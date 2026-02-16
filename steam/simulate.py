"""Core STEAM cascade algorithm."""

import numpy as np
from pathlib import Path
from .constants import (
    hurst_horizontal as H_h,
    hurst_vertical_anisotropy as H_z,
)
from .thermodynamics import recover_diagnostics


def simulate(
    h_profile,
    qt_profile,
    nx, ny, nz,
    dx, dy, dz,
    outer_scale,
    spheroscale,
    surface_pressure=101325.0,
    seed=None,
    plot_dir=None,
):
    """Run STEAM cascade and recover diagnostic fields.

    Parameters
    ----------
    h_profile : ndarray, shape (nz,)
        Mean moist static energy profile [J/kg].
    qt_profile : ndarray, shape (nz,)
        Mean total water mixing ratio profile [kg/kg].
    nx, ny, nz : int
        Grid dimensions (must be powers of 2).
    dx, dy, dz : float
        Grid spacing [m].
    outer_scale : float
        Outer (largest) turbulon scale L [m]. Must be power-of-2 multiple of dx.
    spheroscale : float
        Scale at which horizontal and vertical turbulon sizes are equal [m].
    surface_pressure : float
        Surface pressure [Pa].
    seed : int or None
        Random seed for reproducibility.
    plot_dir : str or Path or None
        If provided, save diagnostic plots (A_h/A_qt and running h/qt fields)
        for each cascade iteration to this directory.

    Returns
    -------
    dict with keys: 'h', 'qt', 'T', 'qv', 'qc', 'qi', 'p' (all 3D arrays,
    shape (nx, ny, nz)).
    """
    rng = np.random.default_rng(seed)

    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Size classes: L, L/2, L/4, ..., dx (Apxeq:mean turbulon amplitude)
    n_classes = int(np.log2(outer_scale / dx)) + 1
    k_values = outer_scale / 2 ** np.arange(n_classes)

    # Vertical size for each k (Apxeq:vertical turbulon size class)
    k_z_values = spheroscale * (k_values / spheroscale) ** H_z

    # Normalization factors (Apxeq:norm factor computation)
    k_z_L = k_z_values[0]  # vertical outer scale
    C_h_L = _compute_normalization(h_profile, k_z_L, dz)
    C_qt_L = _compute_normalization(qt_profile, k_z_L, dz)

    # Scale-dependent amplitudes (Apxeq:mean turbulon amplitude)
    C_h_k = C_h_L * (k_values / outer_scale) ** H_h
    C_qt_k = C_qt_L * (k_values / outer_scale) ** H_h

    # Initialize perturbation fields
    h_perturbation = np.zeros((nx, ny, nz))
    qt_perturbation = np.zeros((nx, ny, nz))

    # Broadcast mean profiles to 3D
    h_mean = np.broadcast_to(h_profile[np.newaxis, np.newaxis, :], (nx, ny, nz))
    qt_mean = np.broadcast_to(qt_profile[np.newaxis, np.newaxis, :], (nx, ny, nz))

    # Cascade from large to small scales
    for i, k in enumerate(k_values):
        print(f'Step {i+1}/{k_values.size}')
        k_z = k_z_values[i]

        # Normalized gradient (Apxeq:amplitude propto gradient normalized)
        G_h = _normalized_gradient(h_mean + h_perturbation, dx, dy, dz)
        G_qt = _normalized_gradient(qt_mean + qt_perturbation, dx, dy, dz)

        # Sparse noise field — same S_k for both h and qt
        S_k = _sparse_noise(nx, ny, nz, k, k_z, dx, dz, rng)

        # Amplitude arrays (Apxeq:amplitude propto gradient normalized)
        A_h = G_h * S_k * C_h_k[i]
        A_qt = G_qt * S_k * C_qt_k[i]

        # Build folded 3D Mexican hat kernel (Apxeq:turbulon shape)
        kernel_folded = _mexican_hat_kernel_folded(
            k, spheroscale, dx, dy, dz, nx, ny,
        )

        # Convolve and accumulate
        h_perturbation += _convolve_periodic_xy_zeropad_z(A_h, kernel_folded)
        qt_perturbation += _convolve_periodic_xy_zeropad_z(A_qt, kernel_folded)

        if plot_dir is not None:
            _save_cascade_plots(
                i, k, k_z, A_h, A_qt,
                h_mean, qt_mean, h_perturbation, qt_perturbation,
                nx, ny, nz, dx, dy, dz, plot_dir,
            )

    # Final 3D fields
    h_3d = np.ascontiguousarray(h_mean + h_perturbation)
    qt_3d = np.ascontiguousarray(qt_mean + qt_perturbation)

    # Thermodynamic recovery
    z_values = np.arange(nz) * dz
    diagnostics = recover_diagnostics(h_3d, qt_3d, z_values, surface_pressure)

    return {
        "h": h_3d,
        "qt": qt_3d,
        **diagnostics,
    }


def _mexican_hat_kernel_folded(k, spheroscale, dx, dy, dz, nx, ny,
                               support_factor=5):
    """3D Mexican hat kernel, periodically folded into (nx, ny) and shifted.

    (Apxeq:turbulon shape) T_k(||r||) = (1 - ||r||²/k²) exp(-||r||²/(2k²))
    (Apxeq:canonical anisotropic norm) for the metric.

    Returns shape (nx, ny, knz) with center rolled to (0, 0) in x,y,
    ready for FFT convolution. Computes the folded result directly by
    summing over periodic tile copies, avoiding allocation of the full
    unfolded kernel.
    """
    k_z = spheroscale * (k / spheroscale) ** H_z
    half_nz = int(np.ceil(support_factor * k_z / dz))
    z_kernel = np.arange(-half_nz, half_nz + 1) * dz
    knz = len(z_kernel)

    # Number of tile copies needed in each direction
    support_x = support_factor * k
    support_y = support_factor * k
    n_tiles_x = int(np.ceil(support_x / (nx * dx)))
    n_tiles_y = int(np.ceil(support_y / (ny * dy)))

    # Base grid positions (0..nx-1, 0..ny-1) in physical coords
    x_base = np.arange(nx) * dx  # (nx,)
    y_base = np.arange(ny) * dy  # (ny,)

    folded = np.zeros((nx, ny, knz))

    for nxi in range(-n_tiles_x, n_tiles_x + 1):
        x = x_base + nxi * nx * dx  # (nx,)
        for nyi in range(-n_tiles_y, n_tiles_y + 1):
            y = y_base + nyi * ny * dy  # (ny,)

            X = x[:, None, None]      # (nx, 1, 1)
            Y = y[None, :, None]      # (1, ny, 1)
            Z = z_kernel[None, None, :]  # (1, 1, knz)

            # Anisotropic norm (Apxeq:canonical anisotropic norm)
            r_norm_sq = spheroscale**2 * (
                (X / spheroscale) ** 2
                + (Y / spheroscale) ** 2
                + (np.abs(Z) / spheroscale) ** (2.0 / H_z)
            )
            ratio_sq = r_norm_sq / k**2
            folded += (1.0 - ratio_sq) * np.exp(-ratio_sq / 2.0)

    # Roll so that kernel center (x=0, y=0) maps to index (0, 0).
    # Currently x=0 is at index 0, so no roll needed.
    return folded


def _mexican_hat_1d(k_z, dz, support_factor=3):
    """1D vertical Mexican hat kernel for normalization computation."""
    half_nz = int(np.ceil(support_factor * k_z / dz))
    z = np.arange(-half_nz, half_nz + 1) * dz
    ratio_sq = (z / k_z) ** 2
    return (1.0 - ratio_sq) * np.exp(-ratio_sq / 2.0)


def _compute_normalization(profile, k_z, dz):
    """Compute C_{Φ,L} via 1D convolution then RMS / k_z.

    (Apxeq:norm factor computation)
    C_{Φ,L} = (1/k_z) * sqrt(<(T_L(x=y=0) * <Φ>_t)²>_z)
    """
    kernel_1d = _mexican_hat_1d(k_z, dz)
    convolved = np.convolve(profile, kernel_1d, mode="same")
    rms = np.sqrt(np.mean(convolved**2))
    # return rms / (100*k_z)   
    if profile[0] < 1:
        return rms / (10*k_z/dz)    
    else:
        return rms / (100*k_z/dz)


def _sparse_noise(nx, ny, nz, k, k_z, dx, dz, rng):
    """Generate sparse N(0,1) noise field.

    Turbulon centers are placed via linspace at exact physical spacings k and
    k_z, then rounded to the nearest grid point. This keeps the maximum
    positional error bounded to half a grid cell regardless of accumulation.
    """
    field = np.zeros((nx, ny, nz))

    n_h = max(1, int(nx * dx / k))
    n_v = max(1, int(nz * dz / k_z))

    # Exact positions via linspace, then snap to nearest grid index
    ix = np.unique(np.rint(np.linspace(0, nx - 1, n_h, endpoint=False)).astype(int))
    iy = np.unique(np.rint(np.linspace(0, ny - 1, n_h, endpoint=False)).astype(int))
    iz = np.unique(np.rint(np.linspace(0, nz - 1, n_v, endpoint=False)).astype(int))

    noise = rng.standard_normal((len(ix), len(iy), len(iz)))
    field[np.ix_(ix, iy, iz)] = noise
    return field


def _normalized_gradient(field_3d, dx, dy, dz):
    """Compute |∇f| / mean(|∇f|), the normalized gradient magnitude.

    (Apxeq:amplitude propto gradient normalized)
    Uses np.gradient for z (non-periodic, one-sided at boundaries) and
    periodic central differences for x,y.
    """
    # Periodic central differences in x,y
    grad_x = (np.roll(field_3d, -1, axis=0) - np.roll(field_3d, 1, axis=0)) / (2 * dx)
    grad_y = (np.roll(field_3d, -1, axis=1) - np.roll(field_3d, 1, axis=1)) / (2 * dy)

    # np.gradient in z (second-order central, one-sided at boundaries)
    grad_z = np.gradient(field_3d, dz, axis=2)

    magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    mean_mag = magnitude.mean()

    if mean_mag > 0:
        return magnitude / mean_mag
    else:
        return np.ones_like(magnitude)


def _convolve_periodic_xy_zeropad_z(field, kernel_folded):
    """FFT-based convolution: periodic in x,y, zero-padded in z.

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz)
    kernel_folded : ndarray, shape (nx, ny, knz)
        Pre-folded kernel with center at (0, 0) in x,y, as produced by
        _mexican_hat_kernel_folded.
    """
    nx, ny, nz = field.shape
    knz = kernel_folded.shape[2]

    # FFT field and kernel in x,y
    F_field = np.fft.fft2(field, axes=(0, 1))
    F_kernel = np.fft.fft2(kernel_folded, axes=(0, 1))

    # Linear convolution along z in Fourier-xy space
    nz_padded = nz + knz - 1
    F_field_z = np.fft.fft(F_field, n=nz_padded, axis=2)
    F_kernel_z = np.fft.fft(F_kernel, n=nz_padded, axis=2)

    result_padded = np.fft.ifft(F_field_z * F_kernel_z, axis=2)

    # Inverse FFT in x,y, take real part
    result_padded = np.fft.ifft2(result_padded, axes=(0, 1)).real

    # mode='same': take central nz slice
    start_z = knz // 2
    return result_padded[:, :, start_z : start_z + nz]


def _save_cascade_plots(
    i, k, k_z, A_h, A_qt,
    h_mean, qt_mean, h_perturbation, qt_perturbation,
    nx, ny, nz, dx, dy, dz, plot_dir,
):
    """Save diagnostic cross-section plots (xz, yz, xy) for one cascade iteration."""
    import matplotlib.pyplot as plt

    x_km = np.arange(nx) * dx / 1000
    y_km = np.arange(ny) * dy / 1000
    z_km = np.arange(nz) * dz / 1000

    h_total = h_mean + h_perturbation
    qt_total = qt_mean + qt_perturbation

    slices = [
        ("xz", x_km, z_km, "x [km]", "z [km]",
         lambda f: f[:, ny // 2, :].T),
        ("yz", y_km, z_km, "y [km]", "z [km]",
         lambda f: f[nx // 2, :, :].T),
        ("xy", x_km, y_km, "x [km]", "y [km]",
         lambda f: f[:, :, nz // 2].T),
    ]

    for label, coord_h, coord_v, xlabel, ylabel, slicer in slices:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))

        # Top row: A_h, A_qt with symmetric colorscale (0 = white)
        for ax, data, name in [
            (axes[0, 0], slicer(A_h), "A_h"),
            (axes[0, 1], slicer(A_qt), "A_qt"),
        ]:
            vlim = np.abs(data).max() or 1.0
            im = ax.pcolormesh(
                coord_h, coord_v, data,
                cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto",
            )
            ax.set_title(f"{name}  (k={k:.0f} m)", fontsize=9)
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax, pad=0.02)

        # Bottom row: running h and qt
        h_slice = slicer(h_total) / 1000
        im2 = axes[1, 0].pcolormesh(
            coord_h, coord_v, h_slice,
            cmap="inferno", shading="auto",
        )
        axes[1, 0].set_title("h (running) [kJ/kg]", fontsize=9)
        axes[1, 0].set_ylabel(ylabel)
        axes[1, 0].set_xlabel(xlabel)
        plt.colorbar(im2, ax=axes[1, 0], pad=0.02)

        qt_slice = slicer(qt_total) * 1000
        im3 = axes[1, 1].pcolormesh(
            coord_h, coord_v, qt_slice,
            cmap="YlGnBu", shading="auto",
        )
        axes[1, 1].set_title("qt (running) [g/kg]", fontsize=9)
        axes[1, 1].set_xlabel(xlabel)
        plt.colorbar(im3, ax=axes[1, 1], pad=0.02)

        fig.suptitle(
            f"Iteration {i} ({label}): k = {k:.0f} m, k_z = {k_z:.0f} m",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"cascade_{label}_{i:02d}_k{k:.0f}m.png", dpi=150,
        )
        plt.close(fig)
