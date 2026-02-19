"""Core STEAM cascade algorithm."""

import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
from scipy.signal import oaconvolve as convolve
import netCDF4
from .constants import (
    hurst_horizontal as H_h,
    hurst_vertical_anisotropy as H_z,
)


def simulate(
    h_profile,
    qt_profile,
    nx, ny,
    dx, dy,
    outer_scale,
    spheroscale,
    domain_height,
    profile_dz,
    output_path,
    sparsity_factors=(1, 1, 2),
    surface_pressure=101325.0,
    seed=None,
):
    """Run STEAM cascade with coarsening, write results to NetCDF.

    Parameters
    ----------
    h_profile : ndarray, shape (n_profile,)
        Mean moist static energy profile [J/kg] at spacing profile_dz.
    qt_profile : ndarray, shape (n_profile,)
        Mean total water mixing ratio profile [kg/kg] at spacing profile_dz.
    nx, ny : int
        Horizontal grid dimensions at finest resolution.
    dx, dy : float
        Horizontal grid spacing at finest resolution [m].
    outer_scale : float
        Outer (largest) turbulon scale L [m]. Must be power-of-2 multiple of dx.
    spheroscale : float
        Scale at which horizontal and vertical turbulon sizes are equal [m].
    domain_height : float
        Vertical extent of domain [m].
    profile_dz : float
        Vertical spacing of input profiles [m].
    output_path : str or Path
        Path to write the output NetCDF file.
    sparsity_factors : tuple of 3 ints
        (s_x, s_y, s_z) sparsity factors. Grid spacing is k/s_x, so
        s=1 gives one grid cell per turbulon and s=2 resolves each
        turbulon with 2 cells.
    surface_pressure : float
        Surface pressure [Pa].
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    Path
        The output_path as a Path object.
    """
    output_path = Path(output_path)
    rng = np.random.default_rng(seed)
    s_x, s_y, s_z = sparsity_factors
    for s, name in zip(sparsity_factors, ('s_x', 's_y', 's_z')):
        if not isinstance(s, int) or s < 1:
            raise ValueError(f"{name} must be a positive integer, got {s}")
    domain_x = nx * dx
    domain_y = ny * dy

    h_profile = np.asarray(h_profile, dtype=np.float32)
    qt_profile = np.asarray(qt_profile, dtype=np.float32)
    if np.any(np.isnan(h_profile)):
        raise ValueError('nans present in h_profile')
    if np.any(np.isnan(qt_profile)):
        raise ValueError('nans present in qt_profile')
    if h_profile.ndim != 1 or qt_profile.ndim != 1:
        raise ValueError("h_profile and qt_profile must be 1D arrays")
    if len(h_profile) == 0 or len(h_profile) != len(qt_profile):
        raise ValueError(
            f"h_profile and qt_profile must have equal nonzero length, "
            f"got {len(h_profile)} and {len(qt_profile)}"
        )
    if outer_scale < dx:
        raise ValueError(
            f"outer_scale ({outer_scale}) must be >= dx ({dx})"
        )
    if domain_height <= 0:
        raise ValueError(f"domain_height must be positive, got {domain_height}")
    z_profile = np.arange(len(h_profile), dtype=np.float32) * profile_dz

    # Size classes: L, L/2, L/4, ..., dx (Apxeq:mean turbulon amplitude)
    n_classes = int(np.log2(outer_scale / dx)) + 1
    k_values = outer_scale / 2 ** np.arange(n_classes)

    # Vertical size for each k (Apxeq:vertical turbulon size class)
    k_z_values = spheroscale * (k_values / spheroscale) ** H_z

    # Normalization factors (Apxeq:norm factor computation)
    k_z_L = k_z_values[0]
    n_large_turbulons = int(domain_height / k_z_L)
    if n_large_turbulons < 1:
        raise ValueError(
            f"domain_height ({domain_height} m) is shorter than the vertical scale of the "
            f"outer-scale turbulons ({k_z_L:.1f} m); increase domain_height or "
            f"decrease outer_scale" 
        )
    C_h_L = _compute_normalization(h_profile, k_z_L, profile_dz, n_large_turbulons)
    C_qt_L = _compute_normalization(qt_profile, k_z_L, profile_dz, n_large_turbulons)

    # Scale-dependent amplitudes (Apxeq:mean turbulon amplitude)
    C_h_k = np.float32(C_h_L) * (k_values / outer_scale).astype(np.float32) ** H_h
    C_qt_k = np.float32(C_qt_L) * (k_values / outer_scale).astype(np.float32) ** H_h

    # Initialize perturbation fields (will be resized each iteration)
    h_perturbation = None
    qt_perturbation = None

    for i, k in enumerate(k_values):
        k_z = k_z_values[i]

        # Coarsened resolution for this scale (oversampled by s)
        dx_k = k / s_x
        dy_k = k / s_y
        dz_k = k_z / s_z
        nx_k = max(2, round(domain_x / dx_k))
        ny_k = max(2, round(domain_y / dy_k))
        nz_k = max(2, round(domain_height / dz_k))

        print(f'Step {i+1:3d}/{k_values.size:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}):  interpolating', end='\r')

        # Interpolate perturbations from previous resolution
        if h_perturbation is None:
            h_perturbation = np.zeros((nx_k, ny_k, nz_k), dtype=np.float32)
            qt_perturbation = np.zeros((nx_k, ny_k, nz_k), dtype=np.float32)
        elif h_perturbation.shape != (nx_k, ny_k, nz_k):
            zoom_factors = (
                nx_k / h_perturbation.shape[0],
                ny_k / h_perturbation.shape[1],
                nz_k / h_perturbation.shape[2],
            )
            h_perturbation = zoom(h_perturbation, zoom_factors, order=1).astype(np.float32)
            qt_perturbation = zoom(qt_perturbation, zoom_factors, order=1).astype(np.float32)

        # Interpolate mean profiles to current vertical grid
        z_k = np.arange(nz_k, dtype=np.float32) * dz_k
        h_mean_1d = np.interp(z_k, z_profile, h_profile).astype(np.float32)
        qt_mean_1d = np.interp(z_k, z_profile, qt_profile).astype(np.float32)
        h_mean = np.broadcast_to(h_mean_1d[np.newaxis, np.newaxis, :], (nx_k, ny_k, nz_k))
        qt_mean = np.broadcast_to(qt_mean_1d[np.newaxis, np.newaxis, :], (nx_k, ny_k, nz_k))

        running_sum_h = h_mean + h_perturbation
        running_sum_qt = qt_mean + qt_perturbation

        # Normalized gradient (Apxeq:amplitude propto gradient normalized)
        print(f'Step {i+1:3d}/{k_values.size:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}):  computing gradients...', end='\r')
        G_h = _normalized_gradient(running_sum_h, dx_k, dy_k, dz_k)
        G_qt = _normalized_gradient(running_sum_qt, dx_k, dy_k, dz_k)

        # Norm for min
        h_min = 315 * 1004
        h_max = 355 * 1004
        qt_min = 0
        qt_max = 30 / 1000
        normalized_distance_to_clamp_h = np.maximum(np.minimum(running_sum_h - h_min, h_max - running_sum_h) / ((h_max-h_min)/2), 0)
        normalized_distance_to_clamp_qt = np.maximum(np.minimum(running_sum_qt - qt_min, qt_max - running_sum_qt) / ((qt_max-qt_min)/2), 0)

        

        # Sparse noise field — same S_k for both h and qt
        S_k = _sparse_noise(nx_k, ny_k, nz_k, s_x, s_y, s_z, rng)
        S_k[:,:,:2] = 0
        S_k[:,:,-2:] = 0
        print('Setting lowest level S_k=0')


        # Amplitude arrays (Apxeq:amplitude propto gradient normalized)
        A_h = G_h * S_k * C_h_k[i] * normalized_distance_to_clamp_h
        A_qt = G_qt * S_k * C_qt_k[i] * normalized_distance_to_clamp_qt
        A_h = G_h * S_k * C_h_k[i]
        A_qt = G_qt * S_k * C_qt_k[i] 
        # Clean memory
        del G_h, S_k, G_qt, normalized_distance_to_clamp_h, normalized_distance_to_clamp_qt

        # Build compact 3D Mexican hat kernel (Apxeq:turbulon shape)
        kernel = _mexican_hat_kernel(k, spheroscale, dx_k, dy_k, dz_k, support_factor=10)

        # Convolve and accumulate (periodic x,y; zero-padded z)
        print(f'Step {i+1:3d}/{k_values.size:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}):  computing convolutions...', end='\r')
        h_perturbation += _convolve_periodic_xy_edgepad_z(A_h, kernel)
        qt_perturbation += _convolve_periodic_xy_edgepad_z(A_qt, kernel)
        print(f'Step {i+1:3d}/{k_values.size:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}):  done                               ')

    # Final fields are at finest resolution (last iteration)
    nz_final = nz_k
    dz_final = dz_k
    z_final = np.arange(nz_final, dtype=np.float32) * dz_final
    h_mean_final = np.interp(z_final, z_profile, h_profile).astype(np.float32)
    qt_mean_final = np.interp(z_final, z_profile, qt_profile).astype(np.float32)

    h_3d = np.ascontiguousarray(h_mean_final[np.newaxis, np.newaxis, :] + h_perturbation)
    qt_3d = np.ascontiguousarray(qt_mean_final[np.newaxis, np.newaxis, :] + qt_perturbation)

    print('Clamping values')
    qt_3d[qt_3d<0] = 0

    # Write NetCDF output
    nx_final, ny_final = h_3d.shape[0], h_3d.shape[1]
    dx_final = (nx * dx) / nx_final
    dy_final = (ny * dy) / ny_final
    print(f"Writing NetCDF to {output_path} ...")
    x_coords = np.arange(nx_final, dtype=np.float32) * dx_final
    y_coords = np.arange(ny_final, dtype=np.float32) * dy_final

    ds = netCDF4.Dataset(output_path, "w", format="NETCDF4")

    # Dimensions
    ds.createDimension("x", nx_final)
    ds.createDimension("y", ny_final)
    ds.createDimension("z", nz_final)
    ds.createDimension("z_profile", len(h_profile))
    ds.createDimension("k", len(k_values))

    # Coordinate variables
    x_var = ds.createVariable("x", "f4", ("x",))
    x_var[:] = x_coords
    x_var.units = "m"
    x_var.long_name = "x coordinate"

    y_var = ds.createVariable("y", "f4", ("y",))
    y_var[:] = y_coords
    y_var.units = "m"
    y_var.long_name = "y coordinate"

    z_var = ds.createVariable("z", "f4", ("z",))
    z_var[:] = z_final
    z_var.units = "m"
    z_var.long_name = "z coordinate (height)"

    # Data variables — chunked and compressed
    h_var = ds.createVariable("h", "f4", ("x", "y", "z"), zlib=True, complevel=4,
                              chunksizes=(min(64, nx_final), min(64, ny_final), nz_final))
    h_var[:] = h_3d
    h_var.units = "J/kg"
    h_var.long_name = "moist static energy"

    qt_var = ds.createVariable("qt", "f4", ("x", "y", "z"), zlib=True, complevel=4,
                               chunksizes=(min(64, nx_final), min(64, ny_final), nz_final))
    qt_var[:] = qt_3d
    qt_var.units = "kg/kg"
    qt_var.long_name = "total water mixing ratio"

    # Profile variables
    zp_var = ds.createVariable("z_profile", "f4", ("z_profile",))
    zp_var[:] = z_profile
    zp_var.units = "m"

    hp_var = ds.createVariable("h_profile", "f4", ("z_profile",))
    hp_var[:] = h_profile
    hp_var.units = "J/kg"
    hp_var.long_name = "input h profile"

    qtp_var = ds.createVariable("qt_profile", "f4", ("z_profile",))
    qtp_var[:] = qt_profile
    qtp_var.units = "kg/kg"
    qtp_var.long_name = "input qt profile"

    # 1D scale arrays
    kv = ds.createVariable("k_values", "f4", ("k",))
    kv[:] = k_values.astype(np.float32)
    kv.units = "m"
    kv.long_name = "horizontal turbulon scale classes"

    kzv = ds.createVariable("k_z_values", "f4", ("k",))
    kzv[:] = k_z_values.astype(np.float32)
    kzv.units = "m"
    kzv.long_name = "vertical turbulon scale classes"

    chk = ds.createVariable("C_h_k", "f4", ("k",))
    chk[:] = C_h_k
    chk.long_name = "scale-dependent h amplitude"

    cqtk = ds.createVariable("C_qt_k", "f4", ("k",))
    cqtk[:] = C_qt_k
    cqtk.long_name = "scale-dependent qt amplitude"

    # Scalar attributes on root group
    ds.nx = np.int32(nx_final)
    ds.ny = np.int32(ny_final)
    ds.dx = np.float32(dx_final)
    ds.dy = np.float32(dy_final)
    ds.dz = np.float32(dz_final)
    ds.outer_scale = np.float32(outer_scale)
    ds.spheroscale = np.float32(spheroscale)
    ds.domain_height = np.float32(domain_height)
    ds.profile_dz = np.float32(profile_dz)
    ds.sparsity_factors = np.array(sparsity_factors, dtype=np.int32)
    ds.surface_pressure = np.float32(surface_pressure)
    ds.seed = np.int32(seed) if seed is not None else -1
    ds.C_h_L = np.float32(C_h_L)
    ds.C_qt_L = np.float32(C_qt_L)
    ds.n_large_turbulons = np.int32(n_large_turbulons)
    ds.H_h = np.float32(H_h)
    ds.H_z = np.float32(H_z)

    ds.close()
    print(f"Written {output_path}")
    return output_path


def _mexican_hat_kernel(k, spheroscale, dx, dy, dz, support_factor=5):
    """Compact 3D Mexican hat kernel, centered, support-sized only.

    (Apxeq:turbulon shape) T_k(||r||) = (1 - ||r||²/k²) exp(-||r||²/(2k²))
    (Apxeq:canonical anisotropic norm) for the metric.

    Returns shape (2*half_nx+1, 2*half_ny+1, 2*half_nz+1) with center
    at the middle of the array. Only extends to support_factor * k
    horizontally and support_factor * k_z vertically.
    """
    k_z = spheroscale * (k / spheroscale) ** H_z
    half_nx = int(np.ceil(support_factor * k / dx))
    half_ny = int(np.ceil(support_factor * k / dy))
    half_nz = int(np.ceil(support_factor * k_z / dz))

    x = np.arange(-half_nx, half_nx + 1, dtype=np.float32) * dx
    y = np.arange(-half_ny, half_ny + 1, dtype=np.float32) * dy
    z = np.arange(-half_nz, half_nz + 1, dtype=np.float32) * dz

    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    # Anisotropic norm (Apxeq:canonical anisotropic norm)
    r_norm_sq = spheroscale**2 * (
        (X / spheroscale) ** 2
        + (Y / spheroscale) ** 2
        + (np.abs(Z) / spheroscale) ** (2.0 / H_z)
    )
    ratio_sq = r_norm_sq / k**2
    kernel = ((1.0 - ratio_sq) * np.exp(-ratio_sq / 2.0)).astype(np.float32)

    # Force zero mean
    # kernel -= kernel.mean()
    return kernel


def _mexican_hat_1d(k_z, dz, support_factor=5):
    """1D vertical Mexican hat kernel (max=1) for normalization computation."""
    half_nz = int(np.ceil(support_factor * k_z / dz))
    z = np.arange(-half_nz, half_nz + 1) * dz
    ratio_sq = (z / k_z) ** 2
    return (1.0 - ratio_sq) * np.exp(-ratio_sq / 2.0)


def _compute_normalization(profile, k_z, dz, n_large_turbulons):
    """Compute C_{Φ,L}: per-turbulon average response of the profile to T_L.

    (Apxeq:norm factor computation)

    Convolves the profile with a 1D Mexican hat at the outer scale, then
    converts the per-sample mean absolute response to a per-turbulon
    average by scaling by n_large_turbulons / n_profile_samples. This
    makes C_L independent of the profile sampling resolution (profile_dz).
    """
    kernel_1d = _mexican_hat_1d(k_z, dz) / (k_z/dz)
    norm_factor = 2    # Need to figure out why this works
    kernel_1d = kernel_1d  / norm_factor

    half = len(kernel_1d) // 2
    padded = np.pad(profile, half, mode='edge')
    convolved = np.convolve(padded, kernel_1d, mode='valid')
    rms = np.mean(np.abs(convolved))
    # return rms * (n_large_turbulons / len(profile))
    return rms


def _sparse_noise(nx, ny, nz, s_x, s_y, s_z, rng):
    """Generate sparse N(0,1) noise field at oversampled resolution.

    When s=1 in all dimensions, returns a full grid of random values.
    When s>1, noise is nonzero every s grid points (one turbulon center
    per k spacing at the oversampled resolution).

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions (at oversampled resolution).
    s_x, s_y, s_z : int
        Oversampling factors (must be positive integers).
    rng : numpy.random.Generator
    """
    if s_x == 1 and s_y == 1 and s_z == 1:
        return rng.standard_normal((nx, ny, nz), dtype=np.float32)

    field = np.zeros((nx, ny, nz), dtype=np.float32)
    ix = np.arange(0, nx, s_x)
    iy = np.arange(0, ny, s_y)
    iz = np.arange(0, nz, s_z)
    field[np.ix_(ix, iy, iz)] = rng.standard_normal(
        (len(ix), len(iy), len(iz)), dtype=np.float32,
    )
    return field


def _normalized_gradient(field_3d, dx, dy, dz):
    """Compute |∇f| / mean(|∇f|), the normalized gradient magnitude.

    (Apxeq:amplitude propto gradient normalized)
    Uses periodic central differences for x,y and np.gradient for z
    (one-sided at boundaries).
    """
    grad_x = (np.roll(field_3d, -1, axis=0) - np.roll(field_3d, 1, axis=0)) / (2 * dx)
    grad_y = (np.roll(field_3d, -1, axis=1) - np.roll(field_3d, 1, axis=1)) / (2 * dy)
    grad_z = np.gradient(field_3d, dz, axis=2)

    magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    mean_mag = magnitude.mean()

    if mean_mag > 0:
        return magnitude / mean_mag
    else:
        return np.ones_like(magnitude)


def _fold_kernel_to_field(kernel, field_shape):
    """Fold a kernel larger than the field onto the field size for periodic axes.

    For x and y (periodic), if the kernel extent exceeds the field size, the
    kernel is wrapped (aliased) onto the field period and re-centered. The z
    axis is left unchanged (handled by edge-padding the field instead).
    """
    result = kernel
    for axis in (0, 1):  # periodic axes only
        n_field = field_shape[axis]
        n_kern = result.shape[axis]
        if n_kern <= n_field:
            continue
        # Sum kernel slices that map to the same periodic index
        folded_shape = list(result.shape)
        folded_shape[axis] = n_field
        folded = np.zeros(folded_shape, dtype=result.dtype)
        center = n_kern // 2
        for i in range(n_kern):
            target = (i - center) % n_field
            slc_src = [slice(None)] * 3
            slc_dst = [slice(None)] * 3
            slc_src[axis] = i
            slc_dst[axis] = target
            folded[tuple(slc_dst)] += result[tuple(slc_src)]
        # Re-center so that index n_field//2 is the origin
        result = np.roll(folded, n_field // 2, axis=axis)
    return result


def _convolve_periodic_xy_edgepad_z(field, kernel):
    """Direct convolution: periodic in x,y, edge-padded in z.

    If the kernel is larger than the field in x or y, it is first folded
    (periodically aliased) to the field size. Then x,y are wrap-padded and
    z is edge-padded before calling oaconvolve with mode='valid'.

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz)
    kernel : ndarray, shape (knx, kny, knz)
        Compact centered kernel from _mexican_hat_kernel.
    """
    kernel = _fold_kernel_to_field(kernel, field.shape)
    # Asymmetric padding: total pad per axis = kernel_size - 1, to get
    # output_size = field_size from oaconvolve mode='valid'
    kx, ky, kz = kernel.shape
    pad_x_l, pad_x_r = kx // 2, (kx - 1) // 2
    pad_y_l, pad_y_r = ky // 2, (ky - 1) // 2
    pad_z_l, pad_z_r = kz // 2, (kz - 1) // 2
    # Wrap-pad x,y; edge-pad z, then convolve and crop
    padded = np.pad(field, ((pad_x_l, pad_x_r), (pad_y_l, pad_y_r), (0, 0)), mode='wrap')
    padded = np.pad(padded, ((0, 0), (0, 0), (pad_z_l, pad_z_r)), mode='edge')
    result = convolve(padded, kernel, mode='valid')
    return result.astype(np.float32)
