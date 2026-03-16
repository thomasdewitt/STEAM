"""Core STEAM cascade algorithm."""

import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
from .constants import (
    hurst_horizontal as H_h,
    hurst_vertical_anisotropy as H_z,
)
from . import turbulons as _turbulons
from .utils import convolve_periodic_xy_zeropad_z_oa
from .output import write_netcdf


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
    sparsity_factors=(1, 1, 1),
    k_factor=2,
    surface_pressure=101325.0,
    seed=None,
    h_min=315 * 1004,
    h_max=355 * 1004,
    qt_min=0.0,
    qt_max=30 / 1000,
    min_distance_to_ground=1,
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
        Outer (largest) turbulon scale L [m]. Must satisfy:
          - outer_scale / dx and outer_scale / dy are positive integer
            powers of k_factor (e.g. k_factor^m for some m >= 0).
          - domain_x = nx*dx and domain_y = ny*dy are integer multiples
            of outer_scale.
    spheroscale : float or ndarray, shape (n_profile,)
        Scale at which horizontal and vertical turbulon sizes are equal [m].
        If a 1D array, it is interpreted as a height-dependent profile at
        the same levels as h_profile. The geometric mean is used for the
        kernel shape and normalization reference; the full profile drives the
        variable-dz grid and height-dependent C factors.
    domain_height : float
        Vertical extent of domain [m].
    profile_dz : float
        Vertical spacing of input profiles [m].
    output_path : str or Path
        Path to write the output NetCDF file.
    sparsity_factors : tuple of 3 ints
        (s_x, s_y, s_z) oversampling factors. Grid spacing at scale k is
        k/(2*s_i), so s=1 is Nyquist sampling and s=2 gives 4 grid cells
        per turbulon width.
    k_factor : int
        Ratio between successive scale classes.
    surface_pressure : float
        Surface pressure [Pa].
    seed : int or None
        Random seed for reproducibility.
    h_min, h_max : float
        Soft-clamp bounds on moist static energy [J/kg].
    qt_min, qt_max : float
        Soft-clamp bounds on total water mixing ratio [kg/kg].
    min_distance_to_ground : int
        Turbulon centers are not placed within min_distance_to_ground × k_z
        of the ground. Must be a non-negative integer.

    Returns
    -------
    Path
        The output_path as a Path object.
    """
    output_path = Path(output_path)
    rng = np.random.default_rng(seed)

    # Input validation
    for s, name in zip(sparsity_factors, ('s_x', 's_y', 's_z')):
        if not isinstance(s, int) or s < 1:
            raise ValueError(f"{name} must be a positive integer, got {s}")
    if not isinstance(min_distance_to_ground, int) or min_distance_to_ground < 0:
        raise ValueError(
            f"min_distance_to_ground must be a non-negative integer, got {min_distance_to_ground}"
        )

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
    if domain_height <= 0:
        raise ValueError(f"domain_height must be positive, got {domain_height}")
    for grid_spacing, axis in ((dx, 'x'), (dy, 'y')):
        ratio = outer_scale / grid_spacing
        if ratio < 1:
            raise ValueError(
                f"outer_scale ({outer_scale}) must be >= d{axis} ({grid_spacing})"
            )
        log_ratio = np.log(ratio) / np.log(k_factor)
        if abs(log_ratio - round(log_ratio)) > 1e-9:
            raise ValueError(
                f"outer_scale/d{axis} must be a positive integer power of "
                f"k_factor={k_factor}, got outer_scale={outer_scale}, "
                f"d{axis}={grid_spacing} (ratio={ratio})"
            )
    for domain_size, axis in ((domain_x, 'x'), (domain_y, 'y')):
        n_tiles = domain_size / outer_scale
        if abs(n_tiles - round(n_tiles)) > 1e-9:
            raise ValueError(
                f"domain_{axis} ({domain_size}) must be an integer multiple of "
                f"outer_scale ({outer_scale}), got ratio={n_tiles}"
            )

    # Spheroscale: scalar or height-dependent profile
    spheroscale_arr = np.asarray(spheroscale, dtype=np.float64)
    if spheroscale_arr.ndim > 0:
        if len(spheroscale_arr) != len(h_profile):
            raise ValueError(
                f"spheroscale array must have the same length as profiles, "
                f"got {len(spheroscale_arr)} vs {len(h_profile)}"
            )
        spheroscale_profile = spheroscale_arr
        # Geometric mean used for kernel shape and normalization reference
        spheroscale_geomean = float(np.exp(np.mean(np.log(spheroscale_arr))))
    else:
        spheroscale_profile = None
        spheroscale_geomean = float(spheroscale_arr)

    z_profile = np.arange(len(h_profile), dtype=np.float64) * profile_dz

    # Scale classes: L, L/k_factor, ..., 2*dx (finest)
    n_classes = int(round(np.log(outer_scale / (2*dx)) / np.log(k_factor))) + 1
    k_values = outer_scale / k_factor ** np.arange(n_classes)
    # k_z_values uses geometric mean spheroscale for NetCDF output reference
    k_z_values = spheroscale_geomean * (k_values / spheroscale_geomean) ** H_z

    k_L = k_values[0]
    k_z_L = k_z_values[0]
    if profile_dz >= k_z_L:
        raise ValueError(
            f"profile_dz ({profile_dz} m) must be less than the vertical outer "
            f"scale k_z_L ({k_z_L:.1f} m); use a finer profile resolution"
        )

    n_large_turbulons = int(domain_height / k_z_L)
    if n_large_turbulons < 1:
        raise ValueError(
            f"domain_height ({domain_height} m) is shorter than the vertical scale of the "
            f"outer-scale turbulons ({k_z_L:.1f} m); increase domain_height or "
            f"decrease outer_scale"
        )

    grids = _compute_all_grids(
        k_values, domain_x, domain_y, domain_height, sparsity_factors,
        spheroscale_geomean, z_profile, spheroscale_profile,
    )

    C_h_k = _compute_normalization(
        h_profile, k_L, spheroscale_geomean, profile_dz,
        z_profile, k_values, outer_scale, grids['z_arrays'], spheroscale_profile,
        sparsity_factors=sparsity_factors,
    )
    C_qt_k = _compute_normalization(
        qt_profile, k_L, spheroscale_geomean, profile_dz,
        z_profile, k_values, outer_scale, grids['z_arrays'], spheroscale_profile,
        sparsity_factors=sparsity_factors,
    )
    # Scalar C_L for NetCDF attribute: outer scale hurst_scale=1
    C_h_L = float(np.mean(C_h_k[0]))
    C_qt_L = float(np.mean(C_qt_k[0]))

    h_pert, qt_pert, final_grid = cascade_loop(
        h_profile, qt_profile, z_profile,
        grids,
        C_h_k, C_qt_k,
        h_min, h_max, qt_min, qt_max,
        min_distance_to_ground,
        sparsity_factors,
        spheroscale_geomean,
        rng,
    )
    
    # Construct final 3D fields
    z_final = final_grid['z'].astype(np.float32)
    h_mean_final = np.interp(z_final, z_profile, h_profile).astype(np.float32)
    qt_mean_final = np.interp(z_final, z_profile, qt_profile).astype(np.float32)

    h_3d = np.ascontiguousarray(h_mean_final[np.newaxis, np.newaxis, :] + h_pert)
    qt_3d = np.ascontiguousarray(qt_mean_final[np.newaxis, np.newaxis, :] + qt_pert)

    nx_final_val = h_3d.shape[0]
    ny_final_val = h_3d.shape[1]
    dx_final = (nx * dx) / nx_final_val
    dy_final = (ny * dy) / ny_final_val
    x_coords = np.arange(nx_final_val, dtype=np.float32) * dx_final
    y_coords = np.arange(ny_final_val, dtype=np.float32) * dy_final
    # dz: scalar for uniform grids, 1D array for variable-spheroscale grids
    dz_out = float(final_grid['dz'][0]) if spheroscale_profile is None else final_grid['dz'].astype(np.float32)

    simulation_params = {
        'nx': nx_final_val,
        'ny': ny_final_val,
        'dx': dx_final,
        'dy': dy_final,
        'dz': dz_out,
        'outer_scale': outer_scale,
        'spheroscale': spheroscale_geomean,
        'domain_height': domain_height,
        'profile_dz': profile_dz,
        'sparsity_factors': sparsity_factors,
        'surface_pressure': surface_pressure,
        'seed': seed,
        'C_h_L': C_h_L,
        'C_qt_L': C_qt_L,
        'n_large_turbulons': n_large_turbulons,
        'H_h': H_h,
        'H_z': H_z,
        'h_min': h_min,
        'h_max': h_max,
        'qt_min': qt_min,
        'qt_max': qt_max,
        'min_distance_to_ground': min_distance_to_ground,
    }

    write_netcdf(
        output_path, h_3d, qt_3d,
        x_coords, y_coords, z_final,
        h_profile, qt_profile, z_profile.astype(np.float32),
        k_values, k_z_values, C_h_k, C_qt_k,
        simulation_params,
    )
    return output_path


def cascade_loop(
    h_profile, qt_profile, z_profile,
    grids,
    C_h_k, C_qt_k,
    h_min, h_max, qt_min, qt_max,
    min_distance_to_ground,
    sparsity_factors,
    spheroscale,
    rng,
    h_perturbation=None,
    qt_perturbation=None,
):
    """Run the multi-scale turbulon cascade over all scale classes.

    Iterates from the outer scale down to the finest scale, adding
    perturbations from each scale class. Can be called directly for
    nested simulations that re-run the cascade on a subdomain.

    Parameters
    ----------
    h_profile, qt_profile : ndarray, shape (n_profile,)
        Mean profiles at heights z_profile.
    z_profile : ndarray, shape (n_profile,)
        Heights of profile levels [m].
    grids : dict
        Output of _compute_all_grids. Contains 1D arrays k, k_z, nx, ny, nz,
        dx, dy, dz (mean, per class) and lists z_arrays, dz_arrays (one 1D
        array per scale class, each of length nz for that class).
    C_h_k, C_qt_k : list of (float or 1D ndarray)
        Scale-dependent amplitudes, one entry per scale class. Scalar for
        uniform spheroscale; 1D array of length nz_k for variable spheroscale.
        Both broadcast correctly over (nx_k, ny_k, nz_k).
    h_min, h_max : float
        Soft-clamp bounds for moist static energy.
    qt_min, qt_max : float
        Soft-clamp bounds for total water mixing ratio.
    min_distance_to_ground : int
        Turbulon centers are not placed within min_distance_to_ground × k_z
        of the ground. The lowest 2 * s_z * min_distance_to_ground z-cells
        of the noise field are zeroed at every scale class.
    sparsity_factors : tuple of 3 ints
        (s_x, s_y, s_z) oversampling factors.
    spheroscale : float
        Geometric-mean spheroscale [m], used only for the turbulon kernel
        (which uses isotropic_norm and is therefore spheroscale-independent).
    rng : numpy.random.Generator
    h_perturbation, qt_perturbation : ndarray or None
        Existing perturbation fields for nested simulations. If None,
        initialized to zero at the first scale class resolution.

    Returns
    -------
    h_perturbation : ndarray, shape (nx_finest, ny_finest, nz_finest)
    qt_perturbation : ndarray, shape (nx_finest, ny_finest, nz_finest)
    final_grid_info : dict
        Keys nx, ny, nz, dx, dy, dz (mean), z (1D coordinate array) from
        the last (finest) iteration.
    """
    s_x, s_y, s_z = sparsity_factors
    n_classes = len(grids['k'])
    n_zero = int(round(2 * s_z * min_distance_to_ground))

    for i in range(n_classes):
        k = grids['k'][i]
        nx_k = int(grids['nx'][i])
        ny_k = int(grids['ny'][i])
        nz_k = int(grids['nz'][i])
        dx_k = float(grids['dx'][i])
        dy_k = float(grids['dy'][i])
        z_k = grids['z_arrays'][i]   # 1D array of z-coordinates (left-edge of each cell)

        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})           interpolating...', end='\r')

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
        h_mean_1d = np.interp(z_k, z_profile, h_profile).astype(np.float32)
        qt_mean_1d = np.interp(z_k, z_profile, qt_profile).astype(np.float32)
        h_mean = np.broadcast_to(h_mean_1d[np.newaxis, np.newaxis, :], (nx_k, ny_k, nz_k))
        qt_mean = np.broadcast_to(qt_mean_1d[np.newaxis, np.newaxis, :], (nx_k, ny_k, nz_k))

        running_sum_h = h_mean + h_perturbation
        running_sum_qt = qt_mean + qt_perturbation

        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})   computing G_* factors...', end='\r')
        # Soft-clamp weights: linearly ramp to zero as field approaches h_min/h_max
        soft_clip_h = np.maximum(0, running_sum_h - h_min) * np.maximum(0, h_max - running_sum_h)
        soft_clip_qt = np.maximum(0, running_sum_qt - qt_min) * np.maximum(0, qt_max - running_sum_qt)
        # Normalize them
        if soft_clip_h.mean() != 0: soft_clip_h = soft_clip_h/soft_clip_h.mean()
        if soft_clip_qt.mean() != 0: soft_clip_qt = soft_clip_qt/soft_clip_qt.mean()

        # G_h function is the clips * normalized gradient magnitude (Apxeq:amplitude propto gradient normalized)
        G_h = _normalized_gradient(running_sum_h, dx_k, dy_k, z_k) * soft_clip_h
        G_qt = _normalized_gradient(running_sum_qt, dx_k, dy_k, z_k) * soft_clip_qt

        # Sparse noise — same S_k for both h and qt (Apxeq:mean turbulon amplitude)
        S_k = _sparse_noise(nx_k, ny_k, nz_k, s_x, s_y, s_z, rng)
        S_k[:, :, :n_zero] = 0

        # Build compact 3D turbulon kernel (Apxeq:turbulon shape)
        kernel = _turbulon_envelope(1, spheroscale, 1/(2*s_x), 1/(2*s_y), 1/(2*s_z),
                                    support_factor=10, norm='isotropic_norm')
        dz_k = z_k[1]-z_k[0]
        kernel_variance = np.sum(kernel**2 * dx_k * dy_k * dz_k)
        # qt_variance = (running_sum_qt-running_sum_qt.mean())**2
        # h_variance = (running_sum_h-running_sum_h.mean())**2
        # qt_variance = _normalized_gradient((normalized_distance_to_clip_qt)**2, dx_k, dy_k, z_k)
        # h_variance = _normalized_gradient((normalized_distance_to_clip_h)**2, dx_k, dy_k, z_k)
        # A_h = S_k * C_h_k[i] * h_variance/h_variance.mean()
        # A_qt = S_k * C_qt_k[i] * qt_variance/qt_variance.mean()
        A_h = S_k * C_h_k[i] * G_h 
        A_qt = S_k * C_qt_k[i] * G_qt
        print(np.count_nonzero(np.isnan(G_h)))
        # print('-------------')
        # print(np.mean(G_h * S_k * C_h_k[i] * normalized_distance_to_clip_h))
        # print(np.mean(A_h))
        # print()
        # print(np.std(G_h * S_k * C_h_k[i] * normalized_distance_to_clip_h))
        # print(np.std(A_h))
        # print(A_h.shape)
        # print('---------')
        # if i == 7: exit()

        # Amplitude arrays (Apxeq:amplitude propto gradient normalized)
        # C_h_k[i] is a scalar or 1D array of shape (nz_k,); both broadcast over (nx,ny,nz)
        # A_h = G_h * S_k * C_h_k[i] * normalized_distance_to_clip_h
        # A_qt = G_qt * S_k * C_qt_k[i] * normalized_distance_to_clip_qt
        # del G_h, S_k, G_qt, normalized_distance_to_clip_h, normalized_distance_to_clip_qt

        # Convolve and accumulate (periodic x,y; zero-padded z)
        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})           computing convolutions...', end='\r')
        h_perturbation += convolve_periodic_xy_zeropad_z_oa(A_h, kernel)
        qt_perturbation += convolve_periodic_xy_zeropad_z_oa(A_qt, kernel)

        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})           done                     ')

    final_grid_info = {
        'nx': nx_k, 'ny': ny_k, 'nz': nz_k,
        'dx': dx_k, 'dy': dy_k,
        'dz': grids['dz_arrays'][i],   # 1D array (uniform for scalar ls, variable for profile ls)
        'z': z_k,
    }
    return h_perturbation, qt_perturbation, final_grid_info


def _compute_all_grids(k_values, domain_x, domain_y, domain_height, sparsity_factors,
                       spheroscale, z_profile=None, spheroscale_profile=None):
    """Precompute grid dimensions and z-coordinate arrays for all scale classes.

    For scalar spheroscale, each scale class has uniform cell spacing. For a
    spheroscale profile, each scale class uses an altitude-dependent dz:
      dz(z) = k_z(z) / (2*s_z)  where  k_z(z) = ls(z) * (k/ls(z))^H_z
    Cells are accumulated from z=0 until domain_height is reached, then all
    dz values are scaled uniformly so their sum equals domain_height exactly.

    Parameters
    ----------
    k_values : ndarray, shape (n_classes,)
    domain_x, domain_y, domain_height : float
    sparsity_factors : tuple of 3 ints
    spheroscale : float
        Geometric-mean (or scalar) spheroscale, used for the uniform-dz case.
    z_profile : ndarray, shape (n_profile,) or None
        Heights at which spheroscale_profile is given. Required if
        spheroscale_profile is not None.
    spheroscale_profile : ndarray, shape (n_profile,) or None
        Height-dependent spheroscale. If None, uniform dz is used.

    Returns
    -------
    dict with keys:
        k, k_z : 1D arrays, shape (n_classes,)
        nx, ny, nz : 1D int arrays, shape (n_classes,)
        dx, dy : 1D float arrays, shape (n_classes,)
        dz : 1D float array, shape (n_classes,) — mean cell height per class
        z_arrays : list of n_classes 1D float64 arrays — left-edge z-coords
        dz_arrays : list of n_classes 1D float64 arrays — cell heights
    """
    s_x, s_y, s_z = sparsity_factors
    n_classes = len(k_values)
    nx_arr = np.empty(n_classes, dtype=np.int64)
    ny_arr = np.empty(n_classes, dtype=np.int64)
    nz_arr = np.empty(n_classes, dtype=np.int64)
    dx_arr = np.empty(n_classes, dtype=np.float64)
    dy_arr = np.empty(n_classes, dtype=np.float64)
    dz_arr = np.empty(n_classes, dtype=np.float64)
    z_arrays = []
    dz_arrays = []

    for i, k in enumerate(k_values):
        dx_k = k / (2 * s_x)
        dy_k = k / (2 * s_y)
        nx_arr[i] = int(round(domain_x / dx_k))
        ny_arr[i] = int(round(domain_y / dy_k))
        dx_arr[i] = dx_k
        dy_arr[i] = dy_k

        if spheroscale_profile is None:
            # Scalar spheroscale: uniform dz
            k_z = spheroscale * (k / spheroscale) ** H_z
            n_turbulons_z = int(np.ceil(domain_height / k_z))
            nz_k = n_turbulons_z * 2 * s_z
            dz_k = domain_height / nz_k
            dz_k_arr = np.full(nz_k, dz_k)
            z_k_arr = np.arange(nz_k, dtype=np.float64) * dz_k
        else:
            # Profile spheroscale: integrate dz(z) = k_z(z)/(2*s_z) from z=0
            def k_z_local(z):
                ls = np.interp(z, z_profile, spheroscale_profile)
                return ls * (k / ls) ** H_z

            z_edges = [0.0]
            while z_edges[-1] < domain_height:
                dz_here = k_z_local(z_edges[-1]) / (2 * s_z)
                z_edges.append(z_edges[-1] + dz_here)

            nz_k = len(z_edges) - 1
            dz_raw = np.diff(z_edges)
            scale_factor = domain_height / np.sum(dz_raw)
            dz_k_arr = dz_raw * scale_factor
            z_k_arr = np.concatenate([[0.0], np.cumsum(dz_k_arr[:-1])])
            dz_k = float(np.mean(dz_k_arr))
            print(f'  k={k:.0f}m: nz={nz_k}, dz scale factor={scale_factor:.6f}')

        nz_arr[i] = nz_k
        dz_arr[i] = dz_k
        z_arrays.append(z_k_arr)
        dz_arrays.append(dz_k_arr)

    k_z_values = spheroscale * (k_values / spheroscale) ** H_z

    return {
        'k': k_values,
        'k_z': k_z_values,
        'nx': nx_arr,
        'ny': ny_arr,
        'nz': nz_arr,
        'dx': dx_arr,
        'dy': dy_arr,
        'dz': dz_arr,
        'z_arrays': z_arrays,
        'dz_arrays': dz_arrays,
    }


def _vertical_scale(norm, k, spheroscale):
    """Return k_z: the z at which norm(0, 0, k_z, spheroscale) = k².

    Add a branch here when adding a new norm to steam/turbulons.py.
    """
    if norm == 'canonical_anisotropic_norm':
        return spheroscale * (k / spheroscale) ** H_z
    elif norm == 'isotropic_norm':
        return k
    else:
        raise ValueError(f"No vertical scale defined for norm {norm!r}")


def _turbulon_envelope(k, spheroscale, dx, dy, dz, support_factor=5,
                       norm='canonical_anisotropic_norm', shape='mexican_hat'):
    """Compact 3D turbulon kernel, centered, trimmed to support extent.

    Looks up norm and shape functions by name from steam.turbulons.
    Raises ValueError if either name is not found there.

    The kernel extends to support_factor * k horizontally and
    support_factor * k_z vertically (k_z = k for isotropic norm).

    Returns array of shape (2*half_nx+1, 2*half_ny+1, 2*half_nz+1)
    with the kernel center at the middle index.
    """
    if norm not in _turbulons.NORMS:
        raise ValueError(f"Unknown norm {norm!r}. Valid options: {sorted(_turbulons.NORMS)}")
    if shape not in _turbulons.SHAPES:
        raise ValueError(f"Unknown shape {shape!r}. Valid options: {sorted(_turbulons.SHAPES)}")

    norm_fn = getattr(_turbulons, norm)
    shape_fn = getattr(_turbulons, shape)

    k_z = _vertical_scale(norm, k, spheroscale)

    half_nx = int(np.ceil(support_factor * k / dx))
    half_ny = int(np.ceil(support_factor * k / dy))
    half_nz = int(np.ceil(support_factor * k_z / dz))

    x = np.arange(-half_nx, half_nx + 1, dtype=np.float32) * dx
    y = np.arange(-half_ny, half_ny + 1, dtype=np.float32) * dy
    z = np.arange(-half_nz, half_nz + 1, dtype=np.float32) * dz

    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    r_norm_sq = norm_fn(X, Y, Z, spheroscale)
    return shape_fn(r_norm_sq, k).astype(np.float32)


def _compute_normalization(profile, k_L, spheroscale, profile_dz,
                           z_profile, k_values, outer_scale, z_arrays,
                           spheroscale_profile=None,
                           norm='canonical_anisotropic_norm', shape='mexican_hat',
                           support_factor=10, sparsity_factors=(1, 1, 1)):
    """Compute scale- and height-dependent amplitude arrays C_k.

    (Apxeq:norm factor computation)

    Measures profile variation at the vertical outer scale k_z_L using a
    first-order Haar wavelet (step function: -1 below center, +1 above),
    then corrects for the 3D kernel energy so that the cascade perturbation
    variance matches the measured profile variation.

      C_k[i] = haar_response / sqrt(E_3d) * (k_i / outer_scale)^H_h

    Parameters
    ----------
    profile : ndarray, shape (n_profile,)
    k_L : float  —  outer-scale horizontal turbulon size
    spheroscale : float  —  geometric-mean spheroscale
    profile_dz : float  —  profile vertical spacing
    z_profile : ndarray, shape (n_profile,)
    k_values : ndarray, shape (n_classes,)
    outer_scale : float
    z_arrays : list of n_classes 1D arrays  —  z-coords per scale class
    spheroscale_profile : ndarray or None
    sparsity_factors : tuple of 3 ints

    Returns
    -------
    list of n_classes entries, each a float32 scalar or 1D float32 array.
    """
    k_z = _vertical_scale(norm, k_L, spheroscale)

    # --- Haar wavelet at scale k_z ---
    n_half = max(1, int(round(k_z / profile_dz)/2))
    if n_half % 2 ==1: n_half += 1
    kernel_haar = np.empty(n_half, dtype=np.float64)
    kernel_haar[:n_half//2] = -1.0 / (n_half/2)
    kernel_haar[n_half//2:] = 1.0 / (n_half/2)

    padded = np.pad(profile, n_half//2, mode='edge')
    response = np.abs(np.convolve(padded, kernel_haar, mode='valid'))

    # --- 3D kernel energy correction ---
    # s_x, s_y, s_z = sparsity_factors
    # kernel_3d = _turbulon_envelope(
    #     1, spheroscale, 1 / (2 * s_x), 1 / (2 * s_y), 1 / (2 * s_z),
    #     support_factor=support_factor, norm='isotropic_norm', shape=shape,
    # )
    # energy_3d = float(np.sum(kernel_3d ** 2))
    energy_3d = 1

    C_k = []
    for i, k in enumerate(k_values):
        hurst_scale = float((k / outer_scale) ** H_h)
        if spheroscale_profile is None:
            C_L = float(np.mean(response)) / np.sqrt(energy_3d)
            C_k.append(np.float32(C_L * hurst_scale))
        else:
            C_profile = response / np.sqrt(energy_3d)
            C_k.append(np.interp(z_arrays[i], z_profile, C_profile).astype(np.float32) * hurst_scale)
    return C_k


def _sparse_noise(nx, ny, nz, factor_x, factor_y, factor_z, rng):
    """Generate sparse N(0,1) noise field at oversampled resolution.

    When s=1 in all dimensions, returns a full grid of random values.
    When s>1, noise is nonzero every s grid points (one turbulon center
    per k spacing at the oversampled resolution).

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions (at oversampled resolution).
    factor_x, factor_y, factor_z : int
        Oversampling factors (must be positive integers).
    rng : numpy.random.Generator
    """
    if factor_x == 1 and factor_y == 1 and factor_z == 1:
        return rng.standard_normal((nx, ny, nz), dtype=np.float32)

    field = np.zeros((nx, ny, nz), dtype=np.float32)
    ix = np.arange(0, nx, factor_x)
    iy = np.arange(0, ny, factor_y)
    iz = np.arange(0, nz, factor_z)
    field[np.ix_(ix, iy, iz)] = rng.standard_normal(
        (len(ix), len(iy), len(iz)), dtype=np.float32,
    )
    return field


def _normalized_gradient(field_3d, dx, dy, z_coords):
    """Compute |∇f| / mean(|∇f|), the normalized gradient magnitude.

    (Apxeq:amplitude propto gradient normalized)
    Uses periodic central differences for x,y and np.gradient for z.
    z_coords may be a scalar spacing (uniform grid) or a 1D array of
    z-positions (non-uniform grid); np.gradient handles both.
    """
    grad_x = (np.roll(field_3d, -1, axis=0) - np.roll(field_3d, 1, axis=0)) / (2 * dx)
    grad_y = (np.roll(field_3d, -1, axis=1) - np.roll(field_3d, 1, axis=1)) / (2 * dy)
    grad_z = np.gradient(field_3d, z_coords, axis=2)

    magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    mean_mag = magnitude.mean()

    if mean_mag > 0:
        return magnitude / mean_mag
    else:
        return np.ones_like(magnitude)
