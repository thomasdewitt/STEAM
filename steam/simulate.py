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
    n_size_classes=None,
    surface_pressure=101325.0,
    seed=None,
    h_min=315 * 1004,
    h_max=355 * 1004,
    qt_min=0.0,
    qt_max=30 / 1000,
    min_distance_to_ground=1,
    turbulon_shape = 'mexican_hat'
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
          - outer_scale must be >= dx.
          - domain_x = nx*dx and domain_y = ny*dy are integer multiples
            of outer_scale.
    spheroscale : float or ndarray, shape (n_profile,)
        Scale at which horizontal and vertical turbulon sizes are equal [m].
        If a 1D array, it is interpreted as a height-dependent profile at
        the same levels as h_profile. Scalars are promoted to uniform arrays.
        The full profile drives the variable-dz grid and height-dependent
        normalization; the arithmetic mean is used only for input validation.
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
    n_size_classes : int or None
        Number of size classes between outer_scale and 2*dx, inclusive.
        If None, use the current dyadic behavior. If an integer, the
        adjacent-class multiplicative gap is derived from the endpoint
        constraint and this class count.
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
    if n_size_classes is not None:
        if not isinstance(n_size_classes, int) or n_size_classes < 2:
            raise ValueError(
                "n_size_classes must be an integer >= 2 when provided, "
                f"got {n_size_classes}"
            )
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
    h_profile_min = float(np.min(h_profile))
    h_profile_max = float(np.max(h_profile))
    qt_profile_min = float(np.min(qt_profile))
    qt_profile_max = float(np.max(qt_profile))
    if h_min > h_profile_min:
        raise ValueError(
            f"h_min ({h_min}) must be <= min(h_profile) ({h_profile_min})"
        )
    if h_max < h_profile_max:
        raise ValueError(
            f"h_max ({h_max}) must be >= max(h_profile) ({h_profile_max})"
        )
    if qt_min > qt_profile_min:
        raise ValueError(
            f"qt_min ({qt_min}) must be <= min(qt_profile) ({qt_profile_min})"
        )
    if qt_max < qt_profile_max:
        raise ValueError(
            f"qt_max ({qt_max}) must be >= max(qt_profile) ({qt_profile_max})"
        )
    if domain_height <= 0:
        raise ValueError(f"domain_height must be positive, got {domain_height}")
    ratio = outer_scale / dx
    if ratio < 1:
        raise ValueError(
            f"outer_scale ({outer_scale}) must be >= dx ({dx})"
        )
    if n_size_classes is not None and outer_scale <= 2 * dx:
        raise ValueError(
            "outer_scale must be greater than 2*dx when n_size_classes is "
            f"provided, got outer_scale={outer_scale} and dx={dx}"
        )
    for domain_size, axis in ((domain_x, 'x'), (domain_y, 'y')):
        n_tiles = domain_size / outer_scale
        if abs(n_tiles - round(n_tiles)) > 1e-9:
            raise ValueError(
                f"domain_{axis} ({domain_size}) must be an integer multiple of "
                f"outer_scale ({outer_scale}), got ratio={n_tiles}"
            )

    # Spheroscale: always promote to 1D profile
    spheroscale_arr = np.asarray(spheroscale, dtype=np.float64)
    if spheroscale_arr.ndim > 0:
        if len(spheroscale_arr) != len(h_profile):
            raise ValueError(
                f"spheroscale array must have the same length as profiles, "
                f"got {len(spheroscale_arr)} vs {len(h_profile)}"
            )
        spheroscale_profile = spheroscale_arr
    else:
        spheroscale_profile = np.full(len(h_profile), float(spheroscale_arr))

    z_profile = np.arange(len(h_profile), dtype=np.float64) * profile_dz

    # Scale classes: L, ..., 2*dx (finest)
    s_x, s_y, s_z = sparsity_factors
    if n_size_classes is None:
        size_class_gap_factor = 2.0
        n_classes = int(
            round(np.log(outer_scale / (2 * dx)) / np.log(size_class_gap_factor))
        ) + 1
    else:
        n_classes = n_size_classes
        size_class_gap_factor = float(
            (outer_scale / (2 * dx)) ** (1.0 / (n_classes - 1))
        )
    k_values = outer_scale / size_class_gap_factor ** np.arange(n_classes)

    # Input validation using arithmetic mean spheroscale
    spheroscale_mean = float(np.mean(spheroscale_profile))
    k_z_L_mean = spheroscale_mean * (outer_scale / spheroscale_mean) ** H_z
    if profile_dz >= k_z_L_mean:
        raise ValueError(
            f"profile_dz ({profile_dz} m) must be less than the vertical outer "
            f"scale k_z_L ({k_z_L_mean:.1f} m); use a finer profile resolution"
        )

    n_large_turbulons = int(domain_height / k_z_L_mean)
    if n_large_turbulons < 1:
        raise ValueError(
            f"domain_height ({domain_height} m) is shorter than the vertical scale of the "
            f"outer-scale turbulons ({k_z_L_mean:.1f} m); increase domain_height or "
            f"decrease outer_scale"
        )

    grids = _compute_all_grids(
        k_values, domain_x, domain_y, domain_height, sparsity_factors,
        spheroscale_profile, z_profile,
    )

    # Interpolate profiles to finest grid for normalization
    z_finest = grids['z_arrays'][-1]
    h_on_finest = np.interp(z_finest, z_profile, h_profile)
    qt_on_finest = np.interp(z_finest, z_profile, qt_profile)
    spheroscale_on_finest = np.interp(z_finest, z_profile, spheroscale_profile)

    # Vertical outer scale in finest-grid points (constant across height)
    k_min = k_values[-1]
    vertical_outer_scale_grid_pts = int(round(2 * s_z * (outer_scale / k_min) ** H_z))

    # Unit turbulon z-slice for normalization correction
    unit_turbulon = _turbulon_envelope(1, 1/(2*s_x), 1/(2*s_y), 1/(2*s_z),
                                     support_factor=10, shape=turbulon_shape)
    spectral_width_correction = spectral_width_normalization(
        turbulon_shape, size_class_gap_factor
    )

    C_h_k = _compute_normalization(
        h_on_finest, vertical_outer_scale_grid_pts,
        k_values, outer_scale, grids,
        unit_turbulon, turbulon_shape
    )
    C_qt_k = _compute_normalization(
        qt_on_finest, vertical_outer_scale_grid_pts,
        k_values, outer_scale, grids,
        unit_turbulon, turbulon_shape
    )
    C_h_k = [c * spectral_width_correction for c in C_h_k]
    C_qt_k = [c * spectral_width_correction for c in C_qt_k]
    # print(C_h_k[0])
    # exit()
    # Scalar C_L for NetCDF attribute: mean of outer-scale C profile
    C_h_L = float(np.mean(C_h_k[0]))
    C_qt_L = float(np.mean(C_qt_k[0]))

    h_pert, qt_pert, final_grid = cascade_loop(
        h_profile, qt_profile, z_profile,
        grids,
        C_h_k, C_qt_k,
        h_min, h_max, qt_min, qt_max,
        min_distance_to_ground,
        sparsity_factors,
        rng,
        turbulon_shape=turbulon_shape,
    )

    # Construct final 3D fields
    z_final = final_grid['z'].astype(np.float32)
    h_mean_final = np.interp(z_final, z_profile, h_profile).astype(np.float32)
    qt_mean_final = np.interp(z_final, z_profile, qt_profile).astype(np.float32)
    spheroscale_final = np.interp(z_final, z_profile, spheroscale_profile).astype(np.float32)

    h_3d = np.ascontiguousarray(h_mean_final[np.newaxis, np.newaxis, :] + h_pert)
    qt_3d = np.ascontiguousarray(qt_mean_final[np.newaxis, np.newaxis, :] + qt_pert)

    nx_final_val = h_3d.shape[0]
    ny_final_val = h_3d.shape[1]
    dx_final = (nx * dx) / nx_final_val
    dy_final = (ny * dy) / ny_final_val
    x_coords = np.arange(nx_final_val, dtype=np.float32) * dx_final
    y_coords = np.arange(ny_final_val, dtype=np.float32) * dy_final

    # k_z_values using arith-mean spheroscale as reference
    k_z_values = spheroscale_mean * (k_values / spheroscale_mean) ** H_z

    simulation_params = {
        'nx': nx_final_val,
        'ny': ny_final_val,
        'dx': dx_final,
        'dy': dy_final,
        'dz': final_grid['dz'].astype(np.float32),
        'outer_scale': outer_scale,
        'spheroscale': spheroscale_final,
        'domain_height': domain_height,
        'profile_dz': profile_dz,
        'sparsity_factors': sparsity_factors,
        'n_size_classes': n_classes,
        'size_class_gap_factor': size_class_gap_factor,
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
    rng,
    h_perturbation=None,
    qt_perturbation=None,
    turbulon_shape = 'mexican_hat'
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
    C_h_k, C_qt_k : list of 1D ndarray
        Scale-dependent amplitudes, one entry per scale class. Each is a
        1D array of length nz_k that broadcasts over (nx_k, ny_k, nz_k).
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

        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})           computing G_* factors...', end='\r')

        # --- DEBUG: soft-clamp options (pick one, comment the rest) ---
        # OPTION A: parabolic [0,1], no mean norm — smooth, peaks at midpoint
        u_h = np.clip((running_sum_h - h_min) / (h_max - h_min), 0, 1)
        soft_clip_h = (4 * u_h * (1 - u_h))
        u_qt = np.clip((running_sum_qt - qt_min) / (qt_max - qt_min), 0, 1)
        soft_clip_qt = (4 * u_qt * (1 - u_qt))
        # soft_clip_h /= soft_clip_h.mean()
        # soft_clip_qt /= soft_clip_qt.mean()

        # OPTION B: tent/triangle — distance to nearest bound, [0,1]
        #   (this is what 614ffed computed but never actually used)
        # soft_clip_h = np.maximum(np.minimum(running_sum_h - h_min, h_max - running_sum_h) / ((h_max - h_min) / 2), 0)
        # soft_clip_qt = np.maximum(np.minimum(running_sum_qt - qt_min, qt_max - running_sum_qt) / ((qt_max - qt_min) / 2), 0)

        # OPTION C: no soft clip (what 614ffed "looking pretty good" actually ran)
        # soft_clip_h = 1.0
        # soft_clip_qt = 1.0
        # --- END DEBUG ---

        # G_h function is the clips * normalized gradient magnitude (Apxeq:amplitude propto gradient normalized)
        G_h = _gradient_magnitude(running_sum_h, dx_k, dy_k, z_k) * soft_clip_h
        G_qt = _gradient_magnitude(running_sum_qt, dx_k, dy_k, z_k) * soft_clip_qt
        mean_h = G_h.mean(axis=(0, 1), keepdims=True)
        mean_h = np.where(mean_h > 0, mean_h, 1.0)
        G_h = G_h / mean_h
        mean_qt = G_qt.mean(axis=(0, 1), keepdims=True)
        mean_qt = np.where(mean_qt > 0, mean_qt, 1.0)
        G_qt = G_qt / mean_qt
        # G_h = soft_clip_h
        # G_qt = soft_clip_qt

        # Sparse noise — same S_k for both h and qt (Apxeq:mean turbulon amplitude)
        S_k = _sparse_noise(nx_k, ny_k, nz_k, s_x, s_y, s_z, rng)
        S_k[:, :, :n_zero] = 0
        S_k[:, :, -n_zero:] = 0

        # Build compact 3D turbulon kernel (Apxeq:turbulon shape)
        kernel = _turbulon_envelope(1, 1/(2*s_x), 1/(2*s_y), 1/(2*s_z),
                                    support_factor=10, shape=turbulon_shape)
        
        # Final turbulon amplitudes
        A_h = S_k * C_h_k[i] * G_h 
        A_qt = S_k * C_qt_k[i] * G_qt
        # A_h = S_k * np.mean(C_h_k[i]) * G_h 
        # A_qt = S_k * np.mean(C_qt_k[i]) * G_qt
        

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
                       spheroscale_profile, z_profile):
    """Precompute grid dimensions and z-coordinate arrays for all scale classes.

    Each scale class uses an altitude-dependent dz:
      dz(z) = k_z(z) / (2*s_z)  where  k_z(z) = ls(z) * (k/ls(z))^H_z
    Cells are accumulated from z=0 until domain_height is reached, then all
    dz values are scaled uniformly so their sum equals domain_height exactly.
    In x and y, target spacings are k/(2*s_x) and k/(2*s_y), but the stored
    dx, dy are the actual spacings implied by the rounded integer grid counts
    so every class spans the domain exactly.

    If spheroscale_profile is scalar, it is promoted to a uniform array.

    Parameters
    ----------
    k_values : ndarray, shape (n_classes,)
    domain_x, domain_y, domain_height : float
    sparsity_factors : tuple of 3 ints
    spheroscale_profile : ndarray, shape (n_profile,)
        Height-dependent spheroscale [m]. Scalar is auto-promoted.
    z_profile : ndarray, shape (n_profile,)
        Heights at which spheroscale_profile is given.

    Returns
    -------
    dict with keys:
        k : 1D array, shape (n_classes,)
        nx, ny, nz : 1D int arrays, shape (n_classes,)
        dx, dy : 1D float arrays, shape (n_classes,)
        dz : 1D float array, shape (n_classes,) — mean cell height per class
        z_arrays : list of n_classes 1D float64 arrays — left-edge z-coords
        dz_arrays : list of n_classes 1D float64 arrays — cell heights
    """
    spheroscale_profile = np.asarray(spheroscale_profile, dtype=np.float64)
    if spheroscale_profile.ndim == 0:
        spheroscale_profile = np.full_like(z_profile, float(spheroscale_profile))

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
        target_dx_k = k / (2 * s_x)
        target_dy_k = k / (2 * s_y)
        nx_arr[i] = int(round(domain_x / target_dx_k))
        ny_arr[i] = int(round(domain_y / target_dy_k))
        dx_arr[i] = domain_x / nx_arr[i]
        dy_arr[i] = domain_y / ny_arr[i]

        # Integrate dz(z) = k_z(z)/(2*s_z) from z=0
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

        nz_arr[i] = nz_k
        dz_arr[i] = dz_k
        z_arrays.append(z_k_arr)
        dz_arrays.append(dz_k_arr)

    return {
        'k': k_values,
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


def _turbulon_envelope(k, dx, dy, dz, support_factor=5, shape='mexican_hat'):
    """Compact 3D isotropic turbulon kernel, centered, trimmed to support.

    The kernel extends to support_factor * k in all directions (isotropic).
    Returns array of shape (2*half_nx+1, 2*half_ny+1, 2*half_nz+1)
    with the kernel center at the middle index.
    """
    if shape not in _turbulons.SHAPES:
        raise ValueError(f"Unknown shape {shape!r}. Valid options: {sorted(_turbulons.SHAPES)}")

    shape_fn = getattr(_turbulons, shape)

    half_nx = int(np.ceil(support_factor * k / dx))
    half_ny = int(np.ceil(support_factor * k / dy))
    half_nz = int(np.ceil(support_factor * k / dz))

    x = np.arange(-half_nx, half_nx + 1, dtype=np.float32) * dx
    y = np.arange(-half_ny, half_ny + 1, dtype=np.float32) * dy
    z = np.arange(-half_nz, half_nz + 1, dtype=np.float32) * dz

    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    r_norm_sq = X**2 + Y**2 + Z**2
    return shape_fn(r_norm_sq, k).astype(np.float32)


def spectral_width_normalization(shape, size_class_gap_factor):
    """Return an overlap correction from kernel spectral width and class spacing.

    The returned factor is a simple linear overlap correction:

        min(1, size_class_gap_factor / spectral_width_factor)

    where ``size_class_gap_factor`` is the multiplicative ratio between
    adjacent size-class values and ``spectral_width_factor`` is a hard-coded
    estimate of the kernel's PSD e-folding width in the same multiplicative
    sense.

    The width constants were estimated once offline from the continuous
    dimensionless kernel shapes at unit scale:

    - ``mexican_hat``: PSD width factor 2.7954
      This comes from the analytic 1D band-pass form of the line-profile PSD,
      ``P(q) ∝ q^4 exp(-q^2)``, using the ratio of the two wavenumbers where
      the PSD falls to ``peak / e`` around the nonzero spectral peak.
    - ``morlet_omega0_6``: PSD width factor 3.0465
      This comes from the corresponding Gaussian-modulated cosine line-profile
      PSD at ``omega0 = 6`` and ``sigma = 1/pi``, again using the ratio of the
      two wavenumbers where the PSD falls to ``peak / e`` around the main
      nonzero spectral peak.

    These constants are heuristic shape descriptors. They are not recomputed
    at runtime because they depend only on the dimensionless kernel family.
    """
    if size_class_gap_factor <= 1:
        raise ValueError(
            "size_class_gap_factor must be greater than 1, "
            f"got {size_class_gap_factor}"
        )

    spectral_width_factor = {
        'mexican_hat': 2.7954,
        'morlet_omega0_6': 3.0465,
    }.get(shape)
    if spectral_width_factor is None:
        raise ValueError(
            f"No spectral width normalization defined for shape {shape!r}"
        )

    if spectral_width_factor < size_class_gap_factor:
        return 1.0
    return float(size_class_gap_factor / spectral_width_factor)


def _compute_normalization(profile_on_finest_grid, vertical_outer_scale_grid_pts,
                           k_values, outer_scale, z_arrays,
                           turbulon, shape='mexican_hat'):
    """Compute scale- and height-dependent amplitude arrays C_k.

    (Apxeq:norm factor computation)

    Measures profile variation at the vertical outer scale using a first-order
    Haar wavelet (step function: -1 below center, +1 above) on the finest
    resolution grid. The Haar width is adjusted so its spectral peak matches
    the envelope shape's spectral peak, then scales by (k/outer_scale)^H_h.

    Parameters
    ----------
    profile_on_finest_grid : ndarray, shape (nz_finest,)
        Profile interpolated to the finest-resolution z-grid.
    vertical_outer_scale_grid_pts : int
        Number of finest-grid cells spanning the vertical outer scale.
        Constant across height: 2 * s_z * (outer_scale / k_min)^H_z.
    k_values : ndarray, shape (n_classes,)
    outer_scale : float
    z_arrays : list of n_classes 1D arrays — z-coords per scale class
    shape : str
        Envelope shape name. Used to match Haar spectral peak to envelope peak.

    Returns
    -------
    list of n_classes 1D float32 arrays.
    """
    # Spectral peak wavelengths in units of k, measured numerically at high
    # resolution.  The Haar peak ratio is per unit of total width.
    HAAR_PEAK_PER_WIDTH = 1.3443
    SHAPE_PEAK_PER_K = {
        'mexican_hat': 1.4170,
        'morlet_omega0_6': 1.0486,
    }
    shape_peak = SHAPE_PEAK_PER_K.get(shape, 1.4170)

    # Adjust Haar width so its spectral peak matches the envelope's
    corrected_width_pts = vertical_outer_scale_grid_pts * shape_peak / HAAR_PEAK_PER_WIDTH
    n_half = max(1, int(round(corrected_width_pts / 2)))

    kernel_haar = np.empty(2 * n_half, dtype=np.float64)
    kernel_haar[:n_half] = -1.0 / n_half
    kernel_haar[n_half:] = 1.0 / n_half

    padded = np.pad(profile_on_finest_grid, (n_half, n_half - 1), mode='edge')
    response = np.abs(np.convolve(padded, kernel_haar, mode='valid'))

    # Correct for turbulon vs Haar sensitivity ratio                                                                                                       
    # mean(|haar|) = 1/n_half by construction; scale by turbulon's mean absolute difference                                          
    haar_sensitivity = float(np.sum(np.abs(kernel_haar)))                                                
    turbulon_sensitivity = float(np.sum(np.abs(turbulon)))   
    # turbulon_sensitivity = float(np.sum(np.abs(turbulon[turbulon.shape[0]//2, turbulon.shape[1]//2, :])))   
    # print(turbulon_sensitivity / haar_sensitivity)
    # exit()
    # response = response * (haar_sensitivity / turbulon_sensitivity) 
    response /= 2.3

    z_finest = z_arrays['z_arrays'][-1]
    C_k = []
    for i, k in enumerate(k_values):
        hurst_scale = float((k / outer_scale) ** H_h)
        C_profile = np.interp(z_arrays['z_arrays'][i], z_finest, response).astype(np.float32) * hurst_scale
        C_k.append(C_profile)
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


def _gradient_magnitude(field_3d, dx, dy, z_coords):
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
    
    return magnitude
