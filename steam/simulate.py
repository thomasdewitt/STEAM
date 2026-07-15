"""Core STEAM cascade algorithm."""

import math
import numpy as np
import netCDF4
from pathlib import Path
from . import constants
from .constants import (
    hurst_horizontal as H_h,
    hurst_vertical_anisotropy as H_z,
)
from . import turbulons as _turbulons
from .utils import (
    convolve_periodic_xy_zeropad_z,
    convolve_periodic_xy_zeropad_z_ndimage,
    convolve_periodic_xy_zeropad_z_oa,
    convolve_fft_xy_oa_z,
    zoom_trilinear,
    zoom_bilinear,
)
from .output import write_netcdf

CONVOLVE = convolve_fft_xy_oa_z
SUPPORT_FACTOR = 5

# ─── FUDGE FACTOR (empirical normalization correction) ──────────────────────
# Divides the Haar response in _compute_normalization, i.e. it scales the whole
# cascade amplitude: LARGER value -> SMALLER perturbations. This is a FUDGE
# FACTOR, kept explicit on purpose. History / tuned values:
#   1.15  original, from the empirical normalization-diagnostic script
#   2.0   EGU hack for a bad profile (May 2026)  [current default, unchanged]
#   DYCOMS-RF01 stratocumulus LWP tuning (2026-06-26, steam_experiment/):
#     ~22.6 for constant spheroscale=1000 m;  ~32 for varying ls 200->2 m.
#     These match qt-variance / LWP but leave MSE variance too low -- a single
#     scalar can't fix both, which is why the constants need to become
#     HEIGHT/FIELD-SPECIFIC (roadmap). Until then LWP-tuned is the pragmatic
#     interim choice for a cloud field.
# Left at 2.0 so other configs are unchanged; override per run via
# `steam.simulate.NORMALIZATION_FUDGE = <value>`.
NORMALIZATION_FUDGE = 2.0

# Optional per-field fudge overrides (independent h vs qt cascade amplitude).
# None -> use NORMALIZATION_FUDGE for both (current behaviour). STEAM carries
# ~3x too much variance in qt relative to h vs LES; a single fudge can match
# sigma_h OR sigma_qt/LWP but not both, so independent fudges (smaller for h ->
# more h variance) are the lever for the qt/h variance ratio. Override via
# steam.simulate.FUDGE_H / FUDGE_QT.
FUDGE_H = None
FUDGE_QT = None

# ─── INTERMITTENCY KNOB: gradient-weight exponent ───────────────────────────
# The per-class gradient-magnitude weights (normalized to per-z-level mean 1) are
# what give the cascade its emergent intermittency. Raising them to a power > 1
# (then renormalizing back to mean 1) sharpens the weighting -> turbulon amplitude
# concentrates where gradients are large -> MORE intermittent / spikier field,
# WITHOUT changing the field mean (so mean LWP is preserved; no re-tuning needed).
# 1.0 = unchanged (current behavior). DYCOMS/TWPICE/GATE under-intermittency work
# (2026-06-26) uses this to push concentration toward the LES. Override at runtime
# via `steam.simulate.GRADIENT_WEIGHT_POWER = <value>`.
GRADIENT_WEIGHT_POWER = 1.0

# ─── WEIGHTING MODE: what concentrates turbulon amplitude ────────────────────
# 'gradient' (default, original): weight by |grad(field)| * soft-clamp -> amplitude
#   concentrates at edges/interfaces.
# 'field'   : weight by the soft-clamp ALONE, i.e. (field-min)*(max-field) (the
#   both-sides-clamped field departure). With max set to ~2x the observed field
#   max, the data only reaches the rising half, so the weight ~ |field-min| over
#   the real range -> amplitude concentrates in the BODY of high-field (moist/warm)
#   regions, not edges. Aimed at building the qt-MSE joint (mixing-line) structure
#   the gradient form misses. Overall amplitude shift is absorbed by re-tuning
#   NORMALIZATION_FUDGE (C_*_k machinery unchanged). Override at runtime via
#   `steam.simulate.WEIGHTING = 'field'`.
WEIGHTING = 'gradient'

# ─── NOISE DISTRIBUTION: shape of the sparse turbulon amplitude S_k ──────────
# 'gaussian' (default): S_k ~ N(0,1), symmetric -> the field PDF is
#   symmetric, which is the wrong shape for convection (SAM/TWPICE qt & MSE at
#   ~4 km are strongly RIGHT-skewed, skew ~ +1.8 / +2.0). Gradient weighting is
#   symmetric in sign, so it cannot manufacture that asymmetry on its own.
# 'levy'    : S_k ~ a Levy alpha-stable (scipy.stats.levy_stable) with index
#   LEVY_ALPHA in (0,2] and skewness LEVY_BETA in [-1,1]. beta > 0 -> heavy right
#   tail. NOTE: for alpha < 2 the variance is INFINITE -> a power-law tail that
#   blows up condensate (LWP spikes ~1e5 g/m^2). Right skew, wrong tail.
# 'gamma'   : S_k ~ centered, unit-variance Gamma(shape=NOISE_GAMMA_K): for
#   G ~ Gamma(k,1), X = (G - k)/sqrt(k). Mean 0, var 1, skew = 2/sqrt(k), with a
#   light EXPONENTIAL tail (finite variance). k is the skew dial: k=1 is the
#   centered exponential (skew +2, ~ the LES MSE skew), k=4 -> skew 1,
#   k -> inf -> Gaussian. Bounded below at -sqrt(k) (gentle bounded subsidence;
#   sparse intense positive updraft tail) -- the convective shape, light-tailed.
# 'skewnorm': S_k ~ centered, unit-variance Azzalini skew-normal (shape
#   NOISE_SKEWNORM_A). Gaussian tails (lightest), continuous skew but capped ~1.
# Unit-variance standardization means these are drop-in for N(0,1) (same fudge).
# Override via steam.simulate.NOISE_DIST / LEVY_* / NOISE_GAMMA_K / NOISE_SKEWNORM_A.
NOISE_DIST = 'gaussian'
LEVY_ALPHA = 1.8
LEVY_BETA = 0.5
NOISE_GAMMA_K = 1.0       # Gamma shape; skew = 2/sqrt(k); 1.0 = centered exponential
NOISE_SKEWNORM_A = 4.0    # skew-normal shape parameter
# NOISE_SIGN: multiply the (zero-mean) draw by this. -1 flips the skew sign.
# STEAM's turbulon kernel is a zero-mean wavelet (mexican-hat/morlet, negative
# side-lobes), so the convolution INVERTS skew: a positive-skew S_k yields a
# negative-skew field and vice versa. Hence a LEFT-skewed noise (NOISE_SIGN=-1
# on a right-skewed dist) produces the RIGHT-skewed (convective) field we want.
NOISE_SIGN = 1.0

# ─── FLUX CASCADE: conserved flux modulates the signed scalar amplitudes ──────
# F is dimensionless and initialized to one. One octave is taken per dyadic size
# class. At each class k:
#     dF = CONVOLVE(FLUX_SCALE * xi * F_prev, psi_k);   F = F_prev + dF
# with xi ~ N(0,1) sparse at turbulon centers. F is clipped >= 0 and normalized
# to per-z-level mean 1 after every step. The *signed center amplitude* xi*F_prev
# replaces the old independent S_k for h/qt:
#     A_scalar = xi * F_prev * G_scalar,k * C_scalar,k.
# Thus scalar amplitudes remain conditionally sign-symmetric and F only modulates
# their local scale; FLUX_SCALE tunes flux intermittency without rescaling h/qt.
#
# For a lognormal cascade, the realized intermittency is approximately
#     C1 ~= kappa * FLUX_SCALE**2 / (2 ln 2),
# where kappa is an envelope-overlap (packing) factor near 0.6 for the default
# Mexican-hat/spacing configuration, calibrated empirically.
FLUX_CASCADE = True
FLUX_SCALE = 0.5
# True means h/qt use the signed turbulon-center source of dF (xi*F_prev).
# The old False mode used positive cumulative F and violated sign symmetry.
FLUX_USE_INCREMENT = True

VALID_ANISOTROPY = ('canonical', 'piecewise_isotropic_below_spheroscale')


def _k_z(anisotropy, k, spheroscale):
    """Vertical scale k_z(k, spheroscale) — the grid-anisotropy function.

    Anisotropy in STEAM lives on the grid (dz_i = k_z(k_i)/(2*s_z)), not in
    the turbulon envelope (which is isotropic in cell-index space).

    Options
    -------
    'canonical' : k_z = spheroscale * (k/spheroscale)**H_z.
    'piecewise_isotropic_below_spheroscale' : canonical for k >= spheroscale,
        k_z = k (isotropic) for k < spheroscale. Continuous at k = spheroscale.
    """
    k = np.asarray(k, dtype=np.float64)
    spheroscale = np.asarray(spheroscale, dtype=np.float64)
    canonical = spheroscale * (k / spheroscale) ** H_z
    if anisotropy == 'canonical':
        return canonical
    if anisotropy == 'piecewise_isotropic_below_spheroscale':
        return np.where(k >= spheroscale, canonical, k)
    raise ValueError(
        f"Unknown anisotropy {anisotropy!r}. "
        f"Valid options: {VALID_ANISOTROPY}"
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
    sparsity_factors=(1, 1, 1),
    n_scale_classes_per_dyad=1,
    surface_pressure=101325.0,
    seed=None,
    h_min=315 * 1004,
    h_max=355 * 1004,
    qt_min=0.0,
    qt_max=30 / 1000,
    min_distance_to_ground=1,
    turbulon_shape='mexican_hat',
    anisotropy='canonical',
    compress=None,
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
    n_scale_classes_per_dyad : int
        Controls the multiplicative spacing of scale classes. The gap
        between adjacent classes is ``2 ** (1 / n_scale_classes_per_dyad)``,
        so 1 (default) yields dyadic (powers-of-2) spacing, 2 yields
        ``sqrt(2)`` spacing, etc. Must be a positive integer. The total
        number of classes is chosen so the finest class sits near 2*dx.
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
    compress : bool or None
        If True, 3D data variables in the output NetCDF are written with
        zlib compression at complevel=4. None (default) uses the module-level
        ``steam.constants.output_compress`` setting.

    Returns
    -------
    Path
        The output_path as a Path object.
    """
    if compress is None:
        compress = constants.output_compress
    output_path = Path(output_path)
    seed_sequence = np.random.SeedSequence(seed)

    # Input validation
    for s, name in zip(sparsity_factors, ('s_x', 's_y', 's_z')):
        if not isinstance(s, int) or s < 1:
            raise ValueError(f"{name} must be a positive integer, got {s}")
    if not isinstance(n_scale_classes_per_dyad, int) or n_scale_classes_per_dyad < 1:
        raise ValueError(
            "n_scale_classes_per_dyad must be a positive integer, "
            f"got {n_scale_classes_per_dyad}"
        )
    if FLUX_CASCADE and n_scale_classes_per_dyad != 1:
        raise ValueError(
            "FLUX_CASCADE requires one dyadic size class per octave; "
            "set n_scale_classes_per_dyad=1"
        )
    if not isinstance(min_distance_to_ground, int) or min_distance_to_ground < 0:
        raise ValueError(
            f"min_distance_to_ground must be a non-negative integer, got {min_distance_to_ground}"
        )
    if anisotropy not in VALID_ANISOTROPY:
        raise ValueError(
            f"anisotropy must be one of {VALID_ANISOTROPY}, got {anisotropy!r}"
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
    if outer_scale <= 2 * dx:
        raise ValueError(
            f"outer_scale ({outer_scale}) must be greater than 2*dx "
            f"({2 * dx}); at least two scale classes are required"
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

    # Scale classes: L, ..., 2*dx (finest).  The multiplicative gap between
    # adjacent classes is fully determined by n_scale_classes_per_dyad; the
    # class count is rounded so the finest class sits near 2*dx.
    s_x, s_y, s_z = sparsity_factors
    size_class_gap_factor = 2.0 ** (1.0 / n_scale_classes_per_dyad)
    n_classes = int(round(
        np.log(outer_scale / (2 * dx)) / np.log(size_class_gap_factor)
    )) + 1
    if n_classes < 2:
        raise ValueError(
            f"Fewer than 2 scale classes would be generated for "
            f"outer_scale={outer_scale}, dx={dx}, "
            f"n_scale_classes_per_dyad={n_scale_classes_per_dyad}"
        )
    k_values = outer_scale / size_class_gap_factor ** np.arange(n_classes)

    # Input validation using arithmetic mean spheroscale
    spheroscale_mean = float(np.mean(spheroscale_profile))
    k_z_L_mean = float(_k_z(anisotropy, outer_scale, spheroscale_mean))
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
        anisotropy=anisotropy,
    )

    # Interpolate profiles to finest grid for normalization
    z_finest = grids['z_arrays'][-1]
    h_on_finest = np.interp(z_finest, z_profile, h_profile)
    qt_on_finest = np.interp(z_finest, z_profile, qt_profile)
    spheroscale_on_finest = np.interp(z_finest, z_profile, spheroscale_profile)

    # Vertical outer scale in finest-grid points (constant across height).
    # Ratio k_z_L / dz_min = 2*s_z * k_z(outer_scale) / k_z(k_min) — only
    # simplifies to (outer_scale/k_min)^H_z under canonical anisotropy.
    k_min = k_values[-1]
    vertical_outer_scale_grid_pts = int(round(
        2 * s_z * _k_z(anisotropy, outer_scale, spheroscale_mean)
        / _k_z(anisotropy, k_min, spheroscale_mean)
    ))

    # Unit turbulon z-slice for normalization correction
    unit_turbulon = _turbulon_envelope(1, 1/(2*s_x), 1/(2*s_y), 1/(2*s_z),
                                     support_factor=SUPPORT_FACTOR, shape=turbulon_shape)
    spectral_width_correction = spectral_width_normalization(
        turbulon_shape, size_class_gap_factor
    )

    C_h_k = _compute_normalization(
        h_on_finest, vertical_outer_scale_grid_pts,
        k_values, outer_scale, grids,
        unit_turbulon, turbulon_shape, fudge=FUDGE_H
    )
    C_qt_k = _compute_normalization(
        qt_on_finest, vertical_outer_scale_grid_pts,
        k_values, outer_scale, grids,
        unit_turbulon, turbulon_shape, fudge=FUDGE_QT
    )
    C_h_k = [c * spectral_width_correction for c in C_h_k]
    C_qt_k = [c * spectral_width_correction for c in C_qt_k]
    # Scalar C_L for NetCDF attribute: mean of outer-scale C profile
    C_h_L = float(np.mean(C_h_k[0]))
    C_qt_L = float(np.mean(C_qt_k[0]))

    child_seeds = seed_sequence.spawn(n_classes)

    h_pert, qt_pert, flux_field, final_grid = cascade_loop(
        h_profile, qt_profile, z_profile,
        grids,
        C_h_k, C_qt_k,
        h_min, h_max, qt_min, qt_max,
        min_distance_to_ground,
        sparsity_factors,
        child_seeds,
        turbulon_shape=turbulon_shape,
    )

    # Construct final 3D fields
    z_final = final_grid['z'].astype(np.float32)
    h_mean_final = np.interp(z_final, z_profile, h_profile).astype(np.float32)
    qt_mean_final = np.interp(z_final, z_profile, qt_profile).astype(np.float32)
    spheroscale_final = np.interp(z_final, z_profile, spheroscale_profile).astype(np.float32)

    h_3d = np.ascontiguousarray(h_mean_final[np.newaxis, np.newaxis, :] + h_pert)
    qt_3d = np.ascontiguousarray(qt_mean_final[np.newaxis, np.newaxis, :] + qt_pert)

    np.clip(h_3d, h_min, h_max, out=h_3d)
    np.clip(qt_3d, qt_min, qt_max, out=qt_3d)

    # Conserved multiplicative flux output (FLUX_CASCADE mode); None otherwise.
    flux_3d = np.ascontiguousarray(flux_field) if flux_field is not None else None

    nx_final_val = h_3d.shape[0]
    ny_final_val = h_3d.shape[1]
    dx_final = (nx * dx) / nx_final_val
    dy_final = (ny * dy) / ny_final_val
    x_coords = np.arange(nx_final_val, dtype=np.float32) * dx_final
    y_coords = np.arange(ny_final_val, dtype=np.float32) * dy_final

    # k_z_values using arith-mean spheroscale as reference
    k_z_values = _k_z(anisotropy, k_values, spheroscale_mean)

    # Align stored C_k onto the output z grid so descendants can inherit
    # the (n_classes, nz_output) table without tracking per-class z arrays.
    # Root simulations are no-pad in z, so z_arrays[i] spans the output z
    # range; interpolation to z_final is clean (and identity for the last
    # class in the no-pad case).
    C_h_k_stored = [np.interp(z_final, grids['z_arrays'][i], C_h_k[i]).astype(np.float32)
                    for i in range(len(k_values))]
    C_qt_k_stored = [np.interp(z_final, grids['z_arrays'][i], C_qt_k[i]).astype(np.float32)
                     for i in range(len(k_values))]

    simulation_params = {
        'nx': nx_final_val,
        'ny': ny_final_val,
        'dx': dx_final,
        'dy': dy_final,
        'dz': final_grid['dz'].astype(np.float32),
        'outer_scale': outer_scale,
        'spheroscale': spheroscale_final,
        'spheroscale_profile': spheroscale_profile,
        'domain_height': domain_height,
        'profile_dz': profile_dz,
        'sparsity_factors': sparsity_factors,
        'n_scale_classes_per_dyad': n_scale_classes_per_dyad,
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
        'turbulon_shape': turbulon_shape,
        'anisotropy': anisotropy,
        'flux_cascade': FLUX_CASCADE,
        'flux_noise_scale': FLUX_SCALE,
        'flux_use_increment': FLUX_USE_INCREMENT,
        'noise_dist': NOISE_DIST,
    }

    write_netcdf(
        output_path, h_3d, qt_3d,
        x_coords, y_coords, z_final,
        h_profile, qt_profile, z_profile.astype(np.float32),
        k_values, k_z_values, C_h_k_stored, C_qt_k_stored,
        simulation_params,
        compress=compress,
        flux_3d=flux_3d,
    )
    return output_path


def cascade_loop(
    h_profile, qt_profile, z_profile,
    grids,
    C_h_k, C_qt_k,
    h_min, h_max, qt_min, qt_max,
    min_distance_to_ground,
    sparsity_factors,
    seeds_or_rng,
    h_perturbation=None,
    qt_perturbation=None,
    turbulon_shape='mexican_hat',
    zero_bottom=True,
    zero_top=True,
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
        array per scale class, each of length nz for that class). Also
        padded_extent_x, padded_extent_y, padded_height, z_min_per_class
        (per-class physical extents, used to shrink the grid between
        classes when pad shrinks with k).
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
    seeds_or_rng : list of SeedSequence, or numpy.random.Generator
        If a list of SeedSequence, each element seeds one size class
        independently. If a Generator, it is used directly for all
        classes (legacy behavior).
    h_perturbation, qt_perturbation : ndarray or None
        Existing perturbation fields for nested simulations. If None,
        initialized to zero at the first scale class resolution.
    zero_bottom, zero_top : bool
        Whether to zero the bottom/top n_zero cells of the sparse noise
        at each class. True when the z-boundary corresponds to ground or
        the top of a full simulation domain; False for elevated insets,
        where turbulon centers may legitimately exist below/above the
        inset's inner z range (in the z-pad) and contribute via their
        kernel tails.

    Returns
    -------
    h_perturbation : ndarray, shape (nx_finest, ny_finest, nz_finest)
    qt_perturbation : ndarray, shape (nx_finest, ny_finest, nz_finest)
    flux : ndarray or None
        Final dimensionless unit-mean flux when ``FLUX_CASCADE`` is enabled.
    final_grid_info : dict
        Keys nx, ny, nz, dx, dy, dz (mean), z (1D coordinate array) from
        the last (finest) iteration.
    """
    s_x, s_y, s_z = sparsity_factors
    n_classes = len(grids['k'])
    n_zero = int(round(2 * s_z * min_distance_to_ground))

    if FLUX_CASCADE:
        if not FLUX_USE_INCREMENT:
            raise ValueError(
                "FLUX_USE_INCREMENT must remain True: cumulative positive F "
                "does not provide sign-symmetric scalar amplitudes"
            )
        if n_classes > 1:
            class_ratios = np.asarray(grids['k'][:-1]) / np.asarray(grids['k'][1:])
            if not np.allclose(class_ratios, 2.0):
                raise ValueError(
                    "FLUX_CASCADE requires dyadic size classes (one octave per step)"
                )

    # Determine whether we have per-class seeds or a shared Generator
    use_per_class_seeds = isinstance(seeds_or_rng, (list, tuple))

    # Kernel is identical every iteration — hoist it out of the loop
    kernel = _turbulon_envelope(1, 1/(2*s_x), 1/(2*s_y), 1/(2*s_z),
                                support_factor=SUPPORT_FACTOR, shape=turbulon_shape)

    # Per-class padded physical extents — used to crop between classes
    # when the padded extent shrinks as k gets smaller.
    padded_extent_x = grids['padded_extent_x']
    padded_extent_y = grids['padded_extent_y']
    padded_height = grids['padded_height']
    z_min_per_class = grids['z_min_per_class']

    prev_dx = None
    prev_dy = None
    prev_dz_mean = None
    prev_padded_extent_x = None
    prev_padded_extent_y = None
    prev_padded_height = None
    prev_z_min = None
    flux = None   # conserved multiplicative flux (FLUX_CASCADE mode); None otherwise

    for i in range(n_classes):
        k = grids['k'][i]
        nx_k = int(grids['nx'][i])
        ny_k = int(grids['ny'][i])
        nz_k = int(grids['nz'][i])
        dx_k = float(grids['dx'][i])
        dy_k = float(grids['dy'][i])
        dz_k_mean = float(grids['dz'][i])
        z_k = grids['z_arrays'][i]   # 1D array of z-coordinates (left-edge of each cell)
        padded_x_i = float(padded_extent_x[i])
        padded_y_i = float(padded_extent_y[i])
        padded_h_i = float(padded_height[i])
        z_min_i = float(z_min_per_class[i])

        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})           interpolating...', end='\r')

        # Interpolate perturbations from previous resolution
        if h_perturbation is None:
            h_perturbation = np.zeros((nx_k, ny_k, nz_k), dtype=np.float32)
            qt_perturbation = np.zeros((nx_k, ny_k, nz_k), dtype=np.float32)
            if FLUX_CASCADE:
                flux = np.ones((nx_k, ny_k, nz_k), dtype=np.float32)
        else:
            # Crop-before-zoom when padded extent shrinks between classes.
            # Cropping happens in physical units at the previous class's
            # resolution, centered on the inner region.
            if prev_padded_extent_x is not None:
                eps = 1e-9
                cur_nx, cur_ny, cur_nz = h_perturbation.shape
                if padded_x_i < prev_padded_extent_x - eps:
                    keep = int(round(padded_x_i / prev_dx))
                    keep = max(1, min(cur_nx, keep))
                    start = (cur_nx - keep) // 2
                    h_perturbation = h_perturbation[start:start+keep, :, :]
                    qt_perturbation = qt_perturbation[start:start+keep, :, :]
                    if flux is not None:
                        flux = flux[start:start+keep, :, :]
                if padded_y_i < prev_padded_extent_y - eps:
                    cur_ny = h_perturbation.shape[1]
                    keep = int(round(padded_y_i / prev_dy))
                    keep = max(1, min(cur_ny, keep))
                    start = (cur_ny - keep) // 2
                    h_perturbation = h_perturbation[:, start:start+keep, :]
                    qt_perturbation = qt_perturbation[:, start:start+keep, :]
                    if flux is not None:
                        flux = flux[:, start:start+keep, :]
                if padded_h_i < prev_padded_height - eps:
                    cur_nz = h_perturbation.shape[2]
                    keep = int(round(padded_h_i / prev_dz_mean))
                    keep = max(1, min(cur_nz, keep))
                    # Offset in z is relative to prev class's bottom (z_min_per_class[i-1]).
                    # New bottom is z_min_i; start index in cells:
                    z_offset = z_min_i - prev_z_min
                    start = int(round(z_offset / prev_dz_mean))
                    start = max(0, min(cur_nz - keep, start))
                    h_perturbation = h_perturbation[:, :, start:start+keep]
                    qt_perturbation = qt_perturbation[:, :, start:start+keep]
                    if flux is not None:
                        flux = flux[:, :, start:start+keep]

            if h_perturbation.shape != (nx_k, ny_k, nz_k):
                h_perturbation = zoom_trilinear(h_perturbation, (nx_k, ny_k, nz_k))
                qt_perturbation = zoom_trilinear(qt_perturbation, (nx_k, ny_k, nz_k))
                if flux is not None:
                    flux = zoom_trilinear(flux, (nx_k, ny_k, nz_k))

        # Interpolate mean profiles to current vertical grid
        h_mean_1d = np.interp(z_k, z_profile, h_profile).astype(np.float32)
        qt_mean_1d = np.interp(z_k, z_profile, qt_profile).astype(np.float32)

        # Sparse noise — same S_k for both h and qt (Apxeq:mean turbulon amplitude)
        if use_per_class_seeds:
            rng = np.random.default_rng(seeds_or_rng[i])
        else:
            rng = seeds_or_rng

        if FLUX_CASCADE:
            # The same Gaussian innovation does two jobs: xi*F_prev is the
            # sign-symmetric scalar turbulon amplitude, and c*xi*F_prev drives
            # the next flux octave. It is never replaced by positive cumulative F.
            xi = _sparse_noise(nx_k, ny_k, nz_k, s_x, s_y, s_z, rng, dist='gaussian')
            if zero_bottom and n_zero > 0:
                xi[:, :, :n_zero] = 0
            if zero_top and n_zero > 0:
                xi[:, :, -n_zero:] = 0
            flux_amplitude, dflux, _ = _advance_flux(
                flux, xi, kernel, FLUX_SCALE,
            )
            S_k = flux_amplitude
            del dflux
        else:
            S_k = _sparse_noise(nx_k, ny_k, nz_k, s_x, s_y, s_z, rng)
            if zero_bottom and n_zero > 0:
                S_k[:, :, :n_zero] = 0
            if zero_top and n_zero > 0:
                S_k[:, :, -n_zero:] = 0

        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})           computing G, convolutions...', end='\r')

        # Process h and qt sequentially to halve peak memory
        for (perturbation_field, mean_1d, var_min, var_max, C_k_i) in (
            (h_perturbation,  h_mean_1d,  h_min,  h_max,  C_h_k[i]),
            (qt_perturbation, qt_mean_1d, qt_min, qt_max, C_qt_k[i]),
        ):
            running_sum = perturbation_field + mean_1d[np.newaxis, np.newaxis, :]

            # Soft-clamp: parabolic, both-sides weight = (field-min)*(max-field)
            u = np.clip((running_sum - var_min) / (var_max - var_min), 0, 1)
            soft_clip = u * (1 - u)
            del u

            if WEIGHTING == 'field':
                # FIELD weighting: the soft-clamp IS the weight (no gradient).
                G = soft_clip
                del running_sum
            else:
                # GRADIENT weighting (original): |grad(field)| * soft-clamp.
                G = _gradient_magnitude(running_sum, dx_k, dy_k, z_k)
                del running_sum
                G *= soft_clip
                del soft_clip
            mean_G = G.mean(axis=(0, 1), keepdims=True)
            G /= np.where(mean_G > 0, mean_G, np.float32(1.0))
            del mean_G

            # Intermittency knob: sharpen the (mean-1) weights, renormalize to
            # mean 1 so the field mean is unchanged. See GRADIENT_WEIGHT_POWER.
            if GRADIENT_WEIGHT_POWER != 1.0:
                G **= np.float32(GRADIENT_WEIGHT_POWER)
                mean_Gp = G.mean(axis=(0, 1), keepdims=True)
                G /= np.where(mean_Gp > 0, mean_Gp, np.float32(1.0))
                del mean_Gp

            # Amplitude: reuse G buffer (G becomes A)
            G *= C_k_i          # 1D broadcast, in-place
            G *= S_k            # in-place; G is now A

            # Convolve and accumulate (periodic x,y; zero-padded z)
            perturbation_field += CONVOLVE(G, kernel)
            del G

        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})           done                     ')

        # Record this class's extent so the next iteration can crop
        prev_dx = dx_k
        prev_dy = dy_k
        prev_dz_mean = dz_k_mean
        prev_padded_extent_x = padded_x_i
        prev_padded_extent_y = padded_y_i
        prev_padded_height = padded_h_i
        prev_z_min = z_min_i

    final_grid_info = {
        'nx': nx_k, 'ny': ny_k, 'nz': nz_k,
        'dx': dx_k, 'dy': dy_k,
        'dz': grids['dz_arrays'][i],   # 1D array (uniform for scalar ls, variable for profile ls)
        'z': z_k,
    }
    return h_perturbation, qt_perturbation, flux, final_grid_info


def _advance_flux(flux, innovation, kernel, flux_noise_scale):
    """Advance the dimensionless flux cascade by one size class.

    ``flux`` is the unit-mean field from the previous (larger) class and
    ``innovation`` is sparse, zero-mean, unit-variance noise at turbulon
    centers. Both inputs are modified in place. The returned
    ``flux_amplitude`` is ``innovation * F_prev``; it is retained separately
    from the convolved field increment because it is the turbulon-center
    amplitude appearing inside the convolution.

    The update is

        F_k = F_{k-1} + conv(psi_k, c * innovation * F_{k-1}).

    Negative values are clipped to zero, then every horizontal level is
    normalized back to mean one. The clip count is measured before clipping.
    """
    innovation *= flux
    flux_amplitude = innovation
    dflux = CONVOLVE(flux_amplitude, kernel)
    dflux *= np.float32(flux_noise_scale)
    flux += dflux

    n_clipped = int(np.count_nonzero(flux < 0))
    np.maximum(flux, np.float32(0.0), out=flux)
    mean_flux = flux.mean(axis=(0, 1), keepdims=True)
    empty_levels = (mean_flux <= 0).reshape(-1)
    if np.any(empty_levels):
        flux[:, :, empty_levels] = np.float32(1.0)
        mean_flux[:, :, empty_levels] = np.float32(1.0)
    flux /= np.where(mean_flux > 0, mean_flux, np.float32(1.0))
    del empty_levels, mean_flux
    return flux_amplitude, dflux, n_clipped


def simulate_flux_only(
    grids,
    sparsity_factors=(1, 1, 1),
    seed=None,
    flux_noise_scale=None,
    min_distance_to_ground=0,
    turbulon_shape='mexican_hat',
    zero_bottom=False,
    zero_top=False,
):
    """Run only the unit-mean flux cascade on root-simulation grids.

    This is a cheap calibration/diagnostic path: it performs one convolution
    per size class and skips the scalar profiles, gradients, and four scalar
    convolutions. ``grids`` must come from :func:`_compute_all_grids` with a
    common physical extent at every class (the root-simulation case).

    Returns
    -------
    flux : float32 ndarray
        Final dimensionless flux field, normalized to horizontal mean one at
        every z level.
    diagnostics : dict
        Per-class and aggregate pre-clip counts/fractions.
    """
    s_x, s_y, s_z = sparsity_factors
    for s, name in zip(sparsity_factors, ('s_x', 's_y', 's_z')):
        if not isinstance(s, int) or s < 1:
            raise ValueError(f"{name} must be a positive integer, got {s}")
    if not isinstance(min_distance_to_ground, int) or min_distance_to_ground < 0:
        raise ValueError(
            "min_distance_to_ground must be a non-negative integer, "
            f"got {min_distance_to_ground}"
        )

    extents = (
        np.asarray(grids['padded_extent_x']),
        np.asarray(grids['padded_extent_y']),
        np.asarray(grids['padded_height']),
    )
    if any(not np.allclose(values, values[0]) for values in extents):
        raise ValueError(
            "simulate_flux_only requires a common physical extent across "
            "classes (root-simulation grids without refinement padding)"
        )

    c = FLUX_SCALE if flux_noise_scale is None else flux_noise_scale
    if not np.isfinite(c) or c < 0:
        raise ValueError(f"flux_noise_scale must be finite and non-negative, got {c}")

    n_classes = len(grids['k'])
    if n_classes > 1:
        class_ratios = np.asarray(grids['k'][:-1]) / np.asarray(grids['k'][1:])
        if not np.allclose(class_ratios, 2.0):
            raise ValueError(
                "simulate_flux_only requires dyadic size classes "
                "(one octave per step)"
            )
    seeds = np.random.SeedSequence(seed).spawn(n_classes)
    n_zero = int(round(2 * s_z * min_distance_to_ground))
    kernel = _turbulon_envelope(
        1, 1/(2*s_x), 1/(2*s_y), 1/(2*s_z),
        support_factor=SUPPORT_FACTOR, shape=turbulon_shape,
    )

    flux = None
    steps = []
    total_clipped = 0
    total_points = 0
    for i, k in enumerate(grids['k']):
        shape = (
            int(grids['nx'][i]),
            int(grids['ny'][i]),
            int(grids['nz'][i]),
        )
        if flux is None:
            flux = np.ones(shape, dtype=np.float32)
        elif flux.shape != shape:
            flux = zoom_trilinear(flux, shape)

        rng = np.random.default_rng(seeds[i])
        innovation = _sparse_noise(
            *shape, s_x, s_y, s_z, rng, dist='gaussian',
        )
        if zero_bottom and n_zero > 0:
            innovation[:, :, :n_zero] = 0
        if zero_top and n_zero > 0:
            innovation[:, :, -n_zero:] = 0

        flux_amplitude, dflux, n_clipped = _advance_flux(
            flux, innovation, kernel, c,
        )
        del flux_amplitude, dflux

        n_points = int(flux.size)
        total_clipped += n_clipped
        total_points += n_points
        steps.append({
            'k': float(k),
            'n_clipped': n_clipped,
            'n_points': n_points,
            'clip_fraction': n_clipped / n_points,
        })

    diagnostics = {
        'flux_noise_scale': float(c),
        'steps': steps,
        'n_clipped': total_clipped,
        'n_points': total_points,
        'clip_fraction': total_clipped / total_points,
        'final_zero_fraction': float(np.count_nonzero(flux == 0) / flux.size),
    }
    return flux, diagnostics


def _compute_all_grids(k_values, inner_extent_x, inner_extent_y, inner_height,
                       sparsity_factors, spheroscale_profile, z_profile, z_min=0.0,
                       pad_x_per_class=None, pad_y_per_class=None,
                       pad_z_below_per_class=None, pad_z_above_per_class=None,
                       anisotropy='canonical'):
    """Precompute grid dimensions and z-coordinate arrays for all scale classes.

    Each scale class has its own padded extent in x, y, and z. The inner
    region (of physical size inner_extent_x × inner_extent_y × inner_height
    starting at z_min) is extended by a per-class physical pad on each side.
    Pad = 0 means no pad (use for periodic/spanning dims or ground/top in z
    where the parent is already naturally zero-padded).

    Each scale class uses an altitude-dependent dz:
      dz(z) = k_z(z) / (2*s_z)  where  k_z(z) = _k_z(anisotropy, k, ls(z))
    Cells are accumulated from (z_min - pad_z_below) until
    (z_min + inner_height + pad_z_above) is reached, then all dz values
    are scaled uniformly so their sum equals the padded z-extent exactly.

    In x and y, target spacings are k/(2*s_x) and k/(2*s_y), but the stored
    dx, dy are the actual spacings implied by the rounded integer grid
    counts so every class spans the padded extent exactly.

    If spheroscale_profile is scalar, it is promoted to a uniform array.

    Parameters
    ----------
    k_values : ndarray, shape (n_classes,)
    inner_extent_x, inner_extent_y, inner_height : float
        Physical size of the inner (un-padded) region [m].
    sparsity_factors : tuple of 3 ints
    spheroscale_profile : ndarray, shape (n_profile,)
        Height-dependent spheroscale [m]. Scalar is auto-promoted.
    z_profile : ndarray, shape (n_profile,)
        Heights at which spheroscale_profile is given.
    z_min : float
        Bottom altitude of the inner region [m]. Default 0.0.
    pad_x_per_class, pad_y_per_class : ndarray or None
        Per-class physical pad on each side [m]. Shape (n_classes,).
        None means zero pad at all classes.
    pad_z_below_per_class, pad_z_above_per_class : ndarray or None
        Per-class physical pad below z_min / above z_min+inner_height [m].
        Shape (n_classes,). None means zero pad at all classes.
    anisotropy : str
        Grid-anisotropy function name. See _k_z docstring for options.

    Returns
    -------
    dict with keys:
        k : 1D array, shape (n_classes,)
        nx, ny, nz : 1D int arrays, shape (n_classes,)
        dx, dy : 1D float arrays, shape (n_classes,)
        dz : 1D float array, shape (n_classes,) — mean cell height per class
        z_arrays : list of n_classes 1D float64 arrays — left-edge z-coords
        dz_arrays : list of n_classes 1D float64 arrays — cell heights
        padded_extent_x, padded_extent_y : 1D float arrays, shape (n_classes,)
            Physical x/y extent of each class (inner + 2*pad).
        padded_height : 1D float array, shape (n_classes,)
            Physical z-extent of each class (inner_height + pad_below + pad_above).
        z_min_per_class : 1D float array, shape (n_classes,)
            Bottom z-edge of each class (z_min - pad_z_below).
    """
    spheroscale_profile = np.asarray(spheroscale_profile, dtype=np.float64)
    if spheroscale_profile.ndim == 0:
        spheroscale_profile = np.full_like(z_profile, float(spheroscale_profile))

    s_x, s_y, s_z = sparsity_factors
    n_classes = len(k_values)

    zeros = np.zeros(n_classes, dtype=np.float64)
    pad_x = zeros if pad_x_per_class is None else np.asarray(pad_x_per_class, dtype=np.float64)
    pad_y = zeros if pad_y_per_class is None else np.asarray(pad_y_per_class, dtype=np.float64)
    pad_zb = zeros if pad_z_below_per_class is None else np.asarray(pad_z_below_per_class, dtype=np.float64)
    pad_za = zeros if pad_z_above_per_class is None else np.asarray(pad_z_above_per_class, dtype=np.float64)

    nx_arr = np.empty(n_classes, dtype=np.int64)
    ny_arr = np.empty(n_classes, dtype=np.int64)
    nz_arr = np.empty(n_classes, dtype=np.int64)
    dx_arr = np.empty(n_classes, dtype=np.float64)
    dy_arr = np.empty(n_classes, dtype=np.float64)
    dz_arr = np.empty(n_classes, dtype=np.float64)
    padded_x_arr = np.empty(n_classes, dtype=np.float64)
    padded_y_arr = np.empty(n_classes, dtype=np.float64)
    padded_h_arr = np.empty(n_classes, dtype=np.float64)
    z_min_arr = np.empty(n_classes, dtype=np.float64)
    z_arrays = []
    dz_arrays = []

    for i, k in enumerate(k_values):
        padded_extent_x = inner_extent_x + 2 * pad_x[i]
        padded_extent_y = inner_extent_y + 2 * pad_y[i]
        padded_height = inner_height + pad_zb[i] + pad_za[i]
        z_min_i = z_min - pad_zb[i]

        target_dx_k = k / (2 * s_x)
        target_dy_k = k / (2 * s_y)
        nx_arr[i] = int(round(padded_extent_x / target_dx_k))
        ny_arr[i] = int(round(padded_extent_y / target_dy_k))
        dx_arr[i] = padded_extent_x / nx_arr[i]
        dy_arr[i] = padded_extent_y / ny_arr[i]

        # Integrate dz(z) = k_z(z)/(2*s_z) from z_min_i
        def k_z_local(z):
            ls = np.interp(z, z_profile, spheroscale_profile)
            return float(_k_z(anisotropy, k, ls))

        z_top = z_min_i + padded_height
        z_edges = [z_min_i]
        while z_edges[-1] < z_top:
            dz_here = k_z_local(z_edges[-1]) / (2 * s_z)
            z_edges.append(z_edges[-1] + dz_here)

        nz_k = len(z_edges) - 1
        dz_raw = np.diff(z_edges)
        scale_factor = padded_height / np.sum(dz_raw)
        dz_k_arr = dz_raw * scale_factor
        z_k_arr = z_min_i + np.concatenate([[0.0], np.cumsum(dz_k_arr[:-1])])
        dz_k = float(np.mean(dz_k_arr))

        nz_arr[i] = nz_k
        dz_arr[i] = dz_k
        padded_x_arr[i] = padded_extent_x
        padded_y_arr[i] = padded_extent_y
        padded_h_arr[i] = padded_height
        z_min_arr[i] = z_min_i
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
        'padded_extent_x': padded_x_arr,
        'padded_extent_y': padded_y_arr,
        'padded_height': padded_h_arr,
        'z_min_per_class': z_min_arr,
    }


def _turbulon_envelope(k, dx, dy, dz, support_factor=SUPPORT_FACTOR, shape='mexican_hat'):
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
                           turbulon, shape='mexican_hat', fudge=None):
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

    # FUDGE FACTOR (empirical normalization correction) — see NORMALIZATION_FUDGE
    # definition near the top of this module for history and tuned values.
    response /= (NORMALIZATION_FUDGE if fudge is None else fudge)

    z_finest = z_arrays['z_arrays'][-1]
    C_k = []
    for i, k in enumerate(k_values):
        hurst_scale = float((k / outer_scale) ** H_h)
        C_profile = np.interp(z_arrays['z_arrays'][i], z_finest, response).astype(np.float32) * hurst_scale
        C_k.append(C_profile)
    return C_k


def _draw_noise(shape, rng, dist=None):
    """Draw zero-mean unit-scale noise, then apply the global NOISE_SIGN flip.

    NOISE_SIGN=-1 negates the (already zero-mean) draw, flipping its skew sign.
    Because STEAM's wavelet kernel inverts skew through the convolution, a
    LEFT-skewed S_k (e.g. NOISE_SIGN=-1 on a right-skewed Gamma) yields a
    RIGHT-skewed field -- the convective direction.

    dist=None uses the module NOISE_DIST (and applies NOISE_SIGN); dist='gaussian'
    forces a plain N(0,1) draw with no sign flip -- used for the flux-cascade
    innovations, which must stay symmetric Gaussian regardless of NOISE_DIST.
    """
    if dist == 'gaussian':
        return rng.standard_normal(shape, dtype=np.float32)
    x = _draw_noise_core(shape, rng)
    if NOISE_SIGN < 0:
        np.negative(x, out=x)
    return x


def _draw_noise_core(shape, rng):
    """Draw the sparse turbulon amplitude block, zero-mean, float32.

    Dispatches on the module global NOISE_DIST ('gaussian' or 'levy').
    Gaussian is N(0,1) (original). Levy is a (LEVY_ALPHA, LEVY_BETA) alpha-stable
    via scipy.stats.levy_stable, generated in z-slabs to cap peak memory
    (the rvs sampler is float64 with several temporaries), cast to float32,
    and centered to exactly zero sample mean so it adds no DC bias to the field.
    """
    if NOISE_DIST == 'gaussian':
        return rng.standard_normal(shape, dtype=np.float32)
    if NOISE_DIST == 'gamma':
        # centered, unit-variance Gamma(k): skew = 2/sqrt(k), exponential tail.
        k = NOISE_GAMMA_K
        x = (rng.gamma(k, 1.0, size=shape) - k) / np.sqrt(k)
        return x.astype(np.float32)
    if NOISE_DIST == 'skewnorm':
        # Azzalini skew-normal via the |N| + N construction (numpy-native, fast),
        # then standardized to zero-mean unit-variance. Gaussian tails.
        a = NOISE_SKEWNORM_A
        delta = a / np.sqrt(1.0 + a * a)
        x = (delta * np.abs(rng.standard_normal(shape))
             + np.sqrt(1.0 - delta * delta) * rng.standard_normal(shape))
        m = delta * np.sqrt(2.0 / np.pi)
        v = 1.0 - 2.0 * delta * delta / np.pi
        return ((x - m) / np.sqrt(v)).astype(np.float32)
    if NOISE_DIST != 'levy':
        raise ValueError(f"unknown NOISE_DIST {NOISE_DIST!r}")
    from scipy.stats import levy_stable
    out = np.empty(shape, dtype=np.float32)
    flat = out.reshape(-1, shape[-1]) if out.ndim > 1 else out.reshape(1, -1)
    nrow, ncol = flat.shape
    # chunk rows so each draw is ~3e7 elements (~0.24 GB f64 + temporaries)
    rstep = max(1, int(3e7 // max(ncol, 1)))
    for r0 in range(0, nrow, rstep):
        r1 = min(nrow, r0 + rstep)
        x = levy_stable.rvs(LEVY_ALPHA, LEVY_BETA, size=(r1 - r0, ncol),
                            random_state=rng)
        flat[r0:r1] = x.astype(np.float32)
    out -= np.float32(out.mean(dtype=np.float64))   # zero-mean (no DC bias)
    return out


def _sparse_noise(nx, ny, nz, factor_x, factor_y, factor_z, rng, dist=None):
    """Generate sparse noise field at oversampled resolution (see _draw_noise
    for the per-sample distribution: N(0,1) or Levy alpha-stable via NOISE_DIST).

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
        return _draw_noise((nx, ny, nz), rng, dist=dist)

    field = np.zeros((nx, ny, nz), dtype=np.float32)
    ix = np.arange(0, nx, factor_x)
    iy = np.arange(0, ny, factor_y)
    iz = np.arange(0, nz, factor_z)
    field[np.ix_(ix, iy, iz)] = _draw_noise((len(ix), len(iy), len(iz)), rng, dist=dist)
    return field


def _gradient_magnitude(field_3d, dx, dy, z_coords):
    """Compute |∇f|, the gradient magnitude (unnormalized).

    Uses periodic central differences for x,y and np.gradient for z.
    z_coords may be a scalar spacing (uniform grid) or a 1D array of
    z-positions (non-uniform grid); np.gradient handles both.

    Memory-efficient: accumulates squared gradients in-place into a single
    result buffer, using ~3 arrays peak instead of 6-7.
    """
    dx_f32 = np.float32(2 * dx)
    dy_f32 = np.float32(2 * dy)

    # X: periodic central difference → result holds grad_x**2
    result = np.roll(field_3d, -1, axis=0)
    result -= np.roll(field_3d, 1, axis=0)
    result /= dx_f32
    result **= 2

    # Y: accumulate grad_y**2 into result
    grad = np.roll(field_3d, -1, axis=1)
    grad -= np.roll(field_3d, 1, axis=1)
    grad /= dy_f32
    grad **= 2
    result += grad
    del grad

    # Z: accumulate grad_z**2 into result
    grad_z = np.gradient(field_3d, z_coords.astype(np.float32) if hasattr(z_coords, 'astype') else z_coords, axis=2)
    grad_z **= 2
    result += grad_z
    del grad_z

    np.sqrt(result, out=result)
    return result


def refine(
    parent_path,
    x_start, x_stop,
    y_start, y_stop,
    dx, dy,
    parent_group='/',
    output_group=None,
    seed=None,
    sparsity_factors=None,
    turbulon_shape=None,
    z_min=None,
    z_max=None,
    anisotropy=None,
    compress=None,
):
    """Refine a subdomain of a parent simulation to finer resolution.

    Loads a completed STEAM simulation (or a previous refinement group),
    extracts a spatial subset (with padding to capture turbulon tails),
    and continues the cascade from the parent's finest scale down to 2*dx.

    For recursive refinement, pass the group path of an existing
    refinement as parent_group (e.g. "refinements/r0").

    Parameters
    ----------
    parent_path : str or Path
        Path to the NetCDF file.
    x_start, x_stop : int
        Index range into parent x-grid for the inner subdomain.
    y_start, y_stop : int
        Index range into parent y-grid for the inner subdomain.
    dx, dy : float
        New finest horizontal resolution [m].
    parent_group : str
        NetCDF group to read parent data from. Default '/' (root).
    output_group : str or None
        NetCDF group name for output. Default: auto-generated
        "refinements/r0", "r1", ...
    seed : int or None
        Random seed. If None, derived from parent seed + group name hash.
    sparsity_factors : tuple of 3 ints or None
        If None, inherit from parent.
    turbulon_shape : str or None
        If None, inherit from parent.
    z_min : float or None
        Bottom altitude of the inset [m].  If None, inherit from parent
        (0 for a root simulation).
    z_max : float or None
        Top altitude of the inset [m].  If None, use parent's top altitude.
    anisotropy : str or None
        Grid-anisotropy function name (see _k_z). If None, inherit from
        the parent group (defaulting to 'canonical' for older files).
    compress : bool or None
        If True, 3D data variables in the output group are written with
        zlib compression at complevel=4. None (default) uses the module-level
        ``steam.constants.output_compress`` setting.

    Returns
    -------
    Path
        The parent_path (with the new group written into it).
    """
    if compress is None:
        compress = constants.output_compress
    parent_path = Path(parent_path)

    # Read parent data from the specified group
    ds = netCDF4.Dataset(parent_path, "r")
    grp = ds if parent_group == '/' else ds[parent_group]

    h_3d = grp.variables["h"][:].astype(np.float32)
    qt_3d = grp.variables["qt"][:].astype(np.float32)
    x_coords = grp.variables["x"][:]
    y_coords = grp.variables["y"][:]
    z_coords = grp.variables["z"][:]
    h_profile = grp.variables["h_profile"][:]
    qt_profile = grp.variables["qt_profile"][:]
    z_profile = grp.variables["z_profile"][:]
    k_values_parent = grp.variables["k_values"][:]
    C_h_k_parent = grp.variables["C_h_k"][:]
    C_qt_k_parent = grp.variables["C_qt_k"][:]
    if "spheroscale_profile" in grp.variables:
        spheroscale_profile = grp.variables["spheroscale_profile"][:]
    else:
        spheroscale_on_z = grp.variables["spheroscale"][:]
        spheroscale_profile = np.interp(z_profile, z_coords, spheroscale_on_z)

    parent_dx = float(grp.dx)
    parent_dy = float(grp.dy)
    domain_height = float(grp.domain_height)
    profile_dz = float(grp.profile_dz)
    surface_pressure = float(grp.surface_pressure)
    parent_seed = int(grp.seed) if int(grp.seed) != -1 else None
    h_min = float(grp.h_min)
    h_max = float(grp.h_max)
    qt_min = float(grp.qt_min)
    qt_max = float(grp.qt_max)
    min_distance_to_ground = int(grp.min_distance_to_ground)
    if not hasattr(grp, 'n_scale_classes_per_dyad'):
        raise ValueError(
            f"Parent group {parent_group!r} has no n_scale_classes_per_dyad "
            f"attribute; cannot extend the cascade. Regenerate the parent "
            f"with the current version of simulate()."
        )
    parent_n_per_dyad = int(grp.n_scale_classes_per_dyad)
    parent_gap = 2.0 ** (1.0 / parent_n_per_dyad)

    if sparsity_factors is None:
        sparsity_factors = tuple(int(v) for v in grp.sparsity_factors)
    if turbulon_shape is None:
        turbulon_shape = grp.turbulon_shape if hasattr(grp, 'turbulon_shape') else 'mexican_hat'
    if anisotropy is None:
        anisotropy = grp.anisotropy if hasattr(grp, 'anisotropy') else 'canonical'
    if anisotropy not in VALID_ANISOTROPY:
        raise ValueError(
            f"anisotropy must be one of {VALID_ANISOTROPY}, got {anisotropy!r}"
        )

    parent_z_min = float(grp.domain_z_min) if hasattr(grp, 'domain_z_min') else 0.0

    # Determine existing refinement groups for auto-naming
    if output_group is None:
        existing = []
        if "refinements" in ds.groups:
            existing = list(ds.groups["refinements"].groups.keys())
        idx = 0
        while f"r{idx}" in existing:
            idx += 1
        output_group = f"refinements/r{idx}"

    ds.close()

    # Resolve vertical extent of the inset
    parent_z_max = parent_z_min + domain_height
    if z_min is None:
        z_min = parent_z_min
    if z_max is None:
        z_max = parent_z_max
    if z_min < parent_z_min - 1e-6 or z_max > parent_z_max + 1e-6:
        raise ValueError(
            f"Requested altitude range [{z_min}, {z_max}] m exceeds parent "
            f"range [{parent_z_min}, {parent_z_max}] m"
        )
    inset_height = z_max - z_min
    if inset_height <= 0:
        raise ValueError(f"z_max ({z_max}) must be greater than z_min ({z_min})")

    # New outer scale = parent's finest k (the "ceiling" for this cascade).
    # k_values start one log-step below new_outer_scale to avoid re-adding
    # turbulons at the overlap scale with the parent.
    new_outer_scale = float(k_values_parent[-1])

    parent_nx = h_3d.shape[0]
    parent_ny = h_3d.shape[1]

    # Extend the parent's log-spaced cascade by extrapolating with the same
    # gap factor until k > 2*dx. Round the refining factor down to the
    # largest gap^n that still satisfies new_outer_scale/gap^n >= 2*dx.
    s_x, s_y, s_z = sparsity_factors
    size_class_gap_factor = parent_gap
    n_classes = int(np.floor(
        np.log(new_outer_scale / (2 * dx)) / np.log(size_class_gap_factor)
    ))
    if n_classes < 1:
        raise ValueError(
            f"Refinement cannot add any classes: new_outer_scale="
            f"{new_outer_scale}, dx={dx}, gap={size_class_gap_factor}. "
            f"Decrease dx or use a parent with a larger finest scale."
        )

    k_values = new_outer_scale / size_class_gap_factor ** np.arange(1, n_classes + 1)

    spheroscale_mean = float(np.mean(spheroscale_profile))
    k_z_values = _k_z(anisotropy, k_values, spheroscale_mean)

    # Inner subdomain physical extent
    inner_nx = x_stop - x_start
    inner_ny = y_stop - y_start
    inner_extent_x = inner_nx * parent_dx
    inner_extent_y = inner_ny * parent_dy

    # Validate: inner extent must be integer multiple of new_outer_scale
    for extent, axis_name in ((inner_extent_x, 'x'), (inner_extent_y, 'y')):
        n_tiles = extent / new_outer_scale
        if abs(n_tiles - round(n_tiles)) > 1e-9:
            raise ValueError(
                f"Inner subdomain {axis_name}-extent ({extent} m) must be an "
                f"integer multiple of new_outer_scale ({new_outer_scale} m), "
                f"got ratio={n_tiles}"
            )

    # Detect spanning dims and parent periodicity.
    # Root sim is periodic in x/y; any nested refinement is non-periodic.
    spans_x = (inner_nx == parent_nx)
    spans_y = (inner_ny == parent_ny)
    parent_is_periodic = (parent_group == '/')

    # Per-class physical pad on each side.
    # Spanning dim → pad = 0 so FFT period = inner_extent = parent_extent.
    # Non-spanning dim → pad = SUPPORT_FACTOR * k_i, shrinking with cascade.
    if spans_x:
        pad_x_per_class = np.zeros(n_classes)
    else:
        pad_x_per_class = SUPPORT_FACTOR * k_values
    if spans_y:
        pad_y_per_class = np.zeros(n_classes)
    else:
        pad_y_per_class = SUPPORT_FACTOR * k_values

    # z pad: 0 at parent ground/top (parent was already zero-padded there);
    # else SUPPORT_FACTOR * k_z_i, clipped to available parent room.
    at_ground = (z_min <= parent_z_min + 1e-6)
    at_top = (z_max >= parent_z_max - 1e-6)

    # If the inset's bottom is above the parent ground, the scalar surface
    # pressure no longer describes the pressure at z=z_min; read the parent's
    # 3D pressure field at levels bracketing z_min for a 2D p_bottom.
    if not at_ground:
        with netCDF4.Dataset(parent_path, "r") as _ds_p:
            _grp_p = _ds_p if parent_group == '/' else _ds_p[parent_group]
            if 'p' not in _grp_p.variables:
                raise ValueError(
                    f"Refining with z_min ({z_min}) above the parent bottom "
                    f"({parent_z_min}) requires pressure diagnostics on the "
                    f"parent group {parent_group!r}. Run "
                    f"steam.thermodynamics.compute_diagnostics on the parent "
                    f"first."
                )
            z_idx_lo = int(np.searchsorted(z_coords, z_min, side='right')) - 1
            z_idx_lo = max(0, min(len(z_coords) - 2, z_idx_lo))
            p_slice_lo = _grp_p.variables['p'][:, :, z_idx_lo].astype(np.float32)
            p_slice_hi = _grp_p.variables['p'][:, :, z_idx_lo + 1].astype(np.float32)
        parent_z_lo = float(z_coords[z_idx_lo])
        parent_z_hi = float(z_coords[z_idx_lo + 1])
    else:
        p_slice_lo = p_slice_hi = None
        parent_z_lo = parent_z_hi = None

    if at_ground:
        pad_z_below_per_class = np.zeros(n_classes)
    else:
        pad_z_below_per_class = np.minimum(
            SUPPORT_FACTOR * k_z_values, z_min - parent_z_min
        )
    if at_top:
        pad_z_above_per_class = np.zeros(n_classes)
    else:
        pad_z_above_per_class = np.minimum(
            SUPPORT_FACTOR * k_z_values, parent_z_max - z_max
        )

    # Pad cell counts at parent resolution (used only for class-0 extraction
    # from the parent's netCDF grid).
    pad_cells_x = int(math.ceil(pad_x_per_class[0] / parent_dx)) if not spans_x else 0
    pad_cells_y = int(math.ceil(pad_y_per_class[0] / parent_dy)) if not spans_y else 0

    # Non-periodic parent: the extraction can't wrap, so the nest must leave
    # enough room on each non-spanning side. Reject otherwise — (4)'s
    # shrinking pad only relaxes cascade memory, not the class-0 extraction.
    if not parent_is_periodic:
        if not spans_x and (x_start < pad_cells_x or x_stop > parent_nx - pad_cells_x):
            raise ValueError(
                f"Refinement of non-periodic parent {parent_group!r} at x=[{x_start},{x_stop}] "
                f"is too close to the parent's x boundary; need at least {pad_cells_x} cells "
                f"of room on each side (parent_nx={parent_nx})"
            )
        if not spans_y and (y_start < pad_cells_y or y_stop > parent_ny - pad_cells_y):
            raise ValueError(
                f"Refinement of non-periodic parent {parent_group!r} at y=[{y_start},{y_stop}] "
                f"is too close to the parent's y boundary; need at least {pad_cells_y} cells "
                f"of room on each side (parent_ny={parent_ny})"
            )

    # Build extraction index arrays at parent resolution.
    if spans_x:
        x_indices = np.arange(x_start, x_stop) % parent_nx
    elif parent_is_periodic:
        x_indices = np.arange(x_start - pad_cells_x, x_stop + pad_cells_x) % parent_nx
    else:
        x_indices = np.arange(x_start - pad_cells_x, x_stop + pad_cells_x)
    if spans_y:
        y_indices = np.arange(y_start, y_stop) % parent_ny
    elif parent_is_periodic:
        y_indices = np.arange(y_start - pad_cells_y, y_stop + pad_cells_y) % parent_ny
    else:
        y_indices = np.arange(y_start - pad_cells_y, y_stop + pad_cells_y)

    # z slice with padded range
    z_range_low = z_min - pad_z_below_per_class[0]
    z_range_high = z_max + pad_z_above_per_class[0]
    z_indices = np.where(
        (z_coords >= z_range_low - 1e-6) & (z_coords < z_range_high + 1e-6)
    )[0]
    if len(z_indices) == 0:
        raise ValueError(
            f"No parent z-levels found in [{z_range_low}, {z_range_high}] m. "
            f"Parent z ranges from {z_coords[0]:.1f} to {z_coords[-1]:.1f} m"
        )
    z_coords_slice = z_coords[z_indices]

    h_pad = h_3d[np.ix_(x_indices, y_indices, z_indices)]
    qt_pad = qt_3d[np.ix_(x_indices, y_indices, z_indices)]

    # Compute perturbation: subtract mean profile
    h_mean_1d = np.interp(z_coords_slice, z_profile, h_profile).astype(np.float32)
    qt_mean_1d = np.interp(z_coords_slice, z_profile, qt_profile).astype(np.float32)
    h_pert_pad = h_pad - h_mean_1d[np.newaxis, np.newaxis, :]
    qt_pert_pad = qt_pad - qt_mean_1d[np.newaxis, np.newaxis, :]

    # Compute grids with per-class pad (shrinks as k shrinks in non-spanning dims).
    grids = _compute_all_grids(
        k_values, inner_extent_x, inner_extent_y, inset_height,
        sparsity_factors, spheroscale_profile, z_profile, z_min=z_min,
        pad_x_per_class=pad_x_per_class,
        pad_y_per_class=pad_y_per_class,
        pad_z_below_per_class=pad_z_below_per_class,
        pad_z_above_per_class=pad_z_above_per_class,
        anisotropy=anisotropy,
    )

    # Inherit amplitude normalization from the parent by extrapolating the
    # parent's (k/outer)^H_h law with the shared gap factor. The parent's
    # last-class amplitude is the anchor: C_parent[-1](z) lives on the
    # parent's output z-grid and already encodes response_root *
    # (k_parent_last/outer_root)^H_h * spectral_width_correction. Because
    # gap is preserved, the correction is the same for every descendant,
    # so multiplying by (k_child/k_parent_last)^H_h reproduces the unified
    # cascade's amplitude at each new class — no re-measurement, no
    # discontinuity at the overlap scale.
    anchor_k = float(k_values_parent[-1])
    anchor_C_h = np.asarray(C_h_k_parent[-1], dtype=np.float64)
    anchor_C_qt = np.asarray(C_qt_k_parent[-1], dtype=np.float64)
    # Trim fill values in case the parent's C_k array was padded to
    # nz_k_max (refine writes per-class rows of varying length).
    anchor_C_h = anchor_C_h[:len(z_coords)]
    anchor_C_qt = anchor_C_qt[:len(z_coords)]
    C_h_k = []
    C_qt_k = []
    for i, k in enumerate(k_values):
        scale = float((k / anchor_k) ** H_h)
        z_i = grids['z_arrays'][i]
        C_h_k.append(
            (np.interp(z_i, z_coords, anchor_C_h) * scale).astype(np.float32)
        )
        C_qt_k.append(
            (np.interp(z_i, z_coords, anchor_C_qt) * scale).astype(np.float32)
        )

    # Seed handling
    if seed is None and parent_seed is not None:
        seed = parent_seed + hash(output_group) % (2**31)
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(n_classes)

    # Run cascade on padded domain
    if FLUX_CASCADE:
        raise NotImplementedError(
            "FLUX_CASCADE is not supported in refine()/nested refinement; "
            "run the flux cascade only on root simulate() calls.")
    h_pert_refined, qt_pert_refined, _flux_unused, final_grid = cascade_loop(
        h_profile, qt_profile, z_profile,
        grids,
        C_h_k, C_qt_k,
        h_min, h_max, qt_min, qt_max,
        min_distance_to_ground,
        sparsity_factors,
        child_seeds,
        h_perturbation=h_pert_pad,
        qt_perturbation=qt_pert_pad,
        turbulon_shape=turbulon_shape,
        zero_bottom=at_ground,
        zero_top=at_top,
    )

    # Trim pad region to get inner subdomain (x, y, z).
    final_nx = final_grid['nx']
    final_ny = final_grid['ny']
    inner_nx_fine = int(round(inner_extent_x / final_grid['dx']))
    inner_ny_fine = int(round(inner_extent_y / final_grid['dy']))
    trim_x = (final_nx - inner_nx_fine) // 2
    trim_y = (final_ny - inner_ny_fine) // 2

    # z trim: find cells whose left-edge lies within [z_min, z_max).
    z_full = final_grid['z']
    dz_full = final_grid['dz']
    z_inner_mask = (z_full >= z_min - 1e-6) & (z_full < z_max - 1e-6)
    z_inner_indices = np.where(z_inner_mask)[0]
    if len(z_inner_indices) == 0:
        # Fallback: use centered trim based on inset_height
        inner_nz_fine = int(round(inset_height / float(final_grid['dz'].mean())))
        z_trim_start = (len(z_full) - inner_nz_fine) // 2
        z_inner_indices = np.arange(z_trim_start, z_trim_start + inner_nz_fine)
    z_trim_start = int(z_inner_indices[0])
    z_trim_stop = int(z_inner_indices[-1]) + 1

    h_pert_inner = h_pert_refined[trim_x:trim_x+inner_nx_fine,
                                   trim_y:trim_y+inner_ny_fine,
                                   z_trim_start:z_trim_stop]
    qt_pert_inner = qt_pert_refined[trim_x:trim_x+inner_nx_fine,
                                     trim_y:trim_y+inner_ny_fine,
                                     z_trim_start:z_trim_stop]

    # Construct final 3D fields
    z_final = z_full[z_trim_start:z_trim_stop].astype(np.float32)
    dz_final = dz_full[z_trim_start:z_trim_stop].astype(np.float32)
    h_mean_final = np.interp(z_final, z_profile, h_profile).astype(np.float32)
    qt_mean_final = np.interp(z_final, z_profile, qt_profile).astype(np.float32)
    spheroscale_final = np.interp(z_final, z_profile, spheroscale_profile).astype(np.float32)

    h_3d_out = np.ascontiguousarray(h_mean_final[np.newaxis, np.newaxis, :] + h_pert_inner)
    qt_3d_out = np.ascontiguousarray(qt_mean_final[np.newaxis, np.newaxis, :] + qt_pert_inner)

    np.clip(h_3d_out, h_min, h_max, out=h_3d_out)
    np.clip(qt_3d_out, qt_min, qt_max, out=qt_3d_out)

    nx_out = h_3d_out.shape[0]
    ny_out = h_3d_out.shape[1]
    dx_final = inner_extent_x / nx_out
    dy_final = inner_extent_y / ny_out
    # World-absolute coords: parent's own first-cell world position + local
    # in-parent offset. Using only x_start * parent_dx skews deeper refinements
    # whose parent doesn't start at world origin (e.g. a cube carved from a
    # strip that is not y-centered on 0).
    parent_x_origin = float(x_coords[0])
    parent_y_origin = float(y_coords[0])
    x_out = (np.arange(nx_out, dtype=np.float32) * dx_final
             + x_start * parent_dx + parent_x_origin)
    y_out = (np.arange(ny_out, dtype=np.float32) * dy_final
             + y_start * parent_dy + parent_y_origin)

    # 2D starting pressure from parent's p, interpolated in z to z_final[0]
    # and bilinearly upsampled to the child horizontal grid.
    if p_slice_lo is not None:
        z_target = float(z_final[0])
        if parent_z_hi > parent_z_lo:
            w = (z_target - parent_z_lo) / (parent_z_hi - parent_z_lo)
            w = float(np.clip(w, 0.0, 1.0))
        else:
            w = 0.0
        p_slice = (1.0 - w) * p_slice_lo + w * p_slice_hi
        if parent_is_periodic:
            ix_inner = np.arange(x_start, x_stop) % parent_nx
            iy_inner = np.arange(y_start, y_stop) % parent_ny
        else:
            ix_inner = np.arange(x_start, x_stop)
            iy_inner = np.arange(y_start, y_stop)
        p_bottom_inner = p_slice[np.ix_(ix_inner, iy_inner)]
        if p_bottom_inner.shape == (nx_out, ny_out):
            p_bottom_field = p_bottom_inner.astype(np.float32)
        else:
            p_bottom_field = zoom_bilinear(p_bottom_inner, (nx_out, ny_out))
    else:
        p_bottom_field = None

    C_h_L = float(np.mean(C_h_k[0]))
    C_qt_L = float(np.mean(C_qt_k[0]))

    # Align stored C_k to the output z grid (see matching block in simulate()
    # for rationale). For refine with elevated / narrow-z insets this also
    # trims off the pad region, so descendants can inherit cleanly.
    C_h_k_stored = [np.interp(z_final, grids['z_arrays'][i], C_h_k[i]).astype(np.float32)
                    for i in range(len(k_values))]
    C_qt_k_stored = [np.interp(z_final, grids['z_arrays'][i], C_qt_k[i]).astype(np.float32)
                     for i in range(len(k_values))]

    simulation_params = {
        'nx': nx_out,
        'ny': ny_out,
        'dx': dx_final,
        'dy': dy_final,
        'dz': dz_final,
        'outer_scale': new_outer_scale,
        'spheroscale': spheroscale_final,
        'spheroscale_profile': spheroscale_profile,
        'domain_height': inset_height,
        'domain_z_min': z_min,
        'profile_dz': profile_dz,
        'sparsity_factors': sparsity_factors,
        'n_scale_classes_per_dyad': parent_n_per_dyad,
        'surface_pressure': surface_pressure,
        'seed': seed,
        'C_h_L': C_h_L,
        'C_qt_L': C_qt_L,
        'n_large_turbulons': 0,  # refinement doesn't define this
        'H_h': H_h,
        'H_z': H_z,
        'h_min': h_min,
        'h_max': h_max,
        'qt_min': qt_min,
        'qt_max': qt_max,
        'min_distance_to_ground': min_distance_to_ground,
        'turbulon_shape': turbulon_shape,
        'anisotropy': anisotropy,
        'parent_group': parent_group,
        'parent_x_slice': np.array([x_start, x_stop], dtype=np.int32),
        'parent_y_slice': np.array([y_start, y_stop], dtype=np.int32),
        'parent_x_offset': float(x_start * parent_dx),
        'parent_y_offset': float(y_start * parent_dy),
    }
    if p_bottom_field is not None:
        simulation_params['p_bottom'] = p_bottom_field

    write_netcdf(
        parent_path, h_3d_out, qt_3d_out,
        x_out, y_out, z_final,
        h_profile, qt_profile, z_profile.astype(np.float32),
        k_values, k_z_values, C_h_k_stored, C_qt_k_stored,
        simulation_params,
        group=output_group,
        compress=compress,
    )
    return parent_path
