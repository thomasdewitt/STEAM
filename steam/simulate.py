"""Core STEAM cascade algorithm."""

import numpy as np
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

# ─── FLUX CASCADE ────────────────────────────────────────────────────────────
# F is dimensionless and initialized to one. One octave is taken per dyadic size
# class. Each substep multiplies the flux by exp(gamma), where gamma is an
# EXTREMAL (maximally skewed, beta=-1) Levy alpha-stable field -- the log-
# generator of a universal-multifractal random measure, so the cascade converges
# to a true multifractal as the substep count grows. gamma is scaled by
# FLUX_SCALE / N_FLUX_SUBSTEPS**(1/alpha): by alpha-stability the per-octave
# generator is EXACTLY invariant in distribution under the substep count, so more
# substeps only make the bounded-below update  F += conv(psi, (exp(gamma)-1)*F)
# clip at zero less often. gamma is shifted so <exp(gamma)> = 1 exactly (the
# alpha-generalization of the lognormal -sigma^2/2 shift; see LEVY_LOG_MEAN),
# which conserves the flux mean. Scalar turbulons inherit the signed first-
# substep multiplier noise, so they are skewed too (SCALAR_NOISE_SIGN).
#
# Realized intermittency calibrates as  C1 ~= A * FLUX_SCALE**alpha  (fit in
# turbulon-analysis/calibration); at alpha=2 the generator is Gaussian and the
# whole scheme reduces to the lognormal cascade.
FLUX_SCALE = 0.5
FLUX_ALPHA = 1.8
N_FLUX_SUBSTEPS = 4
# Sign of the scalar turbulon amplitude relative to the flux multiplier noise.
# The extremal generator is heavy-tailed on the low-multiplier side, so the two
# signs give oppositely skewed scalar fields; +1 gives convective right-skew in
# the h'/qt' interior (256x256x64 seed 20260715: interior skew ~ +2.9 for +1
# vs ~ +0.15 for -1), chosen from the 2026-07-15 both-signs experiment.
SCALAR_NOISE_SIGN = 1.0


def _extremal_levy(alpha, size, rng):
    """Extremal (maximally skewed, beta=-1) Levy alpha-stable draws.

    Chambers-Mallows-Stuck generator, reusing the convention of Thomas's
    ``scaleinvariance.extremal_levy`` (its universal-multifractal log-generator).
    The distribution is maximally skewed toward -inf, so the RIGHT tail is light
    and <exp(gamma)> is finite -- the property that lets the multiplier be
    renormalized to unit mean. At alpha=2 it is exactly N(0, 2), i.e. the
    lognormal case. Valid for alpha in (1, 2].

    Assembled with in-place float32 ops (commutative reorderings of the single
    CMS expression) so only a few field-sized buffers are ever live -- the
    generator is drawn at the flux cascade's finest resolution.
    """
    alpha = float(alpha)
    phi0 = np.float32(-(np.pi / 2.0) * (1.0 - abs(1.0 - alpha)) / alpha)
    sign = np.float32(1.0 if alpha > 1.0 else -1.0)
    eps = np.float32(1e-12)

    phi = (rng.random(size, dtype=np.float32) - np.float32(0.5)) * np.float32(np.pi)
    R = rng.random(size, dtype=np.float32)       # -log(U) ~ Exp(1), all float32
    np.clip(R, eps, None, out=R)
    np.log(R, out=R)
    R *= np.float32(-1.0)

    factor = np.clip(np.cos(phi), eps, None)     # (|a-1| cos phi) ** (-1/a)
    factor *= np.float32(abs(alpha - 1.0))
    factor **= np.float32(-1.0 / alpha)

    shifted = phi - phi0
    shifted *= np.float32(alpha)                 # a (phi - phi0)
    phi -= shifted                               # phi - a(phi - phi0); reuse phi
    np.cos(phi, out=phi)
    np.clip(phi, eps, None, out=phi)
    phi /= R                                     # cos(...)/R
    phi **= np.float32((1.0 - alpha) / alpha)    # -> tail factor
    del R

    np.sin(shifted, out=shifted)                 # sin(a(phi - phi0))
    shifted *= sign
    shifted *= factor
    shifted *= phi
    return shifted


# log<exp(gamma_0)> for the unit-scale extremal generator above. Subtracting
# LEVY_LOG_MEAN * scale**alpha from a scale-times-gamma_0 draw sets the
# multiplier to unit mean (the alpha-generalization of the lognormal -sigma^2/2;
# it equals 1 at alpha=2). There is no closed form in this generator's scale
# convention, so it is measured once from a large deterministic draw (in float64
# for an accurate mean).
LEVY_LOG_MEAN = float(np.log(np.mean(np.exp(
    _extremal_levy(FLUX_ALPHA, 8_000_000, np.random.default_rng(0)).astype(np.float64)
))))


def _sparse_levy(nx, ny, nz, factor_x, factor_y, factor_z, alpha, rng):
    """Sparse extremal-Levy generator field, one draw per turbulon center.

    Zero everywhere except every s grid points (one turbulon center per k
    spacing at the oversampled resolution); at s=1 every cell is a center.
    """
    if factor_x == 1 and factor_y == 1 and factor_z == 1:
        return _extremal_levy(alpha, nx * ny * nz, rng).reshape(nx, ny, nz)

    field = np.zeros((nx, ny, nz), dtype=np.float32)
    ix = np.arange(0, nx, factor_x)
    iy = np.arange(0, ny, factor_y)
    iz = np.arange(0, nz, factor_z)
    field[np.ix_(ix, iy, iz)] = _extremal_levy(
        alpha, len(ix) * len(iy) * len(iz), rng
    ).reshape(len(ix), len(iy), len(iz))
    return field


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
    if n_scale_classes_per_dyad != 1:
        raise ValueError(
            "The flux cascade requires one dyadic size class per octave; "
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
        unit_turbulon, turbulon_shape,
    )
    C_qt_k = _compute_normalization(
        qt_on_finest, vertical_outer_scale_grid_pts,
        k_values, outer_scale, grids,
        unit_turbulon, turbulon_shape,
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

    # The perturbation buffers are no longer needed separately. Reuse them for
    # final fields instead of holding two additional full-domain arrays.
    h_pert += h_mean_final[np.newaxis, np.newaxis, :]
    qt_pert += qt_mean_final[np.newaxis, np.newaxis, :]
    h_3d = np.ascontiguousarray(h_pert)
    qt_3d = np.ascontiguousarray(qt_pert)
    del h_pert, qt_pert

    np.clip(h_3d, h_min, h_max, out=h_3d)
    np.clip(qt_3d, qt_min, qt_max, out=qt_3d)

    flux_3d = np.ascontiguousarray(flux_field)

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
        'flux_noise_scale': FLUX_SCALE,
        'flux_alpha': FLUX_ALPHA,
        'n_flux_substeps': N_FLUX_SUBSTEPS,
        'scalar_noise_sign': SCALAR_NOISE_SIGN,
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
    flux : ndarray
        Final dimensionless unit-mean flux.
    final_grid_info : dict
        Keys nx, ny, nz, dx, dy, dz (mean), z (1D coordinate array) from
        the last (finest) iteration.
    """
    s_x, s_y, s_z = sparsity_factors
    n_classes = len(grids['k'])
    n_zero = int(round(2 * s_z * min_distance_to_ground))
    if not isinstance(N_FLUX_SUBSTEPS, int) or N_FLUX_SUBSTEPS < 1:
        raise ValueError(
            f"N_FLUX_SUBSTEPS must be a positive integer, got {N_FLUX_SUBSTEPS}"
        )

    if n_classes > 1:
        class_ratios = np.asarray(grids['k'][:-1]) / np.asarray(grids['k'][1:])
        if not np.allclose(class_ratios, 2.0):
            raise ValueError(
                "The flux cascade requires dyadic size classes (one octave per step)"
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
    flux = None

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
                    flux = flux[start:start+keep, :, :]
                if padded_y_i < prev_padded_extent_y - eps:
                    cur_ny = h_perturbation.shape[1]
                    keep = int(round(padded_y_i / prev_dy))
                    keep = max(1, min(cur_ny, keep))
                    start = (cur_ny - keep) // 2
                    h_perturbation = h_perturbation[:, start:start+keep, :]
                    qt_perturbation = qt_perturbation[:, start:start+keep, :]
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
                    flux = flux[:, :, start:start+keep]

            if h_perturbation.shape != (nx_k, ny_k, nz_k):
                h_perturbation = zoom_trilinear(h_perturbation, (nx_k, ny_k, nz_k))
                qt_perturbation = zoom_trilinear(qt_perturbation, (nx_k, ny_k, nz_k))
                flux = zoom_trilinear(flux, (nx_k, ny_k, nz_k))

        # Interpolate mean profiles to current vertical grid
        h_mean_1d = np.interp(z_k, z_profile, h_profile).astype(np.float32)
        qt_mean_1d = np.interp(z_k, z_profile, qt_profile).astype(np.float32)

        # Extremal-Levy multiplier — same S_k for both h and qt (Apxeq:mean turbulon amplitude)
        if use_per_class_seeds:
            rng = np.random.default_rng(seeds_or_rng[i])
        else:
            rng = seeds_or_rng

        S_k, _ = _advance_flux(
            flux, rng, kernel, FLUX_SCALE, N_FLUX_SUBSTEPS,
            sparsity_factors, n_zero, zero_bottom, zero_top,
        )

        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})           computing G, convolutions...', end='\r')

        # Process h and qt sequentially to halve peak memory
        for (perturbation_field, mean_1d, var_min, var_max, C_k_i) in (
            (h_perturbation,  h_mean_1d,  h_min,  h_max,  C_h_k[i]),
            (qt_perturbation, qt_mean_1d, qt_min, qt_max, C_qt_k[i]),
        ):
            running_sum = perturbation_field + mean_1d[np.newaxis, np.newaxis, :]

            if WEIGHTING == 'field':
                G = None
            else:
                # GRADIENT weighting (original): |grad(field)| * soft-clamp.
                # Compute it before reusing running_sum as the soft-clip buffer.
                G = _gradient_magnitude(running_sum, dx_k, dy_k, z_k)

            # Soft-clamp in the no-longer-needed running-field buffer. Reordering
            # this after the gradient removes one full-domain live array at the
            # gradient peak without changing the calculation.
            running_sum -= np.float32(var_min)
            running_sum /= np.float32(var_max - var_min)
            np.clip(running_sum, 0, 1, out=running_sum)
            running_sum *= (1 - running_sum)
            if G is None:
                # FIELD weighting: the soft-clamp IS the weight (no gradient).
                G = running_sum
            else:
                G *= running_sum
                del running_sum
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


def _advance_flux(
    flux, rng, kernel, flux_noise_scale, n_flux_substeps,
    sparsity_factors, n_zero=0, zero_bottom=False, zero_top=False,
):
    """Advance the dimensionless flux cascade by one size class.

    Each substep multiplies the flux by exp(gamma) with gamma an extremal-Levy
    generator field (see the FLUX CASCADE note), scaled and shifted to unit
    mean, giving the bounded-below update F += conv(psi, (exp(gamma)-1)*F). The
    scalar turbulon amplitude carries the SIGNED first-substep multiplier noise
    and the entering flux, normalized to mean absolute value one:
    S_k = SCALAR_NOISE_SIGN * (exp(gamma_1)-1) * F_{k-1} / <|exp(gamma_1)-1|>.
    """
    s_x, s_y, s_z = sparsity_factors
    substep_scale = flux_noise_scale / n_flux_substeps ** (1.0 / FLUX_ALPHA)
    shift = np.float32(LEVY_LOG_MEAN * substep_scale ** FLUX_ALPHA)
    substep_scale = np.float32(substep_scale)
    scalar_amplitude = None
    substeps = []

    for substep in range(n_flux_substeps):
        gamma = _sparse_levy(*flux.shape, s_x, s_y, s_z, FLUX_ALPHA, rng)
        if zero_bottom and n_zero > 0:
            gamma[:, :, :n_zero] = 0
        if zero_top and n_zero > 0:
            gamma[:, :, -n_zero:] = 0

        # Unit-mean multiplier noise exp(gamma)-1 at the turbulon centers, and
        # exactly 0 off-centers (no turbulon there); bounded below by -1.
        gamma *= substep_scale
        noise = np.expm1(gamma - shift)
        noise[gamma == 0.0] = np.float32(0.0)

        if substep == 0:
            mean_abs = np.abs(noise).sum() / max(np.count_nonzero(noise), 1)
            scalar_amplitude = noise * flux
            if mean_abs > 0:
                scalar_amplitude *= np.float32(SCALAR_NOISE_SIGN / mean_abs)

        noise *= flux
        flux += CONVOLVE(noise, kernel)

        n_clipped = int(np.count_nonzero(flux < 0))
        np.maximum(flux, np.float32(0.0), out=flux)
        mean_flux = flux.mean(axis=(0, 1), keepdims=True)
        empty_levels = (mean_flux <= 0).reshape(-1)
        if np.any(empty_levels):
            flux[:, :, empty_levels] = np.float32(1.0)
            mean_flux[:, :, empty_levels] = np.float32(1.0)
        flux /= np.where(mean_flux > 0, mean_flux, np.float32(1.0))

        substeps.append({
            'substep': substep + 1,
            'n_clipped': n_clipped,
            'n_points': int(flux.size),
            'clip_fraction': n_clipped / flux.size,
        })

    total_clipped = sum(step['n_clipped'] for step in substeps)
    total_points = int(flux.size) * n_flux_substeps
    diagnostics = {
        'n_clipped': total_clipped,
        'n_points': total_points,
        'clip_fraction': total_clipped / total_points,
        'substeps': substeps,
    }
    return scalar_amplitude, diagnostics


def simulate_flux_only(
    grids,
    sparsity_factors=(1, 1, 1),
    seed=None,
    flux_noise_scale=None,
    n_flux_substeps=None,
    min_distance_to_ground=0,
    turbulon_shape='mexican_hat',
    zero_bottom=False,
    zero_top=False,
):
    """Run only the unit-mean flux cascade on root-simulation grids.

    This is a cheap calibration/diagnostic path: it performs one convolution
    or more convolutions per size class and skips the scalar profiles, gradients, and scalar
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
    n_substeps = N_FLUX_SUBSTEPS if n_flux_substeps is None else n_flux_substeps
    if not isinstance(n_substeps, int) or n_substeps < 1:
        raise ValueError(
            f"n_flux_substeps must be a positive integer, got {n_substeps}"
        )

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
        scalar_amplitude, advance = _advance_flux(
            flux, rng, kernel, c, n_substeps, sparsity_factors,
            n_zero, zero_bottom, zero_top,
        )
        del scalar_amplitude

        total_clipped += advance['n_clipped']
        total_points += advance['n_points']
        steps.append({
            'k': float(k),
            **advance,
        })

    diagnostics = {
        'flux_noise_scale': float(c),
        'n_flux_substeps': n_substeps,
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
    envelope = shape_fn(r_norm_sq, k).astype(np.float32)
    envelope -= envelope.mean()
    return envelope


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

    # FUDGE FACTOR (empirical normalization correction) — see NORMALIZATION_FUDGE
    # definition near the top of this module for history and tuned values.
    response /= NORMALIZATION_FUDGE

    z_finest = z_arrays['z_arrays'][-1]
    C_k = []
    for i, k in enumerate(k_values):
        hurst_scale = float((k / outer_scale) ** H_h)
        C_profile = np.interp(z_arrays['z_arrays'][i], z_finest, response).astype(np.float32) * hurst_scale
        C_k.append(C_profile)
    return C_k


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
