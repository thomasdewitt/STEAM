"""Core STEAM cascade algorithm."""

import zlib
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
    available_memory_bytes,
    fft_convolution_bytes,
    MEMORY_HEADROOM_BYTES,
)
from .output import write_netcdf

CONVOLVE = convolve_fft_xy_oa_z
# Kernel truncation radius in units of k, and (for nests) the halo width.
# At 3k the envelope is 1.5e-18 of its peak -- eleven orders below float32
# resolution, so nothing the dtype can represent is discarded -- while the
# kernel is (3/5)^3 = 0.24x the cells of the former 5k. (2k would be 3.3e-8,
# within a factor of 4 of float32 eps, so it is not safe.) The discrete
# admissibility correction in _turbulon_envelope is independent of this
# choice to 10 digits.
#
# Changing it does change the realized field: the convolution itself moves
# only at roundoff (2.8e-7 relative), but the flux cascade's clip-at-zero
# and renormalize is nonlinear and multiplicative, so a roundoff-level
# perturbation at the coarsest class amplifies down the cascade until the
# realization is entirely different (88% of cells differ by >1e-5). The
# STATISTICS are preserved -- flux volume mean 0.95498 -> 0.95538 (0.04%),
# clipped fraction 0.1491 -> 0.1476 -- but any test asserting a tight
# tolerance on a single realization will move.
SUPPORT_FACTOR = 3

# ─── NESTED REFINEMENT ───────────────────────────────────────────────────────
# refine() continues the cascade of a completed simulation over a subdomain,
# at size classes below the parent's finest. The nest inherits the parent's
# scalar perturbations AND flux over the subdomain plus a halo, and then runs
# the IDENTICAL per-class loop. The only change is scope: every realized-mean
# statistic that the root computes over its whole domain is computed over the
# nest's own extent instead. Global means are no longer available; that
# approximation, taken over the nesting domain, is deliberate.
#
# The halo ("pad") is context, not domain: it exists so the tails of turbulons
# centered outside the nest reach into it, and it is discarded on output. Like
# the ghost cells of a nested LES it is excluded from every domain statistic
# (see _inner_view).

# ─── FLUX CASCADE ────────────────────────────────────────────────────────────
# F is dimensionless and initialized to one. The flux is advanced exactly ONCE
# per size class: each advance multiplies the flux by exp(gamma), where gamma
# is an EXTREMAL (maximally skewed, beta=-1) Levy alpha-stable field -- the
# log-generator of a universal-multifractal random measure. gamma is scaled by
# FLUX_SCALE / n_scale_classes_per_dyad**(1/alpha): adjacent classes are
# separated by scale ratio 2**(1/n), so by alpha-stability the per-OCTAVE
# generator is exactly invariant in distribution under the class density --
# n_scale_classes_per_dyad is the single cascade-density knob, controlling
# both the scalar size classes and the flux cascade. gamma is shifted so
# <exp(gamma)> = 1 exactly (the alpha-generalization of the lognormal
# -sigma^2/2 shift; see LEVY_LOG_MEAN), which conserves the flux mean under
# the bounded-below update  F += conv(psi, (exp(gamma)-1)*F), clipped at
# zero. The clip is the one thing that breaks that conservation, so after
# each class the field is rescaled by a single scalar restoring the VOLUME
# mean it entered the class with -- a corrector for that step's clip bias and
# nothing more. It imposes no value on the flux, which is what lets a nest
# carry its parent region's flux anomaly (see refine).
# Scalar turbulons inherit the signed multiplier noise, so they are
# skewed too; the extremal generator is heavy-tailed on the low-multiplier
# side, giving the h'/qt' interior a convective right-skew.
#
# Realized intermittency calibrates as  C1 = A * FLUX_SCALE**alpha  (fit in
# turbulon-analysis/calibration); at alpha=2 the generator is Gaussian and the
# whole scheme reduces to the lognormal cascade.
# FLUX_SCALE is set so the realized flux C1 = 0.1, the measured TWPICE
# horizontal-wind intermittency; C1 = 1.674 c^1.8 from calibration
# (turbulon-analysis/calibration), so c = (0.1/1.674)^(1/1.8) = 0.21.
# NOTE: that calibration was fit under the since-removed n_flux_substeps=4
# configuration (flux advanced four times per octave); it must be re-fit for
# the single-advance-per-class scheme.
FLUX_SCALE = 0.21
FLUX_ALPHA = 1.8

# Taper buffer in units of the summed mean-absolute amplitudes of the
# remaining cascade (current class and all smaller classes).
BOUND_BUFFER_MULTIPLE = 3

# Interpolation compensation for the resolution dependence of the mean
# absolute turbulon amplitude (2026-07-28 convention, Thomas; replaces
# the hops-remaining ZOOM_RETENTION bookkeeping, whose factors were
# defined against the run's own regrid chain and so could not survive a
# change of output resolution).
#
# At s = 1 the output-grid samples of a poorly resolved class coincide
# with the turbulon centers, so coarse sampling systematically hits the
# envelope peaks: a class at k/dx_out = 2 reads ~2.6x the mean absolute
# amplitude of the same deposit sampled finely. The inflation is a
# property of the OUTPUT sampling alone: after the first trilinear
# regrid the deposit is the chord polygon through its samples -- a
# fixed point of later regrids (paper/concept-figs/s2) -- and
# successively finer sampling of that fixed function converges to its
# continuum mean absolute value. The factor f(k/dx_out) is defined
# relative to that chain limit (f -> 1 for well-resolved classes); each
# class's amplitude is MULTIPLIED by f so every class delivers the same
# amplitude convention on the output grid. The one-time chord-polygon
# loss, common to all classes, is absorbed into the lambda = 1/R
# calibration (HAAR_TO_MHAT below).
#
# Measured on the production cascade path (real noise, normalization,
# anisotropic vertical chain, crop machinery) by single-class A/B runs
# across output resolutions: tests/heavy/interpolation_compensation.py.
# Two effects are bundled, and both are real:
#   (1) sampling inflation at small k/dx -- the big rise 0.36 -> 0.92
#       over 2..32 (the isotropic toy underestimates it: the vertical
#       working grids refine as k_z ~ k^{H_z} per octave and stay
#       under-resolved far longer than the horizontal);
#   (2) a slow chain leak of ~1-3% PER OCTAVE that persists to at least
#       k/dx = 256 (per-hop ratios 0.976, 0.973, 0.974, 0.987 over
#       16..256; the vertical regrids never node-nest, so each hop
#       re-chords the deposit slightly). The old ZOOM_RETENTION tail
#       drift (0.461 -> 0.388) was this leak, misread as a small-grid
#       artifact.
# Because of (2) there is no finite "well-resolved" plateau; the
# reference is CHOSEN at k/dx = 512, the outer class of the production
# square runs (entry extrapolated from the measured 256 -> 512 trend).
# The choice of reference is a single overall constant absorbed into
# HAAR_TO_MHAT; only the shape matters. Values 2..64: 384-km-domain
# probe, 3 seeds (seed spread < 0.3%); 128..256: 96-km-domain deep
# probe ratios (the two probes agree to 4 digits on their shared
# 32->64 hop). Classes deeper than 512 clamp to 1 (slightly
# under-compensated by the ~1%/octave residual leak -- no production
# config goes deeper).
#
# NOTE (nested refinement, OPEN): f depends on the run's own output
# resolution. A nest's inherited content was deposited by the parent
# with f(k/dx_parent) but is re-read at the nest's finer resolution,
# where the correct factor would be larger: the parent's finest class
# arrives in a nest ~2.2x too weak. A single summed field cannot be
# correct at two output resolutions at once; resolution pending
# (per-class increment storage vs. accepting the seam bias).
INTERPOLATION_COMPENSATION = {2: 0.358, 4: 0.600, 8: 0.800, 16: 0.894,
                              32: 0.924, 64: 0.950, 128: 0.975,
                              256: 0.988, 512: 1.0}


def _interpolation_compensation(k_over_dx):
    """f(k/dx_out): log2-linear interpolation of the measured table.

    Clamps to the table ends (>= the last abscissa means well-resolved,
    f = 1). An empty table disables compensation (used by the
    measurement harness itself).
    """
    if not INTERPOLATION_COMPENSATION:
        return 1.0
    xs = np.array(sorted(INTERPOLATION_COMPENSATION), dtype=np.float64)
    fs = np.array([INTERPOLATION_COMPENSATION[x] for x in xs])
    if k_over_dx >= xs[-1]:
        return float(fs[-1])
    if k_over_dx <= xs[0]:
        return float(fs[0])
    return float(np.interp(np.log2(k_over_dx), np.log2(xs), fs))

# Haar -> turbulon-amplitude calibration for C_{Phi,L}, defined operationally:
# lambda = 1/R, where R is the mean absolute vertical Haar fluctuation (at
# scale k_z) of a field of unit-amplitude outer-class turbulons synthesized
# exactly as the cascade delivers them under the INTERPOLATION_COMPENSATION
# convention (coarse s=1 deposit carried down the regrid chain, reference
# k/dx = 512). Setting lambda = 1/R makes the field's Haar fluctuation at
# the outer scale equal the mean profile's by construction -- the crossover
# property checked by tests/heavy/normalization_diagnostic.py.
#
# History: R = 2.005 (May calibration) -> 1.9842 (2026-07-28 kernel A/B for
# the Gaussian-weighted admissibility correction: R_new/R_old = 0.98963 +/-
# 0.00001 over 8 seeds). 2026-07-28 (later): the INTERPOLATION_COMPENSATION
# convention re-anchors delivery at the k/dx = 512 reference; at the
# calibration config's outer class both the old and new conventions sit in
# their clamp regimes, so the delivered amplitude changes by EXACTLY the
# removed boost 1/0.388 and R rescales analytically:
#     R = 1.9842 * 0.388 = 0.7699.
# (R < 1 now simply reflects the weaker delivery convention; the crossover
# is what is calibrated, not R's magnitude.) PENDING: end-to-end crossover
# verification via normalization_diagnostic.py before regeneration.
HAAR_TO_MHAT = 1.0 / 0.7699   # ~= 1.299


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
    # ~100x float32 machine epsilon, matching scaleinvariance's
    # precision-scaled clamp. A float64-scale eps (1e-12) here DISABLES the
    # protection: near the zero crossing of cos(phi - a(phi - phi0)) the
    # float32 absolute error (~6e-8) swamps the true value, occasionally
    # flips its sign into the clamp, and the negative power amplifies the
    # garbage into impossible right-tail draws (audit item 33, 2026-07-27;
    # spurious +64 at alpha=1.5 and +259 at alpha=1.95 in 30M draws with
    # 1e-12; clean at 1e-6). The larger clamp also caps the legitimate
    # far-negative tail at smaller magnitude, which is invisible through
    # exp(gamma) -- both map to multiplier -1.
    eps = np.float32(1e-6)

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
# multiplier to unit mean exactly (the alpha-generalization of the lognormal
# -sigma^2/2; it equals 1 at alpha=2). Closed form (derivation in the
# supplement, S1 cascade loop): relative to a standard maximally skewed
# stable of unit scale -- for which ln<exp(theta X)> = -theta^a sec(pi a/2),
# finite because beta=-1 puts the heavy tail on the side the exponential
# kills -- the generator above differs by the scale factor
# |a-1|^(-1/a) (1+tan^2(pi a/2))^(-1/(2a)), and since (1+tan^2)^(1/2) = |sec|
# the secants cancel:  ln<exp(gamma_0)> = 1/(alpha-1)  exactly.
# (The 8M-draw Monte Carlo this replaces measured 1.2510 vs 1.25; audit
# item 32, applied 2026-07-27.)
LEVY_LOG_MEAN = 1.0 / (FLUX_ALPHA - 1.0)


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
    device='cpu',
    save_class_increments=False,
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
          - each horizontal extent (domain_x = nx*dx, domain_y = ny*dy) is
            either an integer multiple of outer_scale, or smaller than it
            (a narrow strip). For a strip axis, turbulon kernels wider than
            the extent are periodized onto it (fold_kernel_to_field), the
            physically correct limit: an eddy much larger than the strip is
            near-uniform across it. Statistics along a strip axis are not
            meaningful at separations approaching the strip width.
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
        Physical bounds (taper + bounded amplitude-preserving add) on moist
        static energy [J/kg].
    qt_min, qt_max : float
        Physical bounds (taper + bounded amplitude-preserving add) on total
        water mixing ratio [kg/kg].
    min_distance_to_ground : int
        Turbulon centers are not placed within min_distance_to_ground × k_z
        of the ground. Must be a non-negative integer.
    compress : bool or None
        If True, 3D data variables in the output NetCDF are written with
        zlib compression at complevel=4. None (default) uses the module-level
        ``steam.constants.output_compress`` setting.
    device : {'cpu', 'cuda'}
        Where the per-class convolutions run. 'cuda' keeps the finest-grid FFT
        working set off the host — the host holds only its persistent float32
        fields. A 'cuda' request with no usable GPU is an error, never a silent
        CPU fall back.
    save_class_increments : bool
        If True, store each class's actually-added h and qt increments, on
        that class's own working grid, in a ``class_increments`` group of
        the output file (~1.14x one output-grid field per scalar; the
        working grids form a geometric pyramid). Opt in for runs intended
        as refinement parents: refine() uses the stored increments to
        re-scale inherited content from the parent's
        INTERPOLATION_COMPENSATION reference to the nest's own (Thomas's
        ruling, 2026-07-28) — without them a nest inherits the parent's
        finest classes ~2.2x too weak in the seam octaves.

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
        if n_tiles < 1:
            # Narrow strip: kernels wider than the extent are periodized onto
            # it by fold_kernel_to_field. No divisibility requirement.
            continue
        if abs(n_tiles - round(n_tiles)) > 1e-9:
            raise ValueError(
                f"domain_{axis} ({domain_size}) must be an integer multiple of "
                f"outer_scale ({outer_scale}) or smaller than it, "
                f"got ratio={n_tiles}"
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

    # Domain-height gate (strengthened 2026-07-28, Thomas): the implied
    # vertical outer scale k_z,L(z) = l_s(z) (L/l_s(z))^{H_z} must fit
    # inside the domain at EVERY level (max over the profile, not the
    # mean), or the outer-scale normalization window has no room for the
    # crossover. All production configs pass comfortably: square 7.6 km,
    # channel <= 4.5 km (both against 20 km), diagnostic 21.9 km / 40 km.
    k_z_L_max = float(np.max(_k_z(anisotropy, outer_scale, spheroscale_profile)))
    n_large_turbulons = int(domain_height / k_z_L_mean)
    if k_z_L_max >= domain_height:
        raise ValueError(
            f"domain_height ({domain_height} m) is shorter than the vertical scale of the "
            f"outer-scale turbulons (max k_z,L = {k_z_L_max:.1f} m over the "
            f"spheroscale profile); increase domain_height or decrease outer_scale"
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

    # Local vertical outer scale k_z,L(z) from the local spheroscale, so the
    # normalization response (a vertical-gradient measure of the mean profile
    # at scale k_z,L) dies where the profile is flat over the local k_z,L.
    k_z_L_on_finest = _k_z(anisotropy, outer_scale, spheroscale_on_finest)

    # Outer-scale turbulon envelope; its x=y=0 center column is the 1D response
    # used to normalize against the mean profile (Eqn apxeq:norm factor computation).
    unit_turbulon = _turbulon_envelope(1, 1/(2*s_x), 1/(2*s_y), 1/(2*s_z),
                                     support_factor=SUPPORT_FACTOR, shape=turbulon_shape)

    # Pre-flight host-memory estimate so a doomed config fails in milliseconds.
    # The cascade's concurrent peak is EIGHT field-sized float32 arrays at the
    # finest grid, not just the three persistent ones (h, qt, flux): the
    # gradient-weighting peak adds S_k, the running-sum buffer, the gradient
    # accumulator and a np.roll/np.gradient temporary; the flux-advance peak
    # equivalently adds gamma, noise, S_k and the convolution result. On CPU
    # the FFT spectral working set (precisely guarded inside
    # convolve_fft_xy_oa_z) rides on top; on CUDA it lives in VRAM and only
    # the eight host arrays remain. (2026-07-15: a 5-field estimate passed
    # preflight at (2048, 2048, 363) and the kernel OOM-killed the session.)
    nx_f, ny_f, nz_f = int(grids['nx'][-1]), int(grids['ny'][-1]), int(grids['nz'][-1])
    field_bytes = nx_f * ny_f * nz_f * 4
    peak_bytes = 8 * field_bytes
    if device != 'cuda':
        peak_bytes += fft_convolution_bytes(nx_f, ny_f, nz_f, unit_turbulon.shape[2], 4)
    available = available_memory_bytes()
    if available is not None and peak_bytes > available - MEMORY_HEADROOM_BYTES:
        raise MemoryError(
            f"STEAM needs ~{peak_bytes / 1024**3:.1f} GiB (approximate) for the "
            f"finest grid {(nx_f, ny_f, nz_f)} but only "
            f"{available / 1024**3:.1f} GiB is available; refusing to risk OOM. "
            f"Reduce resolution, domain size, or vertical levels."
        )

    C_h_k = _compute_normalization(
        h_on_finest, k_z_L_on_finest,
        k_values, outer_scale, grids,
        n_scale_classes_per_dyad=n_scale_classes_per_dyad,
    )
    C_qt_k = _compute_normalization(
        qt_on_finest, k_z_L_on_finest,
        k_values, outer_scale, grids,
        n_scale_classes_per_dyad=n_scale_classes_per_dyad,
    )
    # Scalar C_L for NetCDF attribute: mean of outer-scale C profile
    C_h_L = float(np.mean(C_h_k[0]))
    C_qt_L = float(np.mean(C_qt_k[0]))

    # Turbulon aspect ratio l_z/l_x per class and level (Eqn eq:turbulon
    # aspect ratio scaling): the deterministic factor that suppresses
    # vertical-gradient advection for pancake-shaped large turbulons and
    # approaches 1 at the spheroscale.
    aspect_k = []
    for i, k in enumerate(k_values):
        ls_i = np.interp(grids['z_arrays'][i], z_profile, spheroscale_profile)
        aspect_k.append((_k_z(anisotropy, k, ls_i) / k).astype(np.float32))

    # Per-class taper buffer widths b_{Phi,i}(z) (see _bound_buffers).
    b_h_k = _bound_buffers(C_h_k, grids['z_arrays'])
    b_qt_k = _bound_buffers(C_qt_k, grids['z_arrays'])

    child_seeds = seed_sequence.spawn(n_classes)

    increment_dir = None
    if save_class_increments:
        import tempfile
        increment_dir = Path(tempfile.mkdtemp(prefix="steam_class_inc_"))

    h_pert, qt_pert, flux_field, final_grid = cascade_loop(
        h_profile, qt_profile, z_profile,
        grids,
        C_h_k, C_qt_k,
        b_h_k, b_qt_k, aspect_k,
        h_min, h_max, qt_min, qt_max,
        min_distance_to_ground,
        sparsity_factors,
        child_seeds,
        n_scale_classes_per_dyad=n_scale_classes_per_dyad,
        turbulon_shape=turbulon_shape,
        device=device,
        increment_dir=increment_dir,
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

    # The per-class bounded amplitude-preserving add guarantees the bounds
    # up to float32 rounding of the (perturbation + mean) reassembly, which
    # can undershoot by ~1 ulp of the field value (observed -1e-9 on qt).
    # Clamp that rounding residue only — this moves values by at most one
    # ulp and is not a physics clip.
    np.clip(h_3d, np.float32(h_min), np.float32(h_max), out=h_3d)
    np.clip(qt_3d, np.float32(qt_min), np.float32(qt_max), out=qt_3d)

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
    if increment_dir is not None:
        from .output import write_class_increments
        import shutil
        write_class_increments(output_path, increment_dir, grids,
                               compress=compress)
        shutil.rmtree(increment_dir, ignore_errors=True)
    return output_path


def cascade_loop(
    h_profile, qt_profile, z_profile,
    grids,
    C_h_k, C_qt_k,
    b_h_k, b_qt_k, aspect_k,
    h_min, h_max, qt_min, qt_max,
    min_distance_to_ground,
    sparsity_factors,
    seeds_or_rng,
    n_scale_classes_per_dyad=1,
    h_perturbation=None,
    qt_perturbation=None,
    flux=None,
    inner_windows=None,
    turbulon_shape='mexican_hat',
    zero_bottom=True,
    zero_top=True,
    device='cpu',
    increment_dir=None,
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
    b_h_k, b_qt_k : list of 1D ndarray
        Per-class taper buffer widths b_{Phi,i}(z), parallel to
        C_h_k/C_qt_k (see _bound_buffers).
    h_min, h_max : float
        Physical bounds (taper + bounded amplitude-preserving add) for moist
        static energy.
    qt_min, qt_max : float
        Physical bounds (taper + bounded amplitude-preserving add) for total
        water mixing ratio.
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
    n_scale_classes_per_dyad : int
        Cascade density: number of size classes per octave. Must match the
        multiplicative spacing of ``grids['k']`` (adjacent-class ratio
        ``2 ** (1 / n_scale_classes_per_dyad)``). Sets the per-class flux
        generator scale (see the FLUX CASCADE note).
    h_perturbation, qt_perturbation : ndarray or None
        Existing perturbation fields for nested simulations. If None,
        initialized to zero at the first scale class resolution.
    flux : ndarray or None
        Existing dimensionless flux field for nested simulations, on the same
        grid as h_perturbation. If None, initialized to one.
    inner_windows : list of (x_start, x_stop, y_start, y_stop) or None
        Per-class index window of the nest's own horizontal extent within the
        padded working grid. None (root) means the whole grid is the domain.
        Every realized-mean statistic — the flux volume mean and the mean
        absolute multiplier noise inside _advance_flux, the advective weight
        norm, and the bounded amplitude-preserving add — is taken over this window,
        so the halo carries context into the domain without contaminating the
        domain's statistics. See _inner_view.
    zero_bottom, zero_top : bool
        Whether to zero the bottom/top n_zero cells of the sparse noise
        at each class. True when the z-boundary corresponds to ground or
        the top of a full simulation domain; False for elevated insets,
        where turbulon centers may legitimately exist below/above the
        inset's inner z range (in the z-pad) and contribute via their
        kernel tails.
    device : {'cpu', 'cuda'}
        Passed to the per-class convolutions; 'cuda' runs the FFTs on the GPU.

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
    if not isinstance(n_scale_classes_per_dyad, int) or n_scale_classes_per_dyad < 1:
        raise ValueError(
            "n_scale_classes_per_dyad must be a positive integer, "
            f"got {n_scale_classes_per_dyad}"
        )

    if n_classes > 1:
        expected_ratio = 2.0 ** (1.0 / n_scale_classes_per_dyad)
        class_ratios = np.asarray(grids['k'][:-1]) / np.asarray(grids['k'][1:])
        if not np.allclose(class_ratios, expected_ratio):
            raise ValueError(
                "grids['k'] spacing is inconsistent with "
                f"n_scale_classes_per_dyad={n_scale_classes_per_dyad} "
                f"(expected adjacent-class ratio {expected_ratio})"
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
        window = None if inner_windows is None else tuple(inner_windows[i])

        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})           interpolating...', end='\r')

        # Interpolate perturbations from previous resolution
        if h_perturbation is None:
            h_perturbation = np.zeros((nx_k, ny_k, nz_k), dtype=np.float32)
            qt_perturbation = np.zeros((nx_k, ny_k, nz_k), dtype=np.float32)
            flux = np.ones((nx_k, ny_k, nz_k), dtype=np.float32)
        else:
            if flux is None:
                flux = np.ones(h_perturbation.shape, dtype=np.float32)
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
            flux, rng, kernel, FLUX_SCALE, n_scale_classes_per_dyad,
            sparsity_factors, n_zero, zero_bottom, zero_top,
            device=device, window=window,
        )

        print(f'Step {i+1:3d}/{n_classes:3d}: k={k:6.0f}m, grid=({nx_k:4d},{ny_k:4d},{nz_k:4d})           computing G, convolutions...', end='\r')

        # Process h and qt sequentially to halve peak memory
        for (name, perturbation_field, mean_1d, C_k_i, b_i, phi_min, phi_max) in (
            ('h',  h_perturbation,  h_mean_1d,  C_h_k[i],  b_h_k[i],  h_min,  h_max),
            ('qt', qt_perturbation, qt_mean_1d, C_qt_k[i], b_qt_k[i], qt_min, qt_max),
        ):
            running_sum = perturbation_field + mean_1d[np.newaxis, np.newaxis, :]

            # Advective weight (Apxeq:advective weight):
            #   W = |∇_h φ| + (ℓ_z/ℓ_x) |∂φ/∂z|,
            # φ the field from classes L > ℓ, ℓ_z/ℓ_x the deterministic
            # turbulon aspect ratio (Δw/Δu = ℓ_z/ℓ_x is kinematics).
            # The flux enters the amplitude ONCE, through S_k (which
            # carries the local larger-scale flux); multiplying W by the
            # flux as well would double-count it (removed 2026-07-22).
            grad_h, grad_z = _gradient_components(running_sum, dx_k, dy_k, z_k)
            W = grad_z
            W *= aspect_k[i]    # 1D broadcast: ℓ_z/ℓ_x
            W += grad_h
            del grad_h

            # Joint product norm (2026-07-27 ruling): W and S_k are
            # CORRELATED — the flux is large where past deposits (and so
            # the running field's gradients) are large, and the correlation
            # compounds down-cascade. Normalizing the factors separately
            # therefore does not normalize the product: <W_hat |S_k|> grew
            # ~9x over 7 octaves and cancelled the k^H_h ladder (audit
            # item 35). Normalize the PRODUCT W*S_k per level by its mean
            # absolute value over turbulon centers, which enforces the
            # mean turbulon amplitude <|A|> = C_k exactly at every level
            # and class. Structureless levels (no centers) keep W = 0.
            #
            # The bound taper participates in the normalized pattern
            # (2026-07-28 ruling, reversing 2026-07-27): proximity to a
            # bound REDISTRIBUTES the level's amplitude toward
            # far-from-bound cells rather than suppressing the level
            # total. The earlier suppress-without-renorm order made the
            # effective ladder C_k*<g_k> with <g_k> strongly
            # scale-dependent (buffers shrink down-cascade), which
            # flattened the production Haar slopes (h ladder bent, qt
            # ladder inverted). With the taper inside the norm the
            # delivered per-level amplitude stays C_k at every class.
            W *= S_k            # joint pattern W*S_k (sparse at centers)
            W *= _bound_taper(running_sum, b_i, phi_min, phi_max)
            del running_sum
            inner = _inner_view(W, window)
            level_sum = np.abs(inner).sum(axis=(0, 1), dtype=np.float64)
            level_cnt = np.count_nonzero(inner, axis=(0, 1))
            level_mean = (level_sum / np.maximum(level_cnt, 1)).astype(np.float32)
            W /= np.where(level_mean > 0, level_mean, np.float32(1.0))[None, None, :]

            W *= C_k_i          # 1D broadcast: mean amplitude C_k(z)

            # Interpolation compensation f(k/dx_out): damp poorly
            # resolved classes so every class delivers the same mean
            # absolute amplitude convention on the output grid (see
            # INTERPOLATION_COMPENSATION). The output spacing is half
            # the finest class by construction, for a root and a nest
            # alike.
            W *= np.float32(_interpolation_compensation(
                2.0 * k / float(grids['k'][n_classes - 1])))

            # Convolve (periodic x,y; zero-padded z), then add through the
            # amplitude-preserving bounded projection (2026-07-28 ruling):
            # each class's added increment is (1) zero-mean per level,
            # (2) amplitude-preserving through the bounding, and (3)
            # bound-respecting pointwise — see _bounded_amplitude_add.
            # This replaces the per-class mean-preserving projection,
            # whose repeated amplitude-capping of coarse-scale excursions
            # (9 compounding clips) was the dominant Haar-slope flattener
            # at production amplitudes.
            increment = CONVOLVE(W, kernel, device=device)
            del W
            if increment_dir is not None:
                before = perturbation_field.copy()
            _bounded_amplitude_add(perturbation_field, mean_1d, increment,
                                   phi_min, phi_max, window=window)
            del increment
            if increment_dir is not None:
                # The ACTUALLY-ADDED increment (post bounded add), on this
                # class's own working grid — the per-class record that lets
                # a nest re-scale inherited content to its own
                # INTERPOLATION_COMPENSATION reference (see refine).
                np.subtract(perturbation_field, before, out=before)
                np.save(Path(increment_dir) / f"c{i:02d}_{name}.npy", before)
                del before

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


def _inner_view(field, window):
    """The simulated domain's own horizontal extent, halo excluded.

    ``window`` is (x_start, x_stop, y_start, y_stop) into the first two axes,
    or None for a root simulation whose domain is the whole array. Works on
    both 3D fields and single 2D levels. Returned as a view, so the caller
    reduces over the domain while still writing back to the full array.
    """
    if window is None:
        return field
    x_start, x_stop, y_start, y_stop = window
    return field[x_start:x_stop, y_start:y_stop]


def _advance_flux(
    flux, rng, kernel, flux_noise_scale, n_scale_classes_per_dyad,
    sparsity_factors, n_zero=0, zero_bottom=False, zero_top=False,
    device='cpu', window=None,
):
    """Advance the dimensionless flux cascade by one size class.

    A single advance multiplies the flux by exp(gamma) with gamma an
    extremal-Levy generator field (see the FLUX CASCADE note), scaled and
    shifted to unit mean, giving the bounded-below update
    F += conv(psi, (exp(gamma)-1)*F). With n = n_scale_classes_per_dyad,
    adjacent classes are separated by scale ratio 2**(1/n), so by
    alpha-stability the per-class generator scale is
    flux_noise_scale / n**(1/alpha) — the per-octave generator is invariant
    in distribution under the class density. The scalar turbulon amplitude
    carries the SIGNED multiplier noise and the entering flux, normalized to
    mean absolute value one:
    S_k = (exp(gamma)-1) * F_{k-1} / <|exp(gamma)-1|>.

    The clip at zero biases the flux downward, so after the update the field is
    rescaled by ONE scalar that restores the VOLUME mean it entered the class
    with (ruling 2026-07-27, audit B14; replaces the per-level unit-mean
    renorm). The renormalization is thereby a pure corrector for the bias of
    the step just taken, and nothing else: it does not impose a value on the
    flux, and it does not touch the realized horizontal-mean structure within
    the volume. A root enters at mean one and stays there. A nest enters at the
    flux anomaly it inherited from its parent over this region, and keeps it --
    which is the whole point of continuing a parent's cascade rather than
    starting a new one. Realized per-level mean fluctuations are left alive:
    they are physical layer-scale intermittency; the flat-dissipation
    idealization is enforced in expectation by the generator shift, not
    realization-by-realization.

    ``window`` (see _inner_view) is the simulated domain, excluding a nest's
    halo. Both the volume means above and the mean absolute multiplier noise
    are taken over it; None means the whole array, as for a root.
    """
    s_x, s_y, s_z = sparsity_factors
    # Captured before the increment: this is what the renormalization restores.
    # float64 accumulator — the finest cascade grid runs to ~1e9 cells.
    entering_mean = float(_inner_view(flux, window).mean(dtype=np.float64))
    per_class_scale = flux_noise_scale / n_scale_classes_per_dyad ** (1.0 / FLUX_ALPHA)
    shift = np.float32(LEVY_LOG_MEAN * per_class_scale ** FLUX_ALPHA)
    per_class_scale = np.float32(per_class_scale)

    gamma = _sparse_levy(*flux.shape, s_x, s_y, s_z, FLUX_ALPHA, rng)
    if zero_bottom and n_zero > 0:
        gamma[:, :, :n_zero] = 0
    if zero_top and n_zero > 0:
        gamma[:, :, -n_zero:] = 0

    # Unit-mean multiplier noise exp(gamma)-1 at the turbulon centers, and
    # exactly 0 off-centers (no turbulon there); bounded below by -1.
    gamma *= per_class_scale
    noise = np.expm1(gamma - shift)
    noise[gamma == 0.0] = np.float32(0.0)

    noise_inner = _inner_view(noise, window)
    mean_abs = np.abs(noise_inner).sum() / max(np.count_nonzero(noise_inner), 1)
    scalar_amplitude = noise * flux
    if mean_abs > 0:
        scalar_amplitude *= np.float32(1.0 / mean_abs)

    noise *= flux
    flux += CONVOLVE(noise, kernel, device=device)

    n_clipped = int(np.count_nonzero(flux < 0))
    np.maximum(flux, np.float32(0.0), out=flux)
    realized_mean = float(_inner_view(flux, window).mean(dtype=np.float64))
    if realized_mean > 0:
        flux *= np.float32(entering_mean / realized_mean)
    else:
        # The whole domain clipped to zero: there is no structure left to
        # rescale, so restore the entering mean as a flat field.
        flux[...] = np.float32(entering_mean)

    diagnostics = {
        'n_clipped': n_clipped,
        'n_points': int(flux.size),
        'clip_fraction': n_clipped / flux.size,
        'entering_mean': entering_mean,
        'realized_mean': realized_mean,
    }
    return scalar_amplitude, diagnostics


def simulate_flux_only(
    grids,
    sparsity_factors=(1, 1, 1),
    seed=None,
    flux_noise_scale=None,
    n_scale_classes_per_dyad=1,
    min_distance_to_ground=0,
    turbulon_shape='mexican_hat',
    zero_bottom=False,
    zero_top=False,
):
    """Run only the unit-mean flux cascade on root-simulation grids.

    This is a cheap calibration/diagnostic path: it performs one convolution
    per size class and skips the scalar profiles, gradients, and scalar
    convolutions. ``grids`` must come from :func:`_compute_all_grids` with a
    common physical extent at every class (the root-simulation case), and its
    ``k`` spacing must match ``n_scale_classes_per_dyad`` (adjacent-class
    ratio ``2 ** (1 / n_scale_classes_per_dyad)``).

    Returns
    -------
    flux : float32 ndarray
        Final dimensionless flux field. Each class restores the volume mean
        the flux entered it with, and the cascade starts from one, so the
        final field has volume mean one.
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
    if not isinstance(n_scale_classes_per_dyad, int) or n_scale_classes_per_dyad < 1:
        raise ValueError(
            "n_scale_classes_per_dyad must be a positive integer, "
            f"got {n_scale_classes_per_dyad}"
        )

    n_classes = len(grids['k'])
    if n_classes > 1:
        expected_ratio = 2.0 ** (1.0 / n_scale_classes_per_dyad)
        class_ratios = np.asarray(grids['k'][:-1]) / np.asarray(grids['k'][1:])
        if not np.allclose(class_ratios, expected_ratio):
            raise ValueError(
                "simulate_flux_only: grids['k'] spacing is inconsistent with "
                f"n_scale_classes_per_dyad={n_scale_classes_per_dyad} "
                f"(expected adjacent-class ratio {expected_ratio})"
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
            flux, rng, kernel, c, n_scale_classes_per_dyad, sparsity_factors,
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
        'n_scale_classes_per_dyad': n_scale_classes_per_dyad,
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
        # max(1, ...): a class whose turbulons are wider than a strip axis
        # still gets one cell there (the periodized kernel spreads it).
        nx_arr[i] = max(1, int(round(padded_extent_x / target_dx_k)))
        ny_arr[i] = max(1, int(round(padded_extent_y / target_dy_k)))
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

    # Enforce discrete admissibility (zero sum) by subtracting a multiple of
    # the envelope's own Gaussian factor rather than a uniform pedestal.
    #
    # The shapes are zero-mean over CONTINUUM 3D space by construction (the
    # (3 - rho) leading constant is the dimension count), but the grid does
    # not resolve that cancellation at the default sparsity: the negative
    # shell is carried by many low-amplitude cells whose count grows as r^2,
    # and at dx = k/2 the discrete sum is +0.131 against a peak tap of 3
    # (1.7% of sum|T|). It falls to 2.7e-12 at dx = k/4 -- the error is a
    # sampling artifact that vanishes super-exponentially as the grid
    # refines, not a defect of the shape.
    #
    # Subtracting the flat mean removes the same DC but spreads the
    # correction uniformly across the whole +/-support_factor*k box, leaving
    # a small step at the truncation radius where the envelope itself is
    # already zero. Subtracting a Gaussian-weighted mean instead keeps the
    # correction localized with the envelope. For mexican_hat this is
    # exactly equivalent to replacing the leading 3 by
    #     A_opt = sum(rho * w) / sum(w),  w = exp(-rho/2)
    # i.e. 2.96781718 at s = 1 (eps = 0.032, 1.07% of 3), and A_opt -> 3 to
    # 12 digits once s = 2 resolves the shell. A_opt depends only on the
    # sparsity factors, and is independent of support_factor (to 10 digits)
    # and of the turbulon aspect ratio -- this kernel is always the
    # isotropic one, anisotropy entering through the grid's physical dz.
    gaussian = np.exp(-r_norm_sq / (2.0 * (k / np.pi) ** 2)).astype(np.float32)
    envelope -= gaussian * (envelope.sum() / gaussian.sum())
    return envelope


def _compute_normalization(profile_on_finest_grid, k_z_L_on_finest,
                           k_values, outer_scale, z_arrays,
                           n_scale_classes_per_dyad=1):
    """Scale- and height-dependent amplitude arrays C_{Phi,k}.

    (Apxeq:norm factor computation)

        C_{Phi,L}(z) = lambda * |Haar_{k_z,L}(<Phi>_t)|(z)

    The Haar fluctuation at scale k_z,L(z) — the mean of the profile over the
    upper half of a k_z,L-wide window minus the mean over the lower half — is
    the characteristic anomaly produced by an overturning eddy of depth k_z,L:
    parcels arriving at z from z +- k_z,L/2 carry the profile difference
    across the eddy. An odd (first-difference) measure is required: an even
    zero-mean kernel (e.g. the envelope column) annihilates constant
    gradients, yet a constant vertical gradient is precisely the case where
    advection must produce variability. The window uses the LOCAL vertical
    outer scale k_z,L(z) from the local spheroscale; half-window MEANS make
    the response independent of profile resolution with no extra bookkeeping.

    HAAR_TO_MHAT converts the measured Haar coefficient to the cascade's
    turbulon amplitude convention, defined operationally: lambda = 1/R with
    R the mean absolute vertical Haar fluctuation of a unit-amplitude
    outer-class turbulon field (R = 1.984 from the envelope shape and s=1
    packing), so the field's Haar fluctuation at the outer scale equals the
    profile's by construction (see the module constant and
    tests/heavy/normalization_diagnostic.py).

    C_{Phi,k} then follows n_c^{-1/alpha} (k/L)^H_h, with n_c =
    n_scale_classes_per_dyad: the same density compensation as the flux
    cascade, ASSUMED to carry over to the scalars because S_k is built from
    the same noise (Var(e^gamma - 1) scales as s^alpha, not s^2, so summed
    per-octave variance is invariant under s ~ n_c^{-1/alpha}). Not verified
    for all scalar statistics; unity at the production n_c = 1.

    Parameters
    ----------
    profile_on_finest_grid : ndarray, shape (nz_finest,)
        Mean profile <Phi>_t interpolated to the finest-resolution z-grid.
    k_z_L_on_finest : ndarray, shape (nz_finest,)
        Local vertical outer scale k_z(outer_scale, spheroscale(z)) [m].
    k_values : ndarray, shape (n_classes,)
    outer_scale : float
        Outer scale L of the ROOT cascade. A nest continues the same
        (k/L)^H_h ladder below its parent's finest class, so it passes the
        root's L here, not its own coarsest class.
    z_arrays : dict with 'z_arrays' — list of per-class z-coordinate arrays.

    Returns
    -------
    list of n_classes 1D float32 arrays.
    """
    z_finest = z_arrays['z_arrays'][-1]
    dz_finest = float(np.mean(np.diff(z_finest)))
    n_p = profile_on_finest_grid.size

    # Per-level Haar half-window k_z,L(z)/2 in finest cells (>= 1 cell).
    n_half = np.maximum(1, np.round(
        0.5 * np.asarray(k_z_L_on_finest) / dz_finest).astype(int))
    n_max = int(n_half.max())
    padded = np.pad(profile_on_finest_grid.astype(np.float64),
                    (n_max, n_max), mode='edge')

    response = np.empty(n_p)
    for i in range(n_p):
        m = int(n_half[i])
        center = i + n_max
        upper = padded[center + 1: center + m + 1].mean()
        lower = padded[center - m: center].mean()
        response[i] = HAAR_TO_MHAT * abs(upper - lower)
    C_k = []
    for i, k in enumerate(k_values):
        hurst_scale = float((k / outer_scale) ** H_h) * n_scale_classes_per_dyad ** (-1.0 / FLUX_ALPHA)
        C_profile = np.interp(z_arrays['z_arrays'][i], z_finest, response).astype(np.float32) * hurst_scale
        C_k.append(C_profile)
    return C_k


def _gradient_components(field_3d, dx, dy, z_coords):
    """Horizontal gradient magnitude |∇_h f| and vertical |∂f/∂z|.

    Periodic central differences in x,y; np.gradient in z (z_coords may be a
    scalar spacing or a 1D array of z-positions). Returns (grad_h, grad_z),
    both non-negative. In-place accumulation keeps the peak at ~3 arrays.
    """
    dx_f32 = np.float32(2 * dx)
    dy_f32 = np.float32(2 * dy)

    # X: periodic central difference → grad_h holds grad_x**2
    grad_h = np.roll(field_3d, -1, axis=0)
    grad_h -= np.roll(field_3d, 1, axis=0)
    grad_h /= dx_f32
    grad_h **= 2

    # Y: accumulate grad_y**2
    grad = np.roll(field_3d, -1, axis=1)
    grad -= np.roll(field_3d, 1, axis=1)
    grad /= dy_f32
    grad **= 2
    grad_h += grad
    del grad
    np.sqrt(grad_h, out=grad_h)

    grad_z = np.gradient(field_3d, z_coords.astype(np.float32) if hasattr(z_coords, 'astype') else z_coords, axis=2)
    np.abs(grad_z, out=grad_z)
    return grad_h, grad_z


def _bound_buffers(C_k, z_arrays):
    """Per-class taper buffer widths b_{Phi,i}(z), one array per size class.

        b_{Phi,i}(z) = BOUND_BUFFER_MULTIPLE * sum_{j >= i} C_{Phi,j}(z)

    — the current class plus all SMALLER classes, a LINEAR sum of
    mean-absolute amplitudes (deliberately not quadrature; first-moment
    discipline). Each C_{Phi,j} lives on its own per-class z-grid and is
    interpolated onto class i's grid before summing.

    Returns a list of n_classes float32 arrays parallel to C_k.
    """
    n_classes = len(C_k)
    buffers = []
    for i in range(n_classes):
        z_i = z_arrays[i]
        total = np.zeros(len(z_i), dtype=np.float64)
        for j in range(i, n_classes):
            total += np.interp(z_i, z_arrays[j], C_k[j])
        buffers.append((BOUND_BUFFER_MULTIPLE * total).astype(np.float32))
    return buffers


def _bound_taper(running_sum, b, phi_min, phi_max):
    """Taper factor g = clip(min((φ - φ_min)/b, (φ_max - φ)/b), 0, 1).

    φ is the running field (running_sum), b the 1D buffer-width profile
    broadcast over (x, y). Computed in float32, in place: running_sum is
    consumed as the output buffer. Where b <= 0 the tiny surrogate divisor
    gives g = 1 strictly inside the bounds and g = 0 at or outside them.
    """
    g = running_sum
    g -= np.float32(phi_min)
    np.minimum(g, np.float32(phi_max - phi_min) - g, out=g)
    g /= np.where(b > 0, b, np.float32(1e-30))
    np.clip(g, np.float32(0.0), np.float32(1.0), out=g)
    return g


def _bounded_amplitude_add(perturbation_field, mean_1d, increment,
                           phi_min, phi_max, window=None, n_rescale=10):
    """Add a class's increment under the three deposit conditions.

    For each z-level, the added perturbation delta satisfies (ruling
    2026-07-28):

      1. zero horizontal mean (over the domain ``window``),
      2. mean absolute amplitude preserved through the bounding —
         <|delta|> equals the raw increment's own pre-clip amplitude A0,
         so the amplitude ladder is set by the normalization upstream
         and the bounding neither double-enforces nor erodes it,
      3. the field stays in [phi_min, phi_max] pointwise.

    All three are achievable iff A0 <= 2*min(<phi> - phi_min,
    phi_max - <phi>): a zero-mean increment's negative mass is capped by
    the level's distance to the lower bound (and vice versa), so the
    up-side headroom cannot be spent without moving the mean. Where A0
    exceeds that ceiling — dry qt levels aloft, where the design ladder
    is infeasible for ANY scheme — the rescale stalls against the clip
    and the level delivers its maximum realizable amplitude, with
    conditions 1 and 3 still exact. Scaling near the bound breaks by
    necessity, not by accident.

    Implementation: candidates are the two-scalar family
    clip(s*d - mu, phi_min - phi, phi_max - phi). A short
    demean -> clip-to-caps -> rescale-to-A0 loop finds s; the exact
    zero mean is then enforced by BISECTION on mu — the level mean of
    clip(d - mu, ...) is continuous, piecewise-linear and monotone
    non-increasing in mu (cf. _project_onto_bounds, the s = 1 member of
    the family). A naive demean-clip iteration is NOT used for the
    final polish: it converges at rate = clip-active fraction, useless
    on levels pinned at a bound (~0.97 aloft), and the residual means
    compound down-cascade (caught 2026-07-28: +54x qt mean at 12 km).

    Modifies perturbation_field in place; consumes ``increment``.
    ``window`` (see _inner_view) is the domain whose statistics are
    enforced; caps are respected everywhere, halo included.
    """
    lo = np.float32(phi_min)
    hi = np.float32(phi_max)
    width = float(phi_max) - float(phi_min)
    for lev in range(perturbation_field.shape[2]):
        phi = perturbation_field[:, :, lev] + mean_1d[lev]
        cap_lo = lo - phi
        cap_hi = hi - phi
        d = increment[:, :, lev]
        dw = _inner_view(d, window)
        a0 = float(np.abs(dw).mean(dtype=np.float64))
        if a0 <= 0.0:
            np.clip(d, cap_lo, cap_hi, out=d)   # halo may still carry mass
            perturbation_field[:, :, lev] += d
            continue
        for _ in range(n_rescale):
            d -= np.float32(dw.mean(dtype=np.float64))
            np.clip(d, cap_lo, cap_hi, out=d)
            m_abs = float(np.abs(dw).mean(dtype=np.float64))
            if m_abs <= 0.0:
                break
            scale = min(a0 / m_abs, 2.0)
            if abs(scale - 1.0) < 1e-4:
                break
            d *= np.float32(scale)
        d -= np.float32(dw.mean(dtype=np.float64))
        np.clip(d, cap_lo, cap_hi, out=d)
        if abs(float(dw.mean(dtype=np.float64))) > 1e-9 * a0:
            mu_lo, mu_hi = -width, width
            trial = np.empty_like(d)
            trial_w = _inner_view(trial, window)
            for _ in range(60):
                mu = 0.5 * (mu_lo + mu_hi)
                np.subtract(d, np.float32(mu), out=trial)
                np.clip(trial, cap_lo, cap_hi, out=trial)
                if float(trial_w.mean(dtype=np.float64)) > 0.0:
                    mu_lo = mu
                else:
                    mu_hi = mu
            np.subtract(d, np.float32(0.5 * (mu_lo + mu_hi)), out=d)
            np.clip(d, cap_lo, cap_hi, out=d)
        perturbation_field[:, :, lev] += d


def _project_onto_bounds(perturbation_field, mean_1d, phi_min, phi_max, window=None):
    """Project onto [phi_min, phi_max] preserving each level's horizontal mean.

    NOTE (2026-07-28): superseded in cascade_loop by
    _bounded_amplitude_add — this is its s = 1 special case (bounds and
    mean, but no amplitude preservation). Retained for diagnostics and
    the archaeology probes.

    For field = perturbation + mean, each level with any out-of-bounds value
    is replaced by clip(field - mu, phi_min, phi_max) with the scalar mu
    chosen so the level's horizontal mean is unchanged — the
    least-squares-closest field satisfying the bounds plus per-level mean
    preservation. The level mean of clip(field - mu, lo, hi) is continuous,
    piecewise-linear, and monotonically non-increasing in mu (each grid
    point's clipped value is non-increasing in mu; strictly decreasing while
    any point is interior), so mu is found by bisection in float64 on the
    safe bracket [-(phi_max - phi_min), +(phi_max - phi_min)]. Levels with
    no violations are untouched. Modifies perturbation_field in place,
    folding the correction into the running perturbation.

    ``window`` (see _inner_view) selects the horizontal extent whose mean is
    preserved. The bounds themselves are still enforced everywhere, halo
    included — an out-of-bounds halo would corrupt the bound taper and the
    gradients that the next size class reads from it.
    """
    lo = float(phi_min)
    hi = float(phi_max)
    width = hi - lo
    for lev in range(perturbation_field.shape[2]):
        field_lev = perturbation_field[:, :, lev].astype(np.float64)
        field_lev += float(mean_1d[lev])
        if field_lev.min() >= lo and field_lev.max() <= hi:
            continue
        field_lev_inner = _inner_view(field_lev, window)
        target = field_lev_inner.mean()
        mu_low, mu_high = -width, width
        for _ in range(60):
            mu = 0.5 * (mu_low + mu_high)
            if np.clip(field_lev_inner - mu, lo, hi).mean() > target:
                mu_low = mu
            else:
                mu_high = mu
            if mu_high - mu_low < 1e-12 * width:
                break
        mu = 0.5 * (mu_low + mu_high)
        np.clip(field_lev - mu, lo, hi, out=field_lev)
        field_lev -= float(mean_1d[lev])
        perturbation_field[:, :, lev] = field_lev


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
    device='cpu',
):
    """Continue a completed simulation's cascade over a subdomain, finer.

    Loads a finished STEAM simulation (or a previous nest), extracts a
    subdomain together with a halo wide enough to hold the tails of turbulons
    centered outside it, and runs the SAME per-class loop for the size classes
    below the parent's finest. The nest inherits the parent's scalar
    perturbations and its flux; nothing is re-seeded from scratch.

    The one thing that changes is scope. A root normalizes the advective
    weight, the flux level means and the bounded amplitude-preserving add over its
    whole domain; a nest has no access to those global means, so it uses its
    own extent instead. That approximation, confined to the nesting domain, is
    the deliberate price of nesting. The halo is context only: it is excluded
    from every such statistic (_inner_view) and discarded on output.

    For recursive refinement, pass the group path of an existing nest as
    parent_group (e.g. "refinements/r0").

    Parameters
    ----------
    parent_path : str or Path
        Path to the NetCDF file. The nest is written back into this file.
    x_start, x_stop : int
        Index range into the parent x-grid for the nest (halo excluded).
    y_start, y_stop : int
        Index range into the parent y-grid for the nest (halo excluded).
    dx, dy : float
        New finest horizontal resolution [m].
    parent_group : str
        NetCDF group to read the parent from. Default '/' (root).
    output_group : str or None
        NetCDF group name for output. Default: auto-generated
        "refinements/r0", "r1", ...
    seed : int or None
        Random seed. If None, derived deterministically from the parent seed
        and the output group name.
    sparsity_factors : tuple of 3 ints or None
        If None, inherit from the parent.
    turbulon_shape : str or None
        If None, inherit from the parent.
    z_min : float or None
        Bottom altitude of the nest [m]. If None, inherit the parent's bottom.
    z_max : float or None
        Top altitude of the nest [m]. If None, use the parent's top.
    anisotropy : str or None
        Grid-anisotropy function name (see _k_z). If None, inherit from the
        parent group. The piecewise-isotropic option genuinely bites here:
        deep nests reach classes with k below the spheroscale.
    compress : bool or None
        If True, 3D data variables in the output group are zlib-compressed at
        complevel=4. None (default) uses ``steam.constants.output_compress``.
    device : {'cpu', 'cuda'}
        Where the per-class convolutions run, as in simulate().

    Returns
    -------
    Path
        The parent_path (with the new group written into it).
    """
    if compress is None:
        compress = constants.output_compress
    parent_path = Path(parent_path)

    ds = netCDF4.Dataset(parent_path, "r")
    grp = ds if parent_group == '/' else ds[parent_group]

    h_3d = grp.variables["h"][:].astype(np.float32)
    qt_3d = grp.variables["qt"][:].astype(np.float32)
    if "flux" not in grp.variables:
        ds.close()
        raise ValueError(
            f"Parent group {parent_group!r} has no flux field; the nest cannot "
            f"continue the flux cascade. Regenerate the parent with the "
            f"current version of simulate()."
        )
    flux_3d = grp.variables["flux"][:].astype(np.float32)
    x_coords = grp.variables["x"][:]
    y_coords = grp.variables["y"][:]
    z_coords = np.asarray(grp.variables["z"][:], dtype=np.float64)
    h_profile = grp.variables["h_profile"][:]
    qt_profile = grp.variables["qt_profile"][:]
    z_profile = np.asarray(grp.variables["z_profile"][:], dtype=np.float64)
    k_values_parent = grp.variables["k_values"][:]
    C_h_k_parent = grp.variables["C_h_k"][:]
    C_qt_k_parent = grp.variables["C_qt_k"][:]
    spheroscale_profile = np.asarray(
        grp.variables["spheroscale_profile"][:], dtype=np.float64)

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
    parent_n_per_dyad = int(grp.n_scale_classes_per_dyad)
    size_class_gap_factor = 2.0 ** (1.0 / parent_n_per_dyad)

    if sparsity_factors is None:
        sparsity_factors = tuple(int(v) for v in grp.sparsity_factors)
    if turbulon_shape is None:
        turbulon_shape = grp.turbulon_shape
    if anisotropy is None:
        anisotropy = grp.anisotropy
    if anisotropy not in VALID_ANISOTROPY:
        raise ValueError(
            f"anisotropy must be one of {VALID_ANISOTROPY}, got {anisotropy!r}"
        )

    parent_z_min = float(grp.domain_z_min) if hasattr(grp, 'domain_z_min') else 0.0
    # Per-axis periodicity of the parent. A root wraps in both; a nest wraps
    # only along an axis it spanned. Files predating the attribute are roots.
    if hasattr(grp, 'periodic_x'):
        parent_periodic_x = bool(grp.periodic_x)
        parent_periodic_y = bool(grp.periodic_y)
    else:
        parent_periodic_x = parent_periodic_y = (parent_group == '/')

    if output_group is None:
        existing = []
        if "refinements" in ds.groups:
            existing = list(ds.groups["refinements"].groups.keys())
        index = 0
        while f"r{index}" in existing:
            index += 1
        output_group = f"refinements/r{index}"

    # For the inherited-content compensation re-scaling below: whether the
    # parent stored per-class increments, and the PARENT's own grid
    # parameters (the nest may override sparsity/anisotropy for itself,
    # but the replay must reconstruct the parent's grids exactly).
    parent_has_increments = "class_increments" in grp.groups
    parent_is_nest = hasattr(grp, "parent_group")
    parent_sparsity = tuple(int(v) for v in grp.sparsity_factors)
    parent_anisotropy = grp.anisotropy if hasattr(grp, "anisotropy") else anisotropy

    ds.close()

    # Vertical extent of the nest.
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
    nest_height = z_max - z_min
    if nest_height <= 0:
        raise ValueError(f"z_max ({z_max}) must be greater than z_min ({z_min})")

    # The nest's classes start one log-step BELOW the parent's finest, so no
    # size class is ever simulated twice.
    new_outer_scale = float(k_values_parent[-1])
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

    parent_nx = h_3d.shape[0]
    parent_ny = h_3d.shape[1]
    inner_nx = x_stop - x_start
    inner_ny = y_stop - y_start
    if inner_nx < 1 or inner_ny < 1:
        raise ValueError(
            f"Empty nest: x=[{x_start},{x_stop}], y=[{y_start},{y_stop}]"
        )
    inner_extent_x = inner_nx * parent_dx
    inner_extent_y = inner_ny * parent_dy

    # Same rule as simulate(): whole tiles, or a strip narrower than one tile
    # (whose over-wide kernels are periodized onto the extent).
    for extent, axis_name in ((inner_extent_x, 'x'), (inner_extent_y, 'y')):
        n_tiles = extent / new_outer_scale
        if n_tiles < 1:
            continue
        if abs(n_tiles - round(n_tiles)) > 1e-9:
            raise ValueError(
                f"Nest {axis_name}-extent ({extent} m) must be an integer "
                f"multiple of new_outer_scale ({new_outer_scale} m) or smaller "
                f"than it, got ratio={n_tiles}"
            )

    spans_x = (inner_nx == parent_nx)
    spans_y = (inner_ny == parent_ny)

    # A nest spanning a NON-periodic parent axis has nowhere to put a halo and
    # cannot borrow context by wrapping: reject rather than silently inventing
    # a periodicity the parent never had.
    if spans_x and not parent_periodic_x:
        raise ValueError(
            f"Refinement of non-periodic parent {parent_group!r} spans its "
            f"full x extent, leaving no room for the halo; refine a subrange"
        )
    if spans_y and not parent_periodic_y:
        raise ValueError(
            f"Refinement of non-periodic parent {parent_group!r} spans its "
            f"full y extent, leaving no room for the halo; refine a subrange"
        )
    periodic_x = spans_x
    periodic_y = spans_y

    # Halo width per class: SUPPORT_FACTOR * k_i in physical units, i.e. the
    # kernel's own reach, so it shrinks with the cascade and stays a constant
    # number of cells. A spanning axis needs none — its FFT period is already
    # the parent's extent.
    pad_x_per_class = np.zeros(n_classes) if spans_x else SUPPORT_FACTOR * k_values
    pad_y_per_class = np.zeros(n_classes) if spans_y else SUPPORT_FACTOR * k_values

    # z is never periodic. At the domain ground/top the parent is already
    # zero-padded, so no halo is needed (and none exists); elsewhere the halo
    # is the kernel's vertical reach, clipped to the room the parent has.
    at_ground = (z_min <= parent_z_min + 1e-6)
    at_top = (z_max >= parent_z_max - 1e-6)
    if at_ground:
        pad_z_below_per_class = np.zeros(n_classes)
    else:
        pad_z_below_per_class = np.minimum(
            SUPPORT_FACTOR * k_z_values, z_min - parent_z_min)
    if at_top:
        pad_z_above_per_class = np.zeros(n_classes)
    else:
        pad_z_above_per_class = np.minimum(
            SUPPORT_FACTOR * k_z_values, parent_z_max - z_max)

    # An elevated nest bottom is no longer at the surface, so the scalar
    # surface pressure does not describe it; read the parent's 3D pressure at
    # the levels bracketing z_min for a 2D starting pressure.
    if not at_ground:
        with netCDF4.Dataset(parent_path, "r") as ds_pressure:
            grp_pressure = (ds_pressure if parent_group == '/'
                            else ds_pressure[parent_group])
            if 'p' not in grp_pressure.variables:
                raise ValueError(
                    f"Refining with z_min ({z_min}) above the parent bottom "
                    f"({parent_z_min}) requires pressure diagnostics on the "
                    f"parent group {parent_group!r}. Run "
                    f"steam.thermodynamics.compute_diagnostics on the parent "
                    f"first."
                )
            z_index_low = int(np.searchsorted(z_coords, z_min, side='right')) - 1
            z_index_low = max(0, min(len(z_coords) - 2, z_index_low))
            pressure_low = grp_pressure.variables['p'][:, :, z_index_low].astype(np.float32)
            pressure_high = grp_pressure.variables['p'][:, :, z_index_low + 1].astype(np.float32)
        parent_z_low = float(z_coords[z_index_low])
        parent_z_high = float(z_coords[z_index_low + 1])
    else:
        pressure_low = pressure_high = None
        parent_z_low = parent_z_high = None

    # Halo width in PARENT cells, needed only to slice class 0 out of the
    # parent's grid; every later class re-derives its own from the grids.
    pad_cells_x = 0 if spans_x else int(np.ceil(pad_x_per_class[0] / parent_dx))
    pad_cells_y = 0 if spans_y else int(np.ceil(pad_y_per_class[0] / parent_dy))

    # Non-periodic parent: the slice cannot wrap, so the nest must leave the
    # halo room on each side. The per-class shrinking of the halo relaxes only
    # the cascade's memory, not this class-0 extraction.
    if not parent_periodic_x and (x_start < pad_cells_x
                                  or x_stop > parent_nx - pad_cells_x):
        raise ValueError(
            f"Refinement of non-periodic parent {parent_group!r} at "
            f"x=[{x_start},{x_stop}] is too close to the parent's x boundary; "
            f"need at least {pad_cells_x} cells of room on each side "
            f"(parent_nx={parent_nx})"
        )
    if not parent_periodic_y and (y_start < pad_cells_y
                                  or y_stop > parent_ny - pad_cells_y):
        raise ValueError(
            f"Refinement of non-periodic parent {parent_group!r} at "
            f"y=[{y_start},{y_stop}] is too close to the parent's y boundary; "
            f"need at least {pad_cells_y} cells of room on each side "
            f"(parent_ny={parent_ny})"
        )

    # Extraction indices at parent resolution; modular where the parent wraps.
    if spans_x:
        x_indices = np.arange(x_start, x_stop) % parent_nx
    elif parent_periodic_x:
        x_indices = np.arange(x_start - pad_cells_x, x_stop + pad_cells_x) % parent_nx
    else:
        x_indices = np.arange(x_start - pad_cells_x, x_stop + pad_cells_x)
    if spans_y:
        y_indices = np.arange(y_start, y_stop) % parent_ny
    elif parent_periodic_y:
        y_indices = np.arange(y_start - pad_cells_y, y_stop + pad_cells_y) % parent_ny
    else:
        y_indices = np.arange(y_start - pad_cells_y, y_stop + pad_cells_y)

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
    flux_pad = np.ascontiguousarray(flux_3d[np.ix_(x_indices, y_indices, z_indices)])
    del h_3d, qt_3d, flux_3d

    h_mean_1d = np.interp(z_coords_slice, z_profile, h_profile).astype(np.float32)
    qt_mean_1d = np.interp(z_coords_slice, z_profile, qt_profile).astype(np.float32)
    h_pert_pad = h_pad - h_mean_1d[np.newaxis, np.newaxis, :]
    qt_pert_pad = qt_pad - qt_mean_1d[np.newaxis, np.newaxis, :]
    del h_pad, qt_pad

    # Re-scale inherited per-class content to the NEST's interpolation-
    # compensation reference (Thomas's ruling, 2026-07-28). f(k/dx) is a
    # property of the run's own output resolution, so content the parent
    # deposited with f(k/dx_parent) is too weak when re-read at the nest's
    # finer dx: without correction the parent's finest class arrives
    # ~2.2x low, right in the parent-nest seam octaves. When the parent
    # stored its per-class increments (simulate(save_class_increments=
    # True)), add the delta (f_nest/f_parent - 1) * increment per class,
    # replaying each increment down the parent's own regrid chain exactly
    # as the cascade carried it. Classes with ratio ~ 1 (all the
    # well-resolved ones) are skipped, so only the parent's finest few
    # classes are replayed.
    if parent_has_increments and INTERPOLATION_COMPENSATION:
        if parent_is_nest:
            print("WARNING: parent is itself a nest; its stored class "
                  "increments cover only its own classes, not content it "
                  "inherited in turn. Skipping the compensation "
                  "re-scaling of inherited content.")
        else:
            parent_grids = _compute_all_grids(
                np.asarray(k_values_parent, dtype=np.float64),
                parent_nx * parent_dx, parent_ny * parent_dy,
                domain_height, parent_sparsity,
                spheroscale_profile, z_profile, z_min=parent_z_min,
                anisotropy=parent_anisotropy,
            )
            ds_inc = netCDF4.Dataset(parent_path, "r")
            inc_root = (ds_inc if parent_group == '/'
                        else ds_inc[parent_group])["class_increments"]
            n_par = len(k_values_parent)
            for j in range(n_par):
                k_j = float(k_values_parent[j])
                ratio = (_interpolation_compensation(k_j / dx)
                         / _interpolation_compensation(k_j / parent_dx))
                if abs(ratio - 1.0) < 1e-3:
                    continue
                for name, pert_pad in (("h", h_pert_pad),
                                       ("qt", qt_pert_pad)):
                    inc = np.asarray(
                        inc_root[f"c{j:02d}"].variables[name][:],
                        dtype=np.float32)
                    for m in range(j + 1, n_par):
                        inc = zoom_trilinear(
                            inc, (int(parent_grids['nx'][m]),
                                  int(parent_grids['ny'][m]),
                                  int(parent_grids['nz'][m])))
                    pert_pad += (np.float32(ratio - 1.0)
                                 * inc[np.ix_(x_indices, y_indices,
                                              z_indices)])
                    del inc
            ds_inc.close()
            # Scaling content up can push the field past a bound; restore
            # pointwise bounds with the mean-preserving projection once.
            _project_onto_bounds(h_pert_pad, h_mean_1d, h_min, h_max)
            _project_onto_bounds(qt_pert_pad, qt_mean_1d, qt_min, qt_max)
    elif INTERPOLATION_COMPENSATION and not parent_has_increments:
        print("NOTE: parent stored no class_increments; inherited content "
              "keeps the parent's interpolation-compensation reference "
              "(finest inherited class ~2.2x weak at the nest's "
              "resolution). Rerun the parent with "
              "save_class_increments=True to enable the re-scaling.")

    # Note: the nest's flux anomaly needs no explicit bookkeeping. Each class
    # restores the volume mean the flux entered that class with (_advance_flux),
    # so the mean this region carried in the parent simply propagates.

    grids = _compute_all_grids(
        k_values, inner_extent_x, inner_extent_y, nest_height,
        sparsity_factors, spheroscale_profile, z_profile, z_min=z_min,
        pad_x_per_class=pad_x_per_class,
        pad_y_per_class=pad_y_per_class,
        pad_z_below_per_class=pad_z_below_per_class,
        pad_z_above_per_class=pad_z_above_per_class,
        anisotropy=anisotropy,
    )

    # Index window of the nest proper inside each class's padded grid.
    inner_windows = []
    for i in range(n_classes):
        dx_i = float(grids['dx'][i])
        dy_i = float(grids['dy'][i])
        nx_i = int(grids['nx'][i])
        ny_i = int(grids['ny'][i])
        window_x_start = min(nx_i - 1, int(round(pad_x_per_class[i] / dx_i)))
        window_x_stop = min(nx_i, window_x_start
                            + max(1, int(round(inner_extent_x / dx_i))))
        window_y_start = min(ny_i - 1, int(round(pad_y_per_class[i] / dy_i)))
        window_y_stop = min(ny_i, window_y_start
                            + max(1, int(round(inner_extent_y / dy_i))))
        inner_windows.append(
            (window_x_start, window_x_stop, window_y_start, window_y_stop))

    # C_{Phi,k}(z) = n_c^{-1/alpha} C_{Phi,L}(z) (k/L)^H_h is ONE ladder from
    # the root outer scale down; the nest just extends it below the parent's
    # finest class. The parent already committed to a C_{Phi,L}(z); its last
    # class, scaled by (k/k_parent)^H_h, extends that same ladder with no seam
    # at the overlap scale. This is literally "continue what the parent was
    # doing": the parent's own loop never re-measures C_{Phi,L} per class.
    # (A re-measured-at-nest-resolution alternative was considered and ruled
    # out 2026-07-27 -- the mean profile is smooth and interpolatable, and
    # re-measuring leaves a step at the overlap scale; see git history.)
    anchor_k = float(k_values_parent[-1])
    anchor_C_h = np.asarray(C_h_k_parent[-1], dtype=np.float64)[:len(z_coords)]
    anchor_C_qt = np.asarray(C_qt_k_parent[-1], dtype=np.float64)[:len(z_coords)]
    C_h_k = []
    C_qt_k = []
    for i, k in enumerate(k_values):
        hurst_scale = float((k / anchor_k) ** H_h)
        z_i = grids['z_arrays'][i]
        C_h_k.append(
            (np.interp(z_i, z_coords, anchor_C_h) * hurst_scale).astype(np.float32))
        C_qt_k.append(
            (np.interp(z_i, z_coords, anchor_C_qt) * hurst_scale).astype(np.float32))
    C_h_L = float(np.mean(C_h_k[0]))
    C_qt_L = float(np.mean(C_qt_k[0]))

    aspect_k = []
    for i, k in enumerate(k_values):
        spheroscale_i = np.interp(grids['z_arrays'][i], z_profile, spheroscale_profile)
        aspect_k.append((_k_z(anisotropy, k, spheroscale_i) / k).astype(np.float32))

    # b_{Phi,i} sums the current class and all SMALLER ones. Every class
    # smaller than the parent's finest lives in this nest, so the sum is whole.
    b_h_k = _bound_buffers(C_h_k, grids['z_arrays'])
    b_qt_k = _bound_buffers(C_qt_k, grids['z_arrays'])

    if seed is None and parent_seed is not None:
        # zlib.crc32, not hash(): str hashing is salted per process, which
        # would make an auto-seeded nest irreproducible across runs.
        seed = int(parent_seed + zlib.crc32(output_group.encode()) % (2 ** 31))
    child_seeds = np.random.SeedSequence(seed).spawn(n_classes)

    h_pert_refined, qt_pert_refined, flux_refined, final_grid = cascade_loop(
        h_profile, qt_profile, z_profile,
        grids,
        C_h_k, C_qt_k,
        b_h_k, b_qt_k, aspect_k,
        h_min, h_max, qt_min, qt_max,
        min_distance_to_ground,
        sparsity_factors,
        child_seeds,
        n_scale_classes_per_dyad=parent_n_per_dyad,
        h_perturbation=h_pert_pad,
        qt_perturbation=qt_pert_pad,
        flux=flux_pad,
        inner_windows=inner_windows,
        turbulon_shape=turbulon_shape,
        # Turbulon centers are suppressed at the DOMAIN surface and top, not
        # at the nest's edges: a nest floating in the interior has centers
        # above and below it whose tails reach in.
        zero_bottom=at_ground,
        zero_top=at_top,
        device=device,
    )

    # Discard the halo.
    window_x_start, window_x_stop, window_y_start, window_y_stop = inner_windows[-1]
    z_full = final_grid['z']
    dz_full = final_grid['dz']
    z_inner_indices = np.where(
        (z_full >= z_min - 1e-6) & (z_full < z_max - 1e-6))[0]
    if len(z_inner_indices) == 0:
        raise ValueError(
            f"No nest z-levels found in [{z_min}, {z_max}] m at the finest class"
        )
    z_trim_start = int(z_inner_indices[0])
    z_trim_stop = int(z_inner_indices[-1]) + 1

    inner = (slice(window_x_start, window_x_stop),
             slice(window_y_start, window_y_stop),
             slice(z_trim_start, z_trim_stop))
    h_pert_inner = h_pert_refined[inner]
    qt_pert_inner = qt_pert_refined[inner]
    flux_inner = np.ascontiguousarray(flux_refined[inner])

    z_final = z_full[z_trim_start:z_trim_stop].astype(np.float32)
    dz_final = dz_full[z_trim_start:z_trim_stop].astype(np.float32)
    h_mean_final = np.interp(z_final, z_profile, h_profile).astype(np.float32)
    qt_mean_final = np.interp(z_final, z_profile, qt_profile).astype(np.float32)
    spheroscale_final = np.interp(z_final, z_profile, spheroscale_profile).astype(np.float32)

    h_3d_out = np.ascontiguousarray(
        h_mean_final[np.newaxis, np.newaxis, :] + h_pert_inner)
    qt_3d_out = np.ascontiguousarray(
        qt_mean_final[np.newaxis, np.newaxis, :] + qt_pert_inner)
    # The per-class bounded amplitude-preserving add guarantees the bounds
    # over the nest exactly as in simulate(); clamp only the ~1 ulp float32
    # rounding residue of the (perturbation + mean) reassembly.
    np.clip(h_3d_out, np.float32(h_min), np.float32(h_max), out=h_3d_out)
    np.clip(qt_3d_out, np.float32(qt_min), np.float32(qt_max), out=qt_3d_out)

    nx_out = h_3d_out.shape[0]
    ny_out = h_3d_out.shape[1]
    dx_final = inner_extent_x / nx_out
    dy_final = inner_extent_y / ny_out
    # World-absolute coordinates: the parent's own first-cell world position
    # plus the offset within the parent. Using x_start * parent_dx alone would
    # skew any nest whose parent does not itself start at the world origin.
    x_out = (np.arange(nx_out, dtype=np.float32) * dx_final
             + x_start * parent_dx + float(x_coords[0]))
    y_out = (np.arange(ny_out, dtype=np.float32) * dy_final
             + y_start * parent_dy + float(y_coords[0]))

    # 2D starting pressure from the parent's p, interpolated in z to the nest
    # bottom and resampled onto the nest's horizontal grid.
    if pressure_low is not None:
        if parent_z_high > parent_z_low:
            weight = float(np.clip(
                (float(z_final[0]) - parent_z_low) / (parent_z_high - parent_z_low),
                0.0, 1.0))
        else:
            weight = 0.0
        pressure_level = (1.0 - weight) * pressure_low + weight * pressure_high
        if parent_periodic_x:
            inner_x_indices = np.arange(x_start, x_stop) % parent_nx
        else:
            inner_x_indices = np.arange(x_start, x_stop)
        if parent_periodic_y:
            inner_y_indices = np.arange(y_start, y_stop) % parent_ny
        else:
            inner_y_indices = np.arange(y_start, y_stop)
        p_bottom_inner = pressure_level[np.ix_(inner_x_indices, inner_y_indices)]
        if p_bottom_inner.shape == (nx_out, ny_out):
            p_bottom_field = p_bottom_inner.astype(np.float32)
        else:
            p_bottom_field = zoom_bilinear(p_bottom_inner, (nx_out, ny_out))
    else:
        p_bottom_field = None

    # Store C_k on the output z grid so descendants inherit it cleanly, with
    # the halo levels trimmed off (see the matching block in simulate()).
    C_h_k_stored = [np.interp(z_final, grids['z_arrays'][i], C_h_k[i]).astype(np.float32)
                    for i in range(n_classes)]
    C_qt_k_stored = [np.interp(z_final, grids['z_arrays'][i], C_qt_k[i]).astype(np.float32)
                     for i in range(n_classes)]

    simulation_params = {
        'nx': nx_out,
        'ny': ny_out,
        'dx': dx_final,
        'dy': dy_final,
        'dz': dz_final,
        'outer_scale': new_outer_scale,
        'spheroscale': spheroscale_final,
        'spheroscale_profile': spheroscale_profile,
        'domain_height': nest_height,
        'domain_z_min': z_min,
        'profile_dz': profile_dz,
        'sparsity_factors': sparsity_factors,
        'n_scale_classes_per_dyad': parent_n_per_dyad,
        'surface_pressure': surface_pressure,
        'seed': seed,
        'C_h_L': C_h_L,
        'C_qt_L': C_qt_L,
        'n_large_turbulons': 0,   # a nest has no outer-scale turbulons of its own
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
        'parent_group': parent_group,
        'parent_x_slice': np.array([x_start, x_stop], dtype=np.int32),
        'parent_y_slice': np.array([y_start, y_stop], dtype=np.int32),
        'parent_x_offset': float(x_start * parent_dx),
        'parent_y_offset': float(y_start * parent_dy),
        'periodic_x': periodic_x,
        'periodic_y': periodic_y,
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
        flux_3d=flux_inner,
    )
    return parent_path
