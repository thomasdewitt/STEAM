"""Tests for bound handling: taper, mean-preserving projection, realized norm."""

import importlib

import numpy as np
import netCDF4

sm = importlib.import_module("steam.simulate")

from steam.simulate import (
    simulate,
    _bound_buffers,
    _bound_taper,
    _project_onto_bounds,
    BOUND_BUFFER_MULTIPLE,
)


# ---------------------------------------------------------------------------
# _project_onto_bounds
# ---------------------------------------------------------------------------

def test_projection_restores_bounds_and_preserves_level_means():
    rng = np.random.default_rng(0)
    nx, ny, nz = 24, 24, 6
    lo, hi = -1.0, 1.0
    mean_1d = np.linspace(-0.3, 0.3, nz).astype(np.float32)

    # float64 field so mean preservation can be checked to 1e-10; the
    # production float32 path is covered by the simulate() smoke test.
    perturbation = rng.normal(0.0, 0.1, size=(nx, ny, nz)).astype(np.float64)
    # Push some values past both bounds on a few levels
    perturbation[0, 0, 1] = 3.0
    perturbation[1, 1, 1] = -3.0
    perturbation[:4, :4, 3] = 2.5
    perturbation[5:8, 5:8, 4] = -2.5

    field_before = perturbation.astype(np.float64) + mean_1d.astype(np.float64)
    means_before = field_before.mean(axis=(0, 1))
    untouched = perturbation[:, :, 0].copy()

    _project_onto_bounds(perturbation, mean_1d, lo, hi)

    field_after = perturbation.astype(np.float64) + mean_1d.astype(np.float64)
    assert field_after.min() >= lo - 1e-6
    assert field_after.max() <= hi + 1e-6
    means_after = field_after.mean(axis=(0, 1))
    np.testing.assert_allclose(means_after, means_before, rtol=1e-10, atol=1e-10)

    # Levels with no violations are bit-identical
    np.testing.assert_array_equal(perturbation[:, :, 0], untouched)


def test_projection_noop_when_within_bounds():
    rng = np.random.default_rng(1)
    perturbation = rng.uniform(-0.4, 0.4, size=(8, 8, 4)).astype(np.float32)
    mean_1d = np.zeros(4, dtype=np.float32)
    copy = perturbation.copy()
    _project_onto_bounds(perturbation, mean_1d, -1.0, 1.0)
    np.testing.assert_array_equal(perturbation, copy)


# ---------------------------------------------------------------------------
# _bound_taper
# ---------------------------------------------------------------------------

def test_taper_reduces_amplitude_near_bound_and_is_one_far_away():
    nx, ny, nz = 4, 4, 5
    lo, hi = 0.0, 10.0
    b = np.full(nz, 2.0, dtype=np.float32)

    field = np.full((nx, ny, nz), 5.0, dtype=np.float32)  # far from bounds
    field[:, :, 1] = 0.5    # within one buffer width of the lower bound
    field[:, :, 2] = -1.0   # outside the lower bound
    field[:, :, 3] = 9.0    # within one buffer width of the upper bound

    g = _bound_taper(field.copy(), b, lo, hi)
    np.testing.assert_array_equal(g[:, :, 0], 1.0)          # far: exactly 1
    np.testing.assert_allclose(g[:, :, 1], 0.25, rtol=1e-6)  # (0.5-0)/2
    np.testing.assert_array_equal(g[:, :, 2], 0.0)          # outside: 0
    np.testing.assert_allclose(g[:, :, 3], 0.5, rtol=1e-6)   # (10-9)/2
    np.testing.assert_array_equal(g[:, :, 4], 1.0)


def test_taper_zero_buffer_is_indicator_of_strict_interior():
    b = np.zeros(3, dtype=np.float32)
    field = np.zeros((2, 2, 3), dtype=np.float32)
    field[:, :, 0] = 0.5    # strictly inside
    field[:, :, 1] = 0.0    # on the bound
    field[:, :, 2] = -1.0   # outside
    g = _bound_taper(field.copy(), b, 0.0, 1.0)
    np.testing.assert_array_equal(g[:, :, 0], 1.0)
    np.testing.assert_array_equal(g[:, :, 1], 0.0)
    np.testing.assert_array_equal(g[:, :, 2], 0.0)


# ---------------------------------------------------------------------------
# _bound_buffers
# ---------------------------------------------------------------------------

def test_bound_buffers_close_the_whole_remaining_ladder():
    """b_i sums the current class and ALL smaller ones — including the classes
    below the run's own finest, which is what makes the taper independent of
    where the run stops. The sum is geometric with ratio 2^(-H_h/n_c)."""
    from steam.constants import hurst_horizontal as H_h
    C_k = [np.full(4, c, dtype=np.float32) for c in (4.0, 2.0, 1.0)]
    b_k = _bound_buffers(C_k, 1)
    tail = 1.0 / (1.0 - 2.0 ** -H_h)
    for i, c in enumerate((4.0, 2.0, 1.0)):
        np.testing.assert_allclose(b_k[i], BOUND_BUFFER_MULTIPLE * tail * c,
                                   rtol=1e-6)
    # Two classes per dyad halve the per-class amplitude step, so more of the
    # ladder is still to come and the buffer is wider.
    assert _bound_buffers(C_k, 2)[0][0] > b_k[0][0]


# ---------------------------------------------------------------------------
# Realized norm
# ---------------------------------------------------------------------------

def _root_grids():
    k_values = np.array([8000.0, 4000.0, 2000.0])
    z_profile = np.arange(17, dtype=np.float64) * 200.0
    return sm._compute_all_grids(
        k_values, 16000.0, 16000.0, 3200.0, (1, 1, 1),
        np.full_like(z_profile, 100.0), z_profile,
    ), z_profile


def test_realized_norm_unit_level_mean_with_structure_zero_without(monkeypatch):
    grids, z_profile = _root_grids()
    h_profile = 330_000.0 + 100.0 * z_profile      # structured at all levels
    # Exactly-zero profile: gradients are exactly 0 (a nonzero constant
    # leaves ~1e-13 np.gradient roundoff, which the realized norm amplifies)
    qt_profile = np.zeros_like(z_profile)
    captured = []

    def fake_advance(flux, rng, kernel, flux_noise_scale, n_scale_classes_per_dyad,
                     sparsity_factors, n_zero, zero_bottom, zero_top,
                     device="cpu", window=None):
        return np.ones_like(flux), {"n_clipped": 0}

    def capture_convolution(field, kernel, device="cpu"):
        captured.append(field.copy())
        return np.zeros_like(field)

    monkeypatch.setattr(sm, "_advance_flux", fake_advance)
    monkeypatch.setattr(sm, "CONVOLVE", capture_convolution)

    ones = [np.ones(int(nz), dtype=np.float32) for nz in grids["nz"]]
    # Tiny buffers: g = 1 everywhere strictly inside the (wide) bounds
    tiny = [np.full(int(nz), 1e-30, dtype=np.float32) for nz in grids["nz"]]
    sm.cascade_loop(
        h_profile, qt_profile, z_profile, grids, ones, ones, tiny, tiny, ones,
        0.0, 1e9, -1.0, 1.0,
        0, (1, 1, 1), np.random.SeedSequence(4).spawn(len(grids["k"])),
    )

    # With C = 1, S = 1, g = 1 the captured amplitude is the normalized
    # product itself, mean absolute value exactly one per level. The
    # interpolation compensation is NOT in it: it is applied to the
    # increment the class adds, when the output is composed, never to the
    # pattern the cascade convolves.
    h_W = captured[0]      # first class, h
    qt_W = captured[1]     # first class, qt
    np.testing.assert_allclose(np.abs(h_W).mean(axis=(0, 1)), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(qt_W, 0.0)


# ---------------------------------------------------------------------------
# Full simulate() smoke: bounds hold without any final clip
# ---------------------------------------------------------------------------

def test_simulate_output_within_bounds(tmp_path):
    nz = 50
    z = np.arange(nz) * 30.0
    h = 340e3 - 20e3 * (z / z.max())
    qt = 0.018 - 0.016 * (z / z.max())
    h_min, h_max = 315 * 1004, 355 * 1004
    qt_min, qt_max = 0.0, 30 / 1000
    out = tmp_path / "bounds.nc"
    simulate(h, qt, nx=16, ny=16, dx=500, dy=500,
             outer_scale=8000, spheroscale=100,
             domain_height=3000, profile_dz=30,
             output_path=out, seed=7,
             h_min=h_min, h_max=h_max, qt_min=qt_min, qt_max=qt_max)
    with netCDF4.Dataset(out) as ds:
        h_3d = ds["h"][:]
        qt_3d = ds["qt"][:]
    h_tol = 1e-6 * (h_max - h_min)
    qt_tol = 1e-6 * (qt_max - qt_min)
    assert h_3d.min() >= h_min - h_tol
    assert h_3d.max() <= h_max + h_tol
    assert qt_3d.min() >= qt_min - qt_tol
    assert qt_3d.max() <= qt_max + qt_tol


# ---------------------------------------------------------------------------
# Interpolation-compensation composition (hop-retention primitive)
# ---------------------------------------------------------------------------

def test_compensation_profiles_reduce_to_canonical_table():
    """With every class above the spheroscale, the composed per-level
    profiles equal the pure-canonical scalar table exactly."""
    k_values = np.array([8000.0, 4000.0, 2000.0, 1000.0])
    z_profile = np.arange(10, dtype=np.float64) * 100.0
    z_arrays = [z_profile.copy() for _ in k_values]
    ls = np.full_like(z_profile, 100.0)     # all k >= ls: canonical
    profiles = sm._compensation_profiles(
        k_values, z_arrays, ls, z_profile,
        'piecewise_isotropic_below_spheroscale')
    for i, k in enumerate(k_values):
        expected = sm._interpolation_compensation(2.0 * k / k_values[-1])
        np.testing.assert_allclose(profiles[i], expected, rtol=1e-6)


def test_compensation_profiles_switch_regime_below_spheroscale():
    """Sub-spheroscale levels compose isotropic hop retentions.

    With the spheroscale between class scales, levels where the
    destination classes sit below ls must use the isotropic per-hop
    factors — larger early losses, faster convergence — so the composed
    factor differs from the canonical one, and matches an explicit
    product of HOP_RETENTION['isotropic'] entries where the whole chain
    is isotropic.
    """
    k_values = np.array([8000.0, 4000.0, 2000.0, 1000.0])
    z_profile = np.arange(10, dtype=np.float64) * 100.0
    z_arrays = [z_profile.copy() for _ in k_values]
    # ls huge: every destination class below ls -> fully isotropic chain
    ls_iso = np.full_like(z_profile, 1e6)
    profiles_iso = sm._compensation_profiles(
        k_values, z_arrays, ls_iso, z_profile,
        'piecewise_isotropic_below_spheroscale')
    d_ref = sm._canonical_delivery_reference()
    r = sm.HOP_RETENTION['isotropic']
    # class 0 (k/dx = 16): hops at y = 2, 4, 8 in the isotropic regime
    np.testing.assert_allclose(
        profiles_iso[0], d_ref / (r[2] * r[4] * r[8]), rtol=1e-6)
    # 'canonical' anisotropy option ignores the spheroscale entirely
    profiles_can = sm._compensation_profiles(
        k_values, z_arrays, ls_iso, z_profile, 'canonical')
    for i, k in enumerate(k_values):
        expected = sm._interpolation_compensation(2.0 * k / k_values[-1])
        np.testing.assert_allclose(profiles_can[i], expected, rtol=1e-6)


def test_simulate_smoke_with_subspheroscale_classes(tmp_path):
    """A cascade crossing the spheroscale runs and respects bounds."""
    nz = 50
    z = np.arange(nz) * 30.0
    h = 340e3 - 20e3 * (z / z.max())
    qt = 0.018 - 0.016 * (z / z.max())
    out = tmp_path / "subsphero.nc"
    simulate(h, qt, nx=16, ny=16, dx=500, dy=500,
             outer_scale=8000, spheroscale=3000.0,
             domain_height=5200, profile_dz=30,  # > k_z_L = 5176 m (gate)
             output_path=out, seed=11,
             h_min=315 * 1004, h_max=355 * 1004, qt_min=0.0, qt_max=0.03,
             anisotropy='piecewise_isotropic_below_spheroscale')
    with netCDF4.Dataset(out) as ds:
        h_3d = ds["h"][:]
        qt_3d = ds["qt"][:]
    assert np.all(np.isfinite(h_3d)) and np.all(np.isfinite(qt_3d))
    assert qt_3d.min() >= -1e-9
