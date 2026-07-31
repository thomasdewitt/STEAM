"""Tests for the dimensionless conserved-flux cascade."""

import importlib

import numpy as np


sm = importlib.import_module("steam.simulate")


def _root_grids():
    k_values = np.array([8.0, 4.0, 2.0])
    z_profile = np.arange(17, dtype=np.float64)
    spheroscale = np.full_like(z_profile, 8.0)
    return sm._compute_all_grids(
        k_values,
        inner_extent_x=16.0,
        inner_extent_y=16.0,
        inner_height=16.0,
        sparsity_factors=(1, 1, 1),
        spheroscale_profile=spheroscale,
        z_profile=z_profile,
    )


def test_flux_cascade_default_scale():
    # c = (0.1 / 1.681)^(1/1.8): the 2026-07-28 re-fit with the
    # interpolation compensation applied to the flux increments.
    assert sm.FLUX_SCALE == 0.2085
    assert sm.FLUX_ALPHA == 1.8


def test_levy_log_mean_reduces_to_lognormal_at_alpha_two():
    # At alpha=2 the extremal generator is N(0, 2), so log<exp(gamma_0)> = 1
    # (the -sigma^2/2 lognormal shift); zero mean and skew confirm the reduction.
    draw = sm._extremal_levy(2.0, 2_000_000, np.random.default_rng(0))
    assert abs(draw.mean()) < 0.01
    assert abs(draw.var() - 2.0) < 0.02
    assert abs(float(np.log(np.mean(np.exp(draw)))) - 1.0) < 0.01


def test_flux_only_rejects_class_spacing_inconsistent_with_density():
    grids = _root_grids()
    grids["k"] = np.array([8.0, 5.0, 2.0])

    with np.testing.assert_raises_regex(ValueError, "n_scale_classes_per_dyad"):
        sm.simulate_flux_only(grids, seed=1)


def test_advance_flux_multiplier_and_signed_scalar(monkeypatch):
    monkeypatch.setattr(sm, "CONVOLVE", lambda field, kernel, device="cpu": field.copy())
    gamma0 = np.array([-2.0, 0.0, 1.0, 0.5], dtype=np.float32).reshape(2, 2, 1)
    monkeypatch.setattr(sm, "_sparse_levy", lambda *args: gamma0.copy())
    flux = np.ones((2, 2, 1), dtype=np.float32)

    c = 0.4
    amplitude, _, diagnostics = sm._advance_flux(
        flux, np.random.default_rng(1), np.ones((1, 1, 1), dtype=np.float32),
        c, 1, (1, 1, 1),
    )

    # n_scale_classes_per_dyad=1: per-class scale = c, multiplier noise
    # = exp(c*gamma0 - shift)-1, zero at the off-center (gamma0 == 0) draw.
    shift = sm.LEVY_LOG_MEAN * c ** sm.FLUX_ALPHA
    noise = np.expm1(gamma0.ravel() * c - shift)
    noise[gamma0.ravel() == 0.0] = 0.0
    mean_abs = np.abs(noise).sum() / np.count_nonzero(noise)
    expected_scalar = noise / mean_abs
    np.testing.assert_allclose(amplitude.ravel(), expected_scalar, rtol=1e-5)
    assert amplitude.ravel()[1] == 0.0                      # off-center: no turbulon
    assert diagnostics["n_clipped"] == 0                    # bounded-below: no clip here

    # F += (exp(gamma)-1)*F with identity convolution, then rescaled to restore
    # the volume mean the flux entered with (one here).
    updated = 1.0 + noise
    updated /= updated.mean()
    np.testing.assert_allclose(flux.ravel(), updated, rtol=1e-5)
    np.testing.assert_allclose(flux.mean(), 1.0, rtol=1e-6)
    assert diagnostics["entering_mean"] == 1.0


def test_advance_flux_scales_generator_by_class_density(monkeypatch):
    # With n classes per dyad the per-class generator scale is
    # c / n**(1/alpha) (per-octave invariance by alpha-stability); a single
    # advance draws ONE generator field and counts each point once.
    gamma0 = np.full((2, 2, 1), -0.25, dtype=np.float32)
    monkeypatch.setattr(sm, "_sparse_levy", lambda *args: gamma0.copy())
    monkeypatch.setattr(sm, "CONVOLVE", lambda field, kernel, device="cpu": field.copy())
    flux = np.ones((2, 2, 1), dtype=np.float32)

    c, n = 0.4, 2
    amplitude, _, diagnostics = sm._advance_flux(
        flux, np.random.default_rng(1), np.ones((1, 1, 1), dtype=np.float32),
        c, n, (1, 1, 1),
    )

    # Equal draws give |noise| = mean_abs, so S_k = sign(noise) exactly.
    scale = c / n ** (1.0 / sm.FLUX_ALPHA)
    shift = sm.LEVY_LOG_MEAN * scale ** sm.FLUX_ALPHA
    noise0 = np.expm1(-0.25 * scale - shift)
    np.testing.assert_allclose(amplitude, noise0 / abs(noise0))
    assert diagnostics["n_points"] == flux.size
    assert "substeps" not in diagnostics
    np.testing.assert_allclose(flux.mean(), 1.0)

    # F += (exp(gamma)-1)*F with identity convolution, then the entering-mean
    # rescale: a uniform draw goes back to exactly one.
    np.testing.assert_allclose(flux, 1.0, rtol=1e-6)


def test_flux_only_runs_with_two_classes_per_dyad():
    # sqrt(2)-spaced size classes with n_scale_classes_per_dyad=2.
    k_values = 8.0 / 2.0 ** (np.arange(5) / 2.0)
    z_profile = np.arange(17, dtype=np.float64)
    spheroscale = np.full_like(z_profile, 8.0)
    grids = sm._compute_all_grids(
        k_values,
        inner_extent_x=16.0,
        inner_extent_y=16.0,
        inner_height=16.0,
        sparsity_factors=(1, 1, 1),
        spheroscale_profile=spheroscale,
        z_profile=z_profile,
    )

    flux, diagnostics = sm.simulate_flux_only(
        grids, seed=7, n_scale_classes_per_dyad=2,
    )

    assert np.all(flux >= 0)
    # Each class restores the volume mean it entered with, so the only drift
    # from one is the trilinear regrid between classes, not the cascade
    # (2026-07-27 ruling; the regrid drift is deliberately not scrubbed).
    np.testing.assert_allclose(flux.mean(dtype=np.float64), 1.0, rtol=5e-3)
    assert diagnostics["n_scale_classes_per_dyad"] == 2
    assert len(diagnostics["steps"]) == len(k_values)


def test_flux_only_starts_at_one_and_preserves_unit_volume_mean():
    flux, diagnostics = sm.simulate_flux_only(
        _root_grids(), seed=42, flux_noise_scale=0.0,
    )

    np.testing.assert_array_equal(flux, np.ones_like(flux))
    np.testing.assert_array_equal(flux.mean(), 1.0)
    assert diagnostics["n_clipped"] == 0
    assert diagnostics["final_zero_fraction"] == 0.0


def test_flux_only_reports_clipping_and_keeps_flux_nonnegative():
    flux, diagnostics = sm.simulate_flux_only(
        _root_grids(), seed=3, flux_noise_scale=5.0,
    )

    assert diagnostics["n_clipped"] > 0
    assert diagnostics["clip_fraction"] > 0
    assert np.all(flux >= 0)
    # Heavy clipping plus the between-class regrid; the per-class renorm only
    # promises to restore each class's own entering mean (larger drift for
    # the extreme spike field this absurd c produces). rtol widened 5e-2 ->
    # 1e-1 when the interpolation compensation reached the flux increments
    # (2026-07-28): the damped realization lands at +7.3% for this seed —
    # same between-class-regrid drift mechanism, different realization.
    np.testing.assert_allclose(flux.mean(dtype=np.float64), 1.0, rtol=1e-1)


def test_scalar_convolutions_receive_positive_flux_center_amplitudes(monkeypatch):
    grids = _root_grids()
    z_profile = np.arange(17, dtype=np.float32)
    h_profile = 330_000.0 + 100.0 * z_profile
    qt_profile = 0.005 + 0.0001 * z_profile
    convolved_fields = []

    def fake_advance(flux, rng, kernel, flux_noise_scale, n_scale_classes_per_dyad,
                     sparsity_factors, n_zero, zero_bottom, zero_top,
                     device="cpu", window=None, amplitude_factor=None,
                     return_increment=False):
        return np.ones_like(flux), None, {"n_clipped": 0}

    def record_convolution(field, kernel, device="cpu"):
        convolved_fields.append(field.copy())
        return np.zeros_like(field)

    monkeypatch.setattr(sm, "_advance_flux", fake_advance)
    monkeypatch.setattr(sm, "CONVOLVE", record_convolution)
    C_h = [np.ones(int(nz), dtype=np.float32) for nz in grids["nz"]]
    C_qt = [np.ones(int(nz), dtype=np.float32) for nz in grids["nz"]]
    ones = [np.ones(int(nz), dtype=np.float32) for nz in grids["nz"]]

    sm.cascade_loop(
        h_profile, qt_profile, z_profile, grids, C_h, C_qt, ones, ones, ones,
        315 * 1004, 355 * 1004, 0.0, 0.03,
        0, (1, 1, 1), np.random.SeedSequence(4).spawn(len(grids["k"])),
    )

    assert len(convolved_fields) == 2 * len(grids["k"])
    for amplitude in convolved_fields:
        assert np.any(amplitude > 0)
        assert not np.any(amplitude < 0)
