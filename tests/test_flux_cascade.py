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
    assert sm.FLUX_SCALE == 0.5


def test_flux_only_rejects_non_dyadic_classes():
    grids = _root_grids()
    grids["k"] = np.array([8.0, 5.0, 2.0])

    with np.testing.assert_raises_regex(ValueError, "dyadic"):
        sm.simulate_flux_only(grids, seed=1)


def test_advance_flux_applies_local_multiplicative_update_and_clips(monkeypatch):
    monkeypatch.setattr(sm, "CONVOLVE", lambda field, kernel: field.copy())
    flux = np.ones((2, 2, 1), dtype=np.float32)
    innovation = np.array([-2.0, 0.0, 1.0, 1.0], dtype=np.float32).reshape(2, 2, 1)

    amplitude, n_clipped = sm._advance_flux(
        flux, innovation, np.ones((1, 1, 1), dtype=np.float32), 1.0,
    )

    np.testing.assert_array_equal(amplitude.ravel(), [-2.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(flux.ravel(), [0.0, 0.8, 1.6, 1.6])
    assert n_clipped == 1
    np.testing.assert_allclose(flux.mean(axis=(0, 1)), 1.0)


def test_flux_only_starts_at_one_and_preserves_unit_horizontal_mean():
    flux, diagnostics = sm.simulate_flux_only(
        _root_grids(), seed=42, flux_noise_scale=0.0,
    )

    np.testing.assert_array_equal(flux, np.ones_like(flux))
    np.testing.assert_array_equal(flux.mean(axis=(0, 1)), 1.0)
    assert diagnostics["n_clipped"] == 0
    assert diagnostics["final_zero_fraction"] == 0.0


def test_flux_only_reports_clipping_and_keeps_flux_nonnegative():
    flux, diagnostics = sm.simulate_flux_only(
        _root_grids(), seed=9, flux_noise_scale=5.0,
    )

    assert diagnostics["n_clipped"] > 0
    assert diagnostics["clip_fraction"] > 0
    assert np.all(flux >= 0)
    np.testing.assert_allclose(flux.mean(axis=(0, 1)), 1.0, atol=5e-7)


def test_scalar_convolutions_receive_signed_flux_center_amplitudes(monkeypatch):
    grids = _root_grids()
    z_profile = np.arange(17, dtype=np.float32)
    h_profile = 330_000.0 + 100.0 * z_profile
    qt_profile = 0.005 + 0.0001 * z_profile
    convolved_fields = []

    def fake_advance(flux, innovation, kernel, flux_noise_scale):
        amplitude = np.ones_like(innovation)
        amplitude[::2, :, :] = -1.0
        return amplitude, 0

    def record_convolution(field, kernel):
        convolved_fields.append(field.copy())
        return np.zeros_like(field)

    monkeypatch.setattr(sm, "_advance_flux", fake_advance)
    monkeypatch.setattr(sm, "CONVOLVE", record_convolution)
    C_h = [np.ones(int(nz), dtype=np.float32) for nz in grids["nz"]]
    C_qt = [np.ones(int(nz), dtype=np.float32) for nz in grids["nz"]]

    sm.cascade_loop(
        h_profile, qt_profile, z_profile, grids, C_h, C_qt,
        315 * 1004, 355 * 1004, 0.0, 0.03,
        0, (1, 1, 1), np.random.SeedSequence(4).spawn(len(grids["k"])),
    )

    assert len(convolved_fields) == 2 * len(grids["k"])
    for amplitude in convolved_fields:
        assert np.any(amplitude > 0)
        assert np.any(amplitude < 0)
