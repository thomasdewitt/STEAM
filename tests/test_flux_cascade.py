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


def test_advance_flux_applies_local_multiplicative_update_and_clips(monkeypatch):
    monkeypatch.setattr(sm, "CONVOLVE", lambda field, kernel: field.copy())
    flux = np.ones((2, 2, 1), dtype=np.float32)
    innovation = np.array([-2.0, 0.0, 1.0, 1.0], dtype=np.float32).reshape(2, 2, 1)

    amplitude, increment, n_clipped = sm._advance_flux(
        flux, innovation, np.ones((1, 1, 1), dtype=np.float32), 1.0,
    )

    np.testing.assert_array_equal(amplitude.ravel(), [-2.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(increment.ravel(), [-2.0, 0.0, 1.0, 1.0])
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
