"""Tests for the Haar-based outer-scale normalization C_{Phi,L}."""

import numpy as np
import pytest

from steam.simulate import _compute_normalization, HAAR_TO_MHAT


def _norm_profile(profile, k_z_L, z, k_values=(1000.0,), outer_scale=1000.0):
    z_arrays = {'z_arrays': [z]}
    return _compute_normalization(
        profile, z, np.full(z.size, k_z_L), np.asarray(k_values), outer_scale,
        z_arrays,
    )


def test_linear_profile_gives_gradient_times_half_window():
    """A constant gradient must produce variability: C_L = lambda * m * k_zL / 2.

    This is the case the previous envelope-column estimator (even kernel)
    annihilated entirely.
    """
    n, dz = 2001, 10.0
    z = np.arange(n) * dz
    slope = 2.0
    k_z_L = 400.0
    C = _norm_profile(100.0 + slope * z, k_z_L, z)
    mid = float(C[0][n // 2])
    expected = HAAR_TO_MHAT * slope * k_z_L / 2
    # The half-window means are exact integrals of the profile's own
    # piecewise-linear interpolant, so this is exact, not approximate.
    assert mid == pytest.approx(expected, rel=1e-6)


def test_constant_profile_gives_zero():
    n, dz = 501, 10.0
    z = np.arange(n) * dz
    C = _norm_profile(np.full(n, 5.0), 400.0, z)
    assert np.all(C[0] == 0.0)


def test_resolution_independence():
    """Halving dz must converge toward, not rescale, the response."""
    slope, k_z_L = 2.0, 400.0
    vals = []
    for n, dz in [(2001, 10.0), (4001, 5.0)]:
        z = np.arange(n) * dz
        C = _norm_profile(100.0 + slope * z, k_z_L, z)
        vals.append(float(C[0][n // 2]))
    coarse, fine = vals
    expected = HAAR_TO_MHAT * slope * k_z_L / 2
    # Not "converges toward" but "is already there": the exact half-window
    # means make the response the same number on any grid.
    assert coarse == pytest.approx(expected, rel=1e-6)
    assert fine == pytest.approx(coarse, rel=1e-6)


def test_hurst_scaling_across_classes():
    """C_k = C_L * (k/L)^H_h is preserved by the Haar estimator."""
    from steam.constants import hurst_horizontal as H_h
    n, dz = 2001, 10.0
    z = np.arange(n) * dz
    z_arrays = {'z_arrays': [z, z]}
    k_values = np.array([1000.0, 500.0])
    C = _compute_normalization(
        100.0 + 2.0 * z, z, np.full(n, 400.0), k_values, 1000.0, z_arrays,
    )
    mid = n // 2
    ratio = float(C[1][mid] / C[0][mid])
    assert ratio == pytest.approx(0.5 ** H_h, rel=1e-5)
