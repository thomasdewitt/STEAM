"""Tests for turbulon envelope kernels."""

import numpy as np
import pytest
from steam.simulate import _turbulon_envelope


@pytest.mark.parametrize("k", [50, 100, 500, 1000])
def test_kernel_is_zero_mean_isotropic(k):
    dx = k / 2
    dy = k / 2
    dz = k / 2

    kernel = _turbulon_envelope(k, dx=dx, dy=dy, dz=dz,
                                support_factor=10, shape='mexican_hat')

    assert abs(float(kernel.mean())) < 1e-8


@pytest.mark.parametrize("sparsity, expected_peak", [(1, 2.96781718), (2, 3.0)])
def test_admissibility_correction_is_gaussian_weighted(sparsity, expected_peak):
    """The zero-sum correction rescales the leading constant, not a pedestal.

    Subtracting a Gaussian-weighted mean is algebraically identical to
    replacing mexican_hat's leading 3 by A_opt = sum(rho w)/sum(w), so the
    kernel peak reports the correction directly. A_opt depends only on the
    sparsity factors: 2.9678 at s = 1, and 3.0 (to float32) once s = 2
    resolves the negative shell that carries the continuum cancellation.
    """
    k = 500.0
    spacing = k / (2 * sparsity)
    kernel = _turbulon_envelope(k, dx=spacing, dy=spacing, dz=spacing,
                                support_factor=5, shape='mexican_hat')

    assert kernel.max() == pytest.approx(expected_peak, abs=1e-5)
    # Zero sum, not merely zero mean: this is the admissibility condition.
    assert abs(float(kernel.sum())) < 1e-5

    # No pedestal: at the truncation radius the kernel must be zero to
    # float32, since the envelope itself has decayed to ~1e-52 there. The
    # old flat-mean subtraction left a constant offset instead.
    assert abs(float(kernel[0, 0, 0])) < 1e-20


def test_admissibility_correction_independent_of_support_factor():
    """A_opt is set by the sampling, not by where the kernel is truncated."""
    peaks = [
        float(_turbulon_envelope(500.0, dx=250.0, dy=250.0, dz=250.0,
                                 support_factor=support,
                                 shape='mexican_hat').max())
        for support in (3, 4, 5, 8)
    ]
    for peak in peaks:
        assert peak == pytest.approx(peaks[0], abs=1e-6)
