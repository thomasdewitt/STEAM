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
