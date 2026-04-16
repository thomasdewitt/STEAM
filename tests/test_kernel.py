"""Tests for turbulon envelope kernels."""

import numpy as np
import pytest
from steam.simulate import _turbulon_envelope


@pytest.mark.parametrize("k", [50, 100, 500, 1000])
def test_kernel_mean_near_zero_isotropic(k):
    """Isotropic Mexican hat kernel mean should be near zero when di = k/2 for all i.

    With isotropic norm, k_z = k, so di = k_i/2 gives dx = dy = dz = k/2.
    The Mexican hat has zero integral by construction; this checks that the
    discrete kernel with 2 cells per scale captures that accurately.
    """
    dx = k / 2
    dy = k / 2
    dz = k / 2

    kernel = _turbulon_envelope(k, dx=dx, dy=dy, dz=dz,
                                support_factor=10, shape='mexican_hat')

    assert abs(kernel.mean()) < 0.01, (
        f"k={k}: kernel mean={kernel.mean():.4e}, shape={kernel.shape}"
    )
