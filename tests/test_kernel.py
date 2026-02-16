"""Tests for the Mexican hat kernel."""

import numpy as np
import pytest
from steam.simulate import _mexican_hat_kernel


@pytest.mark.parametrize("k", [50, 100, 500, 1000])
@pytest.mark.parametrize("spheroscale", [10, 50, 100, 500])
def test_kernel_mean_near_zero(k, spheroscale):
    """Kernel mean should be near zero for various scales and spheroscales.

    Uses oversampling_factors=(1,1,2): dx=dy=k, dz=k_z/2.
    """
    H_z = 5 / 9
    k_z = spheroscale * (k / spheroscale) ** H_z
    dx = k          # oversampling_factor 1 in x
    dy = k          # oversampling_factor 1 in y
    dz = k_z / 2    # oversampling_factor 2 in z

    kernel = _mexican_hat_kernel(k, spheroscale, dx, dy, dz, support_factor=10)

    assert abs(kernel.mean()) < 0.01, (
        f"k={k}, spheroscale={spheroscale}: kernel mean={kernel.mean():.4e}, "
        f"shape={kernel.shape}"
    )
