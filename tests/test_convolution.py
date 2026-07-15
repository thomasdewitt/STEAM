"""Tests comparing convolution utilities against ndimage reference."""

import numpy as np
import pytest
import torch

from steam.utils import (
    convolve_fft_xy_oa_z,
    convolve_periodic_xy_zeropad_z,
    convolve_periodic_xy_zeropad_z_ndimage,
    convolve_periodic_xy_zeropad_z_oa,
    fold_kernel_to_field,
)


def _reference(field, kernel):
    """Mixed BC reference: wrap in x/y and constant-zero in z via ndimage."""
    return convolve_periodic_xy_zeropad_z_ndimage(field, kernel)


def _reference_folded(field, kernel):
    """ndimage reference for folded-kernel operator used by OA/FFT methods."""
    folded = fold_kernel_to_field(kernel, field.shape)
    return convolve_periodic_xy_zeropad_z_ndimage(field, folded)


def _random_field_and_kernel(rng, field_shape, kernel_shape):
    field = rng.standard_normal(field_shape).astype(np.float32)
    kernel = rng.standard_normal(kernel_shape).astype(np.float32)
    return field, kernel


# -------------------------------------------------------------------
# Basic shape and value tests
# -------------------------------------------------------------------

def test_output_shape_matches_input():
    rng = np.random.default_rng(0)
    field, kernel = _random_field_and_kernel(rng, (16, 16, 10), (5, 5, 5))
    result = convolve_periodic_xy_zeropad_z(field, kernel)
    assert result.shape == field.shape


def test_output_dtype_is_float32():
    rng = np.random.default_rng(0)
    field, kernel = _random_field_and_kernel(rng, (8, 8, 6), (3, 3, 3))
    result = convolve_periodic_xy_zeropad_z(field, kernel)
    assert result.dtype == np.float32


# -------------------------------------------------------------------
# Agreement with ndimage reference
# -------------------------------------------------------------------

@pytest.mark.parametrize("field_shape, kernel_shape", [
    ((8, 8, 8), (3, 3, 3)),
    ((16, 16, 10), (5, 5, 5)),
    ((12, 14, 8), (7, 5, 3)),
    ((10, 10, 6), (3, 7, 5)),
    ((20, 20, 15), (9, 9, 7)),
])
def test_matches_ndimage_various_sizes(field_shape, kernel_shape):
    rng = np.random.default_rng(42)
    field, kernel = _random_field_and_kernel(rng, field_shape, kernel_shape)
    result = convolve_periodic_xy_zeropad_z(field, kernel)
    expected = _reference(field, kernel)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_matches_ndimage_asymmetric_kernel():
    """Kernel with different sizes per axis, not symmetric values."""
    rng = np.random.default_rng(7)
    field, kernel = _random_field_and_kernel(rng, (10, 12, 8), (5, 3, 7))
    result = convolve_periodic_xy_zeropad_z(field, kernel)
    expected = _reference(field, kernel)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_matches_ndimage_kernel_1x1x1():
    """Trivial 1x1x1 kernel should just scale."""
    rng = np.random.default_rng(99)
    field = rng.standard_normal((8, 8, 6)).astype(np.float32)
    kernel = np.array([[[2.5]]], dtype=np.float32)
    result = convolve_periodic_xy_zeropad_z(field, kernel)
    expected = _reference(field, kernel)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


# -------------------------------------------------------------------
# Periodicity in x and y
# -------------------------------------------------------------------

def test_periodic_in_x():
    """Shifting the field along x should shift the output along x."""
    rng = np.random.default_rng(10)
    field, kernel = _random_field_and_kernel(rng, (16, 16, 8), (5, 5, 5))
    result_original = convolve_periodic_xy_zeropad_z(field, kernel)
    shifted_field = np.roll(field, 3, axis=0)
    result_shifted = convolve_periodic_xy_zeropad_z(shifted_field, kernel)
    expected_shifted = np.roll(result_original, 3, axis=0)
    np.testing.assert_allclose(result_shifted, expected_shifted, rtol=1e-5, atol=1e-5)


def test_periodic_in_y():
    """Shifting the field along y should shift the output along y."""
    rng = np.random.default_rng(11)
    field, kernel = _random_field_and_kernel(rng, (16, 16, 8), (5, 5, 5))
    result_original = convolve_periodic_xy_zeropad_z(field, kernel)
    shifted_field = np.roll(field, 5, axis=1)
    result_shifted = convolve_periodic_xy_zeropad_z(shifted_field, kernel)
    expected_shifted = np.roll(result_original, 5, axis=1)
    np.testing.assert_allclose(result_shifted, expected_shifted, rtol=1e-5, atol=1e-5)


# -------------------------------------------------------------------
# Zero-padding in z
# -------------------------------------------------------------------

def test_not_periodic_in_z():
    """Shifting along z should NOT produce a simple shift in output."""
    rng = np.random.default_rng(12)
    field, kernel = _random_field_and_kernel(rng, (8, 8, 12), (3, 3, 5))
    result_original = convolve_periodic_xy_zeropad_z(field, kernel)
    shifted_field = np.roll(field, 2, axis=2)
    result_shifted = convolve_periodic_xy_zeropad_z(shifted_field, kernel)
    rolled_original = np.roll(result_original, 2, axis=2)
    # Should NOT match (unlike periodic axes)
    assert not np.allclose(result_shifted, rolled_original, atol=1e-3)


def test_z_boundary_attenuation():
    """A spike near z=0 should spread to fewer z-neighbors than an interior spike."""
    kernel = np.ones((1, 1, 5), dtype=np.float32)
    # Interior spike: kernel spreads across 5 z-levels
    field_interior = np.zeros((8, 8, 20), dtype=np.float32)
    field_interior[4, 4, 10] = 1.0
    response_interior = convolve_periodic_xy_zeropad_z(field_interior, kernel)[4, 4, :]
    # Boundary spike: zero-padding truncates the spread
    field_boundary = np.zeros((8, 8, 20), dtype=np.float32)
    field_boundary[4, 4, 0] = 1.0
    response_boundary = convolve_periodic_xy_zeropad_z(field_boundary, kernel)[4, 4, :]
    # Interior spike touches 5 z-levels, boundary spike touches fewer
    assert np.count_nonzero(response_boundary) < np.count_nonzero(response_interior)


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------

def test_zero_field_gives_zero_output():
    field = np.zeros((8, 8, 6), dtype=np.float32)
    kernel = np.ones((3, 3, 3), dtype=np.float32)
    result = convolve_periodic_xy_zeropad_z(field, kernel)
    np.testing.assert_array_equal(result, 0.0)


def test_zero_kernel_gives_zero_output():
    rng = np.random.default_rng(20)
    field = rng.standard_normal((8, 8, 6)).astype(np.float32)
    kernel = np.zeros((3, 3, 3), dtype=np.float32)
    result = convolve_periodic_xy_zeropad_z(field, kernel)
    np.testing.assert_array_equal(result, 0.0)


def test_identity_kernel():
    """A single 1.0 at kernel center should return the field unchanged."""
    rng = np.random.default_rng(30)
    field = rng.standard_normal((10, 10, 8)).astype(np.float32)
    kernel = np.zeros((3, 3, 3), dtype=np.float32)
    kernel[1, 1, 1] = 1.0
    result = convolve_periodic_xy_zeropad_z(field, kernel)
    np.testing.assert_allclose(result, field, rtol=1e-6, atol=1e-6)


def test_kernel_larger_than_field_in_xy():
    """Kernel bigger than field in x,y should still work (periodic wrapping)."""
    rng = np.random.default_rng(50)
    field = rng.standard_normal((4, 4, 6)).astype(np.float32)
    kernel = rng.standard_normal((9, 9, 3)).astype(np.float32)
    result = convolve_periodic_xy_zeropad_z(field, kernel)
    expected = _reference(field, kernel)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


# -------------------------------------------------------------------
# Coverage for additional utils convolution functions
# -------------------------------------------------------------------

@pytest.mark.parametrize("field_shape, kernel_shape", [
    ((8, 8, 8), (3, 3, 3)),
    ((12, 10, 7), (5, 3, 5)),
    ((4, 5, 8), (9, 7, 3)),
])
def test_oa_matches_ndimage_reference(field_shape, kernel_shape):
    rng = np.random.default_rng(60)
    field, kernel = _random_field_and_kernel(rng, field_shape, kernel_shape)
    result = convolve_periodic_xy_zeropad_z_oa(field, kernel)
    expected = _reference_folded(field, kernel)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("field_shape, kernel_shape", [
    ((8, 8, 8), (3, 3, 3)),
    ((10, 12, 8), (5, 3, 7)),
    ((4, 5, 8), (9, 7, 3)),
])
def test_fft_xy_oa_z_matches_ndimage_reference(field_shape, kernel_shape):
    rng = np.random.default_rng(61)
    field, kernel = _random_field_and_kernel(rng, field_shape, kernel_shape)
    result = convolve_fft_xy_oa_z(field, kernel)
    expected = _reference_folded(field, kernel)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("field_shape, kernel_shape", [
    ((8, 8, 8), (3, 3, 3)),
    ((10, 12, 40), (5, 3, 7)),
    ((16, 16, 60), (9, 9, 11)),
])
def test_fft_xy_oa_z_cuda_matches_cpu(field_shape, kernel_shape):
    """Same seed, same field: the GPU path matches the CPU path to float32."""
    rng = np.random.default_rng(123)
    field, kernel = _random_field_and_kernel(rng, field_shape, kernel_shape)
    cpu = convolve_fft_xy_oa_z(field, kernel, device='cpu')
    gpu = convolve_fft_xy_oa_z(field, kernel, device='cuda')
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)


def test_cuda_request_without_gpu_raises(monkeypatch):
    """device='cuda' with no usable GPU is an error, never a silent fallback."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(1)
    field, kernel = _random_field_and_kernel(rng, (8, 8, 8), (3, 3, 3))
    with pytest.raises(RuntimeError):
        convolve_fft_xy_oa_z(field, kernel, device='cuda')


def test_fold_kernel_to_field_preserves_sum_when_folding():
    rng = np.random.default_rng(70)
    kernel = rng.standard_normal((9, 7, 5)).astype(np.float32)
    folded = fold_kernel_to_field(kernel, (4, 5, 8))
    assert folded.shape == (4, 5, 5)
    np.testing.assert_allclose(folded.sum(), kernel.sum(), rtol=0, atol=1e-5)


def test_fold_kernel_to_field_noop_when_kernel_fits():
    rng = np.random.default_rng(71)
    kernel = rng.standard_normal((3, 5, 3)).astype(np.float32)
    folded = fold_kernel_to_field(kernel, (8, 8, 8))
    assert folded.shape == kernel.shape
    np.testing.assert_array_equal(folded, kernel)
