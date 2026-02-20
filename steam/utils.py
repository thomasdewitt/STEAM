"""Convolution utilities for periodic x/y and zero-padded z."""

import numpy as np
from numba import njit, prange
from scipy import ndimage
from scipy.fft import irfft2, rfft2
from scipy.signal import oaconvolve


@njit(parallel=True, cache=True)
def convolve_periodic_xy_zeropad_z(field, kernel):
    """3D convolution: periodic in axes 0 and 1, zero-padded in axis 2.

    This is a direct spatial-domain implementation replacing the
    scipy.signal.oaconvolve approach. Parallelized over axis 0 via numba.

    Parameters
    ----------
    field : float32 array, shape (nx, ny, nz)
    kernel : float32 array, shape (kx, ky, kz)
        Centered kernel (dimensions should be odd).

    Returns
    -------
    result : float32 array, shape (nx, ny, nz)
        Convolution result with same shape as input field.
    """
    nx, ny, nz = field.shape
    kx, ky, kz = kernel.shape
    half_kx = kx // 2
    half_ky = ky // 2
    half_kz = kz // 2

    result = np.empty((nx, ny, nz), dtype=np.float32)
    # Precompute wrapped horizontal indices so modulo is not in the hot loop.
    sx_lut = np.empty((nx, kx), dtype=np.int64)
    sy_lut = np.empty((ny, ky), dtype=np.int64)

    for ix in range(nx):
        for dkx in range(kx):
            sx_lut[ix, dkx] = (ix - dkx + half_kx) % nx

    for iy in range(ny):
        for dky in range(ky):
            sy_lut[iy, dky] = (iy - dky + half_ky) % ny

    for ix in prange(nx):
        sx_row = sx_lut[ix]
        for iy in range(ny):
            sy_row = sy_lut[iy]
            for iz in range(nz):
                total = np.float64(0.0)
                for dkx in range(kx):
                    # convolution flips the kernel: source = output - kernel + center
                    sx = sx_row[dkx]
                    for dky in range(ky):
                        sy = sy_row[dky]
                        for dkz in range(kz):
                            sz = iz - dkz + half_kz
                            if 0 <= sz < nz:
                                total += field[sx, sy, sz] * kernel[dkx, dky, dkz]
                result[ix, iy, iz] = np.float32(total)

    return result


def fold_kernel_to_field(kernel, field_shape):
    """Fold a kernel larger than the field onto the field size for periodic axes."""
    result = kernel
    for axis in (0, 1):
        n_field = field_shape[axis]
        n_kern = result.shape[axis]
        if n_kern <= n_field:
            continue
        folded_shape = list(result.shape)
        folded_shape[axis] = n_field
        folded = np.zeros(folded_shape, dtype=result.dtype)
        center = n_kern // 2
        for i in range(n_kern):
            target = (i - center) % n_field
            slc_src = [slice(None)] * 3
            slc_dst = [slice(None)] * 3
            slc_src[axis] = i
            slc_dst[axis] = target
            folded[tuple(slc_dst)] += result[tuple(slc_src)]
        result = np.roll(folded, n_field // 2, axis=axis)
    return result


def convolve_periodic_xy_zeropad_z_oa(field, kernel):
    """SciPy OA implementation: periodic in x/y and zero-padded in z."""
    kernel = fold_kernel_to_field(kernel, field.shape)
    kx, ky, kz = kernel.shape
    pad_x_l, pad_x_r = kx // 2, (kx - 1) // 2
    pad_y_l, pad_y_r = ky // 2, (ky - 1) // 2
    pad_z_l, pad_z_r = kz // 2, (kz - 1) // 2
    padded = np.pad(field, ((pad_x_l, pad_x_r), (pad_y_l, pad_y_r), (0, 0)), mode="wrap")
    padded = np.pad(
        padded,
        ((0, 0), (0, 0), (pad_z_l, pad_z_r)),
        mode="constant",
        constant_values=0,
    )
    result = oaconvolve(padded, kernel, mode="valid")
    return result.astype(np.float32)


def convolve_periodic_xy_zeropad_z_ndimage(field, kernel):
    """ndimage implementation: periodic in x/y and zero-padded in z."""
    nx, ny, nz = field.shape
    kx, ky, kz = kernel.shape
    pad_x_l, pad_x_r = kx // 2, (kx - 1) // 2
    pad_y_l, pad_y_r = ky // 2, (ky - 1) // 2
    pad_z_l, pad_z_r = kz // 2, (kz - 1) // 2

    padded = np.pad(field, ((pad_x_l, pad_x_r), (pad_y_l, pad_y_r), (0, 0)), mode="wrap")
    padded = np.pad(
        padded,
        ((0, 0), (0, 0), (pad_z_l, pad_z_r)),
        mode="constant",
        constant_values=0.0,
    )
    filtered = ndimage.convolve(
        padded,
        kernel,
        mode="constant",
        cval=0.0,
        origin=tuple(-1 if (s % 2 == 0) else 0 for s in kernel.shape),
    )
    result = filtered[
        pad_x_l : pad_x_l + nx,
        pad_y_l : pad_y_l + ny,
        pad_z_l : pad_z_l + nz,
    ]
    return result.astype(np.float32)


def convolve_fft_xy_oa_z(field, kernel):
    """Hybrid method: circular FFT in x/y and OA convolution in z."""
    kernel = fold_kernel_to_field(kernel, field.shape)
    nx, ny, _ = field.shape
    kx, ky, _ = kernel.shape
    center_x = (kx - 1) // 2
    center_y = (ky - 1) // 2

    kernel_padded = np.zeros((nx, ny, kernel.shape[2]), dtype=np.float32)
    kernel_padded[:kx, :ky, :] = kernel
    kernel_padded = np.roll(kernel_padded, -center_x, axis=0)
    kernel_padded = np.roll(kernel_padded, -center_y, axis=1)

    field_fft_xy = rfft2(field, axes=(0, 1))
    kernel_fft_xy = rfft2(kernel_padded, axes=(0, 1))
    out_fft_xy = oaconvolve(field_fft_xy, kernel_fft_xy, mode="same", axes=2)
    result = irfft2(out_fft_xy, s=(nx, ny), axes=(0, 1))
    return result.astype(np.float32)
