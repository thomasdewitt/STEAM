"""Convolution utilities for periodic x/y and zero-padded z."""

import numpy as np
import torch
from numba import njit, prange
from scipy import ndimage
from scipy.signal import oaconvolve

MEMORY_HEADROOM_BYTES = 2 * 1024**3  # keep this much RAM free; OOM swap-thrashes the host
CUDA_MEMORY_HEADROOM_BYTES = 1 * 1024**3  # keep this much VRAM free for cuFFT plan caches / fragmentation


def available_memory_bytes():
    """RAM currently available to allocate, from /proc/meminfo (Linux)."""
    with open('/proc/meminfo') as meminfo:
        for line in meminfo:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024  # kB -> bytes
    return None


def fft_convolution_bytes(nx, ny, nz, kz, itemsize):
    """Peak working-set bytes of :func:`convolve_fft_xy_oa_z` for one field.

    Overlap-add along z allocates a real accumulator (nx, ny, nz+kz-1), the
    kernel and one block spectrum (nx, ny, n_fft//2+1) complex, and one real
    block (nx, ny, n_fft), where n_fft is the smallest power of two admitting a
    linear kz-tap block convolution.
    """
    n_fft = 1 << max(4, (2 * kz - 1).bit_length())
    half = n_fft // 2 + 1
    return nx * ny * ((nz + kz - 1) * itemsize     # accumulator
                      + 2 * half * 2 * itemsize    # kernel + block spectra (complex)
                      + n_fft * itemsize)          # block output


def cuda_fft_convolution_bytes(nx, ny, nz, kz, n_fft, itemsize):
    """Peak VRAM of the GPU :func:`convolve_fft_xy_oa_z` at FFT length n_fft.

    Resident on the device: the field, the real accumulator (nz+kz-1), and the
    kernel spectrum; in flight for one block, a block spectrum and its real
    output. cuFFT scratch is budgeted at one more spectral array.
    """
    plane = nx * ny
    spectrum = plane * (n_fft // 2 + 1) * 2 * itemsize   # complex
    return (plane * nz * itemsize                         # field on device
            + plane * (nz + kz - 1) * itemsize            # accumulator
            + spectrum                                    # kernel spectrum
            + spectrum                                    # block spectrum in flight
            + plane * n_fft * itemsize                    # block real output
            + spectrum)                                   # cuFFT workspace


def cuda_block_fft_size(nx, ny, nz, kz, itemsize):
    """Largest power-of-two overlap-add FFT length along z that fits free VRAM.

    A single block covers the whole padded array when that fits; otherwise the
    length shrinks toward the kernel length (block_len = n_fft - kz + 1 >= 1).
    Raises MemoryError if even the minimal block will not fit, mirroring the
    host guard in :func:`convolve_fft_xy_oa_z`.
    """
    free, _ = torch.cuda.mem_get_info()
    budget = free - CUDA_MEMORY_HEADROOM_BYTES
    n_fft = 1 << (nz + kz - 2).bit_length()   # whole padded array in one block
    n_fft_min = 1 << (kz - 1).bit_length()    # smallest power of two >= kz
    while n_fft >= n_fft_min:
        if cuda_fft_convolution_bytes(nx, ny, nz, kz, n_fft, itemsize) <= budget:
            return n_fft
        n_fft >>= 1
    requested = cuda_fft_convolution_bytes(nx, ny, nz, kz, n_fft_min, itemsize)
    raise MemoryError(
        f"convolve_fft_xy_oa_z (cuda) needs {requested / 1024**3:.1f} GiB VRAM "
        f"for the minimal block of field shape {(nx, ny, nz)} (kernel kz={kz}) "
        f"but only {free / 1024**3:.1f} GiB is free; refusing to risk OOM. "
        f"Reduce resolution, domain size, or vertical levels."
    )


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


def convolve_fft_xy_oa_z(field, kernel, device='cpu'):
    """Real-FFT overlap-add convolution: circular x/y, linear zero-padded z.

    Each real z block is transformed in all three dimensions with
    :func:`torch.fft.rfftn`; the one-sided spectral axis is z. This avoids the
    persistent complex horizontal transform and complex overlap accumulator
    required by the former ``rfft2`` + complex-z-FFT implementation. It matches
    the ndimage reference to float32 precision (typically <3e-7 relative error).

    ``device`` selects where the transforms run. 'cuda' moves the field to the
    GPU once and the result back once — there is no per-block host<->device
    streaming. A 'cuda' request on a machine without a usable GPU is an error,
    never a silent fall back to the CPU.
    """
    kernel = fold_kernel_to_field(kernel, field.shape)
    nx, ny, nz = field.shape
    kx, ky, kz = kernel.shape
    cx = (kx - 1) // 2
    cy = (ky - 1) // 2
    cz = (kz - 1) // 2
    nz_lin = nz + kz - 1

    # Overlap-add along z: block_len = n_fft - kz + 1 samples advance per block
    # (n_fft a power of two), while x/y keep their exact periods. On CPU n_fft
    # is the smallest length admitting a linear kz-tap block, and the host guard
    # refuses an oversized spectral set (an OOM here swap-thrashes the host). On
    # CUDA n_fft is sized up to the largest that fits free VRAM — a single block
    # when the whole padded array fits — carrying its own VRAM guard.
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "convolve_fft_xy_oa_z: device='cuda' requested but torch.cuda "
                "is unavailable"
            )
        # Return the caching allocator's reserved-but-idle blocks to the driver
        # so mem_get_info reflects the true free VRAM. Without this, each call in
        # a sequence sees only what the previous call left uncached and sizes the
        # block ever smaller until even the minimal block cannot fit.
        torch.cuda.empty_cache()
        n_fft = cuda_block_fft_size(nx, ny, nz, kz, field.itemsize)
    else:
        n_fft = 1 << max(4, (2 * kz - 1).bit_length())
        requested = fft_convolution_bytes(nx, ny, nz, kz, field.itemsize)
        available = available_memory_bytes()
        if available is not None and requested > available - MEMORY_HEADROOM_BYTES:
            raise MemoryError(
                f"convolve_fft_xy_oa_z needs {requested / 1024**3:.1f} GiB for field "
                f"shape {(nx, ny, nz)} (kernel {(kx, ky, kz)}) but only "
                f"{available / 1024**3:.1f} GiB is available; refusing to risk OOM. "
                f"Reduce resolution, domain size, or vertical levels."
            )

    block_len = n_fft - kz + 1
    transform_shape = (nx, ny, n_fft)
    field_t = torch.from_numpy(field).to(device)
    kernel_t = torch.from_numpy(kernel).to(device)

    kernel_padded = torch.zeros((nx, ny, kz), dtype=kernel_t.dtype, device=device)
    kernel_padded[:kx, :ky, :] = kernel_t
    kernel_padded = torch.roll(kernel_padded, shifts=(-cx, -cy), dims=(0, 1))
    kernel_spectrum = torch.fft.rfftn(
        kernel_padded, s=transform_shape, dim=(0, 1, 2),
    )
    del kernel_t, kernel_padded

    accum = torch.zeros((nx, ny, nz_lin), dtype=field_t.dtype, device=device)
    for start in range(0, nz, block_len):
        end = min(start + block_len, nz)
        block_spectrum = torch.fft.rfftn(
            field_t[..., start:end], s=transform_shape, dim=(0, 1, 2),
        )
        block_spectrum.mul_(kernel_spectrum)
        block_out = torch.fft.irfftn(
            block_spectrum, s=transform_shape, dim=(0, 1, 2),
        )
        del block_spectrum
        seg_len = min(n_fft, nz_lin - start)
        accum[..., start:start + seg_len] += block_out[..., :seg_len]
        del block_out
    del field_t, kernel_spectrum

    trimmed = accum[..., cz:cz + nz].contiguous()
    del accum
    return trimmed.cpu().numpy()


def zoom_trilinear(field, target_shape):
    """Resample a 3D float32 field to target_shape with trilinear interpolation.

    Matches scipy.ndimage.zoom(order=1) to float32 precision (the
    corner-aligned sampling convention — torch's align_corners=True).
    """
    t = torch.from_numpy(field)[None, None]  # NCDHW
    out = torch.nn.functional.interpolate(
        t, size=tuple(target_shape), mode="trilinear", align_corners=True,
    )
    return out[0, 0].numpy().astype(np.float32)


def zoom_bilinear(field, target_shape):
    """Resample a 2D float32 field to target_shape with bilinear interpolation.

    Corner-aligned (torch's align_corners=True).
    """
    t = torch.from_numpy(field.astype(np.float32))[None, None]  # NCHW
    out = torch.nn.functional.interpolate(
        t, size=tuple(target_shape), mode="bilinear", align_corners=True,
    )
    return out[0, 0].numpy().astype(np.float32)
