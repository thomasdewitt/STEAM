"""Benchmark 3D convolutions with periodic x/y and zero-padded z.

Methods compared:
1) steam.utils.convolve_periodic_xy_zeropad_z (Numba direct convolution)
2) steam.utils.convolve_periodic_xy_zeropad_z_ndimage (SciPy ndimage)
3) steam.utils.convolve_periodic_xy_zeropad_z_oa (SciPy OA)
4) Hybrid: FFT in horizontal (x/y), OA in vertical (z)

Default field size is 512x512x128.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

import numpy as np

from steam.utils import (
    convolve_fft_xy_oa_z,
    convolve_periodic_xy_zeropad_z,
    convolve_periodic_xy_zeropad_z_ndimage,
    convolve_periodic_xy_zeropad_z_oa,
)


Array3D = np.ndarray


def _parse_kernel_shapes(spec: str) -> list[tuple[int, int, int]]:
    """Parse format like: '3x3x3,7x7x5,15x15x9'."""
    shapes: list[tuple[int, int, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        dims = item.lower().split("x")
        if len(dims) != 3:
            raise ValueError(f"Invalid kernel spec '{item}'. Use kxXkyXkz.")
        kx, ky, kz = (int(d.strip()) for d in dims)
        if kx < 1 or ky < 1 or kz < 1:
            raise ValueError(f"Kernel dimensions must be positive: {item}")
        shapes.append((kx, ky, kz))
    if not shapes:
        raise ValueError("No kernel shapes parsed.")
    return shapes


def _time_method(
    fn: Callable[[Array3D, Array3D], Array3D],
    field: Array3D,
    kernel: Array3D,
    repeats: int,
) -> tuple[list[float], Array3D]:
    times: list[float] = []
    out: Array3D | None = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(field, kernel)
        times.append(time.perf_counter() - t0)
    assert out is not None
    return times, out


def _fmt_stats(times: list[float]) -> str:
    arr = np.array(times, dtype=np.float64)
    return f"min={arr.min():8.3f}s  mean={arr.mean():8.3f}s  std={arr.std(ddof=0):8.3f}s"


def _maxrss_bytes() -> int:
    """Return ru_maxrss in bytes (platform-adjusted)."""
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss * 1024)


def _fmt_mem(bytes_value: int) -> str:
    return f"{bytes_value / (1024.0 * 1024.0):8.1f}MB"


def _kernel_seed(base_seed: int, kernel_shape: tuple[int, int, int]) -> int:
    kx, ky, kz = kernel_shape
    mix = (kx * 73856093) ^ (ky * 19349663) ^ (kz * 83492791)
    return int((base_seed + mix) % (2**63 - 1))


def _make_field(nx: int, ny: int, nz: int, seed: int) -> Array3D:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nx, ny, nz)).astype(np.float32)


def _make_kernel(kernel_shape: tuple[int, int, int], seed: int) -> Array3D:
    rng = np.random.default_rng(_kernel_seed(seed, kernel_shape))
    return rng.standard_normal(kernel_shape).astype(np.float32)


def _method_table() -> list[tuple[str, Callable[[Array3D, Array3D], Array3D]]]:
    return [
        ("convolve_periodic_xy_zeropad_z", convolve_periodic_xy_zeropad_z),
        ("convolve_periodic_xy_zeropad_z_ndimage", convolve_periodic_xy_zeropad_z_ndimage),
        ("convolve_periodic_xy_zeropad_z_oa", convolve_periodic_xy_zeropad_z_oa),
        ("convolve_fft_xy_oa_z", convolve_fft_xy_oa_z),
    ]


def _run_memory_worker(args: argparse.Namespace) -> None:
    if args.worker_method is None or args.worker_kernel is None:
        raise ValueError("Worker mode requires --worker-method and --worker-kernel.")

    kernel_shape = _parse_kernel_shapes(args.worker_kernel)[0]
    methods = dict(_method_table())
    fn = methods[args.worker_method]
    field = _make_field(args.nx, args.ny, args.nz, args.seed)
    kernel = _make_kernel(kernel_shape, args.seed)
    times, _ = _time_method(fn, field, kernel, repeats=args.repeats)
    payload = {
        "times": times,
        "peak_rss_bytes": _maxrss_bytes(),
    }
    print(json.dumps(payload))


def _measure_method_memory(
    method_name: str,
    kernel_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> int:
    kernel_spec = f"{kernel_shape[0]}x{kernel_shape[1]}x{kernel_shape[2]}"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--nx",
        str(args.nx),
        "--ny",
        str(args.ny),
        "--nz",
        str(args.nz),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--worker-method",
        method_name,
        "--worker-kernel",
        kernel_spec,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout.strip())
    return int(payload["peak_rss_bytes"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=512)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument(
        "--kernels",
        type=str,
        default="3x3x3,7x7x7,15x15x9,31x31x15",
        help="Comma-separated kernel shapes as kxXkyXkz",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check numerical agreement of methods against Numba output.",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        default=True,
        help="Measure peak RSS (process max resident set size) per method (default: on).",
    )
    parser.add_argument(
        "--no-memory",
        dest="memory",
        action="store_false",
        help="Disable peak RSS measurement.",
    )
    parser.add_argument("--worker-method", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-kernel", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_method is not None:
        _run_memory_worker(args)
        return

    field = _make_field(args.nx, args.ny, args.nz, args.seed)
    kernel_shapes = _parse_kernel_shapes(args.kernels)

    methods = _method_table()

    print(f"Field shape: ({args.nx}, {args.ny}, {args.nz}), dtype=float32")
    print(f"Kernels: {kernel_shapes}")
    print(f"Repeats per method: {args.repeats}")
    print(f"Memory reporting: {'on' if args.memory else 'off'}")
    print()

    # Warm up Numba compile outside timings.
    warm_kernel = _make_kernel(kernel_shapes[0], args.seed)
    _ = convolve_periodic_xy_zeropad_z(field, warm_kernel)

    for kernel_shape in kernel_shapes:
        print(f"Kernel {kernel_shape}:")
        kernel = _make_kernel(kernel_shape, args.seed)

        peak_rss_by_method: dict[str, int] = {}
        if args.memory:
            for name, _ in methods:
                peak_rss_by_method[name] = _measure_method_memory(name, kernel_shape, args)

        baseline: Array3D | None = None
        for name, fn in methods:
            times, out = _time_method(fn, field, kernel, repeats=args.repeats)
            line = f"  {name:40s} {_fmt_stats(times)}"
            if args.memory:
                line += f"  peakRSS={_fmt_mem(peak_rss_by_method[name])}"
            if baseline is None:
                baseline = out
            if args.check and baseline is not None and name != "numba_direct":
                max_abs = np.max(np.abs(out - baseline))
                l2 = np.linalg.norm((out - baseline).ravel())
                line += f"  max|diff|={max_abs:.3e}  l2={l2:.3e}"
            print(line)
        print()


if __name__ == "__main__":
    main()
