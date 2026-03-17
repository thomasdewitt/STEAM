"""Benchmark a single 512x512 STEAM simulation.

Reports wall-clock time and peak RSS memory.
"""

from __future__ import annotations

import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from steam.simulate import simulate


def _maxrss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss * 1024)


def _fmt_mem(b: int) -> str:
    return f"{b / (1024.0 * 1024.0):.1f} MB"


def main() -> None:
    nz = 50
    z = np.arange(nz) * 30.0
    h_profile = (340e3 - 20e3 * (z / z.max())).astype(np.float32)
    qt_profile = (0.018 - 0.016 * (z / z.max())).astype(np.float32)

    nx = ny = 700
    dx = dy = 500.0        # 500 m grid spacing → 256 km domain
    outer_scale = nx * dx
    spheroscale = 100.0
    domain_height = 15_000.0
    profile_dz = 30.0

    print(f"nx={nx}, ny={ny}, dx={dx} m, outer_scale={outer_scale/1e3:.0f} km")
    print(f"domain: {nx*dx/1e3:.0f} km × {ny*dy/1e3:.0f} km × {domain_height/1e3:.0f} km")
    print()

    rss_before = _maxrss_bytes()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "bench.nc"
        t0 = time.perf_counter()
        simulate(
            h_profile, qt_profile,
            nx=nx, ny=ny,
            dx=dx, dy=dy,
            outer_scale=outer_scale,
            spheroscale=spheroscale,
            domain_height=domain_height,
            profile_dz=profile_dz,
            output_path=out_path,
            seed=42,
        )
        elapsed = time.perf_counter() - t0

    rss_after = _maxrss_bytes()

    print(f"wall clock : {elapsed:.2f} s")
    print(f"peak RSS   : {_fmt_mem(rss_after)}  (delta {_fmt_mem(rss_after - rss_before)})")


if __name__ == "__main__":
    main()
