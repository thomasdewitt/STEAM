"""Benchmark STEAM simulation over multiple realizations.

Each realization runs in a subprocess so peak RSS is measured fresh per run.
Reports wall-clock time and peak RSS averaged over n_realizations.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _fmt_mem(b: int) -> str:
    return f"{b / (1024.0 * 1024.0):.1f} MB"


def _maxrss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss * 1024)


def _run_worker(args: argparse.Namespace) -> None:
    from steam.simulate import simulate

    z = np.arange(args.nz) * args.profile_dz
    h_profile = (340e3 - 20e3 * (z / z.max())).astype(np.float32)
    qt_profile = (0.018 - 0.016 * (z / z.max())).astype(np.float32)

    t0 = time.perf_counter()
    simulate(
        h_profile, qt_profile,
        nx=args.nx, ny=args.ny,
        dx=args.dx, dy=args.dy,
        outer_scale=args.outer_scale,
        spheroscale=args.spheroscale,
        domain_height=args.domain_height,
        profile_dz=args.profile_dz,
        output_path=args.output_path,
        seed=args.seed,
    )
    payload = {
        "elapsed_s": time.perf_counter() - t0,
        "peak_rss_bytes": _maxrss_bytes(),
    }
    print(json.dumps(payload))


def _run_one(seed: int, nx: int, ny: int, dx: float, dy: float,
             outer_scale: float, spheroscale: float,
             domain_height: float, profile_dz: float,
             nz: int, out_path: str) -> tuple[float, int]:
    """Run a single simulation in a subprocess; return (elapsed_s, peak_rss_bytes)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--seed", str(seed),
        "--nx", str(nx),
        "--ny", str(ny),
        "--dx", str(dx),
        "--dy", str(dy),
        "--outer-scale", str(outer_scale),
        "--spheroscale", str(spheroscale),
        "--domain-height", str(domain_height),
        "--profile-dz", str(profile_dz),
        "--nz", str(nz),
        "--output-path", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"Worker subprocess failed with exit code {result.returncode}")
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError("Worker subprocess produced no stdout payload")
    payload = json.loads(stdout_lines[-1])
    return float(payload["elapsed_s"]), int(payload["peak_rss_bytes"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--ny", type=int, default=1024)
    parser.add_argument("--dx", type=float, default=500.0)
    parser.add_argument("--dy", type=float, default=500.0)
    parser.add_argument("--outer-scale", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--spheroscale", type=float, default=100.0)
    parser.add_argument("--domain-height", type=float, default=15_000.0)
    parser.add_argument("--profile-dz", type=float, default=30.0)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--n-realizations", type=int, default=5)
    parser.add_argument("--output-path", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _run_worker(args)
        return

    nz = args.nz
    nx = args.nx
    ny = args.ny
    dx = args.dx
    dy = args.dy
    outer_scale = nx * dx
    spheroscale = args.spheroscale
    domain_height = args.domain_height
    profile_dz = args.profile_dz
    n_realizations = args.n_realizations

    print(f"nx={nx}, ny={ny}, dx={dx} m, outer_scale={outer_scale/1e3:.0f} km")
    print(f"domain: {nx*dx/1e3:.0f} km × {ny*dy/1e3:.0f} km × {domain_height/1e3:.0f} km")
    print(f"averaging over {n_realizations} realizations")
    print()

    elapsed_times = []
    peak_rss_values = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_realizations):
            out_path = str(Path(tmpdir) / f"bench_{i}.nc")
            elapsed, peak_rss = _run_one(
                seed=i, nx=nx, ny=ny, dx=dx, dy=dy,
                outer_scale=outer_scale, spheroscale=spheroscale,
                domain_height=domain_height, profile_dz=profile_dz,
                nz=nz, out_path=out_path,
            )
            elapsed_times.append(elapsed)
            peak_rss_values.append(peak_rss)
            print(f"  realization {i+1}/{n_realizations}: {elapsed:.2f} s  peak RSS {_fmt_mem(peak_rss)}")

    mean_elapsed = np.mean(elapsed_times)
    std_elapsed = np.std(elapsed_times)
    mean_rss = np.mean(peak_rss_values)
    std_rss = np.std(peak_rss_values)
    print()
    print(f"wall clock : {mean_elapsed:.2f} s ± {std_elapsed:.2f} s")
    print(f"peak RSS   : {_fmt_mem(int(mean_rss))} ± {_fmt_mem(int(std_rss))}")


if __name__ == "__main__":
    main()
