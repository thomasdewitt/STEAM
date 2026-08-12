"""Throughput of the world-keyed generator against the stream it replaced.

Run:  uv run python tests/heavy/keyed_noise_throughput.py

The stream generator was heavily optimized (thread pool over 4 M-element
chunks, in-place float32 CMS, L3-aware; 5.60 -> 1.66 s at the production
finest class, see _extremal_levy's docstring), so "keyed noise is comparable"
is a claim that needs a measurement rather than an argument. Being
counter-based it has no sequential dependency at all, which is why it can
win outright: the stream had to split PCG64's state to parallelize the draw,
and the transform -- which both schemes share, unchanged -- is the same work
either way.

Sizes are the production finest class's draw count (1.8 GiB of float32
draws) and a couple of octaves above it.
"""

import time

import numpy as np

from steam.noise import class_key
from steam.simulate import (
    FLUX_ALPHA,
    NoiseRegion,
    _extremal_levy,
    _keyed_sparse_levy,
)


def _timed(call, repeats=3):
    """Best of `repeats`, after one warm-up (the jitted kernel compiles on
    first call, and the first touch of a fresh array pays page faults)."""
    call()
    best = float('inf')
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    key = class_key(np.random.SeedSequence(0))
    print(f"{'shape':>22} {'draws':>12} {'stream':>9} {'keyed':>9} "
          f"{'ratio':>7} {'GB':>6}")
    for shape in [(512, 512, 64),
                  (1024, 1024, 64),
                  (2048, 2048, 64),
                  (4096, 4096, 115)]:      # production square finest class
        size = shape[0] * shape[1] * shape[2]
        gigabytes = size * 4 / 1024**3

        stream = _timed(
            lambda: _extremal_levy(FLUX_ALPHA, size, np.random.default_rng(0)))
        region = NoiseRegion(key, (0, 0, 0), shape)
        keyed = _timed(
            lambda: _keyed_sparse_levy(shape, (1, 1, 1), FLUX_ALPHA, region))

        print(f"{str(shape):>22} {size:>12,} {stream:>8.2f}s {keyed:>8.2f}s "
              f"{stream / keyed:>6.2f}x {gigabytes:>5.1f}")

    # Sparse case: at s = 2 only 1/8 of cells are centers, and the keyed
    # generator evaluates the hash only at those, so it should track the
    # center count rather than the array size.
    print()
    shape = (2048, 2048, 64)
    for factors in [(1, 1, 1), (2, 2, 2), (2, 2, 1)]:
        region = NoiseRegion(key, (0, 0, 0), shape)
        elapsed = _timed(
            lambda: _keyed_sparse_levy(shape, factors, FLUX_ALPHA, region))
        centers = (len(range(0, shape[0], factors[0]))
                   * len(range(0, shape[1], factors[1]))
                   * len(range(0, shape[2], factors[2])))
        print(f"{str(shape):>22} s={factors} centers={centers:>12,} "
              f"{elapsed:>8.2f}s")


if __name__ == '__main__':
    main()
