# Flux-cascade memory benchmark

Date: 2026-07-15. Canonical model revision: `e74acec` (extremal-Levy flux
multiplier). The current run used seed 123, float32 fields, `FLUX_SCALE=0.5`,
`FLUX_ALPHA=1.8`, the default four flux substeps, uncompressed NetCDF output,
and CPU FFTs.

## Current canonical re-check

| Final grid | Flux substeps | Peak RSS | Simulation wall time |
|---|---:|---:|---:|
| 2048 x 2048 x 84 | 4 | 14.74 GiB | 48.99 s |

Peak RSS was 15,825,485,824 bytes (15,092.3 MiB). It was measured with
`resource.getrusage(RUSAGE_SELF).ru_maxrss` in a fresh worker process. Wall
time was measured around `simulate()`, including NetCDF output but excluding
environment startup. Torch tensors remained on CPU.

The grid used `dx=dy=500 m`, a 1024-km square horizontal domain, outer scale
512 km, spheroscale 100 m, and a 15-km-deep column. The exact command was:

`uv run --frozen --with netCDF4 --with numba python benchmarking/benchmark_simulate.py --worker --seed 123 --nx 2048 --ny 2048 --dx 500 --dy 500 --outer-scale 512000 --spheroscale 100 --domain-height 15000 --profile-dz 30 --nz 50 --output-path /tmp/steam_flux_memory_n4.nc`

Switching the flux innovation from additive Gaussian noise to the extremal-Levy
multiplier raised peak RSS by 10.4% (13.35 -> 14.74 GiB) and runtime by 25.2%
(39.12 -> 48.99 s) versus the Gaussian four-substep baseline. The extra runtime
is the Chambers-Mallows-Stuck generator's transcendentals (four per substep per
class); the extra memory is its few float32 buffers. Drawing the generator in
float32 with in-place buffer reuse is essential here -- a first float64
implementation peaked at 24.7 GiB on this grid.

## Historical memory optimization reference

The real-FFT and buffer-reuse work at `f956a98` was measured with one flux
substep:

| Final grid | Version | Peak RSS | Simulation wall time |
|---|---:|---:|---:|
| 256 x 256 x 84 | before | 884.95 MiB | 0.287 s |
| 256 x 256 x 84 | after | 861.95 MiB | 0.274 s |
| 2048 x 2048 x 84 | before | 15.15 GiB | 26.88 s |
| 2048 x 2048 x 84 | after | 12.93 GiB | 21.09 s |

That optimization changed the convolution to real `rfftn/irfftn`, reused the
soft-clamp and final-field buffers, and released convolution temporaries as
soon as possible. Fixed-seed fields remained numerically equivalent, and the
test suite passed.

The old two-point projection of 49.8 GiB for a 4096 x 4096 x 84 grid applies
to the one-substep model. It was not recomputed from this single canonical
re-check. The measured 3.3% increase suggests less headroom with four substeps,
so a 4096 run remains borderline on a 60-GiB workstation and should be monitored.
