# Flux-cascade memory benchmark

Date: 2026-07-15. Canonical model revision: `439f735`. The current run used
seed 123, float32 fields, `FLUX_SCALE=0.5`, the default four flux substeps,
uncompressed NetCDF output, and CPU FFTs.

## Current canonical re-check

| Final grid | Flux substeps | Peak RSS | Simulation wall time |
|---|---:|---:|---:|
| 2048 x 2048 x 84 | 4 | 13.35 GiB | 39.12 s |

Peak RSS was 14,339,584,000 bytes (13,675.3 MiB). It was measured with
`resource.getrusage(RUSAGE_SELF).ru_maxrss` in a fresh worker process. Wall
time was measured around `simulate()`, including NetCDF output but excluding
environment startup. Torch tensors remained on CPU.

The grid used `dx=dy=500 m`, a 1024-km square horizontal domain, outer scale
512 km, spheroscale 100 m, and a 15-km-deep column. The exact command was:

`uv run --frozen --with netCDF4 --with numba python benchmarking/benchmark_simulate.py --worker --seed 123 --nx 2048 --ny 2048 --dx 500 --dy 500 --outer-scale 512000 --spheroscale 100 --domain-height 15000 --profile-dz 30 --nz 50 --output-path /tmp/steam_flux_memory_n4.nc`

The previous one-substep result on the same grid was 12.93 GiB and 21.09 s.
Four substeps therefore raised measured peak RSS by 3.3% and runtime by 85.5%.
The small memory change is expected because substeps reuse the same flux and
innovation buffers; the runtime increase comes from three additional flux
convolutions per size class.

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
