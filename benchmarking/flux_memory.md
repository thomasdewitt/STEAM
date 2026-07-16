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
re-check. (Framing correction, 2026-07-15: the GIGALES target is 2048 x 2048
horizontal, not 4096 x 4096. At 2048 x 2048 the horizontal plane is not the
problem — the vertical level count is; see the GPU section below.)

## GPU path and the host floor (2026-07-15, revision `46552d6`)

`convolve_fft_xy_oa_z(..., device='cuda')` moves the FFT working set to the
RTX 5080 (16 GiB). The overlap-add z-block is sized adaptively from free VRAM:
single block when the padded array fits, blocking only when the shape demands
it. Measured per convolution (float32, 21^3 kernel, fixed seed, GPU-vs-CPU
max relative error ~1e-6):

| Field shape | CPU | GPU | Speedup | GPU blocks | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| 512 x 512 x 168 | 0.17 s | 0.07 s | 2.5x | 1 (n_fft=256) | small |
| 1024 x 1024 x 247 | 1.02 s | 0.36 s | 2.9x | 1 (n_fft=512) | ~8 GiB |
| 2048 x 1024 x 342 | 3.00 s | 0.87 s | 3.5x | 2 (n_fft=256) | 13.55 GiB |
| 2048 x 2048 x 363 | — | 1.97 s | — | 31 (n_fft=32) | 13.75 GiB |

Full TWPICE-profile cascade at 2048 x 1024 x 342 (10 dyadic classes, four flux
substeps plus two scalar convolutions per class, seed 20260714, compressed
output): simulate 145 s, thermodynamic diagnostics 112 s, 4:20 wall including
startup. Peak host RSS 27.3 GiB, peak VRAM ~13.8 GiB.

**The binding constraint is host RAM, not VRAM, and it scales with vertical
levels.** The GPU removes only the FFT spectral working set. What it cannot
remove is the host floor: up to eight concurrent field-sized float32 arrays
at the finest grid (persistent h/qt/flux plus the flux-advance or
gradient-weighting transients — see the preflight guard in simulate.py),
measured at ~10 field-equivalents of RSS once the CUDA host context and
compressed-NetCDF buffering are included (27.3 GiB / 2.67 GiB per field).

Max vertical levels on this 60-GiB host (~50 GiB available, 2 GiB headroom,
~10 field-equivalents at 2048 x 2048 -> 160 MiB per level): **~300 levels at
2048 x 2048**, ~600 at 2048 x 1024. VRAM would cap at ~370 and ~780 levels
respectively (minimal 32-tap blocks), so the host binds first in both cases.
Under the current spheroscale spec (log-linear 1000 m -> 10 m over 20 km,
isotropic below the spheroscale) a 20-km column demands 342 levels — over
the 2048 x 2048 budget, which is why the production first-look run is
2048 x 1024 (outer scale 102.4 km). Big runs go under
`systemd-run --user --scope -p MemoryMax=...` so the kernel can only ever
kill the run, never the session; the 2026-07-15 session was lost to exactly
this failure mode before the scoping was adopted.
