# Flux-cascade memory benchmark

Date: 2026-07-15. Physics baseline: `730aa1b`. Both runs used seed 123,
float32 model fields, the Gaussian unit-mean flux cascade, uncompressed NetCDF
output, and the same CPU-only execution path. The optimized code is the memory
commit containing this note.

## Outcome

The largest measured case, `2048 x 2048 x 84`, fell from 15.15 GiB to
12.93 GiB peak RSS (-14.7%) and from 26.88 s to 21.09 s (-21.5%). The small
reference case improved less because Python, Torch, and FFT-library startup
memory dominate a 256-grid run.

| Final grid | Version | Peak RSS | Simulation wall time | Memory change | Time change |
|---|---:|---:|---:|---:|---:|
| 256 x 256 x 84 | before | 884.95 MiB | 0.287 s | -- | -- |
| 256 x 256 x 84 | after | 861.95 MiB | 0.274 s | -2.6% | -4.5% |
| 2048 x 2048 x 84 | before | 15.15 GiB | 26.88 s | -- | -- |
| 2048 x 2048 x 84 | after | 12.93 GiB | 21.09 s | -14.7% | -21.5% |

Peak RSS is `resource.getrusage(RUSAGE_SELF).ru_maxrss` from a fresh worker
process. Wall time is measured inside `simulate()`, so environment startup is
excluded. `/usr/bin/time -v` independently reported the same RSS peaks to
rounding. Torch tensors remained on CPU; CUDA allocation was zero, so GPU memory
is not applicable even though the host has an RTX 5080.

The measured configurations were:

- Small: `nx=ny=256`, `dx=dy=500 m`, outer scale 128 km, 15 km deep.
- Large: `nx=ny=2048`, `dx=dy=500 m`, outer scale 512 km (two outer-scale
  tiles across the 1024-km domain), 15 km deep.

The worker command was `uv run --frozen --with netCDF4 --with numba python
benchmarking/benchmark_simulate.py --worker ...`; the two configurations above
supplied the numeric arguments and wrote temporary NetCDF files under `/tmp`.

## Changes

1. The convolution now transforms each real overlap-add block directly with
   `torch.fft.rfftn/irfftn`. The previous path retained a complex horizontal
   `rfft2` field, performed complex FFTs along z, and accumulated complex
   half-plane output. The new accumulator and inverse output are real.
2. Scalar gradient weighting is computed before soft clipping, after which the
   no-longer-needed running-field buffer is reused for the soft-clip weight.
   This removes one full-domain live array at the gradient peak.
3. Final h and qt means are added into their perturbation buffers in place,
   instead of allocating two additional full-domain result arrays.
4. The convolved flux increment and FFT block buffers are deleted as soon as
   their consumers finish. No size-class history is retained; the existing
   cascade already interpolates only the current coarsened fields to the next
   grid.

No dtype or numerical-precision setting changed. STEAM already uses explicit
float32 working fields, so no new float32 opt-in mode was needed.

## Fixed-seed correctness gate

The pre/post files for the small configuration used identical seed 123 and
inputs. Coordinates, scale arrays, and `C_h_k`/`C_qt_k` were bit-identical.
The real-FFT operation order changes float32 rounding, so 3-D fields are not
bit-identical, but they are numerically equivalent:

| Field | Maximum absolute difference | RMSE | RMSE / field-fluctuation std | Correlation |
|---|---:|---:|---:|---:|
| h | 2.90625 J kg^-1 | 0.01678 J kg^-1 | 6.50e-6 | 0.999999999979 |
| qt | 6.63e-7 kg kg^-1 | 1.82e-9 kg kg^-1 | 8.64e-7 | 1.000000000000 |
| F | 3.05e-5 | 7.17e-7 | 4.39e-7 | 1.000000000000 |

The full test suite also passes (`109 passed, 60 skipped`). Convolution tests
compare the new implementation against the mixed-boundary ndimage reference at
`rtol=atol=1e-4`; direct old/new convolution comparisons were within about
`3e-7` relative error.

## Feasible-domain projection

For the same 84-level vertical grid and outer-scale schedule, fitting
`RSS = b + a*N^2` through the measured 256 and 2048 cases projects:

| Horizontal grid | Before | After |
|---|---:|---:|
| 4096 x 4096 x 84 | 58.7 GiB | 49.8 GiB |

The workstation has 60 GiB physical RAM and about 52 GiB available during the
benchmark. Before the change, the next dyadic 4096 grid had effectively no
headroom for the OS and was not a safe run; 2048 was the largest conservative
case. After the change, 4096 is projected to fit with roughly 10 GiB relative to
total RAM (about 2 GiB at the observed concurrent system load). For a production
matched-domain run, stop other memory-heavy jobs and monitor RSS, because 4096
is still borderline rather than comfortable. With an 80%-of-RAM safety budget
(48 GiB), the projected limit is about `4020 x 4020 x 84`; with the measured
52-GiB available budget it is about `4190 x 4190 x 84`.

These projections scale only horizontal area. A GIGALES match with more than 84
vertical levels, larger sparsity factors, or a different finest vertical scale
must be reduced by approximately the square root of the vertical-size ratio.
