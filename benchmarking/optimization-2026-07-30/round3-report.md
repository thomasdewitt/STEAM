# STEAM round-3 optimization — implementation report

2026-07-30 · `turbulon-model` @ `0c130d1` (round-2 committed) plus
`turbulon-analysis` @ `bfbd351` · **all changes uncommitted**.

Changed files:

- `turbulon-model`: `steam/simulate.py`, `steam/output.py`,
  `steam/thermodynamics.py`, `steam/constants.py`, new `tests/test_cuda_path.py`.
  (`steam.egg-info/*` was already dirty before this session.)
- `turbulon-analysis`: `run_production_squares.py` (one line + comment),
  `regen_production.sh` (the cgroup cap).

Runners: `./.venv/bin/python` for tests and micro-benchmarks,
`turbulon-analysis/.venv/bin/python` for production-shaped runs. HEAD reference
is `scratchpad/head3/` (a `git archive HEAD` copy of `steam/`), driven through
`scratchpad/run_head3.py`. Scratch in `/var/tmp/steam-audit/` (large files
deleted afterwards).

**Headline: the production square goes 182.0 s → 60.4 s (3.0x) at 17.49 →
16.61 GiB peak RSS, and the output is bit-identical — 57/57 variables,
including every class increment and every diagnostic.**

---

## Item 1 — class-increment staging off tmpfs

`simulate()` now stages next to the output file:

```python
increment_dir = Path(tempfile.mkdtemp(prefix="steam_class_inc_",
                                      dir=Path(output_path).parent))
```

| check | result |
|---|---|
| staging directory actually used (256² run, `verify_staging.py`) | `/var/tmp/steam-audit/steam_class_inc_vkawhe3l` — beside the output, off tmpfs |
| 256² 7-class output vs HEAD, all groups | **44/44 variables bit-identical** |

Recovers the full 6.5 GB of tmpfs that the mem-io report measured, and is the
prerequisite for the item-8 cap (tmpfs pages are charged to the cgroup and are
not reclaimable under cap pressure).

---

## Item 2 — fused `_gradient_components` (numba)

`_gradient_stencil`, `@njit(parallel=True, cache=True)`, one pass over the
field writing `grad_h` and `grad_z` — **two** full-size arrays where the four
`np.roll` copies plus `np.gradient` held three.

**It is bit-identical**, which needed matching np.gradient's z arithmetic
exactly rather than equivalently. Three details, all documented in the
docstring:

- float32 throughout, *not* float64 coefficients. `np.gradient` keeps the
  input dtype for an inexact input and the coefficients inherit float32 from
  `np.diff` of a float32 z; under NEP 50 the literal `2.` in numpy's uniform
  branch is weak and does not promote. (This contradicts the brief's
  "compute the z coefficients in float64 exactly as np.gradient does" — at
  numpy 2.4.4 that is not what np.gradient does.)
- the uniform-z branch is kept separate: `(f[k+1]-f[k-1])/(2 dz)` rounds
  differently from the general three-term form.
- endpoints are `edge_order=1` one-sided differences (np.gradient's default),
  not the second-order edges.

The gpu-torch report's 1.9e-9 discrepancy did not reappear: +,−,*,/,sqrt
compile to the same IEEE ops, and numba does not contract to FMA with
`fastmath=False`.

| shape | grad_h equal | grad_z equal | old → new |
|---|---|---|---|
| (2048, 2048, 115) production finest | **True** | **True** | 3.771 → 0.216 s (**17.5x**) |
| (4096, 140, 169) nest-like | **True** | **True** | 0.829 → 0.034 s (24.6x) |
| (256, 256, 115) | **True** | **True** | 0.052 → 0.003 s |
| (64, 64, 40), **uniform z** | **True** | **True** | — |
| (17, 13, 9) tiny odd | **True** | **True** | — |

**Memory, measured (not estimated) at the nest finest shape
(16384, 140, 364), one call, 3.11 GiB field:**

| | peak RSS | call |
|---|---:|---:|
| HEAD numpy | 10.389 GiB | 6.06 s |
| fused numba | **7.320 GiB** | **0.24 s** |

**−3.07 GiB**, against the mem-io report's projected −3.11 GiB. On a 29.3 GiB
nest peak that is ~10 %, i.e. ~10 % more nest cells for the same RAM.

**JIT cost:** compilation is lazy (at first call, not at import). Cold compile
0.55 s; with `cache=True` a warm process pays 0.11 s on first call and
0.0001 s thereafter. The cache lands in `steam/__pycache__/` beside the
existing `convolve_periodic_xy_zeropad_z` entries.

**No pyproject change was needed** — `numba>=0.57` has been a declared
dependency since `cc5e3e3` and `steam/utils.py` already imports it. Installed:
0.66.0 (model venv), 0.65.0 (analysis venv).

---

## Item 3 — drop the full-size `before` copy

`_bounded_amplitude_add` gains `record_applied=False`. When set, it leaves the
delta the field ACTUALLY took (`after - before`, not the solved `d` — the
float32 `+=` rounds) in `increment` itself, and `cascade_loop` saves that
array directly. The two per-scalar per-class `perturbation_field.copy()` calls
are gone.

- **CPU**: the difference is formed one level at a time
  (`perturbation_field[:, :, lev].copy()`, 16 MB at 2048²). The `a0 <= 0`
  early return became an `else` so there is a single exit.
- **CUDA**: after-minus-before is done on the device in `e` (which aliases
  `d`), the buffer the bisection has just finished with — the original
  perturbation is still on the host and goes back up. **No extra VRAM**;
  peak VRAM stays at round-2's 5.88 GiB.

| check | result |
|---|---|
| 256² 7-class, `device='cpu'`, all vars + all class increments vs HEAD | **44/44 bit-identical** |
| 256² 7-class, `device='cuda'`, same | **44/44 bit-identical** (better than the ~1e-6 statistical bar asked for — the arithmetic is unchanged, only where the subtraction happens moved) |
| production finest class (2048, 2048, 115) peak RSS, increment recorded | **6.638 → 4.917 GiB (−1.72 GiB)** |
| same, applied-delta mean / std | −7.298955884e-08 / 799.9813903, identical to 10 digits |
| same, CPU call time | 16.28 → 18.13 s (+1.85 s: the per-level copy and subtract) |

The +1.85 s is CPU-path only and only when increments are recorded; production
squares run the CUDA path (no cost there) and the nest never records
increments.

Not touched, and still a full-size copy: `flux_before = flux.copy()` in the
same loop, for the flux increment. Removing it means changing `_advance_flux`'s
contract, which was outside this brief.

---

## Item 4 — blosc_zstd complevel 1

`steam/constants.py` gains `output_compression = "blosc_zstd"`,
`output_complevel = 1`, and `output_compression_min_chunk_bytes = 16 * 1024`
(see the caveat below). All six write sites go through one new function,
`steam.output.compression_kwargs(compress, chunksizes)` — a helper rather than
six inline copies because it now encodes a *rule*, not just a kwarg triple.
`shuffle=True` is never used (silently ignored for non-zlib); the shuffle is
`blosc_shuffle=1`.

Measured through a real production h field (2048 × 2048 × 115, 1.93 GB, from
`runs/steam_sq10_icon_lem_m01.nc`, chunks (64, 64, 115)):

| filter | write | read | on disk | read-back |
|---|---:|---:|---:|---|
| zlib-4 (former shipped) | 20.4 s | 3.4 s | 1.54 GB | exact |
| **blosc_zstd c1** | **1.7 s** | **1.0 s** | **1.28 GB** | **exact** |

**12x faster to write and, on this field, 17 % SMALLER** (the mem-io report's
+5 % was across h+qt+flux on a different member; flux is the field that grows).
System `/usr/bin/ncdump`, entirely outside the venv, reads both the header and
the data.

### ⚠ One finding that needed a decision — blosc fails outright on tiny chunks

The first full 256² run with the new filter **crashed**:

```
RuntimeError: NetCDF: HDF error
Blosc_Filter Error: blosc_filter: Buffer is uncompressible.
```

in `write_class_increments`. blosc's HDF5 filter does not degrade to raw
storage: given a chunk it cannot shrink and no room for its 16-byte header, it
fails the write. Measured threshold on incompressible float32, whole-array
chunk: **fails at ≤ 1024 bytes, succeeds from 1536**. Class 0 of a production
square is 4 × 4 × 12 = **768 bytes**, so this would have taken down every
production square. (zlib and plain `zstd` both handle these fine — it is
specific to blosc.)

I resolved it with a documented, deterministic rule rather than a fallback:
chunks below `output_compression_min_chunk_bytes` (16 KiB, a wide margin over
the measured cliff) are written raw. On the 256² check that is classes 0 and 1
only — 5 kB of a 158 MB file — and a chunk that small is a rounding error
beside the ~11 kB of HDF5 metadata the variable carries anyway. **Flagging it
for Thomas: this is a behaviour rule I chose, not one that was specified.**

### Item 4b — compress the nest (Thomas's ruling, mid-task)

`run_nest` in `turbulon-analysis/run_production_squares.py` now passes
`compress=True` to `refine()`, matching the square. Verified on a real nest
group: written `blosc_zstd c1` (filters confirmed), read back **bit-identical
to the HEAD nest written zlib-4**, and readable by system `ncdump`. Nothing
else in that file was touched.

---

## Item 5 — continuously-fed diagnostics pool

`compute_diagnostics`' batch-synchronous loop is replaced by the fed pool:
`n_workers` chunks in flight, `FIRST_COMPLETED` → write that result and
immediately read + submit the next. Reads and writes stay on the main thread.
The worker cap stays at `min(8, ...)`.

Real square field (2048² × 115 h/qt), same base file for all three:

| variant | wall | peak RSS | file |
|---|---:|---:|---:|
| HEAD (batch, zlib-4) | 60.5 s | 8.39 GiB | 4.96 GB |
| **fed + blosc_zstd c1** | **25.5 s** | 11.10 GiB | 5.35 GB |
| fed + zlib-4 (isolating the loop) | 57.4 s | 11.96 GiB | 4.96 GB |

**blake2b digests of T, qv, qc, qi and p are identical across all three** —
bit-identical, as the docstring guarantees.

This reproduces the mem-io report's key finding: the fed loop alone buys almost
nothing (60.5 → 57.4 s); **the filter is the fix**. Note HEAD measures 60.5 s
here rather than the report's 86.7 s because my base file's h/qt are
blosc-compressed, so the reads are faster for every variant.

Peak RSS rises 8.39 → 11.10 GiB (in-flight chunks now overlap writes). It is
below both binding peaks and the square's diagnostics can only ever coexist
with the nest's 31.5 GB, i.e. ~43 GB against the 50 GB cap. If that margin is
disliked, `n_workers=6` was measured at 24.7 s / 8.4 GB by the mem-io agent —
one constant, no restructuring.

---

## Item 6 — hyperslab parent read in `refine()`

`refine()` no longer reads whole parent h/qt/flux. `parent_nx/ny` come from the
dimensions; the file is reopened after the halo widths are known and the pads
are read as contiguous hyperslabs by `_read_parent_slab`, using
`_contiguous_runs` to split each index array (which can wrap modulo the axis
length on a periodic parent) into one or two ranges.

| check | result |
|---|---|
| small parent + strip nest (r0, no wrap) + wrapping strip nest (r1, y from 0 with pad → two runs), HEAD vs new, all groups | **76/76 variables bit-identical** |
| production prologue peak RSS (`probe_nest_prologue.py`, real m00 parent) | **8.59 → 5.36 GB (−3.23 GB)** |
| production prologue wall | 28.8 → 22.0 s (−6.8 s) |

Less than the mem-io report's projected −7.5 GB: with the parent read gone, the
remaining 5.36 GB prologue peak is the interpolation-compensation replay
(whole class-increment reads and their zooms), which is untouched — cropping
those before zooming is realization-changing (audit §5.5).

---

## Item 7 — CUDA-path test

`tests/test_cuda_path.py`, one test, skipped without a CUDA device. Runs the
same tiny `simulate()` (64², 4 classes, seed 17) on cpu and on cuda and
compares per-level profiles of h and qt, plus bounds.

Margins at the 1e-3 gate: h level-mean 9.3e-08, h level-std **3.8e-05**,
qt level-mean 2.6e-07, qt level-std 2.2e-07 — the gate has ~26x headroom on
the tightest quantity and the two paths do genuinely differ (h level-std
68.89867 cpu vs 68.89834 cuda), so it is not vacuous. Bounds are asserted
exactly on both paths. **1.9 s.**

Level spreads are compared only where the level has spread
(`cpu_std > 1e-3 * max`): qt's topmost levels are dry to ~1e-10 kg/kg and
their relative comparison is meaningless — this is the round-2 report's
"qt level-std max rel = 1" observation, handled rather than tripped over.

---

## Item 8 — co-residency cap

`regen_production.sh` steps [1/3] and [2/3] now run under

```
systemd-run --user --scope -p MemoryMax=50G -p MemorySwapMax=0 --same-dir
```

Verified on this box before use: the flags are accepted, `memory.max` inside
the scope reads **53687091200** (= 50 GiB), and `--same-dir` preserves the
working directory. The comment in the script records why it is continuous
enforcement rather than a preflight (both processes ramp gradually and each
passes its own start-of-run check while the other has barely allocated) and
why an OOM kill is acceptable (both halves of the driver skip completed work
on restart). `run_production_squares.py` logic was not touched apart from item
4b's one line.

---

## Final gate

### Test suite

```
./.venv/bin/python -m pytest tests/ -q   →   144 passed in 9.27 s
```

(143 before, plus the new CUDA test. Nothing in `tests/` was edited; the file
was added.)

### 256² end-to-end bit-identity, combined tree, `device='cpu'`

**44/44 variables bit-identical** to HEAD, including all 7 classes' h/qt/flux
increments.

### Production square, `scratchpad/run_square.py` → `/var/tmp/steam-audit/`

2048², L = 1536 km, ℓ_s = 10 m, 9 classes, `device='cuda'`, seed 2000,
`save_class_increments=True`, then `compute_diagnostics`. GPU idle at 528 MiB
before each run.

| | round-2 (HEAD, re-measured) | round 3 | change |
|---|--:|--:|--:|
| `simulate()` | 125.2 s | **35.9 s** | −89.3 s |
| `compute_diagnostics()` | 56.8 s | **23.7 s** | −33.1 s |
| **square total** | **182.0 s** | **60.4 s** | **−121.6 s (3.0x)** |
| peak RSS | 17.49 GiB | **16.61 GiB** | −0.88 GiB |
| peak VRAM | 5.88 GiB | 5.88 GiB | — |
| tmpfs (RAM) for staging | 6.5 GB | **0** | −6.5 GB |
| output size | 10.86 GB | 11.90 GB | +9.6 % |

HEAD reproduced round-2's published 182.0 s / 17.45 GiB / 5.88 GiB exactly, so
the comparison is like-for-like on the same box in the same session.

Most of `simulate()`'s −89 s is the write phase (zlib-4's 94.5 s → ~10 s); the
gradient stencil accounts for a few seconds of it.

### Production-scale output identity

```
cmp_nc_all.py prod_head.nc prod_new.nc   →   57/57 variables bit-identical
```

All 9 classes' h/qt/flux increments, h/qt/flux, and all five diagnostics
(T, qv, qc, qi, p), at 2048² × 115 on the CUDA path. **Nothing about the
realization moved.**

---

## Open items for Thomas

1. **The 16 KiB compression floor (item 4).** blosc genuinely cannot write
   chunks that small; I chose "write them raw, documented" over "use a filter
   that survives everything but compresses worse". Worth your blessing.
2. **Diagnostics peak RSS 8.4 → 11.1 GiB** for the 60.5 → 25.5 s. Non-binding,
   but `n_workers=6` gives back 2.7 GB for ~1 s if you would rather.
3. **The nest is now compressed** (item 4b): ~+6 s per nest, −4.15 GB per
   member on a filesystem at 94 %. Every member written from now on will carry
   blosc-filtered nest variables; readers need the HDF5 plugin (this box's
   system netCDF has it).
4. **`flux_before = flux.copy()`** in `cascade_loop` is still a full-size copy
   per class when increments are recorded — the same 1.8 GiB pattern item 3
   removed for h and qt, left alone because it needs `_advance_flux`'s contract
   to change.

## Reproduction

```
scratchpad/verify_staging.py OUT.nc          # item 1
scratchpad/verify_gradient.py {identity|mem_old|mem_new|jit}   # item 2
scratchpad/verify_bounded_mem.py {old|new}   # item 3
scratchpad/verify_compress.py                # item 4
scratchpad/verify_diag.py {base|head|new [zlib4]}              # item 5
scratchpad/probe_nest_prologue2.py           # item 6 (STEAM_REPO=... selects the tree)
scratchpad/e2e_square.py / e2e_square_cuda.py / e2e_nest.py / e2e_nest_wrap.py
scratchpad/cmp_nc_all.py A.nc B.nc           # group-walking comparison
scratchpad/run_head3.py SCRIPT.py ...        # run any of the above against head3/
```
