# STEAM performance audit — full 2048² production configuration
2026-07-30 · Ryzen 9950X (16c/32t), 60 GB RAM, RTX 5080 16 GB (sm_120), driver 595.80
Model `turbulon-model@bef0364`, driver `turbulon-analysis/run_production_squares.py`, unmodified.
All runs and outputs in `/tmp/claude-1000/.../scratchpad` and `/var/tmp/steam-audit`; nothing written to `turbulon-analysis/runs/`.

## 0. Headline

| | wall | peak RSS | GPU |
|---|---|---|---|
| one production square (`simulate` + `compute_diagnostics`) | **508 s** (8 m 28 s) | 18.0 GB | 12.8 GB, busy 0.55 % of the time |
| its strip nest (`refine`, CPU) | **657 s** (10 m 57 s) | 31.5 GB | none |
| per member, pipelined | ~657 s (the nest is the critical path) | | |

**One function is 75 % of the entire pipeline.** `_bounded_amplitude_add`
(simulate.py:1840) accounts for 63.5 % of a square and 84.1 % of a nest —
875 s of the 1166 s of work per member. It is single-threaded and spends
that time walking **strided** `[:, :, lev]` slices of a C-ordered
`(nx, ny, nz)` array, i.e. touching one useful float per 460-byte stride.

Two changes, both **verified bit-identical**, give a measured 3.5× (copy the
level to a contiguous buffer) and 6.2× (that, plus 8 threads over the
independent levels). That is a 54–63 % cut in total pipeline wall time with
zero effect on the realization.

The second headline: **the GPU is essentially unused.** All 27 convolutions of
a square take 2.79 s of the 508 s. The driver comment "GPU ~3 min each" no
longer describes this code — the run is 8.5 minutes at 1.00 CPU core.

---

## 1. Square: wall-clock breakdown

`nx=ny=2048`, `dx=dy=3000 m`, `L=1536 km`, `l_s=10 m` constant,
`piecewise_isotropic_below_spheroscale`, `H_h=0.45`, `domain_height=20 km`,
`profile_dz=50 m`, seed 2000, `FLUX_SCALE=(0.05/1.681)^(1/1.8)`,
`compress=True`, `device="cuda"`, `save_class_increments=True`.
Output 10.9 GB (+ 9.2 GB after the nest = 20.0 GB, matching production).

### 1.1 Phases

| phase | s | % of 508 |
|---|---:|---:|
| grid setup, envelope, normalization, compensation, bound buffers | 0.05 | 0.0 |
| **`cascade_loop` (9 size classes)** | **355.9** | **70.0** |
| `write_netcdf` — h, qt, flux, zlib-4 | 37.0 | 7.3 |
| `write_class_increments` — 27 arrays, zlib-4 | 57.5 | 11.3 |
| `compute_diagnostics` — saturation adjustment + Newton | 57.0 | 11.2 |

The whole pre-cascade construction — `_compute_all_grids`,
`_turbulon_envelope`, `_compute_normalization` (the per-level Haar loop),
`_compensation_profiles`, `_bound_buffers` — is **50 milliseconds**. All of it
is 1-D work on ≤ 401-element profiles. It is not worth a single line of
optimization.

### 1.2 Per size class (cascade only)

L = 1536 km down to 2dx = 6 km, 9 dyadic classes. Working grids grow to the
output grid; the class pyramid holds 581 M cells, of which the finest class is
482 M (83 %).

| # | k (m) | grid | Mcell | wall (s) | % of cascade |
|--:|--:|---|--:|--:|--:|
| 1 | 1 536 000 | 8×8×6 | 0.0004 | 0.09 | 0.0 |
| 2 | 768 000 | 16×16×8 | 0.002 | 0.01 | 0.0 |
| 3 | 384 000 | 32×32×12 | 0.012 | 0.01 | 0.0 |
| 4 | 192 000 | 64×64×17 | 0.07 | 0.02 | 0.0 |
| 5 | 96 000 | 128×128×25 | 0.41 | 0.10 | 0.0 |
| 6 | 48 000 | 256×256×37 | 2.4 | 0.55 | 0.2 |
| 7 | 24 000 | 512×512×53 | 13.9 | 3.99 | 1.1 |
| 8 | 12 000 | 1024×1024×78 | 81.8 | 44.56 | 12.5 |
| 9 | **6 000** | **2048×2048×115** | **482.3** | **306.52** | **86.1** |

The finest class is **86 %** of the cascade and **60 %** of the whole square.
The last octave costs **6.9×** the previous one while carrying only 5.9× the
cells — the extra 17 % is the strided-slice cache penalty in
`_bounded_amplitude_add` growing with `nz`. The first six classes together are
0.2 % — cheaper than a single `print`.

### 1.3 Per operation inside the finest class (306.5 s)

| operation | s | % of class |
|---|--:|--:|
| `_bounded_amplitude_add` (×2: h, qt) | **278.83** | **91.0** |
| `_advance_flux` total (×1) | 9.32 | 3.0 |
|   · `_sparse_levy`/`_extremal_levy` — the extremal-Lévy draw | 5.41 | 1.8 |
|   · `expm1`, ⟨\|·\|⟩ norm, `noise*flux`, clip, volume renorm | 3.14 | 1.0 |
|   · `CONVOLVE` flux increment (**cuda**) | 0.77 | 0.3 |
| `_gradient_components` (×2) | 7.47 | 2.4 |
| in-loop array algebra (`running_sum`, `W*=S_k`, level norm, `W*=C_k`, `W*=f`) | 5.4 | 1.8 |
| `zoom_trilinear` (×3: h, qt, flux) | 1.47 | 0.5 |
| `CONVOLVE` scalar amplitude → increment (**cuda**, ×2) | 1.46 | 0.5 |
| `_bound_taper` (×2) | 1.41 | 0.5 |
| `np.save` of class increments to /tmp (×3) | 1.17 | 0.4 |

### 1.4 Whole-square operation totals

| operation | s | % of 508 |
|---|--:|--:|
| `_bounded_amplitude_add` | **322.67** | **63.5** |
| `recover_diagnostics` (133.4 CPU-s over 8 workers → 57.0 s wall) | 57.0 | 11.2 |
| `write_class_increments` | 57.47 | 11.3 |
| `write_netcdf` | 37.04 | 7.3 |
| `_advance_flux` (of which Lévy 6.48, convolve 0.98) | 11.16 | 2.2 |
| `_gradient_components` | 8.94 | 1.8 |
| in-loop array algebra | ~6.2 | 1.2 |
| **all 27 convolutions (GPU)** | **2.79** | **0.55** |
| `zoom_trilinear` (24 calls) | 1.76 | 0.35 |
| `_bound_taper` | 1.66 | 0.33 |
| `.npy` staging of increments | 1.38 | 0.27 |
| grids + normalization + envelope + compensation | 0.05 | 0.01 |

---

## 2. Strip nest `refine()`: wall-clock breakdown

`refine(parent, 0, 2048, 1016, 1032, 375, 375, device="cpu")` — spans x,
48 km in y, refined to dx = 375 m.

**It has three classes, not many.** The nest starts one dyadic step below the
parent's finest (6 km) and stops at 2·dx = 750 m: k = 3000, 1500, 750 m. But
its finest working grid is **835 M cells — 1.7× the square's finest** — because
x spans the full 6144 km at 375 m (16384 cells) and the y halo adds 12 cells to
the 128 kept. Total nest pyramid 1.02 G cells vs the square's 581 M, on CPU.

| phase | s | % of 657 |
|---|--:|--:|
| parent netCDF read (h, qt, flux — 5.8 GB, zlib-inflate, single-threaded) | ~17.0 | 2.6 |
| interpolation-compensation replay: 84 `zoom_trilinear` up the parent chain | 12.6 | 1.9 |
| `_project_onto_bounds` ×2 after the replay | 0.43 | 0.1 |
| class 1 — k=3000 m, 4096×44×169 (30.5 Mcell) | 7.19 | 1.1 |
| class 2 — k=1500 m, 8192×76×248 (154.4 Mcell) | 81.25 | 12.4 |
| **class 3 — k=750 m, 16384×140×364 (834.9 Mcell)** | **533.80** | **81.2** |
| `write_netcdf` into `refinements/r0` | 2.71 | 0.4 |

| operation (whole nest) | s | % of 657 |
|---|--:|--:|
| `_bounded_amplitude_add` | **552.73** | **84.1** |
| `_advance_flux` (Lévy 11.60, convolve 7.26) | 25.81 | 3.9 |
| `zoom_trilinear` (93 calls; 84 are the replay) | 15.88 | 2.4 |
| `_gradient_components` | 15.19 | 2.3 |
| `CONVOLVE` scalar (**cpu**, ×6) | 14.21 | 2.2 |
| `_bound_taper` | 2.91 | 0.4 |
| `write_netcdf` | 2.71 | 0.4 |

Two incidental findings:

* **The nest is written uncompressed.** `run_nest` never passes `compress=`, so
  `refine` falls back to `constants.output_compress = False` while the square
  is written with `compress=True`. That is why the nest write is 2.7 s for
  9.2 GB (page-cache buffered) and why `steam_sq10_icon_lem_m00.nc` is 38 GB
  against 20 GB for the nestless members. Probably unintentional asymmetry;
  it is *fast*, so the question is only whether you want the disk back.
* The nest's y grid is padded to 140 cells and 128 are kept — 9 % of the
  finest-class work is halo that is discarded.

---

## 3. Memory and transfer

| | peak RSS | notes |
|---|---:|---|
| square, cascade (finest class) | **18.0 GB** | `VmHWM` 17.8 GB; matches the 8-field preflight estimate (8 × 1.93 GB = 15.4 GB) plus allocator slack |
| square, diagnostics | 9.2 GB | 8 chunk pairs of (128, 2048, 115) in flight |
| **square, class-increment staging** | **+6.5 GB of tmpfs** | `tempfile.mkdtemp()` lands in `/tmp`, which is tmpfs on this box — the 27 `.npy` files are RAM, not disk, and do **not** appear in RSS |
| nest | **31.5 GB** | `VmHWM` 30.8 GB; peak during the increment replay, where several full parent-grid `inc` arrays (1.93 GB each) are live at once |

**GPU.** Peak `torch` allocation 12.75 GB, reserved 13.46 GB, at the finest
class — `cuda_block_fft_size` deliberately grows `n_fft` to 128 so the whole
padded array is one overlap-add block. That is 80 % of the 16 GB card for a
0.55 % share of the wall clock.

**Transfer churn.** `convolve_fft_xy_oa_z` does one `.to('cuda')` and one
`.cpu()` per call — no per-block streaming, which is right. Over the run that
is ~14 GB of round-trip PCIe traffic. At the finest class each call moves
3.86 GB in 0.73 s, i.e. the GPU convolution is **transfer-dominated**: the
FFTs themselves are a small fraction. Each cuda call also issues
`torch.cuda.empty_cache()`, a synchronizing driver-level free, before sizing
the block; harmless here but it serializes.

**⚠ Co-residency hazard.** Production pipelines the nest (a spawned worker,
31.5 GB) behind the next square (18.0 GB + 6.5 GB tmpfs). Peak concurrent
demand is ≈ **56 GB on a 60 GB box**. That is the configuration that will take
the login session, and it is not guarded: `simulate()` has a preflight
`available_memory_bytes()` check, but `refine()` has none, and neither knows
about the other process. Worth a `MemoryMax=` scope on the nest pool or a
lock, independent of any speed work.

---

## 4. CPU utilization character

* **`cascade_loop` runs at exactly 1.00 core for its entire 356 s** (square)
  and its entire 622 s (nest). Sampled every 0.25 s; no excursion above 1.01
  except the `zoom_trilinear` calls, where torch's 16 intra-op threads briefly
  spike. Everything expensive is single-threaded numpy on ≥ 2 GB arrays.
* **Diagnostics parallelization is ineffective**: 133.4 CPU-seconds of
  `recover_diagnostics` completed in 57.0 s wall — **2.3 effective cores out of
  8 requested, out of 32 available**. The sampler shows a clean square wave
  alternating 1.00 ↔ 5.7 cores. Cause is the batch-synchronous structure at
  thermodynamics.py:195: the main thread reads (and zlib-inflates) all 8 chunks
  of a batch serially, the pool computes, then the main thread writes (and
  zlib-deflates) all 8 serially, and only then does the next batch start. Both
  serial legs are 1 core. `n_workers` is additionally capped at
  `min(8, ...)` — 8 of 32 threads.
* **Nothing is IO-bound.** `/usr/bin/time -v` reports 0 filesystem inputs for
  the square (all page cache) and 21.2 GB of outputs that writeback absorbs
  asynchronously. The write phases sit at 100 % of one core inside zlib:
  `write_netcdf` compresses 5.8 GB of payload in 37 s (156 MB/s) and
  `write_class_increments` 7.0 GB in 57.5 s (121 MB/s) — textbook
  single-threaded zlib-4 rates. The NVMe is idle.

---

## 5. Ranked bottlenecks

Percentages are of the 1166 s of work per production member (square 508 +
nest 657).

### 1. `_bounded_amplitude_add` — 875 s, **75.1 %**

*Why slow:* memory-bound and single-threaded, and pessimally laid out. It
loops `for lev in range(nz)` over the **last** axis of a C-ordered
`(nx, ny, nz)` array, so every `perturbation_field[:, :, lev]`,
`increment[:, :, lev]` and the write-back touches 4.2 M elements at a stride of
`nz*4` = 460 bytes (square) / 1456 bytes (nest). Each 64-byte cache line
delivers 4 useful bytes. On top of that the inner bisection runs a **fixed 60
iterations**, each a subtract + clip + float64 mean over that strided level —
measured 0.179 s per level per 60-iteration solve at 2048², i.e. 41 s of the
99 s a single call takes.

*Fix (measured, at (2048, 2048, 115), `bench_bounded.py`):*

| variant | s | speedup | bit-identical |
|---|--:|--:|:--:|
| shipped | 99.41 | 1.00× | — |
| `np.ascontiguousarray` the level, solve, write back | 28.48 | **3.49×** | **yes** |
| + early exit when `float32(mu_lo) == float32(mu_hi)` | 28.74 | 3.46× | yes |
| + 8 threads over levels | 16.11 | **6.17×** | **yes** |
| + 16 threads over levels | 17.03 | 5.84× | yes |

*Reproducibility:* **exactly invariant.** numpy's pairwise reduction follows
index order, not memory order, so `strided.mean(dtype=np.float64)` and
`contiguous.mean(dtype=np.float64)` return the same bits (verified directly);
every other operation is elementwise. Threading is over `lev`, and levels touch
disjoint slices of `perturbation_field` and `increment` with no shared
reduction. `np.array_equal` against the shipped result was True for all four
variants above. This is the rare case where the largest win carries no
realization risk at all.

*Payoff:* 875 → 250 s (contiguity alone) or 875 → 142 s (threaded). Total
pipeline **1166 → 541 s per member (−54 %)** or **→ 433 s (−63 %)**.

*Note:* the early-exit is provably a no-op for the last ~32 iterations
(`float32(mu)` freezes after 28 bisection steps on a bracket of
2·(h_max−h_min)), but it measured neutral once the data is contiguous — the
rescale loop, not the bisection, dominates then. Keep it as tidiness, not as an
optimization.

*Risk:* low. The only subtlety is that the shipped code mutates `increment`
in place and the caller relies on that (`del increment` immediately after), so a
contiguous-copy version must write `d` back into `increment[:, :, lev]` if you
want to preserve that contract — or drop the contract, since nothing reads it.

### 2. Compressed NetCDF writes on the square — 94.5 s, **8.1 %**

`write_netcdf` 37.0 s + `write_class_increments` 57.5 s, both 100 % of one core
inside zlib complevel 4.

*Fixes, in increasing order of intrusiveness:*
- `complevel=1` typically gives ~2.5× the throughput for ~10 % more bytes →
  ~55 s saved. **Bit-identical data** (compression is lossless; only the file
  bytes change).
- Write class increments straight from `cascade_loop` into the open dataset
  instead of the `.npy` → tmpfs → `np.load` → netCDF round trip. Saves the
  1.4 s of staging, a full re-read, and **6.5 GB of RAM** (see §3).
- If the netCDF4/HDF5 build has zstd or blosc filters, either is 3–5× zlib at
  similar ratio. Check `nc-config --has-zstd`.

*Risk:* none to the data.

### 3. `compute_diagnostics` — 57 s, **4.9 %**

2.3 of 32 cores. The Newton solve itself is only 9.4 s of the 133 CPU-s;
the rest is the saturation-adjustment arithmetic in `recover_diagnostics`.

*Fix:* replace the batch-synchronous loop with a continuously-fed pool
(submit the next chunk's read as soon as a worker frees), and raise the
`n_workers` cap from 8. Reading and writing still serialize on the HDF5 layer,
so realistically 57 → ~25 s.

*Reproducibility:* the docstring already guarantees bit-identity across
`n_workers` — each chunk is an independent per-column calculation. **Exactly
invariant.**

### 4. `_advance_flux` non-convolution work — 37 s, **3.2 %**

`_extremal_levy` is 18 s of it (5.1 s per finest-class draw): the
Chambers–Mallows–Stuck expression is assembled as ~10 sequential full-field
numpy ops, each a separate 1.9 GB pass. A single fused numba/numexpr kernel
would do it in one pass, ~3×.

*Reproducibility:* **exactly invariant** if the per-element float32 operation
order is preserved — it is purely elementwise, no reductions. A numba
`@njit(parallel=True)` version with the identical expression gives identical
bits. (Do **not** reorder the arithmetic: the `eps = 1e-6` clamps and the
`**(-1/alpha)` are order-sensitive by design — see the audit-item-33 comment.)

### 5. Nest setup: parent read + compensation replay — 30 s, **2.6 %**

17 s inflating 5.8 GB of zlib from the parent, 12.6 s in 84 `zoom_trilinear`
calls replaying each parent class's increment up the parent's own regrid chain
to full parent resolution — and each replayed array is then reduced to a
2048×34×115 slice. The replay computes 1.93 GB to keep 8 M cells (0.4 %).

*Fix:* slice first. The nest only ever needs `inc[np.ix_(x_indices, y_indices,
z_indices)]`; the intermediate zooms could be done on an x-full, y-narrow
column of the parent grid. This changes the interpolation stencil at the y
edges of the extracted column, so it is **not** bit-identical — trilinear
interpolation of a cropped array differs from cropping the interpolated array
near the crop boundary. Given that the halo exists precisely to absorb edge
effects it is probably harmless, but it is a realization change. **Low payoff,
nonzero risk — deprioritize.**

### 6. `_gradient_components` — 24 s, **2.1 %**

Four `np.roll` calls, each a full 1.9 GB copy, plus `np.gradient` which
allocates another. A fused numba stencil would cut it ~3×, and is
**exactly invariant** (same central differences, same order). Small absolute
payoff; do it only if you are already writing numba kernels for #4.

### 7. Convolutions — 24 s, **2.1 %** (GPU 2.8 s, CPU nest 21.5 s)

Nothing to fix. Worth stating plainly: at 0.55 % of a square, **the
`device="cuda"` path is not what makes the production run fast, and the whole
squares-on-GPU / nests-on-CPU pipelining in `run_production_squares.py` buys
almost nothing.** Measured at (2048, 2048, 115): CUDA 0.65 s vs CPU 2.48 s —
the GPU saves 1.8 s per call. Running the nest with `device="cuda"` would save
~16 s of its 657 s (and cost 13 GB of VRAM). The pipelining is still worth
keeping for the *overlap* — two processes, two cores — but not for the device.

**Optimizations that would change realizations (listed so they can be
declined knowingly):** moving `_bounded_amplitude_add` or the level-mean
reductions to the GPU or to a `math.fsum`/blocked accumulator; reducing the
bisection iteration count below the float32 freeze point; cropping before the
replay zooms in `refine`; any change to `SUPPORT_FACTOR`. Each perturbs the
field at roundoff and the flux clip amplifies it down-cascade until the
realization is unrecognizable — statistics preserved, single-realization tests
broken (the `SUPPORT_FACTOR` comment at simulate.py:29 documents exactly this).
**None of the top-three optimizations are in this category.**

---

## 6. Operation map — what the cascade actually does, in order, and what it costs

Percentages are of one square's 508 s.

**Before the loop (0.05 s, 0.0 %).** `_compute_all_grids` walks each size
class and integrates `dz(z) = k_z(z)/2` from the ground up to build that
class's own vertical grid; `_turbulon_envelope` builds the single 13³ Mexican-hat
kernel used at every class (it lives in grid-index space, so one array serves
all scales) and subtracts a Gaussian-weighted mean to make it exactly
zero-sum on the grid; `_compute_normalization` measures the mean profile's
vertical Haar fluctuation at the local outer scale `k_z,L(z)` and multiplies by
λ = 0.28927 to set the outer-class amplitude `C_L(z)`, then runs the `(k/L)^H_h`
ladder down; `_compensation_profiles` composes the per-hop retention tables into
the per-class, per-level damping `f` that makes every class deliver the same
amplitude convention on the output grid; `_bound_buffers` sums the remaining
cascade's amplitudes into the taper width `b(z)`. All of this is 1-D arithmetic
on 401-point profiles and is free.

Then, for each of the 9 classes from L = 1536 km down to 2dx = 6 km:

1. **Regrid (0.35 %).** `zoom_trilinear` lifts h′, q_t′ and the flux from the
   previous class's coarser grid onto this class's finer one (torch's
   `interpolate`, on the CPU, corner-aligned). Cheap, and one of the only
   multi-threaded things in the loop.
2. **Draw the noise (1.3 %).** `_sparse_levy` puts one extremal (β = −1)
   Lévy α-stable variable, α = 1.8, at every turbulon center — which at s = 1 is
   every cell. This is the log-generator of the multiplicative cascade.
3. **Advance the flux (2.2 % total).** The flux is multiplied by `exp(γ)` once
   per class, implemented as `F += conv(ψ, (exp(γ)−1)·F·f)`. γ is shifted so
   `⟨exp γ⟩ = 1` exactly; negatives are clipped to zero; then one scalar rescale
   restores the volume mean the field entered with — a corrector for that step's
   clip bias, nothing more. The same `(exp(γ)−1)·F`, normalized to unit mean
   absolute value, becomes `S_k`, the signed amplitude the scalars share.
   **This clip is the nonlinearity that makes single realizations chaotic.**
4. **Advective weight (1.8 %).** `_gradient_components` takes periodic central
   differences in x and y and `np.gradient` in z of the running field
   ⟨Φ⟩ + Φ′; they combine as `W = |∇_h Φ| + (ℓ_z/ℓ_x)|∂Φ/∂z|`. This is where
   turbulons get placed on the existing gradients — an eddy only makes an
   anomaly where there is something to advect.
5. **Bound taper (0.33 %).** `_bound_taper` computes a 0…1 factor that goes to
   zero as the running field approaches h_min/h_max or q_t bounds, so amplitude
   is redistributed away from levels pressed against a physical limit rather
   than clipped there.
6. **Normalize the joint pattern (~1 % in-loop algebra).** `W·S_k·taper` is
   divided, level by level, by its own mean absolute value over turbulon
   centers, then multiplied by `C_k(z)` and the compensation `f`. Normalizing the
   *product* (not the factors separately) is what keeps the `k^H_h` amplitude
   ladder from being eaten by the growing W–flux correlation.
7. **Convolve (0.55 %).** `convolve_fft_xy_oa_z` spreads each cell's amplitude
   over the Mexican-hat envelope: real 3-D FFT, circular in x and y, overlap-add
   and zero-padded in z. **This is the only operation that uses the GPU, and it
   is 0.55 % of the run.**
8. **Add under bounds (63.5 %).** `_bounded_amplitude_add` adds the convolved
   increment level by level, solving for a scale `s` and a shift `μ` such that
   the added field has exactly zero horizontal mean, keeps its intended mean
   absolute amplitude, and leaves h and q_t inside their physical bounds
   pointwise. The shift is found by 60 bisection steps per level. **This one step
   is two-thirds of the entire model.**

**After the loop:** the mean profiles are added back, a 1-ulp clip is applied,
h/q_t/flux are written to NetCDF with zlib-4 (7.3 %), the 27 staged per-class
increments are copied from tmpfs into the file (11.3 %), and
`compute_diagnostics` recovers T, q_v, q_c, q_i and p by a 5-iteration Newton
saturation adjustment with hypsometric pressure, over 8 threads that deliver
2.3 cores (11.2 %).

The strip nest re-runs steps 1–8 unchanged for three more classes down to
750 m, on grids up to 835 M cells, on the CPU, and step 8 is 84 % of it.

---

## 7. Reproduction

```
scratchpad/run_square.py OUT.nc --instrument          # replica of run_square()
scratchpad/run_nest.py    OUT.nc --instrument         # replica of run_nest()
scratchpad/analyze.py     events_square_2048.json
scratchpad/bench_bounded.py 2048,2048,115             # the A/B in §5.1
scratchpad/microbench.py    2048,2048,115
```
Instrumentation is a monkeypatch layer (`scratchpad/instr.py`) of
`perf_counter` wrappers around the named functions plus a 0.25 s RSS/VRAM/CPU
sampler; overhead is below the run-to-run noise (the instrumented square's
508.3 s agrees with `/usr/bin/time` at 510.3 s wall including interpreter
startup). Both heavy runs were confined with
`systemd-run --user --scope -p MemoryMax=45G`.
