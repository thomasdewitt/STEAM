The report file write is blocked by my harness (subagents return findings as text), so here is the full report. Benchmark scripts are in `/tmp/claude-1000/-home-thomas-code-and-data-turbulon-model/68899bbe-b782-4e72-9635-c992412b50b6/scratchpad/gpu-torch/` (`bench_levy.py`, `bench_levy2.py`, `bench_grad.py`, `bench_conv.py`, `bench_conv2.py`, `bench_bounded_gpu.py`, `bench_misc.py`) — the parent can save this text to `report.md` if wanted.

---

# STEAM: torch / GPU opportunity study

2026-07-30 · Ryzen 9950X (16c/32t), 60 GB RAM, RTX 5080 16 GB (sm_120, driver 595.80), torch 2.11.0+cu130 in `turbulon-analysis/.venv`, numpy 2.4.4, numba 0.65.0. Read-only: nothing in `turbulon-model/` was modified. All timings at the production finest-class shape **(2048, 2048, 115)** = 482.3 M cells, 1.80 GiB/float32 field, unless stated. Best-of-3 after warm-up.

Baseline for payoff arithmetic: the audit's 1166 s per member (square 508 + nest 657), of which `_bounded_amplitude_add` was 875 s. With the landed CPU fix (8.6×) that becomes **≈ 390 s per member** — every "% of pipeline" below is against 390 s, not 1166 s.

## 0. Headline

Three things came out of this that matter more than the questions asked.

1. **`_bounded_amplitude_add` on the GPU is 0.78 s against the new CPU version's 17–20 s** — 22× on the solve, **13–15× including a full host round trip**, at 7.2 GiB peak VRAM. It is the one GPU move that is unambiguously worth making: pipeline ~390 → ~307 s per member (−21 %).

2. **⚠ `convolve_fft_xy_oa_z(device='cpu')` silently returns an all-zero array at (2048, 2048, ·)** in this torch build whenever `torch.get_num_threads() >= 4` (default 16). A torch CPU `irfftn` bug, not a STEAM bug, but a live landmine: `simulate(nx=ny=2048, device='cpu')` produces a cascade in which every increment is zero, with no error. Production is unaffected (squares run cuda; no nest grid is 2048²) — but the audit's "CPU 2.48 s vs CUDA 0.65 s" was timing the zero-return. The honest CPU number is **6.29 s**. Details in §2.4.

3. **The Lévy draw can be made 3.0× faster with zero realization change** — 5.94 → 1.97 s — by chunking the arithmetic over a thread pool and splitting the PCG64 stream with `advance()`. Verified bit-identical, both arithmetic and random stream. No GPU, no numba, no new dependency.

Everything else is small, and the answer to the whole-hog question is **no** (§7): after the two items above, ~67 % of the pipeline is zlib and the Newton saturation adjustment, neither of which a GPU-resident cascade touches.

## 1. numpy/scipy → torch swaps in the hot path

### 1.1 The extremal-Lévy draw (audit item #4, 18 s ≡ 4.6 %)

| variant | s | vs shipped | bit-identical | notes |
|---|--:|--:|:--:|---|
| numpy RNG alone (2 × `random(N, f32)`) | 1.73 | — | — | floor for any numpy-RNG variant |
| **numpy shipped** | **5.94** | 1.00× | — | 1.73 RNG + 4.21 arithmetic |
| numpy, chunked over a thread pool (8 thr) | 3.38 | 1.76× | **yes** | 1.73 serial RNG + 1.62 arithmetic |
| **numpy, chunked + parallel PCG64 RNG** | **1.97** | **3.02×** | **yes** | 0.34 RNG + 1.62 arithmetic |
| numba `@njit(parallel=True)`, same op order | 2.52 | 2.36× | **no** | 1.73 RNG + 0.79 arithmetic |
| torch CPU, 16 threads, torch RNG | 3.73 | 1.59× | no | |
| torch CPU, 1 thread | 7.02 | 0.85× | no | slower than numpy |
| torch CUDA, stays on device | 0.11 | 56× | no | peak 7.19 GiB VRAM |
| torch CUDA + `.cpu().numpy()` (pageable) | 0.48 | 12.4× | no | |
| torch CUDA + persistent pinned copy-out | 0.14 | 42× | no | |

**The bit-identical parallelization is the interesting result.** Two independent pieces, both verified:

- *Arithmetic.* Purely elementwise, so slicing the flat array into 4 M-element chunks and running the identical numpy ufunc sequence per chunk in a `ThreadPoolExecutor` (numpy releases the GIL) gives `np.array_equal == True`. It also helps single-threaded — 4.21 → 3.24 s at one thread — because a 4 M-element chunk stays in L3 across the ten passes instead of streaming 1.8 GiB from DRAM ten times.
- *RNG.* `rng.random(n, dtype=float32)` consumes one `uint32` per draw, two per 64-bit PCG64 output, so a chunk of **even** length consumes exactly `len/2` states. Cloning the bit generator and calling `.advance(offset // 2)` per chunk reproduces the stream exactly (verified for both the `phi` draw and the continuation `R` draw), and `bg.advance(n // 2)` on the parent leaves the caller's generator correctly positioned. 1.76 → 0.34 s.

**numba is *not* bit-identical** — contrary to the audit's §5.4 claim. Same operation order, but `np.cos`/`np.log`/`np.sin`/`**` on float32 scalars go through libm in float64 and round, whereas numpy dispatches its own float32 SIMD kernels. Measured: 9.4 M of 20 M draws differ, max |Δ| = 2.4e-4 on draws of σ ≈ 3.2 (~7e-5 relative). Statistically indistinguishable (`⟨exp(0.13γ)⟩` agrees to 6 digits), but it *is* a realization change and should be labelled as one.

**Verdict: worth it — take the chunked bit-identical numpy version.** 3.0× for a ~30-line change, no new dependency, no realization risk, and *faster* than both numba and torch-CPU. Saves ~12 s/member (3 %).

*GPU bandwidth angle:* this kernel is ~10 streaming passes over 1.8 GiB ≈ 18 GiB of traffic; at ~950 GB/s that is 19 ms and we measure 106 ms — the GPU version is **transcendental-bound, not bandwidth-bound**, so there is little more to extract there.

### 1.2 The gradient stencil (audit item #6, 24 s ≡ 6.2 %)

| variant | s | vs shipped | bit-identical | notes |
|---|--:|--:|:--:|---|
| **numpy shipped** | **2.86** | 1.00× | — | 4 × `np.roll` full copies + `np.gradient` |
| numpy, chunked over x with a 1-cell periodic halo (8 thr) | 1.60 | 1.79× | **yes** | bandwidth-bound |
| numba fused stencil (`parallel=True`) | **0.24** | **11.9×** | no | max Δ 1.9e-9 (float32 coefficient rounding) |
| torch CPU, roll-based, 16 threads | 2.04 | 1.41× | no | |
| torch CUDA, stays on device | 0.20 | 14.1× | no | peak 14.3 GiB (roll allocates) |
| torch CUDA + full round trip | 0.88 | 3.3× | no | 10.7 GiB |

The shipped cost is entirely `np.roll` materializing four 1.8 GiB copies plus `np.gradient` allocating a fifth: ~9 arrays' worth of DRAM traffic to produce two. A fused stencil reads the field once and writes two outputs — hence 11.9×.

**Verdict: worth it, as a numba kernel, not as torch.** torch-CPU gains almost nothing (same rolls); torch-CUDA loses its 14× to the round trip. numba's 11.9× saves ~22 s/member (5.6 %). The 1.9e-9 difference is float32 rounding of the `np.gradient` non-uniform-spacing coefficients — a port computing those in float64 (as `np.gradient` does) would plausibly be bit-identical for the z part; x/y differs only through the `sqrt`. For a guaranteed-invariant option, chunked numpy is 1.79× for free.

### 1.3 The in-loop array algebra (~6 s/square)

Measured per scalar field per class at the finest shape (the `W` chain from `W = grad_z` through `W *= comp_k`, including `_bound_taper` and the level normalization): **2.86 s**. Two fields → 5.7 s at the square's finest class, matching the audit's 5.4 s.

Components, all single-threaded numpy: `running_sum = pert + mean_1d` 0.32 s · `_bound_taper` 1.14 s · `np.abs(W).sum(axis=(0,1), dtype=f8)` 0.55 s · `np.count_nonzero(W, axis=(0,1))` 0.28 s · each `W *= <1D>` broadcast pass 0.10 s.

| variant | s | vs shipped | invariance |
|---|--:|--:|---|
| shipped | 2.86 | 1.00× | — |
| chunked over x, 8 threads, factors fused | **1.17** | **2.44×** | roundoff-changing |
| torch CUDA + full round trip | 0.81 | 3.54× | roundoff-changing, 7.2 GiB VRAM |

The chunked version is not bit-identical for two fixable reasons: (a) the level reduction `Σ|W|` accumulates chunk partials, not numpy's pairwise order; (b) I fused `C_k * comp / level_mean` into one multiplication instead of three passes. Keeping three separate passes and reproducing pairwise order restores bit-identity at ~2.0× instead of 2.44×.

**Verdict: worth it as CPU threading (2.4×, ~9 s/member); not worth a GPU trip on its own** — 0.81 s round-trip vs 1.17 s CPU is not a difference worth restructuring for.

### 1.4 Summary of Q1

Torch-on-CPU never wins in this codebase. It lost to numpy on the Lévy draw (3.73 vs 1.97 s) and essentially tied on the gradient (2.04 vs 1.60 s); its intra-op parallelism is real but it pays for it in extra materialization and gives up bit-identity for nothing. **The right CPU tool here is chunked numpy in a thread pool** (bit-identical, 1.8–3.0×), with numba reserved for the one case where fusion buys an order of magnitude (the gradient stencil).

## 2. GPU direct convolution for the fine classes

Kernel 13³ (`SUPPORT_FACTOR = 3`, `s = 1`), 2197 taps; field 482 M cells → 1.06 T MAC = 2.1 TFLOP per convolution.

### 2.1 Speed and memory at (2048, 2048, 115)

| method | s | peak VRAM | notes |
|---|--:|--:|---|
| **FFT-OA CUDA, shipped (`n_fft = 128`)** | **0.62** | **11.88 GiB** | one block covers the padded array |
| FFT-OA CUDA, `n_fft = 64` | 0.64 | 7.88 GiB | 3 blocks |
| FFT-OA CUDA, `n_fft = 32` (= the CPU choice) | 0.64 | **5.88 GiB** | 6 blocks |
| FFT-OA CUDA, `n_fft = 16` | 0.80 | 4.88 GiB | 29 blocks |
| conv3d CUDA, 1 x-chunk | 6.08 | 3.83 GiB | cudnn, TF32 disabled |
| conv3d CUDA, 2 / 4 / 8 x-chunks | 6.04 / 6.05 / 6.02 | 1.92 / 0.97 / **0.50** GiB | |
| conv3d CUDA, `channels_last_3d`, 4 chunks | 6.03 | 1.02 GiB | no help |
| conv3d CPU (torch, 16 chunks) | 12.74 | — | |
| FFT-OA CPU, 2 threads (correct) | 6.29 | — | see §2.4 |
| FFT-OA CPU, 16 threads | *2.63* | — | **returns zeros — invalid** |

Timings include host→device→host transfer in every case. VRAM is `max_memory_allocated()`, i.e. includes the 1.80 GiB field.

### 2.2 Verdict

**Not worth it for speed — conv3d is 10× slower than the FFT.** 6.0 s for 2.1 TFLOP is 0.35 TFLOP/s on a card that does ~55 TFLOP/s fp32: cudnn has no good algorithm for a single-channel 13³ 3-D convolution (implicit-GEMM degenerates at `C_in = C_out = 1`). Chunking does not change the time at all, confirming compute-bound not memory-bound.

**The memory win is real but you do not need conv3d to get it.** The shipped `cuda_block_fft_size` deliberately grows `n_fft` until it fills free VRAM; the 12.8 GiB the audit flagged is *chosen*, not required. Capping `n_fft` at 32 costs **0.02 s** and gives back **6.0 GiB** — strictly better than conv3d's 6.0 GiB-for-5.4 s. Capping at 16 costs 0.18 s and gives back 7.0 GiB.

So for question 2: **cap `n_fft`, don't switch algorithms.** Concretely, `cuda_block_fft_size` should take a *budget* argument rather than "all free VRAM", so a caller wanting the cascade state resident (§7) can ask for a 4 GiB convolution instead of a 12 GiB one. conv3d only becomes interesting below a ~3 GiB budget, and even then only if you'll pay 5.4 s per call.

### 2.3 Accuracy

Against the `scipy.ndimage` reference at (96, 96, 40), as max |Δ| / mean|out|: FFT-OA cpu 2.34e-6 · FFT-OA cuda 3.12e-6 · conv3d cpu 4.29e-6 · conv3d cuda 4.29e-6 · conv3d cuda 4-chunk 4.29e-6.

All four are float32-accurate and mutually consistent; conv3d is very slightly *worse* than the FFT (2197-term serial accumulation), and conv3d CUDA is bit-identical to conv3d CPU and invariant to chunking. **Invariance class: realization-changing at roundoff** for any switch between them (amplified through the flux clip, per the `SUPPORT_FACTOR` comment) — statistics-identical. TF32 must be off (`torch.backends.cudnn.allow_tf32 = False`); with it on, conv3d loses ~3 decimal digits.

### 2.4 ⚠ The CPU FFT convolution returns zeros at 2048²

Isolated to torch, not to STEAM:

```python
torch.fft.irfftn(torch.fft.rfftn(x, s=shape), s=shape)   # x float32
```

returns an all-zero array when `shape[:2] == (2048, 2048)` and `torch.get_num_threads() >= 4`. Characterized:

- The **forward** `rfftn` is correct (matches numpy to 1.3e-6 relative). Only the inverse fails.
- Trigger is the leading 2-D plane being exactly 2048 × 2048, at **any** `nz` (tested 1, 2, 3, 8, 16, 32, 64, 115).
- **Not** a total-size effect: (1448, 1448, 32) and (4096, 1024, 32) — same 67 M / 134 M elements — are correct, as are (1024, 1024, 256), (4096, 4096, 8), (4096, 2048, 32), (2048, 4096, 32), (2047, 2048, 32).
- Correct at 1 and 2 threads, wrong at 4, 8, 16.
- float32 only; float64 correct. Complex-to-complex `ifftn` correct. numpy's `irfftn` correct at the same shape.

Consequences for STEAM:

- All production **nest** grids — (4096, 44, 169), (8192, 76, 248), (16384, 140, 364) — and the common test shapes (1024², 512², 256²) are **correct** (verified through `convolve_fft_xy_oa_z` itself). Production output is not compromised.
- `simulate(nx=ny=2048, device='cpu')` **is** compromised, silently: every class increment is zero. Anything comparing a CPU and a CUDA run at 2048² will see the CPU run produce the bare mean profiles.
- The audit's §5.7 "CUDA 0.65 s vs CPU 2.48 s, the GPU saves 1.8 s per call" should read CUDA 0.62 s vs CPU **6.29 s** — the GPU saves 5.7 s per call at 2048².

Cheapest guard: a shape/thread check in `convolve_fft_xy_oa_z` that raises (or forces `torch.set_num_threads(2)` for the transform) when `nx == ny == 2048 and torch.get_num_threads() >= 4`. Better: a one-off correctness assertion on a small array at import, or use numpy's FFT on the CPU path. Worth reporting upstream.

## 3. Noise generation on the GPU

Measured (§1.1): 0.106 s on device, 0.140 s including a copy-out into a **persistent pinned** host buffer, 0.477 s with a naive pageable `.cpu().numpy()`. Against 1.97 s for the best bit-identical CPU version, that is still 10–14× *including* transfer.

Transfer economics, measured directly (1.93 GB buffer):

| direction | pinned, persistent buffer | pageable |
|---|--:|--:|
| H→D | 33.6 ms (57.5 GB/s) | 76.5 ms (25.2 GB/s) |
| D→H | 33.8 ms (57.1 GB/s) | 104.1 ms (18.5 GB/s) |

`pin_memory()` on a *fresh* 1.93 GB array costs **0.372 s** — more than the copy. Any pinned design must allocate staging buffers once and reuse them; calling `.pin_memory()` per operation is slower than pageable. (This is why the "pinned" rows in §5 are *worse* than "pageable" — that benchmark pins fresh arrays.)

**Verdict: not worth it on its own; worth it only as part of a larger GPU move.** The absolute saving is ~1.8 s per finest-class call, ~6 s per member (1.5 %) — real, but it costs a persistent 1.9 GiB pinned host buffer, 7.2 GiB transient VRAM, a device RNG stream that must be seeded and recorded separately from the numpy one, and a CUDA dependency in the middle of `_advance_flux`. The CPU alternative is free, bit-identical, and gets 3.0×.

Note too that `gamma` is *immediately* followed by four more elementwise full-field host passes (`gamma *= scale`, `expm1` 0.79 s, `noise[gamma == 0] = 0` 0.48 s, `noise * flux`). Downloading `gamma` only to stream it four more times through DRAM is the worst of both worlds; if the draw goes to the GPU, all of `_advance_flux` should — and then `flux` must live there too, which is §7.

## 4. Interpolation — `zoom_trilinear`

| resample | CPU (torch, 16 threads) | CUDA + full round trip |
|---|--:|--:|
| (1024, 1024, 78) → (2048, 2048, 115) | **1.16 s** | 1.44 s |
| (512, 512, 53) → (1024, 1024, 78) | 0.29 s | 0.21 s |

**Confirmed: a non-issue, and the GPU is *slower* at the size that matters.** 1.76 s for all 24 calls in a square is 0.45 % of it; the 15.9 s in the nest is 93 calls of which 84 are the compensation replay — an algorithmic cost (the replay computes 1.93 GiB to keep 8 M cells), not an interpolation cost.

History, one sentence: `2de20d1` (2026-04-16, "Port FFT convolution and trilinear zoom to torch") replaced `scipy.ndimage.zoom(order=1)` with `torch.nn.functional.interpolate(align_corners=True)` and recorded **"~60× faster"** in the commit message — so Thomas's memory of slow interpolation is the `scipy.ndimage` era, fixed three and a half months ago.

## 5. `_bounded_amplitude_add` on the GPU (added scope)

All 115 levels solved simultaneously: `mu_lo`/`mu_hi` are `(nz,)` float64 device vectors, the 60 bisection steps are one vectorized pass each, and the level means are per-level reductions over dims (0, 1). Two regimes: *loose* (the audit's synthetic setup, clip rarely binding) and *pinned* (increment amplitude ramped to 40000 and levels pushed toward `phi_max`, so most levels stall against the clip as q_t does aloft).

| | loose | pinned |
|---|--:|--:|
| CPU, contiguous + 16 threads (the landed fix) | 17.4–19.6 s | 18.1–20.7 s |
| **GPU solve only** | **0.78 s** | **1.02 s** |
| GPU + pageable host round trip (2 up, 1 down) | 1.29 s | 1.52 s |
| speedup incl. transfer | **15.2×** | **13.6×** |
| speedup, field already resident | **24.5×** | **20.3×** |
| peak VRAM | **7.19 GiB** | 7.19 GiB |

(The CPU baseline was noisy — 16.3 to 23.4 s across repeats — because other agents were loading the box. Treat it as "≈ 15–20 s", consistent with the reported 8.6× over the shipped 99 s.)

Thomas's instinct is right and it is why this one wins where the others do not: **the bisection is 60 passes over the whole field**, ~60 × 5.4 GiB ≈ 320 GiB of memory traffic per call. The CPU has ~80 GB/s usable, the 5080 ~950 GB/s. 0.78 s for ~430 GiB achieved is 550 GB/s — 58 % of peak, about right for a strided-reduction workload. Pure bandwidth win; no amount of CPU threading closes it.

**VRAM, and how to lower it.** The 7.19 GiB is 4 field-sized buffers: field, increment, two clip caps. Two notes:

- A naive `t.sum(dim=(0,1), dtype=torch.float64)` **upcasts the entire tensor**, adding 3.6 GiB and 0.7 s per call. Reducing in two stages — `t.sum(dim=0).sum(dim=0, dtype=torch.float64)` — costs nothing in accuracy (float32 tree-sum over 2048, then exact float64 over 2048 partials) and took the measured peak from 12.59 → 7.19 GiB and the solve from 1.47 → 0.78 s. Stated explicitly because it is the single easiest way to get this wrong.
- The caps can be dropped by working in shifted space (`e = d + phi`, clip against the scalar bounds, subtract the per-level mean of `phi`) — I already do this for the bisection, which is why bisection residency is only 2 fields. Doing it for the rescale loop too lands the whole call at **~5.4 GiB**.

**Transfer cost if the field lives on the CPU.** 2 fields up + 1 down = 5.4 GiB. Persistent pinned staging: 3 × 33.7 ms ≈ **0.10 s**. Pageable: ~0.26 s. Either way 8–25 % of GPU time; does not threaten the verdict.

**Invariance: realization-changing.** GPU reduction order differs and the level means feed the bisection, so the whole call differs at roundoff. Measured vs CPU: max |Δ| 0.0625 on a field of mean|·| 757 (8e-5 relative) in the loose regime; per-level delivered amplitudes ⟨|δ|⟩ agree to **3.4e-8** (loose) and **4.1e-6** (pinned); per-level means agree (0.352 vs 0.368 max residual, both ≈ 0 against a 6.5e4 bound width). The three deposit conditions are satisfied to the same tolerance by both. Statistics-identical; single-realization tests break.

**Verdict: worth it, on its own, now.** ~14× on 75 % of the original pipeline. Per member: 875 s (shipped) → ~102 s (landed CPU fix) → **~19 s** (GPU with transfers: 3 square calls + 6 nest calls, nest ones at 835 M cells ≈ 1.4 s solve + 0.3 s transfer each). Pipeline **390 → ~307 s**. It also *reduces* peak host RSS, since the increment need not be materialized on the host at all if the convolution already ran on the device.

## 6. Anything else in the hot loop

Ranked; all small compared to §5.

1. **`noise[gamma == 0.0] = 0` in `_advance_flux` (0.48 s per call, ~1.7 s/member).** At production sparsity `s_x = s_y = s_z = 1` every cell is a turbulon centre, so `gamma` is nonzero everywhere except the `zero_bottom`/`zero_top` bands. The line allocates a 482 MB boolean mask and does a masked scatter over the whole field to zero at most `2 * n_zero` levels. Replacing it with `noise[:, :, :n_zero] = 0` / `[-n_zero:]` when all sparsity factors are 1 is **bit-identical** and free. (At `s > 1` the general path is still needed.)

2. **`save_class_increments=True` costs three extra full-field copies per class.** `flux_before = flux.copy()` and `before = perturbation_field.copy()` for each of h and q_t — 5.4 GiB of allocation and ~1.0 s of memcpy at the finest class, on top of the 6.5 GiB of tmpfs the audit flagged. The `flux` one is avoidable entirely: `_advance_flux` knows its own increment (`CONVOLVE(noise, ...)`) and could return it instead of the caller differencing before/after. The scalar ones are avoidable if `_bounded_amplitude_add` returns the delta it applied (it computes `d` explicitly per level and currently throws it away after `+=`).

3. **`_bound_taper` is 1.14 s per field per class** — 4 passes over 1.8 GiB, of which `g /= np.where(b > 0, b, 1e-30)` rebuilds a `(nz,)` `where` result each call. `b` is a 115-element profile constant within a class; hoist it. (Hoisting the *reciprocal* is not bit-identical — `x/b != x*(1/b)` in float32 — so hoist the `np.where` array and keep the divide.)

4. **`np.count_nonzero(W, axis=(0,1))` costs 0.28 s** and, at `s = 1`, counts almost the whole level every time. It is genuinely needed (taper and gradient both zero cells), but it and `np.abs(W).sum(...)` are two separate 1.8 GiB streams (0.83 s combined) that one numba kernel would fuse into one.

5. **The nest is written uncompressed and the square compressed** — already audit §2, but re-flagged because after the bounded-add fix `write_netcdf` + `write_class_increments` (94.5 s) becomes the **single largest item in the pipeline**, 24 % of 390 s. Nothing in this report touches it, and it is a bigger number than everything in §1 combined.

## 7. The whole-hog option: entire cascade state resident on the GPU

Sketch and budget only, using the per-piece VRAM measured above.

### 7.1 Does it fit? — the square: **yes, barely**

Field = 1.80 GiB. Concurrent residency at the worst moment of a class:

| | GiB |
|---|--:|
| persistent state: `h_perturbation`, `qt_perturbation`, `flux` | 5.40 |
| `S_k` (lives across both field passes) | 1.80 |
| `W` (working pattern) | 1.80 |
| **subtotal, always live** | **9.00** |
| + convolution workspace, FFT `n_fft = 32` (measured 5.88 peak incl. its 1.80 input) | +4.08 → **13.08** |
| or: + `increment` + 2 clip caps during the bounded add | +5.40 → **14.40** |

Usable VRAM is 15.46 GiB minus ~0.5 GiB held by the desktop ≈ **14.9 GiB**. It fits, with 0.5–1.8 GiB of slack — but only if: `n_fft` is capped (the shipped 128 alone blows it: 11.9 GiB for the convolution with 9.0 GiB already live); the bounded add uses the shifted-space formulation (§5), taking the second row to 12.6 GiB; and `save_class_increments` streams to the host as increments are produced rather than adding another resident copy. Also `_gradient_components` currently needs `running_sum`, `grad_h`, `grad_z` live at once (+5.4 GiB) — a device version must fuse it (exactly the numba-style kernel of §1.2, written in torch) or it does not fit.

### 7.2 Does it fit? — the nest: **no**

The finest nest class is 834.9 M cells = **3.11 GiB per field**. Three state fields alone are 9.33 GiB; adding `S_k` and `W` is 15.6 GiB before any convolution workspace. It does not fit on a 16 GiB card, by roughly 2×, and there is no easy decomposition: the nest spans the full 6144 km in x, so x is *periodic* and the FFT convolution cannot be slabbed along it. (Direct conv3d *can* be slabbed — 0.50 GiB at 8 chunks, measured — but that fixes only the workspace, not the 15.6 GiB of state.)

The nest's realistic ceiling is therefore **CPU-resident state with the bounded add offloaded per call** (§5): 6 calls, 9.3 GiB of transfer each, ~1.4 s solve + ~0.3 s pinned transfer. VRAM 12.4 GiB with the current 4-buffer formulation, 9.3 GiB with the shifted-space one.

### 7.3 Which pieces are hard

- **The bounded bisection is *not* hard** — it vectorizes cleanly over levels and is the biggest win (§5). The per-level early-exit conditions in the rescale loop become `torch.where` masks plus one `done.all()` host sync per iteration.
- **`np.count_nonzero(inner, axis=(0,1))` and the `Σ|W|` level reduction** are fine on the GPU (`vector_norm(ord=1, dim=0)`), *provided* you never write `dim=(0,1), dtype=float64` — see the 3.6 GiB upcast trap in §5.
- **No sorting or quantile ops anywhere in the cascade**, so the classic "hard on GPU" reductions do not arise. `compute_diagnostics` has a 5-iteration Newton solve that is embarrassingly parallel and would port trivially — but it runs after the cascade and would need the fields back on the device.
- **`np.interp` for the mean profiles** is 1-D on ≤ 401 points; leave it on the host.
- **The genuinely awkward piece is host↔device orchestration around the netCDF writer**, which is 24 % of the post-fix pipeline and must run on the host.

### 7.4 Expected end-to-end speedup

Building from the audit's phase table, with the CPU bounded-add fix assumed landed:

| configuration | square | nest | member | vs post-fix |
|---|--:|--:|--:|--:|
| shipped (audit) | 508 | 657 | 1166 | — |
| + landed CPU bounded-add fix (8.6×) | 222 | 168 | **390** | 1.00× |
| + **GPU bounded-add only** (§5) | 193 | 114 | **307** | **1.27×** |
| + Lévy/gradient/algebra CPU fixes (§1) | 180 | 100 | **280** | 1.39× |
| + full GPU-resident square cascade (nest stays CPU) | 172 | 100 | **272** | 1.43× |

At the bottom row the member is: `write_netcdf` + `write_class_increments` 94.5 s, `compute_diagnostics` 57 s, nest parent read + compensation replay 30 s — i.e. **~67 % of what is left is zlib, the Newton saturation adjustment, and an algorithmically wasteful replay, none of which the GPU-resident cascade touches.**

**Verdict on whole-hog: not worth it.** The restructure is large (every cascade kernel rewritten, a device RNG stream, a new memory-budget discipline, `save_class_increments` re-plumbed to stream) and it still leaves the nest on the CPU because it does not fit; it changes every realization; and it buys ~35 s per member beyond what the single `_bounded_amplitude_add` offload gives. **Do §5 and stop.** If more is wanted, the next ~150 s are in `complevel`, a continuously-fed diagnostics pool, and slicing before the nest replay — all CPU-side, all in the audit already.

## 8. Recommendation on running convolution on CPU everywhere

**Against — keep the CUDA path**, and the reasoning got stronger, not weaker, during this study.

- The premise was wrong. The audit's "+8 s/square for CPU convolution" rested on a 2.48 s CPU measurement that was **timing an all-zero return** (§2.4). The correct CPU time at 2048² is **6.29 s** against 0.62 s on the GPU, so dropping CUDA costs **+17 s per square** (3 finest-class convolutions), plus the smaller classes — call it +19 s, ~9 % of the post-fix square, not 3 %.
- It is currently *unsafe*: at the default 16 threads, the CPU path at 2048² produces silent zeros. Making CPU-everywhere the production path would require the guard from §2.4 first, and the guard itself forces 2 threads, which is where the 6.29 s comes from.
- The VRAM argument dissolves. You do not need to drop CUDA to recover the 12.8 GiB — capping `n_fft` at 32 recovers 6.0 GiB for 0.02 s, and at 16 recovers 7.0 GiB for 0.18 s (§2.1). That is nearly all of the memory benefit at ~1 % of the time cost.
- The code does not actually simplify much: `convolve_fft_xy_oa_z` already shares one torch implementation across both devices; removing CUDA deletes the VRAM guard and `cuda_block_fft_size`, ~40 lines, and the `device=` parameter would still need to exist for `refine`.

**Conditioning on the GPU-resident question:** since §7 concludes *against* a GPU-resident cascade, the question does not invert — but the recommendation is stable either way. If a GPU-resident square were pursued, the convolution obviously must stay on the device (transferring 1.8 GiB out and back per convolution to run a 6.3 s CPU FFT would be absurd), and the `n_fft` budget cap becomes mandatory rather than merely nice. In the actual recommended configuration — CPU cascade with `_bounded_amplitude_add` offloaded — the GPU is already initialized, already holds a context, and the convolution is free to keep using it.

**Concrete recommendation:** keep `device='cuda'` for squares; add the `n_fft` budget cap (11.9 → 5.9 GiB, ~free); add the 2048²/thread guard to the CPU path so it can never silently return zeros; and consider running the **nest** with `device='cuda'` too — it saves ~14 s of its CPU convolution time and, at the nest's grids, needs far less VRAM than the square (the leading dimension is 16384 and `ny` is only 140, so the OA block is small).

## 9. Per-question verdict table

| # | question | verdict | measured | memory | invariance |
|--:|---|---|---|---|---|
| 1a | Lévy draw → torch/numba | **worth it — as chunked numpy, not torch** | 5.94 → **1.97 s** (3.0×) | unchanged | **exactly invariant** (arithmetic *and* RNG stream verified) |
| 1b | Lévy draw → GPU | worth it only inside a GPU `_advance_flux` | 5.94 → 0.14 s | +7.2 GiB VRAM, +1.9 GiB pinned host | realization-changing (torch RNG) |
| 1c | gradient stencil | **worth it — numba** | 2.86 → **0.24 s** (11.9×) | −7 GiB transient host alloc | roundoff-changing (1.9e-9); float64-coefficient port likely exact |
| 1d | gradient stencil, invariant option | worth it if bit-identity required | 2.86 → 1.60 s (1.79×) | same | **exactly invariant** |
| 1e | in-loop `W` algebra | worth it (CPU threading) | 2.86 → 1.17 s (2.44×) | unchanged | roundoff-changing as written; ~2.0× available bit-identical |
| 2 | GPU direct conv3d | **not worth it** — 10× slower; get the memory from `n_fft` | 6.05 s vs 0.62 s FFT | 0.50 vs 11.88 GiB | roundoff-changing |
| 2′ | cap `n_fft` at 32 | **worth it** | +0.02 s | **11.88 → 5.88 GiB** | roundoff-changing (different OA blocking) |
| 3 | noise on GPU | not worth it standalone | saves ~6 s/member | +7.2 GiB VRAM | realization-changing |
| 4 | `zoom_trilinear` | **non-issue; GPU is slower** | 1.16 s CPU vs 1.44 s CUDA | — | — |
| 5 | misc (§6) | small, mostly free | ~2–3 s/member + 5.4 GiB host | mostly bit-identical | |
| 5′ | **CPU FFT zero-return at 2048²** | **bug — must be guarded** | correctness, not speed | — | — |
| ★ | **`_bounded_amplitude_add` on GPU** | **worth it, do this one** | 17–20 s → **1.3 s** (14×) | 7.2 GiB VRAM (5.4 achievable) | realization-changing; ⟨\|δ\|⟩ agrees to 3e-8 |
| 6 | whole-hog GPU cascade | **not worth it** — square fits at 13–14 GiB, nest does not fit at all, +35 s/member over ★ | — | 14.4 / >15.6 GiB | all realizations change |
| 8 | CPU convolution everywhere | **against** — costs +19 s/square, currently unsafe, and `n_fft` capping gets the memory anyway | 6.29 s vs 0.62 s | 11.88 → 5.88 GiB via `n_fft` instead | — |

## 10. Reproduction

```
cd /home/thomas/code-and-data/turbulon-analysis
.venv/bin/python <scratchpad>/gpu-torch/bench_levy.py        # numpy/numba/torch/CUDA Levy
.venv/bin/python <scratchpad>/gpu-torch/bench_levy2.py       # bit-identical chunked + parallel PCG64
.venv/bin/python <scratchpad>/gpu-torch/bench_grad.py        # gradient stencil, 5 variants
.venv/bin/python <scratchpad>/gpu-torch/bench_conv.py        # conv3d vs FFT-OA, CPU + CUDA
.venv/bin/python <scratchpad>/gpu-torch/bench_conv2.py       # accuracy + n_fft/VRAM sweep
.venv/bin/python <scratchpad>/gpu-torch/bench_bounded_gpu.py # GPU bounded-add, 2 regimes
.venv/bin/python <scratchpad>/gpu-torch/bench_misc.py        # W chain, taper, level norm, zoom
```

The torch CPU FFT bug reproduces in four lines:

```python
import torch
torch.set_num_threads(16)
x = torch.randn(2048, 2048, 32)
y = torch.fft.irfftn(torch.fft.rfftn(x, s=x.shape, dim=(0,1,2)), s=x.shape, dim=(0,1,2))
print(y.abs().mean())   # 1.9e-07 ; correct answer 0.798, and correct at <= 2 threads
```