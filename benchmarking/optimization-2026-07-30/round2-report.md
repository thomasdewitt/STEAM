# STEAM round-2 optimization — implementation report

2026-07-30 · turbulon-model, branch `main`, base `098f3e1` · all changes **uncommitted**
in the working tree (`steam/simulate.py`, `steam/utils.py`; the `steam.egg-info/`
modifications were already there before I started).

GPU was free at the start (530 MiB, desktop only) and at the end (596 MiB). No VRAM
contention from another process was observed during any measurement.

Runner: `/home/thomas/code-and-data/turbulon-analysis/.venv/bin/python`.
Tests: `./.venv/bin/python -m pytest tests/ -q`.
Reference builds: `scratchpad/head_repo/` is a `git archive HEAD` copy of `steam/`,
driven through `scratchpad/run_head.py` (which strips the editable-install meta-path
finder so `sys.path` wins).

---

## Item 1 — guard the torch CPU `irfftn` bug  (`steam/utils.py`)

**Change.** In `convolve_fft_xy_oa_z`, when `device != 'cuda'` and the *transform*
shape's leading two dims are exactly `(2048, 2048)`, drop to `torch.set_num_threads(2)`
for the duration of the call and restore the thread count before returning.
Guarded on the *trigger*, not on the output.

Comment cites **pytorch/pytorch#169670** (multithreaded MKL 2024.x DFTI mis-normalizes a
3-D c2r descriptor with 2048×2048 leading dims; fix PR #169743 closed unmerged) and
records why rescaling by 2048² would be the wrong fix.

**Verification** (`scratchpad/verify1.py`, torch default 16 threads):

| check | result |
|---|---|
| `(2048,2048,32)` CPU vs CUDA, `max|Δ| / mean|ref|` | **8.56e-06** (float32 FFT accuracy) |
| `(2048,2048,32)` CPU `mean|out|` after fix | 35.6543 (CUDA: 35.6543) |
| same shape under HEAD code | 8.50065e-06 — ratio to the correct answer **4.194e6 = 2048²**, exactly the reported mis-normalization |
| thread count after the call | 16 (restored) |
| `(512,512,32)` bit-identical to HEAD | **True** |
| `(1024,1024,40)` bit-identical to HEAD | **True** |
| `(2047,2048,16)` bit-identical to HEAD | **True** |

Note the confirmation of the coordinator's correction: the old output is *not* zero, it is
the right field divided by 2048². Any "is it zero?" detection would indeed have been flaky.

---

## Item 2 — Lévy draw, chunked + parallel PCG64  (`steam/simulate.py`)

**Change.** `_extremal_levy` now (a) draws both uniforms through
`_parallel_uniform_float32`, which splits the PCG64 stream by cloning the bit generator
and `advance(offset // 2)` per chunk, and (b) runs the identical elementwise sequence
per 4 M-element chunk (`_levy_chunk`) on an 8-worker `ThreadPoolExecutor`.
`_sparse_levy` is untouched.

Two details worth keeping in mind when reading it:

- Chunk boundaries are placed on *even offsets past the generator's cached half-state*
  (`state['has_uint32']`), so every chunk after the first starts on a whole 64-bit state.
- The caller's generator is repositioned by copying the **last chunk's** state back, not
  by `advance(n // 2)`. That is what makes an odd `size` correct: it reproduces the
  leftover cached uint32 that a serial draw would have left.

**Verification** (`scratchpad/verify2.py`), against a verbatim copy of the HEAD function:

| check | result |
|---|---|
| values identical, `size` ∈ {0, 1, 7, 4194304, 4194305, 12000001} | **True** (all) |
| downstream stream identical after each (float32 / float64 / integers draws) | **True** (all) |
| odd draw then a second odd draw: values + full bit-generator state | **True / True** |
| `_sparse_levy` factors (1,1,1), (2,2,1), (3,2,4): values and zero-band counts | **identical** (76032/76032, 96624/96624 zeros) |
| production size 2048·2048·115, both φ and R draws | **bit-identical** |
| caller RNG state after, and next 1000 draws | **identical** |
| timing | old **5.60 s** → new **1.66 s** (**3.38×**) |

**Full-sim check.** `scratchpad/e2e_square.py` (256², dx = 3 km, outer 384 km → 7 classes,
ℓ_s = 10 m, icon_lem snap0, anchored bounds, `device='cpu'`, seed 2000,
`save_class_increments=True`) under HEAD vs items 1+2:

> **16/16 netCDF variables bit-identical.**

Test suite after items 1–2: **143 passed**, same as HEAD.

---

## Item 3 — cap the CUDA overlap-add FFT block at 32  (`steam/utils.py`)

**Change.** `CUDA_MAX_BLOCK_FFT = 32`; `cuda_block_fft_size` now starts from
`max(n_fft_min, min(whole_padded_array, 32))` and keeps the existing shrink-to-fit loop
below that, so a VRAM-short machine still steps down and still raises `MemoryError` when
even the minimal block will not fit.

Per the mid-flight note, the docstring frames the cap as **headroom for the item-4
offload and the cascade state at larger grids**, not as a memory saving for its own sake.

**Verification** (`scratchpad/verify3.py`, best of the second call, transfers included):

| shape | n_fft old → new | time old → new | peak VRAM old → new | new vs old `max|Δ|/mean|out|` |
|---|---|---|---|---|
| (2048, 2048, 115) — production finest class | 128 → 32 | 0.761 → 0.924 s | **11.875 → 5.875 GiB** | 3.28e-06 |
| (256, 256, 115) | 128 → 32 | 0.013 → 0.015 s | 0.186 → 0.093 GiB | 2.87e-06 |
| (4096, 44, 169) — nest-shaped | 256 → 32 | 0.065 → 0.062 s | 1.102 → 0.350 GiB | 2.86e-06 |

The +0.16 s at the production shape is larger than the report's +0.02 s; both are noise
against the 39 s the square gained overall. Realization-changing at float32 FFT roundoff,
as expected.

---

## Item 4 — GPU `_bounded_amplitude_add`  (`steam/simulate.py`)

**Change.** `_bounded_amplitude_add` gains a `device` parameter and dispatches to
`_bounded_amplitude_add_cuda` when it is `'cuda'`; `cascade_loop` passes its own `device`
down. The CPU path is untouched and is what the nest keeps using — a genuine device
dispatch, not a fallback.

The CUDA solve does all `nz` levels at once: `a0`, the rescale factor and the bisection
bracket `mu_lo`/`mu_hi` are `(nz,)` float64 device vectors; the rescale loop's per-level
early exits are a `done` mask (which also freezes the demean and the scale for finished
levels, matching the CPU's `break` exactly) with one `done.all()` host sync per iteration;
60 vectorized bisection steps follow. Reductions use the **two-stage**
`sum(dim=0).sum(dim=0, dtype=torch.float64)` (`_level_mean`, `_level_mean_abs`) — never
`dim=(0,1), dtype=float64`.

Window/halo semantics are exact: `_inner_view` is applied to the device tensors for every
reduction, while every clamp runs over the full array, halo included.

**Memory: 3 field-sized device buffers, 5.39 GiB measured at (2048, 2048, 115)** — the
shifted-space target. The clip caps are never materialized:
`clip(d, φ_min−φ, φ_max−φ)` is computed as `clamp(d+φ, φ_min, φ_max) − φ`, and the
bisection runs entirely in the shifted variable `e = d + φ`.

**Function-level verification** (`scratchpad/verify4.py`, `(2048, 2048, 115)`, both
regimes from `bench_bounded_gpu.py`, plus a nest-like window):

| regime / window | CPU | GPU (incl. transfers) | speedup | peak VRAM | per-level ⟨\|δ\|⟩ max rel. diff (gate 1e-4) | per-level mean of δ, max\|·\| CPU → GPU | bound violations |
|---|--:|--:|--:|--:|--:|---|--:|
| loose, no window | 16.06 s | **1.49 s** | 10.8× | 5.39 GiB | **3.66e-08** | 3.18e-06 → 1.56e-02 | 0 of 4.82e8 |
| pinned, no window | 18.24 s | **1.69 s** | 10.8× | 5.39 GiB | **3.40e-06** | 1.14e-03 → 1.08e-01 | 0 of 4.82e8 |
| pinned, window (256,1792,256,1792) | 17.05 s | **1.61 s** | 10.6× | 5.39 GiB | **2.77e-06** | 1.14e-03 → 8.66e-02 | 0 of 4.82e8 |

(bound width 6.5e4, A0 up to 3.2e4; field mean\|·\| 757 loose / 1.51e4 pinned;
max\|GPU−CPU\| on the field 0.063 loose / 0.70 pinned.)

Two things in that table need saying plainly:

1. **Bounds are respected pointwise *exactly* by the GPU path and only approximately by
   the CPU path.** The GPU clamps the field itself, so 0 of 482 344 960 cells fall outside
   `[φ_min, φ_max]`. The CPU path clips the *increment* against float32 caps and then does
   `pert += d`, which can land a few 0.03-sized ulps outside. This is an improvement, but
   it is a behaviour change worth knowing about.

2. **The per-level mean of δ is larger on the GPU: 0.0156 against the CPU's 3.2e-06 in
   the loose regime** (0.11 vs 1.1e-03 pinned). This is not a bug and not a convergence
   failure — 0.0156 is exactly *half a float32 ulp of the field* (φ ≈ 3.3e5, ulp 0.03125).
   The shifted formulation evaluates the bisection trial at field magnitude, so the
   achievable zero-mean residual is quantized at that scale. Three notes:
   - relative to the delivered amplitude it is 2e-5 of A0, and relative to the field 5e-8;
   - the quantization is *per-value*, so aloft (where q_t ≈ 1e-6 and its ulp ≈ 1e-13) the
     residual shrinks with the field — it is not an absolute floor;
   - the cascade re-quantizes at exactly this scale on the very next operation anyway
     (`running_sum = perturbation + mean` is computed at field magnitude), so nothing the
     pipeline preserves is lost.
   Recovering the CPU's residual would require materializing both clip caps *and* keeping
   the original perturbation, i.e. 5 device buffers ≈ 9 GiB instead of 5.4 — against the
   stated priority. **Flagging it for Thomas rather than deciding it silently.** Note the
   report's own 7.2 GiB formulation has the identical residual, since it also applies the
   final result in shifted space.

---

## Statistical acceptance gate (items 3 + 4)

### Ensemble: 8 seeds, 512×512, dx = 3 km, outer 768 km (7 classes), ℓ_s = 10 m, icon_lem snap0, anchored bounds, `device='cuda'`

`scratchpad/ens_run.py` + `ens_table.py`. "old" = `head_repo` (= HEAD; items 1–2 are
bit-identical so HEAD *is* HEAD-before-items-3/4). ξ(q) are horizontal Haar fluctuation
exponents of h from `scaleinvariance.haar_fluctuation` (order [1, 2], periodic, lags
'powers of 1.2'), fitted over 4 ≤ lag ≤ 64 cells (12–192 km), at three mid-levels.
Acceptance: |new mean − old mean| ≤ 2 × (old ensemble standard error).

| quantity | old mean | old sd | new mean | new sd | diff | \|d\|/SE | |
|---|--:|--:|--:|--:|--:|--:|:--|
| h level-mean @ z≈2 km | 318347.5 | 29.33 | 318347.5 | 29.34 | 8.8e-03 | 0.00 | PASS |
| h level-mean @ z≈6 km | 323153.4 | 453.7 | 323153.4 | 453.7 | 7.4e-03 | 0.00 | PASS |
| h level-mean @ z≈12 km | 331242.4 | 352.3 | 331242.4 | 352.3 | 6.5e-03 | 0.00 | PASS |
| h level-mean, RMS over z | 338792.7 | 161.7 | 338792.7 | 161.7 | 3.4e-03 | 0.00 | PASS |
| h level-std @ z≈2 km | 2493.770 | 155.2 | 2493.771 | 155.2 | 9.6e-04 | 0.00 | PASS |
| h level-std @ z≈6 km | 3820.113 | 468.8 | 3820.113 | 468.8 | 1.8e-04 | 0.00 | PASS |
| h level-std @ z≈12 km | 3498.143 | 570.9 | 3498.142 | 570.9 | −1.1e-03 | 0.00 | PASS |
| h level-std, RMS over z | 6368.219 | 94.42 | 6368.219 | 94.42 | −5.5e-05 | 0.00 | PASS |
| qt level-mean @ z≈2 km | 5.336465e-03 | 3.29e-05 | 5.336465e-03 | 3.29e-05 | −2.2e-11 | 0.00 | PASS |
| qt level-mean @ z≈6 km | 1.147478e-03 | 2.38e-04 | 1.147478e-03 | 2.38e-04 | 2.9e-11 | 0.00 | PASS |
| qt level-mean @ z≈12 km | 2.206806e-05 | 7.95e-06 | 2.206806e-05 | 7.95e-06 | 1.5e-12 | 0.00 | PASS |
| qt level-mean, RMS over z | 3.405146e-03 | 1.62e-05 | 3.405146e-03 | 1.62e-05 | 9.1e-11 | 0.00 | PASS |
| qt level-std @ z≈2 km | 2.272983e-03 | 4.17e-05 | 2.272983e-03 | 4.17e-05 | 1.3e-11 | 0.00 | PASS |
| qt level-std @ z≈6 km | 1.671442e-03 | 1.50e-04 | 1.671442e-03 | 1.50e-04 | −9.9e-11 | 0.00 | PASS |
| qt level-std @ z≈12 km | 7.651895e-05 | 1.67e-05 | 7.651895e-05 | 1.67e-05 | 8.6e-15 | 0.00 | PASS |
| qt level-std, RMS over z | 1.162182e-03 | 4.06e-05 | 1.162182e-03 | 4.06e-05 | −2.5e-11 | 0.00 | PASS |
| **flux volume mean** | 1.009406 | 0.01945 | 1.009406 | 0.01945 | 6.8e-09 | 0.00 | PASS |
| **flux clipped fraction** | 1.170681e-03 | 1.45e-05 | 1.170681e-03 | 1.45e-05 | **0** | 0.00 | PASS |
| ξ(1) of h, z-index 40 | 0.4363977 | 0.02426 | 0.4363976 | 0.02426 | −3.3e-08 | 0.00 | PASS |
| ξ(1) of h, z-index 57 | 0.5325791 | 0.02039 | 0.5325790 | 0.02039 | −7.7e-08 | 0.00 | PASS |
| ξ(1) of h, z-index 74 | 0.3615153 | 0.02552 | 0.3615153 | 0.02552 | −2.5e-08 | 0.00 | PASS |
| ξ(2) of h, z-index 40 | 0.3030775 | 0.09178 | 0.3030777 | 0.09178 | 2.5e-07 | 0.00 | PASS |
| ξ(2) of h, z-index 57 | 0.4285605 | 0.07262 | 0.4285597 | 0.07262 | −8.0e-07 | 0.00 | PASS |
| ξ(2) of h, z-index 74 | 0.2198697 | 0.04741 | 0.2198696 | 0.04741 | −2.2e-08 | 0.00 | PASS |

Every |diff|/SE rounds to 0.00 — the gate is not merely met, it is met by six orders of
magnitude, because the two codes track the **same realization** seed by seed rather than
merely the same distribution. Per-seed profile agreement:

| profile | max abs \|new−old\| | max rel |
|---|--:|--:|
| h level-mean | 0.0497 | 1.35e-07 |
| h level-std | 0.0352 | 1.84e-05 |
| qt level-mean | 3.24e-09 | 5.71e-04 |
| qt level-std | 6.15e-10 | 1.0 |

The `qt level-std max rel = 1` entry is a topmost level where q_t's spread is ~0 in one
run and ~6e-10 in the other; the absolute difference is 6e-10 kg/kg. Worth a glance but
not a signal.

That the roundoff perturbation does **not** amplify through the flux clip over 7 classes
here is a genuine (and mildly surprising) result — the `SUPPORT_FACTOR` comment's warning
about amplification is about larger perturbations and/or deeper cascades. It should not be
assumed to hold for the 12-class nest without checking.

Ensemble wall clock: old 5.4 s/member, new 4.4 s/member.

### Full test suite

```
./.venv/bin/python -m pytest tests/ -q   →   143 passed in 8.33 s
```

**No failures; nothing in `tests/` was edited.** Caveat for honest reading: the suite
exercises `device='cpu'` almost everywhere. The only CUDA-marked test is
`test_fft_xy_oa_z_cuda_matches_cpu`, which does cover the item-3 change (and passes);
**there is no test that touches `_bounded_amplitude_add_cuda`.** Its coverage in this
round is the function-level table and the ensemble above.

### Timing: one full production square, before vs after all four items

`scratchpad/run_square.py` (2048², dx = 3 km, outer 1536 km, ℓ_s = 10 m, `device='cuda'`,
seed 2000, `save_class_increments=True`, then `compute_diagnostics`), output to
`/var/tmp/steam-audit/`, wrapped in `scratchpad/timed.py`:

| | HEAD (098f3e1) | items 1–4 | change |
|---|--:|--:|--:|
| `simulate()` | 164.2 s | **125.3 s** | **−38.9 s (−23.7 %)** |
| `compute_diagnostics()` | 56.6 s | 56.7 s | — |
| square total | 220.9 s | **182.0 s** | −38.9 s (−17.6 %) |
| peak RSS | 17.02 GiB | 17.45 GiB | +0.43 GiB (CUDA context/allocator) |
| **peak VRAM** | 11.88 GiB | **5.88 GiB** | **−6.00 GiB** |
| output size | 10.86 GB | 10.86 GB | — |

The `_bounded_amplitude_add` offload accounts for ~34 s of the 39 s (3 finest-class calls
at ~16 s → ~1.5 s), the Lévy draw for ~4 s, the n_fft cap costs back a fraction of a
second. Peak VRAM is now set by the bounded-add's 5.39 GiB rather than the convolution's
11.88, which is the headroom the mid-flight note asked for.

---

## Files

- Changed (uncommitted): `steam/simulate.py`, `steam/utils.py`.
- Verification scripts: `scratchpad/verify1.py`, `verify2.py`, `verify3.py`, `verify4.py`,
  `ens_run.py`, `ens_table.py`, `cmp_nc.py`, `timed.py`, `run_head.py`.
- References: `scratchpad/head_repo/` (HEAD copy of `steam/`),
  `scratchpad/utils_old.py` (HEAD `steam/utils.py`).
- Ensemble summaries: `/var/tmp/steam-audit/ens_{old,new}_300{1..8}.npz`.
  All large netCDFs were deleted after analysis.

## Open questions for Thomas

1. The GPU bounded-add's per-level mean residual is half a float32 ulp of the field
   (0.016 J/kg for h) against the CPU's 3e-06. Argued above to be immaterial and
   unrecoverable below 9 GiB of VRAM — but it is a real relaxation of deposit condition 1
   and you should be the one to accept it.
2. The GPU path now enforces the bounds *strictly*; the CPU path can overshoot by a
   float32 ulp. Different behaviour between the square and the nest.
3. No test covers `_bounded_amplitude_add_cuda`. Worth one, if you want the dispatch
   defended.
