# STEAM round-4 optimization — implementation report

2026-07-30 · `turbulon-model` @ `c9d1dbb` (round 3 committed) and
`turbulon-analysis` @ `3df2bb5` · **all changes uncommitted**.

Goal of the round: **larger simulations** — a 4096² × 115 production-config
square on this box, and the arithmetic for larger nests.

Changed files:

- `turbulon-model`: `steam/simulate.py`, `steam/utils.py`,
  `tests/test_flux_cascade.py`, `tests/test_bounds.py` (two monkeypatched
  fakes and two call sites, because `_advance_flux`'s return changed).
- `turbulon-analysis`: `run_production_squares.py`, `regen_production.sh`.

Runners: `./.venv/bin/python` (model repo) for the test suite,
`turbulon-analysis/.venv/bin/python` for everything production-shaped.
HEAD reference is `scratchpad/head4/` (`git archive HEAD steam tests`),
driven through `scratchpad/run_head4.py`. Scratch in `/var/tmp/steam-audit/`.

**Headline: the 2048² production square is bit-identical to `c9d1dbb`
(57/57 variables) at 58.2 s against round-3's 60.4 s, and its peak RSS falls
16.61 → 15.78 GiB. The finest-class `_advance_flux` falls 11.77 → 8.61 GiB
(−27 %). The GPU bounded add now runs at 4096² × 115 inside 7.88 GiB of VRAM.
The 4096² square itself does NOT fit: it was OOM-killed in the finest class at
52.7 GiB RSS — see item 5 for the honest gap.**

---

## Item 1 — gamma/noise buffer reuse in `_advance_flux`, and the flux increment

Two changes in one function.

**(a) The expm1 chain runs in the generator's own buffer.** HEAD held
`gamma`, the `gamma - shift` temporary and `noise` simultaneously, and kept
the dead `gamma` alive to the end of the call. Now:

```python
generator = _sparse_levy(...)
generator *= per_class_scale
off_center = generator == 0.0   # the generator's own zeros: no turbulon
generator -= shift
np.expm1(generator, out=generator)
noise = generator               # the same buffer, now the multiplier noise
del generator
noise[off_center] = np.float32(0.0)
del off_center
```

Every operation is elementwise, so the arithmetic per cell is exactly what it
was. Three field-sized arrays became one plus a quarter-field boolean.

**I did NOT take the band-slicing / lattice-slicing variant** the brief and
the gpu-torch report suggested (`noise[:, :, :n] = 0` at s = 1). It is a
semantic change, not just an optimization: `noise[gamma == 0] = 0` zeroes any
cell whose generator draw is exactly 0.0, and `tests/test_flux_cascade.py`
asserts precisely that (`amplitude.ravel()[1] == 0.0`, an artificial gamma
containing a 0 with sparsity factors (1,1,1)). Structural slicing would make
that test fail, and would silently redefine "off-centre" from
"generator is zero" to "off the lattice". They differ only in a
probability-zero case, but it is a documented contract, so it is Thomas's
call, not mine. **The mask costs nothing at the peak anyway** — it is live
when 2.25 field-equivalents are held, while the call's peak is 6.

**(b) `_advance_flux` returns its own applied increment**, so
`flux_before = flux.copy()` in `cascade_loop` is gone. The trick that makes it
free: after `CONVOLVE` returns, the `noise` buffer is dead, so it holds the
entering flux for the final `np.subtract(flux, entering_flux, out=increment)`
— literally the same subtraction the caller used to do, hence bit-identical
increments (they feed nest continuation). Signature is now
`(scalar_amplitude, applied_increment, diagnostics)` with
`return_increment=False` by default; without it the second element is None,
`noise` is deleted before the clip/renorm, and nothing extra is held.

| check | result |
|---|---|
| 256² 7-class e2e, `device='cpu'`, all vars + all class increments vs HEAD | **44/44 bit-identical** |
| 256² 7-class e2e, `device='cuda'` | **44/44 bit-identical** |
| production finest class (2048, 2048, 115), `device='cuda'`, increment recorded: peak RSS | **11.769 → 8.608 GiB (−3.16 GiB, −27 %)** |
| same, wall | 5.68 → 4.87 s |
| same, blake2b of scalar amplitude / applied increment / flux | **all three identical** |
| strip nest (windowed) and wrapping strip nest, all groups | 60/60 bit-identical each |

Live field-sized arrays at the finest class, `_advance_flux` plus its caller:
**8 → 6** (HEAD: h, qt, flux, flux_before, gamma, gamma−shift, noise,
convolution result; now: h, qt, flux, noise, S_k, increment).

A third, free change rode along: `noise * flux` was computed twice (once for
`scalar_amplitude`, once in place) and is now computed once, since
`scalar_amplitude` is that same product scaled by `1/mean_abs`. Elementwise,
so bit-identical.

## Item 2 — `running_sum` reused as the `|W|` scratch

`_bound_taper` consumes `running_sum`, so its buffer is dead; the level
normalization now does `np.abs(inner, out=inner_absolute)` into it instead of
allocating a fresh array.

**Bit-identical, root and nest.** A sum over axes (0, 1) of an (x, y, z) array
accumulates in index order, which is the same whether the operand is the
contiguous buffer or a window's strided view of it — verified directly
(contiguous, x-strided and y-strided windows all give `array_equal` sums), and
end to end by the 256² square (44/44), the strip nest (60/60) and the
wrapping strip nest (60/60).

**Honest correction to codex#7's projection:** this is a *speed* item, not a
peak item. HEAD already did `del running_sum` before allocating the `np.abs`
temporary, so the two never coexisted and the peak is unchanged. What it buys
is the allocation and its page faults: at (2048, 2048, 115), 0.493 → 0.265 s
per field per class (measured, best of 3), i.e. ~2–4 s per production square.

## Item 3 — level-batched GPU bounded add

`_bounded_amplitude_add_cuda` is now a driver that walks z in batches and
calls `_bounded_amplitude_add_cuda_levels` (the previous body, unchanged) on
each batch's slabs. The host side of a slab is a strided view; torch uploads
and downloads it directly.

`cuda_level_batch_size` (new, `steam/utils.py`) sizes the batch from a **fixed**
budget, `CUDA_BOUNDED_ADD_BUDGET_BYTES = 8 GiB`, not from the free VRAM of the
moment, and raises `MemoryError` rather than shrinking if the batch will not
fit. That is deliberate and is the one design decision here worth Thomas's
attention: **the batch size is part of the realization** — the per-level means
are float32 reductions whose value depends on how many levels share the
reduction (measured: batching a (2048,2048,115) reduction into 64s changes the
level means at ~1e-6 relative), so sizing it from whatever else happened to be
on the card would make the same seed give different fields on different days.
8 GiB is chosen so that the 2048² × 115 production square (5.78 GiB) is a
single batch and reproduces the pre-batching realization exactly.

**No pinned staging buffer**, against the brief's suggestion: a pinned buffer
of a batch is 1.9–2.7 GiB of *host* RSS held at exactly the moment — the
finest class — where host RSS is what limits the domain size, and the measured
prize was ~0.16 s per call. Memory is the binding constraint, so I took the
transfers pageable.

| shape | | |
|---|---|---|
| (2048, 2048, 115), cuda | HEAD 1.65 s / 5.39 GiB VRAM | new **1.65 s / 5.39 GiB**, field and applied-increment digests **identical** (single batch) |
| (4096, 4096, 115), cuda | HEAD: impossible (3 × 7.19 = 21.6 GiB VRAM) | new **10.80 s, peak VRAM 7.88 GiB**, 42 levels/batch × 3 |
| (4096, 4096, 115), cpu, same inputs | | 86.84 s |
| GPU vs CPU at 4096², per-level delivered amplitude | | max rel diff **5.4e-6**, mean 5.2e-7 |
| GPU vs CPU at 4096², field | | ⟨\|δ\|⟩ 2.42e-2 on mean\|·\| 8972 = **2.7e-6 relative** |

Comfortably inside the round-2 bar (1e-4), and 8.0× faster than the CPU solve
at a shape the card previously could not touch at all.

## Item 3b — streamed CUDA convolution (NOT in the brief; flagged for a ruling)

Discovered while sizing item 5: at 4096² × 115 the GPU convolution cannot run
either, and no amount of `n_fft` shrinking helps, because the field (7.19 GiB)
and the overlap-add accumulator (7.83 GiB) are both resident — 15.5 GiB before
a single spectrum, on a card with 14.5 GiB free. `cuda_block_fft_size` would
have raised `MemoryError` at the finest class.

So `convolve_fft_xy_oa_z` gained a streamed path: the same overlap-add blocks,
one at a time, with the field and the accumulator left on the host and each
block's real output added into the (already trimmed) host output. Nothing
field-sized is resident. `cuda_block_fft_size` now returns `(n_fft, streamed)`
and tries the resident ladder first, exactly as before, falling to the streamed
ladder only when nothing resident fits.

| check | result |
|---|---|
| (1024, 1024, 115) resident vs forced-streamed | **bit-identical**; 0.22 → 0.47 s; peak VRAM 1.47 → 0.60 GiB |
| (2048, 2048, 115) resident vs forced-streamed | **bit-identical**; 0.63 → 1.80 s; peak VRAM 5.88 → 2.41 GiB |
| plan chosen at (2048, 2048, 115) | `(32, False)` — resident, i.e. production is untouched |
| plan chosen at (4096, 4096, 115) | `(32, True)` — streamed |
| plan chosen at the production nest shape (16384, 140, 364) | `(32, False)` — resident, 7.7 GiB |
| plan at (8192, 8192, 115) | clean `MemoryError`, no silent fallback |

It is bit-identical, it is not 4096-specific (the same budget ladder decides
it), and it degrades nothing. But it is unbriefed and it touches the
convolution, so: **Thomas's call.** It is cleanly separable — one constant, one
function, and two hunks in `steam/utils.py` — and reverting it means reverting
`cuda_block_fft_size`'s two-value return with it. Nothing else in round 4
depends on it, since the 4096² square failed on host RAM anyway (item 5).

## Item 4 — production driver serialized

`run_production_squares.py::main` now runs each member's square and then that
member's own nest, in one process, in order; the `ProcessPoolExecutor` and the
`multiprocessing` import are gone. Restartability is untouched (skip-if-exists
on both halves). `regen_production.sh`'s co-residency paragraph is trimmed to
the safety-net sentence it now is, and the cap moves 50G → 55G (one process,
one peak).

The nest still runs `device="cpu"` — see the nest section below for the
measured cuda comparison. **The default was not flipped**; that is Thomas's
call.

## Item 5 — ACCEPTANCE TEST: real 4096² × 115 square. **It does not fit.**

Configuration exactly as `run_production_squares.py::run_square` with
nx = ny = 4096, dx = 3 km (a 12288 km square), outer scale = domain/4 =
3072 km → 10 size classes, constant 10 m spheroscale, `icon_lem` snap0,
anchored bounds, seed 2000, `save_class_increments=True`, `compress=True`,
`device="cuda"`, then `compute_diagnostics`. Nothing weakened: same algorithm,
same precision, all phases, no 4096-specific branch anywhere in the model.
Run under `systemd-run --user --scope -p MemoryMax=57G -p MemorySwapMax=0`.

**Stated before running** (the probe prints it):

| | |
|---|---:|
| one field-sized array at (4096, 4096, 115) | **7.19 GiB** |
| `simulate()`'s own preflight estimate (8 × field) | 57.5 GiB |
| host available at launch | 54.1 GiB |
| → `simulate()` would REFUSE the run | (overridden in the probe, cgroup cap as the guard) |
| live-array arithmetic, `_advance_flux` at the convolution (h, qt, flux, noise, S_k, increment) | 6 × field = 43.1 GiB |
| live-array arithmetic, **gradient stage** (h, qt, flux, S_k, running_sum, grad_h, grad_z) | **7 × field = 50.3 GiB** |
| live-array arithmetic, scalar convolution (h, qt, flux, S_k, W, increment) | 6 × field = 43.1 GiB |

**Result: OOM-killed, in class 10/10, grid (4096, 4096, 115), during
"computing G, convolutions"** — i.e. in the gradient stage, the predicted
7-field peak, on the first scalar of the finest class. Classes 1–9 completed
normally (class 9 is (2048, 2048, 78)).

| measurement | |
|---|---:|
| wall clock to the kill | 86 s |
| **peak process RSS (VmHWM)** | **52.58 GiB** = **7.31 × field** |
| cgroup peak (systemd), incl. page cache of the staging writes | 55.8 GiB |
| peak VRAM | 13.96 GiB of 15.46 (streamed convolution + bounded add) |
| RSS trajectory | 10.5 GiB (class ~7) → 33.6 GiB (class 9) → **52.6 GiB (class 10)** → killed |
| increment staging on disk at the kill | 19 GB |
| diagnostics | never reached |

**The measured peak matches the arithmetic to 4 %**: 7 × 7.19 = 50.3 GiB of
fields plus ~2.3 GiB of interpreter, torch, CUDA context and allocator.

**The honest gap.** Two independent shortfalls, both real:

1. **RAM.** 52.6 GiB of RSS against 54.1 GiB available on a 60 GiB box — it
   very nearly fits, but "nearly" with 1.5 GiB of margin is not a production
   configuration, and the cgroup also has to hold the page cache of a 26 GB
   staging directory and a 47 GB output file being written. That is what
   actually killed it. Then `compute_diagnostics` (11.1 GiB at 2048²) would
   have to run afterwards.
2. **Disk.** The complete output is ~47 GB (2048² is 11.90 GB on disk; the
   uncompressed content scales ×4) and the per-class `.npy` staging peaks at
   ~26 GB, and the two are live at the same moment during the write phase:
   **~73 GB against 65 GB free on a filesystem at 94 %.** This one is not
   marginal and no memory work fixes it.

**What would close the RAM gap:** one field-equivalent, ~7.2 GiB, is available
at exactly the stage that peaked. `_gradient_components` writes `grad_h` and
`grad_z`, and the caller then forms `W = grad_z * aspect + grad_h`. A single
numba kernel writing `W = |∇_h φ| + (ℓ_z/ℓ_x)|∂φ/∂z|` directly would hold
`running_sum` and `W` instead of `running_sum`, `grad_h`, `grad_z` — 7 fields
→ 6 at the peak, 52.6 → ~45.4 GiB, which would fit with ~9 GiB of margin. It
is the same kind of change as round-3's gradient fusion and would be
bit-identical only if the three operations are done in the same order and
precision as now (`W = grad_z; W *= aspect; W += grad_h`), which is
arrangeable. **Recommended as the round-5 headline item.** It does not touch
the disk problem.

## The production nest, measured — and the cuda question from item 4

Real production strip nest (parent 2048² m00-equivalent, spanning x, 48 km
wide, dx = 375 m; finest class (16384, 140, 364), field = 3.11 GiB),
`refine(..., compress=True)`, one process, `MemoryMax=55G`:

| | wall | peak RSS | peak VRAM |
|---|---:|---:|---:|
| `device="cpu"` (production default) | **132.4 s** | **23.43 GiB** | — |
| `device="cuda"` | **64.0 s** | 26.77 GiB | 9.32 GiB of 15.46 |

For scale, the pre-round-3 audit measured this nest at 657 s and 31.5 GB
(29.3 GiB). Rounds 3 and 4 together take it to 132 s and 23.4 GiB.

**Answer to the item-4 question: yes, the nest fits on the card, and it is
2.07× faster end to end — but it costs +3.34 GiB of host RSS** (one
field-equivalent: the device buffers do not replace the host arrays, and the
CUDA context and caching allocator add their own). Host RSS is exactly what
limits how big a nest can be, so this is a speed-versus-size trade, not a free
win. **The default is NOT flipped — Thomas's call.** (The cuda nest is also
realization-changing relative to the cpu one, per round 2.)

## Item 6 — nest-enlargement arithmetic (ESTIMATED, no runs)

Model, calibrated on this round's three real measurements:

**peak RSS ≈ 7 × (one field) + 2.3 GiB**

The 7 is the gradient stage of the finest class (h, qt, flux, S_k,
running_sum, grad_h, grad_z), the same stage that peaked in every run this
round; the 2.3 GiB is interpreter + torch + CUDA context + allocator.
Calibration:

| run | field | predicted | measured |
|---|---:|---:|---:|
| 2048² square (cuda, increments) | 1.80 GiB | 14.9 GiB | 15.78 GiB |
| 4096² square (cuda, increments) | 7.19 GiB | 52.6 GiB | 52.58 GiB |
| production nest (cpu) | 3.11 GiB | 24.1 GiB | 23.43 GiB |

Nest geometry: x is the full 6144 km parent span; y is the strip's own width
plus a 12-cell halo (the kernel half-width, resolution-independent); nz grows
by 2^H_z = 1.474 per octave, not by 2 (measured 115 → 364 over three octaves,
and 53 → 78 → 115 within the square).

| variant | finest grid | field | **estimated peak RSS** | fits under 55 GB (51.2 GiB)? |
|---|---|---:|---:|---|
| **(a) current strip** — 48 km, 375 m | 16384 × 140 × 364 | 3.11 GiB | **24.1 GiB** (measured 23.43) | **yes**, with room |
| **(b) one more octave** — 48 km, 187.5 m | 32768 × 268 × 537 | 17.6 GiB | **125 GiB** | **no**, by 2.5× |
| **(c) 2× wider strip** — 96 km, 375 m | 16384 × 268 × 364 | 5.96 GiB | **44.0 GiB** (47.2 GB) | **yes**, ~7 GiB spare |
| **(d) both** — 96 km, 187.5 m | 32768 × 524 × 537 | 34.4 GiB | **243 GiB** | **no**, by 5× |

Two derived numbers worth having:

- **Widest strip that fits at 375 m:** field ≤ 6.99 GiB → 314 y-cells total →
  **~113 km wide** (2.4× today's 48 km). With the proposed 6-field gradient
  fusion, ~133 km.
- **Depth is the expensive axis**: one octave costs ×5.66 in memory
  (2 × 2 × 1.474 with the halo diluted), so no plausible constant-factor
  saving buys another octave. Another octave of nest needs a bigger machine,
  not better bookkeeping.

(Disk, for completeness: variant (c) roughly doubles the nest's share of the
output, ~+5 GB per member.)



---

## Final gate

### Test suite

```
./.venv/bin/python -m pytest tests/ -q   →   144 passed in 9.6 s
```

Two test files were edited, both because `_advance_flux`'s return became a
3-tuple: the two real call sites in `test_flux_cascade.py` and the two
monkeypatched `fake_advance` stand-ins in `test_flux_cascade.py` and
`test_bounds.py` (which now also accept `return_increment`). No assertion was
weakened or removed.

### End-to-end bit-identity vs `c9d1dbb`, combined tree

| check | result |
|---|---|
| 256² 7-class square, `device='cpu'`, all vars + all class increments | **44/44 bit-identical** |
| 256² 7-class square, `device='cuda'` | **44/44 bit-identical** |
| strip nest (windowed), all groups | **60/60 bit-identical** |
| wrapping strip nest (y from 0, two contiguous runs) | **60/60 bit-identical** |
| **2048² production square, `device='cuda'`, `save_class_increments=True`** | **57/57 bit-identical** (all 9 classes' h/qt/flux increments, h/qt/flux, all five diagnostics) |

### Production square, 2048², vs round 3

| | round 3 | round 4 | change |
|---|--:|--:|--:|
| `simulate()` | 35.9 s | **35.0 s** | −0.9 s |
| `compute_diagnostics()` | 23.7 s | **23.2 s** | −0.5 s |
| **square total** | **60.4 s** (re-measured 59.6 s today on HEAD: 58.9 s) | **58.2 s** | no regression |
| **peak RSS** | 16.61 GiB | **15.78 GiB** | **−0.83 GiB** |
| peak VRAM | 5.88 GiB | 5.88 GiB (allocator 5.39, reserved 6.49) | — |
| output size | 11.90 GB | 11.90 GB | — |

HEAD re-measured today at 58.9 s / same output, so the comparison is
like-for-like on the same box in the same session; round 3's published 60.4 s
was a slightly noisier run of the same code.

Field-equivalents of peak RSS (the number that matters for scaling up):

| nx | field | peak RSS | field-equivalents |
|---|---:|---:|---:|
| 1024 | 0.45 GiB | 3.75 GiB | 8.34 |
| 2048 | 1.80 GiB | 15.78 GiB | 8.78 |
| 4096 | 7.19 GiB | 52.58 GiB | 7.31 (increments on; killed here) |

## Open items for Thomas

1. **The streamed CUDA convolution (item 3b) is unbriefed.** Bit-identical,
   not 4096-specific, cleanly separable. Keep or revert — but note that
   reverting it re-imposes a hard ceiling: no cuda convolution above ~2.4 field
   GiB, which also caps how wide a nest can get on the card.
2. **`CUDA_BOUNDED_ADD_BUDGET_BYTES = 8 GiB` is a realization-defining
   constant.** Changing it changes the batching and hence the level means at
   ~1e-6. It is set so the 2048² square stays a single batch. If it is ever
   raised or lowered, existing 4096²-class output stops being reproducible.
3. **`noise[gamma == 0] = 0` was kept as a mask**, not converted to band or
   lattice slicing, because the slicing variant redefines "off-centre" and
   breaks a test that asserts the current contract. The mask is not at the
   peak, so keeping it costs nothing. Your call whether the contract should
   change.
4. **4096² is out of reach on this box** on two counts (RAM by ~1.5 GiB of
   margin plus page cache; disk by ~8 GB), and the single highest-value next
   change is the fused gradient+W numba kernel (7 → 6 fields at the peak).
5. **Nest on cuda: 2.07× faster, +3.34 GiB host RSS.** Not flipped.

## Reproduction

```
scratchpad/verify_advance_flux.py {old|new} [nx ny nz]     # item 1
scratchpad/verify_bounded_batch.py {cuda|cpu|both} [nx ny nz]  # item 3
scratchpad/verify_stream_conv.py [nx ny nz]                # item 3b
scratchpad/run_square.py OUT.nc [--instrument] [--nx N]    # 2048^2 gate
scratchpad/run_square_4096.py OUT.nc [--preflight-only]    # item 5
scratchpad/run_nest_prod.py PARENT.nc {cpu|cuda}           # nest
scratchpad/e2e_square.py / e2e_square_cuda.py / e2e_nest.py / e2e_nest_wrap.py
scratchpad/cmp_nc_all.py A.nc B.nc
scratchpad/run_head4.py SCRIPT.py ...    # any of the above against head4/
```
