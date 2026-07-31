# STEAM round-5 optimization — implementation report

2026-07-30 · `turbulon-model` @ `255f6c2` (round 4 committed) ·
**all changes uncommitted**. Driver repo `turbulon-analysis` @ `23ef1fa`,
untouched this round.

Changed files: `turbulon-model/steam/simulate.py` only. No test file needed
editing.

Runners: `./.venv/bin/python` (model repo) for the suite,
`turbulon-analysis/.venv/bin/python` for everything production-shaped.
HEAD reference tree is `scratchpad/head5/` (`git archive 255f6c2 steam tests`),
driven through `scratchpad/run_head5.py`. Scratch in `/var/tmp/steam-audit/`.

---

## Item 1 — one fewer live field at the cascade's peak instant

### 1a. The fused advective weight (as briefed)

`cascade_loop` did

```python
grad_h, grad_z = _gradient_components(running_sum, dx_k, dy_k, z_k)
W = grad_z
W *= aspect_k[i]    # 1D broadcast, one value per level
W += grad_h
del grad_h
```

— two full-size arrays out of the stencil, immediately collapsed into one.
It now does

```python
W = _advective_weight(running_sum, dx_k, dy_k, z_k, aspect_k[i])
```

`_advective_weight_stencil` is `_gradient_stencil` with the horizontal
magnitude parked in the OUTPUT array rather than a second buffer, and the
z pass finishing the combination in place. Per cell the arithmetic is
`grad_z * aspect_z[iz] + grad_h`, in float32, in that order — the same three
stored float32 values the two-array path produced, so it is exact rather than
merely equivalent. The z-coefficient preparation both entry points share moved
into `_z_difference_coefficients`, so the two cannot drift apart.
`_gradient_components` is unchanged and still exported (the tests use it).

`aspect_z` is validated to be one value per level rather than silently
broadcast.

| shape | z branch | `np.array_equal(W_old, W_new)` | two-step → fused |
|---|---|---|---|
| (2048, 2048, 115) production finest | uniform | **True** | 0.452 → 0.144 s |
| (2048, 2048, 115) | non-uniform | **True** | 0.457 → 0.144 s |
| (16384, 140, 364) production nest finest | uniform | **True** | 0.680 → 0.174 s |
| (16384, 140, 364) | non-uniform | **True** | 0.693 → 0.173 s |
| (256, 256, 115) | both | **True** | 0.007 → 0.003 s |
| (8, 8, 6) tiny | both | **True** | — |

(3.1–3.9× faster, because the fused kernel writes the output once instead of
writing two arrays and then reading both back for `*=` and `+=`.)

### 1b. What the measurement then said — and the second change it forced

Item 1a alone did **not** move the process peak: 2048² VmHWM went
15.68 → 15.60 GiB. Sampling RSS at 20 ms and attributing it to the enclosing
call showed why. Round 4 inferred that the peak stage was the gradient
(the 4096² OOM printed "computing G, convolutions", which is the banner for
the whole h/qt loop). Measured, the loop's high-water is:

| call, finest class, 2048² | HEAD peak within the call | after 1a |
|---|---:|---:|
| `_advance_flux` | 13.46 GiB | 13.35 |
| `_gradient_components` / `_advective_weight` | **15.68** | 13.63 |
| `_bound_taper` | **15.69** | **15.58** |
| `CONVOLVE` (scalar) | 12.08 | 11.93 |
| `_bounded_amplitude_add` | 12.09 | 11.98 |

`_bound_taper` ties the gradient stage at HEAD and becomes the sole peak once
the gradient is fused, because of one line:

```python
np.minimum(g, np.float32(phi_max - phi_min) - g, out=g)
```

`span - g` is a whole reflected copy of the field. So the taper now takes it
an **x-slab at a time** (x is the slowest axis, so a slab is contiguous
memory); `TAPER_SLAB_BYTES = 64 MiB` sets how many rows one numpy call
covers. Every operation in `_bound_taper` is elementwise, so slabbing changes
the size of the temporary and nothing else — not a single bit.

(First attempt slabbed along z. Bit-identical, but a z-slab is a strided view
with 4-float contiguous runs, and `_bound_taper` went 0.62 → 3.1–4.3 s. The
x-slab version runs at 0.60–0.62 s against HEAD's 0.74–0.82 s.)

### Result

| finest class, 2048², peak RSS within the call | HEAD | round 5 |
|---|---:|---:|
| gradient stage | 15.68 GiB | **13.63** |
| bound taper (the binding stage) | 15.69 GiB | **13.95** |
| whole simulate(), sampled | 15.69 GiB | **13.95** |

Exactly one field-sized array (1.797 GiB at 2048²) removed from the
process peak. Live full-size arrays at the h/qt loop's worst instant:
**7 → 6** (h, qt, flux, S_k, running_sum/g, W; HEAD additionally held either
`grad_h`+`grad_z` or the taper's reflected copy).

### Bit-identity

| check | result |
|---|---|
| 256² 7-class e2e, `device='cpu'`, all vars + all class increments | **44/44 bit-identical** |
| 256² 7-class e2e, `device='cuda'` | **44/44 bit-identical** |
| strip nest (windowed), all groups | **60/60 bit-identical** |
| wrapping strip nest (y from 0, two contiguous runs) | **60/60 bit-identical** |
| **2048² production square, cuda, `save_class_increments=True`** | **57/57 bit-identical** |
| test suite | 144 passed |


### Production square regression, 2048², vs `255f6c2`

| | HEAD `255f6c2` (re-measured today) | round 5 |
|---|--:|--:|
| `simulate()` | 33.9 s | **33.0 s** |
| `compute_diagnostics()` | 22.9 s | **23.2 s** |
| **square total** | **56.8 s** | **56.2 s** |
| **peak RSS (VmHWM)** | 15.68 GiB | **13.87 GiB** (−1.81 = one field) |
| output size | 11.90 GB | 11.90 GB |

Field-equivalents of peak RSS, the number that matters for scaling up:
**8.73 → 7.72** at 2048².

---

## Item 2 — 4096² × 115 ACCEPTANCE TEST: **it fits, and it completed**

### Where the output went

`df -h` at the start of the round: **`/` (btrfs, `/dev/nvme0n1p3`, 929 G) had
153 G free** — the storage Thomas freed. There is no external mount:
`/run/media/thomas`, `/mnt` and `/media` hold nothing with a filesystem
(`lsblk` shows only `nvme0n1` and the zram swap). So the run went to
**`/var/tmp/steam-audit/` on `/`**, which is also where `simulate()` put its
per-class increment staging.

### Configuration — identical to round-4 item 5, nothing weakened

nx = ny = 4096, dx = 3 km (a 12288 km square), outer scale = domain/4 =
3072 km → **10 size classes**, constant 10 m spheroscale, `icon_lem` snap0,
anchored bounds, seed 2000, `save_class_increments=True`, `compress=True`,
`device="cuda"`, then `compute_diagnostics(compress=True)`. Run under
`systemd-run --user --scope -p MemoryMax=58G -p MemorySwapMax=0`.

### Preflight, stated before running

| | |
|---|---:|
| one field-sized array at (4096, 4096, 115) | **7.19 GiB** |
| `simulate()`'s own preflight estimate (8 × field) | 57.5 GiB |
| host available at launch | 54.0 GiB |
| → `simulate()` still REFUSES the run | overridden in the probe; the cgroup cap is the guard |
| live arrays, `_advance_flux` at the convolution | 6 × field = 43.1 GiB |
| live arrays, **gradient stage, now fused** | 6 × field = 43.1 GiB |
| live arrays, **bound taper, now slabbed** | 6 × field = 43.1 GiB |
| live arrays, scalar convolution | 6 × field = 43.1 GiB |

### Result — **COMPLETED, exit 0**

| measurement | round 4 (`255f6c2`) | **round 5** |
|---|---:|---:|
| outcome | **OOM-killed in class 10/10** | **completed** |
| `simulate()` wall | — (killed at 86 s) | **170.5 s** |
| `compute_diagnostics()` wall | never reached | **99.3 s** |
| **total wall** | — | **269.7 s** |
| **peak process RSS (VmHWM)** | 52.58 GiB (7.31 × field) | **51.55 GiB (7.17 × field)** |
| output file | — | **46.45 GB** |
| diagnostics | never reached | **complete: T, p, qc, qv** |
| class increments | — | **all 10 groups (c00…c09), h/qt/flux each** |

Wall for reference: the 2048² square is 56.2 s, so 4096² is 4.8× the work for
4× the cells.

### Sanity check on the output

`scratchpad/sanity_square.py`, against a 2048² production square from the same
code (seed 4242, `/var/tmp/steam-audit/parent4242.nc`):

| | 4096² | 2048² reference |
|---|---|---|
| h per-level mean | 3.183e5 … 4.084e5, all finite | 3.184e5 … 4.084e5 |
| h per-level std | 145.9 … 1.348e4, all finite | 173.1 … 1.360e4 |
| qt per-level mean | 2.75e-7 … 1.301e-2, all finite | 3.01e-7 … 1.301e-2 |
| qt per-level std | 3.46e-6 … 3.479e-3, all finite | 1.94e-6 … 2.902e-3 |
| cloud fraction | max **0.182 at z = 4348 m**, nonzero on 88 of 115 levels | (diagnostics not run on the reference) |

Profile-shape agreement, 4096² interpolated onto the reference's z:

| profile | correlation | relative RMS |
|---|---:|---:|
| h mean | **+1.0000** | 0.0002 |
| h std | +0.9525 | 0.0921 |
| qt mean | +0.9999 | 0.0032 |
| qt std | +0.9779 | 0.0917 |

The means match to 2–3 significant figures; the ~9 % std difference is
expected and in the right direction — the 4096² square has **one more size
class** (10 vs 9) and twice the outer scale, so it carries more variance.
Cloud fraction peaking at 18 % near 4.3 km with a cloud-free layer above and
below is a plausible profile.

**The output file was deleted after the report was written** (test artifact).

---

## Item 3 — narrow-strip +1-octave nest

### (a) Predicted peak RSS by strip width and depth — MEASURED GEOMETRY

The grids are not assumed: `scratchpad/probe_nest_grids.py` builds exactly the
arguments `refine()` builds for an x-spanning strip (`pad_x = 0` because the
strip spans the periodic x axis, `pad_y = SUPPORT_FACTOR·k` per class) and
calls `_compute_all_grids` itself. Parent = the production 2048² square:
6144 km span, dx = 3 km, 9 classes, finest parent turbulon **k = 6000 m**,
gap factor 2 (so one nest class per octave).

Confirmed against the code rather than assumed:

- **x** = 6144 km / dx, and it does double per octave: 16384 → 32768.
- **y** = width/dx + **12 halo cells**, and the 12 is the same at every width
  and every depth (the halo is `3·k` physical, which is a fixed cell count).
- **nz** grows 364 → 535 per octave, i.e. **×1.47 = 2^{H_z}**, not ×2 —
  matching round 4's 2^{5/9} rule.
- So one octave costs **2 × 1 × 1.47 = 2.94×** in cells for a narrow strip
  where the halo dominates y, rising toward round-4's 5.66× for wide strips
  where y also doubles. At 48 km: 3.11 → 17.50 GiB is 5.63×. At 6 km:
  0.62 → 2.87 GiB is only **4.6×**, because the 12 halo cells are a bigger
  share of a narrow strip and they do not shrink.

| width | finest dx | classes | finest grid | field | round-4 model (7×+2.3) | round-5 model (6×+2.3) |
|---|---|---:|---|---:|---:|---:|
| 48 km | 375 m | 3 | 16384 × 140 × 364 | 3.11 GiB | 24.1 GiB | **21.0 GiB** |
| 24 km | 375 m | 3 | 16384 × 76 × 364 | 1.69 GiB | 14.1 | **12.4** |
| 12 km | 375 m | 3 | 16384 × 44 × 364 | 0.98 GiB | 9.1 | **8.2** |
| 6 km | 375 m | 3 | 16384 × 28 × 364 | 0.62 GiB | 6.7 | **6.0** |
| **48 km** | **187.5 m** | 4 | 32768 × 268 × 535 | 17.50 GiB | 124.8 | **107.3** ✗ |
| **24 km** | **187.5 m** | 4 | 32768 × 140 × 535 | 9.14 GiB | 66.3 | **57.2** ✗ |
| **12 km** | **187.5 m** | 4 | 32768 × 76 × 535 | 4.96 GiB | 37.0 | **32.1** ✓ |
| **6 km** | **187.5 m** | 4 | 32768 × 44 × 535 | 2.87 GiB | 22.4 | **19.5** ✓ |

Under a ~50 GiB working budget, **+1 octave becomes reachable at 12 km and
6 km**, and only there — 24 km is 57 GiB even on the optimistic model, and
48 km (round-4's variant (b), 125 GiB) stays out of reach by 2×.

### (b) The two physics costs of narrowing, quantified

**(i) How many outer-nest-class turbulons span the width.** The nest's
largest class is half the parent's finest, **k = 3000 m** (then 1500, 750,
375, and at +1 octave 187.5). Supports across the strip = width / 3000 m:

| width | supports of the widest nest class across the strip |
|---|---:|
| 48 km | 16.0 |
| 24 km | 8.0 |
| 12 km | **4.0** |
| 6 km | **2.0** |

Independent of the finest dx — class 0 of the nest is the same class either
way. At 6 km the widest nest turbulon fits across the strip exactly twice; at
12 km, four times.

**(ii) Samples entering the joint W·S_k per-level normalization at the widest
nest class** (n_y inner cells × n_x centres at that class's own grid), and the
noise on a mean-absolute over N samples, which scales as N^{-1/2}:

| width | N per level at k = 3000 m | noise relative to today's 48 km |
|---|---:|---:|
| 48 km | 131 072 | 1.00× |
| 24 km | 65 536 | 1.41× |
| 12 km | 32 768 | **2.00×** |
| 6 km | 16 384 | **2.83×** |

Again independent of depth. So the narrow-deep options buy an octave at the
price of a 2–2.8× noisier level normalization at the nest's coarsest class,
and 4 or 2 supports across the strip. **These are Thomas's to weigh, not
mine.**

### (c) The real measurement

A fresh 2048² production-config parent, **seed 4242**, increments on, cuda:
32.4 s, 8.38 GB, peak RSS 13.91 GiB (`/var/tmp/steam-audit/parent4242.nc`).
Two nests were then run from it, both `device="cpu"`, both under
`systemd-run --user --scope -p MemoryMax=50G -p MemorySwapMax=0`.

The candidate follows the brief's rule — the **narrowest width from (a) that
both fits and keeps ≥ 2 outer-nest-class supports across the strip**, which is
**6 km** (2.0 supports exactly; 12 km would give 4.0).

| | reference geometry | **candidate** |
|---|---|---|
| strip width | 48 km (16 parent cells) | **6 km (2 parent cells)** |
| finest dx | 375 m | **187.5 m** |
| nest size classes | 3 (3000, 1500, 750 m) | **4 (3000, 1500, 750, 375 m)** |
| finest grid | 16384 × 140 × 364 | **32768 × 44 × 535** |
| one field | 3.11 GiB | 2.87 GiB |
| **wall** | **126.5 s** | **110.7 s** |
| **peak RSS, measured** | **25.49 GiB** | **23.49 GiB** |
| predicted, 6×field + 2.3 | 21.0 | 19.5 |
| predicted, **7×field + 3.5** (recalibrated below) | **25.3** | **23.6** |

**A correction to round 4's nest description**: the current 48 km / 375 m nest
carries **3** classes, 3000 → 750 m, not the "3000/1500/750/375" of round-4
item 6. The +1-octave nest is the one with a 375 m class.

**Why the model needed recalibrating, and what actually binds a nest.** The
round-5 saving reaches the *square's* peak in full (−1.81 GiB of 1.80), but
the reference nest only fell **26.46 → 25.49 GiB** against HEAD (a like-for-like
HEAD run on a copy of the same parent). Tracing the stages of the nest
(`scratchpad/probe_nest_stages.py`, 20 ms sampler attributed to the enclosing
call) says why:

| finest class (16384, 140, 364), round-5 code | peak RSS within the call |
|---|---:|
| **`_advance_flux`** | **25.28 GiB** ← the peak |
| `CONVOLVE` (scalar) | 23.57 |
| `_bound_taper` | 23.30 |
| `_advective_weight` | 23.01 |
| `_bounded_amplitude_add` | 20.51 |

The h/qt loop, which is what item 1 shrank, is now **below** `_advance_flux`.
For nests the binding stage has moved to `_advance_flux` — six live fields
plus the convolution — which round 5 did not touch. **That is the round-6
lever for nests.** (The parent-side prologue and the `zoom_trilinear` replay
peak at only 4.9 GiB and are not close.)

Combining the two squares and the two nests, a peak model that fits all four:

| run | field | 7 × field + b | measured |
|---|---:|---:|---:|
| 2048² square (cuda) | 1.80 GiB | b = 1.3 → 13.9 | **13.87** |
| 4096² square (cuda) | 7.19 GiB | b = 1.3 → 51.6 | **51.22 / 51.55** |
| 48 km nest (cpu) | 3.11 GiB | b = 3.5 → 25.3 | **25.49** |
| 6 km +1 oct nest (cpu) | 2.87 GiB | b = 3.5 → 23.6 | **23.49** |

**peak RSS ≈ 7 × field + 1.3 GiB (square, cuda) or + 3.5 GiB (nest, cpu)** —
the extra ~2.2 GiB in a nest is the parent-side inherited state that stays
live through the cascade. Re-running table (a) with the nest form:

| width | dx | field | **predicted peak** | under ~50 GiB? |
|---|---|---:|---:|---|
| 48 km | 375 m | 3.11 | 25.3 (**measured 25.49**) | yes |
| 24 km | 375 m | 1.69 | 15.3 | yes |
| 12 km | 375 m | 0.98 | 10.4 | yes |
| 6 km | 375 m | 0.62 | 7.8 | yes |
| 48 km | **187.5 m** | 17.50 | 126.0 | **no** |
| 24 km | **187.5 m** | 9.14 | 67.5 | **no** |
| **12 km** | **187.5 m** | 4.96 | **38.2** | **yes** |
| **6 km** | **187.5 m** | 2.87 | 23.6 (**measured 23.49**) | yes |

### Does the narrow deep nest still scale cleanly? — yes

Haar first-order fluctuation exponent ξ(1) of h, from the `scaleinvariance`
package, over **the same physical window for both nests** (horizontal
750–6000 m; vertical 110–440 m, i.e. up to k_z of the parent's finest class).
Horizontal along x with `periodic=True` (the strip spans the periodic axis),
vertical along z. z thinned ×4 for the horizontal fit and x thinned ×8 for the
vertical one, purely to bound the cost — neither thinning can touch the
exponent in the other direction.

| | 48 km / 375 m (r0) | **6 km / 187.5 m (r1)** |
|---|---:|---:|
| **horizontal ξ(1)** | 0.4259 ± 0.0056 | **0.4380 ± 0.0063** |
| **vertical ξ(1)** | 0.7263 ± 0.0080 | **0.7403 ± 0.0004** |

Over each nest's own full scaling range instead (2 cells to the parent's
finest k), the candidate gives horizontal 0.4362 ± 0.0043, vertical
0.7285 ± 0.0043 — the same answer.

The two geometries agree to **0.012 horizontally and 0.014 vertically**,
about 1.4 σ and 1.7 σ. The nominal target is H_h = 0.45, and the narrow deep
nest is if anything marginally *closer* to it than the wide shallow one. The
fluctuation values themselves rise monotonically and smoothly with lag in both
(no kink at the class boundaries). **The narrow deep nest scales as cleanly as
the current geometry** — over these ranges, on this one realization.

### Judgment for Thomas (his call, not mine)

- **+1 octave is now reachable**, at 6 km (23.5 GiB measured) or 12 km
  (38 GiB predicted) — but not at 24 or 48 km, which stay out of reach by
  1.4× and 2.5×.
- **12 km looks like the better buy than 6 km**: it still fits with ~12 GiB
  of headroom, and it doubles the outer-nest-class supports across the strip
  (4.0 vs 2.0) while halving the normalization noise penalty (2.0× vs 2.8×
  relative to today's 48 km). The measurement above is on 6 km because the
  brief asked for the narrowest fitting candidate; 12 km should behave at
  least as well.
- The scaling evidence is one realization and two fit windows. If this is
  going in the paper, it wants an ensemble.

---

## Final gate

| check | result |
|---|---|
| `./.venv/bin/python -m pytest tests/ -q` | **144 passed** (no test file edited, no assertion weakened) |
| 256² 7-class e2e vs `255f6c2`, `device='cpu'`, all vars + all class increments | **44/44 bit-identical** |
| 256² 7-class e2e vs `255f6c2`, `device='cuda'` | **44/44 bit-identical** |
| strip nest (windowed), all groups | **60/60 bit-identical** |
| wrapping strip nest, all groups | **60/60 bit-identical** |
| **2048² production square, cuda, increments, vs `255f6c2`** | **57/57 bit-identical** |
| 2048² production square wall | **56.2 s** (bar: ≤ 58 s; HEAD today 56.8 s) |
| 2048² production square peak RSS | **13.87 GiB** (bar: ≤ ~14.9 GiB) |

Only `steam/simulate.py` changed. All changes **uncommitted**.

Peak VRAM at 4096²: the streamed convolution at the finest class measures
**11.75 GiB** (`scratchpad/verify_stream_conv.py 4096 4096 115`), the batched
bounded add 7.88 GiB (round 4); they do not overlap. The in-run `nvidia-smi`
total was **not** captured this round — the sampler's own argv matched its own
pgrep pattern, so it never terminated and never wrote its file; a spot reading
mid-cascade was 8.9 GiB. Round 4 measured 13.96 GiB of 15.46 in-run for the
identical GPU code path.

## Open items for Thomas

1. **`_bound_taper`'s slabbing was not in the brief** but was required to
   deliver item 1's memory saving — with only the gradient fused, the 2048²
   peak moved by 0.08 GiB, not 1.8. It is bit-identical, one constant plus a
   three-line loop, and it made the taper *faster* (0.74 → 0.61 s). Keep.
2. **`TAPER_SLAB_BYTES = 64 MiB` affects no result** — unlike round-4's
   `CUDA_BOUNDED_ADD_BUDGET_BYTES`, it does not enter any reduction, so it is
   not realization-defining and can be changed freely.
3. **4096² now works end to end** and takes 269.7 s and 46.45 GB of disk. The
   `simulate()` preflight (8 × field) still refuses it and had to be
   overridden in the probe; the true requirement is 7 × field + 1.3 GiB.
   **The preflight's factor should probably come down from 8 to ~7.2** — but
   that is a safety guard and is Thomas's call, not a silent edit.
4. **The nest's binding stage is now `_advance_flux`**, not the h/qt loop.
   Round 6's nest lever is there.
5. **`refine()` on a 12 km / 187.5 m strip** is the recommended next nest
   geometry if +1 octave is wanted — predicted 38 GiB, 4 supports, 2.0×
   normalization noise.
