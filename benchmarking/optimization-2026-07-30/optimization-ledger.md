# STEAM optimization pass — session ledger (2026-07-30)

Ground rules (Thomas):
- **Acceptance bar: statistics-identical, NOT bit-identical** (Thomas's explicit
  ruling, 2026-07-30). Different RNG/seeding (e.g. torch RNG replacing numpy) is
  fine if it wins on speed/memory; individual realizations may differ. Every
  candidate is **numerically verified before acceptance** — verification is
  sequential, not parallel. Verification = matched ensemble statistics at
  production-relevant config (means/std profiles, Haar fluctuation functions /
  scaling exponents, flux K(q)/C1, clipped fraction — the quantities the paper
  reads), not per-cell comparison.
- **Memory weighted slightly over speed** when they trade off — but clarified
  (Thomas, 2026-07-30 evening): memory reduction matters ONLY where it enables
  larger simulations. Binding peaks: host RSS during cascade (18 GB square) and
  nest replay (31.5 GB = global peak), tmpfs staging (6.5 GB of the same RAM),
  and VRAM only where it caps the field size the GPU bounded-add can take.
  Sub-peak transients (write phase, diagnostics) are nice-to-have only.
- Partial-GPU ops with multiple CPU↔GPU transfers disfavored unless clearly much
  better — code simplicity matters.
- Opus for implementation; codex as an additional set of eyes on candidates.

Baseline (measured, perf-audit-2026-07-30.md, turbulon-model@bef0364):
- Production square (2048², L=1536 km, ls=10 m, H_h=0.45): **508 s** wall,
  18.0 GB RSS + 6.5 GB tmpfs (increment staging), 12.8 GB VRAM (0.55 % busy)
- Strip nest refine() (dx=375 m, CPU): **657 s** wall, 31.5 GB RSS
- Per member (nest is critical path): ~1166 s of work
- _bounded_amplitude_add = 75.1 % of all work (322.7 s square, 552.7 s nest)
- Writes (zlib-4, single-thread): 94.5 s/square; diagnostics 57 s at 2.3/32 cores
- ⚠ co-residency: square (18+6.5) + nest (31.5) ≈ 56 GB on a 60 GB box, unguarded

Candidate ideas from Thomas (to hand to investigation agents):
1. numpy/scipy → torch swaps for parallelization (check current landscape).
2. GPU **discrete/direct convolution** for the expensive small classes — kernel ≪ domain
   there, so direct may beat FFT and fit memory better.
3. More work on GPU generally (e.g. noise generation) — but avoid transfer churn.
4. Interpolation (trilinear zoom) — historically slow; check whether that was
   scipy-era or persists under current implementation.

## Gains log

| # | Change | Verified how | Wall before → after | Peak mem before → after | Output-invariant? | Status |
|---|--------|--------------|---------------------|-------------------------|-------------------|--------|
| 1 | `_bounded_amplitude_add`: contiguous per-level copy + 8-thread pool (`098f3e1`) | np.array_equal vs bef0364 at (2048,2048,115) + nest shape; full 256² 7-class simulate() bit-identical (44/44 vars); 143 tests; codex adversarial pass confirmed production bit-identity | function 131→15 s; member est. 1166→~440 s | ~unchanged (per-level scratch ~8×17 MB) | **bit-identical** (production path) | COMMITTED |

| 2 | Round-2 (`0c130d1`): MKL irfftn guard; exact parallel Lévy (bit-identical incl. RNG stream); n_fft≤32 headroom cap; GPU bounded-add (device-dispatched) | items 1–2 bit-identical (16/16 vars e2e); items 3–4 statistics gate: 8-seed ensemble all |Δ|/SE=0.00, per-level ⟨|δ|⟩ ≤4e-6; 143 tests | square simulate 164→125 s; bounded-add 16–18→1.5 s on cuda | VRAM 11.9→5.9 GiB peak | items 1–2 bit; 3–4 realization-changing on cuda only (nest/CPU bit-identical) | COMMITTED |

| 3 | Round-3 (`c9d1dbb` + analysis `3df2bb5`): staging off tmpfs; numba fused gradient; applied-delta increments; blosc_zstd c1 (+16 KiB raw-chunk floor); fed diagnostics; hyperslab nest read; CUDA test; MemoryMax cap; nest compressed | BIT-IDENTICAL at production scale (57/57 vars incl. all increments + diagnostics; y-wrap nest 76/76); 144 tests | square 182→60.4 s; diagnostics 60.5→25.5 s; nest prologue 29→22 s | RSS 17.5→16.6 GiB; tmpfs 6.5→0; finest-class 6.6→4.9 GiB; nest gradient −3.07 GiB; disk −4.15 GB/member | bit-identical | COMMITTED |

Rulings (Thomas, 2026-07-30 evening): numba dependency APPROVED; blosc_zstd/HDF5
filter plugins APPROVED; nest compression YES; pytorch GitHub comment NO (drop);
zarr considered and deferred (blosc captured the win; revisit if 4096² IO binds).

Cumulative (estimated until round-3 re-benchmark): member ~1166 → ~610 s so far
(square 508→~182 incl. diagnostics/writes; nest still ~430 post-round-1);
round 3 in flight targets writes (−84 s), diagnostics (−63 s), gradient (−16 s),
peaks 18→~15 GB square / 31.5→~28 GB nest, tmpfs −5.7 GB.

| 4 | Round-4 (`255f6c2` + analysis `23ef1fa`): 6 live fields (gamma reuse + flux applied-increment); running_sum reuse; level-batched GPU add (fixed 8 GiB budget); streamed CUDA conv; serialized driver | bit-identical: 57/57 at 2048² production, 44/44 cpu+cuda 256², 60/60 both nests; 144 tests (incl. new cuda test) | 2048² square 60.4→58.2 s; nest 657→132 s (cpu) | RSS 16.61→15.78 GiB; finest-class 11.77→8.61 GiB | bit-identical | COMMITTED |

| 5 | Round-5 (`81c27ef`): fused advective weight (numba, in-register); slabbed _bound_taper (the true peak stage); preflight 8→7.5×field | bit-identical 57/57 at 2048² production; 44/44 cpu+cuda; 60/60 nests; 144 tests | 2048² 58.2→56.2 s | RSS 15.78→13.87 GiB (−1 field exactly) | bit-identical | COMMITTED |

**★ 4096²×115 FULLY-COMPLETE CONFIG COMPLETES (2026-07-30 night): 269.7 s,
51.55 GiB peak (7.17×field), 46.45 GB output, all 10 increment classes +
diagnostics, reproduced twice, h-profile corr +1.0000 vs 2048².**
Narrow-deep nest measured: 6 km/187.5 m (+1 octave) runs in 110.7 s / 23.5 GiB
and scales as cleanly as the 48 km/375 m strip (ξ(1) horizontal 0.438 vs 0.426,
vertical 0.740 vs 0.726, ~1.4σ, marginally closer to nominal H_h=0.45).
Judgment pending (Thomas): 12 km/187.5 m predicted 38 GiB — 4.0 outer-class
supports across width, 2.0× normalization noise vs current (6 km: 2.0 supports,
2.8×). Round-6 lever if wanted: _advance_flux is now the binding stage
(25.28 of 25.49 GiB nest peak). Nest classes correction: current nest has 3
classes (3000/1500/750 m).

**Superseded 07-30 evening (round-4) verdict below — kept for the record:**
4096² verdict (measured, fully-complete config): does NOT fit yet. OOM in
class 10/10 ("computing G" 7-field stage) at 52.58 GiB peak (= 7.31 fields,
arithmetic validated to 4%) vs ~54.1 available; disk needs ~73 GB vs 65 free.
Round-5 candidate that closes RAM: fuse gradient+W into one kernel (7→6 fields,
~45 GiB projected). Disk is a config matter (output to Expansion), not code.
Nest options (measured model peak ≈ 7×field + 2.3 GiB): current 24.1 ✓;
+1 octave 125 ✗; 2× strip width 44.0 ✓; widest at 375 m ≈ 113 km (2.4×).
Nest on cuda: 64 s vs 132 s cpu, +3.3 GiB host RSS — default NOT flipped.

Round 4 (queued behind round 3, Thomas endorsed 2026-07-30 night) — goal:
**4096² squares and larger nests**:
- gamma buffer reuse in _advance_flux (codex#5) and running_sum reuse (codex#7)
  — sub-peak at 2048², peak-relevant at 4096²; ~42–46 GB projected at 4096².
- level-batched GPU bounded-add (per-level independence → stream level batches;
  needed once 3 field buffers exceed 16 GB VRAM).
- driver: SERIALIZE square/nest (drop pipelining — it was a throughput trick for
  the old 657 s CPU nest; retires the co-residency hazard entirely).
- ACCEPTANCE TEST: real 4096²×115 production-config run under a MemoryMax scope,
  measured peak RSS. Then apply the same arithmetic to nest enlargement
  (Thomas: "my thoughts exactly").

Candidate pool for round 2 (codex findings + pending agent reports):
- codex#2: eliminate `before` copy + tmpfs staging (write `d` back / direct netCDF); −1.8 GB RSS, −6.5 GB tmpfs, −5.8 GB write-phase duplication (output.py loads h/qt/flux together — mmap one at a time)
- codex#3: diagnostics as fused numba column kernel — ~9.2→~1 GB working set, 57→30–45 s (vs mem-io agent's fed-pool design; pick one)
- codex#4: sorted one-sided breakpoint μ-solve replacing 60-pass bisection — further 1.2–1.8× on bounded-add; statistics-identical only; second stage
- codex#5: reuse gamma buffer in _advance_flux (expm1 out=) — −3.1 GB square / −5.4 GB nest transients, bit-identical
- codex#6: refine() read only needed parent y-slab — −5.7 GB setup, −12–16 s, bit-identical
- codex#7: reuse running_sum as normalization scratch — −1.8–3 GB transient
- codex#8: drop unused saturated qt copy in thermodynamics — free
mem-io agent verdicts (full report: mem-io/report.md):
- CORRECTION to audit: nest 31.5 GB peak is cascade_loop's finest class (~9.4
  field-equivalents, matches preflight formula to 3%), NOT the replay (prologue
  peaks 8.5 GB, enters cascade at 1.3 GB). Audit §5.5 (slice-first replay) DROPPED.
- #1 staging dir= off tmpfs (one line): −5.7 GB both peaks, +1.1 s, bit-identical.
  Prerequisite for the cgroup cap.
- #2 fused _gradient_components: −3.11 GiB nest peak (~10% larger nests — THE nest
  lever), −1.8 GiB square, −16 s, exactly invariant. Needs numba (dependency Q).
- #3 drop `before` copy via bounded-add writing d back: −1.8 GiB square; MUST
  coordinate with round-2 bounded-add rewrite (contract currently relaxed).
- #4 zlib-4 → blosc_zstd c1: −84 s/square write, bit-identical data; portability
  caveat (HDF5 plugin needed by readers; system ncdump here reads it fine).
  zlib-1 = zero-risk fallback (1.4×). THOMAS'S CALL.
- #5 compress the nest (blosc_zstd c1): +6.2 s for −4.15 GB/member disk. THOMAS'S CALL.
- #6 fed diagnostics + blosc_zstd c1 + cap 8→6: 86.7→24 s, bit-identical (blake2b
  verified). Audit's "raise the cap" was wrong direction — 16w slower & +6 GB.
- #7 hyperslab parent read: −7.5 GB prologue (non-binding), −15 s, bit-identical.
- #8 co-residency: systemd-run --scope MemoryMax=50G on the driver (preflight
  provably wouldn't fire — ramps collide later); needs #1 first.
- #9 nest halo: exactly the kernel half-width at every class — correct, no action.

GPU agent verdicts (full report: gpu-torch/report.md):
- ★ GPU bounded-add: 17–20 s → 1.3 s incl. transfers (14×); member 390→~307 s; 7.2 GiB
  VRAM (5.4 achievable, shifted-space); REALIZATION-CHANGING (stats verified: per-level
  ⟨|δ|⟩ agrees 3e-8). Pure bandwidth win — Thomas's instinct confirmed.
- ⚠ BUG: torch CPU irfftn returns silent ZEROS at exactly 2048×2048×* with ≥4 threads
  (float32 only; forward fine; nest shapes fine). Production (cuda) unaffected, but
  simulate(2048², device='cpu') yields a zero cascade. Guard required; report upstream.
- Lévy draw: chunked numpy + PCG64 .advance() split = 3.0× (5.94→1.97 s), BIT-IDENTICAL
  (arithmetic and RNG stream both verified). Better than numba (not bit-identical) & torch.
- Gradient stencil: numba fused 11.9× (roundoff Δ1.9e-9) or chunked numpy 1.79× bit-identical.
- conv3d direct on GPU: 10× SLOWER than FFT (cudnn degenerates at 1 channel). Memory win
  comes instead from capping n_fft at 32: 11.88→5.88 GiB VRAM for +0.02 s.
- CPU-everywhere conv: REJECTED (real CPU cost 6.29 s/call — audit's 2.48 s timed the
  zero-bug — and n_fft cap gets the VRAM back anyway). Keep cuda; consider cuda for nest.
- Whole-hog GPU-resident cascade: REJECTED — square barely fits (13–14 GiB), nest does
  NOT fit (15.6+ GiB state), buys only ~35 s/member beyond the bounded-add offload.
- zoom_trilinear: confirmed non-issue (scipy era ended at 2de20d1, "~60× faster").
- Misc bit-identical freebies: noise[gamma==0] → band-slice zeroing at s=1;
  _bound_taper np.where hoist; _advance_flux could return its own increment.

| 5 | Round-5 (UNCOMMITTED): fused advective weight `_advective_weight` (grad_h+grad_z -> W in one kernel); `_bound_taper`'s reflected `span - g` copy taken an x-slab at a time | bit-identical: 57/57 at 2048^2 production, 44/44 cpu+cuda 256^2, 60/60 both nests; np.array_equal of W at (2048,2048,115) and (16384,140,364), both z branches; 144 tests | 2048^2 square 56.8 -> 56.2 s; gradient stage 0.45 -> 0.14 s; taper 0.74 -> 0.61 s; nest 127.8 -> 126.5 s | 2048^2 RSS 15.68 -> 13.87 GiB (7 -> 6 live fields at the loop peak); nest 26.46 -> 25.49 (nest peak moved to _advance_flux) | bit-identical | UNCOMMITTED |

**4096^2 verdict (measured, fully-complete config): IT FITS.** Completed in
269.7 s (simulate 170.5, diagnostics 99.3), peak RSS 51.55 GiB = 7.17 x field
under MemoryMax=58G, output 46.45 GB on `/` (153 G free -- no external mount
exists; `/var/tmp/steam-audit`). All 10 class-increment groups present,
diagnostics complete, profiles finite and matching the 2048^2 shape
(h mean corr +1.0000). Peak model now **7 x field + 1.3 GiB (square, cuda)**
and **+ 3.5 GiB (nest, cpu)**, calibrated on four runs.
Nest +1 octave (finest dx 187.5 m, one extra class at 375 m) is REACHABLE at
12 km (38 GiB predicted) and 6 km (23.49 GiB measured), not at 24/48 km.
Narrow deep nest scales cleanly: Haar xi(1) of h over a common physical
window agrees with the 48 km/375 m nest to 0.012 (horizontal) and 0.014
(vertical). NOTE: the current 48 km nest has 3 classes (3000-750 m), not the
4 stated in round 4.

Round 6 candidates: `_advance_flux` is now the binding stage for BOTH the
square and the nest (6 live fields + the convolution); `simulate()`'s
preflight factor 8 is now too conservative (true need 7.2) and blocks 4096^2
without an override.
