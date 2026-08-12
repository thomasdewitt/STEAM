# stream_to_disk: out-of-core execution of the STEAM cascade

Design ruled with Thomas 2026-08-12 (full discussion: thomas-context
project-notes/turbulon.md, 2026-08-12 entry). This document is the working
spec for the `stream-to-disk` branch. Edit it as decisions land; it is the
single source of truth for the feature until merge.

## Goal and contract

`simulate(..., stream_to_disk=True)`: same output as the in-RAM run, RAM
bounded, scratch disk instead. The planner picks the "RAM horizon" — the
deepest size class whose working set fits in memory — runs in-RAM above it
exactly as today, and below it pages the working state through RAM in
horizontal tiles, with the full state living in a scratch directory. On
completion the final NetCDF is written to the requested path and scratch is
deleted (kept on failure for resume/debugging).

**The fundamental test: streamed simulation = non-streamed simulation.**
Difference at the float32/FFT rounding floor, with NO seam-correlated
structure (bin the difference field by distance-to-nearest-tile-seam; the
statistics must be flat). Bit-exactness is NOT required (Thomas ruling:
fp32 rounding differences are a non-issue). The outer regression is the
paper-production figure pipeline in turbulon-analysis.

## Why this is exact, not approximate

The streamed run is the same cascade with the loops reordered. Three
couplings block a naive port, each with its fix:

1. **Noise is keyed to draw order.** [FIXED — component 1, 2026-08-12]
   `_sparse_levy` reshaped one linear RNG stream over whatever array it was
   handed, so the same world position drew different values in different
   regions. Fix: counter-based noise, a pure function of (root_seed, class
   index, world lattice site). Any region's noise is now the restriction of
   the root's. Seed compatibility with existing runs is explicitly NOT
   required (Thomas ruling) — realizations changed; the output file records
   `noise_scheme`.

2. **Realized global reductions mid-class.** [SEAM BUILT — component 2,
   2026-08-12; the two-sweep DRIVER is component 3] Per class the model takes: the
   joint product norm <|g W S|> per level over turbulon centers
   (simulate.py cascade_loop, the level_sum/level_cnt block); the flux
   entering/realized volume means and the mean-abs multiplier noise
   (_advance_flux); the bounded add's per-level (s, mu)
   (_bounded_amplitude_add); the final projection and flux output rescale
   (composition). All are per-class x per-level (or volume) scalars. Fix:
   two sweeps over the tile set per class — sweep 1 accumulates partial
   sums, reduce, sweep 2 applies the global scalars. Exact because the raw
   product at class i depends only on state committed at classes < i plus
   class-i noise, never on class i's own normalization: no circularity
   within a class. The reduced scalars ("the ledger") are recorded in the
   output file — kilobytes, and they make any tile replayable tile-locally.

3. **Halo domain of dependence.** [RESOLVED — component 3, 2026-08-12: the
   per-class re-extraction route was taken, and the halo is kernel_half + 1
   cells per side, verified sufficient by the seam test] A tile descending
   multiple classes without
   re-reading neighbors needs its initial halo to cover the accumulated
   domain of dependence: Sigma_j 3 k_j ≈ 6k of the first tiled class
   (constant 12*s cells per side per class on each class's own working
   grid), shrinking down-cascade under the same crop schedule the nest halo
   uses today. Exactly finite because the kernel is truncated at 3k.
   NOTE: with the two-sweep-per-class structure, tiles are re-read from the
   scratch store every class anyway, so the per-class halo only needs to
   cover ONE class's dependencies (3k + gradient stencil + regrid margin),
   same as today's nest halo. The 6k accumulated halo is only needed if a
   tile ever descends multiple classes without re-paging. Start with
   per-class re-extraction (simpler); keep the accumulated-halo option in
   mind if I/O dominates.

## Rulings (Thomas, 2026-08-12)

- Seed/realization reproducibility with existing runs: non-issue.
- Sequential-only execution (no MPI): tiles/parent always run at max-RAM
  size. Tile count grows 4x per octave below the horizon.
- fp32 rounding: non-issue; seam test is the criterion.
- Horizontal tiling only. A partial vertical extent, if requested, is a
  property of the whole tiling (all tiles share it).
- Current region-scope nesting (`refine`) behavior preserved unchanged —
  streaming is a separate execution mode, not a replacement. (Under keyed
  noise, sibling nests become *consistent* — same world noise — rather than
  the accidental noise-clones they are today; an explicit `seed` remains the
  route to an independent realization.)
- GPU residency (tiles sized to VRAM; see 2026-08-07 benchmark notes, capped
  by VRAM at 67% _compose_output) is deferred until streamed=non-streamed
  passes. Design nothing that precludes it: the per-tile work unit must stay
  device-agnostic (it already is — `device=` plumbs through).

## Components, in landing order

Each lands separately, each independently verifiable. 1 and 2 are
prerequisites and are behavior-changing (1) / behavior-neutral (2) on the
NORMAL path; 3-6 are additive.

**STATUS: components 1-5 all landed 2026-08-12. The feature is complete and the
spec is now the merge documentation.** Component 6 (GPU-resident tiles) was
always deferred; see Future work.

### 1. World-keyed noise  [LANDED 2026-08-12]

Implemented as described below. What landed, and what the implementation had
to decide that this spec did not cover:

- **Generator: Philox4x32-10** (`steam/noise.py`), jitted, counter =
  (ix, iy, iz, key2), key = (key0, key1) — 96 bits of per-class key from the
  class's SeedSequence child. Validated against all three published Random123
  known-answer vectors, and the shared round structure at 64-bit width against
  numpy's own Philox bit generator. 4x32 rather than 4x64 because its round
  needs only a 32x32 -> 64 multiply, native in numba's uint64.
- **The uniforms are numpy's own conversion** (24 bits x 2**-24), so they sit
  on the identical discrete lattice the stream drew from; with `_levy_chunk`
  unchanged the two schemes are distributionally identical rather than merely
  close. `_levy_from_uniforms` is the shared transform; `_extremal_levy` and
  `_parallel_uniform_float32` are kept OFF the cascade path as the reference
  the keyed uniforms are tested against (a KS test at three alphas).
- **Throughput: faster, not slower.** 1.06x at 1.93 G draws (6.67 -> 6.29 s),
  1.03-1.13x at smaller sizes; the requirement was 1.5x. Being counter-based
  it has no sequential dependency, and the transform is shared, so there was
  nothing to lose. `tests/heavy/keyed_noise_throughput.py`.
- **`NoiseRegion`** (key, world origin, world shape) is the per-array
  descriptor; `cascade_loop` takes `world_origins` / `world_shapes` (None =
  root) alongside `inner_windows`. This is the seam a tile will use unchanged.
- **`class_seeds` replaces `seeds_or_rng`.** The shared-Generator "legacy
  behavior" is gone — a stream cannot be world-keyed, so it raises rather than
  silently reseeding. Nothing in the package used it.
- **New file attributes:** `noise_scheme` (records which scheme drew the
  file — realizations changed), and `root_domain_x/y/height`, the ROOT's
  domain, inherited unchanged down every generation of nesting. Keyed noise is
  defined on the grid a root run over that domain has at each class, so a
  descendant needs the domain itself, not just its own extent. `refine` on a
  parent lacking them REFUSES: such a file predates the change and its
  realization came from the stream anyway.

LIMITATION, documented per the delegation rather than forced. Two sub-cell
mismatches exist between a nest's grid and the world grid at the same class,
both properties of the existing gridding and not of the keying:

1. `dx` is the padded extent divided by a rounded cell count, so a nest's dx
   differs from the world's at the same class by O(1/nx);
2. a partial-height nest rescales dz to span its own padded height
   (`_compute_all_grids`), so its z levels are not a subset of the world's at
   all — the vertical world index is genuinely ill-defined there.

So the world index is the NEAREST world cell: exact when a nest's grid
coincides with the world's (full-span, full-height) and accurate to a fraction
of a cell otherwise. This costs the actual target nothing — tiles share the
full vertical extent and are exact integer offsets on a common horizontal
grid, so for tiles the restriction is exact in both index and physical space.
For nests it means "consistent with the root to within the regrid the nest was
already doing", which is the same standard the inherited state meets.

The original component description follows.


Replace stream-ordered draws in `_sparse_levy` / `_extremal_levy` /
`_parallel_uniform_float32` with counter-based generation: gamma at class i,
world lattice site (ix, iy, iz) is a pure function of
(root_seed, i, ix, iy, iz). Mechanics:

- Two uniforms per center (the CMS transform in `_levy_chunk` is unchanged),
  from a counter-based generator. Numba is available and already used
  (`_advective_weight_stencil` is jitted); a jitted Philox-style or
  splitmix64-based hash over (key, linear world index) x {0,1} is the
  simplest correct route and trivially parallel. numpy's
  `np.random.Philox(key=..., counter=...)` is the reference to validate the
  hash against IF used; a hand-rolled counter hash must pass distributional
  tests (uniformity, independence across the two lanes, no lattice
  structure).
- Per-class keys: keep `SeedSequence(root_seed).spawn(n_classes)` and derive
  each class's 128-bit key from its child — the existing bookkeeping
  (`root_seed`, `n_classes_consumed`) stays meaningful.
- The center lattice must be anchored to WORLD coordinates: a region whose
  origin sits at world cell (ox, oy, oz) on class i's working grid draws
  its centers at world sites, not array-corner-relative sites. `refine`'s
  extraction knows its index range within the parent; the linear world index
  is over the ROOT's full working grid at that class. Sub-lattice phase for
  s > 1 comes from the world origin, not the array origin.
- Performance: the current generator is heavily optimized (thread pool over
  4M-element chunks, in-place float32 CMS, L3-aware; see _extremal_levy
  docstring — 1.8 GiB of draws at the production finest class). The keyed
  version must be comparable. A jitted hash is embarrassingly parallel and
  should beat the stream (no sequential dependency at all).
- Out of scope for this component: any change to normalization, cascade
  logic, or the streaming machinery. `zero_bottom`/`zero_top` masking and
  sparse placement logic stay as they are (they consume the field, not the
  stream).

Tests: (a) restriction property — noise generated for a subregion equals
the corresponding slice of the root's noise field, exactly, for s = 1 and
s > 1, including nest-continuation classes; (b) distribution unchanged —
moments/quantiles of the new generator against the old CMS pipeline at
matched alpha (the transform is shared, so this tests the uniforms);
(c) existing test suite passes (test_flux_cascade, test_refine identity
tests — realizations change, properties must hold; update any test pinning
exact values to a seed); (d) throughput within ~1.5x of current at a
production-shaped draw.

### 2. Measure/apply split of the class body  [LANDED 2026-08-12]

Implemented as described below. What landed, and the decisions the spec did
not cover:

- **`steam/ledger.py`.** `RunLedger` -> per-class `ClassLedger` -> the flux
  advance's means, the per-scalar product norms, the per-scalar bounded-add
  solve; plus a composition entry (projection mu per scalar, flux output
  clip-and-restore). Every mean is a `MeanReduction(total, count)`, never a
  mean: a mean is not additive across tiles and a sum is.
- **`ledger=` on `cascade_loop`, `simulate`, `refine`.** None records; a
  recorded ledger runs APPLY-ONLY, every solve skipped and every scalar
  injected. Same-device-exact (the GPU's reduction order is not numpy's).
- **BIT-EXACT on all five pinned configurations, CUDA included.** References
  pinned on 959f925 first, by `tests/heavy/make_refactor_references.py`;
  `tests/test_measure_apply.py` is the gate, and it was verified to be a real
  gate (a plausible float64 "improvement" turns it red).
- **The flux advance is FOUR phases**, structured honestly rather than forced
  into two: measure_entering -> apply_update -> measure_realized ->
  apply_rescale. The post-clip mean can only be measured after the update
  whose bias it corrects.

**SPEC CORRECTION — the bounded add's ledger is a scalar SEQUENCE, not
(s, mu).** The spec assumed the solve's candidate family is
clip(s*d - mu, caps), as its own docstring says. The implementation clips
INSIDE the demean/rescale loop, so the composite is a chain of
affine-then-clip steps that no single pair can express: measured on a qt-like
level against the lower bound, clip(s_total*d - mu_total) misses 58% of cells
by up to 11% of the field scale, while replaying the recorded sequence is
bit-exact. So the ledger records, per level: a0, the loop demeans and scales
with their counts, the final demean, and mu (NaN where no bisection ran). This
tiles exactly as well as (s, mu) would have — every recorded entry is a
per-level scalar and every step between them is a pointwise clip — so nothing
downstream is harder, it is just more numbers.

**FOUND during component 2, FIXED as component 3's pre-work.** The flux's
mean-abs multiplier noise was accumulated in **float32** —
`np.abs(noise_inner).sum()` with no dtype argument — unlike the float64 volume
means beside it, while reducing over a field reaching ~1e9 cells at the
production finest class. Left alone in component 2 because bit-exactness was
that component's gate; now `dtype=np.float64`, which

- matches the repo's float64-accumulator convention at that scale, and
- makes the WHOLE ledger exactly additive across tiles. It was the one
  reduction whose per-tile partials were not, so the streamed-vs-in-RAM
  comparison is now "the fp32/FFT floor, period" rather than "the floor plus
  one fuzzy reduction".

It feeds S_k and never the flux state, so it moved the scalar realization;
ruled a non-issue, and the pinned references were regenerated in the same
commit that made the change.

**Float-discipline trap worth knowing for component 3.** The division in a
realized mean is float64 in every case, and that is not a choice: inline, a
float32 total was divided by a numpy int64 count, and NEP 50 promotes that
pair to float64 because an int64 is strong. Coercing the count to a Python int
makes the pair float32 (a Python int is weak) and moved 19 cells of h by one
ULP — caught by the pinned gate, not by reading. `MeanReduction.mean` divides
in float64 explicitly rather than relying on promotion.

**Ledger format: ragged via subgroups** (`ledger/cNN/`), following
`class_increments`. nz differs from class to class, so a padded
(n_classes, nz_max) array plus a validity mask would put every reader in the
business of knowing which entries are real. Each class subgroup carries its
own `z` and `rescale` dimensions. `mu` variables are written with
`fill_value=False` because NaN is the load-bearing "not projected" / "not
bisected" marker and a fill value would swallow it. Kilobytes per run.

The original component description follows.


Refactor `cascade_loop`'s per-class body so every realized global reduction
is hoisted into a MEASURE phase (accumulate partial sums: product level
sums/counts per scalar, flux entering volume mean, noise mean-abs) and an
APPLY phase that takes the reduced scalars as arguments (normalize, convolve,
bounded add, flux update/clip/rescale). On the normal in-RAM path the two
phases run back-to-back on the resident array and output is IDENTICAL to
today (bit-exact — this is a pure refactor and must be validated as one).
Record the reduced scalars per class ("ledger") and write them to the output
file as a new group.

The bounded add is the special case: its (s, mu) is solved against the
data per level. Split it as solve (per level, needs the level's candidate
increment) / apply (clip with given s, mu). In-RAM, solve-then-apply is
what it already does internally; the refactor exposes the seam so the
streamed path can solve on assembled planes.

Tests: bit-exact output equality with main across the existing test suite's
simulate() calls plus one production-shaped small run (same seed — component
1 changed realizations, so the reference is regenerated on this branch, not
main).

### 3. Tile store + streamed driver  [LANDED 2026-08-12]

`steam/streaming.py`. Classes above the RAM horizon run in RAM through
`cascade_loop` unchanged; below it each class runs

    pass 0 REGRID -> pass 1 MEASURE -> reduce -> pass 2 APPLY -> reduce
    -> pass 3 PLANE (the bounded add on assembled z-planes)

**MEASURED** (16² × 10 through 64² × 10, four classes):

| quantity | result |
|---|---|
| streamed vs in-RAM, h_perturbation | 2.6e-06 relative |
| streamed vs in-RAM, qt_perturbation | 5.9e-07 |
| streamed vs in-RAM, flux | 2.4e-07 |
| same floor at 1×1 … 8×8 tiles | yes — error does not grow with tile count |
| **seam test** (seam bin ÷ interior median) | **0.94 / 0.92 / 0.87**, bins flat to 10% |
| ledger: flux entering, mean-abs noise | EXACT |
| ledger: realized mean / product norms | 1e-6 / 1.1e-08 |
| scratch accounting | predicted peak bounds actual, within 3× |

float32 eps is 1.2e-07, so h at ~20 ULP after four classes of a nonlinear
multiplicative cascade is where it should be. The seam bin is if anything
QUIETER than the interior: no seam-correlated structure.

Decisions this spec left open:

- **The store holds WORLD-SIZED fields**, one memory-mapped `.npy` per (class,
  field) — not per-tile files with a manifest. Tiles partition the WORK, not
  the data: horizontal tiling never splits z, and a streamed run is a root, so
  its stored state carries no halo at all (halos are transient, built on
  page-in by wrapping). One mapping per field serves both required access
  patterns — tile+halo is a strided slice, a z-plane across all tiles is
  `array[:, :, lev]` — with none of the stitching bookkeeping. Given this is
  the area that produced the align_corners=True and never-periodic-resample
  findings, fewer indexing sites is the right trade.
- **The tiled regrid is exact and has its own gate test.**
  align_corners=False makes the target→source map a pure dilation, so with
  n_in/n_out written as b/a in lowest terms, a source slab quantized to b lands
  on target-cell boundaries at multiples of a. One extra block of b source
  cells per side (wrapped) supplies the periodic continuation, exactly as
  `utils._wrap_plan` does — WITHOUT it torch clamps at the tile edge, which
  puts a seam in the domain interior. Bit-exact against the full-domain
  resample for a dyadic ladder; a few float32 ULP for non-dyadic ratios.
- **THREE passes per class, not two.** Within a class there is a CHAIN of two
  dependent global reductions: the mean-abs multiplier noise gates S_k, and S_k
  enters the product whose norm is the second reduction. Rather than pay a
  third sweep, pass 1 measures the RAW product (the entering flux in place of
  S_k) and the normalized totals are recovered by one division — a positive
  scalar factors straight out of a sum of absolute values. Costs one float32
  multiply's worth of rounding, which is the floor this path is measured at.
- **Halo = kernel_half + 1 cells per side**, constant across classes because
  the kernel is (SUPPORT_FACTOR·k is 6·s cells at every class), plus one for
  the advective weight's gradient stencil. Verified sufficient at halo = 7 with
  a 13-cell kernel: convolution difference 2.0e-07.
- **Tile visit order is fixed** (world raster, x-major) and documented, because
  the float64 partials accumulate in it. Two streamed runs of one configuration
  are bit-identical; tested.

**BUG THE FUNDAMENTAL TEST CAUGHT, worth recording.** Pass 2 originally wrote
the updated flux back into the field it read tile halos from, so every tile
after the first saw a halo of already-advanced values — a 13% error in the
realized flux mean. Fixed by double-buffering the flux within the class and
swapping at the end of the pass. Caught on the first run of
streamed-equals-resident, which is the argument for writing that test first.

Deferred with reasons: the compensation deficits and `save_for_refinement`
(both belong to the output composition — component 4), and crash resume (the
progress marker is written from the start, so component 5's lifecycle work is
cheap). No public `stream_to_disk` flag yet (component 5); an internal entry
point with forced tiling is what the tests drive.

The original component description follows.


The new machinery. A scratch store (npy or zarr per tile per field, chunked
so a z-plane across tiles is cheaply assemblable — decide during
implementation; plain npy per tile with a manifest is fine for v1) holding
h_perturbation, qt_perturbation, flux, and the deficit fields. The driver:

- Planner: from the grid plan (all shapes known up front), find the RAM
  horizon class via the existing byte calculators (utils.fft_convolution_bytes
  etc. plus the live-field count in cascade_loop's docstrings), choose tile
  counts per class below it (powers of two; tiles sized to fill RAM), and
  compute total scratch bytes. Disk check against the scratch dir up front;
  refuse with the number if short.
- Above the horizon: today's loop, unchanged, on the resident array. At the
  horizon: write the state into the store, switch modes.
- Below, per class: sweep 1 over tiles (read tile + halo from store —
  modular wrap at periodic boundaries, exactly refine's extraction indexing —
  regrid to this class's working grid, generate keyed noise, measure);
  reduce; sweep 2 (re-read, regenerate — recompute-instead-of-persist,
  keyed noise makes it exact — apply with global scalars, convolve, bounded
  add solve deferred, write candidate + state back); plane pass (assemble
  each z-level across tiles, solve (s, mu) with the component-2 solver,
  apply, write back). The flux post-clip rescale is one scalar per class:
  fold it lazily into the next class's read instead of a third sweep.
- Regrid between classes happens on page-in (tile + margin), using the same
  trilinear cell-consistent convention; +1 source-cell margin.
- Crash resume: the store plus a progress marker per (class, sweep) makes
  resume nearly free. Build it in from the start; keep scratch on failure.

RISK CONCENTRATION: regrid + halo + periodic-wrap bookkeeping. This exact
area produced the align_corners=True finding (2026-07-31) and the
never-periodic resample (2026-08-06). Write the seam test FIRST and run it
on every change.

Tests: the fundamental one — toy-size run (e.g. 128x128x32-ish, 3-4 tiled
classes, forced tiny RAM horizon, 4x4 tiles) streamed vs in-RAM: rtol at
the rounding floor, seam-binned difference statistics flat. Plus: tile count
1x1 spanning (= in-RAM path through the store, should be near-bit-exact),
non-square tiles, domain-edge tiles wrapping the periodic boundary,
save_for_refinement increments written correctly from streamed mode.

### 4. Streamed composition, diagnostics, output  [LANDED 2026-08-12]

A streamed run produces a COMPLETE output file.
`simulate(_force_tiling=(horizon, tiles_x, tiles_y))` is the internal entry
point; the public switch and lifecycle are component 5.

**MEASURED**, streamed vs resident written file (32² × 10, four classes, 2×2):

| variable | relative |
|---|---|
| h / qt / flux | 9.1e-08 / 1.8e-07 / 2.6e-07 |
| h_pert / qt_pert / flux_state | 1.0e-06 / 3.6e-07 / 1.6e-07 |
| T / p / qv / qc / qi | 2.0e-07 / 7.7e-08 / 5.3e-07 / 3.1e-06 / 0 |
| **nest from streamed parent** | **9.1e-08 / 1.8e-07 / 3.5e-07** |
| **seam ratio, composed h / qt (4×4)** | **0.93 / 1.04** |

The composed fields are TIGHTER than the states they come from — arithmetic, not
luck: h is ~3.4e5 against a ~1e3 perturbation. The 2×2 seam ratio reads 1.58 on
h, which is sampling noise on a 124-cell bin; 4×4 doubles the sample and reads
0.93. Both recorded so nobody rediscovers that the 2×2 number is noisy.

**Diagnostics needed no streaming work.** `compute_diagnostics` already walks a
written file in x-chunks, so the item reduced to "write h/qt/flux, then call it
as-is" — checked before writing any code. The saturation adjustment did not
amplify anywhere here (0.0000% of cells past 1e-5 of scale for every condensate
variable), and a dedicated test pins that fraction so a configuration that DOES
amplify gets characterized rather than absorbed into a looser tolerance.

**Structure:** composition per-LEVEL on the store's z-major planes (which is what
the projection is), file written per-TILE in x-y with full z (which is what the
output variable's chunks are). Composing straight into the NetCDF variable plane
by plane would touch every chunk per level — the memmap trap one level up. Two
hooks carry it: `write_netcdf(tile_writer=)` and
`write_class_increments(readers=)`.

Everything after the cascade — coordinates, stored ladder, `simulation_params` —
is SHARED between the paths, which is what makes the files comparable attribute
for attribute; `test_file_structure_is_identical` asserts it.

**Bug the ledger round-trip caught:** the composition's projections were merged
into the run ledger AFTER `write_netcdf` serialized the ledger group, so the
streamed file's projection group was empty while every field agreed. Same shape
as the resident-head deficit bug — a quantity the main comparison cannot see.

**Scratch goes beside the output file, never /tmp**, which is tmpfs: scratch
there IS RAM and defeats the feature. Component 5's `scratch_dir=` must detect
and refuse or warn.

Earlier in this component:

**Layout (prerequisite, found in review).** The store is now Z-MAJOR, stored as
(nz, nx, ny) behind an (nx, ny, nz) view. On a C-ordered (nx, ny, nz) field the
elements of one z-level are nz·4 bytes apart, so reading ONE level touches
essentially every page; with the field larger than RAM each level re-reads the
whole field. Measured (`tests/heavy/plane_pass_io.py`, 256³, nz=256, cold):

| pass | (nx,ny,nz) | (nz,nx,ny) |
|---|---|---|
| **per-level** | **256.0×** (= nz) | **13.3×** |
| tile+halo, y=128 | 2.4× | 4.0× |
| tile+halo, y=64 | 5.5× | 16.0× |

The 13.3× is pessimistic (an artifact of evicting between levels; a real plane
pass is a sequential scan). Chosen over dual-layout-plus-transpose: one layout,
no transpose passes, comparable totals. Tile access goes through the RAW buffer
(`read_window`/`write_window`), not the transposed view — slicing the view is an
element-by-element gather and cost 8× at toy scale, all of it strided-copy
overhead. Numerically neutral: floor and seam ratios unchanged to the digit.

Note for anyone measuring this: **/tmp is tmpfs on this box**, and a tmpfs file
never reaches the block layer, so `read_bytes` sits at zero and the benchmark
silently reports "1.0×, no problem". The benchmark now refuses to run there.

**Item 1 — the flux finishes inside its class.** The rescale is applied in pass
3, the true increment formed against the still-live entering buffer, the flux
deficit accumulated, then the double buffer swapped. `pending_flux_rescale` and
its end-of-run special case are gone; "the store always holds a fully advanced
flux" is an invariant.

**Item 2 — deficits** accumulated in the store, regridded with the state,
allocated lazily at the first class with f != 1. Verified against the resident
path at the states' own floor (2.4e-06 / 5.2e-07 / 3.2e-07) across every horizon
and tiling.

**Item 3 — increments** recorded per class into a second store and tested;
writing them into the output file's `class_increments` group waits on the
hyperslab writer below.

**Bug worth recording, caught by the deficit comparison and located in the
RESIDENT HEAD, not the tiles:** the head's `cascade_loop` was called without
`comp_k`, so it computed the default from the finest class of the grids it was
handed — compensating its classes as though the cascade stopped at the horizon.
The states matched perfectly while the deficits were 11% out. An independent
quantity is the only thing that could have shown this.

The original component description follows.


_compose_output (deficit add + final per-level projection), the saturation
adjustment / hydrostatic column solve (column-local), and NetCDF writing
(hyperslab per tile). The final projection is per-level: reuse the plane
pass. Nothing algorithmic changes.

### 5. The switch and lifecycle  [LANDED 2026-08-12]

`simulate(stream_to_disk=False, scratch_dir=None, memory_budget=None,
fresh=False)`. All five components are now in; the feature is complete.

**The switch does not auto-engage.** Left off, a run whose resident working set
exceeds `memory_budget` fails EARLY with a `MemoryError` naming the predicted
bytes, the budget, and the flag — a helpful refusal, because changing execution
mode is the caller's decision and not the planner's. `memory_budget` defaults to
80% of `MemAvailable` from /proc/meminfo (the kernel's own estimate of what a new
allocation can have without swapping, which already discounts the reclaimable
page cache a streamed run leans on).

**Every check runs before any compute:** scratch filesystem type, scratch free
space against the planner's predicted peak, and the OUTPUT file's destination.
That last one matters — a run that streams for hours and then dies inside
`write_netcdf` on a full filesystem is exactly the failure this feature exists to
prevent.

**tmpfs is refused, not warned about.** A RAM-backed scratch directory IS memory,
so streaming to it bounds nothing. Detected via /proc/mounts (statvfs exposes no
`f_type`): longest mount point that is a prefix of the resolved path, then its
type against tmpfs/ramfs/devtmpfs. There is no override — an exotic setup can
name a real filesystem. The default scratch location is beside the output file
for the same reason, since the system temp directory is tmpfs on most Linux
distributions. This is not hypothetical: it silently invalidated the first
version of the I/O benchmark, which reported a meaningless "1.0×".

**RESUME.** A manifest (SHA-256 over the class ladder, every grid shape, the
seed, the tiling plan, the bounds, the input profiles, the flux parameters and
the noise scheme) plus an appended per-(class, phase) progress log. Matching
manifest → resume from the last completed phase; absent or mismatched → refuse,
offering `fresh=True`. A mismatched store is never silently reused: half a
cascade from a different configuration is worse than none.

The structural change resume required: **the plane pass is now double-buffered**,
state and deficit both. It used to mutate the state in place level by level, so a
crash mid-pass left a half-projected field that a re-run would project twice —
and the deficit is accumulated, so it would double-add. Writing to a companion
buffer and swapping at pass end makes "resume = re-run the last incomplete pass"
unconditionally true. Cost: one extra state field of I/O per class per scalar.

Passes 1 and 2 were already re-runnable (pass 1 is pure measurement; pass 2 reads
buffers that survive until the plane pass swaps them). One thing that CANNOT be
recomputed is a finished class's ledger entry — and re-measuring is not an option,
because a partially completed plane pass may already have swapped the entering
flux — so each class's entry is persisted to scratch as it completes and read
back on resume.

**RESUME IS BIT-IDENTICAL** to an uninterrupted run, verified at five
interruption points: mid-pass-2 after the first tile, mid-plane-pass on each
scalar (which is what proves the new double buffer), between classes, and after
the regrid. Not "close" — identical: the partials accumulate in a fixed tile
order and world-keyed noise makes recomputation exact, so there is nothing left
that could differ.

The resident classes above the horizon are not resumable and simply re-run. That
is cheap by construction — they are the classes that fit in memory.

Scratch is destroyed on success and RETAINED on failure, with a printed message
naming the path and saying that re-running resumes.

### The original component 5 description follows.

`stream_to_disk=False/True` (consider 'auto' later, engaging on planner
overflow with a warning), `scratch_dir=` (default: system temp on the same
filesystem as the output path — scratch I/O is the cost driver; document
that it should be NVMe), final copy to requested output path, scratch
cleanup on success, retention + resume instructions on failure.

## Future work

In rough priority order, as of 2026-08-12 with components 1-5 landed:

0. **Streamed apply-only replay.** `ledger=` under `stream_to_disk` raises today:
   the ledger would have to be distributed back over the tiles and planes.
1. **GPU-resident tiles** (the original component 6, below): tiles sized to VRAM
   rather than RAM, per-tile work running device-resident end to end. Nothing in
   1-5 precludes it — the per-tile work unit is a pure function of (slab arrays
   in, slab arrays out, scalars) and `device=` plumbs through. Revisit the
   2026-08-07 benchmark conclusions then.
2. **Production shakedown**: a medium-scale streamed-vs-resident comparison at
   the largest size the resident path can still run, then a demonstration run
   past the RAM wall. The outer regression is the paper-production figure
   pipeline in turbulon-analysis.
3. **`stream_to_disk='auto'`**, engaging on planner overflow with a warning. Held
   deliberately: the current behaviour is a helpful refusal, and auto-engagement
   changes execution mode without the caller asking.
4. **Streamed nests.** `refine` does not take the flag and the driver raises on
   non-root grids (a varying padded extent). A streamed nest would need the nest
   halo to compose with the tile halo, which was never in scope.
5. **A tighter halo.** Currently `kernel_half + 1` per side at every class,
   verified sufficient. The accumulated-halo variant (letting a tile descend
   several classes without re-paging) is only worth it if I/O turns out to
   dominate; measure before building.
6. **Durability limitations — deliberately out of scope.** Resume is a
   convenience, and `fresh=True` is always the fallback. We defend against
   process death at any point (atomic marker and ledger commits, idempotent
   settle, double-buffered phases) and against a host crash to the extent that
   `fsync` on the file and its directory plus POSIX `rename` semantics allow. We
   do **not** defend against filesystem-level torn writes inside a single
   `.npy` field file, non-POSIX rename semantics (some network filesystems), or
   media corruption. A store that fails its manifest stamp is refused rather
   than repaired. This boundary is a decision, not an omission.
7. **Narrow-tile I/O.** The z-major layout costs tile reads a factor of ~2 at
   128-cell tile widths and nothing at production widths. If a configuration ever
   forces very narrow tiles, that constant is where to look.

### 6. (Deferred) GPU-resident tiles

After streamed = non-streamed passes: tiles sized to VRAM instead of RAM,
per-tile work running device-resident end-to-end. Revisit the 2026-08-07
benchmark conclusions then. Nothing in 1-5 may preclude this: keep the
per-tile work unit a pure function of (slab arrays in, slab arrays out,
scalars) with `device=` plumbed through.

## Component 6 — independent codex review, all 12 findings resolved

Thomas asked for a codex pass over the whole branch; it found 12 issues,
concentrated exactly where toy-scale tests are structurally blind. **Two of the
three coverage holes it exposed are now their own test classes.**

**Feature-defeating.** (F1) The resident OOM preflight ran unconditionally,
before the streaming branch — so every run big enough to need streaming was
refused, and the feature could not exceed RAM at all. (F2) Horizontal halos were
derived from the **vertical** kernel width; the kernel's cell extent scales with
its own axis's sparsity, so at s=(3,1,1) x needed 19 halo cells and got 7,
letting per-tile convolution wrap reach the inner region. Reintroducing it
reproduces h error 6.784e-03 against codex's reported ~7e-3.

**Noise restriction.** (F3) The sub-lattice phase was chosen on the UNWRAPPED
index, which is only equivalent when `world_n % s == 0`. Reachable on
narrow-strip grids at s>1; fixed at the root, standard configs unaffected (the
pinned references confirm it).

**Resume crash-consistency** (F4, F5, F6) — one protocol, not three patches.
Every phase is now: pure writes to `*_next`, flush, persist what the marker
implies exists, **atomic marker commit** (temp file + `os.replace`, settle
actions recorded beside it), then **settle** (renames and drops, idempotent,
replayed on every resume). Restart is a single clause: before the marker all
inputs are intact so re-run; after it, settle and continue.

**Manifest** (F7, F8, F9). The fingerprint had omitted H_h and lambda — module
constants read at run time, and actively swept in constants.py, so a resume
across an edit would have spliced two realizations — plus device (CPU/CUDA differ
at ULP, so a mixed resume matches neither pure run) and seven others. `seed=None`
runs could never resume, i.e. the DEFAULT API path was unresumable; the manifest
now carries a seedless digest and the drawn seed. And `prepare_scratch` treated
"no manifest" like `fresh=True` and rmtree'd the directory, so pointing
`scratch_dir` at any existing directory destroyed it — now always refused, with
nothing deleted, and `destroy()` removes only store-owned files.

**API honesty** (F10, F11, F12). `ledger=` under streaming raises rather than
silently measuring afresh; scratch accounting covers every tenant including the
degenerate no-tiling plan; a resume credits the scratch it already owns.

**A bug the new crash-window test caught in my own fix:** the settle protocol was
defined and never invoked on resume — `settle_all()` existed, the call site did
not — and the flux diverged 53%. Defining a mechanism is not wiring it in, and
only a test that crashes inside the window could tell.

**New coverage classes**, both permanent: *won't-fit-in-RAM* (memory patched down
so the preflight would fire; the streamed run must proceed while the resident one
refuses) and *anisotropic sparsity* (s=(3,1,1), (1,3,1), (2,3,1)).

### Round 2 — re-review of the fix commits

Codex re-reviewed the fixes, confirmed the **numerics clean** (per-axis halo,
wrapped-world center selection, preflight gating, seed adoption, `ledger=`
refusal) and found 8 further issues, **all in the crash-durability and accounting
layer**. Two fixes are structural, chosen to kill whole classes rather than the
instances found.

**A — markers are one empty file each, not an append log** (#1). A crash or
ENOSPC part-way through an append leaves a torn line, and the *next* append
concatenates onto it — `"2 plane"` + `"2 plane_h"` reads as one nonsense entry a
tolerant parser silently drops. Silent loss of a completed phase is the worst
failure available here, because the resume then re-runs a phase whose inputs its
settle already consumed. One file per marker makes that impossible by
construction. Every atomic replace now goes through one helper that fsyncs the
file *and* the containing directory: on ext4 a rename can be durable while its
directory entry is not, which would preserve a marker and lose its settle record.

**B — containment, not enumeration** (#4). The store owns
`scratch_dir/steam-scratch-store`, writes nothing outside it, and `destroy()`
rmtree's exactly that subtree. `scratch_dir` is never scanned, never deleted,
never required to be empty, so other contents are *irrelevant* rather than
protected by a list of "files we own" — enumeration whose failure mode is
deleting someone's data. The manifest carries an ownership stamp checked before
any reuse or destruction, so a directory holding a merely-parseable
`manifest.json` (`{}` included) is not mistaken for ours.

**The other six.** (#2) the class ledger was written straight to its live path —
atomic now. (#3) the fingerprint omitted dx/dy and the padded extents; nx=2 at
dx=25 and nx=2 at dx=24 are the same shape and a different answer. (#5) regrid
dropped the previous class by hand outside the protocol — recorded as settle
actions, because a protocol with special cases is not one. (#6) the degenerate
no-tiling accounting floored at 6 final-field equivalents where the concurrent
peak is 9. (#7) `total_bytes()` missed the increment sub-stores; an rglob over
the owned subtree, which falls out of containment. (#8) scratch and output were
checked independently against the *same* free space — the default case, since
scratch lives beside the output and is not cleaned up until after the write.

**Two bugs of my own**, both caught by the new tests rather than by reading: the
store ended up nested inside itself (`simulate` passed the owned root where the
driver expected the scratch dir), and the containment rewrite initially checked
`scratch_dir`'s emptiness rather than the store subdirectory's, refusing every
run whose scratch directory it had just created.

**No realization change**, asserted by the pinned references passing untouched.

## Measured numbers, all configurations (2026-08-12)

One table, so the merge review has them in one place.

| what | measured |
|---|---|
| keyed noise throughput vs the stream it replaced | 1.06× at 1.93 G draws (faster) |
| streamed vs resident, cascade states | h 2.6e-06, qt 5.9e-07, flux 2.4e-07 |
| streamed vs resident, deficits | 2.4e-06 / 5.2e-07 / 3.2e-07 |
| streamed vs resident, written file | h 9.1e-08, qt 1.8e-07, flux 2.6e-07 |
| streamed vs resident, diagnostics | T 2.0e-07, p 7.7e-08, qv 5.3e-07, qc 3.1e-06 |
| nest from a streamed parent vs a resident one | h 9.1e-08, qt 1.8e-07, flux 3.5e-07 |
| **seam ratio, cascade states (4×4)** | **0.94 / 0.92 / 0.87** |
| **seam ratio, composed fields (4×4)** | **0.93 / 1.04** |
| floor invariance with tile count | identical from 1×1 to 8×8 |
| resume vs uninterrupted | **bit-identical**, 5 interruption points |
| per-level I/O amplification, (nx,ny,nz) → (nz,nx,ny) | **256.0× → 13.3×** |
| ledger agreement, flux entering / mean-abs noise | exact |
| scratch accounting | predicted peak bounds actual, within 4× |
| **resume after a torn/failed write** | recovers, bit-identical (9 injection points) |
| **anisotropic s=(3,1,1)** | floor 5.0e-06, seams 1.25 / 1.18 / 1.13 |
| **anisotropic s=(1,3,1)** | floor 3.7e-06, seams 1.13 / 1.10 / 1.05 |
| **anisotropic s=(2,3,1)** | floor 3.5e-06, seams 1.06 / 1.19 / 1.12 |

float32 eps is 1.2e-07 throughout. The seam ratio is the seam-adjacent
difference bin divided by the interior median; ~1 means no seam-correlated
structure, which is the criterion the whole design is held to.

## Merge criteria

- Components 1-5 landed; every measured number above current.
- Full suite green (318 tests) and `tests/heavy/test_nest_identity.py` bit-exact
  on all eight configurations.
- Pinned pre-refactor references unchanged and passing.
- **The branch passed independent codex review (Thomas's explicit request) in
  two rounds — 12 findings then 8 — with all 20 resolved**, each with a
  regression test, and the feature-defeating ones verified non-vacuously by
  reintroducing the bug and watching the test fail. Round 2 confirmed the
  numerics clean and was confined to the resume/durability layer; the remaining
  boundary of that layer is recorded under Future work rather than left implicit.
- Remaining work is in Future work above; nothing there blocks merge.

## Conventions for this branch

- Match the repo's code style: dense, decision-recording comments citing
  rulings and dates where behavior is load-bearing; docstrings that explain
  WHY (see cascade_loop, _advance_flux for the register). Numbered audit
  items live in paper/audit-2026-07-23.md.
- Tests go in tests/ next to their subjects (test_refine.py is the model
  for identity-style tests). Heavy/measurement scripts in tests/heavy/.
- Run tests with `uv run pytest tests/ -x -q` from the repo root (per-project
  .venv via `uv sync`).
- Commit granularity: one component (or coherent slice of one) per commit,
  on this branch (`stream-to-disk`). Do not touch steam/constants.py's
  uncommitted working-tree change (a dropped comment; predates this branch;
  leave it uncommitted and unstaged).
- The paper (paper/) is out of scope on this branch.
