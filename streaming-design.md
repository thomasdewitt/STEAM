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

2. **Realized global reductions mid-class.** Per class the model takes: the
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

3. **Halo domain of dependence.** A tile descending multiple classes without
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

### 2. Measure/apply split of the class body  [delegated, after 1]

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

### 3. Tile store + streamed driver  [delegated, after 2]

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

### 4. Streamed composition, diagnostics, output  [delegated, after 3]

_compose_output (deficit add + final per-level projection), the saturation
adjustment / hydrostatic column solve (column-local), and NetCDF writing
(hyperslab per tile). The final projection is per-level: reuse the plane
pass. Nothing algorithmic changes.

### 5. The switch and lifecycle  [delegated, after 4]

`stream_to_disk=False/True` (consider 'auto' later, engaging on planner
overflow with a warning), `scratch_dir=` (default: system temp on the same
filesystem as the output path — scratch I/O is the cost driver; document
that it should be NVMe), final copy to requested output path, scratch
cleanup on success, retention + resume instructions on failure.

### 6. (Deferred) GPU-resident tiles

After streamed = non-streamed passes: tiles sized to VRAM instead of RAM,
per-tile work running device-resident end-to-end. Revisit the 2026-08-07
benchmark conclusions then. Nothing in 1-5 may preclude this: keep the
per-tile work unit a pure function of (slab arrays in, slab arrays out,
scalars) with `device=` plumbed through.

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
