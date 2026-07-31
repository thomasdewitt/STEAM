# STEAM memory & I/O investigation — staging, compression, diagnostics, co-residency, halo

2026-07-30 · follow-up to `perf-audit-2026-07-30.md` · all measurements on the
production box (9950X, 60 GB, RTX 5080), all benchmarks on **real production
fields** pulled from `runs/steam_sq10_icon_lem_m0{0,1}.nc`.
Scripts in `scratchpad/mem-io/`; scratch data in `/var/tmp/steam-bench-mem-io/`.
Nothing in `turbulon-model/` or `turbulon-analysis/runs/` was modified.

Ranking follows Thomas's clarification: **memory counts only where it moves a
binding peak** — the square cascade (18.0 GB), the nest cascade (31.5 GB, the
global peak and the real cap on nest size), and the 6.5 GB tmpfs staging that
competes with both. Transients below those peaks are reported but ranked low.

---

## 0. The audit's nest-peak attribution is wrong — corrected here first

The audit attributed the nest's 31.5 GB to the interpolation-compensation
replay ("several full parent-grid `inc` arrays live at once"). **It is not the
replay.** I ran `refine()`'s real prologue against a production parent with an
RSS sampler and aborted at the first call into `cascade_loop`
(`probe_nest_prologue.py`):

```
prologue peak RSS  =  8.52 GB      (parent read; the replay itself adds ~2.3 GB transient)
RSS entering cascade_loop =  1.30 GB
prologue wall = 31.4 s, 84 zoom_trilinear calls
```

`refine()` does `del h_3d, qt_3d, flux_3d` immediately after slicing the pads,
and the pads are tiny (2048 × 22 × 115 ≈ 21 MB each). The replay's zoomed `inc`
arrays are freed one per iteration. **The whole prologue is 8.5 GB and it is
not binding.**

The 31.5 GB is `cascade_loop`'s finest class, and it is exactly what
`simulate()`'s own preflight formula predicts:

| nest finest class 16384 × 140 × 364 | GiB |
|---|---:|
| one full-size float32 field | 3.11 |
| 8 × field (the preflight's own estimate) | 24.9 |
| + `fft_convolution_bytes` (CPU path, kz 7–25) | 3.6 – 5.0 |
| **predicted total** | **28.5 – 29.9** |
| **observed peak** (audit `VmHWM` 30.8 GB / 31.5 GB RSS) | **29.3** |

So the nest peak is **~9.4 full-size field-equivalents**, and the lever on nest
size is *the number of simultaneously-live full-size arrays in `cascade_loop`'s
inner loop* — nothing else. This retargets audit item §5.5 ("slice first in the
replay"), which was already deprioritized for being realization-changing: it is
now also **pointless**, since the replay is not the peak. Recommend dropping
§5.5 entirely.

---

## Ranked recommendations

### 1. Move class-increment staging off tmpfs — 5.7 GB of binding RAM, one line

**This is the whole 6.5 GB, and it is a one-line change.**

`simulate()` stages the 27 per-class `.npy` increments via
`tempfile.mkdtemp(prefix="steam_class_inc_")`. `TMPDIR`/`TMP`/`TEMP` are all
unset on this box, so `tempfile.gettempdir()` returns `/tmp`, which is
**tmpfs — RAM**:

```
/tmp      tmpfs   31G
/var/tmp  (on /dev/nvme0n1p3, real NVMe)
```

Measured (`bench_staging.py`), staging 5.79 GB of real finest-class increments:

| staging dir | save | load | **MemAvailable cost** |
|---|---:|---:|---:|
| `/tmp` (tmpfs) | 1.1 s (5263 MB/s) | 1.1 s | **−5.67 GB** |
| `/var/tmp` (NVMe) | 1.2 s (4713 MB/s) | 2.2 s | **−0.30 GB** |

Identical data, identical save throughput, **+1.1 s of load time, and 5.4 GB of
RAM handed back**. tmpfs pages are not reclaimable (only swappable) and are
charged to the writing process's cgroup; NVMe page-cache pages are reclaimable
under pressure. This is exactly the 6.5 GB the audit flagged, and it is the
cheapest item in this report by a wide margin.

The staging peaks at the *end* of the cascade (the finest class's three 1.93 GB
arrays are the last written) — i.e. it coincides with the square's 18.0 GB
cascade peak *and* with the co-resident nest's 31.5 GB. It is a direct
contributor to the 56 GB figure.

**Change** (`steam/simulate.py`, in `simulate()`, the `save_class_increments`
block):

```python
increment_dir = Path(tempfile.mkdtemp(
    prefix="steam_class_inc_", dir=Path(output_path).parent))
```

Staging next to the output guarantees the filesystem has room for it (it is
sized like the output) and keeps it off tmpfs on any Linux box, not just this
one. Worth a comment recording *why*, since the default looks harmless.

- **Bit-identical.** Same `.npy` bytes, different directory.
- **Risk: low.** One caveat: `/` is 93 % full (66 GB free) and holds both
  `runs/` and `/var/tmp`, so the transient 7 GB shares a tight filesystem with
  the 20–38 GB member files. A crash leaves a `steam_class_inc_*` dir beside the
  output instead of in `/tmp` — more visible, but also more likely to be noticed
  and cleaned.
- **Zero-code alternative for tonight's run:** `TMPDIR=/var/tmp python
  run_production_squares.py` gets the identical win without touching the repo,
  because `mkdtemp` honours `TMPDIR`. Good for an immediate run; the `dir=`
  change is the durable fix.

**Deeper variant (write increments straight into the dataset) — not
recommended as the first step.** It requires `simulate()` to create the NetCDF
*before* `cascade_loop` (currently `write_netcdf` opens mode `"w"`, which
truncates, and it runs *after* the cascade), `cascade_loop` to reopen mode
`"a"` per class, and `write_netcdf` to gain an append mode. That is ~40 lines
across `simulate.py` and `output.py` to save the round trip — worth only
**~5–8 s** and no additional RAM beyond what the one-liner already recovers
(page cache is reclaimable). Do it only if the staging directory is disliked on
its own merits.

---

### 2. `_gradient_components` fused stencil — est. ~10 % off *both* binding peaks

The largest remaining lever on nest size. At the peak instant inside the h
iteration the live full-size set is:

`h_perturbation`, `qt_perturbation`, `flux`, `S_k`, `running_sum`, plus
`_gradient_components`' three internal arrays (`grad_h`, the `grad` temporary,
`grad_z`) — **8 full-size arrays**, matching the 8× preflight formula and the
observed 9.4 field-equivalents once FFT buffers are added.

`_gradient_components` builds its result from four `np.roll` calls (each a full
copy) plus `np.gradient`. Its own docstring claims "~3 arrays"; a fused numba
stencil computing `grad_h` and `grad_z` in a single pass needs **2**, removing
one full-size array from the peak:

- nest: −3.11 GiB of 29.3 GiB → **~10 %**, i.e. ~10 % more nest cells for the
  same RAM
- square: −1.80 GiB of ~16.8 GiB
- and it is already audit item §5.6 for speed (~24 s, ~3× faster)

**Exactly invariant** — same central differences, same operation order, purely
elementwise plus `np.gradient`'s fixed stencil.

**Confidence: this one is estimated, not measured.** The array count is read
off the source and corroborated by the peak arithmetic closing to within 3 %,
but I did not implement the numba kernel, so the 10 % is a projection. It is
the best remaining memory-per-effort ratio, and it is the only item here that
makes nests *bigger*.

---

### 3. Drop the `before = perturbation_field.copy()` in increment recording — 1.8 GiB off the square peak · **needs coordination**

When `save_class_increments=True` (production does), `cascade_loop` allocates a
full-size `before = perturbation_field.copy()` per scalar, purely to recover the
added delta by subtraction afterwards. At the square's finest class that is
**1.80 GiB live inside the 18.0 GB peak — ~10 % of the square's binding peak**,
paid only because the increment is recorded.

It is removable if `_bounded_amplitude_add` writes its per-level solved `d` back
into `increment[:, :, lev]`, making `increment` *be* the added delta: one
strided write per level (~1 s), and `cascade_loop` then stores `increment`
directly with no copy and no subtract.

**⚠ Do not action this without talking to the agent currently rewriting
`_bounded_amplitude_add`.** Their in-flight version's docstring explicitly
*relaxes* this contract — "``increment`` is scratch (the solve works on
per-level copies, but no promise it survives)" — and adds a CUDA path
(`_bounded_amplitude_add_cuda`) which would need the same write-back. This
recommendation re-establishes a contract they just removed, so it is a design
question for the two changes together, not an independent edit.

- **Bit-identical** to the shipped field (the write-back changes only what
  `increment` holds on return, and the recorded increment is by construction the
  same array the subtraction produced today).
- Nest is unaffected — `refine()` never passes `increment_dir`.

---

### 4. Compression: replace zlib-4 with **blosc+zstd** — 2–8× faster writes, same or smaller files

Both filters are available in the venv (`netCDF4 1.7.4`, libnetcdf 4.9.3:
`__has_zstandard_support__ = 1`, `__has_blosc_support__ = 1`;
`nc-config --has-stdfilters -> bz2 deflate szip blosc zstd`).

**API note:** these are `compression="zstd"` / `compression="blosc_zstd"` +
`complevel`, *not* `zstd=True`/`blosc=...`. And `shuffle=True` is **silently
ignored** for non-zlib compressors — `filters()` reports `shuffle: False`. Byte
shuffling must be requested as `blosc_shuffle=1`, which is exactly why plain
`compression="zstd"` compresses so much worse than zlib below.

Measured on real production fields, 1.93 GB each, chunks `(64, 64, nz)` as
shipped (`bench_compress.py`; all variants verified `exact=True` on read-back):

| filter | h write | qt write | flux write | inc c08 write | **h+qt+flux on disk** |
|---|---:|---:|---:|---:|---:|
| none | 0.6 s | 0.6 s | 0.7 s | 0.6 s | 5.79 GB |
| **zlib-4 (SHIPPED)** | 20.7 s | 6.4 s | 18.2 s | 19.6 s | **3.23 GB** |
| zlib-1 | 12.5 s | 4.6 s | 15.1 s | 15.8 s | 3.26 GB |
| zstd-1 (no shuffle) | 2.1 s | 1.2 s | 1.7 s | 1.6 s | 3.81 GB |
| **blosc_zstd c1** | 1.8 s | 1.6 s | 2.2 s | 2.2 s | **3.40 GB** |
| **blosc_zstd c5** | 8.2 s | 9.8 s | 10.2 s | 14.7 s | **3.22 GB** |
| blosc_lz4 c5 | 0.9 s | 0.9 s | 0.9 s | 1.6 s | 3.58 GB |

Two clean options, both **bit-identical data** (compression is lossless; only
file bytes change):

- **`blosc_zstd` complevel 5 — strictly dominates zlib-4.** Same total size
  (3.22 vs 3.23 GB; qt is actually *better*, 3.88× vs 3.71×), **1.6–2.6×
  faster to write, 2–3× faster to read**. There is no axis on which zlib-4 wins.
  The safe swap.
- **`blosc_zstd` complevel 1 — the throughput choice.** 8–11× faster than
  zlib-4 for **+5 % bytes** (3.40 vs 3.23 GB). Given writes are 94.5 s/square
  and 5 % of 20 GB is 1 GB, this is the better trade for the production pipeline.

Projected on the square's shipped 94.5 s of write (`write_netcdf` 37.0 s +
`write_class_increments` 57.5 s): **→ ~11 s at blosc_zstd c1** (−84 s/square),
or ~55 s at c5.

Also useful: zlib **inflate** is already fast (660–1140 MB/s measured on the
production file), so the audit's "17 s to inflate the parent" is not a
decompression-rate problem — see item 7.

**Portability caveat (the real risk).** blosc and zstd are HDF5 *filter
plugins*; a reader without them cannot open the variable at all, whereas zlib is
universal. Verified on this box that the **system** netCDF (Fedora build,
entirely separate from the venv) reads both via `ncdump` — so
`turbulon-analysis` and command-line tooling here are fine. The venv has no
`h5py`/`xarray` to test. Flag before SFI/collaborator hand-off; for archival
products zlib-4 remains the conservative choice. If that matters, **zlib-1 is
the zero-risk fallback**: 1.4× faster than zlib-4 for +1 % bytes (the ratio
difference between zlib-1 and zlib-4 is negligible on these fields — 1.63 vs
1.65 on h, 1.24 vs 1.25 on flux).

**Change:** `steam/output.py`, `write_netcdf` and `write_class_increments`
(the `zlib=compress, complevel=4 if compress else 0` kwargs, 5 sites) and
`steam/thermodynamics.py` `compute_diagnostics` (1 site). Cleanest as a shared
helper returning the filter kwargs, so the choice lives in one place —
`constants.py` alongside `output_compress` would fit the house pattern.

---

### 5. The nest-written-uncompressed asymmetry — quantified both ways

Confirmed real: `run_nest` never passes `compress=`, so `refine()` falls back to
`constants.output_compress = False`. In `steam_sq10_icon_lem_m00.nc`,
`refinements/r0` h/qt/flux are all `zlib: False` — 16384 × 128 × 364, **9.16 GB
of payload stored raw**, which is the entire 38 GB vs 20 GB difference.
(The nest also has no diagnostics computed — `run_production_squares` calls
`compute_diagnostics` on the square only.)

Measured on **real nest data** (4096-column slab of `r0`, scaled ×4 —
`bench_nest_compress.py`):

| filter | full-nest write | on disk | vs shipped |
|---|---:|---:|---|
| **none (SHIPPED)** | **2.7 s** | **9.16 GB** | — |
| zlib-4 | 56.3 s | 4.80 GB | +53.6 s to save 4.36 GB |
| zlib-1 | 47.0 s | 4.86 GB | +44.3 s to save 4.30 GB |
| **blosc_zstd c1** | **8.9 s** | **5.01 GB** | **+6.2 s to save 4.15 GB** |
| blosc_zstd c5 | 44.0 s | 4.78 GB | +41.3 s to save 4.38 GB |

Nest fields compress well (h 1.83×, qt 4.47×, flux 1.25×) — better than the
square's, since the 375 m field is smoother per cell.

**This is Thomas's call, but the trade is now lopsided.** With zlib-4 the
question "is 4.4 GB worth 54 s?" is genuinely arguable. With `blosc_zstd c1` it
costs **6.2 s of the nest's 657 s (0.9 %) to save 4.15 GB per member** — 95 % of
the achievable saving for 12 % of the time. Across 6 members that is ~25 GB
recovered on a filesystem that is 93 % full, for 37 s total. If the compression
filter changes at all (item 4), turning the nest on becomes close to free.

No memory effect either way — NetCDF compresses chunk-by-chunk.

---

### 6. Diagnostics: continuously-fed pool **and** lower the worker cap — 86.7 s → 24 s

Working set is 8.9 GB, below both binding peaks, so this is ranked as a speed
item per the clarification. Prototyped and benchmarked against a real
2048² × 115 h/qt file (`bench_diag.py`; **every variant produced identical
blake2b digests of T, qv, qc, qi, p — bit-identity confirmed as the docstring
guarantees**):

| variant | wall | VmHWM | file |
|---|---:|---:|---:|
| **shipped, 8 w, zlib-4** | **86.7 s** | 8.9 GB | 4.84 GB |
| fed, 8 w, zlib-4 | 55.7 s | 10.7 GB | 4.84 GB |
| fed, 4 w, zlib-4 | 52.6 s | 6.1 GB | 4.84 GB |
| fed, 16 w, zlib-4 | 59.5 s | 16.1 GB | 4.84 GB |
| fed, 24 w, zlib-4 | 62.9 s | 16.2 GB | 4.84 GB |
| shipped, 8 w, **uncompressed** | 24.5 s | 8.8 GB | 11.34 GB |
| fed, 8 w, uncompressed | 22.9 s | 9.8 GB | 11.34 GB |
| fed, 2 w, blosc_zstd c1 | 31.0 s | 3.8 GB | 5.23 GB |
| fed, 4 w, blosc_zstd c1 | 28.7 s | 6.2 GB | 5.23 GB |
| fed, 6 w, blosc_zstd c1 | 24.7 s | 8.4 GB | 5.23 GB |
| **fed, 8 w, blosc_zstd c1** | **24.0 s** | 9.8 GB | 5.23 GB |
| fed, 16 w, blosc_zstd c1 | 27.2 s | 16.2 GB | 5.23 GB |

Three findings, the second of which is the important one:

1. The fed loop alone: **86.7 → 55.7 s (1.56×)** at +1.8 GB.
2. **Most of that win is overlapping *deflate*, not compute.** With compression
   off, shipped is 24.5 s and fed is 22.9 s — the batch structure barely matters.
   The shipped loop is slow because the serial zlib-4 leg is long and the pool
   sits idle through it. So **the filter change (item 4) is the real fix**, and
   `fed + blosc_zstd c1` reaches **24.0 s — the audit's ~25 s target — while the
   fed restructure alone plateaus at 52–56 s.**
3. **The `min(8, ...)` worker cap is too high, not too low.** Raising it is
   strictly bad (16 w is slower *and* 6 GB heavier than 8 w in every pairing).
   `fed, 4 w, zlib-4` beats `fed, 8 w, zlib-4` on **both** time and memory
   (52.6 s / 6.1 GB vs 55.7 s / 10.7 GB). The audit's "raise the cap from 8" is
   the wrong direction — the serial HDF5 leg is the floor, and extra workers only
   buy in-flight chunk memory (~850 MB per worker).

**Recommendation:** take the filter change; the fed loop is optional on top.
If both, `n_workers=6` (24.7 s / 8.4 GB) or 4 (28.7 s / 6.2 GB) is the
memory-sane operating point — do not raise the cap.

Prototype is `compute_diagnostics_fed` in `bench_diag.py`: keep exactly
`n_workers` chunks in flight, and on each `FIRST_COMPLETED` write that result
and immediately read + submit the next. Reads and writes stay on the main
thread (HDF5 is not re-entrant); only the ordering changes. Writes land in
completion order rather than chunk order, which is safe because each chunk owns
a disjoint x-slab — confirmed by the identical digests.

- **Bit-identical**, verified across 2/4/6/8/16/24 workers and 4 filters.
- **Risk: low.** ~25 lines replacing the batch loop in `compute_diagnostics`.

---

### 7. Nest parent read — 8.5 GB and ~17 s for data that is 94 % discarded (low rank: not binding)

Not a binding peak (8.5 GB < 31.5 GB), so ranked low, but it is nearly free and
it is the prologue's whole cost. `refine()` reads **entire** parent h, qt and
flux (`grp.variables["h"][:]`, 1.93 GB each, ×2 transiently for the
`.astype(np.float32)` copy → the measured 8.52 GB), then immediately slices out
a 2048 × 22 × 115 pad (~21 MB) and `del`s the originals.

For the production strip only `y ∈ [1013, 1035)` is ever used. With chunks
`(64, 64, 115)` that is 2 of 32 y-chunk columns — **a hyperslab read would touch
1/16 of the data**: prologue peak 8.5 GB → well under 1 GB, and the 31.4 s
prologue (of which ~17 s is the read) → a few seconds.

**Bit-identical**: a NetCDF hyperslab returns the same floats as slicing the
full read. The only care needed is that `y_indices` is computed modulo
`parent_ny` for a periodic parent and can wrap; handle it as one or two
contiguous ranges, or fall back to the current full read when it wraps.

Note this is *not* audit §5.5 (cropping before the replay zooms), which changes
the interpolation stencil and is realization-changing. This one is upstream of
that, and exact.

---

### 8. Co-residency guard — recommendation: a cgroup scope, **not** a preflight

**Recommendation: run the driver under a systemd scope with a hard cap,**
matching the pattern already used for the audit runs:

```
systemd-run --user --scope -p MemoryMax=50G -p MemorySwapMax=0 \
    python run_production_squares.py
```

Rationale — and specifically why *not* the two alternatives:

- **A preflight in `refine()` would not fire.** This is the decisive point.
  Both processes ramp gradually and each passes its own check at start time.
  The driver runs `run_square(m+1)` in the main process immediately after
  submitting nest *m*; at that instant the freshly-spawned nest has barely
  allocated, so the square's existing `available_memory_bytes()` preflight sees
  plenty free and passes. Symmetrically, the nest's preflight would run while
  the *previous* square has already exited. The 56 GB collision happens later,
  when both are at peak, and no start-of-run checkpoint sees it.
  `MemAvailable` is global, so a preflight *would* work if the peaks were
  simultaneous at launch — they are not.
- **Cross-process ordering (serialize square and nest) works but is expensive** —
  it forfeits the overlap, ~508 s per member, and the pipelining is the only
  reason the members are not 1166 s each.
- **A cgroup cap is enforced continuously, not at a checkpoint** — the only
  mechanism that actually matches the failure mode. It protects the login
  session absolutely (the hazard the audit named); the victim is inside the
  scope, and the driver is already restartable in both directions ("members
  whose .nc exists are skipped; nests whose group exists are skipped"), so an
  OOM-killed square or nest is simply redone.

Zero repo change, one command, house-consistent.

**Important coupling:** tmpfs pages are charged to the cgroup that faults them
in, and cannot be reclaimed under cap pressure — so with the current `/tmp`
staging, 6.5 GB of the cap is spent on files. **Item 1 is effectively a
prerequisite** for the cap to be set at a sane value. With item 1 done, peak
concurrent demand falls from ~56 GB to ~49.5 GB and a 50 GB cap has real
headroom on a 60 GB box.

If a code-level guard is also wanted for `refine()` on its own merits (it is the
only entry point with no preflight at all, and it is the 31.5 GB one), mirror
`simulate()`'s formula — `8 * field_bytes + fft_convolution_bytes(...)` for the
finest class, which I verified predicts the observed peak to within 3 % (§0).
That is worth having for direct `refine()` calls outside the pipeline; it just
does not solve co-residency.

---

### 9. Nest halo — exactly right, no action

**The halo is exactly `SUPPORT_FACTOR` and not one cell more. Dropping it.**

`pad_y_per_class = SUPPORT_FACTOR * k_values` (physical metres), and each class
is gridded at `dx = k/2`, so the halo is **uniformly 6 cells per side at every
class** — identically the kernel's own half-width
`half_ny = ceil(SUPPORT_FACTOR * k / dy)`:

| class | k (m) | dy (m) | halo = 3k/dy | kernel half-width | y cells |
|---|---:|---:|---:|---:|---|
| 1 | 3000 | 1500 | 6 | 6 | 64 + 12 = 76 |
| 2 | 1500 | 750 | 6 | 6 | 64→... 76 |
| 3 | 750 | 375 | 6 | 6 | **128 + 12 = 140** |

A turbulon centred one cell beyond the halo contributes exactly zero to the
domain; one cell less and it would not. The 9 % (12/140) is the minimum price of
a correct convolution, not waste. The code comment already says so — "the
kernel's own reach, so it shrinks with the cascade and stays a constant number
of cells" — and it is correct. The only way to shrink it is to lower
`SUPPORT_FACTOR`, which the comment at the top of `simulate.py` rules out
explicitly as realization-changing.

One observation offered without recommendation: strictly, the halo covers the
*convolution's* reach but not the *gradient* chain — `W` computed in the outer
halo cells is itself built from a field missing turbulons beyond the halo, and
the convolution then carries that inward. Exactness against an unnested run
would want ~2× the halo. That is an *increase*, and the design already accepts
it as "the deliberate price of nesting". No action.

---

## Summary table

| # | change | binding-peak effect | speed effect | identity | risk |
|---|---|---|---|---|---|
| 1 | staging `dir=` off tmpfs | **−5.7 GB** (both peaks) | +1.1 s | bit-identical | low |
| 2 | fused `_gradient_components` | **−3.11 GiB nest (~10 %)**, −1.80 GiB square *(estimated)* | −16 s | exactly invariant | med (new numba kernel) |
| 3 | drop `before` copy | −1.80 GiB square (~10 %) | −1 s | bit-identical | **coordinate w/ in-flight work** |
| 4 | zlib-4 → blosc_zstd c1 | none | **−84 s/square** | bit-identical data | low (filter portability) |
| 5 | compress the nest (c1) | none | +6.2 s, **−4.15 GB disk/member** | bit-identical data | low · Thomas's call |
| 6 | fed diagnostics + cap 8→6 | −0.5 GB (non-binding) | **86.7 → 24 s** | bit-identical (verified) | low |
| 7 | hyperslab parent read | −7.5 GB (non-binding) | −15 s | bit-identical | low |
| 8 | `systemd-run` MemoryMax scope | safety | none | n/a | low · needs #1 |
| 9 | nest halo | — | — | — | **no action** |

Per-member totals if 1 + 4 + 5 + 6 are taken (the low-risk set): square
508 → ~400 s, nest 657 → ~663 s, **6.5 GB of RAM returned and ~4 GB/member of
disk saved** — independent of, and composable with, the `_bounded_amplitude_add`
work in flight.

---

## Reproduction

```
scratchpad/mem-io/extract_field.py        # pull real h/qt/flux/increment from a production member
scratchpad/mem-io/bench_compress.py       # §4  filter A/B on 4 real fields
scratchpad/mem-io/bench_nest_compress.py  # §5  filter A/B on real nest data
scratchpad/mem-io/bench_staging.py        # §1  tmpfs vs NVMe staging, MemAvailable delta
scratchpad/mem-io/bench_diag.py V NW F    # §6  diagnostics A/B + digests
scratchpad/mem-io/probe_nest_prologue.py  # §0  real refine() prologue, RSS-sampled, aborts at cascade_loop
```
Logs alongside as `bench_*.log`. Scratch arrays and work files are in
`/var/tmp/steam-bench-mem-io/` (~14 GB) — delete when done; `/var/tmp/steam-audit`
from the earlier audit no longer exists.
