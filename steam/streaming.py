"""Out-of-core execution of the STEAM cascade: tile store, planner, driver.

The streamed run is the SAME cascade with the loops reordered. Component 1 made
the noise a pure function of (root seed, class index, world site), so a region's
noise is the restriction of the root's; component 2 hoisted every realized
global reduction into a measure/reduce/apply seam with additive partials. What
is left, and what this module is, is the driver: page the working state through
RAM in horizontal tiles, with the full state living on scratch disk.

Per class below the RAM horizon:

  pass 0  REGRID    previous class's store -> this class's grid, tile by tile
  pass 1  MEASURE   per tile: regenerate noise, form the RAW product,
                    accumulate flux and product-norm partials
          reduce    add the partials (steam.ledger's `+`)
  pass 2  APPLY     per tile: recompute (keyed noise makes this exact), apply
                    the global scalars, convolve, update the flux, write the
                    state and the candidate scalar increments
          reduce    the post-clip flux mean -> one rescale scalar per class
  pass 3  PLANE     per z-level: assemble the candidate increment and the state
                    across the whole domain, run the EXISTING bounded-add solve
                    on the assembled plane, apply, write back

WHY THE STORE HOLDS WORLD-SIZED FIELDS, NOT TILES. Tiles are a partition of the
WORK, not of the data: horizontal tiling never splits z, and every tile's
statistics are the whole domain's (a streamed run is a root, so there is no
halo in the stored state at all -- halos are transient, built on page-in by
wrapping). One memory-mapped .npy per (class, field) therefore serves both
access patterns the driver needs -- a tile plus halo is a strided slice, a
z-plane across all tiles is `array[:, :, lev]` -- with none of the stitching
bookkeeping a per-tile store would need. Fewer places to get the indexing
wrong, which is the point: this is the area that produced the
align_corners=True finding (2026-07-31) and the never-periodic resample
(2026-08-06).

WHAT IS NOT HERE (component 4): the output composition, the diagnostics column
solve, and NetCDF writing all stay in-RAM. This module produces the cascade
STATES and the ledger, which is what the fundamental test compares.
"""

from math import gcd
from pathlib import Path
import hashlib
import json
import os
import shutil

import numpy as np
import torch

from .ledger import (
    BoundedAddLedger,
    ClassLedger,
    FluxAdvanceLedger,
    FluxOutputLedger,
    MeanReduction,
    PatternLedger,
    ProjectionLedger,
    RunLedger,
    load_class_ledger,
    save_class_ledger,
)

# Field names the store carries between classes. The deficits are absent on
# purpose: they exist only for the output composition (component 4), the cascade
# state never reads them, and carrying them would need the pre-advance flux kept
# live across the lazy rescale.
PROGRESS_FILE = 'progress'
MANIFEST_FILE = 'manifest.json'

STATE_FIELDS = ('h_perturbation', 'qt_perturbation', 'flux')
# Written within a class by pass 2 and consumed by pass 3.
CANDIDATE_FIELDS = ('h_increment', 'qt_increment')
# The interpolation-compensation deficits: sum over classes of (f_i(z) - 1)
# times the increment that class actually added. Carried between classes down
# the same regrid chain as the state, and allocated lazily at the first class
# with f != 1, exactly as cascade_loop does -- the well-resolved classes carry
# none, so on a production ladder these exist only for the last nine.
DEFICIT_FIELDS = {'h': 'h_deficit', 'qt': 'qt_deficit', 'flux': 'flux_deficit'}


# ---------------------------------------------------------------------------
# Tiled regrid -- the risk concentration, so it gets its own exact statement
# ---------------------------------------------------------------------------

def _axis_regrid_plan(n_in, n_out, lo, hi):
    """Source slab and crop that reproduce target cells [lo, hi) of a resample.

    Under align_corners=False the map from target to source is a pure dilation:
    target cell j samples source coordinate (j + 0.5) * n_in/n_out - 0.5. Write
    n_in/n_out as the reduced fraction b/a (b = n_in/g, a = n_out/g with
    g = gcd). Then b source cells span exactly a target cells, so a source slab
    whose ends are multiples of b lands on target-cell boundaries that are
    multiples of a. That is what makes a tiled regrid possible at all: no tile
    needs the whole source array, only a slab quantized to b.

    The slab carries ONE EXTRA BLOCK of b source cells on each side, wrapped,
    and the matching a target cells are cropped off -- the same trick, and the
    same reason, as utils._wrap_plan. Upsampling means n_in/n_out < 1, so the
    first target cell of a block samples a NEGATIVE source coordinate
    (0.5 * n_in/n_out - 0.5 < 0) and the last reaches past the end: without the
    extra block torch would clamp there, replicating the slab's outer half cell
    instead of continuing the field. Clamping at a TILE boundary is worse than
    clamping at the domain boundary, because it puts a seam in the interior --
    which is exactly what the seam test would catch, and did (2026-08-12).

    Returns (src_lo, src_hi, crop_lo, out_len): read source cells
    [src_lo, src_hi) (modularly, on a periodic axis), interpolate to out_len
    target cells, and keep [crop_lo, crop_lo + (hi - lo)).
    """
    if n_out < n_in:
        raise NotImplementedError(
            f"tiled regrid downsamples ({n_in} -> {n_out}); the cascade only "
            f"ever goes finer, and zoom_trilinear refuses this too")
    common = gcd(n_in, n_out)
    b = n_in // common
    a = n_out // common
    block_lo = lo // a
    block_hi = -(-hi // a)          # ceil
    return (block_lo * b - b, block_hi * b + b,
            a + (lo - block_lo * a), (block_hi - block_lo) * a + 2 * a)


def _read_wrapped(source, x_lo, x_hi, y_lo, y_hi):
    """A slab of a world-sized array, wrapping x and y at the domain edges.

    The wrap is the cascade's own periodicity, and it is what lets a tile at the
    domain edge see its true neighbours -- the same modular extraction refine()
    uses to slice a spanning nest out of its parent.
    """
    nx, ny = source.shape[0], source.shape[1]
    x_index = np.arange(x_lo, x_hi) % nx
    y_index = np.arange(y_lo, y_hi) % ny
    # Take x runs then y runs rather than one fancy-index gather: on a memmap
    # that is a handful of contiguous reads instead of a random-access walk.
    rows = np.concatenate([np.asarray(source[start:stop])
                          for start, stop in _runs(x_index)], axis=0)
    return np.concatenate([rows[:, start:stop]
                           for start, stop in _runs(y_index)], axis=1)


def _runs(index):
    """Split a modular index array into (start, stop) contiguous runs."""
    if len(index) == 0:
        return []
    breaks = np.flatnonzero(np.diff(index) != 1) + 1
    out = []
    for piece in np.split(index, breaks):
        out.append((int(piece[0]), int(piece[-1]) + 1))
    return out


def _interpolate_slab(slab, out_shape):
    """torch trilinear interpolate, exactly as steam.utils._zoom calls it.

    The wrapped pad _zoom applies is absent here because the caller already
    extracted a modular slab, which is the same continuation by other means.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(slab, dtype=np.float32))
    out = torch.nn.functional.interpolate(
        tensor[None, None], size=tuple(int(n) for n in out_shape),
        mode='trilinear', align_corners=False)[0, 0].numpy()
    return np.ascontiguousarray(out, dtype=np.float32)


def regrid_window(read_slab, source_shape, target_shape, window, scale=None):
    """Target cells ``window`` of ``zoom_trilinear(source, target_shape)``.

    ``read_slab(x_lo, x_hi, y_lo, y_hi)`` returns that (wrapped) slab of the
    source as a resident (nx, ny, nz) array -- a callable rather than an array
    because the source lives in the store's z-major layout, and only the store
    should know that.

    ``window`` is (x_lo, x_hi, y_lo, y_hi) in target cells; z is never tiled, so
    the whole z extent comes through and its non-periodic (clamped) end
    behaviour is the global one rather than a slab's.

    ``scale`` multiplies the source slab BEFORE interpolation, which is where
    the flux's lazily-folded per-class rescale is applied: the in-RAM path
    rescales at the end of a class and regrids at the start of the next, so
    doing it in this order is what keeps the two paths in step.
    """
    x_lo, x_hi, y_lo, y_hi = window
    nx_in, ny_in, nz_in = source_shape
    nx_out, ny_out, nz_out = target_shape

    src_x_lo, src_x_hi, crop_x, out_x = _axis_regrid_plan(nx_in, nx_out, x_lo, x_hi)
    src_y_lo, src_y_hi, crop_y, out_y = _axis_regrid_plan(ny_in, ny_out, y_lo, y_hi)

    slab = read_slab(src_x_lo, src_x_hi, src_y_lo, src_y_hi)
    if scale is not None:
        slab = slab * np.float32(scale)
    resampled = _interpolate_slab(slab, (out_x, out_y, nz_out))
    return np.ascontiguousarray(
        resampled[crop_x:crop_x + (x_hi - x_lo),
                  crop_y:crop_y + (y_hi - y_lo)])


# ---------------------------------------------------------------------------
# Tile geometry
# ---------------------------------------------------------------------------

def tile_bounds(n, n_tiles):
    """Split n cells into n_tiles as evenly as possible, low tiles taking the
    remainder. Handles a tile count that does not divide the domain, which is
    the common case once the grid stops being a power of two."""
    if n_tiles < 1 or n_tiles > n:
        raise ValueError(f"cannot split {n} cells into {n_tiles} tiles")
    edges = [(index * n) // n_tiles for index in range(n_tiles + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n_tiles)]


def tile_windows(nx, ny, tiles_x, tiles_y):
    """Every tile's (x_lo, x_hi, y_lo, y_hi), in WORLD RASTER ORDER.

    The order is fixed and documented because the partial sums of pass 1 are
    accumulated in it: float64 addition is not associative, so two streamed runs
    of the same configuration agree bit-for-bit only if they visit tiles in the
    same order. x-major, then y.
    """
    return [(x_lo, x_hi, y_lo, y_hi)
            for x_lo, x_hi in tile_bounds(nx, tiles_x)
            for y_lo, y_hi in tile_bounds(ny, tiles_y)]


# ---------------------------------------------------------------------------
# The scratch store
# ---------------------------------------------------------------------------

class TileStore:
    """World-sized float32 fields on scratch disk, one .npy per (class, field).

    Memory-mapped, so the process's own allocations stay bounded by a tile plus
    its halo and the kernel manages the rest as reclaimable page cache. Writes
    are flushed per tile rather than left to accumulate as dirty pages, which is
    what keeps a streamed class from quietly turning into a memory-resident one.

    STORED Z-MAJOR, as (nz, nx, ny), while every caller sees the cascade's usual
    (nx, ny, nz) through a transposed view. The reason is I/O amplification on
    the per-level passes -- the bounded add's plane pass, and component 4's
    projection and composition. On a C-ordered (nx, ny, nz) field the elements
    of one z-level are nz * 4 bytes apart, so a 4 KiB page holds a handful of
    them and reading ONE level touches essentially every page of the file; with
    the field larger than RAM, as it is whenever streaming is worth doing, each
    level then re-reads the whole field from disk. Measured on a 256^3 field at
    nz = 256 with the cache cold (tests/heavy/plane_pass_io.py):

        per-level pass, (nx, ny, nz):   256.0x amplification   <- exactly nz
        per-level pass, (nz, nx, ny):    13.3x

    and the 13.3x is itself pessimistic, an artifact of evicting between levels:
    a real plane pass walks the file sequentially, which is the one pattern
    readahead is built for. Z-major costs the tile reads a factor of ~2 at
    moderate tile widths (4.0x against 2.4x at 128 cells) and nothing at
    production widths, where a row-run is a page or more. That is the trade, and
    it is the right way round: per-level passes outnumber tile passes once the
    composition lands, and 256x is not a constant factor.

    Kept on failure, deleted on success: a crashed run's scratch is the only
    thing that makes it resumable, and the progress marker below is what a
    resume would read. (Resume itself is component 5's lifecycle work; the
    marker exists from the start so it costs nothing to add later.)
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._open = {}          # logical (nx, ny, nz) views
        self._raw = {}           # the (nz, nx, ny) buffers they are views of

    def _path(self, class_index, name):
        return self.directory / f"c{class_index:02d}_{name}.npy"

    def create(self, class_index, name, shape, fill=None):
        """Allocate a field on disk, given its LOGICAL (nx, ny, nz) shape.

        Stored as (nz, nx, ny); the returned array is the (nx, ny, nz) view, so
        callers never see the layout. ``fill`` writes a constant (the flux
        enters the cascade at one; the perturbations at zero).
        """
        nx, ny, nz = (int(n) for n in shape)
        path = self._path(class_index, name)
        raw = np.lib.format.open_memmap(
            path, mode='w+', dtype=np.float32, shape=(nz, nx, ny))
        if fill is not None:
            raw[...] = np.float32(fill)
            raw.flush()
        self._raw[(class_index, name)] = raw
        view = raw.transpose(1, 2, 0)
        self._open[(class_index, name)] = view
        return view

    def open(self, class_index, name):
        """The single memory map of one field.

        Always mapped read-WRITE, and cached, so there is exactly one mapping
        per file. Two mappings of the same file -- one read-only for the tile
        reads, one writable for the tile writes -- would let a read see stale
        pages behind a write, which is the kind of bug that shows up as a seam
        and takes a day to find.
        """
        key = (class_index, name)
        if key in self._open:
            return self._open[key]
        raw = np.load(self._path(class_index, name), mmap_mode='r+')
        self._raw[key] = raw
        view = raw.transpose(1, 2, 0)
        self._open[key] = view
        return view

    def read_window(self, class_index, name, x_lo, x_hi, y_lo, y_hi):
        """A tile plus halo as a contiguous (nx, ny, nz) array, x/y wrapped.

        Reads through the RAW (nz, nx, ny) buffer, where the innermost axis is
        y: a window is then one contiguous run per (level, x) rather than the
        element-by-element gather that slicing the transposed view would be.
        Measured at toy scale, going through the view instead cost 8x on the
        tile passes -- all of it strided-copy overhead, none of it I/O.
        """
        raw = self.plane_array(class_index, name)
        nx, ny = raw.shape[1], raw.shape[2]
        x_runs = _runs(np.arange(x_lo, x_hi) % nx)
        y_runs = _runs(np.arange(y_lo, y_hi) % ny)
        columns = np.concatenate(
            [np.concatenate([raw[:, x_start:x_stop, y_start:y_stop]
                             for y_start, y_stop in y_runs], axis=2)
             for x_start, x_stop in x_runs], axis=1)
        return np.ascontiguousarray(columns.transpose(1, 2, 0))

    def write_window(self, class_index, name, x_lo, x_hi, y_lo, y_hi, data):
        """Write an (nx, ny, nz) tile back. No wrap: a tile's OWN region never
        crosses the domain edge (only its halo does, and halos are discarded)."""
        raw = self.plane_array(class_index, name)
        raw[:, x_lo:x_hi, y_lo:y_hi] = data.transpose(2, 0, 1)

    def plane_array(self, class_index, name):
        """The raw (nz, nx, ny) array, whose ``[level]`` is ONE CONTIGUOUS plane.

        This is the whole point of the z-major layout: a per-level pass reads
        and writes nx*ny contiguous bytes per level instead of touching every
        page of the field. Use it for anything that iterates over z; use
        ``open`` for anything that works in tiles.
        """
        self.open(class_index, name)
        return self._raw[(class_index, name)]

    def exists(self, class_index, name):
        return self._path(class_index, name).exists()

    def close(self, class_index, name):
        self._open.pop((class_index, name), None)
        raw = self._raw.pop((class_index, name), None)
        if raw is not None and hasattr(raw, 'flush'):
            raw.flush()

    def drop(self, class_index, name):
        """Delete a field once nothing will read it again -- what keeps peak
        scratch at two classes' worth rather than the whole cascade's."""
        self.close(class_index, name)
        path = self._path(class_index, name)
        if path.exists():
            path.unlink()

    def rename(self, class_index, name, new_name):
        """Move a field, for a double-buffer swap. Cheap: one directory entry.

        os.replace rather than unlink-then-rename: atomic, so a crash cannot
        leave the target missing, and the whole operation is then idempotent
        (source gone means it already happened).
        """
        self.close(class_index, name)
        self.close(class_index, new_name)
        source = self._path(class_index, name)
        if not source.exists():
            return
        os.replace(source, self._path(class_index, new_name))

    def mark_atomic(self, class_index, phase, pending_renames=(),
                    pending_drops=()):
        """Commit a phase marker ATOMICALLY, recording its settle actions.

        The crash-consistency protocol (2026-08-12, codex review). Every phase
        is: pure writes to *_next names, flush, persist whatever the marker
        implies exists, ATOMIC marker commit, then SETTLE (the renames and
        drops). The marker is written to a temp file and os.replace'd, which is
        atomic on POSIX, so a crash either leaves the marker absent -- all the
        phase's inputs are still intact and it simply re-runs -- or present with
        its settle actions recorded, and the settle is idempotent and re-run on
        resume.

        Without this the windows were real: a crash between a rename and its
        marker stranded the resume with missing inputs, and a crash between two
        renames of a multi-field settle left half the fields swapped.
        """
        record = {
            'class': int(class_index), 'phase': phase,
            'renames': [list(pair) for pair in pending_renames],
            'drops': list(pending_drops),
        }
        settle_path = self.directory / f"settle_{class_index:02d}_{phase}.json"
        temporary = settle_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record))
        os.replace(temporary, settle_path)
        self.mark(class_index, phase)
        self.settle(class_index, phase)

    def settle(self, class_index, phase):
        """Perform (or finish) a marked phase's renames and drops.

        Idempotent by construction: a rename whose source is gone was already
        done, and a drop of a missing file is a no-op. Re-run on every resume for
        every marked phase, which is what makes the restart rule a single clause
        -- before the marker, re-run the phase; after it, settle and continue.
        """
        settle_path = self.directory / f"settle_{class_index:02d}_{phase}.json"
        if not settle_path.exists():
            return
        try:
            record = json.loads(settle_path.read_text())
        except ValueError:
            return
        for source, target in record.get('renames', []):
            source_path = self._path(class_index, source)
            if source_path.exists():
                self.rename(class_index, source, target)
        for name in record.get('drops', []):
            self.drop(class_index, name)

    def settle_all(self, completed):
        """Finish the settle of every already-marked phase, before resuming."""
        for class_index, phase in sorted(completed):
            self.settle(class_index, phase)

    def mark(self, class_index, phase):
        """Record that (class, phase) COMPLETED. Appended, so the file is a log
        rather than a single value: what resume needs is the last completed
        phase, and an append cannot lose an earlier one to a partial write."""
        with open(self.directory / PROGRESS_FILE, 'a') as handle:
            handle.write(f"{class_index} {phase}\n")
            handle.flush()
            os.fsync(handle.fileno())

    def completed(self):
        """The set of (class_index, phase) pairs already finished."""
        path = self.directory / PROGRESS_FILE
        if not path.exists():
            return set()
        done = set()
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                done.add((int(parts[0]), parts[1]))
        return done

    def write_manifest(self, manifest):
        (self.directory / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=1, sort_keys=True))

    def read_manifest(self):
        path = self.directory / MANIFEST_FILE
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except ValueError:
            return None

    def total_bytes(self):
        return sum(path.stat().st_size
                   for path in self.directory.glob("*.npy"))

    OWNED_SUFFIXES = ('.npy',)
    OWNED_NAMES = (PROGRESS_FILE, MANIFEST_FILE)

    def destroy(self):
        """Remove the files this store owns, then the directory if it empties.

        NOT rmtree (2026-08-12, codex review + coordinator): scratch_dir is a
        caller-supplied path, and a blanket rmtree of it would delete whatever
        else happens to live there. Only the store's own artifacts go -- field
        .npy files, per-class ledger npzs, the progress log, the manifest -- and
        the directory is removed only if that leaves it empty. Anything foreign
        survives and keeps the directory alive, which is the visible signal that
        something unexpected was in there.
        """
        for key in list(self._open):
            self.close(*key)
        if not self.directory.exists():
            return
        for path in list(self.directory.iterdir()):
            if path.is_dir():
                # The increments sub-store owns itself; recurse only into ours.
                if path.name == 'increments' or path.name == 'head_increments':
                    shutil.rmtree(path, ignore_errors=True)
                continue
            owned = (path.suffix in self.OWNED_SUFFIXES
                     or path.name in self.OWNED_NAMES
                     or (path.name.startswith('ledger_c')
                         and path.suffix == '.npz')
                     or path.name.startswith('settle_'))
            if owned:
                path.unlink(missing_ok=True)
        try:
            self.directory.rmdir()
        except OSError:
            pass        # something foreign is still in there: leave it be


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

# The cascade's concurrent peak in units of one field, measured 2026-07-30 and
# used by simulate()'s preflight. Reused here per TILE, since a tile runs the
# same class body over a smaller array.
PEAK_FIELDS = 10.5


class TilePlan:
    """Which classes stream, and over how many tiles.

    ``horizon`` is the first class that streams: classes below that index run
    in RAM on the resident array exactly as they do today. ``tiles`` is one
    (tiles_x, tiles_y) per streamed class -- tile counts grow down the cascade
    as the grid does.
    """

    __slots__ = ('horizon', 'tiles', 'halo', 'peak_scratch_bytes')

    def __init__(self, horizon, tiles, halo, peak_scratch_bytes):
        self.horizon = int(horizon)
        self.tiles = dict(tiles)
        self.halo = tuple(int(h) for h in halo)
        self.peak_scratch_bytes = int(peak_scratch_bytes)

    def __repr__(self):
        return (f"TilePlan(horizon={self.horizon}, tiles={self.tiles}, "
                f"halo={self.halo}, "
                f"scratch={self.peak_scratch_bytes / 1024**3:.2f} GiB)")


def _field_bytes(grids, index):
    return (int(grids['nx'][index]) * int(grids['ny'][index])
            * int(grids['nz'][index]) * 4)


def _working_bytes(nx, ny, nz, kernel_nz, device):
    """RAM one class body needs over an (nx, ny, nz) array."""
    from .utils import fft_convolution_bytes
    total = int(PEAK_FIELDS * nx * ny * nz * 4)
    if device != 'cuda':
        total += fft_convolution_bytes(nx, ny, nz, kernel_nz, 4)
    return total


def plan_tiling(grids, kernel_shape, budget_bytes=None, device='cpu',
                force=None, save_increments=False):
    """Choose the RAM horizon, the per-class tile grids, and the halo.

    ``force`` is (horizon, tiles_x, tiles_y) and overrides the search entirely:
    every test runs at a toy scale where nothing would tile on its own, so
    forcing is how the streamed path is exercised at all. The forced tile counts
    apply to every streamed class.

    Halo per side, in cells of the class's own grid, and constant across classes
    because the kernel is: the convolution reaches ``kernel_shape[axis] // 2``
    cells (SUPPORT_FACTOR * k in physical units, which is 6 * s_axis cells at
    every class), and the advective weight's gradient stencil reaches one more.
    Anything beyond that is slack, and slack in a halo is only cost.

    PER AXIS, and that is the point. The kernel's cell extent scales with the
    sparsity factor of ITS OWN axis, so at s = (3, 1, 1) the kernel is
    (37, 13, 13) and x needs 19 cells of halo where y needs 7. Deriving both
    from the VERTICAL width -- which is what this did until 2026-08-12 -- left
    the x halo at 7 and let the convolution's wrap contamination reach the inner
    region: measured h/qt errors ~7e-3 with 6.2x and 17.9x seam enrichment
    (codex review). Every test on this branch had used isotropic sparsity, which
    is exactly why it survived: with s_x == s_y == s_z the wrong derivation gives
    the right number.
    """
    n_classes = len(grids['k'])
    try:
        kernel_shape = tuple(int(n) for n in kernel_shape)
    except TypeError:
        kernel_shape = ()
    if len(kernel_shape) != 3:
        raise ValueError(
            f"plan_tiling needs the kernel's full 3D shape to size the halo per "
            f"axis, got {kernel_shape!r}")
    kernel_nz = kernel_shape[2]
    halo = (kernel_shape[0] // 2 + 1, kernel_shape[1] // 2 + 1)

    if force is not None:
        horizon, tiles_x, tiles_y = force
        tiles = {index: (tiles_x, tiles_y)
                 for index in range(int(horizon), n_classes)}
    else:
        from .utils import available_memory_bytes, MEMORY_HEADROOM_BYTES
        if budget_bytes is None:
            available = available_memory_bytes()
            budget_bytes = ((available - MEMORY_HEADROOM_BYTES)
                            if available is not None else None)
        if budget_bytes is None:
            raise RuntimeError(
                "plan_tiling cannot read available memory and was given no "
                "budget_bytes; pass one rather than guessing")
        horizon = n_classes
        for index in range(n_classes):
            if _working_bytes(int(grids['nx'][index]), int(grids['ny'][index]),
                              int(grids['nz'][index]), kernel_nz,
                              device) > budget_bytes:
                horizon = index
                break
        tiles = {}
        for index in range(horizon, n_classes):
            nx, ny, nz = (int(grids['nx'][index]), int(grids['ny'][index]),
                          int(grids['nz'][index]))
            tiles_x = tiles_y = 1
            # Powers of two, split on whichever axis is longer in cells, until
            # the tile plus its halo fits. A tile must stay wider than the halo
            # or the halo costs more than the tile.
            while True:
                tile_nx = -(-nx // tiles_x) + 2 * halo[0]
                tile_ny = -(-ny // tiles_y) + 2 * halo[1]
                if _working_bytes(tile_nx, tile_ny, nz, kernel_nz,
                                  device) <= budget_bytes:
                    break
                if (nx // tiles_x <= 2 * halo[0]
                        and ny // tiles_y <= 2 * halo[1]):
                    raise MemoryError(
                        f"class {index} ({nx}x{ny}x{nz}) cannot be tiled small "
                        f"enough to fit {budget_bytes / 1024**3:.1f} GiB: the "
                        f"halo ({halo[0]} cells per side) would dominate the "
                        f"tile. Reduce the vertical extent or raise the budget.")
                if nx // tiles_x >= ny // tiles_y:
                    tiles_x *= 2
                else:
                    tiles_y *= 2
            tiles[index] = (tiles_x, tiles_y)

    # Peak scratch. Per class the store can hold, at the widest moment (pass 0,
    # which has both classes live):
    #   previous class: 3 state + 3 deficit                 = 6 fields
    #   this class:     3 state + 3 deficit + 2 candidate
    #                   + 1 flux double-buffer
    #                   + 2 plane-pass companions           = 11 fields
    # The plane-pass companions (state_next, deficit_next) exist one scalar at a
    # time and are what makes a crashed pass re-runnable (component 5).
    # The deficits exist only from the first class with f != 1 onward, and the
    # candidates only within a class; counting them always is deliberate --
    # a planner that UNDER-predicts is worse than no planner, because the run
    # dies deep instead of refusing up front. (Caught 2026-08-12: adding the
    # deficits made the old 3+5 estimate under-predict, and the accounting test
    # failed rather than the run.)
    peak = 0
    for index in range(max(horizon, 1), n_classes):
        peak = max(peak, 6 * _field_bytes(grids, index - 1)
                   + 11 * _field_bytes(grids, index))
    if horizon == 0:
        peak = max(peak, 11 * _field_bytes(grids, 0))
    # Every other scratch tenant, none of which the first version counted
    # (2026-08-12, codex review). An under-predicting planner is worse than no
    # planner: the run dies deep instead of refusing up front.
    if save_increments:
        # Three applied increments per class, kept until the output file is
        # written. A geometric pyramid over the ladder, not a per-class cost.
        peak += 3 * sum(_field_bytes(grids, index)
                        for index in range(horizon, n_classes))
        # The resident head stages its own increments as npy beside the store.
        peak += 3 * sum(_field_bytes(grids, index) for index in range(horizon))
    # The composition stages h, qt and flux at the finest grid before the write.
    peak += 3 * _field_bytes(grids, n_classes - 1)
    if horizon >= n_classes:
        # Degenerate plan: nothing streams, but the plumbing still writes the
        # head's state into a store and composes out of it, so the scratch is
        # not zero. Accounted rather than special-cased, so the number is never
        # a confident lie.
        peak = max(peak, 6 * _field_bytes(grids, n_classes - 1))
    return TilePlan(horizon, tiles, halo, peak)


def check_scratch_space(directory, needed_bytes, already_owned=0):
    """Refuse up front, with the number, rather than dying mid-cascade."""
    directory = Path(directory)
    probe = directory if directory.exists() else directory.parent
    free = shutil.disk_usage(probe).free
    # A resuming run already OWNS the bytes its store occupies, and demanding
    # the full peak on top of them would refuse a resume that fits perfectly
    # well (2026-08-12, codex review).
    needed_bytes = max(0, needed_bytes - int(already_owned))
    if needed_bytes > free:
        raise OSError(
            f"streamed run needs {needed_bytes / 1024**3:.2f} GiB of scratch in "
            f"{directory} but only {free / 1024**3:.2f} GiB is free "
            f"(short by {(needed_bytes - free) / 1024**3:.2f} GiB). Point "
            f"scratch_dir= at a larger filesystem -- ideally NVMe, since "
            f"scratch I/O is this mode's cost driver.")
    return free


# ---------------------------------------------------------------------------
# The streamed driver
# ---------------------------------------------------------------------------

def _multiplier_noise(shape, world_origin, world_shape, class_key_i,
                      sparsity_factors, per_class_scale, shift,
                      n_zero, zero_bottom, zero_top):
    """The unit-mean multiplier noise exp(gamma)-1 over one tile plus its halo.

    Character-for-character _advance_flux's own construction, in one place so
    pass 1 and pass 2 cannot drift apart -- they must produce the identical
    array, which is precisely what world-keyed noise buys and what makes
    recompute-instead-of-persist exact.
    """
    from .simulate import FLUX_ALPHA, NoiseRegion, _keyed_sparse_levy

    generator = _keyed_sparse_levy(
        shape, sparsity_factors, FLUX_ALPHA,
        NoiseRegion(class_key_i, world_origin, world_shape))
    if zero_bottom and n_zero > 0:
        generator[:, :, :n_zero] = 0
    if zero_top and n_zero > 0:
        generator[:, :, -n_zero:] = 0
    generator *= per_class_scale
    off_center = generator == 0.0
    generator -= shift
    np.expm1(generator, out=generator)
    noise = generator
    noise[off_center] = np.float32(0.0)
    return noise


def streamed_cascade_loop(
    h_profile, qt_profile, z_profile,
    grids,
    C_h_k, C_qt_k,
    b_h_k, b_qt_k, aspect_k,
    h_min, h_max, qt_min, qt_max,
    min_distance_to_ground,
    sparsity_factors,
    class_seeds,
    plan,
    scratch_dir,
    n_scale_classes_per_dyad=1,
    flux_noise_scale=None,
    turbulon_shape='mexican_hat',
    zero_bottom=True,
    zero_top=True,
    device='cpu',
    comp_k=None,
    increment_store=None,
    completed=None,
    _crash_hook=None,
):
    """cascade_loop for a ROOT run, paged through RAM in horizontal tiles.

    Same arguments as cascade_loop where they overlap, plus a TilePlan and a
    scratch directory. Classes above ``plan.horizon`` run in RAM through
    cascade_loop itself, unchanged; the rest stream.

    Restricted to a root, deliberately and checked: no inherited state, no
    inner_windows (a streamed run's domain IS the world, so every realized mean
    is over the whole array), no between-class crop (a root's padded extent is
    constant, so the regrid is a pure zoom). Those are the assumptions that make
    the tiling argument simple enough to trust; a streamed nest would need the
    halo bookkeeping to compose with the tile halo, which is not this component.

    ``comp_k`` is the per-class, per-level interpolation compensation, as
    cascade_loop takes it. Where it differs from one the run accumulates the
    three DEFICIT fields in the store, lazily and down the same regrid chain as
    the state; they feed only the output composition, which never touches the
    cascade state.

    ``increment_store`` is a second TileStore. Given one, each class's ACTUALLY
    ADDED increment is persisted into it (per scalar from the plane pass, for
    the flux from the rescale visit) -- what save_for_refinement stores, and
    what a nest replays to compose its own output.

    Returns (h_perturbation, qt_perturbation, flux, deficit_h, deficit_qt,
    deficit_flux, final_grid_info, ledger, store). The fields are memory-mapped
    views on the scratch store rather than resident arrays: assembling them in
    RAM would undo the whole point at production scale, so the composition
    writes them out in hyperslabs. Deficits are None where every class was well
    resolved. The caller owns the store and must destroy() it.
    """
    from .simulate import (
        CONVOLVE, FLUX_ALPHA, FLUX_SCALE, LEVY_LOG_MEAN, SUPPORT_FACTOR,
        _advective_weight, _bound_taper, _bounded_amplitude_add,
        _turbulon_envelope, cascade_loop,
    )
    from .noise import class_key

    s_x, s_y, s_z = sparsity_factors
    n_classes = len(grids['k'])
    n_zero = int(round(2 * s_z * min_distance_to_ground))
    horizon = plan.horizon
    halo_x, halo_y = plan.halo

    for name, values in (('padded_extent_x', grids['padded_extent_x']),
                         ('padded_extent_y', grids['padded_extent_y']),
                         ('padded_height', grids['padded_height'])):
        if not np.allclose(values, values[0]):
            raise ValueError(
                f"streamed_cascade_loop needs a root's grids (constant "
                f"{name} across classes, so the regrid is a pure zoom); got "
                f"a varying one, which means a nest's shrinking pad")

    kernel = _turbulon_envelope(1, 1 / (2 * s_x), 1 / (2 * s_y), 1 / (2 * s_z),
                                support_factor=SUPPORT_FACTOR,
                                shape=turbulon_shape)
    class_keys = [class_key(child) for child in class_seeds]
    if comp_k is None:
        from .simulate import _interpolation_compensation
        k_finest = float(grids['k'][n_classes - 1])
        comp_k = [np.full(int(grids['nz'][i]), np.float32(
                      _interpolation_compensation(2.0 * grids['k'][i] / k_finest)))
                  for i in range(n_classes)]
    ledger = RunLedger(n_classes)
    store = TileStore(scratch_dir)
    # RESUME. ``completed`` is the set of (class, phase) pairs a previous attempt
    # finished, from the store's progress log. Every phase is written to be
    # re-runnable from its start -- pass 1 is pure measurement, pass 2 reads the
    # entering buffers which survive until the plane pass swaps them, and the
    # plane pass double-buffers -- so resume is simply "skip what is marked".
    # The one thing that cannot be recomputed is a completed class's LEDGER
    # entry, so each class's entry is persisted as it completes and read back.
    completed = set() if completed is None else set(completed)
    # Finish any settle a crash interrupted between a marker commit and its
    # renames. Idempotent -- a rename whose source is gone already happened, a
    # drop of a missing file is a no-op -- so replaying it for every marked
    # phase costs nothing and makes the restart rule a single clause.
    store.settle_all(completed)

    def phase_done(index, phase):
        return (index, phase) in completed

    def hook(index, phase):
        if _crash_hook is not None:
            _crash_hook(index, phase)

    def persist(index):
        save_class_ledger(store.directory / f"ledger_c{index:02d}.npz",
                          ledger.classes[index])
    scale_c = FLUX_SCALE if flux_noise_scale is None else flux_noise_scale
    # The resident head stages its per-class increments the way cascade_loop
    # already does (npy per class), beside the streamed store's own.
    head_increment_dir = None
    if increment_store is not None and horizon > 0:
        head_increment_dir = Path(scratch_dir) / "head_increments"
        head_increment_dir.mkdir(parents=True, exist_ok=True)

    # ---- above the horizon: today's resident loop, unchanged ----------------
    if horizon > 0:
        resident = {key: (grids[key][:horizon]
                          if key not in ('z_arrays', 'dz_arrays')
                          else grids[key][:horizon])
                    for key in grids}
        # comp_k[:horizon], NOT the default cascade_loop would compute: the
        # default is derived from the finest class of the grids it is handed, so
        # a sliced head would compensate its classes as though the cascade
        # stopped there. Caught 2026-08-12 by the deficit comparison -- the
        # states matched and the deficits were 11% out.
        (h_pert, qt_pert, flux_field, deficit_h, deficit_qt, deficit_flux, _,
         head_ledger) = cascade_loop(
            h_profile, qt_profile, z_profile, resident,
            C_h_k[:horizon], C_qt_k[:horizon],
            b_h_k[:horizon], b_qt_k[:horizon], aspect_k[:horizon],
            h_min, h_max, qt_min, qt_max,
            min_distance_to_ground, sparsity_factors, class_seeds[:horizon],
            n_scale_classes_per_dyad=n_scale_classes_per_dyad,
            flux_noise_scale=flux_noise_scale,
            turbulon_shape=turbulon_shape,
            zero_bottom=zero_bottom, zero_top=zero_top, device=device,
            comp_k=comp_k[:horizon],
            increment_dir=head_increment_dir,
        )
        for index in range(horizon):
            ledger.classes[index] = head_ledger.classes[index]
        # The head's deficits come across too, on its own final grid, and are
        # regridded onward with the state like any other carried field.
        for name, field in (('h_perturbation', h_pert),
                            ('qt_perturbation', qt_pert),
                            ('flux', flux_field),
                            ('h_deficit', deficit_h),
                            ('qt_deficit', deficit_qt),
                            ('flux_deficit', deficit_flux)):
            if field is None:
                continue
            store.create(horizon - 1, name, field.shape)[...] = field
            store.close(horizon - 1, name)
        del h_pert, qt_pert, flux_field, deficit_h, deficit_qt, deficit_flux
    else:
        # Nothing ran in RAM, so class 0's store IS the cascade's initial state.
        shape = (int(grids['nx'][0]), int(grids['ny'][0]), int(grids['nz'][0]))
        store.create(-1, 'h_perturbation', shape, fill=0.0)
        store.create(-1, 'qt_perturbation', shape, fill=0.0)
        store.create(-1, 'flux', shape, fill=1.0)

    for index in range(max(horizon, 0), n_classes):
        nx_k = int(grids['nx'][index])
        ny_k = int(grids['ny'][index])
        nz_k = int(grids['nz'][index])
        dx_k = float(grids['dx'][index])
        dy_k = float(grids['dy'][index])
        z_k = grids['z_arrays'][index]
        world_shape = (nx_k, ny_k, nz_k)
        tiles_x, tiles_y = plan.tiles[index]
        windows = tile_windows(nx_k, ny_k, tiles_x, tiles_y)
        class_ledger = ledger.classes[index]
        previous = index - 1

        h_mean_1d = np.interp(z_k, z_profile, h_profile).astype(np.float32)
        qt_mean_1d = np.interp(z_k, z_profile, qt_profile).astype(np.float32)
        scalars = (
            ('h', 'h_perturbation', 'h_increment', h_mean_1d, C_h_k[index],
             b_h_k[index], h_min, h_max),
            ('qt', 'qt_perturbation', 'qt_increment', qt_mean_1d,
             C_qt_k[index], b_qt_k[index], qt_min, qt_max),
        )

        if phase_done(index, 'class'):
            ledger.classes[index] = load_class_ledger(
                store.directory / f"ledger_c{index:02d}.npz")
            print(f'Streamed {index + 1:3d}/{n_classes:3d}: resumed, already '
                  f'complete                       ')
            continue

        deficit_i = comp_k[index] - np.float32(1.0)
        class_has_deficit = bool(np.any(deficit_i))
        carried = list(STATE_FIELDS) + [
            name for name in DEFICIT_FIELDS.values()
            if store.exists(previous, name)]

        # ---- pass 0: regrid the previous class's store onto this grid -------
        print(f'Streamed {index + 1:3d}/{n_classes:3d}: '
              f'grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}) '
              f'{tiles_x}x{tiles_y} tiles  regrid...', end='\r')
        for name in (() if phase_done(index, 'regrid') else carried):
            source_shape = store.open(previous, name).shape
            store.create(index, name, world_shape)

            def read_slab(x_lo, x_hi, y_lo, y_hi, _name=name):
                return store.read_window(previous, _name, x_lo, x_hi,
                                         y_lo, y_hi)

            if source_shape == world_shape:
                # Same grid (the first streamed class after a resident head):
                # nothing to resample, so copy plane by plane.
                source_planes = store.plane_array(previous, name)
                target_planes = store.plane_array(index, name)
                for level in range(nz_k):
                    target_planes[level] = source_planes[level]
            else:
                for x_lo, x_hi, y_lo, y_hi in windows:
                    store.write_window(
                        index, name, x_lo, x_hi, y_lo, y_hi,
                        regrid_window(read_slab, source_shape, world_shape,
                                      (x_lo, x_hi, y_lo, y_hi)))
            store.plane_array(index, name).flush()
            store.close(index, name)
        if not phase_done(index, 'regrid'):
            # A class whose compensation differs from one starts the deficits if
            # no earlier class already did. Once they exist they are carried on,
            # even through classes with f == 1, because the accumulated sum is
            # what the composition adds.
            if class_has_deficit:
                for name in DEFICIT_FIELDS.values():
                    if not store.exists(index, name):
                        store.create(index, name, world_shape, fill=0.0)
            # Marker FIRST, then the previous class's fields go. Dropping them
            # before the marker left a window in which a crash stranded the
            # resume with neither the source (deleted) nor a marker saying the
            # regrid had finished (codex review, 2026-08-12).
            store.mark_atomic(index, 'regrid')
            for name in carried:
                store.drop(previous, name)
        hook(index, 'regrid_done')

        per_class_scale = scale_c / n_scale_classes_per_dyad ** (1.0 / FLUX_ALPHA)
        shift = np.float32(LEVY_LOG_MEAN * per_class_scale ** FLUX_ALPHA)
        per_class_scale = np.float32(per_class_scale)

        def read_tile(name, window):
            """Tile plus halo, wrapped at the periodic domain edges."""
            x_lo, x_hi, y_lo, y_hi = window
            return store.read_window(index, name, x_lo - halo_x, x_hi + halo_x,
                                     y_lo - halo_y, y_hi + halo_y)

        def inner_of(window):
            x_lo, x_hi, y_lo, y_hi = window
            return (slice(halo_x, halo_x + (x_hi - x_lo)),
                    slice(halo_y, halo_y + (y_hi - y_lo)))

        def tile_noise(window):
            x_lo, x_hi, y_lo, y_hi = window
            shape = (x_hi - x_lo + 2 * halo_x, y_hi - y_lo + 2 * halo_y, nz_k)
            return _multiplier_noise(
                shape, (x_lo - halo_x, y_lo - halo_y, 0), world_shape,
                class_keys[index], sparsity_factors, per_class_scale, shift,
                n_zero, zero_bottom, zero_top)

        # Passes 1 and 2 are skipped wholesale on resume, and their results
        # come from the persisted class ledger rather than being re-measured:
        # a partially completed plane pass may already have swapped the
        # ENTERING flux for the advanced one, so re-measuring would measure
        # the wrong field. This is what the per-class ledger file is for.
        if phase_done(index, 'apply'):
            ledger.classes[index] = load_class_ledger(
                store.directory / f"ledger_c{index:02d}.npz")
            class_ledger = ledger.classes[index]
            level_means = {scalar: class_ledger.pattern[scalar].level_mean
                           for scalar in ('h', 'qt')}
            mean_abs = class_ledger.flux.noise_abs.mean
            flux_rescale = class_ledger.flux.rescale
        else:
            # ---- pass 1: MEASURE -----------------------------------------------
            print(f'Streamed {index + 1:3d}/{n_classes:3d}: '
                  f'grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}) '
                  f'{tiles_x}x{tiles_y} tiles  measure...', end='\r')
            entering = MeanReduction(np.float64(0.0), 0)
            noise_abs = MeanReduction(np.float64(0.0), 0)
            raw_pattern = {}
            for window in windows:                  # world raster order: see
                inner = inner_of(window)             # tile_windows
                flux_tile = read_tile('flux', window)
                noise = tile_noise(window)

                flux_inner = flux_tile[inner]
                entering = entering + MeanReduction(
                    flux_inner.sum(dtype=np.float64), flux_inner.size)
                noise_inner = noise[inner]
                noise_abs = noise_abs + MeanReduction(
                    np.abs(noise_inner).sum(dtype=np.float64),
                    np.count_nonzero(noise_inner))

                # The RAW product, i.e. with the entering flux in place of S_k: S_k
                # is that product divided by the mean-abs noise, which is not known
                # until this sweep finishes. A positive scalar factors straight out
                # of the level sums, so the normalized totals are recovered below
                # by one division -- at the cost of one float32 multiply's worth of
                # rounding against the in-RAM order, which is the floor this whole
                # path is measured at.
                raw = noise * flux_tile
                for (name, state_name, _, mean_1d, _, b_i, phi_min,
                     phi_max) in scalars:
                    state = read_tile(state_name, window)
                    running_sum = state + mean_1d[np.newaxis, np.newaxis, :]
                    W = _advective_weight(running_sum, dx_k, dy_k, z_k,
                                          aspect_k[index])
                    W *= raw
                    W *= _bound_taper(running_sum, b_i, phi_min, phi_max)
                    W_inner = W[inner]
                    partial = PatternLedger(
                        np.abs(W_inner).sum(axis=(0, 1), dtype=np.float64),
                        np.count_nonzero(W_inner, axis=(0, 1)))
                    raw_pattern[name] = (partial if name not in raw_pattern
                                         else raw_pattern[name] + partial)
                    del state, running_sum, W, W_inner
                del flux_tile, noise, raw

            # ---- reduce ---------------------------------------------------------
            mean_abs = noise_abs.mean
            for name in raw_pattern:
                divisor = mean_abs if mean_abs > 0 else 1.0
                class_ledger.pattern[name] = PatternLedger(
                    raw_pattern[name].level_total / divisor,
                    raw_pattern[name].level_count)
            level_means = {name: class_ledger.pattern[name].level_mean
                           for name in raw_pattern}
            store.mark_atomic(index, 'measure')

            # ---- pass 2: APPLY --------------------------------------------------
            print(f'Streamed {index + 1:3d}/{n_classes:3d}: '
                  f'grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}) '
                  f'{tiles_x}x{tiles_y} tiles  apply...', end='\r')
            for name in CANDIDATE_FIELDS:
                store.create(index, name, world_shape)
            realized = MeanReduction(np.float64(0.0), 0)
            n_clipped = 0
            # DOUBLE BUFFER, and it is load-bearing. Pass 2 reads the ENTERING flux
            # over a tile plus halo and writes the UPDATED flux over the tile. Those
            # regions overlap between neighbouring tiles, so writing back into the
            # field being read would hand every tile after the first a halo of
            # already-advanced values -- caught 2026-08-12 as a 13% error in the
            # realized flux mean, which is exactly the kind of thing the
            # streamed-equals-resident test exists to find.
            flux_next = store.create(index, 'flux_next', world_shape)
            for window in windows:
                x_lo, x_hi, y_lo, y_hi = window
                inner = inner_of(window)
                flux_tile = read_tile('flux', window)
                noise = tile_noise(window)

                # Same order as the in-RAM class body: S_k comes from the ENTERING
                # flux, then the flux advances, then the scalars use S_k.
                noise *= flux_tile
                if mean_abs > 0:
                    S_k = noise * np.float32(1.0 / mean_abs)
                else:
                    S_k = noise.copy()

                # Per-tile FFT. It is periodic over the PADDED TILE rather than the
                # domain, so the wrap contaminates within a kernel radius of the
                # padded tile's edge -- which is inside the halo, and the halo is
                # never written back. Exactly the nest halo's argument.
                increment = CONVOLVE(noise, kernel, device=device)
                del noise
                flux_tile += increment
                del increment
                flux_inner = flux_tile[inner]
                n_clipped += int(np.count_nonzero(flux_inner < 0))
                np.maximum(flux_tile, np.float32(0.0), out=flux_tile)
                flux_inner = flux_tile[inner]
                realized = realized + MeanReduction(
                    flux_inner.sum(dtype=np.float64), flux_inner.size)
                store.write_window(index, 'flux_next', x_lo, x_hi,
                                   y_lo, y_hi, flux_inner)
                del flux_tile

                for (name, state_name, increment_name, mean_1d, C_k_i, b_i,
                     phi_min, phi_max) in scalars:
                    state = read_tile(state_name, window)
                    running_sum = state + mean_1d[np.newaxis, np.newaxis, :]
                    W = _advective_weight(running_sum, dx_k, dy_k, z_k,
                                          aspect_k[index])
                    W *= S_k
                    W *= _bound_taper(running_sum, b_i, phi_min, phi_max)
                    level_mean = level_means[name]
                    W /= np.where(level_mean > 0, level_mean,
                                  np.float32(1.0))[None, None, :]
                    W *= C_k_i
                    candidate = CONVOLVE(W, kernel, device=device)
                    del W, state, running_sum
                    store.write_window(index, increment_name, x_lo, x_hi,
                                       y_lo, y_hi, candidate[inner])
                    del candidate
                del S_k
                hook(index, 'apply_tile')
            store.plane_array(index, 'flux_next').flush()
            # The swap is DEFERRED to pass 3: forming the flux's actually-added
            # increment needs the entering flux ('flux') and the advanced flux
            # ('flux_next') live at the same time, after the rescale.
            for name in CANDIDATE_FIELDS:
                store.plane_array(index, name).flush()

            class_ledger.flux = FluxAdvanceLedger(entering, noise_abs, realized,
                                                  n_clipped)
            flux_rescale = class_ledger.flux.rescale
            if flux_rescale is None:
                raise RuntimeError(
                    f"class {index}: the whole domain clipped to zero, which the "
                    f"in-RAM path handles by flattening the flux to its entering "
                    f"mean. Not reachable at any sane amplitude, and not "
                    f"implemented streamed rather than implemented untested.")
            persist(index)
            store.mark_atomic(index, 'apply')
        hook(index, 'apply_done')

        # ---- pass 3: PLANE PASS (the bounded add) --------------------------
        print(f'Streamed {index + 1:3d}/{n_classes:3d}: '
              f'grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}) '
              f'{tiles_x}x{tiles_y} tiles  planes... ', end='\r')
        # The flux finishes INSIDE its class (2026-08-12). The rescale used to
        # be folded into the next class's read, which left the stored flux in a
        # half-advanced state and made the last class a special case; worse, the
        # flux's actually-added increment is defined across the WHOLE advance
        # including the rescale, so it could not be formed at all. Applying the
        # rescale here -- one multiply per plane, cheap beside the plane pass's
        # own traffic -- makes "the store always holds a fully advanced flux" an
        # invariant, and hands the deficit and save_for_refinement the increment
        # they need.
        if phase_done(index, 'plane_flux'):
            entering_planes = advanced_planes = None      # settled already
        else:
            entering_planes = store.plane_array(index, 'flux')
            advanced_planes = store.plane_array(index, 'flux_next')
            has_flux_deficit = store.exists(index, 'flux_deficit')
            flux_deficit_planes = (store.plane_array(index, 'flux_deficit')
                                   if has_flux_deficit else None)
            flux_deficit_next = None
            if has_flux_deficit:
                store.create(index, 'flux_deficit_next', world_shape)
                flux_deficit_next = store.plane_array(index, 'flux_deficit_next')
            record_flux = increment_store is not None
            if record_flux:
                increment_store.create(index, 'flux', world_shape)
                flux_increment_planes = increment_store.plane_array(index, 'flux')
            # PURE WRITES to a third buffer, never an in-place multiply of
            # flux_next. Rescaling in place made this loop non-idempotent: a
            # crash halfway through and then a resume would rescale the
            # completed prefix a second time (codex review, 2026-08-12).
            store.create(index, 'flux_rescaled', world_shape)
            rescaled_planes = store.plane_array(index, 'flux_rescaled')
            for level in range(nz_k):
                advanced = advanced_planes[level] * np.float32(flux_rescale)
                if flux_deficit_planes is not None or record_flux:
                    applied = advanced - entering_planes[level]
                    if flux_deficit_next is not None:
                        flux_deficit_next[level] = (
                            flux_deficit_planes[level]
                            + np.float32(deficit_i[level]) * applied)
                    if record_flux:
                        flux_increment_planes[level] = applied
                rescaled_planes[level] = advanced
                if level == nz_k // 2:
                    hook(index, 'flux_rescale_mid')
            if record_flux:
                increment_store.plane_array(index, 'flux').flush()
                increment_store.close(index, 'flux')
            rescaled_planes.flush()
            if flux_deficit_next is not None:
                flux_deficit_next.flush()
            # ATOMIC marker, then settle. Every rename and drop below happens
            # AFTER the marker commits and is replayed idempotently on resume,
            # so no crash window can leave the fields half-swapped.
            renames = [('flux_rescaled', 'flux')]
            drops = ['flux_next']
            if flux_deficit_next is not None:
                renames.append(('flux_deficit_next', 'flux_deficit'))
            store.mark_atomic(index, 'plane_flux', pending_renames=renames,
                              pending_drops=drops)

        for (name, state_name, increment_name, mean_1d, _, _, phi_min,
             phi_max) in scalars:
            if phase_done(index, f'plane_{name}'):
                continue
            # DOUBLE BUFFERED, for resume. The plane pass used to mutate the
            # state in place level by level, so a crash mid-pass left a
            # half-projected field that a re-run would project twice. Writing
            # to a companion buffer and swapping at pass end makes "resume =
            # re-run the last incomplete pass" unconditionally true, at one
            # extra state field of I/O per class per scalar. The DEFICIT needs
            # the same treatment for the same reason -- it is accumulated, so a
            # re-run would double-add.
            state_planes = store.plane_array(index, state_name)
            candidate_planes = store.plane_array(index, increment_name)
            store.create(index, state_name + '_next', world_shape)
            state_next_planes = store.plane_array(index, state_name + '_next')
            deficit_name = DEFICIT_FIELDS[name]
            has_deficit = store.exists(index, deficit_name)
            deficit_planes = (store.plane_array(index, deficit_name)
                              if has_deficit else None)
            deficit_next_planes = None
            if has_deficit:
                store.create(index, deficit_name + '_next', world_shape)
                deficit_next_planes = store.plane_array(
                    index, deficit_name + '_next')
            increment_planes = None
            if increment_store is not None:
                increment_store.create(index, name, world_shape)
                increment_planes = increment_store.plane_array(index, name)
            solve = BoundedAddLedger(nz_k)
            for level in range(nz_k):
                # The domain of a streamed run is the world, so an assembled
                # z-plane IS the array the in-RAM bounded add sees at this
                # level -- same shape, same window (none), same solve. The
                # levels are independent, so running them one at a time is the
                # identical arithmetic rather than an approximation of it.
                plane_state = np.array(
                    state_planes[level], dtype=np.float32)[:, :, None]
                plane_candidate = np.array(
                    candidate_planes[level], dtype=np.float32)[:, :, None]
                level_solve = _bounded_amplitude_add(
                    plane_state, mean_1d[level:level + 1], plane_candidate,
                    phi_min, phi_max, window=None, device=device,
                    record_applied=True)
                state_next_planes[level] = plane_state[:, :, 0]
                # record_applied left the ACTUALLY-ADDED delta in the candidate
                # -- after minus before, float32 rounding included, which is
                # what the deficit compensates and what a nest replays.
                applied = plane_candidate[:, :, 0]
                if deficit_next_planes is not None:
                    deficit_next_planes[level] = (
                        deficit_planes[level]
                        + np.float32(deficit_i[level]) * applied)
                if increment_planes is not None:
                    increment_planes[level] = applied
                if level == nz_k // 2:
                    hook(index, f'plane_{name}_mid')
                for field in ('a0', 'n_loop', 'n_scale', 'final_demean', 'mu'):
                    getattr(solve, field)[level] = getattr(level_solve, field)[0]
                solve.demean[level] = level_solve.demean[0]
                solve.scale[level] = level_solve.scale[0]
            class_ledger.bounded_add[name] = solve
            state_next_planes.flush()
            if deficit_next_planes is not None:
                deficit_next_planes.flush()
            if increment_planes is not None:
                increment_store.plane_array(index, name).flush()
                increment_store.close(index, name)
            # Persist BEFORE the marker: the marker's presence is what makes a
            # resume load this ledger file, so committing the marker first left a
            # window where the resume looked for an npz that did not exist yet
            # (codex review, 2026-08-12).
            persist(index)
            renames = [(state_name + '_next', state_name)]
            if deficit_next_planes is not None:
                renames.append((deficit_name + '_next', deficit_name))
            store.mark_atomic(index, f'plane_{name}',
                              pending_renames=renames,
                              pending_drops=[increment_name])
        persist(index)
        store.mark_atomic(index, 'class')
        hook(index, 'class_done')
        print(f'Streamed {index + 1:3d}/{n_classes:3d}: '
              f'grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}) '
              f'{tiles_x}x{tiles_y} tiles  done          ')

    final = n_classes - 1
    final_grid_info = {
        'nx': int(grids['nx'][final]), 'ny': int(grids['ny'][final]),
        'nz': int(grids['nz'][final]), 'dx': float(grids['dx'][final]),
        'dy': float(grids['dy'][final]), 'dz': grids['dz_arrays'][final],
        'z': grids['z_arrays'][final],
    }
    def deficit_or_none(name):
        return (store.open(final, name) if store.exists(final, name) else None)

    return (store.open(final, 'h_perturbation'),
            store.open(final, 'qt_perturbation'),
            store.open(final, 'flux'),
            deficit_or_none('h_deficit'), deficit_or_none('qt_deficit'),
            deficit_or_none('flux_deficit'),
            final_grid_info, ledger, store)


# ---------------------------------------------------------------------------
# Streamed composition (component 4)
# ---------------------------------------------------------------------------
#
# The composition is per-LEVEL work (the mean-preserving projection) followed by
# per-TILE work (the NetCDF write). Those want opposite layouts, and both get
# the one they want: the projection runs on the store's z-major planes, where a
# level is contiguous, and the write reads x-y tiles back out through the store's
# raw-buffer window reader, matching the output variable's (64, 64, nz) chunking.
# Composing straight into the NetCDF variable plane by plane would touch every
# chunk of the file per level -- the same trap as the memmap layout, one level up.

COMPOSED_FIELDS = {'h': 'h_composed', 'qt': 'qt_composed',
                   'flux': 'flux_composed'}


def compose_scalar(store, index, state_name, deficit_name, composed_name,
                   mean_1d, phi_min, phi_max, device='cpu', ledger=None):
    """state + deficit + mean, projected onto the bounds, into a store field.

    The streamed form of _compose_output, level by level. The projection is the
    mean-preserving one (_project_onto_bounds' s = 1 family) and it is per level,
    so a plane at a time IS the whole operation rather than an approximation of
    it -- the levels are independent and every reduction is within a level.

    Returns the ProjectionLedger. The composition is deliberately the LAST thing
    that happens and touches nothing a descendant nest reads: a nest continues
    from the STATE, which is why it is the state the store keeps.
    """
    from .simulate import _project_onto_bounds

    state_planes = store.plane_array(index, state_name)
    nz = state_planes.shape[0]
    has_deficit = store.exists(index, deficit_name)
    deficit_planes = store.plane_array(index, deficit_name) if has_deficit else None
    store.create(index, composed_name,
                 (state_planes.shape[1], state_planes.shape[2], nz))
    composed_planes = store.plane_array(index, composed_name)

    projection = ledger if ledger is not None else ProjectionLedger(
        np.full(nz, np.nan, dtype=np.float64))
    for level in range(nz):
        plane = np.array(state_planes[level], dtype=np.float32)
        if has_deficit:
            plane += deficit_planes[level]
            # Restoring amplitude to under-resolved classes can push the sum
            # past the bounds the per-class bounded add held the STATE inside,
            # so the composition is projected onto them -- exactly as
            # _compose_output does, and only when there is a deficit to add.
            level_ledger = ProjectionLedger(projection.mu[level:level + 1].copy())
            solved = _project_onto_bounds(
                plane[:, :, None], mean_1d[level:level + 1], phi_min, phi_max,
                window=None, device=device,
                ledger=level_ledger if ledger is not None else None)
            projection.mu[level] = solved.mu[0]
        plane += np.float32(mean_1d[level])
        # The bounds now hold up to float32 rounding of the (perturbation +
        # mean) reassembly, which can undershoot by ~1 ulp of the field value.
        # Clamp that residue only -- at most one ulp, not a physics clip.
        np.clip(plane, np.float32(phi_min), np.float32(phi_max), out=plane)
        composed_planes[level] = plane
    composed_planes.flush()
    return projection


def compose_flux(store, index, state_name, deficit_name, composed_name,
                 ledger=None):
    """(flux + deficit) clipped at zero, restored to the state's volume mean.

    The flux analogue of compose_scalar, and its bounds discipline is the flux's
    own: positivity, and the volume mean the uncompensated state carries. Two
    plane passes -- the realized mean cannot be known until the clip has run
    everywhere -- with the entering mean accumulated on the first one's read, so
    it costs no extra traversal.
    """
    state_planes = store.plane_array(index, state_name)
    nz = state_planes.shape[0]
    has_deficit = store.exists(index, deficit_name)
    if not has_deficit:
        # No class was under-resolved: the state IS the output.
        return None, state_name
    deficit_planes = store.plane_array(index, deficit_name)
    store.create(index, composed_name,
                 (state_planes.shape[1], state_planes.shape[2], nz))
    composed_planes = store.plane_array(index, composed_name)

    if ledger is None:
        entering = MeanReduction(np.float64(0.0), 0)
        realized = MeanReduction(np.float64(0.0), 0)
    else:
        entering, realized = ledger.entering, ledger.realized
    for level in range(nz):
        state_plane = np.array(state_planes[level], dtype=np.float32)
        if ledger is None:
            entering = entering + MeanReduction(
                state_plane.sum(dtype=np.float64), state_plane.size)
        plane = state_plane + deficit_planes[level]
        np.maximum(plane, np.float32(0.0), out=plane)
        if ledger is None:
            realized = realized + MeanReduction(
                plane.sum(dtype=np.float64), plane.size)
        composed_planes[level] = plane

    output_ledger = FluxOutputLedger(entering, realized)
    rescale = output_ledger.rescale
    for level in range(nz):
        if rescale is not None:
            composed_planes[level] *= np.float32(rescale)
        else:
            # Everything clipped to zero: restore the entering mean flat, as
            # _compose_flux_output does.
            composed_planes[level] = np.float32(entering.mean)
    composed_planes.flush()
    return output_ledger, composed_name


def tile_writer_from_store(store, index, name_map, tile=64):
    """A write_netcdf `tile_writer`: fill a 3D variable from the store in tiles.

    x-y tiles with the full z extent, which is the shape of the output
    variable's own chunks -- writing z-planes instead would touch every chunk of
    the file per level. Reads come through the store's raw-buffer window reader,
    so the source side is contiguous runs too.
    """
    def write(name, variable, source):
        field = name_map.get(name)
        if field is None:
            variable[:] = source
            return
        class_index, store_name = field
        nx, ny = variable.shape[0], variable.shape[1]
        for x_lo in range(0, nx, tile):
            x_hi = min(x_lo + tile, nx)
            for y_lo in range(0, ny, tile):
                y_hi = min(y_lo + tile, ny)
                variable[x_lo:x_hi, y_lo:y_hi, :] = store.read_window(
                    class_index, store_name, x_lo, x_hi, y_lo, y_hi)
    return write


class IncrementReader:
    """A write_class_increments `reader`, backed by a streamed run's store."""

    __slots__ = ('_store', '_index', 'shape')

    def __init__(self, store, index, shape):
        self._store = store
        self._index = index
        self.shape = tuple(int(n) for n in shape)

    def __call__(self, name, x_lo, x_hi):
        return self._store.read_window(self._index, name, x_lo, x_hi,
                                       0, self.shape[1])


def compose_and_write(store, increment_store, result, grids, plan,
                      profiles, bounds, simulation_params, output_path,
                      class_grids, coordinates, k_arrays, C_stored,
                      save_for_refinement, compress, device):
    """Compose a streamed run's states and write the output file.

    The composition runs on the store's z-major planes (per level, which is what
    the projection is) and the file is written in x-y tiles (which is what the
    NetCDF chunks are). Both layouts get the access they want; neither field is
    ever resident.

    Returns the ProjectionLedger / FluxOutputLedger entries so the caller can
    put them in the run's ledger before it is written.
    """
    from .output import write_class_increments, write_netcdf

    h_profile, qt_profile, z_profile = profiles
    h_min, h_max, qt_min, qt_max = bounds
    final = len(grids['k']) - 1
    z_final = np.asarray(result[6]['z'], dtype=np.float32)
    h_mean = np.interp(z_final, z_profile, h_profile).astype(np.float32)
    qt_mean = np.interp(z_final, z_profile, qt_profile).astype(np.float32)

    projections = {}
    projections['h'] = compose_scalar(
        store, final, 'h_perturbation', 'h_deficit', COMPOSED_FIELDS['h'],
        h_mean, h_min, h_max, device=device)
    projections['qt'] = compose_scalar(
        store, final, 'qt_perturbation', 'qt_deficit', COMPOSED_FIELDS['qt'],
        qt_mean, qt_min, qt_max, device=device)
    flux_output_ledger, flux_field = compose_flux(
        store, final, 'flux', 'flux_deficit', COMPOSED_FIELDS['flux'])

    # Into the ledger BEFORE it is written: write_netcdf serializes the ledger
    # group, so a projection merged after the write would be recorded nowhere.
    # (Caught 2026-08-12 by the ledger round-trip test, which found the streamed
    # file's projection group empty while every field agreed.)
    run_ledger = result[7]
    run_ledger.projection.update(projections)
    if flux_output_ledger is not None:
        run_ledger.flux_output = flux_output_ledger

    # Which store field backs each output variable. The perturbations and the
    # flux STATE are written as they are -- a nest continues from those, so they
    # must not carry the composition.
    name_map = {
        'h': (final, COMPOSED_FIELDS['h']),
        'qt': (final, COMPOSED_FIELDS['qt']),
        'flux': (final, flux_field),
        'h_perturbation': (final, 'h_perturbation'),
        'qt_perturbation': (final, 'qt_perturbation'),
        'flux_state': (final, 'flux'),
    }
    write_netcdf(
        output_path, result[0], result[1],
        coordinates[0], coordinates[1], z_final,
        h_profile, qt_profile, z_profile,
        k_arrays[0], k_arrays[1], C_stored[0], C_stored[1],
        simulation_params, compress=compress,
        flux_3d=result[2],
        h_pert_3d=result[0] if save_for_refinement else None,
        qt_pert_3d=result[1] if save_for_refinement else None,
        flux_state_3d=result[2] if save_for_refinement else None,
        run_ledger=result[7],
        tile_writer=tile_writer_from_store(store, final, name_map),
    )

    if increment_store is not None:
        # Classes above the horizon staged npy the way cascade_loop does; the
        # streamed ones are read straight out of the store.
        readers = []
        for index in range(len(class_grids)):
            if index < plan.horizon:
                readers.append(None)
            else:
                readers.append(IncrementReader(
                    increment_store, index,
                    (int(grids['nx'][index]), int(grids['ny'][index]),
                     int(grids['nz'][index]))))
        write_class_increments(
            output_path, Path(store.directory) / "head_increments",
            class_grids, compress=compress, readers=readers)
    return projections, flux_output_ledger


# ---------------------------------------------------------------------------
# Lifecycle: budgets, refusals, manifest, resume (component 5)
# ---------------------------------------------------------------------------

MEMORY_BUDGET_FRACTION = 0.8


def available_memory_budget(fraction=MEMORY_BUDGET_FRACTION):
    """A default memory budget from MemAvailable, in bytes.

    MemAvailable rather than MemFree: the kernel's own estimate of what a new
    allocation can have without swapping, which already discounts the
    reclaimable page cache a streamed run leans on. The fraction is headroom for
    everything the estimate cannot know about -- the FFT working set's own
    guards, the interpreter, whatever else the box is doing.
    """
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(int(line.split()[1]) * 1024 * fraction)
    except OSError:
        pass
    from .utils import available_memory_bytes
    available = available_memory_bytes()
    return None if available is None else int(available * fraction)


def is_tmpfs(path):
    """Is this path on a RAM-backed filesystem?

    /proc/mounts is the reliable route on Linux (statvfs exposes no f_type):
    walk the mount points, take the longest one that is a prefix of the resolved
    path, and look at its type. tmpfs, ramfs and devtmpfs are all RAM.
    """
    path = Path(path).resolve()
    try:
        entries = Path('/proc/mounts').read_text().splitlines()
    except OSError:
        return False
    best, best_type = -1, None
    for entry in entries:
        parts = entry.split()
        if len(parts) < 3:
            continue
        mount_point, filesystem = parts[1], parts[2]
        mount_point = mount_point.replace('\\040', ' ')
        try:
            resolved = Path(mount_point).resolve()
        except OSError:
            continue
        if resolved == path or resolved in path.parents:
            depth = len(resolved.parts)
            if depth > best:
                best, best_type = depth, filesystem
    return best_type in ('tmpfs', 'ramfs', 'devtmpfs')


def check_scratch_filesystem(directory):
    """Refuse a RAM-backed scratch directory.

    A tmpfs scratch IS memory, so streaming to it bounds nothing and the run
    fails later and more confusingly than it would have here. The system
    temporary directory is tmpfs on most Linux distributions, which is why the
    default scratch lives beside the output file -- and why this check exists
    rather than a comment hoping nobody points scratch_dir at /tmp. No override:
    a genuinely exotic setup can name a real filesystem instead.
    """
    probe = Path(directory)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if is_tmpfs(probe):
        raise OSError(
            f"scratch directory {directory} is on a RAM-backed filesystem "
            f"(tmpfs/ramfs). Streaming to RAM bounds nothing -- the whole point "
            f"is to keep the working state OFF memory -- so this is refused "
            f"rather than run. Pass scratch_dir= pointing at a real "
            f"filesystem, ideally NVMe: scratch I/O is this mode's cost driver."
        )


def check_output_space(output_path, grids, save_for_refinement):
    """Refuse up front if the OUTPUT file will not fit where it is going.

    A run that streams for hours and then dies inside write_netcdf on a full
    filesystem is precisely the failure this feature exists to prevent, and the
    output is not small: h, qt and flux at the finest grid, doubled by
    save_for_refinement, plus the class-increment pyramid.
    """
    field = _field_bytes(grids, len(grids['k']) - 1)
    needed = 3 * field
    if save_for_refinement:
        needed += 3 * field
        needed += 3 * sum(_field_bytes(grids, index)
                          for index in range(len(grids['k'])))
    destination = Path(output_path).parent
    free = shutil.disk_usage(destination if destination.exists()
                             else Path('.')).free
    if needed > free:
        raise OSError(
            f"the output file needs about {needed / 1024**3:.2f} GiB in "
            f"{destination} but only {free / 1024**3:.2f} GiB is free. "
            f"Refusing before the cascade rather than after it.")
    return needed


def build_manifest(grids, plan, seed, fingerprint_inputs):
    """The fingerprint a resume must match before it reuses a scratch store.

    ``fingerprint_inputs`` is every scalar, array and string that would change
    the answer; the caller assembles it so nothing can be forgotten here without
    being visibly absent there. Two digests come out:

    - ``config_sha256`` over everything including the seed;
    - ``config_seedless_sha256`` over everything except it.

    The seedless one exists for ``seed=None`` runs (2026-08-12, codex review).
    An unseeded run draws a fresh seed each time, so its retry would always look
    like a different configuration and the default API path could never resume.
    With the seedless digest matching, the manifest's recorded seed is ADOPTED
    and the run continues -- which is what "re-running the same command resumes"
    has to mean when the command names no seed.

    A scratch store whose manifest does not match is NEVER silently reused --
    half a cascade from a different configuration is worse than no cascade.

    What must be in here, learned the hard way: the module constants H_h and
    lambda are read at RUN time, and Thomas sweeps H_h in constants.py, so a
    resume across an edit to that file would otherwise splice two realizations
    together. Likewise ``device``: the CPU and CUDA paths differ at ULP level, so
    a mixed resume is bit-identical to neither pure run.
    """
    def digest_of(items):
        digest = hashlib.sha256()
        for value in items:
            if isinstance(value, np.ndarray):
                digest.update(np.ascontiguousarray(
                    value, dtype=np.float64).tobytes())
            else:
                digest.update(repr(value).encode())
            digest.update(b'\x1e')     # separator: keeps fields unambiguous
        return digest.hexdigest()

    shared = [
        np.asarray(grids['k'], dtype=np.float64),
        np.asarray(grids['nx'], dtype=np.float64),
        np.asarray(grids['ny'], dtype=np.float64),
        np.asarray(grids['nz'], dtype=np.float64),
        np.asarray(grids['dz'], dtype=np.float64),
    ]
    # Every class's z grid and cell heights: a change in the vertical gridding
    # changes every level's answer without touching nx/ny/nz.
    for index in range(len(grids['k'])):
        shared.append(np.asarray(grids['z_arrays'][index], dtype=np.float64))
        shared.append(np.asarray(grids['dz_arrays'][index], dtype=np.float64))
    shared.extend([plan.horizon, sorted(plan.tiles.items()), plan.halo])
    for key in sorted(fingerprint_inputs):
        shared.append(key)
        shared.append(fingerprint_inputs[key])

    return {
        'config_sha256': digest_of(shared + ['seed', seed]),
        'config_seedless_sha256': digest_of(shared),
        'seed': None if seed is None else int(seed),
        'n_classes': int(len(grids['k'])),
        'horizon': int(plan.horizon),
        'tiles': {str(k): list(v) for k, v in sorted(plan.tiles.items())},
        'halo': list(plan.halo),
        'shapes': [[int(grids['nx'][i]), int(grids['ny'][i]),
                    int(grids['nz'][i])] for i in range(len(grids['k']))],
    }


def prepare_scratch(directory, manifest, fresh=False, seed_was_none=False):
    """Validate, adopt or clear a scratch directory.

    Returns (store, completed_phases, adopted_seed). ``adopted_seed`` is not None
    only when an unseeded run is resuming and has taken the seed the earlier
    attempt drew -- see build_manifest.

    Three cases, and the distinction between the last two is the safety
    property (2026-08-12, codex review):

    - EMPTY or absent: start fresh, write the manifest.
    - A STORE (valid manifest) whose fingerprint differs: refuse, offering
      fresh=True, which discards only the store's own files.
    - NOT A STORE (non-empty, no valid manifest): refuse ALWAYS, and delete
      NOTHING. This is the dangerous case -- scratch_dir is a caller-supplied
      path, and the previous version treated "no manifest" like fresh=True and
      rmtree'd it, so pointing scratch_dir at any existing directory destroyed
      it. A corrupt manifest lands here too, which is the right side to fail on.
    """
    directory = Path(directory)
    store = TileStore(directory)
    non_empty = directory.exists() and any(directory.iterdir())
    existing = store.read_manifest()

    if non_empty and existing is None:
        raise OSError(
            f"scratch directory {directory} is not empty and is not a STEAM "
            f"scratch store (no readable manifest.json). Refusing to touch it: "
            f"this path may hold something else entirely, and a scratch "
            f"directory is deleted on success. Empty it yourself, or point "
            f"scratch_dir= somewhere else. (A corrupt manifest reports here "
            f"too -- if this WAS a store, delete it deliberately.)")

    if existing is not None and not fresh:
        if existing.get('config_sha256') == manifest['config_sha256']:
            return store, store.completed(), None
        if (seed_was_none
                and existing.get('config_seedless_sha256')
                == manifest['config_seedless_sha256']
                and existing.get('seed') is not None):
            # Same configuration in every respect but the seed, and the caller
            # named no seed -- so the earlier attempt's seed IS the one this run
            # means. Adopt it and resume.
            return store, store.completed(), int(existing['seed'])
        raise OSError(
            f"scratch directory {directory} holds a partial run of a DIFFERENT "
            f"configuration (manifest {existing.get('config_sha256', '?')[:12]} "
            f"against {manifest['config_sha256'][:12]}). Refusing to mix them. "
            f"Pass fresh=True to discard it and start over, or point "
            f"scratch_dir= somewhere else.")

    if non_empty:
        store.destroy()                 # a store, and fresh=True was asked for
        store = TileStore(directory)
    store.write_manifest(manifest)
    return store, set(), None


def scratch_hint(grids):
    """A human-readable scratch estimate, for the refusal message.

    About 11 field-equivalents at the finest grid -- the peak the planner
    accounts for -- which is the number worth quoting to someone deciding
    whether to turn streaming on.
    """
    total = 11 * _field_bytes(grids, len(grids['k']) - 1)
    return f"{total / 1024**3:.1f} GiB"
