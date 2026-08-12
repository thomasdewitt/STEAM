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
import shutil

import numpy as np
import torch

from .ledger import (
    BoundedAddLedger,
    FluxAdvanceLedger,
    MeanReduction,
    PatternLedger,
    RunLedger,
)

# Field names the store carries between classes. The deficits are absent on
# purpose: they exist only for the output composition (component 4), the cascade
# state never reads them, and carrying them would need the pre-advance flux kept
# live across the lazy rescale.
STATE_FIELDS = ('h_perturbation', 'qt_perturbation', 'flux')
# Written within a class by pass 2 and consumed by pass 3.
CANDIDATE_FIELDS = ('h_increment', 'qt_increment')


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
        """Move a field, for a double-buffer swap. Cheap: one directory entry."""
        self.close(class_index, name)
        self.close(class_index, new_name)
        target = self._path(class_index, new_name)
        if target.exists():
            target.unlink()
        self._path(class_index, name).rename(target)

    def mark(self, class_index, phase):
        """Progress marker per (class, phase), for resume."""
        (self.directory / "progress").write_text(f"{class_index} {phase}\n")

    def total_bytes(self):
        return sum(path.stat().st_size
                   for path in self.directory.glob("*.npy"))

    def destroy(self):
        for key in list(self._open):
            self.close(*key)
        shutil.rmtree(self.directory, ignore_errors=True)


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


def plan_tiling(grids, kernel_nz, budget_bytes=None, device='cpu', force=None):
    """Choose the RAM horizon, the per-class tile grids, and the halo.

    ``force`` is (horizon, tiles_x, tiles_y) and overrides the search entirely:
    every test runs at a toy scale where nothing would tile on its own, so
    forcing is how the streamed path is exercised at all. The forced tile counts
    apply to every streamed class.

    Halo per side, in cells of the class's own grid, and constant across classes
    because the kernel is: the convolution reaches ``kernel_nz // 2`` cells
    (SUPPORT_FACTOR * k in physical units, which is 6 * s cells at every class),
    and the advective weight's gradient stencil reaches one more. Anything
    beyond that is slack, and slack in a halo is only cost.
    """
    n_classes = len(grids['k'])
    halo_cells = kernel_nz // 2 + 1
    halo = (halo_cells, halo_cells)

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

    # Peak scratch: at class i the previous class's three state fields are still
    # live (pass 0 reads them) alongside this class's three state fields and two
    # candidate increments. The previous class is dropped as soon as pass 0
    # finishes, so this is the true peak rather than the cascade's total.
    peak = 0
    for index in range(max(horizon, 1), n_classes):
        peak = max(peak, 3 * _field_bytes(grids, index - 1)
                   + 5 * _field_bytes(grids, index))
    if horizon == 0:
        peak = max(peak, 5 * _field_bytes(grids, 0))
    return TilePlan(horizon, tiles, halo, peak)


def check_scratch_space(directory, needed_bytes):
    """Refuse up front, with the number, rather than dying mid-cascade."""
    directory = Path(directory)
    probe = directory if directory.exists() else directory.parent
    free = shutil.disk_usage(probe).free
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

    The interpolation-compensation DEFICITS are not carried (component 4). They
    feed only the output composition -- the cascade state never reads them -- so
    the states and the ledger this returns are complete, and the deficits become
    component 4's business along with the composition that consumes them.

    Returns (h_perturbation, qt_perturbation, flux, None, None, None,
    final_grid_info, ledger, store). The three fields are memory-mapped views on
    the scratch store rather than resident arrays: assembling them in RAM would
    undo the whole point at production scale, and component 4 writes them out as
    hyperslabs. The caller owns the store and must destroy() it.
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
    ledger = RunLedger(n_classes)
    store = TileStore(scratch_dir)
    scale_c = FLUX_SCALE if flux_noise_scale is None else flux_noise_scale

    # ---- above the horizon: today's resident loop, unchanged ----------------
    if horizon > 0:
        resident = {key: (grids[key][:horizon]
                          if key not in ('z_arrays', 'dz_arrays')
                          else grids[key][:horizon])
                    for key in grids}
        (h_pert, qt_pert, flux_field, _, _, _, _,
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
        )
        for index in range(horizon):
            ledger.classes[index] = head_ledger.classes[index]
        for name, field in (('h_perturbation', h_pert),
                            ('qt_perturbation', qt_pert),
                            ('flux', flux_field)):
            store.create(horizon - 1, name, field.shape)[...] = field
            store.close(horizon - 1, name)
        del h_pert, qt_pert, flux_field
    else:
        # Nothing ran in RAM, so class 0's store IS the cascade's initial state.
        shape = (int(grids['nx'][0]), int(grids['ny'][0]), int(grids['nz'][0]))
        store.create(-1, 'h_perturbation', shape, fill=0.0)
        store.create(-1, 'qt_perturbation', shape, fill=0.0)
        store.create(-1, 'flux', shape, fill=1.0)

    # The flux's per-class volume-mean restore, folded into the NEXT class's
    # read instead of costing a third sweep over the tiles (spec, component 3).
    # In-RAM order is rescale-then-regrid, so it is applied to the source slab
    # before interpolation -- see regrid_window's `scale`.
    pending_flux_rescale = None

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

        # ---- pass 0: regrid the previous class's store onto this grid -------
        print(f'Streamed {index + 1:3d}/{n_classes:3d}: '
              f'grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}) '
              f'{tiles_x}x{tiles_y} tiles  regrid...', end='\r')
        for name in STATE_FIELDS:
            source_shape = store.open(previous, name).shape
            store.create(index, name, world_shape)
            scale = pending_flux_rescale if name == 'flux' else None

            def read_slab(x_lo, x_hi, y_lo, y_hi, _name=name):
                return store.read_window(previous, _name, x_lo, x_hi,
                                         y_lo, y_hi)

            if source_shape == world_shape:
                # Same grid (the first streamed class after a resident head):
                # nothing to resample, so copy plane by plane.
                source_planes = store.plane_array(previous, name)
                target_planes = store.plane_array(index, name)
                for level in range(nz_k):
                    if scale is None:
                        target_planes[level] = source_planes[level]
                    else:
                        target_planes[level] = (source_planes[level]
                                                * np.float32(scale))
            else:
                for x_lo, x_hi, y_lo, y_hi in windows:
                    store.write_window(
                        index, name, x_lo, x_hi, y_lo, y_hi,
                        regrid_window(read_slab, source_shape, world_shape,
                                      (x_lo, x_hi, y_lo, y_hi), scale=scale))
            store.plane_array(index, name).flush()
            store.close(index, name)
        for name in STATE_FIELDS:
            store.drop(previous, name)
        pending_flux_rescale = None
        store.mark(index, 'regrid')

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
        store.mark(index, 'measure')

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
        store.plane_array(index, 'flux_next').flush()
        store.drop(index, 'flux')
        store.rename(index, 'flux_next', 'flux')
        for name in CANDIDATE_FIELDS:
            store.plane_array(index, name).flush()

        class_ledger.flux = FluxAdvanceLedger(entering, noise_abs, realized,
                                              n_clipped)
        # Folded into the next class's read (see pending_flux_rescale).
        pending_flux_rescale = class_ledger.flux.rescale
        if pending_flux_rescale is None:
            raise RuntimeError(
                f"class {index}: the whole domain clipped to zero, which the "
                f"in-RAM path handles by flattening the flux to its entering "
                f"mean. Not reachable at any sane amplitude, and not "
                f"implemented streamed rather than implemented untested.")
        store.mark(index, 'apply')

        # ---- pass 3: PLANE PASS (the bounded add) --------------------------
        print(f'Streamed {index + 1:3d}/{n_classes:3d}: '
              f'grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}) '
              f'{tiles_x}x{tiles_y} tiles  planes... ', end='\r')
        for (name, state_name, increment_name, mean_1d, _, _, phi_min,
             phi_max) in scalars:
            state_planes = store.plane_array(index, state_name)
            candidate_planes = store.plane_array(index, increment_name)
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
                state_planes[level] = plane_state[:, :, 0]
                for field in ('a0', 'n_loop', 'n_scale', 'final_demean', 'mu'):
                    getattr(solve, field)[level] = getattr(level_solve, field)[0]
                solve.demean[level] = level_solve.demean[0]
                solve.scale[level] = level_solve.scale[0]
            class_ledger.bounded_add[name] = solve
            store.plane_array(index, state_name).flush()
            store.drop(index, increment_name)
        store.mark(index, 'planes')
        print(f'Streamed {index + 1:3d}/{n_classes:3d}: '
              f'grid=({nx_k:4d},{ny_k:4d},{nz_k:4d}) '
              f'{tiles_x}x{tiles_y} tiles  done          ')

    # The last class's rescale has no next read to fold into.
    if pending_flux_rescale is not None:
        flux_store = store.open(n_classes - 1, 'flux')
        for x_lo, x_hi, y_lo, y_hi in windows:
            flux_store[x_lo:x_hi, y_lo:y_hi] *= np.float32(
                pending_flux_rescale)
        store.plane_array(n_classes - 1, 'flux').flush()

    final = n_classes - 1
    final_grid_info = {
        'nx': int(grids['nx'][final]), 'ny': int(grids['ny'][final]),
        'nz': int(grids['nz'][final]), 'dx': float(grids['dx'][final]),
        'dy': float(grids['dy'][final]), 'dz': grids['dz_arrays'][final],
        'z': grids['z_arrays'][final],
    }
    return (store.open(final, 'h_perturbation'),
            store.open(final, 'qt_perturbation'),
            store.open(final, 'flux'),
            None, None, None, final_grid_info, ledger, store)
