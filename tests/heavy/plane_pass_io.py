"""How much disk traffic a per-level pass costs, by storage layout.

Run:  uv run python tests/heavy/plane_pass_io.py

The streamed driver's plane pass (and component 4's projection and composition)
walk the field one z-level at a time. On a C-ordered (x, y, z) array the
elements of one level are nz * 4 bytes apart, so a 4 KiB page holds only
4096 / (nz * 4) useful values and reading ONE level touches essentially every
page of the file. Per-level iteration then costs nz times the field size in
real I/O -- invisible at toy scale, where the page cache absorbs everything,
and fatal at production (nz ~ 500 against a ~0.5 TB finest-class field).

Measured honestly rather than argued: `posix_fadvise(DONTNEED)` evicts the
file's clean pages between phases, so `/proc/self/io`'s read_bytes is real block
I/O rather than page-cache hits. AMPLIFICATION is read_bytes / file_size --
1.0 means the pass read the field once, which is the floor.

THE SCRATCH DIRECTORY MUST BE DISK-BACKED. /tmp is tmpfs on this box, and a
tmpfs file produces no block I/O at all, so read_bytes stays flat at zero and
the measurement silently says nothing. Default is a directory beside the repo
on /home (btrfs, NVMe); pass a path to override. This is the same reason
simulate() stages its class increments beside the output file rather than in
/tmp, and the same reason component 5 should document that scratch_dir wants
NVMe.
"""

import os
import time
from pathlib import Path

import numpy as np


def _is_tmpfs(path):
    """tmpfs files never reach the block layer, so read_bytes would stay flat."""
    import subprocess
    out = subprocess.run(['df', '--output=fstype', str(path)],
                         capture_output=True, text=True).stdout
    return 'tmpfs' in out


def read_bytes_now():
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    raise RuntimeError("/proc/self/io has no read_bytes on this kernel")


def evict(path):
    """Drop this file's clean page cache. No root needed, unlike drop_caches."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def write_field(path, shape):
    array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                      shape=shape)
    rng = np.random.default_rng(0)
    # Fill in slabs so the writer itself does not need the whole field resident.
    for start in range(0, shape[0], max(1, shape[0] // 16)):
        stop = min(start + max(1, shape[0] // 16), shape[0])
        array[start:stop] = rng.normal(
            0, 1, (stop - start,) + shape[1:]).astype(np.float32)
    array.flush()
    del array
    return path.stat().st_size


def time_plane_pass(path, layout, nx, ny, nz, cold=False):
    """Read every z-level, as the plane pass does, and report real I/O.

    ``cold`` evicts before EVERY level, which is the production regime: the
    finest-class field is ~0.5 TB against 60 GiB of RAM, so by the time the pass
    comes back for level 1 nothing of level 0's read survives in cache. With a
    warm cache (field smaller than RAM, i.e. every test in this repo) both
    layouts read the field exactly once and the amplification is invisible --
    which is why this had to be reasoned about rather than caught by a test.
    """
    evict(path)
    array = np.load(path, mmap_mode='r')
    before = read_bytes_now()
    start = time.perf_counter()
    total = 0.0
    for level in range(nz):
        if cold:
            del array
            evict(path)
            array = np.load(path, mmap_mode='r')
        if layout == 'xyz':
            plane = np.ascontiguousarray(array[:, :, level])
        else:
            plane = np.ascontiguousarray(array[level])
        total += float(plane[0, 0])
    elapsed = time.perf_counter() - start
    consumed = read_bytes_now() - before
    del array
    return consumed, elapsed


def time_tile_pass(path, layout, nx, ny, nz, tiles=4, halo=7, cold=False):
    """Read every tile plus its halo, as passes 1 and 2 do."""
    evict(path)
    array = np.load(path, mmap_mode='r')
    before = read_bytes_now()
    start = time.perf_counter()
    if cold:
        pass
    edges = [(i * nx) // tiles for i in range(tiles + 1)]
    for i in range(tiles):
        for j in range(tiles):
            x_lo, x_hi = max(0, edges[i] - halo), min(nx, edges[i + 1] + halo)
            y_lo, y_hi = max(0, edges[j] - halo), min(ny, edges[j + 1] + halo)
            if cold:
                del array
                evict(path)
                array = np.load(path, mmap_mode='r')
            if layout == 'xyz':
                slab = np.ascontiguousarray(array[x_lo:x_hi, y_lo:y_hi, :])
            else:
                slab = np.ascontiguousarray(
                    array[:, x_lo:x_hi, y_lo:y_hi]).transpose(1, 2, 0)
            del slab
    elapsed = time.perf_counter() - start
    consumed = read_bytes_now() - before
    del array
    return consumed, elapsed


def main():
    import sys
    import tempfile

    # Large enough that the page cache cannot hide the pattern, small enough to
    # run in seconds: 64 MiB per layout, with an nz in the production range.
    nx = ny = 256
    nz = 256
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    directory = Path(tempfile.mkdtemp(prefix="steam_io_", dir=str(root)))
    if _is_tmpfs(directory):
        raise SystemExit(
            f"{directory} is on tmpfs: a RAM-backed file does no block I/O, so "
            f"this benchmark would measure nothing. Pass a disk-backed path.")
    try:
        paths = {}
        for layout, shape in (('xyz', (nx, ny, nz)), ('zxy', (nz, nx, ny))):
            paths[layout] = directory / f"{layout}.npy"
            size = write_field(paths[layout], shape)
        print(f"field {nx}x{ny}x{nz} float32 = {size / 1024**2:.0f} MiB, "
              f"nz = {nz}\n")

        print(f"{'pass':<22} {'layout':<7} {'read':>10} {'amplif':>9} "
              f"{'time':>8}")
        for cold in (False, True):
            label = "cold" if cold else "warm"
            for layout in ('xyz', 'zxy'):
                consumed, elapsed = time_plane_pass(paths[layout], layout,
                                                    nx, ny, nz, cold=cold)
                print(f"{'per-level (' + label + ')':<22} {layout:<7} "
                      f"{consumed / 1024**2:>8.0f} M {consumed / size:>8.1f}x "
                      f"{elapsed:>7.2f}s")
        # Tile reads, cold, over tile widths. The zxy layout reads a tile as
        # one row-run per (level, x); a run shorter than a 4 KiB page pulls a
        # whole page for part of its contents, so narrow tiles pay for the
        # plane pass's contiguity. Production tiles are ~1024+ cells wide.
        for tiles in (2, 4, 16):
            for layout in ('xyz', 'zxy'):
                consumed, elapsed = time_tile_pass(paths[layout], layout, nx,
                                                   ny, nz, tiles=tiles,
                                                   cold=True)
                width = ny // tiles
                print(f"{'tile cold, y=' + str(width):<22} {layout:<7} "
                      f"{consumed / 1024**2:>8.0f} M {consumed / size:>8.1f}x "
                      f"{elapsed:>7.2f}s")

        print("\nThe cold per-level row is the production regime: field >> RAM, "
              "so\nnothing survives cache between levels.")
    finally:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == '__main__':
    main()
