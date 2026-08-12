"""Counter-based (world-keyed) random draws for the turbulon cascade.

The cascade's noise used to be a STREAM: `_sparse_levy` reshaped one linear
PCG64 sequence over whatever array it was handed, so the value a turbulon
center received depended on where that center sat in the array rather than
where it sat in the world. Two consequences, one wanted and one a bug:

- Tiling is impossible. A tile is a different array, so the same world
  position draws a different value, and a streamed run cannot be the
  in-RAM run with the loops reordered.
- Sibling nests were noise CLONES. `refine` with seed=None hands every
  sibling the same per-class SeedSequence (which is correct — they continue
  the same root stream), and stream-ordered draws then give two disjoint
  regions the identical sequence of values, laid over different ground.
  Same-config siblings differed only in the state they inherited.

Both fall out of making the draw a pure function of (root_seed, class index,
world lattice site). That is what this module provides: Philox4x32-10, a
counter-based generator, evaluated at the world site itself. Any region's
noise field is then exactly the restriction of the root's noise field to that
region — bit-for-bit, with no bookkeeping, in any order, from any process.

Why Philox rather than a hand-rolled mixing hash (2026-08-12): it is a
published, independently analyzed CBRNG (Salmon et al., SC'11, "Parallel
random numbers: as easy as 1, 2, 3") whose whole design goal is exactly this
use — full statistical quality per counter value, with the counter carrying
structured, low-entropy input like a lattice index. The 4x32 variant is the
one that suits a jitted kernel: its round function needs a 32x32 -> 64 bit
multiply, which is native uint64 arithmetic, where Philox4x64 would need a
64x64 -> 128 multiply synthesized from four 32-bit pieces. Ten rounds is the
Random123 default (the construction is statistically clean from about seven).

The counter is the lattice site and the key is the class: counter words
(ix, iy, iz, key2) against key words (key0, key1). That leaves 96 bits of
per-class key material, drawn from the class's SeedSequence child, and gives
each of the three world axes a full 32-bit range with no packing.

Numba note: every intermediate here is np.uint64 and every constant is
np.uint64. Mixing a uint64 with a Python int in numba promotes the result to
float64 (silently, and the multiply would then lose the low bits), so the
casts are load-bearing rather than decoration.
"""

import numpy as np
from numba import njit, prange

# Philox4x32 round constants (Random123).
_M0 = np.uint64(0xD2511F53)
_M1 = np.uint64(0xCD9E8D57)
# Per-round key bumps: the golden ratio and sqrt(3)-1 in 32-bit fixed point.
_W0 = np.uint64(0x9E3779B9)
_W1 = np.uint64(0xBB67AE85)
_MASK32 = np.uint64(0xFFFFFFFF)
_SHIFT32 = np.uint64(32)
_PHILOX_ROUNDS = 10

# uint32 -> float32 in [0, 1). This is numpy's own float32 conversion --
# 24 random bits scaled by 2**-24 (`(next_uint32() >> 8) * (1.0f/16777216.0f)`
# in numpy's `next_float`) -- reproduced exactly so the keyed uniforms live on
# the identical discrete lattice the stream-ordered generator drew from. The
# CMS transform downstream is unchanged, so matching the uniforms' support
# makes the two schemes distributionally identical, not merely similar.
_SHIFT8 = np.uint64(8)
_TWO_POW_M24 = np.float32(1.0 / 16777216.0)


@njit(inline='always', cache=True)
def _philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Ten Philox4x32 rounds over counter (c0..c3) with key (k0, k1).

    All six arguments and all four returns are uint64 holding 32-bit values.
    Round 0 uses the raw key; the key is bumped before each later round.
    """
    for r in range(_PHILOX_ROUNDS):
        if r > 0:
            k0 = (k0 + _W0) & _MASK32
            k1 = (k1 + _W1) & _MASK32
        product0 = _M0 * c0          # exact: both factors are < 2**32
        product1 = _M1 * c2
        hi0 = product0 >> _SHIFT32
        lo0 = product0 & _MASK32
        hi1 = product1 >> _SHIFT32
        lo1 = product1 & _MASK32
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0), lo1, (hi0 ^ c3 ^ k1), lo0
    return c0, c1, c2, c3


@njit(inline='always', cache=True)
def _uniform_pair(ix, iy, iz, key0, key1, key2):
    """The two float32 uniforms belonging to world site (ix, iy, iz).

    Lane 0 and lane 1 are Philox output words 0 and 1 of the single counter
    value for that site -- one generator call per site, not two, so the two
    lanes cost what one does. Words 2 and 3 are unused and are what a third
    per-center draw would come from if the model ever needs one.
    """
    r0, r1, _, _ = _philox4x32_10(
        np.uint64(ix), np.uint64(iy), np.uint64(iz), key2, key0, key1)
    return (np.float32(r0 >> _SHIFT8) * _TWO_POW_M24,
            np.float32(r1 >> _SHIFT8) * _TWO_POW_M24)


@njit(parallel=True, cache=True)
def keyed_uniforms(world_ix, world_iy, world_iz, key0, key1, key2,
                   lane0, lane1):
    """Fill two flat float32 lanes with the uniforms of a world index box.

    ``world_ix/iy/iz`` are the world lattice indices of the box's cells along
    each axis (already phase-selected and wrapped by the caller, so this
    kernel knows nothing about sparsity or periodicity). ``lane0``/``lane1``
    are length ``nx*ny*nz`` and are filled in C order over
    (world_ix, world_iy, world_iz) -- the layout
    ``field[np.ix_(ix, iy, iz)] = draws.reshape(nx, ny, nz)`` expects.

    Parallel over the slowest axis. At the production finest class that is
    thousands of independent chunks; the coarse classes are too small for the
    parallelism to matter either way.
    """
    n_y = world_iy.shape[0]
    n_z = world_iz.shape[0]
    for a in prange(world_ix.shape[0]):
        ix = np.uint64(world_ix[a])
        for b in range(n_y):
            iy = np.uint64(world_iy[b])
            base = (a * n_y + b) * n_z
            for c in range(n_z):
                u0, u1 = _uniform_pair(ix, iy, np.uint64(world_iz[c]),
                                       key0, key1, key2)
                lane0[base + c] = u0
                lane1[base + c] = u1


def class_key(seed_sequence):
    """The 96-bit Philox key of one size class, from its SeedSequence child.

    Per-class seeding is unchanged from the stream scheme:
    ``SeedSequence(root_seed).spawn(n_classes)[i]`` still identifies class i,
    so ``root_seed`` / ``n_classes_consumed`` bookkeeping keeps its meaning
    and `refine`'s continuation (skip the classes already spent) falls out
    exactly as before. Only what the child seeds changed: a counter-based key
    instead of a stream position.
    """
    if not isinstance(seed_sequence, np.random.SeedSequence):
        seed_sequence = np.random.SeedSequence(seed_sequence)
    state = seed_sequence.generate_state(3, dtype=np.uint32)
    return (np.uint64(state[0]), np.uint64(state[1]), np.uint64(state[2]))


def center_indices(n, origin, world_n, factor, wrap):
    """Turbulon-center cells of one axis, as (local index, world index).

    A center sits at every world site whose index is a multiple of ``factor``
    (the sparsity oversampling), so which LOCAL cells are centers depends on
    the region's world ``origin`` -- the sub-lattice phase comes from the
    world, not from the array corner. A root (origin 0) keeps centers at
    local 0, factor, 2*factor, ..., exactly as before.

    ``wrap`` takes the world index modulo ``world_n``, which is what makes a
    halo that runs off a periodic edge draw the noise of the world cells it
    actually wraps onto. z never wraps.

    The divisibility test is applied to the WRAPPED world index, which matters
    when ``world_n`` is not itself a multiple of ``factor`` (2026-08-12, codex
    review). Testing the unwrapped index first and wrapping afterwards is only
    equivalent when world_n % factor == 0: at world_n = 5, factor = 2,
    origin = 4, cells 0..2 map to world 4, 0, 1 and the true centers are locals
    0 and 1, where the unwrapped test returned locals 0 and 2 -- one center
    invented, one missed. The extent rules make world_n % s == 0 for standard
    configurations (nx_k = 2*s*m*2^i), so this reached only narrow-strip grids at
    s > 1; fixed at the root rather than asserted away, because a silent wrong
    lattice is exactly the class of bug world-keying exists to remove.

    A consequence: the local indices are no longer a regular stride when a
    region crosses the periodic boundary, so callers must scatter by index array
    rather than assuming an arange.
    """
    local_all = np.arange(n, dtype=np.int64)
    world_all = local_all + int(origin)
    if wrap:
        world_all = world_all % int(world_n)
    keep = (world_all % int(factor)) == 0
    return local_all[keep], world_all[keep]
