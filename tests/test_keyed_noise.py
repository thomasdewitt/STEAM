"""Tests for the counter-based (world-keyed) noise core, steam/noise.py.

Two independent legs of validation, because "I wrote a hash and it looked
random" is not a standard the cascade's generator should be held to:

- EXACTNESS. The Philox4x32-10 implementation reproduces the published
  Random123 known-answer vectors, and the round structure it shares with
  Philox4x64-10 reproduces numpy's own Philox bit generator bit for bit.
  Between them these pin the constants, the round assembly, the key-bump
  schedule and the round count against two external references.
- DISTRIBUTION. Uniformity, independence of the two lanes, and absence of
  lattice structure -- the properties a counter-based generator can fail even
  with the arithmetic right, because its input is a low-entropy lattice index
  rather than a stream position.

Plus purity: the value at a world site does not depend on the box it was
generated in. That is the property the whole streaming feature rests on.
"""

import numpy as np
import pytest

from steam.noise import (
    _philox4x32_10,
    center_indices,
    class_key,
    keyed_uniforms,
)


U64 = np.uint64


def _draw(world_ix, world_iy, world_iz, key):
    """Both lanes over an index box, shaped (nx, ny, nz)."""
    shape = (len(world_ix), len(world_iy), len(world_iz))
    n = shape[0] * shape[1] * shape[2]
    lane0 = np.empty(n, dtype=np.float32)
    lane1 = np.empty(n, dtype=np.float32)
    keyed_uniforms(np.asarray(world_ix, dtype=np.int64),
                   np.asarray(world_iy, dtype=np.int64),
                   np.asarray(world_iz, dtype=np.int64),
                   key[0], key[1], key[2], lane0, lane1)
    return lane0.reshape(shape), lane1.reshape(shape)


# ---------------------------------------------------------------------------
# Exactness against external references
# ---------------------------------------------------------------------------

# Random123 kat_vectors, philox4x32 at 10 rounds: counter and key words in,
# four output words out. These are the reference implementation's published
# values, so reproducing them pins every constant and the whole round
# schedule without trusting anything in this repo.
PHILOX4X32_10_KAT = [
    ((0x00000000, 0x00000000, 0x00000000, 0x00000000),
     (0x00000000, 0x00000000),
     (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
     (0xFFFFFFFF, 0xFFFFFFFF),
     (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
     (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


@pytest.mark.parametrize("counter,key,expected", PHILOX4X32_10_KAT)
def test_philox4x32_10_matches_random123_known_answers(counter, key, expected):
    got = _philox4x32_10(U64(counter[0]), U64(counter[1]), U64(counter[2]),
                         U64(counter[3]), U64(key[0]), U64(key[1]))
    assert tuple(int(v) for v in got) == expected


MASK64 = (1 << 64) - 1
_M64 = (0xD2E7470EE14C6C93, 0xCA5A826395121157)
_W64 = (0x9E3779B97F4A7C15, 0xBB67AE8584CAA73B)


def _philox4x64_10(counter, key):
    """Philox4x64-10, the SAME round structure as steam.noise's 4x32 with the
    64-bit constants. Only the word size differs, so agreement with numpy's
    Philox pins the structure the jitted 4x32 kernel is built on."""
    c = list(counter)
    k = list(key)
    for r in range(10):
        if r > 0:
            k[0] = (k[0] + _W64[0]) & MASK64
            k[1] = (k[1] + _W64[1]) & MASK64
        product0 = _M64[0] * c[0]
        product1 = _M64[1] * c[2]
        hi0, lo0 = product0 >> 64, product0 & MASK64
        hi1, lo1 = product1 >> 64, product1 & MASK64
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0]
    return c


@pytest.mark.parametrize("key,counter", [
    ((1, 2), (7, 11, 13, 17)),
    ((0x0123456789ABCDEF, 0xFEDCBA9876543210), (7, 11, 13, 17)),
    ((0, 0), (0, 0, 0, 0)),
    ((2**64 - 1, 2**64 - 1), (2**64 - 2, 5, 6, 7)),
])
def test_philox_round_structure_matches_numpy(key, counter):
    """numpy's Philox generates the block for counter+1 (it increments word 0
    before filling its buffer -- verified from the reported state), and hands
    the four words out in order."""
    bit_generator = np.random.Philox(
        key=np.array(key, dtype=np.uint64),
        counter=np.array(counter, dtype=np.uint64))
    got = [int(v) for v in bit_generator.random_raw(4)]

    advanced = list(counter)
    advanced[0] = (advanced[0] + 1) & MASK64
    if advanced[0] == 0:
        advanced[1] = (advanced[1] + 1) & MASK64
    assert got == _philox4x64_10(advanced, key)


def test_uniforms_live_on_numpys_float32_lattice():
    """The uint32 -> float32 conversion is numpy's own: 24 random bits times
    2**-24. Matching the SUPPORT (not just the moments) is what makes the
    keyed uniforms distributionally identical to the stream-ordered ones,
    since the CMS transform they feed is shared."""
    key = class_key(np.random.SeedSequence(11))
    lane0, lane1 = _draw(np.arange(40), np.arange(40), np.arange(40), key)
    reference = np.random.default_rng(0).random(64000, dtype=np.float32)

    for values in (lane0.ravel(), lane1.ravel(), reference):
        scaled = values.astype(np.float64) * 2.0**24
        assert np.all(scaled == np.round(scaled))
        assert values.min() >= 0.0
        assert values.max() < 1.0
    assert lane0.max() <= 1.0 - 2.0**-24


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def big_draw():
    """~2.1 M draws per lane over a 128 x 128 x 128 world box."""
    key = class_key(np.random.SeedSequence(2718))
    return _draw(np.arange(128), np.arange(128), np.arange(128), key)


def test_uniform_marginals(big_draw):
    for lane in big_draw:
        values = lane.ravel().astype(np.float64)
        assert abs(values.mean() - 0.5) < 5e-4
        assert abs(values.var() - 1.0 / 12.0) < 5e-4
        counts, _ = np.histogram(values, bins=64, range=(0.0, 1.0))
        expected = values.size / 64.0
        chi2 = np.sum((counts - expected) ** 2 / expected)
        # 63 dof: the 1e-6 upper tail is ~140. A generator with visible bias
        # over 2 M draws fails this by orders, not by a hair.
        assert chi2 < 140.0


def test_lanes_are_independent(big_draw):
    lane0, lane1 = (lane.ravel().astype(np.float64) for lane in big_draw)
    n = lane0.size
    assert abs(np.corrcoef(lane0, lane1)[0, 1]) < 5.0 / np.sqrt(n)

    # Joint uniformity on an 8x8 grid: the strong form, since zero
    # correlation alone would not exclude a structured dependence.
    joint, _, _ = np.histogram2d(lane0, lane1, bins=8,
                                 range=((0, 1), (0, 1)))
    expected = n / 64.0
    chi2 = np.sum((joint - expected) ** 2 / expected)
    assert chi2 < 140.0        # 63 dof again


def test_no_lattice_structure_in_a_2d_slice():
    """The failure mode specific to a counter-based generator: the counter is
    a lattice index, so a weak hash leaves periodic structure that a marginal
    histogram cannot see. Checked two ways on a 512 x 512 slice of world
    sites -- small-lag correlations, and flatness of the 2D power spectrum."""
    key = class_key(np.random.SeedSequence(31415))
    lane0, _ = _draw(np.arange(512), np.arange(512), [0], key)
    field = lane0[:, :, 0].astype(np.float64) - 0.5
    n = field.size

    for lag_x in range(4):
        for lag_y in range(4):
            if lag_x == 0 and lag_y == 0:
                continue
            shifted = np.roll(np.roll(field, lag_x, axis=0), lag_y, axis=1)
            correlation = float(np.mean(field * shifted) / np.var(field))
            assert abs(correlation) < 5.0 / np.sqrt(n), (lag_x, lag_y)

    power = np.abs(np.fft.rfft2(field)) ** 2
    power[0, 0] = 0.0
    mean_power = power.sum() / (power.size - 1)
    # White noise gives exponentially distributed power; over ~1.3e5 modes the
    # largest is ~log(N) ~ 12 times the mean. 40 leaves a wide margin while
    # still catching any genuine spectral spike (a periodic artifact would put
    # orders of magnitude into one mode).
    assert power.max() / mean_power < 40.0


def test_distinct_class_keys_give_uncorrelated_fields():
    """Adjacent size classes draw at overlapping world sites, so their keys
    must decorrelate the fields completely -- otherwise the cascade's classes
    share noise."""
    box = (np.arange(64), np.arange(64), np.arange(64))
    children = np.random.SeedSequence(99).spawn(3)
    fields = [_draw(*box, class_key(child))[0].ravel().astype(np.float64)
              for child in children]
    n = fields[0].size
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(np.corrcoef(fields[i], fields[j])[0, 1]) < 5.0 / np.sqrt(n)


# ---------------------------------------------------------------------------
# Purity: the property the streaming feature rests on
# ---------------------------------------------------------------------------

def test_value_at_a_world_site_is_independent_of_the_box():
    """The same world site drawn inside three differently shaped, differently
    offset boxes gives bit-identical values in both lanes."""
    key = class_key(np.random.SeedSequence(5))
    full0, full1 = _draw(np.arange(48), np.arange(40), np.arange(32), key)

    sub0, sub1 = _draw(np.arange(11, 29), np.arange(7, 23), np.arange(3, 19),
                       key)
    np.testing.assert_array_equal(sub0, full0[11:29, 7:23, 3:19])
    np.testing.assert_array_equal(sub1, full1[11:29, 7:23, 3:19])

    # A single site, and a box that is not even contiguous in world space.
    one0, _ = _draw([37], [5], [21], key)
    assert one0[0, 0, 0] == full0[37, 5, 21]
    scattered0, _ = _draw([2, 37, 9], [5], [21, 0], key)
    assert scattered0[1, 0, 0] == full0[37, 5, 21]
    assert scattered0[2, 0, 1] == full0[9, 5, 0]


def test_center_indices_phase_comes_from_the_world_origin():
    """At s > 1 a center sits at every world index divisible by s, so which
    LOCAL cells are centers depends on where the region starts in the world."""
    local, world = center_indices(10, 0, 16, 2, wrap=True)
    np.testing.assert_array_equal(local, [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(world, [0, 2, 4, 6, 8])

    # Origin 5 with s=2: the first world multiple of 2 at or after 5 is 6,
    # which is local cell 1.
    local, world = center_indices(10, 5, 16, 2, wrap=True)
    np.testing.assert_array_equal(local, [1, 3, 5, 7, 9])
    np.testing.assert_array_equal(world, [6, 8, 10, 12, 14])

    local, world = center_indices(7, 4, 9, 3, wrap=True)
    np.testing.assert_array_equal(local, [2, 5])
    np.testing.assert_array_equal(world, [6, 0])       # 9 wraps to 0

    # s = 1: every cell is a center, and the world index is the origin offset.
    local, world = center_indices(4, 30, 32, 1, wrap=True)
    np.testing.assert_array_equal(local, [0, 1, 2, 3])
    np.testing.assert_array_equal(world, [30, 31, 0, 1])


def test_class_key_is_a_pure_function_of_the_seed_sequence():
    """Per-class seeding is unchanged: spawn(n)[i] identifies class i, so a
    nest that skips the classes already spent gets the same keys a deeper root
    run would have used."""
    root = np.random.SeedSequence(1234)
    keys_full = [class_key(child) for child in root.spawn(6)]
    keys_again = [class_key(child)
                  for child in np.random.SeedSequence(1234).spawn(6)]
    assert keys_full == keys_again

    continued = [class_key(child) for child in
                 np.random.SeedSequence(1234).spawn(6)[4:]]
    assert continued == keys_full[4:]
    assert len(set(keys_full)) == 6
