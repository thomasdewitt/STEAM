"""World-keyed noise: the restriction property, and the clone bug it fixes.

The generator is a pure function of (root seed, class index, world lattice
site) as of 2026-08-12, so ANY region's noise field is the restriction of the
root's noise field to that region -- bit for bit. That property is what makes
out-of-core tiling exact rather than approximate, and it is tested here at
both levels: directly on the generator, and end to end through refine().

It also fixes a latent bug it is worth stating plainly. With stream-ordered
draws, `refine(seed=None)` gave every sibling nest the same per-class
SeedSequence -- correct, they continue the same root stream -- and the stream
then handed two DISJOINT regions the identical sequence of values. Two
same-config siblings over different ground were noise clones, differing only
in the state they inherited. See test_sibling_nests_are_not_noise_clones.
"""

import importlib
import shutil

import numpy as np
import pytest
import netCDF4
from scipy import stats

sm = importlib.import_module("steam.simulate")

from steam.noise import class_key
from steam.simulate import (
    FLUX_ALPHA,
    NoiseRegion,
    _extremal_levy,
    _keyed_sparse_levy,
    refine,
    simulate,
)


ALPHA = 1.8


def _world_field(key, world_shape, sparsity_factors, alpha=ALPHA):
    """The class's noise field over the whole world grid."""
    return _keyed_sparse_levy(world_shape, sparsity_factors, alpha,
                              NoiseRegion(key, (0, 0, 0), world_shape))


def _restriction(world_field, origin, shape):
    """The world field restricted to a region, with x/y wrapping as the
    generator wraps them (z does not wrap)."""
    world_nx, world_ny, _ = world_field.shape
    ix = (np.arange(shape[0]) + origin[0]) % world_nx
    iy = (np.arange(shape[1]) + origin[1]) % world_ny
    iz = np.arange(shape[2]) + origin[2]
    return world_field[np.ix_(ix, iy, iz)]


# ---------------------------------------------------------------------------
# (a) The restriction property, on the generator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sparsity_factors", [
    (1, 1, 1),        # every cell a center
    (2, 2, 2),
    (2, 3, 1),        # anisotropic sub-lattice
    (4, 4, 2),
])
@pytest.mark.parametrize("origin", [
    (0, 0, 0),        # the world origin itself
    (7, 5, 3),        # a phase that misses the sub-lattice on every axis
    (8, 12, 4),       # a phase aligned with it
    (13, 1, 0),
])
def test_subregion_noise_is_the_root_restriction(sparsity_factors, origin):
    """The core property. Bit-equal, not close: the region's draws ARE the
    world's draws at those sites, so there is nothing to round."""
    world_shape = (48, 40, 24)
    key = class_key(np.random.SeedSequence(17))
    world = _world_field(key, world_shape, sparsity_factors)

    shape = (16, 14, 12)
    region = _keyed_sparse_levy(
        shape, sparsity_factors, ALPHA,
        NoiseRegion(key, origin, world_shape))
    np.testing.assert_array_equal(region, _restriction(world, origin, shape))


@pytest.mark.parametrize("sparsity_factors", [(1, 1, 1), (2, 2, 2), (3, 2, 1)])
def test_restriction_holds_across_a_periodic_wrap(sparsity_factors):
    """A halo (or a tile) running off a periodic edge covers the world cells it
    wraps onto, and must draw exactly their noise -- otherwise the seam a
    streamed run stitches across the boundary carries a discontinuity in the
    noise itself."""
    world_shape = (36, 36, 18)
    key = class_key(np.random.SeedSequence(23))
    world = _world_field(key, world_shape, sparsity_factors)

    # Origin near the far corner, region wide enough to wrap in x and y.
    origin = (30, 33, 2)
    shape = (14, 10, 8)
    region = _keyed_sparse_levy(
        shape, sparsity_factors, ALPHA,
        NoiseRegion(key, origin, world_shape))
    np.testing.assert_array_equal(region, _restriction(world, origin, shape))

    # A region as wide as the world wraps back onto itself exactly once.
    whole = _keyed_sparse_levy(
        world_shape, sparsity_factors, ALPHA,
        NoiseRegion(key, (world_shape[0], 0, 0), world_shape))
    np.testing.assert_array_equal(whole, world)


@pytest.mark.parametrize("sparsity_factors", [(1, 1, 1), (2, 2, 2)])
def test_restriction_holds_for_nest_continuation_classes(sparsity_factors):
    """A nest's classes are the root's stream continued -- spawn(N)[M:] -- so
    the restriction property has to hold at the classes a root run never
    reached, which is where every nest actually draws."""
    n_consumed, n_new = 5, 3
    root = np.random.SeedSequence(1234)
    continuation = root.spawn(n_consumed + n_new)[n_consumed:]
    # The keys a deeper root run would have used at those same classes.
    deeper = np.random.SeedSequence(1234).spawn(n_consumed + n_new)

    world_shape = (32, 32, 20)
    shape = (12, 10, 9)
    origin = (11, 6, 4)
    for i in range(n_new):
        key = class_key(continuation[i])
        assert key == class_key(deeper[n_consumed + i])
        world = _world_field(key, world_shape, sparsity_factors)
        region = _keyed_sparse_levy(
            shape, sparsity_factors, ALPHA,
            NoiseRegion(key, origin, world_shape))
        np.testing.assert_array_equal(
            region, _restriction(world, origin, shape))


# ---------------------------------------------------------------------------
# (d) Sibling independence: the clone bug
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sparsity_factors", [(1, 1, 1), (2, 2, 2)])
def test_disjoint_regions_of_one_class_are_not_clones(sparsity_factors):
    """Two disjoint regions of the SAME class share a key, so the old scheme
    gave them identical values. They must now differ -- while each still
    equals the world restriction, which is the property that distinguishes
    "consistent" from merely "reseeded"."""
    world_shape = (64, 64, 16)
    key = class_key(np.random.SeedSequence(77))
    world = _world_field(key, world_shape, sparsity_factors)

    shape = (12, 12, 8)
    left = (4, 4, 0)
    right = (36, 40, 0)
    field_left = _keyed_sparse_levy(shape, sparsity_factors, ALPHA,
                                    NoiseRegion(key, left, world_shape))
    field_right = _keyed_sparse_levy(shape, sparsity_factors, ALPHA,
                                     NoiseRegion(key, right, world_shape))

    assert not np.array_equal(field_left, field_right)
    # Not merely unequal: uncorrelated, over the centers they actually have.
    nonzero = (field_left != 0) & (field_right != 0)
    assert nonzero.sum() > 100
    correlation = np.corrcoef(field_left[nonzero].astype(np.float64),
                              field_right[nonzero].astype(np.float64))[0, 1]
    assert abs(correlation) < 5.0 / np.sqrt(nonzero.sum())

    np.testing.assert_array_equal(field_left, _restriction(world, left, shape))
    np.testing.assert_array_equal(field_right, _restriction(world, right, shape))


def test_regions_overlapping_agree_on_the_overlap():
    """The positive form of the same statement, and the one tiling needs: two
    regions that overlap must agree exactly where they do."""
    world_shape = (40, 40, 12)
    key = class_key(np.random.SeedSequence(88))
    first = _keyed_sparse_levy((20, 20, 12), (2, 2, 2), ALPHA,
                               NoiseRegion(key, (0, 0, 0), world_shape))
    second = _keyed_sparse_levy((20, 20, 12), (2, 2, 2), ALPHA,
                                NoiseRegion(key, (12, 8, 0), world_shape))
    np.testing.assert_array_equal(first[12:20, 8:20], second[0:8, 0:12])


# ---------------------------------------------------------------------------
# (b) Distribution unchanged
# ---------------------------------------------------------------------------

def _keyed_flat_draws(seed, size, alpha=ALPHA):
    """`size` keyed draws as a flat sample (s = 1, so every cell is a center)."""
    side = int(np.ceil(size ** (1.0 / 3.0)))
    field = _world_field(class_key(np.random.SeedSequence(seed)),
                         (side, side, side), (1, 1, 1), alpha=alpha)
    return field.ravel()[:size].astype(np.float64)


@pytest.mark.parametrize("alpha", [1.5, ALPHA, 1.95])
def test_keyed_draws_match_the_stream_generators_distribution(alpha):
    """The CMS transform is shared, so this tests the uniforms. Compared on the
    functional the model actually uses -- exp(gamma), whose mean is the thing
    the unit-mean shift is defined against -- plus quantiles across the body
    and a two-sample KS test over the whole distribution."""
    size = 300_000
    keyed = _keyed_flat_draws(101, size, alpha=alpha)
    stream = _extremal_levy(alpha, size,
                            np.random.default_rng(202)).astype(np.float64)

    # <exp(gamma)> = exp(1/(alpha-1)) exactly (LEVY_LOG_MEAN's closed form).
    expected_mean = np.exp(1.0 / (alpha - 1.0))
    for sample in (keyed, stream):
        assert abs(np.mean(np.exp(sample)) / expected_mean - 1.0) < 0.03

    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    keyed_q = np.quantile(keyed, quantiles)
    stream_q = np.quantile(stream, quantiles)
    # The body of an extremal-Levy sample; scale set by the interquartile
    # range so the tolerance means the same thing at every alpha.
    spread = stream_q[5] - stream_q[1]
    np.testing.assert_allclose(keyed_q, stream_q, atol=0.03 * spread)

    # Two independent samples of the same law. At 3e5 vs 3e5 the KS test
    # resolves CDF differences of a few tenths of a percent, so this is a
    # strong statement, and the fixed seeds make it a deterministic one.
    assert stats.ks_2samp(keyed, stream).pvalue > 1e-4


def test_keyed_draws_reduce_to_the_gaussian_case_at_alpha_two():
    """At alpha=2 the extremal generator is exactly N(0, 2), which pins the
    scale of the keyed draws and not just their shape."""
    draws = _keyed_flat_draws(303, 2_000_000, alpha=2.0)
    assert abs(draws.mean()) < 0.01
    assert abs(draws.var() - 2.0) < 0.02
    assert abs(float(np.log(np.mean(np.exp(draws)))) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# End to end through refine()
# ---------------------------------------------------------------------------

PARENT_NZ = 50
PARENT_PROFILE_DZ = 30.0
PARENT_DOMAIN_HEIGHT = 3000.0


def _profiles():
    z = np.arange(PARENT_NZ) * PARENT_PROFILE_DZ
    return 340e3 - 20e3 * (z / z.max()), 0.018 - 0.016 * (z / z.max())


@pytest.fixture(scope="module")
def parent_template(tmp_path_factory):
    h, qt = _profiles()
    out = tmp_path_factory.mktemp("keyed_parent") / "parent.nc"
    simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
             outer_scale=8000, spheroscale=100,
             domain_height=PARENT_DOMAIN_HEIGHT, profile_dz=PARENT_PROFILE_DZ,
             output_path=out, seed=42, save_for_refinement=True)
    return out


@pytest.fixture
def parent_nc(parent_template, tmp_path):
    destination = tmp_path / "parent.nc"
    shutil.copy(parent_template, destination)
    return destination


def _parent_geometry(path):
    with netCDF4.Dataset(path, "r") as ds:
        return {
            'nx': len(ds.dimensions["x"]),
            'dx': float(ds.dx),
            'k_finest': float(ds.variables["k_values"][:][-1]),
            'root_seed': int(ds.root_seed),
            'n_classes_consumed': int(ds.n_classes_consumed),
        }


def _capture_noise(monkeypatch):
    """Record every generator field the cascade draws, with its region."""
    drawn = []
    real = sm._keyed_sparse_levy

    def spy(shape, sparsity_factors, alpha, region):
        field = real(shape, sparsity_factors, alpha, region)
        drawn.append((region, field.copy()))
        return field

    monkeypatch.setattr(sm, "_keyed_sparse_levy", spy)
    return drawn


def _capture_world_geometry(monkeypatch):
    """Record the world origins/shapes refine() hands to the cascade."""
    seen = {}
    real = sm.cascade_loop

    def spy(*args, **kwargs):
        seen['world_origins'] = kwargs.get('world_origins')
        seen['world_shapes'] = kwargs.get('world_shapes')
        seen['grids'] = args[3]
        return real(*args, **kwargs)

    monkeypatch.setattr(sm, "cascade_loop", spy)
    return seen


def test_full_span_full_height_nest_is_anchored_at_the_world_origin(
        parent_nc, monkeypatch):
    """The case where world anchoring is exact rather than nearest-cell: a nest
    spanning the parent in x, y and z has no halo and rescales no dz, so its
    grid at each class IS the world grid there. Origin (0, 0, 0) and matching
    shapes are the check that the anchoring arithmetic is right."""
    geometry = _parent_geometry(parent_nc)
    seen = _capture_world_geometry(monkeypatch)
    new_dx = geometry['k_finest'] / 8

    refine(parent_nc, 0, geometry['nx'], 0, geometry['nx'], new_dx, new_dx,
           output_group="full")

    assert len(seen['world_origins']) >= 2      # more than one nest class
    for i, origin in enumerate(seen['world_origins']):
        assert origin == (0, 0, 0)
        assert seen['world_shapes'][i] == (int(seen['grids']['nx'][i]),
                                           int(seen['grids']['ny'][i]),
                                           int(seen['grids']['nz'][i]))


def test_nest_noise_is_the_root_restriction(parent_nc, monkeypatch):
    """End to end: every generator field a nest draws is the world field of
    that class restricted to the nest's own cells."""
    geometry = _parent_geometry(parent_nc)
    seen = _capture_world_geometry(monkeypatch)
    drawn = _capture_noise(monkeypatch)
    new_dx = geometry['k_finest'] / 8

    refine(parent_nc, 8, 16, 12, 20, new_dx, new_dx, output_group="quarter")

    assert len(drawn) == len(seen['world_origins'])
    # Not anchored at the origin -- otherwise this test proves nothing.
    assert any(origin[:2] != (0, 0) for origin in seen['world_origins'])

    for i, (region, field) in enumerate(drawn):
        assert region.origin == seen['world_origins'][i]
        world = _world_field(region.key, region.world_shape, (1, 1, 1))
        np.testing.assert_array_equal(
            field, _restriction(world, region.origin, field.shape))


def test_sibling_nests_are_not_noise_clones(parent_nc, monkeypatch):
    """THE regression. Two same-config nests over disjoint regions, both with
    seed=None so both continue the root's per-class stream. Before world-keyed
    noise their generator fields were bit-identical; now they differ, while
    each remains the restriction of the same world field."""
    geometry = _parent_geometry(parent_nc)
    new_dx = geometry['k_finest'] / 8

    drawn = _capture_noise(monkeypatch)
    refine(parent_nc, 4, 12, 4, 12, new_dx, new_dx, output_group="left")
    left = list(drawn)
    drawn.clear()
    refine(parent_nc, 20, 28, 20, 28, new_dx, new_dx, output_group="right")
    right = list(drawn)

    assert len(left) == len(right) >= 2
    for (region_left, field_left), (region_right, field_right) in zip(left, right):
        # Same class: same key, same shape -- which is exactly why the stream
        # scheme cloned them.
        assert region_left.key == region_right.key
        assert field_left.shape == field_right.shape
        assert region_left.origin != region_right.origin
        assert not np.array_equal(field_left, field_right)

        world = _world_field(region_left.key, region_left.world_shape, (1, 1, 1))
        np.testing.assert_array_equal(
            field_left, _restriction(world, region_left.origin, field_left.shape))
        np.testing.assert_array_equal(
            field_right, _restriction(world, region_right.origin,
                                      field_right.shape))


def test_same_region_nests_still_agree(parent_nc):
    """The other half of the ruling: consistency, not independence. Two nests
    over the SAME region with seed=None must still be identical -- world
    keying makes that a property of the region rather than of the draw order."""
    geometry = _parent_geometry(parent_nc)
    new_dx = geometry['k_finest'] / 8
    refine(parent_nc, 8, 16, 8, 16, new_dx, new_dx, output_group="a")
    refine(parent_nc, 8, 16, 8, 16, new_dx, new_dx, output_group="b")
    with netCDF4.Dataset(parent_nc, "r") as ds:
        np.testing.assert_array_equal(ds.groups["a"].variables["h"][:],
                                      ds.groups["b"].variables["h"][:])


def test_output_records_the_noise_scheme(parent_nc):
    """Files say which scheme drew them: realizations changed on 2026-08-12,
    so a single-realization comparison across that boundary is meaningless and
    has to be detectable from the file alone."""
    geometry = _parent_geometry(parent_nc)
    new_dx = geometry['k_finest'] / 8
    refine(parent_nc, 8, 16, 8, 16, new_dx, new_dx, output_group="scheme")
    with netCDF4.Dataset(parent_nc, "r") as ds:
        assert ds.noise_scheme == "world_keyed_philox4x32_10"
        assert ds.groups["scheme"].noise_scheme == "world_keyed_philox4x32_10"
        # The world every descendant anchors against.
        assert float(ds.root_domain_x) == 32 * 250.0
        assert float(ds.groups["scheme"].root_domain_x) == 32 * 250.0
        assert float(ds.groups["scheme"].root_domain_height) == PARENT_DOMAIN_HEIGHT


def test_refining_a_pre_keyed_parent_is_refused(parent_nc):
    """A file without the world attributes came from the stream generator, so
    its realization cannot be continued under keyed noise. Refuse with the
    reason rather than inventing a world for it."""
    with netCDF4.Dataset(parent_nc, "a") as ds:
        ds.delncattr("root_domain_x")
    geometry = _parent_geometry(parent_nc)
    new_dx = geometry['k_finest'] / 8
    with pytest.raises(ValueError, match="root_domain_x"):
        refine(parent_nc, 8, 16, 8, 16, new_dx, new_dx)


def test_cascade_loop_rejects_a_shared_generator():
    """The former legacy path. A Generator cannot be world-keyed, so it is an
    error rather than a silent reseed."""
    with pytest.raises(TypeError, match="class_seeds"):
        sm.cascade_loop(
            np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32),
            np.arange(4, dtype=np.float64),
            {'k': np.array([100.0]), 'nx': np.array([4]), 'ny': np.array([4]),
             'nz': np.array([4]), 'dx': np.array([25.0]),
             'dy': np.array([25.0]), 'dz': np.array([25.0]),
             'z_arrays': [np.arange(4, dtype=np.float64)],
             'dz_arrays': [np.ones(4)],
             'padded_extent_x': np.array([100.0]),
             'padded_extent_y': np.array([100.0]),
             'padded_height': np.array([100.0]),
             'z_min_per_class': np.array([0.0])},
            [np.ones(4, dtype=np.float32)], [np.ones(4, dtype=np.float32)],
            [np.ones(4, dtype=np.float32)], [np.ones(4, dtype=np.float32)],
            [np.ones(4, dtype=np.float32)],
            0.0, 1e9, -1.0, 1.0, 0, (1, 1, 1),
            np.random.default_rng(0),
        )
