"""The fundamental test of the streamed cascade: streamed = non-streamed.

Component 3's whole claim is that paging the state through RAM in horizontal
tiles is the same cascade with the loops reordered. Bit-exactness is NOT the
criterion (Thomas's ruling): a per-tile FFT has a different extent from the
full-domain FFT and float64 partial sums accumulate in a different order, so
the two paths differ at the rounding floor. What must NOT happen is
seam-correlated structure -- a difference field that knows where the tile
boundaries are. That is what test_no_seam_structure checks, and it is the test
the whole design stands or falls on.

Everything here runs cascade_loop directly rather than simulate(): component 3
delivers the cascade STATES and the ledger, and the output composition stays
in-RAM until component 4. Comparing states rather than written fields is also
what makes a failure attributable.
"""

import numpy as np
import pytest

from steam.ledger import RunLedger
from steam.simulate import (
    SUPPORT_FACTOR,
    _bound_buffers,
    _compute_all_grids,
    _compute_normalization,
    _k_z,
    _turbulon_envelope,
    cascade_loop,
)
from steam.streaming import (
    TileStore,
    _axis_regrid_plan,
    plan_tiling,
    regrid_window,
    streamed_cascade_loop,
    tile_bounds,
    tile_windows,
)
from steam.utils import zoom_trilinear


PROFILE_NZ = 28
PROFILE_DZ = 30.0
DOMAIN_HEIGHT = 800.0
SPHEROSCALE = 100.0
SEED = 42


def _config(nx=64, dx=125.0, outer_scale=2000.0, n_per_dyad=1,
            sparsity_factors=(1, 1, 1)):
    """Everything cascade_loop needs for a root run, built as simulate() does.

    Deliberately a real configuration rather than ones-everywhere: the product
    norm, the bound taper and the bounded add all behave differently on a field
    with structure, and the seam test is only meaningful on one.
    """
    z_profile = np.arange(PROFILE_NZ) * PROFILE_DZ
    h_profile = (340e3 - 20e3 * (z_profile / z_profile.max())).astype(np.float32)
    qt_profile = (0.018 - 0.016 * (z_profile / z_profile.max())).astype(np.float32)
    spheroscale_profile = np.full(PROFILE_NZ, SPHEROSCALE)

    gap = 2.0 ** (1.0 / n_per_dyad)
    n_classes = int(round(np.log(outer_scale / (2 * dx)) / np.log(gap))) + 1
    k_values = outer_scale / gap ** np.arange(n_classes)
    grids = _compute_all_grids(
        k_values, nx * dx, nx * dx, DOMAIN_HEIGHT, sparsity_factors,
        spheroscale_profile, z_profile)

    k_z_L = _k_z('piecewise_isotropic_below_spheroscale', outer_scale,
                 spheroscale_profile)
    C_h_k = _compute_normalization(h_profile, z_profile, k_z_L, k_values,
                                  outer_scale, grids,
                                  n_scale_classes_per_dyad=n_per_dyad)
    C_qt_k = _compute_normalization(qt_profile, z_profile, k_z_L, k_values,
                                   outer_scale, grids,
                                   n_scale_classes_per_dyad=n_per_dyad)
    aspect_k = []
    for k in k_values:
        ls = np.interp(grids['z_arrays'][len(aspect_k)], z_profile,
                       spheroscale_profile)
        aspect_k.append(
            (_k_z('piecewise_isotropic_below_spheroscale', k, ls) / k
             ).astype(np.float32))
    return dict(
        h_profile=h_profile, qt_profile=qt_profile, z_profile=z_profile,
        grids=grids, C_h_k=C_h_k, C_qt_k=C_qt_k,
        b_h_k=_bound_buffers(C_h_k, n_per_dyad, None, None),
        b_qt_k=_bound_buffers(C_qt_k, n_per_dyad, None, None),
        aspect_k=aspect_k, n_classes=n_classes,
        sparsity_factors=sparsity_factors, n_per_dyad=n_per_dyad,
    )


BOUNDS = (315 * 1004.0, 355 * 1004.0, 0.0, 30 / 1000)


def _run_resident(config):
    seeds = np.random.SeedSequence(SEED).spawn(config['n_classes'])
    return cascade_loop(
        config['h_profile'], config['qt_profile'], config['z_profile'],
        config['grids'], config['C_h_k'], config['C_qt_k'],
        config['b_h_k'], config['b_qt_k'], config['aspect_k'],
        *BOUNDS, 1, config['sparsity_factors'], seeds,
        n_scale_classes_per_dyad=config['n_per_dyad'],
    )


def _run_streamed(config, tmp_path, horizon, tiles_x, tiles_y):
    seeds = np.random.SeedSequence(SEED).spawn(config['n_classes'])
    s_x, s_y, s_z = config['sparsity_factors']
    kernel = _turbulon_envelope(1, 1 / (2 * s_x), 1 / (2 * s_y), 1 / (2 * s_z),
                                support_factor=SUPPORT_FACTOR)
    plan = plan_tiling(config['grids'], kernel.shape[2],
                       force=(horizon, tiles_x, tiles_y))
    return streamed_cascade_loop(
        config['h_profile'], config['qt_profile'], config['z_profile'],
        config['grids'], config['C_h_k'], config['C_qt_k'],
        config['b_h_k'], config['b_qt_k'], config['aspect_k'],
        *BOUNDS, 1, config['sparsity_factors'], seeds,
        plan, tmp_path / f"scratch_{horizon}_{tiles_x}_{tiles_y}",
        n_scale_classes_per_dyad=config['n_per_dyad'],
    )


def _relative(streamed, resident):
    """Max |difference| relative to the resident field's own scale."""
    difference = np.abs(np.asarray(streamed, dtype=np.float64)
                        - np.asarray(resident, dtype=np.float64))
    scale = float(np.max(np.abs(resident)))
    return float(np.max(difference)) / (scale if scale > 0 else 1.0)


# ---------------------------------------------------------------------------
# The tiled regrid, on its own
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source_shape,target_shape,exact", [
    ((8, 8, 6), (16, 16, 10), True),        # dyadic: the production ladder
    ((2, 2, 4), (4, 4, 5), True),
    ((5, 7, 3), (16, 16, 8), True),
    ((4, 4, 5), (6, 6, 7), False),          # ratio 3/2
    ((6, 10, 4), (15, 15, 9), False),       # 5/2 and 3/2
])
@pytest.mark.parametrize("tiles", [(1, 1), (2, 2), (3, 2), (4, 4)])
def test_tiled_regrid_reproduces_the_full_resample(source_shape, target_shape,
                                                   exact, tiles):
    """A tile's regrid must be the full-domain regrid restricted to that tile.

    BIT-EXACT for a dyadic ladder, which is the production case and the one
    that matters. For a non-dyadic ratio the target-to-source coordinate is
    (j + 0.5) * n_in/n_out - 0.5 evaluated at a different j offset per tile, so
    the last bits of the interpolation weights move; that is a few float32 ULP
    and is the documented floor, not a seam (test_no_seam_structure covers the
    consequence).
    """
    if tiles[0] > target_shape[0] or tiles[1] > target_shape[1]:
        pytest.skip("more tiles than cells")
    source = np.random.default_rng(0).normal(
        0, 1, source_shape).astype(np.float32)
    full = zoom_trilinear(source, target_shape)

    def read_slab(x_lo, x_hi, y_lo, y_hi):
        index_x = np.arange(x_lo, x_hi) % source.shape[0]
        index_y = np.arange(y_lo, y_hi) % source.shape[1]
        return source[np.ix_(index_x, index_y)]

    tiled = np.zeros_like(full)
    for window in tile_windows(target_shape[0], target_shape[1], *tiles):
        tiled[window[0]:window[1], window[2]:window[3]] = regrid_window(
            read_slab, source.shape, target_shape, window)
    if exact:
        np.testing.assert_array_equal(tiled, full)
    else:
        np.testing.assert_allclose(tiled, full, rtol=0, atol=1e-6 *
                                   float(np.max(np.abs(full))))


def test_regrid_window_applies_its_scale_before_interpolating():
    """The flux's per-class rescale is folded into the next class's read, and
    the in-RAM order is rescale-then-regrid. Interpolation is linear, so the
    two orders agree in exact arithmetic but not in float32; this pins the
    order that matches the resident path."""
    source = np.random.default_rng(1).normal(0, 1, (8, 8, 6)).astype(np.float32)
    window = (0, 16, 0, 16)

    def read_slab(x_lo, x_hi, y_lo, y_hi):
        index_x = np.arange(x_lo, x_hi) % source.shape[0]
        index_y = np.arange(y_lo, y_hi) % source.shape[1]
        return source[np.ix_(index_x, index_y)]

    scaled = regrid_window(read_slab, source.shape, (16, 16, 10), window,
                           scale=0.75)
    expected = zoom_trilinear(source * np.float32(0.75), (16, 16, 10))
    np.testing.assert_array_equal(scaled, expected)


def test_axis_regrid_plan_quantizes_to_whole_blocks():
    """b source cells span exactly a target cells, and the slab carries one
    extra block per side for the periodic continuation."""
    # 8 -> 16: g = 8, b = 1, a = 2.
    src_lo, src_hi, crop, out_len = _axis_regrid_plan(8, 16, 4, 12)
    assert (src_lo, src_hi) == (4 // 2 * 1 - 1, -(-12 // 2) * 1 + 1)
    assert out_len == (6 - 2) * 2 + 4
    assert crop == 2 + (4 - 2 * 2)
    # 4 -> 6: g = 2, b = 2, a = 3. A window not on a block boundary must be
    # widened to one and cropped back.
    src_lo, src_hi, crop, out_len = _axis_regrid_plan(4, 6, 1, 5)
    assert src_lo == 0 - 2 and src_hi == 4 + 2
    assert crop == 3 + 1
    with pytest.raises(NotImplementedError, match="downsample"):
        _axis_regrid_plan(16, 8, 0, 8)


def test_tile_bounds_handles_uneven_splits():
    assert tile_bounds(8, 2) == [(0, 4), (4, 8)]
    assert tile_bounds(32, 3) == [(0, 10), (10, 21), (21, 32)]
    assert sum(hi - lo for lo, hi in tile_bounds(37, 4)) == 37
    with pytest.raises(ValueError):
        tile_bounds(4, 8)


def test_tile_windows_are_world_raster_order():
    windows = tile_windows(8, 8, 2, 2)
    assert windows == [(0, 4, 0, 4), (0, 4, 4, 8), (4, 8, 0, 4), (4, 8, 4, 8)]


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

def test_planner_finds_a_horizon_and_tiles_below_it():
    config = _config()
    kernel_nz = _turbulon_envelope(1, 0.5, 0.5, 0.5,
                                   support_factor=SUPPORT_FACTOR).shape[2]
    # A budget that fits the coarse classes and not the finest.
    grids = config['grids']
    from steam.streaming import _working_bytes
    budget = _working_bytes(int(grids['nx'][-2]), int(grids['ny'][-2]),
                            int(grids['nz'][-2]), kernel_nz, 'cpu')
    plan = plan_tiling(grids, kernel_nz, budget_bytes=budget)
    assert plan.horizon == config['n_classes'] - 1
    assert plan.tiles[config['n_classes'] - 1][0] >= 2
    assert plan.peak_scratch_bytes > 0


def test_planner_accepts_a_forced_plan():
    config = _config()
    plan = plan_tiling(config['grids'], 13, force=(2, 2, 2))
    assert plan.horizon == 2
    assert set(plan.tiles) == {2, 3}
    assert all(count == (2, 2) for count in plan.tiles.values())


def test_scratch_check_refuses_with_the_number(tmp_path):
    from steam.streaming import check_scratch_space
    with pytest.raises(OSError, match="GiB of scratch"):
        check_scratch_space(tmp_path, 1 << 60)


def test_scratch_accounting_matches_actual_peak(tmp_path):
    """The planner's predicted peak must bound what the run actually writes --
    a planner that under-predicts is worse than none, because the run dies deep
    instead of refusing up front."""
    config = _config()
    plan = plan_tiling(config['grids'], 13, force=(2, 2, 2))
    result = _run_streamed(config, tmp_path, 2, 2, 2)
    store = result[8]
    try:
        actual = store.total_bytes()
        assert actual <= plan.peak_scratch_bytes, (
            f"actual scratch {actual} exceeded predicted peak "
            f"{plan.peak_scratch_bytes}")
        # And not wildly loose: within 3x, or the number is not informative.
        assert plan.peak_scratch_bytes < 3 * actual
    finally:
        store.destroy()


# ---------------------------------------------------------------------------
# THE fundamental test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def resident():
    config = _config()
    return config, _run_resident(config)


@pytest.mark.parametrize("horizon,tiles_x,tiles_y,label", [
    (2, 1, 1, "1x1 spanning"),
    (2, 2, 2, "2x2"),
    (2, 4, 2, "non-square 4x2"),
    (2, 3, 3, "3x3, does not divide the grid"),
    (2, 8, 8, "8x8, tiles narrower than the halo is wide"),
    (1, 2, 2, "deeper horizon"),
    (0, 2, 2, "everything streamed, no resident classes"),
])
def test_streamed_equals_resident(resident, tmp_path, horizon, tiles_x,
                                  tiles_y, label):
    """Streamed cascade states against in-RAM cascade states, at the floor.

    MEASURED on this configuration (16 x 16 x 10 through 64 x 64 x 10, four
    classes, 2026-08-12): max relative difference 2.6e-06 on h_perturbation,
    5.9e-07 on qt_perturbation, 2.4e-07 on the flux -- and the SAME floor at
    every tiling from 1x1 to 8x8, which is the point: the error does not grow
    with the tile count. float32 eps is 1.2e-07, so h sits at ~20 ULP after
    four classes of a nonlinear multiplicative cascade, which is where it
    should be. The assertion is 1e-5, a factor of four of headroom over the
    worst observed value.
    """
    config, (h_res, qt_res, flux_res, _, _, _, _, ledger_res) = resident
    result = _run_streamed(config, tmp_path, horizon, tiles_x, tiles_y)
    store = result[8]
    try:
        for name, streamed, reference in (
                ('h_perturbation', result[0], h_res),
                ('qt_perturbation', result[1], qt_res),
                ('flux', result[2], flux_res)):
            relative = _relative(streamed, reference)
            assert relative < 1e-5, (
                f"{label}: {name} differs by {relative:.3e} relative, past the "
                f"rounding floor")
    finally:
        store.destroy()


def test_no_seam_structure(resident, tmp_path):
    """THE test of the design: the streamed-minus-resident difference must not
    know where the tile boundaries are.

    Bin |difference| by distance to the nearest tile seam (with wrap, since the
    domain is periodic and so is the tiling) and compare the seam-adjacent bins
    against the interior. A halo that is too narrow, a regrid that clamps at a
    tile edge, or a convolution whose wrap contamination reaches the inner
    region all show up here as enrichment in the first bin or two -- and
    nowhere else.
    """
    config, (h_res, qt_res, flux_res, _, _, _, _, _) = resident
    result = _run_streamed(config, tmp_path, 2, 4, 4)
    store = result[8]
    try:
        nx, ny = h_res.shape[0], h_res.shape[1]
        seams_x = [lo for lo, _ in tile_bounds(nx, 4)]
        seams_y = [lo for lo, _ in tile_bounds(ny, 4)]

        def seam_distance(n, seams):
            index = np.arange(n)
            gaps = [np.minimum((index - seam) % n, (seam - index) % n)
                    for seam in seams]
            return np.min(np.stack(gaps), axis=0)

        distance = np.minimum(seam_distance(nx, seams_x)[:, None],
                              seam_distance(ny, seams_y)[None, :])

        for name, streamed, reference in (
                ('h_perturbation', result[0], h_res),
                ('qt_perturbation', result[1], qt_res),
                ('flux', result[2], flux_res)):
            difference = np.abs(np.asarray(streamed, dtype=np.float64)
                                - np.asarray(reference, dtype=np.float64))
            # Collapse z: a seam is a vertical plane, so every level votes.
            column = difference.mean(axis=2)
            bins = []
            for d in range(0, int(distance.max()) + 1):
                mask = distance == d
                if mask.sum() >= 8:
                    bins.append((d, float(column[mask].mean())))
            assert len(bins) >= 3, f"{name}: too few distance bins to judge"
            interior = np.median([value for _, value in bins[1:]])
            on_seam = bins[0][1]
            if interior == 0.0:
                assert on_seam == 0.0, f"{name}: seam differs, interior exact"
                continue
            # A genuine seam artifact is orders of magnitude, not a factor of
            # two: the failures this guards against (clamped regrid, kernel
            # reaching past the halo, a tile reading a neighbour's
            # already-advanced flux) put the seam bin 10-1000x the interior.
            # MEASURED here at 4x4 tiles: 0.94 (h), 0.92 (qt), 0.87 (flux) --
            # the seam bin is if anything QUIETER than the interior, and the
            # bins are flat to within 10% across the whole distance range.
            assert on_seam / interior < 4.0, (
                f"{name}: seam-adjacent difference is {on_seam / interior:.1f}x "
                f"the interior median -- seam-correlated structure. "
                f"bins={[(d, f'{v:.2e}') for d, v in bins[:6]]}")
    finally:
        store.destroy()


def test_streamed_run_is_deterministic(resident, tmp_path):
    """Two streamed runs of the same configuration must be bit-identical to
    each other: the partial sums are accumulated in world raster order, which
    is fixed, so there is nothing left to vary."""
    config, _ = resident
    first = _run_streamed(config, tmp_path / "a", 2, 3, 2)
    second = _run_streamed(config, tmp_path / "b", 2, 3, 2)
    try:
        for index in range(3):
            np.testing.assert_array_equal(np.asarray(first[index]),
                                          np.asarray(second[index]))
    finally:
        first[8].destroy()
        second[8].destroy()


def test_ledger_agrees_between_the_two_paths(resident, tmp_path):
    """The ledger is the run's realized global scalars, so the two paths must
    agree on it to float64 tolerance -- if they did not, they would not be the
    same cascade whatever the fields looked like."""
    config, (_, _, _, _, _, _, _, ledger_res) = resident
    result = _run_streamed(config, tmp_path, 2, 2, 2)
    store = result[8]
    ledger_streamed = result[7]
    try:
        for index in range(config['n_classes']):
            streamed = ledger_streamed.classes[index]
            reference = ledger_res.classes[index]
            for attribute in ('entering', 'noise_abs', 'realized'):
                a = getattr(streamed.flux, attribute).mean
                b = getattr(reference.flux, attribute).mean
                assert abs(a - b) <= 1e-6 * max(abs(b), 1e-30), (
                    f"class {index} flux {attribute}: {a} vs {b}")
            assert (getattr(streamed.flux, 'noise_abs').count
                    == getattr(reference.flux, 'noise_abs').count)
            for name in ('h', 'qt'):
                np.testing.assert_allclose(
                    streamed.pattern[name].level_mean,
                    reference.pattern[name].level_mean,
                    rtol=1e-5, atol=0,
                    err_msg=f"class {index} {name} product norm")
                np.testing.assert_array_equal(
                    streamed.pattern[name].level_count,
                    reference.pattern[name].level_count,
                    err_msg=f"class {index} {name} turbulon center count")
    finally:
        store.destroy()
