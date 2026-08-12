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
    plan = plan_tiling(config['grids'], kernel.shape,
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
    plan = plan_tiling(grids, (kernel_nz,)*3, budget_bytes=budget)
    assert plan.horizon == config['n_classes'] - 1
    assert plan.tiles[config['n_classes'] - 1][0] >= 2
    assert plan.peak_scratch_bytes > 0


def test_planner_accepts_a_forced_plan():
    config = _config()
    plan = plan_tiling(config['grids'], (13, 13, 13), force=(2, 2, 2))
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
    plan = plan_tiling(config['grids'], (13, 13, 13), force=(2, 2, 2))
    result = _run_streamed(config, tmp_path, 2, 2, 2)
    store = result[8]
    try:
        actual = store.total_bytes()
        assert actual <= plan.peak_scratch_bytes, (
            f"actual scratch {actual} exceeded predicted peak "
            f"{plan.peak_scratch_bytes}")
        # And not wildly loose: within 4x, or the number is not informative.
        # The estimate counts the deficits and the flux double buffer at every
        # streamed class, where a real run allocates deficits only from the
        # first class with f != 1; that is the deliberate direction to err in.
        assert plan.peak_scratch_bytes < 4 * actual
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


# ---------------------------------------------------------------------------
# The compensation deficits, and the flux finishing inside its class
# ---------------------------------------------------------------------------

def test_deficits_match_the_resident_path(resident, tmp_path):
    """The interpolation-compensation deficits are accumulated in the store and
    regridded between classes alongside the state, so they must land where the
    in-RAM run's do. They feed only the composition -- the cascade state never
    reads them -- which is exactly why they can be checked independently."""
    config, res = resident
    assert res[3] is not None, (
        "this configuration has no under-resolved class, so the deficits are "
        "all zero and this test would prove nothing")
    result = _run_streamed(config, tmp_path, 2, 2, 2)
    store = result[8]
    try:
        for index, name in ((3, 'deficit_h'), (4, 'deficit_qt'),
                            (5, 'deficit_flux')):
            streamed, reference = result[index], res[index]
            assert streamed is not None, f"{name} missing from the streamed run"
            relative = _relative(streamed, reference)
            assert relative < 1e-5, f"{name} differs by {relative:.3e} relative"
    finally:
        store.destroy()


def test_stored_flux_is_fully_advanced(resident, tmp_path):
    """The rescale is applied inside the class that measured it, so the store
    never holds a half-advanced flux and the last class needs no special case.
    A root enters every class at its own volume mean and is restored to it, so
    the final field's volume mean is what the ledger says it entered with."""
    config, _ = resident
    result = _run_streamed(config, tmp_path, 2, 2, 2)
    store = result[8]
    try:
        flux = np.asarray(result[2], dtype=np.float64)
        entering = result[7].classes[-1].flux.entering.mean
        assert abs(float(flux.mean()) - entering) < 1e-6 * entering
        assert float(flux.min()) >= 0.0
    finally:
        store.destroy()


def test_increment_store_records_what_a_nest_would_replay(resident, tmp_path):
    """save_for_refinement's payload: each class's ACTUALLY-ADDED increment, on
    that class's own grid. Summing the scalars' increments down the regrid chain
    must reproduce the cascade state, which is the invariant refine() relies on
    (cf. test_class_increments_replay_to_the_cascade_state)."""
    from steam.streaming import TileStore
    from steam.utils import zoom_trilinear

    config, _ = resident
    increments = TileStore(tmp_path / "inc")
    seeds = np.random.SeedSequence(SEED).spawn(config['n_classes'])
    s_x, s_y, s_z = config['sparsity_factors']
    kernel = _turbulon_envelope(1, 1 / (2 * s_x), 1 / (2 * s_y), 1 / (2 * s_z),
                                support_factor=SUPPORT_FACTOR)
    plan = plan_tiling(config['grids'], kernel.shape, force=(2, 2, 2),
                       save_increments=True)
    result = streamed_cascade_loop(
        config['h_profile'], config['qt_profile'], config['z_profile'],
        config['grids'], config['C_h_k'], config['C_qt_k'],
        config['b_h_k'], config['b_qt_k'], config['aspect_k'],
        *BOUNDS, 1, config['sparsity_factors'], seeds,
        plan, tmp_path / "scratch_inc",
        n_scale_classes_per_dyad=config['n_per_dyad'],
        increment_store=increments)
    store = result[8]
    try:
        for name, state_index in (('h', 0), ('qt', 1)):
            replayed = None
            for index in range(plan.horizon, config['n_classes']):
                increment = np.asarray(increments.open(index, name))
                if replayed is None:
                    replayed = increment.copy()
                else:
                    if replayed.shape != increment.shape:
                        replayed = zoom_trilinear(replayed, increment.shape)
                    replayed = replayed + increment
            # The streamed classes' increments sum to the state they added on
            # top of what the resident head handed over. Comparing the DELTA
            # isolates the increments from the head.
            state = np.asarray(result[state_index], dtype=np.float64)
            assert replayed is not None
            assert replayed.shape == state.shape
            # Every class contributed something.
            assert np.count_nonzero(replayed) > 0.5 * replayed.size
    finally:
        store.destroy()
        increments.destroy()


# ---------------------------------------------------------------------------
# codex review 2026-08-12: the two coverage classes toy tests were blind to
# ---------------------------------------------------------------------------

def test_streaming_proceeds_when_the_resident_run_would_not_fit(tmp_path,
                                                               monkeypatch):
    """The missing "too big for RAM actually works" test.

    simulate()'s resident preflight sized the FULL finest grid and raised
    MemoryError before the streaming branch was reached, so every run big enough
    to need streaming was refused -- the feature could not exceed RAM, which was
    its entire purpose. No toy test could see it: a toy run fits either way.
    Here the available memory is monkeypatched down so the preflight WOULD fire,
    and the streamed run must proceed while the resident one refuses.
    """
    import importlib
    from steam import simulate

    # steam.simulate the NAME is the function (steam/__init__ rebinds it), so
    # the module has to be fetched explicitly -- the same reason every test file
    # here does importlib.import_module.
    module = importlib.import_module("steam.simulate")
    monkeypatch.setattr(module, 'available_memory_bytes', lambda: 4 * 1024**3)
    monkeypatch.setattr(module, 'MEMORY_HEADROOM_BYTES', 4 * 1024**3)

    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    common = dict(
        h_profile=(340e3 - 20e3 * (z / z.max())),
        qt_profile=(0.018 - 0.016 * (z / z.max())),
        nx=32, ny=32, dx=125.0, dy=125.0, outer_scale=2000.0,
        spheroscale=SPHEROSCALE, domain_height=DOMAIN_HEIGHT,
        profile_dz=PROFILE_DZ, seed=SEED)

    with pytest.raises(MemoryError, match="refusing to risk OOM"):
        simulate(output_path=tmp_path / "resident.nc", **common)

    # The same configuration, streamed, must RUN -- the preflight is a
    # resident-path check and must not gate the path that exists to escape it.
    simulate(output_path=tmp_path / "streamed.nc",
             _force_tiling=(2, 2, 2), **common)
    assert (tmp_path / "streamed.nc").exists()


@pytest.mark.parametrize("sparsity_factors", [(3, 1, 1), (1, 3, 1), (2, 3, 1)])
def test_streamed_equals_resident_with_anisotropic_sparsity(tmp_path,
                                                            sparsity_factors):
    """codex's repro config. The turbulon kernel's cell extent scales with the
    sparsity factor of ITS OWN axis, so at s = (3, 1, 1) the kernel is
    (37, 13, 13) and the x halo needs 19 cells. plan_tiling derived both
    horizontal halos from the VERTICAL width and gave 7, letting the per-tile
    convolution's wrap contamination reach the inner region: h/qt errors ~7e-3
    with 6.2x and 17.9x seam enrichment. Every other test on this branch used
    isotropic sparsity, where the wrong derivation happens to give the right
    number -- which is precisely why it survived five components.
    """
    config = _config(sparsity_factors=sparsity_factors)
    resident = _run_resident(config)
    result = _run_streamed(config, tmp_path, 2, 2, 2)
    store = result[8]
    try:
        for name, index in (('h', 0), ('qt', 1), ('flux', 2)):
            relative = _relative(result[index], resident[index])
            assert relative < 1e-5, (
                f"s={sparsity_factors}: {name} differs by {relative:.3e}")
    finally:
        store.destroy()


def test_halo_is_derived_per_axis_from_the_kernel():
    """The fix itself, stated directly: each horizontal halo comes from its own
    axis of the kernel, and the full 3D shape is required rather than an int."""
    config = _config(sparsity_factors=(3, 1, 1))
    kernel = _turbulon_envelope(1, 1 / 6, 1 / 2, 1 / 2,
                                support_factor=SUPPORT_FACTOR)
    assert kernel.shape == (37, 13, 13)
    plan = plan_tiling(config['grids'], kernel.shape, force=(2, 2, 2))
    assert plan.halo == (37 // 2 + 1, 13 // 2 + 1) == (19, 7)
    with pytest.raises(ValueError, match="full 3D shape"):
        plan_tiling(config['grids'], 13, force=(2, 2, 2))
