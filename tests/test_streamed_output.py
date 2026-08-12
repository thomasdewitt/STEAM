"""End-to-end gate: a streamed run's OUTPUT FILE against the resident one.

Component 3 compared cascade states; this compares the finished product --
every written variable, the diagnostics computed from them, the ledger group,
and finally a nest refined from a streamed parent. That last one closes the loop
on the whole design: if a streamed run is a valid refinement parent, the streamed
path is not a separate thing that happens to look similar, it is the cascade.

Both runs go through simulate() itself, so the non-realization contents of the
two files are built by the same code and any difference is in the data.
"""

import numpy as np
import pytest
import netCDF4

from steam import simulate, refine
from steam.ledger import read_ledger
from steam.thermodynamics import compute_diagnostics


PROFILE_NZ = 28
PROFILE_DZ = 30.0
DOMAIN_HEIGHT = 800.0
SEED = 42

# 32 x 32 x ~10 at the finest class over four classes: small enough for a test
# suite, large enough that a 2x2 tiling has real interior and real seams.
GRID = dict(nx=32, ny=32, dx=125.0, dy=125.0, outer_scale=2000.0)

SCALARS = ('h', 'qt', 'flux')
STATES = ('h_perturbation', 'qt_perturbation', 'flux_state')
DIAGNOSTICS = ('T', 'p', 'qv', 'qc', 'qi')


def _profiles():
    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    return (340e3 - 20e3 * (z / z.max()), 0.018 - 0.016 * (z / z.max()))


def _kwargs(tmp_path, name, **extra):
    h, qt = _profiles()
    kwargs = dict(
        h_profile=h, qt_profile=qt, spheroscale=100.0,
        domain_height=DOMAIN_HEIGHT, profile_dz=PROFILE_DZ,
        output_path=tmp_path / f"{name}.nc", seed=SEED, **GRID)
    kwargs.update(extra)
    return kwargs


def _read(path, names, group='/'):
    with netCDF4.Dataset(path, "r") as ds:
        grp = ds if group == '/' else ds[group]
        return {name: np.asarray(grp.variables[name][:], dtype=np.float64)
                for name in names if name in grp.variables}


def _relative(streamed, resident):
    scale = float(np.max(np.abs(resident)))
    return float(np.max(np.abs(streamed - resident))) / (scale if scale else 1.0)


@pytest.fixture(scope="module")
def streamed_4x4(tmp_path_factory, pair):
    """A second streamed run at 4x4 tiles, for the seam statistic.

    The seam test wants as many seam lines as possible: at 2x2 there are two per
    axis and the seam-adjacent bin holds 124 cells, enough for the ratio to
    wander on sampling noise alone (measured 1.58 on h at 2x2 against 0.93 at
    4x4, with non-monotonic bins either way). 4x4 doubles the sample and is the
    number worth asserting on.
    """
    directory = tmp_path_factory.mktemp("streamed_4x4")
    kwargs = _kwargs(directory, "streamed4", _force_tiling=(2, 4, 4))
    simulate(**kwargs)
    return kwargs["output_path"]


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """One resident run and one streamed run of the same configuration."""
    directory = tmp_path_factory.mktemp("streamed_output")
    resident = _kwargs(directory, "resident", save_for_refinement=True)
    simulate(**resident)
    streamed = _kwargs(directory, "streamed", save_for_refinement=True,
                       _force_tiling=(2, 2, 2))
    simulate(**streamed)
    compute_diagnostics(resident["output_path"])
    compute_diagnostics(streamed["output_path"])
    return resident["output_path"], streamed["output_path"]


# ---------------------------------------------------------------------------
# (1) Every written variable, at the floor
# ---------------------------------------------------------------------------

def test_written_fields_agree(pair):
    """The composed output and the stored cascade states, streamed vs resident.

    MEASURED on this configuration (2026-08-12, 2x2 tiles, horizon 2):

        h 9.1e-08   qt 1.8e-07   flux 2.6e-07
        h_perturbation 1.0e-06   qt_perturbation 3.6e-07   flux_state 1.6e-07

    The composed fields come out TIGHTER than the states they are built from,
    which is arithmetic rather than luck: h is ~3.4e5 while its perturbation is
    ~1e3, so the same absolute difference is a smaller fraction of the composed
    field's scale. The assertion is 1e-5 against a worst observed 1.0e-06.
    """
    resident_path, streamed_path = pair
    resident = _read(resident_path, SCALARS + STATES)
    streamed = _read(streamed_path, SCALARS + STATES)
    assert set(streamed) == set(resident)
    report = {}
    for name in resident:
        report[name] = _relative(streamed[name], resident[name])
    worst = max(report.values())
    assert worst < 1e-5, (
        "streamed output past the rounding floor: "
        + ", ".join(f"{k}={v:.2e}" for k, v in sorted(report.items())))


def test_file_structure_is_identical(pair):
    """Same variables, same dimensions, same attributes -- everything that is
    not the realization. The two files are built by the same code after the
    cascade, so a difference here means the streamed branch skipped something."""
    resident_path, streamed_path = pair
    with netCDF4.Dataset(resident_path) as a, netCDF4.Dataset(streamed_path) as b:
        assert set(a.variables) == set(b.variables)
        assert set(a.groups) == set(b.groups)
        assert {d: len(v) for d, v in a.dimensions.items()} == \
               {d: len(v) for d, v in b.dimensions.items()}
        for attribute in a.ncattrs():
            first, second = a.getncattr(attribute), b.getncattr(attribute)
            if isinstance(first, (str, bytes)):
                assert first == second, attribute
            else:
                np.testing.assert_allclose(
                    np.asarray(first, dtype=np.float64),
                    np.asarray(second, dtype=np.float64),
                    rtol=1e-12, err_msg=attribute)
        for name in ("x", "y", "z", "k_values", "h_profile", "qt_profile"):
            np.testing.assert_allclose(a.variables[name][:],
                                       b.variables[name][:], rtol=1e-12)


# ---------------------------------------------------------------------------
# (2) The seam test, on the COMPOSED fields
# ---------------------------------------------------------------------------

def test_no_seam_structure_in_composed_fields(pair, streamed_4x4):
    """The design's own test, applied to what actually gets written.

    The composition adds a per-level projection on top of the cascade, so a seam
    could in principle be introduced here and nowhere else.

    MEASURED at 4x4 tiles (2026-08-12): h 0.93, qt 1.04 -- both at unity, bins
    flat and non-monotonic across the distance range. At 2x2 the same
    measurement gives 1.58 on h, which is sampling noise on a 124-cell bin, not
    structure; see the streamed_4x4 fixture.
    """
    from steam.streaming import tile_bounds

    resident_path, _ = pair
    streamed_path = streamed_4x4
    resident = _read(resident_path, ('h', 'qt'))
    streamed = _read(streamed_path, ('h', 'qt'))
    nx, ny = resident['h'].shape[0], resident['h'].shape[1]
    seams_x = [lo for lo, _ in tile_bounds(nx, 4)]
    seams_y = [lo for lo, _ in tile_bounds(ny, 4)]

    def seam_distance(n, seams):
        index = np.arange(n)
        return np.min(np.stack([np.minimum((index - s) % n, (s - index) % n)
                                for s in seams]), axis=0)

    distance = np.minimum(seam_distance(nx, seams_x)[:, None],
                          seam_distance(ny, seams_y)[None, :])
    for name in ('h', 'qt'):
        difference = np.abs(streamed[name] - resident[name]).mean(axis=2)
        bins = [(d, float(difference[distance == d].mean()))
                for d in range(int(distance.max()) + 1)
                if (distance == d).sum() >= 8]
        assert len(bins) >= 3
        interior = np.median([value for _, value in bins[1:]])
        if interior == 0.0:
            assert bins[0][1] == 0.0
            continue
        assert bins[0][1] / interior < 4.0, (
            f"{name}: seam bin is {bins[0][1] / interior:.1f}x the interior "
            f"median. bins={[(d, f'{v:.2e}') for d, v in bins[:6]]}")


# ---------------------------------------------------------------------------
# (3) Diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_agree(pair):
    """compute_diagnostics already walks a written file in x-chunks, so the
    streamed path needs nothing new from it -- write h/qt/flux, then call it.
    Asserted rather than assumed, because the saturation adjustment's Newton
    iterations can amplify a float32 difference near a phase boundary: a cell
    sitting exactly at saturation can land on either side of the branch.

    MEASURED (2026-08-12): T 2.0e-07, p 7.7e-08, qv 5.3e-07, qc 3.1e-06,
    qi exactly 0 -- and that last row is VACUOUS: an 800 m toy column never
    freezes, so qi is identically zero in both files. It says nothing about the
    ice path. Not worth an ice-bearing configuration of its own, since the
    partition is a pointwise linear ramp and the liquid path exercises the
    mechanism. No amplification at all on this configuration -- and
    test_condensate_differences_are_not_widespread puts the fraction of cells
    past 1e-5 of the field scale at 0.0000% for all three condensate variables.
    If a configuration ever does amplify, that is where it will show.
    """
    resident_path, streamed_path = pair
    resident = _read(resident_path, DIAGNOSTICS)
    streamed = _read(streamed_path, DIAGNOSTICS)
    assert set(streamed) == set(resident) and len(resident) == len(DIAGNOSTICS)
    report = {name: _relative(streamed[name], resident[name])
              for name in resident}
    # T and p are smooth functions of h, qt; the condensate partition is not,
    # so it gets its own (still tight) allowance and the number is recorded.
    for name in ('T', 'p'):
        assert report[name] < 1e-5, f"{name}={report[name]:.2e}"
    for name in ('qv', 'qc', 'qi'):
        assert report[name] < 1e-3, (
            f"{name}={report[name]:.2e}; if this is a localized phase-boundary "
            f"amplification, characterize it rather than loosening further. "
            f"full report: " + ", ".join(f"{k}={v:.2e}"
                                         for k, v in sorted(report.items())))


def test_condensate_differences_are_not_widespread(pair):
    """If the saturation adjustment amplifies anywhere, it must be at isolated
    cells near a phase boundary rather than across the field -- a broad
    disagreement would mean the composition is wrong, not that Newton branched.
    """
    resident_path, streamed_path = pair
    resident = _read(resident_path, ('qc', 'qi', 'qv'))
    streamed = _read(streamed_path, ('qc', 'qi', 'qv'))
    for name in ('qc', 'qi', 'qv'):
        scale = float(np.max(np.abs(resident[name])))
        if scale == 0.0:
            continue
        loud = np.abs(streamed[name] - resident[name]) > 1e-5 * scale
        fraction = float(loud.sum()) / loud.size
        assert fraction < 1e-3, (
            f"{name}: {fraction:.2%} of cells differ by more than 1e-5 of the "
            f"field scale -- too widespread to be phase-boundary branching")


# ---------------------------------------------------------------------------
# (4) The ledger group
# ---------------------------------------------------------------------------

def test_streamed_file_carries_a_round_tripping_ledger(pair):
    """The streamed file's ledger group must read back, and must agree with the
    resident run's: it is the run's realized global scalars, so disagreement
    would mean the two are not the same cascade whatever the fields look like."""
    resident_path, streamed_path = pair
    with netCDF4.Dataset(resident_path) as ds:
        resident = read_ledger(ds)
    with netCDF4.Dataset(streamed_path) as ds:
        streamed = read_ledger(ds)
    assert streamed is not None
    assert len(streamed.classes) == len(resident.classes)
    for index, (a, b) in enumerate(zip(streamed.classes, resident.classes)):
        for attribute in ('entering', 'noise_abs', 'realized'):
            first = getattr(a.flux, attribute).mean
            second = getattr(b.flux, attribute).mean
            assert abs(first - second) <= 1e-6 * max(abs(second), 1e-30), (
                f"class {index} flux {attribute}")
        for name in ('h', 'qt'):
            np.testing.assert_allclose(a.pattern[name].level_mean,
                                       b.pattern[name].level_mean,
                                       rtol=1e-5,
                                       err_msg=f"class {index} {name} norm")
    # The composition's own entries survived the streamed path.
    assert set(streamed.projection) == set(resident.projection)
    for name in streamed.projection:
        first = streamed.projection[name].mu
        second = resident.projection[name].mu
        np.testing.assert_array_equal(np.isnan(first), np.isnan(second),
                                      err_msg=f"{name} projected levels differ")
        finite = ~np.isnan(first)
        if finite.any():
            np.testing.assert_allclose(first[finite], second[finite],
                                       rtol=1e-4, err_msg=f"{name} mu")


# ---------------------------------------------------------------------------
# (5) A streamed run is a valid refinement parent
# ---------------------------------------------------------------------------

def test_refine_from_a_streamed_parent(pair):
    """The loop closes here. A nest refined from a streamed parent must match a
    nest refined from the matched resident parent: same subregion, same seed
    stream, same world-keyed noise. This exercises save_for_refinement in
    streamed mode (the stored states AND the class_increments the nest replays
    to compose its own output), the world anchoring of component 1, and the
    composition of component 4, all at once.
    """
    resident_path, streamed_path = pair
    with netCDF4.Dataset(resident_path) as ds:
        k_finest = float(ds.variables["k_values"][:][-1])
        assert "class_increments" in ds.groups
    with netCDF4.Dataset(streamed_path) as ds:
        assert "class_increments" in ds.groups, (
            "streamed run did not write class_increments, so it cannot be a "
            "refinement parent")

    new_dx = k_finest / 4
    refine(resident_path, 8, 24, 8, 24, new_dx, new_dx, output_group="nest")
    refine(streamed_path, 8, 24, 8, 24, new_dx, new_dx, output_group="nest")

    resident = _read(resident_path, SCALARS, group="nest")
    streamed = _read(streamed_path, SCALARS, group="nest")
    report = {name: _relative(streamed[name], resident[name])
              for name in resident}
    worst = max(report.values())
    # MEASURED (2026-08-12): h 9.1e-08, qt 1.8e-07, flux 3.5e-07.
    assert worst < 1e-4, (
        "a nest of the streamed parent differs from a nest of the resident "
        "parent past the floor: "
        + ", ".join(f"{k}={v:.2e}" for k, v in sorted(report.items())))
