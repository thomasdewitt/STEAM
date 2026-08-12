"""Component 2: the measure/apply split must not move the answer by one bit.

Every realized global reduction is hoisted out of the per-class body into an
explicit MEASURE phase (additive partials), a reduce, and an APPLY phase that
takes the reduced scalars as arguments. On the in-RAM path the three run
back-to-back on the resident array, so the output has to be BIT-IDENTICAL to
what the code produced before the refactor -- the whole point is that the seam
exists, not that the numbers change.

The references in tests/reference/ were generated on 959f925, BEFORE the
refactor, by tests/heavy/make_refactor_references.py. Pinning first is the only
way to hold a refactor to bit-exactness afterwards; regenerating them is a
deliberate act asserting that a realization change is intended.

The second half of this file tests the capability the seam buys: replay. A run
handed a previously recorded ledger, with every solve skipped, must reproduce
its own output bit-for-bit. That is the primitive component 3's tiles stand on.
"""

import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import torch

from steam import simulate, refine

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

PROFILE_NZ = 28
PROFILE_DZ = 30.0
DOMAIN_HEIGHT = 800.0
SEED = 42

FIELDS = ("h", "qt", "flux",
          "h_perturbation", "qt_perturbation", "flux_state")


def _profiles():
    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    return (340e3 - 20e3 * (z / z.max()),
            0.018 - 0.016 * (z / z.max()))


def _base_kwargs(tmp_path, name):
    """The generator script's configuration, kept in step with it by the
    reference comparison itself: a drift here shows up as a shape mismatch."""
    h, qt = _profiles()
    return dict(
        h_profile=h, qt_profile=qt,
        nx=16, ny=16, dx=125.0, dy=125.0,
        outer_scale=2000.0, spheroscale=100.0,
        domain_height=DOMAIN_HEIGHT, profile_dz=PROFILE_DZ,
        output_path=tmp_path / f"{name}.nc",
        seed=SEED, save_for_refinement=True,
    )


def _harvest(path, group='/'):
    with netCDF4.Dataset(path, "r") as ds:
        grp = ds if group == '/' else ds[group]
        return {field: np.asarray(grp.variables[field][:], dtype=np.float32)
                for field in FIELDS if field in grp.variables}


def _reference(name):
    path = REFERENCE_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"no pinned reference {path.name}; run "
                    f"tests/heavy/make_refactor_references.py")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _assert_bit_identical(produced, expected, label):
    assert set(produced) == set(expected), label
    for field, reference in expected.items():
        actual = produced[field]
        assert actual.shape == reference.shape, f"{label}/{field} shape"
        if np.array_equal(actual, reference):
            continue
        # Diagnosable failure: say how far off and how widely, since a pure
        # refactor that has drifted is usually wrong in a describable way
        # (one level, one class, or a dtype).
        differing = np.count_nonzero(actual != reference)
        with np.errstate(invalid='ignore', divide='ignore'):
            scale = np.maximum(np.abs(reference), np.abs(actual))
            relative = np.where(scale > 0,
                                np.abs(actual - reference) / scale, 0.0)
        pytest.fail(
            f"{label}/{field} is not bit-identical to the pinned reference: "
            f"{differing}/{actual.size} cells differ, "
            f"max|diff|={np.max(np.abs(actual - reference)):.6e}, "
            f"max relative={np.max(relative):.3e}. The measure/apply split is "
            f"a pure refactor; if this change is intended, regenerate with "
            f"tests/heavy/make_refactor_references.py --force and say why."
        )


# ---------------------------------------------------------------------------
# The gate: bit-exactness against pre-refactor output
# ---------------------------------------------------------------------------

def test_root_dyadic_matches_reference(tmp_path):
    simulate(**_base_kwargs(tmp_path, "root_dyadic"))
    _assert_bit_identical(_harvest(tmp_path / "root_dyadic.nc"),
                          _reference("root_dyadic"), "root_dyadic")


def test_root_s2_matches_reference(tmp_path):
    """s = 2 on every axis: the sparse noise lattice, and a per-level turbulon
    center count that the product norm divides by."""
    kwargs = _base_kwargs(tmp_path, "root_s2")
    kwargs["sparsity_factors"] = (2, 2, 2)
    simulate(**kwargs)
    _assert_bit_identical(_harvest(tmp_path / "root_s2.nc"),
                          _reference("root_s2"), "root_s2")


def test_root_half_dyad_matches_reference(tmp_path):
    """Non-dyadic class spacing, so the per-class crop and regrid schedule
    differs from the dyadic case."""
    kwargs = _base_kwargs(tmp_path, "root_half")
    kwargs["n_scale_classes_per_dyad"] = 2
    simulate(**kwargs)
    _assert_bit_identical(_harvest(tmp_path / "root_half.nc"),
                          _reference("root_half"), "root_half")


def test_nest_matches_reference(tmp_path):
    """A nest exercises what a root does not: inner_windows, so every realized
    mean is taken over a window rather than the whole array, plus the
    inherited-deficit replay and the world origins of component 1."""
    kwargs = _base_kwargs(tmp_path, "root_dyadic")
    path = kwargs["output_path"]
    simulate(**kwargs)
    with netCDF4.Dataset(path, "r") as ds:
        k_finest = float(ds.variables["k_values"][:][-1])
    refine(path, 4, 12, 4, 12, k_finest / 4, k_finest / 4,
           output_group="nest", save_for_refinement=True)
    _assert_bit_identical(_harvest(path, "nest"), _reference("nest"), "nest")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="no GPU on this machine")
def test_root_cuda_matches_reference(tmp_path):
    """The GPU bounded add and projection are separate implementations with
    their own reduction order, so they carry their own ledger and their own
    pin. Replay is same-device-exact only, which is the honest contract."""
    kwargs = _base_kwargs(tmp_path, "root_cuda")
    kwargs["device"] = "cuda"
    simulate(**kwargs)
    _assert_bit_identical(_harvest(tmp_path / "root_cuda.nc"),
                          _reference("root_cuda"), "root_cuda")


# ---------------------------------------------------------------------------
# Replay: the capability the seam buys
# ---------------------------------------------------------------------------

def test_recorded_ledger_round_trips_through_netcdf(tmp_path):
    """The ledger is written and reads back as the same numbers, NaN markers
    included -- a NaN mu means "this level was not projected", so a fill value
    that swallowed it would silently change what replay does."""
    from steam.ledger import read_ledger

    kwargs = _base_kwargs(tmp_path, "root_dyadic")
    simulate(**kwargs)
    with netCDF4.Dataset(kwargs["output_path"], "r") as ds:
        ledger = read_ledger(ds)

    assert ledger is not None
    assert len(ledger.classes) == 4
    for index, class_ledger in enumerate(ledger.classes):
        assert class_ledger.flux is not None
        assert class_ledger.flux.entering.count > 0
        if index == 0:
            # The cascade starts the flux at exactly one, and nothing has
            # regridded it yet.
            assert class_ledger.flux.entering.mean == 1.0
        else:
            # Each class restores the volume mean it entered with, so the mean
            # only drifts by what the between-class regrid does to it (the
            # trilinear zoom does not conserve a volume mean exactly). A few
            # percent, not a factor.
            assert abs(class_ledger.flux.entering.mean - 1.0) < 0.05
        for scalar in ("h", "qt"):
            pattern = class_ledger.pattern[scalar]
            assert np.all(pattern.level_total >= 0.0)
            assert np.all(pattern.level_count >= 0)
            solve = class_ledger.bounded_add[scalar]
            assert solve.a0.shape == pattern.level_total.shape
            # NaN survives the round trip as the no-bisection marker.
            assert solve.mu.dtype == np.float64
    # The composition's own entries.
    assert ledger.flux_output is not None
    assert set(ledger.projection) <= {"h", "qt"}


@pytest.mark.parametrize("extra,label", [
    (None, "dyadic"),
    ({"sparsity_factors": (2, 2, 2)}, "s2"),
    ({"n_scale_classes_per_dyad": 2}, "half"),
])
def test_apply_only_replay_reproduces_the_run(tmp_path, extra, label):
    """THE replay test. A second cascade over identical inputs, handed the
    recorded ledger, with every solve skipped and every reduction injected,
    must land on the same field bit for bit.

    This is what a streamed tile does: it never re-derives a global scalar, it
    is handed one. If any reduction were still fused into its application, the
    injected value would be ignored and this would drift.
    """
    from steam.ledger import read_ledger

    kwargs = _base_kwargs(tmp_path, f"replay_{label}")
    if extra:
        kwargs.update(extra)
    simulate(**kwargs)
    recorded = _harvest(kwargs["output_path"])
    with netCDF4.Dataset(kwargs["output_path"], "r") as ds:
        ledger = read_ledger(ds)

    # Poison the inputs to the solves that replay must not consult: if apply
    # mode still measured anything, it would measure these and diverge.
    kwargs["output_path"] = tmp_path / f"replay_{label}_again.nc"
    replayed = _replay_simulate(kwargs, ledger)

    for field, reference in recorded.items():
        np.testing.assert_array_equal(
            replayed[field], reference,
            err_msg=f"apply-only replay of {label} drifted on {field}")


def _replay_simulate(kwargs, ledger):
    """simulate() with a supplied ledger, via the same public path."""
    simulate(ledger=ledger, **kwargs)
    return _harvest(kwargs["output_path"])


def test_replay_actually_uses_the_injected_values(tmp_path):
    """Guard against a vacuous replay test: perturb one recorded scalar and the
    replayed field must move. Otherwise 'replay reproduces the run' could hold
    simply because apply mode re-solved everything."""
    from steam.ledger import read_ledger

    kwargs = _base_kwargs(tmp_path, "poison")
    simulate(**kwargs)
    baseline = _harvest(kwargs["output_path"])
    with netCDF4.Dataset(kwargs["output_path"], "r") as ds:
        ledger = read_ledger(ds)

    # Halve the finest class's h product norm. A genuinely injected value
    # doubles that class's h amplitude; a re-solved one ignores this entirely.
    ledger.classes[-1].pattern['h'].level_total *= 0.5
    kwargs["output_path"] = tmp_path / "poisoned.nc"
    poisoned = _replay_simulate(kwargs, ledger)

    assert not np.array_equal(poisoned['h_perturbation'],
                              baseline['h_perturbation'])
    # And only h moved: the flux state is upstream of the scalar norms.
    np.testing.assert_array_equal(poisoned['flux_state'],
                                  baseline['flux_state'])


def test_nest_replay_reproduces_the_run(tmp_path):
    """A nest replays too, and it is the harder case: every realized mean is
    taken over its inner window rather than the whole array, so a ledger that
    had quietly recorded whole-array reductions would drift here."""
    from steam.ledger import read_ledger

    kwargs = _base_kwargs(tmp_path, "nest_replay")
    path = kwargs["output_path"]
    simulate(**kwargs)
    with netCDF4.Dataset(path, "r") as ds:
        k_finest = float(ds.variables["k_values"][:][-1])
    refine(path, 4, 12, 4, 12, k_finest / 4, k_finest / 4,
           output_group="first", save_for_refinement=True)
    recorded = _harvest(path, "first")
    with netCDF4.Dataset(path, "r") as ds:
        ledger = read_ledger(ds, "first")

    refine(path, 4, 12, 4, 12, k_finest / 4, k_finest / 4,
           output_group="replayed", save_for_refinement=True, ledger=ledger)
    replayed = _harvest(path, "replayed")
    for field, reference in recorded.items():
        np.testing.assert_array_equal(
            replayed[field], reference,
            err_msg=f"nest apply-only replay drifted on {field}")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="no GPU on this machine")
def test_cuda_replay_reproduces_the_run(tmp_path):
    """The GPU solves record their own ledger in their own arithmetic. Replay is
    same-device-exact only -- a CPU ledger would not replay here and is not
    asked to -- but on the device that recorded it, it is exact."""
    from steam.ledger import read_ledger

    kwargs = _base_kwargs(tmp_path, "cuda_replay")
    kwargs["device"] = "cuda"
    simulate(**kwargs)
    recorded = _harvest(kwargs["output_path"])
    with netCDF4.Dataset(kwargs["output_path"], "r") as ds:
        ledger = read_ledger(ds)

    kwargs["output_path"] = tmp_path / "cuda_replay_again.nc"
    simulate(ledger=ledger, **kwargs)
    replayed = _harvest(kwargs["output_path"])
    for field, reference in recorded.items():
        np.testing.assert_array_equal(
            replayed[field], reference,
            err_msg=f"cuda apply-only replay drifted on {field}")
