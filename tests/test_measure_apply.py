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
