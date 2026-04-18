"""Regression tests for STEAM simulation outputs.

Runs simulations with various parameter combinations and compares
against saved reference outputs. If reference data doesn't exist,
run:  python tests/generate_regression_data.py
"""

import numpy as np
import pytest
import netCDF4
from pathlib import Path

from steam.simulate import simulate

REFERENCE_DIR = Path(__file__).parent / "regression_data"

# Each case: (name, kwargs_override)
# All use the same base profiles and small grids for speed.
CASES = {
    "baseline": dict(
        nx=16, ny=16, dx=500, dy=500,
        outer_scale=8000, spheroscale=100,
        domain_height=3000, profile_dz=30,
        seed=42,
    ),
    "morlet": dict(
        nx=16, ny=16, dx=500, dy=500,
        outer_scale=8000, spheroscale=100,
        domain_height=3000, profile_dz=30,
        seed=42,
        turbulon_shape="morlet_omega0_6",
    ),
    "oversampled": dict(
        nx=16, ny=16, dx=500, dy=500,
        outer_scale=8000, spheroscale=100,
        domain_height=3000, profile_dz=30,
        seed=42,
        sparsity_factors=(2, 2, 2),
    ),
    "dense_scale_classes": dict(
        nx=16, ny=16, dx=500, dy=500,
        outer_scale=8000, spheroscale=100,
        domain_height=3000, profile_dz=30,
        seed=42,
        n_scale_classes_per_dyad=2,
    ),
    "asymmetric_grid": dict(
        nx=16, ny=20, dx=500, dy=400,
        outer_scale=8000, spheroscale=100,
        domain_height=3000, profile_dz=30,
        seed=42,
    ),
    "min_distance_2": dict(
        nx=16, ny=16, dx=500, dy=500,
        outer_scale=8000, spheroscale=100,
        domain_height=3000, profile_dz=30,
        seed=42,
        min_distance_to_ground=2,
    ),
    "array_spheroscale": dict(
        nx=16, ny=16, dx=500, dy=500,
        outer_scale=8000, spheroscale="profile",  # sentinel — replaced below
        domain_height=3000, profile_dz=30,
        seed=42,
    ),
    "morlet_oversampled": dict(
        nx=16, ny=16, dx=500, dy=500,
        outer_scale=8000, spheroscale=100,
        domain_height=3000, profile_dz=30,
        seed=42,
        turbulon_shape="morlet_omega0_6",
        sparsity_factors=(2, 1, 1),
    ),
    "different_seed": dict(
        nx=16, ny=16, dx=500, dy=500,
        outer_scale=8000, spheroscale=100,
        domain_height=3000, profile_dz=30,
        seed=123,
    ),
    "custom_clamp_bounds": dict(
        nx=16, ny=16, dx=500, dy=500,
        outer_scale=8000, spheroscale=100,
        domain_height=3000, profile_dz=30,
        seed=42,
        h_min=310 * 1004,
        h_max=360 * 1004,
        qt_min=0.0,
        qt_max=35 / 1000,
    ),
}


def _make_profiles(profile_dz, domain_height):
    """Build the canonical regression test profiles."""
    nz = int(domain_height / profile_dz) + 1
    z = np.arange(nz) * profile_dz
    h = 340e3 - 20e3 * (z / domain_height)
    qt = 0.018 - 0.016 * (z / domain_height)
    return h, qt


def _build_kwargs(case_name, case_params, tmp_path):
    """Build full simulate() kwargs, handling special cases."""
    kw = dict(case_params)
    profile_dz = kw["profile_dz"]
    domain_height = kw["domain_height"]
    h, qt = _make_profiles(profile_dz, domain_height)

    # Handle array spheroscale sentinel
    if kw.get("spheroscale") == "profile":
        nz = len(h)
        z = np.arange(nz) * profile_dz
        kw["spheroscale"] = 80 + 40 * (z / domain_height)

    kw["output_path"] = tmp_path / f"{case_name}.nc"
    return h, qt, kw


def _load_reference(case_name):
    """Load saved reference arrays for a case."""
    path = REFERENCE_DIR / f"{case_name}.npz"
    if not path.exists():
        pytest.skip(
            f"Reference data not found: {path}\n"
            f"Run: python tests/generate_regression_data.py"
        )
    return np.load(path)


@pytest.fixture(params=sorted(CASES.keys()))
def case_name(request):
    return request.param


def test_regression_h(case_name, tmp_path):
    """h field must match reference within small tolerance."""
    params = CASES[case_name]
    h_prof, qt_prof, kw = _build_kwargs(case_name, params, tmp_path)
    ref = _load_reference(case_name)

    simulate(h_prof, qt_prof, **kw)
    ds = netCDF4.Dataset(kw["output_path"], "r")
    h_out = ds.variables["h"][:]
    ds.close()

    np.testing.assert_allclose(
        h_out, ref["h"], rtol=1e-5, atol=1e-7,
        err_msg=f"h field changed for case '{case_name}'",
    )


def test_regression_qt(case_name, tmp_path):
    """qt field must match reference within small tolerance."""
    params = CASES[case_name]
    h_prof, qt_prof, kw = _build_kwargs(case_name, params, tmp_path)
    ref = _load_reference(case_name)

    simulate(h_prof, qt_prof, **kw)
    ds = netCDF4.Dataset(kw["output_path"], "r")
    qt_out = ds.variables["qt"][:]
    ds.close()

    np.testing.assert_allclose(
        qt_out, ref["qt"], rtol=1e-5, atol=1e-7,
        err_msg=f"qt field changed for case '{case_name}'",
    )


def test_regression_coordinates(case_name, tmp_path):
    """x, y, z coordinates must match reference within small tolerance."""
    params = CASES[case_name]
    h_prof, qt_prof, kw = _build_kwargs(case_name, params, tmp_path)
    ref = _load_reference(case_name)

    simulate(h_prof, qt_prof, **kw)
    ds = netCDF4.Dataset(kw["output_path"], "r")
    x_out = ds.variables["x"][:]
    y_out = ds.variables["y"][:]
    z_out = ds.variables["z"][:]
    ds.close()

    np.testing.assert_allclose(x_out, ref["x"], rtol=1e-5, atol=1e-7,
                                err_msg=f"x coords changed for '{case_name}'")
    np.testing.assert_allclose(y_out, ref["y"], rtol=1e-5, atol=1e-7,
                                err_msg=f"y coords changed for '{case_name}'")
    np.testing.assert_allclose(z_out, ref["z"], rtol=1e-5, atol=1e-7,
                                err_msg=f"z coords changed for '{case_name}'")


def test_regression_normalization(case_name, tmp_path):
    """C_h_k and C_qt_k normalization arrays must match reference within small tolerance."""
    params = CASES[case_name]
    h_prof, qt_prof, kw = _build_kwargs(case_name, params, tmp_path)
    ref = _load_reference(case_name)

    simulate(h_prof, qt_prof, **kw)
    ds = netCDF4.Dataset(kw["output_path"], "r")
    C_h = ds.variables["C_h_k"][:]
    C_qt = ds.variables["C_qt_k"][:]
    ds.close()

    np.testing.assert_allclose(C_h, ref["C_h_k"], rtol=1e-5, atol=1e-7,
                                err_msg=f"C_h_k changed for '{case_name}'")
    np.testing.assert_allclose(C_qt, ref["C_qt_k"], rtol=1e-5, atol=1e-7,
                                err_msg=f"C_qt_k changed for '{case_name}'")


def test_regression_k_values(case_name, tmp_path):
    """k_values and k_z_values must match reference."""
    params = CASES[case_name]
    h_prof, qt_prof, kw = _build_kwargs(case_name, params, tmp_path)
    ref = _load_reference(case_name)

    simulate(h_prof, qt_prof, **kw)
    ds = netCDF4.Dataset(kw["output_path"], "r")
    k_out = ds.variables["k_values"][:]
    k_z_out = ds.variables["k_z_values"][:]
    ds.close()

    np.testing.assert_array_equal(k_out, ref["k_values"],
                                   err_msg=f"k_values changed for '{case_name}'")
    np.testing.assert_array_equal(k_z_out, ref["k_z_values"],
                                   err_msg=f"k_z_values changed for '{case_name}'")


def test_regression_attributes(case_name, tmp_path):
    """Key scalar attributes must match reference."""
    params = CASES[case_name]
    h_prof, qt_prof, kw = _build_kwargs(case_name, params, tmp_path)
    ref = _load_reference(case_name)

    simulate(h_prof, qt_prof, **kw)
    ds = netCDF4.Dataset(kw["output_path"], "r")
    attrs = {
        "nx": int(ds.nx),
        "ny": int(ds.ny),
        "dx": float(ds.dx),
        "dy": float(ds.dy),
        "C_h_L": float(ds.C_h_L),
        "C_qt_L": float(ds.C_qt_L),
    }
    ds.close()

    for key in ("nx", "ny", "dx", "dy", "C_h_L", "C_qt_L"):
        assert attrs[key] == ref[key], (
            f"Attribute '{key}' changed for '{case_name}': "
            f"{attrs[key]} vs {ref[key]}"
        )
