"""Sanity check tests for STEAM."""

import warnings

import numpy as np
import pytest
import netCDF4
import torch
from pathlib import Path

from steam.simulate import (
    simulate,
    _compute_all_grids,
    _turbulon_envelope,
    _keyed_sparse_levy,
    check_regrid_ladder,
    NoiseRegion,
    _gradient_components,
)
from steam.utils import (
    fold_kernel_to_field,
    convolve_periodic_xy_zeropad_z_oa,
)
from steam.thermodynamics import recover_diagnostics, compute_diagnostics
from steam.constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
    hurst_vertical_anisotropy as H_z,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_profiles():
    nz = 50
    z = np.arange(nz) * 30.0
    h = 340e3 - 20e3 * (z / z.max())
    qt = 0.018 - 0.016 * (z / z.max())
    return h, qt


@pytest.fixture
def small_nc(tmp_path, simple_profiles):
    """Run a tiny simulate() and return the NC path."""
    h, qt = simple_profiles
    out = tmp_path / "test.nc"
    simulate(h, qt, nx=16, ny=16, dx=500, dy=500,
             outer_scale=8000, spheroscale=100,
             domain_height=3000, profile_dz=30,
             output_path=out, seed=42)
    return out


# ===================================================================
# simulate.py — Input validation
# ===================================================================

def test_rejects_outer_scale_less_than_dx(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="outer_scale"):
        simulate(h, qt, 16, 16, 500, 500, 100, 100, 3000, 30, tmp_path / "x.nc")


def test_allows_dy_not_aligned_with_size_classes(tmp_path, simple_profiles):
    h, qt = simple_profiles
    out = tmp_path / "unaligned_dy.nc"
    simulate(
        h, qt,
        nx=16, ny=20,
        dx=500, dy=400,
        outer_scale=8000, spheroscale=100,
        domain_height=3000, profile_dz=30,
        output_path=out, seed=42,
    )
    assert out.exists()


def test_rejects_mismatched_profile_lengths(tmp_path):
    with pytest.raises(ValueError, match="equal nonzero length"):
        simulate(np.ones(10), np.ones(5), 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc")


def test_rejects_empty_profiles(tmp_path):
    with pytest.raises(ValueError, match="equal nonzero length"):
        simulate(np.array([]), np.array([]), 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc")


def test_rejects_2d_profiles(tmp_path):
    with pytest.raises(ValueError, match="1D"):
        simulate(np.ones((5, 2)), np.ones((5, 2)), 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc")


def test_rejects_zero_domain_height(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="domain_height"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 0, 30, tmp_path / "x.nc")


def test_rejects_negative_domain_height(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="domain_height"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, -100, 30, tmp_path / "x.nc")


def test_rejects_domain_shorter_than_vertical_outer_scale(tmp_path, simple_profiles):
    h, qt = simple_profiles
    # outer_scale=8000, spheroscale=100 => k_z_L ~ 100*(8000/100)^(5/9) ~ 1467 m
    # domain_height=100 < k_z_L => n_large_turbulons=0
    with pytest.raises(ValueError, match="shorter than the vertical"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 100, 30, tmp_path / "x.nc")


def test_domain_height_gate_uses_max_over_spheroscale_profile(tmp_path, simple_profiles):
    """The gate is the MAX of k_z,L(z), not the profile-mean (2026-07-28).

    Spheroscale 400 m near the surface gives k_z,L ~ 2114 m there, taller
    than the 1500 m domain, even though the profile-mean spheroscale
    (~106 m) implies k_z,L ~ 1500 m and would squeak past a mean-based
    check.
    """
    h, qt = simple_profiles
    ls = np.full(len(h), 100.0)
    ls[:2] = 400.0
    with pytest.raises(ValueError, match="shorter than the vertical"):
        simulate(h, qt, 16, 16, 500, 500, 8000, ls, 1500, 30, tmp_path / "x.nc")


def test_rejects_profile_dz_too_coarse(tmp_path):
    # spheroscale=100, outer_scale=8000 => k_z_L ~ 1467 m
    # profile_dz=2000 >= k_z_L => should raise
    nz = 3
    h = np.linspace(340e3, 320e3, nz)
    qt = np.linspace(0.018, 0.002, nz)
    with pytest.raises(ValueError, match="profile_dz"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 4000, 2000, tmp_path / "x.nc")


def test_rejects_zero_sparsity(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="positive integer"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc",
                 sparsity_factors=(0, 1, 1))


def test_rejects_invalid_n_scale_classes_per_dyad(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="n_scale_classes_per_dyad"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc",
                 n_scale_classes_per_dyad=0)
    with pytest.raises(ValueError, match="n_scale_classes_per_dyad"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc",
                 n_scale_classes_per_dyad=1.5)


def test_simulate_runs_with_two_scale_classes_per_dyad(tmp_path, simple_profiles):
    # n_scale_classes_per_dyad is the single cascade-density knob: it sets
    # both the sqrt(2) scalar class spacing and the per-class flux scale.
    h, qt = simple_profiles
    out = tmp_path / "half_octave.nc"
    simulate(h, qt, nx=16, ny=16, dx=500, dy=500,
             outer_scale=8000, spheroscale=100,
             domain_height=3000, profile_dz=30,
             output_path=out, seed=42,
             n_scale_classes_per_dyad=2)
    assert out.exists()
    with netCDF4.Dataset(out) as ds:
        assert int(ds.n_scale_classes_per_dyad) == 2
        k = ds.variables["k_values"][:]
        np.testing.assert_allclose(k[:-1] / k[1:], np.sqrt(2.0), rtol=1e-6)


def test_rejects_float_sparsity(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="positive integer"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc",
                 sparsity_factors=(1.5, 1, 1))


def test_rejects_h_min_above_profile_min(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="h_min"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc",
                 h_min=float(h.min()) + 1.0)


def test_rejects_h_max_below_profile_max(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="h_max"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc",
                 h_max=float(h.max()) - 1.0)


def test_rejects_qt_min_above_profile_min(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="qt_min"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc",
                 qt_min=float(qt.min()) + 1e-6)


def test_rejects_qt_max_below_profile_max(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="qt_max"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc",
                 qt_max=float(qt.max()) - 1e-6)


def test_compute_all_grids_regression_for_dyadic_scale_classes():
    k_values = np.array([8.0, 4.0, 2.0])
    grids = _compute_all_grids(
        k_values=k_values,
        inner_extent_x=32.0,
        inner_extent_y=16.0,
        inner_height=20.0,
        sparsity_factors=(1, 2, 1),
        spheroscale_profile=np.array([2.0, 2.0]),
        z_profile=np.array([0.0, 20.0]),
    )

    np.testing.assert_array_equal(grids["k"], np.array([8.0, 4.0, 2.0]))
    np.testing.assert_array_equal(grids["nx"], np.array([8, 16, 32]))
    np.testing.assert_array_equal(grids["ny"], np.array([8, 16, 32]))
    np.testing.assert_array_equal(grids["nz"], np.array([10, 14, 20]))
    np.testing.assert_allclose(grids["dx"], np.array([4.0, 2.0, 1.0]))
    np.testing.assert_allclose(grids["dy"], np.array([2.0, 1.0, 0.5]))
    np.testing.assert_allclose(grids["dz"], np.array([2.0, 20.0 / 14.0, 1.0]))

    expected_z_arrays = [
        np.arange(10, dtype=np.float64) * 2.0,
        np.arange(14, dtype=np.float64) * (20.0 / 14.0),
        np.arange(20, dtype=np.float64),
    ]
    expected_dz_arrays = [
        np.full(10, 2.0, dtype=np.float64),
        np.full(14, 20.0 / 14.0, dtype=np.float64),
        np.ones(20, dtype=np.float64),
    ]
    for z_actual, z_expected in zip(grids["z_arrays"], expected_z_arrays):
        np.testing.assert_allclose(z_actual, z_expected)
    for dz_actual, dz_expected in zip(grids["dz_arrays"], expected_dz_arrays):
        np.testing.assert_allclose(dz_actual, dz_expected)


def test_compute_all_grids_uses_actual_spacing_from_rounded_counts():
    grids = _compute_all_grids(
        k_values=np.array([7.0, 3.5]),
        inner_extent_x=30.0,
        inner_extent_y=18.0,
        inner_height=12.0,
        sparsity_factors=(1, 1, 1),
        spheroscale_profile=np.array([2.0, 2.0]),
        z_profile=np.array([0.0, 12.0]),
    )

    np.testing.assert_array_equal(grids["nx"], np.array([9, 17]))
    np.testing.assert_array_equal(grids["ny"], np.array([5, 10]))
    np.testing.assert_allclose(grids["dx"], 30.0 / grids["nx"])
    np.testing.assert_allclose(grids["dy"], 18.0 / grids["ny"])


# ===================================================================
# simulate.py — NetCDF output structure
# ===================================================================

def test_output_file_exists(small_nc):
    assert small_nc.exists()


def test_output_contains_required_variables(small_nc):
    ds = netCDF4.Dataset(small_nc, "r")
    expected = {"h", "qt", "x", "y", "z", "h_profile", "qt_profile",
                "z_profile", "k_values", "k_z_values", "C_h_k", "C_qt_k",
                "dz", "spheroscale"}
    assert expected <= set(ds.variables.keys())
    ds.close()


def test_output_contains_required_attributes(small_nc):
    ds = netCDF4.Dataset(small_nc, "r")
    expected = {"nx", "ny", "dx", "dy", "outer_scale",
                "domain_height", "profile_dz", "surface_pressure", "seed",
                "C_h_L", "C_qt_L", "n_large_turbulons", "H_h", "H_z",
                "sparsity_factors"}
    assert expected <= set(ds.ncattrs())
    ds.close()


def test_output_h_qt_shapes_match_dimensions(small_nc):
    ds = netCDF4.Dataset(small_nc, "r")
    nx = len(ds.dimensions["x"])
    ny = len(ds.dimensions["y"])
    nz = len(ds.dimensions["z"])
    assert ds.variables["h"].shape == (nx, ny, nz)
    assert ds.variables["qt"].shape == (nx, ny, nz)
    ds.close()


def test_output_coordinate_spacing(small_nc):
    ds = netCDF4.Dataset(small_nc, "r")
    x = ds.variables["x"][:]
    y = ds.variables["y"][:]
    z = ds.variables["z"][:]
    dz = ds.variables["dz"][:]
    np.testing.assert_allclose(x[1] - x[0], float(ds.dx), rtol=1e-5)
    np.testing.assert_allclose(y[1] - y[0], float(ds.dy), rtol=1e-5)
    # dz[0] should match the first z-spacing
    np.testing.assert_allclose(z[1] - z[0], float(dz[0]), rtol=1e-5)
    ds.close()


def test_output_profiles_roundtrip(small_nc, simple_profiles):
    h_in, qt_in = simple_profiles
    ds = netCDF4.Dataset(small_nc, "r")
    np.testing.assert_allclose(ds.variables["h_profile"][:], h_in.astype(np.float32), rtol=1e-6)
    np.testing.assert_allclose(ds.variables["qt_profile"][:], qt_in.astype(np.float32), rtol=1e-6)
    ds.close()


def test_output_oversampled_dimensions(tmp_path, simple_profiles):
    h, qt = simple_profiles
    out = tmp_path / "over.nc"
    simulate(h, qt, nx=16, ny=16, dx=500, dy=500,
             outer_scale=8000, spheroscale=100,
             domain_height=3000, profile_dz=30,
             output_path=out, seed=42,
             sparsity_factors=(2, 2, 2))
    ds = netCDF4.Dataset(out, "r")
    assert len(ds.dimensions["x"]) == 32
    assert len(ds.dimensions["y"]) == 32
    np.testing.assert_allclose(float(ds.dx), 250.0, rtol=1e-5)
    ds.close()


# ===================================================================
# simulate.py — Determinism & physics
# ===================================================================

def test_seed_reproducibility(tmp_path, simple_profiles):
    h, qt = simple_profiles
    kw = dict(nx=16, ny=16, dx=500, dy=500, outer_scale=8000, spheroscale=100,
              domain_height=3000, profile_dz=30, seed=99)
    p1 = simulate(h, qt, output_path=tmp_path / "a.nc", **kw)
    p2 = simulate(h, qt, output_path=tmp_path / "b.nc", **kw)
    ds1 = netCDF4.Dataset(p1, "r")
    ds2 = netCDF4.Dataset(p2, "r")
    np.testing.assert_array_equal(ds1.variables["h"][:], ds2.variables["h"][:])
    np.testing.assert_array_equal(ds1.variables["qt"][:], ds2.variables["qt"][:])
    ds1.close(); ds2.close()


def test_different_seeds_differ(tmp_path, simple_profiles):
    h, qt = simple_profiles
    kw = dict(nx=16, ny=16, dx=500, dy=500, outer_scale=8000, spheroscale=100,
              domain_height=3000, profile_dz=30)
    p1 = simulate(h, qt, output_path=tmp_path / "a.nc", seed=1, **kw)
    p2 = simulate(h, qt, output_path=tmp_path / "b.nc", seed=2, **kw)
    ds1 = netCDF4.Dataset(p1, "r")
    ds2 = netCDF4.Dataset(p2, "r")
    assert not np.array_equal(ds1.variables["h"][:], ds2.variables["h"][:])
    ds1.close(); ds2.close()


def test_h_mean_matches_profile(small_nc, simple_profiles):
    h_in, _ = simple_profiles
    ds = netCDF4.Dataset(small_nc, "r")
    h = ds.variables["h"][:]
    z = ds.variables["z"][:]
    z_prof = np.arange(len(h_in)) * 30.0
    h_interp = np.interp(z, z_prof, h_in)
    h_xy_mean = h.mean(axis=(0, 1))
    # Perturbations should be modest relative to the profile range
    np.testing.assert_allclose(h_xy_mean, h_interp, rtol=0.5)
    ds.close()


# ===================================================================
# simulate.py — Kernel & helpers
# ===================================================================

def test_kernel_is_symmetric():
    kernel = _turbulon_envelope(500, 100, 100, 50, support_factor=5)
    cx, cy, cz = kernel.shape[0]//2, kernel.shape[1]//2, kernel.shape[2]//2
    # x symmetry
    np.testing.assert_allclose(kernel[:cx, cy, cz], kernel[cx+1:, cy, cz][::-1], atol=1e-6)
    # y symmetry
    np.testing.assert_allclose(kernel[cx, :cy, cz], kernel[cx, cy+1:, cz][::-1], atol=1e-6)
    # z symmetry
    np.testing.assert_allclose(kernel[cx, cy, :cz], kernel[cx, cy, cz+1:][::-1], atol=1e-6)


def test_kernel_center_is_positive():
    kernel = _turbulon_envelope(500, 100, 100, 50, support_factor=5)
    cx, cy, cz = kernel.shape[0]//2, kernel.shape[1]//2, kernel.shape[2]//2
    assert kernel[cx, cy, cz] > 0


def test_kernel_shape_is_odd():
    kernel = _turbulon_envelope(500, 100, 100, 50, support_factor=5)
    assert all(s % 2 == 1 for s in kernel.shape)


def test_sparse_levy_density():
    region = NoiseRegion.root(np.random.SeedSequence(0), (20, 20, 20))
    field = _keyed_sparse_levy((20, 20, 20), (2, 2, 2), 1.8, region)
    n_nonzero = np.count_nonzero(field)
    expected = 10 * 10 * 10  # (20/2)^3
    assert n_nonzero == expected


def test_sparse_levy_s1_all_nonzero():
    region = NoiseRegion.root(np.random.SeedSequence(0), (10, 10, 10))
    field = _keyed_sparse_levy((10, 10, 10), (1, 1, 1), 1.8, region)
    # Probability of any element being exactly 0.0 is essentially 0
    assert np.count_nonzero(field) == 1000


def test_gradient_components_positive_for_random_field():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((20, 20, 20)).astype(np.float32)
    grad_h, grad_z = _gradient_components(field, 1.0, 1.0, 1.0)
    for G in (grad_h, grad_z):
        assert G.shape == field.shape
        assert np.all(G >= 0)
        assert G.mean() > 0


def test_gradient_components_zero_for_uniform_field():
    field = np.ones((10, 10, 10), dtype=np.float32) * 300e3
    grad_h, grad_z = _gradient_components(field, 100.0, 100.0, 50.0)
    np.testing.assert_allclose(grad_h, 0.0, atol=1e-6)
    np.testing.assert_allclose(grad_z, 0.0, atol=1e-6)


def test_gradient_components_split_directions():
    # x-only variation -> grad_h > 0, grad_z == 0; z-only -> the reverse.
    x_ramp = np.tile(np.arange(10, dtype=np.float32)[:, None, None], (1, 10, 10))
    grad_h, grad_z = _gradient_components(x_ramp, 1.0, 1.0, 1.0)
    assert grad_h[3:7].mean() > 0          # interior (rolls wrap at edges)
    np.testing.assert_allclose(grad_z, 0.0, atol=1e-6)
    z_ramp = np.tile(np.arange(10, dtype=np.float32)[None, None, :], (10, 10, 1))
    grad_h, grad_z = _gradient_components(z_ramp, 1.0, 1.0, 1.0)
    assert np.all(grad_z > 0)
    interior = grad_h[:, :, :]  # x,y uniform -> horizontal gradient zero everywhere
    np.testing.assert_allclose(interior, 0.0, atol=1e-6)


def test_fold_kernel_preserves_sum():
    kernel = _turbulon_envelope(500, 100, 100, 50, support_factor=5)
    folded = fold_kernel_to_field(kernel, (8, 8, kernel.shape[2]))
    np.testing.assert_allclose(folded.sum(), kernel.sum(), atol=1e-3)


def test_convolution_preserves_shape():
    field = np.random.default_rng(0).standard_normal((16, 16, 10)).astype(np.float32)
    kernel = _turbulon_envelope(500, 500, 500, 100, support_factor=5)
    result = convolve_periodic_xy_zeropad_z_oa(field, kernel)
    assert result.shape == field.shape


# ===================================================================
# thermodynamics.py
# ===================================================================

def test_recover_unsaturated_roundtrip():
    """For dry conditions, h = cp*T + Lv*qt + g*z should close."""
    nz = 20
    z = np.arange(nz, dtype=np.float64) * 200.0
    # Very dry so nothing saturates
    qt_val = 0.001
    T_target = 290.0 - 6.5e-3 * z
    h_vals = cp * T_target + Lv * qt_val + g * z
    h = np.broadcast_to(h_vals[None, None, :], (4, 4, nz)).copy()
    qt = np.full_like(h, qt_val)
    result = recover_diagnostics(h, qt, z, 101325.0)
    np.testing.assert_allclose(result["T"][:, :, 0], T_target[0], rtol=1e-4)


def test_qc_qi_zero_when_unsaturated():
    nz = 10
    z = np.arange(nz, dtype=np.float64) * 100.0
    qt_val = 0.0005  # very dry
    T_target = 300.0 - 6.5e-3 * z
    h = (cp * T_target + Lv * qt_val + g * z)[None, None, :] * np.ones((2, 2, nz))
    qt = np.full_like(h, qt_val)
    result = recover_diagnostics(h, qt, z, 101325.0)
    np.testing.assert_allclose(result["qc"], 0.0, atol=1e-10)
    np.testing.assert_allclose(result["qi"], 0.0, atol=1e-10)


def test_pressure_decreases_with_height():
    nz = 20
    z = np.arange(nz, dtype=np.float64) * 200.0
    T_target = 290.0 - 6.5e-3 * z
    qt_val = 0.002
    h = (cp * T_target + Lv * qt_val + g * z)[None, None, :] * np.ones((2, 2, nz))
    qt = np.full_like(h, qt_val)
    result = recover_diagnostics(h, qt, z, 101325.0)
    p = result["p"][0, 0, :]
    assert np.all(np.diff(p) < 0), "pressure should decrease with height"


def test_surface_pressure_matches_input():
    nz = 5
    z = np.arange(nz, dtype=np.float64) * 100.0
    h = np.full((2, 2, nz), 340e3)
    qt = np.full((2, 2, nz), 0.01)
    sp = 100000.0
    result = recover_diagnostics(h, qt, z, sp)
    np.testing.assert_allclose(result["p"][:, :, 0], sp)


def test_phase_partition_pure_liquid_above_273K():
    """At T > 273.15 K, lambda=1, so all condensate should be liquid."""
    nz = 3
    z = np.arange(nz, dtype=np.float64) * 100.0
    # Warm temperatures with lots of moisture to force saturation
    T_target = np.full(nz, 300.0)
    qt_val = 0.10  # way above saturation
    h = (cp * T_target + Lv * qt_val + g * z)[None, None, :] * np.ones((2, 2, nz))
    qt = np.full_like(h, qt_val)
    result = recover_diagnostics(h, qt, z, 101325.0)
    # Where T > 273.15, qi should be 0
    warm = result["T"] > 273.15
    if np.any(warm):
        np.testing.assert_allclose(result["qi"][warm], 0.0, atol=1e-10)


def test_phase_partition_pure_ice_below_235K():
    """At T < 235.15 K, lambda=0, so all condensate should be ice."""
    nz = 3
    z = np.arange(nz, dtype=np.float64) * 100.0
    # Very cold: T ~ 200 K
    T_target = np.full(nz, 200.0)
    qt_val = 0.05
    h = (cp * T_target + Lv * qt_val + g * z)[None, None, :] * np.ones((2, 2, nz))
    qt = np.full_like(h, qt_val)
    result = recover_diagnostics(h, qt, z, 101325.0)
    cold = result["T"] < 235.15
    if np.any(cold):
        np.testing.assert_allclose(result["qc"][cold], 0.0, atol=1e-10)


def test_compute_diagnostics_adds_variables(small_nc):
    compute_diagnostics(small_nc)
    ds = netCDF4.Dataset(small_nc, "r")
    assert {"T", "qv", "qc", "qi", "p"} <= set(ds.variables.keys())
    ds.close()


def test_compute_diagnostics_idempotent(small_nc):
    compute_diagnostics(small_nc)
    ds = netCDF4.Dataset(small_nc, "r")
    T1 = ds.variables["T"][:].copy()
    ds.close()
    compute_diagnostics(small_nc)
    ds = netCDF4.Dataset(small_nc, "r")
    T2 = ds.variables["T"][:]
    ds.close()
    np.testing.assert_array_equal(T1, T2)


def test_normalization_invariant_to_profile_dz(tmp_path):
    """Output field std should not depend on profile_dz resolution."""
    domain_height = 3000
    stds = {}
    for profile_dz in [1, 10, 100]:
        nz = int(domain_height / profile_dz) + 1
        z = np.arange(nz) * float(profile_dz)
        h = 340e3 - 20e3 * (z / domain_height)
        qt = 0.018 - 0.016 * (z / domain_height)

        out = tmp_path / f"test_dz{profile_dz}.nc"
        simulate(h, qt, nx=64, ny=64, dx=125, dy=125,
                 outer_scale=8000, spheroscale=100,
                 domain_height=domain_height, profile_dz=profile_dz,
                 output_path=out, seed=42)

        ds = netCDF4.Dataset(out)
        h_3d = ds.variables['h'][:]
        z_out = ds.variables['z'][:]
        # Subtract mean profile to get perturbation std
        h_mean = np.interp(z_out, z, h)
        h_pert = h_3d - h_mean[np.newaxis, np.newaxis, :]
        stds[profile_dz] = float(np.std(h_pert))
        ds.close()

    values = list(stds.values())
    mean_std = np.mean(values)
    for dz, s in stds.items():
        np.testing.assert_allclose(
            s, mean_std, rtol=0.1,
            err_msg=f"profile_dz={dz}: perturbation std={s:.1f} vs mean={mean_std:.1f}"
        )


def test_compute_diagnostics_chunking_matches_full(tmp_path, simple_profiles):
    h, qt = simple_profiles
    kw = dict(nx=16, ny=16, dx=500, dy=500, outer_scale=8000, spheroscale=100,
              domain_height=3000, profile_dz=30, seed=42)
    p1 = simulate(h, qt, output_path=tmp_path / "a.nc", **kw)
    p2 = simulate(h, qt, output_path=tmp_path / "b.nc", **kw)
    compute_diagnostics(p1, chunk_nx=2)
    compute_diagnostics(p2, chunk_nx=9999)
    ds1 = netCDF4.Dataset(p1, "r")
    ds2 = netCDF4.Dataset(p2, "r")
    for name in ("T", "qv", "qc", "qi", "p"):
        np.testing.assert_array_equal(ds1.variables[name][:], ds2.variables[name][:],
                                      err_msg=f"{name} differs between chunk sizes")
    ds1.close(); ds2.close()


def test_compute_diagnostics_parallel_matches_serial(tmp_path, simple_profiles):
    h, qt = simple_profiles
    kw = dict(nx=16, ny=16, dx=500, dy=500, outer_scale=8000, spheroscale=100,
              domain_height=3000, profile_dz=30, seed=42)
    p_serial = simulate(h, qt, output_path=tmp_path / "serial.nc", **kw)
    p_parallel = simulate(h, qt, output_path=tmp_path / "parallel.nc", **kw)
    compute_diagnostics(p_serial, chunk_nx=2, n_workers=1)
    compute_diagnostics(p_parallel, chunk_nx=2, n_workers=4)
    ds1 = netCDF4.Dataset(p_serial, "r")
    ds2 = netCDF4.Dataset(p_parallel, "r")
    for name in ("T", "qv", "qc", "qi", "p"):
        np.testing.assert_array_equal(
            ds1.variables[name][:], ds2.variables[name][:],
            err_msg=f"{name} differs between n_workers=1 and n_workers=4"
        )
    ds1.close(); ds2.close()


def _diagnostics_inputs(nx=24, ny=64, nz=115, seed=0):
    """h/qt with a ~30% saturated fraction, so both branches of the level
    solve are exercised."""
    z = np.linspace(0.0, 3000.0, nz)
    rng = np.random.default_rng(seed)
    h = ((340e3 - 20e3 * (z / 3000.0))[None, None, :]
         + rng.normal(0, 2e3, (nx, ny, nz))).astype(np.float32)
    qt = np.clip((0.018 - 0.016 * (z / 3000.0))[None, None, :]
                 + rng.normal(0, 2e-3, (nx, ny, nz)), 0, None).astype(np.float32)
    return h, qt, z


def test_recover_diagnostics_pressure_accumulator_is_float64():
    """The column march is 115 chained multiplies, so its float32 error
    compounds; the float64 accumulator must hold p to the float32 storage
    granularity of the output (~4e-3 Pa at ~1e5 Pa), not 25x worse.
    """
    h, qt, z = _diagnostics_inputs()
    # Exact answer available from these float32 inputs
    ref = recover_diagnostics(h.astype(np.float64), qt.astype(np.float64),
                              z, 101325.0)
    got = recover_diagnostics(h, qt, z, 101325.0)

    storage_floor = np.abs(
        ref["p"].astype(np.float32).astype(np.float64) - ref["p"]).max()
    err = np.abs(got["p"].astype(np.float64) - ref["p"]).max()
    assert err < 2 * storage_floor, (
        f"pressure error {err:.3e} Pa exceeds twice the float32 storage "
        f"floor {storage_floor:.3e} Pa -- the float64 column accumulator "
        f"in recover_diagnostics has regressed"
    )
    assert got["p"].dtype == np.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_recover_diagnostics_cuda_matches_cpu():
    """The dispatch is not bit-identical (CUDA expf is not glibc's), but must
    agree to the float32 granularity of what gets written."""
    h, qt, z = _diagnostics_inputs()
    cpu = recover_diagnostics(h, qt, z, 101325.0)
    gpu = recover_diagnostics(h, qt, z, 101325.0, device='cuda')

    saturated = float(np.mean(cpu["qc"] + cpu["qi"] > 0))
    assert 0.05 < saturated < 0.95, (
        f"test field is {saturated:.2f} saturated; it must exercise both "
        f"branches of the level solve for this comparison to mean anything"
    )
    tol = {"T": 1e-3, "qv": 1e-6, "qc": 1e-6, "qi": 1e-6, "p": 1.0}
    for name, atol in tol.items():
        assert gpu[name].dtype == np.float32, name
        np.testing.assert_allclose(gpu[name], cpu[name], atol=atol, rtol=0,
                                   err_msg=f"{name} cpu vs cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_recover_diagnostics_cuda_accepts_2d_surface_pressure():
    """Elevated-bottom nests pass a 2D p_bottom rather than a scalar."""
    h, qt, z = _diagnostics_inputs()
    p_bottom = np.full(h.shape[:2], 101325.0, dtype=np.float32)
    p_bottom += np.random.default_rng(1).normal(0, 500, h.shape[:2]).astype(np.float32)
    cpu = recover_diagnostics(h, qt, z, p_bottom)
    gpu = recover_diagnostics(h, qt, z, p_bottom, device='cuda')
    np.testing.assert_allclose(gpu["T"], cpu["T"], atol=1e-3, rtol=0)
    np.testing.assert_allclose(gpu["p"], cpu["p"], atol=1.0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_compute_diagnostics_cuda_matches_cpu(tmp_path, simple_profiles):
    h, qt = simple_profiles
    kw = dict(nx=16, ny=16, dx=500, dy=500, outer_scale=8000, spheroscale=100,
              domain_height=3000, profile_dz=30, seed=42)
    p_cpu = simulate(h, qt, output_path=tmp_path / "cpu.nc", **kw)
    p_gpu = simulate(h, qt, output_path=tmp_path / "gpu.nc", **kw)
    compute_diagnostics(p_cpu, chunk_nx=4)
    compute_diagnostics(p_gpu, chunk_nx=4, device='cuda')
    ds1 = netCDF4.Dataset(p_cpu, "r")
    ds2 = netCDF4.Dataset(p_gpu, "r")
    tol = {"T": 1e-3, "qv": 1e-6, "qc": 1e-6, "qi": 1e-6, "p": 1.0}
    for name, atol in tol.items():
        np.testing.assert_allclose(
            ds2.variables[name][:], ds1.variables[name][:], atol=atol, rtol=0,
            err_msg=f"{name} differs between device='cpu' and device='cuda'"
        )
    ds1.close(); ds2.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_compute_diagnostics_cuda_chunking_matches(tmp_path, simple_profiles):
    """Chunking splits x, and columns are independent, so the cuda path is
    bit-identical across chunk sizes even though it does not match the host."""
    h, qt = simple_profiles
    kw = dict(nx=16, ny=16, dx=500, dy=500, outer_scale=8000, spheroscale=100,
              domain_height=3000, profile_dz=30, seed=42)
    p1 = simulate(h, qt, output_path=tmp_path / "c1.nc", **kw)
    p2 = simulate(h, qt, output_path=tmp_path / "c2.nc", **kw)
    compute_diagnostics(p1, chunk_nx=2, device='cuda')
    compute_diagnostics(p2, chunk_nx=9999, device='cuda')
    ds1 = netCDF4.Dataset(p1, "r")
    ds2 = netCDF4.Dataset(p2, "r")
    for name in ("T", "qv", "qc", "qi", "p"):
        np.testing.assert_array_equal(
            ds1.variables[name][:], ds2.variables[name][:],
            err_msg=f"{name} differs between chunk sizes on cuda")
    ds1.close(); ds2.close()


# ---------------------------------------------------------------------------
# The regrid ladder's wrapped-pad cost, checked before any cascade work
# ---------------------------------------------------------------------------

def _ladder_grids(domain_km, outer_km, n_per_dyad=1, dx=100.0,
                  domain_height=20000.0, spheroscale=10.0):
    outer = outer_km * 1e3
    z = np.arange(400) * 50.0
    profile = np.full(z.size, spheroscale)
    gap = 2.0 ** (1.0 / n_per_dyad)
    n = int(round(np.log(outer / (2 * dx)) / np.log(gap))) + 1
    k = outer / gap ** np.arange(n)
    grids = _compute_all_grids(k, domain_km * 1e3, domain_km * 1e3,
                               domain_height, (1, 1, 1), profile, z)
    return grids, domain_km * 1e3, outer


def test_regrid_ladder_warns_on_a_non_dyadic_domain_ratio():
    """Thomas's 2026-08-13 configuration, reproduced exactly.

    domain 204.8 km against outer_scale 2048 km -- ratio 0.1, not a power of
    two -- so round(2*domain/k_i) drifts and consecutive grid counts share
    little common factor: 51 -> 102 -> 205, where gcd(102, 205) = 1 and the
    wrapped pad is 9x the target volume. NINE of the thirteen hops carried that
    pad; the run only FAILED at the first hop whose target passed the resample's
    volume guard, ten classes in. This warning predicts that in microseconds.

    The class LADDER is dyadic here (k halves exactly at every step), which is
    why the resample's own error message -- "use a dyadic class ladder" -- points
    at the wrong knob for this configuration.
    """
    grids, domain, outer = _ladder_grids(204.8, 2048)
    with pytest.warns(RuntimeWarning, match="WILL FAIL mid-cascade") as caught:
        check_regrid_ladder(grids, domain, outer, 1, (1, 1, 1))
    message = str(caught[0].message)
    # It must name the offending hop and the actual cause, not just complain.
    assert "(102, 102, 144)" in message and "(205, 205, 211)" in message
    assert "9.00x" in message
    assert "not a power of two" in message
    assert "class LADDER is fine" in message
    # And it must say what to change, in the units the caller thinks in.
    assert "256 km" in message or "1638.4 km" in message


@pytest.mark.parametrize("domain_km,outer_km", [
    (256, 2048),        # 1/8
    (128, 2048),        # 1/16
    (512, 2048),        # 1/4
    (64, 2048),         # 1/32
    (204.8, 1638.4),    # his domain, outer moved to make the ratio 1/8
])
def test_a_power_of_two_ratio_is_silent(domain_km, outer_km):
    """The fix, verified: every hop's pad is one source cell, so no warning."""
    grids, domain, outer = _ladder_grids(domain_km, outer_km)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        check_regrid_ladder(grids, domain, outer, 1, (1, 1, 1))


def test_the_check_does_not_cry_wolf_on_clamped_coarse_classes():
    """A 1-cell axis regridding to 1 cell has a pad FACTOR of 9 -- 1 + 2*1 per
    axis, twice -- and costs nothing at all. The test therefore mirrors
    _wrap_plan's own: factor AND absolute size. Factor alone would warn on
    essentially every configuration, which is worse than not warning."""
    from steam.utils import wrap_pad_factor

    assert wrap_pad_factor((1, 1, 5), (1, 1, 7)) == pytest.approx(9.0)
    # ... but a tiny target must not warn.
    grids, domain, outer = _ladder_grids(256, 2048)
    assert int(grids['nx'][0]) == 1          # the clamped coarse class exists
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        check_regrid_ladder(grids, domain, outer, 1, (1, 1, 1))


def test_wrap_pad_factor_scales_with_the_grid_not_just_the_ratio():
    """An exact doubling still pads TWO target cells per side, so its factor is
    ((n + 4) / n)**2 -- 1.56 on a 16-cell axis and 1.004 at 2048. The overhead of
    a well-behaved ladder is therefore negligible exactly where it would matter
    (large grids) and irrelevant where it is large (small ones), which is why the
    check has to weigh factor against absolute size rather than either alone."""
    from steam.utils import wrap_pad_factor

    small = wrap_pad_factor((8, 8, 6), (16, 16, 10))
    large = wrap_pad_factor((1024, 1024, 100), (2048, 2048, 144))
    assert small == pytest.approx((20 / 16) ** 2)      # 1.5625
    assert large == pytest.approx((2052 / 2048) ** 2)  # 1.0039
    assert large < 1.01

    # Coprime counts on both periodic axes: the 9x case that actually bites.
    assert wrap_pad_factor((102, 102, 144), (205, 205, 211)) > 8.9


def test_a_toy_simulation_emits_no_regrid_warning(tmp_path):
    """The configurations the rest of this suite uses must stay silent, or the
    warning is noise and will be ignored when it matters."""
    z = np.arange(28) * 30.0
    h = 340e3 - 20e3 * (z / z.max())
    qt = 0.018 - 0.016 * (z / z.max())
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        simulate(h, qt, nx=16, ny=16, dx=125.0, dy=125.0,
                 outer_scale=2000.0, spheroscale=100.0,
                 domain_height=800.0, profile_dz=30.0,
                 output_path=tmp_path / "quiet.nc", seed=1)
