"""Sanity check tests for STEAM."""

import numpy as np
import pytest
import netCDF4
from pathlib import Path

from steam.simulate import (
    simulate,
    _turbulon_envelope,
    _sparse_noise,
    _normalized_gradient,
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
    h = 340e3 - 30e3 * (z / z.max())
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


def test_rejects_profile_dz_too_coarse(tmp_path):
    # spheroscale=100, outer_scale=8000 => k_z_L ~ 1467 m
    # profile_dz=2000 >= k_z_L => should raise
    nz = 3
    h = np.linspace(340e3, 310e3, nz)
    qt = np.linspace(0.018, 0.002, nz)
    with pytest.raises(ValueError, match="profile_dz"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 4000, 2000, tmp_path / "x.nc")


def test_rejects_zero_sparsity(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="positive integer"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc",
                 sparsity_factors=(0, 1, 1))


def test_rejects_float_sparsity(tmp_path, simple_profiles):
    h, qt = simple_profiles
    with pytest.raises(ValueError, match="positive integer"):
        simulate(h, qt, 16, 16, 500, 500, 8000, 100, 3000, 30, tmp_path / "x.nc",
                 sparsity_factors=(1.5, 1, 1))


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


def test_sparse_noise_density():
    rng = np.random.default_rng(0)
    field = _sparse_noise(20, 20, 20, 2, 2, 2, rng)
    n_nonzero = np.count_nonzero(field)
    expected = 10 * 10 * 10  # (20/2)^3
    assert n_nonzero == expected


def test_sparse_noise_s1_all_nonzero():
    rng = np.random.default_rng(0)
    field = _sparse_noise(10, 10, 10, 1, 1, 1, rng)
    # Probability of any element being exactly 0.0 is essentially 0
    assert np.count_nonzero(field) == 1000


def test_normalized_gradient_mean_is_one():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((20, 20, 20)).astype(np.float32)
    G = _normalized_gradient(field, 1.0, 1.0, 1.0)
    np.testing.assert_allclose(G.mean(), 1.0, atol=0.15)


def test_normalized_gradient_uniform_field_returns_ones():
    field = np.ones((10, 10, 10), dtype=np.float32) * 300e3
    G = _normalized_gradient(field, 100.0, 100.0, 50.0)
    np.testing.assert_array_equal(G, np.ones_like(G))


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
        h = 340e3 - 30e3 * (z / domain_height)
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
