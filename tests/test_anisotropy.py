"""Smoke tests for the grid-anisotropy option."""

import numpy as np
import pytest
import netCDF4

from steam.simulate import simulate, _k_z
from steam.constants import hurst_vertical_anisotropy as H_z


@pytest.fixture
def profiles():
    nz = 101                 # 100 * 30 m spans the 3000 m domains here
    z = np.arange(nz) * 30.0
    h = 340e3 - 20e3 * (z / z.max())
    qt = 0.018 - 0.016 * (z / z.max())
    return h, qt


def test_k_z_canonical_matches_formula():
    k = np.array([10.0, 100.0, 1000.0])
    spheroscale = 100.0
    expected = spheroscale * (k / spheroscale) ** H_z
    got = _k_z('canonical', k, spheroscale)
    np.testing.assert_allclose(got, expected)


def test_k_z_piecewise_isotropic_below_spheroscale():
    spheroscale = 100.0
    k = np.array([10.0, 50.0, 100.0, 500.0, 1000.0])
    got = _k_z('piecewise_isotropic_below_spheroscale', k, spheroscale)
    # Below spheroscale: k_z = k
    np.testing.assert_allclose(got[:2], k[:2])
    # At spheroscale: both formulas agree
    np.testing.assert_allclose(got[2], spheroscale)
    # Above spheroscale: canonical
    canonical_above = spheroscale * (k[3:] / spheroscale) ** H_z
    np.testing.assert_allclose(got[3:], canonical_above)


def test_k_z_continuous_at_spheroscale():
    spheroscale = 100.0
    eps = 1e-6
    k_below = spheroscale - eps
    k_above = spheroscale + eps
    below = float(_k_z('piecewise_isotropic_below_spheroscale', k_below, spheroscale))
    above = float(_k_z('piecewise_isotropic_below_spheroscale', k_above, spheroscale))
    assert abs(below - above) < 1e-4


def test_k_z_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown anisotropy"):
        _k_z('made_up', 100.0, 100.0)


def test_simulate_piecewise_finite(tmp_path, profiles):
    h, qt = profiles
    out = tmp_path / "piecewise.nc"
    # spheroscale sits mid-cascade so both branches of the piecewise get exercised
    simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
             outer_scale=4000, spheroscale=700,
             domain_height=3000, profile_dz=30,
             output_path=out, seed=42,
             anisotropy='piecewise_isotropic_below_spheroscale')

    ds = netCDF4.Dataset(out, "r")
    h_field = ds.variables["h"][:]
    qt_field = ds.variables["qt"][:]
    k_values = ds.variables["k_values"][:]
    k_z_values = ds.variables["k_z_values"][:]
    anisotropy_attr = ds.anisotropy
    spheroscale_profile = ds.variables["spheroscale"][:]
    ds.close()

    assert np.all(np.isfinite(h_field))
    assert np.all(np.isfinite(qt_field))
    assert anisotropy_attr == 'piecewise_isotropic_below_spheroscale'

    # At classes with k < spheroscale_mean, k_z should equal k.
    spheroscale_mean = float(np.mean(spheroscale_profile))
    below = k_values < spheroscale_mean
    above = k_values >= spheroscale_mean
    np.testing.assert_allclose(k_z_values[below], k_values[below], rtol=1e-5)
    canonical_above = spheroscale_mean * (k_values[above] / spheroscale_mean) ** H_z
    np.testing.assert_allclose(k_z_values[above], canonical_above, rtol=1e-5)


def test_simulate_default_is_piecewise(tmp_path, profiles):
    """The default anisotropy is piecewise-isotropic (2026-08-06 ruling).

    Every class in this config sits above the spheroscale, so the k_z law
    coincides with the canonical formula; the attribute pins the default.
    """
    h, qt = profiles
    out = tmp_path / "default.nc"
    simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
             outer_scale=8000, spheroscale=100,
             domain_height=3000, profile_dz=30,
             output_path=out, seed=42)

    ds = netCDF4.Dataset(out, "r")
    k_values = ds.variables["k_values"][:]
    k_z_values = ds.variables["k_z_values"][:]
    anisotropy_attr = ds.anisotropy
    spheroscale_profile = ds.variables["spheroscale"][:]
    ds.close()

    assert anisotropy_attr == 'piecewise_isotropic_below_spheroscale'
    spheroscale_mean = float(np.mean(spheroscale_profile))
    expected = spheroscale_mean * (k_values / spheroscale_mean) ** H_z
    np.testing.assert_allclose(k_z_values, expected, rtol=1e-5)




def test_simulate_rejects_unknown_anisotropy(tmp_path, profiles):
    h, qt = profiles
    out = tmp_path / "bad.nc"
    with pytest.raises(ValueError, match="anisotropy must be one of"):
        simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
                 outer_scale=8000, spheroscale=100,
                 domain_height=3000, profile_dz=30,
                 output_path=out, seed=42,
                 anisotropy='made_up')
