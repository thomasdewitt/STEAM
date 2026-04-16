"""Smoke tests for the grid-anisotropy option."""

import numpy as np
import pytest
import netCDF4

from steam.simulate import simulate, refine, _k_z
from steam.constants import hurst_vertical_anisotropy as H_z


@pytest.fixture
def profiles():
    nz = 50
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


def test_simulate_canonical_unchanged(tmp_path, profiles):
    """The default anisotropy must still be canonical — k_z obeys the old law."""
    h, qt = profiles
    out = tmp_path / "canonical.nc"
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

    assert anisotropy_attr == 'canonical'
    spheroscale_mean = float(np.mean(spheroscale_profile))
    expected = spheroscale_mean * (k_values / spheroscale_mean) ** H_z
    np.testing.assert_allclose(k_z_values, expected, rtol=1e-5)


def test_refine_with_different_anisotropy(tmp_path, profiles):
    """Refine the same canonical parent twice with two different anisotropy functions."""
    h, qt = profiles
    parent = tmp_path / "parent.nc"
    simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
             outer_scale=4000, spheroscale=700,
             domain_height=3000, profile_dz=30,
             output_path=parent, seed=42)

    # Refinement inner must be an integer multiple of new_outer_scale = parent's finest k.
    ds = netCDF4.Dataset(parent, "r")
    parent_dx = float(ds.dx)
    k_finest = float(ds.variables["k_values"][:][-1])
    parent_nx = len(ds.dimensions["x"])
    ds.close()

    # Use spanning x/y so we avoid boundary-pad issues, and a smaller dx so at
    # least one refined class sits below spheroscale (1000 m).
    new_dx = new_dy = k_finest / 4

    refine(parent, 0, parent_nx, 0, parent_nx, new_dx, new_dy,
           seed=99, anisotropy='canonical')
    refine(parent, 0, parent_nx, 0, parent_nx, new_dx, new_dy,
           seed=100, anisotropy='piecewise_isotropic_below_spheroscale')

    ds = netCDF4.Dataset(parent, "r")
    r0 = ds.groups["refinements"].groups["r0"]
    r1 = ds.groups["refinements"].groups["r1"]

    assert r0.anisotropy == 'canonical'
    assert r1.anisotropy == 'piecewise_isotropic_below_spheroscale'

    h0 = r0.variables["h"][:]
    h1 = r1.variables["h"][:]
    assert np.all(np.isfinite(h0))
    assert np.all(np.isfinite(h1))

    # Piecewise k_z_values must differ from canonical at the refined (small-k) classes
    # since the refined k_values fall below spheroscale.
    k_r1 = r1.variables["k_values"][:]
    k_z_r1 = r1.variables["k_z_values"][:]
    spheroscale_mean = float(np.mean(r1.variables["spheroscale"][:]))
    below = k_r1 < spheroscale_mean
    assert np.any(below), "Test setup should produce refined classes below spheroscale"
    np.testing.assert_allclose(k_z_r1[below], k_r1[below], rtol=1e-5)
    ds.close()


def test_refine_inherits_anisotropy_from_parent(tmp_path, profiles):
    h, qt = profiles
    parent = tmp_path / "parent.nc"
    simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
             outer_scale=4000, spheroscale=700,
             domain_height=3000, profile_dz=30,
             output_path=parent, seed=42,
             anisotropy='piecewise_isotropic_below_spheroscale')

    ds = netCDF4.Dataset(parent, "r")
    k_finest = float(ds.variables["k_values"][:][-1])
    parent_nx = len(ds.dimensions["x"])
    ds.close()

    refine(parent, 0, parent_nx, 0, parent_nx, k_finest / 4, k_finest / 4,
           seed=99)  # anisotropy omitted → inherit

    ds = netCDF4.Dataset(parent, "r")
    r0 = ds.groups["refinements"].groups["r0"]
    assert r0.anisotropy == 'piecewise_isotropic_below_spheroscale'
    ds.close()


def test_simulate_rejects_unknown_anisotropy(tmp_path, profiles):
    h, qt = profiles
    out = tmp_path / "bad.nc"
    with pytest.raises(ValueError, match="anisotropy must be one of"):
        simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
                 outer_scale=8000, spheroscale=100,
                 domain_height=3000, profile_dz=30,
                 output_path=out, seed=42,
                 anisotropy='made_up')
