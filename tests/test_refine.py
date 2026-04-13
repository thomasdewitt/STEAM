"""Tests for per-class seeding reproducibility and subdomain refinement."""

import numpy as np
import pytest
import netCDF4
from pathlib import Path

from steam.simulate import (
    simulate,
    cascade_loop,
    refine,
    _compute_all_grids,
    _turbulon_envelope,
    _compute_normalization,
    spectral_width_normalization,
)


@pytest.fixture
def simple_profiles():
    nz = 50
    z = np.arange(nz) * 30.0
    h = 340e3 - 20e3 * (z / z.max())
    qt = 0.018 - 0.016 * (z / z.max())
    return h, qt


@pytest.fixture
def parent_nc(tmp_path, simple_profiles):
    """Run a small parent simulation and return the NC path."""
    h, qt = simple_profiles
    out = tmp_path / "parent.nc"
    simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
             outer_scale=8000, spheroscale=100,
             domain_height=3000, profile_dz=30,
             output_path=out, seed=42)
    return out


class TestPerClassSeeding:
    """Verify that per-class SeedSequence seeding is deterministic and
    independent: running the first M classes, then the remaining N-M
    classes, produces the same result as running all N at once."""

    def test_split_simulation_reproducibility(self, tmp_path, simple_profiles):
        h, qt = simple_profiles
        seed = 123
        outer_scale = 8000
        dx = dy = 250
        nx = ny = 32
        domain_height = 3000
        profile_dz = 30
        spheroscale = 100.0
        sparsity_factors = (1, 1, 1)

        z_profile = np.arange(len(h)) * profile_dz
        spheroscale_profile = np.full(len(h), spheroscale)

        size_class_gap_factor = 2.0
        n_classes = int(
            round(np.log(outer_scale / (2 * dx)) / np.log(size_class_gap_factor))
        ) + 1
        k_values = outer_scale / size_class_gap_factor ** np.arange(n_classes)

        domain_x = nx * dx
        domain_y = ny * dy
        grids = _compute_all_grids(
            k_values, domain_x, domain_y, domain_height,
            sparsity_factors, spheroscale_profile, z_profile,
        )

        # Normalization
        from steam.constants import hurst_vertical_anisotropy as H_z
        s_x, s_y, s_z = sparsity_factors
        z_finest = grids['z_arrays'][-1]
        h_on_finest = np.interp(z_finest, z_profile, h)
        qt_on_finest = np.interp(z_finest, z_profile, qt)
        k_min = k_values[-1]
        vert_outer_pts = int(round(2 * s_z * (outer_scale / k_min) ** H_z))
        unit_turbulon = _turbulon_envelope(1, 1/(2*s_x), 1/(2*s_y), 1/(2*s_z),
                                           support_factor=10)
        spectral_corr = spectral_width_normalization('mexican_hat', size_class_gap_factor)

        C_h_k = _compute_normalization(
            h_on_finest, vert_outer_pts, k_values, outer_scale, grids,
            unit_turbulon, 'mexican_hat'
        )
        C_qt_k = _compute_normalization(
            qt_on_finest, vert_outer_pts, k_values, outer_scale, grids,
            unit_turbulon, 'mexican_hat'
        )
        C_h_k = [c * spectral_corr for c in C_h_k]
        C_qt_k = [c * spectral_corr for c in C_qt_k]

        # Full cascade in one shot
        seed_sequence = np.random.SeedSequence(seed)
        all_seeds = seed_sequence.spawn(n_classes)

        h_pert_full, qt_pert_full, _ = cascade_loop(
            h, qt, z_profile, grids,
            C_h_k, C_qt_k,
            315*1004, 355*1004, 0.0, 30/1000,
            1, sparsity_factors, all_seeds,
        )

        # Split cascade: first half, then second half
        # Re-spawn same seeds (SeedSequence is deterministic)
        seed_sequence2 = np.random.SeedSequence(seed)
        all_seeds2 = seed_sequence2.spawn(n_classes)

        split_point = n_classes // 2

        def split_grids(grids, sl):
            return {
                'k': grids['k'][sl],
                'nx': grids['nx'][sl],
                'ny': grids['ny'][sl],
                'nz': grids['nz'][sl],
                'dx': grids['dx'][sl],
                'dy': grids['dy'][sl],
                'dz': grids['dz'][sl],
                'z_arrays': grids['z_arrays'][sl.start:sl.stop],
                'dz_arrays': grids['dz_arrays'][sl.start:sl.stop],
            }

        first = slice(0, split_point)
        second = slice(split_point, n_classes)

        h_pert_1, qt_pert_1, _ = cascade_loop(
            h, qt, z_profile, split_grids(grids, first),
            C_h_k[:split_point], C_qt_k[:split_point],
            315*1004, 355*1004, 0.0, 30/1000,
            1, sparsity_factors, all_seeds2[:split_point],
        )

        h_pert_2, qt_pert_2, _ = cascade_loop(
            h, qt, z_profile, split_grids(grids, second),
            C_h_k[split_point:], C_qt_k[split_point:],
            315*1004, 355*1004, 0.0, 30/1000,
            1, sparsity_factors, all_seeds2[split_point:],
            h_perturbation=h_pert_1,
            qt_perturbation=qt_pert_1,
        )

        np.testing.assert_array_equal(
            h_pert_2, h_pert_full,
            err_msg="Split cascade does not match full cascade"
        )


class TestRefine:
    """Smoke test for subdomain refinement."""

    def test_refine_creates_group_with_correct_dimensions(self, parent_nc):
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        ds.close()

        # Refine the first quadrant to 2x finer resolution
        half_x = parent_nx // 2
        half_y = parent_ny // 2

        ds = netCDF4.Dataset(parent_nc, "r")
        parent_dx = float(ds.dx)
        parent_dy = float(ds.dy)
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        # New dx = half of parent's finest k / 2 (so there are new classes)
        new_dx = k_finest / 4
        new_dy = k_finest / 4

        refine(
            parent_nc,
            x_start=0, x_stop=half_x,
            y_start=0, y_stop=half_y,
            dx=new_dx, dy=new_dy,
            seed=99,
        )

        # Verify the group exists and has correct structure
        ds = netCDF4.Dataset(parent_nc, "r")
        assert "refinements" in ds.groups
        ref_group = ds.groups["refinements"]
        assert "r0" in ref_group.groups
        r0 = ref_group.groups["r0"]

        # Check variables exist and are finite
        assert "h" in r0.variables
        assert "qt" in r0.variables
        h_refined = r0.variables["h"][:]
        qt_refined = r0.variables["qt"][:]
        assert np.all(np.isfinite(h_refined))
        assert np.all(np.isfinite(qt_refined))

        # Check refinement-specific attributes
        assert hasattr(r0, "parent_group")
        assert hasattr(r0, "parent_x_slice")
        x_slice = r0.parent_x_slice
        assert x_slice[0] == 0
        assert x_slice[1] == half_x

        ds.close()

    def test_refine_auto_group_naming(self, parent_nc):
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        half_x = parent_nx // 2
        half_y = parent_ny // 2
        new_dx = k_finest / 4
        new_dy = k_finest / 4

        # First refinement -> r0
        refine(parent_nc, 0, half_x, 0, half_y, new_dx, new_dy, seed=99)
        # Second refinement -> r1
        refine(parent_nc, 0, half_x, 0, half_y, new_dx, new_dy, seed=100)

        ds = netCDF4.Dataset(parent_nc, "r")
        ref_group = ds.groups["refinements"]
        assert "r0" in ref_group.groups
        assert "r1" in ref_group.groups
        ds.close()

    def test_refine_z_limited(self, parent_nc):
        """Refinement with z_min/z_max produces a vertically limited inset."""
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        parent_domain_height = float(ds.domain_height)
        k_finest = float(ds.variables["k_values"][:][-1])
        parent_z = ds.variables["z"][:]
        ds.close()

        half_x = parent_nx // 2
        half_y = parent_ny // 2
        new_dx = k_finest / 4
        new_dy = k_finest / 4

        # Take the middle third of the domain height
        z_min = parent_domain_height / 3
        z_max = 2 * parent_domain_height / 3
        inset_height = z_max - z_min

        refine(
            parent_nc,
            x_start=0, x_stop=half_x,
            y_start=0, y_stop=half_y,
            dx=new_dx, dy=new_dy,
            z_min=z_min, z_max=z_max,
            seed=99,
        )

        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]

        # domain_height should be the inset span, not the full height
        assert abs(float(r0.domain_height) - inset_height) < 1.0
        assert abs(float(r0.domain_z_min) - z_min) < 1.0

        # z-coordinates should lie within [z_min, z_max]
        z_refined = r0.variables["z"][:]
        assert z_refined[0] >= z_min - 1.0
        assert z_refined[-1] <= z_max + 1.0

        # Fields should be finite
        h_refined = r0.variables["h"][:]
        qt_refined = r0.variables["qt"][:]
        assert np.all(np.isfinite(h_refined))
        assert np.all(np.isfinite(qt_refined))

        ds.close()

    def test_refine_z_limited_bad_range_raises(self, parent_nc):
        """z_min/z_max outside parent range raises ValueError."""
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        parent_domain_height = float(ds.domain_height)
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        half_x = parent_nx // 2
        half_y = parent_ny // 2
        new_dx = k_finest / 4
        new_dy = k_finest / 4

        with pytest.raises(ValueError, match="exceeds parent"):
            refine(
                parent_nc, 0, half_x, 0, half_y, new_dx, new_dy,
                z_min=0, z_max=parent_domain_height + 500,
                seed=99,
            )
