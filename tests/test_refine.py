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
                'padded_extent_x': grids['padded_extent_x'][sl],
                'padded_extent_y': grids['padded_extent_y'][sl],
                'padded_height': grids['padded_height'][sl],
                'z_min_per_class': grids['z_min_per_class'][sl],
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

    def test_refine_spanning_x_is_periodic(self, parent_nc):
        """When inset spans the parent in x, the output should be periodic
        in x with period = parent_domain_x. No padding on the spanning dim."""
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        half_y = parent_ny // 2
        new_dx = k_finest / 4
        new_dy = k_finest / 4

        refine(
            parent_nc,
            x_start=0, x_stop=parent_nx,   # spanning x
            y_start=0, y_stop=half_y,
            dx=new_dx, dy=new_dy,
            seed=7,
        )

        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]
        h = r0.variables["h"][:]
        qt = r0.variables["qt"][:]
        ds.close()

        # Periodicity in x: first column ~= "wrap" of last column under FFT
        # convolution. We don't expect byte-exact equality after the full
        # cascade, but the discrepancy should be small relative to the field's
        # horizontal variability — the cascade ran with period = inner_extent.
        first_col = h[0, :, :]
        last_col = h[-1, :, :]
        # Compare the left-right seam to a mid-field discontinuity to confirm
        # the wraparound is as small as the field's natural continuity.
        seam_gap = np.abs(first_col - last_col).mean()
        interior_gap = np.abs(h[0, :, :] - h[1, :, :]).mean()
        assert seam_gap < 10 * interior_gap, (
            f"x-seam discontinuity {seam_gap:.3e} is much larger than interior "
            f"cell-to-cell variation {interior_gap:.3e} — x periodicity likely "
            f"broken for spanning inset."
        )

    def test_refine_nested_interior(self, parent_nc):
        """Second-level refinement fully inside the first level must succeed
        and not silently wrap around the first-level's (non-periodic) boundary."""
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        k_finest = float(ds.variables["k_values"][:][-1])
        parent_dx = float(ds.dx)
        ds.close()

        # First refinement: quadrant, with some room inside for a second refine
        quarter_x = parent_nx // 2
        quarter_y = parent_ny // 2
        new_dx = k_finest / 4
        new_dy = k_finest / 4

        refine(
            parent_nc,
            x_start=0, x_stop=quarter_x,
            y_start=0, y_stop=quarter_y,
            dx=new_dx, dy=new_dy,
            seed=11,
        )

        # Read r0 geometry
        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]
        r0_nx = len(r0.dimensions["x"])
        r0_ny = len(r0.dimensions["y"])
        r0_dx = float(r0.dx)
        r0_dy = float(r0.dy)
        r0_k_finest = float(r0.variables["k_values"][:][-1])
        ds.close()

        # Pick a second refine fully inside r0 with enough room on each side
        pad_cells_needed = int(np.ceil(5 * r0_k_finest / r0_dx))  # SUPPORT_FACTOR=5
        # Inner region: a tile at least 1 new_outer_scale wide, centered
        new2_outer = r0_k_finest
        inner2_cells_x = int(round(new2_outer / r0_dx))
        inner2_cells_y = int(round(new2_outer / r0_dy))
        x2_start = max(pad_cells_needed, r0_nx // 4)
        y2_start = max(pad_cells_needed, r0_ny // 4)
        x2_stop = x2_start + inner2_cells_x
        y2_stop = y2_start + inner2_cells_y

        # Confirm room on both sides
        assert x2_start >= pad_cells_needed
        assert x2_stop <= r0_nx - pad_cells_needed
        assert y2_start >= pad_cells_needed
        assert y2_stop <= r0_ny - pad_cells_needed

        refine(
            parent_nc,
            x_start=x2_start, x_stop=x2_stop,
            y_start=y2_start, y_stop=y2_stop,
            dx=r0_dx / 2, dy=r0_dy / 2,
            parent_group="refinements/r0",
            seed=13,
        )

        ds = netCDF4.Dataset(parent_nc, "r")
        r1 = ds.groups["refinements"].groups["r1"]
        h = r1.variables["h"][:]
        assert np.all(np.isfinite(h))
        ds.close()

    def test_refine_nested_boundary_rejected(self, parent_nc):
        """Second-level refinement too close to first-level's boundary must raise."""
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        half_x = parent_nx // 2
        half_y = parent_ny // 2
        new_dx = k_finest / 4
        new_dy = k_finest / 4

        refine(
            parent_nc,
            x_start=0, x_stop=half_x,
            y_start=0, y_stop=half_y,
            dx=new_dx, dy=new_dy,
            seed=17,
        )

        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]
        r0_nx = len(r0.dimensions["x"])
        r0_ny = len(r0.dimensions["y"])
        r0_dx = float(r0.dx)
        r0_dy = float(r0.dy)
        r0_k_finest = float(r0.variables["k_values"][:][-1])
        ds.close()

        # Refine r0 starting at x=0 — touches r0's non-periodic boundary.
        # Pick an inner tile that leaves room on the right but 0 on the left.
        new2_outer = r0_k_finest
        inner2_cells_x = int(round(new2_outer / r0_dx))
        inner2_cells_y = int(round(new2_outer / r0_dy))
        with pytest.raises(ValueError, match="non-periodic parent"):
            refine(
                parent_nc,
                x_start=0, x_stop=inner2_cells_x,
                y_start=0, y_stop=inner2_cells_y,
                dx=r0_dx / 2, dy=r0_dy / 2,
                parent_group="refinements/r0",
                seed=19,
            )

    def test_refine_tiny_inner_memory(self, parent_nc):
        """Tiny-inner refinement should keep the finest-class padded grid
        size bounded by inner + constant-cell pad, not by the full
        (inner + SUPPORT*L) extent at parent resolution."""
        from steam.simulate import SUPPORT_FACTOR

        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        parent_dx = float(ds.dx)
        parent_dy = float(ds.dy)
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        # Inner: exactly one new_outer_scale (k_finest) wide in both dims.
        # With k_finest=500m and parent_dx=250m, that's 2 parent pixels.
        inner_cells_x = int(round(k_finest / parent_dx))
        inner_cells_y = int(round(k_finest / parent_dy))
        new_dx = k_finest / 8   # gives 4 refined size classes
        new_dy = k_finest / 8

        x_start = parent_nx // 2
        x_stop = x_start + inner_cells_x
        y_start = parent_ny // 2
        y_stop = y_start + inner_cells_y

        refine(
            parent_nc,
            x_start=x_start, x_stop=x_stop,
            y_start=y_start, y_stop=y_stop,
            dx=new_dx, dy=new_dy,
            seed=23,
        )

        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]
        nx_out = len(r0.dimensions["x"])
        ny_out = len(r0.dimensions["y"])
        ds.close()

        # Output spans just the inner region: nx_out cells at spacing ~new_dx
        # over inner_extent = k_finest. Expected ~8 cells.
        assert nx_out <= 16, (
            f"nx_out={nx_out}; output should reflect inner only (~8), "
            f"not inner+static-pad at parent resolution"
        )

    def test_refine_elevated_places_turbulons_below(self, parent_nc):
        """Elevated z-inset should allow new turbulon centers *below* the inset's
        z_min and *above* its z_max (in the z-pad). We verify indirectly: the
        first and last z-slices of the inset differ across seeds — if they were
        dominated only by the inherited parent field, they'd be identical."""
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
        z_min = parent_domain_height / 3
        z_max = 2 * parent_domain_height / 3

        # Run two refinements with different seeds, same spatial extent
        refine(
            parent_nc, 0, half_x, 0, half_y, new_dx, new_dy,
            z_min=z_min, z_max=z_max, seed=101,
        )
        refine(
            parent_nc, 0, half_x, 0, half_y, new_dx, new_dy,
            z_min=z_min, z_max=z_max, seed=202,
        )

        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]
        r1 = ds.groups["refinements"].groups["r1"]
        h0 = r0.variables["h"][:]
        h1 = r1.variables["h"][:]
        ds.close()

        # Bottom z-slice (within the inset) — if we had suppressed turbulon
        # centers in the bottom band AND below the inset, the bottom rows
        # would show only parent variance; with new centers below the inset
        # and no suppression in the bottom band, seed changes should shift
        # the bottom rows significantly.
        bottom_diff = np.abs(h0[:, :, 0] - h1[:, :, 0])
        mid_diff = np.abs(h0[:, :, h0.shape[2] // 2] - h1[:, :, h0.shape[2] // 2])
        assert bottom_diff.mean() > 0.1 * mid_diff.mean(), (
            f"Bottom-slice seed variation {bottom_diff.mean():.3e} is much smaller "
            f"than mid-slice {mid_diff.mean():.3e}; new turbulons below the inset "
            f"may not be contributing."
        )

    def test_refine_spanning_y_is_periodic(self, parent_nc):
        """Mirror of spanning-x: inset spans parent in y should be periodic in y."""
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        half_x = parent_nx // 2
        new_dx = k_finest / 4
        new_dy = k_finest / 4

        refine(
            parent_nc,
            x_start=0, x_stop=half_x,
            y_start=0, y_stop=parent_ny,   # spanning y
            dx=new_dx, dy=new_dy,
            seed=31,
        )

        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]
        h = r0.variables["h"][:]
        ds.close()

        seam_gap = np.abs(h[:, 0, :] - h[:, -1, :]).mean()
        interior_gap = np.abs(h[:, 0, :] - h[:, 1, :]).mean()
        assert seam_gap < 10 * interior_gap, (
            f"y-seam discontinuity {seam_gap:.3e} is much larger than interior "
            f"{interior_gap:.3e} — y periodicity likely broken for spanning inset."
        )

    def test_refine_doubly_spanning(self, parent_nc):
        """Inset = whole parent in both x and y. Should be periodic in both dims."""
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        new_dx = k_finest / 4
        new_dy = k_finest / 4

        refine(
            parent_nc,
            x_start=0, x_stop=parent_nx,
            y_start=0, y_stop=parent_ny,
            dx=new_dx, dy=new_dy,
            seed=37,
        )

        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]
        h = r0.variables["h"][:]
        ds.close()

        assert np.all(np.isfinite(h))

        x_seam = np.abs(h[0, :, :] - h[-1, :, :]).mean()
        x_interior = np.abs(h[0, :, :] - h[1, :, :]).mean()
        y_seam = np.abs(h[:, 0, :] - h[:, -1, :]).mean()
        y_interior = np.abs(h[:, 0, :] - h[:, 1, :]).mean()
        assert x_seam < 10 * x_interior, f"x-seam {x_seam:.3e} vs {x_interior:.3e}"
        assert y_seam < 10 * y_interior, f"y-seam {y_seam:.3e} vs {y_interior:.3e}"

    def test_refine_spanning_x_with_interior_z(self, parent_nc):
        """Spanning x + interior z-slice: both new code paths composed."""
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        parent_domain_height = float(ds.domain_height)
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        half_y = parent_ny // 2
        new_dx = k_finest / 4
        new_dy = k_finest / 4
        z_min = parent_domain_height / 3
        z_max = 2 * parent_domain_height / 3

        refine(
            parent_nc,
            x_start=0, x_stop=parent_nx,   # spanning x
            y_start=0, y_stop=half_y,
            dx=new_dx, dy=new_dy,
            z_min=z_min, z_max=z_max,
            seed=41,
        )

        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]
        h = r0.variables["h"][:]
        z_refined = r0.variables["z"][:]
        ds.close()

        assert np.all(np.isfinite(h))
        # z bounded by inset range
        assert z_refined[0] >= z_min - 1.0
        assert z_refined[-1] <= z_max + 1.0
        # x still periodic
        x_seam = np.abs(h[0, :, :] - h[-1, :, :]).mean()
        x_interior = np.abs(h[0, :, :] - h[1, :, :]).mean()
        assert x_seam < 10 * x_interior

    def test_refine_reproducibility(self, parent_nc):
        """Same explicit seed produces byte-identical h and qt fields."""
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        half_x = parent_nx // 2
        half_y = parent_ny // 2
        new_dx = k_finest / 4
        new_dy = k_finest / 4

        # Refine twice into different groups with the same explicit seed.
        refine(parent_nc, 0, half_x, 0, half_y, new_dx, new_dy, seed=43)
        refine(parent_nc, 0, half_x, 0, half_y, new_dx, new_dy, seed=43)

        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]
        r1 = ds.groups["refinements"].groups["r1"]
        h0 = r0.variables["h"][:]
        h1 = r1.variables["h"][:]
        qt0 = r0.variables["qt"][:]
        qt1 = r1.variables["qt"][:]
        ds.close()

        np.testing.assert_array_equal(h0, h1)
        np.testing.assert_array_equal(qt0, qt1)

    def test_refine_z_pad_clipped_near_ground(self, parent_nc):
        """Elevated inset with z_min just slightly above parent's ground:
        the desired z-pad-below (SUPPORT * k_z_0) far exceeds the available
        room; the code must clip to z_min - parent_z_min and not error."""
        ds = netCDF4.Dataset(parent_nc, "r")
        parent_nx = len(ds.dimensions["x"])
        parent_ny = len(ds.dimensions["y"])
        parent_dz_vals = ds.variables["dz"][:]
        parent_domain_height = float(ds.domain_height)
        k_finest = float(ds.variables["k_values"][:][-1])
        ds.close()

        half_x = parent_nx // 2
        half_y = parent_ny // 2
        new_dx = k_finest / 4
        new_dy = k_finest / 4

        # Pick z_min equal to the parent's first non-ground z-level: tiny room.
        # Parent z_profile starts at 0 so any small positive z_min is "elevated".
        parent_dz_typ = float(np.mean(parent_dz_vals))
        z_min = parent_dz_typ  # just one parent cell above ground
        z_max = 2 * parent_domain_height / 3

        # Should succeed (pad gets clipped to z_min).
        refine(
            parent_nc,
            x_start=0, x_stop=half_x,
            y_start=0, y_stop=half_y,
            dx=new_dx, dy=new_dy,
            z_min=z_min, z_max=z_max,
            seed=47,
        )

        ds = netCDF4.Dataset(parent_nc, "r")
        r0 = ds.groups["refinements"].groups["r0"]
        h = r0.variables["h"][:]
        assert np.all(np.isfinite(h))
        ds.close()
