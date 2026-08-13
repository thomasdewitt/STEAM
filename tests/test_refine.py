"""Tests for nested subdomain refinement and per-class seeding."""

import importlib
import shutil

import numpy as np
import pytest
import netCDF4

sm = importlib.import_module("steam.simulate")

from steam.simulate import (
    simulate,
    refine,
    cascade_loop,
    _compute_all_grids,
    SUPPORT_FACTOR,
)
from steam.constants import hurst_horizontal as H_h
from steam.thermodynamics import compute_diagnostics
from steam.utils import zoom_trilinear


PARENT_NZ = 50
PARENT_PROFILE_DZ = 30.0
PARENT_DOMAIN_HEIGHT = 3000.0


def _profiles():
    z = np.arange(PARENT_NZ) * PARENT_PROFILE_DZ
    h = 340e3 - 20e3 * (z / z.max())
    qt = 0.018 - 0.016 * (z / z.max())
    return h, qt


@pytest.fixture(scope="module")
def parent_template(tmp_path_factory):
    """One parent simulation, built once and copied per test.

    compute_diagnostics is run so elevated-z nests can read the parent's 3D
    pressure for their 2D p_bottom.
    """
    h, qt = _profiles()
    out = tmp_path_factory.mktemp("parent") / "parent.nc"
    simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
             outer_scale=8000, spheroscale=100,
             domain_height=PARENT_DOMAIN_HEIGHT, profile_dz=PARENT_PROFILE_DZ,
             output_path=out, seed=42, save_for_refinement=True)
    compute_diagnostics(out)
    return out


@pytest.fixture
def parent_nc(parent_template, tmp_path):
    destination = tmp_path / "parent.nc"
    shutil.copy(parent_template, destination)
    return destination


def _parent_geometry(path):
    with netCDF4.Dataset(path, "r") as ds:
        return {
            'nx': len(ds.dimensions["x"]),
            'ny': len(ds.dimensions["y"]),
            'dx': float(ds.dx),
            'dy': float(ds.dy),
            'k_finest': float(ds.variables["k_values"][:][-1]),
        }


# ---------------------------------------------------------------------------
# Per-class seeding: a split cascade must reproduce a single run
# ---------------------------------------------------------------------------

def test_split_cascade_matches_full_cascade(monkeypatch):
    """Running the first M classes then the remaining N-M, handing over the
    perturbations AND the flux, is bit-identical to running all N at once.
    This is the invariant refine() relies on to continue a parent's cascade.

    The interpolation compensation is neutralized here: it is
    deliberately resolution-aware (each run damps its classes by
    f(k/dx) of ITS OWN output grid), so a split run compensates the
    shared classes differently than the full run does. That is the
    intended semantics — a parent's output is correct for the parent's
    grid — but it breaks bit-identity, so the machinery invariant is
    tested with the compensation switched off."""
    monkeypatch.setattr(sm, "INTERPOLATION_COMPENSATION", {})
    h_profile, qt_profile = _profiles()
    z_profile = np.arange(PARENT_NZ) * PARENT_PROFILE_DZ
    spheroscale_profile = np.full(PARENT_NZ, 100.0)
    sparsity_factors = (1, 1, 1)
    outer_scale, dx = 8000.0, 250.0

    n_classes = int(round(np.log(outer_scale / (2 * dx)) / np.log(2.0))) + 1
    k_values = outer_scale / 2.0 ** np.arange(n_classes)
    grids = _compute_all_grids(
        k_values, 32 * dx, 32 * dx, PARENT_DOMAIN_HEIGHT,
        sparsity_factors, spheroscale_profile, z_profile,
    )

    ones = [np.ones(int(nz), dtype=np.float32) for nz in grids["nz"]]
    tiny = [np.full(int(nz), 1e-30, dtype=np.float32) for nz in grids["nz"]]
    bounds = (315 * 1004.0, 355 * 1004.0, 0.0, 30 / 1000)

    def run(class_slice, seeds, h_pert=None, qt_pert=None, flux=None):
        sub = {key: (grids[key][class_slice] if key not in ('z_arrays', 'dz_arrays')
                     else grids[key][class_slice.start:class_slice.stop])
               for key in grids}
        return cascade_loop(
            h_profile, qt_profile, z_profile, sub,
            ones[class_slice], ones[class_slice],
            tiny[class_slice], tiny[class_slice], ones[class_slice],
            *bounds, 1, sparsity_factors, seeds,
            h_perturbation=h_pert, qt_perturbation=qt_pert, flux=flux,
        )

    all_seeds = np.random.SeedSequence(123).spawn(n_classes)
    h_full, qt_full, flux_full, _, _, _, _, _ = run(slice(0, n_classes), all_seeds)

    split = n_classes // 2
    seeds = np.random.SeedSequence(123).spawn(n_classes)
    h_1, qt_1, flux_1, _, _, _, _, _ = run(slice(0, split), seeds[:split])
    h_2, qt_2, flux_2, _, _, _, _, _ = run(slice(split, n_classes), seeds[split:],
                                           h_pert=h_1, qt_pert=qt_1, flux=flux_1)

    np.testing.assert_array_equal(h_2, h_full)
    np.testing.assert_array_equal(qt_2, qt_full)
    np.testing.assert_array_equal(flux_2, flux_full)


# ---------------------------------------------------------------------------
# Geometry and bookkeeping
# ---------------------------------------------------------------------------

def test_refine_creates_group_with_expected_structure(parent_nc):
    geometry = _parent_geometry(parent_nc)
    half_x = geometry['nx'] // 2
    half_y = geometry['ny'] // 2
    new_dx = geometry['k_finest'] / 4

    refine(parent_nc, 0, half_x, 0, half_y, new_dx, new_dx, seed=99)

    with netCDF4.Dataset(parent_nc, "r") as ds:
        r0 = ds.groups["refinements"].groups["r0"]
        assert np.all(np.isfinite(r0.variables["h"][:]))
        assert np.all(np.isfinite(r0.variables["qt"][:]))
        assert np.all(np.isfinite(r0.variables["flux"][:]))
        assert r0.parent_group == "/"
        assert list(r0.parent_x_slice) == [0, half_x]
        assert list(r0.parent_y_slice) == [0, half_y]
        # Non-spanning nest of a periodic root: the nest itself does not wrap.
        assert r0.periodic_x == 0 and r0.periodic_y == 0
        # Output covers the inner region only, at the requested resolution.
        assert abs(float(r0.dx) - new_dx) < 1e-6
        assert len(r0.dimensions["x"]) == round(half_x * geometry['dx'] / new_dx)
        # Every nest class is strictly finer than the parent's finest.
        assert np.all(np.asarray(r0.variables["k_values"][:]) < geometry['k_finest'])


def test_refine_auto_group_naming(parent_nc):
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    new_dx = geometry['k_finest'] / 4
    refine(parent_nc, 0, half, 0, half, new_dx, new_dx, seed=99)
    refine(parent_nc, 0, half, 0, half, new_dx, new_dx, seed=100)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        assert set(ds.groups["refinements"].groups) >= {"r0", "r1"}


def test_refine_reproducible_under_fixed_seed(parent_nc):
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    new_dx = geometry['k_finest'] / 4
    refine(parent_nc, 0, half, 0, half, new_dx, new_dx, seed=43)
    refine(parent_nc, 0, half, 0, half, new_dx, new_dx, seed=43)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        r0 = ds.groups["refinements"].groups["r0"]
        r1 = ds.groups["refinements"].groups["r1"]
        np.testing.assert_array_equal(r0.variables["h"][:], r1.variables["h"][:])
        np.testing.assert_array_equal(r0.variables["qt"][:], r1.variables["qt"][:])
        np.testing.assert_array_equal(r0.variables["flux"][:], r1.variables["flux"][:])


def test_refine_auto_seed_continues_the_root_stream(parent_nc):
    """seed=None continues the root's per-class seed stream, so a nest depends
    on nothing but the root seed and the class index — in particular not on
    the output group name, and not on anything salted per process."""
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    new_dx = geometry['k_finest'] / 4
    refine(parent_nc, 0, half, 0, half, new_dx, new_dx, output_group="a")
    refine(parent_nc, 0, half, 0, half, new_dx, new_dx, output_group="b")
    with netCDF4.Dataset(parent_nc, "r") as ds:
        a = np.asarray(ds.groups["a"].variables["h"][:])
        b = np.asarray(ds.groups["b"].variables["h"][:])
        assert int(ds.groups["a"].n_classes_consumed) > int(ds.n_classes_consumed)
    np.testing.assert_array_equal(a, b)


def test_refine_tiny_inner_keeps_grid_bounded(parent_nc):
    """A nest one outer-scale wide must produce a small output grid: the halo
    shrinks with the size class, so it never blows up the finest grid."""
    geometry = _parent_geometry(parent_nc)
    inner_cells = int(round(geometry['k_finest'] / geometry['dx']))
    x_start = geometry['nx'] // 2
    new_dx = geometry['k_finest'] / 8
    refine(parent_nc, x_start, x_start + inner_cells,
           x_start, x_start + inner_cells, new_dx, new_dx, seed=23)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        r0 = ds.groups["refinements"].groups["r0"]
        assert len(r0.dimensions["x"]) <= 16


# ---------------------------------------------------------------------------
# Inheritance: the nest starts from the parent's field, not from scratch
# ---------------------------------------------------------------------------

def test_nest_with_zero_amplitude_reproduces_the_parent_state(parent_nc):
    """With the turbulon amplitudes set to zero the nest adds nothing, so its
    cascade state must be exactly the parent's state interpolated onto the
    nest grid. That isolates the inheritance path (extraction, zoom, trim)
    from the cascade itself.

    A doubly-spanning nest is used so there is no halo to complicate the
    comparison. The nest re-derives its amplitude ladder from the parent's
    mean profile and lambda, so zeroing lambda zeroes every nest class.
    """
    with netCDF4.Dataset(parent_nc, "r+") as ds:
        ds.lambda_haar_to_mhat = np.float64(0.0)
    geometry = _parent_geometry(parent_nc)
    new_dx = geometry['k_finest'] / 4

    refine(parent_nc, 0, geometry['nx'], 0, geometry['ny'], new_dx, new_dx,
           seed=1, save_for_refinement=True)

    with netCDF4.Dataset(parent_nc, "r") as ds:
        parent_state = np.asarray(ds.variables["h_perturbation"][:],
                                  dtype=np.float32)
        z_profile = np.asarray(ds.variables["z_profile"][:], dtype=np.float64)
        h_profile = np.asarray(ds.variables["h_profile"][:])
        h_min, h_max = float(ds.h_min), float(ds.h_max)
        r0 = ds.groups["refinements"].groups["r0"]
        nest_state = np.asarray(r0.variables["h_perturbation"][:],
                                dtype=np.float32)
        nest_z = np.asarray(r0.variables["z"][:], dtype=np.float64)

    expected = zoom_trilinear(parent_state, nest_state.shape)
    # Levels the interpolated field does not press against a bound are pure
    # interpolation. On the rest the per-class add still enforces the bounds
    # pointwise, so a cell the finer mean profile pushed outside is pulled
    # back -- correctly, and only ever toward the bound.
    field = expected + np.interp(nest_z, z_profile, h_profile)[None, None, :]
    interior = ((field.min(axis=(0, 1)) > h_min + 1.0)
                & (field.max(axis=(0, 1)) < h_max - 1.0))
    assert interior.sum() > 5      # not a vacuous comparison
    np.testing.assert_allclose(nest_state[:, :, interior],
                               expected[:, :, interior], rtol=1e-6, atol=1e-3)

def test_nest_amplitude_ladder_continues_the_parent(parent_nc):
    """C_{Phi,k} must continue the parent's ladder with no step at the overlap
    scale: C_nest(k) = C_parent(k_finest) * (k/k_finest)^H_h."""
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    refine(parent_nc, 0, half, 0, half,
           geometry['k_finest'] / 8, geometry['k_finest'] / 8, seed=3)

    with netCDF4.Dataset(parent_nc, "r") as ds:
        parent_z = np.asarray(ds.variables["z"][:])
        parent_C = np.asarray(ds.variables["C_h_k"][:])[-1]
        r0 = ds.groups["refinements"].groups["r0"]
        nest_z = np.asarray(r0.variables["z"][:])
        nest_C = np.asarray(r0.variables["C_h_k"][:])
        nest_k = np.asarray(r0.variables["k_values"][:])

    anchor = np.interp(nest_z, parent_z, parent_C)
    for i, k in enumerate(nest_k):
        predicted = anchor * (k / geometry['k_finest']) ** H_h
        # Levels where the mean profile is flat carry C ~ 0; their relative
        # error is meaningless, so compare where there is amplitude to compare.
        structured = predicted > 0.02 * predicted.max()
        # The stored table passes through each class's own vertical grid on
        # the way out, so a coarse class picks up interpolation error against
        # this direct parent -> output-grid prediction; the level-mean ratio
        # is free of that and pins the (k/L)^H_h law itself.
        np.testing.assert_allclose(
            nest_C[i][:len(nest_z)][structured], predicted[structured], rtol=5e-2)
        np.testing.assert_allclose(
            nest_C[i][:len(nest_z)][structured].mean(), predicted[structured].mean(),
            rtol=5e-3)


# ---------------------------------------------------------------------------
# Flux
# ---------------------------------------------------------------------------

def test_nest_flux_preserves_the_inherited_entering_mean(parent_nc):
    """Each class restores the volume mean the flux entered it with, so a
    nest's flux keeps the regional anomaly it inherited from the parent
    instead of being scrubbed back to one."""
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    refine(parent_nc, 0, half, 0, half,
           geometry['k_finest'] / 4, geometry['k_finest'] / 4, seed=5)

    with netCDF4.Dataset(parent_nc, "r") as ds:
        parent_flux = np.asarray(ds.variables["flux"][:], dtype=np.float64)
        nest_flux = np.asarray(
            ds.groups["refinements"].groups["r0"].variables["flux"][:],
            dtype=np.float64)

    inherited = parent_flux[:half, :half, :].mean()
    # Interpolation onto the nest's grid perturbs the mean slightly; the
    # cascade itself must not move it at all.
    #
    # The tolerance covers realization spread, not just the regrid. This
    # drift is a property of the PARENT realization (it is identical across
    # refine seeds), and over parent seeds 42-47 it measured +0.225, +0.465,
    # -0.256, -0.171, +0.425, +0.089 percent. The former rtol=5e-3 therefore
    # passed on luck: seed 42 sits at +0.225% but seed 43 clears the bound by
    # 0.035%, and any bit-level change to the cascade -- e.g. the 2026-07-28
    # SUPPORT_FACTOR 5 -> 3 truncation, whose convolution moves only 2.8e-7
    # but whose clip-and-renormalize amplifies that to a different
    # realization -- reshuffles which end of the spread this seed lands on.
    #
    # 2026-08-13, min_distance_to_ground default 1 -> 0: the same measurement
    # over parent seeds 42-47 now gives +2.233, +0.814, -0.872, +0.677, +0.002,
    # +0.064 percent -- max |drift| 2.233% against 0.465% before, so allowing
    # centers at the ground and top widened the spread about fivefold rather
    # than merely reshuffling it. Plausibly because a boundary turbulon is
    # truncated by the vertical zero padding, and the nest's own z grid
    # truncates it differently than the parent's did, so the inherited mean has
    # more room to move across the regrid.
    #
    # rtol is therefore 5e-2, chosen against both bounds rather than to make
    # this seed pass: 2.2x the widest drift measured, and still 3x below the
    # ~15% error that "scrubbed back to one" would produce for this parent
    # (inherited 1.175). The robust form of the claim, which needs no tolerance
    # at all, is test_nest_flux_anomaly_is_not_scrubbed_to_one below.
    np.testing.assert_allclose(nest_flux.mean(), inherited, rtol=5e-2)
    assert np.all(nest_flux >= 0.0)


def test_nest_flux_anomaly_is_not_scrubbed_to_one(parent_nc):
    """A nest carved from a genuinely quiet or busy region must come out
    quiet or busy — not renormalized to the global mean of one."""
    geometry = _parent_geometry(parent_nc)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        parent_flux = np.asarray(ds.variables["flux"][:], dtype=np.float64)
    quarter = geometry['nx'] // 4
    # Pick the quarter-tile whose flux mean is furthest from one.
    corners = [(i, j) for i in range(4) for j in range(4)]
    x_index, y_index = max(corners, key=lambda c: abs(parent_flux[
        c[0] * quarter:(c[0] + 1) * quarter,
        c[1] * quarter:(c[1] + 1) * quarter].mean() - 1.0))
    x_start, y_start = x_index * quarter, y_index * quarter
    inherited = parent_flux[x_start:x_start + quarter,
                            y_start:y_start + quarter].mean()
    assert abs(inherited - 1.0) > 0.02, "parent region is too close to the mean"

    refine(parent_nc, x_start, x_start + quarter, y_start, y_start + quarter,
           geometry['k_finest'] / 4, geometry['k_finest'] / 4, seed=6)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        nest_flux = np.asarray(
            ds.groups["refinements"].groups["r0"].variables["flux"][:],
            dtype=np.float64)

    # Much closer to the inherited anomaly than to one.
    assert abs(nest_flux.mean() - inherited) < 0.25 * abs(inherited - 1.0)


def test_refine_requires_a_parent_flux_state(parent_nc):
    with netCDF4.Dataset(parent_nc, "a") as ds:
        ds.renameVariable("flux_state", "flux_state_disabled")
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    with pytest.raises(ValueError, match="no flux_state field"):
        refine(parent_nc, 0, half, 0, half,
               geometry['k_finest'] / 4, geometry['k_finest'] / 4)


# ---------------------------------------------------------------------------
# Periodicity: spanning axes keep the parent's period, others do not wrap
# ---------------------------------------------------------------------------

def _seam_ratio(field, axis):
    first = np.take(field, 0, axis=axis)
    second = np.take(field, 1, axis=axis)
    last = np.take(field, -1, axis=axis)
    return (np.abs(first - last).mean() / np.abs(first - second).mean())


def test_spanning_strip_is_periodic_along_the_spanned_axis(parent_nc):
    """A long thin strip spanning the parent in x keeps period = parent extent
    in x (halo width zero there), so the x seam is no worse than the field's
    own cell-to-cell variation."""
    geometry = _parent_geometry(parent_nc)
    new_dx = geometry['k_finest'] / 4
    refine(parent_nc, 0, geometry['nx'], 0, geometry['ny'] // 4,
           new_dx, new_dx, seed=7)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        r0 = ds.groups["refinements"].groups["r0"]
        h = np.asarray(r0.variables["h"][:])
        assert r0.periodic_x == 1 and r0.periodic_y == 0
        # No halo in x: the output spans the parent's full x extent.
        assert len(r0.dimensions["x"]) == round(
            geometry['nx'] * geometry['dx'] / new_dx)
    assert _seam_ratio(h, axis=0) < 10.0


def test_spanning_y_is_periodic_along_y(parent_nc):
    geometry = _parent_geometry(parent_nc)
    new_dx = geometry['k_finest'] / 4
    refine(parent_nc, 0, geometry['nx'] // 4, 0, geometry['ny'],
           new_dx, new_dx, seed=31)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        r0 = ds.groups["refinements"].groups["r0"]
        h = np.asarray(r0.variables["h"][:])
        assert r0.periodic_x == 0 and r0.periodic_y == 1
    assert _seam_ratio(h, axis=1) < 10.0


def test_doubly_spanning_nest_is_periodic_in_both(parent_nc):
    geometry = _parent_geometry(parent_nc)
    new_dx = geometry['k_finest'] / 4
    refine(parent_nc, 0, geometry['nx'], 0, geometry['ny'], new_dx, new_dx, seed=37)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        h = np.asarray(
            ds.groups["refinements"].groups["r0"].variables["h"][:])
    assert np.all(np.isfinite(h))
    assert _seam_ratio(h, axis=0) < 10.0
    assert _seam_ratio(h, axis=1) < 10.0


# ---------------------------------------------------------------------------
# Recursive nesting and non-periodic parents
# ---------------------------------------------------------------------------

def _nest_geometry(path, group):
    with netCDF4.Dataset(path, "r") as ds:
        grp = ds[group]
        return {
            'nx': len(grp.dimensions["x"]),
            'ny': len(grp.dimensions["y"]),
            'dx': float(grp.dx),
            'dy': float(grp.dy),
            'k_finest': float(grp.variables["k_values"][:][-1]),
        }


def test_recursive_refinement_inside_a_nest(parent_nc):
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    refine(parent_nc, 0, half, 0, half,
           geometry['k_finest'] / 4, geometry['k_finest'] / 4, seed=11,
           save_for_refinement=True)

    r0 = _nest_geometry(parent_nc, "refinements/r0")
    pad_cells = int(np.ceil(SUPPORT_FACTOR * r0['k_finest'] / r0['dx']))
    inner_cells = int(round(r0['k_finest'] / r0['dx']))
    start = max(pad_cells, r0['nx'] // 4)
    assert start + inner_cells <= r0['nx'] - pad_cells

    refine(parent_nc, start, start + inner_cells, start, start + inner_cells,
           r0['dx'] / 2, r0['dy'] / 2, parent_group="refinements/r0", seed=13)

    with netCDF4.Dataset(parent_nc, "r") as ds:
        r1 = ds.groups["refinements"].groups["r1"]
        assert np.all(np.isfinite(r1.variables["h"][:]))
        assert r1.parent_group == "refinements/r0"
        # The ladder is anchored on the ROOT outer scale throughout, so a
        # grandchild's classes are still strictly finer than its parent's.
        assert np.all(np.asarray(r1.variables["k_values"][:]) < r0['k_finest'])


def test_boundary_touching_nest_of_a_nest_is_rejected(parent_nc):
    """A non-periodic parent cannot supply a halo by wrapping, so a nest that
    would reach past its edge must be refused rather than silently wrapped."""
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    refine(parent_nc, 0, half, 0, half,
           geometry['k_finest'] / 4, geometry['k_finest'] / 4, seed=17,
           save_for_refinement=True)
    r0 = _nest_geometry(parent_nc, "refinements/r0")
    inner_cells = int(round(r0['k_finest'] / r0['dx']))
    with pytest.raises(ValueError, match="non-periodic parent"):
        refine(parent_nc, 0, inner_cells, 0, inner_cells,
               r0['dx'] / 2, r0['dy'] / 2,
               parent_group="refinements/r0", seed=19)


def test_spanning_a_non_periodic_parent_axis_is_rejected(parent_nc):
    """Spanning leaves no room for a halo at all; on a non-periodic axis that
    cannot be papered over by wrapping."""
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    refine(parent_nc, 0, half, 0, half,
           geometry['k_finest'] / 4, geometry['k_finest'] / 4, seed=17,
           save_for_refinement=True)
    r0 = _nest_geometry(parent_nc, "refinements/r0")
    with pytest.raises(ValueError, match="non-periodic parent"):
        refine(parent_nc, 0, r0['nx'], 0, r0['ny'],
               r0['dx'] / 2, r0['dy'] / 2,
               parent_group="refinements/r0", seed=21)


# ---------------------------------------------------------------------------
# Vertical insets
# ---------------------------------------------------------------------------

def test_refine_z_limited(parent_nc):
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    new_dx = geometry['k_finest'] / 4
    z_min = PARENT_DOMAIN_HEIGHT / 3
    z_max = 2 * PARENT_DOMAIN_HEIGHT / 3

    refine(parent_nc, 0, half, 0, half, new_dx, new_dx,
           z_min=z_min, z_max=z_max, seed=99)

    with netCDF4.Dataset(parent_nc, "r") as ds:
        r0 = ds.groups["refinements"].groups["r0"]
        assert abs(float(r0.domain_height) - (z_max - z_min)) < 1.0
        assert abs(float(r0.domain_z_min) - z_min) < 1.0
        z = np.asarray(r0.variables["z"][:])
        assert z[0] >= z_min - 1.0 and z[-1] <= z_max + 1.0
        assert np.all(np.isfinite(r0.variables["h"][:]))
        # Elevated bottom: a 2D starting pressure must be carried over.
        assert "p_bottom" in r0.variables


def test_refine_z_range_outside_parent_raises(parent_nc):
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    new_dx = geometry['k_finest'] / 4
    with pytest.raises(ValueError, match="exceeds parent"):
        refine(parent_nc, 0, half, 0, half, new_dx, new_dx,
               z_min=0, z_max=PARENT_DOMAIN_HEIGHT + 500, seed=99)


def test_refine_z_pad_clipped_near_ground(parent_nc):
    """An inset whose bottom sits one parent cell above ground asks for far
    more z-halo than exists; it must clip rather than fail."""
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    new_dx = geometry['k_finest'] / 4
    with netCDF4.Dataset(parent_nc, "r") as ds:
        z_min = float(np.mean(ds.variables["dz"][:]))
    refine(parent_nc, 0, half, 0, half, new_dx, new_dx,
           z_min=z_min, z_max=2 * PARENT_DOMAIN_HEIGHT / 3, seed=47)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        assert np.all(np.isfinite(
            ds.groups["refinements"].groups["r0"].variables["h"][:]))


def test_elevated_nest_gets_turbulons_centered_outside_it(parent_nc):
    """An interior inset must allow centers below its z_min and above its
    z_max, contributing through their tails. Checked indirectly: changing the
    seed must move the inset's edge slices about as much as its middle."""
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    new_dx = geometry['k_finest'] / 4
    z_min = PARENT_DOMAIN_HEIGHT / 3
    z_max = 2 * PARENT_DOMAIN_HEIGHT / 3

    refine(parent_nc, 0, half, 0, half, new_dx, new_dx,
           z_min=z_min, z_max=z_max, seed=101)
    refine(parent_nc, 0, half, 0, half, new_dx, new_dx,
           z_min=z_min, z_max=z_max, seed=202)

    with netCDF4.Dataset(parent_nc, "r") as ds:
        h0 = np.asarray(ds.groups["refinements"].groups["r0"].variables["h"][:])
        h1 = np.asarray(ds.groups["refinements"].groups["r1"].variables["h"][:])

    middle = h0.shape[2] // 2
    mid_spread = np.abs(h0[:, :, middle] - h1[:, :, middle]).mean()
    for level, name in ((0, "bottom"), (-1, "top")):
        edge_spread = np.abs(h0[:, :, level] - h1[:, :, level]).mean()
        assert edge_spread > 0.1 * mid_spread, (
            f"{name}-slice seed spread {edge_spread:.3e} is far below the "
            f"mid-slice {mid_spread:.3e}; turbulons centered outside the "
            f"inset may not be reaching in"
        )


def test_spanning_x_with_interior_z(parent_nc):
    """Spanning halo-free x composed with an elevated z inset."""
    geometry = _parent_geometry(parent_nc)
    new_dx = geometry['k_finest'] / 4
    z_min = PARENT_DOMAIN_HEIGHT / 3
    z_max = 2 * PARENT_DOMAIN_HEIGHT / 3
    refine(parent_nc, 0, geometry['nx'], 0, geometry['ny'] // 4, new_dx, new_dx,
           z_min=z_min, z_max=z_max, seed=41)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        r0 = ds.groups["refinements"].groups["r0"]
        h = np.asarray(r0.variables["h"][:])
        z = np.asarray(r0.variables["z"][:])
    assert np.all(np.isfinite(h))
    assert z[0] >= z_min - 1.0 and z[-1] <= z_max + 1.0
    assert _seam_ratio(h, axis=0) < 10.0


# ---------------------------------------------------------------------------
# Bounds and anisotropy carry through the nest
# ---------------------------------------------------------------------------

def test_nest_respects_physical_bounds(parent_nc):
    geometry = _parent_geometry(parent_nc)
    half = geometry['nx'] // 2
    refine(parent_nc, 0, half, 0, half,
           geometry['k_finest'] / 8, geometry['k_finest'] / 8, seed=53)
    with netCDF4.Dataset(parent_nc, "r") as ds:
        r0 = ds.groups["refinements"].groups["r0"]
        h = np.asarray(r0.variables["h"][:])
        qt = np.asarray(r0.variables["qt"][:])
        assert h.min() >= float(r0.h_min) - 1e-3
        assert h.max() <= float(r0.h_max) + 1e-3
        assert qt.min() >= float(r0.qt_min) - 1e-9
        assert qt.max() <= float(r0.qt_max) + 1e-9


def test_nest_inherits_and_can_override_anisotropy(parent_nc):
    geometry = _parent_geometry(parent_nc)
    quarter = geometry['nx'] // 4
    # Deep enough that the finest nest classes fall below the 100 m
    # spheroscale, which is where the piecewise option differs at all.
    new_dx = geometry['k_finest'] / 16
    refine(parent_nc, 0, quarter, 0, quarter, new_dx, new_dx, seed=59)
    refine(parent_nc, 0, quarter, 0, quarter, new_dx, new_dx, seed=59,
           anisotropy='canonical')
    with netCDF4.Dataset(parent_nc, "r") as ds:
        r0 = ds.groups["refinements"].groups["r0"]
        r1 = ds.groups["refinements"].groups["r1"]
        # r0 inherits the parent's anisotropy (the piecewise package
        # default, 2026-08-06 ruling); r1 overrides it.
        assert r0.anisotropy == "piecewise_isotropic_below_spheroscale"
        assert r1.anisotropy == "canonical"
        # The nests' classes lie below the 100 m spheroscale, which is
        # where the two options differ at all: the inherited piecewise
        # nest has k_z = k there, the canonical override does not.
        for grp, expect_piecewise in ((r0, True), (r1, False)):
            k_values = np.asarray(grp.variables["k_values"][:])
            k_z_values = np.asarray(grp.variables["k_z_values"][:])
            below = k_values < 100.0
            assert below.any()
            if expect_piecewise:
                np.testing.assert_allclose(k_z_values[below],
                                           k_values[below], rtol=1e-5)
            else:
                expected = 100.0 * (k_values[below] / 100.0) ** (5.0 / 9.0)
                np.testing.assert_allclose(k_z_values[below], expected,
                                           rtol=1e-5)


# ---------------------------------------------------------------------------
# Per-class increment storage + nest compensation re-scaling (2026-07-28)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def parent_with_increments(tmp_path_factory):
    h, qt = _profiles()
    out = tmp_path_factory.mktemp("parent_inc") / "parent.nc"
    simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
             outer_scale=8000, spheroscale=100,
             domain_height=PARENT_DOMAIN_HEIGHT, profile_dz=PARENT_PROFILE_DZ,
             output_path=out, seed=42, save_for_refinement=True)
    compute_diagnostics(out)
    return out


def test_class_increments_replay_to_the_cascade_state(parent_with_increments):
    """Sum of stored increments, replayed down the regrid chain, is the STATE.

    The cascade adds each class's increment on its own working grid and
    the between-class regrids are linear, so replaying every stored
    increment through the same chain and summing must reproduce the final
    perturbation the loop left (up to float32 accumulation order). It is
    the state, not the written h: h carries the interpolation compensation
    and the output projection on top.
    """
    with netCDF4.Dataset(parent_with_increments) as ds:
        state = np.asarray(ds.variables["h_perturbation"][:], dtype=np.float64)
        z_profile = np.asarray(ds.variables["z_profile"][:], dtype=np.float64)
        ls_profile = np.asarray(ds.variables["spheroscale_profile"][:],
                                dtype=np.float64)
        k_values = np.asarray(ds.variables["k_values"][:], dtype=np.float64)
        nx, ny = int(ds.nx), int(ds.ny)
        dx, dy = float(ds.dx), float(ds.dy)
        height = float(ds.domain_height)
        sparsity = tuple(int(v) for v in ds.sparsity_factors)
        anisotropy = ds.anisotropy
        n_par = len(k_values)
        incs = [np.asarray(ds["class_increments"][f"c{j:02d}"]
                           .variables["h"][:], dtype=np.float32)
                for j in range(n_par)]

    grids = _compute_all_grids(k_values, nx * dx, ny * dy, height,
                               sparsity, ls_profile, z_profile,
                               anisotropy=anisotropy)
    total = np.zeros_like(incs[-1], dtype=np.float64)
    for j, inc in enumerate(incs):
        cur = inc
        for m in range(j + 1, n_par):
            cur = zoom_trilinear(cur, (int(grids['nx'][m]),
                                       int(grids['ny'][m]),
                                       int(grids['nz'][m])))
        total += cur

    scale = np.abs(state).mean()
    assert scale > 0
    assert np.abs(total - state).max() < 5e-3 * scale + 1e-2


def test_inherited_classes_are_compensated_up_at_the_nest_grid():
    """An inherited class is further from a nest's output grid than from its
    parent's, so it needs MORE compensation there -- the seam octaves, which
    a nest would otherwise under-represent. Pure function check on the
    factor itself."""
    z = [np.zeros(4)]
    ls, zp = np.full(4, 1e9), np.arange(4, dtype=float)
    parent_ladder = 8000.0 / 2.0 ** np.arange(4)
    nest_ladder = 8000.0 / 2.0 ** np.arange(4, 7)
    at_parent = sm._compensation_profiles(parent_ladder, z, ls, zp, 'canonical')
    at_nest = sm._compensation_profiles(
        np.concatenate([parent_ladder, nest_ladder]), z, ls, zp, 'canonical')
    assert at_nest[0][0] > at_parent[0][0]


def test_nest_output_is_the_state_damped_and_bounded(parent_with_increments,
                                                     tmp_path):
    """The written nest field is its state plus the compensation deficit,
    projected onto the bounds. f <= 1, so the composition carries LESS
    amplitude than the raw state, and it respects the bounds."""
    src = tmp_path / "parent_inc.nc"
    shutil.copy(parent_with_increments, src)
    refine(src, x_start=8, x_stop=24, y_start=8, y_stop=24, dx=125, dy=125,
           seed=7, save_for_refinement=True)

    with netCDF4.Dataset(src) as ds:
        grp = ds["refinements/r0"]
        h = np.asarray(grp.variables["h"][:], dtype=np.float64)
        qt = np.asarray(grp.variables["qt"][:], dtype=np.float64)
        state = np.asarray(grp.variables["h_perturbation"][:], dtype=np.float64)
        h_min, h_max = float(grp.h_min), float(grp.h_max)
        qt_min, qt_max = float(grp.qt_min), float(grp.qt_max)

    output_pert = h - h.mean(axis=(0, 1))[None, None, :]
    assert np.std(output_pert) < np.std(state - state.mean(axis=(0, 1))[None, None, :])
    assert h.min() >= h_min - 1e-3 and h.max() <= h_max + 1e-3
    assert qt.min() >= qt_min - 1e-9 and qt.max() <= qt_max + 1e-9


def test_refine_inherits_parent_flux_scale(tmp_path):
    """A nest's new classes use the PARENT's flux amplitude, not the global.

    Regression for the 2026-08-06 fix: refine read the module FLUX_SCALE at
    call time, which is how the production render nests once ran c=0.2085
    against a c=0.1419 parent. Two refinements of the same parent, one with
    the global left alone and one with it set to an absurd value, must be
    identical -- and both must record the parent's value.
    """
    h, qt = _profiles()
    parent = tmp_path / "parent.nc"
    original = sm.FLUX_SCALE
    try:
        sm.FLUX_SCALE = 0.11
        simulate(h, qt, nx=32, ny=32, dx=250, dy=250,
                 outer_scale=8000, spheroscale=100,
                 domain_height=PARENT_DOMAIN_HEIGHT,
                 profile_dz=PARENT_PROFILE_DZ,
                 output_path=parent, seed=7, save_for_refinement=True)
        refine(parent, 0, 16, 0, 16, 125.0, 125.0,
               output_group="refinements/control")
        sm.FLUX_SCALE = 0.99
        refine(parent, 0, 16, 0, 16, 125.0, 125.0,
               output_group="refinements/mutated")
    finally:
        sm.FLUX_SCALE = original
    with netCDF4.Dataset(parent) as ds:
        control = ds.groups["refinements"].groups["control"]
        mutated = ds.groups["refinements"].groups["mutated"]
        assert float(control.flux_noise_scale) == 0.11
        assert float(mutated.flux_noise_scale) == 0.11
        np.testing.assert_array_equal(control.variables["qt"][:],
                                      mutated.variables["qt"][:])
        np.testing.assert_array_equal(control.variables["flux"][:],
                                      mutated.variables["flux"][:])
