"""NetCDF output writer for STEAM simulations."""

import numpy as np
import netCDF4
from pathlib import Path

from . import constants


def compression_kwargs(compress, chunksizes):
    """netCDF4 filter keywords for a float32 variable with these chunks.

    One place for the filter choice (steam.constants.output_compression) and
    for its two traps: ``shuffle=True`` is SILENTLY IGNORED by every non-zlib
    compressor, so the byte shuffle has to be asked for as ``blosc_shuffle``;
    and a chunk below ``output_compression_min_chunk_bytes`` is written raw,
    because blosc fails the write outright on chunks too small for its header
    (see the constant).
    """
    if not compress:
        return dict(compression=None)
    chunk_bytes = int(np.prod(chunksizes)) * 4
    if chunk_bytes < constants.output_compression_min_chunk_bytes:
        return dict(compression=None)
    return dict(compression=constants.output_compression,
                complevel=constants.output_complevel, blosc_shuffle=1)


def write_netcdf(
    output_path, h_3d, qt_3d,
    x_coords, y_coords, z_coords,
    h_profile, qt_profile, z_profile,
    k_values, k_z_values, C_h_k, C_qt_k,
    simulation_params,
    group=None,
    compress=None,
    flux_3d=None,
    h_pert_3d=None,
    qt_pert_3d=None,
    flux_state_3d=None,
):
    """Write STEAM simulation output to a NetCDF file.

    Parameters
    ----------
    output_path : str or Path
    h_3d : ndarray, shape (nx, ny, nz)
    qt_3d : ndarray, shape (nx, ny, nz)
    x_coords, y_coords, z_coords : 1D arrays of coordinate positions [m]
    h_profile, qt_profile, z_profile : 1D input profile arrays
    k_values, k_z_values : 1D arrays of scale classes
    C_h_k, C_qt_k : 1D arrays of scale-dependent amplitudes
    simulation_params : dict
        Simulation metadata. Contains scalar attributes and 1D arrays:
        - dz : 1D float32 array, shape (nz,) — cell heights
        - spheroscale : 1D float32 array, shape (nz,) — spheroscale profile on output grid
        Plus scalar keys: nx, ny, dx, dy, outer_scale, domain_height,
        profile_dz, sparsity_factors, surface_pressure, seed, C_h_L,
        C_qt_L, n_large_turbulons, H_h, H_z, h_min, h_max, qt_min,
        qt_max, min_distance_to_ground.
    group : str or None
        If None, write to the root of a new file. If provided, open the
        existing file in append mode and write into a NetCDF4 group of this
        name — how refine() stores a nest alongside its parent.
    compress : bool or None
        If True, write the 3D data variables with the
        ``steam.constants.output_compression`` filter. None (default) uses
        the module-level ``steam.constants.output_compress`` setting.

    Returns
    -------
    Path
        The output_path as a Path object.
    """
    if compress is None:
        compress = constants.output_compress

    output_path = Path(output_path)
    nx_final, ny_final, nz_final = h_3d.shape

    if group is not None:
        print(f"Writing NetCDF group '{group}' to {output_path} ...")
        ds_root = netCDF4.Dataset(output_path, "a", format="NETCDF4")
        ds = ds_root.createGroup(group)
    else:
        print(f"Writing NetCDF to {output_path} ...")
        ds_root = netCDF4.Dataset(output_path, "w", format="NETCDF4")
        ds = ds_root

    # Dimensions
    ds.createDimension("x", nx_final)
    ds.createDimension("y", ny_final)
    ds.createDimension("z", nz_final)
    ds.createDimension("z_profile", len(h_profile))
    ds.createDimension("k", len(k_values))

    # Coordinate variables
    x_var = ds.createVariable("x", "f4", ("x",))
    x_var[:] = x_coords
    x_var.units = "m"
    x_var.long_name = "x coordinate"

    y_var = ds.createVariable("y", "f4", ("y",))
    y_var[:] = y_coords
    y_var.units = "m"
    y_var.long_name = "y coordinate"

    z_var = ds.createVariable("z", "f4", ("z",))
    z_var[:] = z_coords
    z_var.units = "m"
    z_var.long_name = "z coordinate (height)"

    # Data variables — chunked, optionally compressed
    field_chunks = (min(64, nx_final), min(64, ny_final), nz_final)
    h_var = ds.createVariable(
        "h", "f4", ("x", "y", "z"), chunksizes=field_chunks,
        **compression_kwargs(compress, field_chunks),
    )
    h_var[:] = h_3d
    h_var.units = "J/kg"
    h_var.long_name = "moist static energy"

    qt_var = ds.createVariable(
        "qt", "f4", ("x", "y", "z"), chunksizes=field_chunks,
        **compression_kwargs(compress, field_chunks),
    )
    qt_var[:] = qt_3d
    qt_var.units = "kg/kg"
    qt_var.long_name = "total water mixing ratio"

    if flux_3d is not None:
        flux_var = ds.createVariable(
            "flux", "f4", ("x", "y", "z"), chunksizes=field_chunks,
            **compression_kwargs(compress, field_chunks),
        )
        flux_var[:] = flux_3d
        flux_var.units = "1"
        flux_var.long_name = "dimensionless conserved flux"

    # The perturbations as the cascade left them, BEFORE the mean profile was
    # added back. h = h_perturbation + <h>(z) in float32 is not invertible to
    # the last bit (the mean is ~1e5 times the perturbation), so a nest that
    # reconstructed the cascade state by subtracting the mean would start from
    # a field a rounding step away from the one its parent finished with.
    # Opt-in, because it doubles the file: pass save_for_refinement=True to
    # simulate() / refine() for runs intended as refinement parents.
    if h_pert_3d is not None:
        for name, field, units in (("h_perturbation", h_pert_3d, "J/kg"),
                                   ("qt_perturbation", qt_pert_3d, "kg/kg")):
            var = ds.createVariable(
                name, "f4", ("x", "y", "z"), chunksizes=field_chunks,
                **compression_kwargs(compress, field_chunks),
            )
            var[:] = field
            var.units = units
            var.long_name = f"{name} as the cascade left it (mean not added)"

    # The flux state the cascade left, before the interpolation-compensation
    # composition (the written `flux` is composed, like h and qt). A nest
    # continues from this, for the same reason as the perturbations above.
    if flux_state_3d is not None:
        var = ds.createVariable(
            "flux_state", "f4", ("x", "y", "z"), chunksizes=field_chunks,
            **compression_kwargs(compress, field_chunks),
        )
        var[:] = flux_state_3d
        var.units = "1"
        var.long_name = ("dimensionless conserved flux as the cascade left "
                         "it (no interpolation compensation)")

    # Profile variables. z_profile and spheroscale_profile are float64: a
    # nest rebuilds its grids from them, and a float32 round-trip of the
    # spheroscale would move every class's dz by a rounding step.
    zp_var = ds.createVariable("z_profile", "f8", ("z_profile",))
    zp_var[:] = z_profile
    zp_var.units = "m"

    hp_var = ds.createVariable("h_profile", "f4", ("z_profile",))
    hp_var[:] = h_profile
    hp_var.units = "J/kg"
    hp_var.long_name = "input h profile"

    qtp_var = ds.createVariable("qt_profile", "f4", ("z_profile",))
    qtp_var[:] = qt_profile
    qtp_var.units = "kg/kg"
    qtp_var.long_name = "input qt profile"

    if 'spheroscale_profile' in simulation_params:
        lsp_var = ds.createVariable("spheroscale_profile", "f8", ("z_profile",))
        lsp_var[:] = np.asarray(simulation_params['spheroscale_profile'], dtype=np.float64)
        lsp_var.units = "m"
        lsp_var.long_name = "spheroscale profile on input profile grid"

    # 1D scale arrays
    kv = ds.createVariable("k_values", "f4", ("k",))
    kv[:] = k_values.astype(np.float32)
    kv.units = "m"
    kv.long_name = "horizontal turbulon scale classes"

    kzv = ds.createVariable("k_z_values", "f4", ("k",))
    kzv[:] = k_z_values.astype(np.float32)
    kzv.units = "m"
    kzv.long_name = "vertical turbulon scale classes"

    # C_h_k / C_qt_k: always 2D (k × nz_k_max)
    nz_k_max = max(len(c) for c in C_h_k)
    ds.createDimension("nz_k_max", nz_k_max)

    chk = ds.createVariable("C_h_k", "f4", ("k", "nz_k_max"), fill_value=np.nan)
    for i, c in enumerate(C_h_k):
        chk[i, :len(c)] = c
    chk.long_name = "scale- and height-dependent h amplitude"

    cqtk = ds.createVariable("C_qt_k", "f4", ("k", "nz_k_max"), fill_value=np.nan)
    for i, c in enumerate(C_qt_k):
        cqtk[i, :len(c)] = c
    cqtk.long_name = "scale- and height-dependent qt amplitude"

    # Scalar attributes on root group
    p = simulation_params
    ds.nx = np.int32(p['nx'])
    ds.ny = np.int32(p['ny'])
    ds.dx = np.float64(p['dx'])
    ds.dy = np.float64(p['dy'])
    # dz: always 1D variable (cell heights per z-level)
    dz_var = ds.createVariable("dz", "f4", ("z",))
    dz_var[:] = np.asarray(p['dz'], dtype=np.float32)
    dz_var.units = "m"
    dz_var.long_name = "cell height"
    ds.outer_scale = np.float64(p['outer_scale'])
    # spheroscale: 1D variable (profile on output grid)
    ls_var = ds.createVariable("spheroscale", "f4", ("z",))
    ls_var[:] = np.asarray(p['spheroscale'], dtype=np.float32)
    ls_var.units = "m"
    ls_var.long_name = "spheroscale profile"
    ds.domain_height = np.float64(p['domain_height'])
    if 'domain_z_min' in p:
        ds.domain_z_min = np.float64(p['domain_z_min'])
    ds.profile_dz = np.float64(p['profile_dz'])
    ds.sparsity_factors = np.array(p['sparsity_factors'], dtype=np.int32)
    ds.surface_pressure = np.float64(p['surface_pressure'])
    ds.seed = np.int32(p['seed']) if p['seed'] is not None else -1
    ds.C_h_L = np.float64(p['C_h_L'])
    ds.C_qt_L = np.float64(p['C_qt_L'])
    ds.n_large_turbulons = np.int32(p['n_large_turbulons'])
    # float64: a nest recomputes its amplitude ladder from these, and a
    # float32 round-trip of the exponent would put its C_k a rounding step
    # off the ladder its parent used.
    ds.H_h = np.float64(p['H_h'])
    ds.H_z = np.float64(p['H_z'])
    ds.lambda_haar_to_mhat = np.float64(p['lambda_haar_to_mhat'])
    ds.h_min = np.float64(p['h_min'])
    ds.h_max = np.float64(p['h_max'])
    ds.qt_min = np.float64(p['qt_min'])
    ds.qt_max = np.float64(p['qt_max'])
    ds.min_distance_to_ground = np.int32(p['min_distance_to_ground'])
    if 'turbulon_shape' in p:
        ds.turbulon_shape = p['turbulon_shape']
    if 'anisotropy' in p:
        ds.anisotropy = p['anisotropy']
    if 'n_scale_classes_per_dyad' in p:
        ds.n_scale_classes_per_dyad = np.int32(p['n_scale_classes_per_dyad'])
    ds.flux_noise_scale = np.float64(p['flux_noise_scale'])
    ds.flux_alpha = np.float64(p['flux_alpha'])
    # Continuation bookkeeping: what a descendant nest needs in order to be
    # the SAME cascade carried further — the root's outer scale (the one
    # (k/L)^H_h ladder), the root's seed, and how many size classes of that
    # seed's per-class stream have already been drawn.
    ds.root_outer_scale = np.float64(p['root_outer_scale'])
    ds.root_seed = np.int32(p['root_seed']) if p['root_seed'] is not None else -1
    ds.n_classes_consumed = np.int32(p['n_classes_consumed'])

    # Refinement-specific attributes. periodic_x / periodic_y record whether
    # the group's own x / y axis wraps: a root always does, a nest only where
    # it spans a parent axis that itself wrapped. Absent means periodic, so
    # root files written before nesting existed still read correctly.
    for attr in ('parent_group', 'parent_x_slice', 'parent_y_slice',
                 'parent_x_offset', 'parent_y_offset',
                 'normalization_source'):
        if attr in p:
            val = p[attr]
            if isinstance(val, str):
                ds.setncattr(attr, val)
            else:
                ds.setncattr(attr, np.array(val))
    for attr in ('periodic_x', 'periodic_y'):
        if attr in p:
            ds.setncattr(attr, np.int8(p[attr]))

    # Optional 2D starting pressure for hydrostatic integration, written by
    # refine() when the nest's bottom is elevated above the parent ground.
    if 'p_bottom' in p:
        pb_var = ds.createVariable(
            "p_bottom", "f4", ("x", "y"),
            **compression_kwargs(compress, (nx_final, ny_final)),
        )
        pb_var[:] = np.asarray(p['p_bottom'], dtype=np.float32)
        pb_var.units = "Pa"
        pb_var.long_name = "starting pressure at nest bottom (z=z[0])"

    ds_root.close()
    print(f"Written {output_path}" + (f" (group '{group}')" if group else ""))
    return output_path


def write_class_increments(output_path, increment_dir, class_grids,
                           group=None, compress=None):
    """Append per-class added increments to an existing STEAM output file.

    Stores, for each size class j of the WHOLE root ladder, the h, qt and
    flux increments the cascade actually added (post bounded add; for the
    flux, the state difference across the advance) on that class's own
    grid, under ``class_increments/c{j:02d}``. Each class subgroup carries
    its own (x, y, z) dimensions, its z-coordinate array, and attributes
    k, dx, dy. Total size is a geometric pyramid, ~1.14x one output-grid
    field per stored field before compression.

    A nest stores the ladder whole -- the classes it inherited, carried
    through from its parent, followed by its own -- so that a nest of a
    nest re-weights exactly the same uniform ladder that a nest of a root
    does.

    Parameters
    ----------
    output_path : str or Path
        Existing NetCDF file written by write_netcdf.
    increment_dir : str or Path
        Directory of ``c{j:02d}_{h,qt}.npy`` files staged by the caller.
    class_grids : list of dict
        One per stored class, in ladder order, with keys k, dx, dy and z
        (the class's 1D z-coordinate array).
    group : str or None
        Parent group to place ``class_increments`` under (None = root).
    compress : bool or None
        compression as in write_netcdf.
    """
    if compress is None:
        compress = constants.output_compress

    increment_dir = Path(increment_dir)
    ds_root = netCDF4.Dataset(output_path, "a", format="NETCDF4")
    base = ds_root if group is None else ds_root[group]
    inc_root = base.createGroup("class_increments")

    for j, class_grid in enumerate(class_grids):
        sub = inc_root.createGroup(f"c{j:02d}")
        arrays = {name: np.load(increment_dir / f"c{j:02d}_{name}.npy")
                  for name in ("h", "qt", "flux")}
        nx_j, ny_j, nz_j = arrays["h"].shape
        sub.createDimension("x", nx_j)
        sub.createDimension("y", ny_j)
        sub.createDimension("z", nz_j)
        class_chunks = (min(64, nx_j), min(64, ny_j), nz_j)
        z_var = sub.createVariable("z", "f8", ("z",))
        z_var[:] = np.asarray(class_grid['z'], dtype=np.float64)
        z_var.units = "m"
        for name, units in (("h", "J/kg"), ("qt", "kg/kg"), ("flux", "1")):
            v = sub.createVariable(
                name, "f4", ("x", "y", "z"), chunksizes=class_chunks,
                **compression_kwargs(compress, class_chunks),
            )
            v[:] = arrays[name]
            v.units = units
            v.long_name = f"class {j} added {name} increment"
        sub.k = np.float64(class_grid['k'])
        sub.dx = np.float64(class_grid['dx'])
        sub.dy = np.float64(class_grid['dy'])

    ds_root.close()
    print(f"Written class_increments ({len(class_grids)} classes) to {output_path}")
    return output_path
