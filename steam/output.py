"""NetCDF output writer for STEAM simulations."""

import numpy as np
import netCDF4
from pathlib import Path

from . import constants


def write_netcdf(
    output_path, h_3d, qt_3d,
    x_coords, y_coords, z_coords,
    h_profile, qt_profile, z_profile,
    k_values, k_z_values, C_h_k, C_qt_k,
    simulation_params,
    compress=None,
    flux_3d=None,
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
    compress : bool or None
        If True, write the 3D data variables with
        zlib compression at complevel=4. None (default) uses the
        module-level ``steam.constants.output_compress`` setting.

    Returns
    -------
    Path
        The output_path as a Path object.
    """
    if compress is None:
        compress = constants.output_compress

    output_path = Path(output_path)
    nx_final, ny_final, nz_final = h_3d.shape

    print(f"Writing NetCDF to {output_path} ...")
    ds = netCDF4.Dataset(output_path, "w", format="NETCDF4")

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
    h_var = ds.createVariable(
        "h", "f4", ("x", "y", "z"), zlib=compress, complevel=4 if compress else 0,
        chunksizes=(min(64, nx_final), min(64, ny_final), nz_final),
    )
    h_var[:] = h_3d
    h_var.units = "J/kg"
    h_var.long_name = "moist static energy"

    qt_var = ds.createVariable(
        "qt", "f4", ("x", "y", "z"), zlib=compress, complevel=4 if compress else 0,
        chunksizes=(min(64, nx_final), min(64, ny_final), nz_final),
    )
    qt_var[:] = qt_3d
    qt_var.units = "kg/kg"
    qt_var.long_name = "total water mixing ratio"

    if flux_3d is not None:
        flux_var = ds.createVariable(
            "flux", "f4", ("x", "y", "z"), zlib=compress, complevel=4 if compress else 0,
            chunksizes=(min(64, nx_final), min(64, ny_final), nz_final),
        )
        flux_var[:] = flux_3d
        flux_var.units = "1"
        flux_var.long_name = "dimensionless conserved flux (horizontal mean 1)"

    # Profile variables
    zp_var = ds.createVariable("z_profile", "f4", ("z_profile",))
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
        lsp_var = ds.createVariable("spheroscale_profile", "f4", ("z_profile",))
        lsp_var[:] = np.asarray(simulation_params['spheroscale_profile'], dtype=np.float32)
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
    ds.dx = np.float32(p['dx'])
    ds.dy = np.float32(p['dy'])
    # dz: always 1D variable (cell heights per z-level)
    dz_var = ds.createVariable("dz", "f4", ("z",))
    dz_var[:] = np.asarray(p['dz'], dtype=np.float32)
    dz_var.units = "m"
    dz_var.long_name = "cell height"
    ds.outer_scale = np.float32(p['outer_scale'])
    # spheroscale: 1D variable (profile on output grid)
    ls_var = ds.createVariable("spheroscale", "f4", ("z",))
    ls_var[:] = np.asarray(p['spheroscale'], dtype=np.float32)
    ls_var.units = "m"
    ls_var.long_name = "spheroscale profile"
    ds.domain_height = np.float32(p['domain_height'])
    ds.profile_dz = np.float32(p['profile_dz'])
    ds.sparsity_factors = np.array(p['sparsity_factors'], dtype=np.int32)
    ds.surface_pressure = np.float32(p['surface_pressure'])
    ds.seed = np.int32(p['seed']) if p['seed'] is not None else -1
    ds.C_h_L = np.float32(p['C_h_L'])
    ds.C_qt_L = np.float32(p['C_qt_L'])
    ds.n_large_turbulons = np.int32(p['n_large_turbulons'])
    ds.H_h = np.float32(p['H_h'])
    ds.H_z = np.float32(p['H_z'])
    ds.h_min = np.float32(p['h_min'])
    ds.h_max = np.float32(p['h_max'])
    ds.qt_min = np.float32(p['qt_min'])
    ds.qt_max = np.float32(p['qt_max'])
    ds.min_distance_to_ground = np.int32(p['min_distance_to_ground'])
    if 'turbulon_shape' in p:
        ds.turbulon_shape = p['turbulon_shape']
    if 'anisotropy' in p:
        ds.anisotropy = p['anisotropy']
    if 'n_scale_classes_per_dyad' in p:
        ds.n_scale_classes_per_dyad = np.int32(p['n_scale_classes_per_dyad'])
    ds.flux_noise_scale = np.float32(p['flux_noise_scale'])
    ds.flux_alpha = np.float32(p['flux_alpha'])
    ds.n_flux_substeps = np.int32(p['n_flux_substeps'])
    ds.scalar_noise_sign = np.float32(p['scalar_noise_sign'])

    ds.close()
    print(f"Written {output_path}")
    return output_path
