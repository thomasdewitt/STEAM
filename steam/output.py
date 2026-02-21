"""NetCDF output writer for STEAM simulations."""

import numpy as np
import netCDF4
from pathlib import Path


def write_netcdf(
    output_path, h_3d, qt_3d,
    x_coords, y_coords, z_coords,
    h_profile, qt_profile, z_profile,
    k_values, k_z_values, C_h_k, C_qt_k,
    simulation_params,
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
        Scalar attributes to write as NetCDF global attributes.
        Required keys: nx, ny, dx, dy, dz, outer_scale, spheroscale,
        domain_height, profile_dz, sparsity_factors, surface_pressure,
        seed, C_h_L, C_qt_L, n_large_turbulons, H_h, H_z,
        h_min, h_max, qt_min, qt_max, min_distance_to_ground.

    Returns
    -------
    Path
        The output_path as a Path object.
    """
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

    # Data variables — chunked and compressed
    h_var = ds.createVariable(
        "h", "f4", ("x", "y", "z"), zlib=True, complevel=4,
        chunksizes=(min(64, nx_final), min(64, ny_final), nz_final),
    )
    h_var[:] = h_3d
    h_var.units = "J/kg"
    h_var.long_name = "moist static energy"

    qt_var = ds.createVariable(
        "qt", "f4", ("x", "y", "z"), zlib=True, complevel=4,
        chunksizes=(min(64, nx_final), min(64, ny_final), nz_final),
    )
    qt_var[:] = qt_3d
    qt_var.units = "kg/kg"
    qt_var.long_name = "total water mixing ratio"

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

    # 1D scale arrays
    kv = ds.createVariable("k_values", "f4", ("k",))
    kv[:] = k_values.astype(np.float32)
    kv.units = "m"
    kv.long_name = "horizontal turbulon scale classes"

    kzv = ds.createVariable("k_z_values", "f4", ("k",))
    kzv[:] = k_z_values.astype(np.float32)
    kzv.units = "m"
    kzv.long_name = "vertical turbulon scale classes"

    # C_h_k / C_qt_k: list of scalars (scalar spheroscale) or list of 1D arrays (profile spheroscale)
    if np.ndim(C_h_k[0]) == 0:
        chk = ds.createVariable("C_h_k", "f4", ("k",))
        chk[:] = np.array(C_h_k, dtype=np.float32)
        chk.long_name = "scale-dependent h amplitude"

        cqtk = ds.createVariable("C_qt_k", "f4", ("k",))
        cqtk[:] = np.array(C_qt_k, dtype=np.float32)
        cqtk.long_name = "scale-dependent qt amplitude"
    else:
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
    # dz: scalar attribute for uniform grids; variable for altitude-dependent spheroscale
    dz = p['dz']
    if np.ndim(dz) == 0:
        ds.dz = np.float32(dz)
    else:
        dz_var = ds.createVariable("dz", "f4", ("z",))
        dz_var[:] = np.asarray(dz, dtype=np.float32)
        dz_var.units = "m"
        dz_var.long_name = "cell height"
    ds.outer_scale = np.float32(p['outer_scale'])
    ds.spheroscale = np.float32(p['spheroscale'])
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

    ds.close()
    print(f"Written {output_path}")
    return output_path
