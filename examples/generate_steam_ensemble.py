#!/usr/bin/env python3
"""Generate a STEAM ensemble and save outputs to NetCDF."""

from pathlib import Path
import sys

import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steam import simulate


# ---- Config ----
Domain_size = 5000
NX = 256
NY = 256
NZ = 256
DX = Domain_size / NX
DY = Domain_size / NY
DZ = Domain_size / NZ
RUNS = 2
BASE_SEED = 0
OUTER_SCALE = 3200.0
SPHEROSCALE = 10.0
SURFACE_PRESSURE = 101325.0
OUTPUT_NETCDF = REPO_ROOT / "examples/steam_ensemble.nc"
DTYPE = np.float32
CASCADE_PLOT_ROOT = None  # e.g. REPO_ROOT / "examples/cascade_plots"


if __name__ == "__main__":
    if not (
        NX > 0
        and NY > 0
        and NZ > 0
        and (NX & (NX - 1) == 0)
        and (NY & (NY - 1) == 0)
        and (NZ & (NZ - 1) == 0)
    ):
        raise ValueError("NX, NY, NZ must be powers of 2.")

    OUTPUT_NETCDF.parent.mkdir(parents=True, exist_ok=True)
    if CASCADE_PLOT_ROOT is not None:
        CASCADE_PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    z = np.arange(NZ, dtype=np.float64) * DZ
    h_profile = 340000.0 - 5.0 * z
    qt_profile = np.maximum(0.015 * np.exp(-z / 2500.0), 1e-6)

    output_vars = ("h", "qt", "T", "qv", "qc", "qi", "p")
    dtype_name = "f4" if DTYPE == np.float32 else "f8"

    print(f"Writing ensemble to: {OUTPUT_NETCDF}")
    with Dataset(OUTPUT_NETCDF, "w", format="NETCDF4") as ds:
        ds.createDimension("run", RUNS)
        ds.createDimension("x", NX)
        ds.createDimension("y", NY)
        ds.createDimension("z", NZ)

        ds.createVariable("run", "i4", ("run",))[:] = np.arange(RUNS, dtype=np.int32)
        ds.createVariable("seed", "i4", ("run",))[:] = np.arange(RUNS, dtype=np.int32) + BASE_SEED
        ds.createVariable("x", "f4", ("x",))[:] = np.arange(NX, dtype=np.float32) * DX
        ds.createVariable("y", "f4", ("y",))[:] = np.arange(NY, dtype=np.float32) * DY
        ds.createVariable("z", "f4", ("z",))[:] = z.astype(np.float32)
        ds.createVariable("h_profile_input", "f8", ("z",))[:] = h_profile
        ds.createVariable("qt_profile_input", "f8", ("z",))[:] = qt_profile

        ds.setncattr("nx", NX)
        ds.setncattr("ny", NY)
        ds.setncattr("nz", NZ)
        ds.setncattr("dx", DX)
        ds.setncattr("dy", DY)
        ds.setncattr("dz", DZ)
        ds.setncattr("outer_scale", OUTER_SCALE)
        ds.setncattr("spheroscale", SPHEROSCALE)
        ds.setncattr("surface_pressure", SURFACE_PRESSURE)
        ds.setncattr("base_seed", BASE_SEED)

        nc_vars = {
            name: ds.createVariable(
                name,
                dtype_name,
                ("run", "x", "y", "z"),
                zlib=True,
                complevel=1,
                chunksizes=(1, min(64, NX), min(64, NY), min(64, NZ)),
            )
            for name in output_vars
        }

        for run_idx in range(RUNS):
            run_seed = BASE_SEED + run_idx
            plot_dir = None if CASCADE_PLOT_ROOT is None else CASCADE_PLOT_ROOT / f"run_{run_idx:02d}"

            print(f"Simulation {run_idx + 1}/{RUNS} (seed={run_seed})")
            result = simulate(
                h_profile=h_profile,
                qt_profile=qt_profile,
                nx=NX,
                ny=NY,
                nz=NZ,
                dx=DX,
                dy=DY,
                dz=DZ,
                outer_scale=OUTER_SCALE,
                spheroscale=SPHEROSCALE,
                surface_pressure=SURFACE_PRESSURE,
                seed=run_seed,
                plot_dir=plot_dir,
            )

            for name in output_vars:
                nc_vars[name][run_idx, :, :, :] = result[name].astype(DTYPE, copy=False)
            ds.sync()

    print("Done.")
