#!/usr/bin/env python3
"""Extract the 2-D slices the pipeline figure needs from the 27 GB STEAM run.

Source: /Volumes/BLUE/STEAM visuals and demos/steam_full_c002_s0030.nc,
written by run_full_field.py beside it (same realization as the campaign's
small_c002_s0030.nc; c = 0.02, l_s = 30 m, L = 20.48 km, seed 7001).

The laptop cannot hold the file, so this reads only what the figure draws:

- an x-z slice (y = mid-domain) of the final fields h, qt, flux, T, qc, qi
- the same x-z slice of every class's increment in h and qt, each carried
  from its own coarse grid to the finest grid by 2-D linear zoom of the
  slice (after linear interpolation in y to the slice position). This is a
  straight zoom, not the cascade's hop-by-hop chain, so per-class
  amplitudes differ a few tens of percent from what the run itself
  delivered (see plot_class_increments.py on BLUE) -- fine for a concept
  figure, not for quantitative claims.
- input profiles, per-class k values, and the run's constants.

Writes steam_pipeline_slices.npz beside this script (~50 MB).
"""

from pathlib import Path

import numpy as np
import netCDF4
from scipy.ndimage import zoom

SRC = Path("/Volumes/BLUE/STEAM visuals and demos/steam_full_c002_s0030.nc")
HERE = Path(__file__).resolve().parent
OUT = HERE / "steam_pipeline_slices.npz"

FIELDS_3D = ["h", "qt", "flux", "T", "qc", "qi"]

# y = 726/1023: the cloudiest x-z slice in the run (5.8% cloudy vs 3.0% at
# mid-domain), found by scanning qc + qi > 1e-5 kg/kg per y plane.
Y_FRAC = 726 / 1023


def yslice(var, frac=Y_FRAC):
    """x-z plane at fractional y position, linearly interpolated in y."""
    ny = var.shape[1]
    fy = frac * (ny - 1)
    j0 = int(np.floor(fy))
    j1 = min(j0 + 1, ny - 1)
    w = fy - j0
    a = np.asarray(var[:, j0, :], dtype=np.float32)
    if j1 != j0 and w > 0:
        a = (1 - w) * a + w * np.asarray(var[:, j1, :], dtype=np.float32)
    return a


def main():
    out = {}
    with netCDF4.Dataset(SRC) as ds:
        nx, nz = ds["h"].shape[0], ds["h"].shape[2]
        for name in FIELDS_3D:
            print(f"reading {name} slice ...", flush=True)
            out[name] = yslice(ds[name])
        for name in ["x", "z", "z_profile", "h_profile", "qt_profile",
                     "k_values", "k_z_values", "dz"]:
            out[name] = np.asarray(ds[name][:])
        out["surface_pressure"] = np.float64(
            np.load(HERE / "steam_pipeline_cm1.npz")["surface_pressure"])

        inc = ds["class_increments"]
        names = sorted(inc.groups)
        out["class_names"] = np.array(names)
        out["class_k"] = np.array([float(inc[n].k) for n in names])
        for n in names:
            for f in ["h", "qt", "flux"]:
                print(f"reading class {n} {f} ...", flush=True)
                a = yslice(inc[n][f])
                if a.shape != (nx, nz):
                    a = zoom(a, (nx / a.shape[0], nz / a.shape[1]),
                             order=1, grid_mode=True, mode="grid-wrap")
                out[f"inc_{f}_{n}"] = a.astype(np.float32)

    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
