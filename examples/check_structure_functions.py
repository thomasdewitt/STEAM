#!/usr/bin/env python3
"""Compare structure functions of STEAM output vs input profiles.

Goal: the output SF should meet the input profile SF at the outer scale.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import netCDF4
import scaleinvariance
from steam import simulate
H_z = 5/9

# ---- Config ----
NX = NY = 128*10
DOMAIN_SIZE = 10000
DX = DY = DOMAIN_SIZE // NX
SPARSITY = 1
OUTER_SCALE = 1000
SPHEROSCALE = 100.0
DOMAIN_HEIGHT = 3600.0
PROFILE_DZ = 30.0
SEED = 42
OUTPUT_PATH = "/Users/thomas/code-and-data/turbulon-model/examples/steam_output.nc"
PLOT_DIR = "/Users/thomas/code-and-data/turbulon-model/examples/plots/steam_sf"

# Toggle: which scaleinvariance function to use
# hurst_fn = scaleinvariance.structure_function_hurst
hurst_fn = scaleinvariance.haar_fluctuation_hurst
# hurst_fn = scaleinvariance.spectral_hurst

USE_SPECTRAL = (hurst_fn is scaleinvariance.spectral_hurst)

# ---- Build input profiles ----
nz_profile = int(DOMAIN_HEIGHT / PROFILE_DZ) + 1
z_profile = np.arange(nz_profile) * PROFILE_DZ
h_profile = 340e3 - 30e3 * (z_profile / z_profile.max())
qt_profile = 0.018 - 0.016 * (z_profile / z_profile.max())

# ---- Run STEAM ----
nc_path = simulate(
    h_profile, qt_profile,
    nx=NX, ny=NY, dx=DX, dy=DY,
    outer_scale=OUTER_SCALE,
    spheroscale=SPHEROSCALE,
    domain_height=DOMAIN_HEIGHT,
    profile_dz=PROFILE_DZ,
    output_path=OUTPUT_PATH,
    seed=SEED,
    sparsity_factors=(SPARSITY, SPARSITY, 2 * SPARSITY),
)

# Read back from NetCDF
ds = netCDF4.Dataset(nc_path, "r")
h_field = ds.variables["h"][:]
qt_field = ds.variables["qt"][:]
dz_output = float(ds.dz)
ds.close()

print(f"Output grid: {h_field.shape}, dz={dz_output:.1f} m")
print(f"h range: {h_field.min():.0f} — {h_field.max():.0f}")
print(f"qt range: {qt_field.min():.6f} — {qt_field.max():.6f}")

# ---- Hurst kwargs differ for spectral vs real-space methods ----
if USE_SPECTRAL:
    fit_kwargs = dict(return_fit=True, max_wavelength=None, min_wavelength=None)
    vert_ref_slope = -11 / 5
    horiz_ref_slope = -5 / 3
    vert_ref_label = "slope -11/5"
    horiz_ref_label = "slope -5/3"
    ylabel = "Power spectral density"
    sep_label_vert = "Vertical wavelength [m]"
    sep_label_horiz = "Horizontal wavelength [m]"
else:
    fit_kwargs = dict(return_fit=True, max_sep=None, min_sep=None)
    vert_ref_slope = 3 / 5
    horiz_ref_slope = 1 / 3
    vert_ref_label = "H = 3/5"
    horiz_ref_label = "H = 1/3"
    ylabel = "Haar fluctuation"
    sep_label_vert = "Vertical separation [m]"
    sep_label_horiz = "Horizontal separation [m]"

# ---- Vertical structure functions ----
h_vert = hurst_fn(h_field, axis=2, **fit_kwargs)
qt_vert = hurst_fn(qt_field, axis=2, **fit_kwargs)
h_prof = hurst_fn(h_profile, **fit_kwargs)
qt_prof = hurst_fn(qt_profile, **fit_kwargs)

vertical_outer_scale = SPHEROSCALE * (OUTER_SCALE / SPHEROSCALE) ** H_z

# Physical separations (lags * grid spacing)
h_vert_sep = np.asarray(h_vert[2]) * dz_output
h_vert_sf = np.asarray(h_vert[3])
h_prof_sep = np.asarray(h_prof[2]) * PROFILE_DZ
h_prof_sf = np.asarray(h_prof[3])

qt_vert_sep = np.asarray(qt_vert[2]) * dz_output
qt_vert_sf = np.asarray(qt_vert[3])
qt_prof_sep = np.asarray(qt_prof[2]) * PROFILE_DZ
qt_prof_sf = np.asarray(qt_prof[3])
# Reference slopes
h_vert_ref = h_vert_sf[0] * (h_vert_sep / h_vert_sep[0]) ** vert_ref_slope
qt_vert_ref = qt_vert_sf[0] * (qt_vert_sep / qt_vert_sep[0]) ** vert_ref_slope

# ---- Horizontal structure functions ----
h_horiz = hurst_fn(h_field, axis=0, **fit_kwargs)
qt_horiz = hurst_fn(qt_field, axis=0, **fit_kwargs)

h_horiz_sep = np.asarray(h_horiz[2]) * DX
h_horiz_sf = np.asarray(h_horiz[3])
qt_horiz_sep = np.asarray(qt_horiz[2]) * DX
qt_horiz_sf = np.asarray(qt_horiz[3])

h_horiz_ref = h_horiz_sf[0] * (h_horiz_sep / h_horiz_sep[0]) ** horiz_ref_slope
qt_horiz_ref = qt_horiz_sf[0] * (qt_horiz_sep / qt_horiz_sep[0]) ** horiz_ref_slope

# ---- Plot ----
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

C_OUTPUT = "#2196F3"
C_PROFILE = "#FF9800"
C_REF = "#888888"

# -- Vertical --
fig, (ax_h, ax_qt) = plt.subplots(1, 2, figsize=(10, 5))

ax_h.loglog(h_vert_sep, h_vert_sf, "o-", lw=1.4, ms=3, color=C_OUTPUT,
            label=f"STEAM output  H={float(h_vert[0]):.3f}")
ax_h.loglog(h_prof_sep, h_prof_sf, "s--", lw=1.2, ms=3, color=C_PROFILE,
            label=f"input profile  H={float(h_prof[0]):.3f}")
ax_h.loglog(h_vert_sep, h_vert_ref, "-", lw=0.8, color=C_REF, label=vert_ref_label)
ax_h.axvline(vertical_outer_scale, color=C_REF, ls=":", alpha=0.6,
             label=f"vert. outer scale ({vertical_outer_scale:.0f} m)")
ax_h.set(xlabel=sep_label_vert, ylabel=ylabel, title="h")
ax_h.grid(True, which="both", alpha=0.2)
ax_h.legend(fontsize=7)

ax_qt.loglog(qt_vert_sep, qt_vert_sf, "o-", lw=1.4, ms=3, color=C_OUTPUT,
             label=f"STEAM output  H={float(qt_vert[0]):.3f}")
ax_qt.loglog(qt_prof_sep, qt_prof_sf, "s--", lw=1.2, ms=3, color=C_PROFILE,
             label=f"input profile  H={float(qt_prof[0]):.3f}")
ax_qt.loglog(qt_vert_sep, qt_vert_ref, "-", lw=0.8, color=C_REF, label=vert_ref_label)
ax_qt.axvline(vertical_outer_scale, color=C_REF, ls=":", alpha=0.6,
              label=f"vert. outer scale ({vertical_outer_scale:.0f} m)")
ax_qt.set(xlabel=sep_label_vert, ylabel=ylabel, title="qt")
ax_qt.grid(True, which="both", alpha=0.2)
ax_qt.legend(fontsize=7)

fig.suptitle("Vertical structure functions", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/sf_vertical.png", dpi=160)
print(f"Saved {PLOT_DIR}/sf_vertical.png")
plt.close(fig)

# -- Horizontal --
fig, (ax_h, ax_qt) = plt.subplots(1, 2, figsize=(10, 5))

ax_h.loglog(h_horiz_sep, h_horiz_sf, "o-", lw=1.4, ms=3, color=C_OUTPUT,
            label=f"STEAM output  H={float(h_horiz[0]):.3f}")
ax_h.loglog(h_horiz_sep, h_horiz_ref, "-", lw=0.8, color=C_REF, label=horiz_ref_label)
ax_h.axvline(OUTER_SCALE, color=C_REF, ls=":", alpha=0.6,
             label=f"outer scale ({OUTER_SCALE:.0f} m)")
ax_h.set(xlabel=sep_label_horiz, ylabel=ylabel, title="h")
ax_h.grid(True, which="both", alpha=0.2)
ax_h.legend(fontsize=7)

ax_qt.loglog(qt_horiz_sep, qt_horiz_sf, "o-", lw=1.4, ms=3, color=C_OUTPUT,
             label=f"STEAM output  H={float(qt_horiz[0]):.3f}")
ax_qt.loglog(qt_horiz_sep, qt_horiz_ref, "-", lw=0.8, color=C_REF, label=horiz_ref_label)
ax_qt.axvline(OUTER_SCALE, color=C_REF, ls=":", alpha=0.6,
              label=f"outer scale ({OUTER_SCALE:.0f} m)")
ax_qt.set(xlabel=sep_label_horiz, ylabel=ylabel, title="qt")
ax_qt.grid(True, which="both", alpha=0.2)
ax_qt.legend(fontsize=7)

fig.suptitle("Horizontal structure functions", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/sf_horizontal.png", dpi=160)
print(f"Saved {PLOT_DIR}/sf_horizontal.png")
plt.close(fig)
