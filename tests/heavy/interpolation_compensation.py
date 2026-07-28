#!/usr/bin/env python
"""Measure INTERPOLATION_COMPENSATION on the production cascade path.

The supplement's Table 1 factor f(k/dx) compensates the resolution
dependence of the mean absolute turbulon amplitude: a class deposited on
its own working grid (Nyquist at s = 1) reads INFLATED on coarse output
grids, because at s = 1 the output samples coincide with the turbulon
centers and so systematically hit the envelope peaks. The inflation is a
property of the OUTPUT sampling alone: after the first trilinear regrid
the deposit is the chord polygon through its samples (a fixed point of
later regrids — see paper/concept-figs/s2), and successively finer
sampling of that fixed polygon converges to its continuum mean-abs. f is
defined relative to that chain limit ("well-resolved", f -> 1); the
one-time chord-polygon loss common to all classes is absorbed into the
lambda = 1/R calibration.

Protocol (production path, not an idealized kernel study): run
simulate() at a fixed physical configuration while halving dx, with the
amplitude ladder zeroed except for ONE target class and the deposit
compensation switched off. The target class's working grid, noise
protocol, normalization, anisotropic vertical chain, crop chain, and
bounded add are exactly production's. f(k/dx) is the ratio of the
mid-band volume-mean |h'| at the deepest run to that at each shallower
run.

Because the vertical working grids refine as k_z ~ k^{H_z} while the
horizontal refine as k, the vertical stays under-resolved longer than an
isotropic toy suggests; this bundled anisotropic chain is why the
production numbers differ from an isotropic-periodic kernel measurement
(which gives 0.46 at k/dx = 2 instead).

Writes interpolation_compensation.npz next to this script and prints the
table. Runtime ~tens of minutes (deepest run is 1024^2).
"""

import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

import importlib

sm = importlib.import_module("steam.simulate")  # NOT `import steam.simulate as
# sm`: steam/__init__ rebinds the attribute `simulate` to the function.

# ---- fixed physical configuration --------------------------------------
# K_TARGET is overridable from the command line (see __main__): probing
# the outer class (96 km) in the same dx sweep reaches k/dx = 512 at the
# same cost as k/dx = 32 for the 12 km class, and the overlap between
# the two probes tests that f depends on k/dx alone.
DOMAIN = 384_000.0          # m, horizontal extent (periodic)
OUTER = 96_000.0            # m
K_TARGET = 12_000.0         # m, the probed class (index 3: 96, 48, 24, 12)
HEIGHT = 8_000.0            # m
PROFILE_DZ = 50.0
SPHEROSCALE = 10.0
DX_LIST = (6000.0, 3000.0, 1500.0, 750.0, 375.0)   # k/dx = 2,4,8,16,32
SEEDS = (0, 1, 2)
BAND = (2_000.0, 6_000.0)   # m, mid-band for the volume mean (avoids
                            # surface/top center exclusion and cutoffs)


def _patched_cascade_loop(orig):
    """Zero every class's amplitude except K_TARGET's."""
    def wrapper(h_profile, qt_profile, z_profile, grids, C_h_k, C_qt_k,
                *args, **kwargs):
        k_values = np.asarray(grids['k'], dtype=float)
        C_h_k = [c if abs(k - K_TARGET) < 1.0 else np.zeros_like(c)
                 for k, c in zip(k_values, C_h_k)]
        C_qt_k = [c if abs(k - K_TARGET) < 1.0 else np.zeros_like(c)
                  for k, c in zip(k_values, C_qt_k)]
        return orig(h_profile, qt_profile, z_profile, grids, C_h_k,
                    C_qt_k, *args, **kwargs)
    return wrapper


def run_one(dx, seed, tmpdir):
    nx = int(round(DOMAIN / dx))
    nz_p = int(HEIGHT / PROFILE_DZ) + 1
    z_p = np.arange(nz_p) * PROFILE_DZ
    h_p = 350e3 - 20e3 * (z_p / HEIGHT)
    qt_p = 0.02 - 0.019 * (z_p / HEIGHT)

    out = Path(tmpdir) / f"probe_dx{int(dx)}_s{seed}.nc"
    sm.simulate(
        h_p, qt_p, nx, nx, dx, dx, OUTER, SPHEROSCALE, HEIGHT,
        PROFILE_DZ, out, seed=seed,
        # bounds far away: the probe measures the raw deposit
        h_min=0.0, h_max=1e9, qt_min=-1.0, qt_max=1.0,
        anisotropy='piecewise_isotropic_below_spheroscale',
    )
    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        z = ds.variables['z'][:]
        band = (z >= BAND[0]) & (z <= BAND[1])
        h = ds.variables['h'][:, :, band]
        hp_out = np.interp(z[band], z_p, h_p).astype(np.float64)
    pert = h.astype(np.float64) - hp_out[None, None, :]
    # per-level mean-abs, then average over the band: keeps levels
    # equally weighted across runs with different vertical grids
    m_lev = np.abs(pert).mean(axis=(0, 1))
    out.unlink()
    return float(m_lev.mean()), int(band.sum())


def main():
    # deposit compensation OFF: measure the raw resolution dependence
    if hasattr(sm, 'ZOOM_RETENTION'):
        sm.ZOOM_RETENTION = (1.0,)
    if hasattr(sm, 'INTERPOLATION_COMPENSATION'):
        sm.INTERPOLATION_COMPENSATION = {}
    sm.cascade_loop = _patched_cascade_loop(sm.cascade_loop)

    M = np.zeros((len(SEEDS), len(DX_LIST)))
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, seed in enumerate(SEEDS):
            for j, dx in enumerate(DX_LIST):
                M[i, j], nlev = run_one(dx, seed, tmpdir)
                print(f"seed {seed} dx {dx:6.0f}  k/dx {K_TARGET/dx:4.0f}  "
                      f"M = {M[i, j]:9.3f}  ({nlev} band levels)",
                      flush=True)

    kdx = np.array([K_TARGET / dx for dx in DX_LIST])
    f = M[:, -1][:, None] / M          # per-seed, relative to deepest run
    here = Path(__file__).parent
    np.savez(here / f"interpolation_compensation_k{int(K_TARGET)}.npz",
             kdx=kdx, M=M, f=f, seeds=np.array(SEEDS),
             k_target=K_TARGET, dx_list=np.array(DX_LIST))
    print(f"\n k/dx    f (mean +/- std over seeds)   [target class "
          f"{K_TARGET:.0f} m; reference: deepest run k/dx = {kdx[-1]:.0f}]")
    for j, x in enumerate(kdx):
        print(f"  {x:4.0f}   {f[:, j].mean():6.3f} +/- {f[:, j].std():.3f}")


if __name__ == '__main__':
    if len(sys.argv) > 1:            # e.g. 96000 for the deep-tail probe
        K_TARGET = float(sys.argv[1])
    if len(sys.argv) > 2:            # optional extra dx entries, comma-sep
        DX_LIST = tuple(float(v) for v in sys.argv[2].split(','))
    main()
