#!/usr/bin/env python
"""Diagnostic: nested cascade = single long cascade (in statistics).

Runs two configurations at matching outer-to-finest dynamic ranges:

  (A) REFERENCE: one simulate() spanning outer_scale down to ~2*ref_dx.
  (B) PARENT+REFINE: simulate() at a coarser parent_dx (so fewer classes),
      followed by a full-domain refine() that extrapolates the parent's
      log-spacing until the reference's finest dx is reached.

If refine() correctly inherits the amplitude normalization (anchoring
C_k_child to C_k_parent[-1] * (k_child/k_parent_last)^H_h) and shifts
its k_values to start at k_split/gap, the combined cascade should
reproduce the reference's H_v = H_h/H_z across the entire vertical
range it resolves.

Reports:
  - Per-class k_values and C_h_k means from both cascades.
  - Vertical Haar fluctuations across seeds; fits H_v and the implied
    relative amplitude in the reference band and in the refined band.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import netCDF4

from steam.simulate import simulate, refine
from steam.constants import hurst_horizontal as H_h, hurst_vertical_anisotropy as H_z
import scaleinvariance
from scaleinvariance import haar_fluctuation

scaleinvariance.set_backend('torch')
#scaleinvariance.set_device('cuda')
scaleinvariance.set_numerical_precision('float32')


# ── Config ────────────────────────────────────────────────────────────────
N_REF         = 10                       # size classes in the reference
N_PARENT      = 5                        # size classes in the parent
NX = NY       = 1024
DOMAIN_WIDTH  = 4_000_000.0              # m
DOMAIN_HEIGHT = 20_000.0                 # m
OUTER_SCALE   = DOMAIN_WIDTH / 2         # m
SPHEROSCALE   = 10.0                     # constant, m
PROFILE_DZ    = 0.6                      # m
N_SEEDS       = 2
BASE_SEED     = 101


def build_profiles() -> tuple[np.ndarray, np.ndarray]:
    nz = int(DOMAIN_HEIGHT / PROFILE_DZ) + 1
    z = np.arange(nz) * PROFILE_DZ
    h  = 350e3 - 20e3 * (z / DOMAIN_HEIGHT) ** 2
    qt = 0.020 - 0.0199 * (z / DOMAIN_HEIGHT)
    return h.astype(np.float32), qt.astype(np.float32)


def run_reference(h, qt, seed, out_path):
    simulate(
        h, qt, nx=NX, ny=NY,
        dx=DOMAIN_WIDTH/NX, dy=DOMAIN_WIDTH/NY,
        outer_scale=OUTER_SCALE, spheroscale=SPHEROSCALE,
        domain_height=DOMAIN_HEIGHT, profile_dz=PROFILE_DZ,
        output_path=out_path, seed=seed,
        h_min=0.9*h.min(), h_max=1.1*h.max(),
        qt_min=0.0, qt_max=1.5*qt.max(),
        n_scale_classes_per_dyad=1,
    )


def run_parent(h, qt, seed, out_path, parent_nx, parent_dx):
    simulate(
        h, qt, nx=parent_nx, ny=parent_nx,
        dx=parent_dx, dy=parent_dx,
        outer_scale=OUTER_SCALE, spheroscale=SPHEROSCALE,
        domain_height=DOMAIN_HEIGHT, profile_dz=PROFILE_DZ,
        output_path=out_path, seed=seed,
        h_min=0.9*h.min(), h_max=1.1*h.max(),
        qt_min=0.0, qt_max=1.5*qt.max(),
        n_scale_classes_per_dyad=1,
    )


def main():
    tmp = Path(tempfile.mkdtemp(prefix='nested_equiv_'))
    print(f"Tempdir: {tmp}")

    # Parent's dx so that parent's finest k = OUTER / 2^(N_PARENT-1) matches
    # the reference's (N_PARENT-1)-th class, with the reference's gap=2.
    # For this, use gap=2 in both.  parent_dx such that outer/(2*parent_dx) =
    # 2^(N_PARENT-1):
    parent_dx = OUTER_SCALE / (2 * 2 ** (N_PARENT - 1))
    parent_nx = int(round(DOMAIN_WIDTH / parent_dx))
    ref_dx = OUTER_SCALE / (2 * 2 ** (N_REF - 1))
    print(f"REF: n_classes={N_REF}, dx={ref_dx:.2f}m, NX={NX}")
    print(f"PARENT: n_classes={N_PARENT}, dx={parent_dx:.2f}m, NX={parent_nx}")
    print(f"REFINE target dx = {ref_dx:.2f}m (= REF dx)\n")

    h, qt = build_profiles()

    ref_cols = []
    nested_cols = []
    z_ref = None
    z_nest = None

    ref_k = ref_Ch = None
    nest_k_parent = nest_k_refine = None
    nest_Ch_parent = nest_Ch_refine = None

    for i in range(N_SEEDS):
        seed = BASE_SEED + i
        print(f"-- seed {seed} --")

        ref_path = tmp / f"ref_seed{seed}.nc"
        run_reference(h, qt, seed, ref_path)
        with netCDF4.Dataset(ref_path) as ds:
            ref_cols.append(ds.variables['h'][:].reshape(-1, len(ds.dimensions['z'])))
            if z_ref is None:
                z_ref = ds.variables['z'][:]
                ref_k = ds.variables['k_values'][:]
                ref_Ch = ds.variables['C_h_k'][:]

        parent_path = tmp / f"parent_seed{seed}.nc"
        run_parent(h, qt, seed, parent_path, parent_nx, parent_dx)

        # Full-domain refine down to the reference's dx.
        refine(
            parent_path,
            x_start=0, x_stop=parent_nx,
            y_start=0, y_stop=parent_nx,
            dx=ref_dx, dy=ref_dx,
            output_group='refinements/nested',
            seed=seed + 10_000,
        )
        with netCDF4.Dataset(parent_path) as ds:
            grp = ds['refinements/nested']
            nested_cols.append(
                grp.variables['h'][:].reshape(-1, len(grp.dimensions['z']))
            )
            if z_nest is None:
                z_nest = grp.variables['z'][:]
                nest_k_parent = ds.variables['k_values'][:]
                nest_Ch_parent = ds.variables['C_h_k'][:]
                nest_k_refine = grp.variables['k_values'][:]
                nest_Ch_refine = grp.variables['C_h_k'][:]

    # ── C_h_k comparison ────────────────────────────────────────────────
    print("\n=== C_h_k at matching scales ===")
    combined_k = np.concatenate([nest_k_parent, nest_k_refine])
    combined_C = np.concatenate(
        [nest_Ch_parent.mean(axis=1), nest_Ch_refine.mean(axis=1)]
    )
    print(f"{'k':>12s}   {'ref mean(C_h)':>15s}   {'nested mean(C_h)':>18s}   ratio")
    for j, k in enumerate(ref_k):
        match = np.argmin(np.abs(combined_k - k))
        cr = float(ref_Ch[j].mean())
        cn = float(combined_C[match])
        print(f"{k:12.2f}   {cr:15.4f}   {cn:18.4f}   {cn/cr:.3f}")

    # ── Haar comparisons ────────────────────────────────────────────────
    ref_data  = np.concatenate(ref_cols,   axis=0)
    nest_data = np.concatenate(nested_cols, axis=0)
    dz_ref  = float(np.mean(np.diff(z_ref)))
    dz_nest = float(np.mean(np.diff(z_nest)))

    lags_r, haar_r = haar_fluctuation(ref_data,  axis=1, lags='powers of 1.05')
    lags_n, haar_n = haar_fluctuation(nest_data, axis=1, lags='powers of 1.05')
    scales_r = lags_r * dz_ref
    scales_n = lags_n * dz_nest

    def fit(scales, vals, lo, hi):
        m = (scales >= lo) & (scales <= hi) & (vals > 0)
        if m.sum() < 2:
            return np.nan
        s, _ = np.polyfit(np.log(scales[m]), np.log(vals[m]), 1)
        return s

    H_v_expected = H_h / H_z
    print(f"\n=== Vertical Haar slopes (expected H_v = {H_v_expected:.3f}) ===")
    # Refined band: below parent's k_z(k_parent_last)
    from steam.simulate import _k_z
    kz_parent_last = float(_k_z('canonical', nest_k_parent[-1], SPHEROSCALE))
    kz_refine_last = float(_k_z('canonical', nest_k_refine[-1], SPHEROSCALE))
    kz_parent_outer = float(_k_z('canonical', nest_k_parent[0], SPHEROSCALE))
    print(f"parent k_z range: [{kz_parent_last:.1f}, {kz_parent_outer:.1f}] m")
    print(f"refine k_z range: [{kz_refine_last:.1f}, {kz_parent_last:.1f}] m")
    band_lo, band_hi = 1.2 * kz_refine_last, 0.8 * kz_parent_last
    band_parent_lo, band_parent_hi = 1.2 * kz_parent_last, 0.8 * kz_parent_outer
    print(f"fit band (refined scales): [{band_lo:.1f}, {band_hi:.1f}] m")
    print(f"  REF slope     = {fit(scales_r, haar_r, band_lo, band_hi):.3f}")
    print(f"  NESTED slope  = {fit(scales_n, haar_n, band_lo, band_hi):.3f}")
    print(f"fit band (parent scales): [{band_parent_lo:.1f}, {band_parent_hi:.1f}] m")
    print(f"  REF slope     = {fit(scales_r, haar_r, band_parent_lo, band_parent_hi):.3f}")
    print(f"  NESTED slope  = {fit(scales_n, haar_n, band_parent_lo, band_parent_hi):.3f}")

    # ── Plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(scales_r, haar_r, 'o-', ms=3, lw=1, label=f'Reference (N={N_REF})',
              color='#2c7bb6')
    ax.loglog(scales_n, haar_n, 's-', ms=3, lw=1,
              label=f'Nested: parent (N={N_PARENT}) + refine', color='#d7191c')
    for x, lab, c in [
        (kz_parent_outer, "$k_z$(outer)", '#1a9641'),
        (kz_parent_last, "$k_z$(parent-k_min)", '#7b3294'),
        (kz_refine_last, "$k_z$(refine-k_min)", '#ff7f00'),
    ]:
        ax.axvline(x, color=c, ls='--', lw=1, alpha=0.7, label=lab)
    # reference H_v slope guide
    xr = np.array([kz_refine_last, kz_parent_outer])
    ax.loglog(xr, 1e4 * (xr/xr[0])**H_v_expected, 'k:', lw=1,
              label=f'slope = {H_v_expected:.3f}')
    ax.set_xlabel('Vertical lag [m]')
    ax.set_ylabel('Haar fluctuation (first order) [J/kg]')
    ax.set_title('Nested cascade vs reference single-cascade')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    out = Path(__file__).parent / 'nested_equivalence.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
