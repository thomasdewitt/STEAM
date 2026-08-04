"""Nest identity: a full-domain nest must equal running the cascade further.

Two tests, both on a small domain so they run in seconds:

  1. full-domain nest identity
       simulate() to a coarse finest grid (dx_a), simulate() to a fine one
       (dx_b < dx_a) with the same seed/domain/profiles, then refine() the
       coarse run over its ENTIRE domain down to dx_b.  The nest must equal
       the direct fine run.

  2. nest-of-nest identity
       coarse -> mid -> fine as two chained full-domain refinements.  Must
       equal the direct fine run as well (nest-within-nest works exactly
       like nest-within-parent).

Run standalone:  python tests/heavy/test_nest_identity.py
"""

import shutil
import sys
from pathlib import Path

import netCDF4
import numpy as np

from steam import simulate, refine

SCRATCH = Path(__file__).resolve().parents[2] / "scratch" / "nest_identity"

# ── small configuration ──────────────────────────────────────────────────
# L = 4000 m, dyadic classes.  Coarse run stops at k = 500 m (dx = 250 m),
# fine run at k = 125 m (dx = 62.5 m): two extra classes.
OUTER_SCALE = 4000.0
DOMAIN_X = 8000.0          # two outer-scale tiles per axis
DOMAIN_Y = 8000.0
DX_COARSE = 250.0
DX_MID = 125.0
DX_FINE = 62.5
DOMAIN_HEIGHT = 6000.0
SPHEROSCALE = 1000.0
PROFILE_DZ = 100.0
SEED = 12345

N_PROFILE = int(DOMAIN_HEIGHT / PROFILE_DZ) + 1
Z_PROFILE = np.arange(N_PROFILE) * PROFILE_DZ
H_PROFILE = np.linspace(320.0, 340.0, N_PROFILE) * 1004.0
QT_PROFILE = np.linspace(0.012, 0.002, N_PROFILE)


def run(path, dx):
    """A root simulation to finest horizontal spacing dx."""
    nx = int(round(DOMAIN_X / dx))
    ny = int(round(DOMAIN_Y / dx))
    return simulate(
        H_PROFILE, QT_PROFILE, nx, ny, dx, dx,
        OUTER_SCALE, SPHEROSCALE, DOMAIN_HEIGHT, PROFILE_DZ, path,
        seed=SEED, save_class_increments=True, save_perturbations=True,
        device='cpu',
    )


def read(path, group='/'):
    with netCDF4.Dataset(path, 'r') as ds:
        grp = ds if group == '/' else ds[group]
        return {name: np.asarray(grp.variables[name][:], dtype=np.float32)
                for name in ('h', 'qt', 'flux')}


def compare(label, nest, direct):
    """Max absolute and relative difference per field.  Returns True if equal."""
    ok = True
    print(f"  {label}")
    for name in ('h', 'qt', 'flux'):
        a, b = nest[name], direct[name]
        if a.shape != b.shape:
            print(f"    {name:5s} SHAPE MISMATCH {a.shape} vs {b.shape}")
            ok = False
            continue
        # Scale against the PERTURBATION, not the profile-dominated field:
        # a difference is only small if it is small next to the structure
        # the cascade actually built.
        scale = float(np.max(np.abs(b - b.mean(axis=(0, 1), keepdims=True))))
        diff = float(np.max(np.abs(a - b)))
        rel = diff / scale if scale > 0 else np.inf
        n_bad = int(np.count_nonzero(a != b))
        print(f"    {name:5s} shape={a.shape} max|diff|={diff:.6e} "
              f"rel={rel:.3e} ({n_bad}/{a.size} cells differ)")
        if rel > 1e-5:
            ok = False
    return ok


def sanity(direct):
    """The reference fine run must itself be a plausible field."""
    h, qt, flux = direct['h'], direct['qt'], direct['flux']
    print("  sanity of the direct fine run:")
    print(f"    h  : mean={h.mean():.1f} std={h.std():.2f} "
          f"range=[{h.min():.1f}, {h.max():.1f}]")
    print(f"    qt : mean={qt.mean():.6f} std={qt.std():.6f} "
          f"range=[{qt.min():.6f}, {qt.max():.6f}]")
    print(f"    flx: mean={flux.mean():.4f} std={flux.std():.4f} "
          f"min={flux.min():.4f}")
    assert h.std() > 1.0, "h field is structureless"
    assert qt.std() > 1e-5, "qt field is structureless"
    assert np.isfinite(h).all() and np.isfinite(qt).all()


def main():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    fine_path = SCRATCH / "fine.nc"
    run(fine_path, DX_FINE)
    direct = read(fine_path)
    sanity(direct)

    nx_coarse = int(round(DOMAIN_X / DX_COARSE))
    ny_coarse = int(round(DOMAIN_Y / DX_COARSE))

    # ── Test 1: one full-domain refinement, coarse -> fine ───────────────
    p1 = SCRATCH / "nest1.nc"
    run(p1, DX_COARSE)
    refine(p1, 0, nx_coarse, 0, ny_coarse, DX_FINE, DX_FINE,
           output_group='refinements/r0', device='cpu')
    print("\nTEST 1  full-domain nest identity (coarse -> fine)")
    ok1 = compare("nest vs direct fine run:", read(p1, 'refinements/r0'), direct)

    # ── Test 2: two chained full-domain refinements ──────────────────────
    p2 = SCRATCH / "nest2.nc"
    run(p2, DX_COARSE)
    refine(p2, 0, nx_coarse, 0, ny_coarse, DX_MID, DX_MID,
           output_group='refinements/r0', device='cpu',
           save_class_increments=True, save_perturbations=True)
    nx_mid = int(round(DOMAIN_X / DX_MID))
    ny_mid = int(round(DOMAIN_Y / DX_MID))
    refine(p2, 0, nx_mid, 0, ny_mid, DX_FINE, DX_FINE,
           parent_group='refinements/r0',
           output_group='refinements/r1', device='cpu')
    print("\nTEST 2  nest-of-nest identity (coarse -> mid -> fine)")
    ok2 = compare("nest-of-nest vs direct fine run:",
                  read(p2, 'refinements/r1'), direct)

    print()
    print(f"TEST 1 (full-domain nest)  : {'PASS' if ok1 else 'FAIL'}")
    print(f"TEST 2 (nest within nest)  : {'PASS' if ok2 else 'FAIL'}")
    return 0 if (ok1 and ok2) else 1


if __name__ == '__main__':
    sys.exit(main())
