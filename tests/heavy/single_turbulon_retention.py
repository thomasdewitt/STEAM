#!/usr/bin/env python
"""Single-turbulon retention probe (Thomas's proposal, 2026-07-28).

The per-hop retentions r(y) were measured on packed production fields
(turbulons at every k/2, heavily overlapping). If ONE turbulon carried
through the same chain of regrids reproduced them, the deep tail could
be probed at a tiny fraction of the cost.

RESULT (2026-07-28): it does not — the retention is an interference
property, not a kernel property. Measured canonical r(y), packed vs
one kernel (3D): 0.597 vs 0.838 at y = 2, 0.750 vs 0.986 at y = 4,
and ~1.002-1.005 (slightly ABOVE one) for every deeper hop, converging
to 1 from above. A lone kernel's |T| is nearly preserved by linear
resampling; the packed field's mean|sum| sits far below the sum of
mean|kernel| through cancellation between the ~12^3 kernels overlapping
each point, and every regrid perturbs that cancellation pattern. The
packed measurements (interpolation_compensation.py) therefore stand as
the only valid source for r(y); any deeper-tail probe must keep the
packed superposition (a packed single-class field without the cascade
machinery is the cheapest faithful option).

Checks this script was built for:
  1. Interference: compare single-kernel r(y) against the packed-field
     values over the measured range (the decisive test — failed).
  2. Dimensionality: a 2D (x,z) variant reaches y ~ 8192 cheaply; 2D
     and 3D single-kernel values agree to <1% at every shared hop, but
     this is moot given (1).

Grids replicate the cascade's: horizontal doubles per regrid; vertical
refines by 2**H_z per regrid in the canonical regime (non-nesting node
counts via rounding, as in production) or doubles in the isotropic
regime. The kernel is the production envelope, isotropic in index space
(the anisotropy lives in the grid's physical spacing, which never enters
here — only the node-count ratios matter).

Usage: python single_turbulon_retention.py [max_y_3d] [max_y_2d]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import importlib

sm = importlib.import_module("steam.simulate")
from steam.constants import hurst_vertical_anisotropy as H_z
from steam.utils import zoom_trilinear, zoom_bilinear

N0 = 14          # cells per side at the deposit stage (domain = 7k per axis:
                 # kernel support is +-3k, so one k of margin; 14 keeps the
                 # y = 512 stage under the memory cap)


def _stage_counts(m, regime):
    nx = N0 * 2 ** m
    if regime == 'isotropic':
        nz = N0 * 2 ** m
    else:
        nz = int(round(N0 * 2 ** (H_z * m)))
    return nx, nz


def run(regime, max_y, ndim):
    kern = sm._turbulon_envelope(1.0, 0.5, 0.5, 0.5).astype(np.float32)
    kw = kern.shape[0]
    if ndim == 3:
        field = np.zeros((N0, N0, N0), dtype=np.float32)
        lo = (N0 - kw) // 2
        field[lo:lo + kw, lo:lo + kw, lo:lo + kw] = kern
    else:
        field = np.zeros((N0, N0), dtype=np.float32)
        lo = (N0 - kw) // 2
        field[lo:lo + kw, lo:lo + kw] = kern[:, kw // 2, :]
    M = [float(np.abs(field).mean(dtype=np.float64))]
    m = 0
    while 2 * 2 ** m < max_y:
        m += 1
        nx, nz = _stage_counts(m, regime)
        if ndim == 3:
            field = zoom_trilinear(field, (nx, nx, nz))
        else:
            field = zoom_bilinear(field, (nx, nz))
        M.append(float(np.abs(field).mean(dtype=np.float64)))
    ys = [2 * 2 ** i for i in range(len(M) - 1)]
    rs = [M[i + 1] / M[i] for i in range(len(M) - 1)]
    return ys, rs


def main():
    max_y_3d = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    max_y_2d = int(sys.argv[2]) if len(sys.argv) > 2 else 8192

    for regime in ("canonical", "isotropic"):
        print(f"starting {regime}...", flush=True)
        packed = sm.HOP_RETENTION[regime]
        print(f"\n=== {regime} ===")
        # Isotropic 3D grids are cubic (vertical refines dyadically), so
        # y = 512 would need a 3584^3 array (184 GB); cap it.
        cap_3d = min(max_y_3d, 128) if regime == 'isotropic' else max_y_3d
        ys3, rs3 = run(regime, cap_3d, 3)
        ys2, rs2 = run(regime, max_y_2d, 2)
        r2 = dict(zip(ys2, rs2))
        print("   y    r_packed   r_single3D   r_single2D")
        for y, r3 in zip(ys3, rs3):
            p = packed.get(y)
            print(f"{y:6d}   {p if p is not None else float('nan'):8.4f}"
                  f"   {r3:10.4f}   {r2.get(y, float('nan')):10.4f}")
        for y in ys2:
            if y > max(ys3):
                print(f"{y:6d}        ---          ---   {r2[y]:10.4f}")


if __name__ == '__main__':
    main()
