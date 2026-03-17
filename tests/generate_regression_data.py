#!/usr/bin/env python
"""Generate reference data for regression tests.

Run this once to create tests/regression_data/*.npz files.
Re-run only when you intentionally change simulation behavior.

Usage:
    python tests/generate_regression_data.py
"""

import sys
from pathlib import Path
import tempfile

import numpy as np
import netCDF4

# Ensure steam is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steam.simulate import simulate

# Import case definitions from the test module
from test_regression import CASES, _make_profiles, _build_kwargs

REFERENCE_DIR = Path(__file__).parent / "regression_data"


def main():
    REFERENCE_DIR.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for case_name, params in sorted(CASES.items()):
            print(f"Generating: {case_name} ...", end=" ", flush=True)

            h_prof, qt_prof, kw = _build_kwargs(case_name, params, tmp_path)
            simulate(h_prof, qt_prof, **kw)

            ds = netCDF4.Dataset(kw["output_path"], "r")

            save_dict = {
                "h": ds.variables["h"][:],
                "qt": ds.variables["qt"][:],
                "x": ds.variables["x"][:],
                "y": ds.variables["y"][:],
                "z": ds.variables["z"][:],
                "C_h_k": ds.variables["C_h_k"][:],
                "C_qt_k": ds.variables["C_qt_k"][:],
                "k_values": ds.variables["k_values"][:],
                "k_z_values": ds.variables["k_z_values"][:],
                "nx": int(ds.nx),
                "ny": int(ds.ny),
                "dx": float(ds.dx),
                "dy": float(ds.dy),
                "C_h_L": float(ds.C_h_L),
                "C_qt_L": float(ds.C_qt_L),
            }
            ds.close()

            out_path = REFERENCE_DIR / f"{case_name}.npz"
            np.savez_compressed(out_path, **save_dict)
            print(f"saved {out_path.name}")

    print(f"\nAll {len(CASES)} reference files written to {REFERENCE_DIR}/")


if __name__ == "__main__":
    main()
