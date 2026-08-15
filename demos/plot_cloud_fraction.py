#!/usr/bin/env python3
"""Cloud fraction profiles for the generated demo fields.

One panel per case: the fraction of cells at each level carrying liquid, ice,
and condensate of either kind, plus the projected cloud cover.

    python demos/plot_cloud_fraction.py             # every field on disk
    python demos/plot_cloud_fraction.py desert-convection

A cell is cloudy at 0.01 g/kg, matching CLOUD_KGKG in
turbulon-analysis/hydrodynamic-comparison. Applied per cell to the field as
written; nothing is coarsened first.

The three curves do not sum. Liquid is qc over the threshold, ice is qi over
it, total is qc + qi over it, so a cell holding 0.006 g/kg of each counts only
in the total.

Cover is the fraction of columns cloudy anywhere, the quantity the fractal
metrics in turbulon-analysis are computed on. It is at least the largest
per-level fraction, and much larger when the field is vertically broken.

Counts accumulate over x-slabs; these fields run to gigabytes per variable and
are never resident.
"""

import sys
from pathlib import Path

import numpy as np
import netCDF4

HERE = Path(__file__).resolve().parent
FIELDS = HERE / "fields"

# Matches turbulon-analysis/hydrodynamic-comparison/scripts/common.py.
CLOUD_KGKG = 0.01e-3

SLAB = 64                       # x-slab width for the streaming pass


def cloud_fraction(path):
    """Per-level liquid/ice/total fractions and the projected cover.

    Counts are int64: a 2048^2 x 400 field has 1.7e9 cells, past int32.
    """
    with netCDF4.Dataset(path) as ds:
        if "qc" not in ds.variables:
            raise SystemExit(
                f"{path.name} has no qc at the root group. This script reads "
                f"the keepers generate_demo_fields.py writes; a file with the "
                f"fields in subgroups is a different layout.")
        z = np.asarray(ds.variables["z"][:], dtype=np.float64)
        qc_v, qi_v = ds.variables["qc"], ds.variables["qi"]
        nx, ny, nz = qc_v.shape
        if z.size != nz:
            raise SystemExit(f"{path.name}: z has {z.size} levels, qc has {nz}")

        counts = {k: np.zeros(nz, dtype=np.int64)
                  for k in ("liquid", "ice", "total")}
        any_cloud = np.zeros((nx, ny), dtype=bool)

        for x0 in range(0, nx, SLAB):
            x1 = min(x0 + SLAB, nx)
            qc = np.asarray(qc_v[x0:x1], dtype=np.float32)
            qi = np.asarray(qi_v[x0:x1], dtype=np.float32)
            liq, ice = qc >= CLOUD_KGKG, qi >= CLOUD_KGKG
            tot = (qc + qi) >= CLOUD_KGKG
            counts["liquid"] += liq.sum(axis=(0, 1))
            counts["ice"] += ice.sum(axis=(0, 1))
            counts["total"] += tot.sum(axis=(0, 1))
            any_cloud[x0:x1] = tot.any(axis=2)

    cells_per_level = float(nx * ny)
    fracs = {k: v / cells_per_level for k, v in counts.items()}
    return z, fracs, float(any_cloud.mean())


def describe(name, z, fracs, cover):
    """Per-case summary printed beside the figure."""
    tot = fracs["total"]
    cloudy = tot > 0.01
    print(f"--- {name}")
    print(f"    cover {cover * 100:5.1f}%   peak level fraction "
          f"{tot.max() * 100:5.1f}% at {z[int(np.argmax(tot))] / 1000:.2f} km")
    if cloudy.any():
        print(f"    cloud from {z[cloudy][0] / 1000:5.2f} to "
              f"{z[cloudy][-1] / 1000:5.2f} km (levels over 1%)")
    else:
        print(f"    no level reaches 1% cloud fraction; "
              f"peak is {tot.max() * 100:.3f}%")
    ice, liq = fracs["ice"], fracs["liquid"]
    mixed = (ice > 0.01) & (liq > 0.01)
    if mixed.any():
        print(f"    mixed phase (both over 1%) {z[mixed][0] / 1000:5.2f} to "
              f"{z[mixed][-1] / 1000:5.2f} km")
    elif (ice > 0.01).any():
        print(f"    ice only, from {z[ice > 0.01][0] / 1000:5.2f} km")
    else:
        print("    liquid only")


def plot(out_png, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(total="#22333b", liquid="#1b4965", ice="#bc4b51")
    styles = dict(total="-", liquid="-", ice="--")
    widths = dict(total=1.6, liquid=1.0, ice=1.0)

    fig, axes = plt.subplots(1, len(results), figsize=(2.9 * len(results), 3.8),
                             squeeze=False)
    for ax, (name, z, fracs, cover) in zip(axes[0], results):
        for key in ("total", "liquid", "ice"):
            ax.plot(fracs[key] * 100, z / 1000, color=colors[key],
                    ls=styles[key], lw=widths[key], label=key)
        ax.set_title(f"{name}\ncover {cover * 100:.1f}%", fontsize=7.5)
        ax.set_xlabel("cloud fraction [%]")
        ax.set_ylim(0, z.max() / 1000)
        ax.margins(x=0.02)
    axes[0][0].set_ylabel("z [km]")
    axes[0][0].legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"cloudy at qc + qi $\\geq$ {CLOUD_KGKG * 1e3:g} g/kg",
                 fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nwrote {out_png}")


def main():
    available = sorted(p for p in FIELDS.glob("demo_*.nc"))
    if not available:
        raise SystemExit(
            f"no demo fields in {FIELDS}. Generate them first with "
            f"demos/generate_demo_fields.py.")
    wanted = sys.argv[1:]
    if wanted:
        paths = []
        for case in wanted:
            p = FIELDS / f"demo_{case}.nc"
            if not p.exists():
                raise SystemExit(
                    f"no field for case {case!r}; on disk: "
                    f"{[q.stem.removeprefix('demo_') for q in available]}")
            paths.append(p)
    else:
        paths = available

    results = []
    for p in paths:
        name = p.stem.removeprefix("demo_")
        try:
            z, fracs, cover = cloud_fraction(p)
        except FileNotFoundError:
            # A concurrent generate_demo_fields.py run deletes a stale keeper
            # before rewriting it, so a listed name can be gone by open time.
            print(f"--- {name}: vanished between listing and opening, "
                  f"skipped")
            continue
        describe(name, z, fracs, cover)
        results.append((name, z, fracs, cover))
    if not results:
        raise SystemExit("no field could be read")
    plot(HERE / "cloud_fraction.png", results)


if __name__ == "__main__":
    main()
