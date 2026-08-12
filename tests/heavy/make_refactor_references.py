"""Pin reference outputs for the measure/apply refactor (component 2).

Run:  uv run python tests/heavy/make_refactor_references.py

Component 2 is a PURE REFACTOR -- every realized global reduction moves out of
the per-class body into an explicit measure/reduce/apply seam, and the in-RAM
answer must not move by one bit. The only trustworthy way to hold a refactor to
that after the fact is to pin its output BEFORE touching it, so these fixtures
were generated on 959f925 (component 1, world-keyed noise) and committed.

Four configurations, chosen for the code paths they exercise rather than for
physical interest:

  root_dyadic   a plain root, s = 1, dyadic classes
  root_s2       s = (2, 2, 2): the sparse noise lattice and a different
                per-level center count in the product norm
  root_half     n_scale_classes_per_dyad = 2: non-dyadic class spacing, so the
                per-class regrid and crop schedule differ
  nest          refine() of root_dyadic: inner_windows (every realized mean
                over a window rather than the whole array), world origins, and
                inherited deficits
  root_cuda     device='cuda': the GPU bounded-add and projection ledgers are
                separate code with their own reduction order, so they need
                their own pin (skipped where no GPU)

Each stores both the WRITTEN output (h, qt, flux -- so the composition's
projection and flux rescale are covered) and the cascade STATE
(h_perturbation, qt_perturbation, flux_state -- so a difference can be
attributed to the cascade or to the composition rather than just observed).

Regenerating these is a deliberate act: it asserts that a realization change is
intended. Component 1 changed realizations by ruling; component 2 must not.
"""

import sys
from pathlib import Path

import netCDF4
import numpy as np

from steam import simulate, refine

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"

# Deliberately tiny. These fixtures live in git forever, and their job is to
# cover code paths, not to be physically interesting: 16 x 16 x ~10 at the
# finest class keeps the whole set under a megabyte against a 4.8 MB repo,
# while still running four size classes through every reduction site.
PROFILE_NZ = 28
PROFILE_DZ = 30.0
DOMAIN_HEIGHT = 800.0
SEED = 42

FIELDS = ("h", "qt", "flux",
          "h_perturbation", "qt_perturbation", "flux_state")


def profiles():
    z = np.arange(PROFILE_NZ) * PROFILE_DZ
    return (340e3 - 20e3 * (z / z.max()),
            0.018 - 0.016 * (z / z.max()))


def base_kwargs(tmp, name):
    h, qt = profiles()
    return dict(
        h_profile=h, qt_profile=qt,
        nx=16, ny=16, dx=125.0, dy=125.0,
        outer_scale=2000.0, spheroscale=100.0,
        domain_height=DOMAIN_HEIGHT, profile_dz=PROFILE_DZ,
        output_path=tmp / f"{name}.nc",
        seed=SEED, save_for_refinement=True,
    )


def harvest(path, group='/'):
    """Every pinned field of one group, as a dict of float32 arrays."""
    with netCDF4.Dataset(path, "r") as ds:
        grp = ds if group == '/' else ds[group]
        out = {}
        for field in FIELDS:
            if field in grp.variables:
                out[field] = np.asarray(grp.variables[field][:],
                                        dtype=np.float32)
    return out


def build(tmp, with_cuda):
    cases = {}

    path = tmp / "root_dyadic.nc"
    simulate(**base_kwargs(tmp, "root_dyadic"))
    cases["root_dyadic"] = harvest(path)

    # A nest of that same root, so the reference covers inner_windows, the
    # world origins of component 1, and the inherited-deficit replay.
    with netCDF4.Dataset(path, "r") as ds:
        k_finest = float(ds.variables["k_values"][:][-1])
    refine(path, 4, 12, 4, 12, k_finest / 4, k_finest / 4,
           output_group="nest", save_for_refinement=True)
    cases["nest"] = harvest(path, "nest")

    kwargs = base_kwargs(tmp, "root_s2")
    kwargs["sparsity_factors"] = (2, 2, 2)
    simulate(**kwargs)
    cases["root_s2"] = harvest(tmp / "root_s2.nc")

    kwargs = base_kwargs(tmp, "root_half")
    kwargs["n_scale_classes_per_dyad"] = 2
    simulate(**kwargs)
    cases["root_half"] = harvest(tmp / "root_half.nc")

    if with_cuda:
        kwargs = base_kwargs(tmp, "root_cuda")
        kwargs["device"] = "cuda"
        simulate(**kwargs)
        cases["root_cuda"] = harvest(tmp / "root_cuda.nc")

    return cases


def main():
    import tempfile
    import torch

    with_cuda = torch.cuda.is_available()
    if not with_cuda:
        print("no CUDA: skipping the root_cuda reference")

    REFERENCE_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cases = build(Path(tmp), with_cuda)

    for name, fields in cases.items():
        target = REFERENCE_DIR / f"{name}.npz"
        if target.exists() and "--force" not in sys.argv:
            print(f"{target.name}: EXISTS, refusing to overwrite "
                  f"(pass --force if a realization change is intended)")
            continue
        np.savez_compressed(target, **fields)
        total = sum(v.nbytes for v in fields.values())
        print(f"{target.name}: {len(fields)} fields, "
              f"{total / 1024**2:.1f} MiB raw, "
              f"{target.stat().st_size / 1024**2:.2f} MiB on disk")


if __name__ == '__main__':
    main()
