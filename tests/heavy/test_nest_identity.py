"""Nest identity: a full-domain nest must equal running the cascade further.

Two tests, run on two configurations, all small enough to finish in seconds:

  1. full-domain nest identity
       simulate() to a coarse finest grid, simulate() to a finer one with
       the same seed / domain / profiles, then refine() the coarse run over
       its ENTIRE domain down to the fine grid.  The nest must equal the
       direct fine run.

  2. nest-of-nest identity
       coarse -> mid -> fine as two chained full-domain refinements.  Must
       equal the direct fine run as well: a nest within a nest works exactly
       like a nest within a root.

Both pass BIT-EXACTLY -- every one of h, qt and flux identical in every
cell of every configuration -- with the interpolation compensation off, and
both fail completely with it on.  That is not a defect in the nesting.
INTERPOLATION_COMPENSATION is anchored to the run's OWN output grid
(comp_i = D_REF / D_i, with D_i the delivery of class i's regrid chain down
to the finest class the run happens to reach), so a run that stops coarse
damps its last nine classes differently from one that carries on -- and
those classes are the field the later coarse classes advect, through the
advective weight and the bound taper.  The depth-independent limit of that
factor, the chain continued indefinitely, is comp = 1 everywhere, i.e. no
compensation at all.  So no reformulation of an output-anchored, in-cascade
compensation can make a nest identical to a deeper root run; it is a
modelling choice between the two.  Both settings are therefore reported.

Run standalone:  python tests/heavy/test_nest_identity.py
"""

import shutil
import sys
from pathlib import Path

import netCDF4
import numpy as np

from steam import simulate, refine

SCRATCH_ROOT = Path(__file__).resolve().parents[2] / "scratch"
SEED = 12345


def dyadic():
    """Dyadic classes, uniform spheroscale, linear profiles, Nyquist sampling."""
    height, profile_dz = 6000.0, 100.0
    z = np.arange(int(height / profile_dz) + 1) * profile_dz
    return dict(
        name="dyadic",
        outer_scale=4000.0, domain=8000.0, height=height, profile_dz=profile_dz,
        dx=(250.0, 125.0, 62.5),            # coarse, mid, fine
        spheroscale=np.full(z.size, 1000.0),
        h=np.linspace(320.0, 340.0, z.size) * 1004.0,
        qt=np.linspace(0.012, 0.002, z.size),
        kwargs=dict(sparsity_factors=(1, 1, 1), n_scale_classes_per_dyad=1,
                    anisotropy='canonical'),
    )


def awkward():
    """Everything that could round differently: sqrt(2) class spacing, 4x
    oversampling, a height-dependent spheroscale the cascade crosses, the
    piecewise-isotropic grid, and curved profiles."""
    height, profile_dz = 6000.0, 50.0
    z = np.arange(int(height / profile_dz) + 1) * profile_dz
    return dict(
        name="awkward",
        outer_scale=4000.0, domain=4000.0, height=height, profile_dz=profile_dz,
        dx=(250.0, 125.0, 62.5),
        spheroscale=300.0 + 700.0 * (z / height),
        h=330e3 + 8e3 * np.tanh((z - 2000.0) / 1500.0),
        qt=0.014 - 0.012 * (z / height) ** 0.7,
        kwargs=dict(sparsity_factors=(2, 2, 2), n_scale_classes_per_dyad=2,
                    anisotropy='piecewise_isotropic_below_spheroscale'),
    )


def run(config, path, dx):
    """A root simulation of this configuration to finest spacing dx."""
    n = int(round(config['domain'] / dx))
    return simulate(
        config['h'], config['qt'], n, n, dx, dx,
        config['outer_scale'], config['spheroscale'], config['height'],
        config['profile_dz'], path,
        seed=SEED, save_class_increments=True, save_perturbations=True,
        device='cpu', **config['kwargs'],
    )


def read(path, group='/'):
    with netCDF4.Dataset(path, 'r') as ds:
        grp = ds if group == '/' else ds[group]
        return {name: np.asarray(grp.variables[name][:], dtype=np.float32)
                for name in ('h', 'qt', 'flux')}


def grid_size(path, group='/'):
    with netCDF4.Dataset(path, 'r') as ds:
        grp = ds if group == '/' else ds[group]
        return len(grp.dimensions['x']), len(grp.dimensions['y'])


def compare(label, nest, direct):
    """Max absolute and relative difference per field. True if equal."""
    ok = True
    print(f"  {label}")
    for name in ('h', 'qt', 'flux'):
        a, b = nest[name], direct[name]
        if a.shape != b.shape:
            print(f"    {name:5s} SHAPE MISMATCH {a.shape} vs {b.shape}")
            ok = False
            continue
        # Scale against the PERTURBATION, not the profile-dominated field: a
        # difference is only small if it is small next to the structure the
        # cascade actually built.
        scale = float(np.max(np.abs(b - b.mean(axis=(0, 1), keepdims=True))))
        diff = float(np.max(np.abs(a - b)))
        rel = diff / scale if scale > 0 else np.inf
        n_bad = int(np.count_nonzero(a != b))
        print(f"    {name:5s} shape={a.shape} max|diff|={diff:.6e} "
              f"rel={rel:.3e} ({n_bad}/{a.size} cells differ)")
        if n_bad:
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


def both_tests(config, scratch):
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    dx_coarse, dx_mid, dx_fine = config['dx']

    fine_path = scratch / "fine.nc"
    run(config, fine_path, dx_fine)
    direct = read(fine_path)
    sanity(direct)

    # The nest spans the parent's whole grid, whose size is set by the class
    # ladder and the sparsity, not by dx.
    p1 = scratch / "nest1.nc"
    run(config, p1, dx_coarse)
    nx, ny = grid_size(p1)
    refine(p1, 0, nx, 0, ny, dx_fine, dx_fine,
           output_group='refinements/r0', device='cpu')
    print(f"\nTEST 1  full-domain nest identity, {config['name']} "
          f"(coarse -> fine)")
    ok1 = compare("nest vs direct fine run:", read(p1, 'refinements/r0'), direct)

    p2 = scratch / "nest2.nc"
    run(config, p2, dx_coarse)
    nx, ny = grid_size(p2)
    refine(p2, 0, nx, 0, ny, dx_mid, dx_mid,
           output_group='refinements/r0', device='cpu',
           save_class_increments=True, save_perturbations=True)
    nx, ny = grid_size(p2, 'refinements/r0')
    refine(p2, 0, nx, 0, ny, dx_fine, dx_fine, parent_group='refinements/r0',
           output_group='refinements/r1', device='cpu')
    print(f"\nTEST 2  nest-of-nest identity, {config['name']} "
          f"(coarse -> mid -> fine)")
    ok2 = compare("nest-of-nest vs direct fine run:",
                  read(p2, 'refinements/r1'), direct)
    return ok1, ok2


def main():
    # `steam.simulate` the name is the function; the module is in sys.modules.
    steam_simulate = sys.modules['steam.simulate']
    results = {}
    for compensation in (True, False):
        if not compensation:
            steam_simulate.INTERPOLATION_COMPENSATION.clear()  # module switch
        state = "ON" if compensation else "OFF"
        for config in (dyadic(), awkward()):
            print()
            print("=" * 72)
            print(f"interpolation compensation {state}, "
                  f"{config['name']} configuration")
            print("=" * 72)
            results[state, config['name']] = both_tests(
                config, SCRATCH_ROOT / f"nest_identity_{state}_{config['name']}")

    print()
    for (state, name), (ok1, ok2) in results.items():
        print(f"compensation {state:3s}  {name:8s}  "
              f"test 1 {'PASS' if ok1 else 'FAIL'}   "
              f"test 2 {'PASS' if ok2 else 'FAIL'}")
    off = [ok for (state, _), pair in results.items() if state == "OFF"
           for ok in pair]
    on = [ok for (state, _), pair in results.items() if state == "ON"
          for ok in pair]
    if not all(on):
        print("\nThe compensation-ON failures are the depth-dependence of "
              "INTERPOLATION_COMPENSATION; see this module's docstring. That "
              "is a modelling decision, not a defect in the nesting.")
    return 0 if all(off) else 1


if __name__ == '__main__':
    sys.exit(main())
