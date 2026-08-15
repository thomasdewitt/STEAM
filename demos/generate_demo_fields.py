#!/usr/bin/env python3
"""Generate the STEAM demo fields for cloudyview's soar viewer.

One parent domain per case, no nest. Output carries qc and qi on the root
group as (x, y, z) float32 in kg/kg, which prebake_demos.py reads with
dims="xyz" and scale=1e3. The demo spec (crop, sun angles, card copy) lives
there, not here.

Geometry constraints, all enforced by simulate():

  - Each horizontal extent (nx*dx, ny*dy) is a whole multiple of outer_scale,
    or smaller than it. Smaller is a strip: kernels wider than the extent are
    periodized onto it, and statistics along that axis stop meaning anything
    near the strip width.
  - k_z,L = l_s (L/l_s)^H_z stays below the domain top. dz is set by dx and by
    which side of the spheroscale the finest class falls on, so a taller
    domain buys levels the cascade has to carry.
  - The profile spans the domain. Above its top h and qt are held constant and
    T falls dry-adiabatically, condensing what is left into a uniform slab.

Domains are periodic in x and y.

Each case runs into a working file holding the full cascade state, tens of GB
at these grids, stripped to the keeper before the working file is deleted.
Runs are serial and size their working set against the whole machine.

soar uploads one fp16 3D texture, nx*ny*nz*2 bytes before gzip, printed per
case. TWP-ICE, the largest field on the rail, is 432 MB raw / 184 MB gzipped.
Chrome caps 3D textures at 2048 per axis.

Usage: python demos/generate_demo_fields.py [CASE ...]
"""

import sys
import time
from pathlib import Path

import numpy as np
import netCDF4

# FLUX_SCALE is a module global rather than a simulate() argument, so the
# amplitude is set by poking the module before each run. It has to be reached
# through importlib: steam/__init__.py binds the simulate FUNCTION to the name
# `steam.simulate`, so `import steam.simulate as m` hands back the function and
# an attribute set on it would go nowhere while the cascade ran at the default.
import importlib
_steam_simulate = importlib.import_module("steam.simulate")

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics, _saturation_mixing_ratio
from steam.constants import specific_heat_dry_air as cp
from steam.constants import latent_heat_vaporization as Lv
from steam.output import compression_kwargs

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "fields"

# Each profile is an npz carrying z_profile, h_profile, qt_profile and
# surface_pressure on a uniform 50 m grid to 20 km.
#
#   cm1        CM1 RCE, all archived timesteps. RH peaks at 85% at 0.65 km.
#   twpice     SAM TWP-ICE squall line, the wettest. Its mean state carries
#              anvil condensate at 12.25-13.9 km, so a domain deep enough to
#              include it starts already clouded there.
#   kslc       KSLC sounding 2026-08-11 00Z. Dry boundary layer moistening
#              monotonically to 75% at 6.2 km.
#   kslc_0420  KSLC sounding 2026-04-20 00Z, 9-16% RH to 5 km and then a sharp
#              moist layer. Moistened to 99% of saturation at its 7.8 km RH
#              peak; below 5 km is the sounding's own.
PROFILE_DIR = HERE / "profiles"
PROFILES = {name: PROFILE_DIR / f"{name}.npz"
            for name in ("cm1", "twpice", "kslc", "kslc_0420")}
PROFILE_DZ = 50.0

DEVICE = "cuda"

STREAM = True
MEMORY_BUDGET = None


CASES = {
    "marine-congestus": dict(
        nx=2048, ny=2048, dx=10.0,          # 20.48 km square
        domain_height=4000.0,
        outer_scale=20480.0,                # one L across each axis
        spheroscale=100.0,
        flux_noise_scale=0.02,
        seed=7001,
        profile="cm1",
    ),
    "stratified": dict(
        nx=2048, ny=2048 , dx=50.0,   # 20.48 km square
        domain_height=15000.0,
        outer_scale=2048.0 * 50,
        spheroscale=10.0,
        flux_noise_scale=0.02,
        seed=7002,
        profile="kslc_0420",
    ),
    "desert-convection": dict(
        nx=2048, ny=2048, dx=25.0,      
        domain_height=15000.0,
        outer_scale=51200.0,
        spheroscale=300.0,
        flux_noise_scale=0.02,              # the campaign's ledger is 0.02 / 0.05 / 0.17
        seed=7003,
        profile="kslc",
    ),
    "desert-convection-coarse": dict(
        nx=512, ny=512, dx=1000.0,      
        domain_height=15000.0,
        outer_scale=51200.0,
        spheroscale=300.0,
        flux_noise_scale=0.02,              # the campaign's ledger is 0.02 / 0.05 / 0.17
        seed=7003,
        profile="kslc",
    ),
}

# Fixed across the cases: moving one of these makes the three stop being a
# set that differs only in geometry and amplitude.
ANISOTROPY = "piecewise_isotropic_below_spheroscale"
TURBULON_SHAPE = "mexican_hat"

# What the keeper carries. Everything else simulate() writes is cascade state
# and diagnostics a renderer has no use for.
KEEP_VARS = ("qc", "qi")
KEEP_AUX = ("x", "y", "z", "z_profile", "dz", "spheroscale", "p_bottom")
KEEP = (*KEEP_AUX, *KEEP_VARS)


def working_path(case):
    return OUT_DIR / f"work_demo_{case}.nc"


def keeper_path(case):
    return OUT_DIR / f"demo_{case}.nc"


def config_spec(case):
    """The run attributes a keeper must match to be this case's field.

    Everything here is recorded by simulate() except spheroscale, which it
    writes as a profile variable; write_keeper puts the constant on the root
    so the comparison stays a scalar one.
    """
    c = CASES[case]
    return {"nx": c["nx"], "ny": c["ny"], "dx": c["dx"],
            "outer_scale": c["outer_scale"],
            "domain_height": c["domain_height"],
            "seed": c["seed"],
            "flux_noise_scale": c["flux_noise_scale"],
            "spheroscale_constant": c["spheroscale"]}


def spec_mismatches(ds, case):
    """Attributes of an existing keeper that disagree with the config."""
    bad = {}
    for attr, want in config_spec(case).items():
        got = getattr(ds, attr, None)
        if got is None or abs(float(got) - want) > 1e-9 * max(1.0, abs(want)):
            bad[attr] = ("missing" if got is None else f"{float(got):g}", want)
    for attr, want in (("demo_case", case),
                       ("profile_host", CASES[case]["profile"])):
        got = getattr(ds, attr, None)
        if got != want:
            bad[attr] = ("missing" if got is None else str(got), want)
    return bad


def texture_bytes(nx, ny, nz):
    """What one fp16 volume costs the viewer, before gzip."""
    return nx * ny * nz * 2


def run_parent(case, out_nc):
    if out_nc.exists():
        print(f"{out_nc.name} exists, skipping the simulation", flush=True)
        return

    c = CASES[case]
    profile_path = PROFILES[c["profile"]]
    if not profile_path.exists():
        raise SystemExit(
            f"case {case!r} is driven by profile {c['profile']!r}, and "
            f"{profile_path.name} is not there. A sounding profile is built "
            f"by demos/make_sounding_profile.py.")
    src = np.load(profile_path)
    profile_top = float(src["z_profile"][-1])
    if c["domain_height"] > profile_top:
        raise SystemExit(
            f"case {case!r} has a {c['domain_height']:.0f} m domain over a "
            f"profile that stops at {profile_top:.0f} m. Above the profile "
            f"top h and qt are held at their last value, and constant h means "
            f"T falls dry-adiabatically: {profile_top / 1000:.0f} km is "
            f"{(float(src['h_profile'][-1]) - 9.81 * profile_top) / 1004 - 273.15:.0f} C "
            f"but {c['domain_height'] / 1000:.0f} km comes out at "
            f"{(float(src['h_profile'][-1]) - 9.81 * c['domain_height']) / 1004 - 273.15:.0f} C, "
            f"which condenses everything left into a uniform ice slab. Lower "
            f"the domain, or build a taller profile -- a sounding reaches "
            f"~32 km, so make_sounding_profile.py can raise DOMAIN_HEIGHT; "
            f"the host profiles are fixed at 20 km by make_input_profiles.py.")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    spheroscale = np.full(src["z_profile"].size, c["spheroscale"])
    surface_pressure = float(src["surface_pressure"])
    qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
    # Anchored bounds, as the square and small-domain campaigns set them.
    h_upper = max(cp * 300.0 + Lv * qt_sat_surface, float(h_profile.max()))
    h_lower = float(h_profile.min()) - 10.0 * cp

    _steam_simulate.FLUX_SCALE = c["flux_noise_scale"]
    print(f"=== {case} === {c['nx']} x {c['ny']} at dx = {c['dx']:g} m "
          f"({c['nx'] * c['dx'] / 1000:.2f} x {c['ny'] * c['dx'] / 1000:.2f} km), "
          f"top {c['domain_height'] / 1000:.1f} km, "
          f"L = {c['outer_scale'] / 1000:.2f} km, "
          f"l_s = {c['spheroscale']:g} m, c = {c['flux_noise_scale']:g}",
          flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    simulate(
        h_profile, qt_profile,
        nx=c["nx"], ny=c["ny"], dx=c["dx"], dy=c["dx"],
        outer_scale=c["outer_scale"],
        spheroscale=spheroscale,
        anisotropy=ANISOTROPY,
        turbulon_shape=TURBULON_SHAPE,
        domain_height=c["domain_height"],
        profile_dz=PROFILE_DZ,
        output_path=str(out_nc),
        surface_pressure=surface_pressure,
        seed=c["seed"],
        h_min=h_lower, h_max=h_upper,
        qt_min=0.0, qt_max=qt_sat_surface,
        compress=True,
        device=DEVICE,
        # No nest, so no refinement state: this file cannot seed one later,
        # and turning that back on means rerunning the case.
        save_for_refinement=False,
        stream_to_disk=STREAM,
        memory_budget=MEMORY_BUDGET,
    )
    compute_diagnostics(str(out_nc), compress=True, device=DEVICE)
    print(f"  simulation done in {time.perf_counter() - t0:.0f} s "
          f"({out_nc.stat().st_size / 1e9:.1f} GB working file)", flush=True)


def write_keeper(case, out_nc, out_keep):
    if out_keep.exists():
        print(f"{out_keep.name} exists, skipping", flush=True)
        return
    tmp = out_keep.with_suffix(".nc.tmp")
    with netCDF4.Dataset(out_nc) as src, netCDF4.Dataset(tmp, "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        dst.source_working_file = out_nc.name
        # Recorded so a keeper made under other settings is caught rather than
        # counted complete; see spec_mismatches.
        dst.demo_case = case
        dst.profile_host = CASES[case]["profile"]
        dst.spheroscale_constant = CASES[case]["spheroscale"]
        dst.kept_variables = " ".join(KEEP_VARS)

        names = [n for n in KEEP if n in src.variables]
        dims_needed = {d for n in names for d in src.variables[n].dimensions}
        for name, dim in src.dimensions.items():
            if name in dims_needed:
                dst.createDimension(name, None if dim.isunlimited() else len(dim))
        for name in names:
            var = src.variables[name]
            chunks = var.chunking()
            chunks = None if chunks == "contiguous" else chunks
            out = dst.createVariable(
                name, var.dtype, var.dimensions, chunksizes=chunks,
                **compression_kwargs(True, chunks or var.shape))
            out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            if var.ndim == 3:                   # copy big fields in slabs
                step = max(1, var.shape[0] // 8)
                for i0 in range(0, var.shape[0], step):
                    out[i0:i0 + step] = var[i0:i0 + step]
            else:
                out[...] = var[...]
    tmp.rename(out_keep)

    with netCDF4.Dataset(out_keep) as ds:
        nx, ny, nz = ds.variables["qc"].shape
    print(f"wrote {out_keep.name} ({out_keep.stat().st_size / 1e9:.2f} GB), "
          f"{nx} x {ny} x {nz} -> "
          f"{texture_bytes(nx, ny, nz) / 1e6:.0f} MB as fp16 before gzip",
          flush=True)


def complete(out_keep, case):
    """True if this keeper is this case's field; fatal if it is another's."""
    with netCDF4.Dataset(out_keep) as ds:
        if not getattr(ds, "kept_variables", ""):
            raise RuntimeError(
                f"{out_keep.name} carries no kept_variables attribute, so it "
                f"is an unstripped run under the keeper's name. Move or "
                f"delete it rather than passing it off as a keeper.")
        bad = spec_mismatches(ds, case)
    if bad:
        detail = ", ".join(f"{a} = {got} (this run: {want})"
                           for a, (got, want) in bad.items())
        raise RuntimeError(
            f"{out_keep.name} was made under a different config: {detail}. "
            f"Move or delete it rather than leaving a stale field on disk.")
    return True


def run_case(case):
    out_nc = working_path(case)
    out_keep = keeper_path(case)
    if out_keep.exists() and not out_nc.exists():
        complete(out_keep, case)
        print(f"case {case} complete, skipping", flush=True)
        return
    run_parent(case, out_nc)
    write_keeper(case, out_nc, out_keep)
    out_nc.unlink()
    print(f"deleted {out_nc.name}", flush=True)


def main():
    cases = sys.argv[1:] or list(CASES)
    for case in cases:
        if case not in CASES:
            raise SystemExit(f"unknown case {case!r} (have {list(CASES)})")
    for case in cases:
        run_case(case)
    print("demo field generation complete", flush=True)


if __name__ == "__main__":
    main()
