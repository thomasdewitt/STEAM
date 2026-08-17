# demos/

Source fields for the STEAM group on cloudyview's soar rail.

Everything here ends at a NetCDF. The demo spec — crop, sun angles, card copy,
the volume packing, the still, `index.json` — lives in
`cloudyview/tools/prebake_demos.py`, which reads `demos/fields/` here directly
(`STEAM_SRC` there, a sibling-repo path) rather than out of a copy under
`cloudyview/data/demos/`. Regenerating a case is enough; there is nothing to
copy across afterwards.

The z crop is not specified there: STEAM specs carry `z="auto"` and prebake
trims to the occupied band, which is what the browser and `witness` both do on
load. That matters because a `--camera-position` is normalized to the domain
box, so the band is the frame of reference for every camera in a spec.

## Scripts

    python demos/generate_demo_fields.py [CASE ...]      # run the cases
    python demos/plot_cloud_fraction.py [CASE ...]       # look at the output
    python demos/make_sounding_profile.py --datetime ... # build a profile

`generate_demo_fields.py` writes `demos/fields/demo_<case>.nc`, one parent
domain per case with no nest, carrying qc and qi on the root group as
`(x, y, z)` float32 in kg/kg — what prebake reads with `dims="xyz"` and
`scale=1e3`. Each case runs into a working file holding the full cascade
state, tens of GB at these grids, and is stripped to the keeper before the
working file is deleted. A keeper that disagrees with the config in force is
refused rather than counted complete.

`plot_cloud_fraction.py` reports per-level liquid, ice and total cloud fraction
plus projected cover. Cloudy is `qc + qi >= 0.01 g/kg`, matching `CLOUD_KGKG`
in turbulon-analysis/hydrodynamic-comparison. The three curves do not sum:
a cell with 0.006 g/kg of each phase counts only in the total.

## Profiles

| name | source | qt sfc / 1 km | RH peak |
|------|--------|---------------|---------|
| `cm1` | CM1 RCE, all timesteps | 14.3 / 10.6 g/kg | 85% at 0.65 km |
| `twpice` | SAM TWP-ICE squall line | 19.4 / 15.4 g/kg | 92% at 5.5 km |
| `kslc` | KSLC sounding 2026-08-11 00Z | 6.4 / 5.6 g/kg | 75% at 6.2 km |
| `kslc_0420` | KSLC sounding 2026-04-20 00Z, moistened | 3.0 / 2.6 g/kg | 99% at 7.9 km |

Profiles live in `demos/profiles/`. `cm1` and `twpice` are copied from
`turbulon-analysis/runs/input_profiles/`; regenerating one means rerunning
`make_input_profiles.py` there. All four are versioned through
`!demos/profiles/*.npz`, since the blanket `*.npz` rule would drop them.

`twpice` carries anvil condensate at 12.25–13.9 km in its mean state, so a
domain deep enough to include it starts already clouded there.

`kslc_0420` is adjusted, not raw: `--moisten-peak` adds a Gaussian to qt at
the RH peak, scaled so the profile reaches `--moisten-target` of saturation
(0.095 g/kg at 7.8 km, 500 m std, taking the maximum from 70% to 99%).
Everything below 5 km is untouched. The amplitude is solved for, not computed:
adding water at fixed h cools the level, which lowers saturation and raises RH
again, so `r_sat - qt` overshoots. The target is the profile maximum, which
sits above the Gaussian's centre because saturation keeps falling with height.
A target of 1.00 reaches saturation with the mean state still clear;
above that it carries condensate, and the script says how much.

## Sounding conventions

Matched to `turbulon-analysis/make_input_profiles.py`: mixing ratios, not
specific humidities; `h = cp*T + g*z + Lv*r_v` with steam's constants; uniform
50 m to 20 km. Two things specific to a sounding:

- **z is height above the station, not sea level.** The model recovers T by
  inverting `T = (h - g*z - Lv*r_v)/cp` from a z starting at 0 at the domain
  bottom. Sea-level heights would warm every recovered temperature by
  `g*z_station/cp`, 12.6 K at Salt Lake's 1289 m. Verified the other way:
  driving `recover_diagnostics` with this profile reproduces the sounding's
  own T to 0.024 K rms and p to 0.1% over 20 km.
- **Levels are cell means.** Ascents are ~5 m resolution, so a 50 m cell holds
  about ten levels and point sampling would alias. An empty cell is an error.

Profiles are smoothed by differentiating, convolving the derivative with a
500 m Gaussian, and integrating back from the surface value. `mode="mirror"`
conserves the derivative's integral, so the top lands where the sounding put
it; under `nearest` the near-surface `dqt/dz` is replicated into the padding
and integrates to −16 g/kg. 500 m rather than 2 km because h and qt are
smoothed independently: at 2 km the moist layer smears into colder air until
the mean state itself is saturated over 7.3–10.0 km.

Which ascent matters more than the station. Of four KSLC soundings, the two
mid-August ones came after rain and are wet in the low levels; 08-11 moistens
monotonically to 74% at 6 km (a layer to build convection through) and 04-20
stays at 9–16% to 5 km before jumping (a high sheet over dry air).

## Cases

Every geometry knob is in the case's own `CASES` entry. Constraints, all
enforced by `simulate()`:

- Each horizontal extent is a whole multiple of `outer_scale`, or smaller than
  it. Smaller means a strip: kernels wider than the extent are periodized onto
  it, and statistics along that axis stop meaning anything near its width.
- `k_z,L = l_s (L/l_s)^H_z` stays below the domain top. Raising the top is not
  free — dz is set by dx and by which side of the spheroscale the finest class
  falls on.
- The profile spans the domain. Above the profile top h and qt are held
  constant and T falls dry-adiabatically: 20 km at −54.9 °C carried to a 30 km
  domain arrives at −152.6 °C, condensing everything left into a uniform slab
  at 0.0100 g/kg with no cascade structure. Host profiles stop at 20 km;
  `make_sounding_profile.py`'s `DOMAIN_HEIGHT` can go to ~32 km.

Domains are periodic in x and y, unlike an LES subvolume.

## Size

soar uploads one fp16 3D texture, `nx*ny*nz*2` bytes before gzip; the script
prints it per case. The largest field on the rail today is TWP-ICE at
1024 × 1024 × 206 — 432 MB raw, 184 MB gzipped. Chrome caps 3D textures at
2048 per axis. Over budget is not wrong, but then the crop at bake time is
choosing the geometry.
