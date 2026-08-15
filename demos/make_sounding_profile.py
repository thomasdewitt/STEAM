#!/usr/bin/env python3
"""Build a STEAM input profile from a radiosonde sounding.

Fetches one sounding from the University of Wyoming archive and writes an npz
with the same keys as cm1.npz, so any case in generate_demo_fields.py can name
it. The CM1 and TWP-ICE profiles are tropical maritime means with a cloud base
a few hundred metres up; a Salt Lake ascent is a deep dry boundary layer over
high terrain, putting the cloud base kilometres higher.

Conventions match turbulon-analysis/make_input_profiles.py: mixing ratios, not
specific humidities (the archive's MIXR column is already one, and a sounding
carries no condensate so qt = r_v); h = cp*T + g*z + Lv*r_v with steam's
constants; uniform 50 m to 20 km.

z IS HEIGHT ABOVE THE STATION, not sea level. The model recovers T by
inverting T = (h - g*z - Lv*r_v)/cp from a z starting at 0 at the domain
bottom, so sea-level heights would warm every recovered temperature by
g*z_station/cp, 12.6 K at Salt Lake's 1289 m. The surface pressure written
alongside is the station's.

Output levels are cell MEANS of the raw ascent, not point samples: these are
~5 m resolution, so a 50 m cell holds about ten levels and point sampling
would alias. An empty cell is an error.

SMOOTHING. A single ascent is not a mean state, and it is rougher in the
gradient than the domain-and-time means STEAM is normally driven with. The
profiles are smoothed by differentiating, convolving the derivative with a
Gaussian, and integrating back from the measured surface value, which pins the
surface exactly. Two constrained choices:

  - mode="mirror" conserves the integral of the derivative, so the top of the
    profile lands where the sounding put it. Under "nearest" the large
    negative near-surface dqt/dz is replicated into the padding and integrates
    to -16 g/kg by 20 km.
  - SMOOTH_STD is 500 m, not 2 km: h and qt are smoothed independently, and
    nothing in that preserves their saturation relationship. At 2 km this
    sounding's moist layer smears upward into colder air until the mean state
    itself is saturated over 7.3-10.0 km, carrying 0.039 g/kg before the
    cascade contributes anything.

MOISTENING is optional and off by default. See moisten_peak.

Usage:
  python demos/make_sounding_profile.py --datetime "2026-04-20 00:00:00" \
      --name kslc_0420 --moisten-target 0.99
  python demos/make_sounding_profile.py --station 72572 --no-plot
"""

import argparse
import html
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from steam.constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
)

HERE = Path(__file__).resolve().parent
PROFILE_DIR = HERE / "profiles"

PROFILE_DZ = 50.0
DOMAIN_HEIGHT = 20000.0

# Gaussian std [m] applied to dh/dz and dqt/dz. 0 disables smoothing. See the
# module docstring for why this is 500 and not 2000.
SMOOTH_STD = 500.0

# --moisten-peak: a Gaussian added to qt, centred on the profile's relative
# humidity peak, scaled so the peak sits at MOISTEN_TARGET of saturation.
MOISTEN_STD = 500.0
MOISTEN_TARGET = 0.99

# The archive's current endpoint. The old cgi-bin/sounding path 404s.
SOURCE = "https://weather.uwyo.edu/wsgi/sounding"
STATION = "72572"        # KSLC, Salt Lake City
STATION_ELEV = None      # taken from the sounding's own first level

# Fixed-width columns of the TEXT:LIST block, in the order they appear.
COLUMNS = ("PRES", "HGHT", "TEMP", "DWPT", "RELH", "MIXR")


def fetch(station, datetime_utc):
    url = (f"{SOURCE}?datetime={urllib.parse.quote(datetime_utc)}"
           f"&id={station}&type=TEXT:LIST")
    with urllib.request.urlopen(url, timeout=90) as r:
        raw = r.read().decode("utf-8", "replace")
    if "Unable to retrieve" in raw:
        raise SystemExit(
            f"the archive has no sounding for station {station} at "
            f"{datetime_utc}. Soundings are launched at 00Z and 12Z and "
            f"appear an hour or two later.")
    block = re.search(r"<PRE>(.*?)</PRE>", raw, re.S)
    if block is None:
        raise SystemExit(f"no data block in the response from {url}")
    return html.unescape(block.group(1)), url


def parse(block):
    """The six columns needed, as float64 arrays, one row per level.

    Rows too short to carry MIXR are dropped: the archive blanks the derived
    humidity columns above where the sensor stops reporting.
    """
    rows = []
    for line in block.splitlines():
        if len(line) < 7 * len(COLUMNS):
            continue
        try:
            rows.append([float(line[7 * i:7 * (i + 1)])
                         for i in range(len(COLUMNS))])
        except ValueError:
            continue           # the header and rule lines land here
    if not rows:
        raise SystemExit("the data block parsed to no usable levels")
    return dict(zip(COLUMNS, np.asarray(rows, dtype=np.float64).T))


def to_steam_profile(sounding):
    """h, qt and the surface pressure, on the raw sounding levels."""
    z = sounding["HGHT"] - sounding["HGHT"][0]        # station-relative
    T = sounding["TEMP"] + 273.15                     # C -> K
    r_v = sounding["MIXR"] / 1000.0                   # g/kg -> kg/kg
    if not np.all(np.diff(z) > 0):
        raise SystemExit("sounding heights are not strictly increasing; "
                         "this parser assumes one clean ascent")
    h = cp * T + g * z + Lv * r_v
    return z, h, r_v, float(sounding["PRES"][0] * 100.0)   # hPa -> Pa


def bin_mean(z_raw, values, z_out, dz):
    """Cell means of `values` on the uniform grid, centred on output levels.

    An empty cell is fatal: averaging there would silently become
    nearest-neighbour.
    """
    edges = np.concatenate([z_out - dz / 2.0, [z_out[-1] + dz / 2.0]])
    idx = np.digitize(z_raw, edges) - 1
    keep = (idx >= 0) & (idx < z_out.size)
    counts = np.bincount(idx[keep], minlength=z_out.size)
    if np.any(counts == 0):
        empty = z_out[counts == 0]
        raise SystemExit(
            f"{empty.size} of {z_out.size} output levels have no sounding "
            f"level in them, the first at z = {empty[0]:.0f} m. The sounding "
            f"is coarser than the {dz:.0f} m grid there.")
    sums = np.bincount(idx[keep], weights=values[keep], minlength=z_out.size)
    return sums / counts


def smooth_via_gradient(f, z, std_m):
    """Differentiate, smooth the derivative, integrate back from f[0].

    The surface value is exact; everything above it moves. See the module
    docstring for why mode="mirror".
    """
    from scipy.ndimage import gaussian_filter1d

    if std_m <= 0:
        return f
    dz = z[1] - z[0]
    dfdz = gaussian_filter1d(np.gradient(f, dz), std_m / dz, mode="mirror")
    return f[0] + np.concatenate([[0.0], np.cumsum(0.5 * (dfdz[1:] + dfdz[:-1]) * dz)])


def relative_humidity(z, h, qt, surface_pressure):
    """qt over the saturation mixing ratio the model recovers.

    Total-water RH. Below saturation qv equals qt.
    """
    from steam.thermodynamics import recover_diagnostics, _saturation_mixing_ratio

    r = recover_diagnostics(h[None, None, :].astype(np.float32),
                            qt[None, None, :].astype(np.float32),
                            z.astype(np.float32), surface_pressure)
    return qt / _saturation_mixing_ratio(r["T"][0, 0], r["p"][0, 0])


def moisten_peak(z, h, qt, surface_pressure, std_m, target):
    """Add a Gaussian to qt at the RH peak, scaled to reach `target`.

    The amplitude is solved for, not computed: adding water at fixed h cools
    the level (h = cp T + g z + Lv qv), which lowers the saturation mixing
    ratio and raises RH again, so (r_sat - qt) overshoots.

    `target` is the maximum of the resulting RH profile, not the value at the
    Gaussian's centre. Saturation keeps falling with height, so the moistened
    maximum lands above the centre: anchoring on the centre put this profile's
    maximum at 99.9889%, a hundredth of a point from a saturated mean layer.
    RH is monotone in the amplitude, so this bisects.

    A target above 1.0 puts real condensate in the mean state.
    """
    rh0 = relative_humidity(z, h, qt, surface_pressure)
    centre_i = int(np.argmax(rh0))
    centre = z[centre_i]
    shape = np.exp(-0.5 * ((z - centre) / std_m) ** 2)

    def rh_max(amp):
        return relative_humidity(z, h, qt + amp * shape, surface_pressure).max()

    # The no-feedback amount to reach saturation at the centre overshoots for
    # any target at or below 100%, and falls short above it, so it grows.
    hi = float(qt[centre_i] / max(rh0[centre_i], 1e-6) - qt[centre_i])
    for _ in range(40):
        if rh_max(hi) >= target:
            break
        hi *= 2.0
    else:
        raise SystemExit(
            f"could not reach {target * 100:.1f}% even at {hi * 1000:.3f} g/kg; "
            f"the target is far enough above saturation that the cooling from "
            f"the added water is outrunning it")
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rh_max(mid) < target:
            lo = mid
        else:
            hi = mid
    amp = 0.5 * (lo + hi)

    qt_out = qt + amp * shape
    rh1 = relative_humidity(z, h, qt_out, surface_pressure)
    print(f"  moistened: {amp * 1000:.3f} g/kg Gaussian, std {std_m:.0f} m, "
          f"centred at {centre / 1000:.2f} km")
    print(f"    RH at centre {rh0[centre_i] * 100:.1f}% -> "
          f"{rh1[centre_i] * 100:.2f}%; profile maximum "
          f"{rh1.max() * 100:.2f}% at {z[int(np.argmax(rh1))] / 1000:.2f} km "
          f"({(1 - rh1.max()) * 100:.2f} points below saturation)")
    print(f"    qt at centre {qt[centre_i] * 1000:.3f} -> "
          f"{qt_out[centre_i] * 1000:.3f} g/kg")
    return qt_out


def check_mean_state(z, h, qt, surface_pressure):
    """Report any condensate the mean profile carries on its own.

    Clouds should come from the cascade's fluctuations about the mean state;
    a mean state saturated over a deep layer starts the run with a slab.
    """
    from steam.thermodynamics import recover_diagnostics

    r = recover_diagnostics(h[None, None, :].astype(np.float32),
                            qt[None, None, :].astype(np.float32),
                            z.astype(np.float32), surface_pressure)
    cond = r["qc"][0, 0] + r["qi"][0, 0]
    wet = cond > 1e-9
    if wet.any():
        print(f"  WARNING: the mean state is saturated at {wet.sum()} levels, "
              f"z = {z[wet][0] / 1000:.1f}-{z[wet][-1] / 1000:.1f} km, up to "
              f"{cond.max() * 1000:.3f} g/kg. The cascade will be adding "
              f"fluctuations to a layer that is already cloud.")
    else:
        print("  mean state is subsaturated at every level, as a driving "
              "profile should be")
    return r


def plot(out_png, z_out, h_out, qt_out, z_raw, h_raw, qt_raw, label,
         h_cell=None, qt_cell=None, surface_pressure=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm1 = PROFILE_DIR / "cm1.npz"
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 4.0), sharey=True)
    raw_c, cell_c, out_c, ref_c = "#c8c8c8", "#7fb3c8", "#1b4965", "#bc4b51"

    for ax, (raw, cell, out, name, unit) in zip(
            axes[:2],
            [(h_raw / 1000.0, None if h_cell is None else h_cell / 1000.0,
              h_out / 1000.0, "h", "kJ/kg"),
             (qt_raw * 1000.0, None if qt_cell is None else qt_cell * 1000.0,
              qt_out * 1000.0, "$q_t$", "g/kg")]):
        ax.plot(raw, z_raw / 1000.0, color=raw_c, lw=0.6, label="raw sounding")
        if cell is not None:
            ax.plot(cell, z_out / 1000.0, color=cell_c, lw=0.8,
                    label="50 m cell mean")
        ax.plot(out, z_out / 1000.0, color=out_c, lw=1.2, label="smoothed")
        ax.set_xlabel(f"{name} [{unit}]")

    if cm1.exists():
        ref = np.load(cm1)
        axes[0].plot(ref["h_profile"] / 1000.0, ref["z_profile"] / 1000.0,
                     color=ref_c, lw=1.0, ls="--", label="CM1")
        axes[1].plot(ref["qt_profile"] * 1000.0, ref["z_profile"] / 1000.0,
                     color=ref_c, lw=1.0, ls="--")

    # The gradients are the smoothness test: a profile can look clean and
    # still carry steps the cascade's weighting differentiates into spikes.
    # This is the panel the smoothing exists for, so it shows both.
    # A 0.1 g/kg bump is invisible on a qt axis running to 15 g/kg.
    if surface_pressure is not None:
        rh_ax = axes[2]
        if qt_cell is not None:
            rh_ax.plot(relative_humidity(z_out, h_out, qt_cell,
                                         surface_pressure) * 100,
                       z_out / 1000, color=cell_c, lw=0.8, label="before")
        rh_ax.plot(relative_humidity(z_out, h_out, qt_out,
                                     surface_pressure) * 100,
                   z_out / 1000, color=out_c, lw=1.2, label="after")
        rh_ax.axvline(100, color="k", lw=0.6, alpha=0.5)
        rh_ax.set_xlabel("RH [%]")
        rh_ax.set_xlim(0, 110)
        rh_ax.legend(fontsize=6, frameon=False)

    for ax, (cell, out, name, unit, scale) in zip(
            axes[3:],
            [(h_cell, h_out, r"$\partial h/\partial z$", "J/kg/m", 1.0),
             (qt_cell, qt_out, r"$\partial q_t/\partial z$", "g/kg/km", 1e6)]):
        if cell is not None:
            ax.plot(np.gradient(cell, z_out) * scale, z_out / 1000.0,
                    color=cell_c, lw=0.6)
        ax.plot(np.gradient(out, z_out) * scale, z_out / 1000.0,
                color=out_c, lw=1.0)
        ax.axvline(0.0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel(f"{name} [{unit}]")

    axes[0].set_ylabel("z above station [km]")
    axes[0].set_ylim(0, DOMAIN_HEIGHT / 1000.0)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(label, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default=STATION, help="WMO number (KSLC = 72572)")
    ap.add_argument("--datetime", default=None,
                    help='UTC, "YYYY-MM-DD HH:00:00"; default is the most '
                         'recent 00Z/12Z launch that has arrived')
    ap.add_argument("--name", default="kslc", help="output stem, <name>.npz")
    ap.add_argument("--smooth-std", type=float, default=SMOOTH_STD,
                    help="Gaussian std [m] on the gradients; 0 disables")
    ap.add_argument("--moisten-peak", action="store_true",
                    help="add a Gaussian to qt at the RH peak, scaled to reach "
                         "--moisten-target of saturation there")
    ap.add_argument("--moisten-std", type=float, default=MOISTEN_STD,
                    help="std [m] of that Gaussian")
    ap.add_argument("--moisten-target", type=float, default=None,
                    help="fraction of saturation to hit at the peak; giving "
                         f"this turns moistening on (default {MOISTEN_TARGET} "
                         "with --moisten-peak). Values above 1.0 are allowed "
                         "and put real condensate in the mean state")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    datetime_utc = args.datetime or latest_launch()
    block, url = fetch(args.station, datetime_utc)
    sounding = parse(block)
    z_raw, h_raw, qt_raw, surface_pressure = to_steam_profile(sounding)

    if z_raw[-1] < DOMAIN_HEIGHT:
        raise SystemExit(
            f"the sounding tops out at {z_raw[-1]:.0f} m above the station, "
            f"below the {DOMAIN_HEIGHT:.0f} m the profile grid needs. "
            f"Extrapolating a balloon that burst early is not something this "
            f"script will do quietly.")

    z_out = np.arange(0.0, DOMAIN_HEIGHT + PROFILE_DZ, PROFILE_DZ)
    h_cell = bin_mean(z_raw, h_raw, z_out, PROFILE_DZ)
    qt_cell = bin_mean(z_raw, qt_raw, z_out, PROFILE_DZ)

    h_out = smooth_via_gradient(h_cell, z_out, args.smooth_std)
    qt_out = smooth_via_gradient(qt_cell, z_out, args.smooth_std)
    # Above the humidity sensor's last report the raw cells are exactly zero
    # and the cumulative sum leaves float64 residue, ~1e-18 against a 1e-2
    # maximum. Clipped. A real undershoot reached -16 g/kg, so the tolerance
    # sits far above the rounding floor and far below the failure.
    tol = 1e-9 * qt_out.max()
    if qt_out.min() < -tol:
        raise SystemExit(
            f"smoothing drove qt to {qt_out.min() * 1000:.3f} g/kg, which is "
            f"not water. The integral of the smoothed derivative has to be "
            f"conserved for this not to happen; see smooth_via_gradient.")
    qt_out = np.maximum(qt_out, 0.0)

    qt_cell_unmoistened = qt_out.copy()
    moisten_on = args.moisten_peak or args.moisten_target is not None
    moisten_target = (MOISTEN_TARGET if args.moisten_target is None
                      else args.moisten_target)
    if moisten_on:
        qt_out = moisten_peak(z_out, h_out, qt_out, surface_pressure,
                              args.moisten_std, moisten_target)
    print(f"  smoothed with a {args.smooth_std:.0f} m Gaussian on the "
          f"gradients" if args.smooth_std > 0 else "  not smoothed")
    check_mean_state(z_out, h_out, qt_out, surface_pressure)

    PROFILE_DIR.mkdir(exist_ok=True)
    out_npz = PROFILE_DIR / f"{args.name}.npz"
    np.savez(
        out_npz,
        z_profile=z_out,
        h_profile=h_out,
        qt_profile=qt_out,
        surface_pressure=np.float64(surface_pressure),
        # Provenance, so a profile on disk can say what it is.
        station=np.array(args.station),
        datetime_utc=np.array(datetime_utc),
        source_url=np.array(url),
        station_elevation=np.float64(sounding["HGHT"][0]),
        smooth_std=np.float64(args.smooth_std),
        moisten_std=np.float64(args.moisten_std if moisten_on else 0.0),
        moisten_target=np.float64(moisten_target if moisten_on else 0.0),
    )
    print(f"wrote {out_npz.name}: {z_out.size} levels to "
          f"{DOMAIN_HEIGHT / 1000:.0f} km, surface {surface_pressure / 100:.1f} hPa, "
          f"station elevation {sounding['HGHT'][0]:.0f} m")
    print(f"  h  {h_out.min() / 1000:.1f} - {h_out.max() / 1000:.1f} kJ/kg")
    print(f"  qt {qt_out.min() * 1000:.3f} - {qt_out.max() * 1000:.2f} g/kg")

    if not args.no_plot:
        plot(PROFILE_DIR / f"{args.name}_profile.png", z_out, h_out, qt_out,
             z_raw, h_raw, qt_raw,
             f"station {args.station}, {datetime_utc} UTC, "
             f"{args.smooth_std:.0f} m gradient smoothing"
             + (f", moistened to {moisten_target * 100:.0f}% at the RH peak"
                if moisten_on else ""),
             h_cell=h_cell,
             # With moistening on, compare against the unmoistened smoothed
             # profile rather than the raw cell means.
             qt_cell=qt_cell_unmoistened if moisten_on else qt_cell,
             surface_pressure=surface_pressure)


def latest_launch():
    """The most recent 00Z or 12Z likely to have been archived.

    Two hours of slack: launch is ~45 min before the synoptic hour and the
    ascent takes about an hour to fly and post.
    """
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)
    hour = 12 if now.hour >= 12 else 0
    return now.replace(hour=hour, minute=0, second=0,
                       microsecond=0).strftime("%Y-%m-%d %H:00:00")


if __name__ == "__main__":
    main()
