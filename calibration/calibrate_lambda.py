#!/usr/bin/env python3
"""End-to-end calibration of the delivery amplitude lambda (HAAR_TO_MHAT).

lambda is fixed by the outer-scale crossover criterion: at the vertical
outer scale k_z,L the vertical Haar fluctuation of the simulated columns
must equal that of the mean profile. Both are measured here from full
STEAM runs, so nothing about the cascade's delivery chain is assumed.

lambda is calibrated PER H_h and does not transfer between them: the ratio
below varies systematically with the horizontal Hurst exponent. Every run
therefore takes an explicit --hurst-horizontal, and every output file is
tagged with it so calibrations at different H_h coexist on disk.

Procedure (one CLI argument each):

  round1  runs the two (h, qt) panels at lambda = 1, reads the mean-profile
          fluctuation and the fixed-slope column line at k_z,L, and reports
          the per-panel ratio
              ratio = F_mean(k_z,L) / F_column_line(k_z,L).
          Since the column fluctuation is proportional to lambda, the
          fitted lambda is the geometric mean of the two ratios.
          Writes round1_Hh<H>.npz (and prints the fitted value).

  round2  re-runs the same panels at the fitted lambda, checks that the
          ratios have moved to ~1, and produces the paper figure.

The column line has slope fixed at H_v = H_h / H_z and is anchored at the
SECOND-SMALLEST lag of the column fluctuation function (Thomas's ruling).
That anchor sits one octave inside the cascade range: the model sets each
class's dz = k_z(z) / (2 s_z), so the output grid Nyquist-samples the
finest class, the smallest Haar lag (2 grid points) is exactly k_z of the
finest turbulon, and the second-smallest lag (4 points) is 2 k_z of it.
There is no sub-turbulon regime in the output by construction.

Extrapolating the fixed-slope line from below is the clean definition
precisely because it ignores what happens near k_z,L: the curvature seen
there is mixing between the mean profile and the turbulence (Thomas's
reading), which is expected physics rather than a defect to fit through.
A mean-residual fit over an intermediate band of lags (comparison_window_
grid_units) is printed alongside as an empirical measure of how much that
mixing bends the curve.

  iterate runs round2 in a fixed-point loop: lambda_{i+1} = lambda_i * r_i,
          stopping when |r_i - 1| < --tol. Under the pre-2026-08-03
          procedure the bound projection clipped the hotter fields and
          made the response sublinear in lambda; with the far bounds now
          used (see run_case) the response should be linear and iterate
          is retained as a convergence check rather than a necessity.
          Writes round2_Hh<H>_iter{i}.npz and the matching figure.

Both rounds render the same 1x2 figure (h left, qt right) from their own
runs: figs/lambda_calibration_Hh<H>_lambda1.png (crossover open, at
lambda = 1) and figs/lambda_calibration_Hh<H>_fitted.png (crossover
closed). The `figure` subcommand re-renders from a saved .npz without
re-running any simulations.

Usage:
    python calibrate_lambda.py round1 --hurst-horizontal 0.45 [--smoke]
    python calibrate_lambda.py round2 --hurst-horizontal 0.45 [--smoke]
    python calibrate_lambda.py iterate --hurst-horizontal 0.45
                                       [--lambda-value X] [--tol 0.02]
                                       [--max-iters 3] [--start-iter 1]
    python calibrate_lambda.py figure --hurst-horizontal 0.45 [--npz FILE]

--smoke runs a tiny 128x64 configuration to exercise the plumbing; it
preserves the 2:1 aspect and outer_scale = x extent = 2 * y extent so the
y-axis fold is exercised. Its numbers are statistically meaningless.

Lives here, not in turbulon-analysis, because what it produces is a model
constant (steam.constants.haar_to_mhat), not an analysis result. Moved
2026-08-05 from turbulon-analysis/lambda_calibration/ (history there:
26590e9 the original pipeline, b86a539 the far-bounds procedure). It sits
outside the packaged steam* tree and needs matplotlib + scaleinvariance,
which the steam package itself does not -- see the `calibration` dependency
group in pyproject.toml.
"""

import argparse
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scaleinvariance

from steam import simulate
from steam.constants import hurst_vertical_anisotropy as H_z
from steam.simulate import _k_z

HERE = Path(__file__).parent
FIGS = HERE / "figs"

# ---- Configuration (production; --smoke shrinks it below) ----
dx = dy = 5000                   # m
nx = 2048                        # x extent 10240 km = exactly one outer scale
ny = 1024                        # y extent 5120 km = L/2: a strip axis, so
                                 # ONLY the outermost class folds along y
                                 # (fold_kernel_to_field periodizes it).
                                 # Accepted deliberately.
outer_scale = dx * nx            # m, horizontal outer scale L = the x extent.
                                 # L never exceeds the domain: variance has to
                                 # build up through the whole scale range.
domain_height = 24000            # m (k_z,L ~ 21.81 km fits inside)
profile_dz = 0.6                 # m
spheroscale = 10                 # m
n_seeds = 3
column_subsample = 4             # every 4th column: the Haar analysis
                                 # allocates several float64 copies
anisotropy = 'canonical'

# Haar lag ladder. The scaleinvariance default 'powers of 1.2' stops at 66
# grid units (15.3 km at dz = 232 m), short of k_z,L = 21.8 km; 1.05 reaches
# 98 units = 22.7 km. Both ladders start [2, 4, 6, ...], so the
# second-smallest-lag anchor is lag 4 = 928 m either way -- the definition is
# unaffected by the change. analyze_panel raises if the ladder falls short.
haar_lag_ladder = 'powers of 1.05'
# Comparison-only window for the mean-residual anchor, in GRID UNITS.
comparison_window_grid_units = (8, 22)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})
color_mean = "#1764ab"
color_column = "#C4442A"
color_line = "#E9A03B"
color_reference = "0.4"


def run_case(hurst_horizontal, haar_to_mhat):
    """Run n_seeds simulations at one H_h and one lambda; pool the columns.

    Returns (z_out, h_columns, qt_columns).
    """
    n_profile = int(domain_height / profile_dz) + 1
    z_profile = np.arange(n_profile) * profile_dz

    # Both mean profiles LINEAR, with very different slopes.
    h_profile = 350e3 - 20e3 * (z_profile / domain_height)
    qt_profile = 0.020 - 0.0199 * (z_profile / domain_height)

    h_column_sets = []
    qt_column_sets = []
    z_out = None
    for seed in range(n_seeds):
        print(f"  H_h={hurst_horizontal}, lambda={haar_to_mhat:.5f}, "
              f"seed {seed + 1}/{n_seeds}...", flush=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "case.nc"
            simulate(
                h_profile, qt_profile, nx, ny, dx, dy,
                outer_scale, spheroscale, domain_height, profile_dz,
                output_path, seed=seed,
                # Bounds pushed far beyond any reachable value (Thomas's
                # ruling, 2026-08-03): lambda is a geometric delivery
                # constant, but the bound projection is case-specific and
                # was clipping the calibration fields -- the same lambda is
                # then applied to subvolumes where bounds are never hit.
                # Far bounds keep the actual-STEAM path (taper, projection
                # machinery all still execute) while guaranteeing neither
                # scalar ever touches a bound in practice.
                h_min=h_profile.min() - 1e7, h_max=h_profile.max() + 1e7,
                qt_min=-1e3, qt_max=1e3,
                n_scale_classes_per_dyad=1,
                hurst_horizontal=hurst_horizontal,
                haar_to_mhat=haar_to_mhat,
            )
            dataset = netCDF4.Dataset(output_path)
            h_3d = dataset.variables['h'][:]
            qt_3d = dataset.variables['qt'][:]
            if z_out is None:
                z_out = dataset.variables['z'][:]
                print(f"    metadata H_h = {dataset.getncattr('H_h')}, "
                      f"lambda = {dataset.getncattr('lambda_haar_to_mhat')}",
                      flush=True)
            dataset.close()

        nz = h_3d.shape[2]
        h_column_sets.append(h_3d.reshape(-1, nz)[::column_subsample])
        qt_column_sets.append(qt_3d.reshape(-1, nz)[::column_subsample])

    return (np.asarray(z_out),
            np.concatenate(h_column_sets, axis=0),
            np.concatenate(qt_column_sets, axis=0))


def analyze_panel(z_out, columns, mean_profile_on_z, hurst_horizontal):
    """Haar fluctuations and the k_z,L amplitude ratios for one panel.

    The mean profile is passed already interpolated to the output z-grid.
    """
    hurst_vertical = hurst_horizontal / H_z
    k_z_L = float(_k_z(anisotropy, outer_scale, spheroscale))
    dz_out = float(z_out[1] - z_out[0])

    # Both calls MUST share the ladder: the log-log interpolation at k_z,L
    # compares the two curves against each other.
    lags, haar_mean = scaleinvariance.haar_fluctuation(
        mean_profile_on_z, order=1, axis=0, lags=haar_lag_ladder)
    _, haar_column = scaleinvariance.haar_fluctuation(
        columns, order=1, axis=1, lags=haar_lag_ladder)
    grid_lags = np.asarray(lags, float)
    physical_lags = grid_lags * dz_out

    # No silent clamping: np.interp holds its endpoint value rather than
    # extrapolating, so a ladder that stops short of k_z,L would quietly
    # return the wrong mean-profile amplitude. The default 'powers of 1.2'
    # ladder does exactly that at nz ~ 103 (tops out at 66 units = 15.3 km
    # against k_z,L = 21.8 km), which is why the ladder is 1.05 here.
    if physical_lags[-1] < k_z_L:
        raise ValueError(
            f"Haar ladder '{haar_lag_ladder}' tops out at "
            f"{physical_lags[-1]:.0f} m, short of k_z,L = {k_z_L:.0f} m: the "
            f"interpolation at k_z,L would clamp silently. Use a denser lag "
            f"ladder or a taller domain."
        )

    log_lags = np.log10(physical_lags)
    log_mean = np.log10(haar_mean)
    log_column = np.log10(haar_column)
    log_k_z_L = np.log10(k_z_L)

    # Column line: slope fixed at H_v, anchored at the second-smallest lag.
    intercept_anchored = log_column[1] - hurst_vertical * log_lags[1]
    # Comparison only. Defined by GRID UNITS, not ladder index: the 1.05
    # ladder is denser than the 1.2 one this window was first written
    # against, so a fixed index slice would silently mean different physical
    # lags. These bounds reproduce the span the old 'points 4-9' covered.
    window = ((grid_lags >= comparison_window_grid_units[0])
              & (grid_lags <= comparison_window_grid_units[1]))
    intercept_window = float(np.mean(
        log_column[window] - hurst_vertical * log_lags[window]))

    column_at_k_z_L = 10 ** (intercept_anchored + hurst_vertical * log_k_z_L)
    column_at_k_z_L_window = 10 ** (intercept_window
                                    + hurst_vertical * log_k_z_L)
    mean_at_k_z_L = 10 ** np.interp(log_k_z_L, log_lags, log_mean)

    # Slope-1 reference through the mean profile, over the template's window.
    mean_fit = ((physical_lags >= 4 * dz_out)
                & (physical_lags <= domain_height / 4))
    intercept_mean = float(np.mean(log_mean[mean_fit] - 1.0 * log_lags[mean_fit]))

    return {
        'physical_lags': physical_lags,
        'haar_mean': haar_mean,
        'haar_column': haar_column,
        'hurst_vertical': hurst_vertical,
        'k_z_L': k_z_L,
        'intercept_anchored': intercept_anchored,
        'intercept_window': intercept_window,
        'intercept_mean': intercept_mean,
        'ratio': mean_at_k_z_L / column_at_k_z_L,
        'ratio_window': mean_at_k_z_L / column_at_k_z_L_window,
    }


def run_all_panels(hurst_horizontal, haar_to_mhat):
    """The two panels (h, qt) at one H_h and one lambda."""
    z_out, h_columns, qt_columns = run_case(hurst_horizontal, haar_to_mhat)

    n_profile = int(domain_height / profile_dz) + 1
    z_profile = np.arange(n_profile) * profile_dz
    h_mean = np.interp(z_out, z_profile,
                       350e3 - 20e3 * (z_profile / domain_height))
    qt_mean = np.interp(z_out, z_profile,
                        0.020 - 0.0199 * (z_profile / domain_height))

    panels = {}
    for field, columns, mean_profile in [('h', h_columns, h_mean),
                                         ('qt', qt_columns, qt_mean)]:
        panel = analyze_panel(z_out, columns, mean_profile, hurst_horizontal)
        panel['n_columns'] = columns.shape[0]
        panels[field] = panel
    return panels


def print_ratios(panels):
    """Print both panels' ratios; return the fitted lambda.

    The definition is the geometric mean over the h and qt panels under the
    second-smallest-lag anchor. The comparison-window anchor is printed
    alongside but never used for the fit.
    """
    print()
    print(f"  field   H_v     ratio (2nd-lag anchor)   "
          f"ratio ({comparison_window_grid_units[0]}-"
          f"{comparison_window_grid_units[1]} grid units)")
    for field, panel in panels.items():
        print(f"  {field:<7s} {panel['hurst_vertical']:<7.3f} "
              f"{panel['ratio']:<24.4f} {panel['ratio_window']:.4f}")

    def geometric_mean(key):
        return float(np.exp(np.mean(np.log(
            [panel[key] for panel in panels.values()]))))

    print()
    print(f"  DEFINITION, 2-panel geometric mean:")
    print(f"    2nd-lag anchor : {geometric_mean('ratio'):.5f}")
    print(f"    {comparison_window_grid_units[0]}-{comparison_window_grid_units[1]} units : "
          f"{geometric_mean('ratio_window'):.5f}")
    return geometric_mean('ratio')


def save_panels(path, panels, haar_to_mhat, fitted_lambda):
    blob = {'haar_to_mhat': haar_to_mhat, 'fitted_lambda': fitted_lambda}
    for field, panel in panels.items():
        for key, value in panel.items():
            blob[f"{field}_{key}"] = value
    np.savez(path, **blob)
    print(f"wrote {path}")


def load_panels(path):
    """Inverse of save_panels: rebuild the panels dict from a round .npz.

    Lets `figure` re-render without re-running the simulations.
    """
    blob = np.load(path)
    panels = {field: {key[len(field) + 1:]: blob[key] for key in blob.files
                      if key.startswith(field + '_')}
              for field in ('h', 'qt')}
    return panels, float(blob['haar_to_mhat']), float(blob['fitted_lambda'])


def make_figure(panels, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    field_labels = {'h': 'Moist static energy $h$',
                    'qt': 'Total water $q_t$'}

    for ax, field in zip(axes, ('h', 'qt')):
        panel = panels[field]
        lags = panel['physical_lags']
        hurst_vertical = panel['hurst_vertical']
        k_z_L = panel['k_z_L']

        ax.loglog(lags / 1e3, panel['haar_mean'], 'o', ms=3,
                  color=color_mean, label='mean profile', zorder=3)
        ax.loglog(lags / 1e3, panel['haar_column'], 's', ms=2.5,
                  color=color_column,
                  label=f"columns (n={panel['n_columns']})", zorder=3)

        line_lags = np.logspace(np.log10(lags[0]), np.log10(k_z_L), 50)
        ax.loglog(line_lags / 1e3,
                  10 ** (panel['intercept_anchored']
                         + hurst_vertical * np.log10(line_lags)),
                  color=color_line, lw=1.3, ls='--',
                  label=f"slope $H_v$ = {hurst_vertical:.2f}", zorder=2)
        ax.loglog(line_lags / 1e3,
                  10 ** (panel['intercept_mean'] + np.log10(line_lags)),
                  color=color_reference, lw=0.9, ls='-.',
                  label='slope 1', zorder=2)

        ax.axvline(k_z_L / 1e3, color="#7B9E87", lw=1.0)
        # Label at the BOTTOM of the rule: the curves rise to the right, so
        # the top-right corner is where the data is.
        ax.annotate(f"$k_{{z,L}}$", (k_z_L / 1e3, 0.0),
                    xycoords=('data', 'axes fraction'), va='bottom',
                    xytext=(3, 5), textcoords='offset points',
                    fontsize=7, color="#7B9E87")
        ax.annotate(f"ratio = {panel['ratio']:.2f}",
                    (0.03, 0.95), xycoords='axes fraction', va='top',
                    fontsize=7.5)

        ax.grid(True, which='both')
        ax.set_title(field_labels[field])
        ax.set_xlabel('vertical lag [km]')
        ax.set_ylabel('$\\hat{M}_1(\\ell_z)$')

    axes[0].legend(fontsize=6.5, loc='lower right')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    print(f"wrote {path}")


def main():
    # CUDA backend for the fluctuation functions. float64 is set EXPLICITLY:
    # scaleinvariance defaults to float32 (backend._numerical_precision), and
    # the h columns are ~3.5e5 J/kg pooled over ~800k columns, so a float32
    # accumulator would put ULP at O(10^-2) against the differences we are
    # measuring. Set before any haar_fluctuation call.
    scaleinvariance.set_backend('torch')
    scaleinvariance.set_device('cuda')
    scaleinvariance.set_numerical_precision('float64')

    parser = argparse.ArgumentParser()
    parser.add_argument('round', choices=['round1', 'round2', 'figure',
                                          'iterate'])
    parser.add_argument('--hurst-horizontal', type=float, required=True,
                        help='H_h to calibrate at. Required and never '
                             'defaulted: lambda is H_h-specific, so a run '
                             'must say which H_h it belongs to. Tags every '
                             'output file.')
    parser.add_argument('--npz', default=None,
                        help="figure only: the round .npz to re-render from "
                             "(default this H_h's round1). Runs no "
                             "simulations; the round-1 vs round-2 title and "
                             "output filename follow the lambda in the file.")
    parser.add_argument('--smoke', action='store_true',
                        help='tiny configuration, plumbing test only')
    parser.add_argument('--lambda-value', type=float, default=None,
                        help='round2/iterate: use this lambda instead of the '
                             'one in round1.npz. The bound projection makes '
                             'the response slightly sublinear in lambda, so '
                             'the round-2 residual need not be exactly 1; '
                             'this flag is the manual version of `iterate`.')
    parser.add_argument('--tol', type=float, default=0.02,
                        help='iterate: stop when |residual - 1| < tol.')
    parser.add_argument('--max-iters', type=int, default=3,
                        help='iterate: give up after this many iterations.')
    parser.add_argument('--start-iter', type=int, default=1,
                        help='iterate: number the first iteration from here. '
                             'Use it to resume without overwriting the '
                             'round2_*_iter*.npz already on disk.')
    args = parser.parse_args()
    hurst_horizontal = args.hurst_horizontal

    global outer_scale, nx, ny, domain_height, profile_dz, n_seeds
    global column_subsample
    if args.smoke:
        # Keeps the 2:1 aspect and outer_scale = x extent = 2 * y extent, so
        # the y-axis fold of the outermost class is actually exercised.
        nx = 128
        ny = 64
        outer_scale = dx * nx
        domain_height = 6000      # k_z,L ~ 4.68 km must fit inside, with
                                  # enough levels that the lag ladder still
                                  # reaches past it (the guard is real: at
                                  # 5200 m it trips, 4642 m vs 4678 m)
        profile_dz = 2.0
        n_seeds = 1
        column_subsample = 1

    # Every output carries its H_h, so calibrations at different H_h coexist.
    tag = f"Hh{hurst_horizontal:g}" + ('_smoke' if args.smoke else '')
    round1_path = HERE / f"round1_{tag}.npz"
    round2_path = HERE / f"round2_{tag}.npz"
    figure_lambda1_path = FIGS / f"lambda_calibration_{tag}_lambda1.png"
    figure_fitted_path = FIGS / f"lambda_calibration_{tag}_fitted.png"
    FIGS.mkdir(exist_ok=True)

    # Startup sanity block. k_z_L must fit inside domain_height (simulate()
    # raises otherwise) and dz/nz follow from the finest class k = 2*dx.
    k_z_L = float(_k_z(anisotropy, outer_scale, spheroscale))
    k_z_finest = float(_k_z(anisotropy, 2 * dx, spheroscale))
    dz_out = k_z_finest / 2
    print(f"grid {nx}x{ny} at dx = {dx} m: domain {nx * dx / 1e3:.0f} x "
          f"{ny * dy / 1e3:.0f} km, outer scale {outer_scale / 1e3:.0f} km "
          f"(x extent / {outer_scale / (ny * dy):.0f} y extents)")
    print(f"k_z,L = {k_z_L / 1e3:.2f} km inside {domain_height / 1e3:.0f} km "
          f"domain; dz ~ {dz_out:.0f} m, nz ~ {int(domain_height / dz_out)}")
    print(f"vertical fit range {k_z_finest:.0f} m -> {k_z_L:.0f} m = "
          f"{np.log2(k_z_L / k_z_finest):.2f} octaves; H_z = {H_z:.4f}")
    print(f"{n_seeds} seed(s); H_h = {hurst_horizontal} -> "
          f"H_v = H_h/H_z = {hurst_horizontal / H_z:.4f}; lambda is the "
          f"geometric mean over the h and qt panels")

    if args.round == 'figure':
        source = Path(args.npz) if args.npz else round1_path
        panels, haar_to_mhat, fitted_lambda = load_panels(source)
        print(f"\nre-rendering from {source} "
              f"(simulations ran at lambda = {haar_to_mhat:.5f})")
        print_ratios(panels)
        if haar_to_mhat == 1.0:
            make_figure(panels,
                        f"$H_h$ = {hurst_horizontal}, round 1: "
                        f"$\\lambda$ = 1 (open crossover); fitted "
                        f"$\\lambda$ = {fitted_lambda:.4f}",
                        figure_lambda1_path)
        else:
            make_figure(panels,
                        f"$H_h$ = {hurst_horizontal}, round 2: fitted "
                        f"$\\lambda$ = {haar_to_mhat:.4f} "
                        f"(closed crossover)",
                        figure_fitted_path)
    elif args.round == 'iterate':
        # Fixed-point loop over the round-2 machinery: the response is
        # slightly sublinear in lambda (the bound projection), so one
        # round-2 pass lands near but not on residual = 1.
        if args.lambda_value is None:
            lambda_value = float(np.load(round1_path)['fitted_lambda'])
            print(f"\nITERATE from lambda = {lambda_value:.5f} "
                  f"(from {round1_path.name}), tol = {args.tol}, "
                  f"max {args.max_iters} iterations")
        else:
            lambda_value = args.lambda_value
            print(f"\nITERATE from lambda = {lambda_value:.5f} "
                  f"(--lambda-value), tol = {args.tol}, "
                  f"max {args.max_iters} iterations")

        last = args.start_iter + args.max_iters - 1
        for iteration in range(args.start_iter, last + 1):
            panels = run_all_panels(hurst_horizontal, lambda_value)
            residual = print_ratios(panels)
            save_panels(HERE / f"round2_{tag}_iter{iteration}.npz",
                        panels, lambda_value, lambda_value)
            make_figure(panels,
                        f"$H_h$ = {hurst_horizontal}, iteration {iteration}: "
                        f"$\\lambda$ = {lambda_value:.4f} "
                        f"(residual {residual:.3f})",
                        FIGS / f"lambda_calibration_{tag}_iter"
                               f"{iteration}.png")
            print(f"ITER {iteration}: lambda={lambda_value:.5f} "
                  f"residual={residual:.5f}")
            if abs(residual - 1.0) < args.tol:
                print(f"CONVERGED lambda={lambda_value:.5f}")
                break
            lambda_value = lambda_value * residual
        else:
            print(f"MAX ITERATIONS: next suggested lambda={lambda_value:.5f}")
    elif args.round == 'round1':
        print(f"\nROUND 1: H_h = {hurst_horizontal}, lambda = 1")
        panels = run_all_panels(hurst_horizontal, 1.0)
        fitted_lambda = print_ratios(panels)
        print(f"\n=== FITTED LAMBDA = {fitted_lambda:.5f} ===")
        print(f"(i.e. HAAR_TO_MHAT = {fitted_lambda:.5f} = 1.0 / "
              f"{1.0 / fitted_lambda:.5f})")
        save_panels(round1_path, panels, 1.0, fitted_lambda)
        make_figure(panels,
                    f"Round 1: simulations at $\\lambda$ = 1 "
                    f"(open crossover); fitted $\\lambda$ = "
                    f"{fitted_lambda:.4f}",
                    figure_lambda1_path)
    else:
        if args.lambda_value is None:
            fitted_lambda = float(np.load(round1_path)['fitted_lambda'])
            print(f"\nROUND 2: lambda = {fitted_lambda:.5f} "
                  f"(from {round1_path.name})")
        else:
            fitted_lambda = args.lambda_value
            print(f"\nROUND 2: lambda = {fitted_lambda:.5f} (--lambda-value)")
        panels = run_all_panels(hurst_horizontal, fitted_lambda)
        residual = print_ratios(panels)
        print(f"\n=== residual ratio at the fitted lambda: {residual:.5f} "
              f"(target 1.0) ===")
        print(f"if you want to iterate: --lambda-value "
              f"{fitted_lambda * residual:.5f}")
        save_panels(round2_path, panels, fitted_lambda, fitted_lambda)
        make_figure(panels,
                    f"Round 2: simulations at the fitted $\\lambda$ = "
                    f"{fitted_lambda:.4f} (closed crossover)",
                    figure_fitted_path)


if __name__ == '__main__':
    main()
