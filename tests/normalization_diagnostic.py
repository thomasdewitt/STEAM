#!/usr/bin/env python
"""Diagnostic: outer-scale normalization via Haar fluctuation crossover.

Runs STEAM with linear h and qt profiles (different slopes), then compares
vertical Haar fluctuations of:
  - The ensemble mean profile (expected slope = 1, smooth/differentiable)
  - Individual 3D output columns (expected slope = H_h/H_z = 3/5 below outer scale)

A correct normalization produces a crossover at the vertical outer scale k_z_L.
"""

import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance
from steam import simulate
from steam.constants import hurst_vertical_anisotropy as H_z, hurst_horizontal as H_h
import netCDF4
from pathlib import Path
import tempfile


def main():
    # ---- Parameters ----
    outer_scale = 5000 * 256       # m, horizontal
    dx = dy = 5000            # m  (L/dx = 32 = 2^5, gives 5 scale classes)
    nx = ny = 256             # domain = 8192 m = 2 * outer_scale
    domain_height = 20000    # m
    profile_dz = .6          # m
    spheroscale = 10    # m
    n_seeds = 3

    # Derived quantities
    H_v = H_h / H_z          # vertical Hurst exponent = (1/3)/(5/9) = 3/5 = 0.6
    k_z_L = spheroscale * (outer_scale / spheroscale) ** H_z

    print(f"Outer scale (horizontal): {outer_scale} m")
    print(f"Vertical outer scale k_z_L: {k_z_L:.0f} m")
    print(f"Expected vertical Hurst: H_v = H_h/H_z = {H_v:.4f}")
    print(f"Spheroscale: {spheroscale} m")
    print(f"H_h = {H_h:.4f}, H_z = {H_z:.4f}")
    print()

    # ---- Profiles: linear with very different slopes ----
    nz_profile = int(domain_height / profile_dz) + 1
    z_profile = np.arange(nz_profile) * profile_dz

    # h: moderate slope (20 kJ/kg over domain)
    # h_profile = 350e3 -20e3 * (z_profile / (domain_height))

    # h: moderate slope (20 kJ/kg over domain) with sharp increase at top
    h_profile = (350e3 - 20e3 * (z_profile / domain_height)) + (370 * 1004 - 330e3) * (np.clip((z_profile - 0.85 * domain_height) / (0.15 * domain_height), 0, 1) ** 2) * (3 - 2 * np.clip((z_profile - 0.85 * domain_height) / (0.15 * domain_height), 0, 1))
    
    # qt: steep relative slope (nearly full range, 18 g/kg over domain)
    qt_profile = 0.020 - 0.0199 * (z_profile / domain_height)

    print(f"h slope: {(h_profile[-1] - h_profile[0]) / domain_height:.2f} J/kg/m")
    print(f"qt slope: {(qt_profile[-1] - qt_profile[0]) / domain_height:.2e} kg/kg/m")
    print()

    # ---- Run simulations ----
    all_h_columns = []
    all_qt_columns = []
    z_out = None

    for seed in range(n_seeds):
        print(f"Running seed {seed + 1}/{n_seeds}...", flush=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.nc"
            simulate(
                h_profile, qt_profile, nx, ny, dx, dy,
                outer_scale, spheroscale, domain_height, profile_dz,
                out, seed=seed, h_min = 0.9 * h_profile.min(), h_max = 1.1 * h_profile.max(), qt_min = 0, qt_max = 1.5 * qt_profile.max()
            )
            ds = netCDF4.Dataset(out)
            h_3d = ds.variables['h'][:]
            qt_3d = ds.variables['qt'][:]
            if z_out is None:
                z_out = ds.variables['z'][:]
            ds.close()

        nz = h_3d.shape[2]
        all_h_columns.append(h_3d.reshape(-1, nz))
        all_qt_columns.append(qt_3d.reshape(-1, nz))

    h_columns = np.concatenate(all_h_columns, axis=0)
    qt_columns = np.concatenate(all_qt_columns, axis=0)
    dz_out = float(z_out[1] - z_out[0])

    print(f"\nOutput grid: dz = {dz_out:.1f} m, nz = {len(z_out)}")
    print(f"Total columns: {h_columns.shape[0]}")
    print()

    # ---- Haar fluctuation analysis ----
    # Mean profile interpolated to output z-grid
    h_mean = np.interp(z_out, z_profile, h_profile)
    qt_mean = np.interp(z_out, z_profile, qt_profile)

    lags, haar_h_mean = scaleinvariance.haar_fluctuation_analysis(h_mean, axis=0)
    _, haar_qt_mean = scaleinvariance.haar_fluctuation_analysis(qt_mean, axis=0)

    # All columns stacked: (n_columns, nz), analyze along axis=1
    _, haar_h_col = scaleinvariance.haar_fluctuation_analysis(h_columns, axis=1)
    _, haar_qt_col = scaleinvariance.haar_fluctuation_analysis(qt_columns, axis=1)

    phys_lags = lags * dz_out

    # ---- Fitting: fixed slopes, fit intercept only ----
    log_lags = np.log10(phys_lags)

    def fit_intercept(log_x, log_y, slope, fit_min, fit_max):
        """Fit log10(y) = intercept + slope * log10(x) over [fit_min, fit_max]."""
        mask = (10**log_x >= fit_min) & (10**log_x <= fit_max)
        n_pts = mask.sum()
        if n_pts < 2:
            print(f"  WARNING: only {n_pts} points in [{fit_min:.0f}, {fit_max:.0f}] m")
            return np.nan
        residuals = log_y[mask] - slope * log_x[mask]
        return float(np.mean(residuals))

    def crossover_scale(intercept_a, slope_a, intercept_b, slope_b):
        """Physical scale where two power laws intersect."""
        if slope_a == slope_b:
            return np.nan
        log_r = (intercept_a - intercept_b) / (slope_b - slope_a)
        return 10**log_r

    # Fitting ranges
    mean_fit_min = 4 * dz_out
    mean_fit_max = domain_height / 4
    col_fit_min = 4 * dz_out
    col_fit_max = k_z_L / 2   # stay below expected crossover

    print(f"Mean fit range: [{mean_fit_min:.0f}, {mean_fit_max:.0f}] m")
    print(f"Column fit range: [{col_fit_min:.0f}, {col_fit_max:.0f}] m")
    print()

    results = {}
    for name, haar_mean, haar_col in [
        ('h', haar_h_mean, haar_h_col),
        ('qt', haar_qt_mean, haar_qt_col),
    ]:
        log_mean = np.log10(haar_mean)
        log_col = np.log10(haar_col)

        int_mean = fit_intercept(log_lags, log_mean, 1.0, mean_fit_min, mean_fit_max)
        int_col = fit_intercept(log_lags, log_col, H_v, col_fit_min, col_fit_max)

        r_cross = crossover_scale(int_mean, 1.0, int_col, H_v)

        results[name] = {
            'haar_mean': haar_mean,
            'haar_col': haar_col,
            'intercept_mean': int_mean,
            'intercept_col': int_col,
            'crossover': r_cross,
        }

        ratio = r_cross / k_z_L if not np.isnan(r_cross) else np.nan
        print(f"{name}:")
        print(f"  Mean profile fit intercept: {int_mean:.4f}")
        print(f"  Column fit intercept (slope={H_v:.3f}): {int_col:.4f}")
        print(f"  Crossover scale: {r_cross:.0f} m")
        print(f"  Expected k_z_L: {k_z_L:.0f} m")
        print(f"  Ratio (crossover / expected): {ratio:.3f}")
        print()

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    haar_axes = axes[0]
    profile_axes = axes[1]

    # Colors
    c_mean = '#2c7bb6'
    c_col = '#d7191c'
    c_fit_mean = '#abd9e9'
    c_fit_col = '#fdae61'
    c_expected = '#1a9641'
    c_measured = '#7b3294'

    for ax, (name, label) in zip(haar_axes, [
        ('h', 'Moist static energy $h$'),
        ('qt', 'Total water $q_t$'),
    ]):
        r = results[name]

        # Data
        ax.loglog(phys_lags, r['haar_mean'], 'o', color=c_mean,
                  ms=4, lw=1.5, label='Mean profile', zorder=3)
        ax.loglog(phys_lags, r['haar_col'], 's', color=c_col,
                  ms=3, lw=1.5, label=f'3D columns (n={h_columns.shape[0]})', zorder=3)

        # Fitted lines spanning the full range
        fit_r = np.logspace(np.log10(phys_lags.min() * 0.8),
                            np.log10(phys_lags.max() * 1.2), 200)
        log_fit_r = np.log10(fit_r)

        line_mean = 10**(r['intercept_mean'] + 1.0 * log_fit_r)
        line_col = 10**(r['intercept_col'] + H_v * log_fit_r)

        ax.loglog(fit_r, line_mean, '--', color=c_fit_mean, lw=2,
                  label='Slope = 1 (smooth)', zorder=2)
        ax.loglog(fit_r, line_col, '--', color=c_fit_col, lw=2,
                  label=f'Slope = {H_v:.2f} (turbulent)', zorder=2)

        # Crossover and expected outer scale
        ax.axvline(k_z_L, color=c_expected, ls='-', lw=2, alpha=0.7,
                   label=f'Expected $k_{{z,L}}$ = {k_z_L:.0f} m')
        if not np.isnan(r['crossover']):
            ax.axvline(r['crossover'], color=c_measured, ls=':', lw=2,
                       label=f'Measured crossover = {r["crossover"]:.0f} m')

        ax.set_xlabel('Vertical separation [m]')
        ax.set_ylabel('Haar fluctuation (first order)')
        ax.set_title(label)
        ax.legend(fontsize=6.5, loc='upper left')
        ax.grid(True, alpha=0.2, which='both')

    profile_specs = [
        ('h', 'Moist static energy $h$', h_columns, h_mean, 'J kg$^{-1}$'),
        ('qt', 'Total water $q_t$', qt_columns, qt_mean, 'kg kg$^{-1}$'),
    ]
    profile_rng = np.random.default_rng(42)

    for ax, (name, label, columns, target_mean, unit) in zip(profile_axes, profile_specs):
        n_show = min(500, columns.shape[0])
        if n_show == columns.shape[0]:
            sample_idx = np.arange(columns.shape[0])
        else:
            sample_idx = profile_rng.choice(columns.shape[0], n_show, replace=False)

        for idx in sample_idx:
            ax.plot(columns[idx], z_out / 1e3, color='black', alpha=0.05, lw=0.3)

        ax.plot(columns.mean(axis=0), z_out / 1e3, color=c_col, lw=2,
                label='Column mean', zorder=3)
        ax.plot(target_mean, z_out / 1e3, color=c_mean, lw=2, ls='--',
                label='Target profile', zorder=4)

        ax.set_xlabel(f'{label} [{unit}]')
        ax.set_ylabel('Height [km]')
        ax.set_title(f'{label} profiles')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Outer-scale normalization diagnostic\n'
        f'$L$={outer_scale} m, $l_s$={spheroscale} m, '
        f'$k_{{z,L}}$={k_z_L:.0f} m, '
        f'$H_h$={H_h:.3f}, $H_z$={H_z:.3f}',
        fontsize=10,
    )
    plt.tight_layout()
    out_path = Path(__file__).parent / 'normalization_diagnostic.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
