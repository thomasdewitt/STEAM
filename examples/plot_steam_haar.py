#!/usr/bin/env python3
"""Plot fluctuation/C1 diagnostics plus 2D slices and 1D trajectories from STEAM NetCDF."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scaleinvariance
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---- Config ----
INPUT_NETCDF = REPO_ROOT / "examples/steam_ensemble.nc"
PLOT_DIR = REPO_ROOT / "examples/plots"
SCALING_METHOD = "haar"  # "haar" or "structure"
MIN_SEP_X = None
MAX_SEP_X = None
MIN_SEP_Y = None  # currently unused by fluctuation analysis
MAX_SEP_Y = None  # currently unused by fluctuation analysis
MIN_SEP_Z = None
MAX_SEP_Z = None


if __name__ == "__main__":
    if SCALING_METHOD not in ("haar", "structure"):
        raise ValueError("SCALING_METHOD must be 'haar' or 'structure'.")

    if SCALING_METHOD == "haar":
        hurst_fn = scaleinvariance.haar_fluctuation_hurst
        c1_scaling_method = "haar_fluctuation"
        fluctuation_label = "Haar fluctuation"
        method_tag = "haar"
    else:
        hurst_fn = scaleinvariance.structure_function_hurst
        c1_scaling_method = "structure_function"
        fluctuation_label = "Structure function"
        method_tag = "structure"

    FLUC_DIR = PLOT_DIR / f"{method_tag}_fluctuations"
    PCOLOR_DIR = PLOT_DIR / "pcolormesh_2d"
    TRAJ_DIR = PLOT_DIR / "trajectories_1d"
    PROFILE_DIR = PLOT_DIR / "profiles"
    FLUC_DIR.mkdir(parents=True, exist_ok=True)
    PCOLOR_DIR.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    with Dataset(INPUT_NETCDF, "r") as ds:
        x = np.asarray(ds.variables["x"][:])
        y = np.asarray(ds.variables["y"][:])
        z = np.asarray(ds.variables["z"][:])
        dx = float(ds.getncattr("dx"))
        dz = float(ds.getncattr("dz"))
        nrun = len(ds.dimensions["run"])
        nx = len(x)
        ny = len(y)
        nz = len(z)
        ix = nx // 2
        iy = ny // 2
        iz = nz // 2
        pcolor_seed = 0
        if "seed" in ds.variables:
            seeds = np.asarray(ds.variables["seed"][:], dtype=int)
            matches = np.where(seeds == pcolor_seed)[0]
            if matches.size == 0:
                raise ValueError(
                    f"Requested pcolormesh seed {pcolor_seed} not found in NetCDF seed variable."
                )
            pcolor_run_idx = int(matches[0])
        else:
            pcolor_run_idx = 0
            print("seed variable not found; using run index 0 for pcolormesh plots.")

        h_in = np.asarray(ds.variables["h_profile_input"][:])[None, :]
        qt_in = np.asarray(ds.variables["qt_profile_input"][:])[None, :]

        h_out = np.asarray(ds.variables["h"][:])
        h_out_H, h_out_H_err, h_out_lags, h_out_vals, h_out_fit = hurst_fn(
            h_out,
            axis=3,
            min_sep=MIN_SEP_Z,
            max_sep=MAX_SEP_Z,
            return_fit=True,
        )
        h_out_c1, h_out_c1_err = scaleinvariance.two_point_intermittency_exponent(
            h_out,
            axis=3,
            scaling_method=c1_scaling_method,
            min_sep=MIN_SEP_Z,
            max_sep=MAX_SEP_Z,
        )
        del h_out

        qt_out = np.asarray(ds.variables["qt"][:])
        qt_out_H, qt_out_H_err, qt_out_lags, qt_out_vals, qt_out_fit = hurst_fn(
            qt_out,
            axis=3,
            min_sep=MIN_SEP_Z,
            max_sep=MAX_SEP_Z,
            return_fit=True,
        )
        qt_out_c1, qt_out_c1_err = scaleinvariance.two_point_intermittency_exponent(
            qt_out,
            axis=3,
            scaling_method=c1_scaling_method,
            min_sep=MIN_SEP_Z,
            max_sep=MAX_SEP_Z,
        )
        del qt_out

        h_in_H, h_in_H_err, h_in_lags, h_in_vals, h_in_fit = hurst_fn(
            h_in,
            axis=1,
            min_sep=MIN_SEP_Z,
            max_sep=MAX_SEP_Z,
            return_fit=True,
        )
        qt_in_H, qt_in_H_err, qt_in_lags, qt_in_vals, qt_in_fit = hurst_fn(
            qt_in,
            axis=1,
            min_sep=MIN_SEP_Z,
            max_sep=MAX_SEP_Z,
            return_fit=True,
        )
        h_in_c1, h_in_c1_err = scaleinvariance.two_point_intermittency_exponent(
            h_in,
            axis=1,
            scaling_method=c1_scaling_method,
            min_sep=MIN_SEP_Z,
            max_sep=MAX_SEP_Z,
        )
        qt_in_c1, qt_in_c1_err = scaleinvariance.two_point_intermittency_exponent(
            qt_in,
            axis=1,
            scaling_method=c1_scaling_method,
            min_sep=MIN_SEP_Z,
            max_sep=MAX_SEP_Z,
        )

        h_out_lags = np.asarray(h_out_lags)
        h_out_vals = np.asarray(h_out_vals)
        h_out_fit = np.asarray(h_out_fit)
        h_in_lags = np.asarray(h_in_lags)
        h_in_vals = np.asarray(h_in_vals)
        h_in_fit = np.asarray(h_in_fit)
        qt_out_lags = np.asarray(qt_out_lags)
        qt_out_vals = np.asarray(qt_out_vals)
        qt_out_fit = np.asarray(qt_out_fit)
        qt_in_lags = np.asarray(qt_in_lags)
        qt_in_vals = np.asarray(qt_in_vals)
        qt_in_fit = np.asarray(qt_in_fit)

        h_x = h_out_lags * dz
        h_y = h_out_vals
        h_ref_v = h_y[0] * (h_x / h_x[0]) ** (3.0 / 5.0)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.loglog(
            h_x,
            h_y,
            "o-",
            lw=1.4,
            ms=4,
            label=(
                f"output (h), H={float(h_out_H):.3f}±{float(h_out_H_err):.3f}, "
                f"C1={float(h_out_c1):.4f}±{float(h_out_c1_err):.4f}"
            ),
        )
        ax.loglog(h_x, h_out_fit, "-", lw=1.2, alpha=0.9, label="output fit")
        ax.loglog(
            h_in_lags * dz,
            h_in_vals,
            "s--",
            lw=1.2,
            ms=4,
            label=(
                f"input profile, H={float(h_in_H):.3f}±{float(h_in_H_err):.3f}, "
                f"C1={float(h_in_c1):.4f}±{float(h_in_c1_err):.4f}"
            ),
        )
        ax.loglog(h_in_lags * dz, h_in_fit, "--", lw=1.1, alpha=0.9, label="input fit")
        ax.loglog(h_x, h_ref_v, "k-.", lw=1.3, label="ref slope 3/5 (vertical)")
        ax.set_xlabel("Vertical separation [m]")
        ax.set_ylabel(fluctuation_label)
        ax.set_title(f"Vertical {fluctuation_label}: h")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(FLUC_DIR / f"{method_tag}_vertical_h_vs_input.png", dpi=160)
        plt.close(fig)

        qt_x = qt_out_lags * dz
        qt_y = qt_out_vals
        qt_ref_v = qt_y[0] * (qt_x / qt_x[0]) ** (3.0 / 5.0)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.loglog(
            qt_x,
            qt_y,
            "o-",
            lw=1.4,
            ms=4,
            label=(
                f"output (qt), H={float(qt_out_H):.3f}±{float(qt_out_H_err):.3f}, "
                f"C1={float(qt_out_c1):.4f}±{float(qt_out_c1_err):.4f}"
            ),
        )
        ax.loglog(qt_x, qt_out_fit, "-", lw=1.2, alpha=0.9, label="output fit")
        ax.loglog(
            qt_in_lags * dz,
            qt_in_vals,
            "s--",
            lw=1.2,
            ms=4,
            label=(
                f"input profile, H={float(qt_in_H):.3f}±{float(qt_in_H_err):.3f}, "
                f"C1={float(qt_in_c1):.4f}±{float(qt_in_c1_err):.4f}"
            ),
        )
        ax.loglog(qt_in_lags * dz, qt_in_fit, "--", lw=1.1, alpha=0.9, label="input fit")
        ax.loglog(qt_x, qt_ref_v, "k-.", lw=1.3, label="ref slope 3/5 (vertical)")
        ax.set_xlabel("Vertical separation [m]")
        ax.set_ylabel(fluctuation_label)
        ax.set_title(f"Vertical {fluctuation_label}: qt")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(FLUC_DIR / f"{method_tag}_vertical_qt_vs_input.png", dpi=160)
        plt.close(fig)

        other_vars = ("T", "qv", "qc", "qi", "p")
        fig, axes = plt.subplots(3, 2, figsize=(11, 12))
        flat_axes = axes.ravel()
        print(f"Computing {fluctuation_label.lower()} for T, qv, qc, qi, p")
        for i, var in enumerate(other_vars):
            field = np.asarray(ds.variables[var][:])
            x_H, x_H_err, x_lags, x_vals, x_fit = hurst_fn(
                field,
                axis=1,
                min_sep=MIN_SEP_X,
                max_sep=MAX_SEP_X,
                return_fit=True,
            )
            z_H, z_H_err, z_lags, z_vals, z_fit = hurst_fn(
                field,
                axis=3,
                min_sep=MIN_SEP_Z,
                max_sep=MAX_SEP_Z,
                return_fit=True,
            )
            x_lags = np.asarray(x_lags)
            x_vals = np.asarray(x_vals)
            x_fit = np.asarray(x_fit)
            z_lags = np.asarray(z_lags)
            z_vals = np.asarray(z_vals)
            z_fit = np.asarray(z_fit)
            x_c1, x_c1_err = scaleinvariance.two_point_intermittency_exponent(
                field,
                axis=1,
                scaling_method=c1_scaling_method,
                min_sep=MIN_SEP_X,
                max_sep=MAX_SEP_X,
            )
            z_c1, z_c1_err = scaleinvariance.two_point_intermittency_exponent(
                field,
                axis=3,
                scaling_method=c1_scaling_method,
                min_sep=MIN_SEP_Z,
                max_sep=MAX_SEP_Z,
            )
            del field

            ax = flat_axes[i]
            ax.loglog(
                x_lags * dx,
                x_vals,
                "o-",
                lw=1.3,
                ms=3.5,
                label=(
                    f"x, H={float(x_H):.3f}±{float(x_H_err):.3f}, "
                    f"C1={float(x_c1):.4f}±{float(x_c1_err):.4f}"
                ),
            )
            ax.loglog(
                z_lags * dz,
                z_vals,
                "s--",
                lw=1.3,
                ms=3.5,
                label=(
                    f"z, H={float(z_H):.3f}±{float(z_H_err):.3f}, "
                    f"C1={float(z_c1):.4f}±{float(z_c1_err):.4f}"
                ),
            )
            ax.loglog(x_lags * dx, x_fit, "-", lw=1.1, alpha=0.9, label="x fit")
            ax.loglog(z_lags * dz, z_fit, "--", lw=1.1, alpha=0.9, label="z fit")
            ax.set_title(var)
            ax.set_xlabel("Separation [m]")
            ax.set_ylabel(fluctuation_label)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)

        for j in range(len(other_vars), len(flat_axes)):
            flat_axes[j].axis("off")
        fig.suptitle(
            f"{fluctuation_label} with C1 (x and z directions)", fontsize=13
        )
        fig.tight_layout()
        fig.savefig(FLUC_DIR / f"{method_tag}_xz_other_variables.png", dpi=160)
        plt.close(fig)

        all_vars = ("h", "qt", "T", "qv", "qc", "qi", "p")
        print("Computing 2D slices, 1D trajectories, and profiles for all variables")
        for var in all_vars:
            var_obj = ds.variables[var]
            field_mean = np.zeros((nx, ny, nz), dtype=np.float64)
            field_pcolor = None
            profile_sum = np.zeros(nz, dtype=np.float64)
            profile_sumsq = np.zeros(nz, dtype=np.float64)
            profile_min = np.full(nz, np.inf, dtype=np.float64)
            profile_max = np.full(nz, -np.inf, dtype=np.float64)

            for run_idx in range(nrun):
                run_field = np.asarray(var_obj[run_idx, :, :, :], dtype=np.float64)
                if run_idx == pcolor_run_idx:
                    field_pcolor = run_field.copy()
                field_mean += run_field
                profile_sum += run_field.sum(axis=(0, 1))
                profile_sumsq += (run_field ** 2).sum(axis=(0, 1))
                profile_min = np.minimum(profile_min, run_field.min(axis=(0, 1)))
                profile_max = np.maximum(profile_max, run_field.max(axis=(0, 1)))
            field_mean /= nrun
            if field_pcolor is None:
                raise RuntimeError("Failed to load pcolormesh field for requested seed.")

            xz = field_pcolor[:, iy, :].T
            yz = field_pcolor[ix, :, :].T
            xy = field_pcolor[:, :, iz].T

            fig, ax = plt.subplots(figsize=(8, 5))
            pcm = ax.pcolormesh(x, z, xz, shading="auto")
            fig.colorbar(pcm, ax=ax, label=var)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            ax.set_title(f"{var} xz (seed {pcolor_seed}, y index {iy})")
            fig.tight_layout()
            fig.savefig(PCOLOR_DIR / f"{var}_xz.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            pcm = ax.pcolormesh(y, z, yz, shading="auto")
            fig.colorbar(pcm, ax=ax, label=var)
            ax.set_xlabel("y [m]")
            ax.set_ylabel("z [m]")
            ax.set_title(f"{var} yz (seed {pcolor_seed}, x index {ix})")
            fig.tight_layout()
            fig.savefig(PCOLOR_DIR / f"{var}_yz.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            pcm = ax.pcolormesh(x, y, xy, shading="auto")
            fig.colorbar(pcm, ax=ax, label=var)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(f"{var} xy (seed {pcolor_seed}, z index {iz})")
            fig.tight_layout()
            fig.savefig(PCOLOR_DIR / f"{var}_xy.png", dpi=160)
            plt.close(fig)

            traj_x = field_pcolor.mean(axis=(1, 2))
            traj_y = field_pcolor.mean(axis=(0, 2))
            traj_z = field_pcolor.mean(axis=(0, 1))

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(x, traj_x, lw=1.6)
            ax.set_xlabel("x [m]")
            ax.set_ylabel(var)
            ax.set_title(f"{var} trajectory along x (seed {pcolor_seed})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(TRAJ_DIR / f"{var}_x.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(y, traj_y, lw=1.6)
            ax.set_xlabel("y [m]")
            ax.set_ylabel(var)
            ax.set_title(f"{var} trajectory along y (seed {pcolor_seed})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(TRAJ_DIR / f"{var}_y.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(z, traj_z, lw=1.6)
            ax.set_xlabel("z [m]")
            ax.set_ylabel(var)
            ax.set_title(f"{var} trajectory along z (seed {pcolor_seed})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(TRAJ_DIR / f"{var}_z.png", dpi=160)
            plt.close(fig)

            sample_count = nrun * nx * ny
            profile_mean = profile_sum / sample_count
            profile_var = profile_sumsq / sample_count - profile_mean ** 2
            profile_std = np.sqrt(np.maximum(profile_var, 0.0))

            fig, ax = plt.subplots(figsize=(6.8, 6.0))
            ax.fill_betweenx(
                z,
                profile_min,
                profile_max,
                color="tab:blue",
                alpha=0.2,
                label="output min-max",
            )
            ax.fill_betweenx(
                z,
                profile_mean - profile_std,
                profile_mean + profile_std,
                color="tab:blue",
                alpha=0.5,
                label="output mean±std",
            )
            ax.plot(
                profile_mean,
                z,
                color="tab:blue",
                lw=2.0,
                alpha=1.0,
                label="output mean",
            )
            if var == "h":
                ax.plot(
                    np.asarray(ds.variables["h_profile_input"][:]),
                    z,
                    color="tab:orange",
                    lw=1.8,
                    alpha=1.0,
                    label="input profile",
                )
            if var == "qt":
                ax.plot(
                    np.asarray(ds.variables["qt_profile_input"][:]),
                    z,
                    color="tab:orange",
                    lw=1.8,
                    alpha=1.0,
                    label="input profile",
                )
            ax.set_xlabel(var)
            ax.set_ylabel("z [m]")
            ax.set_title(f"{var} profile statistics")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8.5)
            fig.tight_layout()
            fig.savefig(PROFILE_DIR / f"{var}_profile.png", dpi=160)
            plt.close(fig)

    print(f"Done. Plots written under: {PLOT_DIR}")
