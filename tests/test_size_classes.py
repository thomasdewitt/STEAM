"""Behavioral tests for different size-class discretizations."""

import math
import numpy as np
import netCDF4

from steam import simulate


def _run_stats(tmp_path, target_gap_factor):
    nx = ny = 64
    dx = dy = 100.0
    outer_scale = 400.0
    domain_height = 600.0
    profile_dz = 10.0
    spheroscale = 100.0

    z_profile = np.arange(int(domain_height / profile_dz) + 1, dtype=np.float64) * profile_dz
    h_profile = 340e3 - 8e3 * (z_profile / domain_height)
    qt_profile = 0.016 - 0.010 * (z_profile / domain_height)

    ratio = outer_scale / (2 * dx)
    n_size_classes = max(
        2,
        int(round(1 + math.log(ratio) / math.log(target_gap_factor))),
    )

    out = tmp_path / f"gap_{str(target_gap_factor).replace('.', '_')}.nc"
    simulate(
        h_profile, qt_profile,
        nx, ny, dx, dy,
        outer_scale, spheroscale, domain_height, profile_dz,
        out,
        seed=7,
        n_size_classes=n_size_classes,
        h_min=float(h_profile.min()) - 1e3,
        h_max=float(h_profile.max()) + 1e3,
        qt_min=0.0,
        qt_max=float(qt_profile.max()) + 5e-3,
    )

    ds = netCDF4.Dataset(out)
    h = np.asarray(ds.variables["h"][:])
    qt = np.asarray(ds.variables["qt"][:])
    z_out = np.asarray(ds.variables["z"][:])
    ds.close()

    h_mean = np.interp(z_out, z_profile, h_profile)[np.newaxis, np.newaxis, :]
    qt_mean = np.interp(z_out, z_profile, qt_profile)[np.newaxis, np.newaxis, :]
    h_pert = h - h_mean
    qt_pert = qt - qt_mean

    return {
        "target_gap_factor": target_gap_factor,
        "n_size_classes": n_size_classes,
        "h_mean": float(h.mean()),
        "qt_mean": float(qt.mean()),
        "h_std": float(h_pert.std()),
        "qt_std": float(qt_pert.std()),
        "h_abs_q95": float(np.quantile(np.abs(h_pert), 0.95)),
        "qt_abs_q95": float(np.quantile(np.abs(qt_pert), 0.95)),
    }


def test_different_size_class_counts_give_grossly_similar_statistics(tmp_path):
    target_gap_factors = [2.0, 1.4, 1.2, 1.05]
    stats = [_run_stats(tmp_path, gap) for gap in target_gap_factors]
    baseline = stats[0]

    for current in stats[1:]:
        np.testing.assert_allclose(
            current["h_mean"], baseline["h_mean"], rtol=1e-3
        )
        np.testing.assert_allclose(
            current["qt_mean"], baseline["qt_mean"], rtol=2e-2
        )

        for key in ("h_std", "qt_std", "h_abs_q95", "qt_abs_q95"):
            ratio = current[key] / baseline[key]
            assert 0.6 <= ratio <= 1.5, (
                f"{key} changed too much for target_gap_factor="
                f"{current['target_gap_factor']}: ratio={ratio:.3f}, "
                f"n_size_classes={current['n_size_classes']}"
            )
