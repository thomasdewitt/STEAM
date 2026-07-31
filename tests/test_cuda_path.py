"""The cuda device dispatch produces the same statistics as the cpu path.

The GPU is a genuine dispatch, not a fallback: the square runs on 'cuda' and
the nest on 'cpu', and the two must be the same model. They are not the same
realization -- the GPU's reduction order differs from numpy's pairwise sum
and the level means feed the bounded add's bisection -- so this compares the
per-level profiles the paper reads, and the bounds, which are exact on both.
"""

import numpy as np
import pytest
import netCDF4
import torch

from steam.simulate import simulate

H_MIN = 315 * 1004.0
H_MAX = 355 * 1004.0
QT_MIN = 0.0
QT_MAX = 30 / 1000.0


def _level_profiles(path):
    with netCDF4.Dataset(path) as ds:
        h = np.asarray(ds["h"][:], dtype=np.float64)
        qt = np.asarray(ds["qt"][:], dtype=np.float64)
    return {"h_mean": h.mean(axis=(0, 1)), "h_std": h.std(axis=(0, 1)),
            "qt_mean": qt.mean(axis=(0, 1)), "qt_std": qt.std(axis=(0, 1)),
            "h_lo": h.min(), "h_hi": h.max(),
            "qt_lo": qt.min(), "qt_hi": qt.max()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu_level_statistics(tmp_path):
    nz = 50
    z = np.arange(nz) * 30.0
    h_profile = 340e3 - 20e3 * (z / z.max())
    qt_profile = 0.018 - 0.016 * (z / z.max())

    profiles = {}
    for device in ("cpu", "cuda"):
        out = tmp_path / f"{device}.nc"
        simulate(h_profile, qt_profile, nx=64, ny=64, dx=500, dy=500,
                 outer_scale=8000, spheroscale=100,
                 domain_height=1200, profile_dz=30,
                 output_path=out, seed=17, device=device)
        profiles[device] = _level_profiles(out)

    cpu = profiles["cpu"]
    gpu = profiles["cuda"]

    # Level means: the amplitude ladder and the mean profile, per level.
    for name, scale in (("h_mean", H_MAX - H_MIN), ("qt_mean", QT_MAX)):
        assert np.abs(gpu[name] - cpu[name]).max() < 1e-3 * scale, name

    # Level spreads: relative, but only where the level actually has spread --
    # qt's topmost levels are dry to ~1e-10 kg/kg and their relative
    # comparison is meaningless.
    for name in ("h_std", "qt_std"):
        active = cpu[name] > 1e-3 * cpu[name].max()
        assert active.any(), name
        rel = np.abs(gpu[name][active] - cpu[name][active]) / cpu[name][active]
        assert rel.max() < 1e-3, f"{name} max rel {rel.max():.3g}"

    # Bounds hold exactly on both paths.
    for tag in ("cpu", "cuda"):
        assert profiles[tag]["h_lo"] >= H_MIN
        assert profiles[tag]["h_hi"] <= H_MAX
        assert profiles[tag]["qt_lo"] >= QT_MIN
        assert profiles[tag]["qt_hi"] <= QT_MAX
