"""Thermodynamic recovery of diagnostic fields from h and qt."""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import netCDF4
from . import constants
from .constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
    gas_constant_dry_air as Rd,
)


def recover_diagnostics(h, qt, z_values, surface_pressure):
    """Recover T, qv, qc, qi, p from 3D h and qt fields.

    Proceeds upward from z=z_values[0], vectorized over (nx, ny) at each level.

    Parameters
    ----------
    h : ndarray, shape (nx, ny, nz)
        Moist static energy [J/kg].
    qt : ndarray, shape (nx, ny, nz)
        Total water mixing ratio [kg/kg].
    z_values : ndarray, shape (nz,)
        Heights [m].
    surface_pressure : float or ndarray, shape (nx, ny)
        Pressure at z=z_values[0] [Pa]. Scalar for a root simulation;
        2D field for an elevated-bottom inset from refine().

    Returns
    -------
    dict with keys 'T', 'qv', 'qc', 'qi', 'p', each shape (nx, ny, nz).
    """
    nx, ny, nz = h.shape
    T = np.empty_like(h)
    qv = np.empty_like(h)
    qc = np.empty_like(h)
    qi = np.empty_like(h)
    p = np.empty_like(h)

    # Surface pressure for all columns
    p[:, :, 0] = surface_pressure

    for iz in range(nz):
        z = z_values[iz]
        p_level = p[:, :, iz]
        h_level = h[:, :, iz]
        qt_level = qt[:, :, iz]

        # Eq:Tdry — dry temperature assuming no condensation
        T_dry = (h_level - Lv * qt_level - g * z) / cp

        # Check saturation
        qvs_dry = _saturation_mixing_ratio(T_dry, p_level)
        saturated = qt_level > qvs_dry

        # Start with unsaturated solution
        T_level = T_dry.copy()
        qv_level = qt_level.copy()

        # Saturated points: Newton solve (Eq:h_saturated)
        if np.any(saturated):
            T_sat = _newton_saturated_T(
                h_level[saturated], qt_level[saturated],
                z, p_level[saturated], T_dry[saturated],
            )
            T_level[saturated] = T_sat
            qv_level[saturated] = _saturation_mixing_ratio(T_sat, p_level[saturated])

        # Phase partition (Eq:phase_partition, Eq:lambda)
        condensate = np.maximum(qt_level - qv_level, 0.0)
        lam = np.clip((T_level - 235.15) / (273.15 - 235.15), 0.0, 1.0)
        qc_level = lam * condensate
        qi_level = (1.0 - lam) * condensate

        T[:, :, iz] = T_level
        qv[:, :, iz] = qv_level
        qc[:, :, iz] = qc_level
        qi[:, :, iz] = qi_level

        # Hypsometric equation for next level (Eq:hypsometric)
        if iz < nz - 1:
            dz = z_values[iz + 1] - z_values[iz]
            Tv = T_level * (1.0 + 0.608 * qv_level)
            p[:, :, iz + 1] = p_level * np.exp(-g * dz / (Rd * Tv))

    return {"T": T, "qv": qv, "qc": qc, "qi": qi, "p": p}


def compute_diagnostics(nc_path, chunk_nx=128, group=None, compress=None,
                        n_workers=None):
    """Compute T, qv, qc, qi, p from h/qt in a NetCDF file, writing in x-chunks.

    Opens the file in r+ mode, reads h, qt, z, and the starting pressure
    (2D ``p_bottom`` if present, else scalar ``surface_pressure``), and appends
    the diagnostic variables T, qv, qc, qi, p (float32, same shape as h/qt).

    Parameters
    ----------
    nc_path : str or Path
        Path to the NetCDF file produced by simulate().
    chunk_nx : int
        Number of x-columns to process at a time.
    group : str or None
        NetCDF group to operate on. None means the root group; pass e.g.
        "refinements/r0" to run diagnostics on a refinement group.
    compress : bool or None
        If True, new diagnostic variables are written with zlib
        compression at complevel=4. None (default) uses the module-level
        ``steam.constants.output_compress`` setting.
    n_workers : int or None
        Number of threads used to compute chunks in parallel. None (default)
        picks ``min(8, n_chunks, cpu_count)``. Pass 1 for serial. Results are
        bit-identical regardless of ``n_workers`` since each chunk is an
        independent per-column calculation.
    """
    if compress is None:
        compress = constants.output_compress

    ds = netCDF4.Dataset(nc_path, "r+")
    grp = ds if group is None else ds[group]
    nx = len(grp.dimensions["x"])
    ny = len(grp.dimensions["y"])
    nz = len(grp.dimensions["z"])
    z_values = grp.variables["z"][:]

    if "p_bottom" in grp.variables:
        starting_pressure = grp.variables["p_bottom"][:]
    else:
        starting_pressure = float(grp.surface_pressure)

    # Create output variables if they don't exist
    diag_names = {"T": ("K", "temperature"),
                  "qv": ("kg/kg", "water vapor mixing ratio"),
                  "qc": ("kg/kg", "cloud liquid water mixing ratio"),
                  "qi": ("kg/kg", "cloud ice mixing ratio"),
                  "p": ("Pa", "pressure")}
    for name, (units, long_name) in diag_names.items():
        if name not in grp.variables:
            v = grp.createVariable(name, "f4", ("x", "y", "z"),
                                   zlib=compress, complevel=4 if compress else 0,
                                   chunksizes=(min(chunk_nx, nx), min(64, ny), nz))
            v.units = units
            v.long_name = long_name

    h_var = grp.variables["h"]
    qt_var = grp.variables["qt"]

    chunks = [(x0, min(x0 + chunk_nx, nx)) for x0 in range(0, nx, chunk_nx)]
    n_chunks = len(chunks)

    if n_workers is None:
        n_workers = min(8, n_chunks, os.cpu_count() or 1)
    n_workers = max(1, min(int(n_workers), n_chunks))

    def _p_slice(x0, x1):
        if isinstance(starting_pressure, np.ndarray):
            return starting_pressure[x0:x1, :]
        return starting_pressure

    def _write_result(x0, x1, result):
        for name in ("T", "qv", "qc", "qi", "p"):
            grp.variables[name][x0:x1, :, :] = result[name].astype(np.float32)

    tag = f" (group '{group}')" if group else ""

    t_start = time.perf_counter()

    def _progress(done):
        elapsed = time.perf_counter() - t_start
        pct = 100.0 * done / n_chunks
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (n_chunks - done) / rate if rate > 0 else 0.0
        print(f"  diagnostics: {done}/{n_chunks} chunks ({pct:5.1f}%) "
              f"[{n_workers}w, {elapsed:5.1f}s elapsed, ~{eta:5.1f}s left]   ",
              end="\r", flush=True)

    if n_workers == 1:
        done = 0
        for x0, x1 in chunks:
            h_chunk = h_var[x0:x1, :, :]
            qt_chunk = qt_var[x0:x1, :, :]
            result = recover_diagnostics(h_chunk, qt_chunk, z_values,
                                         _p_slice(x0, x1))
            _write_result(x0, x1, result)
            done += 1
            _progress(done)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for i in range(0, n_chunks, n_workers):
                batch = chunks[i:i + n_workers]
                futures = {}
                for x0, x1 in batch:
                    h_chunk = h_var[x0:x1, :, :]
                    qt_chunk = qt_var[x0:x1, :, :]
                    fut = pool.submit(recover_diagnostics, h_chunk, qt_chunk,
                                      z_values, _p_slice(x0, x1))
                    futures[fut] = (x0, x1)
                for fut in as_completed(futures):
                    x0, x1 = futures[fut]
                    _write_result(x0, x1, fut.result())
                    done += 1
                    _progress(done)

    ds.close()
    total = time.perf_counter() - t_start
    print(f"  diagnostics: done in {total:.1f}s, written to {nc_path}{tag}"
          "                    ")


def _saturation_vapor_pressure(T):
    """Bolton (1980) saturation vapor pressure [Pa]. (Eq:bolton)"""
    return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))


def _saturation_mixing_ratio(T, p):
    """Saturation mixing ratio [kg/kg]. (Eq:qvsat)"""
    es = _saturation_vapor_pressure(T)
    return 0.622 * es / (p - es)


def _newton_saturated_T(h, qt, z, p, T_guess, n_iterations=5):
    """Vectorized Newton solve for saturated temperature. (Eq:h_saturated)

    Solves f(T) = cp*T + Lv*qvsat(T,p) + g*z - h = 0.
    """
    T = T_guess.copy()
    for _ in range(n_iterations):
        es = _saturation_vapor_pressure(T)
        qvs = 0.622 * es / (p - es)

        f = cp * T + Lv * qvs + g * z - h

        # Eq:fprime_explicit
        des_dT = 4302.6 * es / (T - 29.65) ** 2       # Eq:des_dT
        dqvs_dT = 0.622 * p / (p - es) ** 2 * des_dT  # Eq:dqvsat_dT
        fprime = cp + Lv * dqvs_dT

        T = T - f / fprime

    return T
