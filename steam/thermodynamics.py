"""Thermodynamic recovery of diagnostic fields from h and qt."""

import os
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import netCDF4
import torch
from . import constants
from .output import compression_kwargs
from .utils import (
    available_memory_bytes,
    MEMORY_HEADROOM_BYTES,
    CUDA_MEMORY_HEADROOM_BYTES,
)
from .constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
    gas_constant_dry_air as Rd,
)


def recover_diagnostics(h, qt, z_values, surface_pressure, device='cpu'):
    """Recover T, qv, qc, qi, p from 3D h and qt fields.

    Proceeds upward from z=z_values[0], vectorized over (nx, ny) at each level.

    ``device`` is the caller's own device. On 'cuda' the column solve runs on
    the GPU (see _recover_diagnostics_cuda), transfers included; on 'cpu' it
    runs here. A real dispatch, not a fallback -- the host path is the
    reference. The two agree to ~6e-5 K in T and ~5e-3 Pa in p, which is the
    float32 storage granularity of the output variables; they are not
    bit-identical, because CUDA's expf is not glibc's.

    The pressure march is the one quantity here that ACCUMULATES: p at a level
    is p at the level below times a factor, so a 115-level column is 115
    chained multiplies, and float32 rounding compounds down it. It carries a
    float64 accumulator on both devices -- one 2D slab, not the 3D field --
    which costs nothing measurable and takes the column error from 0.199 Pa to
    0.005 Pa against an exact reference (the float32 output granularity at
    ~1e5 Pa is 0.004 Pa, so this is at the representation floor). Everything
    else is pointwise per level, where float32 is already correctly rounded
    and float64 would buy nothing.

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
        Optional 2D field for elevated-bottom data.

    Returns
    -------
    dict with keys 'T', 'qv', 'qc', 'qi', 'p', each shape (nx, ny, nz).
    """
    if device == 'cuda':
        return _recover_diagnostics_cuda(h, qt, z_values, surface_pressure)

    nx, ny, nz = h.shape
    T = np.empty_like(h)
    qv = np.empty_like(h)
    qc = np.empty_like(h)
    qi = np.empty_like(h)
    p = np.empty_like(h)

    # Surface pressure for all columns. float64 accumulator for the march --
    # see the docstring; the level physics below stays in h's own dtype.
    p_cur = np.empty((nx, ny), dtype=np.float64)
    p_cur[:] = surface_pressure

    for iz in range(nz):
        z = z_values[iz]
        p_level = p_cur.astype(h.dtype)
        p[:, :, iz] = p_level
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
            dz = float(z_values[iz + 1] - z_values[iz])
            Tv = T_level.astype(np.float64) * (
                1.0 + 0.608 * qv_level.astype(np.float64))
            p_cur *= np.exp(-g * dz / (Rd * Tv))

    return {"T": T, "qv": qv, "qc": qc, "qi": qi, "p": p}


def _es_cuda(T):
    """Bolton (1980) saturation vapor pressure [Pa], on the card. (Eq:bolton)"""
    return 611.2 * torch.exp(17.67 * (T - 273.15) / (T - 29.65))


def _level_cuda(h_level, qt_level, p_level, z, n_iterations=5):
    """One level of the column solve, branch-free.

    The host path gathers the saturated points, solves on the compacted
    subset, and scatters back (h_level[saturated]). Here the Newton solve
    runs on EVERY point and torch.where picks the branch: at the ~30%
    saturated fraction of a production field that is ~3x the Newton
    arithmetic, but it buys a gather, a scatter and the divergence, and the
    card is not arithmetic-bound on this kernel. Both orderings are exact --
    the solve reads nothing outside its own point.
    """
    T_dry = (h_level - Lv * qt_level - g * z) / cp

    es_dry = _es_cuda(T_dry)
    qvs_dry = 0.622 * es_dry / (p_level - es_dry)
    saturated = qt_level > qvs_dry

    T = T_dry
    for _ in range(n_iterations):
        es = _es_cuda(T)
        qvs = 0.622 * es / (p_level - es)
        f = cp * T + Lv * qvs + g * z - h_level
        des_dT = 4302.6 * es / (T - 29.65) ** 2       # Eq:des_dT
        dqvs_dT = 0.622 * p_level / (p_level - es) ** 2 * des_dT
        T = T - f / (cp + Lv * dqvs_dT)               # Eq:fprime_explicit

    es_sat = _es_cuda(T)
    qv_sat = 0.622 * es_sat / (p_level - es_sat)

    T_level = torch.where(saturated, T, T_dry)
    qv_level = torch.where(saturated, qv_sat, qt_level)

    # Eq:phase_partition, Eq:lambda
    condensate = torch.clamp(qt_level - qv_level, min=0.0)
    lam = torch.clamp((T_level - 235.15) / (273.15 - 235.15), 0.0, 1.0)
    return T_level, qv_level, lam * condensate, (1.0 - lam) * condensate


def _recover_diagnostics_cuda(h, qt, z_values, surface_pressure):
    """recover_diagnostics on the GPU. Takes and returns host arrays.

    The z-march cannot vectorize -- each level's pressure is the level below
    times a factor -- so this is 115 sequential launches over (nx, ny) slabs,
    not one kernel. That is fine: the slabs are millions of points wide, which
    is where the parallelism is, and the column direction is only 115 long.

    torch.compile was measured SLOWER here (0.265 s against 0.226 s eager on a
    production chunk): 115 small per-level graphs do not amortize the guard
    overhead. Eager is the faster and simpler path.

    Transfers are part of the call. At the production chunk that is 241 MB up
    and 603 MB down against ~0.23 s of compute, and it still runs 8.4x the
    host path end to end.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "recover_diagnostics: device='cuda' requested but torch.cuda "
            "is unavailable"
        )
    dev = torch.device('cuda')
    out_dtype = h.dtype
    ht = torch.as_tensor(np.ascontiguousarray(h), device=dev)
    qtt = torch.as_tensor(np.ascontiguousarray(qt), device=dev)
    nx, ny, nz = ht.shape

    T = torch.empty_like(ht)
    qv = torch.empty_like(ht)
    qc = torch.empty_like(ht)
    qi = torch.empty_like(ht)
    p = torch.empty_like(ht)

    # float64 column accumulator, one 2D slab -- see recover_diagnostics.
    p_cur = torch.empty((nx, ny), dtype=torch.float64, device=dev)
    p_cur[:] = torch.as_tensor(np.asarray(surface_pressure, dtype=np.float64),
                               device=dev)

    for iz in range(nz):
        z = float(z_values[iz])
        p_level = p_cur.to(ht.dtype)
        p[:, :, iz] = p_level
        T_l, qv_l, qc_l, qi_l = _level_cuda(ht[:, :, iz], qtt[:, :, iz],
                                            p_level, z)
        T[:, :, iz] = T_l
        qv[:, :, iz] = qv_l
        qc[:, :, iz] = qc_l
        qi[:, :, iz] = qi_l

        # Hypsometric equation for next level (Eq:hypsometric)
        if iz < nz - 1:
            dz = float(z_values[iz + 1] - z_values[iz])
            Tv = T_l.to(torch.float64) * (1.0 + 0.608 * qv_l.to(torch.float64))
            p_cur *= torch.exp(-g * dz / (Rd * Tv))

    return {name: v.cpu().numpy().astype(out_dtype, copy=False)
            for name, v in (("T", T), ("qv", qv), ("qc", qc), ("qi", qi),
                            ("p", p))}


DIAGNOSTICS_MAX_CHUNK_NX = 128     # tuned at nz=115; see _diagnostics_chunk_nx


def _diagnostics_chunk_nx(nx, ny, nz, itemsize, n_workers, device):
    """x-columns per diagnostics chunk that ``n_workers`` of them fit in memory.

    A chunk is (chunk_nx, ny, nz), and each one in flight holds SEVEN arrays of
    that shape: h and qt in, T/qv/qc/qi/p out. Everything the level loop
    allocates on top of that is a 2D slab and does not scale with chunk_nx.
    So the working set is 7 * chunk_nx * ny * nz * itemsize per worker, and the
    number that has to fit is n_workers times that.

    Sizing this from x-columns alone -- a fixed chunk_nx=128 -- is what the
    default used to be, and it holds only at the nz it was tuned at. 128
    columns of the 2048^2 x 115 production square is 115 MB per array, 1.6 GB
    of VRAM for two workers, which is nothing. The same 128 columns of a
    2048^2 x 835 demo field is 836 MiB per array and 11.4 GiB for two workers,
    which does not fit on a 16 GiB card next to anything else and OOMed at the
    seventh allocation. nz varies by a factor of seven across the cases this is
    run on, so the column count cannot be the fixed quantity -- the bytes have
    to be.

    The chunk size is a pure partition of an embarrassingly parallel
    calculation: every column is solved from its own h, qt and surface
    pressure, reading nothing outside itself. So unlike the cascade's batch
    sizes, this one does not enter the realization, and sizing it from
    whatever VRAM is free at the moment cannot make the same seed give a
    different field. test_compute_diagnostics_chunking_matches_full pins that.

    Capped at DIAGNOSTICS_MAX_CHUNK_NX because the HDF5 chunkshape of the
    diagnostic variables is derived from it -- a chunk sized to fill the card
    would be a poor read granularity for everything downstream.
    """
    per_column = 7 * ny * nz * itemsize
    if device == 'cuda':
        torch.cuda.empty_cache()      # so mem_get_info sees the true free VRAM
        free, _ = torch.cuda.mem_get_info()
        budget = free - CUDA_MEMORY_HEADROOM_BYTES
        where = f"{free / 1024**3:.1f} GiB free VRAM"
    else:
        available = available_memory_bytes()
        if available is None:         # not Linux; no preflight, keep the cap
            return min(nx, DIAGNOSTICS_MAX_CHUNK_NX)
        budget = available - MEMORY_HEADROOM_BYTES
        where = f"{available / 1024**3:.1f} GiB available RAM"
    columns = int(budget // (n_workers * per_column))
    if columns < 1:
        raise MemoryError(
            f"compute_diagnostics (device={device!r}) needs "
            f"{n_workers * per_column / 1024**3:.1f} GiB for a single-column "
            f"chunk of field shape {(nx, ny, nz)} across {n_workers} workers, "
            f"but there is only {where}; refusing to risk OOM. Pass fewer "
            f"n_workers, or free the device."
        )
    return min(nx, DIAGNOSTICS_MAX_CHUNK_NX, columns)


def compute_diagnostics(nc_path, chunk_nx=None, group=None, compress=None,
                        n_workers=None, device='cpu'):
    """Compute T, qv, qc, qi, p from h/qt in a NetCDF file, writing in x-chunks.

    Opens the file in r+ mode, reads h, qt, z, and the starting pressure
    (2D ``p_bottom`` if present, else scalar ``surface_pressure``), and appends
    the diagnostic variables T, qv, qc, qi, p (float32, same shape as h/qt).

    Parameters
    ----------
    nc_path : str or Path
        Path to the NetCDF file produced by simulate().
    chunk_nx : int or None
        Number of x-columns to process at a time. None (default) sizes the
        chunk against free memory on the device actually being used, so that
        ``n_workers`` chunks fit; see _diagnostics_chunk_nx for why a fixed
        column count is the wrong unit. An explicit value is honoured as
        given, with no memory check.
    group : str or None
        NetCDF group to operate on. None means the root group; pass e.g.
        "refinements/r0" to run diagnostics on a refinement group.
    compress : bool or None
        If True, new diagnostic variables are written with the
        ``steam.constants.output_compression`` filter. None (default) uses
        the module-level ``steam.constants.output_compress`` setting.
    n_workers : int or None
        Number of threads used to compute chunks in parallel. None (default)
        picks ``min(8, n_chunks, cpu_count)`` on the host and 2 on 'cuda'.
        Pass 1 for serial. Results are bit-identical regardless of
        ``n_workers`` since each chunk is an independent per-column
        calculation.

        The host default is 8 rather than cpu_count because the solve stops
        scaling there: measured per-chunk 1.88 s at one thread, 0.85 s at
        eight, 0.88 s at sixteen -- a 2.2x ceiling set by memory bandwidth,
        not by thread count. On 'cuda' two is enough to keep the card fed
        while the main thread does NetCDF I/O (measured 8.3 / 5.3 / 5.2 s at
        one / two / three workers, against 26.6 s on the host).
    device : str
        'cpu' (default) or 'cuda'. See recover_diagnostics for what the
        dispatch changes and what it does not. At the production square the
        card runs the whole pass in 5.3 s against the host's 26.6 s; the
        remaining floor is NetCDF I/O, measured at 2.9 s with the physics
        removed entirely.
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

    # Worker count first: the chunk has to be sized so that this many of them
    # fit at once, and the count itself does not depend on the chunking except
    # through the clamp to n_chunks below.
    if n_workers is None:
        n_workers = 2 if device == 'cuda' else min(8, os.cpu_count() or 1)
    n_workers = max(1, int(n_workers))

    if chunk_nx is None:
        chunk_nx = _diagnostics_chunk_nx(
            nx, ny, nz, grp.variables["h"].dtype.itemsize, n_workers, device)

    # Create output variables if they don't exist
    diag_names = {"T": ("K", "temperature"),
                  "qv": ("kg/kg", "water vapor mixing ratio"),
                  "qc": ("kg/kg", "cloud liquid water mixing ratio"),
                  "qi": ("kg/kg", "cloud ice mixing ratio"),
                  "p": ("Pa", "pressure")}
    for name, (units, long_name) in diag_names.items():
        if name not in grp.variables:
            diag_chunks = (min(chunk_nx, nx), min(64, ny), nz)
            v = grp.createVariable(name, "f4", ("x", "y", "z"),
                                   chunksizes=diag_chunks,
                                   **compression_kwargs(compress, diag_chunks))
            v.units = units
            v.long_name = long_name

    h_var = grp.variables["h"]
    qt_var = grp.variables["qt"]

    chunks = [(x0, min(x0 + chunk_nx, nx)) for x0 in range(0, nx, chunk_nx)]
    n_chunks = len(chunks)
    n_workers = min(n_workers, n_chunks)

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
              f"[{n_workers}w x {chunk_nx}col, {elapsed:5.1f}s elapsed, "
              f"~{eta:5.1f}s left]   ", end="\r", flush=True)

    if n_workers == 1:
        done = 0
        for x0, x1 in chunks:
            h_chunk = h_var[x0:x1, :, :]
            qt_chunk = qt_var[x0:x1, :, :]
            result = recover_diagnostics(h_chunk, qt_chunk, z_values,
                                         _p_slice(x0, x1), device=device)
            _write_result(x0, x1, result)
            done += 1
            _progress(done)
    else:
        # Keep exactly n_workers chunks in flight: as soon as ANY worker
        # finishes, write its result and immediately read and submit the next.
        # In fixed batches of n_workers the pool sat idle through the whole
        # serial read/compress/write leg of every batch, which is most of the
        # wall time (measured 86.7 s vs 24.0 s at 2048^2 x 115). Reads and
        # writes stay on the main thread -- the HDF5 layer is not re-entrant.
        # Writes land in completion order rather than chunk order, which is
        # safe because each chunk owns a disjoint x-slab.
        done = 0
        next_chunk = 0
        in_flight = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            while next_chunk < n_chunks or in_flight:
                while len(in_flight) < n_workers and next_chunk < n_chunks:
                    x0, x1 = chunks[next_chunk]
                    next_chunk += 1
                    h_chunk = h_var[x0:x1, :, :]
                    qt_chunk = qt_var[x0:x1, :, :]
                    fut = pool.submit(recover_diagnostics, h_chunk, qt_chunk,
                                      z_values, _p_slice(x0, x1),
                                      device=device)
                    in_flight[fut] = (x0, x1)
                finished, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for fut in finished:
                    x0, x1 = in_flight.pop(fut)
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
