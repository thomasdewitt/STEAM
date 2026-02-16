"""Thermodynamic recovery of diagnostic fields from h and qt."""

import numpy as np
from .constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
    gas_constant_dry_air as Rd,
)


def recover_diagnostics(h, qt, z_values, surface_pressure):
    """Recover T, qv, qc, qi, p from 3D h and qt fields.

    Proceeds upward from the surface, vectorized over (nx, ny) at each z level.

    Parameters
    ----------
    h : ndarray, shape (nx, ny, nz)
        Moist static energy [J/kg].
    qt : ndarray, shape (nx, ny, nz)
        Total water mixing ratio [kg/kg].
    z_values : ndarray, shape (nz,)
        Heights [m].
    surface_pressure : float
        Surface pressure [Pa].

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
