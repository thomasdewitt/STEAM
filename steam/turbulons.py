"""Turbulon norm functions and envelope shape functions.

Norm functions: (X, Y, Z, spheroscale) -> r_norm_sq
    All accept spheroscale; isotropic ignores it.

Shape functions: (r_norm_sq, k) -> kernel array (same shape as r_norm_sq)
    ratio_sq = r_norm_sq / σ², σ = k/π is the dimensionless argument.
    All shapes are designed so that peak PSD occurs at wavelength k.

NORMS and SHAPES are the authoritative registries of valid string names.
Add an entry here when adding a new function.
"""

import numpy as np
from .constants import hurst_vertical_anisotropy as H_z


# Registries — extend these when adding new functions
NORMS = {'isotropic_norm', 'canonical_anisotropic_norm'}
SHAPES = {'mexican_hat', 'morlet_omega0_6'}


# ---------------------------------------------------------------------------
# Norm functions
# ---------------------------------------------------------------------------

def isotropic_norm(X, Y, Z, spheroscale=None):
    """Isotropic Euclidean norm squared: r_norm² = X² + Y² + Z²."""
    return X**2 + Y**2 + Z**2


def canonical_anisotropic_norm(X, Y, Z, spheroscale):
    """Canonical anisotropic STEAM norm squared.

    (Apxeq:canonical anisotropic norm)
    r_norm² = spheroscale² * ((X/spheroscale)² + (Y/spheroscale)²
                               + (|Z|/spheroscale)^(2/H_z))
    """
    return spheroscale**2 * (
        (X / spheroscale) ** 2
        + (Y / spheroscale) ** 2
        + (np.abs(Z) / spheroscale) ** (2.0 / H_z)
    )


# ---------------------------------------------------------------------------
# Shape functions
# ---------------------------------------------------------------------------

def mexican_hat(r_norm_sq, k):
    """3D Mexican hat wavelet envelope.

    (Apxeq:turbulon shape)
    T(r) = (1 - ρ) exp(-ρ/2),  ρ = r_norm² / σ²,  σ = k/π

    σ = k/π places the 3D PSD peak at wavelength k, so dx = k/2
    Nyquist-samples the peak wavelength.
    """
    ratio_sq = r_norm_sq / (k / np.pi) ** 2
    return (1.0 - ratio_sq) * np.exp(-ratio_sq / 2.0)


def morlet_omega0_6(r_norm_sq, k):
    """3D Morlet wavelet envelope with ω₀ = 6.

    T(r) = cos(ω₀ · r_norm) · exp(-ρ/2),
    where ω₀ = 6/k and ρ = r_norm² / σ² with σ = k/π (same as mexican_hat).
    """
    ratio_sq = r_norm_sq / (k / np.pi) ** 2
    omega_0 = 6.0 / k
    return np.cos(omega_0 * np.sqrt(r_norm_sq)) * np.exp(-ratio_sq / 2.0)
