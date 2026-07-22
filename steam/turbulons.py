"""Turbulon envelope shape functions.

Shape functions: (r_norm_sq, k) -> kernel array (same shape as r_norm_sq)
    ratio_sq = r_norm_sq / σ², σ = k/π is the dimensionless argument.
    All shapes are designed so that peak PSD occurs at wavelength k.

SHAPES is the authoritative registry of valid string names.
Add an entry here when adding a new function.

Grid anisotropy is handled separately in simulate._k_z (on the grid, not
in the envelope): the envelope is always evaluated with isotropic
r_norm² = X² + Y² + Z² in cell-index space, and anisotropy enters
through per-class dz = k_z(k, spheroscale) / (2*s_z).
"""

import numpy as np


# Registry — extend when adding new functions
SHAPES = {'mexican_hat', 'morlet_omega0_6'}


def mexican_hat(r_norm_sq, k):
    """3D Mexican hat (Marr) wavelet envelope: -Laplacian of a Gaussian.

    (Apxeq:turbulon shape)
    T(r) = (3 - ρ) exp(-ρ/2),  ρ = r_norm² / σ²,  σ = k/π

    The leading constant equals the number of dimensions: (3 - ρ) makes
    the envelope integrate to zero over 3D space (wavelet admissibility),
    unlike the 1D profile (1 - ρ) rotated into 3D, whose volume integral
    is negative (the r² volume element overweights the negative shell).
    With σ = k/π the shell-integrated PSD peaks near wavelength 1.15 k
    and the 1D center column's PSD near sqrt(2) k.
    """
    ratio_sq = r_norm_sq / (k / np.pi) ** 2
    return (3.0 - ratio_sq) * np.exp(-ratio_sq / 2.0)


def morlet_omega0_6(r_norm_sq, k):
    """3D Morlet wavelet envelope with ω₀ = 6.

    T(r) = cos(ω₀ · r_norm) · exp(-ρ/2),
    where ω₀ = 6/k and ρ = r_norm² / σ² with σ = k/π (same as mexican_hat).
    """
    ratio_sq = r_norm_sq / (k / np.pi) ** 2
    omega_0 = 6.0 / k
    return np.cos(omega_0 * np.sqrt(r_norm_sq)) * np.exp(-ratio_sq / 2.0)
