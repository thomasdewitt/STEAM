"""Physical constants for STEAM."""

specific_heat_dry_air = 1004.0          # cp [J/(kg·K)]
latent_heat_vaporization = 2.5e6        # Lv [J/kg]
gravity = 9.81                          # g [m/s²]
gas_constant_dry_air = 287.04           # Rd [J/(kg·K)]

hurst_horizontal = 0.45              # H_h, Kolmogorov/Corrsin-Obukhov
hurst_vertical_anisotropy = 5/9      # H_z, aspect-ratio scaling exponent

# lambda, the delivery amplitude in the paper: the factor converting the mean
# profile's vertical Haar fluctuation at the local outer scale into the
# outer-class turbulon amplitude, C_{Phi,L}(z) = lambda * |Haar_{k_z,L}(<Phi>)|.
#
# Operational definition (2026-07-30, replaces the earlier unit-turbulon
# measurement): lambda is FITTED end-to-end from the outer-scale crossover
# criterion. Full STEAM runs on linear h and qt profiles give the vertical
# Haar fluctuation of the 3D columns; a line of fixed slope H_v = H_h/H_z is
# anchored at the second-smallest lag -- i.e. extrapolated from BELOW, where
# the cascade scales cleanly -- and read at the vertical outer scale k_z,L.
# lambda is the geometric mean over the h and qt panels of the ratio
# (mean-profile fluctuation) / (column line) there, iterated to a fixed point
# because the bound projection makes the response slightly sublinear.
# Extrapolating from below is deliberate: near k_z,L the profile and the
# turbulence mix, which bends the measured curve, and the criterion is about
# the cascade's amplitude, not the mixing.
#
# WARNING: lambda is calibrated AT a specific hurst_horizontal and is not
# transferable -- the ratio varies systematically with H_h. Changing
# hurst_horizontal above INVALIDATES this value; recalibrate with
# turbulon-analysis/lambda_calibration/calibrate_lambda.py.
#
# Converged 2026-07-30: fixed-point iteration seeded at 0.4, residual 0.992
# at iteration 4; per-field ratios at convergence h = 0.964, qt = 1.021.
haar_to_mhat = 0.28927

# NetCDF output default: whether h, qt, diagnostic variables, and p_bottom
# are written with zlib compression (complevel=4). Overridden by an explicit
# compress= kwarg on simulate(), write_netcdf(), compute_diagnostics().
output_compress = False
