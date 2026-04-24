"""Physical constants for STEAM."""

specific_heat_dry_air = 1004.0          # cp [J/(kg·K)]
latent_heat_vaporization = 2.5e6        # Lv [J/kg]
gravity = 9.81                          # g [m/s²]
gas_constant_dry_air = 287.04           # Rd [J/(kg·K)]

hurst_horizontal = 1 / 3               # H_h, Kolmogorov/Corrsin-Obukhov
hurst_vertical_anisotropy = 5/9      # H_z, aspect-ratio scaling exponent

# NetCDF output default: whether h, qt, diagnostic variables, and p_bottom
# are written with zlib compression (complevel=4). Overridden by an explicit
# compress= kwarg on simulate(), refine(), write_netcdf(), compute_diagnostics().
output_compress = False
