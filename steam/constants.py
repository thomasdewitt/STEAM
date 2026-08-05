"""Physical constants for STEAM."""

specific_heat_dry_air = 1004.0          # cp [J/(kg·K)]
latent_heat_vaporization = 2.5e6        # Lv [J/kg]
gravity = 9.81                          # g [m/s²]
gas_constant_dry_air = 287.04           # Rd [J/(kg·K)]

hurst_horizontal = 0.5               # H_h, Kolmogorov/Corrsin-Obukhov
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
# (mean-profile fluctuation) / (column line) there, iterated to a fixed point:
# the response is sublinear in lambda, and that sublinearity survived the
# far-bounds change below, so it is a property of the delivery chain (the
# mean-profile-gradient terms) rather than of the bound projection.
# Extrapolating from below is deliberate: near k_z,L the profile and the
# turbulence mix, which bends the measured curve, and the criterion is about
# the cascade's amplitude, not the mixing.
#
# WARNING: lambda is calibrated AT a specific hurst_horizontal and is not
# transferable -- the ratio varies systematically with H_h. Changing
# hurst_horizontal above INVALIDATES this value; recalibrate with
# turbulon-analysis/lambda_calibration/calibrate_lambda.py.
#
# Converged 2026-08-04 at H_h = 0.5, under the 2026-08-03 far-bounds
# procedure (calibration bounds pushed beyond any reachable value, so the
# projection never clips: lambda is a geometric delivery constant, but the
# bounds are case-specific). Residual 0.99726; per-field ratios at
# convergence h = 0.9982, qt = 0.9963 -- the h/qt split of the old bounded
# procedure (0.964 / 1.021) was the bound projection and is now gone.
#
# Ledger, same production geometry, for anyone comparing against the paper:
#   H_h = 0.45, bounded procedure (pre-2026-08-03) : 0.28927
#   H_h = 0.45, far bounds                         : 0.25518
#   H_h = 0.50, far bounds                         : 0.19691  <- in use
# So the change from the previously committed value is -11.8% procedure and
# -22.8% H_h, not a single effect.
#
# CAVEAT at this H_h: H_v = H_h/H_z = 0.90, and the realized local slope of
# the column Haar reaches 0.989 near k_z,L -- against the saturation ceiling
# of 1, where H_h = 0.45 still had headroom (0.969). The anchored-from-below
# definition is unaffected (it reads the fitted line, not the curve), but the
# top of the vertical range no longer scales cleanly.
haar_to_mhat = 0.19691

# NetCDF output default: whether h, qt, diagnostic variables, and p_bottom
# are compressed at all. Overridden by an explicit compress= kwarg on
# simulate(), write_netcdf(), compute_diagnostics().
output_compress = False

# The filter used when compression is on. blosc_zstd at complevel 1 writes
# 8-11x faster than the former zlib complevel 4 for 5% more bytes (measured
# on production fields: 20.7 s -> 1.8 s for a 1.93 GB h field), and reads
# 2-3x faster. Byte shuffling must be asked for as blosc_shuffle -- netCDF4
# SILENTLY IGNORES shuffle=True for every non-zlib compressor, and without
# the shuffle zstd loses most of its ratio on float32 fields.
#
# Both blosc and zstd are HDF5 filter PLUGINS: a reader without them cannot
# open the variable at all, where zlib is universal. The system netCDF here
# (ncdump, and turbulon-analysis) has both. Switch to "zlib" complevel 1 for
# anything archival or handed to a collaborator.
output_compression = "blosc_zstd"
output_complevel = 1

# Chunks smaller than this are written RAW even when compression is on.
# blosc's HDF5 filter does not degrade gracefully: given a chunk it cannot
# shrink, and no room for its 16-byte header, it fails the write outright
# with "Buffer is uncompressible". Measured on this box: incompressible
# chunks up to 1024 bytes fail, 1536 and above succeed. The coarse
# class-increment classes are exactly that small -- class 0 of a production
# square is 4 x 4 x 12 = 768 bytes of noise -- so the pyramid's top would
# take the whole write down. The threshold below is a wide margin over the
# measured cliff and costs nothing: a chunk this small is a rounding error
# beside the ~11 kB of HDF5 metadata the variable carries anyway.
output_compression_min_chunk_bytes = 16 * 1024
