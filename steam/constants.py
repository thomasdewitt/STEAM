"""Physical constants for STEAM."""

specific_heat_dry_air = 1004.0          # cp [J/(kg·K)]
latent_heat_vaporization = 2.5e6        # Lv [J/kg]
gravity = 9.81                          # g [m/s²]
gas_constant_dry_air = 287.04           # Rd [J/(kg·K)]

# The two constants below must be changed together
# hurst_horizontal = 0.5               # H_h, Kolmogorov/Corrsin-Obukhov
# haar_to_mhat = 0.19691
hurst_horizontal = 0.45 
haar_to_mhat = 0.25518 
                                     # (0.45 alternative: see haar_to_mhat --
                                     # the two MUST be changed together)
hurst_vertical_anisotropy = 5/9


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
