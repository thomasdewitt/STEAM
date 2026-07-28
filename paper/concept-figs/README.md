# Turbulon concept figures

Three figures explaining the turbulon concept (locked 2026-07-23):

- **w7_quincunx_asym** — what a turbulon is: polar-wireframe sombreros for
  each field (T, p', h, q_t) with signed amplitudes, center shows all four
  superimposed with an asymmetric cutout whose face-on wall carries the
  radial profile fan.
- **v10c_cascade_xz** — vertical slice through the cascade: turbulon
  ellipses at dyadic sizes, aspect ratio (k/l_s)^(4/9) (H_z = 5/9), number
  density per class ~ k^-(1+H_z) (inverse 2D volume). Spheroscale turbulon
  (round, teal) with dimension marker. Locked seed 3.
- **v11c_field_h** — an h field (ink ridgeline) as a superposition of
  turbulons; one strong positive and one negative turbulon highlighted
  (ochre) and extracted as polar wireframes with relative amplitudes.
  Locked seed 1001, sienna connectors.

Supplement figures:

- **s1_working_grid_resolution** — supplement S1 "Working grids". Upper
  panel: the continuum turbulon envelope against its piecewise-constant
  representation on the class's own working grid at Delta x_k = k/2
  (s_x = 1). Lower panel: the same envelope on log axes out to the
  truncation radius, with the current (5k) and candidate (3k)
  support_factor marked against the float32 resolution floor.

  Unlike the three concept figures above, this one uses the *true* 3D
  envelope sampled at the cell centers `simulate._turbulon_envelope`
  actually builds, so the numbers on it are the real kernel taps —
  including the discrete admissibility correction, which is why the peak
  reads A_opt = 2.968 rather than 3. NOTE it is a center-line CUT of a
  3D function: the cut is mean-positive even though the 3D kernel sums
  to zero, because the negative shell lives off-axis where the r^2
  volume element gives it the multiplicity to cancel the core.

Notes:
- Envelope drawn with the 1D-form Mexican hat 2(1-a)e^(-a/2)
  (deeper negative lobe than the true 2D form — a deliberate
  visualization choice); circular domain truncated at R = 1.10 where the
  lobe has decayed.
- Regenerate: `python <name>.py` (needs numpy + matplotlib; PDFs are
  transparent-background vector). v10c/v11c accept `--seed N` for
  exploration; locked seeds are the defaults.
- `turblib.py` = palette/amplitudes/projection/ridgeline machinery;
  `wirelib.py` = circular polar wireframe, cutout wall, axis triads.
- The exploration archive (all rejected variants, point clouds, GIFs)
  lives in `~/Downloads/turbulon-figure/`.
