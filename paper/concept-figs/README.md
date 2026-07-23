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
