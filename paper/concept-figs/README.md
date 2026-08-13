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

- **wavelet_discretization** — `fig:wavelet discretization`: the envelope
  choice only matters near the grid. Mean fluctuation function (q = 1) of
  an fBm field under three envelopes — structure function (field
  differences), Mexican hat, Haar — and their local exponents. All three
  reach H = 1/2 by l ~ 5-8 and agree above it, so the envelope choice
  stops mattering a few cells above the grid; below that each carries a
  transient set by how its taps land on the grid. The script prints those
  crossover scales — they are the number the surrounding paragraph quotes
  for `l >~ ?`. Two insets in panel (a): one realization of the field
  (upper left), and the three envelopes drawn as their actual discrete
  taps at one nominal l (lower right), which is where the asymmetry comes
  from — at l = 8 the structure function uses 2 taps, the Haar 8, the
  Mexican hat 49.

  `FIELD_MODEL` picks the test field, and the choice decides *which*
  discretization the figure measures — there are two. The **analysis**
  envelope is the T the fluctuation function is computed with, which is
  what separates the three curves. The **synthesis** envelope is the T
  the field was built from; STEAM deposits discretized Mexican hats on
  working grids and sums them.

  `'fbm'` (default) is a deposit-and-sum construction, so it carries
  both, and its small-scale transient is the one that applies to the
  model's own fields. `'random_walk'` is a cumsum — the exact discrete
  random walk, no synthesis step — so it isolates the analysis envelope,
  at the cost of not resembling anything STEAM produces. It exists only
  at H = 1/2 and raises if `H_TRUE` is anything else.

  The two differ a lot, and it is worth knowing which you are looking at.
  At H = 1/2, local exponent at l = 1, 2, 3, 4:

  | field | l=1 | 2 | 3 | 4 |
  |---|---|---|---|---|
  | cumsum | 0.500 | 0.500 | 0.500 | 0.500 |
  | `fBm_1D_circulant` | 0.611 | 0.587 | 0.536 | 0.529 |
  | `fBm_1D` (LS2010) | 0.611 | 0.587 | 0.537 | 0.529 |

  The two fBm paths agree to three digits, so the transient is inherent
  to deposit-and-sum synthesis rather than a kernel choice — which is
  itself the paragraph's point about discretization being generic. Under
  `'fbm'` every curve including the structure function approaches H from
  above and the crossovers are l ~ 5-8; under `'random_walk'` the
  structure function is flat at 0.500 and only the extended envelopes
  have a transient, with crossovers at l ~ 10-12.

  Unlike the other figures here this one is *measured*, not drawn: it
  calls `scaleinvariance.wavelet_fluctuation` on the walks, so it needs
  the project venv (`../../.venv/bin/python wavelet_discretization.py`)
  rather than bare numpy + matplotlib. Parameters are at the top of the
  file (walk length, realization count, q, lag ladder, slope-fit width,
  tolerance). `MAX_SEP_FRACTION = 1/32` is deliberate: the Mexican-hat
  kernel is ~5.8 l wide, so at l near L/8 it occupies a sizeable fraction
  of the domain and its exponent drifts high — a finite-domain artifact,
  not the discretization effect the figure is about.

Supplement figures:

- **s1_working_grid_resolution** — main-text appendix (sparsity factors):
  the continuum turbulon envelope against its piecewise-constant
  representation on the class's own working grid at Delta x_k = k/2
  (s_x = 1).

  Unlike the three concept figures above, this one uses the *true* 3D
  envelope sampled at the cell centers `simulate._turbulon_envelope`
  actually builds, so the numbers on it are the real kernel taps —
  including the discrete admissibility correction, which is why the peak
  reads A_opt = 2.968 rather than 3. NOTE it is a center-line CUT of a
  3D function: the cut is mean-positive even though the 3D kernel sums
  to zero, because the negative shell lives off-axis where the r^2
  volume element gives it the multiplicity to cancel the core.

- **s2_interpolation_retention** — why the trilinear-regrid loss is a
  one-time factor and not something that accumulates down the chain. Same
  envelope and same A_opt as s1, but against the *linear interpolant*
  through the k/2 samples rather than the piecewise-constant
  representation. The first regrid replaces the envelope by the chord
  polygon through its samples; the second shortcuts that polygon's own
  kinks; from the third regrid on the field is an exact fixed point of the
  interpolation operator (verified: peak frozen at 2.6152 for m >= 2 at
  every grid size tried, 7 to 513 cells). This is the figure for the
  "interpolation does not preserve the mean absolute value" paragraph.

Notes:
- Envelope drawn with the 1D-form Mexican hat 2(1-a)e^(-a/2)
  (deeper negative lobe than the true 2D form — a deliberate
  visualization choice); circular domain truncated at R = 1.10 where the
  lobe has decayed.
- Regenerate: `python <name>.py` (needs numpy + matplotlib; PDFs are
  transparent-background vector). wavelet_discretization additionally
  needs scaleinvariance >= 0.15 — run it with `../../.venv/bin/python`.
  v10c/v11c accept `--seed N` for
  exploration; locked seeds are the defaults.
- `turblib.py` = palette/amplitudes/projection/ridgeline machinery;
  `wirelib.py` = circular polar wireframe, cutout wall, axis triads.
- The exploration archive (all rejected variants, point clouds, GIFs)
  lives in `~/Downloads/turbulon-figure/`.
