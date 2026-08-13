"""fig:wavelet discretization — the envelope choice only matters near the grid.

Sect. "Distribution of turbulon amplitudes across scale" argues that the
turbulon envelope function T can be replaced by any zero-mean envelope
without changing the power-law exponent of the fluctuation function

    <Delta f^q>(l) = < |T_l * f|^q >_r ,                (eq:wavelet transform)

and that this equivalence holds only once l is well above the grid scale.
This figure measures where "well above" starts.

The test field is a random walk (cumulative sum of iid Gaussians), whose
Hurst exponent is exactly 1/2. Three envelopes are compared, all
L1-normalized so they differ in amplitude but not in slope:

  - field differences, T_l(r) = delta(r + l/2) - delta(r - l/2), i.e. the
    ordinary structure function (eq:delta pair envelope);
  - a Mexican hat (Ricker), the discrete cousin of the turbulon envelope
    the model actually deposits;
  - a Haar wavelet, the difference of the means of the two halves of a
    width-l window.

Panel (a) is the fluctuation function; panel (b) its local exponent
d log <Delta f^q> / d log l, which is the Hurst exponent resolved scale by
scale. The field-difference curve is a power law at every lag -- for a
random walk <|f(x+l) - f(x)|> = sqrt(2 l / pi) exactly, for every integer
l -- so it has no discretization transient at all. The two extended
envelopes do: both start well below 1/2 at l of a few cells and reach it
from below, the Mexican hat by l ~ 10 and the Haar by l ~ 12 at the
default settings. The script prints those crossover scales, which is the
number the surrounding paragraph quotes.

The asymmetry is not a defect of either wavelet, it is what "discretized"
costs: the nominal lag l means different things to different envelopes.
The field difference needs 2 taps, the Haar needs l, and the Mexican hat's
kernel is ~5.8 l wide (sigma = l / sqrt(3), cut at +/- 5 sigma). The inset
draws all three at the same nominal l so the disparity is visible.

MAX_SEP_FRACTION exists for that reason: at l near L/8 the Mexican-hat
kernel is a sizeable fraction of the domain and its exponent drifts high,
which is a finite-domain artifact and not the discretization effect the
figure is about. L/32 keeps every kernel comfortably inside the walk.

FIELD_MODEL picks the test field, and the choice decides WHICH
discretization the figure measures. There are two, not one:

  - the ANALYSIS envelope, the 𝔗 the fluctuation function is computed
    with -- what panel (b) separates the three curves by;
  - the SYNTHESIS envelope, the 𝔗 the field was BUILT from -- STEAM
    deposits discretized Mexican hats on working grids and sums them.

'fbm' (default) is a deposit-and-sum construction, so it carries both,
and its small-scale transient is the one that applies to the model's own
fields. 'random_walk' is a cumsum, the exact discrete random walk, with
no synthesis step at all: it isolates the analysis envelope, at the cost
of not resembling anything STEAM produces. It only exists at H = 1/2.

The difference is large and worth knowing before reading panel (b).
Measured at H = 1/2, 2000 realizations of 4096 cells, local exponent at
l = 1, 2, 3, 4:

    cumsum             0.500  0.500  0.500  0.500   (exact, no transient)
    fBm_1D_circulant   0.611  0.587  0.536  0.529
    fBm_1D (LS2010)    0.611  0.587  0.537  0.529

Both fBm paths agree to three digits, so the transient is inherent to
deposit-and-sum synthesis rather than a kernel choice -- which is the
paragraph's point about discretization being generic. Under 'fbm' every
curve including the structure function approaches H from above; under
'random_walk' the structure function is flat at 0.500 and only the
extended envelopes have a transient.

Run with the project venv (needs scaleinvariance >= 0.15):
    .venv/bin/python wavelet_discretization.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scaleinvariance as si
from turblib import INK, RULE, LABEL, PALETTE, save

# ---------------------------------------------------------------- parameters
FIELD_MODEL = 'fbm'   # 'fbm' (deposit-and-sum, like STEAM) or 'random_walk'
WALK_SIZE = 4096      # length of each realization, in grid cells
N_REALIZATIONS = 2000   # independent fields, analyzed together (see note below)
H_TRUE = 0.5      # target Hurst exponent; 'random_walk' only supports 1/2
Q_ORDER = 1.0      # moment of the fluctuation function
SEED = 20260813

MAX_SEP_FRACTION = 1 / 32   # largest lag, as a fraction of WALK_SIZE
LAG_SPACING = 'powers of 1.2'   # lag ladder; 'all' for every integer
SLOPE_FIT_POINTS = 3          # points per local log-log slope fit (odd)
EXPONENT_TOL = 0.02       # |slope - H| counted as "converged"

SHOW_KERNEL_INSET = True
INSET_LAG = 8          # nominal l at which the inset draws the envelopes
ANNOTATE_CROSSOVER = True

SHOW_WALK_INSET = True
WALK_INSET_CELLS = 2048   # cells of the example trajectory to draw
WALK_INSET_INDEX = 0      # which realization to draw

# envelope name -> (legend label, color, marker)
ENVELOPES = [
    ('structure_function', 'structure function', PALETTE['T'],  'o'),
    ('mexican_hat',        'Mexican hat',        PALETTE['h'],  's'),
    ('haar',               'Haar wavelet',       PALETTE['qt'], '^'),
]


# ------------------------------------------------------------------ analysis
def test_field(n_fields, size, seed, model=FIELD_MODEL, H=H_TRUE):
    """n_fields independent realizations, stacked on axis 0.

    'random_walk' is the exact discrete object -- nothing sits between the
    field and the envelope, which is why its structure function is a power
    law at every lag. It only exists at H = 1/2.

    'fbm' reaches any H, at the cost of a synthesis transient of its own
    at small l (see the module docstring). fBm_1D_circulant takes no noise
    argument, so the seed has to go through the global RNG.
    """
    if model == 'random_walk':
        if not np.isclose(H, 0.5):
            raise ValueError(
                f"FIELD_MODEL='random_walk' is H = 1/2 by construction, "
                f"got H_TRUE = {H}. Use FIELD_MODEL = 'fbm' for other H.")
        rng = np.random.default_rng(seed)
        return np.cumsum(rng.standard_normal((n_fields, size)), axis=1)
    if model == 'fbm':
        si.set_numerical_precision('float64')
        np.random.seed(seed)
        return np.stack([si.fBm_1D_circulant(size, H)
                         for _ in range(n_fields)], axis=0)
    raise ValueError(f"unknown FIELD_MODEL {model!r}")


def local_exponent(lags, values, n_points=SLOPE_FIT_POINTS):
    """d log F / d log l from a sliding log-log fit over n_points lags."""
    x, y = np.log(lags), np.log(values)
    half = n_points // 2
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        ok = np.isfinite(y[lo:hi])
        if ok.sum() >= 2:
            out[i] = np.polyfit(x[lo:hi][ok], y[lo:hi][ok], 1)[0]
    return out


def crossover_lag(lags, exponents, target=H_TRUE, tol=EXPONENT_TOL):
    """Smallest lag beyond which the local exponent never leaves [target +/- tol]."""
    off = np.abs(exponents - target) > tol
    off &= np.isfinite(exponents)
    if not off.any():
        return lags[0]
    last = np.where(off)[0][-1]
    return lags[last + 1] if last + 1 < len(lags) else np.nan


# All realizations go into ONE call per envelope: the fluctuation function is
# an average of |.|^q over positions AND realizations, and averaging per-walk
# slopes instead would not give the same thing.
walk = test_field(N_REALIZATIONS, WALK_SIZE, SEED)
max_sep = int(WALK_SIZE * MAX_SEP_FRACTION)
# fBm_1D_circulant returns a periodic field, so the fluctuations should wrap
# too: every lag then uses all L samples instead of dropping the edge pairs.
periodic = (FIELD_MODEL == 'fbm')

results = {}
for name, label, color, marker in ENVELOPES:
    lags, values = si.wavelet_fluctuation(walk, wavelet=name, order=Q_ORDER,
                                          axis=1, max_sep=max_sep,
                                          lags=LAG_SPACING, periodic=periodic)
    lags = np.asarray(lags, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values) & (values > 0)
    lags, values = lags[keep], values[keep]
    exponents = local_exponent(lags, values)
    results[name] = dict(label=label, color=color, marker=marker, lags=lags,
                         values=values, exponents=exponents,
                         crossover=crossover_lag(lags, exponents))


# ------------------------------------------------------------------ plotting
fig, (ax_a, ax_b) = plt.subplots(
    2, 1, figsize=(7.2, 5.6), sharex=True,
    gridspec_kw=dict(height_ratios=[1.35, 1.0], hspace=0.12))

# (a) the fluctuation functions themselves
for name, _, _, _ in ENVELOPES:
    r = results[name]
    ax_a.plot(r['lags'], r['values'], color=r['color'], lw=1.4, zorder=4,
              marker=r['marker'], ms=3.6, mec='white', mew=0.6)

# reference slope, anchored on the field-difference curve
ref = results['structure_function']
guide_l = np.array([ref['lags'][0], ref['lags'][-1]])
guide = ref['values'][0] * (guide_l / ref['lags'][0]) ** H_TRUE
ax_a.plot(guide_l, guide * 1.42, color=INK, lw=0.9, ls=(0, (4, 2.5)), zorder=3)
tip_l = float(np.exp(0.62 * np.log(guide_l[-1] / guide_l[0])) * guide_l[0])
tip_v = ref['values'][0] * 1.42 * (tip_l / ref['lags'][0]) ** H_TRUE
ax_a.annotate(f'$\\ell^{{{H_TRUE:g}}}$ reference slope', xy=(tip_l, tip_v),
              xytext=(0.70, 0.90), textcoords='axes fraction', color=LABEL,
              fontsize=7.5, ha='right', va='center',
              arrowprops=dict(arrowstyle='->', color=LABEL, lw=0.7,
                              shrinkB=2,
                              connectionstyle='arc3,rad=-0.05'))

ax_a.set_xscale('log')
ax_a.set_yscale('log')
ax_a.set_ylabel('$\\langle |\\mathfrak{T}_\\ell * f|^q \\rangle$')

# (b) the local power-law exponent
ax_b.axhspan(H_TRUE - EXPONENT_TOL, H_TRUE + EXPONENT_TOL, color=RULE,
             alpha=0.55, lw=0, zorder=0)
ax_b.axhline(H_TRUE, color=INK, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
ax_b.text(results['structure_function']['lags'][-1], H_TRUE + 0.012,
          f'$H = {H_TRUE:g}$', color=LABEL, fontsize=8.5, ha='right',
          va='bottom')

for name, _, _, _ in ENVELOPES:
    r = results[name]
    ax_b.plot(r['lags'], r['exponents'], color=r['color'], lw=1.4, zorder=4,
              marker=r['marker'], ms=3.6, mec='white', mew=0.6,
              label=r['label'])

# limits from the data, so the panel survives a change of H or FIELD_MODEL
# (under 'fbm' the curves overshoot H, under 'random_walk' they only undershoot)
seen = np.concatenate([results[n]['exponents'] for n, _, _, _ in ENVELOPES])
seen = seen[np.isfinite(seen)]
y_lo = min(seen.min(), H_TRUE - EXPONENT_TOL)
y_hi = max(seen.max(), H_TRUE + EXPONENT_TOL)
span = y_hi - y_lo
ax_b.set_ylim(y_lo - 0.95 * span, y_hi + 0.12 * span)

ticks = plt.MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]).tick_values(y_lo, y_hi)
ax_b.set_yticks([t for t in ticks
                 if y_lo - 0.06 * span <= t <= y_hi + 0.06 * span])

if ANNOTATE_CROSSOVER:
    # stems dropped into the empty band below the curves, staggered in
    # height so crossovers a factor of ~1.5 apart in l do not collide
    slot = 0
    for name, _, _, _ in ENVELOPES:
        r = results[name]
        lc = r['crossover']
        if not np.isfinite(lc) or lc <= r['lags'][0]:
            continue   # no transient to mark
        depth = y_lo - (0.24 + 0.25 * slot) * span
        ax_b.plot([lc, lc], [H_TRUE - EXPONENT_TOL, depth + 0.04 * span],
                  color=r['color'], lw=0.8, ls=(0, (1.5, 1.8)), zorder=2)
        ax_b.text(lc, depth, f'$\\ell={lc:.0f}$', color=r['color'],
                  fontsize=8, ha='center', va='top')
        slot += 1
ax_b.set_ylabel('$d\\log\\langle\\Delta f^q\\rangle\\,/\\,d\\log\\ell$')
ax_b.set_xlabel('scale $\\ell$ (grid cells)')

# upper-left inset: one of the walks the statistics are taken over
if SHOW_WALK_INSET:
    ins_w = ax_a.inset_axes([0.045, 0.60, 0.30, 0.34])
    trace = walk[WALK_INSET_INDEX, :WALK_INSET_CELLS]
    ins_w.plot(np.arange(len(trace)), trace, color=INK, lw=0.5,
               solid_joinstyle='round')
    ins_w.set_xlim(0, len(trace))
    ins_w.set_xticks([])
    ins_w.set_yticks([])
    for side in ins_w.spines:
        ins_w.spines[side].set_visible(False)
    walk_label = ('random walk $f(x)$' if FIELD_MODEL == 'random_walk'
                  else f'fBm $f(x)$, $H = {H_TRUE:g}$')
    ins_w.text(0.5, -0.04, walk_label, transform=ins_w.transAxes,
               color=LABEL, fontsize=7.5, ha='center', va='top')

# lower-right inset: the three envelopes at one nominal l, on a common cell axis
if SHOW_KERNEL_INSET:
    ins = ax_a.inset_axes([0.66, 0.115, 0.31, 0.34])
    # every kernel zero-padded out to the widest one (the Mexican hat), so
    # the compact envelopes run along zero across the full span rather than
    # stopping at their own support
    taps = {name: si.to_numpy(si.get_wavelet(name).build_kernel(INSET_LAG))
            for name, _, _, _ in ENVELOPES}
    span = int(np.ceil(max((len(k) - 1) / 2 for k in taps.values()))) + 1
    for name, _, color, _ in ENVELOPES:
        kernel = taps[name] / np.abs(taps[name]).max()
        pad = int(round(span - (len(kernel) - 1) / 2))
        kernel = np.concatenate([np.zeros(pad), kernel, np.zeros(pad)])
        offset = np.arange(len(kernel)) - (len(kernel) - 1) / 2
        # step, not line: these are the discrete taps as actually applied
        ins.step(offset, kernel, where='mid', color=color, lw=1.0)
    ins.axhline(0, color=RULE, lw=0.7, zorder=0)
    ins.set_xlim(-span, span)
    ins.set_ylim(-1.25, 1.25)
    ins.set_xticks([])
    ins.set_yticks([])
    for side in ins.spines:
        ins.spines[side].set_visible(False)
    ins.text(0.5, -0.04, f'$\\mathfrak{{T}}_\\ell$ at $\\ell={INSET_LAG}$',
             transform=ins.transAxes, color=LABEL, fontsize=7.5, ha='center',
             va='top')

# ------------------------------------------------------------------- styling
for ax, tag in ((ax_a, '(a)'), (ax_b, '(b)')):
    ax.text(0.0, 1.0, tag, transform=ax.transAxes, color=LABEL, fontsize=9,
            ha='left', va='bottom')
    ax.tick_params(colors=LABEL, labelsize=8, length=3, width=0.8,
                   which='both')
    ax.xaxis.label.set_color(LABEL)
    ax.yaxis.label.set_color(LABEL)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.8)

leg = ax_b.legend(frameon=False, fontsize=8.5, loc='lower right',
                  handlelength=1.6, borderaxespad=0.4)
for text in leg.get_texts():
    text.set_color(LABEL)

fig.savefig('wavelet_discretization_hires.png', dpi=300, bbox_inches='tight',
            pad_inches=0.05, transparent=True)
save(fig, 'wavelet_discretization')

print(f'done  ({N_REALIZATIONS} walks of {WALK_SIZE} cells, q = {Q_ORDER:g}, '
      f'lags 1-{max_sep})')
for name, _, _, _ in ENVELOPES:
    r = results[name]
    lc = r['crossover']
    reached = ('no transient' if lc <= r['lags'][0]
               else f'within {EXPONENT_TOL} of {H_TRUE:g} for l >= {lc:.0f}')
    print(f'  {r["label"]:<18s} exponent at l = {r["lags"][0]:.0f}: '
          f'{r["exponents"][0]:.3f}   {reached}')
