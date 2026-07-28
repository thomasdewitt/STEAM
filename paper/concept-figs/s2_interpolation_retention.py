"""S2 — why the regrid loss is committed once, at the first interpolation.

Supplement S1, "Working grids": a class deposits its turbulons on its own
working grid (Delta x_k = k/2, Nyquist at the class scale) and the field
is then trilinearly interpolated down the remaining chain of grids. The
figure shows why the resulting amplitude loss is a ONE-TIME factor rather
than something that accumulates:

  - the continuum envelope (what the deposit means);
  - its samples at Delta x_k = k/2 (what the working grid holds);
  - the piecewise-linear interpolant through those samples.

The interpolant is the chord polygon through the k/2 samples. Every later
regrid linearly interpolates a function that is ALREADY piecewise linear,
which reproduces it exactly wherever the new nodes fall inside an old
segment. So the chord polygon --- not the envelope --- is what the rest of
the cascade carries, and refining the grid does not walk back toward the
envelope. The dashed curve is the interpolant after four further dyadic
regrids; it lies on the one-regrid polygon to within the line width, the
residual drift coming only from the corner-aligned (align_corners=True)
resampling, which does not exactly nest n -> 2n.

Envelope and leading constant are identical to s1_working_grid_resolution:
the paper's 3D Mexican hat (Apxeq:turbulon shape) cut along a line through
the center,
    T(x) = (A - rho) exp(-rho / 2),  rho = (x / sigma)^2,  sigma = k / pi
with A = A_opt the discrete admissibility correction that
steam.simulate._turbulon_envelope applies (A_opt = 2.9678 at s = 1).

NOTE, as in S1: this is a center-line CUT of a 3D function and is strongly
mean-positive even though the 3D kernel sums to zero. Do not read the panel
as a mean.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from turblib import INK, RULE, LABEL, PALETTE, save

EXACT = INK                 # continuum envelope
COARSE = PALETTE['h']       # ochre — the k/2 samples and their interpolant
LATE = PALETTE['T']         # sienna — after the rest of the regrid chain
HALF_WIDTH = 1.75           # x range, in units of k
SIGMA = 1.0 / np.pi         # k = 1 throughout
N_LATE = 4                  # further dyadic regrids for the dashed curve


def leading_constant(sparsity, support=5.0):
    """A_opt for the 3D kernel at this sparsity — see s1 module docstring."""
    spacing = 1.0 / (2.0 * sparsity)
    half = int(np.ceil(support / spacing))
    axis = np.arange(-half, half + 1) * spacing
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
    rho = (X**2 + Y**2 + Z**2) / SIGMA**2
    weight = np.exp(-rho / 2.0)
    return float((rho * weight).sum() / weight.sum())


A_OPT = leading_constant(1)


def envelope_1d(x, leading=A_OPT):
    """Center-line cut of the 3D Mexican hat with leading constant A."""
    rho = (x / SIGMA) ** 2
    return (leading - rho) * np.exp(-rho / 2.0)


def regrid(values, n_target):
    """One corner-aligned linear regrid, the 1D form of zoom_trilinear.

    torch's align_corners=True maps sample j of n_target onto position
    j*(n_src-1)/(n_target-1) in source index units: endpoints coincide,
    interior nodes do NOT nest under n -> 2n.
    """
    n_src = len(values)
    src = np.arange(n_src, dtype=float)
    dst = np.linspace(0.0, n_src - 1.0, n_target)
    return np.interp(dst, src, values)


# --- the working grid: cell centers at x_m = m * dx, dx = k/2 (s_x = 1) ---
dx = 0.5
m_max = int(np.floor(HALF_WIDTH / dx))
x_cells = np.arange(-m_max, m_max + 1) * dx
t_cells = envelope_1d(x_cells)

# --- one regrid, then the rest of the chain ---
x_fine = np.linspace(-HALF_WIDTH, HALF_WIDTH, 2001)
n1 = 2 * len(x_cells) - 1
x1 = np.linspace(x_cells[0], x_cells[-1], n1)
t1 = regrid(t_cells, n1)

t_late, n_late = t1, n1
for _ in range(N_LATE):
    n_late = 2 * n_late
    t_late = regrid(t_late, n_late)
x_late = np.linspace(x_cells[0], x_cells[-1], n_late)

fig, ax = plt.subplots(figsize=(7.2, 3.6))

ax.plot(x_fine, envelope_1d(x_fine), color=EXACT, lw=1.5, zorder=4,
        label='turbulon envelope $\\mathfrak{T}_k(x)$')

# the chord polygon: draw it as the piecewise-linear function it is
ax.plot(x_cells, t_cells, color=COARSE, lw=1.5, zorder=5,
        label='after one regrid (piecewise linear)')
ax.fill_between(x_cells, envelope_1d(x_cells) * 0 + t_cells,
                envelope_1d(x_cells), color=COARSE, alpha=0.0, lw=0)
# shaded gap between envelope and chord polygon = the one-time loss
chord_on_fine = np.interp(x_fine, x_cells, t_cells)
ax.fill_between(x_fine, chord_on_fine, envelope_1d(x_fine), color=COARSE,
                alpha=0.13, lw=0, zorder=1)

ax.plot(x_late, t_late, color=LATE, lw=1.0, ls=(0, (4, 2.5)), zorder=6,
        label='after all further regrids (fixed point)')

ax.plot(x_cells, t_cells, 'o', ms=4.0, color=COARSE, mec='white', mew=0.7,
        zorder=7, label='working-grid samples, $\\Delta x_k = k/2$')

ax.axhline(0, color=RULE, lw=0.8, zorder=0)

# cell-width dimension marker
y_dim = -0.52
ax.annotate('', xy=(-dx / 2, y_dim), xytext=(dx / 2, y_dim),
            arrowprops=dict(arrowstyle='<->', color=LABEL, lw=0.8,
                            shrinkA=0, shrinkB=0))
ax.text(0.0, y_dim - 0.11, '$\\Delta x_k = k/2$', color=LABEL, fontsize=8,
        ha='center', va='top')

# the one-time loss: the chord polygon shortcuts the envelope between
# samples, and the second regrid shortcuts the polygon's own kinks. From
# the third regrid on the field is a fixed point of the operator.
ax.annotate('the kink is rounded once,\nthen never again',
            xy=(-0.06, t_late.max() - 0.04), xytext=(-1.68, 2.62),
            color=LABEL, fontsize=8, ha='left', va='center',
            arrowprops=dict(arrowstyle='-', color=LABEL, lw=0.7,
                            connectionstyle='arc3,rad=0.18'))
ax.annotate('interpolation shortcuts the\nenvelope between samples',
            xy=(-0.30, np.interp(-0.30, x_fine, chord_on_fine) + 0.22),
            xytext=(-1.68, 1.15), color=LABEL, fontsize=8, ha='left',
            va='center',
            arrowprops=dict(arrowstyle='-', color=LABEL, lw=0.7,
                            connectionstyle='arc3,rad=-0.18'))

ax.set_xlim(-HALF_WIDTH, HALF_WIDTH)
ax.set_ylim(-0.75, 3.35)
ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
ax.set_xticklabels(['$-3k/2$', '$-k$', '$-k/2$', '$0$', '$k/2$', '$k$',
                    '$3k/2$'])
ax.set_yticks([0, 1, 2, 3])
ax.set_ylabel('envelope amplitude')
ax.set_xlabel('horizontal distance from turbulon center')

ax.tick_params(colors=LABEL, labelsize=8, length=3, width=0.8)
ax.xaxis.label.set_color(LABEL)
ax.yaxis.label.set_color(LABEL)
ax.xaxis.label.set_fontsize(9)
ax.yaxis.label.set_fontsize(9)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_color(RULE)
    ax.spines[side].set_linewidth(0.8)

leg = ax.legend(frameon=False, fontsize=8.5, loc='upper right',
                handlelength=1.6, borderaxespad=0.2)
for text in leg.get_texts():
    text.set_color(LABEL)

fig.savefig('s2_interpolation_retention_hires.png', dpi=300,
            bbox_inches='tight', pad_inches=0.05, transparent=True)
save(fig, 's2_interpolation_retention')

drift = np.max(np.abs(t_late - np.interp(x_late, x_cells, t_cells)))
print(f'done  (A_opt = {A_OPT:.6f}, max drift after {N_LATE} further '
      f'regrids = {drift:.4f})')
