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
segment. So a chord polygon --- not the envelope --- is what the rest of
the cascade carries, and refining the grid does not walk back toward the
envelope.

The regrids are cell-consistent (steam.utils.zoom_trilinear, torch's
align_corners=False): each cell splits into two whose centers sit a
quarter of the old spacing either side of the old center. The only thing
this changes about the polygon is where a kink falls between the new
samples, so the first regrid cuts the corners off the polygon --- the
peak most visibly, from 2.968 to 2.262. The peak is then held exactly,
whatever the turbulon's position, because the first regrid leaves a flat
plateau across the center that later samples reproduce. The remaining
kinks keep being shaved a little at every hop, which is the 1-3% per
octave the supplement reports; the dashed curve is four further regrids
and shows how small that is next to the one-time cut.

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
from turblib import INK, RULE, LABEL, PALETTE, page_fontsize, save

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


def regrid(x, values):
    """One cell-consistent halving of the spacing, the 1D zoom_trilinear.

    Each cell splits into two whose centers sit a quarter of the old
    spacing either side of the old center (torch's align_corners=False),
    so there is no free phase and no dependence on where the turbulon
    sits within the domain.
    """
    spacing = (x[1] - x[0]) / 2
    x_new = np.sort(np.concatenate([x - spacing / 2, x + spacing / 2]))
    return x_new, np.interp(x_new, x, values)


# --- the working grid: cell centers at x_m = m * dx, dx = k/2 (s_x = 1) ---
dx = 0.5
m_max = int(np.floor(HALF_WIDTH / dx))
x_cells = np.arange(-m_max, m_max + 1) * dx
t_cells = envelope_1d(x_cells)

# --- one regrid, then the rest of the chain ---
x_fine = np.linspace(-HALF_WIDTH, HALF_WIDTH, 2001)
x1, t1 = regrid(x_cells, t_cells)

x_late, t_late = x1, t1
for _ in range(N_LATE):
    x_late, t_late = regrid(x_late, t_late)

fig, ax = plt.subplots(figsize=(7.2, 3.6))

FS_TICK = page_fontsize(8.0, fig)
FS_AXIS = page_fontsize(9.0, fig)
FS_NOTE = page_fontsize(8.5, fig)

ax.plot(x_fine, envelope_1d(x_fine), color=EXACT, lw=1.5, zorder=4,
        label='turbulon envelope $\\mathfrak{T}_\\ell(x)$')

# the chord polygon: draw it as the piecewise-linear function it is
ax.plot(x_cells, t_cells, color=COARSE, lw=1.5, zorder=5,
        label='chord polygon through the samples')
ax.fill_between(x_cells, envelope_1d(x_cells) * 0 + t_cells,
                envelope_1d(x_cells), color=COARSE, alpha=0.0, lw=0)
# shaded gap between envelope and chord polygon = the one-time loss
chord_on_fine = np.interp(x_fine, x_cells, t_cells)
ax.fill_between(x_fine, chord_on_fine, envelope_1d(x_fine), color=COARSE,
                alpha=0.13, lw=0, zorder=1)

ax.plot(x1, t1, color=LATE, lw=1.4, zorder=6,
        label='after one regrid')
ax.plot(x_late, t_late, color=LATE, lw=1.0, ls=(0, (4, 2.5)), zorder=6,
        label=f'after {N_LATE} more regrids')

ax.plot(x_cells, t_cells, 'o', ms=4.0, color=COARSE, mec='white', mew=0.7,
        zorder=7)

ax.axhline(0, color=RULE, lw=0.8, zorder=0)

# cell-width dimension marker
y_dim = -0.45
ax.annotate('', xy=(-dx / 2, y_dim), xytext=(dx / 2, y_dim),
            arrowprops=dict(arrowstyle='<->', color=LABEL, lw=0.8,
                            shrinkA=0, shrinkB=0))
ax.text(dx / 2 + 0.09, y_dim, '$\\Delta x_\\ell = \\ell/2$', color=INK,
        fontsize=FS_NOTE, ha='left', va='center')

# the one-time loss: the chord polygon shortcuts the envelope between
# samples, and the second regrid shortcuts the polygon's own kinks. From
# the third regrid on the field is a fixed point of the operator.
ax.annotate('the peak is cut once, to 2.262,\nand then held exactly',
            xy=(-0.10, t_late.max()), xytext=(-1.68, 2.62),
            color=INK, fontsize=FS_NOTE, ha='left', va='center',
            arrowprops=dict(arrowstyle='-', color=LABEL, lw=0.7,
                            connectionstyle='arc3,rad=0.18'))
ax.annotate('interpolation shortcuts the\nenvelope between samples',
            xy=(-0.30, np.interp(-0.30, x_fine, chord_on_fine) + 0.22),
            xytext=(-1.68, 1.15), color=INK, fontsize=FS_NOTE, ha='left',
            va='center',
            arrowprops=dict(arrowstyle='-', color=LABEL, lw=0.7,
                            connectionstyle='arc3,rad=-0.18'))

ax.set_xlim(-HALF_WIDTH, HALF_WIDTH)
ax.set_ylim(-0.62, 5.10)
ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
ax.set_xticklabels(['$-3\\ell/2$', '$-\\ell$', '$-\\ell/2$', '$0$',
                    '$\\ell/2$', '$\\ell$', '$3\\ell/2$'])
ax.set_yticks([0, 1, 2, 3])
ax.set_ylabel('envelope amplitude')
ax.set_xlabel('horizontal distance from turbulon center')

ax.tick_params(colors=INK, labelsize=FS_TICK, length=3, width=0.8)
ax.xaxis.label.set_color(INK)
ax.yaxis.label.set_color(INK)
ax.xaxis.label.set_fontsize(FS_AXIS)
ax.yaxis.label.set_fontsize(FS_AXIS)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_color(RULE)
    ax.spines[side].set_linewidth(0.8)

leg = ax.legend(frameon=False, fontsize=FS_NOTE, loc='upper right',
                handlelength=1.6, borderaxespad=0.2)
for text in leg.get_texts():
    text.set_color(INK)

fig.savefig('s2_interpolation_retention_hires.png', dpi=300,
            bbox_inches='tight', pad_inches=0.05, transparent=True)
save(fig, 's2_interpolation_retention')

drift = np.max(np.abs(t_late - np.interp(x_late, x1, t1)))
print(f'done  (A_opt = {A_OPT:.6f}, max drift after {N_LATE} further '
      f'regrids = {drift:.4f})')
