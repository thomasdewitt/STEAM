"""S1 — how coarsely a turbulon is represented on its own working grid.

Supplement S1, "Working grids": each size class works on a grid with
Delta x_k = k / (2 s_x). At s_x = 1 the spacing is half the turbulon
size, so the envelope is carried by only five nonzero cells: the center
(T = 3), the two flanks at +/- k/2 (T = 0.155), and the two negative
cells at +/- k (T = -0.049). This figure shows the continuum envelope
against that piecewise-constant grid representation.

Envelope is the paper's 3D Mexican hat (Apxeq:turbulon shape) evaluated
along a line through the center:
    T(x) = (3 - rho) exp(-rho / 2),  rho = (x / sigma)^2,  sigma = k / pi
which is exactly what steam.turbulons.mexican_hat returns for
r_norm^2 = x^2, and the sample points are those of
steam.simulate._turbulon_envelope (cell-centered, spacing dx).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from turblib import INK, RULE, LABEL, PALETTE, save

EXACT = INK                 # continuum envelope
COARSE = PALETTE['h']       # ochre — the grid representation
HALF_WIDTH = 1.75           # x range, in units of k


def mexican_hat_1d(x, k=1.0):
    """Center-line cut of the 3D (3 - rho) Mexican hat, sigma = k / pi."""
    ratio_sq = (x / (k / np.pi)) ** 2
    return (3.0 - ratio_sq) * np.exp(-ratio_sq / 2.0)


fig, ax = plt.subplots(figsize=(7.2, 3.6))

# --- continuum envelope -------------------------------------------------
x_fine = np.linspace(-HALF_WIDTH, HALF_WIDTH, 2001)
ax.plot(x_fine, mexican_hat_1d(x_fine), color=EXACT, lw=1.5, zorder=4,
        label='turbulon envelope $\\mathfrak{T}_k(x)$')

# --- grid representation at dx = k/2 (s_x = 1) --------------------------
# Cell centers as _turbulon_envelope builds them: x_m = m * dx, m integer.
dx = 0.5
m_max = int(np.floor(HALF_WIDTH / dx))
x_cells = np.arange(-m_max, m_max + 1) * dx
t_cells = mexican_hat_1d(x_cells)

# Piecewise-constant: each cell holds its sampled value across its width.
edges = np.concatenate([x_cells - dx / 2, [x_cells[-1] + dx / 2]])
ax.stairs(t_cells, edges, baseline=None, color=COARSE, lw=1.5, zorder=5,
          label='grid representation, $\\Delta x_k = k/2$')
ax.stairs(t_cells, edges, baseline=0, fill=True, color=COARSE, alpha=0.11,
          lw=0, zorder=1)
ax.plot(x_cells, t_cells, 'o', ms=4.0, color=COARSE, mec='white', mew=0.7,
        zorder=6)

# --- reference lines ----------------------------------------------------
ax.axhline(0, color=RULE, lw=0.8, zorder=0)
# Stems only on the cells that actually carry the envelope; the rest are
# numerically zero and stemming them is clutter.
for xc, tc in zip(x_cells, t_cells):
    if abs(xc) <= 1.0 + 1e-9:
        ax.plot([xc, xc], [-0.42, tc], color=COARSE, lw=0.5,
                ls=(0, (1, 2.5)), alpha=0.55, zorder=2)

# --- cell-width dimension marker ---------------------------------------
y_dim = -0.52
ax.annotate('', xy=(-dx / 2, y_dim), xytext=(dx / 2, y_dim),
            arrowprops=dict(arrowstyle='<->', color=LABEL, lw=0.8,
                            shrinkA=0, shrinkB=0))
ax.text(0.0, y_dim - 0.11, '$\\Delta x_k = k/2$', color=LABEL, fontsize=8,
        ha='center', va='top')

# --- value callouts -----------------------------------------------------
ax.text(-0.20, 2.80, '$3$', color=COARSE, fontsize=8, ha='center',
        va='center')
ax.text(0.60, 0.155 + 0.10, '$0.155$', color=COARSE, fontsize=8,
        ha='left', va='bottom')
ax.text(1.10, -0.049 - 0.08, '$-0.049$', color=COARSE, fontsize=8,
        ha='left', va='top')

# --- axes ---------------------------------------------------------------
ax.set_xlim(-HALF_WIDTH, HALF_WIDTH)
ax.set_ylim(-0.75, 3.35)
ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
ax.set_xticklabels(['$-3k/2$', '$-k$', '$-k/2$', '$0$', '$k/2$', '$k$',
                    '$3k/2$'])
ax.set_yticks([0, 1, 2, 3])
ax.set_xlabel('horizontal distance from turbulon center')
ax.set_ylabel('envelope amplitude')
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

fig.savefig('s1_working_grid_resolution_hires.png', dpi=300,
            bbox_inches='tight', pad_inches=0.05, transparent=True)
save(fig, 's1_working_grid_resolution')
print('done')
