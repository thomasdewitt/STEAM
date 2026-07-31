"""S1 — how coarsely a turbulon is represented on its own working grid.

Supplement S1, "Working grids": each size class works on a grid with
Delta x_k = k / (2 s_x). At s_x = 1 the spacing is half the turbulon
size, so the envelope is carried by only five nonzero cells along the
center line: the center, the two flanks at +/- k/2, and the two negative
cells at +/- k.

The figure shows the continuum envelope against that piecewise-constant
grid representation, and against what the rest of the cascade actually
carries: the deposit after it has been interpolated down N_HOPS = 8
further working grids (k/dx = 2 -> 512, the canonical chain the
interpolation compensation is referenced to in steam.simulate).

The delivered curve is the CHORD POLYGON through the five samples, not a
smoothed envelope: the first regrid replaces the deposit by the straight
lines between its samples, and every later regrid interpolates a function
that is already piecewise linear. Refining the grid does not walk back
toward the envelope -- it converges to the polygon. (Computed here by
running the actual chain, so the statement is measured rather than
asserted; see s2_interpolation_retention for why the loss is one-time.)

One honest caveat, drawn as the shaded wedge at the tip: the grids are
corner-aligned (align_corners=True) and so do NOT node-nest, n -> 2n. A
turbulon whose center falls on a node of the finer grids keeps the full
peak; one falling half a cell off has its tip cut to 2.262, a 24% loss.
Both are real and position-dependent within a single run, which is part
of why the compensation factor is measured on the production cascade
rather than derived from this cut.

Envelope is the paper's 3D Mexican hat (Apxeq:turbulon shape) evaluated
along a line through the center:
    T(x) = (A - rho) exp(-rho / 2),  rho = (x / sigma)^2,  sigma = k / pi
with the leading constant A carrying the discrete admissibility
correction applied in steam.simulate._turbulon_envelope:
    A_opt = sum(rho w) / sum(w),  w = exp(-rho / 2)
summed over the 3D kernel grid. A_opt = 2.9678 at s = 1 and 3.0 (the
continuum value, which makes the 3D integral vanish) once s = 2 resolves
the negative shell. A_opt is computed here from the same closed form the
package uses; tests/test_kernel.py pins the two against each other.

NOTE: this is a center-line CUT of a 3D function. The cut is strongly
mean-positive even though the 3D kernel sums to zero -- the negative
shell lives off-axis, where the r^2 volume element gives it the
multiplicity to cancel the core. Do not read the panel as a mean.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from turblib import INK, RULE, LABEL, PALETTE, save

EXACT = INK                 # continuum envelope
COARSE = PALETTE['h']       # ochre — the grid representation
LATE = PALETTE['T']         # sienna — after the rest of the regrid chain
HALF_WIDTH = 1.75           # upper-panel x range, in units of k
SIGMA = 1.0 / np.pi         # k = 1 throughout
dx = 0.5                    # working-grid spacing k/(2 s_x) at s_x = 1
N_HOPS = 8                  # regrids to the k/dx = 512 reference chain
CHAIN_HALF = 8.0            # wide window for the chain, so edges don't intrude


def leading_constant(sparsity, support=5.0):
    """A_opt for the 3D kernel at this sparsity — see module docstring."""
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


def regrid_chain(phase_cells=0.0):
    """Run the deposit down N_HOPS working grids; return (x, values).

    Each hop halves the spacing, as k halves per dyadic class. The grids
    are corner-aligned and do not node-nest; `phase_cells` offsets the
    first finer grid by that fraction of its own spacing, which is how a
    turbulon's position within the domain shows up.
    """
    m = int(round(CHAIN_HALF / dx))
    x = np.arange(-m, m + 1) * dx           # centers sit ON a node, as in the code
    v = envelope_1d(x)
    for hop in range(1, N_HOPS + 1):
        spacing = dx / 2 ** hop
        offset = phase_cells * spacing if hop == 1 else 0.0
        x_new = np.arange(-CHAIN_HALF + offset, CHAIN_HALF, spacing)
        v = np.interp(x_new, x, v)
        x = x_new
    return x, v


fig, ax = plt.subplots(figsize=(7.2, 3.6))

x_fine = np.linspace(-HALF_WIDTH, HALF_WIDTH, 2001)
ax.plot(x_fine, envelope_1d(x_fine), color=EXACT, lw=1.5, zorder=4,
        label='turbulon envelope $\\mathfrak{T}_k(x)$')

# Grid representation at dx = k/2 (s_x = 1); cell centers as
# _turbulon_envelope builds them: x_m = m * dx, m integer.
m_max = int(np.floor(HALF_WIDTH / dx))
x_cells = np.arange(-m_max, m_max + 1) * dx
t_cells = envelope_1d(x_cells)

edges = np.concatenate([x_cells - dx / 2, [x_cells[-1] + dx / 2]])
ax.stairs(t_cells, edges, baseline=None, color=COARSE, lw=1.5, zorder=5,
          label='grid representation, $\\Delta x_k = k/2$')
ax.stairs(t_cells, edges, baseline=0, fill=True, color=COARSE, alpha=0.11,
          lw=0, zorder=1)
ax.plot(x_cells, t_cells, 'o', ms=4.0, color=COARSE, mec='white', mew=0.7,
        zorder=6)

# What the rest of the cascade carries: the deposit after N_HOPS further
# working grids. Solid = a turbulon centered on a node of the finer grids;
# the wedge to the dotted curve is the tip loss for one centered half a
# cell off, the corner-alignment phase spread within a single run.
x_chain, t_chain = regrid_chain(phase_cells=0.0)
x_chain_off, t_chain_off = regrid_chain(phase_cells=0.5)
window = np.abs(x_chain) <= HALF_WIDTH
window_off = np.abs(x_chain_off) <= HALF_WIDTH
ax.fill_between(x_chain[window], t_chain[window],
                np.interp(x_chain[window], x_chain_off, t_chain_off),
                color=LATE, alpha=0.16, lw=0, zorder=2)
ax.plot(x_chain_off[window_off], t_chain_off[window_off], color=LATE,
        lw=0.9, ls=(0, (2, 2)), alpha=0.8, zorder=3)
ax.plot(x_chain[window], t_chain[window], color=LATE, lw=1.4, zorder=3,
        label=f'carried downstream, after {N_HOPS} regrids')

ax.axhline(0, color=RULE, lw=0.8, zorder=0)

# cell-width dimension marker
y_dim = -0.52
ax.annotate('', xy=(-dx / 2, y_dim), xytext=(dx / 2, y_dim),
            arrowprops=dict(arrowstyle='<->', color=LABEL, lw=0.8,
                            shrinkA=0, shrinkB=0))
ax.text(0.0, y_dim - 0.11, '$\\Delta x_k = k/2$', color=LABEL, fontsize=8,
        ha='center', va='top')

# tap callouts
ax.text(0.0, t_cells[m_max] + 0.10, f'${A_OPT:.3f}$', color=COARSE,
        fontsize=8, ha='center', va='bottom')
ax.text(0.60, t_cells[m_max + 1] + 0.10, f'${t_cells[m_max + 1]:.3f}$',
        color=COARSE, fontsize=8, ha='left', va='bottom')
ax.text(1.10, t_cells[m_max + 2] - 0.08, f'${t_cells[m_max + 2]:.3f}$',
        color=COARSE, fontsize=8, ha='left', va='top')

ax.set_xlim(-HALF_WIDTH, HALF_WIDTH)
ax.set_ylim(-0.75, 3.35)
ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
ax.set_xticklabels(['$-3k/2$', '$-k$', '$-k/2$', '$0$', '$k/2$', '$k$',
                    '$3k/2$'])
ax.set_yticks([0, 1, 2, 3])
ax.set_ylabel('envelope amplitude')

for a in (ax,):
    a.tick_params(colors=LABEL, labelsize=8, length=3, width=0.8)
    a.xaxis.label.set_color(LABEL)
    a.yaxis.label.set_color(LABEL)
    a.xaxis.label.set_fontsize(9)
    a.yaxis.label.set_fontsize(9)
    for side in ('top', 'right'):
        a.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        a.spines[side].set_color(RULE)
        a.spines[side].set_linewidth(0.8)

ax.set_xlabel('horizontal distance from turbulon center')
leg = ax.legend(frameon=False, fontsize=8.5, loc='upper right',
                handlelength=1.6, borderaxespad=0.2)
for text in leg.get_texts():
    text.set_color(LABEL)

fig.savefig('s1_working_grid_resolution_hires.png', dpi=300,
            bbox_inches='tight', pad_inches=0.05, transparent=True)
save(fig, 's1_working_grid_resolution')
print(f'done  (A_opt = {A_OPT:.8f})')
