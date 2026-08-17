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

The delivered curve is a CHORD POLYGON, not a smoothed envelope: the
first regrid replaces the deposit by the straight lines between its
samples, and every later regrid interpolates a function that is already
piecewise linear. Refining the grid does not walk back toward the
envelope -- it is a fixed point after the first hop.

The tip loss is the same for every turbulon in the domain. The regrids
are cell-consistent (steam.utils.zoom_trilinear, torch's
align_corners=False): source and target samples are the cell CENTERS of
grids covering the same extent, so halving the spacing puts the new
samples exactly a quarter of the old spacing either side of every old
one. The turbulon center always falls midway between two samples of the
finer grid and its peak is always cut to (3 T(0) + T(dx)) / 4 = 2.262,
76.2% of 2.968 -- no dependence on where the turbulon sits. Hence a
single delivered curve here, and a per-class scalar compensation that can
absorb the loss exactly. (The earlier corner-aligned convention,
align_corners=True, delivered a position-dependent ramp across the domain
instead; that is why it was replaced.)

The chain is computed here by running the interpolation, not by asserting
the algebra; the 2.262 peak matches steam.utils.zoom_trilinear driven
over the same eight hops at any turbulon position. See
s2_interpolation_retention for why the loss is one-time.

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
from turblib import INK, RULE, LABEL, PALETTE, page_fontsize, save

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


def regrid_chain():
    """Run the deposit down N_HOPS working grids; return (x, values).

    Each hop halves the spacing, as k halves per dyadic class. The grids
    are cell-consistent, so each cell splits into two whose centers sit a
    quarter of the old spacing either side of the old center -- there is
    no free phase, and so no dependence on the turbulon's position.
    """
    m = int(round(CHAIN_HALF / dx))
    x = np.arange(-m, m + 1) * dx     # deposit sits on cell centers, as in the code
    v = envelope_1d(x)
    for _ in range(N_HOPS):
        spacing = (x[1] - x[0]) / 2
        x_new = np.sort(np.concatenate([x - spacing / 2, x + spacing / 2]))
        v = np.interp(x_new, x, v)
        x = x_new
    return x, v


fig, ax = plt.subplots(figsize=(7.2, 3.6))

FS_TICK = page_fontsize(8.0, fig)
FS_AXIS = page_fontsize(9.0, fig)
FS_NOTE = page_fontsize(8.5, fig)

x_fine = np.linspace(-HALF_WIDTH, HALF_WIDTH, 2001)
ax.plot(x_fine, envelope_1d(x_fine), color=EXACT, lw=1.5, zorder=4,
        label='turbulon envelope $\\mathfrak{T}_\\ell(x)$')

# Grid representation at dx = k/2 (s_x = 1); cell centers as
# _turbulon_envelope builds them: x_m = m * dx, m integer.
m_max = int(np.floor(HALF_WIDTH / dx))
x_cells = np.arange(-m_max, m_max + 1) * dx
t_cells = envelope_1d(x_cells)

edges = np.concatenate([x_cells - dx / 2, [x_cells[-1] + dx / 2]])
ax.stairs(t_cells, edges, baseline=None, color=COARSE, lw=1.5, zorder=5,
          label='grid representation')
ax.stairs(t_cells, edges, baseline=0, fill=True, color=COARSE, alpha=0.11,
          lw=0, zorder=1)
ax.plot(x_cells, t_cells, 'o', ms=4.0, color=COARSE, mec='white', mew=0.7,
        zorder=6)

# What the rest of the cascade carries: the deposit after N_HOPS further
# working grids. One curve, because the cell-consistent regrids deliver
# the same shape wherever the turbulon sits (see module docstring).
x_chain, t_chain = regrid_chain()
window = np.abs(x_chain) <= HALF_WIDTH
ax.plot(x_chain[window], t_chain[window], color=LATE, lw=1.4, zorder=3,
        label=f'after {N_HOPS} regrids')

ax.axhline(0, color=RULE, lw=0.8, zorder=0)

# cell-width dimension marker. The label goes ABOVE the arrow: the span
# is only one cell wide, so a centered label below it sits on top of its
# own arrowheads, and the band above is empty.
y_dim = -0.45
ax.annotate('', xy=(-dx / 2, y_dim), xytext=(dx / 2, y_dim),
            arrowprops=dict(arrowstyle='<->', color=LABEL, lw=0.8,
                            shrinkA=0, shrinkB=0))
for tick_x in (-dx / 2, dx / 2):
    ax.plot([tick_x, tick_x], [y_dim - 0.07, y_dim + 0.07], color=LABEL,
            lw=0.8, zorder=1)
ax.text(0.0, y_dim + 0.10, '$\\Delta x_\\ell = \\ell/2$', color=INK,
        fontsize=FS_NOTE, ha='center', va='bottom')

# tap callouts
ax.text(0.0, t_cells[m_max] + 0.10, f'${A_OPT:.3f}$', color=COARSE,
        fontsize=FS_NOTE, ha='center', va='bottom')
ax.text(0.60, t_cells[m_max + 1] + 0.10, f'${t_cells[m_max + 1]:.3f}$',
        color=COARSE, fontsize=FS_NOTE, ha='left', va='bottom')
ax.text(1.10, t_cells[m_max + 2] - 0.08, f'${t_cells[m_max + 2]:.3f}$',
        color=COARSE, fontsize=FS_NOTE, ha='left', va='top')

# delivered peak: what the rest of the cascade actually receives
ax.annotate(f'${t_chain.max():.3f}$', xy=(0.10, t_chain.max()),
            xytext=(0.62, t_chain.max() + 0.16), color=LATE,
            fontsize=FS_NOTE, ha='left', va='center',
            arrowprops=dict(arrowstyle='-', color=LATE, lw=0.7,
                            shrinkA=2, shrinkB=2))

ax.set_xlim(-HALF_WIDTH, HALF_WIDTH)
ax.set_ylim(-0.62, 3.35)
ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
ax.set_xticklabels(['$-3\\ell/2$', '$-\\ell$', '$-\\ell/2$', '$0$',
                    '$\\ell/2$', '$\\ell$', '$3\\ell/2$'])
ax.set_yticks([0, 1, 2, 3])
ax.set_ylabel('envelope amplitude')

for a in (ax,):
    a.tick_params(colors=INK, labelsize=FS_TICK, length=3, width=0.8)
    a.xaxis.label.set_color(INK)
    a.yaxis.label.set_color(INK)
    a.xaxis.label.set_fontsize(FS_AXIS)
    a.yaxis.label.set_fontsize(FS_AXIS)
    for side in ('top', 'right'):
        a.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        a.spines[side].set_color(RULE)
        a.spines[side].set_linewidth(0.8)

ax.set_xlabel('horizontal distance from turbulon center')
leg = ax.legend(frameon=False, fontsize=FS_NOTE, loc='upper left',
                handlelength=1.6, borderaxespad=0.2)
for text in leg.get_texts():
    text.set_color(INK)

fig.savefig('s1_working_grid_resolution_hires.png', dpi=300,
            bbox_inches='tight', pad_inches=0.05, transparent=True)
save(fig, 's1_working_grid_resolution')
print(f'done  (A_opt = {A_OPT:.8f})')
