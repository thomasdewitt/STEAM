"""S1 — how coarsely a turbulon is represented on its own working grid.

Supplement S1, "Working grids": each size class works on a grid with
Delta x_k = k / (2 s_x). At s_x = 1 the spacing is half the turbulon
size, so the envelope is carried by only five nonzero cells along the
center line: the center, the two flanks at +/- k/2, and the two negative
cells at +/- k.

Upper panel: the continuum envelope against that piecewise-constant grid
representation. Lower panel: the same envelope on log axes out to the
truncation radius, showing where support_factor may be cut.

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
MARK = PALETTE['qt']        # teal — support-factor annotations
HALF_WIDTH = 1.75           # upper-panel x range, in units of k
SIGMA = 1.0 / np.pi         # k = 1 throughout
FLOAT32_EPS = 1.2e-7


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


fig, (ax, axl) = plt.subplots(
    2, 1, figsize=(7.2, 5.4), gridspec_kw=dict(height_ratios=[3, 2], hspace=0.55))

# ======================= upper panel: the grid representation ============
x_fine = np.linspace(-HALF_WIDTH, HALF_WIDTH, 2001)
ax.plot(x_fine, envelope_1d(x_fine), color=EXACT, lw=1.5, zorder=4,
        label='turbulon envelope $\\mathfrak{T}_k(x)$')

# Grid representation at dx = k/2 (s_x = 1); cell centers as
# _turbulon_envelope builds them: x_m = m * dx, m integer.
dx = 0.5
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

ax.axhline(0, color=RULE, lw=0.8, zorder=0)
for xc, tc in zip(x_cells, t_cells):
    if abs(xc) <= 1.0 + 1e-9:
        ax.plot([xc, xc], [-0.42, tc], color=COARSE, lw=0.5,
                ls=(0, (1, 2.5)), alpha=0.55, zorder=2)

# cell-width dimension marker
y_dim = -0.52
ax.annotate('', xy=(-dx / 2, y_dim), xytext=(dx / 2, y_dim),
            arrowprops=dict(arrowstyle='<->', color=LABEL, lw=0.8,
                            shrinkA=0, shrinkB=0))
ax.text(0.0, y_dim - 0.11, '$\\Delta x_k = k/2$', color=LABEL, fontsize=8,
        ha='center', va='top')

# tap callouts
ax.text(-0.20, 2.76, f'${A_OPT:.3f}$', color=COARSE, fontsize=8,
        ha='center', va='center')
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

# ======================= lower panel: support factor =====================
x_log = np.linspace(0.0, 5.4, 4001)
axl.semilogy(x_log, np.abs(envelope_1d(x_log)), color=EXACT, lw=1.3,
             zorder=4)

peak = A_OPT
axl.axhline(peak, color=RULE, lw=0.8, zorder=0)
# (the notch near 0.55k is the envelope's zero crossing, where |T| -> 0)
axl.text(2.4, peak * 1.8, 'peak $\\mathfrak{T}_k(0)$', color=LABEL,
         fontsize=7.5, ha='left', va='bottom')

# float32 resolution relative to the peak — below this the kernel is
# numerically indistinguishable from zero.
axl.axhline(peak * FLOAT32_EPS, color=LABEL, lw=0.8, ls=(0, (4, 3)),
            zorder=1)
axl.text(4.15, peak * FLOAT32_EPS * 2.2, 'float32 resolution',
         color=LABEL, fontsize=7.5, ha='left', va='bottom')

for support, style, note in [(3.0, (0, (2, 2)), 'proposed'),
                             (5.0, 'solid', 'current')]:
    axl.axvline(support, color=MARK, lw=1.1, ls=style, zorder=3)
    axl.text(support - 0.08, 1e-38,
             f'support $= {support:.0f}k$\n({note})', color=MARK,
             fontsize=7.5, ha='right', va='bottom', linespacing=1.4)

axl.set_xlim(0, 5.4)
axl.set_ylim(1e-45, 30)
axl.set_yticks([1e-40, 1e-30, 1e-20, 1e-10, 1e0])
axl.set_xticks([0, 1, 2, 3, 4, 5])
axl.set_xticklabels(['$0$', '$k$', '$2k$', '$3k$', '$4k$', '$5k$'])
axl.set_ylabel('$|\\mathfrak{T}_k|$')
axl.set_xlabel('distance from turbulon center')

# ======================= shared styling ==================================
for a in (ax, axl):
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
