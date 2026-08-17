"""The turbulon figure (Fig. "turbulon concept" in main.tex): an
assembled quincunx with the asymmetric cutout center (right wall face-on
at 0 deg, left wall near view-parallel)."""
import numpy as np
import matplotlib.pyplot as plt
from turblib import PALETTE, AMPS, ORDER, INK, clean_axes, page_fontsize, save
from wirelib import wire_polar, triad, wall

THA_DEG = 275
THA = np.radians(THA_DEG)
TRIAD_LABELS = {'qt': '$q_t$', 'h': '$h$', 'T': '$T$', 'p': "$p'$"}
AMP_TEXT = {'qt': '$A = 1.0$', 'h': '$A = 0.7$', 'T': '$A = 0.35$',
            'p': '$A = -0.22$'}

fig, ax = plt.subplots(figsize=(10, 8))
FS_NOTE = page_fontsize(8.5, fig)

d = 2.8
corners = {'T': (-d, d * 0.75), 'p': (d, d * 0.75),
           'h': (-d, -d * 0.75), 'qt': (d, -d * 0.75)}
for f, (cx, cy) in corners.items():
    wire_polar(ax, AMPS[f], PALETTE[f], x0=cx, y0=cy)
    triad(ax, cx - 1.62, cy - 0.62, TRIAD_LABELS[f], PALETTE[f],
          scale=0.3, lw=0.8, fontsize=FS_NOTE)
    # amplitude label mirrors the triad position on the right side, and
    # sits clear of the wireframe (radius 1.1) rather than over it
    ax.text(cx + 1.22, cy - 0.62, AMP_TEXT[f], ha='left', va='center',
            color=PALETTE[f], fontsize=FS_NOTE)

SC = 1.3
far = [(0.0, np.pi)]
near = [(np.pi, THA)]
for f in sorted(ORDER, key=lambda f: abs(AMPS[f])):
    wire_polar(ax, AMPS[f], PALETTE[f], lw=0.32, alpha=0.85,
               theta_ranges=far, zorder=2, n_rings=7, n_spokes=14,
               scale=SC)
wall(ax, THA, lw=1.1, zorder=5, scale=SC)
wall(ax, 2 * np.pi, lw=1.1, zorder=5, scale=SC)
for f in sorted(ORDER, key=lambda f: abs(AMPS[f])):
    wire_polar(ax, AMPS[f], PALETTE[f], lw=0.32, alpha=0.6,
               theta_ranges=near, zorder=8, n_rings=7, n_spokes=14,
               scale=SC)
# center triad: equal clearance to its (1.3x) turbulon as corners have
# to theirs — edge at -1.155*SC, corner clearance ~0.465
triad(ax, -1.155 * SC - 0.465, -0.62 * SC, 'all fields', INK,
      scale=0.3, lw=0.8, fontsize=FS_NOTE)

clean_axes(ax)
# high-res transparent PNG first (save() closes the figure)
fig.savefig('turbulon_figure_hires.png', dpi=300,
            bbox_inches='tight', pad_inches=0.05, transparent=True)
save(fig, 'turbulon_figure')
print('done')
