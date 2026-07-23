"""V11c — h field (ink ridgeline) with two highlighted turbulons (one
strong positive, one strong negative), each extracted at right as a
circular polar wireframe with its relative amplitude.

Seed exploration:
    .venv/bin/python v11c_field_h.py --seed 23
    .venv/bin/python v11c_field_h.py --seed 23 --wire   # also wireframe field

Writes v11c_field_h_s<seed>.png/.pdf and prints the chosen turbulons.
Both chosen turbulons are constrained to |cx|,|cy| < 1.5 so their full
annulus stays inside the domain (half = 2.6).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from turblib import (envelope as env2d, draw_ridges, draw_wireframe,
                     project, clean_axes, save, PALETTE, LABEL, INK)
from wirelib import wire_polar, triad

OCHRE = PALETTE['h']
ARROW_COLORS = {'sienna': PALETTE['T'], 'slate': PALETTE['p'],
                'teal': PALETTE['qt']}
CENTER_BOX = 1.5          # turbulon centers stay this far from the edge

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=1001)
ap.add_argument('--arrow', default='sienna',
                choices=['sienna', 'slate', 'teal'])
ap.add_argument('--wire', action='store_true',
                help='also render the wireframe-field variant')
args = ap.parse_args()

rng = np.random.default_rng(args.seed)

# ---- build field ----
n = 221
half = 2.6
x = np.linspace(-half, half, n)
X, Y = np.meshgrid(x, x)
H = 0.55
field = np.zeros_like(X)
ks = [2.0, 1.0, 0.5, 0.25]
placed = []
for ci, k in enumerate(ks):
    n_t = int(10 * (ks[0] / k) ** 1.6)
    for ti in range(n_t):
        cx, cy = rng.uniform(-half, half, 2)
        amp = rng.normal(0, 1) * k ** H
        field += amp * env2d(X - cx, Y - cy, k=k)
        placed.append((ci, cx, cy, amp, k))

# ---- pick strong positive + strong negative, both well inside domain
cands = [p for p in placed if p[0] == 1 and abs(p[1]) < CENTER_BOX
         and abs(p[2]) < CENTER_BOX]
pos = [p for p in cands if p[3] > 0]
if not pos:
    raise SystemExit(f'seed {args.seed}: no positive class-1 turbulon '
                     'inside the center box — try another seed')
strong = max(pos, key=lambda p: p[3])
negs = [p for p in cands if p[3] < 0
        and np.hypot(p[1] - strong[1], p[2] - strong[2]) > 1.4]
if not negs:
    raise SystemExit(f'seed {args.seed}: no separated negative turbulon '
                     '— try another seed')
weak = min(negs, key=lambda p: p[3])
print(f'seed {args.seed}: strong A={strong[3]:+.2f} at '
      f'({strong[1]:+.2f},{strong[2]:+.2f}); negative A={weak[3]:+.2f} '
      f'at ({weak[1]:+.2f},{weak[2]:+.2f}); ratio '
      f'{weak[3] / strong[3]:+.2f}')


def render(stem, field_style):
    fig, ax = plt.subplots(figsize=(11, 6))
    x_off = -3.1
    if field_style == 'ridge':
        draw_ridges(ax, X, Y, 0.30 * field, color=INK, lw=0.4, step=3,
                    zscale=0.55, ydepth=0.5, yshear=0.28, x0=x_off)
    else:
        draw_wireframe(ax, X, Y, 0.30 * field, color=INK, lw=0.28,
                       step=5, alpha=0.8, zscale=0.55, ydepth=0.5,
                       yshear=0.28, x0=x_off)
    # highlight both turbulons
    tips = []
    for (ci, hx, hy, hamp, hk) in (strong, weak):
        hl = hamp * env2d(X - hx, Y - hy, k=hk)
        mask = np.abs(hl) > 0.25 * abs(hamp)
        crown = (hl * np.sign(hamp)) > 0.25 * abs(hamp)
        Zh = np.where(mask, 0.30 * field, np.nan)
        Zc = np.where(crown, 0.30 * field, np.nan)
        pts = []
        for i in range(0, n, 3):
            if np.any(mask[i]):
                Px, Py = project(X[i], Y[i], Zh[i], zscale=0.55,
                                 ydepth=0.5, yshear=0.28)
                ax.plot(Px + x_off, Py, color=OCHRE, lw=0.85, zorder=3)
            if np.any(crown[i]):
                Pxc, Pyc = project(X[i], Y[i], Zc[i], zscale=0.55,
                                   ydepth=0.5, yshear=0.28)
                ok = ~np.isnan(Pyc)
                pts.append(np.column_stack([Pxc[ok] + x_off, Pyc[ok]]))
        pts = np.concatenate(pts)
        if hamp >= 0:      # strong: anchor rightmost crown point
            tips.append(tuple(pts[np.argmax(pts[:, 0])]))
        else:              # negative: anchor bottommost crown point
            tips.append(tuple(pts[np.argmin(pts[:, 1])]))
    triad(ax, x_off - 4.55, -1.5, '$h$', OCHRE, scale=0.36, lw=0.9)

    # ---- right: the two extracted turbulons
    amps_rel = {'strong': 1.0, 'weak': weak[3] / strong[3]}
    y_positions = {'strong': 1.05, 'weak': -1.35}
    x_off2 = 1.55
    sc = 0.78
    for name, xoff in (('strong',0), ('weak',-.5)):
        y0 = y_positions[name]
        wire_polar(ax, amps_rel[name], OCHRE, x0=x_off2+xoff, y0=y0,
                   lw=0.32, alpha=0.9, n_rings=7, n_spokes=14, scale=sc)
        ax.text(x_off2 + .65+xoff, y0 - 0.55,
                f'$A = {amps_rel[name]:.1f}$', ha='left', va='center',
                color=OCHRE, fontsize=8.5)
    # connectors (teal); the negative one arcs BELOW the field
    routes = {'strong': (0.25, 0.0, -0.12, .9), 'weak': (0.1, -0.25, 0.33,1.5)}
    for (tx, ty), name in zip(tips, ('strong', 'weak')):
        ddx, ddy, rad,arrx = routes[name]
        ax.annotate('', xy=(x_off2 - arrx, y_positions[name] + 0.1),
                    xytext=(tx + ddx, ty + ddy),
                    arrowprops=dict(arrowstyle='->',
                                    color=ARROW_COLORS[args.arrow], lw=0.9,
                                    connectionstyle=f'arc3,rad={rad}'),
                    zorder=4)
    clean_axes(ax) 
    save(fig, stem)


stem = ('v11c_field_h' if args.seed == 1001 and args.arrow == 'sienna'
        else f'v11c_field_h_s{args.seed}_{args.arrow}')
render(stem, 'ridge')
if args.wire:
    render(f'v11d_field_h_wire_s{args.seed}', 'wire')
print('done')
