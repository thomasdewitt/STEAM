"""Circular-domain wireframe turbulons (cartesian-clipped and polar),
plus pie-cutout machinery. Shares the 2D envelope + oblique projection
with turblib.
"""
import numpy as np
from turblib import project, INK, PALETTE, AMPS, ORDER

R_DOMAIN = 1.10          # truncation radius (negative lobe ~ended)


def envelope(x, y, k=1.0):
    """1D-form Mexican hat as a function of radius (deeper negative lobe
    than the true 2D form — a deliberate visualization choice), doubled
    so peak = 2 matches the earlier renders: 2 (1 - a) exp(-a/2)."""
    a = np.pi**2 * (x**2 + y**2) / k**2
    return 2.0 * (1.0 - a) * np.exp(-a / 2.0)


def triad(ax, x0, y0, zlabel, zcolor, scale=0.30, lw=0.7,
          label_color=None, fontsize=7.5):
    """Small unit-vector glyph: x, y (oblique projected), z labeled with
    the field name."""
    if label_color is None:
        label_color = INK
    dirs = {'x': (1.0, 0.0), 'y': (0.32, 0.55), 'z': (0.0, 1.0)}
    # register the glyph's extent in the data limits (annotations don't)
    ax.plot([x0 - 0.05, x0 + scale * 1.15], [y0 - 0.05, y0 + scale * 1.15],
            alpha=0.0, lw=0.1)
    for name, (dx, dy) in dirs.items():
        n = np.hypot(dx, dy)
        ex, ey = scale * dx / n, scale * dy / n
        ax.annotate('', xy=(x0 + ex, y0 + ey), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=lw,
                                    mutation_scale=6, shrinkA=0,
                                    shrinkB=0), zorder=4,
                    annotation_clip=False)
        if name == 'x':
            ax.text(x0 + ex + 0.05, y0 + ey - 0.02, '$x$', ha='left',
                    va='center', color=label_color, fontsize=fontsize)
        elif name == 'y':
            ax.text(x0 + ex + 0.04, y0 + ey + 0.02, '$y$', ha='left',
                    va='bottom', color=label_color, fontsize=fontsize)
        else:
            ax.text(x0 + ex + 0.04, y0 + ey + 0.05, zlabel, ha='right',
                    va='bottom', color=zcolor, fontsize=fontsize)


def _draw_path(ax, x, y, z, color, lw, alpha=1.0, zorder=2,
               x0=0.0, y0=0.0, scale=1.0):
    Px, Py = project(x, y, z)
    ax.plot(scale * Px + x0, scale * Py + y0, color=color, lw=lw,
            alpha=alpha, zorder=zorder, solid_capstyle='round')


def wire_cart(ax, amp, color, R=R_DOMAIN, n_lines=13, lw=0.32,
              alpha=0.9, x0=0.0, y0=0.0, grid_offset=0.0, zorder=2):
    """Cartesian x/y grid lines clipped to circle radius R."""
    span = np.linspace(-R, R, n_lines) + grid_offset
    span = span[np.abs(span) < R]
    t = np.linspace(-R, R, 400)
    for c in span:
        # line y = c
        m = c**2 + t**2 <= R**2
        x, y = t.copy(), np.full_like(t, c)
        z = amp * envelope(x, y)
        x[~m] = np.nan
        _draw_path(ax, x, y, z, color, lw, alpha, zorder, x0, y0)
        # line x = c
        x2, y2 = np.full_like(t, c), t.copy()
        z2 = amp * envelope(x2, y2)
        x2c = x2.copy()
        x2c[~m] = np.nan
        _draw_path(ax, x2c, y2, z2, color, lw, alpha, zorder, x0, y0)
    rim(ax, amp, color, R=R, lw=lw * 1.5, alpha=alpha, x0=x0, y0=y0,
        zorder=zorder)


def rim(ax, amp, color, R=R_DOMAIN, lw=0.5, alpha=1.0, x0=0.0, y0=0.0,
        th0=0.0, th1=2 * np.pi, zorder=2):
    th = np.linspace(th0, th1, 400)
    x, y = R * np.cos(th), R * np.sin(th)
    z = amp * envelope(x, y)
    _draw_path(ax, x, y, z, color, lw, alpha, zorder, x0, y0)


def wire_polar(ax, amp, color, R=R_DOMAIN, n_rings=8, n_spokes=16,
               lw=0.32, alpha=0.9, x0=0.0, y0=0.0, spoke_offset=0.0,
               ring_offset=0.0, wedge=None, theta_ranges=None, zorder=2,
               rim_boost=1.4, scale=1.0):
    """Polar wireframe: concentric rings + radial spokes.
    wedge=(tha, thb): that ccw sector is removed.
    theta_ranges: explicit list of (t0, t1) sectors to draw (overrides
    wedge). Offsets rotate spokes / shift ring radii."""
    if theta_ranges is None:
        if wedge is None:
            theta_ranges = [(0.0, 2 * np.pi)]
        else:
            tha, thb = wedge
            theta_ranges = [(thb, tha + 2 * np.pi)]
    radii = (np.arange(1, n_rings + 1) / n_rings) * R + ring_offset
    radii = radii[(radii > 0.03) & (radii <= R + 1e-9)]
    for r in radii:
        w = lw * (rim_boost if abs(r - R) < 1e-9 else 1.0)
        for (t0, t1) in theta_ranges:
            th = np.linspace(t0, t1, max(40, int(400 * (t1 - t0)
                                                 / (2 * np.pi))))
            x, y = r * np.cos(th), r * np.sin(th)
            z = amp * envelope(x, y)
            _draw_path(ax, x, y, z, color, w, alpha, zorder, x0, y0, scale)
    ths = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False) + spoke_offset
    for thv in ths:
        tv = np.mod(thv, 2 * np.pi)
        ok = False
        for (t0, t1) in theta_ranges:
            if np.mod(tv - t0, 2 * np.pi) <= (t1 - t0) + 1e-9:
                ok = True
                break
        if not ok:
            continue
        rr = np.linspace(0.0, R, 200)
        x, y = rr * np.cos(thv), rr * np.sin(thv)
        z = amp * envelope(x, y)
        _draw_path(ax, x, y, z, color, lw, alpha, zorder, x0, y0, scale)


WALL_FILL = '#f5f4f1'


def wall(ax, th_w, x0=0.0, y0=0.0, lw=1.1, zorder=5, scale=1.0):
    r = np.linspace(0, R_DOMAIN, 300)
    env = envelope(r, 0.0)
    curves = {f: AMPS[f] * env for f in ORDER}
    allz = np.vstack([curves[f] for f in ORDER])
    top = np.maximum(allz.max(axis=0), 0)
    bot = np.minimum(allz.min(axis=0), 0)
    X, Yc = r * np.cos(th_w), r * np.sin(th_w)

    def proj(z):
        Px, Py = project(X, Yc, z)
        return x0 + scale * Px, y0 + scale * Py

    tx, ty = proj(top)
    bx, by = proj(bot)
    ax.fill(np.concatenate([tx, bx[::-1]]), np.concatenate([ty, by[::-1]]),
            color=WALL_FILL, ec='none', zorder=zorder)
    zx, zy = proj(np.zeros_like(r))
    ax.plot(zx, zy, color='#c9c9c9', lw=0.6, zorder=zorder + 1)
    ax.plot([tx[0], bx[0]], [ty[0], by[0]], color=INK, lw=0.5,
            alpha=0.6, zorder=zorder + 1)
    for f in sorted(ORDER, key=lambda f: abs(AMPS[f])):
        px, py = proj(curves[f])
        ax.plot(px, py, color=PALETTE[f], lw=lw, zorder=zorder + 2,
                solid_capstyle='round')
