"""Shared machinery for turbulon concept figures.

Envelope: 2D analog of the paper's 3D Mexican hat (Eq. A1),
    T_k(r) = (2 - pi^2 r^2 / k^2) exp(-pi^2 r^2 / (2 k^2))
zero-mean over 2D space (admissibility), negative skirt included.

Rendering: manual oblique/axonometric projection of a height surface,
drawn as thin ridgelines (y-slices) with optional painter's-algorithm
hidden-line removal. No mpl3d.

Text is typeset by LaTeX itself (text.usetex), so figure annotation is in
the same Computer Modern as the manuscript rather than matplotlib's
DejaVu -- copernicus.cls loads no font package, so the document is plain
CM. This needs a working latex + dvipng on PATH, the same requirement as
building the paper at all.

Annotation text is INK (black), matching the body text it sits next to.
Grey (LABEL) is kept for structure that is not reading matter: rules,
leader lines, arrows, spines, ticks.

Figures are drawn several inches wider than they appear on the page
(\\includegraphics[width=12cm] throughout main.tex), so a font size set
here lands smaller in print. Use page_fontsize() to set sizes in the
points they will actually have on the page.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
})

INK = '#111111'
RULE = '#e3e3e3'
LABEL = '#7a7a7a'

PAGE_WIDTH_IN = 12 / 2.54       # \includegraphics[width=12cm] in main.tex


def page_fontsize(pt, fig):
    """Font size to set so that text renders at `pt` on the printed page."""
    return pt * fig.get_size_inches()[0] / PAGE_WIDTH_IN

# palette: field -> color  (earth-adjacent, tuned for white background)
PALETTE = {
    'qt': '#1F6E6B',   # deep teal        — total water
    'h':  '#C08A2D',   # ochre            — moist static energy
    'T':  '#B5502A',   # sienna           — temperature
    'p':  '#5B6B8C',   # slate blue       — pressure
}
# physically coherent amplitude set for one warm-moist turbulon
AMPS = {'qt': 1.0, 'h': 0.7, 'T': 0.35, 'p': -0.22}
LABELS = {'qt': 'total water $q_t$', 'h': 'moist static energy $h$',
          'T': 'temperature $T$', 'p': 'pressure $p$'}
ORDER = ['T', 'p', 'h', 'qt']   # corner order: TL, TR, BL, BR


def envelope(x, y, k=1.0, aspect=1.0):
    """2D Mexican-hat turbulon envelope; aspect>1 flattens in y."""
    r2 = x**2 + (y * aspect)**2
    a = np.pi**2 * r2 / k**2
    return (2.0 - a) * np.exp(-a / 2.0)


def surface(n=121, half=1.6, k=1.0, aspect=1.0):
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x, y)
    return X, Y, envelope(X, Y, k=k, aspect=aspect)


def project(X, Y, Z, zscale=0.5, ydepth=0.55, yshear=0.32):
    """Oblique projection: screen coords from (x, y, z=height)."""
    Px = X + yshear * Y
    Py = ydepth * Y + zscale * Z
    return Px, Py


def draw_ridges(ax, X, Y, Z, color=INK, lw=0.5, zscale=0.5,
                ydepth=0.55, yshear=0.32, step=4, hidden=True,
                alpha=1.0, x0=0.0, y0=0.0):
    """Draw y=const slices back-to-front with hidden-line removal.

    Hidden-line: keep a running upper envelope on a fine common
    screen-x grid; draw only segments above it (front slices occlude
    those behind, since larger screen-y at same screen-x means visible
    from above... we draw back (large Y) first and occlude by height).
    """
    Px, Py = project(X, Y, Z, zscale, ydepth, yshear)
    Px = Px + x0
    Py = Py + y0
    n = X.shape[0]
    rows = range(0, n, step)   # front (min y) to back; back hidden by front
    if not hidden:
        for i in rows:
            ax.plot(Px[i], Py[i], color=color, lw=lw, alpha=alpha,
                    solid_capstyle='round', zorder=2)
        return
    # common fine screen-x grid for the occlusion envelope
    gx = np.linspace(Px.min(), Px.max(), 2200)
    env_hi = np.full_like(gx, -np.inf)   # highest screen-y drawn so far
    for i in rows:
        sy = np.interp(gx, Px[i], Py[i], left=np.nan, right=np.nan)
        vis = np.isnan(sy) | (sy > env_hi)
        # draw visible runs
        run = None
        for j, v in enumerate(vis):
            good = v and not np.isnan(sy[j])
            if good and run is None:
                run = j
            elif not good and run is not None:
                if j - run > 1:
                    ax.plot(gx[run:j], sy[run:j], color=color, lw=lw,
                            alpha=alpha, solid_capstyle='round', zorder=2)
                run = None
        if run is not None and len(gx) - run > 1:
            ax.plot(gx[run:], sy[run:], color=color, lw=lw, alpha=alpha,
                    solid_capstyle='round', zorder=2)
        env_hi = np.fmax(env_hi, np.where(np.isnan(sy), -np.inf, sy))


def draw_wireframe(ax, X, Y, Z, color=INK, lw=0.35, zscale=0.5,
                   ydepth=0.55, yshear=0.32, step=6, alpha=1.0,
                   x0=0.0, y0=0.0):
    """Transparent full wireframe (both slice directions), no occlusion."""
    Px, Py = project(X, Y, Z, zscale, ydepth, yshear)
    Px, Py = Px + x0, Py + y0
    n = X.shape[0]
    for i in range(0, n, step):
        ax.plot(Px[i], Py[i], color=color, lw=lw, alpha=alpha, zorder=2)
    for j in range(0, n, step):
        ax.plot(Px[:, j], Py[:, j], color=color, lw=lw, alpha=alpha, zorder=2)


def clean_axes(ax):
    ax.set_aspect('equal')
    ax.axis('off')


def save(fig, stem, thumb_px=1400, dpi=200):
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', pad_inches=0.05,
                transparent=True)
    # thumbnail for in-session viewing (< 2000 px)
    w = fig.get_size_inches()[0]
    # fig.savefig(f'{stem}.png', dpi=min(dpi, thumb_px / w),
    #             bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)
