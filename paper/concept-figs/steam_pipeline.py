#!/usr/bin/env python3
"""How a STEAM simulation works, in three steps.

Paper version of the defense-deck figure (steam_pipeline_A3 in
~/Main/Presentations/PhD Defense/code-and-figures/steam-pipeline-fig/), with
the step labels that were placed in Keynote for the talk now drawn here.

Data are real slices of the small-domain STEAM run steam_full_c002_s0030.nc
on BLUE (c = 0.02, l_s = 30 m, L = 20.48 km, seed 7001), extracted by
steam_pipeline_extract.py into steam_pipeline_slices.npz (not committed).
Nothing is drawn by hand. Per-class increments are carried to the finest
grid by a straight 2-D zoom rather than the cascade's hop-by-hop chain, so
per-class amplitudes are off by a few tens of percent -- fine for a concept
figure, not for quantitative claims.

Layout: input profiles + 3 constants (left), then per-variable columns
(h, qt, flux) of cascade increments stacked as a written-out sum -- panel +
panel, vdots, sum rule, full field -- and the diagnosed condensate centered
below. Only the left half of the domain (10.24 km) is shown so panels are
not so wide.

Sized for a landscape page: included at the rotated text width (9 in).
"""

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from turblib import INK, LABEL, PALETTE, save

HERE = Path(__file__).resolve().parent
D = np.load(HERE / "steam_pipeline_slices.npz")

FIGW, FIGH = 12.8, 7.2
PAGE_WIDTH_IN = 9.0      # \textwidth inside a sidewaysfigure


def pt(size):
    """Font size to set so that text renders at `size` pt on the page.

    The saved PDF is cropped to its ink (about 0.93 of FIGW), so scale
    against that rather than the full figure width.
    """
    return size * 0.93 * FIGW / PAGE_WIDTH_IN


FS_BODY = pt(9)          # headers, class labels, constants
FS_SMALL = pt(7.5)       # ticks, axis labels, asides
FS_STEP = pt(10)         # step labels

# Diverging ramps for zero-mean increments, one per variable: the positive
# lobe carries the variable's own hue (matching its profile curve and full
# field), the negative lobe a complementary palette colour.
DIV = {
    "h": LinearSegmentedColormap.from_list("div_h", [
        "#333d54", "#5B6B8C", "#fbf9f5", "#C08A2D", "#7a5417"]),
    "qt": LinearSegmentedColormap.from_list("div_qt", [
        "#7c3117", "#B5502A", "#fbf9f5", "#1F6E6B", "#14504E"]),
    "flux": LinearSegmentedColormap.from_list("div_f", [
        "#2e2e2e", "#6e6e6e", "#fbf9f5", "#5B6B8C", "#333d54"]),
}

# Sequential ramps for the full fields (paper -> variable colour -> ink).
STOPS = [0.00, 0.28, 0.52, 0.76, 1.00]
SEQ = {
    "h": LinearSegmentedColormap.from_list("seq_h", list(zip(
        STOPS, ["#fbf9f5", "#e2dccf", "#c9a25c", "#9c4a28", INK]))),
    "qt": LinearSegmentedColormap.from_list("seq_qt", list(zip(
        STOPS, ["#fbf9f5", "#dbe4e0", "#7fa8a4", "#1F6E6B", INK]))),
    "flux": LinearSegmentedColormap.from_list("seq_f", list(zip(
        STOPS, ["#fbf9f5", "#dfe2e9", "#9aa5bd", "#5B6B8C", INK]))),
}

# Condensate: clear sky in paper tone, cloud toward slate/ink.
CLOUD_CMAP = LinearSegmentedColormap.from_list("cloud", [
    "#fbf9f5", "#c8cedb", "#5B6B8C", INK])

X_KM = D["x"] / 1000.0
NXH = D["h"].shape[0] // 2          # leftmost half of the domain
ASPECT = (NXH * X_KM[1]) / 4.0      # 10.24 km / 4 km = 2.56

CLASSES = [("c01", r"$\ell = 10.24$ km"), ("c04", r"$\ell = 1.28$ km")]
FIELDS = ["h", "qt", "flux"]
COLS = {
    "h": (r"$h$", PALETTE["h"]),
    "qt": (r"$q_t$", PALETTE["qt"]),
    "flux": (r"$\mathcal{F}$", PALETTE["p"]),
}


def half(a):
    return a[:NXH, :]


def field_ax(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(LABEL)
        s.set_linewidth(0.5)
    return ax


def imshow(ax, a, **kw):
    ax.imshow(a.T, origin="lower", aspect="auto", interpolation="bilinear",
              rasterized=True, **kw)


def profile_panels(fig, rect):
    """Two thin profile panels sharing z, drawn inside rect (fig coords)."""
    x0, y0, w, h = rect
    zp = D["z_profile"] / 1000.0
    sel = zp <= 4.0
    axh = fig.add_axes([x0, y0, w * 0.44, h])
    axq = fig.add_axes([x0 + w * 0.56, y0, w * 0.44, h])
    axh.plot(D["h_profile"][sel] / 1000, zp[sel], color=PALETTE["h"],
             lw=1.6)
    axq.plot(D["qt_profile"][sel] * 1000, zp[sel], color=PALETTE["qt"],
             lw=1.6)
    for ax in (axh, axq):
        ax.set_ylim(0, 4)
        ax.tick_params(labelsize=FS_SMALL, colors=LABEL, width=0.5)
        for s in ax.spines.values():
            s.set_edgecolor(LABEL)
            s.set_linewidth(0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axh.set_ylabel("$z$ (km)", fontsize=FS_BODY, color=INK)
    axq.set_yticklabels([])
    axh.set_xlabel(r"$\langle h\rangle$ (kJ kg$^{-1}$)", fontsize=FS_SMALL,
                   color=PALETTE["h"])
    axq.set_xlabel(r"$\langle q_t\rangle$ (g kg$^{-1}$)", fontsize=FS_SMALL,
                   color=PALETTE["qt"])
    axh.set_xticks([320, 335])
    axq.set_xticks([5, 15])


def main():
    fig = plt.figure(figsize=(FIGW, FIGH))

    ncol = len(FIELDS)
    RH = 0.135                                  # row height, fig coords
    CW = RH * (FIGH / FIGW) * ASPECT            # column width, fig coords
    GAP = 0.014
    block_w = ncol * CW + (ncol - 1) * GAP
    RX = 0.63 - block_w / 2

    def col_x(j):
        return RX + j * (CW + GAP)

    # -- step 1: inputs, left column
    profile_panels(fig, [0.040, 0.40, 0.145, 0.30])
    fig.text(0.010, 0.78,
             r"\textbf{1. Inputs:} horizontal" "\n"
             r"mean profiles for $h$, $q_t$",
             fontsize=FS_STEP, color=INK, ha="left", va="bottom",
             linespacing=1.5)
    fig.text(0.040, 0.315,
             r"$L = 20.48$ km\ \ (outer scale)" "\n"
             r"$\ell_s = 30$ m\ \ (spheroscale)" "\n"
             r"$\varpi = 0.02$\ \ (intermittency)",
             fontsize=FS_BODY, color=INK, va="top", linespacing=1.9)

    # -- step 2: the cascade, one column per variable
    y_top = 0.875
    fig.text(RX + block_w / 2, 0.93,
             r"\textbf{2. Cascade:} add perturbations at every size class",
             fontsize=FS_STEP, color=INK, ha="center", va="bottom")
    for j, f in enumerate(FIELDS):
        name, color = COLS[f]
        fig.text(col_x(j) + CW / 2, y_top + 0.017, name,
                 fontsize=FS_BODY * 1.15, color=color, ha="center",
                 va="center")

    # shared per-column increment scale across the shown classes
    lims = {f: max(np.percentile(np.abs(half(D[f"inc_{f}_{n}"])), 99.0)
                   for n, _ in CLASSES) for f in FIELDS}

    y = y_top
    for i, (cname, clab) in enumerate(CLASSES):
        for j, f in enumerate(FIELDS):
            ax = field_ax(fig, [col_x(j), y - RH, CW, RH])
            imshow(ax, half(D[f"inc_{f}_{cname}"]), cmap=DIV[f],
                   vmin=-lims[f], vmax=lims[f])
        fig.text(RX - 0.014, y - RH / 2, clab, fontsize=FS_BODY, color=INK,
                 ha="right", va="center")
        y -= RH
        if i == 0:                     # "+" centered between rows
            for j in range(ncol):
                fig.text(col_x(j) + CW / 2, y - 0.021, "$+$",
                         fontsize=FS_BODY * 1.3, color=INK, ha="center",
                         va="center")
            y -= 0.042
    for j in range(ncol):
        fig.text(col_x(j) + CW / 2, y - 0.023, r"$\vdots$",
                 fontsize=FS_BODY * 1.3, color=INK, ha="center",
                 va="center")
    fig.text(RX - 0.014, y - 0.023, "11 classes,\n20.48 km to 20 m", fontsize=FS_SMALL, color=LABEL, ha="right",
             va="center")
    y -= 0.050
    # sum rule, then the full fields
    for j in range(ncol):
        fig.add_artist(matplotlib.lines.Line2D(
            [col_x(j), col_x(j) + CW], [y, y], transform=fig.transFigure,
            color=INK, lw=1.0))
    y -= 0.008
    for j, f in enumerate(FIELDS):
        ax = field_ax(fig, [col_x(j), y - RH, CW, RH])
        if f == "flux":
            fl = half(D["flux"])
            imshow(ax, fl, cmap=SEQ["flux"],
                   vmin=np.percentile(fl, 1.0),
                   vmax=np.percentile(fl, 99.0))
        else:
            imshow(ax, half(D[f]), cmap=SEQ[f])
    fig.text(RX - 0.014, y - RH / 2,
             r"$\Phi = \langle\Phi\rangle(z)$" "\n"
             r"$\quad\ +\ \sum_\ell \Phi_\ell$",
             fontsize=FS_BODY, color=INK, ha="right", va="center",
             linespacing=1.6)
    y_full_bottom = y - RH

    # -- step 3: diagnose, centered under h and qt
    xc12 = (col_x(0) + col_x(1) + CW) / 2
    ytop3 = y_full_bottom - 0.085
    ax = field_ax(fig, [xc12 - CW / 2, ytop3 - RH, CW, RH])
    a = (half(D["qc"]) + half(D["qi"])) * 1000.0
    imshow(ax, a, cmap=CLOUD_CMAP, vmin=0,
           vmax=np.percentile(a[a > 0], 99.5) if (a > 0).any() else 1)
    fig.text(xc12 - CW / 2 - 0.014, ytop3 - RH / 2,
             r"\textbf{3. Diagnose} using" "\n" "saturation adjustment",
             fontsize=FS_STEP, color=INK, ha="right", va="center",
             linespacing=1.5)
    fig.text(xc12 + CW / 2 + 0.012, ytop3 - RH / 2,
             r"$q_c + q_i$" "\n" r"(also $T$, $p$, $q_v$)",
             fontsize=FS_BODY, color=INK, ha="left", va="center",
             linespacing=1.6)

    # imshow panels are rasters; savefig's default dpi sets their resolution
    matplotlib.rcParams["savefig.dpi"] = 300
    save(fig, "steam_pipeline")
    print("wrote steam_pipeline.pdf")


if __name__ == "__main__":
    main()
