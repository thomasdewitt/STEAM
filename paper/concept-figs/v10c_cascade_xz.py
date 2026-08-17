"""V10c — cascade x-z slice.
- density: exactly 1 turbulon of the largest class (n ~ k^-(1+Hz)),
- smallest class == spheroscale (round), teal + dimension marker,
- pruning: an ink turbulon is removed ONLY if its outline literally
  intersects the teal marks (ellipse / arrow / ticks / label box),
- seed: auto-searched so no large (k >= 2) turbulon intersects the
  marker; force one with --seed N.

    .venv/bin/python v10c_cascade_xz.py --seed 12
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from turblib import clean_axes, save, page_fontsize, INK

TEAL = '#1F6E6B'
HZ = 5.0 / 9.0
W, Hd = 20.0, 5.0
ls_ = 0.5
ks = [8.0, 4.0, 2.0, 1.0, 0.5]
alphas = [0.9, 0.8, 0.68, 0.55, 0.45]
lws = [0.7, 0.6, 0.5, 0.45, 0.4]
N0 = 1.0

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=3,
                help='seed (locked default: 3); auto-search with --seed -1')
args = ap.parse_args()

# ---- teal marker geometry ----
mk_kh = ks[-1]
mk_kv = ls_ * (mk_kh / ls_) ** HZ
mx, mz = 0.86 * W, 0.78 * Hd
adx = mk_kh / 2 + 0.42                      # arrow x-offset from center
TOL = 0.06                                  # 'literally intersect' tol


def teal_hit(ex, ez):
    """Do outline points (ex, ez) touch any teal mark?"""
    # teal ellipse boundary
    rn = np.sqrt(((ex - mx) / (mk_kh / 2)) ** 2
                 + ((ez - mz) / (mk_kv / 2)) ** 2)
    if np.any(np.abs(rn - 1.0) * min(mk_kh, mk_kv) / 2 < TOL):
        return True
    # vertical arrow segment + tick bars at x = mx + adx
    ax_ = mx + adx
    near_x = np.abs(ex - ax_) < TOL + 0.12   # ticks span +-0.12
    in_z = (ez > mz - mk_kv / 2 - TOL) & (ez < mz + mk_kv / 2 + TOL)
    if np.any(near_x & in_z):
        return True
    # label box
    if np.any((ex > ax_ + 0.18) & (ex < ax_ + 0.8)
              & (np.abs(ez - mz) < 0.2)):
        return True
    return False


def build(seed):
    rng = np.random.default_rng(seed)
    keep, big_clash = [], False
    th = np.linspace(0, 2 * np.pi, 240)
    for ci, kh in enumerate(ks):
        kv = ls_ * (kh / ls_) ** HZ
        n_t = max(1, int(round(N0 * (ks[0] / kh) ** (1.0 + HZ))))
        slots = (np.arange(n_t) + rng.uniform(0.15, 0.85, n_t)) / n_t * W
        rng.shuffle(slots)
        for ti in range(n_t):
            cx = slots[ti]
            cz = rng.uniform(kv / 2 + 0.1, Hd - kv / 2 - 0.1)
            ex = cx + kh / 2 * np.cos(th)
            ez = cz + kv / 2 * np.sin(th)
            if teal_hit(ex, ez):
                if kh >= 2.0:
                    big_clash = True
                continue
            keep.append((ci, cx, cz, kh, kv))
    return keep, big_clash


if args.seed >= 0:
    seed = args.seed
    keep, clash = build(seed)
    if clash:
        print(f'note: seed {seed} prunes a LARGE turbulon at the marker')
else:
    seed = 4
    for s in range(4, 80):
        keep, clash = build(s)
        if not clash:
            seed = s
            break
keep, _ = build(seed)
print('seed =', seed)

fig, ax = plt.subplots(figsize=(10, 4.4))
FS_NOTE = page_fontsize(9.0, fig)


def ellipse(cx, cz, kh, kv, **kw):
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx + kh / 2 * np.cos(th), cz + kv / 2 * np.sin(th), **kw)


for (ci, cx, cz, kh, kv) in keep:
    ellipse(cx, cz, kh, kv, color=INK, lw=lws[ci], alpha=alphas[ci])

ellipse(mx, mz, mk_kh, mk_kv, color=TEAL, lw=1.3, zorder=5)
ax.annotate('', xy=(mx + adx, mz + mk_kv / 2),
            xytext=(mx + adx, mz - mk_kv / 2),
            arrowprops=dict(arrowstyle='<->', color=TEAL, lw=0.7,
                            mutation_scale=7, shrinkA=0, shrinkB=0),
            zorder=5)
for zz in (mz - mk_kv / 2, mz + mk_kv / 2):
    ax.plot([mx + adx - 0.12, mx + adx + 0.12], [zz, zz], color=TEAL,
            lw=0.7, zorder=5)
ax.text(mx + adx + 0.24, mz, '$\\ell_s$', ha='left', va='center',
        color=TEAL, fontsize=FS_NOTE, zorder=5)

ax.set_xlim(-0.3, W + 0.3)
ax.set_ylim(-0.3, Hd + 0.3)
clean_axes(ax)
stem = 'v10c_cascade_xz' if seed == 3 else f'v10c_cascade_xz_s{seed}'
save(fig, stem)
print('done ->', stem)
