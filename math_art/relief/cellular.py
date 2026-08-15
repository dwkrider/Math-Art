"""relief.cellular -- Worley's cellular texture, and the crack pattern of it.

Scatter feature points, then let every position report its distance to them.
The nearest gives F1, a field of smooth domes rising from a network of creases;
the difference F2 - F1 is zero exactly on the set of points equidistant from
their two nearest features, which is the Voronoi boundary -- so it draws the
cell walls without ever constructing a Voronoi diagram.  That is the whole
trick of the method: a partition emerges from distances alone, with no
combinatorial geometry and no data structure to maintain.

For a relief this gives the family nothing else here covers: stone setts, dried
mud, reptile skin, cracked glaze -- structure that is irregular yet everywhere
locally organised.

**It periodizes for free**, which is unusual and worth stating.  The seam work
elsewhere in this engine snaps a wavevector to the dual lattice or projects
onto a symmetric basis.  Here it is enough to compare against the *minimum
image* of each feature point -- the nearest of its copies under the panel's
translation lattice -- because distance is all the construction uses.  Wrap the
points and the field is periodic exactly, not approximately.

References:
  Steven Worley, "A Cellular Texture Basis Function", SIGGRAPH 1996, 291-294 --
    F1, F2, the Fn combinations, and the scattering scheme.
  Georgy Voronoi, "Nouvelles applications des parametres continus a la theorie
    des formes quadratiques", J. reine angew. Math. 133, 1908, 97-178 -- the
    partition F2 - F1 draws the boundaries of.
"""

import math

import numpy as np

# What the distances are turned into.  F1 is the plain nearest-feature
# distance; CRACK is Worley's F2 - F1, which vanishes on the cell walls.
MODES = ('F1', 'F2', 'CRACK', 'F2F1_SUM', 'CELL_ID')


def feature_points(count, hx, hy, seed=1, jitter=1.0):
    """`count` scattered points in the panel rectangle.

    Worley scatters by a Poisson process per cube; on a single panel a
    stratified scatter is equivalent and reproducible.  `jitter` 0 collapses
    the scatter onto the lattice itself, which is a legitimate look (a regular
    sett pattern) and a useful test case, since the answer is then known.
    """
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    n = max(1, int(count))
    # Stratify over a near-square grid so the points cover evenly, then jitter
    # within each stratum -- a plain uniform draw clumps, and clumping reads as
    # a mistake rather than as randomness.
    cols = max(1, int(round(math.sqrt(n * hx / max(hy, 1e-9)))))
    rows = max(1, int(math.ceil(n / cols)))
    ix, iy = np.meshgrid(np.arange(cols), np.arange(rows))
    ix = ix.ravel()[:n].astype(float)
    iy = iy.ravel()[:n].astype(float)
    du = 2.0 * hx / cols
    dv = 2.0 * hy / rows
    j = float(np.clip(jitter, 0.0, 1.0))
    px = -hx + (ix + 0.5 + j * (rng.random(n) - 0.5)) * du
    py = -hy + (iy + 0.5 + j * (rng.random(n) - 0.5)) * dv
    return np.stack([px, py], axis=-1)


def distances(X, Y, pts, periodic=False, hx=None, hy=None, keep=2):
    """The `keep` smallest feature distances at every sample.

    With `periodic`, each offset is reduced to its minimum image before the
    distance is taken, so a point near the left edge is also near the right
    one -- which is what makes the field tile.
    """
    P = np.asarray(pts, dtype=float).reshape(-1, 2)
    keep = max(1, min(int(keep), len(P)))
    Lx = 2.0 * float(hx if hx is not None else np.abs(X).max())
    Ly = 2.0 * float(hy if hy is not None else np.abs(Y).max())

    # Chunk over feature points: the full (samples x points) matrix is the
    # obvious formulation and needs gigabytes at panel resolutions.  Keeping a
    # running set of the smallest `keep` costs one pass and bounded memory.
    shape = X.shape
    best = np.full((keep,) + shape, np.inf)
    step = max(1, int(2_000_000 // max(X.size, 1)) or 1)
    for start in range(0, len(P), step):
        blk = P[start:start + step]
        for px, py in blk:
            dx = X - px
            dy = Y - py
            if periodic:
                dx = dx - Lx * np.round(dx / Lx)
                dy = dy - Ly * np.round(dy / Ly)
            d = np.hypot(dx, dy)
            # Insert into the running smallest-k, cheapest first.
            for k in range(keep):
                swap = d < best[k]
                if not swap.any():
                    continue
                d, best[k] = np.where(swap, best[k], d), np.where(swap, d,
                                                                  best[k])
    return best


def worley(X, Y, info, count=48, seed=1, mode='CRACK', jitter=1.0,
           periodic=False, sharp=1.0):
    """Cellular field from scattered feature points.

    `sharp` shapes the crack profile: 1 leaves F2 - F1 linear (a V-groove),
    higher values narrow the wall into a crease, lower ones broaden it into a
    valley.
    """
    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    pts = feature_points(count, hx, hy, seed=seed, jitter=jitter)
    keep = 1 if mode == 'F1' else 2
    d = distances(X, Y, pts, periodic=periodic, hx=hx, hy=hy, keep=keep)

    # Normalise by the mean feature spacing, so the look does not change with
    # the point count -- only the scale does.
    scale = math.sqrt(4.0 * hx * hy / max(1, int(count)))
    if mode == 'F1':
        out = d[0] / scale
    elif mode == 'F2':
        out = d[1] / scale
    elif mode == 'F2F1_SUM':
        out = (d[1] + d[0]) / scale
    elif mode == 'CELL_ID':
        # Flat-topped cells: each plateau is one Voronoi region.  Built from
        # the distances alone, so it inherits the periodicity for free.
        out = np.where(d[1] - d[0] > 0.02 * scale, 1.0, -1.0)
        return out
    else:                                   # CRACK
        out = (d[1] - d[0]) / scale
    out = np.clip(out, 0.0, None)
    s = float(sharp)
    if s != 1.0 and s > 0.0:
        out = out ** s
    m = out.mean()
    return 2.0 * (out - m) / (out.std() or 1.0) * 0.5


def _selftest():
    ok = True
    from . import grid as _grid
    from . import tiling as _tiling

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=160)

    for mode in MODES:
        h = worley(X, Y, info, count=36, seed=3, mode=mode)
        print("cellular: %-9s range [%.2f, %.2f] finite=%s sd=%.4f"
              % (mode, h.min(), h.max(), np.isfinite(h).all(), h.std()))
        ok = ok and np.isfinite(h).all() and h.std() > 1e-6
        ok = ok and h.shape == X.shape

    # F1 is a distance to a point set, so it is 1-Lipschitz: no sample may
    # exceed its neighbour by more than the sample pitch.  This is the
    # property that makes the cell interiors smooth domes rather than noise,
    # and it holds exactly, not approximately.
    d = distances(X, Y, feature_points(36, 1.0, 1.0, seed=3), keep=1)[0]
    slope = float(max(np.abs(np.diff(d, axis=1)).max(),
                      np.abs(np.diff(d, axis=0)).max()) / info['dx'])
    print("cellular: F1 Lipschitz constant %.6f (must not exceed 1)" % slope)
    ok = ok and slope <= 1.0 + 1e-9

    # F2 >= F1 everywhere, by definition of "second nearest".
    dd = distances(X, Y, feature_points(24, 1.0, 1.0, seed=5), keep=2)
    print("cellular: min(F2 - F1) = %.3e (must be >= 0)"
          % float((dd[1] - dd[0]).min()))
    ok = ok and float((dd[1] - dd[0]).min()) >= -1e-12

    # The crack really is the Voronoi boundary: F2 - F1 vanishes there and
    # nowhere else.  Check against a directly computed nearest-point label --
    # the walls must sit exactly where the label changes.
    P = feature_points(16, 1.0, 1.0, seed=7)
    lab = np.argmin(np.stack([np.hypot(X - px, Y - py) for px, py in P]),
                    axis=0)
    boundary = np.zeros(X.shape, dtype=bool)
    boundary[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    boundary[:-1, :] |= lab[:-1, :] != lab[1:, :]
    gap = distances(X, Y, P, keep=2)
    crack = gap[1] - gap[0]
    print("cellular: F2-F1 on the Voronoi boundary %.4f vs %.4f in the "
          "interior" % (crack[boundary].mean(), crack[~boundary].mean()))
    ok = ok and crack[boundary].mean() < 0.25 * crack[~boundary].mean()

    # Zero jitter puts the 36 points on a 6x6 lattice, and with wrapping the
    # field is then EXACTLY periodic under a one-cell translation -- so any two
    # cells must agree to machine precision, not merely look alike.  The grid
    # has to divide evenly for that to be testable: 169 samples spanning the
    # width gives 168 pitches, exactly 28 per cell.  (At 160 samples a cell is
    # 26.67 pitches and the comparison is between different phases of the
    # pattern, which is a broken test rather than a broken field.)
    Xr, Yr, ir = _grid.make_grid(width=2.0, aspect=1.0, resolution=169)
    reg = worley(Xr, Yr, ir, count=36, seed=1, mode='CRACK', jitter=0.0,
                 periodic=True)
    per = (ir['nx'] - 1) // 6
    shift = float(np.abs(reg[:, :per] - reg[:, per:2 * per]).max())
    print("cellular: unjittered lattice, cell-to-cell difference %.2e" % shift)
    ok = ok and shift < 1e-9

    # Periodic wrapping makes it tile -- by minimum image, not by snapping.
    per = worley(X, Y, info, count=36, seed=3, mode='CRACK', periodic=True)
    chk = _tiling.check(per, 'TORUS', info, 'WORLEY')
    non = worley(X, Y, info, count=36, seed=3, mode='CRACK', periodic=False)
    chk_non = _tiling.check(non, 'TORUS', info, 'WORLEY')
    print("cellular: periodic joint step x%.3f curvature x%.3f ok=%s "
          "| unwrapped x%.3f ok=%s"
          % (chk['step'], chk['curvature'], chk['ok'],
             chk_non['step'], chk_non['ok']))
    ok = ok and chk['ok'] and not chk_non['ok']

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
