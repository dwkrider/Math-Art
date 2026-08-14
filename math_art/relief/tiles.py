"""relief.tiles -- flowing patterns from tiles laid down at random.

Two ornament traditions that produce organic-looking relief from entirely
geometric rules, and both tile by construction.

**Truchet.**  Smith's arc variant of Truchet's tile puts two quarter-circle
arcs of radius s/2 at opposite corners of a square cell, in one of two
orientations.  The point is a continuity property: every arc meets each cell
edge *at its midpoint, perpendicular to the edge*, so arcs from any two
adjacent cells join tangentially no matter which orientation either cell drew.
A random assembly therefore produces smooth meandering lanes rather than a
broken scatter -- geometric rules, organic result, and no matching constraint
to satisfy.

**Seigaiha.**  The Japanese "blue sea wave": concentric arcs on a staggered
grid, each fan overlapping its neighbours so only the crown of each circle
shows.  It is the wave pattern that appears throughout *Hamonshu*, the 1903
craftsman's sourcebook, and it is what a wave looks like when it is drawn as
ornament rather than simulated.

References:
  Sebastien Truchet, "Memoire sur les combinaisons", Memoires de l'Academie
    Royale des Sciences, 1704.
  Cyril Stanley Smith (with Pauline Boucher), "The Tiling Patterns of Sebastien
    Truchet and the Topology of Structural Hierarchy", Leonardo 20(4), 1987,
    373-385 -- the translation of Truchet, and the arc variant used here.
  Mori Yuzan, "Hamonshu" (3 vols.), Kyoto, 1903 -- the wave-pattern
    sourcebook; seigaiha is its recurring motif.
"""

import math

import numpy as np


def truchet(X, Y, info, cells=6, seed=1, lane=0.25, multiscale=0):
    """Truchet arc tiling, lifted by distance to the arc set.

    `lane` is the lane width as a fraction of the cell.  `multiscale`
    subdivides a fraction of cells one level finer, in the spirit of Carlson's
    multi-scale Truchet patterns.
    """
    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    n = max(1, int(cells))
    s = 2.0 * hx / n                        # cell side
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

    u = (X + hx) / s                        # cell coordinates
    v = (Y + hy) / s
    ci = np.floor(u).astype(int)
    cj = np.floor(v).astype(int)
    nx = int(math.ceil(2.0 * hx / s))
    ny = int(math.ceil(2.0 * hy / s))
    orient = rng.integers(0, 2, size=(ny + 2, nx + 2))
    if multiscale:
        fine = rng.random((ny + 2, nx + 2)) < float(multiscale)
    else:
        fine = np.zeros((ny + 2, nx + 2), dtype=bool)

    ci_c = np.clip(ci, 0, nx + 1)
    cj_c = np.clip(cj, 0, ny + 1)
    o = orient[cj_c, ci_c]
    f = fine[cj_c, ci_c]

    # Local coordinates inside the cell, in [0, 1).
    lu = u - np.floor(u)
    lv = v - np.floor(v)
    # A finer cell is the same construction at twice the frequency.
    lu = np.where(f, (lu * 2.0) % 1.0, lu)
    lv = np.where(f, (lv * 2.0) % 1.0, lv)
    r = 0.5

    # Two quarter arcs, at opposite corners; orientation picks which pair.
    d_a = np.abs(np.hypot(lu, lv) - r)
    d_b = np.abs(np.hypot(lu - 1.0, lv - 1.0) - r)
    d_c = np.abs(np.hypot(lu - 1.0, lv) - r)
    d_d = np.abs(np.hypot(lu, lv - 1.0) - r)
    d = np.where(o == 0, np.minimum(d_a, d_b), np.minimum(d_c, d_d))

    w = max(float(lane), 1e-3) * 0.5
    t = np.clip(1.0 - d / w, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t) * 2.0 - 1.0


def seigaiha(X, Y, info, cells=6, seed=1, rings=3, crown=0.55):
    """Overlapping fans of concentric arcs on a staggered grid.

    Each circle shows only its crown, the rest hidden by the row in front --
    which is why the motif reads as water rather than as a grid of targets.
    `crown` is the fraction of each circle that stays visible.
    """
    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    n = max(1, int(cells))
    s = 2.0 * hx / n
    R = s * 0.75

    out = np.full(X.shape, -1.0)
    rows = int(math.ceil(2.0 * hy / (s * float(crown)))) + 2
    cols = n + 2
    for j in range(-1, rows):
        cy = -hy + j * s * float(crown)
        stagger = 0.5 * s * (j % 2)
        for i in range(-1, cols):
            cx = -hx + i * s + stagger
            d = np.hypot(X - cx, Y - cy)
            inside = d <= R
            if not inside.any():
                continue
            # Concentric arcs within the crown of this circle.
            band = np.cos(2.0 * math.pi * float(rings) * d / R)
            fade = np.clip(1.0 - d / R, 0.0, 1.0)
            v = np.where(inside, band * (0.35 + 0.65 * fade), -1.0)
            # Nearer rows sit in front: later circles overwrite earlier ones.
            out = np.where(inside, v, out)
    return out


def _selftest():
    ok = True
    from . import grid as _grid
    from . import tiling as _tiling

    X, Y, info = _grid.make_grid(width=2.0, aspect=1.0, resolution=192)

    for name, fn, kw in (("truchet", truchet, dict(cells=6, seed=2)),
                         ("seigaiha", seigaiha, dict(cells=5, seed=1))):
        h = fn(X, Y, info, **kw)
        print("tiles: %-9s range [%.2f, %.2f] finite=%s sd=%.4f"
              % (name, h.min(), h.max(), np.isfinite(h).all(), h.std()))
        ok = ok and np.isfinite(h).all() and h.std() > 1e-3
        ok = ok and h.shape == X.shape

    # Truchet lanes are continuous: the arc set forms long connected runs
    # rather than isolated blobs, which is the whole point of the tile.
    h = truchet(X, Y, info, cells=6, seed=2, lane=0.3)
    lane = h > 0.0
    runs = []
    for row in lane:
        best = cur = 0
        for val in row:
            cur = cur + 1 if val else 0
            best = max(best, cur)
        runs.append(best)
    print("tiles: truchet mean longest lane run = %.1f samples" % np.mean(runs))
    ok = ok and np.mean(runs) > 4.0

    # Determinism, and the seed genuinely changes the layout.
    a = truchet(X, Y, info, cells=6, seed=2)
    b = truchet(X, Y, info, cells=6, seed=2)
    c = truchet(X, Y, info, cells=6, seed=3)
    ok = ok and np.array_equal(a, b) and not np.allclose(a, c)
    print("tiles: truchet repeatable=%s seed-varies=%s"
          % (np.array_equal(a, b), not np.allclose(a, c)))

    # Cell count really is the cell count.
    few = truchet(X, Y, info, cells=3, seed=2)
    many = truchet(X, Y, info, cells=9, seed=2)
    edges = lambda z: int(np.sum(np.abs(np.diff(z[z.shape[0] // 2, :])) > 0.5))
    print("tiles: truchet 3 cells -> %d lane edges, 9 cells -> %d"
          % (edges(few), edges(many)))
    ok = ok and edges(many) > edges(few)

    # Multi-scale subdivision adds detail without breaking anything.
    ms = truchet(X, Y, info, cells=5, seed=4, multiscale=0.4)
    ok = ok and np.isfinite(ms).all() and ms.std() > 1e-3

    # Seigaiha is built on a staggered grid, so alternate rows are offset;
    # its horizontal profile differs between two rows a half-cell apart.
    sg = seigaiha(X, Y, info, cells=5)
    r1 = sg[sg.shape[0] // 3, :]
    r2 = sg[2 * sg.shape[0] // 3, :]
    print("tiles: seigaiha row-to-row rms difference %.3f (staggered)"
          % float(np.sqrt(((r1 - r2) ** 2).mean())))
    ok = ok and float(np.sqrt(((r1 - r2) ** 2).mean())) > 0.05

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
