
# Map L-systems: tissue by developmental division.
#
# WHY THIS IS NOT VORONOI.  The add-on already makes cells by
# PARTITIONING -- voronoi_openwork, organic_wireframe, bubble.  Those
# take a region and cut it up.  A map L-system makes cells by DIVISION:
# it starts with one cell and every cell splits, so the tissue has a
# developmental history.  The difference shows: graded cell sizes
# (recently divided cells are smaller), walls that meet at consistent
# angles because they were laid down in order, and an outline that
# EMERGES from the growth instead of being the boundary you started
# with.  Voronoi randomness looks scattered; this looks grown.
#
# WHICH FORMALISM.  Rozenberg and Salomaa set out three approaches --
# PG0L systems on the dual graph with stencils, direct map processing,
# and graph processing.  This is the second, the mBPMOL line that ABOP
# chapter 7 follows, chosen because its markers correspond to a real
# biological object (the preprophase band of microtubules, which
# predicts where the division plate will form) and because it needs no
# stencil library.
#
# Two constraints hold in all three formalisms and are respected here:
#
#   * NO CELL DISAPPEARS.  Nothing is ever erased; allowing it "would
#     complicate these models quite considerably".
#   * ADJACENCY IS INHERITED.  Two new elements can be adjacent only if
#     the elements that produced them were -- the same locality that
#     makes an L-system propagating.
#
# THE DIVISION PLANE is the whole shape control.  A wall laid down
# PERICLINAL to the surface (parallel to it) thickens the tissue; one
# laid ANTICLINAL (perpendicular) extends it.  The ratio P/A between the
# two is the single dial: low values keep the sheet convex, high values
# drive it concave.
#
# THE SOLVER is the Fracchia mass-spring interpretation: cell and tissue
# shape is the equilibrium between internal turgor pressure and wall
# tension.  Walls are springs; each cell pushes outward with
# p = nRT / (A * h), so a small cell pushes harder -- which is what
# equalises cell sizes without any explicit rule saying so.  Forward
# Euler, because the appeal of mass-spring over finite elements is
# precisely that it is simple enough to actually implement.
#
# Note that map L-systems trigger division by ELAPSED TIME rather than
# by cell size, which is the opposite of the Korn/Errera line -- so the
# UI dial is a step count, not a size threshold.
#
# References:
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990, chapters 6 and 7.
# - Grzegorz Rozenberg and Arto Salomaa, "The Mathematical Theory of L
#   Systems", Academic Press, 1980, chapter 27 -- the three approaches
#   and the two shared constraints.
# - F. David Fracchia, Przemyslaw Prusinkiewicz and Martin de Boer,
#   "Animation of the development of multicellular structures", in
#   Computer Animation '90, Springer, 1990 -- the mass-spring model.
# - Przemyslaw Prusinkiewicz and Adam Runions, "Computational models of
#   plant development and form", New Phytologist 193, 2012, section IV.1
#   -- wall state and wall age, and the Errera to Besson-Dumais line.
# - Leo Errera, "Sur une condition fondamentale d'equilibre des cellules
#   vivantes", C. R. Acad. Sci. Paris 103, 1886 -- the shortest-wall rule.

import numpy as np

ANTICLINAL, PERICLINAL = "ANTICLINAL", "PERICLINAL"


class CellMap:
    """Vertices plus cells, each cell an ordered ring of vertex indices."""

    __slots__ = ("verts", "cells", "ages")

    def __init__(self, verts, cells, ages=None):
        self.verts = np.asarray(verts, dtype=float)
        self.cells = [list(c) for c in cells]
        #: how many divisions each cell has been through -- wall age, in
        #: the sense of the 2012 review
        self.ages = list(ages) if ages else [0] * len(self.cells)

    def copy(self):
        return CellMap(self.verts.copy(), [list(c) for c in self.cells],
                       list(self.ages))

    def centroid(self, ci):
        return self.verts[self.cells[ci]].mean(axis=0)

    def area(self, ci):
        P = self.verts[self.cells[ci]]
        x, y = P[:, 0], P[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                               - np.dot(np.roll(x, -1), y)))

    def areas(self):
        return np.array([self.area(i) for i in range(len(self.cells))])

    def edges(self):
        """Every wall, as an unordered vertex pair."""
        out = set()
        for c in self.cells:
            for i in range(len(c)):
                a, b = c[i], c[(i + 1) % len(c)]
                out.add((min(a, b), max(a, b)))
        return sorted(out)


def square(size=1.0):
    """A one-cell map: the zygote."""
    h = 0.5 * float(size)
    return CellMap([[-h, -h], [h, -h], [h, h], [-h, h]], [[0, 1, 2, 3]])


def _split_cell(cm, ci, direction):
    """Cut cell `ci` with a line through its centroid along `direction`.

    Returns the two new cells' vertex rings, inserting the two wall
    endpoints into the map.  The new wall is shared, so adjacency is
    inherited by construction -- the daughters are adjacent because the
    mother was one cell.
    """
    ring = cm.cells[ci]
    P = cm.verts[ring]
    c = P.mean(axis=0)
    d = np.asarray(direction, dtype=float)[:2]
    d = d / (np.linalg.norm(d) or 1.0)
    nrm = np.array([-d[1], d[0]])

    side = (P - c).dot(nrm)
    hits = []
    n = len(ring)
    for i in range(n):
        s0, s1 = side[i], side[(i + 1) % n]
        if s0 == 0.0:
            s0 = 1e-15
        if s0 * s1 < 0:
            f = s0 / (s0 - s1)
            pt = P[i] + (P[(i + 1) % n] - P[i]) * f
            hits.append((i, pt))
    if len(hits) != 2:
        return None

    (i0, p0), (i1, p1) = hits
    base = len(cm.verts)
    cm.verts = np.vstack([cm.verts, p0, p1])
    a, b = base, base + 1

    left = [a] + [ring[k % n] for k in range(i0 + 1, i1 + 1)] + [b]
    right = [b] + [ring[k % n] for k in range(i1 + 1, i0 + 1 + n)] + [a]
    if len(left) < 3 or len(right) < 3:
        return None
    return left, right


def divide(cm, pa=1.0, seed=0, axis=(0.0, 1.0), jitter=0.25):
    """One developmental step: every cell divides once.

    `pa` is the periclinal/anticlinal ratio.  A periclinal wall lies
    parallel to the tissue surface and thickens it; an anticlinal wall
    is perpendicular and extends it.  Below about 1.25 the sheet stays
    convex; above 2 it turns concave.

    NO CELL IS ERASED, which is the formalism's own constraint: a
    division replaces one cell with two, never with none.
    """
    rng = np.random.default_rng(int(seed))
    out = cm.copy()
    ax = np.asarray(axis, dtype=float)[:2]
    ax = ax / (np.linalg.norm(ax) or 1.0)
    per = np.array([-ax[1], ax[0]])

    new_cells, new_ages = [], []
    p_peri = float(pa) / (1.0 + float(pa))
    for ci in range(len(cm.cells)):
        want_peri = rng.random() < p_peri
        d = per if want_peri else ax
        if jitter:
            a = rng.uniform(-jitter, jitter)
            ca, sa = np.cos(a), np.sin(a)
            d = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
        got = _split_cell(out, ci, d)
        if got is None:
            new_cells.append(list(cm.cells[ci]))
            new_ages.append(cm.ages[ci])
            continue
        left, right = got
        new_cells.extend([left, right])
        new_ages.extend([cm.ages[ci] + 1] * 2)
    out.cells = new_cells
    out.ages = new_ages
    return out


def relax(cm, steps=60, pressure=0.02, stiffness=0.6, dt=0.25,
          rest=None, pin=None):
    """Mass-spring relaxation with turgor pressure.

    Walls pull as springs; each cell pushes its ring outward with
    p = k / A, so a small cell pushes harder than a large one.  That
    single term is what equalises cell size and rounds the tissue --
    there is no rule anywhere saying cells should be similar or convex.

    Forward Euler, scattered with `np.add.at` so every wall and cell
    contributes in one pass.
    """
    V = cm.verts.copy()
    E = np.array(cm.edges(), dtype=int)
    if not len(E):
        return cm
    if rest is None:
        rest = float(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean())
    pin = set() if pin is None else set(pin)

    for _ in range(int(steps)):
        F = np.zeros_like(V)
        # wall tension
        d = V[E[:, 1]] - V[E[:, 0]]
        L = np.linalg.norm(d, axis=1)
        L[L < 1e-12] = 1e-12
        f = (float(stiffness) * (L - rest) / L)[:, None] * d
        np.add.at(F, E[:, 0], f)
        np.add.at(F, E[:, 1], -f)

        # turgor: p = nRT / (A h), so the force is inversely proportional
        # to the cell's area
        for ci, ring in enumerate(cm.cells):
            P = V[ring]
            c = P.mean(axis=0)
            a = max(float(np.abs(0.5 * (
                np.dot(P[:, 0], np.roll(P[:, 1], -1))
                - np.dot(np.roll(P[:, 0], -1), P[:, 1])))), 1e-9)
            out = P - c
            n = np.linalg.norm(out, axis=1)
            n[n < 1e-12] = 1e-12
            F[ring] += (out / n[:, None]) * (float(pressure) / a)

        V = V + F * float(dt)
        for i in pin:
            V[i] = cm.verts[i]
    return CellMap(V, cm.cells, cm.ages)


def layer(steps=5, pa=1.0, seed=0, relax_steps=60, pressure=0.02,
          size=1.0):
    """A cellular layer -- the fern-gametophyte case.

    Divide, relax, repeat.  Relaxing between divisions is what makes the
    tissue look grown: each generation settles before the next divides,
    so cell size is graded by age rather than uniform.
    """
    cm = square(size)
    for k in range(int(steps)):
        cm = divide(cm, pa=pa, seed=int(seed) + k)
        cm = relax(cm, steps=relax_steps, pressure=pressure)
    return cm


def cell_ball(steps=3, seed=0, radius=1.0, relax_steps=40,
              pressure=0.02):
    """The spherical cellular layer -- a Patella vulgata style blastula.

    The same division rule applied to a closed surface: start from an
    octahedron's faces as cells, split each, and re-project onto the
    sphere so the tissue stays a shell.  Returns (verts3d, cells).
    """
    a = 1.0
    V = [(a, 0, 0), (-a, 0, 0), (0, a, 0),
         (0, -a, 0), (0, 0, a), (0, 0, -a)]
    F = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
         [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    V = [np.asarray(p, dtype=float) for p in V]
    rng = np.random.default_rng(int(seed))

    for _ in range(int(steps)):
        newF = []
        for ring in F:
            # Split across the longest diagonal -- Errera's shortest-
            # wall rule, applied on the sphere.  Try each candidate in
            # order of length until one yields two valid rings: a single
            # fixed choice silently failed on some cells and left them
            # undivided, so the shell grew 8 -> 16 -> 24 instead of
            # doubling.
            P = np.array([V[i] for i in ring])
            n = len(ring)
            order = sorted(range(n),
                           key=lambda i: -float(np.linalg.norm(
                               P[i] - P[(i + n // 2) % n])))
            done = False
            for i0 in order:
                i1 = (i0 + n // 2) % n
                if i1 == i0:
                    continue
                mids = []
                for (s, e) in ((i0, (i0 + 1) % n), (i1, (i1 + 1) % n)):
                    m = 0.5 * (V[ring[s]] + V[ring[e]])
                    m = m / (np.linalg.norm(m) or 1.0)
                    m = m * (1.0 + rng.uniform(-0.02, 0.02))
                    mids.append(m)
                left_ids = [ring[k % n] for k in range(i0 + 1, i1 + 1)]
                right_ids = [ring[k % n] for k in range(i1 + 1, i0 + 1 + n)]
                if len(left_ids) < 1 or len(right_ids) < 1:
                    continue
                base = len(V)
                V.extend(mids)
                a0, a1 = base, base + 1
                left = [a0] + left_ids + [a1]
                right = [a1] + right_ids + [a0]
                if len(left) >= 3 and len(right) >= 3:
                    newF.extend([left, right])
                    done = True
                    break
                del V[base:]
            if not done:
                newF.append(ring)
        F = newF

    P = np.array(V)
    P = P / np.linalg.norm(P, axis=1)[:, None] * float(radius)
    return P, F


def convexity(cm):
    """How convex the tissue outline is: hull area over actual area.

    1.0 is convex; the further above, the more the outline dishes in.
    This is what the P/A dial moves.
    """
    V = cm.verts
    pts = V[np.unique([i for c in cm.cells for i in c])]
    if len(pts) < 3:
        return 1.0
    hull = _hull_area(pts)
    tot = float(cm.areas().sum())
    return hull / max(tot, 1e-12)


def _hull_area(pts):
    P = np.asarray(pts, dtype=float)
    idx = np.lexsort((P[:, 1], P[:, 0]))
    P = P[idx]

    def half(Q):
        out = []
        for p in Q:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - \
                        (b[1] - a[1]) * (p[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    h = half(P)[:-1] + half(P[::-1])[:-1]
    H = np.array(h)
    if len(H) < 3:
        return 0.0
    x, y = H[:, 0], H[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                           - np.dot(np.roll(x, -1), y)))


def _selftest():
    # --- the zygote ----------------------------------------------------
    cm = square()
    assert len(cm.cells) == 1
    assert abs(cm.area(0) - 1.0) < 1e-12

    # --- division -------------------------------------------------------
    # Every cell divides in two, and NO CELL DISAPPEARS -- that is a
    # constraint of the formalism, not an implementation detail.
    prev = 1
    c = square()
    for k in range(6):
        c = divide(c, pa=1.0, seed=k, jitter=0.0)
        assert len(c.cells) == 2 * prev, (k, len(c.cells), prev)
        prev = len(c.cells)
        for ring in c.cells:
            assert len(ring) >= 3, ring
        # area is conserved by division: cutting a cell adds nothing
        assert abs(float(c.areas().sum()) - 1.0) < 1e-9, k
        # every cell has positive area -- a degenerate cell is a lost one
        assert float(c.areas().min()) > 0.0, k

    # ADJACENCY IS INHERITED: the two daughters share exactly the new
    # wall, so they are adjacent because their mother was one cell.
    one = divide(square(), pa=1.0, seed=0, jitter=0.0)
    s0, s1 = set(one.cells[0]), set(one.cells[1])
    assert len(s0 & s1) == 2, sorted(s0 & s1)

    # --- pressure relaxation ---------------------------------------------
    # Turgor must INFLATE the tissue, and it must even out cell sizes,
    # since a small cell pushes harder than a large one.
    grown = divide(square(), pa=1.0, seed=3)
    grown = divide(grown, pa=1.0, seed=4)
    before = grown.areas()
    after = relax(grown, steps=80, pressure=0.02).areas()
    assert float(after.sum()) > float(before.sum()), \
        (float(before.sum()), float(after.sum()))
    spread = lambda a: float(a.std() / a.mean())      # noqa: E731
    assert spread(after) < spread(before) + 1e-9, (spread(before),
                                                   spread(after))
    # Zero pressure is not the same as no change -- the springs still
    # equalise wall lengths, and that is not area-neutral.  What must
    # hold is that turgor drives the inflation: more pressure, more
    # area, monotonically.
    tot = [float(relax(grown, steps=80, pressure=q).areas().sum())
           for q in (0.0, 0.01, 0.02, 0.04)]
    assert all(b > a for a, b in zip(tot, tot[1:])), tot

    # --- the P/A dial ----------------------------------------------------
    # This is the one shape control, so it has to demonstrably control
    # the shape: more periclinal division dishes the outline in.
    conv = {}
    for pa in (0.3, 1.0, 4.0):
        L = layer(steps=4, pa=pa, seed=11, relax_steps=40)
        assert len(L.cells) == 16, (pa, len(L.cells))
        assert float(L.areas().min()) > 0.0
        conv[pa] = convexity(L)
    assert conv[4.0] > conv[0.3], conv

    # cells are GRADED, not uniform -- which is the difference from a
    # partitioning method
    L = layer(steps=5, pa=1.0, seed=2)
    a = L.areas()
    assert float(a.max() / a.min()) > 1.5, float(a.max() / a.min())

    # --- the cell ball ----------------------------------------------------
    for s in (1, 2, 3):
        P, F = cell_ball(steps=s, seed=1)
        assert len(F) == 8 * 2 ** s, (s, len(F))
        assert np.all(np.isfinite(P))
        r = np.linalg.norm(P, axis=1)
        # it is a SHELL: every vertex on the sphere
        assert abs(float(r.max() - r.min())) < 1e-9, (float(r.min()),
                                                      float(r.max()))
        for ring in F:
            assert len(ring) >= 3

    print(f"maps: OK -- division doubles cells and conserves area, no "
          f"cell erased, daughters share exactly the new wall, turgor "
          f"inflates and evens sizes, P/A moves convexity "
          f"{conv[0.3]:.3f}->{conv[4.0]:.3f}, cell ball stays a shell")
