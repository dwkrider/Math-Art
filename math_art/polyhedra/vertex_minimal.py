# Vertex-minimal polyhedra of genus 2 and 3 (exact integer data)
#
# The Csaszar polyhedron is a torus built on only seven vertices, the fewest
# any triangulated torus can have.  The same question can be asked of every
# surface.  Heawood's inequality says a triangulated closed surface of Euler
# characteristic chi needs at least (7 + sqrt(49 - 24.chi)) / 2 vertices;
# for the orientable surfaces the bound is attained (Jungerman and Ringel)
# with the single exception of genus 2, where it gives 9 but Huneke showed
# that 10 are needed.  So both genus 2 and genus 3 need exactly TEN vertices.
#
# Having the right number of vertices is a statement about the abstract
# triangulation.  Whether such a triangulation can be built in space with
# flat triangles that do not pass through one another is a separate and much
# harder question, open in general for genus 1 to 4.  For ten vertices it
# has been settled by search: all 865 vertex-minimal triangulations of the
# genus-2 surface, and all 20 of the genus-3 surface, are realisable, and
# nearly all with remarkably small integer coordinates -- every genus-2 one
# fits in general position in the 4x4x4 cube (and none in the 3x3x3), and at
# least 17 of the genus-3 ones in the 5x5x5 cube (and none in the 4x4x4).
#
# Two of those coordinate-minimal realisations are stored here, each chosen
# by its authors for display because a hole is clearly visible: surface
# No. 11909 (genus 2: 10 vertices, 36 edges, 24 triangles) and No. 14542
# (genus 3: 10 vertices, 42 edges, 28 triangles).  The numbers are positions
# in Lutz's catalogue of the 42426 triangulated surfaces with ten vertices.
# The triangles are published unoriented; they are oriented outwards here,
# and everything is checked exactly in integers.
#
# References:
# - Stefan Hougardy, Frank H. Lutz and Mariano Zelke, "Polyhedra of genus 2
#   with 10 vertices and minimal coordinates", Electronic Geometry Models
#   No. 2005.08.001 (2007), arXiv:math/0507592 -- the genus-2 realisation
#   (surface No. 11909) in the 4x4x4 cube.
# - Stefan Hougardy, Frank H. Lutz and Mariano Zelke, "Polyhedra of genus 3
#   with 10 vertices and minimal coordinates", Electronic Geometry Models
#   No. 2006.02.001 (2007), arXiv:math/0604017 -- the genus-3 realisation
#   (surface No. 14542) in the 5x5x5 cube.
# - Frank H. Lutz, "Enumeration and random realization of triangulated
#   surfaces", arXiv:math/0506316 (2005) -- the catalogue of the 42426
#   ten-vertex triangulated surfaces the numbers refer to.
# - Percy J. Heawood, "Map-colour theorem", Quarterly Journal of Pure and
#   Applied Mathematics 24 (1890), 332-338 -- the lower bound on vertices.
# - Mark Jungerman and Gerhard Ringel, "Minimal triangulations on orientable
#   surfaces", Acta Mathematica 145 (1980), 121-154 -- the bound is attained.
# - John Philip Huneke, "A minimum-vertex triangulation", Journal of
#   Combinatorial Theory, Series B 24 (1978), 258-266 -- genus 2 needs ten.

import math
from itertools import combinations

VERTEX_MINIMAL = {
    "MINIMAL_G2": {
        "name": "Ten-Vertex Genus-2 Polyhedron",
        "genus": 2, "cube": 4,
        "V": [(0, 0, 0), (0, 0, 1), (0, 1, 4), (2, 3, 2), (1, 3, 4),
              (3, 3, 3), (3, 4, 0), (2, 1, 1), (4, 0, 2), (3, 2, 1)],
        "T": [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 6),
              (0, 5, 7), (0, 6, 8), (0, 7, 9), (0, 8, 9), (1, 2, 5),
              (1, 3, 4), (1, 4, 7), (1, 5, 8), (1, 7, 8), (2, 4, 5),
              (3, 4, 6), (3, 5, 9), (3, 6, 7), (3, 7, 8), (3, 8, 9),
              (4, 5, 7), (5, 6, 8), (5, 6, 9), (6, 7, 9)],
    },
    "MINIMAL_G3": {
        "name": "Ten-Vertex Genus-3 Polyhedron",
        "genus": 3, "cube": 5,
        "V": [(0, 0, 0), (0, 0, 4), (0, 2, 0), (3, 2, 4), (5, 3, 2),
              (1, 1, 2), (5, 4, 5), (5, 1, 1), (4, 2, 1), (1, 5, 0)],
        "T": [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 6),
              (0, 5, 7), (0, 6, 8), (0, 7, 9), (0, 8, 9), (1, 2, 5),
              (1, 3, 4), (1, 4, 8), (1, 5, 9), (1, 6, 7), (1, 6, 9),
              (1, 7, 8), (2, 3, 6), (2, 3, 9), (2, 4, 8), (2, 5, 6),
              (2, 8, 9), (3, 4, 7), (3, 5, 9), (3, 6, 7), (4, 6, 9),
              (4, 7, 9), (5, 6, 8), (5, 7, 8)],
    },
}


def _orient(p, q, r, s):
    """Six times the signed volume of the tetrahedron pqrs (an integer)."""
    a = [q[i] - p[i] for i in range(3)]
    b = [r[i] - p[i] for i in range(3)]
    c = [s[i] - p[i] for i in range(3)]
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def _oriented(V, T):
    """The triangles with a consistent orientation, facing outwards."""
    tris = [list(t) for t in T]
    by_edge = {}
    for i, t in enumerate(tris):
        for k in range(3):
            by_edge.setdefault(frozenset((t[k], t[(k + 1) % 3])),
                               []).append(i)
    done = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        t = tris[i]
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            for j in by_edge[frozenset((a, b))]:
                if j in done:
                    continue
                u = tris[j]
                # the neighbour must run along the shared edge backwards
                if any(u[m] == a and u[(m + 1) % 3] == b for m in range(3)):
                    tris[j] = u[::-1]
                done.add(j)
                stack.append(j)
    assert len(done) == len(tris), "surface is not connected"
    origin = (0, 0, 0)
    vol6 = sum(_orient(origin, V[a], V[b], V[c]) for a, b, c in tris)
    if vol6 < 0:
        tris = [t[::-1] for t in tris]
    return tris


def build(kind):
    """(V, F): integer vertices as published, triangles oriented outwards."""
    S = VERTEX_MINIMAL[kind]
    V = [tuple(p) for p in S["V"]]
    return V, _oriented(V, S["T"])


def heawood_bound(genus):
    """Heawood's lower bound on the vertices of a triangulated orientable
    surface of this genus."""
    disc = 1 + 48 * genus                  # 49 - 24.chi with chi = 2 - 2g
    root = math.isqrt(disc)
    twice = 7 + root + (0 if root * root == disc else 1)   # ceil(7 + sqrt)
    return (twice + 1) // 2


def _check(kind):
    S = VERTEX_MINIMAL[kind]
    V, F = build(kind)
    n, g = S["cube"], S["genus"]

    # the published box, filled in every direction
    for axis in range(3):
        cs = [p[axis] for p in V]
        assert (min(cs), max(cs)) == (0, n), (kind, axis)
    # general position: no four of the ten points in a plane
    for q in combinations(V, 4):
        assert _orient(*q) != 0, (kind, "four coplanar points", q)

    # a closed oriented surface: each directed edge once, with its reverse
    darts = [(t[k], t[(k + 1) % 3]) for t in F for k in range(3)]
    assert len(set(darts)) == len(darts)
    assert all((b, a) in set(darts) for a, b in darts)
    nE = len(darts) // 2
    assert sorted(tuple(sorted(t)) for t in F) == sorted(S["T"])
    # every vertex link is a single cycle
    for v in range(len(V)):
        ring = {a: b for t in F for k in range(3)
                for a, b in [(t[(k + 1) % 3], t[(k + 2) % 3])] if t[k] == v}
        start = next(iter(ring))
        seen, w = 0, start
        while True:
            w = ring[w]
            seen += 1
            if w == start:
                break
        assert seen == len(ring), (kind, "pinched vertex", v)
    chi = len(V) - nE + len(F)
    assert chi == 2 - 2 * g, (kind, chi)

    # outward, and embedded: in general position two triangles can only
    # meet wrongly if an edge of one passes through the inside of the other
    assert sum(_orient((0, 0, 0), V[a], V[b], V[c]) for a, b, c in F) > 0
    edges = sorted({tuple(sorted(d)) for d in darts})
    for p, q in edges:
        for a, b, c in F:
            if p in (a, b, c) or q in (a, b, c):
                continue
            if _orient(V[a], V[b], V[c], V[p]) * \
                    _orient(V[a], V[b], V[c], V[q]) > 0:
                continue
            s = (_orient(V[p], V[q], V[a], V[b]),
                 _orient(V[p], V[q], V[b], V[c]),
                 _orient(V[p], V[q], V[c], V[a]))
            assert not (all(x > 0 for x in s) or all(x < 0 for x in s)), \
                (kind, "edge pierces triangle", (p, q), (a, b, c))
    return len(V), nE, len(F)


def _selftest():
    assert [heawood_bound(g) for g in range(8)] == [4, 7, 9, 10, 11, 12, 12,
                                                    13]
    counts = _check("MINIMAL_G2")
    assert counts == (10, 36, 24), counts
    # genus 2 is the one orientable exception: one more than Heawood's 9
    assert counts[0] == heawood_bound(2) + 1
    print("genus 2  No. 11909  V=10 E=36 F=24, general position in the "
          "4x4x4 cube, closed, oriented outwards, embedded")
    counts = _check("MINIMAL_G3")
    assert counts == (10, 42, 28), counts
    assert counts[0] == heawood_bound(3)
    print("genus 3  No. 14542  V=10 E=42 F=28, general position in the "
          "5x5x5 cube, closed, oriented outwards, embedded")
    # the embedding test can fail: moving one vertex across the solid
    # makes it report a piercing edge
    broken = dict(VERTEX_MINIMAL["MINIMAL_G3"])
    broken["V"] = list(broken["V"])
    broken["V"][5] = (4, 4, 3)
    VERTEX_MINIMAL["_BROKEN"] = broken
    try:
        _check("_BROKEN")
    except AssertionError:
        pass
    else:
        raise AssertionError("a damaged solid passed the checks")
    finally:
        del VERTEX_MINIMAL["_BROKEN"]
    print("RESULT: OK")
