# Face-neighbourly polyhedra of genus 3 (exact data + shared-edge helpers)
#
# A polyhedron is FACE-NEIGHBOURLY when every one of its faces shares an
# edge with every other face.  The tetrahedron (4 faces, genus 0) and the
# Szilassi polyhedron (7 hexagons, genus 1) were for decades the only known
# examples.  If every pair of faces shares EXACTLY one edge and three faces
# meet at every vertex, then E = n(n-1)/2, 3V = 2E and Euler's formula force
# the genus to be (n-3)(n-4)/12, which is an integer only for n = 0, 3, 4, 7
# (mod 12): after 4 and 7 the next case is 12 faces on a surface of genus 6,
# and whether that one exists is open.
#
# Letting a pair of faces meet along TWO collinear edges opens up n = 8.
# Two such polyhedra are known, both of genus 3 with eight simple non-convex
# nonagonal faces, 24 vertices and 36 edges (equivelar type {9,3}), in which
# 20 pairs of faces share one edge and 8 pairs share two:
#
#   * Mizhaev's polyhedron M -- integer vertices, symmetry group C4 generated
#     by the rotoreflection T(x, y, z) = (y, -x, -z), achiral.  Its eight
#     doubled pairs form a single 8-cycle.
#   * The Rost-Vigh polyhedron P -- rational vertices, symmetry group D2 (the
#     three half-turns about the coordinate axes), chiral in every
#     realisation.  Its eight doubled pairs form two 4-cycles, which is what
#     proves the two solids combinatorially different.
#
# For contrast the module also keeps Mizhaev's OTHER eight-nonagon polyhedron
# of genus 3, the first variant (V1) of his 2020 preprint, of which M is the
# second.  It has the same counts and the same rotoreflection symmetry, but
# it is not face-neighbourly: four pairs of faces never meet, twelve share two
# edges and twelve share one.  So its faces can be coloured as a map with
# only four colours -- the four pairs that never meet each share one --
# against eight for M and P.  The preprint prints each face as a SET of
# vertices in increasing order; the boundary cycles below are the ones the
# geometry forces (two faces meeting at a vertex of this three-valent surface
# share an edge there, so the vertices two faces have in common, sorted along
# the line where their planes meet, pair off into the shared edges).
#
# Everything here is stored and checked EXACTLY (integers / fractions).  The
# vertices of P are not typed in at all: each is derived as the intersection
# of the planes of the three faces that meet there, and the published table
# is kept only as the cross-check.  Mizhaev's face walks are consistently
# orientable but, as printed, his orientation gives inward normals; the
# walks of F1, F2, F7 and F8 are reversed here so that every face is
# counter-clockwise seen from outside.
#
# References:
# - Gergely Rost and Viktor Vigh, "A second eight-faced polyhedron in which
#   every two faces share an edge", arXiv:2609.32998 (2026) -- the
#   polyhedron P: face planes, boundary cycles, vertex table, the D2
#   symmetry and the comparison with Mizhaev's polyhedron.
# - Ruslan Mizhaev, "Integer realization of an equivelar octahedron of
#   genus 3", arXiv:2609.17700 (2026) -- the polyhedron M: integer vertex
#   coordinates, face walks and plane equations.
# - Ruslan Mizhaev, "Equivelar octahedron of genus 3 in 3-space", OSF
#   Preprints (2020), doi:10.31219/osf.io/hvtey -- the original construction,
#   with the two variants V1 (kept here as MIZHAEV_V1) and V2 (the
#   combinatorial type of M).
# - Lajos Szilassi, "Regular toroids", Structural Topology 13 (1986),
#   69-80 -- the seven-faced face-neighbourly torus these two follow.

from fractions import Fraction
from itertools import combinations


def _half_turns(plane):
    """The plane and its images under the half-turns Rz, Rx, Ry."""
    a, b, c, d = plane
    return [(a, b, c, d), (-a, -b, c, d), (a, -b, -c, d), (-a, b, -c, d)]


# --- Rost-Vigh polyhedron P ------------------------------------------------
# Planes a.x + b.y + c.z = d, in the order A1..A4, B1..B4.
_P_PLANES = _half_turns((3, -4, -2, 5)) + _half_turns((-2, 5, -5, 3))
# Boundary cycles (1-based vertex numbers of the paper's Table 2),
# counter-clockwise seen from outside.
_P_CYCLES = [
    [1, 2, 5, 7, 17, 13, 21, 12, 9],        # A1
    [6, 8, 18, 14, 22, 11, 10, 2, 1],       # A2
    [7, 5, 19, 15, 23, 10, 11, 3, 4],       # A3
    [4, 3, 8, 6, 20, 16, 24, 9, 12],        # A4
    [1, 9, 24, 23, 15, 14, 18, 20, 6],      # B1
    [13, 17, 19, 5, 2, 10, 23, 24, 16],     # B2
    [21, 13, 16, 20, 18, 8, 3, 11, 22],     # B3
    [4, 12, 21, 22, 14, 15, 19, 17, 7],     # B4
]
# Table 2 of the paper, one representative per D2 orbit (vertices 1, 5, 9,
# 13, 17, 21); the orbit runs v, Rz v, Rx v, Ry v.  Cross-check only.
_P_TABLE2 = [
    ((-38, 7), (-57, 14), (-5, 2)), ((5, 3), (-1, 15), (2, 15)),
    ((-37, 38), (-5, 4), (-111, 76)), ((13, 11), (-3, 5), (26, 55)),
    ((3, 2), (-1, 4), (1, 4)), ((31, 23), (-62, 115), (3, 5)),
]

# --- Mizhaev's polyhedron M ------------------------------------------------
_M_PLANES = [
    (21, 3, -10, -1440), (21, 3, 10, 1440), (3, 0, 1, 234), (3, 0, -1, -234),
    (0, 3, -1, 234), (0, 3, 1, -234), (3, -21, 10, -1440),
    (3, -21, -10, 1440),
]
_M_VERTS = [
    (-72, 84, 18), (-36, 112, 102), (72, -84, 18), (36, -112, 102),
    (0, 300, 234), (-84, 48, -18), (9, 147, 207), (0, -300, 234),
    (84, -48, -18), (-112, -36, -102), (48, 84, 18), (-147, 9, -207),
    (-9, -147, 207), (84, 72, -18), (112, 36, -102), (-48, -84, 18),
    (147, -9, -207), (-18, 126, 144), (-126, -18, -144), (-300, 0, -234),
    (-84, -72, -18), (18, -126, 144), (126, 18, -144), (300, 0, -234),
]
# Face walks exactly as printed (F1..F8) ...
_M_WALKS = [
    [6, 10, 16, 22, 18, 7, 5, 1, 2], [9, 15, 11, 18, 22, 13, 8, 3, 4],
    [7, 5, 8, 3, 17, 23, 9, 15, 14], [13, 8, 5, 1, 12, 19, 6, 10, 21],
    [1, 2, 11, 18, 7, 14, 24, 20, 12], [3, 4, 16, 22, 13, 21, 20, 24, 17],
    [14, 24, 17, 23, 19, 6, 2, 11, 15], [10, 21, 20, 12, 19, 23, 9, 4, 16],
]
# ... and the ones that must be reversed to face outwards (0-based).
_M_REVERSED = (0, 1, 6, 7)

# --- Mizhaev's 2020 variant V1 ----------------------------------------------
# Planes (the faces of two tetrahedra), in the preprint's order; which face
# lies on which is found by incidence.
_V1_PLANES = [
    (0, 1, 1, -100), (1, 0, -1, -100), (0, 1, -1, 100), (1, 0, 1, 100),
    (1, -7, -4, 384), (7, 1, -4, -384), (1, -7, 4, -384), (7, 1, 4, 384),
]
# Table 2 of the preprint: (numerator, denominator) per coordinate.
_V1_VERTS = [
    ((-308, 1), (-108, 1), (-208, 1)), ((308, 1), (108, 1), (-208, 1)),
    ((16, 1), (0, 1), (-100, 1)), ((-16, 1), (0, 1), (-100, 1)),
    ((28, 1), (4, 1), (-96, 1)), ((-28, 1), (-4, 1), (-96, 1)),
    ((-4272, 41), (-448, 41), (-3652, 41)),
    ((4272, 41), (448, 41), (-3652, 41)),
    ((-182, 1), (-18, 1), (-82, 1)), ((182, 1), (18, 1), (-82, 1)),
    ((-1216, 13), (-336, 13), (-964, 13)),
    ((1216, 13), (336, 13), (-964, 13)),
    ((336, 13), (-1216, 13), (964, 13)),
    ((-336, 13), (1216, 13), (964, 13)),
    ((18, 1), (-182, 1), (82, 1)), ((-18, 1), (182, 1), (82, 1)),
    ((448, 41), (-4272, 41), (3652, 41)),
    ((-448, 41), (4272, 41), (3652, 41)),
    ((-4, 1), (28, 1), (96, 1)), ((4, 1), (-28, 1), (96, 1)),
    ((0, 1), (-16, 1), (100, 1)), ((0, 1), (16, 1), (100, 1)),
    ((-108, 1), (308, 1), (208, 1)), ((108, 1), (-308, 1), (208, 1)),
]
# Table 3 of the preprint, as printed: each face's vertex SET.
_V1_FACE_SETS = [
    [1, 3, 5, 6, 7, 9, 12, 14, 18], [2, 10, 13, 15, 17, 20, 21, 22, 23],
    [8, 12, 13, 16, 18, 19, 20, 21, 23], [2, 4, 5, 6, 8, 10, 11, 13, 17],
    [1, 4, 3, 5, 8, 10, 12, 16, 23], [1, 9, 14, 16, 18, 19, 22, 21, 24],
    [7, 11, 14, 15, 17, 19, 20, 22, 24], [2, 3, 4, 6, 7, 9, 11, 15, 24],
]
# The boundary cycles those sets force, counter-clockwise from outside.
_V1_CYCLES = [
    [3, 6, 5, 12, 18, 14, 7, 9, 1], [10, 23, 21, 22, 20, 13, 17, 15, 2],
    [8, 13, 20, 19, 21, 23, 16, 18, 12], [4, 5, 6, 11, 17, 13, 8, 10, 2],
    [1, 16, 23, 10, 8, 12, 5, 4, 3], [9, 24, 22, 21, 19, 14, 18, 16, 1],
    [14, 19, 20, 22, 24, 15, 17, 11, 7], [2, 15, 24, 9, 7, 11, 6, 3, 4],
]

NEIGHBOURLY = {
    "ROST_VIGH": {
        "name": "Rost-Vigh Polyhedron",
        "labels": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
    },
    "MIZHAEV": {
        "name": "Mizhaev Polyhedron",
        "labels": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
    },
    # not face-neighbourly: kept for contrast (see the header)
    "MIZHAEV_V1": {
        "name": "Mizhaev Polyhedron V1",
        "labels": ["1", "2", "3", "4", "5", "6", "7", "8"],
    },
}


def _det3(a, b, c):
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def _solve3(p, q, r):
    """The point on three planes (a, b, c, d), exactly; None if they do not
    meet in a single point."""
    rows = (p, q, r)
    den = _det3(*(row[:3] for row in rows))
    if den == 0:
        return None
    out = []
    for i in range(3):
        m = [list(row[:3]) for row in rows]
        for j in range(3):
            m[j][i] = rows[j][3]
        out.append(Fraction(_det3(*m), den))
    return tuple(out)


def exact(kind):
    """(V, F, planes): exact vertices (Fractions), 0-based outward face
    cycles, and the face planes (a, b, c, d), one per face."""
    if kind == "ROST_VIGH":
        F = [[v - 1 for v in cyc] for cyc in _P_CYCLES]
        V = []
        for v in range(24):
            on = [_P_PLANES[f] for f, cyc in enumerate(F) if v in cyc]
            V.append(_solve3(*on))
        return V, F, list(_P_PLANES)
    if kind == "MIZHAEV":
        F = [[v - 1 for v in (w[::-1] if f in _M_REVERSED else w)]
             for f, w in enumerate(_M_WALKS)]
        V = [tuple(Fraction(c) for c in p) for p in _M_VERTS]
        return V, F, list(_M_PLANES)
    if kind == "MIZHAEV_V1":
        F = [[v - 1 for v in cyc] for cyc in _V1_CYCLES]
        V = [tuple(Fraction(n, d) for n, d in p) for p in _V1_VERTS]
        planes = []
        for cyc in F:
            on = [p for p in _V1_PLANES
                  if all(_dot(p[:3], V[v]) == p[3] for v in cyc)]
            assert len(on) == 1, "a V1 face is not on exactly one plane"
            planes.append(on[0])
        return V, F, planes
    raise KeyError(kind)


def build(kind):
    """(V, F) in floats, in the published coordinates."""
    V, F, _planes = exact(kind)
    return [tuple(float(c) for c in p) for p in V], F


# --- shared-edge bookkeeping (any closed polyhedron) -------------------------

def shared_edges(F):
    """{(f, g): [(a, b), ...]} -- the edges each pair of faces f < g has in
    common.  Every edge is assumed to lie on exactly two faces."""
    owner = {}
    for f, cyc in enumerate(F):
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            owner.setdefault((min(a, b), max(a, b)), []).append(f)
    out = {}
    for e, fs in owner.items():
        if len(fs) == 2 and fs[0] != fs[1]:
            out.setdefault((min(fs), max(fs)), []).append(e)
    return out


def is_face_neighbourly(F):
    """True when every two faces share at least one edge."""
    n = len(F)
    return n > 1 and len(shared_edges(F)) == n * (n - 1) // 2


def doubled_edges(F):
    """The edges belonging to a pair of faces that shares more than one."""
    return sorted(e for es in shared_edges(F).values() if len(es) > 1
                  for e in es)


def surface_topology(F):
    """(edges, components, genus) of a closed orientable face list; the
    genus is the total over all components."""
    E = set()
    parent = {}

    def find(v):
        while parent.setdefault(v, v) != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for f in F:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            E.add((min(a, b), max(a, b)))
            parent[find(a)] = find(b)
    verts = {v for f in F for v in f}
    comps = len({find(v) for v in verts})
    chi = len(verts) - len(E) + len(F)
    return len(E), comps, comps - chi // 2


def summary_text(name, V, F, note=None):
    """A report line: counts, genus, and the two neighbourliness
    properties when they hold."""
    nE, comps, genus = surface_topology(F)
    text = "%s: V=%d E=%d F=%d" % (name, len(V), nE, len(F))
    if comps == 1:
        text += " (genus %d)" % genus
    else:
        text += " (%d separate surfaces of genus %d)" % (comps,
                                                          genus // comps)
    if is_face_neighbourly(F):
        twice = sum(1 for es in shared_edges(F).values() if len(es) > 1)
        text += "; every two faces share an edge"
        if twice:
            text += " (%d pairs share two)" % twice
    if comps == 1 and nE == len(V) * (len(V) - 1) // 2:
        text += "; every two vertices are joined by an edge"
    if note:
        text += "; " + note
    return text


def face_adjacency(F):
    """{face: [neighbouring faces]} across shared edges."""
    adj = {f: [] for f in range(len(F))}
    for f, g in shared_edges(F):
        adj[f].append(g)
        adj[g].append(f)
    return adj


# --- exact checks --------------------------------------------------------------

def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _newell(V, cyc):
    n = [0, 0, 0]
    for i in range(len(cyc)):
        p, q = V[cyc[i]], V[cyc[(i + 1) % len(cyc)]]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return tuple(n)


def _reflex_corners(V, cyc):
    """Vertices where the (outward, counter-clockwise) boundary turns
    clockwise."""
    n = _newell(V, cyc)
    k = len(cyc)
    return [cyc[i] for i in range(k)
            if _dot(_cross(_sub(V[cyc[i]], V[cyc[i - 1]]),
                           _sub(V[cyc[(i + 1) % k]], V[cyc[i]])), n) < 0]


def _in_closed_polygon(V, cyc, normal, x):
    """Is x (a point of the polygon's plane) inside it or on its boundary?"""
    k = len(cyc)
    for i in range(k):
        a, b = V[cyc[i]], V[cyc[(i + 1) % k]]
        if (_cross(_sub(x, a), _sub(b, a)) == (0, 0, 0)
                and _dot(_sub(x, a), _sub(x, b)) <= 0):
            return True
    drop = max(range(3), key=lambda i: abs(normal[i]))
    u, w = [i for i in range(3) if i != drop]
    inside = False
    for i in range(k):
        a, b = V[cyc[i]], V[cyc[(i + 1) % k]]
        if (a[w] > x[w]) != (b[w] > x[w]):
            if a[u] + (x[w] - a[w]) * (b[u] - a[u]) / (b[w] - a[w]) > x[u]:
                inside = not inside
    return inside


def _on_line(V, cyc, own, other, d, origin):
    """The closed polygon `cyc` cut by the plane `other`: a sorted list of
    closed intervals [lo, hi] (lo == hi for an isolated touching point),
    parametrised by t = x . d along the line where the two planes meet."""
    k = len(cyc)
    side = [_dot(other[:3], V[v]) - other[3] for v in cyc]
    ts = set()
    for i in range(k):
        j = (i + 1) % k
        if side[i] == 0:
            ts.add(_dot(V[cyc[i]], d))
        if side[i] * side[j] < 0:
            s = side[i] / (side[i] - side[j])
            p, q = V[cyc[i]], V[cyc[j]]
            ts.add(_dot(tuple(a + s * (b - a) for a, b in zip(p, q)), d))
    ts = sorted(ts)
    dd = _dot(d, d)
    out = []
    for t in ts:
        out.append([t, t])
    for t0, t1 in zip(ts, ts[1:]):
        mid = (t0 + t1) / 2
        x = tuple(origin[i] + d[i] * mid / dd for i in range(3))
        if _in_closed_polygon(V, cyc, own[:3], x):
            out.append([t0, t1])
    return _merge(out)


def _merge(ivs):
    out = []
    for lo, hi in sorted(ivs):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def _embedding_faults(V, F, planes):
    """Pairs of faces whose intersection is NOT exactly the edges they
    share.  Every face pair is tested on the line where their planes meet,
    in exact arithmetic, so there is no tolerance to tune."""
    shared = shared_edges(F)
    faults = []
    for f, g in combinations(range(len(F)), 2):
        d = _cross(planes[f][:3], planes[g][:3])
        assert d != (0, 0, 0), "parallel face planes"
        origin = _solve3(planes[f], planes[g], d + (0,))
        a = _on_line(V, F[f], planes[f], planes[g], d, origin)
        b = _on_line(V, F[g], planes[g], planes[f], d, origin)
        meet = _merge([[max(p[0], q[0]), min(p[1], q[1])]
                       for p in a for q in b
                       if max(p[0], q[0]) <= min(p[1], q[1])])
        want = _merge([sorted((_dot(V[p], d), _dot(V[q], d)))
                       for p, q in shared.get((f, g), [])])
        if meet != want:
            faults.append((f, g))
    return faults


def _signed_volume(V, F):
    vol = Fraction(0)
    for cyc in F:
        for i in range(1, len(cyc) - 1):
            vol += _det3(V[cyc[0]], V[cyc[i]], V[cyc[i + 1]])
    return vol / 6


def _automorphisms(F, mirror):
    """Number of automorphisms of the map: orientation-preserving ones, or
    (mirror=True) orientation-reversing ones.  A map automorphism is fixed
    by the image of one directed edge, so each candidate image is tried and
    propagated."""
    nxt, prv = {}, {}
    for cyc in F:
        k = len(cyc)
        for i in range(k):
            d = (cyc[i], cyc[(i + 1) % k])
            nxt[d] = (cyc[(i + 1) % k], cyc[(i + 2) % k])
            prv[d] = (cyc[i - 1], cyc[i])

    def flip(d):
        return (d[1], d[0])

    def step(d):
        # "next edge around the face" in the mirror-image map
        if not mirror:
            return nxt[d]
        return flip(prv[flip(d)])

    darts = list(nxt)
    d0 = darts[0]
    count = 0
    for target in darts:
        h = {d0: target}
        stack = [d0]
        ok = True
        while stack and ok:
            d = stack.pop()
            for nd, nt in ((nxt[d], step(h[d])), (flip(d), flip(h[d]))):
                if nd in h:
                    if h[nd] != nt:
                        ok = False
                        break
                else:
                    h[nd] = nt
                    stack.append(nd)
        if ok and len(h) == len(darts) and len(set(h.values())) == len(darts):
            count += 1
    return count


def _isomorphic(F1, F2, mirror):
    """Is there a map isomorphism from F1 onto F2 (orientation-reversing
    when mirror=True)?  Like _automorphisms: one dart's image fixes it."""
    def darts(F):
        nxt, prv = {}, {}
        for cyc in F:
            k = len(cyc)
            for i in range(k):
                d = (cyc[i], cyc[(i + 1) % k])
                nxt[d] = (cyc[(i + 1) % k], cyc[(i + 2) % k])
                prv[d] = (cyc[i - 1], cyc[i])
        return nxt, prv

    n1, _p1 = darts(F1)
    n2, p2 = darts(F2)
    if len(n1) != len(n2):
        return False

    def flip(d):
        return (d[1], d[0])

    def step(d):
        return flip(p2[flip(d)]) if mirror else n2[d]

    d0 = next(iter(n1))
    for target in n2:
        h = {d0: target}
        stack = [d0]
        ok = True
        while stack and ok:
            d = stack.pop()
            for nd, nt in ((n1[d], step(h[d])), (flip(d), flip(h[d]))):
                if nd in h:
                    if h[nd] != nt:
                        ok = False
                        break
                else:
                    h[nd] = nt
                    stack.append(nd)
        if ok and len(h) == len(n1) and len(set(h.values())) == len(n1):
            return True
    return False


def _doubled_cycles(F):
    """Sizes of the connected pieces of the graph joining two faces when
    they share two edges; each piece must be a plain cycle."""
    adj = {}
    for (f, g), es in shared_edges(F).items():
        if len(es) > 1:
            assert len(es) == 2
            adj.setdefault(f, set()).add(g)
            adj.setdefault(g, set()).add(f)
    assert all(len(nb) == 2 for nb in adj.values()), "not a union of cycles"
    seen, sizes = set(), []
    for start in sorted(adj):
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            v = stack.pop()
            if v not in comp:
                comp.add(v)
                stack.extend(adj[v] - comp)
        seen |= comp
        sizes.append(len(comp))
    return sorted(sizes)


def _check(kind, singles=20, doubles=8):
    V, F, planes = exact(kind)
    nV, nF = len(V), len(F)

    # planes in general position; every vertex on exactly its three faces
    for p, q in combinations(planes, 2):
        assert _cross(p[:3], q[:3]) != (0, 0, 0), "parallel planes"
    pts = [_solve3(*t) for t in combinations(planes, 3)]
    assert None not in pts and len(set(pts)) == len(pts), \
        "four planes through one point"
    for v in range(nV):
        mine = {f for f, cyc in enumerate(F) if v in cyc}
        assert len(mine) == 3, (kind, v, mine)
        for f, (a, b, c, d) in enumerate(planes):
            on = a * V[v][0] + b * V[v][1] + c * V[v][2] == d
            assert on == (f in mine), (kind, v, f)
        assert _solve3(*(planes[f] for f in sorted(mine))) == V[v]

    # closed and consistently oriented: each directed edge once, with its
    # reverse; three edges and a single ring of three faces at every vertex
    darts = [(cyc[i], cyc[(i + 1) % 9]) for cyc in F for i in range(9)]
    assert all(len(cyc) == 9 and len(set(cyc)) == 9 for cyc in F)
    assert len(set(darts)) == len(darts) == 72
    assert all((b, a) in set(darts) for a, b in darts)
    nE = len({frozenset(d) for d in darts})
    deg = [0] * nV
    for a, _b in darts:
        deg[a] += 1
    assert set(deg) == {3}, "vertex not three-valent"
    chi = nV - nE + nF
    assert (nV, nE, nF, chi) == (24, 36, 8, -4), (nV, nE, nF, chi)

    # which pairs meet once and twice (all 28, for the face-neighbourly
    # ones), and every doubled pair's two edges on one line
    sh = shared_edges(F)
    assert is_face_neighbourly(F) == (singles + doubles == 28)
    mult = sorted(len(es) for es in sh.values())
    assert mult == [1] * singles + [2] * doubles, mult
    for es in sh.values():
        if len(es) == 2:
            (a, b), (c, d) = es
            u = _sub(V[b], V[a])
            assert _cross(u, _sub(V[c], V[a])) == (0, 0, 0)
            assert _cross(u, _sub(V[d], V[a])) == (0, 0, 0)
    assert len(doubled_edges(F)) == 2 * doubles

    # outward orientation, simple non-convex nonagons, embedded
    vol = _signed_volume(V, F)
    assert vol > 0, "faces not oriented outwards"
    reflex = [len(_reflex_corners(V, cyc)) for cyc in F]
    assert sorted(reflex) == [2, 2, 2, 2, 3, 3, 3, 3], reflex
    faults = _embedding_faults(V, F, planes)
    assert not faults, ("faces meet off their shared edges", kind, faults)
    return V, F, planes, vol, reflex


def _selftest():
    # --- the Rost-Vigh polyhedron P ---
    V, F, planes, vol, reflex = _check("ROST_VIGH")
    # derived vertices against the published Table 2, orbit by orbit
    for k, rep in enumerate(_P_TABLE2):
        x, y, z = (Fraction(n, d) for n, d in rep)
        orbit = [(x, y, z), (-x, -y, z), (x, -y, -z), (-x, y, -z)]
        assert V[4 * k:4 * k + 4] == orbit, ("Table 2 mismatch", k)
    assert vol == Fraction(351105133, 18170460), vol
    assert reflex == [3, 3, 3, 3, 2, 2, 2, 2], reflex
    assert sorted(v + 1 for v in _reflex_corners(V, F[0])) == [7, 9, 13]
    assert sorted(v + 1 for v in _reflex_corners(V, F[4])) == [20, 24]
    # D2: the three half-turns carry vertices to vertices, faces to faces
    index = {p: i for i, p in enumerate(V)}
    cycles = {frozenset(_rotations(c)) for c in F}
    for sx, sy, sz in ((-1, -1, 1), (1, -1, -1), (-1, 1, -1)):
        perm = [index[(sx * p[0], sy * p[1], sz * p[2])] for p in V]
        for c in F:
            assert frozenset(_rotations([perm[v] for v in c])) in cycles
    # the convex hull is the tetrahedron on the four spike tips
    tet = [V[0], V[1], V[2], V[3]]
    for p in V:
        for i in range(4):
            face = [tet[j] for j in range(4) if j != i]
            nrm = _cross(_sub(face[1], face[0]), _sub(face[2], face[0]))
            assert (_dot(nrm, _sub(p, face[0]))
                    * _dot(nrm, _sub(tet[i], face[0]))) >= 0
    assert _doubled_cycles(F) == [4, 4]
    assert (_automorphisms(F, False), _automorphisms(F, True)) == (8, 0)
    print("Rost-Vigh P  V=24 E=36 F=8 genus 3; 20 pairs share one edge, 8 "
          "share two (two 4-cycles); volume 351105133/18170460; embedded; "
          "D2; 8 map automorphisms, none orientation-reversing")

    # --- Mizhaev's polyhedron M ---
    V, F, planes, vol, reflex = _check("MIZHAEV")
    assert vol == 4455360, vol
    # faces F3..F6 carry three reflex corners, the rest two
    assert reflex == [2, 2, 3, 3, 3, 3, 2, 2], reflex
    # the rotoreflection T(x, y, z) = (y, -x, -z)
    index = {p: i for i, p in enumerate(V)}
    perm = [index[(p[1], -p[0], -p[2])] for p in V]
    cycles = {frozenset(_rotations(c)) for c in F}
    for c in F:        # T reverses orientation, so a cycle lands reversed
        assert frozenset(_rotations([perm[v] for v in c][::-1])) in cycles
    assert _doubled_cycles(F) == [8]
    assert (_automorphisms(F, False), _automorphisms(F, True)) == (2, 2)
    print("Mizhaev M    V=24 E=36 F=8 genus 3; 20 pairs share one edge, 8 "
          "share two (one 8-cycle); volume 4455360; embedded; C4 "
          "rotoreflection; 4 map automorphisms, two orientation-reversing")

    # --- Mizhaev's 2020 variant V1: the same counts, not face-neighbourly ---
    V, F, planes, vol, reflex = _check("MIZHAEV_V1", singles=12, doubles=12)
    # the cycles are the ones the printed vertex sets force
    assert [sorted(v + 1 for v in c) for c in F] == \
        [sorted(s) for s in _V1_FACE_SETS]
    sh = shared_edges(F)
    apart = [(f + 1, g + 1) for f, g in combinations(range(8), 2)
             if (f, g) not in sh]
    assert apart == [(1, 2), (3, 8), (4, 6), (5, 7)], apart
    twice = {f: 0 for f in range(8)}
    for (f, g), es in sh.items():
        if len(es) == 2:
            twice[f] += 1
            twice[g] += 1
    assert set(twice.values()) == {3}       # three doubled partners each
    assert sorted(reflex) == [2, 2, 2, 2, 3, 3, 3, 3]
    # the same rotoreflection T as M
    index = {p: i for i, p in enumerate(V)}
    perm = [index[(p[1], -p[0], -p[2])] for p in V]
    cycles = {frozenset(_rotations(c)) for c in F}
    for c in F:
        assert frozenset(_rotations([perm[v] for v in c][::-1])) in cycles
    assert (_automorphisms(F, False), _automorphisms(F, True)) == (4, 4)
    # and a different map from both face-neighbourly solids
    for other in ("MIZHAEV", "ROST_VIGH"):
        Fo = exact(other)[1]
        assert not _isomorphic(F, Fo, False) and \
            not _isomorphic(F, Fo, True), other
    assert _isomorphic(exact("MIZHAEV")[1], exact("MIZHAEV")[1], True)
    print("Mizhaev V1   V=24 E=36 F=8 genus 3; 12 pairs share one edge, 12 "
          "share two, 4 pairs never meet; embedded; C4 rotoreflection; "
          "not isomorphic to M or P")

    # the float build matches the exact data
    for kind in NEIGHBOURLY:
        Vf, Ff = build(kind)
        assert len(Vf) == 24 and len(Ff) == 8
    print("RESULT: OK")


def _rotations(cyc):
    cyc = tuple(cyc)
    return [cyc[i:] + cyc[:i] for i in range(len(cyc))]
