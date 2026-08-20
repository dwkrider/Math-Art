
# Uniform Polyhedra Generator for Blender
#
# Builds the uniform polyhedra by Wythoff's kaleidoscopic construction:
# each solid comes from a Schwarz triangle (p q r) -- a spherical triangle
# with angles pi/p, pi/q, pi/r whose three sides are mirrors.  The three
# reflections generate a finite group (tetrahedral, octahedral or
# icosahedral).  A generator point is placed by the Wythoff symbol (the
# point equidistant from the "active" mirrors, on the others), its orbit
# under the group gives the vertices, and each face is the cyclic or
# dihedral orbit of that point about a rotation axis -- a {n/d} polygon,
# a star polygon when d > 1.  This yields the convex uniform solids, the
# regular star (Kepler-Poinsot) polyhedra, and the non-convex star
# uniforms including the hemipolyhedra (faces through the centre).
#
# This covers all 75 uniform polyhedra: the non-snub cases (Wythoff
# symbols with the bar not first) are built procedurally by the
# construction above; the 11 snubs and the great dirhombicosidodecahedron
# (whose generator point needs a transcendental snub solve) are built
# from stored vertex coordinates evaluated from their published exact
# constructions, with each face's winding density detected for rendering.
#
# Star faces are rendered by fanning each {n/d} polygon from its centre,
# so the star outline shows as a solid.
#
# One further object sits outside that set. Relaxing "two faces to an
# edge" to "an even number" admits exactly one more uniform figure --
# Skilling's great disnub dirhombidodecahedron, four faces deep along
# half of its edges -- and it is built here from U75 by Hart's XOR with
# the compound of twenty octahedra (see build_skilling).
#
# References:
# - W. A. Wythoff, "A relation between the polytopes of the C600-family",
#   Koninklijke Akademie van Wetenschappen te Amsterdam (1918).
# - H. S. M. Coxeter, M. S. Longuet-Higgins, J. C. P. Miller, "Uniform
#   polyhedra", Phil. Trans. Royal Soc. A 246 (1954), 401-450.
# - S. P. Skilling, "The complete set of uniform polyhedra", Phil. Trans.
#   Royal Soc. A 278 (1975) -- completeness, and the great disnub
#   dirhombidodecahedron that the relaxed definition admits.
# - George W. Hart, "Great Disnub Dirhombidodecahedron", Virtual Polyhedra
#   (georgehart.com/virtual-polyhedra/), for the exclusive-OR construction
#   used here.
# - Zvi Har'El, "Uniform Solution for Uniform Polyhedra", Geometriae
#   Dedicata 47 (1993), 57-110 (the construction algorithm followed here).
# - Magnus Wenninger, "Polyhedron Models", Cambridge (1971).
# - Numbering (U1-U75) after Roman Maeder's Kaleido / Mathematica package.

bl_info = {
    "name": "Uniform Polyhedra",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "All 75 uniform polyhedra (Wythoff construction): "
                   "convex, Kepler-Poinsot, hemipolyhedra and star "
                   "uniforms, with duals and star prisms; plus "
                   "Skilling's figure",
    "category": "Add Mesh",
}

import math
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None

try:                                    # the 12 snub / special solids
    from .polyhedra import _uniform_snub_data as _snub_data
except ImportError:
    try:
        from polyhedra import _uniform_snub_data as _snub_data
    except ImportError:
        _snub_data = None


# u, name, wythoff, pqr, symmetry, V, E, F  (all 75; snubs/U75 flagged
# by a bar-first Wythoff symbol or a 4-entry pqr -- not built yet)
UNIFORMS = [
    (1, "Tetrahedron", "3 | 2 3", ["3", "2", "3"], 4, 6, 4),
    (2, "Truncated Tetrahedron", "2 3 | 3", ["2", "3", "3"], 12, 18, 8),
    (3, "Octahemioctahedron", "3/2 3 | 3", ["3/2", "3", "3"], 12, 24, 12),
    (4, "Tetrahemihexahedron", "3/2 3 | 2", ["3/2", "3", "2"], 6, 12, 7),
    (5, "Octahedron", "4 | 2 3", ["4", "2", "3"], 6, 12, 8),
    (6, "Cube", "3 | 2 4", ["3", "2", "4"], 8, 12, 6),
    (7, "Cuboctahedron", "2 | 3 4", ["2", "3", "4"], 12, 24, 14),
    (8, "Truncated Octahedron", "2 4 | 3", ["2", "4", "3"], 24, 36, 14),
    (9, "Truncated Cube", "2 3 | 4", ["2", "3", "4"], 24, 36, 14),
    (10, "Rhombicuboctahedron", "3 4 | 2", ["3", "4", "2"], 24, 48, 26),
    (11, "Truncated Cuboctahedron", "2 3 4 |", ["2", "3", "4"], 48, 72, 26),
    (12, "Snub Cube", "| 2 3 4", ["2", "3", "4"], 24, 60, 38),
    (13, "Small Cubicuboctahedron", "3/2 4 | 4", ["3/2", "4", "4"],
     24, 48, 20),
    (14, "Great Cubicuboctahedron", "3 4 | 4/3", ["3", "4", "4/3"],
     24, 48, 20),
    (15, "Cubohemioctahedron", "4/3 4 | 3", ["4/3", "4", "3"], 12, 24, 10),
    (16, "Cubitruncated Cuboctahedron", "4/3 3 4 |", ["4/3", "3", "4"],
     48, 72, 20),
    (17, "Nonconvex Great Rhombicuboctahedron", "3/2 4 | 2",
     ["3/2", "4", "2"], 24, 48, 26),
    (18, "Small Rhombihexahedron", "3/2 2 4 |", ["3/2", "2", "4"],
     24, 48, 18),
    (19, "Stellated Truncated Hexahedron", "2 3 | 4/3", ["2", "3", "4/3"],
     24, 36, 14),
    (20, "Great Truncated Cuboctahedron", "4/3 2 3 |", ["4/3", "2", "3"],
     48, 72, 26),
    (21, "Great Rhombihexahedron", "4/3 3/2 2 |", ["4/3", "3/2", "2"],
     24, 48, 18),
    (22, "Icosahedron", "5 | 2 3", ["5", "2", "3"], 12, 30, 20),
    (23, "Dodecahedron", "3 | 2 5", ["3", "2", "5"], 20, 30, 12),
    (24, "Icosidodecahedron", "2 | 3 5", ["2", "3", "5"], 30, 60, 32),
    (25, "Truncated Icosahedron", "2 5 | 3", ["2", "5", "3"], 60, 90, 32),
    (26, "Truncated Dodecahedron", "2 3 | 5", ["2", "3", "5"], 60, 90, 32),
    (27, "Rhombicosidodecahedron", "3 5 | 2", ["3", "5", "2"], 60, 120, 62),
    (28, "Truncated Icosidodecahedron", "2 3 5 |", ["2", "3", "5"],
     120, 180, 62),
    (29, "Snub Dodecahedron", "| 2 3 5", ["2", "3", "5"], 60, 150, 92),
    (30, "Small Ditrigonal Icosidodecahedron", "3 | 5/2 3",
     ["3", "5/2", "3"], 20, 60, 32),
    (31, "Small Icosicosidodecahedron", "5/2 3 | 3", ["5/2", "3", "3"],
     60, 120, 52),
    (32, "Small Snub Icosicosidodecahedron", "| 5/2 3 3",
     ["5/2", "3", "3"], 60, 180, 112),
    (33, "Small Dodecicosidodecahedron", "3/2 5 | 5", ["3/2", "5", "5"],
     60, 120, 44),
    (34, "Small Stellated Dodecahedron", "5 | 2 5/2", ["5", "2", "5/2"],
     12, 30, 12),
    (35, "Great Dodecahedron", "5/2 | 2 5", ["5/2", "2", "5"], 12, 30, 12),
    (36, "Dodecadodecahedron", "2 | 5 5/2", ["2", "5", "5/2"], 30, 60, 24),
    (37, "Truncated Great Dodecahedron", "2 5/2 | 5", ["2", "5/2", "5"],
     60, 90, 24),
    (38, "Rhombidodecadodecahedron", "5/2 5 | 2", ["5/2", "5", "2"],
     60, 120, 54),
    (39, "Small Rhombidodecahedron", "2 5/2 5 |", ["2", "5/2", "5"],
     60, 120, 42),
    (40, "Snub Dodecadodecahedron", "| 2 5/2 5", ["2", "5/2", "5"],
     60, 150, 84),
    (41, "Ditrigonal Dodecadodecahedron", "3 | 5/3 5", ["3", "5/3", "5"],
     20, 60, 24),
    (42, "Great Ditrigonal Dodecicosidodecahedron", "3 5 | 5/3",
     ["3", "5", "5/3"], 60, 120, 44),
    (43, "Small Ditrigonal Dodecicosidodecahedron", "5/3 3 | 5",
     ["5/3", "3", "5"], 60, 120, 44),
    (44, "Icosidodecadodecahedron", "5/3 5 | 3", ["5/3", "5", "3"],
     60, 120, 44),
    (45, "Icositruncated Dodecadodecahedron", "3 5 5/3 |",
     ["3", "5", "5/3"], 120, 180, 44),
    (46, "Snub Icosidodecadodecahedron", "| 5/3 3 5", ["5/3", "3", "5"],
     60, 180, 104),
    (47, "Great Ditrigonal Icosidodecahedron", "3/2 | 3 5",
     ["3/2", "3", "5"], 20, 60, 32),
    (48, "Great Icosicosidodecahedron", "3/2 5 | 3", ["3/2", "5", "3"],
     60, 120, 52),
    (49, "Small Icosihemidodecahedron", "3/2 3 | 5", ["3/2", "3", "5"],
     30, 60, 26),
    (50, "Small Dodecicosahedron", "3/2 3 5 |", ["3/2", "3", "5"],
     60, 120, 32),
    (51, "Small Dodecahemidodecahedron", "5/4 5 | 5", ["5/4", "5", "5"],
     30, 60, 18),
    (52, "Great Stellated Dodecahedron", "3 | 2 5/2", ["3", "2", "5/2"],
     20, 30, 12),
    (53, "Great Icosahedron", "5/2 | 2 3", ["5/2", "2", "3"], 12, 30, 20),
    (54, "Great Icosidodecahedron", "2 | 3 5/2", ["2", "3", "5/2"],
     30, 60, 32),
    (55, "Truncated Great Icosahedron", "2 5/2 | 3", ["2", "5/2", "3"],
     60, 90, 32),
    (56, "Rhombicosahedron", "2 5/2 3 |", ["2", "5/2", "3"], 60, 120, 50),
    (57, "Great Snub Icosidodecahedron", "| 2 5/2 3", ["2", "5/2", "3"],
     60, 150, 92),
    (58, "Small Stellated Truncated Dodecahedron", "2 5 | 5/3",
     ["2", "5", "5/3"], 60, 90, 24),
    (59, "Truncated Dodecadodecahedron", "2 5 5/3 |", ["2", "5", "5/3"],
     120, 180, 54),
    (60, "Inverted Snub Dodecadodecahedron", "| 5/3 2 5",
     ["5/3", "2", "5"], 60, 150, 84),
    (61, "Great Dodecicosidodecahedron", "5/2 3 | 5/3",
     ["5/2", "3", "5/3"], 60, 120, 44),
    (62, "Small Dodecahemicosahedron", "5/3 5/2 | 3", ["5/3", "5/2", "3"],
     30, 60, 22),
    (63, "Great Dodecicosahedron", "5/3 5/2 3 |", ["5/3", "5/2", "3"],
     60, 120, 32),
    (64, "Great Snub Dodecicosidodecahedron", "| 5/3 5/2 3",
     ["5/3", "5/2", "3"], 60, 180, 104),
    (65, "Great Dodecahemicosahedron", "5/4 5 | 3", ["5/4", "5", "3"],
     30, 60, 22),
    (66, "Great Stellated Truncated Dodecahedron", "2 3 | 5/3",
     ["2", "3", "5/3"], 60, 90, 32),
    (67, "Nonconvex Great Rhombicosidodecahedron", "5/3 3 | 2",
     ["5/3", "3", "2"], 60, 120, 62),
    (68, "Great Truncated Icosidodecahedron", "2 3 5/3 |",
     ["2", "3", "5/3"], 120, 180, 62),
    (69, "Great Inverted Snub Icosidodecahedron", "| 5/3 2 3",
     ["5/3", "2", "3"], 60, 150, 92),
    (70, "Great Dodecahemidodecahedron", "5/3 5/2 | 5/3",
     ["5/3", "5/2", "5/3"], 30, 60, 18),
    (71, "Great Icosihemidodecahedron", "3/2 3 | 5/3", ["3/2", "3", "5/3"],
     30, 60, 26),
    (72, "Small Retrosnub Icosicosidodecahedron", "| 3/2 3/2 5/2",
     ["3/2", "3/2", "5/2"], 60, 180, 112),
    (73, "Great Rhombidodecahedron", "3/2 5/3 2 |", ["3/2", "5/3", "2"],
     60, 120, 42),
    (74, "Great Retrosnub Icosidodecahedron", "| 2 3/2 5/3",
     ["2", "3/2", "5/3"], 60, 150, 92),
    (75, "Great Dirhombicosidodecahedron", "| 3/2 5/3 3 5/2",
     ["3/2", "5/3", "3", "5/2"], 60, 240, 124),
]

try:                                  # inside the math_art package
    from .polyhedra.fit import fit_cube as _fit_cube
except ImportError:                   # flat import (test runner)
    from polyhedra.fit import fit_cube as _fit_cube


def _frac(x):
    s = str(x)
    if '/' in s:
        p, q = s.split('/')
        return float(p) / float(q), (int(p), int(q))
    return float(x), (int(float(x)), 1)


def _mirrors(pqr):
    o = [_frac(x)[0] for x in pqr]
    n0 = np.array([1.0, 0.0, 0.0])
    c01 = -math.cos(math.pi / o[2])
    n1 = np.array([c01, math.sqrt(max(0.0, 1 - c01 * c01)), 0.0])
    c02 = -math.cos(math.pi / o[1])
    c12 = -math.cos(math.pi / o[0])
    x2 = c02
    y2 = (c12 - n1[0] * x2) / n1[1]
    z2 = math.sqrt(abs(1 - x2 * x2 - y2 * y2))
    return [n0, n1, np.array([x2, y2, z2])]


def _key(v, p=5):
    return (round(float(v[0]), p), round(float(v[1]), p),
            round(float(v[2]), p))


def _mkey(M, p=4):
    return tuple(round(float(x), p) for x in M.ravel())


def _refl_group(ns, maxn=4000):
    refl = [np.eye(3) - 2 * np.outer(n, n) for n in ns]
    seen = {_mkey(np.eye(3)): np.eye(3)}
    frontier = [np.eye(3)]
    while frontier:
        nf = []
        for M in frontier:
            for r in refl:
                N = r @ M
                k = _mkey(N)
                if k not in seen:
                    seen[k] = N
                    nf.append(N)
                    if len(seen) > maxn:
                        raise RuntimeError("symmetry group too large")
        frontier = nf
    return list(seen.values())


def _orbit(P, G):
    d = {}
    for M in G:
        q = M @ P
        d.setdefault(_key(q), q)
    return list(d.values())


def _rot(ax, a):
    x, y, z = ax
    c = math.cos(a)
    s = math.sin(a)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _ring_mask(wythoff):
    toks = wythoff.split()
    bar = toks.index('|')
    mask = []
    idx = 0
    for t in toks:
        if t == '|':
            continue
        mask.append(1.0 if idx < bar else 0.0)
        idx += 1
    return mask


def build_uniform(wythoff, pqr):
    """(verts, faces) where each face is (vertex-index list, density d).
    Non-snub Wythoff symbols only."""
    if np is None:
        raise RuntimeError("uniform polyhedra need NumPy")
    ns = _mirrors(pqr)
    G = _refl_group(ns)
    rm = _ring_mask(wythoff)
    P = np.linalg.solve(np.array([ns[0], ns[1], ns[2]]),
                        np.array(rm, dtype=float))
    P = P / np.linalg.norm(P)
    verts = _orbit(P, G)
    index = {_key(v): i for i, v in enumerate(verts)}

    def vidx(q):
        return index.get(_key(q))
    faces = []
    seen = set()
    for i in range(3):
        j, k = [t for t in range(3) if t != i]
        rc = int(rm[j] + rm[k])
        if rc == 0:
            continue
        _, (num, den) = _frac(pqr[i])
        Vi = np.cross(ns[j], ns[k])
        Vi = Vi / np.linalg.norm(Vi)
        if rc == 2:
            reflj = np.eye(3) - 2 * np.outer(ns[j], ns[j])
            raw = []
            for kk in range(num):
                R = _rot(Vi, 2 * math.pi * kk / num)
                raw.append(R @ P)
                raw.append(R @ (reflj @ P))
            nv = 2 * num
        else:
            raw = [_rot(Vi, 2 * math.pi * kk / num) @ P for kk in range(num)]
            nv = num
        if nv < 3:
            continue
        u = raw[0] - np.dot(raw[0], Vi) * Vi
        u = u / np.linalg.norm(u)
        w = np.cross(Vi, u)

        def az(t):
            d = raw[t] - np.dot(raw[t], Vi) * Vi
            return math.atan2(np.dot(d, w), np.dot(d, u))
        order = sorted(range(len(raw)), key=az)
        rawo = [raw[t] for t in order]
        seed = [rawo[(den * t) % nv] for t in range(nv)]
        for M in G:
            f = [vidx(M @ p) for p in seed]
            if None in f or len(set(f)) != len(f):
                continue
            key = frozenset(f)
            if key not in seen:
                seen.add(key)
                faces.append((f, den))
    return verts, faces


def _face_density(V, f):
    """Winding number of a face about its centre (1 = convex polygon,
    2 = pentagram, ...) -- used to tell pentagons from pentagrams."""
    pts = [V[i] for i in f]
    m = len(f)
    c = [sum(p[k] for p in pts) / m for k in range(3)]
    nx = [0.0, 0.0, 0.0]
    for i in range(m):
        p, q = pts[i], pts[(i + 1) % m]
        nx[0] += (p[1] - q[1]) * (p[2] + q[2])
        nx[1] += (p[2] - q[2]) * (p[0] + q[0])
        nx[2] += (p[0] - q[0]) * (p[1] + q[1])
    ln = math.sqrt(sum(x * x for x in nx)) or 1.0
    nx = [x / ln for x in nx]
    u = [pts[0][k] - c[k] for k in range(3)]
    dp = sum(u[k] * nx[k] for k in range(3))
    u = [u[k] - dp * nx[k] for k in range(3)]
    lu = math.sqrt(sum(x * x for x in u)) or 1.0
    u = [x / lu for x in u]
    w = [nx[1] * u[2] - nx[2] * u[1], nx[2] * u[0] - nx[0] * u[2],
         nx[0] * u[1] - nx[1] * u[0]]
    ang = [math.atan2(sum((p[k] - c[k]) * w[k] for k in range(3)),
                      sum((p[k] - c[k]) * u[k] for k in range(3)))
           for p in pts]
    tot = 0.0
    for i in range(m):
        da = ang[(i + 1) % m] - ang[i]
        while da <= -math.pi:
            da += 2 * math.pi
        while da > math.pi:
            da -= 2 * math.pi
        tot += da
    return max(1, round(abs(tot) / (2 * math.pi)))


def build_snub(u, scale=1.0):
    """The snub / special uniform polyhedra (U12, U29 and the star snubs
    up to the great dirhombicosidodecahedron) from stored coordinates;
    each face is tagged with its winding density for star rendering."""
    if _snub_data is None:
        raise RuntimeError("snub coordinate data not available")
    S = _snub_data.SNUBS[u]
    V = [tuple(c * scale for c in v) for v in S["V"]]
    faces = [(list(f), _face_density(S["V"], f)) for f in S["F"]]
    return V, faces


# --------------------------------------------------------------------------
# Skilling's figure -- the one beyond the seventy-five
# --------------------------------------------------------------------------
# Relax "exactly two faces meet at each edge" to "an even number do", and
# Skilling's computer search found exactly ONE new uniform object: the
# great disnub dirhombidodecahedron, with four faces along some edges.
# It is therefore not one of the 75, and it is the only thing the wider
# definition adds.
#
# Hart's construction is an exclusive-OR of the great dirhombicosi-
# dodecahedron (U75) with the compound of twenty octahedra, and the
# compound never has to be built separately: U75's own sixty vertices
# already carry it.  They form thirty antipodal diameters, and the twenty
# triples of MUTUALLY ORTHOGONAL diameters are exactly the twenty
# octahedra -- each diameter lying in two of them.  Every one of U75's
# forty triangles turns out to be one of the compound's 160, so the XOR
# cancels them and leaves 120 triangles, which with U75's 60 squares and
# 24 pentagrams gives 204 faces on 60 vertices and 240 edges.
#
# The four-faces-to-an-edge property is why this figure needs its own
# check: a self-test asserting every edge is used exactly twice would
# reject a correct build, so the assertion is EVEN multiplicity.

def _skilling_octahedra(V, tol=1e-5):
    """The twenty octahedra hiding in U75's vertex set, as index triples
    of antipodal diameters."""
    diam = []
    taken = set()
    for i, v in enumerate(V):
        if i in taken:
            continue
        for j in range(i + 1, len(V)):
            if j in taken:
                continue
            if all(abs(v[k] + V[j][k]) < tol for k in range(3)):
                taken.add(i)
                taken.add(j)
                diam.append((i, j))
                break
    out = []
    for a in range(len(diam)):
        for b in range(a + 1, len(diam)):
            for c in range(b + 1, len(diam)):
                u, w, x = V[diam[a][0]], V[diam[b][0]], V[diam[c][0]]
                if (abs(sum(u[k] * w[k] for k in range(3))) < tol
                        and abs(sum(u[k] * x[k] for k in range(3))) < tol
                        and abs(sum(w[k] * x[k] for k in range(3))) < tol):
                    out.append((diam[a], diam[b], diam[c]))
    return out


def build_skilling(scale=1.0):
    """The great disnub dirhombidodecahedron (Skilling's figure)."""
    if _snub_data is None:
        raise RuntimeError("snub coordinate data not available")
    S = _snub_data.SNUBS[75]
    V = [tuple(c * scale for c in v) for v in S["V"]]
    tri = set()
    for da, db, dc in _skilling_octahedra(S["V"]):
        for ia in da:
            for ib in db:
                for ic in dc:
                    tri.add(frozenset((ia, ib, ic)))
    drop = {frozenset(f) for f in S["F"] if len(f) == 3}
    faces = [(list(t), 1) for t in sorted(tri - drop, key=sorted)]
    faces += [(list(f), _face_density(S["V"], f))
              for f in S["F"] if len(f) != 3]
    return V, faces


def build_dual(V, faces, big=6.0):
    """Dual by polar reciprocation about the unit sphere: one dual vertex
    per face (its pole), one dual face per original vertex (the poles of
    the faces round that vertex, in cyclic order).  Faces through the
    centre (hemipolyhedra) have poles at infinity -- rendered truncated at
    radius `big`.  Returns (verts, faces-with-density)."""
    if np is None:
        raise RuntimeError("duals need NumPy")
    F = [f for f, _d in faces]
    poles = []
    inf_dirs = []                        # faces through the centre
    for f in F:
        pts = [np.array(V[i], float) for i in f]
        c = sum(pts) / len(pts)
        n = np.zeros(3)
        for i in range(len(f)):
            n += np.cross(pts[i] - c, pts[(i + 1) % len(f)] - c)
        n = n / (np.linalg.norm(n) or 1.0)
        k = float(np.dot(n, pts[0]))
        if abs(k) < 1e-6:                # pole at infinity -- fix up later
            inf_dirs.append(len(poles))
            poles.append(np.array(n))
        else:
            poles.append(np.array(n / k))
    # place the "infinite" hemipolyhedron vertices a bounded distance out,
    # relative to the finite body, so the dual still fits a 2 m cube
    finite = [p for i, p in enumerate(poles) if i not in set(inf_dirs)]
    rf = max((float(np.linalg.norm(p)) for p in finite), default=1.0)
    for i in inf_dirs:
        poles[i] = poles[i] * (big if big < rf else 1.6 * rf)
    # the reciprocation centre is the origin; scale to fit a 2 m cube
    mx = max((abs(float(c)) for p in poles for c in p), default=1.0)
    poles = [tuple(p / mx) for p in poles]
    vf = defaultdict(list)
    for fi, f in enumerate(F):
        for v in f:
            vf[v].append(fi)
    dfaces = []
    for v in range(len(V)):
        fs = vf[v]
        fe = {}
        for fi in fs:
            f = F[fi]
            i = f.index(v)
            fe[fi] = [frozenset((v, f[(i - 1) % len(f)])),
                      frozenset((v, f[(i + 1) % len(f)]))]
        e2f = defaultdict(list)
        for fi in fs:
            for e in fe[fi]:
                e2f[e].append(fi)
        order = [fs[0]]
        cur = fs[0]
        while len(order) < len(fs):
            nxt = None
            for e in fe[cur]:
                for g in e2f[e]:
                    if g != cur and g not in order:
                        nxt = g
                        break
                if nxt is not None:
                    break
            if nxt is None:
                break
            order.append(nxt)
            cur = nxt
        if len(order) >= 3:
            dfaces.append(order)
    out = [(df, _face_density(poles, df)) for df in dfaces]
    return poles, out


def valid_star_step(p, q):
    """Coerce (p, q) to a valid star-polygon step: 1 <= q < p/2, coprime
    with p ({p/q} and {p/(p-q)} are the same star, so q is folded into
    range; a non-coprime q is nudged to the nearest coprime, falling back
    to 1 = a convex prism)."""
    q = max(1, min(int(q), p - 1))
    if q > p / 2:                        # {p/q} == {p/(p-q)}
        q = p - q
    q = max(1, q)
    if math.gcd(p, q) != 1:
        for delta in range(1, p):
            for cand in (q - delta, q + delta):
                if 1 <= cand <= p // 2 and math.gcd(p, cand) == 1:
                    return cand
        return 1
    return q


def build_star_prism(p, q, scale=1.0):
    """Uniform {p/q} star prism: two star polygon bases joined by squares
    (q = 1 gives an ordinary convex prism).  (p, q) is coerced to a valid
    star polygon.  Returns (verts, faces) with each face (indices,
    density)."""
    q = valid_star_step(p, q)
    R = 1.0 / (2 * math.sin(q * math.pi / p)) * scale
    top = [(R * math.cos(2 * math.pi * k / p),
            R * math.sin(2 * math.pi * k / p), 0.5 * scale)
           for k in range(p)]
    bot = [(x, y, -z) for (x, y, z) in top]
    verts = top + bot
    faces = []
    for k in range(p):                       # lateral squares on star edges
        a, b = k, (k + q) % p
        faces.append(([a, b, p + b, p + a], 1))
    faces.append(([(q * k) % p for k in range(p)], q))          # top star
    faces.append(([p + ((q * k) % p) for k in range(p)][::-1], q))  # bottom
    return verts, faces


def build_star_antiprism(p, q, scale=1.0):
    """Uniform {p/q} star antiprism: two star-polygon bases (bottom rotated by
    pi/p) joined by 2p lateral triangles.  The lateral-triangle apex is the
    perpendicular-bisector-plane vertex (equal lateral edges), and the height
    solves the uniform condition (lateral edge = base edge) when real, else
    falls back to a valid crossed/retrograde form.  V=2p, E=4p, F=2p+2."""
    q = valid_star_step(p, q)
    R = scale / (2.0 * math.sin(q * math.pi / p))
    if p % 2 == 1:                                   # apex shift s: 2s == q-1
        s = ((q - 1) * ((p + 1) // 2)) % p           # (mod p)
    else:
        s = (q - 1) // 2                             # q odd when p even
    t = (q - s) % p
    delta = math.pi * (2 * s + 1) / p
    disc = (math.cos(delta) - math.cos(2 * q * math.pi / p)) / 2.0
    h = R * math.sqrt(disc) if disc > 1e-12 else 0.5 * scale
    top = [(R * math.cos(2 * math.pi * k / p),
            R * math.sin(2 * math.pi * k / p), h) for k in range(p)]
    bot = [(R * math.cos(2 * math.pi * k / p + math.pi / p),
            R * math.sin(2 * math.pi * k / p + math.pi / p), -h)
           for k in range(p)]
    verts = top + bot                                # T_k = k, B_k = p + k
    faces = []
    for k in range(p):
        faces.append(([k, (k + q) % p, p + (k + s) % p], 1))
        faces.append(([p + k, p + (k + q) % p, (k + t) % p], 1))
    faces.append(([(q * k) % p for k in range(p)], q))          # top star
    faces.append(([p + (q * k) % p for k in range(p)][::-1], q))  # bottom
    return verts, faces


def build_star_dipyramid(p, q, scale=1.0):
    """Star bipyramid over a {p/q} star polygon (dual of the star prism):
    p star-polygon equator vertices at z=0, apexes at z = +-scale; 2p
    triangular faces.  V=p+2, E=3p, F=2p."""
    q = valid_star_step(p, q)
    r = H = scale
    equ = [(r * math.cos(2 * math.pi * k / p),
            r * math.sin(2 * math.pi * k / p), 0.0) for k in range(p)]
    verts = equ + [(0.0, 0.0, H), (0.0, 0.0, -H)]    # E_k=k, top=p, bot=p+1
    ti, bi = p, p + 1
    faces = []
    for k in range(p):
        a, b = k, (k + q) % p
        faces.append(([a, b, ti], 1))
        faces.append(([b, a, bi], 1))
    return verts, faces


def _newell_normal(pts):
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0, z0 = pts[i]
        x1, y1, z1 = pts[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return nx, ny, nz


def _pole_of_face(verts, idx):
    """Pole of the face plane w.r.t. the unit sphere: n / (n . v0)."""
    pts = [verts[i] for i in idx]
    nx, ny, nz = _newell_normal(pts)
    v0 = pts[0]
    d = nx * v0[0] + ny * v0[1] + nz * v0[2]
    return (nx / d, ny / d, nz / d)


def _link_ordered_faces(incident):
    """Cyclic order of the faces around a vertex via its link (one cycle).
    `incident`: face_index -> (prev_vert, next_vert) of that vertex."""
    from collections import defaultdict
    adj = defaultdict(list)
    for f, (a, b) in incident.items():
        adj[a].append((f, b))
        adj[b].append((f, a))
    start = next(iter(incident))
    _a, b = incident[start]
    order = [start]
    cur_end, cur_face = b, start
    for _ in range(len(incident) - 1):
        nxt = next(((f, o) for (f, o) in adj[cur_end] if f != cur_face), None)
        if nxt is None:
            break
        cur_face, cur_end = nxt
        order.append(cur_face)
    return order


def build_star_trapezohedron(p, q, scale=1.0):
    """Star trapezohedron = polar reciprocal of the {p/q} star antiprism:
    2 axial apexes + a 2p-vertex zig-zag equator, 2p kite faces.  Includes
    the concave star trapezohedra.  V=2p+2, E=4p, F=2p."""
    q = valid_star_step(p, q)
    av, af = build_star_antiprism(p, q, scale)
    poles = [_pole_of_face(av, idx) for (idx, _d) in af]
    from collections import defaultdict
    incident = defaultdict(dict)
    for fi, (idx, _d) in enumerate(af):
        n = len(idx)
        for j in range(n):
            incident[idx[j]][fi] = (idx[(j - 1) % n], idx[(j + 1) % n])
    faces = [(_link_ordered_faces(incident[v]), 1) for v in range(len(av))]
    m = max(max(abs(c) for c in P) for P in poles) or 1.0
    verts = [(x / m * scale, y / m * scale, z / m * scale)
             for (x, y, z) in poles]
    return verts, faces


def _expand_faces(verts, faces):
    """Turn (indices, density) faces into a renderable mesh: convex faces
    stay n-gons, star faces (density > 1) are fanned from their centre so
    the star shows as a solid.  Returns (verts, faces, face_sizes)."""
    verts = list(verts)
    out = []
    fsize = []
    for f, d in faces:
        if d == 1:
            out.append(list(f))
            fsize.append(len(f))
        else:
            c = [sum(verts[i][k] for i in f) / len(f) for k in range(3)]
            ci = len(verts)
            verts.append(tuple(c))
            for i in range(len(f)):
                out.append([f[i], f[(i + 1) % len(f)], ci])
                fsize.append(len(f))
    return verts, out, fsize


# solids this build can construct: the non-snub Wythoff cases plus the
# snub / special solids for which stored coordinates are available.
# Skilling's figure is not one of the 75 (four faces meet at some of its
# edges), so it sits outside UNIFORMS and is appended to BUILDABLE with
# its own id and family.
SKILLING_U = 76
SKILLING_ROW = (SKILLING_U, "Great Disnub Dirhombidodecahedron",
                "| 3/2 5/3 3 5/2", ["3/2", "5/3", "3", "5/2"], 60, 240, 204)

_SNUB_U = set(_snub_data.SNUBS) if _snub_data else set()
BUILDABLE = [row for row in UNIFORMS
             if row[0] in _SNUB_U
             or (not row[2].strip().startswith('|') and len(row[3]) == 3)]
if _snub_data is not None:
    BUILDABLE.append(SKILLING_ROW)


# --- families (to organise the UI) --------------------------------------
# Convex (Platonic + Archimedean), the regular stars (Kepler-Poinsot), the
# hemipolyhedra (faces through the centre), and the remaining star uniforms
# split by symmetry (octahedral vs icosahedral).
_KEPLER_POINSOT = {34, 35, 52, 53}

_FAMILIES = [
    ('CONVEX', "Convex (Platonic & Archimedean)"),
    ('KEPLER', "Regular Star (Kepler-Poinsot)"),
    ('HEMI', "Hemipolyhedra"),
    ('STAR_O', "Star (Octahedral)"),
    ('STAR_I', "Star (Icosahedral)"),
    ('SKILLING', "Beyond the 75 (Skilling's Figure)"),
]


def _symmetry(pqr):
    toks = ' '.join(pqr)
    if '5' in toks:
        return 'I'
    if '4' in toks:
        return 'O'
    return 'T'


def _category(u, name, pqr):
    if u == SKILLING_U:
        return 'SKILLING'
    if 'hemi' in name.lower():
        return 'HEMI'
    if u in _KEPLER_POINSOT:
        return 'KEPLER'
    if not any('/' in x for x in pqr):
        return 'CONVEX'
    return 'STAR_O' if _symmetry(pqr) == 'O' else 'STAR_I'


_BY_CAT = {cat: [] for cat, _lbl in _FAMILIES}
for _row in BUILDABLE:
    _BY_CAT[_category(_row[0], _row[1], _row[3])].append(_row)


def _self_test():
    ok = 0
    bad = 0
    for (u, name, wy, pqr, Ve, Ee, Fe) in BUILDABLE:
        if u == SKILLING_U:
            V, F = build_skilling()
        elif u in _SNUB_U:
            V, F = build_snub(u)
        else:
            V, F = build_uniform(wy, pqr)
        E = set()
        for f, _d in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                E.add((min(a, b), max(a, b)))
        got = (len(V), len(E), len(F))
        if got == (Ve, Ee, Fe):
            ok += 1
        else:
            bad += 1
            print(f"U{u} {name}: got {got} expected {(Ve, Ee, Fe)}")
    print(f"uniform polyhedra: {ok}/{ok + bad} correct")
    # star prisms / antiprisms / dipyramids / trapezohedra
    star = {'prism': (build_star_prism, lambda p: (2 * p, 3 * p, p + 2)),
            'antiprism': (build_star_antiprism, lambda p: (2 * p, 4 * p,
                                                           2 * p + 2)),
            'dipyramid': (build_star_dipyramid, lambda p: (p + 2, 3 * p,
                                                           2 * p)),
            'trapezohedron': (build_star_trapezohedron,
                              lambda p: (2 * p + 2, 4 * p, 2 * p))}
    for (p, q) in [(5, 2), (7, 2), (7, 3), (8, 3)]:
        for nm, (fn, want) in star.items():
            V, F = fn(p, q)
            E = set()
            for f, _d in F:
                for i in range(len(f)):
                    a, b = f[i], f[(i + 1) % len(f)]
                    E.add((min(a, b), max(a, b)))
            chi = len(V) - len(E) + len(F)
            got = (len(V), len(E), len(F))
            tag = "OK" if (got == want(p) and chi == 2) else "BAD"
            print(f"star {nm:13s} {{{p}/{q}}} V={len(V):2d} E={len(E):2d} "
                  f"F={len(F):2d} chi={chi} {tag}")


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, BoolProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _family_items(self, context):
        return [(cat, lbl, "") for cat, lbl in _FAMILIES if _BY_CAT[cat]]

    _SOLID_CACHE = {}

    def _solid_items(self, context):
        cat = self.family
        if cat not in _SOLID_CACHE:
            _SOLID_CACHE[cat] = [(str(u), f"U{u}  {name}", wy)
                                 for (u, name, wy, pqr, V, E, F)
                                 in _BY_CAT.get(cat, [])]
        return _SOLID_CACHE[cat]

    def _family_update(self, context):
        ids = [it[0] for it in _solid_items(self, context)]
        if ids and self.solid not in ids:
            self.solid = ids[0]

    _PALETTE = {3: (0.90, 0.36, 0.23), 4: (0.27, 0.52, 0.79),
                5: (0.30, 0.69, 0.42), 6: (0.95, 0.77, 0.29),
                8: (0.25, 0.72, 0.72), 10: (0.55, 0.60, 0.29)}

    def _material_for(nn):
        name = f"Uniform {nn}-gon"
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            import colorsys
            rgb = _PALETTE.get(nn, colorsys.hsv_to_rgb((nn * 0.618) % 1.0,
                                                       0.55, 0.8))
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.5
        return mat

    def _make_object(context, name, verts, faces, color):
        verts, mfaces, fsize = _expand_faces(verts, faces)
        # centre and fit the 2 m cube.  These were emitted at their raw
        # construction coordinates, which left them both undersized and
        # OFF-ORIGIN (a third of a unit, for the default).
        verts = _fit_cube(verts)
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], mfaces)
        me.validate(clean_customdata=True)
        if color and len(me.polygons) == len(mfaces):
            sizes = sorted(set(fsize))
            slot = {n: i for i, n in enumerate(sizes)}
            for n in sizes:
                me.materials.append(_material_for(n))
            me.polygons.foreach_set('material_index',
                                    [slot[s] for s in fsize])
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    class MESH_OT_uniform_polyhedron_add(bpy.types.Operator):
        """Add a uniform polyhedron by Wythoff construction: convex,
        Kepler-Poinsot and non-convex star uniforms.  Star faces are
        triangulated from their centre so the star shows as a solid."""
        bl_idname = "mesh.uniform_polyhedron_add"
        bl_label = "Uniform Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(name="Family", items=_family_items,
                             update=_family_update)
        solid: EnumProperty(name="Solid", items=_solid_items)
        dual: BoolProperty(
            name="Dual", default=False,
            description="Build the dual by polar reciprocation (the "
                        "Catalan / Kepler-Poinsot / 'cronic' star duals). "
                        "Hemipolyhedra have vertices at infinity, drawn "
                        "truncated")
        coloring: EnumProperty(
            name="Coloring",
            items=[('SIDES', "By Face Size", ""), ('NONE', "None", "")],
            default='SIDES')
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain closed polyhedron"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo Style "
                    "modifier"),
                   ('WIRE', "Struts", "Wireframe modifier"),
                   ('BALLSTICK', "Ball and Stick",
                    "Edges as solid cylindrical struts and vertices "
                    "as small spheres (ball-and-stick model)"),
                   ('WIREFRAME', "Wireframe",
                    "Mesh edges only, displayed as a wireframe"),
                   ('FACETS', "Face Segments",
                    "Split into one inward-extruded, mitre-beveled "
                    "segment per face (best on the convex uniforms)")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.06, min=0.005, max=1.0,
            description="Leonardo face frame width")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness")
        strut_radius: FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.5,
            description="Ball-and-stick edge cylinder radius")
        node_radius: FloatProperty(
            name="Node Radius", default=0.035, min=0.0, max=0.5,
            description="Ball-and-stick vertex sphere radius "
                        "(0 = no nodes)")
        facet_depth: FloatProperty(name="Depth", default=0.15, min=0.01,
                                   max=2.0,
                                   description="Face Segments inward depth")
        facet_gap: FloatProperty(name="Bevel Gap", default=0.0, min=0.0,
                                 max=0.5,
                                 description="Gap between face segments")
        facet_explode: FloatProperty(name="Explode", default=0.1, min=0.0,
                                     max=5.0,
                                     description="Move segments outward")
        facet_separate: BoolProperty(
            name="Separate Meshes", default=False,
            description="Each face segment as its own object")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
            lay.prop(self, 'solid')
            lay.prop(self, 'dual')
            lay.prop(self, 'coloring')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style in ('LEONARDO', 'WIRE'):
                lay.prop(self, 'thickness')
            if self.style == 'BALLSTICK':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')
            if self.style == 'FACETS':
                lay.prop(self, 'facet_depth')
                lay.prop(self, 'facet_gap')
                lay.prop(self, 'facet_explode')
                lay.prop(self, 'facet_separate')
            lay.prop(self, 'scale')

        def execute(self, context):
            ids = [it[0] for it in _solid_items(self, context)]
            sid = self.solid if self.solid in ids else (ids[0] if ids
                                                       else self.solid)
            row = next((r for r in BUILDABLE if str(r[0]) == sid),
                       BUILDABLE[0])
            u, name, wy, pqr, Ve, Ee, Fe = row
            try:
                if u == SKILLING_U:
                    V, F = build_skilling()
                elif u in _SNUB_U:
                    V, F = build_snub(u)
                else:
                    V, F = build_uniform(wy, pqr)
                if self.dual:
                    V, F = build_dual(V, F)
                    name = name + " (dual)"
            except Exception as e:      # noqa: BLE001
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            verts = [tuple(c * self.scale for c in v) for v in V]
            if self.style == 'FACETS':
                try:
                    from .styles import facet_style
                except ImportError:
                    from styles import facet_style
                mat = (_material_for
                       if self.coloring == 'SIDES' else None)
                # uniform faces are (vertex_indices, marker) tuples,
                # and build_uniform returns them with mixed winding;
                # facet_style trusts the winding to aim the inward
                # mitre, so normalise every face to point outward first
                ff = [list(f[0]) for f in F]
                ff = facet_style.wind_outward(verts, ff)
                facet_style.emit_facets(
                    context, verts, ff, name,
                    self.facet_depth, self.facet_gap,
                    self.facet_explode, self.facet_separate, mat)
                self.report({'INFO'}, f"{name}: {len(F)} face segments")
                return {'FINISHED'}
            obj = _make_object(context, name, verts, F,
                               self.coloring == 'SIDES')
            if self.style == 'LEONARDO':
                try:
                    from . import leonardo_style
                except ImportError:
                    import leonardo_style
                leonardo_style.add_modifier(obj, self.border,
                                            self.thickness)
            elif self.style == 'WIRE':
                mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
                mod.thickness = self.thickness
                mod.use_even_offset = False
            elif self.style == 'BALLSTICK':
                try:
                    from .styles import ball_and_stick
                except ImportError:
                    from styles import ball_and_stick
                ball_and_stick.rebuild(obj, self.strut_radius,
                                       self.node_radius)
            elif self.style == 'WIREFRAME':
                obj.display_type = 'WIRE'
            self.report({'INFO'},
                        f"{name}: V={len(V)} F={len(F)}")
            return {'FINISHED'}

    from bpy.props import IntProperty

    _STAR_FORMS = [
        ('PRISM', "Prism", "Two star bases joined by squares"),
        ('ANTIPRISM', "Antiprism", "Two star bases joined by triangles"),
        ('DIPYRAMID', "Dipyramid", "Bipyramid over a star (dual of the prism)"),
        ('TRAPEZOHEDRON', "Trapezohedron",
         "Dual of the star antiprism (incl. concave forms)"),
    ]
    _STAR_BUILD = {'PRISM': build_star_prism, 'ANTIPRISM': build_star_antiprism,
                   'DIPYRAMID': build_star_dipyramid,
                   'TRAPEZOHEDRON': build_star_trapezohedron}

    class MESH_OT_star_prism_add(bpy.types.Operator):
        """Add a uniform {p/q} star prism, antiprism, dipyramid or
        trapezohedron (Step = 1 gives the ordinary convex form)."""
        bl_idname = "mesh.star_prism_add"
        bl_label = "Star Prism / Antiprism"
        bl_options = {'REGISTER', 'UNDO'}

        form: EnumProperty(name="Form", items=_STAR_FORMS, default='PRISM')
        sides: IntProperty(name="Sides (p)", default=5, min=3, max=32)
        step: IntProperty(name="Step (q)", default=2, min=1, max=15,
                          description="Star density; {p/q}, coprime with p")
        coloring: EnumProperty(
            name="Coloring",
            items=[('SIDES', "By Face Size", ""), ('NONE', "None", "")],
            default='SIDES')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            q = valid_star_step(self.sides, self.step)
            V, F = _STAR_BUILD[self.form](self.sides, q, 1.0)
            # centre at origin (symmetric already) and fit a 2 m cube, then
            # apply the user scale, so every {p/q} form stays in [-1, 1]^3
            mx = max((abs(c) for v in V for c in v), default=1.0) or 1.0
            V = [tuple(c / mx * self.scale for c in v) for v in V]
            lbl = dict((k, v) for k, v, _ in _STAR_FORMS)[self.form]
            _make_object(context, f"Star {lbl} {{{self.sides}/{q}}}",
                         V, F, self.coloring == 'SIDES')
            if q != self.step:
                self.report({'INFO'},
                            f"{{{self.sides}/{self.step}}} is not a star "
                            f"polygon; built {{{self.sides}/{q}}}")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.uniform_polyhedron_add",
                             icon='MESH_ICOSPHERE')
        self.layout.operator("mesh.star_prism_add", icon='MESH_CYLINDER')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_uniform_polyhedron_add)
        bpy.utils.register_class(MESH_OT_star_prism_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_star_prism_add)
        bpy.utils.unregister_class(MESH_OT_uniform_polyhedron_add)
