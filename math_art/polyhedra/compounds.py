# Polyhedral compounds: several solids sharing a centre and a symmetry.
#
# Part of the Math Art polyhedra engine (`math_art/polyhedra/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#

import math
import itertools
import numpy as np


PHI = (1 + 5 ** 0.5) / 2


def _seeds():
    tetra = ([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
             [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    cube = ([(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)],
            [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
    octa = ([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)],
            [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
             [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return tetra, cube, octa


def _hull_faces(V):
    """Merged planar faces of the convex hull (small vertex sets only)."""
    n = len(V)
    planes = []
    seen = set()
    for a, b, c in itertools.combinations(range(n), 3):
        A, B, C = V[a], V[b], V[c]
        nx = ((B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1]),
              (B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2]),
              (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]))
        ln = math.sqrt(sum(x * x for x in nx))
        if ln < 1e-9:
            continue
        nx = tuple(x / ln for x in nx)
        d = sum(nx[i] * A[i] for i in range(3))
        if d < 0:
            nx = tuple(-x for x in nx)
            d = -d
        key = tuple(round(x, 5) for x in nx) + (round(d, 5),)
        if key in seen:
            continue
        if all(sum(nx[i] * V[j][i] for i in range(3)) <= d + 1e-7
               for j in range(n)):
            seen.add(key)
            onv = [j for j in range(n)
                   if abs(sum(nx[i] * V[j][i] for i in range(3)) - d) < 1e-6]
            cen = [sum(V[j][i] for j in onv) / len(onv) for i in range(3)]
            ux = [V[onv[0]][i] - cen[i] for i in range(3)]
            uy = [nx[1] * ux[2] - nx[2] * ux[1],
                  nx[2] * ux[0] - nx[0] * ux[2],
                  nx[0] * ux[1] - nx[1] * ux[0]]

            def ang(j):
                dx = [V[j][i] - cen[i] for i in range(3)]
                return math.atan2(sum(dx[i] * uy[i] for i in range(3)),
                                  sum(dx[i] * ux[i] for i in range(3)))
            planes.append(sorted(onv, key=ang))
    return planes


def _midradius(V, F):
    """Radius of the sphere the edges are tangent to (distance to an edge
    LINE, not to its midpoint)."""
    edges = set()
    for f in F:
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            edges.add((min(a, b), max(a, b)))
    best = None
    for a, b in edges:
        p0, p1 = V[a], V[b]
        d = [p1[i] - p0[i] for i in range(3)]
        L = math.sqrt(sum(x * x for x in d)) or 1.0
        cr = [(-p0[1]) * d[2] - (-p0[2]) * d[1],
              (-p0[2]) * d[0] - (-p0[0]) * d[2],
              (-p0[0]) * d[1] - (-p0[1]) * d[0]]
        r = math.sqrt(sum(x * x for x in cr)) / L
        best = r if best is None else min(best, r)
    return best or 1.0


def _dodeca():
    V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    for a in (-1 / PHI, 1 / PHI):
        for b in (-PHI, PHI):
            V += [(0, a, b), (a, b, 0), (b, 0, a)]
    return V, _hull_faces(V)


def _icosa():
    V = []
    for a in (-1, 1):
        for b in (-PHI, PHI):
            V += [(0, a, b), (a, b, 0), (b, 0, a)]
    return V, _hull_faces(V)


def _octa_rotations():
    G = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = signs[i]
            if np.linalg.det(M) > 0:
                G.append(M)
    return G


def _rot(axis, ang):
    a = np.array(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _icosa_rotations():
    gens = [_rot((1, 1, 1), 2 * math.pi / 3),
            _rot((0, 1, PHI), 2 * math.pi / 5)]
    seen = {}
    def key(M):
        return tuple(np.round(M.ravel(), 5))
    ident = np.eye(3)
    seen[key(ident)] = ident
    frontier = [ident]
    while frontier:
        nf = []
        for M in frontier:
            for g in gens:
                N = g @ M
                k = key(N)
                if k not in seen:
                    seen[k] = N
                    nf.append(N)
        frontier = nf
        if len(seen) > 120:
            break
    return list(seen.values())


def _orbit(seed, G):
    """Distinct copies of the seed (V, F) under the rotation group."""
    V0, F0 = seed
    seen = set()
    out = []
    for M in G:
        W = [tuple(M @ np.array(v, float)) for v in V0]
        k = frozenset(tuple(np.round(w, 4)) for w in W)
        if k not in seen:
            seen.add(k)
            out.append((W, [list(f) for f in F0]))
    return out


COMPOUNDS = [
    ('STELLA', "Stella Octangula (2 Tetrahedra)"),
    ('5TETRA', "Compound of 5 Tetrahedra"),
    ('10TETRA', "Compound of 10 Tetrahedra"),
    ('5CUBES', "Compound of 5 Cubes"),
    ('5OCTA', "Compound of 5 Octahedra"),
    ('CUBE_OCTA', "Cube + Octahedron"),
    ('DODECA_ICOSA', "Dodecahedron + Icosahedron"),
]


def build_compound(kind):
    """Return a list of components, each (V, F)."""
    if np is None:
        raise RuntimeError("compounds need NumPy")
    tetra, cube, octa = _seeds()

    def scaled(seed, s):
        V, F = seed
        return ([tuple(s * c for c in v) for v in V], F)
    if kind == 'STELLA':
        return _orbit(tetra, _octa_rotations())
    if kind == '5TETRA':
        return _orbit(tetra, _icosa_rotations())
    if kind == '10TETRA':
        I = _icosa_rotations()
        mirr = ([(-v[0], v[1], v[2]) for v in tetra[0]],
                [list(reversed(f)) for f in tetra[1]])
        return _orbit(tetra, I) + _orbit(mirr, I)
    if kind == '5CUBES':
        return _orbit(cube, _icosa_rotations())
    if kind == '5OCTA':
        return _orbit(scaled(octa, math.sqrt(3)), _icosa_rotations())
    if kind == 'CUBE_OCTA':
        return [cube, scaled(octa, 1.5)]
    if kind == 'DODECA_ICOSA':
        # Build the icosahedron FROM the dodecahedron's face axes rather than
        # from a separately written vertex table.  `_dodeca` and `_icosa` list
        # their cyclic permutations in opposite senses, so pairing them
        # directly leaves the two components rotated 26.565 deg = atan(1/2)
        # apart: a valid figure, but not the dual compound, and its symmetry
        # collapses from Ih to Th.  Deriving one from the other cannot drift.
        dv, df = _dodeca()
        axes = []
        for f in df:
            c = [sum(dv[k][t] for k in f) / len(f) for t in range(3)]
            L = math.sqrt(sum(x * x for x in c)) or 1.0
            axes.append(tuple(x / L for x in c))
        # scale so both components share a midsphere -- the canonical dual
        # compound, with each solid's edges crossing the other's
        iv, ifc = axes, _hull_faces(axes)
        s = _midradius(dv, df) / _midradius(iv, ifc)
        return [(dv, df), ([tuple(s * c for c in v) for v in iv], ifc)]
    raise ValueError(kind)
