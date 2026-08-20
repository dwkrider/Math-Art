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


# --------------------------------------------------------------------------
# The axis-alignment rule
# --------------------------------------------------------------------------
# Michael Harman's systematic construction (unpublished, 1974; described
# on Hart's compounds pages) makes a compound out of one rule:
#
#   align the COMPONENT's n-fold axis with the COMPOUND's m-fold axis,
#   turn the component about that axis, replicate over every m-fold axis
#   of the compound's symmetry group, and drop duplicates.
#
# So it is one parametric operator rather than a table of meshes: pick a
# component, a symmetry group, which axis of each to match, and a phase.
# Two of the choices are discrete (which axis class, which of two mirror
# phases), which is why the named models still need a small table -- but
# the table holds five-tuples, not coordinates.
#
# Turning the phase continuously is exactly the "rotational freedom" of
# Hart's cube compounds: at generic angles the components stay distinct,
# and at special angles pairs coincide and the count drops, which the
# frozenset dedup in _orbit handles on its own.

def _tetra_rotations():
    """The twelve rotations preserving one of the cube's two inscribed
    tetrahedra -- a subgroup of the octahedral group, filtered rather
    than rebuilt."""
    T = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    out = []
    for M in _octa_rotations():
        W = T @ M.T
        d2 = ((W[:, None, :] - T[None, :, :]) ** 2).sum(-1)
        if d2.min(axis=1).max() < 1e-9:
            out.append(M)
    return out


GROUPS = {'T': _tetra_rotations, 'O': _octa_rotations, 'I': _icosa_rotations}

# One representative axis of each order, per symmetry group.
GROUP_AXES = {
    'T': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0)},
    'O': {2: (1.0, 1.0, 0.0), 3: (1.0, 1.0, 1.0), 4: (0.0, 0.0, 1.0)},
    'I': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0), 5: (0.0, 1.0, PHI)},
}

# and of each component solid, in its own frame
COMPONENT_AXES = {
    'TETRA': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0)},
    'CUBE': {2: (1.0, 1.0, 0.0), 3: (1.0, 1.0, 1.0), 4: (0.0, 0.0, 1.0)},
    'OCTA': {2: (1.0, 1.0, 0.0), 3: (1.0, 1.0, 1.0), 4: (0.0, 0.0, 1.0)},
    'DODECA': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0), 5: (0.0, 1.0, PHI)},
    'ICOSA': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0), 5: (0.0, 1.0, PHI)},
}


def _component(kind):
    tetra, cube, octa = _seeds()
    if kind == 'TETRA':
        return tetra
    if kind == 'CUBE':
        return cube
    if kind == 'OCTA':
        V, F = octa
        return ([tuple(math.sqrt(3.0) * c for c in v) for v in V], F)
    if kind == 'DODECA':
        return _dodeca()
    if kind == 'ICOSA':
        return _icosa()
    raise ValueError(kind)


def _align_rotation(src, dst):
    """Rotation carrying direction `src` onto `dst` by the shortest turn."""
    a = np.array(src, float)
    b = np.array(dst, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else _rot((1, 0, 0) if abs(a[0]) < 0.9
                                            else (0, 1, 0), math.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def build_axis_compound(component, group, comp_order, group_order,
                        phase=0.0, scale=1.0):
    """Harman's rule, as a function.

    Align the component's `comp_order`-fold axis with the compound's
    `group_order`-fold axis, turn it by `phase` degrees about that axis,
    and take the orbit under the group.  Returns the distinct components.
    """
    if group not in GROUPS:
        raise ValueError('unknown group %r' % (group,))
    ca = COMPONENT_AXES[component].get(comp_order)
    ga = GROUP_AXES[group].get(group_order)
    if ca is None:
        raise ValueError('%s has no %d-fold axis' % (component, comp_order))
    if ga is None:
        raise ValueError('group %s has no %d-fold axis' % (group,
                                                           group_order))
    V, F = _component(component)
    R = _align_rotation(ca, ga)
    R = _rot(ga, math.radians(phase)) @ R
    seed = ([tuple(scale * (R @ np.array(v, float))) for v in V],
            [list(f) for f in F])
    return _orbit(seed, GROUPS[group]())


# Named compounds reachable by the rule.  Each row is
# (key, label, component, group, component axis, group axis, phase),
# and every count is asserted in the self-test -- transcription is the
# real risk here, not the geometry.
AXIS_COMPOUNDS = [
    ('H_2TETRA', "2 Tetrahedra (Stella Octangula)", 'TETRA', 'O', 3, 3,
     0.0, 2),
    ('H_5TETRA', "5 Tetrahedra", 'TETRA', 'I', 2, 2, 0.0, 5),
    ('H_5CUBES', "5 Cubes", 'CUBE', 'I', 3, 3, 0.0, 5),
    ('H_5OCTA', "5 Octahedra", 'OCTA', 'I', 3, 3, 0.0, 5),
    ('H_2DODECA', "2 Dodecahedra (Octahedral)", 'DODECA', 'O', 3, 3,
     0.0, 2),
    ('H_2ICOSA', "2 Icosahedra (Octahedral)", 'ICOSA', 'O', 3, 3, 0.0, 2),
    ('H_5DODECA', "5 Dodecahedra", 'DODECA', 'I', 3, 3, 0.0, 5),
    ('H_12DODECA', "12 Dodecahedra (Octahedral)", 'DODECA', 'O', 2, 2,
     0.0, 12),
    ('H_12ICOSA', "12 Icosahedra (Octahedral)", 'ICOSA', 'O', 2, 2,
     0.0, 12),
    ('H_12CUBES', "12 Cubes (Octahedral)", 'CUBE', 'O', 4, 3, 0.0, 12),
    ('H_12TETRA', "12 Tetrahedra (Octahedral)", 'TETRA', 'O', 2, 2,
     0.0, 12),
    ('H_12ICOSA_4', "12 Icosahedra (4-fold)", 'ICOSA', 'O', 5, 4, 0.0, 12),
    ('H_30CUBES', "30 Cubes (Icosahedral)", 'CUBE', 'I', 4, 5, 0.0, 30),
    ('H_60DODECA', "60 Dodecahedra (Icosahedral)", 'DODECA', 'I', 5, 3,
     0.0, 60),
]

_AXIS_BY_KEY = {r[0]: r for r in AXIS_COMPOUNDS}


COMPOUNDS = [
    ('STELLA', "Stella Octangula (2 Tetrahedra)"),
    ('5TETRA', "Compound of 5 Tetrahedra"),
    ('10TETRA', "Compound of 10 Tetrahedra"),
    ('5CUBES', "Compound of 5 Cubes"),
    ('5OCTA', "Compound of 5 Octahedra"),
    ('CUBE_OCTA', "Cube + Octahedron"),
    ('DODECA_ICOSA', "Dodecahedron + Icosahedron"),
] + [(k, lbl) for k, lbl, *_rest in AXIS_COMPOUNDS]


def build_compound(kind, phase=None):
    """Return a list of components, each (V, F).

    `phase` overrides a named compound's own turn angle, which is what
    turns the rigid presets into Hart's rotational-freedom families:
    away from the named angle the components separate and the count
    generally rises.
    """
    if kind in _AXIS_BY_KEY:
        _k, _lbl, comp, grp, ca, ga, ph, _n = _AXIS_BY_KEY[kind]
        return build_axis_compound(comp, grp, ca, ga,
                                   ph if phase is None else phase)
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
        return [_dodeca(), scaled(_icosa(), PHI)]
    raise ValueError(kind)
