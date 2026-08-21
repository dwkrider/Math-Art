# Polyhedral compounds: several solids sharing a centre and a symmetry.
#
# Part of the Math Art polyhedra engine (`math_art/polyhedra/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#

import math
import itertools
import numpy as np

try:                                    # stored snub coordinates
    from . import _uniform_snub_data as _snub_data
except ImportError:                     # flat import (test runner)
    try:
        import _uniform_snub_data as _snub_data
    except ImportError:
        _snub_data = None


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


def dihedral_rotations(n):
    """D_n as rotation matrices: n turns about z and n half-turns across.

    Skilling's entries 20-25 are the four infinite prism and antiprism
    families, and they are the only rows that need a group outside
    T/O/I.  Entry 20 is "any prism repeated r times by successive
    rotations of 2 pi / n r about the n-fold prism axis", which is a
    cyclic group acting on a prism; adding the half-turns gives the
    dihedral case that entries 22 and 24 use.
    """
    out = [_rot((0.0, 0.0, 1.0), 2 * math.pi * k / n) for k in range(n)]
    out += [_rot((math.cos(math.pi * k / n), math.sin(math.pi * k / n), 0.0),
                 math.pi) for k in range(n)]
    return out


def cyclic_rotations(n):
    """C_n -- n turns about z."""
    return [_rot((0.0, 0.0, 1.0), 2 * math.pi * k / n) for k in range(n)]


def prism_family(n, r, anti=False, reflect=False, phase=0.0, star_m=0):
    """Skilling 20-25: a prism or antiprism repeated r times about its
    own axis.

    Entry 20/24: the constituent is turned by successive 2 pi / (n r),
    giving 2r copies when a vertical mirror is added and r without.  The
    count is what Skilling tabulates, and it is the reason these are
    families rather than models -- both n and r are free.
    """
    # star_m selects a star base: entries 24/25 are the m-EVEN antiprism,
    # which is the ALIGNED-base second antiprism, not the staggered one
    # (Skilling's rows 22/24 give the rule -- D_nd for m odd, D_nh for m
    # even).  m = 0 means the ordinary convex prism or antiprism.
    if star_m:
        V, F = (star_antiprism_second(n, star_m) if star_m % 2 == 0
                else star_antiprism_solid(n, star_m)) if anti \
            else star_prism_solid(n, star_m)
    else:
        V, F = prism_solid(n, anti)
    comps = []
    seen = set()
    steps = 2 * r if reflect else r
    for k in range(steps):
        a = math.radians(phase) + 2 * math.pi * k / (n * steps)
        R = _rot((0.0, 0.0, 1.0), a)
        W = [tuple(R @ np.array(v, float)) for v in V]
        if reflect and k % 2:
            W = [(x, -y, z) for x, y, z in W]
        key = frozenset(tuple(np.round(w, 4)) for w in W)
        if key in seen:
            continue
        seen.add(key)
        comps.append((W, [list(f) for f in F]))
    return comps


# --- Skilling's Table 1, entries 20-25 ------------------------------------
# The four infinite families.  Skilling's own summary calls them "two
# infinite families of prism compounds and two infinite families of
# antiprism compounds", parametric in the side count n AND the repeat
# count r, which is why they are not rows of fixed models: entry 20 gives
# 2r constituents and 21 gives r, for every n and r.
#
#   20  prism,     C_nh used, 2r constituents, rotational freedom
#   21  prism,     C_nh used,  r constituents, the special case
#   22  antiprism, S_2n used, 2r constituents, rotational freedom
#   23  antiprism, S_2n used,  r constituents, the special case
#
#: key -> (antiprism?, reflect? -> 2r rather than r, star numerator)
#: 24/25 use the m-EVEN star antiprism, whose bases are ALIGNED, giving
#: D_nh where the staggered one gives D_nd.
FAMILIES = {
    'S20_PRISM_FAMILY': (False, True, 0),
    'S21_PRISM_FAMILY': (False, False, 0),
    'S22_ANTI_FAMILY': (True, True, 0),
    'S23_ANTI_FAMILY': (True, False, 0),
    'S24_ANTI_FAMILY': (True, True, 2),
    'S25_ANTI_FAMILY': (True, False, 2),
}

FAMILY_LABELS = [
    ('S20_PRISM_FAMILY', "Skilling 20: 2r Prisms (free)"),
    ('S21_PRISM_FAMILY', "Skilling 21: r Prisms"),
    ('S22_ANTI_FAMILY', "Skilling 22: 2r Antiprisms (free)"),
    ('S23_ANTI_FAMILY', "Skilling 23: r Antiprisms"),
    ('S24_ANTI_FAMILY',
     "Skilling 24: 2r Pentagrammic Antiprisms (free, Sides=5)"),
    ('S25_ANTI_FAMILY',
     "Skilling 25: r Pentagrammic Antiprisms (Sides=5)"),
]


def _with_inversion(rots):
    """A rotation group extended by the centre, giving the full group.

    Needed because Skilling's Table 1 distinguishes O from O_h and I from
    I_h throughout, and the count of constituents differs between them --
    a triangular prism on a three-fold axis gives 4 copies in O and 8 in
    O_h, since O_h contains no mirror perpendicular to a three-fold axis
    and so cannot absorb the prism's own sigma_h.
    """
    return list(rots) + [-M for M in rots]


def _tetra_full():
    """T_d -- the full symmetry of one inscribed tetrahedron, order 24.

    NOT the same as T_h: T_d contains the S4 rotoreflections and the
    diagonal mirrors but no centre, while T_h contains the centre but no
    S4.  Skilling's band 1-19 uses S4 as the generating subgroup, so it
    is T_d that is needed there.  Filter the full octahedral group rather
    than rebuild.
    """
    T = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    out = []
    for M in _with_inversion(_octa_rotations()):
        W = T @ M.T
        d2 = ((W[:, None, :] - T[None, :, :]) ** 2).sum(-1)
        if d2.min(axis=1).max() < 1e-9:
            out.append(M)
    return out


GROUPS = {
    'T': _tetra_rotations,
    'Td': _tetra_full,
    'O': _octa_rotations,
    'I': _icosa_rotations,
    'Oh': lambda: _with_inversion(_octa_rotations()),
    'Ih': lambda: _with_inversion(_icosa_rotations()),
    'Th': lambda: _with_inversion(_tetra_rotations()),
}

# One representative axis of each order, per symmetry group.
GROUP_AXES = {
    'T': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0)},
    'O': {2: (1.0, 1.0, 0.0), 3: (1.0, 1.0, 1.0), 4: (0.0, 0.0, 1.0)},
    'I': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0), 5: (0.0, 1.0, PHI)},
}
GROUP_AXES['Td'] = GROUP_AXES['T']
GROUP_AXES['Th'] = GROUP_AXES['T']
GROUP_AXES['Oh'] = GROUP_AXES['O']
GROUP_AXES['Ih'] = GROUP_AXES['I']

# and of each component solid, in its own frame
COMPONENT_AXES = {
    'TETRA': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0)},
    'CUBE': {2: (1.0, 1.0, 0.0), 3: (1.0, 1.0, 1.0), 4: (0.0, 0.0, 1.0)},
    'OCTA': {2: (1.0, 1.0, 0.0), 3: (1.0, 1.0, 1.0), 4: (0.0, 0.0, 1.0)},
    # NB the dodecahedron's five-fold axis is (0, PHI, 1), NOT (0, 1, PHI)
    # like the icosahedron's.  `_dodeca()` is built out from the cube
    # (+-1, +-1, +-1) and its face normals come out along the OTHER cyclic
    # set, so it is not the dual of `_icosa()` in this frame but a turned
    # copy.  The wrong axis here does not change any component COUNT --
    # a bogus axis and a five-fold-on-three-fold alignment both leave the
    # stabilizer trivial -- so only checking the axis against the solid
    # catches it.
    'DODECA': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0), 5: (0.0, PHI, 1.0)},
    'ICOSA': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0), 5: (0.0, 1.0, PHI)},
    # The tetrahemihexahedron, for Skilling's entry 19.  It stands on the
    # octahedron's own six vertices and twelve edges -- it keeps four of
    # the eight triangles and adds the three equatorial squares -- so its
    # three-fold axes are the octahedron's, the body diagonals.  Only the
    # scale differs (circumradius 1 against the octahedron's sqrt(3)),
    # which is why entry 19 and entry 14 come out sharing a skeleton only
    # after both are normalised.
    'U4': {3: (1.0, 1.0, 1.0)},
}

#: prisms and antiprisms as components, keyed PRISM<n> / ANTI<n>.  Their
#: only alignable axis is the prism axis, which `prism_solid` puts on z.
for _n in (3, 4, 5, 6, 8, 10, 12):
    COMPONENT_AXES['PRISM%d' % _n] = {_n: (0.0, 0.0, 1.0)}
    COMPONENT_AXES['ANTI%d' % _n] = {_n: (0.0, 0.0, 1.0)}
del _n
#: star prisms {n/m}, whose axis is still the n-fold one
for _n, _m in ((5, 2), (10, 3)):
    COMPONENT_AXES['PRISM%d_%d' % (_n, _m)] = {_n: (0.0, 0.0, 1.0)}
COMPONENT_AXES['ANTI5_3'] = {5: (0.0, 0.0, 1.0)}
COMPONENT_AXES['ANTI5_2'] = {5: (0.0, 0.0, 1.0)}
del _n, _m


def to_standard_frame(V, F):
    """Turn a solid so an orthogonal triple of its own half-turn axes
    lies on the coordinate axes.

    Constituents that come from elsewhere are not in this module's frame
    -- `build_uniform` derives its mirrors from the Wythoff symbol, and
    the stored snub coordinates use a third orientation.  Orbiting an
    unaligned constituent silently returns the wrong count (a solid with
    O_h symmetry gave 15 copies under I_h instead of 5, because only an
    order-8 subgroup was actually shared), so the frame has to be fixed
    before the orbit, not assumed.

    Both O and I contain orthogonal triples of half-turn axes, and every
    such triple is equivalent under the group, so any one will do: the
    compound is only defined up to conjugacy anyway.
    """
    S = {tuple(np.round(v, 4)) for v in V}
    cands = [np.array(v, float) for v in V]
    for f in F:
        cands.append(np.array([sum(V[i][k] for i in f) / len(f)
                               for k in range(3)]))
        for i in range(len(f)):
            a, b = V[f[i]], V[f[(i + 1) % len(f)]]
            cands.append(np.array([(a[k] + b[k]) / 2.0 for k in range(3)]))
    # Cross products of the direct candidates, because a rotation axis
    # need not lie along ANY vertex, face centre or edge midpoint.  The
    # octahemioctahedron is the case that forced this: its four-fold axes
    # are the coordinate axes, but its faces are 8 triangles and 4
    # hexagons THROUGH THE CENTRE, so the hexagon centroids are the
    # origin and get skipped, and no other feature points that way.
    # Without the crosses the aligner falls back to two-folds, leaves the
    # solid 45 degrees out, and the orbit returns 15 instead of 5.
    direct = [d / np.linalg.norm(d) for d in cands
              if np.linalg.norm(d) > 1e-9]
    for i, a in enumerate(direct):
        for b in direct[i + 1:]:
            x = np.cross(a, b)
            if np.linalg.norm(x) > 1e-6:
                cands.append(x)

    axes = []                             # (max rotation order, axis)
    for d in cands:
        ln = np.linalg.norm(d)
        if ln < 1e-9:
            continue
        d = d / ln
        if any(abs(abs(float(d @ e)) - 1) < 1e-6 for _o, e in axes):
            continue
        best = 0
        for n in (2, 3, 4, 5):
            R = _rot(d, 2 * math.pi / n)
            if {tuple(np.round(R @ np.array(v, float), 4))
                    for v in V} == S:
                best = n
        if best:
            axes.append((best, d))
    # Take the HIGHEST-order orthogonal pair available.  For an
    # octahedral solid that means its four-fold axes: the tetrahedral
    # subgroup shared with an icosahedral group has its half-turns on
    # exactly those, so landing on an orthogonal pair of ordinary
    # two-folds instead leaves the solid 45 degrees out and the orbit
    # returns 15 copies where Skilling says 5.
    for order in (5, 4, 3, 2):
        sel = [d for o, d in axes if o == order]
        for i, a in enumerate(sel):
            for b in sel[i + 1:]:
                if abs(float(a @ b)) > 1e-6:
                    continue
                M = np.array([b, np.cross(a, b), a])
                if np.linalg.det(M) < 0:
                    M = np.array([b, np.cross(b, a), a])
                return ([tuple(M @ np.array(v, float)) for v in V],
                        [list(f) for f in F])
    raise ValueError('the solid has no orthogonal pair of rotation axes')


#: built constituents, keyed by kind.  `to_standard_frame` is O(n^2) in
#: the candidate directions once the cross products are included, and the
#: same handful of uniforms is rebuilt for row after row.
_COMPONENT_CACHE = {}


def _uniform_component(u):
    """Uniform polyhedron U<u> as (V, F), for use as a constituent.

    Imported LAZILY: `uniform_polyhedra_generator` imports from this
    package, so a module-level import here would close the cycle.
    """
    try:
        from .. import uniform_polyhedra_generator as _uni
    except ImportError:                   # flat import (test runner)
        import uniform_polyhedra_generator as _uni
    row = next(r for r in _uni.UNIFORMS if r[0] == u)
    V, faces = _uni.build_uniform(row[2], row[3])
    return to_standard_frame([tuple(float(c) for c in v) for v in V],
                             [list(f) for f, _d in faces])


def _component(kind):
    if kind in _COMPONENT_CACHE:
        V, F = _COMPONENT_CACHE[kind]
        return ([tuple(v) for v in V], [list(f) for f in F])
    if kind.startswith('U') and kind[1:].isdigit():
        out = _uniform_component(int(kind[1:]))
        _COMPONENT_CACHE[kind] = out
        return ([tuple(v) for v in out[0]], [list(f) for f in out[1]])
    if kind.startswith('PRISM'):
        spec = kind[5:]
        if '_' in spec:                    # PRISM<n>_<m>, a star prism
            n, m = (int(t) for t in spec.split('_'))
            return star_prism_solid(n, m)
        return prism_solid(int(spec), anti=False)
    if kind.startswith('ANTI'):
        spec = kind[4:]
        if '_' in spec:                    # ANTI<n>_<m>, a star antiprism
            n, m = (int(t) for t in spec.split('_'))
            # Skilling's rows 22/24 give the rule and it drives the
            # choice here: | 2 2 n/m is D_nd for m ODD and D_nh for m
            # EVEN, and those are the staggered and aligned-base solids
            # respectively.  So the symbol picks the construction.
            if m % 2 == 0:
                return star_antiprism_second(n, m)
            return star_antiprism_solid(n, m)
        return prism_solid(int(spec), anti=True)
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


def free_orientation_compound(component, group, tilt=23.0, phase=17.0,
                              scale=1.0):
    """A component in GENERAL position, replicated over the group.

    Hart's "central freedom" families: nothing about the constituent is
    aligned with anything, so its stabilizer in the group is trivial and
    the count is the full group order -- 12, 24 or 60 cubes for T, O
    and I.  The two angles are arbitrary and only have to avoid the
    special positions; they are exposed so the operator's Turn dial still
    moves the whole family.
    """
    V, F = _component(component)
    R = (_rot((0.0, 0.0, 1.0), math.radians(phase))
         @ _rot((1.0, 0.0, 0.0), math.radians(tilt)))
    seed = ([tuple(scale * (R @ np.array(v, float))) for v in V],
            [list(f) for f in F])
    return _orbit(seed, GROUPS[group]())


def _two_fold_axes(G):
    """Axes of the half-turns in a group, as unit vectors."""
    out = []
    for M in G:
        if abs(np.trace(M) + 1) > 1e-9 or abs(np.linalg.det(M) - 1) > 1e-9:
            continue                      # not a proper half-turn
        w, v = np.linalg.eig(M)
        for i in range(3):
            if abs(w[i] - 1) < 1e-9:
                a = np.real(v[:, i])
                out.append(a / np.linalg.norm(a))
    return out


#: phase sentinel -- "turn until the component's own half-turn axis lands
#: on one of the group's"
PERP = 'PERP'


def perpendicular_phase(component, group, comp_order, group_order):
    """The turn that puts the component's half-turn axis on the group's.

    Several of Skilling's prism compounds need the constituent's own
    two-fold axes to coincide with two-fold axes of the compound group,
    not merely its main axis with the main axis -- that is what takes the
    stabilizer from C_n up to D_n and so halves the constituent count.
    For the octahedral rows the angle is a round 15 degrees, but the
    icosahedral ones need 37.2388..., which no plausible guess and no
    scan on a round grid will find.  So it is COMPUTED: take the group's
    half-turn axes perpendicular to the alignment axis and measure the
    azimuth of one of them from the component's own.
    """
    ax = np.array(GROUP_AXES[group][group_order], float)
    ax = ax / np.linalg.norm(ax)
    A = _align_rotation(COMPONENT_AXES[component][comp_order],
                        GROUP_AXES[group][group_order])
    # where the COMPONENT's own half-turn axis lies, in its own frame.  A
    # prism's run through its vertical edge midpoints, at a vertex
    # azimuth; an antiprism's bisect a top and a bottom vertex and so sit
    # half a step round, at pi/2n.  Using the vertex azimuth for both
    # silently mis-phases every antiprism row by that half step.
    #
    # NB this pi/2n rule assumes a STAGGERED antiprism.  It is wrong for
    # the aligned-base second antiprism (`star_antiprism_second`), whose
    # rows pass an explicit axis to `perp_phase_from` instead.
    a0 = (math.pi / (2.0 * int(component[4:]))
          if component.startswith('ANTI') else 0.0)
    ref = A @ np.array([math.cos(a0), math.sin(a0), 0.0])
    ref = ref - ref.dot(ax) * ax
    ref = ref / np.linalg.norm(ref)
    side = np.cross(ax, ref)
    best = None
    for n2 in _two_fold_axes(GROUPS[group]()):
        if abs(float(n2.dot(ax))) > 1e-6:
            continue                      # not perpendicular
        a = math.degrees(math.atan2(float(n2.dot(side)),
                                    float(n2.dot(ref)))) % 180.0
        if best is None or a < best:
            best = a
    if best is None:
        raise ValueError('%s has no half-turn axis perpendicular to its '
                         '%d-fold axis' % (group, group_order))
    return best


def perp_half_turn_axis(component, comp_order):
    """A half-turn axis of the component perpendicular to its own
    `comp_order` axis, found from the solid rather than written down.

    The cube's are tidy -- (1, -1, 0) and friends -- but the icosahedral
    solids' are not: the dodecahedron's, perpendicular to its three-fold
    axis, points along (-0.809017, 0.309017, 0.5).  Rather than paste
    that in, walk the edge midpoints and keep the first direction whose
    half-turn maps the vertex set to itself.
    """
    V, F = _component(component)
    ax = np.array(COMPONENT_AXES[component][comp_order], float)
    ax = ax / np.linalg.norm(ax)
    S = {tuple(np.round(v, 4)) for v in V}
    for f in F:
        for i in range(len(f)):
            a, b = V[f[i]], V[f[(i + 1) % len(f)]]
            m = np.array([(a[k] + b[k]) / 2.0 for k in range(3)])
            ln = np.linalg.norm(m)
            if ln < 1e-9 or abs(float((m / ln) @ ax)) > 1e-6:
                continue
            d = m / ln
            R = _rot(d, math.pi)
            if {tuple(np.round(R @ np.array(v, float), 4))
                    for v in V} == S:
                return d
    raise ValueError('%s has no half-turn axis perpendicular to its '
                     '%d-fold axis' % (component, comp_order))


def vertex_mirror_phases(component, group, comp_order, group_order):
    """Turns about the aligned axis that put a constituent VERTEX on one
    of the group's MIRROR PLANES, in closed form.

    This is the condition behind Skilling's octahedral special cases, and
    he says so himself: the footnotes to entries 15 and 16 read "oriented
    as in Fig. 5(a)/(b) with its vertices on icosahedral mirror planes".
    Putting a vertex on a mirror doubles that vertex's stabilizer, which
    halves the number of DISTINCT vertices -- from 120 to 60 -- and that
    is what his "constituents per vertex" column is counting.

    Whether the constituent COUNT also halves is a separate matter and
    is what tells his rows apart: the same construction gives 10 copies
    at two of these turns (15 and 16, genuinely different compounds --
    they share not one vertex), 20 copies with two meeting at every
    vertex at a third (14), and 5 at a fourth (17, the classical
    compound).  So the turns are derived here and the rows select among
    them; none of the four is a fitted constant.

    Closed form rather than a scan, because a scan is exactly what fails
    on this family -- these turns are 22.238756... and 67.908424..., and
    a search on any round grid reports them unreachable.  Writing the
    turned vertex with Rodrigues' formula,

        v(t) = cos t * u + sin t * (n x u) + (1 - cos t)(n.u) n

    the condition m.v(t) = 0 for a mirror normal m is
    P cos t + Q sin t + C = 0 with P = m.u - (n.u)(m.n),
    Q = m.(n x u) and C = (n.u)(m.n) -- one phase-shifted cosine, solved
    directly.
    """
    G = GROUPS[group]()
    mirrors = []
    for M in G:
        if np.linalg.det(M) > 0 or abs(np.trace(M) - 1.0) > 1e-7:
            continue                      # keep plane reflections only
        w, vec = np.linalg.eig(M)
        ax = np.real(vec[:, int(np.argmin(np.real(w)))])
        ax = ax / np.linalg.norm(ax)
        if not any(min(np.linalg.norm(ax - b), np.linalg.norm(ax + b))
                   < 1e-6 for b in mirrors):
            mirrors.append(ax)
    ca = np.array(COMPONENT_AXES[component][comp_order], float)
    ga = np.array(GROUP_AXES[group][group_order], float)
    n = ga / np.linalg.norm(ga)
    A = _align_rotation(ca / np.linalg.norm(ca), n)
    V, _F = _component(component)
    us = []
    for v in V:
        q = A @ np.array(v, float)
        q = q / np.linalg.norm(q)
        if not any(np.linalg.norm(q - b) < 1e-6 for b in us):
            us.append(q)
    period = 360.0 / comp_order
    out = []
    for u in us:
        nu = float(n @ u)
        cr = np.cross(n, u)
        for m in mirrors:
            P = float(m @ u) - nu * float(m @ n)
            Q = float(m @ cr)
            C = nu * float(m @ n)
            R = math.hypot(P, Q)
            if R < 1e-12 or abs(C) > R + 1e-12:
                continue                  # this vertex never reaches it
            ph = math.atan2(Q, P)
            for s in (1.0, -1.0):
                t = math.degrees(ph + s * math.acos(
                    max(-1.0, min(1.0, -C / R))))
                t %= period
                if not any(min(abs(t - x), period - abs(t - x)) < 1e-6
                           for x in out):
                    out.append(t)
    return sorted(out)


def vertex_pairing_phases(component, group, comp_order, group_order):
    """Turns about the aligned axis at which a vertex of one constituent
    lands exactly on a vertex of ANOTHER.

    The mirror rule above cannot serve for Skilling's entry 19, because
    that compound's group is I, which has no mirrors at all.  What fixes
    its one free angle is his own sentence: "the polyhedron vertices
    coincide in pairs, each vertex of one class coalescing with one of
    the other, so true uniformity is recovered."  That is this condition,
    and it is what rescues a constituent which is only BI-uniform in the
    group used -- the tetrahemihexahedron's six vertices split into two
    classes under the C_3 about a triangular face, and only the pairing
    puts them back into one.

    Same closed form as `vertex_mirror_phases`, one conjugation deeper.
    For a group element g the requirement g.v_i(t) = v_j(t) forces the
    n-components to agree first, and that projection is again a single
    phase-shifted cosine in t; each root is then checked as a full vector
    equation, since agreeing along n is necessary and not sufficient.
    """
    G = GROUPS[group]()
    ca = np.array(COMPONENT_AXES[component][comp_order], float)
    ga = np.array(GROUP_AXES[group][group_order], float)
    n = ga / np.linalg.norm(ga)
    A = _align_rotation(ca / np.linalg.norm(ca), n)
    V, _F = _component(component)
    us = []
    for v in V:
        q = A @ np.array(v, float)
        q = q / np.linalg.norm(q)
        if not any(np.linalg.norm(q - b) < 1e-6 for b in us):
            us.append(q)
    period = 360.0 / comp_order
    out = []
    for g in G:
        ng = g.T @ n
        for ui in us:
            perp = ui - float(n @ ui) * n
            cr = np.cross(n, ui)
            for uj in us:
                P = float(ng @ perp)
                Q = float(ng @ cr)
                C = float(n @ ui) * float(ng @ n) - float(n @ uj)
                R = math.hypot(P, Q)
                if R < 1e-12 or abs(C) > R + 1e-12:
                    continue
                ph = math.atan2(Q, P)
                for s in (1.0, -1.0):
                    t = math.degrees(ph + s * math.acos(
                        max(-1.0, min(1.0, -C / R))))
                    M = _rot(n, math.radians(t))
                    if np.linalg.norm(g @ (M @ ui) - M @ uj) > 1e-7:
                        continue          # met along n only, not in full
                    t %= period
                    if not any(min(abs(t - x), period - abs(t - x)) < 1e-5
                               for x in out):
                        out.append(t)
    return sorted(out)


def perp_phase_from(component, group, comp_order, group_order, ref):
    """Turn carrying the component half-turn axis `ref` onto a group one.

    The general form of `perpendicular_phase`, for components whose own
    half-turn axis is not at azimuth 0 in their frame -- a cube aligned
    on its THREE-fold axis, say, whose perpendicular half-turns are its
    edge axes like (1, -1, 0).  `ref` is that axis in the component's own
    frame; only its component perpendicular to the alignment axis counts.
    """
    ax = np.array(GROUP_AXES[group][group_order], float)
    ax = ax / np.linalg.norm(ax)
    A = _align_rotation(COMPONENT_AXES[component][comp_order],
                        GROUP_AXES[group][group_order])
    r = A @ np.array(ref, float)
    r = r - r.dot(ax) * ax
    r = r / np.linalg.norm(r)
    side = np.cross(ax, r)
    best = None
    for n2 in _two_fold_axes(GROUPS[group]()):
        if abs(float(n2.dot(ax))) > 1e-6:
            continue
        a = math.degrees(math.atan2(float(n2.dot(side)),
                                    float(n2.dot(r)))) % 180.0
        if best is None or a < best:
            best = a
    if best is None:
        raise ValueError('%s has no half-turn axis perpendicular to its '
                         '%d-fold axis' % (group, group_order))
    return best


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
    if phase == PERP:
        phase = perpendicular_phase(component, group, comp_order,
                                    group_order)
    V, F = _component(component)
    R = _align_rotation(ca, ga)
    R = _rot(ga, math.radians(phase)) @ R
    seed = ([tuple(scale * (R @ np.array(v, float))) for v in V],
            [list(f) for f in F])
    return _orbit(seed, GROUPS[group]())


# --- a prism or antiprism together with its dual --------------------------
# Hart closes his pyramids chapter with fifteen of these -- prisms with
# their dual dipyramids for n = 3..10, and antiprisms with their dual
# trapezohedra for n = 4..10 -- so they are one parametric pair here
# rather than fifteen stored meshes.
#
# What makes the pair meaningful is that BOTH uniform families are
# already edge-tangent, which is not obvious and is worth the two lines
# of algebra.  Write R for the circumradius of the n-gon of unit edge,
# R = 1 / (2 sin(pi/n)).  For the prism the vertical edge midpoints sit
# at radius R and z = 0, and the polygon edge midpoints at radius
# R cos(pi/n) and z = +-1/2; those agree exactly because
# R sin(pi/n) = 1/2 is the definition of R.  For the antiprism, height h
# is fixed by making the lateral edges unit, h^2 = 1 - 4 R^2 sin^2(pi/2n),
# and then
#
#     d_polygon^2 - d_lateral^2
#         = R^2 (cos^2(pi/n) - cos^2(pi/2n)) + h^2/4
#         = R^2 (cos^2(pi/n) - cos^2(pi/2n) + sin^2(pi/n) - sin^2(pi/2n))
#         = 0.
#
# So each solid has a genuine midsphere and needs no canonicalization
# first: reciprocating in that sphere puts every dual edge crossing its
# own primal edge at a right angle, at the point where both touch it,
# which is the interlocked look Hart's models have.  Reciprocating in
# any other sphere would still give the right dual SHAPE but the wrong
# relative size, and the two would not touch.
def edge_touch(a, b):
    """Foot of the perpendicular from the centre onto the line ab.

    Where an edge touches the midsphere, which is NOT its midpoint in
    general -- it is for the prism, whose edges are symmetric about their
    closest point, but not for the dipyramid, whose apex and equator sit
    at different radii.  Assuming the midpoint there is a real trap: it
    reports a bipyramid tangent to a sphere it is not tangent to.
    """
    d = [b[k] - a[k] for k in range(3)]
    dd = sum(x * x for x in d)
    t = -sum(a[k] * d[k] for k in range(3)) / dd if dd else 0.0
    return [a[k] + t * d[k] for k in range(3)]


def prism_solid(n, anti=False):
    """A uniform n-gonal prism or antiprism of unit edge, axis on z."""
    if n < 3:
        raise ValueError('a prism needs at least 3 sides')
    R = 0.5 / math.sin(math.pi / n)
    if anti:
        h = math.sqrt(max(1.0 - 4.0 * R * R
                          * math.sin(math.pi / (2.0 * n)) ** 2, 0.0))
        off = math.pi / n
    else:
        h = 1.0
        off = 0.0
    V = [(R * math.cos(2 * math.pi * k / n),
          R * math.sin(2 * math.pi * k / n), h / 2.0) for k in range(n)]
    V += [(R * math.cos(2 * math.pi * k / n + off),
           R * math.sin(2 * math.pi * k / n + off), -h / 2.0)
          for k in range(n)]
    return V, _hull_faces(V)


def star_prism_solid(n, m):
    """The uniform prism on the star polygon {n/m}, axis on z.

    Its VERTICES are those of the ordinary n-prism, only rescaled -- the
    star's unit edge spans m steps, so R = 1/(2 sin(m pi/n)) instead of
    1/(2 sin(pi/n)).  That is why Skilling gives the star rows the same
    constituent counts as their convex partners (36/37 match 34/35, and
    41 matches 40): the placement and the stabilizer see the same vertex
    set up to scale, and only the FACES differ.
    """
    if math.gcd(n, m) != 1 or not 1 < m < n / 2:
        raise ValueError('{%d/%d} is not a star polygon' % (n, m))
    R = 0.5 / math.sin(m * math.pi / n)
    V = [(R * math.cos(2 * math.pi * k / n),
          R * math.sin(2 * math.pi * k / n), 0.5) for k in range(n)]
    V += [(x, y, -0.5) for x, y, _z in V]
    top = [(k * m) % n for k in range(n)]        # the star circuit
    F = [top, [n + i for i in reversed(top)]]
    F += [[k, (k + m) % n, n + (k + m) % n, n + k] for k in range(n)]
    return V, F


def star_antiprism_second(n, m):
    """The 'second' antiprism on {n/m} -- Coxeter's s'{2/p}.

    CLM section 10 shows that setting b = c = 0 in equation (10.3) leaves
    [(X+1)^2 - a^2]^2 = 0 with a = 2 cos(pi/p), so rho^2 = 1/(3 -+ a) --
    TWO roots, rho being the circumradius of the VERTEX FIGURE, a
    trapezoid of sides 1, 1, 1, a.  The crossed trapezoid needs a < 1, so
    the second solution exists only for 2 < p < 3; p = 5/2 qualifies.

    Converting rho to the polyhedron: the neighbours of a vertex lie on a
    circle of radius rho at distance sqrt(1 - rho^2) from it, so the
    circumradius is R = 1/(2 sqrt(1 - rho^2)).  The two roots then give

        1/(3+a): R = 0.587785, rings offset pi/n, lateral gap 3pi/n
        1/(3-a): R = 0.656431, rings offset 0,    lateral gap 2pi/n

    and the second of those is this function.  **Its two bases are
    azimuthally ALIGNED, not staggered** -- which is the whole point,
    because a horizontal mirror then maps one base onto the other and the
    symmetry is D_nh rather than the D_nd of an ordinary antiprism.  That
    is exactly the symmetry Skilling's Table 1 records for entries 44/45,
    and it is why no re-wiring of the staggered form can produce them.
    """
    if math.gcd(n, m) != 1 or not 1 < m < n - 1:
        raise ValueError('{%d/%d} is not a star polygon' % (n, m))
    m = min(m, n - m)
    a = 2.0 * math.cos(math.pi * m / n)
    if a >= 1.0:
        raise ValueError('{%d/%d} admits no second antiprism (a = %.4f, '
                         'the crossed trapezoid needs a < 1)' % (n, m, a))
    R = 0.5 / math.sin(m * math.pi / n)
    h2 = 1.0 - 2.0 * R * R * (1.0 - math.cos(2.0 * math.pi / n))
    if h2 <= 1e-12:
        raise ValueError('{%d/%d} admits no second antiprism' % (n, m))
    h = math.sqrt(h2)
    V = [(R * math.cos(2 * math.pi * k / n),
          R * math.sin(2 * math.pi * k / n), h / 2.0) for k in range(n)]
    V += [(x, y, -h / 2.0) for x, y, _z in V]      # bases ALIGNED
    star = [(k * m) % n for k in range(n)]
    F = [star, [n + i for i in reversed(star)]]
    F += [[k, (k + m) % n, n + (k + 1) % n] for k in range(n)]
    F += [[n + k, n + (k + m) % n, (k + 1) % n] for k in range(n)]
    return V, F


def star_antiprism_solid(n, m):
    """The uniform antiprism on the star polygon {n/m}, axis on z.

    Unlike the star PRISM, this one does not close for every plausible
    wiring, and the failures are informative.  A lateral triangle sits on
    one base edge (T_k, T_{k+m}) and reaches one vertex of the other
    base, so that vertex must be a lateral neighbour of BOTH ends.  For
    {5/2} the obvious "nearest bottom vertex" attachment -- the one that
    makes an ordinary antiprism -- leaves T_0 and T_2 with no common
    neighbour, so it does not close at all; only the attachment two steps
    round does, and it fixes h at 0.5257, not the 0.9457 the nearest
    attachment would want.

    For n = 5 the two star circuits {5/2} and {5/3} share an edge set and
    their triangles relabel onto each other, so both arguments return the
    SAME solid -- and that solid has no horizontal mirror, so it is the
    D_5d one.  Skilling's rows 22/24 give the rule (|2 2 n/m is D_nd for
    m odd and D_nh for m even), which identifies it as the CROSSED
    antiprism |2 2 5/3 of his entries 28 and 29.  The D_5h partner
    |2 2 5/2 of entries 44/45 is a different solid this does not build.
    """
    # m may exceed n/2 here, unlike the prism: {5/3} is the retrograde
    # reading of the same pentagram and is the symbol Skilling uses for
    # the crossed antiprism, so it has to be accepted rather than
    # normalised away.
    if math.gcd(n, m) != 1 or not 1 < m < n - 1:
        raise ValueError('{%d/%d} is not a star polygon' % (n, m))
    m = min(m, n - m)                     # {n/m} and {n/(n-m)} are the
    R = 0.5 / math.sin(m * math.pi / n)   # same circuit, opposite sense
    c = 1.0 - 2.0 * R * R * (1.0 - math.cos(3.0 * math.pi / n))
    if c <= 1e-12:
        raise ValueError('{%d/%d} admits no uniform antiprism' % (n, m))
    h = math.sqrt(c)
    V = [(R * math.cos(2 * math.pi * k / n),
          R * math.sin(2 * math.pi * k / n), h / 2.0) for k in range(n)]
    V += [(R * math.cos(2 * math.pi * k / n + math.pi / n),
           R * math.sin(2 * math.pi * k / n + math.pi / n), -h / 2.0)
          for k in range(n)]
    star = [(k * m) % n for k in range(n)]
    F = [star, [n + i for i in reversed(star)]]
    F += [[k, (k + m) % n, n + (k + m + 1) % n] for k in range(n)]
    F += [[n + k, n + (k + m) % n, (k - 1) % n] for k in range(n)]
    return V, F


def prism_and_dual(n, anti=False):
    """The compound of a uniform n-prism (or n-antiprism) and its dual."""
    V, F = prism_solid(n, anti)

    # the midsphere radius, measured rather than assumed
    mids = [math.sqrt(sum(c * c for c in edge_touch(V[f[i]],
                                                    V[f[(i + 1) % len(f)]])))
            for f in F for i in range(len(f))]
    if max(mids) - min(mids) > 1e-9:
        raise ValueError('the %d-gonal %s has no midsphere (%.3g spread)'
                         % (n, 'antiprism' if anti else 'prism',
                            max(mids) - min(mids)))
    rho = mids[0]

    # polar reciprocation in that midsphere: one dual vertex per face,
    # at rho^2 n / d.  The dual of a convex solid is convex, so hulling
    # the poles recovers its faces.
    P = []
    for f in F:
        cen = [sum(V[i][k] for i in f) / len(f) for k in range(3)]
        a, b, c = (V[i] for i in f[:3])
        u = [b[k] - a[k] for k in range(3)]
        w = [c[k] - a[k] for k in range(3)]
        nx = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
              u[0] * w[1] - u[1] * w[0]]
        ln = math.sqrt(sum(x * x for x in nx))
        nx = [x / ln for x in nx]
        d = sum(nx[k] * cen[k] for k in range(3))
        if d < 0:
            nx = [-x for x in nx]
            d = -d
        P.append(tuple(rho * rho * x / d for x in nx))
    return [(V, F), (P, _hull_faces(P))]


# --- Hart's "inscribed pieces" -------------------------------------------
# Six models showing one regular solid sitting inside another, sharing
# its vertices with some of the larger one's.  These are not orbits of
# anything -- they are two named solids in a fixed relative position --
# so they get coordinates rather than a rule.
#
# Five of the six are immediate in the seed frames this module already
# uses, which is the point of choosing those frames:
#
#   * `_dodeca()` literally CONTAINS the cube (+-1, +-1, +-1) as eight of
#     its twenty vertices, so cube-in-dodecahedron is a selection, and
#     tetrahedron-in-dodecahedron selects four of those eight.
#   * the cube's two inscribed tetrahedra are its alternate vertices.
#
# The sixth needs one line of algebra.  Each vertex of an icosahedron
# inscribed in an octahedron divides an octahedron EDGE in the golden
# ratio -- twelve edges, twelve vertices, one each.  Taking `_icosa()`'s
# cyclic coordinates, the vertex (1, phi, 0) must lie on the edge running
# from (d, 0, 0) to (0, d, 0), i.e. on x + y = d, so
#
#     d = 1 + phi = phi^2.
#
# That the CYCLIC set works and the full permutation set does not is
# exactly why the figure is chiral, and why Hart lists a two-icosahedra
# version: the mirror set (0, +-phi, +-1) inscribes in the same
# octahedron the other way round.
def inscribed(kind):
    """One of Hart's six inscribed-piece models, as [(V, F), ...]."""
    tetra, cube, octa = _seeds()
    if kind == 'INS_TETRA_CUBE':
        return [tetra, cube]
    if kind == 'INS_2TETRA_CUBE':
        tv, tf = tetra
        mirror = ([(-x, -y, -z) for x, y, z in tv], tf)
        return [tetra, mirror, cube]
    dv, df = _dodeca()
    if kind in ('INS_CUBE_DODECA', 'INS_TETRA_DODECA'):
        cv = [v for v in dv if all(abs(abs(c) - 1) < 1e-9 for c in v)]
        if len(cv) != 8:
            raise ValueError('the dodecahedron seed lost its inscribed cube')
        if kind == 'INS_TETRA_DODECA':
            cv = [v for v in cv if v[0] * v[1] * v[2] > 0]
        return [(cv, _hull_faces(cv)), (dv, df)]
    if kind in ('INS_ICOSA_OCTA', 'INS_2ICOSA_OCTA'):
        d = PHI * PHI
        # the unit octahedron from _seeds(), scaled out to phi^2 -- its
        # face list is explicit there, and _hull_faces cannot supply one
        # for a solid whose vertices are this degenerate on the axes
        ov = ([tuple(d * c for c in v) for v in octa[0]], octa[1])
        iv, if_ = _icosa()
        out = [(iv, if_)]
        if kind == 'INS_2ICOSA_OCTA':          # the other hand
            mv = [(x, z, y) for x, y, z in iv]
            out.append((mv, _hull_faces(mv)))
        out.append(ov)
        return out
    raise ValueError(kind)


# --- Skilling's Table 1, entries 68-75: duplication of enantiomorphs -----
# The whole band is one idea: a CHIRAL uniform polyhedron has only
# rotations in its symmetry group, so adding the centre of inversion
# turns it into a compound of the solid and its mirror image.  Skilling's
# columns say exactly that -- used symmetry O or I, compound symmetry
# O_h or I_h, two constituents throughout -- and `_orbit` under the full
# group produces it without any placement decision at all, because a
# chiral solid in the standard frame is already correctly positioned.
#
# The eight constituents are all snubs, and the repo already stores their
# coordinates.  Matching Skilling's Wythoff symbols to the U numbers is
# the only real work, and it is by multiset since the two sources order
# the generators differently (Skilling's |2 3 5/2 is U57's | 2 5/2 3):
#
#   68 |2 3 4      CLM 24 -> U12  snub cube
#   69 |2 3 5      CLM 32 -> U29  snub dodecahedron
#   70 |2 3 5/2    CLM 73 -> U57  great snub icosidodecahedron
#   71 |2 3 5/3    CLM 88 -> U69  great inverted snub icosidodecahedron
#   72 |2 3/2 5/3  CLM 90 -> U74  great retrosnub icosidodecahedron
#   73 |2 5/2 5    CLM 49 -> U40  snub dodecadodecahedron
#   74 |2 5/3 5    CLM 76 -> U60  inverted snub dodecadodecahedron
#   75 |3 5/3 5    CLM 58 -> U46  snub icosidodecadodecahedron
#
# The two snubs the repo has that are NOT here -- U32 and U72 -- are the
# achiral ones, which is the consistency check on the mapping: an
# achiral solid would give one constituent, not two, so it cannot appear
# in this band.
def enantiomorph_pair(u):
    """A chiral uniform polyhedron together with its mirror image.

    Mirror the solid directly rather than orbiting it under the full
    group.  Orbiting is what the rest of this module does, but it assumes
    the constituent sits in the same frame as `_icosa_rotations()`, and
    the STORED SNUB COORDINATES DO NOT -- U29 happens to, but the star
    snubs are oriented differently and orbiting them returns 10 or 28
    copies instead of 2.  Inversion needs no frame at all.

    The count is then 2 by construction, so it proves nothing; what is
    worth asserting is the chirality that makes the compound exist.  A
    chiral solid's mirror is a genuinely different vertex set and the
    dedup keeps both; an ACHIRAL one mirrors onto itself and collapses
    to a single constituent, which is exactly why the two achiral snubs
    in the data (U32, U72) are absent from Skilling's band.
    """
    if _snub_data is None:
        raise RuntimeError('snub coordinate data not available')
    S = _snub_data.SNUBS[u]
    V = [tuple(float(c) for c in v) for v in S['V']]
    F = [list(f) for f in S['F']]
    # inversion reverses orientation, so the faces are re-wound to keep
    # the mirror's normals pointing outward like the original's
    W = [(-x, -y, -z) for x, y, z in V]
    key = frozenset(tuple(np.round(v, 4)) for v in V)
    if frozenset(tuple(np.round(v, 4)) for v in W) == key:
        return [(V, F)]                   # achiral: the mirror IS itself
    return [(V, F), (W, [list(reversed(f)) for f in F])]


#: (key, label, U number, compound group, Skilling's constituent count)
ENANTIOMORPHS = [
    ('S68_SNUB_CUBE', "Skilling 68: 2 Snub Cubes", 12, 'Oh', 2),
    ('S69_SNUB_DODECA', "Skilling 69: 2 Snub Dodecahedra", 29, 'Ih', 2),
    ('S70_GT_SNUB_ID', "Skilling 70: 2 Great Snub Icosidodecahedra",
     57, 'Ih', 2),
    ('S71_GT_INV_SNUB_ID',
     "Skilling 71: 2 Great Inverted Snub Icosidodecahedra", 69, 'Ih', 2),
    ('S72_GT_RETRO_SNUB_ID',
     "Skilling 72: 2 Great Retrosnub Icosidodecahedra", 74, 'Ih', 2),
    ('S73_SNUB_DD', "Skilling 73: 2 Snub Dodecadodecahedra", 40, 'Ih', 2),
    ('S74_INV_SNUB_DD', "Skilling 74: 2 Inverted Snub Dodecadodecahedra",
     60, 'Ih', 2),
    ('S75_SNUB_IDD', "Skilling 75: 2 Snub Icosidodecadodecahedra",
     46, 'Ih', 2),
]

_ENANT_BY_KEY = {r[0]: r for r in ENANTIOMORPHS}


# --- Skilling's Table 1, entries 46-67 ------------------------------------
# "Tetrahedral symmetry embedded in octahedral or icosahedral symmetry".
# The constituent is simply PLACED IN ITS OWN STANDARD FRAME and orbited
# under the compound group -- no axis to choose, no phase.  The count
# then falls out of what the two groups share:
#
#   O_h constituent in I_h : they share T_h, 120/24 =  5
#   I_h constituent in O_h : they share T_h,  48/24 =  2
#   I_h constituent in I_h : a DIFFERENT icosahedral group sharing only
#                            T_h,                120/24 =  5
#   T_d constituent in O_h : share T_d,          48/24 =  2
#   T_d constituent in I    : share T,            60/12 =  5
#   T_d constituent in I_h  : share T,           120/12 = 10
#
# The one thing that has to be right is the FRAME -- see
# `to_standard_frame`, without which these came out 15, 24, 30 and 60.
def subgroup_compound(component, group):
    """A constituent in its own standard frame, orbited under `group`.

    One correction is needed for the rows whose constituent has the SAME
    symmetry group as the compound -- the icosahedral ones, entries 47,
    49, 51 and 53.  Those want the constituent placed in a DIFFERENT
    icosahedral group sharing only the tetrahedral subgroup, giving 5
    copies; if it lands in the reference group instead, the orbit is a
    single copy and the compound collapses.

    Whether `to_standard_frame` lands in the reference group or a
    conjugate depends on which solid it is (the great icosahedron did,
    the small stellated dodecahedron did not), so detect it rather than
    tabulate it: a one-copy orbit means the frames coincided, and a
    quarter turn about z fixes it.  That turn is a provably correct
    choice, not a fudge -- it lies in O_h, so it normalizes T_h and keeps
    the shared tetrahedral subgroup, but it is not in I_h, so it does
    move the constituent to a different icosahedral group.
    """
    V, F = _component(component)
    comps = _orbit((V, F), GROUPS[group]())
    if len(comps) == 1:
        R = _rot((0.0, 0.0, 1.0), math.pi / 2)
        comps = _orbit(([tuple(R @ np.array(v, float)) for v in V], F),
                       GROUPS[group]())
    return comps


#: (key, label, component, group, Skilling's constituent count)
SUBGROUP_COMPOUNDS = [
    ('S46_ICOSA', "Skilling 46: 2 Icosahedra (Oh)", 'U22', 'Oh', 2),
    ('S47_ICOSA', "Skilling 47: 5 Icosahedra (Ih)", 'U22', 'Ih', 5),
    ('S54_TRUNCTET', "Skilling 54: 2 Truncated Tetrahedra (Oh)",
     'U2', 'Oh', 2),
    ('S55_TRUNCTET', "Skilling 55: 5 Truncated Tetrahedra (I)",
     'U2', 'I', 5),
    ('S56_TRUNCTET', "Skilling 56: 10 Truncated Tetrahedra (Ih)",
     'U2', 'Ih', 10),
    ('S57_TRUNCCUBE', "Skilling 57: 5 Truncated Cubes", 'U9', 'Ih', 5),
    ('S58_STELLTRUNCHEX',
     "Skilling 58: 5 Stellated Truncated Hexahedra", 'U19', 'Ih', 5),
    ('S59_CUBOCTA', "Skilling 59: 5 Cuboctahedra", 'U7', 'Ih', 5),
    ('S62_RHOMBICUBOCTA', "Skilling 62: 5 Rhombicuboctahedra",
     'U10', 'Ih', 5),
    ('S67_NONCONVEX_GRCO',
     "Skilling 67: 5 Nonconvex Great Rhombicuboctahedra", 'U17', 'Ih', 5),
    # Entry 18 sits in the "miscellaneous" band but is built the same
    # way.  Entry 19, the compound of 20 tetrahemihexahedra, is NOT here:
    # Skilling singles it out as the only uniform compound that cannot be
    # reached by adding symmetry to a group in which the constituent is
    # uniform.  It needs the bi-uniform C_3 placement in which the six
    # vertices fall into two classes and then coalesce in pairs.
    ('S18_TETRAHEMIHEX', "Skilling 18: 5 Tetrahemihexahedra (I)",
     'U4', 'I', 5),
    ('S48_GT_DODECA', "Skilling 48: 2 Great Dodecahedra (Oh)",
     'U35', 'Oh', 2),
    ('S49_GT_DODECA', "Skilling 49: 5 Great Dodecahedra (Ih)",
     'U35', 'Ih', 5),
    ('S50_SM_STELL_DODECA',
     "Skilling 50: 2 Small Stellated Dodecahedra (Oh)", 'U34', 'Oh', 2),
    ('S51_SM_STELL_DODECA',
     "Skilling 51: 5 Small Stellated Dodecahedra (Ih)", 'U34', 'Ih', 5),
    ('S52_GT_ICOSA', "Skilling 52: 2 Great Icosahedra (Oh)",
     'U53', 'Oh', 2),
    ('S53_GT_ICOSA', "Skilling 53: 5 Great Icosahedra (Ih)",
     'U53', 'Ih', 5),
    ('S60_CUBOHEMIOCTA', "Skilling 60: 5 Cubohemioctahedra",
     'U15', 'Ih', 5),
    ('S61_OCTAHEMIOCTA', "Skilling 61: 5 Octahemioctahedra",
     'U3', 'Ih', 5),
    ('S64_SM_CUBICUBOCTA', "Skilling 64: 5 Small Cubicuboctahedra",
     'U13', 'Ih', 5),
    ('S65_GT_CUBICUBOCTA', "Skilling 65: 5 Great Cubicuboctahedra",
     'U14', 'Ih', 5),
    # Entries 63 and 66 were long recorded here as blocked on
    # constituents "with four-part Wythoff symbols that build_uniform
    # does not construct".  That was a misreading of the symbol, not a
    # gap in the engine.  Skilling writes them `2 4 3/2 4/2 |` and
    # `2 4/3 3/2 4/2 |`, the form in which the repeated generator records
    # that Wythoff's construction traverses each face TWICE; the same two
    # solids are written `3/2 2 4 |` and `4/3 3/2 2 |` in this repo's
    # table, and both have always built.  They are the small and great
    # RHOMBIHEXAHEDRA, U18 and U21.
    #
    # Coxeter, Longuet-Higgins & Miller settle it beyond the symbol.
    # Skilling cites their figures 60 and 82, and CLM's Table 7 gives
    # figure 60 as 24 vertices, 48 edges, 12{4} + 6{8} and figure 82 as
    # 24, 48, 12{4} + 6{8/3} -- which are exactly U18 and U21, faces and
    # counts alike.  Skilling's own remark column agrees a third time: he
    # groups 62, 63, 64 as "shared vertices and edge length" and U10,
    # U18, U13 do share a vertex set, as do U14, U21, U17 for 65, 66, 67.
    ('S63_SM_RHOMBIHEX', "Skilling 63: 5 Small Rhombihexahedra",
     'U18', 'Ih', 5),
    ('S66_GT_RHOMBIHEX', "Skilling 66: 5 Great Rhombihexahedra",
     'U21', 'Ih', 5),
]

_SUBGROUP_BY_KEY = {r[0]: r for r in SUBGROUP_COMPOUNDS}

#: Hart's "central freedom" cubes -- (key, label, group, count)
FREE_COMPOUNDS = [
    ('HC_12CUBES_FREE', "Hart: 12 Cubes, central freedom", 'T', 12),
    ('HC_24CUBES', "Hart: 24 Cubes, central freedom", 'O', 24),
    ('HC_60CUBES', "Hart: 60 Cubes, central freedom", 'I', 60),
]

_FREE_BY_KEY = {r[0]: r for r in FREE_COMPOUNDS}


INSCRIBED = [
    ('INS_TETRA_CUBE', "Tetrahedron Inscribed in Cube"),
    ('INS_2TETRA_CUBE', "Two Tetrahedra Inscribed in Cube"),
    ('INS_CUBE_DODECA', "Cube Inscribed in Dodecahedron"),
    ('INS_TETRA_DODECA', "Tetrahedron Inscribed in Dodecahedron"),
    ('INS_ICOSA_OCTA', "Icosahedron Inscribed in Octahedron"),
    ('INS_2ICOSA_OCTA', "Two Icosahedra Inscribed in Octahedron"),
]


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

    # --- Skilling's Table 1, the prism/antiprism-in-polyhedral band ----
    # Entries 26-45 of Skilling (1976) are prism symmetry embedded in
    # octahedral or icosahedral symmetry, which is exactly the axis rule
    # above -- so they are rows, not new machinery.  The subset here is
    # the one whose constituents are ORDINARY prisms and antiprisms; the
    # rest of the band needs star prisms (5/2, 10/3), which the component
    # table does not yet carry.
    #
    # Every count is Skilling's own, read off Table 1, and asserted.  The
    # constituent count is the real check on a row: it is |G| divided by
    # the stabilizer the placement actually achieves, so a mis-phased row
    # comes out at twice the right number rather than merely looking odd.
    ('S26_5ANTI', "Skilling 26: 12 Pentagonal Antiprisms (free)",
     'ANTI5', 'Ih', 5, 5, 9.0, 12),
    ('S27_5ANTI', "Skilling 27: 6 Pentagonal Antiprisms",
     'ANTI5', 'Ih', 5, 5, PERP, 6),
    ('S30_3PRISM', "Skilling 30: 4 Triangular Prisms (O)",
     'PRISM3', 'O', 3, 3, PERP, 4),
    ('S31_3PRISM', "Skilling 31: 8 Triangular Prisms (Oh)",
     'PRISM3', 'Oh', 3, 3, PERP, 8),
    ('S32_3PRISM', "Skilling 32: 10 Triangular Prisms (I)",
     'PRISM3', 'I', 3, 3, PERP, 10),
    ('S33_3PRISM', "Skilling 33: 20 Triangular Prisms (Ih)",
     'PRISM3', 'Ih', 3, 3, PERP, 20),
    ('S34_5PRISM', "Skilling 34: 6 Pentagonal Prisms (I)",
     'PRISM5', 'I', 5, 5, PERP, 6),
    ('S35_5PRISM', "Skilling 35: 12 Pentagonal Prisms (Ih)",
     'PRISM5', 'Ih', 5, 5, PERP, 12),
    ('S38_6PRISM', "Skilling 38: 4 Hexagonal Prisms (Oh)",
     'PRISM6', 'Oh', 6, 3, PERP, 4),
    ('S39_6PRISM', "Skilling 39: 10 Hexagonal Prisms (Ih)",
     'PRISM6', 'Ih', 6, 3, PERP, 10),
    ('S40_10PRISM', "Skilling 40: 6 Decagonal Prisms (Ih)",
     'PRISM10', 'Ih', 10, 5, PERP, 6),
    ('S42_4ANTI', "Skilling 42: 3 Square Antiprisms (O)",
     'ANTI4', 'O', 4, 4, PERP, 3),
    ('S43_4ANTI', "Skilling 43: 6 Square Antiprisms (Oh)",
     'ANTI4', 'Oh', 4, 4, PERP, 6),
    ('S36_5_2PRISM', "Skilling 36: 6 Pentagrammic Prisms (I)",
     'PRISM5_2', 'I', 5, 5, PERP, 6),
    ('S37_5_2PRISM', "Skilling 37: 12 Pentagrammic Prisms (Ih)",
     'PRISM5_2', 'Ih', 5, 5, PERP, 12),
    ('S41_10_3PRISM', "Skilling 41: 6 Decagrammic Prisms (Ih)",
     'PRISM10_3', 'Ih', 10, 5, PERP, 6),
    # --- Skilling's Table 1, entries 1-19 ------------------------------
    # "Miscellaneous", but really one idea: add symmetry to a solid
    # generated by S4 (tetrahedra), C4h (cubes) or S6 (octahedra).  Those
    # are rotoreflection subgroups, so the placement is the axis rule
    # with a free turn, and the special cases are particular angles.
    #
    # Several of the band already ship under their classical names and
    # are NOT duplicated here: 4 is the stella octangula (`STELLA`), 5
    # and 6 the five and ten tetrahedra, 7 and 8 the six and three cubes
    # (`HC_6CUBES_C4`, `HC_3CUBES`), 9 the five cubes, 17 the five
    # octahedra.
    ('S01_TETRA', "Skilling 1: 6 Tetrahedra, Td (free)",
     'TETRA', 'Td', 2, 2, 11.0, 6),
    ('S02_TETRA', "Skilling 2: 12 Tetrahedra, Oh (free)",
     'TETRA', 'Oh', 2, 4, 11.0, 12),
    ('S03_TETRA', "Skilling 3: 6 Tetrahedra, Oh", 'TETRA', 'Oh', 2, 4,
     45.0, 6),
    ('S10_OCTA', "Skilling 10: 4 Octahedra, Th (free)",
     'OCTA', 'Th', 3, 3, 11.0, 4),
    ('S11_OCTA', "Skilling 11: 8 Octahedra, Oh (free)",
     'OCTA', 'Oh', 3, 3, 11.0, 8),
    ('S12_OCTA', "Skilling 12: 4 Octahedra, Oh", 'OCTA', 'Oh', 3, 3,
     60.0, 4),
    ('S13_OCTA', "Skilling 13: 20 Octahedra, Ih (free)",
     'OCTA', 'Ih', 3, 3, 11.0, 20),
    # Entries 14, 15 and 16 are the special positions of 13, and all
    # three are the SAME geometric condition: the octahedron's vertices
    # land on icosahedral mirror planes, which is what Skilling's own
    # footnotes to 15 and 16 say.  `vertex_mirror_phases` solves for
    # those turns in closed form and returns six, of which
    #
    #   [0]  0.000000   5 octahedra           entry 17
    #   [1] 22.238756  10 octahedra           entry 15, figure 5(a)
    #   [2] 44.477512   5 octahedra           entry 17 again
    #   [3] 67.908424  20, two per vertex     entry 14
    #   [4] 82.238756  10 octahedra           entry 16, figure 5(b)
    #   [5] 96.569089  20, two per vertex     entry 14 again
    #
    # so the rows below select among derived turns rather than carrying
    # fitted constants.  15 and 16 are genuinely different compounds and
    # not one solid turned: they share not a single vertex, which is the
    # check that stands in for reading figure 5.  Entry 15 keeps its
    # older, independent derivation via the half-turn axis, and the
    # self-test asserts the two agree -- two routes to one angle being
    # worth more than either alone.
    ('S14_OCTA', "Skilling 14: 20 Octahedra, Ih (two per vertex)",
     'OCTA', 'Ih', 3, 3, vertex_mirror_phases('OCTA', 'Ih', 3, 3)[3], 20),
    ('S15_OCTA', "Skilling 15: 10 Octahedra, Ih", 'OCTA', 'Ih', 3, 3,
     perp_phase_from('OCTA', 'Ih', 3, 3,
                     perp_half_turn_axis('OCTA', 3)), 10),
    ('S16_OCTA', "Skilling 16: 10 Octahedra, Ih (second orientation)",
     'OCTA', 'Ih', 3, 3, vertex_mirror_phases('OCTA', 'Ih', 3, 3)[4], 10),
    # Entry 19, the one compound Skilling says CANNOT be had by adding
    # symmetry to a group in which the constituent is uniform.  His own
    # paragraph is the construction: each tetrahemihexahedron has one of
    # its four triangles normal to an icosahedral three-fold axis, so the
    # symmetry actually used is that C_3 -- order 3, whence 60/3 = 20
    # copies -- and in it the constituent is only BI-uniform, its six
    # vertices splitting into those on that triangle and those not.
    # Uniformity comes back because the vertices coincide in pairs, and
    # THAT is what fixes the free turn: see `vertex_pairing_phases`.
    #
    # Two of its roots give 20 copies, and they are the two
    # enantiomorphous forms Skilling says the compound has, the group
    # being I and not I_h -- one is exactly the mirror image of the
    # other, sharing all 60 squares and not one of the 80 triangles.
    #
    # The construction is confirmed by something outside it: he records
    # that this compound's 60 vertices and 240 edges are shared with the
    # compound of 20 octahedra, and normalised, entry 19's skeleton and
    # entry 14's agree exactly -- two different constituents, two
    # different groups (I against I_h) and two different angle rules
    # landing on one skeleton.
    ('S19_TETRAHEMIHEX',
     "Skilling 19: 20 Tetrahemihexahedra, I (two per vertex)",
     'U4', 'I', 3, 3, vertex_pairing_phases('U4', 'I', 3, 3)[-2], 20),

    # --- compounds of the other regulars ------------------------------
    # Reachable only after the dodecahedron's five-fold axis was
    # corrected above; with the old (0, 1, PHI) the five-fold rows just
    # produced 60 loose copies.  The phases are the candidate angles that
    # put a constituent half-turn axis on a group one, chosen by the
    # resulting count.
    ('H_6DODECA', "6 Dodecahedra", 'DODECA', 'I', 5, 5, 72.0, 6),
    ('H_5ICOSA', "5 Icosahedra", 'ICOSA', 'I', 2, 2, 90.0, 5),
    ('H_6ICOSA', "6 Icosahedra", 'ICOSA', 'I', 5, 5, 36.0, 6),
    ('H_10ICOSA', "10 Icosahedra", 'ICOSA', 'I', 3, 3, 60.0, 10),
    # 104.4775... = the perpendicular-alignment angle PLUS 60.  The bare
    # angle is the fully aligned position, where all ten copies coincide
    # and one dodecahedron comes back; a further 60 degrees is a distinct
    # placement because the dodecahedron's period about a three-fold axis
    # is 120, not 60.  Same shape of trap as the cube's D4 row.
    ('H_10DODECA', "10 Dodecahedra", 'DODECA', 'I', 3, 3,
     perp_phase_from('DODECA', 'I', 3, 3,
                     perp_half_turn_axis('DODECA', 3)) + 60.0, 10),

    # --- Hart's compounds of cubes ------------------------------------
    # Hart labels these "count | G x I / H x I", which IS the
    # subgroup-embedding rule: constituent placed so H is a subgroup of
    # both its own symmetry and the compound's, count = |G| / |H|.  The
    # C_n rows are the axis rule with a free turn (Hart's "rotational
    # freedom"); the D_n rows are the same with the turn locked to where
    # a cube two-fold also lands on a group two-fold.
    #
    # Those locked angles are NOT what perpendicular_phase() returns.  It
    # measures from the component's +x, which is a half-turn axis for a
    # prism but not for a cube on its three-fold axis, and its
    # smallest-candidate rule lands on the FULLY aligned position, where
    # the stabilizer is the whole group and the compound collapses to a
    # single cube.  Each angle below was found by scanning and is checked
    # against Hart's own count.
    #
    # THE FREE ROWS NOW SIT AT HART'S OWN ANGLE.  A row with rotational
    # freedom has no canonical turn, and these carried 11.0 degrees --
    # picked only to be generic, i.e. to be far from any special
    # position.  Generic is exactly what looks WRONG: the cubes land at
    # no relation to each other and the compound reads as a heap.  Hart
    # models each family at a chosen angle, so the angles below are his,
    # recovered from the X3D models in `research/data/hart/x3d` by
    # solving for the turn whose compound has the same set of pairwise
    # angles between cube axes as his.  Each is confirmed a second way,
    # against the full vertex set: same count, and the same
    # rotation-invariant distance spectrum to 2e-3.
    #
    # The named (D_n) rows already agreed with Hart before this and are
    # unchanged -- 3 Cubes with cubes_S4_D4, 4 Cubes with cubes_S4_D3,
    # 6 Cubes 2-fold with cubes_S4_D2, 15 Cubes with cubes_A5_D2 -- which
    # is the check that the CONSTRUCTION was right all along and only the
    # free angles were arbitrary.  (That data is gitignored, so this is
    # recorded here rather than left to a test that cannot run.)
    ('HC_6CUBES_C4', "Hart: 6 Cubes, 4-fold (free)",
     'CUBE', 'O', 4, 4, 27.8171, 6),              # cubes_S4_C4
    ('HC_3CUBES', "Hart: 3 Cubes", 'CUBE', 'O', 4, 4, 45.0, 3),
    ('HC_8CUBES', "Hart: 8 Cubes (free)",
     'CUBE', 'O', 3, 3, 48.5451, 8),              # cubes_S4_C3
    ('HC_4CUBES', "Hart: 4 Cubes (Bakos')", 'CUBE', 'O', 3, 3, 60.0, 4),
    ('HC_12CUBES_C2', "Hart: 12 Cubes (free)",
     'CUBE', 'O', 2, 2, 28.6480, 12),             # cubes_S4_C2_A
    ('HC_6CUBES_D2', "Hart: 6 Cubes, 2-fold", 'CUBE', 'O', 2, 2, 90.0, 6),
    ('HC_20CUBES', "Hart: 20 Cubes (free)",
     'CUBE', 'I', 3, 3, 68.1469, 20),             # cubes_A5_C3
    ('HC_30CUBES_C2', "Hart: 30 Cubes (free)",
     'CUBE', 'I', 2, 2, 129.2706, 30),            # cubes_A5_C2_A_alt1
    ('HC_15CUBES', "Hart: 15 Cubes", 'CUBE', 'I', 2, 2, 45.0, 15),
    # 22.2388..., computed rather than written down: it is the turn about
    # a three-fold axis carrying a cube EDGE half-turn axis onto an
    # icosahedral one.  A 0.05-degree scan steps straight over it and
    # reports the row unreachable -- the same 0.2388 fraction that made
    # the icosahedral prism rows look impossible.
    ('HC_10CUBES', "Hart: 10 Cubes", 'CUBE', 'I', 3, 3,
     perp_phase_from('CUBE', 'I', 3, 3, (1.0, -1.0, 0.0)), 10),
    # and it agrees with Hart's cubes_A5_D3_b (not _a: he models both
    # embeddings of D_3, and this is the second).
    ('HC_4CUBES_T', "Hart: 4 Cubes, tetrahedral (free)",
     'CUBE', 'T', 3, 3, 34.3833, 4),              # cubes_A4_C3
    ('HC_6CUBES_T', "Hart: 6 Cubes, tetrahedral (free)",
     'CUBE', 'T', 2, 2, 22.0834, 6),              # cubes_A4_C2

    ('S28_5_3ANTI', "Skilling 28: 12 Pentagrammic Crossed Antiprisms (free)",
     'ANTI5_3', 'Ih', 5, 5, 9.0, 12),
    ('S29_5_3ANTI', "Skilling 29: 6 Pentagrammic Crossed Antiprisms",
     'ANTI5_3', 'Ih', 5, 5, 18.0, 6),
    ('S44_5_2ANTI', "Skilling 44: 6 Pentagrammic Antiprisms (I)",
     'ANTI5_2', 'I', 5, 5,
     perp_phase_from('ANTI5_2', 'I', 5, 5,
                     perp_half_turn_axis('ANTI5_2', 5)), 6),
    ('S45_5_2ANTI', "Skilling 45: 12 Pentagrammic Antiprisms (Ih)",
     'ANTI5_2', 'Ih', 5, 5,
     perp_phase_from('ANTI5_2', 'Ih', 5, 5,
                     perp_half_turn_axis('ANTI5_2', 5)), 12),
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
    ('PRISM_DUAL', "Prism + Dual Dipyramid"),
    ('ANTIPRISM_DUAL', "Antiprism + Dual Trapezohedron"),
] + list(FAMILY_LABELS) + list(INSCRIBED) \
    + [(k, lbl) for k, lbl, *_rest in ENANTIOMORPHS] \
    + [(k, lbl) for k, lbl, *_rest in FREE_COMPOUNDS]     + [(k, lbl) for k, lbl, *_rest in SUBGROUP_COMPOUNDS] \
    + [(k, lbl) for k, lbl, *_rest in AXIS_COMPOUNDS]

#: The families the dropdown is grouped into, in menu order.  The list
#: had grown to 117 entries in the order the source tables happened to be
#: concatenated, which put Skilling 62, 67, 18 and 48 in consecutive
#: columns and made the thing unreadable.
#:
#: The five Skilling bands are HIS OWN, printed at the head of Table 1
#: ("1-19 miscellaneous, 20-25 prism symmetry embedded in prism
#: symmetry, ...") -- so the grouping is the paper's rather than one
#: invented here, and a row lands in its band by its own number.
_GROUP_ORDER = [
    ('Classical compounds',
     ['STELLA', '5TETRA', '10TETRA', '5CUBES', '5OCTA']),
    ('A solid with its dual',
     ['CUBE_OCTA', 'DODECA_ICOSA', 'PRISM_DUAL', 'ANTIPRISM_DUAL']),
    ('One solid inscribed in another', None),      # filled from INSCRIBED
    ('Regular solids on shared axes', None),       # the H_ rows
    ("Hart's cube compounds", None),               # the HC_ rows
    ('Skilling 1-19: miscellaneous', None),
    ('Skilling 20-25: prism symmetry in prism symmetry', None),
    ('Skilling 26-45: prism symmetry in octahedral or icosahedral', None),
    ('Skilling 46-67: tetrahedral symmetry in octahedral or icosahedral',
     None),
    ('Skilling 68-75: duplication of enantiomorphs', None),
]


#: heading -> the enum key the family selector uses.  Written out rather
#: than slugged from the heading so that rewording a heading cannot
#: silently change a key and invalidate a saved .blend or a script.
_FAMILY_KEYS = {
    'Classical compounds': 'CLASSICAL',
    'A solid with its dual': 'DUAL_PAIR',
    'One solid inscribed in another': 'INSCRIBED',
    'Regular solids on shared axes': 'AXES',
    "Hart's cube compounds": 'HART_CUBES',
    'Skilling 1-19: miscellaneous': 'SK_1_19',
    'Skilling 20-25: prism symmetry in prism symmetry': 'SK_20_25',
    'Skilling 26-45: prism symmetry in octahedral or icosahedral':
        'SK_26_45',
    'Skilling 46-67: tetrahedral symmetry in octahedral or icosahedral':
        'SK_46_67',
    'Skilling 68-75: duplication of enantiomorphs': 'SK_68_75',
}


def compound_families():
    """The families as [(key, heading, [(compound key, label), ...]), ...].

    The two-stage selector's first stage.  Keys come from `_FAMILY_KEYS`
    so that rewording a heading does not invalidate a saved file.
    """
    out = []
    for heading, rows in compound_groups():
        key = _FAMILY_KEYS.get(heading)
        if key is None:                    # the 'Other' catch-all
            key = 'OTHER'
        out.append((key, heading, rows))
    return out


def _skilling_number(label):
    """The Table 1 row a label names, or None."""
    if not label.startswith('Skilling '):
        return None
    head = label[9:].split(':', 1)[0].strip()
    return int(head) if head.isdigit() else None


def compound_groups():
    """The dropdown contents as [(heading, [(key, label), ...]), ...].

    Grouping only -- every key and label is passed through untouched, so
    the object a build is named after still carries its full name.
    """
    by_key = dict(COMPOUNDS)
    inscribed = [k for k, _l in INSCRIBED]
    placed = set()
    out = []
    for heading, fixed in _GROUP_ORDER:
        if fixed is not None:
            rows = [(k, by_key[k]) for k in fixed if k in by_key]
        elif heading.startswith('One solid'):
            rows = [(k, by_key[k]) for k in inscribed if k in by_key]
        elif heading.startswith('Regular solids'):
            rows = [(k, l) for k, l in COMPOUNDS
                    if k.startswith('H_') and _skilling_number(l) is None]
        elif heading.startswith("Hart's cube"):
            rows = [(k, l) for k, l in COMPOUNDS if k.startswith('HC_')]
        else:
            lo, hi = (int(x) for x in
                      heading.split(':')[0].split()[1].split('-'))
            rows = sorted(
                ((k, l) for k, l in COMPOUNDS
                 if lo <= (_skilling_number(l) or -1) <= hi),
                key=lambda kl: _skilling_number(kl[1]))
        rows = [r for r in rows if r[0] not in placed]
        placed.update(k for k, _l in rows)
        if rows:
            out.append((heading, rows))
    # Nothing may go missing: a key that matched no rule would simply
    # vanish from the menu, and the operator would still accept it, so
    # only a count catches it.
    left = [(k, l) for k, l in COMPOUNDS if k not in placed]
    if left:
        out.append(('Other', left))
    return out


_INSCRIBED_KEYS = {k for k, _lbl in INSCRIBED}

#: compounds whose shape is set by the side count rather than a preset
SIDED = {'PRISM_DUAL': False, 'ANTIPRISM_DUAL': True}



def build_compound(kind, phase=None, sides=5, repeat=2):
    """Return a list of components, each (V, F).

    `phase` overrides a named compound's own turn angle, which is what
    turns the rigid presets into Hart's rotational-freedom families:
    away from the named angle the components separate and the count
    generally rises.
    """
    if kind in FAMILIES:
        anti, reflect, star_m = FAMILIES[kind]
        return prism_family(sides, repeat, anti=anti, reflect=reflect,
                            phase=0.0 if phase is None else phase,
                            star_m=star_m)
    if kind in SIDED:
        return prism_and_dual(sides, anti=SIDED[kind])
    if kind in _INSCRIBED_KEYS:
        return inscribed(kind)
    if kind in _ENANT_BY_KEY:
        return enantiomorph_pair(_ENANT_BY_KEY[kind][2])
    if kind in _SUBGROUP_BY_KEY:
        _k, _lbl, comp, grp, _n = _SUBGROUP_BY_KEY[kind]
        return subgroup_compound(comp, grp)
    if kind in _FREE_BY_KEY:
        _k, _lbl, grp, _n = _FREE_BY_KEY[kind]
        return free_orientation_compound(
            'CUBE', grp, phase=17.0 if phase is None else phase)
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
