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


def _with_inversion(rots):
    """A rotation group extended by the centre, giving the full group.

    Needed because Skilling's Table 1 distinguishes O from O_h and I from
    I_h throughout, and the count of constituents differs between them --
    a triangular prism on a three-fold axis gives 4 copies in O and 8 in
    O_h, since O_h contains no mirror perpendicular to a three-fold axis
    and so cannot absorb the prism's own sigma_h.
    """
    return list(rots) + [-M for M in rots]


GROUPS = {
    'T': _tetra_rotations,
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
GROUP_AXES['Th'] = GROUP_AXES['T']
GROUP_AXES['Oh'] = GROUP_AXES['O']
GROUP_AXES['Ih'] = GROUP_AXES['I']

# and of each component solid, in its own frame
COMPONENT_AXES = {
    'TETRA': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0)},
    'CUBE': {2: (1.0, 1.0, 0.0), 3: (1.0, 1.0, 1.0), 4: (0.0, 0.0, 1.0)},
    'OCTA': {2: (1.0, 1.0, 0.0), 3: (1.0, 1.0, 1.0), 4: (0.0, 0.0, 1.0)},
    'DODECA': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0), 5: (0.0, 1.0, PHI)},
    'ICOSA': {2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0), 5: (0.0, 1.0, PHI)},
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
del _n, _m


def _component(kind):
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
    ('S28_5_3ANTI', "Skilling 28: 12 Pentagrammic Crossed Antiprisms (free)",
     'ANTI5_3', 'Ih', 5, 5, 9.0, 12),
    ('S29_5_3ANTI', "Skilling 29: 6 Pentagrammic Crossed Antiprisms",
     'ANTI5_3', 'Ih', 5, 5, 18.0, 6),
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
] + list(INSCRIBED) + [(k, lbl) for k, lbl, *_rest in AXIS_COMPOUNDS]

_INSCRIBED_KEYS = {k for k, _lbl in INSCRIBED}

#: compounds whose shape is set by the side count rather than a preset
SIDED = {'PRISM_DUAL': False, 'ANTIPRISM_DUAL': True}


def build_compound(kind, phase=None, sides=5):
    """Return a list of components, each (V, F).

    `phase` overrides a named compound's own turn angle, which is what
    turns the rigid presets into Hart's rotational-freedom families:
    away from the named angle the components separate and the count
    generally rises.
    """
    if kind in SIDED:
        return prism_and_dual(sides, anti=SIDED[kind])
    if kind in _INSCRIBED_KEYS:
        return inscribed(kind)
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
