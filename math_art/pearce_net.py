# Pearce's Universal Node net -- the coordinate kernel shared by every
# saddle polyhedron of Table 8.1.
#
# THE UNIVERSAL NODE.  Pearce's connector carries branches in 26
# directions of a cubic lattice: the 6 face directions <100>, the 12
# edge directions <110> and the 8 body directions <111>.  Nothing else.
# Every saddle polyhedron in his inventory is a closed circuit-graph
# whose edges are branches of this one net, and every column of Table
# 8.1 -- node valences, branch counts split by direction class, face
# counts, face types, included angles, face-plane directions -- is a
# checksum on that graph.
#
# THE ANGLE THEOREM (verified, not assumed).  The "Included Angles"
# column of Table 8.1 lists exactly angles between pairs of branch
# directions.  Over all C(26,2) = 325 pairs the distinct angles are
#
#     35d16'  cos  sqrt(2/3)   110-111        45d    cos  sqrt(2)/2  100-110
#     54d44'  cos  1/sqrt(3)   100-111        60d    cos  1/2        110-110
#     70d32'  cos  1/3         111-111        90d    cos  0          several
#    109d28'  cos -1/3         111-111       120d    cos -1/2        110-110
#    125d16'  cos -1/sqrt(3)   100-111       135d    cos -sqrt(2)/2  100-110
#    144d44'  cos -sqrt(2/3)   110-111       180d    cos -1          opposite
#
# and every angle Pearce prints is in that set.  Two of them (125d16'
# and 135d) never occur as a face corner in the 53 solids -- a fact
# about which circuits he uses, not a defect of the rule.  The lone
# "69d" in the table (entry 26, a 2-fold 4-gon printed 69d, 90d, 60d,
# 90d) is a misprint for 60d: 2-fold symmetry forces opposite angles
# equal, and 60d is the only legal value that satisfies it.  Angles are
# length-independent, so this needs no length convention at all.
#
# COORDINATES.  Integer EIGHTHS of the conventional cubic cell edge --
# the convention the triamond space-filler already used.  Pearce
# supplies every branch class in a full and a half length (ch. 18), and
# in eighths both are integral:
#
#     class    full            length      half           length
#     <100>    (8,0,0)         1           (4,0,0)        1/2
#     <110>    (4,4,0)         sqrt(2)/2   (2,2,0)        sqrt(2)/4
#     <111>    (4,4,4)         sqrt(3)/2   (2,2,2)        sqrt(3)/4
#
# The full lengths are the nearest-neighbour distances of the simple
# cubic, fcc and bcc lattices respectively, in ratio 1 : sqrt(2)/2 :
# sqrt(3)/2.  Because every branch is an integer vector in eighths,
# a circuit closes exactly when its edge vectors sum to the zero
# integer vector -- exact arithmetic, no tolerance, and the cheapest
# gate there is.
#
# The half-<110> branch (2,2,0) is the edge of the triamond (srs) net
# and the half-<111> branch (2,2,2) is the edge of the diamond net, so
# the classical interstitial-domain solids fall out of the same
# integer lattice as the rest of the table.
#
# References:
# - Peter Pearce, "Structure in Nature is a Strategy for Design", The
#   MIT Press, 1978 (paperback 1990), ch. 8 -- the Universal Node
#   system, saddle polygons and interstitial domains, and Table 8.1's
#   inventory of 53 saddle polyhedra with their node, branch and face
#   specifications.
# - Peter Pearce, ibid., ch. 18 -- the Universal Node connector and
#   its branches in full and half lengths.

from itertools import combinations, product
from math import acos, degrees, gcd, sqrt

import numpy as np

# --------------------------------------------------------------------
# 1.  Branch directions and classes
# --------------------------------------------------------------------

#: primitive integer direction vectors of the three branch classes
_DIR100 = tuple(sorted({tuple(s * (1 if i == k else 0) for i in range(3))
                        for k in range(3) for s in (1, -1)}))
_DIR110 = tuple(sorted({tuple(v) for v in (
    (a, b, 0) for a in (1, -1) for b in (1, -1))}
    | {tuple(v) for v in ((a, 0, b) for a in (1, -1) for b in (1, -1))}
    | {tuple(v) for v in ((0, a, b) for a in (1, -1) for b in (1, -1))}))
_DIR111 = tuple(sorted(product((1, -1), repeat=3)))

#: all 26 Universal Node branch directions, primitive integer vectors
DIRECTIONS = _DIR100 + _DIR110 + _DIR111

CLASS_OF = {}
for _v in _DIR100:
    CLASS_OF[_v] = '100'
for _v in _DIR110:
    CLASS_OF[_v] = '110'
for _v in _DIR111:
    CLASS_OF[_v] = '111'

#: branch vectors in integer eighth-coordinates, full and half length
FULL8 = {'100': 8, '110': 4, '111': 4}
HALF8 = {'100': 4, '110': 2, '111': 2}

CLASSES = ('100', '110', '111')


def primitive(v):
    """Reduce an integer vector to its primitive direction."""
    v = tuple(int(round(x)) for x in v)
    g = 0
    for x in v:
        g = gcd(g, abs(x))
    if g == 0:
        raise ValueError("zero vector has no direction")
    return tuple(x // g for x in v)


def branch_class(v):
    """'100', '110' or '111' -- or None if v is not a branch."""
    return CLASS_OF.get(primitive(v))


def is_branch(v):
    return branch_class(v) is not None


def branch_kind(v):
    """(class, 'FULL'|'HALF') for a branch in eighth-coordinates.

    Raises if the vector is not a Universal Node branch at one of the
    two supplied moduli."""
    c = branch_class(v)
    if c is None:
        raise ValueError("not a branch direction: %r" % (v,))
    p = primitive(v)
    step = abs(v[0] // p[0]) if p[0] else (
        abs(v[1] // p[1]) if p[1] else abs(v[2] // p[2]))
    if step == FULL8[c]:
        return c, 'FULL'
    if step == HALF8[c]:
        return c, 'HALF'
    raise ValueError("branch %r is class %s but modulus %d is neither "
                     "full (%d) nor half (%d)"
                     % (v, c, step, FULL8[c], HALF8[c]))


# --------------------------------------------------------------------
# 2.  Angles between branches
# --------------------------------------------------------------------

def included_angle(u, v):
    """Angle in degrees between two branch vectors (exact cosine)."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    c = float(u @ v) / (float(np.linalg.norm(u)) * float(np.linalg.norm(v)))
    return degrees(acos(max(-1.0, min(1.0, c))))


def dms(deg):
    """(degrees, minutes) rounded the way Pearce prints them."""
    d = int(deg)
    m = int(round((deg - d) * 60.0))
    if m == 60:
        d, m = d + 1, 0
    return d, m


def angle_label(deg):
    """Pearce's printed form, e.g. 54.7356 -> "54d44'"."""
    d, m = dms(deg)
    return "%dd%02d'" % (d, m) if m else "%dd" % d


def angle_set():
    """Every distinct angle between two of the 26 branch directions."""
    seen = {}
    for u, v in combinations(DIRECTIONS, 2):
        a = included_angle(u, v)
        seen.setdefault(angle_label(a), a)
    return dict(sorted(seen.items(), key=lambda kv: kv[1]))


#: the angles Pearce actually prints in Table 8.1 (69d is a misprint
#: for 60d -- see the header).
TABULATED = ("35d16'", "45d", "54d44'", "60d", "70d32'", "90d",
             "109d28'", "120d", "144d44'", "180d")


def angles_match(printed, measured, tol_min=1.0):
    """Does a measured angle agree with a printed one, to the minute?"""
    d, m = dms(measured)
    lab = "%dd%02d'" % (d, m) if m else "%dd" % d
    if lab == printed:
        return True
    # allow a minute of rounding slack on the printed value
    want = _parse_angle(printed)
    return abs(want - measured) <= tol_min / 60.0 + 1e-9


def _parse_angle(lab):
    lab = lab.replace("°", "d").replace("′", "'")
    d, _, rest = lab.partition("d")
    m = rest.strip("'") or "0"
    return float(d) + float(m) / 60.0


# --------------------------------------------------------------------
# 3.  Circuits, closure and face geometry
# --------------------------------------------------------------------

def edge_vectors(loop):
    """Consecutive edge vectors of a closed integer circuit."""
    n = len(loop)
    return [tuple(loop[(i + 1) % n][k] - loop[i][k] for k in range(3))
            for i in range(n)]


def closes(loop):
    """Exact integer closure: do the edge vectors sum to zero?"""
    s = [0, 0, 0]
    for e in edge_vectors(loop):
        for k in range(3):
            s[k] += e[k]
    return s == [0, 0, 0]


def circuit_angles(loop):
    """Included angle at each vertex of a closed circuit, in degrees.

    The angle at vertex i is between the two edges emanating from it,
    i.e. between (prev - v) and (next - v)."""
    n = len(loop)
    out = []
    for i in range(n):
        p = loop[(i - 1) % n]
        v = loop[i]
        q = loop[(i + 1) % n]
        a = tuple(p[k] - v[k] for k in range(3))
        b = tuple(q[k] - v[k] for k in range(3))
        out.append(included_angle(a, b))
    return out


def circuit_branch_classes(loop):
    """Class of every edge of a circuit."""
    return [branch_class(e) for e in edge_vectors(loop)]


def newell_normal(loop):
    """Best-fit (Newell) normal of a skew circuit, unit length."""
    P = np.asarray(loop, float)
    n = len(P)
    nz = np.zeros(3)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        nz[0] += (a[1] - b[1]) * (a[2] + b[2])
        nz[1] += (a[2] - b[2]) * (a[0] + b[0])
        nz[2] += (a[0] - b[0]) * (a[1] + b[1])
    ln = float(np.linalg.norm(nz))
    if ln < 1e-12:
        raise ValueError("degenerate circuit has no normal")
    return nz / ln


def face_plane_class(loop, tol=1e-6):
    """Which branch class the circuit's normal points along.

    Pearce's "Face Plane Directions" column: a face is assigned to
    [100], [110] or [111] according to the direction of its plane's
    normal.  Returns the class string, or None if the normal is not
    along a branch direction."""
    nz = newell_normal(loop)
    for v in DIRECTIONS:
        u = np.asarray(v, float)
        u = u / float(np.linalg.norm(u))
        if float(np.abs(np.abs(u @ nz) - 1.0)) < tol:
            return CLASS_OF[v]
    return None


def is_equilateral_equiangular(loop, tol=1e-9):
    """Does this circuit admit a TRUE spidron nest?

    A spidron nest of congruent triangles needs an equilateral,
    equiangular skew polygon: one edge length and one included angle
    throughout.  This is the exact gate -- no per-solid judgement."""
    P = np.asarray(loop, float)
    n = len(P)
    lens = [float(np.linalg.norm(P[(i + 1) % n] - P[i])) for i in range(n)]
    ang = circuit_angles(loop)
    return (max(lens) - min(lens) <= tol * max(1.0, max(lens))
            and max(ang) - min(ang) <= 1e-6)


# --------------------------------------------------------------------
# 4.  Surface combinatorics
# --------------------------------------------------------------------

def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def edge_counts(faces):
    """edge -> number of faces using it."""
    cnt = {}
    for f in faces:
        n = len(f)
        for i in range(n):
            e = edge_key(f[i], f[(i + 1) % n])
            cnt[e] = cnt.get(e, 0) + 1
    return cnt


def is_closed_surface(faces):
    """Every edge in exactly two faces."""
    cnt = edge_counts(faces)
    return bool(cnt) and all(v == 2 for v in cnt.values())


def euler(verts, faces):
    """(V, E, F, chi) for a face-circuit complex."""
    used = set()
    for f in faces:
        used.update(f)
    E = len(edge_counts(faces))
    V = len(used)
    F = len(faces)
    return V, E, F, V - E + F


def valence_histogram(faces):
    """vertex -> degree in the edge graph, as a {degree: count} map."""
    deg = {}
    for e in edge_counts(faces):
        for v in e:
            deg[v] = deg.get(v, 0) + 1
    hist = {}
    for d in deg.values():
        hist[d] = hist.get(d, 0) + 1
    return hist, deg


def branch_totals(verts, faces):
    """Count the solid's distinct edges by branch class."""
    tot = {c: 0 for c in CLASSES}
    for a, b in edge_counts(faces):
        v = tuple(verts[b][k] - verts[a][k] for k in range(3))
        c = branch_class(v)
        if c is None:
            raise ValueError("edge %r-%r is not a branch: %r" % (a, b, v))
        tot[c] += 1
    return tot


def orientation_consistent(faces):
    """Do the circuits induce opposite directions on every shared edge?

    A consistently oriented closed surface traverses each edge once in
    each direction."""
    seen = {}
    for f in faces:
        n = len(f)
        for i in range(n):
            d = (f[i], f[(i + 1) % n])
            if d in seen:
                return False
            seen[d] = True
    for (a, b) in list(seen):
        if (b, a) not in seen:
            return False
    return True


# --------------------------------------------------------------------
# 5.  Symmetry
# --------------------------------------------------------------------

def polygon_symmetries(loop):
    """(proper, improper, orders) of a closed polygon's own symmetry.

    Rigid maps carrying the vertex cycle onto itself.  Migrated from
    the triamond space-filler, where it established that the n10a
    decagon is equilateral and equiangular yet has only 222 symmetry."""
    P = np.asarray(loop, float)
    m = len(P)
    Q = P - P.mean(axis=0)
    prop = impr = 0
    orders = []
    for s in range(m):
        for rev in (False, True):
            perm = [(s + (-k if rev else k)) % m for k in range(m)]
            B = Q[perm]
            H = Q.T @ B
            U, _S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.abs((R @ Q.T).T - B).max() < 1e-9:
                if np.linalg.det(R) > 0:
                    prop += 1
                    k, M = 1, R
                    while np.abs(M - np.eye(3)).max() > 1e-9:
                        M = M @ R
                        k += 1
                    orders.append(k)
                else:
                    impr += 1
    return prop, impr, orders


def face_symmetry_label(loop):
    """Pearce's per-face symmetry column: '6F'/'4F'/'3F'/'2F',
    'MIRROR' or 'NONE'."""
    prop, impr, orders = polygon_symmetries(loop)
    rot = max([k for k in orders if k > 1], default=1)
    if rot > 1:
        return "%dF" % rot
    return 'MIRROR' if impr else 'NONE'


def cloud_match(A, B, R, tol=1e-6):
    """Does rotation R carry point set A onto point set B?"""
    T = (R @ np.asarray(A, float).T).T
    B = np.asarray(B, float)
    d = np.linalg.norm(T[:, None, :] - B[None, :, :], axis=2)
    return float(d.min(axis=1).max()) < tol and \
        float(d.min(axis=0).max()) < tol


_OCTA = None


def _octahedral_rotations():
    """The 24 proper rotations of the cube, as integer matrices.

    Every solid in Table 8.1 sits in the cubic net, so its rotational
    symmetries are a subgroup of this group -- which makes the axis
    counts of Pearce's second column exactly computable."""
    global _OCTA
    if _OCTA is not None:
        return _OCTA
    mats = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                 (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for sgn in product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = sgn[i]
            if round(float(np.linalg.det(M))) == 1:
                mats.append(M)
    _OCTA = mats
    return mats


def rotation_symmetries(points, tol=1e-6):
    """Proper rotations of the cubic group fixing this point cloud.

    The cloud must already be centred on the solid's centroid."""
    P = np.asarray(points, float)
    P = P - P.mean(axis=0)
    return [R for R in _octahedral_rotations() if cloud_match(P, P, R, tol)]


def _rotation_order(R):
    k, M = 1, R
    while np.abs(M - np.eye(3)).max() > 1e-9:
        M = M @ R
        k += 1
        if k > 12:
            raise ValueError("not a finite rotation")
    return k


#: the 13 rotation axes of the cubic system, as primitive vectors:
#: 3 of <100> (up to 4-fold), 6 of <110> (2-fold), 4 of <111> (3-fold).
CUBIC_AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
              (1, 1, 0), (1, -1, 0), (1, 0, 1),
              (1, 0, -1), (0, 1, 1), (0, 1, -1),
              (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1))


def axis_rotation(axis, k):
    """Rotation by 2*pi/k about `axis`, as a 3x3 matrix."""
    a = np.asarray(axis, float)
    a = a / float(np.linalg.norm(a))
    th = 2.0 * np.pi / k
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return (np.eye(3) + np.sin(th) * K
            + (1.0 - np.cos(th)) * (K @ K))


def axis_counts(points, tol=1e-6):
    """(n2, n3, n4, n6): Pearce's symmetry-axis column.

    Counts AXES, not group elements, and reports each axis at its
    HIGHEST order -- a 4-fold axis also carries a 2-fold rotation but
    is counted once, as 4-fold.  Tested directly against the 13 axes of
    the cubic system rather than by classifying rotation matrices,
    because a 3-fold rotation and its square are two matrices about one
    axis and eigenvector round-off makes them look like two distinct
    axes (which is exactly what an earlier revision did, reporting 6
    three-fold axes for the tetrahedral diamond cell instead of 4).

    No axis can be 6-fold inside the cubic group; a "6-fold" in Table
    8.1 always refers to a FACE's own symmetry, never the solid's."""
    P = np.asarray(points, float)
    P = P - P.mean(axis=0)
    n = {2: 0, 3: 0, 4: 0, 6: 0}
    for ax in CUBIC_AXES:
        best = 1
        for k in (6, 4, 3, 2):
            if cloud_match(P, P, axis_rotation(ax, k), tol):
                best = k
                break
        if best in n:
            n[best] += 1
    return n[2], n[3], n[4], n[6]


def is_chiral(points, tol=1e-6):
    """Does the cloud admit NO improper isometry of the cubic group?"""
    P = np.asarray(points, float)
    P = P - P.mean(axis=0)
    for R in _octahedral_rotations():
        if cloud_match(P, P, -R, tol):
            return False
    return True


# --------------------------------------------------------------------
# 6.  Standard nets, in integer eighth-coordinates
# --------------------------------------------------------------------

#: Wyckoff 8a of I4132 -- the triamond / Laves / srs / (10,3)-a net.
SRS_BASE8 = ((1, 1, 1), (3, 7, 5), (7, 5, 3), (5, 3, 7),
             (5, 5, 5), (7, 3, 1), (3, 1, 7), (1, 7, 3))
SRS_NBR8 = tuple(v for v in (
    (a, b, 0) for a in (2, -2) for b in (2, -2))) + tuple(
    (a, 0, b) for a in (2, -2) for b in (2, -2)) + tuple(
    (0, a, b) for a in (2, -2) for b in (2, -2))

#: the diamond net: two interpenetrating fcc lattices, edge (2,2,2).
DIAMOND_BASE8 = ((0, 0, 0), (4, 4, 0), (4, 0, 4), (0, 4, 4),
                 (2, 2, 2), (6, 6, 2), (6, 2, 6), (2, 6, 6))
DIAMOND_NBR8 = tuple(product((2, -2), repeat=3))

#: the bcc net: corner and body centre, edge (4,4,4).
BCC_BASE8 = ((0, 0, 0), (4, 4, 4))
BCC_NBR8 = tuple(product((4, -4), repeat=3))

#: the fcc net: corners and face centres, edge (4,4,0).
FCC_BASE8 = ((0, 0, 0), (4, 4, 0), (4, 0, 4), (0, 4, 4))
FCC_NBR8 = tuple(v for v in (
    (a, b, 0) for a in (4, -4) for b in (4, -4))) + tuple(
    (a, 0, b) for a in (4, -4) for b in (4, -4)) + tuple(
    (0, a, b) for a in (4, -4) for b in (4, -4))

#: the simple cubic net, edge (8,0,0).
SC_BASE8 = ((0, 0, 0),)
SC_NBR8 = tuple(v for v in (
    (8, 0, 0), (-8, 0, 0), (0, 8, 0), (0, -8, 0), (0, 0, 8), (0, 0, -8)))

NETS = {
    'SRS': (SRS_BASE8, SRS_NBR8),
    'DIAMOND': (DIAMOND_BASE8, DIAMOND_NBR8),
    'BCC': (BCC_BASE8, BCC_NBR8),
    'FCC': (FCC_BASE8, FCC_NBR8),
    'SC': (SC_BASE8, SC_NBR8),
}


def branch_vectors(kinds):
    """Every signed branch vector of the given (class, modulus) kinds.

    `kinds` is an iterable of ('100'|'110'|'111', 'FULL'|'HALF')."""
    out = []
    for cls, mod in kinds:
        step = FULL8[cls] if mod == 'FULL' else HALF8[cls]
        for d in DIRECTIONS:
            if CLASS_OF[d] != cls:
                continue
            out.append(tuple(step * x for x in d))
    return tuple(sorted(set(out)))


def mixed_net(kinds, n=2):
    """A net built from an arbitrary mix of branch classes and moduli.

    This is the actual Universal Net, restricted to the branches a
    given solid uses.  The restriction is what makes it searchable:
    allowing all 26 directions at both moduli gives a node degree of
    52, and enumerating circuits in a graph that dense is hopeless,
    whereas Table 8.1 tells us exactly which classes each solid needs
    -- typically one or two, degree 8 to 20.

    Vertices are generated by breadth-first closure from the origin
    under the allowed branch vectors, so the node set is whatever
    sublattice those branches actually reach (simple cubic for full
    <100>, fcc for full <110>, diamond for half <111>, and so on)."""
    vecs = branch_vectors(kinds)
    if not vecs:
        return [], {}, {}
    lim = 8 * n
    seen = {(0, 0, 0)}
    frontier = [(0, 0, 0)]
    while frontier:
        nxt = []
        for p in frontier:
            for d in vecs:
                q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
                if q in seen:
                    continue
                if not all(0 <= c <= lim for c in q):
                    continue
                seen.add(q)
                nxt.append(q)
        frontier = nxt
    Vi = sorted(seen)
    idx = {p: i for i, p in enumerate(Vi)}
    adj = {p: [q for q in ((p[0] + d[0], p[1] + d[1], p[2] + d[2])
                           for d in vecs) if q in idx] for p in Vi}
    return Vi, idx, adj


def kinds_for_row(branches):
    """Candidate (class, modulus) sets for a Table 8.1 branch column.

    The row says which CLASSES a solid uses but not which modulus, so
    both are offered -- full first, since the classical nets use it."""
    used = [c for c in CLASSES if branches.get(c, 0) > 0]
    if not used:
        return []
    out = []
    for mods in product(('FULL', 'HALF'), repeat=len(used)):
        out.append(tuple(zip(used, mods)))
    return out


def net_chunk(name, n=3):
    """A net over an n^3 block of cells: (verts, index, adjacency).

    Coordinates are integer eighths of the conventional cell edge."""
    base, nbr = NETS[name]
    Vi = []
    for off in product(range(n), repeat=3):
        for b in base:
            Vi.append((b[0] + 8 * off[0], b[1] + 8 * off[1],
                       b[2] + 8 * off[2]))
    idx = {p: i for i, p in enumerate(Vi)}
    adj = {p: [q for q in ((p[0] + d[0], p[1] + d[1], p[2] + d[2])
                           for d in nbr) if q in idx] for p in Vi}
    return Vi, idx, adj


def canonical_ring(cyc):
    """Canonical form of a cyclic index sequence (either direction)."""
    cyc = list(cyc)
    k = cyc.index(min(cyc))
    c1 = tuple(cyc[k:] + cyc[:k])
    c2 = tuple([c1[0]] + list(reversed(c1[1:])))
    return min(c1, c2)


def rings_through(A, B, length, idx, adj):
    """Every circuit of the given length passing through both A and B."""
    rings = set()

    def dfs(path, seen):
        last = path[-1]
        if len(path) == length:
            if A in adj[last] and B in seen:
                rings.add(canonical_ring([idx[p] for p in path]))
            return
        for q in adj[last]:
            if q not in seen:
                dfs(path + [q], seen | {q})

    dfs([A], {A})
    return rings


def shortest_rings_at(A, idx, adj, length):
    """Every circuit of the given length through A."""
    rings = set()

    def dfs(path, seen):
        last = path[-1]
        if len(path) == length:
            if A in adj[last]:
                rings.add(canonical_ring([idx[p] for p in path]))
            return
        for q in adj[last]:
            if q not in seen:
                dfs(path + [q], seen | {q})

    dfs([A], {A})
    return rings


def interstitial_cell(A, B, length, nfaces, idx, adj):
    """The saddle cell spanned between two net vertices.

    Pearce's construction for the classical interstitial domains: take
    the circuits of one length through both A and B; if exactly
    `nfaces` of them exist and they close into a surface, that surface
    is the cell.  Returns the face tuple, or None."""
    R = sorted(rings_through(A, B, length, idx, adj))
    if len(R) != nfaces:
        return None
    if not is_closed_surface(R):
        return None
    return tuple(R)


def all_rings(idx, adj, length, seeds=None):
    """Every circuit of the given length in a net chunk.

    Returned as canonical index tuples.  `seeds` restricts the search
    to circuits through those vertices, which is much faster when only
    the cells around one node are wanted."""
    rings = set()
    for A in (seeds if seeds is not None else idx):
        if A not in adj:
            continue
        ia = idx[A]

        def dfs(path, seen):
            last = path[-1]
            if len(path) == length:
                if A in adj[last]:
                    rings.add(canonical_ring([idx[p] for p in path]))
                return
            for q in adj[last]:
                # only extend through vertices above the seed, so each
                # circuit is found once from its lowest vertex
                if q not in seen and idx[q] > ia:
                    dfs(path + [q], seen | {q})

        dfs([A], {A})
    return rings


def _ring_edges(cyc):
    n = len(cyc)
    return [edge_key(cyc[i], cyc[(i + 1) % n]) for i in range(n)]


def close_surfaces(pool, nfaces, seed=None, limit=64):
    """Closed surfaces assembled from a pool of circuits.

    Grows a face set by repeatedly picking an edge used by exactly one
    chosen face and trying every pool circuit that could pair with it,
    which is how a saddle polyhedron actually closes: each branch is
    shared by exactly two faces.  Yields face tuples of length
    `nfaces`.  This is the generic engine behind Pearce's interstitial
    domains -- the apex-pair recipe works only for the triamond net,
    but every cell in the table closes this way."""
    by_edge = {}
    for r in pool:
        for e in _ring_edges(r):
            by_edge.setdefault(e, []).append(r)

    found = []
    starts = [seed] if seed is not None else sorted(pool)

    def grow(chosen, counts):
        if len(found) >= limit:
            return
        deficient = [e for e, c in counts.items() if c == 1]
        if not deficient:
            if len(chosen) == nfaces:
                found.append(tuple(sorted(chosen)))
            return
        if len(chosen) >= nfaces:
            return
        e = min(deficient)
        for r in by_edge.get(e, ()):
            if r in chosen:
                continue
            es = _ring_edges(r)
            if any(counts.get(x, 0) >= 2 for x in es):
                continue
            for x in es:
                counts[x] = counts.get(x, 0) + 1
            chosen.add(r)
            grow(chosen, counts)
            chosen.discard(r)
            for x in es:
                counts[x] -= 1
                if counts[x] == 0:
                    del counts[x]

    for s in starts:
        c = {}
        for x in _ring_edges(s):
            c[x] = 1
        grow({s}, c)
        if found:
            break
    return found


def find_cell(net, ring_len, nfaces, n=3):
    """The interstitial saddle cell of a net, found by circuit growth.

    Returns (verts, faces) in integer eighth-coordinates, outward
    oriented and compacted, or None."""
    Vi, idx, adj = net_chunk(net, n)
    interior = [p for p in Vi if len(adj[p]) == len(NETS[net][1])
                or len(adj[p]) >= 3]
    pool = all_rings(idx, adj, ring_len,
                     seeds=interior[:max(1, len(interior) // 2)])
    if not pool:
        return None
    for s in sorted(pool):
        got = close_surfaces(pool, nfaces, seed=s, limit=1)
        if got:
            V, F = compact(Vi, got[0])
            return V, orient_faces(V, F)
    return None


def orient_faces(verts, faces):
    """Orient every circuit outward from the solid's centroid.

    Returns faces with each circuit reversed as needed so that its
    Newell normal points away from the centroid, then checks that the
    result is a consistently oriented closed surface."""
    used = sorted({i for f in faces for i in f})
    C = np.asarray([verts[i] for i in used], float).mean(axis=0)
    out = []
    for f in faces:
        loop = [verts[i] for i in f]
        nz = newell_normal(loop)
        mid = np.asarray(loop, float).mean(axis=0)
        out.append(tuple(f) if float(nz @ (mid - C)) > 0
                   else tuple(reversed(f)))
    return tuple(out)


def compact(verts, faces):
    """Drop unused vertices and reindex the circuits."""
    used = sorted({i for f in faces for i in f})
    remap = {old: new for new, old in enumerate(used)}
    V = tuple(tuple(int(x) for x in verts[i]) for i in used)
    F = tuple(tuple(remap[i] for i in f) for f in faces)
    return V, F


# --------------------------------------------------------------------
# 7.  Self-test
# --------------------------------------------------------------------

def _selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-58s %s %s" % (name, "OK" if cond else "BAD", extra))

    print("pearce_net: the Universal Node")
    chk("26 branch directions", len(DIRECTIONS) == 26,
        "%d/%d/%d" % (len(_DIR100), len(_DIR110), len(_DIR111)))
    chk("classes assigned", len(CLASS_OF) == 26)

    # --- the angle theorem -------------------------------------------
    S = angle_set()
    chk("325 direction pairs", len(list(combinations(DIRECTIONS, 2))) == 325)
    chk("12 distinct angles", len(S) == 12, " ".join(S))
    missing = [a for a in TABULATED if a not in S]
    chk("every tabulated angle is a branch-pair angle", not missing,
        "missing %r" % (missing,) if missing else "")
    unused = [a for a in S if a not in TABULATED]
    chk("only 125d16'/135d go unused by Pearce",
        sorted(unused) == sorted(["125d16'", "135d"]), " ".join(unused))
    # exact cosines
    for lab, want in (("70d32'", 1.0 / 3.0), ("109d28'", -1.0 / 3.0),
                      ("54d44'", 1.0 / sqrt(3.0)),
                      ("35d16'", sqrt(2.0 / 3.0)),
                      ("144d44'", -sqrt(2.0 / 3.0))):
        got = np.cos(np.radians(S[lab]))
        chk("cos %-8s exact" % lab, abs(got - want) < 1e-12,
            "%.15f" % got)
    # the 69d misprint: a 2-fold 4-gon cannot have unequal opposite
    # angles, so 69d must be the 60d printed two places later.
    chk("69d is not a branch-pair angle",
        all(abs(v - 69.0) > 0.5 for v in S.values()))

    # --- branch moduli -----------------------------------------------
    chk("full <100> branch", branch_kind((8, 0, 0)) == ('100', 'FULL'))
    chk("half <100> branch", branch_kind((4, 0, 0)) == ('100', 'HALF'))
    chk("full <110> branch", branch_kind((4, 4, 0)) == ('110', 'FULL'))
    chk("half <110> branch (srs edge)",
        branch_kind((2, 2, 0)) == ('110', 'HALF'))
    chk("full <111> branch", branch_kind((4, 4, 4)) == ('111', 'FULL'))
    chk("half <111> branch (diamond edge)",
        branch_kind((2, 2, 2)) == ('111', 'HALF'))
    lens = {c: sqrt(sum(x * x for x in v)) / 8.0
            for c, v in (('100', (8, 0, 0)), ('110', (4, 4, 0)),
                         ('111', (4, 4, 4)))}
    chk("full branch moduli 1 : sqrt2/2 : sqrt3/2",
        abs(lens['100'] - 1.0) < 1e-12
        and abs(lens['110'] - sqrt(2.0) / 2.0) < 1e-12
        and abs(lens['111'] - sqrt(3.0) / 2.0) < 1e-12)

    # --- the decatrihedron, entry 1, straight from the srs net -------
    Vi, idx, adj = net_chunk('SRS', 3)
    deg3 = [p for p in Vi if len(adj[p]) == 3]
    chk("srs net is 3-connected", bool(deg3) and
        all(len(adj[p]) <= 3 for p in Vi),
        "%d of %d interior" % (len(deg3), len(Vi)))
    cell = None
    for p in deg3:
        for s in product((-4, 4), repeat=3):
            q = (p[0] + s[0], p[1] + s[1], p[2] + s[2])
            if q not in idx:
                continue
            c = interstitial_cell(p, q, 10, 3, idx, adj)
            if c is not None:
                cell = (p, q, c)
                break
        if cell:
            break
    chk("decatrihedron found in the srs net", cell is not None)
    if cell:
        V, F = compact(Vi, cell[2])
        F = orient_faces(V, F)
        v, e, f, chi = euler(V, F)
        chk("  V=14 E=15 F=3 chi=2",
            (v, e, f, chi) == (14, 15, 3, 2), "%r" % ((v, e, f, chi),))
        hist, _ = valence_histogram(F)
        chk("  2 primary z3 + 12 secondary z2",
            hist == {3: 2, 2: 12}, "%r" % (hist,))
        chk("  all branches <110>",
            branch_totals(V, F) == {'100': 0, '110': 15, '111': 0},
            "%r" % (branch_totals(V, F),))
        loop = [V[i] for i in F[0]]
        chk("  decagon angles all 120d",
            all(abs(a - 120.0) < 1e-9 for a in circuit_angles(loop)))
        chk("  decagon equilateral + equiangular (true nest)",
            is_equilateral_equiangular(loop))
        chk("  decagon face symmetry 2-fold",
            face_symmetry_label(loop) == '2F')
        chk("  decagon plane direction [110]",
            face_plane_class(loop) == '110')
        chk("  orientation consistent", orientation_consistent(F))
        pts = [V[i] for i in range(len(V))]
        chk("  axes (3 x 2-fold, 1 x 3-fold)",
            axis_counts(pts) == (3, 1, 0, 0), "%r" % (axis_counts(pts),))
        chk("  chiral", is_chiral(pts))

    # --- the diamond tetrahedron, entry 11 ---------------------------
    Vi, idx, adj = net_chunk('DIAMOND', 3)
    deg4 = [p for p in Vi if len(adj[p]) == 4]
    chk("diamond net is 4-connected", bool(deg4))
    cell = find_cell('DIAMOND', 6, 4, n=2)
    chk("diamond tetrahedron found by circuit growth", cell is not None)
    if cell:
        V, F = cell
        v, e, f, chi = euler(V, F)
        chk("  V=10 E=12 F=4 chi=2",
            (v, e, f, chi) == (10, 12, 4, 2), "%r" % ((v, e, f, chi),))
        hist, _ = valence_histogram(F)
        chk("  4 primary z3 + 6 secondary z2",
            hist == {3: 4, 2: 6}, "%r" % (hist,))
        chk("  all branches <111>",
            branch_totals(V, F) == {'100': 0, '110': 0, '111': 12},
            "%r" % (branch_totals(V, F),))
        loop = [V[i] for i in F[0]]
        chk("  hexagon angles all 109d28'",
            all(abs(a - 109.4712206) < 1e-6 for a in circuit_angles(loop)))
        chk("  hexagon is a REGULAR skew hexagon",
            is_equilateral_equiangular(loop)
            and face_symmetry_label(loop) in ('3F', '6F'),
            face_symmetry_label(loop))
        chk("  orientation consistent", orientation_consistent(F))
        pts = [V[i] for i in range(len(V))]
        chk("  axes (3 x 2-fold, 4 x 3-fold) = tetrahedral",
            axis_counts(pts) == (3, 4, 0, 0), "%r" % (axis_counts(pts),))
        chk("  achiral", not is_chiral(pts))

    # --- the bcc tetrahedron, entry 12 -------------------------------
    cell = find_cell('BCC', 4, 4, n=2)
    chk("bcc tetrahedron found", cell is not None)
    if cell:
        V, F = cell
        chk("  V=6 E=8 F=4 chi=2", euler(V, F) == (6, 8, 4, 2),
            "%r" % (euler(V, F),))
        hist, _ = valence_histogram(F)
        chk("  2 primary z4 + 4 secondary z2", hist == {4: 2, 2: 4},
            "%r" % (hist,))
        chk("  8 <111> branches",
            branch_totals(V, F) == {'100': 0, '110': 0, '111': 8})
        loop = [V[i] for i in F[0]]
        chk("  regular 4-gon at 70d32'",
            is_equilateral_equiangular(loop)
            and all(abs(a - 70.5287794) < 1e-6 for a in circuit_angles(loop)))
        chk("  face planes [100]",
            all(face_plane_class([V[i] for i in f]) == '100' for f in F))
        pts = [V[i] for i in range(len(V))]
        chk("  axes (4 x 2-fold, 1 x 4-fold) = tetragonal",
            axis_counts(pts) == (4, 0, 1, 0), "%r" % (axis_counts(pts),))

    print("RESULT:", "OK" if ok else "BAD")
    if not ok:
        raise AssertionError("pearce_net self-test failed")
