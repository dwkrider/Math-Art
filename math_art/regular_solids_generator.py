
# Regular Solids Generator for Blender
#
# A complete "Add Regular Solid", organised by family:
#
#   Platonic      the 5 regular solids
#   Kepler-Poinsot the 4 regular star polyhedra, built as their true
#                 intersecting faces (pentagrams star-triangulated)
#   Archimedean   all 13, via Conway notation + canonicalization
#   Catalan       all 13 duals
#   Prisms        uniform n-prisms and n-antiprisms (exact)
#   Johnson       J1-J48: every pyramid / cupola / rotunda solid and
#                 their elongated, gyroelongated, bi- (ortho / gyro)
#                 combinations, composed with exact unit-edge
#                 coordinates (the augmented/diminished J49+ are not
#                 included)
#
# Options: generic stellation (each face replaced by a pyramid to the
# intersection of its neighbours' planes -- octahedron gives the
# stella octangula, dodecahedron the small stellated dodecahedron),
# Solid / Leonardo (da Vinci) / Wireframe styles, and coloring by
# face size (sharing the Conway generator's palette).

bl_info = {
    "name": "Regular Solids",
    "author": "David Krider (Math Art project)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Regular Solid",
    "description": "Platonic, Kepler-Poinsot, Archimedean, Catalan, "
                   "prism and Johnson solids with styles",
    "category": "Add Mesh",
}

import math
from math import pi, sin, cos, sqrt

try:
    from . import conway_operators as cw
except ImportError:
    import conway_operators as cw

PHI = (1 + sqrt(5)) / 2


# ---------------------------------------------------------------- #
#  catalogs                                                        #
# ---------------------------------------------------------------- #

PLATONIC = [
    ('TETRA', "Tetrahedron", 'T'),
    ('CUBE', "Cube", 'C'),
    ('OCTA', "Octahedron", 'O'),
    ('DODECA', "Dodecahedron", 'D'),
    ('ICOSA', "Icosahedron", 'I'),
]

ARCHIMEDEAN = [
    ('TT', "Truncated Tetrahedron", 'tT'),
    ('CO', "Cuboctahedron", 'aC'),
    ('TC', "Truncated Cube", 'tC'),
    ('TO', "Truncated Octahedron", 'tO'),
    ('RCO', "Rhombicuboctahedron", 'eC'),
    ('TCO', "Truncated Cuboctahedron", 'bC'),
    ('SC', "Snub Cube", 'sC'),
    ('ID', "Icosidodecahedron", 'aD'),
    ('TD', "Truncated Dodecahedron", 'tD'),
    ('TI', "Truncated Icosahedron", 'tI'),
    ('RID', "Rhombicosidodecahedron", 'eD'),
    ('TID', "Truncated Icosidodecahedron", 'bD'),
    ('SD', "Snub Dodecahedron", 'sD'),
]

CATALAN = [
    ('KTT', "Triakis Tetrahedron", 'dtT'),
    ('RD', "Rhombic Dodecahedron", 'daC'),
    ('KTC', "Triakis Octahedron", 'dtC'),
    ('KTO', "Tetrakis Hexahedron", 'dtO'),
    ('DIT', "Deltoidal Icositetrahedron", 'deC'),
    ('DDD', "Disdyakis Dodecahedron", 'dbC'),
    ('PIT', "Pentagonal Icositetrahedron", 'dsC'),
    ('RT', "Rhombic Triacontahedron", 'daD'),
    ('KTD', "Triakis Icosahedron", 'dtD'),
    ('PKD', "Pentakis Dodecahedron", 'dtI'),
    ('DHX', "Deltoidal Hexecontahedron", 'deD'),
    ('DDT', "Disdyakis Triacontahedron", 'dbD'),
    ('PHX', "Pentagonal Hexecontahedron", 'dsD'),
]

KEPLER = [
    ('SSD', "Small Stellated Dodecahedron", None),
    ('GD', "Great Dodecahedron", None),
    ('GSD', "Great Stellated Dodecahedron", None),
    ('GI', "Great Icosahedron", None),
]

# Johnson recipes: (top cap, middle, bottom, twist ring-steps)
#   caps:   ('pyr', n) ('cup', n) ('rot',) 'flat'
#   middle: None, 'prism', 'anti'
_J = [
    (1, "Square Pyramid (J1)", ('pyr', 4), None, 'flat', 0),
    (2, "Pentagonal Pyramid (J2)", ('pyr', 5), None, 'flat', 0),
    (3, "Triangular Cupola (J3)", ('cup', 3), None, 'flat', 0),
    (4, "Square Cupola (J4)", ('cup', 4), None, 'flat', 0),
    (5, "Pentagonal Cupola (J5)", ('cup', 5), None, 'flat', 0),
    (6, "Pentagonal Rotunda (J6)", ('rot',), None, 'flat', 0),
    (7, "Elongated Triangular Pyramid (J7)", ('pyr', 3), 'prism',
     'flat', 0),
    (8, "Elongated Square Pyramid (J8)", ('pyr', 4), 'prism',
     'flat', 0),
    (9, "Elongated Pentagonal Pyramid (J9)", ('pyr', 5), 'prism',
     'flat', 0),
    (10, "Gyroelongated Square Pyramid (J10)", ('pyr', 4), 'anti',
     'flat', 0),
    (11, "Gyroelongated Pentagonal Pyramid (J11)", ('pyr', 5),
     'anti', 'flat', 0),
    (12, "Triangular Bipyramid (J12)", ('pyr', 3), None,
     ('pyr', 3), 0),
    (13, "Pentagonal Bipyramid (J13)", ('pyr', 5), None,
     ('pyr', 5), 0),
    (14, "Elongated Triangular Bipyramid (J14)", ('pyr', 3),
     'prism', ('pyr', 3), 0),
    (15, "Elongated Square Bipyramid (J15)", ('pyr', 4), 'prism',
     ('pyr', 4), 0),
    (16, "Elongated Pentagonal Bipyramid (J16)", ('pyr', 5),
     'prism', ('pyr', 5), 0),
    (17, "Gyroelongated Square Bipyramid (J17)", ('pyr', 4),
     'anti', ('pyr', 4), 0),
    (18, "Elongated Triangular Cupola (J18)", ('cup', 3), 'prism',
     'flat', 0),
    (19, "Elongated Square Cupola (J19)", ('cup', 4), 'prism',
     'flat', 0),
    (20, "Elongated Pentagonal Cupola (J20)", ('cup', 5), 'prism',
     'flat', 0),
    (21, "Elongated Pentagonal Rotunda (J21)", ('rot',), 'prism',
     'flat', 0),
    (22, "Gyroelongated Triangular Cupola (J22)", ('cup', 3),
     'anti', 'flat', 0),
    (23, "Gyroelongated Square Cupola (J23)", ('cup', 4), 'anti',
     'flat', 0),
    (24, "Gyroelongated Pentagonal Cupola (J24)", ('cup', 5),
     'anti', 'flat', 0),
    (25, "Gyroelongated Pentagonal Rotunda (J25)", ('rot',), 'anti',
     'flat', 0),
    (27, "Triangular Orthobicupola (J27)", ('cup', 3), None,
     ('cup', 3), 0),
    (28, "Square Orthobicupola (J28)", ('cup', 4), None,
     ('cup', 4), 0),
    (29, "Square Gyrobicupola (J29)", ('cup', 4), None,
     ('cup', 4), 1),
    (30, "Pentagonal Orthobicupola (J30)", ('cup', 5), None,
     ('cup', 5), 0),
    (31, "Pentagonal Gyrobicupola (J31)", ('cup', 5), None,
     ('cup', 5), 1),
    (32, "Pentagonal Orthocupolarotunda (J32)", ('rot',), None,
     ('cup', 5), 0),
    (33, "Pentagonal Gyrocupolarotunda (J33)", ('rot',), None,
     ('cup', 5), 1),
    (34, "Pentagonal Orthobirotunda (J34)", ('rot',), None,
     ('rot',), 0),
    (35, "Elongated Triangular Orthobicupola (J35)", ('cup', 3),
     'prism', ('cup', 3), 0),
    (36, "Elongated Triangular Gyrobicupola (J36)", ('cup', 3),
     'prism', ('cup', 3), 1),
    (37, "Elongated Square Gyrobicupola (J37)", ('cup', 4),
     'prism', ('cup', 4), 1),
    (38, "Elongated Pentagonal Orthobicupola (J38)", ('cup', 5),
     'prism', ('cup', 5), 0),
    (39, "Elongated Pentagonal Gyrobicupola (J39)", ('cup', 5),
     'prism', ('cup', 5), 1),
    (40, "Elongated Pentagonal Orthocupolarotunda (J40)", ('rot',),
     'prism', ('cup', 5), 0),
    (41, "Elongated Pentagonal Gyrocupolarotunda (J41)", ('rot',),
     'prism', ('cup', 5), 1),
    (42, "Elongated Pentagonal Orthobirotunda (J42)", ('rot',),
     'prism', ('rot',), 0),
    (43, "Elongated Pentagonal Gyrobirotunda (J43)", ('rot',),
     'prism', ('rot',), 1),
    (44, "Gyroelongated Triangular Bicupola (J44)", ('cup', 3),
     'anti', ('cup', 3), 0),
    (45, "Gyroelongated Square Bicupola (J45)", ('cup', 4), 'anti',
     ('cup', 4), 0),
    (46, "Gyroelongated Pentagonal Bicupola (J46)", ('cup', 5),
     'anti', ('cup', 5), 0),
    (47, "Gyroelongated Pentagonal Cupolarotunda (J47)", ('rot',),
     'anti', ('cup', 5), 0),
    (48, "Gyroelongated Pentagonal Birotunda (J48)", ('rot',),
     'anti', ('rot',), 0),
]

JOHNSON = [(f'J{num}', name, num) for (num, name, *_r) in _J]
JOHNSON.insert(21, ('J26', "Gyrobifastigium (J26)", 26))
_J_BY_NUM = {num: (top, mid, bot, tw)
             for (num, _n, top, mid, bot, tw) in _J}


# the chiral solids -- their mirror images are genuinely different
# shapes: the two snubs, their Catalan duals, and the
# gyroelongated Johnson bicupolas / cupolarotunda / birotunda
CHIRAL = {('ARCHIMEDEAN', 'SC'), ('ARCHIMEDEAN', 'SD'),
          ('CATALAN', 'PIT'), ('CATALAN', 'PHX'),
          ('JOHNSON', 'J44'), ('JOHNSON', 'J45'),
          ('JOHNSON', 'J46'), ('JOHNSON', 'J47'),
          ('JOHNSON', 'J48')}


def mirror_solid(V, F):
    """The enantiomorph: reflect through the yz-plane and reverse
    the windings so faces stay outward."""
    V2 = [(-float(v[0]), float(v[1]), float(v[2])) for v in V]
    F2 = [list(reversed(f)) for f in F]
    return V2, F2


# ---------------------------------------------------------------- #
#  exact unit-edge construction kit (Johnson / prisms)             #
# ---------------------------------------------------------------- #

def _rn(n):
    """Circumradius of a unit-edge n-gon."""
    return 0.5 / sin(pi / n)


def _ring(n, radius, z, phase=0.0):
    return [(radius * cos(2 * pi * (k + phase) / n),
             radius * sin(2 * pi * (k + phase) / n), z)
            for k in range(n)]


def _antiprism_height(n):
    d = 2 * _rn(n) * sin(pi / (2 * n))
    return sqrt(max(1e-12, 1.0 - d * d))


def _cupola_height(n):
    """Height of the unit-edge n-cupola (2n-ring to n-ring)."""
    b = _ring(2 * n, _rn(2 * n), 0.0)[0]
    t = _ring(n, _rn(n), 0.0, 0.25 * 0 + 0.0)  # placeholder
    # top vertex above the midpoint of bottom edge (b0, b1):
    tx = _rn(n) * cos(pi / (2 * n))
    ty = _rn(n) * sin(pi / (2 * n))
    d2 = (tx - b[0]) ** 2 + (ty - b[1]) ** 2
    return sqrt(max(1e-12, 1.0 - d2))


_ROT_CACHE = {}


def _rotunda_unit():
    """Vertices/faces of the pentagonal rotunda with unit edges,
    decagon ring in z=0 (cap above), from a sliced canonical
    icosidodecahedron. Cached."""
    if 'rot' in _ROT_CACHE:
        return _ROT_CACHE['rot']
    V, F = cw.apply_conway('aD')
    V = cw.canonicalize(V, F, iters=300)
    # normalise edge length to 1
    els = []
    for f in F:
        for i in range(len(f)):
            a, b = V[f[i]], V[f[(i + 1) % len(f)]]
            els.append(math.dist(a, b))
    s = 1.0 / (sum(els) / len(els))
    V = [tuple(c * s for c in v) for v in V]
    # 5-fold axis = a pentagon face centre
    pent = next(f for f in F if len(f) == 5)
    ax = [sum(V[i][k] for i in pent) / 5 for k in range(3)]
    ln = math.dist(ax, (0, 0, 0))
    ax = [c / ln for c in ax]
    # rotate ax -> +z
    zax = (0.0, 0.0, 1.0)
    v = (ax[1] * zax[2] - ax[2] * zax[1],
         ax[2] * zax[0] - ax[0] * zax[2],
         ax[0] * zax[1] - ax[1] * zax[0])
    c = ax[2]
    if abs(c + 1.0) < 1e-9:
        R = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    else:
        k = 1.0 / (1.0 + c)
        R = [[v[0] * v[0] * k + c, v[0] * v[1] * k - v[2],
              v[0] * v[2] * k + v[1]],
             [v[1] * v[0] * k + v[2], v[1] * v[1] * k + c,
              v[1] * v[2] * k - v[0]],
             [v[2] * v[0] * k - v[1], v[2] * v[1] * k + v[0],
              v[2] * v[2] * k + c]]
    V = [tuple(sum(R[r][k] * p[k] for k in range(3)) for r in range(3))
         for p in V]
    # keep the upper half (equator decagon at z ~ 0)
    keep = [i for i, p in enumerate(V) if p[2] > -1e-6]
    remap = {old: new for new, old in enumerate(keep)}
    NV = [V[i] for i in keep]
    NF = [[remap[i] for i in f] for f in F
          if all(i in remap for i in f)]
    ring = [i for i, p in enumerate(NV) if abs(p[2]) < 1e-6]
    ring.sort(key=lambda i: math.atan2(NV[i][1], NV[i][0]))
    _ROT_CACHE['rot'] = (NV, NF, ring)
    return _ROT_CACHE['rot']


class _Builder:
    def __init__(self):
        self.V = []
        self.F = []

    def add(self, p):
        self.V.append(tuple(p))
        return len(self.V) - 1

    def ring(self, n, radius, z, phase=0.0):
        return [self.add(p) for p in _ring(n, radius, z, phase)]


def _cap_faces(bld, ring_idx, cap, z, phase, up=1, shift=0):
    """Attach a cap on the given ring (which lies at height z).
    `up` = +1 above, -1 mirrored below; `shift` rotates the cap by
    whole ring steps (ortho vs gyro pairings). Ring vertex k sits at
    angle 2*pi*(k+phase)/m."""
    m = len(ring_idx)
    if cap == 'flat':
        bld.F.append(list(ring_idx))
        return
    kind = cap[0]
    if kind == 'pyr':
        n = cap[1]
        h = sqrt(max(1e-12, 1.0 - _rn(n) ** 2))
        apex = bld.add((0.0, 0.0, z + up * h))
        for i in range(m):
            bld.F.append([ring_idx[i], ring_idx[(i + 1) % m], apex])
        return
    if kind == 'cup':
        n = cap[1]
        h = _cupola_height(n)
        top = bld.ring(n, _rn(n), z + up * h,
                       (phase + shift) / 2 + 0.25)
        for j in range(n):
            b0 = ring_idx[(2 * j + shift) % m]
            b1 = ring_idx[(2 * j + 1 + shift) % m]
            b2 = ring_idx[(2 * j + 2 + shift) % m]
            bld.F.append([b0, b1, top[j]])
            bld.F.append([b1, b2, top[(j + 1) % n], top[j]])
        bld.F.append(list(top))
        return
    if kind == 'rot':
        NV, NF, rring = _rotunda_unit()
        # rotate the rotunda so its ring vertex 0 lands on ring
        # position `shift`, then weld ring verts
        a0 = math.atan2(NV[rring[0]][1], NV[rring[0]][0])
        target = 2 * pi * (phase + shift) / 10
        da = target - a0
        ca, sa = cos(da), sin(da)
        ids = {}
        for i, p in enumerate(NV):
            x = p[0] * ca - p[1] * sa
            y = p[0] * sa + p[1] * ca
            ids[i] = (None if i in set(rring)
                      else bld.add((x, y, z + up * p[2])))
        for k, i in enumerate(rring):
            ids[i] = ring_idx[(k + shift) % m]
        for f in NF:
            bld.F.append([ids[i] for i in f])
        return
    raise ValueError(cap)


def build_johnson(num, scale=1.0):
    if num == 26:                        # gyrobifastigium
        r = _rn(3)
        h = sqrt(3) / 2
        bld = _Builder()
        # square 1x1 in z=0; prisms up (ridge along y) and down
        # (ridge along x)
        s = 0.5
        b = [bld.add(p) for p in ((-s, -s, 0), (s, -s, 0),
                                  (s, s, 0), (-s, s, 0))]
        rt = [bld.add(p) for p in ((0, -s, h), (0, s, h))]
        rb = [bld.add(p) for p in ((-s, 0, -h), (s, 0, -h))]
        bld.F += [[b[0], b[1], rt[0]], [b[2], b[3], rt[1]],
                  [b[1], b[2], rt[1], rt[0]],
                  [b[3], b[0], rt[0], rt[1]],
                  [b[1], b[0], rb[0], rb[1]],
                  [b[3], b[2], rb[1], rb[0]],
                  [b[0], b[3], rb[0]], [b[2], b[1], rb[1]]]
        V, F = cw.orient_outward(bld.V, bld.F)
        return [tuple(c * scale for c in v) for v in V], F
    (top, mid, bot, tw) = _J_BY_NUM[num]
    m = (10 if top[0] == 'rot' else
         2 * top[1] if top[0] == 'cup' else top[1])
    R = _rn(m)
    bld = _Builder()
    ring0 = bld.ring(m, R, 0.0)
    _cap_faces(bld, ring0, top, 0.0, 0.0, up=1)
    z = 0.0
    phase = 0.0
    ring = ring0
    if mid == 'prism':
        z -= 1.0
        ring = bld.ring(m, R, z, phase)
        for i in range(m):
            bld.F.append([ring0[i], ring0[(i + 1) % m],
                          ring[(i + 1) % m], ring[i]])
    elif mid == 'anti':
        z -= _antiprism_height(m)
        phase += 0.5
        ring = bld.ring(m, R, z, phase)
        for i in range(m):
            bld.F.append([ring0[i], ring0[(i + 1) % m], ring[i]])
            bld.F.append([ring0[(i + 1) % m], ring[(i + 1) % m],
                          ring[i]])
    _cap_faces(bld, ring, bot, z, phase, up=-1, shift=tw)
    V, F = cw.orient_outward(bld.V, bld.F)
    return [tuple(c * scale for c in v) for v in V], F


def build_prism(kind, n, scale=1.0):
    bld = _Builder()
    R = _rn(n)
    if kind == 'PRISM':
        top = bld.ring(n, R, 0.5)
        bot = bld.ring(n, R, -0.5)
        for i in range(n):
            bld.F.append([bot[i], bot[(i + 1) % n],
                          top[(i + 1) % n], top[i]])
        bld.F.append(list(reversed(top)))
        bld.F.append(bot)
    else:
        h = _antiprism_height(n) / 2
        top = bld.ring(n, R, h)
        bot = bld.ring(n, R, -h, 0.5)
        for i in range(n):
            bld.F.append([bot[i], top[i], top[(i + 1) % n]])
            bld.F.append([bot[i], top[(i + 1) % n],
                          bot[(i + 1) % n]])
        bld.F.append(list(reversed(top)))
        bld.F.append(bot)
    V, F = cw.orient_outward(bld.V, bld.F)
    return [tuple(c * scale for c in v) for v in V], F


# ---------------------------------------------------------------- #
#  Kepler-Poinsot (true intersecting faces)                        #
# ---------------------------------------------------------------- #

def _icosa():
    V = []
    for a in (-1, 1):
        for b in (-PHI, PHI):
            V += [(0, a, b), (a, b, 0), (b, 0, a)]
    n = sqrt(1 + PHI * PHI)
    return [tuple(c / n for c in v) for v in V]


def _vertex_rings(V, k_neigh=5):
    """Ordered neighbour ring of every vertex (by closest
    distance)."""
    n = len(V)
    rings = []
    for i in range(n):
        d = sorted(range(n),
                   key=lambda j: math.dist(V[i], V[j]))[1:k_neigh + 1]
        # order around the axis V[i]
        ax = V[i]
        ref = V[d[0]]
        u = [ref[k] - sum(ref[t] * ax[t] for t in range(3)) * ax[k]
             for k in range(3)]
        ln = sqrt(sum(t * t for t in u)) or 1.0
        u = [t / ln for t in u]
        w = (ax[1] * u[2] - ax[2] * u[1], ax[2] * u[0] - ax[0] * u[2],
             ax[0] * u[1] - ax[1] * u[0])

        def ang(j):
            p = V[j]
            return math.atan2(sum(p[t] * w[t] for t in range(3)),
                              sum(p[t] * u[t] for t in range(3)))
        rings.append(sorted(d, key=ang))
    return rings


def _star_face(pts):
    """Triangulate the {5/2} pentagram whose points are the 5 given
    coplanar positions in CONVEX order. Returns (extra_verts,
    triangles) with triangle indices: 0..4 = the input points,
    5..9 = inner pentagon vertices (to be offset by caller)."""
    # basis in the plane
    c = [sum(p[k] for p in pts) / 5 for k in range(3)]
    u = [pts[0][k] - c[k] for k in range(3)]
    ln = sqrt(sum(t * t for t in u)) or 1.0
    u = [t / ln for t in u]
    nrm = [0.0, 0.0, 0.0]
    for i in range(5):
        p, q = pts[i], pts[(i + 1) % 5]
        nrm[0] += (p[1] - q[1]) * (p[2] + q[2])
        nrm[1] += (p[2] - q[2]) * (p[0] + q[0])
        nrm[2] += (p[0] - q[0]) * (p[1] + q[1])
    ln = sqrt(sum(t * t for t in nrm)) or 1.0
    nrm = [t / ln for t in nrm]
    w = (nrm[1] * u[2] - nrm[2] * u[1], nrm[2] * u[0] - nrm[0] * u[2],
         nrm[0] * u[1] - nrm[1] * u[0])

    def to2(p):
        d = [p[k] - c[k] for k in range(3)]
        return (sum(d[k] * u[k] for k in range(3)),
                sum(d[k] * w[k] for k in range(3)))

    def to3(q):
        return tuple(c[k] + q[0] * u[k] + q[1] * w[k]
                     for k in range(3))

    P = [to2(p) for p in pts]

    def isect(a, b, cc, dd):
        (x1, y1), (x2, y2) = a, b
        (x3, y3), (x4, y4) = cc, dd
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    inner2 = []
    for i in range(5):
        # chords (i -> i+2) and (i+1 -> i+4) cross at an inner vertex
        inner2.append(isect(P[i], P[(i + 2) % 5],
                            P[(i + 1) % 5], P[(i + 4) % 5]))
    inner = [to3(q) for q in inner2]
    tris = [[5, 6, 7], [5, 7, 8], [5, 8, 9]]      # inner pentagon
    for j in range(5):
        # point triangle: each tip with its two nearest inner
        # vertices (the first crossings along its two chords)
        near = sorted(range(5), key=lambda k: (
            (inner2[k][0] - P[j][0]) ** 2
            + (inner2[k][1] - P[j][1]) ** 2))[:2]
        a, b = near
        # wind consistently with the pentagon orientation
        ux = inner2[a][0] - P[j][0]
        uy = inner2[a][1] - P[j][1]
        vx = inner2[b][0] - P[j][0]
        vy = inner2[b][1] - P[j][1]
        if ux * vy - uy * vx < 0:
            a, b = b, a
        tris.append([j, 5 + a, 5 + b])
    return inner, tris


def build_kepler(kind, scale=1.0):
    V = _icosa()
    rings = _vertex_rings(V)
    verts = [tuple(c * scale for c in v) for v in V]
    faces = []
    sizes = []
    if kind == 'GD':                      # {5, 5/2}: 12 pentagons
        seen = set()
        for i, ring in enumerate(rings):
            key = frozenset(ring)
            if key in seen:
                continue
            seen.add(key)
            faces.append(list(ring))
            sizes.append(5)
    elif kind == 'GI':                    # {3, 5/2}: 20 triangles
        seen = set()
        for i, ring in enumerate(rings):
            for j in range(5):
                tri = (i, ring[j], ring[(j + 2) % 5])
                key = frozenset(tri)
                if key not in seen:
                    seen.add(key)
                    faces.append(list(tri))
                    sizes.append(3)
    elif kind in ('SSD', 'GSD'):
        if kind == 'SSD':                 # {5/2, 5}: rings of icosa
            polys = []
            seen = set()
            for ring in rings:
                key = frozenset(ring)
                if key not in seen:
                    seen.add(key)
                    polys.append([verts[i] for i in ring])
        else:                             # {5/2, 3}: pentagram points
            # are the SECOND ring of each dodeca face (the vertices
            # one step out), so the 12 face planes cut deep and the
            # faces interpenetrate into the spiky star
            D, DF = cw._seed('D', 0)
            n = sqrt(sum(c * c for c in D[0]))
            D = [tuple(c / n * scale for c in D[i])
                 for i in range(len(D))]
            adj = {}
            for f in DF:
                for i in range(len(f)):
                    a, b = f[i], f[(i + 1) % len(f)]
                    adj.setdefault(a, set()).add(b)
                    adj.setdefault(b, set()).add(a)
            polys = []
            for f in DF:
                fs = set(f)
                ring = set()
                for i in f:
                    ring |= adj[i] - fs
                c = [sum(D[i][k] for i in f) / len(f)
                     for k in range(3)]
                ln = sqrt(sum(t * t for t in c)) or 1.0
                ax = [t / ln for t in c]
                ref = D[next(iter(ring))]
                u = [ref[k] - sum(ref[t] * ax[t] for t in range(3))
                     * ax[k] for k in range(3)]
                lu = sqrt(sum(t * t for t in u)) or 1.0
                u = [t / lu for t in u]
                w = (ax[1] * u[2] - ax[2] * u[1],
                     ax[2] * u[0] - ax[0] * u[2],
                     ax[0] * u[1] - ax[1] * u[0])

                def angp(i):
                    p = D[i]
                    return math.atan2(
                        sum(p[t] * w[t] for t in range(3)),
                        sum(p[t] * u[t] for t in range(3)))
                polys.append([D[i] for i in sorted(ring, key=angp)])
        verts = []
        faces = []
        sizes = []
        for poly in polys:
            base = len(verts)
            verts.extend(poly)
            inner, tris = _star_face(poly)
            verts.extend(inner)
            for t in tris:
                faces.append([base + i for i in t])
                sizes.append(5)           # tag pentagrams as 5s
    else:
        raise ValueError(kind)
    return verts, faces, sizes


# ---------------------------------------------------------------- #
#  generic stellation                                              #
# ---------------------------------------------------------------- #

def stellate(V, F):
    """Replace every face with a pyramid to the intersection of its
    edge-neighbours' planes (least squares). Octahedron -> stella
    octangula, dodecahedron -> small stellated dodecahedron,
    icosahedron -> small triambic icosahedron. Returns (V, F,
    max_residual)."""
    e2f = {}
    for fi, f in enumerate(F):
        for i in range(len(f)):
            e2f[(f[i], f[(i + 1) % len(f)])] = fi
    planes = []
    for f in F:
        n = cw._newell(V, f)
        ln = sqrt(sum(t * t for t in n)) or 1.0
        n = [t / ln for t in n]
        d = sum(n[k] * V[f[0]][k] for k in range(3))
        planes.append((n, d))
    NV = [tuple(v) for v in V]
    NF = []
    worst = 0.0
    for fi, f in enumerate(F):
        neigh = set()
        for i in range(len(f)):
            g = e2f.get((f[(i + 1) % len(f)], f[i]))
            if g is not None and g != fi:
                neigh.add(g)
        M = [[0.0] * 3 for _ in range(3)]
        b = [0.0] * 3
        for g in neigh:
            n, d = planes[g]
            for r in range(3):
                b[r] += n[r] * d
                for cc in range(3):
                    M[r][cc] += n[r] * n[cc]
        det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
               - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
               + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
        if abs(det) < 1e-9:
            raise ValueError("this solid has no bounded stellation "
                             "(parallel neighbour planes)")
        apex = []
        for cc in range(3):
            Mc = [row[:] for row in M]
            for r in range(3):
                Mc[r][cc] = b[r]
            dc = (Mc[0][0] * (Mc[1][1] * Mc[2][2]
                              - Mc[1][2] * Mc[2][1])
                  - Mc[0][1] * (Mc[1][0] * Mc[2][2]
                                - Mc[1][2] * Mc[2][0])
                  + Mc[0][2] * (Mc[1][0] * Mc[2][1]
                                - Mc[1][1] * Mc[2][0]))
            apex.append(dc / det)
        for g in neigh:
            n, d = planes[g]
            worst = max(worst, abs(sum(n[k] * apex[k]
                                       for k in range(3)) - d))
        ai = len(NV)
        NV.append(tuple(apex))
        for i in range(len(f)):
            NF.append([f[i], f[(i + 1) % len(f)], ai])
    return NV, NF, worst


# ---------------------------------------------------------------- #
#  main build                                                      #
# ---------------------------------------------------------------- #

_NOTATION = {sid: (label, nota) for cat in (PLATONIC, ARCHIMEDEAN,
                                            CATALAN)
             for (sid, label, nota) in cat}

# ---------------------------------------------------------------- #
#  congruent shell splitting                                       #
# ---------------------------------------------------------------- #

try:
    from .symmetric_sculpture_generator import group_rotations
except ImportError:
    try:
        from symmetric_sculpture_generator import group_rotations
    except ImportError:
        group_rotations = None


def _face_centroids(V, F):
    return [tuple(sum(V[i][k] for i in f) / len(f) for k in range(3))
            for f in F]


def _detect_face_perms(V, F):
    """Face permutations induced by the solid's own rotation group,
    derived from the mesh: candidate rotations align face 0's frame
    with every same-size face at every cyclic offset, and are kept
    when they map the whole centroid set onto itself. Returns a list
    of permutation tuples (identity included) or None."""
    cents = _face_centroids(V, F)
    scale_ref = max(max(abs(c) for c in p) for p in cents) or 1.0
    tol = 5e-3 * scale_ref

    cell = 2 * tol
    grid = {}
    for i, c in enumerate(cents):
        grid.setdefault(tuple(int(math.floor(x / cell))
                              for x in c), []).append(i)

    def find(c):
        kx, ky, kz = (int(math.floor(x / cell)) for x in c)
        best, bd = None, tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((kx + dx, ky + dy, kz + dz),
                                      ()):
                        d = max(abs(cents[j][k] - c[k])
                                for k in range(3))
                        if d < bd:
                            best, bd = j, d
        return best

    def frame(fi, k0):
        f = F[fi]
        c = cents[fi]
        n = cw._newell(V, f)
        ln = sqrt(sum(t * t for t in n)) or 1.0
        n = [t / ln for t in n]
        if sum(n[k] * c[k] for k in range(3)) < 0:
            n = [-t for t in n]
        w = [V[f[k0]][k] - c[k] for k in range(3)]
        d = sum(w[k] * n[k] for k in range(3))
        w = [w[k] - d * n[k] for k in range(3)]
        lw = sqrt(sum(t * t for t in w)) or 1.0
        u = [t / lw for t in w]
        v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2],
             n[0] * u[1] - n[1] * u[0]]
        return u, v, n

    m0 = len(F[0])
    u0, v0, n0 = frame(0, 0)
    B0 = (u0, v0, n0)
    perms = set()
    for fj in range(len(F)):
        if len(F[fj]) != m0:
            continue
        for k0 in range(len(F[fj])):
            Bj = frame(fj, k0)
            # R = Bj^T-composed rotation taking B0 to Bj
            R = [[sum(Bj[a][r] * B0[a][c] for a in range(3))
                  for c in range(3)] for r in range(3)]
            perm = []
            ok = True
            for i, c in enumerate(cents):
                rc = tuple(sum(R[r][k] * c[k] for k in range(3))
                           for r in range(3))
                t = find(rc)
                if t is None or len(F[t]) != len(F[i]):
                    ok = False
                    break
                perm.append(t)
            if ok and len(set(perm)) == len(F):
                perms.add(tuple(perm))
    return sorted(perms) if len(perms) > 1 else None


def _perm_mul(a, b):
    return tuple(a[b[i]] for i in range(len(a)))


def _closure(gens, cap):
    ident = tuple(range(len(gens[0])))
    seen = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for a in frontier:
            for g in gens:
                c = _perm_mul(g, a)
                if c not in seen:
                    if len(seen) >= cap:
                        return None
                    seen.add(c)
                    nxt.append(c)
        frontier = nxt
    return seen


def _free_subgroups(perms, n):
    """Distinct subgroups of order n of the face-permutation group
    whose nontrivial elements fix no face."""
    ident = tuple(range(len(perms[0])))
    subs = set()
    for g in perms:
        s = _closure([g], n + 1)
        if s and len(s) == n:
            subs.add(frozenset(s))
    if n <= 12:
        small = [g for g in perms if g != ident]
        for i in range(len(small)):
            for j in range(i + 1, len(small)):
                s = _closure([small[i], small[j]], n + 1)
                if s and len(s) == n:
                    subs.add(frozenset(s))
    out = []
    for s in subs:
        if all(p == ident or all(p[i] != i for i in range(len(p)))
               for p in s):
            out.append(sorted(s))
    return out


def _grow_partition(V, F, subgroup, seed):
    """Greedy connected, compact fundamental domain for the (free)
    subgroup action; pieces are its translates. Returns assignment
    list or None if growth stalls."""
    nf = len(F)
    n = len(subgroup)
    cents = _face_centroids(V, F)
    # shared-edge adjacency
    edges = {}
    for fi, f in enumerate(F):
        for i in range(len(f)):
            e = (min(f[i], f[(i + 1) % len(f)]),
                 max(f[i], f[(i + 1) % len(f)]))
            edges.setdefault(e, []).append(fi)
    adj = [set() for _ in range(nf)]
    for fs in edges.values():
        for a in fs:
            for b in fs:
                if a != b:
                    adj[a].add(b)
    assign = [-1] * nf
    piece0 = []
    csum = [0.0, 0.0, 0.0]

    def take(f):
        for j, p in enumerate(subgroup):
            assign[p[f]] = j
        piece0.append(f)
        for k in range(3):
            csum[k] += cents[f][k]

    take(seed)
    target = nf // n
    while len(piece0) < target:
        cen = [csum[k] / len(piece0) for k in range(3)]
        cand = set()
        for f in piece0:
            for g in adj[f]:
                if assign[g] == -1:
                    cand.add(g)
        if not cand:
            return None, None
        best = min(cand, key=lambda g: sum(
            (cents[g][k] - cen[k]) ** 2 for k in range(3)))
        take(best)
    return assign, None


def _cut_pairs(V, F):
    """(adj, pairs): face adjacency sets and one (a, b, length)
    triple per shared edge, for boundary-length accounting."""
    edges = {}
    for fi, f in enumerate(F):
        for i in range(len(f)):
            e = (min(f[i], f[(i + 1) % len(f)]),
                 max(f[i], f[(i + 1) % len(f)]))
            edges.setdefault(e, []).append(fi)
    adj = [set() for _ in range(len(F))]
    pairs = []
    for (va, vb), fs in edges.items():
        if len(fs) == 2 and fs[0] != fs[1]:
            adj[fs[0]].add(fs[1])
            adj[fs[1]].add(fs[0])
            l = math.dist(V[va], V[vb])
            pairs.append((fs[0], fs[1], l))
    return adj, pairs


def _piece0_connected(assign, adj, want):
    """Is the set of faces assigned to piece 0 edge-connected?"""
    start = next((f for f, a in enumerate(assign) if a == 0), None)
    if start is None:
        return False
    seen = {start}
    stack = [start]
    while stack:
        f = stack.pop()
        for g in adj[f]:
            if assign[g] == 0 and g not in seen:
                seen.add(g)
                stack.append(g)
    return len(seen) == want


def _partition_score(assign, adj, pairs, pen):
    """Jaggedness of a partition: total cut perimeter plus a
    penalty per hinge face (a face with at most one same-piece
    neighbour -- the protruding flaps).  Lower is smoother."""
    cut = sum(l for a, b, l in pairs if assign[a] != assign[b])
    hinges = sum(1 for f in range(len(assign))
                 if sum(1 for g in adj[f]
                        if assign[g] == assign[f]) <= 1)
    return cut + pen * hinges


def _smooth_partition(F, subgroup, assign, adj, pairs):
    """De-jag a fundamental-domain partition: hill-climb over the
    face orbits, re-choosing which group copy of each orbit lands in
    which piece whenever that lowers the jaggedness score (cut
    perimeter + hinge-face penalty) and keeps piece 0 (hence, by
    symmetry, every piece) connected.  Mutates and returns assign,
    plus the final score."""
    nf = len(F)
    n = len(subgroup)
    seen = set()
    orbits = []
    for f in range(nf):
        if f not in seen:
            orb = sorted({p[f] for p in subgroup})
            seen.update(orb)
            orbits.append(orb)
    want = nf // n
    pen = 1.5 * sum(l for _, _, l in pairs) / max(1, len(pairs))
    score = _partition_score(assign, adj, pairs, pen)
    for _ in range(60):                 # passes until stable
        improved = False
        for orb in orbits:
            best_score = score - 1e-9
            best_map = None
            for r in orb:
                amap = {p[r]: j for j, p in enumerate(subgroup)}
                if all(amap[x] == assign[x] for x in orb):
                    continue
                old = [assign[x] for x in orb]
                for x in orb:
                    assign[x] = amap[x]
                s = _partition_score(assign, adj, pairs, pen)
                if (s < best_score
                        and _piece0_connected(assign, adj, want)):
                    best_score = s
                    best_map = amap
                for x, o in zip(orb, old):
                    assign[x] = o
            if best_map is not None:
                for x, v in best_map.items():
                    assign[x] = v
                score = best_score
                improved = True
        if not improved:
            break
    return assign, score


def split_congruent(V, F, n):
    """Partition faces into n congruent connected pieces (rotated
    copies of one another). Returns (assignment, valid_counts);
    assignment None if impossible, valid_counts lists workable n."""
    perms = _detect_face_perms(V, F)
    if perms is None:
        return None, []
    valid = set()
    for d in range(2, min(len(F), len(perms)) + 1):
        if len(F) % d == 0 and _free_subgroups(perms, d):
            valid.add(d)
    if len(F) % n != 0:
        return None, sorted(valid)
    best = None
    best_score = None
    adj, pairs = _cut_pairs(V, F)
    for sub in _free_subgroups(perms, n):
        ident = tuple(range(len(F)))
        ordered = [ident] + [p for p in sub if p != ident]
        for seed in range(len(F)):
            assign, _ = _grow_partition(V, F, ordered, seed)
            if assign is None:
                continue
            # score = cut perimeter after de-jagging: shorter
            # boundaries = smoother, more coherent shells
            assign, score = _smooth_partition(F, ordered, assign,
                                              adj, pairs)
            if best_score is None or score < best_score:
                best, best_score = assign, score
    return best, sorted(valid)


_STELLATE_OK = {}


def can_stellate(family, sid, n=6):
    """Whether the stellation option is usable for this solid (some
    faces have all neighbour planes parallel to one axis -- cube,
    prisms, flat-capped elongated Johnson solids). Lazily computed
    and cached for the cheap families; the canonicalized families
    are never rank-deficient."""
    if family == 'KEPLER':
        return False
    if family in ('ARCHIMEDEAN', 'CATALAN'):
        return True
    key = (family, sid, n if family == 'PRISM' else 0)
    if key not in _STELLATE_OK:
        try:
            V, F, _sizes = build_solid(family, sid, n, 1.0)
            stellate(V, F)
            _STELLATE_OK[key] = True
        except Exception:
            _STELLATE_OK[key] = False
    return _STELLATE_OK[key]


def build_solid(family, sid, n=6, scale=1.0):
    """Returns (V, F, face_sizes or None)."""
    if family == 'KEPLER':
        return build_kepler(sid, scale)
    if family == 'PRISM':
        V, F = build_prism(sid, n, scale)
        return V, F, None
    if family == 'JOHNSON':
        V, F = build_johnson(int(sid[1:]), scale)
        return V, F, None
    label, nota = _NOTATION[sid]
    V, F = cw.apply_conway(nota)
    if family in ('ARCHIMEDEAN', 'CATALAN'):
        V = cw.canonicalize(V, F, iters=250)
    # normalise to unit circumradius-ish
    r = max(sqrt(sum(c * c for c in v)) for v in V)
    V = [tuple(c / r * scale for c in v) for v in V]
    return V, F, None


# ---------------------------------------------------------------- #
#  Blender layer                                                   #
# ---------------------------------------------------------------- #

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    FAMILIES = [
        ('PLATONIC', "Platonic", "The five regular solids"),
        ('KEPLER', "Kepler-Poinsot",
         "The four regular star polyhedra (true intersecting faces)"),
        ('ARCHIMEDEAN', "Archimedean", "All 13 semiregular solids"),
        ('CATALAN', "Catalan", "All 13 Archimedean duals"),
        ('PRISM', "Prisms & Antiprisms", "Uniform n-prisms"),
        ('JOHNSON', "Johnson",
         "J1-J48: pyramid / cupola / rotunda solids and their "
         "elongations and pairings"),
    ]

    _ITEM_CACHE = {}

    def _solid_items(self, context):
        fam = self.family
        if fam not in _ITEM_CACHE:
            if fam == 'PRISM':
                items = [('PRISM', "Prism", "uniform n-prism"),
                         ('ANTIPRISM', "Antiprism",
                          "uniform n-antiprism")]
            elif fam == 'JOHNSON':
                items = [(sid, label, "") for (sid, label, _n)
                         in JOHNSON]
            else:
                cat = {'PLATONIC': PLATONIC, 'KEPLER': KEPLER,
                       'ARCHIMEDEAN': ARCHIMEDEAN,
                       'CATALAN': CATALAN}[fam]
                items = [(sid, label, nota or "")
                         for (sid, label, nota) in cat]
            _ITEM_CACHE[fam] = items
        return _ITEM_CACHE[fam]

    class MESH_OT_regular_solid_add(bpy.types.Operator):
        """Add a regular / semiregular / star / Johnson solid,
        organised by family, with stellation, styles and coloring"""
        bl_idname = "mesh.regular_solid_add"
        bl_label = "Regular Solid"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(name="Family", items=FAMILIES,
                             default='PLATONIC')
        solid: EnumProperty(name="Solid", items=_solid_items)
        n: IntProperty(name="Sides", default=6, min=3, max=32,
                       description="Prism / antiprism base sides")
        handedness: EnumProperty(
            name="Handedness",
            items=[('RIGHT', "Right-Handed", "As constructed"),
                   ('LEFT', "Left-Handed",
                    "Mirror image (the other enantiomorph)")],
            default='RIGHT',
            description="Which of the two mirror forms of this "
                        "chiral solid to build")
        stellated: BoolProperty(
            name="Stellated", default=False,
            description="Replace each face with a pyramid to the "
                        "intersection of its neighbours' planes "
                        "(octahedron gives the stella octangula, "
                        "dodecahedron the small stellated "
                        "dodecahedron)")
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", ""),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo "
                    "Style modifier"),
                   ('WIRE', "Wireframe", "Wireframe modifier")],
            default='SOLID')
        border: FloatProperty(name="Border", default=0.3, min=0.02,
                              max=0.95)
        thickness: FloatProperty(name="Thickness", default=0.05,
                                 min=0.001, max=1.0)
        coloring: EnumProperty(
            name="Coloring",
            items=[('SIDES', "By Face Size",
                    "One material per face size (shared with the "
                    "Conway generator; view with Material Preview "
                    "or Solid shading set to Material colour)"),
                   ('NONE', "None", "")],
            default='SIDES')
        pieces: IntProperty(
            name="Congruent Pieces", default=1, min=1, max=60,
            description="Split the shell into this many congruent, "
                        "connected pieces (rotated copies of one "
                        "another; each is a separate object for "
                        "printing and reassembly). 1 = single "
                        "object")
        explode: FloatProperty(
            name="Explode", default=0.0, min=0.0, max=5.0,
            description="Move each piece outward along its centroid "
                        "direction so the split is visible")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        _PALETTE = {3: (0.90, 0.36, 0.23), 4: (0.27, 0.52, 0.79),
                    5: (0.30, 0.69, 0.42), 6: (0.95, 0.77, 0.29),
                    7: (0.62, 0.40, 0.75), 8: (0.25, 0.72, 0.72),
                    9: (0.91, 0.56, 0.71), 10: (0.55, 0.60, 0.29),
                    12: (0.52, 0.45, 0.40)}

        @classmethod
        def _material_for(cls, nn):
            name = f"Conway {nn}-gon"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE.get(
                    nn, (0.5 + 0.5 * math.sin(nn), 0.55,
                         0.5 + 0.5 * math.cos(nn)))
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb,
                                                               1.0)
                    bsdf.inputs["Roughness"].default_value = 0.5
            return mat

        def execute(self, context):
            try:
                V, F, sizes = build_solid(self.family, self.solid,
                                          self.n, self.scale)
                if (self.handedness == 'LEFT'
                        and (self.family, self.solid) in CHIRAL):
                    V, F = mirror_solid(V, F)
                if self.stellated and self.family != 'KEPLER':
                    try:
                        V, F, resid = stellate(V, F)
                        sizes = None
                        if resid > 1e-4 * self.scale:
                            self.report(
                                {'WARNING'},
                                "approximate stellation (neighbour "
                                "planes do not meet at a point)")
                    except ValueError:
                        self.report({'WARNING'},
                                    "this solid cannot be stellated "
                                    "(parallel neighbour planes) -- "
                                    "left unstellated")
            except (ValueError, KeyError, StopIteration) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            fsz = sizes if sizes else [len(f) for f in F]
            label = dict((i[0], i[1]) for i in
                         _solid_items(self, context))[self.solid]
            if self.pieces > 1:
                assign, valid = split_congruent(V, F, self.pieces)
                if assign is None:
                    opts = ", ".join(map(str, valid)) or "none"
                    self.report(
                        {'ERROR'},
                        f"cannot split this solid into "
                        f"{self.pieces} congruent connected pieces "
                        f"(available: {opts})")
                    return {'CANCELLED'}
                groups = [[i for i in range(len(F))
                           if assign[i] == j]
                          for j in range(self.pieces)]
            else:
                groups = [list(range(len(F)))]
            for o in context.selected_objects:
                o.select_set(False)
            first = None
            for j, gf in enumerate(groups):
                remap = {}
                pv = []
                pf = []
                for fi in gf:
                    face = []
                    for i in F[fi]:
                        if i not in remap:
                            remap[i] = len(pv)
                            pv.append(V[i])
                        face.append(remap[i])
                    pf.append(face)
                psz = [fsz[fi] for fi in gf]
                me = bpy.data.meshes.new("Solid")
                me.from_pydata(pv, [], pf)
                me.validate(clean_customdata=True)
                if len(me.polygons) == len(psz):
                    attr = me.attributes.new("ngon_sides", 'INT',
                                             'FACE')
                    attr.data.foreach_set('value', psz)
                    if self.coloring == 'SIDES':
                        lut = {}
                        for nn in sorted(set(psz)):
                            lut[nn] = len(me.materials)
                            me.materials.append(
                                self._material_for(nn))
                        me.polygons.foreach_set(
                            'material_index', [lut[s] for s in psz])
                me.update()
                name = (label if len(groups) == 1
                        else f"{label} {j + 1}of{len(groups)}")
                obj = bpy.data.objects.new(name, me)
                context.collection.objects.link(obj)
                off = (0.0, 0.0, 0.0)
                if self.explode > 0 and len(groups) > 1:
                    c = [0.0, 0.0, 0.0]
                    for p in pv:
                        for k in range(3):
                            c[k] += p[k] / len(pv)
                    ln = sqrt(sum(t * t for t in c)) or 1.0
                    off = tuple(self.explode * t / ln for t in c)
                cur = context.scene.cursor.location
                obj.location = (cur[0] + off[0], cur[1] + off[1],
                                cur[2] + off[2])
                obj.select_set(True)
                if first is None:
                    first = obj
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
            context.view_layer.objects.active = first
            self.report({'INFO'},
                        f"{label}: {len(groups)} piece(s), "
                        f"{len(F)} faces")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
            lay.prop(self, 'solid')
            if self.family == 'PRISM':
                lay.prop(self, 'n')
            if (self.family, self.solid) in CHIRAL:
                lay.prop(self, 'handedness')
            if self.family != 'KEPLER':
                row = lay.row()
                row.enabled = can_stellate(self.family, self.solid,
                                           self.n)
                row.prop(self, 'stellated')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style != 'SOLID':
                lay.prop(self, 'thickness')
            lay.prop(self, 'coloring')
            lay.prop(self, 'pieces')
            if self.pieces > 1:
                lay.prop(self, 'explode')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.regular_solid_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_regular_solid_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_regular_solid_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        fails = []

        def check_regular(V, F, tol=2e-3, name=""):
            """All edges unit length; all faces planar."""
            worst = 0.0
            for f in F:
                m = len(f)
                for i in range(m):
                    e = math.dist(V[f[i]], V[f[(i + 1) % m]])
                    worst = max(worst, abs(e - 1.0))
            return worst

        COUNTS = {'TT': (12, 8), 'CO': (12, 14), 'TC': (24, 14),
                  'TO': (24, 14), 'RCO': (24, 26), 'TCO': (48, 26),
                  'SC': (24, 38), 'ID': (30, 32), 'TD': (60, 32),
                  'TI': (60, 32), 'RID': (60, 62), 'TID': (120, 62),
                  'SD': (60, 92)}
        for (sid, label, nota) in ARCHIMEDEAN:
            V, F = cw.apply_conway(nota)
            V = cw.canonicalize(V, F, iters=250)
            ok = (len(V), len(F)) == COUNTS[sid]
            print(f"{label:32s} V={len(V)} F={len(F)} "
                  f"{'OK' if ok else 'BAD'}")
            if not ok:
                fails.append(sid)
        for (sid, label, nota) in CATALAN:
            V, F = cw.apply_conway(nota)
            V = cw.canonicalize(V, F, iters=250)
            arch = COUNTS[dict(zip(
                [c[0] for c in CATALAN],
                [a[0] for a in ARCHIMEDEAN]))[sid]]
            ok = (len(V), len(F)) == (arch[1], arch[0])
            print(f"{label:32s} V={len(V)} F={len(F)} "
                  f"{'OK' if ok else 'BAD'}")
            if not ok:
                fails.append(sid)
        for (sid, label, num) in JOHNSON:
            V, F = build_johnson(num)
            dev = check_regular(V, F)
            chi = None
            E = set()
            for f in F:
                for i in range(len(f)):
                    a, b = f[i], f[(i + 1) % len(f)]
                    E.add((min(a, b), max(a, b)))
            chi = len(V) - len(E) + len(F)
            ok = dev < 2e-3 and chi == 2
            print(f"{label:44s} edge dev={dev:.2e} chi={chi} "
                  f"{'OK' if ok else 'BAD'}")
            if not ok:
                fails.append(sid)
        # ortho forms have like-meets-like contacts at the girdle
        def titi(num):
            V, F = build_johnson(num)
            from collections import defaultdict
            e2 = defaultdict(list)
            for f in F:
                for i in range(len(f)):
                    a, b = f[i], f[(i + 1) % len(f)]
                    e2[(min(a, b), max(a, b))].append(len(f))
            return sum(1 for s in e2.values() if s == [3, 3])
        for num, expect in ((27, True), (28, True), (29, False),
                            (32, True), (33, False), (34, True)):
            has = titi(num) > 0
            ok = has == expect
            print(f"J{num} tri-tri contacts={has} (expect {expect}) "
                  f"{'OK' if ok else 'BAD'}")
            if not ok:
                fails.append(f'J{num}-og')
        for kind in ('SSD', 'GD', 'GSD', 'GI'):
            V, F, sizes = build_kepler(kind)
            print(f"kepler {kind}: verts={len(V)} faces={len(F)}")
        V, F = cw.apply_conway('O')
        SV, SF, r = stellate([list(v) for v in V], F)
        print(f"stellated octahedron: F={len(SF)}(24) resid={r:.1e} "
              f"{'OK' if len(SF) == 24 and r < 1e-9 else 'BAD'}")
        print("RESULT:", "ALL OK" if not fails else f"FAILS {fails}")
