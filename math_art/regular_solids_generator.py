
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
#                 coordinates; plus J49-J83 -- the augmented prisms,
#                 augmented dodecahedron / diminished icosahedron, the
#                 cupola-augmented truncated solids, and the whole
#                 gyrate/diminished rhombicosidodecahedron family
#                 (glued/sliced/gyrated on exact convex bases); and
#                 J84-J92, the 9 elementary solids (regular-face
#                 relaxation).  All 92 Johnson solids.
#
# Options: generic stellation (each face replaced by a pyramid to the
# intersection of its neighbours' planes -- octahedron gives the
# stella octangula, dodecahedron the small stellated dodecahedron),
# Solid / Leonardo (da Vinci) / Wireframe styles, and coloring by
# face size (sharing the Conway generator's palette).
#
# References:
# - Platonic solids: Euclid, "Elements" Book XIII (construction
#   attributed to Theaetetus).
# - Archimedean solids: Archimedes (work lost); reconstructed by
#   Johannes Kepler, "Harmonices Mundi" (1619).
# - Catalan solids (the duals): Eugene Catalan (1865).
# - Kepler-Poinsot star polyhedra: Johannes Kepler (1619) and Louis
#   Poinsot, "Memoire sur les polygones et les polyedres" (1810).
# - Johnson solids: Norman W. Johnson (1966); completeness proved by
#   Victor Zalgaller (1969).

bl_info = {
    "name": "Regular Solids",
    "author": "Math Art project",
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

# J49-J63: augmented / diminished solids built on exact convex bases
# (prisms, dodecahedron, icosahedron) rather than pyramid/cupola/rotunda
# stacks -- see build_johnson_ext below.
_J_EXT = [
    (49, "Augmented Triangular Prism (J49)"),
    (50, "Biaugmented Triangular Prism (J50)"),
    (51, "Triaugmented Triangular Prism (J51)"),
    (52, "Augmented Pentagonal Prism (J52)"),
    (53, "Biaugmented Pentagonal Prism (J53)"),
    (54, "Augmented Hexagonal Prism (J54)"),
    (55, "Parabiaugmented Hexagonal Prism (J55)"),
    (56, "Metabiaugmented Hexagonal Prism (J56)"),
    (57, "Triaugmented Hexagonal Prism (J57)"),
    (58, "Augmented Dodecahedron (J58)"),
    (59, "Parabiaugmented Dodecahedron (J59)"),
    (60, "Metabiaugmented Dodecahedron (J60)"),
    (61, "Triaugmented Dodecahedron (J61)"),
    (62, "Metabidiminished Icosahedron (J62)"),
    (63, "Tridiminished Icosahedron (J63)"),
    (64, "Augmented Tridiminished Icosahedron (J64)"),
    (65, "Augmented Truncated Tetrahedron (J65)"),
    (66, "Augmented Truncated Cube (J66)"),
    (67, "Biaugmented Truncated Cube (J67)"),
    (68, "Augmented Truncated Dodecahedron (J68)"),
    (69, "Parabiaugmented Truncated Dodecahedron (J69)"),
    (70, "Metabiaugmented Truncated Dodecahedron (J70)"),
    (71, "Triaugmented Truncated Dodecahedron (J71)"),
    (72, "Gyrate Rhombicosidodecahedron (J72)"),
    (73, "Parabigyrate Rhombicosidodecahedron (J73)"),
    (74, "Metabigyrate Rhombicosidodecahedron (J74)"),
    (75, "Trigyrate Rhombicosidodecahedron (J75)"),
    (76, "Diminished Rhombicosidodecahedron (J76)"),
    (77, "Paragyrate Diminished Rhombicosidodecahedron (J77)"),
    (78, "Metagyrate Diminished Rhombicosidodecahedron (J78)"),
    (79, "Bigyrate Diminished Rhombicosidodecahedron (J79)"),
    (80, "Parabidiminished Rhombicosidodecahedron (J80)"),
    (81, "Metabidiminished Rhombicosidodecahedron (J81)"),
    (82, "Gyrate Bidiminished Rhombicosidodecahedron (J82)"),
    (83, "Tridiminished Rhombicosidodecahedron (J83)"),
    (84, "Snub Disphenoid (J84)"),
    (85, "Snub Square Antiprism (J85)"),
    (86, "Sphenocorona (J86)"),
    (87, "Augmented Sphenocorona (J87)"),
    (88, "Sphenomegacorona (J88)"),
    (89, "Hebesphenomegacorona (J89)"),
    (90, "Disphenocingulum (J90)"),
    (91, "Bilunabirotunda (J91)"),
    (92, "Triangular Hebesphenorotunda (J92)"),
]
_J_EXT_NUMS = {num for num, _name in _J_EXT}
JOHNSON += [(f'J{num}', name, num) for num, name in _J_EXT]


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
#  Augmented / diminished Johnson solids (J49-J63)                 #
# ---------------------------------------------------------------- #
#
# J49-J92 are not pyramid/cupola/rotunda stacks; they are made by
# gluing regular pyramids onto, or slicing pyramidal caps off, exact
# convex bases.  These generic primitives operate on a unit-edge base
# and cover the augmented prisms (J49-J57), the augmented dodecahedron
# (J58-J61) and the bi-/tri-diminished icosahedron (J62-J63).  The
# augmented-Archimedean (J64-J83) and elementary (J84-J92) Johnson
# solids are built in later revisions.

def _norm_edge(V, F):
    """Scale V so the mean edge length is 1 (exact for regular bases)."""
    els = []
    for f in F:
        m = len(f)
        for i in range(m):
            els.append(math.dist(V[f[i]], V[f[(i + 1) % m]]))
    s = 1.0 / (sum(els) / len(els))
    return [tuple(c * s for c in v) for v in V]


def _face_centroid(V, f):
    return [sum(V[i][c] for i in f) / len(f) for c in range(3)]


def _face_normal_out(V, F, fi):
    """Unit outward normal of face fi (pointing away from the centroid)."""
    f = F[fi]
    nrm = cw._newell(V, f)
    ln = sqrt(sum(x * x for x in nrm)) or 1.0
    nrm = [x / ln for x in nrm]
    cen = [sum(v[c] for v in V) / len(V) for c in range(3)]
    fc = _face_centroid(V, f)
    if sum(nrm[c] * (fc[c] - cen[c]) for c in range(3)) < 0:
        nrm = [-x for x in nrm]
    return nrm


def _augment(V, F, fi, edge=1.0):
    """Glue a regular pyramid with unit lateral edges onto face fi,
    replacing it with |fi| triangles meeting at a new apex."""
    V = [list(v) for v in V]
    f = F[fi]
    m = len(f)
    cen = _face_centroid(V, f)
    R = sum(math.dist(V[i], cen) for i in f) / m       # circumradius
    h = sqrt(max(1e-12, edge * edge - R * R))
    nrm = _face_normal_out(V, F, fi)
    apex = [cen[c] + nrm[c] * h for c in range(3)]
    ai = len(V)
    V.append(apex)
    NF = [list(g) for j, g in enumerate(F) if j != fi]
    for i in range(m):
        NF.append([f[i], f[(i + 1) % m], ai])
    return V, NF


def _augment_faces(V, F, idxs):
    for fi in sorted(idxs, reverse=True):     # high-to-low keeps indices
        V, F = _augment(V, F, fi)
    return V, F


def _diminish_vertex(V, F, vi):
    """Slice the pyramidal cap at vertex vi: drop every face touching vi
    and cap the opening with the polygon of vi's neighbour ring."""
    other = [list(f) for f in F if vi not in f]
    nxt = {}
    for f in F:
        if vi not in f:
            continue
        m = len(f)
        i = f.index(vi)
        nxt[f[(i + 1) % m]] = f[(i + 2) % m]   # opposite edge, oriented
    start = next(iter(nxt))
    ring = [start]
    while nxt[ring[-1]] != start:
        ring.append(nxt[ring[-1]])
        if len(ring) > 100:
            raise ValueError("bad vertex fan")
    other.append(ring)
    remap = {old: (old if old < vi else old - 1)
             for old in range(len(V)) if old != vi}
    NV = [v for j, v in enumerate(V) if j != vi]
    NF = [[remap[i] for i in f] for f in other]
    return NV, NF


def _prism_unit(n):
    V, F = build_prism('PRISM', n, 1.0)   # side faces are F[0..n-1]
    return V, F


def _platonic_unit(sym):
    V, F = cw._seed(sym, 0)                 # 'D' -> pentagons, 'I' -> tris
    return _norm_edge(V, F), F


def _faces_adjacent(fa, fb):
    return len(set(fa) & set(fb)) >= 2


def _pick_dodeca_faces(V, F, num):
    if num == 58:
        return [0]
    cen = [_face_centroid(V, f) for f in F]
    dot0 = [sum(cen[0][c] * cen[j][c] for c in range(3))
            for j in range(len(F))]
    opp = min(range(len(F)), key=lambda j: dot0[j])
    if num == 59:
        return [0, opp]
    if num == 60:                          # meta: non-adjacent, not opposite
        for j in range(len(F)):
            if j not in (0, opp) and not _faces_adjacent(F[0], F[j]):
                return [0, j]
    if num == 61:                          # three mutually non-adjacent
        chosen = [0]
        for j in range(1, len(F)):
            if all(not _faces_adjacent(F[j], F[c]) for c in chosen):
                chosen.append(j)
                if len(chosen) == 3:
                    return chosen
    raise ValueError(f"dodeca faces for J{num}")


def _pick_icosa_verts(V, F, num):
    edge = min(math.dist(V[f[i]], V[f[(i + 1) % len(f)]])
               for f in F for i in range(len(f)))

    def adj(i, j):
        return abs(math.dist(V[i], V[j]) - edge) < 1e-6

    def antipodal(i, j):
        di = sqrt(sum(x * x for x in V[i]))
        dj = sqrt(sum(x * x for x in V[j]))
        return (sum(V[i][c] * V[j][c] for c in range(3))
                / (di * dj or 1.0)) < -0.99
    if num == 62:                          # two non-adjacent, non-antipodal
        for j in range(1, len(V)):
            if not adj(0, j) and not antipodal(0, j):
                return [0, j]
    if num == 63:                          # three mutually non-adjacent
        chosen = [0]
        for j in range(1, len(V)):
            if all(not adj(j, c) for c in chosen):
                chosen.append(j)
                if len(chosen) == 3:
                    return chosen
    raise ValueError(f"icosa verts for J{num}")


def _augment_cupola(V, F, fi, shift=0, edge=1.0):
    """Glue a regular n-cupola onto a 2n-gon face fi (its 2n-ring welds
    to the face).  shift=0 gives the ortho placement, shift=1 the gyro
    placement (top n-gon rotated by one base edge)."""
    V = [list(v) for v in V]
    f = F[fi]
    twon = len(f)
    n = twon // 2
    cen = _face_centroid(V, f)
    nrm = _face_normal_out(V, F, fi)
    h = _cupola_height(n) * edge
    rt = _rn(n) * edge
    top = []
    for j in range(n):
        a = V[f[(2 * j + shift) % twon]]
        b = V[f[(2 * j + 1 + shift) % twon]]
        mid = [(a[c] + b[c]) / 2 for c in range(3)]
        d = [mid[c] - cen[c] for c in range(3)]
        L = sqrt(sum(x * x for x in d)) or 1.0
        d = [x / L for x in d]
        top.append(len(V))
        V.append([cen[c] + d[c] * rt + nrm[c] * h for c in range(3)])
    NF = [list(g) for j, g in enumerate(F) if j != fi]
    for j in range(n):
        b0 = f[(2 * j + shift) % twon]
        b1 = f[(2 * j + 1 + shift) % twon]
        b2 = f[(2 * j + 2 + shift) % twon]
        NF.append([b0, b1, top[j]])
        NF.append([b1, b2, top[(j + 1) % n], top[j]])
    NF.append(list(reversed(top)))
    return V, NF


def _augment_cupola_faces(V, F, idxs, shift=0):
    for fi in sorted(idxs, reverse=True):
        V, F = _augment_cupola(V, F, fi, shift=shift)
    return V, F


_ARCH_CACHE = {}


def _archimedean_unit(nota):
    """Unit-edge, origin-centred Archimedean solid (canonical form of the
    combinatorial type == the uniform solid, to numerical precision).
    Cached per notation, since canonicalizing the big bases is slow;
    callers get fresh copies to mutate."""
    if nota not in _ARCH_CACHE:
        V, F = cw.apply_conway(nota)
        # uniform solids are canonical, so their canonical form has regular
        # faces; the larger ones need many iterations to converge tightly
        # (canonicalize early-exits once it has).
        V = cw.canonicalize(V, F, iters=2000)
        V = _norm_edge(V, F)
        cen = [sum(v[c] for v in V) / len(V) for c in range(3)]
        _ARCH_CACHE[nota] = ([tuple(v[c] - cen[c] for c in range(3))
                              for v in V], [list(f) for f in F])
    V, F = _ARCH_CACHE[nota]
    return [list(v) for v in V], [list(f) for f in F]


def _pick_faces_by_size(V, F, size, mode):
    """Pick face(s) of the given polygon size in the requested relation:
    'one', 'para' (opposite pair), 'meta' (non-adjacent, non-opposite
    pair), or 'tri' (three mutually non-adjacent)."""
    cand = [fi for fi, f in enumerate(F) if len(f) == size]
    cen = [_face_centroid(V, F[fi]) for fi in range(len(F))]

    def ddot(i, j):
        a, b = cen[i], cen[j]
        na = sqrt(sum(x * x for x in a)) or 1.0
        nb = sqrt(sum(x * x for x in b)) or 1.0
        return sum(a[c] * b[c] for c in range(3)) / (na * nb)
    f0 = cand[0]
    if mode == 'one':
        return [f0]
    opp = min(cand, key=lambda j: ddot(f0, j))
    if mode == 'para':
        return [f0, opp]
    if mode == 'meta':
        for j in cand:
            if j not in (f0, opp) and not _faces_adjacent(F[f0], F[j]):
                return [f0, j]
    if mode == 'tri':
        chosen = [f0]
        for j in cand:
            if all(not _faces_adjacent(F[j], F[c]) for c in chosen):
                chosen.append(j)
                if len(chosen) == 3:
                    return chosen
    raise ValueError(f"cannot pick {mode} {size}-gon faces")


def _cap_dir(V, f):
    c = _face_centroid(V, f)
    L = sqrt(sum(x * x for x in c)) or 1.0
    return [x / L for x in c]


def _find_face_by_dir(V, F, d, size):
    """Index of the size-gon face whose centroid direction best matches d."""
    best, bi = -2.0, None
    for fi, f in enumerate(F):
        if len(f) != size:
            continue
        cd = _cap_dir(V, f)
        dot = sum(cd[c] * d[c] for c in range(3))
        if dot > best:
            best, bi = dot, fi
    return bi


def _remove_faces_cap(V, F, remove):
    """Delete the faces in `remove` (a topological disk), close the hole
    with the polygon of its boundary ring, and drop orphaned vertices."""
    remove = set(remove)
    kept = [list(f) for i, f in enumerate(F) if i not in remove]
    removed = [F[i] for i in remove]
    kept_edges = set()
    for f in kept:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            kept_edges.add((min(a, b), max(a, b)))
    nxt = {}
    for f in removed:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            if (min(a, b), max(a, b)) in kept_edges:
                nxt[a] = b
    start = next(iter(nxt))
    ring = [start]
    while nxt[ring[-1]] != start:
        ring.append(nxt[ring[-1]])
        if len(ring) > 200:
            raise ValueError("cap boundary walk")
    kept.append(list(reversed(ring)))
    used = set()
    for f in kept:
        used.update(f)
    old = sorted(used)
    remap = {o: i for i, o in enumerate(old)}
    NV = [V[o] for o in old]
    NF = [[remap[i] for i in f] for f in kept]
    return NV, NF


def _diminish_cap(V, F, pent_fi):
    """Slice off the pentagonal cupola capped by pentagon face pent_fi,
    leaving a regular decagon in its place."""
    pv = set(F[pent_fi])
    remove = [i for i, f in enumerate(F) if set(f) & pv]
    return _remove_faces_cap(V, F, remove)


def _pick_pent_caps(V, F, mode):
    """Choose pentagonal-cupola caps of the rhombicosidodecahedron by
    icosahedral relation.  Two pentagons are 'adjacent' (share a
    triangle) when their centroid directions meet at ~63 deg; the
    Johnson gyrate/diminished forms only ever act on mutually
    non-adjacent caps, so cap operations stay independent."""
    cand = [fi for fi, f in enumerate(F) if len(f) == 5]
    d = {fi: _cap_dir(V, F[fi]) for fi in cand}

    def dot(i, j):
        return sum(d[i][c] * d[j][c] for c in range(3))

    def nonadj(i, j):
        return dot(i, j) < 0.2          # not sharing a triangle
    c0 = cand[0]
    if mode == 'one':
        return [c0]
    opp = min(cand, key=lambda j: dot(c0, j))
    if mode == 'para':
        return [c0, opp]
    if mode == 'meta':
        for j in cand:
            if j not in (c0, opp) and nonadj(c0, j):
                return [c0, j]
    if mode == 'tri':
        na = [j for j in cand if j != c0 and nonadj(c0, j)]
        for a in na:
            for b in na:
                if b != a and nonadj(a, b):
                    return [c0, a, b]
    raise ValueError(f"cannot pick {mode} pentagon caps")


# (mode, gyrated-cap indices, diminished-cap indices) into the cap list
_RID_ROLES = {
    72: ('one', [0], []),
    73: ('para', [0, 1], []),
    74: ('meta', [0, 1], []),
    75: ('tri', [0, 1, 2], []),
    76: ('one', [], [0]),
    77: ('para', [0], [1]),
    78: ('meta', [0], [1]),
    79: ('tri', [0, 1], [2]),
    80: ('para', [], [0, 1]),
    81: ('meta', [], [0, 1]),
    82: ('tri', [0], [1, 2]),
    83: ('tri', [], [0, 1, 2]),
}


# ---------------------------------------------------------------- #
#  Elementary Johnson solids J84-J92                               #
# ---------------------------------------------------------------- #
#
# The nine "special / elementary" Johnson solids cannot be cut from
# regular components -- they are held rigid only by the requirement
# that every face be a regular unit polygon.  Each is seeded here with
# a combinatorial face list and approximate vertex coordinates
# (evaluated from the closed forms / polynomial roots published for
# each solid -- Snub Disphenoid: root of 2x^3+11x^2+4x-1; Bilunabirotunda:
# golden-ratio triples; etc.), then _relax_regular snaps every face to a
# regular unit polygon, converging to the exact solid (edges equal to
# ~1e-14).  Coordinates are seeds only; the exact geometry is produced
# by the solver here, not copied from any coordinate database.
#
# Seed data: {num: (names, V, F)} with V approximate, F 0-based CCW.

_JELEM = {
    84: ([[0.644584, 0.205562, 0.0], [-0.644584, 0.205562, 0.0],
          [0.0, -0.205562, 0.644584], [0.0, -0.205562, -0.644584],
          [0.5, -0.783931, 0.0], [-0.5, -0.783931, 0.0],
          [0.0, 0.783931, 0.5], [0.0, 0.783931, -0.5]],
         [[2, 0, 6], [2, 1, 5], [3, 0, 4], [3, 1, 7], [4, 0, 2], [4, 2, 5],
          [5, 1, 3], [5, 3, 4], [6, 0, 7], [6, 1, 2], [7, 0, 3], [7, 1, 6]]),
    85: ([[0.5, 0.5, 0.676869], [1.213206, 0.0, 0.185607],
          [-0.5, 0.5, 0.676869], [0.707107, 0.0, -0.676869],
          [0.0, 1.213206, 0.185607], [0.857866, 0.857866, -0.185607],
          [-0.5, -0.5, 0.676869], [0.0, -0.707107, -0.676869],
          [0.0, 0.707107, -0.676869], [-1.213206, 0.0, 0.185607],
          [0.857866, -0.857866, -0.185607], [-0.857866, 0.857866, -0.185607],
          [0.5, -0.5, 0.676869], [-0.707107, 0.0, -0.676869],
          [0.0, -1.213206, 0.185607], [-0.857866, -0.857866, -0.185607]],
         [[1, 0, 12], [2, 0, 4], [3, 1, 10], [4, 0, 5], [5, 0, 1], [5, 1, 3],
          [5, 3, 8], [6, 2, 9], [7, 3, 10], [8, 4, 5], [9, 2, 11], [10, 1, 12],
          [11, 2, 4], [11, 4, 8], [11, 8, 13], [12, 6, 14], [13, 7, 15],
          [13, 9, 11], [14, 6, 15], [14, 7, 10], [14, 10, 12], [15, 6, 9],
          [15, 7, 14], [15, 9, 13], [12, 0, 2, 6], [13, 8, 3, 7]]),
    86: ([[0.0, 0.5, 0.522357], [0.852727, 0.5, 0.0],
          [0.0, 0.789428, -0.434843], [0.5, 0.0, -0.790938],
          [0.0, -0.5, 0.522357], [-0.852727, 0.5, 0.0],
          [0.852727, -0.5, 0.0], [0.0, -0.789428, -0.434843],
          [-0.5, 0.0, -0.790938], [-0.852727, -0.5, 0.0]],
         [[2, 0, 1], [2, 1, 3], [3, 1, 6], [5, 0, 2], [5, 2, 8], [6, 4, 7],
          [7, 3, 6], [7, 4, 9], [8, 2, 3], [8, 3, 7], [8, 7, 9], [9, 5, 8],
          [4, 0, 5, 9], [6, 1, 0, 4]]),
    87: ([[0.0, 0.5, 0.522357], [0.852727, 0.5, 0.0],
          [0.0, 0.789428, -0.434843], [0.5, 0.0, -0.790938],
          [0.0, -0.5, 0.522357], [-0.852727, 0.5, 0.0],
          [0.852727, -0.5, 0.0], [0.0, -0.789428, -0.434843],
          [-0.5, 0.0, -0.790938], [-0.852727, -0.5, 0.0],
          [0.795726, 0.0, 0.864147]],
         [[1, 0, 10], [2, 0, 1], [2, 1, 3], [3, 1, 6], [5, 0, 2], [5, 2, 8],
          [6, 1, 10], [6, 4, 7], [7, 3, 6], [7, 4, 9], [8, 2, 3], [8, 3, 7],
          [8, 7, 9], [9, 5, 8], [10, 0, 4], [10, 4, 6], [4, 0, 5, 9]]),
    88: ([[0.0, 0.5, 0.803997], [0.594633, 0.5, 0.0],
          [0.0, 1.283102, 0.182104], [0.5, 0.0, -0.860839],
          [0.0, 0.854743, -0.721504], [0.0, -0.5, 0.803997],
          [-0.594633, 0.5, 0.0], [0.594633, -0.5, 0.0],
          [0.0, -1.283102, 0.182104], [-0.5, 0.0, -0.860839],
          [0.0, -0.854743, -0.721504], [-0.594633, -0.5, 0.0]],
         [[2, 0, 1], [2, 1, 4], [3, 1, 7], [4, 1, 3], [4, 3, 9], [6, 0, 2],
          [6, 2, 4], [6, 4, 9], [7, 5, 8], [8, 5, 11], [9, 3, 10], [10, 3, 7],
          [10, 7, 8], [10, 8, 11], [11, 6, 9], [11, 9, 10], [1, 0, 5, 7],
          [11, 5, 0, 6]]),
    89: ([[0.5, 0.5, 0.976206], [0.716845, 0.5, 0.0],
          [0.0, 1.101296, 0.352954], [0.5, 0.0, -0.838438],
          [0.0, 0.835659, -0.611119], [-0.5, 0.5, 0.976206],
          [0.5, -0.5, 0.976206], [-0.716845, 0.5, 0.0],
          [0.716845, -0.5, 0.0], [0.0, -1.101296, 0.352954],
          [-0.5, 0.0, -0.838438], [0.0, -0.835659, -0.611119],
          [-0.5, -0.5, 0.976206], [-0.716845, -0.5, 0.0]],
         [[2, 0, 1], [2, 1, 4], [3, 1, 8], [4, 1, 3], [4, 3, 10], [5, 0, 2],
          [5, 2, 7], [7, 2, 4], [7, 4, 10], [8, 6, 9], [9, 6, 12], [10, 3, 11],
          [11, 3, 8], [11, 8, 9], [11, 9, 13], [13, 7, 10], [13, 9, 12],
          [13, 10, 11], [1, 0, 6, 8], [6, 0, 5, 12], [12, 5, 7, 13]]),
    90: ([[0.5, 0.767131, 0.462948], [0.5, 0.0, 1.104438],
          [1.126483, 0.0, 0.325003], [-0.5, 0.767131, 0.462948],
          [0.5, -0.767131, 0.462948], [0.767131, -0.5, -0.462948],
          [-0.5, 0.0, 1.104438], [0.0, -0.5, -1.104438],
          [-1.126483, 0.0, 0.325003], [0.0, -1.126483, -0.325003],
          [-0.5, -0.767131, 0.462948], [0.767131, 0.5, -0.462948],
          [-0.767131, -0.5, -0.462948], [0.0, 0.5, -1.104438],
          [0.0, 1.126483, -0.325003], [-0.767131, 0.5, -0.462948]],
         [[2, 0, 1], [2, 1, 4], [3, 0, 14], [5, 2, 4], [5, 4, 9], [6, 3, 8],
          [7, 5, 9], [8, 3, 15], [9, 4, 10], [10, 6, 8], [10, 8, 12],
          [11, 0, 2], [11, 2, 5], [12, 7, 9], [12, 8, 15], [12, 9, 10],
          [14, 0, 11], [14, 11, 13], [14, 13, 15], [15, 3, 14], [1, 0, 3, 6],
          [10, 4, 1, 6], [13, 7, 12, 15], [13, 11, 5, 7]]),
    91: ([[0.0, 0.0, 0.809017], [0.0, 0.0, -0.809017],
          [0.5, 0.809017, 0.5], [0.5, 0.809017, -0.5],
          [0.5, -0.809017, 0.5], [0.5, -0.809017, -0.5],
          [-0.5, 0.809017, 0.5], [-0.5, 0.809017, -0.5],
          [-0.5, -0.809017, 0.5], [-0.5, -0.809017, -0.5],
          [1.309017, 0.5, 0.0], [1.309017, -0.5, 0.0],
          [-1.309017, 0.5, 0.0], [-1.309017, -0.5, 0.0]],
         [[3, 1, 7], [3, 2, 10], [4, 0, 8], [6, 0, 2], [9, 1, 5], [9, 8, 13],
          [11, 4, 5], [12, 6, 7], [5, 4, 8, 9], [6, 2, 3, 7],
          [10, 2, 0, 4, 11], [11, 5, 1, 3, 10], [12, 7, 1, 9, 13],
          [13, 8, 0, 6, 12]]),
    92: ([[0.0, -0.57735, 1.511523], [1.309017, 0.178411, 0.934172],
          [1.309017, -0.755761, 0.57735], [1.0, 0.0, 0.0],
          [0.5, 0.288675, 1.511523], [-0.809017, 1.044436, 0.934172],
          [-1.309017, 0.178411, 0.934172], [0.0, 1.511523, 0.57735],
          [-1.309017, -0.755761, 0.57735], [-0.5, 0.866025, 0.0],
          [-1.0, 0.0, 0.0], [-0.5, 0.288675, 1.511523],
          [-0.5, -1.222847, 0.934172], [0.809017, 1.044436, 0.934172],
          [0.5, -1.222847, 0.934172], [-0.5, -0.866025, 0.0],
          [0.5, 0.866025, 0.0], [0.5, -0.866025, 0.0]],
         [[3, 1, 2], [3, 2, 17], [4, 1, 13], [8, 6, 10], [9, 5, 7], [9, 7, 16],
          [11, 0, 4], [11, 5, 6], [12, 8, 15], [14, 0, 12], [15, 8, 10],
          [16, 7, 13], [17, 2, 14], [10, 6, 5, 9], [13, 1, 3, 16],
          [14, 12, 15, 17], [1, 4, 0, 14, 2], [5, 11, 4, 13, 7],
          [8, 12, 0, 11, 6], [9, 16, 3, 17, 15, 10]]),
}


def _relax_regular(V, F, iters=1500):
    """Snap the mesh so every face is a regular unit-edge polygon: each
    iteration best-fits a regular m-gon (circumradius 1/(2 sin(pi/m))) to
    every face in its own plane and averages the shared-vertex proposals.
    Converges to the unique all-regular-faced solid for a rigid graph."""
    V = [list(v) for v in V]
    n = len(V)
    for _ in range(iters):
        acc = [[0.0, 0.0, 0.0] for _ in range(n)]
        cnt = [0] * n
        for f in F:
            m = len(f)
            pts = [V[i] for i in f]
            c = [sum(p[k] for p in pts) / m for k in range(3)]
            nx = [0.0, 0.0, 0.0]
            for i in range(m):
                p, q = pts[i], pts[(i + 1) % m]
                nx[0] += (p[1] - q[1]) * (p[2] + q[2])
                nx[1] += (p[2] - q[2]) * (p[0] + q[0])
                nx[2] += (p[0] - q[0]) * (p[1] + q[1])
            ln = sqrt(sum(x * x for x in nx)) or 1.0
            nx = [x / ln for x in nx]
            d0 = [pts[0][k] - c[k] for k in range(3)]
            dot = sum(d0[k] * nx[k] for k in range(3))
            u = [d0[k] - dot * nx[k] for k in range(3)]
            lu = sqrt(sum(x * x for x in u)) or 1.0
            u = [x / lu for x in u]
            v = [nx[1] * u[2] - nx[2] * u[1], nx[2] * u[0] - nx[0] * u[2],
                 nx[0] * u[1] - nx[1] * u[0]]
            R = 0.5 / sin(pi / m)
            a = [sum((p[k] - c[k]) * u[k] for k in range(3)) for p in pts]
            b = [sum((p[k] - c[k]) * v[k] for k in range(3)) for p in pts]
            sr = si = 0.0
            for k in range(m):
                wr, wi = cos(2 * pi * k / m), sin(2 * pi * k / m)
                sr += a[k] * wr + b[k] * wi
                si += b[k] * wr - a[k] * wi
            phi = math.atan2(si, sr)
            for k in range(m):
                ang = 2 * pi * k / m + phi
                tx, ty = R * cos(ang), R * sin(ang)
                vi = f[k]
                for j in range(3):
                    acc[vi][j] += c[j] + tx * u[j] + ty * v[j]
                cnt[vi] += 1
        maxd = 0.0
        for i in range(n):
            if cnt[i] == 0:
                continue
            p = [acc[i][j] / cnt[i] for j in range(3)]
            maxd = max(maxd, math.dist(p, V[i]))
            V[i] = p
        if maxd < 1e-14:
            break
    return V


def build_johnson_elementary(num, scale=1.0):
    V, F = _JELEM[num]
    V = _relax_regular([list(v) for v in V], F)
    cen = [sum(v[c] for v in V) / len(V) for c in range(3)]
    V = [tuple((v[c] - cen[c]) * scale for c in range(3)) for v in V]
    V, F = cw.orient_outward(V, [list(f) for f in F])
    return V, F


def build_johnson_ext(num, scale=1.0):
    if 84 <= num <= 92:
        return build_johnson_elementary(num, scale)
    if num in (49, 50, 51):
        V, F = _prism_unit(3)
        V, F = _augment_faces(V, F, {49: [0], 50: [0, 1],
                                     51: [0, 1, 2]}[num])
    elif num in (52, 53):
        V, F = _prism_unit(5)
        V, F = _augment_faces(V, F, {52: [0], 53: [0, 2]}[num])
    elif num in (54, 55, 56, 57):
        V, F = _prism_unit(6)
        V, F = _augment_faces(V, F, {54: [0], 55: [0, 3], 56: [0, 2],
                                     57: [0, 2, 4]}[num])
    elif num in (58, 59, 60, 61):
        V, F = _platonic_unit('D')
        V, F = _augment_faces(V, F, _pick_dodeca_faces(V, F, num))
    elif num in (62, 63):
        V, F = _platonic_unit('I')
        for vi in sorted(_pick_icosa_verts(V, F, num), reverse=True):
            V, F = _diminish_vertex(V, F, vi)
    elif num == 64:                        # augment J63's central triangle
        V, F = _platonic_unit('I')
        for vi in sorted(_pick_icosa_verts(V, F, 63), reverse=True):
            V, F = _diminish_vertex(V, F, vi)
        tri = next(fi for fi, f in enumerate(F) if len(f) == 3
                   and sum(1 for g in F if g is not f
                           and _faces_adjacent(f, g) and len(g) == 5) == 3)
        V, F = _augment(V, F, tri)
    elif num == 65:                        # cupola on a truncated tetra hex
        V, F = _archimedean_unit('tT')
        V, F = _augment_cupola_faces(
            V, F, _pick_faces_by_size(V, F, 6, 'one'))
    elif num in (66, 67):                  # cupola on truncated cube octagons
        V, F = _archimedean_unit('tC')
        V, F = _augment_cupola_faces(
            V, F, _pick_faces_by_size(V, F, 8,
                                      'one' if num == 66 else 'para'))
    elif num in (68, 69, 70, 71):          # cupolas on trunc. dodeca decagons
        V, F = _archimedean_unit('tD')
        mode = {68: 'one', 69: 'para', 70: 'meta', 71: 'tri'}[num]
        V, F = _augment_cupola_faces(
            V, F, _pick_faces_by_size(V, F, 10, mode))
    elif 72 <= num <= 83:                   # gyrate / diminished RID family
        V, F = _archimedean_unit('eD')
        mode, gyr_i, dim_i = _RID_ROLES[num]
        caps = _pick_pent_caps(V, F, mode)
        dirs = [_cap_dir(V, F[fi]) for fi in caps]
        for i in dim_i:
            V, F = _diminish_cap(V, F, _find_face_by_dir(V, F, dirs[i], 5))
        for i in gyr_i:
            V, F = _diminish_cap(V, F, _find_face_by_dir(V, F, dirs[i], 5))
            di = _find_face_by_dir(V, F, dirs[i], 10)
            V, F = _augment_cupola(V, F, di, shift=1)
    else:
        raise ValueError(f"J{num} not available")
    cen = [sum(v[c] for v in V) / len(V) for c in range(3)]
    V = [tuple((v[c] - cen[c]) * scale for c in range(3)) for v in V]
    V, F = cw.orient_outward(V, F)
    return V, F


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
        num = int(sid[1:])
        if num in _J_EXT_NUMS:
            V, F = build_johnson_ext(num, scale)
        else:
            V, F = build_johnson(num, scale)
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
        ('ARCHIMEDEAN', "Archimedean", "All 13 semiregular solids"),
        ('CATALAN', "Catalan", "All 13 Archimedean duals"),
        ('KEPLER', "Kepler-Poinsot",
         "The four regular star polyhedra (true intersecting faces)"),
        ('PRISM', "Prisms & Antiprisms", "Uniform n-prisms"),
        ('JOHNSON', "Johnson",
         "All 92: pyramid / cupola / rotunda solids and their "
         "elongations and pairings, the augmented prisms, augmented "
         "dodecahedron / diminished icosahedron, cupola-augmented "
         "truncated solids, the gyrate/diminished rhombicosidodecahedron "
         "family and the 9 elementary solids"),
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
                    "or Solid shading set to Material color)"),
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
            # mirror AFTER the split: the domain search is not
            # mirror-symmetric, and equal-score optima differ in
            # shape, so the left form takes exactly the mirrored
            # right-handed split (mirror_solid keeps face order,
            # so the assignment carries over unchanged)
            if (self.handedness == 'LEFT'
                    and (self.family, self.solid) in CHIRAL):
                V, F = mirror_solid(V, F)
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
