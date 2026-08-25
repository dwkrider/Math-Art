# Spidronised Space-Filler -- the decatrihedron of the triamond net.
#
# Pearce's first space-filling saddle polyhedron, with its faces filled
# by spidron nests after van Ballegooijen, Gailiunas and Erdely's
# "Spidronised Space-fillers" (Bridges 2009), whose Example 1 this
# module implements end to end: the polyhedron, its two chiral
# spidronised forms, the two-polyhedron repeat unit, and blocks of the
# space-filling.
#
# THE NET.  The triamond -- Laves' graph, the (10,3)-a or srs net, the
# K4 crystal -- is the 3-connected net whose vertices sit on Wyckoff
# site 8a of space group I4132: (1/8,1/8,1/8) and equivalents, each
# bonded to its three nearest neighbours.  The three edges at a vertex
# are coplanar at 120 degrees, and the plane's normal is a [111]
# direction, the site's 3-fold axis.  Every shortest circuit of the net
# is a skew DECAGON, and every angle in every circuit is 120 degrees.
#
# THE CELL.  The interstitial domain of the net is a saddle polyhedron
# bounded by THREE of those decagons -- Pearce's "decatrihedron",
# classification [10, 3].  Its two apexes are net vertices a half body
# diagonal apart on a common 3-fold axis; the three decagons are
# exactly the circuits through both apexes, and they close into a
# surface with F = 3, E = 15, V = 14 (Euler 2) -- two 3-valent apexes
# and twelve 2-valent vertices, something no plane-faced polyhedron can
# have.  The cell has the proper symmetry of a triangular prism (D3:
# one 3-fold axis through the apexes, three 2-fold axes across it) and
# is chiral.  The decagon itself is equilateral and equiangular at 120
# degrees yet NOT regular: its proper symmetries are three 2-fold axes
# only (point group 222), it admits no mirror, and it is enantiomorphic
# -- Pearce's figure 8.24, the Bridges paper's nest n10a.  Cell centres
# fall on the MIRROR-IMAGE srs net (Wyckoff 8a negated), which is
# Pearce's observation that the system is an enantiomorphic self-dual.
#
# THE NESTS AND THE MATCHING RULE.  Each decagon is spidronised by the
# Bridges construction (`spidron_math.spidronise`): scale toward the
# centroid, rotate about the Newell normal, triangulate the annulus,
# repeat.  In a space-filling any two coincident faces must be the SAME
# surface, which viewed from the two sides reads clockwise from one
# polyhedron and counter-clockwise from the other -- the paper's rule
# that CW always meets CCW.  With three faces no single decorated form
# can satisfy it (it would need one and a half clockwise faces), so TWO
# spidronised forms are needed and the basic repeat unit is two
# polyhedra sharing one face, with four external faces.  Here the rule
# is realised by 2-colouring the cells: cell adjacency (the dual srs
# net) is bipartite, one colour class winds every face + about its
# outward normal and the other winds -, and a rotation of +t about the
# outward normal IS a rotation of -t about the inward one, so the two
# cells at a shared face build bit-for-bit the same nest.  Every cell
# of one colour is congruent to every other; the two colours are not
# congruent to each other under any proper isometry; and the whole
# packing, like the bare decagon, exists in two enantiomorphic forms.
#
# THE PARAMETERS.  The paper warns that the faces of this polyhedron
# pass very close to one another, so the nest's scale factor and
# rotation angle need care to avoid self-intersection.  A numerical
# triangle-intersection sweep shows the collisions all come from the
# outermost rings swinging past the boundary; once those clear, deeper
# rings never collide, so the ring count is free (the measured limits
# are identical from 3 rings to 8).
#
# The safe region is NOT symmetric in the sign of the twist for a
# single polyhedron -- it cannot be, because the bare cell is CHIRAL.
# Winding +t and -t about the outward normals of a chiral cell are not
# mirror-equivalent operations, and the measured limits differ widely:
# at ring scale 0.60 a colour-0 cell stays intersection-free from
# -38.9 to +14.5 degrees.  A colour-1 cell is the body-centring
# translate of a colour-0 cell with the twist negated, so its region
# is the same interval negated (-14.5 to +38.9).  The space-filling
# layouts contain both forms plus CROSS-cell face pairs, and one of
# those binds slightly before the intra-cell limit: the measured
# symmetric bound is +-14.18 degrees at scale 0.60 (identical for the
# two-cell unit and the 8-cell block).  An earlier revision of
# this header claimed the region itself was symmetric in twist sign --
# that was an artefact of sweeping only the two-cell repeat unit,
# where the intersection of the two mirror-related per-form intervals
# is symmetric by construction; a single polyhedron (Layout =
# Polyhedron builds a colour-0 cell) shows the asymmetry directly as a
# folded-over spike on one twist sign only.  `TWIST_LIMITS` tabulates
# the bisected per-form limits against ring scale, `twist_limits()`
# interpolates them, and the operator clamps the twist to the safe
# interval of the layout being built unless the user opts out.  The
# defaults -- ring scale 0.60, twist 12 degrees -- sit inside the
# symmetric region with margin, and the self-test asserts both the
# defaults and the tabulated limits against a direct sweep.
#
# References:
# - Walt van Ballegooijen, Paul Gailiunas & Daniel Erdely,
#   "Spidronised Space-fillers", Bridges 2009 Conference Proceedings,
#   pp. 271-278 -- Example 1, the decatrihedron: the nest n10a, the
#   CW-meets-CCW rule, the two chiral forms and the two-polyhedron
#   repeat unit with four external faces.
# - Peter Pearce, "Structure in Nature is a Strategy for Design", The
#   MIT Press, 1978, ch. 8 -- the saddle decatrihedron of the Laves
#   network (figs 8.6-8.8, 8.24): its triangular-prism symmetry, the
#   equilateral equiangular 2-fold decagon, and the space-filling as
#   an enantiomorphic self-dual.
# - Daniel Erdely, "Some Surprising New Properties of the Spidrons",
#   Bridges 2005 Conference Proceedings, pp. 179-186 -- the spidron
#   and the hexagonal nest the face decoration generalises.
# - John H. Conway, Heidi Burgiel & Chaim Goodman-Strauss, "The
#   Symmetries of Things", A K Peters, 2008, pp. 351-352 -- the
#   triamond net, its I4132 symmetry and its chirality.

bl_info = {
    "name": "Spidronised Space-Filler",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Polyhedra",
    "description": "Pearce's decatrihedron with spidron-nest faces: "
                   "one saddle polyhedron, the two-cell repeat unit, "
                   "or a block of the triamond space-filling",
    "category": "Add Mesh",
}

from itertools import combinations, product
from math import pi, radians, sqrt

import numpy as np

try:
    from . import spidron_math as sm
    from .spidron_ball_generator import arm_boundary_edges
    from .polyhedra import fit as _fit
    from .patterns import common as pc
    from . import sharp_creases as _sc
except Exception:                       # legacy single-file / CLI use
    import spidron_math as sm
    from spidron_ball_generator import arm_boundary_edges
    from polyhedra import fit as _fit
    from patterns import common as pc
    import sharp_creases as _sc

# srs net vertices per conventional cubic cell, in eighths of the cell
# edge: Wyckoff 8a of I4132.
_BASE8 = ((1, 1, 1), (3, 7, 5), (7, 5, 3), (5, 3, 7),
          (5, 5, 5), (7, 3, 1), (3, 1, 7), (1, 7, 3))
# neighbour offsets (also in eighths): the twelve (+-2, +-2, 0) vectors;
# each vertex matches exactly three of them.
_NBR = ((2, 2, 0), (2, -2, 0), (-2, 2, 0), (-2, -2, 0),
        (2, 0, 2), (2, 0, -2), (-2, 0, 2), (-2, 0, -2),
        (0, 2, 2), (0, 2, -2), (0, -2, 2), (0, -2, -2))
# cell-centre translation classes (x8, mod 8): the dual (mirror) srs
# positions split by the body-centring translation, which is the
# 2-colouring of the cells.  Verified edge by edge in `_selftest`.
_CLASS0 = {(7, 7, 7), (5, 1, 3), (1, 3, 5), (3, 5, 1)}
_CLASS1 = {(3, 3, 3), (1, 5, 7), (5, 7, 1), (7, 1, 5)}

EDGE = sqrt(2.0) / 4.0                  # net edge length, cell edge 1


# --------------------------------------------------------------------
# 1.  The triamond net and its decatrihedral cells
# --------------------------------------------------------------------

def net_chunk(n):
    """The srs net over an n^3 block of unit cells, in integer
    eighth-coordinates.  Returns (verts, index, adjacency)."""
    Vi = []
    for off in product(range(n), repeat=3):
        for b in _BASE8:
            Vi.append((b[0] + 8 * off[0], b[1] + 8 * off[1],
                       b[2] + 8 * off[2]))
    idx = {p: i for i, p in enumerate(Vi)}
    adj = {}
    for p in Vi:
        adj[p] = [q for q in ((p[0] + d[0], p[1] + d[1], p[2] + d[2])
                              for d in _NBR) if q in idx]
    return Vi, idx, adj


def vertex_axis(p, adj):
    """The 3-fold axis at a full-degree vertex: the unit normal of the
    plane its three (coplanar, 120-degree) edges span."""
    ns = adj[p]
    if len(ns) < 3:
        return None
    e = [np.subtract(q, p).astype(float) for q in ns]
    ax = np.cross(e[1] - e[0], e[2] - e[0])
    return ax / np.linalg.norm(ax)


def _canon_ring(cyc):
    """Canonical form of a cyclic vertex-index sequence."""
    k = cyc.index(min(cyc))
    c1 = tuple(cyc[k:] + cyc[:k])
    c2 = tuple([c1[0]] + list(reversed(c1[1:])))
    return min(c1, c2)


def rings_between(A, B, idx, adj):
    """All decagon circuits of the net through both A and B."""
    rings = set()

    def dfs(path, seen):
        last = path[-1]
        if len(path) == 10:
            if A in adj[last] and B in seen:
                rings.add(_canon_ring([idx[p] for p in path]))
            return
        for q in adj[last]:
            if q not in seen:
                dfs(path + [q], seen | {q})

    dfs([A], {A})
    return rings


def _closed_surface(rings):
    """Do these circuits close up (every edge in exactly two)?"""
    cnt = {}
    for cyc in rings:
        for i in range(len(cyc)):
            e = (cyc[i], cyc[(i + 1) % len(cyc)])
            e = e if e[0] < e[1] else (e[1], e[0])
            cnt[e] = cnt.get(e, 0) + 1
    return all(v == 2 for v in cnt.values()), len(cnt)


def cell_between(A, B, idx, adj):
    """The decatrihedron with apexes A and B: the three decagon
    circuits through both, provided they close into a surface."""
    R = sorted(rings_between(A, B, idx, adj))
    if len(R) != 3:
        return None
    ok, ne = _closed_surface(R)
    if not ok or ne != 15:
        return None
    return tuple(R)


def cell_colour(centre8):
    """Which of the two spidronised forms a cell at this centre takes:
    0 for one translation class of cell centres, 1 for its body-centred
    copy.  Adjacent cells always differ (asserted in the self-test), so
    this is the CW/CCW alternation the space-filling requires."""
    key = tuple(int(round(c)) % 8 for c in centre8)
    if key in _CLASS0:
        return 0
    if key in _CLASS1:
        return 1
    raise ValueError("not a decatrihedron centre: %r" % (centre8,))


_CELLS_CACHE = {}


def enumerate_cells(n):
    """All complete decatrihedra in an n^3 chunk of the net.

    Returns (V, cells) where V is the vertex array (unit-cell
    coordinates) and cells maps an apex-pair key to a dict with the
    cell's rings, centre, and colour (= which spidronised form)."""
    if n in _CELLS_CACHE:
        return _CELLS_CACHE[n]
    Vi, idx, adj = net_chunk(n)
    V = np.asarray(Vi, float) / 8.0
    cells = {}
    for p in Vi:
        ax = vertex_axis(p, adj)
        if ax is None:
            continue
        for s in product((-4, 4), repeat=3):
            if float(np.abs(np.cross(np.asarray(s, float), ax)).max()) > 1e-9:
                continue
            q = (p[0] + s[0], p[1] + s[1], p[2] + s[2])
            if q not in idx:
                continue
            key = tuple(sorted((idx[p], idx[q])))
            if key in cells:
                continue
            rings = cell_between(p, q, idx, adj)
            if rings is None:
                continue
            c8 = tuple((pi_ + qi_) / 2.0 for pi_, qi_ in zip(p, q))
            cells[key] = dict(rings=rings, centre8=c8,
                              centre=np.asarray(c8, float) / 8.0,
                              colour=cell_colour(c8))
    _CELLS_CACHE[n] = (V, cells)
    return V, cells


def face_owners(cells):
    """ring -> [cell keys sharing it]."""
    owners = {}
    for key, c in cells.items():
        for f in c['rings']:
            owners.setdefault(f, []).append(key)
    return owners


# --------------------------------------------------------------------
# 2.  Spidronising a cell
# --------------------------------------------------------------------

def nest_face(P, cell_centre, colour, scale, twist, rings, cap=True):
    """The spidron nest on one decagon face of one cell.

    P is the (10, 3) vertex loop; the nest scales toward the centroid
    and rotates about the face normal ORIENTED OUT OF THE CELL, with
    the rotation sense set by the cell's colour.  Because +t about the
    outward normal equals -t about the inward one, the two cells that
    share a face build the identical surface -- the CW-meets-CCW rule.
    Returns (verts, tris, kinds)."""
    P = np.asarray(P, float)
    C = P.mean(axis=0)
    N = sm._best_fit_normal(P)
    if float(N @ (C - np.asarray(cell_centre, float))) < 0.0:
        N = -N
    ch = 1 if colour == 0 else -1
    return sm.spidronise(P, scale, twist, rings, chirality=ch,
                         centre=C, normal=N, cap=cap)


# --------------------------------------------------------------------
# Safe-twist limits (see THE PARAMETERS in the header)
# --------------------------------------------------------------------

# (ring scale, largest clean negative twist, largest clean positive
# twist) in DEGREES for a single colour-0 cell, bisected to 0.2 deg
# with `form_intersects` (the values are identical from 3 to 8 rings,
# so ring count does not enter).  A colour-1 cell is the translated
# mirror-wound form: its interval is this one negated.  Layouts that
# contain both forms get the symmetric intersection of the two.
TWIST_LIMITS = (
    (0.35, 60.00, 52.73),
    (0.40, 60.00, 36.91),
    (0.45, 60.00, 28.24),
    (0.50, 60.00, 22.27),
    (0.55, 56.60, 17.81),
    (0.60, 38.91, 14.53),
    (0.65, 27.30, 12.07),
    (0.70, 20.27, 9.84),
    (0.75, 15.00, 7.85),
    (0.80, 10.90, 5.98),
    (0.85, 7.50, 4.34),
    (0.90, 4.57, 2.70),
    (0.95, 2.11, 1.29),
    (0.97, 1.17, 0.70),
)


# The same measurement over the two-cell repeat unit and the 8-cell
# block at gap 1: (ring scale, symmetric limit).  Within one cell the
# +14.5-degree intra-cell bound governs, but a CROSS-cell collision
# between neighbouring cells binds slightly earlier on the other sign
# (14.18 degrees at scale 0.60, identical for unit and block); the
# infinite packing at -t is the body-centring translate of the packing
# at +t, so its true safe interval is the symmetric minimum of the
# two, which is this column.
TWIST_LIMITS_PACKED = (
    (0.35, 52.73), (0.40, 36.91), (0.45, 28.24), (0.50, 22.27),
    (0.55, 17.81), (0.60, 14.18), (0.65, 11.37), (0.70, 8.91),
    (0.75, 6.91), (0.80, 5.04), (0.85, 3.52), (0.90, 2.23),
    (0.95, 1.05), (0.97, 0.59),
)


def _interp(table, scale):
    s = min(max(float(scale), table[0][0]), table[-1][0])
    for i in range(len(table) - 1):
        r0, r1 = table[i], table[i + 1]
        if s <= r1[0]:
            f = (s - r0[0]) / (r1[0] - r0[0])
            return tuple(a + f * (b - a)
                         for a, b in zip(r0[1:], r1[1:]))
    return tuple(table[-1][1:])


def twist_limits(scale, colour=0):
    """The intersection-free twist interval of one cell, in radians.

    Returns (neg, pos) magnitudes: a cell of `colour` is free of
    self-intersection for twists in [-neg, +pos].  Linear interpolation
    of TWIST_LIMITS; the measured curves are convex decreasing, so the
    chord sits inside the safe region."""
    neg, pos = _interp(TWIST_LIMITS, scale)
    if colour:
        neg, pos = pos, neg
    return radians(neg), radians(pos)


def twist_limit_packed(scale):
    """The symmetric intersection-free twist bound of the space-filling
    layouts (both forms present, gap 1 worst case), in radians."""
    return radians(_interp(TWIST_LIMITS_PACKED, scale)[0])


def _default_cells(layout):
    """The canonical cell selection for the CELL and UNIT layouts:
    (V, cells, keys) with keys the near-centre colour-0 cell, plus its
    lowest-keyed neighbour for UNIT."""
    n = 3
    V, cells = enumerate_cells(n)
    mid = np.full(3, n / 2.0)
    c0 = min((k for k, c in cells.items() if c['colour'] == 0),
             key=lambda k: (float(np.linalg.norm(cells[k]['centre']
                                                 - mid)),
                            cells[k]['centre8']))
    if layout == 'CELL':
        return V, cells, [c0]
    owners = face_owners(cells)
    nbrs = set()
    for f in cells[c0]['rings']:
        for k in owners.get(f, ()):
            if k != c0:
                nbrs.add(k)
    c1 = min(nbrs, key=lambda k: cells[k]['centre8'])
    return V, cells, [c0, c1]


def nest_triangles(V, cells, keys, scale, twist, rings):
    """Every nest triangle of the listed cells, as (3, 3) arrays."""
    tris = []
    for k in keys:
        c = cells[k]
        for f in c['rings']:
            v, fc, _kd = nest_face(V[list(f)], c['centre'], c['colour'],
                                   scale, twist, rings)
            A = np.asarray(v)
            tris.extend(A[list(t)] for t in fc)
    return tris


def _tri_soup_intersects(tris):
    """Any proper intersection in this triangle set?  Pairs sharing a
    vertex (mesh adjacency, and the exactly coincident shared surfaces
    of neighbouring cells) do not count."""
    T = np.asarray(tris, float)
    n = len(T)
    lo = T.min(axis=1)
    hi = T.max(axis=1)
    ov = np.ones((n, n), bool)
    for a in range(3):
        ov &= lo[:, None, a] <= hi[None, :, a] + 1e-12
        ov &= lo[None, :, a] <= hi[:, None, a] + 1e-12
    ii, jj = np.nonzero(np.triu(ov, 1))
    for i, j in zip(ii, jj):
        t1, t2 = T[i], T[j]
        d = np.linalg.norm(t1[:, None, :] - t2[None, :, :], axis=2)
        if float(d.min()) < 1e-9:
            continue
        if _tri_tri(t1, t2):
            return True
    return False


def form_intersects(scale, twist, rings=4, colour=0):
    """Does a single cell of this colour self-intersect at these nest
    parameters?  (The measurement behind TWIST_LIMITS.)"""
    n = 3
    V, cells = enumerate_cells(n)
    mid = np.full(3, n / 2.0)
    key = min((k for k, c in cells.items() if c['colour'] == colour),
              key=lambda k: (float(np.linalg.norm(cells[k]['centre']
                                                  - mid)),
                             cells[k]['centre8']))
    return _tri_soup_intersects(
        nest_triangles(V, cells, [key], scale, twist, rings))


def unit_intersects(scale, twist, rings=4):
    """Does the two-cell repeat unit (at gap 1, shared surface
    coincident and excluded) self-intersect?"""
    V, cells, keys = _default_cells('UNIT')
    return _tri_soup_intersects(
        nest_triangles(V, cells, keys, scale, twist, rings))


COLOR_ITEMS = [
    ('FORM', "Form",
     "One colour per spidronised form, showing the clockwise and "
     "counter-clockwise polyhedra the space-filling alternates"),
    ('CELL', "Cell", "One colour per polyhedron"),
    ('FACE', "Face", "One colour per decagon face"),
    ('RING', "Ring",
     "One colour per spiral ring, banding each face from rim to "
     "centre"),
    ('UNIFORM', "Uniform", "A single material"),
]

NPAL = 12


def build(layout='UNIT', nx=1, ny=1, nz=1, rings=6, scale=0.60,
          twist=radians(12.0), gap=0.92, mirror=False, color_by='FORM',
          open_center=False, limit_twist=False):
    """Build spidronised decatrihedra.

    layout: 'CELL' one polyhedron, 'UNIT' the two-cell repeat unit,
    'BLOCK' every cell whose centre falls in an nx x ny x nz box of
    unit cells.  Each cell is shrunk about its own centre by `gap`;
    at gap = 1 coincident faces of neighbouring cells match exactly.
    `mirror` builds the enantiomorphic space-filling.  `limit_twist`
    clamps the twist to the intersection-free interval of the layout:
    the (asymmetric) colour-0 interval for CELL, the symmetric
    two-form intersection for UNIT and BLOCK.

    Returns (verts, faces, mats, labels, info): labels carries
    (cell_key, face_ring, arm) per triangle for creasing, info the cell
    count, form split, and the twist actually used ('twist',
    'twist_clamped')."""
    twist = float(twist)
    twist_in = twist
    if limit_twist:
        if layout == 'CELL':
            neg, pos = twist_limits(scale)      # colour-0 interval
        else:                                   # both forms + cross-cell
            neg = pos = twist_limit_packed(scale)
        twist = min(max(twist, -neg), pos)
    if layout == 'BLOCK':
        n = max(nx, ny, nz) + 2
        V, cells = enumerate_cells(n)
        lo = np.array([1.0, 1.0, 1.0])
        hi = lo + np.array([nx, ny, nz], float)
        keep = [k for k, c in cells.items()
                if (c['centre'] >= lo - 1e-9).all()
                and (c['centre'] < hi - 1e-9).all()]
    else:
        V, cells, keep = _default_cells(layout)

    verts, faces, mats, labels = [], [], [], []
    face_index = {}
    forms = [0, 0]
    for ci, key in enumerate(sorted(keep)):
        c = cells[key]
        forms[c['colour']] += 1
        for f in c['rings']:
            if f not in face_index:
                face_index[f] = len(face_index)
            P = V[list(f)]
            v, fc, kd = nest_face(P, c['centre'], c['colour'], scale,
                                  twist, rings, cap=not open_center)
            A = np.asarray(v, float)
            A = c['centre'] + (A - c['centre']) * gap
            o = len(verts)
            verts.extend([tuple(p) for p in A])
            faces.extend([tuple(i + o for i in t) for t in fc])
            n_ann = rings * 20
            for ti, k in enumerate(kd):
                if color_by == 'FORM':
                    m = c['colour']
                elif color_by == 'CELL':
                    m = ci % NPAL
                elif color_by == 'FACE':
                    m = face_index[f] % NPAL
                elif color_by == 'RING':
                    m = (ti // 20 if ti < n_ann else rings) % NPAL
                else:
                    m = 0
                mats.append(m)
                arm = (ti % 20) // 2 if ti < n_ann else ti - n_ann
                labels.append((key, f, arm))

    P = np.asarray(verts, float)
    ctr = 0.5 * (P.max(axis=0) + P.min(axis=0))
    P = P - ctr
    if mirror:
        P[:, 0] = -P[:, 0]
        faces = [tuple(reversed(t)) for t in faces]
    verts = [tuple(p) for p in P]
    info = dict(cells=len(keep), faces=len(face_index), forms=forms,
                twist=twist, twist_clamped=(twist != twist_in))
    return verts, faces, mats, labels, info


# --------------------------------------------------------------------
# 3.  Blender operator
# --------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_spidron_spacefill_add(bpy.types.Operator,
                                        AddObjectHelper):
        """Add spidronised decatrihedra: Pearce's three-faced saddle
        space-filler with spidron-nest faces, as one polyhedron, the
        two-cell repeat unit, or a block of the space-filling"""
        bl_idname = "mesh.spidron_spacefill_add"
        bl_label = "Spidronised Space-Filler"
        bl_options = {'REGISTER', 'UNDO'}

        layout_kind: EnumProperty(
            name="Layout", default='UNIT',
            items=[('CELL', "Polyhedron",
                    "A single spidronised decatrihedron: three skew "
                    "decagon faces filled with spiral nests"),
                   ('UNIT', "Repeat Unit",
                    "The basic repeat unit of the space-filling: two "
                    "polyhedra of opposite winding sharing one face, "
                    "with four external faces"),
                   ('BLOCK', "Block",
                    "Every polyhedron whose centre falls in the chosen "
                    "box of lattice cells")],
            description="How much of the space-filling to build")
        nx: IntProperty(
            name="Cells X", default=1, min=1, max=4,
            description="Lattice cells along X (a cell holds eight "
                        "polyhedra)")
        ny: IntProperty(
            name="Cells Y", default=1, min=1, max=4,
            description="Lattice cells along Y")
        nz: IntProperty(
            name="Cells Z", default=1, min=1, max=4,
            description="Lattice cells along Z")
        rings: IntProperty(
            name="Rings", default=6, min=1, max=12,
            description="How many times the spiral step repeats on "
                        "each face")
        scale_step: FloatProperty(
            name="Ring Scale", default=0.60, min=0.35, max=0.97,
            description="How much each ring shrinks toward the centre "
                        "of its face. Beyond about 0.65 at the default "
                        "twist, neighbouring faces of this polyhedron "
                        "start to cross -- they pass very close")
        twist: FloatProperty(
            name="Twist", default=radians(12.0),
            min=radians(-60.0), max=radians(60.0), subtype='ANGLE',
            description="Rotation of each ring toward the face centre. "
                        "The collision-free range depends on the ring "
                        "scale and, for a single polyhedron, on the "
                        "twist sign -- the cell is chiral, so one "
                        "winding direction has far more room than the "
                        "other (about -39 to +14 degrees at the "
                        "default ring scale)")
        limit_twist: BoolProperty(
            name="Limit Twist", default=True,
            description="Clamp the twist to the measured collision-"
                        "free range for this ring scale and layout. "
                        "The faces of this polyhedron pass very close "
                        "to one another; disable to drive past the "
                        "limit and let the outermost rings collide")
        gap: FloatProperty(
            name="Gap Factor", default=0.92, min=0.05, max=1.0,
            description="Scale of each polyhedron about its own centre "
                        "(1.0 = neighbours share their faces exactly)")
        mirror: BoolProperty(
            name="Mirror Form", default=False,
            description="Build the enantiomorphic space-filling -- the "
                        "decagon is chiral, so the whole packing "
                        "exists in two mirror forms")
        color_by: EnumProperty(
            name="Color", items=COLOR_ITEMS, default='FORM',
            description="How materials are assigned across the "
                        "polyhedra")
        smooth: BoolProperty(
            name="Smooth Shading", default=True,
            description="Shade the spiral surfaces smooth instead of "
                        "faceted")
        sharp_edges: BoolProperty(
            name="Sharp Creases", default=True,
            description="Keep the fold between neighbouring spiral "
                        "arms crisp under smooth shading")
        open_center: BoolProperty(
            name="Open Centres", default=False,
            description="Leave the small hole at the centre of each "
                        "face open instead of closing it")

        def execute(self, context):
            V, F, M, labels, info = build(
                self.layout_kind, int(self.nx), int(self.ny),
                int(self.nz), int(self.rings), float(self.scale_step),
                float(self.twist), float(self.gap), self.mirror,
                self.color_by, self.open_center, self.limit_twist)
            if not F:
                self.report({'ERROR'}, "no geometry generated")
                return {'CANCELLED'}
            obj = pc.build_object(context, "Spidron Spacefill", V, F, M,
                                  span=2.0, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no geometry generated")
                return {'CANCELLED'}
            me = obj.data
            if self.smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
                me.update()
            ncrease = 0
            if self.sharp_edges:
                ncrease = _sc.mark_sharp(me, arm_boundary_edges(F, labels))
                if ncrease == 0:
                    self.report({'WARNING'},
                                "no arm boundaries found to crease")
            note = ""
            if info.get('twist_clamped'):
                from math import degrees as _deg
                note = ("  twist limited to %.1f deg (turn off Limit "
                        "Twist to override)" % _deg(info['twist']))
            self.report({'INFO'},
                        "%d polyhedra (%d CW + %d CCW), %d faces  "
                        "V=%d F=%d  creases=%d%s"
                        % (info['cells'], info['forms'][0],
                           info['forms'][1], info['faces'],
                           len(me.vertices), len(me.polygons), ncrease,
                           note))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'layout_kind')
            if self.layout_kind == 'BLOCK':
                for p in ('nx', 'ny', 'nz'):
                    lay.prop(self, p)
            for p in ('rings', 'scale_step', 'twist', 'limit_twist',
                      'gap', 'mirror', 'color_by', 'open_center',
                      'smooth', 'sharp_edges'):
                lay.prop(self, p)
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.spidron_spacefill_add",
                             icon='SNAP_VOLUME')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_spidron_spacefill_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_spidron_spacefill_add)


# --------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------

def _tri_tri(t1, t2, eps=1e-12):
    """Do two triangles properly intersect?  Coincident triangles and
    contact along shared edges do not count."""
    p1, q1, r1 = t1
    p2, q2, r2 = t2
    n2 = np.cross(q2 - p2, r2 - p2)
    d1 = [float(np.dot(n2, x - p2)) for x in (p1, q1, r1)]
    if all(d > eps for d in d1) or all(d < -eps for d in d1):
        return False
    n1 = np.cross(q1 - p1, r1 - p1)
    d2 = [float(np.dot(n1, x - p1)) for x in (p2, q2, r2)]
    if all(d > eps for d in d2) or all(d < -eps for d in d2):
        return False

    def seg_tri(a, b, p, q, r):
        n = np.cross(q - p, r - p)
        da = float(np.dot(n, a - p))
        db = float(np.dot(n, b - p))
        if da * db > -eps:
            return False
        t = da / (da - db)
        x = a + t * (b - a)
        v0, v1, v2 = q - p, r - p, x - p
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-18:
            return False
        u = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        return u > 1e-9 and w > 1e-9 and u + w < 1.0 - 1e-9

    for a, b in ((p1, q1), (q1, r1), (r1, p1)):
        if seg_tri(a, b, p2, q2, r2):
            return True
    for a, b in ((p2, q2), (q2, r2), (r2, p2)):
        if seg_tri(a, b, p1, q1, r1):
            return True
    return False


def _poly_symmetries(P):
    """(proper, improper) symmetry counts of a closed polygon: rigid
    maps carrying the vertex cycle onto itself."""
    P = np.asarray(P, float)
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


def _cloud_match(A, B, R, tol=1e-6):
    """Does rotation R carry point set A onto point set B?"""
    T = (R @ np.asarray(A, float).T).T
    B = np.asarray(B, float)
    d = np.linalg.norm(T[:, None, :] - B[None, :, :], axis=2)
    return float(d.min(axis=1).max()) < tol and \
        float(d.min(axis=0).max()) < tol


def _congruent_proper(cloudA, axisA, twofoldsA, cloudB, axisB,
                      twofoldsB):
    """Is there a PROPER isometry taking decorated cell A onto B?

    Any such isometry must map the undecorated cell onto the
    undecorated cell, whose proper group is D3, so the candidates are
    exactly the rotations aligning A's axis frame with each of B's
    twelve signed D3 frames."""
    def frame(a, t):
        a = a / np.linalg.norm(a)
        t = t - (t @ a) * a
        t = t / np.linalg.norm(t)
        return np.stack([a, t, np.cross(a, t)], axis=1)

    FA = frame(np.asarray(axisA, float), np.asarray(twofoldsA[0], float))
    for sa in (1.0, -1.0):
        for t in twofoldsB:
            for st in (1.0, -1.0):
                FB = frame(sa * np.asarray(axisB, float),
                           st * np.asarray(t, float))
                R = FB @ FA.T
                if _cloud_match(cloudA, cloudB, R):
                    return True
    return False


def _selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-56s %s %s" % (name, "OK" if cond else "BAD", extra))

    print("spidron_spacefill: the triamond net")
    Vi, idx, adj = net_chunk(3)
    V = np.asarray(Vi, float) / 8.0
    full = [p for p in Vi if len(adj[p]) == 3]
    chk("3-connected (interior degree 3)", len(full) > 0 and
        all(len(adj[p]) <= 3 for p in Vi),
        "%d of %d interior" % (len(full), len(Vi)))
    worst_a, worst_p = 0.0, 0.0
    for p in full[:40]:
        e = [np.subtract(q, p).astype(float) / 8.0 for q in adj[p]]
        for i, j in combinations(range(3), 2):
            ca = float(e[i] @ e[j]) / (EDGE * EDGE)
            worst_a = max(worst_a, abs(ca + 0.5))
        ax = np.cross(e[1] - e[0], e[2] - e[0])
        ax /= np.linalg.norm(ax)
        worst_p = max(worst_p, max(abs(float(ei @ ax) - float(e[0] @ ax))
                                   for ei in e))
    chk("edges meet at 120 deg in a plane",
        worst_a < 1e-12 and worst_p < 1e-12,
        "%.1e / %.1e" % (worst_a, worst_p))
    p0 = full[0]
    ax = vertex_axis(p0, adj)
    chk("the plane normal is a body diagonal",
        float(np.abs(np.abs(ax) - 1.0 / sqrt(3.0)).max()) < 1e-12,
        "axis %s" % np.round(ax * sqrt(3.0), 6))

    print("spidron_spacefill: the decatrihedron")
    _V2, cells = enumerate_cells(3)
    chk("cells found", len(cells) > 0, "%d" % len(cells))
    key0 = min(cells)
    c0 = cells[key0]
    rings3 = c0['rings']
    edges = set()
    vs = set()
    for cyc in rings3:
        vs |= set(cyc)
        for i in range(len(cyc)):
            e = (cyc[i], cyc[(i + 1) % 10])
            edges.add(e if e[0] < e[1] else (e[1], e[0]))
    chk("F=3 E=15 V=14, Euler 2",
        len(rings3) == 3 and len(edges) == 15 and len(vs) == 14
        and len(vs) - len(edges) + len(rings3) == 2)
    inc = {}
    for cyc in rings3:
        for x in cyc:
            inc[x] = inc.get(x, 0) + 1
    chk("two 3-valent apexes, twelve 2-valent vertices",
        sorted(inc.values()).count(3) == 2
        and sorted(inc.values()).count(2) == 12)

    P0 = V[list(rings3[0])]
    els = [float(np.linalg.norm(P0[(i + 1) % 10] - P0[i]))
           for i in range(10)]
    chk("decagon is equilateral", max(els) - min(els) < 1e-12,
        "edge %.6f" % els[0])
    angs = []
    for i in range(10):
        a = P0[(i - 1) % 10] - P0[i]
        b = P0[(i + 1) % 10] - P0[i]
        angs.append(float(a @ b) /
                    (np.linalg.norm(a) * np.linalg.norm(b)))
    chk("decagon is equiangular at 120 deg",
        max(abs(c + 0.5) for c in angs) < 1e-12)
    N0 = sm._best_fit_normal(P0)
    h = (P0 - P0.mean(axis=0)) @ N0
    chk("decagon is skew (relief = one edge length)",
        abs((h.max() - h.min()) - EDGE) < 1e-12)
    prop, impr, orders = _poly_symmetries(P0)
    chk("decagon symmetry is 222: identity + three 2-folds",
        prop == 4 and sorted(orders) == [1, 2, 2, 2])
    chk("decagon admits no mirror (enantiomorphic)", impr == 0)
    # ... which is exactly the statement that no PROPER rotation maps
    # the mirror image back onto the original: a proper congruence of
    # the mirror composed with the mirror itself would be an improper
    # self-symmetry, and there are none.
    Pm = P0.copy()
    Pm[:, 0] = -Pm[:, 0]
    Qo = P0 - P0.mean(axis=0)
    Qm = Pm - Pm.mean(axis=0)
    proper_hit = False
    for s in range(10):
        for rev in (False, True):
            perm = [(s + (-k if rev else k)) % 10 for k in range(10)]
            H = Qo.T @ Qm[perm]
            U_, _S, Vt_ = np.linalg.svd(H)
            R_ = Vt_.T @ U_.T
            if (np.linalg.det(R_) > 0
                    and np.abs((R_ @ Qo.T).T - Qm[perm]).max() < 1e-9):
                proper_hit = True
    chk("decagon and its mirror are a chiral pair", not proper_hit)

    # cell symmetry: D3, and chiral
    CV = V[sorted(vs)] - 0.5 * (V[key0[0]] + V[key0[1]])
    axis = V[key0[1]] - V[key0[0]]
    axis = axis / np.linalg.norm(axis)

    def rot(axu, ang):
        axu = axu / np.linalg.norm(axu)
        c, s = np.cos(ang), np.sin(ang)
        x, y, z = axu
        return np.array([
            [c + x * x * (1 - c), x * y * (1 - c) - z * s,
             x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c),
             y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s,
             c + z * z * (1 - c)]])

    chk("cell has the 3-fold axis through its apexes",
        _cloud_match(CV, CV, rot(axis, 2 * pi / 3), 1e-9))
    twofolds = []
    for w in CV:
        u = w - (w @ axis) * axis
        if np.linalg.norm(u) < 1e-9:
            continue
        u = u / np.linalg.norm(u)
        if any(abs(abs(u @ t)) > 1 - 1e-9 for t in twofolds):
            continue
        if _cloud_match(CV, CV, rot(u, pi), 1e-9):
            twofolds.append(u)
    for i, j in combinations(range(len(CV)), 2):
        if len(twofolds) >= 3:
            break
        u = 0.5 * (CV[i] + CV[j])
        u = u - (u @ axis) * axis
        if np.linalg.norm(u) < 1e-6:
            continue
        u = u / np.linalg.norm(u)
        if any(abs(abs(u @ t)) > 1 - 1e-9 for t in twofolds):
            continue
        if _cloud_match(CV, CV, rot(u, pi), 1e-9):
            twofolds.append(u)
    chk("and exactly three 2-fold axes across it (D3)",
        len(twofolds) == 3)
    CM = CV.copy()
    CM[:, 0] = -CM[:, 0]
    axm = axis.copy()
    axm[0] = -axm[0]
    tfm = [t * np.array([-1.0, 1.0, 1.0]) for t in twofolds]
    chk("cell is chiral (mirror not properly congruent)",
        not _congruent_proper(CV, axis, twofolds, CM, axm, tfm))

    print("spidron_spacefill: the packing")
    owners = face_owners(cells)
    share = sorted(set(len(v) for v in owners.values()))
    chk("faces are shared by at most two cells", share[-1] == 2,
        "%d interior" % sum(1 for v in owners.values() if len(v) == 2))
    bad = sum(1 for f, ow in owners.items() if len(ow) == 2
              and cells[ow[0]]['colour'] == cells[ow[1]]['colour'])
    chk("adjacent cells always take opposite forms (bipartite)",
        bad == 0, "%d clashes" % bad)
    cents = set(tuple(int(round(x)) % 8 for x in c['centre8'])
                for c in cells.values())
    chk("cell centres lie on the mirror srs net (self-dual)",
        cents <= (_CLASS0 | _CLASS1), "%d classes" % len(cents))

    # CW meets CCW: the two owners of every shared face build the SAME
    # nest surface at gap 1
    worst = 0.0
    n_checked = 0
    for f, ow in owners.items():
        if len(ow) != 2:
            continue
        P = V[list(f)]
        Va, _fa, _ka = nest_face(P, cells[ow[0]]['centre'],
                                 cells[ow[0]]['colour'], 0.60,
                                 radians(12.0), 4)
        Vb, _fb, _kb = nest_face(P, cells[ow[1]]['centre'],
                                 cells[ow[1]]['colour'], 0.60,
                                 radians(12.0), 4)
        A = np.asarray(Va)
        B = np.asarray(Vb)
        d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        worst = max(worst, float(d.min(axis=1).max()))
        n_checked += 1
        if n_checked >= 40:
            break
    chk("CW meets CCW: shared faces coincide exactly",
        n_checked > 0 and worst < 1e-9,
        "%d faces, worst %.1e" % (n_checked, worst))

    # exactly two spidronised forms
    def decorated(key):
        c = cells[key]
        pts = []
        for f in c['rings']:
            v, _f2, _k2 = nest_face(V[list(f)], c['centre'],
                                    c['colour'], 0.60, radians(12.0), 3)
            pts.extend(v)
        pts = np.asarray(pts, float) - c['centre']
        a = V[key[1]] - V[key[0]]
        a = a / np.linalg.norm(a)
        tf = []
        CVk = np.asarray(sorted(set(tuple(x) for f in c['rings']
                                    for x in V[list(f)])))
        CVk = CVk - c['centre']
        for i, j in combinations(range(len(CVk)), 2):
            if len(tf) >= 3:
                break
            u = 0.5 * (CVk[i] + CVk[j])
            u = u - (u @ a) * a
            if np.linalg.norm(u) < 1e-6:
                continue
            u = u / np.linalg.norm(u)
            if any(abs(abs(u @ t)) > 1 - 1e-9 for t in tf):
                continue
            if _cloud_match(CVk, CVk, rot(u, pi), 1e-9):
                tf.append(u)
        for w in CVk:
            if len(tf) >= 3:
                break
            u = w - (w @ a) * a
            if np.linalg.norm(u) < 1e-9:
                continue
            u = u / np.linalg.norm(u)
            if any(abs(abs(u @ t)) > 1 - 1e-9 for t in tf):
                continue
            if _cloud_match(CVk, CVk, rot(u, pi), 1e-9):
                tf.append(u)
        return pts, a, tf

    by_col = {0: [], 1: []}
    for k, c in cells.items():
        by_col[c['colour']].append(k)
    k0a, k0b = sorted(by_col[0])[:2]
    k1a = sorted(by_col[1])[0]
    d0a, d0b, d1a = decorated(k0a), decorated(k0b), decorated(k1a)
    chk("cells of one colour are congruent (one form each)",
        _congruent_proper(d0a[0], d0a[1], d0a[2],
                          d0b[0], d0b[1], d0b[2]))
    chk("the two forms are NOT congruent -- two forms are needed",
        not _congruent_proper(d0a[0], d0a[1], d0a[2],
                              d1a[0], d1a[1], d1a[2]))

    print("spidron_spacefill: build")
    _v, _f, _m, _lb, info = build('CELL', rings=4)
    chk("single polyhedron builds", len(_f) > 0 and info['cells'] == 1,
        "V=%d F=%d" % (len(_v), len(_f)))
    _v, _f, _m, _lb, info = build('UNIT', rings=4, gap=1.0)
    chk("repeat unit is two cells, one of each form",
        info['cells'] == 2 and info['forms'] == [1, 1])
    chk("repeat unit has four external faces (3 + 3 - 2 shared)",
        info['faces'] == 5,
        "%d distinct decagons, 1 shared" % info['faces'])
    _Vu, cu, (c0u, c1u) = _default_cells('UNIT')
    fs0 = set(cu[c0u]['rings'])
    fs1 = set(cu[c1u]['rings'])
    chk("unit shares exactly one face, four faces free",
        len(fs0 & fs1) == 1 and len(fs0 ^ fs1) == 4)
    chk("no self-intersection at the default parameters",
        not unit_intersects(0.60, radians(12.0), 4))

    print("spidron_spacefill: safe-twist limits (chirality)")
    # The bare cell is chiral, so the safe region of a SINGLE cell is
    # NOT symmetric in twist sign -- and the two forms mirror each
    # other exactly (colour 1 is the translated cell wound the other
    # way).  Sweep both signs and both forms.
    chk("colour 0 clean at -22 deg (scale 0.60)",
        not form_intersects(0.60, radians(-22.0), 3, colour=0))
    chk("colour 0 collides at +22 deg -- asymmetric",
        form_intersects(0.60, radians(22.0), 3, colour=0))
    chk("colour 1 clean at +22 deg -- the mirror interval",
        not form_intersects(0.60, radians(22.0), 3, colour=1))
    chk("colour 1 collides at -22 deg",
        form_intersects(0.60, radians(-22.0), 3, colour=1))
    for s in (0.50, 0.60, 0.80):
        neg, pos = twist_limits(s)
        eps = radians(0.05)
        chk("scale %.2f: clean at both tabulated limits" % s,
            not form_intersects(s, pos - eps, 3)
            and not form_intersects(s, -(neg - eps), 3),
            "-%.1f..+%.1f deg" % (np.degrees(neg), np.degrees(pos)))
        past = radians(1.0)
        chk("scale %.2f: collides just past the limits" % s,
            (pos > radians(59.0) or form_intersects(s, pos + past, 3))
            and (neg > radians(59.0)
                 or form_intersects(s, -(neg + past), 3)))
    # the space-filling layouts get the symmetric packed bound, which
    # a cross-cell collision makes slightly tighter than the intra-
    # cell limit
    _n60, pos60 = twist_limits(0.60)
    sym60 = twist_limit_packed(0.60)
    chk("packed bound is tighter than the intra-cell one",
        sym60 < pos60,
        "%.2f < %.2f deg" % (np.degrees(sym60), np.degrees(pos60)))
    eps = radians(0.05)
    chk("unit clean at both ends of the packed interval",
        not unit_intersects(0.60, sym60 - eps, 3)
        and not unit_intersects(0.60, -(sym60 - eps), 3))
    chk("unit collides just past it",
        unit_intersects(0.60, sym60 + radians(1.0), 3)
        and unit_intersects(0.60, -(sym60 + radians(1.0)), 3))
    # and build() clamps per layout when asked
    _v, _f, _m, _lb, ci = build('CELL', rings=3, twist=radians(22.0),
                                limit_twist=True)
    chk("CELL build clamps +22 deg to the colour-0 limit",
        ci['twist_clamped']
        and abs(ci['twist'] - pos60) < 1e-9)
    _v, _f, _m, _lb, ci = build('CELL', rings=3, twist=radians(-22.0),
                                limit_twist=True)
    chk("CELL build leaves -22 deg alone (room on that side)",
        not ci['twist_clamped'])
    _v, _f, _m, _lb, ci = build('UNIT', rings=3, twist=radians(-22.0),
                                limit_twist=True)
    chk("UNIT build clamps -22 deg to the packed bound",
        ci['twist_clamped'] and abs(ci['twist'] + sym60) < 1e-9)

    _v, _f, _m, _lb, info = build('BLOCK', nx=1, ny=1, nz=1, rings=3)
    chk("one lattice cell holds eight polyhedra",
        info['cells'] == 8 and info['forms'] == [4, 4],
        "%d cells, forms %s" % (info['cells'], info['forms']))
    A = np.asarray(_v, float)
    ctr = 0.5 * (A.max(axis=0) + A.min(axis=0))
    chk("output is centred", float(np.abs(ctr).max()) < 1e-9)
    Af = np.asarray(_fit.fit_cube(_v, 2.0), float)
    ext = Af.max(axis=0) - Af.min(axis=0)
    chk("fits the 2 m cube", abs(float(ext.max()) - 2.0) < 1e-9,
        "extent %.6f" % float(ext.max()))
    vm, fm, _mm, _lm, _im = build('CELL', rings=3, mirror=True)
    v0, f0, _m0, _l0, _i0 = build('CELL', rings=3, mirror=False)
    chk("mirror form differs from the original",
        np.abs(np.asarray(vm) - np.asarray(v0)).max() > 1e-6)

    print("RESULT:", "OK" if ok else "BAD")
    return ok
