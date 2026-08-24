
# Monohedral Tiling Generator for Blender
#
# Tilings of the plane by a SINGLE prototile that need not be regular:
#   * any triangle (paired with its 180 deg rotate -> a parallelogram);
#   * any quadrilateral, even non-convex (180 deg rotations about the
#     edge midpoints -- a centrally symmetric hexagon dimer);
#   * the 3 convex hexagons that tile (Reinhardt Types 1-3);
#   * the 15 convex pentagons that tile the plane;
#   * Durer's tiling (1525), a NON-PERIODIC member;
#   * Durer's ORIGINAL pentagon-and-rhombus tiling (1525) -- strictly
#     DIHEDRAL (two prototiles), kept here as the historical companion
#     of the hexagon version above; see its section for the exception.
#
# Each tiling is described by a TRANSLATIONAL UNIT CELL: two lattice
# vectors b1, b2 plus the tile copies (rotated / reflected / glided)
# that fill one cell.  The cell is replicated over an nx x ny run.  A
# per-tile "type" index records each copy's orientation orbit so the
# TYPE color mode can paint the tile's dance across the plane.
#
# Part of the Pattern Engine (see the `patterns` package).  The heavy lifting
# of color / margin / relief / trim is delegated to the shared
# tiling_generator.cells_from_polys assembler.
#
# References:
# - Karl Reinhardt, "Uber die Zerlegung der Ebene in Polygone" (doctoral
#   dissertation, 1918) -- the 3 convex hexagon types and first 5
#   convex pentagon types that tile the plane.
# - Richard B. Kershner, "On paving the plane" (1968) -- further pentagon
#   types.
# - Marjorie Rice -- amateur discovery of additional pentagon types
#   (1970s).
# - Michael Rao, "Exhaustive search of convex pentagons which tile the
#   plane" (2017) -- proof that exactly 15 convex pentagon types exist.
# - Albrecht Durer, "Underweysung der Messung mit dem Zirckel und
#   Richtscheyt", Book II (Nuremberg, 1525) -- the radial tiling; the
#   construction followed here (base hexagon, wedge offsets and the
#   interleaved second family) is Robert Ferreol's, in "Encyclopedie des
#   formes mathematiques remarquables" (mathcurve.com), "pavage de
#   Durer".  Durer's own n = 5 figure replaces the hexagon by a regular
#   pentagon plus a 36-degree rhombus; that dihedral tiling IS built
#   here (DURER5, radial and two periodic variants).  The substitution
#   is combinatorial, not a dissection -- a unit pentagon and rhomb
#   total area 2.308 against the hexagon's 2.127, so the metric layout
#   had to be reconstructed from the source's figures; the DURER5
#   section records what was established.

bl_info = {
    "name": "Monohedral Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Monohedral tilings: any triangle / quadrilateral, "
                   "the 3 convex hexagon types, and the 15 convex "
                   "pentagon types that tile the plane",
    "category": "Add Mesh",
}

from math import cos, sin, pi, radians, atan2, hypot, sqrt, tan
import numpy as np

try:
    from .patterns import common as pc
    from . import tiling_generator as tg
except Exception:
    from patterns import common as pc
    import tiling_generator as tg


# --------------------------------------------------------------------
# Geometry helpers -- polygons and isometries (pure numpy)
# --------------------------------------------------------------------

def _poly(edges, interior_deg, start_heading=0.0):
    """Build a polygon (CCW) from consecutive edge lengths and interior
    angles.  Vertex 0 sits at the origin; edge i runs from vertex i to
    vertex i+1, leaving vertex 0 along `start_heading`.  At each vertex
    the heading turns LEFT by the exterior angle (pi - interior).  The
    caller is responsible for supplying a consistent (closing) set."""
    n = len(edges)
    pts = [(0.0, 0.0)]
    x, y = 0.0, 0.0
    h = start_heading
    for i in range(n):
        x += edges[i] * cos(h)
        y += edges[i] * sin(h)
        if i < n - 1:
            pts.append((x, y))
        h += pi - radians(interior_deg[(i + 1) % n])
    return np.array(pts, float)


def _solve_pentagon(interior_deg, e0, e1, e2):
    """A closed pentagon from its 5 interior angles (summing to 540)
    and the first three edge lengths e0, e1, e2; the remaining two edge
    lengths are solved from the closure condition sum(edge_i * dir_i)=0,
    so the polygon closes exactly.  Returns the (5, 2) vertex array."""
    # headings of the five edges (edge 0 along +x)
    h = [0.0]
    for i in range(4):
        h.append(h[-1] + pi - radians(interior_deg[(i + 1) % 5]))
    d = [np.array([cos(a), sin(a)]) for a in h]
    # e0 d0 + e1 d1 + e2 d2 + e3 d3 + e4 d4 = 0  ->  [d3 d4] [e3 e4]^T = -S
    S = e0 * d[0] + e1 * d[1] + e2 * d[2]
    M = np.column_stack([d[3], d[4]])
    e3, e4 = np.linalg.solve(M, -S)
    edges = [e0, e1, e2, float(e3), float(e4)]
    return _poly(edges, interior_deg), edges


def _solve_hexagon(interior_deg, e0, e1, e2, e3):
    """A closed hexagon from its 6 interior angles (summing to 720) and
    the first four edge lengths; the last two are solved from closure."""
    h = [0.0]
    for i in range(5):
        h.append(h[-1] + pi - radians(interior_deg[(i + 1) % 6]))
    d = [np.array([cos(a), sin(a)]) for a in h]
    S = e0 * d[0] + e1 * d[1] + e2 * d[2] + e3 * d[3]
    M = np.column_stack([d[4], d[5]])
    e4, e5 = np.linalg.solve(M, -S)
    return _poly([e0, e1, e2, e3, float(e4), float(e5)], interior_deg)


def _place(base, ops):
    """Apply a list of 3x3 isometry matrices to a base polygon, tagging
    each image with its index -> a list of (poly, type) for one cell."""
    return [(pc.apply(M, base), i) for i, M in enumerate(ops)]


# _Rot and _Mir are signature ADAPTERS, not reimplementations: they take
# the centre as a tuple where the engine takes two arguments.  _I was a
# genuine duplicate of the engine's identity and now delegates.
def _Rot(theta, c=(0.0, 0.0)):
    return pc.Rot(theta, c[0], c[1])


def _Mir(ang, c=(0.0, 0.0)):
    return pc.Mir(ang, c[0], c[1])


def _I():
    return pc.I()


def _rot(poly, theta, c=(0.0, 0.0)):
    return pc.apply(pc.Rot(theta, c[0], c[1]), poly)


def _rot180(poly, c):
    return pc.apply(pc.Rot(pi, c[0], c[1]), poly)


def _reflect(poly, P, Q):
    """Reflect a polygon across the line through P and Q."""
    ang = atan2(Q[1] - P[1], Q[0] - P[0])
    return pc.apply(pc.Mir(ang, P[0], P[1]), poly)


def _mid(P, Q):
    return (0.5 * (P[0] + Q[0]), 0.5 * (P[1] + Q[1]))


def _V(poly):
    """Return the vertices as an array (identity, for readability)."""
    return np.asarray(poly, float)


# --------------------------------------------------------------------
# Unit cells
# --------------------------------------------------------------------
#
# _unit_cell(name) -> (b1, b2, cell) where cell is a list of
# (poly, type_index).  build_patch replicates it over nx x ny.

def _tri_cell():
    """Any triangle tiles: the triangle plus its 180 deg rotate about an
    edge midpoint make a parallelogram whose sides are the lattice."""
    A = np.array([0.0, 0.0])
    B = np.array([1.3, 0.0])
    C = np.array([0.35, 0.95])
    tri = np.array([A, B, C])
    M = _mid(B, C)
    tri2 = _rot180(tri, M)
    return (B - A), (C - A), [(tri, 0), (tri2, 1)]


def _quad_cell():
    """Any quadrilateral tiles by 180 deg rotation about edge midpoints;
    the tile plus one rotate form a centrally symmetric hexagon whose
    three edge-pair vectors give the lattice.  A deliberately NON-convex
    (dart) quad is used to show the construction needs no convexity."""
    P0 = np.array([0.0, 0.0])
    P1 = np.array([2.0, 0.4])
    P2 = np.array([1.4, 1.0])           # reflex vertex -> dart
    P3 = np.array([2.2, 2.4])
    quad = np.array([P0, P1, P2, P3])
    # pair across an edge NOT touching the reflex vertex so the dimer
    # stays simple; the lattice (edge-midpoint half-turns) is unchanged.
    M01 = _mid(P0, P1)
    quad2 = _rot180(quad, M01)
    b1 = P2 - P0
    b2 = P3 - P1
    return b1, b2, [(quad, 0), (quad2, 1)]


def _hex1_cell():
    """Hexagon Type 1 (b = e, B + C + D = 360): a centrally symmetric
    hexagon -- opposite sides equal and parallel.  It tiles by pure
    translation; the three edge-pair sums give the lattice."""
    v0 = np.array([1.10, -0.30])
    v1 = np.array([0.90, 0.70])
    v2 = np.array([-0.20, 0.90])
    v = np.array([v0, v1, v2, -v0, -v1, -v2])
    b1 = v0 + v1
    b2 = v1 + v2
    return b1, b2, [(v, 0)]


def _hex2_cell():
    """Hexagon Type 2 (b = e, d = f, B + C + E = 360).  The tile plus its
    180 deg rotate about a side midpoint form a translational dimer."""
    P = _solve_hexagon([130, 130, 100, 130, 130, 100], 1.0, 1.0, 1.0, 1.0)
    m = _mid(P[0], P[1])
    ops = [_I(), _Rot(pi, m)]
    b1 = np.array([1.6428, 0.7660])
    b2 = np.array([0.0, -3.0642])
    return b1, b2, _place(P, ops)


# --------------------------------------------------------------------
# The 15 convex pentagon types
# --------------------------------------------------------------------
#
# Each pentagon type is a family of convex pentagons defined by angle /
# edge constraints; a representative member is built here and tiled by
# its known isometry unit cell (a mix of 180/120/90/60 deg rotations,
# reflections and glides over a lattice).  Pentagon tilings are usually
# NOT edge-to-edge, so the coverage self-test -- not a vertex check --
# is what certifies them.

def _pent1_cell():
    """Pentagon Type 1 (B + C = 180): a generic member, so a pair of
    sides is parallel.  Tiles p2 -- the pentagon plus its 180 deg rotate
    about a side midpoint make a hexagon that tiles by translation."""
    P, _ = _solve_pentagon([100, 110, 70, 140, 120], 2.0, 2.0, 2.0)
    m = _mid(P[1], P[2])
    ops = [_I(), _Rot(pi, m)]
    b1 = np.array([0.6840, 1.8794])
    b2 = np.array([4.2007, -1.1372])
    return b1, b2, _place(P, ops)


def _pent2_cell():
    """Pentagon Type 2 (c = e, B + D = 180): here the highly symmetric
    'floret' member (four 120 deg angles and one 60 deg), tiled p6 as
    six copies pinwheeling about the 60 deg vertex on a hexagonal
    lattice."""
    s = 1.0 / sqrt(3.0)
    P = _poly([2 * s, s, s, s, 2 * s], [60, 120, 120, 120, 120])
    ops = [_Rot(pi / 3.0 * k) for k in range(6)]
    b1 = np.array([-2.5981, 0.5])
    b2 = np.array([0.8660, -2.5])
    return b1, b2, _place(P, ops)


def _pent5_cell():
    """Pentagon Type 5 (a = b, d = e, A = 60, D = 120): a generic member.
    The pentagon plus its 180 deg rotate about a side midpoint form a
    translational dimer."""
    P, _ = _solve_pentagon([60, 160, 80, 120, 120], 0.9358, 0.9358, 0.6304)
    m = _mid(P[1], P[2])
    ops = [_I(), _Rot(pi, m)]
    b1 = np.array([2.2510, -0.5459])
    b2 = np.array([0.5642, 0.8660])
    return b1, b2, _place(P, ops)


# --------------------------------------------------------------------
# Durer's tiling (Underweysung der Messung, Book II, 1525)
# --------------------------------------------------------------------
#
# The prototile is a centrally symmetric hexagon with two angles of
# 2.pi/n and four of (n-1).pi/n -- the zonogon of the three unit vectors
# at -a, 0 and +a, where a = pi/n.  Being centrally symmetric it tiles by
# TRANSLATION, which is the ordinary periodic thing to do with it.
#
# Durer's idea was to tile by ROTATION instead.  Lay a triangular array of
# the hexagons in a wedge, then repeat that wedge by turns of 2.pi/n about
# the centre.  The result is monohedral and covers the plane, but it has
# NO translational symmetry at all -- only the n-fold rotation -- while
# still containing arbitrarily large patches of the periodic tiling made
# from the same piece.  That combination is what makes it interesting: a
# non-periodic monohedral tiling long before non-periodicity was studied,
# and unlike a Penrose tiling its centre is unrepeatable, since no
# neighbourhood of it occurs anywhere else in the plane.

def _durer_hexagon(order):
    """The base hexagon: the zonogon of unit vectors at -a, 0, +a."""
    a = pi / order
    em = np.array([cos(a), -sin(a)])
    e0 = np.array([1.0, 0.0])
    ep = np.array([cos(a), sin(a)])
    P = np.array([np.zeros(2), em, em + e0, em + e0 + ep, e0 + ep, ep])
    area = float(np.sum(P[:, 0] * np.roll(P[:, 1], -1)
                        - np.roll(P[:, 0], -1) * P[:, 1]))
    return P if area > 0 else P[::-1]


def _durer_cell(order=5):
    """The PERIODIC use of the piece: it tiles by translation alone.

    A centrally symmetric hexagon with edge vectors u, v, w tiles by the
    lattice generated by u + v and v + w.
    """
    a = pi / order
    em = np.array([cos(a), -sin(a)])
    e0 = np.array([1.0, 0.0])
    ep = np.array([cos(a), sin(a)])
    return em + e0, e0 + ep, _place(_durer_hexagon(order), [_I()])


def build_durer_radial(order=5, depth=3):
    """Durer's ROTATIONAL layout: 2n wedges of a triangular array.

    Following Durer's construction as set down by Ferreol: wedge m is
    turned by 2.m.pi/n, and within a wedge the piece at (r, k) is offset
    by r + k.e^{ia} + (r-k).e^{-ia} for 0 <= k <= r <= depth.  A second
    family of wedges, offset by one step and turned by a further pi/n,
    interleaves with the first and closes the plane up.
    """
    a = pi / order
    em = np.array([cos(a), -sin(a)])
    ep = np.array([cos(a), sin(a)])
    base = _durer_hexagon(order)
    polys, types = [], []
    for m in range(order):
        for r in range(depth + 1):
            for k in range(r + 1):
                for second in (False, True):
                    off = (np.array([r + 1.0 if second else r, 0.0])
                           + k * ep + (r - k) * em)
                    ang = 2 * m * a + (a if second else 0.0) + a / 2.0
                    polys.append(pc.apply(_Rot(ang), base + off))
                    types.append(r % 8)
    return polys, types


# --------------------------------------------------------------------
# Durer's ORIGINAL pentagon-and-rhombus tiling (DURER5)
# --------------------------------------------------------------------
#
# Durer's own 1525 plate does not show the hexagon: for n = 5 he drew a
# patch of unit REGULAR PENTAGONS and unit 36-degree RHOMBI around a
# central pentagon.  Ferreol presents this as the hexagon tiling with
# each hexagon replaced by a pentagon-and-rhomb motif, but the
# substitution cannot be a dissection (the areas differ), so the actual
# metric layout was reconstructed here from the source's figures and
# verified computationally.  What was established:
#
#   * Only two vertex figures occur: 108+108+108+36 (three pentagons
#     and a rhomb tip) and 108+108+144 (two pentagons and a rhomb's fat
#     corner).  All edges are unit, every edge direction is a multiple
#     of 36 degrees, so every vertex lies in the ring Z[zeta_10] -- the
#     reconstruction was done in exact cyclotomic arithmetic.
#   * The radial tiling is 10 wedges about a central pentagon, wedge j
#     on axis 18 + 36.j degrees.  Even wedges are anchored at the
#     centre pentagon's five VERTICES, odd wedges further out at the
#     apexes of the five first-corona pentagons -- exactly Ferreol's
#     two interleaved hexagon families (Piece / Piece2).
#   * Within a wedge, nodes repeat on the lattice tau_j, tau_{j+1}
#     where tau_j = phi^2 . (cos 36j, sin 36j) -- the golden ratio
#     squared, phi^2 = 2.618..., against 2.cos(18) = 1.902 for the
#     hexagons.  Row n holds n+1 nodes, Durer's 1, 2, 3, ... wedges.
#   * Each node carries one thin rhomb (tip at the node, long axis on
#     the wedge) and two pentagons; tiles shared between wedges (the
#     centre pentagon belongs to all five even wedges) are deduplicated
#     by vertex set.  Growing this to depth 9 (1551 tiles) covers a
#     disc of radius 18 with no gap or overlap, so the arrangement
#     continues to the whole plane -- Durer's patch does extend.
#   * The result has exact 5-fold rotational symmetry, and therefore NO
#     translational symmetry at all (the crystallographic restriction:
#     a plane tiling with a global 5-fold rotation admits no
#     translation), with an unrepeatable centre like the hexagon
#     version.
#
# The same two tiles also tile PERIODICALLY; the source shows both
# variants, built here as unit cells: pmm (mirror pairs of pentagons
# with upright rhombi between the rows) and p2 (strips of alternating
# up/down pentagons with all rhombi leaning the same way, which kills
# the mirrors).  Both are 2 pentagons + 1 rhomb per cell.
#
# This module is otherwise monohedral; DURER5 is the one DIHEDRAL entry,
# kept here deliberately as the historical companion of the DURER
# hexagon above -- same source, same wedge construction, same absent
# translation -- rather than exiled to a module that shares none of
# that machinery.

_D5_DIR = [np.array([cos(pi / 5 * k), sin(pi / 5 * k)]) for k in range(10)]
_D5_PHI2 = (3.0 + sqrt(5.0)) / 2.0          # phi^2, the node lattice pitch
_D5_RC = 1.0 / (2.0 * sin(pi / 5))          # pentagon circumradius
_D5_RIN = 1.0 / (2.0 * tan(pi / 5))         # pentagon inradius


def _d5_pent(v, k):
    """Unit regular pentagon walked CCW from vertex v, first edge along
    direction 36k degrees; interior angles of 108 turn the heading by
    72 = two grid steps, so the edge directions are k, k+2, ... k+8."""
    pts = [np.asarray(v, float)]
    for j in range(4):
        pts.append(pts[-1] + _D5_DIR[(k + 2 * j) % 10])
    return np.array(pts)


def _d5_rhomb(v, k, fat=False):
    """Unit 36-degree rhombus with a corner at v: the sharp (36) corner
    when the edges leave along adjacent grid directions k, k+1, the fat
    (144) corner when they straddle four steps, k and k+4."""
    a, b = _D5_DIR[k % 10], _D5_DIR[(k + (4 if fat else 1)) % 10]
    v = np.asarray(v, float)
    return np.array([v, v + a, v + a + b, v + b])


def _durer5_cell(variant='PMM'):
    """The two PERIODIC pentagon-and-rhombus tilings shown by Ferreol.

    PMM: mirror-image pentagon pairs share a horizontal edge; upright
    rhombi (fat corners on the pair axis) fill the 144-degree notches.
    P2: strips of alternating up/down pentagons; a rhombus leans on
    each down-pentagon's top edge, all leaning the same way, so only
    half-turns survive.  Cells are 2 pentagons + 1 rhomb either way.
    """
    A = np.zeros(2)
    if variant == 'P2':
        C = _D5_DIR[0] + _D5_DIR[2]              # third vertex of the up tile
        cell = [(_d5_pent(A, 0), 0), (_d5_pent(C, 7), 1),
                (_d5_rhomb(C, 0, fat=True), 2)]
        return _D5_PHI2 * _D5_DIR[0], _D5_DIR[1] + _D5_DIR[3], cell
    B = _D5_DIR[0]
    cell = [(_d5_pent(A, 0), 0), (_d5_pent(A, 7), 1),
            (_d5_rhomb(B, 8, fat=True), 2)]
    return (_D5_PHI2 - 1.0) * _D5_DIR[0], _D5_PHI2 * _D5_DIR[2], cell


def build_durer5_radial(depth=3):
    """Durer's original 1525 arrangement: 10 wedges of pentagon-and-rhomb
    nodes about a central pentagon, growing 1, 2, 3, ... per row.

    Wedge j sits on axis 18 + 36j degrees; even wedges anchor at the
    centre pentagon's vertices, odd wedges at the first-corona apexes.
    Node (n, k) is the anchor offset by k.tau_{j+1} + (n-k).tau_j and
    carries a thin rhomb (tip at the node) and two pentagons; tiles on
    wedge seams coincide between wedges and are emitted once.  Types:
    pentagon 0, rhombus 1 -- the two prototiles.
    """
    c0 = np.array([0.5, _D5_RIN])               # centre pentagon's centre
    seen = set()
    polys, types = [], []

    def emit(p, t):
        key = frozenset((round(float(x), 6), round(float(y), 6))
                        for x, y in p)
        if key not in seen:
            seen.add(key)
            polys.append(p)
            types.append(t)

    for j in range(10):
        ax = _D5_DIR[j] + _D5_DIR[(j + 1) % 10]
        ax = ax / np.linalg.norm(ax)            # wedge axis, 18 + 36j deg
        rad = _D5_RC if j % 2 == 0 else _D5_RC + 2.0 * _D5_RIN
        anchor = c0 + rad * ax
        for n in range(depth + 1):
            for k in range(n + 1):
                a = (anchor + _D5_PHI2
                     * (k * _D5_DIR[(j + 1) % 10] + (n - k) * _D5_DIR[j]))
                emit(_d5_rhomb(a, j), 1)
                emit(_d5_pent(a, (4 + j) % 10), 0)
                emit(_d5_pent(a, (7 + j) % 10), 0)
    return polys, types


# Registry of finished tilings: name -> (label, builder)
_CELLS = {
    'TRIANGLE': ("Triangle", _tri_cell),
    'QUAD': ("Quadrilateral", _quad_cell),
    'PENT1': ("Pentagon Type 1", _pent1_cell),
    'PENT2': ("Pentagon Type 2", _pent2_cell),
    'PENT5': ("Pentagon Type 5", _pent5_cell),
    'HEX1': ("Hexagon Type 1", _hex1_cell),
    'HEX2': ("Hexagon Type 2", _hex2_cell),
    'DURER': ("Durer's Tiling", _durer_cell),
    'DURER5': ("Durer Pentagons (1525)", _durer5_cell),
}


def _unit_cell(name):
    return _CELLS[name][1]()


def build_patch(name, nx, ny, layout='LATTICE', order=5, variant='PMM'):
    """Replicate a tiling's unit cell over an nx x ny run of the lattice.
    Returns (polys, types): a list of (N, 2) CCW polygon arrays and a
    parallel list of per-tile orientation-orbit indices.

    The two Durer tilings also accept layout='RADIAL', which abandons
    the lattice for the rotational arrangement; there `nx` sets how many
    rings deep the wedges go and `ny` is unused.  For DURER5 the
    periodic layout picks its wallpaper group through `variant` ('PMM'
    or 'P2').
    """
    if name == 'DURER' and layout == 'RADIAL':
        return build_durer_radial(order, max(1, nx))
    if name == 'DURER5' and layout == 'RADIAL':
        return build_durer5_radial(max(1, nx))
    if name == 'DURER':
        b1, b2, cell = _durer_cell(order)
    elif name == 'DURER5':
        b1, b2, cell = _durer5_cell(variant)
    else:
        b1, b2, cell = _unit_cell(name)
    b1 = np.asarray(b1, float)
    b2 = np.asarray(b2, float)
    polys, types = [], []
    for i in range(nx):
        for j in range(ny):
            off = i * b1 + j * b2
            for poly, t in cell:
                polys.append(np.asarray(poly, float) + off)
                types.append(int(t))
    return polys, types


# ordered list of implemented tiling keys (menu + self-test order)
_ORDER = ['TRIANGLE', 'QUAD', 'PENT1', 'PENT2', 'PENT5', 'HEX1', 'HEX2',
          'DURER', 'DURER5']


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

def _items():
    return [(k, _CELLS[k][0], _CELLS[k][0]) for k in _ORDER]


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_monohedral_add(bpy.types.Operator, AddObjectHelper):
        """Add a monohedral tiling (single irregular prototile)"""
        bl_idname = "mesh.monohedral_add"
        bl_label = "Monohedral Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        tiling: EnumProperty(name="Tiling", items=_items(),
                             default='TRIANGLE',
                             description="Which single-prototile tiling "
                                         "to build")
        nx: IntProperty(name="Cells X", default=5, min=1, max=40,
                        description="Number of unit cells along X")
        ny: IntProperty(name="Cells Y", default=5, min=1, max=40,
                        description="Number of unit cells along Y")
        color_by: EnumProperty(
            name="Color By",
            items=[('SIDES', "By Sides",
                    "Material per polygon side count"),
                   ('TYPE', "By Tile Type",
                    "Material per tile orientation orbit in the cell"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='TYPE',
            description="How tile materials are assigned")
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat 2D mesh; > 0 extrudes each tile")
        trim: BoolProperty(
            name="Trim Boundary", default=False,
            description="Clip the patch to a clean rectangle, removing "
                        "the ragged edge and stray boundary tiles")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object "
                        "(parented to an empty)")
        durer_layout: EnumProperty(
            name="Layout",
            description="How the tiles are repeated (the two Durer "
                        "tilings only)",
            items=[('LATTICE', "Periodic",
                    "Repeat by translation -- the ordinary periodic use "
                    "of the same tiles"),
                   ('RADIAL', "Radial (Durer)",
                    "Durer's rotational arrangement: wedges turned about "
                    "a centre, which covers the plane with NO "
                    "translational symmetry at all")],
            default='RADIAL')
        durer_order: IntProperty(
            name="Order", default=5, min=3, max=16,
            description="Rotational order n; the prototile has two "
                        "angles of 360/n degrees.  Durer drew n = 5 "
                        "(Durer's tiling only)")
        durer5_variant: EnumProperty(
            name="Wallpaper Group",
            description="Which periodic arrangement of the pentagon and "
                        "rhombus to build (Durer pentagons, periodic "
                        "layout only)",
            items=[('PMM', "pmm",
                    "Mirror pairs of pentagons with upright rhombi; two "
                    "perpendicular mirror axes"),
                   ('P2', "p2",
                    "Strips of alternating pentagons with every rhombus "
                    "leaning the same way; half-turns only, no mirrors")],
            default='PMM')

        def execute(self, context):
            label = _CELLS[self.tiling][0]
            lay = (self.durer_layout if self.tiling in ('DURER', 'DURER5')
                   else 'LATTICE')
            cells = tg.cells_from_polys(
                lambda a, b: build_patch(self.tiling, a, b, lay,
                                         self.durer_order,
                                         self.durer5_variant),
                self.nx, self.ny, self.color_by, self.margin,
                self.height, self.trim)
            obj = pc.emit(context, "Monohedral %s" % label, cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no tiling generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s  V=%d F=%d" %
                            (label, len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s  %d tiles" %
                            (label, len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'tiling')
            if self.tiling in ('DURER', 'DURER5'):
                lay.prop(self, 'durer_layout')
            if self.tiling == 'DURER':
                lay.prop(self, 'durer_order')
            if self.tiling == 'DURER5' and self.durer_layout == 'LATTICE':
                lay.prop(self, 'durer5_variant')
            lay.prop(self, 'nx')
            if not (self.tiling in ('DURER', 'DURER5')
                    and self.durer_layout == 'RADIAL'):
                lay.prop(self, 'ny')      # radial depth uses nx alone
            for p in ('color_by', 'margin', 'height'):
                lay.prop(self, p)
            lay.prop(self, 'trim')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.monohedral_add", icon='MESH_GRID')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_monohedral_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_monohedral_add)


# --------------------------------------------------------------------
# Self-test (pure Python): coverage check
# --------------------------------------------------------------------
#
# Pentagon tilings are frequently NOT edge-to-edge, so a vertex-angle
# test would be meaningless.  Instead we sample a dense grid and require
# every sample point to be covered by EXACTLY ONE tile: a count of 0 is
# a gap, >= 2 an overlap.

def _coverage(name, nx=6, ny=6, N=35, half=2.5):
    polys, _ = build_patch(name, nx, ny)
    allv = np.vstack([np.asarray(p, float) for p in polys])
    C = allv.mean(axis=0)
    defects = 0
    for ix in range(N):
        gx = C[0] - half + (2.0 * half) * ix / (N - 1) + 0.0137
        for iy in range(N):
            gy = C[1] - half + (2.0 * half) * iy / (N - 1) + 0.0071
            cnt = 0
            for p in polys:
                if tg._pip(p, gx, gy):
                    cnt += 1
                    if cnt > 1:
                        break
            if cnt != 1:
                defects += 1
    return len(polys), defects


def _selftest():
    bad = []
    for name in _ORDER:
        ntiles, defects = _coverage(name)
        ok = defects == 0
        print("%-22s tiles=%4d defects=%4d %s" %
              (_CELLS[name][0], ntiles, defects, "OK" if ok else "BAD"))
        if not ok:
            bad.append(name)

    # Durer's RADIAL layout, where the whole point is what it does NOT
    # have.  The prototile is checked first (unit sides, two angles of
    # 360/n and four of 180 - 180/n), then the arrangement: it covers
    # without gap or overlap, it IS invariant under a 1/n turn, and it is
    # NOT invariant under any translation carrying one tile onto another.
    for order in (5, 7, 8):
        H = _durer_hexagon(order)
        sides = {round(float(np.linalg.norm(H[(i + 1) % 6] - H[i])), 9)
                 for i in range(6)}
        assert sides == {1.0}, (order, sides)
        angs = []
        for i in range(6):
            u, w = H[i - 1] - H[i], H[(i + 1) % 6] - H[i]
            angs.append(round(np.degrees(np.arccos(
                float(u @ w) / np.linalg.norm(u) / np.linalg.norm(w))), 6))
        assert sorted(angs) == sorted([round(360.0 / order, 6)] * 2
                                      + [round(180.0 - 180.0 / order, 6)]
                                      * 4), (order, angs)
        polys, _t = build_durer_radial(order, 3)
        assert len(polys) == 2 * order * 10, (order, len(polys))
        cen = np.array([np.asarray(p, float).mean(axis=0) for p in polys])
        defects = 0
        for k in range(0, len(cen), 3):    # sample inside the covered disc
            gx, gy = cen[k] + np.array([0.0137, 0.0071])
            if np.hypot(gx, gy) > 2.4:
                continue
            cnt = sum(1 for p in polys if tg._pip(p, gx, gy))
            if cnt != 1:
                defects += 1
        assert defects == 0, (order, defects)
        th = 2 * np.pi / order
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

        def key(A):
            return sorted((round(float(x), 6), round(float(y), 6))
                          for x, y in A)
        assert key(cen) == key(cen @ R.T), ("not %d-fold" % order)
        moved = any(key(cen + (t - cen[0])) == key(cen)
                    for t in cen[1:24])
        assert not moved, ("a translation fixes it -- that is periodic",
                           order)
    print("Durer radial          prototile exact; covers with no gap or "
          "overlap, n-fold rotation yes, translation symmetry none")

    # Durer's ORIGINAL pentagon-and-rhombus tiling.  The p2 periodic
    # variant first (pmm went through _coverage above), then the radial
    # arrangement: every tile a unit regular pentagon or unit 36-degree
    # rhomb, exact coverage, 5-fold rotation present, translation absent.
    polys, _t = build_patch('DURER5', 6, 6, 'LATTICE', variant='P2')
    allv = np.vstack([np.asarray(p, float) for p in polys])
    C = allv.mean(axis=0)
    defects = 0
    for ix in range(25):
        for iy in range(25):
            gx = C[0] - 2.5 + 5.0 * ix / 24 + 0.0137
            gy = C[1] - 2.5 + 5.0 * iy / 24 + 0.0071
            if sum(1 for p in polys if tg._pip(p, gx, gy)) != 1:
                defects += 1
    assert defects == 0, ("DURER5 p2", defects)

    polys, kinds = build_durer5_radial(3)
    assert len(polys) == 261, len(polys)
    for p, kd in zip(polys, kinds):
        P = np.asarray(p, float)
        n = len(P)
        assert n == (5 if kd == 0 else 4), (n, kd)
        angs = []
        for i in range(n):
            u, w = P[i - 1] - P[i], P[(i + 1) % n] - P[i]
            lu, lw = np.linalg.norm(u), np.linalg.norm(w)
            assert abs(lw - 1.0) < 1e-9, lw          # unit edges
            angs.append(round(float(np.degrees(
                np.arccos(np.clip(u @ w / lu / lw, -1, 1)))), 6))
        assert sorted(angs) == ([108.0] * 5 if kd == 0
                                else [36.0, 36.0, 144.0, 144.0]), angs
    c5 = np.array([0.5, _D5_RIN])
    cen = np.array([np.asarray(p, float).mean(axis=0) for p in polys])
    defects = 0
    for k in range(len(cen)):                # sample inside the safe disc
        gx, gy = cen[k] + np.array([0.0137, 0.0071])
        if hypot(gx - c5[0], gy - c5[1]) > 4.5:
            continue
        if sum(1 for p in polys if tg._pip(p, gx, gy)) != 1:
            defects += 1
    assert defects == 0, ("DURER5 radial", defects)
    th = 2 * np.pi / 5
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def key5(A):
        return sorted((round(float(x), 6), round(float(y), 6))
                      for x, y in A)
    assert key5(cen - c5) == key5((cen - c5) @ R.T), "not 5-fold"
    moved = any(key5(cen + (t - cen[0])) == key5(cen) for t in cen[1:24])
    assert not moved, "a translation fixes Durer5 -- that is periodic"
    print("Durer pentagons(1525) tiles exact (unit regular pentagon + "
          "36-deg rhomb); radial covers with no gap or overlap, 5-fold "
          "rotation yes, translation none; pmm and p2 variants cover")

    print("RESULT:", "OK" if not bad else "BAD %s" % bad)
    assert not bad
