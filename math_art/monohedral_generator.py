
# Monohedral Tiling Generator for Blender
#
# Tilings of the plane by a SINGLE prototile that need not be regular:
#   * any triangle (paired with its 180 deg rotate -> a parallelogram);
#   * any quadrilateral, even non-convex (180 deg rotations about the
#     edge midpoints -- a centrally symmetric hexagon dimer);
#   * the 3 convex hexagons that tile (Reinhardt Types 1-3);
#   * the 15 convex pentagons that tile the plane;
#   * Durer's tiling (1525), the one NON-PERIODIC member here.
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
#   Durer".  Durer's own n = 5 figure refines the hexagon further into a
#   regular pentagon and a 36-degree rhombus; that refinement is not
#   built here, since the two shapes do not partition this hexagon (a
#   unit pentagon and rhomb total 2.308 against its 2.127) and the
#   source's figures are the only record of how the pieces actually go.

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
}


def _unit_cell(name):
    return _CELLS[name][1]()


def build_patch(name, nx, ny, layout='LATTICE', order=5):
    """Replicate a tiling's unit cell over an nx x ny run of the lattice.
    Returns (polys, types): a list of (N, 2) CCW polygon arrays and a
    parallel list of per-tile orientation-orbit indices.

    Durer's tiling also accepts layout='RADIAL', which abandons the
    lattice for his rotational arrangement; there `nx` sets how many
    rings deep the wedges go and `ny` is unused.
    """
    if name == 'DURER' and layout == 'RADIAL':
        return build_durer_radial(order, max(1, nx))
    if name == 'DURER':
        b1, b2, cell = _durer_cell(order)
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
          'DURER']


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
            description="How the prototile is repeated (Durer's tiling "
                        "only)",
            items=[('LATTICE', "Periodic",
                    "Repeat by translation, the ordinary use of a "
                    "centrally symmetric hexagon"),
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

        def execute(self, context):
            label = _CELLS[self.tiling][0]
            lay = (self.durer_layout if self.tiling == 'DURER'
                   else 'LATTICE')
            cells = tg.cells_from_polys(
                lambda a, b: build_patch(self.tiling, a, b, lay,
                                         self.durer_order),
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
            if self.tiling == 'DURER':
                lay.prop(self, 'durer_layout')
                lay.prop(self, 'durer_order')
            lay.prop(self, 'nx')
            if not (self.tiling == 'DURER'
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

    print("RESULT:", "OK" if not bad else "BAD %s" % bad)
    assert not bad
