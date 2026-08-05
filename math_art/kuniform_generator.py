
# 2-Uniform Euclidean Tiling Generator for Blender
#
# The 2-uniform tilings of the plane by regular polygons: edge-to-edge
# tilings whose vertices fall into exactly TWO transitivity classes (two
# distinct vertex figures), one step less regular than the 11 vertex-
# transitive Archimedean tilings.  Each is built from a translational
# unit cell -- lattice vectors (b1, b2) plus a small list of regular-
# polygon tiles of edge length 1 -- replicated across the plane by
# translating the tiles over i*b1 + j*b2.
#
# Many of the most attractive 2-uniform tilings are "layered": periodic
# stacks of two 1-uniform strips (e.g. a row of squares alternating with
# a double band of triangles, or rows of hexagons-and-triangles between
# bands of triangles).  Those layered families are what this module
# builds; every unit cell is derived by hand and coverage-verified.
#
# Part of the Pattern Engine (see pattern_common.py); mirrors the layout
# of tiling_generator.py and reuses its geometry helpers.
#
# References:
#   Branko Grunbaum & G. C. Shephard, "Tilings and Patterns" (W. H.
#     Freeman, 1987) -- classification of k-uniform tilings by regular
#     polygons.
#   Otto Krotenheerdt, enumeration of the k-uniform (in particular the
#     20 two-uniform) tilings, 1969.

bl_info = {
    "name": "2-Uniform Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "2-uniform Euclidean tilings (regular polygons with "
                   "two vertex orbits) as colored tile patterns",
    "category": "Add Mesh",
}

from math import sqrt, hypot, atan2
import numpy as np

try:
    from . import pattern_common as pc
    from . import tiling_generator as tg
except Exception:
    import pattern_common as pc
    import tiling_generator as tg


_H = sqrt(3.0) / 2.0                       # height of a unit triangle
_S3 = sqrt(3.0)


# --------------------------------------------------------------------
# Small tile builders (edge length 1)
# --------------------------------------------------------------------

def _tri(a, b, c):
    return np.array([a, b, c], float)


def _sq(x0, y0):
    """Axis-aligned unit square with lower-left corner (x0, y0)."""
    return np.array([[x0, y0], [x0 + 1.0, y0],
                     [x0 + 1.0, y0 + 1.0], [x0, y0 + 1.0]], float)


def _band(y0, x0):
    """One up/down triangle pair of a triangle strip whose bottom edge
    sits on y=y0 with a vertex at x=x0.  Bottom vertices land at
    x0 + integer, top vertices at x0 + 0.5 + integer; the pair tiles the
    strip under translation by (1, 0).  Returns [up, down]."""
    up = _tri((x0, y0), (x0 + 1.0, y0), (x0 + 0.5, y0 + _H))
    dn = _tri((x0 + 1.0, y0), (x0 + 1.5, y0 + _H), (x0 + 0.5, y0 + _H))
    return [up, dn]


# --------------------------------------------------------------------
# The translational unit cells
# --------------------------------------------------------------------
#
# _cell(name) -> (b1, b2, tiles).  Each `tiles` entry is one regular
# polygon (CCW, edge 1) of the fundamental domain; the plane is the set
# of translates tile + i*b1 + j*b2.  Naming is by the two vertex figures.

def _cell(name):
    if name == 'S_TT':
        # Square row + a DOUBLE band of triangles.  The line through the
        # middle of the double band is all-triangle (3^6); the square /
        # triangle interfaces are 3.3.3.4.4.
        b1 = np.array([1.0, 0.0])
        b2 = np.array([0.0, 1.0 + 2.0 * _H])
        tiles = [_sq(0.0, 0.0)]
        tiles += _band(1.0, 0.0)             # lower triangle band
        tiles += _band(1.0 + _H, 0.5)        # upper triangle band
        return b1, b2, tiles

    if name == 'S_TTT':
        # Square row + a TRIPLE band of triangles (a distinct tiling of
        # the same 3^6 / 3.3.3.4.4 vertex pair).
        b1 = np.array([1.0, 0.0])
        b2 = np.array([0.5, 1.0 + 3.0 * _H])
        tiles = [_sq(0.0, 0.0)]
        tiles += _band(1.0, 0.0)
        tiles += _band(1.0 + _H, 0.5)
        tiles += _band(1.0 + 2.0 * _H, 0.0)
        return b1, b2, tiles

    if name == 'SS_T':
        # DOUBLE square row + a single band of triangles.  The line
        # between the two square rows is all-square (4.4.4.4); the
        # square / triangle interface is 3.3.3.4.4.
        b1 = np.array([1.0, 0.0])
        b2 = np.array([0.5, 2.0 + _H])
        tiles = [_sq(0.0, 0.0), _sq(0.0, 1.0)]
        tiles += _band(2.0, 0.0)
        return b1, b2, tiles

    if name == 'SSS_T':
        # TRIPLE square row + a single band of triangles (a distinct
        # tiling of the same 4.4.4.4 / 3.3.3.4.4 vertex pair).
        b1 = np.array([1.0, 0.0])
        b2 = np.array([0.5, 3.0 + _H])
        tiles = [_sq(0.0, 0.0), _sq(0.0, 1.0), _sq(0.0, 2.0)]
        tiles += _band(3.0, 0.0)
        return b1, b2, tiles

    if name == 'HEX_T':
        # A trihexagonal strip (a row of hexagons with the triangles
        # that fill the gaps between them) alternating with a band of
        # triangles.  Hexagon-row interior vertices are 3.6.3.6; every
        # strip / band interface vertex is 3.3.3.3.6.
        b1 = np.array([2.0, 0.0])
        b2 = np.array([0.5, 3.0 * _H])
        hexa = tg._reg(6, 0.0, 0.0, 0.0)               # flat-top, edge 1
        tiles = [hexa]
        # gap triangles of the hexagon row (shared corner at (1, 0))
        tiles.append(_tri((0.5, _H), (1.5, _H), (1.0, 0.0)))
        tiles.append(_tri((0.5, -_H), (1.5, -_H), (1.0, 0.0)))
        # the triangle band riding on the strip's flat top (y = _H)
        tiles.append(_tri((0.5, _H), (1.5, _H), (1.0, 2.0 * _H)))
        tiles.append(_tri((1.5, _H), (2.5, _H), (2.0, 2.0 * _H)))
        tiles.append(_tri((0.0, 2.0 * _H), (1.0, 2.0 * _H), (0.5, _H)))
        tiles.append(_tri((1.0, 2.0 * _H), (2.0, 2.0 * _H), (1.5, _H)))
        return b1, b2, tiles

    if name == 'HEXCOL':
        # Hexagons in ALIGNED columns (vertically adjacent hexagons share
        # an edge), the side gaps filled by rhombi of two triangles.
        # Vertically-shared corners are 3.3.6.6; the shared-corner
        # columns give 3.6.3.6.
        b1 = np.array([2.0, 0.0])
        b2 = np.array([0.0, 2.0 * _H])
        hexa = tg._reg(6, 0.0, 0.0, 0.0)               # flat-top, edge 1
        tiles = [hexa]
        # rhombus gap to the right, split into two triangles
        tiles.append(_tri((1.0, 0.0), (1.5, _H), (0.5, _H)))
        tiles.append(_tri((1.5, _H), (1.0, 2.0 * _H), (0.5, _H)))
        return b1, b2, tiles

    raise ValueError("unknown tiling %r" % name)


def build_patch(name, nx, ny):
    """Return (polys, types) for an nx x ny run of a tiling's unit cell.
    `types` is the tile's index within the cell (for 'By Tile Type')."""
    b1, b2, tiles = _cell(name)
    polys, types = [], []
    for i in range(nx):
        for j in range(ny):
            off = i * b1 + j * b2
            for ti, tile in enumerate(tiles):
                polys.append(np.asarray(tile, float) + off)
                types.append(ti)
    return polys, types


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

TILING_ITEMS = [
    ('S_TT', "3.3.3.3.3.3 / 3.3.3.4.4 (a)",
     "Square row + double triangle band"),
    ('S_TTT', "3.3.3.3.3.3 / 3.3.3.4.4 (b)",
     "Square row + triple triangle band"),
    ('SS_T', "4.4.4.4 / 3.3.3.4.4 (a)",
     "Double square row + triangle band"),
    ('SSS_T', "4.4.4.4 / 3.3.3.4.4 (b)",
     "Triple square row + triangle band"),
    ('HEX_T', "3.6.3.6 / 3.3.3.3.6",
     "Hexagon-triangle strip + triangle band"),
    ('HEXCOL', "3.6.3.6 / 3.3.6.6",
     "Aligned hexagon columns + triangle rhombi"),
]

_LABEL = dict((k, v) for k, v, _ in TILING_ITEMS)


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_kuniform_add(bpy.types.Operator, AddObjectHelper):
        """Add a 2-uniform Euclidean tiling"""
        bl_idname = "mesh.kuniform_add"
        bl_label = "k-Uniform Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        tiling: EnumProperty(name="Tiling", items=TILING_ITEMS,
                             default='S_TT')
        nx: IntProperty(name="Cells X", default=5, min=1, max=40)
        ny: IntProperty(name="Cells Y", default=5, min=1, max=40)
        color_by: EnumProperty(
            name="Color By",
            items=[('SIDES', "By Sides",
                    "Material per polygon side count"),
                   ('TYPE', "By Tile Type",
                    "Material per tile orbit in the unit cell"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='SIDES')
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat 2D mesh; > 0 extrudes each tile")
        trim: BoolProperty(
            name="Trim Boundary", default=False,
            description="Clip the patch to a clean rectangle")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object")

        def execute(self, context):
            cells = tg.cells_from_polys(
                lambda a, b: build_patch(self.tiling, a, b),
                self.nx, self.ny, self.color_by,
                self.margin, self.height, self.trim)
            label = _LABEL[self.tiling]
            obj = pc.emit(context, "k-Uniform %s" % label, cells,
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
            for p in ('tiling', 'nx', 'ny', 'color_by', 'margin',
                      'height'):
                lay.prop(self, p)
            lay.prop(self, 'trim')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.kuniform_add", icon='MESH_GRID')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_kuniform_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_kuniform_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _regular(polys, tol=1e-3):
    """Every edge of every tile has length ~1.0."""
    for p in polys:
        p = np.asarray(p, float)
        m = len(p)
        for k in range(m):
            if abs(hypot(*(p[(k + 1) % m] - p[k])) - 1.0) > tol:
                return False
    return True


def _coverage_defects(polys):
    """Sample a 35x35 grid over centroid +/- 2.0, each point nudged by a
    small irrational offset, and count points not covered by exactly one
    tile (0 = gap, >=2 = overlap)."""
    allv = np.vstack([np.asarray(p, float) for p in polys])
    cx, cy = allv.mean(axis=0)
    defects = 0
    for gi in range(35):
        for gj in range(35):
            x = cx - 2.0 + 4.0 * gi / 34.0 + 0.0137
            y = cy - 2.0 + 4.0 * gj / 34.0 + 0.0071
            cnt = 0
            for p in polys:
                if tg._pip(p, x, y):
                    cnt += 1
                    if cnt > 1:
                        break
            if cnt != 1:
                defects += 1
    return defects


def _vertex_types(polys):
    """Number of distinct interior vertex figures (cyclic side-count
    sequence, canonicalised).  A genuine 2-uniform tiling has 2."""
    from collections import defaultdict
    coords, index = [], {}

    def vid(x, y):
        key = (round(float(x), 4), round(float(y), 4))
        if key not in index:
            index[key] = len(coords)
            coords.append((float(x), float(y)))
        return index[key]

    rings = [[vid(x, y) for x, y in p] for p in polys]
    ecount = defaultdict(int)
    vedges = defaultdict(set)
    vinc = defaultdict(list)                # vertex -> [(poly, corner)]
    for pi_, r in enumerate(rings):
        m = len(r)
        for k in range(m):
            a, b = r[k], r[(k + 1) % m]
            e = (a, b) if a < b else (b, a)
            ecount[e] += 1
            vedges[a].add(e)
            vedges[b].add(e)
        for k, v in enumerate(r):
            vinc[v].append((pi_, k))

    def canon(seq):
        best = None
        for s in (seq, seq[::-1]):
            for r in range(len(s)):
                rot = tuple(s[r:] + s[:r])
                if best is None or rot < best:
                    best = rot
        return best

    figs = set()
    for v, inc in vinc.items():
        if not vedges[v] or any(ecount[e] != 2 for e in vedges[v]):
            continue                        # boundary vertex
        vx, vy = coords[v]
        ordered = sorted(
            inc, key=lambda pk: atan2(
                np.mean([coords[i][1] for i in rings[pk[0]]]) - vy,
                np.mean([coords[i][0] for i in rings[pk[0]]]) - vx))
        figs.add(canon([len(rings[pk[0]]) for pk in ordered]))
    return len(figs)


_ORDER = [k for k, _, _ in TILING_ITEMS]


def _selftest():
    bad = []
    for name in _ORDER:
        polys, _ = build_patch(name, 6, 6)
        defects = _coverage_defects(polys)
        reg = _regular(polys)
        vt = _vertex_types(polys)
        ok = (defects == 0) and reg
        print("%-8s tiles=%4d coverage-defect=%3d types=%d reg=%s  %s"
              % (name, len(polys), defects, vt, reg,
                 "OK" if ok else "BAD"))
        if not ok:
            bad.append(name)
    print("RESULT:", "OK" if not bad else "BAD %s" % bad)
    assert not bad
