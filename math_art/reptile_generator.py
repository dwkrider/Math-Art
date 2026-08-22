
# Rep-Tile Substitution Tiling Generator for Blender
#
# A rep-tile is a shape that dissects into N smaller congruent copies of
# itself (a rep-N tile).  Applying that dissection over and over -- the
# "deflation" of a substitution tiling -- fills the original prototile
# with an ever finer, self-similar patch of sub-tiles.  Each rep-tile
# here is stored as its prototile polygon plus a fixed list of N child
# affine maps, every one a similarity of ratio 1/sqrt(N) whose images
# exactly tile the prototile.  `iterations` deflation steps therefore
# yield N**iterations sub-tiles covering the prototile with no gaps and
# no overlaps; the finished patch is fed through the shared Pattern
# Engine (see the `patterns` package) for color, grout margin and relief:
#
#   L_TROMINO       The L-tromino (three unit squares in an L).  rep-4:
#       the L splits into four half-scale L-trominoes.
#   SPHINX          The sphinx hexiamond (six equilateral triangles).
#       rep-4: it splits into four half-scale sphinxes, one of them a
#       mirror image -- the classic non-lattice substitution tiling.
#   L_TETROMINO     The L-tetromino (four unit squares).  rep-4.
#   P_PENTOMINO     The P-pentomino (a 2x2 block plus one square).
#       rep-4.
#   RIGHT_TRIANGLE  The right isosceles triangle.  rep-2: the altitude
#       from the right angle to the hypotenuse cuts it into two
#       half-size copies.
#
# `types` is each sub-tile's orientation class (rotation + reflection of
# its accumulated map), so the "By Orientation" color mode reveals the
# substitution structure directly.  Every dissection is coverage-checked
# by the pure-Python self-test at the bottom of this file.
#
# References:
# - Solomon W. Golomb, "Replicating figures in the plane" (Mathematical
#   Gazette, 1964) -- coined the term "rep-tile".
# - Martin Gardner, "Mathematical Games" (Scientific American, 1963) --
#   popularized rep-tiles, including the sphinx.

bl_info = {
    "name": "Rep-Tile Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Rep-tile substitution tilings (L-tromino, sphinx, "
                   "L-tetromino, P-pentomino, right triangle)",
    "category": "Add Mesh",
}

from math import sqrt, atan2, degrees

import numpy as np

try:                                  # inside the math_art package
    from .patterns.polygon2d import ensure_ccw as _ensure_ccw
except ImportError:                   # flat import (test runner)
    from patterns.polygon2d import ensure_ccw as _ensure_ccw

try:
    from .patterns import common as pc
    from . import tiling_generator as tg
except Exception:                       # legacy single-file / CLI use
    from patterns import common as pc
    import tiling_generator as tg


_H = sqrt(3.0) / 2.0                     # height of a unit triangle


# --------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------

def _M(a, b, c, d):
    return np.array([[a, b], [c, d]], float)






# --------------------------------------------------------------------
# Rep-tile definitions
#
# Each entry is (prototile vertices, child maps, N) where every child
# map is (A, b) acting as v -> A v + b and the N images A_k(prototile)
# exactly tile the prototile.  The maps were derived by solving each
# tile's exact self-dissection and are coverage-verified in the
# self-test below.
# --------------------------------------------------------------------

_KINDS = {
    # L-tromino: L of three unit squares, notch at the top right.
    'L_TROMINO': (
        [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)],
        [(_M(0.0, 0.5, 0.5, 0.0), np.array([0.0, 0.0])),
         (_M(0.0, 0.5, 0.5, 0.0), np.array([0.5, 0.5])),
         (_M(-0.5, 0.0, 0.0, 0.5), np.array([2.0, 0.0])),
         (_M(0.5, 0.0, 0.0, -0.5), np.array([0.0, 2.0]))],
        4),

    # Sphinx hexiamond: pentagon with edges 1, 1, 2, 3, 1 (unit = one
    # triangle edge).  Four half-scale sphinxes, one reflected.
    'SPHINX': (
        [(0.5, -_H), (1.0, 0.0), (2.0, 0.0), (1.0, 2.0 * _H), (-0.5, -_H)],
        [(_M(-0.25, _H / 2, _H / 2, 0.25), np.array([0.75, -_H / 2])),
         (_M(-0.25, _H / 2, -_H / 2, -0.25), np.array([1.5, _H])),
         (_M(0.25, -_H / 2, -_H / 2, -0.25), np.array([0.0, 0.0])),
         (_M(0.25, -_H / 2, -_H / 2, -0.25),
          np.array([0.75, 1.5 * _H]))],
        4),

    # L-tetromino: L of four unit squares (a 1x3 bar plus one square).
    'L_TETROMINO': (
        [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (0, 2)],
        [(_M(0.5, 0.0, 0.0, 0.5), np.array([1.0, 0.0])),
         (_M(0.0, -0.5, 0.5, 0.0), np.array([1.0, 0.0])),
         (_M(-0.5, 0.0, 0.0, -0.5), np.array([3.0, 1.0])),
         (_M(0.0, 0.5, -0.5, 0.0), np.array([0.0, 2.0]))],
        4),

    # P-pentomino: a 2x2 block with one extra square on top left.
    'P_PENTOMINO': (
        [(0, 0), (2, 0), (2, 2), (1, 2), (1, 3), (0, 3)],
        [(_M(0.5, 0.0, 0.0, 0.5), np.array([0.0, 0.0])),
         (_M(0.0, 0.5, -0.5, 0.0), np.array([0.5, 2.0])),
         (_M(0.5, 0.0, 0.0, -0.5), np.array([0.0, 3.0])),
         (_M(-0.5, 0.0, 0.0, 0.5), np.array([2.0, 0.0]))],
        4),

    # Right isosceles triangle, rep-2 (split by the right-angle altitude).
    'RIGHT_TRIANGLE': (
        [(0, 0), (1, 0), (0, 1)],
        [(_M(-0.5, 0.5, -0.5, -0.5), np.array([0.5, 0.5])),
         (_M(-0.5, -0.5, 0.5, -0.5), np.array([0.5, 0.5]))],
        2),
}


def _orient_key(M):
    """Orientation class of a 2x2 linear part: (reflected?, angle bucket).
    Scale-invariant, so it is stable through the deflation.  Sub-tiles
    sharing a key share a color under the 'By Orientation' mode."""
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    ang = degrees(atan2(M[1, 0], M[0, 0]))
    bucket = int(round(ang / 15.0)) % 24
    return (0 if det > 0.0 else 1, bucket)


# --------------------------------------------------------------------
# Public patch builder
# --------------------------------------------------------------------

def reptile_patch(kind, iterations):
    """Return (polys, types) for a deflated rep-tile patch.

    Starting from one prototile, apply the rep-tile's dissection
    `iterations` times (each tile -> N scaled/rotated/reflected copies).

    polys -- list of (N, 2) CCW float arrays (one per sub-tile)
    types -- list of int orientation-class indices, parallel to polys
    """
    if kind not in _KINDS:
        raise ValueError("unknown rep-tile kind %r" % kind)
    verts, maps, _n = _KINDS[kind]
    proto = np.asarray(verts, float)

    tiles = [(np.eye(2), np.zeros(2))]         # (linear, translation)
    for _ in range(int(iterations)):
        nxt = []
        for M0, t0 in tiles:
            for A, b in maps:
                nxt.append((M0 @ A, M0 @ b + t0))
        tiles = nxt

    polys, keys = [], []
    for M0, t0 in tiles:
        poly = (M0 @ proto.T).T + t0
        polys.append(_ensure_ccw(poly))
        keys.append(_orient_key(M0))

    order = {k: i for i, k in enumerate(sorted(set(keys)))}
    types = [order[k] for k in keys]
    return polys, types


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

KIND_ITEMS = [
    ('L_TROMINO', "L-Tromino (rep-4)",
     "The L-tromino dissects into four half-scale L-trominoes"),
    ('SPHINX', "Sphinx Hexiamond (rep-4)",
     "The sphinx dissects into four half-scale sphinxes (one mirrored)"),
    ('L_TETROMINO', "L-Tetromino (rep-4)",
     "The L-tetromino dissects into four half-scale copies"),
    ('P_PENTOMINO', "P-Pentomino (rep-4)",
     "The P-pentomino dissects into four half-scale copies"),
    ('RIGHT_TRIANGLE', "Right Triangle (rep-2)",
     "The right isosceles triangle dissects into two half-size copies"),
]


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_reptile_add(bpy.types.Operator, AddObjectHelper):
        """Add a rep-tile substitution tiling"""
        bl_idname = "mesh.reptile_add"
        bl_label = "Rep-Tile Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(name="Rep-Tile", items=KIND_ITEMS,
                           default='L_TROMINO',
                           description="Which rep-tile to substitute")
        iterations: IntProperty(
            name="Iterations", default=3, min=0, max=7,
            description="Deflation depth (tile count is N**iterations, "
                        "N = 4 for the rep-4 tiles, 2 for the triangle)")
        color_by: EnumProperty(
            name="Color By",
            items=[('TYPE', "By Orientation",
                    "Material per sub-tile orientation, revealing the "
                    "substitution structure"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='TYPE',
            description="How the tiles are colored")
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat 2D mesh; > 0 extrudes each tile")
        trim: BoolProperty(
            name="Trim Boundary", default=False,
            description="Clip the patch to a clean central rectangle")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object "
                        "(parented to an empty) so tiles can be edited "
                        "individually")

        def execute(self, context):
            polys, types = reptile_patch(self.kind, self.iterations)
            cells = tg.cells_from_polys(
                lambda a, b: (polys, types), 1, 1, self.color_by,
                self.margin, self.height, self.trim)
            label = dict((k, v) for k, v, _ in KIND_ITEMS)[self.kind]
            obj = pc.emit(context, "Rep-Tile %s" % label, cells,
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
            for p in ('kind', 'iterations', 'color_by', 'margin',
                      'height'):
                lay.prop(self, p)
            lay.prop(self, 'trim')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.reptile_add", icon='MESH_PLANE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_reptile_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_reptile_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
#
# A rep-tile's deflation exactly tiles the ORIGINAL prototile region, so
# a grid of points strictly inside the prototile must each be covered by
# exactly one sub-tile (0 gaps, 0 overlaps), and the sub-tile count must
# equal N**iterations.
# --------------------------------------------------------------------

def _coverage(polys, proto, samples=48, off=(0.0137, 0.0071)):
    """Sample a grid over the prototile bounding box; for every point
    strictly inside the prototile, count covering sub-tiles.  Return
    (n_inside, n_gap, n_over)."""
    proto = np.asarray(proto, float)
    x0, y0 = proto[:, 0].min(), proto[:, 1].min()
    x1, y1 = proto[:, 0].max(), proto[:, 1].max()
    n_in = n_gap = n_over = 0
    for ix in range(samples):
        x = x0 + (x1 - x0) * ix / (samples - 1) + off[0]
        for iy in range(samples):
            y = y0 + (y1 - y0) * iy / (samples - 1) + off[1]
            if not tg._pip(proto, x, y):
                continue
            n_in += 1
            cnt = 0
            for p in polys:
                if tg._pip(p, x, y):
                    cnt += 1
                    if cnt >= 2:
                        break
            if cnt == 0:
                n_gap += 1
            if cnt >= 2:
                n_over += 1
    return n_in, n_gap, n_over


def _selftest():
    all_ok = True
    for kind, label, _ in KIND_ITEMS:
        verts, _maps, N = _KINDS[kind]
        polys, types = reptile_patch(kind, 3)
        n_in, n_gap, n_over = _coverage(polys, verts)
        count_ok = (len(polys) == N ** 3)
        ok = count_ok and n_in > 0 and n_gap == 0 and n_over == 0
        all_ok = all_ok and ok
        print("%-15s tiles=%4d (N^3=%4d) inside=%4d gaps=%d over=%d  %s"
              % (kind, len(polys), N ** 3, n_in, n_gap, n_over,
                 "OK" if ok else "BAD"))
    print("RESULT:", "OK" if all_ok else "BAD")
    assert all_ok
