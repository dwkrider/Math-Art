
# Subdivision Tiling Generator for Blender
#
# A finite subdivision rule replaces every tile by a fixed pattern of smaller
# tiles and iterates.  The headline case is Bowers and Stephenson's PENTAGONAL
# rule: one pentagon into six, generating a tiling of the plane in which every
# pentagon at every level is CONFORMALLY REGULAR -- conformally equivalent to a
# regular pentagon, corners preserved -- even though no two are congruent.
#
# The catch, and the reason this generator depends on the circle-packing engine,
# is that the tiles are pentagons only COMBINATORIALLY.  There is no
# straight-line realisation in which they are all regular, so the geometry
# cannot come from the rule; it comes from a packing.  Add a vertex at the
# barycentre of each tile, triangulate, pack that, and draw each tile through
# the circle centres at its corners.  This is what Cannon, Floyd and Parry's
# `tilepack` program does, and under refinement the tiles converge to their true
# conformal shapes.
#
# Three layouts, and the contrast between them is the point:
#
#   CONFORMAL      the maximal packing of the hyperbolic disc -- the correct
#                  realisation, filling the unit disc
#   EUCLIDEAN      packed in the plane with a free boundary
#   COMBINATORIAL  a straight-line Tutte embedding: cheap, and it visibly
#                  degrades with depth
#
# Rules shipped: PENTAGONAL (1 -> 6 pentagons), BARYCENTRIC (1 -> 6 triangles),
# QUAD (1 -> 4 quadrilaterals) and a two-type TRIANGLE_QUAD rule that turns a
# triangle into three quadrilaterals and then quadrilaterals into four.
#
# The engines are in `math_art/subdiv/` and `math_art/packing/`, both
# Blender-free.
#
# References:
# - Philip L. Bowers and Kenneth Stephenson, "A 'regular' pentagonal tiling of
#   the plane", Conformal Geometry and Dynamics 1 (1997), pp. 58-86.
# - J. W. Cannon, W. J. Floyd and W. R. Parry, "Finite subdivision rules",
#   Conformal Geometry and Dynamics 5 (2001), pp. 153-196.
# - Philip L. Bowers and Kenneth Stephenson, "Conformal tilings II: local
#   isomorphism, hierarchy and conformal type", Conformal Geometry and Dynamics
#   23 (2019), pp. 60-101.
# - Burt Rodin and Dennis Sullivan, "The convergence of circle packings to the
#   Riemann mapping", Journal of Differential Geometry 26 (1987), pp. 349-360.

bl_info = {
    "name": "Subdivision Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Subdivision Tiling",
    "description": "Conformal tilings from finite subdivision rules, laid out "
                   "by circle packing",
    "category": "Add Mesh",
}

import math

from . import subdiv
from .subdiv import (BARYCENTRIC, COMBINATORIAL, CONFORMAL, PENTAGONAL, PLANAR,
                     QUAD, TRIANGLE_QUAD, build, realise, tile_count,
                     tile_polygons)

_PALETTE = [
    (0.90, 0.91, 0.94, 1.0), (0.16, 0.22, 0.61, 1.0),
    (0.29, 0.38, 0.78, 1.0), (0.46, 0.56, 0.90, 1.0),
    (0.60, 0.42, 0.09, 1.0), (0.84, 0.64, 0.29, 1.0),
    (0.35, 0.40, 0.47, 1.0), (0.12, 0.14, 0.18, 1.0),
]
_NPAL = len(_PALETTE)


def _centroid(poly):
    n = len(poly)
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


def _area(poly):
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return abs(0.5 * a)


def build_tiling(rule=PENTAGONAL, depth=2, layout=CONFORMAL, output='FACES',
                 inset=0.12, relief=0.0, scale=1.0, color_by='DEPTH',
                 ribbon=0.05):
    """Build the tiling and return (verts, faces, mats, report)."""
    n_tiles = tile_count(rule, depth)
    if n_tiles > 60000:
        raise ValueError("depth %d of the %s rule is %d tiles; reduce the depth"
                         % (depth, rule, n_tiles))
    K = build(rule, depth)
    pos, info = realise(K, layout)
    polys = tile_polygons(K, pos)

    areas = [_area(p) for p in polys]
    amax = max(areas) if areas else 1.0
    amin = min(a for a in areas if a > 0) if any(a > 0 for a in areas) else 1.0

    verts, faces, mats = [], [], []
    for ti, poly in enumerate(polys):
        if color_by == 'UNIFORM':
            mi = 1
        elif color_by == 'SIDES':
            mi = 1 + (len(poly) % (_NPAL - 1))
        elif color_by == 'AREA':
            t = 0.0 if amax <= amin else \
                (math.log(max(areas[ti], 1e-18)) - math.log(amin)) \
                / (math.log(amax) - math.log(amin))
            mi = 1 + int(min(0.999, max(0.0, t)) * (_NPAL - 1))
        else:
            mi = 1 + (ti % (_NPAL - 1))

        cx, cy = _centroid(poly)
        z = 0.0
        if relief:
            r = math.hypot(cx, cy)
            z = relief * math.cos(1.5 * math.pi * min(1.0, r))

        if output == 'EDGES':
            base = len(verts)
            for (px, py) in poly:
                sx = cx + (px - cx) * (1.0 - ribbon)
                sy = cy + (py - cy) * (1.0 - ribbon)
                verts.append((px * scale, py * scale, z * scale))
                verts.append((sx * scale, sy * scale, z * scale))
            n = len(poly)
            for i in range(n):
                a0 = base + 2 * i
                a1 = base + 2 * i + 1
                b0 = base + 2 * ((i + 1) % n)
                b1 = base + 2 * ((i + 1) % n) + 1
                faces.append((a0, b0, b1, a1))
                mats.append(mi)
            continue

        shrunk = [(cx + (px - cx) * (1.0 - inset),
                   cy + (py - cy) * (1.0 - inset)) for (px, py) in poly]
        base = len(verts)
        for (px, py) in shrunk:
            verts.append((px * scale, py * scale, z * scale))
        faces.append(tuple(range(base, base + len(shrunk))))
        mats.append(mi)

    report = ("%s depth %d: %d tiles, %s layout" %
              (rule.lower(), depth, len(polys), info['layout'].lower()))
    if info.get('circles'):
        report += ", %d circles, angle error %.2e" % (info['circles'],
                                                      info['angle_error'])
    if not info.get('converged', True):
        report += " (STALLED)"
    return verts, faces, mats, report


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _tiling_mesh(name, verts, faces, mats, smooth):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in verts], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        for idx, rgba in enumerate(_PALETTE):
            nm = "SubdivisionTiling_%d" % idx
            m = bpy.data.materials.get(nm)
            if m is None:
                m = bpy.data.materials.new(nm)
                m.diffuse_color = rgba
            me.materials.append(m)
        if mats and len(mats) == len(me.polygons):
            me.polygons.foreach_set('material_index', mats)
        if smooth:
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        return me

    class MESH_OT_subdivision_tiling_add(bpy.types.Operator):
        """Add a conformal tiling generated by a finite subdivision rule and
        laid out by circle packing"""
        bl_idname = "mesh.subdivision_tiling_add"
        bl_label = "Subdivision Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        rule: EnumProperty(
            name="Rule",
            items=[(PENTAGONAL, "Pentagonal",
                    "One pentagon into six; the conformally regular pentagonal "
                    "tiling of Bowers and Stephenson"),
                   (BARYCENTRIC, "Barycentric",
                    "One triangle into six, through the edge midpoints and "
                    "the centroid"),
                   (QUAD, "Quadrilateral",
                    "One quadrilateral into four"),
                   (TRIANGLE_QUAD, "Triangle to Quads",
                    "A triangle into three quadrilaterals, then each into "
                    "four; a rule with two tile types")],
            default=PENTAGONAL)
        depth: IntProperty(
            name="Depth", default=2, min=0, max=6,
            description="Subdivision passes; the pentagonal rule multiplies "
                        "the tile count by six each time")
        layout: EnumProperty(
            name="Layout",
            items=[(CONFORMAL, "Conformal",
                    "Laid out by a maximal circle packing of the disc: every "
                    "tile takes its true conformal shape"),
                   (PLANAR, "Packed Plane",
                    "Laid out by a euclidean packing with a free boundary"),
                   (COMBINATORIAL, "Straight Line",
                    "A straight-line embedding; cheap, and it degrades "
                    "visibly as the depth grows")],
            default=CONFORMAL)
        output: EnumProperty(
            name="Output",
            items=[('FACES', "Tiles", "Each tile as a face"),
                   ('EDGES', "Ribbons",
                    "A ribbon around each tile's border, for screens and "
                    "relief panels")],
            default='FACES')
        inset: FloatProperty(
            name="Tile Gap", default=0.12, min=0.0, max=0.6,
            description="Shrink each tile toward its centre, opening a gap "
                        "along the joins")
        ribbon: FloatProperty(
            name="Ribbon Width", default=0.05, min=0.005, max=0.4,
            description="Width of the border ribbon, as a fraction of the tile")
        relief: FloatProperty(
            name="Relief", default=0.0, min=-1.0, max=1.0,
            description="Lift tiles out of the plane by distance from the "
                        "centre, turning the tiling into a panel")
        color_by: EnumProperty(
            name="Material By",
            items=[('DEPTH', "Tile", "A material per tile, cycling"),
                   ('AREA', "Area", "Shade by tile size"),
                   ('SIDES', "Sides", "Shade by how many sides a tile has"),
                   ('UNIFORM', "Uniform", "One material")],
            default='DEPTH')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Shade Smooth", default=False)

        def execute(self, context):
            try:
                verts, faces, mats, report = build_tiling(
                    self.rule, self.depth, self.layout, self.output,
                    self.inset, self.relief, self.scale, self.color_by,
                    self.ribbon)
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            me = _tiling_mesh("SubdivisionTiling", verts, faces, mats,
                              self.smooth)
            obj = bpy.data.objects.new("SubdivisionTiling", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, report)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'rule')
            lay.prop(self, 'depth')
            lay.prop(self, 'layout')
            lay.prop(self, 'output')
            if self.output == 'FACES':
                lay.prop(self, 'inset')
            else:
                lay.prop(self, 'ribbon')
            lay.prop(self, 'relief')
            lay.prop(self, 'color_by')
            lay.prop(self, 'scale')
            lay.prop(self, 'smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.subdivision_tiling_add", icon='MESH_GRID')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_subdivision_tiling_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_subdivision_tiling_add)


def _selftest():
    """Build every rule and layout headlessly."""
    v, f, m, rep = build_tiling(PENTAGONAL, 2, CONFORMAL)
    assert len(f) == 36, (len(f), rep)
    assert len(m) == len(f)
    assert 'STALLED' not in rep, rep
    assert all(len(fc) == 5 for fc in f), "pentagonal tiles must have 5 corners"

    for rule, want in ((BARYCENTRIC, 36), (QUAD, 16), (TRIANGLE_QUAD, 12)):
        depth = 2 if rule != QUAD else 2
        v2, f2, m2, rep2 = build_tiling(rule, depth, PLANAR)
        assert len(f2) == want, (rule, len(f2), want, rep2)
        assert 'STALLED' not in rep2, rep2

    for lay in (CONFORMAL, PLANAR, COMBINATORIAL):
        v3, f3, m3, rep3 = build_tiling(PENTAGONAL, 1, lay)
        assert len(f3) == 6, (lay, len(f3))

    v4, f4, m4, _ = build_tiling(PENTAGONAL, 1, CONFORMAL, output='EDGES')
    assert len(f4) == 6 * 5, len(f4)

    v5, f5, m5, _ = build_tiling(PENTAGONAL, 1, CONFORMAL, relief=0.3)
    assert any(abs(p[2]) > 1e-9 for p in v5), "relief should lift the tiles"

    try:
        build_tiling(PENTAGONAL, 7, CONFORMAL)
        raised = False
    except ValueError:
        raised = True
    assert raised, "an excessive depth must be refused"

    print("subdivision_tiling_generator: all four rules and three layouts "
          "build, ribbons and relief work, excessive depth refused. "
          "RESULT: OK")
