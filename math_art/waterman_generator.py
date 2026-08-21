
# Waterman Polyhedra Generator for Blender
#
# Waterman polyhedra (Steve Waterman): the convex hull of the points of
# the FCC lattice (cubic close packing) that lie within a sphere of
# radius sqrt(2 * root) about the origin. After Antiprism's `waterman`.
#
# References:
# - Waterman polyhedra: Steve Waterman (convex hulls of close-packed
#   lattice points within a given radius).
# - Antiprism (Adrian Rossiter), the `waterman` program.

bl_info = {
    "name": "Waterman Polyhedra",
    "author": "Math Art project (after Antiprism's waterman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Waterman Polyhedron",
    "description": "Waterman polyhedra (convex hulls of FCC sphere points)",
    "category": "Add Mesh",
}

import math

try:                                  # inside the math_art package
    from .polyhedra.fit import fit_cube as _fit_cube
except ImportError:                   # flat import (test runner)
    from polyhedra.fit import fit_cube as _fit_cube


def fcc_points(root):
    """FCC lattice points (x+y+z even) with |p|^2 <= 2*root."""
    r2 = 2 * root
    m = int(math.isqrt(r2)) + 1
    pts = []
    for x in range(-m, m + 1):
        for y in range(-m, m + 1):
            zz = r2 - x * x - y * y
            if zz < 0:
                continue
            for z in range(-m, m + 1):
                if x * x + y * y + z * z <= r2 and (x + y + z) % 2 == 0:
                    pts.append((x, y, z))
    return pts


try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_waterman_add(bpy.types.Operator):
        """Add a Waterman polyhedron (hull of FCC points within
        radius sqrt(2*root))"""
        bl_idname = "mesh.waterman_add"
        bl_label = "Waterman Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        root: IntProperty(
            name="Root", default=10, min=1, max=1000,
            description="Radius^2 / 2 of the ball of FCC points")
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain closed polyhedron"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo Style "
                    "Geometry Nodes modifier (Border and Thickness "
                    "stay editable on the modifier)"),
                   ('WIRE', "Struts",
                    "Struts along the edges (Wireframe modifier)"),
                   ('BALLSTICK', "Ball and Stick",
                    "Edges as solid cylindrical struts and vertices "
                    "as small spheres (ball-and-stick model)"),
                   ('WIREFRAME', "Wireframe",
                    "Mesh edges only, displayed as a wireframe"),
                   ('FACETS', "Face Segments",
                    "Split into one inward-extruded, mitre-beveled "
                    "segment per face")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.06, min=0.005, max=1.0,
            description="Leonardo face frame width, the same on every "
                        "face whatever its size")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness for the Leonardo "
                        "and Wireframe styles")
        strut_radius: FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.5,
            description="Ball-and-stick edge cylinder radius")
        node_radius: FloatProperty(
            name="Node Radius", default=0.035, min=0.0, max=0.5,
            description="Ball-and-stick vertex sphere radius "
                        "(0 = no nodes)")
        facet_depth: FloatProperty(name="Depth", default=0.15, min=0.01,
                                   max=2.0,
                                   description="Face Segments inward depth")
        facet_gap: FloatProperty(name="Bevel Gap", default=0.0, min=0.0,
                                 max=0.5,
                                 description="Gap between face segments")
        facet_explode: FloatProperty(name="Explode", default=0.1, min=0.0,
                                     max=5.0,
                                     description="Move segments outward")
        facet_separate: BoolProperty(
            name="Separate Meshes", default=False,
            description="Each face segment as its own object")
        scale: FloatProperty(name="Scale", default=1.0, min=0.001,
                             max=100.0)

        def execute(self, context):
            pts = fcc_points(self.root)
            if len(pts) < 4:
                self.report({'ERROR'}, "not enough lattice points")
                return {'CANCELLED'}
            # The lattice radius sqrt(2*root) bounds the SPHERE the points
            # were chosen from, not the hull's bounding box, so scaling by
            # it left every Waterman short of the 2 m cube (1.79 across for
            # the default).  Fit the actual points instead.
            pts = _fit_cube(pts, 2.0 * self.scale)
            bm = bmesh.new()
            vlist = [bm.verts.new((p[0], p[1], p[2])) for p in pts]
            bmesh.ops.convex_hull(bm, input=vlist)
            unused = [v for v in bm.verts if not v.link_faces]
            bmesh.ops.delete(bm, geom=unused, context='VERTS')
            bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(0.1),
                                     verts=bm.verts[:], edges=bm.edges[:])
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            me = bpy.data.meshes.new(f"Waterman W{self.root}")
            bm.to_mesh(me)
            bm.free()
            me.update()
            if self.style == 'FACETS':
                Vf = [tuple(v.co) for v in me.vertices]
                Ff = [list(p.vertices) for p in me.polygons]
                bpy.data.meshes.remove(me)
                try:
                    from .styles import facet_style
                except ImportError:
                    from styles import facet_style
                facet_style.emit_facets(
                    context, Vf, Ff, f"Waterman W{self.root}",
                    self.facet_depth, self.facet_gap,
                    self.facet_explode, self.facet_separate)
                self.report({'INFO'}, f"{len(Ff)} face segments")
                return {'FINISHED'}
            obj = bpy.data.objects.new(f"Waterman W{self.root}", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
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
            elif self.style == 'BALLSTICK':
                try:
                    from .styles import ball_and_stick
                except ImportError:
                    from styles import ball_and_stick
                ball_and_stick.rebuild(obj, self.strut_radius,
                                       self.node_radius)
            elif self.style == 'WIREFRAME':
                obj.display_type = 'WIRE'
            self.report({'INFO'},
                        f"W{self.root}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'root')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style in ('LEONARDO', 'WIRE'):
                lay.prop(self, 'thickness')
            if self.style == 'BALLSTICK':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')
            if self.style == 'FACETS':
                lay.prop(self, 'facet_depth')
                lay.prop(self, 'facet_gap')
                lay.prop(self, 'facet_explode')
                lay.prop(self, 'facet_separate')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.waterman_add", icon='MESH_ICOSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_waterman_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_waterman_add)


def _selftest():
    for root in (1, 2, 5, 10, 100):
        print(f"W{root}: {len(fcc_points(root))} lattice points")
