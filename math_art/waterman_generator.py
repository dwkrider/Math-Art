
# Waterman Polyhedra Generator for Blender
#
# Waterman polyhedra (Steve Waterman): the convex hull of the points of
# the FCC lattice (cubic close packing) that lie within a sphere of
# radius sqrt(2 * root) about the origin. After Antiprism's `waterman`.

bl_info = {
    "name": "Waterman Polyhedra",
    "author": "David Krider (Math Art project, after Antiprism's waterman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Waterman Polyhedron",
    "description": "Waterman polyhedra (convex hulls of FCC sphere points)",
    "category": "Add Mesh",
}

import math


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
    from bpy.props import IntProperty, FloatProperty
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
        scale: FloatProperty(name="Scale", default=1.0, min=0.001,
                             max=100.0)

        def execute(self, context):
            pts = fcc_points(self.root)
            if len(pts) < 4:
                self.report({'ERROR'}, "not enough lattice points")
                return {'CANCELLED'}
            s = self.scale / math.sqrt(2 * self.root)
            bm = bmesh.new()
            vlist = [bm.verts.new((p[0] * s, p[1] * s, p[2] * s))
                     for p in pts]
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
            obj = bpy.data.objects.new(f"Waterman W{self.root}", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"W{self.root}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'root')
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


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for root in (1, 2, 5, 10, 100):
            print(f"W{root}: {len(fcc_points(root))} lattice points")
