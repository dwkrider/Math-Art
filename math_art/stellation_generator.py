
# Icosahedron Stellations for Blender
#
# The final (complete) stellation of the icosahedron -- the echidnahedron,
# Wenninger W42, the last of "The Fifty-Nine Icosahedra".  It is the solid
# obtained by extending all twenty face planes of the icosahedron to their
# outermost intersections.  As a simple (triangulated) polyhedron it has 92
# vertices, 270 edges and 180 triangular faces; the vertices lie on three
# concentric icosahedrally-aligned shells -- an inner dodecahedron, a middle
# icosahedron and an outer truncated icosahedron of spike tips.  Because the
# solid is star-shaped about its centre, its faces are the radial (spherical)
# convex hull of those 92 vertices.  This reproduces the exact combinatorics
# (V=92, E=270, F=180); the vertex radii are the published closed forms, so
# the faceting is a faithful model of the echidnahedron.  Output is centred
# and fit to a 2 m cube.
#
# References:
# - H. S. M. Coxeter, P. Du Val, H. T. Flather, J. F. Petrie, "The
#   Fifty-Nine Icosahedra" (1938; 3rd ed. Tarquin 1999).
# - J. C. P. Miller's rules for significant stellations (ibid.).
# - Magnus Wenninger, "Polyhedron Models" (1971), model W42.

bl_info = {
    "name": "Icosahedron Stellations",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "The final stellation of the icosahedron (echidnahedron)",
    "category": "Add Mesh",
}

import math

try:
    from . import conway_operators as cw
except ImportError:
    import conway_operators as cw

PHI = (1 + 5 ** 0.5) / 2


def _unit(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


def echidnahedron_vertices():
    """The 92 vertices of the final stellation, on three icosahedrally
    aligned shells (dodecahedron, icosahedron, truncated icosahedron)."""
    ico, _ = cw._seed('I', 0)
    dod, _ = cw._seed('D', 0)
    tiV, tiF = cw.apply_conway('tI')
    tiV = cw.canonicalize(tiV, tiF, iters=800)
    r_in = math.sqrt(1.5 * (3 + 5 ** 0.5))
    r_mid = math.sqrt(0.5 * (25 + 11 * 5 ** 0.5))
    r_out = math.sqrt(0.5 * (97 + 43 * 5 ** 0.5))
    P = ([tuple(r_in * c for c in _unit(v)) for v in dod]
         + [tuple(r_mid * c for c in _unit(v)) for v in ico]
         + [tuple(r_out * c for c in _unit(v)) for v in tiV])
    return P


try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _hull_faces(P):
        import bmesh
        bm = bmesh.new()
        for p in P:
            bm.verts.new(_unit(p))          # spherical hull = radial faces
        bm.verts.ensure_lookup_table()
        res = bmesh.ops.convex_hull(bm, input=bm.verts)
        junk = res.get('geom_unused', []) + res.get('geom_interior', [])
        if junk:
            bmesh.ops.delete(bm, geom=junk, context='VERTS')
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        idx = {v: i for i, v in enumerate(bm.verts)}
        keep = [tuple(round(c, 5) for c in v.co) for v in bm.verts]
        faces = [[idx[v] for v in f.verts] for f in bm.faces]
        bm.free()
        # map hull vertex order back to the actual-radius points
        pk = {tuple(round(c, 5) for c in _unit(p)): p for p in P}
        V = [pk[k] for k in keep]
        return V, faces

    class MESH_OT_final_stellation_add(bpy.types.Operator):
        """Add the final (complete) stellation of the icosahedron -- the
        echidnahedron (Wenninger W42)"""
        bl_idname = "mesh.final_stellation_add"
        bl_label = "Final Stellation of the Icosahedron"
        bl_options = {'REGISTER', 'UNDO'}

        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            P = echidnahedron_vertices()
            V, F = _hull_faces(P)
            mx = max(abs(c) for v in V for c in v) or 1.0
            s = self.scale / mx
            me = bpy.data.meshes.new("Echidnahedron")
            me.from_pydata([tuple(c * s for c in v) for v in V], [],
                           [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new("Echidnahedron", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"Final stellation: V={len(V)} F={len(F)}")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.final_stellation_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_final_stellation_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_final_stellation_add)
