
# Stellated Surface Weave for Blender
#
# A port of Shengyi Wang's "Stellated Surface Weave"
# (txyyss.github.io/math-art/stellated-surface-weave, source at
# github.com/txyyss/math-art): twelve folded strips interlocking
# along the pentagram planes of a small stellated dodecahedron.
#
# Construction: each pentagram face carries five skeleton arms
# (tip -> bend at (5-sqrt5)/10 along the edge -> face centre). At
# every tip, the five incident arms share a pentagonal-pyramid apex
# whose inner point sits 2*width inward; strips get two side faces
# lying in the adjacent star planes plus a parallelogram closure.
# Toward the face centres the strips stay parallel and meet in
# mitred pentagon junctions. The two extension factors
# sqrt((5+sqrt5)/10) and sqrt((5-sqrt5)/10) keep the bends mitred at
# constant width. The strip width is the one free parameter.
#
# References:
# - Shengyi Wang (txyyss), "Stellated Surface Weave"
#   (txyyss.github.io/math-art/stellated-surface-weave; source at
#   github.com/txyyss/math-art) -- the construction this ports.
# - Underlying solid: the small stellated dodecahedron, one of the
#   Kepler-Poinsot star polyhedra (Johannes Kepler, 1619).

bl_info = {
    "name": "Stellated Surface Weave",
    "author": "Math Art project (after Shengyi Wang / "
              "txyyss)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Stellated Surface Weave",
    "description": "Interlocking folded strips on the small "
                   "stellated dodecahedron",
    "category": "Add Mesh",
}
# The mathematics lives in the sibling `weaving` engine package;
# this module is the Blender layer over it.
try:
    from .weaving.stellated import (build_arms, build_weave,
                                        indexed_ssd_faces)
except ImportError:  # flat import outside the package
    from weaving.stellated import (build_arms, build_weave,
                                       indexed_ssd_faces)








































































try:
    import bpy
    import bmesh
    from bpy.props import (FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_stellated_weave_add(bpy.types.Operator):
        """Twelve interlocking folded strips along the pentagram
        planes of a small stellated dodecahedron (after Shengyi
        Wang's Stellated Surface Weave)"""
        bl_idname = "mesh.stellated_weave_add"
        bl_label = "Stellated Surface Weave"
        bl_options = {'REGISTER', 'UNDO'}

        width: FloatProperty(
            name="Strip Width", default=0.12, min=0.02, max=0.32,
            description="Width of the folded strips; all mitres "
                        "recompute")
        coloring: EnumProperty(
            name="Coloring",
            items=[('STRIP', "Per Strip",
                    "One material per pentagram plane (12 strips; "
                    "view with Material Preview or Solid shading "
                    "set to Material color)"),
                   ('NONE', "None", "No materials")],
            default='STRIP')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                    (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                    (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                    (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                    (0.45, 0.45, 0.85), (0.80, 0.50, 0.30),
                    (0.35, 0.60, 0.55), (0.75, 0.35, 0.45)]

        @classmethod
        def _material_for(cls, i):
            name = f"Weave Strand {i + 1}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE[i % len(cls._PALETTE)]
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb,
                                                               1.0)
                    bsdf.inputs["Roughness"].default_value = 0.45
            return mat

        def execute(self, context):
            verts, faces, tags = build_weave(self.width, self.scale)
            me = bpy.data.meshes.new("StellatedWeave")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            # fit (roughly) within a 2 x scale cube at the origin
            lo = [min(v.co[k] for v in me.vertices)
                  for k in range(3)]
            hi = [max(v.co[k] for v in me.vertices)
                  for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0
                       for k in range(3)) or 1.0
            f = self.scale / half
            for v in me.vertices:
                v.co = [(v.co[k] - (lo[k] + hi[k]) / 2.0) * f
                        for k in range(3)]
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            if len(me.polygons) == len(tags):
                attr = me.attributes.new("strip_index", 'INT',
                                         'FACE')
                attr.data.foreach_set('value', tags)
                if self.coloring == 'STRIP':
                    for i in range(12):
                        me.materials.append(self._material_for(i))
                    me.polygons.foreach_set('material_index', tags)
            me.update()
            obj = bpy.data.objects.new("StellatedWeave", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('width', 'coloring', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.stellated_weave_add",
                             icon='MOD_LATTICE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_stellated_weave_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_stellated_weave_add)


def _selftest():
    from collections import Counter
    faces = indexed_ssd_faces()
    # 12 pentagram faces from 12 tips, 5 arms each
    arms = build_arms(faces)
    tips = set(a.tipIndex for a in arms)
    print(f"faces={len(faces)}(12) tips={len(tips)}(12) "
          f"arms={len(arms)}(60) "
          f"{'OK' if (len(faces), len(tips), len(arms)) == (12, 12, 60) else 'BAD'}")
    for w in (0.06, 0.12, 0.2):
        v, f, t = build_weave(w)
        cnt = Counter()
        for fc in f:
            for i in range(len(fc)):
                a, b = fc[i], fc[(i + 1) % len(fc)]
                cnt[(min(a, b), max(a, b))] += 1
        nonman = sum(1 for c in cnt.values() if c != 2)
        print(f"width {w}: verts={len(v)} faces={len(f)} "
              f"strips={len(set(t))}(12) "
              f"non-2-manifold edges={nonman}")
