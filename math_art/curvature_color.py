
# Curvature Color for Blender
#
# Colors the active mesh by discrete Gaussian curvature, after the
# curvature illustrations in Henry Segerman's "Visualizing
# Mathematics with 3D Printing" (fig 4-4): red where the surface is
# positively curved (sphere-like), white where it is flat, blue where
# it is negatively curved (saddle-like).
#
# The curvature is the classical angle deficit: for each vertex of an
# (internally triangulated) copy of the mesh,
#
#     interior vertex:  kappa = (2*pi - sum of incident angles) / A
#     boundary vertex:  kappa = (  pi - sum of incident angles) / A
#
# where A is the mixed area, approximated barycentrically as one third
# of the area of the incident triangles.  The scalar field can be
# Laplacian-smoothed, is clamped at a percentile of |kappa| (or a
# manual value) and mapped through a diverging blue-white-red ramp
# into a "Curvature" vertex color attribute.  A "Curvature Color"
# material showing that attribute is created (or reused) and assigned.
#
# The operator works on the object's base mesh data: apply modifiers
# first if you want the curvature of the modified surface.  The mesh
# itself is never altered; triangulation happens on a bmesh copy used
# only for the computation.
#
# References:
#   Gaussian curvature and its intrinsic character: C. F. Gauss,
#       "Theorema Egregium" (Disquisitiones generales circa superficies
#       curvas, 1827). The angle-deficit estimate is the discrete
#       Gauss-Bonnet / Descartes total-angular-defect form.

bl_info = {
    "name": "Curvature Color",
    "author": "Math Art project (after Henry Segerman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Object > Curvature Color",
    "description": "Color the active mesh by discrete Gaussian "
                   "curvature (red positive, white flat, blue "
                   "negative)",
    "category": "Object",
}

try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    import bmesh
    import numpy as np
    from math import pi

    ATTR_NAME = "Curvature"
    MAT_NAME = "Curvature Color"

    def vertex_curvature(mesh):
        """Per-vertex discrete Gaussian curvature of a mesh.

        Angle deficit over barycentric mixed area, boundary aware.
        The mesh is not modified: a bmesh copy is triangulated for
        the computation.  Returns a float64 array of len(vertices).
        """
        nv = len(mesh.vertices)
        coords = np.empty(3 * nv)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(-1, 3)

        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        tris = np.array([[l.vert.index for l in f.loops]
                         for f in bm.faces], dtype=np.int64)
        boundary = np.zeros(nv, dtype=bool)
        for e in bm.edges:
            if e.is_boundary:
                boundary[e.verts[0].index] = True
                boundary[e.verts[1].index] = True
        bm.free()

        angle_sum = np.zeros(nv)
        marea = np.zeros(nv)
        if len(tris):
            p = coords[tris]                       # (m, 3, 3)
            area = 0.5 * np.linalg.norm(
                np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                axis=1)
            for k in range(3):
                u = p[:, (k + 1) % 3] - p[:, k]
                v = p[:, (k + 2) % 3] - p[:, k]
                ang = np.arctan2(
                    np.linalg.norm(np.cross(u, v), axis=1),
                    np.einsum('ij,ij->i', u, v))
                np.add.at(angle_sum, tris[:, k], ang)
                np.add.at(marea, tris[:, k], area / 3.0)

        target = np.where(boundary, pi, 2.0 * pi)
        kappa = np.zeros(nv)
        ok = marea > 1e-12                # skip loose/degenerate verts
        kappa[ok] = (target[ok] - angle_sum[ok]) / marea[ok]
        return kappa

    def smooth_field(mesh, values, passes):
        """Uniform-weight Laplacian smoothing over mesh edges."""
        ne = len(mesh.edges)
        if ne == 0 or passes <= 0:
            return values
        ev = np.empty(2 * ne, dtype=np.int64)
        mesh.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        nv = len(values)
        cnt = np.zeros(nv)
        np.add.at(cnt, ev.ravel(), 1.0)
        for _ in range(passes):
            acc = np.zeros(nv)
            np.add.at(acc, ev[:, 0], values[ev[:, 1]])
            np.add.at(acc, ev[:, 1], values[ev[:, 0]])
            mean = np.where(cnt > 0, acc / np.maximum(cnt, 1.0),
                            values)
            values = 0.5 * (values + mean)
        return values

    def ensure_material():
        """Get or (re)build the shared Curvature Color material."""
        mat = bpy.data.materials.get(MAT_NAME)
        if mat is None:
            mat = bpy.data.materials.new(MAT_NAME)
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = attr = out = None
        for node in nt.nodes:
            if node.type == 'BSDF_PRINCIPLED' and bsdf is None:
                bsdf = node
            elif node.type == 'ATTRIBUTE' and attr is None:
                attr = node
            elif node.type == 'OUTPUT_MATERIAL' and out is None:
                out = node
        if out is None:
            out = nt.nodes.new('ShaderNodeOutputMaterial')
        if bsdf is None:
            bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
            bsdf.location = (out.location.x - 280, out.location.y)
        # note: == not `is` -- bpy wraps RNA data in fresh Python
        # objects on every access
        if not any(lk.to_node == out and lk.from_node == bsdf
                   for lk in nt.links):
            nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        if attr is None:
            attr = nt.nodes.new('ShaderNodeAttribute')
            attr.location = (bsdf.location.x - 280, bsdf.location.y)
        attr.attribute_type = 'GEOMETRY'
        attr.attribute_name = ATTR_NAME
        if not any(lk.from_node == attr and lk.to_node == bsdf
                   for lk in nt.links):
            nt.links.new(attr.outputs['Color'],
                         bsdf.inputs['Base Color'])
        return mat

    class OBJECT_OT_curvature_color_add(bpy.types.Operator):
        """Color the active mesh by discrete Gaussian curvature:
        red positive, white flat, blue negative (angle deficit,
        written to a "Curvature" vertex color attribute; apply
        modifiers first to color the modified surface)"""
        bl_idname = "object.curvature_color_add"
        bl_label = "Curvature Color"
        bl_options = {'REGISTER', 'UNDO'}

        smoothing: bpy.props.IntProperty(
            name="Smoothing", default=2, min=0, max=50,
            description="Laplacian smoothing passes applied to the "
                        "curvature field (uniform weights)")
        normalize: bpy.props.EnumProperty(
            name="Normalize",
            items=(('AUTO', "Percentile",
                    "Clamp at a percentile of |curvature|"),
                   ('MANUAL', "Manual",
                    "Clamp at a fixed curvature value")),
            default='AUTO',
            description="How to pick the curvature value mapped to "
                        "full red/blue")
        percentile: bpy.props.FloatProperty(
            name="Percentile", default=90.0, min=50.0, max=100.0,
            subtype='PERCENTAGE',
            description="Percentile of |curvature| clamped to full "
                        "color (auto normalization)")
        clamp: bpy.props.FloatProperty(
            name="Clamp", default=1.0, min=1e-6,
            description="Curvature value mapped to full color "
                        "(manual normalization)")
        invert: bpy.props.BoolProperty(
            name="Invert", default=False,
            description="Swap the roles of the two ramp ends")
        color_pos: bpy.props.FloatVectorProperty(
            name="Positive", subtype='COLOR', size=3,
            min=0.0, max=1.0,
            default=(0.784, 0.188, 0.188),        # ~#C83030 red
            description="Ramp end for positive curvature")
        color_neg: bpy.props.FloatVectorProperty(
            name="Negative", subtype='COLOR', size=3,
            min=0.0, max=1.0,
            default=(0.188, 0.376, 0.784),        # ~#3060C8 blue
            description="Ramp end for negative curvature")

        @classmethod
        def poll(cls, context):
            return (context.active_object is not None
                    and context.active_object.type == 'MESH')

        def execute(self, context):
            obj = context.active_object
            mesh = obj.data
            nv = len(mesh.vertices)
            if nv == 0:
                self.report({'WARNING'}, "Mesh has no vertices")
                return {'CANCELLED'}

            kappa = vertex_curvature(mesh)
            kappa = smooth_field(mesh, kappa, self.smoothing)

            if self.normalize == 'AUTO':
                clamp = float(np.percentile(np.abs(kappa),
                                            self.percentile))
            else:
                clamp = self.clamp
            # floor so flat meshes stay white instead of amplifying
            # float noise into full saturation
            clamp = max(clamp, 1e-3)

            t = np.clip(kappa / clamp, -1.0, 1.0)
            if self.invert:
                t = -t
            tp = np.clip(t, 0.0, None)[:, None]
            tn = np.clip(-t, 0.0, None)[:, None]
            pos = np.asarray(self.color_pos, dtype=np.float64)
            neg = np.asarray(self.color_neg, dtype=np.float64)
            cols = np.ones((nv, 4))
            cols[:, :3] = 1.0 + tp * (pos - 1.0) + tn * (neg - 1.0)

            # vertex color attribute (reuse; recreate if the wrong
            # kind of attribute occupies the name)
            attr = mesh.color_attributes.get(ATTR_NAME)
            if attr is not None and (attr.domain != 'POINT'
                                     or attr.data_type
                                     != 'FLOAT_COLOR'):
                mesh.color_attributes.remove(attr)
                attr = None
            if attr is None:
                attr = mesh.color_attributes.new(
                    ATTR_NAME, 'FLOAT_COLOR', 'POINT')
            attr.data.foreach_set("color", cols.ravel())
            mesh.color_attributes.active_color = attr
            mesh.update()

            # shared material, appended once and made active
            mat = ensure_material()
            slot_index = -1
            for i, slot in enumerate(obj.material_slots):
                if slot.material == mat:
                    slot_index = i
                    break
            if slot_index < 0:
                mesh.materials.append(mat)
                slot_index = len(obj.material_slots) - 1
            obj.active_material_index = slot_index

            self.report(
                {'INFO'},
                "Curvature in [%.4g, %.4g], clamped at %.4g"
                % (kappa.min(), kappa.max(), clamp))
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("object.curvature_color_add",
                             icon='COLOR')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(OBJECT_OT_curvature_color_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_object.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_object.remove(_menu_func)
        bpy.utils.unregister_class(OBJECT_OT_curvature_color_add)
