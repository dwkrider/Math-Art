
# Rotegrity Generator for Blender
#
# Rotegrity ("nexorade") sphere models, after Antiprism's `rotegrity`:
# each edge of a spherical polyhedron becomes a curved strap on the
# sphere, rotated about its midpoint and lengthened so that strap ends
# overlap their neighbours in a woven, tensegrity-like arrangement.
# Twist and extension are free sliders (tune until the ends meet).
#
# References:
#   - Adrian Rossiter, Antiprism and its `rotegrity` program
#     (https://www.antiprism.com) -- the reference implementation.
#   - The term "rotegrity" was introduced by Richard (Dick) Boyt,
#     "Rotegrity" (1970); modern strap-sphere models by Dick
#     Fischbeck.
#   - "Tensegrity" is Buckminster Fuller's term for the floating-
#     compression structures of Kenneth Snelson.
#   - "Nexorade" / reciprocal-frame terminology after Olivier
#     Baverel et al., "Nexorades" (mutually supporting members).

bl_info = {
    "name": "Rotegrity Generator",
    "author": "Math Art project (after Antiprism's rotegrity)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Rotegrity",
    "description": "Rotegrity / nexorade strap-sphere models",
    "category": "Add Mesh",
}



try:                                  # inside the math_art package
    from .polyhedra.seeds import icosa_faces as _icosa_faces
    from .polyhedra.seeds import seed_poly as _shared_seed
except ImportError:                   # flat import (test runner)
    from polyhedra.seeds import icosa_faces as _icosa_faces
    from polyhedra.seeds import seed_poly as _shared_seed

try:                                  # inside the math_art package
    from .polyhedra.fit import fit_cube as _fit_cube
except ImportError:                   # flat import (test runner)
    from polyhedra.fit import fit_cube as _fit_cube
# The mathematics lives in the sibling `weaving` engine package;
# this module is the Blender layer over it.
try:
    from .weaving.rotegrity import (build_rotegrity)
except ImportError:  # flat import outside the package
    from weaving.rotegrity import (build_rotegrity)






# ---- spherical seed polyhedra -------------------------------------------









# ---- strap construction --------------------------------------------------







# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_rotegrity_add(bpy.types.Operator):
        """Add a rotegrity (nexorade) strap-sphere"""
        bl_idname = "mesh.rotegrity_add"
        bl_label = "Rotegrity"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Seed",
            items=[('ICOSA', "Icosahedron", ""), ('OCTA', "Octahedron", ""),
                   ('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
                   ('DODECA', "Dodecahedron", "")],
            default='ICOSA',
            description="Spherical polyhedron whose edges become straps")
        freq: IntProperty(
            name="Geodesic Frequency", default=2, min=1, max=8,
            description="Subdivision of triangular seeds")
        twist: FloatProperty(
            name="Twist", default=18.0, min=-60.0, max=60.0,
            description="Rotation of each strap about its midpoint (deg)")
        extension: FloatProperty(
            name="Extension", default=0.35, min=0.0, max=1.0,
            description="Lengthening of each strap beyond its edge")
        width: FloatProperty(name="Strap Width", default=0.06,
                             min=0.005, max=0.4,
                             description="Width of each strap")
        thickness: FloatProperty(name="Strap Thickness", default=0.025,
                                 min=0.002, max=0.2,
                                 description="Thickness of each strap")
        segments: IntProperty(name="Segments", default=12, min=4, max=64,
                              description="Lengthwise subdivisions along "
                                          "each strap")
        coloring: EnumProperty(
            name="Coloring",
            description="How straps are assigned materials",
            items=[('LENGTH', "By Strap Length",
                    "One material per strap length class -- geodesic "
                    "breakdowns give a few distinct lengths, colored "
                    "as in physical rotegrity kits (view with "
                    "Material Preview or Solid shading set to "
                    "Material color)"),
                   ('STRAP', "Per Strap",
                    "Cycle the palette strap by strap"),
                   ('NONE', "None", "No materials")],
            default='LENGTH')
        scale: FloatProperty(name="Radius", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall radius of the sphere")

        _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                    (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                    (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                    (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                    (0.45, 0.45, 0.85), (0.80, 0.50, 0.30)]

        @classmethod
        def _material_for(cls, i, kind="Strap"):
            name = f"Rotegrity {kind} {i + 1}"
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
            verts, faces, ne, face_strap, strap_len = build_rotegrity(
                self.kind, self.freq, self.twist, self.extension,
                self.width, self.thickness, self.segments, self.scale)
            # the strap thickness pushes the hull a little past the
            # nominal radius, so fit the bounding box rather than trust it
            verts = _fit_cube(verts, 2.0 * self.scale)
            me = bpy.data.meshes.new("Rotegrity")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            if len(me.polygons) == len(face_strap):
                classes = sorted(set(round(l, 5) for l in strap_len))
                strap_cls = [classes.index(round(l, 5))
                             for l in strap_len]
                attr = me.attributes.new("strap_index", 'INT', 'FACE')
                attr.data.foreach_set('value', face_strap)
                attr = me.attributes.new("length_class", 'INT',
                                         'FACE')
                attr.data.foreach_set(
                    'value', [strap_cls[s] for s in face_strap])
                if self.coloring == 'LENGTH':
                    for i in range(len(classes)):
                        me.materials.append(
                            self._material_for(i, "Length"))
                    me.polygons.foreach_set(
                        'material_index',
                        [strap_cls[s] for s in face_strap])
                elif self.coloring == 'STRAP':
                    nmat = min(ne, len(self._PALETTE))
                    for i in range(nmat):
                        me.materials.append(self._material_for(i))
                    me.polygons.foreach_set(
                        'material_index',
                        [s % nmat for s in face_strap])
            me.update()
            obj = bpy.data.objects.new("Rotegrity", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{ne} straps")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'freq', 'twist', 'extension', 'width',
                      'thickness', 'segments', 'coloring', 'scale'):
                if k == 'freq' and self.kind in ('CUBE', 'DODECA'):
                    continue
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.rotegrity_add", icon='SPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_rotegrity_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_rotegrity_add)


def _selftest():
    for kind, freq in (('ICOSA', 1), ('ICOSA', 2), ('CUBE', 1),
                       ('DODECA', 1)):
        v, f, ne, fs, sl = build_rotegrity(kind, freq)
        print(f"{kind} f{freq}: straps={ne} verts={len(v)} "
              f"faces={len(f)}")
