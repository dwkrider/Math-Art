
# Regular Polylinks Generator for Blender
#
# Orderly tangles of regular polygons, after George W. Hart's regular
# polylinks: one polygonal frame per face (plane) of a Platonic solid,
# each rotated about its face normal, scaled, and offset so the frames
# interlock. Classic examples: 4 triangles (tetrahedron), 6 squares
# (cube), 12 pentagons or 6 pentagons (dodecahedron), 8 triangles
# (octahedron), 20 triangles (icosahedron).
#
# References:
#   - Alan Holden, "Orderly Tangles: Cloverleafs, Gordian Knots, and
#     Regular Polylinks" (Columbia University Press, 1983) -- the
#     symmetric interlocked-polygon arrangements.
#   - George W. Hart, "Orderly Tangles Revisited",
#     https://www.georgehart.com/orderly-tangles-revisited/tangles.htm
#     (the regular polylinks this follows).

bl_info = {
    "name": "Regular Polylinks",
    "author": "Math Art project (after George W. Hart)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Regular Polylinks",
    "description": "Interlocked polygon frames from Platonic solids",
    "category": "Add Mesh",
}



try:                                  # inside the math_art package
    from .curve_frames import closed_tube as _shared_closed_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import closed_tube as _shared_closed_tube


try:                                  # inside the math_art package
    from .polyhedra.seeds import icosa_faces as _icosa_faces
    from .polyhedra.seeds import seed_poly
except ImportError:                   # flat import (test runner)
    from polyhedra.seeds import icosa_faces as _icosa_faces
    from polyhedra.seeds import seed_poly
# The mathematics lives in the sibling `weaving` engine package;
# this module is the Blender layer over it.
try:
    from .weaving.links import (build_polylinks)
except ImportError:  # flat import outside the package
    from weaving.links import (build_polylinks)














try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    PRESETS = {
        'T4': ("4 Triangles", dict(kind='TETRA', size=1.55, rotation=30,
                                   offset=-0.45, antipodal=False)),
        'C6': ("6 Squares", dict(kind='CUBE', size=1.45, rotation=25,
                                 offset=-0.4, antipodal=False)),
        'O8': ("8 Triangles", dict(kind='OCTA', size=1.6, rotation=25,
                                   offset=-0.25, antipodal=False)),
        'D6': ("6 Pentagons", dict(kind='DODECA', size=1.5, rotation=20,
                                   offset=-0.9, antipodal=True)),
        'D12': ("12 Pentagons", dict(kind='DODECA', size=1.25, rotation=15,
                                     offset=-0.5, antipodal=False)),
        'I20': ("20 Triangles", dict(kind='ICOSA', size=1.55, rotation=22,
                                     offset=-0.55, antipodal=False)),
    }

    class MESH_OT_polylinks_add(bpy.types.Operator):
        """Interlocked regular polygon frames (after Hart's polylinks)"""
        bl_idname = "mesh.polylinks_add"
        bl_label = "Regular Polylinks"
        bl_options = {'REGISTER', 'UNDO'}

        def _preset_chosen(self, context):
            if self.preset != 'CUSTOM':
                for k, v in PRESETS[self.preset][1].items():
                    setattr(self, k, v)

        preset: EnumProperty(
            name="Preset",
            description="Preset polylink arrangement (Custom to set the "
                        "parameters by hand)",
            items=[('CUSTOM', "Custom", "")] +
                  [(k, v[0], "") for k, v in PRESETS.items()],
            default='T4', update=_preset_chosen)
        kind: EnumProperty(
            name="Solid",
            description="Platonic solid whose face planes host the "
                        "polygon frames",
            items=[('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", ""),
                   ('DODECA', "Dodecahedron", ""),
                   ('ICOSA', "Icosahedron", "")],
            default='TETRA')
        size: FloatProperty(name="Frame Size", default=1.55,
                            min=0.5, max=4.0,
                            description="Size of each polygon frame")
        rotation: FloatProperty(
            name="Rotation", default=30.0, min=-90.0, max=90.0,
            description="Turn of each frame about its face normal (deg)")
        offset: FloatProperty(
            name="Plane Offset", default=-0.45, min=-2.0, max=2.0,
            description="Push of each frame along its normal "
                        "(negative = toward the centre)")
        link_shape: EnumProperty(
            name="Link Shape",
            description="Shape of each link: flat polygon frame, wavy "
                        "circle, or torus knot",
            items=[('POLYGON', "Polygon Frame",
                    "Flat polygon frames (Hart's polylinks)"),
                   ('WAVE', "Wavy Circle",
                    "Radius-modulated circles: round rings that "
                    "weave through each other (after Shengyi "
                    "Wang's polylink add-on)"),
                   ('KNOT', "Torus Knot",
                    "A (p, q x sides) torus knot about each face "
                    "axis, swept with rotation-minimizing frames")],
            default='POLYGON')
        amplitude: FloatProperty(
            name="Wave Amplitude", default=0.35, min=0.0, max=2.0,
            description="Radial wave amplitude (wavy circle) or "
                        "knot minor radius")
        wave_factor: bpy.props.IntProperty(
            name="Wave Factor", default=1, min=1, max=8,
            description="Wave frequency in multiples of the face "
                        "side count")
        knot_p: bpy.props.IntProperty(
            name="Knot p", default=2, min=1, max=8,
            description="Windings around the face axis")
        knot_q_factor: bpy.props.IntProperty(
            name="Knot q Factor", default=1, min=1, max=8,
            description="q = factor x face side count")
        tube_sides: bpy.props.IntProperty(name="Tube Sides",
                                          default=8, min=3, max=24,
                                          description="Sides of the swept "
                                                      "tube cross-section "
                                                      "(wavy / knot links)")
        segments: bpy.props.IntProperty(
            name="Link Segments", default=128, min=24, max=512,
            description="Samples along wavy / knot centerlines")
        width: FloatProperty(name="Frame Width", default=0.14,
                             min=0.02, max=0.9,
                             description="Width of each polygon frame bar")
        thickness: FloatProperty(name="Frame Thickness", default=0.10,
                                 min=0.01, max=1.0,
                                 description="Thickness of each frame bar "
                                             "or swept tube")
        antipodal: BoolProperty(
            name="Antipodal Half", default=False,
            description="Use only one face of each antipodal pair")
        coloring: EnumProperty(
            name="Coloring",
            description="How the frames are coloured",
            items=[('FRAME', "Per Link", "One material per frame, for "
                    "visibility (view with Material Preview or Solid "
                    "shading set to Material color)"),
                   ('PAIR', "Per Parallel Pair",
                    "Iso-color parallel (antipodal) frames, as in "
                    "Hart's paper models: 6 squares in 3 colors"),
                   ('NONE', "None", "No materials")],
            default='FRAME')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size (1.0 fits a 2 m "
                                         "cube)")

        _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                    (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                    (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                    (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                    (0.80, 0.50, 0.30), (0.45, 0.45, 0.85)]

        @classmethod
        def _material_for(cls, i):
            name = f"Polylink {i + 1}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                if i < len(cls._PALETTE):
                    rgb = cls._PALETTE[i]
                else:
                    import colorsys
                    rgb = colorsys.hsv_to_rgb((i * 0.618034) % 1.0,
                                              0.6, 0.8)
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.5
            return mat

        def execute(self, context):
            if self.preset != 'CUSTOM':
                self._preset_chosen(context)
            verts, faces, nf, face_frame, frame_dirs = build_polylinks(
                self.kind, self.size, self.rotation, self.offset,
                self.width, self.thickness, self.antipodal, self.scale,
                self.link_shape, self.amplitude, self.wave_factor,
                self.knot_p, self.knot_q_factor, self.tube_sides,
                self.segments)
            me = bpy.data.meshes.new("Polylinks")
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
            if len(me.polygons) == len(faces):
                # frame membership as face metadata (Geometry Nodes /
                # shader Attribute node: "link_index")
                attr = me.attributes.new("link_index", 'INT', 'FACE')
                attr.data.foreach_set('value', face_frame)
                if self.coloring != 'NONE':
                    if self.coloring == 'PAIR':
                        group = {}
                        gid = []
                        for d in frame_dirs:
                            dd = d if tuple(d) >= tuple(-x for x in d) \
                                else tuple(-x for x in d)
                            key = tuple(round(x, 5) for x in dd)
                            if key not in group:
                                group[key] = len(group)
                            gid.append(group[key])
                    else:
                        gid = list(range(nf))
                    nmats = max(gid) + 1
                    for i in range(nmats):
                        me.materials.append(self._material_for(i))
                    me.polygons.foreach_set(
                        'material_index',
                        [gid[face_frame[i]] for i in range(len(faces))])
            me.update()
            obj = bpy.data.objects.new("Polylinks", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{nf} frames")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'link_shape')
            keys = ['kind', 'size', 'rotation', 'offset']
            if self.link_shape == 'POLYGON':
                keys += ['width', 'thickness']
            else:
                keys += ['thickness', 'amplitude']
                if self.link_shape == 'WAVE':
                    keys += ['wave_factor']
                else:
                    keys += ['knot_p', 'knot_q_factor']
                keys += ['tube_sides', 'segments']
            keys += ['antipodal', 'coloring', 'scale']
            for k in keys:
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.polylinks_add", icon='MESH_CIRCLE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_polylinks_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polylinks_add)
