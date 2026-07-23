
# Organic Wireframe style for Blender: the classic "parametric
# Voronoi sphere" modifier stack, applicable to any mesh.
#
# The whole effect is a live, non-destructive modifier chain on
# the active object:
#
#   Triangulate  -> even triangles for the collapse to chew on
#   Decimate     -> COLLAPSE at a low ratio melts the triangulation
#                   into irregular, Voronoi-looking cells
#   Wireframe    -> replaces the faces with a lattice of struts
#                   along the cell edges (boundary kept, so open
#                   surfaces like minimal-surface patches keep
#                   their rim)
#   Subdivision  -> rounds the lattice into smooth organic branches
#
# Everything stays editable in the modifier stack afterwards; the
# operator only builds the stack and sets smooth shading.

bl_info = {
    "name": "Organic Wireframe (Voronoi style)",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Styles",
    "description": "Parametric Voronoi look via a live "
                   "Triangulate / Decimate / Wireframe / "
                   "Subdivision modifier stack",
    "category": "Object",
}

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class OBJECT_OT_organic_wireframe(bpy.types.Operator):
        """Give the active mesh the organic 'parametric Voronoi'
        look: a live Triangulate / Decimate / Wireframe /
        Subdivision modifier stack (fully adjustable afterwards)"""
        bl_idname = "object.organic_wireframe_add"
        bl_label = "Organic Wireframe (Voronoi)"
        bl_options = {'REGISTER', 'UNDO'}

        ratio: FloatProperty(
            name="Cell Coarseness", default=0.12, min=0.005,
            max=1.0,
            description="Decimate collapse ratio: lower keeps "
                        "fewer, larger cells")
        thickness: FloatProperty(
            name="Strut Thickness", default=0.04, min=0.001,
            max=1.0,
            description="Wireframe strut thickness")
        levels: IntProperty(
            name="Smoothing Levels", default=2, min=0, max=4,
            description="Subdivision levels rounding the struts")
        triangulate: BoolProperty(
            name="Triangulate First", default=True,
            description="Triangulate before decimating, so quad "
                        "grids melt into irregular cells rather "
                        "than stripes")
        even_offset: BoolProperty(
            name="Even Thickness", default=True,
            description="Maintain strut thickness at sharp "
                        "corners (Wireframe even offset)")

        @classmethod
        def poll(cls, context):
            ob = context.active_object
            return (ob is not None and ob.type == 'MESH'
                    and context.mode == 'OBJECT')

        def execute(self, context):
            obj = context.active_object
            if self.triangulate:
                mod = obj.modifiers.new("VoronoiTriangulate",
                                        'TRIANGULATE')
                try:
                    mod.min_vertices = 4
                except AttributeError:
                    pass
            mod = obj.modifiers.new("VoronoiDecimate", 'DECIMATE')
            mod.decimate_type = 'COLLAPSE'
            mod.ratio = self.ratio
            mod = obj.modifiers.new("VoronoiWireframe",
                                    'WIREFRAME')
            mod.thickness = self.thickness
            mod.use_even_offset = self.even_offset
            mod.use_boundary = True
            mod.use_replace = True
            mod = obj.modifiers.new("VoronoiSmooth", 'SUBSURF')
            mod.levels = self.levels
            mod.render_levels = max(self.levels, 2)
            for p in obj.data.polygons:
                p.use_smooth = True
            self.report(
                {'INFO'},
                "modifier stack added; tune it in the modifier "
                "properties or apply to make it permanent")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('ratio', 'thickness', 'levels',
                      'triangulate', 'even_offset'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("object.organic_wireframe_add",
                             icon='MOD_WIREFRAME')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(OBJECT_OT_organic_wireframe)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_object.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_object.remove(_menu_func)
        bpy.utils.unregister_class(OBJECT_OT_organic_wireframe)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        print("organic wireframe style: modifier stack only; run "
              "the extension test suite for coverage")
