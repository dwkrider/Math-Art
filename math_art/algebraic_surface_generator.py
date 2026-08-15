
# Algebraic Surface Generator for Blender
#
# Classical algebraic surfaces -- celebrated cubics, quartics,
# quintics and sextics -- built as implicit level sets f(x,y,z) = 0
# and meshed with the marching-tetrahedra extractor from the sibling
# Minimal Surface Toolkit module.
#
#   Clebsch diagonal cubic   (all 27 lines are real)
#   Cayley nodal cubic       (4 nodes -- the maximum for a cubic)
#   Kummer quartic           (16 nodes at the generic parameter)
#   Barth sextic             (65 nodes -- the maximum for a sextic)
#   Togliatti quintic        (31 nodes -- the maximum for a quintic)
#   Taubin heart, Ding-dong, Chmutov sextic, Tangle cube
#   Monkey saddle           (n-fold: z = Re((x+iy)^n), n = 3 classic)
#
# Geometry only; materials and rendering are left to Blender.
#
# References:
#   Clebsch diagonal cubic: A. Clebsch (1871). Cayley nodal cubic:
#       A. Cayley (1869). Kummer quartic: E. E. Kummer (1864).
#   Barth sextic (65 nodes): W. Barth (1996). Togliatti quintic
#       (31 nodes): E. G. Togliatti (1940). Chmutov surfaces:
#       S. V. Chmutov. Heart surface after G. Taubin (1994).
#   N-fold monkey saddles z = rho^n cos(n*phi) = Re((x+iy)^n) are the
#       graphs of the degree-n harmonic polynomials (real parts of the
#       holomorphic w^n); n = 2 is the ordinary saddle, n = 3 the
#       classic monkey saddle z = x^3 - 3xy^2. Ceramic renditions of
#       these saddle sheets recur in Robert Fathauer's mathematical
#       ceramics (his n-fold saddle forms).

bl_info = {
    "name": "Algebraic Surface Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Algebraic Surface",
    "description": "Classical algebraic surfaces (Clebsch, Cayley, "
                   "Kummer, Barth, Togliatti, ...) as implicit level "
                   "sets meshed by marching tetrahedra",
    "category": "Add Mesh",
}
# The mathematics lives in the sibling `surfaces` engine package;
# this module is the Blender layer over it.
try:
    from .surfaces.algebraic import (PRESETS, build_algebraic, np)
except ImportError:  # flat import outside the package
    from surfaces.algebraic import (PRESETS, build_algebraic, np)








# ==========================================================================
# Implicit fields
# ==========================================================================
# Each field takes numpy sample grids x, y, z plus the Kummer
# parameter mu (ignored by every preset except the Kummer quartic)
# and returns f on the grid; the surface is the zero level set.

























# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _new_object(context, name, verts, faces, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth',
                                [smooth] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    class MESH_OT_algebraic_surface_add(bpy.types.Operator):
        """Add a classical algebraic surface (implicit level set
        meshed by marching tetrahedra)"""
        bl_idname = "mesh.algebraic_surface_add"
        bl_label = "Algebraic Surface"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset",
            items=[(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='CLEBSCH')
        resolution: IntProperty(
            name="Resolution", default=80, min=16, max=256,
            description="Sample grid resolution per axis (algebraic "
                        "surfaces need more than TPMS)")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        mu: FloatProperty(
            name="Kummer Mu", default=1.3, min=1.05, max=2.0,
            description="Kummer quartic parameter (node sharpness); "
                        "used by the Kummer preset only")
        fold: IntProperty(
            name="Fold n", default=3, min=2, max=8,
            description="Saddle fold count: 2 = ordinary saddle, "
                        "3 = monkey saddle, higher = n-fold saddles; "
                        "Monkey Saddle preset only")
        clip: FloatProperty(
            name="Clip Override", default=0.0, min=0.0, max=20.0,
            description="Clip ball radius / box half-extent; "
                        "0 uses the preset default")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this "
                        "thickness")
        smooth: BoolProperty(
            name="Smooth Shading", default=True)

        def execute(self, context):
            label = PRESETS[self.preset][0]
            verts, tris = build_algebraic(
                self.preset, self.resolution, mu=self.mu,
                clip=self.clip, scale=self.scale, fold=self.fold)
            if len(tris) == 0:
                self.report({'ERROR'}, "Empty level set")
                return {'CANCELLED'}
            obj = _new_object(context, label, verts, tris,
                              smooth=self.smooth)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            me = obj.data
            self.report({'INFO'},
                        f"{label}: {len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'resolution')
            if self.preset == 'KUMMER':
                lay.prop(self, 'mu')
            if self.preset == 'MONKEY':
                lay.prop(self, 'fold')
            for k in ('clip', 'scale', 'thickness', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.algebraic_surface_add",
                             icon='SURFACE_NSPHERE')

    _classes = (MESH_OT_algebraic_surface_add,)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


def _selftest():
    # standalone smoke test of the numeric core (requires the
    # Minimal Surface Toolkit importable as a sibling)
    for kind in PRESETS:
        V, T = build_algebraic(kind, 40)
        print(f"{kind:10s}: {len(V):6d} verts {len(T):6d} tris")
