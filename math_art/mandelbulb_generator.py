
# Escape-time Volumetric Fractal Generator for Blender
#
# Solid 3D escape-time fractals -- the Mandelbulb, quaternion Julia
# sets and the Mandelbox -- meshed as watertight objects. Each point
# of a sample grid is iterated under the fractal map; points whose
# orbit stays bounded are INSIDE the set, points whose orbit escapes
# are OUTSIDE. A distance estimate (Hart-Sandin-Kauffman) gives the
# outside points a sub-voxel distance to the boundary, so the zero
# level set of the resulting signed field -- extracted by the sibling
# Minimal Surface Toolkit's marching_tets -- is a smooth, printable
# surface rather than a blocky voxel shell. An optional largest-
# connected-component pass drops the isolated specks these sets throw
# off, leaving a single closed solid centred in a 2 m cube.
#
#   MANDELBULB   z -> z^p + c in White-Nylander spherical form, the
#                classic "power 8" bulb (p is exposed)
#   JULIA        quaternion Julia set z -> z^2 + c for a fixed
#                quaternion c, drawn as its w = const 3D slice
#   MANDELBOX    z -> scale * boxFold(ballFold(z)) + c
#
# References:
# - Mandelbulb: Daniel White and Paul Nylander (2009), the spherical
#   power-p map popularised at fractalforums.com.
# - Quaternion Julia sets: Alan Norton, "Generation and display of
#   geometric fractals in 3-D", Computer Graphics (SIGGRAPH) 16(3),
#   1982, pp. 61-67.
# - Mandelbox: Tom Lowe (2010), fractalforums.com.
# - Distance estimation of escape-time fractals: John C. Hart, Daniel
#   J. Sandin and Louis H. Kauffman, "Ray tracing deterministic 3-D
#   fractals", Computer Graphics (SIGGRAPH) 23(3), 1989, pp. 289-296.

bl_info = {
    "name": "Escape-time Fractals",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Escape-time Fractal",
    "description": "Mandelbulb, quaternion Julia and Mandelbox solids",
    "category": "Add Mesh",
}

import math

import numpy as np
# The mathematics lives in the sibling `ifs` engine package;
# this module is the Blender layer over it.
try:
    from .ifs.escape import (PRESETS, build_escape_fractal)
except ImportError:  # flat import outside the package
    from ifs.escape import (PRESETS, build_escape_fractal)



# ==========================================================================
# Signed distance-estimator fields.  Each returns a callable
# field(X, Y, Z) -> array that is negative inside the set and a
# positive distance estimate outside, ready for marching_tets.
# `cell` is the grid spacing, used as the interior sentinel so the
# zero crossing lands about a cell-fraction outside the last solid
# sample.
# ==========================================================================


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

    class MESH_OT_mandelbulb_add(bpy.types.Operator):
        """Add an escape-time volumetric fractal (Mandelbulb,
        quaternion Julia or Mandelbox) as a solid mesh"""
        bl_idname = "mesh.mandelbulb_add"
        bl_label = "Escape-time Fractal"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Fractal",
            items=[('MANDELBULB', "Mandelbulb",
                    "White-Nylander power-p bulb"),
                   ('JULIA', "Quaternion Julia",
                    "Quaternion Julia set, w = const slice"),
                   ('MANDELBOX', "Mandelbox",
                    "Box/ball-folding Mandelbox (scale 2)")],
            default='MANDELBULB',
            description="Which escape-time fractal to build")
        resolution: IntProperty(
            name="Resolution", default=96, min=24, max=256,
            description="Sample grid cells per axis (cost grows as "
                        "the cube; 96-160 is a good range)")
        power: FloatProperty(
            name="Bulb Power", default=8.0, min=2.0, max=16.0,
            description="Exponent p of the Mandelbulb map (8 is the "
                        "classic bulb); ignored by other presets")
        iterations: IntProperty(
            name="Iterations", default=0, min=0, max=32,
            description="Orbit iterations (0 = per-preset default); "
                        "more sharpens the boundary but adds cost")
        julia_w: FloatProperty(
            name="Julia Slice (w)", default=0.0, min=-1.5, max=1.5,
            description="Which w = const 3D slice of the 4D "
                        "quaternion Julia set to mesh")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Overall size of the result")
        largest_only: BoolProperty(
            name="Largest Part Only", default=True,
            description="Keep only the largest connected component, "
                        "dropping isolated specks")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier of this "
                        "thickness (for a shell / printing)")
        smooth: BoolProperty(name="Smooth Shading", default=True,
                             description="Shade the surface smooth")

        def execute(self, context):
            label = PRESETS[self.preset][0]
            verts, tris = build_escape_fractal(
                self.preset, self.resolution, power=self.power,
                iters=self.iterations, julia_w=self.julia_w,
                scale=self.scale, largest_only=self.largest_only)
            if len(tris) == 0:
                self.report({'ERROR'}, "Empty set at these settings")
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
            if self.preset == 'MANDELBULB':
                lay.prop(self, 'power')
            if self.preset == 'JULIA':
                lay.prop(self, 'julia_w')
            lay.prop(self, 'iterations')
            lay.prop(self, 'scale')
            lay.prop(self, 'largest_only')
            lay.prop(self, 'thickness')
            lay.prop(self, 'smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.mandelbulb_add",
                             icon='META_BALL')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_mandelbulb_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_mandelbulb_add)


def _selftest():
    # standalone self-test: every preset must mesh to a non-empty,
    # closed (no boundary edges) single component
    from collections import Counter
    for kind in ('MANDELBULB', 'JULIA', 'MANDELBOX'):
        verts, tris = build_escape_fractal(kind, res=64)
        edges = Counter()
        for tri in tris:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                edges[(min(a, b), max(a, b))] += 1
        boundary = sum(1 for c in edges.values() if c != 2)
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        print(f"{kind}: V={len(verts)} F={len(tris)} "
              f"boundary_edges={boundary} "
              f"bbox={np.round(hi - lo, 3)} "
              f"{'OK' if len(tris) and boundary == 0 else 'CHECK'}")
        assert len(tris) and boundary == 0
