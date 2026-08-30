
# Realistic Snowflake Generator for Blender
#
# Snow-crystal growth by Reiter's cellular automaton on a triangular
# (hexagonal) lattice. Each cell holds an amount of water s; a cell is
# "frozen" (part of the crystal) once s >= 1 and "receptive" if it is
# frozen or touches a frozen cell. Each step the water is split into a
# receptive part (which stays put and gains a constant vapour input
# gamma) and a non-receptive part (which diffuses to the six
# neighbours with coefficient alpha), after which the two are
# recombined. From a single frozen seed this grows the full range of
# real snow-crystal habits -- hexagonal plates, sectored plates,
# stellar and fern-like dendrites -- all with exact six-fold symmetry
# because the lattice update is isotropic.
#
# The crystal is turned into geometry by extruding each frozen cell to
# a hexagonal prism whose height tracks its ice mass, so the plates,
# ribs and ridges of the real crystal appear as surface relief; the
# plate is symmetric about z = 0 (two-sided) and centred in a 2 m
# cube. Shared hexagon corners are welded, so a plate of equal-height
# cells is a single watertight surface, with risers stitching the
# height steps.
#
# References:
# - Clifford A. Reiter, "A local cellular model for snow crystal
#   growth", Chaos, Solitons & Fractals 23, 2005, pp. 1111-1119.
# - Related, physically richer models: Janko Gravner and David
#   Griffeath, "Modeling snow crystal growth II: A mesoscopic lattice
#   map with plausible dynamics", Physica D 237, 2008, pp. 385-404.
# - Snow-crystal morphology (the temperature/supersaturation habit
#   diagram): Ukichiro Nakaya, "Snow Crystals: Natural and
#   Artificial", Harvard University Press, 1954.

bl_info = {
    "name": "Snowflake (Reiter CA)",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Snowflake",
    "description": "Realistic snow crystals via Reiter's cellular "
                   "automaton",
    "category": "Add Mesh",
}

import math

import numpy as np
# The mathematics lives in the sibling `ifs` engine package;
# this module is the Blender layer over it.
try:
    from .ifs.automaton import (PRESETS, build_preset, build_snowflake,
                                simulate_reiter)
except ImportError:  # flat import outside the package
    from ifs.automaton import (PRESETS, build_preset, build_snowflake,
                               simulate_reiter)



# ==========================================================================
# Geometry: extrude each frozen cell to a mass-scaled hexagonal prism
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

    class MESH_OT_snowflake_add(bpy.types.Operator):
        """Grow a realistic snow crystal with Reiter's cellular
        automaton and build it as a two-sided hexagonal-relief plate"""
        bl_idname = "mesh.snowflake_add"
        bl_label = "Snowflake"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Habit",
            items=[(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='DENDRITE',
            description="Snow-crystal habit to grow")
        radius: IntProperty(
            name="Radius", default=70, min=16, max=180,
            description="Lattice radius in cells; growth stops when "
                        "the crystal nears this edge")
        max_steps: IntProperty(
            name="Max Steps", default=6000, min=100, max=40000,
            description="Cap on automaton steps (growth usually stops "
                        "earlier when it reaches the radius)")
        base: FloatProperty(
            name="Base Thickness", default=0.4, min=0.02, max=3.0,
            description="Prism height of a just-frozen cell (relative "
                        "to cell size, before the fit to 2 m)")
        relief: FloatProperty(
            name="Mass Relief", default=0.6, min=0.0, max=3.0,
            description="Extra height per unit of ice mass above 1, "
                        "giving the ridges and plateaus")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Overall size multiplier")
        smooth: BoolProperty(name="Smooth Shading", default=False,
                             description="Shade the surface smooth")

        def execute(self, context):
            label = PRESETS[self.preset][0]
            verts, faces, ncells, steps = build_preset(
                self.preset, radius=self.radius,
                max_steps=self.max_steps, base=self.base,
                relief=self.relief, scale=self.scale)
            if len(faces) == 0:
                self.report({'ERROR'}, "Crystal did not grow")
                return {'CANCELLED'}
            me = bpy.data.meshes.new("Snowflake")
            me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(f"Snowflake {label}", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{label}: {ncells} cells in {steps} steps, "
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('preset', 'radius', 'max_steps', 'base',
                      'relief', 'scale', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.snowflake_add", icon='FREEZE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_snowflake_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_snowflake_add)


def _selftest():
    from collections import Counter

    def rot60(idx, ar):
        # axial 60-degree rotation (q, r) -> (-r, q + r)
        i, j = idx
        q, r = i - ar, j - ar
        return (-r + ar, (q + r) + ar)

    bad = []
    for kind in PRESETS:
        s, frozen, hexdist, steps = simulate_reiter(
            *PRESETS[kind][1:], radius=40, max_steps=4000)
        ar = frozen.shape[0] // 2
        cells = np.argwhere(frozen)
        # six-fold symmetry: rotating the frozen set by 60 degrees
        # maps it onto itself
        fset = set(map(tuple, cells))
        rset = {rot60(c, ar) for c in fset}
        sym = len(fset & rset) / max(len(fset), 1)
        verts, faces = build_snowflake(frozen, s)
        edges = Counter()
        for f in faces:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                edges[(min(a, b), max(a, b))] += 1
        boundary = sum(1 for c in edges.values() if c != 2)
        ok = len(cells) and sym > 0.99 and boundary == 0
        print(f"{kind}: cells={len(cells)} steps={steps} "
              f"6fold={sym:.3f} V={len(verts)} F={len(faces)} "
              f"boundary_edges={boundary} "
              f"{'OK' if ok else 'CHECK'}")
        if not ok:
            bad.append(kind)
    assert not bad
