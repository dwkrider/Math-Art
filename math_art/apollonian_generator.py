
# Apollonian Gasket / Sphere-Packing Generator for Blender
#
# The Apollonian gasket: begin with mutually tangent circles, inscribe
# a new circle in every curvilinear gap, and recurse forever. The 3D
# analogue packs mutually tangent spheres (Soddy spheres). Both are
# exactly self-similar fractals whose gaps are filled at every scale.
#
# Curvatures obey the Descartes circle theorem (2D) and its Soddy-
# Gosper generalisation (3D). This generator uses the sign-free
# "reflection" form: given a set of mutually tangent circles/spheres
# and one further tangent element c0, the OTHER element tangent to the
# same set is
#     k'   = 2*sum(k_i) - k0
#     k'c' = 2*sum(k_i c_i) - k0 c0
# so each gap is filled without any square-root sign ambiguity, and
# every sub-gap recurses the same way. Recursion stops at a minimum
# radius. Circles are drawn as rings (tori in the plane); spheres as
# icospheres. Output is centred and fit to a 2 m cube.
#
# References:
# - Rene Descartes (1643, letter to Princess Elisabeth); rediscovered
#   by Frederick Soddy, "The Kiss Precise", Nature 137, 1936, p. 1021
#   (verse giving both the 2D theorem and its 3D sphere version).
# - Apollonius of Perga, "Tangencies" (lost; c. 200 BC), the classical
#   problem of tangent circles.
# - Jeffrey C. Lagarias, Colin L. Mallows and Allan R. Wilf, "Beyond
#   the Descartes Circle Theorem", Amer. Math. Monthly 109, 2002,
#   pp. 338-361 (the curvature-centre / reflection formulation).

bl_info = {
    "name": "Apollonian Gasket",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Apollonian Gasket",
    "description": "Apollonian circle gasket and 3D sphere packing",
    "category": "Add Mesh",
}

import cmath
import math

import numpy as np


# The mathematics lives in the sibling `ifs` engine package;
# this module is the Blender layer over it.
try:
    from .ifs.inversive import (_PALETTE, build_apollonian, gasket_2d,
                                packing_3d)
except ImportError:  # flat import outside the package
    from ifs.inversive import (_PALETTE, build_apollonian, gasket_2d,
                               packing_3d)


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

    def _apollonian_mesh(name, verts, faces, mats, smooth):
        """Build a mesh with the palette materials assigned per face."""
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in verts], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        for idx, rgba in enumerate(_PALETTE):
            nm = "Apollonian_%d" % idx
            m = bpy.data.materials.get(nm)
            if m is None:
                m = bpy.data.materials.new(nm)
                m.diffuse_color = rgba
            me.materials.append(m)
        if mats and len(mats) == len(me.polygons):
            me.polygons.foreach_set('material_index', mats)
        if smooth:
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        return me

    class MESH_OT_apollonian_add(bpy.types.Operator):
        """Add an Apollonian gasket (2D rings) or sphere packing"""
        bl_idname = "mesh.apollonian_add"
        bl_label = "Apollonian Gasket"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            items=[('PACKING', "Sphere Packing (3D)",
                    "Mutually tangent Soddy spheres"),
                   ('GASKET', "Gasket (2D)",
                    "Apollonian circles in the plane")],
            default='PACKING',
            description="Build a 3D sphere packing or a 2D circle gasket")
        gasket_style: EnumProperty(
            name="Circle Style",
            items=[('FILLED', "Filled Discs",
                    "Each circle a flat filled face (2D)"),
                   ('TUBE', "Tube Rings",
                    "Each circle a raised tube ring (2D)")],
            default='FILLED',
            description="How the 2D gasket circles are drawn")
        color_by: EnumProperty(
            name="Color By",
            items=[('SIZE', "Size", "Color by circle/sphere radius"),
                   ('DEPTH', "Depth", "Color by inscription generation"),
                   ('UNIFORM', "Uniform", "A single color")],
            default='SIZE',
            description="What the circle/sphere colours are keyed to")
        depth: IntProperty(
            name="Depth", default=6, min=1, max=16,
            description="Gap-filling generations (packing fills largest-"
                        "first to Max Count; a sphere packing never fills "
                        "space fully -- the gaps are a fractal)")
        min_r: FloatProperty(
            name="Min Radius", default=0.0, min=0.0, max=0.5,
            description="Stop inscribing below this radius "
                        "(0 = per-mode default)")
        cap: IntProperty(
            name="Max Count", default=40000, min=10, max=200000,
            description="Hard cap on circles/spheres (packing fills the "
                        "largest this many)")
        tube_ratio: FloatProperty(
            name="Ring Tube", default=0.06, min=0.01, max=0.5,
            description="Tube radius as a fraction of circle radius "
                        "(gasket Tube Rings style)")
        inflate: FloatProperty(
            name="Sphere Inflate", default=1.0, min=1.0, max=1.15,
            description="Grow spheres slightly to fuse contacts for "
                        "printing (packing mode)")
        sphere_res: IntProperty(
            name="Sphere Resolution", default=1, min=1, max=5,
            description="Icosphere subdivision level (packing mode); "
                        "higher rounds spheres so tangent spheres touch")
        ring_seg: IntProperty(name="Ring Segments", default=20,
                              min=6, max=64,
                              description="Segments around each circle ring")
        tube_seg: IntProperty(name="Tube Segments", default=8,
                              min=4, max=32,
                              description="Segments around the ring's tube "
                                          "cross-section")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                            max=100.0,
                            description="Overall size; fits the result into "
                                        "a 2 m cube times this factor")
        smooth: BoolProperty(name="Smooth Shading", default=True,
                             description="Shade the surfaces smooth")
        separate_levels: BoolProperty(
            name="Separate Levels", default=False,
            description="Output each inscription level as its own object "
                        "(parented to an empty) so a level can be hidden "
                        "or deleted individually")

        def execute(self, context):
            verts, faces, mats, levels, n = build_apollonian(
                self.mode, self.depth, self.min_r, self.cap,
                self.tube_ratio, self.inflate, self.ring_seg,
                self.tube_seg, self.scale, self.gasket_style,
                self.color_by, self.sphere_res)
            verts = [tuple(v) for v in np.asarray(verts)]
            for o in context.selected_objects:
                o.select_set(False)
            if self.separate_levels:
                from collections import defaultdict
                groups = defaultdict(
                    lambda: {'vmap': {}, 'V': [], 'F': [], 'M': []})
                for fi, f in enumerate(faces):
                    g = groups[levels[fi]]
                    nf = []
                    for vi in f:
                        if vi not in g['vmap']:
                            g['vmap'][vi] = len(g['V'])
                            g['V'].append(verts[vi])
                        nf.append(g['vmap'][vi])
                    g['F'].append(tuple(nf))
                    g['M'].append(mats[fi])
                parent = bpy.data.objects.new("Apollonian", None)
                context.collection.objects.link(parent)
                parent.location = context.scene.cursor.location
                for lv in sorted(groups):
                    g = groups[lv]
                    me = _apollonian_mesh("Apollonian_L%d" % lv, g['V'],
                                          g['F'], g['M'], self.smooth)
                    ob = bpy.data.objects.new("Apollonian_L%d" % lv, me)
                    context.collection.objects.link(ob)
                    ob.parent = parent
                parent.select_set(True)
                context.view_layer.objects.active = parent
                self.report({'INFO'}, "%s: %d elements in %d level objects"
                            % (self.mode, n, len(groups)))
            else:
                me = _apollonian_mesh("Apollonian", verts, faces, mats,
                                      self.smooth)
                obj = bpy.data.objects.new("Apollonian", me)
                context.collection.objects.link(obj)
                obj.location = context.scene.cursor.location
                obj.select_set(True)
                context.view_layer.objects.active = obj
                self.report({'INFO'}, "%s: %d elements, V=%d F=%d"
                            % (self.mode, n, len(me.vertices),
                               len(me.polygons)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            lay.prop(self, 'depth')
            lay.prop(self, 'min_r')
            lay.prop(self, 'cap')
            lay.prop(self, 'color_by')
            if self.mode == 'GASKET':
                lay.prop(self, 'gasket_style')
                if self.gasket_style == 'TUBE':
                    lay.prop(self, 'tube_ratio')
                    lay.prop(self, 'tube_seg')
                lay.prop(self, 'ring_seg')
            else:
                lay.prop(self, 'sphere_res')
                lay.prop(self, 'inflate')
            lay.prop(self, 'scale')
            lay.prop(self, 'smooth')
            lay.prop(self, 'separate_levels')

    def _menu_func(self, context):
        self.layout.operator("mesh.apollonian_add", icon='MESH_CIRCLE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_apollonian_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_apollonian_add)


def _selftest():
    # 2D: verify the packing is valid (no two positive circles
    # overlap; all inside the outer circle)
    balls = gasket_2d(depth=4, min_r=0.01, cap=3000)
    pos = [b for b in balls if b.k > 0]
    outer = next(b for b in balls if b.k < 0)
    Rout = 1.0 / abs(outer.k)
    bad = 0
    for i in range(len(pos)):
        bi = pos[i]
        ri = 1.0 / bi.k
        if np.linalg.norm(bi.c - outer.c) + ri > Rout + 1e-6:
            bad += 1
        for j in range(i + 1, len(pos)):
            bj = pos[j]
            rj = 1.0 / bj.k
            if (np.linalg.norm(bi.c - bj.c)
                    < ri + rj - 1e-6):
                bad += 1
    print(f"GASKET  depth4: circles={len(balls)} overlaps={bad} "
          f"{'OK' if bad == 0 else 'BAD'}")
    # 3D: same validity check
    sph = packing_3d(depth=3, min_r=0.03, cap=3000)
    bad3 = 0
    for i in range(len(sph)):
        ri = 1.0 / sph[i].k
        for j in range(i + 1, len(sph)):
            rj = 1.0 / sph[j].k
            if (np.linalg.norm(sph[i].c - sph[j].c)
                    < ri + rj - 1e-6):
                bad3 += 1
    v, f, m, lv, n = build_apollonian('PACKING', depth=3, min_r=0.05)
    print(f"PACKING depth3: spheres={len(sph)} overlaps={bad3} "
          f"mesh_V={len(v)} F={len(f)} mats={len(set(m))} "
          f"levels={len(set(lv))} "
          f"{'OK' if bad3 == 0 else 'BAD'}")
    # filled 2D gasket colored by size builds with per-face mats
    gv, gf, gm, glv, gn = build_apollonian('GASKET', depth=4,
                                           min_r=0.02,
                                           gasket_style='FILLED',
                                           color_by='SIZE')
    print(f"GASKET filled: circles={gn} faces={len(gf)} "
          f"mats={len(set(gm))} "
          f"{'OK' if len(gf) == gn and len(gm) == len(gf) else 'BAD'}")
    assert bad == 0 and bad3 == 0
    assert len(gf) == gn and len(gm) == len(gf)
