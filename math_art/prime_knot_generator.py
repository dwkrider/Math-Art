
# Prime Knot Generator for Blender
#
# All 249 prime knots with up to 10 crossings (Rolfsen table), built
# from the minimum braid words of Thomas Gittings ("Minimum braids: a
# complete invariant of knots and links", arXiv:math/0401051, Table
# 1). The braid closure is laid out around a circle -- strands at
# radii by braid level, crossings as over/under bumps -- and then
# relaxed with curve smoothing plus self-repulsion into a rounded,
# KnotPlot-like presentation.
#
# Every braid word in the table is verified programmatically (run
# this file with plain python): the closure permutation must be a
# single cycle, and the Alexander polynomial computed via the reduced
# Burau representation must match Gittings' published value at t=10.
#
# Output styles follow the classic Torus Knot + add-on: a Bezier /
# Poly / NURBS curve with bevel radius, or a mesh tube.
#
# References:
# - Dale Rolfsen, "Knots and Links", Publish or Perish, 1976 (the
#   Rolfsen table and its knot numbering).
# - Knot nomenclature n_k after J. W. Alexander & G. B. Briggs
#   (1926/27), extending P. G. Tait's 19th-century enumeration.
# - Thomas A. Gittings, "Minimum braids: a complete invariant of
#   knots and links", arXiv:math/0401051, 2004 (Table 1 braid words).
# - Verification via the Alexander polynomial (J. W. Alexander, 1928)
#   computed from the reduced Burau representation (Werner Burau, 1935).

bl_info = {
    "name": "Prime Knots",
    "author": "Math Art project (braid table after Thomas "
              "Gittings)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Prime Knot",
    "description": "All prime knots to 10 crossings from verified "
                   "minimum braids",
    "category": "Add Curve",
}


# The engine below this line moved to the Blender-free package
# `math_art/knots/`: the braid table, word parsing, closure embedding, the
# Alexander polynomial, resampling, rope relaxation and the tube sweep.
# This module keeps the registered operator and re-exports the engine names
# so operator code -- and the three sibling generators that had come to
# import this module for them -- read as before.
#
# New code should import the package (`from .knots import build_knot`)
# rather than reaching through this module.

import math

from .knots import (KNOTS, alexander_at, braid_closure_points, build_knot,
                    closed_tube, closure_components, parse_letters,
                    relax_knot, resample_closed)


# ---- Blender layer -----------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           StringProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _ITEMS = [('CUSTOM', "Custom braid",
               "Closure of the braid word entered below")]
    _ITEMS += [(name, name.replace('_', '.'),
                f"{name.split('_')[0]} crossings, minimum braid "
                f"{braid}")
               for (name, braid, _ap) in KNOTS]
    _BRAIDS = {name: braid for (name, braid, _ap) in KNOTS}

    class CURVE_OT_prime_knot_add(bpy.types.Operator):
        """Add a prime knot (Rolfsen table, up to 10 crossings) from
        its verified minimum braid, relaxed into a smooth curve"""
        bl_idname = "curve.prime_knot_add"
        bl_label = "Prime Knot"
        bl_options = {'REGISTER', 'UNDO'}

        knot: EnumProperty(name="Knot", items=_ITEMS, default='3_1')
        braid: StringProperty(
            name="Braid Word", default="AAA",
            description="Letters a..z are braid generators, A..Z "
                        "their inverses (used for Custom)")
        samples: IntProperty(name="Curve Samples", default=240,
                             min=60, max=1000)
        iters: IntProperty(
            name="Relax Iterations", default=150, min=0, max=600,
            description="Smoothing + self-repulsion rounds (0 shows "
                        "the raw braid closure)")
        repel: FloatProperty(
            name="Strand Clearance", default=0.35, min=0.05, max=1.0,
            description="Self-repulsion distance while relaxing")
        mirror: BoolProperty(
            name="Mirror", default=False,
            description="Mirror image of the knot")
        output: EnumProperty(
            name="Output",
            items=[('BEZIER', "Bezier Curve", "auto-smoothed"),
                   ('POLY', "Poly Curve", ""),
                   ('NURBS', "NURBS Curve", ""),
                   ('MESH', "Mesh Tube", "swept tube mesh")],
            default='BEZIER')
        radius: FloatProperty(
            name="Tube Radius", default=0.08, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius")
        resolution: IntProperty(name="Bevel Resolution", default=6,
                                min=1, max=16)
        tube_sides: IntProperty(name="Tube Sides", default=12,
                                min=3, max=32)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            braid = (self.braid if self.knot == 'CUSTOM'
                     else _BRAIDS[self.knot])
            try:
                P = build_knot(braid, self.samples, self.iters,
                               self.repel, self.scale, self.mirror)
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            name = ("Knot " + self.knot.replace('_', '.')
                    if self.knot != 'CUSTOM' else "Knot custom")
            if self.output == 'MESH':
                verts, faces = self._tube(P, self.radius,
                                          self.tube_sides)
                me = bpy.data.meshes.new(name)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
                me.update()
                obj = bpy.data.objects.new(name, me)
            else:
                cu = bpy.data.curves.new(name, 'CURVE')
                cu.dimensions = '3D'
                if self.output == 'BEZIER':
                    sp = cu.splines.new('BEZIER')
                    sp.bezier_points.add(len(P) - 1)
                    for i, p in enumerate(P):
                        bp = sp.bezier_points[i]
                        bp.co = p
                        bp.handle_left_type = 'AUTO'
                        bp.handle_right_type = 'AUTO'
                else:
                    sp = cu.splines.new(self.output)
                    sp.points.add(len(P) - 1)
                    for i, p in enumerate(P):
                        sp.points[i].co = (p[0], p[1], p[2], 1.0)
                    if self.output == 'NURBS':
                        sp.order_u = 4
                sp.use_cyclic_u = True
                cu.bevel_depth = self.radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: braid {braid}, {len(P)} samples")
            return {'FINISHED'}

        @staticmethod
        def _tube(P, radius, sides):
            """Closed tube with parallel-transport frames; the frame
            holonomy is distributed so the seam matches."""
            import numpy as np
            m = len(P)
            T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
            T /= np.linalg.norm(T, axis=1)[:, None]
            N = np.empty_like(P)
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ref, T[0])) > 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            N[0] = np.cross(T[0], ref)
            N[0] /= np.linalg.norm(N[0])
            for i in range(1, m):
                v = N[i - 1] - T[i] * np.dot(N[i - 1], T[i])
                N[i] = v / (np.linalg.norm(v) or 1.0)
            # transport once more to measure the closure twist
            v = N[m - 1] - T[0] * np.dot(N[m - 1], T[0])
            v /= (np.linalg.norm(v) or 1.0)
            B0 = np.cross(T[0], N[0])
            ang = math.atan2(np.dot(v, B0), np.dot(v, N[0]))
            verts = []
            faces = []
            for i in range(m):
                corr = -ang * i / m
                B = np.cross(T[i], N[i])
                for j in range(sides):
                    a = 2 * math.pi * j / sides + corr
                    p = (P[i] + radius
                         * (math.cos(a) * N[i] + math.sin(a) * B))
                    verts.append(tuple(p))
            for i in range(m):
                i2 = (i + 1) % m
                for j in range(sides):
                    j2 = (j + 1) % sides
                    faces.append([i * sides + j, i * sides + j2,
                                  i2 * sides + j2, i2 * sides + j])
            return verts, faces

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'knot')
            if self.knot == 'CUSTOM':
                lay.prop(self, 'braid')
            for k in ('samples', 'iters', 'repel', 'mirror',
                      'output', 'radius'):
                lay.prop(self, k)
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            elif self.radius > 0:
                lay.prop(self, 'resolution')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.prime_knot_add",
                             icon='CURVE_DATA')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_prime_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_prime_knot_add)


def _selftest():
    fails = []
    counts = {}
    for (name, braid, ap) in KNOTS:
        word = parse_letters(braid)
        cr = int(name.split('_')[0])
        counts[cr] = counts.get(cr, 0) + 1
        ok = closure_components(word) == 1 \
            and alexander_at(word) == ap
        if not ok:
            fails.append(name)
            print(f"FAIL {name} {braid}")
    print(f"{len(KNOTS)} knots, per crossings: "
          f"{sorted(counts.items())}")
    print("braid table:",
          "ALL VERIFIED" if not fails else f"FAILURES {fails}")
    # embedding smoke test (no numpy relaxation here)
    for name in ('3_1', '4_1', '8_18', '10_124'):
        braid = dict((n, b) for (n, b, a) in KNOTS)[name]
        pts = braid_closure_points(parse_letters(braid))
        print(f"{name}: closure polyline {len(pts)} points")
    assert not fails
