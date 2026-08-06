
# Dual Helix Generator for Blender
#
# A sculpture of two nested counter-rotating helices joined into one
# closed loop: an outer helix (radius blending join -> outer -> join)
# spirals down while an inner helix (join -> inner -> join) spirals
# back up, both with a cosine height profile, so the two ends meet
# exactly at the top and bottom. The loop closes whenever the two
# turn counts sum to a whole number of turns (auto-snapped by
# default). Generalised from the construction in the user's
# wonder2.blend script.
#
# References:
#   - Two nested counter-rotating helices joined into a closed loop
#     is a classical curve construction; the parametrization here
#     (raised-cosine radius blend, signed-power cosine height,
#     whole-turn closure) is original to this project.
#   - Form inspired by Tom Lawton's kinetic sculpture "Wonder".

bl_info = {
    "name": "Dual Helix",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Dual Helix",
    "description": "Closed loop of two nested counter-rotating "
                   "tapered helices",
    "category": "Add Curve",
}

import math
from math import cos, sin, pi, copysign


def dual_helix_points(outer_radius=1.0, inner_radius=0.35,
                      join_radius=0.75, half_height=2 * pi / 5,
                      outer_turns=1.5, inner_turns=2.5,
                      seg_per_turn=120, z_shape=1.0, scale=1.0):
    """Closed polyline of the dual helix. Returns (points, closed):
    closed is True when the turn counts sum to a whole number so the
    inner helix arrives back at the outer start."""
    closed = abs((outer_turns + inner_turns) % 1.0) < 1e-6 \
        or abs((outer_turns + inner_turns) % 1.0 - 1.0) < 1e-6

    def zprof(c):
        return copysign(abs(c) ** z_shape, c)

    pts = []
    n1 = max(8, int(round(outer_turns * seg_per_turn)))
    for i in range(n1 + 1):
        t = i / n1
        w = (1 + cos(2 * pi * t)) / 2
        r = (w * join_radius + (1 - w) * outer_radius) * scale
        ang = 2 * pi * outer_turns * t
        z = half_height * zprof(cos(pi * t)) * scale
        pts.append((r * cos(ang), r * sin(ang), z))
    a0 = 2 * pi * outer_turns
    n2 = max(8, int(round(inner_turns * seg_per_turn)))
    for i in range(1, n2 + 1):
        s = i / n2
        w = (1 + cos(2 * pi * s)) / 2
        r = (w * join_radius + (1 - w) * inner_radius) * scale
        ang = a0 + 2 * pi * inner_turns * s
        z = -half_height * zprof(cos(pi * s)) * scale
        pts.append((r * cos(ang), r * sin(ang), z))
    if closed:
        pts.pop()                        # last point == first point
    return pts, closed


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, BoolProperty,
                           EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class CURVE_OT_dual_helix_add(bpy.types.Operator):
        """Two nested counter-rotating tapered helices joined into
        one closed sculptural loop"""
        bl_idname = "curve.dual_helix_add"
        bl_label = "Dual Helix"
        bl_options = {'REGISTER', 'UNDO'}

        outer_radius: FloatProperty(name="Outer Radius", default=1.0,
                                    min=0.05, max=20.0)
        inner_radius: FloatProperty(name="Inner Radius", default=0.35,
                                    min=0.01, max=20.0)
        join_radius: FloatProperty(
            name="Join Radius", default=0.75, min=0.02, max=20.0,
            description="Radius where the two helices meet at the "
                        "top and bottom")
        half_height: FloatProperty(name="Half Height",
                                   default=2 * pi / 5, min=0.05,
                                   max=20.0)
        outer_turns: FloatProperty(name="Outer Turns", default=1.5,
                                   min=0.5, max=12.0, step=25)
        inner_turns: FloatProperty(
            name="Inner Turns", default=2.5, min=0.5, max=12.0,
            step=25,
            description="Snapped so the turn counts sum to a whole "
                        "number when Auto Close is on")
        auto_close: BoolProperty(
            name="Auto Close", default=True,
            description="Adjust the inner turns so the loop closes "
                        "exactly (outer + inner turns must be a "
                        "whole number)")
        z_shape: FloatProperty(
            name="Height Profile", default=1.0, min=0.25, max=4.0,
            description="Exponent of the cosine height profile "
                        "(1 = the original; higher flattens the "
                        "middle, lower sharpens it)")
        seg_per_turn: IntProperty(name="Segments Per Turn",
                                  default=120, min=16, max=720)
        spline_type: EnumProperty(
            name="Spline",
            items=[('NURBS', "NURBS", "smooth, as in the original"),
                   ('POLY', "Poly", ""),
                   ('BEZIER', "Bezier", "auto handles")],
            default='NURBS')
        tube_major: FloatProperty(
            name="Tube Major Axis", default=0.09, min=0.0, max=2.0,
            step=1, precision=3,
            description="Half-width of the elliptical tube profile "
                        "(0 = wire only)")
        tube_minor: FloatProperty(
            name="Tube Minor Axis", default=0.045, min=0.0, max=2.0,
            step=1, precision=3,
            description="Half-height of the elliptical tube profile")
        tube_rotation: FloatProperty(
            name="Profile Rotation", default=0.0, min=-90.0,
            max=90.0,
            description="Rotation of the ellipse in the tube "
                        "cross-section (degrees)")
        resolution: IntProperty(name="Profile Resolution", default=6,
                                min=1, max=32)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            inner = self.inner_turns
            if self.auto_close:
                total = round(self.outer_turns + self.inner_turns)
                snapped = max(0.5, total - self.outer_turns)
                if abs(snapped - self.inner_turns) > 1e-6:
                    self.report({'INFO'},
                                f"inner turns snapped to {snapped:g} "
                                f"to close the loop")
                inner = snapped
            pts, closed = dual_helix_points(
                self.outer_radius, self.inner_radius,
                self.join_radius, self.half_height,
                self.outer_turns, inner, self.seg_per_turn,
                self.z_shape, self.scale)
            cu = bpy.data.curves.new("DualHelix", 'CURVE')
            cu.dimensions = '3D'
            if self.spline_type == 'BEZIER':
                sp = cu.splines.new('BEZIER')
                sp.bezier_points.add(len(pts) - 1)
                for i, p in enumerate(pts):
                    bp = sp.bezier_points[i]
                    bp.co = p
                    bp.handle_left_type = 'AUTO'
                    bp.handle_right_type = 'AUTO'
            else:
                sp = cu.splines.new(self.spline_type)
                sp.points.add(len(pts) - 1)
                for i, p in enumerate(pts):
                    sp.points[i].co = (p[0], p[1], p[2], 1.0)
                if self.spline_type == 'NURBS':
                    sp.order_u = 4
            sp.use_cyclic_u = closed
            obj = bpy.data.objects.new("DualHelix", cu)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.tube_major > 0 and self.tube_minor > 0:
                # elliptical sweep profile as a bevel object (as in
                # the original file's scaled bevel circle)
                prof = bpy.data.curves.new("DualHelixProfile",
                                           'CURVE')
                prof.dimensions = '2D'
                prof.resolution_u = self.resolution
                psp = prof.splines.new('BEZIER')
                psp.bezier_points.add(3)
                a, b = self.tube_major, self.tube_minor
                k = 0.5522847                # circle handle factor
                rot = math.radians(self.tube_rotation)
                cr, sr = cos(rot), sin(rot)

                def rot2(x, y):
                    return (x * cr - y * sr, x * sr + y * cr, 0.0)

                data = [((a, 0), (a, -k * b), (a, k * b)),
                        ((0, b), (k * a, b), (-k * a, b)),
                        ((-a, 0), (-a, k * b), (-a, -k * b)),
                        ((0, -b), (-k * a, -b), (k * a, -b))]
                for bp, (co, hl, hr) in zip(psp.bezier_points, data):
                    bp.co = rot2(*co)
                    bp.handle_left = rot2(*hl)
                    bp.handle_right = rot2(*hr)
                    bp.handle_left_type = 'FREE'
                    bp.handle_right_type = 'FREE'
                psp.use_cyclic_u = True
                prof_obj = bpy.data.objects.new("DualHelixProfile",
                                                prof)
                context.collection.objects.link(prof_obj)
                prof_obj.parent = obj
                prof_obj.hide_render = True
                cu.bevel_mode = 'OBJECT'
                cu.bevel_object = prof_obj
                if not closed:
                    cu.use_fill_caps = True
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{len(pts)} points"
                        + (", closed loop" if closed else ", open"))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('outer_radius', 'inner_radius', 'join_radius',
                      'half_height', 'outer_turns', 'inner_turns',
                      'auto_close', 'z_shape', 'seg_per_turn',
                      'spline_type', 'tube_major', 'tube_minor',
                      'tube_rotation', 'resolution', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("curve.dual_helix_add",
                             icon='MOD_SCREW')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_dual_helix_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_dual_helix_add)


def _selftest():
    # default parameters must close exactly
    pts, closed = dual_helix_points()
    gap = math.dist(pts[0], pts[-1])
    n1 = int(round(1.5 * 120))
    # joint between the two helices must be continuous
    j = math.dist(pts[n1], pts[n1 + 1])
    step = math.dist(pts[0], pts[1])
    print(f"points={len(pts)} closed={closed} "
          f"joint step={j:.4f} (~{step:.4f}) "
          f"{'OK' if closed and j < 5 * step else 'BAD'}")
    assert closed and j < 5 * step
    pts, closed = dual_helix_points(outer_turns=2.0,
                                    inner_turns=3.0)
    print(f"2+3 turns closed={closed} "
          f"{'OK' if closed else 'BAD'}")
    assert closed
    pts, closed = dual_helix_points(outer_turns=1.5,
                                    inner_turns=2.0)
    print(f"1.5+2 turns closed={closed} (expect False) "
          f"{'OK' if not closed else 'BAD'}")
    assert not closed
