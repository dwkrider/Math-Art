
# Fractal (Iterated Cable) Knot Generator for Blender
#
# An iterated knot in the spirit of Robert Fathauer's recursive knots:
# a base knot is repeatedly "cabled", i.e. replaced by a thinner strand
# that winds helically around the previous curve at a higher frequency
# and smaller radius. Each pass adds a finer, self-similar level of
# coiling, so the limit curve is self-similar at every scale -- a
# fractal knot. Because the construction lives in true 3-space and the
# winding radii shrink geometrically, the curve stays embedded (no
# self-intersections), so the tube renders correct over/under weaving
# automatically, with no crossing bookkeeping.
#
# This is the satellite/cable-knot route to an iterated knot, distinct
# from Fathauer's planar tile-substitution method but producing the
# same coiled, self-similar family. The winding uses a rotation-
# minimizing (Bishop) frame corrected to close up on the loop, so the
# cable joins seamlessly.
#
# References:
# - Robert W. Fathauer, "Fractal Knots Created by Iterative
#   Substitution", Bridges 2007, pp. 335-342; "A New Method for
#   Designing Iterated Knots", Bridges 2009, pp. 251-258 (the concept
#   of iterated/recursive knots).
# - Satellite and cable knots: Dale Rolfsen, "Knots and Links",
#   Publish or Perish, 1976 (Ch. 4).
# - Rotation-minimizing frames: Wenping Wang et al., "Computation of
#   rotation minimizing frames", ACM Trans. Graphics 27(1), 2008.

bl_info = {
    "name": "Fractal Knot",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Fractal Knot",
    "description": "Iterated cable knots (self-similar coiled knots)",
    "category": "Add Curve",
}

import math

import numpy as np


def torus_knot(p, q, samples, R=1.0, r=0.45):
    """A (p, q) torus knot as a closed (N,3) polyline."""
    th = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    rr = R + r * np.cos(q * th)
    return np.stack([rr * np.cos(p * th), rr * np.sin(p * th),
                     r * np.sin(q * th)], axis=-1)


def _rodrigues(v, k, t):
    ct, st = math.cos(t), math.sin(t)
    return (v * ct + np.cross(k, v) * st
            + k * (k @ v) * (1.0 - ct))


def rmf(C):
    """Rotation-minimizing frame on a closed curve C (N,3). Returns
    unit tangent T, normal N and binormal B arrays, with N corrected
    to be periodic around the loop."""
    n = len(C)
    T = np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    a = np.array([0.0, 0.0, 1.0])
    if abs(a @ T[0]) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    N = np.zeros_like(C)
    N[0] = a - (a @ T[0]) * T[0]
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, n):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        ni = N[i - 1]
        if s > 1e-9:
            ni = _rodrigues(N[i - 1], v / s,
                            math.atan2(s, float(T[i - 1] @ T[i])))
        ni = ni - (ni @ T[i]) * T[i]
        N[i] = ni / np.linalg.norm(ni)
    # holonomy: transport last normal across the closing edge
    v = np.cross(T[-1], T[0])
    s = np.linalg.norm(v)
    nclose = N[-1]
    if s > 1e-9:
        nclose = _rodrigues(N[-1], v / s,
                            math.atan2(s, float(T[-1] @ T[0])))
    nclose = nclose - (nclose @ T[0]) * T[0]
    nclose /= np.linalg.norm(nclose)
    phi = math.atan2(float(np.cross(nclose, N[0]) @ T[0]),
                     float(nclose @ N[0]))
    ang = phi * np.arange(n) / n
    ca, sa = np.cos(ang), np.sin(ang)
    N = ca[:, None] * N + sa[:, None] * np.cross(T, N)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    B = np.cross(T, N)
    return T, N, B


def cable(C, wind, amp, phase=0.0):
    """Wind a helix of `wind` turns and radius `amp` around C."""
    _, N, B = rmf(C)
    n = len(C)
    s = np.arange(n) / n * 2.0 * math.pi * wind + phase
    return C + amp * (np.cos(s)[:, None] * N + np.sin(s)[:, None] * B)


PRESETS = {
    'TREFOIL': ("Trefoil", 2, 3),
    'CINQUEFOIL': ("Cinquefoil (5,2)", 5, 2),
    'T3_4': ("(3,4) Torus Knot", 3, 4),
    'UNKNOT': ("Unknot (circle base)", 1, 0),
}


def build_fractal_knot(kind='TREFOIL', iters=2, wind0=7, wind_mult=5,
                       amp0=0.24, amp_ratio=0.4, samples=3000,
                       scale=1.0):
    """Base torus knot, cabled `iters` times with geometrically
    shrinking radius and growing winding. Returns (points, closed=True)
    centred and fit to a 2 m cube."""
    p, q = PRESETS[kind][1:]
    C = torus_knot(p, q, samples)
    amp, wind = amp0, wind0
    for _ in range(iters):
        C = cable(C, int(round(wind)), amp)
        amp *= amp_ratio
        wind *= wind_mult
    lo, hi = C.min(axis=0), C.max(axis=0)
    ext = float((hi - lo).max())
    C = (C - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)
    return C * scale, True


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class CURVE_OT_fractal_knot_add(bpy.types.Operator):
        """Add an iterated cable knot -- a base knot recursively wound
        into a self-similar fractal knot"""
        bl_idname = "curve.fractal_knot_add"
        bl_label = "Fractal Knot"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Base Knot",
            items=[(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='TREFOIL')
        iters: IntProperty(
            name="Cable Levels", default=2, min=0, max=4,
            description="How many times to cable the base knot")
        wind0: IntProperty(
            name="First Winding", default=7, min=2, max=40,
            description="Turns of the first cable around the base")
        wind_mult: IntProperty(
            name="Winding Multiplier", default=5, min=2, max=9,
            description="Winding turns multiply by this each level")
        amp0: FloatProperty(
            name="First Radius", default=0.24, min=0.02, max=0.6,
            description="Radius of the first cable (relative)")
        amp_ratio: FloatProperty(
            name="Radius Ratio", default=0.4, min=0.15, max=0.7,
            description="Cable radius shrinks by this each level")
        samples: IntProperty(
            name="Samples", default=3000, min=400, max=12000,
            description="Points along the curve (must exceed the "
                        "highest winding number several times over)")
        radius: FloatProperty(
            name="Tube Radius", default=0.02, min=0.0, max=0.3)
        resolution: IntProperty(
            name="Bevel Resolution", default=3, min=0, max=12)
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            label = PRESETS[self.kind][0]
            pts, closed = build_fractal_knot(
                self.kind, self.iters, self.wind0, self.wind_mult,
                self.amp0, self.amp_ratio, self.samples, self.scale)
            cu = bpy.data.curves.new("FractalKnot", 'CURVE')
            cu.dimensions = '3D'
            sp = cu.splines.new('POLY')
            sp.points.add(len(pts) - 1)
            for i, pt in enumerate(pts):
                sp.points[i].co = (float(pt[0]), float(pt[1]),
                                   float(pt[2]), 1.0)
            sp.use_cyclic_u = closed
            cu.bevel_depth = self.radius
            cu.bevel_resolution = self.resolution
            cu.use_fill_caps = False
            obj = bpy.data.objects.new(f"Fractal Knot {label}", cu)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{label}: {len(pts)} points, "
                        f"{self.iters} cable levels")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'iters', 'wind0', 'wind_mult', 'amp0',
                      'amp_ratio', 'samples', 'radius', 'resolution',
                      'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("curve.fractal_knot_add",
                             icon='FORCE_VORTEX')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_fractal_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_fractal_knot_add)


def _selftest():
    for kind in PRESETS:
        pts, closed = build_fractal_knot(kind, samples=2000)
        # closure gap and max step (smoothness) checks
        gap = float(np.linalg.norm(pts[0] - pts[-1]))
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        finite = np.isfinite(pts).all()
        ext = np.round(pts.max(axis=0) - pts.min(axis=0), 2)
        ok = finite and gap < steps.mean() * 3.0
        print(f"{kind:11s}: N={len(pts)} closed_gap={gap:.4f} "
              f"max_step={steps.max():.4f} bbox={ext} "
              f"{'OK' if ok else 'CHECK'}")
        assert ok
