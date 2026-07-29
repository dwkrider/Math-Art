
# L-system Turtle Generator for Blender
#
# Lindenmayer systems drawn by a 3D turtle. A short axiom is rewritten
# by parallel string-substitution rules for a number of iterations,
# then interpreted as turtle commands: draw forward, turn, pitch, roll,
# and push/pop for branching. The resulting poly-lines are emitted as a
# single bevelled curve, so classic space-filling and plant fractals
# come out as printable tubes.
#
# Turtle alphabet:
#   F, A, B ... draw forward one step (which letters draw is per preset)
#   f          move forward without drawing
#   + -        yaw left / right        (about the up axis)
#   & ^        pitch down / up         (about the left axis)
#   \ /        roll left / right       (about the heading axis)
#   |          turn 180 degrees
#   [ ]        push / pop turtle state (a branch)
#   other letters (X, Y ...) are rewrite-only and draw nothing
#
# Presets: the Koch snowflake, Levy C curve, Heighway dragon,
# Sierpinski arrowhead, Gosper (flowsnake) curve, a 2D plant, and a
# 3D bush. Hilbert/Peano/Moore space-filling curves are deliberately
# omitted -- the Space-Filling Curve generator already covers those.
#
# References:
# - Aristid Lindenmayer, "Mathematical models for cellular
#   interactions in development", J. Theoretical Biology 18, 1968.
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990 (the turtle interpretation and
#   most of these production rules).
# - Gosper curve / flowsnake: Martin Gardner, "Mathematical Games",
#   Scientific American 235(6), 1976 (after William Gosper).

bl_info = {
    "name": "L-system Fractals",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > L-system",
    "description": "Koch, dragon, Gosper, Sierpinski arrowhead, "
                   "plants via Lindenmayer systems",
    "category": "Add Curve",
}

import math

import numpy as np


# (axiom, rules, angle_deg, draw_symbols, default_iters, max_iters)
PRESETS = {
    'KOCH': ("Koch Snowflake", "F--F--F", {'F': "F+F--F+F"}, 60.0,
             "F", 3, 5),
    'LEVY': ("Levy C Curve", "F", {'F': "+F--F+"}, 45.0, "F", 10, 14),
    'DRAGON': ("Heighway Dragon", "FX",
               {'X': "X+YF+", 'Y': "-FX-Y"}, 90.0, "F", 11, 16),
    'ARROWHEAD': ("Sierpinski Arrowhead", "A",
                  {'A': "B-A-B", 'B': "A+B+A"}, 60.0, "AB", 6, 9),
    'GOSPER': ("Gosper (Flowsnake)", "A",
               {'A': "A-B--B+A++AA+B-", 'B': "+A-BB--B-A++A+B"},
               60.0, "AB", 4, 5),
    'PLANT': ("Plant", "X",
              {'X': "F+[[X]-X]-F[-FX]+X", 'F': "FF"}, 25.0, "F", 5, 7),
    'BUSH3D': ("3D Bush", "A",
               {'A': "F[&A]////[&A]////[&A]////"}, 25.0, "F", 5, 6),
}


def expand(axiom, rules, iters):
    s = axiom
    for _ in range(iters):
        s = "".join(rules.get(c, c) for c in s)
    return s


def _rot(v, k, t):
    """Rodrigues rotation of vector v about unit axis k by angle t."""
    ct, st = math.cos(t), math.sin(t)
    return v * ct + np.cross(k, v) * st + k * float(np.dot(k, v)) * (1 - ct)


def turtle(s, angle_deg, draw_syms, step=1.0):
    """Interpret the command string, returning a list of poly-lines
    (each an (m,3) array of at least two points)."""
    ang = math.radians(angle_deg)
    H = np.array([0.0, 0.0, 1.0])       # heading (grows +z)
    L = np.array([1.0, 0.0, 0.0])       # left
    U = np.array([0.0, 1.0, 0.0])       # up
    pos = np.zeros(3)
    cur = [pos.copy()]
    lines = []
    stack = []

    def spin(axis, t):
        return _rot(H, axis, t), _rot(L, axis, t), _rot(U, axis, t)

    for c in s:
        if c in draw_syms:
            pos = pos + H * step
            cur.append(pos.copy())
        elif c == 'f':
            if len(cur) >= 2:
                lines.append(np.array(cur))
            pos = pos + H * step
            cur = [pos.copy()]
        elif c == '+':
            H, L, U = spin(U, ang)
        elif c == '-':
            H, L, U = spin(U, -ang)
        elif c == '&':
            H, L, U = spin(L, ang)
        elif c == '^':
            H, L, U = spin(L, -ang)
        elif c == '\\':
            H, L, U = spin(H, ang)
        elif c == '/':
            H, L, U = spin(H, -ang)
        elif c == '|':
            H, L, U = spin(U, math.pi)
        elif c == '[':
            stack.append((pos.copy(), H.copy(), L.copy(), U.copy()))
        elif c == ']':
            if len(cur) >= 2:
                lines.append(np.array(cur))
            pos, H, L, U = stack.pop()
            pos, H, L, U = pos.copy(), H.copy(), L.copy(), U.copy()
            cur = [pos.copy()]
    if len(cur) >= 2:
        lines.append(np.array(cur))
    return lines


def build_lsystem(kind='KOCH', iters=-1, scale=1.0):
    """Return (list of (m,3) poly-lines, closed) centred and fit to a
    2 m cube. `iters` < 0 uses the preset default."""
    label, axiom, rules, angle, draw, dfl, mx = PRESETS[kind]
    it = dfl if iters < 0 else min(iters, mx)
    s = expand(axiom, rules, it)
    lines = turtle(s, angle, draw)
    if not lines:
        return [], False
    allpts = np.vstack(lines)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    ext = float((hi - lo).max())
    cen = 0.5 * (lo + hi)
    f = (2.0 / ext if ext > 1e-9 else 1.0) * scale
    lines = [(ln - cen) * f for ln in lines]
    closed = kind in ('KOCH', 'GOSPER')
    return lines, closed


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

    class CURVE_OT_lsystem_add(bpy.types.Operator):
        """Add an L-system fractal (Koch, dragon, Gosper, plant...) as
        a bevelled curve"""
        bl_idname = "curve.lsystem_add"
        bl_label = "L-system"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="System",
            items=[(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='KOCH')
        iters: IntProperty(
            name="Iterations", default=-1, min=-1, max=16,
            description="Rewrite depth (-1 = preset default; capped "
                        "per system to keep the string manageable)")
        radius: FloatProperty(
            name="Tube Radius", default=0.01, min=0.0, max=0.5,
            description="Bevel radius (0 = bare curve)")
        resolution: IntProperty(
            name="Bevel Resolution", default=2, min=0, max=12)
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            label = PRESETS[self.kind][0]
            lines, closed = build_lsystem(self.kind, self.iters,
                                          self.scale)
            if not lines:
                self.report({'ERROR'}, "Empty L-system")
                return {'CANCELLED'}
            cu = bpy.data.curves.new("Lsystem", 'CURVE')
            cu.dimensions = '3D'
            for ln in lines:
                sp = cu.splines.new('POLY')
                sp.points.add(len(ln) - 1)
                for i, p in enumerate(ln):
                    sp.points[i].co = (float(p[0]), float(p[1]),
                                       float(p[2]), 1.0)
                sp.use_cyclic_u = closed
            cu.bevel_depth = self.radius
            cu.bevel_resolution = self.resolution
            obj = bpy.data.objects.new(f"L-system {label}", cu)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            npts = sum(len(ln) for ln in lines)
            self.report({'INFO'},
                        f"{label}: {len(lines)} strands, {npts} points")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'iters', 'radius', 'resolution',
                      'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("curve.lsystem_add", icon='GRAPH')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_lsystem_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_lsystem_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for kind in PRESETS:
            lines, closed = build_lsystem(kind)
            npts = sum(len(ln) for ln in lines)
            allpts = np.vstack(lines) if lines else np.zeros((1, 3))
            finite = np.isfinite(allpts).all()
            ext = np.round(allpts.max(axis=0) - allpts.min(axis=0), 2)
            ok = bool(lines) and finite
            print(f"{kind:10s}: strands={len(lines)} points={npts} "
                  f"closed={closed} bbox={ext} "
                  f"{'OK' if ok else 'BAD'}")
