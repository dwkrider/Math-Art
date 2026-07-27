
# Fractal Tree Generator for Blender
#
# Parametric n-ary fractal trees, after the Mahler-Segerman "Ternary
# tree mobile" (fig. 7-2 in Segerman, "Visualizing Mathematics with
# 3D Printing"). Two modes:
#
#   TREE   -- recursive branching: a trunk that splits into `arity`
#             children at each tip, each tilted from the parent
#             direction by the branch angle and spread evenly in
#             azimuth, lengths scaled by `ratio` per generation.
#             An azimuth twist rotates each generation's spread
#             (golden-angle-ish values give non-aligned crowns).
#             Emitted as one curve object, one poly spline per branch
#             segment, per-point radius = ratio^level so a round
#             bevel (bevel_depth = trunk radius, point radii carrying
#             the relative taper) yields solid printable limbs.
#             Optional icosphere "leaves" merged at the final tips.
#
#   MOBILE -- a hanging mobile as in the figure: from each hang point
#             a vertical string drops to a horizontal `arity`-armed
#             spreader, each arm tip drops a string to the next
#             level's spreader, and the final tips hang small sphere
#             weights. Successive generations are rotated by the
#             azimuth twist (60 degrees by default in this mode, so
#             alternating spreaders do not align).
#
# Output is either a bevelled CURVE or a converted MESH. Leaf/weight
# spheres are mesh geometry, so whenever spheres are present (Mobile
# mode, or Tree with Leaf Spheres) the result is forced to MESH --
# the Output property then only matters for a sphere-free tree.
# Everything is deterministic: no random seed anywhere.
#
# Note on the twist default: the property defaults to 0 (Tree mode).
# In Mobile mode, while the property is untouched, an implicit
# default of 60 degrees is used instead; setting the slider (to any
# value, including 0) always wins.
#
# The segment budget is capped at 8000; Depth/Levels is clamped (with
# a warning) to stay under it.
#
# Run this file with plain python for a geometry self-test.

bl_info = {
    "name": "Fractal Tree",
    "author": "Math Art project (after Mahler & Segerman's "
              "ternary tree mobile)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Fractal Tree",
    "description": "Parametric n-ary fractal trees and hanging "
                   "mobiles with tapered limbs",
    "category": "Add Curve",
}

import math

import numpy as np

MAX_SEGMENTS = 8000
GOLDEN_ANGLE = math.radians(137.507764)
MOBILE_TWIST = math.radians(60.0)


# ---- geometry core (Blender-independent) -------------------------------

def tree_segment_count(arity, depth):
    """Segments in a full tree: trunk + all branch generations,
    sum_{g=0}^{depth} arity^g."""
    return (arity ** (depth + 1) - 1) // (arity - 1)


def mobile_segment_count(arity, levels):
    """Strings + spreader arms in a full mobile (weights excluded)."""
    strings = tree_segment_count(arity, levels)      # incl. final drops
    arms = strings - 1                               # arity^1 .. ^levels
    return strings + arms


def _basis(d):
    """Right-handed orthonormal (u, v) perpendicular to unit vector
    d, deterministically referenced to the world axes."""
    a = (np.array((0.0, 0.0, 1.0)) if abs(d[2]) < 0.999
         else np.array((1.0, 0.0, 0.0)))
    u = np.cross(a, d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


def build_tree(arity, depth, branch_angle, ratio, twist, trunk_len):
    """Recursive n-ary tree from the origin, trunk along +Z.

    Returns (segments, tips): segments as (p0, p1, r0, r1) tuples
    with relative radii ratio^level, tips as (position, direction)
    of the arity^depth final branch ends."""
    segs, tips = [], []
    sa, ca = math.sin(branch_angle), math.cos(branch_angle)

    def rec(p, d, length, level):
        q = p + d * length
        segs.append((tuple(map(float, p)), tuple(map(float, q)),
                     ratio ** level, ratio ** (level + 1)))
        if level >= depth:
            tips.append((tuple(map(float, q)),
                         tuple(map(float, d))))
            return
        u, v = _basis(d)
        g = level + 1
        for k in range(arity):
            az = 2.0 * math.pi * k / arity + g * twist
            cd = sa * (math.cos(az) * u + math.sin(az) * v) + ca * d
            cd /= np.linalg.norm(cd)
            rec(q, cd, length * ratio, g)

    rec(np.zeros(3), np.array((0.0, 0.0, 1.0)), float(trunk_len), 0)
    return segs, tips


def build_mobile(arity, levels, twist, trunk_len, ratio,
                 string_r=0.3, bar_r=1.0):
    """Hanging mobile from the origin, growing downward.

    Each hang point drops a vertical string of length trunk_len *
    ratio^g to a horizontal arity-armed spreader (arm length the
    same), generation g's arms rotated by g * twist in azimuth; the
    final tips drop a half-length string ending at a weight hang
    point.

    Returns (segments, weights): segments as (p0, p1, r0, r1) with
    relative radii, weights as (hang_point, spreader_z) -- the weight
    sphere is to be centred just below hang_point, and spreader_z is
    the height of the bar it hangs from (always above)."""
    segs, weights = [], []

    def rec(p, g):
        s = ratio ** g
        if g == levels:
            end = (p[0], p[1], p[2] - 0.5 * trunk_len * s)
            segs.append((p, end, string_r * s, string_r * s))
            weights.append((end, p[2]))
            return
        c = (p[0], p[1], p[2] - trunk_len * s)
        segs.append((p, c, string_r * s, string_r * s))
        arm = trunk_len * s
        for k in range(arity):
            az = 2.0 * math.pi * k / arity + g * twist
            tip = (c[0] + arm * math.cos(az),
                   c[1] + arm * math.sin(az), c[2])
            segs.append((c, tip, bar_r * s, 0.7 * bar_r * s))
            rec(tip, g + 1)

    rec((0.0, 0.0, 0.0), 0)
    return segs, weights


def icosphere(subdiv=1):
    """Unit icosphere (verts, tri faces); subdiv=1 gives 42/80."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10),
             (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
             (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [list(v) for v in verts]
    for _ in range(subdiv):
        mid = {}

        def midpoint(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in mid:
                va, vb = verts[a], verts[b]
                verts.append([(va[i] + vb[i]) / 2.0
                              for i in range(3)])
                mid[key] = len(verts) - 1
            return mid[key]

        nf = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), \
                midpoint(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc),
                   (ab, bc, ca)]
        faces = nf
    out = []
    for v in verts:
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        out.append((v[0] / n, v[1] / n, v[2] / n))
    return out, faces


_ICO = None


def _ico():
    global _ICO
    if _ICO is None:
        _ICO = icosphere(1)
    return _ICO


# ---- Blender layer -----------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class CURVE_OT_fractal_tree_add(bpy.types.Operator):
        """Add an n-ary fractal tree or hanging mobile (after the
        Mahler-Segerman ternary tree mobile)"""
        bl_idname = "curve.fractal_tree_add"
        bl_label = "Fractal Tree"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            items=[('TREE', "Tree",
                    "recursive branching tree with tapering limbs"),
                   ('MOBILE', "Mobile",
                    "hanging mobile: strings, spreader bars and "
                    "sphere weights")],
            default='TREE')
        arity: IntProperty(
            name="Arity", default=3, min=2, max=5,
            description="Branches (or spreader arms) per node")
        depth: IntProperty(
            name="Depth", default=5, min=1, max=12,
            description="Branching generations (Tree mode); clamped "
                        "to keep the total under 8000 segments")
        levels: IntProperty(
            name="Levels", default=4, min=1, max=8,
            description="Spreader levels (Mobile mode); clamped to "
                        "keep the total under 8000 segments")
        branch_angle: FloatProperty(
            name="Branch Angle", subtype='ANGLE',
            default=math.radians(35.0), min=0.0,
            max=math.radians(90.0),
            description="Tilt of each child branch away from its "
                        "parent's direction")
        ratio: FloatProperty(
            name="Ratio", default=0.67, min=0.2, max=0.95,
            description="Length and radius scale factor per "
                        "generation")
        azimuth_twist: FloatProperty(
            name="Azimuth Twist", subtype='ANGLE', default=0.0,
            min=-math.pi, max=math.pi,
            description="Extra rotation of each generation's azimuth "
                        "spread (try the golden angle 137.5 for "
                        "non-aligned crowns). Mobile mode uses 60 "
                        "degrees while this is left untouched")
        trunk_length: FloatProperty(
            name="Trunk Length", default=2.0, min=0.01, max=100.0,
            description="Length of the trunk (Tree) or of the "
                        "level-0 string and arms (Mobile)")
        trunk_radius: FloatProperty(
            name="Trunk Radius", default=0.06, min=0.0, max=10.0,
            step=1, precision=3,
            description="Bevel radius at the trunk; limbs taper by "
                        "Ratio per generation (0 = wire)")
        leaf_spheres: BoolProperty(
            name="Leaf Spheres", default=False,
            description="Merge a small icosphere at each final tip "
                        "(Tree mode; forces mesh output)")
        weight_size: FloatProperty(
            name="Sphere Size", default=0.12, min=0.001, max=10.0,
            step=1, precision=3,
            description="Radius of the leaf spheres / mobile weights")
        output: EnumProperty(
            name="Output",
            items=[('MESH', "Mesh",
                    "convert the bevelled curve to a mesh (required "
                    "and forced whenever spheres are present: Mobile "
                    "mode, or Tree with Leaf Spheres)"),
                   ('CURVE', "Curve",
                    "keep a curve object with bevel (only honoured "
                    "when no spheres are requested)")],
            default='MESH')

        def execute(self, context):
            arity = self.arity
            twist = self.azimuth_twist
            if (self.mode == 'MOBILE'
                    and not self.properties.is_property_set(
                        "azimuth_twist")):
                twist = MOBILE_TWIST
            spheres = []          # (center, radius)
            if self.mode == 'TREE':
                depth = self.depth
                while depth > 1 and \
                        tree_segment_count(arity, depth) \
                        > MAX_SEGMENTS:
                    depth -= 1
                if depth < self.depth:
                    self.report(
                        {'WARNING'},
                        f"depth clamped {self.depth} -> {depth} "
                        f"({MAX_SEGMENTS}-segment cap)")
                segs, tips = build_tree(
                    arity, depth, self.branch_angle, self.ratio,
                    twist, self.trunk_length)
                if self.leaf_spheres:
                    spheres = [(p, self.weight_size)
                               for p, _d in tips]
                name = "Fractal Tree"
            else:
                levels = self.levels
                while levels > 1 and \
                        mobile_segment_count(arity, levels) \
                        > MAX_SEGMENTS:
                    levels -= 1
                if levels < self.levels:
                    self.report(
                        {'WARNING'},
                        f"levels clamped {self.levels} -> {levels} "
                        f"({MAX_SEGMENTS}-segment cap)")
                segs, weights = build_mobile(
                    arity, levels, twist, self.trunk_length,
                    self.ratio)
                spheres = [((x, y, z - self.weight_size),
                            self.weight_size)
                           for (x, y, z), _bar_z in weights]
                name = "Fractal Mobile"

            # Center on the origin and fit within a 2 m cube by
            # default: bbox over every segment endpoint (and sphere
            # extent), then translate the midpoint to the origin and
            # uniformly scale the largest extent to 2.0. fit_scale is
            # applied to the bevel too so limb thickness stays in
            # proportion; the object itself keeps an identity transform.
            fit_scale = 1.0
            pts = [p for p0, p1, _r0, _r1 in segs for p in (p0, p1)]
            for (cx, cy, cz), r in spheres:
                pts.append((cx - r, cy - r, cz - r))
                pts.append((cx + r, cy + r, cz + r))
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]
                cen = ((min(xs) + max(xs)) / 2.0,
                       (min(ys) + max(ys)) / 2.0,
                       (min(zs) + max(zs)) / 2.0)
                ext = max(max(xs) - min(xs), max(ys) - min(ys),
                          max(zs) - min(zs))
                fit_scale = 2.0 / ext if ext > 1e-9 else 1.0
                segs = [(((p0[0] - cen[0]) * fit_scale,
                          (p0[1] - cen[1]) * fit_scale,
                          (p0[2] - cen[2]) * fit_scale),
                         ((p1[0] - cen[0]) * fit_scale,
                          (p1[1] - cen[1]) * fit_scale,
                          (p1[2] - cen[2]) * fit_scale),
                         r0, r1)
                        for p0, p1, r0, r1 in segs]
                spheres = [(((cx - cen[0]) * fit_scale,
                             (cy - cen[1]) * fit_scale,
                             (cz - cen[2]) * fit_scale),
                            r * fit_scale)
                           for (cx, cy, cz), r in spheres]

            cu = bpy.data.curves.new(name, 'CURVE')
            cu.dimensions = '3D'
            for p0, p1, r0, r1 in segs:
                sp = cu.splines.new('POLY')
                sp.points.add(1)
                sp.points[0].co = (p0[0], p0[1], p0[2], 1.0)
                sp.points[1].co = (p1[0], p1[1], p1[2], 1.0)
                sp.points[0].radius = r0
                sp.points[1].radius = r1
            cu.bevel_depth = self.trunk_radius * fit_scale
            cu.bevel_resolution = 4
            if self.trunk_radius > 0:
                cu.use_fill_caps = True
            obj = bpy.data.objects.new(name, cu)
            context.collection.objects.link(obj)

            need_mesh = bool(spheres) or self.output == 'MESH'
            if spheres and self.output == 'CURVE':
                self.report({'INFO'},
                            "spheres need mesh output -- converting "
                            "curve to mesh")
            if need_mesh:
                context.view_layer.update()
                dg = context.evaluated_depsgraph_get()
                tmp = bpy.data.meshes.new_from_object(
                    obj.evaluated_get(dg), depsgraph=dg)
                verts = [v.co[:] for v in tmp.vertices]
                faces = [tuple(p.vertices) for p in tmp.polygons]
                edges = ([] if faces else
                         [tuple(e.vertices) for e in tmp.edges])
                bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.curves.remove(cu)
                bpy.data.meshes.remove(tmp)
                iv, ifc = _ico()
                for (cx, cy, cz), r in spheres:
                    base = len(verts)
                    verts.extend((cx + r * x, cy + r * y,
                                  cz + r * z) for x, y, z in iv)
                    faces.extend((a + base, b + base, c + base)
                                 for a, b, c in ifc)
                me = bpy.data.meshes.new(name)
                me.from_pydata(verts, edges, faces)
                me.validate()
                obj = bpy.data.objects.new(name, me)
                context.collection.objects.link(obj)

            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: {len(segs)} segments, "
                        f"{len(spheres)} spheres")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            lay.prop(self, 'arity')
            if self.mode == 'TREE':
                lay.prop(self, 'depth')
                lay.prop(self, 'branch_angle')
            else:
                lay.prop(self, 'levels')
            lay.prop(self, 'ratio')
            lay.prop(self, 'azimuth_twist')
            lay.prop(self, 'trunk_length')
            lay.prop(self, 'trunk_radius')
            if self.mode == 'TREE':
                lay.prop(self, 'leaf_spheres')
                if self.leaf_spheres:
                    lay.prop(self, 'weight_size')
            else:
                lay.prop(self, 'weight_size')
            lay.prop(self, 'output')

    def _menu_func(self, context):
        self.layout.operator("curve.fractal_tree_add",
                             icon='GRAPH')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_fractal_tree_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_fractal_tree_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for arity, depth in ((2, 3), (3, 4), (5, 3)):
            segs, tips = build_tree(arity, depth,
                                    math.radians(35.0), 0.67,
                                    GOLDEN_ANGLE, 2.0)
            assert len(segs) == tree_segment_count(arity, depth)
            assert len(tips) == arity ** depth
            starts = {tuple(round(c, 6) for c in s[0])
                      for s in segs}
            free = [s for s in segs
                    if tuple(round(c, 6) for c in s[1])
                    not in starts]
            assert len(free) == arity ** depth, \
                f"tip endpoints {len(free)} != {arity ** depth}"
            # radius continuity: child start == parent end
            r_at = {tuple(round(c, 6) for c in s[1]): s[3]
                    for s in segs}
            for s in segs:
                key = tuple(round(c, 6) for c in s[0])
                if key in r_at:
                    assert abs(r_at[key] - s[2]) < 1e-9
            print(f"tree  arity={arity} depth={depth}: "
                  f"{len(segs)} segs, {len(tips)} tips OK")
        for arity, levels in ((3, 3), (2, 4)):
            segs, weights = build_mobile(arity, levels,
                                         MOBILE_TWIST, 2.0, 0.67)
            assert len(segs) == mobile_segment_count(arity, levels)
            assert len(weights) == arity ** levels
            assert all(p[2] < bar_z for p, bar_z in weights)
            print(f"mobile arity={arity} levels={levels}: "
                  f"{len(segs)} segs, {len(weights)} weights "
                  f"all below their bars OK")
        v, f = icosphere(1)
        assert len(v) == 42 and len(f) == 80
        print("fractal_tree_generator self-test: ALL OK")
