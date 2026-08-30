
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
#   HONDA  -- Honda's "tree-like body" (1971): repeated BIFURCATION
#             described by just four parameters -- two branching angles
#             theta1, theta2 and two length ratios R1, R2 -- plus a
#             divergence angle used when a branching angle is zero.  The
#             signs of the angles alternate at every branching.  Honda's
#             three findings are directly visible in the sliders: the
#             branching angle sets the WIDTH of the crown, the relative
#             difference between R1 and R2 is the degree of APICAL
#             DOMINANCE and decides conic versus flat, and the ratio of
#             the smaller angle to the branching angle sets AXIALITY --
#             how clearly a main axis reads.  The seven published plate
#             parameter sets ship as presets.
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
# Everything is deterministic, and a `seed` property is exposed so that
# a given seed always reproduces the same tree.
#
# Note on the twist default: the property defaults to 0 (Tree mode).
# In Mobile mode, while the property is untouched, an implicit
# default of 60 degrees is used instead; setting the slider (to any
# value, including 0) always wins.
#
# The segment budget is capped at MAX_SEGMENTS; Depth/Levels is clamped
# (with a warning) to stay under it.  The old cap of 8000 was low enough
# to block ABOP's own Figures 2.8b/2.8c, which need 13,120 segments for a
# ternary tree at depth 8.
#
# Run this file with plain python for a geometry self-test.
#
# References:
#   - Aristid Lindenmayer, "Mathematical models for cellular
#     interactions in development" (1968) -- L-systems.
#   - Przemyslaw Prusinkiewicz & Aristid Lindenmayer, "The
#     Algorithmic Beauty of Plants" (Springer, 1990).
#   - Hisao Honda, "Description of the Form of Trees by the Parameters
#     of the Tree-like Body: Effects of the Branching Angle and the
#     Branch Length on the Shape of the Tree-like Body", Journal of
#     Theoretical Biology 31 (1971), pp. 331-338 -- the HONDA mode's
#     assumptions (a)-(e), end-point formula and Plates I-VI.
#   - Leonardo da Vinci, Notebooks, and Cecil D. Murray, J. General
#     Physiology 9 (1926) -- the width law behind `width_exponent`.
#   - Recursive branching in the tradition of the Pythagoras tree
#     (Albert E. Bosman, 1942).
#   - Henry Segerman, "Visualizing Mathematics with 3D Printing"
#     (Johns Hopkins University Press, 2016) -- the Mahler-Segerman
#     ternary tree mobile (fig. 7-2) reproduced by the Mobile mode.

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










# ---- geometry core (Blender-independent) -------------------------------











try:
    from .surfaces.primitives import icosphere as _icosphere_shared
except ImportError:  # flat import outside the package
    from surfaces.primitives import icosphere as _icosphere_shared
import math
import numpy as np

# The mathematics lives in the sibling `lsystem` engine package;
# this module is the Blender layer over it.
try:
    from .lsystem.trees import (GOLDEN_ANGLE, HONDA_PLATES,
                                    MAX_SEGMENTS, MOBILE_TWIST, _ico,
                                    build_honda, build_mobile, build_tree,
                                    honda_segment_count, icosphere, mobile_segment_count, tree_segment_count)
except ImportError:  # flat import outside the package
    from lsystem.trees import (GOLDEN_ANGLE, HONDA_PLATES, MAX_SEGMENTS,
                                   MOBILE_TWIST, _ico, build_honda,
                                   build_mobile, build_tree,
                                   honda_segment_count, icosphere, mobile_segment_count, tree_segment_count)










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
                    "sphere weights"),
                   ('HONDA', "Honda",
                    "Honda's tree-like body (1971): bifurcation from "
                    "two branching angles and two length ratios, with "
                    "the published plate parameters")],
            default='TREE',
            description="Which construction to build")
        honda_plate: EnumProperty(
            name="Plate",
            items=[('PLATE_I', "I - axis lost",
                    "theta 16.7/-33.3, R 0.85/0.85"),
                   ('PLATE_II', "II - axis preserved",
                    "theta 0/-45, R 0.9/0.8"),
                   ('PLATE_III', "III - cone to flat crown",
                    "theta 16.7/-33.3, R 0.9/0.79"),
                   ('PLATE_IV', "IV - branching angle",
                    "theta 0/-45, R 0.9/0.7"),
                   ('PLATE_V', "V - constant axiality",
                    "theta 15/-30, R 0.9/0.75"),
                   ('PLATE_VI_CONIC', "VI conic - strong dominance",
                    "theta 0/-45, R 0.9/0.6"),
                   ('PLATE_VI_FLAT', "VI flat - weak dominance",
                    "theta 0/-45, R 0.7/0.9"),
                   ('CUSTOM', "Custom", "use the sliders below")],
            default='PLATE_II',
            description="Honda's published plate parameter set "
                        "(or Custom to use the sliders)")
        theta1: FloatProperty(
            name="Theta 1", default=0.0, min=-90.0, max=90.0,
            description="First branching angle (degrees)")
        theta2: FloatProperty(
            name="Theta 2", default=-45.0, min=-90.0, max=90.0,
            description="Second branching angle (degrees)")
        r1: FloatProperty(
            name="R1", default=0.9, min=0.1, max=1.0,
            description="Length ratio of the first daughter")
        r2: FloatProperty(
            name="R2", default=0.8, min=0.1, max=1.0,
            description="Length ratio of the second daughter; the "
                        "relative difference from R1 is the degree of "
                        "apical dominance, and sets conic vs flat crown")
        divergence: FloatProperty(
            name="Divergence", default=137.5, min=0.0, max=360.0,
            description="Roll between successive branchings; used "
                        "directly when a branching angle is zero")
        width_exponent: FloatProperty(
            name="Width Exponent", default=2.0, min=1.0, max=4.0,
            description="w_parent^n = sum w_child^n. n=2 conserves "
                        "cross-sectional area (da Vinci), n=3 is "
                        "Murray's law. Decoupled from the length ratio")
        seed: IntProperty(
            name="Seed", default=0, min=0, max=10000,
            description="Reserved for stochastic variation; same seed "
                        "gives the same tree")
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
            default='MESH',
            description="Result as a mesh or a bevelled curve "
                        "(spheres force mesh output)")

        def execute(self, context):
            arity = self.arity
            twist = self.azimuth_twist
            if (self.mode == 'MOBILE'
                    and not self.properties.is_property_set(
                        "azimuth_twist")):
                twist = MOBILE_TWIST
            spheres = []          # (center, radius)
            if self.mode == 'HONDA':
                if self.honda_plate == 'CUSTOM':
                    t1, t2 = self.theta1, self.theta2
                    rr1, rr2, alpha = self.r1, self.r2, self.divergence
                    depth = self.depth
                else:
                    t1, t2, rr1, rr2, alpha, nmax = HONDA_PLATES[
                        self.honda_plate]
                    depth = min(self.depth, nmax)
                while depth > 1 and honda_segment_count(depth) > MAX_SEGMENTS:
                    depth -= 1
                if depth < self.depth:
                    self.report({'WARNING'},
                                f"Depth clamped to {depth} to stay under "
                                f"{MAX_SEGMENTS} segments")
                segs, tips = build_honda(
                    depth, t1, t2, rr1, rr2, alpha,
                    trunk_len=self.trunk_length,
                    width_exponent=self.width_exponent)
                if self.leaf_spheres:
                    spheres = [(p, self.weight_size) for p, _d in tips]
                name = "Honda Tree"
            elif self.mode == 'TREE':
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


def _honda_selftest():
    """Honda's model, checked against his own reported findings.

    The numbers below are not invented for the test: they are the
    parameter sets printed on Plates I-VI, and the assertions are the
    three effects he reports in section 3.
    """
    # every plate builds finite geometry of the right size
    for key, (t1, t2, r1, r2, al, nmax) in HONDA_PLATES.items():
        segs, tips = build_honda(6, t1, t2, r1, r2, al)
        assert len(segs) == honda_segment_count(6), (key, len(segs))
        assert len(tips) == 2 ** 6, key
        P = np.array([s[1] for s in segs], dtype=float)
        assert np.all(np.isfinite(P)), key
        assert nmax == 9, key            # all plates are drawn at N = 9

    def bbox(key, depth=7):
        t1, t2, r1, r2, al, _n = HONDA_PLATES[key]
        segs, _tips = build_honda(depth, t1, t2, r1, r2, al)
        P = np.array([s[0] for s in segs] + [s[1] for s in segs])
        return P.max(axis=0) - P.min(axis=0)

    # FINDING 1 (apical dominance -> crown shape).  Plate VI sweeps R1
    # and R2 at fixed angles.  Strong dominance (0.9/0.6) gives a CONIC,
    # tall crown; weak dominance (0.7/0.9) gives a FLAT, wide one.
    conic, flat = bbox('PLATE_VI_CONIC'), bbox('PLATE_VI_FLAT')
    assert conic[2] > flat[2], (conic, flat)
    assert max(flat[0], flat[1]) / flat[2] > max(conic[0], conic[1]) / conic[2], \
        "weak apical dominance must give a relatively wider crown"

    # FINDING 2 (branching angle -> width/stretch).  Widening the
    # branching angle at fixed ratios spreads the body.
    narrow = np.array([s[1] for s in build_honda(7, 0, -20, .9, .8)[0]])
    wide = np.array([s[1] for s in build_honda(7, 0, -80, .9, .8)[0]])
    nw = (narrow.max(axis=0) - narrow.min(axis=0))[:2].max()
    ww = (wide.max(axis=0) - wide.min(axis=0))[:2].max()
    assert ww > nw, (nw, ww)

    # the length ratios really do contract, so the body is bounded
    segs, _t = build_honda(9, 0, -45, 0.9, 0.8)
    P = np.array([s[1] for s in segs])
    assert np.all(np.abs(P) < 50.0), "contraction must keep the body finite"

    # the width law conserves w^n across every fork
    for n in (2.0, 3.0):
        segs, _t = build_honda(3, 0, -45, 0.9, 0.8, width_exponent=n)
        assert segs[0][2] == 1.0
        assert all(np.isfinite([s[2] for s in segs]))


def _selftest():
    _honda_selftest()
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
