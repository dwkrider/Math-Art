
# Kleinian Limit Set Generator for Blender
#
# Two Moebius transformations generate a group of the Riemann sphere; the
# closure of an orbit is the LIMIT SET, the fractal the dynamics accumulate on.
# Grandma's recipe builds the pair from their traces, so the whole family of
# once-punctured-torus groups is parameterised by two complex numbers, and the
# picture runs continuously from a wild quasifuchsian Jordan curve to a gasket
# of tangent circles as those traces move.
#
# The Apollonian gasket this add-on already generates is one very special member
# of the family; this is the general case.
#
# Drawn by DEPTH-FIRST SEARCH, not random iteration.  A random walk visits parts
# of the limit set exponentially rarely, leaving holes that never fill; the
# search of Indra's Pearls enumerates reduced words systematically and produces
# the set as a continuous curve, terminating a branch once its image has shrunk
# below a tolerance.  Two details decide whether the output is a curve or a
# spray of dots: the generators are held in the cyclic order a, B, A, b, and
# termination is geometric rather than by word length.
#
# Three outputs, all distinct from anything else in the add-on:
#   CURVE   the limit set as a polyline, ready to bevel
#   ORBIT   the group's images of a seed circle, as rings
#   SLICE   the boundary of the Maskit slice -- the deformation space itself,
#           traced by locating the cusp at every rational p/q with Newton on
#           the trace polynomial
#
# The engine is in `math_art/kleinian/` and is Blender-free.
#
# References:
# - David Mumford, Caroline Series and David Wright, "Indra's Pearls: The Vision
#   of Felix Klein", Cambridge University Press, 2002 (Grandma's recipe in
#   Box 21; the cyclic order in Box 15; the depth-first search in Box 17; the
#   trace recursion and boundary tracing in Boxes 24-28).
# - Bernard Maskit, "Kleinian Groups", Springer Grundlehren 287, 1988.
# - Alan F. Beardon, "The Geometry of Discrete Groups", Springer GTM 91, 1983.
# - David J. Wright, "Searching for the cusp", in "Spaces of Kleinian Groups",
#   London Mathematical Society Lecture Note Series 329, Cambridge University
#   Press, 2006, pp. 301-336.

bl_info = {
    "name": "Kleinian Limit Set",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Kleinian Limit Set",
    "description": "Limit sets of Kleinian groups, and the boundary of the "
                   "Maskit slice",
    "category": "Add Curve",
}

import math

from . import kleinian
from .kleinian import (GRANDMA, MASKIT, PRESETS, boundary, circle_orbit,
                       generators, limit_set)

_PALETTE = [
    (0.90, 0.91, 0.94, 1.0), (0.16, 0.22, 0.61, 1.0),
    (0.29, 0.38, 0.78, 1.0), (0.46, 0.56, 0.90, 1.0),
    (0.60, 0.42, 0.09, 1.0), (0.84, 0.64, 0.29, 1.0),
    (0.35, 0.40, 0.47, 1.0), (0.12, 0.14, 0.18, 1.0),
]
_NPAL = len(_PALETTE)


def _ring(cx, cy, r, tube, seg, tseg):
    verts, faces = [], []
    for i in range(seg):
        a = 2.0 * math.pi * i / seg
        ca, sa = math.cos(a), math.sin(a)
        for j in range(tseg):
            b = 2.0 * math.pi * j / tseg
            rr = r + tube * math.cos(b)
            verts.append((cx + rr * ca, cy + rr * sa, tube * math.sin(b)))
    for i in range(seg):
        for j in range(tseg):
            a0 = i * tseg + j
            b0 = i * tseg + (j + 1) % tseg
            c0 = ((i + 1) % seg) * tseg + (j + 1) % tseg
            d0 = ((i + 1) % seg) * tseg + j
            faces.append((a0, b0, c0, d0))
    return verts, faces


def build_kleinian(mode='CURVE', family=GRANDMA, preset='QUASIFUCHSIAN',
                   ta_re=1.91, ta_im=0.05, tb_re=3.0, tb_im=0.0,
                   mu_re=0.0, mu_im=2.0, epsilon=0.005, max_depth=22,
                   max_points=200000, orbit_depth=4, seed_radius=0.3,
                   tube_ratio=0.06, ring_seg=16, tube_seg=6, denom=24,
                   scale=1.0, color_by='RADIUS'):
    """Return (verts, edges, faces, mats, report).

    CURVE returns edges and no faces; ORBIT and SLICE return faces."""
    if preset != 'CUSTOM' and family == GRANDMA:
        ta, tb = PRESETS[preset]
    else:
        ta = complex(ta_re, ta_im)
        tb = complex(tb_re, tb_im)
    mu = complex(mu_re, mu_im)

    if mode == 'SLICE':
        cusps = boundary(denom=max(2, denom))
        verts = [(m.real * scale, m.imag * scale, 0.0) for (_, _, m) in cusps]
        edges = [(i, i + 1) for i in range(len(verts) - 1)]
        report = "Maskit slice: %d cusps to denominator %d" % (len(cusps),
                                                               denom)
        return verts, edges, [], [], report

    gens = generators(family, ta, tb, mu)

    if mode == 'ORBIT':
        circles = circle_orbit(gens, [(complex(0.0, 0.0), seed_radius)],
                               depth=orbit_depth)
        verts, faces, mats = [], [], []
        rr = [r for (_, r) in circles if r > 0]
        lo = math.log(min(rr)) if rr else 0.0
        hi = math.log(max(rr)) if rr else 1.0
        for (c, r) in circles:
            if r <= 1e-6 or r > 50.0 or abs(c) > 50.0:
                continue
            if color_by == 'UNIFORM':
                mi = 1
            else:
                t = 0.0 if hi <= lo else (math.log(r) - lo) / (hi - lo)
                mi = 1 + int(min(0.999, max(0.0, t)) * (_NPAL - 1))
            rv, rf = _ring(c.real * scale, c.imag * scale, r * scale,
                           max(1e-5, r * tube_ratio * scale), ring_seg,
                           tube_seg)
            base = len(verts)
            verts.extend(rv)
            for f in rf:
                faces.append(tuple(base + i for i in f))
                mats.append(mi)
        return verts, [], faces, mats, ("orbit: %d circles to depth %d"
                                        % (len(circles), orbit_depth))

    pts = limit_set(gens, epsilon=epsilon, max_depth=max_depth,
                    max_points=max_points)
    verts = []
    edges = []
    prev = None
    for z in pts:
        if not (abs(z.real) < 1e6 and abs(z.imag) < 1e6):
            prev = None
            continue
        verts.append((z.real * scale, z.imag * scale, 0.0))
        i = len(verts) - 1
        if prev is not None and abs(z - prev) < 0.25:
            edges.append((i - 1, i))
        prev = z
    report = ("limit set: %d points, %d segments, epsilon %.4g"
              % (len(verts), len(edges), epsilon))
    return verts, edges, [], [], report


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _kleinian_mesh(name, verts, edges, faces, mats, smooth):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in verts],
                       [tuple(int(i) for i in e) for e in edges],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        for idx, rgba in enumerate(_PALETTE):
            nm = "Kleinian_%d" % idx
            m = bpy.data.materials.get(nm)
            if m is None:
                m = bpy.data.materials.new(nm)
                m.diffuse_color = rgba
            me.materials.append(m)
        if mats and len(mats) == len(me.polygons):
            me.polygons.foreach_set('material_index', mats)
        if smooth and len(me.polygons):
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        return me

    class CURVE_OT_kleinian_add(bpy.types.Operator):
        """Add the limit set of a Kleinian group, or the boundary of the
        Maskit slice"""
        bl_idname = "curve.kleinian_add"
        bl_label = "Kleinian Limit Set"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Output",
            items=[('CURVE', "Limit Set",
                    "The limit set as a polyline, ready to bevel"),
                   ('ORBIT', "Circle Orbit",
                    "The group's images of a seed circle, as rings"),
                   ('SLICE', "Maskit Slice",
                    "The boundary of the deformation space, traced through "
                    "its cusps")],
            default='CURVE')
        family: EnumProperty(
            name="Family",
            items=[(GRANDMA, "Grandma's Recipe",
                    "Two generators built from their traces; the commutator "
                    "is parabolic, giving a once-punctured-torus group"),
                   (MASKIT, "Maskit",
                    "The one-parameter slice with b parabolic")],
            default=GRANDMA)
        preset: EnumProperty(
            name="Preset",
            items=[('QUASIFUCHSIAN', "Quasifuchsian",
                    "A wild fractal Jordan curve"),
                   ('GASKET', "Gasket",
                    "Every generator parabolic; the limit set closes into "
                    "tangent circles"),
                   ('SPIRAL', "Spiral", "Deep spiralling whorls"),
                   ('NEAR_FUCHSIAN', "Near Circle",
                    "Close to a round circle"),
                   ('CUSTOM', "Custom", "Set the traces by hand")],
            default='QUASIFUCHSIAN')
        ta_re: FloatProperty(name="Trace A (real)", default=1.91,
                             min=-6.0, max=6.0)
        ta_im: FloatProperty(name="Trace A (imag)", default=0.05,
                             min=-6.0, max=6.0)
        tb_re: FloatProperty(name="Trace B (real)", default=3.0,
                             min=-6.0, max=6.0)
        tb_im: FloatProperty(name="Trace B (imag)", default=0.0,
                             min=-6.0, max=6.0)
        mu_re: FloatProperty(name="Mu (real)", default=0.0, min=-6.0, max=6.0)
        mu_im: FloatProperty(name="Mu (imag)", default=2.0, min=1.01, max=8.0,
                             description="Groups in the Maskit slice need this "
                                         "above 1")
        epsilon: FloatProperty(
            name="Detail", default=0.005, min=0.0002, max=0.2,
            description="A branch stops once its image is smaller than this; "
                        "lower resolves more of the set and costs more")
        max_depth: IntProperty(name="Word Depth", default=22, min=4, max=30)
        max_points: IntProperty(
            name="Point Budget", default=200000, min=1000, max=1000000,
            description="Hard cap on the number of points generated")
        orbit_depth: IntProperty(name="Orbit Depth", default=4, min=1, max=7)
        seed_radius: FloatProperty(name="Seed Circle", default=0.3,
                                   min=0.02, max=2.0)
        denom: IntProperty(
            name="Cusp Detail", default=24, min=2, max=200,
            description="Largest denominator p/q used when tracing the slice "
                        "boundary")
        tube_ratio: FloatProperty(name="Ring Thickness", default=0.06,
                                  min=0.005, max=0.4)
        ring_seg: IntProperty(name="Ring Segments", default=16, min=6, max=64)
        tube_seg: IntProperty(name="Tube Segments", default=6, min=3, max=24)
        color_by: EnumProperty(
            name="Material By",
            items=[('RADIUS', "Circle Size", "Shade rings by radius"),
                   ('UNIFORM', "Uniform", "One material")],
            default='RADIUS')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        smooth: BoolProperty(name="Shade Smooth", default=True)

        def execute(self, context):
            try:
                verts, edges, faces, mats, report = build_kleinian(
                    self.mode, self.family, self.preset,
                    self.ta_re, self.ta_im, self.tb_re, self.tb_im,
                    self.mu_re, self.mu_im, self.epsilon, self.max_depth,
                    self.max_points, self.orbit_depth, self.seed_radius,
                    self.tube_ratio, self.ring_seg, self.tube_seg,
                    self.denom, self.scale, self.color_by)
            except (ValueError, ZeroDivisionError) as exc:
                self.report({'ERROR'}, "%s" % exc)
                return {'CANCELLED'}
            if not verts:
                self.report({'ERROR'}, "no geometry: try a coarser Detail")
                return {'CANCELLED'}
            me = _kleinian_mesh("Kleinian", verts, edges, faces, mats,
                                self.smooth)
            obj = bpy.data.objects.new("Kleinian", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, report)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'SLICE':
                lay.prop(self, 'denom')
            else:
                lay.prop(self, 'family')
                if self.family == GRANDMA:
                    lay.prop(self, 'preset')
                    if self.preset == 'CUSTOM':
                        lay.prop(self, 'ta_re')
                        lay.prop(self, 'ta_im')
                        lay.prop(self, 'tb_re')
                        lay.prop(self, 'tb_im')
                else:
                    lay.prop(self, 'mu_re')
                    lay.prop(self, 'mu_im')
                if self.mode == 'CURVE':
                    lay.prop(self, 'epsilon')
                    lay.prop(self, 'max_depth')
                    lay.prop(self, 'max_points')
                else:
                    lay.prop(self, 'orbit_depth')
                    lay.prop(self, 'seed_radius')
                    lay.prop(self, 'tube_ratio')
                    lay.prop(self, 'ring_seg')
                    lay.prop(self, 'tube_seg')
                    lay.prop(self, 'color_by')
            lay.prop(self, 'scale')
            lay.prop(self, 'smooth')

    def _menu_func(self, context):
        self.layout.operator("curve.kleinian_add", icon='FORCE_TURBULENCE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_kleinian_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_kleinian_add)


def _selftest():
    """Build every mode headlessly."""
    v, e, f, m, rep = build_kleinian('CURVE', preset='GASKET', epsilon=0.02,
                                     max_depth=18)
    assert len(v) > 500 and len(e) > 400, rep
    assert max(math.hypot(p[0], p[1]) for p in v) < 1.001, \
        "the gasket limit set must stay in the unit disc"

    v2, e2, f2, m2, rep2 = build_kleinian('CURVE', preset='QUASIFUCHSIAN',
                                          epsilon=0.02, max_depth=18)
    assert len(v2) > 500 and len(e2) > 400, rep2

    v3, e3, f3, m3, rep3 = build_kleinian('ORBIT', preset='GASKET',
                                          orbit_depth=3)
    assert len(f3) > 0 and len(m3) == len(f3), rep3

    v4, e4, f4, m4, rep4 = build_kleinian('SLICE', denom=12)
    assert len(v4) > 5 and len(e4) == len(v4) - 1, rep4
    assert all(p[1] > 1.0 - 1e-9 for p in v4), \
        "every Maskit cusp must have Im(mu) > 1"

    v5, e5, f5, m5, _ = build_kleinian('CURVE', preset='CUSTOM',
                                       ta_re=1.87, ta_im=0.1,
                                       tb_re=1.87, tb_im=-0.1,
                                       epsilon=0.03, max_depth=16)
    assert len(v5) > 100

    v6, e6, f6, m6, _ = build_kleinian('CURVE', preset='GASKET',
                                       epsilon=0.001, max_depth=26,
                                       max_points=4000)
    assert len(v6) <= 4000, "the point budget must be honoured"

    print("kleinian_generator: limit set, circle orbit and Maskit slice all "
          "build; gasket inside the unit disc, cusps above Im = 1, budget "
          "honoured. RESULT: OK")
