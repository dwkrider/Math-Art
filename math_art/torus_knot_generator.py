
# Torus Knot generator for Blender: the (p, q) torus knots and
# links as rope curves, with the same output styles as the Prime
# Knots generator (bezier / poly / NURBS curve or swept tube
# mesh).
#
# The curve winds p times around the torus axis while looping q
# times through the hole, on a torus of configurable major/minor
# radius.  When gcd(p, q) = d > 1 the result is a torus LINK of d
# components (e.g. (2,2) is the Hopf link, (2,4) Solomon's link,
# (3,3) three fibres of the Hopf fibration), emitted as d splines
# (or merged tube meshes) with optional per-component coloring.
#
# References:
# - Torus knots and links T(p,q): a classical construction of knot
#   theory (see e.g. Dale Rolfsen, "Knots and Links", 1976).
# - The Hopf fibration of S^3, whose fibres are (1,1) torus circles:
#   Heinz Hopf, "Ueber die Abbildungen der dreidimensionalen Sphaere
#   auf die Kugelflaeche", Math. Ann. 104, 1931.

bl_info = {
    "name": "Torus Knots & Links",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Torus Knot",
    "description": "(p, q) torus knots and links as curves or "
                   "tube meshes",
    "category": "Add Curve",
}

import math
from math import gcd, pi

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def torus_link_components(p, q, samples, major=0.7, minor=0.3):
    """List of closed polylines, one per link component.  The
    (p, q) torus link has d = gcd(p, q) components, each a reduced
    (p/d, q/d) torus knot; on the flat torus they are parallel
    lines of slope q'/p' spaced 2 pi / p apart in the tube angle
    (spacing by 2 pi / d would re-trace the same curve when
    p/d > 1)."""
    import numpy as np
    d = gcd(p, q)
    pr, qr = p // d, q // d
    t = np.linspace(0.0, 2.0 * pi, samples, endpoint=False)
    comps = []
    for k in range(d):
        ph = 2.0 * pi * k / p
        rr = major + minor * np.cos(qr * t + ph)
        P = np.stack([rr * np.cos(pr * t), rr * np.sin(pr * t),
                      minor * np.sin(qr * t + ph)], axis=1)
        comps.append(P)
    return comps


if _IN_BLENDER:

    class CURVE_OT_torus_knot_add(bpy.types.Operator):
        """Add a (p, q) torus knot -- or, when gcd(p, q) > 1, a
        torus link of gcd(p, q) components -- as a curve or tube
        mesh"""
        bl_idname = "curve.torus_knot_add"
        bl_label = "Torus Knot"
        bl_options = {'REGISTER', 'UNDO'}

        p: IntProperty(
            name="p", default=2, min=1, max=16,
            description="Windings around the torus axis")
        q: IntProperty(
            name="q", default=3, min=1, max=16,
            description="Loops through the torus hole; "
                        "gcd(p, q) > 1 gives a link with that "
                        "many components")
        major_radius: FloatProperty(
            name="Major Radius", default=0.7, min=0.05, max=10.0,
            description="Torus centre-line radius")
        minor_radius: FloatProperty(
            name="Minor Radius", default=0.3, min=0.01, max=10.0,
            description="Torus tube radius the knot lies on")
        samples: IntProperty(
            name="Samples", default=192, min=24, max=2048,
            description="Points per component")
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
        color_components: BoolProperty(
            name="Color Components", default=True,
            description="One material with a distinct color per "
                        "link component (links only)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            import numpy as np
            comps = torus_link_components(
                self.p, self.q, self.samples,
                self.major_radius, self.minor_radius)
            comps = [np.asarray(c) * self.scale for c in comps]
            d = len(comps)
            name = (f"Torus Knot ({self.p},{self.q})" if d == 1
                    else f"Torus Link ({self.p},{self.q})")
            color = self.color_components and d > 1
            if self.output == 'MESH':
                try:
                    from . import prime_knot_generator as pk
                except ImportError:
                    import prime_knot_generator as pk
                tube = pk.CURVE_OT_prime_knot_add._tube
                verts = []
                faces = []
                midx = []
                for k, P in enumerate(comps):
                    v, f = tube(P, self.radius, self.tube_sides)
                    base = len(verts)
                    verts.extend(v)
                    faces.extend([[base + i for i in face]
                                  for face in f])
                    midx.extend([k] * len(f))
                me = bpy.data.meshes.new(name)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                me.polygons.foreach_set(
                    'use_smooth', [True] * len(me.polygons))
                if color and len(me.polygons) == len(midx):
                    me.polygons.foreach_set('material_index', midx)
                me.update()
                obj = bpy.data.objects.new(name, me)
            else:
                cu = bpy.data.curves.new(name, 'CURVE')
                cu.dimensions = '3D'
                for P in comps:
                    if self.output == 'BEZIER':
                        sp = cu.splines.new('BEZIER')
                        sp.bezier_points.add(len(P) - 1)
                        for i, pnt in enumerate(P):
                            bp = sp.bezier_points[i]
                            bp.co = pnt
                            bp.handle_left_type = 'AUTO'
                            bp.handle_right_type = 'AUTO'
                    else:
                        sp = cu.splines.new(self.output)
                        sp.points.add(len(P) - 1)
                        for i, pnt in enumerate(P):
                            sp.points[i].co = (pnt[0], pnt[1],
                                               pnt[2], 1.0)
                        if self.output == 'NURBS':
                            sp.order_u = 4
                    sp.use_cyclic_u = True
                cu.bevel_depth = self.radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)
            if color:
                import colorsys
                for k in range(d):
                    rgb = colorsys.hsv_to_rgb(k / d, 0.72, 0.9)
                    mat = bpy.data.materials.new(
                        f"{name} C{k + 1}")
                    mat.diffuse_color = (*rgb, 1.0)
                    mat.use_nodes = True
                    node = mat.node_tree.nodes.get(
                        "Principled BSDF")
                    if node:
                        node.inputs["Base Color"].default_value \
                            = (*rgb, 1.0)
                    obj.data.materials.append(mat)
                if self.output != 'MESH':
                    # spline.material_index is clamped to the
                    # material count at assignment time, so set it
                    # only now that the materials exist
                    for k, sp in enumerate(obj.data.splines):
                        sp.material_index = k
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: {d} component(s), "
                        f"{self.samples} samples each")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('p', 'q', 'major_radius', 'minor_radius',
                      'samples', 'output', 'radius'):
                lay.prop(self, k)
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            elif self.radius > 0:
                lay.prop(self, 'resolution')
            if gcd(self.p, self.q) > 1:
                lay.prop(self, 'color_components')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.torus_knot_add",
                             icon='FORCE_VORTEX')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(CURVE_OT_torus_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_torus_knot_add)


def _selftest():
    import numpy as np
    ok_all = True
    for p, q in ((2, 3), (3, 5), (2, 2), (2, 4), (3, 3),
                 (4, 6)):
        comps = torus_link_components(p, q, 200)
        d = gcd(p, q)
        # every point on the torus (R=0.7, r=0.3)
        dev = max(
            abs(np.hypot(np.hypot(P[:, 0], P[:, 1]) - 0.7,
                         P[:, 2]) - 0.3).max()
            for P in comps)
        # components pairwise disjoint
        mind = min(
            np.linalg.norm(A[:, None, :] - B[None, :, :],
                           axis=-1).min()
            for i, A in enumerate(comps)
            for B in comps[i + 1:]) if d > 1 else 1.0
        ok = len(comps) == d and dev < 1e-12 and mind > 1e-3
        ok_all = ok_all and ok
        print(f"({p},{q}): components={len(comps)} (want {d}) "
              f"on-torus dev={dev:.1e} min-sep={mind:.3f} "
              f"{'OK' if ok else 'BAD'}")
    assert ok_all
    print("torus knot standalone tests passed")
