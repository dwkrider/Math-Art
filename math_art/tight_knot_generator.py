# Tight Knot Generator for Blender
#
# Tight (ideal-style) knots and untangling by minimizing the
# tangent-point energy under length and barycenter constraints, with a
# fractional Sobolev (H^s) preconditioner -- the `knots.tangent_point`
# engine.  The energy blows up as the rope approaches self-contact, so
# the flow cannot change the knot type: the braid-table seed relaxes to
# a canonical tight shape (and a tangled unknot -- e.g. the braid
# abaBcBC -- untangles to a round circle).  This is the
# topology-guaranteed upgrade of the Prime Knot generator's smoothing +
# repulsion relaxer, which remains available as the fast preview.
#
# The rope radius can be set automatically from the measured
# Gonzalez-Maddocks thickness, so the swept tube is exactly the
# maximal-thickness rope of the tight shape.
#
# References:
# - G. Buck and J. Orloff, "A simple energy function for knots",
#   Topology Appl. 61 (1995) -- the tangent-point energy.
# - C. Yu, H. Schumacher, K. Crane, "Repulsive Curves", ACM Trans.
#   Graph. 40(2) (2021) -- the fractional Sobolev-Slobodeckij
#   preconditioned flow implemented in `knots/tangent_point.py`.
# - O. Gonzalez and J. H. Maddocks, "Global curvature, thickness, and
#   the ideal shapes of knots", PNAS 96 (1999) -- thickness/ropelength.
# - Thomas A. Gittings, "Minimum braids: a complete invariant of knots
#   and links", arXiv:math/0401051, 2004 (Table 1 braid words).

bl_info = {
    "name": "Tight Knots",
    "author": "Math Art project (braid table after Thomas Gittings)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Tight Knot",
    "description": "Tight knots and untangling via the tangent-point "
                   "energy with Sobolev preconditioning",
    "category": "Add Curve",
}

import numpy as np

from .knots import (KNOTS, braid_closure_points, closed_tube,
                    closure_components, gm_ropelength, gm_thickness,
                    parse_letters, resample_closed, tighten)


def build_tight_knot(braid, samples=140, iters=60, mirror=False,
                     scale=1.0):
    """Seed the braid closure, run the tangent-point flow, and
    normalize the result to fit the standard 2 m cube (centered at the
    origin, max coordinate 1) times `scale`.

    Returns (P, info) with info the flow record plus the thickness and
    ropelength of the NORMALIZED curve."""
    word = parse_letters(braid)
    if closure_components(word) != 1:
        raise ValueError(f"braid {braid!r} closes to a link, not a "
                         "knot")
    P = resample_closed(braid_closure_points(word), samples)
    if mirror:
        P = P * np.array([1.0, 1.0, -1.0])
    P, info = tighten(P, iters=iters)
    P = P - P.mean(axis=0)
    m = float(np.abs(P).max())
    if m > 1e-12:
        P = P * (scale / m)
    info = dict(info)
    info["thickness"] = gm_thickness(P)
    info["ropelength"] = gm_ropelength(P)
    return P, info


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
               "Closure of the braid word entered below (the default "
               "abaBcBC is a 7-crossing tangled UNKNOT: watch it "
               "untangle to a circle)")]
    _ITEMS += [(name, name.replace('_', '.'),
                f"{name.split('_')[0]} crossings, minimum braid "
                f"{braid}")
               for (name, braid, _ap) in KNOTS]
    _BRAIDS = {name: braid for (name, braid, _ap) in KNOTS}

    class CURVE_OT_tight_knot_add(bpy.types.Operator):
        """Add a tight knot: the braid-table seed relaxed under the
        self-avoiding tangent-point energy (knot type provably kept)"""
        bl_idname = "curve.tight_knot_add"
        bl_label = "Tight Knot"
        bl_options = {'REGISTER', 'UNDO'}

        knot: EnumProperty(name="Knot", items=_ITEMS, default='3_1')
        braid: StringProperty(
            name="Braid Word", default="abaBcBC",
            description="Letters a..z are braid generators, A..Z "
                        "their inverses (used for Custom)")
        samples: IntProperty(
            name="Curve Samples", default=140, min=48, max=400,
            description="Polyline resolution; the dense solver is "
                        "O(n^3) per step, keep moderate")
        iters: IntProperty(
            name="Tighten Iterations", default=60, min=0, max=500,
            description="Tangent-point flow steps (the flow usually "
                        "converges in a few tens; 0 shows the seed)")
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
        auto_radius: BoolProperty(
            name="Radius From Thickness", default=True,
            description="Set the rope radius to the measured "
                        "Gonzalez-Maddocks thickness of the tight "
                        "shape (the maximal embedded tube)")
        radius: FloatProperty(
            name="Tube Radius", default=0.08, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius (when not "
                        "taken from the thickness)")
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
                P, info = build_tight_knot(
                    braid, self.samples, self.iters,
                    mirror=self.mirror, scale=self.scale)
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            radius = (0.98 * info["thickness"] if self.auto_radius
                      else self.radius)
            name = ("Tight Knot " + self.knot.replace('_', '.')
                    if self.knot != 'CUSTOM' else "Tight Knot custom")
            if self.output == 'MESH':
                verts, faces = closed_tube(P, radius,
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
                cu.bevel_depth = radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report(
                {'INFO'},
                f"{name}: braid {braid}, E {info['E0']:.1f} -> "
                f"{info['E']:.1f} in {info['iters_run']} steps, "
                f"ropelength {info['ropelength']:.2f}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'knot')
            if self.knot == 'CUSTOM':
                lay.prop(self, 'braid')
            for k in ('samples', 'iters', 'mirror', 'output',
                      'auto_radius'):
                lay.prop(self, k)
            if not self.auto_radius:
                lay.prop(self, 'radius')
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            else:
                lay.prop(self, 'resolution')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.tight_knot_add",
                             icon='FORCE_VORTEX')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_tight_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_tight_knot_add)


def _selftest():
    ok = True
    from .knots.alexander import alexander_from_curve

    # The pipeline must keep the knot type (Alexander gate), reduce the
    # energy, and emit a curve normalized into the 2 m cube.
    P, info = build_tight_knot('AAA', samples=96, iters=15)
    alex = alexander_from_curve(P)
    good = (alex == 91 and info["E"] < info["E0"]
            and abs(float(np.abs(P).max()) - 1.0) < 1e-9
            and float(np.linalg.norm(P.mean(axis=0))) < 1e-9
            and info["thickness"] > 0.0)
    ok &= good
    print(f"tight_knot: trefoil alex={alex} (exp 91), E "
          f"{info['E0']:.1f}->{info['E']:.1f}, thickness "
          f"{info['thickness']:.3f} {'OK' if good else 'FAIL'}")

    # A link-closing braid must be rejected, not silently emitted.
    try:
        build_tight_knot('aa', samples=64, iters=0)
        good = False
    except ValueError:
        good = True
    ok &= good
    print(f"tight_knot: link braid rejected {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("tight_knot self-test failed")
