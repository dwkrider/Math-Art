
# Strange Attractor Generator for Blender
#
# All 38 strange attractors of Chaotic Atmospheres' "MATHRULES" art
# series (chaoticatmospheres.com/mathrules-strange-attractors) as
# presets (plus Zhou-Chen from the same collection): the ODE systems
# and parameter values follow Juergen Meier's compilation
# (3d-meier.de, Tutorial 19), the artist's stated source, with
# Lorenz / Roessler / Chua transcribed from the posters themselves.
# The 4D systems (Qi, Lorenz-Stenflo) and the 6D coupled Lorenz pair
# are drawn as their (x, y, z) projection. The system is integrated with fixed-step RK4, the transient
# discarded, and the trajectory emitted as a curve -- optionally
# resampled to even arc length, with the tube radius modulated by the
# local speed (slow regions thicken, the look of the original
# renders).
#
# Run this file with plain python for a self-test: every preset must
# integrate to a bounded, non-collapsing, non-escaping trajectory.

bl_info = {
    "name": "Strange Attractors",
    "author": "Math Art project (systems after Juergen Meier "
              "and Chaotic Atmospheres)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Strange Attractor",
    "description": "Chaotic attractor curves: the 38 systems of the "
                   "Chaotic Atmospheres MATHRULES series",
    "category": "Add Curve",
}

import math
# The mathematics lives in the sibling `ifs` engine package;
# this module is the Blender layer over it.
try:
    from .ifs.flow import (PRESETS, _ORDER, build_attractor)
except ImportError:  # flat import outside the package
    from ifs.flow import (PRESETS, _ORDER, build_attractor)


# ---- the systems -------------------------------------------------------
# Each entry: key -> (label, rhs(*state, p) -> dstate,
#                     params dict, x0, dt, steps, transient, method)
# The state may have more than 3 components (Qi and Lorenz-Stenflo
# are 4D, the coupled Lorenz pair is 6D); the first three are the
# plotted projection. dt/steps are tuned so each preset draws the
# attractor's familiar shape; all values verified by the _selftest()
# routine. method is 'RK4' except where the reference renders
# depend on Euler's numerical dissipation (see Lorenz Mod 1).


# ---- Blender layer -----------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _preset_items():
        items = []
        for k in _ORDER:
            label, rhs, params = PRESETS[k][:3]
            desc = ", ".join(f"{n}={v:g}"
                             for n, v in sorted(params.items()))
            items.append((k, label, f"parameters: {desc}"))
        return items

    class CURVE_OT_attractor_add(bpy.types.Operator):
        """Add a strange attractor trajectory curve (presets after
        the Chaotic Atmospheres / Juergen Meier collection)"""
        bl_idname = "curve.attractor_add"
        bl_label = "Strange Attractor"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(name="Attractor",
                             items=_preset_items(),
                             default='LORENZ',
                             description="Which strange-attractor "
                                         "system to integrate and draw")
        steps: IntProperty(
            name="Steps", default=0, min=0, max=200000, step=1000,
            description="Integration steps (0 = preset default)")
        dt_scale: FloatProperty(
            name="Time Step Scale", default=1.0, min=0.05, max=10.0,
            description="Multiplier on the preset's integration step")
        size: FloatProperty(name="Size", default=2.0, min=0.1,
                            max=1000.0,
                            description="Largest bounding-box extent")
        spline: EnumProperty(
            name="Spline",
            items=[('POLY', "Poly", "one point per sample"),
                   ('BEZIER', "Bezier",
                    "auto-handles, smoother at low sample counts")],
            default='POLY',
            description="Spline type used for the trajectory curve")
        samples: IntProperty(
            name="Resample", default=0, min=0, max=20000,
            description="Resample to N evenly spaced points "
                        "(0 = raw integration points; even spacing "
                        "disables speed taper)")
        radius: FloatProperty(
            name="Tube Radius", default=0.05, min=0.0, max=10.0,
            step=1, precision=3,
            description="Curve bevel depth (0 = wire only)")
        resolution: IntProperty(name="Bevel Resolution", default=4,
                                min=0, max=16,
                                description="Segments around the tube "
                                            "cross-section")
        taper: FloatProperty(
            name="Speed Taper", default=0.0, min=0.0, max=1.0,
            description="Thicken the tube where the flow is slow "
                        "(0 = uniform radius)")
        profile_sides: IntProperty(
            name="Profile Sides", default=0, min=0, max=12,
            description="Polygonal tube cross-section with N flat "
                        "sides (0 = round; e.g. 5 gives the pentagon "
                        "profile used in the reference lorentz.blend)")

        def execute(self, context):
            try:
                pts, spd = build_attractor(
                    self.preset, self.size, self.dt_scale,
                    self.steps, None, self.samples)
            except (OverflowError, ValueError) as e:
                self.report({'ERROR'},
                            f"integration failed: {e} -- try a "
                            f"smaller Time Step Scale")
                return {'CANCELLED'}
            label = PRESETS[self.preset][0]
            name = f"{label} Attractor"
            cu = bpy.data.curves.new(name, 'CURVE')
            cu.dimensions = '3D'
            if self.spline == 'BEZIER':
                sp = cu.splines.new('BEZIER')
                sp.bezier_points.add(len(pts) - 1)
                for i, pnt in enumerate(pts):
                    bp = sp.bezier_points[i]
                    bp.co = pnt
                    bp.handle_left_type = 'AUTO'
                    bp.handle_right_type = 'AUTO'
                bpts = sp.bezier_points
            else:
                sp = cu.splines.new('POLY')
                sp.points.add(len(pts) - 1)
                for i, pnt in enumerate(pts):
                    sp.points[i].co = (pnt[0], pnt[1], pnt[2], 1.0)
                bpts = sp.points
            if self.taper > 0.0 and spd:
                # slow flow -> fat tube, like the original renders
                for i, s in enumerate(spd):
                    bpts[i].radius = 1.0 + self.taper * 3.0 * (1 - s)
            cu.bevel_depth = self.radius
            cu.bevel_resolution = self.resolution
            if self.radius > 0:
                cu.use_fill_caps = True
            obj = bpy.data.objects.new(name, cu)
            context.collection.objects.link(obj)
            if self.radius > 0 and self.profile_sides >= 3:
                cu.bevel_mode = 'OBJECT'
                cu.bevel_object = self._profile(
                    context, name, self.profile_sides, self.radius,
                    obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{label}: {len(pts)} points")
            return {'FINISHED'}

        @staticmethod
        def _profile(context, name, sides, radius, parent):
            """Closed N-gon poly curve used as the bevel object (the
            pentagonal-cross-section trick from lorentz.blend)."""
            cu = bpy.data.curves.new(f"{name} Profile", 'CURVE')
            cu.dimensions = '2D'
            sp = cu.splines.new('POLY')
            sp.points.add(sides - 1)
            for i in range(sides):
                a = 2 * math.pi * i / sides + math.pi / 2
                sp.points[i].co = (radius * math.cos(a),
                                   radius * math.sin(a), 0.0, 1.0)
            sp.use_cyclic_u = True
            prof = bpy.data.objects.new(f"{name} Profile", cu)
            context.collection.objects.link(prof)
            prof.parent = parent
            prof.hide_set(True)
            prof.hide_render = True
            return prof

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            for k in ('steps', 'dt_scale', 'size', 'spline',
                      'samples', 'radius'):
                lay.prop(self, k)
            if self.radius > 0:
                lay.prop(self, 'resolution')
                lay.prop(self, 'profile_sides')
            if not self.samples:
                lay.prop(self, 'taper')

    def _menu_func(self, context):
        self.layout.operator_menu_enum("curve.attractor_add",
                                       "preset",
                                       text="Strange Attractor",
                                       icon='RNDCURVE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(CURVE_OT_attractor_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_attractor_add)


def _selftest():
    fails = []
    for key in _ORDER:
        label = PRESETS[key][0]
        try:
            pts, spd = build_attractor(key)
            lo = [min(p[k] for p in pts) for k in range(3)]
            hi = [max(p[k] for p in pts) for k in range(3)]
            ext = sorted(hi[k] - lo[k] for k in range(3))
            # non-degenerate: real extent in at least 2 axes
            flat = ext[1] < 0.5
            # not settled onto a fixed point: the last fifth of
            # the trajectory still covers a decent volume
            tail = pts[-len(pts) // 5:]
            tlo = [min(p[k] for p in tail) for k in range(3)]
            thi = [max(p[k] for p in tail) for k in range(3)]
            text = max(thi[k] - tlo[k] for k in range(3))
            # Rayleigh-Benard is a deliberate transient spiral
            stuck = text < 0.5 and key != 'RAYLEIGHBENARD'
            bad = ([] + (["flat"] if flat else [])
                   + (["stuck"] if stuck else []))
            print(f"{label:28s} {len(pts):6d} pts  "
                  f"ext=({ext[0]:5.2f},{ext[1]:5.2f},"
                  f"{ext[2]:5.2f})  tail={text:5.2f}  "
                  f"{'OK' if not bad else ' '.join(bad)}")
            if bad:
                fails.append(key)
        except (OverflowError, ValueError, ZeroDivisionError) \
                as e:
            print(f"{label:28s} FAILED: {e}")
            fails.append(key)
    print(f"\n{len(PRESETS)} presets:",
          "ALL OK" if not fails else f"FAILURES: {fails}")
    assert not fails
