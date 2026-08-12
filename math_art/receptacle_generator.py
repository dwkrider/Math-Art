
# Receptacle Generator -- phyllotaxis on a surface of revolution.
#
# Vogel's sunflower disk and van Iterson's cylinder are usually written
# as two separate constructions.  Ridley's model subsumes both: place
# organs by ARC-LENGTH DENSITY along an arbitrary profile curve, and the
# disk, cone, cylinder, sphere, ovoid, poker and thistle all fall out of
# editing that one profile.  So this operator has a profile enum where a
# naive one would have three generators.
#
# Three things worth knowing about the controls:
#
#   * PARASTICHY mode runs the van Iterson solver backwards -- pick the
#     spiral pair you want to see, e.g. (5,8) for a spruce cone or
#     (5,8,13) for a pineapple, and it reports the divergence and rise
#     that produce it.  That is the INVERSE question, and Ridley does
#     not answer it, which is why both are shipped.
#   * ORGAN SIZE is a function of position, not a constant.  Real
#     capitula grow their florets along the receptacle, and the density
#     integral handles that directly.
#   * The RISE LADDER sweeps h at a fixed golden divergence and steps
#     the visible pair 2/3 -> 3/5 -> 5/8 -> 8/13 -> 13/21 with no
#     discontinuity -- Turing's transition, and the one control here
#     that is really meant to be keyframed.
#
# References: as `lsystem/phyllo.py` -- ABOP section 4.2 and Table 4.1;
# van Iterson (1907); Ridley, Mathematical Biosciences 58, 1982;
# Prusinkiewicz et al., SIGGRAPH 2001 section 7; Vogel (1979);
# Fowler, Prusinkiewicz and Battjes, SIGGRAPH 1992.

bl_info = {
    "name": "Receptacle Phyllotaxis",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Plants > Receptacle",
    "description": "Phyllotaxis on any surface of revolution: cone, "
                   "pineapple, sunflower, poker, thistle",
    "category": "Add Mesh",
}

import numpy as np

try:
    from .lsystem import phyllo
except ImportError:                       # headless / direct execution
    from lsystem import phyllo


#: Published presets (plan 3.7): profile, count, divergence, parastichy.
PRESETS = {
    "SPRUCE_CONE": dict(profile="CURVED_CONE", count=140,
                        divergence=phyllo.GOLDEN_ANGLE, pair=(5, 8),
                        label="Spruce cone (5,8)"),
    "PINEAPPLE":   dict(profile="OVOID", count=200,
                        divergence=phyllo.PINEAPPLE_ANGLE, pair=(8, 13),
                        label="Pineapple (5,8,13 triple contact)"),
    "SUNFLOWER":   dict(profile="DISK", count=800,
                        divergence=phyllo.GOLDEN_ANGLE, pair=(34, 55),
                        label="Sunflower head"),
    "POKER":       dict(profile="POKER", count=300,
                        divergence=phyllo.GOLDEN_ANGLE, pair=(8, 13),
                        label="Kniphofia poker"),
    "THISTLE":     dict(profile="THISTLE", count=260,
                        divergence=phyllo.GOLDEN_ANGLE, pair=(13, 21),
                        label="Thistle head"),
    "PINE_CONE":   dict(profile="CONE", count=120,
                        divergence=phyllo.GOLDEN_ANGLE, pair=(5, 8),
                        label="Pine cone (5,8)"),
}


try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _emit(pts, nor, siz, name, size):
        """A point cloud with normal and size attributes.

        A mesh of points rather than instanced geometry, because that is
        what Geometry Nodes wants as an input and it keeps the operator
        from imposing an organ shape on the user.
        """
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(p) for p in pts], [], [])
        me.update()
        try:
            a = me.attributes.new(name="normal_dir", type='FLOAT_VECTOR',
                                  domain='POINT')
            a.data.foreach_set("vector", nor.ravel())
        except Exception:
            pass
        for attr, vals in (("organ_size", siz),
                           ("index_f", np.arange(len(pts), dtype=float))):
            try:
                a = me.attributes.new(name=attr, type='FLOAT',
                                      domain='POINT')
                a.data.foreach_set("value", vals.astype(float))
            except Exception:
                pass
        return bpy.data.objects.new(name, me)

    class MESH_OT_receptacle_add(bpy.types.Operator):
        """Add a phyllotactic receptacle: organs placed by area density
        on any surface of revolution"""
        bl_idname = "mesh.receptacle_add"
        bl_label = "Receptacle"
        bl_options = {'REGISTER', 'UNDO', 'PRESET'}

        source: EnumProperty(
            name="Source",
            items=[('PRESET', "Preset", "A published receptacle"),
                   ('CUSTOM', "Custom", "Choose the profile directly"),
                   ('PARASTICHY', "By Parastichy", "Solve van Iterson "
                                  "backwards: name the spiral pair and "
                                  "get the divergence and rise"),
                   ('LADDER', "Rise Sweep", "Hold the divergence at the "
                              "golden angle and sweep the rise: the "
                              "visible pair climbs 2/3 - 3/5 - 5/8 - "
                              "8/13 - 13/21 with no discontinuity")],
            default='PRESET')
        preset: EnumProperty(
            name="Preset",
            items=[(k, v["label"], v["label"])
                   for k, v in sorted(PRESETS.items())],
            default='PINE_CONE')
        profile: EnumProperty(
            name="Profile",
            items=[(k, k.replace("_", " ").title(),
                    f"Surface of revolution: {k.lower()}")
                   for k in sorted(phyllo.PROFILES)],
            default='CONE')

        count: IntProperty(name="Organs", default=200, min=3, max=20000)
        divergence: FloatProperty(
            name="Divergence", default=phyllo.GOLDEN_ANGLE,
            min=0.0, max=360.0,
            description="Roll between successive organs; 137.50776 is "
                        "the golden angle")
        parastichy_m: IntProperty(name="Parastichy m", default=5,
                                  min=1, max=89)
        parastichy_n: IntProperty(name="Parastichy n", default=8,
                                  min=1, max=144)

        rise: FloatProperty(
            name="Rise", default=0.05, min=0.0005, max=0.6,
            description="Height gained per organ on the cylinder. "
                        "Lower it and the visible parastichy pair steps "
                        "up the Fibonacci ladder continuously -- "
                        "Turing's transition, and the control here most "
                        "worth keyframing")
        grade: FloatProperty(
            name="Size Gradient", default=0.0, min=-0.9, max=4.0,
            description="Organ radius as 1 + grade*s along the profile. "
                        "Real capitula grow their florets outward, and "
                        "the density integral handles that directly")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            prof, count, div = self.profile, self.count, self.divergence
            note = ""
            if self.source == 'PRESET':
                p = PRESETS[self.preset]
                prof, count, div = p["profile"], p["count"], p["divergence"]
                note = f"  pair {p['pair']}"
            elif self.source == 'LADDER':
                prof, div = 'CYLINDER', phyllo.GOLDEN_ANGLE
                pair = phyllo.contact_pair(div, self.rise)
                note = f"  rise {self.rise:.5f}, visible pair {pair}"
            elif self.source == 'PARASTICHY':
                try:
                    a, h = phyllo.van_iterson(self.parastichy_m,
                                              self.parastichy_n)
                except ValueError as exc:
                    self.report({'ERROR'}, str(exc))
                    return {'CANCELLED'}
                div = a
                note = (f"  van Iterson: alpha={a:.5f} h={h:.6f}, "
                        f"visible {phyllo.contact_pair(a, h)}")

            rho = None
            if abs(self.grade) > 1e-9:
                g = float(self.grade)
                rho = lambda s: 1.0 + g * s        # noqa: E731

            try:
                pts, nor, siz = phyllo.ridley(prof, count=count,
                                              divergence=div, rho=rho)
            except (KeyError, ValueError) as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            if not len(pts):
                self.report({'ERROR'}, "produced no organs")
                return {'CANCELLED'}

            lo, hi = pts.min(axis=0), pts.max(axis=0)
            ext = float((hi - lo).max())
            if ext > 1e-12:
                pts = (pts - (lo + hi) / 2.0) * (2.0 * self.scale / ext)

            name = (PRESETS[self.preset]["label"]
                    if self.source == 'PRESET' else f"Receptacle {prof.title()}")
            ob = _emit(pts, nor, siz, name, self.scale)
            context.collection.objects.link(ob)
            for o in context.selected_objects:
                o.select_set(False)
            ob.select_set(True)
            context.view_layer.objects.active = ob
            self.report({'INFO'},
                        f"{name}: {len(pts)} organs on {prof.lower()}, "
                        f"divergence {div:.5f}{note}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, "source")
            if self.source == 'PRESET':
                lay.prop(self, "preset")
                p = PRESETS[self.preset]
                lay.box().label(text=f"{p['profile'].title()}, "
                                     f"{p['count']} organs, pair {p['pair']}",
                                icon='INFO')
            else:
                if self.source == 'LADDER':
                    lay.prop(self, "rise")
                    pair = phyllo.contact_pair(phyllo.GOLDEN_ANGLE,
                                               self.rise)
                    box = lay.box()
                    box.label(text=f"visible pair {pair}", icon='INFO')
                    box.label(text="golden angle, cylinder")
                    lay.prop(self, "count")
                elif self.source == 'PARASTICHY':
                    lay.prop(self, "parastichy_m")
                    lay.prop(self, "parastichy_n")
                    try:
                        a, h = phyllo.van_iterson(self.parastichy_m,
                                                  self.parastichy_n)
                        box = lay.box()
                        box.label(text=f"alpha = {a:.5f} deg", icon='INFO')
                        box.label(text=f"rise h = {h:.6f}")
                    except ValueError:
                        lay.box().label(text="no solution for this pair",
                                        icon='ERROR')
                else:
                    lay.prop(self, "divergence")
                if self.source != 'LADDER':
                    lay.prop(self, "profile")
                    lay.prop(self, "count")
            lay.prop(self, "grade")
            lay.prop(self, "scale")

    def register():
        bpy.utils.register_class(MESH_OT_receptacle_add)

    def unregister():
        bpy.utils.unregister_class(MESH_OT_receptacle_add)


def _selftest():
    # every preset must name a real profile and place its organs
    for key, p in sorted(PRESETS.items()):
        assert p["profile"] in phyllo.PROFILES, (key, p["profile"])
        pts, nor, siz = phyllo.ridley(p["profile"], count=p["count"],
                                      divergence=p["divergence"])
        assert len(pts) == p["count"], key
        assert np.all(np.isfinite(pts)), key
        ext = pts.max(axis=0) - pts.min(axis=0)
        assert float((ext > 1e-9).sum()) >= 2, (key, ext)

    # the pineapple preset must actually use the triple-contact angle,
    # not the golden angle -- that difference is the whole point of the
    # (5,8,13) row
    assert abs(PRESETS["PINEAPPLE"]["divergence"]
               - phyllo.PINEAPPLE_ANGLE) < 1e-9
    assert abs(PRESETS["PINEAPPLE"]["divergence"]
               - phyllo.GOLDEN_ANGLE) > 0.5

    # The rise sweep must step the visible pair up the ladder without a
    # jump -- plan item 3.8, and the reason `rise` is a float rather than
    # a pair enum.
    ladder = phyllo.rise_ladder()
    rises = [h for _m, _n, h in ladder]
    assert all(b < a for a, b in zip(rises, rises[1:])), rises
    seen = []
    for m, n, h in ladder:
        # sample just inside the rung, since h is its upper edge
        pair = tuple(sorted(phyllo.contact_pair(phyllo.GOLDEN_ANGLE,
                                                h * 0.95)))
        assert pair == tuple(sorted((m, n))), (m, n, pair)
        seen.append(pair)
    assert len(set(seen)) == len(seen), seen
    for (a, b), (c, d) in zip(seen, seen[1:]):
        assert c >= a and d >= b, seen        # climbs, never falls back

    # the size gradient must actually change the placement
    a = phyllo.ridley("DISK", count=200)[0]
    b = phyllo.ridley("DISK", count=200, rho=lambda s: 1.0 + 2.0 * s)[0]
    assert not np.allclose(a, b)

    print(f"receptacle_generator: OK -- {len(PRESETS)} presets place "
          f"organs on real profiles; pineapple uses the 138.1395 triple "
          f"contact, not the golden angle")
