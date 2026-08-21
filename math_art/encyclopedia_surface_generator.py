# Encyclopedia Surfaces generator for Blender.
#
# The Blender layer over `math_art/surfaces/encyclopedia.py`, which holds
# all the mathematics and the self-test.  Two groups of parametric
# surfaces from Ferreol's encyclopedia:
#
#   Zoll surfaces    Tannery's pear, Tannery's hourglass and Zoll's own
#       surface -- surfaces other than the round sphere on which EVERY
#       geodesic closes up, almost all of them at the same length.
#   Darboux surfaces  one rigid curve carried by a motion.  The three
#       classical motions give translation surfaces, surfaces of
#       revolution and helicoids; a general motion gives a Darboux
#       surface that is none of the three.
#
# References:
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables",
#   mathcurve.com, chapters "poire de Tannery" and "surface de Darboux".
# - J. Tannery, Bulletin des sciences mathematiques, 2e serie, 16
#   (1892) 190.
# - O. Zoll, "Ueber Flaechen mit Scharen geschlossener geodaetischer
#   Linien", Mathematische Annalen 57 (1903) 108-133.
# - G. Darboux, "Lecons sur la theorie generale des surfaces", 1887-96.
# - A. L. Besse, "Manifolds all of whose Geodesics are Closed",
#   Springer 1978.

bl_info = {
    "name": "Encyclopedia Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Zoll surfaces (Tannery's pear and hourglass, "
                   "Zoll's surface) and Darboux surfaces swept by a "
                   "rigid curve",
    "category": "Add Mesh",
}

import numpy as np

try:
    from . import rim_curve as _rim
except ImportError:                       # flat import outside the package
    import rim_curve as _rim

try:
    from .surfaces.encyclopedia import (PRESETS, PRESET_ORDER, MOTIONS,
                                        GENERATRICES, build_preset)
except ImportError:                       # flat import outside the package
    from surfaces.encyclopedia import (PRESETS, PRESET_ORDER, MOTIONS,
                                       GENERATRICES, build_preset)


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _new_object(context, name, verts, faces, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(map(float, v)) for v in np.asarray(verts)],
                       [], [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth', [smooth] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    _PRESET_ITEMS = [(k, PRESETS[k][0], PRESETS[k][2])
                     for k in PRESET_ORDER]
    _MOTION_ITEMS = [
        ('GENERAL', "General Darboux Motion",
         "Spin about the axis while tumbling, with the curve's centre "
         "riding a circle: a Darboux surface that is neither a "
         "translation surface, nor a surface of revolution, nor a "
         "helicoid"),
        ('TRANSLATION', "Translation",
         "Slide the curve along a straight line: a translation surface"),
        ('REVOLUTION', "Revolution",
         "Spin the curve about a fixed axis: a surface of revolution"),
        ('HELICOID', "Helicoidal",
         "Screw the curve about a fixed axis: a helicoid"),
    ]
    _GENERATRIX_ITEMS = [
        ('CIRCLE', "Circle",
         "A circular generatrix; under a rotation this gives a cyclic "
         "surface"),
        ('ELLIPSE', "Ellipse", "An elliptical generatrix"),
        ('LEMNISCATE', "Gerono Lemniscate",
         "A figure-eight generatrix"),
        ('ASTROID', "Astroid", "A four-cusped generatrix"),
        ('SEGMENT', "Segment",
         "A straight generatrix, which makes the result a ruled "
         "surface"),
    ]

    class MESH_OT_encyclopedia_surface_add(bpy.types.Operator):
        """Add a Zoll surface (every geodesic closes) or a Darboux
        surface (one rigid curve carried by a motion)"""
        bl_idname = "mesh.encyclopedia_surface_add"
        bl_label = "Encyclopedia Surface"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Surface", items=_PRESET_ITEMS, default='TANNERY_PEAR',
            description="Which surface to build")
        size: FloatProperty(
            name="Size", default=1.0, min=0.05, max=10.0,
            description="The length a in the parametrisation. The "
                        "result is fitted to a 2 m cube afterwards, so "
                        "for the Zoll surfaces this changes nothing "
                        "visible; it is here because the closed "
                        "geodesics have length 2 pi a")
        res_u: IntProperty(
            name="Rings", default=64, min=6, max=400,
            description="Samples along the profile (the generatrix, "
                        "for a Darboux surface)")
        res_v: IntProperty(
            name="Segments", default=96, min=6, max=512,
            description="Samples around the sweep")
        motion: EnumProperty(
            name="Motion", items=_MOTION_ITEMS, default='GENERAL',
            description="How the rigid curve is carried through space "
                        "(Darboux surface only)")
        generatrix: EnumProperty(
            name="Generatrix", items=_GENERATRIX_ITEMS, default='CIRCLE',
            description="The rigid curve that is swept (Darboux "
                        "surface only)")
        curve_size: FloatProperty(
            name="Curve Size", default=0.45, min=0.02, max=5.0,
            description="Size of the swept curve (Darboux only)")
        curve_ratio: FloatProperty(
            name="Curve Ratio", default=0.5, min=0.02, max=5.0,
            description="Minor-to-major ratio of an elliptical "
                        "generatrix (Darboux only)")
        radius: FloatProperty(
            name="Path Radius", default=1.0, min=0.0, max=10.0,
            description="Radius of the circle the curve's centre rides "
                        "(Darboux only)")
        pitch: FloatProperty(
            name="Pitch", default=0.4, min=-5.0, max=5.0,
            description="Rise per radian for the translation and screw "
                        "motions (Darboux only)")
        tilt: FloatProperty(
            name="Tumble", default=0.6, min=0.0, max=3.14,
            description="How far the curve tips out of its plane as it "
                        "goes round; 0 collapses the general motion "
                        "back to a plain revolution (Darboux only)")
        wobbles: IntProperty(
            name="Tumbles", default=3, min=1, max=24,
            description="How many times the curve tips back and forth "
                        "per revolution (Darboux only)")
        turns: FloatProperty(
            name="Turns", default=1.0, min=0.05, max=12.0,
            description="How far the motion runs, in revolutions. A "
                        "whole number closes the sweep and welds the "
                        "seam; anything else leaves it open (Darboux "
                        "only)")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this "
                        "thickness")
        smooth: BoolProperty(name="Smooth Shading", default=True)
        rim: _rim.rim_prop()
        rim_thickness: _rim.rim_thickness_prop()
        rim_smooth: _rim.rim_smooth_prop()
        rim_profile: _rim.rim_profile_prop()
        rim_twist: _rim.rim_twist_prop()
        rim_reeds: _rim.rim_reeds_prop()

        def execute(self, context):
            key = self.preset
            label = PRESETS[key][0]
            kw = {}
            if key == 'DARBOUX':
                kw = dict(motion=self.motion, generatrix=self.generatrix,
                          size=self.curve_size, ratio=self.curve_ratio,
                          radius=self.radius, pitch=self.pitch,
                          tilt=self.tilt, wobbles=self.wobbles,
                          turns=self.turns)
            verts, faces = build_preset(
                key, res_u=self.res_u, res_v=self.res_v, a=self.size,
                scale=self.scale, **kw)
            if not len(faces):
                self.report({'ERROR'}, "Empty surface")
                return {'CANCELLED'}
            obj = _new_object(context, label, verts, faces,
                              smooth=self.smooth)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            nrim = 0
            if self.rim:
                # the rim sweeps the open edge, so it needs triangles
                tris = []
                for f in faces:
                    for i in range(1, len(f) - 1):
                        tris.append((f[0], f[i], f[i + 1]))
                nrim = _rim.add_rim_curve(
                    context, obj, label, np.asarray(verts),
                    np.asarray(tris, dtype=np.int64),
                    self.rim_thickness, self.rim_smooth,
                    self.rim_profile, twist=self.rim_twist,
                    reeds=self.rim_reeds)
            me = obj.data
            self.report({'INFO'},
                        f"{label}: {len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces"
                        + (f", rim {nrim} loop(s)" if self.rim else ""))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'DARBOUX':
                for k in ('motion', 'generatrix', 'curve_size'):
                    lay.prop(self, k)
                if self.generatrix == 'ELLIPSE':
                    lay.prop(self, 'curve_ratio')
                if self.motion in ('GENERAL', 'TRANSLATION'):
                    lay.prop(self, 'radius')
                if self.motion in ('TRANSLATION', 'HELICOID'):
                    lay.prop(self, 'pitch')
                if self.motion == 'GENERAL':
                    lay.prop(self, 'tilt')
                    lay.prop(self, 'wobbles')
                lay.prop(self, 'turns')
            else:
                lay.prop(self, 'size')
            for k in ('res_u', 'res_v', 'scale', 'thickness', 'smooth'):
                lay.prop(self, k)
            _rim.draw_rim(lay, self)

    def _menu_func(self, context):
        self.layout.operator("mesh.encyclopedia_surface_add",
                             icon='SURFACE_NCURVE')

    _classes = (MESH_OT_encyclopedia_surface_add,)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


def _selftest():
    """The Blender layer's own gate: every combination the operator can
    offer has to mesh.  The mathematics is gated in
    `surfaces/encyclopedia.py`; this checks the wiring, and in
    particular that every motion/generatrix pair the two enums allow
    really does build (the SEGMENT generatrix takes a different sampling
    path, and an unclosed `turns` a different face topology)."""
    ok = True
    bad = []
    for key in PRESET_ORDER:
        V, F = build_preset(key, res_u=24, res_v=32)
        if len(F) < 50 or not np.all(np.isfinite(V)):
            bad.append('%s:%d faces' % (key, len(F)))
    ok &= not bad
    print("encyclopedia_generator: %d presets mesh %s"
          % (len(PRESET_ORDER), 'OK' if not bad else 'FAIL ' + ','.join(bad)))

    bad = []
    for mot, _l in MOTIONS:
        for gen, _g in GENERATRICES:
            for turns in (1.0, 2.0, 1.5):
                V, F = build_preset(
                    'DARBOUX', res_u=20, res_v=28, motion=mot,
                    generatrix=gen, turns=turns)
                if len(F) < 30 or not np.all(np.isfinite(V)):
                    bad.append('%s/%s@%.1f:%d' % (mot, gen, turns, len(F)))
                elif any(len(set(f)) != len(f) for f in F):
                    bad.append('%s/%s@%.1f:degenerate' % (mot, gen, turns))
    n = len(MOTIONS) * len(GENERATRICES) * 3
    ok &= not bad
    print("encyclopedia_generator: %d Darboux combinations mesh %s"
          % (n, 'OK' if not bad else 'FAIL ' + ','.join(bad)))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("encyclopedia generator self-test failed")
