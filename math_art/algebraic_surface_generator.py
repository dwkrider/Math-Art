
# Algebraic Surface Generator for Blender
#
# Classical algebraic surfaces -- celebrated cubics, quartics,
# quintics and sextics -- built as implicit level sets f(x,y,z) = 0
# and meshed with the marching-tetrahedra extractor from the sibling
# Minimal Surface Toolkit module.
#
#   Clebsch diagonal cubic   (all 27 lines are real)
#   Cayley nodal cubic       (4 nodes -- the maximum for a cubic)
#   Kummer quartic           (16 nodes at the generic parameter)
#   Barth sextic             (65 nodes -- the maximum for a sextic)
#   Togliatti quintic        (31 nodes -- the maximum for a quintic)
#   Taubin heart, Ding-dong, Chmutov sextic, Tangle cube
#   Monkey saddle           (n-fold: z = Re((x+iy)^n), n = 3 classic)
#
# plus the sixty-three named surfaces of Herwig Hauser's gallery --
# Zitrus, Seepferdchen, Kreuz, Himmel und Hoelle and the rest -- which
# are chosen for their shapes rather than their theorems and several of
# which are deliberately singular (Kreuz is xyz = 0, three planes).
# A Family dropdown picks the group; the Preset list is filtered to it,
# because one flat enum of eighty entries is unusable.
#
# Geometry only; materials and rendering are left to Blender.
#
# References:
# - Clebsch diagonal cubic: A. Clebsch (1871). Cayley nodal cubic:
#   A. Cayley (1869). Kummer quartic: E. E. Kummer (1864).
# - Barth sextic (65 nodes): W. Barth (1996). Togliatti quintic
#   (31 nodes): E. G. Togliatti (1940). Chmutov surfaces:
#   S. V. Chmutov. Heart surface after G. Taubin (1994).
# - N-fold monkey saddles z = rho^n cos(n*phi) = Re((x+iy)^n) are the
#   graphs of the degree-n harmonic polynomials (real parts of the
#   holomorphic w^n); n = 2 is the ordinary saddle, n = 3 the
#   classic monkey saddle z = x^3 - 3xy^2. Ceramic renditions of
#   these saddle sheets recur in Robert Fathauer's mathematical
#   ceramics (his n-fold saddle forms).
# - The Hauser family: H. Hauser, "Bildergalerie algebraischer
#   Flaechen", Universitaet Wien -- equations transcribed from the
#   gallery captions; see the _HAUSER block in
#   math_art/surfaces/algebraic.py for the per-row provenance and
#   for the four surfaces the gallery names but gives no equation.

bl_info = {
    "name": "Algebraic Surface Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Algebraic Surface",
    "description": "Classical algebraic surfaces (Clebsch, Cayley, "
                   "Kummer, Barth, Togliatti, ...) as implicit level "
                   "sets meshed by marching tetrahedra",
    "category": "Add Mesh",
}
import numpy as np

try:
    from . import rim_curve as _rim
except ImportError:  # flat import outside the package
    import rim_curve as _rim

# The mathematics lives in the sibling `surfaces` engine package;
# this module is the Blender layer over it.
try:
    from .surfaces.algebraic import (
        PRESETS, build_algebraic, boundary_loops, FAMILIES,
        SURFACE_FAMILY, HAUSER_EQUATION, FAMILY_RESOLUTION,
        FAMILY_RESOLUTION_DEFAULT, preset_params)
except ImportError:  # flat import outside the package
    from surfaces.algebraic import (
        PRESETS, build_algebraic, boundary_loops, FAMILIES,
        SURFACE_FAMILY, HAUSER_EQUATION, FAMILY_RESOLUTION,
        FAMILY_RESOLUTION_DEFAULT, preset_params)








# ==========================================================================
# Implicit fields
# ==========================================================================
# Each field takes numpy sample grids x, y, z plus the Kummer
# parameter mu (ignored by every preset except the Kummer quartic)
# and returns f on the grid; the surface is the zero level set.

























# ==========================================================================
# Blender layer
# ==========================================================================

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
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth',
                                [smooth] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    # Preset lists, built once: the full union and one list per family.
    # An enum whose items come from a callback cannot carry a static
    # default, so CLEBSCH stays the effective default by being first in
    # PRESETS -- keep it there.
    _PRESET_ITEMS_ALL = [
        (k, v[0], HAUSER_EQUATION.get(k, v[0])) for k, v in PRESETS.items()]
    _PRESET_ITEMS_FAM = {}
    for _k, _v in PRESETS.items():
        _PRESET_ITEMS_FAM.setdefault(
            SURFACE_FAMILY.get(_k, 'CLASSICAL'), []).append(
                (_k, _v[0], HAUSER_EQUATION.get(_k, _v[0])))
    _FAMILY_ITEMS = [
        (f, lab, "%s (%d surfaces)" % (lab, len(_PRESET_ITEMS_FAM.get(f, ()))))
        for f, lab in FAMILIES if _PRESET_ITEMS_FAM.get(f)]

    def _preset_items(self, context):
        # Fall back to the FULL union whenever there is no UI area
        # (context None, or a background/scripted call).  Scripted use
        # -- mesh.algebraic_surface_add(preset='KREUZ') -- and the
        # headless icon/doc renders must not be filtered out by the
        # family, and the stored enum index has to map against the same
        # list on both set and get.
        if context is None or getattr(context, 'area', None) is None:
            return _PRESET_ITEMS_ALL
        return _PRESET_ITEMS_FAM.get(self.family, _PRESET_ITEMS_ALL)

    def _family_default(fam):
        """First preset of a family, and the resolution it wants."""
        items = _PRESET_ITEMS_FAM.get(fam) or _PRESET_ITEMS_ALL
        return items[0][0], FAMILY_RESOLUTION.get(
            fam, FAMILY_RESOLUTION_DEFAULT)

    def _sync_params(op, key):
        """Load a preset's declared defaults into the shared slots.

        Without this, switching from one member of a family to another
        would leave the previous member's coefficients in place and
        silently build a surface that is neither.
        """
        for attr, _lab, kind, dflt, _lo, _hi, _d in preset_params(key):
            v = int(dflt) if kind == 'INT' else float(dflt)
            if getattr(op, attr, None) != v:
                setattr(op, attr, v)

    def _on_preset_change(self, context):
        _sync_params(self, self.preset)

    def _on_family_change(self, context):
        """Switching family must move `preset` with it.

        A dynamic enum stores an INDEX into whatever list the items
        callback last returned, so after the list is swapped the old
        index either points at an unrelated surface or at nothing --
        which is how `PRESETS[self.preset]` came to raise a KeyError on
        an empty string.  Landing on the family's first entry is both
        the fix and the obvious behaviour.

        The resolution moves with it, to whatever the incoming family
        asks for -- overwriting a hand-set value, which is the intent,
        since a family switch is a change of subject.  Every family
        wants the same 120 today, so this is currently a no-op; it stays
        because the per-family override is the right place to put a
        surface that turns out to need more.
        """
        key, res = _family_default(self.family)
        if self.preset != key:
            self.preset = key
        if self.resolution != res:
            self.resolution = res
        _sync_params(self, key)


    class MESH_OT_algebraic_surface_add(bpy.types.Operator):
        """Add a classical algebraic surface (implicit level set
        meshed by marching tetrahedra). Pick a Family, then a Preset
        within it."""
        bl_idname = "mesh.algebraic_surface_add"
        bl_label = "Algebraic Surface"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(
            name="Family",
            items=_FAMILY_ITEMS,
            default='CLASSICAL',
            update=_on_family_change,
            description="Which group of surfaces to choose from; "
                        "filters the Preset list")
        preset: EnumProperty(
            name="Preset",
            items=_preset_items,
            update=_on_preset_change,
            description="The surface to build; the tooltip gives its "
                        "defining equation where one is printed")
        resolution: IntProperty(
            name="Resolution", default=FAMILY_RESOLUTION_DEFAULT,
            min=16, max=256,
            description="Sample grid resolution per axis. Algebraic "
                        "surfaces need more of it than the periodic "
                        "ones: their interest is usually a cusp, a "
                        "self-intersection or a double point, and a "
                        "coarse grid rounds exactly those away")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        # The parameterised presets declare what they take in
        # `PRESET_PARAMS`, and these are the slots that carry it.  The
        # LABEL each is drawn under is the declaring row's, so adding a
        # parameterised surface is a table row rather than another
        # branch here.  The RANGE is not: a property's limits are fixed
        # at registration and cannot be retuned per preset, so each slot
        # carries the widest range any family that uses it needs, and
        # the table's own lo/hi document the intent rather than binding
        # the widget.  `mu` and `fold` keep their own names because they
        # were already public API for scripted calls.
        mu: FloatProperty(
            name="Node Sharpness", default=1.3, min=1.05, max=2.0,
            description="Kummer quartic parameter (node sharpness)")
        fold: IntProperty(
            name="Folds", default=3, min=2, max=8,
            description="Saddle fold count: 2 = ordinary saddle, "
                        "3 = monkey saddle, higher = n-fold saddles")
        k0: FloatProperty(name="k", default=0.0, min=-40.0, max=40.0,
                          step=10, description="Family coefficient")
        k1: FloatProperty(name="k'", default=0.0, min=-40.0, max=40.0,
                          step=10, description="Family coefficient")
        k2: FloatProperty(name="k''", default=0.0, min=-100.0,
                          max=100.0, step=10,
                          description="Family coefficient")
        k3: FloatProperty(name="k'''", default=0.0, min=-200.0,
                          max=200.0, step=10,
                          description="Family coefficient")
        size: FloatProperty(
            name="Size", default=1.0, min=0.05, max=10.0,
            description="The length the family's coefficients are "
                        "measured against; it changes how much of the "
                        "surface the clip ball shows, not the size of "
                        "the finished object")
        clip: FloatProperty(
            name="Clip Override", default=0.0, min=0.0, max=20.0,
            description="Clip ball radius / box half-extent; "
                        "0 uses the preset default")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this "
                        "thickness")
        smooth: BoolProperty(
            name="Smooth Shading", default=True)
        rim: _rim.rim_prop()
        rim_thickness: _rim.rim_thickness_prop()
        rim_smooth: _rim.rim_smooth_prop()
        rim_profile: _rim.rim_profile_prop()
        rim_twist: _rim.rim_twist_prop()
        rim_reeds: _rim.rim_reeds_prop()

        def execute(self, context):
            # belt and braces against a stale enum index: the
            # update callback above keeps preset and family in
            # step, but a re-run from an old redo state (or a
            # scripted call naming a preset that has since gone)
            # must not crash the operator.
            key = self.preset
            if key not in PRESETS:
                key = _family_default(self.family)[0]
            label = PRESETS[key][0]
            extra = {attr: (int(getattr(self, attr)) if kind == 'INT'
                            else float(getattr(self, attr)))
                     for attr, _l, kind, _d, _lo, _hi, _dsc
                     in preset_params(key)}
            # `mu` and `fold` stay named arguments: they were public
            # before this table existed and scripted calls use them.
            extra.pop('mu', None)
            verts, tris = build_algebraic(
                key, self.resolution, mu=self.mu,
                clip=self.clip, scale=self.scale, fold=self.fold,
                **extra)
            if len(tris) == 0:
                self.report({'ERROR'}, "Empty level set")
                return {'CANCELLED'}
            obj = _new_object(context, label, verts, tris,
                              smooth=self.smooth)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            me = obj.data
            nrim = 0
            if self.rim:
                nrim = _rim.add_rim_curve(
                    context, obj, label, verts, tris,
                    self.rim_thickness, self.rim_smooth,
                    self.rim_profile, twist=self.rim_twist,
                        reeds=self.rim_reeds)
            self.report({'INFO'},
                        f"{label}: {len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces"
                        + (f", rim {nrim} loop(s)" if self.rim else ""))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
            lay.prop(self, 'preset')
            eq = HAUSER_EQUATION.get(self.preset)
            if eq:
                row = lay.row()
                row.enabled = False
                row.label(text=eq)
            lay.prop(self, 'resolution')
            # exactly the parameters this preset declares, under the
            # names its own family gives them
            for attr, lab, _k, _d, _lo, _hi, _dsc in preset_params(
                    self.preset):
                lay.prop(self, attr, text=lab)
            for k in ('clip', 'scale', 'thickness', 'smooth'):
                lay.prop(self, k)
            _rim.draw_rim(lay, self)

    def _menu_func(self, context):
        self.layout.operator("mesh.algebraic_surface_add",
                             icon='SURFACE_NSPHERE')

    _classes = (MESH_OT_algebraic_surface_add,)

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
    # standalone smoke test of the numeric core (requires the
    # Minimal Surface Toolkit importable as a sibling)
    for kind in PRESETS:
        V, T = build_algebraic(kind, 40)
        print(f"{kind:10s}: {len(V):6d} verts {len(T):6d} tris")
