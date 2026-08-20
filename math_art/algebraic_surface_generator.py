
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
        FAMILY_RESOLUTION_DEFAULT)
except ImportError:  # flat import outside the package
    from surfaces.algebraic import (
        PRESETS, build_algebraic, boundary_loops, FAMILIES,
        SURFACE_FAMILY, HAUSER_EQUATION, FAMILY_RESOLUTION,
        FAMILY_RESOLUTION_DEFAULT)








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

    def _on_family_change(self, context):
        """Switching family must move `preset` with it.

        A dynamic enum stores an INDEX into whatever list the items
        callback last returned, so after the list is swapped the old
        index either points at an unrelated surface or at nothing --
        which is how `PRESETS[self.preset]` came to raise a KeyError on
        an empty string.  Landing on the family's first entry is both
        the fix and the obvious behaviour.

        The resolution moves too: the record nodal surfaces are packed
        with double points and need a finer grid than the rest (this
        overwrites a hand-set resolution, which is the intent -- the
        family switch is a change of subject).
        """
        key, res = _family_default(self.family)
        if self.preset != key:
            self.preset = key
        if self.resolution != res:
            self.resolution = res


    def _add_rim_curve(context, obj, label, verts, tris, thickness,
                       smooth):
        """Sweep a bevelled curve along the surface's open edge.

        Parented to the surface so the pair moves as one.  Returns the
        number of rim loops found -- zero for a closed surface, which
        is not an error: a cyclide or a Hauser tube simply has no edge.
        """
        loops = boundary_loops(verts, tris, smooth=smooth)
        if not loops:
            return 0
        cu = bpy.data.curves.new(label + " Rim", 'CURVE')
        cu.dimensions = '3D'
        cu.fill_mode = 'FULL'
        cu.bevel_depth = float(thickness)
        cu.bevel_resolution = 4
        cu.use_fill_caps = True
        for pts, closed in loops:
            sp = cu.splines.new('POLY')
            sp.points.add(len(pts) - 1)
            for i, q in enumerate(pts):
                sp.points[i].co = (float(q[0]), float(q[1]),
                                   float(q[2]), 1.0)
            sp.use_cyclic_u = bool(closed)
        rim = bpy.data.objects.new(label + " Rim", cu)
        context.collection.objects.link(rim)
        rim.matrix_world = obj.matrix_world.copy()
        rim.parent = obj
        rim.matrix_parent_inverse = obj.matrix_world.inverted()
        return len(loops)

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
            description="The surface to build; the tooltip gives its "
                        "defining equation where one is printed")
        resolution: IntProperty(
            name="Resolution", default=FAMILY_RESOLUTION_DEFAULT,
            min=16, max=256,
            description="Sample grid resolution per axis (algebraic "
                        "surfaces need more than TPMS; the record "
                        "nodal surfaces need more again)")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        mu: FloatProperty(
            name="Kummer Mu", default=1.3, min=1.05, max=2.0,
            description="Kummer quartic parameter (node sharpness); "
                        "used by the Kummer preset only")
        fold: IntProperty(
            name="Fold n", default=3, min=2, max=8,
            description="Saddle fold count: 2 = ordinary saddle, "
                        "3 = monkey saddle, higher = n-fold saddles; "
                        "Monkey Saddle preset only")
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
            verts, tris = build_algebraic(
                key, self.resolution, mu=self.mu,
                clip=self.clip, scale=self.scale, fold=self.fold)
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
                nrim = _add_rim_curve(
                    context, obj, label, verts, tris,
                    self.rim_thickness, self.rim_smooth)
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
            if self.preset == 'KUMMER':
                lay.prop(self, 'mu')
            if self.preset == 'MONKEY':
                lay.prop(self, 'fold')
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
