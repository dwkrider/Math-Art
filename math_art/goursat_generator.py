# Goursat Surfaces generator for Blender.
#
# The Blender layer over `math_art/surfaces/goursat.py`, which holds all
# the mathematics and the self-test.  Goursat's question was which
# algebraic surfaces carry the full symmetry group of a Platonic solid;
# his answer is a handful of one-parameter families, and this operator is
# a browser for them.  Pick a family, then either a named member or
# Custom and drive the coefficients by hand -- the whole family is
# continuous in them, so sweeping a slider walks the surface through its
# topological transitions (the tetrahedral cubic passes through Cayley's
# four-nodal surface exactly at k = 4).
#
# References:
# - E. Goursat, "Etude des surfaces qui admettent tous les plans de
#   symetrie d'un polyedre regulier", Annales scientifiques de l'Ecole
#   Normale Superieure, 3e serie, 4 (1887) 159-200.
#   http://www.numdam.org/article/ASENS_1887_3_4__159_0.pdf
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables",
#   mathcurve.com, chapter "surface de Goursat" -- the named members and
#   their coefficient tuples.
# - W. Barth, "Two projective surfaces with many nodes admitting the
#   symmetries of the icosahedron", J. Algebraic Geometry 5 (1996)
#   173-186 -- the sextic member.

bl_info = {
    "name": "Goursat Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "The algebraic surfaces with the symmetries of a "
                   "regular polyhedron: octahedral quartics, "
                   "tetrahedral cubics and quartics, dodecahedral "
                   "sextics",
    "category": "Add Mesh",
}

import numpy as np

try:
    from . import rim_curve as _rim
except ImportError:                       # flat import outside the package
    import rim_curve as _rim

# The mathematics lives in the sibling `surfaces` engine package; this
# module is the Blender layer over it.
try:
    from .surfaces.goursat import (PRESETS, PRESET_ORDER, PRESETS_BY_FAMILY,
                                   FAMILIES, FAMILY_LABEL, FAMILY_NCOEFF,
                                   build_goursat, default_coeffs)
except ImportError:                       # flat import outside the package
    from surfaces.goursat import (PRESETS, PRESET_ORDER, PRESETS_BY_FAMILY,
                                  FAMILIES, FAMILY_LABEL, FAMILY_NCOEFF,
                                  build_goursat, default_coeffs)


#: what each family's coefficients are called, in order.  The operator
#: exposes four sliders and relabels them per family rather than
#: carrying twelve properties that are mostly hidden.
COEFF_NAMES = {
    'OCT4': ("k", "k'", "k''"),
    'TET3': ("k", "k'"),
    'TET4': (),
    'DODEC6': ("k", "k'", "k''", "k'''"),
}

#: the "Custom" sentinel appended to every family's preset list
CUSTOM = 'CUSTOM'


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

    # Preset lists, built once: the full union and one list per family,
    # each with a trailing Custom entry.  An enum whose items come from
    # a callback cannot carry a static default, so the effective default
    # is whatever comes first -- keep the octahedral quartics first.
    def _row(k):
        lab, _fam, _c, _r, desc = PRESETS[k]
        return (k, lab, desc)

    _CUSTOM_ROW = (CUSTOM, "Custom",
                   "Drive the family's coefficients from the sliders "
                   "below rather than taking a named member")

    _PRESET_ITEMS_ALL = [_row(k) for k in PRESET_ORDER] + [_CUSTOM_ROW]
    _PRESET_ITEMS_FAM = {
        f: [_row(k) for k in PRESETS_BY_FAMILY[f]] + [_CUSTOM_ROW]
        for f, _l, _fn, _n, _d in FAMILIES}
    _FAMILY_ITEMS = [
        (f, lab, "%s -- degree %d, %d named members"
         % (lab, deg, len(PRESETS_BY_FAMILY[f])))
        for f, lab, _fn, _n, deg in FAMILIES]

    def _preset_items(self, context):
        # Fall back to the FULL union whenever there is no UI area
        # (context None, or a background/scripted call), so that a
        # scripted mesh.goursat_surface_add(preset='DODEC_ROUNDED') and
        # the headless icon/doc renders are not filtered out by the
        # family, and the stored index maps against the same list on
        # both set and get.
        if context is None or getattr(context, 'area', None) is None:
            return _PRESET_ITEMS_ALL
        return _PRESET_ITEMS_FAM.get(self.family, _PRESET_ITEMS_ALL)

    def _on_family_change(self, context):
        """Switching family must move `preset` with it.

        A dynamic enum stores an INDEX into whatever list the items
        callback last returned, so after the list is swapped the old
        index either points at an unrelated surface or at nothing.
        Landing on the family's first named member is both the fix and
        the obvious behaviour; the coefficient sliders follow it, so
        flipping straight to Custom starts from something that meshes
        rather than from the previous family's numbers (which for a
        different degree mean something else entirely).
        """
        keys = PRESETS_BY_FAMILY.get(self.family) or PRESET_ORDER
        if self.preset != keys[0]:
            self.preset = keys[0]
        _sync_coeffs(self, PRESETS[keys[0]][2], PRESETS[keys[0]][3])

    def _sync_coeffs(op, coeffs, clip):
        """Copy a member's coefficients onto the sliders."""
        for i, name in enumerate(('k0', 'k1', 'k2', 'k3')):
            v = float(coeffs[i]) if i < len(coeffs) else 0.0
            if getattr(op, name) != v:
                setattr(op, name, v)
        if op.clip != float(clip):
            op.clip = float(clip)

    def _on_preset_change(self, context):
        """Selecting a named member loads its coefficients and clip, so
        that switching to Custom afterwards starts from where you are
        looking rather than from wherever the sliders were left."""
        row = PRESETS.get(self.preset)
        if row is not None:
            _sync_coeffs(self, row[2], row[3])

    class MESH_OT_goursat_surface_add(bpy.types.Operator):
        """Add a Goursat surface: an algebraic surface carrying the
        full symmetry group of a regular polyhedron. Pick a family,
        then a named member or Custom coefficients."""
        bl_idname = "mesh.goursat_surface_add"
        bl_label = "Goursat Surface"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(
            name="Symmetry",
            items=_FAMILY_ITEMS,
            default='OCT4',
            update=_on_family_change,
            description="Which polyhedron's symmetry group the surface "
                        "carries, and at what degree; filters the "
                        "Surface list")
        preset: EnumProperty(
            name="Surface",
            items=_preset_items,
            update=_on_preset_change,
            description="A named member of the family, or Custom to "
                        "drive the coefficients by hand")
        k0: FloatProperty(
            name="k", default=-1.0, min=-40.0, max=40.0, step=10,
            description="First coefficient of the family")
        k1: FloatProperty(
            name="k'", default=1.0, min=-40.0, max=40.0, step=10,
            description="Second coefficient of the family")
        k2: FloatProperty(
            name="k''", default=1.0, min=-100.0, max=100.0, step=10,
            description="Third coefficient of the family")
        k3: FloatProperty(
            name="k'''", default=0.0, min=-200.0, max=200.0, step=10,
            description="Fourth coefficient (dodecahedral sextics only)")
        size: FloatProperty(
            name="Size", default=1.0, min=0.05, max=10.0,
            description="The length a the coefficients are measured "
                        "against. It rescales the surface within the "
                        "clip ball rather than the finished object, so "
                        "moving it changes how much of the surface is "
                        "shown, not how big the result is")
        resolution: IntProperty(
            name="Resolution", default=120, min=16, max=300,
            description="Sample grid resolution per axis. These "
                        "surfaces carry their interest in nodes, "
                        "contained lines and thin sheets, all of which "
                        "a coarse grid rounds away")
        clip: FloatProperty(
            name="Clip Radius", default=2.4, min=0.2, max=20.0,
            description="Radius of the clip ball, in units of Size. A "
                        "ball rather than a box, because a cubical "
                        "window would break the very symmetry these "
                        "surfaces are built on")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Uniform scale of the finished object; the "
                        "surface is fitted to a 2 m cube first")
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

        def _resolved(self):
            """(label, family, coefficients) for the current state."""
            row = PRESETS.get(self.preset)
            if row is None:                       # Custom, or a stale index
                fam = self.family
                n = FAMILY_NCOEFF[fam]
                co = (self.k0, self.k1, self.k2, self.k3)[:n]
                return "Goursat %s" % FAMILY_LABEL[fam], fam, co
            lab, fam, co, _r, _d = row
            return lab, fam, co

        def execute(self, context):
            label, fam, coeffs = self._resolved()
            verts, tris = build_goursat(
                fam, tuple(float(c) for c in coeffs),
                res=self.resolution, a=self.size, clip=self.clip,
                scale=self.scale)
            if len(tris) == 0:
                self.report({'ERROR'},
                            "Empty level set -- no real points inside "
                            "the clip ball at these coefficients")
                return {'CANCELLED'}
            obj = _new_object(context, label, verts, tris,
                              smooth=self.smooth)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            nrim = 0
            if self.rim:
                nrim = _rim.add_rim_curve(
                    context, obj, label, verts, tris,
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
            lay.prop(self, 'family')
            lay.prop(self, 'preset')
            names = COEFF_NAMES.get(self.family, ())
            custom = self.preset not in PRESETS
            for i, nm in enumerate(names):
                row = lay.row()
                # A named member's coefficients are shown but not
                # editable: typing into them would silently contradict
                # the label above.  Switch to Custom to drive them.
                row.enabled = custom
                row.prop(self, ('k0', 'k1', 'k2', 'k3')[i], text=nm)
            for k in ('size', 'resolution', 'clip', 'scale',
                      'thickness', 'smooth'):
                lay.prop(self, k)
            _rim.draw_rim(lay, self)

    def _menu_func(self, context):
        self.layout.operator("mesh.goursat_surface_add",
                             icon='MESH_ICOSPHERE')

    _classes = (MESH_OT_goursat_surface_add,)

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
    """The Blender layer's own gate: every row the operator can offer
    must resolve to a family that exists, a coefficient count that
    matches it, and a mesh.  The mathematics is gated in
    `surfaces/goursat.py`; this checks the wiring between the two."""
    ok = True
    fams = {f for f, _l, _fn, _n, _d in FAMILIES}
    bad = []
    for key in PRESET_ORDER:
        lab, fam, co, clip, desc = PRESETS[key]
        if fam not in fams:
            bad.append('%s:family %s' % (key, fam))
        elif len(co) != FAMILY_NCOEFF[fam]:
            bad.append('%s:%d coeffs, family wants %d'
                       % (key, len(co), FAMILY_NCOEFF[fam]))
        elif not lab or not desc:
            bad.append('%s:missing label/description' % key)
        elif len(COEFF_NAMES.get(fam, ())) != FAMILY_NCOEFF[fam]:
            bad.append('%s:%s has no slider names' % (key, fam))
    ok &= not bad
    print("goursat_generator: %d presets wire to a declared family %s"
          % (len(PRESET_ORDER), 'OK' if not bad else 'FAIL ' + ','.join(bad)))

    # Custom coefficients must mesh too: the operator lets the user
    # leave every named member behind, and an empty level set there is
    # reported rather than crashed on -- but the DEFAULT custom state
    # (each family's first member) has to produce geometry.
    bad = []
    for f, _l, _fn, _n, _d in FAMILIES:
        co = default_coeffs(f)
        V, T = build_goursat(f, tuple(float(c) for c in co), 40, 1.0,
                             PRESETS[PRESETS_BY_FAMILY[f][0]][3])
        if len(T) < 200 or not np.all(np.isfinite(V)):
            bad.append('%s:%d tris' % (f, len(T)))
    ok &= not bad
    print("goursat_generator: %d families mesh at their default "
          "coefficients %s"
          % (len(FAMILIES), 'OK' if not bad else 'FAIL ' + ','.join(bad)))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("Goursat generator self-test failed")
