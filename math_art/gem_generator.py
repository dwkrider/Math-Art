# Faceted Gemstone Generator for Blender.
#
# Cut gemstones built the way a lapidary specifies them -- as facet
# PLANES, each placed by a mast angle and a position on the index gear --
# and turned into a solid by intersecting the resulting half-spaces.  The
# mathematics lives in the sibling `gems` engine package; this module is
# the Blender layer over it.
#
# The reference design is the Standard Round Brilliant printed in
# Strickland's GemCad manual, and the same file format that manual
# defines (`.ASC`) is what this generator imports and exports, so the
# thousands of published faceting designs are readable directly.  Note
# that third-party design libraries are licensed per designer: import
# them at runtime, do not redistribute them.
#
# Facets are emitted as single n-gons and left flat-shaded on purpose.  A
# gemstone's optical behaviour is entirely a matter of the angles between
# flat mirrors, so smooth shading -- or a bevel modifier rounding the
# edges -- destroys the only thing the geometry is for.
#
# References:
#   Marcel Tolkowsky, "Diamond Design: A Study of the Reflection and
#     Refraction of Light in a Diamond", E. & F. N. Spon, London, 1919.
#   Robert W. Strickland, "GemCad for Windows Version 1.0 User's Guide",
#     GemSoft Enterprises, 2002 -- the .ASC format and the Standard Round
#     Brilliant.
#   Robert H. Long & Norman W. Steele, "Introduction to Meetpoint
#     Faceting", Seattle Faceting Books, 1985.
#   CIBJO Diamond Commission, "The Diamond Book" (Blue Book 2024-1),
#     Annex B 7.2 -- the normative round-brilliant facet arrangement.

try:
    from .gems import (catalogue, design as gem_design, facets as gem_facets,
                       measure as gem_measure)
    from .gems import curved as gem_curved
    from .gems import materials as gem_mat
    from .gems.asc import parse_asc, write_asc
    from .polyhedra.fit import fit_cube
    from .styles import gem_material
except ImportError:                     # flat import outside the package
    from gems import (catalogue, design as gem_design, facets as gem_facets,
                      curved as gem_curved,
                      materials as gem_mat, measure as gem_measure)
    from gems.asc import parse_asc, write_asc
    from polyhedra.fit import fit_cube
    try:
        from styles import gem_material
    except ImportError:
        gem_material = None

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty, StringProperty)
    from bpy_extras.io_utils import ExportHelper, ImportHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

bl_info = {
    "name": "Faceted Gemstone",
    "author": "Math Art project (after Tolkowsky / Strickland / "
              "Long & Steele)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Gems",
    "description": "Gemstone cuts from facet planes; GemCad .ASC import",
    "category": "Add Mesh",
}


def build_gem(cut_design, span=2.0, snap=1e-5):
    """Build a cut design into mesh data.

    Returns `(verts, faces, tier_ids, info)`: vertices fitted to the
    project's `span`-unit cube convention, one n-gon per facet, the index
    of the tier each facet came from, and a report dict carrying the
    structural checks so a caller can refuse to emit a bad stone.
    """
    N, d, tier = gem_design.planes(cut_design)
    P = gem_facets.intersect_halfspaces(N, d, snap=snap)
    chk = gem_facets.polytope_checks(P, N, d)
    verts = fit_cube([tuple(v) for v in P.V.tolist()], span=span)
    # the built solid's volume, in the design's own units, is what a
    # carat weight has to be computed from -- not the fitted mesh's
    info = dict(chk)
    info["volume_design_units"] = gem_facets.volume(P)
    info["dropped"] = len(P.dropped)
    # Measure the SOLID, not the proportions we were handed: that is what
    # makes the report a check rather than an echo.
    try:
        pr = gem_measure.proportions(P, N, d, tier, fold=cut_design.fold or 8)
        info["proportions"] = pr
        info["grades"] = gem_measure.idc_grade(pr)
        info["warnings"] = gem_measure.warnings(pr, ri=cut_design.ri)
    except (ValueError, ZeroDivisionError, KeyError):
        info["proportions"] = info["grades"] = None
        info["warnings"] = []
    return (verts, [list(f) for f in P.faces],
            [int(tier[j]) for j in P.face_plane], info)


def _tier_names(cut_design):
    return [t.name or str(i + 1) for i, t in enumerate(cut_design.tiers)]


if _IN_BLENDER:

    def _emit(context, cut_design, span, snap, label):
        """Create the mesh object for a design and select it."""
        verts, faces, tiers, info = build_gem(cut_design, span=span,
                                              snap=snap)
        if not info["closed"]:
            return None, info
        me = bpy.data.meshes.new(label)
        me.from_pydata(verts, [], faces)
        me.validate(clean_customdata=True)
        # Flat, always: the facets ARE the object.
        me.polygons.foreach_set('use_smooth', [False] * len(me.polygons))
        attr = me.attributes.new("gem_tier", 'INT', 'FACE')
        attr.data.foreach_set('value', tiers)
        me.update()

        obj = bpy.data.objects.new(label, me)
        # The design travels with the object, so the stone can be
        # re-exported, re-cut for another material, or inspected later.
        obj["gem_asc"] = write_asc(cut_design)
        obj["gem_tier_names"] = ",".join(_tier_names(cut_design))
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj, info

    def _report(op, cut_design, info, me, size_mm=0.0, sg=0.0):
        """Report what was MEASURED off the solid, not what was asked for."""
        pr, gr = info.get("proportions"), info.get("grades")
        msg = (f"{cut_design.name or 'gem'}: {len(me.polygons)} facets, "
               f"{len(me.vertices)} vertices")
        if pr is not None:
            msg += (f", table {pr.table_pct * 100:.1f}%, crown "
                    f"{pr.crown_angle:.1f} deg, pavilion "
                    f"{pr.pavilion_angle:.2f} deg, depth "
                    f"{pr.total_depth_pct * 100:.1f}%")
            if gr:
                msg += f" -- proportions {gr['overall']}"
            if size_mm > 0.0 and sg > 0.0:
                ct = gem_measure.carat_from_diameter(pr, size_mm, sg)
                msg += f"; {size_mm:.1f} mm = {ct:.2f} ct"
        if info["dropped"]:
            msg += f", {info['dropped']} facet(s) cut away"
        op.report({'INFO'}, msg)
        # The IDC pathologies are warnings, not errors: a fish-eye stone is
        # a real stone, and being able to make one on purpose is the point.
        for w in info.get("warnings", []):
            op.report({'WARNING'}, w)



    RIG_ITEMS = (
        ('CUSTOM', "Custom",
         "Leave the scene's lighting, world and camera exactly as they are"),
        ('STUDIO', "Gem Studio",
         "Sky, a small bright key for fire and a broad fill for brilliance"),
        ('ASET', "ASET Rig",
         "The AGS three-zone hemisphere: where the returned light came "
         "from. Needs Cycles -- the dome is emissive geometry"),
    )

    def _apply_rig(context, kind):
        """Show one rig and hide the others, building it if it is absent.

        Rigs are found by a `gem_rig` tag rather than by name, so a user
        who renames or duplicates one keeps working.  Switching HIDES the
        rig it replaces rather than deleting it: a rig may have been
        adjusted -- the key moved, the dome resized -- and throwing that
        away because someone looked at the other view would be rude.
        `CUSTOM` hides every rig and touches nothing else, which is what
        makes it safe as the default on an operator that re-runs on every
        redo.
        """
        for o in bpy.data.objects:
            tag = o.get("gem_rig")
            if tag:
                o.hide_viewport = o.hide_render = (tag != kind)
        if kind == 'CUSTOM':
            return
        if not any(o.get("gem_rig") == kind for o in bpy.data.objects):
            if kind == 'STUDIO':
                bpy.ops.mesh.gem_studio_add()
            else:
                bpy.ops.mesh.gem_aset_rig_add()
        # point the camera and the world at the rig now in force
        for o in bpy.data.objects:
            if o.type == 'CAMERA' and o.get("gem_rig") == kind:
                context.scene.camera = o
        if kind == 'ASET':
            # the dome IS the light; a lit world would wash its zones out
            w = bpy.data.worlds.get("ASET World")
            if w is None:
                w = bpy.data.worlds.new("ASET World")
                w["gem_rig"] = 'ASET'
                w.use_nodes = True
                for n in w.node_tree.nodes:
                    if n.type == 'BACKGROUND':
                        n.inputs['Color'].default_value = (0, 0, 0, 1)
                        n.inputs['Strength'].default_value = 0.0
            context.scene.world = w
        else:
            w = next((x for x in bpy.data.worlds
                      if x.get("gem_rig") == 'STUDIO'), None)
            if w is not None:
                context.scene.world = w

    class MESH_OT_gem_add(bpy.types.Operator):
        """Add a faceted gemstone cut from its facet planes"""
        bl_idname = "mesh.gem_add"
        bl_label = "Gemstone"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Cut",
            items=[(k, lbl, desc) for k, lbl, desc in catalogue.cut_items()],
            # Named, not "whatever sorts first": the menu is ordered by
            # family and then alphabetically, so adding a cut would
            # otherwise move the default out from under the user.
            default='ROUND_BRILLIANT')
        rig: EnumProperty(
            name="Viewing Rig", items=RIG_ITEMS, default='CUSTOM',
            description="Lighting to build around the stone. Custom leaves "
                        "the scene alone; the others hide whichever rig "
                        "they replace rather than deleting it")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        # Soft ranges span the IDC grading table end to end, so the slider
        # itself shows how much room the trade recognises; the defaults sit
        # in its Excellent band.
        table_pct: FloatProperty(
            name="Table", default=57.0, min=30.0, max=85.0,
            soft_min=49.0, soft_max=70.0, subtype='PERCENTAGE')
        crown_angle: FloatProperty(
            name="Crown Angle", default=34.5, min=10.0, max=60.0,
            soft_min=26.0, soft_max=40.0,
            description="Degrees above the girdle plane (IDC Excellent: "
                        "32.0 to 36.0)")
        pavilion_angle: FloatProperty(
            name="Pavilion Angle", default=40.75, min=25.0, max=60.0,
            soft_min=38.5, soft_max=43.1,
            description="Degrees below the girdle plane (IDC Excellent: "
                        "40.6 to 41.8)")
        girdle_pct: FloatProperty(
            name="Girdle", default=3.0, min=0.0, max=15.0,
            soft_min=0.5, soft_max=7.5, subtype='PERCENTAGE')
        culet_pct: FloatProperty(
            name="Culet", default=0.0, min=0.0, max=10.0,
            subtype='PERCENTAGE',
            description="0 gives a pointed culet and 57 facets; any larger "
                        "value adds a culet facet, giving 58")
        star_len: FloatProperty(
            name="Star Length", default=55.0, min=5.0, max=95.0,
            subtype='PERCENTAGE',
            description="How far the stars reach from the table edge "
                        "towards the girdle")
        lower_girdle_len: FloatProperty(
            name="Lower Halves", default=78.0, min=5.0, max=95.0,
            subtype='PERCENTAGE',
            description="How far the lower halves reach from the girdle "
                        "towards the culet")
        girdle_facets: IntProperty(name="Girdle Facets", default=16,
                                   min=8, max=128)
        size_mm: FloatProperty(
            name="Size (mm)", default=6.5, min=0.1, max=100.0,
            description="Real diameter. Reported as a carat weight only -- "
                        "the mesh is still fitted to the 2 m cube")
        material: EnumProperty(
            name="Material",
            items=[(k, lbl, desc)
                   for k, lbl, desc in gem_mat.material_items()],
            default=gem_mat.DEFAULT,
            description="Sets the refractive index, the dispersion that "
                        "makes the fire, the absorption that makes the "
                        "colour, and the density that sets carat weight")
        assign_material: BoolProperty(
            name="Assign Material", default=True,
            description="Build and assign a dispersion shader for the "
                        "chosen species")
        bands: EnumProperty(
            name="Dispersion Bands",
            items=[('3', "3 (RGB)", "One glass BSDF per colour channel"),
                   ('6', "6", "Finer fire, roughly twice the shading cost"),
                   ('9', "9", "Finest; for stills of very fiery material")],
            default='3',
            description="Blender has no dispersion of its own, so fire is "
                        "built from this many glass BSDFs at different "
                        "refractive indices. More bands reduce colour "
                        "banding at a proportional cost")
        preview_material: BoolProperty(
            name="Preview (no dispersion)", default=False,
            description="A single Principled BSDF at the sodium-D index: "
                        "cheap and EEVEE-friendly, but with no fire")
        sg: FloatProperty(
            name="Specific Gravity", default=0.0, min=0.0, max=8.0,
            description="Override the material's own density for the carat "
                        "weight; 0 uses the species value")
        snap: FloatProperty(
            name="Meetpoint Tolerance", default=1e-5,
            min=1e-7, max=1e-3, precision=7,
            description="Radius, as a fraction of the girdle radius, "
                        "within which facet planes are taken to meet at "
                        "one point. Published designs carry eight "
                        "decimals, so the facets around a culet do not "
                        "quite concur; this is how close counts as "
                        "concurrent")

        def _parametric(self):
            src = catalogue.CUTS[self.preset].source
            return not isinstance(src, gem_design.CutDesign)

        # (property name, constructor argument, conversion)
        _PROPS = (('table_pct', 'table', 0.01),
                  ('crown_angle', 'crown_angle', 1.0),
                  ('pavilion_angle', 'pavilion_angle', 1.0),
                  ('girdle_pct', 'girdle_pct', 0.01),
                  ('culet_pct', 'culet_pct', 0.01),
                  ('star_len', 'star_len', 0.01),
                  ('lower_girdle_len', 'lower_girdle_len', 0.01),
                  ('girdle_facets', 'girdle_facets', 1))

        def _params(self):
            """Only the proportions the user actually set.

            A preset carries its own proportions -- Tolkowsky's are 53%,
            34.5 and 40.75 -- and passing every slider on every build would
            silently overwrite them with this operator's defaults, so
            picking "Tolkowsky" would quietly not give you Tolkowsky.
            `is_property_set` is true only for values explicitly supplied,
            so an untouched slider defers to the preset.
            """
            if not self._parametric():
                return {}
            allowed = catalogue.accepted_params(self.preset)
            out = {}
            for prop, arg, conv in self._PROPS:
                if arg in allowed and self.properties.is_property_set(prop):
                    v = getattr(self, prop)
                    out[arg] = v * conv if isinstance(conv, float) else v
            return out

        def execute(self, context):
            try:
                D = catalogue.get_cut(self.preset, **self._params())
                obj, info = _emit(context, D, 2.0 * self.scale, self.snap,
                                  catalogue.CUTS[self.preset].label)
            except (ValueError, KeyError, TypeError) as e:
                self.report({'ERROR'}, f"could not build the cut: {e}")
                return {'CANCELLED'}
            if obj is None:
                self.report({'ERROR'},
                            "the facet planes do not close a solid; try a "
                            "larger meetpoint tolerance")
                return {'CANCELLED'}
            spec = gem_mat.get(self.material)
            sg = self.sg if self.sg > 0.0 else spec.sg
            if self.assign_material and gem_material is not None:
                try:
                    obj.data.materials.append(gem_material.gem_material(
                        self.material, size_mm=self.size_mm,
                        bands=int(self.bands),
                        simple=self.preview_material))
                except Exception as e:      # never lose the stone over a shader
                    self.report({'WARNING'},
                                f"material not assigned: {e}")
            pr, gr = info.get("proportions"), info.get("grades")
            if pr is not None:
                # keep the measurement with the stone, so it can be read
                # back without rebuilding
                obj["gem_proportions"] = {k: float(v) for k, v
                                          in pr._asdict().items()}
                obj["gem_carat"] = gem_measure.carat_from_diameter(
                    pr, self.size_mm, sg)
            if gr:
                obj["gem_idc_grade"] = gr["overall"]
            try:
                _apply_rig(context, self.rig)
            except Exception as e:      # never lose the stone over a rig
                self.report({'WARNING'}, f"rig not applied: {e}")
            _report(self, D, info, obj.data, self.size_mm, sg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self._parametric():
                # Only the controls this family actually has: a rose has no
                # table, a step cut takes a list of row angles rather than
                # one.  Showing a slider that the cut cannot use would be
                # showing a lie.
                allowed = catalogue.accepted_params(self.preset)
                col = lay.column(align=True)
                shown = [p for p, arg, _ in self._PROPS if arg in allowed]
                for p in shown:
                    col.prop(self, p)
                if not any(self.properties.is_property_set(p)
                           for p in shown):
                    lay.label(text="using the preset's own proportions",
                              icon='INFO')
            # The measured proportions and their IDC grade are reported in
            # the status bar on build, and stored on the object.  They are
            # deliberately NOT recomputed here: draw() runs on every
            # redraw, and grading means intersecting the half-spaces again.
            lay.separator()
            lay.prop(self, 'rig')
            lay.prop(self, 'material')
            lay.prop(self, 'assign_material')
            if self.assign_material:
                lay.prop(self, 'preview_material')
                if not self.preview_material:
                    lay.prop(self, 'bands')
            lay.prop(self, 'size_mm')
            lay.prop(self, 'sg')
            lay.prop(self, 'scale')
            lay.prop(self, 'snap')

    class IMPORT_MESH_OT_gemcad_asc(bpy.types.Operator, ImportHelper):
        """Import a GemCad/Datavue faceting design (.asc)"""
        bl_idname = "import_mesh.gemcad_asc"
        bl_label = "Import GemCad Design"
        bl_options = {'REGISTER', 'UNDO'}

        filename_ext = ".asc"
        filter_glob: StringProperty(default="*.asc;*.ASC",
                                    options={'HIDDEN'})
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            import os
            try:
                with open(self.filepath, 'r', encoding='utf-8',
                          errors='replace') as f:
                    D = parse_asc(f.read())
            except (OSError, ValueError) as e:
                self.report({'ERROR'}, f"could not read the design: {e}")
                return {'CANCELLED'}
            label = D.name or os.path.splitext(
                os.path.basename(self.filepath))[0]
            try:
                obj, info = _emit(context, D, 2.0 * self.scale, 1e-5, label)
            except ValueError as e:
                self.report({'ERROR'}, f"could not build the design: {e}")
                return {'CANCELLED'}
            if obj is None:
                self.report({'ERROR'},
                            "the design's facet planes do not close a solid")
                return {'CANCELLED'}
            _report(self, D, info, obj.data)
            return {'FINISHED'}

    class EXPORT_MESH_OT_gemcad_asc(bpy.types.Operator, ExportHelper):
        """Export the active gemstone's design as GemCad .asc"""
        bl_idname = "export_mesh.gemcad_asc"
        bl_label = "Export GemCad Design"

        filename_ext = ".asc"
        filter_glob: StringProperty(default="*.asc;*.ASC",
                                    options={'HIDDEN'})

        @classmethod
        def poll(cls, context):
            obj = context.active_object
            return obj is not None and "gem_asc" in obj

        def execute(self, context):
            # Export the DESIGN carried on the object, not the mesh: a
            # mesh has lost the angles, the index gear and the symmetry,
            # and those are what a .ASC file is.
            text = context.active_object.get("gem_asc")
            if not text:
                self.report({'ERROR'},
                            "the active object carries no faceting design")
                return {'CANCELLED'}
            try:
                with open(self.filepath, 'w', encoding='utf-8',
                          newline='\n') as f:
                    f.write(text)
            except OSError as e:
                self.report({'ERROR'}, f"could not write: {e}")
                return {'CANCELLED'}
            self.report({'INFO'}, f"wrote {self.filepath}")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.gem_add", icon='MESH_ICOSPHERE')

    def _import_menu(self, context):
        self.layout.operator("import_mesh.gemcad_asc",
                             text="GemCad Faceting Design (.asc)")

    def _export_menu(self, context):
        self.layout.operator("export_mesh.gemcad_asc",
                             text="GemCad Faceting Design (.asc)")



# --------------------------------------------------------------------------
# Viewing rigs.
#
# A cut stone is almost entirely WHAT IT REFLECTS.  Rendered against a
# black world it renders black, correctly -- which is the commonest way a
# gemstone render goes wrong, and no amount of work on the material fixes
# it.  So the generator ships the lighting rather than leaving it as an
# exercise.
#
# Two rigs, for two different questions.  The STUDIO rig asks "what does
# this stone look like": a sky for the broad reflections, a small bright
# key because fire needs a small source, a soft fill because brilliance
# needs a broad one.  The ASET rig asks "where is this stone's light
# coming from": the American Gem Society's three-zone hemisphere, green
# for 0-45 degrees above the horizon, red for 45-75, blue for 75-90 where
# the observer's own head blocks the light.
#
# Neither rig can be embedded IN the stone: lights, worlds and cameras are
# separate datablocks from a mesh object.  The studio can put everything
# in one collection instead, which is the level at which Blender does let
# a lit subject travel as a unit.
#
# References:
#   Peter Yantzer et al., "Foundation, Research Results and Application of
#     the New AGS Cut Grading System", American Gem Society Laboratories,
#     2005 -- the ASET zone definition reproduced here.
#   Jose M. Sasian, Peter Yantzer, Tom Tivol, "The Optical Design of
#     Gemstones", Optics & Photonics News 14(4), 2003 -- diffuse light
#     favours brilliance, localized light favours fire; and the 76-90 /
#     45-75 / 0-44 degree zones.
# --------------------------------------------------------------------------

    ASET_ZONES = ((0.0, 45.0, (0.05, 0.65, 0.10), "green low indirect"),
                  (45.0, 75.0, (0.85, 0.06, 0.06), "red bright zone"),
                  (75.0, 90.0, (0.05, 0.10, 0.80), "blue head obstruction"))

    def _emission(name, rgb, strength):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        em = nt.nodes.new('ShaderNodeEmission')
        em.inputs['Color'].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        em.inputs['Strength'].default_value = strength
        nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
        return m

    def _aset_dome(radius, rings, segs, skirt=12.0):
        """A hemisphere split into the three ASET elevation zones.

        ASET proper is defined over the hemisphere ABOVE the stone, 0 to
        90 degrees, and a dome springing from exactly z = 0 is therefore
        the literal reading.  It is also wrong in practice: a stone fitted
        to the 2 m cube hangs its pavilion some 0.6 below the girdle
        plane, so rays grazing up past the girdle meet nothing and the
        stone's outline comes back with black bites taken out of it.
        `skirt` carries the low zone a little below the horizon so that
        boundary is continuous.  It is a rendering convenience, not part
        of the AGS definition, which is why it is small and green -- the
        zone light at that elevation would belong to anyway.
        """
        import math as _m
        lo = -abs(skirt)
        elev = [lo + (90.0 - lo) * k / rings for k in range(rings + 1)]
        verts, faces, zone = [], [], []
        for e in elev:
            t = _m.radians(e)
            for s in range(segs):
                a = 2.0 * _m.pi * s / segs
                verts.append((radius * _m.cos(t) * _m.cos(a),
                              radius * _m.cos(t) * _m.sin(a),
                              radius * _m.sin(t)))
        for r in range(rings):
            mid = 0.5 * (elev[r] + elev[r + 1])
            z = 0
            for i, band in enumerate(ASET_ZONES):
                # the skirt below the horizon belongs to the low zone
                if (band[0] <= mid < band[1]
                        or (band[0] == 0.0 and mid < 0.0)
                        or (band[1] >= 90.0 and mid >= band[0])):
                    z = i
            for s in range(segs):
                a = r * segs + s
                b = r * segs + (s + 1) % segs
                c = (r + 1) * segs + (s + 1) % segs
                d = (r + 1) * segs + s
                faces.append((a, b, c, d))
                zone.append(z)
        return verts, faces, zone

    class MESH_OT_gem_aset_rig_add(bpy.types.Operator):
        """Add the AGS ASET hemisphere: green 0-45, red 45-75, blue 75-90"""
        bl_idname = "mesh.gem_aset_rig_add"
        bl_label = "ASET Rig"
        bl_options = {'REGISTER', 'UNDO'}

        radius: FloatProperty(name="Radius", default=8.0, min=2.0, max=60.0)
        strength: FloatProperty(name="Strength", default=6.0, min=0.1,
                                max=100.0)
        skirt: FloatProperty(
            name="Skirt", default=12.0, min=0.0, max=45.0,
            description="Degrees the low zone is carried BELOW the girdle "
                        "plane. ASET proper is the hemisphere above the "
                        "stone, but a pavilion hangs below it, so a dome "
                        "stopping at exactly zero leaves black notches "
                        "bitten out of the outline")
        add_camera: BoolProperty(
            name="Add Camera", default=True,
            description="Orthographic, straight down the table -- the view "
                        "an ASET image is made in")
        use_cycles: BoolProperty(
            name="Switch to Cycles", default=True,
            description="The dome is EMISSIVE GEOMETRY, not light objects, "
                        "because ASET is defined by where light comes from "
                        "rather than by lamps. Cycles treats emissive mesh "
                        "as a real source; EEVEE does not light a scene "
                        "from it, and renders the stone flat grey")

        def execute(self, context):
            verts, faces, zone = _aset_dome(self.radius, 24, 64,
                                            skirt=self.skirt)
            me = bpy.data.meshes.new("ASET Dome")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            for band in ASET_ZONES:
                me.materials.append(_emission("ASET " + band[3], band[2],
                                              self.strength))
            me.polygons.foreach_set('material_index', zone)
            me.flip_normals()          # it lights the stone from inside
            me.update()
            obj = bpy.data.objects.new("ASET Dome", me)
            context.collection.objects.link(obj)
            obj.visible_camera = False     # light the stone, do not film it
            obj["gem_rig"] = 'ASET'

            if self.add_camera:
                cam = bpy.data.cameras.new("ASET Camera")
                cam.type = 'ORTHO'
                cam.ortho_scale = 2.4
                co = bpy.data.objects.new("ASET Camera", cam)
                co.location = (0.0, 0.0, self.radius * 0.9)
                context.collection.objects.link(co)
                co["gem_rig"] = 'ASET'
                context.scene.camera = co
            if self.use_cycles and context.scene.render.engine != 'CYCLES':
                context.scene.render.engine = 'CYCLES'
                self.report({'INFO'}, "switched the scene to Cycles")
            self.report({'INFO'},
                        "ASET rig: green 0-45 deg, red 45-75, blue 75-90. "
                        "The dome is emissive GEOMETRY, so this view needs "
                        "Cycles and Rendered shading -- in EEVEE, or in "
                        "Solid view, the stone shows flat grey")
            return {'FINISHED'}

    class MESH_OT_gem_studio_add(bpy.types.Operator):
        """Add a lighting studio for gemstones: sky, key, fill and camera"""
        bl_idname = "mesh.gem_studio_add"
        bl_label = "Gem Studio"
        bl_options = {'REGISTER', 'UNDO'}

        key_size: FloatProperty(
            name="Key Size", default=0.35, min=0.01, max=10.0,
            description="Small sources make FIRE -- dispersed colour "
                        "flashes; broad ones make brilliance. The single "
                        "most important control over how a stone reads")
        key_energy: FloatProperty(name="Key Power", default=900.0, min=1.0,
                                  max=20000.0)
        fill_energy: FloatProperty(name="Fill Power", default=120.0, min=0.0,
                                   max=20000.0)
        sky_strength: FloatProperty(
            name="Sky", default=1.2, min=0.0, max=20.0,
            description="The environment IS most of a gem's appearance: a "
                        "stone with nothing to reflect renders black, and "
                        "that is physically correct")
        add_camera: BoolProperty(name="Add Camera", default=True)
        use_collection: BoolProperty(
            name="Group in a Collection", default=False,
            description="Lights and cameras cannot live inside a mesh "
                        "object, but a collection can hold the stone and "
                        "its rig together and be saved as an asset")

        def execute(self, context):
            import math as _m
            coll = context.collection
            if self.use_collection:
                coll = bpy.data.collections.new("Gem Studio")
                context.scene.collection.children.link(coll)

            w = bpy.data.worlds.new("Gem Studio World")
            w["gem_rig"] = 'STUDIO'
            context.scene.world = w
            w.use_nodes = True
            nt = w.node_tree
            for n in list(nt.nodes):
                nt.nodes.remove(n)
            out = nt.nodes.new('ShaderNodeOutputWorld')
            bg = nt.nodes.new('ShaderNodeBackground')
            bg.inputs['Strength'].default_value = self.sky_strength
            sky = nt.nodes.new('ShaderNodeTexSky')
            # NOTE: a Gradient texture driven by Generated coordinates does
            # NOT work in a world -- it renders flat and the stone comes out
            # black.  Sky Texture is in world space and does.
            try:
                sky.sky_type = 'MULTIPLE_SCATTERING'
            except TypeError:              # the enum differs across versions
                pass
            sky.sun_elevation = 0.9
            nt.links.new(sky.outputs['Color'], bg.inputs['Color'])
            nt.links.new(bg.outputs['Background'], out.inputs['Surface'])

            key = bpy.data.lights.new("Gem Key", 'AREA')
            key.energy = self.key_energy
            key.size = self.key_size
            ko = bpy.data.objects.new("Gem Key", key)
            coll.objects.link(ko)
            ko["gem_rig"] = 'STUDIO'
            ko.location = (3.2, -2.4, 4.2)
            ko.rotation_euler = (0.62, 0.32, 0.72)

            if self.fill_energy > 0.0:
                fill = bpy.data.lights.new("Gem Fill", 'AREA')
                fill.energy = self.fill_energy
                fill.size = 4.0
                fo = bpy.data.objects.new("Gem Fill", fill)
                coll.objects.link(fo)
                fo["gem_rig"] = 'STUDIO'
                fo.location = (-3.5, 2.0, 2.0)
                fo.rotation_euler = (-0.7, -0.4, -0.9)

            if self.add_camera:
                cam = bpy.data.cameras.new("Gem Camera")
                cam.lens = 70.0
                co = bpy.data.objects.new("Gem Camera", cam)
                coll.objects.link(co)
                co["gem_rig"] = 'STUDIO'
                co.location = (0.0, -4.6, 3.4)
                co.rotation_euler = (_m.radians(52.0), 0.0, 0.0)
                context.scene.camera = co

            self.report({'INFO'},
                        "Gem studio: small key for fire, broad fill for "
                        "brilliance, sky for the reflections. Cycles shows "
                        "the internal reflections; EEVEE approximates them")
            return {'FINISHED'}


    class MESH_OT_gem_cabochon_add(bpy.types.Operator):
        """Add a cabochon: a domed stone, cut on a curve rather than facets"""
        bl_idname = "mesh.gem_cabochon_add"
        bl_label = "Cabochon"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Shape",
            items=[(k, lbl, desc)
                   for k, lbl, desc in gem_curved.cabochon_items()],
            default='SINGLE')
        lw: FloatProperty(name="Length / Width", default=0.0, min=0.0,
                          max=4.0,
                          description="0 uses the preset's own outline")
        height: FloatProperty(
            name="Dome Height", default=0.0, min=0.0, max=2.0,
            description="As a fraction of the width; 0 uses the preset's. "
                        "Asterism and chatoyancy need a tall dome to "
                        "gather their reflections into a line")
        dome: FloatProperty(
            name="Dome Shape", default=0.0, min=0.0, max=6.0,
            description="The profile's superellipse exponent: below 2 is "
                        "pointed like a sugarloaf, 2 is a hemisphere, "
                        "above 2 is a flatter cabochon. 0 uses the preset")
        rings: IntProperty(name="Rings", default=24, min=3, max=200)
        segments: IntProperty(name="Segments", default=64, min=6, max=512)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        material: EnumProperty(
            name="Material",
            items=[(k, lbl, desc)
                   for k, lbl, desc in gem_mat.material_items()],
            default='RUBY',
            description="Cabochons are cut for material that facets would "
                        "waste -- opaque or translucent, or phenomenal")
        assign_material: BoolProperty(name="Assign Material", default=True)
        size_mm: FloatProperty(name="Size (mm)", default=8.0, min=0.1,
                               max=100.0)
        rig: EnumProperty(
            name="Viewing Rig", items=RIG_ITEMS, default='CUSTOM',
            description="Lighting to build around the stone. Custom leaves "
                        "the scene alone")

        def execute(self, context):
            cab = gem_curved.get(self.preset)
            if self.lw > 0.0:
                cab = cab._replace(lw=self.lw)
            if self.height > 0.0:
                cab = cab._replace(height=self.height)
            if self.dome > 0.0:
                cab = cab._replace(dome=self.dome)
            try:
                verts, faces = gem_curved.build_cabochon(
                    cab, rings=self.rings, segments=self.segments)
            except ValueError as e:
                self.report({'ERROR'}, f"could not build the cabochon: {e}")
                return {'CANCELLED'}

            me = bpy.data.meshes.new(cab.name)
            me.from_pydata(fit_cube(verts, span=2.0 * self.scale), [], faces)
            me.validate(clean_customdata=True)
            # A cabochon is a CURVED surface, so unlike every faceted cut
            # here it wants smooth shading -- flat shading would turn the
            # dome back into facets, which is the one thing it is not.
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(cab.name, me)
            obj["gem_cabochon"] = self.preset
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

            spec = gem_mat.get(self.material)
            if self.assign_material and gem_material is not None:
                try:
                    obj.data.materials.append(gem_material.gem_material(
                        self.material, size_mm=self.size_mm, bands=3))
                except Exception as e:
                    self.report({'WARNING'}, f"material not assigned: {e}")
            try:
                _apply_rig(context, self.rig)
            except Exception as e:
                self.report({'WARNING'}, f"rig not applied: {e}")
            self.report({'INFO'},
                        f"{cab.name}: {len(me.vertices)} vertices, "
                        f"{len(me.polygons)} faces, dome "
                        f"{cab.height * 100:.0f}% of the width, "
                        f"{spec.label}")
            return {'FINISHED'}

    ADD_MENU = True
    _CLASSES = (MESH_OT_gem_add, MESH_OT_gem_cabochon_add,
                MESH_OT_gem_studio_add,
                MESH_OT_gem_aset_rig_add,
                IMPORT_MESH_OT_gemcad_asc,
                EXPORT_MESH_OT_gemcad_asc)

    def register():
        for c in _CLASSES:
            bpy.utils.register_class(c)
        bpy.types.TOPBAR_MT_file_import.append(_import_menu)
        bpy.types.TOPBAR_MT_file_export.append(_export_menu)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.types.TOPBAR_MT_file_export.remove(_export_menu)
        bpy.types.TOPBAR_MT_file_import.remove(_import_menu)
        for c in reversed(_CLASSES):
            bpy.utils.unregister_class(c)

else:                                   # importable outside Blender
    ADD_MENU = False

    def register():
        pass

    def unregister():
        pass


def _selftest():
    ok = True

    D = catalogue.get_cut("SRB_GEMCAD")
    verts, faces, tiers, info = build_gem(D)

    good = info["closed"] and info["convex"] and len(faces) == 73
    ok &= good
    print(f"gem_generator: the Standard Round Brilliant builds to "
          f"{len(faces)} facets, closed and convex "
          f"{'OK' if good else 'did not build'}")

    good = len(tiers) == len(faces) and set(tiers) == set(range(7))
    ok &= good
    print(f"gem_generator: every facet carries its tier index, all 7 tiers "
          f"present {'OK' if good else 'tier attribute wrong'}")

    # the 2 m cube convention
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    off = max(abs(min(a) + max(a)) / 2.0 for a in (xs, ys, zs))
    good = abs(ext - 2.0) < 1e-9 and off < 1e-9
    ok &= good
    print(f"gem_generator: fitted to the 2 m cube (extent {ext:.9f}, "
          f"centre offset {off:.1e}) {'OK' if good else 'convention broken'}")

    # faces stay n-gons: the table is one polygon, not a fan
    sizes = sorted(len(f) for f in faces)
    good = max(sizes) >= 8 and len(faces) == 73
    ok &= good
    print(f"gem_generator: facets are n-gons, largest has {max(sizes)} "
          f"sides {'OK' if good else 'triangulated'}")

    # no vertex is used twice in one face, and every index is in range
    good = all(len(set(f)) == len(f) and all(0 <= i < len(verts) for i in f)
               for f in faces)
    ok &= good
    print(f"gem_generator: face index lists are clean "
          f"{'OK' if good else 'malformed'}")

    # the design survives being carried on the object as text
    good = parse_asc(write_asc(D)) == D
    ok &= good
    print(f"gem_generator: the design stored on the object round-trips "
          f"{'OK' if good else 'would be lost'}")

    good = _tier_names(D) == ["G", "1", "2", "A", "B", "C", "T"]
    ok &= good
    print(f"gem_generator: tier names come through for the face attribute "
          f"{'OK' if good else 'names lost'}")

    # scale is honoured
    v2, _, _, _ = build_gem(D, span=5.0)
    ext2 = max(max(a) - min(a) for a in zip(*v2))
    good = abs(ext2 - 5.0) < 1e-9
    ok &= good
    print(f"gem_generator: the span argument scales the result "
          f"(got {ext2:.6f}) {'OK' if good else 'ignored'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gem_generator self-test failed")
