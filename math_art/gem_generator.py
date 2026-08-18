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
    from .gems.asc import parse_asc, write_asc
    from .polyhedra.fit import fit_cube
except ImportError:                     # flat import outside the package
    from gems import (catalogue, design as gem_design, facets as gem_facets,
                      measure as gem_measure)
    from gems.asc import parse_asc, write_asc
    from polyhedra.fit import fit_cube

try:
    import bpy
    from bpy.props import (EnumProperty, FloatProperty, IntProperty,
                           StringProperty)
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
        sg: FloatProperty(
            name="Specific Gravity", default=3.52, min=1.0, max=8.0,
            description="Of the intended material (diamond 3.52, corundum "
                        "4.00, quartz 2.65); sets the carat weight")
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
            pr, gr = info.get("proportions"), info.get("grades")
            if pr is not None:
                # keep the measurement with the stone, so it can be read
                # back without rebuilding
                obj["gem_proportions"] = {k: float(v) for k, v
                                          in pr._asdict().items()}
                obj["gem_carat"] = gem_measure.carat_from_diameter(
                    pr, self.size_mm, self.sg)
            if gr:
                obj["gem_idc_grade"] = gr["overall"]
            _report(self, D, info, obj.data, self.size_mm, self.sg)
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

    ADD_MENU = True
    _CLASSES = (MESH_OT_gem_add, IMPORT_MESH_OT_gemcad_asc,
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
