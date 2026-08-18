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
    from .gems import (catalogue, design as gem_design, facets as gem_facets)
    from .gems.asc import parse_asc, write_asc
    from .polyhedra.fit import fit_cube
except ImportError:                     # flat import outside the package
    from gems import catalogue, design as gem_design, facets as gem_facets
    from gems.asc import parse_asc, write_asc
    from polyhedra.fit import fit_cube

try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, StringProperty
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

    def _report(op, cut_design, info, me):
        op.report({'INFO'},
                  f"{cut_design.name or 'gem'}: {len(me.polygons)} facets, "
                  f"{len(me.vertices)} vertices, "
                  f"meetpoint residual {info['residual']:.1e}"
                  + (f", {info['dropped']} facet(s) cut away"
                     if info["dropped"] else ""))

    class MESH_OT_gem_add(bpy.types.Operator):
        """Add a faceted gemstone cut from its facet planes"""
        bl_idname = "mesh.gem_add"
        bl_label = "Gemstone"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Cut",
            items=[(k, lbl, desc) for k, lbl, desc in catalogue.cut_items()],
            default=catalogue.cut_items()[0][0])
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)
        snap: FloatProperty(
            name="Meetpoint Tolerance", default=1e-5,
            min=1e-7, max=1e-3, precision=7,
            description="Radius, as a fraction of the girdle radius, "
                        "within which facet planes are taken to meet at "
                        "one point. Published designs carry eight "
                        "decimals, so the facets around a culet do not "
                        "quite concur; this is how close counts as "
                        "concurrent")

        def execute(self, context):
            try:
                D = catalogue.get_cut(self.preset)
                obj, info = _emit(context, D, 2.0 * self.scale, self.snap,
                                  catalogue.CUTS[self.preset].label)
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, f"could not build the cut: {e}")
                return {'CANCELLED'}
            if obj is None:
                self.report({'ERROR'},
                            "the facet planes do not close a solid; try a "
                            "larger meetpoint tolerance")
                return {'CANCELLED'}
            _report(self, D, info, obj.data)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
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
