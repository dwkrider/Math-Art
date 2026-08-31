# Import a FOLD crease pattern or folded state into Blender.
#
# The operator over `math_art/crease/`, which holds the mathematics and
# the file handling.  This module is only the Blender skin: read a file,
# build a mesh, tag the creases, and report what the checks found.
#
# WHY IMPORT RATHER THAN DESIGN.  FOLD is the interchange format the
# whole computational-origami ecosystem speaks -- ORIPA, Oriedita,
# Origami Simulator, Flat-Folder, Rabbit Ear, Tachi's tools.  Reading it
# is what lets this add-on concentrate on the half nobody else covers:
# folding, thickening, materialising and rendering.  Designing crease
# patterns is a solved and crowded problem; rendering them is not.
#
# WHAT LANDS IN THE MESH.  Vertices and edges as drawn, faces recovered
# from the plane graph when the file omits them (many editors write only
# lines), and two named attributes carried on the edge domain:
#
#     crease_assignment   int, 0..4 for M V F U B
#     fold_angle          float, radians, signed (valley positive)
#
# They are attributes rather than baked geometry so a later pass can
# re-read them without inferring anything, and so a fold solver can
# write back to the same mesh.  The same discipline the curvature
# colouring uses.
#
# References:
#   E. D. Demaine, J. S. Ku, R. J. Lang, "A New File Standard to
#       Represent Folded Structures," 26th Fall Workshop on
#       Computational Geometry / CG:YRF, 2016 -- the FOLD format.
#   T. C. Hull, "Origametry: Mathematical Methods in Paper Folding"
#       (Cambridge, 2020) -- the flat-foldability conditions reported
#       after import.

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

try:
    from . import crease
except ImportError:                                   # headless import
    import crease


# Attribute encoding for the five FOLD assignments.  Stored as ints
# because Blender has no string attribute domain on edges.
_ASSIGN_CODE = {"M": 0, "V": 1, "F": 2, "U": 3, "B": 4}

_ASSIGN_ITEMS = (
    ("M", "Mountain", "Folds away from the viewer"),
    ("V", "Valley", "Folds towards the viewer"),
    ("F", "Flat", "Marked, but not folded"),
    ("U", "Unassigned", "Undecided"),
    ("B", "Boundary", "The edge of the sheet"),
)


def _build_mesh(name, frame, sheet_size):
    """Turn a crease.Frame into a Blender mesh with crease attributes."""
    verts = frame.verts
    if verts is None:
        raise crease.FoldError(
            "this frame has no vertices_coords, so there is nothing to "
            "place; it describes an abstract fold, not an embedded one")

    # FOLD carries no unit; files are conventionally in sheet units.
    # Scale the sheet's longest side to `sheet_size` and keep the real
    # extent on the object so a later fabrication pass can recover it.
    xy = verts[:, :2]
    extent = float(max(xy.max(axis=0) - xy.min(axis=0))) or 1.0
    scale = sheet_size / extent
    centre = 0.5 * (xy.max(axis=0) + xy.min(axis=0))

    co = []
    for v in verts:
        x = (v[0] - centre[0]) * scale
        y = (v[1] - centre[1]) * scale
        z = (v[2] * scale) if frame.dim >= 3 else 0.0
        co.append((x, y, z))

    edges = [] if frame.edges is None else [tuple(int(i) for i in e)
                                            for e in frame.edges]
    faces = [] if frame.faces is None else [list(f) for f in frame.faces]

    me = bpy.data.meshes.new(name)
    # Faces already reference the same vertices as the edges, so give
    # from_pydata the faces and let it derive the rest; passing both
    # would duplicate every face edge.
    me.from_pydata(co, edges if not faces else [], faces)
    me.update()

    # MAP BY VERTEX PAIR, NOT BY INDEX.
    #
    # These used to be written positionally, guarded by
    # `len(me.edges) == len(edges)` -- and when that guard failed the
    # assignment was dropped in silence, leaving a mesh with no creases
    # and, much later, "this mesh carries no crease_assignment
    # attribute" from the folder.  The guard fails routinely: passing
    # faces makes `from_pydata` derive the edges from them, so any
    # crease bordering no face simply is not in the mesh.  A real file
    # hit it -- Origami Simulator's `6ptHypar-anti.svg` has 500 creases
    # of which 499 border a face, and lost all 500.
    #
    # By pair, a count mismatch is no longer fatal: every edge the mesh
    # does have gets its true assignment, and anything unmatched falls
    # back to unassigned rather than to nothing at all.
    if frame.assignment is not None:
        want = {}
        for k, (a, b) in enumerate(frame.edges):
            code = _ASSIGN_CODE.get(str(frame.assignment[k]), 3)
            want[(int(a), int(b))] = code
            want[(int(b), int(a))] = code
        attr = me.attributes.new("crease_assignment", 'INT', 'EDGE')
        attr.data.foreach_set(
            "value", [want.get(tuple(e.vertices), 3) for e in me.edges])
    if frame.fold_angle is not None:
        ang = {}
        for k, (a, b) in enumerate(frame.edges):
            v = float(frame.fold_angle[k])
            v = 0.0 if v != v else v               # NaN -> 0
            ang[(int(a), int(b))] = v
            ang[(int(b), int(a))] = v
        attr = me.attributes.new("fold_angle", 'FLOAT', 'EDGE')
        attr.data.foreach_set(
            "value", [ang.get(tuple(e.vertices), 0.0) for e in me.edges])

    me.update()
    return me, extent


class MESH_OT_fold_import(bpy.types.Operator, ImportHelper):
    """Import a FOLD crease pattern or folded state"""

    bl_idname = "mesh.fold_import"
    bl_label = "Crease Pattern (.fold/.svg)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".fold"
    # SVG belongs here rather than behind a second menu entry: a user
    # with a crease pattern does not care which interchange format it
    # happens to be in, and the reference libraries are split across
    # both -- Origami Simulator ships its patterns as SVG only.
    filter_glob: StringProperty(default="*.fold;*.json;*.svg",
                                options={'HIDDEN'})

    frame_index: FloatProperty(
        name="Frame", default=0, min=0, max=4096, precision=0,
        description="Which frame of the file to build; a folding motion "
                    "stores its key frames after the first")
    sheet_size: FloatProperty(
        name="Sheet Size", default=2.0, min=0.001, max=1000.0, unit='LENGTH',
        description="Longest side of the imported sheet")
    recover_faces: BoolProperty(
        name="Recover Faces", default=True,
        description="Build faces from the crease graph when the file "
                    "lists only vertices and edges")
    triangulate_cells: BoolProperty(
        name="Triangulate Cells", default=False,
        description="Split every non-triangular panel with a diagonal, "
                    "left unassigned so the solver may bend there. A "
                    "rigid solver holds a quad panel flat, so patterns "
                    "whose panels must bend -- the pleated hypar above "
                    "all -- will not fold at all without this")
    report_checks: BoolProperty(
        name="Report Checks", default=True,
        description="Check developability, Maekawa and Kawasaki after "
                    "import and report any vertices that fail")

    def execute(self, context):
        try:
            frame, rep, order = crease.load_pattern(
                self.filepath,
                frame=int(self.frame_index),
                recover_faces=self.recover_faces,
                validate_it=self.report_checks)
        except (crease.FoldError, crease.SvgError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        # TRIANGULATE BEFORE BUILDING THE MESH, so the diagonals are
        # real creases in the frame rather than a mesh-only change the
        # solver never sees.  They go in UNASSIGNED, not FLAT: FLAT
        # means "creased but not folded" and the solver freezes it, so
        # a flat-tagged diagonal leaves the panel exactly as rigid as
        # the quad it replaced -- which defeats the whole option.
        # UNASSIGNED says what is actually true: nobody decided, and the
        # solver may choose.
        tri_note = None
        if self.triangulate_cells and frame.faces:
            # A face that fan triangulation cannot handle must not take
            # the whole import down with it.  Real files contain them:
            # `6ptHypar-anti.svg` recovers a face that visits one vertex
            # twice -- a pinch point, not a simple polygon -- and the
            # triangulator rightly refuses it.  Import the pattern
            # untriangulated and say so, rather than failing after the
            # file has been read and parsed.
            try:
                tris, diags = crease.triangulate(frame.verts, frame.faces)
            except crease.GraphError as exc:
                tris, diags = None, None
                tri_note = f"could not triangulate: {exc}"
            if diags:
                import numpy as _np
                frame.faces = tris
                frame.edges = _np.vstack(
                    [frame.edges, _np.array(diags, dtype=_np.int64)])
                frame.assignment = _np.concatenate(
                    [frame.assignment,
                     _np.array([crease.UNASSIGNED] * len(diags), dtype="<U1")])
                frame.meta = dict(frame.meta or {})
                frame.meta["triangulated"] = len(diags)

        import os
        name = os.path.splitext(os.path.basename(self.filepath))[0] or "Fold"
        try:
            me, extent = _build_mesh(name, frame, self.sheet_size)
        except crease.FoldError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        # Keep what the file could not: its own units, and whether the
        # frame we built was the flat sheet or a folded state.
        obj["fold_sheet_extent"] = extent
        obj["fold_is_flat"] = bool(frame.is_flat)

        msgs = [f"{frame.n_verts} vertices, {frame.n_edges} edges, "
                f"{frame.n_faces} faces"]
        if not frame.is_flat:
            msgs.append("folded state")
        if order:
            msgs.append(f"{len(order)} layer relations"
                        + ("" if order.stacking() is not None
                           else " (CYCLIC)"))
        # What the SVG reader had to drop, and why.  An importer that
        # discards a third of a file without saying so is worse than one
        # that refuses it: `hypar.svg` alone contributes 94 annotation
        # strokes that must not become creases, and the user should be
        # able to tell that from a mistake.
        st = frame.meta.get("import_stats") if frame.meta else None
        if st:
            msgs.append(crease.svg_io.stats_summary(st))
        if frame.meta and frame.meta.get("triangulated"):
            msgs.append(f"{frame.meta['triangulated']} panel diagonal(s) "
                        f"added, unassigned")
        if tri_note:
            msgs.append(tri_note)
        if rep is not None and rep.checked:
            msgs.append(rep.summary())
        elif rep is not None:
            msgs.append("not a flat pattern, so the fold checks were skipped")
        self.report({'INFO'} if (rep is None or rep) else {'WARNING'},
                    "; ".join(msgs))
        return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(MESH_OT_fold_import.bl_idname,
                         text="Crease Pattern (.fold)")


_CLASSES = (MESH_OT_fold_import,)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
