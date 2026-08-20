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

    if frame.assignment is not None and len(me.edges) == len(edges):
        attr = me.attributes.new("crease_assignment", 'INT', 'EDGE')
        attr.data.foreach_set(
            "value", [_ASSIGN_CODE.get(str(a), 3) for a in frame.assignment])
    if frame.fold_angle is not None and len(me.edges) == len(edges):
        import math
        vals = [0.0 if (a != a) else float(a)      # NaN -> 0
                for a in frame.fold_angle]
        attr = me.attributes.new("fold_angle", 'FLOAT', 'EDGE')
        attr.data.foreach_set("value", vals)

    me.update()
    return me, extent


class MESH_OT_fold_import(bpy.types.Operator, ImportHelper):
    """Import a FOLD crease pattern or folded state"""

    bl_idname = "mesh.fold_import"
    bl_label = "Crease Pattern (.fold)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".fold"
    filter_glob: StringProperty(default="*.fold;*.json", options={'HIDDEN'})

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
        except crease.FoldError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

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
