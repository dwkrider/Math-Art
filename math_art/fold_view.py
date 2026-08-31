# Seeing the crease pattern: mountain, valley, boundary, in colour.
#
# The viewport half of `math_art/crease/`.  Every imported or generated
# pattern already carries a `crease_assignment` attribute on its edge
# domain -- and until now nothing drew it, so a crease pattern was a grey
# mesh and the validator's warnings named vertex indices with no way to
# find them.  This module is the answer to both.
#
# WHY AN OVERLAY AND NOT GEOMETRY.  Colouring creases by building ribbon
# geometry would double the mesh, go stale the moment an edge is
# reassigned, and force a choice of ribbon width that is wrong at some
# zoom.  A GPU overlay draws the edges the mesh already has, reads the
# attribute at draw time, and costs nothing in the file.  The same
# discipline the curvature colouring uses: the data lives on the mesh,
# the appearance is derived.
#
# THE COLOUR CONVENTION is the origami one, not an arbitrary palette:
# mountain red, valley blue.  That is what every crease-pattern diagram
# in the literature uses and what ORIPA, Oriedita and Origami Simulator
# all draw, so a pattern imported from any of them looks like itself
# here.  Boundary is neutral, flat and unassigned are dimmed -- they are
# marks on the paper, not folds.
#
# References:
#   E. D. Demaine, J. S. Ku, R. J. Lang, "A New File Standard to
#       Represent Folded Structures," 2016 -- the M/V/F/U/B assignments
#       being drawn.
#   T. C. Hull, "Origametry" (Cambridge, 2020) -- the flat-foldability
#       conditions whose failures this view locates.

import bpy
import gpu
from bpy.props import BoolProperty, FloatProperty
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

# Mountain red, valley blue -- the standard diagram convention.
_COLOURS = {
    0: (0.85, 0.15, 0.15, 1.0),   # M  mountain
    1: (0.15, 0.35, 0.85, 1.0),   # V  valley
    2: (0.55, 0.55, 0.55, 0.7),   # F  flat: marked, not folded
    3: (0.75, 0.55, 0.15, 0.8),   # U  unassigned
    4: (0.15, 0.15, 0.15, 1.0),   # B  boundary
}

_handle = None
_shader = None


def _crease_batches(obj):
    """Build one line batch per assignment code, in world space."""
    me = obj.data
    attr = me.attributes.get("crease_assignment")
    if attr is None:
        return None
    mw = obj.matrix_world
    # Edge order matches the attribute's edge domain, so they zip.
    by_code = {}
    for e, d in zip(me.edges, attr.data):
        a, b = e.vertices
        by_code.setdefault(int(d.value), []).extend(
            (mw @ me.vertices[a].co, mw @ me.vertices[b].co))
    return by_code


def _draw():
    ctx = bpy.context
    scene = getattr(ctx, "scene", None)
    if scene is None or not getattr(scene, "math_art_show_creases", False):
        return
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    width = float(getattr(scene, "math_art_crease_width", 2.0))
    gpu.state.line_width_set(width)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')

    for obj in ctx.view_layer.objects:
        if obj.type != 'MESH' or not obj.visible_get():
            continue
        try:
            batches = _crease_batches(obj)
        except (ReferenceError, AttributeError):
            continue                       # object went away mid-draw
        if not batches:
            continue
        for code, coords in batches.items():
            if not coords:
                continue
            if code == 4 and not getattr(scene, "math_art_crease_boundary",
                                         True):
                continue
            batch = batch_for_shader(_shader, 'LINES', {"pos": coords})
            _shader.uniform_float("color", _COLOURS.get(code, _COLOURS[3]))
            batch.draw(_shader)

    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('NONE')


class VIEW3D_OT_crease_view(bpy.types.Operator):
    """Show mountain and valley creases in the viewport"""

    bl_idname = "view3d.crease_view"
    bl_label = "Crease Pattern View"
    bl_options = {'REGISTER'}

    enable: BoolProperty(
        name="Show Creases", default=True,
        description="Draw mountain red and valley blue over every mesh "
                    "carrying a crease assignment")

    def execute(self, context):
        context.scene.math_art_show_creases = bool(self.enable)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        n = sum(1 for o in context.view_layer.objects
                if o.type == 'MESH'
                and o.data.attributes.get("crease_assignment") is not None)
        self.report({'INFO'},
                    f"crease view {'on' if self.enable else 'off'}; "
                    f"{n} pattern(s) in view")
        return {'FINISHED'}


class OBJECT_OT_select_bad_vertices(bpy.types.Operator):
    """Select the vertices that fail the flat-foldability checks"""

    bl_idname = "object.select_bad_creases"
    bl_label = "Select Failing Vertices"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and obj.data.attributes.get("crease_assignment") is not None)

    def execute(self, context):
        try:
            from . import crease
        except ImportError:
            import crease
        import numpy as np

        obj = context.active_object
        me = obj.data
        code_to_char = {0: "M", 1: "V", 2: "F", 3: "U", 4: "B"}
        attr = me.attributes["crease_assignment"]
        frame = crease.Frame(
            verts=np.array([(v.co.x, v.co.y, v.co.z) for v in me.vertices]),
            edges=np.array([tuple(e.vertices) for e in me.edges],
                           dtype=np.int64).reshape(-1, 2),
            assignment=np.array([code_to_char.get(int(d.value), "U")
                                 for d in attr.data], dtype="<U1"),
            faces=[list(p.vertices) for p in me.polygons] or None)
        rep = crease.check(frame)
        if not rep.checked:
            self.report({'WARNING'},
                        "not a flat crease pattern, so the checks were "
                        "skipped -- unfold it first")
            return {'CANCELLED'}

        bad = rep.vertices()
        # A report you cannot act on is a dead end, so put the answer in
        # the selection rather than only in the status bar.
        bpy.ops.object.mode_set(mode='OBJECT')
        for v in me.vertices:
            v.select = False
        for e in me.edges:
            e.select = False
        for p in me.polygons:
            p.select = False
        for i in bad:
            me.vertices[i].select = True
        me.update()
        if bad:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='VERT')

        self.report({'INFO'} if not bad else {'WARNING'}, rep.summary())
        return {'FINISHED'}


_CLASSES = (VIEW3D_OT_crease_view, OBJECT_OT_select_bad_vertices)


def register():
    global _handle
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.math_art_show_creases = BoolProperty(
        name="Show Creases", default=True,
        description="Draw crease patterns with mountain red, valley blue")
    bpy.types.Scene.math_art_crease_boundary = BoolProperty(
        name="Show Boundary", default=True,
        description="Include the sheet's boundary edges in the overlay")
    bpy.types.Scene.math_art_crease_width = FloatProperty(
        name="Crease Width", default=2.0, min=1.0, max=8.0,
        description="Line width of the crease overlay")
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
    for attr in ("math_art_show_creases", "math_art_crease_boundary",
                 "math_art_crease_width"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
