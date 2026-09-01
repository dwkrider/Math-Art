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


def _crease_batches(obj, depsgraph=None):
    """Build one line batch per assignment code, in world space.

    POSITIONS COME FROM THE EVALUATED MESH, not the base one.  A folded
    pattern is deformed by shape keys, which leave `obj.data` sitting in
    the flat state -- so reading base coordinates draws the crease
    pattern lying flat underneath a folded model, which is exactly the
    wrong picture and was the first thing this overlay got wrong.

    Topology and the assignment attribute still come from the base mesh:
    shape keys do not change either, and reading them there avoids
    depending on which attributes survive evaluation.  If a modifier HAS
    changed the vertex count then the indices no longer correspond, so
    the base positions are used and the overlay simply sits where the
    unmodified cage is -- wrong-but-stable beats mismatched garbage.
    """
    me = obj.data
    attr = me.attributes.get("crease_assignment")
    if attr is None:
        return None

    coords = None
    if depsgraph is not None:
        try:
            eval_me = obj.evaluated_get(depsgraph).data
            if len(eval_me.vertices) == len(me.vertices):
                coords = [v.co.copy() for v in eval_me.vertices]
        except (AttributeError, ReferenceError, RuntimeError):
            coords = None
    if coords is None:
        coords = [v.co.copy() for v in me.vertices]

    mw = obj.matrix_world
    by_code = {}
    for e, d in zip(me.edges, attr.data):
        a, b = e.vertices
        by_code.setdefault(int(d.value), []).extend(
            (mw @ coords[a], mw @ coords[b]))
    return by_code


def _draw():
    # NO ON/OFF SWITCH.  The overlay only ever draws on a mesh carrying a
    # `crease_assignment` attribute, so on anything else it is already
    # invisible -- a toggle for it was a menu entry that did nothing
    # ninety-nine times out of a hundred.
    ctx = bpy.context
    if getattr(ctx, "scene", None) is None:
        return
    global _shader
    if _shader is None:
        _shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    width = float(getattr(scene, "math_art_crease_width", 2.0))
    gpu.state.line_width_set(width)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')

    try:
        dg = ctx.evaluated_depsgraph_get()
    except (AttributeError, RuntimeError):
        dg = None

    for obj in ctx.view_layer.objects:
        if obj.type != 'MESH' or not obj.visible_get():
            continue
        try:
            batches = _crease_batches(obj, dg)
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


_CLASSES = ()


def register():
    global _handle
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
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
    for attr in ("math_art_crease_boundary", "math_art_crease_width"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
