
# Turning a TurtleOutput into Blender data.
#
# This is the ONLY module in the package that imports bpy.  Everything
# else is pure Python + numpy so that `tests/test_selftests.py` can
# exercise the numerics headlessly, and so `import lsystem` never drags
# Blender in.  The package `__init__` deliberately does not import this
# module for the same reason -- the operator imports `lsystem.emit`
# explicitly.
#
# Three emitters, matching the three things a turtle produces:
#
#   emit_curve      strands -> one CURVE object of POLY splines, with the
#                   per-point `radius` carrying the `!` taper and a round
#                   bevel giving solid, printable limbs.  This is the
#                   mechanism `fractal_tree_generator` already uses.
#   emit_polygons   the { . } captures -> a filled MESH, fan-triangulated
#                   about each loop's centroid so leaves and petals come
#                   out as real, solidifiable blades rather than wire.
#   emit_instances  the ~ placements -> collection instances on the
#                   turtle frame, for real organ meshes.
#
# Colour indices from `'` and `;` become a `colour_index` INT attribute
# on the POINT domain, so a material can read them without a second pass.

import bpy
import numpy as np

from .polyline import round_corners, split_at_reversals


def emit_curve(out, name="L-system", radius=0.01, resolution=2,
               closed=False, taper=True, split_reversals=True, tip=0.0,
               fillet=0.25, fillet_segments=4):
    """A CURVE object with POLY splines.

    `radius` is the bevel depth at width 1.0; per-point radii carry the
    relative taper, which is how `!` becomes visible geometry.

    `split_reversals` breaks each strand at any exact 180-degree turn --
    see `split_at_reversals` for why the bevel needs it.  Disabled
    automatically for a closed figure, where the caller wants one cyclic
    spline.

    `fillet` rounds the corners so the bevel does not pinch at them --
    see `round_corners`.  0 keeps the corners mathematically exact.
    """
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    widths = _width_range(out)
    for s in out.strands:
        if len(s.points) < 2:
            continue
        spans = ([(0, len(s.points))] if not split_reversals
                 else split_at_reversals(s.points))
        # A closed figure normally wants one cyclic spline -- but some
        # closed grammars retrace themselves (Pentaplexity turns a full
        # 180 with `|`), and a reversal inside a cyclic spline gets
        # neither the split nor the fillet, leaving the degenerate frame
        # the split exists to remove.  If the figure really does double
        # back, prefer well-formed open pieces over a broken loop.
        cyclic = closed and len(spans) == 1
        for a, b in spans:
            seg, segw = s.points[a:b], s.widths[a:b]
            if len(seg) < 2:
                continue
            seg, segw = round_corners(seg, segw, fillet, fillet_segments,
                                      closed=cyclic)
            sp = cu.splines.new('POLY')
            sp.points.add(len(seg) - 1)
            for i, p in enumerate(seg):
                sp.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
                if taper:
                    sp.points[i].radius = _rel_width(segw[i], widths, tip)
            sp.use_cyclic_u = cyclic
    cu.bevel_depth = float(radius)
    cu.bevel_resolution = int(resolution)
    obj = bpy.data.objects.new(name, cu)
    return obj


def emit_mesh(out, name="L-system", width_attr=True):
    """A MESH of the strand polylines as edges, with per-vertex width and
    colour attributes.  Useful as a skeleton for the Skin modifier, or
    for Strahler styling."""
    verts, edges, widths, colours, orders = [], [], [], [], []
    for s in out.strands:
        base = len(verts)
        for i, p in enumerate(s.points):
            verts.append((float(p[0]), float(p[1]), float(p[2])))
            widths.append(float(s.widths[i]))
            colours.append(int(s.colours[i]))
            orders.append(int(s.order))
        for i in range(len(s.points) - 1):
            edges.append((base + i, base + i + 1))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, edges, [])
    me.update()
    if width_attr and verts:
        _float_attr(me, "width", widths)
        _int_attr(me, "colour_index", colours)
        _int_attr(me, "strahler", orders)
    return bpy.data.objects.new(name, me)


def emit_polygons(out, name="L-system Blades"):
    """The { . } polygon captures as a filled MESH.

    Each loop is fan-triangulated about its own centroid.  A fan is used
    rather than an ear-clip because the loops the turtle produces are the
    outlines of leaves and petals -- star-shaped about their centroid --
    and a fan is both robust and gives an evenly distributed topology for
    a later Solidify.
    """
    verts, faces, cols = [], [], []
    for pts, colour in out.polygons:
        if len(pts) < 3:
            continue
        base = len(verts)
        centre = pts.mean(axis=0)
        verts.append(tuple(float(c) for c in centre))
        for p in pts:
            verts.append((float(p[0]), float(p[1]), float(p[2])))
        n = len(pts)
        for i in range(n):
            faces.append((base, base + 1 + i, base + 1 + (i + 1) % n))
            cols.append(int(colour))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    if faces:
        _int_attr(me, "colour_index", cols, domain='FACE')
    return bpy.data.objects.new(name, me)


def emit_instances(out, source, name="L-system Organs"):
    """Empties parented to `source` (an object or collection) at each `~`
    placement, oriented on the turtle frame."""
    objs = []
    for i, (nm, mx, scale) in enumerate(out.placements):
        e = bpy.data.objects.new(f"{name}.{i:04d}", None)
        e.empty_display_type = 'PLAIN_AXES'
        e.matrix_world = [list(row) for row in mx]
        e.scale = (scale, scale, scale)
        if source is not None:
            e.instance_type = 'COLLECTION' if hasattr(
                source, "objects") else 'OBJECT'
            if e.instance_type == 'COLLECTION':
                e.instance_collection = source
        objs.append(e)
    return objs


# --- helpers ---------------------------------------------------------

def _width_range(out):
    ws = [s.widths for s in out.strands if len(s.widths)]
    if not ws:
        return (1.0, 1.0)
    allw = np.concatenate(ws)
    lo, hi = float(allw.min()), float(allw.max())
    return (lo, hi if hi > lo else lo + 1e-9)


def _rel_width(w, rng, tip=0.0):
    """Per-point bevel radius, with the widest strand at 1.0.

    `tip` compresses the range into [tip, 1.0].  It is a DISPLAY choice,
    not a change to the width law: da Vinci's rule over a twelve-level
    tree legitimately spans a 300:1 range, so at the default tube radius
    the outer branches come out at 0.00003 m -- well under a pixel, which
    reads as a sparse cloud of dots rather than a tree.  Raising `tip`
    keeps the taper's ORDER while making the fine branches visible.
    """
    lo, hi = rng
    if hi <= lo:
        return 1.0
    r = max(float(w) / hi, 1e-6)
    tip = min(max(float(tip), 0.0), 1.0)
    if tip > 0.0:
        r = tip + (1.0 - tip) * r
    return r


def _float_attr(me, name, values, domain='POINT'):
    try:
        a = me.attributes.new(name=name, type='FLOAT', domain=domain)
        a.data.foreach_set("value", values)
    except Exception:
        pass                      # older Blender, or a name clash


def _int_attr(me, name, values, domain='POINT'):
    try:
        a = me.attributes.new(name=name, type='INT', domain=domain)
        a.data.foreach_set("value", values)
    except Exception:
        pass


def link(context, objs, active=None):
    """Link objects into the scene, select them, make one active."""
    if not isinstance(objs, (list, tuple)):
        objs = [objs]
    objs = [o for o in objs if o is not None]
    for o in objs:
        context.collection.objects.link(o)
    for o in context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    if objs:
        context.view_layer.objects.active = active or objs[0]
    return objs
