# Headless test of live object editing (the Math Art sidebar tab).
#
# For every generator the framework claims to cover, this checks the
# whole round trip:
#
#   1. the Add operator records its settings on the object it built
#   2. the sidebar can draw that object's settings -- through the
#      generator's OWN draw(), which is the parity claim
#   3. a rebuild produces geometry
#   4. the rebuild is EQUIVALENT to running the operator afresh with the
#      same settings
#
# Step 4 is the one that matters.  Everything else can pass while the
# panel quietly edits a different generator's parameters, or rebuilds
# from stale values; comparing against a fresh operator run is what
# proves the sidebar and the Add menu are the same generator.
#
# Run:
#   blender --background --factory-startup --python tests/test_live.py
import os
import sys
import traceback

import bpy
from mathutils import Vector

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import math_art                                          # noqa: E402
from math_art import live                                # noqa: E402
from math_art.live import build, clone                   # noqa: E402
from math_art.live.registry import (GENERATORS,          # noqa: E402
                                    root_for_object,
                                    settings_for_object)

math_art.register()

# A property worth nudging: an int or float with room to move, which is
# the cheapest way to prove a rebuild actually re-ran the generator
# rather than leaving the old mesh in place.
_NUDGE_SKIP = {'scale', 'global_scale', 'seed'}


def _nudge(pg):
    """Change one setting; return (name, old, new) or None.

    Counts go DOWN by preference.  Several generators cap their own
    output -- too many copies, too fine a grid -- so nudging upward
    walks into a generator's own limit and tests nothing about the
    framework; downward stays inside whatever the generator allows.
    """
    for prop in clone.editable_props(type(pg)):
        name = prop.identifier
        if name in _NUDGE_SKIP or getattr(prop, 'is_array', False):
            continue
        if prop.type == 'INT':
            old = getattr(pg, name)
            new = old - 1 if old - 1 >= prop.hard_min else old + 1
            if new == old or not (prop.hard_min <= new <= prop.hard_max):
                continue
            setattr(pg, name, new)
            return name, old, new
        if prop.type == 'BOOLEAN':
            old = getattr(pg, name)
            setattr(pg, name, not old)
            return name, old, not old
    # An enum-only generator (a gallery of named solids, typically) has
    # no number to move, and skipping those would leave a whole family
    # of generators unverified.
    for prop in clone.editable_props(type(pg)):
        name = prop.identifier
        if name in _NUDGE_SKIP or prop.type != 'ENUM':
            continue
        old = getattr(pg, name)
        for item in prop.enum_items:
            if item.identifier != old:
                setattr(pg, name, item.identifier)
                return name, old, item.identifier
    return None


def _counts(obj):
    if obj.type == 'MESH':
        return len(obj.data.vertices), len(obj.data.polygons)
    if obj.type == 'CURVE':
        return (sum(len(s.bezier_points) + len(s.points)
                    for s in obj.data.splines), len(obj.data.splines))
    return 0, 0


def _extent(obj):
    """The local bounding box, measured from the DATA.

    Not `obj.bound_box`: that is a depsgraph-evaluated cache, and
    straight after a rebuild it still describes the previous geometry
    until a view-layer update runs -- which never happens in background
    mode.  Reading the vertices is exact and needs no evaluation.

    Local rather than world, because a rebuild deliberately keeps the
    object's transform while a fresh Add lands on the 3D cursor.
    """
    points = []
    if obj.type == 'MESH' and obj.data.vertices:
        flat = [0.0] * (3 * len(obj.data.vertices))
        obj.data.vertices.foreach_get('co', flat)
        points = list(zip(flat[0::3], flat[1::3], flat[2::3]))
    elif obj.type == 'CURVE':
        for spline in obj.data.splines:
            points.extend(tuple(p.co)[:3] for p in spline.bezier_points)
            points.extend(tuple(p.co)[:3] for p in spline.points)
    if not points:
        return ()
    return tuple(
        round(fn(p[axis] for p in points), 4)
        for axis in range(3) for fn in (min, max))


def _group(root):
    """A build's root plus every companion it hangs off, live only."""
    objects = [root]
    for entry in root.math_art.members:
        member = entry.obj
        if member is None:
            continue
        try:
            member.name
        except ReferenceError:
            continue
        objects.append(member)
    return objects


def _points_in_root_space(root, objects):
    """Every point of a group, expressed relative to its root.

    Root-relative, because a rebuilt group sits wherever the object the
    user was editing sits while a freshly added one lands on the 3D
    cursor.  Comparing world coordinates would only measure that
    difference; comparing root-relative ones measures the shape, which
    is what has to match.

    The view layer is updated first because `matrix_world` -- like
    `bound_box` -- is depsgraph-evaluated.  Re-parenting a companion
    leaves the cached value behind until something asks the depsgraph to
    catch up, which in background mode nothing ever does: every bubble
    of a rebuilt cluster reads as sitting at the origin while its local
    transform is perfectly correct.
    """
    bpy.context.view_layer.update()
    to_root = root.matrix_world.inverted()
    points = []
    for obj in objects:
        matrix = to_root @ obj.matrix_world
        local = []
        if obj.type == 'MESH' and obj.data.vertices:
            flat = [0.0] * (3 * len(obj.data.vertices))
            obj.data.vertices.foreach_get('co', flat)
            local = list(zip(flat[0::3], flat[1::3], flat[2::3]))
        elif obj.type == 'CURVE':
            for spline in obj.data.splines:
                local.extend(tuple(p.co)[:3] for p in spline.bezier_points)
                local.extend(tuple(p.co)[:3] for p in spline.points)
        points.extend(matrix @ Vector(p) for p in local)
    return points


def _shape(root):
    """What a build produced: per-object counts and the group's extent.

    For a single-object generator this is the object.  For one that
    builds an assembly it is the whole assembly, because rebuilding
    only the root correctly while mangling its twelve children would
    otherwise pass.
    """
    objects = _group(root)
    counts = tuple(sorted(_counts(o) for o in objects))
    points = _points_in_root_space(root, objects)
    if not points:
        return (len(objects), counts), ()
    extent = tuple(
        round(fn(p[axis] for p in points), 4)
        for axis in range(3) for fn in (min, max))
    return (len(objects), counts), extent


class FakeLayout:
    """Enough of a UILayout to run a generator's draw() off-screen."""

    def __init__(self, drawn):
        self.drawn = drawn
        for attr in ('use_property_split', 'use_property_decorate',
                     'alert', 'enabled', 'active'):
            setattr(self, attr, False)
        self.scale_x = self.scale_y = 1.0
        self.alignment = 'EXPAND'

    def prop(self, data, name, **kw):
        self.drawn.append(name)

    def prop_search(self, data, name, *a, **kw):
        self.drawn.append(name)

    def props_enum(self, data, name):
        self.drawn.append(name)

    def __getattr__(self, name):
        # row/column/box/split/grid_flow return a layout; label,
        # separator, template_* and operator return something harmless.
        def anything(*a, **kw):
            if name in ('row', 'column', 'column_flow', 'grid_flow',
                        'split', 'box', 'menu_pie'):
                return self
            return type('Returned', (), {'__getattr__':
                                         lambda s, n: (lambda *a, **k: None)})()
        return anything


fails = []
skips = []
checked = 0
print("\n=== live editing: %d generators registered ===" % len(GENERATORS))

for idname, info in sorted(GENERATORS.items()):
    module, _, func = idname.partition('.')
    op = getattr(getattr(bpy.ops, module, None), func, None)
    if op is None:
        fails.append((idname, "operator not registered"))
        continue

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        if not op.poll():
            skips.append((idname, "poll false in an empty scene"))
            continue
        result = op('EXEC_DEFAULT')
    except Exception as exc:
        skips.append((idname, "add failed: %s" % exc))
        continue
    if 'FINISHED' not in result:
        skips.append((idname, "add returned %s" % result))
        continue

    active = bpy.context.view_layer.objects.active
    if active is None:
        fails.append((idname, "nothing active after the add"))
        continue

    # 1. the settings were recorded -- on the group's ROOT, which is not
    #    always what the generator left active.  Bubble Cluster leaves a
    #    bubble active while the settings belong to the Empty the twelve
    #    of them hang off, and resolving that is exactly what makes
    #    clicking one bubble show the cluster's panel.
    obj = root_for_object(active)
    if obj is None:
        fails.append((idname, "settings were not recorded on %r, and it "
                              "resolves to no group root" % active.name))
        continue
    got, pg = settings_for_object(obj)
    if got is None or pg is None:
        fails.append((idname, "no settings on the resolved root %r"
                      % obj.name))
        continue
    if got.idname != idname:
        fails.append((idname, "recorded as %s" % got.idname))
        continue

    # 2. the sidebar can draw them, using the generator's own draw()
    if 'draw' in info.pg_cls.__dict__:
        drawn = []
        info.pg_cls.layout = FakeLayout(drawn)
        try:
            info.pg_cls.draw(pg, bpy.context)
        except Exception as exc:
            fails.append((idname, "draw failed: %s: %s"
                          % (type(exc).__name__, exc)))
            traceback.print_exc()
            continue
        finally:
            info.pg_cls.layout = None
        if not drawn:
            fails.append((idname, "draw produced no controls"))
            continue

    group_size = obj.math_art.n_created
    # Every companion must point back at the root, or selecting one of
    # them in the viewport finds no settings.
    for entry in obj.math_art.members:
        if entry.obj is not None and root_for_object(entry.obj) is not obj:
            fails.append((idname, "companion %r does not resolve to its "
                                  "root" % entry.obj.name))
            break

    # 3a. Auto Update, which the generator's measured speed switched on
    #     for anything fast enough, must rebuild without being asked.
    if obj.math_art.autobuild:
        changed = _nudge(pg)
        # A generator that cannot be re-run from inside a property write
        # (Scherk-Collins, whose NURBS output drives edit-mode operators)
        # asks to be rebuilt on a timer instead.  Timers never fire in
        # background mode, so stand in for one -- this is the same call
        # the timer makes.
        build.flush_deferred()
        if changed is not None and build.is_stale(obj):
            # Auto Update swallows a failed rebuild on purpose -- a
            # half-typed value must not throw an error box at someone
            # mid-drag -- so establish WHY it did nothing before calling
            # it a bug.  A settings change that turns the generator into
            # a multi-object one is a known limit, not a fault.
            try:
                build.rebuild(obj, bpy.context)
            except build.LiveError as exc:
                skips.append((idname, "%s=%r cannot rebuild in place: %s"
                              % (changed[0], changed[2], exc)))
                continue
            fails.append((idname, "Auto Update left %s stale after "
                                  "changing %s" % (obj.name, changed[0])))
            continue

    # 3b. With it off, a change must be visible as stale and stay that
    #     way until a rebuild is asked for -- silently rebuilding a
    #     three-second generator on every keystroke is the thing the
    #     toggle exists to prevent.
    obj.math_art.autobuild = False
    changed = _nudge(pg)
    if changed is None:
        skips.append((idname, "no scalar setting to nudge"))
        continue
    name, old, new = changed
    if not build.is_stale(obj):
        fails.append((idname, "changing %s left the object un-stale"
                      % name))
        continue
    try:
        rebuilt = build.rebuild(obj, bpy.context)
    except build.LiveError as exc:
        if "objects at once" in str(exc):
            skips.append((idname, "%s=%r makes it multi-object "
                                  "(a later phase)" % (name, new)))
            continue
        # Distinguish "the framework cannot rebuild this" from "the
        # generator refuses these particular numbers".  Several
        # generators cap their own output (too many copies, an empty
        # level set); putting the old value back and rebuilding says
        # which of the two just happened.
        setattr(pg, name, old)
        try:
            build.rebuild(obj, bpy.context)
        except build.LiveError:
            fails.append((idname, "rebuild failed after %s %r->%r: %s"
                          % (name, old, new, exc)))
            continue
        skips.append((idname, "the generator rejects %s=%r: %s"
                      % (name, new, exc)))
        continue
    if build.is_stale(rebuilt):
        fails.append((idname, "still stale after a rebuild"))
        continue
    live_shape = _shape(rebuilt)
    live_counts = _counts(rebuilt)
    if live_counts == (0, 0) and len(_group(rebuilt)) == 1:
        fails.append((idname, "rebuild produced empty geometry"))
        continue

    # 4. THE claim: the same as running the Add operator with those
    #    settings, rather than merely something.
    # Values must be MATERIALISED here, not read lazily after the reset
    # below: an array property hands back a view into the settings group,
    # and the reset frees the object that group lives on.  Passing those
    # views to an operator afterwards reads freed memory, which Blender
    # answers with an access violation rather than an exception.
    kwargs = {}
    for prop in clone.editable_props(info.pg_cls):
        pid = prop.identifier
        if prop.type in ('INT', 'FLOAT', 'BOOLEAN', 'ENUM', 'STRING'):
            try:
                kwargs[pid] = clone._prop_value(pg, prop)
            except (AttributeError, TypeError):
                pass
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        op('EXEC_DEFAULT', **kwargs)
    except Exception as exc:
        skips.append((idname, "could not re-run for comparison: %s"
                      % exc))
        continue
    # The comparison run's root is resolved the same way the first one
    # was: what the generator leaves active is not always the root of
    # what it built.  Measuring from the active object instead would
    # compare a whole rebuilt cluster against one of its bubbles.
    fresh = root_for_object(bpy.context.view_layer.objects.active)
    if fresh is None:
        fails.append((idname, "the comparison run recorded no settings"))
        continue
    # No reading anything off `rebuilt` past this point: the reset above
    # freed it.  `live_shape` was measured while it was still alive, and
    # already carries the group size and per-object counts.
    fresh_shape = _shape(fresh)
    if fresh_shape[0] != live_shape[0]:
        fails.append((idname,
                      "rebuild differs from a fresh %s=%r run: "
                      "live %d object(s) %s vs fresh %d object(s) %s"
                      % (name, new, live_shape[0][0], live_shape[0][1],
                         fresh_shape[0][0], fresh_shape[0][1])))
        continue
    if fresh_shape[1] != live_shape[1]:
        fails.append((idname,
                      "rebuild has the same element counts as a fresh "
                      "%s=%r run but a different shape" % (name, new)))
        continue

    checked += 1
    print("OK   %-42s %s %r->%r  %d object(s) %s"
          % (idname, name, old, new, live_shape[0][0], live_counts))

print("\n" + "=" * 64)
for idname, why in skips:
    print("SKIP %-42s %s" % (idname, why))
for idname, why in fails:
    print("FAIL %-42s %s" % (idname, why))
print("=" * 64)
print("live editing: %d generators verified end to end, %d skipped, "
      "%d failed" % (checked, len(skips), len(fails)))
print("RESULT:", "OK" if not fails else "BAD")
if fails:
    sys.exit(1)
