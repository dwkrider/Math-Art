
# Rebuilding a generated object from its stored settings.
#
# A rebuild runs the generator's own `execute()` -- reached through the
# cloned settings group, so it is the same code the Add menu runs -- and
# then moves the result onto the object that already exists, instead of
# leaving a second object behind.
#
# `bpy.ops` is deliberately NOT used.  A rebuild is triggered from a
# property update callback, and calling an operator from one is unsafe;
# calling the underlying `execute()` directly is not, and it also keeps
# the rebuild out of the undo stack as a separate step.
#
# Two ways of landing the result, chosen by what came back:
#
#   transplant  -- one new object of the same type: its data is moved
#                  onto the existing object.  Modifiers, materials,
#                  transform, parenting, constraints, drivers, name and
#                  collection membership all survive, because the object
#                  itself is never replaced.  This is the normal path.
#   replace     -- one new object of a DIFFERENT type: a generator whose
#                  Output setting switched between a curve and a mesh
#                  cannot transplant, because an object's type is fixed
#                  by its data.  The new object takes the old one's
#                  place, with `user_remap` so anything pointing at the
#                  old object follows.
#
# Several objects at once (Bubble Cluster, Seifert Surface, ...) is a
# later phase; such a generator is detected when it is first built and
# its panel says so rather than offering a button that cannot work.

import time

try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

try:
    from . import clone
    from .registry import settings_for_object
except ImportError:                     # flat import outside the package
    import clone
    from registry import settings_for_object


class LiveError(Exception):
    """A rebuild could not be completed; the old geometry is untouched."""


# Set while a rebuild is running.  Rebuilding writes settings back (a
# generator's `execute` assigns to `properties.location`, and preset
# callbacks assign to their siblings), and every one of those writes
# would otherwise trigger another rebuild.
_BUILDING = [False]


def is_building():
    return _BUILDING[0]


# A build slower than this feels like lag rather than like dragging a
# slider, so a generator that takes longer starts with Auto Update off.
# The threshold is applied to the generator's MEASURED time on the
# user's own machine, not to a table baked in here.
AUTO_LIMIT_SECONDS = 0.25


if _IN_BLENDER:

    _DATA_COLLECTIONS = ('objects', 'meshes', 'curves', 'materials')

    def _snapshot():
        """Names of the datablocks that exist now, per collection.

        A build creates meshes and materials as well as objects; without
        a before-picture the ones it discards would pile up, and during
        a drag that is a new orphan mesh every frame.
        """
        return {name: set(getattr(bpy.data, name).keys())
                for name in _DATA_COLLECTIONS}

    def _new_objects(before):
        return [bpy.data.objects[name]
                for name in bpy.data.objects.keys()
                if name not in before['objects']]

    def _purge(before, keep, discard=()):
        """Remove what this build made and then did not use.

        Scratch OBJECTS are removed outright rather than by a user
        count.  An object linked into a collection always has a user --
        the collection -- so a "delete the unused ones" sweep never
        touches any of them, and every rebuild would leave its scratch
        object sitting in the scene.

        `discard` is for datablocks that were already here and have just
        been let go: the mesh an object held before its replacement.
        Being pre-existing, they are invisible to the new-since-`before`
        test, so without this a rebuild leaks the old mesh -- once per
        frame while a slider is being dragged.
        """
        for key in list(bpy.data.objects.keys()):
            if key in before['objects']:
                continue
            block = bpy.data.objects.get(key)
            if block is None or block in keep:
                continue
            try:
                bpy.data.objects.remove(block)
            except (RuntimeError, ReferenceError):
                pass

        candidates = []
        for name in _DATA_COLLECTIONS[1:]:          # objects done above
            store = getattr(bpy.data, name)
            for key in list(store.keys()):
                if key in before[name]:
                    continue
                block = store.get(key)
                if block is not None:
                    candidates.append((store, block))
        for block in discard:
            if block is None:
                continue
            store = getattr(bpy.data, _store_name(block), None)
            if store is not None:
                candidates.append((store, block))

        for store, block in candidates:
            try:
                if block in keep or block.users:
                    continue
                store.remove(block)
            except (RuntimeError, ReferenceError):
                pass

    def _store_name(block):
        """Which `bpy.data` collection a datablock belongs to."""
        return {'Mesh': 'meshes', 'Curve': 'curves',
                'Material': 'materials'}.get(type(block).__name__, '')

    def _copy_modifier(src, obj):
        """Recreate one modifier on another object, settings and all."""
        mod = obj.modifiers.new(src.name, src.type)
        for prop in src.bl_rna.properties:
            name = prop.identifier
            if name in ('rna_type', 'type', 'name') or prop.is_readonly:
                continue
            try:
                setattr(mod, name, getattr(src, name))
            except (AttributeError, TypeError, ValueError):
                pass
        return mod

    def _transplant(obj, src, settings):
        """Move the scratch build's data onto the existing object."""
        old_data = obj.data
        obj.data = src.data

        # Modifiers the generator itself added are part of its output and
        # have to be refreshed; anything the user added afterwards is
        # theirs and is left in place.  The generator's are known by the
        # names recorded when the object was first built.
        owned = {n for n in settings.gen_modifiers.split('\x1f') if n}
        for mod in list(obj.modifiers):
            if mod.name in owned:
                obj.modifiers.remove(mod)
        made = []
        for mod in src.modifiers:
            made.append(_copy_modifier(mod, obj).name)
        settings.gen_modifiers = '\x1f'.join(made)
        return old_data

    def _replace(obj, src, info, settings, context):
        """Put the scratch build in the existing object's place.

        Only for a type change, which a transplant cannot express.  The
        object identity does change here, so `user_remap` is what keeps
        the scene consistent: parents, modifier targets and constraints
        that referred to the old object refer to the new one afterwards.
        """
        name = obj.name
        matrix = obj.matrix_world.copy()
        parent = obj.parent
        parent_inverse = obj.matrix_parent_inverse.copy()
        collections = list(obj.users_collection)

        # The record has to move BEFORE the old object goes away.
        clone.copy_values(obj.math_art, src.math_art)
        clone.copy_values(getattr(obj.math_art, info.slug),
                          getattr(src.math_art, info.slug))

        released = obj.data
        obj.user_remap(src)
        bpy.data.objects.remove(obj)

        src.name = name
        for coll in list(src.users_collection):
            coll.objects.unlink(src)
        for coll in collections:
            coll.objects.link(src)
        if parent is not None:
            src.parent = parent
            src.matrix_parent_inverse = parent_inverse
        src.matrix_world = matrix
        return src, released

    def _geometry_counts(obj):
        """(elements, faces) for whatever kind of data the object holds."""
        data = obj.data
        if obj.type == 'MESH':
            return len(data.vertices), len(data.polygons)
        if obj.type == 'CURVE':
            return (sum(len(s.bezier_points) + len(s.points)
                        for s in data.splines), len(data.splines))
        return 0, 0

    def record_build(obj, info, settings, seconds):
        """Note what the object was built from, and how expensive it was."""
        pg = getattr(obj.math_art, info.slug)
        settings.built_sig = clone.signature(pg)
        settings.last_seconds = float(seconds)
        counts = _geometry_counts(obj)
        settings.built_elements, settings.built_faces = counts

    def is_stale(obj):
        """True if the settings have moved on from the built geometry."""
        info, pg = settings_for_object(obj)
        if info is None or pg is None:
            return False
        return clone.signature(pg) != obj.math_art.built_sig

    def is_hand_edited(obj):
        """True if the geometry no longer matches what was last built.

        A rebuild overwrites the mesh, so this is what the panel warns
        about before that happens.  It only detects a change of size, so
        moving a vertex without adding one goes unnoticed -- a warning
        that fires on the obvious cases is still better than none.
        """
        settings = getattr(obj, 'math_art', None)
        if settings is None or not settings.generator:
            return False
        if not settings.built_elements and not settings.built_faces:
            return False
        return _geometry_counts(obj) != (settings.built_elements,
                                         settings.built_faces)

    def rebuild(obj, context, allow_replace=True):
        """Rebuild `obj` from its stored settings.  Returns the object.

        Raises LiveError, with the old geometry untouched, if the
        generator produced nothing usable.

        `allow_replace=False` refuses the replacement path instead of
        taking it.  An interactive edit passes False because replacement
        FREES the object -- and the caller is a property update callback
        running on a property of that very object, so deleting it there
        would pull the ground out from under the write in progress.  The
        object is left stale, and the Rebuild button, which is an
        operator and so runs after the write has finished, does it.
        """
        info, pg = settings_for_object(obj)
        if info is None or pg is None:
            raise LiveError("this object has no Math Art generator")
        settings = obj.math_art

        view_layer = context.view_layer
        previous_active = view_layer.objects.active
        previously_selected = [o for o in context.selected_objects]

        before = _snapshot()
        _BUILDING[0] = True
        started = time.perf_counter()
        try:
            try:
                result = info.pg_cls.execute(pg, context)
            except Exception as exc:                  # generator failure
                _purge(before, keep=set())
                raise LiveError("%s: %s" % (type(exc).__name__, exc))
            seconds = time.perf_counter() - started

            made = _new_objects(before)
            if not result or 'FINISHED' not in result or not made:
                _purge(before, keep=set())
                raise LiveError("the generator produced no geometry")
            if len(made) > 1:
                _purge(before, keep=set())
                settings.n_created = len(made)
                raise LiveError(
                    "%s builds %d objects at once; editing those in place "
                    "is not supported yet" % (info.label, len(made)))

            src = made[0]
            if src.type != obj.type and not allow_replace:
                # Read the type BEFORE purging: the purge frees `src`,
                # and reaching for an attribute of a freed object to
                # describe why nothing happened raises a second, more
                # confusing error than the one being reported.
                became = src.type.lower()
                _purge(before, keep=set())
                raise LiveError(
                    "changing the output to %s replaces the object; use "
                    "Rebuild" % became)
            if src.type == obj.type:
                released = _transplant(obj, src, settings)
                bpy.data.objects.remove(src)
                built = obj
            else:
                built, released = _replace(obj, src, info, settings,
                                           context)
            keep = {built, built.data}
            keep |= set(built.data.materials or [])
            _purge(before, keep=keep, discard=(released,))

            record_build(built, info, built.math_art, seconds)
            # The object's evaluated state -- its bounds, and any
            # modifier reading its geometry -- is cached until the
            # depsgraph is told the data moved under it.
            built.update_tag()
        finally:
            _BUILDING[0] = False

        # A build selects and activates what it made; the user's
        # selection is theirs, and a rebuild is not a selection change.
        for o in context.selected_objects:
            o.select_set(False)
        for o in previously_selected:
            try:
                o.select_set(True)
            except ReferenceError:
                pass                        # replaced by this rebuild
        try:
            view_layer.objects.active = (previous_active
                                         if previous_active is not None
                                         else built)
        except (ReferenceError, RuntimeError):
            view_layer.objects.active = built
        return built

    def rebuild_quietly(obj, context):
        """Rebuild for an interactive edit: never raise, never report.

        Half-typed numbers and momentarily impossible combinations are
        normal while dragging a slider, and a generator that rejects one
        should not put an error in front of the user -- the stale marker
        already says the geometry is behind the settings.
        """
        try:
            return rebuild(obj, context, allow_replace=False)
        except (LiveError, RuntimeError, ValueError, MemoryError,
                ReferenceError):
            return None

    def auto_update(pg, context):
        """The `update=` attached to every cloned setting.

        `id_data` is what makes one shared settings class safe: the slot
        exists on every object, and this says which object's slot was
        actually edited, so a rebuild lands on that one rather than on
        whatever happens to be active.
        """
        if _BUILDING[0]:
            return
        obj = getattr(pg, 'id_data', None)
        if not isinstance(obj, bpy.types.Object):
            return
        settings = getattr(obj, 'math_art', None)
        if settings is None or not settings.generator:
            return
        if not settings.autobuild:
            return                       # the stale marker says the rest
        rebuild_quietly(obj, context)


def _selftest():
    """The parts that hold without Blender: the staleness contract."""
    ok = AUTO_LIMIT_SECONDS > 0.0
    # A guard that never clears would silently disable every rebuild
    # after the first failure, so the flag has to start down.
    ok = ok and not is_building()
    print("live.build: rebuild guard starts clear, auto limit %.2fs"
          % AUTO_LIMIT_SECONDS)
    err = LiveError("no geometry")
    ok = ok and isinstance(err, Exception) and str(err) == "no geometry"
    print("RESULT:", "OK" if ok else "BAD")
    assert ok
