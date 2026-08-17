
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
    from .registry import info_for_object, settings_for_object
except ImportError:                     # flat import outside the package
    import clone
    from registry import info_for_object, settings_for_object


class LiveError(Exception):
    """A rebuild could not be completed; the old geometry is untouched."""


def choose_root(made, active):
    """Which of the objects a generator made is THE object.

    STRUCTURE decides, and the active object only breaks ties.  It is
    tempting to trust what the generator left active, but that is not
    always the assembly's root: Bubble Cluster parents twelve bubbles to
    an Empty and leaves the FIRST BUBBLE active.  Anchoring the group on
    that bubble makes the Empty a mere companion, so the next rebuild
    deletes it as part of the previous generation and leaves the root
    parented to nothing -- the cluster comes apart.

    So: a candidate is an object no other member of the same build owns.
    Among candidates the active one wins, because a generator that makes
    several independent objects (a curve and its mesh, say) really has
    said which one it considers primary.

    Both the first record and every later rebuild go through here, so
    the anchor cannot drift from one build to the next.
    """
    if not made:
        return None
    members = set(made)
    candidates = [obj for obj in made
                  if obj.parent is None or obj.parent not in members]
    if not candidates:                  # a cycle: no structural answer
        candidates = list(made)
    if active is not None and active in candidates:
        return active
    return candidates[0]


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
        """Which `bpy.data` collection a datablock belongs to.

        By TYPE rather than by class name.  Blender has more than one
        class per collection -- a NURBS surface's data is a
        `SurfaceCurve`, and text is a `TextCurve`, both of which live in
        `bpy.data.curves` -- so matching names exactly silently failed to
        free a Scherk-Collins sculpture's old surface every time its
        NURBS Output was switched off.
        """
        if isinstance(block, bpy.types.Mesh):
            return 'meshes'
        if isinstance(block, bpy.types.Curve):
            return 'curves'
        if isinstance(block, bpy.types.Material):
            return 'materials'
        return ''

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
        if src.data is not None:
            obj.data = src.data
        # An Empty -- which is what a generator that lays its pieces out
        # as separate objects uses for a root -- holds no data at all,
        # and its `data` is read-only.  There is nothing to move; the
        # group hanging off it is the whole of the rebuild.

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

    def _group_members(root):
        """The companion objects a previous build left under `root`.

        Read through object POINTERS rather than by name: Blender clears
        a pointer when its object is deleted and follows it when the
        object is renamed, so a group survives both without this module
        having to notice either.
        """
        members = []
        for entry in getattr(root.math_art, 'members', ()):
            member = entry.obj
            if member is None:
                continue                    # deleted since the last build
            try:
                member.name
            except ReferenceError:
                continue
            members.append(member)
        return members

    def _record_members(root, companions):
        """Note the group a build produced, on the root and the members."""
        settings = root.math_art
        settings.members.clear()
        for companion in companions:
            settings.members.add().obj = companion
            # The back-pointer is what lets someone click a bubble in a
            # bubble cluster and get the cluster's settings, instead of
            # an empty tab and no way to tell what made it.
            companion.math_art.group_root = root
        settings.n_created = len(companions) + 1

    def _adopt(built, src, companions, src_matrix):
        """Move a scratch build's companions onto the kept object.

        Two things have to be true afterwards: the companions hang off
        the object the user has been editing, and they sit where that
        object is rather than where the scratch build was made.

        Anything parented to the scratch root gets re-parented, which
        settles its position on its own -- a local transform against a
        new parent is exactly the offset it already had.  Anything
        unparented has to be moved bodily by the difference between the
        two roots.  Anything parented to another companion is already
        correct, because that companion is being handled too.
        """
        if built is src:                    # replacement: already right
            delta = None
        else:
            delta = built.matrix_world @ src_matrix.inverted()
        collections = list(built.users_collection)
        for companion in companions:
            if companion.parent is src and built is not src:
                basis = companion.matrix_basis.copy()
                inverse = companion.matrix_parent_inverse.copy()
                companion.parent = built
                companion.matrix_parent_inverse = inverse
                companion.matrix_basis = basis
            elif companion.parent is None and delta is not None:
                companion.matrix_world = delta @ companion.matrix_world
            for coll in list(companion.users_collection):
                if coll not in collections:
                    coll.objects.unlink(companion)
            for coll in collections:
                if companion.name not in coll.objects:
                    coll.objects.link(companion)

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

            # Which of the new objects is THE object is decided the same
            # way it was when the settings were first recorded, so the
            # anchor does not wander between builds.
            src = choose_root(made, view_layer.objects.active)
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

            companions = [o for o in made if o is not src]
            # Where the scratch build put its root, needed to move any
            # unparented companion onto the object being kept.
            src_matrix = src.matrix_world.copy()
            superseded = _group_members(obj)

            if src.type == obj.type:
                released = _transplant(obj, src, settings)
                built = obj
                # Re-parent BEFORE the scratch root is freed, or its
                # children lose their parent as it goes.
                _adopt(built, src, companions, src_matrix)
                bpy.data.objects.remove(src)
            else:
                built, released = _replace(obj, src, info, settings,
                                           context)
                _adopt(built, src, companions, src_matrix)

            # The previous generation of companions goes only now that
            # the new one is safely in place.
            discard = [released]
            for member in superseded:
                discard.append(getattr(member, 'data', None))
                try:
                    bpy.data.objects.remove(member)
                except (RuntimeError, ReferenceError):
                    pass

            keep = {built} | set(companions)
            for kept in list(keep):
                data = getattr(kept, 'data', None)
                if data is None:
                    continue
                keep.add(data)
                keep |= set(getattr(data, 'materials', None) or [])
            _purge(before, keep=keep, discard=tuple(discard))

            _record_members(built, companions)
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

    # Objects whose rebuild has been put off until a safe moment, by
    # NAME: the wait is precisely for a point at which object references
    # taken now may no longer be valid.
    _PENDING = set()

    def flush_deferred():
        """Run every rebuild that was put off.  Returns how many ran.

        Called from a timer in a running Blender, and directly by the
        headless tests, where no timer ever fires.
        """
        names = sorted(_PENDING)
        _PENDING.clear()
        done = 0
        for name in names:
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            if settings_for_object(obj)[0] is None:
                continue
            try:
                rebuild(obj, bpy.context)
                done += 1
            except (LiveError, RuntimeError, ValueError, MemoryError,
                    ReferenceError):
                pass
        return done

    def _timer():
        flush_deferred()
        return None                      # one shot

    def schedule_rebuild(obj):
        """Rebuild `obj` at the next safe moment rather than right now.

        Some generators cannot be re-run from inside a property update
        callback at all: Scherk-Collins builds its NURBS output with
        `bpy.ops.object.mode_set` and `bpy.ops.curve.make_segment`, and
        edit-mode operators need a context that a half-finished property
        write does not provide.  Coming back on a timer costs a frame and
        makes them safe -- and, because the write has finished by then, a
        deferred rebuild may also replace the object.
        """
        _PENDING.add(obj.name)
        if not bpy.app.timers.is_registered(_timer):
            bpy.app.timers.register(_timer, first_interval=0.0)

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
        info = info_for_object(obj)
        if info is not None and getattr(info.op_cls, 'math_art_live_defer',
                                        False):
            schedule_rebuild(obj)
            return
        rebuild_quietly(obj, context)


def _selftest():
    """Root choice, which decides what a group hangs off."""
    class FakeObject:
        def __init__(self, name, parent=None):
            self.name = name
            self.parent = parent

        def __repr__(self):
            return self.name

    ok = AUTO_LIMIT_SECONDS > 0.0
    # A guard that never clears would silently disable every rebuild
    # after the first failure, so the flag has to start down.
    ok = ok and not is_building()
    print("live.build: rebuild guard starts clear, auto limit %.2fs"
          % AUTO_LIMIT_SECONDS)
    err = LiveError("no geometry")
    ok = ok and isinstance(err, Exception) and str(err) == "no geometry"

    empty = FakeObject('Bubbles')
    bubbles = [FakeObject('Bubble %03d' % i, empty) for i in range(1, 13)]
    cluster = bubbles + [empty]

    # THE regression that motivated the structural rule: Bubble Cluster
    # leaves a CHILD active.  Trusting that made the parent Empty a
    # companion, and the next rebuild deleted it and orphaned the root.
    ok = ok and choose_root(cluster, bubbles[0]) is empty
    print("live.build: a child left active does not become the root")

    ok = ok and choose_root(cluster, None) is empty
    ok = ok and choose_root(cluster, FakeObject('Cube')) is empty
    print("live.build: an unrelated active object is ignored")

    # Independent objects: no structure to go on, so the generator's
    # choice of active object is the answer.
    curve = FakeObject('Raceme')
    mesh = FakeObject('Florets')
    ok = ok and choose_root([curve, mesh], mesh) is mesh
    ok = ok and choose_root([curve, mesh], None) is curve
    print("live.build: with no hierarchy the active object decides")

    single = FakeObject('Torus Knot')
    ok = ok and choose_root([single], None) is single
    ok = ok and choose_root([], None) is None

    # A parent cycle has no structural answer; it must still return one
    # of the objects rather than raise or hand back None.
    a = FakeObject('A')
    b = FakeObject('B', a)
    a.parent = b
    ok = ok and choose_root([a, b], None) in (a, b)
    print("RESULT:", "OK" if ok else "BAD")
    assert ok
