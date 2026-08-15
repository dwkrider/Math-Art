
# Recording what a generator was asked for, on the object it produced.
#
# Every live-editable operator's `execute` is wrapped once, at
# registration.  The wrapper runs the original, notices which objects
# appeared, and copies the operator's settings onto the one that came
# out.  That is the whole reason none of the 117 generator modules needs
# editing: they keep creating objects exactly as they did, and the
# record is taken from outside.
#
# The wrapper must never be able to stop a generator from working.  A
# failure to record settings costs the sidebar, which is a inconvenience;
# a failure to create the object costs the user their action.  So the
# recording is wrapped in its own `except` and reports to the console.

import time

try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

try:
    from . import clone
    from .build import AUTO_LIMIT_SECONDS, is_building
except ImportError:                     # flat import outside the package
    import clone
    from build import AUTO_LIMIT_SECONDS, is_building


def choose_root(made, active):
    """Which of the objects a generator made is THE object.

    The generator has already said so by leaving it active, which is
    what the Add menu relies on too.  Failing that, an object nothing
    else is parented to is the assembly's root.
    """
    if not made:
        return None
    if active is not None and active in made:
        return active
    for obj in made:
        if obj.parent is None:
            return obj
    return made[0]


if _IN_BLENDER:

    try:
        from .registry import GENERATORS
    except ImportError:
        from registry import GENERATORS

    def record(op, context, info, made, seconds):
        """Copy an operator's settings onto the object it just built."""
        root = choose_root(made, context.view_layer.objects.active)
        if root is None:
            return
        settings = root.math_art
        pg = getattr(settings, info.slug, None)
        if pg is None:
            return
        clone.copy_values(op, pg)
        settings.generator = info.idname
        settings.n_created = len(made)
        settings.gen_modifiers = '\x1f'.join(m.name for m in root.modifiers)
        # Auto Update is offered on the strength of what this generator
        # actually cost on this machine, not a guess: a build that takes
        # two seconds is a button, a build that takes twenty
        # milliseconds is a slider.
        settings.autobuild = (seconds < AUTO_LIMIT_SECONDS
                              and len(made) == 1)
        try:
            from .build import record_build
        except ImportError:
            from build import record_build
        record_build(root, info, settings, seconds)

    def _wrap(info):
        """Wrap one operator's execute so its results are recorded."""
        original = info.op_cls.execute

        def execute(op, context):
            # A rebuild reaches `execute` through the settings group, not
            # through the operator, so this only ever sees a real Add.
            if is_building():
                return original(op, context)
            existing = set(bpy.data.objects.keys())
            started = time.perf_counter()
            result = original(op, context)
            seconds = time.perf_counter() - started
            if result and 'FINISHED' in result:
                try:
                    made = [bpy.data.objects[name]
                            for name in bpy.data.objects.keys()
                            if name not in existing]
                    record(op, context, info, made, seconds)
                except Exception as exc:
                    # Never let bookkeeping cost the user their object.
                    print("Math Art: could not record settings for %s: "
                          "%s: %s" % (info.idname, type(exc).__name__,
                                      exc))
            return result

        execute.__name__ = 'execute'
        execute.__doc__ = getattr(original, '__doc__', None)
        info.orig_execute = original
        info.op_cls.execute = execute

    def install_capture():
        """Wrap every registered generator's execute."""
        for info in GENERATORS.values():
            if info.orig_execute is None:
                _wrap(info)

    def remove_capture():
        """Put every operator's own execute back."""
        for info in GENERATORS.values():
            if info.orig_execute is not None:
                info.op_cls.execute = info.orig_execute
                info.orig_execute = None


def _selftest():
    """Root choice, which decides which object owns the settings."""
    class FakeObject:
        def __init__(self, name, parent=None):
            self.name = name
            self.parent = parent

        def __repr__(self):
            return self.name

    parent = FakeObject('Bubbles')
    children = [FakeObject('Bubble %d' % i, parent) for i in range(3)]
    made = children + [parent]

    ok = choose_root(made, parent) is parent
    print("live.capture: the generator's active object wins")

    # No active object: the one nothing hangs off is the root, even when
    # the children were created first.
    ok = ok and choose_root(made, None) is parent
    print("live.capture: otherwise the unparented object is the root")

    # An active object that belongs to something else must not be
    # mistaken for this build's output.
    outsider = FakeObject('Cube')
    ok = ok and choose_root(made, outsider) is parent
    print("live.capture: an unrelated active object is ignored")

    ok = ok and choose_root([], None) is None
    single = FakeObject('Torus Knot')
    ok = ok and choose_root([single], None) is single
    print("RESULT:", "OK" if ok else "BAD")
    assert ok
