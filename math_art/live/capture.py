
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
    from .build import AUTO_LIMIT_SECONDS, choose_root, is_building
except ImportError:                     # flat import outside the package
    import clone
    from build import AUTO_LIMIT_SECONDS, choose_root, is_building


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
        settings.gen_modifiers = '\x1f'.join(m.name for m in root.modifiers)
        try:
            from .build import _record_members
        except ImportError:
            from build import _record_members
        _record_members(root, [o for o in made if o is not root])
        # Auto Update is offered on the strength of what this generator
        # actually cost on this machine, not a guess: a build that takes
        # two seconds is a button, a build that takes twenty
        # milliseconds is a slider.
        settings.autobuild = seconds < AUTO_LIMIT_SECONDS
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
    """That recording and rebuilding agree on what a group hangs off.

    `choose_root` used to have a copy here as well as in `build`, and
    two copies is exactly how the object that OWNS the settings comes to
    differ from the object a rebuild treats as the root -- after which a
    rebuild quietly rearranges the group.  The rule is tested where it
    lives, in `live.build`; what matters here is that this module uses
    that one and has not grown a second.
    """
    try:
        from . import build as build_module
    except ImportError:
        import build as build_module

    ok = choose_root is build_module.choose_root
    print("live.capture: recording and rebuilding share one root rule")

    for name in ('record', 'install_capture', 'remove_capture'):
        ok = ok and (not _IN_BLENDER or name in globals())
    print("RESULT:", "OK" if ok else "BAD")
    assert ok
