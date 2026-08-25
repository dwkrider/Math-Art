
# Live editing of generated objects -- the Math Art sidebar tab.
#
# Every Math Art generator is an operator that builds an object and then
# forgets what it was asked for: the settings live in the redo panel,
# which lasts until the next action.  This package keeps them.  When a
# generator runs, its settings are recorded on the object it produced;
# selecting that object later brings the generator's own UI back in the
# sidebar, and changing anything there rebuilds the geometry.
#
# Nothing here is generator-specific, and no generator module knows this
# package exists.  Everything is derived from the operators themselves:
#
#   registry -- which generators are live-editable
#   clone    -- an operator's properties, draw() and execute() re-hosted
#               on a PropertyGroup that an object can store
#   capture  -- wraps each operator's execute to record its settings
#   build    -- re-runs a generator and lands the result on the object
#               that already exists
#   panel    -- the sidebar tab, which just hands its layout to clone
#
# Adding a generator to the add-on therefore adds it here too, with its
# full parameter set, on its own.

try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

try:
    from . import build, capture, clone, panel
    from .registry import GENERATORS, GenInfo, slug_for
except ImportError:                     # flat import outside the package
    import build
    import capture
    import clone
    import panel
    from registry import GENERATORS, GenInfo, slug_for


# Generators that keep a bespoke sidebar panel of their own, because it
# does something this framework cannot derive from an Add operator.
OPT_OUT = frozenset({
    # The Relief Panel's sidebar is NOT a duplicate of its Add operator:
    # the operator builds a single field, while the panel composes a
    # STACK of them, with per-layer amplitude, blend mode, mask and
    # placement, and it drives an exact plate solver the layered model
    # does not expose.  Replacing that with the operator's UI would be a
    # loss of function dressed up as a cleanup, so it keeps its own.
    'mesh.relief_panel_add',
})

# Whole operator families that are not generators: the helpers a panel
# drives (add a layer) and the spec-file readers and writers.
OPT_OUT_PREFIXES = ('relief.', 'scherk.')


def _defines_poll(cls):
    """True if the add-on's own code gives this operator a `poll`.

    Blender's own base classes are skipped, because whatever they define
    is true of every operator and so cannot distinguish one from
    another; what matters is whether a generator module chose to gate
    this operator on the selection.
    """
    for klass in cls.__mro__:
        if (getattr(klass, '__module__', '') or '').startswith('bpy'):
            continue
        if 'poll' in klass.__dict__:
            return True
    return False


def is_generator(cls):
    """Can this operator's output be edited in place?

    Three disqualifications, and each is a real category rather than a
    list of names:

      a custom `poll`   -- the operator transforms whatever is selected
                           instead of creating something, so re-running
                           it needs an input that no longer exists
      `filename_ext`    -- an Import/Export helper: its subject is a
                           file, not an object
      opted out         -- it has a panel of its own already
    """
    idname = getattr(cls, 'bl_idname', '') or ''
    if '.' not in idname or idname in OPT_OUT:
        return False
    if idname.startswith(OPT_OUT_PREFIXES):
        return False
    if not hasattr(cls, 'execute'):
        return False
    if hasattr(cls, 'filename_ext'):
        return False
    return not _defines_poll(cls)


if _IN_BLENDER:

    # Everything this package registers, newest last, so unregistering
    # in reverse is correct.
    _REGISTERED = []
    _CONTAINER = [None]

    def _operator_classes(modules):
        """Every operator class the generator modules define."""
        seen = {}
        for module in modules:
            for name in dir(module):
                cls = getattr(module, name, None)
                if not isinstance(cls, type):
                    continue
                if not issubclass(cls, bpy.types.Operator):
                    continue
                idname = getattr(cls, 'bl_idname', None)
                if idname and idname not in seen:
                    seen[idname] = cls
        return seen

    class MathArtGroupMember(bpy.types.PropertyGroup):
        """One companion object of a multi-object build.

        An object POINTER, not a name: Blender clears it when the object
        is deleted and follows it when the object is renamed, so a group
        stays correct through both without any bookkeeping here.
        """
        obj: bpy.props.PointerProperty(type=bpy.types.Object)

    def _container_class():
        """The settings block stored on every Math Art object.

        One pointer per generator.  Blender allocates a nested pointer
        only when something writes to it, so an ordinary object in an
        ordinary scene carries none of this -- the cost is a hundred-odd
        RNA definitions once, not a hundred-odd values per object.
        """
        annotations = {
            'generator': bpy.props.StringProperty(
                name="Generator",
                description="Operator that built this object"),
            'autobuild': bpy.props.BoolProperty(
                name="Auto Update", default=True,
                description="Rebuild the geometry as the settings "
                            "change, rather than on demand"),
            'built_sig': bpy.props.StringProperty(),
            'gen_modifiers': bpy.props.StringProperty(),
            'last_seconds': bpy.props.FloatProperty(default=0.0),
            'n_created': bpy.props.IntProperty(default=1),
            'built_elements': bpy.props.IntProperty(default=0),
            'built_faces': bpy.props.IntProperty(default=0),
            # A generator that builds several objects at once records
            # the companions here, on the root that owns the settings...
            'members': bpy.props.CollectionProperty(
                type=MathArtGroupMember),
            # ...and each companion points back, so selecting one of
            # them finds the settings that made it.
            'group_root': bpy.props.PointerProperty(
                type=bpy.types.Object,
                description="The Math Art object this one is part of"),
        }
        for info in GENERATORS.values():
            annotations[info.slug] = bpy.props.PointerProperty(
                type=info.pg_cls)
        return type('MathArtObjectSettings', (bpy.types.PropertyGroup,),
                    {'__annotations__': annotations})

    def install(modules):
        """Make every eligible generator live-editable.

        Called from the package's `register()` once the generator
        modules have registered their operators, because the operators
        are what everything here is derived from.
        """
        uninstall()
        bpy.utils.register_class(MathArtGroupMember)
        _REGISTERED.append(MathArtGroupMember)
        for idname, cls in sorted(_operator_classes(modules).items()):
            if not is_generator(cls):
                continue
            slug = slug_for(idname)
            try:
                pg_cls = clone.settings_class(cls, slug,
                                              build.auto_update)
                bpy.utils.register_class(pg_cls)
            except Exception as exc:
                print("Math Art: no live settings for %s: %s: %s"
                      % (idname, type(exc).__name__, exc))
                continue
            _REGISTERED.append(pg_cls)
            GENERATORS[idname] = GenInfo(
                idname, getattr(cls, 'bl_label', idname), slug, cls,
                pg_cls)

        container = _container_class()
        bpy.utils.register_class(container)
        _REGISTERED.append(container)
        _CONTAINER[0] = container
        bpy.types.Object.math_art = bpy.props.PointerProperty(
            type=container)

        for cls in panel.CLASSES:
            bpy.utils.register_class(cls)
            _REGISTERED.append(cls)

        capture.install_capture()
        print("Math Art: %d generators are editable in the sidebar"
              % len(GENERATORS))

    def uninstall():
        """Undo `install`, leaving the operators exactly as they were."""
        if not _REGISTERED and not GENERATORS:
            return
        capture.remove_capture()
        if hasattr(bpy.types.Object, 'math_art'):
            del bpy.types.Object.math_art
        for cls in reversed(_REGISTERED):
            try:
                bpy.utils.unregister_class(cls)
            except (RuntimeError, ValueError):
                pass
        _REGISTERED.clear()
        _CONTAINER[0] = None
        GENERATORS.clear()

    # Not a generator: it contributes no menu entry and is wired up by
    # the package rather than by the module list.
    ADD_MENU = False

    def register():
        pass

    def unregister():
        pass


def _selftest():
    """The eligibility rules, which decide what the sidebar covers.

    They are stated as categories rather than as a list of names, so the
    thing worth checking is that each category actually catches what it
    is meant to -- a rule that quietly matches nothing would show up as
    a missing panel long after the change that caused it.
    """
    class Op:
        bl_idname = 'mesh.torus_knot_add'

        def execute(self, context):
            pass

    ok = is_generator(Op)
    print("live: a plain add operator is live-editable")

    class Transformer(Op):
        bl_idname = 'object.leonardo_add'

        @classmethod
        def poll(cls, context):
            return True

    ok = ok and not is_generator(Transformer)
    print("live: an operator with its own poll transforms a selection "
          "and is excluded")

    class Importer(Op):
        bl_idname = 'mesh.thing_import'
        filename_ext = '.txt'

    ok = ok and not is_generator(Importer)
    print("live: an Import/Export helper is excluded")

    class OptedOut(Op):
        bl_idname = 'mesh.relief_panel_add'

    class Helper(Op):
        bl_idname = 'relief.layer_add'

    ok = ok and not is_generator(OptedOut) and not is_generator(Helper)
    print("live: generators with a bespoke panel, and their helpers, "
          "are excluded")

    class NoExecute:
        bl_idname = 'mesh.nothing_add'

    class NoIdname:
        bl_idname = ''

    ok = ok and not is_generator(NoExecute) and not is_generator(NoIdname)

    # A poll inherited from another of the add-on's operators still
    # means "acts on a selection", so the search has to walk the MRO --
    # while whatever Blender's own base classes define is true of every
    # operator and must not disqualify anything.
    class Subclassed(Transformer):
        bl_idname = 'object.leonardo_variant_add'

    ok = ok and not is_generator(Subclassed)

    class FromBlender:
        __module__ = 'bpy_types'

        @classmethod
        def poll(cls, context):
            return True

    class Ordinary(Op, FromBlender):
        bl_idname = 'mesh.child_add'

    ok = ok and is_generator(Ordinary)
    print("live: an inherited poll disqualifies, but Blender's own "
          "base classes do not")
    print("RESULT:", "OK" if ok else "BAD")
    assert ok
