
# Turning a generator operator into an editable settings group.
#
# A Blender operator declares its parameters as annotations holding
# `_PropertyDeferred` objects -- the un-invoked result of
# `bpy.props.IntProperty(...)` and friends.  Those deferred definitions
# can be handed to a second class verbatim, which is what this module
# does: it builds a PropertyGroup carrying the SAME properties as the
# operator, with the same names, defaults, ranges, units, enum items and
# tooltips, plus an `update=` callback that the operator's own
# properties do not need.
#
# The class body is copied too, so the settings group ends up with the
# operator's own `draw()` and `execute()`.  That is the whole point: the
# sidebar draws the generator's real creation UI rather than a
# hand-maintained imitation of it, and a rebuild runs the generator's
# real code rather than a second implementation.  Neither can drift.
#
# Two shims make an operator's code accept a PropertyGroup as `self`:
#
#   `properties`  -- `bpy_extras.object_utils.object_data_add()` reaches
#                    for `operator.properties`; every AddObjectHelper
#                    generator goes through it.
#   `report`      -- operators narrate with `self.report()`; the messages
#                    are kept so the panel can show the last one.

import sys

try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# Properties that belong to a file dialog, not to a generated object.
# Cloning them would put a file path in the sidebar and, worse, make a
# rebuild try to read it.
FILE_PROPS = frozenset({
    'filepath', 'filename', 'directory', 'files', 'filter_glob',
    'check_existing', 'filename_ext', 'hide_props_region',
})

# Placement properties from `AddObjectHelper`.  They are CLONED, because
# `object_data_add()` cannot run without them, but they are never drawn:
# an object that already exists has a transform, and offering a second
# one that only applies to the scratch build would be a control that
# appears to do nothing.
PLACEMENT_PROPS = frozenset({'align', 'location', 'rotation'})

# RNA bookkeeping that every PropertyGroup carries and no generator
# declares.  `name` is the trap: a PropertyGroup inherits one, so a
# settings group appears to have a "Name" setting that the operator it
# mirrors has never heard of.  Left in, it draws a stray Name field in
# the sidebar and makes the settings un-replayable, because passing
# `name=` to the operator is an error.
INTERNAL_PROPS = frozenset({'rna_type', 'name'})

# Class members that must not travel from the operator to the settings
# group.  `bl_*` would confuse registration; the interactive entry points
# (`poll`, `invoke`, `modal`, `cancel`) are never called on a settings
# group and would only mislead anyone reading it.
CLASS_SKIP = frozenset({
    '__dict__', '__weakref__', '__annotations__',
    'bl_rna', 'bl_idname', 'bl_label', 'bl_options', 'bl_description',
    'bl_translation_context', 'bl_undo_group', 'bl_cursor_pending',
    'poll', 'invoke', 'modal', 'cancel', 'check', 'poll_message_set',
    'is_registered', 'order',
})


def _deferred(annotation):
    """True if an annotation is an un-invoked `bpy.props` definition."""
    return hasattr(annotation, 'function') and hasattr(annotation,
                                                       'keywords')


def _resolve(annotation, owner):
    """A property definition, even when the annotation is a string.

    A module written with `from __future__ import annotations` keeps its
    annotations unevaluated, so a class defined there hands out source
    text rather than `_PropertyDeferred` objects.  Blender evaluates
    those itself at registration; a clone has to do the same or the
    properties simply go missing.

    This is not hypothetical: `bpy_extras.object_utils` is written that
    way, so `AddObjectHelper` -- the mixin behind 22 of this add-on's
    generators -- provides `align`, `location` and `rotation` as strings.
    Without them `object_data_add()` raises on every rebuild.
    """
    if _deferred(annotation):
        return annotation
    if not isinstance(annotation, str):
        return None
    namespace = {}
    module = sys.modules.get(getattr(owner, '__module__', ''), None)
    if module is not None:
        namespace.update(vars(module))
    if _IN_BLENDER:
        namespace.update(vars(bpy.props))
    try:
        resolved = eval(annotation, namespace)          # noqa: S307
    except Exception:
        return None
    return resolved if _deferred(resolved) else None


def _chain(existing, ours):
    """Run the operator's own update callback, then the rebuild trigger."""
    if existing is None:
        return ours

    def chained(self, context):
        existing(self, context)
        ours(self, context)
    return chained


def clone_annotations(op_cls, update):
    """Every property definition on an operator, ready for a PropertyGroup.

    The MRO is walked base-first so a mixin's properties
    (`AddObjectHelper`) are picked up and an operator that redefines one
    still wins.  Any `update=` the operator already declared is kept and
    CHAINED in front of ours: 30 properties in this add-on use one to
    apply a preset, and dropping them would break every preset menu.
    """
    out = {}
    for klass in reversed(op_cls.__mro__):
        for name, ann in getattr(klass, '__annotations__', {}).items():
            if name in FILE_PROPS:
                continue
            ann = _resolve(ann, klass)
            if ann is None:
                continue
            kw = dict(ann.keywords)
            # SKIP_SAVE is meaningful for an operator re-run and wrong
            # for stored settings, which exist precisely to be saved.
            kw.pop('options', None)
            kw['update'] = _chain(kw.get('update'), update)
            out[name] = ann.function(**kw)
    return out


if _IN_BLENDER:

    # The last message each settings group reported, so the panel can
    # show what the generator said about the build it just did.
    LAST_REPORT = {}

    def _report(self, level, message):
        LAST_REPORT[self.bl_rna.identifier] = (
            tuple(level)[0] if level else 'INFO', message)

    def settings_class(op_cls, slug, update):
        """Build (and return) the PropertyGroup mirroring `op_cls`."""
        namespace = {k: v for k, v in op_cls.__dict__.items()
                     if k not in CLASS_SKIP}
        namespace['__annotations__'] = clone_annotations(op_cls, update)
        namespace['report'] = _report
        # `properties` on an operator is the operator's own property
        # collection; on the settings group that is the group itself.
        namespace['properties'] = property(lambda self: self)
        namespace['math_art_idname'] = op_cls.bl_idname
        return type('MathArtLive_' + slug, (bpy.types.PropertyGroup,),
                    namespace)

    def _prop_value(owner, prop):
        value = getattr(owner, prop.identifier)
        if getattr(prop, 'is_array', False):
            return tuple(value)
        return value

    def editable_props(pg_cls):
        """The properties of a settings group, in declaration order.

        `rna_type` is RNA bookkeeping and the placement trio is the
        scratch build's business, not the user's.
        """
        return [p for p in pg_cls.bl_rna.properties
                if p.identifier not in INTERNAL_PROPS
                and p.identifier not in PLACEMENT_PROPS
                and not p.is_readonly]

    def copy_values(src, dst):
        """Copy every shared property value from one struct to another.

        Used twice: to record an operator's settings onto the object it
        just built, and to carry settings across when a rebuild has to
        replace the object rather than reuse it.
        """
        copied = 0
        for prop in dst.bl_rna.properties:
            name = prop.identifier
            if name in INTERNAL_PROPS or prop.is_readonly:
                continue
            if not hasattr(src, name):
                continue
            try:
                setattr(dst, name, _prop_value(src, prop))
                copied += 1
            except (TypeError, ValueError, AttributeError):
                # A value the destination cannot hold -- an enum whose
                # items are computed and no longer offer it, typically.
                # Losing one setting is better than losing the record.
                pass
        return copied

    def signature(pg):
        """A short string that changes whenever any setting changes.

        Compared against the signature stored at the last build, this is
        what lets the panel say "these settings are not what you are
        looking at" without rebuilding to find out.
        """
        parts = []
        for prop in pg.bl_rna.properties:
            if prop.identifier in INTERNAL_PROPS or prop.is_readonly:
                continue
            try:
                parts.append('%s=%r' % (prop.identifier,
                                        _prop_value(pg, prop)))
            except AttributeError:
                pass
        return '|'.join(parts)

    def reset_to_defaults(pg):
        """Put every setting back to the value the Add menu would use."""
        for prop in pg.bl_rna.properties:
            if prop.identifier in INTERNAL_PROPS or prop.is_readonly:
                continue
            try:
                if getattr(prop, 'is_array', False):
                    setattr(pg, prop.identifier, tuple(prop.default_array))
                else:
                    setattr(pg, prop.identifier, prop.default)
            except (TypeError, ValueError, AttributeError):
                pass


def _fake_property(**keywords):
    """A stand-in `_PropertyDeferred` for the self-test below.

    It lives at module level because `_resolve` evaluates a string
    annotation in the namespace of the module that declared it, which is
    the behaviour under test.
    """
    return type('_FakeDeferred', (),
                {'function': _fake_property, 'keywords': keywords})()


def _selftest():
    """The chaining and skip rules, without Blender.

    `clone_annotations` needs real `bpy.props` deferreds, so the part
    that can be checked headlessly is the ordering contract of `_chain`
    -- which is where a mistake would be silent: a preset that stops
    applying still looks like a working panel.
    """
    calls = []

    def first(self, context):
        calls.append('operator')

    def second(self, context):
        calls.append('rebuild')

    _chain(first, second)(None, None)
    ok = calls == ['operator', 'rebuild']
    print("live.clone: an existing update callback runs before the "
          "rebuild trigger: %s" % (' -> '.join(calls),))

    calls.clear()
    _chain(None, second)(None, None)
    ok = ok and calls == ['rebuild']

    ok = ok and 'filepath' in FILE_PROPS and 'align' in PLACEMENT_PROPS
    # Placement must be cloned even though it is never drawn, or
    # `object_data_add` raises on every AddObjectHelper generator.
    ok = ok and not (FILE_PROPS & PLACEMENT_PROPS)
    ok = ok and 'poll' in CLASS_SKIP and 'draw' not in CLASS_SKIP
    ok = ok and 'execute' not in CLASS_SKIP
    print("live.clone: draw and execute are copied, poll and invoke are not")

    # A deferred property definition is recognised by shape, not by
    # type, so the check has to be shape-based here too.
    class _Fake:
        function = None
        keywords = {}
    ok = ok and _deferred(_Fake()) and not _deferred(3)

    # A string annotation -- what a module written with
    # `from __future__ import annotations` hands out, which is exactly
    # how `AddObjectHelper` declares align/location/rotation -- has to
    # resolve in the defining module's namespace.  When this regressed,
    # every AddObjectHelper generator rebuilt into a TypeError about a
    # missing `location`, so it is worth a test that needs no Blender.
    class _Owner:
        pass

    resolved = _resolve("_fake_property(name='Align')", _Owner)
    ok = ok and resolved is not None and resolved.keywords == {
        'name': 'Align'}
    print("live.clone: a string annotation resolves to a real property "
          "definition")
    ok = ok and _resolve("NoSuchProperty()", _Owner) is None
    ok = ok and _resolve(3, _Owner) is None
    ok = ok and _resolve("'just a type hint'", _Owner) is None
    print("live.clone: unresolvable and non-property annotations are "
          "skipped rather than raising")
    print("RESULT:", "OK" if ok else "BAD")
    assert ok
