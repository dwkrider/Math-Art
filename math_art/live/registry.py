
# Registry of the live-editable generators.
#
# One mapping, consulted by every other module in this package: a
# generator operator's idname -> the cloned settings class that mirrors
# it.  It lives in its own module so `clone`, `build`, `capture` and
# `panel` can all reach it without importing one another.
#
# Nothing here touches bpy, so the module imports headlessly.


class GenInfo:
    """What the framework knows about one live-editable generator."""

    __slots__ = ('idname', 'label', 'slug', 'op_cls', 'pg_cls',
                 'orig_execute')

    def __init__(self, idname, label, slug, op_cls, pg_cls):
        self.idname = idname
        self.label = label
        self.slug = slug
        self.op_cls = op_cls
        self.pg_cls = pg_cls
        self.orig_execute = None      # set when `capture` wraps execute

    def __repr__(self):
        return "<GenInfo %s>" % self.idname


# idname -> GenInfo, filled by `live.install()` and emptied by
# `live.uninstall()`.
GENERATORS = {}


def slug_for(idname):
    """`curve.torus_knot_add` -> `curve_torus_knot_add`.

    The slug names the settings slot on an object, so it has to be a
    legal Python identifier and stable across sessions -- a stored
    generator name is matched back to its slot by this function alone.
    """
    return idname.replace('.', '_')


def info_for_object(obj):
    """The GenInfo an object was built by, or None."""
    if obj is None:
        return None
    settings = getattr(obj, 'math_art', None)
    if settings is None or not settings.generator:
        return None
    return GENERATORS.get(settings.generator)


def settings_for_object(obj):
    """(GenInfo, settings group) for an object, or (None, None).

    The pair is what every caller actually wants; splitting it saves the
    second lookup and keeps the "is this a generated object" test in one
    place.
    """
    info = info_for_object(obj)
    if info is None:
        return None, None
    return info, getattr(obj.math_art, info.slug, None)


def _selftest():
    ok = slug_for('curve.torus_knot_add') == 'curve_torus_knot_add'
    ok = ok and slug_for('mesh.relief_panel_add').isidentifier()
    # Every idname in the add-on is <module>.<name>; the slug must stay
    # unique or two generators would share one settings slot.
    samples = ['mesh.torus_knot_add', 'curve.torus_knot_add',
               'mesh.tiling_add', 'mesh.tpms_add']
    ok = ok and len({slug_for(s) for s in samples}) == len(samples)
    print("live.registry: slug mapping is identifier-safe and injective")
    print("RESULT:", "OK" if ok else "BAD")
    assert ok
