# Geometry Nodes modifier inputs, across Blender versions.
#
# Up to Blender 5.1 a geometry-nodes modifier stored the values of its
# group inputs as ID properties keyed by socket identifier:
#
#     mod["Socket_2"] = 1.5
#
# From 5.2 those are ordinary RNA properties under `mod.properties.inputs`,
# one struct per socket carrying `value`, `type` (VALUE or ATTRIBUTE) and
# `attribute_name`:
#
#     mod.properties.inputs.Socket_2.value = 1.5
#
# and the modifier refuses ID properties outright -- item assignment and
# even `keys()` raise TypeError.  Every module that drives a modifier
# input goes through here so the version split lives in one place.
#
#     from . import gn_inputs
#     gn_inputs.set_input(mod, item.identifier, value)


def _inputs(mod):
    """The 5.2+ per-socket input structs, or None on older Blender."""
    props = getattr(mod, 'properties', None)
    if props is None:
        return None
    return getattr(props, 'inputs', None)


def set_input(mod, identifier, value):
    """Set the group input `identifier` (e.g. "Socket_2") on `mod`."""
    inputs = _inputs(mod)
    if inputs is None:
        mod[identifier] = value
    else:
        getattr(inputs, identifier).value = value


def get_input(mod, identifier):
    """Read the group input `identifier` from `mod`."""
    inputs = _inputs(mod)
    if inputs is None:
        return mod[identifier]
    return getattr(inputs, identifier).value


def input_data_path(mod, identifier):
    """RNA path to the input's value, relative to the modifier's object --
    what a driver variable's `data_path` wants."""
    base = 'modifiers["%s"]' % mod.name.replace('"', '\\"')
    if _inputs(mod) is None:
        return '%s["%s"]' % (base, identifier)
    return '%s.properties.inputs.%s.value' % (base, identifier)


# Per-socket settings worth carrying over, beyond the value itself:
# whether the input reads a named attribute, and which one.
_SOCKET_FIELDS = ('type', 'attribute_name', 'layer_name', 'value')


def copy_inputs(src, dst):
    """Copy every group-input setting from modifier `src` to `dst`.

    `dst` must already have the same node group assigned.  Inputs that
    cannot take a value (the geometry socket) or that `dst` lacks are
    skipped rather than raised on."""
    inputs = _inputs(src)
    if inputs is None:
        # pre-5.2: values, `_use_attribute` and `_attribute_name`
        # flags are all plain keys.  Modifier types that cannot hold
        # ID properties say so by raising.
        try:
            keys = src.keys()
        except TypeError:
            return
        for key in keys:
            try:
                dst[key] = src[key]
            except (KeyError, TypeError, ValueError):
                pass
        return
    out = _inputs(dst)
    if out is None:
        return
    for prop in inputs.bl_rna.properties:
        ident = prop.identifier
        if ident in ('rna_type', 'name'):
            continue
        a = getattr(inputs, ident, None)
        b = getattr(out, ident, None)
        if a is None or b is None:
            continue
        for field in _SOCKET_FIELDS:
            if not hasattr(a, field):
                continue
            try:
                setattr(b, field, getattr(a, field))
            except (AttributeError, TypeError, ValueError):
                pass


def _selftest():
    """Exercise the helpers on a stand-in for each API generation."""

    class _Legacy(dict):
        name = "GN"

    old = _Legacy()
    set_input(old, "Socket_2", 1.5)
    assert get_input(old, "Socket_2") == 1.5
    assert input_data_path(old, "Socket_2") == 'modifiers["GN"]["Socket_2"]'
    dup = _Legacy()
    copy_inputs(old, dup)
    assert dup["Socket_2"] == 1.5

    class _Socket:
        def __init__(self):
            self.value = 0.0
            self.type = 'VALUE'
            self.attribute_name = ""

    class _Prop:
        def __init__(self, identifier):
            self.identifier = identifier

    class _RNA:
        properties = [_Prop('rna_type'), _Prop('Socket_2')]

    class _Inputs:
        bl_rna = _RNA()

        def __init__(self):
            self.Socket_2 = _Socket()

    class _Props:
        def __init__(self):
            self.inputs = _Inputs()

    class _Modern:
        name = "GN"

        def __init__(self):
            self.properties = _Props()

    new = _Modern()
    set_input(new, "Socket_2", 2.5)
    assert get_input(new, "Socket_2") == 2.5
    assert input_data_path(new, "Socket_2") == \
        'modifiers["GN"].properties.inputs.Socket_2.value'
    new.properties.inputs.Socket_2.type = 'ATTRIBUTE'
    new.properties.inputs.Socket_2.attribute_name = "height"
    dup = _Modern()
    copy_inputs(new, dup)
    s = dup.properties.inputs.Socket_2
    assert (s.value, s.type, s.attribute_name) == (2.5, 'ATTRIBUTE', "height")
    print("gn_inputs selftest OK")
