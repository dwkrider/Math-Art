
# Baked menu-icon loading for the "Math Art" Add menu.
#
# Icons are rendered offline, one PNG per operator, by
# renders/bake_menu_icons.py and committed under math_art/icons/.  The
# manifest's [build] section excludes only __pycache__/ and *.pyc, so
# that folder ships inside the extension zip with no further wiring.
#
# At register() time each PNG is handed to a bpy preview collection.
# `ImagePreviewCollection.load()` is deferred -- Blender decodes the
# image the first time it is actually drawn -- so registering costs a
# path check per entry, not 117 PNG decodes.
#
# Resolution order for a menu entry (see menu_defs.Entry):
#
#   1. entry.builtin is True  -> the built-in glyph, always.
#   2. a loaded preview exists -> the baked render.
#   3. otherwise               -> the built-in glyph.
#
# Rule 3 is what makes baking incremental: an icons/ folder holding
# three PNGs is a perfectly valid build, and the other 114 entries keep
# the icons they have today.

import os

from . import menu_defs

try:
    import bpy
    import bpy.utils.previews
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


ICON_DIR = os.path.join(os.path.dirname(__file__), 'icons')

_previews = None            # bpy preview collection, or None


def icon_filename(op):
    """PNG basename for an operator id ("mesh.oloid_add" -> ...png)."""
    return op.replace('.', '_') + '.png'


def icon_path(op):
    """Absolute path of the baked icon for `op` (may not exist)."""
    return os.path.join(ICON_DIR, icon_filename(op))


def baked_ops():
    """Operator ids that actually have a PNG on disk right now."""
    if not os.path.isdir(ICON_DIR):
        return []
    return sorted(op for op in menu_defs.bakeable_ops()
                  if os.path.isfile(icon_path(op)))


if _IN_BLENDER:

    def load():
        """Register every baked PNG into a preview collection."""
        global _previews
        if _previews is not None:
            return
        if not os.path.isdir(ICON_DIR):
            return                      # un-baked build: all fallbacks
        _previews = bpy.utils.previews.new()
        for op in menu_defs.bakeable_ops():
            path = icon_path(op)
            if not os.path.isfile(path):
                continue                # not baked yet: fall back
            try:
                _previews.load(op, path, 'IMAGE')
            except Exception as e:      # a bad PNG must not break the menu
                print(f"Math Art: icon {op} failed to load: {e}")

    def unload():
        """Drop the preview collection (Blender leak-checks these)."""
        global _previews
        if _previews is not None:
            bpy.utils.previews.remove(_previews)
            _previews = None

    def icon_kwargs(entry):
        """Icon keyword for `lay.operator()` -- baked or built-in.

        Returns {'icon_value': id} when a baked render should be drawn
        and {'icon': NAME} otherwise, so the caller just splats it.
        """
        if not entry.builtin and _previews is not None:
            if entry.op in _previews:
                # icon_id is 0 when Blender has no icon manager to
                # allocate from -- always in --background, and for a
                # preview whose file would not decode.  Passing
                # icon_value=0 draws nothing at all, so a built-in
                # glyph is strictly the better answer.
                icon_id = _previews[entry.op].icon_id
                if icon_id:
                    return {'icon_value': icon_id}
        return {'icon': entry.icon}

else:                                   # headless: no previews at all

    def load():
        pass

    def unload():
        pass

    def icon_kwargs(entry):
        return {'icon': entry.icon}


def _selftest():
    """Path derivation, and drift between icons/ and the menu table."""
    if icon_filename("mesh.oloid_add") != "mesh_oloid_add.png":
        raise AssertionError("icon_filename() derivation changed")
    if os.path.basename(icon_path("curve.torus_knot_add")) \
            != "curve_torus_knot_add.png":
        raise AssertionError("icon_path() basename derivation changed")

    # Distinct operators must not collide on one filename, or baking
    # one would silently overwrite the other.
    names = {}
    for op in menu_defs.unique_ops():
        names.setdefault(icon_filename(op), []).append(op)
    clashes = {k: v for k, v in names.items() if len(v) > 1}
    if clashes:
        raise AssertionError(f"icon filename collision: {clashes}")

    # Drift: every PNG in icons/ must belong to a bakeable entry.  A
    # leftover file means an operator was renamed or pinned to a
    # built-in glyph and its render was never cleaned up.
    if os.path.isdir(ICON_DIR):
        want = {icon_filename(op) for op in menu_defs.bakeable_ops()}
        have = {f for f in os.listdir(ICON_DIR) if f.endswith('.png')}
        orphans = sorted(have - want)
        if orphans:
            raise AssertionError(
                f"icons/ holds {len(orphans)} file(s) with no menu entry "
                f"(renamed or pinned?): {orphans[:5]}")
        print(f"menu_icons: {len(have)}/{len(want)} icons baked")
    else:
        print("menu_icons: no icons/ yet -- all entries use built-ins")
