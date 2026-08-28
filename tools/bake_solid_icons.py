"""Bake one preview icon per saddle polyhedron.

The Saddle Polyhedron operator picks its solid from a GALLERY -- a grid
of thumbnails rather than a text list -- because thirty entries named
"Truncated tetragonal tetrahedron" and "Tetragonal saddle hexahedron"
are not tellable apart by name, and the whole point of the inventory is
what the solids look like.

Blender draws that with `template_icon_view` over an enum whose items
carry preview icon ids, so each solid needs a small PNG.  This renders
them with the same studio rig the menu icons and documentation figures
use, so a solid's gallery thumbnail and its doc figure cannot drift.

    blender --background --factory-startup --python tools/bake_solid_icons.py

Renders only what is missing; pass --all to force a re-bake.  Output
goes to `math_art/icons/solids/<KEY>.png`, which ships inside the
extension zip.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "docs"))
sys.path.insert(0, os.path.join(ROOT, "math_art"))

import bpy                                                # noqa: E402

# bake_menu_icons owns the rig setup, the alpha crop and the write; it
# registers math_art on import via render_docs.  Reuse it rather than
# duplicating a second, drifting copy of the same studio.
import bake_menu_icons as bmi                             # noqa: E402
import pearce_data as pdata                               # noqa: E402

OUT_DIR = os.path.join(ROOT, "math_art", "icons", "solids")

#: A gallery cell is drawn far larger than a menu row, so these are
#: rendered at 128 rather than the menu icons' 64 -- at 64 the solids
#: come out as pale blobs and half of them are not tellable apart,
#: which defeats the point of a gallery.
RES = 128

#: A solid is a closed saddle shell; the plan view of several is a flat
#: silhouette (the diamond tetrahedron reads as a hexagon straight down
#: its 3-fold axis), so everything is shot from the same three-quarter
#: angle the doc figures use.
ORIENT = (0.95, 0.0, 0.5)


def _build(key):
    """Add one solid, framed the way the gallery wants it."""
    bpy.ops.mesh.saddle_polyhedron_add(
        solid=key, face_style='MINIMAL', density=3, smoothness=25,
        layout_kind='SINGLE', smooth=True, colour_by='FACE')


def bake(key, path):
    import render_docs as rd
    rd.clear_sculpts()
    bpy.ops.object.select_all(action='DESELECT')
    try:
        bpy.ops.mesh.saddle_polyhedron_add(
            solid=key, face_style='MINIMAL', density=3, smoothness=25,
            layout_kind='SINGLE', smooth=True, colour_by='FACE')
    except TypeError:
        # older signature still carries the group selector
        _build(key)
    obj = bpy.context.view_layer.objects.active
    if obj is not None:
        obj.rotation_euler = ORIENT
    bmi._render_to(path)


def main(force=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    bmi.RES = RES                    # gallery cells are large
    bmi._setup()
    made = skipped = failed = 0
    for solid in pdata.SOLIDS:
        key = solid['key']
        path = os.path.join(OUT_DIR, "%s.png" % key)
        if os.path.exists(path) and not force:
            skipped += 1
            continue
        try:
            bake(key, path)
            made += 1
            print("[%2d] OK   %s" % (solid['number'], key), flush=True)
        except Exception as exc:
            failed += 1
            print("[%2d] FAIL %s: %s" % (solid['number'], key, exc),
                  flush=True)
    print("\nbaked %d, skipped %d, failed %d -> %s"
          % (made, skipped, failed, OUT_DIR))


main('--all' in sys.argv)
