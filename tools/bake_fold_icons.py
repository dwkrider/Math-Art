"""Bake one preview icon per classical crease pattern.

Crease Pattern picks its pattern from a GALLERY -- a grid of thumbnails
rather than a dropdown -- because the five are told apart by what they
look like, not by their names.

THE THUMBNAILS SHOW THE FOLDED STATE.  Flat, all five are grids of thin
lines, and at 128 px the Miura, the Yoshimura and the waterbomb are
near enough indistinguishable; folded, they are a corrugation, a tube
and a ball.  The picture has to show the thing being chosen between,
and what the user is choosing is a fold.

    blender --background --factory-startup --python tools/bake_fold_icons.py

Renders only what is missing; pass --all to force a re-bake.  Output
goes to `math_art/icons/folds/<PATTERN>.png`, which ships inside the
extension zip.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "docs"))
sys.path.insert(0, os.path.join(ROOT, "math_art"))

import bpy                                                # noqa: E402

# bake_menu_icons owns the rig, the alpha crop and the write, and
# registers math_art on import via render_docs.  Reuse it rather than
# keeping a second studio that can drift.
import bake_menu_icons as bmi                             # noqa: E402

OUT_DIR = os.path.join(ROOT, "math_art", "icons", "folds")

#: A gallery cell is drawn far larger than a menu row, so bake at 128
#: like the saddle solids rather than the menu icons' 64.
RES = 128

#: Per pattern: the build arguments and how far to fold.  These are not
#: the operator defaults -- a thumbnail wants the reading that shows
#: the pattern's character in one glance, which usually means fewer
#: panels than a working sheet and a deeper fold than 70 degrees.
CASES = {
    'MIURA':     dict(rows=4, cols=6, fold=100.0),
    'ACCORDION': dict(rows=4, cols=8, fold=110.0),
    'WATERBOMB': dict(rows=3, cols=4, fold=90.0),
    'YOSHIMURA': dict(rows=4, cols=6, fold=75.0),
    # The hypar is the awkward one.  Its concentric pleats never show
    # in a thumbnail: it has no planar-facet folding at all (Demaine et
    # al. 2011), so what the solver returns is the gross SADDLE, and
    # measuring it confirms the pleats contribute nothing to the
    # silhouette -- 6, 8 and 10 rings all fold to the same
    # 2.00 x 2.00 x 1.23 box.  So the icon is chosen to show the saddle,
    # which is the hypar's actual signature and is what tells it apart
    # from the other four here: square (4 sides, the classical one) and
    # folded hard enough for the warp to be unmistakable.  A gentle fold
    # reads as a flat slab and is worse than useless.
    'HYPAR':     dict(rows=6, cols=4, fold=120.0),
}

#: Three-quarter view.  A folded corrugation seen straight down reads as
#: a flat grid again -- which is the very thing these icons exist to
#: avoid -- so the camera is kept off the sheet normal.
ORIENT = (math.radians(62), 0.0, math.radians(28))


def bake(key, path):
    import render_docs as rd
    rd.clear_sculpts()
    bpy.ops.object.select_all(action='DESELECT')
    case = CASES[key]
    bpy.ops.mesh.crease_pattern_add(
        pattern=key, rows=case['rows'], cols=case['cols'],
        size=2.0, check=False,
        auto_fold=True, fold_angle=math.radians(case['fold']),
        steps=10, animate=False)
    obj = bpy.context.view_layer.objects.active
    if obj is None:
        raise RuntimeError("no object was created")
    obj.rotation_euler = ORIENT
    bmi._render_to(path)


def main(force=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    bmi.RES = RES                    # gallery cells are large
    bmi._setup()
    made = skipped = failed = 0
    for key in CASES:
        path = os.path.join(OUT_DIR, "%s.png" % key)
        if os.path.exists(path) and not force:
            skipped += 1
            continue
        try:
            bake(key, path)
            made += 1
            print("OK   %s" % key, flush=True)
        except Exception as exc:
            failed += 1
            print("FAIL %s: %s" % (key, exc), flush=True)
    print("\nbaked %d, skipped %d, failed %d -> %s"
          % (made, skipped, failed, OUT_DIR))


main('--all' in sys.argv)
