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
sys.path.insert(0, ROOT)

import bpy                                                # noqa: E402

# bake_menu_icons owns the rig, the alpha crop and the write, and
# registers math_art on import via render_docs.  Reuse it rather than
# keeping a second studio that can drift.
import bake_menu_icons as bmi                             # noqa: E402
from math_art.fold_pattern_generator import (_NATURAL_ANGLE,  # noqa: E402
                                             _NATURAL_DRIVE,
                                             _NATURAL_FOLD,
                                             _NATURAL_ROWS,
                                             _NATURAL_SOLVER)

OUT_DIR = os.path.join(ROOT, "math_art", "icons", "folds")

#: A gallery cell is drawn far larger than a menu row, so bake at 128
#: like the saddle solids rather than the menu icons' 64.
RES = 128

#: Per pattern: the icon-specific framing only.  The ring count comes
#: from `_NATURAL_ROWS` and the fold angle from `_NATURAL_FOLD`, both
#: owned by the operator -- so each thumbnail shows exactly what that
#: pattern produces from its own defaults, and cannot drift from it.
#: Only the column count is chosen here, for legibility at 128 px.
CASES = {
    'MIURA':     dict(cols=6),
    'ACCORDION': dict(cols=8),
    # 4x6, the operator's own defaults, NOT a smaller framing.  For a
    # waterbomb the column count is how many cells go round the
    # circumference, so 3x4 closed into a squat, faceted tube while the
    # default 4x6 is visibly cylindrical -- the icon undersold the
    # generator, which is the one thing an icon must not do.
    'WATERBOMB': dict(cols=6),
    'YOSHIMURA': dict(cols=6),
    # Sectors are pinned by the operator (`_PINNED_SIDES`), so `cols` is
    # ignored for these two and they differ only in which pattern is
    # asked for.  Their ring count and fold angle come from the operator
    # tables, so each icon shows what its own defaults produce.
    'HYPAR':     dict(cols=4, steps=14),
    'MONKEY':    dict(cols=6, steps=14),
    'KRESLING':  dict(cols=6),
    'KRESZIG':   dict(cols=6),
    'RESCH':     dict(cols=3),
    'TWIST':     dict(cols=1),
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
        pattern=key, rows=_NATURAL_ROWS[key], cols=case['cols'],
        size=2.0, check=False,
        auto_fold=True,
        # Each pattern's own solver, so the thumbnail shows what the
        # operator's defaults produce.  The Kresling only closes its
        # tube under Bending Paper; baked rigid it reads as a half-open
        # crimp, which is the one thing an icon must not do.
        solver=_NATURAL_SOLVER.get(key, 'RIGID'),
        drive=_NATURAL_DRIVE.get(key, 1.0),
        panel_angle=math.radians(_NATURAL_ANGLE.get(key, 60.0)),
        fold_angle=math.radians(_NATURAL_FOLD[key]),
        # more continuation steps where the fold is deep: the path is
        # solved once and only its end state is kept here, but a deep
        # target still wants smaller strides to stay on the branch.
        steps=case.get('steps', 10), animate=False)
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
