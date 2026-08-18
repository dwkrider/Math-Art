"""Bake the Math Art Add-menu icons: one PNG per menu operator.

Every operator listed in `math_art/menu_defs.py` that is not pinned to a
built-in glyph is run once with its default settings, shot with the same
studio rig the documentation renders use, and written to
`math_art/icons/<mesh_foo_add>.png`.  `math_art/menu_icons.py` loads
that folder at register() time; anything missing simply falls back to
the entry's built-in icon, so a partial bake is a valid build.

Run (bakes only the icons that do not exist yet):

    blender --background --factory-startup --python tools/bake_menu_icons.py

Re-bake everything, or just some operators:

    ... --python tools/bake_menu_icons.py -- --all
    ... --python tools/bake_menu_icons.py -- mesh.oloid_add curve.torus_knot_add

List what would be baked, without rendering:

    ... --python tools/bake_menu_icons.py -- --list

On `--factory-startup`: the project's usual warning against it applies
to testing the *installed extension*, whose operators it would switch
off.  This script does what docs/render_docs.py does instead -- imports
`math_art` from the working tree and registers it by hand -- so it is
testing the branch's code, not the installed zip, and the flag keeps the
render free of user startup settings.  Do not "fix" it by removing the
flag; do not use this script to check whether a build installed.
"""
import argparse
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Euler, Vector

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "docs"))

# render_docs registers math_art on import and owns the studio rig --
# the dome, the four-light setup, the 2 m-cube normalisation and the
# white-plastic fallback material.  Sharing it is the point: menu icons
# that match the documentation renders come for free, and there is only
# one camera to tune.
import render_docs as rd                                  # noqa: E402

from math_art import menu_defs, menu_icons                # noqa: E402

ICON_DIR = menu_icons.ICON_DIR

# Final icon size.  Menu rows draw an icon at roughly 16-20 px, so 64 is
# already generous; the render happens at SS x that and is scaled down,
# which is cheaper than sampling a tiny frame into clean edges.
RES = 64
SUPERSAMPLE = 4
SAMPLES = 48

# Empty border kept around the subject, as a fraction of its half-size.
# The frame is cropped to the subject's alpha, so this is the only thing
# setting how much of the icon the shape fills (0.06 -> about 89%).
MARGIN = 0.06

# Exposure for the plan-view shots.  Lighting a flat panel head-on and
# dropping AgX's highlight rolloff (see _aim_rig) drove these to the
# clipping point -- measured mean value 0.99 with 41% of the hyperbolic
# tiling's pixels pinned at white, which is what washed the colour out.
# -1.8 is where clipping reaches exactly zero across the coloured
# subjects; darker than that only dims the icon without adding
# saturation the pale generator palettes do not have.
PLAN_EXPOSURE = -1.8

# Operators that need help: a bare call is too slow, or its defaults
# make a thumbnail that says nothing.  Keep this list short -- an icon
# should show what the user gets when they click the entry.
OVERRIDES = {
    # A bare tetrahedron reads as a flat triangle; the dodecahedron's
    # pentagons say "regular solid" at a glance.
    "mesh.regular_solid_add": dict(family='PLATONIC', solid='DODECA'),
    # And the uniform operator's whole point is what lies beyond the
    # Platonics, so it gets a Kepler-Poinsot star rather than another
    # convex solid that would look like the entry above.
    "mesh.uniform_polyhedron_add": dict(family='KEPLER', solid='34'),
    # The signature Scherk-Collins form (docs/render_docs.py shoots the
    # same preset).
    "mesh.scherk_collins_add": dict(preset='HEX'),
    # The gyroid is the TPMS everyone recognises.
    "mesh.periodic_minimal_add": dict(periodicity='TRIPLY', surface='G'),
    # Fold with Blender's own cloth solver rather than the internal
    # packing: same surface, far better folds to look at.
    "mesh.crochet_add": dict(physics='CLOTH'),
    # A cube's twist is hidden by its own faces; a tetrahedron has few
    # enough that the ribbon reads.
    "mesh.platonic_twist_add": dict(kind='TETRA'),
    # docs/render_docs.py shoots the twisted torus at these values.
    "mesh.twisted_torus_add": dict(n=6, twist_steps=6),
    # Rods flush with the core hide the interleaving; pushing them out
    # two cells at each end shows how the sticks thread past each other.
    "mesh.polystix_add": dict(overhang=2.0),
    # {7,3} is regular, so every face has the same side count and the
    # default "by sides" colouring yields exactly one material.  Parity
    # gives the classic two-tone (with a seam, since q=3 is odd).
    "mesh.hyperbolic_tiling_add": dict(color_by='PARITY'),
    # Light up the 13 parastichy arms rather than shipping a grey disc.
    "mesh.phyllotaxis_add": dict(color_by='PARASTICHY', parastichy=13),
    # Koch is the one fractal everybody has already seen.
    "curve.lsystem_add": dict(kind='PENTAPLEXITY'),
    # EDGE mode offers only the edge-rewriting generators (ANTIKOCH,
    # CESARO, ELEVEN, KOCH, KOCH_SQUARE, LEVY, MINKOWSKI, QUADKOCH,
    # SEVEN); the flowsnake lives under FASS.  Minkowski's square bumps
    # read at icon size and are nobody's mental image of "a fractal".
    "curve.turtle_curve_add": dict(mode='EDGE', teragon='MINKOWSKI'),
}

# Operators whose subject is a flat panel.  The studio rig's 3/4 view
# collapses these to a thin sliver -- measured bounding-box aspect ran
# 0.21-0.38 against a median of 0.9 for the solids -- so they are shot
# from straight overhead instead.  Everything here is a 2D pattern by
# nature; the Patterns entries with genuine relief (relief panel and
# solid, the modular screen, layer groups) keep the 3/4 view because
# their depth is the point.
PLAN_VIEW = {
    "mesh.frieze_add", "mesh.wallpaper_add", "mesh.tiling_add",
    "mesh.kuniform_add", "mesh.monohedral_add", "mesh.isohedral_add",
    "mesh.aperiodic_add", "mesh.reptile_add", "mesh.voderberg_add",
    "mesh.spiral_tiling_add", "mesh.fractal_tiling_add",
    "mesh.fractal_reptile_add", "mesh.islamic_pattern_add",
    "mesh.celtic_knot_2d_add", "mesh.over_under_screen_add",
    "mesh.knot_carpet_add", "mesh.hyperbolic_tiling_add",
    "mesh.map_lsystem_add",
    # curve-based fractals that are drawn in the plane
    "curve.lsystem_add", "curve.turtle_curve_add",
    "curve.substitution_knot_add", "mesh.fractal_knotwork_add",
    "mesh.snowflake_add",
    # a phyllotaxis head is a flat disc: at 3/4 it foreshortens to a
    # pale ellipse and the parastichy colouring is wasted
    "mesh.phyllotaxis_add",
}

# Per-operator turntable, in radians, for shapes whose default pose is
# ambiguous from the studio camera.  A tetrahedron sitting face-on reads
# as a flat triangle; a sixth of a turn puts an edge toward the camera
# and it reads as a solid again.
ORIENT = {
    "mesh.regular_solid_add": (0.0, 0.0, 0.62),
    # The Klein bottle's default pose puts the handle behind the body,
    # so the self-intersection -- the whole point of the surface -- is
    # hidden.  Half a turn brings it to the front.
    "mesh.topological_surface_add": (0.0, 0.0, math.pi),
    # The IFS default is SIERP_TETRA, a Sierpinski *tetrahedron* -- a
    # solid, not a plane figure, so it wants a turn rather than a plan
    # view (from overhead a tetrahedron just squares off).  An eighth
    # turn puts an edge forward and the recursion reads down the faces.
    "mesh.ifs_add": (0.0, 0.0, math.pi / 8),
}

# Operators that cannot be baked at all.  Each needs a reason.
SKIP = {
    # Builds the phyllotaxis seed positions as a points-only mesh (120
    # verts, 0 faces), so Cycles has nothing to shade and the frame comes
    # back empty.  Giving it faces just for the thumbnail would show
    # something the operator does not actually produce, so it keeps its
    # built-in glyph and the gallery draws it as an ordinary row.
    "mesh.receptacle_add": "points-only mesh, nothing for Cycles to shade",
}


def _invoke(op, kwargs):
    mod, _, fn = op.partition('.')
    getattr(getattr(bpy.ops, mod), fn)(**kwargs)


_CAM_POSE = {}          # 'studio' / 'plan' -> (location, rotation_euler)
_LIGHT_POSE = {}        # light name -> (studio location, plan location)

_LIGHT_NAMES = ("Key Light", "Fill Light", "Rim Light L", "Rim Light R",
                "Top Light")


def _capture_rig():
    """Remember the studio camera and lights, and derive plan-view poses.

    The plan camera keeps the studio camera's distance from the origin
    so the two framings are comparable, and points straight down: a
    camera with no rotation looks along -Z.

    The lights get lifted with it.  The studio rig lights a solid from
    the side, which across a flat panel is grazing light -- it rakes the
    surface, blows the highlights and leaves the colours pale.  Each
    light is therefore re-placed at its own distance but high overhead,
    keeping only a fraction of its horizontal offset, so a panel is lit
    nearly head-on and its materials read at full saturation.
    """
    cam = bpy.data.objects.get("Studio Camera")
    if cam is not None:
        _CAM_POSE['studio'] = (cam.location.copy(),
                               cam.rotation_euler.copy())
        _CAM_POSE['plan'] = (Vector((0.0, 0.0, cam.location.length)),
                             Euler((0.0, 0.0, 0.0)))
    for name in _LIGHT_NAMES:
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        loc = ob.location.copy()
        dist = max(loc.length, 1e-6)
        plan = Vector((loc.x * 0.25, loc.y * 0.25, abs(dist)))
        plan.length = dist          # same distance, mostly overhead
        _LIGHT_POSE[name] = (loc, plan)


def _aim_rig(plan):
    """Point camera and lights at the subject for the chosen view."""
    cam = bpy.data.objects.get("Studio Camera")
    pose = _CAM_POSE.get('plan' if plan else 'studio')
    if cam is not None and pose is not None:
        cam.location, cam.rotation_euler = pose[0].copy(), pose[1].copy()
    for name, (studio, overhead) in _LIGHT_POSE.items():
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        ob.location = (overhead if plan else studio).copy()
        # Area lights are aimed by rotation, not constrained, so re-aim
        # each one at the origin after moving it.
        ob.rotation_euler = (-ob.location).to_track_quat('-Z', 'Y').to_euler()

    # AgX rolls highlights off towards white, which is the right look for
    # a 720 px documentation render and the wrong one for a 64 px icon
    # that has to stay legible by colour.  Plan-view subjects are the
    # flat, coloured ones, so they get the untouched Standard transform.
    scene = bpy.context.scene
    vt = [v.name for v in bpy.types.ColorManagedViewSettings.bl_rna
          .properties["view_transform"].enum_items]
    want = "Standard" if plan else ("AgX" if "AgX" in vt else "Standard")
    if want in vt:
        scene.view_settings.view_transform = want
    scene.view_settings.exposure = PLAN_EXPOSURE if plan else -0.5


def _setup():
    """Studio rig, tuned for a small icon on a transparent background."""
    rd.setup_studio()
    _capture_rig()
    scene = bpy.context.scene
    # The docs rig shoots against a near-black dome; an icon has to sit
    # on whatever colour the menu happens to be, so drop the backdrop
    # and let the film go transparent instead.
    dome = bpy.data.objects.get("Backdrop Dome")
    if dome is not None:
        dome.hide_render = True
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES * SUPERSAMPLE
    scene.render.resolution_y = RES * SUPERSAMPLE


def _square_crop(px, margin=MARGIN):
    """Tightest square around the opaque pixels of an (h, w, 4) array.

    Returns a new (s, s, 4) array, transparent where the crop window
    falls outside the rendered frame.  Cropping to the subject rather
    than trusting the camera is what makes every icon fill its box by
    the same amount: the studio rig frames a 2 m cube, but a flat tiling
    panel and a round sphere occupy very different fractions of it.
    """
    ys, xs = np.nonzero(px[..., 3] > 0.01)
    if not len(ys):
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    half = max(x1 - x0, y1 - y0) / 2.0 * (1.0 + 2.0 * margin)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = max(2, int(round(2.0 * half)))

    out = np.zeros((s, s, 4), dtype=np.float32)
    # Source window, clipped to the frame; destination offset by however
    # much the clip removed, so the subject stays centred.
    sx0, sy0 = int(round(cx - half)), int(round(cy - half))
    h, w = px.shape[:2]
    cx0, cy0 = max(0, sx0), max(0, sy0)
    cx1, cy1 = min(w, sx0 + s), min(h, sy0 + s)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0 - sy0:cy1 - sy0, cx0 - sx0:cx1 - sx0] = px[cy0:cy1, cx0:cx1]
    return out


def _render_to(path):
    """Render, crop to the subject, and write a RES x RES PNG."""
    scene = bpy.context.scene
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)

    img = bpy.data.images.load(path)
    cropped = None
    try:
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        out = _square_crop(buf.reshape(h, w, 4))
        if out is None:
            raise RuntimeError("rendered frame is fully transparent")
        s = out.shape[0]
        cropped = bpy.data.images.new("icon_crop", s, s, alpha=True)
        cropped.pixels.foreach_set(out.reshape(-1))
        # Downsample last: Blender's filter on a supersampled crop gives
        # cleaner edges than asking Cycles for a 64 px frame directly.
        cropped.scale(RES, RES)
        cropped.file_format = 'PNG'
        cropped.save(filepath=path)
    finally:
        bpy.data.images.remove(img)
        if cropped is not None:
            bpy.data.images.remove(cropped)


def bake(op):
    """Bake one operator's icon.  Returns None on success, else why not."""
    rd.clear_sculpts()
    _aim_rig(op in PLAN_VIEW)
    try:
        _invoke(op, OVERRIDES.get(op, {}))
    except Exception as e:
        return f"operator failed: {e!r}"
    subjects = rd.subjects()
    if not subjects:
        return "operator produced no mesh or curve"
    # Diagnose the empty-frame case up front: a mesh with vertices but no
    # faces, or a curve with no bevel, renders nothing in Cycles, and
    # "fully transparent frame" is a confusing way to hear about it.
    if not any(len(getattr(o.data, 'polygons', ())) for o in subjects
               if o.type == 'MESH') and \
            not any(o.type == 'CURVE' for o in subjects):
        return (f"no renderable faces "
                f"({sum(len(getattr(o.data, 'vertices', ())) for o in subjects)}"
                f" verts, 0 faces)")
    if op in ORIENT:
        for o in rd.subjects():
            o.rotation_euler = Euler(ORIENT[op])
    rd.normalize_subjects()
    rd.apply_material()
    path = menu_icons.icon_path(op)
    try:
        _render_to(path)
    except Exception as e:
        # _render_to writes the full-size frame before cropping it, so a
        # crop that raises leaves an uncropped PNG behind.  Left in
        # place it would load as a valid-looking icon at the wrong size,
        # which is worse than having no icon at all.
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return f"render failed: {e!r}"
    return None


def _worklist(args):
    if args.ops:
        unknown = sorted(set(args.ops) - set(menu_defs.unique_ops()))
        if unknown:
            sys.exit(f"not in the menu table: {', '.join(unknown)}")
        pinned = sorted(set(args.ops) - set(menu_defs.bakeable_ops()))
        if pinned:
            sys.exit(f"pinned to a built-in icon in menu_defs.py, "
                     f"so a render would never be shown: "
                     f"{', '.join(pinned)}")
        return args.ops
    ops = [o for o in menu_defs.bakeable_ops() if o not in SKIP]
    if not args.all:
        ops = [o for o in ops if not os.path.isfile(menu_icons.icon_path(o))]
    return ops


def main(argv):
    ap = argparse.ArgumentParser(prog="bake_menu_icons")
    ap.add_argument("ops", nargs="*",
                    help="operator ids to bake (default: the missing ones)")
    ap.add_argument("--all", action="store_true",
                    help="re-bake every operator, not just missing ones")
    ap.add_argument("--list", action="store_true",
                    help="print the work-list and exit")
    args = ap.parse_args(argv)

    ops = _worklist(args)
    if args.list:
        for op in ops:
            print(op)
        print(f"{len(ops)} operator(s); "
              f"{len(menu_defs.bakeable_ops())} bakeable in all, "
              f"{len(SKIP)} skipped")
        return

    os.makedirs(ICON_DIR, exist_ok=True)
    _setup()
    ok, failed = [], []
    for i, op in enumerate(ops, 1):
        t0 = time.time()
        why = bake(op)
        dt = time.time() - t0
        if why is None:
            ok.append(op)
            print(f"[{i}/{len(ops)}] OK   {op}  ({dt:.1f}s)", flush=True)
        else:
            failed.append((op, why))
            print(f"[{i}/{len(ops)}] FAIL {op}  ({dt:.1f}s)  {why}",
                  flush=True)
    print(f"\nbaked {len(ok)}, failed {len(failed)}, "
          f"skipped {len(SKIP)}")
    for op, why in failed:
        print(f"  FAIL {op}: {why}")
    print("DONE")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
