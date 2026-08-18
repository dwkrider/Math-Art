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
}

# Per-operator turntable, in radians, for shapes whose default pose is
# ambiguous from the studio camera.  A tetrahedron sitting face-on reads
# as a flat triangle; a sixth of a turn puts an edge toward the camera
# and it reads as a solid again.
ORIENT = {
    "mesh.regular_solid_add": (0.0, 0.0, 0.62),
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


def _capture_camera():
    """Remember the studio camera pose, and derive the plan-view one.

    Plan view keeps the studio camera's distance from the origin so the
    two framings are comparable, and points it straight down: a camera
    with no rotation looks along -Z.
    """
    cam = bpy.data.objects.get("Studio Camera")
    if cam is None:
        return
    _CAM_POSE['studio'] = (cam.location.copy(), cam.rotation_euler.copy())
    dist = cam.location.length
    _CAM_POSE['plan'] = (Vector((0.0, 0.0, dist)), Euler((0.0, 0.0, 0.0)))


def _aim_camera(plan):
    cam = bpy.data.objects.get("Studio Camera")
    pose = _CAM_POSE.get('plan' if plan else 'studio')
    if cam is None or pose is None:
        return
    cam.location, cam.rotation_euler = pose[0].copy(), pose[1].copy()


def _setup():
    """Studio rig, tuned for a small icon on a transparent background."""
    rd.setup_studio()
    _capture_camera()
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
    _aim_camera(op in PLAN_VIEW)
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
