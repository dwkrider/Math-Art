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

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "docs"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# render_docs registers math_art on import and owns the studio rig --
# the dome, the four-light setup, the 2 m-cube normalisation and the
# white-plastic fallback material.  Sharing it is the point: menu icons
# that match the documentation renders come for free, and there is only
# one camera to tune.
import render_docs as rd                                  # noqa: E402

from math_art import menu_defs, menu_icons                # noqa: E402

# Subject parameters, poses and the plan-view set are shared with
# docs/render_docs.py so a figure and its menu icon cannot drift.
import subjects                                           # noqa: E402

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


def _invoke(op, kwargs):
    mod, _, fn = op.partition('.')
    getattr(getattr(bpy.ops, mod), fn)(**kwargs)


#: The icon look, which is NOT the docs look, and the reasoning is worth
#: keeping because two earlier attempts fixed one half and broke the
#: other.  An icon fails in two different ways depending on its subject:
#:
#: * Coloured subjects wash out.  AgX at -0.5 -- right for a figure on a
#:   page -- drives every channel to clipping, and the saddle palette,
#:   whose colours are (0.85, 0.30, 0.24) and friends, measured 0.21
#:   mean saturation.
#: * White subjects go flat.  render_docs' rig runs TWO rim lights at
#:   750 W each against a 320 W key.  A rim is meant to draw an edge; at
#:   more than twice the key it wraps round and fills the shadow side,
#:   and a white plastic subject keeps no gradient to read the form by.
#:
#: Exposure alone cannot serve both, because it scales the whole rig
#: together: pulling it down to Standard/-3.5 lifted saturation to 0.36
#: and left white subjects a flat grey (0.156 luminance spread).  Two
#: independent changes are needed, one per failure.
#:
#: The tone curve does the colour half.  Khronos PBR Neutral exists to
#: roll highlights off WITHOUT the desaturation and hue shift a filmic
#: curve introduces, which is exactly the complaint; it beat every AgX
#: and Standard variant on saturation and on shading at once.  The rim
#: scale does the shading half, bringing the rims back under the key
#: where three-point practice puts them.
#:
#: The EXPOSURE is set by the convex subjects, and this is the part a
#: sweep on one spiky solid gets wrong.  A star self-shadows, so it
#: shows a wide luminance range under any rig and reads fine at -1.0.
#: A convex white solid inside a five-light surround does not: measured
#: at -1.0, the geodesic sphere renders with mean luminance 0.875 and a
#: 5th-to-95th-percentile range of 0.48 to 0.98, so its shading is real
#: (spread 0.157) but squeezed into the top eighth of the scale where
#: the eye cannot see it.  Nothing is clipped -- it is simply all white.
#: Dropping to -2.5 puts that same sphere at mean 0.62 over a 0.23-0.83
#: range, and the facets appear.  Judge exposure on a BALL, not a star.
#:
#: Measured against Standard/-3.5, on the tetrahedral decahedron:
#: saturation 0.43 vs 0.36; on the geodesic sphere, luminance spread
#: 0.20 vs 0.16 with the range no longer crushed against white.
#: Re-measure with tools/icon_lighting_sweep.py before changing these.
VIEW_TRANSFORM = 'Khronos PBR Neutral'
VIEW_TRANSFORM_FALLBACK = ('AgX', 'Standard')
EXPOSURE = -2.5
RIM_SCALE = 0.35


def _icon_look(scene, plan=False):
    """Colour management and light ratios for an icon, not a figure.

    Call this AFTER `subjects.aim_rig`, never only before it.  aim_rig
    re-points the rig per subject and, in doing so, restores each
    light's energy from the snapshot `capture_rig` took and sets the
    view transform and exposure itself -- so anything configured in
    `_setup` alone is silently overwritten on every bake.  That is what
    made the earlier colour-management fix look like it had no effect
    on the menu icons: it never reached them.

    Restoring-then-scaling is also why the rim multiply here is safe to
    run once per bake: aim_rig has just reset the energy to the
    captured full value, so the scale never compounds.

    Plan views are left alone.  They are flat coloured tilings shot
    head-on, and subjects.aim_rig already dims the rig to a quarter and
    kills the specular lobe for them -- a separate, working fix for a
    different problem.
    """
    if plan:
        return
    vs = scene.view_settings
    for name in (VIEW_TRANSFORM,) + VIEW_TRANSFORM_FALLBACK:
        # `view_transform` is a DYNAMIC enum -- its items come from the
        # loaded OCIO config at runtime, so bl_rna reports none of them
        # and the only way to test a name is to assign it and look.
        try:
            vs.view_transform = name
        except TypeError:
            continue
        if vs.view_transform == name:
            break
    vs.exposure = EXPOSURE
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.name.startswith("Rim Light"):
            obj.data.energy *= RIM_SCALE


def _setup():
    """Studio rig, tuned for a small icon on a transparent background."""
    rd.setup_studio()
    subjects.capture_rig()
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
    _icon_look(scene)
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
    """Bake one operator's icon.  Returns None on success, else why not.

    A few subjects render in an environment of their own -- a gemstone
    lit by the plastic-studio rig with nothing to reflect comes out
    black.  It has to stay up through the render, so it is torn down
    here rather than inside `_bake`, whichever way that returns.
    """
    teardown = subjects.enter_environment(op)
    try:
        return _bake(op)
    finally:
        teardown()


def _bake(op):
    rd.clear_sculpts()
    plan = op in subjects.PLAN_VIEW
    subjects.aim_rig(plan)
    # aim_rig has just overwritten the view transform, the exposure and
    # every light's energy; put the icon look back on top of it.
    _icon_look(bpy.context.scene, plan)
    # Operators that transform a selection need something to act on;
    # the setup builds it, and it is dropped once consumed so only the
    # generated surface is framed and rendered.
    helpers = subjects.run_setup(op)
    try:
        _invoke(op, subjects.params_for(op))
    except Exception as e:
        subjects.drop_setup(helpers)
        return f"operator failed: {e!r}"
    subjects.drop_setup(helpers)
    # Working geometry an operator leaves beside its result -- an
    # editable motif, guide rings -- must come out of the frame before
    # anything is measured, because the framing fits the COMBINED
    # bounding box.  render_docs.py does this for the hero and variant
    # shots; without it here the icon and the hero disagree, and the
    # symmetric sculpture's icon shrank to a speck above its guides.
    subjects.hide_helpers(op)
    subs = rd.subjects()
    if not subs:
        return "operator produced no mesh or curve"
    # Diagnose the empty-frame case up front: a mesh with vertices but no
    # faces, or a curve with no bevel, renders nothing in Cycles, and
    # "fully transparent frame" is a confusing way to hear about it.
    # Counted on the EVALUATED object: a generator may store nothing but
    # a point cloud and build every face in a Geometry Nodes modifier --
    # the symmetric sculpture instances its motif over 30 rotation
    # points -- so the stored mesh says "0 faces" for a subject that
    # renders perfectly well.
    deps = bpy.context.evaluated_depsgraph_get()

    def _counts(o):
        try:
            data = o.evaluated_get(deps).data
        except Exception:
            data = o.data
        return (len(getattr(data, 'vertices', ())),
                len(getattr(data, 'polygons', ())))

    tally = [_counts(o) for o in subs if o.type == 'MESH']
    if not any(f for _v, f in tally) and \
            not any(o.type == 'CURVE' for o in subs):
        return (f"no renderable faces "
                f"({sum(v for v, _f in tally)} verts, 0 faces)")
    subjects.pose_subjects(op, rd.subjects())
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
    ops = [o for o in menu_defs.bakeable_ops() if o not in subjects.SKIP]
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
              f"{len(subjects.SKIP)} skipped")
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
          f"skipped {len(subjects.SKIP)}")
    for op, why in failed:
        print(f"  FAIL {op}: {why}")
    print("DONE")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
