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


#: The icon look, which is the docs studio look with ONE change.
#:
#: The tone curve and the rim scale live in subjects.py, because
#: aim_rig is the call every render path makes and setting them there
#: is what stops hero figures, gallery variants and icons drifting
#: apart.  The reasoning for both is recorded beside them.
#:
#: What differs here is the exposure, and only the exposure.  An icon
#: is cropped to 64 px, where a shallow gradient has far fewer pixels
#: to read across than the same gradient has in a 720 px figure, so it
#: wants to sit lower on the curve: -2.5 against the figures' -2.0.
#: Measured on the tetrahedral decahedron, saturation 0.43 against the
#: 0.36 of the Standard/-3.5 look this replaced; on a geodesic sphere,
#: luminance spread 0.20 against 0.16, with the range no longer
#: crushed up against white.
#:
#: Re-measure with tools/icon_lighting_sweep.py before changing this,
#: and judge exposure on a BALL, not a star -- a spiky solid
#: self-shadows and looks fine two stops too bright.
VIEW_TRANSFORM = subjects.STUDIO_VIEW_TRANSFORM
VIEW_TRANSFORM_FALLBACK = ('AgX', 'Standard')
EXPOSURE = -2.5

#: Per-operator exceptions, for subjects that are not lit by the studio
#: rig at all.  `exposure` replaces EXPOSURE; `world` multiplies the
#: strength of every background shader in the world the subject brought.
#:
#: The two gemstones are the case.  subjects._env_gem_studio hides the
#: docs lights entirely and installs the add-on's own Gem Studio -- a
#: sky world plus a small key and fill -- because a faceted stone has
#: no appearance of its own, only what it refracts.  EXPOSURE was
#: measured on diffuse plastic under the studio rig, so applying it
#: here just underexposed a different rig by two stops: the cabochon
#: baked at mean luminance 0.14, a near-black lozenge in the menu.
#:
#: Raising the SKY rather than the exposure is what fixes a stone,
#: because most of its light arrives through the world, and it lifts
#: the shadowed body without flattening the facets the way more key
#: light would.  At -1.5 with the sky at x3 the cabochon reaches 0.42
#: and the faceted stone 0.72.  x6 lifts the cabochon further but
#: costs the faceted stone its facets (saturation 0.19, p95 0.97 --
#: it goes pale and the edges stop reading), so x3 is the ceiling.
LOOK_OVERRIDES = {
    "mesh.gem_add": dict(exposure=-1.5, world=3.0),
    "mesh.gem_cabochon_add": dict(exposure=-1.5, world=3.0),
}


def _boost_world(scene, factor):
    """Scale every background shader in the current world.

    The base strength is remembered on the world datablock the first
    time it is touched, so re-applying the boost to a world that
    survives between bakes sets it rather than compounding it.
    """
    w = scene.world
    if w is None or not w.use_nodes:
        return
    for node in w.node_tree.nodes:
        inp = node.inputs.get("Strength") if node.inputs else None
        if inp is None:
            continue
        key = "_icon_base_strength_%s" % node.name
        base = w.get(key)
        if base is None:
            base = float(inp.default_value)
            w[key] = base
        inp.default_value = base * factor


def _icon_look(scene, plan=False, op=None):
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
    over = LOOK_OVERRIDES.get(op, {})
    vs = scene.view_settings
    subjects._set_view_transform(scene, VIEW_TRANSFORM,
                                 *VIEW_TRANSFORM_FALLBACK)
    vs.exposure = over.get("exposure", EXPOSURE)
    # Set the rim energies ABSOLUTELY, from the strengths capture_rig
    # recorded, rather than multiplying what is there.  aim_rig has
    # usually applied the same scale already, and a second multiply
    # would compound it to 0.12; bake_solid_icons never calls aim_rig
    # at all, so it needs the scale applied here.  Deriving both from
    # the captured base makes this correct either way, and idempotent.
    for name, base in getattr(subjects, "_LIGHT_ENERGY", {}).items():
        if not name.startswith("Rim Light"):
            continue
        ob = bpy.data.objects.get(name)
        if ob is not None:
            ob.data.energy = base * subjects.STUDIO_RIM_SCALE
    if "world" in over:
        _boost_world(scene, over["world"])


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
    _icon_look(bpy.context.scene, plan, op)
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
