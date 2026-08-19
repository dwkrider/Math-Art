"""Regenerate the documentation renders.

Every generator is shot with one consistent studio rig -- a black-velvet
dome, soft key/fill/rim/top lights, a white-plastic material on any
object the generator left uncolored (objects that carry their own
materials keep their colors), Cycles + AgX.  Because every generator now
outputs centered on the origin and fitting a 2 m cube, a single camera
and lighting setup frames them all.

The set of figures is the Add menu: every operator in
`math_art/menu_defs.py` gets one, named by `subjects.slug_for()`, which
is also the name of its page under `docs/generators/`.  Adding a
generator therefore needs no edit here at all.

Run (renders the hero figures that are missing):

    blender --background --factory-startup --python docs/render_docs.py

Re-render everything, or just some (by slug):

    ... --python docs/render_docs.py -- --all
    ... --python docs/render_docs.py -- twisted_polyhedron prime_knot

The per-option variant galleries, which are much the larger job:

    ... --python docs/render_docs.py -- variants
    ... --python docs/render_docs.py -- variants regular_solids

List what is present and what is missing, without rendering:

    ... --python docs/render_docs.py -- --list

Tune SAMPLES / RES for quality vs speed.
"""
import bpy
import bmesh
import sys
import os
import math
from mathutils import Vector, Euler

SAMPLES = 96
RES = 720

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "tools"))
import math_art  # noqa: E402
math_art.register()

# The menu table is the list of things that need a figure.
from math_art import menu_defs  # noqa: E402

# Subject parameters, poses and the flat-subject (plan view) set are
# shared with tools/bake_menu_icons.py, so a figure here and the menu
# icon for the same operator cannot drift apart.
import subjects as subject_cfg  # noqa: E402
IMG = os.path.join(PROJ, "docs", "images")
os.makedirs(IMG, exist_ok=True)

STUDIO = {"Backdrop Dome", "Key Light", "Fill Light", "Rim Light L",
          "Rim Light R", "Top Light", "Studio Camera"}


# ---------------------------------------------------------------- studio
def setup_studio():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_uv_sphere_add(radius=30, segments=96,
                                          ring_count=48)
    dome = bpy.context.object
    dome.name = "Backdrop Dome"
    bm = bmesh.new()
    bm.from_mesh(dome.data)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(dome.data)
    bm.free()
    dome.data.polygons.foreach_set("use_smooth",
                                   [True] * len(dome.data.polygons))
    velvet = bpy.data.materials.new("Black Velvet")
    velvet.use_nodes = True
    vb = velvet.node_tree.nodes.get("Principled BSDF")
    vb.inputs["Base Color"].default_value = (0.006, 0.006, 0.008, 1)
    vb.inputs["Roughness"].default_value = 1.0
    for k, v in (("Sheen Weight", 0.6), ("Sheen Roughness", 0.35)):
        if k in vb.inputs:
            vb.inputs[k].default_value = v
    dome.data.materials.append(velvet)
    rr = 1.55

    def area(name, off, energy, size):
        la = bpy.data.lights.new(name, 'AREA')
        la.energy = energy
        la.size = size
        o = bpy.data.objects.new(name, la)
        o.location = Vector(off) * rr
        o.rotation_euler = (-o.location).to_track_quat('-Z', 'Y').to_euler()
        o.visible_camera = False
        scene.collection.objects.link(o)
    area("Key Light", (1.8, -1.9, 1.6), 320 * rr * rr, rr * 3.0)
    area("Fill Light", (-2.4, -0.8, 0.5), 70 * rr * rr, rr * 5.0)
    area("Rim Light L", (-1.7, 1.9, 1.1), 750 * rr * rr, rr * 1.0)
    area("Rim Light R", (1.9, 1.7, 0.8), 750 * rr * rr, rr * 1.0)
    area("Top Light", (0.0, 0.3, 2.6), 150 * rr * rr, rr * 2.4)
    world = scene.world or bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Color"] \
        .default_value = (0.004, 0.004, 0.005, 1)
    cam_d = bpy.data.cameras.new("Studio Camera")
    cam_d.lens = 72
    cam = bpy.data.objects.new("Studio Camera", cam_d)
    scene.collection.objects.link(cam)
    view = Vector((1.35, -2.2, 0.95)).normalized()
    cam.location = view * (rr * 5.0)
    cam.rotation_euler = (-cam.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.film_transparent = False
    vt = [v.name for v in bpy.types.ColorManagedViewSettings.bl_rna
          .properties["view_transform"].enum_items]
    if "AgX" in vt:
        scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.5


_PLASTIC = [None]


def apply_material():
    if _PLASTIC[0] is None:
        m = bpy.data.materials.new("White Plastic")
        m.use_nodes = True
        sb = m.node_tree.nodes.get("Principled BSDF")
        sb.inputs["Base Color"].default_value = (0.84, 0.84, 0.86, 1)
        sb.inputs["Roughness"].default_value = 0.38
        _PLASTIC[0] = m
    for o in bpy.data.objects:
        if o.name in STUDIO or o.type not in ('MESH', 'CURVE'):
            continue
        if not any(m is not None for m in o.data.materials):
            o.data.materials.append(_PLASTIC[0])


def clear_sculpts():
    for o in list(bpy.data.objects):
        if o.name not in STUDIO and o.type in ('MESH', 'CURVE'):
            bpy.data.objects.remove(o, do_unlink=True)


def subjects():
    return [o for o in bpy.data.objects
            if o.name not in STUDIO and o.type in ('MESH', 'CURVE')]


def normalize_subjects(target=2.0):
    """Center all sculpture objects on the origin and scale them so
    their combined (modifier-evaluated) bounding box spans `target`, so
    every generator -- whatever its native size -- frames identically."""
    deps = bpy.context.evaluated_depsgraph_get()
    mn = Vector((1e18, 1e18, 1e18))
    mx = -mn
    found = False
    for o in subjects():
        oe = o.evaluated_get(deps)
        me = None
        try:
            me = oe.to_mesh()
        except Exception:
            me = None
        pts = ([oe.matrix_world @ v.co for v in me.vertices]
               if me and len(me.vertices)
               else [o.matrix_world @ Vector(c) for c in o.bound_box])
        for w in pts:
            found = True
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
        if me is not None:
            oe.to_mesh_clear()
    if not found:
        return
    center = (mn + mx) / 2.0
    ext = max(mx[i] - mn[i] for i in range(3))
    s = (target / ext) if ext > 1e-9 else 1.0
    for o in subjects():
        o.location = (o.location - center) * s
        o.scale = o.scale * s


def render(slug):
    bpy.context.scene.render.filepath = os.path.join(IMG, slug + ".png")
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------- tasks
def styled(base_fn, style_op, **kw):
    def run():
        base_fn()
        ob = bpy.context.active_object
        for o in bpy.context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        getattr(bpy.ops.object, style_op)(**kw)
    run.op = getattr(base_fn, 'op', None)   # carry the subject through
    return run


def O(op, **kw):
    """Task that runs `op` with its canonical parameters.

    Parameters come from tools/subjects.py, which the menu-icon baker
    reads as well, so a figure here and the icon in the Add menu show
    the same object.  Anything passed as `kw` wins, for the rare figure
    that deliberately wants a different variant from the icon.
    """
    mod, _, fn = op.partition('.')
    args = subject_cfg.params_for(op, **kw)
    run = lambda: getattr(getattr(bpy.ops, mod), fn)(**args)   # noqa: E731
    run.op = op          # so main() can find the subject's pose/view
    return run


# The figure set is the MENU, not a list kept alongside it.  Every
# operator in math_art/menu_defs.py gets a figure at docs/images/
# <slug>.png and a page at docs/generators/<slug>.md; the slug comes
# from subjects.slug_for().  Nothing has to be added here when a
# generator is added -- which is the whole point, since the previous
# hand-written table had drifted to 52 of 128 operators, and its entry
# for the periodic minimal surfaces still called an operator id that
# had since been renamed.
#
# Only the exceptions are written down: STYLED, for the operators that
# restyle an existing object and so need one built first, and SCENES,
# for figures that are a whole set-up rather than a subject on the
# studio rig.
def _menu_tasks():
    tasks = {}
    for op in menu_defs.unique_ops():
        if op in subject_cfg.SKIP:
            continue
        tasks[subject_cfg.slug_for(op)] = O(op)
    return tasks


# Operators that restyle whatever is selected rather than adding a
# shape.  Each needs a base object built first, and the base is not the
# subject: the truncated icosahedron under the Leonardo style is just a
# solid with enough faces to show the strut work off.
STYLED = {
    "leonardo": (O("mesh.regular_solid_add", family='ARCHIMEDEAN',
                   solid='TI'), "leonardo_add"),
    "curvature_color": (O("mesh.twisted_torus_add"), "curvature_color_add"),
    "voronoi_openwork": (O("mesh.geodesic_add"), "voronoi_openwork_add"),
    "organic_wireframe": (O("mesh.geodesic_add"), "organic_wireframe_add"),
    "strahler": (O("curve.fractal_tree_add"), "strahler_add"),
}

TASKS = _menu_tasks()
TASKS.update({slug: styled(base, op)
              for slug, (base, op) in STYLED.items()})


# generators whose material is incidental (motif default etc.) -- render
# them in the neutral white plastic instead of their own material
FORCE_PLASTIC = {"symmetric_sculpture"}

# per-generator object rotation (radians) for a more legible pose,
# applied before framing
ROTATE = {
    "tpms": (math.radians(35), math.radians(20), math.radians(15)),
}


# ------------------------------------------------ custom scenes
def scene_stereographic():
    """A perforated sphere with a point light at its north pole casting
    the pattern down onto a plane -- the stereographic projection made
    physical (a la Segerman's shadow sculptures)."""
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Color"] \
        .default_value = (0.0, 0.0, 0.0, 1)
    # perforated sphere: south pole at origin, north pole at (0,0,2R);
    # a hyperbolic {7,3} tiling projects to a Poincare-disc pattern
    bpy.ops.mesh.stereographic_add(pattern='TILING', tile_p=7,
                                   tile_q=3, radius=1.0)
    sph = bpy.context.active_object
    plastic = bpy.data.materials.new("Sphere Plastic")
    plastic.use_nodes = True
    plastic.node_tree.nodes.get("Principled BSDF") \
        .inputs["Roughness"].default_value = 0.45
    sph.data.materials.append(plastic)
    for p in sph.data.polygons:
        p.use_smooth = True
    # catch plane at z=0 (the projection plane)
    bpy.ops.mesh.primitive_plane_add(size=26, location=(0, 0, 0))
    plane = bpy.context.active_object
    pm = bpy.data.materials.new("Catch Plane")
    pm.use_nodes = True
    pb = pm.node_tree.nodes.get("Principled BSDF")
    pb.inputs["Base Color"].default_value = (0.85, 0.85, 0.87, 1)
    pb.inputs["Roughness"].default_value = 0.7
    plane.data.materials.append(pm)
    # bright point light just inside the north pole -> sharp shadow
    la = bpy.data.lights.new("Projector", 'POINT')
    la.energy = 6000
    la.shadow_soft_size = 0.01
    lo = bpy.data.objects.new("Projector", la)
    lo.location = (0.0, 0.0, 1.9)
    scene.collection.objects.link(lo)
    # gentle fill so the sphere body is not pure black
    fa = bpy.data.lights.new("Fill", 'AREA')
    fa.energy = 60
    fa.size = 6
    fo = bpy.data.objects.new("Fill", fa)
    fo.location = (-4, -5, 4)
    fo.rotation_euler = (-Vector(fo.location)).to_track_quat(
        '-Z', 'Y').to_euler()
    fo.visible_camera = False
    scene.collection.objects.link(fo)
    # camera: 3/4 view taking in the sphere and the projected disc
    cam_d = bpy.data.cameras.new("Cam")
    cam_d.lens = 40
    cam = bpy.data.objects.new("Cam", cam_d)
    scene.collection.objects.link(cam)
    cam.location = Vector((5.5, -8.5, 5.0))
    cam.rotation_euler = (Vector((0, 0, 0.7)) - cam.location) \
        .to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    vt = [v.name for v in bpy.types.ColorManagedViewSettings.bl_rna
          .properties["view_transform"].enum_items]
    if "AgX" in vt:
        scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 1.5
    render("stereographic")


SCENES = {"stereographic": scene_stereographic}



# ---------------- variant galleries: one render per selector option ---
# The grid on each doc page is resolved from the operator's own enum by
# tools/subjects.variants_for(), so the ids and labels here are the
# same ones the Add menu shows.  This replaced a ~350-line hand-copied
# table whose labels could -- and did -- drift from the operators.
#
# Variants render smaller than the hero figures: the page displays them
# at 200 px, so shooting them at the full 720 px spent about five times
# the bytes on detail no reader ever sees.
VARIANT_RES = 320
VARIANT_SAMPLES = 64


def _variant_path(slug, vid):
    return os.path.join(IMG, "variants", "%s__%s.png" % (slug, vid))


def render_variants(only=None, missing_only=True):
    import json
    vdir = os.path.join(IMG, "variants")
    os.makedirs(vdir, exist_ok=True)
    manifest = {}
    mpath = os.path.join(vdir, "_manifest.json")
    if os.path.exists(mpath):
        try:
            manifest = json.load(open(mpath, encoding="utf-8"))
        except Exception:
            manifest = {}

    ops = [op for op in menu_defs.unique_ops()
           if op not in subject_cfg.SKIP]
    if only:
        ops = [op for op in ops if subject_cfg.slug_for(op) in only]

    setup_studio()
    subject_cfg.capture_rig()
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = VARIANT_RES
    scene.cycles.samples = VARIANT_SAMPLES

    done = failed = skipped = 0
    for op in ops:
        slug = subject_cfg.slug_for(op)
        try:
            vs = subject_cfg.variants_for(op)
        except Exception as e:
            print("FAIL resolve", slug, repr(e))
            failed += 1
            continue
        if not vs:
            continue
        # The manifest records the gallery as *declared*, so a page
        # still lists a variant whose render failed this run rather
        # than silently losing it; insert_variants.py drops any whose
        # image is genuinely absent.
        manifest[slug] = [([v[0], v[1], v[3]] if v[3] else [v[0], v[1]])
                          for v in vs]
        json.dump(manifest, open(mpath, "w", encoding="utf-8"), indent=1)
        subject_cfg.aim_rig(op in subject_cfg.PLAN_VIEW)
        for vid, _label, kw, _group in vs:
            path = _variant_path(slug, vid)
            if missing_only and os.path.exists(path):
                skipped += 1
                continue
            clear_sculpts()
            teardown = subject_cfg.enter_environment(op)
            helpers = subject_cfg.run_setup(op)
            try:
                O(op, **kw)()
                subject_cfg.drop_setup(helpers)
                if slug in FORCE_PLASTIC:
                    for o in subjects():
                        o.data.materials.clear()
                if slug in ROTATE:
                    for o in subjects():
                        o.rotation_euler = Euler(ROTATE[slug])
                else:
                    subject_cfg.pose_subjects(op, subjects())
                normalize_subjects()
                apply_material()
                scene.render.filepath = path
                bpy.ops.render.render(write_still=True)
                print("OK", slug, vid)
                done += 1
            except Exception as e:
                print("FAIL", slug, vid, repr(e))
                failed += 1
            finally:
                teardown()
    print("VARIANTS done=%d failed=%d already-present=%d"
          % (done, failed, skipped))

def render_heroes(todo, missing_only):
    std = [s for s in todo if s in TASKS]
    scenes = [s for s in todo if s in SCENES]
    for s in todo:
        if s not in TASKS and s not in SCENES:
            print("SKIP unknown", s)
    done = failed = skipped = 0
    if std:
        setup_studio()
        subject_cfg.capture_rig()
        for slug in std:
            path = os.path.join(IMG, slug + ".png")
            if missing_only and os.path.exists(path):
                skipped += 1
                continue
            clear_sculpts()
            task = TASKS[slug]
            op = getattr(task, 'op', None)
            # A gemstone renders black under the plastic-studio rig --
            # its whole appearance is what it refracts -- so some
            # operators bring their own world.  Set up before building,
            # tear down however the render goes.
            teardown = subject_cfg.enter_environment(op)
            # Operators that transform a selection (the minimal span)
            # have nothing to show until they are given input geometry.
            helpers = subject_cfg.run_setup(op)
            try:
                # Flat subjects are shot from overhead in both renderers
                # -- at 3/4 a tiling collapses to a sliver.
                subject_cfg.aim_rig(op in subject_cfg.PLAN_VIEW)
                task()
                subject_cfg.drop_setup(helpers)
                if slug in FORCE_PLASTIC:
                    for o in subjects():
                        o.data.materials.clear()
                # A slug-specific ROTATE still wins, for a figure that
                # wants a pose the icon has no use for; otherwise the
                # subject's shared pose applies.
                if slug in ROTATE:
                    for o in subjects():
                        o.rotation_euler = Euler(ROTATE[slug])
                else:
                    subject_cfg.pose_subjects(op, subjects())
                normalize_subjects()
                apply_material()
                render(slug)
                print("OK", slug)
                done += 1
            except Exception as e:
                print("FAIL", slug, repr(e))
                failed += 1
            finally:
                teardown()
    for slug in scenes:
        if missing_only and os.path.exists(
                os.path.join(IMG, slug + ".png")):
            skipped += 1
            continue
        try:
            SCENES[slug]()
            print("OK", slug)
            done += 1
        except Exception as e:
            print("FAIL", slug, repr(e))
            failed += 1
    print("HEROES done=%d failed=%d already-present=%d"
          % (done, failed, skipped))
    return failed


def main():
    argv = (sys.argv[sys.argv.index("--") + 1:]
            if "--" in sys.argv else [])
    # Rendering everything is now hours of Cycles across 128 figures and
    # ~1450 variants, so the default is to shoot only what is missing --
    # the same contract tools/bake_menu_icons.py has.  `--all` forces a
    # re-render, which is what to use after changing the studio rig.
    missing_only = "--all" not in argv
    want_variants = "variants" in argv
    only = [a for a in argv
            if a not in ("variants", "--all", "--variants-too")] or None

    if "--list" in argv:
        for slug in sorted(set(TASKS) | set(SCENES)):
            have = os.path.exists(os.path.join(IMG, slug + ".png"))
            print(("  have " if have else "  MISSING ") + slug)
        return

    failed = 0
    if not want_variants or "--variants-too" in argv:
        failed += render_heroes(only or (list(TASKS) + list(SCENES)),
                                missing_only)
    if want_variants or "--variants-too" in argv:
        render_variants(only, missing_only)
    print("DONE")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
