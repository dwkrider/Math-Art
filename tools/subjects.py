"""Canonical render subjects, shared by the docs and the menu icons.

One operator, one subject.  Both renderers read this module, so a
documentation figure and the menu icon beside it show the same object
in the same pose without anyone having to remember to update two files:

    docs/render_docs.py       720 px figures in docs/images/
    tools/bake_menu_icons.py   64 px icons in math_art/icons/

What lives here is what decides *which object you are looking at*:
the operator's parameters, its pose, and whether it is a flat subject
that has to be shot from overhead.  What does NOT live here is
presentation that legitimately differs between a full-page figure and a
16 px menu row -- resolution, sample count, exposure, background,
cropping.  Those stay with each renderer.

The module is importable without Blender (the data half is plain
dicts); the rig helpers at the bottom appear only when bpy is present.
"""

# --------------------------------------------------------------------
# Subject parameters: operator id -> kwargs passed to the operator
# --------------------------------------------------------------------
# A bare default is the right subject for most generators -- an icon
# should show what you get when you click the entry -- so only the ones
# whose defaults under-sell them are listed.
PARAMS = {
    # -- solids ---------------------------------------------------
    # A bare tetrahedron reads as a flat triangle at icon size; the
    # dodecahedron's pentagons say "regular solid" at a glance.  (The
    # docs previously shot a snub cube here, which is Archimedean
    # rather than regular; the dodecahedron suits the operator's name
    # better in both places.)
    "mesh.regular_solid_add": dict(family='PLATONIC', solid='DODECA'),
    # The uniform operator's whole point is what lies beyond the
    # Platonics, so it gets a Kepler-Poinsot star rather than another
    # convex solid that would duplicate the entry above.
    "mesh.uniform_polyhedron_add": dict(family='KEPLER', solid='34'),
    "mesh.polytope4d_add": dict(kind='CELL120'),
    "mesh.waterman_add": dict(root=20),
    "mesh.spiked_polyhedron_add": dict(preset='MODERN'),
    "mesh.woven_polyhedron_add": dict(solid='ICOSA'),
    "mesh.poly_weave_add": dict(kind='CUBE'),
    "mesh.rotegrity_add": dict(kind='ICOSA', freq=1),
    "mesh.tangle_add": dict(kind='T5'),
    # A cube's twist is hidden behind its own faces; a tetrahedron has
    # few enough that the ribbon reads.
    "mesh.platonic_twist_add": dict(kind='TETRA'),
    "mesh.twisted_torus_add": dict(n=6, twist_steps=6),
    "mesh.sphericon_add": dict(sides=7, coloring='NONE'),

    # -- surfaces -------------------------------------------------
    "mesh.scherk_collins_add": dict(preset='HEX'),
    "mesh.parametric_minimal_add": dict(surface='ENNEPER'),
    # The gyroid is the TPMS everyone recognises.
    "mesh.periodic_minimal_add": dict(periodicity='TRIPLY', surface='G'),
    "mesh.minimal_knot_span_add": dict(p=2, q=3),
    "mesh.minimal_surface_polyhedron_add": dict(mode='SADDLE'),
    "mesh.algebraic_surface_add": dict(preset='CLEBSCH'),
    "mesh.curiosity_surface_add": dict(surface='FRESNEL'),
    "mesh.ruled_surface_add": dict(mode='HYPERBOLOID', output='RODS',
                                   family='BOTH'),
    "mesh.spherical_harmonic_add": dict(form='OFFSET', degree=4, order=2),
    "mesh.orbital_add": dict(mode='ATOMIC', n=3, l=2, m=-2),
    "mesh.topological_surface_add": dict(preset='KLEIN'),
    "mesh.seifert_surface_add": dict(preset='TREFOIL'),
    "mesh.bubble_cluster_add": dict(separate=True, color=True),
    # Fold with Blender's own cloth solver rather than the internal
    # packing: same surface, far better folds to look at.
    "mesh.crochet_add": dict(physics='CLOTH'),

    # -- curves and fractals --------------------------------------
    "curve.attractor_add": dict(preset='LORENZ'),
    # Koch is the one fractal everybody has already seen.
    "curve.lsystem_add": dict(kind='PENTAPLEXITY'),
    # EDGE mode offers only the edge-rewriting generators (ANTIKOCH,
    # CESARO, ELEVEN, KOCH, KOCH_SQUARE, LEVY, MINKOWSKI, QUADKOCH,
    # SEVEN); the flowsnake lives under FASS.
    "curve.turtle_curve_add": dict(mode='EDGE', teragon='MINKOWSKI'),

    # -- patterns -------------------------------------------------
    # {7,3} is regular, so every face has the same side count and the
    # default by-sides colouring yields exactly one material.  Parity
    # gives the classic two-tone (with a seam, since q=3 is odd).
    "mesh.hyperbolic_tiling_add": dict(color_by='PARITY'),
    # Light up the 13 parastichy arms rather than shipping a grey disc.
    "mesh.phyllotaxis_add": dict(color_by='PARASTICHY', parastichy=13),

    # -- weaves ---------------------------------------------------
    # Rods flush with the core hide the interleaving; pushing them out
    # two cells at each end shows how the sticks thread past each other.
    "mesh.polystix_add": dict(overhang=2.0),
}


# --------------------------------------------------------------------
# Pose: operator id -> rotation_euler, radians
# --------------------------------------------------------------------
import math                                              # noqa: E402

ORIENT = {
    # A tetrahedron sitting face-on reads as a flat triangle; a sixth
    # of a turn puts an edge toward the camera and it reads as a solid.
    "mesh.regular_solid_add": (0.0, 0.0, 0.62),
    # The Klein bottle's default pose puts the handle behind the body,
    # hiding the self-intersection -- the whole point of the surface.
    "mesh.topological_surface_add": (0.0, 0.0, math.pi),
    # The IFS default is SIERP_TETRA, a Sierpinski *tetrahedron*: a
    # solid, not a plane figure, so it wants a turn rather than a plan
    # view (from overhead a tetrahedron simply squares off).
    "mesh.ifs_add": (0.0, 0.0, math.pi / 8),
}


# --------------------------------------------------------------------
# Flat subjects: shot from straight overhead
# --------------------------------------------------------------------
# The studio rig's 3/4 view collapses a flat panel to a thin sliver --
# measured bounding-box aspect ran 0.21-0.38 against a median near 0.9
# for the solids.  The Patterns entries with genuine relief (relief
# panel and solid, the modular screen, layer groups) are deliberately
# absent: their depth is the subject.
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


# --------------------------------------------------------------------
# Operators no renderer can shoot.  Each needs a reason.
# --------------------------------------------------------------------
SKIP = {
    # Builds the phyllotaxis seed positions as a points-only mesh (120
    # verts, 0 faces), so Cycles has nothing to shade and the frame
    # comes back empty.  Giving it faces just for the thumbnail would
    # show something the operator does not actually produce.
    "mesh.receptacle_add": "points-only mesh, nothing for Cycles to shade",
}


def params_for(op, **extra):
    """Canonical kwargs for `op`, with `extra` taking precedence."""
    kw = dict(PARAMS.get(op, ()))
    kw.update(extra)
    return kw


def _selftest():
    """Structural checks (no Blender required)."""
    for name, table in (("PARAMS", PARAMS), ("ORIENT", ORIENT)):
        for op in table:
            prefix, _, rest = op.partition('.')
            if prefix not in ('mesh', 'curve', 'object') or not rest:
                raise AssertionError(f"{name}: bad operator id {op!r}")
    for op in PLAN_VIEW | set(SKIP):
        prefix, _, rest = op.partition('.')
        if prefix not in ('mesh', 'curve', 'object') or not rest:
            raise AssertionError(f"bad operator id {op!r}")
    for op, rot in ORIENT.items():
        if len(rot) != 3 or not all(isinstance(a, float) for a in rot):
            raise AssertionError(f"ORIENT[{op}] is not three floats")
    # A plan view of a subject that also carries a turntable would fight
    # itself: the rotation is about Z, which is the plan camera's axis.
    both = PLAN_VIEW & set(ORIENT)
    if both:
        raise AssertionError(f"both plan view and ORIENT: {sorted(both)}")
    if set(SKIP) & PLAN_VIEW:
        raise AssertionError("an operator is both skipped and plan view")
    if params_for("mesh.polystix_add", overhang=9.0)["overhang"] != 9.0:
        raise AssertionError("params_for() does not honour overrides")
    print(f"subjects: {len(PARAMS)} parameterised, {len(ORIENT)} posed, "
          f"{len(PLAN_VIEW)} plan view, {len(SKIP)} skipped")


# --------------------------------------------------------------------
# Rig helpers (Blender only) -- shared so both renderers aim the studio
# camera and lights identically for a given subject.
# --------------------------------------------------------------------
try:
    import bpy
    from mathutils import Euler, Vector
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    LIGHT_NAMES = ("Key Light", "Fill Light", "Rim Light L",
                   "Rim Light R", "Top Light")

    # Exposure for the plan-view shots.  Lighting a flat panel head-on
    # and dropping AgX's highlight rolloff drove these to the clipping
    # point -- measured mean value 0.99, with 41% of the hyperbolic
    # tiling's pixels pinned at white, which is what washed the colour
    # out.  -1.8 is where clipping reaches exactly zero across the
    # coloured subjects; darker only dims the icon without recovering
    # saturation the pale generator palettes do not have.
    PLAN_EXPOSURE = -1.8
    STUDIO_EXPOSURE = -0.5

    _CAM_POSE = {}
    _LIGHT_POSE = {}

    def capture_rig():
        """Record the studio poses and derive the plan-view ones.

        The plan camera keeps the studio camera's distance so the two
        framings are comparable, and points straight down (a camera
        with no rotation looks along -Z).  The lights are lifted with
        it: the studio rig lights a solid from the side, which across a
        flat panel is grazing light -- it rakes the surface, blows the
        highlights and leaves the colours pale.  Each light keeps its
        distance but only a quarter of its horizontal offset, so a
        panel is lit nearly head-on.
        """
        _CAM_POSE.clear()
        _LIGHT_POSE.clear()
        cam = bpy.data.objects.get("Studio Camera")
        if cam is not None:
            _CAM_POSE['studio'] = (cam.location.copy(),
                                   cam.rotation_euler.copy())
            _CAM_POSE['plan'] = (Vector((0.0, 0.0, cam.location.length)),
                                 Euler((0.0, 0.0, 0.0)))
        for name in LIGHT_NAMES:
            ob = bpy.data.objects.get(name)
            if ob is None:
                continue
            loc = ob.location.copy()
            dist = max(loc.length, 1e-6)
            plan = Vector((loc.x * 0.25, loc.y * 0.25, abs(dist)))
            plan.length = dist
            _LIGHT_POSE[name] = (loc, plan)

    def aim_rig(plan):
        """Point camera, lights and view transform for the chosen view."""
        cam = bpy.data.objects.get("Studio Camera")
        pose = _CAM_POSE.get('plan' if plan else 'studio')
        if cam is not None and pose is not None:
            cam.location = pose[0].copy()
            cam.rotation_euler = pose[1].copy()
        for name, (studio, overhead) in _LIGHT_POSE.items():
            ob = bpy.data.objects.get(name)
            if ob is None:
                continue
            ob.location = (overhead if plan else studio).copy()
            # Area lights are aimed by rotation, not constrained, so
            # re-aim each one at the origin after moving it.
            ob.rotation_euler = (-ob.location).to_track_quat(
                '-Z', 'Y').to_euler()

        # AgX rolls highlights towards white, which is right for a full
        # page figure and wrong for a flat, coloured subject that has to
        # stay legible by colour.
        scene = bpy.context.scene
        vt = [v.name for v in bpy.types.ColorManagedViewSettings.bl_rna
              .properties["view_transform"].enum_items]
        want = "Standard" if plan else ("AgX" if "AgX" in vt else "Standard")
        if want in vt:
            scene.view_settings.view_transform = want
        scene.view_settings.exposure = (PLAN_EXPOSURE if plan
                                        else STUDIO_EXPOSURE)

    def pose_subjects(op, objects):
        """Apply the canonical pose for `op` to `objects`, if any."""
        rot = ORIENT.get(op)
        if rot is None:
            return False
        for ob in objects:
            ob.rotation_euler = Euler(rot)
        return True
