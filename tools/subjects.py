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
    # The spanned saddle is built on a circle in XY and one in XZ, and
    # the studio camera looks very nearly down the second circle's axis
    # -- straight on, the four lobes overlap into a featureless blob.
    # A quarter turn puts that circle edge-on and the lobes separate.
    "object.minimal_span": (0.0, 0.0, math.pi / 2),
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


# --------------------------------------------------------------------
# Documentation slugs: operator id -> docs/generators/<slug>.md
# --------------------------------------------------------------------
# One slug names three things: the page file, the hero render in
# docs/images/<slug>.png, and that generator's variant renders under
# docs/images/variants/<slug>__<id>.png.
#
# The slug is derived mechanically -- drop the `mesh.` / `curve.` /
# `object.` prefix and the `_add` suffix -- so a new generator needs no
# entry here at all.  Only pages whose historical name differs from
# that rule are listed, which is 14 of them; renaming a page is a
# one-line edit.  The alternative, a full 128-row table, is a second
# copy of menu_defs.py that would rot the moment someone forgot it.
SLUG_OVERRIDE = {
    "mesh.algebraic_surface_add": "algebraic",
    "mesh.minimal_knot_span_add": "knot_span",
    "curve.math_link_add": "link",
    "mesh.minimal_surface_polyhedron_add": "minimal_polyhedron",
    "mesh.regular_solid_add": "regular_solids",
    "mesh.seifert_surface_add": "seifert",
    "curve.space_filling_add": "space_filling_curve",
    "mesh.spacefill_add": "spacefill_solids",
    "mesh.topological_surface_add": "topological",
    "mesh.woven_polyhedron_add": "twisted_polyhedron",
    "mesh.poly_weave_add": "weave",
    "mesh.zonohedron_add": "zonohedra",
    # The operator was renamed `tpms_add` -> `periodic_minimal_add`;
    # the page keeps the acronym everyone searches for, and keeps its
    # URL.  This override is what stops the two drifting apart again.
    "mesh.periodic_minimal_add": "tpms",
    "object.symmetric_sculpture_add": "symmetric_sculpture",
}


def slug_for(op):
    """docs/generators/<slug>.md for `op` (see SLUG_OVERRIDE)."""
    if op in SLUG_OVERRIDE:
        return SLUG_OVERRIDE[op]
    base = op.partition('.')[2]
    return base[:-4] if base.endswith('_add') else base


# --------------------------------------------------------------------
# Variant galleries: which property makes this "a different shape"
# --------------------------------------------------------------------
# A generator's doc page carries a grid of every option of its main
# selector.  Those ids and labels are already declared once, in the
# operator's own EnumProperty, so naming the property is enough -- the
# renderer reads `enum_items` for the rest.  Transcribing them by hand
# (the previous approach, ~350 lines) meant a label could disagree with
# the menu, and every added enum option silently missed the gallery.
# Every property name below was read back off the registered operator
# (tools/check_variants.py re-checks them), not guessed: a stale name
# here silently produces an empty gallery.
VARIANT_SELECTOR = {
    # -- surfaces --
    "mesh.scherk_collins_add": "preset",
    "mesh.seifert_surface_add": "preset",
    "mesh.algebraic_surface_add": "preset",
    "mesh.topological_surface_add": "preset",
    "mesh.curiosity_surface_add": "surface",
    "mesh.helical_surface_add": "surface",
    "mesh.hyperbolic_surface_add": "preset",
    "mesh.squeeze_add": "seed",
    "mesh.vertex_vortices_add": "seed",
    "mesh.minimal_surface_polyhedron_add": "seed",
    "mesh.supershape_add": "preset",
    "mesh.crochet_add": "preset",
    # The four closure modes are what change the form; the outline
    # shapes (kind_a/kind_b) are a second axis the page describes in
    # prose rather than multiplying the gallery by 25.
    "mesh.dform_add": "mode",
    # -- polyhedra --
    "mesh.zonohedron_add": "kind",
    "mesh.polytope4d_add": "kind",
    "mesh.spiked_polyhedron_add": "preset",
    "mesh.hyperbolic_honeycomb_add": "preset",
    "mesh.spacefill_add": "kind",
    "mesh.symmetrohedron_add": "group",
    "mesh.conway_add": "example",
    "mesh.polytwister_add": "shape",
    "mesh.toroidal_polyhedron_add": "solid",
    "mesh.polyhedron_compound_add": "compound",
    "mesh.notable_polyhedron_add": "solid",
    "mesh.biscribed_solid_add": "solid",
    "mesh.icosahedron_stellation_add": "solid",
    "mesh.general_stellation_add": "seed",
    "mesh.star_prism_add": "form",
    "mesh.polyhedral_torus_add": "tiling",
    "mesh.interlocking_add": "family",
    # -- fractals --
    "mesh.sponge_add": "kind",
    "mesh.fractal_polyhedron_add": "kind",
    "curve.space_filling_add": "kind",
    "mesh.mandelbulb_add": "preset",
    "mesh.snowflake_add": "preset",
    "mesh.apollonian_add": "mode",
    "curve.lsystem_add": "kind",
    "curve.turtle_curve_add": "mode",
    "mesh.fractal_tiling_add": "kind",
    "mesh.fractal_reptile_add": "family",
    "mesh.fractal_knotwork_add": "substrate",
    # -- plants --
    "curve.inflorescence_add": "archetype",
    "mesh.leaf_add": "shape",
    "curve.growth_add": "mode",
    "mesh.map_lsystem_add": "mode",
    "mesh.phyllotaxis_add": "form",
    "curve.fractal_tree_add": "mode",
    # -- knots --
    "curve.prime_knot_add": "knot",
    "curve.attractor_add": "preset",
    "curve.math_link_add": "preset",
    "curve.harmonic_knot_add": "preset",
    "curve.petal_knot_add": "preset",
    "curve.rational_knot_add": "preset",
    "curve.fractal_knot_add": "kind",
    "curve.substitution_knot_add": "base",
    "curve.tight_knot_add": "knot",
    "curve.hopf_fibration_add": "preset",
    "mesh.hopf_torus_add": "preset",
    "mesh.rolling_knot_add": "mode",
    "mesh.invariant_manifold_add": "system",
    # -- weaves --
    "mesh.polylinks_add": "preset",
    "mesh.tangle_add": "kind",
    "mesh.poly_weave_add": "kind",
    "mesh.rotegrity_add": "kind",
    "mesh.woven_polyhedron_add": "solid",
    "mesh.woven_double_shell_add": "solid",
    "mesh.turks_head_add": "surface",
    # `preset` carries a CUSTOM entry and duplicates; the packing is
    # the actual family of stick arrangements.
    "mesh.polystix_add": "packing",
    "curve.celtic_knot_add": "source",
    # -- patterns --
    "mesh.frieze_add": "group",
    "mesh.wallpaper_add": "group",
    "mesh.layer_add": "group",
    "mesh.tiling_add": "tiling",
    "mesh.kuniform_add": "tiling",
    "mesh.monohedral_add": "tiling",
    "mesh.isohedral_add": "tiling",
    "mesh.aperiodic_add": "kind",
    "mesh.reptile_add": "kind",
    "mesh.voderberg_add": "kind",
    "mesh.spiral_tiling_add": "family",
    "mesh.islamic_pattern_add": "preset",
    "mesh.celtic_knot_2d_add": "preset",
    "mesh.over_under_screen_add": "weave",
    "mesh.knot_carpet_add": "source",
    "mesh.modular_screen_add": "preset",
    "mesh.relief_panel_add": "preset",
    "mesh.relief_solid_add": "preset",
    "mesh.hyperbolic_tiling_add": "model",
    # -- rollers / odds --
    "mesh.oloid_add": "kind",
    "mesh.platonic_twist_add": "kind",
    "mesh.stereographic_add": "pattern",
    "mesh.constant_width_add": "kind",
    "mesh.monostatic_body_add": "kind",
    "mesh.steinmetz_add": "kind",
    "mesh.koman_add": "kind",
    "mesh.gem_add": "preset",
    "mesh.gem_cabochon_add": "preset",
    "mesh.bubble_cluster_add": "seed",
    "mesh.relaxed_bubble_add": "bubbles",
    "mesh.cmc_capillary_add": "mode",
    "mesh.orbifold_sphere_add": "signature",
    "object.symmetric_sculpture_add": "preset",
}

# Two-level selectors: (group property, item property).  The item enum
# is a callback that depends on the group -- reading it off the type
# yields nothing -- so the renderer sets the group on an operator
# properties instance first, then reads the item list.  The grouping
# is also what gives the page its "### Platonic / ### Archimedean"
# subheadings.
VARIANT_GROUP = {
    "mesh.regular_solid_add": ("family", "solid"),
    "mesh.uniform_polyhedron_add": ("family", "solid"),
    "mesh.canonical_polyhedron_add": ("family", "solid"),
    "mesh.parametric_minimal_add": ("family", "surface"),
    "mesh.periodic_minimal_add": ("periodicity", "surface"),
}

# Groups to render, where a two-level selector reaches further than the
# page usefully can.  The regular-solids operator grew derived families
# (hulls, propellors, chamfers) that are Conway operations on the
# classical ones rather than new solids to enumerate; the page covers
# the six classical families and says so.
VARIANT_GROUP_ONLY = {
    "mesh.regular_solid_add": ("PLATONIC", "ARCHIMEDEAN", "CATALAN",
                               "KEPLER", "PRISM", "JOHNSON"),
}

# Kwargs applied to every variant of a generator, where the gallery
# needs a setting held constant to stay comparable.
VARIANT_COMMON = {
    "mesh.minimal_surface_polyhedron_add": dict(mode='SADDLE'),
    "mesh.bubble_cluster_add": dict(separate=True, color=True),
    "mesh.sphericon_add": dict(coloring='NONE'),
    "mesh.periodic_minimal_add": dict(periodicity='TRIPLY', cells=1),
    "mesh.spherical_harmonic_add": dict(degree=4, order=2),
}

# Enum ids to leave out of a gallery, with a reason.  Keep this short:
# an option worth shipping is usually worth a thumbnail.
VARIANT_SKIP = {
    # 20 vertices cubed exceeds the generator's own copy cap, so the
    # default generation count cannot build it (see VARIANT_EXTRA,
    # which renders it at two generations instead).
    "mesh.fractal_polyhedron_add": {"DODECA"},
}

# Ids skipped in every gallery.  A "custom" entry is the operator
# saying "use the sliders below" -- it has no canonical appearance, so
# its thumbnail would just be whatever the other defaults happen to
# make, sitting in the grid as if it were a named form.
GENERIC_SKIP_IDS = {"CUSTOM", "NONE"}

# Ceiling on one generator's gallery.  The renderer prints what it
# dropped rather than truncating quietly -- a silently capped grid
# reads as "this is the complete set" when it is not.  Raise it per
# operator where the complete set genuinely is the point of the page.
VARIANT_MAX_DEFAULT = 48
VARIANT_MAX = {
    # The 59 stellations of the icosahedron are a named, closed,
    # historically complete list (Coxeter et al.); a partial gallery
    # would misrepresent it.
    "mesh.icosahedron_stellation_add": 64,
    # Likewise the 92 Johnson solids, across all families on one page.
    "mesh.regular_solid_add": 160,
    "mesh.uniform_polyhedron_add": 96,
    # Minimal surfaces are what this project is chiefly about, and the
    # families are the point of the page; do not truncate them.
    "mesh.parametric_minimal_add": 96,
    "mesh.periodic_minimal_add": 96,
    "mesh.canonical_polyhedron_add": 96,
}

# Galleries whose entries are combinations of properties rather than
# one enum, so there is nothing to introspect.  Same 3-tuple shape the
# renderer builds internally: (id, label, kwargs).
VARIANT_EXTRA = {
    "mesh.fractal_polyhedron_add": [
        ("DODECA", "Dodecahedron", dict(kind='DODECA', generations=2)),
    ],
    "mesh.geodesic_add": [
        ("ICOSA", "Icosahedron", dict(base='ICOSA')),
        ("OCTA", "Octahedron", dict(base='OCTA')),
        ("TETRA", "Tetrahedron", dict(base='TETRA')),
        ("GOLDBERG", "Goldberg Dual", dict(base='ICOSA', dual=True)),
    ],
    "curve.torus_knot_add": [
        ("2_3", "Trefoil (2, 3)", dict(p=2, q=3)),
        ("2_5", "Cinquefoil (2, 5)", dict(p=2, q=5)),
        ("2_7", "(2, 7)", dict(p=2, q=7)),
        ("3_4", "(3, 4)", dict(p=3, q=4)),
        ("3_5", "(3, 5)", dict(p=3, q=5)),
        ("5_2", "(5, 2)", dict(p=5, q=2)),
    ],
    "mesh.sphericon_add": [
        (str(n), lab, dict(sides=n))
        for n, lab in ((3, "Triangular (3)"), (4, "Sphericon (4)"),
                       (5, "Pentagonal (5)"), (6, "Hexagonal (6)"),
                       (7, "Heptagonal (7)"), (8, "Octagonal (8)"))
    ],
    "mesh.spherical_harmonic_add": [
        (f, lab, dict(form=f))
        for f, lab in (("OFFSET", "Offset Sphere"),
                       ("ABS", "Absolute Lobes"),
                       ("SIGNED", "Signed Lobes"),
                       ("BOURKE", "Bourke Family"))
    ],
    "mesh.ruled_surface_add": [
        ("HYPERBOLOID", "Stick Hyperboloid", dict(mode='HYPERBOLOID')),
        ("HYPERBOLOID_RODS", "Stick Hyperboloid (Rulings)",
         dict(mode='HYPERBOLOID', output='RODS', family='BOTH')),
        ("HELICAL_CONE", "Compound Helical Cone",
         dict(mode='HELICAL_CONE')),
        ("SPIRAL", "Spiral Ruled", dict(mode='SPIRAL')),
        ("SPIRAL_ROSETTE", "Spiral Ruled (Rosette)",
         dict(mode='SPIRAL', tightness=0.0, petals=5, petal_amp=0.4)),
        ("PLUCKER", "Plucker Cylindroid",
         dict(mode='CONOID', conoid_kind='PLUCKER')),
        ("WALLIS", "Wallis Conical Edge",
         dict(mode='CONOID', conoid_kind='WALLIS')),
        ("WHITNEY", "Whitney Umbrella",
         dict(mode='CONOID', conoid_kind='WHITNEY')),
        ("TANGENT_DEV", "Tangent Developable", dict(mode='TANGENT_DEV')),
        ("HELICOID", "Helicoid", dict(mode='HELICOID')),
        ("TWIST_STRIP", "Twisted Strip (Mobius)",
         dict(mode='TWIST_STRIP', half_twists=1)),
        ("HYPAR", "Hyperbolic Paraboloid", dict(mode='HYPAR')),
    ],
    # The atomic half is indexed by the quantum numbers (n, l, m), not
    # by an enum, so there is nothing to introspect; the molecular half
    # has a 17-entry `preset` but mixing the two lists by hand is what
    # puts them in teaching order on the page.
    "mesh.orbital_add": [
        ("1s", "1s", dict(mode='ATOMIC', n=1, l=0, m=0)),
        ("2s", "2s (radial node)", dict(mode='ATOMIC', n=2, l=0, m=0)),
        ("2pz", "2p_z", dict(mode='ATOMIC', n=2, l=1, m=0)),
        ("3pz", "3p_z", dict(mode='ATOMIC', n=3, l=1, m=0)),
        ("3dxy", "3d_xy", dict(mode='ATOMIC', n=3, l=2, m=-2)),
        ("3dz2", "3d_z2", dict(mode='ATOMIC', n=3, l=2, m=0)),
        ("4fz3", "4f_z3", dict(mode='ATOMIC', n=4, l=3, m=0)),
        ("sigma1s", "sigma 1s",
         dict(mode='MOLECULAR', preset='SIGMA_1S')),
        ("sigmastar1s", "sigma* 1s",
         dict(mode='MOLECULAR', preset='SIGMA_STAR_1S')),
        ("pi2px", "pi 2p_x", dict(mode='MOLECULAR', preset='PI_2PX')),
        ("sp3", "sp3 hybrid", dict(mode='MOLECULAR', preset='SP3')),
        ("water", "H2O lone pair",
         dict(mode='MOLECULAR', preset='WATER_LONE_PAIR')),
        ("benzene", "benzene pi",
         dict(mode='MOLECULAR', preset='BENZENE_PI', huckel_k=0)),
        ("cloud", "pi 2p_x probability cloud",
         dict(mode='MOLECULAR', preset='PI_2PX', display='CLOUD',
              shells=3)),
    ],
    "mesh.ifs_add": [
        ("ABC124", "ABC tile (1,2,4)",
         dict(mode='RADIX', preset='ABC_124')),
        ("ABC128", "ABC tile (1,2,8), self-similar",
         dict(mode='RADIX', preset='ABC_128')),
        ("ABC134", "ABC tile (1,3,4)",
         dict(mode='RADIX', preset='ABC_134')),
        ("TWINA", "Twindragon A", dict(mode='RADIX', preset='TWIN_A')),
        ("TWIND", "Twindragon D", dict(mode='RADIX', preset='TWIN_D')),
        ("TWING", "Twindragon G", dict(mode='RADIX', preset='TWIN_G')),
        ("GASKET", "Cube gasket (4 holes)",
         dict(mode='RADIX', preset='CUBE', holes=4)),
        ("EXACT", "ABC (1,2,4), exact level-k cubes",
         dict(mode='RADIX', preset='ABC_124', tile_output='EXACT')),
        ("SIERPTETRA", "Sierpinski tetrahedron",
         dict(mode='IFS', dimension='3', ifs_preset='SIERP_TETRA',
              output='SOLIDS')),
        ("MENGER", "Menger sponge",
         dict(mode='IFS', dimension='3', ifs_preset='MENGER',
              output='SOLIDS', seed_solid='CUBE', depth=3)),
        ("VOXEL", "Sierpinski octahedron (voxels)",
         dict(mode='IFS', dimension='3', ifs_preset='SIERP_OCTA',
              output='VOXEL')),
        ("ISO", "Sierpinski tetrahedron (smooth)",
         dict(mode='IFS', dimension='3', ifs_preset='SIERP_TETRA',
              output='ISO')),
        ("BMMSIERP", "Sierpinski triangle in 3D (Bandt et al.)",
         dict(mode='IFS', dimension='3', ifs_preset='BMM_SIERP',
              output='ISO')),
        ("BMMSIERPREV", "...and its reverse fractal",
         dict(mode='IFS', dimension='3', ifs_preset='BMM_SIERP',
              output='ISO', reverse=True)),
        ("BMMTETRA", "Modified fractal tetrahedron (Bandt et al.)",
         dict(mode='IFS', dimension='3', ifs_preset='BMM_TETRA',
              output='ISO')),
        ("BMMCUBE", "Modified cube (Bandt et al.)",
         dict(mode='IFS', dimension='3', ifs_preset='BMM_CUBE',
              output='ISO')),
        ("FERN", "Barnsley fern (2-D)",
         dict(mode='IFS', dimension='2', ifs_preset='FERN2D',
              output='RELIEF')),
        ("SIERPTRI", "Sierpinski triangle (2-D)",
         dict(mode='IFS', dimension='2', ifs_preset='SIERP_TRI',
              output='RELIEF')),
        ("DRAGON", "Heighway dragon (2-D)",
         dict(mode='IFS', dimension='2', ifs_preset='DRAGON',
              output='RELIEF')),
        ("LEVY", "Levy C curve (2-D)",
         dict(mode='IFS', dimension='2', ifs_preset='LEVY',
              output='RELIEF')),
        ("KOCH", "Koch curve (2-D)",
         dict(mode='IFS', dimension='2', ifs_preset='KOCH',
              output='RELIEF')),
    ],
}


def load_menu_defs():
    """`math_art.menu_defs` without importing the package.

    `math_art/__init__.py` imports bpy, so a plain
    `from math_art import menu_defs` only works inside Blender.  The
    menu table itself is deliberately bpy-free, so the documentation
    tools and the docs test -- which run under plain Python -- load it
    straight off disk instead.
    """
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "math_art", "menu_defs.py")
    spec = importlib.util.spec_from_file_location("_menu_defs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    # -- documentation tables ------------------------------------
    if slug_for("mesh.oloid_add") != "oloid":
        raise AssertionError("slug_for() default rule is broken")
    if slug_for("mesh.periodic_minimal_add") != "tpms":
        raise AssertionError("slug_for() ignores SLUG_OVERRIDE")
    if slug_for("object.minimal_span") != "minimal_span":
        raise AssertionError("slug_for() mishandles an op with no _add")
    slugs = {}
    for op in list(SLUG_OVERRIDE):
        s = slug_for(op)
        if s in slugs:
            raise AssertionError(f"slug {s!r}: {op} and {slugs[s]}")
        slugs[s] = op
    for name, table in (("VARIANT_SELECTOR", VARIANT_SELECTOR),
                        ("VARIANT_COMMON", VARIANT_COMMON),
                        ("VARIANT_EXTRA", VARIANT_EXTRA),
                        ("VARIANT_GROUP", VARIANT_GROUP),
                        ("VARIANT_MAX", VARIANT_MAX)):
        for op in table:
            prefix, _, rest = op.partition('.')
            if prefix not in ('mesh', 'curve', 'object') or not rest:
                raise AssertionError(f"{name}: bad operator id {op!r}")
    # One operator cannot be both a one-enum gallery and a two-level
    # one; the renderer would have to guess which table wins.
    both = set(VARIANT_SELECTOR) & set(VARIANT_GROUP)
    if both:
        raise AssertionError(f"selector and group both set: {sorted(both)}")
    for op, entries in VARIANT_EXTRA.items():
        ids = [e[0] for e in entries]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"VARIANT_EXTRA[{op}]: duplicate ids")
        for e in entries:
            if len(e) != 3 or not isinstance(e[2], dict):
                raise AssertionError(f"VARIANT_EXTRA[{op}]: bad entry {e}")
    for op in VARIANT_GROUP_ONLY:
        if op not in VARIANT_GROUP:
            raise AssertionError(f"VARIANT_GROUP_ONLY[{op}] has no group")
    n_gal = len(set(VARIANT_SELECTOR) | set(VARIANT_GROUP)
                | set(VARIANT_EXTRA))
    print(f"subjects: {len(PARAMS)} parameterised, {len(ORIENT)} posed, "
          f"{len(PLAN_VIEW)} plan view, {len(SKIP)} skipped, "
          f"{len(SLUG_OVERRIDE)} slug overrides, {n_gal} galleries")


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

    # ----------------------------------------------------------------
    # Input geometry for operators that transform a selection
    # ----------------------------------------------------------------
    # Most generators add a shape from nothing.  A few instead act on
    # whatever is selected, and so have nothing to show until they are
    # given something to act on.  A setup builds that input, leaves it
    # selected, and returns the objects it made so the renderer can drop
    # them once the operator has consumed them.

    def _setup_minimal_span():
        """Two unit circles at right angles, for the span to bridge.

        `object.minimal_span` polls for one or two selected curves or
        meshes and builds the minimal surface across them.  Two coaxial
        circles would give a catenoid; crossing them at a right angle
        gives the four-lobed saddle, which is the more telling picture
        of what the operator does.  The circles are created unfilled, so
        they add no faces of their own to the render.
        """
        made = []
        for rot in ((0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)):
            bpy.ops.mesh.primitive_circle_add(
                vertices=128, radius=1.0, fill_type='NOTHING',
                location=(0.0, 0.0, 0.0), rotation=rot)
            made.append(bpy.context.active_object)
        for ob in bpy.context.selected_objects:
            ob.select_set(False)
        for ob in made:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = made[0]
        return made

    SETUP = {
        "object.minimal_span": _setup_minimal_span,
    }

    # ----------------------------------------------------------------
    # Render environments
    # ----------------------------------------------------------------
    # A setup builds geometry the operator consumes; an environment
    # changes the world the subject is rendered *in*, and has to stay up
    # through the render and then be undone.  Gemstones need one: their
    # appearance is almost entirely what they reflect and refract, so a
    # stone lit by the plastic-studio rig against no environment renders
    # black -- correctly, but uselessly.

    def _env_gem_studio():
        """The add-on's own Gem Studio: sky world, small key, fill.

        `mesh.gem_studio_add` builds the rig the gem generator is meant
        to be seen under.  The documentation studio's lights are hidden
        while it is up, since the gem rig brings its own key and fill
        and doubling them floods out the fire.  Returns a callable that
        puts the scene back.
        """
        scene = bpy.context.scene
        saved_world, saved_cam = scene.world, scene.camera
        hidden = []
        for name in LIGHT_NAMES:
            ob = bpy.data.objects.get(name)
            if ob is not None and not ob.hide_render:
                ob.hide_render = True
                hidden.append(ob)
        before = set(bpy.data.objects)
        bpy.ops.mesh.gem_studio_add()
        made = [o for o in bpy.data.objects if o not in before]

        def teardown():
            for ob in made:
                try:
                    bpy.data.objects.remove(ob, do_unlink=True)
                except Exception:
                    pass
            for ob in hidden:
                ob.hide_render = False
            scene.world = saved_world
            scene.camera = saved_cam

        return teardown

    ENVIRONMENT = {
        "mesh.gem_add": _env_gem_studio,
        "mesh.gem_cabochon_add": _env_gem_studio,
    }

    def enter_environment(op):
        """Set up `op`'s render environment; returns a teardown callable."""
        fn = ENVIRONMENT.get(op)
        return fn() if fn is not None else (lambda: None)

    # ----------------------------------------------------------------
    # Resolving a gallery to a concrete list of renders
    # ----------------------------------------------------------------
    # A static EnumProperty can be read straight off the operator's RNA.
    # A *dynamic* one -- items supplied by a callback -- cannot: RNA
    # only invokes the callback for a live UI, so `enum_items` comes
    # back empty in background Blender (verified; this is why the
    # two-level galleries need the resolvers below rather than the same
    # code path).  Each resolver reads the generator module's own
    # catalogue, which is the list the callback itself is built from.

    import importlib                                       # noqa: E402

    class _Shim:
        """Stand-in for an operator instance, for an items callback.

        The callbacks take (self, context) and read one property off
        `self`.  Handing them an object with just that property is
        enough, and avoids depending on a live operator.
        """

        def __init__(self, **kw):
            self.__dict__.update(kw)

    def _mod(name):
        return importlib.import_module("math_art." + name)

    def _pairs(items):
        return [(it[0], it[1]) for it in items]

    def _groups_regular_solid():
        m = _mod("regular_solids_generator")
        return {fam[0]: _pairs(m._solid_items(_Shim(family=fam[0]), None))
                for fam in m.FAMILIES}

    def _groups_by_family(modname):
        """uniform / canonical: both levels are callbacks."""
        def resolve():
            m = _mod(modname)
            out = {}
            for fid, _label, *_ in m._family_items(_Shim(), None):
                out[fid] = _pairs(m._solid_items(_Shim(family=fid), None))
            return out
        return resolve

    def _groups_parametric():
        # `_surface_items` deliberately returns the *union* list in a
        # background context (scripted calls must not be family-
        # filtered), so grouping has to come from the catalogue dict
        # the callback filters against, not from the callback.
        m = _mod("minimal_surface_toolkit")
        return {fam: _pairs(items)
                for fam, items in m._SURF_ITEMS_FAM.items()}

    def _groups_periodic():
        m = _mod("minimal_surface_toolkit")
        return {per: _pairs(items)
                for per, items in m._PERIODIC_ITEMS.items()}

    GROUP_RESOLVER = {
        "mesh.regular_solid_add": _groups_regular_solid,
        "mesh.uniform_polyhedron_add":
            _groups_by_family("uniform_polyhedra_generator"),
        "mesh.canonical_polyhedron_add":
            _groups_by_family("canonical_polyhedra_generator"),
        "mesh.parametric_minimal_add": _groups_parametric,
        "mesh.periodic_minimal_add": _groups_periodic,
    }

    def _static_enum_items(op, prop):
        """(id, label) pairs for a plain static EnumProperty."""
        mod, _, fn = op.partition('.')
        rna = getattr(getattr(bpy.ops, mod), fn).get_rna_type()
        if prop not in rna.properties:
            raise KeyError(f"{op}: no property {prop!r}")
        p = rna.properties[prop]
        if p.type != 'ENUM':
            raise TypeError(f"{op}.{prop} is {p.type}, not ENUM")
        return [(i.identifier, i.name) for i in p.enum_items]

    def variants_for(op):
        """Resolve `op`'s doc gallery to [(id, label, kwargs, group)].

        `group` is None for a flat gallery and the group's label for a
        two-level one, which is what puts the "### Archimedean"
        subheadings on the page.  Returns [] when the operator has no
        gallery declared.  Raises on a gallery that is declared but
        resolves to nothing -- a stale property name must be loud, not
        silently produce an empty grid.
        """
        common = dict(VARIANT_COMMON.get(op, ()))
        skip = set(VARIANT_SKIP.get(op, ())) | GENERIC_SKIP_IDS
        out = []

        if op in VARIANT_GROUP:
            gprop, iprop = VARIANT_GROUP[op]
            resolver = GROUP_RESOLVER.get(op)
            if resolver is None:
                raise KeyError(f"{op}: VARIANT_GROUP with no resolver")
            groups = resolver()
            labels = dict(_static_enum_items(op, gprop))
            only = VARIANT_GROUP_ONLY.get(op)
            for gid, items in groups.items():
                if only and gid not in only:
                    continue
                for vid, label in items:
                    if vid in skip:
                        continue
                    out.append((vid, label,
                                dict(common, **{gprop: gid, iprop: vid}),
                                labels.get(gid, gid)))
        elif op in VARIANT_SELECTOR:
            prop = VARIANT_SELECTOR[op]
            for vid, label in _static_enum_items(op, prop):
                if vid in skip:
                    continue
                out.append((vid, label, dict(common, **{prop: vid}), None))

        # VARIANT_SKIP is deliberately NOT applied here.  Its usual job
        # is to drop an option the default parameters cannot build, and
        # the matching VARIANT_EXTRA entry is how that option comes
        # back with parameters that work -- filtering it out again
        # would undo the fix.
        for vid, label, kw in VARIANT_EXTRA.get(op, ()):
            out.append((vid, label, dict(common, **kw), None))

        if (op in VARIANT_SELECTOR or op in VARIANT_GROUP) and not out:
            raise ValueError(
                f"{op}: gallery declared but resolved to nothing -- "
                f"stale property name in subjects.VARIANT_*?")

        # A two-level gallery can repeat an id across groups (the same
        # solid id in two families); qualify those so the rendered file
        # names stay unique.
        seen, uniq = {}, []
        for vid, label, kw, grp in out:
            n = seen.get(vid, 0)
            seen[vid] = n + 1
            uniq.append((vid if not n else f"{vid}_{n}", label, kw, grp))

        cap = VARIANT_MAX.get(op, VARIANT_MAX_DEFAULT)
        if len(uniq) > cap:
            print(f"  NOTE {op}: {len(uniq)} variants, capped at {cap} "
                  f"-- dropped {[v[0] for v in uniq[cap:]]}")
            uniq = uniq[:cap]
        return uniq

    def run_setup(op):
        """Build `op`'s input geometry.  Returns the objects it made."""
        fn = SETUP.get(op)
        return list(fn()) if fn is not None else []

    def drop_setup(objects):
        """Remove setup geometry once the operator has consumed it."""
        for ob in objects:
            try:
                bpy.data.objects.remove(ob, do_unlink=True)
            except Exception:
                pass
