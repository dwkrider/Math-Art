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
    # A bare noble faceting is a self-intersecting wireframe-ish solid
    # and reads as mush when shaded; the great dodecahedron -- faceting
    # 1 of the icosahedral vertex set -- has big obvious pentagons and
    # says "this is a polyhedron through someone else's vertices".
    "mesh.noble_faceting_add": dict(seed='ICOSA', index=1),
    # The compound of five tetrahedra is the operator's own default and
    # the clearest advertisement for it: five interpenetrating solids
    # whose separateness is obvious at icon size, where the stella
    # octangula just reads as one spiky ball.
    "mesh.polyhedron_compound_add": dict(family='CLASSICAL',
                                         compound='5TETRA'),
    # -- spidrons -------------------------------------------------
    # The whole hexagonal subdivision coloured BY ARM: six spiral
    # limbs meeting at the centre, which is the picture the word
    # "spidron" means.  Six rings keep the outer triangles big enough
    # to read at icon size (the deeper rings are invisible anyway),
    # and no grout -- the margin would cut each arm into slivers where
    # the solid arm is the subject.
    "mesh.spidron_rosette_add": dict(layout_kind='FIGURE',
                                     arm_parts=6, rings=6,
                                     arm='SPIDRON', color_by='ARM'),
    # Flat, the nest is just a hexagon of triangles -- the whole point
    # is that it folds, so shoot it mid-fold where the crumple reads.
    # (`import math` lives below these tables, so angles are literal
    # radians here: 0.698132 rad = 40 deg, 0.383972 rad = 22 deg.)
    "mesh.spidron_nest_add": dict(fold=0.698132, rings=7,
                                  cap_center=True),
    # The dodecahedron is Nylander's subject and the one the name
    # "spidroball" refers to.  The operator defaults ARE his published
    # ball (Advance twist, ring scale 2/3, 8 rings, relief 0.10557,
    # uniform chirality).  The twist is an EXCESS on top of the node
    # step: Nylander's own is +0.532 deg, and -0.261799 rad = -15 deg
    # opens the arms out from that, which reads better at icon size
    # and is the operator's default, so icon and defaults agree.
    "mesh.spidron_ball_add": dict(seed='DODECA', rings=8,
                                  scale_step=0.666667, twist=-0.261799,
                                  relief=0.105573, chirality='CW',
                                  twist_style='ADVANCE',
                                  relief_style='WOVEN',
                                  color_by='PAIR', colors=5),
    # The two-cell repeat unit IS the subject: on a single decatrihedron
    # the space-filling's CW-meets-CCW rule is invisible, and the pair
    # of opposite-winding polyhedra (coloured by form) is the paper's
    # own Figure 3.  A small gap keeps the two solids readable as two.
    # Pearce's diamond tetrahedron is the famous one: four REGULAR
    # skew hexagons, the interstitial domain of the diamond net, and
    # the solid that reads unmistakably as a saddle polyhedron at icon
    # size.  Relaxed to the minimal surface, which is Pearce's own
    # definition of the face.
    # No `family` here: the group selector is gone and the solid is
    # picked from a single gallery, so passing one is a hard error.
    "mesh.saddle_polyhedron_add": dict(solid='DIAMOND_TETRAHEDRON',
                                       face_style='MINIMAL',
                                       density=4, smoothness=40),
    # -- solids ---------------------------------------------------
    # Escher's Solid (the stellated rhombic dodecahedron of "Waterfall")
    # is the most recognisable of the notable polyhedra.
    "mesh.notable_polyhedron_add": dict(solid='ESCHER'),
    # A bare tetrahedron reads as a flat triangle at icon size; the
    # dodecahedron's pentagons say "regular solid" at a glance.  (The
    # docs previously shot a snub cube here, which is Archimedean
    # rather than regular; the dodecahedron suits the operator's name
    # better in both places.)
    "mesh.regular_solid_add": dict(family='PLATONIC', solid='DODECA'),
    # The bare default is a rhombic triacontahedron, which the zonohedra
    # entry above already shows.  The dissection is what this operator
    # adds that nothing else has: Kowalewski's twenty golden rhombohedra,
    # colour-matched by zone triple.  Assembled rather than exploded --
    # the blocks close up into the solid exactly, so the icon reads as a
    # polyhedron whose faces are colour-coded by the block behind them,
    # which is the point; exploded it reads as debris at 64 px.
    "mesh.zonish_add": dict(mode='DISSECTION', seed='ICOSA',
                            explode=0.0, color='BLOCK'),
    # Halfway is the whole point of the operator -- either end
    # is just a solid we already ship.  The cuboctahedron reads
    # clearly at icon size and its dual is the rhombic
    # dodecahedron, so both families are recognisable.
    "mesh.transpolyhedron_add": dict(seed='CO', blend=0.5),
    # The pyritohedron at its default 1/phi IS the regular dodecahedron,
    # which the regular-solid entry already shows.  Pulled off that
    # value it becomes the pyrite crystal form, which is the point.
    "mesh.twelve_faced_add": dict(solid='PYRITOHEDRON', shape=0.35),
    # Thirty squares is the slide-together everyone recognises, and the
    # colour rotation is how the paper models are actually made.
    # The twenty triangles, not the thirty squares.  Once the squares
    # were placed correctly on the rhombidodecadodecahedron they pack
    # almost closed, and at 128 px the icon became a pale ball with no
    # hint of what the generator does.  The triangle model keeps Hart's
    # "prominent stars" open, so the panels and the interlocking stay
    # legible.  Thickness is nudged up from the operator default of 0.01,
    # which suits a printable template but vanishes at icon size.
    "mesh.slide_together_add": dict(model='T20', colors=True,
                                    thickness=0.03),
    # The uniform operator's whole point is what lies beyond the
    # Platonics, so it gets a Kepler-Poinsot star rather than another
    # convex solid that would duplicate the entry above.
    "mesh.uniform_polyhedron_add": dict(family='KEPLER', solid='34'),
    "mesh.polytope4d_add": dict(kind='CELL120'),
    # Csaszar and Szilassi are the mathematically famous toroids but
    # both read as a crumpled scrap at thumbnail size; the Borromean
    # ring polyhedron says "toroidal" instantly -- three rectangular
    # rings interlocked, with the holes plainly visible.
    "mesh.toroidal_polyhedron_add": dict(solid='BORROMEAN'),
    # The K = +1 family defaults to the sphere, which at icon size is
    # a plain ball and says nothing the UV-sphere primitive does not.
    # The spindle -- a lemon with a conical tip at each pole -- is the
    # one that reads as "constant curvature, not a sphere".
    "mesh.spherical_surface_add": dict(preset='SPINDLE'),
    "mesh.waterman_add": dict(root=20),
    "mesh.spiked_polyhedron_add": dict(preset='MODERN'),
    "mesh.woven_polyhedron_add": dict(solid='ICOSA'),
    "mesh.poly_weave_add": dict(kind='CUBE'),
    "mesh.rotegrity_add": dict(kind='ICOSA', freq=1),
    "mesh.tangle_add": dict(kind='T5'),
    # The dodecahedron's twelve pentagons give the twist far more to
    # act on than a tetrahedron's four triangles, so the spiralling
    # reads as a twist rather than as a bent triangle.
    "mesh.platonic_twist_add": dict(kind='DODECA'),
    "mesh.twisted_torus_add": dict(n=6, twist_steps=6),
    # The Clifford torus is a torus; the vesicle is the shape the
    # Willmore energy is famous for -- the biconcave discocyte that
    # the Helfrich model predicts and a red blood cell actually is.
    # Of the three modes the Clifford torus and the inflated ring both
    # render as a plain doughnut; the relaxed vesicle is the one with a
    # shape of its own.  It settles to the prolate CAPSULE branch, not
    # the biconcave discocyte the module header describes: measured
    # dimensions are (2.0, 0.55, 0.55) for seed_shape='OBLATE' and
    # 'PROLATE' alike, at reduced volume 0.65 and 0.60.  No pose is set
    # for it -- the capsule is a solid of revolution, so rotating about
    # its own axis does nothing and any other angle just foreshortens
    # it.
    "mesh.willmore_add": dict(mode='VESICLE'),
    "mesh.sphericon_add": dict(sides=7, coloring='NONE'),

    # -- surfaces -------------------------------------------------
    "mesh.scherk_collins_add": dict(preset='HEX'),
    "mesh.parametric_minimal_add": dict(surface='ENNEPER'),
    # The gyroid is the TPMS everyone recognises.  Two cells per axis
    # rather than one: a single cell is a fragment, while the 2x2x2
    # block shows the channels threading through one another, which is
    # the thing about a gyroid.
    "mesh.periodic_minimal_add": dict(periodicity='TRIPLY', surface='G',
                                      cells=2),
    "mesh.minimal_knot_span_add": dict(p=2, q=3),
    "mesh.minimal_surface_polyhedron_add": dict(mode='SADDLE', seed='ICOSA'),
    "mesh.algebraic_surface_add": dict(preset='CLEBSCH'),
    "mesh.curiosity_surface_add": dict(surface='FRESNEL'),
    "mesh.ruled_surface_add": dict(mode='HYPERBOLOID', output='RODS',
                                   family='BOTH'),
    "mesh.spherical_harmonic_add": dict(form='OFFSET', degree=4, order=2),
    # 3d_z2: the two-lobes-and-a-torus orbital that is the one everybody
    # pictures from a chemistry textbook, and unmistakable in silhouette.
    "mesh.orbital_add": dict(mode='ATOMIC', n=3, l=2, m=0),
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

    # A single soap bubble is a sphere; the triple is where the
    # generator's point lives -- three films meeting at 120 degrees
    # along a Plateau border.
    "mesh.relaxed_bubble_add": dict(bubbles='TRIPLE'),
    # Uncoloured, a gasket is a heap of white spheres and the nesting
    # is invisible; colouring by radius separates the generations.
    "mesh.apollonian_add": dict(color_by='SIZE'),

    # -- patterns -------------------------------------------------
    # The bare default is a plain relief; a reaction-diffusion field
    # wrapped on a torus shows what the generator is actually for --
    # the pattern reads at a glance and the hole says "solid", not
    # "panel", which is the distinction against the relief panel.
    "mesh.relief_solid_add": dict(preset='TURING_SPHERE', regime='MAZE'),
    # {7,3} is regular, so every face has the same side count and the
    # default by-sides colouring yields exactly one material.  Parity
    # gives the classic two-tone (with a seam, since q=3 is odd).
    "mesh.hyperbolic_tiling_add": dict(color_by='PARITY'),
    # The bubble is the whole point, and it is localised: too many
    # Delaunay periods shrink it to a speck on a long pipe.
    # The default Willmore (1,3) member at render-grade sampling: three
    # lobes wrap into a cleanly readable trefoil-like sleeve, and the
    # Willmore shape is the mathematically canonical representative.
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

# The gyroid's own pose, shared by its hero and its gallery entries.
GYROID_POSE = (-0.1967, 0.1944, -1.8216)

ORIENT = {
    # The diamond tetrahedron's 3-fold axis is vertical by
    # construction, and straight down it the four hexagons stack into a
    # flat hexagonal silhouette -- the one view that hides the saddle
    # curvature the generator exists to show.  Tip it off the axis so
    # two faces turn toward the camera.
    "mesh.saddle_polyhedron_add": (1.309, 0.175, 0.349),
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
    # The folded nest is a relief: straight down hides the fold and
    # straight on hides the spiral, so tip it into a three-quarter view.
    "mesh.spidron_nest_add": (0.9, 0.0, 0.4),
    # The spanned saddle is built on a circle in XY and one in XZ, and
    # the studio camera looks very nearly down the second circle's axis
    # -- straight on, the four lobes overlap into a featureless blob.
    # A quarter turn puts that circle edge-on and the lobes separate.
    "object.minimal_span": (0.0, 0.0, math.pi / 2),
    # Posed in the Blender viewport and converted to the studio
    # camera's frame (R = q_studio . q_view^-1).  Looks down the
    # channels so the openings read as holes rather than as dents.
    # The hero is the gyroid, so the operator-level pose is the
    # gyroid's; the gallery poses per surface (see VARIANT_ORIENT).
    "mesh.periodic_minimal_add": GYROID_POSE,
    # A Meissner solid seen down a vertex axis is a Reuleaux triangle:
    # a flat-looking rounded triangle that says nothing about its being
    # a solid.  Turned off that axis, the three curved edges and the
    # tetrahedral structure both read.
    "mesh.constant_width_add": (math.radians(70), math.radians(15),
                                math.radians(45)),
}


# --------------------------------------------------------------------
# Per-variant poses
# --------------------------------------------------------------------
# ORIENT is keyed by operator, which is right for a hero -- one
# operator, one subject -- but too blunt for a gallery, where the
# entries are genuinely different surfaces.  The gyroid pose looks down
# the gyroid's channels; applied to Schwarz P or Neovius it is just an
# arbitrary tilt of a different shape.
#
# An operator listed here takes its gallery poses from THIS table and
# does not fall back to ORIENT, so a variant left out gets no pose at
# all.  That is the point: only the surfaces named are posed.
VARIANT_ORIENT = {
    "mesh.periodic_minimal_add": {
        "G": GYROID_POSE,           # Gyroid
        "CG": GYROID_POSE,          # Complementary Gyroid
        "GPRIME": GYROID_POSE,      # G' Alternating Gyroid
        # PGD is the P-Gyroid-D Bonnet family and defaults to the P
        # end of it, so it is deliberately not posed as a gyroid.
    },
}


# --------------------------------------------------------------------
# Flat subjects: shot from straight overhead
# --------------------------------------------------------------------
# The studio rig's 3/4 view collapses a flat panel to a thin sliver --
# measured bounding-box aspect ran 0.21-0.38 against a median near 0.9
# for the solids.  The Patterns entries with genuine relief (relief
# panel and solid, the modular screen, the over-under screen, layer
# groups) are deliberately absent: their depth is the subject.  The
# over-under screen especially -- it is a *woven* screen whose whole
# point is ribbons passing in front of and behind one another, and
# from straight overhead that reads as a flat pattern.
PLAN_VIEW = {
    "mesh.frieze_add", "mesh.wallpaper_add", "mesh.tiling_add",
    "mesh.kuniform_add", "mesh.monohedral_add", "mesh.isohedral_add",
    "mesh.aperiodic_add", "mesh.reptile_add", "mesh.voderberg_add",
    "mesh.spiral_tiling_add", "mesh.fractal_tiling_add",
    "mesh.spidron_rosette_add",
    "mesh.fractal_reptile_add", "mesh.islamic_pattern_add",
    "mesh.celtic_knot_2d_add",
    "mesh.knot_carpet_add", "mesh.hyperbolic_tiling_add",
    "mesh.map_lsystem_add",
    # curve-based fractals that are drawn in the plane
    "curve.lsystem_add", "curve.turtle_curve_add",
    "curve.substitution_knot_add", "mesh.fractal_knotwork_add",
    "mesh.snowflake_add",
    # a phyllotaxis head is a flat disc: at 3/4 it foreshortens to a
    # pale ellipse and the parastichy colouring is wasted
    "mesh.phyllotaxis_add",
    # a leaf is a blade with venation -- the outline and the veins are
    # the subject, and both are only legible face-on
    "mesh.leaf_add",
}


# --------------------------------------------------------------------
# Helper objects an operator leaves beside its result
# --------------------------------------------------------------------
# Some operators build working geometry alongside the thing they make:
# a source motif to edit, guide rings to align by.  Those belong in the
# viewport, not in the figure -- and they do real damage there, because
# the framing fits the *combined* bounding box.  The symmetric
# sculpture's motif sits well off to one side, so including it shrank
# the sculpture to a speck in the middle of the frame.
#
# They are HIDDEN, not deleted: the sculpture instances its motif
# through a Geometry Nodes modifier, so removing the motif removes the
# geometry too and the frame comes back empty.  Hiding is enough, since
# the renderer skips hidden objects when it measures the subject.
HIDE_AFTER = {
    "object.symmetric_sculpture_add": ("SymSculpt Motif",
                                       "SymSculpt Guides"),
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
    "mesh.topological_surface_add": "preset",
    "mesh.curiosity_surface_add": "surface",
    "mesh.helical_surface_add": "surface",
    "mesh.hyperbolic_surface_add": "preset",
    "mesh.delaunay_surface_add": "mode",
    "mesh.bryant_surface_add": "mode",
    "mesh.squeeze_add": "seed",
    "mesh.vertex_vortices_add": "seed",
    "mesh.minimal_surface_polyhedron_add": "seed",
    "mesh.supershape_add": "preset",
    "mesh.crochet_add": "preset",
    "mesh.willmore_add": "mode",
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
    "mesh.saddle_polyhedron_add": "solid",
    "mesh.notable_polyhedron_add": "solid",
    "mesh.biscribed_solid_add": "solid",
    # Stellation's variants are its 59 Crennell figures, not its seven
    # seeds: `solid` is the selector that changes the shape most, and the
    # 59 are a closed historical list (see VARIANT_MAX below).  The seeds
    # other than the icosahedron are reached through `preset` and are not
    # enumerated here.  (The former mesh.general_stellation_add entry went
    # when that operator was merged into this one.)
    "mesh.icosahedron_stellation_add": "solid",
    "mesh.noble_faceting_add": "seed",
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
    "curve.tight_link_add": "link",
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
    "mesh.celtic_knot_2d_add": "substrate",
    "mesh.transpolyhedron_add": "seed",
    "mesh.slide_together_add": "model",
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
    "mesh.algebraic_surface_add": ("family", "preset"),
    "mesh.polyhedron_compound_add": ("family", "compound"),
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
    # span the substrates with the 3D woven interlace (the operator's
    # interlace_mode defaults to FLAT, but the woven form is what reads).
    "mesh.celtic_knot_2d_add": dict(interlace_mode='WOVEN'),
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
    # The polygonal spiral needs its own n/arms/angle triple: at the
    # shared defaults the operator reports the angles degenerate
    # (A=50, B=310, C=-180).  It wants a hand-tuned VARIANT_EXTRA
    # entry from someone who knows the family, not a broken thumbnail.
    "mesh.spiral_tiling_add": {"POLY"},
}

# Ids skipped in every gallery.  A "custom" entry is the operator
# saying "use the sliders below" -- it has no canonical appearance, so
# its thumbnail would just be whatever the other defaults happen to
# make, sitting in the grid as if it were a named form.  "ACTIVE"
# means "use the selected object", which in a headless render is
# nothing at all: the operator correctly refuses ("no active mesh
# object; pick a built-in seed instead").
GENERIC_SKIP_IDS = {"CUSTOM", "NONE", "ACTIVE"}

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
    # 10 classical + 63 Hauser; the Hauser family is a named gallery
    # and a partial one would misrepresent it.
    "mesh.algebraic_surface_add": 96,
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


def module_for(op):
    """Path of the generator module that declares `op`, by bl_idname.

    Found by scanning rather than registered anywhere, so it works
    outside Blender -- which is what lets the docs test tell whether a
    figure predates the code that draws it.
    """
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "math_art")
    needle = 'bl_idname = "%s"' % op
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(root, name)
        try:
            with open(path, encoding="utf-8") as fh:
                if needle in fh.read():
                    return path
        except OSError:
            continue
    return None


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
    for op, table in VARIANT_ORIENT.items():
        prefix, _, rest = op.partition('.')
        if prefix not in ('mesh', 'curve', 'object') or not rest:
            raise AssertionError(f"VARIANT_ORIENT: bad operator id {op!r}")
        for vid, rot in table.items():
            if len(rot) != 3 or not all(isinstance(a, float) for a in rot):
                raise AssertionError(
                    f"VARIANT_ORIENT[{op}][{vid}] is not three floats")
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
    # Exposure and light level for the plan view.  The rig lifts five
    # lights -- two of them 750 W rim lights -- to nearly overhead, and
    # a flat panel facing them comes back washed: measured mean
    # saturation 0.10 against a palette whose red, (0.85, 0.30, 0.24),
    # is 0.43 saturated in sRGB.  Sweeping both knobs:
    #
    #     exposure  lights   saturation  median value
    #       -1.8     x1.00      0.103       0.925
    #       -3.0     x0.25      0.312       0.667
    #       -3.5     x0.25      ~0.37       ~0.60
    #       -4.0     x0.25      0.423       0.529
    #       -5.0     x0.25      0.476       0.376   (too dark)
    #
    # -3.5 with quarter lights is the balance: saturation roughly
    # tripled while the panel still reads as lit rather than murky.
    PLAN_EXPOSURE = -3.5
    PLAN_LIGHT_SCALE = 0.25

    # The 3/4 studio look.  Khronos PBR Neutral is a tone curve built to
    # roll highlights off WITHOUT the desaturation and hue shift a
    # filmic curve introduces, which is exactly what was wanted here:
    # under AgX the subjects clipped, and the saddle palette -- colours
    # like (0.85, 0.30, 0.24) -- measured 0.21 mean saturation against
    # 0.43 under this curve.
    #
    # The exposure is set by the CONVEX subjects, and judging it on a
    # spiky one is the trap.  A star self-shadows, so it shows a wide
    # luminance range under any rig; a convex white solid inside a
    # five-light surround does not.  At -0.5 the geodesic sphere renders
    # at mean luminance 0.92 across a 0.58-0.99 range: its shading is
    # present but crushed against white, with nothing clipped -- it is
    # simply all bright.  At -2.0 the same sphere sits at mean 0.75
    # across 0.31-0.96 and the facets appear.  Judge it on a ball.
    #
    # Menu icons go two thirds of a stop darker still (see
    # tools/bake_menu_icons.py), because they are cropped to 64 px where
    # a shallow gradient has far fewer pixels to read across.
    STUDIO_EXPOSURE = -2.0
    STUDIO_VIEW_TRANSFORM = "Khronos PBR Neutral"
    STUDIO_RIM_SCALE = 0.35

    def _set_view_transform(scene, *names):
        """Set the first view transform this Blender actually offers.

        `view_transform` is a DYNAMIC enum: its items come from the
        loaded OCIO config at runtime, so `bl_rna` reports NONE of them
        and testing membership that way rejects every name, including
        the ones in use.  (That silently disabled the transform choice
        here for as long as it has been written that way.)  Assigning
        and reading back is the only reliable probe.
        """
        vs = scene.view_settings
        for name in names:
            try:
                vs.view_transform = name
            except TypeError:
                continue
            if vs.view_transform == name:
                return name
        return vs.view_transform

    # Focal lengths, solved rather than chosen.  Subjects are
    # normalised into a 2 m cube, so the guarantee the frame has to
    # keep is that the cube fits however it is turned.  Projecting its
    # eight corners through this rig and requiring the worst to land at
    # 0.97 of the half-frame (a 3% border) gives these two values -- one
    # per view, because the overhead camera sits the same distance away
    # but the cube subtends less from straight above.
    #
    # Deriving them from measured silhouettes instead put the real cube
    # corner at 1.014 and clipped six figures; the cube is what the
    # normalisation promises, so the cube is what these are solved
    # against.  Re-solve with tools/solve_lens.py if the rig's distance
    # or view direction ever changes.
    STUDIO_LENS = 84.0
    PLAN_LENS = 117.5

    _CAM_POSE = {}
    _LIGHT_POSE = {}
    _LIGHT_ENERGY = {}

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
        _LIGHT_ENERGY.clear()
        for name in LIGHT_NAMES:
            ob = bpy.data.objects.get(name)
            if ob is None:
                continue
            loc = ob.location.copy()
            dist = max(loc.length, 1e-6)
            plan = Vector((loc.x * 0.25, loc.y * 0.25, abs(dist)))
            plan.length = dist
            _LIGHT_POSE[name] = (loc, plan)
            _LIGHT_ENERGY[name] = ob.data.energy

    def aim_rig(plan):
        """Point camera, lights and view transform for the chosen view."""
        cam = bpy.data.objects.get("Studio Camera")
        pose = _CAM_POSE.get('plan' if plan else 'studio')
        if cam is not None and pose is not None:
            cam.location = pose[0].copy()
            cam.rotation_euler = pose[1].copy()
            cam.data.lens = PLAN_LENS if plan else STUDIO_LENS
        for name, (studio, overhead) in _LIGHT_POSE.items():
            ob = bpy.data.objects.get(name)
            if ob is None:
                continue
            ob.location = (overhead if plan else studio).copy()
            # Area lights are aimed by rotation, not constrained, so
            # re-aim each one at the origin after moving it.
            ob.rotation_euler = (-ob.location).to_track_quat(
                '-Z', 'Y').to_euler()
            # Kill the specular lobe for the plan view.  A flat panel
            # lit nearly head-on picks up a broad white sheen across
            # its whole surface, and white added equally to every
            # channel is precisely what destroys saturation: the
            # pattern palette is strongly coloured -- (0.85, 0.30,
            # 0.24) is a 0.72-saturation red -- yet measured 0.08 in
            # the render.  Note this cannot be fixed with exposure:
            # saturation is a ratio between channels, so scaling them
            # all together leaves it exactly where it was.  Solids at
            # 3/4 keep their highlights, which is what reads as form.
            ob.visible_glossy = not plan
            base = _LIGHT_ENERGY.get(name)
            if base is not None:
                ob.data.energy = base * (PLAN_LIGHT_SCALE if plan else 1.0)

        # Colour management, and for the 3/4 view the light ratio with
        # it.  Both are set HERE because aim_rig is the one call every
        # render path makes -- hero figures, gallery variants and menu
        # icons alike -- so this is the only place they cannot drift
        # apart.  Anything a caller sets beforehand is overwritten by
        # this function; a caller wanting to differ has to act after it.
        scene = bpy.context.scene
        want = "Standard" if plan else STUDIO_VIEW_TRANSFORM
        _set_view_transform(scene, want, "AgX", "Standard")
        scene.view_settings.exposure = (PLAN_EXPOSURE if plan
                                        else STUDIO_EXPOSURE)
        if not plan:
            # Rim light is meant to draw an EDGE.  The rig builds two of
            # them at 750 W against a 320 W key, so at full strength
            # their wrap light reaches round into the shadow side and
            # fills it -- and a white subject, which is most of them,
            # then has no gradient left to read its form by.  Measured
            # on a geodesic sphere, the facets simply disappear.  This
            # brings them back under the key, where three-point practice
            # puts them.  Plan views keep their own treatment above.
            for name, _pose in _LIGHT_POSE.items():
                if not name.startswith("Rim Light"):
                    continue
                ob = bpy.data.objects.get(name)
                if ob is not None:
                    ob.data.energy *= STUDIO_RIM_SCALE

    def pose_subjects(op, objects):
        """Apply the canonical pose for `op` to `objects`, if any."""
        rot = ORIENT.get(op)
        if rot is None:
            return False
        for ob in objects:
            ob.rotation_euler = Euler(rot)
        return True

    def pose_variant(op, vid, objects):
        """Pose one gallery entry.

        An operator with a VARIANT_ORIENT table is posed per variant
        and does NOT inherit its operator-level ORIENT: the gallery
        entries are different surfaces, and one surface's pose is
        meaningless on another.
        """
        table = VARIANT_ORIENT.get(op)
        rot = table.get(vid) if table is not None else ORIENT.get(op)
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
        # The backdrop dome is a radius-30 sphere enclosing the scene,
        # so it also seals the stone off from the sky world the gem rig
        # installs -- and that sky is not decoration, it is the thing
        # the stone reflects and refracts.  Sealing it in is most of why
        # both gem figures rendered black.
        #
        # Hiding the dome outright would fix the stone and wreck the
        # picture: the sky would then be the background, and the gems
        # would be the only two figures in the gallery not shot against
        # black velvet.  So drop the dome out of the *reflection* rays
        # only and leave it visible to the camera -- the stone sees the
        # sky, the reader sees the same backdrop as everywhere else.
        dome = bpy.data.objects.get("Backdrop Dome")
        dome_rays = dome_mats = None
        if dome is not None:
            dome_rays = (dome.visible_diffuse, dome.visible_glossy,
                         dome.visible_transmission)
            dome.visible_diffuse = False
            dome.visible_glossy = False
            dome.visible_transmission = False
            # ...and the velvet has to stop being *lit*, too.  The gem
            # sky is bright, and a 0.006-albedo surface under it comes
            # back mid-grey (measured 0.246), not black.  A zero
            # emission shader ignores lighting entirely, so the camera
            # sees the same black behind a gem as behind everything
            # else, whatever the rig does to the world.
            dome_mats = list(dome.data.materials)
            black = bpy.data.materials.new("Gem Backdrop Black")
            black.use_nodes = True
            nt = black.node_tree
            nt.nodes.clear()
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 0.0
            nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
            dome.data.materials.clear()
            dome.data.materials.append(black)
        before = set(bpy.data.objects)
        # add_camera=False is essential, not tidiness: the operator
        # otherwise points scene.camera at its own Gem Camera, which is
        # framed for a stone at the size the gem generator makes one.
        # The docs rig then scales the stone to the 2 m cube and films
        # it with a camera aimed somewhere else entirely -- which is
        # how both gem figures came out pure black (measured mean
        # 0.0003) while the render still "succeeded".  We want the gem
        # studio's world and lights; we keep our own camera.
        bpy.ops.mesh.gem_studio_add(add_camera=False)
        made = [o for o in bpy.data.objects if o not in before]
        # Whatever the rig arrives hidden for, it is no use to us that
        # way: these are the only lights left once the studio's are off.
        for ob in made:
            if ob.type == 'LIGHT':
                ob.hide_render = False

        def teardown():
            for ob in made:
                try:
                    bpy.data.objects.remove(ob, do_unlink=True)
                except Exception:
                    pass
            for ob in hidden:
                ob.hide_render = False
            if dome is not None and dome_rays is not None:
                (dome.visible_diffuse, dome.visible_glossy,
                 dome.visible_transmission) = dome_rays
                dome.data.materials.clear()
                for m in dome_mats:
                    dome.data.materials.append(m)
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

    def _groups_algebraic():
        # Same shape as _groups_parametric: `_preset_items` returns
        # the union list in a background context on purpose, so the
        # grouping comes from the per-family catalogue the callback
        # filters against.
        m = _mod("algebraic_surface_generator")
        return {fam: _pairs(items)
                for fam, items in m._PRESET_ITEMS_FAM.items()}

    def _groups_compound():
        # Compound is a two-stage family -> compound selector, exactly
        # like uniform/canonical above; its catalogue is the authority
        # (both enums are dynamic callbacks).  compound_families() is
        # [(family key, heading, [(compound key, label), ...]), ...].
        m = _mod("compound_generator")
        return {key: _pairs(rows)
                for key, _head, rows in m._cmp.compound_families()}

    GROUP_RESOLVER = {
        "mesh.regular_solid_add": _groups_regular_solid,
        "mesh.uniform_polyhedron_add":
            _groups_by_family("uniform_polyhedra_generator"),
        "mesh.canonical_polyhedron_add":
            _groups_by_family("canonical_polyhedra_generator"),
        "mesh.parametric_minimal_add": _groups_parametric,
        "mesh.periodic_minimal_add": _groups_periodic,
        "mesh.algebraic_surface_add": _groups_algebraic,
        "mesh.polyhedron_compound_add": _groups_compound,
    }

    # Flat (single-level) galleries whose selector is a DYNAMIC enum, so
    # _static_enum_items reads nothing off the RNA type.  The resolver
    # calls the module's items callback with a shim, exactly like the
    # two-level GROUP_RESOLVER above.
    SELECTOR_RESOLVER = {
        "mesh.saddle_polyhedron_add": lambda: _pairs(
            _mod("saddle_polyhedron_generator")._solid_items(_Shim(), None)),
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
                # A module catalogue can be keyed more finely than the
                # operator's own group enum.  The parametric surface
                # table carries SINGLY/DOUBLY keys, which are families
                # of the *periodic* operator -- passing one as
                # `family=` is a TypeError, and those surfaces already
                # have their thumbnails on the periodic page.  Group
                # ids the operator will not accept are not ours.
                #
                # Only when there is a static list to check against:
                # the uniform and canonical operators have a *dynamic*
                # `family` too, so `labels` is empty for them and the
                # resolver's ids are the authority.
                if labels and gid not in labels:
                    continue
                for vid, label in items:
                    if vid in skip:
                        continue
                    out.append((vid, label,
                                dict(common, **{gprop: gid, iprop: vid}),
                                labels.get(gid, gid)))
        elif op in VARIANT_SELECTOR:
            prop = VARIANT_SELECTOR[op]
            items = (SELECTOR_RESOLVER[op]() if op in SELECTOR_RESOLVER
                     else _static_enum_items(op, prop))
            for vid, label in items:
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

    def hide_helpers(op):
        """Hide the working objects `op` leaves beside its result.

        Matched by name prefix (see HIDE_AFTER), because the operator
        names them deterministically and there is no other marker.
        """
        prefixes = HIDE_AFTER.get(op)
        if not prefixes:
            return 0
        hidden = 0
        for ob in bpy.data.objects:
            if any(ob.name.startswith(p) for p in prefixes):
                ob.hide_render = True
                hidden += 1
        return hidden

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
