
# Declarative description of the "Math Art" Add-menu tree.
#
# This module is deliberately free of any `bpy` import: it is plain data
# describing which operator sits in which submenu, with what label and
# what icon.  Two very different consumers read it:
#
#   * math_art/__init__.py builds the Blender menu classes from it at
#     register() time, and
#   * renders/bake_menu_icons.py iterates it headlessly to render one
#     thumbnail per entry into math_art/icons/.
#
# Keeping the table in one bpy-free place is what keeps those two in
# sync -- an entry cannot appear in a menu without the bake harness
# knowing it exists, and `_selftest()` below fails if they drift.
#
# Entry fields
#   op       operator id ("mesh.foo_add").  None marks a separator.
#   text     label override, or None to use the operator's own bl_label.
#   icon     built-in Blender icon.
#   builtin  True to pin the entry to its built-in icon (see below).
#
# Icon resolution, in order:
#
#   1. `builtin=True`  -- always draw the built-in `icon`.  The baker
#      skips the entry entirely, so no PNG is ever produced for it.
#      Use this where a built-in glyph genuinely reads better than a
#      render would: the Styles operators modify an existing object
#      rather than making a shape, so a thumbnail of "a sculpture"
#      would say nothing about what the entry does.
#   2. a baked math_art/icons/<op>.png exists -- draw that.
#   3. otherwise -- draw the built-in `icon`.
#
# So every entry carries a sensible built-in icon regardless: an
# un-baked build, a partially-baked build and a build where someone
# judged the render worse than the glyph all look correct.  Flipping
# one entry between a render and a glyph is a one-word edit here.

from collections import namedtuple

Entry = namedtuple('Entry', 'op text icon builtin')
Menu = namedtuple('Menu', 'idname label icon entries')


def _e(op, icon, text=None, builtin=False):
    return Entry(op, text, icon, builtin)


SEP = Entry(None, None, None, False)


# --------------------------------------------------------------------
# Submenus, in the order they appear under Add > Math Art
# --------------------------------------------------------------------

SURFACES = Menu(
    "VIEW3D_MT_math_art_minimal", "Surfaces", 'SURFACE_NSPHERE', [
        _e("mesh.scherk_collins_add", 'MESH_TORUS',
           "Scherk-Collins Sculpture"),
        _e("mesh.parametric_minimal_add", 'SURFACE_NSPHERE',
           "Minimal Surfaces"),
        _e("mesh.periodic_minimal_add", 'MESH_ICOSPHERE',
           "Minimal Surfaces (Periodic)"),
        _e("mesh.minimal_knot_span_add", 'MESH_TORUS'),
        # Spans whatever curves are selected rather than adding a shape,
        # so it has no subject of its own -- the baker builds it two
        # crossed circles to bridge (tools/subjects.py SETUP).
        _e("object.minimal_span", 'OUTLINER_OB_SURFACE'),
        _e("mesh.seifert_surface_add", 'MOD_SIMPLIFY', "Seifert Surface"),
        _e("mesh.algebraic_surface_add", 'SURFACE_NSURFACE'),
        _e("mesh.topological_surface_add", 'MESH_TORUS'),
        _e("mesh.minimal_surface_polyhedron_add", 'MESH_UVSPHERE'),
        _e("mesh.squeeze_add", 'MOD_SIMPLEDEFORM'),
        _e("mesh.vertex_vortices_add", 'FORCE_VORTEX'),
        _e("mesh.helical_surface_add", 'MOD_SCREW'),
        _e("mesh.ruled_surface_add", 'MOD_SCREW'),
        _e("mesh.dform_add", 'MOD_CLOTH', "D-Form"),
        _e("mesh.curiosity_surface_add", 'SURFACE_DATA'),
        _e("mesh.invariant_manifold_add", 'SURFACE_NSURFACE'),
        _e("mesh.supershape_add", 'SURFACE_NSPHERE'),
        _e("mesh.spherical_harmonic_add", 'SURFACE_NSPHERE'),
        _e("mesh.orbital_add", 'META_BALL'),
        _e("mesh.hyperbolic_surface_add", 'MESH_CAPSULE'),
        # One constant-mean-curvature generator: Delaunay surfaces and
        # their roulettes, bubbletons grafted onto them, Wente's closed
        # torus, and the elastic tori of the spherical space form.
        _e("mesh.delaunay_surface_add", 'META_CAPSULE',
           "CMC Surfaces"),
        _e("mesh.bryant_surface_add", 'MESH_UVSPHERE'),
        _e("mesh.crochet_add", 'MOD_CLOTH'),
        _e("mesh.willmore_add", 'MESH_TORUS', "Willmore Surface"),
    ])

POLYHEDRA = Menu(
    "VIEW3D_MT_math_art_polyhedra", "Polyhedra", 'MESH_ICOSPHERE', [
        _e("mesh.regular_solid_add", 'MESH_ICOSPHERE'),
        _e("mesh.uniform_polyhedron_add", 'MESH_ICOSPHERE'),
        _e("mesh.biscribed_solid_add", 'MESH_ICOSPHERE'),
        _e("mesh.polyhedron_compound_add", 'MESH_ICOSPHERE'),
        _e("mesh.toroidal_polyhedron_add", 'MESH_TORUS'),
        _e("mesh.polyhedral_torus_add", 'MESH_TORUS'),
        _e("mesh.notable_polyhedron_add", 'MESH_ICOSPHERE'),
        _e("mesh.canonical_polyhedron_add", 'MESH_ICOSPHERE'),
        _e("mesh.icosahedron_stellation_add", 'MESH_ICOSPHERE'),
        _e("mesh.general_stellation_add", 'MESH_ICOSPHERE'),
        _e("mesh.star_prism_add", 'MESH_CYLINDER'),
        _e("mesh.conway_add", 'MESH_ICOSPHERE'),
        _e("mesh.zonohedron_add", 'MESH_UVSPHERE'),
        # Zonohedrification, zonish polyhedra and the rhombohedral
        # dissections are one seed-plus-star construction, so one entry.
        _e("mesh.zonish_add", 'MESH_ICOSPHERE', "Zonohedrification"),
        _e("mesh.transpolyhedron_add", 'MESH_ICOSPHERE'),
        _e("mesh.twelve_faced_add", 'MESH_ICOSPHERE'),
        _e("mesh.waterman_add", 'MESH_ICOSPHERE'),
        _e("mesh.symmetrohedron_add", 'MESH_ICOSPHERE'),
        _e("mesh.polytope4d_add", 'MESH_CUBE'),
        _e("mesh.polytwister_add", 'MESH_TORUS'),
        _e("mesh.hyperbolic_honeycomb_add", 'META_BALL'),
        _e("mesh.spacefill_add", 'SNAP_VOLUME'),
        _e("mesh.interlocking_add", 'MOD_BUILD'),
        _e("mesh.geodesic_add", 'MESH_UVSPHERE'),
        _e("mesh.spiked_polyhedron_add", 'LIGHT_SUN'),
    ])

FRACTALS = Menu(
    "VIEW3D_MT_math_art_fractals", "Fractals", 'MESH_CUBE', [
        _e("mesh.sponge_add", 'MESH_CUBE'),
        _e("mesh.mandelbulb_add", 'META_BALL'),
        _e("mesh.snowflake_add", 'FREEZE'),
        _e("mesh.apollonian_add", 'MESH_CIRCLE'),
        _e("mesh.fractal_polyhedron_add", 'OUTLINER_OB_POINTCLOUD'),
        _e("curve.space_filling_add", 'CURVE_DATA'),
        _e("curve.lsystem_add", 'GRAPH'),
        _e("curve.turtle_curve_add", 'MOD_CURVE'),
        _e("curve.fractal_tree_add", 'GRAPH'),
        _e("mesh.fractal_tiling_add", 'MESH_CIRCLE'),
        _e("mesh.fractal_knotwork_add", 'OUTLINER_OB_CURVES'),
        _e("mesh.fractal_reptile_add", 'MOD_TRIANGULATE'),
        _e("mesh.ifs_add", 'MOD_REMESH'),
    ])

PLANTS = Menu(
    "VIEW3D_MT_math_art_plants", "Plants & Growth", 'OUTLINER_OB_CURVES', [
        _e("curve.lsystem_add", 'GRAPH'),
        _e("curve.inflorescence_add", 'OUTLINER_OB_CURVES'),
        _e("mesh.receptacle_add", 'MESH_UVSPHERE'),
        _e("mesh.leaf_add", 'OUTLINER_OB_MESH'),
        _e("curve.growth_add", 'OUTLINER_OB_FORCE_FIELD'),
        _e("mesh.map_lsystem_add", 'MESH_GRID'),
        _e("curve.fractal_tree_add", 'GRAPH'),
        _e("mesh.phyllotaxis_add", 'MESH_CIRCLE'),
    ])

KNOTS = Menu(
    "VIEW3D_MT_math_art_knots", "Knots & Curves", 'FORCE_VORTEX', [
        _e("curve.prime_knot_add", 'FORCE_VORTEX'),
        _e("curve.tight_knot_add", 'FORCE_VORTEX'),
        _e("curve.tight_link_add", 'LINKED'),
        _e("curve.torus_knot_add", 'FORCE_VORTEX'),
        _e("curve.hopf_fibration_add", 'FORCE_MAGNETIC'),
        # Hopf tori, Willmore tori (over free elasticae) and
        # constrained Willmore tori (over constrained elasticae) are
        # one construction with one dial, so they are one entry.
        _e("mesh.hopf_torus_add", 'MESH_TORUS',
           "Hopf & Willmore Tori"),
        _e("curve.harmonic_knot_add", 'FORCE_HARMONIC'),
        _e("curve.petal_knot_add", 'CURVE_NCIRCLE'),
        _e("curve.rational_knot_add", 'MOD_CURVE'),
        _e("curve.fractal_knot_add", 'FORCE_VORTEX'),
        _e("curve.substitution_knot_add", 'CURVE_NCURVE'),
        _e("curve.math_link_add", 'LINKED'),
        _e("mesh.antoine_add", 'LINKED'),
        _e("curve.attractor_add", 'RNDCURVE', "Strange Attractor"),
        _e("curve.dual_helix_add", 'MOD_SCREW'),
        _e("mesh.rolling_knot_add", 'PHYSICS'),
    ])

WEAVES = Menu(
    "VIEW3D_MT_math_art_weaves", "Weaves & Tangles", 'MOD_LATTICE', [
        _e("mesh.polylinks_add", 'MESH_CIRCLE'),
        _e("mesh.polystix_add", 'MESH_CYLINDER'),
        _e("mesh.tangle_add", 'MOD_BOOLEAN'),
        _e("mesh.poly_weave_add", 'MOD_LATTICE'),
        _e("mesh.rotegrity_add", 'SPHERE'),
        _e("mesh.stellated_weave_add", 'MOD_LATTICE'),
        _e("curve.celtic_knot_add", 'MOD_LATTICE'),
        _e("mesh.woven_polyhedron_add", 'MESH_ICOSPHERE'),
        _e("mesh.woven_double_shell_add", 'MESH_UVSPHERE'),
        _e("mesh.turks_head_add", 'CURVE_NCIRCLE'),
    ])

PATTERNS = Menu(
    "VIEW3D_MT_math_art_patterns", "Patterns", 'MESH_GRID', [
        _e("mesh.frieze_add", 'MOD_ARRAY'),
        _e("mesh.wallpaper_add", 'MOD_MIRROR'),
        _e("mesh.layer_add", 'MOD_SOLIDIFY'),
        _e("mesh.modular_screen_add", 'MOD_WIREFRAME'),
        _e("mesh.relief_panel_add", 'MOD_DISPLACE'),
        _e("mesh.relief_solid_add", 'MATSPHERE'),
        SEP,
        _e("mesh.tiling_add", 'MESH_GRID'),
        _e("mesh.kuniform_add", 'MESH_GRID'),
        _e("mesh.monohedral_add", 'MESH_PLANE'),
        _e("mesh.isohedral_add", 'MOD_UVPROJECT'),
        _e("mesh.aperiodic_add", 'MESH_ICOSPHERE'),
        _e("mesh.reptile_add", 'MESH_GRID'),
        _e("mesh.voderberg_add", 'FORCE_VORTEX'),
        _e("mesh.spiral_tiling_add", 'FORCE_VORTEX'),
        _e("mesh.islamic_pattern_add", 'SOLO_ON'),
        _e("mesh.celtic_knot_2d_add", 'MOD_LATTICE'),
        _e("mesh.over_under_screen_add", 'MESH_GRID'),
        _e("mesh.knot_carpet_add", 'MESH_CIRCLE'),
        SEP,
        _e("mesh.hyperbolic_tiling_add", 'MESH_CIRCLE'),
    ])

# non-spherical shapes that roll ("rolloids"): rollers of constant
# width, developable rollers, and balancing solids
ROLLERS = Menu(
    "VIEW3D_MT_math_art_rollers", "Rollers", 'MESH_CAPSULE', [
        _e("mesh.oloid_add", 'MESH_CAPSULE'),
        _e("mesh.sphericon_add", 'MESH_CAPSULE'),
        _e("mesh.steinmetz_add", 'MESH_CYLINDER'),
        _e("mesh.orbis_add", 'MESH_TORUS'),
        _e("mesh.constant_width_add", 'MESH_CIRCLE'),
        _e("mesh.monostatic_body_add", 'MESH_UVSPHERE', "Monostatic Body"),
        _e("mesh.rolling_knot_add", 'PHYSICS'),
    ])

# Only generators that live nowhere else.  Everything with a home of
# its own is listed THERE and not here: the rollers (oloid, sphericon,
# Steinmetz, orbis, constant-width, Gomboc) in Rollers, polytwisters
# and hyperbolic honeycombs in Polyhedra, phyllotaxis in Plants.  A
# generator appearing in two menus makes both of them harder to read
# and neither of them authoritative.
ODDS = Menu(
    "VIEW3D_MT_math_art_odds", "Odds & Ends", 'MESH_TORUS', [
        _e("mesh.gem_add", 'MESH_ICOSPHERE', "Faceted Gemstone"),
        _e("mesh.gem_cabochon_add", 'MESH_CAPSULE', "Cabochon Gemstone"),
        _e("mesh.koman_add", 'MOD_SCREW', "Koman Developable"),
        _e("mesh.platonic_twist_add", 'MOD_SCREW'),
        _e("mesh.twisted_torus_add", 'MESH_TORUS'),
        _e("mesh.stereographic_add", 'LIGHT_POINT'),
        _e("mesh.orbifold_sphere_add", 'MOD_MIRROR'),
        _e("mesh.bubble_cluster_add", 'SPHERE'),
        _e("mesh.relaxed_bubble_add", 'META_BALL'),
        _e("mesh.cmc_capillary_add", 'MATFLUID'),
    ])

# Styles restyle whatever is already selected rather than adding a
# shape of their own, so every entry is pinned to its built-in glyph:
# a render here would show some arbitrary sample object and say nothing
# about what the operator does.
STYLES = Menu(
    "VIEW3D_MT_math_art_styles", "Styles", 'MOD_SOLIDIFY', [
        _e("object.leonardo_add", 'MESH_ICOSPHERE', builtin=True),
        _e("object.curvature_color_add", 'COLORSET_01_VEC', builtin=True),
        _e("object.voronoi_openwork_add", 'MOD_REMESH',
           "Voronoi Openwork (Experimental)", builtin=True),
        _e("object.organic_wireframe_add", 'MOD_WIREFRAME', builtin=True),
        _e("object.strahler_add", 'MOD_SIMPLIFY', builtin=True),
    ])


# Draw order of the submenus under Add > Math Art.  STYLES is listed
# last by the root menu, after the ROOT_ENTRIES below, so it is not
# part of this run.
MENU_ORDER = (SURFACES, POLYHEDRA, FRACTALS, PLANTS, KNOTS, WEAVES,
              PATTERNS, ROLLERS, ODDS)

# Every menu that gets a generated class, in registration order.
ALL_MENUS = MENU_ORDER + (STYLES,)

# Operators drawn directly on the root "Math Art" menu rather than in
# any submenu.  Symmetric Sculpture is a single operator with no family
# around it, so a submenu of one would be noise -- but it still has to
# be listed *here* rather than only in __init__.py's draw(), because
# this module is what the icon baker and the documentation coverage
# gate enumerate.  An operator drawn straight by __init__.py and absent
# from this table is invisible to both: it silently gets no baked icon
# and no doc page, which is exactly what happened to this entry.
ROOT_ENTRIES = (
    _e("object.symmetric_sculpture_add", 'MOD_MIRROR'),
)


def iter_entries():
    """Yield every non-separator Entry across all menus, in menu order.

    Entries repeated across menus (an L-system is both a fractal and a
    plant) are yielded once per appearance; use `unique_ops()` when a
    single pass per operator is wanted, as the icon baker needs.  The
    root-level entries come last, matching their draw order.
    """
    for menu in ALL_MENUS:
        for entry in menu.entries:
            if entry.op is not None:
                yield entry
    for entry in ROOT_ENTRIES:
        if entry.op is not None:
            yield entry


def unique_ops():
    """Sorted list of the distinct operator ids used by the menus."""
    return sorted({e.op for e in iter_entries()})


def bakeable_ops():
    """Sorted operator ids that a baked icon is actually wanted for.

    This is the authoritative work-list for the icon baker: one PNG per
    operator, however many menus that operator appears in, minus the
    entries pinned to a built-in glyph.  An operator counts as pinned
    only if EVERY appearance of it is pinned -- if it is a plain entry
    in one menu and a pinned one in another, the PNG is still needed.
    """
    return sorted({e.op for e in iter_entries() if not e.builtin})


def _selftest():
    """Structural checks on the table (no Blender required)."""
    seen_idnames = set()
    for menu in ALL_MENUS:
        if not menu.idname.startswith("VIEW3D_MT_math_art_"):
            raise AssertionError(f"odd menu idname: {menu.idname}")
        if menu.idname in seen_idnames:
            raise AssertionError(f"duplicate menu idname: {menu.idname}")
        seen_idnames.add(menu.idname)
        if not menu.label or not menu.icon:
            raise AssertionError(f"{menu.idname}: missing label/icon")

        # An operator listed twice in the SAME menu is always a mistake;
        # across menus it can be deliberate (fractal_tree is a fractal
        # and a plant), so that is checked per-menu only.
        ops = [e.op for e in menu.entries if e.op is not None]
        if len(ops) != len(set(ops)):
            dupes = sorted({o for o in ops if ops.count(o) > 1})
            raise AssertionError(f"{menu.idname}: repeated {dupes}")
        if not ops:
            raise AssertionError(f"{menu.idname}: no entries")

        for entry in menu.entries:
            if entry.op is None:            # separator
                if entry.text or entry.icon or entry.builtin:
                    raise AssertionError("separator carries text/icon")
                continue
            prefix, _, rest = entry.op.partition('.')
            if prefix not in ('mesh', 'curve', 'object') or not rest:
                raise AssertionError(f"bad operator id: {entry.op}")
            # Every entry needs a built-in icon whether or not it is
            # pinned to one: it is what an un-baked build falls back to.
            if not entry.icon:
                raise AssertionError(f"{entry.op}: no fallback icon")
            if entry.icon != entry.icon.upper():
                raise AssertionError(f"{entry.op}: icon not upper-case")
            if not isinstance(entry.builtin, bool):
                raise AssertionError(f"{entry.op}: builtin not a bool")

    # Root-level entries get the same structural checks as menu ones,
    # and must not duplicate an operator that already has a submenu home.
    submenu_ops = {e.op for m in ALL_MENUS for e in m.entries if e.op}
    for entry in ROOT_ENTRIES:
        if entry.op is None:
            raise AssertionError("ROOT_ENTRIES holds a separator")
        prefix, _, rest = entry.op.partition('.')
        if prefix not in ('mesh', 'curve', 'object') or not rest:
            raise AssertionError(f"bad root operator id: {entry.op}")
        if not entry.icon or entry.icon != entry.icon.upper():
            raise AssertionError(f"{entry.op}: bad/missing fallback icon")
        if entry.op in submenu_ops:
            raise AssertionError(f"{entry.op}: drawn at root AND in a submenu")

    # The root ordering must cover every menu exactly once.  Menu is a
    # namedtuple holding a list, so it is unhashable -- compare on the
    # idnames, which are unique by the check above.
    order = [m.idname for m in ALL_MENUS]
    if len(set(order)) != len(order):
        raise AssertionError("ALL_MENUS repeats a menu")
    if STYLES.idname in [m.idname for m in MENU_ORDER]:
        raise AssertionError("STYLES is drawn separately, not in MENU_ORDER")

    ops, bakeable = unique_ops(), bakeable_ops()
    if len(ops) < 100:
        raise AssertionError(f"only {len(ops)} operators -- table truncated?")
    if not set(bakeable) <= set(ops):
        raise AssertionError("bakeable_ops() is not a subset of unique_ops()")
    print(f"menu_defs: {len(ALL_MENUS)} menus, "
          f"{len(list(iter_entries()))} entries, "
          f"{len(ops)} distinct operators, "
          f"{len(bakeable)} to bake "
          f"({len(ops) - len(bakeable)} pinned to built-in icons)")
