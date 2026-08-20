
# Math Art -- a Blender extension bundling mathematical sculpture
# generators. Each module is also usable on its own as a legacy
# single-file add-on; installed as an extension they share one
# "Math Art" submenu under Add.
#
# Modules are loaded resiliently: names listed below that do not
# exist yet (work in progress) are skipped with a console note and
# activate automatically once their file appears. Menu entries for
# operators that are not registered are hidden.

import importlib

import bpy

_MODULE_NAMES = [
    'scherk_collins_generator',
    'minimal_surface_toolkit',
    'seifert_surface_generator',
    'conway_operators',
    'zonohedra_generator',
    'zonish_generator',
    'waterman_generator',
    'rotegrity_generator',
    'weave_generator',
    'polylinks_generator',
    'polystix_generator',
    'platonic_twist_generator',
    'fractal_polyhedron_generator',
    'symmetrohedron_generator',
    'twisted_torus_generator',
    'polytope4d_generator',
    'tangle_generator',
    'symmetric_sculpture_generator',
    'leonardo_style',
    'sponge_generator',
    'mandelbulb_generator',
    'snowflake_generator',
    'l_system_generator',
    'turtle_curve_generator',
    'inflorescence_generator',
    'receptacle_generator',
    'leaf_generator',
    'growth_generator',
    'map_lsystem_generator',
    'strahler_style',
    'antoine_generator',
    'fractal_knot_generator',
    'apollonian_generator',
    'hyperbolic_surface_generator',
    'delaunay_generator',
    'bryant_generator',
    'crochet_generator',
    'dform_generator',
    'koman_generator',
    'space_curve_generator',
    'oloid_generator',
    'gem_generator',
    'sphericon_generator',
    'steinmetz_generator',
    'orbis_generator',
    'monostatic_body_generator',
    'constant_width_generator',
    'prime_knot_generator',
    'tight_knot_generator',
    'regular_solids_generator',
    'uniform_polyhedra_generator',
    'compound_generator',
    'toroidal_polyhedron_generator',
    'biscribed_solids_generator',
    'stellation_engine',
    'general_stellation',
    'other_polyhedra_generator',
    'canonical_polyhedra_generator',
    'attractor_generator',
    'dual_helix_generator',
    'stellated_weave_generator',
    'stereographic_projection_generator',
    'hyperbolic_honeycomb_generator',
    'algebraic_surface_generator',
    'invariant_manifold_generator',
    'spacefill_generator',
    'interlocking_generator',
    'link_generator',
    'topological_surface_generator',
    'hyperbolic_tiling_generator',
    'geodesic_generator',
    'curvature_color',
    'orbifold_sphere_generator',
    'fractal_tree_generator',
    'spiked_polyhedron_generator',
    'celtic_knot_generator',
    'voronoi_openwork',
    'torus_knot_generator',
    'hopf_fibration_generator',
    'polytwister_generator',
    'organic_wireframe',
    'minimal_polyhedron_generator',
    'squeeze_generator',
    'vortex_generator',
    'bubble_generator',
    'cmc_generator',
    'phyllotaxis_generator',
    'helical_surface_generator',
    'ruled_surface_generator',
    'curiosity_surface_generator',
    'supershape_generator',
    'spherical_harmonic_generator',
    'orbital_generator',
    'rolling_knot_generator',
    'woven_polyhedron_generator',
    'wallpaper_generator',
    'frieze_generator',
    'tiling_generator',
    'layer_generator',
    'modular_screen_generator',
    'relief_generator',
    'relief_solid_generator',
    'kuniform_generator',
    'monohedral_generator',
    'isohedral_generator',
    'aperiodic_generator',
    'reptile_generator',
    'voderberg_generator',
    'spiral_tiling_generator',
    'fractal_tiling_generator',
    'fractal_reptile_generator',
    'ifs_generator',
    'islamic_pattern_generator',
    'celtic_knot_2d_generator',
    'over_under_screen_generator',
    'knot_carpet_generator',
    'woven_double_shell_generator',
    'harmonic_knot_generator',
    'petal_knot_generator',
    'turks_head_generator',
    'rational_knot_generator',
    'substitution_knot_generator',
    'fractal_knotwork_generator',
    'willmore_generator',
]
# NOTE: engine subpackages (`minsurf`, `seifert`, `lsystem`, `knots`,
# `patterns`, `polyhedra`, `curve_frames`, `ifs`) and the `styles` namespace
# are deliberately absent -- they register nothing and are imported by their
# operator modules.  `pattern_common` used to be listed here; it is now
# `patterns/common.py`, and its `register` was always a no-op re-exported
# from `patterns.emit` (ADD_MENU = False), so nothing is lost by dropping it.
#
# `live` is absent for a different reason: it is not a generator but a
# layer over all of them, and it has to be installed AFTER the modules
# below have registered their operators, since it is built from them.

from . import live                                       # noqa: E402

# The Add-menu tree is data (menu_defs) plus an icon resolver
# (menu_icons); the classes further down are generated from them.
from . import menu_defs, menu_icons                       # noqa: E402

_MODULES = []
for _nm in _MODULE_NAMES:
    try:
        _MODULES.append(importlib.import_module('.' + _nm,
                                                __package__))
    except Exception as _e:                    # WIP module: skip
        print(f"Math Art: skipping module {_nm}: {_e}")


def _op(lay, idname, **kw):
    """Menu entry that hides itself while its operator is absent."""
    mod, _, fn = idname.partition('.')
    if hasattr(bpy.types, f"{mod.upper()}_OT_{fn}"):
        lay.operator(idname, **kw)


def _draw_entries(lay, entries):
    """Draw one menu's worth of table entries."""
    for entry in entries:
        if entry.op is None:
            lay.separator()
            continue
        kw = dict(menu_icons.icon_kwargs(entry))
        if entry.text is not None:
            kw['text'] = entry.text
        _op(lay, entry.op, **kw)


def _make_menu(spec):
    """Build a bpy.types.Menu class from a menu_defs.Menu record."""
    # `spec` is captured by closure, not passed as a default argument:
    # Blender rejects a Menu whose draw() takes anything but
    # (self, context).
    def draw(self, context):
        _draw_entries(self.layout, spec.entries)

    return type(spec.idname, (bpy.types.Menu,),
                {'bl_idname': spec.idname,
                 'bl_label': spec.label,
                 'draw': draw})


# The ten submenus are generated from the table in menu_defs.py; only
# the root menu below is hand-written, since it holds submenu links and
# the Symmetric Sculpture enum rather than a flat run of operators.
_SUBMENUS = tuple(_make_menu(s) for s in menu_defs.ALL_MENUS)


# --------------------------------------------------------------------
# Gallery menu (PROTOTYPE -- under evaluation against the plain rows)
# --------------------------------------------------------------------
# A submenu drawn as a grid of large thumbnails instead of 20 px rows.
# `template_icon` is the only way to draw a preview above icon size, and
# it comes with two constraints that shape everything here:
#
#   * it takes an icon_value and nothing else, so it cannot draw a
#     built-in glyph -- an entry with no baked render has to fall back
#     to an ordinary row, listed under the grid; and
#   * it draws a picture, not a button.  The clickable control is the
#     operator underneath each thumbnail.
#
# At GALLERY_SCALE = 4 the preview lands near 65 px on a 1.0-scale UI,
# which is what the 64 px bake was sized for.  Going bigger means
# re-baking at RES = 128 or the thumbnails start to soften.
GALLERY_COLUMNS = 4
GALLERY_SCALE = 4.0


def _draw_gallery(lay, entries):
    """Draw entries as a thumbnail grid, plus rows for the un-baked."""
    withpv = [(e, menu_icons.preview_id(e)) for e in entries
              if e.op is not None]
    grid = [(e, pv) for e, pv in withpv if pv]
    plain = [e for e, pv in withpv if not pv]

    # Blender right-aligns a ragged final row: three items after a row of
    # four land under columns 2..4 rather than 1..3.  grid_flow does it,
    # hand-built rows of columns do it, and padding the count out does
    # not help -- neither label(text="") nor template_icon(icon_value=0)
    # reserves any width.
    #
    # So the grid is built as real columns instead, and the items are
    # dealt down them: column j takes indices j, j+COLS, j+2*COLS...
    # Reading across the columns still gives the table's order, every
    # column is its own layout so the pitch cannot drift, and a short
    # last row simply leaves its rightmost columns one item shorter.
    cols = min(GALLERY_COLUMNS, len(grid))
    row = lay.row(align=False)
    for j in range(cols):
        col = row.column(align=False)
        for entry, pv in grid[j::cols]:
            cell = col.column(align=True)
            cell.template_icon(icon_value=pv, scale=GALLERY_SCALE)
            kw = {'text': entry.text} if entry.text is not None else {}
            _op(cell, entry.op, **kw)

    if plain:
        lay.separator()
        _draw_entries(lay, plain)


def _make_gallery_menu(spec):
    """Gallery variant of `spec`: same label, distinct idname.

    The label is unchanged because the gallery *is* the menu now -- only
    the class needs a separate idname so it can coexist with the plain
    row version, which stays registered for comparison.
    """
    idname = spec.idname + "_gallery"

    def draw(self, context):
        _draw_gallery(self.layout, spec.entries)

    return type(idname, (bpy.types.Menu,),
                {'bl_idname': idname,
                 'bl_label': spec.label,
                 'draw': draw})


# Every content submenu is drawn as a gallery.  STYLES is deliberately
# not: all five of its entries are pinned to built-in glyphs, so its
# gallery would have nothing to put in the grid and would fall back to
# exactly the plain menu -- a duplicate class for an identical result.
#
# The plain row-per-entry menus stay registered alongside the galleries
# so a single build can still be compared either way.
GALLERY_MENUS = menu_defs.MENU_ORDER

_GALLERIES = tuple(_make_gallery_menu(s) for s in GALLERY_MENUS)
_GALLERY_BY_IDNAME = {g.bl_idname: g for g in _GALLERIES}


class VIEW3D_MT_math_art_add(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_add"
    bl_label = "Math Art"

    def draw(self, context):
        lay = self.layout
        for spec in menu_defs.MENU_ORDER:
            gal = _GALLERY_BY_IDNAME.get(spec.idname + "_gallery")
            lay.menu(gal.bl_idname if gal else spec.idname,
                     icon=spec.icon)
        lay.separator()
        # Plain entries, not operator_menu_enum: a preset belongs in the
        # redo panel with the rest of the settings, not spilled up into
        # the Add menu as a submenu of its own.  The list lives in
        # menu_defs so the icon baker and the docs gate can see it.
        _draw_entries(lay, menu_defs.ROOT_ENTRIES)
        # Styles stays a plain row menu -- see GALLERY_MENUS above.
        lay.menu(menu_defs.STYLES.idname, icon=menu_defs.STYLES.icon)


_MENUS = _SUBMENUS + _GALLERIES + (VIEW3D_MT_math_art_add,)


# The "Math Art" line in Blender's own Add menu borrows a baked render
# rather than a built-in glyph, so the add-on is recognisable before any
# of its submenus are open.  The Polyhedral Tangle reads well at the
# ~20 px an Add-menu row gives it: dense, symmetric, and unmistakably
# not one of Blender's primitives.  MATSHADERBALL is the fallback for an
# un-baked build.
ROOT_ICON_OP = "mesh.tangle_add"
ROOT_ICON_FALLBACK = 'MATSHADERBALL'


def _menu_func(self, context):
    self.layout.separator()
    self.layout.menu("VIEW3D_MT_math_art_add",
                     **menu_icons.op_icon_kwargs(ROOT_ICON_OP,
                                                 ROOT_ICON_FALLBACK))


_ACTIVE = []


def register():
    _ACTIVE.clear()
    for m in _MODULES:
        try:
            m.ADD_MENU = False   # entries live in the Math Art menu
            m.register()
            _ACTIVE.append(m)
        except Exception as e:
            print(f"Math Art: register failed for "
                  f"{m.__name__}: {e}")
    menu_icons.load()        # baked icons; a no-op on an un-baked build
    for c in _MENUS:
        bpy.utils.register_class(c)
    bpy.types.VIEW3D_MT_add.append(_menu_func)
    # Last, and never fatally: the sidebar is built by reading the
    # operators that were just registered, and an add-on that fails to
    # load because its panel could not be derived would be a poor trade
    # for a convenience.
    try:
        live.install(_ACTIVE)
    except Exception as e:
        print(f"Math Art: live object editing unavailable: {e}")


def unregister():
    try:
        live.uninstall()
    except Exception as e:
        print(f"Math Art: live uninstall failed: {e}")
    bpy.types.VIEW3D_MT_add.remove(_menu_func)
    for c in reversed(_MENUS):
        bpy.utils.unregister_class(c)
    menu_icons.unload()      # Blender leak-checks preview collections
    for m in reversed(_ACTIVE):
        try:
            m.unregister()
        except Exception as e:
            print(f"Math Art: unregister failed for "
                  f"{m.__name__}: {e}")
    _ACTIVE.clear()
