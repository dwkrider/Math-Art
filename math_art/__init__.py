
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
    'crochet_generator',
    'dform_generator',
    'koman_generator',
    'space_curve_generator',
    'oloid_generator',
    'sphericon_generator',
    'steinmetz_generator',
    'orbis_generator',
    'monostatic_body_generator',
    'constant_width_generator',
    'prime_knot_generator',
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


class VIEW3D_MT_math_art_add(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_add"
    bl_label = "Math Art"

    def draw(self, context):
        lay = self.layout
        for spec in menu_defs.MENU_ORDER:
            lay.menu(spec.idname, icon=spec.icon)
        lay.separator()
        if hasattr(bpy.types, 'OBJECT_OT_symmetric_sculpture_add'):
            lay.operator_menu_enum("object.symmetric_sculpture_add",
                                   "preset",
                                   text="Symmetric Sculpture "
                                        "(Experimental)",
                                   icon='MOD_MIRROR')
        lay.menu(menu_defs.STYLES.idname, icon=menu_defs.STYLES.icon)


_MENUS = _SUBMENUS + (VIEW3D_MT_math_art_add,)


def _menu_func(self, context):
    self.layout.separator()
    self.layout.menu("VIEW3D_MT_math_art_add", icon='MATSHADERBALL')


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
