
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
    'fractal_surface_generator',
    'l_system_generator',
    'antoine_generator',
    'fractal_knot_generator',
    'apollonian_generator',
    'hyperbolic_surface_generator',
    'crochet_generator',
    'space_curve_generator',
    'oloid_generator',
    'sphericon_generator',
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
    'rolling_knot_generator',
    'woven_polyhedron_generator',
    'pattern_common',
    'wallpaper_generator',
    'frieze_generator',
    'tiling_generator',
    'layer_generator',
    'modular_screen_generator',
    'kuniform_generator',
    'monohedral_generator',
    'isohedral_generator',
    'aperiodic_generator',
    'reptile_generator',
    'voderberg_generator',
    'spiral_tiling_generator',
    'fractal_tiling_generator',
    'fractal_reptile_generator',
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
    'we_builders',
    'minimal_surface_zoo',
]

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


class VIEW3D_MT_math_art_minimal(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_minimal"
    bl_label = "Surfaces"

    def draw(self, context):
        lay = self.layout
        _op(lay, "mesh.scherk_collins_add",
            text="Scherk-Collins Sculpture", icon='MESH_TORUS')
        _op(lay, "mesh.parametric_minimal_add",
            text="Minimal Surfaces", icon='SURFACE_NSPHERE')
        _op(lay, "mesh.periodic_minimal_add",
            text="Minimal Surfaces (Periodic)",
            icon='MESH_ICOSPHERE')
        _op(lay, "mesh.minimal_knot_span_add", icon='MESH_TORUS')
        _op(lay, "object.minimal_span", icon='OUTLINER_OB_SURFACE')
        _op(lay, "mesh.seifert_surface_add",
            text="Seifert Surface", icon='MOD_SIMPLIFY')
        _op(lay, "mesh.algebraic_surface_add",
            icon='SURFACE_NSURFACE')
        _op(lay, "mesh.topological_surface_add", icon='MESH_TORUS')
        _op(lay, "mesh.minimal_surface_polyhedron_add",
            icon='MESH_UVSPHERE')
        _op(lay, "mesh.squeeze_add", icon='MOD_SIMPLEDEFORM')
        _op(lay, "mesh.vertex_vortices_add", icon='FORCE_VORTEX')
        _op(lay, "mesh.helical_surface_add", icon='MOD_SCREW')
        _op(lay, "mesh.ruled_surface_add", icon='MOD_SCREW')
        _op(lay, "mesh.curiosity_surface_add",
            icon='SURFACE_DATA')
        _op(lay, "mesh.supershape_add", icon='SURFACE_NSPHERE')
        _op(lay, "mesh.hyperbolic_surface_add", icon='MESH_CAPSULE')
        _op(lay, "mesh.crochet_add", icon='MOD_CLOTH')


class VIEW3D_MT_math_art_polyhedra(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_polyhedra"
    bl_label = "Polyhedra"

    def draw(self, context):
        lay = self.layout
        _op(lay, "mesh.regular_solid_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.uniform_polyhedron_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.biscribed_solid_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.polyhedron_compound_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.toroidal_polyhedron_add", icon='MESH_TORUS')
        _op(lay, "mesh.polyhedral_torus_add", icon='MESH_TORUS')
        _op(lay, "mesh.notable_polyhedron_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.canonical_polyhedron_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.icosahedron_stellation_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.general_stellation_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.star_prism_add", icon='MESH_CYLINDER')
        _op(lay, "mesh.conway_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.zonohedron_add", icon='MESH_UVSPHERE')
        _op(lay, "mesh.waterman_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.symmetrohedron_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.polytope4d_add", icon='MESH_CUBE')
        _op(lay, "mesh.hyperbolic_honeycomb_add", icon='META_BALL')
        _op(lay, "mesh.spacefill_add", icon='SNAP_VOLUME')
        _op(lay, "mesh.interlocking_add", icon='MOD_BUILD')
        _op(lay, "mesh.geodesic_add", icon='MESH_UVSPHERE')
        _op(lay, "mesh.spiked_polyhedron_add", icon='LIGHT_SUN')


class VIEW3D_MT_math_art_fractals(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_fractals"
    bl_label = "Fractals"

    def draw(self, context):
        lay = self.layout
        _op(lay, "mesh.sponge_add", icon='MESH_CUBE')
        _op(lay, "mesh.mandelbulb_add", icon='META_BALL')
        _op(lay, "mesh.snowflake_add", icon='FREEZE')
        _op(lay, "mesh.fractal_surface_add", icon='RNDCURVE')
        _op(lay, "mesh.apollonian_add", icon='MESH_CIRCLE')
        _op(lay, "mesh.fractal_polyhedron_add",
            icon='OUTLINER_OB_POINTCLOUD')
        _op(lay, "curve.space_filling_add", icon='CURVE_DATA')
        _op(lay, "curve.lsystem_add", icon='GRAPH')
        _op(lay, "curve.fractal_tree_add", icon='GRAPH')
        _op(lay, "mesh.fractal_tiling_add", icon='MESH_CIRCLE')
        _op(lay, "mesh.fractal_knotwork_add", icon='OUTLINER_OB_CURVES')
        _op(lay, "mesh.fractal_reptile_add", icon='MOD_TRIANGULATE')


class VIEW3D_MT_math_art_knots(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_knots"
    bl_label = "Knots & Curves"

    def draw(self, context):
        lay = self.layout
        _op(lay, "curve.prime_knot_add", icon='FORCE_VORTEX')
        _op(lay, "curve.torus_knot_add", icon='FORCE_VORTEX')
        _op(lay, "curve.harmonic_knot_add", icon='FORCE_HARMONIC')
        _op(lay, "curve.petal_knot_add", icon='CURVE_NCIRCLE')
        _op(lay, "curve.rational_knot_add", icon='MOD_CURVE')
        _op(lay, "curve.fractal_knot_add", icon='FORCE_VORTEX')
        _op(lay, "curve.substitution_knot_add", icon='CURVE_NCURVE')
        _op(lay, "curve.math_link_add", icon='LINKED')
        _op(lay, "mesh.antoine_add", icon='LINKED')
        if hasattr(bpy.types, 'CURVE_OT_attractor_add'):
            lay.operator_menu_enum("curve.attractor_add", "preset",
                                   text="Strange Attractor",
                                   icon='RNDCURVE')
        _op(lay, "curve.dual_helix_add", icon='MOD_SCREW')
        _op(lay, "mesh.rolling_knot_add", icon='PHYSICS')


class VIEW3D_MT_math_art_weaves(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_weaves"
    bl_label = "Weaves & Tangles"

    def draw(self, context):
        lay = self.layout
        _op(lay, "mesh.polylinks_add", icon='MESH_CIRCLE')
        _op(lay, "mesh.polystix_add", icon='MESH_CYLINDER')
        _op(lay, "mesh.tangle_add", icon='MOD_BOOLEAN')
        _op(lay, "mesh.poly_weave_add", icon='MOD_LATTICE')
        _op(lay, "mesh.rotegrity_add", icon='SPHERE')
        _op(lay, "mesh.stellated_weave_add", icon='MOD_LATTICE')
        _op(lay, "curve.celtic_knot_add", icon='MOD_LATTICE')
        _op(lay, "mesh.woven_polyhedron_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.woven_double_shell_add", icon='MESH_UVSPHERE')
        _op(lay, "mesh.turks_head_add", icon='CURVE_NCIRCLE')


class VIEW3D_MT_math_art_odds(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_odds"
    bl_label = "Odds & Ends"

    def draw(self, context):
        lay = self.layout
        _op(lay, "mesh.platonic_twist_add", icon='MOD_SCREW')
        _op(lay, "mesh.twisted_torus_add", icon='MESH_TORUS')
        _op(lay, "mesh.oloid_add", icon='MESH_CAPSULE')
        _op(lay, "mesh.sphericon_add", icon='MESH_CAPSULE')
        _op(lay, "mesh.monostatic_body_add", text="Monostatic Body",
            icon='MESH_UVSPHERE')
        _op(lay, "mesh.constant_width_add", icon='MESH_CIRCLE')
        _op(lay, "mesh.stereographic_add", icon='LIGHT_POINT')
        _op(lay, "mesh.orbifold_sphere_add", icon='MOD_MIRROR')
        _op(lay, "mesh.bubble_cluster_add", icon='SPHERE')
        _op(lay, "mesh.phyllotaxis_add", icon='OUTLINER_OB_POINTCLOUD')


class VIEW3D_MT_math_art_patterns(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_patterns"
    bl_label = "Patterns"

    def draw(self, context):
        lay = self.layout
        _op(lay, "mesh.frieze_add", icon='MOD_ARRAY')
        _op(lay, "mesh.wallpaper_add", icon='MOD_MIRROR')
        _op(lay, "mesh.layer_add", icon='MOD_SOLIDIFY')
        _op(lay, "mesh.modular_screen_add", icon='MOD_WIREFRAME')
        lay.separator()
        _op(lay, "mesh.tiling_add", icon='MESH_GRID')
        _op(lay, "mesh.kuniform_add", icon='MESH_GRID')
        _op(lay, "mesh.monohedral_add", icon='MESH_PLANE')
        _op(lay, "mesh.isohedral_add", icon='MOD_UVPROJECT')
        _op(lay, "mesh.aperiodic_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.reptile_add", icon='MESH_GRID')
        _op(lay, "mesh.voderberg_add", icon='FORCE_VORTEX')
        _op(lay, "mesh.spiral_tiling_add", icon='FORCE_VORTEX')
        _op(lay, "mesh.islamic_pattern_add", icon='SOLO_ON')
        _op(lay, "mesh.celtic_knot_2d_add", icon='MOD_LATTICE')
        _op(lay, "mesh.over_under_screen_add", icon='MESH_GRID')
        _op(lay, "mesh.knot_carpet_add", icon='MESH_CIRCLE')
        lay.separator()
        _op(lay, "mesh.hyperbolic_tiling_add", icon='MESH_CIRCLE')


class VIEW3D_MT_math_art_styles(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_styles"
    bl_label = "Styles"

    def draw(self, context):
        lay = self.layout
        _op(lay, "object.leonardo_add", icon='MESH_ICOSPHERE')
        _op(lay, "object.curvature_color_add",
            icon='COLORSET_01_VEC')
        _op(lay, "object.voronoi_openwork_add",
            text="Voronoi Openwork (Experimental)", icon='MOD_REMESH')
        _op(lay, "object.organic_wireframe_add",
            icon='MOD_WIREFRAME')


class VIEW3D_MT_math_art_add(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_add"
    bl_label = "Math Art"

    def draw(self, context):
        lay = self.layout
        lay.menu("VIEW3D_MT_math_art_minimal",
                 icon='SURFACE_NSPHERE')
        lay.menu("VIEW3D_MT_math_art_polyhedra",
                 icon='MESH_ICOSPHERE')
        lay.menu("VIEW3D_MT_math_art_fractals", icon='MESH_CUBE')
        lay.menu("VIEW3D_MT_math_art_knots", icon='FORCE_VORTEX')
        lay.menu("VIEW3D_MT_math_art_weaves", icon='MOD_LATTICE')
        lay.menu("VIEW3D_MT_math_art_patterns", icon='MESH_GRID')
        lay.menu("VIEW3D_MT_math_art_odds", icon='MESH_TORUS')
        lay.separator()
        if hasattr(bpy.types, 'OBJECT_OT_symmetric_sculpture_add'):
            lay.operator_menu_enum("object.symmetric_sculpture_add",
                                   "preset",
                                   text="Symmetric Sculpture "
                                        "(Experimental)",
                                   icon='MOD_MIRROR')
        lay.menu("VIEW3D_MT_math_art_styles", icon='MOD_SOLIDIFY')


_MENUS = (VIEW3D_MT_math_art_minimal, VIEW3D_MT_math_art_polyhedra,
          VIEW3D_MT_math_art_fractals, VIEW3D_MT_math_art_knots,
          VIEW3D_MT_math_art_weaves, VIEW3D_MT_math_art_patterns,
          VIEW3D_MT_math_art_odds, VIEW3D_MT_math_art_styles,
          VIEW3D_MT_math_art_add)


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
    for c in _MENUS:
        bpy.utils.register_class(c)
    bpy.types.VIEW3D_MT_add.append(_menu_func)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_menu_func)
    for c in reversed(_MENUS):
        bpy.utils.unregister_class(c)
    for m in reversed(_ACTIVE):
        try:
            m.unregister()
        except Exception as e:
            print(f"Math Art: unregister failed for "
                  f"{m.__name__}: {e}")
    _ACTIVE.clear()
