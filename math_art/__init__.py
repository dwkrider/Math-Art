
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
    'platonic_twist_generator',
    'fractal_polyhedron_generator',
    'symmetrohedron_generator',
    'twisted_torus_generator',
    'polytope4d_generator',
    'tangle_generator',
    'symmetric_sculpture_generator',
    'leonardo_style',
    'sponge_generator',
    'space_curve_generator',
    'oloid_generator',
    'prime_knot_generator',
    'regular_solids_generator',
    'attractor_generator',
    'dual_helix_generator',
    'stellated_weave_generator',
    'stereographic_projection_generator',
    'hyperbolic_honeycomb_generator',
    'algebraic_surface_generator',
    'spacefill_generator',
    'link_generator',
    'topological_surface_generator',
    'hyperbolic_tiling_generator',
    'geodesic_generator',
    'curvature_color',
    'orbifold_sphere_generator',
    'fractal_tree_generator',
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
        lay.operator_menu_enum("mesh.scherk_collins_add", "preset",
                               text="Scherk-Collins Sculpture",
                               icon='MESH_TORUS')
        _op(lay, "mesh.parametric_minimal_add",
            icon='SURFACE_NSPHERE')
        _op(lay, "mesh.tpms_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.minimal_knot_span_add", icon='MESH_TORUS')
        _op(lay, "object.minimal_span", icon='OUTLINER_OB_SURFACE')
        _op(lay, "mesh.seifert_surface_add", icon='MOD_SIMPLIFY')
        _op(lay, "mesh.algebraic_surface_add",
            icon='SURFACE_NSURFACE')
        _op(lay, "mesh.topological_surface_add", icon='MESH_TORUS')


class VIEW3D_MT_math_art_polyhedra(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_polyhedra"
    bl_label = "Polyhedra"

    def draw(self, context):
        lay = self.layout
        _op(lay, "mesh.regular_solid_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.conway_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.zonohedron_add", icon='MESH_UVSPHERE')
        _op(lay, "mesh.waterman_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.symmetrohedron_add", icon='MESH_ICOSPHERE')
        _op(lay, "mesh.polytope4d_add", icon='MESH_CUBE')
        _op(lay, "mesh.hyperbolic_honeycomb_add", icon='META_BALL')
        _op(lay, "mesh.spacefill_add", icon='SNAP_VOLUME')
        _op(lay, "mesh.geodesic_add", icon='MESH_UVSPHERE')


class VIEW3D_MT_math_art_fractals(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_fractals"
    bl_label = "Fractals"

    def draw(self, context):
        lay = self.layout
        _op(lay, "mesh.sponge_add", icon='MESH_CUBE')
        _op(lay, "mesh.fractal_polyhedron_add",
            icon='OUTLINER_OB_POINTCLOUD')
        _op(lay, "curve.space_filling_add", icon='CURVE_DATA')
        _op(lay, "curve.fractal_tree_add", icon='GRAPH')


class VIEW3D_MT_math_art_knots(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_knots"
    bl_label = "Knots & Curves"

    def draw(self, context):
        lay = self.layout
        _op(lay, "curve.prime_knot_add", icon='FORCE_VORTEX')
        _op(lay, "curve.math_link_add", icon='LINKED')
        if hasattr(bpy.types, 'CURVE_OT_attractor_add'):
            lay.operator_menu_enum("curve.attractor_add", "preset",
                                   text="Strange Attractor",
                                   icon='RNDCURVE')
        _op(lay, "curve.dual_helix_add", icon='MOD_SCREW')


class VIEW3D_MT_math_art_weaves(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_weaves"
    bl_label = "Weaves & Tangles"

    def draw(self, context):
        lay = self.layout
        _op(lay, "mesh.polylinks_add", icon='MESH_CIRCLE')
        _op(lay, "mesh.tangle_add", icon='MOD_BOOLEAN')
        _op(lay, "mesh.poly_weave_add", icon='MOD_LATTICE')
        _op(lay, "mesh.rotegrity_add", icon='SPHERE')
        _op(lay, "mesh.stellated_weave_add", icon='MOD_LATTICE')


class VIEW3D_MT_math_art_odds(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_odds"
    bl_label = "Odds & Ends"

    def draw(self, context):
        lay = self.layout
        _op(lay, "mesh.platonic_twist_add", icon='MOD_SCREW')
        _op(lay, "mesh.twisted_torus_add", icon='MESH_TORUS')
        _op(lay, "mesh.oloid_add", icon='MESH_CAPSULE')
        _op(lay, "mesh.stereographic_add", icon='LIGHT_POINT')
        _op(lay, "mesh.hyperbolic_tiling_add", icon='MESH_CIRCLE')
        _op(lay, "mesh.orbifold_sphere_add", icon='MOD_MIRROR')


class VIEW3D_MT_math_art_styles(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_styles"
    bl_label = "Styles"

    def draw(self, context):
        lay = self.layout
        _op(lay, "object.leonardo_add", icon='MESH_ICOSPHERE')
        _op(lay, "object.curvature_color_add",
            icon='COLORSET_01_VEC')


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
        lay.menu("VIEW3D_MT_math_art_odds", icon='MESH_TORUS')
        lay.separator()
        if hasattr(bpy.types, 'OBJECT_OT_symmetric_sculpture_add'):
            lay.operator_menu_enum("object.symmetric_sculpture_add",
                                   "preset",
                                   text="Symmetric Sculpture",
                                   icon='MOD_MIRROR')
        lay.menu("VIEW3D_MT_math_art_styles", icon='MOD_SOLIDIFY')


_MENUS = (VIEW3D_MT_math_art_minimal, VIEW3D_MT_math_art_polyhedra,
          VIEW3D_MT_math_art_fractals, VIEW3D_MT_math_art_knots,
          VIEW3D_MT_math_art_weaves, VIEW3D_MT_math_art_odds,
          VIEW3D_MT_math_art_styles, VIEW3D_MT_math_art_add)


def _menu_func(self, context):
    self.layout.separator()
    self.layout.menu("VIEW3D_MT_math_art_add", icon='FUND')


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
