
# Math Art -- a Blender extension bundling mathematical sculpture
# generators. Each module is also usable on its own as a legacy
# single-file add-on; installed as an extension they share one
# "Math Art" submenu under Add > Mesh.
#
#   Scherk-Collins sculptures      (after Sequin's Sculpture Generator I)
#   Minimal surfaces               (parametric, TPMS, Plateau solver)
#   Seifert surfaces               (after van Wijk & Cohen's SeifertView)
#   Conway polyhedron notation     |
#   Zonohedra & spirallohedra      |
#   Waterman polyhedra             |  after Adrian Rossiter's Antiprism
#   Rotegrity spheres              |
#   Polyhedral weaves              |

from . import scherk_collins_generator
from . import minimal_surface_toolkit
from . import seifert_surface_generator
from . import conway_operators
from . import zonohedra_generator
from . import waterman_generator
from . import rotegrity_generator
from . import weave_generator
from . import polylinks_generator
from . import platonic_twist_generator
from . import fractal_polyhedron_generator
from . import symmetrohedron_generator
from . import twisted_torus_generator
from . import polytope4d_generator
from . import tangle_generator
from . import symmetric_sculpture_generator
from . import leonardo_style
from . import sponge_generator
from . import space_curve_generator
from . import oloid_generator
from . import prime_knot_generator
from . import regular_solids_generator
from . import dual_helix_generator
from . import attractor_generator

import bpy

_MODULES = (scherk_collins_generator, minimal_surface_toolkit,
            seifert_surface_generator, conway_operators,
            zonohedra_generator, waterman_generator,
            rotegrity_generator, weave_generator,
            polylinks_generator, platonic_twist_generator,
            fractal_polyhedron_generator, symmetrohedron_generator,
            twisted_torus_generator, polytope4d_generator,
            tangle_generator, symmetric_sculpture_generator,
            leonardo_style, sponge_generator, space_curve_generator,
            oloid_generator, prime_knot_generator,
            regular_solids_generator, attractor_generator,
            dual_helix_generator)


class VIEW3D_MT_math_art_minimal(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_minimal"
    bl_label = "Minimal Surfaces"

    def draw(self, context):
        lay = self.layout
        lay.operator_menu_enum("mesh.scherk_collins_add", "preset",
                               text="Scherk-Collins Sculpture",
                               icon='MESH_TORUS')
        lay.operator("mesh.parametric_minimal_add",
                     icon='SURFACE_NSPHERE')
        lay.operator("mesh.tpms_add", icon='MESH_ICOSPHERE')
        lay.operator("mesh.minimal_knot_span_add", icon='MESH_TORUS')
        lay.operator("object.minimal_span",
                     icon='OUTLINER_OB_SURFACE')
        lay.operator("mesh.seifert_surface_add", icon='MOD_SIMPLIFY')


class VIEW3D_MT_math_art_polyhedra(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_polyhedra"
    bl_label = "Polyhedra"

    def draw(self, context):
        lay = self.layout
        lay.operator("mesh.regular_solid_add", icon='MESH_ICOSPHERE')
        lay.operator("mesh.conway_add", icon='MESH_ICOSPHERE')
        lay.operator("mesh.zonohedron_add", icon='MESH_UVSPHERE')
        lay.operator("mesh.waterman_add", icon='MESH_ICOSPHERE')
        lay.operator("mesh.symmetrohedron_add",
                     icon='MESH_ICOSPHERE')
        lay.operator("mesh.polytope4d_add", icon='MESH_CUBE')


class VIEW3D_MT_math_art_fractals(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_fractals"
    bl_label = "Fractals"

    def draw(self, context):
        lay = self.layout
        lay.operator("mesh.sponge_add", icon='MESH_CUBE')
        lay.operator("mesh.fractal_polyhedron_add",
                     icon='OUTLINER_OB_POINTCLOUD')
        lay.operator("curve.space_filling_add", icon='CURVE_DATA')


class VIEW3D_MT_math_art_knots(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_knots"
    bl_label = "Knots && Curves"

    def draw(self, context):
        lay = self.layout
        lay.operator("curve.prime_knot_add", icon='FORCE_VORTEX')
        lay.operator_menu_enum("curve.attractor_add", "preset",
                               text="Strange Attractor",
                               icon='RNDCURVE')
        lay.operator("curve.dual_helix_add", icon='MOD_SCREW')


class VIEW3D_MT_math_art_weaves(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_weaves"
    bl_label = "Weaves && Tangles"

    def draw(self, context):
        lay = self.layout
        lay.operator("mesh.polylinks_add", icon='MESH_CIRCLE')
        lay.operator("mesh.tangle_add", icon='MOD_BOOLEAN')
        lay.operator("mesh.poly_weave_add", icon='MOD_LATTICE')
        lay.operator("mesh.rotegrity_add", icon='SPHERE')


class VIEW3D_MT_math_art_sculpture(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_sculpture"
    bl_label = "Sculpture"

    def draw(self, context):
        lay = self.layout
        lay.operator("mesh.platonic_twist_add", icon='MOD_SCREW')
        lay.operator("mesh.twisted_torus_add", icon='MESH_TORUS')
        lay.operator("mesh.oloid_add", icon='MESH_CAPSULE')
        lay.operator_menu_enum("object.symmetric_sculpture_add",
                               "preset", text="Symmetric Sculpture",
                               icon='MOD_MIRROR')
        lay.operator("object.leonardo_add", icon='MESH_ICOSPHERE')


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
        lay.menu("VIEW3D_MT_math_art_sculpture", icon='MOD_MIRROR')


_MENUS = (VIEW3D_MT_math_art_minimal, VIEW3D_MT_math_art_polyhedra,
          VIEW3D_MT_math_art_fractals, VIEW3D_MT_math_art_knots,
          VIEW3D_MT_math_art_weaves, VIEW3D_MT_math_art_sculpture,
          VIEW3D_MT_math_art_add)


def _menu_func(self, context):
    self.layout.separator()
    self.layout.menu("VIEW3D_MT_math_art_add", icon='FUND')


def register():
    for m in _MODULES:
        m.ADD_MENU = False       # entries live in the Math Art submenu
        m.register()
    for c in _MENUS:
        bpy.utils.register_class(c)
    bpy.types.VIEW3D_MT_add.append(_menu_func)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_menu_func)
    for c in reversed(_MENUS):
        bpy.utils.unregister_class(c)
    for m in reversed(_MODULES):
        m.unregister()
