
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

import bpy

_MODULES = (scherk_collins_generator, minimal_surface_toolkit,
            seifert_surface_generator, conway_operators,
            zonohedra_generator, waterman_generator,
            rotegrity_generator, weave_generator,
            polylinks_generator, platonic_twist_generator,
            fractal_polyhedron_generator, symmetrohedron_generator,
            twisted_torus_generator, polytope4d_generator)


class VIEW3D_MT_math_art_add(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_math_art_add"
    bl_label = "Math Art"

    def draw(self, context):
        lay = self.layout
        lay.operator_menu_enum("mesh.scherk_collins_add", "preset",
                               text="Scherk-Collins Sculpture",
                               icon='MESH_TORUS')
        lay.separator()
        lay.operator("mesh.parametric_minimal_add",
                     icon='SURFACE_NSPHERE')
        lay.operator("mesh.tpms_add", icon='MESH_ICOSPHERE')
        lay.operator("mesh.minimal_knot_span_add", icon='MESH_TORUS')
        lay.operator("object.minimal_span", icon='OUTLINER_OB_SURFACE')
        lay.separator()
        lay.operator("mesh.seifert_surface_add", icon='MOD_SIMPLIFY')
        lay.separator()
        lay.operator("mesh.conway_add", icon='MESH_ICOSPHERE')
        lay.operator("mesh.zonohedron_add", icon='MESH_UVSPHERE')
        lay.operator("mesh.waterman_add", icon='MESH_ICOSPHERE')
        lay.operator("mesh.rotegrity_add", icon='SPHERE')
        lay.operator("mesh.poly_weave_add", icon='MOD_LATTICE')
        lay.separator()
        lay.operator("mesh.polylinks_add", icon='MESH_CIRCLE')
        lay.operator("mesh.platonic_twist_add", icon='MOD_SCREW')
        lay.operator("mesh.fractal_polyhedron_add",
                     icon='OUTLINER_OB_POINTCLOUD')
        lay.operator("mesh.symmetrohedron_add", icon='MESH_ICOSPHERE')
        lay.operator("mesh.twisted_torus_add", icon='MESH_TORUS')
        lay.operator("mesh.polytope4d_add", icon='MESH_CUBE')


def _menu_func(self, context):
    self.layout.separator()
    self.layout.menu("VIEW3D_MT_math_art_add", icon='FUND')


def register():
    for m in _MODULES:
        m.ADD_MENU = False       # entries live in the Math Art submenu
        m.register()
    bpy.utils.register_class(VIEW3D_MT_math_art_add)
    bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)


def unregister():
    bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
    bpy.utils.unregister_class(VIEW3D_MT_math_art_add)
    for m in reversed(_MODULES):
        m.unregister()
