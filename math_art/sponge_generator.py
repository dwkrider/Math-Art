
# Fractal Sponge Generator for Blender
#
# Subtractive volume fractals: the Menger sponge and its relatives
# built by recursively keeping a subset of grid subcells, plus the
# point-contact Sierpinski solids. Grid sponges emit only exterior
# faces (faces between a solid cell and an empty neighbour), so the
# result is a single watertight surface even at high levels.
#
#   MENGER    3x3x3, drop cells sharing 2+ centre coordinates (20 kept)
#   VICSEK    3x3x3, keep the centre + 6 face neighbours (3D plus sign)
#   CARPET    Sierpinski carpet: 3x3 in the plane, one cube thick
#   MOSELY    Mosely snowflake: 3x3x3, drop the 8 corners (19 kept)
#   MOSELYL   Mosely snowflake (light): drop the 8 corners AND the
#             body centre (18 kept), leaving a hollow at each stage
#   CANTOR    Cantor dust: 3x3x3, keep only the 8 corners (2^3); the
#             copies never touch, so it is a cloud of separate cubes
#   TETRA     Sierpinski tetrahedron: 4 half-scale copies at vertices
#   OCTA      Sierpinski octahedron: 6 half-scale copies at vertices
#
# References:
# - Menger sponge: Karl Menger, "Allgemeine Raeume und Cartesische
#   Raeume", Proc. Akad. Wetensch. Amsterdam 29, 1926, pp. 1125-1128.
# - Sierpinski carpet (and its 3D tetra/octa analogues): Waclaw
#   Sierpinski, C. R. Acad. Sci. Paris 162, 1916, pp. 629-632.
# - Vicsek fractal: Tamas Vicsek, "Fractal models for diffusion
#   controlled aggregation", J. Phys. A 16, 1983, pp. L647-L652.
# - Mosely snowflake: named for Jeannine Mosely (of business-card
#   Menger-sponge fame); the Sierpinski-Menger snowflake family is
#   surveyed in M. Kalinski, "On the variations of the Sierpinski and
#   Menger sponges and the Mosely snowflake" (2017).
# - Cantor dust: the 3-fold Cartesian product of Georg Cantor's set,
#   "Ueber unendliche, lineare Punktmannigfaltigkeiten V", Math. Ann.
#   21, 1883, pp. 545-591.
# - Self-similar dimension: Benoit B. Mandelbrot, "The Fractal
#   Geometry of Nature", W. H. Freeman, 1982.

bl_info = {
    "name": "Fractal Sponges",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Fractal Sponge",
    "description": "Menger sponge, Vicsek fractal, Sierpinski solids",
    "category": "Add Mesh",
}


_FACE_DIRS = (
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
)
# The mathematics lives in the sibling `ifs` engine package;
# this module is the Blender layer over it.
try:
    from .ifs.menger import (MAX_LEVEL, build_corner_sponge,
                             build_grid_sponge, build_sponge, sponge_cells)
except ImportError:  # flat import outside the package
    from ifs.menger import (MAX_LEVEL, build_corner_sponge, build_grid_sponge,
                            build_sponge, sponge_cells)



try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_sponge_add(bpy.types.Operator):
        """Add a fractal sponge (Menger, Vicsek, Sierpinski...)"""
        bl_idname = "mesh.sponge_add"
        bl_label = "Fractal Sponge"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Fractal",
            items=[('MENGER', "Menger Sponge",
                    "3x3x3, centre and face-centre cells removed"),
                   ('TETRA', "Sierpinski Tetrahedron",
                    "4 half-scale copies at the corners"),
                   ('OCTA', "Sierpinski Octahedron",
                    "6 half-scale copies at the corners"),
                   ('VICSEK', "Vicsek Fractal",
                    "3D plus-sign: centre + 6 face neighbours"),
                   ('CARPET', "Sierpinski Carpet",
                    "The flat 3x3 carpet, one cell thick"),
                   ('MOSELY', "Mosely Snowflake",
                    "3x3x3 with the 8 corners removed (19 kept)"),
                   ('MOSELYL', "Mosely Snowflake (light)",
                    "Corners and body centre removed (18 kept, "
                    "hollow)"),
                   ('CANTOR', "Cantor Dust",
                    "Only the 8 corners kept: a cloud of separate "
                    "cubes")],
            default='MENGER')
        level: IntProperty(
            name="Level", default=3, min=0, max=6,
            description="Recursion depth (grid sponges are capped "
                        "per kind to keep meshes manageable)")
        size: FloatProperty(name="Size", default=2.0, min=0.05,
                            max=100.0)

        def execute(self, context):
            lv = min(self.level, MAX_LEVEL[self.kind])
            if lv != self.level:
                self.report({'WARNING'},
                            f"level capped at {lv} for this kind")
            verts, faces = build_sponge(self.kind, lv, self.size)
            me = bpy.data.meshes.new("Sponge")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new(f"Sponge {self.kind.title()}",
                                       me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"level {lv}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'level', 'size'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.sponge_add", icon='MESH_CUBE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_sponge_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_sponge_add)


def _selftest():
    # cell counts follow the fractal's replication factor
    for kind, factor in (('MENGER', 20), ('VICSEK', 7),
                         ('CARPET', 8), ('MOSELY', 19),
                         ('MOSELYL', 18), ('CANTOR', 8)):
        for lv in (1, 2, 3):
            cells = sponge_cells(kind, lv)
            ok = len(cells) == factor ** lv
            print(f"{kind} L{lv}: cells={len(cells)}"
                  f"({factor ** lv}) {'OK' if ok else 'BAD'}")
    # watertightness: every edge shared by exactly two faces
    from collections import Counter
    for kind in ('MENGER', 'VICSEK', 'CARPET', 'MOSELY',
                 'MOSELYL', 'CANTOR'):
        v, f = build_grid_sponge(kind, 2)
        cnt = Counter()
        for fc in f:
            for i in range(len(fc)):
                a, b = fc[i], fc[(i + 1) % len(fc)]
                cnt[(min(a, b), max(a, b))] += 1
        man = all(c == 2 for c in cnt.values())
        print(f"{kind} L2 surface: verts={len(v)} faces={len(f)} "
              f"manifold={man} {'OK' if man else 'BAD'}")
    for kind, copies, bf in (('TETRA', 4, 4), ('OCTA', 6, 8)):
        v, f = build_corner_sponge(kind, 3)
        ok = len(f) == bf * copies ** 3
        print(f"{kind} L3: faces={len(f)}({bf * copies ** 3}) "
              f"{'OK' if ok else 'BAD'}")
