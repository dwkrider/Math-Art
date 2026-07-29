
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


def _grid_rule(kind):
    """Which of the 3x3x3 (or 3x3x1) subcells survive one step."""
    keep = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                mid = (x == 1, y == 1, z == 1)
                if kind == 'MENGER':
                    ok = sum(mid) < 2
                elif kind == 'VICSEK':
                    ok = sum(mid) >= 2
                elif kind == 'CARPET':
                    ok = z == 0 and not (mid[0] and mid[1])
                elif kind == 'MOSELY':
                    ok = sum(mid) >= 1          # keep all but 8 corners
                elif kind == 'MOSELYL':
                    ok = 1 <= sum(mid) <= 2     # also drop body centre
                elif kind == 'CANTOR':
                    ok = sum(mid) == 0          # keep only the corners
                else:
                    raise ValueError(kind)
                if ok:
                    keep.append((x, y, z))
    return keep


def sponge_cells(kind, level):
    """Occupied unit cells of the level-n grid sponge, as integer
    coordinates in a 3^n grid."""
    rule = _grid_rule(kind)
    cells = [(0, 0, 0)]
    for _ in range(level):
        cells = [(3 * cx + dx, 3 * cy + dy, 3 * cz + dz)
                 for (cx, cy, cz) in cells
                 for (dx, dy, dz) in rule]
    return cells


_FACE_DIRS = (
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
)


def build_grid_sponge(kind, level, size=2.0):
    """Watertight exterior surface of a grid sponge: only faces
    between a solid cell and an empty neighbour are emitted, with
    shared vertices."""
    cells = sponge_cells(kind, level)
    occ = set(cells)
    n = 3 ** level
    s = size / n
    off = size / 2.0
    zoff = (size / 2.0 if kind != 'CARPET'
            else s / 2.0)          # carpet: one cell thick, centred
    verts = []
    vid = {}
    faces = []

    def vertex(ix, iy, iz):
        key = (ix, iy, iz)
        i = vid.get(key)
        if i is None:
            i = len(verts)
            vid[key] = i
            verts.append((ix * s - off, iy * s - off, iz * s - zoff))
        return i

    for (cx, cy, cz) in cells:
        for (d, corners) in _FACE_DIRS:
            if (cx + d[0], cy + d[1], cz + d[2]) in occ:
                continue
            faces.append([vertex(cx + a, cy + b, cz + c)
                          for (a, b, c) in corners])
    return verts, faces


_TETRA = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
_TETRA_F = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
_OCTA = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
         (0, 0, 1), (0, 0, -1)]
_OCTA_F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
           (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]


def build_corner_sponge(kind, level, size=2.0):
    """Sierpinski tetrahedron / octahedron: recursive half-scale
    copies at the vertices, meeting at points. One closed solid per
    smallest copy."""
    base_v, base_f = ((_TETRA, _TETRA_F) if kind == 'TETRA'
                      else (_OCTA, _OCTA_F))
    centres = [(0.0, 0.0, 0.0)]
    sc = size / 2.0
    for _ in range(level):
        sc /= 2.0
        centres = [(c[0] + v[0] * sc, c[1] + v[1] * sc,
                    c[2] + v[2] * sc)
                   for c in centres for v in base_v]
    verts = []
    faces = []
    for c in centres:
        base = len(verts)
        for v in base_v:
            verts.append((c[0] + v[0] * sc, c[1] + v[1] * sc,
                          c[2] + v[2] * sc))
        for f in base_f:
            faces.append([base + i for i in f])
    return verts, faces


GRID_KINDS = ('MENGER', 'VICSEK', 'CARPET', 'MOSELY', 'MOSELYL',
              'CANTOR')
MAX_LEVEL = {'MENGER': 4, 'VICSEK': 5, 'CARPET': 6,
             'MOSELY': 3, 'MOSELYL': 3, 'CANTOR': 4,
             'TETRA': 6, 'OCTA': 5}


def build_sponge(kind='MENGER', level=3, size=2.0):
    level = min(level, MAX_LEVEL[kind])
    if kind in GRID_KINDS:
        return build_grid_sponge(kind, level, size)
    return build_corner_sponge(kind, level, size)


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


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
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
