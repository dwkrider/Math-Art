
# Notable / Other Polyhedra for Blender
#
# A gallery of famous individual polyhedra that don't fit the parametric
# families: the Schonhardt polyhedron (the twisted octahedron that cannot be
# triangulated without extra vertices), Jessen's orthogonal icosahedron (a
# shaky polyhedron whose dihedral angles are all right angles), Durer's solid
# (the truncated rhombohedron from Melencolia I), the Bilinski dodecahedron
# (the "other" rhombic dodecahedron, of golden rhombi), and a polyhedral
# realization of Felix Klein's regular map {3,7}_8 on a genus-3 surface.
#
# References:
# - E. Schonhardt, "Ueber die Zerlegung von Dreieckspolyedern in
#   Tetraeder", Math. Ann. 98 (1928), 309-312.
# - B. Jessen, "Orthogonal icosahedra", Nordisk Mat. Tidskr. 15 (1967).
# - A. Durer, "Melencolia I" (1514); analysis: see Weitzel, Schreiber.
# - S. Bilinski, "Ueber die Rhombenisoeder", Glasnik Mat.-Fiz. Astr. 15
#   (1960), 251-263.
# - F. Klein, "Ueber die Transformation siebenter Ordnung der elliptischen
#   Functionen", Math. Ann. 14 (1878) (the quartic / map {3,7}_8);
#   polyhedral realization: E. Schulte & J. M. Wills, "A polyhedral
#   realization of Felix Klein's map {3,7}_8 on a Riemann surface of genus
#   3", J. London Math. Soc. 32 (1985), 539-547.  (The Klein coordinates
#   here are a construction on two homothetic truncated tetrahedra,
#   verified to carry the full 168 automorphisms of {3,7}_8.)

bl_info = {
    "name": "Notable Polyhedra",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Schonhardt, Jessen's orthogonal icosahedron, Durer's "
                   "solid, Bilinski dodecahedron, Klein's regular map",
    "category": "Add Mesh",
}

import math

PHI = (1 + 5 ** 0.5) / 2


def schonhardt(twist_deg=40.0):
    """Schonhardt polyhedron: two parallel triangles, the top twisted so the
    three long side diagonals become concave (non-convex; cannot be split
    into tetrahedra without a new vertex)."""
    tw = math.radians(twist_deg)
    A = [(math.cos(2 * math.pi * k / 3), math.sin(2 * math.pi * k / 3), -0.5)
         for k in range(3)]
    B = [(math.cos(2 * math.pi * k / 3 + tw),
          math.sin(2 * math.pi * k / 3 + tw), 0.5) for k in range(3)]
    V = A + B
    F = [[2, 1, 0], [3, 4, 5]]
    for i in range(3):
        a0, a1, b0, b1 = i, (i + 1) % 3, 3 + i, 3 + (i + 1) % 3
        F += [[a0, a1, b0], [a1, b1, b0]]      # concave split
    return V, F


GALLERY = {
    "JESSEN": {"name": "Jessen's Orthogonal Icosahedron",
               "V": [[-2, -1, 0], [-2, 1, 0], [-1, 0, -2], [-1, 0, 2],
                     [0, -2, -1], [0, -2, 1], [0, 2, -1], [0, 2, 1],
                     [1, 0, -2], [1, 0, 2], [2, -1, 0], [2, 1, 0]],
               "F": [[0, 2, 4], [5, 3, 0], [6, 2, 1], [1, 3, 7], [4, 8, 10],
                     [10, 9, 5], [11, 8, 6], [7, 9, 11], [3, 2, 0],
                     [0, 4, 10], [10, 5, 0], [1, 2, 3], [11, 6, 1],
                     [1, 7, 11], [6, 4, 2], [3, 5, 7], [4, 6, 8], [9, 7, 5],
                     [8, 9, 10], [11, 9, 8]]},
    "DURER": {"name": "Durer's Solid (Melencolia I)",
              "V": [[0.0, 0.419469524122, -0.64771662102],
                    [-0.363271264003, -0.209734762061, -0.64771662102],
                    [0.363271264003, -0.209734762061, -0.64771662102],
                    [0.0, 0.678715947274, -0.367200443531],
                    [-0.587785252292, -0.339357973637, -0.367200443531],
                    [0.587785252292, -0.339357973637, -0.367200443531],
                    [0.0, -0.678715947274, 0.367200443531],
                    [0.587785252292, 0.339357973637, 0.367200443531],
                    [-0.587785252292, 0.339357973637, 0.367200443531],
                    [0.0, -0.419469524122, 0.64771662102],
                    [0.363271264003, 0.209734762061, 0.64771662102],
                    [-0.363271264003, 0.209734762061, 0.64771662102]],
              "F": [[2, 1, 0], [9, 10, 11], [1, 4, 8, 3, 0], [9, 6, 5, 7, 10],
                    [2, 5, 6, 4, 1], [10, 7, 3, 8, 11], [0, 3, 7, 5, 2],
                    [11, 8, 4, 6, 9]]},
    "BILINSKI": {"name": "Bilinski Dodecahedron",
                 "V": [[0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                       [PHI, 0, PHI], [PHI, 0, -PHI], [-PHI, 0, PHI],
                       [-PHI, 0, -PHI], [PHI, 1, 0], [PHI, -1, 0],
                       [-PHI, 1, 0], [-PHI, -1, 0], [0, 0, PHI * PHI],
                       [0, 0, -PHI * PHI]],
                 "F": [[10, 0, 8, 1], [8, 0, 12, 4], [12, 0, 10, 6],
                       [13, 1, 8, 5], [10, 1, 13, 7], [9, 2, 11, 3],
                       [12, 2, 9, 4], [11, 2, 12, 6], [9, 3, 13, 5],
                       [13, 3, 11, 7], [8, 4, 9, 5], [11, 6, 10, 7]]},
    "KLEIN": {"name": "Klein Regular Map {3,7}_8 (genus 3)",
              "V": [[1, 1, 3], [1, -3, -1], [-3, 1, -1], [1, 3, 1],
                    [-1.5, -0.5, 0.5], [-1, -1, 3], [1.5, 0.5, 0.5],
                    [1.5, -0.5, -0.5], [0.5, -1.5, -0.5], [3, 1, 1],
                    [1, -1, -3], [-0.5, -0.5, 1.5], [0.5, -0.5, -1.5],
                    [-1.5, 0.5, -0.5], [3, -1, -1], [-1, -3, 1],
                    [-0.5, 0.5, -1.5], [-3, -1, 1], [0.5, 0.5, 1.5],
                    [-0.5, -1.5, 0.5], [0.5, 1.5, 0.5], [-1, 1, -3],
                    [-0.5, 1.5, -0.5], [-1, 3, -1]],
              "F": [[0, 9, 3], [0, 3, 4], [0, 4, 5], [0, 5, 6], [0, 6, 7],
                    [0, 7, 8], [0, 8, 9], [1, 13, 10], [1, 14, 11],
                    [1, 15, 12], [1, 16, 13], [1, 10, 14], [1, 11, 15],
                    [1, 12, 16], [2, 19, 17], [2, 20, 18], [2, 21, 19],
                    [2, 22, 20], [2, 23, 21], [2, 17, 22], [2, 18, 23],
                    [3, 11, 4], [4, 13, 5], [5, 15, 6], [6, 10, 7],
                    [7, 12, 8], [8, 14, 9], [9, 16, 3], [3, 18, 11],
                    [4, 21, 13], [5, 17, 15], [6, 20, 10], [7, 23, 12],
                    [8, 19, 14], [9, 22, 16], [3, 23, 18], [4, 19, 21],
                    [5, 22, 17], [6, 18, 20], [7, 21, 23], [8, 17, 19],
                    [9, 20, 22], [3, 16, 23], [4, 11, 19], [5, 13, 22],
                    [6, 15, 18], [7, 10, 21], [8, 12, 17], [9, 14, 20],
                    [10, 20, 14], [11, 18, 15], [12, 23, 16], [13, 21, 10],
                    [14, 19, 11], [15, 17, 12], [16, 22, 13]]},
}

ITEMS = [("SCHONHARDT", "Schonhardt Polyhedron", "twisted octahedron"),
         ("JESSEN", "Jessen's Orthogonal Icosahedron", "all right angles"),
         ("DURER", "Durer's Solid", "Melencolia I"),
         ("BILINSKI", "Bilinski Dodecahedron", "golden rhombi"),
         ("KLEIN", "Klein Regular Map {3,7} (genus 3)", "")]


def build(kind):
    if kind == 'SCHONHARDT':
        V, F = schonhardt()
    else:
        S = GALLERY[kind]
        V = [tuple(float(c) for c in v) for v in S["V"]]
        F = [list(f) for f in S["F"]]
    cen = [sum(v[i] for v in V) / len(V) for i in range(3)]
    V = [tuple(v[i] - cen[i] for i in range(3)) for v in V]
    mx = max((abs(c) for v in V for c in v), default=1.0) or 1.0
    return [tuple(c / mx for c in v) for v in V], F


def _self_test():
    want = {'SCHONHARDT': (6, 12, 8, 2), 'JESSEN': (12, 30, 20, 2),
            'DURER': (12, 18, 8, 2), 'BILINSKI': (14, 24, 12, 2),
            'KLEIN': (24, 84, 56, -4)}
    for kind, _lbl, _d in ITEMS:
        V, F = build(kind)
        E = {}
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                k = (min(a, b), max(a, b))
                E[k] = E.get(k, 0) + 1
        chi = len(V) - len(E) + len(F)
        e2 = all(v == 2 for v in E.values())
        print(f"{kind:11s} V={len(V):2d} E={len(E):2d} F={len(F):2d} "
              f"chi={chi:3d} edge-in-2={e2}  want{want[kind]}")


try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_notable_polyhedron_add(bpy.types.Operator):
        """Add a notable individual polyhedron (Schonhardt, Jessen, Durer,
        Bilinski, or Klein's genus-3 regular map)"""
        bl_idname = "mesh.notable_polyhedron_add"
        bl_label = "Notable Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(name="Solid", items=ITEMS)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            V, F = build(self.solid)
            label = dict((i[0], i[1]) for i in ITEMS)[self.solid]
            me = bpy.data.meshes.new(label)
            me.from_pydata([tuple(c * self.scale for c in v) for v in V],
                           [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new(label, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{label}: V={len(V)} F={len(F)}")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.notable_polyhedron_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_notable_polyhedron_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_notable_polyhedron_add)


if __name__ == "__main__" and not _IN_BLENDER:
    _self_test()
