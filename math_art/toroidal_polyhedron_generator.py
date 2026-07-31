
# Toroidal Polyhedra for Blender
#
# Genus-1 (toroidal) polyhedra.  The Csaszar polyhedron is the unique
# (besides the tetrahedron) polyhedron with no diagonals: 7 vertices, 21
# edges forming the complete graph K7, and 14 triangular faces embedded on
# a torus.  Its dual, the Szilassi polyhedron, has 7 hexagonal faces every
# pair of which shares an edge -- the toroidal analogue of the tetrahedron.
# Both are stored from their original published coordinates and centred /
# fit to a 2 m cube on build.
#
# References:
# - Akos Csaszar, "A polyhedron without diagonals", Acta Sci. Math.
#   Szeged 13 (1949-50), 140-142.
# - Lajos Szilassi, "Regular toroids", Structural Topology 13 (1986),
#   69-80; and "On three classes of regular toroids".
# - B. M. Stewart, "Adventures Among the Toroids" (1970/1980), for the
#   toroidal-polyhedron tradition.

bl_info = {
    "name": "Toroidal Polyhedra",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Polyhedra",
    "description": "Toroidal (genus-1) polyhedra: the Csaszar and "
                   "Szilassi polyhedra",
    "category": "Add Mesh",
}

import math

# Original published coordinates (Szilassi's tables); not unit-scaled.
TOROIDS = {
    "CSASZAR": {
        "name": "Csaszar Polyhedron",
        "V": [(3.0, -3.0, 0.0), (3.0, 3.0, 1.0), (1.0, 2.0, 3.0),
              (-1.0, -2.0, 3.0), (-3.0, -3.0, 1.0), (-3.0, 3.0, 0.0),
              (0.0, 0.0, 15.0)],
        "F": [[0, 5, 1], [0, 1, 3], [4, 1, 2], [3, 2, 0], [1, 5, 6],
              [2, 1, 6], [0, 2, 6], [5, 0, 4], [5, 4, 2], [1, 4, 3],
              [2, 3, 5], [4, 0, 6], [3, 4, 6], [5, 3, 6]],
    },
    "SZILASSI": {
        "name": "Szilassi Polyhedron",
        "V": [(-12.0, 0.0, 12.0), (-2.0, 5.0, -8.0), (0.0, -12.6, -12.0),
              (3.75, 3.75, -3.0), (-7.0, -2.5, 2.0), (-7.0, 0.0, 2.0),
              (-4.5, 2.5, 2.0), (12.0, 0.0, 12.0), (2.0, -5.0, -8.0),
              (0.0, 12.6, -12.0), (-3.75, -3.75, -3.0), (7.0, 2.5, 2.0),
              (7.0, 0.0, 2.0), (4.5, -2.5, 2.0)],
        "F": [[7, 11, 6, 3, 1, 0], [0, 1, 9, 2, 5, 4], [8, 10, 3, 6, 5, 2],
              [1, 3, 10, 13, 12, 9], [9, 12, 11, 7, 8, 2],
              [0, 4, 13, 10, 8, 7], [4, 5, 6, 11, 12, 13]],
    },
}

TOROID_ITEMS = [("CSASZAR", "Csaszar Polyhedron", "7 vertices, 14 "
                 "triangles, K7 (no diagonals)"),
                ("SZILASSI", "Szilassi Polyhedron", "7 hexagons, every "
                 "pair sharing an edge (dual of Csaszar)")]


def build_toroid(kind):
    """Vertices/faces of the toroid, centred and fit to a 2 m cube."""
    S = TOROIDS[kind]
    V = [list(v) for v in S["V"]]
    c = [sum(v[i] for v in V) / len(V) for i in range(3)]
    V = [[v[i] - c[i] for i in range(3)] for v in V]
    mx = max(abs(x) for v in V for x in v) or 1.0
    V = [tuple(x / mx for x in v) for v in V]
    return V, [list(f) for f in S["F"]]


def _self_test():
    for kind, S in TOROIDS.items():
        V, F = build_toroid(kind)
        E = {}
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                k = (min(a, b), max(a, b))
                E[k] = E.get(k, 0) + 1
        chi = len(V) - len(E) + len(F)
        e2 = all(v == 2 for v in E.values())
        # planarity
        maxpl = 0.0
        for f in F:
            pts = [V[i] for i in f]
            cen = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
            nx = [0.0, 0.0, 0.0]
            for i in range(len(f)):
                p, q = pts[i], pts[(i + 1) % len(f)]
                nx[0] += (p[1] - q[1]) * (p[2] + q[2])
                nx[1] += (p[2] - q[2]) * (p[0] + q[0])
                nx[2] += (p[0] - q[0]) * (p[1] + q[1])
            ln = math.sqrt(sum(x * x for x in nx)) or 1.0
            nx = [x / ln for x in nx]
            for p in pts:
                maxpl = max(maxpl, abs(sum(nx[i] * (p[i] - cen[i])
                                          for i in range(3))))
        print(f"{kind:9s} V={len(V):2d} E={len(E):2d} F={len(F):2d} "
              f"chi={chi} edge-in-2={e2} planar={maxpl:.1e}")


try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_toroidal_polyhedron_add(bpy.types.Operator):
        """Add a toroidal (genus-1) polyhedron: the Csaszar polyhedron
        (no diagonals) or its dual the Szilassi polyhedron"""
        bl_idname = "mesh.toroidal_polyhedron_add"
        bl_label = "Toroidal Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(name="Solid", items=TOROID_ITEMS)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            V, F = build_toroid(self.solid)
            me = bpy.data.meshes.new(TOROIDS[self.solid]["name"])
            me.from_pydata([tuple(c * self.scale for c in v) for v in V],
                           [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new(TOROIDS[self.solid]["name"], me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{TOROIDS[self.solid]['name']}: "
                        f"V={len(V)} F={len(F)} (genus 1)")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.toroidal_polyhedron_add",
                             icon='MESH_TORUS')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_toroidal_polyhedron_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_toroidal_polyhedron_add)


if __name__ == "__main__" and not _IN_BLENDER:
    _self_test()
