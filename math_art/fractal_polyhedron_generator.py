
# Fractal Polyhedron Generator for Blender
#
# Recursive clusters of Platonic solids, after the fractal-polyhedron
# construction on George W. Hart's pages: every generation places a
# scaled copy of the solid at each vertex, face or edge of the current
# copies. Vertex mode with scale 1/2 on a tetrahedron and parents
# removed is the classic Sierpinski tetrahedron; face mode grows
# spiky coral-like clusters.

bl_info = {
    "name": "Fractal Polyhedra",
    "author": "David Krider (Math Art project, after George W. Hart)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Fractal Polyhedron",
    "description": "Recursive Platonic solid clusters",
    "category": "Add Mesh",
}

import math

PHI = (1 + 5 ** 0.5) / 2


def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def _icosa_faces(V):
    n = len(V)
    emin = None
    d2 = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = sum((V[i][k] - V[j][k]) ** 2 for k in range(3))
            d2[(i, j)] = d
            emin = d if emin is None else min(emin, d)
    adj = {i: set() for i in range(n)}
    for (i, j), d in d2.items():
        if d < emin * 1.2:
            adj[i].add(j)
            adj[j].add(i)
    fs = set()
    for i in range(n):
        for j in adj[i]:
            for k in adj[i] & adj[j]:
                fs.add(tuple(sorted((i, j, k))))
    faces = []
    for f in fs:
        a, b, c = (V[i] for i in f)
        nx = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
              (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
              (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        cen = [(a[k] + b[k] + c[k]) / 3 for k in range(3)]
        if sum(nx[k] * cen[k] for k in range(3)) < 0:
            faces.append([f[0], f[2], f[1]])
        else:
            faces.append(list(f))
    return faces


def seed_poly(kind):
    if kind == 'TETRA':
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        F = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    elif kind == 'OCTA':
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
        F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    elif kind == 'CUBE':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    elif kind == 'ICOSA':
        V = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = _icosa_faces(V)
    elif kind == 'DODECA':
        IV = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                IV += [(0, a, b), (a, b, 0), (b, 0, a)]
        IF = _icosa_faces(IV)
        V = [tuple(sum(IV[i][k] for i in f) / 3 for k in range(3))
             for f in IF]
        F = []
        for vi in range(len(IV)):
            adj = [fi for fi, f in enumerate(IF) if vi in f]
            nrm = _unit(IV[vi])
            u0 = V[adj[0]]
            u = [u0[k] - sum(u0[j] * nrm[j] for j in range(3)) * nrm[k]
                 for k in range(3)]
            w = (nrm[1] * u[2] - nrm[2] * u[1],
                 nrm[2] * u[0] - nrm[0] * u[2],
                 nrm[0] * u[1] - nrm[1] * u[0])
            def ang(fi):
                d = V[fi]
                return math.atan2(sum(d[k] * w[k] for k in range(3)),
                                  sum(d[k] * u[k] for k in range(3)))
            F.append(sorted(adj, key=ang))
    else:
        raise ValueError(kind)
    return [list(v) for v in V], [list(f) for f in F]


def _anchors(V, F, mode):
    if mode == 'VERTS':
        return [tuple(v) for v in V]
    if mode == 'EDGES':
        seen = set()
        out = []
        for f in F:
            m = len(f)
            for i in range(m):
                a, b = f[i], f[(i + 1) % m]
                k = (min(a, b), max(a, b))
                if k not in seen:
                    seen.add(k)
                    out.append(tuple((V[a][c] + V[b][c]) / 2
                                     for c in range(3)))
        return out
    # FACES
    return [tuple(sum(V[i][c] for i in f) / len(f) for c in range(3))
            for f in F]


MAX_COPIES = 30000


def build_fractal(kind='TETRA', mode='VERTS', generations=3,
                  child_scale=0.5, spread=1.0, keep_parents=False,
                  scale=1.0):
    """Returns (verts, faces, n_copies). Each copy is (origin, size);
    children sit at anchors of their parent, scaled by child_scale."""
    V, F = seed_poly(kind)
    anchors = _anchors(V, F, mode)
    copies = [((0.0, 0.0, 0.0), 1.0)]
    all_copies = list(copies) if keep_parents else []
    for g in range(generations):
        nxt = []
        for (o, s) in copies:
            for a in anchors:
                nxt.append((tuple(o[k] + a[k] * s * spread
                                  for k in range(3)),
                            s * child_scale))
        copies = nxt
        if keep_parents:
            all_copies.extend(copies)
        if len(copies) * len(V) > MAX_COPIES * 4:
            raise ValueError(
                f"too many copies ({len(copies)}); lower generations")
    final = all_copies if keep_parents else copies
    if len(final) > MAX_COPIES:
        raise ValueError(f"too many copies ({len(final)})")
    verts = []
    faces = []
    for (o, s) in final:
        base = len(verts)
        for v in V:
            verts.append(tuple((o[k] + v[k] * s) * scale
                               for k in range(3)))
        for f in F:
            faces.append([base + i for i in f])
    return verts, faces, len(final)


try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, IntProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_fractal_poly_add(bpy.types.Operator):
        """Recursive Platonic solid cluster (vertex mode, scale 0.5,
        tetrahedron = the Sierpinski tetrahedron)"""
        bl_idname = "mesh.fractal_polyhedron_add"
        bl_label = "Fractal Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Solid",
            items=[('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", ""),
                   ('DODECA', "Dodecahedron", ""),
                   ('ICOSA', "Icosahedron", "")],
            default='TETRA')
        mode: EnumProperty(
            name="Anchors",
            items=[('VERTS', "Vertices", "children at vertices"),
                   ('FACES', "Faces", "children at face centres"),
                   ('EDGES', "Edges", "children at edge midpoints")],
            default='VERTS')
        generations: IntProperty(name="Generations", default=3,
                                 min=1, max=7)
        child_scale: FloatProperty(name="Child Scale", default=0.5,
                                   min=0.1, max=0.9)
        spread: FloatProperty(
            name="Spread", default=1.0, min=0.5, max=3.0,
            description="Distance multiplier from parent to children")
        keep_parents: BoolProperty(
            name="Keep Parents", default=False,
            description="Keep every generation (off = only the last, "
                        "e.g. the Sierpinski gasket)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            try:
                verts, faces, nc = build_fractal(
                    self.kind, self.mode, self.generations,
                    self.child_scale, self.spread, self.keep_parents,
                    self.scale)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            me = bpy.data.meshes.new("FractalPolyhedron")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new("FractalPolyhedron", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{nc} copies")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'mode', 'generations', 'child_scale',
                      'spread', 'keep_parents', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.fractal_polyhedron_add",
                             icon='OUTLINER_OB_POINTCLOUD')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_fractal_poly_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_fractal_poly_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for kind, mode, gen, cs, keep in (
                ('TETRA', 'VERTS', 3, 0.5, False),
                ('TETRA', 'VERTS', 4, 0.5, False),
                ('CUBE', 'FACES', 3, 0.45, True),
                ('ICOSA', 'VERTS', 2, 0.4, True)):
            v, f, n = build_fractal(kind, mode, gen, cs,
                                    keep_parents=keep)
            print(f"{kind} {mode} g{gen}: copies={n} verts={len(v)}")
