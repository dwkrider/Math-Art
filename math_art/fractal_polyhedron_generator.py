
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
    "author": "Math Art project (after George W. Hart)",
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


def _mat_mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def _mat_vec(A, v):
    return tuple(sum(A[i][k] * v[k] for k in range(3)) for i in range(3))


def _axis_rot(axis, ang):
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return ((c + x * x * C, x * y * C - z * s, x * z * C + y * s),
            (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
            (z * x * C - y * s, z * y * C + x * s, c + z * z * C))


_ID3 = ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))


def _euler_rot(rx, ry, rz):
    R = _axis_rot((1, 0, 0), math.radians(rx))
    R = _mat_mul(_axis_rot((0, 1, 0), math.radians(ry)), R)
    return _mat_mul(_axis_rot((0, 0, 1), math.radians(rz)), R)


def build_fractal(kind='CUBE', mode='VERTS', generations=3,
                  child_scale=0.5, spread=1.27, keep_parents=True,
                  push=0.0, twist=0.0, rot_x=0.0, rot_y=0.0, rot_z=0.0,
                  scale=1.0):
    """Returns (verts, faces, n_copies, face_gen). Each copy carries
    (origin, size, orientation, generation). Children sit at the
    anchors of their parent (scaled by spread, shifted radially by
    push), rotated by `twist` about the anchor direction plus an
    XYZ rotation -- all cumulative across generations."""
    V, F = seed_poly(kind)
    anchors = _anchors(V, F, mode)
    Re = _euler_rot(rot_x, rot_y, rot_z)
    tw = math.radians(twist)
    copies = [((0.0, 0.0, 0.0), 1.0, _ID3, 0)]
    all_copies = list(copies) if keep_parents else []
    for g in range(generations):
        nxt = []
        for (o, s, R, _gen) in copies:
            for a in anchors:
                al = math.sqrt(sum(x * x for x in a)) or 1.0
                adir = tuple(x / al for x in a)
                dir_w = _mat_vec(R, adir)
                pos = tuple(o[k]
                            + _mat_vec(R, a)[k] * s * spread
                            + dir_w[k] * push * s for k in range(3))
                Rc = R
                if abs(tw) > 1e-12:
                    Rc = _mat_mul(_axis_rot(dir_w, tw), Rc)
                Rc = _mat_mul(Rc, Re)
                nxt.append((pos, s * child_scale, Rc, g + 1))
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
    face_gen = []
    for (o, s, R, gen) in final:
        base = len(verts)
        for v in V:
            p = _mat_vec(R, v)
            verts.append(tuple((o[k] + p[k] * s) * scale
                               for k in range(3)))
        for f in F:
            faces.append([base + i for i in f])
            face_gen.append(gen)
    return verts, faces, len(final), face_gen


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
            default='CUBE')
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
            name="Spread", default=1.27, min=0.5, max=3.0,
            description="Distance multiplier from parent to children")
        keep_parents: BoolProperty(
            name="Keep Parents", default=True,
            description="Keep every generation (off = only the last, "
                        "e.g. the Sierpinski gasket)")
        push: FloatProperty(
            name="Push", default=0.0, min=-1.0, max=2.0,
            description="Extra shift of each child along its anchor "
                        "direction (in parent-size units)")
        twist: FloatProperty(
            name="Twist", default=0.0, min=-180.0, max=180.0,
            description="Rotation of each child about its anchor "
                        "direction (cumulative per generation)")
        rot_x: FloatProperty(name="Child Rotate X", default=0.0,
                             min=-180.0, max=180.0)
        rot_y: FloatProperty(name="Child Rotate Y", default=0.0,
                             min=-180.0, max=180.0)
        rot_z: FloatProperty(name="Child Rotate Z", default=0.0,
                             min=-180.0, max=180.0)
        coloring: EnumProperty(
            name="Coloring",
            items=[('GEN', "Per Generation",
                    "One material per generation (view with Material "
                    "Preview or Solid shading set to Material colour)"),
                   ('NONE', "None", "No materials")],
            default='GEN')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        _PALETTE = [(0.85, 0.82, 0.75), (0.90, 0.36, 0.23),
                    (0.27, 0.52, 0.79), (0.95, 0.77, 0.29),
                    (0.30, 0.69, 0.42), (0.62, 0.40, 0.75),
                    (0.25, 0.72, 0.72), (0.91, 0.56, 0.71)]

        @classmethod
        def _material_for(cls, g):
            name = f"Fractal Gen {g}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE[g % len(cls._PALETTE)]
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.55
            return mat

        def execute(self, context):
            try:
                verts, faces, nc, face_gen = build_fractal(
                    self.kind, self.mode, self.generations,
                    self.child_scale, self.spread, self.keep_parents,
                    self.push, self.twist, self.rot_x, self.rot_y,
                    self.rot_z, self.scale)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            me = bpy.data.meshes.new("FractalPolyhedron")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            if len(me.polygons) == len(faces):
                attr = me.attributes.new("generation", 'INT', 'FACE')
                attr.data.foreach_set('value', face_gen)
                if self.coloring == 'GEN':
                    gens = sorted(set(face_gen))
                    slot = {g: i for i, g in enumerate(gens)}
                    for g in gens:
                        me.materials.append(self._material_for(g))
                    me.polygons.foreach_set(
                        'material_index',
                        [slot[g] for g in face_gen])
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
                      'spread', 'keep_parents'):
                lay.prop(self, k)
            col = lay.column(align=True)
            for k in ('push', 'twist', 'rot_x', 'rot_y', 'rot_z'):
                col.prop(self, k)
            lay.prop(self, 'coloring')
            lay.prop(self, 'scale')

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
            v, f, n, fg = build_fractal(kind, mode, gen, cs, spread=1.0,
                                        keep_parents=keep, twist=15,
                                        rot_z=10)
            print(f"{kind} {mode} g{gen}: copies={n} verts={len(v)} "
                  f"gens={sorted(set(fg))}")
