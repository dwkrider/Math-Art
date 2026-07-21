
# Polyhedral Weave Generator for Blender
#
# Basket-weave spheres after Antiprism's `poly_weave`: the weave strands
# are the straight-ahead circuits of the seed polyhedron's medial graph
# (the 4-valent graph on edge midpoints). Each circuit becomes a closed
# ribbon that alternates over/under at every crossing -- the classic
# woven-ball construction (a cube gives the 4-strand hexagonal weave,
# an icosahedron the 6-strand decagonal weave, geodesic spheres give
# dense triaxial weaves).

bl_info = {
    "name": "Polyhedral Weave Generator",
    "author": "David Krider (Math Art project, after Antiprism's poly_weave)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Polyhedral Weave",
    "description": "Woven-strand spheres from polyhedra",
    "category": "Add Mesh",
}

import math
from math import sin, cos

PHI = (1 + 5 ** 0.5) / 2


# ---- seeds (self-contained copy of the rotegrity helpers) ---------------

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
    else:
        raise ValueError(kind)
    return [_unit(v) for v in V], [list(f) for f in F]


def geodesic(V, F, freq):
    if freq <= 1:
        return V, F
    verts = list(V)
    key = {}
    faces = []

    def vid(p):
        k = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if k not in key:
            key[k] = len(verts)
            verts.append(p)
        return key[k]

    for f in F:
        A, B, C = (V[i] for i in f)
        grid = {}
        for i in range(freq + 1):
            for j in range(freq + 1 - i):
                k = freq - i - j
                grid[(i, j)] = vid(_unit(tuple(
                    (i * A[c] + j * B[c] + k * C[c]) / freq
                    for c in range(3))))
        for i in range(freq):
            for j in range(freq - i):
                faces.append([grid[(i, j)], grid[(i + 1, j)],
                              grid[(i, j + 1)]])
                if j < freq - i - 1:
                    faces.append([grid[(i + 1, j)], grid[(i + 1, j + 1)],
                                  grid[(i, j + 1)]])
    return verts, faces


# ---- weave strands: straight-ahead circuits of the medial graph ---------

def weave_strands(V, F):
    """Returns (midpoints, strands, over) where strands are cyclic lists
    of edge ids and over[(strand_step)] gives +1/-1 crossing parity."""
    e2f = {}
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            e2f[(f[i], f[(i + 1) % m])] = (fi, i)
    eid = {}
    for (a, b) in e2f:
        k = (min(a, b), max(a, b))
        if k not in eid:
            eid[k] = len(eid)
    mid = [None] * len(eid)
    for (a, b), i in eid.items():
        mid[i] = _unit(tuple((V[a][c] + V[b][c]) / 2 for c in range(3)))

    def face_edge(fi, off):
        f = F[fi]
        m = len(f)
        a, b = f[off % m], f[(off + 1) % m]
        return eid[(min(a, b), max(a, b))]

    # for edge e=(a,b): fL contains a->b, fR contains b->a.
    # neighbours in cyclic order: fL@a, fL@b, fR@b, fR@a
    # straight-ahead pairs: (fL@a <-> fR@b), (fL@b <-> fR@a)
    opposite = {}   # (e, neighbour) -> next neighbour
    diag = {}       # (e, neighbour) -> which diagonal (0 or 1)
    for (a, b) in list(e2f.keys()):
        if a > b:
            continue
        e = eid[(a, b)]
        fL, iL = e2f[(a, b)]
        fR, iR = e2f[(b, a)]
        nLa = face_edge(fL, iL - 1)   # edge of fL before e (shares a)
        nLb = face_edge(fL, iL + 1)   # edge of fL after e (shares b)
        nRb = face_edge(fR, iR - 1)   # edge of fR before e (shares b)
        nRa = face_edge(fR, iR + 1)   # edge of fR after e (shares a)
        opposite[(e, nLa)] = nRb
        opposite[(e, nRb)] = nLa
        opposite[(e, nLb)] = nRa
        opposite[(e, nRa)] = nLb
        diag[(e, nLa)] = 0
        diag[(e, nRb)] = 0
        diag[(e, nLb)] = 1
        diag[(e, nRa)] = 1

    used = set()
    strands = []
    overs = []
    for start_key in sorted(opposite.keys()):
        if start_key in used:
            continue
        e, came = start_key
        strand = []
        over = []
        cur, prev = e, came
        while True:
            k = (cur, prev)
            if k in used:
                break
            used.add(k)
            strand.append(cur)
            over.append(1 if diag[k] == 0 else -1)
            nxt_neigh = opposite[k]
            used.add((cur, nxt_neigh))
            prev, cur = cur, nxt_neigh
            if (cur, prev) == start_key or cur == e and prev == came:
                break
            if len(strand) > 10 * len(eid):
                raise RuntimeError("strand failed to close")
        strands.append(strand)
        overs.append(over)
    return mid, strands, overs


# ---- ribbon geometry -----------------------------------------------------

def _slerp(a, b, t):
    d = max(-1.0, min(1.0, sum(a[k] * b[k] for k in range(3))))
    om = math.acos(d)
    if om < 1e-9:
        return a
    sa = sin((1 - t) * om) / sin(om)
    sb = sin(t * om) / sin(om)
    return _unit(tuple(a[k] * sa + b[k] * sb for k in range(3)))


def build_weave(kind='CUBE', freq=1, width=0.10, thickness=0.03,
                amplitude=0.05, subdiv=6, smooth_rounds=2, scale=1.0):
    V, F = seed_poly(kind)
    if kind in ('TETRA', 'OCTA', 'ICOSA'):
        V, F = geodesic(V, F, freq)
    mid, strands, overs = weave_strands(V, F)

    verts = []
    faces = []
    for strand, over in zip(strands, overs):
        L = len(strand)
        # sampled center path with radial over/under modulation
        pts = []
        for i in range(L):
            p0 = mid[strand[i]]
            p1 = mid[strand[(i + 1) % L]]
            for s in range(subdiv):
                t = s / subdiv
                sp = _slerp(p0, p1, t)
                # cosine blend of crossing offsets
                r0 = 1.0 + amplitude * over[i]
                r1 = 1.0 + amplitude * over[(i + 1) % L]
                r = r0 + (r1 - r0) * (1 - cos(math.pi * t)) / 2
                pts.append(tuple(c * r for c in sp))
        # Laplacian smoothing of the closed path (shape only, radius kept)
        for _ in range(smooth_rounds):
            n = len(pts)
            new = []
            for i in range(n):
                a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
                q = tuple((a[k] + 2 * b[k] + c[k]) / 4 for k in range(3))
                r = math.sqrt(sum(x * x for x in b))
                new.append(tuple(x / (math.sqrt(sum(y * y for y in q))
                                      or 1) * r for x in q))
            pts = new
        # sweep a box profile
        n = len(pts)
        base = len(verts)
        for i in range(n):
            p = pts[i]
            nxt = pts[(i + 1) % n]
            prv = pts[i - 1]
            tang = _unit(tuple(nxt[k] - prv[k] for k in range(3)))
            rad = _unit(p)
            side = _unit((tang[1] * rad[2] - tang[2] * rad[1],
                          tang[2] * rad[0] - tang[0] * rad[2],
                          tang[0] * rad[1] - tang[1] * rad[0]))
            for (sw, sr) in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                verts.append(tuple(
                    (p[k] + side[k] * sw * width / 2
                     + rad[k] * sr * thickness / 2) * scale
                    for k in range(3)))
        for i in range(n):
            i2 = (i + 1) % n
            for j in range(4):
                faces.append([base + 4 * i + j,
                              base + 4 * i + (j + 1) % 4,
                              base + 4 * i2 + (j + 1) % 4,
                              base + 4 * i2 + j])
    return verts, faces, len(strands)


# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_weave_add(bpy.types.Operator):
        """Add a woven-strand sphere (strands = straight-ahead circuits
        of the seed polyhedron's medial graph)"""
        bl_idname = "mesh.poly_weave_add"
        bl_label = "Polyhedral Weave"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Seed",
            items=[('CUBE', "Cube (4 strands)", ""),
                   ('ICOSA', "Icosahedron (6 strands)", ""),
                   ('OCTA', "Octahedron (4 strands)", ""),
                   ('TETRA', "Tetrahedron (3 strands)", "")],
            default='CUBE')
        freq: IntProperty(
            name="Geodesic Frequency", default=1, min=1, max=6,
            description="Subdivision of triangular seeds")
        width: FloatProperty(name="Strand Width", default=0.10,
                             min=0.01, max=0.5)
        thickness: FloatProperty(name="Strand Thickness", default=0.03,
                                 min=0.005, max=0.2)
        amplitude: FloatProperty(
            name="Weave Amplitude", default=0.05, min=0.0, max=0.3,
            description="Radial over/under offset at crossings")
        subdiv: IntProperty(name="Path Subdivision", default=6,
                            min=2, max=24)
        smooth_rounds: IntProperty(name="Smoothing", default=2,
                                   min=0, max=10)
        scale: FloatProperty(name="Radius", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            verts, faces, ns = build_weave(
                self.kind, self.freq, self.width, self.thickness,
                self.amplitude, self.subdiv, self.smooth_rounds,
                self.scale)
            me = bpy.data.meshes.new("Weave")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Weave", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{ns} strands")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'freq', 'width', 'thickness', 'amplitude',
                      'subdiv', 'smooth_rounds', 'scale'):
                if k == 'freq' and self.kind == 'CUBE':
                    continue
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.poly_weave_add", icon='MOD_LATTICE')

    def register():
        bpy.utils.register_class(MESH_OT_weave_add)
        bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_weave_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for kind, freq in (('CUBE', 1), ('OCTA', 1), ('ICOSA', 1),
                           ('TETRA', 1), ('ICOSA', 2), ('ICOSA', 3)):
            V, F = seed_poly(kind)
            if kind != 'CUBE':
                V, F = geodesic(V, F, freq)
            mid, strands, overs = weave_strands(V, F)
            total = sum(len(s) for s in strands)
            ne = len(mid)
            print(f"{kind} f{freq}: strands={len(strands)} "
                  f"lengths={sorted(set(len(s) for s in strands))} "
                  f"crossings-covered={total} (2x edges = {2 * ne}) "
                  f"{'OK' if total == 2 * ne else 'BAD'}")
