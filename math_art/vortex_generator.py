
# Vertex Vortices generator for Blender, inspired by Robert
# Fathauer's "Vertices Vortices" sculpture: each face of a
# polyhedron is divided by spokes from its centre to each of its
# vertices, the spokes are bent within the face plane with equal
# chirality (a pinwheel), the original edges are deleted, and a
# minimal surface spans each group of four bent spokes.
#
# Every original edge yields exactly one such group: the two
# spokes reaching its endpoints from each adjacent face's centre
# form a closed four-sided frame straddling the deleted edge, so
# there is one saddle patch per edge and every spoke borders
# exactly two patches -- the welded result is a closed surface
# with swirling vortices meeting at the polyhedron's vertices.

bl_info = {
    "name": "Vertex Vortices",
    "author": "Math Art project (after Robert Fathauer)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Polyhedra with pinwheel-bent face spokes and "
                   "minimal surfaces spanning each deleted edge",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def build_vortex(V, F, bend=0.11, m=12, iterations=30,
                 reverse=False):
    """(verts, faces, edge_ids): the vortex surface.  Each face's
    centre-to-vertex spokes bend within the face plane by `bend` x
    the spoke length (4t(1-t) profile), all with the same
    chirality; one membrane patch (a Coons patch relaxed toward
    minimal area) spans the four bent spokes around each original
    edge."""
    V = [np.asarray(v, float) for v in V]
    centers = [np.mean([V[i] for i in f], axis=0) for f in F]
    sgn = -1.0 if reverse else 1.0

    def newell(f):
        n = np.zeros(3)
        for i in range(len(f)):
            a, b = V[f[i]], V[f[(i + 1) % len(f)]]
            n += np.cross(a, b)
        return n / (np.linalg.norm(n) or 1.0)

    # one bent spoke per (face, vertex), centre -> vertex
    spoke = {}
    for fi, f in enumerate(F):
        n = newell(f)
        C = centers[fi]
        for v in f:
            d = V[v] - C
            L = np.linalg.norm(d)
            perp = np.cross(n, d / (L or 1.0)) * sgn
            pts = []
            for j in range(m + 1):
                t = j / m
                pts.append(C + d * t
                           + perp * (bend * L * 4.0 * t
                                     * (1.0 - t)))
            spoke[(fi, v)] = pts

    # the face containing each DIRECTED edge a->b
    dir_face = {}
    for fi, f in enumerate(F):
        for i in range(len(f)):
            dir_face[(f[i], f[(i + 1) % len(f)])] = fi
    if (set(dir_face) != {(b, a) for a, b in dir_face}
            or len(dir_face) != sum(len(f) for f in F)):
        raise ValueError("seed must be a closed manifold mesh "
                         "with consistent winding")

    try:
        try:
            from .minimal_surface_toolkit import minimize_area
        except ImportError:
            from minimal_surface_toolkit import minimize_area
    except Exception:
        minimize_area = None

    vid = {}
    verts = []
    faces = []
    eids = []

    def emit(p, weld):
        if not weld:
            verts.append(tuple(p))
            return len(verts) - 1
        key = tuple(np.round(p, 9))
        if key not in vid:
            vid[key] = len(verts)
            verts.append(tuple(p))
        return vid[key]

    for ei, (a, b) in enumerate(k for k in dir_face if k[0] < k[1]):
        fa = dir_face[(a, b)]         # edge runs a->b in face fa
        fb = dir_face[(b, a)]
        # the frame is a curved quad with corners centre_a, a,
        # centre_b, b; a Coons patch over its four bent spokes is
        # the initial membrane
        bot = np.array(spoke[(fa, a)])           # centre_a -> a
        top = np.array(spoke[(fb, b)][::-1])     # b -> centre_b
        lef = np.array(spoke[(fa, b)])           # centre_a -> b
        rig = np.array(spoke[(fb, a)][::-1])     # a -> centre_b
        c00, c10 = bot[0], bot[-1]
        c01, c11 = top[0], top[-1]
        n1 = m + 1
        P = np.empty((n1, n1, 3))
        for iu in range(n1):
            u = iu / m
            for iv in range(n1):
                v = iv / m
                P[iu, iv] = ((1 - v) * bot[iu] + v * top[iu]
                             + (1 - u) * lef[iv] + u * rig[iv]
                             - ((1 - u) * (1 - v) * c00
                                + u * (1 - v) * c10
                                + (1 - u) * v * c01
                                + u * v * c11))
        P = P.reshape(-1, 3)
        tris = []
        for iu in range(m):
            for iv in range(m):
                q = iu * n1 + iv
                tris.append((q, q + n1, q + 1))
                tris.append((q + 1, q + n1, q + n1 + 1))
        fixed = np.zeros(len(P), dtype=bool)
        for iu in range(n1):
            fixed[iu * n1] = fixed[iu * n1 + m] = True
        fixed[:n1] = fixed[-n1:] = True
        if minimize_area is not None and iterations > 0:
            minimize_area(P, np.array(tris), fixed,
                          outer_iters=iterations)
        elif iterations > 0:
            nbrs = [set() for _ in range(len(P))]
            for t in tris:
                for q in range(3):
                    nbrs[t[q]].add(t[(q + 1) % 3])
                    nbrs[t[(q + 1) % 3]].add(t[q])
            nbl = [sorted(s) for s in nbrs]
            for _ in range(iterations * 4):
                Q = np.empty_like(P)
                for i, s in enumerate(nbl):
                    Q[i] = P[s].mean(axis=0)
                P[~fixed] += 0.6 * (Q[~fixed] - P[~fixed])
        local = [emit(p, bool(fixed[i])) for i, p in enumerate(P)]
        for t in tris:
            faces.append([local[i] for i in t])
            eids.append(ei)

    # orient outward (patch winding follows the frame loop, which
    # is consistent but may come out inside-out as a whole)
    A = np.array(verts)
    vol = sum(np.dot(A[f[0]], np.cross(A[f[1]], A[f[2]]))
              for f in faces)
    if vol < 0:
        faces = [f[::-1] for f in faces]
    return verts, faces, eids


def _seed(name):
    try:
        from . import spiked_polyhedron_generator as sp
    except ImportError:
        import spiked_polyhedron_generator as sp
    V, F = sp._seed(name)
    return [tuple(v) for v in V], [list(f) for f in F]


if _IN_BLENDER:

    class MESH_OT_vertex_vortices_add(bpy.types.Operator):
        """Bend each face's centre-to-vertex spokes into a
        pinwheel, delete the original edges and span a minimal
        surface over the four spokes around each edge (after
        Robert Fathauer's Vertices Vortices)"""
        bl_idname = "mesh.vertex_vortices_add"
        bl_label = "Vertex Vortices"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(
            name="Seed",
            items=[('TETRA', "Tetrahedron", ""),
                   ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", ""),
                   ('DODECA', "Dodecahedron",
                    "the Vertices Vortices"),
                   ('ICOSA', "Icosahedron", "")],
            default='DODECA')
        bend: FloatProperty(
            name="Bend", default=0.11, min=0.0, max=0.6,
            description="Sideways bend of each spoke, as a "
                        "fraction of its length (about 0.11 in "
                        "the original sculpture)")
        samples: IntProperty(
            name="Spoke Samples", default=12, min=3, max=48,
            description="Subdivisions of each bent spoke")
        iterations: IntProperty(
            name="Solver Iterations", default=30, min=0, max=200,
            description="Plateau area-minimization iterations "
                        "per patch")
        reverse: BoolProperty(
            name="Reverse Swirl", default=False,
            description="Mirror the vortex chirality")
        smooth: BoolProperty(name="Smooth Shading", default=False)
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="Solidify modifier thickness (0 = raw "
                        "surface)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            V, F = _seed(self.seed)
            try:
                verts, faces, eids = build_vortex(
                    V, F, self.bend, self.samples,
                    self.iterations, self.reverse)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            name = ("Vertices Vortices" if self.seed == 'DODECA'
                    else f"Vortices ({self.seed.title()})")
            # fit (roughly) within a 2 x scale cube at the origin
            lo = [min(v[k] for v in verts) for k in range(3)]
            hi = [max(v[k] for v in verts) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0 for k in range(3)) \
                or 1.0
            s = self.scale / half
            verts = [tuple((v[k] - (lo[k] + hi[k]) / 2.0) * s
                           for k in range(3)) for v in verts]
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set(
                'use_smooth', [self.smooth] * len(me.polygons))
            attr = me.attributes.new("edge_index", 'INT', 'FACE')
            if len(me.polygons) == len(eids):
                attr.data.foreach_set('value', eids)
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('seed', 'bend', 'samples', 'iterations',
                      'reverse', 'smooth', 'thickness', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.vertex_vortices_add",
                             icon='FORCE_VORTEX')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_vertex_vortices_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_vertex_vortices_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        CUBE_V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
                  for z in (-1, 1)]
        CUBE_F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                  [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        verts, faces, eids = build_vortex(
            CUBE_V, CUBE_F, bend=0.11, m=8, iterations=20)
        # 12 edges -> 12 patches
        assert len(set(eids)) == 12
        cnt = {}
        for f in faces:
            for i in range(len(f)):
                e = frozenset((f[i], f[(i + 1) % len(f)]))
                cnt[e] = cnt.get(e, 0) + 1
        watertight = all(c == 2 for c in cnt.values())
        finite = all(all(math.isfinite(c) for c in v)
                     for v in verts)
        # consistent orientation: each welded edge traversed once
        # in each direction
        dcnt = {}
        for f in faces:
            for i in range(len(f)):
                d = (f[i], f[(i + 1) % len(f)])
                dcnt[d] = dcnt.get(d, 0) + 1
        oriented = all(c == 1 for c in dcnt.values())
        A = np.array(verts)
        vol = sum(np.dot(A[f[0]], np.cross(A[f[1]], A[f[2]]))
                  for f in faces) / 6.0
        print(f"cube vortex: V={len(verts)} F={len(faces)} "
              f"watertight={watertight} oriented={oriented} "
              f"finite={finite} vol={vol:.3f}")
        assert watertight and oriented and finite and vol > 0
        # chirality flips with reverse
        v2, _, _ = build_vortex(CUBE_V, CUBE_F, bend=0.11, m=8,
                                iterations=0, reverse=True)
        assert not np.allclose(np.array(verts)[:10],
                               np.array(v2)[:10])
        print("vortex standalone tests passed")
