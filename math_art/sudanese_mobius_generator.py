
# Sudanese Mobius Band Generator for Blender
#
# The "Sudanese" Mobius band is the most symmetric embedding of the
# Mobius strip as a MINIMAL surface: it lives inside the round
# 3-sphere S^3 with a great circle as its single boundary, and is
# then stereographically projected into ordinary 3-space (a projection
# that keeps the boundary a perfect round circle).
#
# It is one half of H. Blaine Lawson's minimally-immersed Klein bottle
# in S^3.  Writing R^4 = C^2, Lawson's Klein bottle is the set
#
#     x(t, u) = ( cos t cos u, sin t cos u,
#                 cos 2t sin u, sin 2t sin u ) in S^3,
#         0 <= t < pi,   0 <= u < 2pi.
#
# Taking the half u in [0, pi] gives an embedded Mobius band: the
# t = 0 and t = pi edges glue with a flip (u <-> pi - u), and the
# u = 0 / u = pi edges join into one great-circle boundary in the
# x1-x2 plane.  Its symmetry group in S^3 is a full O(2), which is
# why the projected model looks so clean.
#
# We stereographically project S^3 -> R^3 from the point
# p = (-1, 0, -1, 0)/sqrt(2).  Numerically this is the point of S^3
# farthest from the band (the band never comes closer than a cos-
# distance of 1 - 1/sqrt(2) ~ 0.293), so the projection stays finite
# and bounded and the result is watertight along its two welded seams.
#
# The nickname "Sudanese" is not geographic: it honours the topologists
# Sue Goodman and Daniel Asimov (SUe + DANiel -> "Sue-Dan-ese"), who
# studied the surface in the 1970s.
#
# References:
#   - H. Blaine Lawson, Jr., "Complete Minimal Surfaces in S^3",
#     Annals of Mathematics 92 (1970), 335-374 (the tau_{m,k} family;
#     the Klein bottle / Mobius band arise for the twisted cases).
#   - George K. Francis, "A Topological Picturebook", Springer 1987
#     (the Sudanese Mobius band and the Sue/Dan naming).
#   - John M. Sullivan, "The Optiverse and other sphere eversions" and
#     the Berlin DDG course notes on Lawson's surfaces and the
#     Sudanese Mobius band.

bl_info = {
    "name": "Sudanese Mobius Band",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Sudanese Mobius Band",
    "description": "Lawson's minimal Mobius band in S^3, stereographically "
                   "projected to R^3 (after Lawson; named for Goodman & "
                   "Asimov)",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi, sqrt


def _vkey(p):
    return (round(p[0], 7), round(p[1], 7), round(p[2], 7))


class _Mesh:
    """Vertex-welding mesh builder (welds the Mobius seams)."""

    def __init__(self):
        self.verts = []
        self.faces = []
        self._ids = {}

    def vid(self, p):
        k = _vkey(p)
        i = self._ids.get(k)
        if i is None:
            i = len(self.verts)
            self._ids[k] = i
            self.verts.append(p)
        return i

    def quad(self, a, b, c, d):
        ids = []
        for p in (a, b, c, d):
            i = self.vid(p)
            if i not in ids:
                ids.append(i)
        if len(ids) >= 3:
            self.faces.append(ids)


# Stereographic projection S^3 -> R^3 from p, read off in the
# orthonormal basis (e1, e2, e3) of the hyperplane p^perp.
_R2 = sqrt(2.0)
_P = (-1.0 / _R2, 0.0, -1.0 / _R2, 0.0)
_E1 = (0.0, 1.0, 0.0, 0.0)
_E2 = (0.0, 0.0, 0.0, 1.0)
_E3 = (1.0 / _R2, 0.0, -1.0 / _R2, 0.0)


def _dot4(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def _s3_point(t, u):
    """Lawson's Klein-bottle map into S^3 subset R^4."""
    return (cos(t) * cos(u), sin(t) * cos(u),
            cos(2 * t) * sin(u), sin(2 * t) * sin(u))


def _project(x):
    d = _dot4(x, _P)
    s = 1.0 - d                       # > 0 for every band point
    q = (x[0] - d * _P[0], x[1] - d * _P[1],
         x[2] - d * _P[2], x[3] - d * _P[3])
    inv = 1.0 / s
    q = (q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv)
    return (_dot4(q, _E1), _dot4(q, _E2), _dot4(q, _E3))


def build_sudanese_mobius(u_segments=180, v_segments=40, scale=1.0):
    """Watertight Sudanese Mobius band.

    t (u_segments) runs 0..pi around the band; u (v_segments) runs
    0..pi across it.  Grid points on S^3 are stereographically
    projected; coincident seam vertices weld automatically.
    """
    m = _Mesh()
    nt = max(12, u_segments)
    nu = max(4, v_segments)

    def pt(i, j):
        t = pi * i / nt
        u = pi * j / nu
        return _project(_s3_point(t, u))

    grid = [[pt(i, j) for j in range(nu + 1)] for i in range(nt + 1)]
    for i in range(nt):
        for j in range(nu):
            m.quad(grid[i][j], grid[i + 1][j],
                   grid[i + 1][j + 1], grid[i][j + 1])

    if scale != 1.0:
        m.verts = [(v[0] * scale, v[1] * scale, v[2] * scale)
                   for v in m.verts]
    return m.verts, m.faces


try:
    import bpy
    import bmesh
    from bpy.props import IntProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_sudanese_mobius_add(bpy.types.Operator):
        """Add the Sudanese Mobius band -- Lawson's minimal Mobius """ \
            """strip in S^3, stereographically projected to R^3"""
        bl_idname = "mesh.sudanese_mobius_add"
        bl_label = "Sudanese Mobius Band"
        bl_options = {'REGISTER', 'UNDO'}

        u_segments: IntProperty(
            name="Around", default=180, min=24, max=512,
            description="Segments along the band (the t direction)")
        v_segments: IntProperty(
            name="Across", default=40, min=6, max=200,
            description="Segments across the band (the u direction)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            verts, faces = build_sudanese_mobius(
                self.u_segments, self.v_segments, self.scale)

            def _fit(vs):
                # centre on the bounding-box midpoint and scale so the
                # largest extent is 2.0 * scale (a ~2 m cube).  scale
                # divides out of the ratio, so it is not applied twice.
                if not vs:
                    return vs
                xs = [v[0] for v in vs]
                ys = [v[1] for v in vs]
                zs = [v[2] for v in vs]
                cx = 0.5 * (min(xs) + max(xs))
                cy = 0.5 * (min(ys) + max(ys))
                cz = 0.5 * (min(zs) + max(zs))
                ext = max(max(xs) - min(xs), max(ys) - min(ys),
                          max(zs) - min(zs))
                s = (2.0 * self.scale / ext) if ext > 1e-9 else 1.0
                return [((v[0] - cx) * s, (v[1] - cy) * s,
                         (v[2] - cz) * s) for v in vs]

            me = bpy.data.meshes.new("Sudanese Mobius")
            me.from_pydata(_fit(verts), [], faces)
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Sudanese Mobius", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'u_segments')
            lay.prop(self, 'v_segments')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.sudanese_mobius_add",
                             icon='MOD_SIMPLEDEFORM')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_sudanese_mobius_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_sudanese_mobius_add)


def _selftest():
    from collections import Counter, defaultdict
    v, f = build_sudanese_mobius(120, 40)

    # every projected point must be finite and bounded
    for p in v:
        for c in p:
            assert math.isfinite(c), "non-finite vertex (projection blew up)"

    # edge census: a Mobius band has Euler characteristic 0 and a
    # single boundary loop of degree-2 vertices.
    cnt = Counter()
    for fc in f:
        n = len(fc)
        for i in range(n):
            a, b = fc[i], fc[(i + 1) % n]
            cnt[(min(a, b), max(a, b))] += 1
    border = [e for e, c in cnt.items() if c == 1]
    chi = len(v) - len(cnt) + len(f)

    adj = defaultdict(list)
    for a, b in border:
        adj[a].append(b)
        adj[b].append(a)
    maxdeg = max((len(a) for a in adj.values()), default=0)
    loops = 0
    seen = set()
    for s in list(adj):
        if s in seen:
            continue
        loops += 1
        cur, prev = s, None
        while True:
            seen.add(cur)
            nxt = [x for x in adj[cur] if x != prev]
            if not nxt or nxt[0] == s:
                break
            prev, cur = cur, nxt[0]

    ok = (chi == 0 and loops == 1 and maxdeg == 2)
    print(f"sudanese_mobius: verts={len(v)} faces={len(f)} chi={chi}(0) "
          f"boundary_loops={loops}(1) max_boundary_deg={maxdeg}(2) "
          f"{'OK' if ok else 'BAD'}")
    assert ok
