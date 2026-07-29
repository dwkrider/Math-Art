
# Apollonian Gasket / Sphere-Packing Generator for Blender
#
# The Apollonian gasket: begin with mutually tangent circles, inscribe
# a new circle in every curvilinear gap, and recurse forever. The 3D
# analogue packs mutually tangent spheres (Soddy spheres). Both are
# exactly self-similar fractals whose gaps are filled at every scale.
#
# Curvatures obey the Descartes circle theorem (2D) and its Soddy-
# Gosper generalisation (3D). This generator uses the sign-free
# "reflection" form: given a set of mutually tangent circles/spheres
# and one further tangent element c0, the OTHER element tangent to the
# same set is
#     k'   = 2*sum(k_i) - k0
#     k'c' = 2*sum(k_i c_i) - k0 c0
# so each gap is filled without any square-root sign ambiguity, and
# every sub-gap recurses the same way. Recursion stops at a minimum
# radius. Circles are drawn as rings (tori in the plane); spheres as
# icospheres. Output is centred and fit to a 2 m cube.
#
# References:
# - Rene Descartes (1643, letter to Princess Elisabeth); rediscovered
#   by Frederick Soddy, "The Kiss Precise", Nature 137, 1936, p. 1021
#   (verse giving both the 2D theorem and its 3D sphere version).
# - Apollonius of Perga, "Tangencies" (lost; c. 200 BC), the classical
#   problem of tangent circles.
# - Jeffrey C. Lagarias, Colin L. Mallows and Allan R. Wilf, "Beyond
#   the Descartes Circle Theorem", Amer. Math. Monthly 109, 2002,
#   pp. 338-361 (the curvature-centre / reflection formulation).

bl_info = {
    "name": "Apollonian Gasket",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Apollonian Gasket",
    "description": "Apollonian circle gasket and 3D sphere packing",
    "category": "Add Mesh",
}

import cmath
import math

import numpy as np


class Ball:
    """A circle or sphere: signed curvature k (negative if enclosing)
    and centre c (a 2- or 3-vector)."""
    __slots__ = ('k', 'c')

    def __init__(self, k, c):
        self.k = float(k)
        self.c = np.asarray(c, dtype=float)


def reflect(group, c0):
    """The element tangent to every ball in `group` other than c0."""
    ksum = sum(b.k for b in group)
    csum = sum(b.k * b.c for b in group)
    k = 2.0 * ksum - c0.k
    c = (2.0 * csum - c0.k * c0.c) / k
    return Ball(k, c)


def _overlaps(n, out, tol=1e-4):
    """True if positive ball n interpenetrates any positive ball in
    out. In 3D, naive Descartes reflection over every quadruple of the
    seed over-fills; rejecting overlaps keeps the packing valid."""
    rn = 1.0 / n.k
    for b in out:
        if b.k <= 0.0:
            continue
        if np.linalg.norm(n.c - b.c) < rn + 1.0 / b.k - tol:
            return True
    return False


# ---- 2D gasket ----------------------------------------------------------

def _base_2d():
    """Outer circle (-1) with two inner circles (2), plus the two
    Descartes circles (3) inscribed in the top and bottom gaps."""
    A = Ball(-1.0, [0.0, 0.0])
    B = Ball(2.0, [0.5, 0.0])
    C = Ball(2.0, [-0.5, 0.0])
    kA, kB, kC = A.k, B.k, C.k
    zA, zB, zC = 0.0 + 0j, 0.5 + 0j, -0.5 + 0j
    S = kA + kB + kC
    k4 = S + 2.0 * math.sqrt(max(kA * kB + kB * kC + kC * kA, 0.0))
    zsum = kA * zA + kB * zB + kC * zC
    zroot = cmath.sqrt(kA * kB * zA * zB + kB * kC * zB * zC
                       + kC * kA * zC * zA)
    tops = []
    for zr in (zsum + 2 * zroot, zsum - 2 * zroot):
        z = zr / k4
        tops.append(Ball(k4, [z.real, z.imag]))
    return A, B, C, tops


def _rec2(a, b, c, c0, out, depth, min_r, cap):
    if len(out) >= cap:
        return
    n = reflect([a, b, c], c0)
    if n.k <= 0.0 or 1.0 / n.k < min_r or depth < 0:
        return
    out.append(n)
    _rec2(a, b, n, c, out, depth - 1, min_r, cap)
    _rec2(a, c, n, b, out, depth - 1, min_r, cap)
    _rec2(b, c, n, a, out, depth - 1, min_r, cap)


def gasket_2d(depth, min_r, cap):
    A, B, C, tops = _base_2d()
    out = [A, B, C] + tops
    for t in tops:
        _rec2(A, B, t, C, out, depth, min_r, cap)
        _rec2(A, C, t, B, out, depth, min_r, cap)
        _rec2(B, C, t, A, out, depth, min_r, cap)
    return out


# ---- 3D packing ---------------------------------------------------------

_TETRA = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
                  dtype=float)
_TETRA /= math.sqrt(3.0)


def _base_3d():
    """Outer sphere (-1) with four equal inner spheres at tetrahedron
    vertices, all mutually tangent and tangent to the outer sphere."""
    # solve 1.2247 r = 1 - r  (tetra circumradius = 1 - r)
    circ = math.sqrt(3.0 / 8.0) * 2.0            # per unit tangent dist
    # centres at distance (1-r); edge = 2r; circumradius = circ * r
    r = 1.0 / (1.0 + circ)
    k = 1.0 / r
    outer = Ball(-1.0, [0.0, 0.0, 0.0])
    inner = [Ball(k, d * (1.0 - r)) for d in _TETRA]
    return [outer] + inner


def _rec3(a, b, c, d, c0, out, depth, min_r, cap):
    if len(out) >= cap:
        return
    n = reflect([a, b, c, d], c0)
    if n.k <= 0.0 or 1.0 / n.k < min_r or depth < 0:
        return
    if _overlaps(n, out):
        return
    out.append(n)
    _rec3(a, b, c, n, d, out, depth - 1, min_r, cap)
    _rec3(a, b, d, n, c, out, depth - 1, min_r, cap)
    _rec3(a, c, d, n, b, out, depth - 1, min_r, cap)
    _rec3(b, c, d, n, a, out, depth - 1, min_r, cap)


def packing_3d(depth, min_r, cap):
    base = _base_3d()
    out = list(base[1:])                         # keep the inner spheres
    for i in range(5):
        four = [base[j] for j in range(5) if j != i]
        _rec3(four[0], four[1], four[2], four[3], base[i],
              out, depth, min_r, cap)
    return out


# ---- meshing ------------------------------------------------------------

def _icosphere():
    """Unit icosphere: icosahedron subdivided once (42 verts)."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
         (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
         (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, dtype=float) for p in v]
    mid = {}

    def midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key not in mid:
            m = (verts[i] + verts[j]) / 2.0
            verts.append(m)
            mid[key] = len(verts) - 1
        return mid[key]

    faces = []
    for a, b, c in f:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    V = np.array([p / np.linalg.norm(p) for p in verts])
    return V, faces


def _ring(centre, R, tube, nu, nv):
    verts, faces = [], []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        cu, su = math.cos(u), math.sin(u)
        for j in range(nv):
            vv = 2 * math.pi * j / nv
            rr = R + tube * math.cos(vv)
            verts.append((centre[0] + rr * cu, centre[1] + rr * su,
                          tube * math.sin(vv)))
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            faces.append((a, b, c, d))
    return verts, faces


def build_apollonian(mode='PACKING', depth=5, min_r=0.0, cap=4000,
                     tube_ratio=0.25, inflate=1.0, ring_seg=20,
                     tube_seg=8, scale=1.0):
    """Build the gasket (2D rings) or sphere packing (3D). Returns
    (verts, faces, n) centred and fit to a 2 m cube."""
    verts, faces = [], []
    if mode == 'GASKET':
        mr = min_r if min_r > 0 else 0.01
        balls = gasket_2d(depth, mr, cap)
        for b in balls:
            R = 1.0 / abs(b.k)
            v, f = _ring((b.c[0], b.c[1]), R, tube_ratio * R,
                         ring_seg, tube_seg)
            base = len(verts)
            verts.extend(v)
            faces.extend([tuple(base + i for i in fc) for fc in f])
        n = len(balls)
    else:
        mr = min_r if min_r > 0 else 0.03
        balls = packing_3d(depth, mr, cap)
        SV, SF = _icosphere()
        for b in balls:
            r = inflate / b.k
            base = len(verts)
            verts.extend((b.c + r * p).tolist() for p in SV)
            faces.extend([tuple(base + i for i in fc) for fc in SF])
        n = len(balls)
    V = np.asarray(verts)
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    V = (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)
    return V * scale, faces, n


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_apollonian_add(bpy.types.Operator):
        """Add an Apollonian gasket (2D rings) or sphere packing"""
        bl_idname = "mesh.apollonian_add"
        bl_label = "Apollonian Gasket"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            items=[('PACKING', "Sphere Packing (3D)",
                    "Mutually tangent Soddy spheres"),
                   ('GASKET', "Gasket (2D rings)",
                    "Apollonian circles drawn as flat rings")],
            default='PACKING')
        depth: IntProperty(
            name="Depth", default=5, min=1, max=12,
            description="Maximum recursion depth")
        min_r: FloatProperty(
            name="Min Radius", default=0.0, min=0.0, max=0.5,
            description="Stop inscribing below this radius "
                        "(0 = per-mode default)")
        cap: IntProperty(
            name="Max Count", default=4000, min=10, max=40000,
            description="Hard cap on circles/spheres")
        tube_ratio: FloatProperty(
            name="Ring Tube", default=0.25, min=0.03, max=0.5,
            description="Ring tube radius as a fraction of circle "
                        "radius (gasket mode)")
        inflate: FloatProperty(
            name="Sphere Inflate", default=1.0, min=1.0, max=1.15,
            description="Grow spheres slightly to fuse contacts for "
                        "printing (packing mode)")
        ring_seg: IntProperty(name="Ring Segments", default=20,
                              min=6, max=64)
        tube_seg: IntProperty(name="Tube Segments", default=8,
                              min=4, max=32)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                            max=100.0)
        smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            verts, faces, n = build_apollonian(
                self.mode, self.depth, self.min_r, self.cap,
                self.tube_ratio, self.inflate, self.ring_seg,
                self.tube_seg, self.scale)
            me = bpy.data.meshes.new("Apollonian")
            me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Apollonian", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{self.mode}: {n} elements, "
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            lay.prop(self, 'depth')
            lay.prop(self, 'min_r')
            lay.prop(self, 'cap')
            if self.mode == 'GASKET':
                lay.prop(self, 'tube_ratio')
            else:
                lay.prop(self, 'inflate')
            lay.prop(self, 'ring_seg')
            lay.prop(self, 'tube_seg')
            lay.prop(self, 'scale')
            lay.prop(self, 'smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.apollonian_add", icon='MESH_CIRCLE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_apollonian_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_apollonian_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        # 2D: verify the packing is valid (no two positive circles
        # overlap; all inside the outer circle)
        balls = gasket_2d(depth=4, min_r=0.01, cap=3000)
        pos = [b for b in balls if b.k > 0]
        outer = next(b for b in balls if b.k < 0)
        Rout = 1.0 / abs(outer.k)
        bad = 0
        for i in range(len(pos)):
            bi = pos[i]
            ri = 1.0 / bi.k
            if np.linalg.norm(bi.c - outer.c) + ri > Rout + 1e-6:
                bad += 1
            for j in range(i + 1, len(pos)):
                bj = pos[j]
                rj = 1.0 / bj.k
                if (np.linalg.norm(bi.c - bj.c)
                        < ri + rj - 1e-6):
                    bad += 1
        print(f"GASKET  depth4: circles={len(balls)} overlaps={bad} "
              f"{'OK' if bad == 0 else 'BAD'}")
        # 3D: same validity check
        sph = packing_3d(depth=3, min_r=0.03, cap=3000)
        bad3 = 0
        for i in range(len(sph)):
            ri = 1.0 / sph[i].k
            for j in range(i + 1, len(sph)):
                rj = 1.0 / sph[j].k
                if (np.linalg.norm(sph[i].c - sph[j].c)
                        < ri + rj - 1e-6):
                    bad3 += 1
        v, f, n = build_apollonian('PACKING', depth=3, min_r=0.05)
        print(f"PACKING depth3: spheres={len(sph)} overlaps={bad3} "
              f"mesh_V={len(v)} F={len(f)} "
              f"{'OK' if bad3 == 0 else 'BAD'}")
