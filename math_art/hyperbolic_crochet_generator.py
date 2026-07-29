
# Hyperbolic Crochet Generator for Blender
#
# Surfaces of constant negative Gaussian curvature -- the shapes Daina
# Taimina popularised by crocheting the hyperbolic plane (increase
# stitches at a constant ratio and the fabric ruffles). Two families:
#
#   EXACT   Smooth constant-curvature surfaces from the sine-Gordon
#           equation: the pseudosphere (tractricoid), Dini's twisted
#           pseudosphere, and Kuen's surface. These have K = -1
#           everywhere but, by Hilbert's theorem, only cover a patch of
#           the hyperbolic plane -- they cannot ruffle.
#
#   RUFFLE  The crochet itself: an annular mesh whose intrinsic metric
#           is hyperbolic (row i has circumference 2*pi*R*sinh(rho/R),
#           growing exponentially) is relaxed under edge (stretch) and
#           bending (Laplacian) energy. Because the exponential length
#           cannot lie flat, the sheet buckles into self-similar
#           ruffles -- the fractal, lettuce/kale-like morphology. The
#           crochet increase ratio N (increase one stitch every N)
#           sets the curvature radius R = h / ln(1 + 1/N) ~ N*h.
#
# References:
# - Daina Taimina, "Crocheting Adventures with Hyperbolic Planes",
#   A K Peters, 2009; David W. Henderson and Daina Taimina,
#   "Crocheting the hyperbolic plane", Math. Intelligencer 23(2),
#   2001, pp. 17-28.
# - Pseudosphere / constant-curvature surfaces: Eugenio Beltrami
#   (1868); Ulisse Dini (1865); Theodor Kuen (1884).
# - Ruffling as elastic energy minimization with recursive (fractal)
#   buckling: John A. Gemmer, E. Sharon, S. C. Venkataramani et al.,
#   "Isometric immersions, energy minimization and self-similar
#   buckling in non-Euclidean elastic sheets", and "Distributed branch
#   points and the shape of elastic surfaces with constant negative
#   curvature", arXiv:2006.14461.
# - Hilbert's theorem (no complete C^2 isometric immersion of H^2 in
#   R^3): D. Hilbert, Trans. AMS 2, 1901, pp. 87-99.

bl_info = {
    "name": "Hyperbolic Crochet",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Hyperbolic Crochet",
    "description": "Constant-negative-curvature surfaces and ruffled "
                   "crocheted hyperbolic planes",
    "category": "Add Mesh",
}

import math

import numpy as np


# ==========================================================================
# Exact constant-curvature surfaces (parametric)
# ==========================================================================

def _pseudosphere(U, V):
    x = np.cosh(U) ** -1 * np.cos(V)
    y = np.cosh(U) ** -1 * np.sin(V)
    z = U - np.tanh(U)
    return x, y, z


def _dini(U, V, twist=0.2):
    x = np.cos(V) * np.sin(U)
    y = np.sin(V) * np.sin(U)
    z = np.cos(U) + np.log(np.tan(U / 2.0)) + twist * V
    return x, y, z


def _kuen(U, V):
    denom = 1.0 + (U * np.sin(V)) ** 2
    x = 2.0 * (np.cos(U) + U * np.sin(U)) * np.sin(V) / denom
    y = 2.0 * (np.sin(U) - U * np.cos(U)) * np.sin(V) / denom
    z = np.log(np.tan(V / 2.0)) + 2.0 * np.cos(V) / denom
    return x, y, z


# (function, u range, v range, wrap_v)
EXACT = {
    'PSEUDOSPHERE': (_pseudosphere, (0.0, 3.0), (0.0, 2 * math.pi),
                     True),
    'DINI': (_dini, (0.05, 1.5), (0.0, 12.0 * math.pi), False),
    'KUEN': (_kuen, (-4.5, 4.5), (0.08, math.pi - 0.08), False),
}


def _grid_faces(nu, nv, wrap_v):
    faces = []
    for i in range(nu - 1):
        for j in range(nv - (0 if wrap_v else 1)):
            j1 = (j + 1) % nv
            faces.append((i * nv + j, (i + 1) * nv + j,
                          (i + 1) * nv + j1, i * nv + j1))
    return faces


def build_exact(kind, ures, vres):
    fn, (u0, u1), (v0, v1), wrap = EXACT[kind]
    us = np.linspace(u0, u1, ures)
    vs = (np.linspace(v0, v1, vres, endpoint=False) if wrap
          else np.linspace(v0, v1, vres))
    U, Vv = np.meshgrid(us, vs, indexing='ij')
    x, y, z = fn(U, Vv)
    V = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    faces = _grid_faces(ures, vres, wrap)
    return _center(V), faces


# ==========================================================================
# Ruffled crochet: intrinsic-hyperbolic annular mesh + PBD relaxation
# ==========================================================================

def _join_rings(sa, na, sb, nb):
    """Bridge two concentric rings of different vertex counts with a
    triangle strip, advancing whichever ring is 'behind' in angle."""
    faces = []
    ia = ib = 0
    while ia < na or ib < nb:
        a0 = sa + ia % na
        b0 = sb + ib % nb
        if ib >= nb or (ia < na and (ia + 1) / na <= (ib + 1) / nb):
            faces.append((a0, b0, sa + (ia + 1) % na))
            ia += 1
        else:
            faces.append((a0, b0, sb + (ib + 1) % nb))
            ib += 1
    return faces


def _crochet_mesh(ratio_n, rows, stitch, max_stitches):
    """The actual crochet: each row's stitch count grows with the
    (exponential) hyperbolic circumference, so every stitch stays size
    h. Each ring is seeded with a PHASE-COHERENT PERIOD-DOUBLING
    cascade of z-ruffles (base wavenumber m0, then m0*2, m0*4, ...
    activated as the excess grows) -- the doubling radii are the
    surface's distributed branch points, giving the self-similar
    kale/lettuce morphology. Edge rest lengths are the true hyperbolic
    distances between the vertices' (rho, theta) parameters."""
    h = stitch
    R = h / math.log(1.0 + 1.0 / ratio_n)
    rng = np.random.default_rng(0)
    m0 = 3                                        # cascade base
    P, faces, rings, prev = [], [], [], None
    for i in range(rows):
        rho = (i + 1) * h
        circ = 2.0 * math.pi * R * math.sinh(rho / R)
        n = int(max(6, min(max_stitches, round(circ / h))))
        start = len(P)
        theta = 2.0 * math.pi * np.arange(n) / n
        excess = circ / (2.0 * math.pi * rho)     # > 1
        z = np.zeros(n)
        if excess > 1.005:
            # arclength-matching target wavenumber (small-slope):
            # m_t = (2 rho / a) sqrt(excess - 1)
            m_t = (2.0 * rho / (0.7 * h)) * math.sqrt(excess - 1.0)
            ks = []
            k = 0
            while m0 * 2 ** k <= max(m_t, m0):
                ks.append(k)
                k += 1
            for k in ks:
                mk = m0 * 2 ** k
                if mk > n // 2 - 1:
                    break
                ak = 0.7 * h if k == ks[-1] else 0.35 * h
                z = z + ak * np.sin(mk * theta)
        z = z + 0.03 * rho + rng.normal(0.0, 0.02 * h, n)  # conical bias
        for j in range(n):
            P.append([rho * math.cos(theta[j]),
                      rho * math.sin(theta[j]), float(z[j])])
        if prev is not None:
            faces += _join_rings(prev[0], prev[1], start, n)
        rings.append((start, n))
        prev = (start, n)
        if len(P) > 60000:
            break
    P = np.array(P)
    eset = set()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            eset.add((a, b) if a < b else (b, a))
    edges = list(eset)
    E0 = np.array([e[0] for e in edges])
    E1 = np.array([e[1] for e in edges])
    # true hyperbolic rest lengths from each vertex's (rho, theta)
    rr = np.linalg.norm(P[:, :2], axis=1)
    th = np.arctan2(P[:, 1], P[:, 0])
    ch = (np.cosh(rr[E0] / R) * np.cosh(rr[E1] / R)
          - np.sinh(rr[E0] / R) * np.sinh(rr[E1] / R)
          * np.cos(th[E0] - th[E1]))
    REST = R * np.arccosh(np.maximum(ch, 1.0))
    nbr = [set() for _ in range(len(P))]
    for a, b in edges:
        nbr[a].add(int(b))
        nbr[b].add(int(a))
    pin = np.arange(rings[0][1])
    tris = np.array(faces, dtype=np.int64)
    return P, E0, E1, REST, nbr, tris, pin, P[pin].copy()


def _repel(P, nbr, radius, strength):
    """Short-range self-repulsion between non-adjacent vertices, via a
    uniform spatial hash -- keeps ruffles from passing through each
    other without a full collision solver."""
    from collections import defaultdict
    keys = np.floor(P / radius).astype(np.int64)
    buckets = defaultdict(list)
    for idx in range(len(P)):
        buckets[(keys[idx, 0], keys[idx, 1], keys[idx, 2])].append(idx)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)]
    dP = np.zeros_like(P)
    r2 = radius * radius
    for idx in range(len(P)):
        kx, ky, kz = keys[idx]
        ni = nbr[idx]
        pi = P[idx]
        for dx, dy, dz in offs:
            for j in buckets.get((kx + dx, ky + dy, kz + dz), ()):
                if j <= idx or j in ni:
                    continue
                d = pi - P[j]
                l2 = float(d @ d)
                if 1e-12 < l2 < r2:
                    L = math.sqrt(l2)
                    f = strength * (radius - L) / radius * (d / L)
                    dP[idx] += f
                    dP[j] -= f
    return dP


def _relax(P, E0, E1, REST, nbr, pin, pin_pos, iters, smooth,
           repel_r, repel_s):
    """Jacobi edge-length constraints + Laplacian bending, with
    periodic self-repulsion; the inner ring is pinned."""
    n = len(P)
    valence = np.zeros(n)
    np.add.at(valence, E0, 1.0)
    np.add.at(valence, E1, 1.0)
    valence = np.maximum(valence, 1.0)[:, None]
    for it in range(iters):
        d = P[E1] - P[E0]
        L = np.linalg.norm(d, axis=1)
        L = np.where(L < 1e-9, 1e-9, L)
        corr = (1.0 - REST / L)[:, None] * d      # single-damped Jacobi
        dP = np.zeros_like(P)
        np.add.at(dP, E0, corr)
        np.add.at(dP, E1, -corr)
        P = P + dP / valence
        if smooth > 0.0:
            nsum = np.zeros_like(P)
            np.add.at(nsum, E0, P[E1])
            np.add.at(nsum, E1, P[E0])
            P = P + smooth * (nsum / valence - P)
        if repel_s > 0.0 and it % 3 == 0 and it > 0:
            P = P + _repel(P, nbr, repel_r, repel_s)
        P[pin] = pin_pos
    return P


def build_ruffle(ratio_n=6, rows=16, stitch=0.1, max_stitches=400,
                 iters=300, smooth=0.06, repel=0.5):
    P, E0, E1, REST, nbr, faces, pin, pin_pos = _crochet_mesh(
        ratio_n, rows, stitch, max_stitches)
    P = _relax(P, E0, E1, REST, nbr, pin, pin_pos, iters, smooth,
               0.4 * stitch, repel)
    return _center(P), faces


# ==========================================================================
# helpers
# ==========================================================================

def _center(V):
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    return (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)


def mean_curvature(V, faces):
    """Mean Gaussian curvature via angle defect / area (Gauss-Bonnet):
    for a constant-K surface this returns ~K. Boundary vertices are
    excluded."""
    n = len(V)
    ang = np.zeros(n)
    area = np.zeros(n)
    deg = np.zeros(n)
    tris = []
    for f in faces:
        if len(f) == 4:
            tris.append((f[0], f[1], f[2]))
            tris.append((f[0], f[2], f[3]))
        else:
            tris.append((f[0], f[1], f[2]))
    for a, b, c in tris:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            u = V[j] - V[i]
            w = V[k] - V[i]
            cs = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)
                                 + 1e-12)
            ang[i] += math.acos(max(-1.0, min(1.0, cs)))
        ar = 0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a]))
        for i in (a, b, c):
            area[i] += ar / 3.0
            deg[i] += 1
    # per-vertex Gaussian curvature K_i = angle-defect / barycentric
    # area; the MEDIAN over interior vertices is robust to the badly
    # stretched triangles some parametrisations produce near their
    # singular boundaries
    interior = deg >= 6
    defect = 2 * math.pi - ang
    ki = defect[interior] / np.maximum(area[interior], 1e-12)
    return float(np.median(ki)) if ki.size else 0.0


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

    class MESH_OT_hyperbolic_crochet_add(bpy.types.Operator):
        """Add a constant-negative-curvature surface or a ruffled
        crocheted hyperbolic plane"""
        bl_idname = "mesh.hyperbolic_crochet_add"
        bl_label = "Hyperbolic Crochet"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            items=[('RUFFLE', "Ruffled Crochet",
                    "Intrinsic hyperbolic sheet buckled into ruffles"),
                   ('PSEUDOSPHERE', "Pseudosphere (exact)",
                    "Tractricoid, K = -1"),
                   ('DINI', "Dini's Surface (exact)",
                    "Twisted pseudosphere"),
                   ('KUEN', "Kuen's Surface (exact)",
                    "K = -1 Kuen surface")],
            default='RUFFLE')
        # exact params
        ures: IntProperty(name="U Resolution", default=64, min=8,
                          max=400)
        vres: IntProperty(name="V Resolution", default=96, min=8,
                          max=400)
        # ruffle params
        ratio_n: IntProperty(
            name="Increase Ratio N", default=6, min=2, max=24,
            description="Crochet: increase 1 stitch every N. Small N = "
                        "tight curvature and dense ruffles")
        rows: IntProperty(
            name="Rows", default=16, min=3, max=30,
            description="Number of crochet rows (radial growth); "
                        "stitch count per row grows automatically")
        stitch: FloatProperty(
            name="Stitch Size", default=0.1, min=0.02, max=0.5,
            description="Size of one (square) stitch")
        max_stitches: IntProperty(
            name="Max Stitches/Row", default=400, min=24, max=2000,
            description="Cap on the (growing) stitch count per row")
        iters: IntProperty(
            name="Relax Steps", default=300, min=0, max=3000,
            description="Buckling relaxation iterations")
        smooth: FloatProperty(
            name="Bending", default=0.15, min=0.0, max=0.45,
            description="Laplacian bending: higher = larger, smoother "
                        "ruffles; lower = finer, crinklier")
        repel: FloatProperty(
            name="Self-Repulsion", default=0.5, min=0.0, max=1.0,
            description="Push apart ruffles that get too close "
                        "(reduces self-intersection)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                            max=100.0)
        shade_smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            if self.mode == 'RUFFLE':
                verts, faces = build_ruffle(
                    self.ratio_n, self.rows, self.stitch,
                    self.max_stitches, self.iters, self.smooth,
                    self.repel)
                name = "Hyperbolic Crochet"
            else:
                verts, faces = build_exact(self.mode, self.ures,
                                           self.vres)
                name = EXACT_LABEL[self.mode]
            verts = verts * self.scale
            me = bpy.data.meshes.new("HyperbolicCrochet")
            me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.shade_smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
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
            lay.prop(self, 'mode')
            if self.mode == 'RUFFLE':
                for k in ('ratio_n', 'rows', 'stitch', 'max_stitches',
                          'iters', 'smooth', 'repel'):
                    lay.prop(self, k)
            else:
                lay.prop(self, 'ures')
                lay.prop(self, 'vres')
            lay.prop(self, 'scale')
            lay.prop(self, 'shade_smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.hyperbolic_crochet_add",
                             icon='MOD_CLOTH')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_hyperbolic_crochet_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_hyperbolic_crochet_add)


EXACT_LABEL = {'PSEUDOSPHERE': "Pseudosphere", 'DINI': "Dini Surface",
               'KUEN': "Kuen Surface"}


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        # exact surfaces: mean Gaussian curvature should be ~ -1
        # (Dini with twist 0.2 -> -1/(1+0.2^2) = -0.96)
        for kind in ('PSEUDOSPHERE', 'DINI', 'KUEN'):
            V, F = build_exact(kind, 80, 80)
            # curvature computed BEFORE the fit-to-cube rescale
            fn, (u0, u1), (v0, v1), wrap = EXACT[kind]
            us = np.linspace(u0, u1, 80)
            vs = (np.linspace(v0, v1, 80, endpoint=False) if wrap
                  else np.linspace(v0, v1, 80))
            U, Vv = np.meshgrid(us, vs, indexing='ij')
            x, y, z = fn(U, Vv)
            Vraw = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
            K = mean_curvature(Vraw, F)
            print(f"{kind:13s}: V={len(V)} F={len(F)} meanK={K:+.3f}")
        # ruffled: measured on the UNSCALED mesh so curvature is
        # meaningful. Targets: strain ~ 1 (near-isometric), K ~ -1/R^2,
        # buckled (z-extent), few close non-neighbour pairs.
        import time
        h0 = 0.1
        Rc = h0 / math.log(1.0 + 1.0 / 6)
        t0 = time.time()
        V, E0, E1, REST, nbr0, F, pin, pinp = _crochet_mesh(6, 14, h0,
                                                            400)
        V = _relax(V, E0, E1, REST, nbr0, pin, pinp, 250, 0.06,
                   0.4 * h0, 0.5)
        dt = time.time() - t0
        strain = float(np.mean(np.linalg.norm(V[E1] - V[E0], axis=1)
                               / REST))
        zext = float(V[:, 2].max() - V[:, 2].min())
        K = mean_curvature(V, F)
        print(f"   (target K = -1/R^2 = {-1.0 / Rc ** 2:+.2f}, "
              f"strain={strain:.3f})")
        # self-intersection proxy: non-adjacent vertex pairs closer
        # than 0.4 * median edge length
        nbr = [set() for _ in range(len(V))]
        for f in F:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                nbr[a].add(b); nbr[b].add(a)
        med = np.median([np.linalg.norm(V[f[0]] - V[f[1]]) for f in F])
        thr = 0.4 * med
        from collections import defaultdict
        buck = defaultdict(list)
        for i, p in enumerate(V):
            buck[tuple((p / thr).astype(int))].append(i)
        close = 0
        for i, p in enumerate(V):
            kx, ky, kz = (p / thr).astype(int)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in buck.get((kx + dx, ky + dy, kz + dz), ()):
                            if j > i and j not in nbr[i] and \
                                    np.linalg.norm(p - V[j]) < thr:
                                close += 1
        finite = np.isfinite(V).all()
        ok = finite and zext > 0.05 and K < 0
        print(f"RUFFLE       : V={len(V)} F={len(F)} z_extent={zext:.3f} "
              f"meanK={K:+.2f} close_pairs={close} time={dt:.1f}s "
              f"{'OK' if ok else 'CHECK'}")
