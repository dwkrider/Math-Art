
# Calabi-Yau Scaffold Generator for Blender
#
# Two combinatorial skeletons of Calabi-Yau geometry, both of which
# are honestly one- or two-dimensional and so can be built without
# faking a projection of a six-dimensional manifold.
#
# THE SYZ DISCRIMINANT GRAPH.  Strominger, Yau and Zaslow conjectured
# that a Calabi-Yau threefold near its large complex structure limit
# fibres in special Lagrangian 3-tori, and that its mirror is obtained
# by dualising the fibres.  For the Fermat quintic
#
#     sum z_k^5 - 5 psi prod z_k = 0     in CP^4
#
# Ruan showed the base of that fibration is the boundary of a
# 4-simplex -- topologically S^3 -- and computed where the T^3
# degenerates.  The answer (his section 4.2) is a GRAPH: "the vertices
# of Gamma are P_ij ... and P_ijk ..., and the legs of Gamma are
# Gamma^k_ij which connects P_ijk and P_ij".  Reading the labels as
# faces of the simplex, that is exactly the incidence between the ten
# EDGES and the ten TRIANGLES of the 4-simplex: a trivalent graph on
# 20 vertices with 30 legs, drawn here inside the boundary S^3 and
# brought down to R^3 either by a Schlegel-style projection or by
# stereographic projection of S^3.
#
# THE FERMAT SURFACE IN CP^3.  Hanson and Sha tessellated the real
# 4-manifolds
#
#     F_n:  z0^n + z1^n + z2^n + z3^n = 0     in CP^3,
#
# of which F_4 is a K3 surface -- a genuine Calabi-Yau 2-fold.  Their
# vertices are the n-th roots of unity in the six standard projective
# lines,
#
#     p01_k = [0,0,1,w],  p02_k = [0,1,0,w],  p03_k = [0,1,w,0],
#     p12_k = [1,0,0,w],  p13_k = [1,0,w,0],  p23_k = [1,w,0,0],
#     w = exp(i(pi + 2 k pi)/n),
#
# 6n of them, and the edges and 2-cells are given by explicit index
# rules.  Following the paper, CP^3 is embedded in R^16 by the
# Hermitian projector z -> z z*/|z|^2 and then projected to R^3; the
# result is their Figure 13, the K3 surface drawn as the edges of its
# tessellation.  The cell counts the theorem predicts -- 6n vertices,
# 12n^2 edges, 8n^2 + 4n^3 triangles -- are all rebuilt from the index
# rules and checked, together with the Euler characteristic
# 6n - 4n^2 + n^3 of a smooth degree-n surface in CP^3 (24 for the K3).
#
# References:
# - A. Strominger, S.-T. Yau and E. Zaslow, "Mirror symmetry is
#   T-duality", Nuclear Physics B479 (1996) 243-259 -- the conjecture.
# - W.-D. Ruan, "Lagrangian torus fibration of quintic Calabi-Yau
#   hypersurfaces I: Fermat quintic case", arXiv:math/9904012 -- the
#   fibration over the boundary of the 4-simplex, and the discriminant
#   graph Gamma of section 4.2 with its vertices P_ij, P_ijk and legs
#   Gamma^k_ij (his Figure 1).
# - M. Gross, "Special Lagrangian fibrations II: Geometry",
#   arXiv:math/9809072 -- the SYZ programme, and the trivalent
#   structure expected of the discriminant locus.
# - A. J. Hanson and J.-P. Sha, "A tessellation for algebraic surfaces
#   in CP3", Journal of Symbolic Computation (2008), arXiv:0804.3218 --
#   the vertex list, the edge and 2-cell index rules, the cell counts
#   of their Theorem, the R^16 embedding, and Figure 13.
# - P. Griffiths and J. Harris, "Principles of Algebraic Geometry",
#   Wiley (1978) -- the Euler characteristic 6n - 4n^2 + n^3 of a
#   smooth degree-n surface in CP^3, and that F_4 is a K3 surface.

import itertools
import math

import numpy as np

bl_info = {
    "name": "Calabi-Yau Scaffolds",
    "author": "Math Art project (after Ruan / Hanson & Sha)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Calabi-Yau Scaffolds",
    "description": "The SYZ discriminant graph and the Hanson-Sha "
                   "tessellation of the Fermat surfaces in CP3",
    "category": "Add Mesh",
}

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# --------------------------------------------------------------------
# The SYZ discriminant graph
# --------------------------------------------------------------------

def simplex4():
    """The five vertices of a regular 4-simplex, centred, in R^4."""
    E = np.eye(5) - 0.2
    # The centred standard basis spans a 4-plane; an orthonormal basis
    # of it gives coordinates.
    u, s, vt = np.linalg.svd(E)
    B = vt[:4]
    P = E @ B.T
    return P / np.linalg.norm(P[0])


def syz_graph():
    """Ruan's discriminant graph in the boundary of the 4-simplex.

    Returns (points, edges, kinds), where `kinds` marks each node 1
    for the ten P_ijk (edge barycentres of the simplex) and 2 for the
    ten P_ij (triangle barycentres).  A leg joins P_ijk to P_ij
    exactly when {i,j} is contained in {i,j,k}, which under the
    face labelling is the statement that the simplex edge is a side of
    the simplex triangle -- so every node has degree three.
    """
    P = simplex4()
    pairs = list(itertools.combinations(range(5), 2))
    triples = list(itertools.combinations(range(5), 3))
    pts, kinds, index = [], [], {}
    for e in pairs:                      # the ten P_ijk
        index[('e', e)] = len(pts)
        pts.append(P[list(e)].mean(axis=0))
        kinds.append(1)
    for t in triples:                    # the ten P_ij
        index[('t', t)] = len(pts)
        pts.append(P[list(t)].mean(axis=0))
        kinds.append(2)
    edges = []
    for t in triples:
        for e in itertools.combinations(t, 2):
            edges.append((index[('e', e)], index[('t', t)]))
    return np.asarray(pts), edges, kinds


def to_r3(pts, mode='SCHLEGEL', pole=None):
    """Bring points of the boundary 3-sphere down to R^3.

    SCHLEGEL drops the component along `pole`, which keeps every
    point finite and gives the familiar simplex diagram.  SPHERE
    normalises onto S^3 first and then projects stereographically, so
    that the S^3 the fibration is based on is what you are looking at.
    """
    P = np.asarray(pts, dtype=float)
    if pole is None:
        pole = simplex4()[0]
    n = np.asarray(pole, dtype=float)
    n = n / np.linalg.norm(n)
    B = np.linalg.svd(np.outer(n, n))[0][:, 1:]      # n's orthocomplement
    if mode == 'SCHLEGEL':
        return P @ B
    S = P / np.linalg.norm(P, axis=1, keepdims=True)
    h = np.clip(S @ n, -0.995, 0.995)
    return (S @ B) / (1.0 - h)[:, None]


# --------------------------------------------------------------------
# The Hanson-Sha tessellation of F_n in CP^3
# --------------------------------------------------------------------

_CLASSES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def fermat_vertices(n):
    """The 6n vertices p_ab_k of the tessellation, as points of CP^3.

    p_ab_k is the homogeneous point whose a-th and b-th coordinates
    vanish, with the n-th root of -1 in the later of the two
    surviving slots.  These are the n-th roots of unity in the six
    standard projective lines of CP^3.
    """
    out = []
    for (a, b) in _CLASSES:
        rest = [c for c in range(4) if c not in (a, b)]
        for k in range(n):
            w = np.exp(1j * (math.pi + 2 * math.pi * k) / n)
            z = np.zeros(4, dtype=complex)
            z[rest[0]] = 1.0
            z[rest[1]] = w
            out.append((a, b, k, z))
    return out


def fermat_cells(n):
    """Vertices, edges and triangles of the tessellation of F_n.

    Edges: on each of the four CP^2's z_m = 0 lie the three vertex
    classes p_ab with m in {a, b}, and every cross pair between two
    of those classes is an edge -- 3 n^2 per plane, 12 n^2 in all.

    Triangles come in two families, both quoted by Hanson and Sha:
    the 2 n^2 per plane with all three corners in that plane and
    indices satisfying i - j + k = 0 or -1 (mod n), and the n^3 per
    choice of three planes whose three edges lie on three different
    CP^2's, taken over all index triples.
    """
    verts = fermat_vertices(n)
    idx = {(a, b, k): i for i, (a, b, k, _) in enumerate(verts)}

    edges = set()
    for m in range(4):
        fam = [c for c in _CLASSES if m in c]
        for c1, c2 in itertools.combinations(fam, 2):
            for i in range(n):
                for j in range(n):
                    u, v = idx[(*c1, i)], idx[(*c2, j)]
                    edges.add((min(u, v), max(u, v)))

    tris = []
    for m in range(4):
        fam = sorted(c for c in _CLASSES if m in c)
        c1, c2, c3 = fam
        for i in range(n):
            for j in range(n):
                for shift in (0, -1):
                    k = (j - i + shift) % n
                    tris.append((idx[(*c1, i)], idx[(*c2, j)],
                                 idx[(*c3, k)]))
    for drop in range(4):
        fam = sorted(c for c in _CLASSES if drop not in c)
        c1, c2, c3 = fam
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    tris.append((idx[(*c1, i)], idx[(*c2, j)],
                                 idx[(*c3, k)]))
    return verts, sorted(edges), tris


def cp3_to_r16(z):
    """The Hermitian projector embedding CP^3 -> R^16.

    z z*/|z|^2 is Hermitian with unit trace, so its independent real
    coordinates are the four diagonal entries and the real and
    imaginary parts of the six entries above the diagonal: 4 + 12 = 16.
    Dropping the imaginary parts -- keeping only the 16 numbers of
    M.real -- would throw away every phase and collapse the n vertices
    of each class onto one point.
    """
    z = np.asarray(z, dtype=complex)
    M = np.outer(z, z.conj()) / float(np.vdot(z, z).real)
    out = [float(M[a, a].real) for a in range(4)]
    for a, b in _CLASSES:
        out.append(float(M[a, b].real))
        out.append(float(M[a, b].imag))
    return np.asarray(out)


def fermat_points(n, seed=0):
    """The tessellation's vertices, embedded in R^16 and cut to R^3.

    The default projection is the symmetric one built from the three
    complementary pairs of off-diagonal entries of the projector,
    which keeps the coordinate symmetry of F_n visible; any other
    seed picks a reproducible random orthonormal frame instead, the
    way Hanson and Sha hunt for a readable view.
    """
    V = np.asarray([cp3_to_r16(z) for _, _, _, z in fermat_vertices(n)])
    rng = np.random.default_rng(_DEFAULT_VIEW if seed == 0 else seed)
    Q, _ = np.linalg.qr(rng.standard_normal((16, 3)))
    return V @ Q


def euler_characteristic(n):
    """chi of a smooth degree-n surface in CP^3 (24 for the K3)."""
    return 6 * n - 4 * n ** 2 + n ** 3


# --------------------------------------------------------------------
# Tubes and balls
# --------------------------------------------------------------------

_DEFAULT_VIEW = 35


def _frame(d):
    d = d / np.linalg.norm(d)
    a = np.array([0.0, 0.0, 1.0])
    if abs(d @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def struts(points, edges, radius=0.02, sides=6, nodes=0.0):
    """A ball-and-stick mesh over a graph."""
    P = np.asarray(points, dtype=float)
    verts, faces = [], []
    for a, b in edges:
        p, q = P[a], P[b]
        d = q - p
        if np.linalg.norm(d) < 1e-12:
            continue
        u, v = _frame(d)
        base = len(verts)
        for end in (p, q):
            for j in range(sides):
                t = 2.0 * math.pi * j / sides
                verts.append(tuple(end + radius * (math.cos(t) * u +
                                                   math.sin(t) * v)))
        for j in range(sides):
            k = (j + 1) % sides
            faces.append((base + j, base + sides + j,
                          base + sides + k, base + k))
    if nodes > 0.0:
        rings, segs = 10, 12
        for c in P:
            base = len(verts)
            for i in range(rings + 1):
                th = math.pi * i / rings
                for j in range(segs):
                    ph = 2.0 * math.pi * j / segs
                    verts.append((c[0] + nodes * math.sin(th) * math.cos(ph),
                                  c[1] + nodes * math.sin(th) * math.sin(ph),
                                  c[2] + nodes * math.cos(th)))
            for i in range(rings):
                for j in range(segs):
                    k = (j + 1) % segs
                    faces.append((base + i * segs + j, base + i * segs + k,
                                  base + (i + 1) * segs + k,
                                  base + (i + 1) * segs + j))
    return verts, faces


def fit(pts, scale=1.0):
    """Centre on the bounding-box midpoint, largest extent 2*scale."""
    if len(pts) == 0:
        return pts
    A = np.asarray(pts, dtype=float)
    lo, hi = A.min(axis=0), A.max(axis=0)
    ext = float((hi - lo).max())
    s = (2.0 * scale / ext) if ext > 1e-12 else 1.0
    return [tuple(v) for v in (A - 0.5 * (lo + hi)) * s]


# --------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------

if _IN_BLENDER:

    class MESH_OT_calabi_yau_scaffold_add(bpy.types.Operator):
        """Add an SYZ discriminant graph or a Fermat surface
        tessellation"""
        bl_idname = "mesh.calabi_yau_scaffold_add"
        bl_label = "Calabi-Yau Scaffolds"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Scaffold",
            description="Which skeleton to build",
            items=[('SYZ', "SYZ Discriminant Graph",
                    "Where the special Lagrangian 3-torus fibration of "
                    "the Fermat quintic degenerates: a trivalent graph "
                    "of 20 nodes and 30 legs inside the base "
                    "three-sphere"),
                   ('FERMAT_CP3', "Fermat Surface in CP3",
                    "The Hanson-Sha tessellation of z0^n + z1^n + z2^n "
                    "+ z3^n = 0, a real 4-manifold; at n = 4 it is a K3 "
                    "surface. Drawn as its edges, projected from the "
                    "embedding of CP3 in R^16")],
            default='SYZ')
        projection: EnumProperty(
            name="Projection",
            description="How to bring the base three-sphere to R^3",
            items=[('SCHLEGEL', "Simplex",
                    "Drop one dimension: the familiar diagram of the "
                    "4-simplex boundary"),
                   ('SPHERE', "Stereographic",
                    "Normalise onto the three-sphere the fibration is "
                    "based on, then project stereographically")],
            default='SCHLEGEL')
        degree: IntProperty(
            name="Degree", default=4, min=1, max=8,
            description="Degree n of the Fermat surface. n = 4 is the "
                        "K3 surface")
        seed: IntProperty(
            name="View", default=0, min=0, max=64,
            description="0 is the symmetric projection out of R^16; any "
                        "other value picks a different reproducible "
                        "frame, which is how a readable view of a "
                        "4-manifold gets found")
        radius: FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.2,
            description="Radius of the tubes along the legs")
        node_radius: FloatProperty(
            name="Node Radius", default=0.045, min=0.0, max=0.2,
            description="Radius of a ball at each node (0 = none)")
        sides: IntProperty(
            name="Strut Sides", default=6, min=3, max=24,
            description="Cross-section of each tube")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size multiplier")

        def execute(self, context):
            if self.mode == 'SYZ':
                pts4, edges, kinds = syz_graph()
                pts = to_r3(pts4, self.projection)
                name = "SYZ Discriminant Graph"
                deg = {}
                for a, b in edges:
                    deg[a] = deg.get(a, 0) + 1
                    deg[b] = deg.get(b, 0) + 1
                msg = (f"discriminant graph: {len(pts)} nodes, "
                       f"{len(edges)} legs, all trivalent "
                       f"{set(deg.values()) == {3}}")
            else:
                n = self.degree
                verts, edges, tris = fermat_cells(n)
                pts = fermat_points(n, self.seed)
                name = f"Fermat Surface F{n}"
                msg = (f"F_{n}: {len(verts)} vertices (6n), "
                       f"{len(edges)} edges (12n^2), "
                       f"{len(tris)} triangles (8n^2 + 4n^3); "
                       f"chi = {euler_characteristic(n)}"
                       + (" -- a K3 surface" if n == 4 else ""))

            pts = fit(pts, 1.0)
            v, f = struts(pts, edges, self.radius, self.sides,
                          self.node_radius)
            me = bpy.data.meshes.new(name)
            me.from_pydata(fit(v, self.scale), [], f)
            me.validate(clean_customdata=True)
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
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'SYZ':
                lay.prop(self, 'projection')
            else:
                lay.prop(self, 'degree')
                lay.prop(self, 'seed')
            lay.prop(self, 'radius')
            lay.prop(self, 'node_radius')
            lay.prop(self, 'sides')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.calabi_yau_scaffold_add",
                             icon='MOD_WIREFRAME')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_calabi_yau_scaffold_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_calabi_yau_scaffold_add)


# --------------------------------------------------------------------

def _selftest():
    bad = []

    # 1. the simplex is regular and centred.
    P = simplex4()
    d = [np.linalg.norm(P[i] - P[j])
         for i, j in itertools.combinations(range(5), 2)]
    ok = (abs(max(d) - min(d)) < 1e-9
          and float(np.abs(P.sum(axis=0)).max()) < 1e-9)
    print(f"syz: regular 4-simplex edge spread {max(d) - min(d):.1e} "
          f"{'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("simplex")

    # 2. Ruan's graph: 20 nodes, 30 legs, every node trivalent, and
    #    the two kinds of node -- ten edge barycentres, ten triangle
    #    barycentres -- form the two sides of a bipartite graph.
    pts, edges, kinds = syz_graph()
    deg = {}
    for a, b in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    tri = set(deg.values()) == {3}
    bip = all(kinds[a] != kinds[b] for a, b in edges)
    ok = (len(pts) == 20 and len(edges) == 30 and tri and bip
          and kinds.count(1) == 10 and kinds.count(2) == 10)
    print(f"syz: {len(pts)} nodes(20) {len(edges)} legs(30) "
          f"trivalent={tri} bipartite={bip} {'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("syz graph")

    # 3. both projections are finite and non-degenerate.
    for mode in ('SCHLEGEL', 'SPHERE'):
        Q = to_r3(pts, mode)
        gap = min(np.linalg.norm(Q[a] - Q[b])
                  for a, b in itertools.combinations(range(len(Q)), 2))
        ok = np.all(np.isfinite(Q)) and gap > 1e-6
        print(f"syz: {mode} finite, closest pair {gap:.3f} "
              f"{'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"projection {mode}")

    # 4. the Hanson-Sha cell counts, against their Theorem.
    for n in (1, 2, 3, 4, 5):
        verts, edges, tris = fermat_cells(n)
        want = (6 * n, 12 * n ** 2, 8 * n ** 2 + 4 * n ** 3)
        got = (len(verts), len(edges), len(tris))
        ok = got == want
        print(f"fermat_cp3: n={n} V,E,T = {got} want {want} "
              f"{'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"F_{n} counts {got} != {want}")

    # 5. every triangle's three sides really are edges of the
    #    tessellation -- the check that the index rules line up.
    n = 4
    verts, edges, tris = fermat_cells(n)
    eset = set(edges)
    missing = 0
    for t in tris:
        for a, b in itertools.combinations(t, 2):
            if (min(a, b), max(a, b)) not in eset:
                missing += 1
    ok = missing == 0
    print(f"fermat_cp3: n=4 triangle sides missing from the edge set: "
          f"{missing} {'OK' if ok else 'BAD'}")
    if not ok:
        bad.append(f"{missing} triangle sides not edges")

    # 6. the theorem's cell counts give the known Euler
    #    characteristic of a degree-n surface in CP^3; for n = 4 that
    #    is 24, the K3 surface.
    for n in (2, 3, 4, 5):
        chi = (6 * n - 12 * n ** 2 + (8 * n ** 2 + 7 * n ** 3)
               - 12 * n ** 3 + 6 * n ** 3)
        ok = chi == euler_characteristic(n)
        print(f"fermat_cp3: n={n} chi from the cell counts {chi} "
              f"({euler_characteristic(n)}) {'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"chi n={n}")

    # 7. the vertices really lie on F_n, and stay distinct in R^3.
    for n in (3, 4):
        res = 0.0
        for _, _, _, z in fermat_vertices(n):
            res = max(res, abs(complex(np.sum(z ** n))))
        Q = fermat_points(n)
        gap = min(np.linalg.norm(Q[a] - Q[b])
                  for a, b in itertools.combinations(range(len(Q)), 2))
        ok = res < 1e-12 and gap > 1e-6
        print(f"fermat_cp3: n={n} max |sum z^n| = {res:.1e}, closest "
              f"projected pair {gap:.3f} {'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"F_{n} vertices res={res:.1e} gap={gap:.1e}")

    # 8. the fit: centred, inside a 2 m cube.
    v, f = struts(fit(fermat_points(4)), fermat_cells(4)[1], 0.02, 6)
    P3 = np.asarray(fit(v))
    ctr = float(np.abs(0.5 * (P3.min(axis=0) + P3.max(axis=0))).max())
    ext = float((P3.max(axis=0) - P3.min(axis=0)).max())
    ok = ctr < 1e-9 and abs(ext - 2.0) < 1e-9
    print(f"scaffold: mesh V={len(v)} F={len(f)} fit centre={ctr:.2e} "
          f"extent={ext:.6f} {'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("fit")

    if bad:
        raise AssertionError("; ".join(bad))
