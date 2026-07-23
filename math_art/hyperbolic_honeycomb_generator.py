
# Hyperbolic Honeycomb Generator for Blender
#
# The edge framework of a {p,q,r} honeycomb of hyperbolic 3-space,
# rendered inside the Poincare ball (after the figures in Segerman's
# "Visualizing Mathematics with 3D Printing", figs 4-21..4-23). Struts
# follow hyperbolic geodesics and thin toward the boundary sphere by
# the local conformal factor, so the infinite honeycomb appears to
# crowd into the ball's skin.
#
# The construction works in the hyperboloid model in Minkowski R^{3,1}
# (signature +,+,+,-): the four mirror normals of the {p,q,r} Coxeter
# orthoscheme are recovered from their Gram matrix by eigen-
# factorization, the honeycomb vertex is the dual point on mirrors
# 1,2,3, and the fundamental edge is that vertex plus its reflection
# in mirror 0. Enumerating the vertex-stabilizer subgroup (the finite
# symmetry group of the vertex figure {q,r}) gives the complete edge
# star of that vertex; a breadth-first walk vertex-to-vertex through
# the 1-skeleton (group elements as Lorentz matrices, deduped by
# rounded entries and positions) then sweeps out the framework, and
# (x,y,z,t) -> (x,y,z)/(1+t) drops it into the Poincare ball.
#
# Compact presets: {4,3,5}, {5,3,4}, {3,5,3}, {5,3,5}. Paracompact
# presets: {6,3,3} (ideal cell centers) and {3,3,6} (ideal vertices --
# every strut is a complete geodesic between boundary points).

bl_info = {
    "name": "Hyperbolic Honeycomb",
    "author": "David Krider (Math Art project)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Hyperbolic Honeycomb",
    "description": "{p,q,r} honeycombs of H^3 as geodesic strut "
                   "frameworks in the Poincare ball",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi, sqrt

import numpy as np


# --------------------------------------------------------------------------
# Coxeter simplex in the hyperboloid model
# --------------------------------------------------------------------------

_J = np.diag((1.0, 1.0, 1.0, -1.0))    # Minkowski metric (+,+,+,-)


def _mink(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] - a[3] * b[3]


def gram_matrix(p, q, r):
    """Gram matrix of the four mirror normals of the {p,q,r}
    orthoscheme: unit normals, adjacent mirrors meet at pi/m along
    the chain p, q, r, non-adjacent mirrors are perpendicular."""
    G = np.eye(4)
    for i, m in enumerate((p, q, r)):
        G[i, i + 1] = G[i + 1, i] = -math.cos(math.pi / m)
    return G


def classify_symbol(p, q, r, tol=1e-8):
    """SPHERICAL / EUCLIDEAN / HYPERBOLIC by the Gram signature
    ((4,0), degenerate, (3,1)); INDEFINITE for anything wilder."""
    w = np.linalg.eigvalsh(gram_matrix(p, q, r))     # ascending
    if w[0] > tol:
        return 'SPHERICAL'
    if abs(w[0]) <= tol:
        return 'EUCLIDEAN'
    if w[1] > tol:
        return 'HYPERBOLIC'
    return 'INDEFINITE'


def simplex_data(p, q, r):
    """Concrete mirror normals N (rows n_0..n_3) realizing the Gram
    matrix in R^{3,1}, plus the honeycomb vertex P_0 (the dual point
    orthogonal to mirrors 1,2,3). Returns (N, P0, ideal) where ideal
    marks a lightlike P_0 (paracompact vertex on the boundary sphere,
    normalized to t = 1). Raises ValueError for non-realizable
    symbols."""
    kind = classify_symbol(p, q, r)
    if kind != 'HYPERBOLIC':
        raise ValueError(
            f"{{{p},{q},{r}}} is {kind.lower()}, not a honeycomb of "
            "hyperbolic 3-space")
    G = gram_matrix(p, q, r)
    # eigen-factorization G = Q diag(w) Q^T with w[0] < 0 < w[1..3]:
    # spread sqrt(w) over three spacelike slots and sqrt(-w[0]) over
    # the timelike slot, so N J N^T = G exactly
    w, Q = np.linalg.eigh(G)
    N = np.empty((4, 4))
    N[:, :3] = Q[:, 1:] * np.sqrt(w[1:])
    N[:, 3] = Q[:, 0] * math.sqrt(-w[0])
    # dual basis: column i of M = inv(N J) satisfies <m_i, n_j> = d_ij,
    # so m_0 lies on mirrors 1, 2 and 3 -- the honeycomb vertex
    M = np.linalg.inv(N @ _J)
    m0 = M[:, 0]
    q0 = _mink(m0, m0)
    scale = float(m0 @ m0)
    if q0 > 1e-7 * scale:
        raise ValueError(
            f"vertex figure {{{q},{r}}} of {{{p},{q},{r}}} is "
            "hyperbolic: the honeycomb has no vertices inside H^3")
    if abs(q0) <= 1e-7 * scale:
        return N, m0 / m0[3], True          # ideal vertex, t = 1
    P0 = m0 / math.sqrt(-q0)                # <P0, P0> = -1
    if P0[3] < 0:
        P0 = -P0
    return N, P0, False


# --------------------------------------------------------------------------
# Group walk: edges of the honeycomb
# --------------------------------------------------------------------------

def _ball(x):
    """Hyperboloid (x,y,z,t) -> Poincare ball (x,y,z)/(1+t)."""
    return x[:3] / (1.0 + x[3])


def _pkey(x):
    """Position dedupe key: rounded ball-model coordinates."""
    return tuple(np.round(_ball(x), 5))


def _gkey(g):
    """Group-element dedupe key: rounded matrix entries."""
    return tuple(np.round(g, 4).ravel())


# an ideal vertex ({q,r} Euclidean, e.g. {3,3,6}) has infinitely many
# incident edges; only this many of its star are kept
_IDEAL_STAR_CAP = 24


def _vertex_star(gens, v1, ideal):
    """Elements of the vertex stabilizer <r_1, r_2, r_3> (the
    symmetry group of the vertex figure {q,r}) carrying the
    fundamental edge to each distinct edge at P_0 -- one coset
    representative per edge, found by BFS over words. The subgroup is
    finite for compact-type vertices (at most 120 elements), so the
    walk closes on its own; for ideal vertices it is infinite and the
    star is capped."""
    ident = np.eye(4)
    reps = [ident]
    pts = {_pkey(v1)}
    seen = {_gkey(ident)}
    frontier = [ident]
    cap = _IDEAL_STAR_CAP if ideal else 10 ** 9
    for _ in range(16):                # longest element of H3 is 15
        if not frontier or len(reps) >= cap:
            break
        nxt = []
        for w in frontier:
            for R in gens[1:]:
                h = w @ R
                k = _gkey(h)
                if k in seen:
                    continue
                seen.add(k)
                nxt.append(h)
                pk = _pkey(h @ v1)
                if pk not in pts and len(reps) < cap:
                    pts.add(pk)
                    reps.append(h)
        frontier = nxt
    return reps


def honeycomb_edges(p, q, r, depth=7, max_edges=15000, prune=None):
    """BFS vertex-to-vertex through the honeycomb 1-skeleton, up to
    the given graph distance from the fundamental vertex. Each
    reached vertex g P_0 emits its complete edge star g w (P_0,
    r_0 P_0); its neighbors continue the walk via g w r_0. Vertices
    and edges are deduped by rounded ball coordinates; the walk stops
    early at max_edges, and vertices outside ball radius `prune` (if
    given) are not expanded. Returns (edge list of (A, B) hyperboloid
    endpoint pairs, ideal flag)."""
    N, P0, ideal = simplex_data(p, q, r)
    # reflection in mirror i: x -> x - 2 <x, n_i> n_i
    gens = [np.eye(4) - 2.0 * np.outer(N[i], _J @ N[i])
            for i in range(4)]
    v0 = P0
    v1 = gens[0] @ P0
    star = _vertex_star(gens, v1, ideal)
    steps = [w @ gens[0] for w in star]    # g w r_0 maps P_0 -> g w v1
    edges = {}
    seen_v = {_pkey(v0)}
    frontier = [np.eye(4)]
    full = False
    for _ in range(depth):
        if full or not frontier:
            break
        nxt = []
        for g in frontier:
            a = g @ v0
            ka = _pkey(a)
            for w, st in zip(star, steps):
                b = g @ (w @ v1)
                kb = _pkey(b)
                ek = (ka, kb) if ka <= kb else (kb, ka)
                if ek not in edges:
                    edges[ek] = (a, b)
                    if len(edges) >= max_edges:
                        full = True
                        break
                if kb not in seen_v:
                    seen_v.add(kb)
                    bb = _ball(b)
                    if prune is None or float(bb @ bb) <= prune * prune:
                        nxt.append(g @ st)
            if full:
                break
        frontier = nxt
    return list(edges.values()), ideal


def sample_geodesic(A, B, segments, ideal):
    """Sample the hyperbolic geodesic between hyperboloid points A, B:
    interpolate in R^{3,1} and renormalize onto <x,x> = -1 (the
    geodesic lies in the plane spanned by A and B), then map to the
    ball. Long near-boundary edges curve properly this way. Ideal
    endpoints are lightlike, so trim the parameter slightly off 0 and
    1 (the strut has already thinned to nothing there). Returns
    (ball points, conformal factors (1-r^2)/2)."""
    lo, hi = (0.002, 0.998) if ideal else (0.0, 1.0)
    pts = []
    confs = []
    for k in range(segments + 1):
        s = lo + (hi - lo) * k / segments
        x = (1.0 - s) * A + s * B
        nrm2 = -_mink(x, x)
        x = x / math.sqrt(max(nrm2, 1e-12))
        b = _ball(x)
        pts.append((b[0], b[1], b[2]))
        confs.append((1.0 - float(b @ b)) * 0.5)
    return pts, confs


# --------------------------------------------------------------------------
# Strut / sphere mesh helpers
# --------------------------------------------------------------------------

def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _unit3(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return (v[0] / l, v[1] / l, v[2] / l)


def add_strut(verts, faces, pts, radii, sides):
    """Closed tube along a polyline with per-point radii."""
    n = len(pts)
    rings = []
    prev_n = None
    for i in range(n):
        if i == 0:
            t = _unit3(_sub3(pts[1], pts[0]))
        elif i == n - 1:
            t = _unit3(_sub3(pts[-1], pts[-2]))
        else:
            t = _unit3(_sub3(pts[i + 1], pts[i - 1]))
        if prev_n is None:
            ref = (0.0, 0.0, 1.0) if abs(t[2]) < 0.9 else (1.0, 0.0, 0.0)
            u = _unit3(_cross3(t, ref))
        else:
            # re-project previous normal for a stable frame
            d = sum(prev_n[k] * t[k] for k in range(3))
            u = _unit3(tuple(prev_n[k] - d * t[k] for k in range(3)))
        w = _cross3(t, u)
        prev_n = u
        ring = []
        for s in range(sides):
            a = 2 * pi * s / sides
            ring.append(len(verts))
            verts.append(tuple(pts[i][k]
                               + radii[i] * (cos(a) * u[k] + sin(a) * w[k])
                               for k in range(3)))
        rings.append(ring)
    for i in range(n - 1):
        r0, r1 = rings[i], rings[i + 1]
        for s in range(sides):
            s2 = (s + 1) % sides
            faces.append([r0[s], r0[s2], r1[s2], r1[s]])
    faces.append(list(reversed(rings[0])))
    faces.append(list(rings[-1]))


def add_sphere(verts, faces, center, radius, seg=8, rings=6):
    base = len(verts)
    verts.append((center[0], center[1], center[2] + radius))
    for r in range(1, rings):
        th = pi * r / rings
        for s in range(seg):
            a = 2 * pi * s / seg
            verts.append((center[0] + radius * sin(th) * cos(a),
                          center[1] + radius * sin(th) * sin(a),
                          center[2] + radius * cos(th)))
    verts.append((center[0], center[1], center[2] - radius))
    last = len(verts) - 1
    ring0 = lambda r: base + 1 + (r - 1) * seg
    for s in range(seg):
        s2 = (s + 1) % seg
        faces.append([base, ring0(1) + s2, ring0(1) + s])
    for r in range(1, rings - 1):
        for s in range(seg):
            s2 = (s + 1) % seg
            faces.append([ring0(r) + s, ring0(r) + s2,
                          ring0(r + 1) + s2, ring0(r + 1) + s])
    for s in range(seg):
        s2 = (s + 1) % seg
        faces.append([last, ring0(rings - 1) + s, ring0(rings - 1) + s2])


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build_honeycomb(p=4, q=3, r=5, depth=7, max_edges=15000,
                    cutoff=0.97, thickness=0.07, sides=6,
                    arc_segments=8, node_spheres=True,
                    sphere_factor=1.6, ball_radius=1.0):
    """Mesh data for the {p,q,r} framework. Edges whose hyperbolic
    midpoint lands outside ball radius `cutoff` are culled; strut
    radii scale with the local conformal factor (1-r^2)/2, which is
    what makes the framework thin toward the boundary sphere.
    Returns (verts, faces, node count, strut count)."""
    edge_list, ideal = honeycomb_edges(p, q, r, depth, max_edges,
                                       prune=cutoff)
    verts = []
    faces = []
    nodes = {}
    floor = 1e-4 * ball_radius
    kept = 0
    for (A, B) in edge_list:
        mid = A + B                       # timelike: normalize, map
        mid = mid / math.sqrt(-_mink(mid, mid))
        bm = _ball(mid)
        if float(bm @ bm) > cutoff * cutoff:
            continue
        pts, confs = sample_geodesic(A, B, arc_segments, ideal)
        spts = [tuple(c * ball_radius for c in pt) for pt in pts]
        radii = [max(thickness * c * ball_radius, floor)
                 for c in confs]
        add_strut(verts, faces, spts, radii, sides)
        kept += 1
        for P in (A, B):
            b = _ball(P)
            nodes.setdefault(tuple(np.round(b, 5)), b)
    if node_spheres and not ideal:
        for b in nodes.values():
            r2 = float(b @ b)
            if r2 > cutoff * cutoff:
                continue
            rad = max(thickness * (1.0 - r2) * 0.5 * ball_radius,
                      floor) * sphere_factor
            add_sphere(verts, faces,
                       tuple(c * ball_radius for c in b), rad)
    return verts, faces, len(nodes), kept


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, IntProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


PRESETS = {
    'H435': (4, 3, 5),
    'H534': (5, 3, 4),
    'H353': (3, 5, 3),
    'H535': (5, 3, 5),
    'H633': (6, 3, 3),
    'H336': (3, 3, 6),
}


if _IN_BLENDER:

    class MESH_OT_hyperbolic_honeycomb_add(bpy.types.Operator):
        """{p,q,r} honeycomb of hyperbolic 3-space as a geodesic
        strut framework in the Poincare ball, thinning toward the
        boundary sphere"""
        bl_idname = "mesh.hyperbolic_honeycomb_add"
        bl_label = "Hyperbolic Honeycomb"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Honeycomb",
            items=[('H435', "{4,3,5} Order-5 Cubic",
                    "Cubes, five around each edge (compact)"),
                   ('H534', "{5,3,4} Order-4 Dodecahedral",
                    "Dodecahedra, four around each edge (compact)"),
                   ('H353', "{3,5,3} Icosahedral",
                    "Icosahedra, three around each edge (compact)"),
                   ('H535', "{5,3,5} Order-5 Dodecahedral",
                    "Dodecahedra, five around each edge (compact)"),
                   ('H633', "{6,3,3} Hexagonal Tiling Cells",
                    "Paracompact: cells are {6,3} planes with ideal "
                    "centers on the boundary sphere"),
                   ('H336', "{3,3,6} Order-6 Tetrahedral",
                    "Paracompact: ideal vertices, every strut a "
                    "complete geodesic between boundary points"),
                   ('CUSTOM', "Custom {p,q,r}",
                    "Any hyperbolic Schlafli symbol (validated)")],
            default='H435')
        p: IntProperty(name="p", default=4, min=3, max=12,
                       description="Cell face size (cells are {p,q})")
        q: IntProperty(name="q", default=3, min=3, max=12,
                       description="Faces per cell vertex")
        r: IntProperty(name="r", default=5, min=3, max=12,
                       description="Cells around each edge")
        depth: IntProperty(
            name="Word Depth", default=7, min=1, max=12,
            description="Reflection word length of the group walk "
                        "(more depth, more edges, deeper into the "
                        "ball's skin)")
        max_edges: IntProperty(
            name="Edge Cap", default=15000, min=10, max=200000,
            description="Stop the group walk early past this many "
                        "edges")
        cutoff: FloatProperty(
            name="Radius Cutoff", default=0.97, min=0.5, max=0.999,
            precision=3,
            description="Cull struts whose midpoint lies outside "
                        "this ball radius")
        thickness: FloatProperty(
            name="Strut Thickness", default=0.07, min=0.005, max=0.5,
            step=1, precision=3,
            description="Strut radius before the conformal thinning "
                        "(a strut at the ball center gets half this)")
        sides: IntProperty(name="Strut Sides", default=6, min=3,
                           max=16)
        arc_segments: IntProperty(
            name="Arc Segments", default=8, min=1, max=32,
            description="Samples per geodesic strut (long "
                        "near-boundary edges need the curvature)")
        node_spheres: BoolProperty(
            name="Node Spheres", default=True,
            description="Spheres at the honeycomb vertices, scaled "
                        "by the same conformal factor (skipped for "
                        "ideal-vertex honeycombs like {3,3,6})")
        sphere_factor: FloatProperty(name="Sphere Size", default=1.6,
                                     min=1.0, max=4.0)
        ball_radius: FloatProperty(name="Ball Radius", default=1.0,
                                   min=0.01, max=100.0,
                                   description="Overall scale: radius "
                                               "of the Poincare ball")

        def execute(self, context):
            if self.preset == 'CUSTOM':
                p, q, r = self.p, self.q, self.r
            else:
                p, q, r = PRESETS[self.preset]
            try:
                verts, faces, nn, ne = build_honeycomb(
                    p, q, r, self.depth, self.max_edges, self.cutoff,
                    self.thickness, self.sides, self.arc_segments,
                    self.node_spheres, self.sphere_factor,
                    self.ball_radius)
            except ValueError as err:
                self.report({'ERROR'}, str(err))
                return {'CANCELLED'}
            me = bpy.data.meshes.new("HyperbolicHoneycomb")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("HyperbolicHoneycomb", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{{{p},{q},{r}}}: {nn} vertices, "
                        f"{ne} edges")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'CUSTOM':
                col = lay.column(align=True)
                for k in ('p', 'q', 'r'):
                    col.prop(self, k)
            lay.prop(self, 'depth')
            lay.prop(self, 'max_edges')
            lay.prop(self, 'cutoff')
            col = lay.column(align=True)
            for k in ('thickness', 'sides', 'arc_segments'):
                col.prop(self, k)
            lay.prop(self, 'node_spheres')
            if self.node_spheres:
                lay.prop(self, 'sphere_factor')
            lay.prop(self, 'ball_radius')

    def _menu_func(self, context):
        self.layout.operator("mesh.hyperbolic_honeycomb_add",
                             icon='MESH_UVSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_hyperbolic_honeycomb_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_hyperbolic_honeycomb_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for name, (p, q, r) in PRESETS.items():
            E, ideal = honeycomb_edges(p, q, r, depth=6)
            print(f"{{{p},{q},{r}}}: {len(E):5d} edges at depth 6"
                  f"{'  (ideal vertices)' if ideal else ''}")
