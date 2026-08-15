# Honeycombs of hyperbolic 3-space.
#
# Part of the Math Art hyperbolic engine (`math_art/hyperbolic/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# The regular honeycombs {p,q,r} of H^3 -- four with finite cells and
# finite vertex figures, and the paracompact ones whose cells or vertex
# figures are horospherical.  Rendered in the Poincare ball, where the
# faces are spherical caps meeting the boundary sphere orthogonally.
#
# References:
# - H. S. M. Coxeter, "Regular Honeycombs in Hyperbolic Space",
#   Proceedings of the International Congress of Mathematicians, 1954.

import math
from math import cos, sin, pi, sqrt
import numpy as np


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
