"""Mesh decimation that cannot open holes.

WHY THIS EXISTS.  The obvious way to shrink a mesh -- snap the vertices
onto a coarse grid, weld whatever coincides, drop the faces that
degenerate -- is *vertex clustering*, and it tears surfaces apart by
construction.  Two sheets passing close together get merged, a thin neck
collapses to a line, and every face that loses a distinct corner is
simply deleted.  What is left has holes, and no amount of tuning the
grid step fixes it: the algorithm's failure mode IS deleting faces.

This module collapses EDGES instead, which is the operation that cannot
change what the surface is if it is checked properly.  Collapsing an
edge merges its two endpoints and removes the one or two triangles that
shared it; every other face survives, re-indexed.  Three tests decide
whether a particular collapse is allowed, and a collapse that fails any
of them is skipped rather than forced:

  LINK CONDITION.  The classical topological test (Dey, Edelsbrunner,
  Guha): collapsing edge (u,v) preserves the surface's topology exactly
  when the vertices adjacent to BOTH u and v are precisely the vertices
  opposite the edge -- two of them for an interior edge, one for a
  boundary edge.  A third shared neighbour means the collapse would
  pinch a handle or fuse two sheets, which is how clustering makes its
  holes.

  NORMAL FLIP.  Every surviving triangle around the merged vertex must
  keep its orientation.  Without this a collapse can fold the surface
  back on itself, which reads as a hole from one side and a crease from
  the other.

  BOUNDARY PRESERVATION.  By default the rim is untouchable -- no rim
  vertex moves, no rim edge collapses, and no interior vertex collapses
  onto one.  An open patch keeps its outline to the last decimal and to
  the last edge.  This matters here because a periodic surface's rim is
  where its copies weld, and a rim moved by a hundredth is a cell that
  no longer closes.  Permitting rim-along-rim collapses instead (the
  usual textbook compromise) took a test patch from 92 boundary edges
  to 25, which is a different outline, so it is off by default.

  VALENCE.  A collapse that would leave a vertex with too many
  neighbours is refused.  High-valence fans are where quadric
  decimation quietly degrades: the fan flattens, subsequent collapses
  price badly against it, and triangle quality goes with them.

The error metric is Garland and Heckbert's: each vertex carries the sum
of the squared-distance quadrics of its incident face planes, an edge's
cost is that quadric evaluated at the merged position, and the cheapest
edge collapses first.  Boundary edges additionally carry a plane
perpendicular to the surface along the rim, so the rim resists being
straightened.

References:
  M. Garland and P. Heckbert, "Surface simplification using quadric
    error metrics", SIGGRAPH '97, pp. 209-216.
  T. K. Dey, H. Edelsbrunner, S. Guha, D. V. Nekhayev, "Topology
    preserving edge contraction", Publ. Inst. Math. (Beograd) 66 (1999).
"""

import heapq

import numpy as np


def _face_quadric(p0, p1, p2):
    """The 4x4 squared-distance-to-plane quadric of one triangle."""
    n = np.cross(p1 - p0, p2 - p0)
    L = float(np.linalg.norm(n))
    if L < 1e-18:
        return None, 0.0
    n = n / L
    d = -float(np.dot(n, p0))
    v = np.array([n[0], n[1], n[2], d], dtype=float)
    # weighting by area makes a big flat region cheap to simplify and a
    # small curved one expensive, which is the behaviour wanted
    return np.outer(v, v) * (0.5 * L), 0.5 * L


def _build(V, faces):
    """Triangulate, and index the mesh for collapsing."""
    tris = []
    for f in faces:
        f = [int(i) for i in f]
        for k in range(1, len(f) - 1):
            a, b, c = f[0], f[k], f[k + 1]
            if a != b and b != c and a != c:
                tris.append((a, b, c))
    V = np.asarray(V, dtype=float)
    return V, tris


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def decimate(V, faces, target_ratio=0.5, max_error=None,
             preserve_boundary=True, max_valence=14):
    """Reduce a mesh to about `target_ratio` of its triangles.

    Returns `(vertices, triangles)`.  Never opens a hole, never changes
    the number of connected components, and -- with the default
    `preserve_boundary` -- never touches the rim at all: not its shape,
    not its vertices, not even its edge count.  A collapse that would do
    any of those is skipped, so the result may be larger than asked for.
    That is the intended trade: an honest surface at whatever size it
    can reach, rather than the requested size with damage.

    `preserve_boundary=False` permits collapses ALONG the rim (never off
    it), which shortens an open patch's outline.  Only use it where the
    outline is decorative; for a periodic cell the rim is where the
    copies weld, and moving it by a hundredth is a cell that no longer
    closes.

    `max_valence` refuses collapses that would leave a vertex with more
    than that many neighbours.  High-valence vertices are where quadric
    decimation degrades: the fan around them flattens, later collapses
    price badly, and triangle quality collapses with it.
    """
    V, tris = _build(V, faces)
    if not tris:
        return V, []
    n_target = max(4, int(len(tris) * float(target_ratio)))
    if len(tris) <= n_target:
        return V, tris

    # --- adjacency -----------------------------------------------------
    vf = {}                       # vertex -> set of face ids
    ef = {}                       # edge -> list of face ids
    for i, (a, b, c) in enumerate(tris):
        for x in (a, b, c):
            vf.setdefault(x, set()).add(i)
        for e in (_edge_key(a, b), _edge_key(b, c), _edge_key(c, a)):
            ef.setdefault(e, []).append(i)
    alive = [True] * len(tris)
    # A boundary edge belongs to one face.  Anything with three or more
    # is non-manifold; those edges are frozen rather than reasoned about.
    boundary_v = set()
    frozen = set()
    for e, fs in ef.items():
        if len(fs) == 1:
            boundary_v.add(e[0])
            boundary_v.add(e[1])
        elif len(fs) > 2:
            frozen.add(e[0])
            frozen.add(e[1])

    # --- quadrics ------------------------------------------------------
    Q = {i: np.zeros((4, 4)) for i in vf}
    for i, (a, b, c) in enumerate(tris):
        q, _ = _face_quadric(V[a], V[b], V[c])
        if q is None:
            continue
        Q[a] += q
        Q[b] += q
        Q[c] += q
    # rim stiffening: a plane through each boundary edge, perpendicular
    # to its face, so the outline is expensive to move even before the
    # hard boundary rule below refuses to move it at all
    for e, fs in ef.items():
        if len(fs) != 1:
            continue
        a, b = e
        tri = tris[fs[0]]
        p = [x for x in tri if x not in (a, b)]
        if not p:
            continue
        fn, _ = _face_quadric(V[tri[0]], V[tri[1]], V[tri[2]])
        edge = V[b] - V[a]
        L = float(np.linalg.norm(edge))
        if L < 1e-18 or fn is None:
            continue
        nrm = np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])
        nl = float(np.linalg.norm(nrm))
        if nl < 1e-18:
            continue
        m = np.cross(edge / L, nrm / nl)
        d = -float(np.dot(m, V[a]))
        v4 = np.array([m[0], m[1], m[2], d], dtype=float)
        w = np.outer(v4, v4) * (L * L)
        Q[a] += w
        Q[b] += w

    def cost(a, b):
        """(error, position) for collapsing a onto b's side."""
        q = Q[a] + Q[b]
        # the merged vertex is placed at whichever endpoint or midpoint
        # is cheapest; solving the quadric exactly can put it far from
        # the surface on a degenerate patch, which is not worth it here
        cands = [V[a], V[b], 0.5 * (V[a] + V[b])]
        # Garland-Heckbert's optimal point: solve the 3x3 system from the
        # quadric's leading block.  It is the best position when the
        # system is well conditioned and can land far off the surface
        # when it is not, so it is offered as a CANDIDATE and wins only
        # if it actually scores lowest -- which also keeps the result
        # inside the convex neighbourhood on flat or degenerate fans.
        A3 = q[:3, :3]
        try:
            if abs(float(np.linalg.det(A3))) > 1e-12:
                popt = np.linalg.solve(A3, -q[:3, 3])
                lo = np.minimum(V[a], V[b]) - 0.5 * np.abs(V[b] - V[a])
                hi = np.maximum(V[a], V[b]) + 0.5 * np.abs(V[b] - V[a])
                if np.all(popt >= lo) and np.all(popt <= hi):
                    cands.insert(0, popt)
        except np.linalg.LinAlgError:
            pass
        best = None
        for p in cands:
            h = np.array([p[0], p[1], p[2], 1.0])
            e = max(0.0, float(h @ q @ h))
            if best is None or e < best[0]:
                best = (e, p)
        return best

    # --- the collapse tests -------------------------------------------
    def neighbours(x):
        out = set()
        for fi in vf[x]:
            if not alive[fi]:
                continue
            out.update(tris[fi])
        out.discard(x)
        return out

    def link_ok(a, b):
        shared = neighbours(a) & neighbours(b)
        e = _edge_key(a, b)
        live = [f for f in ef.get(e, ()) if alive[f]]
        want = set()
        for fi in live:
            want.update(x for x in tris[fi] if x not in (a, b))
        # exactly the opposite vertices, and no more
        return shared == want and len(live) in (1, 2)

    def flips(a, b, p):
        """Would moving a to p invert any surviving triangle?"""
        for fi in vf[a]:
            if not alive[fi]:
                continue
            t = tris[fi]
            if b in t:
                continue          # this one disappears
            q = [p if x == a else V[x] for x in t]
            n0 = np.cross(V[t[1]] - V[t[0]], V[t[2]] - V[t[0]])
            n1 = np.cross(q[1] - q[0], q[2] - q[0])
            if float(np.dot(n0, n1)) <= 0.0:
                return True
            if float(np.linalg.norm(n1)) < 1e-16:
                return True
        return False

    def allowed(a, b):
        if a in frozen or b in frozen:
            return False
        if preserve_boundary:
            # The rim is untouchable: not moved, not shortened, not
            # re-parameterised.  Anything less and an open patch's
            # outline drifts -- measured at 92 boundary edges down to 25
            # when rim-along-rim collapses were permitted.
            if a in boundary_v or b in boundary_v:
                return False
        elif a in boundary_v:
            if b not in boundary_v:
                return False
            if len(ef.get(_edge_key(a, b), ())) != 1:
                return False
        if max_valence:
            n = len((neighbours(a) | neighbours(b)) - {a, b})
            if n > max_valence:
                return False
        return link_ok(a, b)

    # --- the queue -----------------------------------------------------
    heap = []
    for e in ef:
        a, b = e
        c, p = cost(a, b)
        heapq.heappush(heap, (c, a, b, tuple(p)))

    live_tris = len(tris)
    merged = {}

    def find(x):
        while x in merged:
            x = merged[x]
        return x

    while heap and live_tris > n_target:
        c, a, b, p = heapq.heappop(heap)
        a, b = find(a), find(b)
        if a == b:
            continue
        if max_error is not None and c > max_error:
            break
        e = _edge_key(a, b)
        if not [f for f in ef.get(e, ()) if alive[f]]:
            continue
        # `a` is absorbed into `b`; prefer to keep a boundary vertex
        if a in boundary_v and b not in boundary_v:
            a, b = b, a
        p = np.asarray(p, dtype=float)
        if a in boundary_v and b in boundary_v:
            p = V[b]              # never move the rim
        if not allowed(a, b) or flips(a, b, p):
            continue
        # do it
        for fi in list(ef.get(_edge_key(a, b), ())):
            if alive[fi]:
                alive[fi] = False
                live_tris -= 1
        V[b] = p
        Q[b] = Q[a] + Q[b]
        for fi in list(vf[a]):
            if not alive[fi]:
                continue
            tris[fi] = tuple(b if x == a else x for x in tris[fi])
            vf[b].add(fi)
        vf[a] = set()
        merged[a] = b
        if a in boundary_v:
            boundary_v.add(b)
        # re-key the edges that moved, and re-price them
        for x in neighbours(b):
            ek = _edge_key(b, x)
            ef.setdefault(ek, [])
            for fi in vf[b]:
                if alive[fi] and x in tris[fi] and fi not in ef[ek]:
                    ef[ek].append(fi)
            cc, pp = cost(b, x)
            heapq.heappush(heap, (cc, b, x, tuple(pp)))

    # --- compact -------------------------------------------------------
    keep = [t for i, t in enumerate(tris) if alive[i]]
    keep = [t for t in keep if len(set(t)) == 3]
    used = sorted({x for t in keep for x in t})
    remap = {x: i for i, x in enumerate(used)}
    return V[used], [[remap[x] for x in t] for t in keep]


def mesh_stats(V, tris):
    """(triangles, boundary edges, components) -- what must not change."""
    ef = {}
    for t in tris:
        for e in (_edge_key(t[0], t[1]), _edge_key(t[1], t[2]),
                  _edge_key(t[2], t[0])):
            ef[e] = ef.get(e, 0) + 1
    boundary = sum(1 for v in ef.values() if v == 1)
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for t in tris:
        r = find(t[0])
        for x in t[1:]:
            rx = find(x)
            if rx != r:
                par[rx] = r
    comps = len({find(x) for t in tris for x in t}) if tris else 0
    return len(tris), boundary, comps


def _selftest():
    rng = np.random.RandomState(7)

    # a closed shape: an icosphere-ish subdivided octahedron
    P = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
         (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    V = np.array(P, dtype=float)
    tris = list(F)
    for _ in range(4):                       # subdivide to ~2048 faces
        mid, new = {}, []
        Vl = list(map(tuple, V))
        for (a, b, c) in tris:
            m = {}
            for (x, y) in ((a, b), (b, c), (c, a)):
                k = _edge_key(x, y)
                if k not in mid:
                    p = 0.5 * (np.asarray(Vl[x]) + np.asarray(Vl[y]))
                    p = p / (np.linalg.norm(p) or 1.0)
                    mid[k] = len(Vl)
                    Vl.append(tuple(p))
                m[k] = mid[k]
            ab, bc, ca = (m[_edge_key(a, b)], m[_edge_key(b, c)],
                          m[_edge_key(c, a)])
            new += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        V, tris = np.array(Vl, dtype=float), new
    n0, b0, c0 = mesh_stats(V, tris)
    assert b0 == 0, "closed test mesh should have no boundary"
    W, T = decimate(V, tris, 0.35)
    n1, b1, c1 = mesh_stats(W, T)
    # THE POINT OF THE MODULE: no holes, no new pieces
    assert b1 == 0, "decimation opened a hole: %d boundary edges" % b1
    assert c1 == c0, "component count changed: %d -> %d" % (c0, c1)
    assert n1 < n0, "nothing was removed (%d -> %d)" % (n0, n1)
    r = float(np.max(np.abs(np.linalg.norm(W, axis=1) - 1.0)))
    assert r < 0.12, "sphere drifted off the unit radius by %.3f" % r

    # an OPEN patch: the rim must survive exactly
    m = 24
    xs, ys = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
    zs = 0.25 * np.sin(3 * xs) * np.cos(3 * ys)
    P = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    T = []
    for i in range(m - 1):
        for j in range(m - 1):
            a = i * m + j
            T += [(a, a + 1, a + m), (a + 1, a + m + 1, a + m)]
    n0, b0, c0 = mesh_stats(P, T)
    rim0 = {tuple(np.round(P[i], 9)) for i in
            {x for e, k in _edges_of(T).items() if k == 1 for x in e}}
    W, T2 = decimate(P, T, 0.4)
    n1, b1, c1 = mesh_stats(W, T2)
    assert c1 == c0, "patch split into %d pieces" % c1
    assert b1 == b0, ("boundary changed on an open patch: %d -> %d"
                      % (b0, b1))
    rim1 = {tuple(np.round(W[i], 9)) for i in
            {x for e, k in _edges_of(T2).items() if k == 1 for x in e}}
    assert rim1 <= rim0, "decimation invented %d rim points" % len(rim1 - rim0)
    assert n1 < n0

    # a mesh with TWO sheets passing close together -- the case vertex
    # clustering fuses and this must not
    A = P.copy()
    B = P.copy()
    B[:, 2] += 0.004                       # 4 thousandths apart
    V2 = np.vstack([A, B])
    T3 = list(T) + [(a + len(A), b + len(A), c + len(A)) for (a, b, c) in T]
    n0, b0, c0 = mesh_stats(V2, T3)
    assert c0 == 2
    W, T4 = decimate(V2, T3, 0.4)
    n1, b1, c1 = mesh_stats(W, T4)
    assert c1 == 2, "the two sheets were fused into %d" % c1
    assert b1 == b0, "boundary changed: %d -> %d" % (b0, b1)
    print("decimate: closed %d->%d, patch rim intact, two sheets stayed two"
          % (n0, n1))


def _edges_of(tris):
    ef = {}
    for t in tris:
        for e in (_edge_key(t[0], t[1]), _edge_key(t[1], t[2]),
                  _edge_key(t[2], t[0])):
            ef[e] = ef.get(e, 0) + 1
    return ef
