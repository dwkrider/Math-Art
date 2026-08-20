# The planar crease graph: faces, adjacency, and triangulation.
#
# Part of the Math Art crease engine (`math_art/crease/`).  Python and
# numpy only -- no `bpy`.
#
# WHY THIS EXISTS.  FOLD does not require `faces_vertices`; plenty of
# real files carry only vertices and edges, because the editors that
# wrote them work in terms of lines.  Everything downstream -- folding,
# rendering, layer ordering -- needs faces.  Recovering them is a planar
# graph problem, not a parsing problem, so it lives here and is called
# explicitly rather than happening silently inside the reader.
#
# HOW FACES ARE RECOVERED.  The standard planar-subdivision walk.  Give
# every undirected edge two half-edges; at each vertex sort the outgoing
# half-edges by angle; then the face to the left of a half-edge is found
# by repeatedly taking the NEXT half-edge CLOCKWISE from the twin at the
# far end.  Each cycle traced this way is a face.  Exactly one cycle
# comes out with negative signed area -- that is the outer face, and it
# is discarded.
#
#     next(h) = twin(h) rotated one step clockwise about its origin
#
# This is O(E log E) for the angular sorts and O(E) for the walk, and it
# needs no tolerance beyond the angular comparison itself, because it
# never asks whether a point is inside a polygon.
#
# WHAT IT ASSUMES, AND WHAT IT REFUSES.  The input must be a PLANE graph:
# edges may meet only at shared endpoints.  Crossing edges are a
# different (and much messier) problem -- they need an arrangement with
# intersection points inserted -- and a crease pattern that has them is
# almost always a broken export rather than a design intent.  So they are
# detected and reported, not silently repaired.
#
# TRIANGULATION.  Both fold solvers need triangles: the rigid solver
# because a k-gon panel with k > 3 is not determined by its edge lengths,
# and the compliant solver because its bending term is per triangle pair.
# Diagonals added here are tagged `F` (flat) so they are visible to the
# solver but carry no fold, and so a round trip back out to FOLD does not
# claim the user drew them.
#
# References:
#   E. D. Demaine, J. O'Rourke, "Geometric Folding Algorithms"
#       (Cambridge, 2007), ch. 12 -- crease patterns as plane graphs.
#   T. Tachi, "Simulation of Rigid Origami," Origami^4, 2009 -- why
#       every k-gon is triangulated before the constraint assembly, and
#       the resulting DOF bookkeeping.

import numpy as np

from .fold_io import FLAT, FoldError


class GraphError(FoldError):
    """A crease graph that is not a plane graph, or not traversable."""


def _half_edges(n_verts, edges):
    """Build the sorted half-edge structure.

    Returns (origin, twin, order), where half-edge 2i runs from
    edges[i][0] to edges[i][1], half-edge 2i+1 is its twin, and `order`
    maps a half-edge to the next one clockwise about its origin.
    """
    m = len(edges)
    origin = np.empty(2 * m, dtype=np.int64)
    target = np.empty(2 * m, dtype=np.int64)
    origin[0::2] = edges[:, 0]
    target[0::2] = edges[:, 1]
    origin[1::2] = edges[:, 1]
    target[1::2] = edges[:, 0]
    twin = np.arange(2 * m, dtype=np.int64) ^ 1
    return origin, target, twin


def build_faces(verts, edges, tol=1e-9):
    """Recover the faces of a plane graph.

    `verts` is (n, 2) or (n, 3); only the first two coordinates are used,
    so a flat 3-D frame works directly.  Returns a list of faces, each a
    list of vertex indices in counter-clockwise order.  The outer face is
    not included.

    Raises GraphError if the graph is not planar as drawn, or if a walk
    fails to close -- both of which mean the file is not what it claims.
    """
    verts = np.asarray(verts, dtype=float)
    edges = np.asarray(edges, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] < 2:
        raise GraphError("build_faces needs 2-D or 3-D vertex coordinates")
    if not len(edges):
        return []
    xy = verts[:, :2]

    _check_no_crossings(xy, edges, tol)

    origin, target, twin = _half_edges(len(verts), edges)
    n_half = len(origin)

    # Angular order of the half-edges leaving each vertex.
    d = xy[target] - xy[origin]
    ang = np.arctan2(d[:, 1], d[:, 0])
    by_vertex = {}
    for h in range(n_half):
        by_vertex.setdefault(int(origin[h]), []).append(h)
    # next_cw[h]: the half-edge one step CLOCKWISE from h about origin[h]
    next_cw = np.empty(n_half, dtype=np.int64)
    for v, hs in by_vertex.items():
        hs_sorted = sorted(hs, key=lambda h: ang[h])
        k = len(hs_sorted)
        for i, h in enumerate(hs_sorted):
            next_cw[h] = hs_sorted[(i - 1) % k]

    # Walk: the face to the left of h continues at next_cw[twin[h]].
    visited = np.zeros(n_half, dtype=bool)
    faces = []
    for start in range(n_half):
        if visited[start]:
            continue
        cycle = []
        h = start
        for _ in range(n_half + 1):
            if visited[h]:
                break
            visited[h] = True
            cycle.append(int(origin[h]))
            h = int(next_cw[int(twin[h])])
            if h == start:
                break
        else:
            raise GraphError("face walk did not close; graph is malformed")
        if len(cycle) >= 3:
            faces.append(cycle)

    if not faces:
        return []

    # Exactly one traced cycle is the outer boundary: it is the one whose
    # signed area is negative (the others are CCW and positive).  A tree
    # of edges with no enclosed area yields none, which is legitimate.
    areas = np.array([_signed_area(xy, f) for f in faces])
    keep = [f for f, a in zip(faces, areas) if a > tol]
    return keep


def _signed_area(xy, face):
    p = xy[np.asarray(face, dtype=np.int64)]
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _check_no_crossings(xy, edges, tol):
    """Reject edges that cross away from a shared endpoint.

    Brute force over a bounding-box filter.  Crease patterns are small
    (thousands of edges at most) and this runs once at import, so the
    quadratic worst case is not worth an interval tree.
    """
    m = len(edges)
    p = xy[edges[:, 0]]
    q = xy[edges[:, 1]]
    lo = np.minimum(p, q) - tol
    hi = np.maximum(p, q) + tol
    for i in range(m):
        # bounding-box overlap with every later edge, vectorised
        cand = np.nonzero(
            (lo[i + 1:, 0] <= hi[i, 0]) & (hi[i + 1:, 0] >= lo[i, 0]) &
            (lo[i + 1:, 1] <= hi[i, 1]) & (hi[i + 1:, 1] >= lo[i, 1])
        )[0]
        for off in cand:
            j = i + 1 + int(off)
            if set(edges[i].tolist()) & set(edges[j].tolist()):
                continue                     # share an endpoint: fine
            if _segments_cross(p[i], q[i], p[j], q[j], tol):
                raise GraphError(
                    f"edges {i} and {j} cross away from a shared vertex; "
                    "this is not a plane graph. Split the edges at their "
                    "intersection before importing.")


def _orient(a, b, c):
    return ((b[0] - a[0]) * (c[1] - a[1]) -
            (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross(a, b, c, d, tol):
    d1, d2 = _orient(c, d, a), _orient(c, d, b)
    d3, d4 = _orient(a, b, c), _orient(a, b, d)
    if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
       ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
        return True
    # Collinear overlap counts as crossing: two creases lying on top of
    # one another is as broken as two that cross.
    for x, y, z in ((c, d, a), (c, d, b), (a, b, c), (a, b, d)):
        if abs(_orient(x, y, z)) <= tol and _on_segment(x, y, z, tol):
            return True
    return False


def _on_segment(a, b, p, tol):
    return (min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and
            min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol)


def vertex_rings(n_verts, edges, verts):
    """For each vertex, its neighbours in counter-clockwise order.

    This is what the validity checks consume: Kawasaki and the angle sum
    are both statements about consecutive sector angles about a vertex,
    which only makes sense once the neighbours are cyclically ordered.
    """
    xy = np.asarray(verts, dtype=float)[:, :2]
    rings = [[] for _ in range(n_verts)]
    for a, b in np.asarray(edges, dtype=np.int64):
        rings[a].append(int(b))
        rings[b].append(int(a))
    out = []
    for v, nb in enumerate(rings):
        if not nb:
            out.append([])
            continue
        d = xy[np.asarray(nb, dtype=np.int64)] - xy[v]
        order = np.argsort(np.arctan2(d[:, 1], d[:, 0]))
        out.append([nb[i] for i in order])
    return out


def triangulate(verts, faces, assignment_of=None):
    """Fan-triangulate every face, reporting the diagonals added.

    Returns (triangles, diagonals) where `diagonals` is a list of vertex
    pairs that were not edges of the input.  Callers append those to the
    edge list tagged `F`, so the solver sees them and a FOLD export does
    not claim the user creased them.

    Fan triangulation is correct for convex faces and adequate for the
    mildly non-convex ones crease patterns produce; it is not a general
    polygon triangulator, and a strongly non-convex face would need ear
    clipping.  That case is reported rather than silently mis-triangulated.
    """
    xy = np.asarray(verts, dtype=float)[:, :2]
    tris, diags = [], []
    for f in faces:
        k = len(f)
        if k < 3:
            continue
        if k == 3:
            tris.append(tuple(int(i) for i in f))
            continue
        if not _is_convex(xy, f):
            raise GraphError(
                f"face {tuple(f)} is strongly non-convex; fan "
                "triangulation would produce inverted triangles")
        hub = int(f[0])
        for i in range(1, k - 1):
            tris.append((hub, int(f[i]), int(f[i + 1])))
            if i > 1:
                diags.append((hub, int(f[i])))
    return tris, diags


def _is_convex(xy, face, tol=1e-12):
    p = xy[np.asarray(face, dtype=np.int64)]
    n = len(p)
    sign = 0
    for i in range(n):
        o = _orient(p[i], p[(i + 1) % n], p[(i + 2) % n])
        if abs(o) <= tol:
            continue
        s = 1 if o > 0 else -1
        if sign and s != sign:
            return False
        sign = s
    return True


def _selftest():
    # --- a 2x2 grid of unit squares ---------------------------------
    xs, ys = np.meshgrid(np.arange(3.0), np.arange(3.0))
    V = np.stack([xs.ravel(), ys.ravel()], axis=1)
    E = []
    for r in range(3):
        for c in range(3):
            v = r * 3 + c
            if c < 2:
                E.append((v, v + 1))
            if r < 2:
                E.append((v, v + 3))
    E = np.array(E, dtype=np.int64)

    faces = build_faces(V, E)
    assert len(faces) == 4, f"expected 4 squares, got {len(faces)}"
    for f in faces:
        assert len(f) == 4
        assert _signed_area(V[:, :2], f) > 0      # all CCW
    assert abs(sum(_signed_area(V[:, :2], f) for f in faces) - 4.0) < 1e-9

    # --- faces of a single triangle ---------------------------------
    Vt = np.array([[0., 0.], [1., 0.], [0., 1.]])
    Et = np.array([[0, 1], [1, 2], [2, 0]])
    ft = build_faces(Vt, Et)
    assert len(ft) == 1 and len(ft[0]) == 3
    assert abs(_signed_area(Vt, ft[0]) - 0.5) < 1e-12

    # --- a tree encloses nothing ------------------------------------
    Vy = np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.]])
    Ey = np.array([[0, 1], [0, 2], [0, 3]])
    assert build_faces(Vy, Ey) == []

    # --- flat 3-D input is accepted ---------------------------------
    V3 = np.hstack([V, np.zeros((len(V), 1))])
    assert len(build_faces(V3, E)) == 4

    # --- crossing edges are refused, not repaired -------------------
    Vx = np.array([[0., 0.], [1., 1.], [1., 0.], [0., 1.]])
    Ex = np.array([[0, 1], [2, 3]])
    try:
        build_faces(Vx, Ex)
    except GraphError as exc:
        assert "cross" in str(exc)
    else:
        raise AssertionError("crossing edges should raise")

    # sharing an endpoint is not a crossing
    Vs = np.array([[0., 0.], [1., 0.], [0., 1.]])
    Es = np.array([[0, 1], [0, 2]])
    build_faces(Vs, Es)

    # --- counter-clockwise neighbour rings --------------------------
    Vr = np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    Er = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
    rings = vertex_rings(5, Er, Vr)
    # CCW, but the cycle starts wherever atan2 puts -pi -- here at the
    # south neighbour.  Compare cyclically, not by first element.
    assert len(rings[0]) == 4 and set(rings[0]) == {1, 2, 3, 4}
    doubled = rings[0] + rings[0]
    assert any(doubled[i:i + 4] == [1, 2, 3, 4] for i in range(4)), rings[0]

    # --- triangulation ----------------------------------------------
    tris, diags = triangulate(V, faces)
    assert len(tris) == 8                          # 4 quads -> 8 triangles
    assert len(diags) == 4                         # one diagonal per quad
    for a, b in diags:
        assert not ((E[:, 0] == a) & (E[:, 1] == b)).any()
        assert not ((E[:, 0] == b) & (E[:, 1] == a)).any()
    # every triangle keeps the parent's orientation
    for t in tris:
        assert _signed_area(V[:, :2], list(t)) > 0

    # a strongly non-convex face is reported
    Vc = np.array([[0., 0.], [2., 0.], [2., 2.], [1., 0.5], [0., 2.]])
    try:
        triangulate(Vc, [[0, 1, 2, 3, 4]])
    except GraphError as exc:
        assert "non-convex" in str(exc)
    else:
        raise AssertionError("non-convex fan should raise")

    print("RESULT: OK  crease.graph")


# NOTE: no __main__ guard -- tests/test_selftests.py discovers and runs
# _selftest() headlessly (see CLAUDE.md).
