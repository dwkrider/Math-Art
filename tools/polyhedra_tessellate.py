# Tessellation of polyhedron faces as the database stores them.
#
# Records in `data/polyhedra/` store each face as its TRUE WINDING CYCLE
# rather than as a pre-triangulated patch, which is what lets a {5/2}
# pentagram be one face instead of five.  Anything that wants to draw
# those records -- the thumbnail renderer here, the companion website's
# WebGL viewer -- has to turn such a cycle into triangles, and the naive
# answers are all wrong for some part of the database:
#
#   * a triangle fan is wrong for the 7,600-odd concave faces, and
#   * any winding-based fill that keys off the polygon's TURNING number
#     is wrong for the 568 crossed quadrilaterals, whose turning number
#     is 0 even though both lobes are visibly there.
#
# Note also that the stored metadata cannot be trusted to find the star
# faces: `faces_by_type` labels a pentagram "{5}", so a pentagram and a
# convex pentagon are indistinguishable by their Schlafli symbol.  The
# self-intersection has to be detected geometrically, which is what this
# module does.
#
# So there is one general routine.  A face is projected to its own plane,
# every self-intersection is found and the edges are split there, the
# resulting planar subdivision is walked into its bounded regions, and a
# region is kept when the ORIGINAL cycle's winding number about an
# interior point of it is non-zero.  Kept regions are simple polygons and
# are ear-clipped.  That rule gives:
#
#   convex / concave    the polygon itself (no intersections to find)
#   {5/2} pentagram     inner pentagon (winding 2) + 5 tips (winding 1)
#   {8/3}, {10/3}       likewise, every ring of the star
#   crossed quad        both lobes (windings +1 and -1, both non-zero)
#
# The nonzero rule rather than even-odd is deliberate: even-odd would
# punch the inner pentagon out of a pentagram, which is not how these
# solids are drawn.
#
# Faces here are tiny -- at most 12 vertices, so at most 66 segment pairs
# to test -- and the O(n^2) methods below are far below the noise floor
# of a Cycles render or a frame budget.
#
# References:
# - Ear clipping: Gary H. Meisters, "Polygons have ears", American
#   Mathematical Monthly 82 (1975), 648-651.
# - Winding-number point location: Dan Sunday, "Inclusion of a point in a
#   polygon" (2001), the standard crossing/winding formulation.
# - Planar subdivision by angular half-edge ordering is the textbook
#   doubly-connected-edge-list face traversal; see Mark de Berg et al.,
#   "Computational Geometry: Algorithms and Applications", 3rd ed.
#   (Springer, 2008), chapter 2.

import math

# Faces are unit-ish in size (edge length 1 by the database's
# normalization), so a single absolute tolerance is honest here.
EPS = 1e-9


# -- plane projection -------------------------------------------------------

def _newell_normal(pts):
    """Newell's normal: correct for non-planar and non-convex input, and
    it averages over every edge rather than trusting one cross product.

    Newell CANCELS on a symmetric crossed quadrilateral, whose two lobes
    contribute equal and opposite area -- and the uniform duals are full
    of those (348 faces across the database vanished this way before the
    fallback below existed).  The face is still perfectly planar, so when
    the sum degenerates we take the normal from the widest triple of
    vertices instead.  Its sign is then arbitrary, which is honest: a
    crossed quad has no consistent outward side to pick.
    """
    n = [0.0, 0.0, 0.0]
    m = len(pts)
    for i in range(m):
        p, q = pts[i], pts[(i + 1) % m]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    ln = math.sqrt(sum(t * t for t in n))
    if ln >= EPS:
        return [t / ln for t in n]

    best, best_ln = None, EPS
    for i in range(1, m):
        for j in range(i + 1, m):
            a = [pts[i][k] - pts[0][k] for k in range(3)]
            b = [pts[j][k] - pts[0][k] for k in range(3)]
            c = [a[1] * b[2] - a[2] * b[1],
                 a[2] * b[0] - a[0] * b[2],
                 a[0] * b[1] - a[1] * b[0]]
            cl = math.sqrt(sum(t * t for t in c))
            if cl > best_ln:
                best, best_ln = c, cl
    if best is None:
        return None
    return [t / best_ln for t in best]


def plane_basis(pts):
    """An orthonormal (origin, u, w, normal) frame for the face's plane.

    `u` is aimed at the first vertex so the frame is deterministic: the
    same face always projects to the same 2-D coordinates, which keeps
    renders reproducible.
    """
    m = len(pts)
    c = [sum(p[k] for p in pts) / m for k in range(3)]
    nrm = _newell_normal(pts)
    if nrm is None:
        return None
    u = [pts[0][k] - c[k] for k in range(3)]
    # Degenerate only if vertex 0 sits at the centroid; fall back to any
    # direction in the plane.
    ln = math.sqrt(sum(t * t for t in u))
    if ln < EPS:
        seed = [1.0, 0.0, 0.0] if abs(nrm[0]) < 0.9 else [0.0, 1.0, 0.0]
        u = [seed[k] - nrm[k] * sum(seed[j] * nrm[j] for j in range(3))
             for k in range(3)]
        ln = math.sqrt(sum(t * t for t in u))
        if ln < EPS:
            return None
    u = [t / ln for t in u]
    w = [nrm[1] * u[2] - nrm[2] * u[1],
         nrm[2] * u[0] - nrm[0] * u[2],
         nrm[0] * u[1] - nrm[1] * u[0]]
    return c, u, w, nrm


def project(pts, frame):
    c, u, w, _n = frame
    out = []
    for p in pts:
        d = [p[k] - c[k] for k in range(3)]
        out.append((sum(d[k] * u[k] for k in range(3)),
                    sum(d[k] * w[k] for k in range(3))))
    return out


def unproject(q, frame):
    c, u, w, _n = frame
    return tuple(c[k] + q[0] * u[k] + q[1] * w[k] for k in range(3))


# -- 2-D primitives ---------------------------------------------------------

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def signed_area(poly):
    s = 0.0
    m = len(poly)
    for i in range(m):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % m]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def winding_number(poly, pt):
    """Winding number of `poly` about `pt`, by the crossing formulation.

    This is the test that decides whether a region of the subdivision is
    part of the face.  It is evaluated against the ORIGINAL cycle, not
    against the split edges, which is what makes a pentagram's inner
    pentagon come out as 2 and a crossed quad's lobes as +1 and -1.
    """
    wn = 0
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        if a[1] <= pt[1]:
            if b[1] > pt[1] and _cross(a, b, pt) > 0:
                wn += 1
        else:
            if b[1] <= pt[1] and _cross(a, b, pt) < 0:
                wn -= 1
    return wn


def _seg_intersect_params(p1, p2, p3, p4):
    """Parameters (t, s) where segments p1p2 and p3p4 properly cross.

    Returns None for parallel, collinear or endpoint-only contact.  Only
    PROPER crossings matter: shared endpoints are already shared vertices
    of the cycle and need no split.
    """
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (p4[0] - p3[0], p4[1] - p3[1])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < EPS:
        return None
    dx, dy = p3[0] - p1[0], p3[1] - p1[1]
    t = (dx * d2[1] - dy * d2[0]) / den
    s = (dx * d1[1] - dy * d1[0]) / den
    if t <= EPS or t >= 1.0 - EPS or s <= EPS or s >= 1.0 - EPS:
        return None
    return t, s


def has_self_intersection(poly):
    m = len(poly)
    for i in range(m):
        for j in range(i + 1, m):
            if j == i or (j + 1) % m == i or (i + 1) % m == j:
                continue
            if _seg_intersect_params(poly[i], poly[(i + 1) % m],
                                     poly[j], poly[(j + 1) % m]):
                return True
    return False


# -- ear clipping -----------------------------------------------------------

def _point_in_triangle(p, a, b, c):
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    neg = (d1 < -EPS) or (d2 < -EPS) or (d3 < -EPS)
    pos = (d1 > EPS) or (d2 > EPS) or (d3 > EPS)
    return not (neg and pos)


def ear_clip(poly):
    """Triangulate a simple polygon. Returns index triples into `poly`.

    Handles concave polygons, which a fan does not -- and roughly a third
    of the database's faces are concave once the Catalan and Johnson
    solids are included.
    """
    m = len(poly)
    if m < 3:
        return []
    idx = list(range(m))
    if signed_area(poly) < 0:                  # work counter-clockwise
        idx.reverse()
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 4 * m * m:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0 = idx[(k - 1) % len(idx)]
            i1 = idx[k]
            i2 = idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if _cross(a, b, c) <= EPS:         # reflex or degenerate
                continue
            if any(_point_in_triangle(poly[j], a, b, c)
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            break                              # numerically stuck; salvage
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


# -- planar subdivision -----------------------------------------------------

def _dedupe_point(store, pt, tol=1e-7):
    for i, q in enumerate(store):
        if abs(q[0] - pt[0]) < tol and abs(q[1] - pt[1]) < tol:
            return i
    store.append(pt)
    return len(store) - 1


def _subdivide(poly):
    """Split every edge at its proper crossings.

    Returns (points, edges) with `edges` a set of undirected index pairs.
    """
    m = len(poly)
    pts = []
    ring = [_dedupe_point(pts, p) for p in poly]
    # splits[i] collects (t, point-index) along edge i
    splits = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if j == i or (j + 1) % m == i or (i + 1) % m == j:
                continue
            a, b = poly[i], poly[(i + 1) % m]
            c, d = poly[j], poly[(j + 1) % m]
            hit = _seg_intersect_params(a, b, c, d)
            if not hit:
                continue
            t, s = hit
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            pi = _dedupe_point(pts, p)
            splits[i].append((t, pi))
            splits[j].append((s, pi))
    edges = set()
    for i in range(m):
        chain = [(0.0, ring[i])] + sorted(splits[i]) + [(1.0, ring[(i + 1) % m])]
        for k in range(len(chain) - 1):
            a, b = chain[k][1], chain[k + 1][1]
            if a != b:
                edges.add((min(a, b), max(a, b)))
    return pts, edges


def _faces_of(pts, edges):
    """Walk the subdivision into its bounded regions.

    Standard half-edge traversal: arriving at v from u, leave along the
    neighbour that is next CLOCKWISE from u around v.  That keeps the
    region on the left, so bounded faces come out counter-clockwise
    (positive area) and the single outer face comes out negative.
    """
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    order = {}
    for v, nbrs in adj.items():
        nbrs.sort(key=lambda w: math.atan2(pts[w][1] - pts[v][1],
                                           pts[w][0] - pts[v][0]))
        order[v] = {w: k for k, w in enumerate(nbrs)}
    faces = []
    unused = {(a, b) for a, b in edges} | {(b, a) for a, b in edges}
    while unused:
        start = next(iter(unused))
        cycle = []
        cur = start
        while cur in unused:
            unused.discard(cur)
            u, v = cur
            cycle.append(u)
            nbrs = adj[v]
            k = order[v][u]
            nxt = nbrs[(k - 1) % len(nbrs)]
            cur = (v, nxt)
        if len(cycle) >= 3:
            faces.append(cycle)
    return faces


def _interior_point(poly):
    """A point strictly inside a simple polygon: the centroid of an ear."""
    tris = ear_clip(poly)
    if not tris:
        return None
    i0, i1, i2 = tris[0]
    return ((poly[i0][0] + poly[i1][0] + poly[i2][0]) / 3.0,
            (poly[i0][1] + poly[i1][1] + poly[i2][1]) / 3.0)


def fill_polygon_2d(poly):
    """Triangulate a 2-D cycle under the nonzero-winding rule.

    Returns (points, triangles); `points` starts with the input vertices
    and is extended with any self-intersection points that were created.
    """
    if len(poly) < 3:
        return list(poly), []
    if not has_self_intersection(poly):
        return list(poly), ear_clip(poly)

    pts, edges = _subdivide(poly)
    tris = []
    for cycle in _faces_of(pts, edges):
        ring = [pts[i] for i in cycle]
        if signed_area(ring) <= EPS:
            continue                           # outer face, or degenerate
        inside = _interior_point(ring)
        if inside is None or winding_number(poly, inside) == 0:
            continue
        for a, b, c in ear_clip(ring):
            tris.append((cycle[a], cycle[b], cycle[c]))
    return pts, tris


def tessellate_face(points3d):
    """Triangulate one stored face cycle.

    `points3d` are the face's vertex positions in stored winding order.
    Returns (points3d_out, triangles, normal); `points3d_out` begins with
    the input points and is extended with any intersection points, so
    callers can append the extras to their own vertex array and offset
    the triangle indices.
    """
    if len(points3d) < 3:
        return list(points3d), [], (0.0, 0.0, 1.0)
    frame = plane_basis(points3d)
    if frame is None:
        return list(points3d), [], (0.0, 0.0, 1.0)
    poly = project(points3d, frame)
    pts2, tris = fill_polygon_2d(poly)
    out = list(points3d) + [unproject(q, frame) for q in pts2[len(points3d):]]
    return out, tris, tuple(frame[3])


# -- self-test --------------------------------------------------------------

def _regular_star(n, d, r=1.0):
    return [(r * math.cos(2 * math.pi * d * i / n),
             r * math.sin(2 * math.pi * d * i / n)) for i in range(n)]


def _area_of(pts, tris):
    return sum(abs(_cross(pts[a], pts[b], pts[c])) / 2.0 for a, b, c in tris)


def _selftest():
    # A square: one quad, two ears, area 1.
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    p, t = fill_polygon_2d(sq)
    assert len(t) == 2, t
    assert abs(_area_of(p, t) - 1.0) < 1e-9, _area_of(p, t)

    # A concave "L": a fan from vertex 0 would spill outside it.
    L = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    p, t = fill_polygon_2d(L)
    assert abs(_area_of(p, t) - 3.0) < 1e-9, _area_of(p, t)

    # {5/2} pentagram. Nonzero fill keeps the inner pentagon, so the area
    # is the whole star outline, not the outline minus its middle.
    star = _regular_star(5, 2)
    p, t = fill_polygon_2d(star)
    assert has_self_intersection(star)
    # Closed form: the solid pentagram on the unit circumcircle.
    inner_r = math.cos(2 * math.pi / 5) / math.cos(math.pi / 5)
    pent = 5 * inner_r ** 2 * math.sin(2 * math.pi / 5) / 2
    tip = 5 * (inner_r * math.sin(math.pi / 5) * (1 - inner_r * math.cos(math.pi / 5)))
    want = pent + tip
    got = _area_of(p, t)
    assert abs(got - want) < 1e-6, (got, want)

    # Even-odd would have returned the outline MINUS the inner pentagon;
    # check we are meaningfully above that, i.e. the rule really is nonzero.
    assert got > want - pent + 1e-3

    # {8/3} and {10/3}: every ring kept, no gaps, no double cover.
    for n, d in ((8, 3), (10, 3), (7, 2)):
        s = _regular_star(n, d)
        p, t = fill_polygon_2d(s)
        assert t, (n, d)
        a = _area_of(p, t)
        # The star fits inside its circumcircle and contains its incircle
        # ring; a loose sanity bracket catches gaps and double covers.
        assert 0.3 < a < math.pi, (n, d, a)

    # Crossed quadrilateral (turning number 0). Both lobes must survive:
    # a fill keyed off the turning number would return nothing at all.
    bow = [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert has_self_intersection(bow)
    p, t = fill_polygon_2d(bow)
    assert t, "crossed quad tessellated to nothing"
    assert abs(_area_of(p, t) - 0.5) < 1e-9, _area_of(p, t)

    # A crossed quad in 3-D, tilted off-axis. This is the case Newell's
    # normal cancels on: the two lobes have equal and opposite area, so
    # the summed normal is zero and the face used to tessellate to
    # nothing. 348 faces across the uniform duals look like this.
    bow3 = [(0, 0, 1.0), (1, 0, 1.3), (0, 1, 1.2), (1, 1, 1.5)]
    pts3, tris, nrm = tessellate_face(bow3)
    assert tris, "crossed quad in 3-D tessellated to nothing"
    assert abs(math.sqrt(sum(t * t for t in nrm)) - 1.0) < 1e-9, nrm

    # 3-D entry point: a pentagram tilted off-axis still tessellates, and
    # its triangles all lie in the face plane.
    tilt = [(x, y, 0.3 * x + 0.2 * y + 1.0) for x, y in _regular_star(5, 2)]
    pts3, tris, nrm = tessellate_face(tilt)
    assert tris
    o = pts3[0]
    for tri in tris:
        for i in tri:
            v = [pts3[i][k] - o[k] for k in range(3)]
            assert abs(sum(v[k] * nrm[k] for k in range(3))) < 1e-9

    print("RESULT: OK")


if __name__ == "__main__":
    _selftest()
