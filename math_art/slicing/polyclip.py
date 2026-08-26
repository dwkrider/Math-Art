# 2-D polygon toolkit for the fabrication slicer: area, containment,
# boolean difference and outward offset.
#
# WHY A CLIPPER AND NOT "BOUNDARY SURGERY".  Cutting a slot into a slice
# is subtracting a rectangle that is open at the rim.  It is tempting to
# do that by splicing the rectangle into the outline where its two long
# sides cross -- which is what `slide_together_generator` does for
# Hart's slide-togethers, and it works there because those outlines are
# analytic polygons with a handful of vertices.  A slice through an
# arbitrary mesh is not that: it is a dense, noisy polyline with
# hundreds of vertices, where a slot side can cross the rim more than
# twice, a mouth can straddle a vertex, and a thin neck can let the
# rectangle leave and re-enter.  Every one of those cases silently
# produces a wrong outline under splicing.  So the slot is subtracted by
# a real polygon boolean instead.
#
# The boolean is Weiler-Atherton: walk the subject while outside the
# clip, and at every crossing switch to the clip boundary walked the
# other way, until the traversal closes.  Degeneracies -- an
# intersection landing on a vertex, or two edges collinear -- are NOT
# patched up with special cases.  They are detected and raised, and the
# caller retries with the clip polygon perturbed by a hair.  A slot
# moved by a micron is still the same slot; a clipper with twenty
# degenerate branches is not still correct.
#
# Everything here is plain Python on (x, y) tuples: no bpy, no numpy, so
# it self-tests headlessly.

import math


class DegenerateClip(Exception):
    """The clip touches the subject exactly on a vertex or along an
    edge, so the crossing sequence is ambiguous.  Retry perturbed."""


# ------------------------------------------------------------------ #
#  basic polygon predicates                                          #
# ------------------------------------------------------------------ #

def signed_area(poly):
    """Twice-signed area / 2: positive when the ring is CCW."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def area(poly):
    return abs(signed_area(poly))


def perimeter(poly):
    n = len(poly)
    if n < 2:
        return 0.0
    t = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        t += math.hypot(x1 - x0, y1 - y0)
    return t


def as_ccw(poly):
    return list(poly) if signed_area(poly) >= 0.0 else list(poly)[::-1]


def as_cw(poly):
    return list(poly) if signed_area(poly) <= 0.0 else list(poly)[::-1]


def bounds(poly):
    """(xmin, ymin, xmax, ymax)."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def centroid(poly):
    """Area centroid; falls back to the vertex mean if degenerate."""
    n = len(poly)
    a = signed_area(poly)
    if abs(a) < 1e-18:
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    cx = cy = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cr = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    return (cx / (6.0 * a), cy / (6.0 * a))


def point_in_polygon(pt, poly):
    """Crossing-number test.  Points exactly on the boundary are
    reported inconsistently by design -- callers that care must not ask
    about boundary points."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xc:
                inside = not inside
    return inside


def polygon_distance(pt, poly):
    """Shortest distance from `pt` to the polygon's boundary."""
    x, y = pt
    best = float('inf')
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 <= 0.0:
            d = math.hypot(x - ax, y - ay)
        else:
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
            d = math.hypot(x - (ax + t * dx), y - (ay + t * dy))
        best = min(best, d)
    return best


# ------------------------------------------------------------------ #
#  line / polygon interval query                                     #
# ------------------------------------------------------------------ #

def line_intervals(poly, origin, direction, eps=1e-12):
    """Intervals of the infinite line `origin + t*direction` that lie
    inside the simple polygon `poly`, as a sorted list of (t0, t1).

    This is the query that finds a slot's span: where the crossing line
    of two slice planes actually passes through the material.  Crossings
    are gathered, sorted, and paired off -- valid because a line meets a
    closed simple curve an even number of times.

    A VERTEX EXACTLY ON THE LINE is resolved by the same half-open rule
    the plane sectioning uses: on the line counts as the negative side.
    This is not a tolerance, it is a convention, and it gives the exact
    right answer in both cases.  Where the line genuinely passes
    THROUGH a vertex, the two incident edges classify differently and
    exactly one crossing is produced, at the vertex itself.  Where the
    line merely TOUCHES a vertex from outside, both edges cross there
    and the pair collapses to a zero-length interval, which is dropped
    -- a tangential touch is not material.

    The alternative, nudging the line off the vertex, is worse than it
    looks: the two families querying a shared crossing line each drop
    their own plane's normal component of the nudge, so any nonzero
    nudge puts them on slightly DIFFERENT lines, and on a sloped
    surface that shows up directly as the two disagreeing about where
    the material is.  A convention costs nothing and keeps them exact.
    """
    ox, oy = origin
    dx, dy = direction
    L = math.hypot(dx, dy)
    if L <= 0.0:
        raise ValueError("line_intervals: zero direction")
    dx, dy = dx / L, dy / L
    nx, ny = -dy, dx                      # left normal of the line

    ts = []
    n = len(poly)
    side = [(p[0] - ox) * nx + (p[1] - oy) * ny for p in poly]
    for i in range(n):
        j = (i + 1) % n
        sa, sb = side[i], side[j]
        if (sa > 0.0) == (sb > 0.0):      # on the line counts as below
            continue
        f = sa / (sa - sb)
        px = poly[i][0] + f * (poly[j][0] - poly[i][0])
        py = poly[i][1] + f * (poly[j][1] - poly[i][1])
        ts.append((px - ox) * dx + (py - oy) * dy)

    ts.sort()
    if len(ts) % 2:
        raise DegenerateClip("odd crossing count")
    out = [(ts[k], ts[k + 1]) for k in range(0, len(ts), 2)]
    return [(a, b) for a, b in out if b - a > eps]


# ------------------------------------------------------------------ #
#  Weiler-Atherton difference                                        #
# ------------------------------------------------------------------ #

def _seg_intersect(a0, a1, b0, b1, eps):
    """Proper crossing of two open segments -> (ta, tb, point), else
    None.  Touching at an endpoint or running collinear is degenerate
    and raises, because the traversal cannot be sequenced through it."""
    ax, ay = a0
    bx, by = a1
    cx, cy = b0
    dx, dy = b1
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    den = rx * sy - ry * sx
    qpx, qpy = cx - ax, cy - ay
    if abs(den) < eps:
        # parallel; collinear *and* overlapping is the bad case
        if abs(qpx * ry - qpy * rx) < eps:
            rr = rx * rx + ry * ry
            if rr > 0.0:
                t0 = (qpx * rx + qpy * ry) / rr
                t1 = t0 + (sx * rx + sy * ry) / rr
                lo, hi = min(t0, t1), max(t0, t1)
                if hi > eps and lo < 1.0 - eps:
                    raise DegenerateClip("collinear overlapping edges")
        return None
    ta = (qpx * sy - qpy * sx) / den
    tb = (qpx * ry - qpy * rx) / den
    on_a = -eps <= ta <= 1.0 + eps
    on_b = -eps <= tb <= 1.0 + eps
    if not (on_a and on_b):
        return None
    if (abs(ta) < eps or abs(ta - 1.0) < eps
            or abs(tb) < eps or abs(tb - 1.0) < eps):
        raise DegenerateClip("intersection lands on a vertex")
    return (ta, tb, (ax + ta * rx, ay + ta * ry))


class _Node:
    __slots__ = ('pt', 'is_x', 'entering', 'partner', 'used')

    def __init__(self, pt, is_x=False):
        self.pt = pt
        self.is_x = is_x
        self.entering = False
        self.partner = None
        self.used = False


def _build_rings(subject, clip, eps):
    """Both rings with intersection nodes spliced in, cross-linked."""
    sn = [_Node(p) for p in subject]
    cn = [_Node(p) for p in clip]
    s_ins = [[] for _ in subject]          # per subject edge: (t, node)
    c_ins = [[] for _ in clip]

    for i in range(len(subject)):
        a0, a1 = subject[i], subject[(i + 1) % len(subject)]
        for j in range(len(clip)):
            b0, b1 = clip[j], clip[(j + 1) % len(clip)]
            hit = _seg_intersect(a0, a1, b0, b1, eps)
            if hit is None:
                continue
            ta, tb, pt = hit
            na = _Node(pt, True)
            nb = _Node(pt, True)
            na.partner, nb.partner = nb, na
            s_ins[i].append((ta, na))
            c_ins[j].append((tb, nb))

    def weave(base, ins):
        out = []
        for i, node in enumerate(base):
            out.append(node)
            for _, x in sorted(ins[i], key=lambda kv: kv[0]):
                out.append(x)
        return out

    return weave(sn, s_ins), weave(cn, c_ins)


def difference(subject, clip, eps=1e-9):
    """`subject` minus `clip`, both simple polygons.  Returns a list of
    CCW rings (possibly empty if the clip swallows the subject, or more
    than one if the clip cuts the subject in two).

    Raises DegenerateClip on any touching/collinear configuration; the
    caller is expected to perturb the clip and retry.
    """
    subj = as_ccw(subject)
    clp = as_ccw(clip)

    sring, cring = _build_rings(subj, clp, eps)
    xs = [n for n in sring if n.is_x]
    if not xs:
        # no crossings: either disjoint (keep subject) or nested
        if point_in_polygon(subj[0], clp):
            return []                       # subject entirely removed
        if point_in_polygon(clp[0], subj):
            raise DegenerateClip("clip strictly interior: would need a hole")
        return [subj]

    # classify each subject crossing by whether the subject is heading
    # into the clip there, sampling just past the crossing
    ns = len(sring)
    for i, node in enumerate(sring):
        if not node.is_x:
            continue
        nxt = sring[(i + 1) % ns]
        mid = ((node.pt[0] + nxt.pt[0]) * 0.5,
               (node.pt[1] + nxt.pt[1]) * 0.5)
        node.entering = point_in_polygon(mid, clp)
        node.partner.entering = node.entering

    cindex = {id(n): i for i, n in enumerate(cring)}
    out = []
    guard = 0
    limit = 8 * (len(sring) + len(cring)) + 64

    for start in sring:
        # begin a ring at a point where the subject LEAVES the clip:
        # from there the subject boundary is outside, which is the part
        # we are keeping
        if not (start.is_x and not start.entering and not start.used):
            continue
        ring = []
        node = start
        on_subject = True
        while True:
            guard += 1
            if guard > limit:
                raise DegenerateClip("traversal did not close")
            node.used = True
            if node.partner is not None:
                node.partner.used = True
            ring.append(node.pt)
            if on_subject:
                i = None
                for k, n in enumerate(sring):
                    if n is node:
                        i = k
                        break
                node = sring[(i + 1) % len(sring)]
            else:
                i = cindex[id(node)]
                node = cring[(i - 1) % len(cring)]
            if node is start or (node.partner is not None
                                 and node.partner is start):
                break
            if node.is_x:
                # switch rails at every crossing
                ring.append(node.pt)
                node.used = True
                node.partner.used = True
                node = node.partner
                on_subject = not on_subject
        # drop consecutive duplicates
        clean = []
        for p in ring:
            if not clean or (abs(p[0] - clean[-1][0]) > eps
                             or abs(p[1] - clean[-1][1]) > eps):
                clean.append(p)
        if len(clean) >= 3 and area(clean) > eps:
            out.append(as_ccw(clean))
    return out


def difference_robust(subject, clip, eps=1e-9, jitter=None, tries=6):
    """`difference`, retrying with the clip nudged when a degeneracy is
    hit.  Returns (rings, nudged) or raises DegenerateClip if every
    attempt degenerated."""
    if jitter is None:
        x0, y0, x1, y1 = bounds(subject)
        jitter = 1e-7 * max(x1 - x0, y1 - y0, 1e-6)
    # a deterministic spiral of nudges: no RNG, so runs reproduce
    steps = [(0.0, 0.0), (1.0, 0.37), (-0.73, 0.91), (0.51, -1.13),
             (-1.29, -0.44), (1.77, 1.61)]
    last = None
    for k in range(min(tries, len(steps))):
        ox, oy = steps[k]
        moved = [(p[0] + ox * jitter, p[1] + oy * jitter) for p in clip]
        try:
            return difference(subject, moved, eps), k > 0
        except DegenerateClip as exc:
            last = exc
    raise DegenerateClip(f"still degenerate after {tries} nudges: {last}")


# ------------------------------------------------------------------ #
#  offsetting (kerf compensation)                                    #
# ------------------------------------------------------------------ #

def offset_polygon(poly, dist, miter_limit=4.0):
    """Miter-offset a simple ring by `dist` (positive grows a CCW ring).

    Kerf compensation only ever asks for offsets far smaller than the
    features, so a miter offset with a limit is enough and no
    self-intersection cleanup is done.  Asking for a large offset on a
    ring with tight concavities WILL produce self-intersections; that is
    a documented limit of this routine, not a bug to be surprised by.
    """
    if abs(dist) < 1e-15:
        return list(poly)
    ring = as_ccw(poly)
    n = len(ring)
    out = []
    for i in range(n):
        p = ring[i]
        a = ring[(i - 1) % n]
        b = ring[(i + 1) % n]
        e0 = (p[0] - a[0], p[1] - a[1])
        e1 = (b[0] - p[0], b[1] - p[1])
        l0 = math.hypot(*e0) or 1.0
        l1 = math.hypot(*e1) or 1.0
        n0 = (e0[1] / l0, -e0[0] / l0)     # outward normal of a CCW ring
        n1 = (e1[1] / l1, -e1[0] / l1)
        mx, my = n0[0] + n1[0], n0[1] + n1[1]
        ml = math.hypot(mx, my)
        if ml < 1e-12:
            out.append((p[0] + n1[0] * dist, p[1] + n1[1] * dist))
            continue
        mx, my = mx / ml, my / ml
        cosh = mx * n1[0] + my * n1[1]
        scale = 1.0 / cosh if abs(cosh) > 1e-9 else miter_limit
        scale = max(-miter_limit, min(miter_limit, scale))
        out.append((p[0] + mx * dist * scale, p[1] + my * dist * scale))
    return out


def bridge_holes(outer, holes):
    """Merge holes into their outline, giving ONE simple ring.

    A ring with holes is not something most triangulators take: the
    holes have to be connected to the outline by a pair of coincident
    "bridge" edges, after which the whole thing is a single simple
    polygon that happens to touch itself along those bridges.  Because
    the bridge has zero width, the merged ring encloses exactly
    outline-minus-holes -- which is what the area check in the
    self-test pins down.

    The construction is the standard one: take the hole's rightmost
    vertex, cast a ray to the right, and join it to a visible vertex of
    the outline.  Where the ray lands in the middle of an edge, the
    candidate is that edge's righter endpoint, and any reflex vertex
    lying inside the triangle formed can block the join -- so the
    blocker with the shallowest angle is used instead, which is
    guaranteed visible.

    Holes are merged rightmost-first, so a hole joined earlier is
    already part of the outline when a hole further left looks for
    something to see.
    """
    ring = list(as_ccw(outer))
    for hole in sorted((as_cw(h) for h in holes),
                       key=lambda h: -max(p[0] for p in h)):
        if len(hole) < 3:
            continue
        m = max(range(len(hole)), key=lambda i: hole[i][0])
        M = hole[m]

        # nearest edge crossing the ray M -> +x
        best_t, best_i, best_pt = float('inf'), None, None
        n = len(ring)
        for i in range(n):
            a, b = ring[i], ring[(i + 1) % n]
            if (a[1] > M[1]) == (b[1] > M[1]):
                continue
            t = a[0] + (M[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if t >= M[0] - 1e-12 and t < best_t:
                best_t, best_i, best_pt = t, i, (t, M[1])
        if best_i is None:
            continue                       # hole is not inside: skip it

        a, b = ring[best_i], ring[(best_i + 1) % n]
        p_idx = best_i if a[0] > b[0] else (best_i + 1) % n
        P = ring[p_idx]

        # any reflex vertex inside triangle (M, hit, P) blocks the view
        def inside(p, q, r, s):
            def side(u, v, w):
                return ((v[0] - u[0]) * (w[1] - u[1])
                        - (v[1] - u[1]) * (w[0] - u[0]))
            d1, d2, d3 = side(p, q, s), side(q, r, s), side(r, p, s)
            neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            return not (neg and pos)

        best_ang = None
        for i, R in enumerate(ring):
            if i == p_idx or R[0] < M[0]:
                continue
            prv, nxt = ring[(i - 1) % n], ring[(i + 1) % n]
            crs = ((R[0] - prv[0]) * (nxt[1] - R[1])
                   - (R[1] - prv[1]) * (nxt[0] - R[0]))
            if crs >= 0:                   # convex: cannot block
                continue
            if not inside(M, best_pt, P, R):
                continue
            ang = abs(math.atan2(R[1] - M[1], R[0] - M[0]))
            if best_ang is None or ang < best_ang:
                best_ang, p_idx = ang, i

        rotated = hole[m:] + hole[:m]
        ring = (ring[:p_idx + 1] + rotated + [rotated[0]]
                + ring[p_idx:])
    return ring


def arc_points(cx, cy, r, a0, a1, segments=12):
    """Polyline approximation of an arc, endpoints included."""
    out = []
    for k in range(segments + 1):
        t = a0 + (a1 - a0) * (k / segments)
        out.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return out


def dogbone(poly, radius, max_angle=math.pi * 0.5 + 1e-6, segments=16):
    """Add dog-bone relief at sharp interior corners of a cut ring.

    A round tool cannot reach into a corner sharper than its own radius,
    so the corner is over-cut with a circle tangent to both edges,
    pushed into the material along the corner bisector.  Only reflex
    corners of the CCW ring -- the ones that are inside corners of the
    CUT -- qualify, and only those tighter than `max_angle`.

    Returns (ring, n_relieved).  With radius 0 this is the identity,
    which is why the caller need not special-case "no tool".
    """
    if radius <= 0.0:
        return list(poly), 0
    ring = as_ccw(poly)
    n = len(ring)
    out = []
    hits = 0
    for i in range(n):
        p = ring[i]
        a = ring[(i - 1) % n]
        b = ring[(i + 1) % n]
        u = (a[0] - p[0], a[1] - p[1])
        v = (b[0] - p[0], b[1] - p[1])
        lu = math.hypot(*u) or 1.0
        lv = math.hypot(*v) or 1.0
        u = (u[0] / lu, u[1] / lu)
        v = (v[0] / lv, v[1] / lv)
        cross = u[0] * v[1] - u[1] * v[0]
        ang = math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1])))
        # cross > 0 here marks the corner that bites into the material
        if cross <= 0.0 or ang > max_angle:
            out.append(p)
            continue
        bx, by = u[0] + v[0], u[1] + v[1]
        bl = math.hypot(bx, by)
        if bl < 1e-12:
            out.append(p)
            continue
        bx, by = bx / bl, by / bl
        # u + v bisects the wedge between the two edges, which at a
        # reflex corner is the CUT side; so stepping back along it puts
        # the circle centre INSIDE the material, which is where an
        # over-cut has to reach
        cx, cy = p[0] - bx * radius, p[1] - by * radius
        base = math.atan2(p[1] - cy, p[0] - cx)
        # a FULL circle that starts and ends at the corner itself --
        # starting anywhere else (the antipode, say) would jump the
        # outline across the circle instead of looping around it.
        # Swept clockwise inside a CCW ring, so the loop SUBTRACTS its
        # area: a dog-bone removes material, it does not add any.
        out.extend(arc_points(cx, cy, radius,
                              base, base - 2.0 * math.pi, segments))
        hits += 1

    clean = []
    for q in out:
        if not clean or (abs(q[0] - clean[-1][0]) > 1e-15
                         or abs(q[1] - clean[-1][1]) > 1e-15):
            clean.append(q)
    return clean, hits


# ------------------------------------------------------------------ #

def _notched_square():
    """A CCW square with a rectangular notch cut in from the top rim --
    the shape a slot actually produces, reused by several checks."""
    sq = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    notch = [(0.8, 1.5), (1.2, 1.5), (1.2, 2.5), (0.8, 2.5)]
    rings, _ = difference_robust(sq, notch)
    return rings[0]


def _selftest():
    sq = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

    assert abs(area(sq) - 4.0) < 1e-12, "square area"
    assert abs(perimeter(sq) - 8.0) < 1e-12, "square perimeter"
    assert signed_area(sq) > 0.0, "CCW square has positive area"
    assert signed_area(sq[::-1]) < 0.0, "reversed square is CW"
    cx, cy = centroid(sq)
    assert abs(cx - 1.0) < 1e-12 and abs(cy - 1.0) < 1e-12, "centroid"

    assert point_in_polygon((1.0, 1.0), sq), "inside"
    assert not point_in_polygon((3.0, 1.0), sq), "outside"

    # line intervals: a horizontal line across the square
    iv = line_intervals(sq, (-1.0, 1.0), (1.0, 0.0))
    assert len(iv) == 1, "one interval across a square"
    assert abs(iv[0][0] - 1.0) < 1e-9 and abs(iv[0][1] - 3.0) < 1e-9, \
        "interval endpoints"

    # a line straight through two opposite vertices is answered
    # exactly, by convention rather than by perturbation: the diagonal
    # of the square, of length 2*sqrt(2)
    iv = line_intervals(sq, (0.0, 0.0), (1.0, 1.0))
    assert len(iv) == 1, f"the diagonal is one span, got {iv}"
    assert abs((iv[0][1] - iv[0][0]) - 2.0 * math.sqrt(2.0)) < 1e-9, \
        f"diagonal length {iv[0][1] - iv[0][0]}"

    # a line that merely grazes a corner from outside touches, but
    # encloses no material, so it yields nothing rather than a sliver
    tri = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
    assert line_intervals(tri, (0.0, 2.0), (1.0, 0.0)) == [], \
        "a tangential touch at the apex is not a span"

    # non-convex: a line across the arms of a U meets it in TWO
    # intervals -- the multi-interval case that decides whether a
    # crossing is assemblable (see slots.py)
    ushape = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (2.0, 3.0),
              (2.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0)]
    iv = line_intervals(ushape, (-1.0, 2.0), (1.0, 0.0))
    assert len(iv) == 2, f"U shape gives two intervals, got {len(iv)}"
    assert abs(iv[0][0] - 1.0) < 1e-9 and abs(iv[0][1] - 2.0) < 1e-9, \
        "left arm interval"
    assert abs(iv[1][0] - 3.0) < 1e-9 and abs(iv[1][1] - 4.0) < 1e-9, \
        "right arm interval"

    # difference: a notch cut in from the rim
    notch = [(0.8, 1.5), (1.2, 1.5), (1.2, 2.5), (0.8, 2.5)]
    rings, _ = difference_robust(sq, notch)
    assert len(rings) == 1, f"notch leaves one ring, got {len(rings)}"
    got = area(rings[0])
    assert abs(got - (4.0 - 0.4 * 0.5)) < 1e-6, \
        f"notched area {got}"

    # difference: a clip that misses entirely changes nothing
    far = [(9.0, 9.0), (10.0, 9.0), (10.0, 10.0), (9.0, 10.0)]
    rings, _ = difference_robust(sq, far)
    assert len(rings) == 1 and abs(area(rings[0]) - 4.0) < 1e-12, \
        "disjoint clip is a no-op"

    # difference: a clip spanning the middle cuts the subject in two
    band = [(-1.0, 0.9), (3.0, 0.9), (3.0, 1.1), (-1.0, 1.1)]
    rings, _ = difference_robust(sq, band)
    assert len(rings) == 2, f"band splits the square, got {len(rings)}"
    assert abs(sum(area(r) for r in rings) - (4.0 - 2.0 * 0.2)) < 1e-6, \
        "split areas"

    # a degeneracy must be survived by perturbation, not by luck
    flush = [(0.5, 1.0), (1.5, 1.0), (1.5, 2.0), (0.5, 2.0)]
    rings, nudged = difference_robust(sq, flush)
    assert rings, "flush clip still produces a result"

    # offset: growing a square by d adds 2d to each side
    big = offset_polygon(sq, 0.1)
    assert abs(area(big) - 2.2 * 2.2) < 1e-9, f"offset area {area(big)}"
    small = offset_polygon(sq, -0.1)
    assert abs(area(small) - 1.8 * 1.8) < 1e-9, "inward offset"
    assert offset_polygon(sq, 0.0) == list(sq), "zero offset is identity"

    # dogbone: no tool, no change; with a tool, only sharp inside
    # corners of the cut gain material
    same, hits = dogbone(sq, 0.0)
    assert same == list(sq) and hits == 0, "no dogbone without a tool"
    notched, hits = dogbone(_notched_square(), 0.05)
    assert hits == 2, \
        f"a rectangular notch has two inside corners, got {hits}"
    # --- bridging holes into one ring -----------------------------
    # the bridge has zero width, so the merged ring encloses exactly
    # the material: outline minus holes, to the last decimal
    outer10 = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    h1 = [(2.0, 2.0), (2.0, 4.0), (4.0, 4.0), (4.0, 2.0)]
    h2 = [(6.0, 6.0), (6.0, 8.0), (8.0, 8.0), (8.0, 6.0)]

    merged = bridge_holes(outer10, [h1])
    assert abs(area(merged) - (100.0 - 4.0)) < 1e-9, \
        f"one hole bridged: area {area(merged)}"
    merged = bridge_holes(outer10, [h1, h2])
    assert abs(area(merged) - (100.0 - 8.0)) < 1e-9, \
        f"two holes bridged: area {area(merged)}"
    assert len(merged) == 4 + 2 * (4 + 2), (
        "each hole adds its own ring plus the two repeated vertices "
        f"that make the there-and-back bridge: {len(merged)}")

    # order must not matter: the same holes given the other way round
    other = bridge_holes(outer10, [h2, h1])
    assert abs(area(other) - area(merged)) < 1e-9, \
        "hole order must not change the merged area"

    # no holes is the identity
    assert bridge_holes(outer10, []) == list(outer10), "no holes, no change"

    # a hole outside the outline is ignored rather than corrupting it
    stray = [(20.0, 20.0), (20.0, 21.0), (21.0, 21.0), (21.0, 20.0)]
    assert abs(area(bridge_holes(outer10, [stray])) - 100.0) < 1e-9, \
        "a hole that is not inside the outline is skipped"

    # a hole tucked against the outline still bridges cleanly -- this
    # is the case a scanline fill is most likely to drop
    tight = [(9.0, 4.0), (9.0, 6.0), (9.6, 6.0), (9.6, 4.0)]
    assert abs(area(bridge_holes(outer10, [tight])) - (100.0 - 1.2)) < 1e-9, \
        "a hole close to the rim bridges too"

    base_area = area(_notched_square())
    # the over-cut is a 16-gon, not a circle, so compare against the
    # exact polygonal area -- a pi*r^2 estimate is off by more than the
    # tolerance worth using here
    loop = 0.5 * 16 * 0.05 ** 2 * math.sin(2.0 * math.pi / 16)
    expect = base_area - 2 * loop
    assert abs(area(notched) - expect) < 1e-9, (
        "each dogbone over-cuts one tool circle out of the material: "
        f"expected ~{expect}, got {area(notched)}")

    return True
