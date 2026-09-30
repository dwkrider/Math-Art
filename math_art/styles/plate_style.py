
# Shared "Finger-Jointed Plates" style: every face of a closed polyhedral
# shell becomes a flat plate of real material thickness, joined to its
# neighbours by interlocking finger (box) joints cut for the edge's true
# dihedral angle -- a kit of parts for a laser cutter, with an assembled
# 3D preview and an SVG/DXF cut layout.
#
# THE CONSTRAINT.  A laser cuts square to the sheet; it cannot bevel.
# At 90 degrees that is the ordinary box joint.  At any other dihedral
# the two plates' slabs overlap in a parallelogram J (side t/|sin phi|)
# that a square-cut plate cannot fill exactly, so every edge carries a
# surface error of t|cos phi|, and the design question is where to put
# it.  Each plate's OUTER face lies on the polyhedron, so outer
# dimensions are true.  Along each edge the two plates take turns
# owning J: the owner's tab ends in a square cut at signed set-back
# c_tab, and the partner is cut back to the smallest depth s that
# clears the tab.  Both come from J, not from case-by-case formulas:
#
#   Filled   c_tab = min over J's corners of their projection onto the
#            plate's in-plane axis -- the tab covers J (no groove; on a
#            convex obtuse edge a lip stands proud by t|cos phi|).
#   Flush    on a convex edge, the smallest c_tab >= that whose tab
#            stays inside the solid: max(0, t cot phi).  On a reflex
#            edge J lies inside the solid, and Flush = Filled.
#   s        max, over (owner's tab) cut with (partner's slab), of the
#            projection onto the partner's axis.  A four-line clip.
#
# Closed forms (the self-test's oracles): the finger step s - c_tab is
# t cot(phi/2) (Filled) or t / sin(phi) (Flush) on acute edges,
# t tan(phi/2) (Filled) or t sin(phi) (Flush) on obtuse ones, and
# t(1+|cos phi|)/|sin phi| on reflex ones.
#
# VERTICES.  Each edge END is owned by one of its plates: at every
# vertex a total order is put on the plates that meet there, and the
# higher plate owns the end.  A total order has no cycles, so the top
# plate's fingers run right into the corner and the vertex closes --
# the pinwheel assignment would leave a hole.  Equal owners at both
# ends make the finger count odd, different owners make it even, and
# the end fingers are stretched until the outline's turn at the corner
# lands inside them.  What a square-cut corner cannot resolve
# analytically -- at a vertex of degree four or more, two plates that
# share no edge can both reach into it -- is resolved by one rule:
# wherever two plates' prisms still overlap, found exactly by clipping
# convex prism pieces, the lower-ranked plate cuts the finger run
# under the overlap back until it clears (a perpendicular cut removes
# a whole column).  The outline stays a finger profile throughout, so
# no general polygon booleans are needed.
#
# ASSEMBLY.  Finger side-walls in both plates are planes normal to the
# edge, so they mate at any angle.  A plate jointed on two or more
# non-parallel edges can then move only along its own normal, and on a
# convex shell with Flush joints every plate -- the last one included
# -- presses in along it.  Otherwise (Filled corner tabs can catch; a
# reflex edge hides its tab behind the partner) an order is found by
# taking the kit apart: repeatedly lift out a plate that clears every
# plate still present -- through the outside, or through the inside
# while the shell is open -- and reverse.
#
# References:
# - D. Beyer, S. Gurevich, S. Mueller, H.-T. Chen and P. Baudisch,
#   "Platener: Low-Fidelity Fabrication of 3D Objects by Substituting 3D
#   Print with Laser-Cut Plates", Proceedings of CHI 2015, ACM,
#   doi:10.1145/2702123.2702225.
# - P. Baudisch, A. Silber, Y. Kommana, M. Gruner, L. Wall, K. Reuss,
#   L. Heilman, R. Kovacs, D. Rechlitz and T. Roumen, "Kyub: A 3D Editor
#   for Modeling Sturdy Laser-Cut Objects", Proceedings of CHI 2019,
#   ACM, paper 566, doi:10.1145/3290605.3300796.
# - F. Heller, J. Thar, D. Lewandowski, M. Hartmann, P. Schoonbrood,
#   S. Stoenner, S. Voelker and J. Borchers, "CutCAD - An Open-Source
#   Tool to Design 3D Objects in 2D", Proceedings of DIS 2018, ACM,
#   pp. 1135-1139, doi:10.1145/3196709.3196800.
# - C. Zheng, E. Y.-L. Do and J. Budd, "Joinery: Parametric Joint
#   Generation for Laser Cut Assemblies", Proceedings of Creativity and
#   Cognition 2017, ACM, pp. 63-74, doi:10.1145/3059454.3059459.
# - C. Robeller and Y. Weinand, "Interlocking Folded Plate - Integral
#   Mechanical Attachment for Structural Wood Panels", International
#   Journal of Space Structures 30, no. 2 (2015), pp. 111-122.
# - C. Robeller and Y. Weinand, "A 3D Cutting Method for Integral 1DOF
#   Multiple-Tab-and-Slot Joints for Timber Plates, Using 5-Axis CNC
#   Cutting Technology", World Conference on Timber Engineering (WCTE
#   2016), Vienna, pp. 2576-2584.
# - N. Rogeau, P. Latteur and Y. Weinand, "An integrated design tool for
#   timber plate structures to generate joints geometry, fabrication
#   toolpath, and robot trajectories", Automation in Construction 130
#   (2021) 103875, doi:10.1016/j.autcon.2021.103875.
# - T. Roumen, I. Apel, J. Shigeyama, A. Muhammad and P. Baudisch,
#   "Kerf-Canceling Mechanisms: Making Laser-Cut Mechanisms Operate
#   Across Different Laser Cutters", Proceedings of UIST 2020, ACM,
#   doi:10.1145/3379337.3415895.
# - F. Heller, R. Ramakers and K. Luyten, "LaserSVG: Responsive
#   Laser-Cutter Templates", arXiv:2209.00116 (2022).

import math
from math import sqrt, sin, cos, pi

try:
    from . import net_style as _net
    from ..slicing import polyclip as pc
    from ..slicing import glyphs as _glyphs
    from ..slicing import layout as _layout
    from ..slicing.parts import Part
except ImportError:                                  # pragma: no cover
    from styles import net_style as _net
    from slicing import polyclip as pc
    from slicing import glyphs as _glyphs
    from slicing import layout as _layout
    from slicing.parts import Part

_sub, _add, _mul = _net._sub, _net._add, _net._mul
_dot, _cross, _len, _unit = _net._dot, _net._cross, _net._len, _net._unit


# Defaults, in millimetres.  Kept here rather than only on the operator
# so the headless builder and the self-test agree with the UI.
DEFAULTS = dict(
    thickness=3.0,          # material thickness t
    size=150.0,             # longest dimension of the assembled model
    finger=8.0,             # target finger width along an edge
    fit='FLUSH',            # 'FLUSH' or 'FILLED'
    clearance=0.05,         # gap between finger side-walls, per joint
    labels='INSIDE',        # 'INSIDE', 'OUTSIDE' or 'NONE'
    coplanar_deg=0.5,       # faces this close to coplanar become one plate
    planar_tol=0.1,         # worst allowed face non-flatness (mm)
    max_step=4.0,           # deepest allowed finger, in thicknesses
    shallow_step=1.0,       # shallower than this (in t) is reported
    limit=300,              # most plates a kit may have
)

PLATE_LIMIT = DEFAULTS['limit']

# deepest finger a Flush joint cuts on a reflex edge, in thicknesses
REFLEX_STEP = 1.5


# ---------------------------------------------------------------- #
#  small 2D helpers                                                 #
# ---------------------------------------------------------------- #

def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _clip_halfplane(poly, f):
    """Sutherland-Hodgman: keep the part of `poly` where f(p) >= 0."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        fp, fq = f(p), f(q)
        if fp >= 0.0:
            out.append(p)
        if (fp >= 0.0) != (fq >= 0.0):
            s = fp / (fp - fq)
            out.append((p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])))
    return out


def _segments_cross(a, b, c, d, eps):
    """Proper crossing (or overlap) of segments ab and cd."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    return False


def ring_is_simple(ring, eps=1e-9):
    """No two non-adjacent edges of the ring cross."""
    n = len(ring)
    if n < 3:
        return False
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(a, b, ring[j], ring[(j + 1) % n], eps):
                return False
    return True


def _drop_repeats(ring, eps):
    out = []
    for p in ring:
        if not out or abs(p[0] - out[-1][0]) > eps \
                or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    while len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps \
            and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    # collinear runs too: they cost the ear-clipper and the booleans
    k = 0
    while len(out) > 3 and k < len(out):
        a, b, c = out[k - 1], out[k], out[(k + 1) % len(out)]
        if abs(_cross2((b[0] - a[0], b[1] - a[1]),
                       (c[0] - b[0], c[1] - b[1]))) <= eps * eps and \
                (b[0] - a[0]) * (c[0] - b[0]) \
                + (b[1] - a[1]) * (c[1] - b[1]) >= 0.0:
            out.pop(k)
        else:
            k += 1
    return out


def ear_clip(ring):
    """Triangulate a simple polygon (any winding) into index triples of
    a CCW copy of it; returns (ccw_ring, triangles).

    Of the valid ears, the most compact (shortest longest side) is cut
    first.  A finger outline is a comb, and plain ear clipping fans its
    teeth into long slivers reaching across the plate; compact ears keep
    each finger's triangles local, which is what lets bounding boxes
    prune the prism-overlap tests."""
    P = pc.as_ccw(list(ring))
    n = len(P)
    idx = list(range(n))
    tris = []

    def area2(i0, i1, i2):
        a, b, c = P[i0], P[i1], P[i2]
        return _cross2((b[0] - a[0], b[1] - a[1]), (c[0] - a[0], c[1] - a[1]))

    def inside(p, a, b, c):
        return (_cross2((b[0] - a[0], b[1] - a[1]),
                        (p[0] - a[0], p[1] - a[1])) >= 0.0 and
                _cross2((c[0] - b[0], c[1] - b[1]),
                        (p[0] - b[0], p[1] - b[1])) >= 0.0 and
                _cross2((a[0] - c[0], a[1] - c[1]),
                        (p[0] - c[0], p[1] - c[1])) >= 0.0)

    def d2(i, j):
        return (P[i][0] - P[j][0]) ** 2 + (P[i][1] - P[j][1]) ** 2

    # doubly linked ring; ear status is recomputed only for the two
    # neighbours of each vertex cut, so the whole run is ~O(n * reflex)
    nxt = {idx[k]: idx[(k + 1) % n] for k in range(n)}
    prv = {idx[k]: idx[k - 1] for k in range(n)}
    alive = set(idx)

    def is_reflex(i):
        return area2(prv[i], i, nxt[i]) <= 1e-14

    reflex = {i for i in alive if is_reflex(i)}

    def ear_size(i):
        i0, i2 = prv[i], nxt[i]
        if i in reflex:
            return None
        a, b, c = P[i0], P[i], P[i2]
        for j in reflex:
            if j in (i0, i, i2) or P[j] == a or P[j] == b or P[j] == c:
                continue
            if inside(P[j], a, b, c):
                return None
        return max(d2(i0, i), d2(i, i2), d2(i0, i2))

    size = {i: ear_size(i) for i in alive}
    while len(alive) > 3:
        cand = [(sz, i) for i, sz in size.items() if sz is not None]
        if not cand:
            # a vertex that stopped being reflex can unblock ears that
            # were never re-checked: refresh everything before giving up
            size = {i: ear_size(i) for i in alive}
            cand = [(sz, i) for i, sz in size.items() if sz is not None]
        if not cand:
            # numerically flat leftovers: drop the flattest vertex
            i = min(alive, key=lambda i: abs(area2(prv[i], i, nxt[i])))
        else:
            i = min(cand)[1]
            tris.append((prv[i], i, nxt[i]))
        a0, a2 = prv[i], nxt[i]
        nxt[a0], prv[a2] = a2, a0
        alive.discard(i)
        reflex.discard(i)
        del size[i]
        for j in (a0, a2):
            if is_reflex(j):
                reflex.add(j)
            else:
                reflex.discard(j)
        for j in (a0, a2):
            size[j] = ear_size(j)
    if len(alive) == 3:
        i = next(iter(alive))
        idx = [prv[i], i, nxt[i]]
    if len(idx) == 3 and area2(*idx) > 1e-14:
        tris.append(tuple(idx))
    return P, tris


# ---------------------------------------------------------------- #
#  the edge cross-section                                           #
# ---------------------------------------------------------------- #

def section_frames(phi):
    """Cross-section frame of an edge with interior dihedral `phi`.

    The edge is the origin.  Plate A lies along +x with its inward
    normal +y; plate B's in-plane direction b makes the angle phi with
    A's through the solid, and nB is B's inward normal.  Valid for any
    phi in (0, 2 pi): on a reflex edge both slabs still sit on the
    solid side of their faces."""
    a = (1.0, 0.0)
    nA = (0.0, 1.0)
    b = (cos(phi), sin(phi))
    nB = (sin(phi), -cos(phi))
    return a, nA, b, nB


def joint_corners(phi, t):
    """The four corners of J (the slabs' overlap), and their positions
    along A's in-plane axis: x = (w + y cos phi) / sin phi."""
    sp, cp = sin(phi), cos(phi)
    xs = []
    for y in (0.0, t):
        for w in (0.0, t):
            xs.append((w + y * cp) / sp)
    return xs


def tab_end(phi, t, fit, reflex_step=REFLEX_STEP):
    """Signed set-back of the owner's tab end, along its own plate.

    On a reflex edge J lies behind both faces, inside the solid, so the
    fingers interlock only behind the edge line and a Flush joint need
    not fill all of J: the tab is cut short where the finger step
    reaches `reflex_step` thicknesses.  What it leaves unfilled is a
    void inside the model.  Above 270 degrees nothing of it shows at
    all; below, the partner's notch on the surface shrinks with the
    tab.  Filled keeps covering J, however deep that takes it."""
    xs = joint_corners(phi, t)
    filled = min(xs)
    if fit == 'FLUSH' and phi < pi:
        return max(0.0, t * cos(phi) / sin(phi))
    if fit == 'FLUSH' and reflex_step is not None:
        cap = reflex_step * t
        if slot_root(phi, t, filled) - filled <= cap:
            return filled
        # the step falls monotonically as the tab is shortened towards
        # the edge line (where it is zero), so bisect for the cap
        lo, hi = filled, 0.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if slot_root(phi, t, mid) - mid > cap:
                lo = mid
            else:
                hi = mid
        return hi
    return filled


def slot_root(phi, t, c_tab):
    """Smallest cut-back of the partner that clears a tab ending at
    `c_tab`: the owner's half-strip cut with the partner's slab,
    projected onto the partner's axis."""
    a, nA, b, nB = section_frames(phi)
    big = abs(c_tab) + 1000.0 * t
    tab = [(c_tab, 0.0), (big, 0.0), (big, t), (c_tab, t)]
    X = _clip_halfplane(tab, lambda p: p[0] * nB[0] + p[1] * nB[1])
    X = _clip_halfplane(X, lambda p: t - (p[0] * nB[0] + p[1] * nB[1]))
    if not X:
        return 0.0
    return max(p[0] * b[0] + p[1] * b[1] for p in X)


def edge_joint(phi, t, fit, reflex_step=REFLEX_STEP):
    """(c_tab, s) for an edge: tab end and partner root.  The joint is
    symmetric under swapping the plates, so one pair serves both."""
    c = tab_end(phi, t, fit, reflex_step)
    return c, slot_root(phi, t, c)


# ---------------------------------------------------------------- #
#  plates: merging coplanar faces, rings, sides                      #
# ---------------------------------------------------------------- #

class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _area_vector(P):
    n = [0.0, 0.0, 0.0]
    m = len(P)
    for i in range(m):
        a, b = P[i], P[(i + 1) % m]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    return (0.5 * n[0], 0.5 * n[1], 0.5 * n[2])


def _group_faces(V, F, coplanar_deg, planar_tol):
    """Union faces across edges where they are coplanar."""
    normals = [_unit(_area_vector([V[i] for i in f])) for f in F]
    uf = _UF(len(F))
    cmax = cos(math.radians(coplanar_deg))
    for uses in _net.edge_table(F).values():
        (a, _ka), (b, _kb) = uses
        if _dot(normals[a], normals[b]) < cmax:
            continue
        # angle alone is not coplanarity: parallel faces that happen to
        # share an edge are, but check the far vertices anyway
        n = normals[a]
        o = V[F[a][0]]
        if all(abs(_dot(_sub(V[i], o), n)) <= planar_tol for i in F[b]):
            uf.union(a, b)
    groups = {}
    for fi in range(len(F)):
        groups.setdefault(uf.find(fi), []).append(fi)
    return [groups[k] for k in sorted(groups)]


def _plate_rings(V, F, faces, face_plate):
    """Boundary rings of a group of faces, as lists of directed edges
    (a, b, neighbour_plate), material on the left."""
    directed = {}
    for fi in faces:
        f = F[fi]
        for k in range(len(f)):
            directed[(f[k], f[(k + 1) % len(f)])] = fi
    owner_of = {}
    for fi, f in enumerate(F):
        for k in range(len(f)):
            owner_of[(f[k], f[(k + 1) % len(f)])] = fi
    out_edges = {}
    for (a, b) in directed:
        if (b, a) in directed:
            continue
        nb = face_plate[owner_of[(b, a)]]
        out_edges.setdefault(a, []).append((a, b, nb))
    rings = []
    while out_edges:
        a0 = min(out_edges)
        ring = []
        cur = a0
        while True:
            lst = out_edges.get(cur)
            if not lst:
                break
            e = lst.pop()
            if not lst:
                del out_edges[cur]
            ring.append(e)
            cur = e[1]
            if cur == a0:
                break
        if ring:
            rings.append(ring)
    return rings


def _ring_sides(ring):
    """Split a ring of directed edges into maximal runs sharing one
    neighbour plate.  Each side: (vertex list, neighbour)."""
    m = len(ring)
    nbs = [e[2] for e in ring]
    start = 0
    for k in range(m):
        if nbs[k] != nbs[k - 1]:
            start = k
            break
    sides = []
    k = 0
    while k < m:
        i = (start + k) % m
        verts = [ring[i][0], ring[i][1]]
        nb = nbs[i]
        k += 1
        while k < m and nbs[(start + k) % m] == nb:
            verts.append(ring[(start + k) % m][1])
            k += 1
        sides.append((verts, nb))
    return sides


# ---------------------------------------------------------------- #
#  convex prism pieces and their intersections                       #
# ---------------------------------------------------------------- #

def _prism(frame, tri2, t):
    """Triangular prism under a 2D triangle of a plate, as (faces,
    planes): faces are 3D polygons, planes (n, d) with n.x <= d
    inside."""
    o, u, w, n = frame
    top = [_add(o, _add(_mul(u, p[0]), _mul(w, p[1]))) for p in tri2]
    bot = [_sub(p, _mul(n, t)) for p in top]
    faces = [list(top), list(reversed(bot))]
    planes = [(n, _dot(n, top[0])),
              (_mul(n, -1.0), -_dot(n, bot[0]))]
    for k in range(3):
        a, b = top[k], top[(k + 1) % 3]
        faces.append([a, bot[k], bot[(k + 1) % 3], b])
        sn = _unit(_cross(_sub(b, a), n))
        planes.append((sn, _dot(sn, a)))
    return faces, planes, top + bot


def _clip_poly3(faces, plane, eps):
    """Clip a convex polytope (list of face polygons) by n.x <= d."""
    nrm, d = plane
    out, cap = [], []
    for f in faces:
        g = []
        m = len(f)
        for i in range(m):
            p, q = f[i], f[(i + 1) % m]
            fp, fq = d - _dot(nrm, p), d - _dot(nrm, q)
            if fp >= -eps:
                g.append(p)
            if (fp >= -eps) != (fq >= -eps):
                s = fp / (fp - fq)
                x = _add(p, _mul(_sub(q, p), s))
                g.append(x)
                cap.append(x)
        if len(g) >= 3:
            out.append(g)
    if len(cap) >= 3:
        c = _mul(cap[0], 0.0)
        for p in cap:
            c = _add(c, p)
        c = _mul(c, 1.0 / len(cap))
        e1 = _unit(_sub(cap[0], c)) if _len(_sub(cap[0], c)) > 0 else \
            _unit(_cross(nrm, (1.0, 0.0, 0.0)))
        e2 = _cross(nrm, e1)
        cap.sort(key=lambda p: math.atan2(_dot(_sub(p, c), e2),
                                          _dot(_sub(p, c), e1)))
        out.append(cap)
    return out


def _volume(faces):
    if not faces:
        return 0.0
    pts = [p for f in faces for p in f]
    c = _mul(pts[0], 0.0)
    for p in pts:
        c = _add(c, p)
    c = _mul(c, 1.0 / len(pts))
    vol = 0.0
    for f in faces:
        for k in range(1, len(f) - 1):
            a, b, cc = _sub(f[0], c), _sub(f[k], c), _sub(f[k + 1], c)
            vol += abs(_dot(a, _cross(b, cc))) / 6.0
    return vol


def _separated(A, B, eps):
    """True when a face plane of either prism has the other entirely
    on its outside: a cheap, exact rejection before any clipping."""
    for (P, Q) in ((A, B), (B, A)):
        for (nrm, d) in P[1]:
            if all(_dot(nrm, q) > d + eps for q in Q[2]):
                return True
    return False


def prism_overlap(A, B, eps):
    """Intersection of two convex prisms (from `_prism`): (volume,
    vertices)."""
    if _separated(A, B, eps):
        return 0.0, []
    faces = A[0]
    for pl in B[1]:
        # a plane with all of A already inside it clips nothing
        nrm, d = pl
        if all(_dot(nrm, q) <= d + eps for f in faces for q in f):
            continue
        faces = _clip_poly3(faces, pl, eps)
        if not faces:
            return 0.0, []
    return _volume(faces), [p for f in faces for p in f]


def _bbox3(pts):
    return (min(p[0] for p in pts), min(p[1] for p in pts),
            min(p[2] for p in pts), max(p[0] for p in pts),
            max(p[1] for p in pts), max(p[2] for p in pts))


def _bbox_hit(a, b, pad):
    return not (a[3] + pad < b[0] or b[3] + pad < a[0] or
                a[4] + pad < b[1] or b[4] + pad < a[1] or
                a[5] + pad < b[2] or b[5] + pad < a[2])


def bridge_holes(outer, holes):
    """One weakly simple ring equivalent to `outer` minus `holes`: each
    hole is spliced in along a bridge from its rightmost vertex to an
    outline vertex it can see (holes taken right to left, so a later
    bridge never has to cross an earlier one)."""
    ring = pc.as_ccw(list(outer))
    for h in sorted((pc.as_cw(list(h)) for h in holes),
                    key=lambda h: -max(p[0] for p in h)):
        m = max(range(len(h)), key=lambda k: (h[k][0], h[k][1]))
        M = h[m]
        others = [g for g in holes]

        def visible(V):
            for R in [ring] + others:
                n = len(R)
                for k in range(n):
                    a, b = R[k], R[(k + 1) % n]
                    if a == V or b == V or a == M or b == M:
                        continue
                    if _segments_cross(M, V, a, b, 1e-12):
                        return False
            return True

        cands = sorted(range(len(ring)),
                       key=lambda k: (ring[k][0] - M[0]) ** 2
                       + (ring[k][1] - M[1]) ** 2)
        v = next((k for k in cands if visible(ring[k])), cands[0])
        hole_seq = h[m:] + h[:m] + [M]
        ring = ring[:v + 1] + hole_seq + ring[v:]
    return ring


def plate_pieces(plate, t, ring=None):
    """Convex prism pieces of a plate (or of one ring given in its
    frame), with their bounding boxes."""
    if ring is None:
        ring = plate['outer']
        if plate.get('holes'):
            ring = bridge_holes(ring, plate['holes'])
    P, tris = ear_clip(ring)
    out = []
    for (i, j, k) in tris:
        pr = _prism(plate['frame'], (P[i], P[j], P[k]), t)
        out.append((pr, _bbox3([p for f in pr[0] for p in f])))
    return out


# ---------------------------------------------------------------- #
#  the builder                                                      #
# ---------------------------------------------------------------- #

def _settings(kw):
    s = dict(DEFAULTS)
    for k, v in kw.items():
        if k not in s:
            raise TypeError(f"unknown plate setting {k!r}")
        s[k] = v
    return s


def _to3(frame, p):
    o, u, w, _n = frame
    return _add(o, _add(_mul(u, p[0]), _mul(w, p[1])))


def _to2(frame, p):
    o, u, w, _n = frame
    d = _sub(p, o)
    return (_dot(d, u), _dot(d, w))


def build_plates(V, F, **kw):
    """Turn a closed polyhedral shell into finger-jointed plates.

    All lengths are millimetres: the shell is scaled so its longest
    dimension is `size`.  Returns a dict

        plates   [{outer, holes, frame, label, rank, error, ...}]
                 outlines in the plate's own 2D frame, outer face up
        joints   per side: plates, phi, c_tab, s, n, owners, number
        scale    mm per model unit
        order    an assembly order (plate indices) or None
        report   counts, angle and depth ranges, warnings

    Raises ValueError with a user-readable message when the shell
    cannot be made into plates at all.
    """
    S = _settings(kw)
    t = float(S['thickness'])
    if t <= 0.0:
        raise ValueError("material thickness must be positive")

    diag = _net._bbox_diag(V)
    V, F = _net.weld_vertices(V, F, 1e-6 * diag)
    bad = _net.check_surface(V, F)
    if bad:
        raise ValueError(bad)
    F = _net.orient_consistently(V, F)
    # the Net's majority vote on "outward" can tie (a torus with a
    # diamond cross-section faces the hole as often as away from it),
    # and inside-out plates would swap every convex edge for a reflex
    # one; the signed volume of the closed shell cannot tie
    vol = 0.0
    for f in F:
        a = V[f[0]]
        for k in range(1, len(f) - 1):
            vol += _dot(a, _cross(V[f[k]], V[f[k + 1]]))
    if vol < 0.0:
        F = [list(reversed(f)) for f in F]

    lo = [min(v[k] for v in V) for k in range(3)]
    hi = [max(v[k] for v in V) for k in range(3)]
    ext = max(hi[k] - lo[k] for k in range(3)) or 1.0
    scale = S['size'] / ext
    Vm = [(v[0] * scale, v[1] * scale, v[2] * scale) for v in V]

    # --- every face must be flat at output scale -------------------
    worst = 0.0
    for f in F:
        P = [Vm[i] for i in f]
        n = _unit(_area_vector(P))
        c = _mul(P[0], 0.0)
        for p in P:
            c = _add(c, p)
        c = _mul(c, 1.0 / len(P))
        worst = max(worst, max(abs(_dot(_sub(p, c), n)) for p in P))
    if worst > S['planar_tol']:
        raise ValueError(
            f"its faces are not flat (up to {worst:.2f} mm out of plane "
            f"at {S['size']:.0f} mm), and a plate cannot bend")

    groups = _group_faces(Vm, F, S['coplanar_deg'], S['planar_tol'])
    if len(groups) > S['limit']:
        raise ValueError(
            f"{len(groups)} plates is more than the {S['limit']} this "
            f"style makes; it is meant for polyhedra, not smooth meshes")
    face_plate = [0] * len(F)
    for pi_, g in enumerate(groups):
        for fi in g:
            face_plate[fi] = pi_

    # --- plate frames and boundary rings ---------------------------
    plates = []
    for pi_, g in enumerate(groups):
        av = (0.0, 0.0, 0.0)
        for fi in g:
            av = _add(av, _area_vector([Vm[i] for i in F[fi]]))
        n = _unit(av)
        rings = _plate_rings(Vm, F, g, face_plate)
        o = Vm[rings[0][0][0]]
        u = _sub(Vm[rings[0][0][1]], o)
        u = _unit(_sub(u, _mul(n, _dot(u, n))))
        w = _cross(n, u)
        plates.append({'index': pi_, 'faces': g, 'frame': (o, u, w, n),
                       'rings_v': [_ring_sides(r) for r in rings],
                       'sides': len(set(
                           e[2] for r in rings for e in r)),
                       'error': None, 'label': f"P{pi_ + 1}",
                       'ngon': (len(F[g[0]]) if len(g) == 1
                                else len(rings[0]))})

    # --- joints: one record per shared side ------------------------
    joints = {}
    for P in plates:
        for sides in P['rings_v']:
            for verts, nb in sides:
                key = (min(P['index'], nb), max(P['index'], nb),
                       min(verts[0], verts[-1]), max(verts[0], verts[-1]))
                rec = joints.setdefault(key, {'plates': (key[0], key[1]),
                                              'walk': {}})
                rec['walk'][P['index']] = (verts[0], verts[-1])
    for key, J in joints.items():
        A, B = J['plates']
        va, vb = J['walk'][A]
        pa, pb = Vm[va], Vm[vb]
        e = _unit(_sub(pb, pa))
        nA, nB = plates[A]['frame'][3], plates[B]['frame'][3]
        a_in = _unit(_cross(nA, e))
        b_in = _unit(_cross(nB, _mul(e, -1.0)))
        ang = math.acos(max(-1.0, min(1.0, _dot(a_in, b_in))))
        convex = _dot(b_in, nA) < 0.0
        phi = ang if convex else 2.0 * pi - ang
        J.update(va=va, vb=vb, L=_len(_sub(pb, pa)), phi=phi,
                 convex=convex)
        if abs(sin(phi)) < 1e-6:
            raise ValueError("two neighbouring faces are coplanar but "
                             "were not merged")
        c, s = edge_joint(phi, t, S['fit'])
        J.update(c=c, s=s, step=s - c)

    # --- depth guards ---------------------------------------------
    deep = [J for J in joints.values() if J['step'] > S['max_step'] * t]
    if deep:
        worst = max(deep, key=lambda J: J['step'])
        msg = (f"{len(deep)} edge(s) need fingers deeper than "
               f"{S['max_step']:g} x the thickness (worst "
               f"{worst['step'] / t:.1f} t at "
               f"{math.degrees(worst['phi']):.0f} degrees)")
        if S['fit'] == 'FILLED':
            ok = all(edge_joint(J['phi'], t, 'FLUSH')[1]
                     - edge_joint(J['phi'], t, 'FLUSH')[0]
                     <= S['max_step'] * t for J in joints.values())
            if ok:
                msg += "; Flush joint edges would work"
        raise ValueError(msg)

    # --- end ownership: an acyclic order at every vertex -----------
    at_vertex = {}
    for key, J in joints.items():
        for v in (J['va'], J['vb']):
            at_vertex.setdefault(v, set()).update(J['plates'])
    owned = [0] * len(plates)
    rank = {}
    for v in sorted(at_vertex):
        order = sorted(at_vertex[v], key=lambda p: (owned[p], p))
        # first in `order` ranks highest
        rank[v] = {p: len(order) - k for k, p in enumerate(order)}
        for key, J in joints.items():
            if v in (J['va'], J['vb']):
                A, B = J['plates']
                owned[A if rank[v][A] > rank[v][B] else B] += 1
    for key, J in joints.items():
        A, B = J['plates']
        J['own_a'] = A if rank[J['va']][A] > rank[J['va']][B] else B
        J['own_b'] = A if rank[J['vb']][A] > rank[J['vb']][B] else B

    # --- how long each end finger must be ------------------------------
    # At a plate corner the two sides' end runs are straight lines at
    # their own depths (tab end or root, by who owns that end), and the
    # outline turns where they meet.  That meeting point must fall
    # inside both end runs or the outline folds back across a finger,
    # so each joint end needs an end run at least that long.
    need = {}
    for P in plates:
        me = P['index']
        for sides in P['rings_v']:
            geo = []
            for verts, nb in sides:
                key = (min(me, nb), max(me, nb),
                       min(verts[0], verts[-1]), max(verts[0], verts[-1]))
                A2 = _to2(P['frame'], Vm[verts[0]])
                B2 = _to2(P['frame'], Vm[verts[-1]])
                L = sqrt((B2[0] - A2[0]) ** 2 + (B2[1] - A2[1]) ** 2)
                e2 = ((B2[0] - A2[0]) / L, (B2[1] - A2[1]) / L)
                geo.append((key, verts[0], verts[-1], A2, e2, L))
            for i in range(len(geo)):
                kp, _vp0, vp1, pA2, pe2, pL = geo[i - 1]
                kh, vh0, _vh1, A2, e2, L = geo[i]
                den = _cross2(pe2, e2)
                if abs(den) < 1e-9:
                    continue
                pn2 = (-pe2[1], pe2[0])
                n2 = (-e2[1], e2[0])
                # not only the end runs' own lines: the next runs in
                # (the other depth) must not cross near the corner
                # either, so take every depth pairing
                Jp, Jh = joints[kp], joints[kh]
                for dp in (Jp['c'], Jp['s']):
                    for dh in (Jh['c'], Jh['s']):
                        p0 = (pA2[0] + pn2[0] * dp, pA2[1] + pn2[1] * dp)
                        q0 = (A2[0] + n2[0] * dh, A2[1] + n2[1] * dh)
                        lam = _cross2((q0[0] - p0[0], q0[1] - p0[1]),
                                      e2) / den
                        corner = (p0[0] + pe2[0] * lam,
                                  p0[1] + pe2[1] * lam)
                        u_here = ((corner[0] - A2[0]) * e2[0]
                                  + (corner[1] - A2[1]) * e2[1])
                        back_prev = pL - lam
                        for (k, v, d) in ((kp, vp1, back_prev),
                                          (kh, vh0, u_here)):
                            need[(k, v)] = max(need.get((k, v), 0.0), d)

    # --- finger patterns --------------------------------------------
    # One pattern per edge, read by each plate in its own direction.
    # Segments alternate owners; the end segments belong to the end
    # owners and are stretched to reach past the corner turn.
    w_target = max(float(S['finger']), 2.0 * S['clearance'] + 1e-3)
    pad = 0.5 * t
    fingerless = 0
    for key, J in joints.items():
        L = J['L']
        odd = J['own_a'] == J['own_b']
        e0 = max(0.0, need.get((key, J['va']), 0.0)) + pad
        e1 = max(0.0, need.get((key, J['vb']), 0.0)) + pad
        n = max(1, int(round(L / w_target)))
        if odd and n % 2 == 0:
            n += 1
        if not odd and n % 2 == 1:
            n += 1
        # shrink the count until the inner fingers and the stretched
        # ends both fit (never below the parity minimum)
        while True:
            if n <= 2:
                break
            inner = (L - max(e0, L / n) - max(e1, L / n)) / (n - 2)
            if inner >= max(0.5 * w_target, 1.5 * t):
                break
            n -= 2
        J['n'] = n
        other = {J['plates'][0]: J['plates'][1],
                 J['plates'][1]: J['plates'][0]}
        if n == 1:
            cuts = [0.0, L]
        elif n == 2:
            m = min(max(0.5 * L, e0), L - e1) if e0 + e1 <= L else 0.5 * L
            cuts = [0.0, m, L]
        else:
            a0 = max(e0, L / n)
            a1 = max(e1, L / n)
            inner = (L - a0 - a1) / (n - 2)
            cuts = [0.0] + [a0 + inner * k for k in range(n - 1)] + [L]
            cuts[-2] = L - a1
        segs = []
        for k in range(n):
            owner = J['own_a'] if k % 2 == 0 else other[J['own_a']]
            segs.append((cuts[k], cuts[k + 1], owner))
        J['segs'] = segs
        if n == 1:
            fingerless += 1

    # --- edge numbers -------------------------------------------------
    for num, key in enumerate(sorted(joints), start=1):
        joints[key]['number'] = num

    # --- outlines, then the vertex clean-up ---------------------------
    clr = 0.5 * float(S['clearance'])
    eps = 1e-9 * S['size']
    for P in plates:
        _profile(P, joints, Vm, clr)
        _trace(P, eps)
    cleaned, residual = _vertex_cleanup(plates, rank, at_vertex, joints, Vm,
                                        t, S['size'], eps)

    # --- assembly order ----------------------------------------------

    # --- labels and edge numbers --------------------------------------
    for P in plates:
        P['engrave'] = []
    if S['labels'] != 'NONE':
        _engrave(plates, joints, Vm, S)

    steps = [J['step'] for J in joints.values()]
    phis = [math.degrees(J['phi']) for J in joints.values()]
    shallow = sum(1 for x in steps if x < S['shallow_step'] * t - 1e-9)
    report = {
        'plates': len(plates), 'faces': len(F), 'edges': len(joints),
        'phi_min': min(phis), 'phi_max': max(phis),
        'step_min': min(steps), 'step_max': max(steps),
        'shallow': shallow, 'fingerless': fingerless,
        'reflex': sum(1 for J in joints.values() if not J['convex']),
        'cleaned': cleaned, 'residual': residual,
        'errors': sum(1 for P in plates if P['error']),
        'lips': (sum(1 for J in joints.values()
                     if J['convex'] and J['c'] < -1e-9)),
        'lip_height': max([t * abs(cos(J['phi'])) for J in joints.values()
                           if J['convex'] and J['c'] < -1e-9] or [0.0]),
        'thickness': t, 'fit': S['fit'],
    }
    result = {'plates': plates, 'joints': joints, 'scale': scale,
              'order': None, 'forced': [], 'report': report, 'settings': S}
    result['order'] = assembly_order(result)
    report['order_known'] = result['order'] is not None
    report['forced'] = len(result['forced'])
    return result


# ---------------------------------------------------------------- #
#  vertex clean-up                                                  #
# ---------------------------------------------------------------- #

def _profile(P, joints, Vm, clr):
    """Each side of the plate as a run list [u0, u1, depth] in the
    side's own coordinates (u along it from its start, depth measured
    inward), from the edge's shared finger pattern."""
    me = P['index']
    frame = P['frame']
    prof = []
    for sides in P['rings_v']:
        ring = []
        for verts, nb in sides:
            key = (min(me, nb), max(me, nb),
                   min(verts[0], verts[-1]), max(verts[0], verts[-1]))
            J = joints[key]
            A2 = _to2(frame, Vm[verts[0]])
            B2 = _to2(frame, Vm[verts[-1]])
            L = sqrt((B2[0] - A2[0]) ** 2 + (B2[1] - A2[1]) ** 2)
            e2 = ((B2[0] - A2[0]) / L, (B2[1] - A2[1]) / L)
            n2 = (-e2[1], e2[0])
            segs = J['segs']
            if verts[0] != J['va']:
                segs = [(J['L'] - u1, J['L'] - u0, o)
                        for (u0, u1, o) in reversed(segs)]
            k_scale = L / J['L']
            runs = []
            for i, (u0, u1, own) in enumerate(segs):
                u0, u1 = u0 * k_scale, u1 * k_scale
                mine = own == me
                # clearance: every tab gives up clr at each side it
                # shares with the partner's tab
                sgn = 1.0 if mine else -1.0
                if i > 0:
                    u0 += sgn * clr
                if i < len(segs) - 1:
                    u1 -= sgn * clr
                runs.append([u0, u1, J['c'] if mine else J['s']])
            ring.append({'key': key, 'A2': A2, 'e2': e2, 'n2': n2, 'L': L,
                         'runs': runs, 's': J['s']})
        prof.append(ring)
    P['prof'] = prof


def _corner(pv, sd):
    """Where the outline turns at the corner between side `pv` and the
    side `sd` after it: (runs dropped at the end of pv, runs dropped at
    the start of sd, the point), or None if nothing fits.

    Normally the last run of one side meets the first run of the next.
    But a run cut deep near a corner (the vertex clean-up does this) can
    swallow whole fingers of the other side, and then the outline has to
    turn where the runs that survive actually meet: the pair whose
    meeting point lies within both of them, dropping fewest runs."""
    pA2, pe2, pn2, pruns = pv['A2'], pv['e2'], pv['n2'], pv['runs']
    A2, e2, n2, runs = sd['A2'], sd['e2'], sd['n2'], sd['runs']
    den = _cross2(pe2, e2)
    if abs(den) < 1e-9:
        return None
    best = None
    for k in range(len(pruns) + len(runs) - 1):
        for dp in range(0, k + 1):
            dh = k - dp
            if dp >= len(pruns) or dh >= len(runs):
                continue
            rp, rh = pruns[-1 - dp], runs[dh]
            p0 = (pA2[0] + pn2[0] * rp[2], pA2[1] + pn2[1] * rp[2])
            q0 = (A2[0] + n2[0] * rh[2], A2[1] + n2[1] * rh[2])
            lam = _cross2((q0[0] - p0[0], q0[1] - p0[1]), e2) / den
            pt = (p0[0] + pe2[0] * lam, p0[1] + pe2[1] * lam)
            u = (pt[0] - A2[0]) * e2[0] + (pt[1] - A2[1]) * e2[1]
            # the end run may run past its side's end, the first run
            # start before its side's start; inner runs must contain it
            ok_p = lam >= rp[0] - 1e-9 and (dp == 0 or lam <= rp[1] + 1e-9)
            ok_h = u <= rh[1] + 1e-9 and (dh == 0 or u >= rh[0] - 1e-9)
            if ok_p and ok_h:
                best = (dp, dh, pt)
                break
        if best is not None:
            break
    return best


def _trace(P, eps):
    """Walk the profile into outline rings; set P's outer, holes and
    error."""
    rings2 = []
    corner_bad = False
    for prof in P['prof']:
        m = len(prof)
        corners = []
        for i in range(m):
            sd, pv = prof[i], prof[i - 1]
            c = _corner(pv, sd)
            if c is None:
                pA2, pe2, pn2, pL = pv['A2'], pv['e2'], pv['n2'], pv['L']
                A2, n2 = sd['A2'], sd['n2']
                dp, dh = pv['runs'][-1][2], sd['runs'][0][2]
                if abs(_cross2(pe2, sd['e2'])) < 1e-9:
                    # straight-through corner: a jog, not an intersection
                    c = (0, 0, [(pA2[0] + pe2[0] * pL + pn2[0] * dp,
                                 pA2[1] + pe2[1] * pL + pn2[1] * dp),
                                (A2[0] + n2[0] * dh, A2[1] + n2[1] * dh)])
                else:
                    corner_bad = True
                    c = (0, 0, [(A2[0] + n2[0] * dh, A2[1] + n2[1] * dh)])
            else:
                c = (c[0], c[1], [c[2]])
            corners.append(c)
        ring = []
        for i in range(m):
            sd = prof[i]
            A2, e2, n2, runs = sd['A2'], sd['e2'], sd['n2'], sd['runs']
            ring.extend(corners[i][2])
            first = corners[i][1]
            last = len(runs) - 1 - corners[(i + 1) % m][0]
            if first > last:
                # both corners swallowed this whole side
                corner_bad = True
                continue
            for j in range(first, last):
                u = runs[j][1]
                u_next = runs[j + 1][0]
                ring.append((A2[0] + e2[0] * u + n2[0] * runs[j][2],
                             A2[1] + e2[1] * u + n2[1] * runs[j][2]))
                ring.append((A2[0] + e2[0] * u_next + n2[0] * runs[j + 1][2],
                             A2[1] + e2[1] * u_next + n2[1] * runs[j + 1][2]))
        rings2.append(_drop_repeats(ring, eps))
    # outer = largest; the rest are holes
    rings2.sort(key=lambda r: -pc.area(r))
    P['outer'] = pc.as_ccw(rings2[0])
    P['holes'] = [pc.as_cw(r) for r in rings2[1:]]
    P['error'] = None
    if corner_bad:
        P['error'] = ("a corner too tight for this stock: make the model "
                      "bigger or the stock thinner")
    elif not ring_is_simple(P['outer'], eps):
        P['error'] = ("fingers cross at a corner too tight for this "
                      "stock: make the model bigger or the stock thinner")
    P.pop('_pieces', None)


def _pieces(P, t):
    if '_pieces' not in P:
        P['_pieces'] = plate_pieces(P, t)
    return P['_pieces']


def _conflicts(P, Q, t, eps, vol_tol, window=None):
    """Overlap pieces between two plates: list of (volume, points)."""
    out = []
    QQ = [(qa, bb) for (qa, bb) in _pieces(Q, t)
          if window is None or _bbox_hit(bb, window, 0.0)]
    for (pa, ba) in _pieces(P, t):
        if window is not None and not _bbox_hit(ba, window, 0.0):
            continue
        for (qa, bb) in QQ:
            if not _bbox_hit(ba, bb, eps):
                continue
            vol, pts = prism_overlap(pa, qa, eps)
            if vol > vol_tol:
                out.append((vol, pts))
    return out


def _deepen(P, pts3, grow):
    """Cut the plate back far enough, run by run, to clear the given
    points: each point is charged to the side whose cut line it sits
    nearest and to the run it falls in, and that run's depth is pushed
    just past it.  The outline stays a finger profile -- no general
    polygon booleans, so nothing to degenerate."""
    want = {}
    for q in pts3:
        x, y = _to2(P['frame'], q)
        best = None
        for ri, prof in enumerate(P['prof']):
            for si, sd in enumerate(prof):
                A2, e2, n2 = sd['A2'], sd['e2'], sd['n2']
                u = (x - A2[0]) * e2[0] + (y - A2[1]) * e2[1]
                d = (x - A2[0]) * n2[0] + (y - A2[1]) * n2[1]
                runs = sd['runs']
                uc = max(runs[0][0], min(runs[-1][1], u))
                k = 0
                while k < len(runs) - 1 and uc > runs[k][1]:
                    k += 1
                # how far inside this run's cut line the point sits,
                # plus how far beyond the side's ends it lies
                slack = d - runs[k][2]
                off = max(0.0, runs[0][0] - u, u - runs[-1][1])
                score = off * 10.0 + abs(slack)
                if best is None or score < best[0]:
                    best = (score, ri, si, k, d)
        _s, ri, si, k, d = best
        kk = (ri, si, k)
        want[kk] = max(want.get(kk, -1e30), d + grow)
    changed = False
    for (ri, si, k), d in want.items():
        sd = P['prof'][ri][si]
        run = sd['runs'][k]
        if d > run[2] + 1e-12:
            # an end finger that loses at its corner gives the corner up
            # outright -- cut back to the partner's root, as if it had
            # never owned that end -- rather than being shaved a sliver
            # at a time (which converges, but only geometrically)
            if k in (0, len(sd['runs']) - 1) and run[2] < sd['s'] - 1e-12:
                run[2] = max(d, sd['s'])
            else:
                # overshoot: an overlap that wraps round a corner only
                # shows the part inside the current outline, and cutting
                # exactly to it converges geometrically; twice the
                # intrusion clears it in a pass or two for a few tenths
                # of a millimetre
                run[2] = run[2] + 2.0 * (d - run[2])
            changed = True
    return changed


def _vertex_cleanup(plates, rank, at_vertex, joints, Vm, t, size, eps):
    """Where two plates meeting at a vertex still overlap, the lower-
    ranked one gives up the column of material above the overlap.

    Returns (runs cut deeper, overlap volume left afterwards)."""
    vol_tol = 1e-7 * t ** 3
    grow = 0.01                 # mm; far below any laser's kerf
    reach = {}
    for J in joints.values():
        r = max(abs(J['c']), J['s'], t,
                J['segs'][0][1] - J['segs'][0][0],
                J['segs'][-1][1] - J['segs'][-1][0])
        for v in (J['va'], J['vb']):
            reach[v] = max(reach.get(v, 0.0), r)
    removed = 0

    verts_of = {}
    for v, ps_ in at_vertex.items():
        for q in ps_:
            verts_of.setdefault(q, set()).add(v)

    tries = {}

    def sweep(fix, which):
        nonlocal removed
        total = 0.0
        touched = set()
        for v in sorted(which):
            ps = sorted(at_vertex[v])
            r = reach[v] + 2.0 * t
            p = Vm[v]
            win = (p[0] - r, p[1] - r, p[2] - r, p[0] + r, p[1] + r,
                   p[2] + r)
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    A, B = plates[ps[i]], plates[ps[j]]
                    if A['error'] or B['error']:
                        continue
                    hits = _conflicts(A, B, t, eps, vol_tol, win)
                    if not hits:
                        continue
                    total += sum(h[0] for h in hits)
                    if not fix:
                        continue
                    # a pair that keeps overlapping however it is cut is
                    # not a corner problem (faces closer together than
                    # the stock is thick, say): stop grinding at it and
                    # let the report say so
                    pair = (A['index'], B['index'])
                    tries[pair] = tries.get(pair, 0) + 1
                    if tries[pair] > 25:
                        continue
                    loser = A if rank[v][A['index']] < rank[v][B['index']]                         else B
                    winner = B if loser is A else A
                    pts = [q for h in hits for q in h[1]]
                    # a cut that would break the plate is undone, and the
                    # other plate gives way instead; if neither can, the
                    # overlap is left for the report
                    for who in (loser, winner):
                        saved = [[list(r) for r in sd['runs']]
                                 for ring in who['prof'] for sd in ring]
                        if not _deepen(who, pts, grow):
                            break
                        _trace(who, eps)
                        if not who['error']:
                            removed += 1
                            touched.add(who['index'])
                            break
                        k = 0
                        for ring in who['prof']:
                            for sd in ring:
                                sd['runs'] = saved[k]
                                k += 1
                        _trace(who, eps)
        return total, touched

    # the first pass looks everywhere; after that only round the plates
    # that were just cut, since nothing else has moved
    which = set(at_vertex)
    for _pass in range(40):
        _total, touched = sweep(True, which)
        if not touched:
            break
        which = set().union(*(verts_of.get(q, ()) for q in touched))
    residual, _ = sweep(False, set(at_vertex))
    return removed, residual


# ---------------------------------------------------------------- #
#  assembly order                                                   #
# ---------------------------------------------------------------- #

ORDER_SEARCH_LIMIT = 120


def assembly_order(result):
    """An order in which the plates can be pressed in one at a time,
    each along its own normal from outside -- or None if none was found
    (or the kit is too big to search).

    A convex shell with Flush joints needs no search: every plate clears
    its neighbours along its normal, whatever is already in place, so
    any order works (the self-test checks exactly that).  Otherwise the
    order is found by taking the model apart: repeatedly lift out a
    plate that clears every plate still present, then reverse.  The
    last plate in must go in from outside; the others may also come
    from inside, while the shell is still open."""
    plates = result['plates']
    joints = result['joints']
    n = len(plates)
    if result['settings']['fit'] == 'FLUSH' and \
            all(J['convex'] for J in joints.values()):
        nbrs = {P['index']: set() for P in plates}
        for J in joints.values():
            A, B = J['plates']
            nbrs[A].add(B)
            nbrs[B].add(A)
        order, seen, queue = [], set(), [0]
        while queue:
            p = queue.pop(0)
            if p in seen:
                continue
            seen.add(p)
            order.append(p)
            queue.extend(sorted(nbrs[p] - seen))
        return order
    if n > ORDER_SEARCH_LIMIT:
        return None
    t = result['settings']['thickness']
    reach = max([abs(J['c']) + J['s'] for J in joints.values()] + [t])
    lift = 2.0 * reach + 2.0 * t
    # overlap is additive plate by plate, so which plates block each
    # plate's way out -- through the outside, or through the inside
    # while the shell is open -- is worked out once, and the take-apart
    # below is pure bookkeeping
    out_by, in_by = {}, {}
    for p in range(n):
        out_by[p] = set(overlap_volume(result, lift=lift, depth=lift + t,
                                       only={p}, per_plate=True))
        in_by[p] = set(overlap_volume(result, lift=0.0, depth=t + lift,
                                      only={p}, per_plate=True))
    present = set(range(n))
    removed, forced = [], []
    while present:
        pick = None
        for p in sorted(present):
            if not (out_by[p] & present) or                     (removed and not (in_by[p] & present)):
                pick = p
                break
        if pick is None:
            # a closed loop of plates that lock one another (the inner
            # ring of a torus): no rigid press-in exists, so one plate
            # has to be sprung or glued in -- the one fewest block
            pick = min(sorted(present), key=lambda p: min(
                len(out_by[p] & present), len(in_by[p] & present)))
            forced.append(pick)
        present.discard(pick)
        removed.append(pick)
    result['forced'] = forced
    return list(reversed(removed))


# ---------------------------------------------------------------- #
#  engraving and the cut layout                                     #
# ---------------------------------------------------------------- #

def _engrave(plates, joints, Vm, S):
    """Record where each plate's label and edge numbers go, in the
    plate's own 2D frame (outer face up).  The strokes themselves are
    set at layout time, after any mirroring, so the text reads
    correctly on whichever face it is cut into."""
    for P in plates:
        me = P['index']
        marks = []
        for sides in P['rings_v']:
            for verts, nb in sides:
                key = (min(me, nb), max(me, nb),
                       min(verts[0], verts[-1]), max(verts[0], verts[-1]))
                J = joints[key]
                A2 = _to2(P['frame'], Vm[verts[0]])
                B2 = _to2(P['frame'], Vm[verts[-1]])
                L = sqrt((B2[0] - A2[0]) ** 2 + (B2[1] - A2[1]) ** 2)
                e2 = ((B2[0] - A2[0]) / L, (B2[1] - A2[1]) / L)
                n2 = (-e2[1], e2[0])
                marks.append({'text': str(J['number']),
                              'mid': (0.5 * (A2[0] + B2[0]),
                                      0.5 * (A2[1] + B2[1])),
                              'inward': n2, 'L': L,
                              'depth': max(J['s'], J['c'], 0.0)})
        P['marks'] = marks


def _text_strokes_at(text, height, centre, direction):
    """Single-stroke text centred on `centre`, baseline along
    `direction` (unit 2D)."""
    w = _glyphs.text_width(text, height)
    strokes = _glyphs.text_strokes(text, height, (-0.5 * w, -0.5 * height))
    dx, dy = direction
    return [[(centre[0] + x * dx - y * dy, centre[1] + x * dy + y * dx)
             for (x, y) in s] for s in strokes]


def _inside(strokes, outer, holes):
    for s in strokes:
        for p in s:
            if not pc.point_in_polygon(p, outer):
                return False
            if any(pc.point_in_polygon(p, h) for h in holes):
                return False
    return True


def plate_parts(result):
    """One slicing `Part` per plate, laid out for cutting: viewed from
    the face that gets engraved (the inside face for INSIDE labels, so
    the outline is mirrored), with its label and edge numbers set as
    single-stroke engraving."""
    S = result['settings']
    mirror = S['labels'] == 'INSIDE'
    t = S['thickness']

    def m(p):
        return (-p[0], p[1]) if mirror else p

    parts = []
    for P in result['plates']:
        outer = [m(p) for p in P['outer']]
        holes = [[m(p) for p in h] for h in P['holes']]
        part = Part(outer, holes, family='P', index=P['index'])
        part.label = P['label']
        strokes = []
        if S['labels'] != 'NONE':
            for mk in P.get('marks', ()):
                ni = m(mk['inward'])
                mid = m(mk['mid'])
                # baseline along the edge, the text's "up" pointing
                # into the plate, so it reads from inside the part
                d = (ni[1], -ni[0])
                h = min(0.25 * mk['L'] / max(1, len(mk['text'])),
                        max(1.5, 1.2 * t))
                placed = None
                while h >= 1.0:
                    off = mk['depth'] + 0.8 * t + 0.5 * h
                    c = (mid[0] + ni[0] * off, mid[1] + ni[1] * off)
                    trial = _text_strokes_at(mk['text'], h, c, d)
                    if _inside(trial, part.outer, part.holes):
                        placed = trial
                        break
                    h *= 0.8
                if placed:
                    strokes.extend(placed)
            cx, cy = pc.centroid(part.outer)
            x0, y0, x1, y1 = pc.bounds(part.outer)
            h = max(1.5, min(0.12 * min(x1 - x0, y1 - y0), 8.0))
            while h >= 1.0:
                trial = _text_strokes_at(part.label, h, (cx, cy), (1.0, 0.0))
                if _inside(trial, part.outer, part.holes):
                    strokes.extend(trial)
                    break
                h *= 0.8
        part.engrave = strokes
        if P['error']:
            part.fail('plate', P['error'])
        parts.append(part)
    return parts


def cut_layout(result, sheet_width=600.0, sheet_height=400.0, kerf=0.15,
               margin=5.0, name='Plates'):
    """Nest the plates onto sheets: (Drawing, report).  Kerf is applied
    to the placed copies only, by the slicer's own nesting."""
    if kerf > 0.0:
        for J in result['joints'].values():
            w = J['L'] / max(1, J['n'])
            if 0.5 * kerf >= 0.5 * min(w, max(J['step'], 1e-9)):
                raise ValueError(
                    f"a kerf of {kerf:g} mm is too wide for these fingers")
    parts = plate_parts(result)
    return _layout.nest(parts, sheet_width, sheet_height, margin=margin,
                        kerf=kerf, label_height=0.0, name=name)


# ---------------------------------------------------------------- #
#  preview geometry and the assembly guide (headless)               #
# ---------------------------------------------------------------- #

def preview_geometry(result, explode=0.0):
    """The assembled kit as mesh data in the MODEL's own units (so it
    sits exactly over the solid it came from): each plate a closed
    prism of its outline, pushed out along its normal by `explode`
    (0..1, as a fraction of the model's half-size).  Returns (verts,
    faces, plate_of_face)."""
    s = result['scale']
    t = result['settings']['thickness']
    push = explode * 0.5 * result['settings']['size']
    verts, faces, owner = [], [], []
    for P in result['plates']:
        o, u, w, n = P['frame']
        o = _add(o, _mul(n, push))
        index = {}

        def vid(p2, level):
            key = (round(p2[0], 9), round(p2[1], 9), level)
            k = index.get(key)
            if k is None:
                q = _add(o, _add(_mul(u, p2[0]), _mul(w, p2[1])))
                if level:
                    q = _sub(q, _mul(n, t))
                k = len(verts)
                verts.append((q[0] / s, q[1] / s, q[2] / s))
                index[key] = k
            return k

        rings = [P['outer']] + list(P['holes'])
        Q, tris = ear_clip(bridge_holes(P['outer'], P['holes'])
                           if P['holes'] else P['outer'])
        for (i, j, k) in tris:
            faces.append([vid(Q[i], 0), vid(Q[j], 0), vid(Q[k], 0)])
            faces.append([vid(Q[k], 1), vid(Q[j], 1), vid(Q[i], 1)])
            owner.extend((P['index'], P['index']))
        for ring in rings:
            m = len(ring)
            for a in range(m):
                p, q = ring[a], ring[(a + 1) % m]
                faces.append([vid(p, 0), vid(p, 1), vid(q, 1), vid(q, 0)])
                owner.append(P['index'])
    return verts, faces, owner


def assembly_guide(result, layout_report=None):
    """Plain-text build notes, stored with the kit."""
    R = result['report']
    S = result['settings']
    t = S['thickness']
    lines = [f"{R['plates']} plates, {R['edges']} jointed edges, "
             f"{t:g} mm stock, model {S['size']:g} mm across."]
    if layout_report:
        lines.append(f"Cut on {layout_report['sheets']} sheet(s).")
    if S['labels'] == 'INSIDE':
        lines.append("Cut with the engraved side up; the engraved face "
                     "goes on the inside.")
    elif S['labels'] == 'OUTSIDE':
        lines.append("Cut with the engraved side up; the engraved face "
                     "shows on the outside.")
    lines.append("Matching numbers mark the two edges that join.")
    lines.append(f"Dihedral angles {R['phi_min']:.1f} to "
                 f"{R['phi_max']:.1f} degrees; fingers "
                 f"{R['step_min']:.1f} to {R['step_max']:.1f} mm deep.")
    if R['lips']:
        lines.append(f"{R['lips']} edge(s) have lips standing up to "
                     f"{R['lip_height']:.1f} mm proud: sand them flush.")
    if R['shallow']:
        lines.append(f"{R['shallow']} edge(s) have shallow fingers; glue "
                     f"them.")
    order = result['order']
    if order is None:
        lines.append("Too many plates to work out an assembly order; "
                     "expect to glue the last few.")
    elif R['reflex'] or S['fit'] == 'FILLED':
        lines.append("Assembly order: " + " ".join(
            result['plates'][p]['label'] for p in order))
        forced = result['forced']
        if forced and len(forced) <= 10:
            lines.append("These close a ring of plates that lock one "
                         "another, so flex or glue them in: " + " ".join(
                             result['plates'][p]['label'] for p in forced))
        elif forced:
            lines.append(f"{len(forced)} plates close rings that lock one "
                         f"another (inside corners hook both ways): glue "
                         f"this model rather than press-fitting it.")
    else:
        lines.append("Any assembly order works; press each plate in "
                     "along its face.")
    lines.append("Measure your stock: the joints are cut for exactly the "
                 "thickness given.")
    return "\n".join(lines)


try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    from bpy.props import EnumProperty, FloatProperty

    # the custom property holding the build notes, beside the layout
    GUIDE_KEY = 'math_art_plate_guide'

    def plate_enum_item():
        """The ('PLATES', ...) entry for a generator's style enum."""
        return ('PLATES', "Finger-Jointed Plates",
                "Build the shell from flat plates of real material "
                "thickness, joined by finger joints cut for each edge's "
                "true angle, with a cut layout to export for a laser "
                "cutter")

    class PlateStyleProps:
        """Property mixin for operators offering the Finger-Jointed
        Plates style.  Inherit it alongside `bpy.types.Operator`, the
        way `net_style.NetStyleProps` is used; never register it
        itself."""
        plate_thickness: FloatProperty(
            name="Material Thickness", default=3.0, min=0.5, max=30.0,
            description="Thickness of the sheet stock, in millimetres. "
                        "Measure your stock: the joints are cut for "
                        "exactly this thickness")
        plate_size: FloatProperty(
            name="Model Size", default=150.0, min=20.0, max=3000.0,
            description="Longest dimension of the finished model, in "
                        "millimetres")
        finger_width: FloatProperty(
            name="Finger Width", default=8.0, min=2.0, max=100.0,
            description="Target width of each finger along an edge, in "
                        "millimetres; every edge rounds it to a whole "
                        "number of fingers")
        joint_fit: EnumProperty(
            name="Joint Edges",
            items=[('FLUSH', "Flush",
                    "Nothing stands proud of the surface; edges that are "
                    "not square show a small groove"),
                   ('FILLED', "Filled",
                    "The fingers fill each joint completely; edges that "
                    "are not square leave a small lip to sand flush")],
            default='FLUSH',
            description="How each joint handles the surface error of a "
                        "square-cut plate meeting another at an angle")
        joint_clearance: FloatProperty(
            name="Joint Clearance", default=0.05, min=0.0, max=1.0,
            description="Gap left between neighbouring fingers, in "
                        "millimetres. 0 is a press fit; more makes "
                        "assembly easier and suits glued joints")
        plate_labels: EnumProperty(
            name="Labels",
            items=[('INSIDE', "Inside",
                    "Engrave part and edge numbers on the inside face"),
                   ('OUTSIDE', "Outside",
                    "Engrave part and edge numbers on the outside face"),
                   ('NONE', "None", "No engraving")],
            default='INSIDE',
            description="Where the part numbers and matching edge "
                        "numbers are engraved")
        plate_explode: FloatProperty(
            name="Explode", default=0.0, min=0.0, max=1.0,
            subtype='FACTOR',
            description="Pull the plates apart along their faces to show "
                        "how they fit together")
        cut_sheet_width: FloatProperty(
            name="Sheet Width", default=600.0, min=10.0, max=5000.0,
            description="Width of the stock sheet, in millimetres")
        cut_sheet_height: FloatProperty(
            name="Sheet Height", default=400.0, min=10.0, max=5000.0,
            description="Height of the stock sheet, in millimetres")
        cut_kerf: FloatProperty(
            name="Kerf", default=0.15, min=0.0, max=1.0,
            description="Width of material the beam burns away, in "
                        "millimetres. The outlines are offset by half "
                        "of it so the plates come out to size")

    def draw_plate_props(lay, op):
        """The Finger-Jointed Plates block of an operator's `draw()`."""
        lay.prop(op, 'plate_thickness')
        lay.prop(op, 'plate_size')
        lay.prop(op, 'joint_fit')
        lay.prop(op, 'finger_width')
        lay.prop(op, 'joint_clearance')
        lay.prop(op, 'plate_labels')
        lay.prop(op, 'plate_explode')
        box = lay.box()
        box.label(text="Fabrication")
        box.prop(op, 'cut_sheet_width')
        box.prop(op, 'cut_sheet_height')
        box.prop(op, 'cut_kerf')
        box.operator("object.fabrication_slice_export",
                     text="Export SVG / DXF", icon='EXPORT')

    def emit_plates_from_operator(op, context, V, F, label,
                                  material_fn=None, hint=None):
        """The whole PLATES branch of an operator's `execute()`: guard,
        build, preview, cut layout, report.  Returns {'FINISHED'} or
        {'CANCELLED'}."""
        try:
            from .. import fabrication_slicer as _fs
        except ImportError:
            import fabrication_slicer as _fs
        try:
            res = build_plates(
                V, F, thickness=op.plate_thickness, size=op.plate_size,
                finger=op.finger_width, fit=op.joint_fit,
                clearance=op.joint_clearance, labels=op.plate_labels)
            drawing, lay_rep = cut_layout(
                res, op.cut_sheet_width, op.cut_sheet_height, op.cut_kerf,
                name=f"{label} plates")
        except ValueError as e:
            msg = f"{label} cannot be made from plates: {e}"
            if hint:
                msg += " -- " + hint
            op.report({'ERROR'}, msg)
            return {'CANCELLED'}

        verts, faces, owner = preview_geometry(res, op.plate_explode)
        me = bpy.data.meshes.new("Plates")
        me.from_pydata(verts, [], faces)
        me.validate(clean_customdata=True)
        if len(me.polygons) == len(faces):
            att = me.attributes.new("plate_id", 'INT', 'FACE')
            att.data.foreach_set('value', owner)
            if material_fn is not None:
                lut, slot = {}, []
                for pid in owner:
                    nn = res['plates'][pid]['ngon']
                    if nn not in lut:
                        lut[nn] = len(me.materials)
                        me.materials.append(material_fn(nn))
                    slot.append(lut[nn])
                me.polygons.foreach_set('material_index', slot)
        me.polygons.foreach_set('use_smooth', [False] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(f"{label} plates", me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        # kept on the OBJECT, which is where the exporter looks
        obj[_fs.DRAWING_KEY] = _fs.drawing_to_json(drawing)
        obj[GUIDE_KEY] = assembly_guide(res, lay_rep)

        R = res['report']
        op.report({'INFO'},
                  f"{label}: {R['plates']} plates on {lay_rep['sheets']} "
                  f"sheet(s), {R['edges']} jointed edges, fingers "
                  f"{R['step_min']:.1f}-{R['step_max']:.1f} mm deep")
        if R['errors'] or lay_rep['errors']:
            op.report({'WARNING'},
                      f"{R['errors']} plate(s) have corners too tight for "
                      f"this stock and are on the ERROR layer; make the "
                      f"model bigger or the stock thinner")
        if R['residual'] > 0.0:
            op.report({'WARNING'},
                      f"plates still overlap by {R['residual']:.2f} mm^3 "
                      f"at some corners")
        if lay_rep['oversize']:
            op.report({'WARNING'},
                      f"{lay_rep['oversize']} plate(s) do not fit the "
                      f"sheet")
        if R['shallow']:
            op.report({'WARNING'},
                      f"{R['shallow']} edge(s) have fingers shallower "
                      f"than the stock is thick; glue recommended")
        if R['lips']:
            op.report({'INFO'},
                      f"{R['lips']} edge(s) leave lips up to "
                      f"{R['lip_height']:.1f} mm proud to sand flush")
        if res['order'] is None:
            op.report({'WARNING'},
                      "too many plates to work out an assembly order")
        elif res['forced']:
            op.report({'INFO'},
                      f"{len(res['forced'])} plate(s) close a locked ring "
                      f"and must be flexed or glued in (see the build "
                      f"notes)")
        return {'FINISHED'}

    def register():
        pass

    def unregister():
        pass


# ---------------------------------------------------------------- #
#  independent checks (used by the self-test and the report)        #
# ---------------------------------------------------------------- #

def overlap_volume(result, lift=0.0, depth=None, only=None, present=None,
                   per_plate=False):
    """Total volume (mm^3) shared by any two plates, computed exactly
    from convex prism pieces over the WHOLE plates -- no vertex windows,
    so it is independent of the clean-up it checks.

    With `depth`, the plate(s) in `only` are swept instead: their prisms
    start `lift` above the outer face and reach `depth` down, which is
    the region a plate passes through when it is pressed in along its
    normal from outside."""
    t = result['settings']['thickness']
    eps = 1e-9 * result['settings']['size']
    tol = 1e-7 * t ** 3
    plates = result['plates']
    pieces = []
    for P in plates:
        if only is not None and P['index'] in only and depth is not None:
            o, u, w, n = P['frame']
            fr = (_add(o, _mul(n, lift)), u, w, n)
            pieces.append(plate_pieces(dict(P, frame=fr), depth))
        else:
            pieces.append(_pieces(P, t))
    boxes = []
    for pl in pieces:
        pts = [(b[0], b[1], b[2]) for (_pr, b) in pl] + \
              [(b[3], b[4], b[5]) for (_pr, b) in pl]
        boxes.append(_bbox3(pts))
    total = 0.0
    by = {}
    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            if only is not None and i not in only and j not in only:
                continue
            if present is not None and (i not in present
                                        or j not in present):
                continue
            if not _bbox_hit(boxes[i], boxes[j], eps):
                continue
            for (pa, ba) in pieces[i]:
                if not _bbox_hit(ba, boxes[j], eps):
                    continue
                for (qa, bb) in pieces[j]:
                    if not _bbox_hit(ba, bb, eps):
                        continue
                    vol, _pts = prism_overlap(pa, qa, eps)
                    if vol > tol:
                        total += vol
                        other = j if i in (only or ()) else i
                        by[other] = by.get(other, 0.0) + vol
    return by if per_plate else total


def insertion_blocked(result):
    """Plates that cannot be pressed in along their own normal as the
    LAST plate, with every other plate already in place."""
    t = result['settings']['thickness']
    bad = []
    for P in result['plates']:
        if overlap_volume(result, lift=2.0 * t, depth=3.0 * t,
                          only={P['index']}) > 0.0:
            bad.append(P['index'])
    return bad


# ---------------------------------------------------------------- #

def _selftest():
    try:
        from .. import regular_solids_generator as rs
    except ImportError:
        import regular_solids_generator as rs
    fails = []

    def bad(msg):
        print("   BAD:", msg)
        fails.append(msg)

    t = 3.0

    # --- 1. the cross-section against its closed forms --------------
    for deg in (20, 45, 60, 70.53, 90, 109.47, 116.57, 120, 138.19, 150,
                170, 200, 240, 270, 300):
        phi = math.radians(deg)
        sp, cp = sin(phi), cos(phi)
        for fit in ('FLUSH', 'FILLED'):
            c, s = edge_joint(phi, t, fit)
            if phi < pi / 2:
                want_s = t / math.tan(phi / 2)
                want_step = want_s if fit == 'FILLED' else t / sp
            elif phi < pi:
                want_s = t * sp if fit == 'FLUSH' else t / sp
                want_step = t * sp if fit == 'FLUSH' else \
                    t * math.tan(phi / 2)
            else:
                want_s = None
                want_step = t * (1 + abs(cp)) / abs(sp)
                if fit == 'FLUSH':
                    # Flush caps a reflex finger; the rest of J is a
                    # hidden void
                    want_step = min(want_step, REFLEX_STEP * t)
            if want_s is not None and abs(s - want_s) > 1e-9:
                bad(f"root at {deg} {fit}: {s} != {want_s}")
            if abs((s - c) - want_step) > 1e-7:
                bad(f"step at {deg} {fit}: {s - c} != {want_step}")
            # the tab and the partner's material must not overlap
            a, nA, b, nB = section_frames(phi)
            big = abs(c) + 1000.0 * t
            X = [(c, 0.0), (big, 0.0), (big, t), (c, t)]
            for f in (lambda p: p[0] * nB[0] + p[1] * nB[1],
                      lambda p: t - (p[0] * nB[0] + p[1] * nB[1]),
                      lambda p: p[0] * b[0] + p[1] * b[1] - s):
                X = _clip_halfplane(X, f)
            if len(X) > 2 and pc.area(X) > 1e-9:
                bad(f"tab overlaps partner at {deg} {fit}")
            if fit == 'FILLED' and c > min(joint_corners(phi, t)) + 1e-12:
                bad(f"Filled tab leaves J uncovered at {deg}")

    # --- test solids --------------------------------------------------
    def solid(fam, sid):
        return rs.build_solid(fam, sid)[:2]

    def l_prism():
        L = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        V = [(x, y, 0.0) for x, y in L] + [(x, y, 1.0) for x, y in L]
        F = [list(reversed(range(6))), list(range(6, 12))]
        for k in range(6):
            j = (k + 1) % 6
            F.append([k, j, j + 6, k + 6])
        return V, F

    def picture_frame():
        # a square ring: the top and bottom are each four coplanar
        # trapezoids that must merge into one plate with a hole
        o = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
        i = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        V = [(x, y, 0.0) for x, y in o + i] + \
            [(x, y, 1.0) for x, y in o + i]
        F = []
        for k in range(4):
            j = (k + 1) % 4
            F.append([k + 8, j + 8, j + 12, k + 12])      # top
            F.append([k, k + 4, j + 4, j])                # bottom
            F.append([k, j, j + 8, k + 8])                # outer wall
            F.append([k + 4, k + 12, j + 12, j + 4])      # inner wall
        return V, F

    def split_cube():
        V, F = solid('PLATONIC', 'CUBE')
        F2 = []
        for f in F:
            F2.append([f[0], f[1], f[2]])
            F2.append([f[0], f[2], f[3]])
        return V, F2

    # --- 2. the cube is the ordinary box joint ---------------------
    V, F = solid('PLATONIC', 'CUBE')
    r1 = build_plates(V, F, fit='FLUSH')
    r2 = build_plates(V, F, fit='FILLED')
    if r1['report']['plates'] != 6 or r1['report']['edges'] != 12:
        bad(f"cube: {r1['report']['plates']} plates, "
            f"{r1['report']['edges']} edges")
    for J in r1['joints'].values():
        if abs(J['c']) > 1e-9 or abs(J['s'] - t) > 1e-9:
            bad(f"cube joint not a box joint: c={J['c']} s={J['s']}")
    for P, Q in zip(r1['plates'], r2['plates']):
        if len(P['outer']) != len(Q['outer']) or any(
                abs(p[0] - q[0]) > 1e-9 or abs(p[1] - q[1]) > 1e-9
                for p, q in zip(P['outer'], Q['outer'])):
            bad("cube: Flush and Filled differ")
            break
    # edge numbers: every number on exactly the two plates of its edge
    seen = {}
    for P in r1['plates']:
        for mk in P['marks']:
            seen.setdefault(mk['text'], []).append(P['index'])
    for J in r1['joints'].values():
        if sorted(seen.get(str(J['number']), [])) != sorted(J['plates']):
            bad(f"edge number {J['number']} not on its two plates")
    if len(seen) != len(r1['joints']):
        bad("edge numbers do not pair up")

    # --- 3/4. exact non-interpenetration, and insertion ---------------
    cases = [('cube', solid('PLATONIC', 'CUBE'), True),
             ('tetra', solid('PLATONIC', 'TETRA'), True),
             ('octa', solid('PLATONIC', 'OCTA'), True),
             ('icosa', solid('PLATONIC', 'ICOSA'), True),
             ('dodeca', solid('PLATONIC', 'DODECA'), True),
             ('cubocta', solid('ARCHIMEDEAN', 'CO'), True),
             ('L-prism', l_prism(), False)]
    for name, (V, F), convex in cases:
        for fit in ('FLUSH', 'FILLED'):
            try:
                r = build_plates(V, F, fit=fit)
            except ValueError as e:
                bad(f"{name} {fit}: {e}")
                continue
            errs = [P['error'] for P in r['plates'] if P['error']]
            if errs:
                bad(f"{name} {fit}: plate errors {errs[:2]}")
            ov = overlap_volume(r)
            if ov > 0.0:
                bad(f"{name} {fit}: plates overlap by {ov:.4f} mm^3")
            if r['report']['residual'] > 0.0:
                bad(f"{name} {fit}: clean-up left "
                    f"{r['report']['residual']:.4f} mm^3")
            if convex and fit == 'FLUSH':
                # the claim that lets the builder skip the search
                blocked = insertion_blocked(r)
                if blocked:
                    bad(f"{name} {fit}: plates {blocked} cannot go in "
                        f"last along their normal")
            if r['order'] is None or \
                    sorted(r['order']) != list(range(len(r['plates']))):
                bad(f"{name} {fit}: no assembly order")
    # near-flat degree-5 vertices, where plates that share no edge
    # overlap widely: the case that needed the clean-up to iterate
    r = build_plates(*solid('ARCHIMEDEAN', 'SD'))
    ov = overlap_volume(r)
    if ov > 0.0 or r['report']['errors']:
        bad(f"snub dodecahedron: overlap {ov:.4f} mm^3, "
            f"{r['report']['errors']} plate error(s)")
    # the L-prism has one reflex edge and a non-convex face
    r = build_plates(*l_prism())
    if r['report']['reflex'] != 1:
        bad(f"L-prism: {r['report']['reflex']} reflex edges, want 1")

    # the prism pieces tile each plate exactly
    for name, (V, F), _cv in cases:
        r = build_plates(V, F)
        for P in r['plates']:
            Q, tris = ear_clip(P['outer'])
            tot = sum(abs(_cross2((Q[j][0] - Q[i][0], Q[j][1] - Q[i][1]),
                                  (Q[k][0] - Q[i][0], Q[k][1] - Q[i][1])))
                      for (i, j, k) in tris) / 2.0
            if abs(tot - pc.area(P['outer'])) > 1e-6 * pc.area(P['outer']):
                bad(f"{name}: triangulation covers {tot:.3f} of "
                    f"{pc.area(P['outer']):.3f} mm^2")
                break

    # a ring torus: reflex edges round the hole, and an inner ring of
    # plates that lock one another, so it can only close with some
    # plates flexed or glued in -- which the order must say
    try:
        from .. import toroidal_polyhedron_generator as tg
    except ImportError:
        import toroidal_polyhedron_generator as tg
    V, F = tg.build_polyhedral_torus(12, 4, 1.0, 0.4, 0.0)
    r = build_plates(V, F)
    R = r['report']
    if R['reflex'] != 24 or R['errors']:
        bad(f"torus: {R['reflex']} reflex edges, {R['errors']} errors")
    ov = overlap_volume(r)
    if ov > 0.0:
        bad(f"torus: plates overlap by {ov:.4f} mm^3")
    if r['order'] is None or             sorted(r['order']) != list(range(R['plates'])):
        bad("torus: no assembly order")
    elif not 0 < len(r['forced']) <= 12:
        bad(f"torus: {len(r['forced'])} plates forced, want the inner "
            f"ring at most")
    # Filled still refuses reflex edges this flat
    try:
        build_plates(V, F, fit='FILLED')
        bad("torus Filled (201-degree edges) should be refused")
    except ValueError:
        pass

    # --- 5. coplanar merge ---------------------------------------------
    r = build_plates(*split_cube())
    if r['report']['plates'] != 6:
        bad(f"split cube made {r['report']['plates']} plates, want 6")

    V, F = picture_frame()
    r = build_plates(V, F)
    holed = sum(1 for P in r['plates'] if P['holes'])
    if r['report']['plates'] != 10 or holed != 2:
        bad(f"picture frame: {r['report']['plates']} plates, {holed} with "
            f"a hole; want 10 and 2")
    ov = overlap_volume(r)
    if ov > 0.0:
        bad(f"picture frame: plates overlap by {ov:.4f} mm^3")
    for P in r['plates']:
        if P['holes']:
            Q, tris = ear_clip(bridge_holes(P['outer'], P['holes']))
            tot = sum(abs(_cross2((Q[j][0] - Q[i][0], Q[j][1] - Q[i][1]),
                                  (Q[k][0] - Q[i][0], Q[k][1] - Q[i][1])))
                      for (i, j, k) in tris) / 2.0
            want = pc.area(P['outer']) - sum(pc.area(h) for h in P['holes'])
            if abs(tot - want) > 1e-6 * want:
                bad(f"holed plate triangulates to {tot:.2f}, want {want:.2f}")

    # --- 6. guards ------------------------------------------------------
    try:
        build_plates(*solid('ARCHIMEDEAN', 'SD'), fit='FILLED')
        bad("snub dodecahedron Filled should be refused")
    except ValueError as e:
        if 'Flush' not in str(e):
            bad(f"refusal should suggest Flush: {e}")
    V, F = solid('PLATONIC', 'CUBE')
    V = [tuple(c * (1.05 if i == 0 else 1.0) for c in v)
         for i, v in enumerate(V)]
    try:
        build_plates(V, F)
        bad("a warped cube should be refused")
    except ValueError:
        pass
    try:
        build_plates(*solid('PLATONIC', 'ICOSA'), limit=10)
        bad("the plate limit should refuse 20 plates")
    except ValueError:
        pass
    r = build_plates(*solid('PLATONIC', 'ICOSA'))
    if r['report']['shallow'] != 30:
        bad(f"icosahedron Flush: {r['report']['shallow']} shallow edges, "
            f"want 30 (step 0.67 t)")

    # --- 7. the cut layout ---------------------------------------------
    r = build_plates(*solid('PLATONIC', 'DODECA'))
    d, rep = cut_layout(r, 600.0, 400.0, kerf=0.15)
    if rep['placed'] != 12 or rep['errors']:
        bad(f"layout: {rep}")
    layers = {e.layer for s in d.sheets for e in s.entities}
    if not {'CUT', 'ENGRAVE'} <= layers:
        bad(f"layout layers {layers}")
    for P, part in zip(r['plates'], plate_parts(r)):
        if abs(pc.area(part.outer) - pc.area(P['outer'])) > 1e-6:
            bad("mirroring changed a plate's area")
            break
    cut = [e.points for s in d.sheets for e in s.entities
           if e.layer == 'CUT']
    if any(not ring_is_simple(ring, 1e-9) for ring in cut):
        bad("a kerf-compensated outline crosses itself")
    if not sum(pc.area(g) for g in cut) > \
            sum(pc.area(P['outer']) for P in r['plates']):
        bad("kerf compensation should grow the outlines")

    # --- 8. the preview mesh ---------------------------------------------
    V, F = solid('PLATONIC', 'CUBE')
    r = build_plates(V, F)
    verts, faces, owner = preview_geometry(r)
    ext = max(max(v[k] for v in V) - min(v[k] for v in V) for k in range(3))
    got = max(max(v[k] for v in verts) - min(v[k] for v in verts)
              for k in range(3))
    if abs(got - ext) > 1e-6 * ext:
        bad(f"preview is {got} across, the solid {ext}")
    edges = {}
    for f in faces:
        for k in range(len(f)):
            e = (min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)]))
            edges[e] = edges.get(e, 0) + 1
    if any(c != 2 for c in edges.values()):
        bad("preview plates are not closed solids")
    if len(set(owner)) != 6:
        bad("preview faces not tagged with their plates")

    if fails:
        raise AssertionError(f"{len(fails)} failure(s): {fails[0]}")
    print("   plate_style: all checks passed")
