
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
# t cot(phi/2) on acute edges, t tan(phi/2) (Filled) or t sin(phi)
# (Flush) on obtuse ones, and t(1+|cos phi|)/|sin phi| on reflex ones.
#
# VERTICES.  Each edge END is owned by one of its plates: at every
# vertex a total order is put on the plates that meet there, and the
# higher plate owns the end.  A total order has no cycles, so the top
# plate's fingers run right into the corner and the vertex closes --
# the pinwheel assignment would leave a hole.  Equal owners at both
# ends make the finger count odd, different owners make it even.  What
# a square-cut corner cannot resolve analytically is resolved by one
# rule: wherever two plates' prisms still overlap near a vertex, the
# lower-ranked plate gives up the whole column of material above the
# overlap (a perpendicular cut removes columns), found by clipping
# convex prism pieces and subtracted as a 2D polygon.
#
# ASSEMBLY.  Finger side-walls in both plates are planes normal to the
# edge, so they mate at any angle.  A plate jointed on two or more
# non-parallel edges can then move only along its own normal, and on a
# convex shell every plate -- the last one included -- presses in
# along it.  A non-convex shell needs an order: a plate goes in from
# outside only if every neighbour already placed meets it across a
# convex edge.  One is searched for greedily and reported.
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


def _hull2(pts):
    """Convex hull, CCW, of 2D points (Andrew's monotone chain)."""
    P = sorted(set((round(p[0], 12), round(p[1], 12)) for p in pts))
    if len(P) < 3:
        return P
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and _cross2(
                (lo[-1][0] - lo[-2][0], lo[-1][1] - lo[-2][1]),
                (p[0] - lo[-2][0], p[1] - lo[-2][1])) <= 0.0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and _cross2(
                (hi[-1][0] - hi[-2][0], hi[-1][1] - hi[-2][1]),
                (p[0] - hi[-2][0], p[1] - hi[-2][1])) <= 0.0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


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
    a CCW copy of it; returns (ccw_ring, triangles)."""
    P = pc.as_ccw(list(ring))
    n = len(P)
    idx = list(range(n))
    tris = []

    def convex(i0, i1, i2):
        a, b, c = P[i0], P[i1], P[i2]
        return _cross2((b[0] - a[0], b[1] - a[1]),
                       (c[0] - a[0], c[1] - a[1])) > 1e-14

    def inside(p, a, b, c):
        d1 = _cross2((b[0] - a[0], b[1] - a[1]), (p[0] - a[0], p[1] - a[1]))
        d2 = _cross2((c[0] - b[0], c[1] - b[1]), (p[0] - b[0], p[1] - b[1]))
        d3 = _cross2((a[0] - c[0], a[1] - c[1]), (p[0] - c[0], p[1] - c[1]))
        return d1 >= 0.0 and d2 >= 0.0 and d3 >= 0.0

    guard = 0
    while len(idx) > 3 and guard < 10 * n * n + 100:
        guard += 1
        m = len(idx)
        cut = False
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            if not convex(i0, i1, i2):
                continue
            a, b, c = P[i0], P[i1], P[i2]
            if any(inside(P[j], a, b, c) for j in idx
                   if j not in (i0, i1, i2)
                   and P[j] != a and P[j] != b and P[j] != c):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            cut = True
            break
        if not cut:
            # numerically flat leftovers: drop the flattest vertex
            best = min(range(m), key=lambda k: abs(_cross2(
                (P[idx[k]][0] - P[idx[k - 1]][0],
                 P[idx[k]][1] - P[idx[k - 1]][1]),
                (P[idx[(k + 1) % m]][0] - P[idx[k - 1]][0],
                 P[idx[(k + 1) % m]][1] - P[idx[k - 1]][1]))))
            idx.pop(best)
    if len(idx) == 3 and convex(*idx):
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


def tab_end(phi, t, fit):
    """Signed set-back of the owner's tab end, along its own plate."""
    xs = joint_corners(phi, t)
    filled = min(xs)
    if fit == 'FLUSH' and phi < pi:
        return max(0.0, t * cos(phi) / sin(phi))
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


def edge_joint(phi, t, fit):
    """(c_tab, s) for an edge: tab end and partner root.  The joint is
    symmetric under swapping the plates, so one pair serves both."""
    c = tab_end(phi, t, fit)
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
    return faces, planes


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


def prism_overlap(A, B, eps):
    """Intersection of two convex prisms (from `_prism`): (volume,
    vertices)."""
    faces = A[0]
    for pl in B[1]:
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


def plate_pieces(plate, t, ring=None):
    """Convex prism pieces of a plate (or of one ring given in its
    frame), with their bounding boxes."""
    ring = plate['outer'] if ring is None else ring
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
                       'error': None, 'label': f"P{pi_ + 1}"})

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
    order = _assembly_order(plates, joints)

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
    return {'plates': plates, 'joints': joints, 'scale': scale,
            'order': order, 'report': report, 'settings': S}


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


def _trace(P, eps):
    """Walk the profile into outline rings; set P's outer, holes and
    error."""
    rings2 = []
    corner_bad = False
    for prof in P['prof']:
        ring = []
        m = len(prof)
        for i in range(m):
            sd, pv = prof[i], prof[i - 1]
            A2, e2, n2, runs = sd['A2'], sd['e2'], sd['n2'], sd['runs']
            pA2, pe2, pn2, pL, pruns = (pv['A2'], pv['e2'], pv['n2'],
                                        pv['L'], pv['runs'])
            # the corner at this side's start: the previous side's last
            # run line meets this side's first run line
            d_prev = pruns[-1][2]
            d_here = runs[0][2]
            p0 = (pA2[0] + pn2[0] * d_prev, pA2[1] + pn2[1] * d_prev)
            q0 = (A2[0] + n2[0] * d_here, A2[1] + n2[1] * d_here)
            den = _cross2(pe2, e2)
            if abs(den) < 1e-9:
                # straight-through corner: a jog, not an intersection
                ring.append((pA2[0] + pe2[0] * pL + pn2[0] * d_prev,
                             pA2[1] + pe2[1] * pL + pn2[1] * d_prev))
                ring.append(q0)
            else:
                dq = (q0[0] - p0[0], q0[1] - p0[1])
                lam = _cross2(dq, e2) / den
                corner = (p0[0] + pe2[0] * lam, p0[1] + pe2[1] * lam)
                # the corner must land inside both end runs, or the
                # outline folds back over a finger
                u_here = ((corner[0] - A2[0]) * e2[0]
                          + (corner[1] - A2[1]) * e2[1])
                if lam < pruns[-1][0] - 1e-9 or u_here > runs[0][1] + 1e-9:
                    corner_bad = True
                ring.append(corner)
            for j in range(len(runs) - 1):
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
        P['error'] = ("fingers too wide for this corner; reduce the finger "
                      "width")
    elif not ring_is_simple(P['outer'], eps):
        P['error'] = "outline crosses itself"
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
            if k in (0, len(sd['runs']) - 1) and run[2] < sd['s']:
                d = max(d, sd['s'])
            run[2] = d
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

    def sweep(fix):
        nonlocal removed
        total = 0.0
        touched = 0
        for v in sorted(at_vertex):
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
                    loser = A if rank[v][A['index']] < rank[v][B['index']] \
                        else B
                    pts = [q for h in hits for q in h[1]]
                    if _deepen(loser, pts, grow):
                        removed += 1
                        touched += 1
                        _trace(loser, eps)
        return total, touched

    for _pass in range(8):
        _total, touched = sweep(True)
        if not touched:
            break
    residual, _ = sweep(False)
    return removed, residual


# ---------------------------------------------------------------- #
#  assembly order                                                   #
# ---------------------------------------------------------------- #

def _assembly_order(plates, joints):
    """A plate can be pressed in from outside when every neighbour
    already in place meets it across a convex edge (or every one across
    a reflex edge).  Greedy: place the plate with the most placed
    neighbours that is still insertable.  None if it gets stuck."""
    nbrs = {P['index']: [] for P in plates}
    for J in joints.values():
        A, B = J['plates']
        nbrs[A].append((B, J['convex']))
        nbrs[B].append((A, J['convex']))
    if all(J['convex'] for J in joints.values()):
        # convex shell: any order works; build outward from plate 0
        order, seen = [], set()
        queue = [0]
        while queue:
            p = queue.pop(0)
            if p in seen:
                continue
            seen.add(p)
            order.append(p)
            queue.extend(q for q, _c in sorted(nbrs[p]) if q not in seen)
        return order
    placed, order = set(), []
    while len(order) < len(plates):
        best = None
        for p in range(len(plates)):
            if p in placed:
                continue
            kinds = {c for q, c in nbrs[p] if q in placed}
            if len(kinds) > 1:
                continue
            score = sum(1 for q, _c in nbrs[p] if q in placed)
            if best is None or score > best[0]:
                best = (score, p)
        if best is None:
            return None
        placed.add(best[1])
        order.append(best[1])
    return order


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
