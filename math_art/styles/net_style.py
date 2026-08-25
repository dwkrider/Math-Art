
# Shared "Papercraft Net" style: edge-unfolding a closed polyhedral
# surface into the flat plane, with fold lines, glue tabs and matching
# edge numbers -- a template you could cut out and fold back up.
#
# HOW IT WORKS.  Cutting a closed surface along a spanning tree of its
# EDGE graph leaves the complementary spanning tree of its FACE (dual)
# graph intact, and what remains is a topological disk that flattens.
# The unfolding is built directly from that dual tree: the root face is
# laid into the plane by its own frame, and every child is carried onto
# the plane by the unique orientation-preserving isometry that puts its
# copy of the shared edge onto the parent's already-placed copy.  Since
# the shared edge runs in opposite directions in the two faces' windings,
# that isometry drops the child on the far side of the hinge with no
# reflection logic, and every face lands printed-side-up.
#
# WHAT IS NOT PROMISED.  A net may overlap itself.  Whether every convex
# polyhedron has SOME non-overlapping edge unfolding is Duerer's
# conjecture -- open since 1525 -- and for non-convex surfaces the answer
# is outright no (Bern et al. exhibit ununfoldable polyhedra with convex
# faces).  So overlap is detected rather than assumed away: faces are
# tested pairwise, the worst offender's hinge is cut, and its subtree
# becomes a separate island packed elsewhere in the plane.  The result is
# always a valid buildable net; the honest question is how many pieces it
# took, which is reported.
#
# FOLDING.  The same dual tree drives a rigid refold.  Each hinge carries
# a signed angle recovered numerically -- not from a normal-sign
# convention -- as the rotation carrying the parent's full-fold placement
# to the child's, which provably fixes both hinge endpoints and so is a
# rotation about the hinge line.  Composing those rotations scaled by a
# parameter t gives the flat net at t = 0 and the original solid, in its
# original position, at t = 1, with every face rigid at every step in
# between.
#
# References:
# - Albrecht Duerer, "Underweysung der Messung mit dem Zirckel und
#   Richtscheyt" (Nuremberg, 1525) -- the first systematic use of
#   polyhedral nets, and the origin of Duerer's conjecture that every
#   convex polyhedron has a non-overlapping edge unfolding.
# - G. C. Shephard, "Convex polytopes with convex nets", Mathematical
#   Proceedings of the Cambridge Philosophical Society 78 (1975),
#   pp. 389-403.
# - M. Bern, E. D. Demaine, D. Eppstein, E. Kuo, A. Mantler and
#   J. Snoeyink, "Ununfoldable polyhedra with convex faces",
#   Computational Geometry: Theory and Applications 24 (2003),
#   pp. 51-62 (preliminary version CCCG 1999).
# - E. D. Demaine, D. Eppstein, J. Erickson, G. W. Hart and
#   J. O'Rourke, "Vertex-unfoldings of simplicial manifolds",
#   Proceedings of the 18th Annual ACM Symposium on Computational
#   Geometry (2002).
# - Z. Abel and E. D. Demaine, "Edge-unfolding orthogonal polyhedra is
#   strongly NP-complete", Proceedings of the 23rd Canadian Conference
#   on Computational Geometry (2011).
# - S. Takahashi, H.-Y. Wu, S. H. Saw, C.-C. Lin and H.-C. Yen,
#   "Optimized topological surgery for unfolding 3D meshes", Computer
#   Graphics Forum 30, no. 7 (Pacific Graphics 2011).
# - E. D. Demaine and J. O'Rourke, "Geometric Folding Algorithms:
#   Linkages, Origami, Polyhedra", Cambridge University Press (2007),
#   Part III -- unfolding, including the steepest-edge and random-tree
#   heuristics this module's search follows.

import math
import random
from math import sqrt, sin, cos, atan2, pi


# ---------------------------------------------------------------- #
#  small vector helpers                                            #
# ---------------------------------------------------------------- #

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _len(a):
    return sqrt(_dot(a, a))


def _unit(a):
    L = _len(a) or 1.0
    return (a[0] / L, a[1] / L, a[2] / L)


def _newell(P):
    """Unit normal of a (planar, non-degenerate) 3D polygon."""
    n = [0.0, 0.0, 0.0]
    m = len(P)
    for i in range(m):
        a, b = P[i], P[(i + 1) % m]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    return _unit(tuple(n))


def _bbox_diag(V):
    if not V:
        return 1.0
    lo = [min(v[k] for v in V) for k in range(3)]
    hi = [max(v[k] for v in V) for k in range(3)]
    return sqrt(sum((hi[k] - lo[k]) ** 2 for k in range(3))) or 1.0


# --- rigid transforms, stored as (R, d) with p -> R p + d ---------- #

_IDENT = (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
          (0.0, 0.0, 0.0))


def _apply(T, p):
    R, d = T
    return (R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + d[0],
            R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + d[1],
            R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + d[2])


def _compose(A, B):
    """A after B."""
    RA, dA = A
    RB, dB = B
    R = tuple(tuple(sum(RA[i][k] * RB[k][j] for k in range(3))
                    for j in range(3)) for i in range(3))
    return (R, _apply(A, dB))


def _invert(T):
    R, d = T
    Ri = tuple(tuple(R[j][i] for j in range(3)) for i in range(3))
    di = tuple(-sum(Ri[i][k] * d[k] for k in range(3)) for i in range(3))
    return (Ri, di)


def _frame_matrix(e1, e2, e3):
    """Rotation whose columns are the given orthonormal basis."""
    return ((e1[0], e2[0], e3[0]),
            (e1[1], e2[1], e3[1]),
            (e1[2], e2[2], e3[2]))


def _frame_map(so, s1, s2, s3, to, t1, t2, t3):
    """Rigid map carrying the orthonormal frame (so; s1,s2,s3) onto
    (to; t1,t2,t3)."""
    S = _frame_matrix(s1, s2, s3)
    Tm = _frame_matrix(t1, t2, t3)
    # R = Tm . S^T
    R = tuple(tuple(sum(Tm[i][k] * S[j][k] for k in range(3))
                    for j in range(3)) for i in range(3))
    Rso = (sum(R[i][k] * so[k] for k in range(3)) for i in range(3))
    d = tuple(t - r for t, r in zip(to, Rso))
    return (R, d)


def _rot_axis(point, axis, ang):
    """Rotation by `ang` about the line through `point` along `axis`."""
    x, y, z = _unit(axis)
    c, s = cos(ang), sin(ang)
    C = 1.0 - c
    R = ((c + x * x * C, x * y * C - z * s, x * z * C + y * s),
         (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
         (z * x * C - y * s, z * y * C + x * s, c + z * z * C))
    Rp = tuple(sum(R[i][k] * point[k] for k in range(3))
               for i in range(3))
    return (R, tuple(point[i] - Rp[i] for i in range(3)))


def _rot_to_axis_angle(R):
    """Axis and angle of a rotation matrix."""
    tr = R[0][0] + R[1][1] + R[2][2]
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-12:
        return ((0.0, 0.0, 1.0), 0.0)
    if pi - ang < 1e-6:                     # 180 deg: axis from R + I
        best, ax = -1.0, (0.0, 0.0, 1.0)
        for k in range(3):
            col = (R[0][k] + (1.0 if k == 0 else 0.0),
                   R[1][k] + (1.0 if k == 1 else 0.0),
                   R[2][k] + (1.0 if k == 2 else 0.0))
            if _len(col) > best:
                best, ax = _len(col), col
        return (_unit(ax), pi)
    s = 2.0 * sin(ang)
    ax = ((R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s,
          (R[1][0] - R[0][1]) / s)
    return (_unit(ax), ang)


def _interp_rigid(T, t):
    """Rigid motion `t` of the way from the identity to T (screw
    interpolation of the rotation, straight line for the translation)."""
    R, d = T
    ax, ang = _rot_to_axis_angle(R)
    Rt, _ = _rot_axis((0.0, 0.0, 0.0), ax, ang * t)
    return (Rt, (d[0] * t, d[1] * t, d[2] * t))


# ---------------------------------------------------------------- #
#  surface conditioning: weld, validate, orient                    #
# ---------------------------------------------------------------- #

def weld_vertices(V, F, tol):
    """Merge vertices that coincide within `tol`, remapping the faces
    and dropping the repeats a merge can create.

    A no-op for meshes that already share indices; the point is the
    generators that emit each face with its own copy of its corners.
    """
    cell = tol if tol > 0 else 1e-9
    buckets = {}
    remap = [0] * len(V)
    out = []
    for i, v in enumerate(V):
        key = (int(math.floor(v[0] / cell)), int(math.floor(v[1] / cell)),
               int(math.floor(v[2] / cell)))
        hit = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in buckets.get((key[0] + dx, key[1] + dy,
                                          key[2] + dz), ()):
                        if _len(_sub(v, out[j])) <= tol:
                            hit = j
                            break
                    if hit is not None:
                        break
                if hit is not None:
                    break
            if hit is not None:
                break
        if hit is None:
            hit = len(out)
            out.append(tuple(float(c) for c in v))
            buckets.setdefault(key, []).append(hit)
        remap[i] = hit
    NF = []
    for f in F:
        g = []
        for i in f:
            j = remap[i]
            if not g or (g[-1] != j and g[0] != j):
                g.append(j)
        if len(g) >= 3:
            NF.append(g)
    return out, NF


def edge_table(F):
    """{(a, b) sorted: [(face, slot), ...]} over every face corner."""
    tab = {}
    for fi, f in enumerate(F):
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            tab.setdefault((min(a, b), max(a, b)), []).append((fi, k))
    return tab


def check_surface(V, F):
    """None if F is a closed, connected, edge-manifold surface;
    otherwise a sentence saying what is wrong."""
    if len(F) < 4:
        return "too few faces to unfold"
    tab = edge_table(F)
    bad = [e for e, uses in tab.items() if len(uses) != 2]
    if bad:
        opens = sum(1 for e in bad if len(tab[e]) == 1)
        many = len(bad) - opens
        parts = []
        if opens:
            parts.append(f"{opens} edge(s) with only one face")
        if many:
            parts.append(f"{many} edge(s) shared by more than two faces")
        return ("this surface is not a closed two-faces-per-edge "
                "shell (" + ", ".join(parts) + ")")
    adj = {}
    for uses in tab.values():
        (a, _ka), (b, _kb) = uses
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen = {0}
    stack = [0]
    while stack:
        for nb in adj.get(stack.pop(), ()):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if len(seen) != len(F):
        return (f"this surface falls into {len(F) - len(seen) and 2} or "
                "more disconnected shells")
    return None


def orient_consistently(V, F):
    """Re-wind faces so every shared edge is traversed once in each
    direction, then flip the whole surface if the majority of its faces
    end up facing inward.

    Not all of the callers guarantee a consistent winding, and for a
    self-intersecting surface (the great dodecahedron, say) "outward" is
    not even defined -- but the unfolding only needs the windings to
    AGREE, and the majority test is what keeps an ordinary convex solid
    coming out printed-side-up.
    """
    F = [list(f) for f in F]
    tab = edge_table(F)
    adj = {}
    for e, uses in tab.items():
        if len(uses) != 2:
            continue
        (a, ka), (b, kb) = uses
        adj.setdefault(a, []).append((b, e, ka, kb))
        adj.setdefault(b, []).append((a, e, kb, ka))
    done = {0}
    stack = [0]
    while stack:
        fi = stack.pop()
        for (nb, e, kh, kn) in adj.get(fi, ()):
            if nb in done:
                continue
            f, g = F[fi], F[nb]
            # the shared edge must run one way in fi and the other in nb
            da = (f[kh], f[(kh + 1) % len(f)])
            db = (g[kn], g[(kn + 1) % len(g)])
            if da == db:
                F[nb] = list(reversed(g))
            done.add(nb)
            stack.append(nb)
        # a flip invalidates the cached slots, so rebuild lazily
        if len(done) < len(F) and not stack:
            tab = edge_table(F)
            adj = {}
            for e, uses in tab.items():
                if len(uses) != 2:
                    continue
                (a, ka), (b, kb) = uses
                adj.setdefault(a, []).append((b, e, ka, kb))
                adj.setdefault(b, []).append((a, e, kb, ka))
            stack = [next(iter(done))]
    cen = tuple(sum(v[k] for v in V) / len(V) for k in range(3))
    votes = 0
    for f in F:
        P = [V[i] for i in f]
        fc = tuple(sum(p[k] for p in P) / len(P) for k in range(3))
        votes += 1 if _dot(_newell(P), _sub(fc, cen)) >= 0 else -1
    if votes < 0:
        F = [list(reversed(f)) for f in F]
    return F


def face_frames(V, F):
    """Per-face orthonormal frame (origin, u, w, n) with n the winding
    normal and (u, w, n) right-handed, the face's own 2D coordinates in
    (u, w), and the worst out-of-plane deviation seen."""
    frames = []
    local = []
    worst = 0.0
    for f in F:
        P = [V[i] for i in f]
        o = P[0]
        n = _newell(P)
        u = _unit(_sub(P[1], P[0]))
        u = _unit(_sub(u, _mul(n, _dot(u, n))))
        w = _cross(n, u)
        frames.append((o, u, w, n))
        pts = []
        for p in P:
            d = _sub(p, o)
            pts.append((_dot(d, u), _dot(d, w)))
            worst = max(worst, abs(_dot(d, n)))
        local.append(pts)
    return frames, local, worst


# ---------------------------------------------------------------- #
#  spanning tree of the dual graph                                 #
# ---------------------------------------------------------------- #

def face_adjacency(F):
    """[[(neighbour, slot_here, slot_there), ...], ...]"""
    adj = [[] for _ in F]
    for uses in edge_table(F).values():
        if len(uses) != 2:
            continue
        (a, ka), (b, kb) = uses
        adj[a].append((b, ka, kb))
        adj[b].append((a, kb, ka))
    return adj


def spanning_tree(F, adj, root, rng=None):
    """Dual spanning tree.  `parent[fi]` is None for the root, else
    (parent face, slot in this face, slot in the parent).  Returns
    (parent, breadth-first order)."""
    parent = [None] * len(F)
    seen = [False] * len(F)
    seen[root] = True
    order = [root]
    frontier = [root]
    while frontier:
        if rng is None:
            fi = frontier.pop(0)
            nbs = adj[fi]
        else:
            fi = frontier.pop(rng.randrange(len(frontier)))
            nbs = list(adj[fi])
            rng.shuffle(nbs)
        for (nb, kh, kn) in nbs:
            if not seen[nb]:
                seen[nb] = True
                parent[nb] = (fi, kn, kh)
                order.append(nb)
                frontier.append(nb)
    return parent, order


# ---------------------------------------------------------------- #
#  planar layout                                                   #
# ---------------------------------------------------------------- #

def _place(local, hinge_local, hinge_target):
    """2D rigid motion carrying the directed segment `hinge_local` onto
    `hinge_target`, applied to every point of `local`."""
    (a0, a1) = hinge_local
    (b0, b1) = hinge_target
    da = (a1[0] - a0[0], a1[1] - a0[1])
    db = (b1[0] - b0[0], b1[1] - b0[1])
    la = sqrt(da[0] ** 2 + da[1] ** 2) or 1.0
    lb = sqrt(db[0] ** 2 + db[1] ** 2) or 1.0
    ca, sa = da[0] / la, da[1] / la
    cb, sb = db[0] / lb, db[1] / lb
    # rotation by (angle of db) - (angle of da)
    c = ca * cb + sa * sb
    s = ca * sb - sa * cb
    out = []
    for p in local:
        x, y = p[0] - a0[0], p[1] - a0[1]
        out.append((b0[0] + c * x - s * y, b0[1] + s * x + c * y))
    return out


def layout(F, local, parent, order):
    """Flat 2D position of every face corner, one list per face."""
    flat = [None] * len(F)
    flat[order[0]] = list(local[order[0]])
    for fi in order[1:]:
        pfi, kn, kh = parent[fi]
        m, mp = len(F[fi]), len(F[pfi])
        # the shared edge, as this face walks it
        a0 = local[fi][kn]
        a1 = local[fi][(kn + 1) % m]
        # ... and as the parent walks it, reversed to match direction
        b1 = flat[pfi][kh]
        b0 = flat[pfi][(kh + 1) % mp]
        flat[fi] = _place(local[fi], (a0, a1), (b0, b1))
    return flat


# ---------------------------------------------------------------- #
#  overlap detection (convex polygons, separating axis)             #
# ---------------------------------------------------------------- #

def poly_overlap(P, Q, eps):
    """Penetration depth of two convex 2D polygons, 0.0 if they are
    separated or merely touching (within `eps`)."""
    best = float('inf')
    for poly in (P, Q):
        m = len(poly)
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            ax, ay = -(b[1] - a[1]), b[0] - a[0]
            L = sqrt(ax * ax + ay * ay)
            if L < 1e-15:
                continue
            ax, ay = ax / L, ay / L
            p0 = min(ax * p[0] + ay * p[1] for p in P)
            p1 = max(ax * p[0] + ay * p[1] for p in P)
            q0 = min(ax * p[0] + ay * p[1] for p in Q)
            q1 = max(ax * p[0] + ay * p[1] for p in Q)
            gap = min(p1, q1) - max(p0, q0)
            if gap <= eps:
                return 0.0
            best = min(best, gap)
    return 0.0 if best == float('inf') else best


def _bounds(P):
    xs = [p[0] for p in P]
    ys = [p[1] for p in P]
    return (min(xs), min(ys), max(xs), max(ys))


def find_overlaps(flat, faces, eps):
    """Overlapping pairs among `faces`, worst first."""
    box = {fi: _bounds(flat[fi]) for fi in faces}
    order = sorted(faces, key=lambda fi: box[fi][0])
    hits = []
    for ii, fi in enumerate(order):
        bi = box[fi]
        for fj in order[ii + 1:]:
            bj = box[fj]
            if bj[0] > bi[2] - eps:
                break
            if bj[3] < bi[1] + eps or bj[1] > bi[3] - eps:
                continue
            d = poly_overlap(flat[fi], flat[fj], eps)
            if d > 0.0:
                hits.append((d, fi, fj))
    hits.sort(reverse=True)
    return hits


# ---------------------------------------------------------------- #
#  islands: cut hinges until nothing overlaps                      #
# ---------------------------------------------------------------- #

def split_islands(F, parent, order, flat, eps):
    """Partition the faces into islands that are each overlap-free.

    Cutting a hinge does not move anything: the subtree below it keeps
    the layout it already had, and only stops being tested against the
    faces it left behind (the islands are packed apart later).  So the
    loop is simply "cut the worst offender's hinge and re-test".
    """
    depth = [0] * len(F)
    kids = [[] for _ in F]
    for fi in order[1:]:
        pfi = parent[fi][0]
        depth[fi] = depth[pfi] + 1
        kids[pfi].append(fi)
    roots = [order[0]]
    island_of = [order[0]] * len(F)
    members = {order[0]: list(order)}
    cuts = set()
    pending = [order[0]]
    while pending:
        r = pending.pop()
        hits = find_overlaps(flat, members[r], eps)
        if not hits:
            continue
        _d, fi, fj = hits[0]
        # detach the deeper of the pair; never the island's own root
        cand = fi if depth[fi] >= depth[fj] else fj
        if cand == r:
            cand = fj if cand == fi else fi
        if cand == r:                       # both are the root: impossible
            continue
        sub = []
        stack = [cand]
        while stack:
            x = stack.pop()
            sub.append(x)
            stack.extend(k for k in kids[x] if island_of[k] == island_of[r])
        cuts.add(cand)
        subset = set(sub)
        members[r] = [x for x in members[r] if x not in subset]
        members[cand] = sub
        for x in sub:
            island_of[x] = cand
        roots.append(cand)
        pending.extend((r, cand))
    return [(r, members[r]) for r in roots], cuts, island_of


# ---------------------------------------------------------------- #
#  packing                                                         #
# ---------------------------------------------------------------- #

def pack_islands(islands, extents, gap):
    """Shelf-pack island bounding boxes, widest first, into a roughly
    square block centred on the origin.  Returns a 2D offset per
    island root."""
    if len(islands) == 1:
        r, _m = islands[0]
        x0, y0, x1, y1 = extents[r]
        return {r: (-(x0 + x1) / 2.0, -(y0 + y1) / 2.0)}
    boxes = [(r, extents[r][2] - extents[r][0],
              extents[r][3] - extents[r][1]) for r, _m in islands]
    area = sum((w + gap) * (h + gap) for _r, w, h in boxes)
    row_w = max(sqrt(area) * 1.1, max(w for _r, w, _h in boxes) + gap)
    boxes.sort(key=lambda b: -b[2])
    off = {}
    cx, cy, rowh, total_w = 0.0, 0.0, 0.0, 0.0
    for (r, w, h) in boxes:
        if cx > 0.0 and cx + w > row_w:
            cy -= rowh + gap
            total_w = max(total_w, cx - gap)
            cx, rowh = 0.0, 0.0
        off[r] = (cx - extents[r][0], cy - extents[r][3])
        cx += w + gap
        rowh = max(rowh, h)
    total_w = max(total_w, cx - gap)
    total_h = -(cy - rowh)
    return {r: (o[0] - total_w / 2.0, o[1] + total_h / 2.0)
            for r, o in off.items()}


# ---------------------------------------------------------------- #
#  fold angles and fold transforms                                 #
# ---------------------------------------------------------------- #

def full_fold_maps(V, F, flat):
    """`A[fi]`: the rigid map carrying face fi's flat placement back to
    its position in the original solid.

    Built from orthonormal frames on both sides rather than by solving
    for a transform, so it is exactly rigid by construction.
    """
    A = []
    for fi, f in enumerate(F):
        P = flat[fi]
        so = (P[0][0], P[0][1], 0.0)
        s1 = _unit((P[1][0] - P[0][0], P[1][1] - P[0][1], 0.0))
        s3 = (0.0, 0.0, 1.0)
        s2 = _cross(s3, s1)
        Q = [V[i] for i in f]
        to = Q[0]
        t3 = _newell(Q)
        t1 = _unit(_sub(Q[1], Q[0]))
        t1 = _unit(_sub(t1, _mul(t3, _dot(t1, t3))))
        t2 = _cross(t3, t1)
        A.append(_frame_map(so, s1, s2, s3, to, t1, t2, t3))
    return A


def fold_angles(F, parent, order, flat, A, island_of):
    """Signed hinge angle and mountain/valley class for every face that
    has a parent inside its own island.

    The angle is recovered from M = A_parent^-1 . A_child.  Both maps
    agree on the two hinge vertices -- they are the same two points of
    the solid -- so M fixes them, which makes it a rotation about the
    flat hinge line and lets the angle be read off a third point.  No
    dihedral-sign convention is involved, so this stays correct for the
    non-convex and self-intersecting solids too.
    """
    theta = [0.0] * len(F)
    sign = [0] * len(F)
    hinge = [None] * len(F)
    for fi in order:
        if parent[fi] is None or island_of[fi] != island_of[parent[fi][0]]:
            continue
        pfi, kn, _kh = parent[fi]
        m = len(F[fi])
        p0 = (flat[fi][kn][0], flat[fi][kn][1], 0.0)
        p1 = (flat[fi][(kn + 1) % m][0], flat[fi][(kn + 1) % m][1], 0.0)
        ax = _unit(_sub(p1, p0))
        M = _compose(_invert(A[pfi]), A[fi])
        # a corner of this face that is off the hinge
        q = None
        for k in range(m):
            if k in (kn, (kn + 1) % m):
                continue
            cand = (flat[fi][k][0], flat[fi][k][1], 0.0)
            r = _sub(cand, p0)
            perp = _sub(r, _mul(ax, _dot(r, ax)))
            if _len(perp) > 1e-9:
                q, v1 = cand, perp
                break
        if q is None:
            continue
        r2 = _sub(_apply(M, q), p0)
        v2 = _sub(r2, _mul(ax, _dot(r2, ax)))
        ang = atan2(_dot(_cross(v1, v2), ax), _dot(v1, v2))
        theta[fi] = ang
        hinge[fi] = (p0, p1)
        if abs(ang) < 1e-7:
            sign[fi] = 0                    # coplanar: a printed line only
        else:
            # mountain if the face folds AWAY from the printed side
            cen = [0.0, 0.0, 0.0]
            for k in range(m):
                cen[0] += flat[fi][k][0] / m
                cen[1] += flat[fi][k][1] / m
            moved = _apply(_rot_axis(p0, ax, ang), (cen[0], cen[1], 0.0))
            sign[fi] = 1 if moved[2] < 0.0 else -1
    return theta, sign, hinge


def fold_transforms(F, parent, order, island_of, theta, hinge, A, t):
    """Rigid placement of every face at fold parameter `t`.

    Island roots interpolate from the identity (flat, where they were
    packed) to their full-fold map, so a many-piece net does not merely
    fold -- its pieces travel back together into the assembled solid.
    """
    T = [None] * len(F)
    for fi in order:
        par = parent[fi]
        if par is None or island_of[fi] != island_of[par[0]]:
            T[fi] = _interp_rigid(A[fi], t)
        else:
            p0, p1 = hinge[fi]
            T[fi] = _compose(T[par[0]],
                             _rot_axis(p0, _sub(p1, p0), theta[fi] * t))
    return T


# ---------------------------------------------------------------- #
#  welding the net: faces share only their hinge vertices          #
# ---------------------------------------------------------------- #

def weld_net(F, parent, order, island_of):
    """Union-find over (face, corner), joined at the two ends of every
    hinge.  Everything else stays split -- which is exactly the cut
    pattern of a paper net."""
    idx = {}
    for fi, f in enumerate(F):
        for k in range(len(f)):
            idx[(fi, k)] = (fi, k)

    def find(x):
        while idx[x] != x:
            idx[x] = idx[idx[x]]
            x = idx[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            idx[ra] = rb

    for fi in order:
        par = parent[fi]
        if par is None or island_of[fi] != island_of[par[0]]:
            continue
        pfi, kn, kh = par
        m, mp = len(F[fi]), len(F[pfi])
        union((fi, kn), (pfi, (kh + 1) % mp))
        union((fi, (kn + 1) % m), (pfi, kh))
    label = {}
    net = {}
    for key in idx:
        r = find(key)
        if r not in label:
            label[r] = len(label)
        net[key] = label[r]
    return net, len(label)


# ---------------------------------------------------------------- #
#  glue tabs                                                       #
# ---------------------------------------------------------------- #

def _tab_quad(a, b, depth, shoulder):
    """Trapezoid standing on the directed edge a->b, on its right."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    L = sqrt(ex * ex + ey * ey) or 1.0
    ex, ey = ex / L, ey / L
    mx, my = ey, -ex                        # right of a->b
    s = min(shoulder, 0.4 * L)
    return [a, b,
            (b[0] + depth * mx - s * ex, b[1] + depth * my - s * ey),
            (a[0] + depth * mx + s * ex, a[1] + depth * my + s * ey)]


def make_tabs(F, flat, cut_halves, members_of, tab_frac, eps, med_edge):
    """One trapezoidal tab per cut pair, shrunk or moved to the mating
    edge where it would collide, dropped if neither side has room."""
    tabs = {}
    dropped = 0
    accepted = {}                           # island root -> [quad, ...]
    groups = {}
    for fi, r in enumerate(members_of):
        groups.setdefault(r, []).append(fi)
    # Try the largest tab EITHER side can host before settling for a
    # smaller one, laying the shoulders back as it goes.  Shortening
    # alone is nearly useless: a trapezoid scaled about its base keeps
    # the same angular spread at its base corners, and that spread is
    # what has to fit.  The room outside a cut edge is the angular
    # defect of the vertex it runs to -- about 11 degrees on a
    # sixty-vertex solid -- so the schedule has to reach splays that
    # shallow or those edges can never be tabbed at any depth.
    for pid, (h0, h1) in enumerate(cut_halves):
        placed = None
        sides = (h0, h1) if h0 < h1 else (h1, h0)
        for shrink, splay in ((1.0, 60.0), (0.85, 50.0), (0.7, 40.0),
                              (0.55, 30.0), (0.4, 22.0), (0.3, 15.0),
                              (0.25, 9.0)):
            tan = math.tan(math.radians(splay))
            for (fi, k) in sides:
                m = len(F[fi])
                a = flat[fi][k]
                b = flat[fi][(k + 1) % m]
                L = sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
                root = members_of[fi]
                # the third term keeps the trapezoid's top edge positive
                # once the shoulders are laid back a long way
                depth = shrink * min(tab_frac * L, 0.5 * med_edge,
                                     0.35 * L * tan)
                quad = _tab_quad(a, b, depth, depth / tan)
                clash = False
                for fj in groups.get(root, ()):
                    if fj != fi and poly_overlap(quad, flat[fj],
                                                 eps) > 0.0:
                        clash = True
                        break
                if not clash:
                    for other in accepted.get(root, ()):
                        if poly_overlap(quad, other, eps) > 0.0:
                            clash = True
                            break
                if not clash:
                    placed = ((fi, k), quad)
                    accepted.setdefault(root, []).append(quad)
                    break
            if placed:
                break
        if placed:
            tabs[pid] = placed
        else:
            dropped += 1
    return tabs, dropped


# ---------------------------------------------------------------- #
#  a stroke font for the edge-match numbers                        #
# ---------------------------------------------------------------- #

_SEG = {
    'a': ((0.0, 1.0), (1.0, 1.0)), 'b': ((1.0, 1.0), (1.0, 0.5)),
    'c': ((1.0, 0.5), (1.0, 0.0)), 'd': ((0.0, 0.0), (1.0, 0.0)),
    'e': ((0.0, 0.5), (0.0, 0.0)), 'f': ((0.0, 1.0), (0.0, 0.5)),
    'g': ((0.0, 0.5), (1.0, 0.5)),
}
_DIGIT = {'0': "abcdef", '1': "bc", '2': "abged", '3': "abgcd",
          '4': "fgbc", '5': "afgcd", '6': "afgedc", '7': "abc",
          '8': "abcdefg", '9': "abcdfg"}


def digit_strokes(number, height=1.0, weight=0.16):
    """Quads spelling `number` in a seven-segment hand, laid out with
    its baseline on y = 0 starting at x = 0.

    This is the fallback letterform, used outside Blender (so the
    self-test can run headless) and if the real font cannot be
    tessellated.  Inside Blender the numbers come from Blender's own
    font instead -- see `_font_glyphs` -- which is what a reader
    actually wants to read off a printed net.  Either way the glyphs are
    mesh, so they ride the face they are printed on as it folds.
    """
    w = 0.55 * height
    pitch = w + 0.30 * height
    t = weight * height * 0.5
    out = []
    for di, ch in enumerate(str(int(number))):
        ox = di * pitch
        for s in _DIGIT.get(ch, ""):
            (x0, y0), (x1, y1) = _SEG[s]
            ax, ay = ox + x0 * w, y0 * height
            bx, by = ox + x1 * w, y1 * height
            dx, dy = bx - ax, by - ay
            L = sqrt(dx * dx + dy * dy) or 1.0
            dx, dy = dx / L, dy / L
            nx, ny = -dy * t, dx * t
            out.append([(ax + nx, ay + ny), (bx + nx, by + ny),
                        (bx - nx, by - ny), (ax - nx, ay - ny)])
    return out


def _place_number(glyphs, a, b, inward, height, along=0.5, lift=0.0):
    """Glyph outlines centred `along` the directed edge a->b and
    standing `inward` from it.  `glyphs` are polygons with their
    baseline on y = 0, from whichever letterform source is in use."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    L = sqrt(ex * ex + ey * ey) or 1.0
    ex, ey = ex / L, ey / L
    nx, ny = (-ey, ex) if inward > 0 else (ey, -ex)
    quads = glyphs
    if not quads:
        return []
    xs = [p[0] for q in quads for p in q]
    span = max(xs) - min(xs)
    ox = -span / 2.0
    oy = 0.25 * height
    out = []
    for q in quads:
        poly = []
        for (px, py) in q:
            gx, gy = px + ox, py + oy
            poly.append((a[0] + ex * (L * along + gx) + nx * gy,
                         a[1] + ey * (L * along + gx) + ny * gy,
                         lift))
        out.append(poly)
    return out


# ---------------------------------------------------------------- #
#  the whole net                                                   #
# ---------------------------------------------------------------- #

def build_net(V, F, mode='BFS', seed=0, fold=0.0, tabs=True,
              tab_size=0.1, numbers=True, gap=0.1, tries=None,
              glyph_fn=None):
    """Unfold (V, F) and return the net as ready-to-build mesh data.

    `glyph_fn(number, height)` supplies the letterform for the edge
    numbers as polygons on a y = 0 baseline; it defaults to the built-in
    stroke font so this stays runnable with no Blender.

    Raises ValueError with a user-readable message if the surface is not
    a closed, connected, two-faces-per-edge shell.
    """
    if glyph_fn is None:
        glyph_fn = digit_strokes
    diag = _bbox_diag(V)
    tol = 1e-6 * diag
    V, F = weld_vertices(V, F, tol)
    bad = check_surface(V, F)
    if bad:
        raise ValueError(bad)
    F = orient_consistently(V, F)
    frames, local, planarity = face_frames(V, F)
    adj = face_adjacency(F)
    eps = 1e-7 * diag

    areas = []
    for fi, f in enumerate(F):
        P = local[fi]
        s = 0.0
        for k in range(len(P)):
            q = P[(k + 1) % len(P)]
            s += P[k][0] * q[1] - q[0] * P[k][1]
        areas.append(abs(s) / 2.0)
    root0 = max(range(len(F)), key=lambda i: areas[i])

    def attempt(rt, rng):
        parent, order = spanning_tree(F, adj, rt, rng)
        flat = layout(F, local, parent, order)
        return parent, order, flat

    best = None
    if mode == 'SEARCH':
        if tries is None:
            tries = max(4, min(40, int(4000 / max(1, len(F)))))
        rng = random.Random(seed)
        cands = [attempt(root0, None)]
        for _ in range(tries):
            cands.append(attempt(rng.randrange(len(F)), rng))
        # rank by the cheap proxy first, split only the best few
        ranked = sorted(
            cands,
            key=lambda c: len(find_overlaps(c[2], list(range(len(F))), eps)))
        for cand in ranked[:3]:
            parent, order, flat = cand
            isl, cuts, iof = split_islands(F, parent, order, flat, eps)
            score = (len(isl), _spread(flat))
            if best is None or score < best[0]:
                best = (score, parent, order, flat, isl, cuts, iof)
    if best is None:
        parent, order, flat = attempt(root0, None)
        isl, cuts, iof = split_islands(F, parent, order, flat, eps)
        best = ((len(isl), 0.0), parent, order, flat, isl, cuts, iof)
    _score, parent, order, flat, islands, _cuts, island_of = best

    # --- cut edges, paired ---
    tab_pairs = []
    tree_edges = set()
    for fi in order:
        par = parent[fi]
        if par is not None and island_of[fi] == island_of[par[0]]:
            pfi, kn, kh = par
            tree_edges.add((fi, kn))
            tree_edges.add((pfi, kh))
    for uses in edge_table(F).values():
        (a, ka), (b, kb) = uses
        if (a, ka) in tree_edges:
            continue
        tab_pairs.append(((a, ka), (b, kb)))

    # --- tabs and numbers live in flat space, so place them first ---
    med = sorted(
        sqrt((flat[fi][(k + 1) % len(F[fi])][0] - flat[fi][k][0]) ** 2
             + (flat[fi][(k + 1) % len(F[fi])][1] - flat[fi][k][1]) ** 2)
        for fi in range(len(F)) for k in range(len(F[fi])))
    med_edge = med[len(med) // 2] if med else 1.0
    tab_map, dropped = ({}, 0)
    if tabs:
        tab_map, dropped = make_tabs(F, flat, tab_pairs, island_of,
                                     tab_size, eps, med_edge)

    # --- pack the islands apart ---
    extents = {}
    for (r, mem) in islands:
        pts = [p for fi in mem for p in flat[fi]]
        for pid, ((fi, _k), quad) in tab_map.items():
            if island_of[fi] == r:
                pts.extend(quad)
        extents[r] = _bounds(pts)
    offs = pack_islands(islands, extents, gap)
    for (r, mem) in islands:
        ox, oy = offs[r]
        for fi in mem:
            flat[fi] = [(p[0] + ox, p[1] + oy) for p in flat[fi]]
    for pid in list(tab_map):
        (fi, k), quad = tab_map[pid]
        ox, oy = offs[island_of[fi]]
        tab_map[pid] = ((fi, k), [(p[0] + ox, p[1] + oy) for p in quad])

    # --- fold ---
    A = full_fold_maps(V, F, flat)
    theta, sign, hinge = fold_angles(F, parent, order, flat, A, island_of)
    T = fold_transforms(F, parent, order, island_of, theta, hinge, A,
                        max(0.0, min(1.0, fold)))

    # --- emit ---
    net_of, nverts = weld_net(F, parent, order, island_of)
    verts = [None] * nverts
    owner = [0] * nverts
    for (fi, k), vi in net_of.items():
        if verts[vi] is None:
            owner[vi] = fi
            p = flat[fi][k]
            verts[vi] = _apply(T[fi], (p[0], p[1], 0.0))
    faces, kinds, sides = [], [], []
    for fi, f in enumerate(F):
        faces.append([net_of[(fi, k)] for k in range(len(f))])
        kinds.append(0)
        sides.append(len(f))

    def emit_poly(fi, poly, kind):
        base = len(verts)
        for p in poly:
            z = p[2] if len(p) > 2 else 0.0
            verts.append(_apply(T[fi], (p[0], p[1], z)))
        faces.append(list(range(base, len(verts))))
        kinds.append(kind)
        sides.append(0)

    lift = 0.0015 * diag
    pair_id = {}
    for pid, (h0, h1) in enumerate(tab_pairs):
        pair_id[h0] = pid + 1
        pair_id[h1] = pid + 1
    for pid, ((fi, k), quad) in sorted(tab_map.items()):
        base = len(verts)
        m = len(F[fi])
        # the tab's base edge reuses the face's own two net vertices
        v0, v1 = net_of[(fi, k)], net_of[(fi, (k + 1) % m)]
        for p in quad[2:]:
            verts.append(_apply(T[fi], (p[0], p[1], 0.0)))
        faces.append([v0, v1] + list(range(base, len(verts))))
        kinds.append(1)
        sides.append(0)
    if numbers:
        for (h0, h1) in tab_pairs:
            pid = pair_id[h0]
            for (fi, k) in (h0, h1):
                m = len(F[fi])
                a, b = flat[fi][k], flat[fi][(k + 1) % m]
                L = sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
                h = min(0.30 * L, 0.45 * med_edge)
                on_tab = tab_map.get(pid - 1, ((None, None), None))[0]
                if on_tab == (fi, k):
                    quad = tab_map[pid - 1][1]
                    ctr = ((quad[2][0] + quad[3][0]) / 2.0,
                           (quad[2][1] + quad[3][1]) / 2.0)
                    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                    d = sqrt((ctr[0] - mid[0]) ** 2
                             + (ctr[1] - mid[1]) ** 2)
                    h = min(h, 0.5 * d)
                    polys = _place_number(glyph_fn(pid, h), a, b, -1, h,
                                          0.5, lift)
                else:
                    polys = _place_number(glyph_fn(pid, h), a, b, +1, h,
                                          0.5, lift)
                for poly in polys:
                    emit_poly(fi, poly, 2)

    # --- edge tags, by net vertex pair ---
    folds, cuts_out = [], []
    for fi in order:
        par = parent[fi]
        if par is None or island_of[fi] != island_of[par[0]]:
            continue
        _p, kn, _kh = par
        m = len(F[fi])
        # reported with the papercraft sign, not the raw rotation's:
        # positive means a mountain (the crease rises toward the printed
        # side), negative a valley.  The rotation that folds the model
        # keeps its own sign in `theta` and is unaffected.
        folds.append((net_of[(fi, kn)], net_of[(fi, (kn + 1) % m)],
                      sign[fi], sign[fi] * abs(theta[fi])))
    for (h0, h1) in tab_pairs:
        pid = pair_id[h0]
        for (fi, k) in (h0, h1):
            m = len(F[fi])
            cuts_out.append((net_of[(fi, k)], net_of[(fi, (k + 1) % m)],
                             pid))

    return {
        'verts': verts, 'faces': faces, 'kinds': kinds, 'sides': sides,
        'folds': folds, 'cuts': cuts_out,
        'info': {'islands': len(islands), 'faces': len(F),
                 'tabs': len(tab_map), 'dropped': dropped,
                 'pairs': len(tab_pairs), 'planarity': planarity,
                 'mountains': sum(1 for f in folds if f[2] > 0),
                 'valleys': sum(1 for f in folds if f[2] < 0),
                 'flat_folds': sum(1 for f in folds if f[2] == 0)},
    }


def _spread(flat):
    pts = [p for P in flat for p in P]
    x0, y0, x1, y1 = _bounds(pts)
    return (x1 - x0) * (y1 - y0)


try:
    import bpy
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _tag_material(name, rgb):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.6
        return mat

    _FONT_CACHE = {}

    def _font_glyphs(number, height):
        """`number` set in Blender's own font and tessellated to filled
        outlines, scaled to `height` with its baseline on y = 0.

        Real type rather than the seven-segment strokes: these are the
        numbers someone reads while gluing the model up, so they should
        look like numbers.  The glyphs are still mesh and still part of
        the net object, so they fold with the face they sit on -- which
        text objects, being separate objects with their own transforms,
        would not do.  Cached per number at unit height; the caller's
        height is a plain scale.
        """
        key = int(number)
        polys = _FONT_CACHE.get(key)
        if polys is None:
            polys = []
            cu = ob = None
            try:
                cu = bpy.data.curves.new(f"_net_num_{key}", type='FONT')
                cu.body = str(key)
                cu.align_x = 'LEFT'
                cu.align_y = 'BOTTOM_BASELINE'
                # a printed number is a few millimetres tall, so the
                # default curve resolution spends hundreds of triangles
                # per digit on smoothness nobody can see; 4 keeps the
                # letterforms and cuts the tessellation by ~4x
                cu.resolution_u = 4
                ob = bpy.data.objects.new(f"_net_num_{key}", cu)
                me = ob.to_mesh()
                if me is not None and len(me.polygons):
                    pts = [(v.co.x, v.co.y) for v in me.vertices]
                    raw = [[pts[i] for i in p.vertices]
                           for p in me.polygons]
                    ys = [p[1] for f in raw for p in f]
                    xs = [p[0] for f in raw for p in f]
                    # digits have no descender, so the glyph box IS the
                    # cap height: normalise it to 1 and rebase to (0, 0)
                    span = (max(ys) - min(ys)) or 1.0
                    x0, y0 = min(xs), min(ys)
                    polys = [[((x - x0) / span, (y - y0) / span)
                              for (x, y) in f] for f in raw]
                ob.to_mesh_clear()
            except (RuntimeError, AttributeError):
                polys = []                  # fall back to the strokes
            finally:
                # both datablocks, always: removing the object leaves
                # the font curve behind at zero users, and one per
                # number would quietly fill the file with orphan data
                if ob is not None:
                    bpy.data.objects.remove(ob)
                if cu is not None:
                    bpy.data.curves.remove(cu)
            _FONT_CACHE[key] = polys
        if not polys:
            return digit_strokes(number, height)
        return [[(x * height, y * height) for (x, y) in f]
                for f in polys]

    def emit_net(context, V, F, label, fold=0.0, mode='BFS', seed=0,
                 tabs=True, tab_size=0.1, numbers=True, gap=0.1,
                 material_fn=None):
        """Build the net object.  Raises ValueError (with a message fit
        for `report`) if the surface cannot be unfolded."""
        net = build_net(V, F, mode=mode, seed=seed, fold=fold, tabs=tabs,
                        tab_size=tab_size, numbers=numbers, gap=gap,
                        glyph_fn=_font_glyphs)
        me = bpy.data.meshes.new("Net")
        me.from_pydata(net['verts'], [], net['faces'])
        me.validate(clean_customdata=True)
        nface = len(me.polygons)
        if nface == len(net['faces']):
            att = me.attributes.new("net_part", 'INT', 'FACE')
            att.data.foreach_set('value', net['kinds'])
            att = me.attributes.new("ngon_sides", 'INT', 'FACE')
            att.data.foreach_set('value', net['sides'])
            lut = {}
            slot = []
            for kind, nn in zip(net['kinds'], net['sides']):
                if kind == 1:
                    key = 'tab'
                elif kind == 2:
                    key = 'ink'
                else:
                    key = nn
                if key not in lut:
                    lut[key] = len(me.materials)
                    if key == 'tab':
                        me.materials.append(
                            _tag_material("Net Tab", (0.82, 0.78, 0.70)))
                    elif key == 'ink':
                        me.materials.append(
                            _tag_material("Net Ink", (0.06, 0.06, 0.08)))
                    else:
                        me.materials.append(
                            _tag_material(f"Net {nn}-gon", (0.7, 0.7, 0.7))
                            if material_fn is None else material_fn(nn))
                slot.append(lut[key])
            me.polygons.foreach_set('material_index', slot)
        # per-edge tags, looked up by vertex pair (never by index order)
        pos = {}
        for e in me.edges:
            a, b = e.vertices
            pos[(min(a, b), max(a, b))] = e.index
        fs = [0] * len(me.edges)
        fa = [0.0] * len(me.edges)
        em = [0] * len(me.edges)
        sharp, crease = [], []
        for (a, b, s, ang) in net['folds']:
            ei = pos.get((min(a, b), max(a, b)))
            if ei is None:
                continue
            fs[ei] = s
            fa[ei] = ang
            if s != 0:
                sharp.append((a, b))
                if s < 0:
                    crease.append((a, b))
        for (a, b, pid) in net['cuts']:
            ei = pos.get((min(a, b), max(a, b)))
            if ei is not None:
                em[ei] = pid
        for name, kind, data in (("fold_sign", 'INT', fs),
                                 ("fold_angle", 'FLOAT', fa),
                                 ("edge_match", 'INT', em)):
            att = me.attributes.new(name, kind, 'EDGE')
            att.data.foreach_set('value', data)
        me.update()
        try:
            from .. import sharp_creases
        except ImportError:
            import sharp_creases
        sharp_creases.mark_sharp(me, sharp, crease=False)
        if crease:
            want = {(min(a, b), max(a, b)) for a, b in crease}
            lyr = me.edge_creases if hasattr(me, 'edge_creases') else None
            if lyr is None:
                lyr = me.attributes.new("crease_edge", 'FLOAT', 'EDGE')
            for e in me.edges:
                a, b = e.vertices
                if (min(a, b), max(a, b)) in want:
                    lyr.data[e.index].value = 1.0
        obj = bpy.data.objects.new(f"{label} net", me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj, net['info']

    def register():
        pass

    def unregister():
        pass


def _selftest():
    try:
        from .. import regular_solids_generator as rs
    except ImportError:
        import regular_solids_generator as rs
    fails = []

    def bad(msg):
        print("   BAD:", msg)
        fails.append(msg)

    # --- weld and the manifold gate ---
    cube = rs.build_solid('PLATONIC', 'CUBE')[:2]
    CV, CF = cube
    loose_v, loose_f = [], []
    for f in CF:
        b = len(loose_v)
        loose_v.extend(CV[i] for i in f)
        loose_f.append(list(range(b, len(loose_v))))
    wv, wf = weld_vertices(loose_v, loose_f, 1e-6)
    print(f"weld: 6 loose quads -> V={len(wv)} F={len(wf)} (expect 8, 6)")
    if len(wv) != 8 or len(wf) != 6:
        bad("weld did not rejoin the cube")
    if check_surface(wv, wf) is not None:
        bad("welded cube failed the surface check")

    # --- separating axis ---
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    if poly_overlap(sq, [(1, 0), (2, 0), (2, 1), (1, 1)], 1e-9) != 0.0:
        bad("edge-sharing squares reported as overlapping")
    if poly_overlap(sq, [(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)],
                    1e-9) <= 0.0:
        bad("overlapping squares reported as clear")

    # --- unfolding is an isometry, islands are clean, fold is exact ---
    CASES = [('PLATONIC', 'TETRA', 1e-9), ('PLATONIC', 'CUBE', 1e-9),
             ('PLATONIC', 'DODECA', 1e-9), ('PLATONIC', 'ICOSA', 1e-9),
             ('ARCHIMEDEAN', 'TI', 1e-5), ('CATALAN', 'DDT', 1e-5),
             ('JOHNSON', 'J92', 1e-5), ('KEPLER', 'GD', 1e-9)]
    for (fam, sid, tol) in CASES:
        V, F, _s = rs.build_solid(fam, sid)
        V = rs.fit_cube(V)
        try:
            net = build_net(V, F, tabs=True, numbers=False)
        except ValueError as e:
            bad(f"{fam}/{sid}: {e}")
            continue
        info = net['info']
        # every island overlap-free: re-derive from the emitted faces
        W, FF = weld_vertices(V, F, 1e-6 * _bbox_diag(V))
        FF = orient_consistently(W, FF)
        frames, local, _p = face_frames(W, FF)
        adj = face_adjacency(FF)
        par, order = spanning_tree(FF, adj, 0, None)
        flat = layout(FF, local, par, order)
        eps = 1e-7 * _bbox_diag(W)
        isl, _c, iof = split_islands(FF, par, order, flat, eps)
        clean = all(not find_overlaps(flat, mem, eps) for _r, mem in isl)
        # areas preserved by the unfolding
        worst_a = 0.0
        for fi, f in enumerate(FF):
            P3 = [W[i] for i in f]
            n = _newell(P3)
            a3 = abs(sum(_dot(n, _cross(P3[k], P3[(k + 1) % len(P3)]))
                         for k in range(len(P3)))) / 2.0
            P2 = flat[fi]
            a2 = abs(sum(P2[k][0] * P2[(k + 1) % len(P2)][1]
                         - P2[(k + 1) % len(P2)][0] * P2[k][1]
                         for k in range(len(P2)))) / 2.0
            worst_a = max(worst_a, abs(a3 - a2))
        # refold reproduces the solid exactly
        folded = build_net(V, F, fold=1.0, tabs=False, numbers=False)
        pts = folded['verts']
        nf = len(FF)
        worst_f = 0.0
        for fi in range(nf):
            for k, vi in enumerate(folded['faces'][fi]):
                worst_f = max(worst_f,
                              _len(_sub(pts[vi], W[FF[fi][k]])))
        flat_net = build_net(V, F, fold=0.0, tabs=False, numbers=False)
        worst_z = max(abs(p[2]) for p in flat_net['verts'])
        ok = (clean and worst_a < tol * 100 and worst_f < tol * 100
              and worst_z < 1e-9)
        print(f"{fam}/{sid:6s} F={info['faces']:4d} islands="
              f"{info['islands']:3d} area dev={worst_a:.1e} "
              f"refold dev={worst_f:.1e} flat z={worst_z:.1e} "
              f"{'OK' if ok else 'BAD'}")
        if not ok:
            bad(f"{fam}/{sid}: unfold/refold check")

    # --- the fold is rigid at every t, not just the ends ---
    V, F, _s = rs.build_solid('PLATONIC', 'DODECA')
    V = rs.fit_cube(V)
    W, FF = weld_vertices(V, F, 1e-9)
    half = build_net(V, F, fold=0.5, tabs=False, numbers=False)
    worst = 0.0
    for fi, f in enumerate(FF):
        face = half['faces'][fi]
        for k in range(len(f)):
            d3 = _len(_sub(W[f[k]], W[f[(k + 1) % len(f)]]))
            dn = _len(_sub(half['verts'][face[k]],
                           half['verts'][face[(k + 1) % len(f)]]))
            worst = max(worst, abs(d3 - dn))
    print(f"rigidity at t=0.5: worst edge drift={worst:.1e} "
          f"{'OK' if worst < 1e-9 else 'BAD'}")
    if worst >= 1e-9:
        bad("faces distort mid-fold")

    # --- a convex solid folds all-mountain; a stellation has flat folds ---
    net = build_net(V, F, tabs=False, numbers=False)
    i = net['info']
    print(f"dodecahedron folds: M={i['mountains']} V={i['valleys']} "
          f"flat={i['flat_folds']} "
          f"{'OK' if i['valleys'] == 0 and i['mountains'] else 'BAD'}")
    if i['valleys'] or not i['mountains']:
        bad("convex solid should fold all-mountain")
    # A non-convex solid must fold BOTH ways: the spike edges are ridges
    # and the creases between neighbouring spikes are troughs.  (It has no
    # coplanar hinges: the triangle a spike raises over an edge lies in its
    # NEIGHBOUR's face plane, and the neighbour's lies in this one, so the
    # two meet at the base solid's dihedral rather than flat.)
    OV, OF, _s = rs.build_solid('PLATONIC', 'OCTA')
    SV, SF, _r = rs.stellate([list(v) for v in OV], OF)
    sn = build_net(rs.fit_cube(SV), SF, tabs=False, numbers=False)
    si = sn['info']
    ok = si['mountains'] > 0 and si['valleys'] > 0
    print(f"stella octangula folds: M={si['mountains']} "
          f"V={si['valleys']} flat={si['flat_folds']} "
          f"{'OK' if ok else 'BAD'}")
    if not ok:
        bad("a non-convex solid should fold both ways")

    # --- coplanar neighbours are classified flat, not mountain/valley ---
    # A cube with one face cut along its diagonal: the diagonal is a
    # genuine mesh edge between two coplanar triangles, so it prints as a
    # line but is not folded.  Rooting the tree at the first triangle
    # makes the other one its child across exactly that hinge.
    CV, CF, _s = rs.build_solid('PLATONIC', 'CUBE')
    q = CF[0]
    SPLIT = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] + \
            [list(f) for f in CF[1:]]
    SV2, SF2 = weld_vertices(CV, SPLIT, 1e-9)
    SF2 = orient_consistently(SV2, SF2)
    _fr, loc, _pl = face_frames(SV2, SF2)
    padj = face_adjacency(SF2)
    par, order = spanning_tree(SF2, padj, 0, None)
    fl = layout(SF2, loc, par, order)
    _isl, _c, iof = split_islands(SF2, par, order, fl, 1e-9)
    Amap = full_fold_maps(SV2, SF2, fl)
    _th, sg, _hg = fold_angles(SF2, par, order, fl, Amap, iof)
    nflat = sum(1 for fi in range(len(SF2))
                if par[fi] is not None and sg[fi] == 0)
    print(f"coplanar hinge on a split cube face: flat folds={nflat} "
          f"{'OK' if nflat == 1 else 'BAD'}")
    if nflat != 1:
        bad("a coplanar hinge was not classified flat")

    # --- the star solids whose faces do not weld to a surface ---
    # Their intersecting faces are emitted as separate polygons that cross
    # one another, so welding by position leaves edges carrying four faces
    # rather than two.  Refusing is the honest answer; GD is the Kepler
    # solid that does unfold.
    for sid in ('SSD', 'GSD', 'GI'):
        SV, SF, _s = rs.build_kepler(sid)
        try:
            build_net(SV, SF)
            bad(f"{sid} unfolded but should have been refused")
        except ValueError as e:
            okmsg = "two-faces-per-edge" in str(e)
            print(f"kepler {sid} refused: {str(e)[:58]}... "
                  f"{'OK' if okmsg else 'BAD'}")
            if not okmsg:
                bad(f"{sid} refused with the wrong message")

    # --- tabs and numbers ---
    V, F, _s = rs.build_solid('PLATONIC', 'CUBE')
    net = build_net(rs.fit_cube(V), F, tabs=True, numbers=True)
    i = net['info']
    ok = i['pairs'] == 7 and i['tabs'] + i['dropped'] == 7
    print(f"cube tabs: pairs={i['pairs']} tabs={i['tabs']} "
          f"dropped={i['dropped']} {'OK' if ok else 'BAD'}")
    if not ok:
        bad("cube should have 7 cut pairs, one tab each")
    seen = {}
    for (_a, _b, pid) in net['cuts']:
        seen[pid] = seen.get(pid, 0) + 1
    if sorted(seen) != list(range(1, i['pairs'] + 1)) or \
            any(c != 2 for c in seen.values()):
        bad("edge-match numbers are not a perfect pairing")
    if not digit_strokes(42) or not digit_strokes(7):
        bad("stroke font produced no geometry")

    print("RESULT:", "ALL OK" if not fails else f"FAILS {fails}")
    assert not fails
