# Half-space intersection: turning a set of facet planes into a solid.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python and numpy
# only -- no `bpy`.
#
# A cut gemstone is specified as PLANES, never as vertices: a design gives
# an angle, an index and a centre-to-facet distance per facet, and the
# stone is the intersection of the resulting half-spaces.  This is the
# dual of the convex-hull problem the polyhedron engine solves in
# `polyhedra/hull.py` (points -> faces), and it is not interchangeable
# with it.
#
# The algorithm is direct vertex enumeration with least-squares meetpoint
# snapping:
#
#   1. drop duplicate planes (a repeated index, or two tiers that coincide)
#   2. for every triple of planes, solve for their common point
#   3. keep the points satisfying every constraint -- those are the
#      candidate vertices
#   4. CLUSTER coincident candidates and replace each cluster by the
#      least-squares solution over all the planes meeting there
#   5. assemble one n-gon face per plane from its incident vertices
#
# Step 4 is the whole point.  Faceting designs are published to eight
# decimals, so at a MEETPOINT where many facets converge -- the culet of a
# 161-facet Portuguese cut has sixteen planes through one point -- the
# planes do not quite meet.  C(16,3) = 560 triples each produce a slightly
# different point, and any method that treats them as distinct vertices
# produces a fuzz of micro-facets instead of a culet.  Solving
#
#     min_p  sum_i (n_i . p - d_i)^2
#
# over every plane incident to the cluster gives one point, the best one,
# and reports how far the design's own planes were from concurrent.
#
# Two alternatives were rejected.  Iterative plane CLIPPING (Sutherland-
# Hodgman in 3-D) accumulates error exactly at the many-plane meetpoints
# where exactness matters most.  Building the DUAL point set and taking
# its convex hull is worse still at the same spot: sixteen planes through
# one meetpoint become sixteen nearly-coplanar dual points, i.e. sliver
# facets.  Enumeration computes every vertex directly from its own
# defining planes, so no error accumulates from vertex to vertex.
#
# MERGING CANDIDATES IS THE HARD PART, and distance is the wrong
# criterion for it.  A triple of planes locates a point through a 3x3
# solve whose determinant is the volume of the parallelepiped of the three
# unit normals; as that volume shrinks the point runs away, amplifying the
# design's own precision by 1/det.  Published designs carry FIVE decimals
# -- measured across 3,680 of them -- so the scatter reaches ~3e-4 of the
# girdle radius, while genuine distinct vertices are sometimes closer than
# that.  No radius separates the two populations, which is why tuning the
# merge distance plateaued at 40% of real designs closing however it was
# swept.
#
# The combinatorial criterion has no such problem.  A vertex IS the set of
# planes through it, so two candidates are the same vertex when they share
# three or more incident planes -- three being what determines a point.
# Scatter changes how far a candidate sits from its planes, never which
# planes they are.  Merging by incidence first, with distance kept only
# for points lying on fewer than three planes, takes real designs from 40%
# to 90% closed and convex.
#
# It also removed the lower bound on the conditioning floor.  Before, a
# high-valence culet shattered unless |det| was held above ~1e-3; now the
# Portuguese stress case closes at every floor from 1e-9 to 1e-1, because
# the scattered points are merged by what they lie on rather than by where
# they landed.  Only the UPPER bound survives, and for a different reason:
# by 3e-2 the enumeration is rejecting so many triples that legitimate
# shallow vertices of the round brilliant go missing.  The self-test
# asserts both facts.
#
# The tolerance that matters, `snap`, is in units of the girdle radius,
# and by default it is INFERRED from the data: `infer_precision` recovers
# the decimal grid the offsets were printed on, and the merge radius is a
# small multiple of that quantum, clamped to [1e-5, 3e-4].  The floor is
# anchored between two scales: real cutters place facets to about 0.01 mm
# and 0.2 degrees (Sasian et al.), i.e. ~1e-3 of the radius, so 1e-5 is a
# hundred times finer than any physical stone, yet ten orders coarser than
# float64 noise (~1e-15).  Anything near the floor should give the same
# solid, which is what the self-test checks -- stability across a decade
# of `snap`, not a single residual threshold.
#
# One tolerance must NOT ride along with `snap`: the face-membership net
# used after the least-squares snap.  `snap` has to cover the pre-snap
# scatter of the candidates, but a SNAPPED vertex sits within the planes'
# own concurrency residual (about one printed quantum) of every plane
# through it, so recognising it on its faces needs only a fixed, small
# margin -- and that margin must stay under the ~1e-3 spacing of the
# cutter's genuinely distinct features.  Scaling it as 10 x an inferred
# snap of 2e-4 handed faces a 2e-3 net, and a five-decimal Portuguese cut
# stopped closing: rings captured neighbouring vertices, and adjacent
# clusters were given identical incidence sets and least-squares snapped
# onto coincident duplicate points.  Hence `_FACE_TOL_CAP` below, and a
# self-test that builds that exact stone.
#
# References:
#   Robert W. Strickland, "GemCad for Windows Version 1.0 User's Guide",
#     GemSoft Enterprises -- the facet/plane model this implements.
#   Jose M. Sasian, Peter Yantzer, Tom Tivol, "The Optical Design of
#     Gemstones", Optics & Photonics News 14(4), 2003 -- the 0.01 mm /
#     0.2 degree cutting precision the tolerance is anchored to.
#   B. Grunbaum, "Convex Polytopes", 2nd ed., 2003 -- half-space
#     intersection and the face lattice.

import itertools
from collections import defaultdict
from typing import NamedTuple

import numpy as np

# Ceiling on the face-membership tolerance, in girdle radii.  Vertices are
# least-squares snapped before faces are assembled, so a vertex sits within
# the design's own concurrency residual of every plane through it -- the
# face net never needs to widen past this, however coarse the merge radius
# `snap` had to be.  It must stay well under ~1e-3, the physical spacing of
# distinct features (Sasian et al.'s 0.01 mm cutting precision): by 2e-3 a
# real Portuguese cut's faces start capturing neighbouring vertices.
_FACE_TOL_CAP = 3e-4


class Polytope(NamedTuple):
    """A solid: vertices, one n-gon per surviving plane, and provenance."""

    V: np.ndarray             # (k, 3) float64
    faces: tuple              # tuple of tuples of vertex indices, wound outward
    face_plane: tuple         # index into the ORIGINAL plane arrays per face
    residual: float           # worst meetpoint residual, in girdle radii
    dropped: tuple            # planes that carry no facet (cut away)


def _dedupe_planes(N, d, R, tol):
    """Indices of distinct planes; a repeated plane keeps its tightest copy.

    Duplicates arise from an index listed twice in a tier, or two tiers
    that happen to coincide.  Left in, each duplicate would raise a second
    identical face and break the "every edge borders exactly two faces"
    check, so they are removed rather than tolerated downstream.
    """
    q = 1e-9
    best = {}
    for i in range(len(N)):
        key = (round(float(N[i, 0]) / q), round(float(N[i, 1]) / q),
               round(float(N[i, 2]) / q))
        j = best.get(key)
        if j is None or d[i] < d[j] - tol * R:
            best[key] = i
    return np.array(sorted(best.values()), dtype=int)


def _candidate_vertices(N, d, R, tol, tol_contain):
    """Points where three planes meet that satisfy every other constraint.

    Enumerated one anchor plane at a time so the working set stays
    bounded: with M planes this is C(M, 3) triples in M numpy batches,
    never one array of them all.
    """
    M = len(N)
    out = []
    for i in range(M - 2):
        rest = np.arange(i + 1, M)
        if len(rest) < 2:
            break
        jj, kk = np.triu_indices(len(rest), k=1)
        j, k = rest[jj], rest[kk]
        na = N[i]
        nb, nc = N[j], N[k]
        cbc = np.cross(nb, nc)
        det = cbc @ na
        live = np.abs(det) > tol
        if not live.any():
            continue
        j, k, cbc, det = j[live], k[live], cbc[live], det[live]
        p = (d[i] * cbc
             + d[j][:, None] * np.cross(nc[live], na)
             + d[k][:, None] * np.cross(na, nb[live])) / det[:, None]
        # cheap radial cull first -- it removes the large majority
        p = p[np.einsum('ij,ij->i', p, p) <= (3.0 * R) ** 2]
        if not len(p):
            continue
        inside = np.all(N @ p.T <= (d + tol_contain * R)[:, None], axis=0)
        if inside.any():
            out.append(p[inside])
    if not out:
        return np.zeros((0, 3))
    return np.vstack(out)


def infer_precision(d):
    """The quantum the offsets were written to, from the offsets themselves.

    A published design is a TEXT file, so its distances carry however many
    decimals the author printed -- five, across the whole Datavue library.
    A design computed here carries full double precision.  Those want
    merge radii two orders apart, and no single default serves both: at
    1e-4 a 64-sided girdle's finely spaced vertices merge into each other,
    and at 1e-5 a five-decimal design's scatter never merges at all.

    Rather than make the caller declare it, recover it: find the coarsest
    grid the offsets all sit on.  Five-decimal data reports 1e-5 and
    computed data falls through to the floor.
    """
    a = np.abs(np.asarray(d, dtype=float))
    a = a[a > 0]
    if not len(a):
        return 1e-12
    for k in range(2, 10):
        q = 10.0 ** (-k)
        if np.all(np.abs(a / q - np.round(a / q)) < 1e-6):
            return q
    return 1e-12


def _dedupe_points(P, q):
    """Collapse candidates that agree to within `q` before clustering.

    Vertex enumeration is hugely redundant at a meetpoint: every one of
    the C(k, 3) triples drawn from the k planes through it produces the
    same point to within the design's own precision.  For k = 120 that is
    280,840 copies of one vertex.  Snapping to a grid of side `q` and
    taking uniques is an O(n log n) way to throw the copies away; `q` is
    chosen far finer than the clustering radius so it can only remove
    genuine duplicates, never merge distinct vertices -- the union-find
    pass that follows still does the real work.
    """
    if len(P) < 2:
        return P
    key = np.round(P / q).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return P[np.sort(first)]


def _merge_by_incidence(P, N, d, R, tol_inc):
    """Group candidates that lie on the SAME PLANES; returns a label array.

    Distance clustering asks "are these two points close?", and on real
    design data that question has no good answer: a vertex where facets
    meet at a shallow angle is located by an ill-conditioned solve, so its
    copies scatter further than distinct vertices sometimes lie apart.
    Measured on the Datavue library, the scatter reaches 3e-4 of the
    girdle radius while genuine vertices can be closer than that -- there
    is no radius that separates them, which is why every tolerance sweep
    plateaued at 40%.

    The combinatorial question has a clean answer.  A vertex IS the set of
    planes through it, so two candidates are the same vertex when they
    share three or more incident planes -- three because that is what it
    takes to determine a point.  Scatter cannot change which planes a
    point lies on, only how far from them it sits, so this criterion is
    stable exactly where the metric one is not.

    `tol_inc` is deliberately generous: it decides only MEMBERSHIP, and
    the least-squares snap afterwards decides position.
    """
    n = len(P)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # which planes does each candidate lie on?
    near2 = (30.0 * tol_inc * R) ** 2
    on = np.abs(P @ N.T - d) <= tol_inc * R          # (k, M) booleans
    inc = [np.nonzero(row)[0] for row in on]
    by_plane = defaultdict(list)
    for i, planes_i in enumerate(inc):
        for pl in planes_i:
            by_plane[pl].append(i)

    for i, planes_i in enumerate(inc):
        if len(planes_i) < 3:
            continue
        shared = defaultdict(int)
        for pl in planes_i:
            for j in by_plane[pl]:
                if j > i:
                    shared[j] += 1
        for j, c in shared.items():
            # Incidence AND proximity, not incidence alone.  Sharing three
            # planes is necessary but not sufficient: measured on the
            # parametric brilliant, 41 genuine vertices collapse to 26 on
            # the incidence test by itself, because near-coincident planes
            # let neighbouring vertices report a third shared plane they do
            # not really lie on.  The distance gate is deliberately loose --
            # it only has to separate distinct vertices, while incidence
            # does the work of tolerating scatter.
            if c >= 3 and np.dot(P[i] - P[j], P[i] - P[j]) <= near2:
                union(i, j)

    roots, label = {}, np.empty(n, dtype=int)
    placed = np.array([len(inc[i]) >= 3 for i in range(n)])
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        label[i] = roots[r]
    return label, placed


def _cluster_points(P, h):
    """Group points lying within `h` of each other; returns a label array.

    A uniform grid of side `h` plus union-find.  Points sharing a cell are
    merged unconditionally (they are within h*sqrt(3) by construction);
    only across cell boundaries is a distance test needed.

    The cross-boundary test is done in chunks, because the redundancy at a
    meetpoint is extreme: 120 planes through one culet yield C(120, 3) =
    280,840 triples that all land on the same point, and a naive outer
    product between two such cells asks for 107 GiB.  Callers should feed
    this a pre-deduplicated set (see `_dedupe_points`); the chunking is
    the second line of defence.
    """
    n = len(P)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    cells = defaultdict(list)
    for i, c in enumerate(map(tuple, np.floor(P / h).astype(np.int64))):
        cells[c].append(i)
    for members in cells.values():
        for m in members[1:]:
            union(members[0], m)
    # forward half of the 26-neighbourhood, so each pair is visited once
    nbrs = [o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)]
    h2 = h * h
    for c, members in cells.items():
        A = P[members]
        for o in nbrs:
            other = cells.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]))
            if not other:
                continue
            B = P[other]
            for s in range(0, len(A), 512):        # chunked: bound the memory
                blk = A[s:s + 512]
                dist2 = ((blk[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
                for a, b in zip(*np.nonzero(dist2 <= h2)):
                    union(members[s + a], other[b])
    roots = {}
    label = np.empty(n, dtype=int)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        label[i] = roots[r]
    return label


def _snap_meetpoints(P, label, N, d, R, tol_face):
    """One least-squares point per cluster, over every plane meeting it."""
    m = label.max() + 1 if len(label) else 0
    V = np.zeros((m, 3))
    worst = 0.0
    for c in range(m):
        blob = P[label == c]
        centre = blob.mean(axis=0)
        inc = np.abs(N @ centre - d) <= tol_face * R
        if inc.sum() >= 3:
            A, b = N[inc], d[inc]
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            # a least-squares "solution" is only meaningful if it stayed
            # put; a rank-deficient incidence set can throw it far away
            V[c] = sol if np.linalg.norm(sol - centre) <= tol_face * R * 10 \
                else centre
        else:
            V[c] = centre
        if inc.any():
            worst = max(worst, float(np.max(np.abs(N[inc] @ V[c] - d[inc]))))
    return V, worst / R


def _assemble_faces(V, N, d, R, tol_face):
    """One outward-wound n-gon per plane, from the vertices lying on it."""
    faces, owner, dropped = [], [], []
    for j in range(len(N)):
        on = np.nonzero(np.abs(V @ N[j] - d[j]) <= tol_face * R)[0]
        if len(on) < 3:
            dropped.append(j)
            continue
        pts = V[on]
        centre = pts.mean(axis=0)
        n = N[j]
        u = np.cross(n, [0.0, 0.0, 1.0])
        if np.linalg.norm(u) < 1e-9:
            u = np.cross(n, [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(n, u)
        rel = pts - centre
        order = np.argsort(np.arctan2(rel @ w, rel @ u))
        ring = on[order]
        loop = V[ring]
        # Newell normal: area, and the winding sense against the outward n
        nl = np.cross(loop - np.roll(loop, 1, axis=0),
                      loop - np.roll(loop, -1, axis=0)).sum(axis=0)
        if np.linalg.norm(nl) < 1e-12 * R * R:
            dropped.append(j)
            continue
        if nl @ n < 0:
            ring = ring[::-1]
        faces.append(tuple(int(i) for i in ring))
        owner.append(j)
    return tuple(faces), tuple(owner), tuple(dropped)


def intersect_halfspaces(N, d, snap=None, tol=3e-3, tol_inc=None):
    """Build the solid {x : N x <= d}.

    `snap` is the meetpoint merge radius in units of the girdle radius and
    `tol` the conditioning floor on a plane triple's determinant; both are
    derived in the module docstring.  Raises ValueError if the half-spaces
    do not bound a solid.
    """
    N = np.asarray(N, dtype=float)
    d = np.asarray(d, dtype=float)
    if N.ndim != 2 or N.shape[1] != 3 or len(N) != len(d):
        raise ValueError(f"need matching (M, 3) normals and (M,) offsets, "
                         f"got {N.shape} and {d.shape}")
    if len(N) < 4:
        raise ValueError(f"{len(N)} half-spaces cannot bound a solid")
    R = float(np.max(np.abs(d)))
    if not R > 0.0:
        raise ValueError("all facet distances are zero")
    if snap is None:
        # 20 quanta: enough to swallow the scatter an ill-conditioned
        # triple makes of the last printed digit, far less than the gap
        # between genuine vertices even on a 64-sided girdle.  Clamped at
        # both ends, because round data misleads the inference -- a cube's
        # offsets are all exactly 1, which sits on a 0.01 grid and would
        # otherwise ask for a merge radius that swallows the whole solid.
        snap = min(3e-4, max(1e-5, 20.0 * infer_precision(d) / R))

    keep = _dedupe_planes(N, d, R, tol)
    Nk, dk = N[keep], d[keep]

    P = _candidate_vertices(Nk, dk, R, tol, tol_contain=snap)
    if len(P) < 4:
        raise ValueError(f"only {len(P)} candidate vertices; the half-spaces "
                         f"do not bound a solid")

    P = _dedupe_points(P, snap * R / 16.0)
    # Incidence is PRIMARY, and distance is not allowed to override it.
    # Unioning the two partitions was tried and is wrong: a merge radius
    # coarse enough for five-decimal design data then swallows genuinely
    # distinct vertices of an exactly-computed one, and the parametric
    # brilliant collapsed from 41 vertices to 25.  Distance is applied
    # only to the leftovers -- points lying on fewer than three planes,
    # which incidence cannot place at all.
    label, placed = _merge_by_incidence(
        P, Nk, dk, R,
        tol_inc=snap if tol_inc is None else tol_inc)
    loose = np.nonzero(~placed)[0]
    if len(loose):
        sub = _cluster_points(P[loose], snap * R)
        base = int(label.max()) + 1 if len(label) else 0
        label[loose] = base + sub
    # renumber densely
    uniq = {v: k for k, v in enumerate(sorted(set(label.tolist())))}
    label = np.array([uniq[v] for v in label.tolist()], dtype=int)
    # The face tolerance answers a different question from `snap` and must
    # not inflate with it.  `snap` has to cover the PRE-snap scatter of the
    # candidates; after the least-squares snap a vertex sits within the
    # planes' own concurrency residual (~ the printed quantum) of every
    # plane through it, so recognising it there needs only a small, fixed
    # margin.  Left at 10 * snap, an inferred snap of 2e-4 gave faces a
    # 2e-3 net -- wide enough to catch NEIGHBOURING vertices of a real
    # Portuguese cut (features sit ~1e-3 apart, the cutter's own placement
    # precision), which broke closure two ways at once: rings picked up
    # foreign vertices, and adjacent clusters were handed identical
    # incidence sets and snapped onto coincident points.
    tol_face = min(10.0 * snap, _FACE_TOL_CAP)
    V, residual = _snap_meetpoints(P, label, Nk, dk, R, tol_face=tol_face)
    faces, owner, dropped = _assemble_faces(V, Nk, dk, R, tol_face=tol_face)

    used = sorted({i for f in faces for i in f})
    if len(used) != len(V):                       # drop unreferenced points
        remap = {old: new for new, old in enumerate(used)}
        V = V[used]
        faces = tuple(tuple(remap[i] for i in f) for f in faces)
    return Polytope(V=V, faces=faces,
                    face_plane=tuple(int(keep[j]) for j in owner),
                    residual=residual,
                    dropped=tuple(int(keep[j]) for j in dropped))


def polytope_checks(P, N=None, d=None, tol=1e-6):
    """Structural report on a built solid.

    Returns a dict with `closed` (every edge borders exactly two faces),
    `euler` (V - E + F), `manifold`, `aspect` (longest/shortest bounding
    box side) and, when the defining planes are supplied, `convex` and the
    worst constraint violation.

    The aspect ratio is reported because this repository has been bitten
    by it: a derived solid can satisfy every topological check and still
    have collapsed flat, so "it is watertight and Euler is 2" is not on
    its own evidence that the geometry is right.
    """
    edges = defaultdict(int)
    for f in P.faces:
        for a, b in zip(f, f[1:] + f[:1]):
            edges[(min(a, b), max(a, b))] += 1
    counts = set(edges.values())
    V, F, E = len(P.V), len(P.faces), len(edges)
    lo, hi = P.V.min(axis=0), P.V.max(axis=0)
    ext = hi - lo
    out = {
        "V": V, "E": E, "F": F,
        "euler": V - E + F,
        "closed": counts == {2},
        "manifold": counts <= {2},
        "extent": tuple(float(x) for x in ext),
        "aspect": float(ext.max() / ext.min()) if ext.min() > 0 else float("inf"),
        "residual": P.residual,
    }
    if N is not None and d is not None:
        viol = float(np.max(np.asarray(N) @ P.V.T
                            - np.asarray(d)[:, None])) if V else 0.0
        R = float(np.max(np.abs(d)))
        out["violation"] = viol / R
        out["convex"] = viol <= tol * R
    return out


def volume(P):
    """Enclosed volume by the divergence theorem over the n-gon faces."""
    total = 0.0
    for f in P.faces:
        loop = P.V[list(f)]
        a = loop[0]
        for b, c in zip(loop[1:-1], loop[2:]):
            total += np.dot(a, np.cross(b - a, c - a))
    return abs(total) / 6.0


# --------------------------------------------------------------------------

def _cube_planes():
    N = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0],
                  [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
    return N, np.ones(6)


def _portuguese_stress(tiers=5, index_step=4, gear=96):
    """A pavilion of many tiers whose facets all converge on one culet.

    The stress case the meetpoint snapping exists for: every tier is aimed
    so its planes pass through the same point on the axis, so the culet is
    a vertex of very high valence and the triples through it disagree at
    the level of the input's own precision.  Distances are rounded to
    eight decimals on purpose -- that is the precision published designs
    carry, and rounding is what makes the planes miss each other.
    """
    import math
    apex = -1.0
    N, d = [], []
    for t in range(tiers):
        ang = math.radians(38.0 + 2.0 * t)
        for i in range(0, gear, index_step):
            phi = 2.0 * math.pi * i / gear
            n = (math.sin(ang) * math.cos(phi), math.sin(ang) * math.sin(phi),
                 -math.cos(ang))
            off = round(n[2] * apex, 8)            # plane through (0, 0, apex)
            N.append(n)
            d.append(off)
    for i in range(0, gear, index_step):           # girdle
        phi = 2.0 * math.pi * i / gear
        N.append((math.cos(phi), math.sin(phi), 0.0))
        d.append(1.0)
    N.append((0.0, 0.0, 1.0))                      # table
    d.append(0.35)
    return np.array(N), np.array(d)


def _selftest():
    import time
    try:
        from . import design as _design
    except ImportError:
        import design as _design

    ok = True

    # --- the cube: the smallest complete check -----------------------------
    N, d = _cube_planes()
    P = intersect_halfspaces(N, d)
    chk = polytope_checks(P, N, d)
    good = (len(P.V) == 8 and len(P.faces) == 6
            and all(len(f) == 4 for f in P.faces) and chk["closed"]
            and chk["euler"] == 2 and chk["convex"])
    ok &= good
    print(f"gems.facets: six half-spaces give a cube -- 8 vertices, 6 quads "
          f"(got {len(P.V)}, {len(P.faces)}) {'OK' if good else 'wrong'}")
    good = abs(volume(P) - 8.0) < 1e-12
    ok &= good
    print(f"gems.facets: its volume is 8 (got {volume(P):.12f}) "
          f"{'OK' if good else 'wrong'}")

    # faces must come out as single n-gons, never triangle fans
    good = all(len(f) == 4 for f in P.faces)
    ok &= good
    print(f"gems.facets: each planar face is one polygon, not a fan "
          f"{'OK' if good else 'fragmented'}")

    # a duplicated plane must not raise a second coincident face
    N2 = np.vstack([N, N[0:1]])
    d2 = np.concatenate([d, d[0:1]])
    P2 = intersect_halfspaces(N2, d2)
    good = len(P2.faces) == 6 and polytope_checks(P2)["closed"]
    ok &= good
    print(f"gems.facets: a repeated plane is deduplicated "
          f"(got {len(P2.faces)} faces) {'OK' if good else 'duplicated'}")

    # --- the Standard Round Brilliant --------------------------------------
    Ns, ds, tid = _design.planes(_design.SRB_GEMCAD)
    t0 = time.perf_counter()
    S = intersect_halfspaces(Ns, ds)
    dt = time.perf_counter() - t0
    chk = polytope_checks(S, Ns, ds)
    good = len(S.faces) == 73 and not S.dropped
    ok &= good
    print(f"gems.facets: the SRB keeps all 73 facets "
          f"(got {len(S.faces)}, dropped {len(S.dropped)}) "
          f"{'OK' if good else 'facets missing'}")
    good = chk["closed"] and chk["euler"] == 2
    ok &= good
    print(f"gems.facets: the SRB is closed with Euler 2 "
          f"(V{chk['V']} - E{chk['E']} + F{chk['F']} = {chk['euler']}) "
          f"{'OK' if good else 'not a closed surface'}")
    good = chk["convex"]
    ok &= good
    print(f"gems.facets: every SRB vertex satisfies every constraint "
          f"(worst {chk['violation']:.2e}) {'OK' if good else 'not convex'}")
    good = S.residual < 1e-6
    ok &= good
    print(f"gems.facets: SRB meetpoint residual {S.residual:.2e} < 1e-6 "
          f"{'OK' if good else 'meetpoints do not meet'}")
    good = 1.0 <= chk["aspect"] <= 3.0
    ok &= good
    print(f"gems.facets: the SRB has not collapsed flat "
          f"(bbox aspect {chk['aspect']:.3f}) {'OK' if good else 'degenerate'}")

    # the facet count splits the way the trade counts it
    tiers = [tid[j] for j in S.face_plane]
    girdle = sum(1 for t in tiers if t == 0)
    pav = sum(1 for t in tiers if t in (1, 2))
    crown = sum(1 for t in tiers if t in (3, 4, 5, 6))
    good = (girdle, pav, crown) == (16, 24, 33)
    ok &= good
    print(f"gems.facets: 16 girdle + 24 pavilion + 33 crown "
          f"(got {girdle} + {pav} + {crown}) {'OK' if good else 'wrong split'}")

    # 8-fold symmetry: rotating by 45 degrees must map the solid to itself
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    rot = S.V @ Rz.T
    dist = np.abs(rot[:, None, :] - S.V[None, :, :]).max(axis=2).min(axis=1)
    worst = float(dist.max())
    good = worst < 1e-9
    ok &= good
    print(f"gems.facets: the SRB is invariant under a 45 degree turn "
          f"(worst {worst:.2e}) {'OK' if good else 'asymmetric'}")

    # --- the meetpoint stress case ----------------------------------------
    # 120 pavilion planes aimed through one culet -- harsher than any real
    # design (a 161-facet Portuguese cut has ~16 there), and the case that
    # both tolerances exist for.
    Np, dp = _portuguese_stress()
    counts = []
    for snap in (3e-6, 1e-5, 3e-5):
        Q = intersect_halfspaces(Np, dp, snap=snap)
        cq = polytope_checks(Q, Np, dp)
        counts.append((len(Q.V), len(Q.faces), cq["closed"]))
    good = len(set(counts)) == 1 and counts[0][2]
    ok &= good
    print(f"gems.facets: {len(Np)} planes converging on a culet give the same "
          f"solid across a decade of snap {counts} "
          f"{'OK' if good else 'unstable'}")

    Q = intersect_halfspaces(Np, dp)
    apex = Q.V[np.argmin(Q.V[:, 2])]
    near = int(np.sum(np.linalg.norm(Q.V - apex, axis=1) < 1e-3))
    good = near == 1
    ok &= good
    print(f"gems.facets: the culet is a single vertex, not a cluster "
          f"(got {near}) {'OK' if good else 'fragmented'}")

    # --- the conditioning window -------------------------------------------
    # The default is the geometric centre of a validated window; assert the
    # WINDOW, so a change that narrows it is caught here and not in a
    # user's stone.  Below 1e-3 the culet shatters; by 3e-2 the round
    # brilliant starts losing legitimate shallow vertices.
    Nc, dc = _cube_planes()
    window, edges = (1e-3, 3e-3, 1e-2), []
    for t in window:
        rows = []
        for (Na, da, want) in ((Nc, dc, (8, 6)), (Ns, ds, (57, 73))):
            A = intersect_halfspaces(Na, da, tol=t)
            rows.append((len(A.V), len(A.faces)) == want
                        and polytope_checks(A, Na, da)["closed"])
        B = intersect_halfspaces(Np, dp, tol=t)
        rows.append(polytope_checks(B, Np, dp)["closed"])
        edges.append(all(rows))
    good = all(edges)
    ok &= good
    print(f"gems.facets: cube, brilliant and culet all survive the whole "
          f"conditioning window {window} {'OK' if good else 'window broken'}")

    # Incidence merging removed the window's LOWER bound: the culet used
    # to shatter below 1e-3 and now survives every floor down to 1e-9,
    # because scattered candidates are merged by the planes they lie on
    # rather than by where they landed.
    low = all(polytope_checks(intersect_halfspaces(Np, dp, tol=t),
                              Np, dp)["closed"]
              for t in (1e-9, 1e-6, 1e-3))
    ok &= low
    print(f"gems.facets: the culet survives every conditioning floor from "
          f"1e-9 up -- incidence merging removed the lower bound "
          f"{'OK' if low else 'the lower bound is back'}")

    # The UPPER bound is still real, and for a different reason: too high
    # a floor rejects the triples that locate legitimate shallow vertices.
    starved = intersect_halfspaces(Ns, ds, tol=3e-2)
    good = len(starved.faces) < 73
    ok &= good
    print(f"gems.facets: too high a floor still starves the brilliant "
          f"(3e-2 loses {73 - len(starved.faces)} facets) "
          f"{'OK' if good else 'upper bound claim is stale'}")

    # --- published precision: the five-decimal Portuguese ------------------
    # The regression this file once shipped: with offsets carrying five
    # decimals (what the Datavue library prints), the inferred merge
    # radius rises to 2e-4, and a face tolerance scaled as 10 x snap
    # handed every facet a 2e-3 net -- wide enough to catch NEIGHBOURING
    # vertices of a real Portuguese cut, whose features sit ~1e-3 apart.
    # The net must widen with nothing: a snapped vertex sits within about
    # one quantum of its planes, so `_FACE_TOL_CAP` bounds it.
    global _FACE_TOL_CAP
    try:
        from . import brilliant as _brilliant
    except ImportError:
        import brilliant as _brilliant
    Nr, dr, _ = _design.planes(_brilliant.portuguese())
    dr5 = np.round(dr, 5)
    R5 = float(np.max(np.abs(dr5)))
    inferred = min(3e-4, max(1e-5, 20.0 * infer_precision(dr5) / R5))
    good = inferred >= 1e-4
    ok &= good
    print(f"gems.facets: five-decimal offsets inflate the inferred snap "
          f"(got {inferred:.0e}) {'OK' if good else 'inference is dead'}")
    G = intersect_halfspaces(Nr, dr5)
    cg = polytope_checks(G, Nr, dr5, tol=1e-3)
    apex = G.V[np.argmin(G.V[:, 2])]
    near = int(np.sum(np.linalg.norm(G.V - apex, axis=1) < 5e-3))
    sep = np.linalg.norm(G.V[:, None, :] - G.V[None, :, :], axis=2)
    np.fill_diagonal(sep, np.inf)
    good = (len(G.faces) == 161 and cg["closed"] and cg["convex"]
            and near == 1 and float(sep.min()) > 1e-3)
    ok &= good
    print(f"gems.facets: a five-decimal Portuguese cut still closes at the "
          f"inferred defaults -- 161 facets, one culet, no coincident "
          f"vertices (got {len(G.faces)} faces, culet {near}, min "
          f"separation {float(sep.min()):.1e}) "
          f"{'OK' if good else 'the published-precision regression is back'}")
    # ... and the cap is load-bearing: uncapped, the same stone fails.
    cap = _FACE_TOL_CAP
    try:
        _FACE_TOL_CAP = 1e9
        loosed = polytope_checks(intersect_halfspaces(Nr, dr5),
                                 Nr, dr5, tol=1e-3)["closed"]
    finally:
        _FACE_TOL_CAP = cap
    good = not loosed and 10.0 * 1e-5 <= _FACE_TOL_CAP < 1e-3
    ok &= good
    print(f"gems.facets: the face-tolerance cap is doing the work -- "
          f"uncapped, the same stone does not close "
          f"{'OK' if good else 'cap claim is stale'}")

    # --- performance -------------------------------------------------------
    t0 = time.perf_counter()
    intersect_halfspaces(Np, dp)
    big = time.perf_counter() - t0
    good = big < 5.0
    ok &= good
    print(f"gems.facets: {len(Np)} planes in {big:.2f}s, SRB in {dt:.2f}s "
          f"{'OK' if good else 'too slow for an operator'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.facets self-test failed")
