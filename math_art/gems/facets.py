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
# Enumeration has a second tolerance, and it is not a formality.  A triple
# of planes locates a point by solving a 3x3 system whose determinant is
# the volume of the parallelepiped of the three unit normals; as that
# volume goes to zero the normals become coplanar and the point runs away.
# Published designs carry eight decimals, so a plane's offset is uncertain
# by ~1e-8, and a triple with |det| = d amplifies that to ~1e-8 / d.  For
# the point to stay inside the merge radius `snap` we therefore need
#
#     |det| > (input precision) / (snap radius) ~ 1e-8 / 1e-5 = 1e-3
#
# and triples below that carry no information -- the vertex they claim to
# define is pinned by better-conditioned triples through the same point.
# Measured on the fixtures below, the window that gets every one of them
# right is |det| in [1e-3, 1e-2]: under 1e-3 a high-valence culet shatters
# into a rosette of micro-vertices, and by 3e-2 the enumeration has begun
# discarding legitimate shallow vertices of the round brilliant.  The
# default sits at the geometric centre of that window, and the self-test
# asserts the whole window rather than the single default, so narrowing it
# is caught rather than discovered later.
#
# The tolerance that matters, `snap`, is in units of the girdle radius.
# Its default of 1e-5 sits deliberately between two scales: real cutters
# place facets to about 0.01 mm and 0.2 degrees (Sasian et al.), i.e.
# ~1e-3 of the radius, so 1e-5 is a hundred times finer than any physical
# stone; and float64 noise on these magnitudes is ~1e-15, so it is ten
# orders coarser than the arithmetic.  Anything in that window should give
# the same solid, which is what the self-test checks -- stability across a
# decade of `snap`, not a single residual threshold.
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


def intersect_halfspaces(N, d, snap=1e-5, tol=3e-3):
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

    keep = _dedupe_planes(N, d, R, tol)
    Nk, dk = N[keep], d[keep]

    P = _candidate_vertices(Nk, dk, R, tol, tol_contain=snap)
    if len(P) < 4:
        raise ValueError(f"only {len(P)} candidate vertices; the half-spaces "
                         f"do not bound a solid")

    P = _dedupe_points(P, snap * R / 16.0)
    label = _cluster_points(P, snap * R)
    V, residual = _snap_meetpoints(P, label, Nk, dk, R, tol_face=10.0 * snap)
    faces, owner, dropped = _assemble_faces(V, Nk, dk, R, tol_face=10.0 * snap)

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

    shattered = polytope_checks(intersect_halfspaces(Np, dp, tol=1e-6),
                                Np, dp)["closed"]
    starved = intersect_halfspaces(Ns, ds, tol=3e-2)
    good = (not shattered) and len(starved.faces) < 73
    ok &= good
    print(f"gems.facets: outside the window it does break, as documented "
          f"(1e-6 leaves the culet open; 3e-2 loses "
          f"{73 - len(starved.faces)} brilliant facets) "
          f"{'OK' if good else 'window claim is stale'}")

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
