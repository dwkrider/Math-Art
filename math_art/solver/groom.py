# In-loop mesh grooming: flips, tangential smoothing, splits, collapses.
#
# Part of the shared solver core (`math_art/solver/`).  NumPy only.
#
# A relaxation that never touches its mesh degrades it: triangles
# collapse into slivers, cotangent weights go wild, and the solver
# either folds the surface or has to be capped.  Surface Evolver's
# practical superiority comes from continuously grooming the mesh
# between gradient steps (`u` equiangulate, `V` vertex average, `t`/`l`
# pop/refine).  Two layers here:
#
# Count-preserving layer (safe for callers holding quad display lists):
#
#   * delaunay_flips: flip the diagonal of the quad formed by the two
#     triangles of an interior edge when the opposite angles sum to
#     more than pi (equivalently, when the edge's cotangent weight is
#     negative) by more than a hysteresis margin, with a quality and a
#     normal-consistency veto so a flip can never invert a triangle.
#     Directly repairs the negative cotan weights the Laplace solvers
#     depend on.
#
#   * tangential_smooth: move each free vertex toward the area-weighted
#     average of its neighbours, with the displacement projected onto
#     the tangent plane (the normal component removed), so mesh quality
#     improves with (to first order) zero shape change -- Evolver's `V`
#     in VOLKEEP mode.
#
# Both operate in place on (V, T) index arrays and preserve counts, so
# callers' pinned-vertex masks and quad face lists stay valid.
#
# Topology-changing layer (returns NEW arrays plus remapped masks --
# vertex/triangle counts change, so quad display lists do NOT survive):
#
#   * split_long_edges: split interior edges longer than 4/3 of the
#     target length at their exact midpoint (Botsch-Kobbelt).  An exact
#     midpoint split leaves the surface point set unchanged, so on an
#     embedded mesh it cannot create a self-intersection; only the
#     degeneracy/pinned vetoes apply.
#
#   * collapse_short_edges: collapse interior edges shorter than 4/5 of
#     the target length (Botsch-Kobbelt hysteresis) and/or shorter than
#     sqrt(2*step) (Evolver's autopop criterion: an edge that motion by
#     mean curvature at time step `step` would close within one step).
#     Every candidate runs the full static safety-veto battery -- link
#     condition, pinned/boundary/feature protection, label-class
#     compatibility, hysteresis bound, normal-inversion, sliver floor,
#     and an exact static self-intersection test -- and is REFUSED, not
#     repaired, on any failure.  The design point (measured on the IPC
#     collision-guard branch): a topology operation that breaks
#     embeddedness is unrecoverable downstream, so verify-and-refuse
#     beats verify-and-repair.
#
#   * nullspace_smooth: El Topo's null-space smoothing -- per vertex,
#     eigendecompose A = sum(area_i n_i n_i^T) over incident triangles
#     and project the Laplacian displacement onto the small-eigenvalue
#     subspace.  On smooth regions this is tangential smoothing; on a
#     ridge it moves only along the ridge; at a corner the vertex
#     freezes.  No feature tagging needed.
#
#   * groom_topo: one Botsch-Kobbelt cycle (split, collapse, flip,
#     smooth) with an optional exact global crossing verification that
#     reverts the whole cycle if it added a crossing.
#
# References:
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992) -- equiangulation with hysteresis, tangential vertex
#       averaging, autopop (trirevis.c, veravg.c, verpopst.c:
#       autopop_detect uses the length threshold sqrt(2*scale)).
#   M. Botsch and L. Kobbelt, "A Remeshing Approach to Multiresolution
#       Modeling", Symposium on Geometry Processing (2004) -- the
#       incremental remeshing cycle: split edges > 4/3 L, collapse
#       edges < 4/5 L into the hysteresis bracket, flip, relax.
#   T. Brochu and R. Bridson, "Robust Topological Operations for
#       Dynamic Explicit Surfaces", SIAM J. Sci. Comput. 31(4) (2009)
#       -- static safety vetoes on improvement operations, null-space
#       smoothing (El Topo, BSD-2-Clause; meshsmoother.cpp:196-315,
#       edgecollapser.cpp safety tests).
#   T. K. Dey, H. Edelsbrunner, S. Guha, D. V. Nekhayev, "Topology
#       preserving edge contraction", Publ. Inst. Math. 66 (1999) --
#       the link condition for collapse.

import math

import numpy as np


def _tri_normals_areas(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    a2 = np.linalg.norm(n, axis=1)
    return n / np.maximum(a2, 1e-300)[:, None], 0.5 * a2


def _tri_min_quality(V, tri_list):
    """Min radius-ratio quality over a list of index triples."""
    q = 1.0
    for (i, j, k) in tri_list:
        a = np.linalg.norm(V[j] - V[k])
        b = np.linalg.norm(V[k] - V[i])
        c = np.linalg.norm(V[i] - V[j])
        s = 0.5 * (a + b + c)
        A = 0.5 * np.linalg.norm(np.cross(V[j] - V[i], V[k] - V[i]))
        q = min(q, 8.0 * A * A / max(s * a * b * c, 1e-300))
    return q


def delaunay_flips(V, T, margin=1e-3, max_sweeps=3, tri_groups=None):
    """Equiangulate: flip interior edges whose opposite angles sum to
    more than pi (cot_c + cot_d < -margin), one non-conflicting batch
    per sweep.  Modifies T in place; returns the number of flips.

    Vetoes per candidate: the replacement edge must not already exist;
    neither new triangle may be degenerate or worse in min quality than
    the pair it replaces; and the new pair's normals must agree with
    the old pair's average normal (no inversion).

    tri_groups (ntri,) restricts flips to pairs in the SAME group --
    for multi-material (region-pair-labeled) meshes a flip across two
    different films would corrupt the labeling, so it is refused (the
    per-face labels then stay valid untouched: a flip rewrites both
    face slots within one film).  Non-manifold edges (Plateau borders,
    where three films meet) are never flip candidates in the first
    place, since only edges shared by exactly two faces qualify."""
    total = 0
    for _ in range(max_sweeps):
        # vectorised edge table: directed corner edges, sorted for the
        # undirected key, grouped by np.unique
        ntri = len(T)
        de = np.concatenate([T[:, [1, 2]], T[:, [2, 0]], T[:, [0, 1]]])
        tri_of = np.tile(np.arange(ntri), 3)
        corner_of = np.repeat(np.arange(3), ntri)
        key = np.sort(de, axis=1)
        uniq, inv, counts = np.unique(key, axis=0, return_inverse=True,
                                      return_counts=True)
        order = np.argsort(inv, kind='stable')
        # interior manifold edges appear exactly twice
        interior = counts == 2
        starts = np.zeros(len(uniq), dtype=np.int64)
        starts[1:] = np.cumsum(counts)[:-1]
        e_ids = np.nonzero(interior)[0]
        if not len(e_ids):
            break
        u1 = order[starts[e_ids]]
        u2 = order[starts[e_ids] + 1]
        t1s, c1s = tri_of[u1], corner_of[u1]
        t2s, c2s = tri_of[u2], corner_of[u2]
        aa, bb = uniq[e_ids, 0], uniq[e_ids, 1]
        cc = T[t1s, c1s]
        dd = T[t2s, c2s]
        good = cc != dd
        if tri_groups is not None:
            good &= np.asarray(tri_groups)[t1s] == np.asarray(tri_groups)[t2s]
        # cotangents at the two opposite corners, batched
        Ua = V[aa] - V[cc]
        Va = V[bb] - V[cc]
        Ub = V[aa] - V[dd]
        Vb = V[bb] - V[dd]
        cr1 = np.maximum(np.linalg.norm(np.cross(Ua, Va), axis=1), 1e-300)
        cr2 = np.maximum(np.linalg.norm(np.cross(Ub, Vb), axis=1), 1e-300)
        score = (np.einsum('ij,ij->i', Ua, Va) / cr1
                 + np.einsum('ij,ij->i', Ub, Vb) / cr2)
        sel = np.nonzero(good & (score < -margin))[0]
        if not len(sel):
            break
        edge_set = {tuple(k) for k in uniq}
        cand = [(float(score[i]), (int(aa[i]), int(bb[i])),
                 int(t1s[i]), int(t2s[i]), int(cc[i]), int(dd[i]))
                for i in sel]
        cand.sort()                           # most non-Delaunay first
        used = np.zeros(len(T), dtype=bool)
        flips = 0
        nrm, _ = _tri_normals_areas(V, T)
        for score, (a, b), t1, t2, c, d in cand:
            if used[t1] or used[t2]:
                continue
            ck = (c, d) if c < d else (d, c)
            if ck in edge_set:
                continue                      # new edge already present
            # orientation-consistent replacement: find winding of edge
            # (a, b) in t1 so the flipped pair keeps the surface
            # orientation
            tri1 = [int(x) for x in T[t1]]
            i = tri1.index(a)
            if tri1[(i + 1) % 3] != b:
                a, b = b, a                   # t1 traverses b -> a
            newA = (a, d, c)
            newB = (d, b, c)
            old_q = _tri_min_quality(V, [tuple(int(x) for x in T[t1]),
                                         tuple(int(x) for x in T[t2])])
            new_q = _tri_min_quality(V, [newA, newB])
            if new_q <= old_q:
                continue                      # improvement margin veto
            ref = nrm[t1] + nrm[t2]
            nA = np.cross(V[newA[1]] - V[newA[0]], V[newA[2]] - V[newA[0]])
            nB = np.cross(V[newB[1]] - V[newB[0]], V[newB[2]] - V[newB[0]])
            if (nA @ ref) <= 0.0 or (nB @ ref) <= 0.0:
                continue                      # inversion veto
            T[t1] = newA
            T[t2] = newB
            used[t1] = used[t2] = True
            edge_set.discard((min(a, b), max(a, b)))
            edge_set.add(ck)
            flips += 1
        total += flips
        if flips == 0:
            break
    return total


def tangential_smooth(V, T, fixed=None, lam=0.25, iters=1):
    """Area-weighted neighbour averaging with the normal component of
    the displacement projected out (pure tangential regularisation --
    Evolver's `V` in VOLKEEP mode).  Modifies V in place."""
    n = len(V)
    for _ in range(iters):
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        fa = 0.5 * np.linalg.norm(fn, axis=1)
        # vertex normals (area-weighted)
        vn = np.zeros((n, 3))
        for k in range(3):
            np.add.at(vn, T[:, k], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)
        # area-weighted neighbour average over triangle corners: each
        # corner pulls toward the opposite edge's endpoints, weighted by
        # the triangle area (a robust stand-in for Evolver's per-edge
        # facet-area weights, identical on closed fans)
        acc = np.zeros((n, 3))
        wsum = np.zeros(n)
        for k in range(3):
            i = T[:, k]
            j1 = T[:, (k + 1) % 3]
            j2 = T[:, (k + 2) % 3]
            np.add.at(acc, i, fa[:, None] * (V[j1] + V[j2]))
            np.add.at(wsum, i, 2.0 * fa)
        target = acc / np.maximum(wsum, 1e-300)[:, None]
        disp = target - V
        disp -= np.einsum('ij,ij->i', disp, vn)[:, None] * vn
        if fixed is not None:
            disp[fixed] = 0.0
        V += lam * disp
    return V


def groom(V, T, fixed=None, flips=True, smooth_lam=0.25,
          flip_margin=1e-3, tri_groups=None):
    """One groom cycle: equiangulate, then tangentially smooth.
    Modifies V and T in place; returns the number of edge flips.
    tri_groups (see delaunay_flips) keeps flips inside one film on
    labeled multi-material meshes."""
    nf = (delaunay_flips(V, T, margin=flip_margin, tri_groups=tri_groups)
          if flips else 0)
    if smooth_lam > 0.0:
        tangential_smooth(V, T, fixed=fixed, lam=smooth_lam)
    return nf


# --------------------------------------------------------------------------
# topology-changing layer: splits, collapses, null-space smoothing
# --------------------------------------------------------------------------

def _edge_table(T):
    """Undirected edge table of a triangle list.

    Returns (uniq, counts, order, starts, tri_of, dir_edges) where
    `uniq` is the (ne, 2) sorted vertex-pair array, `counts` the number
    of incident triangles per edge, and for edge e the incident
    (triangle, directed edge as traversed) records are
    `(tri_of[order[starts[e] + k]], dir_edges[order[starts[e] + k]])`
    for k < counts[e]."""
    ntri = len(T)
    de = np.concatenate([T[:, [1, 2]], T[:, [2, 0]], T[:, [0, 1]]])
    tri_of = np.tile(np.arange(ntri), 3)
    key = np.sort(de, axis=1)
    uniq, inv, counts = np.unique(key, axis=0, return_inverse=True,
                                  return_counts=True)
    order = np.argsort(inv, kind='stable')
    starts = np.zeros(len(uniq), dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    return uniq, counts, order, starts, tri_of, de


def _pierce_pairs(V, Eids, Tids, tol=1e-7):
    """Count (edge, triangle) pairs where the open segment strictly
    pierces the open triangle interior, excluding pairs that share a
    vertex id.  The same Moeller-Trumbore predicate as the repo's
    embeddedness gate (minsurf.plateau._selfx_crossings), on explicit
    edge / triangle index lists.  Bbox-prefiltered."""
    Eids = np.asarray(Eids, dtype=np.int64).reshape(-1, 2)
    Tids = np.asarray(Tids, dtype=np.int64).reshape(-1, 3)
    if not len(Eids) or not len(Tids):
        return 0
    P0, P1 = V[Eids[:, 0]], V[Eids[:, 1]]
    elo, ehi = np.minimum(P0, P1), np.maximum(P0, P1)
    A, B, C = V[Tids[:, 0]], V[Tids[:, 1]], V[Tids[:, 2]]
    tlo = np.minimum(np.minimum(A, B), C)
    thi = np.maximum(np.maximum(A, B), C)
    count = 0
    for i0 in range(0, len(Eids), 512):
        i1 = min(i0 + 512, len(Eids))
        ov = np.all((elo[i0:i1, None, :] <= thi[None, :, :] + tol)
                    & (ehi[i0:i1, None, :] >= tlo[None, :, :] - tol),
                    axis=2)
        ei, ti = np.nonzero(ov)
        ei += i0
        if not len(ei):
            continue
        share = np.zeros(len(ei), dtype=bool)
        for cc in range(2):
            for dd in range(3):
                share |= (Eids[ei, cc] == Tids[ti, dd])
        ei, ti = ei[~share], ti[~share]
        if not len(ei):
            continue
        o = V[Eids[ei, 0]]
        d = V[Eids[ei, 1]] - o
        e1 = V[Tids[ti, 1]] - V[Tids[ti, 0]]
        e2 = V[Tids[ti, 2]] - V[Tids[ti, 0]]
        pv = np.cross(d, e2)
        det = np.einsum('ij,ij->i', e1, pv)
        good = np.abs(det) > 1e-14
        inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
        tv = o - V[Tids[ti, 0]]
        uu = np.einsum('ij,ij->i', tv, pv) * inv
        qv = np.cross(tv, e1)
        vv = np.einsum('ij,ij->i', d, qv) * inv
        tt = np.einsum('ij,ij->i', e2, qv) * inv
        count += int(np.sum(good & (uu > tol) & (vv > tol)
                            & (uu + vv < 1 - tol)
                            & (tt > tol) & (tt < 1 - tol)))
    return count


def crossing_count(V, T, tol=1e-7):
    """Exact self-intersection count of the whole mesh: edges piercing
    the open interior of triangles they share no vertex with.  Same
    predicate as the repo embeddedness gate; provided here so the
    solver core stays free of engine imports."""
    T = np.asarray(T, dtype=np.int64)
    E = np.unique(np.sort(np.concatenate(
        [T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    return _pierce_pairs(np.asarray(V, float), E, T, tol=tol)


def _median_edge_length(V, T):
    E = np.unique(np.sort(np.concatenate(
        [T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    return float(np.median(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)))


def _new_vetoes():
    return {"pinned": 0, "boundary": 0, "nonmanifold": 0, "link": 0,
            "valence": 0, "label": 0, "hysteresis": 0, "inversion": 0,
            "quality": 0, "selfx": 0, "locked": 0}


def split_long_edges(V, T, target, fixed=None, tri_groups=None,
                     high=4.0 / 3.0, quality_floor=0.05, max_sweeps=3,
                     vetoes=None):
    """Split interior manifold edges longer than ``high * target`` at
    their exact midpoint (Botsch-Kobbelt 2004, the refinement half of
    the 4/3 -- 4/5 hysteresis bracket).

    Because the midpoint lies on the edge, a split changes the mesh's
    point set not at all: on an embedded mesh it cannot create a
    self-intersection, so no intersection test is run.  Static vetoes:

      * pinned: both endpoints fixed (a rim edge -- rim resolution
        belongs to the caller), refused;
      * boundary / non-manifold edges: never candidates;
      * quality: none of the four child triangles may fall below
        ``min(quality_floor, quality of the parent pair)`` in
        radius-ratio quality, and none may be degenerate.

    Labels: each child inherits its parent triangle's group, which is
    always label-consistent.  Returns
    ``(V2, T2, fixed2, tri_groups2, n_splits)`` -- NEW arrays; the
    midpoint vertices are appended free (never fixed).  ``vetoes`` (a
    dict from ``_new_vetoes``) is tallied in place when given."""
    V = np.asarray(V, dtype=float)
    T = np.asarray(T, dtype=np.int64)
    fx = (np.zeros(len(V), dtype=bool) if fixed is None
          else np.asarray(fixed, dtype=bool).copy())
    groups = None if tri_groups is None else np.asarray(tri_groups).copy()
    if vetoes is None:
        vetoes = _new_vetoes()
    thresh = float(high) * float(target)
    total = 0
    for _ in range(max_sweeps):
        uniq, counts, order, starts, tri_of, de = _edge_table(T)
        interior = counts == 2
        aa, bb = uniq[:, 0], uniq[:, 1]
        L = np.linalg.norm(V[aa] - V[bb], axis=1)
        cand_e = np.nonzero(interior & (L > thresh))[0]
        vetoes["boundary"] += int(np.sum((counts == 1) & (L > thresh)))
        vetoes["nonmanifold"] += int(np.sum((counts > 2) & (L > thresh)))
        if not len(cand_e):
            break
        cand = sorted(((float(L[e]), int(e)) for e in cand_e),
                      reverse=True)
        used = np.zeros(len(T), dtype=bool)
        newV, newT, newG = [], [], []
        replaced = np.zeros(len(T), dtype=bool)
        nxt = len(V)
        n_sweep = 0
        for _, e in cand:
            a, b = int(uniq[e, 0]), int(uniq[e, 1])
            if fx[a] and fx[b]:
                vetoes["pinned"] += 1
                continue
            u1 = order[starts[e]]
            u2 = order[starts[e] + 1]
            t1, t2 = int(tri_of[u1]), int(tri_of[u2])
            if used[t1] or used[t2]:
                vetoes["locked"] += 1
                continue
            # directed edges as traversed by each triangle, and the
            # opposite corners
            p1, q1 = int(de[u1, 0]), int(de[u1, 1])
            p2, q2 = int(de[u2, 0]), int(de[u2, 1])
            c = int(np.sum(T[t1]) - p1 - q1)
            d = int(np.sum(T[t2]) - p2 - q2)
            mid = 0.5 * (V[a] + V[b])
            m = nxt
            # child triangles, orientation preserved
            kids = [(p1, m, c), (m, q1, c), (p2, m, d), (m, q2, d)]
            # quality veto, evaluated with the midpoint appended lazily
            Vq = {m: mid}

            def _q(tris):
                q = 1.0
                for (i, j, k) in tris:
                    Pi = Vq[i] if i in Vq else V[i]
                    Pj = Vq[j] if j in Vq else V[j]
                    Pk = Vq[k] if k in Vq else V[k]
                    ea = np.linalg.norm(Pj - Pk)
                    eb = np.linalg.norm(Pk - Pi)
                    ec = np.linalg.norm(Pi - Pj)
                    s = 0.5 * (ea + eb + ec)
                    Aq = 0.5 * np.linalg.norm(np.cross(Pj - Pi, Pk - Pi))
                    q = min(q, 8.0 * Aq * Aq / max(s * ea * eb * ec,
                                                   1e-300))
                return q

            old_q = _tri_min_quality(V, [tuple(int(x) for x in T[t1]),
                                         tuple(int(x) for x in T[t2])])
            new_q = _q(kids)
            if new_q < min(quality_floor, old_q) or new_q <= 0.0:
                vetoes["quality"] += 1
                continue
            used[t1] = used[t2] = True
            replaced[t1] = replaced[t2] = True
            newV.append(mid)
            newT.extend(kids)
            if groups is not None:
                newG.extend([groups[t1], groups[t1],
                             groups[t2], groups[t2]])
            nxt += 1
            n_sweep += 1
        if n_sweep == 0:
            break
        V = np.vstack([V, np.asarray(newV)])
        fx = np.concatenate([fx, np.zeros(len(newV), dtype=bool)])
        keep = ~replaced
        T = np.vstack([T[keep], np.asarray(newT, dtype=np.int64)])
        if groups is not None:
            groups = np.concatenate([groups[keep], np.asarray(newG)])
        total += n_sweep
    return V, T, fx, groups, total


def collapse_short_edges(V, T, target=None, fixed=None, tri_groups=None,
                         low=0.8, high=4.0 / 3.0, autopop_step=None,
                         quality_floor=0.05, selfx_veto=True,
                         max_sweeps=3, tol=1e-7, vetoes=None):
    """Collapse interior manifold edges shorter than ``low * target``
    (Botsch-Kobbelt 2004: 4/5 of the target, the coarsening half of the
    hysteresis bracket) and/or shorter than ``sqrt(2 * autopop_step)``
    (Brakke's Evolver, verpopst.c autopop_detect: under motion by mean
    curvature with time step `scale`, a feature of size L closes in
    time ~ L^2/2, so an edge with L < sqrt(2*scale) is predicted to
    collapse within one step and is removed *before* it degenerates).

    Placement: if one endpoint is protected (fixed, on the boundary, or
    on a non-manifold/triple edge) the other is merged into it and it
    does not move; if both are free the survivor moves to the midpoint;
    if both are protected the collapse is refused.  Static vetoes, each
    refusing (never repairing) the operation:

      * link condition (Dey-Edelsbrunner-Guha-Nekhayev 1999): the
        common vertex neighbours of the endpoints must be exactly the
        two opposite corners, else the collapse pinches the surface;
      * an interior edge between two boundary vertices is refused (it
        would pinch the patch and change the boundary count);
      * valence: an opposite corner may not drop below valence 4
        (interior) / 3 (boundary);
      * label classes (``tri_groups``): refused if the edge's two
        triangles carry different labels (the edge lies ON a label
        seam) or the endpoints touch different label sets (the merge
        would rewire a seam) -- LosTopos-style label safety;
      * hysteresis: no post-collapse edge at the survivor may exceed
        ``high * target`` (prevents split/collapse oscillation);
      * inversion: every surviving triangle of the two 1-rings must
        keep a positive dot with its pre-collapse normal;
      * quality: no surviving triangle of the 1-rings may fall below
        ``min(quality_floor, their pre-collapse minimum)``;
      * static self-intersection (El Topo edgecollapser-style, exact,
        same predicate as the embeddedness gate): with the collapse
        applied, no edge of the modified 1-ring triangles may pierce
        any triangle, and no nearby triangle's edge may pierce a
        modified triangle.  Refused on any hit.

    Returns ``(V2, T2, fixed2, tri_groups2, n_collapses)`` -- NEW,
    compacted arrays (dead vertices dropped, indices remapped)."""
    V = np.asarray(V, dtype=float).copy()
    T = np.asarray(T, dtype=np.int64).copy()
    fx = (np.zeros(len(V), dtype=bool) if fixed is None
          else np.asarray(fixed, dtype=bool).copy())
    groups = None if tri_groups is None else np.asarray(tri_groups).copy()
    if vetoes is None:
        vetoes = _new_vetoes()
    thresh = 0.0
    if target is not None:
        thresh = max(thresh, float(low) * float(target))
    if autopop_step is not None and autopop_step > 0.0:
        thresh = max(thresh, math.sqrt(2.0 * float(autopop_step)))
    if thresh <= 0.0:
        return V, T, fx, groups, 0
    hyst = (float(high) * float(target)) if target is not None else None
    total = 0
    for _ in range(max_sweeps):
        n = len(V)
        uniq, counts, order, starts, tri_of, de = _edge_table(T)
        boundary_v = np.zeros(n, dtype=bool)
        feature_v = np.zeros(n, dtype=bool)
        be = counts == 1
        boundary_v[uniq[be].ravel()] = True
        nm = counts > 2
        feature_v[uniq[nm].ravel()] = True
        protected = fx | boundary_v | feature_v
        # vertex -> incident triangles (list of arrays via argsort)
        vert_of = T.ravel()
        tri_idx = np.repeat(np.arange(len(T)), 3)
        vorder = np.argsort(vert_of, kind='stable')
        vsorted = vert_of[vorder]
        vstarts = np.searchsorted(vsorted, np.arange(n + 1))
        inc = [tri_idx[vorder[vstarts[i]:vstarts[i + 1]]]
               for i in range(n)]
        # vertex -> neighbour sets
        nbr = [set() for _ in range(n)]
        for (aa, bb) in uniq:
            nbr[aa].add(int(bb))
            nbr[bb].add(int(aa))
        valence = np.array([len(s) for s in nbr])
        if groups is not None:
            vgroups = [set() for _ in range(n)]
            for ti_, tri in enumerate(T):
                g = groups[ti_]
                for vv in tri:
                    vgroups[vv].add(g)
        L = np.linalg.norm(V[uniq[:, 0]] - V[uniq[:, 1]], axis=1)
        cand_e = np.nonzero((counts == 2) & (L < thresh))[0]
        if not len(cand_e):
            break
        cand = sorted((float(L[e]), int(e)) for e in cand_e)
        # live tri bboxes for the static intersection prefilter
        if selfx_veto:
            tA, tB, tC = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
            tlo = np.minimum(np.minimum(tA, tB), tC)
            thi = np.maximum(np.maximum(tA, tB), tC)
        locked = np.zeros(n, dtype=bool)
        dead_tri = np.zeros(len(T), dtype=bool)
        alive_v = np.ones(n, dtype=bool)
        n_sweep = 0
        for _, e in cand:
            a, b = int(uniq[e, 0]), int(uniq[e, 1])
            if locked[a] or locked[b]:
                vetoes["locked"] += 1
                continue
            u1 = order[starts[e]]
            u2 = order[starts[e] + 1]
            t1, t2 = int(tri_of[u1]), int(tri_of[u2])
            c = int(np.sum(T[t1]) - a - b)
            d = int(np.sum(T[t2]) - a - b)
            if locked[c] or locked[d]:
                vetoes["locked"] += 1
                continue
            if protected[a] and protected[b]:
                vetoes["pinned"] += 1
                continue
            if boundary_v[a] and boundary_v[b]:
                # interior edge between two rim vertices: pinch
                vetoes["boundary"] += 1
                continue
            # link condition
            common = nbr[a] & nbr[b]
            if common != {c, d}:
                vetoes["link"] += 1
                continue
            if valence[c] < (3 if boundary_v[c] else 4) or \
               valence[d] < (3 if boundary_v[d] else 4):
                vetoes["valence"] += 1
                continue
            if groups is not None:
                if groups[t1] != groups[t2] or vgroups[a] != vgroups[b]:
                    vetoes["label"] += 1
                    continue
            keep, gone = (a, b)
            if protected[b]:
                keep, gone = b, a
            newpos = (V[keep].copy() if protected[keep]
                      else 0.5 * (V[a] + V[b]))
            ring = np.unique(np.concatenate([inc[a], inc[b]]))
            ring = ring[~dead_tri[ring]]
            mod = ring[(ring != t1) & (ring != t2)]
            if not len(mod):
                vetoes["valence"] += 1
                continue
            # simulated modified triangles (gone -> keep, keep moved)
            Tm = T[mod].copy()
            Tm[Tm == gone] = keep
            oldpos = V[keep].copy()
            # hysteresis: new edges at the survivor
            if hyst is not None:
                others = np.unique(Tm[Tm != keep])
                if len(others) and float(np.max(np.linalg.norm(
                        V[others] - newpos, axis=1))) > hyst:
                    vetoes["hysteresis"] += 1
                    continue
            # inversion + quality vetoes over the surviving 1-rings
            n_old, a_old = _tri_normals_areas(V, T[mod])
            q_old = _tri_min_quality(V, [tuple(int(x) for x in t)
                                         for t in T[mod]])
            V[keep] = newpos
            moved_ok = True
            n_new, a_new = _tri_normals_areas(V, Tm)
            scale = math.sqrt(max(float(np.max(a_old)), 1e-300))
            if np.any(a_new < 1e-12 * scale * scale) or \
               np.any(np.einsum('ij,ij->i', n_new, n_old) <= 0.0):
                vetoes["inversion"] += 1
                moved_ok = False
            if moved_ok:
                q_new = _tri_min_quality(V, [tuple(int(x) for x in t)
                                             for t in Tm])
                if q_new < min(quality_floor, q_old):
                    vetoes["quality"] += 1
                    moved_ok = False
            if moved_ok and selfx_veto:
                # exact static intersection test around the op
                rlo = np.minimum(np.min(V[np.unique(Tm)], axis=0),
                                 np.minimum(oldpos, V[gone]))
                rhi = np.maximum(np.max(V[np.unique(Tm)], axis=0),
                                 np.maximum(oldpos, V[gone]))
                near = np.nonzero(~dead_tri
                                  & np.all(tlo <= rhi + tol, axis=1)
                                  & np.all(thi >= rlo - tol, axis=1))[0]
                near = near[(near != t1) & (near != t2)]
                mod_set = set(int(x) for x in mod)
                far = np.array([tt for tt in near
                                if int(tt) not in mod_set],
                               dtype=np.int64)
                # candidate triangle list: far tris as-is + modified
                Tcand = (np.vstack([T[far], Tm]) if len(far)
                         else Tm)
                # edges of the modified region (gone remapped to keep)
                em = np.concatenate([Tm[:, [0, 1]], Tm[:, [1, 2]],
                                     Tm[:, [2, 0]]])
                em = np.unique(np.sort(em, axis=1), axis=0)
                hits = _pierce_pairs(V, em, Tcand, tol=tol)
                if hits == 0 and len(far):
                    ef = np.concatenate([T[far][:, [0, 1]],
                                         T[far][:, [1, 2]],
                                         T[far][:, [2, 0]]])
                    ef = np.unique(np.sort(ef, axis=1), axis=0)
                    # far triangles contain neither endpoint (every
                    # triangle touching a or b is in the ring), so
                    # their edge geometry is unchanged; id-based
                    # shared-vertex exclusion against Tm is exact
                    hits = _pierce_pairs(V, ef, Tm, tol=tol)
                if hits:
                    vetoes["selfx"] += 1
                    moved_ok = False
            if not moved_ok:
                V[keep] = oldpos
                continue
            # commit
            T[mod] = Tm
            dead_tri[t1] = dead_tri[t2] = True
            alive_v[gone] = False
            if selfx_veto:
                mA, mB, mC = V[Tm[:, 0]], V[Tm[:, 1]], V[Tm[:, 2]]
                tlo[mod] = np.minimum(np.minimum(mA, mB), mC)
                thi[mod] = np.maximum(np.maximum(mA, mB), mC)
                tlo[t1] = tlo[t2] = np.inf
                thi[t1] = thi[t2] = -np.inf
            lock_set = (nbr[a] | nbr[b] | {a, b})
            for vv in lock_set:
                locked[vv] = True
            n_sweep += 1
        if n_sweep == 0:
            break
        total += n_sweep
        keepT = ~dead_tri
        T = T[keepT]
        if groups is not None:
            groups = groups[keepT]
        used_v = np.zeros(len(V), dtype=bool)
        used_v[T.ravel()] = True
        remap = -np.ones(len(V), dtype=np.int64)
        remap[used_v] = np.arange(int(np.sum(used_v)))
        V = V[used_v]
        fx = fx[used_v]
        T = remap[T]
    return V, T, fx, groups, total


def nullspace_smooth(V, T, fixed=None, lam=0.5, iters=1,
                     rank_ratio=0.03):
    """El Topo null-space smoothing (Brochu-Bridson 2009,
    meshsmoother.cpp:196-315): per vertex build
    ``A = sum(area_i n_i n_i^T)`` over the incident triangles,
    eigendecompose, and apply only the component of the area-weighted
    Laplacian displacement lying in the small-eigenvalue subspace
    (eigenvalues < rank_ratio * lambda_max; El Topo's
    G_EIGENVALUE_RANK_RATIO = 0.03).  On a smooth patch that subspace
    is the tangent plane; on a ridge or triple line it is the ridge
    direction; at a corner it is empty and the vertex freezes -- the
    surface is redistributed without being moved, with no feature
    tagging.  Modifies V in place."""
    n = len(V)
    T = np.asarray(T, dtype=np.int64)
    for _ in range(iters):
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        a2 = np.linalg.norm(fn, axis=1)
        fa = 0.5 * a2
        nhat = fn / np.maximum(a2, 1e-300)[:, None]
        contrib = fa[:, None, None] * (nhat[:, :, None]
                                       * nhat[:, None, :])
        A = np.zeros((n, 3, 3))
        for k in range(3):
            np.add.at(A, T[:, k], contrib)
        w, Q = np.linalg.eigh(A)          # ascending eigenvalues
        lmax = np.maximum(w[:, 2], 1e-300)
        small = w < rank_ratio * lmax[:, None]
        # area-weighted neighbour average (same stencil as
        # tangential_smooth)
        acc = np.zeros((n, 3))
        wsum = np.zeros(n)
        for k in range(3):
            i = T[:, k]
            j1 = T[:, (k + 1) % 3]
            j2 = T[:, (k + 2) % 3]
            np.add.at(acc, i, fa[:, None] * (V[j1] + V[j2]))
            np.add.at(wsum, i, 2.0 * fa)
        target = acc / np.maximum(wsum, 1e-300)[:, None]
        disp = target - V
        coef = np.einsum('nj,nji->ni', disp, Q)      # components on e_i
        coef = np.where(small, coef, 0.0)
        dproj = np.einsum('ni,nji->nj', coef, Q)
        if fixed is not None:
            dproj[np.asarray(fixed, dtype=bool)] = 0.0
        touched = np.zeros(n, dtype=bool)
        touched[T.ravel()] = True
        dproj[~touched] = 0.0
        V += lam * dproj
    return V


def groom_topo(V, T, fixed=None, tri_groups=None, target=None,
               cycles=1, do_split=True, do_collapse=True, do_flips=True,
               smooth="nullspace", smooth_lam=0.5, smooth_iters=1,
               flip_margin=1e-3, low=0.8, high=4.0 / 3.0,
               autopop_step=None, quality_floor=0.05, selfx_veto=True,
               verify_selfx=False, rank_ratio=0.03):
    """Full Botsch-Kobbelt grooming cycle(s): split long edges,
    collapse short edges, flip to Delaunay, smooth -- the SGP 2004
    incremental-remeshing order, with two substitutions stated up
    front: the flip pass optimises the Delaunay (opposite-angle)
    criterion with an improvement margin rather than Botsch-Kobbelt's
    valence balancing (measured better for the cotan solvers this repo
    runs), and the relaxation pass is El Topo null-space smoothing
    rather than plain tangential relaxation.

    ``target=None`` uses the median edge length at entry.  With
    ``verify_selfx=True`` every cycle is bracketed by an exact global
    crossing count and reverted wholesale if the count rose -- the
    IPC-branch finding is that grooming outside a guard is exactly
    where embeddedness dies, and a refused groom is recoverable while a
    crossed mesh is not.

    Returns ``(V2, T2, fixed2, tri_groups2, info)`` where info carries
    the operation and veto tallies."""
    V = np.asarray(V, dtype=float).copy()
    T = np.asarray(T, dtype=np.int64).copy()
    fx = (np.zeros(len(V), dtype=bool) if fixed is None
          else np.asarray(fixed, dtype=bool).copy())
    groups = None if tri_groups is None else np.asarray(tri_groups).copy()
    if target is None:
        target = _median_edge_length(V, T)
    vetoes = _new_vetoes()
    info = {"target": float(target), "splits": 0, "collapses": 0,
            "flips": 0, "cycles_run": 0, "cycles_reverted": 0,
            "vetoes": vetoes}
    x0 = crossing_count(V, T) if verify_selfx else 0
    for _ in range(int(cycles)):
        snap = (V.copy(), T.copy(), fx.copy(),
                None if groups is None else groups.copy())
        ns = nc = nf = 0
        if do_split:
            V, T, fx, groups, ns = split_long_edges(
                V, T, target, fixed=fx, tri_groups=groups, high=high,
                quality_floor=quality_floor, vetoes=vetoes)
        if do_collapse:
            V, T, fx, groups, nc = collapse_short_edges(
                V, T, target=target, fixed=fx, tri_groups=groups,
                low=low, high=high, autopop_step=autopop_step,
                quality_floor=quality_floor, selfx_veto=selfx_veto,
                vetoes=vetoes)
        if do_flips:
            nf = delaunay_flips(V, T, margin=flip_margin,
                                tri_groups=groups)
        if smooth == "nullspace" and smooth_lam > 0.0:
            nullspace_smooth(V, T, fixed=fx, lam=smooth_lam,
                             iters=smooth_iters, rank_ratio=rank_ratio)
        elif smooth == "tangential" and smooth_lam > 0.0:
            tangential_smooth(V, T, fixed=fx, lam=smooth_lam,
                              iters=smooth_iters)
        if verify_selfx:
            x1 = crossing_count(V, T)
            if x1 > x0:
                V, T, fx, groups = snap
                info["cycles_reverted"] += 1
                break
        info["splits"] += ns
        info["collapses"] += nc
        info["flips"] += nf
        info["cycles_run"] += 1
        if ns == 0 and nc == 0 and nf == 0:
            break
    return V, T, fx, groups, info


def _selftest():
    ok = True

    # A flat non-Delaunay quad: two skinny triangles over the short
    # diagonal.  One flip must make it Delaunay and improve quality.
    V = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                  [2.0, 0.35, 0.0], [2.0, -0.35, 0.0]])
    T = np.array([[0, 1, 2], [1, 0, 3]])
    from_q = _tri_min_quality(V, [tuple(t) for t in T])
    nf = delaunay_flips(V, T)
    to_q = _tri_min_quality(V, [tuple(t) for t in T])
    edges = {tuple(sorted((t[i], t[(i + 1) % 3])))
             for t in T for i in range(3)}
    good = nf == 1 and (2, 3) in edges and to_q > from_q
    ok &= good
    print(f"groom: non-Delaunay quad flipped ({nf} flip, q {from_q:.3f}"
          f" -> {to_q:.3f}) {'OK' if good else 'FAIL'}")

    # Flipping must preserve orientation: total signed area (z) of the
    # flat mesh unchanged in sign for every triangle.
    zs = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])[:, 2]
    good = bool(np.all(zs > 0) or np.all(zs < 0))
    ok &= good
    print(f"groom: orientation preserved after flip "
          f"{'OK' if good else 'FAIL'}")

    # The same non-Delaunay quad with its two triangles in DIFFERENT
    # groups (two films): the flip must be refused, T untouched.
    Vg2 = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                    [2.0, 0.35, 0.0], [2.0, -0.35, 0.0]])
    Tg2 = np.array([[0, 1, 2], [1, 0, 3]])
    T_before = Tg2.copy()
    nf = delaunay_flips(Vg2, Tg2, tri_groups=np.array([0, 1]))
    good = nf == 0 and np.array_equal(Tg2, T_before)
    ok &= good
    print(f"groom: cross-film flip refused under tri_groups ({nf} flips) "
          f"{'OK' if good else 'FAIL'}")

    # A Delaunay mesh must be left alone (hysteresis).
    Vd = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.5, 0.9, 0],
                   [0.5, -0.9, 0]])
    Td = np.array([[0, 1, 2], [1, 0, 3]])
    nf = delaunay_flips(Vd, Td)
    good = nf == 0
    ok &= good
    print(f"groom: Delaunay mesh untouched ({nf} flips) "
          f"{'OK' if good else 'FAIL'}")

    # Tangential smoothing of a bumpy-but-flat-boundary grid: on a
    # PLANE, tangential means in-plane -- z must stay (near) zero and
    # the interior spacing must even out; pinned rim must not move.
    m = 7
    xs, ys = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
    Vg = np.stack([xs.ravel(), ys.ravel(), np.zeros(m * m)], axis=1)
    rng = np.random.default_rng(3)
    interior = np.ones(m * m, dtype=bool)
    idx = np.arange(m * m).reshape(m, m)
    interior[idx[0]] = interior[idx[-1]] = False
    interior[idx[:, 0]] = interior[idx[:, -1]] = False
    Vg[interior, :2] += rng.normal(0.0, 0.04, (int(interior.sum()), 2))
    Tg = []
    for r in range(m - 1):
        for cc in range(m - 1):
            a, b = idx[r, cc], idx[r, cc + 1]
            c2, d2 = idx[r + 1, cc + 1], idx[r + 1, cc]
            Tg.append([a, b, c2])
            Tg.append([a, c2, d2])
    Tg = np.array(Tg)
    rim0 = Vg[~interior].copy()
    L0 = None
    for arr in (None,):
        E = np.unique(np.sort(np.concatenate(
            [Tg[:, [0, 1]], Tg[:, [1, 2]], Tg[:, [2, 0]]]), axis=1), axis=0)
        L0 = np.linalg.norm(Vg[E[:, 0]] - Vg[E[:, 1]], axis=1)
    cv0 = float(np.std(L0) / np.mean(L0))
    tangential_smooth(Vg, Tg, fixed=~interior, lam=0.5, iters=10)
    L1 = np.linalg.norm(Vg[E[:, 0]] - Vg[E[:, 1]], axis=1)
    cv1 = float(np.std(L1) / np.mean(L1))
    rim_moved = float(np.max(np.abs(Vg[~interior] - rim0)))
    zmax = float(np.max(np.abs(Vg[:, 2])))
    good = cv1 < cv0 and rim_moved == 0.0 and zmax < 1e-9
    ok &= good
    print(f"groom: tangential smoothing evens spacing (cv {cv0:.4f} -> "
          f"{cv1:.4f}), rim pinned, z drift {zmax:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # ---- topology-changing layer -------------------------------------

    def _chi(Vx, Tx):
        Ex = np.unique(np.sort(np.concatenate(
            [Tx[:, [0, 1]], Tx[:, [1, 2]], Tx[:, [2, 0]]]), axis=1),
            axis=0)
        used = len(np.unique(Tx.ravel()))
        return used - len(Ex) + len(Tx)

    # A 3x3 unit grid: every interior edge is "long" for target=0.6 and
    # must split at its midpoint; boundary edges are never split; the
    # disk Euler characteristic survives; the surface point set is
    # unchanged so the mesh stays embedded.
    m3 = 3
    idx3 = np.arange(m3 * m3).reshape(m3, m3)
    xs3, ys3 = np.meshgrid(np.arange(m3, dtype=float),
                           np.arange(m3, dtype=float))
    Vsp = np.stack([xs3.ravel(), ys3.ravel(), np.zeros(m3 * m3)], axis=1)
    Tsp = []
    for r in range(m3 - 1):
        for cc in range(m3 - 1):
            a3, b3 = idx3[r, cc], idx3[r, cc + 1]
            c3, d3 = idx3[r + 1, cc + 1], idx3[r + 1, cc]
            Tsp.append([a3, b3, c3])
            Tsp.append([a3, c3, d3])
    Tsp = np.array(Tsp, dtype=np.int64)
    chi0 = _chi(Vsp, Tsp)
    V2, T2, fx2, _, nsp = split_long_edges(Vsp, Tsp, target=0.6)
    zs2 = np.cross(V2[T2[:, 1]] - V2[T2[:, 0]],
                   V2[T2[:, 2]] - V2[T2[:, 0]])[:, 2]
    good = (nsp > 0 and _chi(V2, T2) == chi0
            and crossing_count(V2, T2) == 0
            and bool(np.all(zs2 > 0)) and not np.any(fx2[len(Vsp):]))
    ok &= good
    print(f"groom: split_long_edges {nsp} splits, chi {chi0} preserved, "
          f"orientation kept, midpoints free {'OK' if good else 'FAIL'}")

    # A pinned-rim edge (both endpoints fixed) must never split.
    Vp = np.array([[0.0, 0, 0], [2.0, 0, 0], [2.0, 2, 0], [0.0, 2, 0]])
    Tp = np.array([[0, 1, 2], [0, 2, 3]])
    vt = _new_vetoes()
    _, Tq, _, _, nsp2 = split_long_edges(
        Vp, Tp, target=0.5, fixed=np.ones(4, dtype=bool), vetoes=vt)
    good = nsp2 == 0 and vt["pinned"] > 0 and np.array_equal(Tq, Tp)
    ok &= good
    print(f"groom: split of all-pinned mesh refused "
          f"(pinned veto x{vt['pinned']}) {'OK' if good else 'FAIL'}")

    # Collapse: a planar 4-spoke fan with one short interior edge.  The
    # collapse must merge the free vertex INTO the protected (boundary)
    # one, which must not move; chi preserved; counts drop by (1 vert,
    # 2 tris); the autopop threshold sqrt(2*step) decides candidacy.
    def _fan5():
        Vf = np.array([[0.0, 0, 0],      # 0 = a (interior, free)
                       [0.4, 0, 0],      # 1 = b (boundary)
                       [0.0, 0.5, 0],    # 2 = c
                       [0.0, -0.5, 0],   # 3 = d
                       [-0.4, 0, 0]])    # 4 = e
        Tf = np.array([[0, 1, 2], [0, 2, 4], [0, 4, 3], [0, 3, 1]],
                      dtype=np.int64)
        return Vf, Tf

    Vf, Tf = _fan5()
    V2, T2, fx2, _, nc = collapse_short_edges(Vf, Tf,
                                              autopop_step=0.005)
    good = nc == 0 and len(V2) == 5
    ok &= good
    print(f"groom: autopop step=0.005 (thresh {math.sqrt(0.01):.2f}) "
          f"collapses nothing {'OK' if good else 'FAIL'}")

    Vf, Tf = _fan5()
    bpos = Vf[1].copy()
    V2, T2, fx2, _, nc = collapse_short_edges(Vf, Tf,
                                              autopop_step=0.101)
    good = (nc == 1 and len(V2) == 4 and len(T2) == 2
            and _chi(V2, T2) == 1
            and any(np.allclose(v, bpos) for v in V2)
            and crossing_count(V2, T2) == 0)
    ok &= good
    print(f"groom: autopop step=0.101 (thresh {math.sqrt(0.202):.2f}) "
          f"collapses the short edge, survivor unmoved "
          f"{'OK' if good else 'FAIL'}")

    # Both endpoints protected -> pinned veto.
    Vf, Tf = _fan5()
    vt = _new_vetoes()
    fixboth = np.zeros(5, dtype=bool)
    fixboth[0] = True                    # a fixed; b is boundary
    V2, T2, _, _, nc = collapse_short_edges(Vf, Tf, autopop_step=0.101,
                                            fixed=fixboth, vetoes=vt)
    good = nc == 0 and vt["pinned"] > 0
    ok &= good
    print(f"groom: collapse with both endpoints protected refused "
          f"(pinned veto x{vt['pinned']}) {'OK' if good else 'FAIL'}")

    # Label-class veto: the same fan with the edge's two triangles in
    # different label groups (edge ON a seam) must refuse.
    Vf, Tf = _fan5()
    vt = _new_vetoes()
    V2, T2, _, g2, nc = collapse_short_edges(
        Vf, Tf, autopop_step=0.101,
        tri_groups=np.array([0, 0, 1, 1]), vetoes=vt)
    good = nc == 0 and vt["label"] > 0
    ok &= good
    print(f"groom: collapse across a label seam refused "
          f"(label veto x{vt['label']}) {'OK' if good else 'FAIL'}")

    # Link-condition veto: endpoints share THREE neighbours (2, 3, 4)
    # but the edge's opposite corners are only {2, 3} -- collapsing
    # would pinch the surface.
    Vl = np.array([[0.2, 0, 0], [0.8, 0, 0], [0.5, 0.8, 0],
                   [0.5, -0.8, 0], [0.5, -1.6, 0],
                   [-0.5, 0.5, 0], [-0.5, -0.5, 0]])
    Tl = np.array([[0, 1, 2], [1, 0, 3], [0, 4, 3], [4, 1, 3],
                   [0, 2, 5], [0, 5, 6], [0, 6, 4]],
                  dtype=np.int64)
    vt = _new_vetoes()
    V2, T2, _, _, nc = collapse_short_edges(Vl, Tl, autopop_step=0.245,
                                            vetoes=vt)
    good = nc == 0 and vt["link"] > 0
    ok &= good
    print(f"groom: pinching collapse refused (link veto x{vt['link']}) "
          f"{'OK' if good else 'FAIL'}")

    # Inversion veto: a folded 5-spoke fan where merging the free
    # vertex into the survivor drags one triangle across its own plane.
    Vi = np.array([[0.05, 0, 0],         # 0 = a (interior)
                   [-0.5, 0, 0],         # 1 = b (boundary survivor)
                   [0.0, 1.0, 0],        # 2 = p0
                   [-0.4, 0.8, 0],       # 3 = c
                   [-0.4, -0.8, 0],      # 4 = d
                   [0.0, -1.0, 0]])      # 5 = p1
    Ti = np.array([[0, 2, 3], [0, 3, 1], [0, 1, 4], [0, 4, 5],
                   [0, 5, 2]], dtype=np.int64)
    vt = _new_vetoes()
    V2, T2, _, _, nc = collapse_short_edges(Vi, Ti, autopop_step=0.18,
                                            vetoes=vt)
    good = nc == 0 and vt["inversion"] > 0
    ok &= good
    print(f"groom: inverting collapse refused "
          f"(inversion veto x{vt['inversion']}) {'OK' if good else 'FAIL'}")

    # Sliver veto: same combinatorics, flap placed so the collapse
    # keeps orientation but squashes the flap below the quality floor.
    Vq = np.array([[0.05, 0, 0],         # 0 = a
                   [-0.5, 0, 0],         # 1 = b
                   [0.75, 0.012, 0],     # 2 = p0
                   [-0.4, 0.8, 0],       # 3 = c
                   [-0.4, -0.8, 0],      # 4 = d
                   [0.75, -0.012, 0]])   # 5 = p1
    Tq = np.array([[0, 2, 3], [0, 3, 1], [0, 1, 4], [0, 4, 5],
                   [0, 5, 2]], dtype=np.int64)
    vt = _new_vetoes()
    V2, T2, _, _, nc = collapse_short_edges(Vq, Tq, autopop_step=0.18,
                                            vetoes=vt)
    good = nc == 0 and vt["quality"] > 0
    ok &= good
    print(f"groom: sliver-creating collapse refused "
          f"(quality veto x{vt['quality']}) {'OK' if good else 'FAIL'}")

    # Static self-intersection veto: a bent fan whose collapse sweeps a
    # new edge through a free-floating blade triangle.  With the veto
    # disabled the collapse happens and the exact crossing counter
    # catches it (proving the configuration is genuine); with the veto
    # on it must be refused with the mesh untouched.
    def _blade():
        Vx = np.array([[0.0, 0, 0],          # 0 = a
                       [0.4, 0, 0],          # 1 = b (survivor)
                       [0.0, 0.5, 0],        # 2 = c
                       [0.0, -0.5, 0],       # 3 = d
                       [-0.4, 0, 0.3],       # 4 = e (lifted)
                       [0.05, 0.04, 0.10],   # 5 blade
                       [0.05, -0.04, 0.10],  # 6 blade
                       [0.05, 0.0, 0.18]])   # 7 blade
        Tx = np.array([[0, 1, 2], [0, 2, 4], [0, 4, 3], [0, 3, 1],
                       [5, 6, 7]], dtype=np.int64)
        return Vx, Tx

    Vx, Tx = _blade()
    x_before = crossing_count(Vx, Tx)
    V2, T2, _, _, nc_off = collapse_short_edges(Vx, Tx,
                                                autopop_step=0.101,
                                                selfx_veto=False)
    x_unsafe = crossing_count(V2, T2)
    Vx, Tx = _blade()
    vt = _new_vetoes()
    V3, T3, _, _, nc_on = collapse_short_edges(Vx, Tx,
                                               autopop_step=0.101,
                                               vetoes=vt)
    good = (x_before == 0 and nc_off == 1 and x_unsafe > 0
            and nc_on == 0 and vt["selfx"] > 0
            and np.array_equal(T3, Tx) and np.allclose(V3, Vx))
    ok &= good
    print(f"groom: crossing-creating collapse (unsafe run: {x_unsafe} "
          f"crossings) refused by the static test "
          f"(selfx veto x{vt['selfx']}) {'OK' if good else 'FAIL'}")

    # Hysteresis veto: the collapse would hand the survivor an edge
    # longer than 4/3 of the target.
    Vh = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.0, 0.5, 0],
                   [0.0, -0.5, 0], [-2.0, 0, 0]])
    Th = np.array([[0, 1, 2], [0, 2, 4], [0, 4, 3], [0, 3, 1]],
                  dtype=np.int64)
    vt = _new_vetoes()
    V2, T2, _, _, nc = collapse_short_edges(Vh, Th, target=0.15,
                                            vetoes=vt)
    good = nc == 0 and vt["hysteresis"] > 0
    ok &= good
    print(f"groom: over-long-result collapse refused "
          f"(hysteresis veto x{vt['hysteresis']}) "
          f"{'OK' if good else 'FAIL'}")

    # Null-space smoothing on a tent: ridge vertices may move only
    # along the ridge (their motion across it or off the surface is in
    # the large-eigenvalue subspace); flat-flank vertices stay exactly
    # in their plane; the pinned rim does not move.
    mt = 7
    xst, yst = np.meshgrid(np.linspace(0.0, 4.0, mt),
                           np.linspace(0.0, 4.0, mt))
    zt = 1.0 - 0.5 * np.abs(xst - 2.0)
    Vt = np.stack([xst.ravel(), yst.ravel(), zt.ravel()], axis=1)
    idxt = np.arange(mt * mt).reshape(mt, mt)
    Tt = []
    for r in range(mt - 1):
        for cc in range(mt - 1):
            a4, b4 = idxt[r, cc], idxt[r, cc + 1]
            c4, d4 = idxt[r + 1, cc + 1], idxt[r + 1, cc]
            Tt.append([a4, b4, c4])
            Tt.append([a4, c4, d4])
    Tt = np.array(Tt, dtype=np.int64)
    rim = np.zeros(mt * mt, dtype=bool)
    rim[idxt[0]] = rim[idxt[-1]] = True
    rim[idxt[:, 0]] = rim[idxt[:, -1]] = True
    # nudge interior vertices along the surface so there is something
    # to smooth (y-shifts keep every vertex exactly on the tent)
    rngt = np.random.default_rng(11)
    Vt0 = Vt.copy()
    for i in range(mt * mt):
        if not rim[i]:
            Vt[i, 1] += rngt.normal(0.0, 0.08)
    Vpre = Vt.copy()
    nullspace_smooth(Vt, Tt, fixed=rim, lam=0.7, iters=4)
    ridge = (~rim) & (np.abs(Vpre[:, 0] - 2.0) < 1e-9)
    flank = (~rim) & ~ridge
    ridge_dx = float(np.max(np.abs(Vt[ridge, 0] - Vpre[ridge, 0])))
    ridge_dz = float(np.max(np.abs(Vt[ridge, 2] - Vpre[ridge, 2])))
    ridge_dy = float(np.max(np.abs(Vt[ridge, 1] - Vpre[ridge, 1])))
    plane_resid = float(np.max(np.abs(
        Vt[flank, 2] - (1.0 - 0.5 * np.abs(Vt[flank, 0] - 2.0)))))
    rim_moved = float(np.max(np.abs(Vt[rim] - Vpre[rim])))
    good = (ridge_dx < 1e-9 and ridge_dz < 1e-9 and ridge_dy > 1e-4
            and plane_resid < 1e-9 and rim_moved == 0.0)
    ok &= good
    print(f"groom: null-space smoothing moves ridge only along ridge "
          f"(dx {ridge_dx:.1e}, dz {ridge_dz:.1e}, dy {ridge_dy:.1e}), "
          f"flanks stay planar ({plane_resid:.1e}), rim pinned "
          f"{'OK' if good else 'FAIL'}")

    # groom_topo end-to-end on a graded noisy disk: uniformity (edge
    # CV) must improve, the disk chi and embeddedness must survive, the
    # pinned rim must not move, and the cycle verifier must report no
    # reverts.
    mg = 9
    tg = np.linspace(0.0, 1.0, mg) ** 1.7
    xsg, ysg = np.meshgrid(tg, np.linspace(0.0, 1.0, mg))
    Vg2 = np.stack([xsg.ravel(), ysg.ravel(), np.zeros(mg * mg)], axis=1)
    idxg = np.arange(mg * mg).reshape(mg, mg)
    Tg2 = []
    for r in range(mg - 1):
        for cc in range(mg - 1):
            a5, b5 = idxg[r, cc], idxg[r, cc + 1]
            c5, d5 = idxg[r + 1, cc + 1], idxg[r + 1, cc]
            Tg2.append([a5, b5, c5])
            Tg2.append([a5, c5, d5])
    Tg2 = np.array(Tg2, dtype=np.int64)
    rimg = np.zeros(mg * mg, dtype=bool)
    rimg[idxg[0]] = rimg[idxg[-1]] = True
    rimg[idxg[:, 0]] = rimg[idxg[:, -1]] = True
    rim_xy = Vg2[rimg].copy()

    def _cv(Vx, Tx):
        Ex = np.unique(np.sort(np.concatenate(
            [Tx[:, [0, 1]], Tx[:, [1, 2]], Tx[:, [2, 0]]]), axis=1),
            axis=0)
        Lx = np.linalg.norm(Vx[Ex[:, 0]] - Vx[Ex[:, 1]], axis=1)
        return float(np.std(Lx) / np.mean(Lx))

    cv0 = _cv(Vg2, Tg2)
    V2, T2, fx2, _, info = groom_topo(Vg2, Tg2, fixed=rimg, cycles=4,
                                      verify_selfx=True)
    cv1 = _cv(V2, T2)
    zs2 = np.cross(V2[T2[:, 1]] - V2[T2[:, 0]],
                   V2[T2[:, 2]] - V2[T2[:, 0]])[:, 2]
    rim2 = V2[fx2]
    rim_ok = (len(rim2) == len(rim_xy)
              and np.allclose(np.sort(rim2, axis=0),
                              np.sort(rim_xy, axis=0)))
    good = (cv1 < cv0 and _chi(V2, T2) == 1
            and crossing_count(V2, T2) == 0
            and bool(np.all(zs2 > 0)) and rim_ok
            and info["splits"] > 0 and info["collapses"] > 0
            and info["cycles_reverted"] == 0)
    ok &= good
    print(f"groom: groom_topo on graded disk cv {cv0:.3f} -> {cv1:.3f} "
          f"({info['splits']} splits, {info['collapses']} collapses, "
          f"{info['flips']} flips, vetoes {sum(info['vetoes'].values())}),"
          f" chi/rim/embeddedness kept {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("solver.groom self-test failed")
