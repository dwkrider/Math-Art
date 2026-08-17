# In-loop mesh grooming: Delaunay edge flips + tangential smoothing.
#
# Part of the shared solver core (`math_art/solver/`).  NumPy only.
#
# A relaxation that never touches its mesh degrades it: triangles
# collapse into slivers, cotangent weights go wild, and the solver
# either folds the surface or has to be capped.  Surface Evolver's
# practical superiority comes from continuously grooming the mesh
# between gradient steps (`u` equiangulate, `V` vertex average).  This
# module implements the two cheapest, safest of those operations for a
# fixed vertex count (no splits or collapses -- those need the full
# safety-veto machinery and are deliberately deferred):
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
# References:
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992) -- equiangulation with hysteresis, tangential vertex
#       averaging (trirevis.c, veravg.c).
#   M. Botsch and L. Kobbelt, "A Remeshing Approach to Multiresolution
#       Modeling", Symposium on Geometry Processing (2004) -- the
#       remesh-during-deformation loop this is a subset of.
#   T. Brochu and R. Bridson, "Robust Topological Operations for
#       Dynamic Explicit Surfaces", SIAM J. Sci. Comput. 31(4) (2009)
#       -- static safety vetoes on improvement operations.

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

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("solver.groom self-test failed")
