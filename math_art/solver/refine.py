# refine -- midpoint 1->4 mesh refinement for coarse-to-fine
# continuation (S5 of research/geometric-solver-survey.md).
#
# The continuation pattern: converge a solve on a COARSE mesh, refine
# every triangle 1->4 by edge midpoints (projecting the new vertices
# onto whatever constraints apply -- walls, level sets, a target
# metric), rescale the resolution-dependent thresholds, and re-solve.
# Global reorganization then happens at coarse cost, and -- the point
# for buckling-unstable problems -- short-wavelength modes simply do
# not exist on the coarse mesh, so the coarse solve selects the
# long-wavelength shape and the fine solve only adds detail.  This is
# the standard "nested iteration" of multigrid, and Surface Evolver's
# canonical converge-then-`r`efine workflow.
#
# NumPy only -- no bpy, no SciPy -- so it imports and self-tests
# headlessly like the rest of `math_art/solver/`.
#
# Threshold rescaling convention under one 1->4 split (documented here
# once; callers own their thresholds): edge lengths halve, so
# `min_len -> min_len/2`, areas quarter (`min_area -> min_area/4`),
# and explicit step-size caps tied to the edge scale halve.
#
# References:
# - W. L. Briggs, V. E. Henson, S. F. McCormick, "A Multigrid
#   Tutorial", 2nd ed., SIAM, 2000 (nested iteration / full multigrid).
# - K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2),
#   1992, pp. 141-165 (converge-then-refine workflow; midpoint
#   refinement with constraint projection).
# - M. Botsch, L. Kobbelt, M. Pauly, P. Alliez, B. Levy, "Polygon Mesh
#   Processing", A K Peters, 2010, ch. 6 (uniform 1-to-4 subdivision).

import numpy as np


def subdivide(V, T, project=None):
    """One uniform 1->4 midpoint subdivision of a triangle mesh.

    V : (n, 3) float vertex positions
    T : (m, 3) int triangle indices
    project : optional callable ``project(P_new, parents) -> P_new``
        applied to the newly created midpoint positions only, so the
        caller can put new vertices exactly on their constraints (wall
        level sets, a boundary curve, a target-metric chart, ...).

    Returns ``(V2, T2, parents)``:

    * ``V2`` -- (n + n_e, 3): the original vertices first (indices
      unchanged), then one midpoint per unique undirected edge.
    * ``T2`` -- (4m, 3): each input triangle (a, b, c) with midpoints
      mab, mbc, mca becomes (a, mab, mca), (b, mbc, mab),
      (c, mca, mbc), (mab, mbc, mca) -- orientation preserved.
    * ``parents`` -- (n_e, 2) int: the endpoints of the edge each new
      vertex bisects; ``V2[n + k]`` is the midpoint of
      ``V[parents[k, 0]]`` and ``V[parents[k, 1]]``.  Use it to
      interpolate any per-vertex attribute (see ``interp``) or to
      propagate pin masks (a midpoint is pinned iff both parents are).
    """
    V = np.asarray(V, dtype=float)
    T = np.asarray(T, dtype=np.int64)
    n = len(V)
    # unique undirected edges, and per-triangle edge ids
    ea = T[:, [0, 1, 2]].ravel()
    eb = T[:, [1, 2, 0]].ravel()
    lo = np.minimum(ea, eb)
    hi = np.maximum(ea, eb)
    key = lo * np.int64(n) + hi
    ukey, inv = np.unique(key, return_inverse=True)
    parents = np.stack([ukey // n, ukey % n], axis=1)
    mid = 0.5 * (V[parents[:, 0]] + V[parents[:, 1]])
    if project is not None:
        mid = np.asarray(project(mid, parents), dtype=float)
        if mid.shape != (len(parents), 3):
            raise ValueError("project() must return the (n_e, 3) "
                             "midpoint array it was given")
    V2 = np.concatenate([V, mid], axis=0)
    m = len(T)
    # inv follows the row-major ravel above: triangle t's edges (a,b),
    # (b,c), (c,a) sit at flat positions 3t, 3t+1, 3t+2
    eid = inv.reshape(m, 3)
    mab = n + eid[:, 0]
    mbc = n + eid[:, 1]
    mca = n + eid[:, 2]
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    T2 = np.concatenate([
        np.stack([a, mab, mca], axis=1),
        np.stack([b, mbc, mab], axis=1),
        np.stack([c, mca, mbc], axis=1),
        np.stack([mab, mbc, mca], axis=1),
    ], axis=0)
    return V2, T2, parents


def interp(A, parents):
    """Midpoint-interpolate a per-vertex attribute array ``A`` (any
    trailing shape) to the refined mesh: returns the (n + n_e, ...)
    array whose first n rows are ``A`` and whose new rows are the
    parent-pair averages."""
    A = np.asarray(A)
    return np.concatenate([A, 0.5 * (A[parents[:, 0]] + A[parents[:, 1]])],
                          axis=0)


def edges_of(T):
    """Unique undirected edge array (n_e, 2) of a triangle mesh."""
    T = np.asarray(T, dtype=np.int64)
    n = int(T.max()) + 1
    ea = T[:, [0, 1, 2]].ravel()
    eb = T[:, [1, 2, 0]].ravel()
    lo = np.minimum(ea, eb)
    hi = np.maximum(ea, eb)
    ukey = np.unique(lo * np.int64(n) + hi)
    return np.stack([ukey // n, ukey % n], axis=1)


def _boundary_edge_count(T):
    T = np.asarray(T, dtype=np.int64)
    n = int(T.max()) + 1
    ea = T[:, [0, 1, 2]].ravel()
    eb = T[:, [1, 2, 0]].ravel()
    key = np.minimum(ea, eb) * np.int64(n) + np.maximum(ea, eb)
    _u, cnt = np.unique(key, return_counts=True)
    return int(np.sum(cnt == 1))


def _selftest():
    rng = np.random.default_rng(7)

    # --- disk-topology test mesh: perturbed triangulated grid -------
    k = 6
    xs, ys = np.meshgrid(np.linspace(0, 1, k), np.linspace(0, 1, k))
    V = np.stack([xs.ravel(), ys.ravel(),
                  0.1 * np.sin(3 * xs.ravel()) * np.cos(2 * ys.ravel())],
                 axis=1)
    V += 0.01 * rng.normal(size=V.shape)
    T = []
    for i in range(k - 1):
        for j in range(k - 1):
            a = i * k + j
            T.append((a, a + 1, a + k))
            T.append((a + 1, a + k + 1, a + k))
    T = np.array(T, dtype=np.int64)

    V2, T2, parents = subdivide(V, T)

    # face count x4, original vertices bitwise untouched
    assert len(T2) == 4 * len(T)
    assert np.array_equal(V2[:len(V)], V)

    # Euler characteristic preserved (disk: chi = 1)
    def chi(Vn, Tn):
        n_e = len(edges_of(Tn))
        return Vn - n_e + len(Tn)
    assert chi(len(V), T) == 1 and chi(len(V2), T2) == 1, "chi changed"

    # boundary edge count doubles under midpoint split
    assert _boundary_edge_count(T2) == 2 * _boundary_edge_count(T)

    # orientation preserved: summed signed normals agree
    def area_vec(Vn, Tn):
        return np.cross(Vn[Tn[:, 1]] - Vn[Tn[:, 0]],
                        Vn[Tn[:, 2]] - Vn[Tn[:, 0]]).sum(axis=0)
    nv0, nv1 = area_vec(V, T), area_vec(V2, T2)
    assert float(nv0 @ nv1) > 0.9 * float(nv0 @ nv0), "orientation flip"

    # mean edge length reduced ~2x (exactly 2x on each bisected edge;
    # interior midpoint triangles keep it within a few percent overall)
    def mean_len(Vn, Tn):
        E = edges_of(Tn)
        return float(np.mean(np.linalg.norm(Vn[E[:, 0]] - Vn[E[:, 1]],
                                            axis=1)))
    r = mean_len(V, T) / mean_len(V2, T2)
    assert 1.8 < r < 2.2, f"edge length ratio {r}"

    # per-vertex attribute interpolation matches position midpoints
    A2 = interp(V, parents)
    assert np.allclose(A2, np.concatenate(
        [V, 0.5 * (V[parents[:, 0]] + V[parents[:, 1]])]), atol=0.0)

    # --- constraint projection: sphere mesh stays on the sphere -----
    # octahedron -> subdivide with radial projection; every vertex must
    # sit on the unit sphere exactly (constrained vertices stay on
    # their constraint)
    So = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                   [0, 0, 1], [0, 0, -1]], dtype=float)
    To = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                   [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]],
                  dtype=np.int64)

    def _proj(P, par):
        return P / np.linalg.norm(P, axis=1, keepdims=True)

    Vs, Ts = So, To
    for _ in range(3):
        Vs, Ts, par = subdivide(Vs, Ts, project=_proj)
    rad = np.linalg.norm(Vs, axis=1)
    assert float(np.abs(rad - 1.0).max()) < 1e-12, "left the constraint"
    assert chi(len(Vs), Ts) == 2, "sphere chi changed"
    # closed mesh stays closed
    assert _boundary_edge_count(Ts) == 0

    # pin-mask propagation convention: midpoint pinned iff both parents
    pin = np.zeros(len(V), dtype=bool)
    pin[:k] = True                       # one boundary row pinned
    pin2 = pin[parents[:, 0]] & pin[parents[:, 1]]
    # every pinned midpoint's parents are adjacent along the pinned row
    assert pin2.sum() == k - 1

    print(f"refine: grid V={len(V)}->{len(V2)} T={len(T)}->{len(T2)} "
          f"len_ratio={r:.3f} sphere_r_max_dev="
          f"{float(np.abs(rad - 1.0).max()):.2e} RESULT: OK")


# NOTE: no __main__ guard -- tests/test_selftests.py discovers and runs
# _selftest() headlessly (see CLAUDE.md).
