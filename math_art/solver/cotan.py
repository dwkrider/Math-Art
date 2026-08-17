# Cotangent-Laplacian weights, with intrinsic mollification.
#
# Part of the shared solver core (`math_art/solver/`).  NumPy only.
#
# The cotangent Laplacian is the discretization behind every fairing and
# Plateau solve in this project (Pinkall-Polthier).  Two of its consumers
# historically clamped the cotangents to keep their linear solves tame
# (`minsurf/plateau.py` clipped to [0.01, 20]; `seifert/fair.py` floored
# at 1e-3) -- but a clamp fires on EVERY obtuse corner (~20-25% of the
# corners of a typical relaxed mesh, measured in tests/bench), and each
# firing biases the solve toward the uniform-weight (Tutte) solution.
#
# Intrinsic mollification (Sharp and Crane) is the principled fix: work
# from edge LENGTHS, add the smallest uniform epsilon that makes every
# triangle inequality hold strictly, and build the cotangents from the
# mollified lengths.  On a non-degenerate mesh epsilon is zero and the
# result is exactly the raw cotangent weight, negative values included
# (a negative cotangent on an obtuse corner is correct geometry, and the
# assembled Laplacian is positive semi-definite regardless of the signs,
# because it is the Hessian of the Dirichlet energy).  Only genuinely
# degenerate triangles are touched.
#
# References:
#   U. Pinkall and K. Polthier, "Computing discrete minimal surfaces and
#       their conjugates", Experimental Mathematics 2(1) (1993).
#   N. Sharp and K. Crane, "A Laplacian for Nonmanifold Triangle
#       Meshes", Computer Graphics Forum 39(5) (SGP) (2020) -- intrinsic
#       mollification.
#   A. I. Bobenko and B. A. Springborn, "A discrete Laplace-Beltrami
#       operator for simplicial surfaces", Discrete and Computational
#       Geometry 38(4) (2007) -- the intrinsic (length-based) viewpoint.

import numpy as np


def mollified_lengths(V, T, delta_rel=1e-6):
    """Per-triangle edge lengths (l0, l1, l2), with l_k opposite corner
    k, mollified: the smallest uniform epsilon is added to every length
    so that every triangle satisfies l_i + l_j - l_k >= delta strictly,
    with delta = delta_rel * mean edge length (Sharp-Crane).  Returns
    (L, eps) where L is (ntri, 3)."""
    P0, P1, P2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    L = np.stack([
        np.linalg.norm(P1 - P2, axis=1),
        np.linalg.norm(P2 - P0, axis=1),
        np.linalg.norm(P0 - P1, axis=1),
    ], axis=1)
    delta = delta_rel * float(np.mean(L))
    # deficit of each triangle inequality: l_i + l_j - l_k (k the third)
    worst = 0.0
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        deficit = delta - (L[:, i] + L[:, j] - L[:, k])
        if len(deficit):
            worst = max(worst, float(np.max(deficit)))
    eps = worst  # adding eps to all three lengths adds +eps to each lhs
    if eps > 0.0:
        L = L + eps
    return L, eps


def corner_cotans_from_lengths(L):
    """Corner cotangents (ntri, 3) from per-triangle edge lengths, corner
    k opposite length l_k: cot_k = (l_i^2 + l_j^2 - l_k^2) / (4 A), with
    A from Heron's formula on the (mollified) lengths."""
    a, b, c = L[:, 0], L[:, 1], L[:, 2]
    s = 0.5 * (a + b + c)
    A2 = np.maximum(s * (s - a) * (s - b) * (s - c), 1e-300)
    A = np.sqrt(A2)
    cots = np.empty_like(L)
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        cots[:, k] = (L[:, i] ** 2 + L[:, j] ** 2 - L[:, k] ** 2) / (4.0 * A)
    return cots


def edge_cotan_weights(V, T, mode="raw", clamp_lo=0.01, clamp_hi=20.0,
                       delta_rel=1e-6):
    """Half-edge cotangent weights for the meshes' cotan Laplacian.

    Returns (E, W): E (3*ntri, 2) directed vertex-index pairs (the edge
    opposite each corner, in corner order 0,1,2) and W the weight
    0.5*cot(angle at the opposite corner).  An interior edge appears
    twice (once per adjacent triangle); consumers accumulate with
    np.add.at exactly as before.

    mode = "raw"      raw cotangents (no clamp; PSD by construction)
           "mollify"  cotangents from mollified intrinsic lengths --
                      identical to "raw" away from degenerate triangles
           "clamp"    np.clip(cot, clamp_lo, clamp_hi) -- the historical
                      plateau.py behaviour, kept for A/B comparison
    """
    if mode == "mollify":
        L, _eps = mollified_lengths(V, T, delta_rel)
        cots = corner_cotans_from_lengths(L)
    else:
        P = [V[T[:, k]] for k in range(3)]
        cots = np.empty((len(T), 3))
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            u = P[i] - P[k]
            v = P[j] - P[k]
            cr = np.linalg.norm(np.cross(u, v), axis=1)
            cots[:, k] = (np.einsum('ij,ij->i', u, v)
                          / np.maximum(cr, 1e-12))
        if mode == "clamp":
            cots = np.clip(cots, clamp_lo, clamp_hi)
        elif mode != "raw":
            raise ValueError(f"unknown cotan mode {mode!r}")
    # block order matches the historical plateau assembly -- edge (a, b)
    # with the cotangent at apex c for (a,b,c) in (0,1,2),(1,2,0),(2,0,1)
    # -- so the baseline path stays bitwise identical under np.add.at
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    W = 0.5 * np.concatenate([cots[:, 2], cots[:, 0], cots[:, 1]])
    return E, W


def _selftest():
    ok = True

    # On the unit right triangle the corner cotans are (1, 0-ish, 1)...
    # use an equilateral instead where all three are 1/sqrt(3), and a
    # right isoceles where they are (0, 1, 1).
    Veq = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
    Teq = np.array([[0, 1, 2]])
    L, eps = mollified_lengths(Veq, Teq)
    c = corner_cotans_from_lengths(L)
    good = eps == 0.0 and np.allclose(c, 1 / np.sqrt(3), atol=1e-12)
    ok &= good
    print(f"cotan: equilateral cotans {c.ravel()} eps={eps} "
          f"{'OK' if good else 'FAIL'}")

    # Raw mode must equal the classical 3-D formula, and mollify must
    # equal raw on a healthy mesh.
    rng = np.random.default_rng(7)
    V = rng.normal(size=(6, 3))
    T = np.array([[0, 1, 2], [2, 1, 3], [3, 4, 5]])
    E1, W1 = edge_cotan_weights(V, T, mode="raw")
    E2, W2 = edge_cotan_weights(V, T, mode="mollify")
    good = np.array_equal(E1, E2) and np.allclose(W1, W2, rtol=1e-9)
    ok &= good
    print(f"cotan: mollify == raw on non-degenerate mesh "
          f"(max diff {np.max(np.abs(W1 - W2)):.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # A degenerate (collinear) triangle must yield FINITE weights in
    # mollify mode -- that is the whole point.
    Vd = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    Td = np.array([[0, 1, 2]])
    _, Wd = edge_cotan_weights(Vd, Td, mode="mollify", delta_rel=1e-4)
    good = bool(np.all(np.isfinite(Wd)))
    ok &= good
    print(f"cotan: degenerate triangle stays finite under mollification "
          f"(|W|max={np.max(np.abs(Wd)):.3g}) {'OK' if good else 'FAIL'}")

    # The assembled Laplacian must be PSD in raw mode even with obtuse
    # triangles present: x^T L x = Dirichlet energy >= 0.
    Vo = rng.normal(size=(8, 3))
    To = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [4, 5, 6], [4, 6, 7]])
    E, W = edge_cotan_weights(Vo, To, mode="raw")
    n = len(Vo)
    Lm = np.zeros((n, n))
    for (a, b), w in zip(E, W):
        Lm[a, a] += w
        Lm[b, b] += w
        Lm[a, b] -= w
        Lm[b, a] -= w
    evals = np.linalg.eigvalsh(Lm)
    good = evals.min() > -1e-10
    ok &= good
    print(f"cotan: raw Laplacian PSD (min eig {evals.min():.2e}) "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("solver.cotan self-test failed")
