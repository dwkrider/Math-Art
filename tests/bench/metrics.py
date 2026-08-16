# Quality metrics for the solver benchmark.
#
# Every metric is a plain function of arrays, deterministic, and chosen
# as a proxy for something visible in the art output:
#
#   * area / analytic-area error      -- how minimal a "minimal" surface is;
#   * mean-curvature residual         -- |H| should -> 0 on a true minimal
#                                        surface; bumps show as shading blemishes;
#   * triangle radius-ratio, min angle, edge-length CV, valence deviation
#                                     -- sliver/starburst artifacts show directly
#                                        in specular shading and in later
#                                        subdivision;
#   * cotan clamp fraction            -- how much the positivity clamp is
#                                        biasing the solve toward the Tutte
#                                        (uniform-weight) solution;
#   * self-intersection count         -- embeddedness, the hard visual failure;
#   * tangency/planarity spreads      -- the definition of canonical/biscribed
#                                        forms, visible as broken symmetry;
#   * knot clearance / length         -- rope thickness and tautness;
#   * hyperbolic edge distortion      -- how faithfully the buckled sheet
#                                        carries the target metric.
import math

import numpy as np


# --------------------------------------------------------------------------
# triangle-mesh metrics
# --------------------------------------------------------------------------

def mesh_area(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return 0.5 * float(np.sum(np.linalg.norm(n, axis=1)))


def _edge_lengths(V, T):
    a = np.linalg.norm(V[T[:, 1]] - V[T[:, 2]], axis=1)
    b = np.linalg.norm(V[T[:, 2]] - V[T[:, 0]], axis=1)
    c = np.linalg.norm(V[T[:, 0]] - V[T[:, 1]], axis=1)
    return a, b, c


def tri_quality(V, T):
    """Radius-ratio quality q = 2 r_in / r_circ = 8 A^2 / (s a b c);
    1.0 for equilateral, -> 0 for slivers.  Also the minimum corner
    angle (degrees) and the count of degenerate triangles."""
    a, b, c = _edge_lengths(V, T)
    s = 0.5 * (a + b + c)
    area2 = np.linalg.norm(
        np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    A = 0.5 * area2
    scale = float(np.mean(a + b + c) / 3.0)
    degen = int(np.sum(A < 1e-12 * scale * scale + 1e-300))
    denom = np.maximum(s * a * b * c, 1e-300)
    q = 8.0 * A * A / denom
    # corner angles via the law of cosines
    angs = []
    for (e0, e1, e2) in ((a, b, c), (b, c, a), (c, a, b)):
        cosv = (e1 * e1 + e2 * e2 - e0 * e0) / np.maximum(2 * e1 * e2, 1e-300)
        angs.append(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))
    angs = np.stack(angs)
    return {
        "q_min": float(np.min(q)),
        "q_mean": float(np.mean(q)),
        "q_p05": float(np.percentile(q, 5)),
        "min_angle_deg": float(np.min(angs)),
        "degenerate": degen,
    }


def edge_len_cv(V, T):
    """Coefficient of variation of unique edge lengths."""
    E = np.unique(np.sort(np.concatenate(
        [T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return float(np.std(L) / max(np.mean(L), 1e-300))


def valence_rms_dev(T, n, boundary=None):
    """RMS deviation of interior vertex valence from 6."""
    E = np.unique(np.sort(np.concatenate(
        [T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    val = np.zeros(n)
    np.add.at(val, E[:, 0], 1)
    np.add.at(val, E[:, 1], 1)
    mask = np.ones(n, dtype=bool)
    if boundary is not None:
        mask &= ~boundary
    used = np.zeros(n, dtype=bool)
    used[T.ravel()] = True
    mask &= used
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean((val[mask] - 6.0) ** 2)))


def cotan_clamp_fraction(V, T, lo=0.01, hi=20.0):
    """Fraction of corner cotangents outside [lo, hi] -- i.e. how often
    plateau._cotan_weights' positivity clamp fires and biases the solve.
    Also reports the genuinely-negative fraction (obtuse corners)."""
    total = 0
    clamped = 0
    negative = 0
    for (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = V[T[:, a]] - V[T[:, c]]
        w = V[T[:, b]] - V[T[:, c]]
        cross = np.linalg.norm(np.cross(u, w), axis=1)
        cot = np.einsum('ij,ij->i', u, w) / np.maximum(cross, 1e-12)
        total += len(cot)
        clamped += int(np.sum((cot < lo) | (cot > hi)))
        negative += int(np.sum(cot < 0.0))
    return {"clamp_frac": clamped / max(total, 1),
            "neg_cot_frac": negative / max(total, 1)}


def mean_curvature_residual(V, T, free=None):
    """Discrete mean curvature |H| per interior vertex from the RAW
    (unclamped) cotan Laplacian over the barycentric vertex area,
    nondimensionalised by the mean edge length.  ->0 on a discrete
    minimal surface; the residual is what shows as shading bumps."""
    n = len(V)
    lap = np.zeros((n, 3))
    Astar = np.zeros(n)
    for (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = V[T[:, a]] - V[T[:, c]]
        w = V[T[:, b]] - V[T[:, c]]
        cr = np.cross(u, w)
        crn = np.maximum(np.linalg.norm(cr, axis=1), 1e-300)
        cot = np.einsum('ij,ij->i', u, w) / crn
        d = V[T[:, b]] - V[T[:, a]]
        contrib = 0.5 * cot[:, None] * d
        np.add.at(lap, T[:, a], contrib)
        np.add.at(lap, T[:, b], -contrib)
    tri_area = 0.5 * np.linalg.norm(
        np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    for k in range(3):
        np.add.at(Astar, T[:, k], tri_area / 3.0)
    Hvec = lap / (2.0 * np.maximum(Astar, 1e-300))[:, None]
    Hmag = np.linalg.norm(Hvec, axis=1)
    a, b, c = _edge_lengths(V, T)
    lmean = float(np.mean(np.concatenate([a, b, c])))
    mask = np.zeros(n, dtype=bool)
    mask[T.ravel()] = True
    if free is not None:
        mask &= free
    h = Hmag[mask] * lmean            # dimensionless
    if len(h) == 0:
        return {"H_rms": 0.0, "H_max": 0.0}
    return {"H_rms": float(np.sqrt(np.mean(h * h))),
            "H_max": float(np.max(h))}


def selfx_count(V, T):
    """Triangle-mesh self-intersections, via the existing counter in
    math_art.minsurf.plateau."""
    from math_art.minsurf import plateau
    return int(plateau._selfx_crossings(np.asarray(V, float),
                                        np.asarray(T, np.int64)))


def max_move(V0, V1, mask=None):
    d = np.linalg.norm(np.asarray(V1) - np.asarray(V0), axis=1)
    if mask is not None:
        d = d[mask]
    return float(np.max(d)) if len(d) else 0.0


def mean_move(V0, V1, mask=None):
    d = np.linalg.norm(np.asarray(V1) - np.asarray(V0), axis=1)
    if mask is not None:
        d = d[mask]
    return float(np.mean(d)) if len(d) else 0.0


# --------------------------------------------------------------------------
# polyhedron metrics (canonical / biscribed forms)
# --------------------------------------------------------------------------

def _edge_list(F):
    return sorted({tuple(sorted((f[i], f[(i + 1) % len(f)])))
                   for f in F for i in range(len(f))})


def tangent_spread(V, F):
    """Spread of origin distances to each edge's nearest point, relative
    to their mean.  Zero iff every edge is tangent to one sphere -- the
    canonical-form residual (same definition as the module self-test)."""
    P = np.asarray(V, float)
    d = []
    for a, b in _edge_list(F):
        A, B = P[a], P[b]
        AB = B - A
        t = -np.dot(A, AB) / max(np.dot(AB, AB), 1e-30)
        t = min(1.0, max(0.0, t))
        d.append(np.linalg.norm(A + t * AB))
    d = np.asarray(d)
    return float((d.max() - d.min()) / max(d.mean(), 1e-12))


def sphere_spreads(V, F):
    """(vertex-radius spread, face-distance spread), both relative --
    the two residuals that define the biscribed form."""
    P = np.asarray(V, float)
    rv = np.linalg.norm(P, axis=1)
    fd = []
    for f in F:
        c = P[list(f)].mean(axis=0)
        n = np.zeros(3)
        for i in range(len(f)):
            n += np.cross(P[f[i]] - c, P[f[(i + 1) % len(f)]] - c)
        ln = np.linalg.norm(n)
        if ln > 1e-12:
            fd.append(abs(float(np.dot(c, n / ln))))
    fd = np.asarray(fd)
    return (float(rv.std() / rv.mean()),
            float(fd.std() / max(fd.mean(), 1e-30)))


def planarity_max(V, F):
    """Max out-of-plane deviation over faces, relative to circumradius."""
    P = np.asarray(V, float)
    m = 0.0
    for f in F:
        Q = P[list(f)]
        c = Q.mean(axis=0)
        U, S, Vt = np.linalg.svd(Q - c)
        m = max(m, float(np.abs((Q - c) @ Vt[2]).max()))
    R = float(np.linalg.norm(P - P.mean(0), axis=1).max())
    return m / max(R, 1e-30)


# --------------------------------------------------------------------------
# curve (knot) metrics
# --------------------------------------------------------------------------

def curve_length(P):
    return float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())


def min_far_gap(P, excl=6):
    """Closest approach between points separated by more than `excl`
    along the closed curve -- the rope-thickness proxy."""
    n = len(P)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    sep = np.minimum(sep, n - sep)
    return float(d[sep > excl].min())


def ropelength_proxy(P, excl=6):
    """length / clearance: lower = tighter rope at equal clearance.
    Not the Gonzalez-Maddocks thickness (no doubly-critical test), but
    monotone in the same quantities and cheap."""
    return curve_length(P) / max(min_far_gap(P, excl), 1e-30)


def turning_rms_deg(P):
    """RMS turning angle between consecutive segments (degrees):
    smoothness of the relaxed rope."""
    d = np.roll(P, -1, 0) - P
    dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
    cosv = np.clip(np.einsum('ij,ij->i', dn, np.roll(dn, 1, 0)), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosv))
    return float(np.sqrt(np.mean(ang * ang)))


# --------------------------------------------------------------------------
# hyperbolic-sheet metrics
# --------------------------------------------------------------------------

def edge_metric_error(P, E0, E1, REST):
    """Relative edge-length error against the target (hyperbolic) rest
    lengths: the buckling-vs-flattening signature."""
    L = np.linalg.norm(P[E1] - P[E0], axis=1)
    rel = (L - REST) / np.maximum(REST, 1e-30)
    return {"edge_err_rms": float(np.sqrt(np.mean(rel * rel))),
            "edge_err_max": float(np.max(np.abs(rel)))}
