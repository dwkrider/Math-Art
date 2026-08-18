# Tangent-point energy flow: tight knots and untangling.
#
# Part of the Math Art knot engine (`math_art/knots/`).  NumPy/stdlib
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules and
# import this package.
#
# The tangent-point energy of a closed curve gamma is
#
#     E = integral integral  |P_perp(T(x)) (x - y)|^alpha / |x - y|^beta
#
# where P_perp(T(x)) projects onto the plane normal to the tangent at x.
# For beta = 2 alpha the kernel is (2 / r(x,y))^alpha with r(x,y) the
# radius of the circle through y tangent to the curve at x, so the
# energy blows up as the curve approaches self-contact: the flow is
# self-avoiding by construction, unlike a smoothing + 1/r repulsion
# heuristic.  This module implements the variant **(alpha, beta) =
# (3, 6)** -- the default of the Yu-Schumacher-Crane reference
# implementation (scene_file.cpp, `repel_curve`) -- discretized exactly
# as their vertex-based quadrature `TPESC::tpe_total`: vertex tangents
# average the two incident edge directions, vertex masses are dual
# (half-edge) lengths, and the sum runs over all ordered vertex pairs
# i != j (the kernel is finite for neighbours, so no exclusion window
# is needed).  The gradient is the exact analytic gradient of that
# discrete energy (all three channels: displacement, tangent, and dual
# length), validated against finite differences in the self-test.
#
# Plain L2 descent on this energy is catastrophically stiff.  The
# preconditioner is the point: the descent direction is the gradient in
# the **fractional Sobolev metric H^s, s = (beta - 1)/alpha (= 5/3
# here)**, whose Gram matrix is assembled densely as 4x4 blocks over
# all non-adjacent edge pairs (a NumPy port of the reference
# implementation's dense path, `SobolevCurves::SobolevGramMatrix`:
# a high-order Ds^T Ds term with kernel |m_I - m_J|^-(2s-1) and a
# low-order midpoint term with kernel k_2^(3+2s), both scaled by
# |I| |J|).  Constraint rows (mass-weighted barycenter, plus total or
# per-edge length -- without a length constraint the energy is
# minimized by inflating the curve to infinity) are appended to the
# same matrix and the bordered dense system is solved directly; the
# step is an Evolver-style parabola line search (shared
# `solver.descent.parabola_line_search`) with a tunneling cap, followed
# by the same constraint-backprojection Newton loop as the reference
# (`LSBackproject` / `BackprojectConstraints`), reusing the iteration's
# factorization.  Barnes-Hut and multigrid acceleration (the paper's
# large-n machinery) are deliberately skipped: at the curve resolutions
# used for art (n of a few hundred) the dense solve is milliseconds.
#
# The Gonzalez-Maddocks thickness (minimum over local radius of
# curvature and pairwise tangent-point radius) and the ropelength
# L / thickness are provided as readouts.
#
# References:
#   G. Buck and J. Orloff, "A simple energy function for knots",
#       Topology Appl. 61 (1995) 205-214 -- the tangent-point energy.
#   O. Gonzalez and J. H. Maddocks, "Global curvature, thickness, and
#       the ideal shapes of knots", PNAS 96 (1999) 4769-4773 -- global
#       radius of curvature, thickness, ideal (tight) knots.
#   P. Strzelecki and H. von der Mosel, "Tangent-point self-avoidance
#       energies for curves", J. Knot Theory Ramifications 21(5)
#       (2012) -- regularity theory for these energies.
#   C. Yu, H. Schumacher, K. Crane, "Repulsive Curves", ACM Trans.
#       Graph. 40(2) (2021) -- the fractional Sobolev-Slobodeckij
#       preconditioner and the discretization ported here; reference
#       implementation `repulsive-curves` (MIT License, (c) 2019
#       Christopher Yu).
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992) -- the parabola line search reused from solver.descent.
#   T. Ashton, J. Cantarella, M. Piatek, E. Rawdon, "Knot Tightening
#       by Constrained Gradient Descent", Experimental Mathematics
#       20(1) (2011) -- reference ropelength values for tight knots
#       (ideal trefoil ~ 32.7429).

import math

import numpy as np

try:                                  # inside the math_art package
    from ..solver.descent import parabola_line_search
except ImportError:                   # flat import (test runner)
    from solver.descent import parabola_line_search


# ----------------------------------------------------------------------
# discrete geometry of a closed polyline
# ----------------------------------------------------------------------

def _geometry(P):
    """Edges, edge lengths, unit edge tangents, vertex tangents (with
    the pre-normalization sum and its norm, needed by the gradient
    chain rule), and dual (vertex mass) lengths of a closed polyline."""
    e = np.roll(P, -1, axis=0) - P            # edge i: vertex i -> i+1
    L = np.linalg.norm(e, axis=1)
    L = np.maximum(L, 1e-300)
    that = e / L[:, None]
    u = that + np.roll(that, 1, axis=0)       # at vertex i: t_{i-1}+t_i
    un = np.maximum(np.linalg.norm(u, axis=1), 1e-300)
    T = u / un[:, None]
    l = 0.5 * (L + np.roll(L, 1, axis=0))     # dual length at vertex i
    return e, L, that, u, un, T, l


def _pair_terms(P, T, alpha, beta):
    """All-pairs displacement D, distance r (diag set to 1), tangential
    component c = D . T_i, normal projection p and its norm pn, and the
    kernel matrix K = pn^alpha / r^beta (diag zeroed)."""
    D = P[:, None, :] - P[None, :, :]
    r = np.linalg.norm(D, axis=2)
    np.fill_diagonal(r, 1.0)
    c = np.einsum('ijk,ik->ij', D, T)
    p = D - c[:, :, None] * T[:, None, :]
    pn = np.linalg.norm(p, axis=2)
    K = pn ** alpha / r ** beta
    np.fill_diagonal(K, 0.0)
    return D, r, c, p, pn, K


# ----------------------------------------------------------------------
# energy and its exact gradient
# ----------------------------------------------------------------------

def tp_energy(P, alpha=3.0, beta=6.0):
    """Discrete tangent-point energy of a closed polyline: the
    reference implementation's vertex quadrature
    sum_{i != j} K(x_i, x_j) l_i l_j (ordered pairs)."""
    P = np.asarray(P, dtype=float)
    _e, _L, _that, _u, _un, T, l = _geometry(P)
    _D, _r, _c, _p, _pn, K = _pair_terms(P, T, alpha, beta)
    return float(np.sum(K * l[:, None] * l[None, :]))


def tp_gradient(P, alpha=3.0, beta=6.0):
    """Exact analytic gradient of `tp_energy` (n, 3).

    Three channels, matching the product/chain rule of the discrete
    energy: the displacement channel (d/dD of the kernel), the tangent
    channel (kernel depends on the vertex tangent T_i, which depends on
    x_{i-1}, x_i, x_{i+1}), and the mass channel (dual lengths)."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    _e, L, that, _u, un, T, l = _geometry(P)
    D, r, c, p, pn, K = _pair_terms(P, T, alpha, beta)
    ll = l[:, None] * l[None, :]

    # guard: where the normal projection vanishes (exactly collinear
    # pairs) its direction is undefined; the true contribution -> 0 for
    # alpha > 1, so zero it (reference guards with 1e-10 the same way).
    pn_safe = np.where(pn > 1e-12, pn, 1.0)
    apn = np.where(pn > 1e-12, alpha * pn_safe ** (alpha - 2.0), 0.0)

    inv_rb = r ** (-beta)
    np.fill_diagonal(inv_rb, 0.0)

    # --- displacement channel: dK/dD = alpha pn^(a-2) p / r^b
    #                                   - beta pn^a D / r^(b+2)
    coef_p = apn * inv_rb * ll                      # (n, n)
    coef_d = beta * pn ** alpha * inv_rb / (r * r) * ll
    Vd = coef_p[:, :, None] * p - coef_d[:, :, None] * D
    grad = Vd.sum(axis=1) - Vd.sum(axis=0)

    # --- tangent channel: dK/dT_i = -alpha pn^(a-2) (D.T_i) p / r^b
    S = -np.einsum('ij,ijk->ik', coef_p * c, p)     # (n, 3) at vertex i
    q = (S - T * np.einsum('ik,ik->i', T, S)[:, None]) / un[:, None]
    tm = np.roll(that, 1, axis=0)                   # t_{i-1} at slot i
    Lm = np.roll(L, 1, axis=0)
    rp = (q - that * np.einsum('ik,ik->i', that, q)[:, None]) / L[:, None]
    rm = (q - tm * np.einsum('ik,ik->i', tm, q)[:, None]) / Lm[:, None]
    grad += np.roll(rp, 1, axis=0) - np.roll(rm, -1, axis=0) + (rm - rp)

    # --- mass channel: dE/dl_i weights c_i = row + column kernel sums
    cw = (K * l[None, :]).sum(axis=1) + (K * l[:, None]).sum(axis=0)
    cwp = np.roll(cw, -1)                           # c_{i+1}
    cwm = np.roll(cw, 1)                            # c_{i-1}
    grad += 0.5 * (tm * (cw + cwm)[:, None] - that * (cw + cwp)[:, None])
    return grad


# ----------------------------------------------------------------------
# fractional Sobolev Gram matrix (dense reference path)
# ----------------------------------------------------------------------

def sobolev_gram(P, alpha=3.0, beta=6.0, diag_eps=0.0):
    """Scalar (n x n) Sobolev-Slobodeckij Gram matrix of the H^s inner
    product, s = (beta - 1)/alpha: dense 4x4 blocks over all ordered
    pairs of non-adjacent edges (port of the reference
    `SobolevCurves::SobolevGramMatrix`).  Constants span its null
    space, so it is only used bordered by constraint rows."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    e, L, that, _u, _un, _T, l = _geometry(P)
    mid = P + 0.5 * e
    s = (beta - 1.0) / alpha
    sig = 2.0 * (s - 1.0) + 1.0                  # 7/3 at (3, 6)

    dm = mid[:, None, :] - mid[None, :, :]
    rmid = np.linalg.norm(dm, axis=2)
    np.fill_diagonal(rmid, 1.0)

    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    sep = np.minimum(sep, n - sep)
    ii, jj = np.nonzero(sep >= 2)                # ordered pairs

    rm = rmid[ii, jj]
    Li, Lj = L[ii], L[jj]
    LL = Li * Lj
    # high-order term: 1 / |m_I - m_J|^(2s-1), times hat-gradient dots
    whi = rm ** (-sig) * LL
    tdot = np.einsum('ik,ik->i', that[ii], that[jj])
    paa = whi / (Li * Li)
    pbb = whi / (Lj * Lj)
    pab = whi * tdot / LL
    # low-order term: k_2^(4+sig) kernel at edge midpoints (symmetrized
    # over the two tangents), times hat-midpoint products (+-1/2)^2
    dvec = dm[ii, jj]
    ci = np.einsum('ik,ik->i', dvec, that[ii])
    cj = np.einsum('ik,ik->i', dvec, that[jj])
    r2 = rm * rm
    kf = 0.5 * ((r2 - ci * ci) + (r2 - cj * cj)) / rm ** (4.0 + sig)
    wlo = 0.25 * kf * LL

    u0, u1 = ii, (ii + 1) % n
    v0, v1 = jj, (jj + 1) % n
    A = np.zeros((n, n))
    for (a, b, val) in (
            (u0, u0, paa + wlo), (u0, u1, -paa + wlo),
            (u1, u0, -paa + wlo), (u1, u1, paa + wlo),
            (v0, v0, pbb + wlo), (v0, v1, -pbb + wlo),
            (v1, v0, -pbb + wlo), (v1, v1, pbb + wlo),
            (u0, v0, -pab - wlo), (u0, v1, pab - wlo),
            (u1, v0, pab - wlo), (u1, v1, -pab - wlo),
            (v0, u0, -pab - wlo), (v0, u1, pab - wlo),
            (v1, u0, pab - wlo), (v1, u1, -pab - wlo)):
        np.add.at(A, (a, b), val)
    if diag_eps:
        A[idx, idx] += diag_eps * l
    return A


# ----------------------------------------------------------------------
# constraints (mass-weighted barycenter + length)
# ----------------------------------------------------------------------

def _constraint_rows(P, mode):
    """Constraint Jacobian C (m, 3n) and current values (m,) for the
    mass-weighted barycenter (3 rows) plus either the total length
    (1 row, mode="total") or every edge length (n rows, mode="edge").
    Coordinates are interleaved: column 3i + c is vertex i, axis c."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    _e, L, that, _u, _un, _T, l = _geometry(P)
    Ltot = float(L.sum())
    tm = np.roll(that, 1, axis=0)                # t_{i-1} at slot i

    rows = []
    vals = []
    # barycenter (mass-weighted, matching the reference's DualLength
    # weights; d(bary)/dx is dominated by the l_i / Ltot diagonal --
    # like the reference we use exactly that row, and let the Newton
    # backprojection absorb the dropped weight-variation terms)
    w = l / Ltot
    for c in range(3):
        row = np.zeros(3 * n)
        row[3 * np.arange(n) + c] = w
        rows.append(row)
        vals.append(float(np.dot(w, P[:, c])))
    if mode == "total":
        row = np.zeros(3 * n)
        g = tm - that                            # dL/dx_i
        for c in range(3):
            row[3 * np.arange(n) + c] = g[:, c]
        rows.append(row)
        vals.append(Ltot)
    elif mode == "edge":
        ar = np.arange(n)
        for k in range(n):
            row = np.zeros(3 * n)
            row[3 * k: 3 * k + 3] = -that[k]
            k1 = (k + 1) % n
            row[3 * k1: 3 * k1 + 3] = that[k]
            rows.append(row)
        vals.extend(L.tolist())
        _ = ar
    else:
        raise ValueError(f"unknown length constraint mode {mode!r}")
    return np.asarray(rows), np.asarray(vals, dtype=float)


def _saddle_inverse(P, alpha, beta, mode, precondition, diag_eps):
    """Inverse of the bordered system [[M, C^T], [C, 0]] where M is the
    H^s Gram (expanded to 3n) or, unpreconditioned, the diagonal L2
    mass matrix diag(l_i)."""
    n = len(P)
    C, _vals = _constraint_rows(P, mode)
    m = C.shape[0]
    A = np.zeros((3 * n + m, 3 * n + m))
    if precondition:
        G = sobolev_gram(P, alpha, beta, diag_eps=diag_eps)
        A[:3 * n, :3 * n] = np.kron(G, np.eye(3))
    else:
        _e, _L, _that, _u, _un, _T, l = _geometry(P)
        A[np.arange(3 * n), np.arange(3 * n)] = np.repeat(l, 3)
    A[:3 * n, 3 * n:] = C.T
    A[3 * n:, :3 * n] = C
    return np.linalg.inv(A), m


# ----------------------------------------------------------------------
# thickness / ropelength readout (Gonzalez-Maddocks)
# ----------------------------------------------------------------------

def gm_thickness(P):
    """Discrete Gonzalez-Maddocks thickness: the minimum over (a) the
    circumradius of every consecutive vertex triple (local radius of
    curvature) and (b) the tangent-point radius r = |d|^2 / (2 |P_perp
    d|) over all ordered vertex pairs with ring separation >= 2 (which
    attains |d|/2 at doubly-critical pairs).  The thickness of the
    round circle of radius R is exactly R under this readout."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    _e, _L, _that, _u, _un, T, _l = _geometry(P)
    D = P[:, None, :] - P[None, :, :]
    r = np.linalg.norm(D, axis=2)
    np.fill_diagonal(r, 1.0)
    c = np.einsum('ijk,ik->ij', D, T)
    perp2 = np.maximum(r * r - c * c, 0.0)
    perp = np.sqrt(perp2)
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    sep = np.minimum(sep, n - sep)
    mask = (sep >= 2) & (perp > 1e-12)
    rtp = np.where(mask, r * r / (2.0 * np.maximum(perp, 1e-300)),
                   np.inf)
    pair_min = float(rtp.min())
    # local circumradius of consecutive triples
    a = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    b = np.linalg.norm(np.roll(P, 1, 0) - P, axis=1)
    cc = np.linalg.norm(np.roll(P, -1, 0) - np.roll(P, 1, 0), axis=1)
    area2 = np.linalg.norm(np.cross(np.roll(P, -1, 0) - P,
                                    np.roll(P, 1, 0) - P), axis=1)
    circ = np.where(area2 > 1e-300, a * b * cc / (2.0 * area2), np.inf)
    return float(min(pair_min, float(circ.min())))


def gm_ropelength(P):
    """Length / thickness -- the dimensionless tightness readout (the
    ideal trefoil's published value is ~ 32.7429)."""
    L = float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())
    return L / gm_thickness(P)


# ----------------------------------------------------------------------
# the flow driver
# ----------------------------------------------------------------------

def _min_gap(P):
    """Closest approach between vertices with ring separation >= 2."""
    n = len(P)
    r = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    sep = np.minimum(sep, n - sep)
    return float(r[sep >= 2].min())


def tighten(P, iters=100, alpha=3.0, beta=6.0, length_mode="edge",
            precondition=True, diag_eps=0.0, backproj_tol=1e-6,
            grad_tol=1e-10, callback=None):
    """Minimize the tangent-point energy of the closed polyline `P`
    under mass-weighted-barycenter plus length constraints
    (`length_mode`: "edge" fixes every edge length, keeping the
    parametrization uniform; "total" fixes only the total length).

    `precondition=True` descends in the fractional Sobolev H^s metric
    (the point of this module); `precondition=False` is plain L2
    descent on the same energy, kept only for A/B measurement.

    Returns (P_out, info) where info records per-iteration energy,
    accepted step, constraint violation, and clearance; the energy is
    non-increasing up to the (recorded) backprojection perturbation.
    """
    x = np.asarray(P, dtype=float).copy()
    n = len(x)
    _C0, targets = _constraint_rows(x, length_mode)
    hist = []
    E = tp_energy(x, alpha, beta)
    E0 = E
    last_s = None
    converged = False
    for it in range(int(iters)):
        dE = tp_gradient(x, alpha, beta)
        Ainv, m = _saddle_inverse(x, alpha, beta, length_mode,
                                  precondition, diag_eps)
        rhs = np.zeros(3 * n + m)
        rhs[:3 * n] = dE.ravel()
        g = (Ainv @ rhs)[:3 * n]
        gnorm = float(np.linalg.norm(g))
        if gnorm < grad_tol:
            converged = True
            break
        d = -g
        dmax = float(np.max(np.linalg.norm(d.reshape(n, 3), axis=1)))
        gap = _min_gap(x)
        s_max = 0.45 * gap / max(dmax, 1e-300)   # tunneling cap
        init = 1.0 / gnorm if gnorm > 1.0 else 1.0 / math.sqrt(gnorm)
        if last_s is not None and last_s > 1e-12:
            init = min(last_s * 1.5, init * 4.0)
        s0 = min(init, s_max)

        def _energy_flat(xf):
            return tp_energy(xf.reshape(n, 3), alpha, beta)

        xf = x.ravel()
        _x_ls, s_used, _E_ls, _ne = parabola_line_search(
            _energy_flat, xf, d, s0, s_max=s_max)
        if s_used <= 0.0:
            converged = True
            break

        # constraint backprojection (Newton, reusing the factorization;
        # halve the step if it cannot be restored)
        viol = np.inf
        for _attempt in range(8):
            y = (xf + s_used * d).reshape(n, 3).copy()
            for _newton in range(3):
                _Cy, vals = _constraint_rows(y, length_mode)
                phi = targets - vals
                viol = float(np.max(np.abs(phi)))
                if viol < backproj_tol:
                    break
                rhs2 = np.zeros(3 * n + m)
                rhs2[3 * n:] = phi
                y += (Ainv @ rhs2)[:3 * n].reshape(n, 3)
            _Cy, vals = _constraint_rows(y, length_mode)
            viol = float(np.max(np.abs(targets - vals)))
            if viol < backproj_tol:
                break
            s_used *= 0.5
            if s_used < 1e-15:
                y = x.copy()
                break
        E_new = tp_energy(y, alpha, beta)
        hist.append({"it": it + 1, "E": E_new, "s": s_used,
                     "viol": viol, "gap": _min_gap(y),
                     "rise": max(0.0, E_new - E)})
        x = y
        E = E_new
        last_s = s_used
        if callback is not None:
            callback(it + 1, x)
    info = {"iters_run": len(hist), "converged": converged,
            "E0": E0, "E": E, "history": hist,
            "viol_max": max((h["viol"] for h in hist), default=0.0),
            "rise_max": max((h["rise"] for h in hist), default=0.0)}
    return x, info


# ----------------------------------------------------------------------
# links: multi-component curves
# ----------------------------------------------------------------------
#
# A link is a list of disjoint closed polylines.  The tangent-point
# double integral simply runs over all pairs of points across all
# components, so the inter-component terms of the SAME kernel are what
# keeps components from passing through each other -- no new energy is
# needed.  Everything below generalizes the single-curve machinery by
# storing the components concatenated and replacing the circular
# `np.roll` neighbour access with per-component successor/predecessor
# index arrays; for a single component the two are the same permutation,
# so the link path reproduces the knot path bitwise (asserted in the
# self-test).
#
# Constraints for links:
#   * length is constrained PER COMPONENT ("edge" mode fixes every edge
#     length, which is per-component by construction; "total" mode gets
#     one row per component) -- a single global length row would let one
#     component inflate while another collapses;
#   * the barycenter constraint stays GLOBAL (3 rows).  Per-component
#     barycenters would freeze the relative positions of the components,
#     which are physical degrees of freedom the flow needs (chain links
#     rearrange, the Hopf components settle to their standoff).  Only
#     the global translation is a null direction of the energy, and the
#     Gram matrix's low-order (midpoint-kernel) term is nonzero on
#     per-component-constant fields, so the global rows are exactly what
#     solvability requires (checked in the self-test).
#
# Additional reference for tight links:
#   J. Cantarella, R. B. Kusner, J. M. Sullivan, "On the minimum
#       ropelength of knots and links", Invent. Math. 150 (2002) --
#       proves the ropelength-tight Hopf link is two round circles in
#       perpendicular planes, each through the other's centre (the
#       geometry the Hopf bench case measures against).

class _LinkTopology:
    """Index bookkeeping for a link stored as concatenated components:
    per-vertex successor/predecessor within the owning component,
    component ids, offsets, and the 'ring separation' matrix (circular
    index distance within a component; cross-component pairs are marked
    2, i.e. always 'far', which is the only way it is consumed)."""

    def __init__(self, sizes):
        sizes = [int(s) for s in sizes]
        if not sizes:
            raise ValueError("link needs at least one component")
        if any(s < 4 for s in sizes):
            raise ValueError("each component needs >= 4 vertices")
        self.sizes = sizes
        self.K = len(sizes)
        self.off = [0]
        for s in sizes:
            self.off.append(self.off[-1] + s)
        N = self.N = self.off[-1]
        nxt = np.arange(1, N + 1)
        prv = np.arange(-1, N - 1)
        cid = np.empty(N, dtype=np.int64)
        for k in range(self.K):
            a, b = self.off[k], self.off[k + 1]
            nxt[b - 1] = a
            prv[a] = b - 1
            cid[a:b] = k
        self.nxt, self.prv, self.cid = nxt, prv, cid
        sep = np.full((N, N), 2, dtype=np.int64)
        for k in range(self.K):
            a, b = self.off[k], self.off[k + 1]
            nk = b - a
            idx = np.arange(nk)
            s = np.abs(idx[:, None] - idx[None, :])
            s = np.minimum(s, nk - s)
            sep[a:b, a:b] = s
        self.sep = sep

    def split(self, X):
        return [X[self.off[k]:self.off[k + 1]].copy()
                for k in range(self.K)]


def _link_concat(comps):
    comps = [np.asarray(Q, dtype=float) for Q in comps]
    topo = _LinkTopology([len(Q) for Q in comps])
    return np.concatenate(comps, axis=0), topo


def _geometry_link(X, topo):
    """`_geometry` with per-component circular neighbours."""
    e = X[topo.nxt] - X
    L = np.linalg.norm(e, axis=1)
    L = np.maximum(L, 1e-300)
    that = e / L[:, None]
    u = that + that[topo.prv]
    un = np.maximum(np.linalg.norm(u, axis=1), 1e-300)
    T = u / un[:, None]
    l = 0.5 * (L + L[topo.prv])
    return e, L, that, u, un, T, l


def _tp_energy_X(X, topo, alpha, beta):
    _e, _L, _that, _u, _un, T, l = _geometry_link(X, topo)
    _D, _r, _c, _p, _pn, K = _pair_terms(X, T, alpha, beta)
    return float(np.sum(K * l[:, None] * l[None, :]))


def tp_energy_link(comps, alpha=3.0, beta=6.0):
    """Discrete tangent-point energy of a link (list of closed
    polylines): the same vertex quadrature as `tp_energy`, summed over
    all ordered vertex pairs of the whole link (intra- AND
    inter-component)."""
    X, topo = _link_concat(comps)
    return _tp_energy_X(X, topo, alpha, beta)


def _tp_gradient_X(X, topo, alpha, beta):
    """Exact analytic gradient of `_tp_energy_X` (N, 3); the
    single-curve `tp_gradient` with neighbour rolls replaced by the
    topology's successor/predecessor maps."""
    nxt, prv = topo.nxt, topo.prv
    _e, L, that, _u, un, T, l = _geometry_link(X, topo)
    D, r, c, p, pn, K = _pair_terms(X, T, alpha, beta)
    ll = l[:, None] * l[None, :]

    pn_safe = np.where(pn > 1e-12, pn, 1.0)
    apn = np.where(pn > 1e-12, alpha * pn_safe ** (alpha - 2.0), 0.0)

    inv_rb = r ** (-beta)
    np.fill_diagonal(inv_rb, 0.0)

    coef_p = apn * inv_rb * ll
    coef_d = beta * pn ** alpha * inv_rb / (r * r) * ll
    Vd = coef_p[:, :, None] * p - coef_d[:, :, None] * D
    grad = Vd.sum(axis=1) - Vd.sum(axis=0)

    S = -np.einsum('ij,ijk->ik', coef_p * c, p)
    q = (S - T * np.einsum('ik,ik->i', T, S)[:, None]) / un[:, None]
    tm = that[prv]
    Lm = L[prv]
    rp = (q - that * np.einsum('ik,ik->i', that, q)[:, None]) / L[:, None]
    rm = (q - tm * np.einsum('ik,ik->i', tm, q)[:, None]) / Lm[:, None]
    grad += rp[prv] - rm[nxt] + (rm - rp)

    cw = (K * l[None, :]).sum(axis=1) + (K * l[:, None]).sum(axis=0)
    cwp = cw[nxt]
    cwm = cw[prv]
    grad += 0.5 * (tm * (cw + cwm)[:, None] - that * (cw + cwp)[:, None])
    return grad


def tp_gradient_link(comps, alpha=3.0, beta=6.0):
    """Gradient of `tp_energy_link`, returned per component."""
    X, topo = _link_concat(comps)
    return topo.split(_tp_gradient_X(X, topo, alpha, beta))


def _sobolev_gram_X(X, topo, alpha, beta, diag_eps=0.0, fast=False):
    """`sobolev_gram` over a link: dense blocks over all ordered pairs
    of non-adjacent edges, where edges of DIFFERENT components are never
    adjacent (they share no vertex), so every cross-component pair
    contributes.  `fast=True` accumulates via one flat `np.bincount`
    instead of sixteen `np.add.at` scatters -- the summation visits the
    identical entries in the identical order, and the self-test asserts
    the two assemblies agree bitwise."""
    N = topo.N
    e, L, that, _u, _un, _T, l = _geometry_link(X, topo)
    mid = X + 0.5 * e
    s = (beta - 1.0) / alpha
    sig = 2.0 * (s - 1.0) + 1.0

    dm = mid[:, None, :] - mid[None, :, :]
    rmid = np.linalg.norm(dm, axis=2)
    np.fill_diagonal(rmid, 1.0)

    ii, jj = np.nonzero(topo.sep >= 2)

    rm = rmid[ii, jj]
    Li, Lj = L[ii], L[jj]
    LL = Li * Lj
    whi = rm ** (-sig) * LL
    tdot = np.einsum('ik,ik->i', that[ii], that[jj])
    paa = whi / (Li * Li)
    pbb = whi / (Lj * Lj)
    pab = whi * tdot / LL
    dvec = dm[ii, jj]
    ci = np.einsum('ik,ik->i', dvec, that[ii])
    cj = np.einsum('ik,ik->i', dvec, that[jj])
    r2 = rm * rm
    kf = 0.5 * ((r2 - ci * ci) + (r2 - cj * cj)) / rm ** (4.0 + sig)
    wlo = 0.25 * kf * LL

    u0, u1 = ii, topo.nxt[ii]
    v0, v1 = jj, topo.nxt[jj]
    entries = (
        (u0, u0, paa + wlo), (u0, u1, -paa + wlo),
        (u1, u0, -paa + wlo), (u1, u1, paa + wlo),
        (v0, v0, pbb + wlo), (v0, v1, -pbb + wlo),
        (v1, v0, -pbb + wlo), (v1, v1, pbb + wlo),
        (u0, v0, -pab - wlo), (u0, v1, pab - wlo),
        (u1, v0, pab - wlo), (u1, v1, -pab - wlo),
        (v0, u0, -pab - wlo), (v0, u1, pab - wlo),
        (v1, u0, pab - wlo), (v1, u1, -pab - wlo))
    if fast:
        lin = np.concatenate([a * N + b for (a, b, _v) in entries])
        w = np.concatenate([v for (_a, _b, v) in entries])
        A = np.bincount(lin, weights=w,
                        minlength=N * N).reshape(N, N)
    else:
        A = np.zeros((N, N))
        for (a, b, val) in entries:
            np.add.at(A, (a, b), val)
    if diag_eps:
        A[np.arange(N), np.arange(N)] += diag_eps * l
    return A


def sobolev_gram_link(comps, alpha=3.0, beta=6.0, diag_eps=0.0):
    """H^s Gram matrix of a link (scalar, N x N over the concatenated
    vertices).  Its null space is the GLOBAL constants only: the
    low-order midpoint term sees differing per-component constants."""
    X, topo = _link_concat(comps)
    return _sobolev_gram_X(X, topo, alpha, beta, diag_eps=diag_eps)


def _constraint_rows_X(X, topo, mode):
    """Constraint Jacobian and values for a link: global mass-weighted
    barycenter (3 rows) plus per-component total length (K rows,
    mode="total") or every edge length (N rows, mode="edge")."""
    N = topo.N
    _e, L, that, _u, _un, _T, l = _geometry_link(X, topo)
    Ltot = float(L.sum())
    tm = that[topo.prv]

    rows = []
    vals = []
    w = l / Ltot
    for c in range(3):
        row = np.zeros(3 * N)
        row[3 * np.arange(N) + c] = w
        rows.append(row)
        vals.append(float(np.dot(w, X[:, c])))
    if mode == "total":
        g = tm - that                            # dL/dx_i
        for k in range(topo.K):
            row = np.zeros(3 * N)
            ar = np.arange(topo.off[k], topo.off[k + 1])
            for c in range(3):
                row[3 * ar + c] = g[ar, c]
            rows.append(row)
            vals.append(float(L[topo.off[k]:topo.off[k + 1]].sum()))
    elif mode == "edge":
        for k in range(N):
            row = np.zeros(3 * N)
            row[3 * k: 3 * k + 3] = -that[k]
            k1 = topo.nxt[k]
            row[3 * k1: 3 * k1 + 3] = that[k]
            rows.append(row)
        vals.extend(L.tolist())
    else:
        raise ValueError(f"unknown length constraint mode {mode!r}")
    return np.asarray(rows), np.asarray(vals, dtype=float)


def _saddle_matrix_X(X, topo, alpha, beta, mode, precondition,
                     diag_eps, fast=False):
    """The bordered matrix [[M, C^T], [C, 0]] for a link (M = H^s Gram
    expanded to 3N, or the diagonal L2 mass matrix)."""
    N = topo.N
    C, _vals = _constraint_rows_X(X, topo, mode)
    m = C.shape[0]
    A = np.zeros((3 * N + m, 3 * N + m))
    if precondition:
        G = _sobolev_gram_X(X, topo, alpha, beta, diag_eps=diag_eps,
                            fast=fast)
        for c in range(3):        # G (kron) I3 without the kron temp
            A[c:3 * N:3, c:3 * N:3] = G
    else:
        _e, _L, _that, _u, _un, _T, l = _geometry_link(X, topo)
        A[np.arange(3 * N), np.arange(3 * N)] = np.repeat(l, 3)
    A[:3 * N, 3 * N:] = C.T
    A[3 * N:, :3 * N] = C
    return A, m


def _min_gap_X(X, topo):
    """Closest approach over 'far' pairs: same component with ring
    separation >= 2, or any cross-component pair."""
    r = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return float(r[topo.sep >= 2].min())


def min_intercomponent_gap(comps):
    """Minimum vertex-vertex distance between DISTINCT components --
    the interpenetration readout (must stay strictly positive)."""
    X, topo = _link_concat(comps)
    if topo.K < 2:
        return float("inf")
    r = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return float(r[topo.cid[:, None] != topo.cid[None, :]].min())


def gm_thickness_link(comps):
    """Gonzalez-Maddocks thickness of a link: minimum over local
    circumradii (per component) and tangent-point radii over all far
    pairs, INCLUDING cross-component pairs (which attain |d|/2 at
    doubly-critical pairs, so inter-component contact bounds the
    thickness exactly as self-contact does)."""
    X, topo = _link_concat(comps)
    _e, _L, _that, _u, _un, T, _l = _geometry_link(X, topo)
    D = X[:, None, :] - X[None, :, :]
    r = np.linalg.norm(D, axis=2)
    np.fill_diagonal(r, 1.0)
    c = np.einsum('ijk,ik->ij', D, T)
    perp2 = np.maximum(r * r - c * c, 0.0)
    perp = np.sqrt(perp2)
    mask = (topo.sep >= 2) & (perp > 1e-12)
    rtp = np.where(mask, r * r / (2.0 * np.maximum(perp, 1e-300)),
                   np.inf)
    pair_min = float(rtp.min())
    a = np.linalg.norm(X[topo.nxt] - X, axis=1)
    b = np.linalg.norm(X[topo.prv] - X, axis=1)
    cc = np.linalg.norm(X[topo.nxt] - X[topo.prv], axis=1)
    area2 = np.linalg.norm(np.cross(X[topo.nxt] - X,
                                    X[topo.prv] - X), axis=1)
    circ = np.where(area2 > 1e-300, a * b * cc / (2.0 * area2), np.inf)
    return float(min(pair_min, float(circ.min())))


def gm_ropelength_link(comps):
    """Total length / link thickness."""
    X, topo = _link_concat(comps)
    L = float(np.linalg.norm(X[topo.nxt] - X, axis=1).sum())
    return L / gm_thickness_link(comps)


def linking_matrix(comps):
    """Pairwise Gauss linking numbers (K x K integer matrix, zero
    diagonal) and the largest deviation of the raw integrals from the
    nearest integer -- the topological gate for link flows (the flow
    may not change any entry)."""
    from .relax import linking_number
    K = len(comps)
    M = np.zeros((K, K), dtype=np.int64)
    dev = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            lk = linking_number(comps[i], comps[j])
            M[i, j] = M[j, i] = int(round(lk))
            dev = max(dev, abs(lk - round(lk)))
    return M, dev


# ----------------------------------------------------------------------
# saddle solvers: exact dense, and lagged factorization reuse
# ----------------------------------------------------------------------

class _DenseSaddleSolver:
    """Exact reference path: assemble and invert the bordered system
    every iteration (what `tighten` does inline)."""

    def __init__(self, alpha, beta, mode, precondition, diag_eps):
        self.args = (alpha, beta, mode, precondition, diag_eps)
        self.Ainv = None
        self.m = 0
        self.factorizations = 0

    def prepare(self, X, topo, it, gap):
        alpha, beta, mode, precondition, diag_eps = self.args
        A, self.m = _saddle_matrix_X(X, topo, alpha, beta, mode,
                                     precondition, diag_eps)
        self.Ainv = np.linalg.inv(A)
        self.factorizations += 1

    def solve(self, rhs):
        return self.Ainv @ rhs

    def stale(self):
        return False

    def force_refresh(self, X, topo, it, gap):
        pass


class _LaggedSaddleSolver:
    """Refresh the dense factorization only every `refresh_every`
    iterations, or sooner if the geometry has drifted by more than
    `refresh_drift` times the clearance recorded at the last refresh.

    Profiling drove this choice: at n = 400 the dense inverse is ~94%
    of an iteration (2.6 s of 2.8 s here), and the O(n^2) energy /
    gradient / assembly terms are two orders of magnitude cheaper -- so
    reusing one factorization across a few iterations removes almost
    the whole O(n^3) cost.  Correctness does not depend on freshness:
    the reused matrix is still an SPD metric on the constraint tangent
    space (of the slightly earlier geometry), so -A^-1 grad is still a
    descent direction for the line search to vet, and the constraint
    backprojection Newton iterates against the CURRENT constraint
    values (a frozen Jacobian, exactly like the reference
    implementation's reuse of one factorization across its
    backprojection).  The flow driver refreshes and retries once
    whenever the line search or the backprojection fails, so a stale
    metric can cost iterations but not correctness; the A/B bench
    holds it to the dense path's converged energy.  The Gram assembly
    uses the bincount fast path (bitwise-identical accumulation,
    asserted in the self-test)."""

    def __init__(self, alpha, beta, mode, precondition, diag_eps,
                 refresh_every=8, refresh_drift=0.2):
        self.args = (alpha, beta, mode, precondition, diag_eps)
        self.refresh_every = max(1, int(refresh_every))
        self.refresh_drift = float(refresh_drift)
        self.Ainv = None
        self.m = 0
        self.age = 0
        self.X_ref = None
        self.gap_ref = None
        self.factorizations = 0

    def _refresh(self, X, topo, gap):
        alpha, beta, mode, precondition, diag_eps = self.args
        A, self.m = _saddle_matrix_X(X, topo, alpha, beta, mode,
                                     precondition, diag_eps, fast=True)
        self.Ainv = np.linalg.inv(A)
        self.age = 0
        self.X_ref = X.copy()
        self.gap_ref = gap
        self.factorizations += 1

    def prepare(self, X, topo, it, gap):
        if self.Ainv is None or self.age >= self.refresh_every:
            self._refresh(X, topo, gap)
            return
        drift = float(np.max(np.linalg.norm(X - self.X_ref, axis=1)))
        if drift > self.refresh_drift * self.gap_ref:
            self._refresh(X, topo, gap)
            return
        self.age += 1

    def solve(self, rhs):
        return self.Ainv @ rhs

    def stale(self):
        return self.age > 0

    def force_refresh(self, X, topo, it, gap):
        self._refresh(X, topo, gap)


def tighten_link(comps, iters=100, alpha=3.0, beta=6.0,
                 length_mode="edge", precondition=True, diag_eps=0.0,
                 backproj_tol=1e-6, grad_tol=1e-10, solver="dense",
                 refresh_every=8, refresh_drift=0.2, callback=None):
    """Minimize the tangent-point energy of a link (list of closed
    polylines) under a global mass-weighted barycenter plus
    PER-COMPONENT length constraints.  The inter-component kernel terms
    keep components from passing through each other, and the tunneling
    cap (0.45 x clearance, where clearance includes cross-component
    pairs) keeps a discrete step from jumping a strand through another
    between samples -- so pairwise linking numbers are preserved.

    `solver`: "dense" refactors the bordered system every iteration
    (exact reference path -- for one component it reproduces `tighten`
    bitwise); "lagged" reuses the factorization for up to
    `refresh_every` iterations (or until the geometry drifts by
    `refresh_drift` x clearance), removing the dominant O(n^3) cost --
    profiled at ~94% of an iteration at n = 400.

    Returns (comps_out, info); info adds `inter_gap` per iteration (the
    minimum cross-component clearance) and `factorizations`."""
    X, topo = _link_concat(comps)
    n = topo.N
    if solver == "dense":
        sad = _DenseSaddleSolver(alpha, beta, length_mode,
                                 precondition, diag_eps)
    elif solver == "lagged":
        sad = _LaggedSaddleSolver(alpha, beta, length_mode,
                                  precondition, diag_eps,
                                  refresh_every=refresh_every,
                                  refresh_drift=refresh_drift)
    else:
        raise ValueError(f"unknown solver {solver!r}")

    def _inter_gap(Y):
        if topo.K < 2:
            return float("inf")
        r = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
        return float(r[topo.cid[:, None] != topo.cid[None, :]].min())

    x = X.copy()
    _C0, targets = _constraint_rows_X(x, topo, length_mode)
    hist = []
    E = _tp_energy_X(x, topo, alpha, beta)
    E0 = E
    last_s = None
    converged = False
    it = 0
    retried = False
    while it < int(iters):
        it += 1
        dE = _tp_gradient_X(x, topo, alpha, beta)
        gap = _min_gap_X(x, topo)
        sad.prepare(x, topo, it, gap)
        m = sad.m
        rhs = np.zeros(3 * n + m)
        rhs[:3 * n] = dE.ravel()
        g = sad.solve(rhs)[:3 * n]
        gnorm = float(np.linalg.norm(g))
        if gnorm < grad_tol:
            converged = True
            break
        d = -g
        dmax = float(np.max(np.linalg.norm(d.reshape(n, 3), axis=1)))
        s_max = 0.45 * gap / max(dmax, 1e-300)   # tunneling cap
        init = 1.0 / gnorm if gnorm > 1.0 else 1.0 / math.sqrt(gnorm)
        if last_s is not None and last_s > 1e-12:
            init = min(last_s * 1.5, init * 4.0)
        s0 = min(init, s_max)

        def _energy_flat(xf):
            return _tp_energy_X(xf.reshape(n, 3), topo, alpha, beta)

        xf = x.ravel()
        _x_ls, s_used, _E_ls, _ne = parabola_line_search(
            _energy_flat, xf, d, s0, s_max=s_max)
        if s_used <= 0.0:
            if sad.stale() and not retried:
                # a stale metric can propose a non-descent direction;
                # refresh once and redo this iteration before giving up
                sad.force_refresh(x, topo, it, gap)
                it -= 1
                retried = True
                continue
            converged = True
            break
        retried = False

        viol = np.inf
        for _attempt in range(8):
            y = (xf + s_used * d).reshape(n, 3).copy()
            for _newton in range(3):
                _Cy, vals = _constraint_rows_X(y, topo, length_mode)
                phi = targets - vals
                viol = float(np.max(np.abs(phi)))
                if viol < backproj_tol:
                    break
                rhs2 = np.zeros(3 * n + m)
                rhs2[3 * n:] = phi
                y += sad.solve(rhs2)[:3 * n].reshape(n, 3)
            _Cy, vals = _constraint_rows_X(y, topo, length_mode)
            viol = float(np.max(np.abs(targets - vals)))
            if viol < backproj_tol:
                break
            s_used *= 0.5
            if s_used < 1e-15:
                y = x.copy()
                break
        E_new = _tp_energy_X(y, topo, alpha, beta)
        hist.append({"it": it, "E": E_new, "s": s_used,
                     "viol": viol, "gap": _min_gap_X(y, topo),
                     "inter_gap": _inter_gap(y),
                     "rise": max(0.0, E_new - E)})
        x = y
        E = E_new
        last_s = s_used
        if callback is not None:
            callback(it, x)
    info = {"iters_run": len(hist), "converged": converged,
            "E0": E0, "E": E, "history": hist,
            "viol_max": max((h["viol"] for h in hist), default=0.0),
            "rise_max": max((h["rise"] for h in hist), default=0.0),
            "inter_gap_min": min((h["inter_gap"] for h in hist),
                                 default=float("inf")),
            "factorizations": sad.factorizations}
    return topo.split(x), info


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------

def _circle(n, R=1.0):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([R * np.cos(t), R * np.sin(t),
                     np.zeros_like(t)], axis=1)


def _selftest():
    ok = True
    rng = np.random.default_rng(7)

    # 1. The analytic gradient must match central finite differences on
    # a generic closed curve (a perturbed trefoil-ish curve).
    t = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    P = np.stack([np.cos(2 * t) + 0.4 * np.cos(3 * t),
                  np.sin(2 * t) - 0.3 * np.sin(t),
                  0.5 * np.sin(3 * t)], axis=1)
    P += 0.01 * rng.standard_normal(P.shape)
    g = tp_gradient(P)
    h = 1e-6
    gfd = np.zeros_like(P)
    for i in range(len(P)):
        for c in range(3):
            Pp = P.copy()
            Pp[i, c] += h
            Pm = P.copy()
            Pm[i, c] -= h
            gfd[i, c] = (tp_energy(Pp) - tp_energy(Pm)) / (2 * h)
    rel = np.linalg.norm(g - gfd) / np.linalg.norm(gfd)
    good = rel < 1e-5
    ok &= good
    print(f"tangent_point: gradient vs FD rel err {rel:.2e} "
          f"{'OK' if good else 'FAIL'}")

    # 2. Circle energy converges to the analytic value pi^2 / (2 R)
    # (for the circle the tangent-point radius equals R for EVERY pair,
    # so the continuous kernel is the constant 1/(8 R^3)).  The vertex
    # quadrature omits the diagonal, so first-order convergence is the
    # expected rate -- measure and report it.
    R = 1.0
    E_exact = math.pi ** 2 / (2.0 * R)
    errs = []
    ns = (32, 64, 128, 256)
    for n in ns:
        errs.append(abs(tp_energy(_circle(n, R)) - E_exact) / E_exact)
    orders = [math.log2(errs[k] / errs[k + 1])
              for k in range(len(errs) - 1)]
    good = errs[0] > errs[-1] and 0.7 < orders[-1] < 2.3
    ok &= good
    print(f"tangent_point: circle E err {errs[0]:.2e} -> {errs[-1]:.2e}"
          f" order ~{orders[-1]:.2f} {'OK' if good else 'FAIL'}")

    # 3. The round circle is a critical point: the constraint-projected
    # Sobolev gradient vanishes (the raw gradient is a uniform radial
    # field, which lies in the span of the length-constraint rows), and
    # a short flow keeps it round to machine precision.
    C = _circle(96)
    n = len(C)
    dE = tp_gradient(C)
    Ainv, m = _saddle_inverse(C, 3.0, 6.0, "edge", True, 0.0)
    rhs = np.zeros(3 * n + m)
    rhs[:3 * n] = dE.ravel()
    g = (Ainv @ rhs)[:3 * n]
    gn = np.linalg.norm(g) / max(np.linalg.norm(dE), 1e-30)
    C2, info = tighten(C, iters=5)
    rad = np.linalg.norm(C2 - C2.mean(0), axis=1)
    cv = float(np.std(rad) / np.mean(rad))
    good = gn < 1e-8 and cv < 1e-6
    ok &= good
    print(f"tangent_point: circle critical (|Pg|/|dE|={gn:.1e}, "
          f"radius cv after flow {cv:.1e}) {'OK' if good else 'FAIL'}")

    # 4. GM thickness of the circle is exactly R; ropelength 2 pi.
    tau = gm_thickness(_circle(128, 2.0))
    rl = gm_ropelength(_circle(128, 2.0))
    good = abs(tau - 2.0) < 1e-2 and abs(rl - 2 * math.pi) < 0.05
    ok &= good
    print(f"tangent_point: circle thickness {tau:.4f} (exp 2), "
          f"ropelength {rl:.4f} (exp {2 * math.pi:.4f}) "
          f"{'OK' if good else 'FAIL'}")

    # 5. A short trefoil flow: energy monotone (up to the recorded
    # backprojection perturbation), constraints held, clearance stays
    # strictly positive, and the flow makes real progress.
    from .braid import braid_closure_points, parse_letters
    from .resample import resample_closed
    P0 = resample_closed(braid_closure_points(parse_letters('AAA')), 96)
    L0 = float(np.linalg.norm(np.roll(P0, -1, 0) - P0, axis=1).sum())
    P1, info = tighten(P0, iters=30)
    L1 = float(np.linalg.norm(np.roll(P1, -1, 0) - P1, axis=1).sum())
    gaps = [h["gap"] for h in info["history"]]
    good = (info["E"] < info["E0"]
            and info["rise_max"] < 1e-6 * info["E0"]
            and info["viol_max"] < 1e-5
            and min(gaps) > 0.0
            and abs(L1 - L0) / L0 < 1e-6)
    ok &= good
    print(f"tangent_point: trefoil 30 iters E {info['E0']:.3f} -> "
          f"{info['E']:.3f}, rise_max {info['rise_max']:.1e}, viol "
          f"{info['viol_max']:.1e}, len drift {abs(L1 - L0) / L0:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # 6. The preconditioner must beat plain L2 descent on the same
    # energy and budget -- the reason this module exists.
    P2, info2 = tighten(P0, iters=10)
    P3, info3 = tighten(P0, iters=10, precondition=False)
    good = info2["E"] < info3["E"]
    ok &= good
    print(f"tangent_point: 10 iters H^s E={info2['E']:.4f} vs L2 "
          f"E={info3['E']:.4f} {'OK' if good else 'FAIL'}")

    # ------------------------------------------------------------------
    # links
    # ------------------------------------------------------------------

    # 7. For a SINGLE component the link path must reproduce the knot
    # path bitwise: the neighbour maps are the same permutation np.roll
    # applies, and every arithmetic operation is written in the same
    # order.  (This is the guard that keeps the two code paths from
    # drifting apart.)
    Pk = P  # the generic perturbed curve from test 1
    Xl, topo1 = _link_concat([Pk])
    e_knot = tp_energy(Pk)
    e_link = _tp_energy_X(Xl, topo1, 3.0, 6.0)
    g_knot = tp_gradient(Pk)
    g_link = _tp_gradient_X(Xl, topo1, 3.0, 6.0)
    G_knot = sobolev_gram(Pk)
    G_link = _sobolev_gram_X(Xl, topo1, 3.0, 6.0)
    Ck, vk = _constraint_rows(Pk, "edge")
    Cl, vl = _constraint_rows_X(Xl, topo1, "edge")
    good = (e_knot == e_link
            and np.array_equal(g_knot, g_link)
            and np.array_equal(G_knot, G_link)
            and np.array_equal(Ck, Cl) and np.array_equal(vk, vl))
    ok &= good
    print(f"tangent_point: link path == knot path (1 comp, bitwise) "
          f"{'OK' if good else 'FAIL'}")

    from .braid import braid_closure_points, parse_letters  # noqa: F811
    from .resample import resample_closed  # noqa: F811
    P0k = resample_closed(braid_closure_points(parse_letters('AAA')), 64)
    a_knot, ia = tighten(P0k, iters=3)
    a_link, ib = tighten_link([P0k], iters=3)
    good = (ia["E"] == ib["E"] and np.array_equal(a_knot, a_link[0]))
    ok &= good
    print(f"tangent_point: tighten_link([P]) == tighten(P) after 3 "
          f"iters (bitwise) {'OK' if good else 'FAIL'}")

    # 8. The bincount fast assembly must agree BITWISE with the add.at
    # assembly (same entries visited in the same order) -- on a real
    # 2-component link so cross-component blocks are exercised.
    t2 = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    A2 = np.stack([np.cos(t2), np.sin(t2), np.zeros_like(t2)], axis=1)
    B2 = np.stack([1.0 + np.cos(t2), np.zeros_like(t2),
                   np.sin(t2)], axis=1)
    Xh, topoh = _link_concat([A2, B2])
    Gs = _sobolev_gram_X(Xh, topoh, 3.0, 6.0, fast=False)
    Gf = _sobolev_gram_X(Xh, topoh, 3.0, 6.0, fast=True)
    good = np.array_equal(Gs, Gf)
    ok &= good
    print(f"tangent_point: gram bincount == add.at (bitwise) "
          f"{'OK' if good else 'FAIL'}")

    # 9. Null space of the link Gram: global constants only.  A
    # per-component constant must NOT be in the null space (the
    # low-order midpoint term sees it) -- this is what lets the single
    # global barycenter constraint make the bordered system solvable.
    ones = np.ones(topoh.N)
    percomp = np.where(topoh.cid == 0, 1.0, -1.0)
    r_ones = np.linalg.norm(Gs @ ones) / np.linalg.norm(Gs)
    r_pc = np.linalg.norm(Gs @ percomp) / np.linalg.norm(Gs)
    good = r_ones < 1e-12 and r_pc > 1e-6
    ok &= good
    print(f"tangent_point: gram null space global-only "
          f"(|G 1|={r_ones:.1e}, |G percomp|={r_pc:.1e}) "
          f"{'OK' if good else 'FAIL'}")

    # 10. Multi-component gradient vs central finite differences on a
    # generic 2-component link (same bar as the knot gradient).
    rng2 = np.random.default_rng(11)
    t3 = np.linspace(0.0, 2.0 * np.pi, 14, endpoint=False)
    Ca = np.stack([np.cos(t3), np.sin(t3), 0.2 * np.sin(2 * t3)],
                  axis=1) + 0.01 * rng2.standard_normal((14, 3))
    Cb = np.stack([1.0 + 0.9 * np.cos(t3), 0.1 * np.cos(3 * t3),
                   0.9 * np.sin(t3)], axis=1) \
        + 0.01 * rng2.standard_normal((14, 3))
    Xg, topog = _link_concat([Ca, Cb])
    gl = _tp_gradient_X(Xg, topog, 3.0, 6.0)

    def _fd(h):
        g = np.zeros_like(Xg)
        for i in range(topog.N):
            for cax in range(3):
                Xp = Xg.copy()
                Xp[i, cax] += h
                Xm = Xg.copy()
                Xm[i, cax] -= h
                g[i, cax] = (_tp_energy_X(Xp, topog, 3.0, 6.0)
                             - _tp_energy_X(Xm, topog, 3.0, 6.0)) / (2 * h)
        return g

    # Richardson-extrapolate the central differences (kills the h^2
    # truncation term, which otherwise floors this stiff two-circle
    # configuration at ~1e-9 regardless of the gradient's exactness).
    gfd2 = (4.0 * _fd(1e-5) - _fd(2e-5)) / 3.0
    rel2 = np.linalg.norm(gl - gfd2) / np.linalg.norm(gfd2)
    good = rel2 < 1e-8
    ok &= good
    print(f"tangent_point: link gradient vs FD (Richardson) rel err "
          f"{rel2:.2e} {'OK' if good else 'FAIL'}")

    # 11. A short Hopf-link flow: linking number preserved, both
    # component lengths individually preserved, energy decreases,
    # inter-component clearance strictly positive throughout.
    t4 = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    Ha = np.stack([np.cos(t4), np.sin(t4), np.zeros_like(t4)], axis=1)
    Hb = np.stack([1.0 + np.cos(t4), np.zeros_like(t4),
                   np.sin(t4)], axis=1)
    lk0, _dev0 = linking_matrix([Ha, Hb])
    La0 = float(np.linalg.norm(np.roll(Ha, -1, 0) - Ha, axis=1).sum())
    Lb0 = float(np.linalg.norm(np.roll(Hb, -1, 0) - Hb, axis=1).sum())
    out, infoh = tighten_link([Ha, Hb], iters=15)
    lk1, dev1 = linking_matrix(out)
    La1 = float(np.linalg.norm(np.roll(out[0], -1, 0) - out[0],
                               axis=1).sum())
    Lb1 = float(np.linalg.norm(np.roll(out[1], -1, 0) - out[1],
                               axis=1).sum())
    good = (np.array_equal(lk0, lk1) and abs(lk0[0, 1]) == 1
            and dev1 < 0.05
            and infoh["E"] < infoh["E0"]
            and infoh["inter_gap_min"] > 0.0
            and infoh["viol_max"] < 1e-5
            and abs(La1 - La0) / La0 < 1e-6
            and abs(Lb1 - Lb0) / Lb0 < 1e-6)
    ok &= good
    print(f"tangent_point: Hopf 15 iters lk={lk1[0, 1]:+d} kept, E "
          f"{infoh['E0']:.2f}->{infoh['E']:.2f}, inter gap "
          f"{infoh['inter_gap_min']:.3f}, len drift "
          f"{max(abs(La1 - La0) / La0, abs(Lb1 - Lb0) / Lb0):.1e} "
          f"{'OK' if good else 'FAIL'}")

    # 12. The lagged solver must land on the dense answer: same
    # converged energy to a tight relative tolerance, topology intact,
    # and it must actually skip factorizations.
    outd, infod = tighten_link([Ha, Hb], iters=15, solver="dense")
    outl, infol = tighten_link([Ha, Hb], iters=15, solver="lagged")
    lkd, _ = linking_matrix(outd)
    lkl, _ = linking_matrix(outl)
    ed, el = infod["E"], infol["E"]
    rel = abs(el - ed) / ed
    good = (rel < 1e-4 and np.array_equal(lkd, lkl)
            and infol["factorizations"] < infod["factorizations"])
    ok &= good
    print(f"tangent_point: lagged vs dense E rel diff {rel:.1e}, "
          f"factorizations {infol['factorizations']} vs "
          f"{infod['factorizations']} {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("tangent_point self-test failed")
