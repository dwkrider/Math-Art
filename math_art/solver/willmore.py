# Willmore / spontaneous-curvature bending energy with its EXACT
# discrete gradient, and a constrained quasi-Newton minimizer.
#
# Part of the shared solver core (`math_art/solver/`).  NumPy only.
#
# Energy (star discretization, Evolver's sq_mean_curvature family /
# Mem3DG's Helfrich form with bending modulus 1):
#
#     E = sum_v a_v (H_v - h0)^2,
#     H_v = sigma_v |g_v| / (2 a_v),
#
# with g_v the vertex area gradient (the integrated mean-curvature
# vector, by the cotan identity), a_v the barycentric vertex area
# (star/3), and sigma_v the sign of g_v against the outward vertex
# normal, so H is +1/R on an outward sphere of radius R.  h0 = 0 is
# the Willmore energy int H^2 dA: 4*pi for every round sphere (its
# global minimum over closed surfaces) and 2*pi^2 for the Clifford
# torus, the minimum among all tori (Marques-Neves).  h0 != 0 is the
# prescribed-mean-curvature ("spontaneous curvature") variant
# int (H - h0)^2 dA of Helfrich membrane theory.
#
# The gradient is the exact first variation of this DISCRETE energy --
# no finite differences, no frozen terms.  Writing E in the closed form
#
#     E = sum_v [ |g_v|^2/(4 a_v) - h0 sigma_v |g_v| + h0^2 a_v ],
#
# the chain rule needs (dg/dV)^T u and (da/dV)^T s for
#
#     u_v = dE/dg_v = (1/(2 a_v) - h0 sigma_v / |g_v|) g_v,
#     s_v = dE/da_v = h0^2 - H_v^2,
#
# and both transposes are available analytically: dg/dV is the HESSIAN
# of total area (symmetric, so (dg/dV)^T u is the directional
# derivative of the assembled area gradient along the field u), and the
# per-face assembly G_{f,k} = n_hat x e_k / 2 (e_k the edge opposite
# corner k) differentiates in closed form,
#
#     dG_{f,k} = ( dn_hat x e_k + n_hat x de_k ) / 2,
#     dn = sum_k u_k x (p_{k+1} - p_{k+2}),   dn_hat = (I - n n^T) dn / |n|.
#
# sigma is piecewise constant and exact away from the measure-zero set
# g_v . n_v = 0 (the self-test verifies the whole gradient against
# central differences to ~1e-11).  This is the same first variation
# Mem3DG assembles from its three variational vectors; deriving it as
# the area-Hessian action keeps the code a direct transcription of the
# calculus above.  Evolver's star_perp variant (H through the normal
# component of g only) is also implemented, with ITS exact gradient
# chaining through the vertex normals.
#
# Stability, measured rather than assumed: line-searched descent on
# either discrete form is monotone by construction, but both forms
# admit sub-continuum minima reached by TANGENTIAL vertex bunching --
# reparametrization modes that are energy-flat in the continuum but
# energy-lowering discretely (Evolver's manual documents the same
# family of instabilities for its star forms).  The minimizer
# therefore restricts motion to the vertex normal field by default
# (the continuous Willmore flow IS a normal flow) and delegates
# tangential spacing to groom cycles (solver/groom), which measured
# out as the difference between a 48x24 torus flow degenerating past
# the Clifford shape by iteration ~800 and one terminating AT it.
#
# Constraints: enclosed volume and/or total area, held by the same
# Gram-matrix Lagrange projection + damped Newton restore as
# solver/volume (Evolver's fixvol scheme); the line search evaluates
# the energy at RESTORED trial positions, so accepted steps are
# monotone in E on the constraint manifold by construction.  Fixed
# area + volume at reduced volume v < 1 is the classical vesicle
# problem (red-blood-cell discocytes); fixed volume with h0 != 0
# gives "inflated" prescribed-curvature forms.
#
# References:
#   T. J. Willmore, "Note on embedded surfaces", An. Sti. Univ. "Al.
#       I. Cuza" Iasi Sect. I a Mat. 11B (1965), 493-496.
#   W. Helfrich, "Elastic properties of lipid bilayers: theory and
#       possible experiments", Z. Naturforsch. C 28 (1973), 693-703.
#   F. C. Marques and A. Neves, "Min-max theory and the Willmore
#       conjecture", Annals of Mathematics 179(2) (2014), 683-782 --
#       the Clifford torus (ratio sqrt(2), W = 2 pi^2) minimizes W
#       among all tori.
#   C. Zhu, C. T. Lee, P. Rangamani, "Mem3DG: Modeling membrane
#       mechanochemical dynamics in 3D using discrete differential
#       geometry", Biophysical Reports 2(3), 100062 (2022) -- the
#       discrete Helfrich energy and its exact first variation.
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992) -- the star sq_mean_curvature discretization
#       (sqcurve3.c) and the Gram/restore constraint scheme.
#   U. Seifert, K. Berndl, R. Lipowsky, "Shape transformations of
#       vesicles: Phase diagram for spontaneous-curvature and
#       bilayer-coupling models", Phys. Rev. A 44 (1991), 1182-1202 --
#       discocyte/prolate/stomatocyte shapes at reduced volume < 1.

import numpy as np

try:
    from . import descent as _descent
    from . import groom as _groom
    from . import volume as _svol
except ImportError:                      # flat (path-based) headless import
    import descent as _descent           # type: ignore
    import groom as _groom               # type: ignore
    import volume as _svol               # type: ignore


# --------------------------------------------------------------------------
# energy and exact gradient
# --------------------------------------------------------------------------

def _face_frames(V, T):
    """Per-face geometry: opposite edges e (ntri, 3, 3) with e[:, k] =
    p_{k+2} - p_{k+1}, unnormalized normals n, lengths nl, unit normals
    nh, and the corner area-gradients G (ntri, 3, 3),
    G[:, k] = nh x e_k / 2."""
    P = V[T]                             # (ntri, 3corners, 3)
    e = np.stack([P[:, 2] - P[:, 1],
                  P[:, 0] - P[:, 2],
                  P[:, 1] - P[:, 0]], axis=1)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nl = np.maximum(np.linalg.norm(n, axis=1), 1e-300)
    nh = n / nl[:, None]
    G = 0.5 * np.cross(nh[:, None, :], e)
    return P, e, n, nl, nh, G


def vertex_area_data(V, T):
    """(g, a, nvec): per-vertex area gradient g (the cotan identity,
    assembled from the per-face normal-cross-edge form), barycentric
    vertex area a = star/3, and the raw area-weighted vertex normal
    nvec (sum of unnormalized face normals over the star)."""
    _P, _e, n, nl, _nh, G = _face_frames(V, T)
    nverts = len(V)
    g = np.zeros((nverts, 3))
    nvec = np.zeros((nverts, 3))
    a = np.zeros(nverts)
    for k in range(3):
        np.add.at(g, T[:, k], G[:, k])
        np.add.at(nvec, T[:, k], n)
        np.add.at(a, T[:, k], nl / 6.0)  # A_f/3 = |n|/6
    return g, a, nvec


def willmore_energy(V, T, h0=0.0, variant="star"):
    """E = sum_v a_v (H_v - h0)^2 over the closed triangle mesh (V, T).

    variant "star" (default) is the classical H = sigma |g|/(2a) star
    form; "perp" is Evolver's star_perp_sq_mean_curvature, measuring H
    through the component of the area gradient along the area-weighted
    vertex normal, H = (g . n_hat)/(2a) -- naturally signed.  Measured
    honestly: as MINIMIZATION energies both admit sub-continuum minima
    reached by tangential vertex bunching (a 48x24 torus flow free to
    move tangentially reaches the Clifford shape and then degenerates
    past it), and perp is WORSE there, since tangential components of
    g cost it nothing, making crumpling free.  The effective cure is
    the minimizer's normal-only motion (see minimize_willmore), under
    which both variants are stable and agree to ~0.1%; star keeps the
    smaller Mobius-invariance residual, hence the default."""
    g, a, nvec = vertex_area_data(V, T)
    a = np.maximum(a, 1e-300)
    if variant == "perp":
        nl = np.maximum(np.linalg.norm(nvec, axis=1), 1e-300)
        gperp = np.einsum('ij,ij->i', g, nvec) / nl
        E = float(np.sum(gperp * gperp / (4.0 * a)))
        if h0:
            E += float(np.sum(-h0 * gperp + h0 * h0 * a))
        return E
    if variant != "star":
        raise ValueError(f"unknown willmore variant {variant!r}")
    gn = np.linalg.norm(g, axis=1)
    E = float(np.sum(gn * gn / (4.0 * a)))
    if h0:
        sigma = np.sign(np.einsum('ij,ij->i', g, nvec))
        sigma[sigma == 0.0] = 1.0
        E += float(np.sum(-h0 * sigma * gn + h0 * h0 * a))
    return E


def area_hessian_apply(V, T, U):
    """(Hess total-area) applied to the vertex field U (n, 3):
    the directional derivative of the assembled area gradient along U,
    differentiated in closed form per face."""
    P, e, n, nl, nh, _G = _face_frames(V, T)
    Uc = U[T]                            # (ntri, 3corners, 3)
    # dn = sum_k u_k x (p_{k+1} - p_{k+2}) = sum_k u_k x (-e_k)
    dn = -np.sum(np.cross(Uc, e), axis=1)
    dnh = (dn - nh * np.einsum('ij,ij->i', nh, dn)[:, None]) / nl[:, None]
    # de[:, k] = u_{k+2} - u_{k+1}
    de = np.stack([Uc[:, 2] - Uc[:, 1],
                   Uc[:, 0] - Uc[:, 2],
                   Uc[:, 1] - Uc[:, 0]], axis=1)
    dG = 0.5 * (np.cross(dnh[:, None, :], e) + np.cross(nh[:, None, :], de))
    out = np.zeros_like(U)
    for k in range(3):
        np.add.at(out, T[:, k], dG[:, k])
    return out


def willmore_gradient(V, T, h0=0.0, variant="star"):
    """(E, grad): the energy and its exact gradient (n, 3).

    Both variants chain-rule through every geometric ingredient --
    the area gradient g (via the area-Hessian action), the barycentric
    areas a, and (perp variant) the area-weighted vertex normals n_v --
    so the returned gradient is the exact first variation of the
    discrete energy, verified against central differences to ~1e-11."""
    P, e, n, nl, nh, G = _face_frames(V, T)
    nverts = len(V)
    g = np.zeros((nverts, 3))
    nvec = np.zeros((nverts, 3))
    a = np.zeros(nverts)
    for k in range(3):
        np.add.at(g, T[:, k], G[:, k])
        np.add.at(nvec, T[:, k], n)
        np.add.at(a, T[:, k], nl / 6.0)
    a = np.maximum(a, 1e-300)

    if variant == "perp":
        nvl = np.maximum(np.linalg.norm(nvec, axis=1), 1e-300)
        nhat = nvec / nvl[:, None]
        gperp = np.einsum('ij,ij->i', g, nhat)
        Hd = gperp / (2.0 * a) - h0      # H_v - h0
        E = float(np.sum(gperp * gperp / (4.0 * a)))
        if h0:
            E += float(np.sum(-h0 * gperp + h0 * h0 * a))
        # dE/dg_v = (H - h0) nhat_v
        U = Hd[:, None] * nhat
        grad = area_hessian_apply(V, T, U)
        # dE/da_v = h0^2 - H^2 = -(H - h0)(H + h0)
        s = -Hd * (gperp / (2.0 * a) + h0)
        sbar = (s[T[:, 0]] + s[T[:, 1]] + s[T[:, 2]]) / 3.0
        for k in range(3):
            np.add.at(grad, T[:, k], sbar[:, None] * G[:, k])
        # dE/d(nvec_v) = (I - nhat nhat^T) (H - h0) g / |nvec|; nvec_v
        # sums the unnormalized face normals over the star, and
        # d n_f = sum_k delta_k x (-e_k), so the transpose scatters
        # M_f x e_k to corner k with M_f the per-face sum of dE/dnvec.
        m = (Hd[:, None] * (g - gperp[:, None] * nhat)) / nvl[:, None]
        M = m[T[:, 0]] + m[T[:, 1]] + m[T[:, 2]]
        for k in range(3):
            np.add.at(grad, T[:, k], np.cross(M, e[:, k]))
        return E, grad

    if variant != "star":
        raise ValueError(f"unknown willmore variant {variant!r}")
    gn = np.linalg.norm(g, axis=1)
    H2 = gn * gn / (4.0 * a * a)         # H_v^2
    E = float(np.sum(gn * gn / (4.0 * a)))
    w = 1.0 / (2.0 * a)                  # u_v = w_v g_v
    s = -H2                              # s_v = h0^2 - H_v^2
    if h0:
        sigma = np.sign(np.einsum('ij,ij->i', g, nvec))
        sigma[sigma == 0.0] = 1.0
        E += float(np.sum(-h0 * sigma * gn + h0 * h0 * a))
        w = w - h0 * sigma / np.maximum(gn, 1e-300)
        s = s + h0 * h0
    U = w[:, None] * g
    grad = area_hessian_apply(V, T, U)
    # + sum_f mean(s over corners) * G_{f,k} scattered to corner k
    sbar = (s[T[:, 0]] + s[T[:, 1]] + s[T[:, 2]]) / 3.0
    for k in range(3):
        np.add.at(grad, T[:, k], sbar[:, None] * G[:, k])
    return E, grad


# --------------------------------------------------------------------------
# constrained minimization
# --------------------------------------------------------------------------

def _closed_labels(T):
    lab = np.zeros((len(T), 2), dtype=np.int64)
    lab[:, 1] = 1
    return lab


def _restore_constraints(V, T, labels, vol_target, area_target,
                         max_rounds=12, tol=1e-12):
    """Damped Newton on the active constraint values (volume and/or
    area), in place; the same Gram scheme as solver.volume's
    restore_volumes generalized to mixed constraint rows.  Returns the
    worst relative deficit after."""
    if vol_target is None and area_target is None:
        return 0.0
    for _ in range(max_rounds):
        rows, deficit, scale = [], [], []
        if vol_target is not None:
            vol = float(_svol.region_volumes(V, T, labels, 1)[0])
            rows.append(_svol.volume_gradients(V, T, labels, 1)[0])
            deficit.append(vol_target - vol)
            scale.append(max(abs(vol_target), 1e-300))
        if area_target is not None:
            area = _svol.mesh_area(V, T)
            g, _a, _n = vertex_area_data(V, T)
            rows.append(g)
            deficit.append(area_target - area)
            scale.append(max(abs(area_target), 1e-300))
        rows = np.stack(rows)
        deficit = np.asarray(deficit)
        rel = float(np.max(np.abs(deficit) / np.asarray(scale)))
        if rel < tol:
            return rel
        A = np.einsum('inj,mnj->im', rows, rows)
        try:
            mu = np.linalg.solve(A, deficit)
        except np.linalg.LinAlgError:
            mu = np.linalg.lstsq(A, deficit, rcond=None)[0]
        step = np.einsum('i,inj->nj', mu, rows)
        s = 1.0
        for _try in range(8):
            V_n = V + s * step
            worst = []
            if vol_target is not None:
                worst.append(abs(vol_target - float(_svol.region_volumes(
                    V_n, T, labels, 1)[0])) / scale[0])
            if area_target is not None:
                worst.append(abs(area_target - _svol.mesh_area(V_n, T))
                             / scale[-1])
            if max(worst) < rel:
                V[:] = V_n
                break
            s *= 0.5
        else:
            return rel                   # no damping factor improves
    rows_rel = []
    if vol_target is not None:
        rows_rel.append(abs(vol_target - float(_svol.region_volumes(
            V, T, labels, 1)[0])) / scale[0])
    if area_target is not None:
        rows_rel.append(abs(area_target - _svol.mesh_area(V, T))
                        / scale[-1])
    return float(max(rows_rel))


def minimize_willmore(V, T, h0=0.0, iters=300, vol_target=None,
                      area_target=None, groom_every=0, groom_smooth=0.25,
                      lbfgs_m=8, h0_seed="laplacian", step_cap=1.0,
                      e_tol=1e-11, variant="star", normal_only=True):
    """Line-searched L-BFGS descent on E = int (H - h0)^2 dA for the
    CLOSED, outward-oriented triangle mesh (V, T).  V is modified in
    place; T is modified in place only when grooming flips edges.

    vol_target / area_target (None = unconstrained) hold the enclosed
    volume / total area by Gram-projected directions plus a damped
    Newton restore; the line search evaluates E at RESTORED trial
    positions, so every accepted step is monotone in E on the
    constraint manifold by construction.  With h0 = 0 the energy is
    exactly scale-invariant, so the unconstrained flow has a flat
    scale direction (harmless to the quasi-Newton method).

    groom_every > 0 runs a groom cycle (Delaunay flips + tangential
    smoothing) before iterations g, 2g, ...: the Willmore energy is
    parametrization-invariant in the continuum, so the discrete soft
    modes are exactly the tangential vertex motions grooming
    regularizes; each cycle resets the L-BFGS history and re-restores
    the constraints.

    h0_seed: "laplacian" seeds the inverse Hessian with the cotan
    (L + eps D)^-1 solve of solver.descent.LaplacianH0 -- one factor
    of k^2 of the fourth-order bending stiffness, making steps far
    less resolution-bound -- or "identity" for the standard scaled
    identity.  step_cap bounds the largest per-step vertex move at
    cap * mean edge length.

    normal_only=True (default) restricts every step to the vertex
    NORMAL field: the continuous Willmore flow is a normal flow, and
    tangential vertex motion is pure reparametrization -- the soft
    mode a quasi-Newton method otherwise accelerates until vertices
    bunch and the discrete energy drops below its continuum value
    through mesh degeneration (measured: an unrestricted 48x24 torus
    flow reaches the Clifford shape by iteration ~300 and then
    degenerates by iteration ~800; the normal-only flow holds it).
    Tangential spacing is managed by the groom cycles instead.

    Returns a dict: E, E0, iters_run, grooms_run, rise_max (the worst
    accepted energy rise outside groom iterations; <= 0 up to roundoff
    by construction), drift_max (worst post-restore constraint
    residual), history."""
    V = np.asarray(V)
    T = np.asarray(T)
    labels = _closed_labels(T)
    constrained = vol_target is not None or area_target is not None

    def _restore(Varr):
        return _restore_constraints(Varr, T, labels, vol_target,
                                    area_target)

    def _energy(x):
        Vc = np.array(x, float)
        _restore(Vc)
        return willmore_energy(Vc, T, h0, variant)

    lb = _descent.LBFGS(m=lbfgs_m)
    if h0_seed == "laplacian":
        seed = _descent.LaplacianH0(eps=1e-3, tol=1e-2, max_iters=100)
    elif h0_seed == "identity":
        seed = None
    else:
        raise ValueError(f"unknown h0_seed {h0_seed!r}")

    drift_max = _restore(V)
    E_prev, g_prev = willmore_gradient(V, T, h0, variant)
    E0 = E_prev
    history = []
    grooms_run = 0
    rise_max = 0.0
    flat_streak = 0
    x_prev = None
    gh_prev = None
    it = 0
    for it in range(1, iters + 1):
        groomed = False
        if groom_every and (it - 1) and (it - 1) % groom_every == 0:
            _groom.groom(V, T, smooth_lam=groom_smooth)
            drift_max = max(drift_max, _restore(V))
            grooms_run += 1
            groomed = True
            lb.reset()
            x_prev = None
            E_prev, g_prev = willmore_gradient(V, T, h0, variant)
        gh = g_prev
        if constrained:
            rows = []
            if vol_target is not None:
                rows.append(_svol.volume_gradients(V, T, labels, 1)[0])
            if area_target is not None:
                rows.append(vertex_area_data(V, T)[0])
            rows = np.stack(rows)
            gh, _lam = _svol.project_velocity(g_prev, rows)
        if x_prev is not None:
            lb.push((V - x_prev).ravel(), (gh - gh_prev).ravel())
        gmax = float(np.max(np.linalg.norm(gh, axis=1)))
        if gmax < 1e-300:
            break
        if seed is not None:
            seed.update(V, T)
        if normal_only:
            nv = vertex_area_data(V, T)[2]
            nv = nv / np.maximum(np.linalg.norm(nv, axis=1,
                                                keepdims=True), 1e-300)

        def _project_dir(flat):
            dd = flat.reshape(len(V), 3)
            if normal_only:
                dd = np.einsum('ij,ij->i', dd, nv)[:, None] * nv
            if constrained:
                dd, _ = _svol.project_velocity(dd, rows)
            return dd

        d = _project_dir(-lb.direction(gh.ravel(), h0=seed))
        slope = float(np.einsum('nj,nj->', gh, d))
        if not (slope < 0.0):
            lb.reset()
            d = _project_dir(-(seed(gh.ravel()) if seed is not None
                               else gh.ravel().copy()))
            slope = float(np.einsum('nj,nj->', gh, d))
        Lmean = float(np.mean(np.linalg.norm(
            V[T[:, 1]] - V[T[:, 0]], axis=1)))
        dmax = float(np.max(np.linalg.norm(d, axis=1)))
        s_max = step_cap * Lmean / max(dmax, 1e-300)
        x1, s, E1, nev = _descent.armijo_backtrack(
            _energy, V, d, slope, s0=min(1.0, s_max), E0=E_prev,
            s_max=s_max)
        if s == 0.0:
            lb.reset()
            d = _project_dir(-(seed(gh.ravel()) if seed is not None
                               else gh.ravel().copy()))
            dmax = float(np.max(np.linalg.norm(d, axis=1)))
            s_max = step_cap * Lmean / max(dmax, 1e-300)
            x1, s, E1, nev2 = _descent.parabola_line_search(
                _energy, V, d, min(1.0, s_max), s_max=s_max)
            nev += nev2
        if s == 0.0:
            break                        # no downhill scale: converged
        x_prev = V.copy()
        gh_prev = gh
        V[:] = x1
        drift_max = max(drift_max, _restore(V))
        E, g_prev = willmore_gradient(V, T, h0, variant)
        if not groomed:
            rise_max = max(rise_max, (E - E_prev) / max(abs(E_prev),
                                                        1e-300))
        history.append({"it": it, "E": E, "s": s, "n_evals": nev,
                        "groomed": groomed})
        if abs(E - E_prev) < e_tol * abs(E_prev):
            flat_streak += 1
            if flat_streak >= 3:
                E_prev = E
                break
        else:
            flat_streak = 0
        E_prev = E
    return {"E": E_prev, "E0": E0, "iters_run": it,
            "grooms_run": grooms_run, "rise_max": rise_max,
            "drift_max": drift_max, "history": history}


# --------------------------------------------------------------------------
# seed meshes and measurement helpers
# --------------------------------------------------------------------------

def torus_mesh(nu=48, nv=24, R=2.0, r=1.0):
    """Outward-oriented triangulated torus of revolution about z:
    (V, T) with V[(i*nv + j)] at angles u_i (major), v_j (tube)."""
    u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    rho = R + r * np.cos(vv)
    V = np.stack([rho * np.cos(uu), rho * np.sin(uu),
                  r * np.sin(vv)], axis=-1).reshape(-1, 3)
    T = []
    for i in range(nu):
        i1 = (i + 1) % nu
        for j in range(nv):
            j1 = (j + 1) % nv
            a = i * nv + j
            b = i1 * nv + j
            c = i1 * nv + j1
            d = i * nv + j1
            T.append([a, b, c])
            T.append([a, c, d])
    return V, np.asarray(T, dtype=np.int64)


def mobius_invert(V, center, radius=1.0):
    """Sphere inversion x -> c + R^2 (x - c)/|x - c|^2 -- a Mobius
    transformation of R^3; the continuous Willmore energy of a closed
    surface not passing through c is invariant under it."""
    X = V - np.asarray(center, float)
    r2 = np.einsum('ij,ij->i', X, X)
    return np.asarray(center, float) + (radius * radius / r2)[:, None] * X


def fit_torus_of_revolution(V):
    """Least-squares torus-of-revolution fit (axis = z through the
    centroid): returns (R, r_mean, r_cv, ratio) where ratio = R/r_mean
    (sqrt(2) for the Clifford shape) and r_cv the relative spread of
    the tube radius (0 for an exact torus of revolution)."""
    P = V - V.mean(axis=0)
    rho = np.hypot(P[:, 0], P[:, 1])
    z = P[:, 2]

    def spread(R):
        rr = np.hypot(rho - R, z)
        return float(np.std(rr)), float(np.mean(rr))

    lo, hi = 0.25 * float(np.mean(rho)), 1.5 * float(np.max(rho))
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = lo, hi
    for _ in range(80):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        if spread(c)[0] < spread(d)[0]:
            b = d
        else:
            a = c
    R = 0.5 * (a + b)
    sd, rm = spread(R)
    return R, rm, sd / max(rm, 1e-300), R / max(rm, 1e-300)


def torus_willmore_analytic(t):
    """int H^2 dA of the torus of revolution with ratio t = R/r:
    pi^2 t^2 / sqrt(t^2 - 1); minimized at t = sqrt(2) with 2 pi^2."""
    return np.pi * np.pi * t * t / np.sqrt(t * t - 1.0)


def _icosphere(subdiv):
    try:
        from ..surfaces.primitives import icosphere
    except ImportError:
        from surfaces.primitives import icosphere    # type: ignore
    SV, SF = icosphere(subdiv, 'per_level')
    return np.asarray(SV, float), np.asarray(SF, dtype=np.int64)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(5)
    ok = True
    fourpi = 4.0 * np.pi
    twopi2 = 2.0 * np.pi * np.pi

    # --- per-face area-gradient assembly == the cotan identity --------
    SV, SF = _icosphere(1)
    g_face, a, nvec = vertex_area_data(SV, SF)
    g_cot = _svol.area_gradient(SV, SF, cotan_mode="raw")
    err = float(np.max(np.abs(g_face - g_cot))) \
        / max(float(np.max(np.abs(g_cot))), 1e-300)
    good = err < 1e-12
    ok &= good
    print(f"willmore: face-assembled area gradient == cotan identity "
          f"(rel {err:.1e}) {'OK' if good else 'FAIL'}")

    # --- area Hessian: symmetric, and matches FD of the area gradient -
    Vh = SV + 0.05 * rng.normal(size=SV.shape)
    U1 = rng.normal(size=Vh.shape)
    U2 = rng.normal(size=Vh.shape)
    H1 = area_hessian_apply(Vh, SF, U1)
    H2 = area_hessian_apply(Vh, SF, U2)
    sym = abs(float(np.sum(U2 * H1) - np.sum(U1 * H2))) \
        / max(abs(float(np.sum(U2 * H1))), 1e-300)
    h = 1e-6
    fd = (vertex_area_data(Vh + h * U1, SF)[0]
          - vertex_area_data(Vh - h * U1, SF)[0]) / (2.0 * h)
    hess_err = float(np.max(np.abs(H1 - fd))) \
        / max(float(np.max(np.abs(fd))), 1e-300)
    good = sym < 1e-11 and hess_err < 1e-7
    ok &= good
    print(f"willmore: area Hessian symmetric ({sym:.1e}), vs FD "
          f"{hess_err:.1e} {'OK' if good else 'FAIL'}")

    # --- gradient vs central differences, both variants, h0 = 0/!=0 ---
    for variant in ("star", "perp"):
        for h0 in (0.0, 1.3):
            E, grad = willmore_gradient(Vh, SF, h0, variant)
            d = rng.normal(size=Vh.shape)
            d /= float(np.max(np.abs(d)))
            h = 3e-6
            fd = (willmore_energy(Vh + h * d, SF, h0, variant)
                  - willmore_energy(Vh - h * d, SF, h0, variant)) \
                / (2.0 * h)
            an = float(np.sum(grad * d))
            err = abs(fd - an) / max(abs(fd), 1e-300)
            good = err < 5e-9
            ok &= good
            print(f"willmore: {variant} gradient vs FD (h0={h0}) "
                  f"rel err {err:.2e} {'OK' if good else 'FAIL'}")

    # --- round sphere: E -> 4 pi under refinement ---------------------
    errs = []
    for sub in (2, 3):
        SVs, SFs = _icosphere(sub)
        errs.append(abs(willmore_energy(SVs, SFs) - fourpi) / fourpi)
    good = errs[1] < 0.5 * errs[0] and errs[1] < 5e-3
    ok &= good
    print(f"willmore: sphere E/4pi - 1 = {errs[0]:.2e} -> {errs[1]:.2e} "
          f"under refinement {'OK' if good else 'FAIL'}")

    # --- torus of revolution vs the analytic pi^2 t^2/sqrt(t^2-1) -----
    errs = []
    for (nu, nv) in ((32, 16), (64, 32)):
        Vt, Tt = torus_mesh(nu, nv, R=3.0, r=1.0)
        Ean = torus_willmore_analytic(3.0)
        errs.append(abs(willmore_energy(Vt, Tt) - Ean) / Ean)
    good = errs[1] < 0.35 * errs[0] and errs[1] < 2e-2
    ok &= good
    print(f"willmore: torus t=3 E vs analytic rel {errs[0]:.2e} -> "
          f"{errs[1]:.2e} under refinement {'OK' if good else 'FAIL'}")

    # --- Mobius invariance: E unchanged under sphere inversion up to
    # discretization error, shrinking under refinement -----------------
    resids = []
    for (nu, nv) in ((32, 16), (64, 32)):
        Vt, Tt = torus_mesh(nu, nv, R=2.0, r=1.0)
        E1 = willmore_energy(Vt, Tt)
        Vm = mobius_invert(Vt, center=[4.5, 0.0, 0.0], radius=2.0)
        E2 = willmore_energy(Vm, Tt)
        resids.append(abs(E2 - E1) / E1)
    good = resids[1] < 0.5 * resids[0] and resids[1] < 2e-2
    ok &= good
    print(f"willmore: Mobius invariance residual {resids[0]:.2e} -> "
          f"{resids[1]:.2e} under refinement {'OK' if good else 'FAIL'}")

    # --- the headline: a fat torus flows to the Clifford shape.  The
    # normal-only flow TERMINATES at its stationary point (no downhill
    # scale left), so the iteration budget is an upper bound, not a
    # tuning knob ------------------------------------------------------
    Vt, Tt = torus_mesh(24, 12, R=3.0, r=1.0)
    info = minimize_willmore(Vt, Tt, iters=300, groom_every=10,
                             step_cap=0.5)
    R, rm, rcv, ratio = fit_torus_of_revolution(Vt)
    e_rel = (info["E"] - twopi2) / twopi2
    good = (abs(ratio - np.sqrt(2.0)) < 0.08 and abs(e_rel) < 0.05
            and info["rise_max"] <= 1e-12 and rcv < 0.05
            and info["iters_run"] < 300)
    ok &= good
    print(f"willmore: coarse torus flow ratio {ratio:.4f} (sqrt2 = "
          f"{np.sqrt(2):.4f}), E/2pi^2 - 1 = {e_rel:+.3f}, tube cv "
          f"{rcv:.3f}, terminated at it {info['iters_run']}, max rise "
          f"{info['rise_max']:.1e} {'OK' if good else 'FAIL'}")

    # --- vesicle smoke: reduced volume 0.65 at fixed area+volume gives
    # an oblate (discocyte-family) shape with E > 4 pi, constraints
    # tight, monotone -------------------------------------------------
    SVv, SFv = _icosphere(2)
    area0 = _svol.mesh_area(SVv, SFv)
    vol0 = float(_svol.region_volumes(SVv, SFv,
                                      _closed_labels(SFv), 1)[0])
    red = 0.65
    Vv = SVv * np.array([1.0, 1.0, 0.6])       # oblate seed
    info = minimize_willmore(Vv, SFv, iters=150, vol_target=red * vol0,
                             area_target=area0, groom_every=10)
    ext = Vv.max(axis=0) - Vv.min(axis=0)
    oblate = ext[2] / (0.5 * (ext[0] + ext[1]))
    good = (info["E"] > fourpi and info["drift_max"] < 1e-9
            and info["rise_max"] <= 1e-12 and oblate < 0.75)
    ok &= good
    print(f"willmore: vesicle v=0.65 E={info['E']:.3f} (> 4pi = "
          f"{fourpi:.3f}), z/xy extent {oblate:.3f}, drift "
          f"{info['drift_max']:.1e}, max rise {info['rise_max']:.1e} "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("solver.willmore self-test failed")
