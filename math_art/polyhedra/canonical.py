# Canonical forms: spherize, canonicalize, biscribe.
#
# Part of the Math Art polyhedra engine (`math_art/polyhedra/`).  Python
# and stdlib only -- no `bpy` -- so the engine imports and self-tests
# headlessly.
#
# A combinatorial polyhedron has many geometric realisations; these pick
# distinguished ones.
#
#   canonicalize  the CANONICAL FORM: every edge tangent to a common
#                 sphere, the tangent points averaging to its centre,
#                 and the whole thing symmetric.  Reached by alternately
#                 pulling edges towards tangency and recentring the
#                 tangent points -- the "conjugate-based" relaxation.
#                 It exists and is unique up to Mobius transformation
#                 for every polyhedral graph (Steinitz).
#   biscribe      the BISCRIBED form: circumscribed and inscribed
#                 spheres concentric, so vertices lie on one sphere and
#                 faces are tangent to another.
#   spherize      the cheap approximation: push vertices out to a common
#                 radius.  Not canonical, but often what is wanted.
#
# References:
# - E. Steinitz, "Uber isoperimetrische Probleme bei konvexen
#   Polyedern", Journal fur die reine und angewandte Mathematik 159,
#   1928 -- existence of the canonical form.
# - G. W. Hart, "Calculating Canonical Polyhedra", Mathematica in
#   Education and Research 6, 1997, pp. 5-10 -- the relaxation used here.
# - P. W. Messer, "Closed-Form Expressions for Uniform Polyhedra and
#   Their Duals", Discrete and Computational Geometry 27, 2002.
# - A. Rossiter, Antiprism (antiprism.com), `canonical` -- the
#   circle-packing / "ambo" canonicalization reimplemented in
#   canonicalize_ambo.  Includes ideas and algorithms by George W. Hart.
# - O. Schramm, "How to cage an egg", Inventiones Mathematicae 107,
#   1992 -- the midsphere as two orthogonal circle packings, the
#   mathematics the ambo change of variables exposes.

import math

import numpy as np

try:
    from ..solver import descent as _descent
except ImportError:                      # flat (path-based) headless import
    try:
        from solver import descent as _descent
    except ImportError:
        _descent = None


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def spherize(V, F, iters=0):
    c = [sum(v[i] for v in V) / len(V) for i in range(3)]
    out = []
    for v in V:
        d = _sub(v, c)
        ln = math.sqrt(sum(x * x for x in d)) or 1.0
        out.append([d[i] / ln for i in range(3)])
    return out


def canonicalize(V, F, iters=200, lam_t=0.3, lam_p=0.5, trace=None,
                 adaptive=True):
    """George Hart's canonicalization: iterate edge-tangency to the unit
    sphere, recentering on the edge tangency points, and face
    planarization, until converged (or iters).  `trace`, if given, is
    called as trace(it, P) after every iteration (instrumentation for
    the benchmark harness; does not alter the numerics).

    adaptive=True scales both gains by an antiprism-style controller
    (grow on residual improvement, shrink on regression -- Rossiter's
    `canonical`, after Hart), measured in tests/bench to reach a
    ~100x smaller tangency spread on the solids where the fixed gains
    stall against the iteration cap."""
    if np is None:
        return V
    P = np.array(V, dtype=np.float64)
    P -= P.mean(axis=0)
    P /= np.mean(np.linalg.norm(P, axis=1))
    edges = set()
    for f in F:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            edges.add((min(a, b), max(a, b)))
    E = np.array(sorted(edges))
    Fi = [np.array(f) for f in F]
    gain = None
    if adaptive and _descent is not None:
        cap = 1.0 / max(lam_t, lam_p)
        gain = _descent.AdaptiveGain(gain=1.0, up=1.01, down=0.995,
                                     lo=0.05, hi=cap)
    for it in range(iters):
        prev = P.copy()
        g = gain.gain if gain is not None else 1.0
        A = P[E[:, 0]]
        B = P[E[:, 1]]
        d = B - A
        t = -np.einsum('ij,ij->i', A, d) / np.maximum(
            np.einsum('ij,ij->i', d, d), 1e-12)
        t = np.clip(t, 0.0, 1.0)
        C = A + t[:, None] * d
        # recentre on the edge tangency points (Hart), then push edges
        # to tangency with the unit sphere
        P -= C.mean(axis=0)
        C -= C.mean(axis=0)
        cl = np.linalg.norm(C, axis=1, keepdims=True)
        corr = C / np.maximum(cl, 1e-9) * (1.0 - cl)
        adj = np.zeros_like(P)
        cnt = np.zeros(len(P))
        np.add.at(adj, E[:, 0], corr)
        np.add.at(adj, E[:, 1], corr)
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
        P += (lam_t * g) * adj / np.maximum(cnt, 1)[:, None]
        maxdist = 0.0
        for f in Fi:
            Q = P[f]
            c = Q.mean(axis=0)
            Qc = Q - c
            nrm = np.zeros(3)
            for i in range(len(f)):
                p = Qc[i]
                q = Qc[(i + 1) % len(f)]
                nrm += np.cross(p, q)
            ln = np.linalg.norm(nrm)
            if ln < 1e-12:
                continue
            nrm /= ln
            dist = Qc @ nrm
            maxdist = max(maxdist, float(np.max(np.abs(dist))))
            P[f] -= (lam_p * g) * dist[:, None] * nrm
        if gain is not None:
            # residual: tangency spread of the entering iterate plus the
            # worst out-of-plane deviation seen this pass
            gain.update(float(cl.max() - cl.min()) + maxdist)
        if trace is not None:
            trace(it, P)
        if np.max(np.linalg.norm(P - prev, axis=1)) < 1e-7:
            break
    return [list(map(float, p)) for p in P]


def _oriented_faces(V, F):
    """A copy of F wound consistently (each shared edge traversed once
    in each direction across its two faces) and outward (positive
    enclosed volume).  Raises ValueError if the map is not a closed
    orientable 2-manifold (an edge not shared by exactly two faces, or
    an inconsistent winding reachable two ways)."""
    F = [list(f) for f in F]
    # undirected edge -> list of (face, direction as traversed)
    e2f = {}
    for fi, f in enumerate(F):
        m = len(f)
        if m < 3:
            raise ValueError("face with fewer than 3 vertices")
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            e2f.setdefault((min(a, b), max(a, b)), []).append(fi)
    for key, fs in e2f.items():
        if len(fs) != 2:
            raise ValueError(f"edge {key} shared by {len(fs)} faces "
                             f"(need a closed 2-manifold)")
    # BFS across shared edges, flipping to make traversals opposite
    seen = [False] * len(F)
    for start in range(len(F)):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        while stack:
            fi = stack.pop()
            dirs = {}
            f = F[fi]
            for i in range(len(f)):
                dirs[(f[i], f[(i + 1) % len(f)])] = True
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                key = (min(a, b), max(a, b))
                other = [g for g in e2f[key] if g != fi]
                gi = other[0] if other else fi
                g = F[gi]
                same = any(g[j] == a and g[(j + 1) % len(g)] == b
                           for j in range(len(g)))
                if not seen[gi]:
                    if same:
                        F[gi] = list(reversed(g))
                    seen[gi] = True
                    stack.append(gi)
                elif same and gi != fi:
                    raise ValueError("non-orientable face winding")
    # outward: positive signed volume (fan-triangulated)
    P = np.asarray(V, dtype=np.float64)
    vol = 0.0
    for f in F:
        for i in range(1, len(f) - 1):
            vol += float(np.linalg.det(
                np.stack([P[f[0]], P[f[i]], P[f[i + 1]]])))
    if vol < 0.0:
        F = [list(reversed(f)) for f in F]
    return F


def _ambo_structure(F):
    """Combinatorics of the ambo (rectified) polyhedron of the oriented
    face list F: one ambo vertex per base edge, ambo faces = the base
    vertex figures (first, in base-vertex order) then the base faces.
    Returns (E, faces_flat, faces_next, starts, counts, VF, VFig,
    n_vfaces) where E is the (nE, 2) base-edge array in ambo-vertex
    order, VF the (nE, 4) cyclic incident-face array (alternating
    vertex-figure face / base face), and VFig the (nE, 4) cyclic
    neighbouring-ambo-vertex array matching VF's cycle."""
    nV = 1 + max(max(f) for f in F)
    eidx = {}
    E = []
    de_face = {}                 # directed edge -> face index
    de_next = {}                 # directed edge -> next directed edge
    de_prev = {}
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            key = (min(a, b), max(a, b))
            if key not in eidx:
                eidx[key] = len(E)
                E.append(key)
            if (a, b) in de_face:
                raise ValueError("inconsistent winding (directed edge "
                                 "repeated)")
            de_face[(a, b)] = fi
            de_next[(a, b)] = (b, f[(i + 2) % m])
            de_prev[(a, b)] = (f[(i - 1) % m], a)
    if len(de_face) != 2 * len(E):
        raise ValueError("not a closed surface")

    def _eid(a, b):
        return eidx[(min(a, b), max(a, b))]

    # vertex figures: cycle of neighbours around each base vertex, by
    # the half-edge walk  (v, w) -> (v, u)  with (u, v) preceding
    # (v, w) in its face
    vstart = {}
    for (a, b) in de_face:
        vstart.setdefault(a, b)
    vfaces = []
    for v in range(nV):
        w0 = vstart.get(v)
        if w0 is None:
            raise ValueError(f"isolated vertex {v}")
        cyc = []
        w = w0
        while True:
            cyc.append(_eid(v, w))
            u, _ = de_prev[(v, w)]
            w = u
            if w == w0:
                break
            if len(cyc) > len(E):
                raise ValueError("vertex-figure walk failed to close")
        if len(cyc) < 3:
            raise ValueError(f"vertex {v} has degree {len(cyc)} < 3")
        vfaces.append(cyc)
    bfaces = [[_eid(f[i], f[(i + 1) % len(f)]) for i in range(len(f))]
              for f in F]
    afaces = vfaces + bfaces     # vertex figures FIRST (recovery order)

    # flat face arrays for reduceat-based centroids and Newell normals
    counts = np.array([len(f) for f in afaces])
    starts = np.zeros(len(afaces), dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    faces_flat = np.array([v for f in afaces for v in f], dtype=np.int64)
    faces_next = np.array([f[(i + 1) % len(f)] for f in afaces
                           for i in range(len(f))], dtype=np.int64)

    # per ambo vertex m(a,b): incident faces in cyclic order
    #   [base face containing a->b, vertex figure of b,
    #    base face containing b->a, vertex figure of a]
    # and the matching cyclic neighbour ring
    #   [m(next edge of f1 at b), m(edge entering b in f2),
    #    m(next edge of f2 at a), m(edge entering a in f1)]
    nE = len(E)
    VF = np.zeros((nE, 4), dtype=np.int64)
    VFig = np.zeros((nE, 4), dtype=np.int64)
    for (a, b), fi in de_face.items():
        k = _eid(a, b)
        if (min(a, b), max(a, b)) != (a, b):
            continue             # handle each edge once, via (a<b) dir
        f2 = de_face[(b, a)]
        VF[k] = (nV + fi, b, nV + f2, a)
        VFig[k] = (_eid(*de_next[(a, b)]), _eid(*de_prev[(b, a)]),
                   _eid(*de_next[(b, a)]), _eid(*de_prev[(a, b)]))
    return (np.array(E, dtype=np.int64), faces_flat, faces_next, starts,
            counts, VF, VFig, nV)


def _ambo_face_planes(P, faces_flat, faces_next, starts, counts):
    """Centroids and unit Newell normals of every ambo face, by
    contiguous reduceat (the faces are stored flat and in order)."""
    cent = np.add.reduceat(P[faces_flat], starts, axis=0) \
        / counts[:, None]
    nrm = np.add.reduceat(np.cross(P[faces_flat], P[faces_next]),
                          starts, axis=0)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-30)
    return cent, nrm


def canonicalize_ambo(V, F, iters=10000, factor=0.01, factor_max=0.5,
                      tol=1e-12, point_type="centroid", trace=None):
    """Circle-packing ("ambo") canonicalization, reimplemented from the
    method Adrian Rossiter devised for Antiprism's `canonical` program
    (base/canonical.cc, make_planar_unit; antiprism.com).  Includes
    ideas and algorithms by George W. Hart.

    Change of variables: iterate on the edge-tangency points (the ambo
    polyhedron -- one vertex per base edge, each with exactly four
    incident faces alternating between base-face circles and
    vertex-figure circles of the midsphere's two orthogonal circle
    packings; Schramm 1992).  Two of Hart's three competing soft nudges
    become EXACT projections in these variables -- unit-sphere tangency
    by renormalizing every ambo vertex each iteration, the tangency
    centroid by exact centroid subtraction -- leaving only face
    coplanarity and circle orthogonality as relaxed residuals, driven
    by a small adaptive gain (x1.01 on improvement, x0.995 on
    regression, capped at factor_max; Rossiter's defaults 0.01 / 0.5).
    The base is recovered at the end by polar reciprocation of the ambo
    face planes in the unit sphere (vertex-figure faces -> base
    vertices), so every base edge is tangent to the unit midsphere.

    point_type: "centroid" starts the ambo vertices at edge midpoints
    (Antiprism's default, better for general input); "nearpoint" at the
    edge points nearest the origin (better for near-canonical input).
    trace(it, P), if given, receives the ambo iterate (instrumentation
    for the benchmark harness; does not alter the numerics).  Raises
    ValueError on input that is not a closed orientable polyhedron with
    all vertex degrees >= 3.

    Measured in tests/bench (see
    research/plans/antiprism-canonicalization-plan.md): converges to
    machine-precision tangency spreads (~5e-16) in 4-8x LESS wall time
    than 400 iterations of `canonicalize` on every catalog solid --
    at equal wall time it is 2.6e5x-8.8e13x tighter -- because the
    whole iteration vectorizes over a fixed-width (nE, 4) array with
    no Python face loop."""
    if point_type not in ("centroid", "nearpoint"):
        raise ValueError(f"unknown point_type {point_type!r}")
    Fo = _oriented_faces(V, F)
    (E, faces_flat, faces_next, starts, counts, VF, VFig,
     n_vfaces) = _ambo_structure(Fo)
    B = np.array(V, dtype=np.float64)
    B -= B.mean(axis=0)
    A0, B0 = B[E[:, 0]], B[E[:, 1]]
    if point_type == "nearpoint":
        d = B0 - A0
        t = -np.einsum('ij,ij->i', A0, d) / np.maximum(
            np.einsum('ij,ij->i', d, d), 1e-30)
        P = A0 + np.clip(t, 0.0, 1.0)[:, None] * d
    else:
        P = 0.5 * (A0 + B0)
    P = P - P.mean(axis=0)
    scale = np.mean(np.linalg.norm(P, axis=1))
    if scale > 0:
        P /= scale
    roll = np.array([1, 2, 3, 0])
    last_max_diff2 = 0.0
    for it in range(iters):
        cent, nrm = _ambo_face_planes(P, faces_flat, faces_next,
                                      starts, counts)
        N4 = nrm[VF]                       # (nE, 4, 3)
        C4 = cent[VF]
        # coplanarity: mean of the projections onto the 4 face planes
        tproj = np.einsum('vkd,vkd->vk', C4 - P[:, None, :], N4)
        off = factor * np.einsum('vk,vkd->vd', tproj, N4) / 4.0
        # tangency-point centroid: exact recentring
        off -= P.mean(axis=0)
        # orthogonality: each pair of opposing circles must be tangent
        # at the vertex -- project onto the planes through the origin
        # normal to cross(n_opposite, n)
        for i in (0, 1):
            u = np.cross(N4[:, i + 2], N4[:, i])
            u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True),
                            1e-30)
            off -= (factor * 0.5) \
                * np.einsum('vd,vd->v', P, u)[:, None] * u
        # unscrambling: a vertex outside the cycle of its 4 neighbours
        # (any positive triple product) is pulled to their centroid
        W = P[VFig]                        # (nE, 4, 3)
        trips = np.einsum('vkd,vd->vk',
                          np.cross(W, W[:, roll, :]), P)
        bad = (trips > 0.0).any(axis=1)
        if bad.any():
            off[bad] += 0.5 * (W[bad].mean(axis=1) - P[bad])
        max_diff2 = float(np.max(np.einsum('vd,vd->v', off, off)))
        # apply, then project exactly onto the unit sphere
        P = P + off
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-30)
        if max_diff2 < last_max_diff2:
            factor = min(factor * 1.01, factor_max)
        else:
            factor *= 0.995
        last_max_diff2 = max_diff2
        if trace is not None:
            trace(it, P)
        width = float(np.max(P.max(axis=0) - P.min(axis=0)))
        if math.sqrt(max_diff2) < tol * max(width, 1e-30):
            break
    # recover the base by polar reciprocation of the vertex-figure
    # face planes in the unit sphere: v = n / (n . c) (sign-free)
    cent, nrm = _ambo_face_planes(P, faces_flat, faces_next,
                                  starts, counts)
    nv, cv = nrm[:n_vfaces], cent[:n_vfaces]
    dv = np.einsum('vd,vd->v', nv, cv)
    dv = np.where(np.abs(dv) < 1e-30, 1e-30, dv)
    Vout = nv / dv[:, None]
    return [list(map(float, p)) for p in Vout]


def _tangent_spread_of(V, F):
    """Relative spread of the edge-tangency distances (the module's own
    convergence gauge, shared by the self-test and the bench)."""
    P = np.asarray(V, dtype=np.float64)
    d = []
    for f in F:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            if a < b:
                A, B = P[a], P[b]
                AB = B - A
                t = -float(np.dot(A, AB)) / max(float(np.dot(AB, AB)),
                                                1e-30)
                t = min(1.0, max(0.0, t))
                d.append(float(np.linalg.norm(A + t * AB)))
    d = np.asarray(d)
    return float((d.max() - d.min()) / max(d.mean(), 1e-12))


def canonicalize_best(V, F, hart_iters=200):
    """Best-available canonicalization for generator call sites: the
    circle-packing ambo method (canonicalize_ambo) run to its own
    convergence -- measured in tests/bench to be several times FASTER
    than even 400 Hart iterations while reaching machine-precision
    tangency (see research/plans/antiprism-canonicalization-plan.md) --
    with the Hart relaxation (`hart_iters` iterations) as the fallback
    when the input is not a closed orientable polyhedron (the ambo
    change of variables needs one) or the ambo iteration fails to
    actually converge (tangency spread checked, not assumed)."""
    try:
        Va = canonicalize_ambo(V, F)
        if _tangent_spread_of(Va, F) < 1e-9:
            return Va
    except ValueError:
        pass
    return canonicalize(V, F, iters=hart_iters)


def _face_planes(P, Fi):
    """Outward unit Newell normals, origin distances and centroids of
    every face of P."""
    ns = np.zeros((len(Fi), 3))
    ds = np.zeros(len(Fi))
    cs = np.zeros((len(Fi), 3))
    for fi, f in enumerate(Fi):
        Q = P[f]
        c = Q.mean(axis=0)
        n = np.zeros(3)
        for i in range(len(f)):
            n += np.cross(Q[i] - c, Q[(i + 1) % len(f)] - c)
        ln = np.linalg.norm(n) or 1.0
        n /= ln
        if n @ c < 0:
            n = -n
        ns[fi], ds[fi], cs[fi] = n, n @ c, c
    return ns, ds, cs


def _biscribe_energy(P, Fi):
    """The biscribed-form residual energy of research/biscribed-solver-
    research.md: circumsphere spread + insphere spread + planarity."""
    R = np.linalg.norm(P, axis=1)
    E = float(np.sum((R - R.mean()) ** 2))
    ns, ds, cs = _face_planes(P, Fi)
    E += float(np.sum((ds - ds.mean()) ** 2))
    for fi, f in enumerate(Fi):
        off = (P[f] - cs[fi]) @ ns[fi]
        E += float(np.sum(off * off))
    return E


def biscribe(V, F, iters=2500, step=0.1, trace=None, line_search=True,
             init="ambo"):
    """Biscribed form: all vertices on a circumsphere AND all faces tangent
    to a concentric insphere.  Starts from the canonical (edge-tangent)
    form, then drives the vertex radii and the face-plane distances to
    common values with a damped summed-force step (circumsphere +
    insphere + planarity, recentred each step) -- the summed force is what
    keeps it stable where a sequential projection diverges.  Not every
    solid HAS a biscribed form (rectified solids and several truncations do
    not); returns (verts, converged) so the caller can report failure.

    line_search=True (default; measured in tests/bench) minimises the
    residual energy along the summed-force direction with an Evolver-
    style optimizing scale each step: ~7x fewer iterations to residuals
    ~1000x tighter, and non-existent forms are reported in a handful of
    iterations (energy-stall window) instead of burning the whole
    budget.  Convergence additionally requires convexity -- the sphere
    conditions alone are also satisfied by self-intersecting
    pseudo-solutions (the truncated cube finds one), which are not
    biscribed forms.  line_search=False keeps the historical fixed-step
    loop.

    init selects the canonical-form initialisation: "ambo" (default;
    the circle-packing method with Hart fallback -- measured in
    tests/bench: existence classification 8/8 unchanged, residual
    spreads tied at ~1e-14, total catalog wall time 1.6x lower because
    the init dominates it) or "hart" (the historical init, kept
    A/B-able and pinning the pre-branch behaviour bitwise)."""
    if np is None:
        return V, False
    if init == "ambo":
        P0 = canonicalize_best(V, F)
    elif init == "hart":
        P0 = canonicalize(V, F)
    else:
        raise ValueError(f"unknown biscribe init {init!r}")
    P = np.array(P0, dtype=np.float64)
    P -= P.mean(axis=0)
    P /= np.mean(np.linalg.norm(P, axis=1)) or 1.0
    Fi = [np.array(f) for f in F]
    use_ls = bool(line_search) and _descent is not None
    gs = 1.0 if use_ls else step         # unit-gain direction under LS
    s_prev = step
    E_hist = []
    for it in range(iters):
        R = np.linalg.norm(P, axis=1)
        Rb = R.mean()
        dX = (gs * ((Rb - R) / np.maximum(R, 1e-9))[:, None]) * P
        ns, ds, cs = _face_planes(P, Fi)
        rb = ds.mean()
        for fi, f in enumerate(Fi):
            n = ns[fi]
            push = gs * (rb - ds[fi])
            for v in f:
                dX[v] += push * n + gs * (n @ (cs[fi] - P[v])) * n
        dX -= dX.mean(axis=0)
        if use_ls:
            # Evolver-style optimizing scale on the summed-force
            # direction (the monotone-decrease fix the research note
            # asked for), with an energy-stall window that reports
            # non-existence early instead of burning the whole budget
            P, s_used, E_now, _ = _descent.parabola_line_search(
                lambda Q: _biscribe_energy(Q, Fi), P, dX,
                s0=max(s_prev, 1e-6), s_max=1.0)
            if trace is not None:
                trace(it, P)
            E_hist.append(E_now)
            if s_used == 0.0:            # no downhill scale left
                break
            s_prev = s_used
            if s_used * float(np.max(np.abs(dX))) < 1e-13:
                break
            if len(E_hist) >= 60 and E_hist[-1] > E_hist[-60] * (1 - 1e-9):
                break                    # stalled: converged or no form
            continue
        P += dX
        if trace is not None:
            trace(it, P)
        if R.max() / max(R.min(), 1e-9) > 50:
            return [list(map(float, p)) for p in P], False
        if np.max(np.abs(dX)) < 1e-11:
            break
    R = np.linalg.norm(P, axis=1)
    ns, ds, cs = _face_planes(P, Fi)
    dd = np.abs(ds)
    # Convexity gate: equal radii and equal face distances can also be
    # satisfied by a self-intersecting (non-convex) realization -- the
    # monotone line search actually finds one for the truncated cube,
    # whose genuine biscribed form does not exist.  Such a shape is not
    # the biscribed form of the solid, so every vertex must lie on or
    # inside every face plane before convergence is claimed.
    convex_viol = 0.0
    for fi in range(len(Fi)):
        convex_viol = max(convex_viol,
                          float(np.max(P @ ns[fi] - ds[fi])))
    converged = ((R.max() - R.min() < 1e-5)
                 and (dd.max() - dd.min() < 1e-5)
                 and convex_viol < 1e-6 * float(R.mean()))
    return [list(map(float, p)) for p in P], converged


def _selftest():
    """What each canonical form actually claims, checked as a geometric
    property rather than as a fixed vertex list."""
    try:
        from .conway import apply_conway
        from .seeds import seed_poly
    except ImportError:
        from conway import apply_conway
        from seeds import seed_poly
    ok = True

    def edge_list(F):
        return sorted({tuple(sorted((f[i], f[(i + 1) % len(f)])))
                       for f in F for i in range(len(f))})

    def tangent_spread(V, F):
        """How far from a COMMON tangent sphere the edges are: the
        spread of the distances from the origin to each edge's nearest
        point, relative to their mean.  Zero is canonical."""
        P = np.asarray(V, float)
        d = []
        for a, b in edge_list(F):
            A, B = P[a], P[b]
            AB = B - A
            t = -np.dot(A, AB) / max(np.dot(AB, AB), 1e-30)
            t = min(1.0, max(0.0, t))
            d.append(np.linalg.norm(A + t * AB))
        d = np.asarray(d)
        return float((d.max() - d.min()) / max(d.mean(), 1e-12))

    # A Platonic solid is ALREADY canonical, so canonicalize must leave
    # its edge tangency perfect -- a fixed point, not just an improvement.
    bad = []
    for kind in ('TETRA', 'CUBE', 'OCTA', 'DODECA', 'ICOSA'):
        V, F = seed_poly(kind)
        s = tangent_spread(*(canonicalize(V, F, iters=200), F))[0] \
            if False else tangent_spread(canonicalize(V, F, iters=200), F)
        if s > 1e-6:
            bad.append(f"{kind}:{s:.2e}")
    good = not bad
    ok &= good
    print(f"canonical: the Platonic solids are fixed points "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # On a solid that is NOT canonical, the relaxation must actually
    # reduce the tangency spread by a large factor -- otherwise it is
    # doing nothing and the fixed-point test above would still pass.
    # gC, pC and wC -- NOT kC or aD, which turn out to be edge-tangent
    # already (the kis height happens to land there, and ambo(D) is the
    # icosidodecahedron, which is canonical).  Testing an improvement on
    # a solid that starts perfect measures nothing.
    bad = []
    for text in ('gC', 'pC', 'wC'):
        V, F = apply_conway(text)
        before = tangent_spread(V, F)
        after = tangent_spread(canonicalize(V, F, iters=400), F)
        if not (after < before / 10.0):
            bad.append(f"{text}:{before:.3f}->{after:.3f}")
    good = not bad
    ok &= good
    print(f"canonical: relaxation cuts the tangency spread tenfold on "
          f"non-canonical solids {'OK' if good else 'FAIL ' + ','.join(bad)}")

    # The ambo (circle-packing) canonicalization must reach the SAME
    # fixed points -- and, unlike the Hart relaxation, it must reach
    # machine-precision tangency on the non-canonical solids too: the
    # tangency and centroid conditions are exact projections in the
    # ambo variables, so anything worse than ~1e-9 means the change of
    # variables or the reciprocation is wrong, not merely slow.
    def centroid_off(V, F):
        P = np.asarray(V, float)
        pts = []
        for a, b in edge_list(F):
            A, B = P[a], P[b]
            AB = B - A
            t = -np.dot(A, AB) / max(np.dot(AB, AB), 1e-30)
            pts.append(A + min(1.0, max(0.0, t)) * AB)
        return float(np.linalg.norm(np.mean(pts, axis=0)))

    def planarity_of(V, F):
        P = np.asarray(V, float)
        worst = 0.0
        for f in F:
            Q = P[list(f)]
            c = Q.mean(axis=0)
            Qc = Q - c
            _w, vv = np.linalg.eigh(Qc.T @ Qc)
            worst = max(worst, float(np.abs(Qc @ vv[:, 0]).max()))
        return worst

    bad = []
    for kind in ('TETRA', 'CUBE', 'OCTA', 'DODECA', 'ICOSA'):
        V, F = seed_poly(kind)
        s = tangent_spread(canonicalize_ambo(V, F), F)
        if s > 1e-9:
            bad.append(f"{kind}:{s:.2e}")
    good = not bad
    ok &= good
    print(f"canonical: ambo keeps the Platonic fixed points "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    bad = []
    for text in ('gC', 'pC', 'wC'):
        V, F = apply_conway(text)
        Va = canonicalize_ambo(V, F)
        s = tangent_spread(Va, F)
        c = centroid_off(Va, F)
        p = planarity_of(Va, F)
        if s > 1e-9 or c > 1e-9 or p > 1e-9:
            bad.append(f"{text}:s={s:.1e},c={c:.1e},p={p:.1e}")
    good = not bad
    ok &= good
    print(f"canonical: ambo reaches machine-precision tangency, "
          f"centred tangency points and planar faces "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # the nearpoint initialisation is for near-canonical input; it must
    # land on the same form
    V, F = apply_conway('tI')
    s = tangent_spread(canonicalize_ambo(V, F, point_type="nearpoint"), F)
    good = s < 1e-9
    ok &= good
    print(f"canonical: ambo nearpoint init converges (s={s:.1e}) "
          f"{'OK' if good else 'FAIL'}")

    # non-manifold input must RAISE, not silently return nonsense --
    # a single open square has every edge on one face only
    try:
        canonicalize_ambo([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                          [[0, 1, 2, 3]])
        good = False
    except ValueError:
        good = True
    ok &= good
    print(f"canonical: ambo rejects non-closed input "
          f"{'OK' if good else 'FAIL'}")

    # spherize puts every vertex on one sphere, by definition.
    V, F = apply_conway('kC')
    P = np.asarray(spherize(V, F, iters=1), float)
    r = np.linalg.norm(P, axis=1)
    good = float(r.std() / r.mean()) < 1e-9
    ok &= good
    print(f"canonical: spherize puts every vertex on one sphere "
          f"{'OK' if good else 'FAIL'}")

    # biscribe: vertices on a common sphere AND faces tangent to a
    # concentric one -- BOTH, which is what separates it from spherize.
    # It returns (verts, converged), and the flag is not decoration: not
    # every solid has a biscribed form.  Rectified solids do not, so the
    # test asserts BOTH directions -- convergence where the form exists,
    # and honest failure where it does not.  Asserting tangency on a
    # rectified solid would be asserting something false.
    def spheres(V, F):
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
                float(fd.std() / fd.mean()))

    bad = []
    for text in ('C', 'D', 'kC', 'kD'):          # these HAVE a biscribed form
        V, F = apply_conway(text)
        Vb, conv = biscribe(V, F, iters=800)
        rs, fs = spheres(Vb, F)
        if not conv or rs > 1e-9 or fs > 1e-9:
            bad.append(f"{text}:conv={conv},r={rs:.1e},f={fs:.1e}")
    good = not bad
    ok &= good
    print(f"canonical: biscribe reaches both spheres where the form exists "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # The rectified solids: no biscribed form, and it must SAY so rather
    # than return a plausible-looking near-miss as success.
    bad = []
    for text in ('aC', 'aD'):
        V, F = apply_conway(text)
        Vb, conv = biscribe(V, F, iters=800)
        rs, fs = spheres(Vb, F)
        if conv or fs < 1e-3:
            bad.append(f"{text}:conv={conv},f={fs:.1e}")
    good = not bad
    ok &= good
    print(f"canonical: rectified solids report no biscribed form "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("canonical self-test failed")
