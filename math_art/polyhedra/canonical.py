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


def biscribe(V, F, iters=2500, step=0.1, trace=None, line_search=True):
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
    loop."""
    if np is None:
        return V, False
    P = np.array(canonicalize(V, F), dtype=np.float64)
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
