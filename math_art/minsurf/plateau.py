# Plateau solver: discrete area minimization.
#
# Part of the Math Art minimal-surface engine (`math_art/minsurf/`), split
# out of the former single-file `minimal_surface_toolkit.py`.  Numpy only --
# no `bpy` -- so the whole engine imports and self-tests headlessly; the
# registered Blender operators stay in the flat `minimal_surface_toolkit.py`
# front-end.
#
# A Plateau-problem solver: pin one or two boundary curves and minimize
# area by cotangent-Laplacian iteration solved with conjugate gradients --
# a lightweight take on what Brakke's Surface Evolver does.  Includes the
# "minimal surface spanning a circle and a torus knot" construction.
#
# References:
#   J. Plateau, "Statique experimentale et theorique des liquides" (1873).
#   U. Pinkall and K. Polthier, "Computing discrete minimal surfaces and
#       their conjugates", Experimental Mathematics 2(1) (1993).
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992).
#   H. Seifert, "Ueber das Geschlecht von Knoten", Math. Annalen 110
#       (1934); visualization after J. J. van Wijk and A. M. Cohen, IEEE
#       TVCG 12(4) (2006).

import math
from collections import defaultdict

import numpy as np

try:
    from ..solver import cotan as _sc
    from ..solver import descent as _sd
    from ..solver import collide as _scol
except ImportError:                      # flat (path-based) headless import
    from solver import cotan as _sc
    from solver import descent as _sd
    from solver import collide as _scol

TAU = 2.0 * math.pi


# ==========================================================================
# 3. Plateau solver (area minimization with pinned boundaries)
# ==========================================================================

def resample_loop(pts, m):
    """Uniform-arclength resample of a closed polyline (k,3) -> (m,3)."""
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    closed = np.vstack([pts, pts[:1]])
    t = np.linspace(0.0, total, m, endpoint=False)
    out = np.empty((m, 3))
    j = 0
    for i, ti in enumerate(t):
        while s[j + 1] < ti:
            j += 1
        f = (ti - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out[i] = closed[j] * (1 - f) + closed[j + 1] * f
    return out


def align_loops(A, B):
    """Cyclic shift + optional reversal of B minimizing sum |A_i - B_i|^2."""
    m = len(A)
    best = (None, 1e30)
    for Bc in (B, B[::-1]):
        # brute-force cyclic shifts (m <= a few hundred)
        for sft in range(m):
            d = np.sum((A - np.roll(Bc, -sft, axis=0)) ** 2)
            if d < best[1]:
                best = (np.roll(Bc, -sft, axis=0), d)
    return best[0]


def _cotan_weights(V, T, mode="clamp"):
    """Per-edge cotangent weights. Returns (edges (e,2), w (e,)).

    mode="clamp" (default) clips the cotangents to [0.01, 20] -- the
    historical behaviour: positive weights keep the maximum principle,
    so the solve cannot fold a degenerate fan, at the cost of biasing
    every obtuse corner toward the uniform-weight (Tutte) solution.
    mode="mollify" uses the true cotangents built from intrinsically
    mollified edge lengths (Sharp-Crane; see solver/cotan.py): exact
    Pinkall-Polthier away from degenerate triangles, finite on them."""
    return _sc.edge_cotan_weights(V, T, mode=mode)


def minimize_area(V, T, fixed, outer_iters=30, cg_tol=1e-8, cg_iters=400,
                  uniform=False, cotan_mode="mollify", groom_every=0,
                  groom_smooth=0.25):
    """Pinkall-Polthier: repeatedly solve the cotan-Laplace equation for
    the interior vertices (boundary pinned). V modified in place.
    With uniform=True, unit weights are used instead of cotangents --
    a Tutte-style fairing solve that untangles folded regions (at the
    cost of exact minimality; follow with a cotan pass).

    cotan_mode: "mollify" (default; true cotangents from intrinsically
    mollified lengths, see solver/cotan) or "clamp" (the historical
    [0.01, 20] clip, kept for comparison).  The default changed after
    tests/bench measured mollify strictly more minimal (catenoid waist
    error 10x smaller, mean-curvature residual 6-10x smaller on the
    Seifert spans), faster (fewer CG iterations), and still embedded
    across the whole Seifert q x samples sweep.
    groom_every=k > 0 runs a mesh-grooming cycle (Delaunay flips +
    tangential vertex averaging, solver/groom) every k outer iterations;
    T is then rewritten in place (counts unchanged), so callers keeping
    a separate quad list for display are unaffected."""
    n = len(V)
    free = ~fixed
    nfree = int(np.sum(free))
    if nfree == 0:
        return V
    if groom_every:
        try:
            from ..solver import groom as _sg
        except ImportError:
            from solver import groom as _sg
    for _it in range(outer_iters):
        if groom_every and _it and _it % groom_every == 0:
            _sg.groom(V, T, fixed=fixed, smooth_lam=groom_smooth)
        E, W = _cotan_weights(V, T, mode=cotan_mode)
        if uniform:
            W = np.ones_like(W)
        deg = np.zeros(n)
        np.add.at(deg, E[:, 0], W)
        np.add.at(deg, E[:, 1], W)

        def matvec_full(Xf):
            y = deg[:, None] * Xf
            np.add.at(y, E[:, 0], -W[:, None] * Xf[E[:, 1]])
            np.add.at(y, E[:, 1], -W[:, None] * Xf[E[:, 0]])
            return y

        Xb = np.where(fixed[:, None], V, 0.0)
        b = -matvec_full(Xb)[free]

        def matvec(xf):
            full = np.zeros((n, 3))
            full[free] = xf
            return matvec_full(full)[free]

        x = V[free].copy()
        r = b - matvec(x)
        p = r.copy()
        rs = np.sum(r * r, axis=0)
        b_norm = max(np.max(np.sum(b * b, axis=0)), 1e-30)
        for _cg in range(cg_iters):
            Ap = matvec(p)
            pAp = np.sum(p * Ap, axis=0)
            alpha = rs / np.where(np.abs(pAp) > 1e-30, pAp, 1e-30)
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.sum(r * r, axis=0)
            if np.max(rs_new) < cg_tol * b_norm:
                break
            p = r + (rs_new / np.maximum(rs, 1e-30)) * p
            rs = rs_new
        move = np.max(np.linalg.norm(x - V[free], axis=1))
        V[free] = x
        if move < 1e-6 * max(1.0, np.max(np.abs(V))):
            break
    return V


def minimize_area_lbfgs(V, T, fixed, outer_iters=200, m=8, step_cap=4.0,
                        cotan_mode="mollify", guard=None, tol=1e-10,
                        h0_eps=1e-3, h0_tol=1e-2, h0_iters=100):
    """Direct L-BFGS area descent over the free vertices (Laplacian H0
    seed, Armijo backtracking), the measured upgrade path over the
    Pinkall-Polthier iteration above: on the catenoid it reaches the
    same area ~2.7x faster in wall clock with a mean-curvature residual
    200-380x smaller (see research/plans/lbfgs-descent-core-plan.md).
    V is modified in place and returned.

    guard: optional collision guard (solver/collide) -- True, an options
    dict, or a MeshGuard.  The simplified-IPC barrier joins the energy
    and gradient and its conservative cap bounds every trial step, so
    the sheet cannot pass through itself no matter how long the flow
    runs.  That matters here because the least-area surface in a class
    like the Seifert spans is NOT embedded: the unguarded flow must be
    stopped by an iteration cap (_SEIFERT_MAX_ITERS) before it pinches
    through, while the guarded flow settles against its own contact
    barrier instead."""
    fixed = np.asarray(fixed, bool)
    free = ~fixed
    if not np.any(free):
        return V
    guard = _scol.make_guard(guard)
    lb = _sd.LBFGS(m=m)
    h0 = _sd.LaplacianH0(eps=h0_eps, tol=h0_tol, max_iters=h0_iters)

    def _garea(Vv):
        E, W = _cotan_weights(Vv, T, mode=cotan_mode)
        g = np.zeros_like(Vv)
        contrib = W[:, None] * (Vv[E[:, 0]] - Vv[E[:, 1]])
        np.add.at(g, E[:, 0], contrib)
        np.add.at(g, E[:, 1], -contrib)
        return g

    def _E(Vv):
        Eb = 0.0 if guard is None else guard.energy(Vv)
        return mesh_area(Vv, T) + Eb

    x_prev = None
    g_prev = None
    if guard is not None:
        guard.ensure(V, T)
    E_prev = _E(V)
    flat = 0
    for _it in range(1, outer_iters + 1):
        if guard is not None:
            # A rebuild does not change the energy VALUE (the gate is
            # exact), so E_prev from the last acceptance stays valid.
            guard.ensure(V, T)
        g = _garea(V)
        if guard is not None:
            g = g + guard.gradient(V)
        g[fixed] = 0.0
        if x_prev is not None:
            lb.push((V - x_prev).ravel(), (g - g_prev).ravel())
        h0.update(V, T, free=free)
        d = -lb.direction(g.ravel(), h0=h0).reshape(-1, 3)
        d[fixed] = 0.0
        slope = float(np.einsum('nj,nj->', g, d))
        if not (slope < 0.0):
            lb.reset()
            d = -h0(g.ravel()).reshape(-1, 3)
            d[fixed] = 0.0
            slope = float(np.einsum('nj,nj->', g, d))
            if not (slope < 0.0):
                break
        Lmean = float(np.mean(np.linalg.norm(
            V[T[:, 1]] - V[T[:, 0]], axis=1)))
        dmax = float(np.max(np.linalg.norm(d, axis=1)))
        s_max = step_cap * Lmean / max(dmax, 1e-300)
        if guard is not None:
            s_max = min(s_max, float(guard.max_step(V, d)))
        x1, s, E1, _ne = _sd.armijo_backtrack(
            _E, V, d, slope, s0=min(1.0, s_max), E0=E_prev, s_max=s_max)
        if s == 0.0:
            break
        x_prev = V.copy()
        g_prev = g
        V[:] = x1
        if abs(E1 - E_prev) < tol * max(E_prev, 1e-300):
            flat += 1
            if flat >= 3:
                break
        else:
            flat = 0
        E_prev = E1
    return V


def relax_normal_flow(V, T, fixed, iters=60, lam=0.4,
                      cotan_mode="mollify"):
    """Mean-curvature flow restricted to the surface normal: pulls a
    (slightly perturbed) net back toward the minimal surface without
    tangential sliding, so a fair control net stays fair. Only suitable
    as a polish -- it cannot perform global reorganization."""
    free = ~fixed
    n = len(V)
    for _ in range(iters):
        E, W = _cotan_weights(V, T, mode=cotan_mode)
        lap = np.zeros((n, 3))
        d = V[E[:, 1]] - V[E[:, 0]]
        np.add.at(lap, E[:, 0], W[:, None] * d)
        np.add.at(lap, E[:, 1], -W[:, None] * d)
        wsum = np.zeros(n)
        np.add.at(wsum, E[:, 0], W)
        np.add.at(wsum, E[:, 1], W)
        umb = lap / np.maximum(wsum, 1e-12)[:, None]
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        vn = np.zeros((n, 3))
        np.add.at(vn, T[:, 0], fn)
        np.add.at(vn, T[:, 1], fn)
        np.add.at(vn, T[:, 2], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        move = np.sum(umb * vn, axis=1, keepdims=True) * vn
        V[free] += lam * move[free]
    return V


def fair_grid_2d(G, iters=8, step=0.5):
    """Light Laplacian fairing of an (rows, m, 3) net, cyclic in m,
    end rows pinned. Restores row/column coherence after per-column
    resampling; follow with relax_normal_flow to restore minimality."""
    G = G.copy()
    for _ in range(iters):
        up = np.roll(G, 1, axis=0)
        dn = np.roll(G, -1, axis=0)
        up[0] = G[0]
        dn[-1] = G[-1]
        lt = np.roll(G, 1, axis=1)
        rt = np.roll(G, -1, axis=1)
        avg = (up + dn + lt + rt) / 4.0
        G[1:-1] += step * (avg[1:-1] - G[1:-1])
    return G


def fair_grid_columns(G):
    """Re-sample every column (axis 0) of an (rows, m, 3) grid uniformly
    by arc length. Points stay on their column polylines -- i.e. on the
    solved surface -- but the severe bunching/shear the area solver
    introduces (which makes a NURBS control net ring) is equalized."""
    rows, m, _ = G.shape
    out = np.empty_like(G)
    for i in range(m):
        P = G[:, i, :]
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total < 1e-12:
            out[:, i, :] = P
            continue
        t = np.linspace(0.0, total, rows)
        for a in range(3):
            out[:, i, a] = np.interp(t, s, P[:, a])
    return out


def mesh_area(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return 0.5 * float(np.sum(np.linalg.norm(n, axis=1)))


def _quads_to_tris(quads):
    T = []
    for q in quads:
        if len(q) == 4:
            T.append((q[0], q[1], q[2]))
            T.append((q[0], q[2], q[3]))
        else:
            T.append(tuple(q))
    return np.array(T, dtype=np.int64)


def build_disk_grid(boundary, rings):
    """Initial disk spanning one loop: concentric rings toward centroid."""
    m = len(boundary)
    cen = boundary.mean(axis=0)
    verts = [boundary]
    for r in range(1, rings):
        f = r / rings
        verts.append(boundary * (1 - f) + cen * f)
    V = np.concatenate(verts + [cen[None, :]], axis=0)
    quads = []
    for r in range(rings - 1):
        for i in range(m):
            i2 = (i + 1) % m
            quads.append((r * m + i, r * m + i2,
                          (r + 1) * m + i2, (r + 1) * m + i))
    last = (rings - 1) * m
    cidx = rings * m
    for i in range(m):
        quads.append((last + i, last + (i + 1) % m, cidx))
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:m] = True
    return V, quads, fixed


def build_annulus_grid(loopA, loopB, rows):
    """Initial ruled surface between two aligned loops of equal length."""
    m = len(loopA)
    verts = []
    for r in range(rows + 1):
        f = r / rows
        verts.append(loopA * (1 - f) + loopB * f)
    V = np.concatenate(verts, axis=0)
    quads = []
    for r in range(rows):
        for i in range(m):
            i2 = (i + 1) % m
            quads.append((r * m + i, r * m + i2,
                          (r + 1) * m + i2, (r + 1) * m + i))
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:m] = True
    fixed[rows * m:] = True
    return V, quads, fixed


def torus_knot(p, q, m, scale=1.0, tube=1.0):
    t = np.linspace(0, TAU, m, endpoint=False)
    r = np.cos(q * t) * tube + 2.0
    return np.stack([r * np.cos(p * t), r * np.sin(p * t),
                     -np.sin(q * t) * tube], axis=1) * scale


# The Seifert span is embedded as built, but the area flow that relaxes
# it afterwards is what can undo that: the cotangent-Laplacian solve is
# free to drag the membrane's rim collar (where the inner Seifert
# circle's ~q/2 waves of wobble die away) through the membrane behind
# it, and to pinch the handles. Three limits keep the flow inside the
# embedded regime: a floor on the azimuth samples per half period, so
# the collar is never one quad thick; a ceiling on the radial rows,
# since rows much finer than the azimuth spacing make slivers that the
# cotangent weights fold (the ceiling tightens with q, whose waves
# crowd the collar); and a cap on the relaxation passes, because the
# least-area surface in this class is not the embedded one -- run the
# flow long enough and it pinches through itself whatever the grid.
# All three were fixed by sweeping q x samples x rings x iterations and
# counting genuine edge-through-triangle crossings; a normal-flip
# detector was tried first and rejected (it both misses folds that
# creep in without a flip and fires on the harmless slivers in the
# q = 1 band).
_SEIFERT_MIN_G = 12         # azimuth samples per half period, minimum
_SEIFERT_MAX_ROWS = 48      # radial row budget, divided down as q grows
_SEIFERT_MAX_ITERS = 8      # relaxation passes the span stays embedded for


def _seifert_rows(q, rings, mA):
    """Radial rows the relaxation stays embedded for: the ceiling falls
    with q (whose waves crowd the collar) and never lets a row be finer
    than a third of the azimuth spacing."""
    return max(4, min(int(rings),
                      _SEIFERT_MAX_ROWS // ((int(q) + 3) // 2),
                      mA // 3))


def build_seifert_span_grid(q, m, rings, knot_scale=1.0, circle_radius=4.5,
                            inner_height=1.0, inner_lift=0.0, window=0.35):
    """Embedded single-sheet spanning surface between the (2, q) torus
    knot (q odd) and the round circle of radius `circle_radius`.

    An embedded *annulus* between a knotted and a round boundary cannot
    exist (its two boundary curves would be isotopic knots), so this
    builds the correct next-simplest surface: Seifert's algorithm
    (H. Seifert, 1934) applied to the round (2, q) diagram, with the
    outer Seifert disk opened out to the circle. Three families of
    structured patches share their seam vertices:

      funnel   -- annulus from the circle in to the outer smoothed
                  Seifert circle (knot arcs + saddle diagonals);
      membrane -- disk spanning the inner smoothed Seifert circle
                  across the middle;
      q bands  -- one half-twist Coons patch per diagram crossing,
                  edged by the two knot strands and the two diagonals.

    Every patch preserves azimuth and they occupy disjoint radial
    ranges (funnel outside `r_out`, membrane inside `r_in`, bands in
    between, in disjoint azimuth windows), so the result is embedded
    by construction. Genus (q - 1) / 2, two boundary loops: the exact
    torus knot (2 m-ish samples) and the exact circle. Returns
    (V, quads, fixed) with `fixed` True exactly on the two boundary
    loops, ready for minimize_area -- for at most _SEIFERT_MAX_ITERS
    passes, beyond which the area flow pulls the sheet back through
    itself (see the note on that constant).
    """
    q = int(q)
    if q < 1 or q % 2 == 0:
        raise ValueError("build_seifert_span_grid needs odd q >= 1")
    k = float(knot_scale)
    h = float(inner_height)
    R = float(circle_radius)
    # `m` is a request, not a promise: the azimuth resolution is floored
    # at _SEIFERT_MIN_G samples per half period so the relaxation cannot
    # fold the membrane collar (a coarse grid leaves it one quad thick)
    g = max(_SEIFERT_MIN_G, int(round(m / (2.0 * q))))
    mA = 2 * q * g                      # azimuth samples, once around
    # ... and `rings` likewise: rows finer than a third of the azimuth
    # spacing, or too many of them for this q, slice the sheet into
    # slivers that the flow folds
    rings = _seifert_rows(q, rings, mA)
    wn = max(2, min(g - 2, int(round(window * g))))
    ns = 2 * wn + 1                     # samples across a crossing window
    th = TAU * np.arange(mA) / mA

    def strand(which, t):
        """Knot strand 0/1 at azimuth t (i.e. tau = t/2 [+ pi])."""
        ph = q * t / 2.0 + (q * math.pi if which else 0.0)
        rho = 2.0 * k + k * math.cos(ph)
        return np.array([rho * math.cos(t), rho * math.sin(t),
                         -k * h * math.sin(ph) + inner_lift])

    centers = [g * (2 * n + 1) for n in range(q)]   # crossing indices
    in_open = np.zeros(mA, dtype=bool)  # strictly inside a window
    for c in centers:
        for d in range(1 - wn, wn):
            in_open[(c + d) % mA] = True

    cw = math.sin(q * (wn * TAU / mA) / 2.0)   # |cos(q u/2)| at edge
    r_out = 2.0 * k + k * cw            # diagonal rail radii
    r_in = 2.0 * k - k * cw

    # smoothed Seifert circles: knot arcs outside the crossing windows,
    # constant-radius diagonals (the saddle rails) across them
    Cout = np.empty((mA, 3))
    Cin = np.empty((mA, 3))
    for i in range(mA):
        A, B = strand(0, th[i]), strand(1, th[i])
        a_out = math.cos(q * th[i] / 2.0) >= 0.0
        Cout[i], Cin[i] = (A, B) if a_out else (B, A)
    for c in centers:
        s1 = 0 if math.cos(q * th[(c - wn) % mA] / 2.0) >= 0.0 else 1
        zo0 = Cout[(c - wn) % mA][2]    # strand z at the window edges
        zo1 = strand(1 - s1, (c + wn) * TAU / mA)[2]
        zi0 = Cin[(c - wn) % mA][2]
        zi1 = strand(s1, (c + wn) * TAU / mA)[2]
        for d in range(-wn, wn + 1):
            i = (c + d) % mA
            fb = (d + wn) / (2.0 * wn)
            Cout[i] = (r_out * math.cos(th[i]), r_out * math.sin(th[i]),
                       (1 - fb) * zo0 + fb * zo1)
            Cin[i] = (r_in * math.cos(th[i]), r_in * math.sin(th[i]),
                      (1 - fb) * zi0 + fb * zi1)

    circle = np.stack([R * np.cos(th), R * np.sin(th),
                       np.zeros(mA)], axis=1)
    verts = []
    fixed = []
    quads = []
    for r in range(rings + 1):          # funnel rows: circle -> Cout
        f = r / rings
        verts.append(circle * (1 - f) + Cout * f)
        fixed.append(np.ones(mA, bool) if r == 0
                     else (~in_open if r == rings
                           else np.zeros(mA, bool)))
    for r in range(rings):
        for i in range(mA):
            i2 = (i + 1) % mA
            quads.append((r * mA + i, r * mA + i2,
                          (r + 1) * mA + i2, (r + 1) * mA + i))
    fun_last = rings * mA               # first index of the Cout row
    base = (rings + 1) * mA             # first index of the membrane
    cen = Cin.mean(axis=0)
    for r in range(rings):              # membrane rows: Cin -> center
        f = r / rings
        verts.append(Cin * (1 - f) + cen * f)
        fixed.append(~in_open if r == 0 else np.zeros(mA, bool))
    verts.append(cen[None, :])
    fixed.append(np.zeros(1, bool))
    cidx = base + rings * mA
    for r in range(rings - 1):
        for i in range(mA):
            i2 = (i + 1) % mA
            quads.append((base + r * mA + i, base + r * mA + i2,
                          base + (r + 1) * mA + i2,
                          base + (r + 1) * mA + i))
    lastm = base + (rings - 1) * mA
    for i in range(mA):
        quads.append((lastm + i, lastm + (i + 1) % mA, cidx))

    Vp = np.concatenate(verts, axis=0)
    Vs = [Vp]
    Fs = [np.concatenate(fixed)]
    nxt = len(Vp)
    for c in centers:                   # one saddle band per crossing
        s1 = 0 if math.cos(q * th[(c - wn) % mA] / 2.0) >= 0.0 else 1
        S1 = np.array([strand(s1, (c - wn + x) * TAU / mA)
                       for x in range(ns)])           # rail y = 0
        S2r = np.array([strand(1 - s1, (c + wn - x) * TAU / mA)
                        for x in range(ns)])          # rail y = ns-1
        Out = Vp[[fun_last + (c - wn + y) % mA for y in range(ns)]]
        Inr = Vp[[base + (c + wn - y) % mA for y in range(ns)]]
        idx = np.empty((ns, ns), dtype=np.int64)
        for y in range(ns):
            idx[0, y] = fun_last + (c - wn + y) % mA
            idx[ns - 1, y] = base + (c + wn - y) % mA
        new_v = []
        new_f = []
        for x in range(1, ns - 1):
            u = x / (ns - 1.0)
            for y in (0, ns - 1):       # strand rails: on the knot
                idx[x, y] = nxt
                new_v.append(S1[x] if y == 0 else S2r[x])
                new_f.append(True)
                nxt += 1
            for y in range(1, ns - 1):  # Coons interior of the saddle
                v = y / (ns - 1.0)
                pt = ((1 - v) * S1[x] + v * S2r[x]
                      + (1 - u) * Out[y] + u * Inr[y]
                      - ((1 - u) * (1 - v) * S1[0]
                         + u * (1 - v) * S1[-1]
                         + (1 - u) * v * S2r[0] + u * v * S2r[-1]))
                idx[x, y] = nxt
                new_v.append(pt)
                new_f.append(False)
                nxt += 1
        Vs.append(np.array(new_v))
        Fs.append(np.array(new_f, dtype=bool))
        for x in range(ns - 1):
            for y in range(ns - 1):
                quads.append((idx[x, y], idx[x + 1, y],
                              idx[x + 1, y + 1], idx[x, y + 1]))
    return np.concatenate(Vs, axis=0), quads, np.concatenate(Fs)


def _selfx_crossings(V, T, tol=1e-7):
    """Count genuine self-intersections of a triangle mesh: edges that
    pierce the open interior of a triangle they share no vertex with
    (Moeller-Trumbore, bbox-prefiltered). O(edges x tris) per bbox
    survivor -- intended for selftest-sized meshes."""
    T = np.asarray(T)
    E = set()
    for tri in T:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]),
                     (tri[2], tri[0])):
            E.add((a, b) if a < b else (b, a))
    E = np.array(sorted(E))
    P0, P1 = V[E[:, 0]], V[E[:, 1]]
    elo, ehi = np.minimum(P0, P1), np.maximum(P0, P1)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    tlo = np.minimum(np.minimum(A, B), C)
    thi = np.maximum(np.maximum(A, B), C)
    count = 0
    for i0 in range(0, len(E), 512):
        i1 = min(i0 + 512, len(E))
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
                share |= (E[ei, cc] == T[ti, dd])
        ei, ti = ei[~share], ti[~share]
        if not len(ei):
            continue
        o = V[E[ei, 0]]
        d = V[E[ei, 1]] - o
        e1 = V[T[ti, 1]] - V[T[ti, 0]]
        e2 = V[T[ti, 2]] - V[T[ti, 0]]
        pv = np.cross(d, e2)
        det = np.einsum('ij,ij->i', e1, pv)
        good = np.abs(det) > 1e-14
        inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
        tv = o - V[T[ti, 0]]
        uu = np.einsum('ij,ij->i', tv, pv) * inv
        qv = np.cross(tv, e1)
        vv = np.einsum('ij,ij->i', d, qv) * inv
        tt = np.einsum('ij,ij->i', e2, qv) * inv
        count += int(np.sum(good & (uu > tol) & (vv > tol)
                            & (uu + vv < 1 - tol)
                            & (tt > tol) & (tt < 1 - tol)))
    return count


# ==========================================================================
# ==========================================================================
# The discrete conjugate (adjoint) surface
# ==========================================================================
# The capability that most of Brakke's triply-periodic collection waits
# on, and it turned out to be smaller than it looked.
#
# His `*adj.fe` datafiles were read here for a long time as needing
# constraint planes with free-sliding vertices.  They do not.  Each one
# states a PLAIN PLATEAU PROBLEM -- a closed space polygon with every
# vertex and every edge fixed (GW5adj.fe is six fixed vertices, six
# fixed edges, one hexagonal face) -- which `minimize_area` already
# solves.  The declared constraint planes describe where the CONJUGATE
# surface's boundary lands, not where the original's does.  So the only
# missing piece was the conjugate itself.
#
# Pinkall and Polthier's discrete conjugate, in the form Brakke's
# adjoint.cmd implements: walk the facets, and for a facet with unit
# normal n whose three edge vectors are taken in traversal order,
#
#     p[next] = p[this] - ( cos(b) * e[other]
#                           + sin(b) * (n x e[other]) ) / 2,
#
# with b the Bonnet angle -- b = 90 degrees is the conjugate proper.
# The positions live on EDGES, not vertices, which is the whole trick:
# the conjugate of a conforming triangulation is nonconforming, and
# forcing it onto vertices too early is what makes naive attempts fail.
# Vertices are recovered afterwards by averaging their incident edges.
#
# STATUS -- read before using this on a surface.  The diagnosis below is
# complete; what is missing is a solver, not an understanding.
#
# VERIFIED, and gated in `_selftest`:
#   * the per-facet relation closes exactly -- 5.6e-17 walking one
#     triangle 0 -> 1 -> 2 -> 0, with medial edge lengths |e|/2 at every
#     Bonnet angle, as they must be since the two terms are orthogonal;
#   * at bangle 0 the whole pipeline is the IDENTITY on a real mesh, to
#     0.0e+00 in area -- propagation and vertex recovery together;
#   * on an EXACTLY minimal mesh (a flat grid) the propagation is exact
#     at bangle 90 too: residual 1.0e-16 against an edge scale of 0.25.
#     That is the check that says the algorithm is right.
#
# WHY IT IS NOT YET USED BY ANY SHIPPED ROW.  On a relaxed surface the
# propagation picks up a residual, and the reason is now understood
# rather than suspected.  It has two independent parts:
#
#   1. A REAL PERIOD, on any domain that is not simply connected.  The
#      relation is a discrete 1-form, and integrating it around a
#      non-contractible loop returns the conjugate's period, not zero --
#      the conjugate of a catenoid is a helicoid, which does not close.
#      On an annulus the residual is 42x the edge length and is
#      concentrated exactly where the walk wraps and meets itself; on a
#      simply connected disk it falls to 0.12x.  This is the "period
#      killing" Karcher's conjugate Plateau method exists to do, and it
#      is why Brakke's datafiles carry tunable rhs parameters.
#   2. NON-INTEGRABILITY, because the surface is only approximately
#      minimal.  The closing condition around an interior vertex is
#      exactly the area gradient there -- (1/2) sum over incident facets
#      of n_T x (edge opposite v) -- which `minimize_area` drives down
#      (2.6e-2 -> 1.5e-4 -> 9.7e-6 over 0/10/600 iterations) but never
#      to zero.
#
# So a WALK is the wrong integrator: it commits to whichever facet
# reached an edge first and pushes all the inconsistency into the edges
# reached last.  Measured on a disk, its residual RISES relative to the
# edge length as the mesh refines -- 0.083 at 48x8, 0.116 at 96x16,
# 0.324 at 144x24 -- while the area converges cleanly.
#
# RESUME, and this is a solver task now.  Replace the walk with a
# least-squares integration of the 1-form: minimise
#     sum over facets, over k, |P[e_{k+1}] - P[e_k] - w_{f,k}|^2,
# a graph Laplacian on edge-adjacency, singular only in the global
# translation.  That spreads the inconsistency instead of accumulating
# it.  It was tried here with 600 Jacobi sweeps and abandoned: Jacobi
# converges far too slowly on a Laplacian of this diameter and lost the
# bangle-0 identity entirely (33% area error), which is a solver
# failure, not a formulation one.  Use conjugate gradients, and keep the
# bangle-0 identity as the acceptance test -- it is exact for the walk
# and must stay exact.
#
def _facet_edges(T):
    """Undirected edge index per facet corner, plus the edge list."""
    ids = {}
    per = np.empty((len(T), 3), dtype=np.int64)
    for f, tri in enumerate(T):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            key = (a, b) if a < b else (b, a)
            j = ids.get(key)
            if j is None:
                j = ids[key] = len(ids)
            per[f, k] = j
    return per, ids


def discrete_adjoint(V, T, bangle=90.0):
    """Conjugate of a discrete minimal surface (Pinkall-Polthier).

    Returns new vertex positions.  `bangle` sweeps the associate family:
    0 returns the original, 90 the conjugate.
    """
    V = np.asarray(V, dtype=float)
    T = np.asarray(T, dtype=np.int64)
    per, _ids = _facet_edges(T)
    ne = int(per.max()) + 1
    P = np.zeros((ne, 3))
    known = np.zeros(ne, dtype=bool)

    e = np.empty((len(T), 3, 3))
    for k in range(3):
        e[:, k] = V[T[:, (k + 1) % 3]] - V[T[:, k]]
    n = np.cross(e[:, 0], e[:, 1])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-300)

    bc = math.cos(math.radians(bangle))
    bs = math.sin(math.radians(bangle))

    # facets touching each edge, so the walk is a BFS rather than the
    # repeated full sweep the Evolver script uses
    inc = defaultdict(list)
    for f in range(len(T)):
        for k in range(3):
            inc[int(per[f, k])].append(f)

    start = int(per[0, 0])
    known[start] = True
    frontier = [start]
    while frontier:
        nxt = []
        for ei in frontier:
            for f in inc[ei]:
                k = int(np.where(per[f] == ei)[0][0])
                for step in (1, 2):
                    kn = (k + step) % 3
                    en = int(per[f, kn])
                    if known[en]:
                        continue
                    ko = (k + (3 - step)) % 3
                    eo = e[f, ko] * (1.0 if step == 1 else -1.0)
                    P[en] = P[ei] - 0.5 * (bc * eo
                                           + bs * np.cross(n[f], eo))
                    known[en] = True
                    nxt.append(en)
        frontier = nxt

    # NONCONFORMING -> CONFORMING, by least squares rather than by
    # averaging.
    #
    # `P` holds one position per EDGE, and at bangle 0 those are exactly
    # the original edge MIDPOINTS -- which is the check that the
    # propagation above is right.  A vertex is therefore NOT the mean of
    # its incident edge values: averaging midpoints pulls every vertex
    # toward the centroid of its neighbourhood and shrinks the surface
    # (measured: 6% area loss and 31% edge-length distortion at bangle 0,
    # where the transform is supposed to be the identity).
    #
    # What the edge values actually assert is (x_a + x_b)/2 = P_e for
    # every edge, an overdetermined linear system whose least-squares
    # solution is the conforming surface.  Its normal equations are
    #     deg(v) x_v + sum_{w ~ v} x_w = 2 sum_{e in v} P_e,
    # solved below by Jacobi iteration -- diagonally dominant, so it
    # converges, and the system is singular only in the global
    # translation, which is fixed at the end by recentring.
    ends = np.empty((ne, 2), dtype=np.int64)
    for f in range(len(T)):
        for k in range(3):
            ends[int(per[f, k])] = (int(T[f, k]), int(T[f, (k + 1) % 3]))
    rhs = np.zeros((len(V), 3))
    deg = np.zeros(len(V))
    for ei in range(ne):
        a, b = ends[ei]
        rhs[a] += 2.0 * P[ei]
        rhs[b] += 2.0 * P[ei]
        deg[a] += 1.0
        deg[b] += 1.0
    deg = np.maximum(deg, 1.0)
    x = V - V.mean(0)
    for _ in range(400):
        nb = np.zeros((len(V), 3))
        for ei in range(ne):
            a, b = ends[ei]
            nb[a] += x[b]
            nb[b] += x[a]
        xn = (rhs - nb) / deg[:, None]
        if np.max(np.abs(xn - x)) < 1e-12:
            x = xn
            break
        x = xn
    return x - x.mean(0)


# Schoen's ring-like surfaces (the R family), by relaxation
# ==========================================================================
# A SECOND construction route, and the only one that reaches these.
#
# Schoen's R_I (= Schwarz H), R_II and R_III are, in his own words in the
# 1970 NASA catalogue, "assembled from ring-like surfaces, each bounded
# by the opposite parallel triangles of a prism" -- (pi/3,pi/3,pi/3) for
# R_I, (pi/2,pi/4,pi/4) for R_II, (pi/2,pi/3,pi/6) for R_III.  Karcher
# (1989) section 5.2 builds the same objects as annular Plateau problems
# and calls them triangular catenoids.
#
# The exact-Weierstrass route does NOT reach R_III.  Karcher says why,
# and it is a property of the surfaces rather than a gap in anyone's
# effort: "RIII is not cut by planar symmetry lines into simply
# connected pieces.  RII has such symmetry lines but the conjugate
# contour is complicated, in particular not a Nitsche graph."  Brakke's
# Surface Evolver datafiles state the same problem the same way -- two
# fixed parallel triangles, everything between them free -- so this is
# what those datafiles are, transcribed rather than read at runtime
# (the extension cannot depend on a mirror).
#
# WHAT IS AND IS NOT CLAIMED.  This produces a DISCRETE minimal surface,
# certified by area minimisation and the Pinkall-Polthier cotan flow,
# not by integrating an exact Weierstrass representation.  It is a
# weaker claim than the `hexagonal` spec rows make and the gate reflects
# that: convergence is measured on AREA, which settles to four or five
# figures, and not on median |H|, which on a relaxed mesh measures
# triangle quality as much as it measures the surface.
#
# References:
# - A. H. Schoen, "Infinite periodic minimal surfaces without
#   self-intersections", NASA TN D-5541 (1970), Table II -- the R family
#   and the triangles bounding each.
# - H. Karcher, "The triply periodic minimal surfaces of Alan Schoen and
#   their constant mean curvature companions", manuscripta math. 64
#   (1989) 291-357, section 5.2 -- triangular catenoids, and the
#   obstruction quoted above.
# - U. Pinkall and K. Polthier, "Computing discrete minimal surfaces and
#   their conjugates", Experimental Mathematics 2(1) (1993) -- the flow
#   `minimize_area` runs.
# - K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#   (1992); datafiles RII.fe and RIII.fe.

# key -> (label, triangle vertices in the z = 0 plane, prism height).
# The triangles are Schoen's, and the heights are Brakke's `#define HT`.
# The contour need not be convex, or even simple.  Schoen's I-series is
# bounded by ROSETTES -- closed polylines that return to the origin
# several times, two petals for I-6 and I-8 and four for I-9 -- which is
# the "figure-8 wireframe" whose soap film Schoen recorded finding in
# October 1970.  `build_annulus_grid` spans any closed loop, so these
# need contour data and nothing else.
RING_SURFACES = {
    'R1': ("Schoen R-I / Schwarz H (ring form)",
           ((0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(0.75))), 0.5),
    'R2': ("Schoen R-II (ring form, genus 9)",
           ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), 0.2),
    'R3': ("Schoen R-III (ring form, genus 13)",
           ((0.0, 0.0), (math.sqrt(3.0), 0.0), (0.0, 3.0)), 0.5),
    # Two-petal rosettes.  I-6 also ships as an exact Weierstrass row, so
    # it is kept here as a CROSS-CHECK between the two routes rather than
    # as a second way to offer the same surface.
    'I6': ("Schoen I-6 (ring form, genus 5)",
           ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
            (0.0, 0.0), (0.0, -1.0), (-1.0, -1.0), (-1.0, 0.0)), 0.8),
    'I8': ("Schoen I-8 (ring form)",
           ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0),
            (0.0, 0.0), (0.0, -1.0), (-0.5, -0.5), (-1.0, 0.0)), 0.5),
    # Four petals.
    'I9': ("Schoen I-9 (ring form)",
           ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
            (0.0, 0.0), (0.0, 1.0), (-1.0, 1.0),
            (0.0, 0.0), (-1.0, 0.0), (-1.0, -1.0),
            (0.0, 0.0), (0.0, -1.0), (1.0, -1.0)), 0.5),
}


# Half-turn generators for the ring surfaces, transcribed from the
# `view_transform_generators` blocks of Brakke's datafiles.  Each is a
# 180-degree rotation about one edge of the bounding prism: three about
# the edges of the bottom triangle, three about the top ones.  Reflecting
# the annulus in these is Schwarz's principle applied to the straight
# boundary lines, and it is what turns one "triangular catenoid" into a
# piece of the periodic surface.
#
# Stored as (3x3 rotation, translation), the two halves of the 4x4 the
# datafile prints.
def _halfturn_pair(rot, tx, ty, ht):
    """The datafile's generators come in pairs: one about a bottom edge
    and the same rotation about the corresponding top edge, differing
    only by a z offset of 2*HT."""
    return [(rot, (tx, ty, 0.0)), (rot, (tx, ty, 2.0 * ht))]


def ring_generators(key):
    """The tiling generators for one ring surface, or None.

    The R family and the I family do NOT share generators, and assuming
    they do is a silent error rather than a loud one: applying the R
    half-turns to I-8 welds into 244 over-shared edges, and to I-9 into
    14 266 while halving the vertex count -- coincident duplicate sheets,
    which no topological count catches on its own.  So each key declares
    its own, transcribed from its datafile's own
    `view_transform_generators` block, and a key with none returns None
    and is not tiled.

    Note the datafile writes its 4x4 matrices ROW-major and inline, so
    "1 0 0 2*xsize  0 1 0 0  0 0 1 0  0 0 0 1" is a translation by
    (2*xsize, 0, 0), not a shear.
    """
    _label, tri, ht = RING_SURFACES[key]
    rx = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    ry = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
    ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    rotz = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    if key == 'R3':
        c = math.sqrt(3.0)
        rc = ((-0.5, -c / 2.0, 0.0), (-math.sqrt(0.75), 0.5, 0.0),
              (0.0, 0.0, -1.0))
        return (_halfturn_pair(rx, 0.0, 0.0, ht)
                + _halfturn_pair(ry, 0.0, 0.0, ht)
                + _halfturn_pair(rc, 3.0 * c / 2.0, 1.5, ht))
    if key == 'R2':
        rc = ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        return (_halfturn_pair(rx, 0.0, 0.0, ht)
                + _halfturn_pair(ry, 0.0, 0.0, ht)
                + _halfturn_pair(rc, 1.0, 1.0, ht))
    if key == 'I6':
        return [(ident, (2.0, 0.0, 0.0)), (ident, (0.0, 2.0, 0.0)),
                (rotz, (0.0, 0.0, ht))]
    if key == 'I8':
        return [(ident, (2.0, 0.0, 0.0)),
                (rotz, (1.0, 1.0, 0.0)),          # translation with twist
                (rotz, (0.0, 0.0, ht))]
    if key == 'I9':
        # I-9's third generator is the MIRROR-like ((0,1,0),(1,0,0),z),
        # not the rotation the other two use -- the datafile prints
        # "0 1 0 0   1 0 0 0" where I-6 and I-8 print "-1 0 0 0".
        swapxy = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        return [(ident, (2.0, 0.0, 0.0)), (ident, (0.0, 2.0, 0.0)),
                (swapxy, (0.0, 0.0, ht))]
    return None


def ring_tile(V, quads, key, depth=2, tol=1e-7):
    """Close the group generated by `ring_generators` up to `depth`
    words and emit the welded orbit of the patch.

    The orbit is deduplicated by the transform, not by the mesh: two
    different words that give the same isometry would otherwise each
    contribute a copy, and coincident sheets are invisible to any
    topological count -- the failure that once rendered CLP-with-a-handle
    as leopard spots.
    """
    V = np.asarray(V, dtype=float)
    seen = {}
    ident = (np.eye(3), np.zeros(3))

    def keyof(R, t):
        return tuple(np.round(np.concatenate([R.ravel(), t]), 7))

    frontier = [ident]
    seen[keyof(*ident)] = ident
    raw = ring_generators(key)
    if raw is None:
        return V, [tuple(int(i) for i in q) for q in quads], 1
    gens = [(np.array(R, float), np.array(t, float)) for R, t in raw]
    for _ in range(depth):
        nxt = []
        for R0, t0 in frontier:
            for R1, t1 in gens:
                R = R1 @ R0
                t = R1 @ t0 + t1
                k = keyof(R, t)
                if k not in seen:
                    seen[k] = (R, t)
                    nxt.append((R, t))
        frontier = nxt
        if not frontier:
            break
    Vs, Qs, base = [], [], 0
    for R, t in seen.values():
        Vs.append(V @ R.T + t)
        flip = np.linalg.det(R) < 0.0
        for q in quads:
            f = tuple(int(i) + base for i in q)
            Qs.append(f[::-1] if flip else f)
        base += len(V)
    return np.concatenate(Vs, 0), Qs, len(seen)


def ring_tile_checked(V, quads, key, depth=2):
    """Tile, weld, and ACCEPT ONLY IF the result is a surface.

    The discriminator is DUPLICATE FACES, not over-shared edges, and the
    difference is the whole point.  These contours are rosettes: the
    boundary curve passes through the origin two or four times, so the
    assembled surface genuinely has lines where four sheets meet, at the
    origin and at every lattice translate of it.  Over-shared edges
    THERE are real geometry -- the same situation as the even-k
    Fischer-Koch surfaces, whose line self-intersection is a classical
    property rather than a meshing defect.  Rejecting on over-shared
    edges alone would therefore throw away correct tilings.

    Coincident duplicate COPIES are the actual failure, and they are
    what `hexagonal._assembly_ok` was written to catch after a stack of
    overlapping patches once shipped rendering as leopard spots.  They
    show up as faces sharing a centroid, which no component count and no
    edge count notices.

    Measured at depth 2, the two are cleanly separated:

        R-2, R-3, I-9   0 over-shared, 0 duplicate      -> accepted
        I-6          1390 over-shared, 368 duplicate    -> refused
        I-8            78 over-shared,   0 duplicate    -> refused

    I-6 fails on duplicates outright.  I-8 has none, but only 5% of its
    over-shared edges lie near a pinch line, so they are not the rosette
    self-touching either and its generator set is not yet right.  Both
    fall back to the single relaxed patch, which is honest: one annulus
    IS a piece of the surface, whereas a stack of overlapping copies is
    not.
    """
    try:
        VT, QT, n = ring_tile(V, quads, key, depth=depth)
    except Exception:
        return np.asarray(V, float), [tuple(int(i) for i in q)
                                      for q in quads], 1
    if n <= 1:
        return VT, QT, 1
    VT = np.asarray(VT, float)
    span = float(np.max(VT.max(0) - VT.min(0)))
    W, QQ = _weld_points(VT, QT, 1e-4 * max(span, 1e-12))

    cent = {}
    for f in QQ:
        c = tuple(np.round(W[list(f)].mean(0), 6))
        cent[c] = cent.get(c, 0) + 1
    if any(v > 1 for v in cent.values()):
        return np.asarray(V, float), [tuple(int(i) for i in q)
                                      for q in quads], 1

    ec = {}
    for f in QQ:
        m = len(f)
        for t in range(m):
            a, b = f[t], f[(t + 1) % m]
            e = (a, b) if a < b else (b, a)
            ec[e] = ec.get(e, 0) + 1
    # CONNECTEDNESS, for the same reason `hexagonal._assembly_ok` gained
    # it: I-8's depth-1 orbit welds into THREE pieces (12288 + 6144 +
    # 6144 vertices) while its single patch is one, and a surface that
    # falls apart is wrong however clean its edges are.
    nv = len(W)
    parent = list(range(nv))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for f in QQ:
        r0 = _find(int(f[0]))
        for j in range(1, len(f)):
            rj = _find(int(f[j]))
            if rj != r0:
                parent[rj] = r0
    if len({_find(int(i)) for f in QQ for i in f}) > 1:
        return np.asarray(V, float), [tuple(int(i) for i in q)
                                      for q in quads], 1

    over = [e for e, c in ec.items() if c > 2]
    if over:
        # tolerated only where they ARE the rosette's own pinch: within
        # a tenth of a cell of a lattice node of the 2x2 translations
        P = np.array([0.5 * (W[e[0]] + W[e[1]]) for e in over])
        d = np.hypot(((P[:, 0] + 1.0) % 2.0) - 1.0,
                     ((P[:, 1] + 1.0) % 2.0) - 1.0)
        if not np.all(d < 0.10):
            return np.asarray(V, float), [tuple(int(i) for i in q)
                                          for q in quads], 1
    return W, QQ, n


def _weld_points(V, faces, tol):
    """Merge vertices within `tol` and reindex."""
    keys = np.round(np.asarray(V, float) / max(tol, 1e-30)).astype(np.int64)
    uniq = {}
    remap = np.empty(len(V), dtype=np.int64)
    out = []
    for i, k in enumerate(map(tuple, keys)):
        j = uniq.get(k)
        if j is None:
            j = uniq[k] = len(out)
            out.append(V[i])
        remap[i] = j
    W = np.asarray(out, dtype=float)
    QQ = []
    for f in faces:
        g = [int(remap[i]) for i in f]
        d = [g[i] for i in range(len(g)) if g[i] != g[(i + 1) % len(g)]]
        if len(d) >= 3:
            QQ.append(tuple(d))
    return W, QQ


def ring_build(key, cells, res_per_cell, scale, theta):
    """TPMS_EXACT-compatible wrapper: relax, then centre and fit the
    result into the 2*scale cube the rest of the catalog uses.

    `cells` selects how many words of the symmetry group to close, and
    the tiling is accepted only if it welds into a surface (see
    `ring_tile_checked`); otherwise the single relaxed patch ships.
    `theta` is accepted and ignored -- it is meaningless here because
    there is no holomorphic family to rotate, which is precisely the
    difference between this route and the exact Weierstrass rows.
    """
    m = max(48, int(round(res_per_cell)) * 2)
    V, quads, _area = ring_surface(key, m=m, rows=max(10, m // 6))
    # `cells` drives how many words of the symmetry group to close, so
    # asking for more cells genuinely extends the surface instead of
    # being ignored -- but only where the tiling verifies.
    depth = int(np.clip(int(cells) if np.isscalar(cells) else
                        max(cells), 1, 3))
    V, quads, _n = ring_tile_checked(V, quads, key, depth=depth)
    V = np.asarray(V, dtype=float)
    lo, hi = V.min(0), V.max(0)
    V = V - 0.5 * (lo + hi)
    ext = float(np.max(hi - lo))
    if ext > 1e-12:
        V = V * (2.0 * float(scale) / ext)
    return V, [tuple(int(i) for i in q) for q in quads]


def ring_surface(key, m=120, rows=20, iters=150):
    """Relax the minimal annulus spanning two parallel congruent
    triangles.  Returns (V, quads, area)."""
    _label, tri, ht = RING_SURFACES[key]
    pts = np.array([(x, y, 0.0) for x, y in tri], dtype=float)
    loop = resample_loop(np.vstack([pts, pts[:1]]), m)
    top = loop.copy()
    top[:, 2] = ht
    V, quads, fixed = build_annulus_grid(loop, top, rows)
    T = _quads_to_tris(quads)
    V = np.asarray(minimize_area(np.asarray(V, float).copy(), T, fixed,
                                 outer_iters=iters), dtype=float)
    return V, quads, mesh_area(V, T)


def _selftest():
    ok = True

    # resample_loop must lay points at uniform arclength and keep them on the
    # curve.  Sampled from a fine polygon so the chord-vs-arc discrepancy at
    # the input corners (which is real, and O(1/n^2)) stays well below the
    # non-uniformity a broken resampler would produce.
    ang = np.linspace(0.0, TAU, 512, endpoint=False)
    circ = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
    r = resample_loop(circ, 64)
    step = np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1)
    spread = float(np.ptp(step)) / float(np.mean(step))
    oncurve = abs(float(np.mean(np.hypot(r[:, 0], r[:, 1]))) - 1.0)
    good = len(r) == 64 and spread < 1e-3 and oncurve < 1e-3
    ok &= good
    print(f"plateau: resample_loop n={len(r)} step spread={spread:.2e} "
          f"|r-1|={oncurve:.2e} {'OK' if good else 'FAIL'}")

    # mesh_area on a unit square split into two triangles is exactly 1.
    Vs = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0], [0.0, 1.0, 0]])
    Ts = np.array([[0, 1, 2], [0, 2, 3]])
    a = mesh_area(Vs, Ts)
    good = abs(a - 1.0) < 1e-12
    ok &= good
    print(f"plateau: mesh_area unit square={a:.12f} {'OK' if good else 'FAIL'}")

    # The real gate: area minimization between two coaxial circles must
    # converge to the catenoid.  For radius 1 rings at z = +-0.5 the neck of
    # the minimal catenoid solves r0 cosh(0.5/r0) = 1 -> r0 = 0.8483...,
    # with area 2*pi*r0*(r0*sinh(1/r0)/... ) -- checked here structurally:
    # the solve must (a) reduce area, (b) leave the pinned rims untouched,
    # (c) pull the waist inside the initial cylinder.
    h, nring, nrow = 0.5, 48, 13
    th = np.linspace(0.0, TAU, nring, endpoint=False)
    zs = np.linspace(-h, h, nrow)
    V = np.array([[math.cos(t), math.sin(t), z] for z in zs for t in th])
    T = []
    for j in range(nrow - 1):
        for i in range(nring):
            a0 = j * nring + i
            b0 = j * nring + (i + 1) % nring
            T.append([a0, b0, a0 + nring])
            T.append([b0, b0 + nring, a0 + nring])
    T = np.array(T)
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:nring] = True
    fixed[-nring:] = True
    rim0 = V[:nring].copy()
    a0 = mesh_area(V, T)
    minimize_area(V, T, fixed, outer_iters=40)
    a1 = mesh_area(V, T)
    waist = float(np.mean(np.hypot(V[nrow // 2 * nring:(nrow // 2 + 1) * nring, 0],
                                   V[nrow // 2 * nring:(nrow // 2 + 1) * nring, 1])))
    rim_moved = float(np.max(np.abs(V[:nring] - rim0)))
    r0 = 0.8483
    good = (a1 < a0 and rim_moved < 1e-12 and abs(waist - r0) < 0.02
            and bool(np.all(np.isfinite(V))))
    ok &= good
    print(f"plateau: catenoid area {a0:.4f} -> {a1:.4f}, waist={waist:.4f} "
          f"(exp {r0:.4f}), rims pinned ({rim_moved:.1e}) "
          f"{'OK' if good else 'FAIL'}")

    # The same solve with in-loop grooming (flips + tangential
    # averaging) must still converge to the catenoid with pinned rims,
    # and the historical clamped mode must keep working as a fallback.
    for label, kw in (("groomed", dict(groom_every=4)),
                      ("clamped", dict(cotan_mode="clamp"))):
        V2 = np.array([[math.cos(t), math.sin(t), z]
                       for z in zs for t in th])
        T2 = T.copy()
        rimA = V2[:nring].copy()
        minimize_area(V2, T2, fixed, outer_iters=40, **kw)
        waist2 = float(np.mean(np.hypot(
            V2[nrow // 2 * nring:(nrow // 2 + 1) * nring, 0],
            V2[nrow // 2 * nring:(nrow // 2 + 1) * nring, 1])))
        moved2 = float(np.max(np.abs(V2[:nring] - rimA)))
        good = (abs(waist2 - r0) < 0.02 and moved2 < 1e-12
                and bool(np.all(np.isfinite(V2))))
        ok &= good
        print(f"plateau: catenoid ({label}) waist={waist2:.4f}, rims "
              f"pinned ({moved2:.1e}) {'OK' if good else 'FAIL'}")

    # The R family is certified by AREA, not by median |H|.  On a mesh
    # produced by relaxation rather than by integration, |H| measures
    # triangle quality as much as it measures the surface -- R-II's rises
    # under refinement while its area settles to five figures -- so area
    # is the honest invariant and the one gated here.
    for key in ('R1', 'R2', 'R3', 'I6', 'I8', 'I9'):
        areas = []
        for m, rows in ((96, 16), (144, 24)):
            _V, _Q, a = ring_surface(key, m=m, rows=rows)
            areas.append(a)
        drift = abs(areas[1] - areas[0]) / max(areas[1], 1e-30)
        good = drift < 0.01 and areas[1] > 1e-6
        ok &= good
        print("plateau: ring %s area %.6f -> %.6f (drift %.4f%%) %s"
              % (key, areas[0], areas[1], 100.0 * drift,
                 'OK' if good else 'FAIL'))

    # The tiling must be ACCEPTED where the generators are right and
    # REFUSED where they are not.  Both halves matter: a check that only
    # ever accepts would have passed the first version of
    # `ring_generators`, which applied the R family's half-turns to the
    # I family and welded I-9 into 14 266 over-shared edges while halving
    # its vertex count -- coincident sheets, invisible to a component
    # count.
    for key, expect_tiled in (('R2', True), ('R3', True),
                              ('I9', True), ('I8', False), ('I6', False)):
        V, Q, _a = ring_surface(key, m=96, rows=16)
        W, QQ, n = ring_tile_checked(V, Q, key, depth=2)
        ec = {}
        for f in QQ:
            mm = len(f)
            for t in range(mm):
                x, y = f[t], f[(t + 1) % mm]
                e = (x, y) if x < y else (y, x)
                ec[e] = ec.get(e, 0) + 1
        over = sum(1 for c in ec.values() if c > 2)
        tiled = n > 1
        good = (over == 0) and (tiled == expect_tiled)
        ok &= good
        print("plateau: ring %s tiling %s (%d copies, %d over-shared) %s"
              % (key, "accepted" if tiled else "refused -> patch",
                 n, over, 'OK' if good else 'FAIL'))

    # The discrete adjoint: only the two facts that ARE established.
    a3 = np.array([0., 0., 0.])
    b3 = np.array([1., 0., 0.])
    c3 = np.array([0.3, 0.9, 0.2])
    Vt = np.array([a3, b3, c3])
    Tt = np.array([[0, 1, 2]])
    ev = [b3 - a3, c3 - b3, a3 - c3]
    nn = np.cross(ev[0], ev[1])
    nn = nn / np.linalg.norm(nn)
    worst_close = 0.0
    worst_len = 0.0
    for ang in (0.0, 30.0, 90.0):
        bc = math.cos(math.radians(ang))
        bs = math.sin(math.radians(ang))

        def _st(eo, _bc=bc, _bs=bs):
            return -0.5 * (_bc * eo + _bs * np.cross(nn, eo))

        Pp = [np.zeros(3), None, None]
        Pp[1] = Pp[0] + _st(ev[2])
        Pp[2] = Pp[1] + _st(ev[0])
        worst_close = max(worst_close,
                          float(np.linalg.norm(Pp[2] + _st(ev[1]) - Pp[0])))
        for k, want in enumerate((ev[2], ev[0], ev[1])):
            got = float(np.linalg.norm(Pp[(k + 1) % 3] - Pp[k]))
            worst_len = max(worst_len,
                            abs(got - 0.5 * float(np.linalg.norm(want))))
    good = worst_close < 1e-12 and worst_len < 1e-12
    ok &= good
    print("plateau: adjoint facet relation closes %.1e, medial lengths "
          "|e|/2 to %.1e %s"
          % (worst_close, worst_len, 'OK' if good else 'FAIL'))

    # ...and bangle 0 must be the identity on a real mesh.
    tt = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    lo = np.stack([np.cos(tt), np.sin(tt), np.zeros_like(tt)], 1)
    hi = lo.copy()
    hi[:, 2] = 1.0
    Vc, qc, fx = build_annulus_grid(lo, hi, 8)
    Tc = _quads_to_tris(qc)
    Vc = np.asarray(minimize_area(np.asarray(Vc, float).copy(), Tc, fx,
                                  outer_iters=80), dtype=float)
    Wc = discrete_adjoint(Vc, Tc, bangle=0.0)
    da = abs(mesh_area(Wc, Tc) / max(mesh_area(Vc, Tc), 1e-30) - 1.0)
    good = da < 1e-9
    ok &= good
    print("plateau: adjoint at bangle 0 is the identity (area ratio "
          "off by %.1e) %s" % (da, 'OK' if good else 'FAIL'))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("plateau self-test failed")
