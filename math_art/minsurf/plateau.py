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
import numpy as np

try:
    from ..solver import cotan as _sc
except ImportError:                      # flat (path-based) headless import
    from solver import cotan as _sc

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

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("plateau self-test failed")
