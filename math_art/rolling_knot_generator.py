
# Smooth-Rolling Knot generator for Blender, after Brodeur,
# Vidulis, Dandy & Pauly, "Smooth-Rolling Knots" (Bridges 2025),
# building on Morton's tritangentless trefoils, the rolling
# analysis of Eget, Lucas & Taalman (Bridges 2020) and the
# smooth-rolling Two-Disk Rollers of Engelhardt & Ucke.
#
# Morton's (p,2) torus knots roll on a plane but their centre
# of mass bobs up and down.  Following the paper, the knot is
# stretched and its two exterior lobes are pinned onto the two
# orthogonal ellipses of a smooth-rolling Two-Disk Roller (TDR,
# half axes alpha/beta, centre distance gamma, smooth-rolling
# when gamma^2 = 4 alpha^2 - 2 beta^2); the interior is morphed
# smoothly inside the TDR's convex hull.  The centre of mass
# then stays at constant height while rolling (rho = 0).
#
# THICKNESS.  The paper (and prior work) treats the knot as an
# ideal curve.  For a physical tube of radius r two exact facts
# make the ideal solution surprisingly robust: the support
# function of the thickened solid is the curve's plus r in every
# direction (Minkowski sum with a ball), and the centre of mass
# of a closed circular tube equals the curve's arc-length
# centroid exactly (the curvature term integrates to zero around
# a closed loop).  What DOES break smooth rolling is strand
# fusion: where tubes overlap, the merged volume is counted once
# and the solid's centre of mass shifts.  This generator
# measures that shift exactly (arc centroid plus a fine, local
# overlap correction), rebalances the interior of the knot so
# the thick solid's centre of mass returns to the rolling
# centre, and reports the achieved rho for both the ideal curve
# and the thick solid.
#
# FAIRNESS.  Rolling correctly and LOOKING right are different
# problems, and the stages above solve only the first: they leave
# the free interior carrying a curvature profile that oscillates
# between the taut pinned lobes, which reads as straight runs
# meeting at corners rather than one continuous sweep.  The paper
# optimizes a spline whose energy includes a curvature term at
# the interior/exterior junction (Eq. 3); this implementation
# instead closes with a separate fairing pass -- a true
# arc-length bending-energy solve over the free interior, with
# the contact arcs held -- and keeps its result only when the
# curve comes out fairer WITHOUT costing rolling accuracy or
# strand clearance.  `curvature_variation` is the measure used,
# and `info['tv_kappa_raw']`/`info['tv_kappa']` report it either
# side of the pass.  See the stage-3 comment in
# `build_rolling_knot` for why the earlier stages cannot do this
# themselves.

bl_info = {
    "name": "Rolling Knot",
    "author": "Math Art project (after Brodeur, Vidulis, "
              "Dandy & Pauly)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > "
                "Knots & Curves",
    "description": "Smooth-rolling knots: Morton (p,2) knots "
                   "morphed onto a two-disk roller, optimized "
                   "for the actual tube thickness",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty,
                           EnumProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# ---------------------------------------------------------------- #
#  geometry core (no bpy)                                          #
# ---------------------------------------------------------------- #

def morton(p, a, zs=1.0, n=512):
    """Morton's (p,2) knot (trefoil for p=3), z scaled by zs."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    b = np.sqrt(1 - a * a)
    c = a / (1 + b)
    den = 1 - b * np.sin(2 * t)
    return np.stack([c * a * np.cos(p * t) / den,
                     c * a * np.sin(p * t) / den,
                     zs * c * b * np.cos(2 * t) / den], 1)


def _fib_dirs(m):
    i = np.arange(m) + 0.5
    ph = np.arccos(1 - 2 * i / m)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph),
                     np.sin(th) * np.sin(ph), np.cos(ph)], 1)


def _runs_of(mask):
    n = len(mask)
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    br = np.where(np.diff(idx) > 4)[0]
    runs = np.split(idx, br + 1)
    if len(runs) > 1 and idx[0] == 0 and idx[-1] == n - 1:
        runs[0] = np.concatenate([runs[-1], runs[0]])
        runs = runs[:-1]
    return runs


def lobe_runs(K, m=6000):
    """The two exterior lobes: maximal runs of curve samples
    that appear on the convex hull, plus the centre-line
    direction between their centres."""
    U = _fib_dirs(m)
    am = np.argmax(U @ K.T, axis=1)
    ext = np.zeros(len(K), bool)
    ext[np.unique(am)] = True
    runs = sorted(_runs_of(ext), key=len)[-2:]
    cens = [K[r].mean(0) for r in runs]
    e1 = cens[1] - cens[0]
    e1[2] = 0.0
    e1 /= np.linalg.norm(e1)
    if (cens[0] @ e1) > 0:
        runs = runs[::-1]
    return runs, e1


class TDR:
    """Smooth-rolling two-disk roller: two identical ellipses in
    perpendicular planes sharing the centre line e1, half axis
    alpha on the line and beta across, centres gamma apart with
    gamma^2 = 4 alpha^2 - 2 beta^2 (Engelhardt & Ucke)."""

    def __init__(self, e1, alpha, beta, tilt=1.0):
        z = np.array([0.0, 0.0, 1.0])
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(z, self.e1)
        self.alpha, self.beta = alpha, beta
        self.gamma = np.sqrt(
            max(4 * alpha ** 2 - 2 * beta ** 2, 1e-12))
        self.n = [(self.e2 + tilt * z) / np.sqrt(2),
                  (self.e2 - tilt * z) / np.sqrt(2)]
        self.v = [np.cross(self.n[0], self.e1),
                  np.cross(self.n[1], self.e1)]
        self.c = [-0.5 * self.gamma * self.e1,
                  +0.5 * self.gamma * self.e1]

    def point(self, side, th):
        return (self.c[side]
                + self.alpha * np.cos(th)[:, None] * self.e1
                + self.beta * np.sin(th)[:, None]
                * self.v[side])

    def closest(self, side, P):
        d = P - self.c[side]
        xi = d @ self.e1
        eta = d @ self.v[side]
        th = np.arctan2(eta * self.alpha, xi * self.beta)
        for _ in range(15):
            ct, st = np.cos(th), np.sin(th)
            f = ((self.alpha ** 2 - self.beta ** 2) * ct * st
                 - xi * self.alpha * st + eta * self.beta * ct)
            fp = ((self.alpha ** 2 - self.beta ** 2)
                  * (ct * ct - st * st)
                  - xi * self.alpha * ct
                  - eta * self.beta * st)
            th = th - f / np.where(np.abs(fp) < 1e-12,
                                   1e-12, fp)
        q = self.point(side, th)
        return q, np.sum((P - q) ** 2, 1), th


def _nelder_mead(f, x0, step, iters=220):
    n = len(x0)
    xs = [np.array(x0, float)]
    for i in range(n):
        x = np.array(x0, float)
        x[i] += step[i]
        xs.append(x)
    fs = [f(x) for x in xs]
    for _ in range(iters):
        o = np.argsort(fs)
        xs = [xs[i] for i in o]
        fs = [fs[i] for i in o]
        c = np.mean(xs[:-1], axis=0)
        xr = c + (c - xs[-1])
        fr = f(xr)
        if fr < fs[0]:
            xe = c + 2 * (c - xs[-1])
            fe = f(xe)
            xs[-1], fs[-1] = ((xe, fe) if fe < fr
                              else (xr, fr))
        elif fr < fs[-2]:
            xs[-1], fs[-1] = xr, fr
        else:
            xc = c + 0.5 * (xs[-1] - c)
            fc = f(xc)
            if fc < fs[-1]:
                xs[-1], fs[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    xs[i] = xs[0] + 0.5 * (xs[i] - xs[0])
                    fs[i] = f(xs[i])
    o = np.argsort(fs)
    return xs[o[0]], fs[o[0]]


def rho_eval(K, com, runs, rt=0.0, m=3000, newton=4):
    """Range-to-average ratio of the centre-of-mass height over
    the rolling motion: directions are refined onto the exact
    bitangent band (equal support on both lobes), the physical
    resting states of a two-lobed roller.  rt adds the tube
    radius (Minkowski offset) to every support height."""
    A = K - com
    m0 = np.zeros(len(K), bool)
    m0[runs[0]] = True
    m1 = np.zeros(len(K), bool)
    m1[runs[1]] = True
    i0all = np.where(m0)[0]
    i1all = np.where(m1)[0]
    size = np.linalg.norm(A, axis=1).max()
    heights = []
    for u in _fib_dirs(m):
        d = A @ u
        h0 = d[m0].max()
        h1 = d[m1].max()
        if abs(h0 - h1) > 0.08 * size:
            continue
        ok = True
        for _ in range(newton):
            d = A @ u
            i0 = i0all[d[m0].argmax()]
            i1 = i1all[d[m1].argmax()]
            w = A[i0] - A[i1]
            diff = d[i0] - d[i1]
            wt = w - (u @ w) * u
            den = wt @ wt
            if den < 1e-12:
                ok = False
                break
            u = u - (diff / den) * wt
            u /= np.linalg.norm(u)
        if not ok:
            continue
        d = A @ u
        h0 = d[m0].max()
        h1 = d[m1].max()
        if abs(h0 - h1) > 1e-5 * size:
            continue
        gm = d.max()
        if gm > max(h0, h1) + 5e-3 * size:
            continue
        heights.append(gm + rt)
    z = np.array(heights)
    if len(z) < 10:
        return np.inf, 0
    return (z.max() - z.min()) / z.mean(), len(z)


def com_thick(K, rt):
    """Centre of mass of the tube of radius rt around the
    closed curve K.  Exactly the arc-length centroid, minus the
    mass over-count where distant strands come closer than 2 rt
    (fused regions weigh once); that correction is integrated on
    a fine grid restricted to the near-contact zones.  Returns
    (com, overlap_fraction)."""
    n = len(K)
    seg = np.linalg.norm(np.roll(K, -1, 0) - K, axis=1)
    w = (seg + np.roll(seg, 1)) / 2
    L = w.sum()
    c_arc = (K * w[:, None]).sum(0) / L
    if rt <= 0:
        return c_arc, 0.0
    # distant-strand near contacts
    gap = max(4, n // 12)
    D = np.linalg.norm(K[:, None, :] - K[None, :, :], axis=2)
    pi_ = np.arange(n)
    circ = np.abs(pi_[:, None] - pi_[None, :])
    circ = np.minimum(circ, n - circ)
    close = (D < 2.0 * rt) & (circ > gap)
    ii, jj = np.where(close)
    if not len(ii):
        return c_arc, 0.0
    mids = 0.5 * (K[ii] + K[jj])
    h = rt / 5.0
    keys = set()
    off = np.arange(-6, 7)
    OX, OY, OZ = np.meshgrid(off, off, off, indexing='ij')
    stencil = np.stack([OX.ravel(), OY.ravel(),
                        OZ.ravel()], 1)
    stencil = stencil[np.linalg.norm(stencil, axis=1)
                      <= 6.01]
    for mfrac in mids[::max(1, len(mids) // 400)]:
        base = np.floor(mfrac / h).astype(int)
        for s in stencil:
            keys.add(tuple(base + s))
    if not keys:
        return c_arc, 0.0
    V = np.array(sorted(keys), float) * h + h / 2
    excess = np.zeros(len(V))
    for chunk in range(0, len(V), 8192):
        Pv = V[chunk:chunk + 8192]
        d2 = ((Pv[:, None, :] - K[None, :, :]) ** 2).sum(2)
        near = d2 <= rt * rt
        for k in range(len(Pv)):
            idx = np.where(near[k])[0]
            if len(idx) < 2:
                continue
            gaps = np.diff(idx)
            nclust = 1 + (gaps > gap).sum()
            if idx[0] + n - idx[-1] <= gap and nclust > 1:
                nclust -= 1
            if nclust >= 2:
                excess[chunk + k] = nclust - 1
    dv = h ** 3
    m_over = excess.sum() * dv
    if m_over <= 0:
        return c_arc, 0.0
    v_tube = math.pi * rt * rt * L
    c_over = (V * excess[:, None]).sum(0) * dv / m_over
    denom = v_tube - m_over
    com = (c_arc * v_tube - c_over * m_over) / denom
    return com, m_over / v_tube


def _bend_operator(K):
    """Arc-length second-derivative operator D2 and vertex masses M
    for a closed polyline: (D2 x)_i is the three-point estimate of
    x''(s) on the UNEVEN spacing actually present, so x^T D2^T M D2 x
    is the discrete bending energy integral |x''(s)|^2 ds.

    The distinction from the index-space operator in `solve` is the
    whole point of the fairing stage -- see the stage-3 comment."""
    n = len(K)
    hh = np.linalg.norm(np.roll(K, -1, 0) - K, axis=1)
    hh = np.maximum(hh, 1e-300)
    D2 = np.zeros((n, n))
    i = np.arange(n)
    hm, hp = hh[(i - 1) % n], hh[i]
    D2[i, (i - 1) % n] = 2.0 / (hm * (hm + hp))
    D2[i, i] = -2.0 / (hm * hp)
    D2[i, (i + 1) % n] = 2.0 / (hp * (hm + hp))
    return D2, 0.5 * (hh + np.roll(hh, 1))


def curvature_variation(K):
    """Total variation of discrete curvature around a closed polyline,
    sum |kappa_i+1 - kappa_i| with kappa_i the turning angle over the
    dual edge length.

    This is the standard fairness functional (integral |dkappa/ds|):
    a curve that sweeps has slowly varying curvature, so a LOWER value
    is a smoother-looking curve.  Cheap enough to evaluate inside the
    fairing loop, which is what lets that loop detect its own failure
    mode and back off."""
    e = np.roll(K, -1, 0) - K
    h = np.maximum(np.linalg.norm(e, axis=1), 1e-300)
    T = e / h[:, None]
    turn = np.arccos(np.clip((T * np.roll(T, 1, 0)).sum(1), -1.0, 1.0))
    kap = turn / (0.5 * (h + np.roll(h, 1)))
    return float(np.abs(np.diff(np.concatenate([kap, kap[:1]]))).sum())


def _max_turn(K):
    """Largest turning angle (radians) between consecutive segments of
    a closed polyline -- the faceting the swept tube shows."""
    e = np.roll(K, -1, 0) - K
    T = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-300)
    return float(np.arccos(np.clip((T * np.roll(T, 1, 0)).sum(1),
                                   -1.0, 1.0)).max())


def resample_uniform(K, m, runs=None):
    """Re-sample a closed polyline at m points of uniform ARC LENGTH,
    interpolating it with a periodic cubic spline where that helps and
    simply subdividing it where it does not.

    `morton` samples uniformly in the knot's parameter, and the
    rational denominator makes that sparsest exactly on the outer
    lobes -- the part of the silhouette the eye lands on.  Measured on
    the (3,2) default, vertex spacing varies about 12:1 and the worst
    polyline turn is ~4.8 degrees per segment ON THE LOBES, which is a
    chord sagitta the swept tube shows as faceting however smoothly it
    is shaded, while the interior is over-tessellated for no benefit.
    Re-sampling evens that out.

    With `runs`, the index runs marking the contact cores are carried
    across by arc-length fraction and returned alongside, so the
    rolling diagnostics still describe the curve that ships."""
    K = np.asarray(K, dtype=float)
    n = len(K)
    m = int(m)
    h = np.linalg.norm(np.roll(K, -1, 0) - K, axis=1)
    h = np.maximum(h, 1e-300)
    t = np.concatenate([[0.0], np.cumsum(h)])
    L = t[-1]

    # periodic cubic spline: second derivatives from the cyclic
    # tridiagonal system (dense solve -- n is a few hundred, and this
    # module already factors matrices of exactly this size)
    i = np.arange(n)
    hm, hp = h[(i - 1) % n], h[i]
    A = np.zeros((n, n))
    A[i, (i - 1) % n] += hm
    A[i, i] += 2.0 * (hm + hp)
    A[i, (i + 1) % n] += hp
    rhs = 6.0 * ((np.roll(K, -1, 0) - K) / hp[:, None]
                 - (K - np.roll(K, 1, 0)) / hm[:, None])
    M = np.linalg.solve(A, rhs)

    # evaluate densely in the chord parameter, then walk that dense
    # polyline to place the m samples at equal arc length
    dense = max(8 * m, 4 * n)
    td = np.linspace(0.0, L, dense, endpoint=False)
    seg = np.clip(np.searchsorted(t, td, side='right') - 1, 0, n - 1)
    u = (td - t[seg])[:, None]
    hs = h[seg][:, None]
    yi, yj = K[seg], K[(seg + 1) % n]
    Mi, Mj = M[seg], M[(seg + 1) % n]
    b = (yj - yi) / hs - hs * (2.0 * Mi + Mj) / 6.0
    P = yi + b * u + 0.5 * Mi * u ** 2 + (Mj - Mi) / (6.0 * hs) * u ** 3

    hd = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    hd = np.maximum(hd, 1e-300)
    sd = np.concatenate([[0.0], np.cumsum(hd)])
    Ld = sd[-1]
    want = np.linspace(0.0, Ld, m, endpoint=False)
    j = np.searchsorted(sd, want, side='right') - 1
    j = np.clip(j, 0, dense - 1)
    frac = (want - sd[j]) / hd[j]
    out = P[j] + frac[:, None] * (P[(j + 1) % dense] - P[j])

    # ...and the same thing WITHOUT interpolating, by walking the
    # original polygon.  A cubic through data that has a genuine
    # corner in it rings: on the kinkier knots the spline made the
    # worst turn angle worse rather than better ((5,2) at a = 0.4:
    # 17.0 -> 28.3 degrees), because it manufactures overshoot the
    # polygon never had.  Subdividing cannot do that -- every sample
    # lands on an existing edge -- so compute both and keep whichever
    # actually turns more gently.  On a curve that is already fair the
    # spline wins comfortably ((3,2): 4.80 -> 2.38 degrees).
    wl = np.linspace(0.0, L, m, endpoint=False)
    jl = np.clip(np.searchsorted(t, wl, side='right') - 1, 0, n - 1)
    fl = (wl - t[jl]) / h[jl]
    lin = K[jl] + fl[:, None] * (K[(jl + 1) % n] - K[jl])
    if _max_turn(lin) < _max_turn(out):
        out, fw = lin, wl / L
    else:
        fw = want / Ld
    if runs is None:
        return out

    # carry the core runs over by arc-length fraction (they are
    # contiguous, possibly wrapping)
    new_runs = []
    for r in runs:
        f0, f1 = t[r[0]] / L, t[r[-1]] / L
        sel = ((fw >= f0) | (fw <= f1)) if f1 < f0 \
            else ((fw >= f0) & (fw <= f1))
        new_runs.append(np.where(sel)[0])
    return out, new_runs


def build_rolling_knot(p=3, a=0.5, mode='SMOOTH', rt=0.05,
                       thick_aware=True, n=512, w_lap=600.0,
                       int_weight=0.4, clearance=0.0,
                       balance_iters=4, w_fair=1e-4,
                       fair_rounds=3, fair_erode=2,
                       fair_data=0.05, resample=2.0):
    """(K, info): the optimized closed centreline and a dict of
    diagnostics (rho values, TDR parameters, overlap)."""
    K0 = morton(p, a, 1.0, n)
    runs0, e1 = lobe_runs(K0)
    info = {}
    com0, _ = com_thick(K0, 0)
    info['rho_raw'] = rho_eval(K0, com0, runs0)[0]
    if mode == 'MORTON':
        cm, ov = com_thick(K0, rt)
        info['rho'] = info['rho_raw']
        info['rho_thick'] = rho_eval(K0, cm, runs0, rt)[0]
        info['overlap'] = ov
        return K0, info

    # ----- stage 1: fit the smooth-rolling TDR (zs, al, be)
    nrm = np.linalg.svd(
        (K0[runs0[0]] - K0[runs0[0]].mean(0)),
        full_matrices=False)[2][2]
    e2 = np.cross([0.0, 0.0, 1.0], e1)
    if nrm @ e2 < 0:
        nrm = -nrm
    tilt = 1.0 if nrm[2] > 0 else -1.0
    zs0 = abs(nrm[2]) / max(np.linalg.norm(nrm[:2]), 1e-9)
    P = K0[np.concatenate(runs0)]
    b0 = np.sqrt(2) * np.abs(P[:, 2] * zs0).max()
    T = np.abs(P @ e1).max()
    al = max(T / 2, b0 / np.sqrt(2) + 1e-3)
    for _ in range(60):
        g = np.sqrt(max(4 * al * al - 2 * b0 * b0, 1e-12))
        al += 0.5 * (T - (g / 2 + al))
    a0 = al

    def cost(x):
        zs, al, be = x
        if not (0.3 <= zs <= 3.0) or al <= 1e-3 \
                or be <= 1e-3 \
                or 4 * al * al - 2 * be * be <= 1e-9:
            return 1e9
        K = K0 * np.array([1, 1, zs])
        tdr = TDR(e1, al, be, tilt)
        tot = 0.0
        for side, r in enumerate(runs0):
            _, d2, _ = tdr.closest(side, K[r])
            tot += np.mean(d2)
        return tot

    x, resid = _nelder_mead(cost, [zs0, a0, b0],
                            [0.05, 0.05, 0.05])
    zs, al, be = x
    info['tdr'] = (zs, al, be,
                   np.sqrt(max(4 * al * al - 2 * be * be, 0)))
    info['fit_resid'] = resid
    K = K0 * np.array([1, 1, zs])
    if mode == 'STRETCHED':
        runs, _ = lobe_runs(K)
        cmc, _ = com_thick(K, 0)
        info['rho'] = rho_eval(K, cmc, runs)[0]
        cm, ov = com_thick(K, rt)
        info['rho_thick'] = rho_eval(K, cm, runs, rt)[0]
        info['overlap'] = ov
        return K, info
    tdr = TDR(e1, al, be, tilt)
    runs, _ = lobe_runs(K)

    # ----- needed contact arcs (rolling touches ~240 degrees)
    th_s = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    E = np.vstack([tdr.point(0, th_s), tdr.point(1, th_s)])
    eruns = [np.arange(256), np.arange(256, 512)]
    need = []
    A = E
    m0 = np.zeros(512, bool)
    m0[:256] = True
    hitth = [[], []]
    for u in _fib_dirs(2000):
        d = A @ u
        h0 = d[:256].max()
        h1 = d[256:].max()
        if abs(h0 - h1) > 0.08:
            continue
        for _ in range(4):
            d = A @ u
            i0 = d[:256].argmax()
            i1 = 256 + d[256:].argmax()
            w = A[i0] - A[i1]
            wt = w - (u @ w) * u
            den = wt @ wt
            if den < 1e-12:
                break
            u = u - ((d[i0] - d[i1]) / den) * wt
            u /= np.linalg.norm(u)
        d = A @ u
        if abs(d[:256].max() - d[256:].max()) > 1e-5:
            continue
        hitth[0].append(th_s[d[:256].argmax()])
        hitth[1].append(th_s[d[256:].argmax() % 256])
    for side in range(2):
        th_hit = np.array(hitth[side])
        cm_a = np.arctan2(np.sin(th_hit).mean(),
                          np.cos(th_hit).mean())
        rel = (th_hit - cm_a + np.pi) % (2 * np.pi) - np.pi
        need.append((cm_a + rel.min(), cm_a + rel.max()))

    # ----- stage 2: pin grown lobes onto the contact arcs by
    # arc length.  Only the CORE of each zone (covering the
    # contact range) is clamped; hinge zones at both ends stay
    # free with targets continuing along the ellipse and a
    # tapering data weight, so the curve peels off the ellipse
    # smoothly instead of kinking at the junction.
    K2 = K.copy()
    runs2 = []
    wdat = np.full(n, int_weight)
    for side, r in enumerate(runs):
        lo, ln = r[0], len(r)
        grow = ln // 2
        lo = (lo - grow) % n
        ln = min(ln + 2 * grow, n // 2 - 8)
        hl = max(4, ln // 6)
        idx = np.array([(lo + k) % n for k in range(ln)])
        ta = need[side][0] - 0.05
        tb = need[side][1] + 0.05
        Pr = K[idx]
        s = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(Pr, axis=0), axis=1))])
        s /= s[-1]
        # core maps onto [ta, tb]; hinges continue beyond
        th_map = ta + (tb - ta) * ((s - s[hl])
                                   / (s[ln - hl - 1] - s[hl]))
        q_f = tdr.point(side, th_map)
        q_b = tdr.point(side, th_map[::-1])
        K2[idx] = (q_f if np.sum((q_f - Pr) ** 2)
                   <= np.sum((q_b - Pr) ** 2) else q_b)
        runs2.append(idx[hl:ln - hl])
        # data-weight taper: ellipse-hugging next to the core,
        # fading to the knot target across the hinge and the
        # first interior stretch
        for kk in range(hl):
            f = (kk + 1.0) / (hl + 1.0)
            wdat[idx[hl - 1 - kk]] = 1.0 - f
            wdat[idx[ln - hl + kk]] = 1.0 - f
        for kk in range(hl):
            f = (kk + 1.0) / (hl + 1.0)
            wdat[(idx[0] - 1 - kk) % n] = f
            wdat[(idx[-1] + 1 + kk) % n] = f

    ext = np.zeros(n, bool)
    for r in runs2:
        ext[r] = True
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = -2
        D[i, (i + 1) % n] = 1
        D[i, (i - 1) % n] = 1
    DTD = D.T @ D
    fix = np.where(ext)[0]
    q0 = K2.copy()

    def solve(target, q0v, wvec):
        Kn = K2.copy()
        seg = np.linalg.norm(np.roll(Kn, -1, 0) - Kn, axis=1)
        w = (seg + np.roll(seg, 1)) / 2
        w = w / w.sum()
        w_b = 1e6
        Am = (np.diag(wvec) + w_lap * DTD
              + w_b * np.outer(w, w))
        for dim in range(3):
            Af = Am.copy()
            bf = wvec * q0v[:, dim] + w_b * target[dim] * w
            for i in fix:
                Af[i, :] = 0
                Af[i, i] = 1
                bf[i] = q0v[i, dim]
            Kn[:, dim] = np.linalg.solve(Af, bf)
        return Kn

    th_f = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    Ef = np.vstack([tdr.point(0, th_f), tdr.point(1, th_f)])
    U = _fib_dirs(6000)
    hT = (U @ Ef.T).max(1)
    size = np.linalg.norm(K2, axis=1).max()
    eps = 2e-4 * size

    def clamp(Kn):
        for _ in range(10):
            viol = (U @ Kn.T) - (hT - eps)[:, None]
            bad = [i for i in np.where(viol.max(0) > 0)[0]
                   if not ext[i]]
            if not bad:
                break
            for i in bad:
                j = viol[:, i].argmax()
                Kn[i] -= viol[j, i] * U[j]
        return Kn

    K2 = clamp(solve(np.zeros(3), q0, wdat))

    # ----- minimum intraknot clearance: distant strands must
    # keep min_d = 2 rt + clearance between centrelines
    # (surface-to-surface gap = clearance).  Rope-style
    # relaxation: full-strength repulsion of violating pairs
    # alternated with light local fairing, anchored by the
    # pinned cores; the balance solves afterwards hold the
    # separated shape with boosted data weights.
    gap_par = max(4, n // 12)
    idxn = np.arange(n)
    circ = np.abs(idxn[:, None] - idxn[None, :])
    circ = np.minimum(circ, n - circ)
    far = circ > gap_par
    min_d = (2 * rt + clearance) if clearance > 0 else 0.0

    def repel(Kn):
        """Each free point steps once toward resolving its
        worst violating pair (stable, no cumulative
        overshoot)."""
        Dm = np.linalg.norm(Kn[:, None, :] - Kn[None, :, :],
                            axis=2)
        Dv = np.where(far & (Dm < min_d), Dm, np.inf)
        dmin = Dv.min(1)
        viol = np.isfinite(dmin) & (~ext)
        if not viol.any():
            return False
        j_star = Dv.argmin(1)
        disp = np.zeros_like(Kn)
        for i in np.where(viol)[0]:
            j = j_star[i]
            d = max(dmin[i], 1e-9)
            u = (Kn[i] - Kn[j]) / d
            disp[i] = u * 0.5 * (min_d - d)
        Kn += disp
        return True

    def relax_clearance(Kn, iters, ref):
        for it in range(iters):
            moved = repel(Kn)
            avg = 0.5 * (np.roll(Kn, -1, 0)
                         + np.roll(Kn, 1, 0))
            Kn[~ext] += (0.25 * (avg[~ext] - Kn[~ext])
                         + 0.10 * (ref[~ext] - Kn[~ext]))
            if it % 10 == 9:
                Kn = clamp(Kn)
            if not moved and it > 20:
                break
        return clamp(Kn)

    w_bal = wdat
    if min_d > 0:
        ref0 = K2.copy()
        K2 = relax_clearance(K2, 80, ref0)
        q0 = K2.copy()          # hold the separated shape
        w_bal = np.where(ext, wdat, 2.0)

    # ----- balance: drive the (thick) COM to the rolling
    # centre; with thick_aware the fused-strand mass counts once
    Tb = np.zeros(3)
    ov = 0.0
    for _ in range(balance_iters):
        cm, ov = com_thick(K2, rt if thick_aware else 0.0)
        if np.linalg.norm(cm) < 1e-5 * size:
            break
        Tb = Tb - cm
        K2 = clamp(solve(Tb, q0, w_bal))
        if min_d > 0:
            K2 = relax_clearance(K2, 15, q0)

    # ----- stage 3: fair the free interior against a TRUE
    # arc-length bending energy.
    #
    # Stage 2's regularizer (w_lap * DTD) is a squared second
    # difference in INDEX space, and morton() samples uniformly in
    # the parameter rather than in arc length -- the outer lobes come
    # out several times sparser than the interior.  Since the
    # continuous stiffness such an operator represents scales as h^3,
    # the interior is smoothed roughly a hundred times more weakly
    # per unit length than the lobes: fairing is strongest exactly
    # where it is least needed.  The clearance stage then holds that
    # shape in place (q0 = K2, interior data weight 2.0), so raising
    # w_lap barely moves the result -- measured, 600 -> 5000 changes
    # the curvature variation by under 5%.
    #
    # Hence a separate closing pass, in arc length, on the geometry
    # that actually shipped: minimize the bending energy with the
    # contact cores held and only a weak pull toward the current
    # shape, then re-apply every invariant the earlier stages
    # enforce -- hull clamp, strand clearance, COM balance -- so the
    # rolling property cannot regress.  Measured at (3,2): curvature
    # variation 83.8 -> 20.9, which is the fairness of the un-morphed
    # Morton seed, with rho unchanged to five decimals.
    #
    # Too stiff a weight and the fairing fights the clearance
    # repulsion instead of the sampling, and then fairness, rho or
    # both blow up (the threshold falls as p rises).  So the pass is
    # speculative: run it, judge the RESULT, and keep it only if it is
    # better on every axis that matters -- otherwise halve the weight
    # and try again, and failing that ship the curve unchanged.
    #
    # The judgement has to be on the whole pass rather than round by
    # round.  Each round's COM correction is applied to the NEXT
    # round's solve, so a single round is measured before its own
    # balance has been applied and can look far worse than the
    # trajectory it belongs to: on (5,2) round 0 alone reads rho
    # 0.00046 -> 0.00145, while the three rounds together finish at
    # 0.00032, BETTER than where they started.  A per-round test
    # throws that away.
    if w_fair > 0.0 and fair_rounds > 0:
        # release the last `fair_erode` samples at each end of every
        # core: the contact ranges were computed with a +-0.05 rad
        # margin, so the shortened core still spans the true contact
        # arc, and letting curvature turn over across the junction
        # halves the second-order step there
        keepf = np.zeros(n, bool)
        for r in runs2:
            keepf[r[fair_erode:len(r) - fair_erode]
                  if 2 * fair_erode < len(r) else r] = True
        fixf = np.where(keepf)[0]

        def _rho_of(Kx):
            cx, _ = com_thick(Kx, 0)
            return rho_eval(Kx, cx, runs2)[0]

        def _gap_of(Kx):
            Dx = np.linalg.norm(Kx[:, None, :] - Kx[None, :, :],
                                axis=2)
            return Dx[far].min() - 2 * rt

        entry, entry_Tb = K2.copy(), Tb.copy()
        tv0, rho0, gap0 = (curvature_variation(entry), _rho_of(entry),
                           _gap_of(entry))
        gap_floor = min(gap0, min_d - 2 * rt) - 1e-9
        wf = float(w_fair)
        trace = []
        # Don't even try when the clearance constraint is already
        # BINDING.  Fairing pulls the interior in; if the strands are
        # already as close as they are allowed to be, the repulsion
        # just pushes back, the two fight to a standstill, and the
        # curve ends up more kinked than it started.  Measured over
        # (3,2)/(5,2)/(7,2)/(9,2) at a = 0.35/0.4/0.5/0.65, "entry gap
        # exceeds 1.15x the requested clearance" separates every case
        # that gained from every case that had to be rejected -- and
        # rejecting costs three dense passes, so it is worth not
        # starting.  The result guard below still backstops this.
        #
        # Three attempts spans the weight range that was ever useful
        # (about 1e-5 to 3e-4); below that the solve barely moves the
        # curve, the result converges back to the entry shape, and it
        # fails the improvement test anyway.
        took = False
        attempts = 3 if (min_d <= 0.0
                         or gap0 > 1.15 * (min_d - 2 * rt)) else 0
        for _attempt in range(attempts):
            Kc, Tc = entry.copy(), entry_Tb.copy()
            for _ in range(int(fair_rounds)):
                D2, Mv = _bend_operator(Kc)
                seg = np.linalg.norm(np.roll(Kc, -1, 0) - Kc, axis=1)
                w = (seg + np.roll(seg, 1)) / 2
                w = w / w.sum()
                w_b = 1e6
                Am = (wf * ((D2.T * Mv) @ D2)
                      + np.diag(np.full(n, fair_data))
                      + w_b * np.outer(w, w))
                Kn = Kc.copy()
                for dim in range(3):
                    Af = Am.copy()
                    bf = (fair_data * entry[:, dim]
                          + w_b * Tc[dim] * w)
                    for i in fixf:
                        Af[i, :] = 0
                        Af[i, i] = 1
                        bf[i] = Kc[i, dim]
                    Kn[:, dim] = np.linalg.solve(Af, bf)
                Kn = clamp(Kn)
                if min_d > 0:
                    # restore the separation the solve ate into.  Raw
                    # repulsion, NOT relax_clearance: that routine's
                    # Laplacian-plus-reference pull drags the curve
                    # back toward its pre-fairing shape and undoes
                    # most of the gain (measured, (3,2): 26 -> 47).
                    for _ in range(25):
                        if not repel(Kn):
                            break
                    Kn = clamp(Kn)
                Kc = Kn
                cm, _ovc = com_thick(Kc, rt if thick_aware else 0.0)
                Tc = Tc - cm
            tv1, rho1, gap1 = (curvature_variation(Kc), _rho_of(Kc),
                               _gap_of(Kc))
            trace.append((wf, tv1, rho1, gap1))
            if (tv1 <= 0.98 * tv0 and rho1 <= 1.05 * rho0 + 1e-5
                    and gap1 >= gap_floor):
                K2, Tb, took = Kc, Tc, True
                break
            wf *= 0.5
        info['w_fair'] = wf if took else 0.0
        info['fair_trace'] = trace
        # both at the OPTIMIZATION resolution, so they compare: the
        # curvature estimate kappa = turn / ds grows at a genuine
        # corner as the sampling is refined, so a value measured
        # after the resample below is not on the same scale as one
        # measured before it
        info['tv_kappa_raw'] = tv0
        info['tv_kappa_fair'] = curvature_variation(K2)
    # ----- even out the sampling for the swept tube.  Done HERE, not
    # in the operator, so every diagnostic below describes the curve
    # that actually ships; the contact runs ride along by arc length.
    #
    # Kept only when it makes the worst turn GENTLER.  On a fair curve
    # that is automatic -- twice the samples, half the turn per
    # segment ((3,2): 4.8 -> 2.4 degrees).  On one that the fairing
    # pass had to decline, the seed's parameter sampling is dense
    # exactly where the curve corners hardest, and spreading the
    # samples evenly hands those corners fewer vertices to turn
    # through: the geometry is the same but the faceting is not
    # improved, so leave such a curve at its own sampling.
    if resample and resample > 0:
        Kr, runs_r = resample_uniform(
            K2, max(16, int(round(resample * n))), runs2)
        if _max_turn(Kr) <= _max_turn(K2):
            K2, runs2 = Kr, runs_r
            nn = len(K2)
            ii = np.arange(nn)
            cc = np.abs(ii[:, None] - ii[None, :])
            far = np.minimum(cc, nn - cc) > max(4, nn // 12)

    info['tv_kappa'] = curvature_variation(K2)
    cmc, _ = com_thick(K2, 0)
    info['rho'] = rho_eval(K2, cmc, runs2)[0]
    cm, ov = com_thick(K2, rt)
    info['rho_thick'] = rho_eval(K2, cm, runs2, rt)[0]
    info['overlap'] = ov
    Df = np.linalg.norm(K2[:, None, :] - K2[None, :, :],
                        axis=2)
    info['gap'] = Df[far].min() - 2 * rt
    return K2, info


def tube_mesh(K, rt, sides=16):
    """Closed swept tube: parallel-transport frames with the
    seam holonomy untwisted, circular cross-section."""
    n = len(K)
    T = np.roll(K, -1, 0) - np.roll(K, 1, 0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.zeros_like(K)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(T[0] @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    N[0] = np.cross(T[0], ref)
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, n):
        v = N[i - 1] - (N[i - 1] @ T[i]) * T[i]
        N[i] = v / (np.linalg.norm(v) or 1.0)
    # seam holonomy
    v = N[n - 1] - (N[n - 1] @ T[0]) * T[0]
    v /= (np.linalg.norm(v) or 1.0)
    B0 = np.cross(T[0], N[0])
    ang = math.atan2(v @ B0, v @ N[0])
    verts = []
    for i in range(n):
        B = np.cross(T[i], N[i])
        corr = -ang * i / n
        ca, sa = math.cos(corr), math.sin(corr)
        Ni = ca * N[i] + sa * B
        Bi = np.cross(T[i], Ni)
        for k in range(sides):
            ph = 2 * math.pi * k / sides
            verts.append(K[i] + rt * (math.cos(ph) * Ni
                                      + math.sin(ph) * Bi))
    faces = []
    for i in range(n):
        i2 = (i + 1) % n
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((i * sides + k, i * sides + k2,
                          i2 * sides + k2, i2 * sides + k))
    return [tuple(v) for v in verts], faces


# ---------------------------------------------------------------- #
#  Blender layer                                                   #
# ---------------------------------------------------------------- #

if _IN_BLENDER:

    class MESH_OT_rolling_knot_add(bpy.types.Operator):
        """Smooth-rolling (p,2) knot: Morton's knot morphed onto
        a smooth-rolling two-disk roller so its centre of mass
        stays at constant height while rolling, optionally
        rebalanced for the actual tube thickness (after Brodeur,
        Vidulis, Dandy & Pauly, Bridges 2025)"""
        bl_idname = "mesh.rolling_knot_add"
        bl_label = "Rolling Knot"
        bl_options = {'REGISTER', 'UNDO'}

        p: IntProperty(
            name="Lobes p", default=3, min=3, max=9,
            description="Odd (p,2) torus knot parameter "
                        "(3 = trefoil; even values are rounded "
                        "down)")
        a: FloatProperty(
            name="Shape a", default=0.5, min=0.15, max=0.9,
            description="Morton shape parameter")
        mode: EnumProperty(
            name="Mode",
            description="How much to reshape the raw Morton knot",
            items=[('SMOOTH', "Smooth-Rolling",
                    "Morph onto the two-disk roller "
                    "(constant-height rolling)"),
                   ('STRETCHED', "Stretched",
                    "Only the optimal vertical stretch"),
                   ('MORTON', "Morton",
                    "The raw Morton knot")],
            default='SMOOTH')
        rt: FloatProperty(
            name="Tube Radius", default=0.05, min=0.005,
            max=0.3,
            description="Tube radius, in units of the "
                        "(unit-scale) curve")
        thick_aware: BoolProperty(
            name="Optimize For Thickness", default=True,
            description="Rebalance the interior so the THICK "
                        "solid's centre of mass (fused strands "
                        "weigh once) sits at the rolling "
                        "centre")
        clearance: FloatProperty(
            name="Min Gap", default=0.02, min=0.0, max=0.5,
            description="Minimum surface-to-surface gap "
                        "between strands of the thick knot "
                        "(0 = strands may touch and fuse)")
        smoothing: FloatProperty(
            name="Smoothness", default=1.0, min=0.0, max=4.0,
            description="How hard to fair the interior of the "
                        "curve: 0 leaves the raw optimized shape, "
                        "1 is the measured sweet spot, higher "
                        "trades knot-shape fidelity for a calmer "
                        "sweep (the flow backs off on its own if "
                        "a setting would cost rolling accuracy)")
        samples: IntProperty(
            name="Curve Samples", default=512, min=128,
            max=1024,
            description="Number of points along the centreline curve")
        sides: IntProperty(
            name="Tube Sides", default=16, min=6, max=48,
            description="Number of sides around the tube cross-section")
        smooth: BoolProperty(name="Smooth Shading",
                             default=True,
                             description="Shade the tube smoothly "
                                         "rather than faceted")
        scale: FloatProperty(name="Scale", default=1.0,
                             min=0.01, max=100.0,
                             description="Overall size of the knot")

        def execute(self, context):
            p = self.p - (1 - self.p % 2)   # force odd
            # the UI knob drives the stage-3 fairing weight (w_lap,
            # the stage-2 index-space one it used to drive, is inert
            # by the time the clearance stage has pinned the shape --
            # see the stage-3 comment in build_rolling_knot)
            K, info = build_rolling_knot(
                p, self.a, self.mode, self.rt,
                self.thick_aware, self.samples,
                clearance=self.clearance,
                w_fair=1e-4 * float(self.smoothing))
            verts, faces = tube_mesh(K, self.rt, self.sides)
            name = f"Rolling Knot ({p},2)"
            # fit (roughly) within a 2 x scale cube at origin
            lo = [min(v[k] for v in verts) for k in range(3)]
            hi = [max(v[k] for v in verts) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0
                       for k in range(3)) or 1.0
            s = self.scale / half
            verts = [tuple((v[k] - (lo[k] + hi[k]) / 2.0) * s
                           for k in range(3)) for v in verts]
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set(
                'use_smooth',
                [self.smooth] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            msg = (f"{name}: rho raw={info['rho_raw']:.4f}"
                   f" curve={info.get('rho', 0):.4f}"
                   f" thick={info.get('rho_thick', 0):.4f}"
                   f" overlap={100 * info.get('overlap', 0):.1f}%"
                   f" gap={info.get('gap', 0):.3f}")
            if 'tv_kappa_raw' in info:
                msg += (f" fairness={info['tv_kappa_raw']:.0f}"
                        f"->{info['tv_kappa_fair']:.0f}"
                        + ("" if info.get('w_fair') else " (declined)"))
            self.report({'INFO'}, msg)
            print("Rolling Knot:", msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('p', 'a', 'mode', 'rt', 'thick_aware',
                      'clearance', 'smoothing', 'samples',
                      'sides', 'smooth', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.rolling_knot_add",
                             icon='FORCE_VORTEX')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_rolling_knot_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_rolling_knot_add)


def _selftest():
    # the smooth-rolling TDR itself must have rho ~ 0
    tdr = TDR(np.array([1.0, -1.0, 0]), 0.5, 0.35)
    th = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    E = np.vstack([tdr.point(0, th), tdr.point(1, th)])
    eruns = [np.arange(1024), np.arange(1024, 2048)]
    r_tdr, nb = rho_eval(E, np.zeros(3), eruns)
    print(f"smooth TDR: rho={r_tdr:.6f} (band {nb})")
    assert r_tdr < 1e-3
    # no-overlap tube COM is EXACTLY the arc centroid
    K = morton(3, 0.5, 1.0, 256)
    c0, ov0 = com_thick(K, 0.01)
    seg = np.linalg.norm(np.roll(K, -1, 0) - K, axis=1)
    w = (seg + np.roll(seg, 1)) / 2
    c_arc = (K * w[:, None]).sum(0) / w.sum()
    assert ov0 == 0.0 and np.allclose(c0, c_arc)
    print("thin tube: zero overlap, exact arc centroid")
    # optimization: trefoil and (5,2)
    for p in (3, 5):
        K, info = build_rolling_knot(p, 0.5, 'SMOOTH',
                                     rt=0.06, n=512)
        # kink check: worst turning angle between
        # consecutive segments must stay gentle
        T = np.roll(K, -1, 0) - K
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        turn = np.degrees(np.arccos(np.clip(
            (T * np.roll(T, 1, 0)).sum(1), -1, 1)))
        print(f"p={p}: rho raw={info['rho_raw']:.4f} -> "
              f"curve={info['rho']:.5f} "
              f"thick={info['rho_thick']:.5f} "
              f"overlap={100 * info['overlap']:.2f}% "
              f"max turn={turn.max():.1f}deg")
        assert info['rho_raw'] > 0.05
        assert info['rho'] < 0.01
        assert info['rho_thick'] < 0.02
        assert turn.max() < 12.0, turn.max()
    # minimum clearance: fat tube with a requested gap
    K, info = build_rolling_knot(3, 0.5, 'SMOOTH',
                                 rt=0.09, n=512,
                                 clearance=0.05)
    print(f"clearance run: gap={info['gap']:.4f} "
          f"(want >= 0.04) rho={info['rho']:.5f} "
          f"thick={info['rho_thick']:.5f}")
    assert info['gap'] >= 0.04
    assert info['rho'] < 0.01 and info['rho_thick'] < 0.01
    # tube mesh is closed
    verts, faces = tube_mesh(K, 0.05, sides=8)
    cnt = {}
    for f in faces:
        for i in range(len(f)):
            e = frozenset((f[i], f[(i + 1) % len(f)]))
            cnt[e] = cnt.get(e, 0) + 1
    assert all(c == 2 for c in cnt.values())
    finite = all(all(math.isfinite(c) for c in v)
                 for v in verts)
    assert finite
    print(f"tube: V={len(verts)} F={len(faces)} "
          f"watertight, finite")

    # the spline resample must reproduce the curve, not just
    # subdivide it: a circle stays a circle, and the sampling comes
    # out uniform in arc length with the contact runs carried across
    tt = np.linspace(0, 2 * np.pi, 97, endpoint=False)
    circ = np.stack([np.cos(tt), np.sin(tt), 0 * tt], 1)
    C2, cruns = resample_uniform(circ, 400, [np.arange(0, 20),
                                             np.arange(50, 70)])
    rad = np.linalg.norm(C2, axis=1)
    hh = np.linalg.norm(np.roll(C2, -1, 0) - C2, axis=1)
    assert abs(rad.mean() - 1.0) < 1e-4 and rad.std() < 1e-5
    assert hh.max() / hh.min() < 1.001, hh.max() / hh.min()
    assert len(cruns) == 2 and all(len(r) > 50 for r in cruns)
    print(f"resample: circle radius {rad.mean():.6f} cv "
          f"{rad.std():.1e}, spacing ratio {hh.max() / hh.min():.5f}")

    # the fairing pass may never ship a curve that is less fair, rolls
    # worse, or has closer strands than the one it was handed -- it
    # measures all three and declines the pass otherwise.  Checked on
    # a case that accepts (3,2) and one that historically declined
    # (9,2), so both branches are exercised.
    for p, a in ((3, 0.5), (9, 0.5)):
        K, info = build_rolling_knot(p, a, 'SMOOTH', rt=0.05, n=384,
                                     clearance=0.02)
        tv0, tv1 = info['tv_kappa_raw'], info['tv_kappa_fair']
        took = bool(info['w_fair'])
        print(f"fairing p={p} a={a}: TV {tv0:.1f} -> {tv1:.1f} "
              f"({'taken' if took else 'declined'}), "
              f"rho={info['rho']:.5f} gap={info['gap']:+.4f}")
        # declining must leave the curve exactly as it was, and
        # taking the pass must be a real improvement -- never a wash
        # dressed up as one
        assert tv1 == tv0 if not took else tv1 <= 0.98 * tv0
        assert info['rho'] < 0.01 and info['gap'] > 0.0
    print("rolling knot standalone tests passed")
