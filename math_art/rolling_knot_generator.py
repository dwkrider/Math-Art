
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


def build_rolling_knot(p=3, a=0.5, mode='SMOOTH', rt=0.05,
                       thick_aware=True, n=512, w_lap=200.0,
                       balance_iters=4):
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
    wdat = np.ones(n)
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
    A0 = np.diag(wdat) + w_lap * (D.T @ D)
    fix = np.where(ext)[0]
    q0 = K2.copy()

    def solve(target):
        Kn = K2.copy()
        seg = np.linalg.norm(np.roll(Kn, -1, 0) - Kn, axis=1)
        w = (seg + np.roll(seg, 1)) / 2
        w = w / w.sum()
        w_b = 1e6
        Am = A0 + w_b * np.outer(w, w)
        for dim in range(3):
            Af = Am.copy()
            bf = wdat * q0[:, dim] + w_b * target[dim] * w
            for i in fix:
                Af[i, :] = 0
                Af[i, i] = 1
                bf[i] = q0[i, dim]
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

    K2 = clamp(solve(np.zeros(3)))
    # ----- balance: drive the (thick) COM to the rolling
    # centre; with thick_aware the fused-strand mass counts once
    Tb = np.zeros(3)
    ov = 0.0
    for _ in range(balance_iters):
        cm, ov = com_thick(K2, rt if thick_aware else 0.0)
        if np.linalg.norm(cm) < 1e-5 * size:
            break
        Tb = Tb - cm
        K2 = clamp(solve(Tb))
    cmc, _ = com_thick(K2, 0)
    info['rho'] = rho_eval(K2, cmc, runs2)[0]
    cm, ov = com_thick(K2, rt)
    info['rho_thick'] = rho_eval(K2, cm, runs2, rt)[0]
    info['overlap'] = ov
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
        samples: IntProperty(
            name="Curve Samples", default=512, min=128,
            max=1024)
        sides: IntProperty(
            name="Tube Sides", default=16, min=6, max=48)
        smooth: BoolProperty(name="Smooth Shading",
                             default=True)
        scale: FloatProperty(name="Scale", default=1.0,
                             min=0.01, max=100.0)

        def execute(self, context):
            p = self.p - (1 - self.p % 2)   # force odd
            K, info = build_rolling_knot(
                p, self.a, self.mode, self.rt,
                self.thick_aware, self.samples)
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
                   f" overlap={100 * info.get('overlap', 0):.1f}%")
            self.report({'INFO'}, msg)
            print("Rolling Knot:", msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('p', 'a', 'mode', 'rt', 'thick_aware',
                      'samples', 'sides', 'smooth', 'scale'):
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


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
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
        print("rolling knot standalone tests passed")
