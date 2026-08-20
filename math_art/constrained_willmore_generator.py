
# Constrained Willmore Hopf tori -- Heller's equivariant family in closed
# form (Weierstrass elliptic functions), meshed through the Hopf fibration.
#
# A Hopf torus (Pinkall 1985) is the preimage under the Hopf fibration of a
# closed curve gamma on S^2; it is constrained Willmore -- a critical point
# of the bending energy W = int (H^2 + 1) dA under conformal variations --
# exactly when gamma is a CONSTRAINED ELASTIC curve: a critical point of
# int kappa^2 ds with prescribed length AND enclosed area,
#
#     kappa'' + kappa^3/2 + (mu + G) kappa + lambda = 0 ,           (EL)
#
# with multipliers mu (length) and lambda (area) on the sphere of curvature
# G.  Heller (2013) parametrises ALL of these curves in closed form.  Her
# Theorem 2 writes a lift of the curve directly in Weierstrass sigma/zeta
# functions on a lattice with real invariants g2, g3:
#
#     ghat1 = sigma(x + x0 - rho)/sigma(x + x0) * exp( zeta(rho)(x + x0))
#     ghat2 = sigma(x + x0 + rho)/sigma(x + x0) * exp(-zeta(rho)(x + x0))
#
# and the S^2 curve is [ghat1 : ghat2] in CP^1 = C u {inf} = S^2.  The pair
# is already the horizontal Hopf lift, so the torus needs no integration at
# all: everything below is evaluation of theta series.
#
# The pieces, and where they come from:
#
#  * LATTICE.  Wavelike curves (the only ones that close on S^2 with
#    lambda = 0 and mu + G > 0, Heller Cor. 3) live on the RHOMBIC lattice
#    (D = g2^3 - 27 g3^2 < 0): periods 2 and 1 + 2it, shape parameter t.
#    Heller's omega1 = 1 (real half period), omega3 = 2it (imaginary
#    half-lattice point; omega3 = omega1 mod Gamma exactly as she notes for
#    D < 0).
#
#  * SPECTRAL POINT.  rho is purely imaginary, and E = P(rho) < e3 = P(w3)
#    sweeps (-inf, e3).  The sphere curvature is G = 4(e3 - E) > 0
#    (Theorem 3), the length multiplier is mu + G = 6 e3, so the whole
#    shape story is: pick the lattice, pick rho.
#
#  * CLOSURE (Prop. 4).  The curve closes after n periods of P with
#    winding data m when g(rho) = eta1 rho - zeta(rho) omega1 = m pi i/(2n),
#    gcd(m, n) = 1.  Im g falls from +inf (rho -> 0) to a minimum and rises
#    back to pi at omega3, so a target BELOW pi (m < 2n, the branch that
#    carries every Willmore and free-elastic curve) has TWO roots -- the
#    closing function must be scanned, not bisected blindly.  The measured
#    geometric winding is 2n - m on this branch, so the operator exposes
#    w = 2n - m ("winding") and n ("lobes"); m > 2n gives exotic
#    multi-wrapped curves (winding m), exposed as "high wrap".
#
#  * THE SPACE-FORM NORMALISATION (Heller Sect. 3.4) -- the subtle step.
#    [ghat1 : ghat2] is only Moebius-equivalent to the arclength elastic
#    curve; the leftover gauge is z -> r z, r real.  Requiring arclength,
#    |gamma'(0)|^2_G = 1, turns out to mean: r is THE constant making
#        |ghat2|^2 + r^2 |ghat1|^2  constant along the curve
#    (then the lift (r ghat1, ghat2) has constant norm and projects to a
#    constant-speed curve).  r^2 comes out of a 2x2 least-squares fit and
#    the residual flatness is ~1e-15; skipping this step leaves a curve
#    whose speed wobbles by up to 14% and whose curvature is ~4% off --
#    close enough to look right and wrong enough to fail every gate.
#
#  * CURVATURE (Lemma 2).  kappa(x) = 4 Im zeta(x + x0) + C with the real
#    constant C fixed by Re P + kappa^2/8 + b = 0; the fitted b gives an
#    independent value of G through G = 8b - 4E (Lemma 1 vs Lemma 2), which
#    agrees with the measured speed^2 to machine precision.  x0 must be
#    PURELY IMAGINARY (a real x0 collapses the curve onto a great circle);
#    x0 = omega3/2 gives lambda = 0 (elastic curves: b = e3/2 exactly), and
#    sliding x0 along i(0, |omega3|) sweeps Heller's isospectral family of
#    genuinely constrained (lambda != 0) elastic curves -- same closure,
#    same lattice, different enclosed-area multiplier.
#
#  * SPECIAL POINTS on the shape axis (checked in the self-test):
#      E = -2 e3   <=>  mu = -G/2  <=>  the Hopf torus is WILLMORE
#                  (Pinkall's functional int (kappa^2+1) ds), and the curve
#                  reproduces the Langer-Singer elastica of the existing
#                  Hopf generator: winding/lobes w/n in (0, 2 - sqrt 2);
#      E = -e3/2   <=>  mu = 0     <=>  FREE elastica, w/n in (1/2, 1/sqrt2).
#
#  * ENERGY (Theorem 5).  W = (16 n eta1 - 8 n omega1 E) pi / sqrt(G),
#    reproduced by the meshed torus and by the analytic curvature integral.
#
# References:
# - Lynn Heller, "Constrained Willmore tori and elastic curves in
#   2-dimensional space forms", Comm. Anal. Geom. 22 (2014), no. 2;
#   arXiv:1303.1445 -- Theorem 2 (the sigma/zeta parametrisation), Prop. 4
#   (closure), Theorem 3 (S^2 curves), Sect. 3.4 (space-form gauge),
#   Theorem 5 (energy and conformal type).
# - Lynn Heller, "Equivariant constrained Willmore tori in the 3-sphere",
#   Math. Z. 278 (2014); arXiv:1211.4137 -- the equivariant classification
#   this family realises.
# - Christoph Bohle, "Constrained Willmore tori in the 4-sphere", J. Diff.
#   Geom. 86 (2010), 71-131 -- constrained Willmore tori are of finite
#   spectral genus (the general theory behind the spectral parameter E).
# - C. Bohle, G. P. Peters, U. Pinkall, "Constrained Willmore surfaces",
#   Calc. Var. PDE 32 (2008), 263-277 -- the Euler-Lagrange equation (EL).
# - Ulrich Pinkall, "Hopf tori in S^3", Invent. Math. 81 (1985), 379-386 --
#   Hopf tori, W = pi int (kappa^2 + 1) ds, all conformal classes.
# - Joel Langer, David A. Singer, "The total squared curvature of closed
#   curves", J. Diff. Geom. 20 (1984), 1-22 -- the elastica ODE and the
#   free/Willmore closure windows the self-test reduces to.
# - NIST DLMF ch. 23 (Weierstrass functions): the sigma/zeta/P quasi-
#   periodicity used for argument reduction.

# This module is a LIBRARY, not a generator: it has no operator of its
# own.  Its curves are exposed as presets on `mesh.hopf_torus_add`
# (math_art/hopf_fibration_generator.py), because a constrained
# Willmore torus IS a Hopf torus -- over a constrained elastic curve
# rather than a free one.  Keeping the operator there keeps one menu
# entry for one construction; keeping the mathematics here keeps a
# large, self-contained numeric core out of the Hopf module.

_UNUSED_bl_info = {
    "name": "Constrained Willmore Torus",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Heller's equivariant constrained Willmore Hopf tori: "
                   "closed (constrained) elastic curves on S^2 in "
                   "Weierstrass closed form, lifted through the Hopf "
                   "fibration",
    "category": "Add Mesh",
}

import math
from math import gcd, pi

import numpy as np

try:
    from .minsurf.elliptic import _Lattice
    from .hopf_fibration_generator import (build_hopf_torus,
                                           _clear_of_poles,
                                           _geodesic_curvature,
                                           willmore_energy)
except ImportError:                       # flat import outside the package
    from minsurf.elliptic import _Lattice                  # type: ignore
    from hopf_fibration_generator import (build_hopf_torus,   # type: ignore
                                          _clear_of_poles,
                                          _geodesic_curvature,
                                          willmore_energy)

TAU = 2.0 * pi
_WILLMORE_MAX = 2.0 - math.sqrt(2.0)      # sup w/n for Willmore closure
_FREE_WINDOW = (0.5, 1.0 / math.sqrt(2.0))


# --------------------------------------------------------------------------
# Rhombic Weierstrass lattice with argument reduction
# --------------------------------------------------------------------------

class _Rhombic:
    """Weierstrass P, P', zeta, sigma on the rhombic (wavelike, D < 0)
    lattice with periods 2 and 1 + 2it, evaluated at arbitrary complex
    arguments by reduction to the fundamental cell.  P and zeta reduce
    with the standard quasi-period constants; sigma carries the
    exponential multiplier sigma(z + Om) = eps sigma(z) e^{H (z + Om/2)}
    (DLMF 23.2.12-23.2.17)."""

    def __init__(self, t):
        self.t = float(t)
        self.tau = 0.5 + 1j * self.t
        self.p1 = 2.0                      # real period
        self.p2 = 1.0 + 2j * self.t        # second period
        self.w3 = 2j * self.t              # imaginary half-lattice point
        self.L = _Lattice(1.0, self.tau)
        self.eta1 = float(np.real(self.L.eta1))   # zeta(1); real
        # zeta(tau) by Legendre:  eta1 * tau - eta2 * 1 = pi i / 2
        self.eta2 = self.eta1 * self.tau - 0.5j * pi
        self.d1 = 2.0 * self.eta1
        self.d2 = 2.0 * self.eta2

    def _reduce(self, z):
        z = np.asarray(z, dtype=complex)
        k = np.rint(z.imag / (2.0 * self.t))
        j = np.rint(0.5 * (z.real - k))
        return z - j * self.p1 - k * self.p2, j, k

    def wp(self, z):
        zr, _, _ = self._reduce(z)
        return self.L.wp(zr)

    def wp_prime(self, z):
        zr, _, _ = self._reduce(z)
        return self.L.wp_prime(zr)

    def zeta(self, z):
        zr, j, k = self._reduce(z)
        return self.L.zeta(zr) + j * self.d1 + k * self.d2

    def sigma(self, z):
        zr, j, k = self._reduce(z)
        Om = j * self.p1 + k * self.p2
        H = j * self.d1 + k * self.d2
        eps = np.where(((j + k + j * k).astype(np.int64) % 2) == 0,
                       1.0, -1.0)
        return eps * self.L.sigma(zr) * np.exp(H * (zr + 0.5 * Om))

    def invariants(self):
        """(g2, g3) from P'^2 = 4P^3 - g2 P - g3 at two generic points."""
        zs = np.array([0.31 + 0.17j, 0.52 + 0.41j])
        P = self.wp(zs)
        rhs = self.wp_prime(zs) ** 2 - 4.0 * P ** 3
        A = np.stack([-P, -np.ones_like(P)], axis=1)
        g2, g3 = np.linalg.solve(A, rhs)
        return g2, g3


# --------------------------------------------------------------------------
# Closure (Heller Prop. 4): scan-and-bisect the NON-monotone closing map
# --------------------------------------------------------------------------

def _closure_roots(lat, target, ngrid=1600):
    """All rho = iy, y in (0, |w3|), with Im[eta1 rho - zeta(rho)] = target.
    The closing function falls from +inf to a t-dependent minimum in
    (pi/2, pi) and rises back to pi at w3, so there are 0, 1 or 2 roots
    below pi and exactly 1 above.  Returns (roots ascending, min of Im g)."""
    ymax = abs(lat.w3)
    ys = np.linspace(ymax * 1e-6, ymax * (1.0 - 1e-9), ngrid)
    g = (lat.eta1 * 1j * ys - lat.zeta(1j * ys)).imag - target
    roots = []
    for c in np.nonzero(np.sign(g[1:]) != np.sign(g[:-1]))[0]:
        ylo, yhi = ys[c], ys[c + 1]
        glo = g[c]
        for _ in range(90):
            ym = 0.5 * (ylo + yhi)
            gm = (lat.eta1 * 1j * ym - lat.zeta(1j * ym)).imag - target
            if gm * glo > 0.0:
                ylo = ym
            else:
                yhi = ym
        roots.append(0.5 * (ylo + yhi))
    return roots, float(g.min() + target)


def _pick_root(roots, branch):
    return roots[-1] if branch == 'UPPER' else roots[0]


# --------------------------------------------------------------------------
# The curve family (Heller Thm. 2 + Sect. 3.4 gauge)
# --------------------------------------------------------------------------

def heller_curve(t, n, m, branch='UPPER', x0_frac=0.5, samples=512):
    """Closed constrained elastic curve on the UNIT sphere.

    t        rhombic lattice shape (Im tau);
    n        lobes = periods of P until closure;
    m        Heller's closure integer, gcd(m, n) = 1 (geometric winding
             2n - m for m < 2n, m itself for m > 2n);
    branch   'UPPER'/'LOWER': which of the two closure roots when m < 2n;
    x0_frac  phase x0 = i * x0_frac * |w3|; 0.5 <=> lambda = 0 (elastic);
    samples  points along the curve (excludes the endpoint).

    Returns (P, info): P an (samples, 3) array on S^2, info a dict with
    G, E, e3, mu1 = (mu+G)/G, lam (area multiplier, unit-sphere units),
    kappa (unit-sphere geodesic curvature at the samples), and the
    diagnostics the self-test gates on."""
    lat = _Rhombic(t)
    target = m * pi / (2.0 * n)
    roots, gmin = _closure_roots(lat, target)
    if not roots:
        raise ValueError(
            f"no closed curve: lattice t={t:.3f} reaches closing angles "
            f">= {gmin:.4f}, target m pi/2n = {target:.4f}; increase the "
            f"shape parameter")
    rho = 1j * _pick_root(roots, branch)
    E = float(lat.wp(rho).real)
    e3 = float(lat.wp(lat.w3).real)
    x0 = 1j * float(x0_frac) * abs(lat.w3)

    x = np.arange(samples) * (2.0 * n / samples)
    zx = x + x0
    zr = lat.zeta(rho)
    s0 = lat.sigma(zx)
    g1 = lat.sigma(zx - rho) / s0 * np.exp(zr * zx)
    g2 = lat.sigma(zx + rho) / s0 * np.exp(-zr * zx)

    # Wronskian det(ghat, ghat') -- constant for an exact solution pair
    lg1 = lat.zeta(zx - rho) - lat.zeta(zx) + zr
    lg2 = lat.zeta(zx + rho) - lat.zeta(zx) - zr
    Wr = g1 * g2 * (lg2 - lg1)
    wr_dev = float(np.abs(Wr - Wr.mean()).max() / np.abs(Wr.mean()))

    # Sect. 3.4 gauge z -> r z:  |g2|^2 + r^2 |g1|^2 = const
    p = np.abs(g1) ** 2
    q = np.abs(g2) ** 2
    vp = p - p.mean()
    r2 = -float(np.dot(q - q.mean(), vp) / np.dot(vp, vp))
    if r2 <= 0.0:
        raise ValueError(
            "not a spherical curve here (G <= 0): this phase/shape puts "
            "the constrained elastic curve in R^2 or H^2 (Heller Rem. 9); "
            "move the area phase toward 0.5 or switch branch")
    cnorm = float((q + r2 * p).mean())
    flat = float(np.std(q + r2 * p) / cnorm)

    # unit-sphere curve (stereographic chart z = r g1/g2)
    w = math.sqrt(r2) * g1 / g2
    den = 1.0 + np.abs(w) ** 2
    P = np.stack([2.0 * w.real / den, 2.0 * w.imag / den,
                  (np.abs(w) ** 2 - 1.0) / den], axis=1)

    # analytic speed |dP/dx| = 2 r |Wr| / (q + r^2 p): constant = sqrt(G)
    speed = 2.0 * math.sqrt(r2) * np.abs(Wr) / (q + r2 * p)
    G_speed = float(speed.mean()) ** 2

    # Lemma 2: kappa = 4 Im zeta + C, Re P + kappa^2/8 + b = 0
    v = lat.zeta(zx).imag
    u = lat.wp(zx).real + 2.0 * v ** 2
    vv = v - v.mean()
    C = -float(np.dot(u - u.mean(), vv) / np.dot(vv, vv))
    b = -(float((u + C * v).mean()) + C * C / 8.0)
    G_b = 8.0 * b - 4.0 * E
    G_dev = abs(G_speed - G_b) / abs(G_b)
    sG = math.sqrt(G_b)
    kappa_u = (4.0 * v + C) / sG          # geodesic curvature, unit sphere

    # Euler-Lagrange fit on the unit sphere:
    #   kappa'' + kappa^3/2 + mu1 kappa + lam = 0  (mu1 = (mu+G)/G)
    kpp = -4.0 * lat.wp_prime(zx).imag / (sG ** 3)
    rhs = -(kpp + 0.5 * kappa_u ** 3)
    A = np.stack([kappa_u, np.ones_like(kappa_u)], axis=1)
    (mu1, lam), *_ = np.linalg.lstsq(A, rhs, rcond=None)
    el_res = float(np.abs(kpp + 0.5 * kappa_u ** 3 + mu1 * kappa_u + lam)
                   .max() / max(np.abs(kpp).max(), 1.0))

    # closure of the point curve
    xe = np.array([0.0, 2.0 * n]) + x0
    g1e = lat.sigma(xe - rho) / lat.sigma(xe) * np.exp(zr * xe)
    g2e = lat.sigma(xe + rho) / lat.sigma(xe) * np.exp(-zr * xe)
    we = math.sqrt(r2) * g1e / g2e
    dne = 1.0 + np.abs(we) ** 2
    Pe = np.stack([2.0 * we.real / dne, 2.0 * we.imag / dne,
                   (np.abs(we) ** 2 - 1.0) / dne], axis=1)
    closure = float(np.linalg.norm(Pe[1] - Pe[0]))

    # Theorem 5 energy (Hopf):  W = (16 n eta1 - 8 n E) pi / sqrt(G)
    W_closed = (16.0 * n * lat.eta1 - 8.0 * n * E) * pi / sG
    # analytic curvature integral: W = pi int (1 + kappa^2) ds, ds = sG dx
    dx = 2.0 * n / samples
    W_analytic = pi * sG * dx * float(np.sum(1.0 + kappa_u ** 2))

    info = dict(lat=lat, rho=rho, E=E, e3=e3, G=G_b, mu1=float(mu1),
                lam=float(lam), kappa=kappa_u, length=2.0 * n * sG,
                W_closed=W_closed, W_analytic=W_analytic,
                wronskian_dev=wr_dev, norm_flat=flat, G_dev=G_dev,
                el_res=el_res, closure=closure, roots=roots)
    return P, info


# --------------------------------------------------------------------------
# Shape solves: the Willmore and free-elastica points of the family
# --------------------------------------------------------------------------

_SHAPE_CACHE = {}


def solve_shape(n, m, condition, branch='UPPER',
                tlo=0.34, thi=3.2, nscan=48):
    """Lattice shape t at which the closed (n, m) curve satisfies a shape
    condition along the given closure branch:

      'WILLMORE':  E = -2 e3   (mu = -G/2; the Hopf torus is Willmore)
      'FREE':      E = -e3/2   (mu = 0; free elastica)

    Scans t (the closure roots exist only above a t-threshold when the
    target angle is below pi, and the condition function is not monotone),
    then bisects the bracketing interval."""
    key = (n, m, condition, branch)
    if key in _SHAPE_CACHE:
        return _SHAPE_CACHE[key]
    target = m * pi / (2.0 * n)
    fac = 2.0 if condition == 'WILLMORE' else 0.5

    def val(t):
        lat = _Rhombic(t)
        roots, _ = _closure_roots(lat, target)
        if not roots:
            return None
        y = _pick_root(roots, branch)
        return float(lat.wp(1j * y).real + fac * lat.wp(lat.w3).real)

    def bisect(a_, b_, va):
        for _ in range(70):
            mid = 0.5 * (a_ + b_)
            vm = val(mid)
            if vm is None or np.sign(vm) == np.sign(va):
                a_ = mid
            else:
                b_ = mid
        return 0.5 * (a_ + b_)

    ts = np.linspace(tlo, thi, nscan)
    vals = [val(float(tt)) for tt in ts]
    for i in range(nscan - 1):
        va, vb = vals[i], vals[i + 1]
        if vb is None:
            continue
        if va is None:
            # existence edge: the closure roots appear somewhere inside
            # (ts[i], ts[i+1]) -- the sign change may hide in the sliver
            # just above the edge (that is exactly where the Willmore
            # point of (w, n) = (1, 3) lives), so locate the edge first.
            a_, b_ = float(ts[i]), float(ts[i + 1])
            for _ in range(60):
                mid = 0.5 * (a_ + b_)
                if val(mid) is None:
                    a_ = mid
                else:
                    b_ = mid
            va = val(b_)
            if va is None or np.sign(va) == np.sign(vb):
                continue
            tsol = bisect(b_, float(ts[i + 1]), va)
            _SHAPE_CACHE[key] = tsol
            return tsol
        if np.sign(va) == np.sign(vb):
            continue
        tsol = bisect(float(ts[i]), float(ts[i + 1]), va)
        _SHAPE_CACHE[key] = tsol
        return tsol
    raise ValueError(
        f"no {condition} curve with lobes {n} and this winding: "
        f"Willmore needs w/n < 2 - sqrt(2) = 0.5858, free elastica "
        f"needs 1/2 < w/n < 1/sqrt(2)")


def resolve_params(lobes, winding, family, shape, branch, phase,
                   high_wrap=False):
    """Map operator-facing parameters to (t, m, branch, x0_frac)."""
    n, w = int(lobes), int(winding)
    if gcd(w, n) != 1:
        raise ValueError(f"winding {w} and lobes {n} must be coprime")
    if not high_wrap and not (1 <= w < n):
        raise ValueError(f"winding must be in 1..{n - 1}")
    m = (2 * n + w) if high_wrap else (2 * n - w)
    x0f = 0.5
    if family == 'WILLMORE':
        if high_wrap or not (w / n < _WILLMORE_MAX):
            raise ValueError(
                f"Willmore tori need winding/lobes < 2 - sqrt(2) = "
                f"{_WILLMORE_MAX:.4f}; got {w}/{n}")
        t = solve_shape(n, m, 'WILLMORE', 'UPPER')
        branch = 'UPPER'
    elif family == 'FREE':
        if high_wrap or not (_FREE_WINDOW[0] < w / n < _FREE_WINDOW[1]):
            raise ValueError(
                f"free elasticae need 1/2 < winding/lobes < 1/sqrt(2) = "
                f"{_FREE_WINDOW[1]:.4f}; got {w}/{n}")
        t = solve_shape(n, m, 'FREE', 'UPPER')
        branch = 'UPPER'
    else:
        t = float(shape)
        if family == 'CONSTRAINED':
            x0f = min(max(float(phase), 0.02), 0.98)
        # walk t up until the closure target is reachable
        for _ in range(40):
            lat = _Rhombic(t)
            roots, _ = _closure_roots(lat, m * pi / (2.0 * n))
            if roots:
                break
            t *= 1.12
        else:
            raise ValueError("closure not reachable; raise the shape "
                             "parameter")
    return t, m, branch, x0f


# --------------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------------

def _selftest():
    ok_all = True

    def check(label, value, tol, fmt="{:.2e}"):
        nonlocal ok_all
        good = value < tol
        ok_all = ok_all and good
        print(f"cw: {label} = " + fmt.format(value) +
              f" (tol {tol:g}) {'OK' if good else 'BAD'}")
        return good

    # 1) lattice engine identities on two shapes
    for t in (0.6, 1.1):
        lat = _Rhombic(t)
        z = np.array([0.23 + 0.11j, -0.4 + 0.35j, 1.7 - 0.6j, 3.1 + 1.9j])
        h = 1e-6
        fd = (lat.sigma(z + h) - lat.sigma(z - h)) / (2 * h * lat.sigma(z))
        check(f"t={t} sigma'/sigma - zeta", float(np.abs(fd - lat.zeta(z))
                                                  .max()), 1e-8)
        q2 = np.abs(lat.sigma(z + lat.p2) + lat.sigma(z)
                    * np.exp(lat.d2 * (z + lat.p2 / 2)))
        check(f"t={t} sigma quasi-periodicity",
              float((q2 / np.abs(lat.sigma(z + lat.p2))).max()), 1e-10)
        g2, g3 = lat.invariants()
        zz = np.array([0.2 + 0.3j, 1.1 + 0.7j])
        ode = np.abs(lat.wp_prime(zz) ** 2
                     - (4 * lat.wp(zz) ** 3 - g2 * lat.wp(zz) - g3))
        check(f"t={t} P ODE residual",
              float((ode / np.abs(lat.wp(zz)) ** 3).max()), 1e-10)
        check(f"t={t} D<0 (wavelike)",
              float(np.sign((g2 ** 3 - 27 * g3 ** 2).real)), 0.0,
              fmt="{:+.0f}")

    # 2) elastic curve (lambda = 0): n=3, w=1 -> m=5, generic shape
    P, info = heller_curve(0.7, 3, 5, 'UPPER', 0.5, samples=2048)
    check("elastic Wronskian dev", info['wronskian_dev'], 1e-12)
    check("elastic 3.4-gauge flatness", info['norm_flat'], 1e-12)
    check("elastic G(speed) vs G(8b-4E)", info['G_dev'], 1e-12)
    check("elastic EL residual", info['el_res'], 1e-12)
    check("elastic |lambda|", abs(info['lam']), 1e-12)
    check("elastic closure", info['closure'], 1e-10)
    check("elastic mu1 vs 6 e3/G",
          abs(info['mu1'] - 6.0 * info['e3'] / info['G']), 1e-10)
    kappa = info['kappa']
    lob = int(np.sum((kappa > np.roll(kappa, 1))
                     & (kappa > np.roll(kappa, -1))))
    check("elastic lobe count |n-3|", abs(lob - 3), 0.5, fmt="{:.0f}")
    km, _, _ = _geodesic_curvature(P)
    kdev = min(float(np.abs(km - kappa).max()),
               float(np.abs(km + kappa).max())) / float(np.abs(kappa).max())
    check("elastic measured kappa vs Lemma 2", kdev, 1e-4)

    # 3) constrained elastic (lambda != 0), same closure, off-centre phase
    _, ci = heller_curve(0.7, 3, 5, 'LOWER', 0.35, samples=2048)
    check("constrained EL residual", ci['el_res'], 1e-12)
    check("constrained closure", ci['closure'], 1e-10)
    good = abs(ci['lam']) > 1e-3
    ok_all = ok_all and good
    print(f"cw: constrained |lambda| = {abs(ci['lam']):.4f} (> 1e-3) "
          f"{'OK' if good else 'BAD'}")

    # 4) WILLMORE point reduces to the Langer-Singer elastica of the
    #    existing Hopf generator (independent Jacobi-function route)
    tstar = solve_shape(3, 5, 'WILLMORE', 'UPPER')
    Pw, wi = heller_curve(tstar, 3, 5, 'UPPER', 0.5, samples=4096)
    check("willmore |E + 2 e3|", abs(wi['E'] + 2 * wi['e3']), 1e-6)
    check("willmore mu1 - 1/2", abs(wi['mu1'] - 0.5), 1e-6)
    check("willmore W closed form vs analytic",
          abs(wi['W_closed'] - wi['W_analytic']) / wi['W_closed'], 1e-9)
    try:
        from .hopf_fibration_generator import spherical_elastica
    except ImportError:
        from hopf_fibration_generator import spherical_elastica
    Q = spherical_elastica(1, 3, 4096)
    Wq, Lq = willmore_energy(Q)
    Wm, Lm = willmore_energy(Pw)
    kq, _, _ = _geodesic_curvature(Q)
    kw = wi['kappa']
    check("willmore vs elastica(1,3): W", abs(Wm - Wq) / Wq, 1e-5)
    check("willmore vs elastica(1,3): L", abs(Lm - Lq) / Lq, 1e-5)
    check("willmore vs elastica(1,3): kappa_max",
          abs(float(np.abs(kw).max()) - float(np.abs(kq).max()))
          / float(np.abs(kq).max()), 1e-5)

    # 5) FREE point: mu = 0 exactly (w=2, n=3 sits in the LS free window)
    tfree = solve_shape(3, 4, 'FREE', 'UPPER')
    _, fi = heller_curve(tfree, 3, 4, 'UPPER', 0.5, samples=2048)
    check("free |mu1 - 1|", abs(fi['mu1'] - 1.0), 1e-6)
    check("free |lambda|", abs(fi['lam']), 1e-8)

    # 6) constrained Willmore criticality on the MESH: the discrete
    #    Willmore gradient's normal density falls under refinement for the
    #    Willmore torus and plateaus for an elastic-but-not-Willmore one
    #    (its continuum gradient is the nonzero constraint combination).
    try:
        from .solver.willmore import willmore_gradient, vertex_area_data
    except ImportError:
        from solver.willmore import willmore_gradient, vertex_area_data

    def grad_density(Pc, mpsi):
        verts, faces, _ = build_hopf_torus(Pc, mpsi)
        V = np.asarray(verts, float)
        T = []
        for f in faces:
            a_, b_, c_, d_ = f
            T.append([a_, b_, c_])
            T.append([a_, c_, d_])
        T = np.asarray(T, dtype=np.int64)
        _E, grad = willmore_gradient(V, T)
        g, a, nvec = vertex_area_data(V, T)
        nl = np.maximum(np.linalg.norm(nvec, axis=1), 1e-300)
        gn = np.einsum('ij,ij->i', grad, nvec / nl[:, None]) / a
        A = a.sum()
        return math.sqrt(float(np.sum(a * gn * gn) / A)) * A ** 1.5

    Pw64, _ = heller_curve(tstar, 3, 5, 'UPPER', 0.5, samples=64)
    Pw128, _ = heller_curve(tstar, 3, 5, 'UPPER', 0.5, samples=128)
    fall_w = grad_density(Pw64, 32) / grad_density(Pw128, 64)
    Pc64, _ = heller_curve(0.75, 3, 5, 'UPPER', 0.5, samples=64)
    Pc128, _ = heller_curve(0.75, 3, 5, 'UPPER', 0.5, samples=128)
    fall_c = grad_density(Pc64, 32) / grad_density(Pc128, 64)
    good = fall_w > 3.0
    ok_all = ok_all and good
    print(f"cw: Willmore mesh gradient falloff x{fall_w:.2f} (> 3) "
          f"{'OK' if good else 'BAD'}")
    good = fall_c < 1.5
    ok_all = ok_all and good
    print(f"cw: non-Willmore control falloff x{fall_c:.2f} (< 1.5, "
          f"plateau) {'OK' if good else 'BAD'}")

    assert ok_all, "constrained_willmore_generator self-test FAILED"
    print("cw: all checks OK")
