
# Constrained Willmore tori -- Heller's equivariant families in closed
# form (Weierstrass elliptic functions).  Two constructions from the same
# paper share this module:
#
#   ROUTE A (Hopf tori):        constrained elastic curves on S^2, lifted
#                               through the Hopf fibration; exposed as the
#                               CONSTRAINED preset on mesh.hopf_torus_add.
#   ROUTE B (tori of revolution): elastic curves in the hyperbolic plane,
#                               rotated about an axis; exposed as the
#                               ELASTIC_TORUS mode on
#                               mesh.delaunay_surface_add (they are the
#                               CMC surfaces of revolution of the SPHERICAL
#                               space form, so they extend the Delaunay
#                               family there).
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
# ROUTE B -- constrained Willmore TORI OF REVOLUTION (Heller Sect. 3.5-6).
# Rotating a closed curve of the upper half plane about the x-axis gives a
# torus that is constrained Willmore exactly when the curve is ELASTIC
# (lambda = 0) in the upper half plane viewed as H^2 (Langer-Singer 1984).
# The same Theorem-2 formulas deliver these curves; what changes:
#
#  * LATTICE.  Closed elastic curves in H^2 are ORBITLIKE (D > 0), so the
#    lattice is RECTANGULAR: periods 2 and 2iT, omega1 = 1, omega3 = iT,
#    real roots e1 > e2 > e3 of P3.  The wavelike case D < 0 admits at
#    most one closed curve, the elastic figure-eight (Heller Prop. 5 /
#    Example 1), which yields NO Willmore torus; the rectangular lattice
#    excludes it structurally.
#
#  * CLOSURE (Theorem 4).  rho = omega1 + iy, y in (0, T); the closing
#    function g = eta1 rho - zeta(rho) omega1 is then PURELY IMAGINARY,
#    rising monotonically (measured, all shapes) from 0 at y -> 0 to
#    pi/2 at omega3 -- so unlike Route A every target m pi/(2n) with
#    0 < m < n, gcd(m, n) = 1, has exactly one root at every lattice
#    shape, and the n-lobed curve exists for all n > 1 (Heller Thm. 4).
#    E = P(rho) sweeps (e2, e1), so P3(E) < 0 ALWAYS on this branch:
#    by Proposition 6 every torus built here is CMC in S^3 (the H^3
#    branches of Prop. 6 need P3(E) > 0 or a wavelike curve, neither of
#    which closes here).
#
#  * MODEL (Sect. 3.4).  g imaginary <=> the monodromy of [ghat1:ghat2]
#    is a ROTATION about 0, so the curve lives in the POINCARE DISC with
#    0 the fixed point -- not the upper half plane (a real g would give a
#    dilation z -> rz, the half-plane's isometry).  The 3.4 gauge is the
#    disc analogue of Route A's: r^2 is THE constant making
#        |ghat2|^2 - r^2 |ghat1|^2   constant along the curve
#    (sign flipped against the S^2 case), and then w = r ghat1/ghat2
#    stays inside the unit disc and runs at constant Poincare speed
#    sqrt(-G), G = 4(e3 - E) = 8b - 4E < 0.  x0 = omega3/2 again gives
#    exactly lambda = 0 (b = e3/2 to machine precision), which is the
#    ONLY admissible phase here: an off-centre x0 makes the curve
#    constrained (lambda != 0) elastic, and its revolution torus is NOT
#    constrained Willmore -- so Route B does not expose the phase.
#
#  * SURFACE.  Cayley-map the disc to the upper half plane (0 -> i) and
#    revolve about the boundary axis: p = (u, v cos phi, v sin phi).
#    The disc rotation freedom (an S^3 isometry -- it is the elliptic
#    rotation about i of the profile plane) is spent balancing the
#    profile, minimising max |z| so the stereographic picture is compact.
#    The n-lobed symmetry pins the curve's elliptic centre to the disc
#    centre, whose Cayley image i is the profile-plane trace of the
#    great circle fixed by the revolution -- which is why this placement,
#    and no rescaling of it, is the one where the torus is CMC in the
#    UNIT 3-sphere (checked to 1e-13: H_S3 = (1+|p|^2) H_R3 / 2 + nu . p
#    is constant along the profile).  The n = 2 family interpolates the
#    embedded two-lobed CMC tori between the doubled geodesic sphere
#    (W -> 8 pi as T -> 0) and the bifurcation homogeneous torus at
#    H = 1/sqrt(3) (W -> 4 pi^2 / sqrt(3) as T -> inf) -- both limits are
#    closed-form anchors the self-test pins.
#
#  * ENERGY (Theorem 5, revolution case).  As printed the paper gives
#    W = 8 n eta1 pi - 4 n omega1 P(omega3) pi, which is the bending
#    energy (pi/2) int kappa^2 ds read in CURVATURE-G units; the Willmore
#    energy of the actual surface -- int H^2 dA over the revolution torus
#    in R^3, which is what the mesh integrates -- carries the same
#    1/sqrt(-G) normalisation the paper writes explicitly in its Hopf
#    formula:
#        W(f) = (8 n eta1 - 4 n omega1 P(omega3)) pi / sqrt(-G) ,
#    verified here three independent ways (closed form, analytic
#    curvature integral, finite differences on the sampled profile).
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
# - Joel Langer, David A. Singer, "Curves in the hyperbolic plane and mean
#   curvature of tori in 3-space", Bull. London Math. Soc. 16 (1984),
#   531-534 -- a torus of revolution is Willmore-critical exactly when its
#   profile is elastic in H^2, and W = (pi/2) int kappa^2 ds (Route B).
# - Ulrich Pinkall, Ivan Sterling, "On the classification of constant mean
#   curvature tori", Ann. of Math. 130 (1989), 407-451 -- the CMC tori in
#   S^3 that Route B's Proposition-6 branch produces are the rotational
#   members of this classification.
# - NIST DLMF ch. 23 (Weierstrass functions): the sigma/zeta/P quasi-
#   periodicity used for argument reduction.

# This module is a LIBRARY, not a generator: it has no operator of its
# own.  Route A's curves are exposed as presets on `mesh.hopf_torus_add`
# (math_art/hopf_fibration_generator.py), because a constrained
# Willmore Hopf torus IS a Hopf torus -- over a constrained elastic
# curve rather than a free one.  Route B's tori are exposed as the
# ELASTIC_TORUS mode on `mesh.delaunay_surface_add`
# (math_art/delaunay_generator.py), because they are exactly the CMC
# surfaces of revolution of the spherical space form -- the members of
# the Delaunay family that close into compact tori, which the rolling
# conics of R^3 can never do.  One menu entry per construction; the
# large numeric core stays here.

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
# Route B: rectangular (orbitlike, D > 0) lattice with argument reduction
# --------------------------------------------------------------------------

class _Rectangular:
    """Weierstrass P, P', zeta, sigma on the rectangular lattice with
    periods 2 and 2iT (omega1 = 1 real, omega3 = iT imaginary), evaluated
    at arbitrary complex arguments by reduction to the fundamental cell.
    Real invariants with D = g2^3 - 27 g3^2 > 0: the ORBITLIKE case, the
    only one whose elastic curves close in H^2 (Heller Thm. 4 vs Prop. 5)."""

    def __init__(self, t):
        self.t = float(t)
        self.tau = 1j * self.t
        self.p1 = 2.0                      # real period
        self.p2 = 2j * self.t              # imaginary period
        self.w3 = 1j * self.t              # imaginary half period
        self.L = _Lattice(1.0, self.tau)
        self.eta1 = float(np.real(self.L.eta1))   # zeta(1); real
        # zeta(w3) by Legendre:  eta1 * w3 - eta3 * 1 = pi i / 2
        self.eta3 = self.eta1 * self.w3 - 0.5j * pi
        self.d1 = 2.0 * self.eta1
        self.d3 = 2.0 * self.eta3

    def _reduce(self, z):
        z = np.asarray(z, dtype=complex)
        k = np.rint(z.imag / (2.0 * self.t))
        j = np.rint(0.5 * z.real)
        return z - j * self.p1 - k * self.p2, j, k

    def wp(self, z):
        zr, _, _ = self._reduce(z)
        return self.L.wp(zr)

    def wp_prime(self, z):
        zr, _, _ = self._reduce(z)
        return self.L.wp_prime(zr)

    def zeta(self, z):
        zr, j, k = self._reduce(z)
        return self.L.zeta(zr) + j * self.d1 + k * self.d3

    def sigma(self, z):
        zr, j, k = self._reduce(z)
        Om = j * self.p1 + k * self.p2
        H = j * self.d1 + k * self.d3
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


def _h2_closure_roots(lat, target, ngrid=1200):
    """All rho = omega1 + iy, y in (0, T), with Im g(rho) = target, where
    g = eta1 rho - zeta(rho) omega1 (Heller Thm. 4; purely imaginary on
    this segment).  Im g rises from 0 (y -> 0) to pi/2 (y -> T); measured
    monotone at every lattice shape, but scanned rather than trusted."""
    T = lat.t
    ys = np.linspace(T * 1e-6, T * (1.0 - 1e-9), ngrid)
    rho = 1.0 + 1j * ys

    def gval(r):
        return (lat.eta1 * r - lat.zeta(r)).imag

    g = gval(rho) - target
    roots = []
    for c in np.nonzero(np.sign(g[1:]) != np.sign(g[:-1]))[0]:
        ylo, yhi = ys[c], ys[c + 1]
        glo = g[c]
        for _ in range(90):
            ym = 0.5 * (ylo + yhi)
            if (gval(1.0 + 1j * ym) - target) * glo > 0.0:
                ylo = ym
            else:
                yhi = ym
        roots.append(0.5 * (ylo + yhi))
    return roots


def _h2_eval(lat, rho, r, x0, x):
    """The disc curve w = r ghat1/ghat2 of Heller (3.5) and its first two
    x-derivatives, all in closed form (log-derivatives are zeta sums,
    their derivatives P sums)."""
    zx = np.asarray(x, dtype=float) + x0
    zr = lat.zeta(rho)
    s0 = lat.sigma(zx)
    g1 = lat.sigma(zx - rho) / s0 * np.exp(zr * zx)
    g2 = lat.sigma(zx + rho) / s0 * np.exp(-zr * zx)
    lg1 = lat.zeta(zx - rho) - lat.zeta(zx) + zr
    lg2 = lat.zeta(zx + rho) - lat.zeta(zx) - zr
    w = r * g1 / g2
    wp = w * (lg1 - lg2)
    dlg1 = -lat.wp(zx - rho) + lat.wp(zx)
    dlg2 = -lat.wp(zx + rho) + lat.wp(zx)
    wpp = wp * (lg1 - lg2) + w * (dlg1 - dlg2)
    return w, wp, wpp, (g1, g2, lg1, lg2)


def heller_h2_curve(t, n, m, samples=1024, branch=0):
    """Closed elastic curve in the hyperbolic plane (Poincare disc model).

    t        rectangular lattice shape (omega3 = it);
    n        lobes = periods of P until closure (n >= 2);
    m        winding number, 1 <= m < n, gcd(m, n) = 1;
    samples  points along the curve (endpoint excluded);
    branch   which closure root if several (measured: always one).

    Returns (w, info): w complex samples in the unit disc, info a dict
    with the lattice data, G < 0, E, e3, the analytic derivatives, and
    every diagnostic the self-test gates on.  The phase is pinned to
    x0 = omega3/2, the unique lambda = 0 (elastic) phase; off-centre
    phases give constrained elastic curves whose revolution tori are NOT
    constrained Willmore, so Route B has no phase freedom to expose."""
    n, m = int(n), int(m)
    if n < 2:
        raise ValueError("tori of revolution need lobes n >= 2 "
                         "(Heller Thm. 4)")
    if not (1 <= m < n) or gcd(m, n) != 1:
        raise ValueError(f"winding must be coprime to lobes and in "
                         f"1..{n - 1}; got {m}/{n}")
    lat = _Rectangular(t)
    g2i, g3i = lat.invariants()
    D = float((g2i ** 3 - 27.0 * g3i ** 2).real)
    if D <= 0.0:
        # unreachable on a rectangular lattice; kept as the Prop.-5 guard
        raise ValueError(
            "wavelike lattice (D <= 0): the only closed elastic curve "
            "there is the figure-eight, which yields no Willmore torus "
            "(Heller Prop. 5, Example 1)")
    target = m * pi / (2.0 * n)
    roots = _h2_closure_roots(lat, target)
    if not roots:
        raise ValueError(f"no closure root for target {target:.4f} at "
                         f"shape {t}")
    rho = 1.0 + 1j * roots[min(int(branch), len(roots) - 1)]
    E = float(lat.wp(rho).real)
    e1 = float(lat.wp(1.0).real)
    e2 = float(lat.wp(1.0 + lat.w3).real)
    e3 = float(lat.wp(lat.w3).real)
    P3E = 4.0 * E ** 3 - float(g2i.real) * E - float(g3i.real)
    x0 = 0.5 * lat.w3                     # lambda = 0, and nothing else

    x = np.arange(samples) * (2.0 * n / samples)
    w1, _, _, (g1v, g2v, lg1, lg2) = _h2_eval(lat, rho, 1.0, x0, x)

    # Wronskian det(ghat, ghat') -- constant for an exact solution pair
    Wr = g1v * g2v * (lg2 - lg1)
    wr_dev = float(np.abs(Wr - Wr.mean()).max() / np.abs(Wr.mean()))

    # Sect. 3.4 gauge, disc version: |g2|^2 - r^2 |g1|^2 = const
    p = np.abs(g1v) ** 2
    q = np.abs(g2v) ** 2
    vp = p - p.mean()
    r2 = float(np.dot(q - q.mean(), vp) / np.dot(vp, vp))
    if r2 <= 0.0:
        raise ValueError("gauge failed: not a disc (H^2) curve here")
    r = math.sqrt(r2)
    flat = float(np.std(q - r2 * p) / abs((q - r2 * p).mean()))

    w, wp, wpp, _ = _h2_eval(lat, rho, r, x0, x)
    if float(np.abs(w).max()) >= 1.0:
        raise ValueError("curve leaves the Poincare disc: shape too "
                         "extreme for the theta series")

    # constant Poincare speed = sqrt(-G)
    speed = 2.0 * np.abs(wp) / (1.0 - np.abs(w) ** 2)
    G_speed = -float(speed.mean()) ** 2
    speed_dev = float(speed.std() / speed.mean())

    # Lemma 2: kappa_G = 4 Im zeta + C ;  Re P + kappa^2/8 + b = 0
    zx = x + x0
    v = lat.zeta(zx).imag
    u = lat.wp(zx).real + 2.0 * v ** 2
    vv = v - v.mean()
    C = -float(np.dot(u - u.mean(), vv) / np.dot(vv, vv))
    b = -(float((u + C * v).mean()) + C * C / 8.0)
    G_b = 8.0 * b - 4.0 * E
    G_dev = abs(G_speed - G_b) / abs(G_b)
    sG = math.sqrt(-G_b)
    kappa_h = (4.0 * v + C) / sG      # geodesic curvature, H^2 (G = -1)

    # Euler-Lagrange fit in H^2 units: k'' + k^3/2 + (mu + G) k + lam = 0
    kpp = -4.0 * lat.wp_prime(zx).imag / (sG ** 3)
    rhs = -(kpp + 0.5 * kappa_h ** 3)
    A = np.stack([kappa_h, np.ones_like(kappa_h)], axis=1)
    (muG, lam), *_ = np.linalg.lstsq(A, rhs, rcond=None)
    el_res = float(np.abs(kpp + 0.5 * kappa_h ** 3 + muG * kappa_h + lam)
                   .max() / max(np.abs(kpp).max(), 1.0))

    # closure of the point curve and its tangent
    we, wpe, _, _ = _h2_eval(lat, rho, r, x0, np.array([0.0, 2.0 * n]))
    closure = float(abs(we[1] - we[0]))
    t_closure = float(abs(wpe[1] - wpe[0]))

    # Theorem 5 energy with the 1/sqrt(-G) surface normalisation, and the
    # analytic curvature integral it must equal
    W_closed = (8.0 * n * lat.eta1 - 4.0 * n * e3) * pi / sG
    dx = 2.0 * n / samples
    W_analytic = 0.5 * pi * sG * dx * float(np.sum(kappa_h ** 2))

    info = dict(lat=lat, rho=rho, r=r, x0=x0, E=E, e1=e1, e2=e2, e3=e3,
                D=D, P3E=P3E, G=G_b, b=b, muG=float(muG), lam=float(lam),
                kappa=kappa_h, length=2.0 * n * sG, n=n, m=m,
                W_closed=W_closed, W_analytic=W_analytic,
                wronskian_dev=wr_dev, norm_flat=flat, G_dev=G_dev,
                speed_dev=speed_dev, el_res=el_res, closure=closure,
                t_closure=t_closure, roots=roots, wp=wp, wpp=wpp)
    return w, info


def _disc_to_profile(w, wp, wpp, theta):
    """Rotate the disc by theta, Cayley-map to the upper half plane
    (0 -> i) and return the profile z = u + iv (v > 0) with derivatives."""
    ph = np.exp(1j * float(theta))
    wr, wpr, wppr = w * ph, wp * ph, wpp * ph
    z = 1j * (1.0 + wr) / (1.0 - wr)
    zp = 2j * wpr / (1.0 - wr) ** 2
    zpp = 2j * (wppr * (1.0 - wr) + 2.0 * wpr ** 2) / (1.0 - wr) ** 3
    return z, zp, zpp


def _profile_invariants(z, zp, zpp):
    """(H_R3, H_S3, speed) along the profile of the revolution torus
    p = (u, v cos phi, v sin phi), from the analytic derivatives.
    H_S3 = (1 + |p|^2) H_R3 / 2 + nu . p  is the mean curvature of the
    stereographic preimage in the UNIT 3-sphere; it is phi-invariant, and
    CONSTANT exactly when the torus is CMC in S^3 (Heller Prop. 6)."""
    uu, vv = z.real, z.imag
    sp = np.abs(zp)
    tg = zp / sp
    k1 = (zpp * np.conj(zp)).imag / sp ** 3       # meridian curvature
    k2 = -tg.real / vv                            # parallel curvature
    H3 = 0.5 * (k1 + k2)
    nu = 1j * tg                                  # profile normal
    HS3 = (0.5 * (1.0 + uu * uu + vv * vv) * H3
           + nu.real * uu + nu.imag * vv)
    return H3, HS3, sp


def _balance_theta(w, coarse=360):
    """Disc rotation minimising max |z| of the Cayley image -- the
    S^3-isometric placement with the most compact stereographic picture."""
    ths = np.linspace(0.0, TAU, coarse, endpoint=False)
    best_t, best_m = 0.0, np.inf
    for th in ths:
        wr = w * np.exp(1j * th)
        mx = float(np.abs(1j * (1.0 + wr) / (1.0 - wr)).max())
        if mx < best_m:
            best_t, best_m = float(th), mx
    lo, hi = best_t - TAU / coarse, best_t + TAU / coarse
    for _ in range(40):
        for th in (0.5 * (lo + best_t), 0.5 * (best_t + hi)):
            wr = w * np.exp(1j * th)
            mx = float(np.abs(1j * (1.0 + wr) / (1.0 - wr)).max())
            if mx < best_m:
                best_t, best_m = th, mx
        lo, hi = 0.5 * (lo + best_t), 0.5 * (best_t + hi)
    return best_t


def build_revolution_torus(t, n, m, ures=256, vres=64, scale=1.0,
                           branch=0, spin=0.0):
    """Route B surface: the constrained Willmore torus of revolution over
    the closed n-lobed elastic curve in H^2, meshed in R^3 and fitted to
    the 2 m cube.  Returns (verts, faces, info).

    `spin` (radians) turns the profile plane's elliptic placement away
    from the balanced position: an ISOMETRY of S^3 (rotation about the
    great circle the revolution fixes), so every invariant -- H in S^3,
    Willmore energy, conformal type -- is untouched; only the
    stereographic appearance in R^3 changes.  spin = 0 is the compact
    symmetric picture; increasing it slides one lobe toward the
    projection pole, reproducing the stacked-bubble views of Heller's
    Figure 1 (rendered by N. Schmitt).

    The profile is resampled uniformly in EUCLIDEAN arclength (the curve
    is arclength in H^2, whose Euclidean speed varies by an order of
    magnitude), by inverting the cumulative arclength on a dense grid and
    re-evaluating the closed form -- no interpolation of positions."""
    w, info = heller_h2_curve(t, n, m, samples=max(1024, 4 * ures),
                              branch=branch)
    theta = _balance_theta(w) + float(spin)
    lat, rho, r, x0 = info['lat'], info['rho'], info['r'], info['x0']

    # dense pass for the arclength table
    xs = np.arange(w.size) * (2.0 * n / w.size)
    z, zp, zpp = _disc_to_profile(w, info['wp'], info['wpp'], theta)
    sp = np.abs(zp)
    dx = 2.0 * n / w.size
    s_cum = np.concatenate([[0.0], np.cumsum(0.5 * (sp[1:] + sp[:-1])
                                             * dx)])
    s_tot = float(s_cum[-1] + 0.5 * (sp[-1] + sp[0]) * dx)
    s_want = np.arange(ures) * (s_tot / ures)
    x_res = np.interp(s_want, s_cum, xs)

    wR, wpR, wppR, _ = _h2_eval(lat, rho, r, x0, x_res)
    zR, zpR, zppR = _disc_to_profile(wR, wpR, wppR, theta)
    H3, HS3, spR = _profile_invariants(zR, zpR, zppR)

    # Willmore energy over the sampled surface, geometry only: PERIODIC
    # central differences on the resampled profile points (independent of
    # the analytic derivatives above; the profile is closed, so one-sided
    # endpoint stencils would poison the quadrature)
    uu, vv = zR.real, zR.imag

    def d1(f):
        return 0.5 * (np.roll(f, -1) - np.roll(f, 1))

    def d2(f):
        return np.roll(f, -1) - 2.0 * f + np.roll(f, 1)

    du, dv = d1(uu), d1(vv)
    ds = np.hypot(du, dv)
    kg = (du * d2(vv) - dv * d2(uu)) / ds ** 3
    Hg = 0.5 * (kg - du / (ds * vv))
    W_mesh = 2.0 * pi * float(np.sum(Hg ** 2 * vv * ds))

    phi = np.arange(vres) * (TAU / vres)
    V = np.empty((ures * vres, 3))
    V[:, 0] = np.repeat(uu, vres)
    V[:, 1] = (vv[:, None] * np.cos(phi)[None, :]).ravel()
    V[:, 2] = (vv[:, None] * np.sin(phi)[None, :]).ravel()
    faces = []
    for i in range(ures):
        i1 = (i + 1) % ures
        for j in range(vres):
            j1 = (j + 1) % vres
            faces.append((i * vres + j, i * vres + j1,
                          i1 * vres + j1, i1 * vres + j))
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    V = (V - 0.5 * (lo + hi)) * ((2.0 / ext if ext > 1e-9 else 1.0)
                                 * scale)
    info.update(theta=theta,
                H_S3=float(HS3.mean()),
                H_S3_dev=float(np.abs(HS3 - HS3.mean()).max()),
                W_mesh=W_mesh,
                W_rel=abs(W_mesh - info['W_closed']) / info['W_closed'],
                profile=zR)
    return V, faces, info


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

    # ------------------------------------------------------------------
    # Route B: tori of revolution over elastic curves in H^2
    # ------------------------------------------------------------------

    # 7) rectangular lattice engine identities
    for t in (0.6, 1.1):
        lat = _Rectangular(t)
        z = np.array([0.23 + 0.11j, -0.4 + 0.35j, 1.7 - 0.6j, 3.1 + 1.9j])
        h = 1e-6
        fd = (lat.sigma(z + h) - lat.sigma(z - h)) / (2 * h * lat.sigma(z))
        check(f"rB t={t} sigma'/sigma - zeta",
              float(np.abs(fd - lat.zeta(z)).max()), 1e-8)
        q2 = np.abs(lat.sigma(z + lat.p2) + lat.sigma(z)
                    * np.exp(lat.d3 * (z + lat.p2 / 2)))
        check(f"rB t={t} sigma quasi-periodicity",
              float((q2 / np.abs(lat.sigma(z + lat.p2))).max()), 1e-10)
        g2, g3 = lat.invariants()
        zz = np.array([0.2 + 0.3j, 1.1 + 0.7j])
        ode = np.abs(lat.wp_prime(zz) ** 2
                     - (4 * lat.wp(zz) ** 3 - g2 * lat.wp(zz) - g3))
        check(f"rB t={t} P ODE residual",
              float((ode / np.abs(lat.wp(zz)) ** 3).max()), 1e-10)
        check(f"rB t={t} D>0 (orbitlike)",
              -float(np.sign((g2 ** 3 - 27 * g3 ** 2).real)), 0.0,
              fmt="{:+.0f}")

    # 8) the n=2, m=1 elastic curve in H^2: every curve-level gate
    wB, bi = heller_h2_curve(0.8, 2, 1, samples=2048)
    check("rB Wronskian dev", bi['wronskian_dev'], 1e-12)
    check("rB 3.4-gauge flatness", bi['norm_flat'], 1e-12)
    check("rB speed constancy", bi['speed_dev'], 1e-12)
    check("rB G(speed) vs G(8b-4E)", bi['G_dev'], 1e-12)
    check("rB lambda=0 pin |b - e3/2|",
          abs(bi['b'] - 0.5 * bi['e3']), 1e-10)
    check("rB EL residual", bi['el_res'], 1e-12)
    check("rB |lambda|", abs(bi['lam']), 1e-12)
    check("rB mu+G fit vs 6 e3/(-G)",
          abs(bi['muG'] - 6.0 * bi['e3'] / (-bi['G'])), 1e-10)
    check("rB curve closure", bi['closure'], 1e-10)
    check("rB tangent closure", bi['t_closure'], 1e-10)
    check("rB in disc: 1 - max|w|", -(float(np.abs(wB).max()) - 1.0),
          1.0, fmt="{:.3f}")
    kb = bi['kappa']
    lobB = int(np.sum((kb > np.roll(kb, 1)) & (kb > np.roll(kb, -1))))
    check("rB lobe count |n-2|", abs(lobB - 2), 0.5, fmt="{:.0f}")
    check("rB orbitlike P3(E) < 0", float(np.sign(bi['P3E'])), 0.0,
          fmt="{:+.0f}")

    # 9) Theorem 5 energy: closed form (surface-normalised by 1/sqrt(-G))
    #    against the analytic curvature integral, and the family's two
    #    closed-form anchors -- W -> 8 pi at the doubled-sphere end and
    #    W -> 4 pi^2 / sqrt(3) at the H = 1/sqrt(3) homogeneous-torus
    #    bifurcation (the sanity anchors of the 2-lobed CMC family)
    check("rB W closed form vs analytic",
          abs(bi['W_closed'] - bi['W_analytic']) / bi['W_closed'], 1e-9)
    _, biS = heller_h2_curve(0.35, 2, 1, samples=2048)
    check("rB sphere-doubling limit |W - 8pi|/8pi",
          abs(biS['W_closed'] - 8.0 * pi) / (8.0 * pi), 1e-3)
    _, biC = heller_h2_curve(5.0, 2, 1, samples=2048)
    Wbif = 4.0 * pi * pi / math.sqrt(3.0)
    check("rB bifurcation limit |W - 4pi^2/sqrt3|/W",
          abs(biC['W_closed'] - Wbif) / Wbif, 1e-3)

    # 10) the H_S3 formula itself, against ground truth: the flat torus
    #     |z1|^2 = c2 in S^3 has H_S3 = (2 c2 - 1)/(2 sqrt(c2 (1 - c2)));
    #     its stereographic image is a round torus of revolution
    c2 = 0.3
    cc, ss = math.sqrt(c2), math.sqrt(1.0 - c2)
    aF = np.linspace(0.1, 2.0 * pi + 0.1, 257)
    zF = (cc * np.sin(aF) + 1j * ss) / (1.0 - cc * np.cos(aF))
    zpF = (cc * np.cos(aF) * (1.0 - cc * np.cos(aF))
           - (cc * np.sin(aF) + 1j * ss) * cc * np.sin(aF)) \
        / (1.0 - cc * np.cos(aF)) ** 2
    hF = 1e-5
    zppF = ((cc * np.sin(aF + hF) + 1j * ss) / (1 - cc * np.cos(aF + hF))
            - 2.0 * zF
            + (cc * np.sin(aF - hF) + 1j * ss)
            / (1 - cc * np.cos(aF - hF))) / hF ** 2
    _, HS3F, _ = _profile_invariants(zF, zpF, zppF)
    HwantF = (2.0 * c2 - 1.0) / (2.0 * cc * ss)
    check("rB H_S3 formula vs flat torus",
          float(np.abs(np.abs(HS3F) - abs(HwantF)).max()), 1e-4)

    # 11) Proposition 6, the point of it all: the built tori are CMC in
    #     the unit S^3 -- H_S3 constant along the profile
    for (tt, nn, mm) in ((0.8, 2, 1), (0.8, 3, 1), (1.2, 3, 2)):
        VB, FB, ib = build_revolution_torus(tt, nn, mm, ures=220, vres=48)
        check(f"rB CMC in S^3 ({nn},{mm}) t={tt}: H dev",
              ib['H_S3_dev'] / max(abs(ib['H_S3']), 0.1), 1e-8)
        check(f"rB W mesh vs closed ({nn},{mm})", ib['W_rel'], 5e-3)
        VB = np.asarray(VB)
        extB = VB.max(0) - VB.min(0)
        good = (len(FB) == 220 * 48 and np.isfinite(VB).all()
                and float(extB.max()) <= 2.0 + 1e-9
                and float(extB.min() / extB.max()) > 0.15)
        ok_all = ok_all and good
        print(f"cw: rB build ({nn},{mm}): V={len(VB)} F={len(FB)} "
              f"aspect {extB.min() / extB.max():.3f} "
              f"H_S3={ib['H_S3']:+.4f} {'OK' if good else 'BAD'}")

    # 12) spin is an S^3 isometry: it must change the R^3 picture and
    #     change NOTHING measured in S^3
    _, _, ia = build_revolution_torus(0.8, 2, 1, ures=200, vres=32)
    _, _, ic = build_revolution_torus(0.8, 2, 1, ures=200, vres=32,
                                      spin=0.9)
    check("rB spin invariance of H_S3",
          abs(ia['H_S3'] - ic['H_S3']), 1e-10)
    check("rB spin: still CMC", ic['H_S3_dev'], 1e-8)
    check("rB spin invariance of W (closed)",
          abs(ia['W_closed'] - ic['W_closed']), 1e-12)
    check("rB spin W mesh vs closed", ic['W_rel'], 5e-3)

    assert ok_all, "constrained_willmore_generator self-test FAILED"
    print("cw: all checks OK")
