# Weierstrass elliptic-function engine (Jacobi-theta series).
#
# Part of the Math Art minimal-surface engine (`math_art/minsurf/`), split
# out of the former single-file `minimal_surface_toolkit.py`.  Numpy only --
# no `bpy` -- so the whole engine imports and self-tests headlessly; the
# registered Blender operators stay in the flat `minimal_surface_toolkit.py`
# front-end.
#
# Also provides the JACOBI elliptic functions sn/cn/dn and the complete
# elliptic integrals K(m), E(m) for real argument and real parameter
# m = k^2 in [0,1), evaluated by the arithmetic-geometric mean rather than
# by the theta series above.  The AGM is the right tool here: it converges
# quadratically for every m in range (the nome q -> 1 as m -> 1, where the
# theta series above would need unboundedly many terms), it needs no
# lattice object, and it is a dozen lines of numpy.  These are what the
# spherical elastica of Langer-Singer needs -- its curvature is
# k(s) = sqrt(a) cn(rs, p) -- so they are used by the Hopf-torus generator
# to build Pinkall's Willmore tori.
#
# References:
#   Weierstrass P, P' and zeta via Jacobi theta functions: DLMF 23.6
#   (elliptic functions) and DLMF 20.2 (the theta q-series).
#   K. Weierstrass, Mathematische Werke (1894-1927).
#   Jacobi sn/cn/dn by descending Landen/AGM, and K, E from the same
#   iteration: M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
#   Functions" (1964), 16.4 (AGM scale) and 17.6 (complete integrals);
#   DLMF 22.20(ii) and 19.8.  C. G. J. Jacobi, "Fundamenta nova theoriae
#   functionum ellipticarum" (1829).

import math
import numpy as np

TAU = 2.0 * math.pi

# --------------------------------------------------------------------------
# Jacobi elliptic functions and complete elliptic integrals (real, by AGM)
# --------------------------------------------------------------------------
# Parameter convention: everything below takes the PARAMETER m = k^2, not
# the modulus k.  (Both conventions are common and mixing them is the
# classic way to get a subtly wrong curve; the argument name is `m`
# throughout, and helpers that want a modulus say so.)

_AGM_MAX = 60          # quadratic convergence: ~7 iterations at m = 1-1e-12
_AGM_TOL = 1e-16


def _agm_scale(m):
    """Descending AGM for parameter m: returns (a, c) lists with
    a[0] = 1, b[0] = sqrt(1-m), c[0] = sqrt(m), iterated until c is
    negligible.  Shared by ellipk/ellipe and jacobi_sncndn."""
    m = float(m)
    if not (0.0 <= m < 1.0):
        raise ValueError(f"parameter m must lie in [0,1), got {m}")
    a, b, c = 1.0, math.sqrt(1.0 - m), math.sqrt(m)
    A, C = [a], [c]
    for _ in range(_AGM_MAX):
        if abs(c) < _AGM_TOL:
            break
        a, b, c = 0.5 * (a + b), math.sqrt(a * b), 0.5 * (a - b)
        A.append(a)
        C.append(c)
    return A, C


def ellipk(m):
    """Complete elliptic integral of the first kind K(m), m = k^2.
    K = pi / (2 * AGM(1, sqrt(1-m)))."""
    A, _ = _agm_scale(m)
    return math.pi / (2.0 * A[-1])


def ellipe(m):
    """Complete elliptic integral of the second kind E(m), m = k^2.
    E = K * (1 - sum_n 2^(n-1) c_n^2)   (A&S 17.6.4; the n = 0 term is
    c_0^2/2 = m/2)."""
    A, C = _agm_scale(m)
    K = math.pi / (2.0 * A[-1])
    s = sum(2.0 ** (n - 1) * c * c for n, c in enumerate(C))
    return K * (1.0 - s)


def jacobi_sncndn(u, m):
    """Jacobi sn(u|m), cn(u|m), dn(u|m) for real u (scalar or ndarray) and
    real parameter m = k^2 in [0,1).  Returns (sn, cn, dn) as ndarrays of
    u's shape.

    Descending Landen transformation on the AGM scale (A&S 16.4): carry
    phi down from phi_N = 2^N a_N u by
        phi_{n-1} = (phi_n + arcsin((c_n / a_n) sin phi_n)) / 2 ,
    then sn = sin phi_0, cn = cos phi_0, dn = sqrt(1 - m sn^2).

    dn is taken from the identity rather than accumulated, which keeps
    m sn^2 + dn^2 = 1 exact to rounding; the clip guards only against a
    negative epsilon under the root at the turning points."""
    u = np.asarray(u, dtype=float)
    A, C = _agm_scale(m)
    N = len(A) - 1
    phi = (2.0 ** N) * A[N] * u
    for n in range(N, 0, -1):
        phi = 0.5 * (phi + np.arcsin(np.clip((C[n] / A[n]) * np.sin(phi),
                                             -1.0, 1.0)))
    sn = np.sin(phi)
    cn = np.cos(phi)
    dn = np.sqrt(np.clip(1.0 - m * sn * sn, 0.0, None))
    return sn, cn, dn


def jacobi_am(u, m):
    """Jacobi amplitude am(u|m): the angle phi with sn(u|m) = sin(phi).

    Free from the descending Landen recursion above -- phi_0 IS the
    amplitude -- and unlike arcsin(sn) it does not fold at the quarter
    periods, so it keeps growing monotonically with u.  That matters
    wherever the amplitude appears inside another integral rather than
    inside a trigonometric function."""
    u = np.asarray(u, dtype=float)
    A, C = _agm_scale(m)
    N = len(A) - 1
    phi = (2.0 ** N) * A[N] * u
    for n in range(N, 0, -1):
        phi = 0.5 * (phi + np.arcsin(np.clip((C[n] / A[n]) * np.sin(phi),
                                             -1.0, 1.0)))
    return phi


# Gauss-Legendre nodes/weights on [-1, 1], built once by Newton iteration
# on the Legendre polynomials (numpy's leggauss would do, but this keeps
# the module's "no scipy, and no surprises" character and is exact to
# rounding).
def _leggauss(n):
    i = np.arange(1, n, dtype=float)
    beta = i / np.sqrt(4.0 * i * i - 1.0)          # Golub-Welsch
    J = np.diag(beta, -1) + np.diag(beta, 1)
    x, V = np.linalg.eigh(J)
    return x, 2.0 * V[0, :] ** 2


_GL_X, _GL_W = _leggauss(48)


def ellippi(n, phi, m, segments=None):
    """Incomplete elliptic integral of the THIRD kind,

        Pi(n; phi | m) = int_0^phi dt / ((1 - n sin^2 t) sqrt(1 - m sin^2 t))

    for real n < 1, real m < 1 and real phi (scalar or ndarray).

    Evaluated by composite 48-point Gauss-Legendre over sub-intervals of
    length <= pi/2.  The integrand is analytic away from n sin^2 t = 1 and
    m sin^2 t = 1, so Gauss-Legendre converges spectrally and 48 nodes per
    half-period is far past machine precision; the subdivision is only
    there to stop a long phi from making one panel span many oscillations.

    NOT valid for n >= 1, where the integrand has a pole inside the range
    and the integral is a Cauchy principal value -- that case raises
    rather than returning a plausible wrong number.  (Carlson's R_J would
    be the general answer if it is ever needed.)"""
    if n >= 1.0:
        raise ValueError(
            f"ellippi: n = {n} >= 1 puts a pole inside the interval; "
            f"the principal-value branch is not implemented")
    if m >= 1.0:
        raise ValueError(f"ellippi: m = {m} >= 1 is out of range")
    ph = np.asarray(phi, dtype=float)
    if segments is None:
        segments = int(np.ceil(float(np.abs(ph).max()) /
                               (0.5 * math.pi))) + 1
    segments = max(1, int(segments))
    # t = ph * (s + (xi+1)/2) / segments over s = 0 .. segments-1
    s = np.arange(segments, dtype=float)
    # nodes for every (sample, segment, gauss-node)
    frac = (s[:, None] + 0.5 * (_GL_X[None, :] + 1.0)) / segments
    t = ph[..., None, None] * frac                 # (..., seg, node)
    st2 = np.sin(t) ** 2
    f = 1.0 / ((1.0 - n * st2) * np.sqrt(1.0 - m * st2))
    w = _GL_W[None, :] * 0.5 / segments
    return ph * np.sum(f * w, axis=(-2, -1))


# ==========================================================================
# Weierstrass elliptic-function engine (Jacobi-theta series, numpy only)
# ==========================================================================
# Provides Weierstrass P, P' and zeta for a lattice given by half-periods,
# via the Jacobi theta functions (DLMF 23.6 for the elliptic functions,
# DLMF 20.2 for the theta q-series). The nome q = exp(i*pi*tau) is small
# for the lattices we use, so ~a dozen terms of each series reach 1e-15.
#
# Costa and Chen-Gackstatter both live on the square (lemniscatic) torus:
#   periods 1, i ; half-periods w1 = 1/2, w3 = i/2 ; tau = i ; q = e^-pi ;
#   g2 = Gamma(1/4)^8 / (16 pi^2) = 189.0727... , g3 = 0 ,
#   e1 = P(1/2) = 6.87519... , and (this lattice) g2 = 4 e1^2.

_THETA_TERMS = 16   # q^((n+.5)^2) underflows long before this for our q


def _theta1_series(xi, q):
    """theta1(xi) and its first three xi-derivatives (t0..t3), where xi is
    a complex ndarray. DLMF 20.2.1 differentiated term by term."""
    n = np.arange(_THETA_TERMS)
    a = ((-1.0) ** n) * q ** ((n + 0.5) ** 2)          # (terms,)
    k = (2 * n + 1).astype(float)
    ang = np.multiply.outer(np.asarray(xi, dtype=complex), k)
    s, c = np.sin(ang), np.cos(ang)
    t0 = 2.0 * np.sum(a * s, axis=-1)
    t1 = 2.0 * np.sum(a * k * c, axis=-1)
    t2 = -2.0 * np.sum(a * k ** 2 * s, axis=-1)
    t3 = -2.0 * np.sum(a * k ** 3 * c, axis=-1)
    return t0, t1, t2, t3


class _Lattice:
    """Weierstrass P, P', zeta on the lattice with real half-period w1 and
    ratio tau = w3/w1 (Im tau > 0). All methods are vectorized over z."""

    def __init__(self, w1, tau):
        self.w1 = float(w1)
        self.q = np.exp(1j * math.pi * tau)
        self.c = math.pi / (2.0 * self.w1)             # dxi/dz
        # quasi-period eta1 = zeta(w1)  (DLMF 23.6.8), from theta1 at 0
        n = np.arange(_THETA_TERMS)
        a = ((-1.0) ** n) * self.q ** ((n + 0.5) ** 2)
        k = (2 * n + 1).astype(float)
        t1_0 = 2.0 * np.sum(a * k)                      # theta1'(0)
        t3_0 = -2.0 * np.sum(a * k ** 3)               # theta1'''(0)
        self.eta1 = -(math.pi ** 2 / (12.0 * self.w1)) * (t3_0 / t1_0)

    def zeta(self, z):
        z = np.asarray(z, dtype=complex)
        t0, t1, _, _ = _theta1_series(self.c * z, self.q)
        return (self.eta1 / self.w1) * z + self.c * (t1 / t0)

    def wp(self, z):
        z = np.asarray(z, dtype=complex)
        t0, t1, t2, _ = _theta1_series(self.c * z, self.q)
        r1 = t1 / t0
        return -self.eta1 / self.w1 - self.c ** 2 * (t2 / t0 - r1 ** 2)

    def wp_prime(self, z):
        z = np.asarray(z, dtype=complex)
        t0, t1, t2, t3 = _theta1_series(self.c * z, self.q)
        r1 = t1 / t0
        return -self.c ** 3 * (t3 / t0 - 3.0 * r1 * (t2 / t0) + 2.0 * r1 ** 3)


# The square torus shared by Costa and Chen-Gackstatter.
_SQUARE = _Lattice(0.5, 1j)


def _selftest():
    # The square (lemniscatic) lattice is pinned by closed-form constants, so
    # a transposed index or a dropped theta term shows up immediately.
    ok = True
    L = _SQUARE

    # e1 = P(w1) = P(1/2), and on this lattice g3 = 0 so g2 = 4 e1^2.
    e1 = L.wp(0.5).real
    g2 = 4.0 * e1 ** 2
    good = abs(e1 - 6.875185818) < 1e-6 and abs(g2 - 189.0727215) < 1e-4
    ok &= good
    print(f"elliptic: e1={e1:.9f} (exp 6.875185818) g2={g2:.6f} "
          f"(exp 189.0727215) {'OK' if good else 'FAIL'}")

    # The defining ODE: P'^2 = 4 P^3 - g2 P - g3, with g3 = 0 here.  This is
    # the real check -- it ties P and P' together at arbitrary points.
    z = np.array([0.2 + 0.3j, 0.37 + 0.11j, 0.6 + 0.44j, 0.13 + 0.71j])
    resid = float(np.max(np.abs(L.wp_prime(z) ** 2
                                - (4.0 * L.wp(z) ** 3 - g2 * L.wp(z)))))
    good = resid < 1e-8
    ok &= good
    print(f"elliptic: max|P'^2-(4P^3-g2 P)|={resid:.3e} "
          f"{'OK' if good else 'FAIL'}")

    # Double periodicity on the period lattice (1, i): P(z + w) = P(z).
    per = float(max(np.max(np.abs(L.wp(z + 1.0) - L.wp(z))),
                    np.max(np.abs(L.wp(z + 1j) - L.wp(z)))))
    good = per < 1e-8
    ok &= good
    print(f"elliptic: max|P(z+w)-P(z)|={per:.3e} {'OK' if good else 'FAIL'}")

    # P is even, P' is odd -- catches a sign slip in the theta derivatives.
    par = float(max(np.max(np.abs(L.wp(-z) - L.wp(z))),
                    np.max(np.abs(L.wp_prime(-z) + L.wp_prime(z)))))
    good = par < 1e-8
    ok &= good
    print(f"elliptic: parity residual={par:.3e} {'OK' if good else 'FAIL'}")

    # Legendre-style consistency of the quasi-period: zeta(z+1) - zeta(z)
    # must equal the constant 2*eta1 everywhere.
    d = L.zeta(z + 1.0) - L.zeta(z)
    quasi = float(np.max(np.abs(d - 2.0 * L.eta1)))
    good = quasi < 1e-8
    ok &= good
    print(f"elliptic: max|zeta(z+1)-zeta(z)-2eta1|={quasi:.3e} "
          f"{'OK' if good else 'FAIL'}")

    ok &= _selftest_jacobi()
    ok &= _selftest_ellippi()

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("elliptic self-test failed")


def _selftest_jacobi():
    """Jacobi sn/cn/dn and the complete integrals.  Every check here is
    against a value known in closed form or an identity that ties the
    three functions together, so a swapped a/c index or an m-vs-k mixup
    fails loudly."""
    ok = True

    # K and E at m = 0 are pi/2; at m = 1/2 they are the classical
    # lemniscatic-adjacent constants (DLMF 19.5 / A&S table 17.1).
    k0 = (abs(ellipk(0.0) - math.pi / 2) < 1e-15
          and abs(ellipe(0.0) - math.pi / 2) < 1e-15)
    kh = (abs(ellipk(0.5) - 1.8540746773013719) < 1e-13
          and abs(ellipe(0.5) - 1.3506438810476755) < 1e-13)
    good = k0 and kh
    ok &= good
    print(f"jacobi: K(0)={ellipk(0.0):.12f} K(.5)={ellipk(0.5):.12f} "
          f"E(.5)={ellipe(0.5):.12f} {'OK' if good else 'FAIL'}")

    # m = 0 degenerates to circular functions: sn=sin, cn=cos, dn=1.
    u = np.linspace(-7.0, 7.0, 401)
    sn, cn, dn = jacobi_sncndn(u, 0.0)
    deg = float(max(np.max(np.abs(sn - np.sin(u))),
                    np.max(np.abs(cn - np.cos(u))),
                    np.max(np.abs(dn - 1.0))))
    good = deg < 1e-12
    ok &= good
    print(f"jacobi: m=0 vs (sin,cos,1) residual={deg:.3e} "
          f"{'OK' if good else 'FAIL'}")

    # The two Pythagorean identities, over a range of m including the
    # p^2 < 1/2 band the spherical elastica lives in.
    worst = 0.0
    for m in (0.05, 0.25, 0.4999, 0.75, 0.95, 0.999):
        sn, cn, dn = jacobi_sncndn(u, m)
        worst = max(worst,
                    float(np.max(np.abs(sn ** 2 + cn ** 2 - 1.0))),
                    float(np.max(np.abs(m * sn ** 2 + dn ** 2 - 1.0))))
    good = worst < 1e-12
    ok &= good
    print(f"jacobi: max identity residual={worst:.3e} "
          f"{'OK' if good else 'FAIL'}")

    # Quarter-period values: sn(K)=1, cn(K)=0, dn(K)=sqrt(1-m)=k'.
    worst = 0.0
    for m in (0.1, 0.3, 0.5, 0.8):
        s, c, d = jacobi_sncndn(ellipk(m), m)
        worst = max(worst, abs(float(s) - 1.0), abs(float(c)),
                    abs(float(d) - math.sqrt(1.0 - m)))
    good = worst < 1e-10
    ok &= good
    print(f"jacobi: max|(sn,cn,dn)(K)-(1,0,k')|={worst:.3e} "
          f"{'OK' if good else 'FAIL'}")

    # Derivative law d(sn)/du = cn*dn, by central differences.  This is
    # the check that would catch a wrong Landen descent, since it probes
    # the u-scaling rather than the pointwise identities.
    m, h = 0.36, 1e-5
    x = np.linspace(0.3, 5.0, 200)
    s1, _, _ = jacobi_sncndn(x + h, m)
    s0, _, _ = jacobi_sncndn(x - h, m)
    _, c, d = jacobi_sncndn(x, m)
    der = float(np.max(np.abs((s1 - s0) / (2 * h) - c * d)))
    good = der < 1e-8
    ok &= good
    print(f"jacobi: max|d(sn)/du - cn*dn|={der:.3e} "
          f"{'OK' if good else 'FAIL'}")

    # Periodicity: sn has period 4K, cn has period 4K, dn has period 2K.
    m = 0.42
    K = ellipk(m)
    s0, c0, d0 = jacobi_sncndn(x, m)
    s4, c4, _ = jacobi_sncndn(x + 4.0 * K, m)
    _, _, d2 = jacobi_sncndn(x + 2.0 * K, m)
    per = float(max(np.max(np.abs(s4 - s0)), np.max(np.abs(c4 - c0)),
                    np.max(np.abs(d2 - d0))))
    good = per < 1e-9
    ok &= good
    print(f"jacobi: max period residual (4K,4K,2K)={per:.3e} "
          f"{'OK' if good else 'FAIL'}")

    return ok


def _selftest_ellippi():
    """The amplitude and the third-kind integral, each against a closed
    form that does NOT go through the same quadrature."""
    ok = True

    # 1) am is the true amplitude: sin(am(u)) = sn(u), and unlike
    #    arcsin(sn) it keeps increasing rather than folding at each
    #    quarter period.
    for m in (0.0, 0.3, 0.75, 0.96):
        u = np.linspace(-9.0, 9.0, 401)
        phi = jacobi_am(u, m)
        sn = jacobi_sncndn(u, m)[0]
        dev = float(np.abs(np.sin(phi) - sn).max())
        monotone = bool(np.all(np.diff(phi) > 0.0))
        good = dev < 1e-12 and monotone
        ok &= good
        print(f"ellippi: am m={m:.2f} max|sin(am)-sn|={dev:.2e} "
              f"monotone={monotone} {'OK' if good else 'FAIL'}")

    # 2) Pi(n; phi | 0) = arctan(sqrt(1-n) tan phi)/sqrt(1-n), continued
    #    across the branch -- exact, and independent of the quadrature.
    for n in (-2.0, -0.4, 0.0, 0.35, 0.8):
        phi = np.linspace(0.05, 1.4, 40)
        got = ellippi(n, phi, 0.0)
        r = math.sqrt(1.0 - n)
        want = np.arctan(r * np.tan(phi)) / r
        dev = float(np.abs(got - want).max())
        good = dev < 1e-13
        ok &= good
        print(f"ellippi: Pi(n={n:+.2f}; phi|0) vs closed form "
              f"max dev {dev:.2e} {'OK' if good else 'FAIL'}")

    # 3) The COMPLETE case at n = m: Pi(m | m) = E(m)/(1-m), which ties
    #    the new quadrature to the AGM series for E.
    for m in (0.1, 0.4, 0.7, 0.9):
        got = float(ellippi(m, math.pi / 2.0, m))
        want = ellipe(m) / (1.0 - m)
        dev = abs(got - want)
        good = dev < 1e-12
        ok &= good
        print(f"ellippi: Pi(m={m:.1f}|m)={got:.12f} vs E/(1-m)="
              f"{want:.12f} dev {dev:.1e} {'OK' if good else 'FAIL'}")

    # 4) n = 0 reduces to the first kind, whose inverse is the amplitude:
    #    am(F(phi|m) | m) = phi.  This closes the loop between the new
    #    quadrature and the AGM recursion, with no shared arithmetic.
    for m in (0.2, 0.55, 0.88):
        phi = np.linspace(0.1, 3.0, 30)
        F = ellippi(0.0, phi, m)
        dev = float(np.abs(jacobi_am(F, m) - phi).max())
        good = dev < 1e-12
        ok &= good
        print(f"ellippi: am(F(phi|m)|m) == phi, m={m:.2f} max dev "
              f"{dev:.2e} {'OK' if good else 'FAIL'}")

    # 5) additivity across a quarter period, i.e. the composite panels
    #    agree with a single long integration
    m, n = 0.6, 0.25
    a = float(ellippi(n, 2.5, m))
    b = float(ellippi(n, 1.0, m)) + float(
        ellippi(n, 2.5, m) - ellippi(n, 1.0, m))
    long_ = float(ellippi(n, 12.0, m, segments=40))
    short = float(ellippi(n, 12.0, m, segments=200))
    dev = abs(long_ - short)
    good = abs(a - b) < 1e-14 and dev < 1e-12
    ok &= good
    print(f"ellippi: panel independence over phi=12 dev {dev:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # 6) n >= 1 is a principal value and must be refused, not guessed
    refused = 0
    for bad in (1.0, 2.5):
        try:
            ellippi(bad, 1.0, 0.5)
        except ValueError:
            refused += 1
    good = refused == 2
    ok &= good
    print(f"ellippi: {refused}/2 principal-value cases refused "
          f"{'OK' if good else 'FAIL'}")

    return ok
