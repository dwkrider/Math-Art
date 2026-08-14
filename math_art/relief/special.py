"""relief.special -- classical special functions, NumPy only.

Blender ships NumPy and no SciPy, so every special function the relief engine
needs is implemented here.  Each is chosen for numerical behaviour rather than
for looking like the textbook formula:

* **Bessel J_m** by its integral representation.  The integrand is periodic and
  analytic, so the trapezoid rule converges geometrically -- and it does so as
  a *cliff*: below a threshold node count the answer is badly wrong, above it
  the error drops to machine precision at once.  `bessel_nodes` gives the rule.
* **Bessel zeros** by McMahon's asymptotic expansion refined with Newton.
* **Zernike radial polynomials** through the Jacobi recurrence, NOT the finite
  factorial sum.  The factorial form loses all significance well inside the
  range a user can reach with an "aberration order" control -- it is already
  wrong in the tenth digit by n = 20 and useless by n = 50 -- so it is not
  offered at all.
* **Hermite and Laguerre** by their three-term recurrences.

References:
  NIST Digital Library of Mathematical Functions, https://dlmf.nist.gov/ --
    10.9.2 (the Bessel integral representation, integer order), 10.21.19
    (McMahon's expansion for the zeros), 10.6 (the derivative used by Newton),
    18.9 (Jacobi, Hermite and Laguerre recurrences).
  Lloyd N. Trefethen and J. A. C. Weideman, "The Exponentially Convergent
    Trapezoidal Rule", SIAM Review 56(3), 2014, 385-458 -- why the trapezoid
    rule is the right quadrature for a periodic analytic integrand.
  Frits Zernike, "Beugungstheorie des Schneidenverfahrens und seiner
    verbesserten Form, der Phasenkontrastmethode", Physica 1, 1934, 689-704.
  Robert J. Noll, "Zernike polynomials and atmospheric turbulence", Journal of
    the Optical Society of America 66(3), 1976, 207-211 -- one of the two
    common index conventions; this module exposes (n, m) natively instead,
    because Noll and the OSA/ANSI scheme disagree on ordering and sign.
"""

import math

import numpy as np


# ------------------------------------------------------------------
# Bessel functions of the first kind, integer order
# ------------------------------------------------------------------

def bessel_nodes(x_max, m=0):
    """Trapezoid nodes needed for machine accuracy up to `x_max`.

    The error behaves like J_{2N-m}(x), which collapses once 2N - m exceeds x;
    N >= (x_max + m)/2 + 20 puts the result comfortably past the cliff.
    """
    return int(max(32, math.ceil((float(x_max) + abs(int(m))) / 2.0) + 20))


def besselj(m, x, nodes=None):
    """J_m(x) for integer order m, via DLMF 10.9.2.

        J_m(x) = (1/pi) * integral_0^pi cos(m tau - x sin tau) d tau

    Accuracy is *absolute* (~1e-15); the relative error is poor where J_m is
    exponentially small, which does not matter for a height field.  Integer
    order only -- the general-order representation carries an extra
    non-periodic term and this quadrature would silently mis-handle it.
    """
    m = int(m)
    x = np.asarray(x, dtype=float)
    n = nodes or bessel_nodes(float(np.abs(x).max()) if x.size else 1.0, m)
    tau = np.linspace(0.0, math.pi, n + 1)
    w = np.full(n + 1, 2.0)
    w[0] = w[-1] = 1.0                     # trapezoid weights
    w *= (math.pi / n) / 2.0
    arg = m * tau[None, :] - x.reshape(-1, 1) * np.sin(tau)[None, :]
    return ((np.cos(arg) * w[None, :]).sum(axis=1) / math.pi).reshape(x.shape)


def besselj_prime(m, x, nodes=None):
    """d/dx J_m(x) = (J_{m-1}(x) - J_{m+1}(x)) / 2   (DLMF 10.6)."""
    return 0.5 * (besselj(m - 1, x, nodes) - besselj(m + 1, x, nodes))


def bessel_zero(m, n):
    """The n-th positive zero j_{m,n} of J_m (n starts at 1).

    McMahon's expansion (DLMF 10.21.19) supplies the seed, Newton polishes it.
    """
    m = int(m)
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    mu = 4.0 * m * m
    a = (n + 0.5 * m - 0.25) * math.pi
    a8 = 8.0 * a
    z = (a
         - (mu - 1.0) / a8
         - 4.0 * (mu - 1.0) * (7.0 * mu - 31.0) / (3.0 * a8 ** 3)
         - 32.0 * (mu - 1.0) * (83.0 * mu * mu - 982.0 * mu + 3779.0)
         / (15.0 * a8 ** 5)
         - 64.0 * (mu - 1.0) * (6949.0 * mu ** 3 - 153855.0 * mu * mu
                                + 1585743.0 * mu - 6277237.0)
         / (105.0 * a8 ** 7))
    nodes = bessel_nodes(z + 5.0, m + 1)
    for _ in range(60):
        f = float(besselj(m, np.array([z]), nodes)[0])
        fp = float(besselj_prime(m, np.array([z]), nodes)[0])
        if abs(fp) < 1e-300:
            break
        step = f / fp
        z -= step
        if abs(step) < 1e-15 * max(1.0, abs(z)):
            break
    return z


def bessel_zeros(m, count):
    """The first `count` positive zeros of J_m."""
    return np.array([bessel_zero(m, i + 1) for i in range(int(count))])


# ------------------------------------------------------------------
# Zernike radial polynomials
# ------------------------------------------------------------------

def zernike_radial(n, m, rho):
    """R_n^m(rho) via R_n^m = rho^m * P_{(n-m)/2}^{(0,m)}(2 rho^2 - 1).

    The Jacobi three-term recurrence is stable to high order; the equivalent
    factorial sum is not, and is deliberately absent from this module.
    """
    n = int(n)
    m = int(abs(m))
    if (n - m) % 2 or n < m:
        raise ValueError("Zernike needs n >= |m| with n - |m| even")
    rho = np.asarray(rho, dtype=float)
    jmax = (n - m) // 2
    x = 2.0 * rho * rho - 1.0
    p_prev = np.ones_like(x)                       # P_0^{(0,m)}
    if jmax == 0:
        return rho ** m * p_prev
    p = ((m + 2.0) * x - m) / 2.0                  # P_1^{(0,m)}
    for j in range(2, jmax + 1):
        c1 = (2.0 * j + m - 1.0) * ((2.0 * j + m) * (2.0 * j + m - 2.0) * x
                                    - m * m)
        c2 = 2.0 * (j - 1.0) * (j + m - 1.0) * (2.0 * j + m)
        den = 2.0 * j * (j + m) * (2.0 * j + m - 2.0)
        p, p_prev = (c1 * p - c2 * p_prev) / den, p
    return rho ** m * p


def zernike(n, m, rho, theta):
    """Zernike polynomial Z_n^m: cosine for m >= 0, sine for m < 0."""
    r = zernike_radial(n, m, rho)
    if m == 0:
        return r
    return r * (np.cos(m * theta) if m > 0 else np.sin(-m * theta))


# ------------------------------------------------------------------
# Hermite and Laguerre
# ------------------------------------------------------------------

def hermite(n, x):
    """Physicists' Hermite H_n by the recurrence H_{k+1} = 2x H_k - 2k H_{k-1}."""
    n = int(n)
    x = np.asarray(x, dtype=float)
    h_prev = np.ones_like(x)
    if n == 0:
        return h_prev
    h = 2.0 * x
    for k in range(1, n):
        h, h_prev = 2.0 * x * h - 2.0 * k * h_prev, h
    return h


def hermite_function(n, x):
    """psi_n(x) = H_n(x) exp(-x^2/2) / sqrt(2^n n! sqrt(pi)).

    Recursed with the Gaussian already folded in, so it does not overflow at
    the orders where the bare polynomial does.
    """
    n = int(n)
    x = np.asarray(x, dtype=float)
    psi_prev = np.exp(-0.5 * x * x) / math.pi ** 0.25
    if n == 0:
        return psi_prev
    psi = math.sqrt(2.0) * x * psi_prev
    for k in range(1, n):
        psi, psi_prev = (x * math.sqrt(2.0 / (k + 1.0)) * psi
                         - math.sqrt(k / (k + 1.0)) * psi_prev), psi
    return psi


def laguerre(p, alpha, x):
    """Generalised Laguerre L_p^alpha by its three-term recurrence."""
    p = int(p)
    x = np.asarray(x, dtype=float)
    l_prev = np.ones_like(x)
    if p == 0:
        return l_prev
    l = 1.0 + alpha - x
    for k in range(1, p):
        l, l_prev = (((2.0 * k + 1.0 + alpha - x) * l
                      - (k + alpha) * l_prev) / (k + 1.0), l)
    return l


def _selftest():
    ok = True

    # Bessel against published values.
    checks = [
        (0, 0.0, 1.0),
        (1, 1.0, 0.4400505857449335),
        (0, 1.0, 0.7651976865579666),
        (2, 5.0, 0.0465651162777522),
    ]
    worst = 0.0
    for m, x, want in checks:
        got = float(besselj(m, np.array([x]))[0])
        worst = max(worst, abs(got - want))
    print("special: besselj worst abs err vs published = %.2e" % worst)
    ok = ok and worst < 1e-13

    # The node-count cliff is real: too few nodes is badly wrong, and the
    # rule's count is converged (doubling it changes nothing).
    bad = float(besselj(0, np.array([60.0]), nodes=16)[0])
    good = float(besselj(0, np.array([60.0]))[0])
    conv = float(besselj(0, np.array([60.0]),
                         nodes=2 * bessel_nodes(60.0, 0))[0])
    print("special: J0(60): 16 nodes -> %.4f, rule (%d nodes) -> %.12f, "
          "doubled -> %.12f" % (bad, bessel_nodes(60.0, 0), good, conv))
    ok = ok and abs(good - conv) < 1e-14        # the rule is converged
    ok = ok and abs(bad - good) > 1e-3          # and 16 nodes is not

    # Zeros against published values.
    z_checks = [((0, 1), 2.404825557695773), ((0, 2), 5.520078110286311),
                ((1, 1), 3.831705970207512), ((2, 3), 11.619841172149059)]
    worstz = 0.0
    for (m, n), want in z_checks:
        worstz = max(worstz, abs(bessel_zero(m, n) - want))
    print("special: bessel_zero worst abs err = %.2e" % worstz)
    ok = ok and worstz < 1e-10
    # ...and they really are zeros.
    zz = bessel_zeros(3, 5)
    resid = float(np.abs(besselj(3, zz)).max())
    print("special: |J3(j_{3,n})| max over 5 zeros = %.2e" % resid)
    ok = ok and resid < 1e-12

    # Zernike: R_n^n = rho^n, and the recurrence stays accurate at high order.
    rho = np.linspace(0.0, 1.0, 51)
    ok = ok and np.abs(zernike_radial(6, 6, rho) - rho ** 6).max() < 1e-13
    # R_n^m(1) = 1 for every valid (n, m) -- a strong structural check.
    one = max(abs(float(zernike_radial(n, m, np.array([1.0]))[0]) - 1.0)
              for n in range(0, 41) for m in range(0, n + 1)
              if (n - m) % 2 == 0)
    print("special: max |R_n^m(1) - 1| over n<=40 = %.2e" % one)
    ok = ok and one < 1e-9
    # Orthogonality: int_0^1 R_n^m R_k^m rho drho = delta / (2n+2).
    r = np.linspace(0.0, 1.0, 20001)
    a = zernike_radial(6, 2, r)
    b = zernike_radial(8, 2, r)
    cross = float(np.trapezoid(a * b * r, r))
    self_ = float(np.trapezoid(a * a * r, r))
    print("special: Zernike <R6,2|R8,2>=%.2e, <R6,2|R6,2>=%.6f (want %.6f)"
          % (cross, self_, 1.0 / 14.0))
    ok = ok and abs(cross) < 1e-6 and abs(self_ - 1.0 / 14.0) < 1e-6

    # Hermite / Laguerre against closed forms.
    x = np.linspace(-2.0, 2.0, 41)
    ok = ok and np.abs(hermite(3, x) - (8 * x ** 3 - 12 * x)).max() < 1e-10
    ok = ok and np.abs(laguerre(2, 0.0, x)
                       - (1.0 - 2.0 * x + 0.5 * x * x)).max() < 1e-10
    # Hermite functions stay normalised where the raw polynomial overflows.
    xx = np.linspace(-14.0, 14.0, 40001)
    nrm = float(np.trapezoid(hermite_function(30, xx) ** 2, xx))
    print("special: ||psi_30||^2 = %.9f (want 1)" % nrm)
    ok = ok and abs(nrm - 1.0) < 1e-6

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
