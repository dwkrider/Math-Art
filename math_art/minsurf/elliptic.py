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
# Carlson's symmetric integral R_F additionally accepts COMPLEX
# arguments (principal square roots throughout; arguments must avoid the
# negative real axis), and `elliptic_f` builds the incomplete integral
# of the first kind F(phi | m) on it for complex phi and complex m --
# which is what the toroidal Karcher-Scherk family needs to trim its
# ends.
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
#   Carlson symmetric integrals (complex R_F, incomplete F):
#   B. C. Carlson, "Computing elliptic integrals by duplication",
#   Numer. Math. 33 (1979), 1-16; B. C. Carlson, "Numerical computation
#   of real or complex elliptic integrals", Numer. Algorithms 10 (1995),
#   13-26; DLMF 19.36 (the duplication method).

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


# --------------------------------------------------------------------------
# Carlson symmetric forms -- for the third-kind integral past n = 1
# --------------------------------------------------------------------------
# The Gauss-Legendre `ellippi` below is exact and fast while n < 1, but at
# n >= 1 the integrand 1/((1 - n sin^2 t) sqrt(1 - m sin^2 t)) has a pole
# inside the range and the integral is a Cauchy PRINCIPAL VALUE, which no
# amount of quadrature refinement will produce.  Carlson's R_J takes its
# fourth argument negative precisely for that case and returns the
# principal value by construction, so the two together cover the whole
# parameter range.
#
# That case is not academic here: a bubbleton on a NODOID has
# characteristic n = 6.43 at necksize -0.4 (see
# math_art/bubbleton_generator.py), so without the principal value the
# nodoid half of the bubbleton family is unreachable.
#
# `carlson_rf` below additionally accepts COMPLEX arguments (scalars or
# ndarrays), which is what `elliptic_f` -- the incomplete first-kind
# integral at complex amplitude and parameter -- is built on.  That pair
# exists for the toroidal Karcher-Scherk family, whose end-trimming
# needs F(phi, m) off the real axis.
#
# References:
#   B. C. Carlson, "Computing elliptic integrals by duplication",
#   Numer. Math. 33 (1979), 1-16 (the duplication algorithm and the
#   fifth-order series in the symmetric deviations).
#   B. C. Carlson, "Numerical computation of real or complex elliptic
#   integrals", Numer. Algorithms 10 (1995), 13-26 (complex arguments,
#   choice of square-root branch, error bounds).
#   DLMF 19.36 (the duplication method in reference form); code layout
#   of the real R_C/R_J duplications as in Press et al., Numerical
#   Recipes, Sect. 6.11.

_RC_ERRTOL = 1.2e-3
_RJ_ERRTOL = 1.5e-3
_RF_MAXITER = 100


def carlson_rc(x, y):
    """Degenerate symmetric integral R_C(x, y) = R_F(x, y, y).
    Handles y < 0 by the principal value."""
    if y > 0.0:
        xt, yt, w = x, y, 1.0
    else:                               # principal value for y < 0
        xt, yt = x - y, -y
        w = math.sqrt(x) / math.sqrt(xt) if x > 0.0 else 0.0
    while True:
        alamb = 2.0 * math.sqrt(xt) * math.sqrt(yt) + yt
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        ave = (xt + yt + yt) / 3.0
        sfac = (yt - ave) / ave
        if abs(sfac) <= _RC_ERRTOL:
            break
    poly = 1.0 + sfac * sfac * (0.3 + sfac * (1.0 / 7.0
                                              + sfac * (0.375
                                                        + sfac * 9.0 / 22.0)))
    return w * poly / math.sqrt(ave)


def carlson_rf(x, y, z, rtol=1e-16):
    """Symmetric elliptic integral of the FIRST kind,

        R_F(x, y, z) = (1/2) int_0^inf dt / sqrt((t+x)(t+y)(t+z)),

    for real or COMPLEX x, y, z -- scalars or ndarrays, broadcast
    elementwise.  At most one argument may be zero, and none may lie on
    the negative real axis.

    Branch convention (the load-bearing choice): every square root taken
    here -- the three roots inside each duplication step and the final
    1/sqrt(mean) -- is the PRINCIPAL branch, cut along (-inf, 0), with
    Re sqrt >= 0 (numpy sends a negative real with +0 imaginary part to
    +i sqrt|.|).  Carlson (1995, Sect. 2) shows that for arguments in
    the plane cut along the negative real axis this choice keeps the
    duplication iterates in the cut plane and makes the algorithm
    converge to the analytic continuation of the real integral -- the
    principal branch of R_F, homogeneous of degree -1/2 with the
    principal k**(-1/2).  Mixing branches between the three roots (or
    using a root with Re < 0) lands on a different sheet, which is why
    the rule is applied uniformly.  An argument exactly ON the cut is
    ambiguous -- numpy's +0 imaginary part silently picks the upper
    side -- so callers must keep arguments off it.

    Algorithm (DLMF 19.36; Carlson 1979): repeatedly form
    lambda = sqrt(x)sqrt(y) + sqrt(y)sqrt(z) + sqrt(z)sqrt(x) and
    replace (x, y, z) by ((x+lambda)/4, (y+lambda)/4, (z+lambda)/4) --
    the duplication theorem leaves R_F invariant while shrinking the
    arguments' relative deviations from their mean by a factor of ~4
    per step -- then finish with the fifth-order Taylor series in the
    elementary symmetric functions e2, e3 of those deviations.  The
    series truncation error is < eps**6 / (4 (1 - eps)) with eps the
    largest deviation (Carlson 1995), so iterating until
    eps <= (4 rtol)**(1/6) yields relative error ~rtol (default: full
    double precision).  Real nonnegative inputs return plain floats or
    float arrays; scalar inputs return a scalar."""
    want_complex = (np.iscomplexobj(np.asarray(x))
                    or np.iscomplexobj(np.asarray(y))
                    or np.iscomplexobj(np.asarray(z)))
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0 and np.ndim(z) == 0
    xt, yt, zt = np.broadcast_arrays(np.asarray(x, dtype=complex),
                                     np.asarray(y, dtype=complex),
                                     np.asarray(z, dtype=complex))
    errtol = (4.0 * float(rtol)) ** (1.0 / 6.0)
    for _ in range(_RF_MAXITER):
        ave = (xt + yt + zt) / 3.0
        dx = (ave - xt) / ave
        dy = (ave - yt) / ave
        dz = (ave - zt) / ave
        if max(float(np.max(np.abs(dx))), float(np.max(np.abs(dy))),
               float(np.max(np.abs(dz)))) <= errtol:
            break
        sx, sy, sz = np.sqrt(xt), np.sqrt(yt), np.sqrt(zt)
        lam = sx * sy + sy * sz + sz * sx
        xt = 0.25 * (xt + lam)
        yt = 0.25 * (yt + lam)
        zt = 0.25 * (zt + lam)
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    res = np.asarray((1.0 + (e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2
                      + e3 / 14.0) / np.sqrt(ave))
    if not want_complex and np.all(np.abs(res.imag)
                                   <= 1e-10 * (1.0 + np.abs(res.real))):
        res = res.real
    return res.item() if scalar else res


def carlson_rj(x, y, z, p):
    """Symmetric elliptic integral of the THIRD kind,
    R_J = (3/2) int_0^inf dt / ((t+p) sqrt((t+x)(t+y)(t+z))).

    p < 0 returns the Cauchy principal value -- that branch is the whole
    point of using Carlson here."""
    a = b = rcx = 0.0
    if p > 0.0:
        xt, yt, zt, pt = x, y, z, p
    else:
        xt, zt = min(x, y, z), max(x, y, z)
        yt = x + y + z - xt - zt
        a = 1.0 / (yt - p)
        b = a * (zt - yt) * (yt - xt)
        pt = yt + b
        rho = xt * zt / yt
        tau = p * pt / yt
        rcx = carlson_rc(rho, tau)
    total, fac = 0.0, 1.0
    while True:
        sx, sy, sz = math.sqrt(xt), math.sqrt(yt), math.sqrt(zt)
        alamb = sx * (sy + sz) + sy * sz
        alpha = (pt * (sx + sy + sz) + sx * sy * sz) ** 2
        beta = pt * (pt + alamb) ** 2
        total += fac * carlson_rc(alpha, beta)
        fac *= 0.25
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        zt = 0.25 * (zt + alamb)
        pt = 0.25 * (pt + alamb)
        ave = 0.2 * (xt + yt + zt + pt + pt)
        dx = (ave - xt) / ave
        dy = (ave - yt) / ave
        dz = (ave - zt) / ave
        dp = (ave - pt) / ave
        if max(abs(dx), abs(dy), abs(dz), abs(dp)) <= _RJ_ERRTOL:
            break
    c1, c2, c3, c4 = 3.0 / 14.0, 1.0 / 3.0, 3.0 / 22.0, 3.0 / 26.0
    c5, c6, c7, c8 = 0.75 * c3, 1.5 * c4, 0.5 * c2, c3 + c3
    ea = dx * (dy + dz) + dy * dz
    eb = dx * dy * dz
    ec = dp * dp
    ed = ea - 3.0 * ec
    ee = eb + 2.0 * dp * (ea - ec)
    ans = 3.0 * total + fac * (
        1.0 + ed * (-c1 + c5 * ed - c6 * ee)
        + eb * (c7 + dp * (-c8 + dp * c4))
        + dp * ea * (c2 - dp * c3) - c2 * dp * ec) / (ave * math.sqrt(ave))
    if p <= 0.0:
        ans = a * (b * ans + 3.0 * (rcx - carlson_rf(xt, yt, zt)))
    return ans


def _ellippi_quarter(n, phi, m):
    """Pi(n; phi | m) for a single phi in [0, pi/2], via Carlson."""
    if phi <= 0.0:
        return 0.0
    sp = math.sin(phi)
    cp2 = math.cos(phi) ** 2
    q = 1.0 - m * sp * sp
    p4 = 1.0 - n * sp * sp
    return (sp * carlson_rf(cp2, q, 1.0)
            + (n / 3.0) * sp ** 3 * carlson_rj(cp2, q, 1.0, p4))


def ellippi_pv(n, phi, m):
    """Pi(n; phi | m) valid for ANY real n, including n >= 1 where the
    integral is a Cauchy principal value.  Scalar or array `phi`.

    Carlson's formula is stated for an amplitude in [0, pi/2], so a
    general phi is folded down first.  The integrand is pi-periodic and
    symmetric about pi/2, giving
        Pi(phi + k pi) = Pi(phi) + 2 k Pi_complete ,
        Pi(pi - phi)   = 2 Pi_complete - Pi(phi) ,
        Pi(-phi)       = -Pi(phi) ,
    and folding is what makes this usable on the long, monotonically
    growing amplitudes that come out of `jacobi_am`."""
    if m >= 1.0:
        raise ValueError(f"ellippi_pv: m = {m} is out of range")
    ph = np.asarray(phi, dtype=float)
    scalar = (ph.ndim == 0)
    flat = np.atleast_1d(ph).ravel()
    complete = _ellippi_quarter(n, 0.5 * math.pi, m)
    out = np.empty_like(flat)
    for i, val in enumerate(flat):
        sgn = 1.0
        t = float(val)
        if t < 0.0:
            sgn, t = -1.0, -t
        k = math.floor(t / math.pi)
        t -= k * math.pi                       # t now in [0, pi)
        if t <= 0.5 * math.pi:
            base = _ellippi_quarter(n, t, m)
        else:
            base = 2.0 * complete - _ellippi_quarter(n, math.pi - t, m)
        out[i] = sgn * (2.0 * k * complete + base)
    res = out.reshape(np.atleast_1d(ph).shape)
    return float(res[0]) if scalar else res


def ellippi(n, phi, m, segments=None):
    """Incomplete elliptic integral of the THIRD kind,

        Pi(n; phi | m) = int_0^phi dt / ((1 - n sin^2 t) sqrt(1 - m sin^2 t))

    for real n < 1, real m < 1 and real phi (scalar or ndarray).

    Evaluated by composite 48-point Gauss-Legendre over sub-intervals of
    length <= pi/2.  The integrand is analytic away from n sin^2 t = 1 and
    m sin^2 t = 1, so Gauss-Legendre converges spectrally and 48 nodes per
    half-period is far past machine precision; the subdivision is only
    there to stop a long phi from making one panel span many oscillations.

    For n >= 1 the integrand has a pole inside the range and the
    integral is a Cauchy principal value, which no refinement of a
    quadrature can produce; those calls are delegated to `ellippi_pv`,
    which gets it from Carlson's R_J.  The Gauss-Legendre route is kept
    for n < 1 because it is faster and vectorised."""
    if n >= 1.0:
        # a pole sits inside the range; hand over to the Carlson route,
        # which returns the Cauchy principal value by construction
        return ellippi_pv(n, phi, m)
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


def elliptic_f(phi, m):
    """Incomplete elliptic integral of the FIRST kind,

        F(phi | m) = int_0^phi dtheta / sqrt(1 - m sin^2 theta),

    for real or COMPLEX amplitude phi and parameter m = k^2 -- scalars
    or ndarrays, broadcast elementwise.  F(pi/2 | m) = K(m) = ellipk(m).

    Evaluated through Carlson's symmetric form (DLMF 19.25.5),

        F(phi | m) = sin(phi) R_F(cos^2 phi, 1 - m sin^2 phi, 1),

    which represents the principal branch of F on the strip
    |Re phi| <= pi/2.  A general amplitude is first folded into that
    strip by the quasi-period relation F(phi + k pi | m) =
    F(phi | m) + 2 k K(m) (DLMF 19.2.10), with the complete integral
    K(m) = R_F(0, 1 - m, 1) taken from the same complex-capable code,
    so long real amplitudes and complex m work together.

    Branch caveats are inherited from `carlson_rf` (principal square
    roots, arguments off the negative real axis): here that means
    1 - m sin^2 phi must stay off (-inf, 0].  For real phi and real
    m in [0, 1) it always does; a real m with m sin^2 phi > 1 lands
    exactly on the cut and numpy's +0 imaginary part silently picks the
    upper side, so pass such parameters explicitly complex if a
    particular side is intended (the result is then returned complex).
    Real inputs with real results come back as plain floats/arrays."""
    want_complex = (np.iscomplexobj(np.asarray(phi))
                    or np.iscomplexobj(np.asarray(m)))
    scalar = np.ndim(phi) == 0 and np.ndim(m) == 0
    ph = np.asarray(phi, dtype=complex)
    mm = np.asarray(m, dtype=complex)
    k = np.round(ph.real / math.pi)          # fold Re phi into +-pi/2
    ph0 = ph - math.pi * k
    s, c = np.sin(ph0), np.cos(ph0)
    res = s * np.asarray(carlson_rf(c * c, 1.0 - mm * s * s, 1.0),
                         dtype=complex)
    if np.any(k != 0.0):
        K = np.asarray(carlson_rf(0.0, 1.0 - mm, 1.0), dtype=complex)
        res = res + (2.0 * k) * K
    res = np.asarray(res)
    if not want_complex and np.all(np.abs(res.imag)
                                   <= 1e-10 * (1.0 + np.abs(res.real))):
        res = res.real
    return res.item() if scalar else res


# ==========================================================================
# Complex gamma and the Gauss hypergeometric function 2F1 (numpy only)
# ==========================================================================
# Added for the CMC-1 trinoids (math_art/trinoid.py): the canonical
# solutions of their Fuchsian system are built from 2F1, and the
# connection matrices and the unitarising constant r involve gamma
# ratios.  Blender ships numpy but not scipy, so both live here.
#
# References:
#   C. Lanczos, "A precision approximation of the gamma function",
#   J. SIAM Numer. Anal. B 1 (1964), 86-96 (the g = 7, n = 9 coefficient
#   set popularised by Numerical Recipes / Boost).
#   Gauss 2F1 series and its connection formulas: DLMF 15.2.1 (series),
#   15.8.1 (Pfaff), 15.8.2 (z -> 1/z), 15.8.4 (z -> 1-z); also
#   Abramowitz & Stegun 15.3.4-15.3.9.

_LANCZOS_G = 7.0
_LANCZOS_C = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
              771.32342877765313, -176.61502916214059, 12.507343278686905,
              -0.13857109526572012, 9.9843695780195716e-6,
              1.5056327351493116e-7)


def cgamma(z):
    """Gamma(z) for complex scalar or ndarray z, by the Lanczos
    approximation (g = 7, 9 terms) with the reflection formula for
    Re z < 1/2.  Relative accuracy ~1e-13 away from the poles.  Validated
    in the self-test against the duplication formula, |Gamma(1+iy)|^2 =
    pi y / sinh(pi y), and integer factorials -- none of which share the
    Lanczos algebra."""
    z = np.asarray(z, dtype=complex)
    refl = z.real < 0.5
    zz = np.where(refl, 1.0 - z, z)
    x = np.full_like(zz, _LANCZOS_C[0])
    for k, ck in enumerate(_LANCZOS_C[1:], start=1):
        x = x + ck / (zz - 1.0 + k)
    t = zz - 0.5 + _LANCZOS_G
    g = math.sqrt(2.0 * math.pi) * t ** (zz - 0.5) * np.exp(-t) * x
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(refl, math.pi / (np.sin(math.pi * z) * g), g)
    return out


def rcgamma(z):
    """1 / Gamma(z), which unlike Gamma itself is ENTIRE: it vanishes at
    the non-positive integers instead of blowing up.  Used for the
    denominators of the 2F1 connection coefficients, where a pole of a
    denominator gamma legitimately means 'this coefficient is zero' and
    must not produce inf/nan."""
    z = np.asarray(z, dtype=complex)
    lower = z.real < 0.5
    # 1/Gamma(z) = sin(pi z) Gamma(1-z) / pi  where Gamma(z) may have
    # poles; plain reciprocal where it cannot (Re z >= 1/2).
    return np.where(lower,
                    np.sin(math.pi * z) * cgamma(1.0 - z) / math.pi,
                    1.0 / cgamma(np.where(lower, 1.0, z)))


_HYP_MAXTERMS = 8000
_HYP_TOL = 1e-15


def _hyp_series(a, b, c, w):
    """The defining series sum (a)_n (b)_n / ((c)_n n!) w^n over a flat
    complex ndarray w with |w| < 1.  Terminates when every term is below
    _HYP_TOL relative to its partial sum."""
    s = np.ones_like(w)
    t = np.ones_like(w)
    n = 0
    while n < _HYP_MAXTERMS:
        t = t * ((a + n) * (b + n) / ((c + n) * (n + 1.0))) * w
        s = s + t
        n += 1
        if n < 48 or n % 16 == 0:
            if np.all(np.abs(t) <= _HYP_TOL * np.abs(s)):
                return s
    raise RuntimeError(
        f"hyp2f1 series did not converge (max |w| = {np.abs(w).max():.6f});"
        " the argument is too close to exp(+-i pi/3), where all Kummer"
        " transformations degenerate")


def _hyp_direct(a, b, c, z):
    """2F1 on the region covered by the series and the Pfaff
    transformation F(a,b;c;z) = (1-z)^(-a) F(a, c-b; c; z/(z-1))
    (DLMF 15.8.1); per point, whichever argument is smaller."""
    out = np.empty_like(z)
    wp = z / (z - 1.0)
    m = np.abs(wp) < np.abs(z)
    if np.any(~m):
        out[~m] = _hyp_series(a, b, c, z[~m])
    if np.any(m):
        out[m] = (1.0 - z[m]) ** (-a) * _hyp_series(a, c - b, c, wp[m])
    return out


def _near_int(x, tol=1e-8):
    x = complex(x)
    return abs(x.imag) < tol and abs(x.real - round(x.real)) < tol


def hyp2f1(a, b, c, z):
    """Gauss hypergeometric function 2F1(a, b; c; z) for complex scalar
    parameters and complex scalar-or-ndarray z, on the principal branch
    (cut along [1, infinity)).

    Algorithm: per point, the argument is moved into the unit disk by
    whichever of the Kummer transformations gives the smallest modulus --
    the identity (series), Pfaff z/(z-1) (DLMF 15.8.1), the z -> 1-z
    connection (DLMF 15.8.4), or the z -> 1/z connection (DLMF 15.8.2),
    the last two combined with Pfaff on their sub-series.  Together these
    reach every z except a neighbourhood of the two points
    exp(+-i pi/3), where ALL six Kummer arguments have modulus 1 and the
    series is refused rather than returned half-converged.

    Limitations (raised, not mis-answered): c must not be a non-positive
    integer; the z -> 1-z route needs c-a-b non-integer and the
    z -> 1/z route needs a-b non-integer (the log-degenerate cases of
    DLMF 15.8.8/15.8.10 are not implemented).  When the best route is
    degenerate the next-best non-degenerate route is used if its
    argument still lies inside the disk.

    Validated in the self-test against closed forms ((1-z)^(-a),
    -log(1-z)/z, arcsin(z)/z), Gauss's theorem at z = 1, the Euler
    transformation, the hypergeometric ODE itself at scattered points in
    every region, and route-against-route agreement across the region
    boundaries."""
    a, b, c = complex(a), complex(b), complex(c)
    if _near_int(c) and c.real <= 0.5:
        raise ValueError(f"hyp2f1: c = {c} is a non-positive integer")
    z = np.asarray(z, dtype=complex)
    scalar = (z.ndim == 0)
    flat = np.atleast_1d(z).ravel().copy()
    out = np.empty_like(flat)

    with np.errstate(divide='ignore', invalid='ignore'):
        m_dir = np.minimum(np.abs(flat), np.abs(flat / (flat - 1.0)))
        m_1mz = np.minimum(np.abs(1.0 - flat), np.abs(1.0 - 1.0 / flat))
        m_inv = np.minimum(np.abs(1.0 / flat), np.abs(1.0 / (1.0 - flat)))
    m_1mz = np.where(np.isfinite(m_1mz), m_1mz, np.inf)
    m_inv = np.where(np.isfinite(m_inv), m_inv, np.inf)
    deg_1mz = _near_int(c - a - b)
    deg_inv = _near_int(a - b)
    if deg_1mz:
        m_1mz = np.full_like(m_1mz, np.inf)
    if deg_inv:
        m_inv = np.full_like(m_inv, np.inf)
    route = np.argmin(np.stack([m_dir, m_1mz, m_inv]), axis=0)

    sel = route == 0
    if np.any(sel):
        out[sel] = _hyp_direct(a, b, c, flat[sel])

    sel = route == 1                        # z -> 1-z  (DLMF 15.8.4)
    if np.any(sel):
        w = 1.0 - flat[sel]
        cab = c - a - b
        # denominator gammas via rcgamma: a pole there means the
        # coefficient is zero, not inf
        k1 = complex(cgamma(c) * cgamma(cab)
                     * rcgamma(c - a) * rcgamma(c - b))
        k2 = complex(cgamma(c) * cgamma(-cab) * rcgamma(a) * rcgamma(b))
        out[sel] = (k1 * _hyp_direct(a, b, a + b - c + 1.0, w)
                    + k2 * w ** cab
                    * _hyp_direct(c - a, c - b, cab + 1.0, w))

    sel = route == 2                        # z -> 1/z  (DLMF 15.8.2)
    if np.any(sel):
        zz = flat[sel]
        w = 1.0 / zz
        k1 = complex(cgamma(c) * cgamma(b - a)
                     * rcgamma(b) * rcgamma(c - a))
        k2 = complex(cgamma(c) * cgamma(a - b)
                     * rcgamma(a) * rcgamma(c - b))
        out[sel] = (k1 * (-zz) ** (-a)
                    * _hyp_direct(a, a - c + 1.0, a - b + 1.0, w)
                    + k2 * (-zz) ** (-b)
                    * _hyp_direct(b, b - c + 1.0, b - a + 1.0, w))

    res = out.reshape(np.atleast_1d(z).shape)
    return complex(res.ravel()[0]) if scalar else res


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
        self.t1p0 = t1_0            # theta1'(0), needed by sigma

    def zeta(self, z):
        z = np.asarray(z, dtype=complex)
        t0, t1, _, _ = _theta1_series(self.c * z, self.q)
        return (self.eta1 / self.w1) * z + self.c * (t1 / t0)

    def wp(self, z):
        z = np.asarray(z, dtype=complex)
        t0, t1, t2, _ = _theta1_series(self.c * z, self.q)
        r1 = t1 / t0
        return -self.eta1 / self.w1 - self.c ** 2 * (t2 / t0 - r1 ** 2)

    def sigma(self, z):
        """Weierstrass sigma, from the same theta series:

            sigma(z) = (2 w1 / pi) exp(eta1 z^2 / (2 w1))
                       theta1(pi z / (2 w1)) / theta1'(0) .

        Normalised so that sigma(z)/z -> 1 as z -> 0 -- the standard
        convention, and the one Heller's closed-form elastic curve
        assumes (his paper prints the limit at infinity, which is a
        typo).  Unlike P and zeta, sigma is NOT elliptic; it is only
        quasi-periodic, which is exactly why it can build a curve that
        does not close until a period condition is imposed."""
        z = np.asarray(z, dtype=complex)
        t0, _, _, _ = _theta1_series(self.c * z, self.q)
        return ((2.0 * self.w1 / math.pi)
                * np.exp(self.eta1 * z * z / (2.0 * self.w1))
                * t0 / self.t1p0)

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
    ok &= _selftest_carlson()
    ok &= _selftest_sigma()
    ok &= _selftest_hyp2f1()

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

    # 6) Carlson agrees with Gauss-Legendre wherever BOTH are valid.
    #    Two independent algorithms -- duplication theorem against
    #    spectral quadrature -- so agreement is real evidence.
    worst = 0.0
    for n in (-3.0, -0.5, 0.0, 0.4, 0.9):
        for m in (0.0, 0.3, 0.85):
            phi = np.linspace(-7.0, 7.0, 25)
            a = ellippi(n, phi, m)
            b = ellippi_pv(n, phi, m)
            worst = max(worst, float(np.abs(a - b).max()
                                     / max(1.0, np.abs(a).max())))
    good = worst < 1e-9
    ok &= good
    print(f"ellippi: Carlson vs Gauss-Legendre, max relative "
          f"{worst:.2e} {'OK' if good else 'FAIL'}")

    # 7) The principal-value branch (n >= 1) against its own closed form
    #    at m = 0, where
    #    Pi(n; phi|0) = atanh(sqrt(n-1) tan phi)/sqrt(n-1) for n > 1.
    worst = 0.0
    for n in (1.7, 4.0, 9.0):
        rt = math.sqrt(n - 1.0)
        phi = np.linspace(0.05, 1.4, 30)
        got = ellippi_pv(n, phi, 0.0)
        want = np.arctanh(np.clip(rt * np.tan(phi), -0.999999999,
                                  0.999999999)) / rt
        # only compare below the pole, where the principal value and the
        # elementary form describe the same branch
        keep = rt * np.tan(phi) < 0.98
        worst = max(worst, float(np.abs(got[keep] - want[keep]).max()))
    good = worst < 1e-9
    ok &= good
    print(f"ellippi: principal value vs atanh form (n > 1, m = 0) max "
          f"dev {worst:.2e} {'OK' if good else 'FAIL'}")

    # 8) The fold is consistent: Pi(phi + k pi) = Pi(phi) + 2 k Pi_c,
    #    which is what makes long amplitudes usable.
    n, m = 3.5, 0.6
    pc = ellippi_pv(n, math.pi / 2.0, m)
    base = ellippi_pv(n, 0.7, m)
    dev = max(abs(float(ellippi_pv(n, 0.7 + k * math.pi, m))
                  - (base + 2.0 * k * pc)) for k in (1, 2, 5))
    good = dev < 1e-9
    ok &= good
    print(f"ellippi: pi-fold consistency (n = 3.5 > 1) dev {dev:.1e} "
          f"{'OK' if good else 'FAIL'}")

    return ok


def _selftest_carlson():
    """Complex-capable R_F and elliptic_f, against exact values, the
    defining symmetries, the AGM complete integral, the Gauss-Legendre
    first-kind path, and an independent quadrature of the defining
    integral along complex amplitudes."""
    ok = True

    # 1) Normalisation: R_F(1,1,1) = 1 EXACTLY -- zero deviations mean
    #    the series is 1 and sqrt(1) is exact, so == is fair here.
    v = carlson_rf(1.0, 1.0, 1.0)
    good = (v == 1.0)
    ok &= good
    print(f"carlson: R_F(1,1,1)={v!r} (exact 1.0) "
          f"{'OK' if good else 'FAIL'}")

    # 2) R_F(x,x,x) = x^(-1/2) on the principal branch, real and complex.
    worst = 0.0
    for xv in (0.25, 2.0, 7.5, 0.3 + 0.4j, 2.0 - 1.0j, -1.0 + 2.0j):
        worst = max(worst, abs(carlson_rf(xv, xv, xv)
                               - 1.0 / np.sqrt(complex(xv))))
    good = worst < 1e-14
    ok &= good
    print(f"carlson: R_F(x,x,x) vs x^-1/2 (real+complex) max dev "
          f"{worst:.1e} {'OK' if good else 'FAIL'}")

    # 3) Full permutation symmetry, including a zero argument.
    worst = 0.0
    for (a, b, c) in ((0.31 + 0.22j, 1.7, 2.3 - 0.6j),
                      (0.0, 1.2 + 0.5j, 2.0)):
        vals = [carlson_rf(*p) for p in
                ((a, b, c), (a, c, b), (b, a, c),
                 (b, c, a), (c, a, b), (c, b, a))]
        worst = max(worst, max(abs(vv - vals[0]) for vv in vals))
    good = worst < 1e-13
    ok &= good
    print(f"carlson: permutation symmetry max dev {worst:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # 4) Homogeneity R_F(kx,ky,kz) = R_F(x,y,z)/sqrt(k) -- valid on the
    #    principal branches while the scaled arguments stay in the right
    #    half plane, which these do.
    x0, y0, z0 = 0.4 + 0.3j, 1.1, 2.2 - 0.7j
    base = carlson_rf(x0, y0, z0)
    worst = 0.0
    for kv in (4.0, 0.5, 2.0 + 1.5j, 0.3 - 0.2j):
        worst = max(worst, abs(carlson_rf(kv * x0, kv * y0, kv * z0)
                               - base / np.sqrt(complex(kv))))
    good = worst < 1e-13
    ok &= good
    print(f"carlson: homogeneity max dev {worst:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # 5) elliptic_f closes on the AGM: F(pi/2 | m) = K(m).
    worst = 0.0
    for m in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95):
        worst = max(worst, abs(elliptic_f(0.5 * math.pi, m) - ellipk(m)))
    good = worst < 5e-14
    ok &= good
    print(f"carlson: F(pi/2|m) vs AGM K(m), m in [0,0.95], max dev "
          f"{worst:.1e} {'OK' if good else 'FAIL'}")

    # 6) Real path incl. the pi-fold, against the Gauss-Legendre
    #    first-kind quadrature (n = 0 third kind) -- two independent
    #    algorithms over amplitudes well past pi/2.
    phi = np.linspace(-4.0, 4.0, 41)
    worst = 0.0
    for m in (0.2, 0.65):
        worst = max(worst, float(np.max(np.abs(
            elliptic_f(phi, m) - ellippi(0.0, phi, m)))))
    good = worst < 1e-12
    ok &= good
    print(f"carlson: F(phi|m) vs Gauss-Legendre over phi in [-4,4] max "
          f"dev {worst:.1e} {'OK' if good else 'FAIL'}")

    # 7) The COMPLEX path: elliptic_f against a direct Gauss-Legendre
    #    quadrature of the defining integral along the straight segment
    #    from 0 to phi (theta = phi*t, t in [0,1]) -- fully independent
    #    of the duplication.  The integrand is analytic near the path
    #    for these (phi, m), so the quadrature reference is itself
    #    converged to rounding (96- vs 160-node agreement is checked).
    xg, wg = _leggauss(96)
    x2, w2 = _leggauss(160)
    worst, qconv = 0.0, 0.0
    for phv, mv in ((0.7 + 0.4j, 0.35 + 0.20j),
                    (1.1 - 0.3j, 0.60 - 0.25j),
                    (0.4 + 0.9j, -0.50 + 0.30j),
                    (2.2 + 0.3j, 0.30 + 0.10j)):   # Re > pi/2: pi-fold
        def q(xn, wn):
            t = 0.5 * (xn + 1.0)
            f = 1.0 / np.sqrt(1.0 - mv * np.sin(phv * t) ** 2)
            return phv * 0.5 * np.sum(wn * f)
        ref, ref2 = q(xg, wg), q(x2, w2)
        qconv = max(qconv, abs(ref - ref2) / abs(ref))
        worst = max(worst, abs(elliptic_f(phv, mv) - ref) / abs(ref))
    good = worst < 1e-12 and qconv < 1e-13
    ok &= good
    print(f"carlson: F(phi|m) complex vs independent quadrature, max "
          f"rel {worst:.1e} (quadrature self-agreement {qconv:.1e}) "
          f"{'OK' if good else 'FAIL'}")

    # 8) Vectorised elementwise path agrees with per-scalar calls.
    ph_arr = np.array([0.7 + 0.4j, 1.1 - 0.3j, 0.4 + 0.9j, 2.2 + 0.3j])
    m_arr = np.array([0.35 + 0.20j, 0.60 - 0.25j, -0.50 + 0.30j,
                      0.30 + 0.10j])
    vec = elliptic_f(ph_arr, m_arr)
    worst = max(abs(vec[i] - elliptic_f(ph_arr[i], m_arr[i]))
                for i in range(len(ph_arr)))
    good = worst < 1e-13
    ok &= good
    print(f"carlson: elementwise array vs scalar calls max dev "
          f"{worst:.1e} {'OK' if good else 'FAIL'}")

    return ok


def _selftest_hyp2f1():
    """cgamma and hyp2f1, each against closed forms and identities that
    do NOT share the implementation."""
    ok = True

    # -- gamma --------------------------------------------------------
    # factorials, Gamma(1/2), and the closed form
    # |Gamma(1+iy)|^2 = pi y / sinh(pi y); none of these run through the
    # reflection/Lanczos path being checked in the same way.
    d1 = max(abs(complex(cgamma(n + 1)) - math.factorial(n))
             / math.factorial(n) for n in range(1, 11))
    d2 = abs(complex(cgamma(0.5)) - math.sqrt(math.pi))
    ys = np.array([0.3, 1.1, 2.7])
    d3 = float(np.abs(np.abs(cgamma(1.0 + 1j * ys)) ** 2
                      - math.pi * ys / np.sinh(math.pi * ys)).max())
    # duplication formula at complex points (tests both half-planes)
    zs = np.array([0.3 + 0.7j, -1.4 + 0.9j, 2.2 - 1.3j, -0.7 - 0.2j])
    dup = (cgamma(2.0 * zs) - 2.0 ** (2.0 * zs - 1.0) / math.sqrt(math.pi)
           * cgamma(zs) * cgamma(zs + 0.5))
    d4 = float((np.abs(dup) / np.abs(cgamma(2.0 * zs))).max())
    good = d1 < 1e-12 and d2 < 1e-13 and d3 < 1e-12 and d4 < 1e-11
    ok &= good
    print(f"cgamma: factorials {d1:.1e}, sqrt(pi) {d2:.1e}, "
          f"|G(1+iy)| {d3:.1e}, duplication {d4:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # -- hyp2f1 closed forms, spanning all three routes -----------------
    # 2F1(a, b; b; z) = (1-z)^(-a) holds on the whole cut plane; with
    # a - b and c - a - b non-integer it exercises direct, 1-z and 1/z.
    a, b = 0.31 + 0.14j, 1.62
    pts = np.array([0.3 + 0.2j, -0.7 + 0.1j, 0.9 + 0.05j, 0.97 - 0.4j,
                    -4.0 + 2.0j, 3.0 + 4.0j, 12.0 - 5.0j, -25.0 - 1.0j,
                    0.55 + 0.83j, 0.55 - 0.83j])   # last two near e^(i pi/3)
    got = hyp2f1(a, b, b, pts)
    want = (1.0 - pts) ** (-a)
    d1 = float((np.abs(got - want) / np.abs(want)).max())
    good = d1 < 1e-11
    ok &= good
    print(f"hyp2f1: (1-z)^(-a) closed form, all routes, max rel "
          f"{d1:.1e} {'OK' if good else 'FAIL'}")

    # 2F1(1,1;2;z) = -log(1-z)/z (principal log) -- the log-degenerate
    # parameter case, so only the direct/Pfaff region applies.
    pts = np.array([0.4 + 0.3j, -0.8 - 0.5j, 0.85 + 0.1j, -0.2 + 0.9j])
    got = hyp2f1(1.0, 1.0, 2.0, pts)
    want = -np.log(1.0 - pts) / pts
    d2 = float((np.abs(got - want) / np.abs(want)).max())
    # 2F1(1/2,1/2;3/2;z^2) = arcsin(z)/z
    zs = np.array([0.3, 0.55, 0.8 + 0.1j, 0.2 - 0.6j])
    got = hyp2f1(0.5, 0.5, 1.5, zs * zs)
    want = np.arcsin(zs) / zs
    d3 = float((np.abs(got - want) / np.abs(want)).max())
    good = d2 < 1e-12 and d3 < 1e-12
    ok &= good
    print(f"hyp2f1: -log(1-z)/z {d2:.1e}, arcsin(z)/z {d3:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # Gauss's theorem: 2F1(a,b;c;1) = G(c)G(c-a-b) / (G(c-a)G(c-b)),
    # summed DIRECTLY (tail ~ n^(a+b-c-1), convergent for c-a-b > 0) --
    # ties the series to cgamma with no connection formula involved.
    a, b, c = 0.3, 0.45, 2.2
    n = np.arange(400000, dtype=float)
    ratios = (a + n) * (b + n) / ((c + n) * (1.0 + n))
    direct = 1.0 + float(np.cumprod(ratios).sum())
    gauss = complex(cgamma(c) * cgamma(c - a - b)
                    / (cgamma(c - a) * cgamma(c - b))).real
    d4 = abs(direct - gauss) / gauss
    good = d4 < 1e-7                     # tail ~ N^-(c-a-b) = 5e-9
    ok &= good
    print(f"hyp2f1: Gauss theorem at z=1, series vs gamma ratio, rel "
          f"{d4:.1e} {'OK' if good else 'FAIL'}")

    # Euler transformation F(a,b;c;z) = (1-z)^(c-a-b) F(c-a,c-b;c;z):
    # different parameters, different series, same value.
    a, b, c = 0.27, 1.13, 1.71
    pts = np.array([0.35 + 0.45j, -0.6 - 0.3j, 0.92 + 0.02j, 2.5 + 1.5j])
    lhs = hyp2f1(a, b, c, pts)
    rhs = (1.0 - pts) ** (c - a - b) * hyp2f1(c - a, c - b, c, pts)
    d5 = float((np.abs(lhs - rhs) / np.abs(lhs)).max())
    good = d5 < 1e-11
    ok &= good
    print(f"hyp2f1: Euler transformation, max rel {d5:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # The hypergeometric ODE z(1-z)F'' + (c-(a+b+1)z)F' - abF = 0 at
    # scattered points in EVERY region -- independent of all the
    # identities used to build the evaluator.
    a, b, c = 0.42, 0.87 + 0.33j, 1.29
    worst = 0.0
    h = 1e-4      # balances FD truncation against roundoff in f''
    for z0 in (0.31 + 0.22j, -0.62 + 0.41j, 0.93 + 0.31j, 0.93 - 0.31j,
               1.62 + 0.35j, 1.62 - 0.35j, -3.1 + 1.2j, 4.2 - 2.2j):
        f0 = hyp2f1(a, b, c, z0)
        fp = hyp2f1(a, b, c, z0 + h)
        fm = hyp2f1(a, b, c, z0 - h)
        d1n = (fp - fm) / (2.0 * h)
        d2n = (fp - 2.0 * f0 + fm) / (h * h)
        resid = z0 * (1.0 - z0) * d2n + (c - (a + b + 1.0) * z0) * d1n \
            - a * b * f0
        scale = max(abs(a * b * f0), abs(d1n), 1.0)
        worst = max(worst, abs(resid) / scale)
    # FD-limited: roundoff in the second difference is ~4 eps/h^2 = 9e-8
    # amplified by |z(1-z)| up to ~25 at the far sample points; a wrong
    # route or branch would fail this at O(1).
    good = worst < 1e-5
    ok &= good
    print(f"hyp2f1: ODE residual over all regions, max {worst:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # Smoothness ACROSS the region boundaries, where the evaluator
    # switches routes: sample a radial triplet straddling |z| = 1 (the
    # direct <-> 1/z boundary) and |1-z| = |z| (direct <-> 1-z).  If a
    # route joins with the wrong branch or coefficient, the second
    # difference jumps to O(1); a correct join leaves only the smooth
    # F'' h^2 ~ 1e-6.  (Params must keep c-a-b and a-b non-integer, or
    # the connection routes are rightly disabled and the direct series
    # rightly refuses |z| -> 1.)
    a, b, c = 0.42, 0.79, 1.31
    h = 1e-3
    worst = 0.0
    for th in np.linspace(0.45, 2.7, 9):     # avoid z = 1 itself
        e = np.exp(1j * th)
        f = [hyp2f1(a, b, c, r * e) for r in (1.0 - h, 1.0, 1.0 + h)]
        worst = max(worst, abs(f[0] - 2.0 * f[1] + f[2]) / abs(f[1]))
    for y in np.linspace(0.35, 2.0, 7):      # the Re z = 1/2 boundary
        f = [hyp2f1(a, b, c, x + 1j * y) for x in (0.5 - h, 0.5, 0.5 + h)]
        worst = max(worst, abs(f[0] - 2.0 * f[1] + f[2]) / abs(f[1]))
    good = worst < 1e-4
    ok &= good
    print(f"hyp2f1: route-boundary smoothness (2nd diff), max "
          f"{worst:.1e} {'OK' if good else 'FAIL'}")

    return ok


def _selftest_sigma():
    """Weierstrass sigma, against the two identities that pin it."""
    ok = True
    for w1, tau in ((0.5, 1j), (0.5, 0.8j), (0.7, 1.6j)):
        L = _Lattice(w1, tau)
        z = np.array([0.07 + 0.03j, 0.13 - 0.09j, 0.21 + 0.17j])

        # sigma'/sigma = zeta -- the defining relation
        h = 1e-6
        dlog = (np.log(L.sigma(z + h)) - np.log(L.sigma(z - h))) / (2 * h)
        d1 = float(np.abs(dlog - L.zeta(z)).max())

        # normalisation sigma(z)/z -> 1 as z -> 0 (NOT as z -> infinity,
        # which is what Heller's paper prints; that is a typo there)
        t = np.array([1e-5, 3e-5, 1e-4])
        d2 = float(np.abs(L.sigma(t) / t - 1.0).max())

        # quasi-periodicity sigma(z + 2w1) = -exp(2 eta1 (z + w1)) sigma(z)
        lhs = L.sigma(z + 2.0 * w1)
        rhs = -np.exp(2.0 * L.eta1 * (z + w1)) * L.sigma(z)
        d3 = float(np.abs(lhs - rhs).max())

        good = d1 < 1e-7 and d2 < 1e-9 and d3 < 1e-12
        ok &= good
        print(f"sigma w1={w1} tau={tau}: |sigma'/sigma - zeta|={d1:.1e}, "
              f"|sigma(z)/z - 1|={d2:.1e}, quasi-period {d3:.1e} "
              f"{'OK' if good else 'FAIL'}")
    return ok
