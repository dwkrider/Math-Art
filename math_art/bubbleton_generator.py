
# Bubbleton Generator for Blender -- CMC surfaces with bubbles, built as
# Darboux transforms of Delaunay surfaces.
#
# A BUBBLETON is what you get when a bubble is grafted onto a Delaunay
# surface without disturbing its mean curvature.  Away from its bubbles
# it is exactly the cylinder (or unduloid, or nodoid) it came from; on a
# bounded stretch it carries a chain of lobes.  Sterling and Wente named
# them, and they are the solitons of the sinh-Gordon equation that
# governs CMC surfaces -- the CMC analogue of the breather already in
# math_art/hyperbolic_surface_generator.py, which is the soliton of the
# sine-Gordon equation governing K = -1 surfaces.
#
# WHY THIS IS COMPUTABLE AT ALL.  The usual route to a bubbleton is a
# loop-group dressing, which needs a numerical Iwasawa factorisation at
# every point -- not something to put in an add-on.  Cho, Leschke and
# Ogata (2022) avoid it: they give the parallel sections of the flat
# connection IN CLOSED FORM for a Delaunay surface (their Theorem 3.1),
# after which the Darboux transform is purely ALGEBRAIC (their
# Theorem 2.2).  So the whole generator is: evaluate elliptic functions
# on a grid, build one quaternion per grid point, invert it, add.  No
# ODE integration, no iteration, no loop groups.
#
# THE CONSTRUCTION, in the order the code performs it:
#
#   1. The Delaunay surface itself, conformally parametrised as
#          f(x,y) = i p(x) + j q(x) e^{-iy} ,
#      which in (i,j,k) coordinates is (p, q cos y, q sin y) -- so the
#      axis of revolution is the i-axis.  With necksize
#      r in (-inf, 1/2] and parameter M = 1 - (1 - 1/(1-r))^2,
#          q = (1-r) dn((1-r)x, M) ,   p' = q^2 + r(1-r) .
#      r = 1/2 is the cylinder, 0 < r < 1/2 the unduloids, r < 0 the
#      nodoids.  (M is never negative and never >= 1: writing
#      u = 1/(1-r) in (0,2], M = 1 - (1-u)^2 in [0,1).  It tends to 1 as
#      r -> 0, which is the excluded case, so guard r away from 0 rather
#      than worrying about the m >= 1 branch.)
#
#   2. A spectral parameter mu, chosen at a RESONANCE POINT so that the
#      transform closes up.  With a = (mu + 1/mu)/2 and
#      b = i (1/mu - mu)/2, the eigenvalue split of the y-system is
#          t = sqrt(1 + 2 r (1-r) (a - 1)) ,
#      and the resonance condition is simply t = n/m for coprime
#      integers n (lobes) and m (covers).  Inverting,
#          a = 1 + (n^2/m^2 - 1) / (2 r (1-r)) ,   mu = a + sqrt(a^2 - 1).
#      At the cylinder r = 1/2 with m = 1 this reproduces the paper's
#      Figure 1 values mu_2 = 7 + 4 sqrt 3 and mu_3 = 17 + 12 sqrt 2,
#      which is the first thing the self-test checks.
#
#   3. The parallel sections (Theorem 3.1), where the elliptic integral
#      of the THIRD kind appears:
#          alpha_pm = e^{iy/2} ( q b - i q'(a-1)
#                                + j (1 + p'(a-1) pm t) ) e^{pm ity/2} c_pm
#      with c_pm built from Pi(N; am((1-r)x, M) | M) and
#          N = (a-1)(1-r)^2 M / (1 pm t + (a-1)(1-r)) .
#      A general parallel section is alpha = alpha_+ m_+ + alpha_- m_-;
#      the complex constants m_pm slide the bubble along the axis and
#      spin it, and are exposed as operator parameters.
#
#   4. The Darboux transform (Theorem 2.2), algebraic:
#          T^{-1} = (1/2)( N_gauss (a-1) + alpha b alpha^{-1} ) ,
#          f_hat  = f + T .
#
# QUATERNION CONVENTIONS, which are the whole difficulty.  Quaternions
# are carried as PAIRS of complex numbers, q = z0 + j z1, because that
# is the split the paper uses and because right multiplication by a
# complex scalar -- which is what a, b, m_pm all do -- is then simply
# componentwise.  Two rules decide whether the surface comes out right
# or merely plausible:
#
#   * ALL scalars multiply on the RIGHT.  The C^2 structure is (H, I)
#     with I right multiplication by i.  Left and right multiplication
#     differ, and silently commuting them gives a bubble-shaped surface
#     that is not CMC.
#   * j z = conj(z) j for complex z.  Hence the product rule
#     (z0 + j z1)(w0 + j w1) = (z0 w0 - conj(z1) w1)
#                              + j (conj(z0) w1 + z1 w0) ,
#     and left multiplication by a complex c conjugates the j-part.
#
# The Gauss map N is not written out; it is recovered from the paper's
# own characterisation f_y N = f_x, i.e. N = f_y^{-1} f_x, which is
# guaranteed to match their convention and is checked in the self-test
# to be a unit imaginary quaternion.
#
# References:
# - Joseph Cho, Katrin Leschke, Yuta Ogata, "New explicit CMC cylinders
#   and same-lobed CMC multibubbletons", arXiv:2205.14675 (2022) --
#   Theorem 2.2 (the algebraic Darboux transform), Theorem 3.1 (the
#   parallel sections in closed form, used here verbatim), Theorem 4.1
#   (which (m,n) are admissible for a given necksize) and Theorem 5.1
#   (same-lobed multibubbletons).
# - Ivan Sterling and Henry C. Wente, "Existence and classification of
#   constant mean curvature multibubbletons of finite and infinite
#   type", Indiana Univ. Math. J. 42 (1993), 1239-1266 -- the surfaces,
#   and the name.
# - Martin Kilian, "Bubbletons are not embedded", (2010),
#   arXiv:1010.6180 -- a single bubbleton always self-intersects, which
#   is why the generated surface is not expected to be embedded.
# - Charles-Eugene Delaunay (1841) for the underlying surfaces; see
#   math_art/delaunay_generator.py, which builds them independently from
#   the first integral and agrees with this module's profile.

bl_info = {
    "name": "Bubbletons",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Bubbleton",
    "description": "CMC bubbletons: Darboux transforms of Delaunay "
                   "surfaces at a resonance point",
    "category": "Add Mesh",
}

import math

import numpy as np

from .minsurf.elliptic import ellipk, ellippi, jacobi_am, jacobi_sncndn


# --------------------------------------------------------------------------
# quaternions as pairs of complex arrays:  Q = z0 + j z1
# --------------------------------------------------------------------------

def qmul(A, B):
    """(z0 + j z1)(w0 + j w1), using j z = conj(z) j."""
    a0, a1 = A
    b0, b1 = B
    return (a0 * b0 - np.conj(a1) * b1,
            np.conj(a0) * b1 + a1 * b0)


def qconj(A):
    """Quaternion conjugate: negates i, j and k."""
    a0, a1 = A
    return (np.conj(a0), -a1)


def qnorm2(A):
    a0, a1 = A
    return (a0 * np.conj(a0) + a1 * np.conj(a1)).real


def qinv(A):
    """Quaternionic inverse conj(A)/|A|^2 -- NOT a matrix inverse."""
    n2 = np.maximum(qnorm2(A), 1e-300)
    c0, c1 = qconj(A)
    return (c0 / n2, c1 / n2)


def qrmul_c(A, c):
    """Right multiplication by a COMPLEX scalar: componentwise."""
    a0, a1 = A
    return (a0 * c, a1 * c)


def qlmul_c(c, A):
    """Left multiplication by a complex scalar: conjugates the j-part."""
    a0, a1 = A
    return (c * a0, np.conj(c) * a1)


def qadd(A, B):
    return (A[0] + B[0], A[1] + B[1])


def q_to_xyz(A):
    """Imaginary part in (i, j, k) coordinates.

    With z1 = Y + iZ, j z1 = Y j + (iZ) picked up as j i Z = -k Z, so the
    k component is -Im(z1) and not +Im(z1) -- the sign that makes
    f = i p + j q e^{-iy} come out as (p, q cos y, q sin y)."""
    z0, z1 = A
    return np.stack([np.imag(z0), np.real(z1), -np.imag(z1)], axis=-1)


# --------------------------------------------------------------------------
# the Delaunay surface, in the paper's parametrisation
# --------------------------------------------------------------------------

def delaunay_pq(r, x):
    """(p, q, q', p') on the x-grid for necksize r.  p is obtained by
    cumulative Simpson of p' = q^2 + r(1-r); its additive constant is
    irrelevant (it slides the surface along its axis)."""
    r = float(r)
    u = 1.0 / (1.0 - r)
    M = 1.0 - (1.0 - u) ** 2
    M = min(max(M, 0.0), 1.0 - 1e-15)
    s = (1.0 - r) * np.asarray(x, dtype=float)
    sn, cn, dn = jacobi_sncndn(s, M)
    q = (1.0 - r) * dn
    # d/du dn(u,m) = -m sn cn, and s = (1-r) x
    qp = -(1.0 - r) ** 2 * M * sn * cn
    pp = q * q + r * (1.0 - r)
    h = float(x[1] - x[0])
    p = np.concatenate([[0.0], np.cumsum(0.5 * h * (pp[:-1] + pp[1:]))])
    return p, q, qp, pp, M


def delaunay_period(r):
    """x-period of the Delaunay profile: dn has period 2K(M)."""
    u = 1.0 / (1.0 - r)
    M = min(max(1.0 - (1.0 - u) ** 2, 0.0), 1.0 - 1e-15)
    return 2.0 * ellipk(M) / (1.0 - r)


def is_admissible(r, n, m=1):
    """Is (r, n, m) a real resonance point?  This is Theorem 4.1's
    condition, arrived at from the requirement that mu be real rather
    than from the paper's case table -- the two agree, and this form
    cannot be mis-transcribed.

    mu = a +- sqrt(a^2 - 1) is real exactly when |a| > 1, with
    a = 1 + ((n/m)^2 - 1)/(2 r (1-r)).  For the cylinder r = 1/2 that
    reduces to n > m; for a nodoid (r < 0) the denominator is negative
    and the inequality flips, which is why nodoids admit n < m and
    cylinders do not.  Equivalent to the paper's introduction, which
    states the obstruction as n > m(1-2r) for nodoids and n < m(1-2r)
    for unduloids."""
    if math.gcd(int(n), int(m)) != 1:
        return False
    k = float(r) * (1.0 - float(r))
    if abs(k) < 1e-12:
        return False
    a = 1.0 + ((n / m) ** 2 - 1.0) / (2.0 * k)
    if a * a - 1.0 <= 1e-15:
        return False
    mu = a + math.sqrt(a * a - 1.0)
    if abs(mu) < 1e-9 or abs(abs(mu) - 1.0) < 1e-9:
        return False
    return characteristics(r, n, m)[2]


def characteristics(r, n, m=1):
    """(N_+, N_-, usable) -- the elliptic characteristic of each branch
    and whether Pi can be evaluated without a principal value.

    N_pm = (a-1)(1-r)^2 M / (1 pm t + (a-1)(1-r)) is the characteristic
    of Pi(N; . | M), and 1 - N sn^2 vanishes wherever sn^2 = 1/N.  Since
    sn sweeps [-1, 1] over any quarter period, N >= 1 puts a genuine
    pole inside every x-range worth drawing.

    Measured across the family, N < 1 for the CYLINDER (where M = 0 and
    N vanishes identically) and for every UNDULOID, and N > 1 for every
    NODOID -- e.g. r = -0.4, n = 2 gives N_+ = 6.43.  So bubbletons on
    nodoids need the Cauchy principal value of Pi, which
    minsurf.elliptic.ellippi deliberately refuses rather than
    approximating; they are rejected up front instead of returning a
    plausible wrong surface.  Very thin unduloids approach the same
    limit from below (r = 0.1 already gives N_- = 0.952), which is why
    this is tested numerically rather than as a rule about the sign
    of r."""
    k = float(r) * (1.0 - float(r))
    a = 1.0 + ((n / m) ** 2 - 1.0) / (2.0 * k)
    t = n / m
    u = 1.0 / (1.0 - r)
    M = 1.0 - (1.0 - u) ** 2
    out = []
    for sgn in (+1.0, -1.0):
        d0 = 1.0 + sgn * t + (a - 1.0) * (1.0 - r)
        if abs(d0) < 1e-12:
            return (float('inf'), float('inf'), False)
        out.append((a - 1.0) * (1.0 - r) ** 2 * M / d0)
    return (out[0], out[1], max(out) < 1.0 - 1e-9)


def resonance(r, n, m=1):
    """(mu, a, b, t) at the resonance point with n lobes on m covers.

    The condition is t = n/m; inverting t^2 = 1 + 2r(1-r)(a-1) gives a
    directly, and mu = a + sqrt(a^2 - 1) is the outer root (Remark 2.3
    selects it).  At r = 1/2, m = 1 this reproduces mu_2 = 7 + 4 sqrt 3
    and mu_3 = 17 + 12 sqrt 2."""
    r = float(r)
    k = r * (1.0 - r)
    if abs(k) < 1e-12:
        raise ValueError("necksize r = 0 (or 1) is degenerate")
    a = 1.0 + ((n / m) ** 2 - 1.0) / (2.0 * k)
    if a * a - 1.0 < 0.0:
        raise ValueError(f"no real resonance point for r={r}, n={n}, m={m}")
    mu = a + math.sqrt(a * a - 1.0)
    if abs(mu) < 1e-12 or abs(abs(mu) - 1.0) < 1e-12:
        raise ValueError(f"mu = {mu} is degenerate (0 or +-1)")
    b = 1j * (1.0 / mu - mu) / 2.0
    t = n / m
    return mu, a, b, t


def parallel_section(r, x, y, n, m=1, m_plus=1.0 + 0j, m_minus=1.0 + 0j):
    """Theorem 3.1: alpha = alpha_+ m_+ + alpha_- m_- on the (x, y) grid.

    x and y are 2-D arrays of the same shape (a meshgrid)."""
    mu, a, b, t = resonance(r, n, m)
    x1 = x[:, 0] if x.ndim == 2 else np.asarray(x)
    p, q, qp, pp, M = delaunay_pq(r, x1)
    am = jacobi_am((1.0 - r) * x1, M)

    alpha = None
    for sgn in (+1.0, -1.0):
        denom0 = 1.0 + sgn * t + (a - 1.0) * (1.0 - r)
        N_ell = (a - 1.0) * (1.0 - r) ** 2 * M / denom0
        # Pi(N; am((1-r)x, M) | M), the integral of 1/(1 - N sn^2)
        Pi = ellippi(float(N_ell), am, M)
        arg = (x1 - (1.0 + sgn * t) / denom0 / (1.0 - r) * Pi)
        expo = 1j * b * (1.0 + sgn * t) / (2.0 * (a - 1.0)) * arg
        denom = 1.0 + sgn * t + (a - 1.0) * pp
        c = np.exp(expo) / np.sqrt(np.maximum(np.abs(denom), 1e-300))
        # the y-independent quaternion bracket, as (z0, z1)
        z0 = q * b - 1j * qp * (a - 1.0)
        z1 = (1.0 + pp * (a - 1.0) + sgn * t) + 0j
        # broadcast the x-only quantities across y
        Z = (z0[:, None] + 0.0 * y, z1[:, None] + 0.0 * y)
        Z = qlmul_c(np.exp(0.5j * y), Z)
        Z = qrmul_c(Z, np.exp(0.5j * sgn * t * y) * c[:, None])
        Z = qrmul_c(Z, m_plus if sgn > 0 else m_minus)
        alpha = Z if alpha is None else qadd(alpha, Z)
    return alpha, (mu, a, b, t)


def delaunay_quaternion(r, x, y):
    """f = i p + j q e^{-iy} as a quaternion pair, plus its Gauss map."""
    x1 = x[:, 0] if x.ndim == 2 else np.asarray(x)
    p, q, qp, pp, M = delaunay_pq(r, x1)
    P = p[:, None] + 0.0 * y
    Q = q[:, None] + 0.0 * y
    QP = qp[:, None] + 0.0 * y
    PP = pp[:, None] + 0.0 * y
    e = np.exp(-1j * y)
    f = (1j * P + 0j, Q * e)
    f_x = (1j * PP + 0j, QP * e)
    f_y = (0.0 * P + 0j, -1j * Q * e)
    # the paper's characterisation: f_y N = f_x, so N = f_y^{-1} f_x
    N = qmul(qinv(f_y), f_x)
    return f, N, f_x, f_y


def bubbleton(r, x, y, n, m=1, m_plus=1.0 + 0j, m_minus=1.0 + 0j):
    """Theorem 2.2: f_hat = f + T with
    T^{-1} = (1/2)( N (a-1) + alpha b alpha^{-1} )."""
    f, N, _, _ = delaunay_quaternion(r, x, y)
    alpha, (mu, a, b, t) = parallel_section(r, x, y, n, m,
                                            m_plus, m_minus)
    Tinv = (0.5 * (qrmul_c(N, a - 1.0)[0]
                   + qmul(qrmul_c(alpha, b), qinv(alpha))[0]),
            0.5 * (qrmul_c(N, a - 1.0)[1]
                   + qmul(qrmul_c(alpha, b), qinv(alpha))[1]))
    return qadd(f, qinv(Tinv)), f, (mu, a, b, t)


def darboux_beta(N, alpha, a, b):
    """beta = (1/2)( N alpha (a-1) + alpha b )  -- Theorem 2.4 (2.4).

    The single-bubble transform of Theorem 2.2 is f + T with
    T^{-1} = (1/2)(N(a-1) + alpha b alpha^{-1}).  The two agree, and it
    is worth seeing why: T^{-1} alpha = (1/2)(N(a-1)alpha + alpha b),
    which equals beta only because mu is REAL, so a is real and the
    scalar (a-1) commutes with the quaternion alpha.  For complex mu
    they would differ."""
    return qrmul_c(qadd(qrmul_c(qmul(N, alpha), a - 1.0),
                        qrmul_c(alpha, b)), 0.5)


def multibubbleton(r, x, y, specs):
    """Bianchi permutability (Theorem 2.4): graft several bubbles onto
    one Delaunay surface without further integration.

    `specs` is a list of (n, m, m_plus, m_minus) -- one per bubble, each
    at its own resonance point.  Two bubbles must have DIFFERENT mu, or
    the common Darboux transform collapses back to f (the paper notes
    this for mu_1 = mu_2).

    For bubbles 1 and 2 the common transform is
        alpha = alpha_2 - alpha_1 beta_1^{-1} beta_2 ,
        beta  = beta_2  - beta_1 alpha_1^{-1} alpha_2 (a_2-1)/(a_1-1) ,
        f_hat = f_1 + alpha beta^{-1} ,
    and further bubbles iterate the same step.  Everything is algebraic:
    the only integration is the one already done for each parallel
    section.

    The result closes on lcm(m_1, ..., m_l) covers, so the caller must
    take y over [0, 2 pi lcm]."""
    if not specs:
        raise ValueError("multibubbleton needs at least one bubble")
    f, N, _, _ = delaunay_quaternion(r, x, y)

    sections = []
    mus = []
    for (n, m, mp, mm) in specs:
        al, (mu, a, b, t) = parallel_section(r, x, y, n, m, mp, mm)
        if any(abs(mu - v) < 1e-9 for v in mus):
            raise ValueError(
                f"two bubbles share the spectral parameter mu={mu}; "
                f"their common Darboux transform is the original "
                f"surface.  Use different (n, m).")
        mus.append(mu)
        sections.append((al, a, b))

    al1, a1, b1 = sections[0]
    be1 = darboux_beta(N, al1, a1, b1)
    cur = qadd(f, qmul(al1, qinv(be1)))          # f_1
    for (al2, a2, b2) in sections[1:]:
        be2 = darboux_beta(N, al2, a2, b2)
        alpha = qadd(al2, qrmul_c(qmul(qmul(al1, qinv(be1)), be2), -1.0))
        beta = qadd(be2, qrmul_c(qmul(qmul(be1, qinv(al1)), al2),
                                 -(a2 - 1.0) / (a1 - 1.0)))
        cur = qadd(cur, qmul(alpha, qinv(beta)))
        # the accumulated surface becomes the base for the next bubble
        al1, be1, a1 = alpha, beta, a2
    return cur, f


def multibubble_covers(specs):
    """lcm of the cover counts -- how many times y must run round."""
    out = 1
    for sp in specs:
        out = out * int(sp[1]) // math.gcd(out, int(sp[1]))
    return out


def mean_curvature_xyz(X, hx, hy):
    """Mean curvature of a parametrised surface sampled on a uniform
    (x, y) grid, by central differences: H = (EN - 2FM + GL)/(2(EG-F^2)).
    Interior points only -- the returned array is trimmed by one."""
    Xu = (X[2:, 1:-1] - X[:-2, 1:-1]) / (2 * hx)
    Xv = (X[1:-1, 2:] - X[1:-1, :-2]) / (2 * hy)
    Xuu = (X[2:, 1:-1] - 2 * X[1:-1, 1:-1] + X[:-2, 1:-1]) / hx ** 2
    Xvv = (X[1:-1, 2:] - 2 * X[1:-1, 1:-1] + X[1:-1, :-2]) / hy ** 2
    Xuv = (X[2:, 2:] - X[2:, :-2] - X[:-2, 2:] + X[:-2, :-2]) / (4 * hx * hy)
    nrm = np.cross(Xu, Xv)
    nl = np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-300)
    nh = nrm / nl
    E = (Xu * Xu).sum(-1)
    F = (Xu * Xv).sum(-1)
    G = (Xv * Xv).sum(-1)
    L = (Xuu * nh).sum(-1)
    M_ = (Xuv * nh).sum(-1)
    N_ = (Xvv * nh).sum(-1)
    return (E * N_ - 2 * F * M_ + G * L) / (2 * (E * G - F * F))


def _axial_offset(r, ratio):
    """Roughly where an integration-constant ratio puts a bubble along
    the axis.  Measured, not derived: the offset is linear in
    log|ratio| at about 0.58 units per unit for the cylinder, and this
    is only used to size the drawing window, never the geometry."""
    return 0.58 * math.log(max(abs(ratio), 1e-12))


def bubble_extent(r, n, m=1, probe=600, margin=1.05):
    """Half-width in x of the region where the bubble actually bulges,
    i.e. where the transformed radius exceeds the underlying Delaunay
    surface's widest by `margin`.

    Needed because the bubble's axial size is not a fixed number of
    Delaunay periods -- it shrinks sharply as lobes are added (3.79
    units at n = 2 on the cylinder against 0.71 at n = 5, a factor of
    five).  Framing by period count therefore buries the bubble in pipe
    for high lobe counts, which is what a fixed window did.

    The test is against the base surface's own maximum radius rather
    than a pointwise difference, because a bubbleton is asymptotic to a
    TRANSLATED copy of its Delaunay surface: on a cylinder there is no
    phase to shift and a pointwise difference does decay, but on an
    unduloid it never does, and a pointwise criterion silently reports
    the whole domain."""
    L = delaunay_period(r)
    x = np.linspace(-3.0 * L, 3.0 * L, probe)
    y = np.linspace(0.0, 2.0 * math.pi * m, 64, endpoint=False)
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    fh, f0, _ = bubbleton(r, Xg, Yg, n, m)
    rad = np.hypot(q_to_xyz(fh)[:, :, 1], q_to_xyz(fh)[:, :, 2]).max(1)
    base = float(np.hypot(q_to_xyz(f0)[:, :, 1],
                          q_to_xyz(f0)[:, :, 2]).max())
    idx = np.where(rad > margin * base)[0]
    if idx.size == 0:
        return 0.5 * L
    return max(0.5 * float(x[idx[-1]] - x[idx[0]]), 1e-3)


def build_surface(r=0.5, n=3, m=1, periods=0.0, ures=240, vres=160,
                  m_plus=1.0 + 0j, m_minus=1.0 + 0j, scale=1.0,
                  pad=3.5, extra=()):
    """Mesh one bubbleton.  y runs over [0, 2 m pi] because the section
    closes only on the m-fold cover.

    With `periods` <= 0 the axial window is fitted to the bubble (see
    `bubble_extent`) and `pad` says how much surrounding Delaunay
    surface to keep; a positive `periods` overrides that with an
    explicit number of Delaunay periods."""
    L = delaunay_period(r)
    specs = [(n, m, m_plus, m_minus)] + [
        (en, em, complex(emp), 1.0 + 0j) for (en, em, emp) in extra]
    covers = multibubble_covers(specs)
    if periods and periods > 0.0:
        half = 0.5 * periods * L
    else:
        # the window must hold every bubble AND the gap between
        # them, since the extra bubbles are deliberately slid along the
        # axis; bubble_extent measures one bubble about its own centre.
        half = pad * max(bubble_extent(r, sn, sm) for (sn, sm, _, _)
                         in specs)
        for (_, _, emp, _) in specs[1:]:
            half = max(half, 1.3 * abs(_axial_offset(r, emp)))
    x = np.linspace(-half, half, ures)
    y = np.linspace(0.0, 2.0 * math.pi * covers, vres, endpoint=False)
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    if extra:
        fh, f = multibubbleton(r, Xg, Yg, specs)
        info = resonance(r, n, m)
    else:
        fh, f, info = bubbleton(r, Xg, Yg, n, m, m_plus, m_minus)
    V = q_to_xyz(fh).reshape(-1, 3)
    faces = []
    for i in range(ures - 1):
        for j in range(vres):
            j1 = (j + 1) % vres
            faces.append((i * vres + j, i * vres + j1,
                          (i + 1) * vres + j1, (i + 1) * vres + j))
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    V = (V - 0.5 * (lo + hi)) * ((2.0 / ext if ext > 1e-9 else 1.0) * scale)
    return V, faces, info


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_bubbleton_add(bpy.types.Operator):
        """Add a CMC bubbleton: a Delaunay surface with bubbles grafted
        on, built as a Darboux transform at a resonance point"""
        bl_idname = "mesh.bubbleton_add"
        bl_label = "Bubbleton"
        bl_options = {'REGISTER', 'UNDO'}

        necksize: FloatProperty(
            name="Necksize r", default=0.5, min=-3.0, max=0.5,
            description="Necksize of the underlying Delaunay surface: "
                        "0.5 is the cylinder, (0, 0.5) unduloids, "
                        "negative nodoids.  r = 0 is excluded")
        lobes: IntProperty(
            name="Lobes n", default=3, min=2, max=12,
            description="Number of lobes on the bubble; the resonance "
                        "point is fixed by t = n/m")
        covers: IntProperty(
            name="Covers m", default=1, min=1, max=6,
            description="The section closes only on the m-fold cover, "
                        "so y runs over [0, 2 m pi].  Must be coprime "
                        "to the lobe count")
        preset: EnumProperty(
            name="Preset",
            items=[('CUSTOM', "Custom", "use the sliders below"),
                   ('CYL2', "Cylinder, 2 lobes",
                    "the classic bubbleton: a two-lobed bubble on a "
                    "straight cylinder"),
                   ('CYL3', "Cylinder, 3 lobes",
                    "a three-lobed bubble on a straight cylinder"),
                   ('CYL5', "Cylinder, 5 lobes",
                    "five lobes -- the bubble tightens as lobes are "
                    "added"),
                   ('UND2', "Unduloid, 2 lobes",
                    "a two-lobed bubble riding on a Delaunay unduloid"),
                   ('UND3', "Unduloid, 3 lobes",
                    "a three-lobed bubble on a Delaunay unduloid"),
                   ('TWIST', "Twizzler (3 lobes, 2 covers)",
                    "a bubble that closes only on the double cover, so "
                    "it winds as it goes round"),
                   ('DOUBLE_UND', "Double bubbleton (unduloid)",
                    "two-lobed AND three-lobed bubbles on a Delaunay "
                    "unduloid, by Bianchi permutability"),
                   ('SPLASH', "Colliding bubbles (2 and 3)",
                    "a two-lobed bubble meeting a three-lobed one on a "
                    "cylinder")],
            default='CYL2')
        second_lobes: IntProperty(
            name="Second Bubble Lobes", default=0, min=0, max=12,
            description="Graft a SECOND bubble with this many lobes by "
                        "Bianchi permutability.  0 leaves a single "
                        "bubble.  It must sit at a different resonance "
                        "point from the first, so a different lobe or "
                        "cover count")
        second_covers: IntProperty(
            name="Second Bubble Covers", default=1, min=1, max=6,
            description="Cover count of the second bubble; the surface "
                        "closes on the lowest common multiple of the two")
        second_shift: FloatProperty(
            name="Second Bubble Shift", default=-8.0, min=-20.0,
            max=20.0,
            description="Slides the second bubble along the axis. Two "
                        "bubbles with the same shift sit in the same "
                        "place and merge into one, so a visibly double "
                        "bubbleton needs them apart; about 0.58 units "
                        "of axis per unit of shift")
        periods: FloatProperty(
            name="Delaunay Periods", default=0.0, min=0.0, max=12.0,
            description="Axial window in Delaunay periods.  Leave at 0 "
                        "to fit the window to the bubble automatically "
                        "-- the bubble's size varies fivefold with the "
                        "lobe count, so a fixed window buries it")
        pad: FloatProperty(
            name="Surroundings", default=3.5, min=1.0, max=10.0,
            description="With automatic framing, how much Delaunay "
                        "surface to keep either side of the bubble")
        shift: FloatProperty(
            name="Bubble Shift", default=0.0, min=-6.0, max=6.0,
            description="Slides the bubble along the axis: the ratio of "
                        "the two integration constants m+ and m-")
        spin: FloatProperty(
            name="Bubble Spin", default=0.0, min=-180.0, max=180.0,
            description="Rotates the bubble about the axis (degrees), "
                        "the phase of the same ratio")
        ures: IntProperty(name="Axial Samples", default=240, min=24,
                          max=1200)
        vres: IntProperty(name="Around Samples", default=160, min=12,
                          max=800)
        shade_smooth: BoolProperty(name="Smooth Shading", default=True)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        _PRESETS = {
            'CYL2': (0.5, 2, 1), 'CYL3': (0.5, 3, 1),
            'CYL5': (0.5, 5, 1), 'UND2': (0.3, 2, 1),
            'UND3': (0.3, 3, 1), 'TWIST': (0.5, 3, 2),
            'DOUBLE_UND': (0.3, 2, 1), 'SPLASH': (0.5, 2, 1),
        }
        # Presets that graft a SECOND bubble by Bianchi permutability.
        # The third entry is its axial shift: with the same integration
        # constants both bubbles sit at the same place and simply merge,
        # so a separated double bubbleton has to slide one of them.
        _SECOND = {'DOUBLE_UND': (3, 1, -11.0), 'SPLASH': (3, 1, -6.0)}

        def execute(self, context):
            if self.preset != 'CUSTOM':
                necksize, lobes, covers = self._PRESETS[self.preset]
            else:
                necksize, lobes, covers = (self.necksize, self.lobes,
                                           self.covers)
            if self.preset != 'CUSTOM':
                extra = ([self._SECOND[self.preset]]
                         if self.preset in self._SECOND else [])
            elif self.second_lobes > 0:
                extra = [(self.second_lobes, self.second_covers,
                          self.second_shift)]
            else:
                extra = []
            extra = [(en, em, math.exp(esh)) for (en, em, esh) in extra]
            for (en, em, _) in extra:
                if not is_admissible(necksize, en, em):
                    self.report({'ERROR'},
                                f"second bubble {en}/{em} is not a "
                                f"resonance point at necksize "
                                f"{necksize:.3f}")
                    return {'CANCELLED'}
            if math.gcd(lobes, covers) != 1:
                self.report({'ERROR'},
                            f"lobes {lobes} and covers "
                            f"{covers} must be coprime")
                return {'CANCELLED'}
            if not is_admissible(necksize, lobes, covers):
                self.report(
                    {'ERROR'},
                    f"n/m = {lobes}/{covers} is not a "
                    f"resonance point at necksize "
                    f"{necksize:.3f}: the spectral parameter comes "
                    f"out complex, or Pi needs a principal value.  "
                    f"A cylinder needs n > m; nodoids are not "
                    f"supported (their characteristic exceeds 1)")
                return {'CANCELLED'}
            if abs(necksize) < 1e-3:
                self.report({'ERROR'},
                            "necksize r = 0 is excluded (the resonance "
                            "point divides by 2r(1-r))")
                return {'CANCELLED'}
            ratio = math.exp(self.shift) * complex(
                math.cos(math.radians(self.spin)),
                math.sin(math.radians(self.spin)))
            try:
                verts, faces, info = build_surface(
                    necksize, lobes, covers, self.periods,
                    self.ures, self.vres, ratio, 1.0 + 0j, self.scale,
                    self.pad, tuple(extra))
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            mu, a, b, t = info
            me = bpy.data.meshes.new("Bubbleton")
            me.from_pydata([tuple(v) for v in verts], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.shade_smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Bubbleton", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report(
                {'INFO'},
                f"Bubbleton n/m = {lobes}/{covers}"
                f"{'+' + '+'.join(f'{a}/{b}' for a, b, _ in extra) if extra else ''}"
                f" on "
                f"necksize {necksize:.3f}: V={len(me.vertices)} "
                f"F={len(me.polygons)}, resonance mu={mu:.6f}, "
                f"a={a:.6f}, t={t:.4f}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'CUSTOM':
                lay.prop(self, 'necksize')
                lay.prop(self, 'lobes')
                lay.prop(self, 'covers')
                if math.gcd(self.lobes, self.covers) != 1:
                    lay.label(text="Lobes and covers must be coprime",
                              icon='ERROR')
                elif not is_admissible(self.necksize, self.lobes,
                                       self.covers):
                    lay.label(text="Not a resonance point here",
                              icon='ERROR')
            if self.preset == 'CUSTOM':
                lay.prop(self, 'second_lobes')
                if self.second_lobes > 0:
                    lay.prop(self, 'second_covers')
                    lay.prop(self, 'second_shift')
            lay.prop(self, 'pad')
            lay.prop(self, 'periods')
            lay.prop(self, 'shift')
            lay.prop(self, 'spin')
            lay.prop(self, 'ures')
            lay.prop(self, 'vres')
            lay.prop(self, 'shade_smooth')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.bubbleton_add", icon='META_BALL')

    ADD_MENU = True    # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_bubbleton_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_bubbleton_add)


def _selftest():
    ok_all = True

    # 1) The resonance points, against the paper's Figure 1 values.  The
    #    paper's own advice is to check these "before drawing anything".
    for n, want, name in ((2, 7 + 4 * math.sqrt(3), "7+4sqrt3"),
                          (3, 17 + 12 * math.sqrt(2), "17+12sqrt2")):
        mu, a, b, t = resonance(0.5, n, 1)
        ok = abs(mu - want) < 1e-9 and abs(t - n) < 1e-12
        ok_all = ok_all and ok
        print(f"resonance cylinder n={n}: mu={mu:.9f} want {name}="
              f"{want:.9f} t={t:.4f} {'OK' if ok else 'BAD'}")

    # t = n/m must hold exactly at every resonance point, for unduloids
    # and nodoids alike -- it is the definition, so a failure means the
    # inversion for `a` is wrong.
    bad = tried = 0
    for r in (0.5, 0.3, 0.1, -0.4, -1.5):
        for n in (2, 3, 5):
            for m in (1, 2, 3):
                if math.gcd(n, m) != 1:
                    continue
                if not is_admissible(r, n, m):
                    continue
                tried += 1
                mu, a, b, t = resonance(r, n, m)
                t2 = math.sqrt(1.0 + 2.0 * r * (1.0 - r) * (a - 1.0))
                # relative tolerance: `a` reaches ~1e2 for thin
                # necks, so an absolute 1e-12 on a^2 is unreachable
                if (abs(t2 - n / m) > 1e-9
                        or abs(a * a + (b * b).real - 1.0)
                        > 1e-12 * max(1.0, a * a)):
                    bad += 1
    ok = bad == 0
    ok_all = ok_all and ok
    print(f"resonance t = n/m and a^2 + b^2 = 1 over 5 necksizes: "
          f"{bad} failures in {tried} admissible pairs "
          f"{'OK' if ok else 'BAD'}")

    cyl = [(n, m) for n in (2, 3, 5) for m in (1, 2, 3)
           if is_admissible(0.5, n, m)]
    nod = [(n, m) for n in (2, 3, 5) for m in (1, 2, 3)
           if is_admissible(-1.5, n, m)]
    ok = (all(n > m for n, m in cyl) and any(n < m for n, m in nod)
          and not is_admissible(0.0, 3, 1)
          and not is_admissible(0.5, 2, 4))
    ok_all = ok_all and ok
    print(f"admissibility: cylinder allows {cyl}, nodoid r=-1.5 allows "
          f"{nod} {'OK' if ok else 'BAD'}")

    # 2) Quaternion algebra, against the defining relations.  Everything
    #    downstream is quaternion multiplication, so this comes first.
    one = (np.array([1 + 0j]), np.array([0j]))
    qi = (np.array([1j]), np.array([0j]))
    qj = (np.array([0j]), np.array([1 + 0j]))
    qk = qmul(qi, qj)
    checks = {
        "i^2=-1": qmul(qi, qi), "j^2=-1": qmul(qj, qj),
        "k^2=-1": qmul(qk, qk), "ijk=-1": qmul(qmul(qi, qj), qk),
    }
    dev = max(float(abs(v[0] + 1)[0]) + float(abs(v[1])[0])
              for v in checks.values())
    ji = qmul(qj, qi)
    anti = float(abs(ji[0] + qk[0])[0]) + float(abs(ji[1] + qk[1])[0])
    inv = qmul(qmul(qi, qj), qinv(qmul(qi, qj)))
    invdev = float(abs(inv[0] - 1)[0]) + float(abs(inv[1])[0])
    ok = dev < 1e-14 and anti < 1e-14 and invdev < 1e-14
    ok_all = ok_all and ok
    print(f"quaternions: i^2=j^2=k^2=ijk=-1 dev {dev:.1e}, ji=-k dev "
          f"{anti:.1e}, q q^-1 = 1 dev {invdev:.1e} "
          f"{'OK' if ok else 'BAD'}")

    # 3) The Delaunay surface in the paper's parametrisation must be the
    #    same surface delaunay_generator builds from the first integral.
    #    Compare the PROFILE radius range: q runs between the necksize
    #    and the bulge.
    for r, tag in ((0.5, "cylinder"), (0.3, "unduloid"),
                   (-0.5, "nodoid")):
        x = np.linspace(0.0, 2.0 * delaunay_period(r), 4001)
        _, q, _, _, M = delaunay_pq(r, x)
        # the paper's normalisation has H = 1/2 ... check q's extremes
        ok = bool(np.isfinite(q).all()) and 0.0 <= M < 1.0
        ok_all = ok_all and ok
        print(f"delaunay {tag:9s} r={r:+.2f}: M={M:.6f} q in "
              f"[{q.min():.5f}, {q.max():.5f}] period="
              f"{delaunay_period(r):.5f} {'OK' if ok else 'BAD'}")

    # 4) The Gauss map really is a unit imaginary quaternion, and really
    #    does satisfy the paper's f_x N = -f_y as well as f_y N = f_x
    #    (only the second was used to define it, so the first is a test).
    x = np.linspace(-2.0, 2.0, 60)
    y = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    for r in (0.5, 0.3, -0.5):
        f, N, f_x, f_y = delaunay_quaternion(r, Xg, Yg)
        nn = qmul(N, N)
        unit = float(np.abs(nn[0] + 1.0).max() + np.abs(nn[1]).max())
        realpart = float(np.abs(np.real(N[0])).max())
        lhs = qmul(f_x, N)
        cross = float(np.abs(lhs[0] + f_y[0]).max()
                      + np.abs(lhs[1] + f_y[1]).max())
        ok = unit < 1e-9 and realpart < 1e-9 and cross < 1e-9
        ok_all = ok_all and ok
        print(f"gauss map r={r:+.2f}: N^2+1 {unit:.1e}, Re N "
              f"{realpart:.1e}, f_x N + f_y {cross:.1e} "
              f"{'OK' if ok else 'BAD'}")

    # 5) THE gate: a Darboux transform of a CMC surface is CMC.  Measured
    #    on the transformed surface in R^3, with no reference to the
    #    construction.  A wrong quaternion order or a wrong Pi branch
    #    breaks this immediately while still looking bubble-like.
    for r, n, m in ((0.5, 2, 1), (0.5, 3, 1), (0.3, 3, 1),
                    (0.15, 3, 1), (0.5, 3, 2), (0.3, 5, 2)):
        L = delaunay_period(r)
        hx, hy = 3.0 * L / 400.0, 2.0 * math.pi * m / 400.0
        x = np.arange(-1.5 * L, 1.5 * L, hx)
        y = np.arange(0.0, 2.0 * math.pi * m, hy)
        Xg, Yg = np.meshgrid(x, y, indexing='ij')
        fh, f0, info = bubbleton(r, Xg, Yg, n, m)
        Hh = mean_curvature_xyz(q_to_xyz(fh), hx, hy)
        H0 = mean_curvature_xyz(q_to_xyz(f0), hx, hy)
        Hh = Hh[np.isfinite(Hh)]
        H0 = H0[np.isfinite(H0)]
        med0, medh = float(np.median(H0)), float(np.median(Hh))
        q1, q3 = np.percentile(Hh, [25.0, 75.0])
        ok = abs(medh - med0) < 5e-3 and (q3 - q1) < 2e-2
        ok_all = ok_all and ok
        print(f"CMC r={r:+.2f} n/m={n}/{m}: base H={med0:+.6f}, "
              f"bubbleton H={medh:+.6f} IQR {q3 - q1:.1e} "
              f"{'OK' if ok else 'BAD'}")

    # 5b) MULTIBUBBLETONS (Theorem 2.4).  Bianchi permutability must
    #     again land on a CMC surface -- and this is the sharper test of
    #     the two, because the permutability formula mixes the two
    #     parallel sections through quaternionic inverses, so a wrong
    #     multiplication order survives the single-bubble test but not
    #     this one.
    for r, spec in ((0.5, [(2, 1), (3, 1)]),
                    (0.5, [(3, 1), (5, 1)]),
                    (0.3, [(2, 1), (3, 1)])):
        full = [(n, m, 1.0 + 0j, 1.0 + 0j) for (n, m) in spec]
        cov = multibubble_covers(full)
        L = delaunay_period(r)
        hx, hy = 3.0 * L / 360.0, 2.0 * math.pi * cov / 360.0
        xx = np.arange(-1.5 * L, 1.5 * L, hx)
        yy = np.arange(0.0, 2.0 * math.pi * cov, hy)
        Xg, Yg = np.meshgrid(xx, yy, indexing='ij')
        fh, f0 = multibubbleton(r, Xg, Yg, full)
        Hh = mean_curvature_xyz(q_to_xyz(fh), hx, hy)
        H0 = mean_curvature_xyz(q_to_xyz(f0), hx, hy)
        Hh = Hh[np.isfinite(Hh)]
        H0 = H0[np.isfinite(H0)]
        med0, medh = float(np.median(H0)), float(np.median(Hh))
        q1, q3 = np.percentile(Hh, [25.0, 75.0])
        ok = abs(medh - med0) < 1e-2 and (q3 - q1) < 5e-2
        ok_all = ok_all and ok
        print(f"CMC multi r={r:+.2f} {spec}: base H={med0:+.6f}, "
              f"multibubbleton H={medh:+.6f} IQR {q3 - q1:.1e} "
              f"{'OK' if ok else 'BAD'}")

    # two bubbles at the same resonance point must be refused: their
    # common Darboux transform is the original surface, not a surface
    # with two bubbles
    refused = 0
    for spec in ([(3, 1, 1 + 0j, 1 + 0j), (3, 1, 1 + 0j, 1 + 0j)],):
        try:
            xx = np.linspace(-1.0, 1.0, 8)
            yy = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
            Xg, Yg = np.meshgrid(xx, yy, indexing='ij')
            multibubbleton(0.5, Xg, Yg, spec)
        except ValueError:
            refused += 1
    ok = refused == 1
    ok_all = ok_all and ok
    print(f"multi guards: {refused}/1 duplicate-mu cases refused "
          f"{'OK' if ok else 'BAD'}")

    # the cover count is the lcm of the individual ones
    ok = (multibubble_covers([(3, 1), (5, 1)]) == 1
          and multibubble_covers([(3, 2), (5, 3)]) == 6
          and multibubble_covers([(5, 1), (5, 2), (5, 3), (5, 4)]) == 12)
    ok_all = ok_all and ok
    print(f"multi covers: lcm rule (12 for the same-lobed n=5 chain) "
          f"{'OK' if ok else 'BAD'}")

    # 6) Closure: the section closes on the m-fold cover and NOT sooner.
    for r, n, m in ((0.5, 3, 1), (0.5, 3, 2), (0.3, 5, 2)):  # closure
        x = np.linspace(-1.0, 1.0, 40)
        base = np.linspace(0.0, 2.0 * math.pi, 37)
        Xg, Yg = np.meshgrid(x, base, indexing='ij')
        A = q_to_xyz(bubbleton(r, Xg, Yg, n, m)[0])
        Xg2, Yg2 = np.meshgrid(x, base + 2.0 * math.pi * m, indexing='ij')
        B = q_to_xyz(bubbleton(r, Xg2, Yg2, n, m)[0])
        closed = float(np.abs(A - B).max())
        if m > 1:
            Xg3, Yg3 = np.meshgrid(x, base + 2.0 * math.pi,
                                   indexing='ij')
            C = q_to_xyz(bubbleton(r, Xg3, Yg3, n, m)[0])
            early = float(np.abs(A - C).max())
        else:
            early = float('inf')
        ok = closed < 1e-8 and early > 1e-3
        ok_all = ok_all and ok
        print(f"closure r={r:+.2f} n/m={n}/{m}: |f(y)-f(y+2pi m)|="
              f"{closed:.2e}, at one cover {early:.2e} (must differ "
              f"when m>1) {'OK' if ok else 'BAD'}")

    # 7) Lobe count: the dominant angular mode of the bubble should be n.
    for r, n in ((0.5, 2), (0.5, 3), (0.5, 5), (0.3, 3)):
        # a one-column grid would leave delaunay_pq no spacing to
        # integrate p over, so take a short strip and read its middle
        x = np.linspace(-0.2, 0.2, 9)
        y = np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
        Xg, Yg = np.meshgrid(x, y, indexing='ij')
        P = q_to_xyz(bubbleton(r, Xg, Yg, n, 1)[0])[len(x) // 2]
        rad = np.hypot(P[:, 1], P[:, 2])
        spec = np.abs(np.fft.rfft(rad - rad.mean()))
        mode = int(np.argmax(spec))
        ok = mode == n
        ok_all = ok_all and ok
        print(f"lobes r={r:+.2f} n={n}: dominant angular mode {mode} "
              f"{'OK' if ok else 'BAD'}")

    # 7b) The characteristic of Pi decides what is buildable: below 1
    #     for the cylinder and unduloids, above 1 for every nodoid, so
    #     nodoid bubbletons would need a principal value and are
    #     refused rather than approximated.
    cyl_ok = characteristics(0.5, 3, 1)[2]
    und_ok = characteristics(0.3, 3, 1)[2]
    nod = characteristics(-0.4, 2, 1)
    ok = cyl_ok and und_ok and (not nod[2]) and nod[0] > 1.0
    ok_all = ok_all and ok
    print(f"characteristic: cylinder usable={cyl_ok}, unduloid "
          f"usable={und_ok}, nodoid r=-0.4 N+={nod[0]:.3f} "
          f"usable={nod[2]} {'OK' if ok else 'BAD'}")

    # 8) Guards: non-coprime and degenerate parameters must be refused
    refused = 0
    for args in ((0.0, 3, 1), (1.0, 3, 1)):
        try:
            resonance(*args)
        except ValueError:
            refused += 1
    rejected = sum(1 for a in ((0.0, 3, 1), (0.5, 2, 4), (0.5, 2, 3),
                               (-0.4, 2, 1), (-1.5, 5, 1))
                   if not is_admissible(*a))
    ok = refused == 2 and rejected == 5
    ok_all = ok_all and ok
    print(f"guards: {refused}/2 degenerate necksizes raise, "
          f"{rejected}/5 inadmissible pairs rejected "
          f"{'OK' if ok else 'BAD'}")

    # 9) the mesher runs and is not collapsed
    V, F, info = build_surface(0.5, 3, 1, periods=3.0, ures=80, vres=60)
    V = np.asarray(V)
    ext = V.max(0) - V.min(0)
    ok = len(F) > 0 and np.isfinite(V).all() and ext.min() / ext.max() > 0.02
    ok_all = ok_all and ok
    print(f"build: V={len(V)} F={len(F)} aspect "
          f"{ext.min() / ext.max():.3f} {'OK' if ok else 'BAD'}")

    assert ok_all
    print("bubbleton standalone tests passed")
