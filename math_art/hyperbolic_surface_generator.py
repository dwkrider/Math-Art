
# Hyperbolic Surfaces Generator for Blender
#
# Smooth surfaces of constant negative Gaussian curvature (K = -1) from
# the classical theory of pseudospherical surfaces / the sine-Gordon
# equation:
#
#   PSEUDOSPHERE  the tractricoid -- the surface of revolution of a
#                 tractrix (Beltrami's model of the hyperbolic plane)
#   DINI          Dini's surface -- a pseudosphere sheared into a
#                 twisting helical band (twist exposed as a parameter)
#   KUEN          Kuen's surface -- a bounded K = -1 surface with a
#                 characteristic bulb and cusp
#   MINDING_BULGE and MINDING_SPINDLE -- the other two K = -1 surfaces
#                 of revolution (see below)
#   BREATHER      the surface of the sine-Gordon BREATHER: a bound
#                 soliton/antisoliton pair, cusped and many-lobed
#   TWO_SOLITON, THREE_SOLITON, FOUR_SOLITON -- the surfaces of the
#                 sine-Gordon multisoliton solutions, built by iterating
#                 the Backlund transformation algebraically (Bianchi
#                 permutability) and reading the surface off the frame
#                 with Sym's formula
#
# Pseudospherical surfaces and the sine-Gordon equation are the same
# subject twice over.  In Chebyshev (asymptotic) coordinates a K = -1
# surface has I = du^2 + 2 cos(phi) du dv + dv^2 and II = 2 sin(phi) du dv
# with phi the angle between the asymptotic lines, and the Gauss-Codazzi
# equations reduce to exactly phi_uv = sin(phi).  So each soliton
# solution is a surface: the 1-soliton is Dini's, and the breather --
# the solution that stays localised and oscillates rather than
# travelling -- is the BREATHER preset here.
#
# The three surfaces of revolution.  Writing a surface of revolution
# with the metric ds^2 = du^2 + f(u)^2 dv^2 (so u is arclength along the
# meridian), the Gauss curvature is K = -f''/f, so K = -1 forces
#     f'' = f ,   hence   f(u) = A cosh u + B sinh u ,
# and the sign of A^2 - B^2 splits the K = -1 surfaces of revolution
# into exactly THREE types -- a classification due to Minding (1839):
#
#   A^2 = B^2  parabolic    f = C e^{-u}    the pseudosphere (above)
#   A^2 > B^2  hyperbolic   f = a cosh u    the MINDING BULGE
#   A^2 < B^2  conic        f = a sinh u    the MINDING SPINDLE
#
# Embedded as r(u) = f(u), z(u) = integral sqrt(1 - f'(u)^2) du, which
# is real only while |f'| <= 1.  That bound is the whole story of these
# surfaces and is exposed rather than hidden: the bulge runs out at
# |u| = asinh(1/a) in a pair of flared rims, and the spindle at
# u = acosh(1/a) in an equator, past which no isometric embedding in
# R^3 continues.  Both therefore realise only a STRIP of the hyperbolic
# plane -- the same Hilbert obstruction as the pseudosphere -- and the
# spindle additionally closes up at two conical points, where it is not
# an immersion at all.  All three are locally isometric to each other
# and to the hyperbolic plane; they differ only in how they are placed
# in R^3, which is Minding's point.
#
# Expect a visible crease around the spindle's equator: it is real, not
# a meshing artifact.  Where |f'| -> 1 the meridian meets that circle
# with a VERTICAL tangent -- writing s for the arclength short of it,
# r_max - r ~ s while z_eq - z ~ s^{3/2}, so dr/dz blows up like
# s^{-1/2}.  The two mirrored halves therefore share a tangent line but
# not a curvature, and the closed spindle is C^1 and not C^2.  The
# bulge shows the same thing, less obtrusively, at its two rims.
#
# By Hilbert's theorem there is no complete C^2 isometric immersion of
# the whole hyperbolic plane in R^3, so each of these covers only a
# patch; the crocheted (ruffled) realisation of the *whole* plane lives
# in the separate Crochet generator. Output is centred and fit to a
# 2 m cube; pair it with the Curvature Colour operator to see the
# constant negative curvature as a uniform tint.
#
# References:
# - Pseudosphere: Eugenio Beltrami, "Saggio di interpretazione della
#   geometria non-euclidea", Giornale di Matematiche 6, 1868.
# - Dini's surface: Ulisse Dini, 1865.
# - Kuen's surface: Theodor Kuen, "Ueber Flaechen von constantem
#   Kruemmungsmass", Sitzungsber. Bayer. Akad. Wiss., 1884.
# - Hilbert's theorem: D. Hilbert, "Ueber Flaechen von constanter
#   Gausscher Kruemmung", Trans. AMS 2, 1901, pp. 87-99.
# - Minding bulge and spindle: Ferdinand Minding, "Wie sich entscheiden
#   laesst, ob zwei gegebene krumme Flaechen auf einander abwickelbar
#   sind oder nicht; nebst Bemerkungen ueber die Flaechen von
#   unveraenderlichem Kruemmungsmasse", J. reine angew. Math. (Crelle)
#   19 (1839), 370-387 -- the classification of the constant-curvature
#   surfaces of revolution into the parabolic, hyperbolic and conic
#   types, and the theorem that all surfaces of equal constant
#   curvature are locally isometric.
# - Breather surface: the sine-Gordon breather goes back to Bour (1862)
#   and Backlund (1883) for the transformation theory; the modern
#   soliton-surface framing (Lax pair, Sym's immersion formula, and the
#   spectral classification in which "breather" names a conjugate pair
#   of branch points) is
#   A. I. Bobenko, "Surfaces in terms of 2 by 2 matrices: old and new
#   integrable cases", in Harmonic Maps and Integrable Systems, Vieweg
#   1994, Sect. 8; and M. Melko and I. Sterling, "Application of soliton
#   theory to the construction of pseudospherical surfaces in R^3",
#   Ann. Global Anal. Geom. 11 (1993), 65-107.
# - Backlund transformation: A. V. Backlund, "Om ytor med konstant
#   negativ krokning", Lunds Universitets Arsskrift 19 (1883) -- the
#   line congruence carrying one pseudospherical surface to another,
#   equivalently the map between solutions of the sine-Gordon equation.
# - Bianchi permutability: L. Bianchi, "Sulla trasformazione di Backlund
#   per le superficie pseudosferiche", Rend. Accad. Naz. Lincei (5) 1
#   (1892), 3-12 -- two Backlund transforms commute, and their common
#   image is given algebraically; iterating it builds the multisoliton
#   surfaces without further integration.
# - Sym's formula: A. Sym, "Soliton surfaces and their application", in:
#   Soliton geometry from spectral problems, Lecture Notes in Physics
#   239, Springer, Berlin 1985, 154-231 -- the immersion as the
#   logarithmic derivative of the frame in the spectral parameter,
#   F = 2 rho Phi^{-1} dPhi/dt; used here exactly as stated in Bobenko
#   1994, Theorem 11.

bl_info = {
    "name": "Hyperbolic Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Hyperbolic Surface",
    "description": "Pseudosphere, Dini, Kuen, Minding, breather and "
                   "multi-soliton constant-negative-curvature surfaces",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    from . import rim_curve as _rim
except ImportError:  # flat import outside the package
    import rim_curve as _rim


def _pseudosphere(U, V, twist=0.0, a=0.5, breather_a=0.4,
                  amsler_angle=90.0, soliton_spread=1.8,
                  soliton_cross=False):
    x = np.cosh(U) ** -1 * np.cos(V)
    y = np.cosh(U) ** -1 * np.sin(V)
    z = U - np.tanh(U)
    return x, y, z


def _dini(U, V, twist=0.2, a=0.5, breather_a=0.4,
          amsler_angle=90.0, soliton_spread=1.8, soliton_cross=False):
    x = np.cos(V) * np.sin(U)
    y = np.sin(V) * np.sin(U)
    z = np.cos(U) + np.log(np.tan(U / 2.0)) + twist * V
    return x, y, z


def _cum_simpson(g, t):
    """Cumulative integral of the callable `g` over the uniform grid
    `t`, Simpson's rule per interval (so O(h^4), with `g` also sampled
    at the interval midpoints).  Returns an array the length of `t`
    starting at 0."""
    h = float(t[1] - t[0])
    y = g(t)
    ym = g(0.5 * (t[:-1] + t[1:]))
    seg = (h / 6.0) * (y[:-1] + 4.0 * ym + y[1:])
    return np.concatenate([[0.0], np.cumsum(seg)])


def _minding_profile(kind, a, n=2001):
    """Meridian of a Minding surface as tables (t, r, z), with the
    normalised profile parameter t running over [-1, 1].

    Both meridians are integrated in a variable that makes the
    integrand SMOOTH, because the naive one is not.  With u the
    arclength, z' = sqrt(1 - f'(u)^2) vanishes like a square root where
    |f'| -> 1, so quadrature in u converges only like h^{3/2} at
    exactly the rim that gives these surfaces their shape.  Substituting
    f'(u) = sin(phi) removes it:

      BULGE   f = a cosh u,  f' = a sinh u = sin(phi),  phi in [-pi/2, pi/2]
              r = sqrt(a^2 + sin^2 phi)
              dz = cos^2(phi) dphi / sqrt(a^2 + sin^2 phi)
              -- smooth on the closed interval; t = 2 phi / pi.

      SPINDLE f = a sinh u,  f' = a cosh u = sin(phi),  phi in [asin a, pi/2]
              r = sqrt(sin^2 phi - a^2)
              dz = cos^2(phi) dphi / sqrt(sin^2 phi - a^2)
              -- now singular at the TIP instead (u |-> phi has a
              critical point there), cured by the second substitution
              phi = phi0 + tau^2, under which dz/dtau and r are both
              smooth and vanish linearly.  The half meridian tip ->
              equator is then mirrored to close the spindle.

    So each surface is integrated in the variable that is smooth for it,
    rather than one shared variable that is smooth for neither."""
    if kind == 'MINDING_BULGE':
        # t in [-1, 1] <-> phi in [-pi/2, pi/2]
        t = np.linspace(-1.0, 1.0, n)
        phi = t * (math.pi / 2.0)
        r = np.sqrt(a * a + np.sin(phi) ** 2)

        def dz_dt(tt):
            p = tt * (math.pi / 2.0)
            s = np.sin(p)
            return (math.pi / 2.0) * np.cos(p) ** 2 / np.sqrt(a * a + s * s)

        z = _cum_simpson(dz_dt, t)
        return t, r, z - 0.5 * (z[0] + z[-1])       # centre the waist

    if kind != 'MINDING_SPINDLE':
        raise ValueError(f"not a Minding surface: {kind!r}")

    phi0 = math.asin(min(max(a, 1e-9), 1.0 - 1e-12))
    span = math.pi / 2.0 - phi0
    if span <= 1e-12:                                # a -> 1: degenerate
        span = 1e-12
    T = math.sqrt(span)
    tau = np.linspace(0.0, T, n)                     # 0 = tip, T = equator

    def _phi(tt):
        return phi0 + tt * tt

    def _r_of(tt):
        s = np.sin(_phi(tt))
        return np.sqrt(np.maximum(s * s - a * a, 0.0))

    def dz_dtau(tt):
        p = _phi(tt)
        s = np.sin(p)
        rad = np.sqrt(np.maximum(s * s - a * a, 0.0))
        # limit at the tip: sin^2 phi - a^2 ~ 2 a cos(phi0) tau^2, so
        # dz/dtau -> 2 cos^2(phi0) / sqrt(2 a cos phi0), a finite value.
        lim = 2.0 * math.cos(phi0) ** 2 / math.sqrt(
            2.0 * a * math.cos(phi0)) if a > 0 else 0.0
        out = np.full_like(np.asarray(tt, dtype=float), lim)
        good = rad > 1e-14
        out[good] = (2.0 * tt[good] * np.cos(p[good]) ** 2 / rad[good])
        return out

    W = _cum_simpson(dz_dtau, tau)                   # height above the tip
    z_half = W[-1] - W                               # height below equator
    r_half = _r_of(tau)
    s_half = 1.0 - tau / T                           # 1 at tip, 0 at equator

    order = np.argsort(s_half)                       # make t increasing
    sp, rp, zp = s_half[order], r_half[order], z_half[order]
    t = np.concatenate([-sp[::-1][:-1], sp])
    r = np.concatenate([rp[::-1][:-1], rp])
    z = np.concatenate([-zp[::-1][:-1], zp])
    return t, r, z


def _minding(kind):
    """Build the (U, V, twist, a) surface function for a Minding type.
    The profile tables are rebuilt per call because they depend on `a`;
    at n = 2001 that is well under a millisecond."""
    def fn(U, V, twist=0.0, a=0.5, breather_a=0.4,
           amsler_angle=90.0, soliton_spread=1.8, soliton_cross=False):
        t, r, z = _minding_profile(kind, a)
        ru = np.interp(U, t, r)
        zu = np.interp(U, t, z)
        return ru * np.cos(V), ru * np.sin(V), zu
    return fn


def _breather(U, V, twist=0.0, a=0.5, breather_a=0.4,
              amsler_angle=90.0, soliton_spread=1.8,
              soliton_cross=False):
    """The breather surface: the pseudospherical surface built from the
    BREATHER solution of the sine-Gordon equation.

    Pseudospherical surfaces correspond to solutions of phi_uv = sin phi
    through Chebyshev (asymptotic) coordinates, in which
        I  = du^2 + 2 cos(phi) du dv + dv^2 ,
        II = 2 sin(phi) du dv ,
    so K = -1 for every solution.  The one-soliton gives Dini's surface
    (already here); the breather -- a soliton/antisoliton pair bound
    into a localised oscillation -- integrates in closed form to

        w   = sqrt(1 - b^2) ,
        D   = b [ w^2 cosh^2(b u) + b^2 sin^2(w v) ] ,
        x   = -u + 2 w^2 cosh(b u) sinh(b u) / D ,
        y + i z = 2 w cosh(b u) e^{i v} (w cos(w v) - i sin(w v)) / D ,

    with the breather parameter 0 < b < 1 setting how tightly bound it
    is.  D is bounded below by b w^2 > 0, so the formula has no poles;
    the surface is nevertheless NOT an immersion everywhere -- it
    carries cusp edges where EG - F^2 -> 0, which is generic for
    soliton surfaces and is a feature of the picture, not a defect of
    the parametrisation.

    U and V arrive NORMALISED to [-1, 1] and are rescaled here, because
    the useful window depends on b.  The bulb decays like
    1/cosh^2(b u), so it spans |u| ~ 1/b, while x carries a bare -u that
    keeps growing: take u too wide and the whole surface renders as a
    thin needle with a speck of structure at the middle (at b = 0.8 the
    classic |u| <= 13 window is 96% needle).  In v the lobes come from
    sin(w v), of period pi/w, so that extent scales like 1/w."""
    b = min(max(breather_a, 1e-3), 1.0 - 1e-6)
    w = math.sqrt(1.0 - b * b)
    U = U * (1.6 / b)
    V = V * (11.0 / w)
    ca, sa = np.cosh(b * U), np.sinh(b * U)
    cw, sw = np.cos(w * V), np.sin(w * V)
    cv, sv = np.cos(V), np.sin(V)
    D = b * (w * w * ca * ca + b * b * sw * sw)
    x = -U + 2.0 * w * w * ca * sa / D
    y = 2.0 * w * ca * (w * cw * cv + sw * sv) / D
    z = 2.0 * w * ca * (w * cw * sv - sw * cv) / D
    return x, y, z


# --------------------------------------------------------------------------
# Multisoliton surfaces: iterated Backlund transformation + Sym's formula
# --------------------------------------------------------------------------
# The surfaces of the N-soliton solutions of sine-Gordon, built the way
# Bobenko 1994 (Sect. 8) frames the subject: a K = -1 surface is a
# solution phi of phi_xy = sin(phi) seen through its SU(2) moving frame
# Phi(x, y, lambda), and the immersion is Sym's formula
#
#     F = 2 rho Phi^{-1} dPhi/dt ,        lambda = e^t ,
#
# evaluated at lambda = 1 (Theorem 11 there).  The frame satisfies
# Phi_x = A Phi, Phi_y = B Phi with, after a lambda-independent diagonal
# gauge that leaves Sym's formula untouched,
#
#     A = (i/2)(phi_x sigma3 - lambda sigma1) ,
#     B = (i/(2 lambda)) [[0, e^{i phi}], [e^{-i phi}, 0]] ,
#
# whose compatibility is exactly phi_xy = sin(phi).  The VACUUM phi = 0
# has the closed-form frame Phi0 = exp((i/2)(y/lambda - lambda x) sigma1)
# -- its Sym surface is a straight line, the degenerate K = -1 "surface"
# every soliton is grafted onto.
#
# One Backlund transform with speed beta > 0 is one DARBOUX STEP on the
# frame: Phi -> D(lambda) Phi with
#
#     D(lambda) = lambda I - S ,   S = i beta [[0, sigma], [conj(sigma), 0]] ,
#
# where sigma = h1/h2 is the ratio of the components of h = Phi(i beta) h0,
# a solution of the Lax pair at the imaginary spectral point lambda = i beta.
# Matching powers of lambda in D_x + D A = A~ D and D_y + D B = B~ D forces
# e^{i phi~} = sigma^2 e^{-i phi} and, writing sigma = e^{i psi}, reduces
# every condition to the CLASSICAL Backlund system
#
#     ((phi~ + phi)/2)_x = phi_x - beta sin((phi~ + phi)/2) ,
#     ((phi~ + phi)/2)_y = (1/beta) sin((phi - phi~)/2) ,
#
# so the step is Backlund's transformation, done algebraically.  That
# |sigma| = 1 holds identically -- phi~ stays real -- is the reality
# theory: at lambda = i beta the antilinear map h -> sigma1 conj(h)
# preserves solutions, and the initial vector h0 is chosen fixed by it
# (up to the parity sign the accumulated factors contribute).  The
# self-test measures | |sigma| - 1 | before normalising, so a broken
# choice of h0 would be caught rather than silently absorbed.
#
# ITERATION IS BIANCHI PERMUTABILITY.  The k-th step needs the previous
# frame at its own spectral point, Phi_{k-1}(i beta_k) -- a product of
# the already-known D_j(i beta_k) with the closed-form vacuum -- so N
# solitons cost N(N+1)/2 matrix products and no integration at all.
# Two equal speeds make D_j(i beta_k) singular (det = beta_j^2 -
# beta_k^2): the permutability construction degenerates, which is why
# the speeds are a strictly increasing ladder and duplicates raise.
#
# Sym's formula TELESCOPES over the steps.  dD/dlambda = I, so each step
# adds one conjugated term to G = Phi^{-1} dPhi/dlambda:
#
#     F = 2 G0 + sum_k Phi_{k-1}(1)^{-1} (2/(1 + beta_k^2)) S_k Phi_{k-1}(1)
#
# (the identity part of D^{-1} = (I + S)/(1 + beta^2) is pure trace and
# drops out of the su(2) projection).  The vacuum term 2 G0 =
# -i (x + y) sigma1 is the bare axis the lobes ride on -- the same
# linear carrier the breather's -u term is.  Every ingredient is
# algebraic, so K = -1 holds to machine precision, and the self-test
# checks the full chain: |sigma| = 1, phi solves sine-Gordon, the first
# fundamental form is the Chebyshev net dx^2 + 2 cos(phi) dx dy + dy^2,
# K = -1, and the winding of phi counts exactly N solitons.


def _m2(x, d00, d01, d10, d11):
    """Assemble a (..., 2, 2) complex matrix field from four entries
    broadcast against the grid `x`."""
    out = np.zeros(np.shape(x) + (2, 2), dtype=complex)
    out[..., 0, 0] = d00
    out[..., 0, 1] = d01
    out[..., 1, 0] = d10
    out[..., 1, 1] = d11
    return out


def _m2inv(M):
    """Inverse of a (..., 2, 2) field, via the adjugate."""
    det = (M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0])
    return _m2(M[..., 0, 0], M[..., 1, 1], -M[..., 0, 1],
               -M[..., 1, 0], M[..., 0, 0]) / det[..., None, None]


def _sg_vacuum(x, y, lam):
    """The vacuum frame Phi0 = exp((i/2)(y/lam - lam x) sigma1) =
    cos(w) I + i sin(w) sigma1.  For real lam this is in SU(2); at the
    soliton points lam = i beta the argument w is imaginary and cos/sin
    become the real cosh/sinh of the soliton phase (beta x + y/beta)/2."""
    w = 0.5 * (y / lam - lam * x)
    cw, sw = np.cos(w), np.sin(w)
    return _m2(w, cw, 1j * sw, 1j * sw, cw)


def _sg_dmat(lam, beta, sigma):
    """The Darboux factor D(lam) = lam I - S with
    S = i beta [[0, sigma], [conj(sigma), 0]]."""
    return _m2(sigma, lam, -1j * beta * sigma,
               -1j * beta * np.conj(sigma), lam)


def _sg_sigmas(x, y, betas, charges):
    """The unimodular Riccati fields sigma_k of the iterated Backlund
    transformation, one per soliton, plus the worst deviation of |sigma|
    from 1 (a measurement of the reality theory, not a knob).

    charges[k] = +-1 flips the k-th soliton between kink and antikink by
    turning the initial vector h0: sigma at the base point is e^{+-i pi/2}.
    The parity factor eps = (-1)^k compensates the sign the accumulated
    Darboux factors put in front of the antilinear symmetry
    h -> sigma1 conj(h), keeping h0 in its fixed set."""
    for j in range(len(betas)):
        for k in range(j + 1, len(betas)):
            if abs(float(betas[j]) - float(betas[k])) < 1e-9:
                raise ValueError(
                    f"two solitons share the speed {betas[j]}: the "
                    f"Darboux factor D(i beta) is singular there and "
                    f"the permutability construction degenerates.  "
                    f"Use distinct speeds.")
    sigs = []
    dev = 0.0
    for k, beta in enumerate(betas):
        lam = 1j * float(beta)
        M = _sg_vacuum(x, y, lam)
        for j in range(k):
            M = _sg_dmat(lam, betas[j], sigs[j]) @ M
        eps = 1.0 if k % 2 == 0 else -1.0
        z = np.exp(0.25j * math.pi * charges[k])
        h1 = M[..., 0, 0] * z + M[..., 0, 1] * (eps * np.conj(z))
        h2 = M[..., 1, 0] * z + M[..., 1, 1] * (eps * np.conj(z))
        sig = h1 / h2
        dev = max(dev, float(np.abs(np.abs(sig) - 1.0).max()))
        sigs.append(sig / np.abs(sig))
    return sigs, dev


def _sg_eiphi(x, y, betas, charges):
    """e^{i phi} for the N-soliton angle field, branch-free: each step
    maps e^{i phi} -> sigma^2 e^{-i phi}.  Used by the window fitter and
    the self-test (phi itself lives on a 2 pi N winding, so the smooth
    object is the exponential, not the angle)."""
    sigs, _ = _sg_sigmas(x, y, betas, charges)
    c = np.ones(np.shape(x), dtype=complex)
    for s in sigs:
        c = s * s * np.conj(c)
    return c


def _multisoliton_xyz(x, y, betas, charges):
    """Sym's formula for the N-times-transformed frame, telescoped:
    F = -i(x+y) sigma1 + sum_k Phi_{k-1}(1)^{-1} c_k S_k Phi_{k-1}(1)
    with c_k = 2/(1 + beta_k^2), projected to R^3 by v_a = Re((i/2)
    tr(F sigma_a)).  x, y are the ASYMPTOTIC (characteristic)
    coordinates of the sine-Gordon equation."""
    sigs, _ = _sg_sigmas(x, y, betas, charges)
    v1 = np.asarray(x + y, dtype=float).copy()
    v2 = np.zeros(np.shape(v1))
    v3 = np.zeros(np.shape(v1))
    P = _sg_vacuum(x, y, 1.0)
    for k, beta in enumerate(betas):
        b = float(beta)
        S = _m2(sigs[k], 0.0, 1j * b * sigs[k],
                1j * b * np.conj(sigs[k]), 0.0)
        R = _m2inv(P) @ S @ P
        c = 2.0 / (1.0 + b * b)
        # v_a = Re((i/2) tr(R sigma_a)) per Pauli matrix
        v1 += c * np.real(0.5j * (R[..., 0, 1] + R[..., 1, 0]))
        v2 += c * np.real(0.5 * (R[..., 1, 0] - R[..., 0, 1]))
        v3 += c * np.real(0.5j * (R[..., 0, 0] - R[..., 1, 1]))
        P = _sg_dmat(1.0, b, sigs[k]) @ P
    return v1, v2, v3


def _soliton_speeds(n, spread):
    """A geometric ladder of n Backlund speeds centred on 1:
    beta_k = spread^(k - (n+1)/2).  Any strictly increasing ladder
    works; the geometric one keeps every consecutive pair at the same
    ratio, which is the single knob worth exposing."""
    s = min(max(float(spread), 1.05), 4.0)
    return [s ** (k - 0.5 * (n - 1)) for k in range(n)]


def _soliton_window(betas, charges):
    """Drawing window in the LAB coordinates xi = x + y, tau = x - y.

    The two directions are bounded by different mathematics, so they
    are fitted differently:

    tau is the WRAP.  Each kink is a line soliton -- its core extends
    for ever -- but the vacuum frame Phi0(1) = exp((i/2)(y - x) sigma1)
    conjugates every soliton's Sym term, rotating it about the axis at
    exactly unit rate in tau, whatever its speed.  So one full wind of
    every helical band is one 2 pi span of tau, and a larger window
    would re-cover the same lobes with overlapping mesh rather than
    show more surface.  The window takes a whisker more than 2 pi so
    the bands visibly continue past their seam.

    xi is the EXTENT, and is measured rather than guessed: the density
    1 - cos(phi) vanishes in the vacuum (phi = 0 mod 2 pi) and reaches
    2 at a kink core, so its support along the axis is the picture.
    Fitted because it moves with both the speed ladder and the
    arrangement -- a fixed window buries a tight ladder in whisker or
    crops a wide one."""
    tau = 3.6
    gx = np.linspace(-16.0, 16.0, 301)
    gt = np.linspace(-tau, tau, 61)
    XI, TA = np.meshgrid(gx, gt, indexing='ij')
    c = _sg_eiphi(0.5 * (XI + TA), 0.5 * (XI - TA), betas, charges)
    hot = (1.0 - np.real(c)) > 0.4
    if not hot.any():
        return -4.0, 4.0, -tau, tau
    xi = XI[hot]
    pad = 1.5
    return (float(xi.min()) - pad, float(xi.max()) + pad, -tau, tau)


def _msoliton(count):
    """Build the preset surface function for the `count`-soliton
    surface.  U, V arrive NORMALISED to [-1, 1] (like the breather's)
    and are mapped to the fitted lab-coordinate window, so the mesh axes
    follow the surface's axis and girth rather than the sine-Gordon
    characteristics, which run diagonally."""
    def fn(U, V, twist=0.0, a=0.5, breather_a=0.4, amsler_angle=90.0,
           soliton_spread=1.8, soliton_cross=False):
        betas = _soliton_speeds(count, soliton_spread)
        charges = [(-1.0 if soliton_cross and k % 2 else 1.0)
                   for k in range(count)]
        x0, x1, t0, t1 = _soliton_window(betas, charges)
        xi = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * U
        ta = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * V
        return _multisoliton_xyz(0.5 * (xi + ta), 0.5 * (xi - ta),
                                 betas, charges)
    return fn


# --------------------------------------------------------------------------
# Amsler's surface: the pseudospherical surface through two straight lines
# --------------------------------------------------------------------------
# Amsler characterised the K = -1 surface containing two intersecting
# STRAIGHT LINES.  It is the Lorentz-invariant reduction of sine-Gordon:
# in asymptotic coordinates put phi(u,v) = omega(uv), and since
# phi_uv = (s omega')' with s = uv, the equation phi_uv = sin(phi)
# collapses to the ordinary differential equation
#
#     d/dr ( r domega/dr ) = sin(omega) ,          r = u v ,
#
# a Painleve III equation.  r = 0 is a regular singular point: writing
# w = r omega_r the system is omega_r = w/r, w_r = sin(omega), and the
# series through it is omega = omega_0 + sin(omega_0) r + O(r^2).  So the
# family is indexed by the single angle omega_0 = omega(0), which is
# exactly the angle at which the two straight lines cross.
#
# That the lines ARE straight falls out of the frame system rather than
# being imposed.  Along v = 0 we have phi_u = omega'(uv) v = 0, so the
# u-equations reduce to e1' = 0 with X_u = e1: the u-axis image is a
# straight line.  Along u = 0 the same happens in v, and X_v there is the
# constant vector cos(omega_0) e1 + sin(omega_0) e2 -- a second straight
# line, meeting the first at angle omega_0.
#
# The surface runs into a cuspidal edge where sin(phi) = 0, i.e. where
# omega reaches 0 or pi, because II = 2 sin(phi) du dv degenerates there.
# omega moves away from omega_0 in both directions along r, so the domain
# is fitted to keep omega strictly inside (0, pi) rather than fixed.


def _amsler_omega(omega0, r_max, n=8001):
    """Solve d/dr(r omega_r) = sin(omega) outward from the regular
    singular point r = 0 in both directions.  Returns (r, omega) with r
    increasing, ready for np.interp.

    The equation is NOT symmetric in r: replacing r by -r turns it into
    (s omega_s)_s = r sin(omega), so the halves are integrated
    separately and the solution is not an even function of r.  An even
    omega is the signature of having coded w_r = r sin(omega) instead of
    w_r = sin(omega), which is worth checking for -- it is the mistake
    that first produced K = -0.4 here instead of -1."""
    def half(sign):
        r0 = sign * 1e-7
        w = math.sin(omega0) * r0
        om = omega0 + math.sin(omega0) * r0
        rs = np.linspace(r0, sign * r_max, n)
        h = rs[1] - rs[0]
        outo = np.empty(n)
        outo[0] = om
        for k in range(n - 1):
            r = rs[k]

            def d(rr, oo, ww):
                return (ww / rr, math.sin(oo))

            k1 = d(r, om, w)
            k2 = d(r + 0.5 * h, om + 0.5 * h * k1[0], w + 0.5 * h * k1[1])
            k3 = d(r + 0.5 * h, om + 0.5 * h * k2[0], w + 0.5 * h * k2[1])
            k4 = d(r + h, om + h * k3[0], w + h * k3[1])
            om += (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            w += (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            outo[k + 1] = om
        return rs, outo

    rp, op = half(+1.0)
    rm, om_ = half(-1.0)
    r = np.concatenate([rm[::-1], [0.0], rp])
    o = np.concatenate([om_[::-1], [omega0], op])
    return r, o


def amsler_span(omega0, margin=0.08, r_max=6.0):
    """Half-width a of the (u, v) square on which omega stays inside
    (margin, pi - margin), so the patch stops short of its cuspidal
    edges.  |uv| <= a^2 over that square, hence the square root."""
    r, o = _amsler_omega(omega0, r_max, 4001)
    bad = (o > math.pi - margin) | (o < margin)
    rr = float(np.abs(r[bad]).min()) if bad.any() else r_max
    return math.sqrt(max(min(rr, r_max), 1e-6))


def _amsler(U, V, twist=0.0, a=0.5, breather_a=0.4, amsler_angle=90.0,
            soliton_spread=1.8, soliton_cross=False):
    """Amsler's surface on the NORMALISED square U, V in [-1, 1], scaled
    internally to the largest square clear of the cuspidal edges.

    The immersion comes from integrating the frame system of asymptotic
    coordinates.  For
        I  = du^2 + 2 cos(phi) du dv + dv^2 ,  II = 2 sin(phi) du dv
    and the frame e1 = X_u, e3 = N, e2 = e3 x e1, the conditions
    |X_u| = |X_v| = 1, X_uu . N = X_vv . N = 0 (the lines are asymptotic)
    and X_uv . N = sin(phi) force

        d_u:  e1' = -phi_u e2 ,  e2' = phi_u e1 + e3 ,  e3' = -e2
        d_v:  e1' = sin(phi) e3 , e2' = -cos(phi) e3 ,
              e3' = -sin(phi) e1 + cos(phi) e2
        X_u = e1 ,   X_v = cos(phi) e1 + sin(phi) e2 .

    Only the v-equations need integrating: along v = 0 the u-equations
    have phi_u = 0 and solve in closed form (e1 constant, e2 and e3
    rotating), which is precisely the straight line the surface is named
    for, so it serves as an exact initial condition."""
    om0 = math.radians(min(max(amsler_angle, 5.0), 175.0))
    span = amsler_span(om0)
    u = np.ascontiguousarray(U[:, 0]) * span
    v = np.ascontiguousarray(V[0, :]) * span
    R, OM = _amsler_omega(om0, span * span + 0.5)

    nu, nv = len(u), len(v)
    e1 = np.tile(np.array([1.0, 0.0, 0.0]), (nu, 1))
    e2 = np.stack([np.zeros(nu), np.cos(u), np.sin(u)], axis=1)
    e3 = np.stack([np.zeros(nu), -np.sin(u), np.cos(u)], axis=1)
    X = np.stack([u, np.zeros(nu), np.zeros(nu)], axis=1)

    def deriv(st, vv):
        E1, E2, E3, _ = st
        ph = np.interp(u * vv, R, OM)[:, None]
        sn, cs = np.sin(ph), np.cos(ph)
        return (sn * E3, -cs * E3, -sn * E1 + cs * E2, cs * E1 + sn * E2)

    def rk4(st, vv, h):
        k1 = deriv(st, vv)
        s2 = tuple(st[i] + 0.5 * h * k1[i] for i in range(4))
        k2 = deriv(s2, vv + 0.5 * h)
        s3 = tuple(st[i] + 0.5 * h * k2[i] for i in range(4))
        k3 = deriv(s3, vv + 0.5 * h)
        s4 = tuple(st[i] + h * k3[i] for i in range(4))
        k4 = deriv(s4, vv + h)
        return tuple(st[i] + (h / 6.0)
                     * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                     for i in range(4))

    j0 = int(np.argmin(np.abs(v)))
    out = [None] * nv
    out[j0] = X.copy()
    cur = (e1, e2, e3, X)
    for j in range(j0, nv - 1):
        cur = rk4(cur, v[j], v[j + 1] - v[j])
        out[j + 1] = cur[3].copy()
    cur = (e1, e2, e3, X)
    for j in range(j0, 0, -1):
        cur = rk4(cur, v[j], v[j - 1] - v[j])
        out[j - 1] = cur[3].copy()
    P = np.transpose(np.array(out), (1, 0, 2))
    return P[..., 0], P[..., 1], P[..., 2]


def _kuen(U, V, twist=0.0, a=0.5, breather_a=0.4,
          amsler_angle=90.0, soliton_spread=1.8, soliton_cross=False):
    denom = 1.0 + (U * np.sin(V)) ** 2
    x = 2.0 * (np.cos(U) + U * np.sin(U)) * np.sin(V) / denom
    y = 2.0 * (np.sin(U) - U * np.cos(U)) * np.sin(V) / denom
    z = np.log(np.tan(V / 2.0)) + 2.0 * np.cos(V) / denom
    return x, y, z


# (label, function, u range, v range, wrap_v)
PRESETS = {
    'PSEUDOSPHERE': ("Pseudosphere", _pseudosphere, (0.0, 3.0),
                     (0.0, 2 * math.pi), True),
    'DINI': ("Dini Surface", _dini, (0.05, 1.5),
             (0.0, 12.0 * math.pi), False),
    'KUEN': ("Kuen Surface", _kuen, (-4.5, 4.5),
             (0.08, math.pi - 0.08), False),
    # u is the NORMALISED profile parameter t of _minding_profile, not
    # arclength -- the arclength extent depends on `a`.  The spindle
    # stops a hair short of +-1 because r -> 0 there: meshing the two
    # conical points exactly would collapse a whole ring of vertices
    # onto one another and hand Blender a fan of degenerate quads.  At
    # 0.999 the remaining opening is ~0.1% of the equatorial radius.
    'MINDING_BULGE': ("Minding Bulge", _minding('MINDING_BULGE'),
                      (-1.0, 1.0), (0.0, 2 * math.pi), True),
    'MINDING_SPINDLE': ("Minding Spindle", _minding('MINDING_SPINDLE'),
                        (-0.999, 0.999), (0.0, 2 * math.pi), True),
    # u decays like 1/cosh^2(b u), so a modest u window already holds
    # the whole breather; v runs over several beats of the two
    # incommensurate frequencies 1 and w = sqrt(1 - b^2).
    'BREATHER': ("Breather Surface", _breather, (-1.0, 1.0),
                 (-1.0, 1.0), False),
    # normalised squares; _msoliton fits the lab-coordinate window to
    # the measured support of the solitons (see _soliton_window)
    'TWO_SOLITON': ("Two-Soliton Surface", _msoliton(2), (-1.0, 1.0),
                    (-1.0, 1.0), False),
    'THREE_SOLITON': ("Three-Soliton Surface", _msoliton(3), (-1.0, 1.0),
                      (-1.0, 1.0), False),
    'FOUR_SOLITON': ("Four-Soliton Surface", _msoliton(4), (-1.0, 1.0),
                     (-1.0, 1.0), False),
    # normalised square; _amsler scales it to the largest patch
    # that stays clear of the cuspidal edges for this angle
    'AMSLER': ("Amsler Surface", _amsler, (-1.0, 1.0),
               (-1.0, 1.0), False),
}

# the presets built from sine-Gordon multisolitons, in one place so the
# operator's draw() and the self-test agree on which they are
SOLITON_PRESETS = ('TWO_SOLITON', 'THREE_SOLITON', 'FOUR_SOLITON')


def _center(V):
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    return (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)


def _grid_faces(nu, nv, wrap_v):
    faces = []
    for i in range(nu - 1):
        for j in range(nv - (0 if wrap_v else 1)):
            j1 = (j + 1) % nv
            faces.append((i * nv + j, (i + 1) * nv + j,
                          (i + 1) * nv + j1, i * nv + j1))
    return faces


def build_surface(kind, ures, vres, twist=0.2, scale=1.0,
                  minding_a=0.5, breather_a=0.4,
                  amsler_angle=90.0, soliton_spread=1.8,
                  soliton_cross=False):
    label, fn, (u0, u1), (v0, v1), wrap = PRESETS[kind]
    us = np.linspace(u0, u1, ures)
    vs = (np.linspace(v0, v1, vres, endpoint=False) if wrap
          else np.linspace(v0, v1, vres))
    U, Vv = np.meshgrid(us, vs, indexing='ij')
    x, y, z = fn(U, Vv, twist, minding_a, breather_a,
                 amsler_angle, soliton_spread, soliton_cross)
    V = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    faces = _grid_faces(ures, vres, wrap)
    return _center(V) * scale, faces


def mean_curvature(V, faces):
    """Median per-vertex Gaussian curvature via angle defect / area;
    ~ -1 for these surfaces. Robust to the stretched triangles the
    parametrisations produce near their singular boundaries."""
    n = len(V)
    ang = np.zeros(n)
    area = np.zeros(n)
    deg = np.zeros(n)
    tris = []
    for f in faces:
        if len(f) == 4:
            tris.append((f[0], f[1], f[2]))
            tris.append((f[0], f[2], f[3]))
        else:
            tris.append((f[0], f[1], f[2]))
    for a, b, c in tris:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            u = V[j] - V[i]
            w = V[k] - V[i]
            cs = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)
                                 + 1e-12)
            ang[i] += math.acos(max(-1.0, min(1.0, cs)))
        ar = 0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a]))
        for i in (a, b, c):
            area[i] += ar / 3.0
            deg[i] += 1
    interior = deg >= 6
    defect = 2 * math.pi - ang
    ki = defect[interior] / np.maximum(area[interior], 1e-12)
    return float(np.median(ki)) if ki.size else 0.0


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

    class MESH_OT_hyperbolic_surface_add(bpy.types.Operator):
        """Add a smooth constant-negative-curvature surface
        (pseudosphere, Dini, Kuen, breather or multi-soliton)"""
        bl_idname = "mesh.hyperbolic_surface_add"
        bl_label = "Hyperbolic Surface"
        bl_options = {'REGISTER', 'UNDO'}
        rim: _rim.rim_prop()
        rim_thickness: _rim.rim_thickness_prop()
        rim_smooth: _rim.rim_smooth_prop()
        rim_profile: _rim.rim_profile_prop()
        rim_twist: _rim.rim_twist_prop()
        rim_reeds: _rim.rim_reeds_prop()

        preset: EnumProperty(
            name="Surface",
            description="Which constant-negative-curvature surface to build",
            items=[(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='PSEUDOSPHERE')
        ures: IntProperty(name="U Resolution", default=64, min=8,
                          max=400,
                          description="Mesh divisions along the surface's "
                                      "u direction")
        vres: IntProperty(name="V Resolution", default=96, min=8,
                          max=400,
                          description="Mesh divisions along the surface's "
                                      "v direction")
        twist: FloatProperty(
            name="Twist", default=0.2, min=0.0, max=2.0,
            description="Helical shear of Dini's surface "
                        "(curvature = -1/(1+twist^2))")
        amsler_angle: FloatProperty(
            name="Crossing Angle", default=90.0, min=10.0, max=170.0,
            description="Angle (degrees) at which Amsler's two straight "
                        "lines cross.  It is the whole parameter: the "
                        "surface is the unique K = -1 surface through "
                        "two lines meeting at this angle")
        soliton_spread: FloatProperty(
            name="Speed Ratio", default=1.8, min=1.05, max=3.0,
            description="Ratio between the speeds of consecutive "
                        "solitons: values near 1 pack the sheets "
                        "tightly, larger values pull them apart")
        soliton_style: EnumProperty(
            name="Arrangement",
            description="How the solitons are oriented against each "
                        "other",
            items=[('CHAIN', "Chain",
                    "All solitons oriented alike, strung as a chain of "
                    "lobes along the axis"),
                   ('CROSSING', "Crossing",
                    "Alternating orientations, meeting in a compact "
                    "crossing at the middle")],
            default='CHAIN')
        breather_a: FloatProperty(
            name="Breather b", default=0.4, min=0.05, max=0.95,
            description="Breather parameter b in (0, 1): small b gives "
                        "a long, loosely bound breather with many "
                        "lobes; b near 1 a short tightly bound one")
        minding_a: FloatProperty(
            name="Waist / Girth", default=0.5, min=0.05, max=3.0,
            description="Shape parameter a of the Minding surfaces: "
                        "the bulge is r = a cosh(u), so a is its waist "
                        "radius; the spindle is r = a sinh(u) and needs "
                        "a < 1, its equator sitting at sqrt(1 - a^2)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                            max=100.0,
                            description="Overall size of the surface")
        shade_smooth: BoolProperty(name="Smooth Shading", default=True,
                                   description="Shade the surface smoothly "
                                               "rather than faceted")

        def execute(self, context):
            label = PRESETS[self.preset][0]
            a = self.minding_a
            if self.preset == 'MINDING_SPINDLE' and a >= 1.0:
                # r = a sinh(u) needs |f'| = a cosh(u) <= 1, which is
                # unsatisfiable for a >= 1: there is no spindle at all.
                self.report({'WARNING'},
                            f"Minding spindle needs a < 1 (got {a:.2f}); "
                            f"clamped to 0.95")
                a = 0.95
            verts, faces = build_surface(self.preset, self.ures,
                                         self.vres, self.twist,
                                         self.scale, a, self.breather_a,
                                         self.amsler_angle,
                                         self.soliton_spread,
                                         self.soliton_style == 'CROSSING')
            me = bpy.data.meshes.new("HyperbolicSurface")
            me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.shade_smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(label, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{label}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            if self.rim:
                _ob = context.active_object
                if _ob is not None:
                    _rim.add_rim_from_object(
                        context, _ob, _ob.name,
                        self.rim_thickness, self.rim_smooth,
                        self.rim_profile, twist=self.rim_twist,
                        reeds=self.rim_reeds)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'ures')
            lay.prop(self, 'vres')
            if self.preset == 'DINI':
                lay.prop(self, 'twist')
            if self.preset == 'BREATHER':
                lay.prop(self, 'breather_a')
            if self.preset == 'AMSLER':
                lay.prop(self, 'amsler_angle')
            if self.preset in SOLITON_PRESETS:
                lay.prop(self, 'soliton_spread')
                lay.prop(self, 'soliton_style')
            if self.preset in ('MINDING_BULGE', 'MINDING_SPINDLE'):
                lay.prop(self, 'minding_a')
                if (self.preset == 'MINDING_SPINDLE'
                        and self.minding_a >= 1.0):
                    lay.label(text="Spindle needs a < 1", icon='ERROR')
            lay.prop(self, 'scale')
            lay.prop(self, 'shade_smooth')

            _rim.draw_rim(lay, self)
    def _menu_func(self, context):
        self.layout.operator("mesh.hyperbolic_surface_add",
                             icon='MESH_CAPSULE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_hyperbolic_surface_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_hyperbolic_surface_add)


def gauss_curvature_param(fn, u, v, twist=0.2, a=0.5, breather_a=0.4,
                          h=1e-3):
    """Gaussian curvature K = (LN - M^2)/(EG - F^2) computed from the
    PARAMETRISATION by central differences, at the sample points
    (u, v).

    Far sharper than the mesh angle-defect estimate in
    `mean_curvature` -- it converges like h^2 and reaches ~1e-6 -- so it
    is the right instrument for checking a closed-form surface really is
    pseudospherical.  Only valid for the closed-form presets: the
    Minding profiles are table-interpolated, and differentiating a
    linear interpolant twice returns noise."""
    def X(uu, vv):
        x, y, z = fn(uu, vv, twist, a, breather_a)
        return np.stack([x, y, z], axis=-1)

    Xu = (X(u + h, v) - X(u - h, v)) / (2 * h)
    Xv = (X(u, v + h) - X(u, v - h)) / (2 * h)
    Xuu = (X(u + h, v) - 2 * X(u, v) + X(u - h, v)) / (h * h)
    Xvv = (X(u, v + h) - 2 * X(u, v) + X(u, v - h)) / (h * h)
    Xuv = (X(u + h, v + h) - X(u + h, v - h)
           - X(u - h, v + h) + X(u - h, v - h)) / (4 * h * h)
    nrm = np.cross(Xu, Xv)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True),
                           1e-300)
    E = (Xu * Xu).sum(-1)
    F = (Xu * Xv).sum(-1)
    G = (Xv * Xv).sum(-1)
    L = (Xuu * nrm).sum(-1)
    M = (Xuv * nrm).sum(-1)
    N = (Xvv * nrm).sum(-1)
    return (L * N - M * M) / (E * G - F * F)


def _selftest():
    ok_all = True

    # 1) each surface has constant Gaussian curvature ~ -1
    #    (Dini with twist 0.2 -> -1/(1+0.2^2) = -0.96)
    want = {'PSEUDOSPHERE': -1.0, 'DINI': -1.0 / 1.04, 'KUEN': -1.0,
            'MINDING_BULGE': -1.0, 'MINDING_SPINDLE': -1.0,
            'BREATHER': -1.0, 'AMSLER': -1.0, 'TWO_SOLITON': -1.0,
            'THREE_SOLITON': -1.0, 'FOUR_SOLITON': -1.0}
    for kind in PRESETS:
        fn = PRESETS[kind][1]
        _, _, (u0, u1), (v0, v1), wrap = PRESETS[kind]
        us = np.linspace(u0, u1, 80)
        vs = (np.linspace(v0, v1, 80, endpoint=False) if wrap
              else np.linspace(v0, v1, 80))
        U, Vv = np.meshgrid(us, vs, indexing='ij')
        x, y, z = fn(U, Vv, 0.2, 0.5)
        Vraw = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
        faces = _grid_faces(80, 80, wrap)
        K = mean_curvature(Vraw, faces)
        ok = abs(K - want[kind]) < 0.05
        ok_all = ok_all and ok
        print(f"{kind:16s}: V={len(Vraw)} F={len(faces)} "
              f"meanK={K:+.3f} (want {want[kind]:+.3f}) "
              f"{'OK' if ok else 'BAD'}")

    # 2) The Minding profiles, checked against the ODE they come from
    #    rather than against a curvature estimate.  Re-deriving the
    #    arclength u from the generated (r, z) table and comparing r(u)
    #    with a cosh u / a sinh u exercises the whole substitution and
    #    quadrature chain, and is far sharper than an angle-defect
    #    measurement on a mesh.
    for kind, f_exact in (('MINDING_BULGE', np.cosh),
                          ('MINDING_SPINDLE', np.sinh)):
        for a in (0.3, 0.5, 0.8):
            t, r, z = _minding_profile(kind, a, n=4001)
            if kind == 'MINDING_SPINDLE':
                t, r, z = t[len(t) // 2:], r[len(r) // 2:], z[len(z) // 2:]
            # arclength along the meridian, measured from the waist
            # (bulge) or from the equator (spindle)
            seg = np.hypot(np.diff(r), np.diff(z))
            u = np.concatenate([[0.0], np.cumsum(seg)])
            if kind == 'MINDING_BULGE':
                u = u - np.interp(0.0, t, u)      # u = 0 at the waist
                # the table spans BOTH rims, |u| <= asinh(1/a)
                u_end = 2.0 * math.asinh(1.0 / a)
            else:
                u = u[-1] - u                     # u = 0 at the tip
                u_end = math.acosh(1.0 / a)       # tip -> equator only
            r_exact = a * f_exact(u)
            err = float(np.abs(r - r_exact).max())
            # the arclength extent must also hit the analytic |f'| <= 1
            # bound, which is what makes these strips and not planes
            span_err = abs(abs(u[0] - u[-1]) - u_end)
            ok = err < 5e-4 and span_err < 5e-4
            ok_all = ok_all and ok
            print(f"{kind:16s} a={a:.1f}: max|r - a {f_exact.__name__} u| "
                  f"= {err:.2e}  arclength {abs(u[0] - u[-1]):.5f} vs "
                  f"{u_end:.5f} {'OK' if ok else 'BAD'}")

    # 3) |f'| <= 1 everywhere on the generated profile -- outside that
    #    bound z' = sqrt(1 - f'^2) is imaginary and there is no surface.
    #    Equivalently: the profile is nowhere steeper than 45 degrees in
    #    dr/du, i.e. |dr| <= |ds| along the meridian.
    for kind in ('MINDING_BULGE', 'MINDING_SPINDLE'):
        for a in (0.2, 0.6, 0.9):
            t, r, z = _minding_profile(kind, a, n=4001)
            ds = np.hypot(np.diff(r), np.diff(z))
            slope = float(np.max(np.abs(np.diff(r)) / np.maximum(ds, 1e-15)))
            ok = slope <= 1.0 + 1e-9 and np.isfinite(z).all()
            ok_all = ok_all and ok
            print(f"{kind:16s} a={a:.1f}: max|dr/ds| = {slope:.6f} "
                  f"(must be <= 1) {'OK' if ok else 'BAD'}")

    # 4) the spindle really closes to a point at both ends, and the
    #    bulge really does not (its waist is r = a)
    t, r, _ = _minding_profile('MINDING_SPINDLE', 0.5, n=4001)
    ok = r[0] < 1e-6 and r[-1] < 1e-6 and abs(r.max()
                                              - math.sqrt(1 - 0.25)) < 1e-6
    ok_all = ok_all and ok
    print(f"spindle ends    : r(tips)={r[0]:.2e}/{r[-1]:.2e} "
          f"r(equator)={r.max():.6f} (want {math.sqrt(0.75):.6f}) "
          f"{'OK' if ok else 'BAD'}")
    t, r, _ = _minding_profile('MINDING_BULGE', 0.5, n=4001)
    ok = abs(r.min() - 0.5) < 1e-9 and abs(r.max()
                                           - math.sqrt(1.25)) < 1e-9
    ok_all = ok_all and ok
    print(f"bulge waist/rim : {r.min():.6f} / {r.max():.6f} "
          f"(want 0.500000 / {math.sqrt(1.25):.6f}) "
          f"{'OK' if ok else 'BAD'}")

    # 5) The closed-form surfaces, checked against K = -1 analytically.
    #    This is the acceptance gate for the BREATHER: a pseudospherical
    #    surface is exactly a solution of the sine-Gordon equation seen
    #    through Chebyshev coordinates, so if the closed form is right
    #    K is identically -1 and if a single coefficient is wrong it is
    #    not.  (The first candidate transcription of the breather gave
    #    median K = -0.18, which is how the slip was caught.)
    #
    #    The MEDIAN is the statistic, not the max: the breather carries
    #    genuine cusp edges where EG - F^2 -> 0 and the finite-difference
    #    estimate blows up there.  That is the surface's real singular
    #    locus, not an error in the formula.
    rng = np.random.default_rng(20260819)
    cases = [('PSEUDOSPHERE', _pseudosphere, (0.2, 3.0), (0.0, 6.2),
              0.0, -1.0),
             ('DINI twist 0', _dini, (0.2, 1.4), (0.0, 6.2), 0.0, -1.0),
             ('DINI twist 0.5', _dini, (0.2, 1.4), (0.0, 6.2), 0.5,
              -1.0 / 1.25),
             ('KUEN', _kuen, (-4.0, 4.0), (0.3, 2.8), 0.0, -1.0)]
    for b in (0.2, 0.4, 0.6, 0.8):
        cases.append((f'BREATHER b={b}', _breather, (-0.9, 0.9),
                      (-0.9, 0.9), 0.0, -1.0))
    for i, (label, fn, (u0, u1), (v0, v1), tw, K_want) in enumerate(cases):
        uu = rng.uniform(u0, u1, 500)
        vv = rng.uniform(v0, v1, 500)
        ba = (0.2, 0.4, 0.6, 0.8)[i - 4] if label.startswith('BREATHER') \
            else 0.4
        K = gauss_curvature_param(fn, uu, vv, twist=tw, breather_a=ba)
        K = K[np.isfinite(K)]
        med = float(np.median(K))
        # interquartile spread, immune to the cusp outliers
        q1, q3 = np.percentile(K, [25.0, 75.0])
        ok = abs(med - K_want) < 2e-4 and (q3 - q1) < 5e-3
        ok_all = ok_all and ok
        print(f"K(analytic) {label:16s}: median {med:+.7f} "
              f"(want {K_want:+.7f}) IQR {q3 - q1:.2e} "
              f"{'OK' if ok else 'BAD'}")

    # 5b) AMSLER.  Two checks, both of them the definition rather than a
    #     restatement of the construction.
    #
    #     (a) K = -1 on the RAW surface.  It has to be the raw one:
    #         build_surface rescales every generator into the 2 m cube,
    #         and a uniform scaling by lambda sends K to -1/lambda^2, so
    #         the meshed surface reports about -1.73 while being
    #         perfectly correct.  What survives scaling is that K is
    #         CONSTANT, and that is checked too.
    #
    #     (b) The two straight lines.  Amsler's surface is characterised
    #         by containing them, and they are not put in by hand -- they
    #         come out of the frame system, so their straightness is a
    #         real test of it.  Their crossing angle must be omega_0.
    for ang in (50.0, 90.0, 130.0):
        span = amsler_span(math.radians(ang))
        nn = 161
        t = np.linspace(-1.0, 1.0, nn)
        U, Vv = np.meshgrid(t, t, indexing='ij')
        x, y, z = _amsler(U, Vv, amsler_angle=ang)
        X = np.stack([x, y, z], axis=-1)
        h = (t[1] - t[0]) * span
        Xu = (X[2:, 1:-1] - X[:-2, 1:-1]) / (2 * h)
        Xv = (X[1:-1, 2:] - X[1:-1, :-2]) / (2 * h)
        Xuu = (X[2:, 1:-1] - 2 * X[1:-1, 1:-1] + X[:-2, 1:-1]) / h ** 2
        Xvv = (X[1:-1, 2:] - 2 * X[1:-1, 1:-1] + X[1:-1, :-2]) / h ** 2
        Xuv = (X[2:, 2:] - X[2:, :-2] - X[:-2, 2:]
               + X[:-2, :-2]) / (4 * h * h)
        nv_ = np.cross(Xu, Xv)
        nv_ = nv_ / np.maximum(np.linalg.norm(nv_, axis=-1, keepdims=True),
                               1e-300)
        E = (Xu * Xu).sum(-1)
        F = (Xu * Xv).sum(-1)
        G = (Xv * Xv).sum(-1)
        L = (Xuu * nv_).sum(-1)
        M = (Xuv * nv_).sum(-1)
        N = (Xvv * nv_).sum(-1)
        K = (L * N - M * M) / (E * G - F * F)
        K = K[np.isfinite(K)]
        med = float(np.median(K))
        q1, q3 = np.percentile(K, [25.0, 75.0])
        ok = abs(med + 1.0) < 5e-3 and (q3 - q1) < 1e-4
        ok_all = ok_all and ok
        print(f"AMSLER angle={ang:5.1f}: span={span:.4f} K median "
              f"{med:+.6f} IQR {q3 - q1:.1e} {'OK' if ok else 'BAD'}")

        # the two asymptotic lines through the origin must be straight,
        # and must cross at omega_0
        mid = nn // 2
        line_u = X[:, mid, :]
        line_v = X[mid, :, :]
        def straightness(P):
            d = P[1:] - P[:-1]
            d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True),
                               1e-300)
            return float(np.abs(d - d[len(d) // 2]).max())
        su, sv = straightness(line_u), straightness(line_v)
        du = line_u[-1] - line_u[0]
        dv = line_v[-1] - line_v[0]
        cosang = float(np.dot(du, dv)
                       / (np.linalg.norm(du) * np.linalg.norm(dv)))
        got = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        ok = su < 1e-6 and sv < 1e-6 and abs(got - ang) < 1e-3
        ok_all = ok_all and ok
        print(f"   straight lines: deviation {su:.1e} / {sv:.1e}, "
              f"crossing angle {got:.4f} deg (want {ang:.1f}) "
              f"{'OK' if ok else 'BAD'}")

    # 5c) MULTISOLITON surfaces (iterated Backlund + Sym).  Five
    #     instruments, each aimed at a different failure mode:
    #
    #     (a) | |sigma| - 1 | BEFORE normalisation.  The reality of the
    #         Backlund transform is a theorem, not an enforcement: if
    #         the initial vectors h0 broke the antilinear symmetry the
    #         Riccati field would leave the unit circle and phi would
    #         go complex.  Measuring the deviation catches that;
    #         normalising without measuring would absorb it silently.
    #     (b) the angle field solves sine-Gordon.  Branch-free, via
    #         c = e^{i phi}: phi_xy - sin(phi) =
    #         Im(c_xy/c - c_x c_y/c^2) - Im(c), so the 2 pi N winding
    #         of phi never touches an arctan.
    #     (c) the first fundamental form is the Chebyshev net
    #         E = G = 1, F = cos(phi) -- Theorem 11's (8.16) at
    #         lambda = 1, and the sharpest full-pipeline check there
    #         is, since it ties the SURFACE derivatives to the ANGLE
    #         field point by point.
    #     (d) K = -1 through the operator-facing preset function,
    #         window fitting included.
    #     (e) the winding of phi along a line crossing every soliton,
    #         and the count of kink cores along it.  An N-soliton that
    #         silently collapsed to fewer solitons cannot pass either;
    #         a CROSSING pair is kink + antikink, so its winding is 0
    #         and the core count still says 2.
    #
    #     All derivatives are central differences at random INTERIOR
    #     points (the endpoint-difference lesson: boundaries report
    #     their own truncation error, not the surface's).
    rng2 = np.random.default_rng(20260907)
    for n, cross in ((2, False), (2, True), (3, False), (4, False)):
        betas = _soliton_speeds(n, 1.8)
        charges = [(-1.0 if cross and k % 2 else 1.0) for k in range(n)]
        tag = f"{n}-soliton {'crossing' if cross else 'chain':8s}"

        gx = np.linspace(-6.0, 6.0, 41)
        X0, Y0 = np.meshgrid(gx, gx, indexing='ij')
        _, dev = _sg_sigmas(X0, Y0, betas, charges)

        h = 1e-3
        xr = rng2.uniform(-2.0, 2.0, 400)
        yr = rng2.uniform(-2.0, 2.0, 400)

        def C(xx, yy, _b=betas, _c=charges):
            return _sg_eiphi(xx, yy, _b, _c)

        c0 = C(xr, yr)
        cx = (C(xr + h, yr) - C(xr - h, yr)) / (2 * h)
        cy = (C(xr, yr + h) - C(xr, yr - h)) / (2 * h)
        cxy = (C(xr + h, yr + h) - C(xr + h, yr - h)
               - C(xr - h, yr + h) + C(xr - h, yr - h)) / (4 * h * h)
        resid = float(np.abs(np.imag(cxy / c0 - cx * cy / (c0 * c0))
                             - np.imag(c0)).max())

        def XYZ(xx, yy, _b=betas, _c=charges):
            return np.stack(_multisoliton_xyz(xx, yy, _b, _c), axis=-1)

        Xu = (XYZ(xr + h, yr) - XYZ(xr - h, yr)) / (2 * h)
        Xv = (XYZ(xr, yr + h) - XYZ(xr, yr - h)) / (2 * h)
        E = (Xu * Xu).sum(-1)
        Fc = (Xu * Xv).sum(-1)
        G = (Xv * Xv).sum(-1)
        cheb = max(float(np.abs(E - 1.0).max()),
                   float(np.abs(G - 1.0).max()),
                   float(np.abs(Fc - np.real(c0)).max()))

        key = {2: 'TWO_SOLITON', 3: 'THREE_SOLITON',
               4: 'FOUR_SOLITON'}[n]

        def fnk(uu, vv, tw, aa, ba, _f=PRESETS[key][1], _c=cross):
            return _f(uu, vv, tw, aa, ba, soliton_cross=_c)

        uu = rng2.uniform(-0.9, 0.9, 400)
        vv = rng2.uniform(-0.9, 0.9, 400)
        K = gauss_curvature_param(fnk, uu, vv)
        K = K[np.isfinite(K)]
        med = float(np.median(K))
        q1, q3 = np.percentile(K, [25.0, 75.0])

        t = np.linspace(-60.0, 30.0, 24001)
        cline = _sg_eiphi(t, 0.6 * t + 12.0, betas, charges)
        wind = float(np.angle(cline[1:] / cline[:-1]).sum()
                     / (2.0 * math.pi))
        w_want = 0.0 if cross and n % 2 == 0 else float(n)
        hot = (1.0 - np.real(cline)) > 1.5
        cores = int(np.count_nonzero(hot[1:] & ~hot[:-1])
                    + (1 if hot[0] else 0))

        ok = (dev < 1e-9 and resid < 1e-4 and cheb < 1e-4
              and abs(med + 1.0) < 1e-4 and (q3 - q1) < 1e-5
              and abs(abs(wind) - w_want) < 1e-2 and cores == n)
        ok_all = ok_all and ok
        print(f"{tag}: |sigma|-1 {dev:.1e}, sine-Gordon {resid:.1e}, "
              f"Chebyshev {cheb:.1e} {'OK' if ok else 'BAD'}")
        print(f"   K median {med:+.7f} IQR {q3 - q1:.1e}, winding "
              f"{wind:+.3f} (want {w_want:.0f}), cores {cores}/{n} "
              f"{'OK' if ok else 'BAD'}")

    # equal Backlund speeds make the permutability step singular and
    # must be refused, not approximated
    try:
        _sg_sigmas(np.zeros((2, 2)), np.zeros((2, 2)),
                   [1.0, 1.0], [1.0, 1.0])
        guard = False
    except ValueError:
        guard = True
    ok_all = ok_all and guard
    print(f"soliton guards  : duplicate speed refused "
          f"{'OK' if guard else 'BAD'}")

    # 6) the mesher runs for every preset and produces finite geometry
    for kind in PRESETS:
        V, F = build_surface(kind, 40, 48, 0.2, 1.0, 0.5)
        ok = len(V) == 40 * 48 and len(F) > 0 and np.isfinite(V).all()
        ok_all = ok_all and ok
        print(f"build {kind:16s}: V={len(V)} F={len(F)} "
              f"{'OK' if ok else 'BAD'}")

    assert ok_all
    print("hyperbolic surface standalone tests passed")
