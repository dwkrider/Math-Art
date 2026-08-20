# CMC-1 trinoids in hyperbolic 3-space, in hypergeometric functions.
#
# Numeric core for the TRINOID mode of `mesh.bryant_surface_add`
# (math_art/bryant_generator.py).  Numpy only, no bpy: everything here
# imports and self-tests headlessly.
#
# A trinoid is a complete CMC-1 (Bryant) surface of genus zero with
# three catenoidal ends.  Bobenko, Pavlyukevich and Springborn [BPS]
# construct all of them explicitly:
#
#   1. GLOBAL SPINOR DATA.  The immersion is F = Psi Psi* (their (7.1),
#      the same Hermitian-matrix form as the rest of bryant_generator),
#      where Psi solves Psi_z = [[PQ, P^2], [-Q^2, -PQ]] Psi (their
#      (3.12)) with spinors P, Q holomorphic on the trinoid's OWN
#      Riemann surface CP^1 \ {0, 1, inf}:
#          P = p0/z + p1/(z-1) + p_inf,   Q = q0/z + q1/(z-1) + q_inf.
#
#   2. REDUCTION TO A FUCHSIAN SYSTEM (their Proposition 1).  A gauge
#      Psi = D Phi with the explicit matrix D of (5.4) turns (3.12)
#      into Phi_z = (A_0/z + A_1/(z-1)) Phi with constant traceless
#      A_0 = diag(alpha, -alpha), A_1 = [[beta, gamma],[delta, -beta]].
#      Here the linear-system part of D (alpha_i, beta_i) is solved
#      numerically from the determinant condition of their proof, the
#      parameter k as a root of the quadratic that kills the 1/z^2
#      term, and A_0, A_1 are read off the gauged system by sampling --
#      the same coefficients as their (5.6), but derived rather than
#      transcribed, and verified Fuchsian at a third point.
#
#   3. HYPERGEOMETRIC SOLUTION (their Theorem 4).  The canonical
#      solutions Phi^(0), Phi^(1), Phi^(inf) at the three singular
#      points are 2x2 matrices of Gauss 2F1 functions (their
#      (6.5)-(6.7)) with a = alpha+tau+rho, b = alpha+tau-rho,
#      c = 2 alpha, where alpha = 1/2 - sqrt(d0), tau = sqrt(d1),
#      rho = sqrt(d_inf) and
#          d0    = 1/4 + <pq>_10 + <pq>_0inf,
#          d1    = 1/4 + <pq>_10 + <pq>_1inf,
#          d_inf = 1/4 + <pq>_0inf + <pq>_1inf,
#      <pq>_ij = p_i q_j - p_j q_i.  Each converges near its own end;
#      the connection matrices E_nu (Phi^(0) = Phi^(nu) E_nu) glue them.
#      E_1 and E_inf are computed numerically by matching at points
#      where two series both converge -- and E_1 is cross-checked
#      against their closed form (6.9) in the self-test.  Phi^(inf)
#      carries an extra branch cut on (0,1) from its 2F1(...; 1/z)
#      entries, so E_inf takes one value in the upper half-plane and
#      another in the lower; both are stored.
#
#   4. UNITARIZATION (their Theorem 6).  F is well defined on the
#      thrice-punctured sphere iff the monodromy group is unitarizable:
#      (i) alpha, tau, rho real, (ii) sin(pi a) sin(pi b) sin(pi(a-c))
#      sin(pi(b-c)) < 0 -- equivalently their Proposition 2: the
#      fractional parts |{sqrt(d_nu)}| lie in the tetrahedral region
#      (7.3).  The unitarizing R = diag(r, 1/r) is found here from the
#      monodromy matrix itself, r^4 = -conj(M1_01)/M1_10 (which must
#      come out real and positive -- this IS condition (ii) in numeric
#      form), and cross-checked against their gamma-function formula
#      (7.2) in the self-test.  Inadmissible parameters are REFUSED.
#
#   5. NORMALISATION (their (7.5)).  The residual isometry freedom is
#      fixed by placing the three ends on the sphere at infinity at
#      (-1/2, 0, -sqrt(3)/2), (1, 0, 0), (-1/2, 0, sqrt(3)/2):
#          p0 = (2 - sqrt 3) q0,  p1 = -q1,  p_inf = (2 + sqrt 3) q_inf,
#      making (d0, d1, d_inf) a complete set of moduli -- the trinoids
#      are a three-parameter family.  d0 = d1 = d_inf gives the
#      symmetric trinoids; the paper reports a threshold D_0 ~ 0.2332
#      below which they are embedded and above which they are not, and
#      the self-test reproduces that transition on the mesh.
#
#   6. DRAWING PARAMETERISATION (their Section 7, construction recipe).
#      The z-plane splits into three domains, one per end, via
#          z(w) = ((z1 - zinf)/(z1 - z0)) (w - z0)/(w - zinf),
#          w = ((1 + wt)/(1 - wt))^(2/3) z_j,   |wt| <= 1,
#      z0 = e^(i pi/6), z1 = e^(i 5pi/6), zinf = e^(i 3pi/2).  wt = 0
#      is the end, |wt| = 1 the domain boundary, wt = +-1 the two
#      umbilic points e^(-+ i pi/3) shared by all three domains.  The
#      three annular grids are welded along their boundary rays into
#      one mesh of the correct topology (a sphere minus three disks).
#
# References:
# - A. I. Bobenko, T. V. Pavlyukevich, B. A. Springborn, "Hyperbolic
#   constant mean curvature one surfaces: spinor representation and
#   trinoids in hypergeometric functions", Math. Z. 245 (2003), 63-91;
#   arXiv:math/0206021.  Everything above.
# - R. L. Bryant, "Surfaces of mean curvature one in hyperbolic space",
#   Asterisque 154-155 (1987), 321-347.  The representation itself.
# - M. Umehara, K. Yamada, "Complete surfaces of constant mean
#   curvature 1 in the hyperbolic 3-space", Ann. of Math. 137 (1993),
#   611-638.  The classification the explicit formulas realise.

import math

import numpy as np

from .minsurf.elliptic import hyp2f1, cgamma

TAU = 2.0 * math.pi
SQRT3 = math.sqrt(3.0)

# (7.5): p_j = END_MULT[j] * q_j pins the ends to the three ideal points
END_MULT = (2.0 - SQRT3, -1.0, 2.0 + SQRT3)
# where those ends land on the unit sphere at infinity, in (x1, x2, x3)
END_POINTS = (np.array([-0.5, 0.0, -0.5 * SQRT3]),
              np.array([1.0, 0.0, 0.0]),
              np.array([-0.5, 0.0, 0.5 * SQRT3]))

# the w-plane anchors of the drawing parameterisation
_Z0 = complex(np.exp(1j * math.pi / 6.0))
_Z1 = complex(np.exp(1j * 5.0 * math.pi / 6.0))
_ZINF = complex(np.exp(1j * 3.0 * math.pi / 2.0))
_ZCROSS = (_Z1 - _ZINF) / (_Z1 - _Z0)          # = e^(-i pi/3), an umbilic
_WCEN = (_Z0, _Z1, _ZINF)


# --------------------------------------------------------------------------
# admissibility -- Theorem 6 via Proposition 2
# --------------------------------------------------------------------------

def _frac_half(x):
    """The fractional part into [-1/2, 1/2), as in [BPS] Section 7."""
    return x - math.floor(x + 0.5)


def admissible(d0, d1, dinf):
    """(ok, reason): is (d0, d1, d_inf) the data of an actual trinoid?

    Implements [BPS] Proposition 2 -- the unitarizable-monodromy region
    -- plus the genericity condition (6.3) (2 alpha, 2 tau, 2 rho not
    integers; the half-integer cases need different, degenerate
    formulas) and non-degeneracy of the normalised spinor data.  The
    region test is elementary trigonometry on sqrt(d); the compile step
    additionally verifies, numerically, that the unitarising r^4 comes
    out real and positive, which is Theorem 6 (ii) in the form 'the
    formula must have a real solution'."""
    ds = (float(d0), float(d1), float(dinf))
    for nm, d in zip("01i", ds):
        if not d > 0.0:
            return False, f"d{nm} = {d:g} must be positive"
        if abs(2.0 * math.sqrt(d) - round(2.0 * math.sqrt(d))) < 1e-3:
            return False, (f"2 sqrt(d{nm}) = {2*math.sqrt(d):g} is (nearly) "
                           "an integer: a degenerate, resonant end")
    A = 0.5 * (ds[0] + ds[1] - ds[2] - 0.25)
    B = 0.5 * (ds[0] + ds[2] - ds[1] - 0.25)
    C = 0.5 * (ds[1] + ds[2] - ds[0] - 0.25)
    for nm, v in (("<pq>_10", A), ("<pq>_0inf", B), ("<pq>_1inf", C)):
        if abs(v) < 1e-9:
            return False, f"{nm} = 0: degenerate spinor data"
    D1, D2, D3 = (abs(_frac_half(math.sqrt(d))) for d in ds)
    m = 1e-9
    if not (D1 + D2 + D3 > 0.5 + m):
        return False, (f"|{{sqrt d}}| sum {D1+D2+D3:.4f} <= 1/2: monodromy "
                       "not unitarizable ([BPS] Prop. 2)")
    for lhs, txt in ((D1 + D2 - D3, "d0,d1 vs dinf"),
                     (D1 + D3 - D2, "d0,dinf vs d1"),
                     (D2 + D3 - D1, "d1,dinf vs d0")):
        if not (lhs < 0.5 - m):
            return False, (f"|{{sqrt d}}| condition fails ({txt}): "
                           "monodromy not unitarizable ([BPS] Prop. 2)")
    return True, "admissible"


# --------------------------------------------------------------------------
# branch-consistent powers
# --------------------------------------------------------------------------
# z-powers cut along (-inf, 0] (principal branch); (z-1)-powers cut
# along [1, inf) with arg(z-1) in [0, 2 pi).  All formulas below use the
# SAME two cuts, so each canonical solution is single-valued off them
# and the connection matrices are honest constants.  F = Psi Psi* is
# continuous ACROSS the cuts because every branch jump is a unitary
# right factor (that is Theorem 6 at work) -- verified in the self-test.

def _zpow(z, s):
    return np.exp(s * np.log(z))


def _wpow(z, s):
    w = z - 1.0
    return np.exp(s * (np.log(np.abs(w)) + 1j * (np.angle(w) % TAU)))


# --------------------------------------------------------------------------
# the compiled surface
# --------------------------------------------------------------------------

class TrinoidSurface:
    """All z-independent data of one trinoid: spinor residues, the
    Fuchsian coefficients, connection matrices, and the unitarising R.
    Compiling raises ValueError on inadmissible or degenerate data."""

    def __init__(self, d0, d1, dinf):
        ok, why = admissible(d0, d1, dinf)
        if not ok:
            raise ValueError(f"not a trinoid: {why}")
        self.d = (float(d0), float(d1), float(dinf))

        # ---- spinor data under the normalisation (7.5) ----
        A = 0.5 * (d0 + d1 - dinf - 0.25)      # <pq>_10
        B = 0.5 * (d0 + dinf - d1 - 0.25)      # <pq>_0inf
        C = 0.5 * (d1 + dinf - d0 - 0.25)      # <pq>_1inf
        self.brackets = (A, B, C)
        m0, m1, mi = END_MULT
        q0 = complex(np.sqrt(complex(
            (A / (m1 - m0)) * (B / (m0 - mi)) / (C / (m1 - mi)))))
        q1 = (A / (m1 - m0)) / q0
        qi = (B / (m0 - mi)) / q0
        self.q = (q0, q1, qi)
        self.p = (m0 * q0, m1 * q1, mi * qi)

        # ---- Fuchsian exponents ----
        self.alpha = 0.5 - math.sqrt(d0)
        self.tau = math.sqrt(d1)
        self.rho = math.sqrt(dinf)
        self.a = self.alpha + self.tau + self.rho
        self.b = self.alpha + self.tau - self.rho
        self.c = 2.0 * self.alpha
        Delta = A * B + A * C + B * C
        self.Delta = Delta

        # ---- gauge D = B(z) C(z) M, coefficients solved not transcribed
        p0, p1, pi_ = self.p
        Amat = np.array([[qi, pi_, 0, 0],
                         [0, 0, q0, p0],
                         [q1, p1, q1, p1],
                         [q0 + q1, p0 + p1, qi, pi_]], dtype=complex)
        self.a1, self.a2, self.b1, self.b2 = np.linalg.solve(
            Amat, np.array([0.0, 0.0, 0.0, 1.0], dtype=complex))
        # k kills the 1/z^2 term: root of  ahat k^2 +
        # (Delta^2 - 2 Delta A B) k - Delta^2 (A + B) = 0, choosing the
        # root whose gauged residue at 0 is [[alpha, 0], [mu, -alpha]]
        ahat = Delta * Delta + A * B * C
        roots = np.roots([ahat, Delta * Delta - 2.0 * Delta * A * B,
                          -Delta * Delta * (A + B)])
        best = None
        for k in roots:
            A0h, A1h, res = self._residues(complex(k))
            score = abs(A0h[0, 1]) + abs(A0h[0, 0] - self.alpha) + res
            if best is None or score < best[0]:
                best = (score, complex(k), A0h, A1h, res)
        self.kscore, self.k, A0h, A1h, self.fuchs_res = best
        if self.kscore > 1e-8:
            raise ValueError("Fuchsian reduction failed (residue "
                             f"structure off by {self.kscore:.2e}); "
                             "the parameters are too degenerate")
        self.mu = A0h[1, 0]
        self.Mg = np.array([[2.0 * self.alpha / self.mu, 0.0],
                            [1.0, 1.0]], dtype=complex)
        Minv = np.linalg.inv(self.Mg)
        self.A0f = Minv @ A0h @ self.Mg
        self.A1f = Minv @ A1h @ self.Mg
        self.beta = self.A1f[0, 0]
        self.gamma = self.A1f[0, 1]
        self.delta = self.A1f[1, 0]
        # eigenvalue identities (6.2) tie the reduction to the exponents
        e1 = abs(self.beta ** 2 + self.gamma * self.delta - self.tau ** 2)
        e2 = abs((self.alpha + self.beta) ** 2 + self.gamma * self.delta
                 - self.rho ** 2)
        if max(e1, e2) > 1e-8:
            raise ValueError(f"exponent identities violated ({e1:.1e}, "
                             f"{e2:.1e}): degenerate parameters")

        # ---- connection matrices, from overlap points ----
        z1 = 0.5 + 0.4j
        self.E1 = np.linalg.solve(self._phi1(np.array([z1]))[0],
                                  self._phi0(np.array([z1]))[0])
        zu, zd = 1.2 + 0.55j, 1.2 - 0.55j       # Phi^(inf) cuts (0,1) too
        self.Einf_up = np.linalg.solve(
            self._phiinf(np.array([zu]))[0],
            self._phi1(np.array([zu]))[0]) @ self.E1
        self.Einf_dn = np.linalg.solve(
            self._phiinf(np.array([zd]))[0],
            self._phi1(np.array([zd]))[0]) @ self.E1

        # ---- unitarising R = diag(r, 1/r) from the monodromy at z = 1
        L1 = np.diag([np.exp(2j * math.pi * self.tau),
                      np.exp(-2j * math.pi * self.tau)])
        Mo1 = np.linalg.solve(self.E1, L1 @ self.E1)
        self.M1 = Mo1
        r4 = -np.conj(Mo1[0, 1]) / Mo1[1, 0]
        if not (abs(r4.imag) < 1e-6 * abs(r4) and r4.real > 0.0):
            raise ValueError(f"monodromy not unitarizable (r^4 = {r4:g}); "
                             "Theorem 6 (ii) fails numerically")
        self.r4 = float(r4.real)
        r = self.r4 ** 0.25
        self.R = np.diag([r, 1.0 / r]).astype(complex)
        U = np.diag([1.0 / r, r]) @ Mo1 @ self.R
        self.unit_res = float(np.abs(U @ U.conj().T - np.eye(2)).max())
        if self.unit_res > 1e-8:
            raise ValueError("unitarization residual "
                             f"{self.unit_res:.2e} -- refusing")

        # ---- normalise det Psi to 1 (det D Phi is a nonzero constant:
        # A_0, A_1 are traceless, so det Phi is constant, and
        # det D = det M is too; F = Psi Psi* then has det exactly 1)
        z = np.array([0.3 + 0.2j])
        self.scale = 1.0 / np.sqrt(np.linalg.det(
            self._D(z)[0] @ self._phi0(z)[0]))

    # ---- pieces of the gauge ----
    def _PQ(self, z):
        p0, p1, pi_ = self.p
        q0, q1, qi = self.q
        return (p0 / z + p1 / (z - 1.0) + pi_,
                q0 / z + q1 / (z - 1.0) + qi)

    def _B(self, z):
        P, Q = self._PQ(z)
        B = np.empty(z.shape + (2, 2), dtype=complex)
        B[..., 0, 0] = P
        B[..., 0, 1] = self.a1 * z + self.b1
        B[..., 1, 0] = -Q
        B[..., 1, 1] = self.a2 * z + self.b2
        return B

    def _D(self, z):
        s = _wpow(z, 0.5)
        C = np.zeros(z.shape + (2, 2), dtype=complex)
        C[..., 0, 0] = s
        C[..., 1, 0] = self.k / (z * s)
        C[..., 1, 1] = 1.0 / s
        return self._B(z) @ C @ self.Mg

    def _residues(self, k):
        """A0, A1 of the gauged system, sampled at two points and
        verified Fuchsian at a third."""
        def ahat_at(zc):
            z = np.array([zc])
            p0, p1, pi_ = self.p
            q0, q1, qi = self.q
            Amat = np.empty((2, 2), dtype=complex)
            P, Q = self._PQ(np.array([zc]))
            P, Q = P[0], Q[0]
            Amat[0, 0], Amat[0, 1] = P * Q, P * P
            Amat[1, 0], Amat[1, 1] = -Q * Q, -P * Q
            Bm = self._B(z)[0]
            Pp = -p0 / zc ** 2 - p1 / (zc - 1.0) ** 2
            Qp = -q0 / zc ** 2 - q1 / (zc - 1.0) ** 2
            Bp = np.array([[Pp, self.a1], [-Qp, self.a2]], dtype=complex)
            Atil = np.linalg.solve(Bm, Amat @ Bm - Bp)
            s = np.sqrt(zc - 1.0)          # branch cancels here
            Cm = np.array([[s, 0.0], [k / (zc * s), 1.0 / s]],
                          dtype=complex)
            Cp = np.array(
                [[0.5 / s, 0.0],
                 [k * (-1.0 / (zc * zc * s)
                       - 0.5 / (zc * s * (zc - 1.0))),
                  -0.5 / (s * (zc - 1.0))]], dtype=complex)
            return np.linalg.solve(Cm, Atil @ Cm - Cp)

        za, zb, zc = 2.7 + 1.3j, -1.4 + 0.8j, 0.4 + 2.1j
        Ha, Hb = ahat_at(za), ahat_at(zb)
        Mc = np.array([[1.0 / za, 1.0 / (za - 1.0)],
                       [1.0 / zb, 1.0 / (zb - 1.0)]], dtype=complex)
        stacked = np.linalg.solve(
            Mc, np.stack([Ha.ravel(), Hb.ravel()]))
        A0 = stacked[0].reshape(2, 2)
        A1 = stacked[1].reshape(2, 2)
        res = float(np.abs(A0 / zc + A1 / (zc - 1.0)
                           - ahat_at(zc)).max())
        return A0, A1, res

    # ---- canonical solutions (6.5)-(6.7), vectorized over z ----
    def _phi0(self, z):
        a, b, c = self.a, self.b, self.c
        al, ta = self.alpha, self.tau
        w = _wpow(z, ta)
        M = np.empty(z.shape + (2, 2), dtype=complex)
        M[..., 0, 0] = (-(2 * al + 1) / self.delta * _zpow(z, al) * w
                        * hyp2f1(a, b, c, z))
        M[..., 0, 1] = (_zpow(z, 1 - al) * w
                        * hyp2f1(a - c + 1, b - c + 1, 2 - c, z))
        M[..., 1, 0] = (_zpow(z, 1 + al) * w
                        * hyp2f1(a + 1, b + 1, c + 2, z))
        M[..., 1, 1] = ((2 * al - 1) / self.gamma * _zpow(z, -al) * w
                        * hyp2f1(a - c, b - c, -c, z))
        return M

    def _phi1(self, z):
        a, b, c = self.a, self.b, self.c
        al, ta = self.alpha, self.tau
        be, g, d = self.beta, self.gamma, self.delta
        wp = _wpow(z, ta)
        wm = _wpow(z, -ta)
        M = np.empty(z.shape + (2, 2), dtype=complex)
        M[..., 0, 0] = ((be + ta) / d * _zpow(z, al) * wp
                        * hyp2f1(a, b, a + b - c + 1, 1.0 - z))
        M[..., 0, 1] = (_zpow(z, al) * wm
                        * hyp2f1(c - a, c - b, c - a - b + 1, 1.0 - z))
        M[..., 1, 0] = (_zpow(z, -al) * wp
                        * hyp2f1(a - c, b - c, a + b - c + 1, 1.0 - z))
        M[..., 1, 1] = (-(be + ta) / g * _zpow(z, -al) * wm
                        * hyp2f1(-a, -b, c - a - b + 1, 1.0 - z))
        return M

    def _phiinf(self, z):
        a, b, c = self.a, self.b, self.c
        ta, ro = self.tau, self.rho
        be, g = self.beta, self.gamma
        w = _wpow(z, ta)
        zm = _zpow(z, -ta - ro)
        zp = _zpow(z, -ta + ro)
        M = np.empty(z.shape + (2, 2), dtype=complex)
        M[..., 0, 0] = (g * (c - a) / (a * (be + ta)) * zm * w
                        * hyp2f1(a, a - c + 1, a - b + 1, 1.0 / z))
        M[..., 0, 1] = zp * w * hyp2f1(b, b - c + 1, b - a + 1, 1.0 / z)
        M[..., 1, 0] = zm * w * hyp2f1(a + 1, a - c, a - b + 1, 1.0 / z)
        M[..., 1, 1] = (b * (be + ta) / (g * (c - b)) * zp * w
                        * hyp2f1(b + 1, b - c, b - a + 1, 1.0 / z))
        return M

    # ---- the immersion ----
    def psi(self, z):
        """Psi with det = 1 on a flat complex ndarray z, choosing per
        point the canonical solution that converges best there."""
        z = np.asarray(z, dtype=complex).ravel()
        Psi = np.empty(z.shape + (2, 2), dtype=complex)
        which = np.argmin(np.stack(
            [np.abs(z), np.abs(1.0 - z), 1.0 / np.abs(z)]), axis=0)
        for rep in (0, 1, 2):
            m = which == rep
            if not np.any(m):
                continue
            zz = z[m]
            if rep == 0:
                Psi[m] = self._D(zz) @ self._phi0(zz) @ self.R
            elif rep == 1:
                Psi[m] = self._D(zz) @ self._phi1(zz) @ self.E1 @ self.R
            else:
                up = zz.imag >= 0.0
                Ph = self._phiinf(zz)
                Dm = self._D(zz)
                out = np.empty_like(Ph)
                if np.any(up):
                    out[up] = Dm[up] @ Ph[up] @ self.Einf_up @ self.R
                if np.any(~up):
                    out[~up] = Dm[~up] @ Ph[~up] @ self.Einf_dn @ self.R
                Psi[m] = out
        return self.scale * Psi

    def hyperboloid(self, z):
        """Minkowski points (x0, x1, x2, x3), <x,x> = -1, x0 > 0, for a
        complex ndarray z; shape z.shape + (4,)."""
        z = np.asarray(z, dtype=complex)
        Psi = self.psi(z.ravel())
        F = Psi @ np.conj(np.swapaxes(Psi, -1, -2))
        x0 = 0.5 * np.real(F[..., 0, 0] + F[..., 1, 1])
        x3 = 0.5 * np.real(F[..., 0, 0] - F[..., 1, 1])
        X = np.stack([x0, np.real(F[..., 0, 1]), np.imag(F[..., 0, 1]),
                      x3], axis=-1)
        return X.reshape(z.shape + (4,))


_CACHE = {}


def get_surface(d0, d1, dinf):
    """Compiled TrinoidSurface, cached on the rounded parameter triple."""
    key = (round(float(d0), 9), round(float(d1), 9), round(float(dinf), 9))
    if key not in _CACHE:
        if len(_CACHE) > 8:
            _CACHE.clear()
        _CACHE[key] = TrinoidSurface(*key)
    return _CACHE[key]


# --------------------------------------------------------------------------
# drawing parameterisation and mesh
# --------------------------------------------------------------------------

def z_of_wt(wt, domain):
    """[BPS] Section 7 recipe: unit-disk coordinate wt of domain
    0/1/2 (= ends z = 0, 1, inf) -> the z-plane."""
    wt = np.asarray(wt, dtype=complex)
    w = ((1.0 + wt) / (1.0 - wt)) ** (2.0 / 3.0) * _WCEN[domain]
    return _ZCROSS * (w - _Z0) / (w - _ZINF)


def trinoid_points(u, v, d0, d1, dinf, domain=1):
    """Hyperboloid points on the log-polar chart wt = exp(u + iv) of one
    domain -- the (u, v) parameterisation the mean-curvature gate
    differentiates.  u < 0 (u -> -inf is the end), v the angle."""
    S = get_surface(d0, d1, dinf)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    wt = np.exp(u + 1j * v)
    return S.hyperboloid(z_of_wt(wt, domain))


def _to_model(X, model):
    if model == 'POINCARE':
        return X[..., 1:] / (1.0 + X[..., :1])
    if model == 'KLEIN':
        return X[..., 1:] / np.maximum(X[..., :1], 1e-12)
    if model == 'HYPERBOLOID':
        return X[..., 1:].copy()
    raise ValueError(f"unknown model {model!r}")


def build_trinoid_mesh(d0, d1, dinf, nr=28, nth=96, rmin=0.03,
                       model='POINCARE'):
    """Welded three-domain trinoid mesh.

    Returns (verts (N,3), faces list of quads, info dict).  Radial rows
    run log-spaced from the domain boundary |wt| = 1 down to rmin (the
    ends); angular columns sample midpoints so no vertex lands exactly
    on the umbilic corner points wt = +-1, where the hypergeometric
    series converge slowest.  The three annuli share their outer rings:
    the upper half-arc of domain j is the SAME curve as the lower
    half-arc of domain j+1, welded by index."""
    nth = int(nth) - (int(nth) % 2)            # seam matching needs even
    nth = max(nth, 12)
    nr = max(int(nr), 4)
    S = get_surface(d0, d1, dinf)
    rr = np.exp(np.linspace(0.0, math.log(rmin), nr))
    th = (np.arange(nth) + 0.5) * TAU / nth
    Rg, Tg = np.meshgrid(rr, th, indexing='ij')
    wt = Rg * np.exp(1j * Tg)
    pts = []
    det_dev = 0.0
    for dom in range(3):
        X = S.hyperboloid(z_of_wt(wt, dom))
        scale = np.sum(X * X, axis=-1)
        mink = (-X[..., 0] ** 2 + X[..., 1] ** 2 + X[..., 2] ** 2
                + X[..., 3] ** 2)
        det_dev = max(det_dev, float((np.abs(mink + 1.0)
                                      / np.maximum(scale, 1.0)).max()))
        pts.append(_to_model(X, model))
    idx = np.empty((3, nr, nth), dtype=int)
    verts = []
    for dom in range(3):                       # canonical vertices
        for i in range(nr):
            for kk in range(nth):
                if i == 0 and kk >= nth // 2:
                    continue
                idx[dom, i, kk] = len(verts)
                verts.append(pts[dom][i, kk])
    for dom in range(3):                       # seam references
        for kk in range(nth // 2, nth):
            idx[dom, 0, kk] = idx[(dom + 2) % 3, 0, nth - 1 - kk]
    faces = []
    for dom in range(3):
        for i in range(nr - 1):
            for kk in range(nth):
                k1 = (kk + 1) % nth
                faces.append((idx[dom, i, kk], idx[dom, i, k1],
                              idx[dom, i + 1, k1], idx[dom, i + 1, kk]))
    info = {'det_dev': det_dev, 'r4': S.r4,
            'exponents': (S.alpha, S.tau, S.rho)}
    return np.array(verts), faces, info


# --------------------------------------------------------------------------
# mesh self-intersection -- the embeddedness detector
# --------------------------------------------------------------------------

def mesh_self_intersects(verts, faces):
    """(hit, count): does the triangulated mesh cross itself?

    Broad phase: axis-aligned-box binning on a uniform grid.  Narrow
    phase: every edge of one triangle against the other triangle
    (Moller-Trumbore, strict interior hits only), both ways, skipping
    pairs that share a vertex.  Used by the self-test to reproduce the
    symmetric-trinoid embeddedness transition at D_0 ~ 0.2332."""
    tris = []
    for f in faces:
        tris.append((f[0], f[1], f[2]))
        tris.append((f[0], f[2], f[3]))
    tris = np.asarray(tris)
    V = np.asarray(verts)
    T = V[tris]                                    # (nt, 3, 3)
    lo, hi = T.min(axis=1), T.max(axis=1)
    cell = max(float(np.linalg.norm(hi - lo, axis=1).mean()) * 1.5,
               float((hi.max(0) - lo.min(0)).max()) / 64.0)
    org = lo.min(axis=0)
    k0 = np.floor((lo - org) / cell).astype(int)
    k1 = np.floor((hi - org) / cell).astype(int)
    from collections import defaultdict
    grid = defaultdict(list)
    for t in range(len(tris)):
        for gx in range(k0[t, 0], k1[t, 0] + 1):
            for gy in range(k0[t, 1], k1[t, 1] + 1):
                for gz in range(k0[t, 2], k1[t, 2] + 1):
                    grid[(gx, gy, gz)].append(t)
    cand = set()
    for lst in grid.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                cand.add((lst[i], lst[j]) if lst[i] < lst[j]
                         else (lst[j], lst[i]))
    pairs = [(x, y) for x, y in cand
             if not (set(tris[x]) & set(tris[y]))
             and not (np.any(lo[x] > hi[y]) or np.any(lo[y] > hi[x]))]
    if not pairs:
        return False, 0
    pairs = np.asarray(pairs)

    def seg_hits_tri(P0, P1, A, B, C):
        d = P1 - P0
        e1, e2 = B - A, C - A
        pv = np.cross(d, e2)
        det = np.einsum('ij,ij->i', e1, pv)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = P0 - A
        uu = np.einsum('ij,ij->i', tv, pv) * inv
        qv = np.cross(tv, e1)
        vv = np.einsum('ij,ij->i', d, qv) * inv
        tt = np.einsum('ij,ij->i', e2, qv) * inv
        eps = 1e-9
        return (ok & (uu > eps) & (vv > eps) & (uu + vv < 1.0 - eps)
                & (tt > eps) & (tt < 1.0 - eps))

    a = tris[pairs[:, 0]]
    b = tris[pairs[:, 1]]
    hits = 0
    for (s0, s1) in ((0, 1), (1, 2), (2, 0)):
        hits += int(seg_hits_tri(V[a[:, s0]], V[a[:, s1]],
                                 V[b[:, 0]], V[b[:, 1]], V[b[:, 2]]).sum())
        hits += int(seg_hits_tri(V[b[:, s0]], V[b[:, s1]],
                                 V[a[:, 0]], V[a[:, 1]], V[a[:, 2]]).sum())
    return hits > 0, hits


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _selftest():
    ok_all = True
    rng = np.random.default_rng(20260819)

    def gate(ok, msg):
        nonlocal ok_all
        ok_all = ok_all and bool(ok)
        print(f"{msg} {'OK' if ok else 'BAD'}")

    # 1) admissibility: the Prop. 2 region test refuses what it should
    cases = [((0.16, 0.18, 0.20), True),     # the default trinoid
             ((0.2332, 0.2332, 0.2332), True),
             ((0.36, 0.36, 0.36), True),     # beyond d = 1/4, still in D
             ((0.02, 0.02, 0.02), False),    # sum <= 1/2
             ((0.16, 0.20, 0.23), False),    # pairwise condition fails
             ((0.25, 0.25, 0.25), False),    # resonant AND degenerate
             ((-0.1, 0.2, 0.2), False)]
    for d, want in cases:
        got, why = admissible(*d)
        gate(got == want, f"admissible{d} = {got} ({why[:48]})"
             f" want {want}")
    # ... and it must agree with Theorem 6 (ii)'s sine product wherever
    # the data is non-degenerate (the two are equivalent by Prop. 2)
    agree = True
    for _ in range(300):
        d = rng.uniform(0.01, 1.2, 3)
        sq = np.sqrt(d)
        if min(abs(2 * s - round(2 * s)) for s in sq) < 5e-3:
            continue
        A = 0.5 * (d[0] + d[1] - d[2] - 0.25)
        B = 0.5 * (d[0] + d[2] - d[1] - 0.25)
        C = 0.5 * (d[1] + d[2] - d[0] - 0.25)
        if min(abs(A), abs(B), abs(C)) < 1e-6:
            continue
        al, ta, ro = 0.5 - sq[0], sq[1], sq[2]
        a, b, c = al + ta + ro, al + ta - ro, 2 * al
        prod = (math.sin(math.pi * a) * math.sin(math.pi * b)
                * math.sin(math.pi * (a - c)) * math.sin(math.pi * (b - c)))
        if abs(prod) < 1e-9:
            continue
        agree &= (admissible(*d)[0] == (prod < 0.0))
    gate(agree, "Prop. 2 region == sign of Theorem 6 (ii) sine product "
         "over 300 random triples")

    # 2) the compiled Fuchsian reduction, on an asymmetric trinoid
    S = TrinoidSurface(0.16, 0.18, 0.20)
    gate(S.fuchs_res < 1e-10 and S.kscore < 1e-10,
         f"Fuchsian reduction: third-point residual {S.fuchs_res:.1e}, "
         f"residue structure {S.kscore:.1e}")

    # 3) Psi = D Phi^(0) R solves the ORIGINAL spinor equation (3.12):
    #    dPsi = [[PQ, P^2], [-Q^2, -PQ]] Psi.  This ties the gauge D,
    #    the hypergeometric solutions and the scaling together against
    #    the one equation everything came from.
    worst = 0.0
    h = 1e-5
    for z0 in (0.35 + 0.3j, 0.52 - 0.44j, -0.8 + 0.6j, 1.7 + 0.8j,
               1.7 - 0.8j, 0.25 + 0.1j):
        zs = np.array([z0 - h, z0, z0 + h])
        P3 = S.psi(zs)
        dnum = (P3[2] - P3[0]) / (2.0 * h)
        P, Q = S._PQ(np.array([z0]))
        P, Q = P[0], Q[0]
        Amat = np.array([[P * Q, P * P], [-Q * Q, -P * Q]])
        want = Amat @ P3[1]
        worst = max(worst, float(np.abs(dnum - want).max()
                                 / np.abs(want).max()))
    gate(worst < 1e-6, f"Psi solves the spinor ODE (3.12), FD residual "
         f"{worst:.1e}")

    # 4) connection matrix E1 against the paper's closed form (6.9) --
    #    the numeric matching and the gamma-function formula share
    #    nothing but the canonical solutions
    a, b, c = S.a, S.b, S.c
    al, ta = S.alpha, S.tau
    g, d_, be = S.gamma, S.delta, S.beta
    G = cgamma
    E1p = np.array([
        [-(2 * al + 1) / (be + ta) * G(c) * G(c - a - b)
         / (G(c - a) * G(c - b)),
         (2 * al - 1) / g * G(-c) * G(c - a - b) / (G(-a) * G(-b))],
        [-(2 * al + 1) / d_ * G(c) * G(a + b - c) / (G(a) * G(b))
         * np.exp(2j * math.pi * ta),
         -(2 * al - 1) / (be + ta) * G(-c) * G(a + b - c)
         / (G(a - c) * G(b - c)) * np.exp(2j * math.pi * ta)]],
        dtype=complex)
    dev = float(np.abs(S.E1 - E1p).max() / np.abs(E1p).max())
    gate(dev < 1e-10, f"E1 numeric vs closed form (6.9): {dev:.1e}")
    #    ... and r^4 against the gamma formula (7.2)
    r4p = complex((2 * al - 1) ** 2 * d_ * G(-c) ** 2 * G(a) * G(b)
                  * G(c - a) * G(c - b)
                  / ((2 * al + 1) ** 2 * g * G(c) ** 2 * G(-a) * G(-b)
                     * G(a - c) * G(b - c)))
    dev = abs(S.r4 - r4p) / abs(r4p)
    gate(dev < 1e-9 and S.unit_res < 1e-10,
         f"r^4 monodromy {S.r4:.6f} vs (7.2) {r4p.real:.6f} "
         f"({dev:.1e}); unitarity {S.unit_res:.1e}")

    # 5) one immersion: all three representations give the same F, and
    #    F is continuous across all three branch cuts (the jumps are
    #    unitary exactly because the monodromy was unitarized)
    def F_at(z):
        Ps = S.psi(np.array([z]))[0]
        return Ps @ Ps.conj().T

    def F_rep(z, rep):
        zz = np.array([z])
        if rep == 0:
            Ps = (S._D(zz) @ S._phi0(zz) @ S.R)[0]
        elif rep == 1:
            Ps = (S._D(zz) @ S._phi1(zz) @ S.E1 @ S.R)[0]
        else:
            E = S.Einf_up if z.imag >= 0 else S.Einf_dn
            Ps = (S._D(zz) @ S._phiinf(zz) @ E @ S.R)[0]
        Ps = S.scale * Ps
        return Ps @ Ps.conj().T

    w1 = max(float(np.abs(F_rep(z, 0) - F_rep(z, 1)).max()
                   / np.abs(F_rep(z, 0)).max())
             for z in (0.5 + 0.35j, 0.52 - 0.4j))
    w2 = max(float(np.abs(F_rep(z, 1) - F_rep(z, 2)).max()
                   / np.abs(F_rep(z, 1)).max())
             for z in (1.25 + 0.6j, 1.25 - 0.6j))
    gate(max(w1, w2) < 1e-10, f"F agrees across representations: "
         f"0 vs 1 {w1:.1e}, 1 vs inf {w2:.1e}")
    eps = 1e-7
    wc = 0.0
    for x in (-1.4, 0.45, 2.3):
        Fa, Fb = F_at(x + 1j * eps), F_at(x - 1j * eps)
        wc = max(wc, float(np.abs(Fa - Fb).max() / np.abs(Fa).max()))
    gate(wc < 1e-5, f"F continuous across the cuts (offset {eps:g}): "
         f"{wc:.1e}")

    # 6) on the hyperboloid, relatively, over the whole drawing domain
    worst = 0.0
    worst_x0 = np.inf
    for dom in range(3):
        rr = np.exp(rng.uniform(math.log(0.01), 0.0, 120))
        th = rng.uniform(0.0, TAU, 120)
        X = S.hyperboloid(z_of_wt(rr * np.exp(1j * th), dom))
        mink = (-X[..., 0] ** 2 + X[..., 1] ** 2 + X[..., 2] ** 2
                + X[..., 3] ** 2)
        scale = np.maximum(np.sum(X * X, axis=-1), 1.0)
        worst = max(worst, float((np.abs(mink + 1.0) / scale).max()))
        worst_x0 = min(worst_x0, float(X[..., 0].min()))
    gate(worst < 1e-10 and worst_x0 > 0.0,
         f"on the hyperboloid: relative |<x,x>+1| max {worst:.1e}, "
         f"x0 > 0")

    # 7) |H| = 1 measured in H^3 from the Minkowski fundamental forms --
    #    shares no algebra with the construction.  Median and IQR over
    #    all three domains, r in [0.05, 0.95].
    G4 = np.array([-1.0, 1.0, 1.0, 1.0])

    def mink(x, y):
        return np.sum(x * y * G4, axis=-1)

    h = 1e-4
    Hs = []
    for dom in range(3):
        for _ in range(30):
            u = rng.uniform(math.log(0.05), math.log(0.95))
            v = rng.uniform(0.0, TAU)

            def Pt(uu, vv):
                return trinoid_points(uu, vv, 0.16, 0.18, 0.20,
                                      domain=dom)
            X0 = Pt(u, v)
            Xu = (Pt(u + h, v) - Pt(u - h, v)) / (2 * h)
            Xv = (Pt(u, v + h) - Pt(u, v - h)) / (2 * h)
            Xuu = (Pt(u + h, v) - 2 * X0 + Pt(u - h, v)) / h ** 2
            Xvv = (Pt(u, v + h) - 2 * X0 + Pt(u, v - h)) / h ** 2
            Xuv = (Pt(u + h, v + h) - Pt(u + h, v - h)
                   - Pt(u - h, v + h) + Pt(u - h, v - h)) / (4 * h ** 2)
            n = np.linalg.svd(np.stack([X0, Xu, Xv]))[2][-1] * G4
            n = n / math.sqrt(abs(mink(n, n)))
            E, Fm, Gm = mink(Xu, Xu), mink(Xu, Xv), mink(Xv, Xv)
            L, Mm, N = mink(Xuu, n), mink(Xuv, n), mink(Xvv, n)
            Hs.append((E * N - 2 * Fm * Mm + Gm * L)
                      / (2 * (E * Gm - Fm ** 2)))
    Hs = np.abs(np.array(Hs))
    Hs = Hs[np.isfinite(Hs)]
    med = float(np.median(Hs))
    q1, q3 = np.percentile(Hs, [25.0, 75.0])
    gate(abs(med - 1.0) < 2e-3 and (q3 - q1) < 1e-2,
         f"|H| in H^3 over 3 domains: median {med:.6f} (want 1), "
         f"IQR {q3 - q1:.1e}, n={len(Hs)}")

    # 8) the three ends approach the three normalised ideal points of
    #    (7.5), each the right one, monotonically in depth
    ends_ok = True
    for dom in range(3):
        radii = []
        for rr_ in (0.03, 0.01, 0.003):
            th = np.linspace(0.0, TAU, 8, endpoint=False)
            X = S.hyperboloid(z_of_wt(rr_ * np.exp(1j * th), dom))
            P = X[..., 1:] / (1.0 + X[..., :1])
            cen = P.mean(axis=0)
            radii.append(float(np.linalg.norm(cen)))
        dirv = cen / np.linalg.norm(cen)
        err = float(np.linalg.norm(dirv - END_POINTS[dom]))
        mono = radii[0] < radii[1] < radii[2]
        ends_ok &= (err < 5e-3 and mono)
        print(f"  end {dom}: -> {np.round(dirv, 4)} vs "
              f"{np.round(END_POINTS[dom], 4)}, err {err:.1e}, "
              f"|p| {radii[0]:.3f} < {radii[1]:.3f} < {radii[2]:.3f}")
    gate(ends_ok, "three ends at the (7.5) ideal points, monotone "
         "approach")

    # 9) mesh: welded seams coincide, finite, inside the Poincare ball
    V, Fc, info = build_trinoid_mesh(0.16, 0.18, 0.20, nr=18, nth=48,
                                     rmin=0.04)
    rad = np.linalg.norm(V, axis=1)
    gate(np.isfinite(V).all() and len(Fc) > 0 and float(rad.max()) < 1.0
         and info['det_dev'] < 1e-10,
         f"mesh builds: V={len(V)} F={len(Fc)} max radius "
         f"{rad.max():.4f} < 1, det dev {info['det_dev']:.1e}")

    # 10) the embeddedness transition of the symmetric trinoids at
    #     D_0 ~ 0.2332 ([BPS] Fig. 3): embedded on one side, mesh
    #     self-intersection on the other, bracketed both coarsely and
    #     at the paper's own quoted value.
    for d, want in ((0.20, False), (0.28, True)):
        V, Fc, _ = build_trinoid_mesh(d, d, d, nr=22, nth=60, rmin=0.05)
        hit, nh = mesh_self_intersects(V, Fc)
        gate(hit == want, f"symmetric d={d}: self-intersects={hit} "
             f"({nh} hits), want {want}")
    fine = {}
    for d in (0.2332, 0.2340):
        V, Fc, _ = build_trinoid_mesh(d, d, d, nr=34, nth=110,
                                      rmin=0.015)
        fine[d] = mesh_self_intersects(V, Fc)
    gate((not fine[0.2332][0]) and fine[0.2340][0],
         f"embeddedness flips between d=0.2332 ({fine[0.2332][0]}) and "
         f"d=0.2340 ({fine[0.2340][0]}) -- paper's D_0 ~ 0.2332")

    assert ok_all
    print("trinoid standalone tests passed")
