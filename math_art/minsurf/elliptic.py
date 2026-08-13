# Weierstrass elliptic-function engine (Jacobi-theta series).
#
# Part of the Math Art minimal-surface engine (`math_art/minsurf/`), split
# out of the former single-file `minimal_surface_toolkit.py`.  Numpy only --
# no `bpy` -- so the whole engine imports and self-tests headlessly; the
# registered Blender operators stay in the flat `minimal_surface_toolkit.py`
# front-end.
#
# References:
#   Weierstrass P, P' and zeta via Jacobi theta functions: DLMF 23.6
#   (elliptic functions) and DLMF 20.2 (the theta q-series).
#   K. Weierstrass, Mathematische Werke (1894-1927).

import math
import numpy as np

TAU = 2.0 * math.pi


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

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("elliptic self-test failed")
