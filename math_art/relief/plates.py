"""relief.plates -- vibration modes of membranes and free plates.

The geometric half of the relief catalogue: the standing-wave patterns that a
drum or a plate actually makes.  Two families, and the difference between them
is visible rather than academic.

A **membrane** (a drumhead) is held at its rim, so every mode vanishes there
and the relief dies at the frame.  A **plate** is stiff and here left *free* at
its edges, so its nodal lines run into the boundary and organise the whole
panel -- which is what a sand Chladni photograph looks like.

The free plate is governed by the biharmonic operator with free edges, and its
modes are not available in closed form.  They are obtained here by the
Rayleigh-Ritz method Ritz invented for exactly this problem in 1909, with the
eigenfunctions of a free-free bar as the trial basis.  With s = 10 (121 basis
functions) the resulting 121x121 generalised eigenproblem reproduces the
published eigenvalues to every printed digit in milliseconds.

The naive "Chladni formula" cos(m pi x)cos(n pi y) + chi * (swap) is a
*membrane* combination, and it is Rayleigh's approximation, not Ritz's -- Ritz's
paper is precisely the demonstration that it is insufficient.  It is offered
alongside as a fast, freely-mixable art control, but the exact solver is the
default and the honest one.

References:
  Ernst F. F. Chladni, "Entdeckungen ueber die Theorie des Klanges",
    Weidmanns Erben und Reich, Leipzig, 1787.
  Gustav Kirchhoff, "Ueber das Gleichgewicht und die Bewegung einer elastischen
    Scheibe", Journal fuer die reine und angewandte Mathematik 40, 1850, 51-88
    -- the plate equation and its free-edge conditions.
  Walther Ritz, "Theorie der Transversalschwingungen einer quadratischen Platte
    mit freien Raendern", Annalen der Physik (4) 28, 1909, 737-786.
  Lord Rayleigh, "The Theory of Sound", Macmillan, 1877, chs. IX-X -- the
    circular membrane, and the cosine-combination approximation for the free
    square plate.
  Martin J. Gander and Felix Kwok, "Chladni Figures and the Tacoma Bridge:
    Motivating PDE Eigenvalue Problems via Vibrating Plates", SIAM Review
    54(3), 2012, 573-596 -- the modern exposition followed here: the energy
    functional (their 2.2), the weak form (2.8) and the free-free bar basis
    (3.2).
  Mary D. Waller, "Chladni Figures: A Study in Symmetry", G. Bell and Sons,
    1961 -- the experimental atlas, and the symmetry-class law used as a
    structural check on any plate solver.
"""

import math

import numpy as np

from . import special as _special


# ------------------------------------------------------------------
# Free-free bar eigenfunctions (the Ritz trial basis)
# ------------------------------------------------------------------

def beam_root(m):
    """k_m for the m-th free-free bar mode (m >= 2).

    The characteristic equations are tan k + tanh k = 0 (m even) and
    tan k - tanh k = 0 (m odd).  They are solved in the pole-free form
    sin k +/- cos k tanh k = 0, because tan's poles sit right next to the
    roots and wreck a naive bracket.
    """
    m = int(m)
    if m < 2:
        return 0.0
    even = (m % 2 == 0)

    def f(k):
        return math.sin(k) + (math.cos(k) * math.tanh(k) if even
                              else -math.cos(k) * math.tanh(k))

    seed = (2.0 * m - 1.0) * math.pi / 4.0
    lo, hi = seed - 0.45, seed + 0.45
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:                    # widen once if the bracket missed
        lo, hi = seed - 0.9, seed + 0.9
        flo, fhi = f(lo), f(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


def beam_basis(x, s):
    """Values and 2nd/1st derivatives of bar modes u_0..u_s on [-1, 1].

    Returns (U, U1, U2), each of shape (s+1, len(x)).  u_0 and u_1 are the
    rigid-body modes (piston and tilt) with k = 0; they carry no bending
    energy but are essential -- a free plate has them, and omitting them
    changes the spectrum.
    """
    x = np.asarray(x, dtype=float)
    n = int(s) + 1
    U = np.zeros((n, x.size))
    U1 = np.zeros((n, x.size))
    U2 = np.zeros((n, x.size))
    for m in range(n):
        if m == 0:
            U[m] = 1.0 / math.sqrt(2.0)
        elif m == 1:
            U[m] = math.sqrt(1.5) * x
            U1[m] = math.sqrt(1.5)
        else:
            k = beam_root(m)
            ck, sk = math.cos(k), math.sin(k)
            chk, shk = math.cosh(k), math.sinh(k)
            if m % 2 == 0:
                den = math.sqrt(chk * chk + ck * ck)
                U[m] = (chk * np.cos(k * x) + ck * np.cosh(k * x)) / den
                U1[m] = k * (-chk * np.sin(k * x) + ck * np.sinh(k * x)) / den
                U2[m] = k * k * (-chk * np.cos(k * x)
                                 + ck * np.cosh(k * x)) / den
            else:
                den = math.sqrt(abs(shk * shk - sk * sk))
                U[m] = (shk * np.sin(k * x) + sk * np.sinh(k * x)) / den
                U1[m] = k * (shk * np.cos(k * x) + sk * np.cosh(k * x)) / den
                U2[m] = k * k * (-shk * np.sin(k * x)
                                 + sk * np.sinh(k * x)) / den
    return U, U1, U2


def _grams(s, quad=None):
    """1-D Gram matrices A, B, C, D over [-1, 1] by Gauss-Legendre.

        A = int u_i'' u_k''    B = int u_i u_k
        C = int u_i'' u_k      D = int u_i'  u_k'
    """
    nq = quad or (8 * (int(s) + 1) + 40)
    xq, wq = np.polynomial.legendre.leggauss(nq)
    U, U1, U2 = beam_basis(xq, s)
    A = (U2 * wq) @ U2.T
    B = (U * wq) @ U.T
    C = (U2 * wq) @ U.T
    D = (U1 * wq) @ U1.T
    return A, B, C, D


def free_plate_modes(s=10, poisson=0.225, aspect=1.0, count=16, quad=None):
    """Eigenvalues and mode coefficients of a FREE rectangular plate.

    Rayleigh-Ritz over the separable basis w_mn = u_m(x) u_n(y/aspect), with
    the weak form of Gander-Kwok (2.8),

        a(u,v) = int u_xx v_xx + u_yy v_yy
                 + mu (u_yy v_xx + u_xx v_yy) + 2(1-mu) u_xy v_xy .

    In Kronecker form, with the y-extent scaled by `aspect` = a,

        K = A(x)B + a^-4 B(x)A + mu a^-2 (C^T(x)C + C(x)C^T)
            + 2(1-mu) a^-2 D(x)D
        M = B(x)B

    where (x) is the Kronecker product.  NumPy's `eigh` takes no second
    matrix, so the generalised problem is reduced with a Cholesky factor of M
    -- reaching for `eig` instead would return a non-symmetric solver's
    complex noise.

    Returns `(lam, vecs, s)`: eigenvalues ascending (the first three are the
    rigid-body modes, ~0), and the coefficient matrix whose columns are modes.
    """
    A, B, C, D = _grams(s, quad)
    a2 = float(aspect) ** -2
    a4 = float(aspect) ** -4
    mu = float(poisson)
    K = (np.kron(A, B) + a4 * np.kron(B, A)
         + mu * a2 * (np.kron(C.T, C) + np.kron(C, C.T))
         + 2.0 * (1.0 - mu) * a2 * np.kron(D, D))
    M = np.kron(B, B)
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)

    L = np.linalg.cholesky(M)
    Li = np.linalg.inv(L)
    lam, vec = np.linalg.eigh(Li @ K @ Li.T)
    vec = Li.T @ vec
    k = min(int(count), lam.size)
    return lam[:k], vec[:, :k], int(s)


_MODE_CACHE = {}


def free_plate_cached(s=10, poisson=0.225, aspect=1.0, count=24):
    """`free_plate_modes` memoised on its arguments.

    The redo panel re-runs the whole generator on every slider drag, and the
    121x121 eigensolve does not need repeating when only the relief depth
    moved.
    """
    key = (int(s), round(float(poisson), 6), round(float(aspect), 6),
           int(count))
    hit = _MODE_CACHE.get(key)
    if hit is None:
        hit = free_plate_modes(s=s, poisson=poisson, aspect=aspect,
                               count=count)
        if len(_MODE_CACHE) > 24:          # keep it bounded
            _MODE_CACHE.clear()
        _MODE_CACHE[key] = hit
    return hit


def plate_mode_field(X, Y, coeffs, s, aspect=1.0):
    """Evaluate one Ritz mode (a coefficient vector) on the sample grid."""
    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    xs = X[0, :] / hx
    ys = Y[:, 0] / hy
    Ux, _, _ = beam_basis(xs, s)
    Uy, _, _ = beam_basis(ys, s)
    a = np.asarray(coeffs, dtype=float).reshape(s + 1, s + 1)
    return Uy.T @ a.T @ Ux            # (ny, nx)


# ------------------------------------------------------------------
# Closed-form membranes
# ------------------------------------------------------------------

def rect_membrane(X, Y, m=2, n=3):
    """Clamped rectangular membrane mode sin(m pi u) sin(n pi v), u,v in [0,1]."""
    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    u = 0.5 * (X / hx + 1.0)
    v = 0.5 * (Y / hy + 1.0)
    return np.sin(m * math.pi * u) * np.sin(n * math.pi * v)


def circular_membrane(X, Y, m=1, n=1, phase=0.0):
    """Drumhead mode J_m(j_{m,n} r/R) cos(m theta + phase), clamped at r = R."""
    R = min(float(np.abs(X).max()), float(np.abs(Y).max())) or 1.0
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    j = _special.bessel_zero(int(m), int(n))
    out = _special.besselj(int(m), np.clip(r / R, 0.0, 1.0) * j)
    if m:
        out = out * np.cos(m * th + phase)
    return np.where(r <= R, out, 0.0)


def chladni_rayleigh(X, Y, m=2, n=3, chi=1.0):
    """Rayleigh's cosine combination for a free square plate.

    cos(m pi u) cos(n pi v) + chi * cos(n pi u) cos(m pi v).

    Fast and decent for low modes, and `chi` is a pleasant art control -- but
    it is an approximation, and for a real plate chi is not free: pairs with
    m + n even are split by the plate operator into just chi = +1 and -1 at
    two different frequencies.  Use `free_plate_modes` for the physics.
    """
    hx = float(np.abs(X).max()) or 1.0
    hy = float(np.abs(Y).max()) or 1.0
    u = 0.5 * (X / hx + 1.0)
    v = 0.5 * (Y / hy + 1.0)
    return (np.cos(m * math.pi * u) * np.cos(n * math.pi * v)
            + float(chi) * np.cos(n * math.pi * u) * np.cos(m * math.pi * v))


def _selftest():
    ok = True

    # --- the gate: reproduce the published free-square-plate eigenvalues ---
    # Gander & Kwok, SIAM Review 54(3), 2012, Table 4.1 (Poisson ratio 0.225,
    # the value Ritz used to match Chladni's glass plates).
    lam, vec, s = free_plate_modes(s=10, poisson=0.225, count=18)
    nonzero = lam[lam > 1.0]
    want = [12.5, 26.1, 35.8, 81.2, 81.2, 236.3, 236.3, 270.0]
    got = [round(float(v), 1) for v in nonzero[:len(want)]]
    print("plates: first rigid-body eigenvalues = %s (want ~0)"
          % np.round(lam[:3], 6))
    print("plates: computed %s" % got)
    print("plates: published %s" % want)
    ok = ok and all(abs(g - w) < 0.05 for g, w in zip(got, want))
    ok = ok and float(np.abs(lam[:3]).max()) < 1e-6      # 3 rigid-body modes

    # Degeneracy is physics: modes 4/5 and 6/7 are genuine pairs.
    print("plates: degenerate pairs %.4f/%.4f and %.4f/%.4f"
          % (nonzero[3], nonzero[4], nonzero[5], nonzero[6]))
    ok = ok and abs(nonzero[3] - nonzero[4]) < 1e-3
    ok = ok and abs(nonzero[5] - nonzero[6]) < 1e-3

    # The Ritz bound is variational: a bigger basis can only lower it.
    lam8, _, _ = free_plate_modes(s=8, poisson=0.225, count=8)
    lam12, _, _ = free_plate_modes(s=12, poisson=0.225, count=8)
    n8 = lam8[lam8 > 1.0][0]
    n12 = lam12[lam12 > 1.0][0]
    print("plates: first mode s=8 -> %.6f, s=12 -> %.6f (must not increase)"
          % (n8, n12))
    ok = ok and n12 <= n8 + 1e-9

    # Poisson's ratio really moves the spectrum (the toy model cannot).
    lg, _, _ = free_plate_modes(s=8, poisson=0.225, count=8)
    lb, _, _ = free_plate_modes(s=8, poisson=0.33, count=8)
    print("plates: first mode mu=0.225 -> %.4f, mu=0.33 -> %.4f"
          % (lg[lg > 1.0][0], lb[lb > 1.0][0]))
    ok = ok and abs(lg[lg > 1.0][0] - lb[lb > 1.0][0]) > 0.1

    # Mode fields evaluate on a grid.
    from . import grid as _grid
    X, Y, _ = _grid.make_grid(width=2.0, aspect=1.0, resolution=61)
    f = plate_mode_field(X, Y, vec[:, 3], s)
    print("plates: mode field range [%.3f, %.3f]" % (f.min(), f.max()))
    ok = ok and np.isfinite(f).all() and f.std() > 1e-9

    # --- membranes ---
    d = circular_membrane(X, Y, m=1, n=2)
    R = min(float(np.abs(X).max()), float(np.abs(Y).max()))
    rim = np.abs(np.hypot(X, Y) - R) < 0.02
    print("plates: drumhead |value| at the clamped rim <= %.2e"
          % float(np.abs(d[rim]).max()))
    ok = ok and float(np.abs(d[rim]).max()) < 5e-2 and np.isfinite(d).all()

    r = rect_membrane(X, Y, 2, 3)
    ok = ok and abs(float(r[0, :].max())) < 1e-9      # vanishes on the frame
    ok = ok and abs(float(r[:, 0].max())) < 1e-9

    c = chladni_rayleigh(X, Y, 2, 3, 1.0)
    ok = ok and np.isfinite(c).all() and c.std() > 1e-6

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
