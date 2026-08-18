# Step-size control and quasi-Newton descent for relaxation loops.
#
# Part of the shared solver core (`math_art/solver/`).  NumPy only.
#
# Mechanisms, all replacing hand-tuned fixed gains:
#
#   * parabola_line_search -- Surface Evolver's "optimizing scale":
#     evaluate the energy at the current scale; double while it falls,
#     halve while it rises (bounded), then jump to the vertex of the
#     parabola through the best bracket and keep the better of vertex
#     and bracket.  Costs 2-5 energy evaluations, needs no gradient
#     beyond the step direction, and turns "diverged" into "small step".
#
#   * AdaptiveGain -- the antiprism `canonical` controller: grow the
#     gain slightly on improvement, cut it on regression.  Zero extra
#     energy evaluations; suits fixed-point iterations whose "residual"
#     is cheap but whose energy is not.
#
#   * LBFGS + armijo_backtrack -- limited-memory BFGS by the two-loop
#     recursion over a small (s, y) history, with a curvature guard
#     (a pair is REJECTED when s.y is not sufficiently positive --
#     without the guard the inverse-Hessian approximation degrades
#     silently on non-convex energies) and Armijo backtracking from the
#     quasi-Newton unit step.  With exact line searches and full
#     history, L-BFGS on an n-dimensional quadratic terminates in at
#     most n steps (it reproduces conjugate gradients) -- the self-test
#     checks exactly that.
#
#   * LaplacianH0 -- an inverse-Hessian SEED for LBFGS: instead of the
#     scaled identity, apply (L + eps D)^-1 by warm-started Jacobi-PCG,
#     L the cotan Laplacian of the current mesh (solver/cotan) and D
#     its diagonal.  The area Hessian of a triangle mesh is the cotan
#     Laplacian to leading order (Pinkall-Polthier), so this seed makes
#     the quasi-Newton step approximately mesh-resolution-independent:
#     the stiff short-wavelength modes that force tiny explicit steps
#     are divided out -- precisely what mobility scaling plus nonlinear
#     CG lacks at high mesh resolution.
#
# References:
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992) -- the optimizing-scale line search (iterate.c).
#   A. Rossiter, Antiprism (antiprism.com), `canonical` -- the adaptive
#       gain schedule.  Includes ideas and algorithms by George W. Hart.
#   J. Nocedal and S. J. Wright, "Numerical Optimization", 2nd ed.,
#       Springer (2006) -- ch. 3 line searches (Armijo condition);
#       Alg. 7.4/7.5 (two-loop L-BFGS recursion, curvature condition).
#   D. C. Liu and J. Nocedal, "On the limited memory BFGS method for
#       large scale optimization", Math. Programming 45 (1989).
#   Y. Zhu, R. Bridson, D. M. Kaufman, "Blended cured quasi-Newton for
#       distortion optimization", ACM Trans. Graph. 37(4) (2018) --
#       Laplacian-seeded quasi-Newton for geometry optimization.
#   U. Pinkall and K. Polthier, "Computing discrete minimal surfaces
#       and their conjugates", Experimental Mathematics 2(1) (1993) --
#       the area gradient is the cotan Laplacian applied to positions
#       (so L is the natural inverse-Hessian seed for area descent).

import numpy as np


def parabola_line_search(energy, x, d, s0, s_max=None, max_double=12,
                         max_halve=20):
    """Minimise energy(x + s*d) over s > 0, Evolver-style.

    energy: callable(ndarray) -> float.  x: current point.  d: step
    direction (same shape).  s0: starting scale (e.g. the last accepted
    one).  Returns (x_new, s_used, E_new, n_evals).  If no downhill
    scale is found, returns (x, 0.0, E(x), n_evals) -- caller decides
    whether that means convergence or non-existence.
    """
    E0 = float(energy(x))
    n_evals = 1
    s = float(s0)
    if s_max is not None:
        s = min(s, float(s_max))
    E1 = float(energy(x + s * d))
    n_evals += 1
    pts = [(0.0, E0), (s, E1)]
    if E1 < E0:
        # downhill: double until it rises
        for _ in range(max_double):
            s2 = 2.0 * s
            if s_max is not None and s2 > s_max:
                break
            E2 = float(energy(x + s2 * d))
            n_evals += 1
            pts.append((s2, E2))
            if E2 >= pts[-2][1]:
                break
            s = s2
    else:
        # uphill: halve until it falls (or give up)
        found = False
        for _ in range(max_halve):
            s = 0.5 * s
            E1 = float(energy(x + s * d))
            n_evals += 1
            pts.append((s, E1))
            if E1 < E0:
                found = True
                break
        if not found:
            return x, 0.0, E0, n_evals
    # parabola through the best point and its neighbours in s-order
    pts.sort()
    Es = [p[1] for p in pts]
    ib = int(np.argmin(Es))
    if 0 < ib < len(pts) - 1:
        (sa, Ea), (sb, Eb), (sc, Ec) = pts[ib - 1], pts[ib], pts[ib + 1]
        denom = 2.0 * (Ea * (sb - sc) + Eb * (sc - sa) + Ec * (sa - sb))
        if abs(denom) > 1e-300:
            sv = ((Ea - Ec) * sb * sb + (Eb - Ea) * sc * sc
                  + (Ec - Eb) * sa * sa) / denom
            if sa < sv < sc and sv > 0.0:
                Ev = float(energy(x + sv * d))
                n_evals += 1
                if Ev < Eb:
                    return x + sv * d, sv, Ev, n_evals
    sb, Eb = pts[ib]
    if sb == 0.0:                        # best is no move at all
        return x, 0.0, E0, n_evals
    return x + sb * d, sb, Eb, n_evals


class AdaptiveGain:
    """Antiprism-style gain schedule: track a scalar residual; grow the
    gain by `up` when it improves, shrink by `down` when it regresses.
    """

    def __init__(self, gain=1.0, up=1.01, down=0.995, lo=1e-3, hi=4.0):
        self.gain = float(gain)
        self.up = up
        self.down = down
        self.lo = lo
        self.hi = hi
        self._last = None

    def update(self, residual):
        """Feed the current residual; returns the gain to use next."""
        if self._last is not None:
            if residual < self._last:
                self.gain = min(self.gain * self.up, self.hi)
            else:
                self.gain = max(self.gain * self.down, self.lo)
        self._last = float(residual)
        return self.gain


class LBFGS:
    """Two-loop limited-memory BFGS (Nocedal-Wright Alg. 7.4/7.5).

    Maintains up to `m` curvature pairs (s, y) = (x_new - x_old,
    g_new - g_old) as flat 1-D arrays.  `push` REJECTS a pair whose
    curvature s.y <= curv_eps * |s| |y| (counted in `skips`): on a
    non-convex energy an indefinite pair would silently corrupt the
    inverse-Hessian approximation, and skipping it keeps the operator
    positive definite (the standard cautious update).

    `direction(g, h0=...)` returns H g (so the DESCENT direction is
    its negative).  `h0` seeds the recursion: a callable q -> H0 q
    (e.g. LaplacianH0), or None for the scaled identity
    gamma I, gamma = s.y / y.y of the newest pair -- the standard
    Barzilai-Borwein-style seed.

    After `reset_after_skips` CONSECUTIVE rejected pairs the whole
    history is dropped: with Armijo-only line searches (no Wolfe
    curvature condition) a non-convex stretch can reject every new
    pair, and a frozen stale history then crawls -- measured on
    Rosenbrock, where the frozen model plateaued while a reset run
    converged.  A reset also clears the skip streak."""

    def __init__(self, m=8, curv_eps=1e-8, reset_after_skips=3):
        self.m = int(m)
        self.curv_eps = float(curv_eps)
        self.reset_after_skips = int(reset_after_skips)
        self.S = []
        self.Y = []
        self.rho = []
        self.skips = 0
        self.resets = 0
        self._skip_streak = 0

    def __len__(self):
        return len(self.S)

    def reset(self):
        self.S.clear()
        self.Y.clear()
        self.rho.clear()
        self._skip_streak = 0

    def push(self, s, y):
        """Offer a curvature pair; returns True if stored."""
        s = np.asarray(s, float).ravel()
        y = np.asarray(y, float).ravel()
        sy = float(s @ y)
        ns = float(np.linalg.norm(s))
        ny = float(np.linalg.norm(y))
        if not np.isfinite(sy) or sy <= self.curv_eps * ns * ny:
            self.skips += 1
            self._skip_streak += 1
            if (self.reset_after_skips
                    and self._skip_streak >= self.reset_after_skips
                    and len(self.S)):
                self.reset()
                self.resets += 1
            return False
        self._skip_streak = 0
        self.S.append(s)
        self.Y.append(y)
        self.rho.append(1.0 / sy)
        if len(self.S) > self.m:
            self.S.pop(0)
            self.Y.pop(0)
            self.rho.pop(0)
        return True

    def direction(self, g, h0=None):
        """H g by the two-loop recursion (flat in, flat out)."""
        q = np.asarray(g, float).ravel().copy()
        k = len(self.S)
        alphas = [0.0] * k
        for i in range(k - 1, -1, -1):
            alphas[i] = self.rho[i] * float(self.S[i] @ q)
            q -= alphas[i] * self.Y[i]
        if h0 is not None:
            r = np.asarray(h0(q), float).ravel().copy()
        elif k:
            y = self.Y[-1]
            gamma = 1.0 / (self.rho[-1] * float(y @ y))
            r = gamma * q
        else:
            r = q
        for i in range(k):
            beta = self.rho[i] * float(self.Y[i] @ r)
            r += (alphas[i] - beta) * self.S[i]
        return r


def armijo_backtrack(energy, x, d, slope, s0=1.0, E0=None, c1=1e-4,
                     shrink=0.5, max_backtracks=25, s_max=None):
    """Armijo (sufficient-decrease) backtracking line search.

    energy: callable(ndarray) -> float.  d: descent direction.
    slope: the directional derivative g.d at x (must be negative for a
    descent direction; a non-negative slope returns immediately with
    s = 0 so the caller can reset its quasi-Newton state).  Starts at
    s0 (the quasi-Newton unit step), optionally capped at s_max, and
    accepts the first s with E(x + s d) <= E0 + c1 s slope.

    Returns (x_new, s_used, E_new, n_evals) -- the same contract as
    parabola_line_search; s_used = 0.0 means no acceptable step."""
    if E0 is None:
        E0 = float(energy(x))
        n_evals = 1
    else:
        E0 = float(E0)
        n_evals = 0
    slope = float(slope)
    if not (slope < 0.0):
        return x, 0.0, E0, n_evals
    s = float(s0)
    if s_max is not None:
        s = min(s, float(s_max))
    for _ in range(max_backtracks):
        x_n = x + s * d
        E_n = float(energy(x_n))
        n_evals += 1
        if E_n <= E0 + c1 * s * slope:
            return x_n, s, E_n, n_evals
        s *= shrink
    return x, 0.0, E0, n_evals


class LaplacianH0:
    """Cotan-Laplacian inverse-Hessian seed for LBFGS on mesh vertex
    problems: apply z = (L + eps D)^-1 q per coordinate, where L is the
    (mollified) cotan Laplacian of the current mesh, D its diagonal,
    and eps a small relative shift that makes the operator SPD (L alone
    annihilates constants).  Solved by Jacobi-preconditioned CG,
    warm-started from the previous application, to a loose relative
    tolerance -- an inexact seed is standard for quasi-Newton and the
    surrounding L-BFGS updates absorb the error.

    `update(V, T, free=None)` rebuilds the weights for the current
    geometry (call it whenever V or T changed); rows of non-free
    vertices act as the identity so pinned vertices never move."""

    def __init__(self, eps=1e-3, tol=1e-2, max_iters=100,
                 cotan_mode="mollify"):
        self.eps = float(eps)
        self.tol = float(tol)
        self.max_iters = int(max_iters)
        self.cotan_mode = cotan_mode
        self._E = None
        self._W = None
        self._diag = None
        self._free = None
        self._n = 0
        self._warm = None
        self.last_iters = 0

    def update(self, V, T, free=None):
        try:
            from . import cotan as _cotan
        except ImportError:                  # flat headless import
            import cotan as _cotan           # type: ignore
        E, W = _cotan.edge_cotan_weights(V, T, mode=self.cotan_mode)
        self.set_weights(E, W, len(V), free=free)

    def set_weights(self, E, W, n, free=None):
        """Install prebuilt half-edge cotan weights (a caller that has
        just assembled the area gradient already holds them)."""
        deg = np.zeros(n)
        np.add.at(deg, E[:, 0], W)
        np.add.at(deg, E[:, 1], W)
        # guard: a fully isolated vertex would have a zero row
        deg = np.maximum(deg, 1e-300)
        self._E = E
        self._W = W
        self._diag = deg
        self._n = n
        self._free = (None if free is None
                      else np.asarray(free, bool))
        if self._warm is not None and self._warm.shape[0] != n:
            self._warm = None

    def _matvec(self, X):
        """(L + eps D) X for X (n, 3); non-free rows act as identity."""
        E, W, deg = self._E, self._W, self._diag
        Y = (1.0 + self.eps) * deg[:, None] * X
        # off-diagonal: -w_ij x_j scattered to i (and symmetrically),
        # via bincount per column (much faster than np.add.at here)
        n = self._n
        for c in range(3):
            xc = X[:, c]
            Y[:, c] -= np.bincount(E[:, 0], W * xc[E[:, 1]], minlength=n)
            Y[:, c] -= np.bincount(E[:, 1], W * xc[E[:, 0]], minlength=n)
        if self._free is not None:
            Y[~self._free] = X[~self._free]
        return Y

    def __call__(self, q):
        """Apply the seed to a flat 3n vector (or (n, 3) array)."""
        if self._E is None:
            raise RuntimeError("LaplacianH0.update(V, T) must be called "
                               "before applying the seed")
        Q = np.asarray(q, float).reshape(self._n, 3)
        if self._free is not None:
            # solve the Dirichlet-restricted (free-vertex) system: the
            # non-free entries of both input and output are pinned to 0,
            # keeping the PCG operator SPD on the working subspace
            Q = Q.copy()
            Q[~self._free] = 0.0
        X = (self._warm.copy() if self._warm is not None
             else np.zeros_like(Q))
        if self._free is not None:
            X[~self._free] = 0.0
        R = Q - self._matvec(X)
        Minv = 1.0 / ((1.0 + self.eps) * self._diag)
        Z = Minv[:, None] * R
        P = Z.copy()
        rz = np.einsum('ij,ij->', R, Z)
        q_norm2 = max(float(np.einsum('ij,ij->', Q, Q)), 1e-300)
        it = 0
        for it in range(1, self.max_iters + 1):
            AP = self._matvec(P)
            pAp = float(np.einsum('ij,ij->', P, AP))
            if not (pAp > 0.0):
                break
            a = rz / pAp
            X += a * P
            R -= a * AP
            if float(np.einsum('ij,ij->', R, R)) < self.tol ** 2 * q_norm2:
                break
            Z = Minv[:, None] * R
            rz_new = np.einsum('ij,ij->', R, Z)
            P = Z + (rz_new / max(rz, 1e-300)) * P
            rz = rz_new
        self.last_iters = it
        self._warm = X.copy()
        return X.ravel() if np.asarray(q).ndim == 1 else X


def _selftest():
    ok = True

    # 1-D quadratic: the parabola fit must land essentially on the
    # minimum in one call.
    xstar = np.array([1.7])
    energy = lambda x: float((x[0] - xstar[0]) ** 2)
    x = np.array([0.0])
    d = np.array([1.0])
    x1, s, E1, ne = parabola_line_search(energy, x, d, s0=0.3)
    good = abs(x1[0] - 1.7) < 1e-9 and E1 < 1e-17 and ne <= 12
    ok &= good
    print(f"descent: quadratic line search hits x*={x1[0]:.6f} "
          f"(E={E1:.1e}, {ne} evals) {'OK' if good else 'FAIL'}")

    # Overlong initial scale: must halve into the bracket, still land.
    x2, s2, E2, ne2 = parabola_line_search(energy, x, d, s0=100.0)
    good = E2 < energy(x) and 0 < s2 < 100.0
    ok &= good
    print(f"descent: uphill start recovers (s={s2:.3g}, E={E2:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # Ascent direction: must refuse to move rather than climb.
    x3, s3, E3, ne3 = parabola_line_search(energy, np.array([1.7]),
                                           np.array([1.0]), s0=0.5)
    good = s3 == 0.0 and abs(E3 - 0.0) < 1e-18
    ok &= good
    print(f"descent: at the minimum no move is taken (s={s3}) "
          f"{'OK' if good else 'FAIL'}")

    # Rosenbrock descent with the line search must monotonically
    # decrease and roughly converge in a few hundred iterations.
    def rosen(p):
        return float((1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2)

    def rosen_grad(p):
        return np.array([
            -2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2),
            200 * (p[1] - p[0] ** 2)])

    p = np.array([-1.2, 1.0])
    s = 1e-3
    E_prev = rosen(p)
    mono = True
    for _ in range(400):
        g = rosen_grad(p)
        p, s_used, E, _ = parabola_line_search(rosen, p, -g, s0=max(s, 1e-9))
        if E > E_prev + 1e-12:
            mono = False
        E_prev = E
        s = s_used if s_used > 0 else s * 0.5
    good = mono and rosen(p) < 1e-2
    ok &= good
    print(f"descent: Rosenbrock monotone to E={rosen(p):.2e} "
          f"{'OK' if good else 'FAIL'}")

    # AdaptiveGain: improvement grows the gain, regression shrinks it.
    ag = AdaptiveGain(gain=1.0)
    ag.update(1.0)
    g1 = ag.update(0.5)
    g2 = ag.update(0.7)
    good = g1 > 1.0 and g2 < g1
    ok &= good
    print(f"descent: adaptive gain up {g1:.4f} / down {g2:.4f} "
          f"{'OK' if good else 'FAIL'}")

    # LBFGS two-loop recursion on a known quadratic: with full history
    # and EXACT line searches, L-BFGS reproduces conjugate gradients and
    # must reach the minimizer of 0.5 x'Ax - b'x in at most n steps.
    # This is a structural property of the recursion (Nocedal-Wright
    # ch. 7), so it cannot be passed by a merely-decreasing method.
    rng = np.random.default_rng(3)
    nq = 12
    Aq = rng.normal(size=(nq, nq))
    Aq = Aq @ Aq.T + nq * np.eye(nq)     # SPD, decently conditioned
    bq = rng.normal(size=nq)
    xq = np.zeros(nq)
    lb = LBFGS(m=nq)
    gq = Aq @ xq - bq
    steps = 0
    for steps in range(1, nq + 1):
        d = -lb.direction(gq)
        t = -(gq @ d) / (d @ Aq @ d)     # exact line search
        s = t * d
        xq = xq + s
        g_new = Aq @ xq - bq
        lb.push(s, g_new - gq)
        gq = g_new
        if np.linalg.norm(gq) < 1e-9 * np.linalg.norm(bq):
            break
    x_star = np.linalg.solve(Aq, bq)
    err = float(np.linalg.norm(xq - x_star) / np.linalg.norm(x_star))
    good = steps <= nq and err < 1e-8
    ok &= good
    print(f"descent: LBFGS solves n={nq} quadratic in {steps} steps "
          f"(rel err {err:.1e}) {'OK' if good else 'FAIL'}")

    # Curvature guard: a negative-curvature pair must be REJECTED and
    # counted, leaving the stored history unchanged.
    k_before = len(lb)
    stored = lb.push(np.array([1.0] * nq), np.array([-1.0] * nq))
    good = (not stored) and len(lb) == k_before and lb.skips == 1
    ok &= good
    print(f"descent: negative-curvature pair skipped "
          f"(skips={lb.skips}) {'OK' if good else 'FAIL'}")

    # Armijo: refuses an ascent direction; accepts the unit step on a
    # well-scaled quadratic and satisfies the sufficient decrease.
    Efun = lambda x: float(0.5 * x @ x)
    x0 = np.array([1.0, -2.0])
    _, s_up, _, _ = armijo_backtrack(Efun, x0, x0, slope=float(x0 @ x0))
    d0 = -x0
    xn, s_ok, En, ne = armijo_backtrack(Efun, x0, d0,
                                        slope=float(x0 @ d0))
    good = (s_up == 0.0 and s_ok == 1.0 and En == 0.0)
    ok &= good
    print(f"descent: Armijo rejects ascent (s={s_up}), unit step on "
          f"quadratic (s={s_ok}, E={En:.1e}) {'OK' if good else 'FAIL'}")

    # LBFGS + Armijo on Rosenbrock must reach the minimum much faster
    # than the plain line-searched descent above (400 its -> ~1e-2).
    def rosen2(p):
        return float((1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2)

    def rosen2_grad(p):
        return np.array([
            -2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2),
            200 * (p[1] - p[0] ** 2)])

    p = np.array([-1.2, 1.0])
    lb = LBFGS(m=8)
    g = rosen2_grad(p)
    E_prev = rosen2(p)
    its_used = 0
    for its_used in range(1, 121):
        d = -lb.direction(g)
        slope = float(g @ d)
        if slope >= 0.0:
            lb.reset()
            d = -g
            slope = float(g @ d)
        p_new, s, E_new, _ = armijo_backtrack(rosen2, p, d, slope,
                                              E0=E_prev)
        if s == 0.0:
            lb.reset()
            p_new, s, E_new, _ = armijo_backtrack(rosen2, p, -g,
                                                  -float(g @ g),
                                                  E0=E_prev)
            if s == 0.0:
                break
        g_new = rosen2_grad(p_new)
        lb.push(p_new - p, g_new - g)
        p, g, E_prev = p_new, g_new, E_new
        if E_prev < 1e-14:
            break
    good = rosen2(p) < 1e-10 and its_used < 120
    ok &= good
    print(f"descent: LBFGS+Armijo Rosenbrock E={rosen2(p):.1e} in "
          f"{its_used} its (plain descent: 1e-2 in 400) "
          f"{'OK' if good else 'FAIL'}")

    # LaplacianH0: the PCG application must match the dense solve of
    # (L + eps D) z = q on a small irregular mesh, with pinned rows
    # acting as identity-on-zero.
    Vm = rng.normal(size=(9, 3)) * 0.3 + np.array(
        [[i % 3, i // 3, 0.0] for i in range(9)])
    Tm = []
    for r in range(2):
        for c in range(2):
            a = r * 3 + c
            Tm += [[a, a + 1, a + 3], [a + 1, a + 4, a + 3]]
    Tm = np.array(Tm)
    freem = np.ones(9, dtype=bool)
    freem[0] = False
    h0 = LaplacianH0(eps=1e-3, tol=1e-10, max_iters=500)
    h0.update(Vm, Tm, free=freem)
    qm = rng.normal(size=(9, 3))
    qm[0] = 0.0
    zm = h0(qm)
    # dense reference on the free subsystem
    try:
        from . import cotan as _cotan
    except ImportError:
        import cotan as _cotan               # type: ignore
    Em, Wm = _cotan.edge_cotan_weights(Vm, Tm, mode="mollify")
    Ld = np.zeros((9, 9))
    for (a, b), w in zip(Em, Wm):
        Ld[a, a] += w
        Ld[b, b] += w
        Ld[a, b] -= w
        Ld[b, a] -= w
    Ld += 1e-3 * np.diag(np.diag(Ld))
    idx = np.nonzero(freem)[0]
    z_ref = np.zeros((9, 3))
    z_ref[idx] = np.linalg.solve(Ld[np.ix_(idx, idx)], qm[idx])
    err = float(np.max(np.abs(zm - z_ref)) / np.max(np.abs(z_ref)))
    good = err < 1e-7 and np.all(zm[0] == 0.0)
    ok &= good
    print(f"descent: LaplacianH0 PCG vs dense solve rel err {err:.1e}, "
          f"pinned row zero {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("solver.descent self-test failed")
