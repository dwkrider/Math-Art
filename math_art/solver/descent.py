# Step-size control for gradient-style relaxation loops.
#
# Part of the shared solver core (`math_art/solver/`).  NumPy only.
#
# Two mechanisms, both replacing hand-tuned fixed gains:
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
# References:
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992) -- the optimizing-scale line search (iterate.c).
#   A. Rossiter, Antiprism (antiprism.com), `canonical` -- the adaptive
#       gain schedule.  Includes ideas and algorithms by George W. Hart.
#   J. Nocedal and S. J. Wright, "Numerical Optimization", 2nd ed.,
#       Springer (2006), ch. 3 -- line-search fundamentals.

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

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("solver.descent self-test failed")
