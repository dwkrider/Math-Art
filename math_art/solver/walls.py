# Level-set wall constraints: vertices sliding on analytic surfaces.
#
# Part of the shared solver core (`math_art/solver/`).  NumPy only.
#
# A wall is a scalar level set f(x) = 0 with an analytic gradient.
# Vertices flagged to a wall are
#
#   * Newton-projected back onto it after every move,
#         x <- x - f(x) grad f / |grad f|^2,
#     iterated to tolerance -- one constraint projects ALL its vertices
#     at once (the common case), so the loop vectorizes;
#   * velocity tangent-projected, v <- v - (v.n) n with
#     n = grad f / |grad f|, so the motion slides along the wall.
#
# One-sided walls (f >= 0 is the allowed side) use the simple
# activation rule: positions are projected only when violating, and a
# velocity is constrained only at a vertex ON the wall whose velocity
# points into the forbidden side.  The full active-set bookkeeping
# (hit bits, release on wrong-signed multipliers -- Evolver
# fixvol.c:one_sided_lagrange_adjust) is deliberately NOT implemented:
# it buys rocking-drop dynamics nobody has asked for.
#
# Composition with volume constraints happens in solver/volume.py:
# `evolve(walls=...)` wall-projects the velocity AND the volume
# gradients before the Gram solve (so the projected motion is tangent
# to both constraint sets and the KKT point is the true constrained
# equilibrium), and the volume restore re-projects positions onto the
# walls inside its Newton loop.
#
# References:
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992) -- level-set constraint walls; the projection scheme is
#       cnstrnt.c (constr_proj, project_v_constr) of the 2.70 source.
#   K. A. Brakke, "Surface Evolver Manual", v2.70 (2013), ch. on
#       constraints -- one-sided constraints and constraint energies.

import numpy as np


class Wall:
    """Level set f(x) = 0 with analytic gradient; subclasses implement
    f(X) -> (n,) and grad(X) -> (n, 3) for X of shape (n, 3).
    one_sided=True means f >= 0 is the allowed side."""

    one_sided = False

    def f(self, X):
        raise NotImplementedError

    def grad(self, X):
        raise NotImplementedError


class PlaneWall(Wall):
    """f = (x - point) . n_hat, n_hat the unit normal."""

    def __init__(self, point, normal, one_sided=False):
        self.point = np.asarray(point, float)
        n = np.asarray(normal, float)
        self.normal = n / np.linalg.norm(n)
        self.one_sided = bool(one_sided)

    def f(self, X):
        return (np.asarray(X, float) - self.point) @ self.normal

    def grad(self, X):
        return np.broadcast_to(self.normal,
                               np.shape(np.asarray(X, float))).copy()


class SphereWall(Wall):
    """f = sign * (|x - c| - R).  sign=+1: one-sided keeps vertices
    OUTSIDE the sphere; sign=-1 keeps them INSIDE."""

    def __init__(self, center, radius, one_sided=False, sign=1.0):
        self.center = np.asarray(center, float)
        self.radius = float(radius)
        self.sign = float(sign)
        self.one_sided = bool(one_sided)

    def f(self, X):
        d = np.asarray(X, float) - self.center
        return self.sign * (np.linalg.norm(d, axis=-1) - self.radius)

    def grad(self, X):
        d = np.asarray(X, float) - self.center
        r = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-300)
        return self.sign * d / r


class CylinderWall(Wall):
    """f = sign * (dist to axis - R); axis through `point` along unit
    `axis`.  sign=+1: one-sided keeps vertices OUTSIDE the cylinder."""

    def __init__(self, point, axis, radius, one_sided=False, sign=1.0):
        self.point = np.asarray(point, float)
        a = np.asarray(axis, float)
        self.axis = a / np.linalg.norm(a)
        self.radius = float(radius)
        self.sign = float(sign)
        self.one_sided = bool(one_sided)

    def _perp(self, X):
        d = np.asarray(X, float) - self.point
        return d - np.outer(d @ self.axis, self.axis)

    def f(self, X):
        return self.sign * (np.linalg.norm(self._perp(X), axis=-1)
                            - self.radius)

    def grad(self, X):
        p = self._perp(X)
        r = np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-300)
        return self.sign * p / r


class FunctionWall(Wall):
    """General wall from callables f(X)->(n,) and grad(X)->(n,3)."""

    def __init__(self, f, grad, one_sided=False):
        self._f = f
        self._grad = grad
        self.one_sided = bool(one_sided)

    def f(self, X):
        return np.asarray(self._f(np.asarray(X, float)), float)

    def grad(self, X):
        return np.asarray(self._grad(np.asarray(X, float)), float)


# --------------------------------------------------------------------------
# projection operators
# --------------------------------------------------------------------------

def newton_project(wall, X, mask=None, tol=1e-12, max_rounds=30):
    """Project the masked rows of X (in place) onto the wall: to f = 0
    for a two-sided wall, back to f >= 0 for a one-sided one (rows
    already on the allowed side are untouched).  Vectorized Newton
    x += -f grad f / |grad f|^2.  Returns the worst remaining
    violation (max |f| two-sided, max(-f, 0) one-sided)."""
    idx = (np.arange(len(X)) if mask is None
           else np.nonzero(np.asarray(mask, bool))[0])
    if len(idx) == 0:
        return 0.0
    for _ in range(max_rounds):
        fv = wall.f(X[idx])
        act = (fv < -tol) if wall.one_sided else (np.abs(fv) > tol)
        if not act.any():
            break
        rows = idx[act]
        g = wall.grad(X[rows])
        g2 = np.maximum(np.einsum('ij,ij->i', g, g), 1e-300)
        X[rows] -= (fv[act] / g2)[:, None] * g
    fv = wall.f(X[idx])
    if wall.one_sided:
        return float(max(0.0, -fv.min())) if len(fv) else 0.0
    return float(np.max(np.abs(fv))) if len(fv) else 0.0


def tangent_project(vel, wall, X, mask=None, act_tol=1e-9):
    """Remove the wall-normal component of `vel` (in place) at the
    constrained rows: v <- v - (v.n) n.  For a one-sided wall only
    rows ON the wall (f <= act_tol) with velocity INTO the forbidden
    side (v.n < 0) are constrained -- the simple activation rule.
    Returns the worst |v.n| that was removed."""
    idx = (np.arange(len(X)) if mask is None
           else np.nonzero(np.asarray(mask, bool))[0])
    if len(idx) == 0:
        return 0.0
    n = wall.grad(X[idx])
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    vn = np.einsum('ij,ij->i', vel[idx], n)
    if wall.one_sided:
        fv = wall.f(X[idx])
        act = (fv <= act_tol) & (vn < 0.0)
        idx, n, vn = idx[act], n[act], vn[act]
    if len(idx) == 0:
        return 0.0
    vel[idx] -= vn[:, None] * n
    return float(np.max(np.abs(vn)))


def project_all(V, walls, tol=1e-12, sweeps=5):
    """Project V (in place) onto every (wall, mask) pair.  Disjoint
    masks need one sweep; a vertex under several walls gets alternating
    projections (exact only where the walls meet transversally --
    prefer `fixed=` for genuinely intersecting constraints).  Returns
    the worst residual after the final sweep."""
    worst = np.inf
    for _ in range(max(1, sweeps)):
        for w, m in walls:
            newton_project(w, V, m, tol=tol)
        worst = wall_residual(V, walls)
        if worst <= tol:
            break
    return worst


def wall_residual(V, walls):
    """Worst violation over every (wall, mask) pair (no movement)."""
    worst = 0.0
    for w, m in walls:
        idx = (np.arange(len(V)) if m is None
               else np.nonzero(np.asarray(m, bool))[0])
        if len(idx) == 0:
            continue
        fv = w.f(V[idx])
        if w.one_sided:
            worst = max(worst, float(max(0.0, -fv.min())))
        else:
            worst = max(worst, float(np.max(np.abs(fv))))
    return worst


def normal_component_max(field, V, walls):
    """max |field . n_hat| over the TWO-SIDED constrained rows -- the
    velocity-tangency residual.  One-sided rows are excluded: an
    inactive vertex legitimately moves off its wall."""
    worst = 0.0
    for w, m in walls:
        if w.one_sided:
            continue
        idx = (np.arange(len(V)) if m is None
               else np.nonzero(np.asarray(m, bool))[0])
        if len(idx) == 0:
            continue
        n = w.grad(V[idx])
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True),
                           1e-300)
        worst = max(worst, float(np.max(np.abs(
            np.einsum('ij,ij->i', field[idx], n)))))
    return worst


def tangent_all(field, V, walls, act_tol=1e-9):
    """Tangent-project a per-vertex 3-vector field (in place) against
    every (wall, mask) pair; returns the field.  Sequential projection
    is exact for disjoint masks; for a vertex under two walls it is the
    alternating approximation (see project_all)."""
    for w, m in walls:
        tangent_project(field, w, V, m, act_tol=act_tol)
    return field


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(7)
    ok = True

    plane = PlaneWall([0.0, 0.0, 1.0], [0.0, 1.0, 2.0])
    sph = SphereWall([0.5, -0.2, 0.0], 1.3)
    cyl = CylinderWall([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.8)

    # --- analytic gradients vs central differences ---------------------
    X = rng.normal(0.0, 2.0, (40, 3))
    h = 1e-6
    for name, w in (("plane", plane), ("sphere", sph), ("cyl", cyl)):
        g = w.grad(X)
        worst = 0.0
        for k in range(3):
            dX = np.zeros(3)
            dX[k] = h
            fd = (w.f(X + dX) - w.f(X - dX)) / (2 * h)
            worst = max(worst, float(np.max(np.abs(fd - g[:, k]))))
        good = worst < 1e-8
        ok &= good
        print(f"walls: {name} grad vs FD max err {worst:.2e} "
              f"{'OK' if good else 'FAIL'}")

    # --- Newton projection: machine-precision residual, and for these
    # constant-|grad| walls the Newton path IS the closest point -------
    for name, w, closest in (
            ("plane", plane,
             lambda P: P - np.outer(w0.f(P), w0.normal)),
            ("sphere", sph,
             lambda P: sph.center + 1.3 * (P - sph.center)
             / np.linalg.norm(P - sph.center, axis=1, keepdims=True)),
    ):
        w0 = w
        P = rng.normal(0.0, 2.0, (60, 3))
        Q = P.copy()
        res = newton_project(w, Q)
        exact = closest(P)
        dev = float(np.max(np.linalg.norm(Q - exact, axis=1)))
        good = res < 1e-12 and dev < 1e-9
        ok &= good
        print(f"walls: {name} projection residual {res:.1e}, closest-"
              f"point dev {dev:.1e} {'OK' if good else 'FAIL'}")
    Q = rng.normal(0.0, 2.0, (60, 3))
    res = newton_project(cyl, Q)
    good = res < 1e-12
    ok &= good
    print(f"walls: cylinder projection residual {res:.1e} "
          f"{'OK' if good else 'FAIL'}")

    # --- masked projection leaves unmasked rows untouched --------------
    P = rng.normal(0.0, 2.0, (30, 3))
    mask = np.zeros(30, dtype=bool)
    mask[::3] = True
    Q = P.copy()
    newton_project(sph, Q, mask)
    good = (np.array_equal(Q[~mask], P[~mask])
            and float(np.max(np.abs(sph.f(Q[mask])))) < 1e-12)
    ok &= good
    print(f"walls: masked projection touches only its rows "
          f"{'OK' if good else 'FAIL'}")

    # --- tangency: v.n = 0 after projection, exactly where asked ------
    P = rng.normal(0.0, 2.0, (30, 3))
    newton_project(sph, P)
    v = rng.normal(size=(30, 3))
    v0 = v.copy()
    tangent_project(v, sph, P)
    n = sph.grad(P)
    vn = float(np.max(np.abs(np.einsum('ij,ij->i', v, n))))
    good = vn < 1e-14
    ok &= good
    print(f"walls: tangent projection max |v.n| {vn:.1e} "
          f"{'OK' if good else 'FAIL'}")
    # the removed component is normal: v0 - v parallel to n
    dv = v0 - v
    cross = np.linalg.norm(np.cross(dv, n), axis=1)
    good = float(np.max(cross)) < 1e-12
    ok &= good
    print(f"walls: removed component is purely normal "
          f"{'OK' if good else 'FAIL'}")

    # --- one-sided: only violators move; only on-wall inward-moving
    # velocities are constrained ---------------------------------------
    floor = PlaneWall([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], one_sided=True)
    P = rng.normal(0.0, 1.0, (50, 3))
    above = P[:, 2] > 0
    Q = P.copy()
    res = newton_project(floor, Q)
    good = (np.array_equal(Q[above], P[above])
            and float(Q[:, 2].min()) >= -1e-12 and res < 1e-12)
    ok &= good
    print(f"walls: one-sided projects only violators (res {res:.1e}) "
          f"{'OK' if good else 'FAIL'}")
    # velocities: on-wall + pushing down -> constrained; else untouched
    Q2 = Q.copy()
    Q2[0], Q2[1], Q2[2] = (0.3, 0.1, 0.0), (0.4, 0.2, 0.0), (0.1, 0.0, 0.5)
    v = np.zeros((50, 3))
    v[0] = (1.0, 0.0, -2.0)      # on wall, pushing in -> constrained
    v[1] = (1.0, 0.0, +2.0)      # on wall, lifting off -> free
    v[2] = (0.0, 0.0, -2.0)      # off wall -> free
    tangent_project(v, floor, Q2)
    good = (np.allclose(v[0], (1.0, 0.0, 0.0))
            and np.allclose(v[1], (1.0, 0.0, 2.0))
            and np.allclose(v[2], (0.0, 0.0, -2.0)))
    ok &= good
    print(f"walls: one-sided activation rule "
          f"{'OK' if good else 'FAIL'}")

    # --- FunctionWall + project_all/tangent_all/wall_residual ----------
    fw = FunctionWall(lambda X: X[:, 0] + X[:, 1] ** 2 - 1.0,
                      lambda X: np.stack(
                          [np.ones(len(X)), 2.0 * X[:, 1],
                           np.zeros(len(X))], axis=1))
    P = rng.normal(0.0, 1.0, (40, 3))
    m1 = np.zeros(40, dtype=bool)
    m1[:20] = True
    m2 = ~m1
    pairs = [(fw, m1), (sph, m2)]
    worst = project_all(P, pairs)
    good = worst < 1e-12 and wall_residual(P, pairs) < 1e-12
    ok &= good
    print(f"walls: project_all over two masked walls residual "
          f"{worst:.1e} {'OK' if good else 'FAIL'}")
    v = rng.normal(size=(40, 3))
    tangent_all(v, P, pairs)
    n1 = fw.grad(P[m1])
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    n2 = sph.grad(P[m2])
    t1 = float(np.max(np.abs(np.einsum('ij,ij->i', v[m1], n1))))
    t2 = float(np.max(np.abs(np.einsum('ij,ij->i', v[m2], n2))))
    good = max(t1, t2) < 1e-14
    ok &= good
    print(f"walls: tangent_all max |v.n| {max(t1, t2):.1e} "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("solver.walls self-test failed")
