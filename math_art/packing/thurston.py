# Thurston's relaxation for circle packing radii, in the Collins-Stephenson
# formulation.
#
# The problem: assign a radius to every vertex of a triangulation so that the
# angle sum around each interior vertex is exactly 2*pi (or a prescribed "aim"),
# with the boundary radii fixed by a boundary condition.  Thurston's scheme
# sweeps the interior vertices, resetting each radius to the value that closes
# its own flower, and repeats.  Monotonicity of the angle sum in the radius
# (angles.py) makes each such solve well posed, and the sweep converges.
#
# Two refinements from Collins-Stephenson turn that into a usable algorithm.
#
# THE UNIFORM NEIGHBOUR MODEL removes the inner root-find.  Solving
# theta(r) = aim exactly for a general flower needs an iteration; instead one
# computes the "reference label" -- the uniform petal radius that reproduces the
# flower's true angle sum at the current radius -- and solves the UNIFORM
# problem, which inverts in closed form.  Their Lemma 3.1 guarantees the step
# moves toward the true solution.  Sweep counts are unchanged, but each sweep
# costs one closed-form update per vertex instead of dozens of angle-sum
# evaluations.
#
# SUPER-STEP ACCELERATION reduces the number of sweeps, which matters because
# the unaccelerated error-reduction factor behaves as lambda ~ N/(N + C) with
# C ~ 30: it tends to 1 as the packing grows, so plain sweeping stalls.  After
# each sweep the radius vector moves in a nearly constant direction, so
# extrapolating along it is safe provided the extrapolated labels stay in range
# (r > 0 euclidean, 0 < s < 1 hyperbolic).  On their 223-vertex test this cut
# 297 sweeps to 46 (euclidean) and 220 to 36 (hyperbolic).
#
# Spherical packings are NOT computed here -- the spherical angle-sum kernel is
# not monotone, so this method does not apply.  See `sphere.py`.
#
# References:
# - Charles R. Collins and Kenneth Stephenson, "A circle packing algorithm",
#   Computational Geometry: Theory and Applications 25 (2003), pp. 233-256
#   (Lemma 2.2; the Uniform Neighbour Model of section 3.2; the algorithm of
#   Table 1; the acceleration results of Table 2).
# - William P. Thurston, "The Geometry and Topology of Three-Manifolds",
#   Princeton lecture notes, 1980, chapter 13.
# - Kenneth Stephenson, "Introduction to Circle Packing: The Theory of Discrete
#   Analytic Functions", Cambridge University Press, 2005.

import math

from . import angles
from .angles import EUCLIDEAN, HYPERBOLIC

MAXIMAL = 'MAXIMAL'
PRESCRIBED = 'PRESCRIBED'
FREE = 'FREE'

TWO_PI = 2.0 * math.pi


class PackResult:
    """Radii plus the diagnostics a caller should report."""

    __slots__ = ('radii', 'geom', 'sweeps', 'residual', 'converged',
                 'max_angle_error')

    def __init__(self, radii, geom, sweeps, residual, converged,
                 max_angle_error):
        self.radii = radii
        self.geom = geom
        self.sweeps = sweeps
        self.residual = residual
        self.converged = converged
        self.max_angle_error = max_angle_error

    def __repr__(self):
        return ("PackResult(%s, %d circles, %d sweeps, angle error %.3e, %s)"
                % (self.geom, len(self.radii), self.sweeps,
                   self.max_angle_error,
                   'converged' if self.converged else 'STALLED'))


def _in_range(geom, v):
    if geom == HYPERBOLIC:
        return 1e-14 < v < 1.0 - 1e-14
    return v > 1e-14


def _clip(geom, v):
    if geom == HYPERBOLIC:
        return min(max(v, 1e-14), 1.0 - 1e-14)
    return max(v, 1e-14)


def pack(K, geom=EUCLIDEAN, boundary=FREE, boundary_radii=None, aims=None,
         initial=None, tol=1e-10, max_sweeps=4000, accelerate=True,
         delta=0.02):
    """Solve for packing radii on the complex K.

    geom            EUCLIDEAN (radii) or HYPERBOLIC (s-radii, s = exp(-2r)).
    boundary        MAXIMAL   -- hyperbolic only; boundary circles are
                                 horocycles (s = 0), giving the maximal packing
                                 of the disc, i.e. the discrete Riemann map.
                    PRESCRIBED -- boundary radii taken from `boundary_radii`.
                    FREE      -- boundary radii held at their initial values.
    aims            per-vertex target angle sums; default 2*pi at interior
                    vertices.  Cone points are set by raising an aim.
    Returns a PackResult.
    """
    if geom not in (EUCLIDEAN, HYPERBOLIC):
        raise ValueError("pack() handles EUCLIDEAN and HYPERBOLIC only; "
                         "spherical packings go through sphere.py")
    if boundary == MAXIMAL and geom != HYPERBOLIC:
        raise ValueError("the MAXIMAL boundary condition is hyperbolic")

    nv = K.nv
    if aims is None:
        aims = [TWO_PI] * nv
    K.validate(aim=max(aims[v] for v in K.interior) if K.interior else None)

    # ---- initial label ---------------------------------------------------
    if initial is not None:
        R = list(initial)
    elif geom == HYPERBOLIC:
        # start with labels deliberately too large (small s, big radius): the
        # Perron "superpacking" side, where angle sums are too small
        R = [0.05] * nv
    else:
        R = [0.5] * nv

    # ---- boundary condition ---------------------------------------------
    if boundary == MAXIMAL:
        for w in K.boundary:
            R[w] = 0.0                       # horocycle
    elif boundary == PRESCRIBED:
        if boundary_radii is None:
            raise ValueError("PRESCRIBED needs boundary_radii")
        for w in K.boundary:
            R[w] = _clip(geom, boundary_radii[w])

    free = [v for v in sorted(K.interior)]
    if not free:
        return PackResult(R, geom, 0, 0.0, True, 0.0)

    flowers = K.flowers
    unm = (angles.unm_update_hyperbolic if geom == HYPERBOLIC
           else angles.unm_update_euclidean)

    # ---- Collins-Stephenson Table 1 --------------------------------------
    c = tol + 1.0
    lam = -1.0
    flag = 0
    sweeps = 0
    while c > tol and sweeps < max_sweeps:
        c0, lam0, flag0 = c, lam, flag
        R0 = list(R)
        c = 0.0
        for v in free:
            petals = [R[u] for u in flowers[v]]
            theta = angles.angle_sum(geom, R[v], petals, closed=True)
            aim = aims[v]
            nr = unm(R[v], theta, aim, len(petals))
            R[v] = _clip(geom, nr)
            d = theta - aim
            c += d * d
        c = math.sqrt(c)
        lam = c / c0 if c0 > 0.0 else -1.0
        flag = 1
        sweeps += 1

        if accelerate and flag0 == 1 and 0.0 < lam < 1.0:
            c *= lam
            if abs(lam - lam0) < delta:
                lam = lam / (1.0 - lam)
            # largest step keeping every label legal, capped at half of it
            lam_star = _max_step(geom, R, R0)
            step = min(lam, 0.5 * lam_star)
            if step > 0.0:
                for v in free:
                    R[v] = _clip(geom, R[v] + step * (R[v] - R0[v]))
            flag = 0

    worst = 0.0
    for v in free:
        petals = [R[u] for u in flowers[v]]
        worst = max(worst, abs(angles.angle_sum(geom, R[v], petals) - aims[v]))
    return PackResult(R, geom, sweeps, c, c <= tol, worst)


def _max_step(geom, R, R0):
    """Largest t with R + t*(R - R0) still a legal label everywhere."""
    best = float('inf')
    for v in range(len(R)):
        d = R[v] - R0[v]
        if d == 0.0:
            continue
        if geom == HYPERBOLIC:
            lim = (1.0 - R[v]) / d if d > 0 else (0.0 - R[v]) / d
        else:
            if d >= 0:
                continue
            lim = (0.0 - R[v]) / d
        if lim > 0.0:
            best = min(best, lim)
    return 0.0 if best == float('inf') else best


def h_radius(s):
    """Hyperbolic radius from an s-value (inf for a horocycle)."""
    if s <= 0.0:
        return float('inf')
    return -0.5 * math.log(s)


def s_value(r):
    """s-value from a hyperbolic radius."""
    return math.exp(-2.0 * r)


def _selftest():
    """Convergence and accuracy checks; raises AssertionError on failure."""
    from . import complexes

    # 1. euclidean, prescribed boundary: the boundary-value packing used as the
    #    reference case throughout the plan
    K, xy = complexes.hex_disc(5)
    assert K.nv == 91
    br = [0.0] * K.nv
    for w in K.boundary:
        x, y = xy[w]
        br[w] = 0.35 * (1.0 + 0.75 * math.cos(3.0 * math.atan2(y, x)))
    res = pack(K, EUCLIDEAN, boundary=PRESCRIBED, boundary_radii=br)
    assert res.converged, res
    assert res.max_angle_error < 1e-9, res.max_angle_error

    # 2. acceleration changes speed, not the answer
    slow = pack(K, EUCLIDEAN, boundary=PRESCRIBED, boundary_radii=br,
                accelerate=False)
    assert slow.converged
    drift = max(abs(a - b) for a, b in zip(slow.radii, res.radii))
    assert drift < 1e-6, "acceleration changed the solution by %.3e" % drift
    assert res.sweeps <= slow.sweeps, (res.sweeps, slow.sweeps)

    # 3. a uniform boundary gives the exactly regular hexagonal packing
    K2, _ = complexes.hex_disc(3)
    br2 = [1.0] * K2.nv
    r2 = pack(K2, EUCLIDEAN, boundary=PRESCRIBED, boundary_radii=br2)
    assert r2.converged
    for v in K2.interior:
        assert abs(r2.radii[v] - 1.0) < 1e-8, \
            "uniform boundary must give uniform radii, got %.6f" % r2.radii[v]

    # 4. hyperbolic maximal packing: horocycles on the boundary, and every
    #    interior angle sum still 2*pi
    res_h = pack(K2, HYPERBOLIC, boundary=MAXIMAL)
    assert res_h.converged, res_h
    assert res_h.max_angle_error < 1e-8, res_h.max_angle_error
    for w in K2.boundary:
        assert res_h.radii[w] == 0.0
    for v in K2.interior:
        assert 0.0 < res_h.radii[v] < 1.0

    # 5. cone point: raising one aim to 4*pi is solvable and is honoured
    aims = [TWO_PI] * K2.nv
    centre = min(K2.interior, key=lambda v: abs(v))
    aims[centre] = 2.0 * TWO_PI
    res_c = pack(K2, EUCLIDEAN, boundary=PRESCRIBED, boundary_radii=br2,
                 aims=aims)
    assert res_c.max_angle_error < 1e-8, res_c.max_angle_error

    # 6. s-value round trip
    for r in (0.1, 1.0, 5.0):
        assert abs(h_radius(s_value(r)) - r) < 1e-12
    assert h_radius(0.0) == float('inf')

    print("packing.thurston: boundary-value and maximal packings converge, "
          "angle error < 1e-8, acceleration speed-only, cone points honoured. "
          "RESULT: OK")
