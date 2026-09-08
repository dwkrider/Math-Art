# Angle-sum kernels for circle packing, in the three constant-curvature
# geometries.
#
# A circle packing assigns a radius to every vertex of a triangulation so that
# circles at the ends of each edge are tangent.  The whole computation turns on
# one quantity: the angle a circle of radius r subtends at its own centre in the
# triangle it forms with two tangent neighbours of radii r1 and r2.  The three
# triangle side lengths are always
#
#     a = r + r1,   b = r + r2,   c = r1 + r2
#
# and the angle at the centre follows from the law of cosines for the geometry.
#
# EUCLIDEAN    cos(alpha) = (a^2 + b^2 - c^2) / (2 a b)
# HYPERBOLIC   cos(alpha) = (cosh a cosh b - cosh c) / (sinh a sinh b)
# SPHERICAL    cos(alpha) = (cos c - cos a cos b) / (sin a sin b)
#
# S-RADII (hyperbolic).  A maximal packing of the disc drives the boundary radii
# to infinity, where cosh overflows and the formula above is unusable -- and
# those are exactly the packings one most wants.  Following Stephenson's
# CirclePack we therefore store a hyperbolic radius as its "s-value"
#
#     s = exp(-2 r)  in  (0, 1],      s -> 0  as  r -> infinity
#
# with s = 0 denoting a horocycle (a circle of infinite hyperbolic radius,
# internally tangent to the unit circle).  Writing A = s*s1 and B = s*s2, and
# multiplying numerator and denominator of the hyperbolic law of cosines by
# 4 exp(-a-b), every exponential collapses into products of s-values:
#
#     cos(alpha) = [ (1 + A)(1 + B) - 2 s (1 + s1 s2) ] / [ (1 - A)(1 - B) ]
#
# which is bounded, overflow-free, and continuous down to s = 0.
#
# MONOTONICITY.  In the euclidean and hyperbolic cases the angle sum of a closed
# flower is strictly decreasing in r, tends to k*pi as r -> 0 and to 0 as
# r -> infinity, so a target angle sum A has a unique solution exactly when
# 0 < A < k*pi (Collins-Stephenson, Lemma 2.2).  The spherical kernel is NOT
# monotone -- alpha falls and then rises again once the sides pass pi/2 -- so it
# must not be driven by a monotone solver; spherical packings are computed by
# projection instead (see `sphere.py`).
#
# References:
# - Kenneth Stephenson, "Introduction to Circle Packing: The Theory of Discrete
#   Analytic Functions", Cambridge University Press, 2005.
# - Charles R. Collins and Kenneth Stephenson, "A circle packing algorithm",
#   Computational Geometry: Theory and Applications 25 (2003), pp. 233-256
#   (Lemma 2.2 monotonicity; the Uniform Neighbour Model).
# - William P. Thurston, "The Geometry and Topology of Three-Manifolds",
#   Princeton lecture notes, 1980, chapter 13 (the relaxation scheme).

import math

EUCLIDEAN = 'EUCLIDEAN'
HYPERBOLIC = 'HYPERBOLIC'
SPHERICAL = 'SPHERICAL'

GEOMETRIES = (EUCLIDEAN, HYPERBOLIC, SPHERICAL)


def _clamp_acos(x):
    """acos with the argument pinned to [-1, 1] against rounding drift."""
    if x <= -1.0:
        return math.pi
    if x >= 1.0:
        return 0.0
    return math.acos(x)


def corner_euclidean(r, r1, r2):
    """Angle at the centre of the circle of radius r, between tangencies with
    circles of radii r1 and r2."""
    a = r + r1
    b = r + r2
    c = r1 + r2
    return _clamp_acos((a * a + b * b - c * c) / (2.0 * a * b))


def corner_hyperbolic(s, s1, s2):
    """Hyperbolic corner angle, in s-radii (s = exp(-2r); s = 0 is a horocycle).

    Derived by multiplying the hyperbolic law of cosines through by
    4*exp(-a-b); see the module header.  Stable for every s in [0, 1)."""
    A = s * s1
    B = s * s2
    den = (1.0 - A) * (1.0 - B)
    if den <= 1e-300:
        # both neighbours coincide with a unit-s (zero radius) circle
        return 0.0
    num = (1.0 + A) * (1.0 + B) - 2.0 * s * (1.0 + s1 * s2)
    return _clamp_acos(num / den)


def corner_spherical(r, r1, r2):
    """Spherical corner angle.  Correct as a formula, but NOT monotone in r --
    see the module header before using it inside a solver."""
    a = r + r1
    b = r + r2
    c = r1 + r2
    if a >= math.pi or b >= math.pi or c >= math.pi:
        return 0.0
    sa = math.sin(a)
    sb = math.sin(b)
    if sa <= 1e-15 or sb <= 1e-15:
        return 0.0
    return _clamp_acos((math.cos(c) - math.cos(a) * math.cos(b)) / (sa * sb))


def corner(geom, r, r1, r2):
    """Corner angle in the named geometry.  For HYPERBOLIC the arguments are
    s-radii, not radii."""
    if geom == EUCLIDEAN:
        return corner_euclidean(r, r1, r2)
    if geom == HYPERBOLIC:
        return corner_hyperbolic(r, r1, r2)
    if geom == SPHERICAL:
        return corner_spherical(r, r1, r2)
    raise ValueError("unknown geometry %r" % (geom,))


def angle_sum(geom, r, petals, closed=True):
    """Sum of the corner angles around a vertex of radius r whose neighbours, in
    cyclic order, have the radii in `petals`.

    `closed` selects an interior vertex (the flower wraps) as against a boundary
    vertex (the flower is a fan and the last pair is not joined)."""
    n = len(petals)
    if n < 2:
        return 0.0
    total = 0.0
    last = n if closed else n - 1
    for k in range(last):
        total += corner(geom, r, petals[k], petals[(k + 1) % n])
    return total


# --------------------------------------------------------------------------
# The Uniform Neighbour Model: a closed-form replacement for the inner solve.
#
# For a flower of k petals all of the same radius rhat around a centre of radius
# r, every corner is equal, so the euclidean angle sum is
#
#     theta = 2 k asin( rhat / (r + rhat) )
#
# which inverts directly.  Given the TRUE angle sum theta at the current radius
# r0, the "reference label" rhat is the uniform petal radius reproducing it;
# solving the uniform problem at the target aim then gives the update.  By
# Collins-Stephenson Lemma 3.1 this step always moves toward the true solution.
# --------------------------------------------------------------------------

def unm_update_euclidean(r0, theta, aim, k):
    """One Uniform Neighbour Model step, euclidean.  Two sines, no iteration."""
    s = math.sin(theta / (2.0 * k))
    if s <= 0.0 or s >= 1.0:
        return r0
    rhat = r0 * s / (1.0 - s)
    t = math.sin(aim / (2.0 * k))
    if t <= 0.0 or t >= 1.0:
        return r0
    return rhat * (1.0 - t) / t


def unm_update_hyperbolic(s0, theta, aim, k):
    """One Uniform Neighbour Model step in s-radii.

    The uniform hyperbolic flower satisfies, with x the petal s-value and s the
    centre s-value,

        cos(theta/k) = [ (1 + s x)^2 - 2 s (1 + x^2) ] / (1 - s x)^2

    Solving that quadratic in x gives the reference label; re-solving at the aim
    gives the update.  Falls back to no change where the quadratic has no root
    in (0, 1), which happens only for labels already outside the feasible set."""
    x = _uniform_petal_h(s0, theta / k)
    if x is None:
        return s0
    return _uniform_centre_h(x, aim / k)


def _uniform_petal_h(s, alpha):
    """s-value of the uniform petal giving corner angle `alpha` around centre s."""
    ca = math.cos(alpha)
    # (1 + s x)^2 - 2 s (1 + x^2) = ca (1 - s x)^2
    # => x^2 (s^2 - 2 s - ca s^2) + x (2 s + 2 ca s) + (1 - 2 s - ca) = 0
    A = s * s - 2.0 * s - ca * s * s
    B = 2.0 * s + 2.0 * ca * s
    C = 1.0 - 2.0 * s - ca
    return _quad_root_unit(A, B, C)


def _uniform_centre_h(x, alpha):
    """s-value of the centre giving corner angle `alpha` with uniform petals x."""
    ca = math.cos(alpha)
    # same relation, solved for s instead
    A = x * x - ca * x * x
    B = 2.0 * x - 2.0 - 2.0 * x * x + 2.0 * ca * x
    C = 1.0 - ca
    return _quad_root_unit(A, B, C)


def _quad_root_unit(A, B, C):
    """Root of A t^2 + B t + C lying in [0, 1), or None."""
    if abs(A) < 1e-14:
        if abs(B) < 1e-14:
            return None
        t = -C / B
        return t if 0.0 <= t < 1.0 else None
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return None
    rt = math.sqrt(disc)
    for t in ((-B + rt) / (2.0 * A), (-B - rt) / (2.0 * A)):
        if 0.0 <= t < 1.0:
            return t
    return None


def _selftest():
    """Numeric checks; raises AssertionError on failure."""
    # 1. euclidean corner against an explicit construction
    for (r, r1, r2) in ((1.0, 1.0, 1.0), (0.3, 1.7, 0.9), (2.5, 0.2, 0.2)):
        a, b, c = r + r1, r + r2, r1 + r2
        want = math.acos((a * a + b * b - c * c) / (2 * a * b))
        got = corner(EUCLIDEAN, r, r1, r2)
        assert abs(got - want) < 1e-12, (got, want)

    # 2. the three equal-radius circles subtend exactly 60 degrees
    assert abs(corner(EUCLIDEAN, 1.0, 1.0, 1.0) - math.pi / 3.0) < 1e-12

    # 3. hyperbolic s-radii agree with the direct cosh form at moderate radii
    for (r, r1, r2) in ((0.5, 0.7, 0.4), (1.0, 1.0, 1.0), (0.2, 1.3, 0.8)):
        a, b, c = r + r1, r + r2, r1 + r2
        want = math.acos((math.cosh(a) * math.cosh(b) - math.cosh(c))
                         / (math.sinh(a) * math.sinh(b)))
        got = corner(HYPERBOLIC, math.exp(-2 * r), math.exp(-2 * r1),
                     math.exp(-2 * r2))
        assert abs(got - want) < 1e-10, (r, r1, r2, got, want)

    # 4. horocycles: s = 0 is finite and gives a zero angle at the centre
    assert abs(corner(HYPERBOLIC, 0.0, 0.5, 0.5)) < 1e-12
    v = corner(HYPERBOLIC, 0.4, 0.0, 0.0)
    assert 0.0 <= v <= math.pi and math.isfinite(v)

    # 5. euclidean angle sum is strictly decreasing in r; spherical is not
    petals = [0.6, 0.9, 0.4, 1.2, 0.7]
    prev = angle_sum(EUCLIDEAN, 0.05, petals)
    for r in (0.2, 0.5, 0.9, 1.4, 2.0, 3.0):
        cur = angle_sum(EUCLIDEAN, r, petals)
        assert cur < prev, "euclidean angle sum must decrease in r"
        prev = cur

    # 6. the UNM update reproduces the exact solve
    for petals in ([0.5] * 5, [0.3, 1.1, 0.7, 0.9], [1.0] * 6):
        k = len(petals)
        r = 1.0
        for _ in range(400):
            th = angle_sum(EUCLIDEAN, r, petals)
            r = unm_update_euclidean(r, th, 2 * math.pi, k)
        assert abs(angle_sum(EUCLIDEAN, r, petals) - 2 * math.pi) < 1e-12, r

    # 7. the hyperbolic UNM update likewise drives the angle sum to its aim
    petals = [0.4] * 6
    s = 0.5
    for _ in range(400):
        th = angle_sum(HYPERBOLIC, s, petals)
        s = unm_update_hyperbolic(s, th, 2 * math.pi, len(petals))
        s = min(max(s, 1e-12), 1.0 - 1e-12)
    assert abs(angle_sum(HYPERBOLIC, s, petals) - 2 * math.pi) < 1e-9

    print("packing.angles: euclidean/hyperbolic kernels agree with the direct "
          "forms, s-radii stable at horocycles, UNM converges. RESULT: OK")
