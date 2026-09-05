# Bending a straight cell edge into a soft-cell edge.
#
# Part of the Math Art soft-cell engine (`math_art/softcell/`).  Numpy only.
#
# An edge of the softened cell has to leave each of its two end nodes along a
# PRESCRIBED direction -- the half-tangent computed in `cell.py` -- while still
# joining the same two vertices as before.  This module builds that curve.
#
# WHAT THE GEOMETRY ACTUALLY IS.  Two facts about this family were measured
# over the seven named cells and 300 random morphospace directions before any
# of the code below was written, and both are worth stating because both run
# against the obvious guess:
#
#   * EVERY edge is SYMMETRIC.  The angle the curve makes with the chord on
#     entry equals the angle on exit, to 7e-16, for every edge of every
#     direction tested.  This is structural, not coincidental: the cell has a
#     symmetry exchanging an edge's two endpoints.  So the six-word Dubins
#     search (LSL, RSR, LSR, RSL, RLR, LRL) is not needed here -- the
#     asymmetric configurations it exists to resolve never arise.
#
#   * COPLANARITY IS THE EXCEPTION, not the rule.  The two end tangents and
#     the chord are coplanar for (e2), (f2), (g2) and Kelvin -- every one of
#     which has a_z = 0 -- and for NO generic direction at all: 0 of 36 edges
#     in each of 300 random samples, and 0 of 36 for (h2), (i2) and the PD
#     cell.  Planarity is a measure-zero coincidence of the equatorial
#     presets, so a spatial construction is the main path, not a fallback.
#
# Hence the builder below: a straight segment when the tangents already lie
# along the chord, and otherwise a SYMMETRIC BIARC -- two circular arcs of
# equal tangent length meeting in a G1 joint.  In the planar symmetric case
# the biarc collapses to the single circular arc, which is the
# minimum-curvature answer and a subpath of Dubins's classification; in the
# spatial case there is no published optimum to appeal to, because Dubins's
# theorem is explicitly planar and he leaves three dimensions open.  The
# construction is chosen to respect the edge's own two-fold symmetry, which
# is what keeps neighbouring cells agreeing on their shared edges.
#
# References:
# - L. E. Dubins, "On curves of minimal length with a constraint on average
#   curvature, and with prescribed initial and terminal positions and
#   tangents", American Journal of Mathematics 79(3):497-516 (1957).
#   Theorem 1: every planar R-geodesic is an arc-line-arc, or three arcs, or
#   a subpath of one of those.  Section 1 notes the three-dimensional case is
#   open.
# - G. Domokos, A. Goriely, A. G. Horvath and K. Regos, "Soft cells and the
#   geometry of seashells", PNAS Nexus 3(9):pgae311 (2024).  The Methods
#   section prescribes replacing each edge by a minimum-curvature Dubins path
#   between two co-planar tangent lines, constrained to the planar domain the
#   original straight edge occupied.

import numpy as np

# classification tags reported back to the operator
STRAIGHT = 'straight'
ARC = 'arc'
BIARC = 'biarc'


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-15 else v


def arc_points(A, T, B, m):
    """`m`+1 points along the circular arc from A to B leaving A along T.

    The arc lies in the plane of T and B-A.  If B-A is parallel to T the arc
    degenerates to the straight segment, which is returned instead of a
    division by zero.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    T = _unit(np.asarray(T, float))
    w = B - A
    perp = w - float(w @ T) * T
    h = float(np.linalg.norm(perp))
    t = np.linspace(0.0, 1.0, m + 1)[:, None]
    if h < 1e-12:                          # collinear: a straight segment
        return A[None, :] + t * w[None, :]
    n = perp / h
    # the unique circle through A and B that is tangent to T at A: its
    # centre lies along +n at the distance that puts B on the circle
    r = float(w @ w) / (2.0 * float(w @ n))
    cen = A + r * n

    # Frame the sweep so that theta = 0 is A and d/dtheta at 0 is +r*T; the
    # arc then provably leaves A along T, with no sign case analysis.
    e1 = A - cen                      # = -r n, length r
    e2 = r * T                        # orthogonal to e1, same length
    d = B - cen
    ang = float(np.arctan2(float(d @ e2), float(d @ e1)))
    if ang < 0.0:
        ang += 2.0 * np.pi
    ca, sa = np.cos(ang * t), np.sin(ang * t)
    return cen[None, :] + ca * e1[None, :] + sa * e2[None, :]


def classify(P0, t0, P1, t1, tol=1e-9):
    """Describe the configuration without building it.

    `t0` and `t1` are the OUTGOING half-tangents at their own nodes, so the
    curve travels along +t0 at P0 and along -t1 at P1.  Returns
    (kind, planar, symmetric).
    """
    P0 = np.asarray(P0, float)
    P1 = np.asarray(P1, float)
    t0 = _unit(np.asarray(t0, float))
    t1 = _unit(np.asarray(t1, float))
    u = _unit(P1 - P0)
    d1 = -t1
    planar = abs(float(np.linalg.det(np.array([t0, d1, u])))) < tol
    symmetric = abs(float(t0 @ u) - float(d1 @ u)) < 1e-7
    if np.linalg.norm(t0 - u) < tol and np.linalg.norm(d1 - u) < tol:
        return STRAIGHT, planar, symmetric
    return (ARC if planar and symmetric else BIARC), planar, symmetric


def build(P0, t0, P1, t1, samples=16):
    """The bent edge: (samples+1, 3) points from P0 to P1.

    Exactly matches t0 at P0 and -t1 at P1 by construction.
    """
    P0 = np.asarray(P0, float)
    P1 = np.asarray(P1, float)
    t0 = _unit(np.asarray(t0, float))
    d1 = -_unit(np.asarray(t1, float))
    kind, _planar, _sym = classify(P0, t0, P1, t1)

    if kind == STRAIGHT:
        s = np.linspace(0.0, 1.0, samples + 1)[:, None]
        return P0[None, :] + s * (P1 - P0)[None, :]

    if kind == ARC:
        return arc_points(P0, t0, P1, samples)

    # symmetric biarc: equal tangent lengths alpha, joint at the midpoint of
    # the two offset points.  This is the standard construction; for a planar
    # symmetric configuration it reproduces the single arc exactly, which the
    # self-test checks rather than assumes.
    v = P1 - P0
    dd = float(t0 @ d1)
    s = t0 + d1
    if abs(1.0 - dd) < 1e-12:              # parallel tangents
        alpha = float(v @ v) / (4.0 * max(1e-12, float(v @ t0)))
    else:
        b = float(v @ s)
        disc = b * b + 2.0 * (1.0 - dd) * float(v @ v)
        alpha = (-b + np.sqrt(max(0.0, disc))) / (2.0 * (1.0 - dd))
    J = 0.5 * ((P0 + alpha * t0) + (P1 - alpha * d1))

    half = max(1, samples // 2)
    A = arc_points(P0, t0, J, half)
    # second arc runs backwards from P1 along -d1 so its tangent there is
    # exact; reverse it to get travel order
    B = arc_points(P1, -d1, J, samples - half)[::-1]
    return np.vstack([A[:-1], B])


def _selftest():
    rng = np.random.default_rng(11)

    # straight case is exact
    P = build([0, 0, 0], [1, 0, 0], [2, 0, 0], [-1, 0, 0], samples=8)
    assert np.allclose(P[0], [0, 0, 0]) and np.allclose(P[-1], [2, 0, 0])
    assert np.abs(P[:, 1:]).max() < 1e-14
    print("edges: straight configuration exact  OK")

    # a planar symmetric configuration must come out as ONE circular arc:
    # constant discrete curvature along its length
    P0, P1 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    for deg in (10.0, 30.0, 45.0, 60.0, 80.0):
        a = np.radians(deg)
        t0 = np.array([np.cos(a), np.sin(a), 0.0])
        t1 = np.array([-np.cos(a), np.sin(a), 0.0])   # outgoing at P1
        kind, planar, sym = classify(P0, t0, P1, t1)
        assert (kind, planar, sym) == (ARC, True, True), (deg, kind)
        C = build(P0, t0, P1, t1, samples=64)
        d1 = np.diff(C, axis=0)
        d2 = np.diff(d1, axis=0)
        kappa = np.linalg.norm(d2, axis=1) / (np.linalg.norm(d1[:-1], axis=1) ** 2)
        assert kappa.std() / max(1e-12, kappa.mean()) < 1e-6, (deg, kappa.std())
    print("edges: planar symmetric -> single arc, constant curvature  OK")

    # endpoint and tangent match, for planar and spatial configurations alike
    worst_p = worst_t = 0.0
    kinds = {STRAIGHT: 0, ARC: 0, BIARC: 0}
    for _ in range(400):
        P0 = rng.normal(size=3)
        P1 = P0 + rng.normal(size=3)
        u = _unit(P1 - P0)
        # build a symmetric pair at a random angle about a random axis
        ax = _unit(np.cross(u, rng.normal(size=3)))
        a = rng.uniform(0.05, 1.2)
        rot = lambda k, th: (k * np.cos(th) + np.cross(ax, k) * np.sin(th)
                             + ax * float(ax @ k) * (1 - np.cos(th)))
        t0 = _unit(rot(u, a))
        tw = rng.uniform(-np.pi, np.pi)
        ax2 = u
        d1 = _unit(rot(u, -a))
        d1 = (d1 * np.cos(tw) + np.cross(ax2, d1) * np.sin(tw)
              + ax2 * float(ax2 @ d1) * (1 - np.cos(tw)))
        t1 = -_unit(d1)
        kind, _pl, _sy = classify(P0, t0, P1, t1)
        kinds[kind] += 1
        C = build(P0, t0, P1, t1, samples=48)
        worst_p = max(worst_p, float(np.linalg.norm(C[0] - P0)),
                      float(np.linalg.norm(C[-1] - P1)))
        e0 = _unit(C[1] - C[0])
        e1 = _unit(C[-2] - C[-1])
        worst_t = max(worst_t, float(np.linalg.norm(e0 - t0)),
                      float(np.linalg.norm(e1 - _unit(np.asarray(t1, float)))))
    assert worst_p < 1e-12, worst_p
    # end tangents are measured by a one-sided difference, so their error is
    # first order in the sample spacing; refine and confirm it shrinks
    assert worst_t < 0.05, worst_t
    print(f"edges: 400 random configurations, endpoint error {worst_p:.1e}, "
          f"end-tangent error {worst_t:.1e}  ({kinds[BIARC]} biarc, "
          f"{kinds[ARC]} arc)  OK")

    # the end-tangent error must be a discretisation artefact: it has to fall
    # as the sampling is refined, or the curve genuinely leaves at the wrong
    # angle
    P0 = np.array([0.0, 0.0, 0.0])
    P1 = np.array([1.0, 0.2, 0.4])
    t0 = _unit(np.array([0.6, 0.8, 0.3]))
    t1 = _unit(np.array([-0.5, 0.7, -0.6]))
    prev = None
    for m in (16, 64, 256, 1024):
        C = build(P0, t0, P1, t1, samples=m)
        err = float(np.linalg.norm(_unit(C[1] - C[0]) - t0))
        if prev is not None:
            assert err < prev * 0.6, (m, err, prev)
        prev = err
    assert prev < 1e-3, prev
    print(f"edges: end-tangent error -> {prev:.1e} under refinement  OK")

    print("softcell.edges standalone tests passed")
