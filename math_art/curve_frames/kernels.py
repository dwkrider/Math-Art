# Rotation-minimizing transport kernels.
#
# Part of `math_art/curve_frames/` -- the moving-frame engine.  NumPy only,
# no `bpy`, so it imports and self-tests headlessly.
#
# The two rotation-minimizing kernels.  Both carry a normal along a curve
# with the least possible rotation about the tangent; they differ in how
# the tangents arrive and in accuracy.
#
#   _transport         axis-angle between consecutive tangents, derived
#                      from the sampled points.  Second order.
#   transport_normals  DOUBLE REFLECTION.  Takes EXACT tangents and
#                      reflects across the chord, so it accounts for the
#                      curve between samples: fourth order.
#
# References:
#   Jules Bloomenthal, "Calculation of reference frames along a space
#     curve", Graphics Gems (1990) -- the axis-angle construction.
#   Wenping Wang, Bert Juttler, Dayue Zheng and Yang Liu, "Computation of
#     rotation minimizing frames", ACM TOG 27(1), 2008 -- double
#     reflection.

import math

import numpy as np


def transport_normals(points, tangents, first_normal):
    """Propagate `first_normal` along a curve by DOUBLE REFLECTION.

    The alternative kernel to `frames()`, for callers that already hold
    exact tangents (an analytic curve rather than a sampled polyline).
    Each step is a composition of two reflections, hence a rotation, so
    orthonormality is preserved exactly in exact arithmetic; taking the
    first reflection across the chord makes it fourth-order accurate.

    Returns the array of normals only -- cross with the tangents for the
    third axis.  Wang, Juttler, Zheng and Liu, ACM TOG 27(1), 2008.
    """
    points = np.asarray(points, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    normals = np.empty_like(points)
    normals[0] = _unit(first_normal
                       - (first_normal @ tangents[0]) * tangents[0])
    for i in range(len(points) - 1):
        v1 = points[i + 1] - points[i]
        c1 = v1 @ v1
        if c1 < 1e-18:                       # duplicated sample
            normals[i + 1] = normals[i]
            continue
        reflected_normal = normals[i] - (2 / c1) * (v1 @ normals[i]) * v1
        reflected_tangent = tangents[i] - (2 / c1) * (v1 @ tangents[i]) * v1
        v2 = tangents[i + 1] - reflected_tangent
        c2 = v2 @ v2
        normals[i + 1] = (
            reflected_normal
            if c2 < 1e-18
            else reflected_normal - (2 / c2) * (v2 @ reflected_normal) * v2
        )
    return normals


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def _transport(h0, h1, l0):
    """Carry `l0` from tangent `h0` to `h1` by the minimal rotation.

    This is the whole of parallel transport: rotate about h0 x h1 by the
    angle between them.  When the tangents agree (a straight section) the
    rotation is the identity, which is exactly why this is defined where
    the Frenet frame is not.
    """
    axis = np.cross(h0, h1)
    m = float(np.linalg.norm(axis))
    if m < 1e-12:
        # parallel (straight) or antiparallel (a cusp)
        if float(np.dot(h0, h1)) > 0.0:
            return _ortho(l0, h1)
        return _ortho(-l0, h1)
    axis = axis / m
    ang = math.atan2(m, float(np.dot(h0, h1)))
    c, s = math.cos(ang), math.sin(ang)
    rot = (l0 * c + np.cross(axis, l0) * s +
           axis * float(np.dot(axis, l0)) * (1.0 - c))
    return _ortho(rot, h1)


def _ortho(v, h):
    """Re-orthogonalise `v` against `h` and normalise.  Called every
    step so accumulated float drift never lets the frame shear."""
    v = v - h * float(np.dot(v, h))
    m = float(np.linalg.norm(v))
    if m < 1e-12:
        alt = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(alt, h))) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        v = np.cross(alt, h)
        m = float(np.linalg.norm(v))
    return v / m


def _selftest():
    ok = True
    # a cubic Bezier, whose tangents we can supply EXACTLY -- which is the
    # case double reflection is for
    t = np.linspace(0.0, 1.0, 200)[:, None]
    p0, p1, p2, p3 = (np.array(p, float) for p in
                      ((0, 0, 0), (1, 0, 0.4), (2, 1, -0.4), (3, 0, 0)))
    s = 1.0 - t
    P = s**3 * p0 + 3 * s**2 * t * p1 + 3 * s * t**2 * p2 + t**3 * p3
    D = 3 * s**2 * (p1 - p0) + 6 * s * t * (p2 - p1) + 3 * t**2 * (p3 - p2)
    T = D / np.linalg.norm(D, axis=1)[:, None]

    N = transport_normals(P, T, np.array([0.0, 0.0, 1.0]))
    perp = float(np.max(np.abs((N * T).sum(axis=1))))
    unitness = float(np.max(np.abs(np.linalg.norm(N, axis=1) - 1.0)))
    good = perp < 1e-9 and unitness < 1e-9
    ok &= good
    print(f"kernels: double reflection stays orthonormal "
          f"(perp {perp:.1e}) {'OK' if good else 'FAIL'}")

    # IT REALLY IS THE FOURTH-ORDER KERNEL.  Measured against the
    # second-order projection transport the gap is NOT zero -- it is the
    # difference between the methods -- and it must shrink as the curve is
    # sampled more finely.  A kernel that had silently degraded to
    # projection would report ~0 here, which orthonormality cannot detect.
    def gap(n):
        tt = np.linspace(0.0, 1.0, n)[:, None]
        ss = 1.0 - tt
        Q = ss**3 * p0 + 3 * ss**2 * tt * p1 + 3 * ss * tt**2 * p2 + tt**3 * p3
        Dq = (3 * ss**2 * (p1 - p0) + 6 * ss * tt * (p2 - p1)
              + 3 * tt**2 * (p3 - p2))
        Tq = Dq / np.linalg.norm(Dq, axis=1)[:, None]
        M = transport_normals(Q, Tq, np.array([0.0, 0.0, 1.0]))
        tot = 0.0
        for i in range(1, len(Q)):
            prev = M[i - 1] - Tq[i] * float(np.dot(M[i - 1], Tq[i]))
            prev /= max(float(np.linalg.norm(prev)), 1e-15)
            c = max(-1.0, min(1.0, float(np.dot(prev, M[i]))))
            tot += abs(math.acos(c))
        return tot
    g1, g2, g3 = gap(100), gap(200), gap(400)
    good = g1 > g2 > g3 and g1 < 0.05
    ok &= good
    print(f"kernels: gap vs projection transport {g1:.2e} -> {g2:.2e} -> "
          f"{g3:.2e} as samples double {'OK' if good else 'FAIL'}")

    # the axis-angle kernel: on a straight run it must not rotate at all
    h0 = np.array([0.0, 0.0, 1.0])
    l0 = np.array([1.0, 0.0, 0.0])
    good = np.allclose(_transport(h0, h0, l0), l0, atol=1e-12)
    ok &= good
    print(f"kernels: axis-angle transport is the identity on a straight run "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("kernels self-test failed")
