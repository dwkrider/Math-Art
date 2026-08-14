# Unit tangents of a sampled curve.
#
# Part of `math_art/curve_frames/` -- the moving-frame engine.  NumPy only,
# no `bpy`, so it imports and self-tests headlessly.
#
# Unit tangents of a sampled curve, open or closed.  Never a zero vector:
# a repeated point inherits its neighbour's direction rather than
# producing a NaN downstream.

import numpy as np


def tangents(points, closed=False):
    """Unit tangents at each sample of a polyline.

    Central differences inside, one-sided at the ends -- and never a
    zero vector: a repeated point inherits its neighbour's direction
    rather than producing a NaN downstream.
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n < 2:
        return np.tile(np.array([0.0, 0.0, 1.0]), (max(n, 1), 1))
    T = np.zeros_like(P)
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T[1:-1] = P[2:] - P[:-2]
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    norms = np.linalg.norm(T, axis=1)
    bad = norms < 1e-12
    if bad.any():
        # inherit from the previous good tangent, else +Z
        last = np.array([0.0, 0.0, 1.0])
        for i in range(n):
            if bad[i]:
                T[i] = last
                norms[i] = 1.0
            else:
                last = T[i] / norms[i]
    return T / norms[:, None]


def closed_tangents(points):
    """Unit tangents of a CLOSED polyline, by central differences that
    wrap.  The convention every closed sweep in this repo already uses."""
    P = np.asarray(points, dtype=float)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)


def _selftest():
    ok = True
    t = np.linspace(0, 4 * np.pi, 200)
    helix = np.stack([np.cos(t), np.sin(t), 0.15 * t], axis=1)

    T = tangents(helix)
    good = (len(T) == len(helix)
            and np.allclose(np.linalg.norm(T, axis=1), 1.0, atol=1e-9))
    ok &= good
    print(f"tangents: unit length on an open curve {'OK' if good else 'FAIL'}")

    # a closed curve wraps: the first and last tangents are computed from
    # neighbours across the seam, not one-sided
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)
    Tc = closed_tangents(ring)
    To = tangents(ring, closed=True)
    good = (np.allclose(np.linalg.norm(Tc, axis=1), 1.0, atol=1e-9)
            and np.allclose(np.abs((Tc * To).sum(axis=1)), 1.0, atol=1e-6))
    ok &= good
    print(f"tangents: closed wrap agrees with closed=True "
          f"{'OK' if good else 'FAIL'}")

    # on a circle the tangent is perpendicular to the radius everywhere
    perp = float(np.max(np.abs((Tc * ring).sum(axis=1))))
    good = perp < 1e-9
    ok &= good
    print(f"tangents: perpendicular to the radius on a circle "
          f"({perp:.1e}) {'OK' if good else 'FAIL'}")

    # a repeated point must inherit a direction, not produce NaN
    dup = np.array([[0., 0., 0.], [1., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    Td = tangents(dup)
    good = bool(np.all(np.isfinite(Td)))
    ok &= good
    print(f"tangents: a duplicated sample stays finite "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("tangents self-test failed")
