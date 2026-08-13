# Plane isometries as homogeneous matrices.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`), split out of
# the former single-file `pattern_common.py`.  Python + numpy only -- no
# `bpy` -- so the engine imports and self-tests headlessly; the registered
# operators stay in their flat generator modules and import this package.
#
# Plane isometries as 3x3 homogeneous matrices acting on [x, y, 1]:
# translation, rotation about a point, reflection in a line, and glide.
# Composition is matrix multiplication, which is what lets a group be
# stored as a list of coset representatives.

from math import cos, sin, pi, hypot, gcd            # noqa: F401
import numpy as np


# 2D isometries as 3x3 homogeneous matrices acting on [x, y, 1]
# --------------------------------------------------------------------

def I():
    return np.eye(3)


def T(tx, ty):
    """Translation."""
    return np.array([[1.0, 0.0, tx],
                     [0.0, 1.0, ty],
                     [0.0, 0.0, 1.0]])


def Rot(theta, cx=0.0, cy=0.0):
    """Rotation by theta about (cx, cy)."""
    c, s = cos(theta), sin(theta)
    m = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return T(cx, cy) @ m @ T(-cx, -cy)


def Mir(angle, cx=0.0, cy=0.0):
    """Reflection across the line through (cx, cy) at `angle`."""
    c, s = cos(2 * angle), sin(2 * angle)
    m = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, 1.0]])
    return T(cx, cy) @ m @ T(-cx, -cy)


def Glide(angle, dist, cx=0.0, cy=0.0):
    """Reflect across the line at `angle` through (cx, cy), then
    translate `dist` along that line."""
    return T(dist * cos(angle), dist * sin(angle)) @ Mir(angle, cx, cy)


def apply(M, pts):
    """Apply a 3x3 homogeneous transform to an (N, 2) point array."""
    pts = np.asarray(pts, dtype=float)
    hom = np.column_stack([pts, np.ones(len(pts))])
    return (hom @ M.T)[:, :2]


def _selftest():
    ok = True
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.8], [-0.5, 0.2]])

    # identity, and translation composing additively
    good = (np.allclose(apply(I(), pts), pts)
            and np.allclose(apply(T(0.3, -0.2) @ T(0.1, 0.5), pts),
                            apply(T(0.4, 0.3), pts)))
    ok &= good
    print(f"isometry: identity and translation composition "
          f"{'OK' if good else 'FAIL'}")

    # every isometry preserves pairwise distances -- the defining property
    def dists(P):
        return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    d0 = dists(pts)
    bad = []
    for name, M in (('Rot', Rot(0.7, 0.2, -0.1)), ('Mir', Mir(0.4, 0.1, 0.2)),
                    ('Glide', Glide(0.9, 0.35, 0.0, 0.0)),
                    ('T', T(1.3, -2.0))):
        if not np.allclose(dists(apply(M, pts)), d0, atol=1e-12):
            bad.append(name)
    good = not bad
    ok &= good
    print(f"isometry: all four preserve distances "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # orientation: rotation and translation preserve it, reflection and
    # glide reverse it (determinant of the linear part)
    import numpy.linalg as la
    def det2(M):
        return float(la.det(M[:2, :2]))
    good = (det2(Rot(0.7)) > 0 and det2(T(1, 2)) > 0
            and det2(Mir(0.4)) < 0 and det2(Glide(0.9, 0.3)) < 0)
    ok &= good
    print(f"isometry: rotations preserve orientation, reflections reverse it "
          f"{'OK' if good else 'FAIL'}")

    # a reflection is an involution; a rotation by 2pi/n has order n
    good = np.allclose(apply(Mir(0.4, 0.1, 0.2) @ Mir(0.4, 0.1, 0.2), pts),
                       pts, atol=1e-12)
    M = I()
    for _ in range(6):
        M = M @ Rot(2 * pi / 6, 0.2, 0.3)
    good = good and np.allclose(apply(M, pts), pts, atol=1e-10)
    ok &= good
    print(f"isometry: reflection is an involution, 6-fold rotation has order 6 "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("isometry self-test failed")
