"""Curves and moving frames for the band geometry.

The paper frames its tubes with the Frenet frame.  That frame is undefined
wherever the curvature vanishes and flips sign through an inflection, which for
a cubic Bezier with nearly collinear control points is a real hazard.  This
module uses the double-reflection rotation-minimising frame instead: same cost
per step, fourth-order accurate, and stable for nearly collinear tangents, so it
needs no special cases.

  W. Wang, B. Juttler, D. Zheng and Y. Liu, "Computation of Rotation Minimizing
  Frames", ACM TOG 27(1), 2008.

The twist logic is the paper's: transport a frame along the band, then add
whatever rotation is needed to land on the requested number of half-turns while
still meeting the fixed orientation at the far end.
"""

from __future__ import annotations

import numpy as np

try:                                  # inside the math_art package
    from ..curve_frames import transport_normals as _transport_normals
except ImportError:                   # flat import (headless test runner)
    from curve_frames import transport_normals as _transport_normals

__all__ = ["unit", "bezier", "rotation_minimising_frames", "twist_to_close"]


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def bezier(p0, p1, p2, p3, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Cubic Bezier positions and unit tangents at ``samples`` parameters."""
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    t = np.linspace(0.0, 1.0, samples)[:, None]
    s = 1.0 - t
    points = s**3 * p0 + 3 * s**2 * t * p1 + 3 * s * t**2 * p2 + t**3 * p3
    derivative = 3 * s**2 * (p1 - p0) + 6 * s * t * (p2 - p1) + 3 * t**2 * (p3 - p2)
    return points, unit(derivative)


def rotation_minimising_frames(
    points: np.ndarray, tangents: np.ndarray, first_normal: np.ndarray
) -> np.ndarray:
    """Propagate ``first_normal`` along the curve by double reflection.

    The kernel is shared: it lives in the repo-wide :mod:`curve_frames`,
    because the band geometry here is not the only thing that needs a
    rotation-minimising frame.  This wrapper is the name the rest of the
    package (and the paper's vocabulary) uses for it.
    """
    return _transport_normals(points, tangents, first_normal)


def twist_to_close(
    transported: np.ndarray,
    target: np.ndarray,
    tangent: np.ndarray,
    half_turns: int,
) -> float:
    """Total rotation to apply along a band.

    ``transported`` is the frame's width vector after being carried to the far
    end, ``target`` the width direction the far end must actually have.  The two
    admissible closures differ by pi, so the parity of ``half_turns`` selects
    one; full turns are then added to reach the requested twist.  This is the
    paper's phi(t) with its integer T, with the frame swapped for an RMF.
    """
    reference = unit(transported - (transported @ tangent) * tangent)
    binormal = np.cross(tangent, reference)
    goal = unit(target - (target @ tangent) * tangent)
    residual = np.arctan2(binormal @ goal, reference @ goal)   # in (-pi, pi]
    if half_turns % 2:                       # odd: land on -goal instead
        residual += np.pi if residual <= 0 else -np.pi
    wanted = half_turns * np.pi
    return residual + round((wanted - residual) / (2 * np.pi)) * 2 * np.pi


def _selftest():
    ok = True

    # The Bezier evaluator: endpoints interpolated, tangents unit, and the
    # tangent direction agreeing with a finite difference of the positions.
    p0, p1, p2, p3 = (0, 0, 0), (1, 0, 0.4), (2, 1, -0.4), (3, 0, 0)
    pts, tans = bezier(p0, p1, p2, p3, 200)
    fd = pts[1:] - pts[:-1]
    fd = fd / np.linalg.norm(fd, axis=1)[:, None]
    good = (np.allclose(pts[0], p0) and np.allclose(pts[-1], p3)
            and np.allclose(np.linalg.norm(tans, axis=1), 1.0, atol=1e-12)
            and float(np.max(np.linalg.norm(fd - tans[:-1], axis=1))) < 0.02)
    ok &= good
    print(f"seifert.frames: bezier endpoints + unit tangents "
          f"{'OK' if good else 'FAIL'}")

    # The RMF kernel (shared with curve_frames) must return unit normals
    # perpendicular to the tangent everywhere.
    n0 = np.array([0.0, 0.0, 1.0])
    N = rotation_minimising_frames(pts, tans, n0)
    perp = float(np.max(np.abs((N * tans).sum(axis=1))))
    unitness = float(np.max(np.abs(np.linalg.norm(N, axis=1) - 1.0)))
    good = perp < 1e-9 and unitness < 1e-9
    ok &= good
    print(f"seifert.frames: RMF perp={perp:.2e} |N|-1={unitness:.2e} "
          f"{'OK' if good else 'FAIL'}")

    # THE KERNEL IS THE FOURTH-ORDER ONE.  Measuring the frame against a
    # naive projection transport does NOT give zero -- it gives the gap
    # between the two methods, because double reflection reflects across
    # the chord and so accounts for the curve between samples.  That gap
    # is the thing worth asserting: it must be small, and must shrink
    # rapidly as the curve is sampled more finely.  (A kernel that had
    # silently degraded to projection transport would report ~0 here, so
    # this catches the regression that a perpendicularity check cannot.)
    def projection_gap(n):
        P, T = bezier(p0, p1, p2, p3, n)
        M = rotation_minimising_frames(P, T, n0)
        tot = 0.0
        for i in range(1, len(P)):
            prev = M[i - 1] - T[i] * float(np.dot(M[i - 1], T[i]))
            prev /= max(np.linalg.norm(prev), 1e-15)
            c = max(-1.0, min(1.0, float(np.dot(prev, M[i]))))
            tot += abs(np.arccos(c))
        return tot

    g1, g2, g3 = projection_gap(100), projection_gap(200), projection_gap(400)
    good = g1 > g2 > g3 and g1 < 0.05
    ok &= good
    print(f"seifert.frames: double-reflection vs projection gap "
          f"{g1:.2e} -> {g2:.2e} -> {g3:.2e} as samples double "
          f"{'OK' if good else 'FAIL'}")

    # twist_to_close must land on the requested number of half-turns, with
    # the parity choosing which of the two admissible closures is taken.
    tangent = np.array([0.0, 0.0, 1.0])
    transported = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])
    bad = []
    for ht in (0, 1, 2, 3, 4, -1, -2):
        ang = twist_to_close(transported, target, tangent, ht)
        # rotating `transported` by `ang` must reach +-target, matching parity
        c, s = np.cos(ang), np.sin(ang)
        got = transported * c + np.cross(tangent, transported) * s
        aligned = float(np.dot(got, target))
        want = -1.0 if ht % 2 else 1.0
        if abs(aligned - want) > 1e-9:
            bad.append(f"{ht}:{aligned:+.3f}")
    good = not bad
    ok &= good
    print(f"seifert.frames: twist_to_close parity over 7 half-turn counts "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("seifert.frames self-test failed")
