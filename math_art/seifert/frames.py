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

    Each step is the composition of two reflections, hence a rotation, so
    orthonormality is preserved exactly in exact arithmetic.
    """
    points = np.asarray(points, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    normals = np.empty_like(points)
    normals[0] = unit(first_normal - (first_normal @ tangents[0]) * tangents[0])
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
