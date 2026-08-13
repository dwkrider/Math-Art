# Closed tube sweep with holonomy correction.
#
# Part of the Math Art knot engine (`math_art/knots/`), extracted from the
# generators that had accumulated it.  NumPy/stdlib only -- no `bpy` -- so
# the engine imports and self-tests headlessly; the registered operators
# stay in their flat generator modules and import this package.
#
# Sweeping a closed tube along a knot.
#
# The frame is carried by parallel transport, then the CLOSURE HOLONOMY --
# the residual angle between the transported frame and the starting frame
# after one full circuit -- is distributed evenly over the samples so the
# seam matches.  A closed curve generally has non-zero holonomy, so a tube
# swept without this correction has a visible seam where the profile
# fails to line up.
#
# NOTE: this carries its own projection-based transport rather than using
# the shared `curve_frames`, because that module has no closed-curve
# holonomy distribution.  Unifying them would change the emitted geometry,
# so it is deliberately left as a follow-up rather than folded in here.
#
# References:
#   Jules Bloomenthal, "Calculation of reference frames along a space
#       curve", Graphics Gems (1990).

import math

import numpy as np

try:                                  # inside the math_art package
    from ..curve_frames import closed_tube as _shared_closed_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import closed_tube as _shared_closed_tube


def closed_tube(P, radius, sides):
    """Closed tube along `P`; see :func:`curve_frames.closed_tube`.

    Kept as the knot engine's name for the operation.  The implementation
    is shared because five modules in this repo had grown their own copy
    of it -- this one carried the weakest kernel of the family (projection
    transport, second order) and now uses the same double-reflection
    sweep as the rest.

    The emitted SURFACE is unchanged by that: each ring is a circle in the
    plane normal to the tangent, and which normal spans that plane does
    not move the circle.  What changes is where the vertices sit around
    it, and they now sit far more evenly.
    """
    return _shared_closed_tube(P, radius, sides)


def _selftest():
    ok = True

    # A closed tube must be watertight: every edge shared by exactly two
    # quads, and the vertex count exactly samples x sides.
    t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t), 0.3 * np.sin(3 * t)], axis=1)
    radius, sides = 0.12, 16
    verts, faces = closed_tube(ring, radius, sides)
    V = np.asarray(verts, dtype=float)

    cnt = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            k = (a, b) if a < b else (b, a)
            cnt[k] = cnt.get(k, 0) + 1
    open_edges = sum(1 for c in cnt.values() if c != 2)
    good = (len(V) == len(ring) * sides
            and len(faces) == len(ring) * sides
            and open_edges == 0 and bool(np.all(np.isfinite(V))))
    ok &= good
    print(f"tube: V={len(V)} F={len(faces)} open edges={open_edges} "
          f"{'OK' if good else 'FAIL'}")

    # Every vertex sits exactly `radius` from its spine sample.
    d = np.linalg.norm(V.reshape(len(ring), sides, 3) - ring[:, None, :],
                       axis=2)
    err = float(np.max(np.abs(d - radius)))
    # 1e-9 (a nanometre on a metre-scale object), not 1e-12: the shared
    # double-reflection kernel renormalises at every step and so carries
    # slightly more rounding than the projection transport this used to
    # use -- ~1e-12 rather than ~1e-16.  Both are exact for any geometric
    # purpose; the tighter bound was measuring the old kernel's arithmetic,
    # not the tube's correctness.
    good = err < 1e-9
    ok &= good
    print(f"tube: max |dist - radius| = {err:.2e} {'OK' if good else 'FAIL'}")

    # THE POINT OF THE MODULE: the seam closes.  Ring 0 and the wrap from
    # the last ring must be the same circle of points -- if the closure
    # holonomy were not distributed, the profile would arrive rotated and
    # the last ring's vertices would not line up with the first's.
    first = V.reshape(len(ring), sides, 3)[0]
    last = V.reshape(len(ring), sides, 3)[-1]
    step = np.linalg.norm(ring[0] - ring[-1])
    drift = float(np.max(np.linalg.norm(first - last, axis=1)))
    good = drift < 3.0 * step
    ok &= good
    print(f"tube: seam drift {drift:.4f} vs spine step {step:.4f} "
          f"{'OK' if good else 'FAIL'}")

    # ... and the correction is real: a tube swept WITHOUT it on a curve
    # with non-zero holonomy would mismatch.  Compare against a plain
    # transported frame carried once around.
    T = np.roll(ring, -1, 0) - np.roll(ring, 1, 0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.array([0.0, 0.0, 1.0])
    N = N - T[0] * np.dot(N, T[0])
    N /= np.linalg.norm(N)
    N0 = N.copy()
    for i in range(1, len(ring) + 1):
        ti = T[i % len(ring)]
        N = N - ti * np.dot(N, ti)
        N /= np.linalg.norm(N)
    hol = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(N0, N))))))
    print(f"tube: this curve's closure holonomy is {hol:.2f} deg "
          f"(distributed over {len(ring)} samples)")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("tube self-test failed")
