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


def closed_tube(P, radius, sides):
    """Closed tube with parallel-transport frames; the frame
    holonomy is distributed so the seam matches."""
    m = len(P)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.empty_like(P)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, T[0])) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    N[0] = np.cross(T[0], ref)
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, m):
        v = N[i - 1] - T[i] * np.dot(N[i - 1], T[i])
        N[i] = v / (np.linalg.norm(v) or 1.0)
    # transport once more to measure the closure twist
    v = N[m - 1] - T[0] * np.dot(N[m - 1], T[0])
    v /= (np.linalg.norm(v) or 1.0)
    B0 = np.cross(T[0], N[0])
    ang = math.atan2(np.dot(v, B0), np.dot(v, N[0]))
    verts = []
    faces = []
    for i in range(m):
        corr = -ang * i / m
        B = np.cross(T[i], N[i])
        for j in range(sides):
            a = 2 * math.pi * j / sides + corr
            p = (P[i] + radius
                 * (math.cos(a) * N[i] + math.sin(a) * B))
            verts.append(tuple(p))
    for i in range(m):
        i2 = (i + 1) % m
        for j in range(sides):
            j2 = (j + 1) % sides
            faces.append([i * sides + j, i * sides + j2,
                          i2 * sides + j2, i2 * sides + j])
    return verts, faces


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
    good = err < 1e-12
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
