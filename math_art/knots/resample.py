# Arclength resampling of closed polylines.
#
# Part of the Math Art knot engine (`math_art/knots/`), extracted from the
# generators that had accumulated it.  NumPy/stdlib only -- no `bpy` -- so
# the engine imports and self-tests headlessly; the registered operators
# stay in their flat generator modules and import this package.
#
# Uniform-arclength resampling of closed polylines, by cumulative chord
# length.  Every knot in this package is relaxed as a polyline, and each
# relaxation step needs its samples redistributed or the rope bunches.

import numpy as np


def resample_closed(pts, m):
    """Resample a closed polyline to m evenly spaced points."""
    P = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    want = np.linspace(0.0, total, m, endpoint=False)
    out = np.empty((m, 3))
    j = 0
    for i, s in enumerate(want):
        while cum[j + 1] < s:
            j += 1
        t = (s - cum[j]) / (seg[j] or 1.0)
        out[i] = P[j] + t * (P[(j + 1) % len(P)] - P[j])
    return out


def resample_loops(loops, samples):
    """Resample each closed loop, distributing points by arc length
    so an average component gets `samples` points."""
    lens = []
    for lp in loops:
        Q = np.asarray(lp, dtype=float)
        lens.append(np.linalg.norm(np.roll(Q, -1, 0) - Q,
                                   axis=1).sum())
    avg = sum(lens) / len(lens)
    return [resample_closed(lp, max(32, int(round(samples
                                                     * ln / avg))))
            for lp, ln in zip(loops, lens)]


def _selftest():
    ok = True
    t = np.linspace(0.0, 2.0 * np.pi, 57, endpoint=False)
    ring = np.stack([np.cos(t), 1.3 * np.sin(t), 0.2 * np.sin(3 * t)], axis=1)

    for m in (16, 40, 128, 500):
        P = resample_closed(ring, m)
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        spread = float(np.ptp(seg)) / float(np.mean(seg))
        g = len(P) == m and bool(np.all(np.isfinite(P))) and spread < 0.05
        ok &= g
        print(f"resample: {len(ring)} -> {m:4d} step spread={spread:.4f} "
              f"{'OK' if g else 'FAIL'}")

    # Total length is preserved to the chord-vs-arc error, which SHRINKS as
    # the sample count rises -- the check that the redistribution is by
    # arclength and not by index.
    def length(P):
        return float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum())

    l0 = length(ring)
    errs = [abs(length(resample_closed(ring, m)) - l0) / l0
            for m in (64, 256, 1024)]
    g = errs[0] > errs[-1] and errs[-1] < 2e-3
    ok &= g
    print(f"resample: length error {errs[0]:.2e} -> {errs[-1]:.2e} as samples "
          f"rise {'OK' if g else 'FAIL'}")

    # resample_loops distributes points by arclength: a long loop gets more
    # points than a short one, and every loop gets at least the floor of 32.
    small = 0.25 * ring
    loops = resample_loops([ring, small], 100)
    g = (len(loops) == 2 and len(loops[0]) > len(loops[1])
         and min(len(lp) for lp in loops) >= 32)
    ok &= g
    print(f"resample: loops by arclength -> {[len(lp) for lp in loops]} "
          f"{'OK' if g else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("resample self-test failed")
