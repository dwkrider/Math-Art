# The 2 m cube convention.
#
# Part of the Math Art polyhedron engine (`math_art/polyhedra/`).  Python
# and numpy only -- no `bpy`.
#
# Every generator in this extension emits geometry CENTRED ON THE ORIGIN
# and scaled so its largest extent is 2.0 units -- a 2 m cube.  That is
# what makes objects from different generators comparable in the viewport
# and printable at a predictable size, and it is why a knot, a minimal
# surface and a polyhedron all arrive the same size.
#
# The rule is easy to state and easy to forget: several polyhedron
# generators emitted their raw mathematical coordinates instead, so a
# regular solid arrived at 1.15 units across, a Waterman at 1.79, and a
# uniform polyhedron was not centred at all.  Keeping the convention in
# one function, next to the seeds it applies to, is the point of this
# module.
#
# Note this normalises the BOUNDING BOX, not the circumradius: a shape is
# "as large as fits" in the cube, which for a sphere-like solid is nearly
# the same thing and for a flat or elongated one is visibly better.

import numpy as np


def fit_cube(V, span=2.0):
    """Centre `V` on the origin and scale so its largest extent is `span`.

    Returns a new list of vertices; the input is not modified.  A
    degenerate (zero-extent) input is centred but not scaled, rather than
    raising or producing infinities.
    """
    A = np.asarray(V, dtype=float)
    if A.size == 0:
        return [list(v) for v in A]
    lo, hi = A.min(axis=0), A.max(axis=0)
    A = A - 0.5 * (lo + hi)
    ext = float(np.max(hi - lo))
    if ext > 1e-12:
        A = A * (span / ext)
    return [[float(c) for c in v] for v in A]


def cube_fit_error(V, span=2.0):
    """(centre offset, extent) for checking the convention holds."""
    A = np.asarray(V, dtype=float)
    if A.size == 0:
        return 0.0, 0.0
    lo, hi = A.min(axis=0), A.max(axis=0)
    return (float(np.max(np.abs(0.5 * (lo + hi)))), float(np.max(hi - lo)))


def _selftest():
    ok = True
    rng = np.random.default_rng(7)

    # the convention itself, on awkward inputs
    cases = {
        'offset cube': np.array([[3., 3., 3.], [7., 5., 4.], [5., 4., 3.5]]),
        'flat sheet': np.array([[0., 0., 0.], [4., 0., 0.], [4., 3., 0.]]),
        'thin needle': np.array([[0., 0., 0.], [0., 0., 9.], [0., .001, 4.]]),
        'random cloud': rng.normal(size=(40, 3)) * 7.0 + 11.0,
    }
    bad = []
    for name, P in cases.items():
        c, e = cube_fit_error(fit_cube(P))
        if c > 1e-12 or abs(e - 2.0) > 1e-12:
            bad.append(f"{name}:|c|={c:.1e},ext={e:.6f}")
    good = not bad
    ok &= good
    print(f"fit: centred and exactly 2.0 across on {len(cases)} shapes "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # it must not distort: relative distances are preserved exactly
    P = cases['random cloud']
    Q = np.asarray(fit_cube(P), float)
    dp = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    dq = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
    s = float(dq[dp > 0].mean() / dp[dp > 0].mean())
    good = float(np.max(np.abs(dq - dp * s))) < 1e-9
    ok &= good
    print(f"fit: a pure similarity -- all distances scale by one factor "
          f"{'OK' if good else 'FAIL'}")

    # idempotent, and `span` respected
    good = (abs(cube_fit_error(fit_cube(fit_cube(P)))[1] - 2.0) < 1e-12
            and abs(cube_fit_error(fit_cube(P, span=5.0), 5.0)[1] - 5.0) < 1e-12)
    ok &= good
    print(f"fit: idempotent, and span is honoured {'OK' if good else 'FAIL'}")

    # a degenerate input must not blow up
    z = fit_cube(np.zeros((4, 3)))
    good = np.all(np.isfinite(np.asarray(z, float)))
    ok &= good
    print(f"fit: a zero-extent input stays finite {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("fit self-test failed")
