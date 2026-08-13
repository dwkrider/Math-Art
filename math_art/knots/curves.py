# Closed-form knot and link curves.
#
# Part of the Math Art knot engine (`math_art/knots/`), extracted from the
# generators that had accumulated it.  NumPy/stdlib only -- no `bpy` -- so
# the engine imports and self-tests headlessly; the registered operators
# stay in their flat generator modules and import this package.
#
# Closed-form space curves for knots and links.
#
# The (p, q) torus link has d = gcd(p, q) components, each a reduced
# (p/d, q/d) torus knot.  On the flat torus the components are parallel
# lines of slope q'/p', spaced 2 pi / p apart in the tube angle -- NOT
# 2 pi / d, which would re-trace the same curve whenever p/d > 1.

import math

import numpy as np
from math import gcd


def torus_link_components(p, q, samples, R=1.25, r=0.55):
    """(p,q) torus link: gcd(p,q) components, each a (p/g, q/g)-type
    curve, phase-offset by 2*pi*k/p in the tube angle (the distinct
    cosets k mod g give the g disjoint components)."""
    g = gcd(p, q)
    s = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    comps = []
    for k in range(g):
        th = (p // g) * s
        ph = (q // g) * s + 2 * math.pi * k / p
        rad = R + r * np.cos(ph)
        comps.append(np.stack([rad * np.cos(th), rad * np.sin(th),
                               r * np.sin(ph)], axis=1))
    return comps


def _selftest():
    ok = True

    # Component count is gcd(p, q) -- the whole point of the construction.
    cases = [(2, 3, 1), (3, 5, 1), (2, 4, 2), (4, 6, 2), (6, 9, 3),
             (3, 3, 3), (2, 7, 1)]
    bad = []
    for p, q, want in cases:
        comps = torus_link_components(p, q, 96)
        if len(comps) != want:
            bad.append(f"({p},{q}):{len(comps)}!={want}")
    good = not bad
    ok &= good
    print(f"curves: gcd component count over {len(cases)} cases "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # Every component is a closed curve lying ON the torus: the distance
    # from the tube's core circle of radius R must equal r everywhere.
    R, r = 1.25, 0.55
    for p, q in ((2, 3), (4, 6)):
        for P in torus_link_components(p, q, 240, R=R, r=r):
            rho = np.hypot(P[:, 0], P[:, 1])
            d = np.hypot(rho - R, P[:, 2])
            g = (bool(np.all(np.isfinite(P)))
                 and float(np.max(np.abs(d - r))) < 1e-9)
            ok &= g
            if not g:
                print(f"curves: ({p},{q}) off-torus by "
                      f"{float(np.max(np.abs(d - r))):.2e} FAIL")
    print(f"curves: components lie on the torus {'OK' if ok else 'FAIL'}")

    # Distinct components must be disjoint -- if the 2 pi k / p phase
    # spacing were wrong (2 pi / d is the classic error) they would
    # coincide instead.
    comps = torus_link_components(4, 6, 240)
    A, B = comps[0], comps[1]
    gap = float(np.min(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)))
    good = gap > 1e-3
    ok &= good
    print(f"curves: (4,6) components disjoint, min gap={gap:.4f} "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("curves self-test failed")
