# Planar polygon predicates shared by the tiling generators.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operators stay in their flat generator
# modules and import this package.
#
# These are small, but they were written five times: `aperiodic`,
# `fractal_reptile`, `fractal_tiling`, `reptile` and `voderberg` each
# carried their own `_signed_area` and `_ensure_ccw`.  Winding is the kind
# of thing every tiling backend needs and nobody thinks to share.
#
# References:
#   The shoelace formula for the area of a simple polygon is usually
#   credited to A. L. F. Meister (1769) and to C. F. Gauss; the "surveyor's
#   formula" name is later.  For a simple polygon its sign is positive
#   exactly when the vertices wind counter-clockwise.

import numpy as np


def signed_area(poly):
    """Shoelace area of a closed polygon given as an (N, 2) array.

    Positive when the vertices wind counter-clockwise.  The magnitude is
    the enclosed area for a SIMPLE polygon; for a self-intersecting one it
    is the sum of the signed areas of its loops, which is why this is a
    winding test and not an area measurement.
    """
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_ccw(poly):
    """Return the polygon wound counter-clockwise (positive area)."""
    poly = np.asarray(poly, float)
    if signed_area(poly) < 0.0:
        return poly[::-1].copy()
    return poly


def to_xy(z):
    """Complex points -> (N, 2) float array.

    NOTE: three generators carry their own version of this and they do
    NOT agree on the empty case -- the list-comprehension form returns
    shape (0,) where this one returns (0, 2).  They are therefore left
    alone for now rather than silently unified; migrating them needs a
    check of what each caller does with an empty tile list.
    """
    z = np.asarray(z)
    return np.column_stack([z.real, z.imag]).astype(float)


def _selftest():
    ok = True

    # A unit square, counter-clockwise, has area +1; reversed, -1.
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    good = (abs(signed_area(sq) - 1.0) < 1e-12
            and abs(signed_area(sq[::-1]) + 1.0) < 1e-12)
    ok &= good
    print(f"polygon2d: unit square area {signed_area(sq):+.3f} / "
          f"{signed_area(sq[::-1]):+.3f} {'OK' if good else 'FAIL'}")

    # A regular n-gon of circumradius 1 has area (n/2) sin(2 pi / n),
    # approaching pi.  This checks the formula, not just the sign.
    bad = []
    for n in (3, 4, 5, 6, 12, 60):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        p = np.stack([np.cos(t), np.sin(t)], axis=1)
        want = 0.5 * n * np.sin(2 * np.pi / n)
        if abs(signed_area(p) - want) > 1e-12:
            bad.append(f"{n}:{signed_area(p):.6f}!={want:.6f}")
    good = not bad
    ok &= good
    print(f"polygon2d: regular n-gon areas match (n/2)sin(2pi/n) "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # ensure_ccw is idempotent, orientation-fixing, and preserves the
    # vertex SET (it may only reverse the order).
    cw = sq[::-1]
    fixed = ensure_ccw(cw)
    good = (signed_area(fixed) > 0 and signed_area(ensure_ccw(fixed)) > 0
            and np.allclose(np.sort(fixed, axis=0), np.sort(sq, axis=0))
            and np.allclose(ensure_ccw(sq), sq))
    ok &= good
    print(f"polygon2d: ensure_ccw fixes, is idempotent, keeps the vertices "
          f"{'OK' if good else 'FAIL'}")

    # Translation invariance -- the area of a tile must not depend on where
    # in the plane the tiling put it.
    off = sq + np.array([13.5, -7.25])
    good = abs(signed_area(off) - signed_area(sq)) < 1e-12
    ok &= good
    print(f"polygon2d: area is translation invariant "
          f"{'OK' if good else 'FAIL'}")

    # to_xy accepts a python list of complex and a numpy complex array
    # alike, and agrees with the list-comprehension form on non-empty input.
    zs = [0 + 0j, 1 + 0j, 0.5 + 0.75j]
    ref = np.array([[p.real, p.imag] for p in zs], float)
    good = (np.allclose(to_xy(zs), ref)
            and np.allclose(to_xy(np.asarray(zs)), ref)
            and to_xy(zs).shape == (3, 2))
    ok &= good
    print(f"polygon2d: to_xy on list and array {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("polygon2d self-test failed")
