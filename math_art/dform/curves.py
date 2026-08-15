# Plane curves for D-form construction -- the flat outlines that get glued.
#
# Part of the Math Art D-form engine (`math_art/dform/`).  Python and
# numpy only -- no `bpy` -- so the engine imports and self-tests headlessly.
#
# A D-form is built from two flat pieces whose boundary curves have the
# SAME PERIMETER; that single precondition is what this module exists to
# guarantee.  Getting two *different* shapes to the same perimeter is the
# awkward step in practice -- even an ellipse has no elementary perimeter
# formula -- so we follow Orduno et al. (Bridges 2016) and do it on the
# discrete polyline: sample densely, measure the polygon perimeter, and
# uniformly scale one curve by the ratio.  Uniform scaling multiplies
# perimeter by exactly the scale factor, so the match is exact for the
# polygon we actually mesh, which is the only thing the solver sees.
#
# Curve library: ellipse, superellipse, egg, rounded regular polygon
# (an exact Minkowski sum of polygon and disc, so it is exactly convex),
# and the Cassinian oval -- convex for b/a >= sqrt(2), peanut-waisted
# below it, which the D-form theorems do not cover but which makes
# interesting shapes anyway.
#
# References:
#   R. R. Orduno, N. Winard, S. Bierwagen, D. Shell, N. Kalantar,
#       A. Borhani, E. Akleman, "A Mathematical Approach to Obtain
#       Isoperimetric Shapes for D-Form Construction," Bridges 2016,
#       pp. 277-284 -- the equal-perimeter procedure and the angle-defect
#       density mu(t) = 2*pi - theta_0(t) - theta_1(t).
#   G. D. Cassini (1680), the ovals; convexity threshold b/a = sqrt(2).

import math

import numpy as np

CURVE_KINDS = ('ELLIPSE', 'SUPERELLIPSE', 'EGG', 'ROUNDED_POLYGON',
               'CASSINI')

# dense sampling used before arclength resampling; the curve library is
# analytic, so this only has to beat the seam resolution comfortably
_DENSE = 2048


def _ellipse(t, aspect):
    return np.stack([np.cos(t), aspect * np.sin(t)], axis=1)


def _superellipse(t, aspect, n):
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / max(n, 1e-6)
    return np.stack([np.sign(c) * np.abs(c) ** e,
                     aspect * np.sign(s) * np.abs(s) ** e], axis=1)


def _egg(t, aspect, e):
    # an oval fattened at one end: y is modulated by (1 + e cos t), which
    # keeps x = cos t and moves the widest point off the short axis.
    # e is clamped away from -1, where the normalisation divides by zero
    # (the Blender UI already bounds it, but the engine is callable on
    # its own and must not hand the solver a NaN mesh)
    e = min(max(float(e), 0.0), 0.95)
    return np.stack([np.cos(t),
                     aspect * np.sin(t) * (1.0 + e * np.cos(t)) / (1.0 + e)],
                    axis=1)


def _rounded_polygon(n_samples, sides, corner):
    """Regular `sides`-gon Minkowski-summed with a disc of radius `corner`.

    Built as geometry rather than as a polar radius: circular arcs at the
    corners joined by straight segments parallel to the edges.  That is
    the exact offset body, so the result is exactly convex for every
    parameter value -- which matters because the D-form theorems assume
    convexity and a "rounded polygon" faked in polar coordinates is not
    reliably convex.
    """
    k = max(3, int(sides))
    c = min(max(float(corner), 1e-4), 0.999)
    # polygon of circumradius (1 - c) so the offset body has extent ~1
    rp = 1.0 - c
    verts = np.stack([rp * np.cos(2 * np.pi * np.arange(k) / k
                                  + np.pi / 2),
                      rp * np.sin(2 * np.pi * np.arange(k) / k
                                  + np.pi / 2)], axis=1)
    # the arc at vertex j spans the outward normals of its two edges
    pts = []
    per_corner = max(3, n_samples // k // 2)
    for j in range(k):
        prv, cur, nxt = verts[j - 1], verts[j], verts[(j + 1) % k]
        e_in = cur - prv
        e_out = nxt - cur
        # outward normal of a CCW edge (dx, dy) is (dy, -dx)
        a0 = math.atan2(-e_in[0], e_in[1])
        a1 = math.atan2(-e_out[0], e_out[1])
        while a1 < a0:
            a1 += 2 * np.pi
        a = np.linspace(a0, a1, per_corner)
        pts.append(cur + c * np.stack([np.cos(a), np.sin(a)], axis=1))
    return np.concatenate(pts, axis=0)


def _cassini(t, ratio):
    """Cassinian oval |P-F1|.|P-F2| = b^2 with a = 1, b = `ratio`.

    Polar form r^4 - 2 r^2 cos(2th) + 1 - b^4 = 0, outer root.  A single
    oval needs b > a; it is convex only for b/a >= sqrt(2).
    """
    b = max(float(ratio), 1.0001)
    c2 = np.cos(2 * t)
    disc = c2 ** 2 - 1.0 + b ** 4
    r2 = c2 + np.sqrt(np.maximum(disc, 0.0))
    r = np.sqrt(np.maximum(r2, 0.0))
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def curve_points(kind, aspect=1.0, super_n=3.0, egg=0.35, sides=3,
                 corner=0.35, cassini=1.6, samples=_DENSE):
    """Densely sampled closed curve, normalised to unit largest extent."""
    t = np.linspace(0.0, 2 * np.pi, samples, endpoint=False)
    if kind == 'SUPERELLIPSE':
        P = _superellipse(t, aspect, super_n)
    elif kind == 'EGG':
        P = _egg(t, aspect, egg)
    elif kind == 'ROUNDED_POLYGON':
        P = _rounded_polygon(samples, sides, corner)
        P = P * np.array([1.0, aspect])
    elif kind == 'CASSINI':
        P = _cassini(t, cassini)
        P = P * np.array([1.0, aspect])
    else:
        P = _ellipse(t, aspect)
    ext = float(np.max(P.max(axis=0) - P.min(axis=0)))
    if ext > 1e-12:
        P = P / ext
    return P - 0.5 * (P.max(axis=0) + P.min(axis=0))


def perimeter(P):
    """Perimeter of the closed polyline `P` (n,2) or (n,3)."""
    d = np.roll(P, -1, axis=0) - P
    return float(np.sum(np.linalg.norm(d, axis=1)))


def polygon_area(P):
    """Signed area of a closed planar polygon (positive = CCW)."""
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def resample_arclength(P, n):
    """Resample a closed polyline to `n` points uniform in arclength."""
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    want = np.linspace(0.0, total, n, endpoint=False)
    idx = np.searchsorted(s, want, side='right') - 1
    idx = np.clip(idx, 0, len(P) - 1)
    frac = (want - s[idx]) / np.maximum(seg[idx], 1e-15)
    nxt = (idx + 1) % len(P)
    return P[idx] + frac[:, None] * (P[nxt] - P[idx])


def match_perimeter(A, B):
    """Uniformly scale `B` so its DISCRETE perimeter equals `A`'s.

    The discrete perimeter is the one that matters: the solver glues the
    sampled polygons, so matching the analytic arclength would leave a
    residual mismatch of exactly the discretisation error.  Returns
    (B_scaled, scale).
    """
    pa, pb = perimeter(A), perimeter(B)
    s = pa / pb if pb > 1e-15 else 1.0
    return B * s, s


def turning_angles(P):
    """Exterior turning angle at each vertex of a closed polygon.

    Sums to 2*pi for a simple closed curve (the 2D Gauss-Bonnet
    identity); the interior angle at vertex i is pi - turn[i].
    """
    prv = P - np.roll(P, 1, axis=0)
    nxt = np.roll(P, -1, axis=0) - P
    a0 = np.arctan2(prv[:, 1], prv[:, 0])
    a1 = np.arctan2(nxt[:, 1], nxt[:, 0])
    d = a1 - a0
    return (d + np.pi) % (2 * np.pi) - np.pi


def interior_angles(P):
    """Interior angle at each vertex of a closed polygon (radians)."""
    return np.pi - turning_angles(P)


def is_convex(P, tol=1e-9):
    """True when every turn has the same sign (a convex simple polygon)."""
    t = turning_angles(P)
    return bool(np.all(t >= -tol) or np.all(t <= tol))


def ensure_ccw(P):
    """Return `P` wound counter-clockwise."""
    return P if polygon_area(P) >= 0.0 else P[::-1].copy()


def seam_correspondence(n, join_offset=0.0, flip=False):
    """Map seam index -> index into the second curve's boundary ring.

    The second piece is traversed in reverse by default, which is what
    makes the glued surface consistently oriented; `flip` restores the
    forward traversal (a genuinely different, usually more twisted,
    D-form from the same pair of outlines).  `join_offset` is the join
    point: the phase phi in [0,1) at which the two boundaries first meet,
    and the parameter that redistributes curvature along the seam.
    """
    i = np.arange(n)
    step = i if flip else -i
    return (int(round(join_offset * n)) + step) % n


def mu_density(ang_a, ang_b):
    """Angle-defect density mu = 2*pi - theta_0 - theta_1 along the seam.

    Orduno et al. (Bridges 2016), Eq. 1: the curvature of a D-form lives
    entirely on the seam, and this is how much sits at each seam vertex.
    Summing it over a closed seam gives 4*pi (their Eq. 3).
    """
    return 2 * np.pi - ang_a - ang_b


def _selftest():
    ok = True

    # every curve in the library must close up, be simple, and (except a
    # waisted Cassinian) be convex -- the D-form precondition
    cases = [('ELLIPSE', dict(aspect=0.6)),
             ('SUPERELLIPSE', dict(aspect=0.8, super_n=3.0)),
             ('EGG', dict(aspect=0.7, egg=0.35)),
             ('ROUNDED_POLYGON', dict(sides=3, corner=0.35)),
             ('ROUNDED_POLYGON', dict(sides=5, corner=0.2)),
             ('CASSINI', dict(cassini=1.6))]
    bad = []
    for kind, kw in cases:
        P = curve_points(kind, **kw)
        if not is_convex(P, tol=1e-6):
            bad.append(kind + str(kw.get('sides', '')))
    good = not bad
    ok &= good
    print(f"curves: library is convex {'OK' if good else 'FAIL ' + ','.join(bad)}")

    # the waisted Cassinian must be detected as NON-convex, or the UI
    # warning that depends on it is worthless
    good = not is_convex(curve_points('CASSINI', cassini=1.15), tol=1e-6)
    ok &= good
    print(f"curves: waisted Cassinian detected non-convex "
          f"{'OK' if good else 'FAIL'}")

    # 2D Gauss-Bonnet: total turning of a simple closed curve is 2*pi
    worst = 0.0
    for kind, kw in cases:
        P = resample_arclength(curve_points(kind, **kw), 256)
        worst = max(worst, abs(float(np.sum(turning_angles(P))) - 2 * np.pi))
    good = worst < 1e-9
    ok &= good
    print(f"curves: total turning = 2pi (err {worst:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # arclength resampling must be uniform to machine precision
    P = resample_arclength(curve_points('EGG', aspect=0.6, egg=0.4), 200)
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    spread = float(seg.max() - seg.min()) / float(seg.mean())
    # chords, not arcs: samples are equally spaced ALONG the polyline, so
    # the straight-line gap is short by ~(k*l)^2/24 where the curve bends
    # hardest.  A percent is the geometry, not a defect.
    good = spread < 1e-2
    ok &= good
    print(f"curves: arclength resample uniform (spread {spread:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # the isoperimetric step: two DIFFERENT shapes, exactly equal
    # discrete perimeters, and equal seam segment lengths after
    # resampling -- the precondition the whole solver rests on
    A = resample_arclength(curve_points('ELLIPSE', aspect=0.6), 160)
    B = resample_arclength(curve_points('ROUNDED_POLYGON', sides=3,
                                        corner=0.3), 160)
    B, s = match_perimeter(A, B)
    err = abs(perimeter(A) - perimeter(B)) / perimeter(A)
    la = np.linalg.norm(np.roll(A, -1, axis=0) - A, axis=1)
    lb = np.linalg.norm(np.roll(B, -1, axis=0) - B, axis=1)
    seg_err = float(np.max(np.abs(la.mean() - lb.mean())))
    good = err < 1e-12 and seg_err < 1e-12
    ok &= good
    print(f"curves: perimeter match exact (rel {err:.2e}, scale {s:.4f}) "
          f"{'OK' if good else 'FAIL'}")

    # mu must integrate to 4*pi over the seam: the Gauss-Bonnet content
    # of Orduno et al. Eq. 3.  Interior angles of both pieces sum with
    # the 2*pi at each seam vertex to exactly 4*pi total.
    n = 240
    A = resample_arclength(curve_points('ELLIPSE', aspect=0.6), n)
    B = resample_arclength(curve_points('EGG', aspect=0.8, egg=0.3), n)
    B, _ = match_perimeter(A, B)
    b = seam_correspondence(n, 0.25)
    mu = mu_density(interior_angles(A), interior_angles(B)[b])
    tot = float(np.sum(mu))
    good = abs(tot - 4 * np.pi) < 1e-9
    ok &= good
    print(f"curves: total seam defect = 4pi (got {tot:.9f}) "
          f"{'OK' if good else 'FAIL'}")

    # the correspondence must be a permutation for every offset/flip
    good = True
    for off in (0.0, 0.25, 0.5, 0.777):
        for fl in (False, True):
            b = seam_correspondence(64, off, fl)
            good &= sorted(b.tolist()) == list(range(64))
    ok &= good
    print(f"curves: seam correspondence is a permutation "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("dform.curves self-test failed")
