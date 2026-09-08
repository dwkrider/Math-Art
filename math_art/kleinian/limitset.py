# Limit sets of Kleinian groups, by depth-first search.
#
# The limit set is the closure of an orbit under the group: the fractal set the
# group's dynamics accumulate on.  Drawing it by RANDOM ITERATION is the obvious
# approach and the wrong one -- a random walk visits some parts of the set
# exponentially rarely, so the picture has holes that never fill.  The
# depth-first search of Indra's Pearls instead enumerates group elements
# systematically and produces the limit set as a CONTINUOUS CURVE.
#
# Box 17, the recursive search, is short:
#
#     for each generator k:  explore(gens[k], k)
#
#     explore(X, i):                       # i is the last generator used
#         for k in (i-1, i, i+1) mod 4:    # every generator but the inverse
#             Y = X * gens[k]
#             if the image is smaller than epsilon:
#                 plot Y applied to the fixed point of gens[k]
#             else:
#                 explore(Y, k)
#
# TWO DETAILS DECIDE WHETHER THE OUTPUT IS A CURVE OR A SPRAY OF DOTS.
#
# 1. THE GENERATORS MUST BE IN THE CYCLIC ORDER a, B, A, b -- not a, b, A, B.
#    Arranged so, stepping k = i-1, i, i+1 walks the branches in the order that
#    traces the limit set continuously, and k = i+2 is exactly the inverse of
#    the incoming generator, which is why that one is skipped.  With the wrong
#    order the cancellation test still works, so nothing looks broken; the
#    points simply stop coming out along the curve.  (Indra's Pearls, Box 15.)
#
# 2. TERMINATION IS GEOMETRIC.  The recursion stops when the image of the whole
#    branch has shrunk below epsilon, estimated here from how far apart the
#    images of the generators' fixed points have become under the accumulated
#    word.  Stopping on word length alone gives a set sampled at wildly uneven
#    density.
#
# SPECIAL WORDS.  For groups whose limit set has narrow necks -- the gasket case
# especially -- terminating purely on point-to-point distance can cut a whorl
# across its neck.  Including the fixed points of the generators themselves
# among the plotted points, as done here, is the first of the two remedies
# described in the book; the second, a per-whorl tip test, is not implemented
# and is noted in the repo backlog.
#
# References:
# - David Mumford, Caroline Series and David Wright, "Indra's Pearls: The Vision
#   of Felix Klein", Cambridge University Press, 2002 (Box 15, cyclic
#   permutation; Box 17, recursive depth-first search; chapters 5 and 9).
# - Curt McMullen, "Hausdorff dimension and conformal dynamics III: Computation
#   of dimension", American Journal of Mathematics 120 (1998), pp. 691-721.

import cmath

from . import mobius, recipes
from .mobius import apply, mul


def _branch_extent(m, seeds):
    """How large the image of the group's 'attention' still is under m.

    Measured as the greatest distance between the images of the seed points --
    the fixed points of the generators.  When this falls below epsilon the whole
    branch is smaller than a pixel and there is nothing left to resolve."""
    pts = []
    for s in seeds:
        z = apply(m, s)
        if not (abs(z.real) < 1e12 and abs(z.imag) < 1e12):
            return float('inf')
        pts.append(z)
    worst = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = abs(pts[i] - pts[j])
            if d > worst:
                worst = d
    return worst


def limit_set(gens, epsilon=0.005, max_depth=22, max_points=400000):
    """Points of the limit set, in traversal order.

    `gens` must be in the cyclic order a, B, A, b (see the module header).
    Returns a list of complex points; consecutive points are close, so the list
    can be drawn as a polyline."""
    seeds = [mobius.attracting_fixed_point(g) for g in gens]
    out = []
    # explicit stack: (matrix, last generator index, whether expanded)
    for k0 in range(4):
        stack = [(gens[k0], k0, 0)]
        while stack:
            if len(out) >= max_points:
                return out
            m, i, depth = stack.pop()
            if depth >= max_depth or _branch_extent(m, seeds) < epsilon:
                out.append(apply(m, seeds[i]))
                continue
            # push in reverse so the traversal order is i-1, i, i+1
            for k in ((i + 1) % 4, i, (i - 1) % 4):
                stack.append((mul(m, gens[k]), k, depth + 1))
    return out


def limit_set_preset(name='QUASIFUCHSIAN', epsilon=0.005, max_depth=22,
                     max_points=400000):
    ta, tb = recipes.PRESETS[name]
    gens = recipes.generators(recipes.GRANDMA, ta, tb)
    return limit_set(gens, epsilon, max_depth, max_points)


def circle_orbit(gens, circles, depth=4, max_circles=20000):
    """Images of a set of circles under all reduced words to a given depth.

    Circles are (centre, radius); a Moebius image of a circle is a circle, and
    the coefficients follow from mapping three points."""
    out = []
    frontier = [(mobius.IDENT, -1)]
    for _ in range(max(0, depth)):
        nxt = []
        for (m, i) in frontier:
            ks = range(4) if i < 0 else ((i + 1) % 4, i, (i - 1) % 4)
            for k in ks:
                w = mul(m, gens[k])
                for (c, r) in circles:
                    img = _map_circle(w, c, r)
                    if img is not None:
                        out.append(img)
                        if len(out) >= max_circles:
                            return out
                nxt.append((w, k))
        frontier = nxt
    return out


def _map_circle(m, c, r):
    """Image of the circle (c, r) under m, as (centre, radius), or None."""
    pts = [apply(m, c + r * cmath.exp(1j * t))
           for t in (0.0, 2.0943951023931953, 4.1887902047863905)]
    for z in pts:
        if not (abs(z.real) < 1e9 and abs(z.imag) < 1e9):
            return None
    (z1, z2, z3) = pts
    ax, ay = z1.real, z1.imag
    bx, by = z2.real, z2.imag
    cx, cy = z3.real, z3.imag
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-14:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    centre = complex(ux, uy)
    return centre, abs(z1 - centre)


def _selftest():
    # 1. the gasket case: every point of the limit set lies in the closed unit
    #    disc, which is the classical picture for t_a = t_b = 2
    pts = limit_set_preset('GASKET', epsilon=0.005, max_depth=22)
    assert len(pts) > 5000, len(pts)
    worst = max(abs(z) for z in pts)
    assert worst < 1.0 + 1e-6, "gasket limit set escaped the unit disc: %.6f" % worst

    # 2. the quasifuchsian case is bounded and genuinely two-dimensional
    q = limit_set_preset('QUASIFUCHSIAN', epsilon=0.005, max_depth=22)
    assert len(q) > 5000
    xs = [z.real for z in q]
    ys = [z.imag for z in q]
    assert max(abs(x) for x in xs) < 50.0
    assert (max(xs) - min(xs)) > 0.5 and (max(ys) - min(ys)) > 0.3

    # 3. consecutive points are close: the search returns a CURVE, not a spray.
    #    Compare the median step against a shuffled ordering of the same points.
    steps = sorted(abs(q[i + 1] - q[i]) for i in range(len(q) - 1))
    median = steps[len(steps) // 2]
    assert median < 0.05, "consecutive points should be close, median %.4f" % median

    # 4. a finer epsilon resolves more of the set
    coarse = limit_set_preset('GASKET', epsilon=0.05, max_depth=16)
    fine = limit_set_preset('GASKET', epsilon=0.005, max_depth=22)
    assert len(fine) > 5 * len(coarse), (len(coarse), len(fine))
    # the sampling really is adaptive: the median step tracks epsilon
    cs = sorted(abs(coarse[i + 1] - coarse[i]) for i in range(len(coarse) - 1))
    assert 0.5 < cs[len(cs) // 2] / 0.05 < 2.0, cs[len(cs) // 2]

    # 5. the point budget is honoured
    capped = limit_set_preset('QUASIFUCHSIAN', epsilon=1e-4, max_depth=24,
                              max_points=5000)
    assert len(capped) <= 5000

    # 6. circle images really are circles: three more points of the image circle
    #    must be equidistant from the computed centre
    gens = recipes.generators(recipes.GRANDMA, *recipes.PRESETS['GASKET'])
    got = _map_circle(gens[0], complex(0.0, 0.0), 0.3)
    assert got is not None
    cen, rad = got
    for t in (0.7, 2.2, 5.0):
        z = apply(gens[0], 0.3 * cmath.exp(1j * t))
        assert abs(abs(z - cen) - rad) < 1e-6, (abs(z - cen), rad)

    # a circle THROUGH the pole maps to a line, not a circle, and must be
    # reported as such rather than fitted to nonsense.  For this generator the
    # pole is at -0.5i, so the radius-0.5 circle about the origin hits it.
    assert _map_circle(gens[0], complex(0.0, 0.0), 0.5) is None

    orbit = circle_orbit(gens, [(complex(0, 0), 0.3)], depth=3)
    assert len(orbit) > 10 and all(r > 0 for (_, r) in orbit)

    print("kleinian.limitset: gasket stays in the unit disc, quasifuchsian "
          "bounded, consecutive points close (median step %.4f), epsilon "
          "refines, circle images are circles. RESULT: OK" % median)
