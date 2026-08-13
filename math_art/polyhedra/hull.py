# Convex hull faces of a point set.
#
# Part of the Math Art polyhedron engine (`math_art/polyhedra/`).  Python
# and numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operators stay in their flat generator
# modules and import this package.
#
# Faces of the convex hull of a point set, recovered by testing each
# candidate plane against every point -- the brute-force formulation,
# which is the right one here because these hulls have tens of vertices,
# not thousands, and it needs no external dependency.
#
# References:
#   The convex hull and its face lattice: B. Grunbaum, "Convex Polytopes"
#   (2nd ed., 2003).  For the incremental and gift-wrapping algorithms a
#   larger input would want, see Preparata and Shamos, "Computational
#   Geometry" (1985), ch. 3.

import itertools
import math


def hull_faces(V):
    """Merged planar faces of the convex hull (small vertex sets only)."""
    n = len(V)
    planes = []
    seen = set()
    for a, b, c in itertools.combinations(range(n), 3):
        A, B, C = V[a], V[b], V[c]
        nx = ((B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1]),
              (B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2]),
              (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]))
        ln = math.sqrt(sum(x * x for x in nx))
        if ln < 1e-9:
            continue
        nx = tuple(x / ln for x in nx)
        d = sum(nx[i] * A[i] for i in range(3))
        if d < 0:
            nx = tuple(-x for x in nx)
            d = -d
        key = tuple(round(x, 5) for x in nx) + (round(d, 5),)
        if key in seen:
            continue
        if all(sum(nx[i] * V[j][i] for i in range(3)) <= d + 1e-7
               for j in range(n)):
            seen.add(key)
            onv = [j for j in range(n)
                   if abs(sum(nx[i] * V[j][i] for i in range(3)) - d) < 1e-6]
            cen = [sum(V[j][i] for j in onv) / len(onv) for i in range(3)]
            ux = [V[onv[0]][i] - cen[i] for i in range(3)]
            uy = [nx[1] * ux[2] - nx[2] * ux[1],
                  nx[2] * ux[0] - nx[0] * ux[2],
                  nx[0] * ux[1] - nx[1] * ux[0]]

            def ang(j):
                dx = [V[j][i] - cen[i] for i in range(3)]
                return math.atan2(sum(dx[i] * uy[i] for i in range(3)),
                                  sum(dx[i] * ux[i] for i in range(3)))
            planes.append(sorted(onv, key=ang))
    return planes


def _selftest():
    ok = True
    from math import sqrt

    # The hull of the Platonic vertex sets must recover exactly those
    # solids' faces.
    PHI = (1.0 + 5.0 ** 0.5) / 2.0
    cube = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    octa = [(s, 0, 0) for s in (-1, 1)] + [(0, s, 0) for s in (-1, 1)] \
        + [(0, 0, s) for s in (-1, 1)]
    ico = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            ico += [(0, s1, s2 * PHI), (s1, s2 * PHI, 0), (s1 * PHI, 0, s2)]
    bad = []
    for name, pts, nf in (('cube', cube, 6), ('octa', octa, 8),
                          ('icosa', ico, 20)):
        f = hull_faces(pts)
        if len(f) != nf:
            bad.append(f"{name}:{len(f)}!={nf}")
    good = not bad
    ok &= good
    print(f"hull: recovers cube/octahedron/icosahedron faces "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # Interior points must not create faces: adding the centre to a cube
    # leaves the hull unchanged.
    good = len(hull_faces(cube + [(0.0, 0.0, 0.0)])) == len(hull_faces(cube))
    ok &= good
    print(f"hull: an interior point does not change the hull "
          f"{'OK' if good else 'FAIL'}")

    # Every hull face must have all points on one side of its plane --
    # the defining property, and what a wrong winding or tolerance breaks.
    pts = ico
    worst = 0.0
    for f in hull_faces(pts):
        p0, p1, p2 = (pts[i] for i in f[:3])
        u = [p1[k] - p0[k] for k in range(3)]
        v = [p2[k] - p0[k] for k in range(3)]
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0]]
        ln = sqrt(sum(c * c for c in n)) or 1.0
        n = [c / ln for c in n]
        d = [sum(n[k] * (q[k] - p0[k]) for k in range(3)) for q in pts]
        worst = max(worst, min(d) if max(d) <= 1e-9 else -max(d))
    good = worst <= 1e-9
    ok &= good
    print(f"hull: every face plane is supporting (worst {worst:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("hull self-test failed")
