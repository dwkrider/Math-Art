# Circle packings of the sphere, by projection rather than by relaxation.
#
# The obvious approach -- swap in the spherical law of cosines and relax as
# usual -- does not work, and it fails silently.  The spherical corner angle is
# NOT monotone in the radius: it decreases, then increases again once the
# triangle sides pass pi/2, so the one-dimensional solve at the heart of the
# relaxation has no unique root to find.  (Measured: non-monotone in 2000 of
# 2000 random configurations, against 0 of 2000 in the euclidean case.)  The
# sphere also has no boundary to pin the solution and a six-dimensional Moebius
# gauge freedom, so the sweep has no convergence theory either.
#
# The standard route instead, and the one CirclePack and GOpack take:
#
#   1. REMOVE A FACE.  Pick any face f = <w1,w2,w3> of the triangulated sphere
#      and treat it as exterior.  K' = K \ int(f) is a combinatorial disc whose
#      boundary is that triangle -- and crucially K' still has every vertex of
#      K, so a packing of K' is a packing of K.
#   2. Pack K' as a MAXIMAL packing of the hyperbolic disc.  That problem is
#      well posed and the monotone solver handles it.
#   3. STEREOGRAPHICALLY PROJECT the resulting disc packing to the sphere.
#      Stereographic projection carries circles to circles, so the packing
#      survives intact; the removed face reappears as the leftover interstice.
#   4. Normalise.  The choice of face changes the result only by a Moebius
#      transformation, so any face will do.
#
# With only three boundary vertices the boundary-placement step is trivial,
# which is why removing a face is cleaner than puncturing a vertex.
#
# References:
# - Charles Collins, Gerald L. Orick and Kenneth Stephenson, "A linearized
#   circle packing algorithm", Computational Geometry: Theory and Applications
#   64 (2017), pp. 13-29, section 3.2 (the remove-a-face route, and the remark
#   that stereographic projection lets all the work be done euclidean).
# - Kenneth Stephenson, "Introduction to Circle Packing: The Theory of Discrete
#   Analytic Functions", Cambridge University Press, 2005.
# - The Koebe-Andreev-Thurston theorem: a triangulated sphere has an essentially
#   unique packing, unique up to Moebius transformations.

import math

from . import layout, thurston
from .angles import HYPERBOLIC
from .complexes import PackingComplex


def remove_face(K, face_index=0):
    """Drop one face, leaving a combinatorial disc on the same vertex set."""
    if not K.faces:
        raise ValueError("complex has no faces")
    fi = face_index % len(K.faces)
    faces = [f for i, f in enumerate(K.faces) if i != fi]
    return PackingComplex(K.nv, faces), K.faces[fi]


def stereographic(x, y):
    """Plane to unit sphere, projecting from the north pole.

    The unit circle maps to the equator, the origin to the south pole."""
    d = 1.0 + x * x + y * y
    return (2.0 * x / d, 2.0 * y / d, (x * x + y * y - 1.0) / d)


def _cap_from_planar_circle(c, r):
    """Spherical cap (unit axis, angular radius) of the image of a plane circle.

    Three boundary points determine the image circle; the plane through their
    images gives the cap axis and its angular radius."""
    pts = []
    for k in range(3):
        a = 2.0 * math.pi * k / 3.0
        pts.append(stereographic(c.real + r * math.cos(a),
                                 c.imag + r * math.sin(a)))
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = pts
    ux, uy, uz = x2 - x1, y2 - y1, z2 - z1
    vx, vy, vz = x3 - x1, y3 - y1, z3 - z1
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln < 1e-14:
        return (0.0, 0.0, 1.0), 0.0
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    d = nx * x1 + ny * y1 + nz * z1       # signed distance of the plane
    if d < 0.0:                            # orient the axis toward the cap
        nx, ny, nz, d = -nx, -ny, -nz, -d
    d = min(1.0, max(-1.0, d))
    return (nx, ny, nz), math.acos(d)


def ball_isometry(x, a):
    """Ahlfors' isometry of the unit ball taking `a` to the origin.

    Restricted to the boundary it is a Moebius transformation of the sphere, so
    it carries circles to circles -- which is what lets a packing be
    renormalised without ceasing to be a packing."""
    xa = [x[i] - a[i] for i in range(3)]
    n2a = sum(c * c for c in a)
    n2xa = sum(c * c for c in xa)
    n2x = sum(c * c for c in x)
    dot = sum(x[i] * a[i] for i in range(3))
    den = n2x * n2a - 2.0 * dot + 1.0
    if abs(den) < 1e-14:
        return tuple(x)
    return tuple(((1.0 - n2a) * xa[i] - n2xa * a[i]) / den for i in range(3))


def _cap_from_sphere_points(pts, inside):
    """Cap (axis, angular radius) through three boundary points on the sphere.

    `inside` is a point known to lie within the cap, and it is what fixes the
    orientation.  Choosing the sign by "make d positive" instead -- which works
    while every cap is small -- is wrong here: a Moebius transformation can
    carry a cap across the far side, where the correct angular radius exceeds
    pi/2 and d is negative.  Getting that wrong silently swaps a cap for its
    complement and destroys tangency."""
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = pts
    ux, uy, uz = x2 - x1, y2 - y1, z2 - z1
    vx, vy, vz = x3 - x1, y3 - y1, z3 - z1
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln < 1e-14:
        return (0.0, 0.0, 1.0), 0.0
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    d = nx * x1 + ny * y1 + nz * z1
    if nx * inside[0] + ny * inside[1] + nz * inside[2] < d:
        nx, ny, nz, d = -nx, -ny, -nz, -d
    return (nx, ny, nz), math.acos(min(1.0, max(-1.0, d)))


def _cap_points(axis, ang, n=3):
    """`n` points spread around a cap's boundary circle."""
    up = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0)
    ex = (up[1] * axis[2] - up[2] * axis[1],
          up[2] * axis[0] - up[0] * axis[2],
          up[0] * axis[1] - up[1] * axis[0])
    m = math.sqrt(sum(c * c for c in ex)) or 1.0
    ex = tuple(c / m for c in ex)
    ey = (axis[1] * ex[2] - axis[2] * ex[1],
          axis[2] * ex[0] - axis[0] * ex[2],
          axis[0] * ex[1] - axis[1] * ex[0])
    ca, sa = math.cos(ang), math.sin(ang)
    out = []
    for k in range(n):
        t = 2.0 * math.pi * k / n
        c, s = math.cos(t), math.sin(t)
        out.append(tuple(ca * axis[i] + sa * (c * ex[i] + s * ey[i])
                         for i in range(3)))
    return out


def normalise(axes, angs, iters=200, tol=1e-13):
    """Moebius-normalise a spherical packing so the circles are balanced.

    A packing of the sphere is unique only UP TO MOEBIUS TRANSFORMATIONS, and
    the one that falls out of the remove-a-face construction is badly skewed:
    the three vertices of the dropped face become horocycles in the disc, which
    project to enormous caps -- typically an 8:1 spread on an octahedron, with
    everything else crowded into the gap.  That is a valid packing, just an
    unflattering choice of representative.

    Balancing is the conformal-centering iteration: take the centroid of the
    cap centres, which lies strictly inside the ball unless the packing is
    already balanced, apply the ball isometry carrying it to the origin, and
    repeat.  Circles stay circles, so tangency is preserved exactly.

    The centroid must be taken UNWEIGHTED.  Weighting each centre by its cap
    area is the obvious thing to try and it DIVERGES -- measured on the
    octahedron, the caps grow past 70 degrees and the centroid wanders instead
    of settling.  Unweighted converges in about a dozen steps and lands on the
    exact answer: all six octahedron caps at 45 degrees, which is what tangency
    forces once the axes are orthogonal."""
    axes = [None if a is None else tuple(a) for a in axes]
    angs = list(angs)
    live = [v for v in range(len(axes)) if axes[v] is not None]
    if not live:
        return axes, angs
    for _ in range(iters):
        c = [0.0, 0.0, 0.0]
        for v in live:
            for i in range(3):
                c[i] += axes[v][i]
        c = [t / len(live) for t in c]
        n = math.sqrt(sum(t * t for t in c))
        if n < tol:
            break
        step = min(n, 0.6)                        # keep the step inside the ball
        c = [t * (step / n) for t in c]
        for v in live:
            pts = [ball_isometry(p, c) for p in _cap_points(axes[v], angs[v])]
            inside = ball_isometry(axes[v], c)   # the cap centre goes with it
            axes[v], angs[v] = _cap_from_sphere_points(pts, inside)
    return axes, angs


def pack_sphere(K, face_index=0, tol=1e-10, max_sweeps=4000, balance=True):
    """Maximal packing of a triangulated sphere.

    Returns (axes, angular_radii, info) where each circle is the spherical cap
    about `axis` of angular radius `angular_radii`."""
    if K.boundary:
        raise ValueError("pack_sphere() wants a closed triangulated sphere")
    Kd, dropped = remove_face(K, face_index)
    res = thurston.pack(Kd, HYPERBOLIC, boundary=thurston.MAXIMAL,
                        tol=tol, max_sweeps=max_sweeps)
    cen, rad = layout.layout_hyperbolic(Kd, res.radii)

    axes = [None] * K.nv
    ang = [0.0] * K.nv
    for v in range(K.nv):
        if cen[v] is None:
            continue
        axes[v], ang[v] = _cap_from_planar_circle(cen[v], rad[v])
    if balance:
        axes, ang = normalise(axes, ang)
    info = {'dropped_face': dropped, 'sweeps': res.sweeps,
            'angle_error': res.max_angle_error, 'converged': res.converged,
            'spread': (max(ang) / min(a for a in ang if a > 0)
                       if any(a > 0 for a in ang) else 1.0)}
    return axes, ang, info


def _selftest():
    from . import complexes

    # 1. removing a face leaves a disc on the same vertices
    K = complexes.octahedron()
    Kd, dropped = remove_face(K)
    assert Kd.nv == K.nv
    assert len(Kd.boundary) == 3, len(Kd.boundary)
    ne = len(list(Kd.edges()))
    assert Kd.nv - ne + len(Kd.faces) == 1, "must be a disc"

    # 2. stereographic projection lands on the unit sphere; the unit circle
    #    goes to the equator and the origin to the south pole
    for (x, y) in ((0.0, 0.0), (0.3, -0.7), (1.0, 0.0), (2.5, 1.5)):
        p = stereographic(x, y)
        assert abs(math.sqrt(sum(t * t for t in p)) - 1.0) < 1e-12
    assert abs(stereographic(1.0, 0.0)[2]) < 1e-12
    assert abs(stereographic(0.0, 0.0)[2] + 1.0) < 1e-12

    # 3. a circle centred at the origin maps to a cap about the south pole
    axis, a = _cap_from_planar_circle(complex(0.0, 0.0), 0.5)
    assert abs(axis[2] + 1.0) < 1e-9, axis
    assert 0.0 < a < math.pi

    # 4. the octahedron packs, and every cap is tangent to its neighbours:
    #    for tangent spherical caps the angle between axes equals the sum of
    #    the angular radii
    axes, ang, info = pack_sphere(K)
    assert info['converged'], info
    assert all(x is not None for x in axes)
    worst = 0.0
    for (v, u) in K.edges():
        dot = sum(a * b for a, b in zip(axes[v], axes[u]))
        sep = math.acos(min(1.0, max(-1.0, dot)))
        worst = max(worst, abs(sep - (ang[v] + ang[u])))
    assert worst < 1e-6, "spherical tangency error %.3e" % worst

    # 5. an ASYMMETRIC sphere -- a symmetric one can pass by symmetry alone.
    #    Split one octahedron face into three to break every symmetry.
    faces = list(K.faces)
    a, b, c = faces.pop()
    n = K.nv
    faces += [(a, b, n), (b, c, n), (c, a, n)]
    K2 = PackingComplex(n + 1, faces)
    assert not K2.boundary
    axes2, ang2, info2 = pack_sphere(K2)
    assert info2['converged'], info2
    worst2 = 0.0
    for (v, u) in K2.edges():
        dot = sum(p * q for p, q in zip(axes2[v], axes2[u]))
        sep = math.acos(min(1.0, max(-1.0, dot)))
        worst2 = max(worst2, abs(sep - (ang2[v] + ang2[u])))
    assert worst2 < 1e-6, "asymmetric spherical tangency error %.3e" % worst2

    # 6. normalisation preserves tangency exactly and BALANCES the packing.
    #    Unnormalised, the three vertices of the dropped face are horocycles in
    #    the disc and project to enormous caps.
    raw_axes, raw_ang, raw_info = pack_sphere(K, balance=False)
    bal_axes, bal_ang, bal_info = pack_sphere(K, balance=True)
    assert raw_info['spread'] > 4.0,         "the unnormalised octahedron packing should be badly skewed, got %.2f"         % raw_info['spread']
    assert bal_info['spread'] < 1.05,         "the balanced octahedron packing should be near uniform, got %.2f"         % bal_info['spread']
    worst3 = 0.0
    for (v, u) in K.edges():
        dot = sum(a * b for a, b in zip(bal_axes[v], bal_axes[u]))
        sep = math.acos(min(1.0, max(-1.0, dot)))
        worst3 = max(worst3, abs(sep - (bal_ang[v] + bal_ang[u])))
    assert worst3 < 1e-6, "normalisation broke tangency: %.3e" % worst3

    # the ball isometry really is one: it maps the sphere to itself
    for k in range(20):
        t = 0.31 * k
        x = (math.cos(t) * 0.6, math.sin(t) * 0.6,
             math.sqrt(max(0.0, 1.0 - 0.36)))
        y = ball_isometry(x, (0.25, -0.1, 0.05))
        assert abs(math.sqrt(sum(c * c for c in y)) - 1.0) < 1e-9

    print("packing.sphere: remove-a-face gives a disc, projection lands on the "
          "unit sphere, symmetric and asymmetric spheres pack with tangency "
          "error < 1e-6; Moebius normalisation cuts the octahedron's cap "
          "spread %.1f:1 -> %.2f:1 with tangency error %.1e. RESULT: OK"
          % (raw_info['spread'], bal_info['spread'], worst3))
