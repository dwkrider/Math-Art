
# Moving frames along a space curve: orienting a cross-section as it
# sweeps.
#
# Every generator in this repo that sweeps a profile along a space curve --
# knots, ruled surfaces, space-filling curves, tree limbs, tubes of any
# kind -- needs a moving reference frame, and the obvious choice is broken.
#
# (This module was called `turtle_frame.py` until the knot engine was
# extracted, after the differential-turtle-geometry formulation of the
# `frames` construction below.  The name suggested an L-system feature,
# which is why several generators grew their own copies of a tube sweep
# rather than finding it here; the L-system provenance is preserved in the
# references, where it belongs.)
#
# THE PROBLEM WITH THE FRENET FRAME.  The Frenet frame is built from the
# curve's own derivatives, so it is:
#
#   * UNDEFINED where the curvature vanishes -- along any straight
#     section, and at every inflection point;
#   * DISCONTINUOUS across an inflection, where the normal flips through
#     180 degrees.
#
# A tube swept with it visibly pops and twists at exactly the places a
# designed curve is most likely to have.
#
# THE FIX.  Prusinkiewicz, Mundermann, Karwowski and Lane define the
# frame by TURTLE rotation rates instead of by curve derivatives.  Write
# the frame as heading H, left L, up U with H x L = U.  Framing a GIVEN
# curve whose unit tangent at the next sample is T' needs
#
#     d(theta_U) =  T' . L         d(theta_L) = -T' . U
#
# and that is all: two of the three rotational degrees of freedom are
# fixed, and the third -- rotation about H itself -- is multiplied by
# zero and remains FREE.
#
# That free parameter IS the choice of frame:
#
#   * choose the roll so L stays in the osculating plane  -> Frenet;
#   * choose no roll at all (omega_H = 0)                 -> PARALLEL
#     TRANSPORT, which minimises rotation about the tube axis.
#
# So one implementation subsumes both, is defined everywhere including
# straight sections, and gains free TWIST (an extra roll per unit length)
# as a bonus.  Parallel transport is the default because it is what you
# almost always want: no gratuitous spin in the swept profile.
#
# TWO KERNELS, ONE FRAME.  `frames()` below transports by the axis-angle
# rotation between consecutive tangents, which it derives from the sampled
# points.  When the tangents are known EXACTLY -- an analytic curve such as
# a Bezier, rather than a polyline -- the double-reflection method of Wang
# et al. is the better kernel: same cost per step, but fourth-order
# accurate instead of second, because its first reflection is taken across
# the chord and so accounts for the curve between samples.
# `transport_normals()` exposes it.  Both compute the same object, a
# rotation-minimising frame; they differ in how the tangents arrive and in
# the order of accuracy, which is why both are kept.
#
# References:
# - Przemyslaw Prusinkiewicz, Lars Mundermann, Radoslaw Karwowski and
#   Brendan Lane, "The use of positional information in the modeling of
#   plants", Proceedings of SIGGRAPH 2001, pp. 289-300 -- section 4 and
#   Appendix A.1 (the fundamental theorem of differential turtle
#   geometry: given the rotation rates and an initial frame, the moving
#   frame and the curve are uniquely determined).
# - Jules Bloomenthal, "Calculation of reference frames along a space
#   curve", Graphics Gems, 1990 -- the rotation-minimising construction
#   that parallel transport implements.
# - Andrew J. Hanson and Hui Ma, "Parallel transport approach to curve
#   framing", Indiana University TR-425, 1995.
# - Wenping Wang, Bert Juttler, Dayue Zheng and Yang Liu, "Computation of
#   rotation minimizing frames", ACM TOG 27(1), 2008 -- the
#   double-reflection method used by `transport_normals`.

import math

import numpy as np

PARALLEL, FRENET, FIXED_UP = 'PARALLEL', 'FRENET', 'FIXED_UP'


def tangents(points, closed=False):
    """Unit tangents at each sample of a polyline.

    Central differences inside, one-sided at the ends -- and never a
    zero vector: a repeated point inherits its neighbour's direction
    rather than producing a NaN downstream.
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n < 2:
        return np.tile(np.array([0.0, 0.0, 1.0]), (max(n, 1), 1))
    T = np.zeros_like(P)
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T[1:-1] = P[2:] - P[:-2]
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    norms = np.linalg.norm(T, axis=1)
    bad = norms < 1e-12
    if bad.any():
        # inherit from the previous good tangent, else +Z
        last = np.array([0.0, 0.0, 1.0])
        for i in range(n):
            if bad[i]:
                T[i] = last
                norms[i] = 1.0
            else:
                last = T[i] / norms[i]
    return T / norms[:, None]


def frames(points, mode=PARALLEL, up=(0.0, 0.0, 1.0), twist=0.0,
           closed=False, initial_left=None):
    """(H, L, U) arrays, one orthonormal frame per sample.

    `mode`:
      PARALLEL  rotation-minimising; the frame is carried along the curve
                by the smallest rotation that maps each tangent to the
                next.  Defined everywhere, including straight runs.
      FRENET    L is placed in the osculating plane.  Provided for
                compatibility and comparison; it is undefined where the
                curvature vanishes, and this implementation falls back to
                parallel transport at those samples rather than emitting
                NaNs.
      FIXED_UP  L is H x up, i.e. the profile is kept level.  Cheap and
                stable, but degenerates where H is parallel to `up`.

    `twist` is an extra roll about H, in degrees over the whole curve --
    the free third degree of freedom, exposed rather than hidden.
    """
    H = tangents(points, closed=closed)
    n = len(H)
    up = np.asarray(up, dtype=float)
    if np.linalg.norm(up) < 1e-12:
        up = np.array([0.0, 0.0, 1.0])
    up = up / np.linalg.norm(up)

    L = np.zeros_like(H)
    U = np.zeros_like(H)

    if initial_left is not None:
        l0 = np.asarray(initial_left, dtype=float)
        l0 = l0 - H[0] * float(np.dot(l0, H[0]))
    else:
        l0 = np.cross(up, H[0])
        if np.linalg.norm(l0) < 1e-9:
            # tangent parallel to `up`: any perpendicular will do
            alt = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(alt, H[0]))) > 0.9:
                alt = np.array([0.0, 1.0, 0.0])
            l0 = np.cross(alt, H[0])
    l0 = l0 / max(np.linalg.norm(l0), 1e-12)
    L[0] = l0
    U[0] = np.cross(H[0], L[0])

    if mode == FIXED_UP:
        for i in range(n):
            li = np.cross(up, H[i])
            m = np.linalg.norm(li)
            if m < 1e-9:
                li = L[i - 1] if i else l0
            else:
                li = li / m
            L[i] = li
            U[i] = np.cross(H[i], L[i])
    else:
        for i in range(1, n):
            L[i] = _transport(H[i - 1], H[i], L[i - 1])
            U[i] = np.cross(H[i], L[i])
        if mode == FRENET:
            L, U = _reframe_frenet(points, H, L, U, closed)

    if twist:
        total = math.radians(twist)
        for i in range(n):
            a = total * (i / max(n - 1, 1))
            c, s = math.cos(a), math.sin(a)
            li, ui = L[i].copy(), U[i].copy()
            L[i] = li * c + ui * s
            U[i] = -li * s + ui * c

    return H, L, U


def transport_normals(points, tangents, first_normal):
    """Propagate `first_normal` along a curve by DOUBLE REFLECTION.

    The alternative kernel to `frames()`, for callers that already hold
    exact tangents (an analytic curve rather than a sampled polyline).
    Each step is a composition of two reflections, hence a rotation, so
    orthonormality is preserved exactly in exact arithmetic; taking the
    first reflection across the chord makes it fourth-order accurate.

    Returns the array of normals only -- cross with the tangents for the
    third axis.  Wang, Juttler, Zheng and Liu, ACM TOG 27(1), 2008.
    """
    points = np.asarray(points, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    normals = np.empty_like(points)
    normals[0] = _unit(first_normal
                       - (first_normal @ tangents[0]) * tangents[0])
    for i in range(len(points) - 1):
        v1 = points[i + 1] - points[i]
        c1 = v1 @ v1
        if c1 < 1e-18:                       # duplicated sample
            normals[i + 1] = normals[i]
            continue
        reflected_normal = normals[i] - (2 / c1) * (v1 @ normals[i]) * v1
        reflected_tangent = tangents[i] - (2 / c1) * (v1 @ tangents[i]) * v1
        v2 = tangents[i + 1] - reflected_tangent
        c2 = v2 @ v2
        normals[i + 1] = (
            reflected_normal
            if c2 < 1e-18
            else reflected_normal - (2 / c2) * (v2 @ reflected_normal) * v2
        )
    return normals


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def closed_tangents(points):
    """Unit tangents of a CLOSED polyline, by central differences that
    wrap.  The convention every closed sweep in this repo already uses."""
    P = np.asarray(points, dtype=float)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)


def closure_holonomy(points, tangents, normals):
    """The angle by which a transported frame fails to return to itself
    after one circuit of a closed curve.

    A closed curve generally has non-zero holonomy: carry a normal all the
    way round and it comes back rotated about the tangent.  Sweeping a
    tube without accounting for it leaves that entire rotation as a twist
    discontinuity at the seam.

    Measured with atan2 over the FULL range.  Using `sign * acos(dot)`
    instead is a real trap -- it is the formulation that left a 1.7 radian
    kink at the seam of two generators' tubes, because it recovers the
    wrong branch and the error lands entirely on the closing edge.
    """
    P = np.asarray(points, dtype=float)
    T = np.asarray(tangents, dtype=float)
    N = np.asarray(normals, dtype=float)
    # one more double-reflection step, across the closing edge
    wrapped = transport_normals(np.stack([P[-1], P[0]]),
                                np.stack([T[-1], T[0]]), N[-1])[1]
    wrapped = _unit(wrapped - T[0] * float(np.dot(wrapped, T[0])))
    binormal = np.cross(T[0], N[0])
    return math.atan2(float(np.dot(wrapped, binormal)),
                      float(np.dot(wrapped, N[0])))


def closed_frames(points):
    """(T, N, B) for a closed curve, with the closure holonomy already
    distributed evenly so the frame joins up at the seam.

    The kernel is `transport_normals` -- double reflection, fourth order.
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    T = closed_tangents(P)
    ref = (np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9
           else np.array([1.0, 0.0, 0.0]))
    n0 = _unit(np.cross(T[0], ref))
    N = transport_normals(P, T, n0)
    theta = closure_holonomy(P, T, N)
    B = np.cross(T, N)
    # spread the residual evenly: ring i takes i/n of it
    corr = -theta * np.arange(n) / n
    c, s = np.cos(corr)[:, None], np.sin(corr)[:, None]
    return T, N * c + B * s, -N * s + B * c


def closed_tube(points, radius, sides, weld=False):
    """Sweep a closed round tube of `radius` along `points`.

    Returns (verts, faces) with `len(points) * sides` vertices and the
    same number of quads.  Vertices are ordered ring by ring.

    `weld=True` additionally joins each ring to the next by the whole-ring
    rotation that best aligns them (nearest-vertex).  That absorbs any
    residual fractional twist into a clean shift at the seam rather than a
    fold, which matters when `sides` is small.
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n < 3:
        return [], []
    T, N, B = closed_frames(P)
    ang = 2.0 * math.pi * np.arange(sides) / sides
    ca, sa = np.cos(ang), np.sin(ang)
    rings = (P[:, None, :]
             + radius * (ca[None, :, None] * N[:, None, :]
                         + sa[None, :, None] * B[:, None, :]))
    verts = [tuple(v) for v in rings.reshape(-1, 3)]

    shift = np.zeros(n, dtype=int)
    if weld:
        for i in range(n):
            j = (i + 1) % n
            d = np.linalg.norm(rings[j][None, :, :] - rings[i][:, None, :],
                               axis=2)
            # rotate ring j by the offset that best matches ring i
            best, bo = None, 0
            for o in range(sides):
                tot = float(np.sum(d[np.arange(sides),
                                     (np.arange(sides) + o) % sides]))
                if best is None or tot < best:
                    best, bo = tot, o
            shift[j] = (shift[i] + bo) % sides

    faces = []
    for i in range(n):
        j = (i + 1) % n
        for k in range(sides):
            k2 = (k + 1) % sides
            a = i * sides + (k + shift[i]) % sides
            b = i * sides + (k2 + shift[i]) % sides
            c2 = j * sides + (k2 + shift[j]) % sides
            d2 = j * sides + (k + shift[j]) % sides
            faces.append([a, b, c2, d2])
    return verts, faces


def _transport(h0, h1, l0):
    """Carry `l0` from tangent `h0` to `h1` by the minimal rotation.

    This is the whole of parallel transport: rotate about h0 x h1 by the
    angle between them.  When the tangents agree (a straight section) the
    rotation is the identity, which is exactly why this is defined where
    the Frenet frame is not.
    """
    axis = np.cross(h0, h1)
    m = float(np.linalg.norm(axis))
    if m < 1e-12:
        # parallel (straight) or antiparallel (a cusp)
        if float(np.dot(h0, h1)) > 0.0:
            return _ortho(l0, h1)
        return _ortho(-l0, h1)
    axis = axis / m
    ang = math.atan2(m, float(np.dot(h0, h1)))
    c, s = math.cos(ang), math.sin(ang)
    rot = (l0 * c + np.cross(axis, l0) * s +
           axis * float(np.dot(axis, l0)) * (1.0 - c))
    return _ortho(rot, h1)


def _ortho(v, h):
    """Re-orthogonalise `v` against `h` and normalise.  Called every
    step so accumulated float drift never lets the frame shear."""
    v = v - h * float(np.dot(v, h))
    m = float(np.linalg.norm(v))
    if m < 1e-12:
        alt = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(alt, h))) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        v = np.cross(alt, h)
        m = float(np.linalg.norm(v))
    return v / m


def _reframe_frenet(points, H, L, U, closed):
    """Rotate each parallel-transported frame about H so that L lies in
    the osculating plane -- recovering the Frenet frame as the special
    case the 2001 paper describes.

    Where the curvature vanishes the osculating plane is undefined; those
    samples keep their transported frame instead of producing NaNs.
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    Lo, Uo = L.copy(), U.copy()
    for i in range(n):
        if closed:
            a, b = P[(i - 1) % n], P[(i + 1) % n]
        else:
            a, b = P[max(i - 1, 0)], P[min(i + 1, n - 1)]
        second = (b - 2.0 * P[i] + a)
        nrm = second - H[i] * float(np.dot(second, H[i]))
        m = float(np.linalg.norm(nrm))
        if m < 1e-9:
            continue                      # straight / inflection: keep PT
        nrm = nrm / m
        Lo[i] = nrm
        Uo[i] = np.cross(H[i], nrm)
    return Lo, Uo


def sweep(points, profile, mode=PARALLEL, radii=None, up=(0.0, 0.0, 1.0),
          twist=0.0, closed=False):
    """Sweep a 2-D `profile` along `points`, returning (verts, faces).

    `profile` is a list of (x, y) pairs in the frame's (L, U) plane;
    `radii` optionally scales it per sample, which is how a tapered limb
    is produced.
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n < 2:
        return [], []
    H, L, U = frames(P, mode=mode, up=up, twist=twist, closed=closed)
    prof = np.asarray(profile, dtype=float)
    k = len(prof)
    if radii is None:
        radii = np.ones(n)
    radii = np.asarray(radii, dtype=float)

    verts = []
    for i in range(n):
        for (px, py) in prof:
            verts.append(tuple(P[i] + (L[i] * px + U[i] * py) * radii[i]))
    faces = []
    rings = n if closed else n - 1
    for i in range(rings):
        j = (i + 1) % n
        for a in range(k):
            b = (a + 1) % k
            faces.append((i * k + a, i * k + b, j * k + b, j * k + a))
    return verts, faces


def welded_tube(points, radius, sides):
    """`closed_tube` with the nearest-vertex ring weld enabled.

    The form nine generators used to reach for as
    `knot_carpet_generator._tube_welded`.  Kept as a named function rather
    than a default argument so those call sites read the same and cannot
    silently lose the weld.
    """
    return closed_tube(points, radius, sides, weld=True)


def _selftest():
    # --- orthonormality on a hard curve -------------------------------
    t = np.linspace(0, 4 * math.pi, 400)
    helix = np.stack([np.cos(t), np.sin(t), t * 0.15], axis=1)
    for mode in (PARALLEL, FRENET, FIXED_UP):
        H, L, U = frames(helix, mode=mode)
        assert np.allclose(np.linalg.norm(H, axis=1), 1, atol=1e-9), mode
        assert np.allclose(np.linalg.norm(L, axis=1), 1, atol=1e-9), mode
        assert np.allclose(np.linalg.norm(U, axis=1), 1, atol=1e-9), mode
        assert np.allclose((H * L).sum(axis=1), 0, atol=1e-8), mode
        assert np.allclose(np.cross(H, L) - U, 0, atol=1e-8), \
            f"{mode}: H x L must equal U"

    # --- THE POINT: an inflection ------------------------------------
    # A cubic S-curve has zero curvature at its centre, where the Frenet
    # normal is undefined and flips.  Parallel transport must sail
    # through it with a small, bounded change of frame.
    s = np.linspace(-1.0, 1.0, 401)
    scurve = np.stack([s, s ** 3, np.zeros_like(s)], axis=1)
    _H, Lp, _U = frames(scurve, mode=PARALLEL)
    steps = np.linalg.norm(np.diff(Lp, axis=0), axis=1)
    assert steps.max() < 0.15, \
        f"parallel transport jumped by {steps.max():.3f} at the inflection"

    # a straight segment is the other case Frenet cannot do at all
    line = np.stack([np.zeros(50), np.zeros(50), np.linspace(0, 5, 50)],
                    axis=1)
    H, L, U = frames(line, mode=PARALLEL)
    assert np.all(np.isfinite(L)), "a straight run must still be framed"
    assert np.allclose(L, L[0], atol=1e-9), \
        "with no turning there should be no rotation at all"

    # --- FRENET really is the osculating-plane choice -----------------
    # On a circle the Frenet normal points at the centre.
    a = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    circle = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)
    _H, Lf, _U = frames(circle, mode=FRENET, closed=True)
    inward = -circle / np.linalg.norm(circle, axis=1)[:, None]
    dots = np.abs((Lf * inward).sum(axis=1))
    assert dots.mean() > 0.99, dots.mean()

    # --- parallel transport minimises twist versus fixed-up ----------
    Hh, Lh, _ = frames(helix, mode=PARALLEL)
    _, Lu, _ = frames(helix, mode=FIXED_UP)
    def total_turn(Lx, Hx):
        tot = 0.0
        for i in range(1, len(Lx)):
            prev = _transport(Hx[i - 1], Hx[i], Lx[i - 1])
            c = max(-1.0, min(1.0, float(np.dot(prev, Lx[i]))))
            tot += abs(math.acos(c))
        return tot
    assert total_turn(Lh, Hh) < total_turn(Lu, Hh), \
        "PARALLEL must accumulate less roll than FIXED_UP"
    assert total_turn(Lh, Hh) < 1e-6, "PARALLEL should accumulate ~none"

    # --- twist is applied, and is exactly what was asked for ---------
    _H, L0, _U = frames(line, mode=PARALLEL, twist=0.0)
    _H, L90, _U = frames(line, mode=PARALLEL, twist=90.0)
    c = max(-1.0, min(1.0, float(np.dot(L0[-1], L90[-1]))))
    assert abs(math.degrees(math.acos(c)) - 90.0) < 1e-6

    # --- sweep produces a closed, correctly sized tube ---------------
    prof = [(math.cos(x), math.sin(x)) for x in
            np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    verts, faces = sweep(helix, prof, radii=np.full(len(helix), 0.1))
    assert len(verts) == len(helix) * 8
    assert len(faces) == (len(helix) - 1) * 8
    V = np.array(verts)
    assert np.all(np.isfinite(V))
    # every vertex sits 0.1 from its spine sample
    d = np.linalg.norm(V.reshape(len(helix), 8, 3) - helix[:, None, :],
                       axis=2)
    assert abs(d.mean() - 0.1) < 1e-9, d.mean()

    # --- CLOSED CURVES: holonomy must be measured and distributed ----
    # A closed curve generally comes back rotated about its own tangent.
    # A planar circle does not (zero holonomy), so it is the control.
    a = np.linspace(0, 2 * math.pi, 160, endpoint=False)
    circle = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)
    T, N, B = closed_frames(circle)
    assert abs(closure_holonomy(circle, closed_tangents(circle),
                                transport_normals(circle,
                                                  closed_tangents(circle),
                                                  N[0]))) < 1e-6,         "a planar circle must have zero closure holonomy"
    assert np.allclose(np.linalg.norm(N, axis=1), 1, atol=1e-9)
    assert np.allclose((T * N).sum(axis=1), 0, atol=1e-9)
    assert np.allclose(np.cross(T, N) - B, 0, atol=1e-8)

    # a wavy closed ring HAS holonomy, and it must be spread EVENLY -- the
    # regression that matters.  Measuring it with sign*acos instead of
    # atan2 recovers the wrong branch, so the loop does not close and the
    # whole residual lands on the single seam quad.  That shipped in two
    # generators as a ~1.7 radian kink; this asserts no step is an outlier.
    t = np.linspace(0, 2 * math.pi, 160, endpoint=False)
    r = 1.0 + 0.3 * np.cos(6 * t)
    wavy = np.stack([r * np.cos(t), r * np.sin(t), 0.4 * np.sin(6 * t)],
                    axis=1)
    verts, faces = closed_tube(wavy, 0.09, 12)
    assert len(verts) == len(wavy) * 12 and len(faces) == len(wavy) * 12

    R = np.asarray(verts).reshape(len(wavy), 12, 3)
    u = R[:, 0, :] - wavy
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    Tw = closed_tangents(wavy)
    steps = []
    for i in range(len(wavy)):
        j = (i + 1) % len(wavy)
        moved = _transport(Tw[i], Tw[j], u[i])
        d = max(-1.0, min(1.0, float(np.dot(moved, u[j]))))
        steps.append(abs(math.acos(d)))
    steps = np.array(steps)
    assert steps.max() < 5.0 * steps.mean(), (
        f"seam discontinuity: worst ring step {steps.max():.3f} vs mean "
        f"{steps.mean():.4f} -- the closure holonomy was not distributed")

    # watertight, and every vertex exactly one radius from its spine sample
    cnt = {}
    for f in faces:
        for x, y in zip(f, f[1:] + f[:1]):
            k = (x, y) if x < y else (y, x)
            cnt[k] = cnt.get(k, 0) + 1
    assert all(c == 2 for c in cnt.values()), "closed tube must be watertight"
    d = np.linalg.norm(R - wavy[:, None, :], axis=2)
    assert abs(d - 0.09).max() < 1e-9, abs(d - 0.09).max()

    print("curve_frames: closed-curve OK -- planar circle has zero holonomy, "
          "wavy ring's is distributed evenly (no seam outlier), tube "
          "watertight and on-radius")

    print("curve_frames: OK -- orthonormal on helix/circle/line, survives "
          "an inflection, Frenet recovers the osculating plane, parallel "
          "transport minimises roll, twist exact, sweep closes")
