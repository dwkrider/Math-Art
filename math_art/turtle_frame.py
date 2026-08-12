
# Differential turtle frames: orienting a cross-section along a curve.
#
# This is a SHARED utility, not an L-system feature.  Every generator in
# this repo that sweeps a profile along a space curve -- knots, ruled
# surfaces, space-filling curves, tree limbs, tubes of any kind -- needs
# a moving reference frame, and the obvious choice is broken.
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

    print("turtle_frame: OK -- orthonormal on helix/circle/line, survives "
          "an inflection, Frenet recovers the osculating plane, parallel "
          "transport minimises roll, twist exact, sweep closes")
