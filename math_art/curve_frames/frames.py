# Moving frames, open and closed.
#
# Part of `math_art/curve_frames/` -- the moving-frame engine.  NumPy only,
# no `bpy`, so it imports and self-tests headlessly.
#
# Full (heading, left, up) frames, and the closed-curve case.
#
# THE FRENET FRAME IS NOT USABLE HERE.  Built from the curve's own
# derivatives, it is undefined wherever curvature vanishes -- any straight
# run -- and flips through 180 degrees at every inflection.  Parallel
# transport is defined everywhere and is the default.
#
# CLOSED CURVES CARRY HOLONOMY.  Transport a frame once round a loop and
# it returns rotated about the tangent.  Carried naively that entire
# residual lands on the seam, folding a swept tube through itself;
# `closed_frames` measures it and spreads it evenly.
#
# References:
#   Przemyslaw Prusinkiewicz, Lars Mundermann, Radoslaw Karwowski and
#     Brendan Lane, "The use of positional information in the modeling of
#     plants", SIGGRAPH 2001 -- differential turtle geometry, the
#     formulation this construction comes from and which gave the module
#     its former name.
#   Andrew J. Hanson and Hui Ma, "Parallel transport approach to curve
#     framing", Indiana University TR-425 (1995).

import math

import numpy as np

from .kernels import _ortho, _transport, _unit, transport_normals
from .tangents import closed_tangents, tangents

#: the three ways to spend the frame's one free rotational degree of
#: freedom: none (rotation-minimizing), the osculating plane, or a
#: fixed world up-vector
PARALLEL, FRENET, FIXED_UP = 'PARALLEL', 'FRENET', 'FIXED_UP'


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


def _selftest():
    ok = True
    t = np.linspace(0, 4 * math.pi, 400)
    helix = np.stack([np.cos(t), np.sin(t), t * 0.15], axis=1)

    # orthonormality in all three modes
    bad = []
    for mode in (PARALLEL, FRENET, FIXED_UP):
        H, L, U = frames(helix, mode=mode)
        if not (np.allclose(np.linalg.norm(H, axis=1), 1, atol=1e-9)
                and np.allclose(np.linalg.norm(L, axis=1), 1, atol=1e-9)
                and np.allclose((H * L).sum(axis=1), 0, atol=1e-8)
                and np.allclose(np.cross(H, L) - U, 0, atol=1e-8)):
            bad.append(mode)
    good = not bad
    ok &= good
    print(f"frames: orthonormal, H x L = U, in all three modes "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # THE POINT: an inflection, where the Frenet normal is undefined and
    # flips.  Parallel transport must sail through with a bounded change.
    s = np.linspace(-1.0, 1.0, 401)
    scurve = np.stack([s, s ** 3, np.zeros_like(s)], axis=1)
    _H, Lp, _U = frames(scurve, mode=PARALLEL)
    step = float(np.max(np.linalg.norm(np.diff(Lp, axis=0), axis=1)))
    good = step < 0.15
    ok &= good
    print(f"frames: survives an inflection (worst step {step:.3f}) "
          f"{'OK' if good else 'FAIL'}")

    # a straight run is the other case Frenet cannot do at all
    line = np.stack([np.zeros(50), np.zeros(50), np.linspace(0, 5, 50)],
                    axis=1)
    _H, Ll, _U = frames(line, mode=PARALLEL)
    good = np.all(np.isfinite(Ll)) and np.allclose(Ll, Ll[0], atol=1e-9)
    ok &= good
    print(f"frames: a straight run is framed, with no rotation at all "
          f"{'OK' if good else 'FAIL'}")

    # FRENET really is the osculating-plane choice: on a circle its normal
    # points at the centre
    a = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    circle = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)
    _H, Lf, _U = frames(circle, mode=FRENET, closed=True)
    inward = -circle / np.linalg.norm(circle, axis=1)[:, None]
    good = float(np.mean(np.abs((Lf * inward).sum(axis=1)))) > 0.99
    ok &= good
    print(f"frames: FRENET recovers the osculating plane on a circle "
          f"{'OK' if good else 'FAIL'}")

    # twist is exactly what was asked for
    _H, L0, _U = frames(line, mode=PARALLEL, twist=0.0)
    _H, L90, _U = frames(line, mode=PARALLEL, twist=90.0)
    c = max(-1.0, min(1.0, float(np.dot(L0[-1], L90[-1]))))
    good = abs(math.degrees(math.acos(c)) - 90.0) < 1e-6
    ok &= good
    print(f"frames: a 90 degree twist rotates the frame by 90 degrees "
          f"{'OK' if good else 'FAIL'}")

    # CLOSED CURVES.  A planar circle has ZERO holonomy -- the control --
    # while a wavy ring has real holonomy that must be spread evenly.
    T, N, B = closed_frames(circle)
    hol = closure_holonomy(circle, closed_tangents(circle),
                           transport_normals(circle, closed_tangents(circle),
                                             N[0]))
    good = abs(hol) < 1e-6
    ok &= good
    print(f"frames: a planar circle has zero closure holonomy "
          f"({hol:.1e}) {'OK' if good else 'FAIL'}")

    tt = np.linspace(0, 2 * math.pi, 160, endpoint=False)
    r = 1.0 + 0.3 * np.cos(6 * tt)
    wavy = np.stack([r * np.cos(tt), r * np.sin(tt), 0.4 * np.sin(6 * tt)],
                    axis=1)
    Tw, Nw, Bw = closed_frames(wavy)
    steps = []
    for i in range(len(wavy)):
        j = (i + 1) % len(wavy)
        prev = _transport(Tw[i], Tw[j], Nw[i])
        c = max(-1.0, min(1.0, float(np.dot(prev, Nw[j]))))
        steps.append(abs(math.acos(c)))
    steps = np.asarray(steps)
    good = float(steps.max()) < 5.0 * float(np.median(steps))
    ok &= good
    print(f"frames: a wavy ring's holonomy is spread, no seam outlier "
          f"(worst {math.degrees(steps.max()):.2f} deg vs median "
          f"{math.degrees(np.median(steps)):.2f}) {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("frames self-test failed")
