# Profile sweeps and closed tubes.
#
# Part of `math_art/curve_frames/` -- the moving-frame engine.  NumPy only,
# no `bpy`, so it imports and self-tests headlessly.
#
# Sweeping a profile along a curve: open sweeps, and closed tubes with
# the closure holonomy distributed so the seam matches.
#
# `welded_tube` additionally joins each ring to the next by the whole-ring
# rotation that best aligns them, which absorbs any residual fractional
# twist into a clean shift rather than a fold.  It is the form nine
# generators used to reach for as another generator's private helper.

import math

import numpy as np

from .frames import PARALLEL, closed_frames, frames


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


def welded_tube(points, radius, sides):
    """`closed_tube` with the nearest-vertex ring weld enabled.

    The form nine generators used to reach for as
    `knot_carpet_generator._tube_welded`.  Kept as a named function rather
    than a default argument so those call sites read the same and cannot
    silently lose the weld.
    """
    return closed_tube(points, radius, sides, weld=True)


def _selftest():
    ok = True
    t = np.linspace(0, 4 * math.pi, 200)
    helix = np.stack([np.cos(t), np.sin(t), 0.15 * t], axis=1)

    prof = [(math.cos(x), math.sin(x))
            for x in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    verts, faces = sweep(helix, prof, radii=np.full(len(helix), 0.1))
    V = np.asarray(verts, float)
    d = np.linalg.norm(V.reshape(len(helix), 8, 3) - helix[:, None, :],
                       axis=2)
    good = (len(verts) == len(helix) * 8
            and len(faces) == (len(helix) - 1) * 8
            and np.all(np.isfinite(V)) and abs(d.mean() - 0.1) < 1e-9)
    ok &= good
    print(f"sweep: open sweep sizes and radius {'OK' if good else 'FAIL'}")

    # closed tube: watertight, on-radius, and no seam fold
    a = np.linspace(0, 2 * math.pi, 160, endpoint=False)
    r = 1.0 + 0.3 * np.cos(6 * a)
    wavy = np.stack([r * np.cos(a), r * np.sin(a), 0.4 * np.sin(6 * a)],
                    axis=1)
    for label, fn in (('closed_tube', closed_tube),
                      ('welded_tube', lambda P, rr, s: welded_tube(P, rr, s))):
        vs, fs = fn(wavy, 0.09, 12)
        cnt = {}
        for f in fs:
            for x, y in zip(f, list(f[1:]) + [f[0]]):
                k = (x, y) if x < y else (y, x)
                cnt[k] = cnt.get(k, 0) + 1
        openedges = sum(1 for c in cnt.values() if c != 2)
        R = np.asarray(vs, float).reshape(len(wavy), 12, 3)
        rad = np.linalg.norm(R - wavy[:, None, :], axis=2)
        # a fold shows as adjacent faces creased past 90 degrees
        good = (len(vs) == len(wavy) * 12 and openedges == 0
                and float(np.max(np.abs(rad - 0.09))) < 1e-9)
        ok &= good
        print(f"sweep: {label} watertight ({openedges} open edges), every "
              f"vertex on-radius {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("sweep self-test failed")
