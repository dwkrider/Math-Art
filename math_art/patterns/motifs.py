# Motif library and isometry-type colouring.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`), split out of
# the former single-file `pattern_common.py`.  Python + numpy only -- no
# `bpy` -- so the engine imports and self-tests headlessly; the registered
# operators stay in their flat generator modules and import this package.
#
# The motif library the pattern operators stamp into a fundamental
# domain, and the classification that colours each copy by the kind of
# isometry that produced it (identity / rotation / reflection / glide).
# Colouring by isometry type is what makes a wallpaper group legible:
# it shows the group acting, not just the pattern repeating.

from math import cos, sin, pi, hypot, gcd            # noqa: F401
import numpy as np
from .isometry import apply


# Motif library + color classification (shared by the pattern ops)
# --------------------------------------------------------------------

def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _comma():
    """A chiral paisley: a tapering ribbon swept along an arc with a
    round head, so its handedness reveals reflections and glides."""
    t = np.linspace(0.0, 1.0, 22)
    a0, a1 = np.deg2rad(205.0), np.deg2rad(-25.0)
    ang = a0 + (a1 - a0) * t
    R, cx, cy = 0.24, 0.52, 0.50
    spine = np.column_stack([cx + R * np.cos(ang), cy + R * np.sin(ang)])
    half = 0.135 * (1.0 - t) ** 0.7 + 0.004
    d = np.gradient(spine, axis=0)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    left = spine + nrm * half[:, None]
    right = spine - nrm * half[:, None]
    c0, h0 = spine[0], half[0]
    base = np.arctan2(right[0, 1] - c0[1], right[0, 0] - c0[0])
    cap = np.array([[c0[0] + h0 * np.cos(base + np.pi * s),
                     c0[1] + h0 * np.sin(base + np.pi * s)]
                    for s in np.linspace(0.0, 1.0, 9)])
    if np.dot(cap[len(cap) // 2] - c0, -d[0]) < 0:
        cap = np.array([[c0[0] + h0 * np.cos(base - np.pi * s),
                         c0[1] + h0 * np.sin(base - np.pi * s)]
                        for s in np.linspace(0.0, 1.0, 9)])
    return [np.vstack([left, right[::-1], cap])]


MOTIFS = ('ARROW', 'F', 'L', 'COMMA', 'ZIG', 'TRIANGLE')


def motif(kind):
    """One or more asymmetric polygons in the unit cell [0, 1]^2.  All
    are chiral so reflections and glides in the group stay visible."""
    if kind == 'F':
        return [_rect(0.30, 0.18, 0.42, 0.82),
                _rect(0.42, 0.70, 0.72, 0.82),
                _rect(0.42, 0.46, 0.64, 0.58)]
    if kind == 'L':
        return [_rect(0.32, 0.20, 0.48, 0.80),
                _rect(0.48, 0.20, 0.74, 0.36)]
    if kind == 'ARROW':
        return [np.array([[0.22, 0.44], [0.58, 0.44], [0.58, 0.30],
                          [0.82, 0.52], [0.58, 0.66], [0.58, 0.56],
                          [0.22, 0.56]])]
    if kind == 'COMMA':
        return _comma()
    if kind == 'ZIG':
        return [_rect(0.26, 0.66, 0.72, 0.80),
                _rect(0.26, 0.20, 0.72, 0.34),
                _rect(0.40, 0.20, 0.56, 0.80)]
    return [np.array([[0.25, 0.2], [0.78, 0.32], [0.35, 0.8]])]  # TRI


def iso_type(g):
    """Classify a coset isometry: 0 identity, 1 rotation, 2 reflection,
    3 glide."""
    A = g[:2, :2]
    if A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0] > 0.0:
        return 0 if np.allclose(A, np.eye(2), atol=1e-6) else 1
    return 2 if np.allclose(g @ g, np.eye(3), atol=1e-6) else 3


def kind_of(color_by, M, g, cell, gi):
    """Material index for one replicated copy, per color mode."""
    if color_by == 'OP':
        return iso_type(g)
    if color_by == 'HAND':
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        return 0 if det > 0 else 1
    if color_by == 'CELL':
        return cell % len(PALETTE_RGBA)      # wrap: avoid a material per cell
    return gi                                            # COPY


# a 12-way categorical palette (shared by every generator's coloring);
# the Blender PALETTE below aliases this so there is one source of truth
PALETTE_RGBA = [(0.85, 0.30, 0.24, 1.0), (0.20, 0.45, 0.70, 1.0),
                (0.95, 0.77, 0.30, 1.0), (0.35, 0.65, 0.38, 1.0),
                (0.60, 0.35, 0.68, 1.0), (0.90, 0.55, 0.22, 1.0),
                (0.30, 0.72, 0.72, 1.0), (0.82, 0.47, 0.60, 1.0),
                (0.52, 0.55, 0.25, 1.0), (0.45, 0.40, 0.75, 1.0),
                (0.72, 0.60, 0.45, 1.0), (0.42, 0.42, 0.47, 1.0)]


def _selftest():
    ok = True
    # every motif is a non-empty set of finite 2D polylines
    bad = []
    for k in MOTIFS:
        strokes = motif(k)
        if not strokes:
            bad.append(f"{k}:empty")
            continue
        for s in strokes:
            a = np.asarray(s, float)
            if a.ndim != 2 or a.shape[1] != 2 or not np.all(np.isfinite(a)):
                bad.append(f"{k}:shape{a.shape}")
    good = not bad
    ok &= good
    print(f"motifs: {len(MOTIFS)} motifs are finite 2D polylines "
          f"{'OK' if good else 'FAIL ' + ','.join(bad[:3])}")

    # A motif must be CHIRAL for a wallpaper pattern to be legible -- if it
    # had a mirror symmetry you could not see reflections acting.  Check
    # that no motif matches its own x-reflection as a point set.
    bad = []
    for k in MOTIFS:
        P = np.vstack([np.asarray(s, float) for s in motif(k)])
        M = P.copy()
        M[:, 0] *= -1.0
        a = np.round(np.sort(P, axis=0), 9)
        b = np.round(np.sort(M, axis=0), 9)
        if P.shape == M.shape and np.allclose(a, b):
            bad.append(k)
    print(f"motifs: chirality check -- mirror-symmetric motifs: "
          f"{bad if bad else 'none'} (informational)")

    # iso_type classifies the four kinds, and kind_of indexes the palette
    from .isometry import Glide, I, Mir, Rot
    got = {iso_type(I()), iso_type(Rot(0.7)), iso_type(Mir(0.3)),
           iso_type(Glide(0.3, 0.5))}
    good = len(got) >= 3
    ok &= good
    print(f"motifs: iso_type distinguishes {len(got)} kinds {sorted(got)} "
          f"{'OK' if good else 'FAIL'}")

    good = len(PALETTE_RGBA) >= 4 and all(
        len(c) == 4 and all(0.0 <= v <= 1.0 for v in c)
        for c in PALETTE_RGBA)
    ok &= good
    print(f"motifs: palette has {len(PALETTE_RGBA)} valid RGBA entries "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("motifs self-test failed")
