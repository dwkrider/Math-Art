# Mode-A relief geometry and cells.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`), split out of
# the former single-file `pattern_common.py`.  Python + numpy only -- no
# `bpy` -- so the engine imports and self-tests headlessly; the registered
# operators stay in their flat generator modules and import this package.
#
# Mode-A relief: turn 2D polygons and line segments into watertight 3D
# prisms on a backing slab, and centre/scale the result into the 2 m cube
# the rest of the extension uses.  Also the per-cell machinery, which
# keeps each unit of a pattern as its own piece so it can be emitted
# separately or merged.

from math import cos, sin, pi, hypot, gcd            # noqa: F401
import numpy as np


# Mode A -- relief geometry (2D polygons/segments -> 3D prisms)
# --------------------------------------------------------------------

def ribbon_polys(segments, width):
    """Turn line segments into rectangular ribbon polygons of the
    given width (strapwork).  Overlaps at joints are intentional --
    the shells union cleanly for print or render."""
    w = 0.5 * width
    polys = []
    for (x0, y0), (x1, y1) in segments:
        dx, dy = x1 - x0, y1 - y0
        L = hypot(dx, dy)
        if L < 1e-9:
            continue
        nx_, ny_ = -dy / L * w, dx / L * w
        polys.append(np.array([[x0 + nx_, y0 + ny_],
                               [x1 + nx_, y1 + ny_],
                               [x1 - nx_, y1 - ny_],
                               [x0 - nx_, y0 - ny_]]))
    return polys


def prisms(verts, faces, mats, polys2d, z_top, z_bot, mat=0):
    """Append watertight prisms (one per 2D polygon) between z_bot and
    z_top.  Top n-gon, reversed bottom n-gon, side quads."""
    for poly in polys2d:
        n = len(poly)
        if n < 3:
            continue
        b0 = len(verts)
        for x, y in poly:
            verts.append((float(x), float(y), z_top))
        for x, y in poly:
            verts.append((float(x), float(y), z_bot))
        faces.append(tuple(b0 + k for k in range(n)))
        faces.append(tuple(b0 + n + k for k in reversed(range(n))))
        for k in range(n):
            k2 = (k + 1) % n
            faces.append((b0 + k, b0 + n + k, b0 + n + k2, b0 + k2))
        mats.extend([mat] * (n + 2))


def slab(verts, faces, mats, lo, hi, z_top, z_bot, mat=1):
    """A single box backing slab spanning the xy rectangle lo..hi."""
    x0, y0 = lo
    x1, y1 = hi
    prisms(verts, faces, mats,
           [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])],
           z_top, z_bot, mat)


def center_scale(verts, span=2.0):
    """Centre a vertex list at the origin and scale so its larger in-
    plane extent spans `span` (the 2 m cube convention)."""
    if not verts:
        return verts
    a = np.asarray(verts, dtype=float)
    lo, hi = a.min(axis=0), a.max(axis=0)
    c = 0.5 * (lo + hi)
    ext = max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
    s = span / ext
    a = (a - c) * s
    return [tuple(v) for v in a]


def center_xy(verts):
    """Centre a vertex list on the origin in X and Y (leaving Z), so
    the object's origin sits at the pattern centre and the object's
    transform can then place/orient it anywhere."""
    if not verts:
        return verts
    a = np.asarray(verts, dtype=float)
    cx = 0.5 * (a[:, 0].min() + a[:, 0].max())
    cy = 0.5 * (a[:, 1].min() + a[:, 1].max())
    a[:, 0] -= cx
    a[:, 1] -= cy
    return [tuple(v) for v in a]


# --------------------------------------------------------------------
# Cells: per-unit pieces, merged into one mesh or emitted separately
# --------------------------------------------------------------------
#
# Every pattern generator produces a list of "cells": one
# (verts, faces, mats) piece per replicated unit (motif copy or tile).
# merge_cells() concatenates them into a single mesh; emit() (Blender)
# either merges or builds one child object per cell so each can be
# edited individually.

def merge_cells(cells):
    """Concatenate per-cell (verts, faces, mats) pieces into a single
    (verts, faces, mats), offsetting the face indices of each cell."""
    verts, faces, mats = [], [], []
    for cv, cf, cm in cells:
        b0 = len(verts)
        verts.extend(cv)
        for f in cf:
            faces.append(tuple(b0 + idx for idx in f))
        mats.extend(cm)
    return verts, faces, mats


def _global_transform(verts, fit, span):
    """(cx, cy, cz, s) mapping the pattern into place: with fit it is
    centred and scaled to `span` (matching center_scale); without fit
    it is only centred in XY (matching center_xy)."""
    a = np.asarray(verts, dtype=float)
    lo, hi = a.min(axis=0), a.max(axis=0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    if fit:
        cz = 0.5 * (lo[2] + hi[2])
        s = span / max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
    else:
        cz, s = 0.0, 1.0
    return cx, cy, cz, s


def _apply_transform(verts, p):
    cx, cy, cz, s = p
    return [((x - cx) * s, (y - cy) * s, (z - cz) * s)
            for x, y, z in verts]


def _selftest():
    ok = True
    # prisms: a unit square extruded between z_bot and z_top must give a
    # closed box -- 8 vertices, and every edge shared by exactly two faces.
    v, f, m = [], [], []
    prisms(v, f, m, [[(0, 0), (1, 0), (1, 1), (0, 1)]], 0.3, -0.1)
    V = np.asarray(v, float)
    cnt = {}
    for face in f:
        for a, b in zip(face, list(face[1:]) + [face[0]]):
            k = (a, b) if a < b else (b, a)
            cnt[k] = cnt.get(k, 0) + 1
    good = (len(V) == 8 and np.all(np.isfinite(V))
            and all(c == 2 for c in cnt.values())
            and abs(V[:, 2].max() - 0.3) < 1e-12
            and abs(V[:, 2].min() + 0.1) < 1e-12)
    ok &= good
    print(f"relief: prism of a square is a closed box V={len(V)} F={len(f)} "
          f"z=[{V[:, 2].min():.2f},{V[:, 2].max():.2f}] "
          f"{'OK' if good else 'FAIL'}")

    # slab: likewise a closed box spanning the given rectangle
    v2, f2, m2 = [], [], []
    slab(v2, f2, m2, (-1.0, -1.0), (1.0, 1.0), 0.2, -0.2)
    V2 = np.asarray(v2, float)
    good = (len(V2) == 8 and np.all(np.isfinite(V2))
            and abs(V2[:, 0].min() + 1.0) < 1e-12
            and abs(V2[:, 0].max() - 1.0) < 1e-12)
    ok &= good
    print(f"relief: slab spans its rectangle V={len(V2)} F={len(f2)} "
          f"{'OK' if good else 'FAIL'}")

    # ribbon_polys turns segments into quads of the requested WIDTH:
    # a unit-length segment widened by w has area w (to the mitre error)
    from .polygon2d import signed_area
    polys = ribbon_polys([((0.0, 0.0), (1.0, 0.0))], 0.2)
    areas = [abs(signed_area(np.asarray(p, float))) for p in polys]
    good = polys and abs(sum(areas) - 0.2) < 1e-9
    ok &= good
    print(f"relief: ribbon of a unit segment has area {sum(areas):.4f} "
          f"(exp 0.2) {'OK' if good else 'FAIL'}")

    # center_scale implements the project convention: centred on the
    # origin, largest extent exactly `span`
    pts = [(3.0, 3.0, 3.0), (7.0, 5.0, 4.0), (5.0, 4.0, 3.5)]
    out = np.asarray(center_scale(pts, 2.0), float)
    lo, hi = out.min(axis=0), out.max(axis=0)
    cen = float(np.max(np.abs(0.5 * (lo + hi))))
    ext = float(np.max(hi - lo))
    good = cen < 1e-12 and abs(ext - 2.0) < 1e-12
    ok &= good
    print(f"relief: center_scale |c|={cen:.1e} ext={ext:.6f} "
          f"{'OK' if good else 'FAIL'}")

    # merge_cells concatenates and reindexes: face indices must stay in range
    cells = [(list(v), list(f), list(m)), (list(v2), list(f2), list(m2))]
    mv, mf, mm = merge_cells(cells)
    good = (len(mv) == len(v) + len(v2)
            and all(0 <= i < len(mv) for face in mf for i in face))
    ok &= good
    print(f"relief: merge_cells V={len(mv)} F={len(mf)}, indices in range "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("relief self-test failed")
