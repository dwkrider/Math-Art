# The placed-tiles interchange format.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`), split out of
# the former single-file `pattern_common.py`.  Python + numpy only -- no
# `bpy` -- so the engine imports and self-tests headlessly; the registered
# operators stay in their flat generator modules and import this package.
#
# The placed-tiles interchange format.
#
# Every tiling backend -- periodic, substitution, hyperbolic -- emits the
# SAME structure: prototiles plus a list of placements (which prototile,
# and the 2D transform putting it there).  The relief and output layers
# consume only that, so any backend composes with any cell mode instead
# of the two multiplying into one-off generators.

from math import cos, sin, pi, hypot, gcd            # noqa: F401
import numpy as np
from .isometry import apply


# Placed-tiles interchange format
# --------------------------------------------------------------------

class Tiling:
    """The common output of every tiling backend.

    * polys    -- list of (M, 2) arrays: filled tile polygons
    * segments -- list of ((x0, y0), (x1, y1)) motif line segments
    * types    -- parallel to polys: an int per tile for coloring
    """

    def __init__(self):
        self.polys = []
        self.segments = []
        self.types = []

    def add_poly(self, pts, kind=0):
        self.polys.append(np.asarray(pts, dtype=float))
        self.types.append(int(kind))

    def add_segment(self, a, b):
        self.segments.append((tuple(a), tuple(b)))

    def replicate(self, isometries):
        """Return a new Tiling with polys and segments imaged under
        every transform in `isometries`."""
        out = Tiling()
        for M in isometries:
            for p, k in zip(self.polys, self.types):
                out.add_poly(apply(M, p), k)
            for a, b in self.segments:
                q = apply(M, [a, b])
                out.add_segment(q[0], q[1])
        return out

    def bbox(self):
        pts = []
        pts += [p for p in self.polys]
        if self.segments:
            pts.append(np.array([c for s in self.segments for c in s]))
        if not pts:
            return np.zeros(2), np.zeros(2)
        allp = np.vstack(pts)
        return allp.min(axis=0), allp.max(axis=0)

    def clip(self, lo, hi, margin=0.51):
        """Keep polys/segments whose centroid lies in [lo, hi]
        (padded by `margin` cells so partial tiles at the frame stay)."""
        out = Tiling()
        lo = np.asarray(lo) - margin
        hi = np.asarray(hi) + margin
        for p, k in zip(self.polys, self.types):
            c = p.mean(axis=0)
            if np.all(c >= lo) and np.all(c <= hi):
                out.add_poly(p, k)
        for a, b in self.segments:
            c = 0.5 * (np.array(a) + np.array(b))
            if np.all(c >= lo) and np.all(c <= hi):
                out.add_segment(a, b)
        return out


def _selftest():
    ok = True
    from .isometry import I, Rot, T

    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    t = Tiling()
    t.add_poly(sq, 0)
    t.add_poly([(2.0, 2.0), (3.0, 2.0), (2.5, 3.0)], 1)
    t.add_segment((0.0, 0.0), (1.0, 1.0))
    good = (len(t.polys) == 2 and len(t.types) == 2
            and len(t.segments) == 1 and t.types == [0, 1]
            and t.polys[0].shape == (4, 2))
    ok &= good
    print(f"placed: add_poly/add_segment keep tiles, kinds and segments "
          f"{'OK' if good else 'FAIL'}")

    # bbox must cover every vertex
    lo, hi = t.bbox()
    allpts = np.vstack(t.polys)
    good = (np.all(allpts >= np.asarray(lo) - 1e-12)
            and np.all(allpts <= np.asarray(hi) + 1e-12))
    ok &= good
    print(f"placed: bbox {np.round(lo,3)}..{np.round(hi,3)} covers all tiles "
          f"{'OK' if good else 'FAIL'}")

    # replicate under the identity is a copy; under k isometries it
    # multiplies the counts by k and preserves the kind tags
    one = t.replicate([I()])
    good = (len(one.polys) == len(t.polys)
            and np.allclose(one.polys[0], t.polys[0])
            and one.types == t.types)
    four = t.replicate([I(), T(5, 0), T(0, 5), Rot(np.pi / 2, 9, 9)])
    good = good and (len(four.polys) == 4 * len(t.polys)
                     and len(four.segments) == 4 * len(t.segments)
                     and sorted(four.types) == sorted(t.types * 4))
    ok &= good
    print(f"placed: replicate x1 is a copy, x4 multiplies "
          f"({len(four.polys)} polys, {len(four.segments)} segs) "
          f"{'OK' if good else 'FAIL'}")

    # replicate must be an ISOMETRY of the whole tiling: a rigid motion
    # cannot change any tile's area
    from .polygon2d import signed_area
    a0 = sorted(round(abs(signed_area(p)), 9) for p in t.polys)
    a1 = sorted(round(abs(signed_area(p)), 9) for p in four.polys)
    good = a1 == sorted(a0 * 4)
    ok &= good
    print(f"placed: replication preserves every tile's area "
          f"{'OK' if good else 'FAIL'}")

    # clip keeps what is inside and drops what is far outside
    near = t.replicate([I(), T(50.0, 50.0)]).clip((-1.0, -1.0), (4.0, 4.0))
    good = len(near.polys) == len(t.polys)
    ok &= good
    print(f"placed: clip kept {len(near.polys)} of "
          f"{2 * len(t.polys)} (far copies dropped) "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("placed self-test failed")
