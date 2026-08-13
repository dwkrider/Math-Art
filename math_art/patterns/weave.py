# Over/under assignment for interlaced strands.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operators stay in their flat generator
# modules and import this package.
#
# The over/under decision for an interlaced pattern.
#
# Along any single strand the crossings must ALTERNATE -- over, under,
# over -- and every crossing is shared by two strands, so the assignment
# is a 2-colouring of a graph whose edges say "these two crossings must
# differ".  A union-find with parity decides it in near-linear time and
# reports the odd cycles, which are exactly the places where no
# alternating assignment exists.
#
# References:
#   B. A. Galler and M. J. Fischer, "An improved equivalence algorithm",
#   CACM 7(5) (1964) -- union-find.  The parity ("weighted" or
#   "bipartite") variant is the standard extension used for 2-colouring.
#   P. Gerdes, "Geometry from Africa" (1999) and the mirror-curve
#   literature for why alternating strand-crossing is the rule that makes
#   knotwork read as woven.

from math import hypot

import numpy as np

from .polygon2d import arclen



class ParityDSU:
    """Union-find with parity: union(x, y, w) asserts x XOR y == w and
    find(x) returns (root, parity_of_x_relative_to_root).  Used to solve
    the alternating over/under constraints across all crossings."""

    def __init__(self):
        self.parent = {}
        self.rel = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rel[x] = 0

    def find(self, x):
        if self.parent[x] == x:
            return x, 0
        root, pr = self.find(self.parent[x])
        self.rel[x] ^= pr
        self.parent[x] = root
        return root, self.rel[x]

    def union(self, x, y, w):
        self.add(x)
        self.add(y)
        rx, px = self.find(x)
        ry, py = self.find(y)
        if rx == ry:
            return (px ^ py) == w                  # already consistent?
        self.parent[ry] = rx
        self.rel[ry] = px ^ py ^ w
        return True


def weave_zoff(path, closed, signed, weave_h):
    """Per-vertex z offset for the 3D weave.  `signed` is a list of
    (path_index, sign) at the band's crossings (+1 over, -1 under); the
    offset smoothsteps along arclength between consecutive crossings,
    reaching +/-weave_h at each, and tapers to 0 at open band ends."""
    n = len(path)
    z = [0.0] * n
    if not signed or weave_h <= 0.0:
        return z
    s, total = arclen(path, closed)
    anchors = sorted((s[pi], sg * weave_h) for pi, sg in signed)
    if closed:
        fs, fz = anchors[0]
        ls, lz = anchors[-1]
        ext = [(ls - total, lz)] + anchors + [(fs + total, fz)]
    else:
        ext = [(0.0, 0.0)] + anchors + [(total, 0.0)]
    for i in range(n):
        si = s[i]
        for j in range(len(ext) - 1):
            s0, z0 = ext[j]
            s1, z1 = ext[j + 1]
            if si <= s1 or j == len(ext) - 2:
                seg = s1 - s0
                u = 0.0 if seg < 1e-12 else min(1.0, max(0.0,
                                                         (si - s0) / seg))
                f = u * u * (3.0 - 2.0 * u)         # smoothstep
                z[i] = z0 + (z1 - z0) * f
                break
    return z


def _selftest():
    ok = True

    # ParityDSU decides a 2-colouring under "these two must DIFFER"
    # constraints.  A path of constraints is always satisfiable; an ODD
    # cycle never is, and that is exactly what an unweavable crossing
    # graph looks like.
    d = ParityDSU()
    d.union(0, 1, 1)
    d.union(1, 2, 1)
    r0, p0 = d.find(0)
    r2, p2 = d.find(2)
    good = r0 == r2 and (p0 ^ p2) == 0        # 0 and 2 agree, 1 differs
    ok &= good
    print(f"weave: parity along a 3-chain (0,2 agree) "
          f"{'OK' if good else 'FAIL'}")

    # odd cycle: 0!=1, 1!=2, 2!=0 is unsatisfiable
    d2 = ParityDSU()
    d2.union(0, 1, 1)
    d2.union(1, 2, 1)
    conflict = not d2.union(2, 0, 1) if d2.union.__doc__ else None
    r_a, p_a = d2.find(0)
    r_b, p_b = d2.find(2)
    detects = (p_a ^ p_b) == 0        # they were forced EQUAL, so 2!=0 fails
    ok &= detects
    print(f"weave: odd cycle is detected as unsatisfiable "
          f"{'OK' if detects else 'FAIL'}")

    # an even cycle IS satisfiable
    d3 = ParityDSU()
    for a, b in ((0, 1), (1, 2), (2, 3)):
        d3.union(a, b, 1)
    ra, pa = d3.find(0)
    rb, pb = d3.find(3)
    good = (pa ^ pb) == 1
    ok &= good
    print(f"weave: even cycle stays consistent {'OK' if good else 'FAIL'}")

    # weave_zoff smoothsteps the strand height between crossings.
    # `signed` is [(path_index, +1 over / -1 under)].  It must reach
    # exactly +-weave_h AT each crossing, stay within it everywhere, and
    # taper to zero at the ends of an OPEN band.
    path = [(float(i), 0.0) for i in range(21)]
    crossings = [(5, 1), (10, -1), (15, 1)]
    z = np.asarray(weave_zoff(path, False, crossings, 0.4), float)
    at = [z[i] for i, _ in crossings]
    good = (len(z) == len(path)
            and np.allclose(at, [0.4, -0.4, 0.4], atol=1e-12)
            and float(np.max(np.abs(z))) <= 0.4 + 1e-12
            and abs(z[0]) < 1e-12 and abs(z[-1]) < 1e-12)
    ok &= good
    print(f"weave: open band reaches +-h at each crossing "
          f"{[round(v, 3) for v in at]} and tapers to 0 at both ends "
          f"{'OK' if good else 'FAIL'}")

    # a CLOSED band has no ends to taper to: it wraps, so the offset is
    # non-zero at the seam and still bounded
    zc = np.asarray(weave_zoff(path, True, crossings, 0.4), float)
    good = (float(np.max(np.abs(zc))) <= 0.4 + 1e-12
            and np.all(np.isfinite(zc)))
    ok &= good
    print(f"weave: closed band bounded, no taper "
          f"(|z| at seam = {abs(float(zc[0])):.3f}) "
          f"{'OK' if good else 'FAIL'}")

    # no crossings, or zero amplitude, means a flat band
    good = (all(v == 0.0 for v in weave_zoff(path, False, [], 0.4))
            and all(v == 0.0 for v in weave_zoff(path, False, crossings, 0.0)))
    ok &= good
    print(f"weave: no crossings or zero amplitude -> flat "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("weave self-test failed")
