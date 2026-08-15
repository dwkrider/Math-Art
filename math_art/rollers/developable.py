# The oloid and the sphericon: developable rollers.
#
# Part of the Math Art rollers engine (`math_art/rollers/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# The oloid is the convex hull of two perpendicular circles of equal
# radius, each passing through the other's centre.  Its surface is
# developable -- it can be unrolled flat -- and it rolls in a wobbling
# path that touches every point of its surface, with the centre of mass
# staying at constant height.
#
# References:
# - P. Schatz, "Rhythmusforschung und Technik", 1975 -- the oloid.
# - C. J. Roberts, the sphericon, described in I. Stewart, "Cone with a
#   Twist", Scientific American, October 1999.
# - H. Dirnbock and H. Stachel, "The Development of the Oloid", Journal
#   for Geometry and Graphics 1, 1997, pp. 105-118.

import math
from math import cos, sin, pi, sqrt, radians


def _vkey(p):
    return (round(p[0], 9), round(p[1], 9), round(p[2], 9))


class _Mesh:
    def __init__(self):
        self.verts = []
        self.faces = []
        self._ids = {}

    def vid(self, p):
        k = _vkey(p)
        i = self._ids.get(k)
        if i is None:
            i = len(self.verts)
            self._ids[k] = i
            self.verts.append(p)
        return i

    def quad(self, a, b, c, d):
        ids = [self.vid(p) for p in (a, b, c, d)]
        uniq = []
        for i in ids:
            if i not in uniq:
                uniq.append(i)
        if len(uniq) >= 3:
            self.faces.append(uniq)


def build_oloid(segments=96, scale=1.0):
    """Exact oloid surface from the Dirnboeck-Stachel ruling; two
    mirror strips welded along the circle arcs. Watertight."""
    m = _Mesh()
    T = 2 * pi / 3
    n = segments

    def A(t):
        return (sin(t) * scale, (-0.5 - cos(t)) * scale, 0.0)

    def B(t, sign):
        c = cos(t)
        z2 = max(0.0, 1.0 + 2.0 * c)
        z = sqrt(z2) / (1.0 + c)
        if z < 1e-6:                     # end rulings: weld exactly
            z = 0.0
        return (0.0, (0.5 - c / (1.0 + c)) * scale,
                sign * z * scale)

    for i in range(n):
        t0 = -T + 2 * T * i / n
        t1 = -T + 2 * T * (i + 1) / n
        m.quad(A(t0), A(t1), B(t1, 1.0), B(t0, 1.0))
        m.quad(A(t1), A(t0), B(t0, -1.0), B(t1, -1.0))
    return m.verts, m.faces


def roller_circles(segments=96, scale=1.0, aspect=1.0):
    """Point cloud of the two-circle roller / wobbler: two circles in
    perpendicular planes, centres sqrt(2) apart (hulled in Blender).
    `aspect` != 1 stretches each disc along the separation axis into an
    ellipse -- David Hirsch's "ellipsoloid" variant."""
    d = sqrt(2.0) / 2.0
    pts = []
    for i in range(segments):
        a = 2 * pi * i / segments
        pts.append((cos(a) * scale, (-d + sin(a) * aspect) * scale, 0.0))
        pts.append((0.0, (d + cos(a) * aspect) * scale, sin(a) * scale))
    return pts


def build_ruled(segments=96, separation=1.0, incline=0.0, phase=0.0,
                scale=1.0):
    """Kit Wallace's ruled strip: straight lines from circle 1 (xy
    plane, centred at the origin) to circle 2 (xz plane, offset along
    x, optionally inclined), sample i to sample i+phase."""
    m = _Mesh()
    n = segments
    inc = radians(incline)
    ci, si = cos(inc), sin(inc)

    def c1(i):
        a = 2 * pi * i / n
        return (cos(a) * scale, sin(a) * scale, 0.0)

    def c2(i):
        a = 2 * pi * (i / n + phase)
        x, y, z = cos(a), 0.0, sin(a)
        # incline about the z axis, then offset along x
        x, y = x * ci - y * si, x * si + y * ci
        return ((x + separation) * scale, y * scale, z * scale)

    for i in range(n):
        m.quad(c1(i), c1(i + 1), c2(i + 1), c2(i))
    return m.verts, m.faces


def build_antioloid(segments=128, phase=0.0, scale=1.0):
    """The anti-oloid (cf. the Matter Collection piece): the ruled
    band between the same two circles as the oloid -- perpendicular
    planes, each centre on the other's circumference -- but with
    rulings connecting points travelling around the FULL circles in
    step, sweeping through the interior. Unlike a Mobius strip it is
    two-sided, and unlike the oloid it is not developable."""
    m = _Mesh()
    n = segments

    def A(t):
        return (sin(t) * scale, (-0.5 - cos(t)) * scale, 0.0)

    def B(t):
        return (0.0, (0.5 + cos(t)) * scale, sin(t) * scale)

    # the canonical anti-oloid pairs each point with the half-turn
    # opposite point on the other circle (phase 0 here); other phases
    # give crossed variants
    p = 2 * pi * phase + pi
    for i in range(n):
        t0 = 2 * pi * i / n
        t1 = 2 * pi * (i + 1) / n
        m.quad(A(t0), A(t1), B(t1 + p), B(t0 + p))
    return m.verts, m.faces


def build_mobius(segments=192, scale=1.0):
    """Kit Wallace's true one-edged ruled Mobius strip: rulings
    f(x) -> f(x + 1/2) along a double-loop edge curve."""
    m = _Mesh()

    def f(x):
        r = 1.0 - 0.15 * sin(2 * pi * x + radians(30))
        return (r * cos(4 * pi * x) * scale,
                r * sin(4 * pi * x) * scale,
                0.2 * cos(2 * pi * x) * scale)

    n = segments

    def fx(i):
        return f((i % (2 * n)) / (2 * n))

    for i in range(n):
        m.quad(fx(i), fx(i + 1), fx(i + 1 + n), fx(i + n))
    return m.verts, m.faces
