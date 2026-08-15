# Bodies of constant width.
#
# Part of the Math Art rollers engine (`math_art/rollers/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# A convex body has constant width w if every pair of parallel
# supporting planes is w apart -- it rolls like a sphere without being
# one.  The Reuleaux polygons are the planar case; Meissner's two
# tetrahedra are the three-dimensional bodies of least volume for their
# width, still only conjecturally so.
#
# References:
# - F. Reuleaux, "The Kinematics of Machinery", 1876.
# - E. Meissner and F. Schilling, "Drei Gipsmodelle von Flachen
#   konstanter Breite", Zeitschrift fur Mathematik und Physik 60, 1912.
# - T. Bayen, T. Lachand-Robert and E. Oudet, "Analytic parametrization
#   of three-dimensional bodies of constant width", Archive for Rational
#   Mechanics and Analysis 186, 2007.

import math
from math import sin, cos, pi, sqrt


# ---------------------------------------------------------------------
# small mesh helper (welds coincident vertices, e.g. the poles / seams)
# ---------------------------------------------------------------------
def _vkey(p):
    return (round(p[0], 6), round(p[1], 6), round(p[2], 6))


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

    def face(self, pts):
        ids = []
        for p in pts:
            i = self.vid(p)
            if i not in ids:
                ids.append(i)
        if len(ids) >= 3:
            self.faces.append(ids)


def _revolve(profile, theta_segments):
    """Revolve a list of (x>=0, y) profile points about the y axis.
    profile runs from top pole to bottom pole; endpoints on the axis
    weld into single vertices."""
    m = _Mesh()
    nth = max(8, theta_segments)

    def ring(px, py, j):
        a = 2.0 * pi * (j % nth) / nth
        return (px * cos(a), py, px * sin(a))

    for i in range(len(profile) - 1):
        x0, y0 = profile[i]
        x1, y1 = profile[i + 1]
        for j in range(nth):
            m.face([ring(x0, y0, j), ring(x0, y0, j + 1),
                    ring(x1, y1, j + 1), ring(x1, y1, j)])
    return m.verts, m.faces


# ---------------------------------------------------------------------
# 1. Reuleaux polygon solids of revolution
# ---------------------------------------------------------------------
def _reuleaux_polygon_profile(n, width, samples_per_arc):
    """Right-half (x>=0) profile of a regular Reuleaux n-gon (n odd),
    oriented with a vertex at the top and symmetric about the y axis,
    ordered from the top pole to the bottom pole."""
    if n % 2 == 0:
        n += 1                                    # Reuleaux polygons are odd
    m = (n - 1) // 2
    R = width / (2.0 * sin(m * pi / n))           # circumradius
    verts = [(R * cos(pi / 2 + 2 * pi * j / n),
              R * sin(pi / 2 + 2 * pi * j / n)) for j in range(n)]

    # sample every arc: arc opposite vertex k, centred at verts[k],
    # radius = width, from verts[k+m] to verts[k+m+1].
    boundary = []
    for k in range(n):
        cx, cy = verts[k]
        a = verts[(k + m) % n]
        b = verts[(k + m + 1) % n]
        a0 = math.atan2(a[1] - cy, a[0] - cx)
        a1 = math.atan2(b[1] - cy, b[0] - cx)
        # go the short way
        d = (a1 - a0 + pi) % (2 * pi) - pi
        for s in range(samples_per_arc):
            t = a0 + d * s / samples_per_arc
            boundary.append((cx + width * cos(t), cy + width * sin(t)))

    # right half, ordered top -> bottom (convex + symmetric => y strictly
    # decreasing down the right side)
    right = [(abs(x), y) for (x, y) in boundary if x >= -1e-9]
    right.sort(key=lambda p: -p[1])
    # dedupe consecutive
    prof = [right[0]]
    for p in right[1:]:
        if abs(p[0] - prof[-1][0]) > 1e-7 or abs(p[1] - prof[-1][1]) > 1e-7:
            prof.append(p)
    return prof


def build_reuleaux_revolution(n=3, width=2.0, theta_segments=128,
                              samples_per_arc=48):
    prof = _reuleaux_polygon_profile(n, width, samples_per_arc)
    return _revolve(prof, theta_segments)


# ---------------------------------------------------------------------
# 2/3. Reuleaux tetrahedron and Meissner bodies (radial construction)
# ---------------------------------------------------------------------
_TETRA = [(1.0, 1.0, 1.0), (1.0, -1.0, -1.0),
          (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _tetra_verts(width):
    """Regular tetra vertices, centroid at origin, edge length = width."""
    e = math.dist(_TETRA[0], _TETRA[1])           # = 2*sqrt(2)
    k = width / e
    return [(v[0] * k, v[1] * k, v[2] * k) for v in _TETRA]


def _exit_t(u, c, w):
    """Positive t where ray t*u (from origin) leaves the ball B(c, w)."""
    b = _dot(u, c)
    disc = b * b - (_dot(c, c) - w * w)
    if disc < 0.0:
        return None
    return b + sqrt(disc)


def _spindle_hit(u, A, B, w):
    """t where ray t*u meets the surface of revolution (about axis A-B)
    of a radius-w arc -- the Meissner rounded edge patch.  Returns None
    if the ray misses the thin lens."""
    mx = (0.5 * (A[0] + B[0]), 0.5 * (A[1] + B[1]), 0.5 * (A[2] + B[2]))
    ax = (B[0] - A[0], B[1] - A[1], B[2] - A[2])
    an = sqrt(_dot(ax, ax))
    n = (ax[0] / an, ax[1] / an, ax[2] / an)
    c3 = w * sqrt(3.0) / 2.0

    def f(t):
        d = (u[0] * t - mx[0], u[1] * t - mx[1], u[2] * t - mx[2])
        z = _dot(d, n)
        perp = (d[0] - z * n[0], d[1] - z * n[1], d[2] - z * n[2])
        rho = sqrt(_dot(perp, perp))
        val = w * w - z * z
        if val < 0.0:
            return rho + c3                       # definitely outside lens
        return rho - (sqrt(val) - c3)

    lo, hi, steps = 1e-6, 1.6 * w, 240
    prev, pt = f(lo), lo
    for s in range(1, steps + 1):
        t = lo + (hi - lo) * s / steps
        cur = f(t)
        if prev <= 0.0 < cur:                     # inside -> outside crossing
            a, b = pt, t
            for _ in range(50):
                mid = 0.5 * (a + b)
                if f(mid) <= 0.0:
                    a = mid
                else:
                    b = mid
            return 0.5 * (a + b)
        prev, pt = cur, t
    return None


def _rounded_edges(kind):
    """Index pairs of the three tetra edges to round.  M_V: three edges
    meeting at vertex 0.  M_E: three edges forming the opposite
    triangle (1,2,3)."""
    if kind == 'MEISSNER_E':
        return [(1, 2), (2, 3), (1, 3)]
    return [(0, 1), (0, 2), (0, 3)]               # MEISSNER_V


def _radius(u, cen, w, rounded):
    """Radial distance to the body boundary along unit dir u.

    Each rounded edge's spindle is a small closed lens sitting on that
    edge, so a ray from the centroid only ever crosses the lens of the
    edge it actually exits over; taking the min over all three
    spindles therefore localises the rounding automatically.  (An
    earlier version instead *classified* the exit by its two nearest
    spheres, but that tie is unstable near the vertices where three
    spheres meet and produced a rippled surface.)"""
    ts = [_exit_t(u, c, w) for c in cen]
    r0 = min(t for t in ts if t is not None)
    if not rounded:
        return r0
    r = r0
    for (i, j) in rounded:
        hit = _spindle_hit(u, cen[i], cen[j], w)
        if hit is not None and hit < r:
            r = hit
    return r


def build_tetra_body(kind='MEISSNER_V', width=2.0, phi_segments=96,
                     theta_segments=160):
    """Radial (star-shaped) mesh of the Reuleaux / Meissner tetra."""
    cen = _tetra_verts(width)
    rounded = _rounded_edges(kind) if kind.startswith('MEISSNER') else []
    m = _Mesh()
    nphi = max(8, phi_segments)
    nth = max(8, theta_segments)

    def pt(i, j):
        phi = pi * i / nphi
        theta = 2.0 * pi * (j % nth) / nth
        u = (sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi))
        r = _radius(u, cen, width, rounded)
        return (r * u[0], r * u[1], r * u[2])

    grid = [[pt(i, j) for j in range(nth)] for i in range(nphi + 1)]
    for i in range(nphi):
        for j in range(nth):
            jn = (j + 1) % nth
            m.face([grid[i][j], grid[i][jn], grid[i + 1][jn], grid[i + 1][j]])
    return m.verts, m.faces
