# Curved-surface mapping for the Pattern Engine.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operators stay in their flat generator
# modules.
#
# Every tiling backend in this extension builds its tiles in a flat
# parameter domain and extrudes them along +Z.  This module lets the
# same tiles be laid on a CURVED surface instead: a surface supplies, for
# each domain point, a 3D position and the outward unit normal there, and
# relief is offset along that normal rather than along +Z.
#
# Three surfaces are provided.
#
#   PLANE   the identity map, so the curved code path can be exercised
#       (and unit-tested) against the flat one it generalises.
#
#   TORUS   the flat torus R^2 / Lambda, realised as the standard torus
#       of revolution.  The quotient is what makes the tiling exact: a
#       tiling periodic under a sublattice of Lambda descends to the
#       torus with no seam and no defect, because opposite sides of the
#       fundamental domain are genuinely identified.  The EMBEDDING in
#       R^3, however, is not isometric -- by Gauss's Theorema Egregium
#       the flat torus has zero curvature everywhere and no smooth
#       surface of revolution does, so the donut necessarily stretches
#       the outer equator and compresses the inner one.  (C^1 isometric
#       embeddings do exist, by Nash-Kuiper; they are nowhere twice
#       differentiable and are not what anybody wants to look at.)  The
#       distortion is therefore a property of the picture, not an error
#       in the tiling.
#
#   SPHERE  the equirectangular chart: domain x is longitude, domain y is
#       latitude.  Unlike the torus this is NOT a quotient -- the sphere
#       is not flat, so no Euclidean tiling maps onto it without
#       distortion, and the chart is singular at both poles.  It is
#       offered for the cases where a recognisable pattern on a ball is
#       the point; `projection/stereographic.py` gives the conformal
#       alternative.
#
# The one genuinely new piece here is refinement.  `prisms.prisms`
# consumes tile rings verbatim, which is correct in the plane and wrong
# on a curved surface: a straight chord between two mapped corners cuts
# below the surface, so tiles sink away from their neighbours and meet at
# visible creases.  Every ring is therefore subdivided before mapping,
# and each tile is built as a polar grid (concentric rows from its
# centroid out to the boundary) with every vertex mapped, so the tile
# hugs the surface instead of chording across it.
#
# Watertightness between neighbouring tiles is by construction, in two
# steps: corners shared by adjacent tiles are snapped to one
# representative point, and edge subdivision is made deterministic in the
# UNORDERED pair of endpoints, so a tile walking an edge a->b and its
# neighbour walking the same edge b->a emit bit-identical interior
# points.  Without both, adjacent tiles sample from marginally different
# endpoints and the shared edge splits into two nearly-coincident
# polylines -- the classic hairline crack.
#
# References:
# - Carl Friedrich Gauss, "Disquisitiones generales circa superficies
#   curvas" (1827) -- the Theorema Egregium: Gaussian curvature is an
#   intrinsic invariant, so a flat torus admits no isometric embedding as
#   a smooth surface of revolution.
# - John F. Nash, "C^1 isometric imbeddings", Annals of Mathematics 60
#   (1954), and Nicolaas H. Kuiper, "On C^1-isometric imbeddings",
#   Indagationes Mathematicae 17 (1955) -- the C^1 isometric embeddings
#   of the flat torus that the smooth theory forbids.
# - The polar-grid curved cell and the shared-corner canonicalisation
#   follow the treatment already used for the hyperbolic hemisphere and
#   pseudosphere models in `math_art/hyperbolic/tilings.py`.

from math import ceil, cos, pi, sin                # noqa: F401
import numpy as np


# Default subdivision target, in units of the pattern domain: no tile
# edge longer than this is mapped as a single chord.  0.25 puts four
# samples along a unit-cell edge, which is enough for the 2 m cube
# convention without inflating the vertex count.
DEFAULT_MAX_EDGE = 0.25


# Hard caps, so a degenerate `max_edge` cannot explode the mesh.
_MAX_EDGE_SPLITS = 64


_MAX_RADIAL = 6


# --------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------

class Surface:
    """A parameterisation of a surface over the flat pattern domain.

    Subclasses map domain points (x, y) to a 3D position together with
    the OUTWARD UNIT NORMAL there, so relief is offset along the normal
    instead of along +Z.

    `periods` records the domain periods (px, py) -- the widths after
    which the surface closes up on itself -- or None on an axis that does
    not close.  A caller that wants an exact, seamless tiling must lay
    its tiles on a lattice commensurate with these.
    """

    periods = (None, None)

    def at(self, pts):
        """(N, 2) domain points -> ((N, 3) positions, (N, 3) normals)."""
        raise NotImplementedError

    def offset(self, pts, d):
        """(N, 2) domain points, displaced `d` along the surface normal."""
        P, nrm = self.at(pts)
        return P + nrm * float(d)


class PlaneSurface(Surface):
    """The identity map: (x, y) -> (x, y, 0), normal +Z.

    Exists so the curved path degenerates exactly to the flat one, which
    is what makes the two testable against each other.
    """

    def at(self, pts):
        P = np.asarray(pts, float).reshape(-1, 2)
        pos = np.column_stack([P[:, 0], P[:, 1], np.zeros(len(P))])
        nrm = np.zeros((len(P), 3))
        nrm[:, 2] = 1.0
        return pos, nrm


class TorusSurface(Surface):
    """The flat torus of domain size `width` x `height`, embedded as the
    standard torus of revolution with the given major and minor radii.

    Domain x becomes the major angle u and domain y the minor angle v:

        u = 2 pi (x - x0) / width,   v = 2 pi (y - y0) / height
        P = ((R + r cos v) cos u, (R + r cos v) sin u, r sin v)
        n = (cos v cos u, cos v sin u, sin v)

    Both angles are periodic, so a tile crossing either seam wraps
    continuously with no special handling -- which is precisely the
    property that makes a lattice-commensurate tiling seamless.  Offset
    along n by d is exactly the torus of minor radius r + d, so relief
    reads as a radial thickening.
    """

    def __init__(self, width, height, major=1.0, minor=0.4,
                 x0=0.0, y0=0.0):
        self.width = float(width)
        self.height = float(height)
        self.major = float(major)
        self.minor = float(minor)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.periods = (self.width, self.height)

    def at(self, pts):
        P = np.asarray(pts, float).reshape(-1, 2)
        u = 2.0 * pi * (P[:, 0] - self.x0) / self.width
        v = 2.0 * pi * (P[:, 1] - self.y0) / self.height
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        ring = self.major + self.minor * cv
        pos = np.column_stack([ring * cu, ring * su, self.minor * sv])
        nrm = np.column_stack([cv * cu, cv * su, sv])
        return pos, nrm


class SphereSurface(Surface):
    """The sphere under the equirectangular chart: domain x is longitude
    over `width`, domain y is latitude over `height`.

        theta = 2 pi (x - x0) / width
        phi   = pi ((y - y0) / height - 1/2)
        P     = R (cos phi cos theta, cos phi sin theta, sin phi)

    A point of a sphere is its own outward normal (up to the radius), so
    relief offsets radially.  Only the longitude closes up; latitude runs
    pole to pole and the chart is singular at both ends, where a whole
    domain row collapses to a point.  Area distortion is severe near the
    poles -- this chart is honest about being a chart, not a quotient.
    """

    def __init__(self, width, height, radius=1.0, x0=0.0, y0=0.0):
        self.width = float(width)
        self.height = float(height)
        self.radius = float(radius)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.periods = (self.width, None)

    def at(self, pts):
        P = np.asarray(pts, float).reshape(-1, 2)
        th = 2.0 * pi * (P[:, 0] - self.x0) / self.width
        ph = pi * ((P[:, 1] - self.y0) / self.height - 0.5)
        cph = np.cos(ph)
        nrm = np.column_stack([cph * np.cos(th), cph * np.sin(th),
                               np.sin(ph)])
        return nrm * self.radius, nrm


# --------------------------------------------------------------------
# Refinement
# --------------------------------------------------------------------

def _ordered(a, b):
    """The pair (a, b) in a deterministic order, plus whether it was
    swapped.  Keying subdivision on the UNORDERED pair is what makes two
    tiles sharing an edge emit bit-identical interior points."""
    if (float(a[0]), float(a[1])) <= (float(b[0]), float(b[1])):
        return a, b, False
    return b, a, True


def edge_points(a, b, max_edge=DEFAULT_MAX_EDGE, cap=_MAX_EDGE_SPLITS):
    """Points along the segment a -> b, EXCLUDING b, spaced at most
    `max_edge` apart in the pattern domain.

    Deterministic in the unordered pair {a, b}: the samples are computed
    in a canonical direction and reversed if needed, so the neighbouring
    tile walking b -> a produces the very same interior floats and the
    shared edge is one polyline rather than two.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    length = float(np.hypot(*(b - a)))
    n = int(min(cap, max(1, ceil(length / max(float(max_edge), 1e-9)))))
    p, q, flipped = _ordered(a, b)
    t = (np.arange(n + 1, dtype=float) / n)[:, None]
    pts = p[None, :] * (1.0 - t) + q[None, :] * t
    if flipped:
        pts = pts[::-1]
    return pts[:-1]


def refine_poly(poly, max_edge=DEFAULT_MAX_EDGE, cap=_MAX_EDGE_SPLITS):
    """A closed 2D polygon with vertices inserted along every edge, so no
    edge spans more than `max_edge` of the pattern domain.

    The inserted points are collinear, so the polygon's area and shape
    are unchanged -- this only buys the resolution a curved map needs.
    """
    P = np.asarray(poly, float)
    if len(P) < 3:
        return P
    out = [edge_points(P[k], P[(k + 1) % len(P)], max_edge, cap)
           for k in range(len(P))]
    return np.vstack(out)


def refine_segment(a, b, max_edge=DEFAULT_MAX_EDGE, cap=_MAX_EDGE_SPLITS):
    """An open segment a -> b subdivided to `max_edge`, both ends kept."""
    pts = edge_points(a, b, max_edge, cap)
    return np.vstack([pts, np.asarray(b, float)[None, :]])


def canonicalize_corners(polys, eps=1e-6):
    """Snap tile corners that coincide to ONE representative point.

    Backends compute each tile independently -- by substitution, by a
    lattice offset, by a group word -- so a corner physically shared by
    several tiles arrives with slightly different floats in each.  Mapped
    to a curved surface and subdivided, those near-duplicates open into
    hairline cracks.  Matching on a spatial hash (rather than by
    rounding, which would split pairs straddling a rounding boundary)
    gives every shared corner a single canonical position.
    """
    cell = 2.0 * float(eps)
    grid = {}

    def canon(x, y):
        gx, gy = int(np.floor(x / cell)), int(np.floor(y / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for bx, by in grid.get((gx + dx, gy + dy), ()):
                    if abs(bx - x) <= eps and abs(by - y) <= eps:
                        return bx, by
        grid.setdefault((gx, gy), []).append((x, y))
        return x, y

    out = []
    for poly in polys:
        P = np.asarray(poly, float)
        out.append(np.array([canon(float(x), float(y)) for x, y in P],
                            dtype=float))
    return out


# --------------------------------------------------------------------
# Curved cells
# --------------------------------------------------------------------

def _polar_rows(ring, max_edge, cap=_MAX_RADIAL):
    """Concentric rows of domain points, from a tile's centroid out to
    its (already refined) boundary ring.

    A single flat fan from centroid to rim chords straight across the
    surface, so a large tile sinks below it and meets its neighbours in a
    crease.  Interior rows, each mapped onto the surface in turn, remove
    that dip.  Small tiles get one row and behave exactly like the fan.
    """
    c = ring.mean(axis=0)
    dmax = float(np.max(np.linalg.norm(ring - c, axis=1)))
    k = int(min(cap, max(1, ceil(dmax / max(float(max_edge), 1e-9)))))
    rows = [c[None, :]]
    for a in range(1, k):
        t = a / float(k)
        rows.append(c[None, :] * (1.0 - t) + ring * t)
    rows.append(ring)
    return rows, k


def _grid_faces(base, n, k, reverse=False):
    """Triangles of one polar grid: row 0 is the single centre vertex,
    rows 1..k each hold n vertices.  Vertex a,i is at
    base + (0 if a == 0 else 1 + (a - 1) * n + i)."""

    def vid(a, i):
        return base + (0 if a == 0 else 1 + (a - 1) * n + i)

    faces = []
    for a in range(k):
        for i in range(n):
            j = (i + 1) % n
            if reverse:
                faces.append((vid(a, i), vid(a + 1, j), vid(a + 1, i)))
            else:
                faces.append((vid(a, i), vid(a + 1, i), vid(a + 1, j)))
            if a > 0:                        # the centre row is a fan
                if reverse:
                    faces.append((vid(a, i), vid(a, j), vid(a + 1, j)))
                else:
                    faces.append((vid(a, i), vid(a + 1, j), vid(a, j)))
    return faces


def _grid_verts(rows, surf, off, n, k):
    """Map every row onto the surface, offset along the normal."""
    verts = []
    for a in range(k + 1):
        P, nrm = surf.at(rows[a])
        Q = P + nrm * float(off)
        if a == 0:
            verts.append(tuple(Q[0]))
        else:
            verts.extend(tuple(p) for p in Q[:n])
    return verts


def surface_patch(verts, faces, mats, polys2d, surf, off=0.0, mat=0,
                  max_edge=DEFAULT_MAX_EDGE):
    """Append one single-sided curved patch per 2D polygon: the tile's
    polar grid mapped onto `surf` and displaced `off` along the normal.

    This is the curved twin of the zero-height branch in
    `tiling_generator.cells_from_polys`, which emits a flat tile as a
    single n-gon -- correct in the plane, and a chord through the surface
    anywhere else.
    """
    for poly in polys2d:
        ring = refine_poly(poly, max_edge)
        if len(ring) < 3:
            continue
        rows, k = _polar_rows(ring, max_edge)
        n = len(ring)
        b0 = len(verts)
        verts.extend(_grid_verts(rows, surf, off, n, k))
        new = _grid_faces(b0, n, k)
        faces.extend(new)
        mats.extend([mat] * len(new))


def surface_prisms(verts, faces, mats, polys2d, surf, off_top, off_bot,
                   mat=0, max_edge=DEFAULT_MAX_EDGE):
    """Append watertight curved prisms (one per 2D polygon) between the
    surfaces offset `off_bot` and `off_top` along the normal.

    The curved twin of `prisms.prisms`: outer shell, reversed inner
    shell, and perimeter walls joining the two boundary rings.  Because
    both shells are polar grids of the SAME refined ring, the walls are
    quads and the solid closes exactly.
    """
    for poly in polys2d:
        ring = refine_poly(poly, max_edge)
        if len(ring) < 3:
            continue
        rows, k = _polar_rows(ring, max_edge)
        n = len(ring)
        b0 = len(verts)
        verts.extend(_grid_verts(rows, surf, off_top, n, k))
        b1 = len(verts)
        verts.extend(_grid_verts(rows, surf, off_bot, n, k))

        new = _grid_faces(b0, n, k)
        new += _grid_faces(b1, n, k, reverse=True)
        for i in range(n):                             # perimeter walls
            j = (i + 1) % n
            outer_i = b0 + 1 + (k - 1) * n + i
            outer_j = b0 + 1 + (k - 1) * n + j
            inner_i = b1 + 1 + (k - 1) * n + i
            inner_j = b1 + 1 + (k - 1) * n + j
            new.append((outer_i, outer_j, inner_j, inner_i))
        faces.extend(new)
        mats.extend([mat] * len(new))


def make_surface(kind, width, height, major=1.0, minor=0.4, radius=1.0):
    """Build a surface by name: 'PLANE', 'TORUS' or 'SPHERE'.

    `width` and `height` are the extent of the pattern domain that should
    be wrapped once around the surface -- for an exact torus tiling they
    are the fundamental domain's lattice spans times the repeat counts.
    """
    if kind == 'PLANE':
        return PlaneSurface()
    if kind == 'TORUS':
        return TorusSurface(width, height, major, minor)
    if kind == 'SPHERE':
        return SphereSurface(width, height, radius)
    raise ValueError("unknown surface %r" % (kind,))


# --------------------------------------------------------------------

def _selftest():
    ok = True
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    # PLANE must reproduce the flat map exactly, or the curved path is
    # not a generalisation of the flat one.
    pl = PlaneSurface()
    P, N = pl.at(sq)
    good = (np.allclose(P[:, :2], sq) and np.allclose(P[:, 2], 0.0)
            and np.allclose(N, np.array([0.0, 0.0, 1.0])))
    ok &= good
    print("surfacemap: PLANE is the identity with +Z normal "
          f"{'OK' if good else 'FAIL'}")

    # TORUS: every mapped point must satisfy the torus equation
    # (hypot(x, y) - R)^2 + z^2 = r^2 ...
    R, r = 1.3, 0.45
    tor = TorusSurface(3.0, 2.0, R, r)
    dom = np.random.default_rng(7).uniform(-4.0, 4.0, size=(400, 2))
    P, N = tor.at(dom)
    res = (np.hypot(P[:, 0], P[:, 1]) - R) ** 2 + P[:, 2] ** 2 - r * r
    good = float(np.max(np.abs(res))) < 1e-12
    ok &= good
    print(f"surfacemap: TORUS points lie on the torus, max |res| "
          f"{float(np.max(np.abs(res))):.2e} {'OK' if good else 'FAIL'}")

    # ... and the reported normal must be the true unit normal: unit
    # length, and perpendicular to both tangents (checked by central
    # differences, whose own truncation error sets the tolerance).
    h = 1e-6
    dx = (tor.at(dom + [h, 0.0])[0] - tor.at(dom - [h, 0.0])[0]) / (2 * h)
    dy = (tor.at(dom + [0.0, h])[0] - tor.at(dom - [0.0, h])[0]) / (2 * h)
    unit = float(np.max(np.abs(np.linalg.norm(N, axis=1) - 1.0)))
    perp = max(float(np.max(np.abs(np.sum(N * dx, axis=1)))),
               float(np.max(np.abs(np.sum(N * dy, axis=1)))))
    good = unit < 1e-12 and perp < 1e-6
    ok &= good
    print(f"surfacemap: TORUS normal is unit ({unit:.1e}) and tangent-"
          f"perpendicular ({perp:.1e}) {'OK' if good else 'FAIL'}")

    # Periodicity in both angles is the whole point of the torus: a tile
    # crossing either seam must land exactly where its image lands.
    a = tor.at(dom)[0]
    b = tor.at(dom + [tor.width, 0.0])[0]
    c = tor.at(dom + [0.0, tor.height])[0]
    wrap = max(float(np.max(np.abs(a - b))), float(np.max(np.abs(a - c))))
    good = wrap < 1e-12
    ok &= good
    print(f"surfacemap: TORUS wraps in both periods, max drift "
          f"{wrap:.1e} {'OK' if good else 'FAIL'}")

    # Offsetting along the normal by d is exactly the torus of minor
    # radius r + d -- i.e. relief reads as a radial thickening.
    d = 0.07
    Q = tor.offset(dom, d)
    res2 = ((np.hypot(Q[:, 0], Q[:, 1]) - R) ** 2 + Q[:, 2] ** 2
            - (r + d) ** 2)
    good = float(np.max(np.abs(res2))) < 1e-12
    ok &= good
    print("surfacemap: TORUS normal offset is the r+d torus "
          f"{'OK' if good else 'FAIL'}")

    # SPHERE: radius exact, normal radial.
    sph = SphereSurface(4.0, 2.0, 1.7)
    P, N = sph.at(dom)
    rad = float(np.max(np.abs(np.linalg.norm(P, axis=1) - 1.7)))
    radial = float(np.max(np.abs(P - N * 1.7)))
    good = rad < 1e-12 and radial < 1e-12
    ok &= good
    print(f"surfacemap: SPHERE radius {rad:.1e} and radial normal "
          f"{radial:.1e} {'OK' if good else 'FAIL'}")

    # refine_poly inserts collinear points only: no edge over max_edge,
    # and the area is untouched.
    from .polygon2d import signed_area
    fine = refine_poly(sq, 0.25)
    seg = np.linalg.norm(np.roll(fine, -1, axis=0) - fine, axis=1)
    a0 = abs(signed_area(sq))
    a1 = abs(signed_area(fine))
    good = (float(np.max(seg)) <= 0.25 + 1e-12 and abs(a1 - a0) < 1e-12
            and len(fine) == 16)
    ok &= good
    print(f"surfacemap: refine_poly V={len(fine)} max edge "
          f"{float(np.max(seg)):.3f} area {a1:.6f} (exp {a0:.6f}) "
          f"{'OK' if good else 'FAIL'}")

    # The watertightness contract: two tiles sharing an edge must emit
    # bit-identical interior points, whichever way each walks it.  Exact
    # equality is the test, not a tolerance -- a tolerance would pass the
    # very float drift this exists to prevent.
    fwd = edge_points((0.3, -0.7), (2.1, 1.9), 0.25)
    rev = edge_points((2.1, 1.9), (0.3, -0.7), 0.25)
    good = (len(fwd) == len(rev)
            and np.array_equal(fwd[1:], rev[1:][::-1]))
    ok &= good
    print("surfacemap: shared edge subdivides identically both ways "
          f"({len(fwd)} pts) {'OK' if good else 'FAIL'}")

    # canonicalize_corners must fuse near-duplicate corners and leave
    # genuinely distinct ones alone.
    p1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    p2 = np.array([[1.0 + 3e-9, 1e-9], [2.0, 0.0], [1.5, 1.0]])
    c1, c2 = canonicalize_corners([p1, p2], eps=1e-6)
    fused = np.array_equal(c1[1], c2[0])
    kept = len({tuple(v) for v in np.vstack([c1, c2])}) == 5
    good = fused and kept
    ok &= good
    print(f"surfacemap: corners fused={fused} distinct kept={kept} "
          f"{'OK' if good else 'FAIL'}")

    # A curved prism must be a closed manifold: every edge in exactly two
    # faces.  This is the check that catches a mismatched inner shell or
    # a missing perimeter wall.
    v, f, m = [], [], []
    surface_prisms(v, f, m, [sq], tor, 0.12, 0.0, mat=2, max_edge=0.34)
    cnt = {}
    for face in f:
        for x, y in zip(face, list(face[1:]) + [face[0]]):
            k = (x, y) if x < y else (y, x)
            cnt[k] = cnt.get(k, 0) + 1
    closed = all(c == 2 for c in cnt.values())
    good = (closed and len(m) == len(f) and set(m) == {2}
            and np.all(np.isfinite(np.asarray(v, float))))
    ok &= good
    print(f"surfacemap: curved prism closed V={len(v)} F={len(f)} "
          f"every-edge-twice={closed} {'OK' if good else 'FAIL'}")

    # A patch is single-sided, and with zero offset its vertices lie ON
    # the surface -- the property the whole module exists to provide.
    v2, f2, m2 = [], [], []
    surface_patch(v2, f2, m2, [sq], sph, 0.0, mat=1, max_edge=0.34)
    V2 = np.asarray(v2, float)
    onsurf = float(np.max(np.abs(np.linalg.norm(V2, axis=1) - 1.7)))
    good = onsurf < 1e-12 and len(f2) == len(m2) and len(f2) > 0
    ok &= good
    print(f"surfacemap: patch vertices lie on the sphere ({onsurf:.1e}) "
          f"F={len(f2)} {'OK' if good else 'FAIL'}")

    # PLANE + surface_patch must agree with the flat n-gon it replaces:
    # same area, in the same place.
    v3, f3, m3 = [], [], []
    surface_patch(v3, f3, m3, [sq], pl, 0.0, mat=0, max_edge=0.34)
    V3 = np.asarray(v3, float)
    good = (np.allclose(V3[:, 2], 0.0)
            and abs(V3[:, 0].min()) < 1e-12 and abs(V3[:, 0].max() - 1) < 1e-12
            and abs(V3[:, 1].min()) < 1e-12 and abs(V3[:, 1].max() - 1) < 1e-12)
    ok &= good
    print("surfacemap: PLANE patch stays flat and in place "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("surfacemap self-test failed")
