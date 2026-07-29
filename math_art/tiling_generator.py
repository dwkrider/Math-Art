
# Uniform Euclidean Tiling Generator for Blender
#
# The 19 named uniform tilings of the plane by regular polygons:
# the 3 regular tilings, the 8 Archimedean (semiregular) tilings, and
# the 8 Laves tilings (their duals).  Each is built from a translational
# unit cell -- lattice vectors plus a small list of regular-polygon
# tiles -- replicated across the plane.  The Laves tilings are computed
# as the geometric DUAL of the matching Archimedean patch (join the
# centroids of the polygons around every interior vertex), so only the
# 11 base tilings carry hand-placed coordinates.
#
# Part of the Pattern Engine (see pattern_common.py).
#
# References:
#   Johannes Kepler, "Harmonices Mundi" (1619) -- first systematic
#     depiction of the regular and Archimedean (semiregular) tilings by
#     regular polygons.
#   Branko Grunbaum & G. C. Shephard, "Tilings and Patterns" (W. H.
#     Freeman, 1987) -- modern classification of the uniform tilings and
#     their Laves (dual) tilings.

bl_info = {
    "name": "Uniform Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "The 19 uniform Euclidean tilings (regular, "
                   "Archimedean, Laves) as colored tile patterns",
    "category": "Add Mesh",
}

from math import cos, sin, pi, hypot, sqrt, atan2, acos
from collections import defaultdict
import numpy as np

try:
    from . import pattern_common as pc
except Exception:
    import pattern_common as pc


_H = sqrt(3.0) / 2.0                       # height of a unit triangle


# --------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------

def _reg(n, cx, cy, rot=0.0, edge=1.0):
    """The n vertices (CCW) of a regular n-gon centred at (cx, cy),
    edge length `edge` (circumradius = edge / (2 sin(pi/n)))."""
    R = edge / (2.0 * sin(pi / n))
    return np.array([[cx + R * cos(rot + 2.0 * pi * k / n),
                      cy + R * sin(rot + 2.0 * pi * k / n)]
                     for k in range(n)])


def _eqtri(a, b, ref):
    """Third vertex of the equilateral triangle on edge (a, b), chosen
    on the side FARTHER from the reference point `ref` (outward)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ref = np.asarray(ref, float)
    m = 0.5 * (a + b)
    d = b - a
    nrm = np.array([-d[1], d[0]])
    nrm = nrm / (hypot(nrm[0], nrm[1]) + 1e-12)
    h = hypot(d[0], d[1]) * _H
    p1, p2 = m + h * nrm, m - h * nrm
    if hypot(*(p1 - ref)) >= hypot(*(p2 - ref)):
        return p1
    return p2


# --------------------------------------------------------------------
# The 11 base tilings: (b1, b2, [tiles])  -- one translational cell
# --------------------------------------------------------------------

def _base_cell(name):
    """Return (b1, b2, tiles) for a regular/Archimedean tiling, where
    tiles is a list of (N, 2) CCW polygon-vertex arrays making up one
    translational unit cell.  Edge length 1 throughout."""
    if name == 'SQUARE':
        b1, b2 = (1.0, 0.0), (0.0, 1.0)
        tiles = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)]

    elif name == 'TRI':
        b1, b2 = (1.0, 0.0), (0.5, _H)
        tiles = [np.array([[0, 0], [1, 0], [0.5, _H]], float),
                 np.array([[1, 0], [1.5, _H], [0.5, _H]], float)]

    elif name == 'HEX':
        s3 = sqrt(3.0)
        b1, b2 = (s3, 0.0), (s3 / 2.0, 1.5)
        tiles = [_reg(6, 0.0, 0.0, pi / 6)]

    elif name == 'TRIHEX':                       # 3.6.3.6
        s3 = sqrt(3.0)
        b1, b2 = (2.0, 0.0), (1.0, s3)
        hexa = _reg(6, 0.0, 0.0, 0.0)            # flat-top, edge 1
        tdn = np.array([[1, 0], [0.5, _H], [1.5, _H]], float)
        tup = np.array([[-0.5, _H], [0.5, _H], [0.0, s3]], float)
        tiles = [hexa, tdn, tup]

    elif name == 'ELONGTRI':                     # 3.3.3.4.4
        b1, b2 = (1.0, 0.0), (0.5, 1.0 + _H)
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        tup = np.array([[0, 1], [1, 1], [0.5, 1 + _H]], float)
        tdn = np.array([[1, 1], [1.5, 1 + _H], [0.5, 1 + _H]], float)
        tiles = [sq, tup, tdn]

    elif name == 'SNUBSQ':                        # 3.3.4.3.4 (chiral)
        s = sqrt(2.0 + sqrt(3.0))
        b1, b2 = (s, 0.0), (0.0, s)
        A = _reg(4, 0.0, 0.0, pi / 3.0)           # square, +15 deg
        B = _reg(4, s / 2.0, s / 2.0, pi / 6.0)   # square, -15 deg
        tiles = [A, B]
        for k in range(4):
            a, b = A[k], A[(k + 1) % 4]
            tiles.append(np.array([a, b, _eqtri(a, b, (0.0, 0.0))]))

    elif name == 'SNUBHEX':                       # 3.3.3.3.6 (chiral)
        s3 = sqrt(3.0)
        b1, b2 = (2.5, _H), (0.5, 3.0 * _H)
        phi = -2.0 * atan2(s3, 9.0)
        b1a = np.array(b1)
        b2a = np.array(b2)
        H0 = _reg(6, 0.0, 0.0, phi)
        Hb1 = _reg(6, b1a[0], b1a[1], phi)
        Hb2 = _reg(6, b2a[0], b2a[1], phi)
        Hbb = _reg(6, b1a[0] + b2a[0], b1a[1] + b2a[1], phi)
        tiles = [H0]
        for k in range(6):                        # 6 edge triangles
            a, b = H0[k], H0[(k + 1) % 6]
            tiles.append(np.array([a, b, _eqtri(a, b, (0.0, 0.0))]))
        tiles.append(np.array([H0[1], Hb1[3], Hb2[5]]))   # filler F1
        tiles.append(np.array([Hb1[2], Hb2[0], Hbb[4]]))  # filler F2

    elif name == 'RHOMBITRIHEX':                  # 3.4.6.4
        d = 1.0 + sqrt(3.0)
        b1, b2 = (d * _H, d * 0.5), (0.0, d)
        hexa = _reg(6, 0.0, 0.0, 0.0)             # flat-top, edge 1
        tiles = [hexa]
        for k in (0, 1, 2):                        # squares, normals 30/90/150
            a, b = hexa[k], hexa[(k + 1) % 6]
            e = b - a
            nrm = np.array([-e[1], e[0]])
            nrm = nrm / (hypot(nrm[0], nrm[1]) + 1e-12)
            # ensure outward (away from origin)
            if np.dot(0.5 * (a + b), nrm) < 0:
                nrm = -nrm
            tiles.append(np.array([a, b, b + nrm, a + nrm]))
        # two triangles, one at each of two hexagon corners
        v0 = hexa[0]
        tiles.append(np.array([v0, v0 + np.array([_H, 0.5]),
                               v0 + np.array([_H, -0.5])]))
        v1 = hexa[1]
        tiles.append(np.array([v1, v1 + np.array([_H, 0.5]),
                               v1 + np.array([0.0, 1.0])]))

    elif name == 'TRUNCHEX':                      # 3.12.12
        d = 2.0 + sqrt(3.0)
        b1, b2 = (d, 0.0), (d / 2.0, d * _H)
        dod = _reg(12, 0.0, 0.0, pi / 12.0)
        right = dod + np.array(b1, float)          # right dodecagon
        # two triangles filling the gaps toward the right neighbour
        t1 = np.array([dod[0], right[4], dod[1]])
        t2 = np.array([dod[11], right[7], dod[10]])
        tiles = [dod, t1, t2]

    elif name == 'TRUNCSQ':                        # 4.8.8
        d = 1.0 + sqrt(2.0)
        b1, b2 = (d, 0.0), (0.0, d)
        oct8 = _reg(8, 0.0, 0.0, pi / 8.0)
        c = d / 2.0
        r = sqrt(2.0) / 2.0
        sq = np.array([[c, c - r], [c + r, c], [c, c + r], [c - r, c]])
        tiles = [oct8, sq]

    elif name == 'TRUNCTRIHEX':                    # 4.6.12
        d = 3.0 + sqrt(3.0)
        b1, b2 = (d, 0.0), (d / 2.0, d * _H)
        dod = _reg(12, 0.0, 0.0, pi / 12.0)
        tiles = [dod]
        # two hexagon orbits at +/- (1+sqrt3) at 30 deg from a dodecagon
        r6 = 1.0 + sqrt(3.0)
        tiles.append(_reg(6, r6 * _H, r6 * 0.5, 0.0))
        tiles.append(_reg(6, r6 * _H, -r6 * 0.5, 0.0))
        # three squares on dodecagon edges (normals 0, 60, 120 deg)
        ap = 0.5 / sin(pi / 12.0) * cos(pi / 12.0)   # dodecagon apothem
        for ang in (0.0, pi / 3.0, 2.0 * pi / 3.0):
            nx_, ny_ = cos(ang), sin(ang)
            mx, my = ap * nx_, ap * ny_
            tx, ty = -ny_, nx_                       # along-edge dir
            a = np.array([mx + 0.5 * tx, my + 0.5 * ty])
            b = np.array([mx - 0.5 * tx, my - 0.5 * ty])
            out = np.array([nx_, ny_])
            tiles.append(np.array([a, b, b + out, a + out]))

    else:
        raise ValueError("unknown tiling %r" % name)

    return np.array(b1, float), np.array(b2, float), tiles


def _base_iter(name, nx, ny):
    """Yield (poly, tile_index) for every tile of a base tiling over an
    nx x ny run of the unit cell."""
    b1, b2, tiles = _base_cell(name)
    for i in range(nx):
        for j in range(ny):
            off = i * b1 + j * b2
            for ti, tile in enumerate(tiles):
                yield np.asarray(tile, float) + off, ti


# --------------------------------------------------------------------
# Dual (Laves) construction
# --------------------------------------------------------------------

LAVES_OF = {
    'RHOMBILLE': 'TRIHEX', 'CAIRO': 'SNUBSQ', 'FLORET': 'SNUBHEX',
    'PRISMATIC': 'ELONGTRI', 'DELTOIDAL': 'RHOMBITRIHEX',
    'TRIAKIS': 'TRUNCHEX', 'TETRAKIS': 'TRUNCSQ',
    'KISRHOMBILLE': 'TRUNCTRIHEX',
}


def _weld(polys):
    """Weld shared vertices (round to 1e-4) -> (coords, rings) where
    rings[i] is the welded-index ring of polys[i]."""
    coords, index = [], {}

    def vid(x, y):
        key = (round(float(x), 4), round(float(y), 4))
        if key not in index:
            index[key] = len(coords)
            coords.append((float(x), float(y)))
        return index[key]

    rings = [[vid(x, y) for x, y in p] for p in polys]
    return coords, rings


def _dual(polys):
    """The dual tiling of a placed patch: one tile per INTERIOR vertex,
    formed by the centroids of the polygons around it in cyclic order.
    Boundary vertices (touching an unmatched edge) are skipped."""
    coords, rings = _weld(polys)
    ecount = defaultdict(int)
    vedges = defaultdict(set)
    vpolys = defaultdict(list)
    for pi_, r in enumerate(rings):
        m = len(r)
        for k in range(m):
            a, b = r[k], r[(k + 1) % m]
            e = (a, b) if a < b else (b, a)
            ecount[e] += 1
            vedges[a].add(e)
            vedges[b].add(e)
        for v in r:
            vpolys[v].append(pi_)
    cents = [np.mean(np.asarray(p, float), axis=0) for p in polys]
    duals = []
    for v, pl in vpolys.items():
        if not vedges[v] or any(ecount[e] != 2 for e in vedges[v]):
            continue
        pls = sorted(set(pl),
                     key=lambda q: atan2(cents[q][1] - coords[v][1],
                                         cents[q][0] - coords[v][0]))
        if len(pls) < 3:
            continue
        duals.append(np.array([cents[q] for q in pls]))
    return duals


def _build_tiling(name, nx, ny):
    """Return (polys, types) for any of the 19 tilings.  `types` is a
    per-tile orbit index for the 'By Tile Type' color mode."""
    if name in LAVES_OF:
        polys = [p for p, _ in _base_iter(LAVES_OF[name], nx + 2, ny + 2)]
        duals = _dual(polys)
        return duals, [len(d) for d in duals]
    polys, types = [], []
    for p, t in _base_iter(name, nx, ny):
        polys.append(p)
        types.append(t)
    return polys, types


def substrate_faces(name, nx, ny):
    """Shared canonical substrate source: the polygon list for ANY of the
    19 uniform tilings named in `TILING_ITEMS`, including the Laves duals.

    For a regular / Archimedean tiling this is the NX x NY run of unit
    cells from `_base_iter`; for a Laves dual it is the geometric dual of
    the matching Archimedean patch (over-built by two cells so the dual
    has a solid interior).  Returns a list of (M, 2) CCW polygon-vertex
    arrays -- the same polygons `_build_tiling` produces, without the
    per-tile type indices -- so every pattern generator can draw its
    tiling substrate from one canonical set and stay in sync."""
    if name in LAVES_OF:
        base = [p for p, _ in _base_iter(LAVES_OF[name], nx + 2, ny + 2)]
        return _dual(base)
    return [p for p, _ in _base_iter(name, nx, ny)]


# --------------------------------------------------------------------
# Mesh assembly
# --------------------------------------------------------------------

_SIDE_MAT = {3: 0, 4: 1, 5: 2, 6: 3, 8: 4, 12: 5}


def _clip_rect(poly, rect):
    """Sutherland-Hodgman clip of a polygon to an axis-aligned
    rectangle (x0, y0, x1, y1).  Returns the clipped (M, 2) polygon,
    or [] if nothing survives."""
    x0, y0, x1, y1 = rect

    def clip(pts, keep, cut):
        out = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            ka, kb = keep(a), keep(b)
            if ka:
                out.append(a)
                if not kb:
                    out.append(cut(a, b))
            elif kb:
                out.append(cut(a, b))
        return out

    def cx(a, b, v):
        t = (v - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
        return (v, a[1] + t * (b[1] - a[1]))

    def cy(a, b, v):
        t = (v - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
        return (a[0] + t * (b[0] - a[0]), v)

    pts = [(float(x), float(y)) for x, y in poly]
    pts = clip(pts, lambda p: p[0] >= x0, lambda a, b: cx(a, b, x0))
    if len(pts) < 3:
        return []
    pts = clip(pts, lambda p: p[0] <= x1, lambda a, b: cx(a, b, x1))
    if len(pts) < 3:
        return []
    pts = clip(pts, lambda p: p[1] >= y0, lambda a, b: cy(a, b, y0))
    if len(pts) < 3:
        return []
    pts = clip(pts, lambda p: p[1] <= y1, lambda a, b: cy(a, b, y1))
    return np.array(pts) if len(pts) >= 3 else []


def _pip(poly, x, y):
    """Point-in-polygon (ray casting)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def _covered(P, rect):
    """True if the whole rectangle boundary lies on tiles.  Since the
    tiling interior is gap-free, a covered perimeter means a covered
    rectangle (the covered region is essentially convex)."""
    x0, y0, x1, y1 = rect
    pts = []
    for t in np.linspace(0.0, 1.0, 11):
        pts += [(x0 + t * (x1 - x0), y0), (x0 + t * (x1 - x0), y1),
                (x0, y0 + t * (y1 - y0)), (x1, y0 + t * (y1 - y0))]
    for x, y in pts:
        if not any(_pip(p, x, y) for p in P):
            return False
    return True


def _trim_rect(polys, nx, ny, pad):
    """A centred axis-aligned rectangle ~nx x ny cells across, shrunk
    until it lies wholly inside the covered (padded) region -- so skew
    lattices clip to a clean rectangle with no ragged edge."""
    P = [np.asarray(p, float) for p in polys]
    allv = np.vstack(P)
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    c = 0.5 * (lo + hi)
    hw0 = 0.5 * (hi[0] - lo[0]) * nx / (nx + pad)
    hh0 = 0.5 * (hi[1] - lo[1]) * ny / (ny + pad)
    rect = (c[0] - hw0, c[1] - hh0, c[0] + hw0, c[1] + hh0)
    for t in (1.0, 0.93, 0.86, 0.79, 0.72, 0.65, 0.58, 0.5, 0.42):
        rect = (c[0] - hw0 * t, c[1] - hh0 * t,
                c[0] + hw0 * t, c[1] + hh0 * t)
        if _covered(P, rect):
            break
    return rect


def build(name, nx, ny, color_by='SIDES', margin=0.0, height=0.0,
          trim=False):
    """Merged (verts, faces, mats) for a tiling patch."""
    return pc.merge_cells(build_cells(name, nx, ny, color_by,
                                      margin, height, trim))


def build_cells(name, nx, ny, color_by='SIDES', margin=0.0,
                height=0.0, trim=False):
    """One (verts, faces, mats) cell per tile of a named uniform
    tiling.  With trim, the patch is over-built and clipped to a clean
    central rectangle."""
    return cells_from_polys(lambda a, b: _build_tiling(name, a, b),
                            nx, ny, color_by, margin, height, trim)


def cells_from_polys(build_fn, nx, ny, color_by='SIDES', margin=0.0,
                     height=0.0, trim=False, pad=2):
    """Shared tiling->cells assembly, reused by the k-uniform and
    monohedral generators.  `build_fn(nx, ny)` returns (polys, types)
    for a patch; this applies optional trim (over-build + clip to a
    clean rectangle), per-tile color, margin inset and relief height,
    and returns one (verts, faces, mats) cell per surviving tile."""
    if trim:
        polys, types = build_fn(nx + pad, ny + pad)
        rect = _trim_rect(polys, nx, ny, pad)
    else:
        polys, types = build_fn(nx, ny)
        rect = None
    cells = []
    for poly, typ in zip(polys, types):
        p = np.asarray(poly, float)
        n0 = len(p)                              # original side count
        if color_by == 'SIDES':
            kind = _SIDE_MAT.get(n0, 6)
        elif color_by == 'TYPE':
            kind = int(typ)
        else:
            kind = 0
        if rect is not None:
            p = _clip_rect(p, rect)
            if len(p) < 3:
                continue
        if margin > 0.0:
            c = p.mean(axis=0)
            p = c + (p - c) * (1.0 - margin)
        n = len(p)
        cv, cf, cm = [], [], []
        if height > 0.0:
            pc.prisms(cv, cf, cm, [p], height, 0.0, kind)
        else:
            for x, y in p:
                cv.append((float(x), float(y), 0.0))
            cf.append(tuple(range(n)))
            cm.append(kind)
        cells.append((cv, cf, cm))
    return cells


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

TILING_ITEMS = [
    ('TRI', "Triangular (3.3.3.3.3.3)", "Regular triangular tiling"),
    ('SQUARE', "Square (4.4.4.4)", "Regular square tiling"),
    ('HEX', "Hexagonal (6.6.6)", "Regular hexagonal tiling"),
    ('TRIHEX', "Trihexagonal (3.6.3.6)", "Archimedean"),
    ('SNUBSQ', "Snub Square (3.3.4.3.4)", "Archimedean (chiral)"),
    ('SNUBHEX', "Snub Hexagonal (3.3.3.3.6)", "Archimedean (chiral)"),
    ('ELONGTRI', "Elongated Triangular (3.3.3.4.4)", "Archimedean"),
    ('RHOMBITRIHEX', "Rhombitrihexagonal (3.4.6.4)", "Archimedean"),
    ('TRUNCHEX', "Truncated Hexagonal (3.12.12)", "Archimedean"),
    ('TRUNCSQ', "Truncated Square (4.8.8)", "Archimedean"),
    ('TRUNCTRIHEX', "Truncated Trihexagonal (4.6.12)", "Archimedean"),
    ('RHOMBILLE', "Rhombille (dual 3.6.3.6)", "Laves dual"),
    ('CAIRO', "Cairo Pentagonal (dual 3.3.4.3.4)", "Laves dual"),
    ('FLORET', "Floret Pentagonal (dual 3.3.3.3.6)", "Laves dual"),
    ('PRISMATIC', "Prismatic Pentagonal (dual 3.3.3.4.4)", "Laves dual"),
    ('DELTOIDAL', "Deltoidal Trihexagonal (dual 3.4.6.4)", "Laves dual"),
    ('TRIAKIS', "Triakis Triangular (dual 3.12.12)", "Laves dual"),
    ('TETRAKIS', "Tetrakis Square (dual 4.8.8)", "Laves dual"),
    ('KISRHOMBILLE', "Kisrhombille (dual 4.6.12)", "Laves dual"),
]


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_tiling_add(bpy.types.Operator, AddObjectHelper):
        """Add a uniform Euclidean tiling"""
        bl_idname = "mesh.tiling_add"
        bl_label = "Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        tiling: EnumProperty(name="Tiling", items=TILING_ITEMS,
                             default='RHOMBITRIHEX')
        nx: IntProperty(name="Cells X", default=5, min=1, max=40)
        ny: IntProperty(name="Cells Y", default=5, min=1, max=40)
        color_by: EnumProperty(
            name="Color By",
            items=[('SIDES', "By Sides",
                    "Material per polygon side count"),
                   ('TYPE', "By Tile Type",
                    "Material per tile orbit in the unit cell"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='SIDES')
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat 2D mesh; > 0 extrudes each tile")
        trim: BoolProperty(
            name="Trim Boundary", default=False,
            description="Clip the patch to a clean rectangle, removing "
                        "the ragged edge and stray boundary tiles")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object "
                        "(parented to an empty) so tiles can be edited "
                        "individually")

        def execute(self, context):
            cells = build_cells(
                self.tiling, self.nx, self.ny, self.color_by,
                self.margin, self.height, self.trim)
            label = dict((k, v) for k, v, _ in TILING_ITEMS)[self.tiling]
            obj = pc.emit(context, "Tiling %s" % label, cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no tiling generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s  V=%d F=%d" %
                            (self.tiling, len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s  %d tiles" %
                            (self.tiling, len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for p in ('tiling', 'nx', 'ny', 'color_by', 'margin',
                      'height'):
                lay.prop(self, p)
            lay.prop(self, 'trim')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.tiling_add", icon='MESH_GRID')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_tiling_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_tiling_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _check_valid(polys):
    """Weld a patch and verify every interior vertex (all incident
    edges shared by exactly two tiles) has interior angles summing to
    2*pi.  Return (#interior vertices, max |anglesum - 2pi|)."""
    coords, rings = _weld(polys)
    ecount = defaultdict(int)
    vedges = defaultdict(set)
    vinc = defaultdict(list)
    for pi_, r in enumerate(rings):
        m = len(r)
        for k in range(m):
            a, b = r[k], r[(k + 1) % m]
            e = (a, b) if a < b else (b, a)
            ecount[e] += 1
            vedges[a].add(e)
            vedges[b].add(e)
        for k, v in enumerate(r):
            vinc[v].append((pi_, k))
    C = [np.array(c) for c in coords]
    nint, maxerr = 0, 0.0
    for v, inc in vinc.items():
        if any(ecount[e] != 2 for e in vedges[v]):
            continue
        nint += 1
        s = 0.0
        for pi_, k in inc:
            r = rings[pi_]
            m = len(r)
            V = C[r[k]]
            a = C[r[k - 1]]
            b = C[r[(k + 1) % m]]
            va, vb = a - V, b - V
            cosang = np.dot(va, vb) / (np.linalg.norm(va) *
                                       np.linalg.norm(vb) + 1e-12)
            s += acos(max(-1.0, min(1.0, cosang)))
        maxerr = max(maxerr, abs(s - 2.0 * pi))
    return nint, maxerr


def _check_regular(polys, tol=1e-3):
    """Every edge of every tile has length ~1.0."""
    for p in polys:
        p = np.asarray(p, float)
        m = len(p)
        for k in range(m):
            if abs(hypot(*(p[(k + 1) % m] - p[k])) - 1.0) > tol:
                return False
    return True


_ORDER = [k for k, _, _ in TILING_ITEMS]


if __name__ == "__main__":
    bad = []
    for name in _ORDER:
        laves = name in LAVES_OF
        if laves:
            polys, _ = _build_tiling(name, 4, 4)
        else:
            polys = [p for p, _ in _base_iter(name, 4, 4)]
        reg_ok = True if laves else _check_regular(polys)
        nint, maxerr = _check_valid(polys)
        ok = reg_ok and nint > 0 and maxerr < 1e-3
        print("%-13s tiles=%4d int=%4d maxerr=%.2e reg=%s %s" %
              (name, len(polys), nint, maxerr, reg_ok,
               "OK" if ok else "BAD"))
        if not ok:
            bad.append(name)
    print("RESULT:", "OK" if not bad else "BAD %s" % bad)
