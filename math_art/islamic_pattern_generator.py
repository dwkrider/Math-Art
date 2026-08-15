
# Islamic Star Pattern Generator for Blender
#
# Geometric star patterns ("girih" strapwork) built by Hankin's
# "polygons in contact" (PIC) method.  An underlying tiling of regular
# polygons is laid down; on every edge, from its MIDPOINT, a pair of
# rays leaves at a fixed CONTACT ANGLE to the edge, one ray directed
# into each incident tile.  Within a tile the rays from its edges
# extend and meet -- each ray running until it reaches the next ray
# around the tile -- forming that tile's star motif.  Because the
# contact points sit at shared edge midpoints, the strapwork is
# continuous from tile to tile.  The contact angle is the primary
# knob: small angles give spiky acute stars, large angles obtuse ones.
#
# The substrate can be any of our uniform tilings (reused from
# tiling_generator): the square grid (4/8-point stars), the hexagonal
# tiling (6/12-point), the triangular tiling (3/6-point), the 4.8.8
# truncated-square tiling (the iconic 8-point star-and-cross), and the
# twelve-fold 3.12.12 and 4.6.12 tilings.
#
# Beyond the plain PIC star, three further motif families are built on
# the same continuous-band machinery:
#
#   * ROSETTE -- the classic Islamic rosette (an n-pointed star ringed
#     by n petal / kite shapes), realized by Kaplan's rosette transform:
#     a regular n-gon is subdivided into a central n-gon plus a ring of
#     n pentagons, and the PIC / inference construction is applied to
#     those cells.  The petals emerge from the pentagons, the central
#     star from the inner n-gon, and because every original edge is split
#     at its midpoint the strapwork stays continuous from tile to tile.
#   * INFERENCE -- Kaplan's motif inference generalizes PIC to any
#     (irregular) tile: contact rays leave each edge midpoint at the
#     contact angle to that edge's true normal and are truncated at their
#     nearest crossing, so mixed / non-regular substrates (e.g. the Laves
#     duals) still knit a consistent, edge-continuous motif.
#   * STAR -- a regular star polygon {n/d} placed in each tile, its tips
#     anchored at the edge midpoints so it too feeds the band tracer.
#
# Output is strapwork as flat RIBBONS (default), optionally given
# relief by extrusion to a height on an optional backing slab, in the
# modular-constructivist relief tradition of the Pattern Engine (the
# `patterns` package).
#
# The strapwork can further be made to WEAVE over and under itself, as
# true Islamic / Celtic knotwork does.  At every crossing one strand is
# assigned OVER and the other UNDER so that following any single band the
# sense strictly alternates -- the alternating-knot condition -- decided
# by the classical checkerboard (medial-graph) construction, solved here
# in its equivalent "alternate along each strand" form.  Two renderings
# are offered: FLAT breaks each band where it goes under (2D interlaced
# knotwork), while WOVEN lifts and dips each band at its crossings into a
# genuine 3D woven surface.  This over/under assignment is exactly how
# Celtic and Islamic interlace choose which strand rides on top.
#
# References:
#   E. H. Hankin, "The Drawing of Geometric Patterns in Saracenic Art"
#     (Memoirs of the Archaeological Survey of India, no. 15, 1925) --
#     the "polygons in contact" construction.
#   Craig S. Kaplan, "Computer Generated Islamic Star Patterns"
#     (Bridges 2000) -- the algorithmic PIC formulation reused here.
#   Craig S. Kaplan, "Islamic Star Patterns from Polygons in Contact"
#     (Graphics Interface 2005) -- motif inference and the rosette.
#   Craig S. Kaplan, "Computer Graphics and Geometric Ornamental
#     Design" (Ph.D. dissertation, University of Washington, 2002) --
#     the rosette transform and inference algorithms implemented here.
#   A. J. Lee, "Islamic Star Patterns" (Muqarnas 4, 1987) -- the
#     traditional rosette (star + petal ring) construction.
#   Jay Bonner, "Islamic Geometric Patterns: Their Historical
#     Development and Traditional Methods of Construction" (2017).
#   Branko Grunbaum & G. C. Shephard, "Tilings and Patterns" (1987) --
#     the uniform tilings used as PIC substrates.
#   Peter J. Lu & Paul J. Steinhardt, "Decagonal and Quasi-Crystalline
#     Tilings in Medieval Islamic Architecture" (Science 315, 2007) --
#     the girih tiles and the quasiperiodic 10-fold strapwork, realized
#     here as polygons-in-contact decoration of a Penrose quasilattice.
#   Peter R. Cromwell, "The Search for Quasi-Periodicity in Islamic Art"
#     (Mathematical Intelligencer 31, 2009) -- girih construction.

bl_info = {
    "name": "Islamic Star Pattern",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Polygons-in-contact Islamic star patterns "
                   "(strapwork ribbons, over/under interlacing, "
                   "optional relief)",
    "category": "Add Mesh",
}

from math import cos, sin, radians, hypot, pi, tan, atan2
import numpy as np

# --- the interlace engine moved to the patterns package ---------------
# Mitred ribbons, band faces, strand smoothing and the over/under solver
# were written here and then reached for PRIVATELY by celtic_knot_2d,
# knot_carpet, substitution_knot and fractal_knotwork -- about eighty
# call sites into this module's underscore names.  They now live in
# `patterns.ribbon`, `patterns.overunder` and `patterns.polygon2d`, with
# public names.  They are re-bound to their old private names here so
# this module's own several hundred call sites read unchanged.
try:
    from .patterns.polygon2d import arclen as _arclen
    from .patterns.polygon2d import line_intersection as _line_line
    from .patterns.polygon2d import unit as _unit
    from .patterns.ribbon import (angle_cut_piece as _angle_cut_piece,
                                  band_ribbon_faces, band_ribbon_faces_z,
                                  catmull_rom, cut_band as _cut_band,
                                  cut_cap_on_edge as _cut_cap_on_edge,
                                  miter as _miter, miter_ribbon)
    from .patterns.overunder import ParityDSU as _ParityDSU
    from .patterns.overunder import weave_zoff as _weave_zoff
except ImportError:                   # flat import (test runner)
    from patterns.polygon2d import arclen as _arclen
    from patterns.polygon2d import line_intersection as _line_line
    from patterns.polygon2d import unit as _unit
    from patterns.ribbon import (angle_cut_piece as _angle_cut_piece,
                                 band_ribbon_faces, band_ribbon_faces_z,
                                 catmull_rom, cut_band as _cut_band,
                                 cut_cap_on_edge as _cut_cap_on_edge,
                                 miter as _miter, miter_ribbon)
    from patterns.overunder import ParityDSU as _ParityDSU
    from patterns.overunder import weave_zoff as _weave_zoff


try:
    from .patterns import common as pc
    from . import tiling_generator as tg
    from . import aperiodic_generator as ap
except Exception:                       # legacy single-file / CLI use
    from patterns import common as pc
    import tiling_generator as tg
    import aperiodic_generator as ap


# --------------------------------------------------------------------
# Substrates: reuse tiling_generator's regular-polygon tilings
# --------------------------------------------------------------------
#
# Every substrate here is one of tiling_generator's base tilings (all
# regular polygons, edge length 1, sharing whole edges), so the PIC
# construction lands clean contact points on the shared edges.

SUBSTRATE_ITEMS = [
    ('SQUARE', "Square (4.4.4.4)",
     "Square grid -> 4/8-point stars"),
    ('TRI', "Triangular (3.3.3.3.3.3)",
     "Triangular tiling -> 3/6-point stars"),
    ('HEX', "Hexagonal (6.6.6)",
     "Hexagonal tiling -> 6/12-point stars"),
    ('TRIHEX', "Trihexagonal (3.6.3.6)",
     "Hexagon-triangle tiling -> 3/6-point stars"),
    ('SNUBSQ', "Snub Square (3.3.4.3.4)",
     "Snub-square Archimedean tiling -> mixed 4/8-point stars"),
    ('SNUBHEX', "Snub Hexagonal (3.3.3.3.6)",
     "Snub-hexagonal Archimedean tiling -> 6-fold stars"),
    ('ELONGTRI', "Elongated Triangular (3.3.3.4.4)",
     "Triangle-square Archimedean tiling"),
    ('RHOMBITRIHEX', "Rhombitrihexagonal (3.4.6.4)",
     "Hexagon-square-triangle tiling -> 6/12-point stars"),
    ('TRUNCSQ', "Truncated Square (4.8.8)",
     "Octagon-square tiling -> the iconic 8-point star-and-cross"),
    ('TRUNCHEX', "Truncated Hexagonal (3.12.12)",
     "Dodecagon-triangle tiling -> 12-point stars"),
    ('TRUNCTRIHEX', "Truncated Trihexagonal (4.6.12)",
     "Dodecagon-hexagon-square tiling -> 12-point stars"),
    ('RHOMBILLE', "Rhombille (dual 3.6.3.6)",
     "Irregular rhombi -- needs motif inference"),
    ('CAIRO', "Cairo Pentagonal (dual 3.3.4.3.4)",
     "Irregular pentagons -- star per tile"),
    ('FLORET', "Floret Pentagonal (dual 3.3.3.3.6)",
     "Irregular pentagons -- star per tile"),
    ('PRISMATIC', "Prismatic Pentagonal (dual 3.3.3.4.4)",
     "Irregular pentagons -- star per tile"),
    ('DELTOIDAL', "Deltoidal Trihexagonal (dual 3.4.6.4)",
     "Irregular kites -- vertex stars"),
    ('TRIAKIS', "Triakis Triangular (dual 3.12.12)",
     "Irregular triangles -- vertex stars"),
    ('TETRAKIS', "Tetrakis Square (dual 4.8.8)",
     "Irregular right triangles -- vertex stars"),
    ('KISRHOMBILLE', "Kisrhombille (dual 4.6.12)",
     "Irregular 30-60-90 triangles -- vertex stars"),
    ('GIRIH', "Girih (quasiperiodic 10-fold)",
     "Penrose quasilattice, girih strapwork -> 10-fold stars, no "
     "repeating unit cell (Lu & Steinhardt 2007)"),
]

# Substrates whose tiles are irregular polygons (the Laves duals): these
# build star-forming strapwork rather than the regular-polygon PIC/rosette
# -- per tile where the tiles are large enough to host a star (the
# pentagonal duals), per vertex where the tiles are triangles or quads
# (see `_irregular_motifs`).  Drawn straight from tiling_generator's shared
# dual construction, so every Laves tiling is available.
_LAVES_SUBSTRATES = tuple(tg.LAVES_OF.keys())

# The Laves duals whose tiles are 3- or 4-gons (too small to host their own
# star): their stars sit at the tiling VERTICES instead of per tile.  The
# pentagonal duals (CAIRO / FLORET / PRISMATIC) host a star per tile.
_VERTEX_STAR_LAVES = ('RHOMBILLE', 'DELTOIDAL', 'TRIAKIS', 'TETRAKIS',
                      'KISRHOMBILLE')

# Contact angle (degrees) giving the classic motif for each substrate.
# Regular / Archimedean substrates not listed here fall back to 30 deg;
# the irregular (Laves) substrates ignore the contact angle (their star
# strapwork is anchored on the shared edge midpoints).
DEFAULT_CONTACT = {
    'SQUARE': 30.0, 'TRI': 30.0, 'HEX': 30.0,
    'TRIHEX': 30.0, 'SNUBSQ': 30.0, 'SNUBHEX': 30.0,
    'ELONGTRI': 30.0, 'RHOMBITRIHEX': 30.0,
    'TRUNCSQ': 30.0, 'TRUNCHEX': 30.0, 'TRUNCTRIHEX': 30.0,
    'RHOMBILLE': 35.0, 'CAIRO': 35.0, 'FLORET': 35.0,
    'PRISMATIC': 35.0, 'DELTOIDAL': 35.0, 'TRIAKIS': 35.0,
    'TETRAKIS': 35.0, 'KISRHOMBILLE': 35.0, 'GIRIH': 72.0,
}

# Historical presets: (substrate, contact angle, motif, star_d).
PRESET_ITEMS = [
    ('CUSTOM', "Custom", "Use the substrate, motif and angle below"),
    ('STARCROSS8', "8-fold Star-and-Cross (4.8.8)",
     "The classic octagon star with square crosses"),
    ('TWELVE', "12-fold Star (3.12.12)",
     "Twelve-point dodecagon stars"),
    ('SIX', "6-fold Star (hexagonal)",
     "Six-point stars on the hexagonal tiling"),
    ('FOUREIGHT', "4/8 Star (square)",
     "Eight-point stars on the square grid"),
    ('ROSETTE8', "8-fold Rosette (4.8.8)",
     "Octagonal rosettes -- star ringed by eight petals"),
    ('ROSETTE12', "12-fold Rosette (3.12.12)",
     "The iconic twelve-petalled dodecagon rosette"),
    ('ROSETTE12B', "12-fold Rosette (4.6.12)",
     "Twelve-fold rosettes on the 4.6.12 tiling"),
    ('STAR8', "8-point Star {8/3} (4.8.8)",
     "Regular star polygons in the octagons"),
    ('INFER_RHOMBI', "Star Weave (Rhombille)",
     "Six-point vertex stars across irregular rhombi"),
    ('INFER_CAIRO', "Star Weave (Cairo)",
     "Five-point stars across irregular pentagons"),
    ('GIRIH10', "Girih 10-fold (quasiperiodic)",
     "Quasiperiodic ten-fold stars on a Penrose girih quasilattice"),
]
# preset -> (substrate, contact angle, motif, star_d)
PRESETS = {
    'STARCROSS8':   ('TRUNCSQ', 30.0, 'PIC', 2),
    'TWELVE':       ('TRUNCHEX', 30.0, 'PIC', 2),
    'SIX':          ('HEX', 30.0, 'PIC', 2),
    'FOUREIGHT':    ('SQUARE', 30.0, 'PIC', 2),
    'ROSETTE8':     ('TRUNCSQ', 34.0, 'ROSETTE', 2),
    'ROSETTE12':    ('TRUNCHEX', 32.0, 'ROSETTE', 2),
    'ROSETTE12B':   ('TRUNCTRIHEX', 32.0, 'ROSETTE', 2),
    'STAR8':        ('TRUNCSQ', 30.0, 'STAR', 3),
    'INFER_RHOMBI': ('RHOMBILLE', 35.0, 'PIC', 2),
    'INFER_CAIRO':  ('CAIRO', 35.0, 'PIC', 2),
    'GIRIH10':      ('GIRIH', 72.0, 'PIC', 2),
}

# Material slot per polygon side count, for the STAR_ORDER color mode.
_ORDER_MAT = {3: 0, 4: 1, 5: 2, 6: 3, 8: 4, 12: 5}
# Backing slab uses the last palette slot (a neutral gray).
_BACKING_MAT = len(pc.PALETTE_RGBA) - 1


# --------------------------------------------------------------------
# Polygons in contact (Hankin / Kaplan)
# --------------------------------------------------------------------

def _ray_param(o0, d0, o1, d1):
    """Parameter s>=0 along ray o0 + s*d0 where it crosses ray
    o1 + t*d1 with t>=0, or None.  Both rays are half-lines."""
    det = d0[0] * (-d1[1]) - (-d1[0]) * d0[1]
    if abs(det) < 1e-12:
        return None
    bx, by = o1[0] - o0[0], o1[1] - o0[1]
    s = (bx * (-d1[1]) - (-d1[0]) * by) / det
    t = (d0[0] * by - bx * d0[1]) / det
    if s <= 1e-7 or t <= 1e-7:
        return None
    return s


def star_segments(poly, angle_deg):
    """PIC strapwork segments for one convex tile (Hankin / Kaplan).

    From each edge midpoint two rays leave into the tile, each making
    `angle_deg` with the edge (symmetric about the inward normal): one
    "left" ray turning toward the next vertex and one "right" ray
    toward the previous vertex.  Each ray is truncated at its nearest
    intersection with a ray of the OPPOSITE hand -- Kaplan's standard
    construction.  A left ray of one edge and a right ray of another
    are mirror images across a polygon symmetry axis, so they always
    meet mutually on that axis, knitting a clean star: small angles
    give grazing rays and blunt motifs; larger angles cross deeper
    into acute, many-pointed stars.  Returns ((x0, y0), (x1, y1))
    segments whose union is the closed motif, every segment starting
    at an edge midpoint (the contact point that guarantees continuity
    across the shared edge).
    """
    theta = radians(angle_deg)
    ct, st = cos(theta), sin(theta)
    poly = np.asarray(poly, float)
    n = len(poly)
    c = poly.mean(axis=0)
    left, right = [], []                         # (origin, direction)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        m = 0.5 * (a + b)
        e = b - a
        e = e / (hypot(e[0], e[1]) + 1e-12)      # along-edge unit
        nrm = c - m
        nrm = nrm / (hypot(nrm[0], nrm[1]) + 1e-12)   # inward unit
        left.append((m, ct * e + st * nrm))      # turns toward vtx i+1
        right.append((m, -ct * e + st * nrm))    # turns toward vtx i
    segs = []
    for src, others in ((left, right), (right, left)):
        for o0, d0 in src:
            best = None
            for o1, d1 in others:
                s = _ray_param(o0, d0, o1, d1)
                if s is not None and (best is None or s < best):
                    best = s
            if best is None:
                continue
            tip = (o0[0] + best * d0[0], o0[1] + best * d0[1])
            segs.append((tuple(o0), tip))
    return segs


# --------------------------------------------------------------------
# Motif inference (Kaplan) -- PIC generalized to irregular tiles
# --------------------------------------------------------------------

def _is_regular(poly, tol=1e-4):
    """True if `poly` is (numerically) a regular polygon: all edges the
    same length and all vertices equidistant from the centroid."""
    poly = np.asarray(poly, float)
    n = len(poly)
    if n < 3:
        return False
    c = poly.mean(axis=0)
    r = [hypot(*(poly[i] - c)) for i in range(n)]
    e = [hypot(*(poly[(i + 1) % n] - poly[i])) for i in range(n)]
    return (max(r) - min(r) < tol * max(r)
            and max(e) - min(e) < tol * max(e))


def infer_segments(poly, angle_deg):
    """Kaplan's motif inference: the PIC construction for an arbitrary
    (convex) tile, cast so the motif is a single CLOSED loop with no
    dangling ends.  From each edge midpoint two rays leave at exactly
    `angle_deg` to the edge (measured from the edge's TRUE normal, so a
    ray and its neighbour's mirror stay collinear across a shared edge --
    the contact stays continuous even on irregular tiles).  Rather than
    truncating each ray independently at its nearest crossing (which on
    an irregular tile leaves one ray ending in the middle of another --
    a stray stub), the rays are truncated MUTUALLY: around each vertex j
    the "left" ray of the preceding edge and the "right" ray of the
    following edge -- the two rays that both aim into the wedge at j --
    are cut at their common crossing, a shared node the band passes
    through.  Each edge midpoint therefore joins exactly two segments
    (one to each neighbouring vertex node) and the whole motif closes
    into one 2n-gon loop: continuous across shared edges, free of stubs."""
    theta = radians(angle_deg)
    ct, st = cos(theta), sin(theta)
    poly = np.asarray(poly, float)
    n = len(poly)
    c = poly.mean(axis=0)
    mids, left, right = [], [], []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        m = 0.5 * (a + b)
        e = b - a
        e = e / (hypot(e[0], e[1]) + 1e-12)
        nrm = np.array([-e[1], e[0]])            # true edge normal
        if np.dot(c - m, nrm) < 0.0:             # orient inward
            nrm = -nrm
        mids.append((float(m[0]), float(m[1])))
        left.append((ct * e[0] + st * nrm[0],    # turns toward vertex i+1
                     ct * e[1] + st * nrm[1]))
        right.append((-ct * e[0] + st * nrm[0],  # turns toward vertex i
                      -ct * e[1] + st * nrm[1]))
    # Every left ray is matched to a right ray at their crossing, both
    # truncated MUTUALLY there (a shared node), so each midpoint joins
    # exactly two segments and the motif closes with no stray stubs.  The
    # matching is greedy shortest-first, so rays pair with their nearest
    # partner and the motif hugs the tile (bounded, star-like) rather than
    # shooting off to a distant near-parallel crossing.
    crs = []
    for i in range(n):
        for k in range(n):
            s = _ray_param(mids[i], left[i], mids[k], right[k])
            if s is None:
                continue
            t = _ray_param(mids[k], right[k], mids[i], left[i])
            if t is None:
                continue
            crs.append((max(s, t), i, k,
                        (mids[i][0] + s * left[i][0],
                         mids[i][1] + s * left[i][1])))
    crs.sort()
    lused = [False] * n
    rused = [False] * n
    tip_of_left = [None] * n
    tip_of_right = [None] * n
    for _d, i, k, p in crs:
        if lused[i] or rused[k]:
            continue
        lused[i] = rused[k] = True
        tip_of_left[i] = tip_of_right[k] = p
    # Any ray left unmatched (rare, non-convex cells): pair the leftovers
    # on their infinite-line crossing so no midpoint is left dangling.
    li = [i for i in range(n) if not lused[i]]
    ri = [k for k in range(n) if not rused[k]]
    for i in li:
        best = None
        for k in ri:
            if rused[k]:
                continue
            p = _line_line(mids[i], (mids[i][0] + left[i][0],
                                     mids[i][1] + left[i][1]),
                           mids[k], (mids[k][0] + right[k][0],
                                     mids[k][1] + right[k][1]))
            if p is None:
                continue
            dd = hypot(p[0] - mids[i][0], p[1] - mids[i][1])
            if best is None or dd < best[0]:
                best = (dd, k, p)
        if best is not None:
            _dd, k, p = best
            rused[k] = True
            tip_of_left[i] = tip_of_right[k] = p
    segs = []
    for i in range(n):
        if tip_of_left[i] is not None:
            segs.append((mids[i], tip_of_left[i]))
        if tip_of_right[i] is not None:
            segs.append((mids[i], tip_of_right[i]))
    return segs


# --------------------------------------------------------------------
# Rosette (Kaplan's rosette transform) -- star + petal ring
# --------------------------------------------------------------------
#
# A regular n-gon is subdivided into a central n-gon P' (radius r',
# rotated by pi/n so its vertices point at P's edge midpoints) plus a
# ring of n pentagons, each spanning one vertex of P.  Applying the PIC
# / inference construction to those cells yields the classic rosette:
# the inner n-gon carries the central star, the pentagons carry the
# petals.  Every original edge of P is split at its own midpoint, so a
# rosette tile's boundary contacts (the quarter-points of P's edges)
# coincide with the neighbour's -- the strapwork stays continuous.

def _rosette_rprime(R, n):
    """Kaplan's inner radius: the connecting segments equal half of the
    polygon's side, giving a well-proportioned rosette."""
    d = pi / n
    return R * cos(d) - R * sin(d) * tan(pi * (n - 2) / (4.0 * n))


def rosette_cells(poly, frac=0.0):
    """Subdivide a regular n-gon into a central n-gon plus n pentagons
    (Kaplan's rosette transform).  `frac` > 0 overrides the inner-polygon
    radius as that fraction of the apothem; frac <= 0 uses Kaplan's
    proportion."""
    poly = np.asarray(poly, float)
    n = len(poly)
    c = poly.mean(axis=0)
    R = float(np.mean([hypot(*(poly[i] - c)) for i in range(n)]))
    mids = [0.5 * (poly[i] + poly[(i + 1) % n]) for i in range(n)]
    mang = [atan2(mids[i][1] - c[1], mids[i][0] - c[0]) for i in range(n)]
    a = R * cos(pi / n)                           # apothem
    rp = a * frac if frac > 0.0 else _rosette_rprime(R, n)
    rp = max(1e-3, min(rp, 0.98 * a))
    W = [c + np.array([rp * cos(mang[k]), rp * sin(mang[k])])
         for k in range(n)]
    cells = [np.array(W)]                          # central n-gon
    for k in range(n):                             # ring pentagons
        k1 = (k + 1) % n
        cells.append(np.array([mids[k], poly[k1], mids[k1], W[k1], W[k]]))
    return cells


def rosette_segments(poly, angle_deg, frac=0.0):
    """Strapwork segments of the rosette motif for a regular tile.  The
    central n-gon carries a crisp regular star polygon (its tips anchored
    at the cell's edge midpoints, which are exactly the contact points
    the surrounding pentagons infer to -- so the star knits to the petal
    ring), and each pentagon is filled by motif inference to make the
    petals."""
    cells = rosette_cells(poly, frac)
    n = len(cells[0])
    core_d = max(2, min((n - 1) // 2, n // 5 + 2))
    segs = list(star_polygon_segments(cells[0], core_d))
    for cell in cells[1:]:
        segs.extend(infer_segments(cell, angle_deg))
    return segs


# --------------------------------------------------------------------
# Regular star polygon {n/d} motif
# --------------------------------------------------------------------



def star_polygon_segments(poly, d):
    """A regular star polygon {n/d} inscribed in a tile with its n tips
    anchored at the edge midpoints (the contact points, so it stays
    continuous across shared edges).  `d` is the star's density: the
    concave (inner) vertices are the crossings of the {n/d} chords, so
    larger d gives sharper, more overlapping points.  Returns the 2n
    outline segments; falls back to inference if d is out of range."""
    poly = np.asarray(poly, float)
    n = len(poly)
    d = int(d)
    if n < 5 or d < 2 or d > (n - 1) // 2:
        return infer_segments(poly, 30.0)
    tips = [tuple(0.5 * (poly[i] + poly[(i + 1) % n])) for i in range(n)]
    inner = []
    for k in range(n):
        p = _line_line(tips[k], tips[(k + d) % n],
                       tips[(k + 1) % n], tips[(k + 1 - d) % n])
        if p is None:
            return infer_segments(poly, 30.0)
        inner.append(p)
    segs = []
    for k in range(n):
        segs.append((tips[k], inner[k]))
        segs.append((inner[k], tips[(k + 1) % n]))
    return segs


def _star_outline_from_tips(tips, d):
    """The outline of a regular-ish star polygon {n/d} through the given
    ordered `tips` (any placement, not necessarily on a circle).  Each
    concave (inner) vertex is the crossing of the two {n/d} chords that
    span the gap between consecutive tips, so the result is a single
    closed 2n-gon: n outer points and n inner notches, every vertex of
    degree two -- dangling-free by construction.  With `d` < 2 (or too
    large, or a non-crossing gap) it degenerates gracefully to the convex
    ring {n/1} through the tips."""
    tips = [(float(t[0]), float(t[1])) for t in tips]
    n = len(tips)
    d = int(d)
    if n < 3:
        return []
    if n < 5 or d < 2 or d > (n - 1) // 2:
        return [(tips[k], tips[(k + 1) % n]) for k in range(n)]
    inner = []
    for k in range(n):
        p = _line_line(tips[k], tips[(k + d) % n],
                       tips[(k + 1) % n], tips[(k + 1 - d) % n])
        if p is None:
            return [(tips[k], tips[(k + 1) % n]) for k in range(n)]
        inner.append(p)
    segs = []
    for k in range(n):
        segs.append((tips[k], inner[k]))
        segs.append((inner[k], tips[(k + 1) % n]))
    return segs


def _star_density(n):
    """A pleasing star density for an n-pointed star (matches the rosette
    core rule): 2 for pentagons/hexagons, growing slowly with n."""
    return max(2, min((n - 1) // 2, n // 5 + 2))


def _tile_star_motifs(polys):
    """Per-tile star motifs for an irregular substrate whose tiles can
    each host a star polygon (n >= 5, e.g. the Cairo pentagons): a
    regular-ish {n/d} star with its n tips anchored at the tile's edge
    midpoints.  Because the tips sit on the shared edge midpoints, each
    tile's star kisses its neighbour's at those points and the strapwork
    stays continuous; each star is a closed loop, so there are no stubs.
    Returns (rep_poly, order, segments) per tile."""
    out = []
    for poly in polys:
        poly = np.asarray(poly, float)
        n = len(poly)
        tips = [tuple(0.5 * (poly[i] + poly[(i + 1) % n]))
                for i in range(n)]
        d = _star_density(n)
        segs = _star_outline_from_tips(tips, d)
        if segs:
            out.append((poly, n, segs))
    return out


def _vertex_star_motifs(polys):
    """Vertex-centered star motifs for an irregular substrate whose tiles
    are too small to host a star (e.g. the Rhombille rhombi).  A star
    sits at every fully surrounded tiling VERTEX, its tips anchored at the
    midpoints of the edges radiating from that vertex; high-valence
    vertices (the 6-valent Rhombille corners) become six-point stars, the
    3-valent corners small triangles.  Every edge midpoint is a tip of the
    star at each of its two endpoints, so the stars kiss at the midpoints
    and the strapwork is continuous; each star is a closed loop, so there
    are no stubs.  Returns (rep_poly, order, segments) per vertex star."""
    vs = _NodeSet(1e-6)
    vcoord = {}
    edge_count = {}
    vinc = {}
    for poly in polys:
        poly = np.asarray(poly, float)
        n = len(poly)
        ids = [vs.add((float(poly[i][0]), float(poly[i][1])))
               for i in range(n)]
        for i in range(n):
            ia, ib = ids[i], ids[(i + 1) % n]
            key = (ia, ib) if ia < ib else (ib, ia)
            edge_count[key] = edge_count.get(key, 0) + 1
            m = (float(0.5 * (poly[i][0] + poly[(i + 1) % n][0])),
                 float(0.5 * (poly[i][1] + poly[(i + 1) % n][1])))
            vinc.setdefault(ia, {})[key] = m
            vinc.setdefault(ib, {})[key] = m
            vcoord[ia] = (float(poly[i][0]), float(poly[i][1]))
            vcoord[ib] = (float(poly[(i + 1) % n][0]),
                          float(poly[(i + 1) % n][1]))
    out = []
    for vid, edges in vinc.items():
        if any(edge_count[k] != 2 for k in edges):
            continue                              # boundary / incomplete
        cx, cy = vcoord[vid]
        tips = sorted(edges.values(),
                      key=lambda p: atan2(p[1] - cy, p[0] - cx))
        m = len(tips)
        if m < 3:
            continue
        d = _star_density(m) if m >= 5 else 1
        segs = _star_outline_from_tips(tips, d)
        if segs:
            out.append((np.asarray(tips, float), m, segs))
    return out


def _irregular_motifs(polys, substrate):
    """Star-forming strapwork for an irregular (Laves-dual) substrate.
    Duals with pentagonal tiles (Cairo / Floret / Prismatic) host a star
    PER TILE; duals whose tiles are triangles or quads (Rhombille and the
    kite / triangle duals -- too small to carry their own star) place a
    star at every tiling VERTEX instead.  Both routes emit closed star
    outlines whose tips land on shared edge midpoints -- authentic star
    patterns with continuous, stub-free strapwork, unlike the old edge-to-
    edge inference weave."""
    if substrate in _VERTEX_STAR_LAVES:
        return _vertex_star_motifs(polys)
    return _tile_star_motifs(polys)


# --------------------------------------------------------------------
# Girih -- quasiperiodic 10-fold strapwork (Lu & Steinhardt 2007)
# --------------------------------------------------------------------
#
# The girih ("knot") tiles decorate a QUASIPERIODIC 10-fold tiling, so
# unlike the periodic substrates above the pattern never repeats.  We
# realize it on a Penrose (P3) quasilattice -- the two rhombi Lu &
# Steinhardt show are equivalent to the girih tiling -- inflated from
# aperiodic_generator's five-fold sun seed, every tile sharing one edge
# length.  Each rhomb is decorated with Hankin polygons-in-contact lines
# crossing each edge midpoint at a fixed contact angle (the 72/108/144
# girih geometry); because the contacts land on shared edge midpoints the
# strapwork is continuous across the whole patch, and where five and ten
# tiles meet it closes into the characteristic ten-point stars and
# decagonal rosettes (crispest near a 72-degree contact on this rhombic
# quasilattice, the default).  It then flows through the same continuous-
# band pipeline, inheriting mitered / curved ribbons, BAND color,
# interlace and relief.

def _girih_polys(gens):
    """The rhombi of a Penrose (P3) quasilattice inflated `gens` times
    from the five-fold sun seed -- a finite quasiperiodic 10-fold patch
    of equal-edge tiles for girih decoration.  Boundary half-triangles
    (unequal edges) are dropped so every contact lands on a shared edge
    midpoint, and the whole patch is scaled to unit edge length to match
    the periodic substrates (so ribbon widths read the same)."""
    tiles = [np.asarray(p, float)
             for _t, p in ap._p3_rhombs(max(0, int(gens))) if len(p) == 4]
    if not tiles:
        return tiles
    e = tiles[0]
    edge = hypot(e[1][0] - e[0][0], e[1][1] - e[0][1])
    s = 1.0 / edge if edge > 1e-12 else 1.0
    return [p * s for p in tiles]


def _girih_rect(polys, frac=0.60):
    """A centered trim rectangle over the inner `frac` of the girih patch,
    dropping the ragged quasilattice boundary so the strapwork reads as a
    clean quasiperiodic field with no boundary stubs."""
    allv = np.vstack([np.asarray(p, float) for p in polys])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    c = 0.5 * (lo + hi)
    hw = 0.5 * (hi[0] - lo[0]) * frac
    hh = 0.5 * (hi[1] - lo[1]) * frac
    return (c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh)


def _girih_motifs(polys, contact_deg):
    """Girih strapwork for the quasilattice: every rhomb decorated by
    polygons-in-contact inference at the girih contact angle.  Inference
    is mutual / dangling-free, so the decorated patch is continuous and
    stub-free across all shared edges."""
    return [(p, len(p), infer_segments(p, contact_deg)) for p in polys]


def _augment_edges(poly):
    """Insert each edge's midpoint as an extra (straight-angle) vertex,
    doubling the edge count.  Inference then lands two contact points per
    original edge -- at its quarter-points -- matching a rosette
    neighbour, so small filler tiles knit to rosettes continuously."""
    poly = np.asarray(poly, float)
    n = len(poly)
    out = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        out.append(a)
        out.append(0.5 * (a + b))
    return np.array(out)


def _tile_segments(poly, contact_deg, motif, star_d, rosette_frac):
    """Strapwork segments for one tile under the chosen motif.  Regular
    tiles use the exact motif; any irregular tile falls back to motif
    inference, so mixed / non-regular substrates knit continuously."""
    reg = _is_regular(poly)
    n = len(poly)
    if not reg:
        return infer_segments(poly, contact_deg)
    if motif == 'ROSETTE':
        if n >= 5:
            return rosette_segments(poly, contact_deg, rosette_frac)
        # small filler tiles: simple motif, but split edges at their
        # midpoints so boundary contacts meet the rosette's quarter-points
        return infer_segments(_augment_edges(poly), contact_deg)
    if motif == 'STAR' and n >= 5:
        return star_polygon_segments(poly, star_d)
    return star_segments(poly, contact_deg)       # PIC (regular)


# --------------------------------------------------------------------
# Substrate + motif assembly
# --------------------------------------------------------------------

def _clip_segment(p0, p1, rect):
    """Liang-Barsky clip of segment p0->p1 to axis-aligned rectangle
    rect = (x0, y0, x1, y1).  Returns (a, b) clipped, or None."""
    x0, y0, x1, y1 = rect
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]),
                 (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if abs(p) < 1e-15:
            if q < 0.0:
                return None                       # parallel, outside
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None
            if r < t1:
                t1 = r
    if t0 > t1:
        return None
    a = (p0[0] + t0 * dx, p0[1] + t0 * dy)
    b = (p0[0] + t1 * dx, p0[1] + t1 * dy)
    if hypot(b[0] - a[0], b[1] - a[1]) < 1e-9:
        return None
    return a, b


def _placed_tiles(substrate, NX, NY):
    """(poly, tile_index) for every tile of a substrate over an NX x NY
    run.  Regular substrates come from tiling_generator's base tilings;
    the irregular Laves substrates from its dual construction."""
    if substrate in _LAVES_SUBSTRATES:
        polys, _types = tg._build_tiling(substrate, NX, NY)
        return [(np.asarray(p, float), i) for i, p in enumerate(polys)]
    return list(tg._base_iter(substrate, NX, NY))


def tile_motifs(substrate, nx, ny, contact_deg, trim=False, pad=2,
                motif='PIC', star_d=2, rosette_frac=0.0, girih_gens=3):
    """Return (motifs, rect): one (poly, order, segments) per substrate
    tile, where `order` is the polygon side count and `segments` are
    that tile's strapwork segments under the chosen `motif` (PIC,
    ROSETTE or STAR; irregular tiles fall back to inference), clipped to
    `rect` when trim.  `rect` is the trim rectangle or None.  The GIRIH
    substrate ignores nx/ny and instead builds a `girih_gens`-generation
    quasiperiodic Penrose patch."""
    if substrate == 'GIRIH':
        polys = _girih_polys(girih_gens)
        rect = _girih_rect(polys) if trim else None
        raw = _girih_motifs(polys, contact_deg)
        motifs = []
        for poly, order, segs in raw:
            if rect is not None:
                segs = [cs for cs in
                        (_clip_segment(a, b, rect) for a, b in segs)
                        if cs is not None]
            if not segs:
                continue
            motifs.append((np.asarray(poly, float), order, segs))
        return motifs, rect
    NX, NY = (nx + pad, ny + pad) if trim else (nx, ny)
    placed = _placed_tiles(substrate, NX, NY)
    rect = None
    if trim:
        rect = tg._trim_rect([p for p, _ in placed], nx, ny, pad)
    if substrate in _LAVES_SUBSTRATES:
        # Irregular substrates: build star-forming strapwork globally
        # (per tile for Cairo, per vertex for Rhombille) rather than the
        # old edge-to-edge inference weave.
        raw = _irregular_motifs([p for p, _ in placed], substrate)
    else:
        raw = [(np.asarray(poly, float), len(poly),
                _tile_segments(poly, contact_deg, motif, star_d,
                               rosette_frac))
               for poly, _ti in placed]
    motifs = []
    for poly, order, segs in raw:
        if rect is not None:
            clipped = []
            for a, b in segs:
                cs = _clip_segment(a, b, rect)
                if cs is not None:
                    clipped.append(cs)
            segs = clipped
        if not segs:
            continue
        motifs.append((np.asarray(poly, float), order, segs))
    return motifs, rect


# --------------------------------------------------------------------
# Strapwork arrangement: continuous bands across the whole patch
# --------------------------------------------------------------------
#
# The PIC construction hands us short line segments, one per ray, each
# starting at an edge midpoint (a contact point).  Rendering each
# segment as its own rectangle would leave the strapwork as disjoint
# quads.  Instead we glue the segments into an ARRANGEMENT -- endpoints
# snapped to shared nodes -- and trace the maximal "straightest-
# through" bands that weave from tile to tile: a band crosses each
# contact point straight (the two adjacent tiles' rays are collinear
# there) and bends at the star points, exactly the classic interlaced
# strapwork topology.  Each traced band is then built as ONE continuous
# mitered ribbon of shared vertices, not a pile of rectangles.



def _flatten_segments(motifs):
    """Flatten per-tile motifs into parallel lists (segments, orders,
    tiles): each strapwork segment with the side count and index of the
    tile whose PIC motif produced it (used for coloring)."""
    segments, orders, tiles = [], [], []
    for ti, (_poly, order, segs) in enumerate(motifs):
        for a, b in segs:
            segments.append((a, b))
            orders.append(order)
            tiles.append(ti)
    return segments, orders, tiles


class _NodeSet:
    """Snap 2D points to shared node ids within `tol` via a small
    spatial hash, so coincident contact points from neighboring tiles
    merge to one node (the seam that makes bands continuous)."""

    def __init__(self, tol=1e-6):
        self.tol = tol
        self.pts = []
        self.grid = {}

    def add(self, p):
        t = self.tol
        gx, gy = round(p[0] / t), round(p[1] / t)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in self.grid.get((gx + dx, gy + dy), ()):
                    q = self.pts[idx]
                    if abs(q[0] - p[0]) <= t and abs(q[1] - p[1]) <= t:
                        return idx
        idx = len(self.pts)
        self.pts.append((float(p[0]), float(p[1])))
        self.grid.setdefault((gx, gy), []).append(idx)
        return idx


def _pair_at_node(incident):
    """Pair a node's incident half-edges "straightest-through": match
    the two directions closest to anti-parallel first (greedy), so a
    band continues as straight as it can.  `incident` is a list of
    (seg_index, unit_direction_pointing_away_from_node).  Returns a
    dict seg_index -> partner seg_index; an unpaired segment is absent
    and terminates its band there (a flat cap)."""
    n = len(incident)
    pairs = []
    for i in range(n):
        di = incident[i][1]
        for j in range(i + 1, n):
            dj = incident[j][1]
            pairs.append((di[0] * dj[0] + di[1] * dj[1], i, j))
    pairs.sort()                                  # most anti-parallel first
    taken = [False] * n
    out = {}
    for _dot, i, j in pairs:
        if taken[i] or taken[j]:
            continue
        taken[i] = taken[j] = True
        out[incident[i][0]] = incident[j][0]
        out[incident[j][0]] = incident[i][0]
    return out


def build_arrangement(segments, tol=1e-6):
    """Snap all segment endpoints to shared nodes and pair the half-
    edges at every node.  Returns (pts, seg_nodes, pair): node
    coordinates, per-segment (na, nb) node ids (None for degenerate),
    and node -> {seg: partner_seg}."""
    ns = _NodeSet(tol)
    seg_nodes = []
    for a, b in segments:
        na, nb = ns.add(a), ns.add(b)
        seg_nodes.append(None if na == nb else (na, nb))
    incident = {}
    for si, nn in enumerate(seg_nodes):
        if nn is None:
            continue
        na, nb = nn
        pa, pb = ns.pts[na], ns.pts[nb]
        d = _unit(pb[0] - pa[0], pb[1] - pa[1])
        incident.setdefault(na, []).append((si, d))
        incident.setdefault(nb, []).append((si, (-d[0], -d[1])))
    pair = {node: _pair_at_node(lst) for node, lst in incident.items()}
    return ns.pts, seg_nodes, pair


def trace_bands(seg_nodes, pair):
    """Trace maximal bands through the arrangement.  Each band is
    (node_path, seg_path): the ordered node ids of its polyline and the
    segment ids it walks.  A closed band repeats its first node last."""
    def other(seg, node):
        a, b = seg_nodes[seg]
        return b if a == node else a

    used = [False] * len(seg_nodes)
    bands = []
    for s0 in range(len(seg_nodes)):
        if seg_nodes[s0] is None or used[s0]:
            continue
        a0, b0 = seg_nodes[s0]
        node_path, seg_path, local = [a0, b0], [s0], {s0}
        for start_node, forward in ((b0, True), (a0, False)):
            cur_seg, cur_node = s0, start_node
            while True:
                nxt = pair.get(cur_node, {}).get(cur_seg)
                if nxt is None or nxt in local:
                    break                          # dead end or closed loop
                nn = other(nxt, cur_node)
                if forward:
                    node_path.append(nn)
                    seg_path.append(nxt)
                else:
                    node_path.insert(0, nn)
                    seg_path.insert(0, nxt)
                local.add(nxt)
                cur_seg, cur_node = nxt, nn
        for s in local:
            used[s] = True
        bands.append((node_path, seg_path))
    return bands








def _band_kind(color_by, band_index, seg_path, orders):
    """Material index for a whole band, per color mode.  A band spans
    many tiles, so STAR_ORDER colors by the dominant star order along
    the band, TILE colors per band, and BAND gives every continuous
    ribbon its own palette color (band index modulo the palette)."""
    if color_by == 'STAR_ORDER':
        from collections import Counter
        order = Counter(orders[s] for s in seg_path).most_common(1)[0][0]
        return _ORDER_MAT.get(order, 6)
    if color_by == 'BAND':
        return band_index % len(pc.PALETTE_RGBA)  # a color per ribbon
    if color_by == 'TILE':
        return band_index % _BACKING_MAT          # keep clear of backing
    return 0                                      # UNIFORM






def strapwork_bands(substrate, nx, ny, contact_deg, ribbon_width,
                    trim=False, curved=False, subdiv=8,
                    motif='PIC', star_d=2, rosette_frac=0.0,
                    girih_gens=3):
    """Trace the continuous strapwork bands of a patch.  Returns
    (bands, orders): each band is
    (left, right, closed, seg_path, path) -- the mitered ribbon
    boundaries, whether it is a closed loop, the segment ids it walks,
    and the centerline polyline (Catmull-Rom smoothed when `curved`)."""
    motifs, _rect = tile_motifs(substrate, nx, ny, contact_deg, trim,
                                motif=motif, star_d=star_d,
                                rosette_frac=rosette_frac,
                                girih_gens=girih_gens)
    segments, orders, _tiles = _flatten_segments(motifs)
    if not segments:
        return [], orders
    pts, seg_nodes, pair = build_arrangement(segments)
    bands = []
    for node_path, seg_path in trace_bands(seg_nodes, pair):
        closed = len(node_path) >= 4 and node_path[0] == node_path[-1]
        poly = [pts[n] for n in node_path]
        if closed:
            poly = poly[:-1]
        if len(poly) < 2:
            continue
        path = catmull_rom(poly, closed, subdiv) if curved else poly
        left, right = miter_ribbon(path, ribbon_width, closed)
        bands.append((left, right, closed, seg_path, path))
    return bands, orders


# --------------------------------------------------------------------
# Over/under interlacing (weaving) -- alternating knotwork
# --------------------------------------------------------------------
#
# By default the traced strapwork lies perfectly flat: where two bands
# cross they simply share the plane.  Real Islamic (and Celtic) knotwork
# instead WEAVES -- at every crossing one strand rides OVER and the other
# passes UNDER, and following any single strand the sense strictly
# alternates over / under / over / under (the alternating-knot
# condition).  We reproduce that here as an option.
#
# A crossing is an interior 4-valent node whose "straightest-through"
# pairing joins its four half-edges into two through-strands (two
# different bands passing straight through each other).  To decide, at
# every crossing, which strand is over, we use the standard method that
# guarantees a globally consistent alternating diagram: it is equivalent
# to a checkerboard 2-coloring of the arrangement's faces (the medial-
# graph coloring).  We solve it directly as its single-strand dual --
# "alternate along every strand" -- as a system of XOR (parity)
# constraints between consecutive crossings on each band, resolved with a
# parity union-find.  Odd-cycle parity clashes (rare boundary cases) are
# dropped gracefully rather than raising.  This is exactly how Celtic /
# Islamic knotwork chooses its over/under.
#
# Two renderings are offered.  FLAT keeps the flat mesh but BREAKS each
# band where it goes under, leaving a gap a little wider than the over-
# band's ribbon, so the over-band reads as crossing on top -- classic
# 2D interlaced knotwork.  WOVEN gives every band a z-height that swings
# smoothly (smoothstep along arclength) to +weave_height where it is
# over a crossing and -weave_height where it is under, turning the
# ribbon into a genuine 3D woven surface (and undulating the extruded
# ribbon too when relief height is also set).
#
# References for the interlacing (in addition to Kaplan and Bonner
# above): the alternating-knot / checkerboard construction is the
# classical way Celtic and Islamic interlace assign over and under.






def _traced_patch(substrate, nx, ny, contact_deg, trim,
                  motif, star_d, rosette_frac, girih_gens=3):
    """(pts, seg_nodes, pair, traced, orders) for a patch: the snapped
    arrangement, its half-edge pairing, the traced bands and per-segment
    star orders.  None if the patch produced no strapwork."""
    motifs, _rect = tile_motifs(substrate, nx, ny, contact_deg, trim,
                                motif=motif, star_d=star_d,
                                rosette_frac=rosette_frac,
                                girih_gens=girih_gens)
    segments, orders, _tiles = _flatten_segments(motifs)
    if not segments:
        return None
    pts, seg_nodes, pair = build_arrangement(segments)
    traced = trace_bands(seg_nodes, pair)
    return pts, seg_nodes, pair, traced, orders


def _find_crossings(pair):
    """Crossing nodes of the arrangement: interior 4-valent nodes whose
    pairing joins the four half-edges into two through-strands.  Returns
    node -> (pairA, pairB), each pair a frozenset of the two segment ids
    of one straight-through strand."""
    crossings = {}
    for node, pd in pair.items():
        if len(pd) != 4:
            continue
        prs = []
        seen = set()
        for s0, s1 in pd.items():
            key = frozenset((s0, s1))
            if key not in seen:
                seen.add(key)
                prs.append(key)
        if len(prs) == 2 and all(len(p) == 2 for p in prs):
            crossings[node] = (prs[0], prs[1])
    return crossings


def _ref_pairs(crossings):
    """Deterministic reference orientation per crossing: (ref, other)
    dicts mapping node -> the pair chosen as the parity reference (the
    one holding the smallest segment id) and its complement."""
    ref, other = {}, {}
    for node, (pA, pB) in crossings.items():
        if min(pA) <= min(pB):
            ref[node], other[node] = pA, pB
        else:
            ref[node], other[node] = pB, pA
    return ref, other


def _band_geometry(pts, node_path, seg_path, curved, subdiv):
    """(closed, poly, path, stations, pair_here) for one traced band:
    the closed flag, its control polyline `poly`, the rendered `path`
    (Catmull-Rom densified when curved), the node id at every control
    point, and the two-segment strand-pair the band uses passing through
    each control point (None at an open end where the band caps)."""
    closed = len(node_path) >= 4 and node_path[0] == node_path[-1]
    poly = [pts[nd] for nd in node_path]
    if closed:
        poly = poly[:-1]
    m = len(poly)
    stations = node_path[:-1] if closed else list(node_path)
    pair_here = []
    if closed:
        L = len(seg_path)
        for k in range(m):
            pair_here.append(frozenset((seg_path[(k - 1) % L],
                                        seg_path[k])))
    else:
        for k in range(m):
            si = seg_path[k - 1] if k > 0 else None
            so = seg_path[k] if k < m - 1 else None
            pair_here.append(None if (si is None or so is None)
                             else frozenset((si, so)))
    path = (catmull_rom(poly, closed, subdiv) if curved
            else [(float(p[0]), float(p[1])) for p in poly])
    return closed, poly, path, stations, pair_here


def _solve_over(crossings, ref, other, band_seqs):
    """Solve the alternating over/under assignment.  `band_seqs` is one
    (seq, closed) per band, seq an ordered list of (node, a) at the
    band's crossings where a = 1 if the band uses the reference strand.
    Requiring the band to alternate over/under gives, between consecutive
    crossings, x_i XOR x_j == 1 XOR a_i XOR a_j (x = "reference is over").
    We resolve all such parity constraints with a union-find, then read
    off over[node] = ref (if x == 1) or other (if x == 0)."""
    dsu = _ParityDSU()
    for node in crossings:
        dsu.add(node)
    for seq, closed in band_seqs:
        L = len(seq)
        if L < 2:
            continue
        rng = range(L) if closed else range(L - 1)
        for i in rng:
            n0, a0 = seq[i]
            n1, a1 = seq[(i + 1) % L]
            dsu.union(n0, n1, 1 ^ a0 ^ a1)         # clash -> dropped
    over = {}
    for node in crossings:
        _r, x = dsu.find(node)
        over[node] = ref[node] if x == 1 else other[node]
    return over








# --------------------------------------------------------------------
# FLAT-interlace under-cord cut: cut ALONG the over band's edge
# --------------------------------------------------------------------
#
# When a band goes UNDER at a crossing its centerline is broken (see
# `_cut_band`) leaving a gap, so the over band reads on top.  A plain
# flat cap ends the under band PERPENDICULAR to its own direction, which
# does not line up with the over band it tucks beneath -- at an oblique
# crossing that leaves a triangular sliver of gap.  Instead the under
# band should be cut ALONG the over band's edge: the cap a straight
# segment parallel to (and flush against) the over band's side, so the
# under cord tucks cleanly beneath the over cord.  The helpers below take
# the over band's local tangent + half-width at the crossing and slide
# the under band's two rail cap points onto the over band's edge line.

def _strand_tangent(node, strand, seg_nodes, pts):
    """Unit through-direction at `node` of a strand (a pair of segment ids
    that meet there): the direction across the node from one neighbouring
    node to the other -- the over band's LOCAL tangent at the crossing,
    used to cut an under band flush along the over band's edge.  Returns
    None when degenerate."""
    try:
        sa, sb = tuple(strand)
    except ValueError:
        return None
    na, nb = seg_nodes[sa], seg_nodes[sb]
    if na is None or nb is None:
        return None
    fa = na[1] if na[0] == node else na[0]
    fb = nb[1] if nb[0] == node else nb[0]
    pa, pb = pts[fa], pts[fb]
    d = _unit(pb[0] - pa[0], pb[1] - pa[1])
    return None if d == (0.0, 0.0) else d






def _interlaced_cells(substrate, nx, ny, contact_deg, ribbon_width,
                      color_by, trim, height, backing, base,
                      curved, subdiv, motif, star_d, rosette_frac,
                      mode, weave_height, girih_gens=3):
    """Interlaced (woven) strapwork cells.  Traces the bands, assigns a
    consistent alternating over/under at every crossing, then renders
    either FLAT knotwork (under-bands broken at crossings) or a WOVEN 3D
    surface (bands rise/dip at crossings).  One (verts, faces, mats) cell
    per ribbon piece; an optional backing slab is appended."""
    tp = _traced_patch(substrate, nx, ny, contact_deg, trim,
                        motif, star_d, rosette_frac, girih_gens)
    if tp is None:
        return []
    pts, seg_nodes, pair, traced, orders = tp
    crossings = _find_crossings(pair)
    ref, other = _ref_pairs(crossings)

    step = subdiv if curved else 1
    bands, band_seqs = [], []
    for node_path, seg_path in traced:
        closed, poly, path, stations, pair_here = _band_geometry(
            pts, node_path, seg_path, curved, subdiv)
        if len(poly) < 2:
            continue
        cross = []                                 # (node, pair, path_idx)
        for k, nd in enumerate(stations):
            ph = pair_here[k]
            if nd in crossings and ph is not None and ph in crossings[nd]:
                cross.append((nd, ph, k * step))
        seq = [(nd, 1 if ph == ref[nd] else 0) for nd, ph, _pi in cross]
        band_seqs.append((seq, closed))
        bands.append((closed, path, seg_path, cross))

    over = _solve_over(crossings, ref, other, band_seqs)

    cells = []
    all_verts = []
    for bi, (closed, path, seg_path, cross) in enumerate(bands):
        kind = _band_kind(color_by, bi, seg_path, orders)
        signed = [(pi, 1 if ph == over[nd] else -1)
                  for nd, ph, pi in cross]
        if mode == 'WOVEN':
            zoff = _weave_zoff(path, closed, signed, weave_height)
            left, right = miter_ribbon(path, ribbon_width, closed)
            cv, cf = band_ribbon_faces_z(left, right, closed, height, zoff)
            if cf:
                cells.append((cv, cf, [kind] * len(cf)))
                all_verts.extend(cv)
        else:                                      # FLAT
            under = [pi for pi, sg in signed if sg < 0]
            ugeo = []
            if under:
                s, total = _arclen(path, closed)
                cut_s = sorted(s[pi] for pi in under)
                margin = max(0.02, 0.25 * ribbon_width)
                half = 0.5 * (ribbon_width + margin)
                pieces = _cut_band(path, closed, cut_s, half, s, total)
                # Over-band local tangent + half-width at each of this
                # band's under-crossings, so the under band is cut ALONG
                # the over band's edge (flush, parallel) instead of with a
                # perpendicular cap.  Ribbon width is uniform, so the over
                # band's half-width is ribbon_width / 2.
                h_o = 0.5 * ribbon_width
                gap = 0.5 * margin                 # hairline beyond the edge
                maxreach = 3.0 * ribbon_width
                cut_gate = 1.5 * half
                for nd, ph, _pi in cross:
                    if ph == over[nd]:
                        continue                   # this band rides OVER here
                    t_o = _strand_tangent(nd, over[nd], seg_nodes, pts)
                    if t_o is None:
                        continue
                    ugeo.append((pts[nd], t_o, (-t_o[1], t_o[0])))
            else:
                pieces = [(path, closed)]
            npieces = len(pieces)
            for pj, (sp, sp_closed) in enumerate(pieces):
                if len(sp) < 2:
                    continue
                left, right = miter_ribbon(sp, ribbon_width, sp_closed)
                if ugeo and not sp_closed:
                    # For an open band the first piece's start and the last
                    # piece's end are genuine band termini, not cuts; every
                    # other piece end is an interlace cut.  A cut loop
                    # (closed band) has cuts at both ends of every piece.
                    start_struct = closed or pj != 0
                    end_struct = closed or pj != npieces - 1
                    _angle_cut_piece(left, right, sp, start_struct,
                                     end_struct, ugeo, h_o, gap,
                                     maxreach, cut_gate)
                cv, cf = band_ribbon_faces(left, right, sp_closed, height)
                if cf:
                    cells.append((cv, cf, [kind] * len(cf)))
                    all_verts.extend(cv)

    if backing and all_verts:
        a = np.asarray(all_verts, float)
        lo = (a[:, 0].min(), a[:, 1].min())
        hi = (a[:, 0].max(), a[:, 1].max())
        cv, cf, cm = [], [], []
        pc.slab(cv, cf, cm, lo, hi, 0.0, -base, mat=_BACKING_MAT)
        cells.append((cv, cf, cm))
    return cells


def build_cells(substrate, nx, ny, contact_deg, ribbon_width,
                color_by='UNIFORM', trim=False, height=0.0,
                backing=False, base=0.08, curved=False, subdiv=8,
                motif='PIC', star_d=2, rosette_frac=0.0,
                interlace=False, interlace_mode='FLAT',
                weave_height=0.05, girih_gens=3):
    """One (verts, faces, mats) cell per continuous strapwork BAND: the
    band as a single mitered ribbon (flat, or extruded to `height`;
    straight-mitered or Catmull-Rom curved).  An optional backing slab
    is appended as a final cell.  With `interlace` the bands instead
    weave over/under at crossings (FLAT knotwork or a WOVEN 3D surface)."""
    if interlace:
        return _interlaced_cells(
            substrate, nx, ny, contact_deg, ribbon_width, color_by,
            trim, height, backing, base, curved, subdiv, motif,
            star_d, rosette_frac, interlace_mode, weave_height,
            girih_gens)
    bands, orders = strapwork_bands(
        substrate, nx, ny, contact_deg, ribbon_width, trim,
        curved, subdiv, motif, star_d, rosette_frac, girih_gens)
    cells = []
    all_verts = []
    for bi, (left, right, closed, seg_path, _path) in enumerate(bands):
        cv, cf = band_ribbon_faces(left, right, closed, height)
        if not cf:
            continue
        kind = _band_kind(color_by, bi, seg_path, orders)
        cm = [kind] * len(cf)
        cells.append((cv, cf, cm))
        all_verts.extend(cv)
    if backing and all_verts:
        a = np.asarray(all_verts, float)
        lo = (a[:, 0].min(), a[:, 1].min())
        hi = (a[:, 0].max(), a[:, 1].max())
        cv, cf, cm = [], [], []
        pc.slab(cv, cf, cm, lo, hi, 0.0, -base, mat=_BACKING_MAT)
        cells.append((cv, cf, cm))
    return cells


def build(substrate, nx, ny, contact_deg, ribbon_width,
          color_by='UNIFORM', trim=False, height=0.0,
          backing=False, base=0.08, curved=False, subdiv=8,
          motif='PIC', star_d=2, rosette_frac=0.0,
          interlace=False, interlace_mode='FLAT', weave_height=0.05,
          girih_gens=3):
    """Merged (verts, faces, mats) for one strapwork patch."""
    return pc.merge_cells(build_cells(
        substrate, nx, ny, contact_deg, ribbon_width, color_by,
        trim, height, backing, base, curved, subdiv,
        motif, star_d, rosette_frac,
        interlace, interlace_mode, weave_height, girih_gens))


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _emit_curve(context, name, bands, span=2.0, operator=None):
        """Build a single Blender curve object whose POLY splines are
        the band centerlines (one spline per continuous ribbon path),
        centered and scaled to the `span` cube.  Placed via the
        operator's Align, matching the mesh path."""
        paths = [(list(path), closed)
                 for (_l, _r, closed, _sp, path) in bands
                 if len(path) >= 2]
        if not paths:
            return None
        allv = np.asarray([p for path, _c in paths for p in path], float)
        lo, hi = allv.min(axis=0), allv.max(axis=0)
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        s = span / max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        for path, closed in paths:
            sp = cu.splines.new('POLY')
            sp.points.add(len(path) - 1)
            for i, (x, y) in enumerate(path):
                sp.points[i].co = ((x - cx) * s, (y - cy) * s, 0.0, 1.0)
            sp.use_cyclic_u = bool(closed)
        obj = bpy.data.objects.new(name, cu)
        context.collection.objects.link(obj)
        if operator is not None:
            from bpy_extras import object_utils
            obj.matrix_world = object_utils.add_object_align_init(
                context, operator)
        else:
            obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    class MESH_OT_islamic_pattern_add(bpy.types.Operator,
                                      AddObjectHelper):
        """Add an Islamic star pattern (polygons in contact)"""
        bl_idname = "mesh.islamic_pattern_add"
        bl_label = "Islamic Star Pattern"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset", items=PRESET_ITEMS, default='STARCROSS8',
            description="Historical star pattern; Custom exposes the "
                        "substrate and contact angle")
        substrate: EnumProperty(
            name="Substrate", items=SUBSTRATE_ITEMS, default='TRUNCSQ',
            description="Underlying tiling; the last two (Laves duals) "
                        "are irregular and rely on motif inference")
        motif: EnumProperty(
            name="Motif",
            items=[('PIC', "Polygons in Contact",
                    "Hankin's star motif (inference on irregular tiles)"),
                   ('ROSETTE', "Rosette",
                    "Classic rosette: an n-star ringed by n petals"),
                   ('STAR', "Star Polygon {n/d}",
                    "A regular star polygon inscribed in each tile")],
            default='PIC',
            description="How each tile's strapwork is generated")
        contact_angle: FloatProperty(
            name="Contact Angle", default=30.0, min=10.0, max=80.0,
            description="Angle (degrees) of the rays to each edge; the "
                        "primary knob -- small = spiky, large = obtuse")
        star_d: IntProperty(
            name="Star Density d", default=3, min=2, max=6,
            description="Density of the {n/d} star polygon (STAR motif); "
                        "larger d gives sharper, more overlapping points")
        rosette_frac: FloatProperty(
            name="Rosette Core", default=0.4, min=0.0, max=0.95,
            description="Inner-polygon radius of the rosette as a "
                        "fraction of the apothem (controls the central "
                        "star size); 0 = Kaplan's proportion")
        nx: IntProperty(name="Cells X", default=5, min=1, max=40)
        ny: IntProperty(name="Cells Y", default=5, min=1, max=40)
        girih_gens: IntProperty(
            name="Girih Generations", default=4, min=1, max=6,
            description="Inflation depth of the quasiperiodic Penrose "
                        "girih patch (GIRIH substrate); higher = larger "
                        "patch, more tiles")
        trim: BoolProperty(
            name="Trim Boundary", default=True,
            description="Clip the strapwork to a clean central "
                        "rectangle, removing the ragged edge")
        ribbon_width: FloatProperty(
            name="Ribbon Width", default=0.14, min=0.01, max=0.6,
            description="Width of the strapwork ribbons (edge units)")
        color_by: EnumProperty(
            name="Color By",
            items=[('UNIFORM', "Uniform", "A single material"),
                   ('TILE', "By Tile",
                    "Material per substrate polygon"),
                   ('STAR_ORDER', "By Star Order",
                    "Material per polygon side count (star order)"),
                   ('BAND', "By Ribbon",
                    "A distinct color per continuous strapwork band")],
            default='STAR_ORDER')
        curved: BoolProperty(
            name="Curved Ribbons", default=False,
            description="Flow each band along a Catmull-Rom spline "
                        "through its nodes instead of straight segments")
        smoothness: IntProperty(
            name="Smoothness", default=8, min=2, max=32,
            description="Spline subdivisions per band segment "
                        "(curved ribbons only)")
        output: EnumProperty(
            name="Output",
            items=[('RIBBON', "Ribbon Mesh",
                    "Filled strapwork ribbons (supports relief)"),
                   ('CURVE', "Centerline Curves",
                    "Band centerlines as a Blender curve object")],
            default='RIBBON',
            description="Build filled ribbon meshes or the band "
                        "centerlines as curve splines")
        interlace: BoolProperty(
            name="Interlace (Weave)", default=False,
            description="Weave the bands over and under at crossings "
                        "(alternating knotwork); off = flat strapwork")
        interlace_mode: EnumProperty(
            name="Interlace Mode",
            items=[('FLAT', "Flat Knotwork",
                    "Break the under-band at each crossing so the "
                    "over-band reads on top (2D interlace, height 0)"),
                   ('WOVEN', "Woven (3D)",
                    "Raise each band where it is over and dip it where "
                    "under, into a genuine 3D woven surface")],
            default='FLAT',
            description="How the over/under weave is rendered")
        weave_height: FloatProperty(
            name="Weave Height", default=0.05, min=0.0, max=0.5,
            description="Z amplitude of the woven ribbons at crossings "
                        "(WOVEN interlace only)")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat ribbons; > 0 extrudes the strapwork")
        backing: BoolProperty(
            name="Backing Slab", default=False,
            description="Add a slab behind raised strapwork (relief "
                        "only)")
        base: FloatProperty(
            name="Base Thickness", default=0.08, min=0.01, max=1.0)
        separate: BoolProperty(
            name="Separate Motifs", default=False,
            description="Output each tile's motif as its own mesh "
                        "object (parented to an empty)")

        def _resolved(self):
            """(substrate, contact, motif, star_d) after any preset."""
            if self.preset != 'CUSTOM':
                return PRESETS[self.preset]
            return (self.substrate, self.contact_angle, self.motif,
                    self.star_d)

        def execute(self, context):
            substrate, contact, motif, star_d = self._resolved()
            if self.output == 'CURVE':
                bands, _orders = strapwork_bands(
                    substrate, self.nx, self.ny, contact,
                    self.ribbon_width, self.trim, self.curved,
                    self.smoothness, motif, star_d, self.rosette_frac,
                    self.girih_gens)
                obj = _emit_curve(context, "Islamic Star", bands,
                                  operator=self)
                if obj is None:
                    self.report({'ERROR'}, "no pattern generated")
                    return {'CANCELLED'}
                obj["math_art_pattern"] = True
                self.report({'INFO'}, "%s %s theta=%.0f  %d bands" %
                            (substrate, motif, contact, len(bands)))
                return {'FINISHED'}
            cells = build_cells(
                substrate, self.nx, self.ny, contact,
                self.ribbon_width, self.color_by, self.trim,
                self.height, self.backing, self.base, self.curved,
                self.smoothness, motif, star_d, self.rosette_frac,
                self.interlace, self.interlace_mode, self.weave_height,
                self.girih_gens)
            obj = pc.emit(context, "Islamic Star", cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no pattern generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s theta=%.0f  V=%d F=%d" %
                            (substrate, contact,
                             len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s  %d motifs" %
                            (substrate, len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            sub, _c, motif, _d = self._resolved()
            if self.preset == 'CUSTOM':
                lay.prop(self, 'substrate')
                lay.prop(self, 'motif')
                lay.prop(self, 'contact_angle')
            if motif == 'STAR':                       # shown for presets too
                lay.prop(self, 'star_d')
            if motif == 'ROSETTE':
                lay.prop(self, 'rosette_frac')
            if sub == 'GIRIH':                        # quasiperiodic patch
                lay.prop(self, 'girih_gens')
            else:
                lay.prop(self, 'nx')
                lay.prop(self, 'ny')
            lay.prop(self, 'trim')
            lay.prop(self, 'ribbon_width')
            lay.prop(self, 'curved')
            if self.curved:
                lay.prop(self, 'smoothness')
            lay.prop(self, 'output')
            if self.output == 'RIBBON':
                lay.prop(self, 'color_by')
                lay.prop(self, 'interlace')
                if self.interlace:
                    lay.prop(self, 'interlace_mode')
                    if self.interlace_mode == 'WOVEN':
                        lay.prop(self, 'weave_height')
                lay.prop(self, 'height')
                lay.prop(self, 'backing')
                if self.backing:
                    lay.prop(self, 'base')
                lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.islamic_pattern_add",
                             icon='SOLO_ON')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_islamic_pattern_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_islamic_pattern_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _poly_area(pts):
    """Signed area of a 2D polygon (shoelace)."""
    a = 0.0
    n = len(pts)
    for k in range(n):
        x0, y0 = pts[k]
        x1, y1 = pts[(k + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def _patch_bands(name, contact):
    """(pts, seg_nodes, pair, traced, tiles) for a 4x4 untrimmed patch --
    the exact-contact arrangement used by the continuity tests."""
    placed = list(tg._base_iter(name, 4, 4))
    motifs = [(np.asarray(p, float), len(p), star_segments(p, contact))
              for p, _ in placed]
    segments, _orders, tiles = _flatten_segments(motifs)
    pts, seg_nodes, pair = build_arrangement(segments)
    return pts, seg_nodes, pair, trace_bands(seg_nodes, pair), tiles


def _motif_patch(substrate, contact, motif, star_d, frac, nx, ny):
    """(pts, seg_nodes, pair, traced, tiles) for a patch under any motif
    -- the arrangement used by the Phase-2 continuity tests."""
    motifs, _rect = tile_motifs(substrate, nx, ny, contact, False, 2,
                                motif, star_d, frac)
    segments, _orders, tiles = _flatten_segments(motifs)
    pts, seg_nodes, pair = build_arrangement(segments)
    return pts, seg_nodes, pair, trace_bands(seg_nodes, pair), tiles


def _node_degrees(seg_nodes):
    """Valence of every arrangement node (how many strapwork segments
    touch it)."""
    deg = {}
    for nn in seg_nodes:
        if nn is None:
            continue
        for nd in nn:
            deg[nd] = deg.get(nd, 0) + 1
    return deg


def _interior_danglers(pts, seg_nodes, rect):
    """The dangling-end count that is the real acceptance criterion for
    continuous strapwork: arrangement nodes touched by exactly ONE
    segment (a loose end) that are NOT on the trim-rectangle boundary.
    A clean pattern has none -- every interior end chains onward.  On the
    trim edge degree-1 nodes are expected (the clip cut the ribbon)."""
    deg = _node_degrees(seg_nodes)
    if rect is None:
        return [nd for nd, d in deg.items() if d == 1]
    x0, y0, x1, y1 = rect
    t = 1e-6
    out = []
    for nd, d in deg.items():
        if d != 1:
            continue
        p = pts[nd]
        on_edge = (abs(p[0] - x0) < t or abs(p[0] - x1) < t
                   or abs(p[1] - y0) < t or abs(p[1] - y1) < t)
        if not on_edge:
            out.append(nd)
    return out


def _star_polygon_count(motifs, order):
    """How many tiles carry a star-polygon motif of the given point
    `order` -- a closed 2*order-segment star outline (n tips + n inner
    notches).  Used to assert Cairo/Rhombille/girih actually form stars
    rather than a plain weave."""
    n = 0
    for _poly, od, segs in motifs:
        if od == order and len(segs) == 2 * order:
            n += 1
    return n


def _self_test():
    ok = True
    print("substrate    tiles  segs bands maxseg xtile  cont "
          "shared  quads  miter")
    for name in ('SQUARE', 'TRI', 'HEX', 'TRUNCSQ', 'TRUNCHEX',
                 'TRUNCTRIHEX'):
        contact = DEFAULT_CONTACT[name]
        placed = list(tg._base_iter(name, 4, 4))
        pts, seg_nodes, pair, traced, tiles = _patch_bands(name, contact)
        nsegs = sum(1 for s in seg_nodes if s is not None)

        # (1) continuity: bands are multi-segment (the strapwork
        # actually continues) and at least one band weaves across a
        # tile boundary (spans >= 2 distinct substrate tiles).
        maxseg = 0
        xtile = 0
        n_multi = 0
        longest = None
        for node_path, seg_path in traced:
            nseg = len(seg_path)
            nt = len(set(tiles[s] for s in seg_path))
            maxseg = max(maxseg, nseg)
            xtile = max(xtile, nt)
            if nseg >= 2:
                n_multi += 1
            if longest is None or nseg > len(longest[1]):
                longest = (node_path, seg_path)
        cont_ok = maxseg >= 2 and xtile >= 2 and n_multi > 0

        # (2)/(3)/(4) on the longest band: ONE continuous ribbon
        # (vertices shared along the path -- 2 per station, not 4 per
        # segment); every ribbon quad non-degenerate; miters do not
        # invert (all quads keep one orientation, width stays bounded).
        shared_ok = quad_ok = miter_ok = False
        if longest is not None:
            node_path, seg_path = longest
            closed = len(node_path) >= 4 and node_path[0] == node_path[-1]
            poly = [pts[n] for n in node_path]
            if closed:
                poly = poly[:-1]
            left, right = miter_ribbon(poly, 0.14, closed)
            cv, cf = band_ribbon_faces(left, right, closed, 0.0)
            want_faces = len(poly) if closed else len(poly) - 1
            shared_ok = (len(cv) == 2 * len(poly)
                         and len(cf) == want_faces
                         and len(cv) < 4 * len(seg_path))
            areas = [_poly_area([cv[i][:2] for i in f]) for f in cf]
            quad_ok = bool(areas) and all(abs(a) > 1e-9 for a in areas)
            same_sign = all(a > 0 for a in areas) or all(a < 0
                                                         for a in areas)
            widths = [hypot(left[i][0] - right[i][0],
                            left[i][1] - right[i][1])
                      for i in range(len(poly))]
            miter_ok = (same_sign and min(widths) > 1e-6
                        and max(widths) <= 0.14 * 4.0 + 1e-6)

        good = cont_ok and shared_ok and quad_ok and miter_ok
        ok = ok and good
        print("%-11s  %5d %5d %5d %6d %5d   %-4s  %-5s  %-5s  %-5s %s" %
              (name, len(placed), nsegs, len(traced), maxseg, xtile,
               cont_ok, shared_ok, quad_ok, miter_ok,
               "OK" if good else "BAD"))

    # (5) curved spline: Catmull-Rom densifies each band yet keeps every
    # node -- especially the shared contact points -- exactly anchored,
    # so curved bands still meet across edges.
    _pts, _sn, _pr, traced, _tiles = _patch_bands('HEX', 30.0)
    anchor_err = 0.0
    grew = True
    for node_path, seg_path in traced:
        closed = len(node_path) >= 4 and node_path[0] == node_path[-1]
        poly = [_pts[n] for n in node_path]
        if closed:
            poly = poly[:-1]
        if len(poly) < 3:
            continue
        sm = catmull_rom(poly, closed, 6)
        if len(sm) <= len(poly):
            grew = False
        if not closed:
            for i, p in enumerate(poly):
                q = sm[i * 6]
                anchor_err = max(anchor_err,
                                 hypot(q[0] - p[0], q[1] - p[1]))
            anchor_err = max(anchor_err,
                             hypot(sm[-1][0] - poly[-1][0],
                                   sm[-1][1] - poly[-1][1]))
    curve_ok = grew and anchor_err < 1e-9
    ok = ok and curve_ok
    print("curved spline  anchor.err=%.2e  grew=%s  %s" %
          (anchor_err, grew, "OK" if curve_ok else "BAD"))

    # (6) color modes all build; BAND gives many distinct ribbon colors,
    # all within the palette.
    color_ok = True
    for cb in ('UNIFORM', 'TILE', 'STAR_ORDER', 'BAND'):
        cells = build_cells('TRUNCSQ', 4, 4, 25.0, 0.14, cb, trim=True)
        mats = set(m for _v, _f, cm in cells for m in cm)
        if not cells or not all(c[1] for c in cells) \
                or any(m >= len(pc.PALETTE_RGBA) for m in mats):
            color_ok = False
    band_cells = build_cells('TRUNCSQ', 4, 4, 30.0, 0.14, 'BAND',
                             trim=True)
    band_colors = len(set(m for _v, _f, cm in band_cells for m in cm))
    color_ok = color_ok and band_colors > 1
    ok = ok and color_ok
    print("color modes    band colors=%d  %s" %
          (band_colors, "OK" if color_ok else "BAD"))

    # (7) trimmed / curved / relief+backing builds are all non-empty.
    flat = build_cells('TRUNCSQ', 4, 4, 25.0, 0.14, 'STAR_ORDER',
                       trim=True)
    curved = build_cells('TRUNCSQ', 3, 3, 30.0, 0.14, 'UNIFORM',
                         trim=True, curved=True, subdiv=6)
    relief = build_cells('HEX', 3, 3, 30.0, 0.14, 'STAR_ORDER',
                         trim=True, height=0.3, backing=True)
    build_ok = all(len(c) > 0 and all(x[1] for x in c)
                   for c in (flat, curved, relief))
    ok = ok and build_ok
    print("builds         flat=%d curved=%d relief=%d  %s" %
          (len(flat), len(curved), len(relief),
           "OK" if build_ok else "BAD"))

    # (8) Continuity / dangling ends -- the real acceptance criterion.
    # For EVERY motif x substrate (PIC/ROSETTE/STAR on the regular
    # tilings, and the irregular Cairo / Rhombille / quasiperiodic girih)
    # the TRIMMED strapwork must be fully continuous: the count of
    # interior dangling ends (degree-1 arrangement nodes not on the trim
    # boundary) MUST be zero -- no loose star-point or petal-tip stubs,
    # no basket-weave danglers.  Each case must also knit across tile
    # boundaries (a band spanning >= 2 tiles), form multi-segment bands
    # (not a shower of stubs) and build non-degenerate ribbon cells.
    print("motif    substrate    segs bands shared xtile dngl "
          " cont  knit  ribbon")
    p2_ok = True
    #  motif, substrate, contact, star_d, frac, nx, ny, girih_gens
    p2_cases = [('PIC', 'SQUARE', 30.0, 2, 0.4, 4, 4, 0),
                ('PIC', 'HEX', 30.0, 2, 0.4, 4, 4, 0),
                ('PIC', 'TRUNCSQ', 30.0, 2, 0.4, 4, 4, 0),
                ('PIC', 'TRUNCHEX', 30.0, 2, 0.4, 3, 3, 0),
                ('PIC', 'TRUNCTRIHEX', 30.0, 2, 0.4, 3, 3, 0),
                ('ROSETTE', 'TRUNCSQ', 34.0, 2, 0.4, 4, 4, 0),
                ('ROSETTE', 'TRUNCHEX', 32.0, 2, 0.4, 3, 3, 0),
                ('ROSETTE', 'TRUNCTRIHEX', 32.0, 2, 0.4, 3, 3, 0),
                ('STAR', 'TRUNCSQ', 30.0, 3, 0.4, 4, 4, 0),
                ('STAR', 'TRUNCHEX', 30.0, 4, 0.4, 3, 3, 0),
                ('PIC', 'CAIRO', 35.0, 2, 0.4, 3, 3, 0),
                ('PIC', 'RHOMBILLE', 35.0, 2, 0.4, 3, 3, 0),
                ('PIC', 'GIRIH', 72.0, 2, 0.4, 0, 0, 4)]
    for motif, sub, contact, sd, fr, nx, ny, gg in p2_cases:
        # trimmed arrangement (for the dangling-end criterion)
        motifs_t, rect = tile_motifs(sub, nx, ny, contact, True, 2,
                                     motif, sd, fr, gg)
        segs_t, _ord_t, tiles_t = _flatten_segments(motifs_t)
        pts_t, seg_nodes_t, _pair_t = build_arrangement(segs_t)
        dngl = len(_interior_danglers(pts_t, seg_nodes_t, rect))
        dngl_ok = (dngl == 0)

        # untrimmed arrangement (for continuity / knit metrics)
        motifs_u, _r = tile_motifs(sub, nx, ny, contact, False, 2,
                                   motif, sd, fr, gg)
        segments, _orders, tiles = _flatten_segments(motifs_u)
        pts, seg_nodes, pair = build_arrangement(segments)
        traced = trace_bands(seg_nodes, pair)
        nsegs = sum(1 for s in seg_nodes if s is not None)

        node_tiles = {}
        for si, nn in enumerate(seg_nodes):
            if nn is None:
                continue
            for nd in nn:
                node_tiles.setdefault(nd, set()).add(tiles[si])
        shared = sum(1 for v in node_tiles.values() if len(v) >= 2)
        xtile = 0
        n_multi = 0
        longest = None
        for node_path, seg_path in traced:
            xtile = max(xtile, len(set(tiles[s] for s in seg_path)))
            if len(seg_path) >= 2:
                n_multi += 1
            if longest is None or len(seg_path) > len(longest[1]):
                longest = (node_path, seg_path)
        cont_ok = shared > 0 and xtile >= 2
        knit_ok = bool(traced) and n_multi >= 0.5 * len(traced)

        ribbon_ok = False
        if longest is not None:
            node_path, seg_path = longest
            closed = (len(node_path) >= 4
                      and node_path[0] == node_path[-1])
            poly = [pts[n] for n in node_path]
            if closed:
                poly = poly[:-1]
            if len(poly) >= 2:
                left, right = miter_ribbon(poly, 0.10, closed)
                cv, cf = band_ribbon_faces(left, right, closed, 0.0)
                areas = [_poly_area([cv[i][:2] for i in f]) for f in cf]
                widths = [hypot(left[i][0] - right[i][0],
                                left[i][1] - right[i][1])
                          for i in range(len(poly))]
                ribbon_ok = (bool(areas)
                             and all(abs(a) > 1e-12 for a in areas)
                             and min(widths) > 1e-6
                             and max(widths) <= 0.10 * 4.0 + 1e-6)

        cells = build_cells(sub, nx, ny, contact, 0.10, 'STAR_ORDER',
                            trim=True, motif=motif, star_d=sd,
                            rosette_frac=fr, girih_gens=gg)
        build_p2 = bool(cells) and all(c[1] for c in cells)

        good = (nsegs > 0 and cont_ok and knit_ok and ribbon_ok
                and build_p2 and dngl_ok)
        p2_ok = p2_ok and good
        print("%-8s %-11s %5d %5d %6d %5d %4d  %-4s  %-5s  %-5s %s" %
              (motif, sub, nsegs, len(traced), shared, xtile, dngl,
               cont_ok, knit_ok, ribbon_ok,
               "OK" if good else "BAD"))
    ok = ok and p2_ok

    # (8b) Stars really form on the irregular / quasiperiodic substrates
    # (not the old starless weave).  Cairo builds a five-point star in
    # every pentagon; Rhombille builds six-point stars at its high-valence
    # vertices; girih forms ten-point star loops.  We detect the star-
    # polygon outlines directly (closed 2n-segment tip+notch figures) and,
    # for girih, a long closed band that rings a ten-fold star.
    cairo_m, _r = tile_motifs('CAIRO', 3, 3, 35.0, False)
    cairo_stars = _star_polygon_count(cairo_m, 5)
    rhomb_m, _r = tile_motifs('RHOMBILLE', 3, 3, 35.0, False)
    rhomb_stars = _star_polygon_count(rhomb_m, 6)
    gm, _r = tile_motifs('GIRIH', 0, 0, 72.0, False, girih_gens=4)
    gsegs, _go, _gt = _flatten_segments(gm)
    gpts, gsn, gpair = build_arrangement(gsegs)
    gtraced = trace_bands(gsn, gpair)
    girih_star_loops = sum(
        1 for np_, sp in gtraced
        if len(np_) >= 4 and np_[0] == np_[-1] and len(sp) >= 10)
    star_ok = (cairo_stars > 0 and rhomb_stars > 0
               and girih_star_loops > 0)
    ok = ok and star_ok
    print("stars          cairo5=%d rhombille6=%d girih10loops=%d  %s" %
          (cairo_stars, rhomb_stars, girih_star_loops,
           "OK" if star_ok else "BAD"))

    # (8c) Girih is a genuine EDGE-TO-EDGE quasiperiodic patch: every
    # rhomb edge is shared (its midpoint contact is a node touched by two
    # tiles) and the strapwork weaves right across those seams.
    gm2, _r = tile_motifs('GIRIH', 0, 0, 72.0, False, girih_gens=4)
    g2segs, _o, g2tiles = _flatten_segments(gm2)
    g2pts, g2sn, g2pair = build_arrangement(g2segs)
    g2node_tiles = {}
    for si, nn in enumerate(g2sn):
        if nn is None:
            continue
        for nd in nn:
            g2node_tiles.setdefault(nd, set()).add(g2tiles[si])
    g2shared = sum(1 for v in g2node_tiles.values() if len(v) >= 2)
    g2xtile = max(len(set(g2tiles[s] for s in sp))
                  for _np, sp in trace_bands(g2sn, g2pair))
    girih_ok = (len(gm2) > 40 and g2shared > 0 and g2xtile >= 3)
    ok = ok and girih_ok
    print("girih patch    tiles=%d shared=%d bandspans<=%d tiles  %s" %
          (len(gm2), g2shared, g2xtile, "OK" if girih_ok else "BAD"))

    # (9) PIC regression: the default motif on a regular substrate must
    # reproduce the plain star_segments motif byte-for-byte (Phase-1
    # behaviour is unchanged for motif == PIC).
    reg_placed = list(tg._base_iter('TRUNCSQ', 2, 2))
    pic_ok = True
    for poly, _t in reg_placed:
        a = star_segments(poly, 30.0)
        b = _tile_segments(poly, 30.0, 'PIC', 2, 0.4)
        if len(a) != len(b) or any(
                hypot(a[i][0][0] - b[i][0][0], a[i][0][1] - b[i][0][1])
                + hypot(a[i][1][0] - b[i][1][0], a[i][1][1] - b[i][1][1])
                > 1e-12 for i in range(len(a))):
            pic_ok = False
    ok = ok and pic_ok
    print("PIC regression byte-identical=%s  %s" %
          (pic_ok, "OK" if pic_ok else "BAD"))

    # (10) Interlacing / weaving.  For each case: the over/under
    # assignment must (a) strictly ALTERNATE along every band (no two
    # consecutive crossings of a band share sense) and (b) be CONSISTENT
    # at every crossing (exactly two passages, one over and one under);
    # FLAT interlace must actually BREAK the under-bands (more ribbon
    # pieces than plain bands, by roughly the number of under-crossings);
    # WOVEN must give the ribbon z-variation with the right sign at each
    # crossing (over -> +weave, under -> -weave); and interlace=False
    # must be byte-identical to the plain build.
    # (At exactly 30 deg the HEX crossings collapse onto tiling vertices
    # as degenerate higher-valence confluences; trimmed HEX therefore has
    # no clean 4-valent crossings -- handled gracefully as no weave.  The
    # untrimmed HEX patch does cross cleanly, so it is used here.)
    print("substrate  motif    cross alt   consist "
          "flatgap wovenz sign")
    il_ok = True
    il_cases = [('TRUNCSQ', 'PIC', 30.0, 2, 0.4, True),
                ('TRUNCHEX', 'PIC', 30.0, 2, 0.4, True),
                ('SQUARE', 'PIC', 30.0, 2, 0.4, True),
                ('HEX', 'PIC', 30.0, 2, 0.4, False),
                ('TRUNCSQ', 'ROSETTE', 34.0, 2, 0.4, True),
                ('TRUNCHEX', 'ROSETTE', 32.0, 2, 0.4, True),
                ('GIRIH', 'PIC', 72.0, 2, 0.4, True)]
    for sub, motif, contact, sd, fr, il_trim in il_cases:
        tp = _traced_patch(sub, 4, 4, contact, il_trim, motif, sd, fr)
        pts, _sn, pair, traced, orders = tp
        crossings = _find_crossings(pair)
        ref, other = _ref_pairs(crossings)

        # per-band crossing sequences (straight bands for the test):
        # keep each band's path so the woven z-offsets can be checked.
        band_seqs, band_cross = [], []
        for node_path, seg_path in traced:
            closed, poly, path, stations, pair_here = _band_geometry(
                pts, node_path, seg_path, False, 1)
            if len(poly) < 2:
                continue
            cross = []
            for k, nd in enumerate(stations):
                ph = pair_here[k]
                if (nd in crossings and ph is not None
                        and ph in crossings[nd]):
                    cross.append((k, nd, ph))
            seq = [(nd, 1 if ph == ref[nd] else 0) for _k, nd, ph in cross]
            band_seqs.append((seq, closed))
            band_cross.append((closed, path, cross))
        over = _solve_over(crossings, ref, other, band_seqs)
        ncross = len(crossings)

        # (a) alternation along each band (consecutive crossings differ)
        alt_ok = True
        for closed, path, cross in band_cross:
            flags = [ph == over[nd] for _k, nd, ph in cross]
            for i in range(len(flags) - 1):
                if flags[i] == flags[i + 1]:
                    alt_ok = False

        # (b) consistency at every crossing: exactly one over, one under
        passages = {}
        for closed, path, cross in band_cross:
            for _k, nd, ph in cross:
                passages.setdefault(nd, []).append(ph == over[nd])
        consist_ok = (bool(passages)
                      and all(len(v) == 2 and sum(v) == 1
                              for v in passages.values()))

        # (c) FLAT breaks the under-bands: more pieces than plain bands
        plain = build_cells(sub, 4, 4, contact, 0.12, 'BAND', trim=il_trim,
                            motif=motif, star_d=sd, rosette_frac=fr)
        flat = build_cells(sub, 4, 4, contact, 0.12, 'BAND', trim=il_trim,
                           motif=motif, star_d=sd, rosette_frac=fr,
                           interlace=True, interlace_mode='FLAT')
        flat_ok = bool(flat) and len(flat) > len(plain)

        # (d) WOVEN z-variation reaching +/- the weave height
        wh = 0.1
        woven = build_cells(sub, 4, 4, contact, 0.12, 'BAND', trim=il_trim,
                            motif=motif, star_d=sd, rosette_frac=fr,
                            interlace=True, interlace_mode='WOVEN',
                            weave_height=wh)
        zs = [v[2] for cv, _cf, _cm in woven for v in cv]
        wovenz_ok = bool(zs) and max(zs) > 0.5 * wh and min(zs) < -0.5 * wh

        # (e) sign correctness: on a band with >=2 crossings, the woven
        # z-offset is >= +weave at every over-crossing and <= -weave at
        # every under-crossing.
        sign_ok = True
        checked = False
        for closed, path, cross in band_cross:
            if len(cross) < 2:
                continue
            signed = [(k, 1 if ph == over[nd] else -1)
                      for k, nd, ph in cross]
            zoff = _weave_zoff(path, closed, signed, wh)
            for k, nd, ph in cross:
                if ph == over[nd] and zoff[k] < wh - 1e-9:
                    sign_ok = False
                if ph != over[nd] and zoff[k] > -wh + 1e-9:
                    sign_ok = False
            checked = True
            break
        sign_ok = sign_ok and checked

        good = (ncross > 0 and alt_ok and consist_ok and flat_ok
                and wovenz_ok and sign_ok)
        il_ok = il_ok and good
        print("%-10s %-8s %5d %-5s %-6s  %-6s  %-5s %-4s %s" %
              (sub, motif, ncross, alt_ok, consist_ok, flat_ok,
               wovenz_ok, sign_ok, "OK" if good else "BAD"))
    ok = ok and il_ok

    # (11) interlace=False regression: identical to the plain build even
    # when interlace mode / weave height are (ignored) non-defaults.
    a = build_cells('TRUNCSQ', 4, 4, 30.0, 0.14, 'STAR_ORDER', trim=True)
    b = build_cells('TRUNCSQ', 4, 4, 30.0, 0.14, 'STAR_ORDER', trim=True,
                    interlace=False, interlace_mode='WOVEN',
                    weave_height=0.2)
    reg_ok = (a == b)
    ok = ok and reg_ok
    print("interlace=False regression byte-identical=%s  %s" %
          (reg_ok, "OK" if reg_ok else "BAD"))

    print("RESULT:", "OK" if ok else "BAD")
    return ok
