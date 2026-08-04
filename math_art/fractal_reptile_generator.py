
# Fractal Rep-Tile (Polygon f-tiling) Generator for Blender
#
# Fathauer fractal tilings ("f-tilings") built from rep-tile prototiles
# rather than the kite/dart of the sibling Fractal Tiling generator.  A
# single similar prototile is grown edge-to-edge in progressively
# smaller generations; the interior is gap-free at every depth and only
# the LIMIT boundary is fractal.  Each prototile here is a rep-tile (it
# dissects into scaled copies of itself), so the "long" edge of a child
# glues exactly onto the "short" edge of its parent, which fixes the
# generation ratio.
#
#   RIGHT_TRIANGLE  isosceles right triangle (45-45-90).  The hypotenuse
#       is sqrt(2) times a leg, so a child scaled by 1/sqrt(2) has its
#       hypotenuse equal in length to the parent's leg.  SEED: two
#       triangles sharing their hypotenuse form a unit square (the
#       4-fold-symmetric generating polyomino).  RECURSION: onto every
#       exposed leg (short edge) of the newest generation, glue one
#       child triangle -- hypotenuse over the leg, right-angle apex
#       pointing outward -- scaled by s = 1/sqrt(2).  Neighbouring
#       children meet leg-to-leg (edge-to-edge); the tiles shrink toward
#       a fractal boundary.
#
# Besides the f-tiling, three further constructions are implemented:
#
# COMPLEX-BASE SELF-AFFINE REPTILES.  A Gaussian-integer base b with
# N = |b|^2 and a digit set D of N residue representatives (digits may
# themselves be complex) defines a radix system on Z[i]: every string
# (d_0 ... d_{k-1}) maps to the Gaussian integer sum_j d_j b^j.  Placing
# a unit square at each of the N^k positions yields a polyomino that
# converges (after rescaling) to a rep-N self-affine tile -- Bandt's
# integer-matrix theorem, catalogued by Vince.  The base -1+i with
# D = {0,1} gives the twindragon (base +1+i FAILS unique representation,
# per Gilbert/Knuth); -2+i with the symmetric digits {0,1,-1,i,-i} gives
# the round rep-5 dragon; collinear digits {0..4} over 2+i and 1+2i give
# the thin dragons whose aspect ratios Mekhontsev computed.
#
# The same radix construction over the EISENSTEIN ring Z[w],
# w = e^(2 pi i/3), with base 3+w = 5/2+(sqrt3/2)i (norm 7) and digits
# {0} + the six sixth-roots of unity, places Voronoi hexagons of the
# triangular lattice: the rep-7 FLOWSNAKE, whose union converges to
# the Gosper island (Vince Fig 4/7; Gardner's 1976 "flowsnake";
# Mandelbrot's Gosper curve).
#
# REFLECTION & FOLDABLE REP-TILES.  Sajid-Husain-Kumar (2026) extend
# the digit-system maps w_j = M^-1 x + d_j by composing with
# reflections/rotations sigma_j from the symmetry group of M.  Each
# map is held in the closed conjugate-affine form
# w(z) = A z + B conj(z) + C and every length-k word is applied to a
# unit-square seed, giving m^k rotated/reflected quads coloured by the
# outermost digit.  Implemented: the LEVY DRAGON (rep-2, x-axis
# reflection; Levy's 1938 curve), the LEAF rep-tile (rep-2, y-axis
# reflection) and the FOLDABLE rep-4 tile (M = 2I, the digit system
# {0,1,i,-1-i} of Vince Fig 6 with the last map an orientation-
# reversing quarter-turn reflection, Theorem 4.4).  Only twists from
# the lattice symmetry group admit the open set condition; a free
# irrational twist angle leaves positive-measure overlaps.
#
# FATHAUER ISOMETRY INFLATION (PENTABOLO).  Following Fathauer's own
# construction, a half-square right triangle {(0,0),(1,0),(1/2,1/2)} is
# grown by five fixed plane isometries M0..M4 (identity, +/-90 and 180
# degree rotations about points of the half-integer lattice).  The union
# of the five images is the 5-triangle pentabolo; each further level
# re-applies the five maps conjugated by the sqrt(5) expansion
# z -> (1+2i)^k z (same rotation angles, inflated centres), giving
# 5^k congruent unit triangles at level k with no overlaps -- a rep-5
# gasket built from isometries rather than contractions.
#
# FRACTAL GASKETS (HOLES).  Fathauer's gaskets (Bridges 2006 and the
# Bridges 2016 "Fractal Gaskets" paper below) drop sub-copies from a
# rep-N dissection: keeping only N - h of the substitution digits (or
# IFS/isometry maps) at EVERY level replaces the solid tile with a
# Sierpinski-like gasket whose holes repeat at all scales.  The
# `holes` option here removes the LAST h entries of the digit/map
# list -- the kept positions are still distinct radix strings (resp.
# open-set-condition images), so the (N-h)^k cells never overlap --
# e.g. holes=1 on the rep-4 digit system {0,1,i} leaves the classic
# Sierpinski triangle arrangement, and holes=1 on the pentabolo keeps
# a 4-isometry gasket.  holes=0 (default) is the full solid rep-tile;
# the right-triangle f-tiling grows edge-by-edge rather than by
# substitution, so it ignores the option.
#
# The pure-Python self-test verifies, for each kind: the f-tiling is
# edge-to-edge, gap-free and shrinks by exactly 1/sqrt(2) per level;
# the complex-base reptiles place N^k unit cells at distinct Gaussian-
# integer positions; the pentabolo has 5^k triangles with zero sampled
# overlaps.  Gasket (holes > 0) runs are checked for the reduced
# count (N-holes)^k and the same distinctness / non-overlap.
#
# References:
# - Andrew Vince, "Rep-tiling Euclidean space", Aequationes
#   Mathematicae 50, 1995, pp. 191-213.  The radix-system catalog:
#   Fig 3 (rep-2 twindragon, base -1+i), Fig 4/7 (rep-7 flowsnake /
#   Gosper island, Eisenstein base 3+omega), Fig 5 (rep-5 dragon,
#   base -2+i, symmetric digits), Fig 6 (rep-4, base 2, digits
#   {0,1,i,-1-i}).
# - Martin Gardner, "Mathematical Games: In which 'monster' curves
#   force redefinition of the word 'curve'", Scientific American 235,
#   Dec 1976, pp. 124-133.  Names the flowsnake (Gosper's curve);
#   see also Benoit B. Mandelbrot, "The Fractal Geometry of Nature",
#   W. H. Freeman, 1982, ch. 6.
# - Paul Levy, "Les courbes planes ou gauches et les surfaces
#   composees de parties semblables au tout", Journal de l'Ecole
#   Polytechnique II-19, 1938, pp. 227-291.  The Levy C curve /
#   dragon, the earliest reflection-generated rep-tile.
# - Christoph Bandt, "Self-similar sets 5. Integer matrices and fractal
#   tilings of R^n", Proceedings of the AMS 112, 1991, pp. 549-562.
#   The integer-matrix + residue-digit-set theorem behind all the
#   complex-base tiles.
# - Solomon W. Golomb, "Replicating figures in the plane",
#   Mathematical Gazette 48, 1964, pp. 403-412.  Defines rep-k figures.
# - Dmitry Mekhontsev, "The aspect ratio of the Twin Dragon is
#   1/phi", arXiv:2604.05010, 2026.  Aspect ratios of collinear-digit
#   dragons over Z[i]: (5,4) base 2+i AR 1/phi^3; (5,2) base 1+2i AR
#   sqrt(2)-1; symmetric-digit base -2+i AR 1/phi.
# - Mohammad Sajid, Akhlaq Husain, Krishnendra S. Kumar, "Fractal
#   rep-tiles of the plane via reflections and integer matrices",
#   Frontiers in Physics 14:1699796, 2026.  Extends the digit-system
#   maps with reflections/rotations from the symmetry group of the
#   matrix: the Levy dragon (Fig 2c), the leaf rep-tile (Fig 2d) and
#   the foldable chiral rep-4 family (Theorem 4.4) implemented here.
# - Robert W. Fathauer, "Fractal Tilings Based on Dissections of
#   Polyominoes", Bridges 2006, pp. 293-300; "Fractal gaskets:
#   rep-tiles, Hamiltonian cycles, and spatial development", Bridges
#   2016, pp. 217-224.  The f-tiling grower and the pentabolo
#   5-isometry inflation.  Online: Fractal Tiling Encyclopedia,
#   https://www.mathartfun.com/encyclopedia/Introduction.html .

bl_info = {
    "name": "Fractal Rep-Tile Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Fractal rep-tiles: right-triangle f-tiling, "
                   "complex-base reptiles (twindragon, rep-5, "
                   "flowsnake) and reflection tiles (Levy dragon, "
                   "leaf, foldable rep-4)",
    "category": "Add Mesh",
}

from collections import Counter
from math import sqrt

import numpy as np

try:
    from . import pattern_common as pc
    from . import tiling_generator as tg
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc
    import tiling_generator as tg


# --------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------

def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(poly):
    poly = np.asarray(poly, float)
    return poly[::-1].copy() if _signed_area(poly) < 0.0 else poly


def _key(p, nd=6):
    return (round(float(p[0]), nd), round(float(p[1]), nd))


def _edge_keys(tri):
    return [frozenset((_key(tri[i]), _key(tri[(i + 1) % 3])))
            for i in range(3)]


def _leg_indices(tri):
    """Indices of the two SHORT edges (legs); the longest is the hyp."""
    lens = [np.linalg.norm(tri[(i + 1) % 3] - tri[i]) for i in range(3)]
    hyp = int(np.argmax(lens))
    return [i for i in range(3) if i != hyp]


def _hyp_index(tri):
    lens = [np.linalg.norm(tri[(i + 1) % 3] - tri[i]) for i in range(3)]
    return int(np.argmax(lens))


def _child_on_edge(p1, p2, centroid):
    """Right-isosceles child with hypotenuse p1-p2 and its right-angle
    apex on the outward side (away from `centroid`)."""
    mid = 0.5 * (p1 + p2)
    d = p2 - p1
    h = float(np.linalg.norm(d))
    n = np.array([-d[1], d[0]], float)
    n /= np.linalg.norm(n)
    if np.dot(n, mid - centroid) < 0.0:
        n = -n
    return np.array([p1, p2, mid + n * (h * 0.5)])


# --------------------------------------------------------------------
# Right-triangle f-tiling
# --------------------------------------------------------------------

_MAX_TILES = 120000

# seed: two right-isosceles triangles sharing the diagonal of a unit
# square (4-fold-symmetric generating polyomino)
_SEED = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
         np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]


def _triangle_patch(iterations):
    """Grow the f-tiling: onto every exposed leg of the newest
    generation, glue a 1/sqrt(2)-scaled child (hyp over the leg, apex
    outward).  An edge already shared (internal) is skipped, so
    neighbouring branches meet edge-to-edge."""
    tiles = [(0, t.copy()) for t in _SEED]
    edge_use = Counter()
    for _, t in tiles:
        for ek in _edge_keys(t):
            edge_use[ek] += 1
    frontier = list(range(len(tiles)))
    for _ in range(int(iterations)):
        legs = sum(len(_leg_indices(tiles[i][1])) for i in frontier)
        if len(tiles) + legs > _MAX_TILES:
            break
        new_idx = []
        for ti in frontier:
            lvl, t = tiles[ti]
            c = t.mean(axis=0)
            for li in _leg_indices(t):
                p1, p2 = t[li], t[(li + 1) % 3]
                if edge_use[frozenset((_key(p1), _key(p2)))] != 1:
                    continue                      # internal leg
                child = _child_on_edge(p1, p2, c)
                tiles.append((lvl + 1, child))
                new_idx.append(len(tiles) - 1)
                for ek in _edge_keys(child):
                    edge_use[ek] += 1
        frontier = new_idx
    return tiles


# --------------------------------------------------------------------
# Complex-base self-affine reptiles (Bandt 1991; Vince 1995)
#
# A Gaussian-integer base b with N = |b|^2 and a complete residue digit
# set D (|D| = N; digits may be complex, e.g. {0,1,-1,i,-i}) defines a
# rep-N tile: every b-adic "integer" sum_{j} d_j b^j (d_j in D) is a
# distinct Gaussian-integer lattice point, and the unit cell placed at
# each of the N**k length-k strings forms a polyomino that -- because b
# carries a rotation -- crinkles toward a fractal reptile as k grows.
# N copies compound into a b-times-larger replica (self-similar).
# Every cell is the same size, so there is no "shrinking generation" to
# colour by; instead cells are coloured by their two most significant
# digits (d0, d1) -> the N-fold substitution nested one level deep
# (N^2 classes), so the recursive structure reads directly, as in
# Vince's and Fathauer's figures.
# --------------------------------------------------------------------

_UNIT = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _base_reptile(b, digits, iterations, cell=_UNIT, holes=0):
    """(polys, types) for the rep-N reptile of base b with digit set
    `digits` (ints or complex; positions stay lattice points).  A copy
    of `cell` (default the unit square; a Voronoi hexagon for the
    Eisenstein flowsnake) is placed at every position.  The colour
    index encodes the top two substitution digits (d0*N + d1) so both
    the outer N copies and the next level of nesting are visible.
    holes > 0 drops the last `holes` digits at every level -- the
    Fathauer gasket: (N-holes)^k cells at still-distinct positions
    (subsets of distinct radix strings stay distinct)."""
    digits = list(digits)[:max(1, len(digits) - int(holes))]
    cells = [(complex(0.0, 0.0), 0)]      # (position, colour)
    n = len(digits)
    for lvl in range(int(iterations)):
        if len(cells) * n > _MAX_TILES:
            break
        nxt = []
        for p, col in cells:
            for di, d in enumerate(digits):
                if lvl == 0:              # most significant digit
                    ncol = di
                elif lvl == 1:            # fold in the 2nd digit
                    ncol = col * n + di
                else:                     # deeper digits don't recolour
                    ncol = col
                nxt.append((p * b + d, ncol))
        cells = nxt
    polys, types = [], []
    for p, col in cells:
        polys.append(cell + [p.real, p.imag])
        types.append(int(col))
    return polys, types


# --------------------------------------------------------------------
# Flowsnake / Gosper island (Vince 1995 Fig 4/7): rep-7, hexagonal
#
# The same radix construction over the Eisenstein ring Z[w],
# w = e^(2 pi i/3): base beta = 3 + w = 5/2 + (sqrt3/2) i (norm 7),
# digit set {0} + the six sixth-roots of unity (the origin and its 6
# triangular-lattice neighbours).  Positions sum d_j beta^j are
# triangular-lattice points; the cell is the lattice's Voronoi hexagon
# (circumradius 1/sqrt3), so distinct positions tile with no overlap
# and the union converges to the Gosper island bounded by the
# flowsnake curve (contraction 1/sqrt7, twist -arctan(sqrt3/5) ~ -19.1
# degrees per level).
# --------------------------------------------------------------------

_FLOW_BASE = 2.5 + 0.5 * sqrt(3.0) * 1j          # 3 + w, norm 7
_FLOW_DIGITS = [0j] + [np.exp(1j * k * np.pi / 3.0) for k in range(6)]

# Voronoi hexagon of the triangular lattice {1, w}: vertices at
# (1/sqrt3) e^(i(pi/6 + k pi/3)), centred on the lattice point
_HEXCELL = np.array(
    [[np.cos(np.pi / 6.0 + k * np.pi / 3.0) / sqrt(3.0),
      np.sin(np.pi / 6.0 + k * np.pi / 3.0) / sqrt(3.0)]
     for k in range(6)])

# unit Eisenstein step (60 deg): hex-cell centres live on Z + Z*_W6, and
# the polyhex radix reptiles use an Eisenstein base b (|b|^2 = n) with a
# ROTATING argument so the boundary is fractal (a real/axis-aligned base
# such as -2 or 3 gives a non-fractal parallelogram instead).
_W6 = np.exp(1j * np.pi / 3.0)


# --------------------------------------------------------------------
# Reflection & foldable rep-tiles (Sajid-Husain-Kumar 2026)
#
# IFS maps w_j(z) = sigma_j M^-1 z + d_j where sigma_j is drawn from
# the symmetry group of the integer matrix M (reflections/rotations),
# not just the identity -- Levy's 1938 dragon is the earliest example.
# Each map is kept in the closed conjugate-affine form
#     w(z) = A z + B conj(z) + C          (complex A, B, C),
# which is closed under composition.  The attractor is rendered by
# applying every length-k word to a unit-square seed: m^k small
# (rotated/reflected) quads, coloured by the OUTERMOST word digit.
# These satisfy the open set condition, so quad interiors overlap only
# in a measure-zero boundary set (the self-test samples this).
# --------------------------------------------------------------------

def _w_apply(w, z):
    """Apply map w = (A, B, C): z -> A z + B conj(z) + C."""
    a, b, c = w
    return a * z + b * np.conj(z) + c


def _w_compose(w2, w1):
    """Composition w2 o w1 in the closed form (A, B, C)."""
    a2, b2, c2 = w2
    a1, b1, c1 = w1
    return (a2 * a1 + b2 * np.conj(b1),
            a2 * b1 + b2 * np.conj(a1),
            a2 * c1 + b2 * np.conj(c1) + c2)


_W_ID = (1 + 0j, 0j, 0j)

# unit-square seed for the IFS attractor renders
_IFS_SEED = np.array([0j, 1 + 0j, 1 + 1j, 1j])

# Levy dragon (rep-2): M^-1 = (1-i)/2; second map reflects in the
# x-axis (conj) before contracting  [Fig 2c; Levy 1938]
_LEVY_MAPS = [((1 - 1j) / 2, 0j, 0j),
              (0j, (1 + 1j) / 2, 1 + 0j)]

# Leaf rep-tile (rep-2): same contraction, second map reflects in the
# y-axis (z -> -conj z)  [Fig 2d].  Translation (3+i)/2: with the
# printed translation 1 the two maps satisfy w1 = w0 o g for a
# reflection g fixing the unit square, so the system degenerates
# (every quad duplicated); (3+i)/2 -- the same residue class mod 1+i
# -- yields the compact curled-leaf tile with measure-zero overlaps.
_LEAF_MAPS = [((1 - 1j) / 2, 0j, 0j),
              (0j, -(1 + 1j) / 2, 1.5 + 0.5j)]

# Foldable rep-4 (Theorem 4.4): M = 2I with digits {0, 1, i, -1-i}
# (Vince Fig 6's residue system) and the LAST map orientation-
# reversing: w3(z) = (i/2) conj(z) - (1+i)/2, i.e. the spec form
# (1/2) e^{-i theta} conj(z) + d with theta locked to a quarter turn.
# The twist must come from the symmetry group of the square lattice:
# an irrational twist angle (e.g. 0.30 rad) provably cannot satisfy
# the open set condition -- measured overlap stays ~9% even with
# optimised translations -- whereas this lattice reflection tiles
# exactly (sampled overlap 0).  The reflected fourth copy is what
# folds the Sierpinski-relative gasket into a triangular tile.
_FOLD_MAPS = [
    (0.5 + 0j, 0j, 0j),                    # z/2            d = 0
    (0.5 + 0j, 0j, 0.5 + 0j),              # z/2 + 1/2      d = 1
    (0.5 + 0j, 0j, 0.5j),                  # z/2 + i/2      d = i
    (0j, 0.5j, -0.5 - 0.5j),               # i conj(z)/2 - (1+i)/2
]


def _ifs_reptile(maps, iterations, holes=0):
    """(polys, types) for the attractor of a conjugate-affine IFS:
    every length-k word applied to the unit-square seed gives m^k
    small quads, coloured by the outermost word digit (m classes).
    holes > 0 drops the last `holes` maps at every level (gasket);
    a sub-system of an OSC system still satisfies the OSC."""
    maps = list(maps)[:max(1, len(maps) - int(holes))]
    m = len(maps)
    words = [(_W_ID, 0)]
    for lvl in range(int(iterations)):
        if len(words) * m > _MAX_TILES:
            break
        nxt = []
        for w, col in words:
            for mi, wm in enumerate(maps):
                nxt.append((_w_compose(w, wm),
                            mi if lvl == 0 else col))
        words = nxt
    polys, types = [], []
    for w, col in words:
        z = _w_apply(w, _IFS_SEED)
        polys.append(np.column_stack([z.real, z.imag]))
        types.append(int(col))
    return polys, types


# --------------------------------------------------------------------
# Fathauer pentabolo (Bridges 2006/2016): 5-isometry inflation
#
# Seed: the half-square right triangle {(0,0),(1,0),(1/2,1/2)}.  Five
# fixed isometries of the plane (identity; +90/180 deg rotations about
# 1/2+1/2i; 180 deg about 1/2; -90 deg about 1) send the seed to the
# 5-triangle pentabolo.  Each further level inflates by sqrt(5): the
# same five maps are re-applied CONJUGATED by the expansion
# D(z) = (1+2i)^k z, i.e. rotations of the same angles about the
# inflated (still half-integer) centres.  Since 1+2i is a Gaussian
# integer of norm 5, every triangle stays a quarter-cell of the unit
# grid, so level k is exactly 5^k congruent unit triangles with no
# overlap (verified by the self-test).  Colour = index of the
# OUTERMOST (last-applied) map, giving 5 first-level colour classes.
# --------------------------------------------------------------------

_PENTA_C = 0.5 + 0.5j

_PENTA_MAPS = [
    lambda z: z,                                    # M0 identity
    lambda z: _PENTA_C + 1j * (z - _PENTA_C),       # M1 rot +90 @ 1/2+1/2i
    lambda z: 2.0 * _PENTA_C - z,                   # M2 rot 180 @ 1/2+1/2i
    lambda z: 1.0 - z,                              # M3 rot 180 @ 1/2
    lambda z: 1.0 - 1j * (z - 1.0),                 # M4 rot -90 @ 1
]

_PENTA_LAMBDA = 1 + 2j            # inflation, |lambda|^2 = 5


def _pentabolo(iterations, holes=0):
    """(polys, types) for Fathauer's rep-5 pentabolo inflation: 5^k
    unit right triangles, coloured by the outermost map index.
    holes > 0 drops the last `holes` isometries at every level --
    Fathauer's Bridges-2016 fractal gasket -- leaving (5-holes)^k
    still-disjoint triangles."""
    maps = _PENTA_MAPS[:max(1, len(_PENTA_MAPS) - int(holes))]
    tris = [(np.array([0.0, 1.0, 0.5 + 0.5j], complex), 0)]
    for lvl in range(int(iterations)):
        if len(tris) * len(maps) > _MAX_TILES:
            break
        f = _PENTA_LAMBDA ** lvl              # conjugating expansion
        nxt = []
        for verts, _col in tris:
            for mi, m in enumerate(maps):
                nxt.append((f * m(verts / f), mi))
        tris = nxt
    polys = [np.column_stack([v.real, v.imag]) for v, _ in tris]
    types = [int(c) for _, c in tris]
    return polys, types


# builders take `iterations` (and a gasket `holes` count) and return
# (polys, types)
def _triangle_ftiling(iterations, holes=0):
    # edge-grown f-tiling, not an N-map substitution: `holes` ignored
    raw = _triangle_patch(iterations)
    return ([_ensure_ccw(t) for _, t in raw],
            [int(lvl) for lvl, _ in raw])


KINDS = {
    # Fathauer polygon-substitution kinds (triangle cells)
    'RIGHT_TRIANGLE': dict(build=_triangle_ftiling, tri=True),
    'PENTABOLO': dict(build=_pentabolo, tri_cells=True, n=5),
    # Complex-base radix-system kinds (unit-square cells)
    'TWINDRAGON': dict(                       # Vince Fig 3, AR 1/phi
        build=lambda it, h=0: _base_reptile(-1 + 1j, [0, 1], it,
                                            holes=h), n=2),
    'REP4': dict(                             # Vince Fig 6
        build=lambda it, h=0: _base_reptile(2 + 0j, [0, 1, 1j, -1 - 1j],
                                            it, holes=h), n=4),
    'REP5': dict(                             # Vince Fig 5, AR 1/phi
        build=lambda it, h=0: _base_reptile(-2 + 1j, [0, 1, -1, 1j, -1j],
                                            it, holes=h), n=5),
    'REP5_THIN': dict(                        # Mekhontsev (5,4)
        build=lambda it, h=0: _base_reptile(2 + 1j, [0, 1, 2, 3, 4], it,
                                            holes=h), n=5),
    'REP5B': dict(                            # Mekhontsev (5,2)
        build=lambda it, h=0: _base_reptile(1 + 2j, [0, 1, 2, 3, 4], it,
                                            holes=h), n=5),
    # Fathauer polyomino reptiles (Bridges 2025): the radix DIGIT SET is
    # the polyomino's own cell positions (a complete residue system mod
    # the base b, |b|^2 = n); the resulting self-affine tile is the
    # filled reptile.  (Domino = TWINDRAGON, X-pentomino = REP5, bar-
    # pentomino = REP5_THIN/REP5B already appear above.)
    'Z_PENTOMINO': dict(                      # Z pentomino, rep-5
        build=lambda it, h=0: _base_reptile(
            -2 - 1j, [0, 1, 1 + 1j, 1 + 2j, 2 + 2j], it, holes=h), n=5),
    'Y_PENTOMINO': dict(                      # Y pentomino, rep-5
        build=lambda it, h=0: _base_reptile(
            -2 + 1j, [0, 1j, 2j, 3j, 1 + 1j], it, holes=h), n=5),
    'P_PENTOMINO': dict(                      # P pentomino, rep-5
        build=lambda it, h=0: _base_reptile(
            -2 + 1j, [0, 1, 1j, 1 + 1j, 2j], it, holes=h), n=5),
    'Z_OCTOMINO': dict(                       # Z octomino, rep-8
        build=lambda it, h=0: _base_reptile(
            -2 - 2j, [0, 1, 2, 3, 1 + 1j, 2 + 1j, 3 + 1j, 4 + 1j], it,
            holes=h), n=8),
    'Z_TETROMINO': dict(                      # Z tetromino -> twindragon
        build=lambda it, h=0: _base_reptile(
            -2j, [0, 1, 1 + 1j, 2 + 1j], it, holes=h), n=4),
    # Eisenstein radix-system kinds (hexagon cells); polyhex reptiles
    'FLOWSNAKE': dict(                        # Vince Fig 4/7 (heptahex)
        build=lambda it, h=0: _base_reptile(_FLOW_BASE, _FLOW_DIGITS,
                                            it, cell=_HEXCELL, holes=h),
        n=7, samp=True),
    'TRIHEX': dict(                           # trihex, rep-3
        build=lambda it, h=0: _base_reptile(
            -2 + _W6, [0, 1, _W6], it, cell=_HEXCELL, holes=h),
        n=3, samp=True),
    'TRIHEX_BAR': dict(                       # bar trihex, rep-3
        build=lambda it, h=0: _base_reptile(
            -2 + _W6, [0, 1, 2], it, cell=_HEXCELL, holes=h),
        n=3, samp=True),
    'TETRAHEX': dict(                         # tetrahex, rep-4
        build=lambda it, h=0: _base_reptile(
            2 - 2 * _W6, [0, 1 - _W6, -1, -_W6], it, cell=_HEXCELL,
            holes=h), n=4, samp=True),
    'HEX9': dict(                             # 9-hex, rep-9
        build=lambda it, h=0: _base_reptile(
            3 * (_W6 - 1),
            [0, _W6 - 1, 1 - _W6, -1, -_W6, _W6, 1, -2 + _W6, -1 - _W6],
            it, cell=_HEXCELL, holes=h), n=9, samp=True),
    'HEX13': dict(                            # 13-hex, rep-13
        build=lambda it, h=0: _base_reptile(
            -4 + _W6,
            [0, _W6 - 1, 1 - _W6, -1, -_W6, _W6, 1, -2 + _W6, -1 - _W6,
             2 * _W6 - 1, 1 - 2 * _W6, 1 + _W6, 2 - _W6],
            it, cell=_HEXCELL, holes=h), n=13, samp=True),
    # Reflection / foldable IFS kinds (quad cells; OSC attractors)
    'LEVY_DRAGON': dict(                      # SHK Fig 2c; Levy 1938
        build=lambda it, h=0: _ifs_reptile(_LEVY_MAPS, it, holes=h),
        n=2, samp=True, osc=True),
    'LEAF': dict(                             # SHK Fig 2d
        build=lambda it, h=0: _ifs_reptile(_LEAF_MAPS, it, holes=h),
        n=2, samp=True, osc=True),
    'FOLDABLE4': dict(                        # SHK Theorem 4.4
        build=lambda it, h=0: _ifs_reptile(_FOLD_MAPS, it, holes=h),
        n=4, samp=True, osc=True),
}


def fractal_patch(kind, iterations, holes=0):
    """Return (polys, types): CCW tile arrays and their level/colour
    indices.  holes > 0 drops that many substitution digits/maps at
    every level (Fathauer's fractal-gasket option); 0 is the full
    rep-tile.  The edge-grown RIGHT_TRIANGLE f-tiling has no
    substitution digits and ignores the option."""
    if kind not in KINDS:
        raise ValueError("unknown fractal rep-tile %r" % kind)
    polys, types = KINDS[kind]['build'](iterations, int(holes))
    return [_ensure_ccw(np.asarray(p, float)) for p in polys], types


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

# Grouped by polyform family.  The polyomino reptiles are Fathauer's
# radix construction (Bridges 2025): the digit set is the polyomino's
# cell positions, giving a solid rep-n self-affine tile.
KIND_ITEMS = [
    # --- Polyomino reptiles (square cells) ---
    ('TWINDRAGON', "Square: Domino (twindragon, rep-2)",
     "Domino, base -1+i, digits {0,1}: the Gilbert/Knuth twindragon "
     "(Vince Fig 3), aspect ratio 1/phi"),
    ('REP4', "Square: Square tetromino (rep-4)",
     "Base 2, digits {0,1,i,-1-i}: rep-4 Sierpinski-relative reptile "
     "(Vince Fig 6)"),
    ('Z_TETROMINO', "Square: Z-tetromino (twindragon, rep-4)",
     "Z-tetromino cell set {0,1,1+i,2+i} over base -2i = (-1+i)^2: the "
     "rep-4 route to the twindragon tile (Fathauer, Bridges 2025)"),
    ('REP5', "Square: X-pentomino (round rep-5)",
     "X-pentomino, base -2+i, digits {0,1,-1,i,-i}: the round rep-5 "
     "dragon (Vince Fig 5), aspect ratio 1/phi"),
    ('Z_PENTOMINO', "Square: Z-pentomino (rep-5)",
     "Z-pentomino cell set {0,1,1+i,1+2i,2+2i} over base -2-i: the "
     "rep-5 Z-pentomino fractal reptile (Fathauer, Bridges 2025)"),
    ('Y_PENTOMINO', "Square: Y-pentomino (rep-5)",
     "Y-pentomino cell set over base -2+i: the rep-5 Y-pentomino "
     "fractal reptile (Fathauer, Bridges 2025)"),
    ('P_PENTOMINO', "Square: P-pentomino (rep-5)",
     "P-pentomino cell set over base -2+i: the rep-5 P-pentomino "
     "fractal reptile (Fathauer, Bridges 2025)"),
    ('REP5_THIN', "Square: Bar pentomino (thin rep-5)",
     "Bar pentomino, base 2+i, collinear digits {0..4}: thin rep-5 "
     "dragon, aspect ratio 1/phi^3 (Mekhontsev (5,4))"),
    ('REP5B', "Square: Bar pentomino (rep-5, base 1+2i)",
     "Bar pentomino, base 1+2i, collinear digits {0..4}: rep-5 dragon, "
     "aspect ratio sqrt(2)-1 (Mekhontsev (5,2))"),
    ('Z_OCTOMINO', "Square: Z-octomino (rep-8)",
     "Z-octomino cell set over base -2-2i: the rep-8 Z-octomino "
     "fractal reptile (Fathauer, Bridges 2025)"),
    # --- Polyhex reptiles (hexagon cells) ---
    ('TRIHEX', "Hex: Trihex (rep-3)",
     "Trihex {0,1,w} over Eisenstein base -2+w (w=e^{i pi/3}): the "
     "rep-3 trihex fractal reptile (Fathauer, Bridges 2026)"),
    ('TRIHEX_BAR', "Hex: Trihex bar (rep-3)",
     "Bar trihex {0,1,2} over Eisenstein base -2+w: the rep-3 straight "
     "trihex fractal reptile (Fathauer, Bridges 2026)"),
    ('TETRAHEX', "Hex: Tetrahex (rep-4)",
     "Tetrahex over Eisenstein base 2-2w (theta=60): the rep-4 tetrahex "
     "fractal reptile (Fathauer, Bridges 2026)"),
    ('FLOWSNAKE', "Hex: Heptahex / Gosper (rep-7)",
     "Eisenstein base 5/2+(sqrt3/2)i, digits {0}+sixth roots of "
     "unity, hexagon cells: the rep-7 Gosper island bounded by the "
     "flowsnake curve (Vince Fig 4/7; Gardner 1976)"),
    ('HEX9', "Hex: 9-hex (rep-9)",
     "A compact 9-hex over Eisenstein base 3(w-1): the rep-9 polyhex "
     "fractal reptile (Fathauer, Bridges 2026)"),
    ('HEX13', "Hex: 13-hex (rep-13)",
     "A compact 13-hex over Eisenstein base -4+w: the rep-13 polyhex "
     "fractal reptile (Fathauer, Bridges 2026)"),
    # --- Reflection / foldable reptiles ---
    ('LEVY_DRAGON', "Reflection: Levy Dragon (rep-2)",
     "Two maps contracting by (1-i)/2, the second reflected in the "
     "x-axis: the Levy dragon rep-tile (Levy 1938; "
     "Sajid-Husain-Kumar Fig 2c)"),
    ('LEAF', "Reflection: Leaf Rep-Tile (rep-2)",
     "Same rep-2 contraction with the second map reflected in the "
     "y-axis (z to -conj z): the curled-leaf rep-tile "
     "(Sajid-Husain-Kumar Fig 2d)"),
    ('FOLDABLE4', "Reflection: Foldable Rep-4 (reflected gasket)",
     "Rep-4 digit system {0,1,i,-1-i} with an orientation-reversing "
     "quarter-turn map i*conj(z)/2: foldable reflection rep-4 tile "
     "(Sajid-Husain-Kumar Theorem 4.4; digits of Vince Fig 6)"),
    # --- Classic f-tiling & isometry ---
    ('RIGHT_TRIANGLE', "Classic: Right Triangle (f-tiling)",
     "Isosceles right-triangle f-tiling: square seed, a 1/sqrt2 child "
     "glued to every exposed leg; tiles shrink to a fractal boundary"),
    ('PENTABOLO', "Bolo: Pentabolo (rep-5 isometry gasket)",
     "Fathauer 5-isometry inflation of a half-square triangle: 5^k "
     "unit triangles, sqrt5 inflation per level, coloured by "
     "first-level copy"),
]

# Two-level Family -> Shape taxonomy for the operator UI, derived from
# the "Family: Shape" label prefixes above.
_FAMILY_ORDER = [('SQUARE', "Square"), ('HEX', "Hex"),
                 ('IAMOND', "Iamond"), ('RHOMB', "Rhomb"),
                 ('BOLO', "Bolo"), ('KITE', "Kite"),
                 ('RECTANGLE', "Rectangle"), ('REFLECTION', "Reflection"),
                 ('CLASSIC', "Classic")]
_FAMILY_LABELS = dict(_FAMILY_ORDER)
_FAMILY_OF = {}                 # kind id -> family id
_SHAPES_BY_FAMILY = {}          # family id -> [(kind, short label, desc)]
for _kid, _lbl, _desc in KIND_ITEMS:
    _fam = _lbl.split(':', 1)[0].strip().upper()
    if _fam not in _FAMILY_LABELS:
        _fam = 'CLASSIC'
    _FAMILY_OF[_kid] = _fam
    _short = _lbl.split(':', 1)[-1].strip()
    _SHAPES_BY_FAMILY.setdefault(_fam, []).append((_kid, _short, _desc))
_KIND_LABEL = dict((k, v) for k, v, _ in KIND_ITEMS)


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _shape_items_cache = {}

    def _shape_items(self, context):
        """Dynamic Shape enum: only the reptiles in the chosen Family.
        The returned list is cached in a module global so Blender does
        not garbage-collect the strings."""
        fam = getattr(self, 'family', 'SQUARE')
        items = _SHAPES_BY_FAMILY.get(fam) or _SHAPES_BY_FAMILY['CLASSIC']
        _shape_items_cache[fam] = items
        return items

    def _on_family(self, context):
        """When the Family changes, snap Shape to that family's first
        reptile so the Shape enum is never left on a now-invalid id."""
        shapes = _SHAPES_BY_FAMILY.get(self.family)
        if shapes:
            self.shape = shapes[0][0]

    class MESH_OT_fractal_reptile_add(bpy.types.Operator,
                                      AddObjectHelper):
        """Add a Fathauer fractal tiling built from a rep-tile
        prototile"""
        bl_idname = "mesh.fractal_reptile_add"
        bl_label = "Fractal Rep-Tile"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(
            name="Family",
            items=[(fid, fname, "%s fractal reptiles" % fname)
                   for fid, fname in _FAMILY_ORDER
                   if fid in _SHAPES_BY_FAMILY],
            default='SQUARE', update=_on_family,
            description="Polyform base-shape family")
        shape: EnumProperty(
            name="Shape", items=_shape_items,
            description="Which reptile within the chosen family")
        iterations: IntProperty(
            name="Iterations", default=6, min=0, max=14,
            description="Growth depth; each generation glues a "
                        "1/sqrt2-scaled child to every exposed leg "
                        "(capped to keep the mesh manageable)")
        holes: IntProperty(
            name="Holes", default=0, min=0, max=6,
            description="Fractal-gasket option (Fathauer): drop this "
                        "many of the rep-N substitution digits/maps "
                        "at every level, leaving (N-holes)^k cells "
                        "with self-similar holes; 0 = solid rep-tile "
                        "(ignored by the Right Triangle f-tiling)")
        color_by: EnumProperty(
            name="Color By",
            items=[('TYPE', "By Level",
                    "Right-triangle: a material per generation, "
                    "revealing the shrinking levels toward the fractal "
                    "boundary. Reptiles (twindragon, rep-5): a material "
                    "per pair of top substitution digits, showing the "
                    "outer copies and their nested sub-copies"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='TYPE')
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat 2D mesh; > 0 extrudes each tile")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object")

        def execute(self, context):
            kind = self.shape
            if kind not in KINDS:      # family just switched: shape id
                shapes = (_SHAPES_BY_FAMILY.get(self.family)
                          or _SHAPES_BY_FAMILY['CLASSIC'])
                kind = shapes[0][0]    # fall back to family's first shape
            polys, types = fractal_patch(kind, self.iterations,
                                         self.holes)
            cells = tg.cells_from_polys(
                lambda a, b: (polys, types), 1, 1, self.color_by,
                self.margin, self.height, False)
            label = _KIND_LABEL.get(kind, kind)
            obj = pc.emit(context, "Fractal RepTile %s" % label, cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no tiling generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s  V=%d F=%d" %
                            (label, len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s  %d tiles" %
                            (label, len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for p in ('family', 'shape', 'iterations', 'holes',
                      'color_by', 'margin', 'height'):
                lay.prop(self, p)
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.fractal_reptile_add",
                             icon='MOD_TRIANGULATE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_fractal_reptile_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_fractal_reptile_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _edges(poly):
    return [(poly[i], poly[(i + 1) % 3]) for i in range(3)]


def _hyp_paired(polys):
    """Every tile's hypotenuse (longest edge) must be shared full-edge
    with exactly one other tile (a child's hyp glues onto a parent's
    leg, and the seed pair shares its hyp)."""
    use = Counter()
    for p in polys:
        for a, b in _edges(p):
            use[frozenset((_key(a), _key(b)))] += 1
    for p in polys:
        h = _hyp_index(p)
        if use[frozenset((_key(p[h]), _key(p[(h + 1) % 3])))] != 2:
            return False
    return True


def _coverage(polys, pts, tol=1e-9):
    cnt = np.zeros(len(pts), int)
    for p in polys:
        lo, hi = p.min(axis=0), p.max(axis=0)
        m = ((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) &
             (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
        if not m.any():
            continue
        q = pts[m]
        ok = np.ones(len(q), bool)
        m3 = len(p)
        for i in range(m3):
            ax, ay = p[i]
            ex, ey = p[(i + 1) % m3] - p[i]
            ok &= (ex * (q[:, 1] - ay) - ey * (q[:, 0] - ax)) > tol
        cnt[m] += ok
    return cnt


def _boundary_loops(polys):
    use = Counter()
    pts = {}
    for p in polys:
        for a, b in _edges(p):
            ka, kb = _key(a), _key(b)
            use[frozenset((ka, kb))] += 1
            pts[ka], pts[kb] = a, b
    adj = {}
    for ek, c in use.items():
        if c != 1:
            continue
        a, b = tuple(ek)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in adj.values()):
        return None, None, False
    loops, visited = [], set()
    for start in adj:
        if start in visited:
            continue
        loop, prev, cur = [start], None, start
        visited.add(start)
        while True:
            nxt = [w for w in adj[cur] if w != prev]
            nxt = nxt[0] if nxt else prev
            if nxt == start:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        loops.append(np.array([pts[v] for v in loop], float))
    loops.sort(key=lambda lp: -abs(_signed_area(lp)))
    return loops[0], loops[1:], True


def _inside_loop(loop, pts):
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(loop)
    for i in range(n):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        if not cond.any():
            continue
        dy = (y2 - y1) if abs(y2 - y1) > 1e-30 else 1e-30
        xi = x1 + (y - y1) * (x2 - x1) / dy
        inside ^= cond & (x < xi)
    return inside


if __name__ == "__main__":
    all_ok = True
    for kind, label, _ in KIND_ITEMS:
        spec = KINDS[kind]
        tri = spec.get('tri', False)          # f-tiling (shrinking)
        tri_cells = spec.get('tri_cells', False)  # unit-triangle cells
        samp = spec.get('samp', False)        # hex/quad sampling cells
        osc = spec.get('osc', False)          # measure-zero boundary
        if tri:
            depths = (4, 7)
        elif tri_cells:
            depths = (2, 4)
        elif osc:
            depths = (8, 10) if spec['n'] == 2 else (5, 6)
        elif samp:
            depths = (2, 4)
        else:
            depths = (4, 7)
        for depth in depths:
            polys, types = fractal_patch(kind, depth)
            covered = 0
            if tri or tri_cells or samp:
                # polygon cells: dense-sampling overlap test
                allv = np.vstack(polys)
                lo, hi = allv.min(0), allv.max(0)
                gx, gy = np.meshgrid(
                    np.linspace(lo[0], hi[0], 240) + 0.00131,
                    np.linspace(lo[1], hi[1], 240) + 0.00069)
                pts = np.column_stack([gx.ravel(), gy.ravel()])
                cov = _coverage(polys, pts)
                overlaps = int((cov >= 2).sum())
                covered = int((cov >= 1).sum())
            else:
                # axis-aligned unit squares on the Gaussian-integer
                # lattice: overlap iff two cells share a position
                pos = Counter(_key(p.min(axis=0)) for p in polys)
                overlaps = sum(c - 1 for c in pos.values() if c > 1)
            if tri:
                e2e = _hyp_paired(polys)
                per = {}
                for p, t in zip(polys, types):
                    per[t] = max(per.get(t, 0.0),
                                 max(np.hypot(*(a - b))
                                     for a, b in _edges(p)))
                lv = sorted(per)
                sim = all(abs(per[lv[i + 1]] - per[lv[i]] / sqrt(2.0))
                          <= 1e-6 * per[lv[i]]
                          for i in range(len(lv) - 1))
                ok = e2e and sim and overlaps == 0
                extra = "edge2edge=%s shrink=%s" % (e2e, sim)
            elif osc:
                # OSC attractor: only measure-zero boundary overlaps
                # are allowed -- sampled hits must be a negligible
                # fraction of the covered points.  The 5% bound leaves
                # room for the Levy dragon, whose near-dimension-2
                # boundary (cf. Duvall-Keesling) keeps a few percent
                # of samples doubled at practical depths; the leaf and
                # foldable tiles measure ~0.
                n = spec['n']
                capped = len(polys) < n ** depth
                frac = overlaps / float(max(covered, 1))
                ok = ((len(polys) == n ** depth or capped)
                      and frac < 0.05)
                extra = "rep-%d count=%d(%d)%s ovfrac=%.4f" % (
                    n, len(polys), n ** depth,
                    " capped" if capped else "", frac)
            else:
                n = spec['n']
                capped = len(polys) < n ** depth
                ok = overlaps == 0 and (len(polys) == n ** depth
                                        or capped)
                extra = "rep-%d count=%d(%d)%s" % (
                    n, len(polys), n ** depth,
                    " capped" if capped else "")
            all_ok = all_ok and ok
            print("%-14s d=%d tiles=%5d overlaps=%d  %-28s %s"
                  % (kind, depth, len(polys), overlaps, extra,
                     "OK" if ok else "BAD"))
    # gasket runs (holes > 0): the same builders with substitution
    # digits/maps dropped must give exactly (N-holes)^k cells, still
    # distinct and non-overlapping (Fathauer's fractal gaskets)
    for kind, h, depth in (('REP4', 1, 7), ('REP5', 2, 7),
                           ('FOLDABLE4', 1, 6), ('PENTABOLO', 1, 4)):
        spec = KINDS[kind]
        n_eff = spec['n'] - h
        polys, types = fractal_patch(kind, depth, holes=h)
        if spec.get('samp') or spec.get('tri_cells'):
            allv = np.vstack(polys)
            lo, hi = allv.min(0), allv.max(0)
            gx, gy = np.meshgrid(
                np.linspace(lo[0], hi[0], 240) + 0.00131,
                np.linspace(lo[1], hi[1], 240) + 0.00069)
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            cov = _coverage(polys, pts)
            overlaps = int((cov >= 2).sum())
            if spec.get('osc'):
                covered = int((cov >= 1).sum())
                ov_ok = overlaps / float(max(covered, 1)) < 0.05
            else:
                ov_ok = overlaps == 0
        else:
            # lattice cells: distinct iff no two share a position
            pos = Counter(_key(p.min(axis=0)) for p in polys)
            overlaps = sum(c - 1 for c in pos.values() if c > 1)
            ov_ok = overlaps == 0
        ok = ov_ok and len(polys) == n_eff ** depth
        all_ok = all_ok and ok
        print("%-14s d=%d holes=%d tiles=%5d overlaps=%d "
              "count=(%d-%d)^%d=%d  %s"
              % (kind, depth, h, len(polys), overlaps,
                 spec['n'], h, depth, n_eff ** depth,
                 "OK" if ok else "BAD"))
    print("RESULT:", "OK" if all_ok else "BAD")
