
# Fractal Rep-Tile (Polygon f-tiling) Generator for Blender
#
# The headline feature is a full catalogue of Fathauer's POLYFORM
# fractal reptiles (Bridges 2025 & 2026): every polyomino, polyiamond,
# polyhex, polyrhomb, polybolo, polykite and polyrectangle reptile on
# his Fractal Diversions site, built by the unified polyform-iteration
# method (scale a tessellating polyform by 1/sqrt(n), rotate/reflect by
# a grid-allowed angle, re-assemble n copies, iterate).  See the
# "General polyform-iteration reptiles" section below for the method and
# citations.  The older constructions kept here (right-triangle
# f-tiling, complex-base radix reptiles, reflection/foldable IFS tiles,
# the pentabolo isometry gasket) are grouped under "Classic".
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
    "description": "Fractal rep-tiles: the full Fathauer polyform "
                   "catalogue (polyominoes, polyiamonds, polyhexes, "
                   "polyrhombs, polybolos, polykites, polyrectangles) "
                   "plus classic right-triangle f-tiling, complex-base "
                   "reptiles and reflection/foldable tiles",
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


def _ifs_reptile(maps, iterations, holes=0, seed=_IFS_SEED):
    """(polys, types) for the attractor of a conjugate-affine IFS:
    every length-k word applied to the `seed` cell (default the unit
    square; a triangle/hexagon/rhombus/kite for the polyform reptiles)
    gives m^k small cells, coloured by the outermost word digit (m
    classes -- the m cells of the generating polyform).
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
        z = _w_apply(w, seed)
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


# --------------------------------------------------------------------
# General polyform-iteration reptiles (Fathauer, Bridges 2025 & 2026)
#
# A starting polyform of n cells that tiles the plane by translation is
# scaled by 1/sqrt(n) about the origin and rotated by theta (or
# reflected about the theta/2 line); n copies are then re-assembled in
# exactly the arrangement by which the polyform was built.  Iterating
# converges to a rep-n self-replicating fractal tile.  This is precisely
# an IFS of n conjugate-affine maps w_i = T_i o S, where
#     S(z) = (e^{i theta}/sqrt n) z          (rotation), or
#     S(z) = (e^{i theta}/sqrt n) conj(z)    (reflection / mirror),
# and T_i is the rigid placement of cell i -- so the same _ifs_reptile
# renderer above draws every one of them, coloured by the outermost word
# digit (the n cells of the generating polyform).
#
# The allowed (n, theta) pairs are those for which sqrt(n) e^{-i theta}
# maps the cell grid onto itself: on the square grid the Gaussian
# integers a+bi with a^2+b^2 = n (n = 2,4,5,8,9,10,13,...); on the
# triangular / hexagonal / rhombille / deltoidal grid the Eisenstein
# integers a + b w with a^2+ab+b^2 = n (n = 3,4,7,9,12,13,...).  theta =
# -arg(base).  n = 6 is not an allowed grid norm, but the 6-rectangle
# (sqrt2 x 1 cells, base 2-sqrt2 i) and 6-rhombus (diagonals 1 and
# sqrt15/3) reptiles obtain it on a deliberately distorted grid.
#
# Reference:
# - Robert Fathauer, "Iterating Polyominoes to Create Fractal Reptiles",
#   Bridges 2025, pp. 93-100 (square-grid method + angle table).
# - Robert Fathauer, "Iterating Polyiamonds, Polyhexes, and other
#   Polyforms to Create Fractal Reptiles", Bridges 2026, pp. 85-92
#   (triangle/hex/rhombille/deltoidal grids + angle table).
# --------------------------------------------------------------------

_E2 = np.exp(1j * np.pi / 3.0)               # 60-degree lattice unit
_H3 = sqrt(3.0) / 2.0
_HEX_A1 = sqrt(3.0) + 0j                      # unit-edge hex lattice
_HEX_A2 = sqrt(3.0) * np.exp(1j * np.pi / 3.0)

# base cells (unit edge), as complex-vertex arrays
_CELL_TRI = np.array([0j, 1 + 0j, _E2])
_CELL_HEX = np.array([np.exp(1j * (np.pi / 6.0 + k * np.pi / 3.0))
                      for k in range(6)])
_CELL_RHO = np.array([0j, 1 + 0j, 1 + _E2, _E2])   # 60-120 (rhombille)
_KM = [_H3 * np.exp(1j * np.radians(60.0 * k)) for k in range(6)]
_CELL_KITE = np.array([0j, _KM[0], np.exp(1j * np.radians(30.0)), _KM[1]])
_CELL_BOLO = np.array([0j, 1 + 0j, 0.5 + 0.5j])    # half-square triangle
_CELL_RECT = np.array([0j, sqrt(2.0) + 0j, sqrt(2.0) + 1j, 1j])
_RU = sqrt(15.0) / 6.0 + 0.5j                       # 6-rhombus edges
_RV = sqrt(15.0) / 6.0 - 0.5j
_CELL_RHO6 = np.array([0j, _RU, _RU + _RV, _RV])


def _pf_gen(n, theta, mirror=False):
    """generator S = (1/sqrt n) e^{i theta} as (A, B, C); mirror uses
    the conjugate term (reflection about the theta/2 line)."""
    a = (1.0 / sqrt(n)) * np.exp(1j * np.radians(theta))
    return (0j, a, 0j) if mirror else (a, 0j, 0j)


def _pf_rigid(rot=0.0, t=0j):
    """rigid placement T(z) = e^{i rot} z + t as (A, B, C)."""
    return (np.exp(1j * np.radians(rot)), 0j, complex(t))


def _pf_maps(placements, n, theta, mirror=False):
    """the n IFS maps w_i = T_i o S for a polyform."""
    s = _pf_gen(n, theta, mirror)
    return [_w_compose(t, s) for t in placements]


# cell-placement builders (each returns a list of (A, B, C) rigids)
def _sq_place(ts):
    return [_pf_rigid(0.0, t) for t in ts]


def _tri_place(ijs):
    """triangular-grid cells (i, j, down): up-triangle translated, or a
    180-degree-rotated down-triangle."""
    out = []
    for i, j, s in ijs:
        base = i + j * _E2
        out.append(_pf_rigid(180.0, base + 1 + _E2) if s
                   else _pf_rigid(0.0, base))
    return out


def _hex_place(mks):
    """hex-grid cells at integer combos of the lattice vectors,
    re-centred on their centroid (so the origin is the symmetry
    centre)."""
    pts = [m * _HEX_A1 + k * _HEX_A2 for m, k in mks]
    c = sum(pts) / len(pts)
    return [_pf_rigid(0.0, p - c) for p in pts]


def _hex_fund(n, p, q, off=0j, r=8):
    """compact fundamental domain: the n hex-lattice cells nearest the
    origin, one per residue class of the Eisenstein base (p, q) -- a
    provably tiling rep-n polyhex (Bandt 1991)."""
    key = lambda m, k: (((p + q) * m + q * k) % n, ((-q) * m + p * k) % n)
    cand = sorted(((m, k) for m in range(-r, r + 1)
                   for k in range(-r, r + 1)),
                  key=lambda mk: abs(mk[0] * _HEX_A1 + mk[1] * _HEX_A2 - off))
    seen, mks = set(), []
    for m, k in cand:
        rk = key(m, k)
        if rk in seen:
            continue
        seen.add(rk)
        mks.append((m, k))
        if len(mks) == n:
            break
    return _hex_place(mks)


def _rect_fund(off=0j, r=7):
    """the 6-rectangle fundamental domain on the sqrt2 x 1 lattice with
    base 2 - sqrt2 i (norm 6)."""
    g1, g2 = sqrt(2.0) + 0j, 1j
    key = lambda m, k: ((2 * m - k) % 6, (2 * m + 2 * k) % 6)
    cand = sorted(((m, k) for m in range(-r, r + 1)
                   for k in range(-r, r + 1)),
                  key=lambda mk: abs(mk[0] * g1 + mk[1] * g2 - off))
    seen, ts = set(), []
    for m, k in cand:
        rk = key(m, k)
        if rk in seen:
            continue
        seen.add(rk)
        ts.append(m * g1 + k * g2)
        if len(ts) == 6:
            break
    return _sq_place(ts)


def _rt_place(rts):
    """explicit (rot_deg, x, y) placements (rhombille / deltoidal /
    tetrakis lattices, verified numerically)."""
    return [_pf_rigid(r, complex(x, y)) for r, x, y in rts]


# ---- the polyform preset registry -------------------------------------
# Each entry: (label, seed cell, n, theta, mirror, placements, grid).
# grid drives the self-test's overlap check ('square' -> lattice, else
# dense sampling).  Every preset is verified (cell count = n^k, sampled
# overlap ~ 0) by the self-test below.
def _pf(label, seed, n, theta, mirror, placements, grid):
    return dict(label=label, seed=np.asarray(seed, complex), n=n,
                theta=theta, mirror=mirror, cells=placements, grid=grid,
                pf=True)


_PF = {
    # --- polyominoes (square grid) ---
    'DOMINO': _pf("Domino (twindragon)", _IFS_SEED, 2,
                  45.0, False, _sq_place([0, 1]), 'square'),
    'Z_TETROMINO': _pf("Z-Tetromino (twindragon)", _IFS_SEED, 4, 90.0,
                       False, _sq_place([0, 1, 1 + 1j, 2 + 1j]), 'square'),
    'X_PENTOMINO': _pf("X-Pentomino", _IFS_SEED, 5, 26.565, False,
                       _sq_place([0, 1, -1, 1j, -1j]), 'square'),
    'Z_PENTOMINO': _pf("Z-Pentomino", _IFS_SEED, 5, -26.565, False,
                       _sq_place([0, 1, 1 + 1j, 1 + 2j, 2 + 2j]), 'square'),
    'Y_PENTOMINO': _pf("Y-Pentomino", _IFS_SEED, 5, 26.565, False,
                       _sq_place([0, 1j, 2j, 3j, 1 + 1j]), 'square'),
    'P_PENTOMINO': _pf("P-Pentomino", _IFS_SEED, 5, 26.565, False,
                       _sq_place([0, 1, 1j, 1 + 1j, 2j]), 'square'),
    'BAR_PENTOMINO': _pf("Bar Pentomino", _IFS_SEED, 5, 63.435, False,
                         _sq_place([0, 1, 2, 3, 4]), 'square'),
    'Z_OCTOMINO': _pf("Z-Octomino", _IFS_SEED, 8, 45.0, False,
                      _sq_place([0, 1, 2, 3, 1 + 1j, 2 + 1j, 3 + 1j,
                                 4 + 1j]), 'square'),
    'I_OCTOMINO': _pf("I-Octomino", _IFS_SEED, 8, 45.0, False,
                      _sq_place([0, 1, 2, 3, 4, 5, 6, 7]), 'square'),
    'FOURFOLD_OCTOMINO': _pf("Four-fold Octomino", _IFS_SEED, 8, 45.0,
                             False, _sq_place([0, -1, -1 - 1j, -1j, 1,
                                               -1 + 1j, -2 - 1j, -2j]),
                             'square'),
    # --- polyiamonds (triangular grid) ---
    'HEPTIAMOND': _pf("Heptiamond", _CELL_TRI, 7, 19.107, False,
                      _tri_place([(-1, 0, 1), (0, -1, 1), (0, 0, 0),
                                  (0, 0, 1), (1, -1, 0), (1, 0, 0),
                                  (1, 0, 1)]), 'tri'),
    'IAMOND12': _pf("12-Iamond", _CELL_TRI, 12, 90.0, False,
                    _tri_place([(-2, 0, 1), (-2, 1, 0), (-1, 0, 0),
                                (-1, 0, 1), (0, 0, 0), (0, 0, 1),
                                (1, -1, 1), (1, 0, 0), (1, 0, 1),
                                (2, -1, 1), (2, 0, 0), (2, 0, 1)]), 'tri'),
    'IAMOND13': _pf("13-Iamond", _CELL_TRI, 13, 73.898, True,
                    _tri_place([(-1, 0, 1), (0, -2, 1), (0, -1, 1),
                                (0, 0, 0), (0, 0, 1), (1, -2, 0),
                                (1, -2, 1), (1, -1, 0), (1, -1, 1),
                                (1, 0, 0), (2, -1, 0), (2, -1, 1),
                                (3, -1, 0)]), 'tri'),
    # --- polyhexes (hexagonal grid) ---
    'TRIHEX': _pf("Trihex (3-fold)", _CELL_HEX, 3, 30.0, False,
                  _hex_place([(0, 0), (1, 0), (0, 1)]), 'hex'),
    'TRIHEX_BAR': _pf("Trihex (bar)", _CELL_HEX, 3, 30.0, False,
                      _hex_place([(0, 0), (1, 0), (2, 0)]), 'hex'),
    'TETRAHEX': _pf("Tetrahex", _CELL_HEX, 4, 0.0, False,
                    _hex_place([(0, 0), (1, 0), (0, 1), (-1, 1)]), 'hex'),
    'HEPTAHEX': _pf("Heptahex (6-fold / Gosper)", _CELL_HEX, 7, 40.893,
                    False, _hex_place([(0, 0), (1, 0), (0, 1), (-1, 1),
                                       (-1, 0), (0, -1), (1, -1)]), 'hex'),
    'HEX9': _pf("9-Hex (3-fold)", _CELL_HEX, 9, 0.0, False,
                _hex_fund(9, 3, 0), 'hex'),
    'HEX12': _pf("12-Hex", _CELL_HEX, 12, 30.0, False,
                 _hex_fund(12, 4, -2), 'hex'),
    'HEX13': _pf("13-Hex", _CELL_HEX, 13, 13.898, False,
                 _hex_fund(13, 4, -1), 'hex'),
    # --- polyrhombs (rhombille grid, 60-120 rhombus) ---
    'TRIRHOMB': _pf("Trirhomb (terdragon envelope)", _CELL_RHO, 3, 90.0,
                    False, _rt_place([(60, -1.5, -_H3), (60, -2.5, -_H3),
                                      (60, -3.5, -_H3)]), 'rho'),
    'TETRARHOMB': _pf("Tetrarhomb", _CELL_RHO, 4, 60.0, False,
                      _rt_place([(60, -2.5, -_H3), (60, -3.5, -_H3),
                                 (60, -1.5, -_H3), (60, -4.5, -_H3)]),
                      'rho'),
    'HEPTARHOMB': _pf("Heptarhomb", _CELL_RHO, 7, 40.893, False,
                      _rt_place([(60, 0.5, -_H3), (60, 1.5, -_H3),
                                 (120, 3.0, 0.0), (60, -1.5, -_H3),
                                 (60, 2.5, -_H3), (60, -2.5, -_H3),
                                 (60, 3.5, -_H3)]), 'rho'),
    # --- polykites (deltoidal grid, 60-90-120-90 kite) ---
    'TRIKITE': _pf("Trikite", _CELL_KITE, 3, -30.0, False,
                   _rt_place([(0, -_H3, -1.5), (60, -_H3, -1.5),
                              (300, -_H3, -1.5)]), 'kite'),
    'TETRAKITE': _pf("Tetrakite", _CELL_KITE, 4, 60.0, False,
                     _rt_place([(120, _H3, -1.5), (60, _H3, -1.5),
                                (180, _H3, -1.5), (240, 0.0, 0.0)]),
                     'kite'),
    'TETRAKITE_0': _pf("Tetrakite (0-deg)", _CELL_KITE, 4, 0.0, False,
                       _rt_place([(120, _H3, -1.5), (60, _H3, -1.5),
                                  (180, _H3, -1.5), (240, 0.0, 0.0)]),
                       'kite'),
    # --- polybolo (tetrakis-square grid, half-square triangle) ---
    'TETRABOLO': _pf("Tetrabolo", _CELL_BOLO, 4, 0.0, False,
                     _rt_place([(90, -2.0, -1.0), (270, -2.0, 0.0),
                                (180, -2.0, 0.0), (0, -3.0, -1.0)]),
                     'bolo'),
    # --- n = 6 on distorted grids ---
    'RECT6': _pf("6-Rectangle (distorted grid)", _CELL_RECT, 6, 35.264,
                 False, _rect_fund(), 'rect'),
    'RHOMB6': _pf("6-Rhombus (distorted grid)", _CELL_RHO6, 6, 52.239,
                  False, _sq_place([m * _RU + k * _RV for m, k in
                                    [(0, 0), (-1, 0), (0, -1), (0, 1),
                                     (1, 0), (-1, 1)]]), 'rho'),
}


def _polyform_build(kind, iterations, holes=0, mirror=False):
    """(polys, types) for a polyform reptile preset; `mirror` toggles the
    preset's own mirror flag (reflect vs. rotate generator)."""
    s = _PF[kind]
    maps = _pf_maps(s['cells'], s['n'], s['theta'],
                    s['mirror'] ^ bool(mirror))
    return _ifs_reptile(maps, iterations, holes, seed=s['seed'])


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
    # Eisenstein radix-system kind (hexagon cells)
    'FLOWSNAKE': dict(                        # Vince Fig 4/7
        build=lambda it, h=0: _base_reptile(_FLOW_BASE, _FLOW_DIGITS,
                                            it, cell=_HEXCELL, holes=h),
        n=7, samp=True),
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


# register the polyform presets as KINDS metadata (build routed through
# _polyform_build via fractal_patch, so the mirror toggle reaches them)
for _k, _s in _PF.items():
    KINDS[_k] = dict(n=_s['n'], pf=True, grid=_s['grid'])


def fractal_patch(kind, iterations, holes=0, mirror=False):
    """Return (polys, types): CCW tile arrays and their level/colour
    indices.  holes > 0 drops that many substitution digits/maps at
    every level (Fathauer's fractal-gasket option); 0 is the full
    rep-tile.  The edge-grown RIGHT_TRIANGLE f-tiling has no
    substitution digits and ignores the option.  `mirror` toggles the
    reflect-vs-rotate generator of the polyform presets (ignored by the
    legacy kinds)."""
    if kind in _PF:
        polys, types = _polyform_build(kind, iterations, int(holes),
                                       mirror)
    elif kind in KINDS:
        polys, types = KINDS[kind]['build'](iterations, int(holes))
    else:
        raise ValueError("unknown fractal rep-tile %r" % kind)
    return [_ensure_ccw(np.asarray(p, float)) for p in polys], types


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

_PF_ORDER = [
    'DOMINO', 'Z_TETROMINO', 'X_PENTOMINO', 'Z_PENTOMINO', 'Y_PENTOMINO',
    'P_PENTOMINO', 'BAR_PENTOMINO', 'Z_OCTOMINO', 'I_OCTOMINO',
    'FOURFOLD_OCTOMINO',
    'HEPTIAMOND', 'IAMOND12', 'IAMOND13',
    'TRIHEX', 'TRIHEX_BAR', 'TETRAHEX', 'HEPTAHEX', 'HEX9', 'HEX12',
    'HEX13',
    'TRIRHOMB', 'TETRARHOMB', 'HEPTARHOMB',
    'TRIKITE', 'TETRAKITE', 'TETRAKITE_0',
    'TETRABOLO',
    'RECT6', 'RHOMB6',
]

_GRID_NAME = {'square': "Polyomino", 'tri': "Polyiamond", 'hex': "Polyhex",
              'rho': "Polyrhomb", 'kite': "Polykite", 'bolo': "Polybolo",
              'rect': "Polyrectangle"}


def _pf_item(k):
    s = _PF[k]
    fam = _GRID_NAME[s['grid']]
    desc = ("%s rep-%d reptile (Fathauer 2025/2026): iterate the "
            "polyform by scaling 1/sqrt(%d) and rotating %g deg%s; the "
            "Mirror option reflects instead" %
            (fam, s['n'], s['n'], s['theta'],
             ", mirrored" if s['mirror'] else ""))
    return (k, "%s: %s" % (fam, s['label']), desc)


KIND_ITEMS = [_pf_item(k) for k in _PF_ORDER] + [
    ('RIGHT_TRIANGLE', "Classic: Right Triangle (f-tiling)",
     "Isosceles right-triangle f-tiling: square seed, a 1/sqrt2 child "
     "glued to every exposed leg; tiles shrink to a fractal boundary"),
    ('PENTABOLO', "Classic: Pentabolo (rep-5 isometry gasket)",
     "Fathauer 5-isometry inflation of a half-square triangle: 5^k "
     "unit triangles, sqrt5 inflation per level, coloured by "
     "first-level copy"),
    ('TWINDRAGON', "Classic: Twindragon (rep-2, base -1+i)",
     "Gaussian base -1+i, digits {0,1}: the Gilbert/Knuth twindragon "
     "(Vince Fig 3), aspect ratio 1/phi"),
    ('REP4', "Rep-4 Reptile (base 2)",
     "Base 2, digits {0,1,i,-1-i}: rep-4 Sierpinski-relative reptile "
     "(Vince Fig 6)"),
    ('REP5', "Rep-5 Dragon (round, base -2+i)",
     "Base -2+i, symmetric digits {0,1,-1,i,-i}: the round rep-5 "
     "dragon (Vince Fig 5), aspect ratio 1/phi"),
    ('REP5_THIN', "Rep-5 Dragon (thin, AR 1/phi^3)",
     "Base 2+i, collinear digits {0..4}: thin rep-5 dragon, aspect "
     "ratio 1/phi^3 (Mekhontsev (5,4))"),
    ('REP5B', "Rep-5 Dragon (base 1+2i, AR sqrt2-1)",
     "Base 1+2i, collinear digits {0..4}: rep-5 dragon, aspect ratio "
     "sqrt(2)-1 (Mekhontsev (5,2))"),
    ('FLOWSNAKE', "Flowsnake / Gosper (rep-7)",
     "Eisenstein base 5/2+(sqrt3/2)i, digits {0}+sixth roots of "
     "unity, hexagon cells: the rep-7 Gosper island bounded by the "
     "flowsnake curve (Vince Fig 4/7; Gardner 1976)"),
    ('LEVY_DRAGON', "Levy Dragon (rep-2, reflection)",
     "Two maps contracting by (1-i)/2, the second reflected in the "
     "x-axis: the Levy dragon rep-tile (Levy 1938; "
     "Sajid-Husain-Kumar Fig 2c)"),
    ('LEAF', "Leaf Rep-Tile (rep-2, reflection)",
     "Same rep-2 contraction with the second map reflected in the "
     "y-axis (z to -conj z): the curled-leaf rep-tile "
     "(Sajid-Husain-Kumar Fig 2d)"),
    ('FOLDABLE4', "Foldable Rep-4 (reflected gasket)",
     "Rep-4 digit system {0,1,i,-1-i} with an orientation-reversing "
     "quarter-turn map i*conj(z)/2: foldable reflection rep-4 tile "
     "(Sajid-Husain-Kumar Theorem 4.4; digits of Vince Fig 6)"),
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

    class MESH_OT_fractal_reptile_add(bpy.types.Operator,
                                      AddObjectHelper):
        """Add a Fathauer fractal tiling built from a rep-tile
        prototile"""
        bl_idname = "mesh.fractal_reptile_add"
        bl_label = "Fractal Rep-Tile"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(name="Tiling", items=KIND_ITEMS,
                           default='RIGHT_TRIANGLE')
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
        mirror: BoolProperty(
            name="Mirror", default=False,
            description="Polyform reptiles only: reflect (instead of "
                        "rotate) between generations, giving the mirror "
                        "variant of the tile; ignored by the classic "
                        "kinds")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object")

        def execute(self, context):
            polys, types = fractal_patch(self.kind, self.iterations,
                                         self.holes, self.mirror)
            cells = tg.cells_from_polys(
                lambda a, b: (polys, types), 1, 1, self.color_by,
                self.margin, self.height, False)
            label = dict((k, v) for k, v, _ in KIND_ITEMS)[self.kind]
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
            for p in ('kind', 'iterations', 'holes', 'color_by',
                      'margin', 'height'):
                lay.prop(self, p)
            if self.kind in _PF:
                lay.prop(self, 'mirror')
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
        pf = spec.get('pf', False)            # general polyform reptile
        if tri:
            depths = (4, 7)
        elif tri_cells:
            depths = (2, 4)
        elif osc:
            depths = (8, 10) if spec['n'] == 2 else (5, 6)
        elif samp:
            depths = (2, 4)
        elif pf:
            depths = (2, {2: 11, 3: 7, 4: 6, 5: 5, 6: 4, 7: 4, 8: 4,
                          9: 3, 12: 3, 13: 3}.get(spec['n'], 4))
        else:
            depths = (4, 7)
        for depth in depths:
            polys, types = fractal_patch(kind, depth)
            covered = 0
            if tri or tri_cells or samp or pf:
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
            elif pf:
                # polyform reptile: exact tiling except a measure-zero
                # fractal boundary, so sampled overlap must be tiny
                n = spec['n']
                capped = len(polys) < n ** depth
                frac = overlaps / float(max(covered, 1))
                ok = ((len(polys) == n ** depth or capped)
                      and frac < 0.02)
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
