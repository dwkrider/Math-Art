
# Aperiodic Tiling Generator for Blender
#
# Non-periodic plane tilings -- finite patches whose local patterns
# repeat everywhere yet never form a translational period.  Three
# families are produced, each as a fixed patch of colored tiles fed
# through the shared Pattern Engine (see pattern_common.py) for color,
# grout margin and relief:
#
#   PENROSE_P3  Penrose rhombs (thick/thin).  Built by deflating the
#       two Robinson triangles -- an acute golden triangle (apex 36
#       deg) and an obtuse golden gnomon (apex 108 deg) -- from a
#       ten-triangle "sun" seed.  Each deflation step subdivides every
#       triangle into golden-ratio-smaller children (thin -> 1 thin +
#       1 thick, thick -> 2 thick + 1 thin); the resulting edge-to-edge
#       triangle mesh is then merged along shared base edges back into
#       the thin (36/144) and thick (72/108) rhombi.  types: 0 thin,
#       1 thick.
#
#   PENROSE_P2  Penrose kite & dart.  Built by deflating its own two
#       Robinson half-tiles -- the half-kite (acute golden triangle,
#       36-72-72 deg) and the half-dart (golden gnomon, 36-36-108 deg)
#       -- from a ten-half-kite "sun" seed (five kites around the
#       origin).  Each step subdivides a half-kite into 2 half-kites +
#       1 half-dart and a half-dart into 1 half-kite + 1 half-dart,
#       every child phi times smaller; mirror halves are then merged
#       across their symmetry axis into the full kite (convex quad,
#       angles 72/72/144/72) and dart (arrowhead quad, angles
#       72/36/216/36 with one reflex vertex).  Both tiles use the same
#       two edge lengths, in golden ratio.  types: 0 kite, 1 dart.
#
#   AMMANN_BEENKER  The octagonal (8-fold) aperiodic tiling of squares
#       and 45-degree rhombs.  Built by de Bruijn's generalized dual
#       (multigrid) method on four line families at 0/45/90/135 deg:
#       the dual of the line arrangement is exactly a rhombus tiling
#       whose prototiles are the unit square (perpendicular families)
#       and the 45-degree rhomb (adjacent families).  types: 0 square,
#       1 rhomb.  (This is the octagonal tiling produced rigorously as
#       a line-arrangement dual rather than by the 1+sqrt(2) inflation,
#       which lets every patch be coverage-verified directly.)
#
#   HAT / TURTLE / CHEVRON / COMET / CUSTOM  The 2023 einstein (aperiodic
#       monotile) and its whole one-parameter family Tile(a,b).  Every
#       member is a 14-edge polygon on the [3.4.6.4] kite lattice with
#       eight edges of length a (in the 60-degree directions a*xi^k,
#       xi = exp(2*pi*i/6)) and six of length b (the perpendicular
#       directions i*b*xi^k); a kite semi-angle A in [0, 90] sets
#       a = cos(A), b = sin(A).  Named members: Hat = Tile(1, sqrt3),
#       Turtle = Tile(sqrt3, 1), Chevron = Tile(0, 1), Comet = Tile(1, 0)
#       (the hat is the case where two collinear a-edges merge, giving its
#       familiar 13-vertex outline).  A patch is grown by the metatile
#       substitution on four clusters H, T, P, F -- inflate an H seed,
#       then expand each cluster into its hats -- and the whole family is
#       obtained by re-laying the same edge-to-edge combinatorics with the
#       parametrized tile.  The tiling is forced to use reflected tiles
#       (asymptotically about one in seven).  types: 0 unreflected, 1
#       reflected (or, when colored by cluster, the top-level H/T/P/F
#       metatile each tile belongs to).
#
#   SPECTRE  The 2023 chiral einstein.  Tile(1,1), the equilateral member
#       of the hat family.  The STRAIGHT-edged Tile(1,1) is only WEAKLY
#       chiral: on its own it still admits periodic tilings once reflected
#       copies are allowed, so it is not a strict chiral monotile.  Strict
#       chirality -- tilings by rotations and translations only, with no
#       reflections possible at all -- holds for the CURVED Spectre, whose
#       every edge is replaced by the standard S-curve (option
#       spectre_curved, on by default) so that no reflected copy can ever
#       mate.  Either way our substitution output is itself reflection-free
#       (rotations and translations only); the straight tile is offered so
#       the patch can be coverage-verified with simple polygons.  A patch
#       is grown by the nine-metatile substitution (Gamma..Psi), inflating
#       a Delta seed.  types: 0 ordinary spectre, 1 the paired tiles of
#       each "mystic" (Gamma) cluster.
#
# References:
#   Roger Penrose, "The role of aesthetics in pure and applied
#     mathematical research" (1974) -- the P2 and P3 aperiodic tilings.
#   N. G. de Bruijn, "Algebraic theory of Penrose's non-periodic tilings
#     of the plane" (1981) -- the multigrid / pentagrid dual method used
#     here for Ammann-Beenker.
#   Robert Ammann and F. P. M. Beenker -- the octagonal Ammann-Beenker
#     tiling (Beenker report, 1982).
#   David Smith, Joseph Samuel Myers, Craig S. Kaplan & Chaim
#     Goodman-Strauss, "An aperiodic monotile" and "A chiral aperiodic
#     monotile" (2023) -- the "hat" and "spectre" einsteins, the
#     continuous Tile(a,b) family that interpolates hat, turtle, chevron
#     and comet, and the curved-edge Spectre that is strictly chiral.

bl_info = {
    "name": "Aperiodic Tiling",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Penrose (rhombs, kite/dart) and Ammann-Beenker "
                   "octagonal aperiodic tilings as colored tile patterns",
    "category": "Add Mesh",
}

import cmath
from collections import deque
from math import pi, cos, sin, floor, radians, hypot

import numpy as np

try:
    from . import pattern_common as pc
    from . import tiling_generator as tg
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc
    import tiling_generator as tg


PHI = (1.0 + 5.0 ** 0.5) / 2.0


# --------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------

def _to_xy(pts):
    """Complex points -> (N, 2) float array."""
    return np.array([[p.real, p.imag] for p in pts])


def _close(x, y, tol=1e-7):
    return abs(x - y) <= tol * max(abs(x), abs(y), 1.0)


def _vkey(z, nd=7):
    return (round(z.real, nd), round(z.imag, nd))


def _signed_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(poly):
    """Return the polygon wound counter-clockwise (positive area)."""
    poly = np.asarray(poly, float)
    if _signed_area(poly) < 0.0:
        return poly[::-1].copy()
    return poly


# --------------------------------------------------------------------
# Penrose Robinson-triangle deflation (P3 rhombs, P2 kite/dart)
# --------------------------------------------------------------------

def _p3_seed():
    """Ten Robinson half-triangles forming the 5-fold "sun" wheel about
    the origin; every seed triangle is thin (color 0), apex at the
    origin, legs of length 1."""
    tris = []
    for i in range(10):
        b = cmath.rect(1.0, (2 * i - 1) * pi / 10)
        c = cmath.rect(1.0, (2 * i + 1) * pi / 10)
        if i % 2 == 0:
            b, c = c, b          # mirror every second wedge
        tris.append((0, 0j, b, c))
    return tris


def _p3_deflate(triangles):
    """One deflation step.  Thin (0) -> 1 thin + 1 thick; thick (1) ->
    2 thick + 1 thin.  Every child is scaled down by phi."""
    out = []
    for color, a, b, c in triangles:
        if color == 0:
            p = a + (b - a) / PHI
            out.append((0, c, p, b))
            out.append((1, p, c, a))
        else:
            q = b + (a - b) / PHI
            r = b + (c - b) / PHI
            out.append((1, r, c, a))
            out.append((1, q, r, b))
            out.append((0, r, q, a))
    return out


def _p3_triangles(generations):
    tris = _p3_seed()
    for _ in range(generations):
        tris = _p3_deflate(tris)
    return tris


def _apex_base(a, b, c):
    """For an isosceles Robinson triangle, return (apex, base0, base1):
    the apex is the vertex between the two equal legs; base0/base1 are
    the endpoints of the (unequal) base edge opposite it."""
    dab = abs(a - b)
    dbc = abs(b - c)
    dca = abs(c - a)
    if _close(dab, dca):          # legs a-b, a-c equal -> apex a
        return a, b, c
    if _close(dab, dbc):          # legs b-a, b-c equal -> apex b
        return b, a, c
    return c, a, b                 # legs c-a, c-b equal -> apex c


def _p3_rhombs(generations):
    """Deflate to Robinson triangles, then merge same-color triangles
    that share a base edge back into rhombi.  Returns a list of
    (type, (N, 2) array); type 0 = thin rhomb, 1 = thick rhomb.  Tiles
    left unpaired on the patch boundary stay as half-triangles."""
    tris = _p3_triangles(generations)
    groups = {}
    for color, a, b, c in tris:
        apex, b0, b1 = _apex_base(a, b, c)
        key = (color, frozenset((_vkey(b0), _vkey(b1))))
        groups.setdefault(key, []).append((apex, b0, b1))

    tiles = []
    for (color, _key), lst in groups.items():
        merged = False
        if len(lst) == 2:
            ap1, b0, b1 = lst[0]
            ap2 = lst[1][0]
            quad = _to_xy([ap1, b0, ap2, b1])
            a_tri = abs(_signed_area(_to_xy([ap1, b0, b1])))
            a_tri2 = abs(_signed_area(_to_xy([ap2, b0, b1])))
            # accept the merge only if the quad is exactly the union of
            # the two half-triangles (guards against any stray pairing)
            if abs(abs(_signed_area(quad)) - (a_tri + a_tri2)) < 1e-9:
                tiles.append((color, quad))
                merged = True
        if not merged:
            for apex, b0, b1 in lst:
                tiles.append((color, _to_xy([apex, b0, b1])))
    return tiles


# --------------------------------------------------------------------
# P2 kite/dart deflation
#
# Half-tile conventions (each stored as (color, a, b, c)):
#   color 0, half-kite  -- acute golden triangle, 36 deg apex at `a`,
#       equal legs a-b (kite boundary) and a-c (the kite's symmetry
#       axis, shared with the mirror half), short base b-c.
#   color 1, half-dart  -- golden gnomon, 108 deg apex at `a`, equal
#       short legs a-b (the dart's symmetry axis) and a-c (dart
#       boundary), long base b-c.
# Keeping track of which leg is the symmetry axis is what lets the
# mirror halves be re-joined into whole kites and darts afterwards.
# --------------------------------------------------------------------

def _p2_seed_sun():
    """Ten half-kites forming the 5-fold "sun" (five whole kites)
    about the origin: apexes at the origin, legs of length 1, each
    half's axis leg lying on an odd multiple of 36 degrees so that
    consecutive halves pair into kites."""
    tris = []
    for i in range(10):
        u = cmath.rect(1.0, pi * i / 5.0)
        v = cmath.rect(1.0, pi * (i + 1) / 5.0)
        if i % 2 == 0:
            tris.append((0, 0j, u, v))     # axis on the odd ray v
        else:
            tris.append((0, 0j, v, u))     # axis on the odd ray u
    return tris


def _p2_seed_star():
    """Ten half-darts forming the 5-fold "star" (five whole darts)
    about the origin: each dart's 72-degree vertex sits at the origin,
    its 108-degree reflex apex one short-leg out along an even multiple
    of 36 degrees, and its two long-base edges shared with the
    neighbouring darts.  Mirror halves pair across each dart's axis into
    whole darts, giving a five-pointed star centre (dual to the sun)."""
    tris = []
    for k in range(5):
        ax = 2.0 * pi * k / 5.0                    # axis / reflex-apex ray
        a = cmath.rect(1.0, ax)                    # 108-deg reflex apex
        cp = cmath.rect(PHI, ax + pi / 5.0)        # wing tip, +36 deg
        cm = cmath.rect(PHI, ax - pi / 5.0)        # wing tip, -36 deg
        tris.append((1, a, 0j, cp))                # half-dart, axis a-origin
        tris.append((1, a, 0j, cm))                # its mirror across a-origin
    return tris


def _p2_seed_cartwheel():
    """Conway's cartwheel: the decagonal configuration that occurs at the
    heart of every Penrose kite-and-dart tiling.  It is a sun with two
    coronas grown around it, which is exactly two inflations of the sun
    seed.  `_p2_deflate` subdivides a tile in place (finer tiles, same
    extent), so subdividing the sun twice and rescaling the result back
    to unit-length tiles performs two geometric inflations -- growing the
    patch outward by phi**2 while keeping the tiles unit sized.  The
    outcome is a central regular-decagon hub ringed by ten spokes with
    D5 symmetry.  `_p2_tiles` then deflates this outward exactly as it
    does the SUN and STAR seeds, so the same coverage/shape checks hold.
    (A similarity of a legal patch is still an edge-to-edge legal patch,
    which is what lets the cartwheel coverage-verify.)"""
    tris = _p2_seed_sun()
    for _ in range(2):
        tris = _p2_deflate(tris)
    s = PHI ** 2                                # restore unit-length tiles
    return [(color, a * s, b * s, c * s) for color, a, b, c in tris]


_P2_SEEDS = {'SUN': _p2_seed_sun, 'STAR': _p2_seed_star,
             'CARTWHEEL': _p2_seed_cartwheel}


def _p2_deflate(triangles):
    """One kite/dart deflation step.  Half-kite -> 2 half-kites + 1
    half-dart; half-dart -> 1 half-kite + 1 half-dart.  Every child is
    scaled down by phi, with the golden-ratio split points placed so
    children meet edge-to-edge and each child's axis leg stays the
    edge it shares with its mirror half in the full tiling."""
    out = []
    for color, a, b, c in triangles:
        if color == 0:
            # split the boundary leg a-b at p (|ap| = |ab| / phi^2)
            # and the axis leg a-c at s (|as| = |ac| / phi)
            p = a + (b - a) / PHI ** 2
            s = a + (c - a) / PHI
            out.append((0, b, p, s))       # half-kite, axis b-s
            out.append((0, b, c, s))       # its mirror across b-s
            out.append((1, p, a, s))       # half-dart, axis p-a
        else:
            # split the long base b-c at w (|bw| = |bc| / phi)
            w = b + (c - b) / PHI
            out.append((0, b, w, a))       # half-kite, axis b-a
            out.append((1, w, c, a))       # half-dart, axis w-c
    return out


def _p2_tiles(generations, seed='SUN'):
    """Deflate the chosen central seed, then merge each pair of mirror
    halves across their shared symmetry-axis edge into a whole tile:
    kite (convex quad, type 0) or dart (arrowhead quad with one reflex
    vertex, type 1).  Halves left unpaired on the ragged patch
    boundary stay as half-triangles.  `seed` selects the symmetric
    starting patch: 'SUN' (five kites round the centre) or 'STAR'
    (five darts round the centre)."""
    tris = _P2_SEEDS.get(seed, _p2_seed_sun)()
    for _ in range(generations):
        tris = _p2_deflate(tris)

    groups = {}
    for color, a, b, c in tris:
        ax0, ax1 = (a, c) if color == 0 else (a, b)
        key = (color, frozenset((_vkey(ax0), _vkey(ax1))))
        groups.setdefault(key, []).append((a, b, c))

    tiles = []
    for (color, _key), lst in groups.items():
        merged = False
        if len(lst) == 2:
            (a1, b1, c1), (a2, b2, c2) = lst
            if color == 0:      # kite: axis a-c, quad a, b1, c, b2
                quad = _to_xy([a1, b1, c1, b2])
            else:               # dart: axis a-b, quad b, c1, a, c2
                quad = _to_xy([b1, c1, a1, c2])
            t1 = abs(_signed_area(_to_xy([a1, b1, c1])))
            t2 = abs(_signed_area(_to_xy([a2, b2, c2])))
            # accept the merge only if the quad is exactly the union
            # of the two halves (guards against any stray pairing)
            if abs(abs(_signed_area(quad)) - (t1 + t2)) < 1e-9:
                tiles.append((color, quad))
                merged = True
        if not merged:
            for a, b, c in lst:
                tiles.append((color, _to_xy([a, b, c])))
    return tiles


# --------------------------------------------------------------------
# Ammann-Beenker octagonal tiling (de Bruijn generalized dual method)
# --------------------------------------------------------------------

# Four line-family normals at 0, 45, 90, 135 degrees.
_AB_DIRS = [np.array([cos(pi * j / 4.0), sin(pi * j / 4.0)])
            for j in range(4)]
# Generic offsets: distinct and chosen so no three grid lines are
# concurrent (which would spoil the pure square/rhomb dual).
_AB_GAMMA = [0.1000, 0.1700, 0.2700, 0.0500]


def _ab_gamma(phase, asymmetry):
    """Four multigrid shifts from a 2-parameter control.  asymmetry=0
    gives near-equal shifts -> the most 8-fold-symmetric tiling; higher
    asymmetry spreads them apart -> a generic, off-centre tiling.
    phase slides all shifts together, cycling through the tiling
    family.  A minimum spread is kept so no three grid lines are ever
    concurrent (which would leave gaps)."""
    g = np.array(_AB_GAMMA)
    mean = g.mean()
    a = 0.15 + 0.85 * float(asymmetry)
    return list(mean + a * (g - mean) + float(phase))


def _ammann_beenker(generations, gamma=None):
    """de Bruijn dual of a four-family octagonal grid.  Each pair of
    grid lines contributes one tile whose four corners are the images
    of the four grid cells meeting at their intersection.  `gamma` is
    the four grid shifts (phases) -- different shifts give different
    valid Ammann-Beenker tilings.  Returns a list of (type, (4, 2)
    array); type 0 = square, 1 = 45-degree rhomb."""
    if gamma is None:
        gamma = _AB_GAMMA
    n = len(_AB_DIRS)
    M = int(generations) + 3           # line index half-range
    tiles = []
    for j in range(n):
        nj = _AB_DIRS[j]
        for k in range(j + 1, n):
            nk = _AB_DIRS[k]
            det = nj[0] * nk[1] - nj[1] * nk[0]
            if abs(det) < 1e-9:
                continue
            square = (abs(k - j) == 2)     # perpendicular families
            for mj in range(-M, M + 1):
                cj = mj + gamma[j]
                for mk in range(-M, M + 1):
                    ck = mk + gamma[k]
                    # intersection p of the two lines nj.p=cj, nk.p=ck
                    px = (cj * nk[1] - ck * nj[1]) / det
                    py = (ck * nj[0] - cj * nk[0]) / det
                    p = np.array([px, py])
                    # fixed cell indices in the other two families
                    base = [0] * n
                    ok = True
                    for l in range(n):
                        if l == j or l == k:
                            continue
                        val = float(_AB_DIRS[l].dot(p)) - gamma[l]
                        if abs(val - round(val)) < 1e-6:
                            ok = False     # singular: skip degenerate tile
                            break
                        base[l] = floor(val)
                    if not ok:
                        continue
                    corners = []
                    for a, b in ((mj - 1, mk - 1), (mj, mk - 1),
                                 (mj, mk), (mj - 1, mk)):
                        idx = list(base)
                        idx[j] = a
                        idx[k] = b
                        pt = np.zeros(2)
                        for l in range(n):
                            pt = pt + idx[l] * _AB_DIRS[l]
                        corners.append(pt)
                    tiles.append((0 if square else 1, np.array(corners)))
    return tiles


# --------------------------------------------------------------------
# Einstein monotiles (2023): the "hat" and the "spectre"
#
# Both are generated by metatile substitution.  Tiles and metatiles are
# carried as 2x3 affine maps stored row-major as [a, b, c, d, e, f]
# (x' = a*x + b*y + c, y' = d*x + e*y + f); the small matrix helpers
# below mirror the classic reference construction.  Points are plain
# (x, y) tuples throughout this section.
# --------------------------------------------------------------------

_HR3 = 3.0 ** 0.5 / 2.0
_EIN_IDENT = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _ein_mul(A, B):
    """Compose two affine maps: (A . B)."""
    return [A[0] * B[0] + A[1] * B[3],
            A[0] * B[1] + A[1] * B[4],
            A[0] * B[2] + A[1] * B[5] + A[2],
            A[3] * B[0] + A[4] * B[3],
            A[3] * B[1] + A[4] * B[4],
            A[3] * B[2] + A[4] * B[5] + A[5]]


def _ein_inv(T):
    det = T[0] * T[4] - T[1] * T[3]
    return [T[4] / det, -T[1] / det, (T[1] * T[5] - T[2] * T[4]) / det,
            -T[3] / det, T[0] / det, (T[2] * T[3] - T[0] * T[5]) / det]


def _ein_trot(ang):
    c = cos(ang)
    s = sin(ang)
    return [c, -s, 0.0, s, c, 0.0]


def _ein_ttrans(tx, ty):
    return [1.0, 0.0, tx, 0.0, 1.0, ty]


def _ein_tpt(M, p):
    return (M[0] * p[0] + M[1] * p[1] + M[2],
            M[3] * p[0] + M[4] * p[1] + M[5])


def _ein_rot_about(p, ang):
    return _ein_mul(_ein_ttrans(p[0], p[1]),
                    _ein_mul(_ein_trot(ang), _ein_ttrans(-p[0], -p[1])))


def _ein_match_seg(p, q):
    """Similarity mapping the unit x-interval onto segment p->q."""
    return [q[0] - p[0], p[1] - q[1], p[0], q[1] - p[1], q[0] - p[0], p[1]]


def _ein_match_two(p1, q1, p2, q2):
    """Similarity carrying segment p1->q1 onto segment p2->q2."""
    return _ein_mul(_ein_match_seg(p2, q2), _ein_inv(_ein_match_seg(p1, q1)))


def _ein_intersect(p1, q1, p2, q2):
    """Intersection point of lines p1-q1 and p2-q2."""
    d = ((q2[1] - p2[1]) * (q1[0] - p1[0]) -
         (q2[0] - p2[0]) * (q1[1] - p1[1]))
    ua = ((q2[0] - p2[0]) * (p1[1] - p2[1]) -
          (q2[1] - p2[1]) * (p1[0] - p2[0])) / d
    return (p1[0] + ua * (q1[0] - p1[0]), p1[1] + ua * (q1[1] - p1[1]))


def _ein_det(T):
    return T[0] * T[4] - T[1] * T[3]


def _ein_linpart(T):
    """The 2x2 linear part (a, b, d, e) of an affine map, translation
    dropped."""
    return (T[0], T[1], T[3], T[4])


class _EinLeaf:
    """A single monotile: a fixed outline plus a color label.  `quad`
    holds the four mating keys used by the spectre substitution (unused
    by the hat)."""
    __slots__ = ('shape', 'label', 'quad')

    def __init__(self, shape, label, quad=None):
        self.shape = shape
        self.label = label
        self.quad = quad


class _EinMeta:
    """A metatile: an outline and a list of (transform, child) pairs.
    `quad` holds the four mating keys used by the spectre substitution."""
    __slots__ = ('shape', 'children', 'quad')

    def __init__(self, shape):
        self.shape = shape
        self.children = []
        self.quad = shape

    def add(self, T, geom):
        self.children.append((T, geom))

    def eval_child(self, n, i):
        T, geom = self.children[n]
        return _ein_tpt(T, geom.shape[i])

    def recentre(self):
        cx = sum(p[0] for p in self.shape) / len(self.shape)
        cy = sum(p[1] for p in self.shape) / len(self.shape)
        self.shape = [(p[0] - cx, p[1] - cy) for p in self.shape]
        M = _ein_ttrans(-cx, -cy)
        self.children = [(_ein_mul(M, T), g) for T, g in self.children]


def _ein_collect(geom, S, out):
    """Descend the metatile tree, emitting (label, det, outline) per
    monotile with the accumulated transform S applied."""
    if isinstance(geom, _EinLeaf):
        out.append((geom.label, _ein_det(S),
                    [_ein_tpt(S, p) for p in geom.shape]))
        return
    for T, child in geom.children:
        _ein_collect(child, _ein_mul(S, T), out)


def _ein_collect_tagged(geom, S, cluster, level, depth, out):
    """Like _ein_collect, but also threads a cluster tag (which
    top-level child of the root this leaf descends from) and a level tag
    (which second-level child), so tiles can be colored by supertile
    membership.  Emits (label, det, outline, cluster, level) per leaf."""
    if isinstance(geom, _EinLeaf):
        out.append((geom.label, _ein_det(S),
                    [_ein_tpt(S, p) for p in geom.shape], cluster, level))
        return
    for idx, (T, child) in enumerate(geom.children):
        c = idx if depth == 0 else cluster
        l = idx if depth == 1 else level
        _ein_collect_tagged(child, _ein_mul(S, T), c, l, depth + 1, out)


# --------------------------------------------------------------------
# The hat.  Its single tile is a 13-gon of eight kites from the [3.4.6.4]
# kite lattice.  Four metatiles H, T, P, F (clusters of hats) inflate on
# a fixed substitution; each H carries one reflected hat, so the tiling
# is forced to use mirrored tiles (asymptotically about one in seven).
# --------------------------------------------------------------------

def _hex_pt(x, y):
    return (x + 0.5 * y, _HR3 * y)


# The hat outline on the kite lattice (index order used by the metatiles).
_HAT_OUTLINE = [
    _hex_pt(0, 0), _hex_pt(-1, -1), _hex_pt(0, -2), _hex_pt(2, -2),
    _hex_pt(2, -1), _hex_pt(4, -2), _hex_pt(5, -1), _hex_pt(4, 0),
    _hex_pt(3, 0), _hex_pt(2, 2), _hex_pt(0, 3), _hex_pt(0, 2),
    _hex_pt(-1, 2)]


def _hat_bases():
    """The four seed metatiles H, T, P, F, each a cluster of whole hats.
    Reflected hats carry the label 'H1'."""
    ho = _HAT_OUTLINE
    hat = _EinLeaf(ho, 'H')
    hat_r = _EinLeaf(ho, 'H1')       # the reflected hat
    hat_t = _EinLeaf(ho, 'T')
    hat_p = _EinLeaf(ho, 'P')
    hat_f = _EinLeaf(ho, 'F')

    H_o = [(0, 0), (4, 0), (4.5, _HR3), (2.5, 5 * _HR3),
           (1.5, 5 * _HR3), (-0.5, _HR3)]
    H = _EinMeta(H_o)
    H.add(_ein_match_two(ho[5], ho[7], H_o[5], H_o[0]), hat)
    H.add(_ein_match_two(ho[9], ho[11], H_o[1], H_o[2]), hat)
    H.add(_ein_match_two(ho[5], ho[7], H_o[3], H_o[4]), hat)
    H.add(_ein_mul(_ein_ttrans(2.5, _HR3),
                   _ein_mul([-0.5, -_HR3, 0, _HR3, -0.5, 0],
                            [0.5, 0, 0, 0, -0.5, 0])), hat_r)

    T = _EinMeta([(0, 0), (3, 0), (1.5, 3 * _HR3)])
    T.add([0.5, 0, 0.5, 0, 0.5, _HR3], hat_t)

    P_o = [(0, 0), (4, 0), (3, 2 * _HR3), (-1, 2 * _HR3)]
    P = _EinMeta(P_o)
    P.add([0.5, 0, 1.5, 0, 0.5, _HR3], hat_p)
    P.add(_ein_mul(_ein_ttrans(0, 2 * _HR3),
                   _ein_mul([0.5, _HR3, 0, -_HR3, 0.5, 0],
                            [0.5, 0, 0, 0, 0.5, 0])), hat_p)

    F_o = [(0, 0), (3, 0), (3.5, _HR3), (3, 2 * _HR3), (-1, 2 * _HR3)]
    F = _EinMeta(F_o)
    F.add([0.5, 0, 1.5, 0, 0.5, _HR3], hat_f)
    F.add(_ein_mul(_ein_ttrans(0, 2 * _HR3),
                   _ein_mul([0.5, _HR3, 0, -_HR3, 0.5, 0],
                            [0.5, 0, 0, 0, 0.5, 0])), hat_f)
    return [H, T, P, F]


# Substitution: assemble 29 child metatiles in a fixed configuration,
# gluing each new metatile edge-to-edge against ones already placed.
# Rule forms: [name] seed; [child, edge, name, new_edge] match one edge;
# [childP, edgeP, childQ, edgeQ, name, new_edge] bridge two placed tiles.
_HAT_RULES = [
    ['H'],
    [0, 0, 'P', 2], [1, 0, 'H', 2], [2, 0, 'P', 2], [3, 0, 'H', 2],
    [4, 4, 'P', 2], [0, 4, 'F', 3], [2, 4, 'F', 3],
    [4, 1, 3, 2, 'F', 0],
    [8, 3, 'H', 0], [9, 2, 'P', 0], [10, 2, 'H', 0], [11, 4, 'P', 2],
    [12, 0, 'H', 2], [13, 0, 'F', 3], [14, 2, 'F', 1], [15, 3, 'H', 4],
    [8, 2, 'F', 1], [17, 3, 'H', 0], [18, 2, 'P', 0], [19, 2, 'H', 2],
    [20, 4, 'F', 3], [20, 0, 'P', 2], [22, 0, 'H', 2], [23, 4, 'F', 3],
    [23, 0, 'F', 3], [16, 0, 'P', 2],
    [9, 4, 0, 2, 'T', 2],
    [4, 0, 'F', 3]]


def _hat_patch_meta(H, T, P, F):
    """Build the 29-metatile substitution cluster from the four inputs."""
    shapes = {'H': H, 'T': T, 'P': P, 'F': F}
    ret = _EinMeta([])
    for r in _HAT_RULES:
        if len(r) == 1:
            ret.add(list(_EIN_IDENT), shapes[r[0]])
        elif len(r) == 4:
            T0, geom = ret.children[r[0]]
            poly = geom.shape
            pp = _ein_tpt(T0, poly[(r[1] + 1) % len(poly)])
            qq = _ein_tpt(T0, poly[r[1]])
            nshp = shapes[r[2]]
            np_ = nshp.shape
            ret.add(_ein_match_two(np_[r[3]], np_[(r[3] + 1) % len(np_)],
                                   pp, qq), nshp)
        else:
            Tp, gp = ret.children[r[0]]
            Tq, gq = ret.children[r[2]]
            pp = _ein_tpt(Tq, gq.shape[r[3]])
            qq = _ein_tpt(Tp, gp.shape[r[1]])
            nshp = shapes[r[4]]
            np_ = nshp.shape
            ret.add(_ein_match_two(np_[r[5]], np_[(r[5] + 1) % len(np_)],
                                   pp, qq), nshp)
    return ret


def _hat_inflate(patch):
    """Regroup a substitution cluster into the next-generation H, T, P, F
    metatiles (each claims its own subset of the cluster's children)."""
    bps1 = patch.eval_child(8, 2)
    bps2 = patch.eval_child(21, 2)
    rbps = _ein_tpt(_ein_rot_about(bps1, -2.0 * pi / 3.0), bps2)
    p72 = patch.eval_child(7, 2)
    p252 = patch.eval_child(25, 2)
    llc = _ein_intersect(bps1, rbps, patch.eval_child(6, 2), p72)

    w = (patch.eval_child(6, 2)[0] - llc[0],
         patch.eval_child(6, 2)[1] - llc[1])
    nH = [llc, bps1]
    w = _ein_tpt(_ein_trot(-pi / 3), w)
    nH.append((nH[1][0] + w[0], nH[1][1] + w[1]))
    nH.append(patch.eval_child(14, 2))
    w = _ein_tpt(_ein_trot(-pi / 3), w)
    nH.append((nH[3][0] - w[0], nH[3][1] - w[1]))
    nH.append(patch.eval_child(6, 2))
    new_H = _EinMeta(nH)
    for ch in [0, 9, 16, 27, 26, 6, 1, 8, 10, 15]:
        new_H.add(patch.children[ch][0], patch.children[ch][1])

    nP = [p72, (p72[0] + bps1[0] - llc[0], p72[1] + bps1[1] - llc[1]),
          bps1, llc]
    new_P = _EinMeta(nP)
    for ch in [7, 2, 3, 4, 28]:
        new_P.add(patch.children[ch][0], patch.children[ch][1])

    nF = [bps2, patch.eval_child(24, 2), patch.eval_child(25, 0), p252,
          (p252[0] + llc[0] - bps1[0], p252[1] + llc[1] - bps1[1])]
    new_F = _EinMeta(nF)
    for ch in [21, 20, 22, 23, 24, 25]:
        new_F.add(patch.children[ch][0], patch.children[ch][1])

    aaa = nH[2]
    bbb = (nH[1][0] + nH[4][0] - nH[5][0], nH[1][1] + nH[4][1] - nH[5][1])
    ccc = _ein_tpt(_ein_rot_about(bbb, -pi / 3), aaa)
    new_T = _EinMeta([bbb, ccc, aaa])
    new_T.add(patch.children[11][0], patch.children[11][1])

    for m in (new_H, new_T, new_P, new_F):
        m.recentre()
    return [new_H, new_T, new_P, new_F]


def _hat_tiles(generations):
    """Inflate from an H seed; return a list of (type, (N, 2) outline).
    type 0 = unreflected hat, 1 = reflected hat."""
    tiles = _hat_bases()
    for _ in range(int(generations)):
        tiles = _hat_inflate(_hat_patch_meta(*tiles))
    raw = []
    _ein_collect(tiles[0], list(_EIN_IDENT), raw)
    out = []
    for label, det, poly in raw:
        refl = 1 if (label == 'H1' or det < 0.0) else 0
        out.append((refl, _to_xy([complex(x, y) for x, y in poly])))
    return out


# --------------------------------------------------------------------
# The whole Tile(a,b) family (hat, turtle, chevron, comet, ...)
#
# Every member of the family shares one 14-edge combinatorial outline: an
# alternation of eight a-edges and six b-edges whose DIRECTIONS are fixed
# multiples of 30 degrees (a-edges on the even multiples, b-edges on the
# odd ones) and whose only free parameters are the two lengths a and b.
# At (a, b) = (1, sqrt3) two collinear a-edges merge and this reproduces
# the hat's classic 13-vertex outline exactly; every other choice deforms
# it continuously.  Because the substitution is edge-to-edge and every
# edge direction is length-independent, the whole tiling is obtained by a
# discrete developing map: run the exact hat substitution once to recover
# the combinatorics (which tile mates which, and each tile's rigid
# rotation/reflection), then re-lay the parametrized tile edge-to-edge.
# --------------------------------------------------------------------

# Directions (degrees) and a/b type of the 14 boundary edges, in order,
# in the hat's own frame.  Walking these with lengths (a for True,
# b for False) draws Tile(a,b); (1, sqrt3) reproduces _HAT_OUTLINE with
# one extra (collinear) vertex inserted where the hat's two a-edges merge.
_TILE_EDGE_DIR = [210, 300, 0, 0, 60, 330, 30, 120, 180, 90, 150, 240,
                  180, 270]
_TILE_EDGE_ISA = [False, True, True, True, True, False, False, True, True,
                  False, False, True, True, False]


def _tile_ab_outline(a, b):
    """The Tile(a,b) boundary as 14 (x, y) vertices: eight a-length edges
    and six b-length edges in the fixed 30-degree directions.  Degenerate
    members (a = 0 chevron, b = 0 comet) leave zero-length edges, i.e.
    repeated vertices, which _dedupe_poly then removes."""
    pts = []
    x = y = 0.0
    for d, isa in zip(_TILE_EDGE_DIR, _TILE_EDGE_ISA):
        pts.append((x, y))
        L = a if isa else b
        r = radians(d)
        x += L * cos(r)
        y += L * sin(r)
    return pts


def _insert_mid23(poly):
    """Insert the midpoint of edge (2, 3) at index 3, upgrading a
    13-vertex hat outline to the 14-vertex family indexing (the hat's two
    merged a-edges become two explicit edges, matching _tile_ab_outline)."""
    return (list(poly[:3]) +
            [((poly[2][0] + poly[3][0]) / 2.0,
              (poly[2][1] + poly[3][1]) / 2.0)] + list(poly[3:]))


def _poly_mirror_x(poly):
    return [(-p[0], p[1]) for p in poly]


def _dedupe_poly(poly, tol=1e-6):
    """Drop coincident and straight-angle (collinear) vertices, so the
    hat comes out as its exact 13-gon and degenerate members collapse to
    their true vertex counts."""
    pts = [(float(p[0]), float(p[1])) for p in poly]
    n = len(pts)
    out = []
    for i in range(n):
        a = pts[i - 1]
        b = pts[i]
        c = pts[(i + 1) % n]
        if abs(b[0] - a[0]) <= tol and abs(b[1] - a[1]) <= tol:
            continue
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        d1 = hypot(b[0] - a[0], b[1] - a[1])
        d2 = hypot(c[0] - b[0], c[1] - b[1])
        if d1 > tol and d2 > tol and abs(cross) <= tol * max(1.0, d1 * d2):
            continue
        out.append(b)
    return out


def _hat_reference(generations):
    """Run the exact hat substitution and collect every tile as
    (label, det, 13-vertex outline, cluster, level)."""
    tiles = _hat_bases()
    for _ in range(int(generations)):
        tiles = _hat_inflate(_hat_patch_meta(*tiles))
    raw = []
    _ein_collect_tagged(tiles[0], list(_EIN_IDENT), -1, -1, 0, raw)
    return raw


# Named members of the family, as exact (a, b) up to scale.  HAT keeps the
# integer (1, sqrt3) so it is numerically identical to the legacy hat.
_SQ3 = 3.0 ** 0.5
_TILE_PRESETS = {
    'HAT':     (1.0, _SQ3),
    'TURTLE':  (_SQ3, 1.0),
    'CHEVRON': (0.0, 1.0),
    'COMET':   (1.0, 0.0),
}


def _tile_ab_from(kind, semi_angle):
    """Resolve a hat-family kind to its (a, b).  CUSTOM reads the kite
    semi-angle A in degrees, a = cos(A), b = sin(A)."""
    if kind in _TILE_PRESETS:
        return _TILE_PRESETS[kind]
    A = radians(max(0.0, min(90.0, float(semi_angle))))
    return (cos(A), sin(A))


def _hatfamily_tiles(a, b, generations):
    """Tile(a,b) patch via the developing map.  Returns a list of
    (refl, (N, 2) outline, cluster, level); refl marks reflected tiles."""
    raw = _hat_reference(generations)
    U = _tile_ab_outline(a, b)
    Um = _poly_mirror_x(U)
    ho14 = _insert_mid23(_HAT_OUTLINE)
    ho14m = _poly_mirror_x(ho14)
    n = len(U)
    N = len(raw)

    ref14, refl, base, refbase, R, clu, lev = ([] for _ in range(7))
    for label, det, poly, cl, lv in raw:
        p14 = _insert_mid23(poly)
        r = 1 if (label == 'H1' or det < 0.0) else 0
        rb = ho14m if r else ho14
        ref14.append(p14)
        refl.append(r)
        clu.append(cl)
        lev.append(lv)
        base.append(Um if r else U)
        refbase.append(rb)
        S = _ein_match_two(rb[0], rb[1], p14[0], p14[1])
        R.append(_ein_linpart(S))

    def key(p):
        return (round(p[0], 3), round(p[1], 3))

    # Undirected shared-edge adjacency (reflected tiles wind the opposite
    # way, so directed matching would miss their edges).
    edgemap = {}
    adj = [[] for _ in range(N)]
    for h in range(N):
        P = ref14[h]
        for i in range(n):
            uk = frozenset((key(P[i]), key(P[(i + 1) % n])))
            if len(uk) < 2:                     # zero-length (degenerate)
                continue
            if uk in edgemap:
                h2, i2 = edgemap[uk]
                adj[h].append((i, h2, i2))
                adj[h2].append((i2, h, i))
            else:
                edgemap[uk] = (h, i)

    def apply(h, i, t):
        rl = R[h]
        p = base[h][i]
        return (rl[0] * p[0] + rl[1] * p[1] + t[0],
                rl[2] * p[0] + rl[3] * p[1] + t[1])

    # Root translation taken from the reference so (1, sqrt3) reproduces
    # the legacy hat patch exactly, position and all.
    S0 = _ein_match_two(refbase[0][0], refbase[0][1],
                        ref14[0][0], ref14[0][1])
    t = [None] * N
    t[0] = (S0[2], S0[5])
    dq = deque([0])
    while dq:
        h = dq.popleft()
        P = ref14[h]
        for (i, h2, i2) in adj[h]:
            if t[h2] is not None:
                continue
            P2 = ref14[h2]
            # match the shared edge vertex-to-vertex (works whichever way
            # the neighbour winds), using one corresponding vertex pair
            dst = i2 if key(P[i]) == key(P2[i2]) else (i2 + 1) % n
            wx, wy = apply(h, i, t[h])
            rl = R[h2]
            p = base[h2][dst]
            t[h2] = (wx - (rl[0] * p[0] + rl[1] * p[1]),
                     wy - (rl[2] * p[0] + rl[3] * p[1]))
            dq.append(h2)

    out = []
    for h in range(N):
        if t[h] is None:
            continue
        poly = _dedupe_poly([apply(h, i, t[h]) for i in range(n)])
        out.append((refl[h], _to_xy([complex(x, y) for x, y in poly]),
                    clu[h], lev[h]))
    return out


# --------------------------------------------------------------------
# The spectre.  Tile(1,1) is the equilateral member of the hat family; it
# tiles by rotations and translations only (no reflections), so it is a
# chiral monotile.  Nine metatiles Gamma, Delta, Theta, Lambda, Xi, Pi,
# Sigma, Phi, Psi inflate on a fixed rule; Gamma is the "mystic" pair.
# --------------------------------------------------------------------

# The Tile(1,1) outline (equal edge lengths), 14 vertices.
_SPECTRE_OUTLINE = [
    (0.0, 0.0), (1.0, 0.0),
    (1.5, -0.8660254037844386),
    (2.366025403784439, -0.36602540378443865),
    (2.366025403784439, 0.6339745962155614),
    (3.366025403784439, 0.6339745962155614),
    (3.866025403784439, 1.5),
    (3.0, 2.0),
    (2.133974596215561, 1.5),
    (1.6339745962155614, 2.3660254037844393),
    (0.6339745962155614, 2.3660254037844393),
    (-0.3660254037844386, 2.3660254037844393),
    (-0.866025403784439, 1.5),
    (0.0, 1.0)]
_SPECTRE_KEYS = [_SPECTRE_OUTLINE[3], _SPECTRE_OUTLINE[5],
                 _SPECTRE_OUTLINE[7], _SPECTRE_OUTLINE[11]]
_SPECTRE_LABELS = ['Delta', 'Theta', 'Lambda', 'Xi', 'Pi',
                   'Sigma', 'Phi', 'Psi']


def _spectre_bases():
    """The nine seed metatiles.  Eight are a single spectre; Gamma is the
    two-spectre 'mystic' cluster (both placed by rotation/translation)."""
    so = _SPECTRE_OUTLINE
    sys = {}
    for lab in _SPECTRE_LABELS:
        sys[lab] = _EinLeaf(so, lab, list(_SPECTRE_KEYS))
    mystic = _EinMeta(list(_SPECTRE_KEYS))
    mystic.add(list(_EIN_IDENT), _EinLeaf(so, 'Gamma1'))
    mystic.add(_ein_mul(_ein_ttrans(so[8][0], so[8][1]),
                        _ein_trot(pi / 6.0)), _EinLeaf(so, 'Gamma2'))
    sys['Gamma'] = mystic
    return sys


_SPECTRE_SUPER = {
    'Gamma':  ['Pi', 'Delta', None, 'Theta', 'Sigma', 'Xi', 'Phi', 'Gamma'],
    'Delta':  ['Xi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
    'Theta':  ['Psi', 'Delta', 'Pi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
    'Lambda': ['Psi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
    'Xi':     ['Psi', 'Delta', 'Pi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma'],
    'Pi':     ['Psi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma'],
    'Sigma':  ['Xi', 'Delta', 'Xi', 'Phi', 'Sigma', 'Pi', 'Lambda', 'Gamma'],
    'Phi':    ['Psi', 'Delta', 'Psi', 'Phi', 'Sigma', 'Pi', 'Phi', 'Gamma'],
    'Psi':    ['Psi', 'Delta', 'Psi', 'Phi', 'Sigma', 'Psi', 'Phi', 'Gamma']}


def _spectre_inflate(sys):
    """One inflation of the nine-metatile system.  The seven placement
    frames Ts are derived from the shared quad, then each supertile lists
    which sub-metatile sits in each frame."""
    quad = sys['Delta'].quad
    R = [-1.0, 0, 0, 0, 1.0, 0]
    t_rules = [(60, 3, 1), (0, 2, 0), (60, 3, 1), (60, 3, 1),
               (0, 2, 0), (60, 3, 1), (-120, 3, 3)]
    Ts = [list(_EIN_IDENT)]
    total = 0
    rot = list(_EIN_IDENT)
    tquad = list(quad)
    for ang, frm, to in t_rules:
        total += ang
        if ang != 0:
            rot = _ein_trot(total * pi / 180.0)
            tquad = [_ein_tpt(rot, quad[i]) for i in range(4)]
        src = _ein_tpt(Ts[-1], quad[frm])
        ttt = _ein_ttrans(src[0] - tquad[to][0], src[1] - tquad[to][1])
        Ts.append(_ein_mul(ttt, rot))
    Ts = [_ein_mul(R, t) for t in Ts]

    super_quad = [_ein_tpt(Ts[6], quad[2]), _ein_tpt(Ts[5], quad[1]),
                  _ein_tpt(Ts[3], quad[2]), _ein_tpt(Ts[0], quad[1])]
    out = {}
    for lab, subs in _SPECTRE_SUPER.items():
        sup = _EinMeta(list(super_quad))
        for idx in range(8):
            if subs[idx] is None:
                continue
            sup.add(Ts[idx], sys[subs[idx]])
        out[lab] = sup
    return out


# The standard Spectre edge curve: each straight edge is replaced by an
# S-curve that is point-symmetric about the edge midpoint, so the reversed
# edge a neighbour walks carries the exact complementary curve and the two
# nest with neither gap nor overlap.  Modelled as a cubic Bezier whose two
# control points bulge to opposite sides (a shallow S).
_SPECTRE_CURVE_AMP = 0.14           # bulge, as a fraction of edge length
_SPECTRE_CURVE_SEG = 8              # polyline segments per curved edge


def _s_curve(p, q, amp, seg):
    """Sample the S-curve from p toward q, returning `seg` points starting
    at p (q excluded, since the next edge begins there)."""
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    nx, ny = -dy, dx                        # left normal (edge length long)
    c1 = (p[0] + dx / 3.0 + amp * nx, p[1] + dy / 3.0 + amp * ny)
    c2 = (q[0] - dx / 3.0 - amp * nx, q[1] - dy / 3.0 - amp * ny)
    pts = []
    for k in range(seg):
        u = k / float(seg)
        w = 1.0 - u
        bx = (w * w * w * p[0] + 3 * w * w * u * c1[0] +
              3 * w * u * u * c2[0] + u * u * u * q[0])
        by = (w * w * w * p[1] + 3 * w * w * u * c1[1] +
              3 * w * u * u * c2[1] + u * u * u * q[1])
        pts.append((bx, by))
    return pts


def _curve_poly(poly, amp=_SPECTRE_CURVE_AMP, seg=_SPECTRE_CURVE_SEG):
    """Replace every straight edge of a CCW polygon with the Spectre
    S-curve.  Winding must be consistent (all CCW) so the left-normal
    bulge is complementary between edge-sharing neighbours."""
    poly = _ensure_ccw(np.asarray(poly, float))
    n = len(poly)
    out = []
    for i in range(n):
        out.extend(_s_curve(poly[i], poly[(i + 1) % n], amp, seg))
    return _to_xy([complex(x, y) for x, y in out])


def _spectre_tiles(generations, curved=False):
    """Inflate from a Delta seed; return a list of
    (type, (N, 2) outline, cluster, level).  The tiling is monochiral:
    type 0 for ordinary spectres, 1 for the two tiles of each 'mystic'
    (Gamma) cluster.  With `curved`, every edge becomes the Spectre
    S-curve (the strictly chiral tile); otherwise the straight Tile(1,1)
    polygon is used so the patch can be coverage-verified."""
    sys = _spectre_bases()
    for _ in range(int(generations)):
        sys = _spectre_inflate(sys)
    raw = []
    _ein_collect_tagged(sys['Delta'], list(_EIN_IDENT), -1, -1, 0, raw)
    out = []
    for label, _det, poly, cl, lv in raw:
        typ = 1 if label in ('Gamma1', 'Gamma2') else 0
        arr = _to_xy([complex(x, y) for x, y in poly])
        if curved:
            arr = _curve_poly(arr)
        out.append((typ, arr, cl, lv))
    return out


# --------------------------------------------------------------------
# Public patch builder
# --------------------------------------------------------------------

# Hat-family kinds share one generator; only their (a, b) differs.
_HAT_FAMILY = ('HAT', 'TURTLE', 'CHEVRON', 'COMET', 'CUSTOM')


def _tag_of(color_by, typ, cluster, level):
    """Pick the per-tile material index for the requested coloring.  TYPE
    keeps the tile-kind (0/1); CLUSTER / SUPERTILE_LEVEL use the threaded
    supertile tags; UNIFORM is a single material."""
    if color_by == 'CLUSTER' and cluster is not None and cluster >= 0:
        return int(cluster)
    if color_by == 'SUPERTILE_LEVEL' and level is not None and level >= 0:
        return int(level)
    if color_by == 'UNIFORM':
        return 0
    return int(typ)


def aperiodic_patch(kind, generations, seed='SUN', phase=0.0,
                    asymmetry=0.0, semi_angle=60.0, spectre_curved=True,
                    color_by='TYPE'):
    """Return (polys, types) for a finite aperiodic tile patch.

    polys -- list of (N, 2) CCW float arrays (one per tile)
    types -- list of int tile-kind indices, parallel to polys
    seed  -- central-patch choice; only used by PENROSE_P2 ('SUN' or
             'STAR'), ignored by every other kind
    phase, asymmetry -- Ammann-Beenker multigrid shift controls; only
             used by AMMANN_BEENKER, ignored by every other kind
    semi_angle -- kite semi-angle A (degrees) for the CUSTOM hat-family
             tile, a = cos A, b = sin A; ignored by the fixed presets and
             every non-hat kind
    spectre_curved -- replace each Spectre edge with the S-curve (the
             strictly chiral tile); only used by SPECTRE
    color_by -- 'TYPE', 'UNIFORM', 'CLUSTER' or 'SUPERTILE_LEVEL'; selects
             which per-tile tag is written into `types`
    """
    tagged = None                      # list of (type, poly, cluster, lev)
    if kind == 'PENROSE_P3':
        raw = _p3_rhombs(generations)
        scale = PHI ** generations     # unit-edge tiles, patch ~ phi^g
    elif kind == 'PENROSE_P2':
        raw = _p2_tiles(generations, seed)
        scale = PHI ** generations
    elif kind == 'AMMANN_BEENKER':
        raw = _ammann_beenker(generations, _ab_gamma(phase, asymmetry))
        scale = 1.0
    elif kind in _HAT_FAMILY:
        a, b = _tile_ab_from(kind, semi_angle)
        tagged = _hatfamily_tiles(a, b, generations)
        scale = 1.0
    elif kind == 'SPECTRE':
        tagged = _spectre_tiles(generations, spectre_curved)
        scale = 1.0
    else:
        raise ValueError("unknown aperiodic kind %r" % kind)

    polys, types = [], []
    if tagged is not None:
        for t, poly, cluster, level in tagged:
            poly = _ensure_ccw(np.asarray(poly, float) * scale)
            polys.append(poly)
            types.append(_tag_of(color_by, t, cluster, level))
    else:
        for t, poly in raw:
            poly = _ensure_ccw(np.asarray(poly, float) * scale)
            polys.append(poly)
            types.append(int(t))
    return polys, types


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

KIND_ITEMS = [
    ('PENROSE_P3', "Penrose Rhombs (P3)",
     "Thick/thin Penrose rhombi via Robinson-triangle deflation"),
    ('PENROSE_P2', "Penrose Kite & Dart (P2)",
     "Kites and darts via half-kite/half-dart triangle deflation"),
    ('AMMANN_BEENKER', "Ammann-Beenker (Octagonal)",
     "Squares and 45-degree rhombs, the 8-fold octagonal tiling"),
    ('HAT', "Hat (Einstein Monotile)",
     "The 2023 aperiodic monotile Tile(1,sqrt3) via H/T/P/F metatile "
     "substitution; color marks the reflected hats (about one in seven)"),
    ('TURTLE', "Turtle",
     "Tile(sqrt3,1), the turtle member of the hat family -- same "
     "substitution, longer a-edges"),
    ('CHEVRON', "Chevron",
     "Tile(0,1), the degenerate chevron member of the hat family "
     "(a-edges shrink to zero)"),
    ('COMET', "Comet",
     "Tile(1,0), the degenerate comet member of the hat family "
     "(b-edges shrink to zero)"),
    ('CUSTOM', "Custom Tile(a,b)",
     "Any member of the hat family; the Kite Semi-Angle slider sets "
     "a = cos(A), b = sin(A)"),
    ('SPECTRE', "Spectre (Chiral Monotile)",
     "The 2023 chiral aperiodic monotile Tile(1,1); with curved edges it "
     "is strictly chiral, color marks the 'mystic' pairs"),
]


# Central-patch seed for the Penrose kite & dart (P2) tiling only.
P2_SEED_ITEMS = [
    ('SUN', "Sun (Five Kites)",
     "Deflate from five kites meeting at the centre -- a sunburst core"),
    ('STAR', "Star (Five Darts)",
     "Deflate from five darts meeting at the centre -- a 5-pointed-star "
     "core (dual to the sun)"),
    ('CARTWHEEL', "Cartwheel",
     "Deflate from Conway's cartwheel -- a central decagon hub ringed by "
     "ten spokes (the sun grown by two coronas)"),
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

    class MESH_OT_aperiodic_add(bpy.types.Operator, AddObjectHelper):
        """Add an aperiodic tiling patch"""
        bl_idname = "mesh.aperiodic_add"
        bl_label = "Aperiodic Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(name="Tiling", items=KIND_ITEMS,
                           default='PENROSE_P3')
        seed: EnumProperty(
            name="P2 Seed", items=P2_SEED_ITEMS, default='SUN',
            description="Central patch the Penrose kite & dart deflation "
                        "starts from (only affects the P2 tiling)")
        generations: IntProperty(
            name="Generations", default=4, min=0, max=7,
            description="Deflation / grid depth (each Penrose step "
                        "multiplies the tile count by roughly phi^2)")
        color_by: EnumProperty(
            name="Color By",
            items=[('TYPE', "By Tile Type",
                    "Material per tile kind (thin/thick, kite/dart, "
                    "square/rhomb, reflected/mystic)"),
                   ('CLUSTER', "By Cluster",
                    "Material per top-level metatile cluster (einstein "
                    "monotiles only)"),
                   ('SUPERTILE_LEVEL', "By Supertile Level",
                    "Material per second-level supertile (einstein "
                    "monotiles only)"),
                   ('UNIFORM', "Uniform", "A single material")],
            default='TYPE')
        margin: FloatProperty(
            name="Margin", default=0.0, min=0.0, max=0.45,
            description="Inset each tile toward its centroid, leaving "
                        "grout lines between tiles")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat 2D mesh; > 0 extrudes each tile")
        trim: BoolProperty(
            name="Trim Boundary", default=False,
            description="Clip the roughly-circular patch to a clean "
                        "central rectangle, removing the ragged edge")
        separate: BoolProperty(
            name="Separate Tiles", default=False,
            description="Output each tile as its own mesh object "
                        "(parented to an empty) so tiles can be edited "
                        "individually")
        phase: FloatProperty(
            name="Phase", default=0.0, min=0.0, max=1.0,
            description="Ammann-Beenker grid phase: slides through the "
                        "family of tilings (only affects Ammann-Beenker)")
        asymmetry: FloatProperty(
            name="Asymmetry", default=0.0, min=0.0, max=1.0,
            description="Ammann-Beenker symmetry break: 0 = the 8-fold "
                        "symmetric tiling, higher = off-centre / generic "
                        "(only affects Ammann-Beenker)")
        semi_angle: FloatProperty(
            name="Kite Semi-Angle", default=60.0, min=0.0, max=90.0,
            description="Custom hat-family tile: kite semi-angle A in "
                        "degrees, a = cos(A), b = sin(A) (60 = Hat, "
                        "30 = Turtle, 45 = Spectre; only affects Custom)")
        spectre_curved: BoolProperty(
            name="Curved Edges", default=True,
            description="Replace each Spectre edge with the standard "
                        "S-curve, giving the strictly chiral tile (only "
                        "affects Spectre)")

        def execute(self, context):
            polys, types = aperiodic_patch(
                self.kind, self.generations, self.seed, self.phase,
                self.asymmetry, self.semi_angle, self.spectre_curved,
                self.color_by)
            # CLUSTER / SUPERTILE_LEVEL are already baked into `types`, so
            # feed cells_from_polys plain TYPE coloring (or UNIFORM).
            cb = 'UNIFORM' if self.color_by == 'UNIFORM' else 'TYPE'
            cells = tg.cells_from_polys(
                lambda a, b: (polys, types), 1, 1, cb,
                self.margin, self.height, self.trim)
            label = dict((k, v) for k, v, _ in KIND_ITEMS)[self.kind]
            obj = pc.emit(context, "Aperiodic %s" % label, cells,
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
            lay.prop(self, 'kind')
            if self.kind == 'PENROSE_P2':
                lay.prop(self, 'seed')
            if self.kind == 'AMMANN_BEENKER':
                lay.prop(self, 'phase')
                lay.prop(self, 'asymmetry')
            if self.kind == 'CUSTOM':
                lay.prop(self, 'semi_angle')
            if self.kind == 'SPECTRE':
                lay.prop(self, 'spectre_curved')
            for p in ('generations', 'color_by', 'margin', 'height'):
                lay.prop(self, p)
            lay.prop(self, 'trim')
            lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.aperiodic_add", icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_aperiodic_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_aperiodic_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _coverage(polys, samples=30, half=3.0, off=(0.0137, 0.0071)):
    """Sample a `samples` x `samples` grid over an interior window of
    half-width `half`, centred on the patch bbox centre and nudged by
    `off` to avoid landing on tile edges.  Return (n_covered, n_over):
    points covered by >=1 tile, and points covered by >=2 (overlap
    defects)."""
    allv = np.vstack([np.asarray(p, float) for p in polys])
    cx = 0.5 * (allv[:, 0].min() + allv[:, 0].max())
    cy = 0.5 * (allv[:, 1].min() + allv[:, 1].max())
    n_cov = n_over = 0
    for ix in range(samples):
        x = cx - half + 2.0 * half * ix / (samples - 1) + off[0]
        for iy in range(samples):
            y = cy - half + 2.0 * half * iy / (samples - 1) + off[1]
            cnt = 0
            for p in polys:
                if tg._pip(p, x, y):
                    cnt += 1
                    if cnt >= 2:
                        break
            if cnt >= 1:
                n_cov += 1
            if cnt >= 2:
                n_over += 1
    return n_cov, n_over


def _reflex_count(poly):
    """Number of reflex (interior angle > 180 deg) vertices of a CCW
    simple polygon."""
    poly = _ensure_ccw(poly)
    n = len(poly)
    cnt = 0
    for k in range(n):
        v1 = poly[k] - poly[k - 1]
        v2 = poly[(k + 1) % n] - poly[k]
        if v1[0] * v2[1] - v1[1] * v2[0] < -1e-9:
            cnt += 1
    return cnt


def _p2_selftest(seed='SUN'):
    """P2-specific checks: kite:dart ratio -> phi, no loose triangles
    in the patch interior, correct kite/dart shapes and golden-ratio
    edges.  Returns (ok, message).  `seed` selects the central patch."""
    polys, types = aperiodic_patch('PENROSE_P2', 5, seed)
    kites = sum(1 for p, t in zip(polys, types)
                if len(p) == 4 and t == 0)
    darts = sum(1 for p, t in zip(polys, types)
                if len(p) == 4 and t == 1)
    kd = kites / darts if darts else 0.0
    kd_ok = darts > 0 and abs(kd - PHI) / PHI <= 0.05

    # every tile well inside the patch must be a quad (kite or dart);
    # only the ragged patch boundary may carry unpaired half-triangles.
    # An interior stray is a triangle that is fully surrounded (all three
    # edges shared with a neighbour); a boundary half always keeps at
    # least one exposed edge.  This edge test is patch-shape agnostic, so
    # it holds for the round SUN patch and the 5-pointed STAR patch alike.
    emul = {}
    for p in polys:
        n = len(p)
        for k in range(n):
            e = frozenset((_vkey(complex(*p[k])),
                           _vkey(complex(*p[(k + 1) % n]))))
            emul[e] = emul.get(e, 0) + 1
    stray = 0
    for p in polys:
        n = len(p)
        if n != 3:
            continue
        if all(emul[frozenset((_vkey(complex(*p[k])),
                               _vkey(complex(*p[(k + 1) % n]))))] >= 2
               for k in range(n)):
            stray += 1

    shapes_ok = True
    for p, t in zip(polys, types):
        if len(p) != 4:
            continue
        edges = sorted(float(np.hypot(*(p[(k + 1) % 4] - p[k])))
                       for k in range(4))
        golden = (_close(edges[0], edges[1], 1e-6) and
                  _close(edges[2], edges[3], 1e-6) and
                  _close(edges[2] / edges[0], PHI, 1e-6))
        reflex = _reflex_count(p)
        if not golden or reflex != (0 if t == 0 else 1):
            shapes_ok = False
            break
    ok = kd_ok and stray == 0 and shapes_ok
    msg = ("seed=%-4s kites=%d darts=%d k:d=%.4f(~phi=%.4f) "
           "interior_tris=%d shapes=%s  %s"
           % (seed, kites, darts, kd, PHI, stray,
              "ok" if shapes_ok else "bad",
              "OK" if ok else "BAD"))
    return ok, msg


def _hat_selftest():
    """HAT-specific check: the tiling must contain both reflected and
    unreflected hats in a ratio near 1:6-7.  Returns (ok, message)."""
    _polys, types = aperiodic_patch('HAT', 3)
    unref = sum(1 for t in types if t == 0)
    refl = sum(1 for t in types if t == 1)
    ratio = unref / refl if refl else 0.0
    both = refl > 0 and unref > 0
    ok = both and 5.0 <= ratio <= 8.5           # asymptote near 1:6-7
    msg = ("unreflected=%d reflected=%d ratio=1:%.2f both_present=%s  %s"
           % (unref, refl, ratio, both, "OK" if ok else "BAD"))
    return ok, msg


def _hat_regression():
    """The HAT preset must reproduce the legacy hat patch exactly: same
    tile count, same tiles (as unordered centroid + area sets)."""
    new, _t = aperiodic_patch('HAT', 2)
    old = [poly for _typ, poly in _hat_tiles(2)]

    def sig(P):
        s = []
        for p in P:
            p = np.asarray(p, float)
            c = p.mean(axis=0)
            s.append((round(float(c[0]), 4), round(float(c[1]), 4),
                      round(abs(_signed_area(p)), 4)))
        return sorted(s)
    same = len(new) == len(old) and sig(new) == sig(old)
    msg = ("legacy_tiles=%d new_tiles=%d identical=%s  %s"
           % (len(old), len(new), same, "OK" if same else "BAD"))
    return same, msg


def _mono_half(polys, frac=0.20):
    """A coverage half-width guaranteed to sit inside the (roughly round)
    monotile patch, whatever the family member's overall scale."""
    allv = np.vstack([np.asarray(p, float) for p in polys])
    ext = min(allv[:, 0].max() - allv[:, 0].min(),
              allv[:, 1].max() - allv[:, 1].min())
    return frac * ext


if __name__ == "__main__":
    all_ok = True
    n_samples = 30 * 30                         # _coverage grid size
    # Per-kind generation depths (lo, mid, hi); the mid patch is coverage
    # tested.  The Penrose/Ammann triples are unchanged; the monotiles use
    # smaller depths so the tested patch is a few hundred to ~1000 tiles.
    gens_map = {'PENROSE_P3': (3, 4, 5), 'PENROSE_P2': (3, 4, 5),
                'AMMANN_BEENKER': (3, 4, 5), 'HAT': (1, 2, 3),
                'TURTLE': (1, 2, 3), 'CHEVRON': (1, 2, 3),
                'COMET': (1, 2, 3), 'CUSTOM': (1, 2, 3),
                'SPECTRE': (2, 3, 4)}
    mono_kinds = ('HAT', 'TURTLE', 'CHEVRON', 'COMET', 'CUSTOM', 'SPECTRE')
    # the hat family patches are round and bbox-centred, so a patch-relative
    # window fits any family member whatever its overall scale; the spectre
    # grows asymmetrically from its Delta seed, so it keeps the fixed window.
    relative_half = ('HAT', 'TURTLE', 'CHEVRON', 'COMET', 'CUSTOM')
    for kind, label, _ in KIND_ITEMS:
        g_lo, g_mid, g_hi = gens_map[kind]
        n3 = len(aperiodic_patch(kind, g_lo)[0])
        polys, types = aperiodic_patch(kind, g_mid)
        n4 = len(polys)
        n5 = len(aperiodic_patch(kind, g_hi)[0])
        chalf = _mono_half(polys) if kind in relative_half else 3.0
        n_cov, n_over = _coverage(polys, half=chalf)
        grows = n3 < n4 < n5
        nquad = sum(1 for p in polys if len(p) == 4)

        ok = (n_over == 0 and n_cov > 0 and grows and n4 > 0)
        extra = "quads=%d" % nquad
        if kind in ('PENROSE_P3', 'PENROSE_P2'):
            ratio = n5 / n4 if n4 else 0.0
            ratio_ok = 2.2 <= ratio <= 3.05         # ~ phi^2 = 2.618
            ok = ok and ratio_ok
            extra = "ratio=%.3f(~2.618)" % ratio
        elif kind in mono_kinds:
            # a gap-free monotile tiling: every interior sample is covered
            gapless = (n_cov == n_samples)
            ok = ok and gapless
            n_type1 = sum(1 for t in types if t == 1)
            extra = "type1=%d gapfree=%s" % (n_type1, gapless)
        print("%-16s tiles=%5d covered=%3d overlaps=%3d grow=%-5s %s  %s"
              % (kind, n4, n_cov, n_over, str(grows), extra,
                 "OK" if ok else "BAD"))
        all_ok = all_ok and ok
        if kind == 'PENROSE_P2':
            # every P2 seed must coverage-verify and pass the kite/dart
            # shape/ratio/interior checks; the main line above already
            # covered the default SUN patch, so re-cover STAR too
            for p2_seed in ('SUN', 'STAR', 'CARTWHEEL'):
                sp, _st = aperiodic_patch('PENROSE_P2', g_mid, p2_seed)
                s_cov, s_over = _coverage(sp)
                p2_ok, p2_msg = _p2_selftest(p2_seed)
                seed_ok = p2_ok and s_over == 0 and s_cov > 0
                print("%-16s %s overlaps=%d  %s"
                      % ("  P2 kite/dart", p2_msg, s_over,
                         "OK" if seed_ok else "BAD"))
                all_ok = all_ok and seed_ok
        if kind == 'HAT':
            h_ok, h_msg = _hat_selftest()
            print("%-16s %s" % ("  hat reflect", h_msg))
            all_ok = all_ok and h_ok
            r_ok, r_msg = _hat_regression()
            print("%-16s %s" % ("  hat regress", r_msg))
            all_ok = all_ok and r_ok
        if kind == 'SPECTRE':
            # the straight Tile(1,1) is the default-tested patch above;
            # additionally verify the curved Spectre still tiles gap-free
            # and that curving genuinely adds vertices (the S-curves)
            cp, _ct = aperiodic_patch('SPECTRE', g_mid, spectre_curved=True)
            sp, _st = aperiodic_patch('SPECTRE', g_mid, spectre_curved=False)
            c_cov, c_over = _coverage(cp, half=3.0)
            curved_more = (sum(len(p) for p in cp) >
                           sum(len(p) for p in sp))
            cur_ok = (c_over == 0 and c_cov == n_samples and curved_more)
            print("%-16s curved_cov=%d overlaps=%d curves_added=%s  %s"
                  % ("  spectre curve", c_cov, c_over, curved_more,
                     "OK" if cur_ok else "BAD"))
            all_ok = all_ok and cur_ok
    print("RESULT:", "OK" if all_ok else "BAD")
