
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
# Besides the f-tiling, two further constructions are implemented:
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
# The pure-Python self-test verifies, for each kind: the f-tiling is
# edge-to-edge, gap-free and shrinks by exactly 1/sqrt(2) per level;
# the complex-base reptiles place N^k unit cells at distinct Gaussian-
# integer positions; the pentabolo has 5^k triangles with zero sampled
# overlaps.
#
# References:
# - Andrew Vince, "Rep-tiling Euclidean space", Aequationes
#   Mathematicae 50, 1995, pp. 191-213.  The radix-system catalog:
#   Fig 3 (rep-2 twindragon, base -1+i), Fig 5 (rep-5 dragon, base
#   -2+i, symmetric digits), Fig 6 (rep-4, base 2, digits
#   {0,1,i,-1-i}).
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
#   matrix.
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
    "description": "Fathauer fractal rep-tiles: right-triangle "
                   "f-tiling and complex-base reptiles (twindragon, "
                   "rep-5 ...)",
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


def _base_reptile(b, digits, iterations):
    """(polys, types) for the rep-N reptile of Gaussian base b with
    digit set `digits` (ints or complex; positions stay Gaussian
    integers).  The colour index encodes the top two substitution
    digits (d0*N + d1) so both the outer N copies and the next level of
    nesting are visible."""
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
        polys.append(_UNIT + [p.real, p.imag])
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


def _pentabolo(iterations):
    """(polys, types) for Fathauer's rep-5 pentabolo inflation: 5^k
    unit right triangles, coloured by the outermost map index."""
    tris = [(np.array([0.0, 1.0, 0.5 + 0.5j], complex), 0)]
    for lvl in range(int(iterations)):
        if len(tris) * 5 > _MAX_TILES:
            break
        f = _PENTA_LAMBDA ** lvl              # conjugating expansion
        nxt = []
        for verts, _col in tris:
            for mi, m in enumerate(_PENTA_MAPS):
                nxt.append((f * m(verts / f), mi))
        tris = nxt
    polys = [np.column_stack([v.real, v.imag]) for v, _ in tris]
    types = [int(c) for _, c in tris]
    return polys, types


# builders take `iterations` and return (polys, types)
def _triangle_ftiling(iterations):
    raw = _triangle_patch(iterations)
    return ([_ensure_ccw(t) for _, t in raw],
            [int(lvl) for lvl, _ in raw])


KINDS = {
    # Fathauer polygon-substitution kinds (triangle cells)
    'RIGHT_TRIANGLE': dict(build=_triangle_ftiling, tri=True),
    'PENTABOLO': dict(build=_pentabolo, tri_cells=True, n=5),
    # Complex-base radix-system kinds (unit-square cells)
    'TWINDRAGON': dict(                       # Vince Fig 3, AR 1/phi
        build=lambda it: _base_reptile(-1 + 1j, [0, 1], it), n=2),
    'REP4': dict(                             # Vince Fig 6
        build=lambda it: _base_reptile(2 + 0j, [0, 1, 1j, -1 - 1j],
                                       it), n=4),
    'REP5': dict(                             # Vince Fig 5, AR 1/phi
        build=lambda it: _base_reptile(-2 + 1j, [0, 1, -1, 1j, -1j],
                                       it), n=5),
    'REP5_THIN': dict(                        # Mekhontsev (5,4)
        build=lambda it: _base_reptile(2 + 1j, [0, 1, 2, 3, 4], it),
        n=5),
    'REP5B': dict(                            # Mekhontsev (5,2)
        build=lambda it: _base_reptile(1 + 2j, [0, 1, 2, 3, 4], it),
        n=5),
}


def fractal_patch(kind, iterations):
    """Return (polys, types): CCW tile arrays and their level/colour
    indices."""
    if kind not in KINDS:
        raise ValueError("unknown fractal rep-tile %r" % kind)
    polys, types = KINDS[kind]['build'](iterations)
    return [_ensure_ccw(np.asarray(p, float)) for p in polys], types


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

KIND_ITEMS = [
    ('RIGHT_TRIANGLE', "Right Triangle (f-tiling)",
     "Isosceles right-triangle f-tiling: square seed, a 1/sqrt2 child "
     "glued to every exposed leg; tiles shrink to a fractal boundary"),
    ('PENTABOLO', "Pentabolo (rep-5 gasket)",
     "Fathauer 5-isometry inflation of a half-square triangle: 5^k "
     "unit triangles, sqrt5 inflation per level, coloured by "
     "first-level copy"),
    ('TWINDRAGON', "Twindragon (rep-2, base -1+i)",
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
            polys, types = fractal_patch(self.kind, self.iterations)
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
            for p in ('kind', 'iterations', 'color_by', 'margin',
                      'height'):
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
        if tri:
            depths = (4, 7)
        elif tri_cells:
            depths = (2, 4)
        else:
            depths = (4, 7)
        for depth in depths:
            polys, types = fractal_patch(kind, depth)
            if tri or tri_cells:
                # triangle cells: dense-sampling overlap test
                allv = np.vstack(polys)
                lo, hi = allv.min(0), allv.max(0)
                gx, gy = np.meshgrid(
                    np.linspace(lo[0], hi[0], 240) + 0.00131,
                    np.linspace(lo[1], hi[1], 240) + 0.00069)
                pts = np.column_stack([gx.ravel(), gy.ravel()])
                overlaps = int((_coverage(polys, pts) >= 2).sum())
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
    print("RESULT:", "OK" if all_ok else "BAD")
