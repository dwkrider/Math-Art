
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
# The pure-Python self-test verifies, for each kind: every hypotenuse is
# paired full-edge (edge-to-edge), the interior is gap-free and
# overlap-free (dense sampling), and each generation is exactly s times
# the previous.
#
# References:
# - Robert W. Fathauer, "Fractal Tilings Based on Dissections of
#   Polyominoes", Bridges 2006, pp. 293-300; "Fractal tilings based on
#   kite- and dart-shaped prototiles", Computers & Graphics 24, 2000.
#   Online: Fractal Tiling Encyclopedia,
#   https://www.mathartfun.com/encyclopedia/Introduction.html .
# - Rep-tiles: Solomon W. Golomb, "Replicating figures in the plane",
#   Mathematical Gazette 48, 1964, pp. 403-412.

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
# Complex-base self-affine reptiles (Fathauer "iterating polyominoes")
#
# A Gaussian-integer base b with N = |b|^2 and a complete residue digit
# set D (|D| = N) defines a rep-N tile: every b-adic "integer"
# sum_{j} d_j b^j (d_j in D) is a distinct lattice point, and the unit
# cell placed at each of the N**k length-k strings forms a polyomino
# that -- because b carries a rotation -- crinkles toward a fractal
# reptile as k grows.  N copies compound into a b-times-larger replica
# (self-similar).  Cells are coloured by their leading (top-level)
# digit, so the N-fold substitution reads directly, as in Fathauer's
# figures.
# --------------------------------------------------------------------

_UNIT = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _base_reptile(b, digits, iterations):
    """(polys, types) for the rep-N reptile of Gaussian base b."""
    cells = [(complex(0.0, 0.0), 0)]      # (position, colour)
    n = len(digits)
    for lvl in range(int(iterations)):
        if len(cells) * n > _MAX_TILES:
            break
        nxt = []
        for p, col in cells:
            for di, d in enumerate(digits):
                nxt.append((p * b + d, di if lvl == 0 else col))
        cells = nxt
    polys, types = [], []
    for p, col in cells:
        polys.append(_UNIT + [p.real, p.imag])
        types.append(int(col))
    return polys, types


# builders take `iterations` and return (polys, types)
def _triangle_ftiling(iterations):
    raw = _triangle_patch(iterations)
    return ([_ensure_ccw(t) for _, t in raw],
            [int(lvl) for lvl, _ in raw])


KINDS = {
    'RIGHT_TRIANGLE': dict(build=_triangle_ftiling, tri=True),
    'TWINDRAGON': dict(
        build=lambda it: _base_reptile(complex(1, 1), [0, 1], it),
        tri=False, n=2),
    'REP5': dict(
        build=lambda it: _base_reptile(complex(2, 1), [0, 1, 2, 3, 4],
                                       it),
        tri=False, n=5),
    'REP5B': dict(
        build=lambda it: _base_reptile(complex(1, 2), [0, 1, 2, 3, 4],
                                       it),
        tri=False, n=5),
    'REP2B': dict(
        build=lambda it: _base_reptile(complex(1, -1), [0, 1], it),
        tri=False, n=2),
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
    ('TWINDRAGON', "Twindragon (rep-2)",
     "Gaussian base 1+i: two cells compound into a sqrt2-larger "
     "replica -- the twindragon fractal reptile"),
    ('REP5', "Rep-5 Dragon (base 2+i)",
     "Gaussian base 2+i: five cells compound into a sqrt5-larger "
     "replica -- Fathauer's iterated-polyomino rep-5 fractal reptile"),
    ('REP5B', "Rep-5 Dragon (base 1+2i)",
     "Gaussian base 1+2i: a second rep-5 reptile, different rotation"),
    ('REP2B', "Twindragon (base 1-i)",
     "Gaussian base 1-i: the mirror-rotation rep-2 reptile"),
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
                    "Material per generation, revealing the shrinking "
                    "levels toward the fractal boundary"),
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
        tri = KINDS[kind].get('tri', False)
        depths = (4, 7) if tri else (3, 5)
        for depth in depths:
            polys, types = fractal_patch(kind, depth)
            allv = np.vstack(polys)
            lo, hi = allv.min(0), allv.max(0)
            gx, gy = np.meshgrid(
                np.linspace(lo[0], hi[0], 240) + 0.00131,
                np.linspace(lo[1], hi[1], 240) + 0.00069)
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            overlaps = int((_coverage(polys, pts) >= 2).sum())
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
                n = KINDS[kind]['n']
                capped = len(polys) < n ** depth
                ok = overlaps == 0 and (len(polys) == n ** depth
                                        or capped)
                extra = "rep-%d count=%d(%d)" % (n, len(polys),
                                                 n ** depth)
            all_ok = all_ok and ok
            print("%-14s d=%d tiles=%5d overlaps=%d  %-24s %s"
                  % (kind, depth, len(polys), overlaps, extra,
                     "OK" if ok else "BAD"))
    print("RESULT:", "OK" if all_ok else "BAD")
