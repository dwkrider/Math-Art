
# Fractal Tiling Generator for Blender
#
# Fathauer fractal tilings ("f-tilings") -- edge-to-edge tilings by a
# single kite prototile in progressively smaller similar copies.  The
# interior of the patch is a clean, gap-free tiling at every depth; only
# the LIMIT boundary is fractal, crinkling into self-similar lobes with
# singular wrap-around points.  After Robert Fathauer, "Fractal tilings
# based on kite- and dart-shaped prototiles" (Computers & Graphics 24,
# 2000) and "Fractal tilings based on v-shaped prototiles" (Computers &
# Graphics 26, 2002); the shipped kinds reproduce entries of his online
# Fractal Tiling Encyclopedia,
# https://www.mathartfun.com/encyclopedia/Introduction.html .
#
# PROTOTILE.  A right kite for even rotational order k: apex angle
# A = 360/k between the two LONG edges, two 90 degree side corners, and
# a tip of 180 - A between the two SHORT edges, with contraction ratio
#
#       s = short / long = tan(180 / k).
#
# CONSTRUCTION (Fathauer's method, Introduction figs. 1-4):
#
#   1. GENERATION 0 -- k kites are placed apex-to-apex around a central
#      point, all long edges paired with a neighbor's long edge.  Their
#      union is a regular k-gon (a hexagon for k = 6) whose corners are
#      the kite tips and whose side midpoints are the 90 degree corners.
#   2. RECURSION -- the exterior angle at a lone exposed tip is
#      360 - (180 - A) = 180 + A = (k/2 + 1) * A, so a FAN of
#      n = k/2 + 1 children scaled by s, apexes at the tip, fills it
#      EXACTLY: the two outermost fan children glue a long edge onto the
#      parent's two short edges (child long = s * long = parent short,
#      endpoint to endpoint), and consecutive fan children pair long
#      edges with each other.  Where two neighboring fans meet, the two
#      90 degree corners complete the straight angle, and the fans' end
#      children lie back-to-back along a shared short edge with their
#      tips COINCIDING (first over the seed polygon's side midpoints);
#      such a doubled tip has only 2A degrees free and takes a fan of
#      just 2 children.  Every vertex of the patch therefore closes to
#      exactly 360 degrees or lies on the outer boundary.  Repeat on the
#      new tips: each generation is s times smaller and only the newest
#      generation's short edges are exposed.
#
# All long edges are therefore paired (internal) at every depth: the
# interior is provably gap-free and overlap-free, and the lone-tip fan
# count k/2 + 1 is exactly Fathauer's g-number (second-generation tiles
# per first-generation tile, e.g. 24/6 = 4 for r6).  Tile counts per
# generation for r6 run 6, 24, 60, 132, 276, ... (a' = 2a + 12).
# Shipped kinds, in his (pN, rK, gG, s..., mM)
# classification -- polygon edges p, rotation order r, generation ratio
# g, scaling ratio s, mirror lines m:
#
#       KITE_R6   (p4, r6,  g4, s.577, m2)   A=60  s=tan30   = 0.5774
#       KITE_R8   (p4, r8,  g5, s.414, m2)   A=45  s=tan22.5 = 0.4142
#       KITE_R12  (p4, r12, g7, s.268, m2)   A=30  s=tan15   = 0.2679
#
# Odd orders are impossible for this family -- the fan count k/2 + 1
# must be an integer -- which is why the encyclopedia lists no r5 kite
# f-tiling.  The pure-Python self-test below verifies, for each kind:
# the g-number (gen-1 count = k * (k/2 + 1)), full-edge edge-to-edge
# pairing of every long edge (endpoints coincide to 1e-6), a gap-free
# interior (dense sampling inside the union outline of the previous
# generations), zero overlaps, and the exact per-generation shrink
# ratio s.

bl_info = {
    "name": "Fractal Tiling",
    "author": "Math Art project",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Fathauer f-tilings: kite rosettes whose similar "
                   "tiles shrink edge-to-edge toward a fractal boundary",
    "category": "Add Mesh",
}

import cmath
from collections import Counter
from math import pi, cos, tan

import numpy as np

try:
    from . import pattern_common as pc
    from . import tiling_generator as tg
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc
    import tiling_generator as tg


# --------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------

def _to_xy(pts):
    """List of complex points -> (N, 2) float array."""
    return np.array([[p.real, p.imag] for p in pts], float)


def _signed_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(poly):
    poly = np.asarray(poly, float)
    if _signed_area(poly) < 0.0:
        return poly[::-1].copy()
    return poly


# --------------------------------------------------------------------
# Kite f-tiling
#
# A tile is stored as (level, T, L, B, R): a right kite with apex T
# (angle A, between the two LONG edges T-L and T-R), two 90 degree side
# corners L and R, and tip B (angle 180 - A, between the two SHORT
# edges L-B and B-R).  In generation 0 the long edges are paired around
# the origin; a grafted tile has its apex T on the parent tip and its
# long edges paired with the parent short edges / its fan siblings.
# --------------------------------------------------------------------

# rotational orders offered, with contraction ratio s = tan(180/k);
# the tip fan places k/2 + 1 children (Fathauer's g-number)
_KITE_K = {'KITE_R6': 6, 'KITE_R8': 8, 'KITE_R12': 12}

# complete generations only: growth stops before a generation that
# would push the patch past this many tiles (keeps meshes manageable)
_MAX_TILES = 150000


def _kite_proto(k):
    """Prototile vertices (T, L, B, R) in local coordinates: apex T at
    the origin, axis along +x, long edge length 1."""
    ha = pi / k                                   # half the apex angle A/2
    return (0j,
            cmath.rect(1.0, ha),                  # L (upper side corner)
            complex(1.0 / cos(ha), 0.0),          # B (tip, on the axis)
            cmath.rect(1.0, -ha))                 # R (lower side corner)


def _kite_seed(k):
    """Generation 0: k prototiles apex-to-apex about the origin, long
    edges paired with neighbors; their union is a regular k-gon."""
    proto = _kite_proto(k)
    a = 2.0 * pi / k
    tiles = []
    for i in range(k):
        rot = cmath.rect(1.0, a * i)
        t, l, b, r = (rot * p for p in proto)
        tiles.append((0, t, l, b, r))
    return tiles


def _ckey(z, nd=6):
    return (round(z.real, nd), round(z.imag, nd))


def _edge_keys(tile):
    """Endpoint-pair keys of the tile's four edges."""
    _, t, l, b, r = tile
    kt, kl, kb, kr = _ckey(t), _ckey(l), _ckey(b), _ckey(r)
    return (frozenset((kt, kl)), frozenset((kl, kb)),
            frozenset((kb, kr)), frozenset((kr, kt)))


def _kite_patch(k, iterations):
    """Grow the f-tiling patch.  Each pass finds every exposed tip
    VERTEX of the newest generation and fills its free angle exactly
    with a fan of s-scaled children, apexes on the vertex:

      * a lone tip exposes 180 + A degrees -> k/2 + 1 children;
      * where the tips of two neighboring branches coincide (they meet
        back-to-back along a shared short edge, first over the seed
        polygon's side midpoints) only 2A degrees remain -> 2 children;
      * a vertex whose angles already sum to 360 degrees is closed.

    The fan's outermost children glue their long edges onto the exposed
    short edges bounding the free sector, and consecutive children pair
    long edges, so every vertex closes to exactly 360 degrees or stays
    on the outer boundary and no tile ever overlaps another."""
    A = 2.0 * pi / k                              # apex angle
    ha = pi / k
    tip_ang = pi - A                              # prototile tip angle
    tiles = _kite_seed(k)
    edge_use = Counter()
    for tl in tiles:
        for ek in _edge_keys(tl):
            edge_use[ek] += 1
    frontier = list(range(len(tiles)))
    for _ in range(int(iterations)):
        if len(tiles) + (k // 2 + 1) * len(frontier) > _MAX_TILES:
            break
        groups = {}
        for ti in frontier:                       # tips sharing a vertex
            groups.setdefault(_ckey(tiles[ti][3]), []).append(ti)
        new_idx = []
        for idxs in groups.values():
            # children needed to close the free angle at this vertex
            n = int(round((2.0 * pi - len(idxs) * tip_ang) / A))
            if n <= 0:
                continue                          # vertex already closed
            # the free sector is bounded by the two still-exposed short
            # edges at the vertex (a short edge shared by two coincident
            # tips is internal and excluded)
            rays = []
            for ti in idxs:
                lvl, t, l, b, r = tiles[ti]
                eks = _edge_keys(tiles[ti])
                for pt, ek in ((l, eks[1]), (r, eks[2])):
                    if edge_use[ek] == 1:
                        rays.append((pt - b) / abs(pt - b))
            if len(rays) != 2:
                continue                          # crowded wrap-around
            lvl, t, l, b, r = tiles[idxs[0]]
            ls = abs(b - r)                       # short edge = child long
            ua, ub = rays
            # sweep from ray ua to ray ub in n steps of the apex angle,
            # in whichever direction lands exactly on ub (the free way)
            rot = cmath.rect(1.0, A)
            if abs(ua * rot ** n - ub) > abs(ua * rot.conjugate() ** n - ub):
                rot = rot.conjugate()
            if abs(ua * rot ** n - ub) > 1e-6:
                continue                          # blocked by older tiles
            roth = cmath.rect(1.0, A / 2.0)
            if rot.imag < 0.0:
                roth = roth.conjugate()
            tipd = ls / cos(ha)                   # child tip distance
            u = ua
            for _j in range(n):
                u2 = u * rot
                child = (lvl + 1,
                         b,                       # apex on the tip vertex
                         b + ls * u2,             # L
                         b + tipd * u * roth,     # B (child tip, outward)
                         b + ls * u)              # R
                tiles.append(child)
                new_idx.append(len(tiles) - 1)
                for ek in _edge_keys(child):
                    edge_use[ek] += 1
                u = u2
        frontier = new_idx
    return tiles


def _poly_of(tile):
    _, t, l, b, r = tile
    return _to_xy([t, l, b, r])


# --------------------------------------------------------------------
# Public patch builder
# --------------------------------------------------------------------

def fractal_patch(kind, iterations):
    """Return (polys, types) for a finite fractal-tiling patch.

    polys -- list of (4, 2) CCW float arrays (one kite per tile)
    types -- list of int GENERATION (level) indices, parallel to polys,
             so 'By Level' color reveals the shrinking generations.
    """
    if kind not in _KITE_K:
        raise ValueError("unknown fractal kind %r" % kind)
    raw = _kite_patch(_KITE_K[kind], iterations)
    polys, types = [], []
    for tile in raw:
        polys.append(_ensure_ccw(_poly_of(tile)))
        types.append(int(tile[0]))
    return polys, types


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

KIND_ITEMS = [
    ('KITE_R6', "Kite (6-fold)",
     "Fathauer f-tiling (p4, r6, g4, s.577, m2): hexagon of 60-degree "
     "kites, four children per tip, ratio s=0.577"),
    ('KITE_R8', "Kite (8-fold)",
     "Fathauer f-tiling (p4, r8, g5, s.414, m2): octagon of 45-degree "
     "kites, five children per tip, ratio s=0.414"),
    ('KITE_R12', "Kite (12-fold)",
     "Fathauer f-tiling (p4, r12, g7, s.268, m2): 12-gon of 30-degree "
     "kites, seven children per tip, ratio s=0.268"),
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

    class MESH_OT_fractal_tiling_add(bpy.types.Operator, AddObjectHelper):
        """Add a Fathauer fractal tiling patch"""
        bl_idname = "mesh.fractal_tiling_add"
        bl_label = "Fractal Tiling"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(name="Tiling", items=KIND_ITEMS,
                           default='KITE_R6')
        iterations: IntProperty(
            name="Iterations", default=4, min=0, max=9,
            description="Growth depth; each generation surrounds every "
                        "exposed tip with its fan of s-scaled tiles "
                        "(deep growth is capped to keep the mesh "
                        "manageable)")
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
            description="Output each tile as its own mesh object "
                        "(parented to an empty) so tiles can be edited "
                        "individually")

        def execute(self, context):
            polys, types = fractal_patch(self.kind, self.iterations)
            cells = tg.cells_from_polys(
                lambda a, b: (polys, types), 1, 1, self.color_by,
                self.margin, self.height, False)
            label = dict((k, v) for k, v, _ in KIND_ITEMS)[self.kind]
            obj = pc.emit(context, "Fractal %s" % label, cells,
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
        self.layout.operator("mesh.fractal_tiling_add",
                             icon='MOD_TRIANGULATE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_fractal_tiling_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_fractal_tiling_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _edges(poly):
    n = len(poly)
    return [(poly[i], poly[(i + 1) % n]) for i in range(n)]


def _edge_len(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _key(pt, nd=6):
    return (round(float(pt[0]), nd), round(float(pt[1]), nd))


def _edge_use_map(polys):
    """Counter of rounded endpoint-pair keys over every edge of every
    tile (a key seen twice is an exact full-edge pairing)."""
    use = Counter()
    for p in polys:
        for a, b in _edges(p):
            use[frozenset((_key(a), _key(b)))] += 1
    return use


def _long_edges_paired(polys, use):
    """Every LONG edge of every tile must be paired -- shared, endpoint
    to endpoint within 1e-6, with exactly one other tile (a neighbor's
    long edge or the parent's short edge).  This is the edge-to-edge
    check: only short edges may lie on the open boundary."""
    for p in polys:
        e = _edges(p)
        lens = [_edge_len(a, b) for a, b in e]
        order = sorted(range(4), key=lambda i: -lens[i])
        for i in order[:2]:                       # the two longest edges
            if use[frozenset((_key(e[i][0]), _key(e[i][1])))] != 2:
                return False
    return True


def _boundary_loops(polys):
    """Chain the edges used exactly once into closed cycles.  Returns
    (outer loop, hole loops, ok): the outer loop is the cycle of
    largest area, any others are enclosed pockets -- at finite depth
    the tiling leaves a tiny uncovered pocket around each boundary
    SINGULARITY (visible as white dots in Fathauer's own renders),
    which shrinks to a point as generations increase.  ok is False if
    any boundary vertex does not touch exactly 2 boundary edges."""
    use = Counter()
    pts = {}
    for p in polys:
        for a, b in _edges(p):
            ka, kb = _key(a), _key(b)
            use[frozenset((ka, kb))] += 1
            pts[ka] = a
            pts[kb] = b
    adj = {}
    for ek, c in use.items():
        if c != 1:
            continue
        a, b = tuple(ek)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in adj.values()):
        return None, None, False
    loops = []
    visited = set()
    for start in adj:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev, cur = None, start
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
    """Even-odd point-in-polygon for an arbitrary closed loop,
    vectorized over the sample points."""
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


def _coverage(polys, pts, tol=1e-9):
    """How many (convex, CCW) tiles cover each sample point."""
    cnt = np.zeros(len(pts), int)
    for p in polys:
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        m = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
             (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
        if not m.any():
            continue
        q = pts[m]
        ok = np.ones(len(q), bool)
        for i in range(len(p)):
            ax, ay = p[i]
            ex, ey = p[(i + 1) % len(p)] - p[i]
            ok &= (ex * (q[:, 1] - ay) - ey * (q[:, 0] - ax)) > tol
        cnt[m] += ok
    return cnt


def _grid(polys, samples=200, off=(0.00137, 0.00071)):
    """Dense sample grid over the joint bbox, nudged off tile edges."""
    allv = np.vstack(polys)
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    xs = np.linspace(lo[0], hi[0], samples) + off[0]
    ys = np.linspace(lo[1], hi[1], samples) + off[1]
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _self_similar(polys, types, s, tol=1e-6):
    """Each generation's tile size (max edge) is s times the previous
    generation's.  Returns (ok, per-level size dict)."""
    per = {}
    for p, t in zip(polys, types):
        per[t] = max(per.get(t, 0.0),
                     max(_edge_len(a, b) for a, b in _edges(p)))
    lv = sorted(per)
    ok = all(abs(per[lv[i + 1]] - s * per[lv[i]]) <= tol * per[lv[i]]
             for i in range(len(lv) - 1))
    return ok, per


if __name__ == "__main__":
    DEPTH = {'KITE_R6': 5, 'KITE_R8': 4, 'KITE_R12': 3}
    all_ok = True
    for kind, label, _ in KIND_ITEMS:
        k = _KITE_K[kind]
        g = DEPTH[kind]
        s = tan(pi / k)
        n_fan = k // 2 + 1
        polys, types = fractal_patch(kind, g)
        # Fathauer's g-number: k seed tiles, k * (k/2 + 1) in gen 1,
        # and strictly growing generations after that
        per_lvl = Counter(types)
        counts_ok = (per_lvl[0] == k and per_lvl[1] == k * n_fan and
                     all(per_lvl[lv + 1] > per_lvl[lv]
                         for lv in range(1, g)))
        # full-edge edge-to-edge pairing of every long edge
        e2e_ok = _long_edges_paired(polys, _edge_use_map(polys))
        # gap-free interior: chain the union outline of generations
        # <= g-1; every sample inside it and outside the finest-scale
        # singular pockets must be covered exactly once (the newest
        # generation's fractal fringe may lie outside the outline)
        sub = [p for p, t in zip(polys, types) if t < g]
        outer, holes, loop_ok = _boundary_loops(sub)
        n_gap = -1
        n_holes = -1
        if loop_ok:
            # a pocket is only legitimate at the finest scale: bounded
            # by the sub-patch's shortest (generation g-1) short edges
            pocket_max = 6.0 * (s ** g) ** 2
            n_holes = len(holes)
            loop_ok = all(abs(_signed_area(h)) <= pocket_max
                          for h in holes)
        pts = _grid(polys)
        cov = _coverage(polys, pts)
        n_over = int(np.sum(cov >= 2))
        if loop_ok:
            inside = _inside_loop(outer, pts)
            for h in holes:
                inside &= ~_inside_loop(h, pts)
            n_gap = int(np.sum((cov == 0) & inside))
        sim_ok, per = _self_similar(polys, types, s)
        ok = (counts_ok and e2e_ok and loop_ok and n_gap == 0 and
              n_over == 0 and sim_ok)
        all_ok = all_ok and ok
        print("%-9s k=%-2d s=%.4f fan=g%d depth=%d tiles=%5d "
              "counts=%-5s edge2edge=%-5s outline=%-5s singular=%2d "
              "gaps=%2d overlaps=%2d shrink=%-5s  %s"
              % (kind, k, s, n_fan, g, len(polys), str(counts_ok),
                 str(e2e_ok), str(loop_ok), n_holes, n_gap, n_over,
                 str(sim_ok), "OK" if ok else "BAD"))
        sizes = " ".join("L%d=%.4f" % (lv, per[lv]) for lv in sorted(per))
        print("          sizes: %s" % sizes)
    print("RESULT:", "OK" if all_ok else "BAD")
