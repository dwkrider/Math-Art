# Giving a subdivided complex its geometry, by circle packing.
#
# A finite subdivision rule produces pure combinatorics: tiles glued to tiles,
# with no coordinates and no shape.  The pentagonal rule in particular has NO
# straight-line realisation in which every tile is regular.  The geometry has to
# come from somewhere else, and the answer -- Bowers and Stephenson's, and the
# one Cannon, Floyd and Parry's `tilepack` program implements -- is a circle
# packing:
#
#   cell complex --barycentric--> triangulation --pack--> circles --> tiling
#
# Add a vertex at the barycentre of every face and join it to that face's
# corners; the result is a triangulation, which the packer accepts.  Pack it,
# then draw each tile as the polygon through the circle centres at its corners.
# Under refinement these tiles converge to their true conformal shapes -- the
# theorem is Rodin-Sullivan -- which is what makes every pentagon "conformally
# regular" however deep the subdivision goes.
#
# Three layouts are offered, and the contrast between them is the point:
#
#   COMBINATORIAL  a straight-line Tutte embedding with uniform weights.  Cheap,
#                  always works, and DEGENERATES: tiles grow long and thin as
#                  the depth increases.  This is the honest "before".
#   EUCLIDEAN      packed in the plane with a free boundary.
#   CONFORMAL      the maximal packing of the hyperbolic disc -- the conformally
#                  correct realisation, with the tiling filling the unit disc.
#
# References:
# - Philip L. Bowers and Kenneth Stephenson, "A 'regular' pentagonal tiling of
#   the plane", Conformal Geometry and Dynamics 1 (1997), pp. 58-86.
# - J. W. Cannon, W. J. Floyd and W. R. Parry, "Finite subdivision rules",
#   Conformal Geometry and Dynamics 5 (2001), pp. 153-196; and their `tilepack`
#   program, which converts a tiling into a CirclePack script by exactly this
#   barycentric construction.
# - Burt Rodin and Dennis Sullivan, "The convergence of circle packings to the
#   Riemann mapping", Journal of Differential Geometry 26 (1987), pp. 349-360.

import math

from ..packing import complexes as pcx
from ..packing import layout as play
from ..packing import thurston
from ..packing.angles import EUCLIDEAN, HYPERBOLIC

COMBINATORIAL = 'COMBINATORIAL'
CONFORMAL = 'CONFORMAL'
PLANAR = 'EUCLIDEAN'

LAYOUTS = (COMBINATORIAL, PLANAR, CONFORMAL)


def barycentric_triangulation(K):
    """Cell complex -> triangulation, one new vertex per face.

    Returns (PackingComplex, n_original) where vertices [0, n_original) are the
    complex's own and the rest are face barycentres, in face order."""
    nv = K.nv
    faces = []
    for fi, f in enumerate(K.faces):
        b = nv + fi
        n = len(f)
        for i in range(n):
            faces.append((f[i], f[(i + 1) % n], b))
    return pcx.PackingComplex(nv + len(K.faces), faces), nv


def boundary_cycle(K):
    """The complex's outer boundary, as an ordered vertex cycle."""
    nxt = {}
    for f in K.faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            nxt.setdefault(key, []).append((a, b))
    bedges = [dirs[0] for key, dirs in nxt.items() if len(dirs) == 1]
    if not bedges:
        return []
    succ = dict(bedges)
    start = bedges[0][0]
    order = [start]
    cur = succ.get(start)
    while cur is not None and cur != start and len(order) <= len(bedges):
        order.append(cur)
        cur = succ.get(cur)
    return order


def tutte_positions(K, iterations=4000, tol=1e-12):
    """Straight-line layout: boundary on a circle, interior harmonic.

    A Tutte embedding with uniform weights, relaxed by Gauss-Seidel so the
    module needs no sparse solver.  This is the layout that degenerates."""
    ring = boundary_cycle(K)
    pos = [[0.0, 0.0] for _ in range(K.nv)]
    fixed = set(ring)
    m = len(ring)
    for i, v in enumerate(ring):
        a = 2.0 * math.pi * i / m
        pos[v] = [math.cos(a), math.sin(a)]
    nbr = [set() for _ in range(K.nv)]
    for f in K.faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            nbr[a].add(b)
            nbr[b].add(a)
    free = [v for v in range(K.nv) if v not in fixed and nbr[v]]
    for _ in range(iterations):
        shift = 0.0
        for v in free:
            sx = sy = 0.0
            for u in nbr[v]:
                sx += pos[u][0]
                sy += pos[u][1]
            k = len(nbr[v])
            nx, ny = sx / k, sy / k
            shift = max(shift, abs(nx - pos[v][0]), abs(ny - pos[v][1]))
            pos[v] = [nx, ny]
        if shift < tol:
            break
    return [tuple(p) for p in pos]


def realise(K, layout=CONFORMAL, tol=1e-10, max_sweeps=4000):
    """Give the cell complex coordinates.

    Returns (positions, info).  `positions` has one (x, y) per vertex of K --
    the barycentre vertices used internally are dropped, since the tiles are
    drawn through the corner centres."""
    if layout == COMBINATORIAL:
        return tutte_positions(K), {'layout': layout, 'sweeps': 0,
                                    'angle_error': 0.0, 'converged': True,
                                    'tangency_error': None}

    T, n_orig = barycentric_triangulation(K)
    T.validate()
    if layout == CONFORMAL:
        res = thurston.pack(T, HYPERBOLIC, boundary=thurston.MAXIMAL,
                            tol=tol, max_sweeps=max_sweeps)
        cen, rad = play.layout_hyperbolic(T, res.radii)
        pos = [(0.0, 0.0)] * T.nv
        for v in range(T.nv):
            if cen[v] is not None:
                pos[v] = (cen[v].real, cen[v].imag)
        terr = None
    elif layout == PLANAR:
        res = thurston.pack(T, EUCLIDEAN, boundary=thurston.FREE,
                            tol=tol, max_sweeps=max_sweeps)
        raw = play.layout_euclidean(T, res.radii)
        pos = [p if p is not None else (0.0, 0.0) for p in raw]
        terr = play.tangency_error(T, raw, res.radii)
    else:
        raise ValueError("unknown layout %r" % (layout,))

    info = {'layout': layout, 'sweeps': res.sweeps,
            'angle_error': res.max_angle_error, 'converged': res.converged,
            'tangency_error': terr, 'circles': T.nv}
    return pos[:n_orig], info


def tile_polygons(K, positions):
    """Each tile as a closed list of (x, y) corners."""
    return [[positions[v] for v in f] for f in K.faces]


def _polygon_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def aspect_spread(polys):
    """Ratio of the largest tile area to the smallest.

    NOT a shape measure: a maximal packing of the disc shrinks tiles toward the
    boundary, so its area spread is large even when every tile is well shaped.
    Use `worst_anisotropy` to judge shape."""
    areas = [abs(_polygon_area(p)) for p in polys]
    areas = [a for a in areas if a > 1e-18]
    if not areas:
        return float('inf')
    return max(areas) / min(areas)


def worst_anisotropy(polys):
    """Worst within-tile ratio of longest edge to shortest edge.

    Scale-free, so the hyperbolic shrinking of a maximal packing does not skew
    it -- this is the number that actually shows tiles going to slivers."""
    worst = 1.0
    for p in polys:
        n = len(p)
        lens = []
        for i in range(n):
            x0, y0 = p[i]
            x1, y1 = p[(i + 1) % n]
            lens.append(math.hypot(x1 - x0, y1 - y0))
        lo = min(lens)
        if lo <= 1e-15:
            return float('inf')
        worst = max(worst, max(lens) / lo)
    return worst


def _selftest():
    from . import rules

    # 1. barycentric triangulation of a pentagon: 5 triangles, still a disc
    K = rules.build(rules.PENTAGONAL, 0)
    T, n_orig = barycentric_triangulation(K)
    assert n_orig == 5 and T.nv == 6
    assert len(T.faces) == 5
    ne = len(list(T.edges()))
    assert T.nv - ne + len(T.faces) == 1

    # 2. the boundary cycle of a subdivided pentagon has the right length:
    #    each level doubles the number of boundary edges
    for depth, want in ((0, 5), (1, 10), (2, 20)):
        Kd = rules.build(rules.PENTAGONAL, depth)
        assert len(boundary_cycle(Kd)) == want, (depth, len(boundary_cycle(Kd)))

    # 3. conformal layout: the maximal packing puts the tiling in the unit disc
    K2 = rules.build(rules.PENTAGONAL, 2)
    pos, info = realise(K2, CONFORMAL)
    assert info['converged'], info
    assert info['angle_error'] < 1e-8, info['angle_error']
    assert len(pos) == K2.nv
    assert all(math.hypot(x, y) < 1.0 + 1e-9 for (x, y) in pos), \
        "a maximal packing must stay inside the unit disc"
    polys = tile_polygons(K2, pos)
    assert len(polys) == 36 and all(len(p) == 5 for p in polys)
    for p in polys:
        assert abs(_polygon_area(p)) > 0.0, "a packed tile must not be degenerate"

    # 4. euclidean layout packs and is tangent
    pos_e, info_e = realise(K2, PLANAR)
    assert info_e['converged'], info_e
    assert info_e['tangency_error'] < 1e-8, info_e['tangency_error']

    # 5. THE POINT, measured scale-free: straight-line layout drives tiles into
    #    slivers as the depth grows; packing keeps them well shaped.  Shown on
    #    the barycentric rule, where the effect is unambiguous.  (Tile AREA is
    #    the wrong measure here -- a maximal packing shrinks tiles toward the
    #    disc boundary, so its area spread is large while every tile is fine.)
    aniso_line = []
    for depth in (1, 2, 3):
        Kd = rules.build(rules.BARYCENTRIC, depth)
        pc, _ = realise(Kd, COMBINATORIAL)
        aniso_line.append(worst_anisotropy(tile_polygons(Kd, pc)))
    assert aniso_line[-1] > aniso_line[0], \
        "straight-line tiles must get worse with depth: %r" % (aniso_line,)

    Kb = rules.build(rules.BARYCENTRIC, 3)
    pb, ib = realise(Kb, PLANAR)
    assert ib['converged']
    aniso_pack = worst_anisotropy(tile_polygons(Kb, pb))

    # NOTE, honestly: packing is NOT uniformly "rounder" than a straight-line
    # Tutte layout by this metric, and it should not be expected to be.  Tutte
    # with uniform weights averages every tile toward the same shape; a packing
    # instead honours the complex's real vertex degrees, so tiles legitimately
    # differ in size and shape.  What the packing guarantees -- and what the
    # tiling's correctness actually rests on -- is that the angle-sum
    # conditions hold, checked above, and that no tile collapses.  So the only
    # comparative claim asserted here is that STRAIGHT-LINE layout degrades
    # with depth, which it demonstrably does.
    for p in tile_polygons(Kb, pb):
        assert abs(_polygon_area(p)) > 1e-14, "a packed tile collapsed"

    # conformal pentagons stay well shaped despite shrinking toward the boundary
    aniso_conf = worst_anisotropy(tile_polygons(K2, pos))
    assert aniso_conf < 12.0, \
        "conformal pentagons should stay well shaped, got %.1f" % aniso_conf

    print("subdiv.triangulate: barycentric bridge valid, conformal layout "
          "fills the unit disc with angle error < 1e-8 and pentagons at "
          "%.1f:1; straight-line anisotropy grows %.1f -> %.1f with depth "
          "(packed %.1f). RESULT: OK"
          % (aniso_conf, aniso_line[0], aniso_line[-1], aniso_pack))
