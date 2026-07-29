
# 2D Celtic Knot Generator for Blender
#
# Classic grid-and-barriers Celtic knotwork: an interlaced plait laid on
# a rectangular grid, its cords woven strictly over-and-under, carved
# into bordered panels and knots by internal BARRIERS.  This is the
# construction of the Insular and Pictish stone-carvers as formalized by
# the modern mathematical accounts.
#
# THE GRID.  Work on an integer lattice (i, j), 0 <= i <= 2W, 0 <= j <=
# 2H, over a W x H array of cells (each cell two units wide):
#   * PRIMARY points  i, j both even     -- the cell corners.
#   * SECONDARY points i, j both odd     -- the cell centres, about which
#                                           the cord makes its crossings.
#   * EDGE-MIDPOINTS  i + j odd          -- the midpoints of the cell
#                                           walls; the cord threads the
#                                           grid through these, and every
#                                           BARRIER occupies one of them.
# The cord runs as a set of 45-degree diagonal strands between edge-
# midpoints.  Where a wall carries no barrier the two strands crossing
# there pass straight through one another (a crossing); where a wall
# carries a barrier the strand cannot pass and REFLECTS (a 90-degree
# bounce), exactly like a light ray in a box of mirrors.  The outer
# border is a full ring of barriers (a bordered panel); internal
# barriers carve the design.  This "mirror-curve" reading of knotwork is
# due to Jablan; the loop count of a plain W x H bordered plait is
# gcd(W, H).
#
# THREADING.  The strand state (an edge-midpoint together with a diagonal
# direction) has a single deterministic successor -- cross, or reflect at
# a barrier -- and that map is a bijection, so iterating it PARTITIONS all
# directed strand-segments into disjoint CLOSED loops (the separate
# cords).  This is proved by construction and checked by the self-test.
#
# OVER / UNDER.  Celtic interlace is an alternating diagram: following any
# cord the sense runs over, under, over, under..., and each crossing has
# exactly one strand over and one under.  Because a reflection swaps a
# strand between the two diagonal families, the alternation reduces to a
# parity (XOR) constraint between consecutive crossings along every cord,
# solved globally with a parity union-find -- the same alternating-knot
# assignment the Islamic strapwork engine uses.  Two renderings follow:
# FLAT breaks the under-cord at each crossing (2D interlace); WOVEN lifts
# and dips each cord into a genuine 3D woven surface.
#
# The traced cords flow through the shared Pattern Engine
# (pattern_common) and reuse the strapwork machinery of the Islamic
# generator -- mitered / Catmull-Rom ribbons, the over/under interlace,
# relief and backing -- so a knot is built as continuous ribbons, not a
# pile of quads.
#
# ARBITRARY TILINGS (the medial / general-graph knot).  Beyond the
# square grid the same cord lives on ANY planar tiling by Christian
# Mercat's classic "knot on any graph" construction: put a CROSSING at
# every tiling EDGE midpoint and route the cord around the faces,
# weaving over and under.  Reusing tiling_generator's regular-polygon
# substrates (triangular 3^6, hexagonal 6^3, and the Archimedean snub-
# square 3.3.4.3.4, trihexagonal 3.6.3.6 and truncated-hexagonal
# 3.12.12 tilings), the cord's control points are the shared edge
# midpoints and each corner-segment threads one edge; the same
# cross/reflect successor is a bijection, so orbits are again closed
# cords, over/under is a checkerboard 2-coloring (with a graceful
# frustration seam where the faces are not bipartite), a wall on an
# edge removes its crossing and the cord reflects, and the tiling's
# outer boundary edges act as the border.  All of it flows through the
# unchanged ribbon / interlace / relief pipeline.
#
# TUBE & ROPE profiles.  A cord need not be a flat ribbon: its
# centerline can also be swept into a round TUBE or a twisted ROPE of
# several helical sub-strands (breaking the under-tube at crossings for
# flat knotwork, rising and dipping it for the woven 3D surface), so the
# knot reads as genuine cordage rather than tape.
#
# SYMMETRY-GROUP BARRIERS.  A raw procedural or seeded barrier set is
# rarely balanced.  Replicating that seed under a point group about the
# panel/lattice centre -- every rotation (and, for the dihedral groups,
# every mirror) that maps the lattice to itself -- and unioning the
# complete orbits yields a barrier set that is provably invariant under
# the group, so the carved knot comes out balanced: C_n gives n-fold
# rotational symmetry, D_n adds the mirrors.  The square lattice admits
# the 90-degree groups C2/C4/D2/D4; the triangular / hexagonal family
# admits the 60/120-degree groups C3/C6/D3/D6.
#
# LENTICULAR TURNS.  Where a smoothed cord reflects off a wall it makes a
# U-turn; the plain Catmull-Rom rounds it into a semicircular cap, but
# the historical Insular carvers drew a POINTED lenticular (pointed-oval
# / leaf) turn -- the two rails curving back to meet at a soft point
# offset from the wall.  The pointed option reshapes the local spline at
# every reflection vertex to that leaf, tunable from the round cap (the
# limit) to a sharp tip.
#
# HISTORICAL PANELS.  Beyond the plain / cross-break / framed presets, a
# handful of named classic panels are supplied as fixed barrier layouts
# on the square grid -- the three-cornered TRIQUETRA (trinity knot), the
# two-cord JOSEPHINE knot, a repeating KELLS border motif, and a filled
# CARPET-page repeat -- following the grid-drawing tradition of the Bains
# and Meehan.
#
# References:
#   George Bain, "Celtic Art: The Methods of Construction" (1951) -- the
#     grid-and-plait method taught to a century of Celtic artists.
#   Iain Bain, "Celtic Knotwork" (1986) -- the systematic grid, breaks
#     and border construction reproduced here.
#   Andrew Glassner, "Celtic Knotwork" (Andrew Glassner's Notebook, IEEE
#     Computer Graphics & Applications 19(5)-(6), 1999) -- the algorithmic
#     grid / barrier / crossing formulation.
#   Christian Mercat, "Les entrelacs des enluminures celtes" (2005) and
#     "Celtic Knotwork" -- the mathematical (mirror-curve) treatment.
#   Peter R. Cromwell, "Celtic Knotwork: Mathematical Art" (Mathematical
#     Intelligencer 15(1), 1993) -- the topology of the plait and its
#     components.
#   Aidan Meehan, "Celtic Design: Knotwork" (1991) -- the grid-cell
#     drawing tradition.
#   Slavik V. Jablan, "Mirror curves" and "Symmetry, Ornament and
#     Modularity" (2002) -- knotwork as billiard / mirror curves, and the
#     component count of the plain plait.
#   Christian Mercat, "Knots and links from a graph" / "Les entrelacs des
#     enluminures celtes" -- the construction of an interlaced knot on the
#     medial graph of ANY planar graph, generalizing the grid plait to
#     arbitrary tilings (used here for the non-square substrates).

bl_info = {
    "name": "Celtic Knot 2D",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Grid-and-barriers 2D Celtic knotwork -- interlaced "
                   "plaits, bordered knots, over/under weave, relief",
    "category": "Add Mesh",
}

from math import gcd, hypot, cos, sin, pi
import random as _random

import numpy as np

try:
    from . import pattern_common as pc
    from . import islamic_pattern_generator as isl
    from . import tiling_generator as tg
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc
    import islamic_pattern_generator as isl
    import tiling_generator as tg


# The four diagonal travel directions on the (i, j) lattice.  NE and SW
# lie on slope +1 (family 0); NW and SE on slope -1 (family 1).  A wall
# reflection flips exactly one component and so swaps a strand's family.
_DIRS = ((1, 1), (-1, -1), (-1, 1), (1, -1))


def _family(d):
    """0 for the slope +1 diagonal family (NE/SW), 1 for slope -1
    (NW/SE)."""
    return 0 if d in ((1, 1), (-1, -1)) else 1


# --------------------------------------------------------------------
# Barriers
# --------------------------------------------------------------------
#
# A barrier is an edge-midpoint (i + j odd) that blocks the cord.  The
# border ring is handled implicitly (see `_is_barrier`); the sets built
# here hold only the INTERNAL barriers that carve the design.  An
# internal edge-midpoint is one strictly inside the panel: a vertical
# wall (i even) at 0 < i < 2W, or a horizontal wall (j odd... ) -- more
# precisely any (i, j), i + j odd, with 0 < i < 2W and 0 < j < 2H.

def _internal_slots(W, H):
    """Every internal edge-midpoint slot of a W x H grid (candidate
    barriers), as a sorted list of (i, j)."""
    out = []
    for i in range(0, 2 * W + 1):
        for j in range(0, 2 * H + 1):
            if (i + j) % 2 == 0:
                continue
            if 0 < i < 2 * W and 0 < j < 2 * H:
                out.append((i, j))
    return out


def _wall_v(B, col, W, H, r0, r1):
    """Add a vertical internal wall at cell-column boundary `col`,
    spanning cell rows r0..r1-1 (a run of edge-midpoints (2col, 2row+1))."""
    i = 2 * col
    if not (0 < i < 2 * W):
        return
    for row in range(max(0, r0), min(H, r1)):
        B.add((i, 2 * row + 1))


def _wall_h(B, row, W, H, c0, c1):
    """Add a horizontal internal wall at cell-row boundary `row`, spanning
    cell columns c0..c1-1 (edge-midpoints (2col+1, 2row))."""
    j = 2 * row
    if not (0 < j < 2 * H):
        return
    for col in range(max(0, c0), min(W, c1)):
        B.add((2 * col + 1, j))


# Historical panels: recommended grid sizes (grid_w, grid_h) at which the
# named motif reads best; the barrier layouts below scale/clamp to the
# user's actual grid.
_PRESET_HINTS = {
    'TRIQUETRA': (4, 4), 'JOSEPHINE': (6, 3),
    'KELLS_BORDER': (8, 3), 'CARPET': (6, 6),
}


def _preset_barriers(name, W, H):
    """Internal barriers for a named preset panel."""
    B = set()
    if name == 'PLAIN':
        return B
    if name == 'TRIQUETRA':
        return _triquetra_barriers(W, H)
    if name == 'JOSEPHINE':
        return _josephine_barriers(W, H)
    if name == 'KELLS_BORDER':
        return _kells_border_barriers(W, H)
    if name == 'CARPET':
        return _carpet_barriers(W, H)
    if name == 'CROSSBREAK':
        # a single vertical and horizontal internal break through the
        # centre -- the classic four-panel knot
        ic = 2 * (W // 2)
        jc = 2 * (H // 2)
        if 0 < ic < 2 * W:
            for j in range(1, 2 * H, 2):
                B.add((ic, j))
        if 0 < jc < 2 * H:
            for i in range(1, 2 * W, 2):
                B.add((i, jc))
        return B
    if name == 'PANEL':
        # an inset rectangular ring of barriers: a knot framing a central
        # motif
        i0, i1 = 2, 2 * W - 2
        j0, j1 = 2, 2 * H - 2
        if i1 > i0 and j1 > j0:
            for j in range(1, 2 * H, 2):
                B.add((i0, j))
                B.add((i1, j))
            for i in range(1, 2 * W, 2):
                B.add((i, j0))
                B.add((i, j1))
        return B
    return B


# --------------------------------------------------------------------
# Historical named panels (square-substrate barrier layouts)
# --------------------------------------------------------------------
#
# Each panel is a fixed barrier layout that yields a recognizable classic
# knot.  The layouts are written for their recommended grid (see
# _PRESET_HINTS) but scale/clamp to the user's grid_w/grid_h so they stay
# valid at any size.

def _triquetra_barriers(W, H):
    """The three-cornered TRINITY knot (triquetra): three lobes about the
    panel centre.  A true 3-fold figure does not sit on the square
    lattice, so the three leaves are carved with a downward wedge of
    walls -- an apex wall up top and two splayed walls below -- giving the
    familiar three interlaced lobes.  Best on a 4x4 panel."""
    B = set()
    W = max(3, W)
    H = max(3, H)
    cx = W // 2
    cy = H // 2
    # apex leaf: a short horizontal wall above centre with a stem down to it
    _wall_h(B, min(H - 1, cy + 1), W, H, cx - 1, cx + 1)
    _wall_v(B, cx, W, H, cy, min(H, cy + 1))
    # two lower leaves: splayed vertical walls with feet turning inward
    _wall_v(B, max(1, cx - 1), W, H, max(0, cy - 1), cy + 1)
    _wall_v(B, min(W - 1, cx + 1), W, H, max(0, cy - 1), cy + 1)
    _wall_h(B, max(1, cy - 1), W, H, cx - 1, cx + 1)
    return B


def _josephine_barriers(W, H):
    """The JOSEPHINE / carrick two-cord knot: two loops sitting side by
    side, linked down the middle.  A central vertical break splits the
    panel into two knots, and a pair of short horizontal breaks pinch the
    waist so the two cords read as an interlocked pair.  Best on a wide
    panel (about 6x3)."""
    B = set()
    W = max(4, W)
    H = max(2, H)
    mid = W // 2
    # central vertical divider, but broken at the waist so the two cords
    # thread through one another rather than sealing into two panels
    waist = H // 2
    _wall_v(B, mid, W, H, 0, waist)
    _wall_v(B, mid, W, H, waist + 1, H)
    # pinch the waist of each loop
    _wall_h(B, waist, W, H, 0, mid - 1)
    _wall_h(B, waist, W, H, mid + 1, W)
    return B


def _kells_border_barriers(W, H):
    """A repeating KELLS-style border motif: a running plait chained into
    a row of linked rings around a plain centre.  Horizontal walls fence
    off a one-cell-deep border band top and bottom, and vertical walls on
    an alternating spacing chop that band into a chain of rings -- the
    edge ornament of an illuminated page.  Best on a wide panel."""
    B = set()
    W = max(3, W)
    H = max(3, H)
    # fence the top and bottom border bands off the central field
    _wall_h(B, 1, W, H, 0, W)
    _wall_h(B, H - 1, W, H, 0, W)
    # chop each band into a chain of rings on an alternating spacing
    for col in range(2, W - 1, 2):
        _wall_v(B, col, W, H, 0, 1)
        _wall_v(B, col, W, H, H - 1, H)
    # close the central field into its own panel so the border reads apart
    _wall_v(B, 1, W, H, 1, H - 1)
    _wall_v(B, W - 1, W, H, 1, H - 1)
    return B


def _carpet_barriers(W, H):
    """A filled CARPET-page repeat: the panel tiled into a regular field
    of identical small knots by a lattice of internal walls every two
    cells (the dense all-over ornament of a carpet page).  Best on a
    larger panel (6x6 or more)."""
    B = set()
    W = max(4, W)
    H = max(4, H)
    for col in range(2, W, 2):
        _wall_v(B, col, W, H, 0, H)
    for row in range(2, H, 2):
        _wall_h(B, row, W, H, 0, W)
    return B


def _sym_lines(N, sp):
    """Cell-boundary indices in 1..N-1 on which to raise a procedural wall
    so the walls are MIRROR-SYMMETRIC about the panel centre and evenly
    spaced -- a regular array of equal (or as-equal-as-possible) knot
    cells, with the leftover margin split evenly at both edges instead of
    piling up on one side.

    A panel of N cells is cut into `n` sub-panels by n-1 interior walls at
    k_m = round(N*m/n).  Because round(N-x) == N-round(x) for integer N,
    the set is exactly invariant under k -> N-k, so the motif tessellates
    symmetrically.  `n` is chosen near N/sp (honouring the requested
    spacing); its parity is matched to N so a symmetric integer placement
    exists (an unpaired central wall is only possible when N is even)."""
    sp = max(1, int(sp))
    if N < 2:
        return []
    target = N / float(sp)
    n = max(1, min(int(round(target)), N))
    # An ODD division count places walls in mirror pairs and works for any
    # N; an EVEN count implies a lone central wall, which lands on an
    # integer boundary only when N is even.  So only when n is even AND N
    # is odd must we shift to the nearest ODD count (keeping the requested
    # spacing as closely as possible).
    if n % 2 == 0 and N % 2 == 1:
        cand = [c for c in (n - 1, n + 1) if 1 <= c <= N]
        n = min(cand, key=lambda c: (abs(c - target), -c))
    lines = sorted({int(round(N * m / float(n))) for m in range(1, n)})
    return [k for k in lines if 0 < k < N]


def _procedural_barriers(W, H, spacing, verticals, horizontals):
    """Internal walls on a regular `spacing` (in cells), placed
    symmetrically about the panel centre (see `_sym_lines`), optionally
    vertical and/or horizontal -- a clean, balanced, repeating array of
    knot cells rather than a lopsided one."""
    B = set()
    sp = max(1, int(spacing))
    if verticals:
        for k in _sym_lines(W, sp):
            i = 2 * k
            for j in range(1, 2 * H, 2):
                B.add((i, j))
    if horizontals:
        for k in _sym_lines(H, sp):
            j = 2 * k
            for i in range(1, 2 * W, 2):
                B.add((i, j))
    return B


def _random_barriers(W, H, seed, density):
    """Seeded internal barriers, each candidate slot included with
    probability `density`."""
    rng = _random.Random(int(seed))
    B = set()
    for slot in _internal_slots(W, H):
        if rng.random() < density:
            B.add(slot)
    return B


# --------------------------------------------------------------------
# Symmetry-group barriers (square lattice)
# --------------------------------------------------------------------
#
# The square panel [0, 2W] x [0, 2H] has the point group of the square
# (D4) about its centre (W, H) ONLY when it is square (W == H); a
# rectangular panel keeps the rectangle group D2.  A group element is an
# integer map of the (i, j) lattice that fixes the centre and carries the
# lattice to itself; the 90-degree elements swap the two axes, so they
# need W == H.  Symmetrizing a seed barrier set = unioning the COMPLETE
# orbit of every seed wall under the group (a wall is symmetrized only
# when its whole orbit lands on valid internal slots), which yields a set
# provably invariant under the group.

_SQUARE_GROUPS = ('NONE', 'C2', 'C4', 'D2', 'D4')
_HEX_GROUPS = ('NONE', 'C3', 'C6', 'D3', 'D6')
_SNUB_GROUPS = ('NONE', 'C2', 'C4')

# group name -> (rotation order n, has mirrors)
_GROUP_SPEC = {'C2': (2, False), 'C3': (3, False), 'C4': (4, False),
               'C6': (6, False), 'D2': (2, True), 'D3': (3, True),
               'D4': (4, True), 'D6': (6, True)}


def _square_group_ops(group, W, H):
    """Integer maps (i, j) -> (i, j) of the point group about the panel
    centre (W, H).  The 90-degree elements need a square panel (W == H);
    on a rectangle C4 degrades to C2 and D4 to D2."""
    if group == 'NONE':
        return [lambda i, j: (i, j)]
    square = (W == H)
    e = lambda i, j: (i, j)
    r180 = lambda i, j: (2 * W - i, 2 * H - j)
    mx = lambda i, j: (2 * W - i, j)                  # flip about vert. axis
    my = lambda i, j: (i, 2 * H - j)                  # flip about horiz. axis
    r90 = lambda i, j: (W + H - j, H - W + i)         # +90 about centre
    r270 = lambda i, j: (W - H + j, H + W - i)        # -90 about centre
    md = lambda i, j: (W - H + j, H - W + i)          # main-diagonal mirror
    ma = lambda i, j: (W + H - j, H + W - i)          # anti-diagonal mirror
    if group == 'C2':
        return [e, r180]
    if group == 'D2':
        return [e, r180, mx, my]
    if group == 'C4':
        return [e, r90, r180, r270] if square else [e, r180]
    if group == 'D4':
        return ([e, r90, r180, r270, mx, my, md, ma] if square
                else [e, r180, mx, my])
    return [e]


def _square_effective_group(group, W, H):
    """The group actually applied on a W x H panel (C4/D4 degrade to
    C2/D2 on a non-square panel)."""
    if group in ('C4', 'D4') and W != H:
        return 'C2' if group == 'C4' else 'D2'
    return group


def _valid_slot(i, j, W, H):
    return (i + j) % 2 == 1 and 0 < i < 2 * W and 0 < j < 2 * H


def _symmetrize_square(barriers, group, W, H):
    """Union the complete group orbit of every seed barrier (skipping any
    whose orbit strays off the internal slots), giving a barrier set
    invariant under the point group about the panel centre."""
    if group == 'NONE':
        return set(barriers)
    ops = _square_group_ops(group, W, H)
    out = set()
    for (i, j) in barriers:
        orbit = [op(i, j) for op in ops]
        if all(_valid_slot(oi, oj, W, H) for (oi, oj) in orbit):
            out.update(orbit)
    return out


# --------------------------------------------------------------------
# The threading permutation (mirror-curve billiard)
# --------------------------------------------------------------------

def _is_barrier(i, j, W, H, border, barriers):
    """True if the wall at edge-midpoint (i, j) blocks the cord."""
    if border and (i == 0 or i == 2 * W or j == 0 or j == 2 * H):
        return True
    return (i, j) in barriers


def _step(state, W, H, border, barriers):
    """Deterministic successor of a directed strand-segment.  A state is
    ((i, j), (di, dj)): located at edge-midpoint (i, j), about to travel
    in diagonal direction (di, dj).  Advance one segment to the next
    edge-midpoint; if that wall is a barrier the strand reflects (the
    perpendicular component flips).  With `border` off, opposite sides of
    the panel are identified (a tileable torus plait)."""
    (i, j), (di, dj) = state
    ni, nj = i + di, j + dj
    if not border:
        ni %= 2 * W
        nj %= 2 * H
    if _is_barrier(ni, nj, W, H, border, barriers):
        if ni % 2 == 0:                    # vertical wall -> flip di
            di = -di
        else:                              # horizontal wall -> flip dj
            dj = -dj
    return ((ni, nj), (di, dj))


def _all_states(W, H, border):
    """Every valid directed strand-segment of the panel."""
    states = []
    imax = 2 * W if not border else 2 * W + 1
    jmax = 2 * H if not border else 2 * H + 1
    for i in range(0, imax):
        for j in range(0, jmax):
            if (i + j) % 2 == 0:
                continue
            for d in _DIRS:
                ni, nj = i + d[0], j + d[1]
                if border and (ni < 0 or ni > 2 * W
                               or nj < 0 or nj > 2 * H):
                    continue
                states.append(((i, j), d))
    return states


def _cycles(W, H, border, barriers):
    """Partition every directed strand-segment into closed orbits of the
    threading map (a permutation, so the orbits are disjoint cycles that
    exhaust the state set)."""
    states = _all_states(W, H, border)
    seen = set()
    cycles = []
    for s in states:
        if s in seen:
            continue
        cyc = []
        cur = s
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            cur = _step(cur, W, H, border, barriers)
        cycles.append(cyc)
    return cycles, len(states)


def _seg_key(state, W, H, border):
    """Undirected segment (a frozenset of its two endpoints) drawn by a
    directed state -- the same for a cord and its reverse, so it groups
    the two orientations of one physical cord."""
    (i, j), (di, dj) = state
    ni, nj = i + di, j + dj
    if not border:
        ni %= 2 * W
        nj %= 2 * H
    return frozenset(((i, j), (ni, nj)))


def trace_cords(W, H, border, barriers):
    """Trace the closed cords of the plait.  Returns a list of cords,
    each a list of records (pos, is_barrier, dir) in cyclic order (an
    oriented closed loop), one cord per physical closed strand (its two
    orientations collapsed)."""
    cycles, _n = _cycles(W, H, border, barriers)
    groups = {}
    for cyc in cycles:
        sig = frozenset(_seg_key(s, W, H, border) for s in cyc)
        groups.setdefault(sig, cyc)          # keep one orientation
    cords = []
    for cyc in groups.values():
        rec = []
        for (pos, d) in cyc:
            isb = _is_barrier(pos[0], pos[1], W, H, border, barriers)
            rec.append((pos, isb, d))
        cords.append(rec)
    return cords


def count_loops(W, H, border, barriers):
    """Number of separate closed cords."""
    return len(trace_cords(W, H, border, barriers))


def _tile_split(rec, W, H):
    """Split a border-off (toroidal) cord into open, in-tile PIECES so
    that no drawn segment ever crosses a seam.

    A cord on the torus is a closed orbit whose stored lattice points wrap
    modulo the grid; drawn naively, a wrapping step joins opposite edges
    with one long diagonal spanning the whole panel (the border-off
    "spazz").  Here the cord is walked in UNWRAPPED coordinates (each step
    is a genuine unit diagonal), the continuous polyline is chopped at
    every tile boundary it crosses -- boundary crossings fall exactly on
    lattice points, since every step is a unit diagonal -- and each run is
    translated back into the fundamental tile [0,2W]x[0,2H].  A cord that
    never leaves the tile is returned as ONE closed piece; a cord that
    wraps becomes several open pieces that leave one edge exactly where
    the continuation enters the opposite edge, so the plait tiles
    seamlessly.

    Returns a list of (points, closed); each `points` is a list of
    (draw_xy, rec_item) where rec_item is the source (pos, is_barrier,
    dir) of that lattice point (used for the over/under crossings)."""
    Lx, Ly = 2 * W, 2 * H
    L = len(rec)
    # unwrapped vertices u[0..L] (u[L] closes the winding); every
    # consecutive pair differs by exactly one diagonal step.
    u = [None] * (L + 1)
    u[0] = (rec[0][0][0], rec[0][0][1])
    for k in range(L):
        dx, dy = rec[k][2]
        u[k + 1] = (u[k][0] + dx, u[k][1] + dy)
    # the tile cell each unit-diagonal segment lives in (its midpoint is
    # strictly interior, so the floor is unambiguous)
    cells = []
    for k in range(L):
        mx = (u[k][0] + u[k + 1][0]) * 0.5
        my = (u[k][1] + u[k + 1][1]) * 0.5
        cells.append((mx // Lx, my // Ly))
    # no seam crossed anywhere -> the whole cord fits one tile, closed
    cut = [cells[k] != cells[(k - 1) % L] for k in range(L)]
    if not any(cut):
        pts = [((float(rec[k][0][0]), float(rec[k][0][1])), rec[k])
               for k in range(L)]
        return [(pts, True)]
    start = cut.index(True)
    pieces = []
    cur = []
    for n in range(L):
        k = (start + n) % L
        ox, oy = cells[k]
        ds = (float(u[k][0] - ox * Lx), float(u[k][1] - oy * Ly))
        de = (float(u[k + 1][0] - ox * Lx), float(u[k + 1][1] - oy * Ly))
        if not cur:
            cur.append((ds, rec[k]))
        cur.append((de, rec[(k + 1) % L]))
        if cells[(k + 1) % L] != cells[k]:
            pieces.append((cur, False))
            cur = []
    if cur:
        pieces.append((cur, False))
    return pieces


# --------------------------------------------------------------------
# Over / under assignment (alternating knot, parity union-find)
# --------------------------------------------------------------------

def _solve_over(cords):
    """Assign a consistent alternating over/under.  For each cord, the
    over/under sense must flip between consecutive crossings; since a
    reflection swaps the strand's diagonal family, that is the parity
    constraint  x[c1] XOR x[c2] = 1 XOR (reflections between).  Resolve
    all constraints with a parity union-find and return x: crossing
    position -> bit (1 = the slope +1 family rides over there)."""
    dsu = isl._ParityDSU()
    keys = set()
    for rec in cords:
        L = len(rec)
        cross_idx = [k for k in range(L) if not rec[k][1]]
        for pos, isb, d in rec:
            if not isb:
                keys.add(pos)
        C = len(cross_idx)
        for a in range(C):
            k1 = cross_idx[a]
            k2 = cross_idx[(a + 1) % C]
            # count barrier positions strictly between k1 and k2 (cyclic)
            r = 0
            k = (k1 + 1) % L
            while k != k2:
                if rec[k][1]:
                    r += 1
                k = (k + 1) % L
            dsu.union(rec[k1][0], rec[k2][0], 1 ^ (r & 1))
    x = {}
    for key in keys:
        dsu.add(key)
        _root, par = dsu.find(key)
        x[key] = par
    return x


def _cord_signed(rec, x):
    """(control_pts, cross): the control polyline of one cord and its
    crossings as (control_index, over_sign, pos) with +1 over, -1 under
    and `pos` the crossing's lattice point (used to look up the over
    cord's tangent when cutting an under cord flush)."""
    control = [(float(pos[0]), float(pos[1])) for pos, _b, _d in rec]
    cross = []
    for k, (pos, isb, d) in enumerate(rec):
        if isb:
            continue
        fam = _family(d)
        over = x.get(pos, 0) ^ fam           # cord rides over here?
        cross.append((k, 1 if over else -1, pos))
    return control, cross


def _render_pieces(rec, x, W, H, border):
    """The drawable pieces of one cord as (control, cross, closed).  With a
    border every cord is a single CLOSED loop (unchanged behaviour); with
    the border off a cord is split into seam-free in-tile pieces (see
    `_tile_split`) -- one closed piece if it never wraps, else several open
    ones -- so no ribbon spans the panel."""
    if border:
        control, cross = _cord_signed(rec, x)
        return [(control, cross, True)]
    out = []
    for pts, closed in _tile_split(rec, W, H):
        control = [dp for dp, _e in pts]
        cross = []
        for idx, (_dp, (pos, isb, d)) in enumerate(pts):
            if isb:
                continue
            fam = _family(d)
            over = x.get(pos, 0) ^ fam
            cross.append((idx, 1 if over else -1, pos))
        out.append((control, cross, closed))
    return out


# --------------------------------------------------------------------
# Single-cord search
# --------------------------------------------------------------------

def make_single_cord(W, H, border, barriers, budget=4000):
    """Greedily toggle INTERNAL barriers until the plait is a single
    closed cord.  Toggling one wall merges or splits two cords (changing
    the count by one), so a merge always exists while more than one cord
    remains.  Returns (barriers, ok); ok is False if the budget ran out
    (log and keep the best found)."""
    B = set(barriers)
    cur = count_loops(W, H, border, B)
    slots = _internal_slots(W, H)
    it = 0
    while cur > 1 and it < budget:
        improved = False
        for s in slots:
            it += 1
            if it >= budget:
                break
            if s in B:
                B.discard(s)
            else:
                B.add(s)
            n = count_loops(W, H, border, B)
            if n < cur:
                cur = n
                improved = True
                break
            if s in B:                        # revert
                B.discard(s)
            else:
                B.add(s)
        if not improved:
            break
    return B, cur == 1


# --------------------------------------------------------------------
# Ribbon assembly (reusing the Islamic strapwork machinery)
# --------------------------------------------------------------------

_BACKING_MAT = len(pc.PALETTE_RGBA) - 1


def _split_at(path, closed, idxs):
    """Split a (closed) densified path into arcs at the sorted control
    indices `idxs`, returning [(subpath, arc_parity)] -- used for the
    over/under two-tone (CHECKER) coloring."""
    n = len(path)
    if not idxs or n < 2:
        return [(path, 0)]
    idxs = sorted(set(i % n for i in idxs))
    pieces = []
    m = len(idxs)
    for a in range(m if closed else m - 1):
        i0 = idxs[a]
        i1 = idxs[(a + 1) % m]
        if i1 > i0:
            sp = path[i0:i1 + 1]
        else:                                 # wrap across the seam
            sp = path[i0:] + path[:i1 + 1]
        if len(sp) >= 2:
            pieces.append((sp, a % 2))
    return pieces


def _path_tangent(path, i, closed):
    """Unit tangent of a drawn path at vertex `i` (from its neighbours),
    or None if degenerate."""
    n = len(path)
    if n < 2:
        return None
    if closed:
        a, b = path[(i - 1) % n], path[(i + 1) % n]
    else:
        a, b = path[max(0, i - 1)], path[min(n - 1, i + 1)]
    d = isl._unit(b[0] - a[0], b[1] - a[1])
    return None if d == (0.0, 0.0) else d


# --------------------------------------------------------------------
# Lenticular (pointed-leaf) reflection turns
# --------------------------------------------------------------------
#
# A smoothed cord makes a U-turn wherever it reflects off a wall; the
# plain Catmull-Rom rounds that into a semicircular cap.  The historical
# Celtic turn is instead LENTICULAR -- the two rails curve back to meet
# at a soft point offset from the wall (a pointed oval / leaf).  We keep
# the exact Catmull-Rom baseline (so ROUNDED is byte-identical and the
# crossing sample indices are untouched) and, for POINTED, add a local
# displacement over the two spline segments adjacent to each reflection
# vertex that pulls the curve toward two straight rails meeting at a tip.
# The displacement is scaled by `tightness` and vanishes at tightness 0,
# so POINTED continuously reduces to the round cap.

def _reflect_indices(ncontrol, cross):
    """Control indices of a piece that are REFLECTION (U-turn) vertices --
    every control point that is not a crossing."""
    cs = set(k for k, _sg, _pos in cross)
    return [k for k in range(ncontrol) if k not in cs]


def _pointed_path(path, control, closed, subdiv, tightness, reflect_idx):
    """Reshape the Catmull-Rom `path` so each reflection vertex in
    `reflect_idx` turns as a pointed leaf.  Returns a new list of xy
    tuples; the sample count and every crossing sample are unchanged."""
    n = len(path)
    if (tightness <= 0.0 or subdiv < 2 or len(control) < 3
            or not reflect_idx or n < 3):
        return path
    ncon = len(control)
    P = [np.asarray(p, float) for p in path]
    disp = [np.zeros(2) for _ in range(n)]

    def ctrl(k):
        return np.asarray(control[k % ncon] if closed
                          else control[min(max(k, 0), ncon - 1)], float)

    for i in reflect_idx:
        if not closed and (i <= 0 or i >= ncon - 1):
            continue                              # open-piece terminus
        A = ctrl(i)
        prev = ctrl(i - 1)
        nxt = ctrl(i + 1)
        mid = 0.5 * (prev + nxt)
        out = A - mid
        depth = float(np.hypot(out[0], out[1]))
        if depth < 1e-9:
            continue                              # not a genuine U-turn
        out = out / depth
        tip = A + out * (tightness * depth)
        m = (i * subdiv) % n if closed else i * subdiv
        # incoming half: samples prev-control .. apex (exclude apex)
        for o in range(subdiv):
            idx = (m - subdiv + o) % n if closed else (m - subdiv + o)
            if idx < 0:
                continue
            s = o / float(subdiv)
            target = prev + (tip - prev) * s
            disp[idx] = disp[idx] + tightness * (target - P[idx])
        # outgoing half: apex .. next-control (include apex once)
        for o in range(subdiv + 1):
            idx = (m + o) % n if closed else (m + o)
            if idx >= n:
                continue
            s = o / float(subdiv)
            target = tip + (nxt - tip) * s
            disp[idx] = disp[idx] + tightness * (target - P[idx])
    return [(float(P[k][0] + disp[k][0]), float(P[k][1] + disp[k][1]))
            for k in range(n)]


def _smooth_path(control, closed, subdiv, corner_style, tightness,
                 reflect_idx):
    """The smoothed centerline of one cord piece: the Catmull-Rom spline
    (corner_style ROUNDED -- byte-identical to the classic path) or the
    same spline with pointed lenticular reflection turns (POINTED)."""
    path = isl.catmull_rom(control, closed, subdiv)
    if corner_style != 'POINTED':
        return path
    return _pointed_path(path, control, closed, subdiv, tightness,
                         reflect_idx)


def _over_tangents(cords, x, W, H, border, style, subdiv,
                   corner_style='ROUNDED', corner_tightness=0.5):
    """Unit tangent of the OVER cord at every crossing lattice point,
    read from the SAME smoothed path the ribbon uses.  Cutting an under
    cord flush along the over cord needs the over cord's LOCAL direction
    at the crossing; in SMOOTH style that deviates from the lattice
    diagonals near border / break turns (and near POINTED leaf turns), so
    it must be measured from the Catmull-Rom curve rather than assumed to
    be a 45-degree diagonal."""
    tan = {}
    for rec in cords:
        for control, cross, closed in _render_pieces(rec, x, W, H, border):
            if len(control) < 2:
                continue
            smoothed = (style == 'SMOOTH' and len(control) >= 3
                        and subdiv >= 2)
            step = subdiv if smoothed else 1
            if style == 'SMOOTH':
                path = _smooth_path(control, closed, subdiv, corner_style,
                                    corner_tightness,
                                    _reflect_indices(len(control), cross))
            else:
                path = control
            for k, sg, pos in cross:
                if sg < 0:
                    continue                      # only the OVER passage
                t = _path_tangent(path, k * step, closed)
                if t is not None:
                    tan[pos] = t
    return tan


def _revert_spike_caps(left, right, orig, width, limit=1.8):
    """Undo a flush end-cut that overshot into a spike.  The over-edge
    flush cut (`isl._angle_cut_piece`) reprojects each cut cap onto the
    over cord's edge line; at a SHALLOW oblique crossing (the medial
    tilings have crossings well below 90 degrees) one rail can slide far
    along the near-parallel edge, stretching the cap into a long spike.
    Where the recut cap is longer than `limit` cord-widths -- i.e. much
    longer than the clean ~width/sin(theta) flush cap of a well-conditioned
    crossing -- restore that end's original (square-cut) rail points, so
    the cord ends in a clean cap instead of a sliver."""
    for idx in (0, -1):
        lx, ly = left[idx]
        rx, ry = right[idx]
        if hypot(lx - rx, ly - ry) > limit * width:
            left[idx], right[idx] = orig[idx]


def _piece_cell(control, cross, closed, width, style, subdiv, interlace,
                mode, weave_height, height, color_by, loop_index,
                over_tan=None, profile='FLAT', tube_sides=8,
                rope_strands=3, rope_twist=6.0, corner_style='ROUNDED',
                corner_tightness=0.5, oblique_caps=False):
    """Sub-cells (verts, faces, mats) for one drawable piece of a cord.  A
    bordered cord is a single closed piece; a border-off cord may be
    several open pieces (`closed` carries the distinction to the ribbon
    machinery, which handles both).  `over_tan` maps each crossing lattice
    point to the over cord's local tangent, so an under cord is cut flush
    ALONG the over cord's edge (FLAT interlace).

    With `profile` != 'FLAT' the flat ribbon is replaced by a swept round
    TUBE or twisted ROPE along the same centerline (see `_profile_cells`);
    FLAT keeps the exact ribbon geometry (byte-identical).  `corner_style`
    POINTED reshapes the smoothed reflection turns into pointed leaves.
    `oblique_caps` (set on the medial-tiling path) guards the flush cut
    against spikes at the tilings' shallow crossings; it is off on the
    square grid, whose 90-degree crossings never spike, keeping that path
    byte-identical."""
    if len(control) < 2:
        return []
    if profile != 'FLAT':
        return _profile_cells(control, cross, closed, width, style, subdiv,
                              interlace, mode, weave_height, color_by,
                              loop_index, profile, tube_sides,
                              rope_strands, rope_twist, corner_style,
                              corner_tightness)
    # catmull_rom only subdivides when it has >= 3 points and subdiv >= 2;
    # otherwise it returns the control points unchanged, so the crossing
    # index multiplier must stay 1 (short border-off seam pieces).
    smoothed = style == 'SMOOTH' and len(control) >= 3 and subdiv >= 2
    step = subdiv if smoothed else 1
    path = (_smooth_path(control, closed, subdiv, corner_style,
                         corner_tightness,
                         _reflect_indices(len(control), cross))
            if style == 'SMOOTH' else control)
    signed = [(k * step, sg) for k, sg, _pos in cross]

    def matof(default):
        if color_by == 'LOOP':
            return loop_index % len(pc.PALETTE_RGBA)
        if color_by == 'UNIFORM':
            return 0
        return default

    sub_cells = []

    if color_by == 'CHECKER':
        # two-tone by over/under: cut at every crossing and color arcs by
        # alternating parity (which tracks the over/under sense)
        idxs = [k * step for k, _sg, _pos in cross]
        for sp, par in _split_at(path, closed, idxs):
            left, right = isl.miter_ribbon(sp, width, False)
            cv, cf = isl.band_ribbon_faces(left, right, False, height)
            if cf:
                sub_cells.append((cv, cf, [par] * len(cf)))
    elif interlace and mode == 'WOVEN':
        zoff = isl._weave_zoff(path, closed, signed, weave_height)
        left, right = isl.miter_ribbon(path, width, closed)
        cv, cf = isl.band_ribbon_faces_z(left, right, closed, height, zoff)
        if cf:
            sub_cells.append((cv, cf, [matof(0)] * len(cf)))
    elif interlace and mode == 'FLAT':
        under = [(k * step, pos) for k, sg, pos in cross if sg < 0]
        ugeo = []
        if under:
            s, total = isl._arclen(path, closed)
            cut_s = sorted(s[pi] for pi, _pos in under)
            margin = max(0.02, 0.25 * width)
            half = 0.5 * (width + margin)
            pieces = isl._cut_band(path, closed, cut_s, half, s, total)
            # Cut the under cord flush ALONG the over cord's edge (a cap
            # parallel to the over cord, not perpendicular to the under
            # cord).  The over cord's tangent is read from its smoothed
            # path, so this holds for SMOOTH as well as ANGULAR cords.
            h_o = 0.5 * width
            gap = 0.5 * margin
            maxreach = 3.0 * width
            cut_gate = 1.5 * half
            if over_tan:
                for pi, pos in under:
                    t_o = over_tan.get(pos)
                    if t_o is None:
                        continue
                    ugeo.append((path[pi], t_o, (-t_o[1], t_o[0])))
        else:
            pieces = [(path, closed)]
        npieces = len(pieces)
        for pj, (sp, sp_closed) in enumerate(pieces):
            if len(sp) < 2:
                continue
            left, right = isl.miter_ribbon(sp, width, sp_closed)
            if ugeo and not sp_closed:
                # for an open (border-off) piece the first sub-piece's
                # start and the last's end are genuine termini, not cuts;
                # a closed cord has cuts at both ends of every sub-piece.
                start_struct = closed or pj != 0
                end_struct = closed or pj != npieces - 1
                orig = {0: (left[0], right[0]), -1: (left[-1], right[-1])}
                isl._angle_cut_piece(left, right, sp, start_struct,
                                     end_struct, ugeo, h_o, gap,
                                     maxreach, cut_gate)
                if oblique_caps:
                    _revert_spike_caps(left, right, orig, width)
            cv, cf = isl.band_ribbon_faces(left, right, sp_closed, height)
            if cf:
                sub_cells.append((cv, cf, [matof(0)] * len(cf)))
    else:                                     # plain flat ribbon
        left, right = isl.miter_ribbon(path, width, closed)
        cv, cf = isl.band_ribbon_faces(left, right, closed, height)
        if cf:
            sub_cells.append((cv, cf, [matof(0)] * len(cf)))

    return sub_cells


def _cord_cell(rec, x, W, H, border, width, style, subdiv, interlace, mode,
               weave_height, height, color_by, loop_index, over_tan=None,
               profile='FLAT', tube_sides=8, rope_strands=3,
               rope_twist=6.0, corner_style='ROUNDED', corner_tightness=0.5):
    """Build one merged (verts, faces, mats) cell for a single cord,
    gathering all of its drawable pieces (one when bordered, several across
    the seams when border-off)."""
    sub_cells = []
    for control, cross, closed in _render_pieces(rec, x, W, H, border):
        sub_cells += _piece_cell(control, cross, closed, width, style,
                                 subdiv, interlace, mode, weave_height,
                                 height, color_by, loop_index, over_tan,
                                 profile, tube_sides, rope_strands,
                                 rope_twist, corner_style, corner_tightness)
    if not sub_cells:
        return None
    return pc.merge_cells(sub_cells)


def build_cells(W, H, border, barriers, cord_width=0.25, style='ANGULAR',
                interlace=True, interlace_mode='FLAT', weave_height=0.06,
                color_by='CHECKER', height=0.0, backing=False, base=0.08,
                subdiv=8, substrate='SQUARE', profile='FLAT', tube_sides=8,
                rope_strands=3, rope_twist=6.0, trim=False,
                corner_style='ROUNDED', corner_tightness=0.5):
    """One merged (verts, faces, mats) cell per cord, plus an optional
    backing slab.  `cord_width` is a fraction of a cell (2 lattice
    units).  `substrate` selects the square grid (default) or a tiling
    medial knot; `profile` selects the flat ribbon (default), a round
    tube, or a twisted rope; `corner_style` POINTED gives lenticular
    reflection turns (SMOOTH only)."""
    if substrate != 'SQUARE':
        return _tiling_build_cells(
            substrate, W, H, border, barriers, cord_width, style,
            interlace, interlace_mode, weave_height, color_by, height,
            backing, base, subdiv, profile, tube_sides, rope_strands,
            rope_twist, trim, corner_style, corner_tightness)
    cords = trace_cords(W, H, border, barriers)
    x = _solve_over(cords)
    width = max(0.02, cord_width * 2.0)
    # Over cord tangents at every crossing, for the flush FLAT-interlace
    # cut (under cord cut ALONG the over cord's edge).
    over_tan = (_over_tangents(cords, x, W, H, border, style, subdiv,
                               corner_style, corner_tightness)
                if interlace and interlace_mode == 'FLAT' else None)
    cells = []
    all_verts = []
    for li, rec in enumerate(cords):
        cell = _cord_cell(rec, x, W, H, border, width, style, subdiv,
                          interlace, interlace_mode, weave_height, height,
                          color_by, li, over_tan, profile, tube_sides,
                          rope_strands, rope_twist, corner_style,
                          corner_tightness)
        if cell is None or not cell[1]:
            continue
        cells.append(cell)
        all_verts.extend(cell[0])
    if backing and all_verts:
        a = np.asarray(all_verts, float)
        lo = (a[:, 0].min(), a[:, 1].min())
        hi = (a[:, 0].max(), a[:, 1].max())
        cv, cf, cm = [], [], []
        pc.slab(cv, cf, cm, lo, hi, 0.0, -base, mat=_BACKING_MAT)
        cells.append((cv, cf, cm))
    return cells


def cord_paths(W, H, border, barriers, style='ANGULAR', subdiv=8,
               substrate='SQUARE', trim=False, corner_style='ROUNDED',
               corner_tightness=0.5):
    """Cord centerlines (control polyline, closed?) for the CURVE output.
    Bordered cords are single closed splines; border-off cords are split
    into seam-free in-tile pieces so no spline shoots across the panel."""
    if substrate != 'SQUARE':
        return _tiling_cord_paths(substrate, W, H, border, barriers,
                                  style, subdiv, trim, corner_style,
                                  corner_tightness)
    cords = trace_cords(W, H, border, barriers)
    out = []
    for rec in cords:
        if border:
            runs = [([(float(p[0]), float(p[1])) for p, _b, _d in rec],
                     [k for k, (_p, isb, _d) in enumerate(rec) if isb],
                     True)]
        else:
            runs = [([dp for dp, _e in pts],
                     [k for k, (_dp, (_p, isb, _d)) in enumerate(pts)
                      if isb], closed)
                    for pts, closed in _tile_split(rec, W, H)]
        for control, reflect_idx, closed in runs:
            if len(control) < 2:
                continue
            if style == 'SMOOTH':
                path = _smooth_path(control, closed, subdiv, corner_style,
                                    corner_tightness, reflect_idx)
            else:
                path = control
            out.append((path, closed))
    return out


def resolve_barriers(source, W, H, border, preset='PLAIN', wall_spacing=2,
                     walls_v=True, walls_h=True, seed=0, density=0.25,
                     single_cord=False, symmetry='NONE'):
    """The internal barrier set for a chosen source, applying the point-
    group symmetrization (PROCEDURAL / RANDOM seeds) and the single-cord
    search when requested.  Returns (barriers, note)."""
    note = ""
    if source == 'PRESET':
        return _preset_barriers(preset, W, H), note
    if source == 'PROCEDURAL':
        B = _procedural_barriers(W, H, wall_spacing, walls_v, walls_h)
    else:                                     # RANDOM
        B = _random_barriers(W, H, seed, density)
    if symmetry not in _SQUARE_GROUPS:
        symmetry = 'NONE'
    if symmetry != 'NONE':
        eff = _square_effective_group(symmetry, W, H)
        if eff != symmetry:
            note = "%s needs a square panel; using %s" % (symmetry, eff)
        B = _symmetrize_square(B, eff, W, H)
    if source == 'RANDOM' and single_cord:
        B, ok = make_single_cord(W, H, border, B)
        if not ok:
            note = (note + "; " if note else "") + \
                "single-cord search did not reach 1 loop"
    return B, note


# ====================================================================
# Arbitrary tilings: the medial / general-graph knot (Mercat)
# ====================================================================
#
# On any planar tiling the cord lives on the MEDIAL graph: a crossing at
# every edge midpoint, and around each face the cord connects consecutive
# edge midpoints (a corner-segment threading exactly one edge).  A dart
# is a directed corner-segment (face f, corner k, sense s): s = +1 leaves
# the corner across its CCW-outgoing edge, s = -1 across its incoming
# edge.  Its successor crosses that edge -- straight through the crossing
# (switch face AND far vertex) when the edge is an unwalled interior edge,
# else reflecting (U-turn within the same face) at a walled or boundary
# edge.  The successor is a bijection on darts, so its orbits partition
# every corner-segment into closed cords -- exactly the square grid's
# invariant, on an arbitrary graph.

_SUBSTRATE_TG = {
    'TRIANGLE': 'TRI', 'HEX': 'HEX', 'SNUBSQUARE': 'SNUBSQ',
    'TRIHEX': 'TRIHEX', 'TRUNCHEX': 'TRUNCHEX',
}


def _poly_ccw(coords, ring):
    """`ring` reordered so its vertices run counter-clockwise (positive
    signed area), so every face shares each edge in the opposite direction
    to its neighbour -- the consistent orientation the successor needs."""
    area = 0.0
    n = len(ring)
    for k in range(n):
        x0, y0 = coords[ring[k]]
        x1, y1 = coords[ring[(k + 1) % n]]
        area += x0 * y1 - x1 * y0
    return ring if area > 0.0 else ring[::-1]


def _tiling_patch(substrate, nx, ny, trim, pad=2):
    """Welded faces of a tiling patch as (coords, rings).  Reuses
    tiling_generator's regular-polygon substrates; with `trim` the patch
    is over-built and only faces whose centroid lies inside a clean
    central rectangle (`tg._trim_rect`) are kept -- a face SELECTION, so
    no polygon is cut and the medial knot stays a set of closed loops."""
    name = _SUBSTRATE_TG[substrate]
    if trim:
        polys = [p for p, _ in tg._base_iter(name, nx + pad, ny + pad)]
        rect = tg._trim_rect(polys, nx, ny, pad)
        x0, y0, x1, y1 = rect
        kept = [p for p in polys
                if x0 <= float(np.asarray(p, float).mean(axis=0)[0]) <= x1
                and y0 <= float(np.asarray(p, float).mean(axis=0)[1]) <= y1]
        polys = kept if kept else polys
    else:
        polys = [p for p, _ in tg._base_iter(name, nx, ny)]
    coords, rings = tg._weld(polys)
    rings = [_poly_ccw(coords, r) for r in rings]
    return coords, rings


def _tiling_topology(coords, rings):
    """Medial-graph topology of a welded patch: per-edge incident faces
    and midpoint, per-face vertex->index maps, and the interior / boundary
    edge-key lists.  An edge key is (min_vid, max_vid)."""
    edge_faces = {}
    edge_mid = {}
    vert_in_face = []
    for fi, r in enumerate(rings):
        n = len(r)
        vmap = {}
        for k in range(n):
            vmap[r[k]] = k
            a, b = r[k], r[(k + 1) % n]
            key = (a, b) if a < b else (b, a)
            edge_faces.setdefault(key, []).append(fi)
            if key not in edge_mid:
                ax, ay = coords[a]
                bx, by = coords[b]
                edge_mid[key] = (0.5 * (ax + bx), 0.5 * (ay + by))
        vert_in_face.append(vmap)
    interior = sorted(k for k, fs in edge_faces.items() if len(fs) == 2)
    boundary = sorted(k for k, fs in edge_faces.items() if len(fs) == 1)
    return {'edge_faces': edge_faces, 'edge_mid': edge_mid,
            'vert_in_face': vert_in_face, 'rings': rings, 'coords': coords,
            'interior': interior, 'boundary': boundary}


def _tiling_is_crossing(topo, key, walls):
    """An edge midpoint is a CROSSING (the cord passes straight through)
    iff the edge is shared by two faces and carries no wall; otherwise it
    REFLECTS (a boundary edge, or one walled by a barrier)."""
    return len(topo['edge_faces'][key]) == 2 and key not in walls


def _tiling_step(dart, topo, walls):
    """Deterministic successor of a dart (f, k, s) -- cross or reflect."""
    rings = topo['rings']
    ef = topo['edge_faces']
    vif = topo['vert_in_face']
    f, k, s = dart
    r = rings[f]
    n = len(r)
    if s > 0:
        a, b = r[k], r[(k + 1) % n]           # outgoing edge
    else:
        a, b = r[(k - 1) % n], r[k]           # incoming edge
    near = r[k]
    far = b if s > 0 else a
    key = (a, b) if a < b else (b, a)
    if _tiling_is_crossing(topo, key, walls):
        faces = ef[key]
        f2 = faces[0] if faces[1] == f else faces[1]
        k2 = vif[f2][far]
        r2 = rings[f2]
        n2 = len(r2)
        s2 = -1 if r2[(k2 + 1) % n2] == near else +1
        return (f2, k2, s2)
    if s > 0:                                  # reflect: U-turn in face f
        return (f, (k + 1) % n, +1)
    return (f, (k - 1) % n, -1)


def _tiling_exit(dart, topo, walls):
    """(exit_key, is_reflect, label): the edge midpoint the dart threads
    to, whether it reflects, and -- at a crossing -- which of the
    crossing's two strands (0/1) this passage rides, so one strand rides
    over and the other under."""
    rings = topo['rings']
    ef = topo['edge_faces']
    f, k, s = dart
    r = rings[f]
    n = len(r)
    if s > 0:
        a, b = r[k], r[(k + 1) % n]
    else:
        a, b = r[(k - 1) % n], r[k]
    near = r[k]
    key = (a, b) if a < b else (b, a)
    if not _tiling_is_crossing(topo, key, walls):
        return key, True, 0
    ka, kb = key
    f0, f1 = sorted(ef[key])
    if (f == f0 and near == ka) or (f == f1 and near == kb):
        return key, False, 0
    return key, False, 1


def _tiling_all_darts(topo):
    darts = []
    for f, r in enumerate(topo['rings']):
        for k in range(len(r)):
            darts.append((f, k, +1))
            darts.append((f, k, -1))
    return darts


def _tiling_cycles(topo, walls):
    """Partition every dart into orbits of the successor permutation."""
    darts = _tiling_all_darts(topo)
    seen = set()
    cycles = []
    for d0 in darts:
        if d0 in seen:
            continue
        cyc = []
        cur = d0
        while cur not in seen:
            seen.add(cur)
            cyc.append(cur)
            cur = _tiling_step(cur, topo, walls)
        cycles.append(cyc)
    return cycles, len(darts)


def _tiling_trace_cords(topo, walls):
    """Trace the closed cords of the medial knot.  A dart-cycle and its
    reverse use the same set of corners and describe one physical cord;
    keep one orientation each.  Returns cords, each a list of records
    (coord_xy, is_reflect, strand_label, crossing_id)."""
    cycles, _n = _tiling_cycles(topo, walls)
    em = topo['edge_mid']
    used = set()
    cords = []
    for cyc in cycles:
        corners = frozenset((f, k) for f, k, _s in cyc)
        if corners & used:
            continue
        used |= corners
        rec = []
        for dart in cyc:
            key, isr, lab = _tiling_exit(dart, topo, walls)
            rec.append((em[key], isr, lab, key))
        cords.append(rec)
    return cords


def _tiling_count_loops(topo, walls):
    return len(_tiling_trace_cords(topo, walls))


def _tiling_solve_over(cords):
    """Alternating over/under across the tiling's crossings.  Between
    consecutive crossings on a cord the reference-strand parity must flip:
    x[c1] XOR x[c2] = 1 XOR label1 XOR label2, solved with a parity union-
    find (isl._ParityDSU).  Non-bipartite tilings frustrate some
    constraints; the union-find simply drops the clashing ones (a graceful
    seam) rather than raising."""
    dsu = isl._ParityDSU()
    keys = set()
    for rec in cords:
        L = len(rec)
        cidx = [k for k in range(L) if not rec[k][1]]
        for _coord, isr, _lab, cid in rec:
            if not isr:
                keys.add(cid)
        C = len(cidx)
        for a in range(C):
            k1 = cidx[a]
            k2 = cidx[(a + 1) % C]
            dsu.union(rec[k1][3], rec[k2][3],
                      1 ^ rec[k1][2] ^ rec[k2][2])
    x = {}
    for key in sorted(keys):
        dsu.add(key)
        _r, par = dsu.find(key)
        x[key] = par
    return x


def _tiling_render_piece(rec, x):
    """(control, cross, closed) for one tiling cord: the edge-midpoint
    control polyline and its crossings as (control_index, over_sign,
    crossing_id).  Every tiling cord is a single closed loop."""
    control = [(float(c[0][0]), float(c[0][1])) for c in rec]
    cross = []
    for k, (_coord, isr, lab, cid) in enumerate(rec):
        if isr:
            continue
        over = x.get(cid, 0) ^ lab
        cross.append((k, 1 if over else -1, cid))
    return control, cross, True


def _tiling_over_tangents(cords, x, style, subdiv, corner_style='ROUNDED',
                          corner_tightness=0.5):
    """Over-cord tangent at every crossing (read from the SAME smoothed
    path the ribbon uses), for the flush FLAT-interlace cut -- the tiling
    analogue of `_over_tangents`.  Reading the tangent from the drawn path
    (rather than assuming a lattice direction) is what makes the flush cut
    correct at the oblique 60/120-degree crossings of the medial tilings,
    for ANGULAR as well as SMOOTH and POINTED cords."""
    tan = {}
    for rec in cords:
        control, cross, closed = _tiling_render_piece(rec, x)
        if len(control) < 2:
            continue
        smoothed = (style == 'SMOOTH' and len(control) >= 3
                    and subdiv >= 2)
        step = subdiv if smoothed else 1
        if style == 'SMOOTH':
            path = _smooth_path(control, closed, subdiv, corner_style,
                                corner_tightness,
                                _reflect_indices(len(control), cross))
        else:
            path = control
        for k, sg, cid in cross:
            if sg < 0:
                continue
            t = _path_tangent(path, k * step, closed)
            if t is not None:
                tan[cid] = t
    return tan


def _tiling_cord_cell(rec, x, width, style, subdiv, interlace, mode,
                      weave_height, height, color_by, loop_index, over_tan,
                      profile, tube_sides, rope_strands, rope_twist,
                      corner_style='ROUNDED', corner_tightness=0.5):
    """One merged (verts, faces, mats) cell for a single tiling cord,
    reusing the shared ribbon / tube pipeline (`_piece_cell`)."""
    control, cross, closed = _tiling_render_piece(rec, x)
    sub = _piece_cell(control, cross, closed, width, style, subdiv,
                      interlace, mode, weave_height, height, color_by,
                      loop_index, over_tan, profile, tube_sides,
                      rope_strands, rope_twist, corner_style,
                      corner_tightness, oblique_caps=True)
    if not sub:
        return None
    return pc.merge_cells(sub)


def _tiling_stripe_walls(topo, spacing, verticals, horizontals):
    """Procedural barriers on a tiling: wall every interior edge that
    straddles one of a set of evenly spaced vertical and/or horizontal
    fault lines -- a clean, regular array of break lines across the patch
    (the tiling analogue of the square grid's procedural walls)."""
    coords = topo['coords']
    interior = topo['interior']
    if not interior:
        return set()
    xs, ys = [], []
    for a, b in interior:
        xs += [coords[a][0], coords[b][0]]
        ys += [coords[a][1], coords[b][1]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sp = max(1, int(spacing))

    def lines(lo, hi):
        span = hi - lo
        n = max(1, int(round(span / float(sp))))
        return [lo + span * m / float(n) for m in range(1, n)]

    walls = set()
    vlines = lines(x0, x1) if verticals else []
    hlines = lines(y0, y1) if horizontals else []
    for key in interior:
        a, b = key
        xa, xb = coords[a][0], coords[b][0]
        ya, yb = coords[a][1], coords[b][1]
        if any((xa - lx) * (xb - lx) < -1e-12 for lx in vlines):
            walls.add(key)
        elif any((ya - ly) * (yb - ly) < -1e-12 for ly in hlines):
            walls.add(key)
    return walls


def _tiling_make_single_cord(topo, walls, budget=3000):
    """Greedily toggle interior-edge walls until the medial knot is a
    single closed cord (each toggle changes the loop count by one, so a
    merge exists while more than one cord remains).  Returns (walls, ok)."""
    W = set(walls)
    cur = _tiling_count_loops(topo, W)
    slots = list(topo['interior'])
    it = 0
    while cur > 1 and it < budget:
        improved = False
        for s in slots:
            it += 1
            if it >= budget:
                break
            if s in W:
                W.discard(s)
            else:
                W.add(s)
            n = _tiling_count_loops(topo, W)
            if n < cur:
                cur = n
                improved = True
                break
            if s in W:
                W.discard(s)
            else:
                W.add(s)
        if not improved:
            break
    return W, cur == 1


# --------------------------------------------------------------------
# Symmetry-group barriers (tiling substrates)
# --------------------------------------------------------------------
#
# The triangular / hexagonal family carries the 60/120-degree point
# groups (C3/C6/D3/D6) about a lattice symmetry centre; the snub-square
# lattice carries the 90-degree rotation groups (C2/C4).  A group element
# acts on the plane; a wall (an interior edge) maps to the interior edge
# whose midpoint lands on the image of its midpoint.  The rotation centre
# (and mirror axis) is chosen by scanning lattice-feature candidates and
# taking the one that carries the most edge midpoints back onto edge
# midpoints, so it is a genuine symmetry centre of the patch.  As on the
# square grid, only COMPLETE orbits are unioned, giving a set provably
# invariant under the group.

def _tiling_mid_index(topo):
    """dict rounded(midpoint) -> interior edge key, for symmetry lookups."""
    idx = {}
    for key in topo['interior']:
        mx, my = topo['edge_mid'][key]
        idx[(round(mx, 3), round(my, 3))] = key
    return idx


def _tiling_lookup_edge(mid_index, p):
    return mid_index.get((round(float(p[0]), 3), round(float(p[1]), 3)))


def _tiling_symmetry_matrices(topo, group):
    """Point-group transforms for a tiling patch: pick the rotation centre
    (and, for dihedral groups, the mirror axis) that best maps the patch's
    edge midpoints onto themselves, and return the list of 3x3 matrices.
    Returns None for the trivial group."""
    spec = _GROUP_SPEC.get(group)
    if spec is None:
        return None
    n, mir = spec
    coords = topo['coords']
    interior = topo['interior']
    mids = [topo['edge_mid'][k] for k in interior]
    if not mids:
        return None
    mid_index = _tiling_mid_index(topo)
    bx = 0.5 * (min(m[0] for m in mids) + max(m[0] for m in mids))
    by = 0.5 * (min(m[1] for m in mids) + max(m[1] for m in mids))
    # candidate symmetry centres: face centroids, vertices, edge midpoints
    cands = []
    for r in topo['rings']:
        pts = np.asarray([coords[v] for v in r], float)
        c = pts.mean(axis=0)
        cands.append((float(c[0]), float(c[1])))
    cands += [(float(x), float(y)) for x, y in coords]
    cands += [(float(m[0]), float(m[1])) for m in mids]
    ang = 2.0 * pi / n

    def rot_score(cx, cy):
        M = pc.Rot(ang, cx, cy)
        q = pc.apply(M, mids)
        return sum(1 for p in q if _tiling_lookup_edge(mid_index, p))

    best, bestc = -1, (bx, by)
    for (cx, cy) in cands:
        sc = rot_score(cx, cy)
        # prefer higher overlap, then nearer the patch centre
        d = (cx - bx) ** 2 + (cy - by) ** 2
        if sc > best or (sc == best
                         and d < (bestc[0] - bx) ** 2 + (bestc[1] - by) ** 2):
            best, bestc = sc, (cx, cy)
    cx, cy = bestc
    mats = [pc.Rot(ang * k, cx, cy) for k in range(n)]
    if mir:
        # choose the mirror axis (through the centre) with the best overlap
        baxis, bscore = 0.0, -1
        for t in range(180):
            a = pi * t / 180.0
            M = pc.Mir(a, cx, cy)
            q = pc.apply(M, mids)
            sc = sum(1 for p in q if _tiling_lookup_edge(mid_index, p))
            if sc > bscore:
                bscore, baxis = sc, a
        mats += [pc.Mir(baxis + pi * k / n, cx, cy) for k in range(n)]
    return mats


def _tiling_symmetrize(walls, topo, group):
    """Union the complete group orbit of every seed wall (skipping walls
    whose orbit strays off the interior edges), giving a wall set
    invariant under the point group of the patch."""
    if group == 'NONE' or not walls:
        return set(walls)
    mats = _tiling_symmetry_matrices(topo, group)
    if not mats:
        return set(walls)
    mid_index = _tiling_mid_index(topo)
    em = topo['edge_mid']
    out = set()
    for key in walls:
        p0 = em[key]
        orbit = []
        ok = True
        for M in mats:
            q = pc.apply(M, [p0])[0]
            k = _tiling_lookup_edge(mid_index, q)
            if k is None:
                ok = False
                break
            orbit.append(k)
        if ok:
            out.update(orbit)
    return out


def _tiling_groups_for(substrate):
    """The symmetry groups exposed for a tiling substrate."""
    if substrate == 'SNUBSQUARE':
        return _SNUB_GROUPS
    return _HEX_GROUPS


def _tiling_resolve_walls(substrate, nx, ny, trim, source, seed, density,
                          wall_spacing, walls_v, walls_h, single_cord,
                          symmetry='NONE'):
    """The walled-edge set for a tiling knot under a chosen barrier source
    (PRESET -> the plain knot; PROCEDURAL -> regular fault lines; RANDOM ->
    seeded edge-walls, optionally solved to a single cord), symmetrized
    under the chosen point group.  Returns (walls, note)."""
    coords, rings = _tiling_patch(substrate, nx, ny, trim)
    topo = _tiling_topology(coords, rings)
    note = ""
    if symmetry not in _tiling_groups_for(substrate):
        symmetry = 'NONE'
    if source == 'PROCEDURAL':
        walls = _tiling_stripe_walls(topo, wall_spacing, walls_v, walls_h)
        walls = _tiling_symmetrize(walls, topo, symmetry)
    elif source == 'RANDOM':
        rng = _random.Random(int(seed))
        walls = set(k for k in topo['interior'] if rng.random() < density)
        walls = _tiling_symmetrize(walls, topo, symmetry)
        if single_cord:
            walls, ok = _tiling_make_single_cord(topo, walls)
            if not ok:
                note = "single-cord search did not reach 1 loop"
    else:                                      # PRESET -> plain knot
        walls = set()
    return walls, note


def _tiling_cord_count(substrate, nx, ny, trim, walls):
    coords, rings = _tiling_patch(substrate, nx, ny, trim)
    topo = _tiling_topology(coords, rings)
    return _tiling_count_loops(topo, walls if walls else set())


def _tiling_build_cells(substrate, nx, ny, border, walls, cord_width,
                        style, interlace, mode, weave_height, color_by,
                        height, backing, base, subdiv, profile, tube_sides,
                        rope_strands, rope_twist, trim,
                        corner_style='ROUNDED', corner_tightness=0.5):
    """One merged (verts, faces, mats) cell per cord of a tiling medial
    knot, plus an optional backing slab -- the tiling analogue of
    `build_cells`, flowing through the same ribbon / tube pipeline."""
    coords, rings = _tiling_patch(substrate, nx, ny, trim)
    topo = _tiling_topology(coords, rings)
    walls = walls if walls else set()
    cords = _tiling_trace_cords(topo, walls)
    x = _tiling_solve_over(cords)
    # Tiling edges have unit length; scale the cord width to the tile so
    # the ribbon reads the same across substrates.
    width = max(0.02, cord_width * 1.2)
    over_tan = (_tiling_over_tangents(cords, x, style, subdiv, corner_style,
                                      corner_tightness)
                if interlace and mode == 'FLAT' and profile == 'FLAT'
                else None)
    cells = []
    all_verts = []
    for li, rec in enumerate(cords):
        cell = _tiling_cord_cell(rec, x, width, style, subdiv, interlace,
                                 mode, weave_height, height, color_by, li,
                                 over_tan, profile, tube_sides,
                                 rope_strands, rope_twist, corner_style,
                                 corner_tightness)
        if cell is None or not cell[1]:
            continue
        cells.append(cell)
        all_verts.extend(cell[0])
    if backing and all_verts:
        a = np.asarray(all_verts, float)
        lo = (a[:, 0].min(), a[:, 1].min())
        hi = (a[:, 0].max(), a[:, 1].max())
        cv, cf, cm = [], [], []
        pc.slab(cv, cf, cm, lo, hi, 0.0, -base, mat=_BACKING_MAT)
        cells.append((cv, cf, cm))
    return cells


def _tiling_cord_paths(substrate, nx, ny, border, walls, style, subdiv,
                       trim, corner_style='ROUNDED', corner_tightness=0.5):
    """Cord centerlines (path, closed) for the CURVE output on a tiling."""
    coords, rings = _tiling_patch(substrate, nx, ny, trim)
    topo = _tiling_topology(coords, rings)
    cords = _tiling_trace_cords(topo, walls if walls else set())
    out = []
    for rec in cords:
        control = [(float(c[0][0]), float(c[0][1])) for c in rec]
        if len(control) < 2:
            continue
        reflect_idx = [k for k, c in enumerate(rec) if c[1]]
        if style == 'SMOOTH':
            path = _smooth_path(control, True, subdiv, corner_style,
                                corner_tightness, reflect_idx)
        else:
            path = control
        out.append((path, True))
    return out


# ====================================================================
# TUBE / ROPE profile: sweep a round or twisted cross-section along a cord
# ====================================================================
#
# The flat ribbon is replaced by a swept solid: a round TUBE, or a ROPE of
# several helical sub-strands twisting about the cord centerline.  The
# sweep runs along the SAME centerline the ribbon uses, so it composes
# with angular / smooth cords, both substrates, color-by-loop, and the
# interlace: FLAT breaks the under-tube at crossings (via `isl._cut_band`,
# exactly as the ribbon breaks the under-cord), WOVEN lifts and dips the
# tube in z (via `isl._weave_zoff`).

def _frames(path3, closed):
    """An approximate parallel-transport frame (unit side, up) at every
    vertex of a 3D polyline: the previous side vector re-orthogonalized
    against the new tangent, so the frame does not spin as the cord
    curves (clean, twist-free tubes)."""
    n = len(path3)
    P = [np.asarray(p, float) for p in path3]
    T = []
    for i in range(n):
        if closed:
            a, b = P[(i - 1) % n], P[(i + 1) % n]
        else:
            a, b = P[max(0, i - 1)], P[min(n - 1, i + 1)]
        d = b - a
        L = float(np.linalg.norm(d))
        T.append(d / L if L > 1e-12 else np.array([1.0, 0.0, 0.0]))
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, T[0]))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    side = np.cross(T[0], ref)
    side /= (np.linalg.norm(side) + 1e-12)
    sides = [side]
    ups = [np.cross(T[0], side)]
    for i in range(1, n):
        s = sides[-1] - float(np.dot(sides[-1], T[i])) * T[i]
        ln = float(np.linalg.norm(s))
        if ln < 1e-9:
            s = np.cross(T[i], ups[-1])
            ln = float(np.linalg.norm(s))
        s = s / (ln + 1e-12)
        sides.append(s)
        ups.append(np.cross(T[i], s))
    return P, sides, ups


def _stitch_rings(rings, closed, cap):
    """(verts, faces) from a list of equal-length vertex rings: quads
    between consecutive rings, plus triangle-fan end caps on an open
    sweep so the tube is watertight."""
    cv, base = [], []
    for ring in rings:
        base.append(len(cv))
        cv.extend(ring)
    cf = []
    m = len(rings)
    nseg = len(rings[0]) if rings else 0
    span = m if closed else m - 1
    for i in range(span):
        a, b = base[i], base[(i + 1) % m]
        for j in range(nseg):
            j2 = (j + 1) % nseg
            cf.append((a + j, a + j2, b + j2, b + j))
    if cap and not closed and m >= 1 and nseg >= 3:
        c0 = len(cv)
        cv.append(tuple(np.mean(np.asarray(rings[0], float), axis=0)))
        for j in range(nseg):
            cf.append((c0, base[0] + (j + 1) % nseg, base[0] + j))
        c1 = len(cv)
        cv.append(tuple(np.mean(np.asarray(rings[-1], float), axis=0)))
        for j in range(nseg):
            cf.append((c1, base[-1] + j, base[-1] + (j + 1) % nseg))
    return cv, cf


def _sweep_tube(path3, closed, radius, nseg):
    """Sweep a circular cross-section of `radius` (nseg facets) along a 3D
    centerline -> (verts, faces)."""
    nseg = max(3, int(nseg))
    if len(path3) < 2:
        return [], []
    P, sides, ups = _frames(path3, closed)
    rings = []
    for i in range(len(P)):
        ring = []
        for j in range(nseg):
            th = 2.0 * pi * j / nseg
            pt = P[i] + radius * (cos(th) * sides[i] + sin(th) * ups[i])
            ring.append((float(pt[0]), float(pt[1]), float(pt[2])))
        rings.append(ring)
    return _stitch_rings(rings, closed, cap=True)


def _sweep_rope(path3, closed, radius, strands, nseg, twist):
    """Sweep a ROPE -- `strands` smaller tubes helically twisting about the
    centerline at `twist` radians per unit arclength -> (verts, faces)."""
    if len(path3) < 2:
        return [], []
    nseg = max(3, int(nseg))
    strands = max(2, int(strands))
    P, sides, ups = _frames(path3, closed)
    s, total = isl._arclen(path3, closed)
    r_major = 0.5 * radius
    r_minor = 0.55 * radius
    cv, cf = [], []
    for st in range(strands):
        phase = 2.0 * pi * st / strands
        rings = []
        for i in range(len(P)):
            phi = phase + twist * s[i]
            c = P[i] + r_major * (cos(phi) * sides[i] + sin(phi) * ups[i])
            ring = []
            for j in range(nseg):
                th = 2.0 * pi * j / nseg
                pt = c + r_minor * (cos(th) * sides[i]
                                    + sin(th) * ups[i])
                ring.append((float(pt[0]), float(pt[1]), float(pt[2])))
            rings.append(ring)
        scv, scf = _stitch_rings(rings, closed, cap=True)
        b0 = len(cv)
        cv.extend(scv)
        cf.extend(tuple(b0 + q for q in f) for f in scf)
    return cv, cf


def _profile_cells(control, cross, closed, width, style, subdiv, interlace,
                   mode, weave_height, color_by, loop_index, profile,
                   tube_sides, rope_strands, rope_twist,
                   corner_style='ROUNDED', corner_tightness=0.5):
    """Tube / rope sub-cells for one drawable piece of a cord, mirroring
    the ribbon branches of `_piece_cell`: CHECKER splits at crossings,
    WOVEN rides the weave z-offset, FLAT breaks the under-passages, else a
    continuous swept solid."""
    smoothed = style == 'SMOOTH' and len(control) >= 3 and subdiv >= 2
    step = subdiv if smoothed else 1
    path = (_smooth_path(control, closed, subdiv, corner_style,
                         corner_tightness,
                         _reflect_indices(len(control), cross))
            if style == 'SMOOTH' else [tuple(p) for p in control])
    signed = [(k * step, sg) for k, sg, _pos in cross]
    r = 0.5 * width

    def matof(default):
        if color_by == 'LOOP':
            return loop_index % len(pc.PALETTE_RGBA)
        if color_by == 'UNIFORM':
            return 0
        return default

    def sweep(path3, cl, mat):
        if profile == 'ROPE':
            cv, cf = _sweep_rope(path3, cl, r, rope_strands, tube_sides,
                                 rope_twist)
        else:
            cv, cf = _sweep_tube(path3, cl, r, tube_sides)
        return (cv, cf, [mat] * len(cf)) if cf else None

    sub_cells = []
    if color_by == 'CHECKER':
        idxs = [k * step for k, _sg, _pos in cross]
        for sp, par in _split_at(path, closed, idxs):
            c = sweep([(p[0], p[1], 0.0) for p in sp], False, par)
            if c:
                sub_cells.append(c)
        return sub_cells
    if interlace and mode == 'WOVEN':
        zoff = isl._weave_zoff(path, closed, signed, weave_height)
        p3 = [(path[i][0], path[i][1], zoff[i]) for i in range(len(path))]
        c = sweep(p3, closed, matof(0))
        if c:
            sub_cells.append(c)
        return sub_cells
    if interlace and mode == 'FLAT':
        under = [k * step for k, sg, _pos in cross if sg < 0]
        if under:
            s, total = isl._arclen(path, closed)
            cut_s = sorted(s[pi] for pi in under)
            margin = max(0.02, 0.25 * width)
            half = 0.5 * (width + margin)
            pieces = isl._cut_band(path, closed, cut_s, half, s, total)
        else:
            pieces = [(path, closed)]
        for sp, sp_closed in pieces:
            if len(sp) < 2:
                continue
            c = sweep([(p[0], p[1], 0.0) for p in sp], sp_closed, matof(0))
            if c:
                sub_cells.append(c)
        return sub_cells
    c = sweep([(p[0], p[1], 0.0) for p in path], closed, matof(0))
    if c:
        sub_cells.append(c)
    return sub_cells


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


_PRESET_ITEMS = [
    ('PLAIN', "Plain Plait", "No internal walls -- the open plait"),
    ('CROSSBREAK', "Cross Break",
     "A central vertical and horizontal break -- a four-panel knot"),
    ('PANEL', "Framed Panel",
     "An inset rectangular ring of barriers framing a central motif"),
    ('TRIQUETRA', "Triquetra (Trinity)",
     "The three-cornered trinity knot -- best on a 4x4 panel"),
    ('JOSEPHINE', "Josephine Knot",
     "The two-cord carrick / Josephine knot -- best on a wide 6x3 panel"),
    ('KELLS_BORDER', "Kells Border",
     "A repeating illuminated-border plait -- best on a wide panel"),
    ('CARPET', "Carpet Page",
     "A filled all-over carpet-page repeat -- best on a 6x6+ panel"),
]

# Symmetry-group enum items, filtered per substrate by _symmetry_items.
# Kept at module scope so the dynamic EnumProperty callback returns stable
# string references (avoiding the Blender enum-GC pitfall).
_SYM_ITEMS_SQUARE = [
    ('NONE', "None", "No symmetrization (seed barriers as generated)"),
    ('C2', "C2 (180 rot)", "Two-fold rotational symmetry"),
    ('C4', "C4 (90 rot)", "Four-fold rotational symmetry (square panel)"),
    ('D2', "D2 (rot + mirror)", "Two-fold rotations with mirrors"),
    ('D4', "D4 (rot + mirror)",
     "Full square symmetry: 90-deg rotations + mirrors (square panel)"),
]
_SYM_ITEMS_HEX = [
    ('NONE', "None", "No symmetrization (seed barriers as generated)"),
    ('C3', "C3 (120 rot)", "Three-fold rotational symmetry"),
    ('C6', "C6 (60 rot)", "Six-fold rotational symmetry"),
    ('D3', "D3 (rot + mirror)", "Three-fold rotations with mirrors"),
    ('D6', "D6 (rot + mirror)", "Six-fold rotations with mirrors"),
]
_SYM_ITEMS_SNUB = [
    ('NONE', "None", "No symmetrization (seed barriers as generated)"),
    ('C2', "C2 (180 rot)", "Two-fold rotational symmetry"),
    ('C4', "C4 (90 rot)", "Four-fold rotational symmetry"),
]


if _IN_BLENDER:

    def _symmetry_items(self, context):
        """Group choices compatible with the current substrate: the
        square groups on the grid, the 60/120-degree groups on the
        triangular / hexagonal family, and the square-rotation groups on
        the (chiral) snub-square tiling."""
        if self.substrate == 'SQUARE':
            return _SYM_ITEMS_SQUARE
        if self.substrate == 'SNUBSQUARE':
            return _SYM_ITEMS_SNUB
        return _SYM_ITEMS_HEX

    def _emit_curve(context, name, paths, span=2.0, operator=None):
        """Build a curve object whose cyclic POLY splines are the cord
        centerlines, centered and scaled to the span cube."""
        paths = [(list(p), c) for p, c in paths if len(p) >= 2]
        if not paths:
            return None
        allv = np.asarray([p for pa, _c in paths for p in pa], float)
        lo, hi = allv.min(axis=0), allv.max(axis=0)
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        s = span / max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        for pa, closed in paths:
            sp = cu.splines.new('POLY')
            sp.points.add(len(pa) - 1)
            for i, (px, py) in enumerate(pa):
                sp.points[i].co = ((px - cx) * s, (py - cy) * s, 0.0, 1.0)
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

    class MESH_OT_celtic_knot_2d_add(bpy.types.Operator, AddObjectHelper):
        """Add a 2D Celtic knot (grid-and-barriers plait)"""
        bl_idname = "mesh.celtic_knot_2d_add"
        bl_label = "Celtic Knot 2D"
        bl_options = {'REGISTER', 'UNDO'}

        substrate: EnumProperty(
            name="Substrate",
            items=[('SQUARE', "Square Grid",
                    "The classic grid plait (the mirror-curve billiard)"),
                   ('TRIANGLE', "Triangular (3.3.3.3.3.3)",
                    "Medial knot on the triangular tiling"),
                   ('HEX', "Hexagonal (6.6.6)",
                    "Medial knot on the hexagonal tiling"),
                   ('SNUBSQUARE', "Snub Square (3.3.4.3.4)",
                    "Medial knot on the snub-square Archimedean tiling"),
                   ('TRIHEX', "Trihexagonal (3.6.3.6)",
                    "Medial knot on the trihexagonal Archimedean tiling"),
                   ('TRUNCHEX', "Truncated Hexagonal (3.12.12)",
                    "Medial knot on the truncated-hexagonal tiling")],
            default='SQUARE',
            description="The graph the cord lives on: the square grid, or "
                        "a tiling's medial graph (Mercat's knot on any "
                        "graph)")
        grid_w: IntProperty(name="Grid Width", default=5, min=1, max=40)
        grid_h: IntProperty(name="Grid Height", default=5, min=1, max=40)
        trim: BoolProperty(
            name="Trim Boundary", default=True,
            description="Keep only tiling faces inside a clean central "
                        "rectangle (tiling substrates)")
        border: BoolProperty(
            name="Border", default=True,
            description="Ring the panel with barriers (bordered knot); "
                        "off = an open, tileable plait")
        barriers: EnumProperty(
            name="Barriers",
            items=[('PRESET', "Preset", "A named barrier panel"),
                   ('PROCEDURAL', "Procedural",
                    "Internal walls on a regular spacing"),
                   ('RANDOM', "Random", "Seeded internal barriers")],
            default='PRESET',
            description="Source of the internal barriers that carve the "
                        "knot")
        preset: EnumProperty(name="Panel", items=_PRESET_ITEMS,
                             default='PLAIN')
        wall_spacing: IntProperty(
            name="Wall Spacing", default=2, min=1, max=20,
            description="Internal walls every this many cells "
                        "(procedural)")
        walls_v: BoolProperty(name="Vertical Walls", default=True)
        walls_h: BoolProperty(name="Horizontal Walls", default=True)
        seed: IntProperty(name="Seed", default=0, min=0, max=100000)
        density: FloatProperty(
            name="Barrier Density", default=0.25, min=0.0, max=1.0,
            description="Fraction of internal walls made barriers "
                        "(random)")
        single_cord: BoolProperty(
            name="Single Cord", default=False,
            description="Adjust random barriers to weave one continuous "
                        "closed cord where feasible")
        symmetry: EnumProperty(
            name="Symmetry",
            items=_symmetry_items,
            description="Replicate the procedural / random barrier seed "
                        "under a point group about the panel centre, so "
                        "the knot comes out balanced (Cn rotational, Dn "
                        "rotational + mirror)")
        cord_width: FloatProperty(
            name="Cord Width", default=0.25, min=0.02, max=0.6,
            description="Cord ribbon width as a fraction of a cell")
        profile: EnumProperty(
            name="Profile",
            items=[('FLAT', "Flat Ribbon",
                    "The classic flat interlaced tape"),
                   ('TUBE', "Round Tube",
                    "Sweep a round cross-section along each cord"),
                   ('ROPE', "Twisted Rope",
                    "Sweep several helical sub-strands (cordage)")],
            default='FLAT',
            description="Cross-section swept along each cord's centerline")
        tube_sides: IntProperty(
            name="Tube Sides", default=8, min=3, max=24,
            description="Facets around the tube / rope cross-section")
        rope_strands: IntProperty(
            name="Rope Strands", default=3, min=2, max=5,
            description="Number of twisted sub-strands (rope profile)")
        rope_twist: FloatProperty(
            name="Rope Twist", default=8.0, min=0.0, max=40.0,
            description="Rope sub-strand twist, radians per unit length")
        style: EnumProperty(
            name="Style",
            items=[('ANGULAR', "Angular",
                    "Straight mitered cords with sharp corners"),
                   ('SMOOTH', "Smooth",
                    "Catmull-Rom curved cords with rounded U-turns")],
            default='SMOOTH')
        smoothness: IntProperty(
            name="Smoothness", default=8, min=2, max=32,
            description="Spline subdivisions per cord segment (smooth)")
        corner_style: EnumProperty(
            name="Corner Style",
            items=[('ROUNDED', "Rounded",
                    "The classic round Catmull-Rom U-turn"),
                   ('POINTED', "Pointed (Lenticular)",
                    "Reflection turns curve back to a soft point -- the "
                    "authentic Celtic leaf / pointed-oval turn")],
            default='ROUNDED',
            description="Shape of the U-turn where a smooth cord reflects "
                        "off a wall")
        corner_tightness: FloatProperty(
            name="Corner Tightness", default=0.5, min=0.0, max=1.0,
            description="How sharp the pointed leaf turn is (0 = the round "
                        "cap, 1 = a sharp tip)")
        interlace: BoolProperty(
            name="Interlace (Weave)", default=True,
            description="Weave the cords over and under (alternating "
                        "knotwork)")
        interlace_mode: EnumProperty(
            name="Interlace Mode",
            items=[('FLAT', "Flat Knotwork",
                    "Break the under-cord at crossings (2D interlace)"),
                   ('WOVEN', "Woven (3D)",
                    "Raise/dip each cord into a 3D woven surface")],
            default='FLAT')
        weave_height: FloatProperty(
            name="Weave Height", default=0.06, min=0.0, max=0.5,
            description="Z amplitude of the woven cords (woven only)")
        color_by: EnumProperty(
            name="Color By",
            items=[('UNIFORM', "Uniform", "A single material"),
                   ('LOOP', "By Cord",
                    "A distinct color per closed cord"),
                   ('CHECKER', "Over/Under",
                    "Two-tone by the over/under weave")],
            default='LOOP')
        output: EnumProperty(
            name="Output",
            items=[('RIBBON', "Ribbon Mesh",
                    "Filled cord ribbons (supports relief)"),
                   ('CURVE', "Centerline Curves",
                    "Cord centerlines as a Blender curve object")],
            default='RIBBON')
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat ribbons; > 0 extrudes the cords")
        backing: BoolProperty(
            name="Backing Slab", default=False,
            description="Add a slab behind the knot")
        base: FloatProperty(
            name="Base Thickness", default=0.08, min=0.01, max=1.0)
        separate: BoolProperty(
            name="Separate Cords", default=False,
            description="Output each closed cord as its own object")

        def execute(self, context):
            W, H = self.grid_w, self.grid_h
            tiling = self.substrate != 'SQUARE'
            if tiling:
                barriers, note = _tiling_resolve_walls(
                    self.substrate, W, H, self.trim, self.barriers,
                    self.seed, self.density, self.wall_spacing,
                    self.walls_v, self.walls_h, self.single_cord,
                    self.symmetry)
            else:
                barriers, note = resolve_barriers(
                    self.barriers, W, H, self.border, self.preset,
                    self.wall_spacing, self.walls_v, self.walls_h,
                    self.seed, self.density, self.single_cord,
                    self.symmetry)
            if note:
                self.report({'WARNING'}, note)
            if self.output == 'CURVE':
                paths = cord_paths(W, H, self.border, barriers,
                                   self.style, self.smoothness,
                                   self.substrate, self.trim,
                                   self.corner_style, self.corner_tightness)
                obj = _emit_curve(context, "Celtic Knot 2D", paths,
                                  operator=self)
                if obj is None:
                    self.report({'ERROR'}, "no knot generated")
                    return {'CANCELLED'}
                obj["math_art_pattern"] = True
                self.report({'INFO'}, "%s  %d cords" %
                            (self.substrate, len(paths)))
                return {'FINISHED'}
            cells = build_cells(
                W, H, self.border, barriers, self.cord_width,
                self.style, self.interlace, self.interlace_mode,
                self.weave_height, self.color_by, self.height,
                self.backing, self.base, self.smoothness,
                self.substrate, self.profile, self.tube_sides,
                self.rope_strands, self.rope_twist, self.trim,
                self.corner_style, self.corner_tightness)
            obj = pc.emit(context, "Celtic Knot 2D", cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no knot generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if tiling:
                n_cords = _tiling_cord_count(self.substrate, W, H,
                                             self.trim, barriers)
            else:
                n_cords = count_loops(W, H, self.border, barriers)
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s  %d cords  V=%d F=%d" %
                            (self.substrate, n_cords,
                             len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s  %d cords" %
                            (self.substrate, len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            tiling = self.substrate != 'SQUARE'
            lay.prop(self, 'substrate')
            lay.prop(self, 'grid_w')
            lay.prop(self, 'grid_h')
            if tiling:
                lay.prop(self, 'trim')
            else:
                lay.prop(self, 'border')
            lay.prop(self, 'barriers')
            if self.barriers == 'PRESET':
                if not tiling:                 # presets are square-only
                    lay.prop(self, 'preset')
            elif self.barriers == 'PROCEDURAL':
                lay.prop(self, 'wall_spacing')
                lay.prop(self, 'walls_v')
                lay.prop(self, 'walls_h')
                lay.prop(self, 'symmetry')
            else:                              # RANDOM
                lay.prop(self, 'seed')
                lay.prop(self, 'density')
                lay.prop(self, 'single_cord')
                lay.prop(self, 'symmetry')
            lay.prop(self, 'cord_width')
            lay.prop(self, 'style')
            if self.style == 'SMOOTH':
                lay.prop(self, 'smoothness')
                lay.prop(self, 'corner_style')
                if self.corner_style == 'POINTED':
                    lay.prop(self, 'corner_tightness')
            lay.prop(self, 'output')
            if self.output == 'RIBBON':
                lay.prop(self, 'profile')
                if self.profile != 'FLAT':
                    lay.prop(self, 'tube_sides')
                    if self.profile == 'ROPE':
                        lay.prop(self, 'rope_strands')
                        lay.prop(self, 'rope_twist')
                lay.prop(self, 'interlace')
                if self.interlace:
                    lay.prop(self, 'interlace_mode')
                    if self.interlace_mode == 'WOVEN':
                        lay.prop(self, 'weave_height')
                lay.prop(self, 'color_by')
                if self.profile == 'FLAT':
                    lay.prop(self, 'height')
                lay.prop(self, 'backing')
                if self.backing:
                    lay.prop(self, 'base')
                lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.celtic_knot_2d_add", icon='MOD_LATTICE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_celtic_knot_2d_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_celtic_knot_2d_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _check_partition(W, H, border, barriers):
    """Every directed segment used exactly once; every cord closes."""
    states = _all_states(W, H, border)
    cycles, n = _cycles(W, H, border, barriers)
    used = [s for cyc in cycles for s in cyc]
    all_once = len(used) == n and len(set(used)) == n == len(states)
    # closure: stepping the last state of a cycle returns to the first
    closes = True
    for cyc in cycles:
        if _step(cyc[-1], W, H, border, barriers) != cyc[0]:
            closes = False
            break
    return all_once, closes


def _check_over_under(W, H, border, barriers):
    """Over/under strictly alternates along every cord, and each crossing
    has exactly one strand over and one under."""
    cords = trace_cords(W, H, border, barriers)
    x = _solve_over(cords)
    alternates = True
    visits = {}                               # crossing -> list of overs
    for rec in cords:
        _control, cross = _cord_signed(rec, x)
        signs = [sg for _k, sg, _pos in cross]
        L = len(signs)
        for a in range(L):
            if signs[a] == signs[(a + 1) % L]:
                alternates = False
        for k, sg, _pos in cross:
            pos = rec[k][0]
            visits.setdefault(pos, []).append(sg > 0)
    one_over = True
    for pos, ov in visits.items():
        if len(ov) != 2 or ov[0] == ov[1]:
            one_over = False
            break
    return alternates, one_over, len(visits)


def _check_local(W, H, barriers):
    """Border-off render pieces are seam-free: every drawn segment is a
    single local diagonal (|dx| <= 1 and |dy| <= 1), catching the "long
    diagonal shooting across the panel" bug directly; every piece is
    non-degenerate; and the pieces cover every physical cord segment
    exactly once (no dangling, no run-off)."""
    cords = trace_cords(W, H, False, barriers)
    x = _solve_over(cords)
    max_dx = max_dy = 0.0
    drawn = 0
    physical = sum(len(rec) for rec in cords)   # one segment per lattice pt
    for rec in cords:
        for control, _cross, closed in _render_pieces(rec, x, W, H, False):
            if len(control) < 2:
                return False, False, False
            span = len(control) if closed else len(control) - 1
            drawn += span
            for a in range(span):
                bx, by = control[a]
                cx, cy = control[(a + 1) % len(control)]
                max_dx = max(max_dx, abs(cx - bx))
                max_dy = max(max_dy, abs(cy - by))
    local = max_dx <= 1.0 + 1e-9 and max_dy <= 1.0 + 1e-9
    covers = drawn == physical
    return local, covers, (max_dx, max_dy)


def _check_symmetry(W, H, sp):
    """Procedural barriers are mirror-symmetric about the panel centre
    (invariant under i -> 2W-i and j -> 2H-j) -- a balanced, repeating
    array rather than a lopsided one."""
    B = _procedural_barriers(W, H, sp, True, True)
    mv = all((2 * W - i, j) in B for (i, j) in B)
    mh = all((i, 2 * H - j) in B for (i, j) in B)
    return mv and mh, len(B)


def _check_reflection():
    """A single internal barrier reflects the cord: adding one wall to a
    plain plait changes the threading (and hence the loop count)."""
    W = H = 4
    base = count_loops(W, H, True, set())
    B = {(4, 3)}                              # one internal vertical wall
    changed = count_loops(W, H, True, B) != base
    return changed


def _nondegenerate_ribbons():
    """A built patch has real, finite geometry."""
    B = _preset_barriers('CROSSBREAK', 5, 5)
    cells = build_cells(5, 5, True, B, interlace=True,
                        interlace_mode='FLAT', color_by='LOOP')
    faces = sum(len(c[1]) for c in cells)
    finite = all(all(np.isfinite(v).all() for v in c[0]) for c in cells)
    return faces > 0 and finite


# The pre-change SQUARE + FLAT geometry, hashed, so the self-test can
# assert the byte-identity of the classic grid path across all the
# tiling / profile additions.
_SQUARE_FLAT_HASHES = {
    'a': 'd23e805be4289d9ad8c7524bf72704ad0f6e500f2e3782ccb33215959a251169',
    'b': 'd2eda752b23ff1fb91149023645abe4acb4c4ef1741973acb7e25e1b0f65f127',
    'c': '26990291f3f26680fef63c671afd78e3330a1a67627e3f1c6d99cc70c0df8fed',
}


def _cells_hash(cells):
    import hashlib
    import json
    s = []
    for cv, cf, cm in cells:
        s.append(('V', [tuple(round(x, 6) for x in v) for v in cv]))
        s.append(('F', [tuple(f) for f in cf]))
        s.append(('M', list(cm)))
    return hashlib.sha256(json.dumps(s).encode()).hexdigest()


def _check_square_byte_identical():
    """SQUARE + FLAT output is byte-identical to the pre-change grid path
    (guards the classic plait against the tiling / profile additions)."""
    a = _cells_hash(build_cells(
        5, 5, True, _preset_barriers('CROSSBREAK', 5, 5), 0.25, 'SMOOTH',
        True, 'FLAT', 0.06, 'LOOP', 0.0, False, 0.08, 8))
    b = _cells_hash(build_cells(
        6, 4, True, set(), 0.3, 'ANGULAR', True, 'WOVEN', 0.06, 'CHECKER',
        0.1, True, 0.08, 8))
    c = _cells_hash(build_cells(
        6, 6, True, _random_barriers(6, 6, 3, 0.2), 0.25, 'SMOOTH', True,
        'FLAT', 0.06, 'LOOP', 0.0, False, 0.08, 8))
    return (a == _SQUARE_FLAT_HASHES['a']
            and b == _SQUARE_FLAT_HASHES['b']
            and c == _SQUARE_FLAT_HASHES['c'])


_TILING_SUBSTRATES = ('TRIANGLE', 'HEX', 'SNUBSQUARE', 'TRIHEX', 'TRUNCHEX')


def _check_tiling(substrate, nx=4, ny=4, trim=False, walls=None):
    """Full medial-knot invariants for a tiling substrate: the successor
    is a bijection on darts, its orbits partition every corner-segment
    into closed cords, no cord segment spans more than one edge, and every
    crossing has exactly one strand over and one under.  Returns a dict of
    booleans and counts."""
    walls = walls or set()
    coords, rings = _tiling_patch(substrate, nx, ny, trim)
    topo = _tiling_topology(coords, rings)
    darts = _tiling_all_darts(topo)
    succ = [_tiling_step(d, topo, walls) for d in darts]
    bij = len(set(succ)) == len(darts) and set(succ) == set(darts)
    cycles, n = _tiling_cycles(topo, walls)
    covered = [d for cyc in cycles for d in cyc]
    partition = (len(covered) == n == len(darts)
                 and len(set(covered)) == n)
    closes = all(_tiling_step(cyc[-1], topo, walls) == cyc[0]
                 for cyc in cycles)
    cords = _tiling_trace_cords(topo, walls)
    spans1 = True
    for rec in cords:
        L = len(rec)
        for k in range(L):
            if not (set(rec[k][3]) & set(rec[(k + 1) % L][3])):
                spans1 = False
    x = _tiling_solve_over(cords)
    visits = {}
    alternates = True
    for rec in cords:
        cidx = [k for k in range(len(rec)) if not rec[k][1]]
        signs = [x.get(rec[k][3], 0) ^ rec[k][2] for k in cidx]
        C = len(signs)
        for a in range(C):
            if signs[a] == signs[(a + 1) % C]:
                alternates = False
        for k in cidx:
            visits.setdefault(rec[k][3], []).append(
                x.get(rec[k][3], 0) ^ rec[k][2])
    one_over = all(len(v) == 2 and v[0] != v[1] for v in visits.values())
    return {'bij': bij, 'partition': partition, 'closes': closes,
            'spans1': spans1, 'one_over': one_over,
            'alternates': alternates, 'cords': len(cords),
            'crossings': len(visits)}


def _check_tiling_geometry(substrate, profile, mode='FLAT'):
    """A tiling knot builds finite, non-empty geometry under a profile."""
    cells = _tiling_build_cells(
        substrate, 4, 4, True, set(), 0.25, 'SMOOTH', True, mode, 0.06,
        'LOOP', 0.0, False, 0.08, 8, profile, 8, 3, 8.0, True)
    faces = sum(len(c[1]) for c in cells)
    finite = all(all(np.isfinite(v).all() for v in c[0]) for c in cells)
    return faces > 0 and finite


def _check_profile_geometry():
    """TUBE and ROPE profiles build finite, non-degenerate geometry on the
    square grid, across the interlace modes and color modes."""
    ok = True
    B = _preset_barriers('CROSSBREAK', 5, 5)
    for profile in ('TUBE', 'ROPE'):
        for mode, cby in (('FLAT', 'LOOP'), ('WOVEN', 'UNIFORM'),
                          ('FLAT', 'CHECKER')):
            cells = build_cells(5, 5, True, B, 0.25, 'SMOOTH', True, mode,
                                0.06, cby, 0.0, False, 0.08, 8,
                                substrate='SQUARE', profile=profile,
                                tube_sides=8, rope_strands=3,
                                rope_twist=8.0)
            faces = sum(len(c[1]) for c in cells)
            finite = all(all(np.isfinite(v).all() for v in c[0])
                         for c in cells)
            ok = ok and faces > 0 and finite
    return ok


def _check_square_symmetry(group, W, H, seed=3, density=0.3):
    """A symmetrized square barrier set is invariant under every element of
    its point group (applying each element leaves the set unchanged), and
    the resulting knot still partitions into closed cords with a valid
    over/under."""
    B = _random_barriers(W, H, seed, density)
    S = _symmetrize_square(B, group, W, H)
    ops = _square_group_ops(group, W, H)
    invariant = all({op(i, j) for (i, j) in S} == S for op in ops)
    p, c = _check_partition(W, H, True, S)
    alt, one, _nx = _check_over_under(W, H, True, S)
    return (invariant and p and c and alt and one), len(S), len(ops)


def _check_tiling_symmetry(substrate, group, nx, ny, trim=False):
    """A symmetrized tiling wall set is invariant under every element of
    its point group, and yields a valid closed-cord knot.  Symmetrizing
    the whole interior edge set exercises every orbit and guarantees a
    non-trivial invariant result."""
    coords, rings = _tiling_patch(substrate, nx, ny, trim)
    topo = _tiling_topology(coords, rings)
    seed_walls = set(topo['interior'])
    S = _tiling_symmetrize(seed_walls, topo, group)
    mats = _tiling_symmetry_matrices(topo, group)
    mid_index = _tiling_mid_index(topo)
    em = topo['edge_mid']
    invariant = True
    for M in mats:
        mapped = set()
        good = True
        for key in S:
            k = _tiling_lookup_edge(mid_index, pc.apply(M, [em[key]])[0])
            if k is None:
                good = False
                break
            mapped.add(k)
        if not good or mapped != S:
            invariant = False
            break
    r = _check_tiling(substrate, nx, ny, trim=trim, walls=S)
    valid = r['bij'] and r['partition'] and r['one_over']
    return (invariant and valid and len(S) > 0), len(S), len(mats)


def _check_pointed_corner():
    """POINTED reflection turns: the pointed path has the same sample count
    as the round one, reduces EXACTLY to the round cap at tightness 0, is
    finite, and genuinely deviates (non-degenerate) at tightness > 0."""
    W = H = 5
    B = _preset_barriers('CROSSBREAK', W, H)
    cords = trace_cords(W, H, True, B)
    x = _solve_over(cords)
    subdiv = 8
    same = reduces = finite = True
    max_dev = 0.0
    tested = 0
    for rec in cords:
        for control, cross, closed in _render_pieces(rec, x, W, H, True):
            reflect = _reflect_indices(len(control), cross)
            if not reflect or len(control) < 3:
                continue
            tested += 1
            rp = _smooth_path(control, closed, subdiv, 'ROUNDED', 0.5,
                              reflect)
            p0 = _smooth_path(control, closed, subdiv, 'POINTED', 0.0,
                              reflect)
            pt = _smooth_path(control, closed, subdiv, 'POINTED', 0.6,
                              reflect)
            same = same and len(rp) == len(pt) == len(p0)
            reduces = reduces and all(
                abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12
                for a, b in zip(rp, p0))
            finite = finite and all(np.isfinite(q).all() for q in pt)
            for a, b in zip(rp, pt):
                max_dev = max(max_dev, hypot(a[0] - b[0], a[1] - b[1]))
    # geometry with pointed turns is non-empty and finite
    cells = build_cells(W, H, True, B, 0.25, 'SMOOTH', True, 'FLAT', 0.06,
                        'LOOP', 0.0, False, 0.08, 8, 'SQUARE', 'FLAT', 8,
                        3, 8.0, False, 'POINTED', 0.6)
    faces = sum(len(c[1]) for c in cells)
    geo_ok = faces > 0 and all(all(np.isfinite(v).all() for v in c[0])
                               for c in cells)
    return (same and reduces and finite and max_dev > 1e-6 and geo_ok
            and tested > 0), max_dev


def _check_pointed_tiling():
    """POINTED turns compose with the medial-tiling ribbon (finite, non-
    empty geometry, and byte-identical to ROUNDED at tightness 0)."""
    a = _cells_hash(_tiling_build_cells(
        'TRIHEX', 4, 4, True, set(), 0.25, 'SMOOTH', True, 'FLAT', 0.06,
        'LOOP', 0.0, False, 0.08, 8, 'FLAT', 8, 3, 8.0, True, 'ROUNDED',
        0.5))
    b = _cells_hash(_tiling_build_cells(
        'TRIHEX', 4, 4, True, set(), 0.25, 'SMOOTH', True, 'FLAT', 0.06,
        'LOOP', 0.0, False, 0.08, 8, 'FLAT', 8, 3, 8.0, True, 'POINTED',
        0.0))
    cells = _tiling_build_cells(
        'HEX', 4, 4, True, set(), 0.25, 'SMOOTH', True, 'FLAT', 0.06,
        'LOOP', 0.0, False, 0.08, 8, 'FLAT', 8, 3, 8.0, True, 'POINTED',
        0.6)
    faces = sum(len(c[1]) for c in cells)
    finite = all(all(np.isfinite(v).all() for v in c[0]) for c in cells)
    return a == b and faces > 0 and finite


def _check_historical_preset(name, W, H):
    """A historical panel preset yields a valid closed-cord knot: the
    threading partitions into closed cords with a valid alternating
    over/under, and the ribbon builds finite geometry."""
    B = _preset_barriers(name, W, H)
    p, c = _check_partition(W, H, True, B)
    alt, one, _nx = _check_over_under(W, H, True, B)
    n = count_loops(W, H, True, B)
    cells = build_cells(W, H, True, B, interlace=True, interlace_mode='FLAT',
                        color_by='LOOP', style='SMOOTH')
    faces = sum(len(c[1]) for c in cells)
    finite = all(all(np.isfinite(v).all() for v in c[0]) for c in cells)
    return (p and c and alt and one and n >= 1 and faces > 0 and finite), n


def _check_tiling_flush_cut(substrate, nx=5, ny=5, trim=True):
    """The FLAT-interlace under-cord endcap on a medial tiling is cut
    parallel to (flush along) the over-cord's edge.  For every under-
    crossing whose over-cord tangent is known, the reprojected cap segment
    (the pair of rail cap points on the cut end) must run parallel to that
    over tangent -- validating the oblique 60/120-degree tiling cut, for
    ANGULAR as well as SMOOTH cords."""
    coords, rings = _tiling_patch(substrate, nx, ny, trim)
    topo = _tiling_topology(coords, rings)
    cords = _tiling_trace_cords(topo, set())
    x = _tiling_solve_over(cords)
    style = 'ANGULAR'
    subdiv = 8
    over_tan = _tiling_over_tangents(cords, x, style, subdiv)
    width = max(0.02, 0.25 * 1.2)
    margin = max(0.02, 0.25 * width)
    half = 0.5 * (width + margin)
    h_o = 0.5 * width
    gap = 0.5 * margin
    maxreach = 3.0 * width
    cut_gate = 1.5 * half
    checked = 0
    parallel = True
    worst = 0.0
    for rec in cords:
        control, cross, closed = _tiling_render_piece(rec, x)
        path = control
        under = [(k, pos) for k, sg, pos in cross if sg < 0]
        if not under:
            continue
        s, total = isl._arclen(path, closed)
        cut_s = sorted(s[k] for k, _pos in under)
        pieces = isl._cut_band(path, closed, cut_s, half, s, total)
        ugeo = []
        for k, pos in under:
            t_o = over_tan.get(pos)
            if t_o is not None:
                ugeo.append((path[k], t_o, (-t_o[1], t_o[0])))
        if not ugeo:
            continue
        for sp, sp_closed in pieces:
            if len(sp) < 2:
                continue
            left, right = isl.miter_ribbon(sp, width, sp_closed)
            orig = {0: (left[0], right[0]), -1: (left[-1], right[-1])}
            isl._angle_cut_piece(left, right, sp, True, True, ugeo, h_o,
                                 gap, maxreach, cut_gate)
            _revert_spike_caps(left, right, orig, width)
            # each end cap: no cap is a spike, and every cap that was cut
            # flush (moved from its square terminus) is parallel to the
            # nearest over-crossing tangent.
            for idx in (0, -1):
                cap = (right[idx][0] - left[idx][0],
                       right[idx][1] - left[idx][1])
                clen = hypot(cap[0], cap[1])
                if clen > 1.8 * width + 1e-9:
                    parallel = False          # spike survived
                if clen < 1e-9:
                    continue
                moved = (hypot(left[idx][0] - orig[idx][0][0],
                               left[idx][1] - orig[idx][0][1]) > 1e-9
                         or hypot(right[idx][0] - orig[idx][1][0],
                                  right[idx][1] - orig[idx][1][1]) > 1e-9)
                if not moved:
                    continue                  # clean square/perp cap
                P = sp[idx]
                best = min(ugeo, key=lambda g: hypot(P[0] - g[0][0],
                                                     P[1] - g[0][1]))
                t_o = best[1]
                cx = abs((cap[0] / clen) * t_o[1] - (cap[1] / clen) * t_o[0])
                worst = max(worst, cx)
                checked += 1
                if cx > 0.02:              # ~1.1 degree tolerance
                    parallel = False
    return parallel and checked > 0, checked, worst


if __name__ == "__main__":
    ok = True

    # 1. threading partitions all segments into closed cords
    for (W, H, bd) in [(5, 5, True), (6, 4, True), (7, 3, True),
                       (5, 5, False), (4, 6, False)]:
        B = _random_barriers(W, H, 3, 0.2)
        p, c = _check_partition(W, H, bd, B)
        ok = ok and p and c
        print("partition %dx%d border=%-5s : all-once=%s closes=%s"
              % (W, H, bd, p, c))

    # 2. plain bordered plait loop count == gcd(W, H)
    gcd_ok = True
    for (W, H) in [(5, 5), (6, 4), (8, 6), (7, 3), (9, 6), (12, 8)]:
        n = count_loops(W, H, True, set())
        exp = gcd(W, H)
        good = n == exp
        gcd_ok = gcd_ok and good
        print("plain plait %2dx%-2d : loops=%d  gcd=%d  %s"
              % (W, H, n, exp, "OK" if good else "BAD"))
    ok = ok and gcd_ok

    # 3. over/under alternates + one-over-one-under per crossing
    for (W, H, src) in [(5, 5, 'PLAIN'), (6, 5, 'CROSSBREAK'),
                        (5, 5, 'PANEL')]:
        B = _preset_barriers(src, W, H)
        alt, one, nx = _check_over_under(W, H, True, B)
        ok = ok and alt and one
        print("interlace %dx%d %-10s : alternate=%s one-over=%s "
              "crossings=%d" % (W, H, src, alt, one, nx))

    # 4. barriers reflect
    refl = _check_reflection()
    ok = ok and refl
    print("barrier reflects (changes threading): %s" % refl)

    # 4b. border-off (toroidal) pieces are local and cover every segment
    for (W, H) in [(5, 5), (6, 4), (4, 6), (7, 5)]:
        for src in (set(), _procedural_barriers(W, H, 2, True, True),
                    _random_barriers(W, H, 5, 0.2)):
            loc, cov, mx = _check_local(W, H, src)
            ok = ok and loc and cov
        print("border-off local %dx%d : local=%s covers=%s maxstep=%s"
              % (W, H, loc, cov, mx))

    # 4c. border-off over/under: consistent for even dims, graceful (no
    # crash / still one-over-one-under) when odd parity frustrates it
    for (W, H) in [(6, 4), (4, 6), (8, 6)]:
        alt, one, nx = _check_over_under(W, H, False, set())
        ok = ok and alt and one
        print("border-off weave %dx%d even : alternate=%s one-over=%s "
              "crossings=%d" % (W, H, alt, one, nx))
    for (W, H) in [(5, 5), (7, 5)]:
        alt, one, nx = _check_over_under(W, H, False, set())  # no crash
        print("border-off weave %dx%d odd  : alternate=%s one-over=%s "
              "crossings=%d (frustration handled)" % (W, H, alt, one, nx))

    # 4d. border stays byte-identical when the toroidal path is unused
    for (W, H) in [(5, 5), (6, 4), (7, 3)]:
        n = count_loops(W, H, True, set())
        good = n == gcd(W, H)
        ok = ok and good
    print("bordered gcd loop counts unchanged: %s" % good)

    # 4e. procedural walls are mirror-symmetric (balanced repeating array)
    sym_ok = True
    for (W, H, sp) in [(6, 6, 2), (5, 5, 2), (7, 4, 3), (8, 8, 2),
                       (9, 6, 3), (5, 7, 2)]:
        s, nb = _check_symmetry(W, H, sp)
        sym_ok = sym_ok and s
        print("procedural symmetric %dx%d sp=%d : %s (%d walls)"
              % (W, H, sp, s, nb))
    ok = ok and sym_ok

    # 5. RANDOM single_cord reaches one loop where feasible
    sc_ok = True
    for seed in (1, 2, 3, 7, 11):
        B = _random_barriers(6, 6, seed, 0.15)
        B2, good = make_single_cord(6, 6, True, B)
        n = count_loops(6, 6, True, B2)
        sc_ok = sc_ok and (n == 1)
        print("single-cord seed=%2d : loops=%d ok=%s" % (seed, n, n == 1))
    ok = ok and sc_ok

    # 6. non-degenerate ribbons
    nd = _nondegenerate_ribbons()
    ok = ok and nd
    print("ribbons non-degenerate: %s" % nd)

    # 6b. border-off ribbons build finite, non-empty geometry across the
    # options (plain / procedural, interlace flat & woven, color modes)
    nd2 = True
    for (W, H, use_proc, mode, cby) in [
            (6, 6, False, 'FLAT', 'LOOP'),
            (6, 6, True, 'FLAT', 'CHECKER'),
            (6, 4, True, 'WOVEN', 'UNIFORM')]:
        B = (_procedural_barriers(W, H, 2, True, True) if use_proc
             else set())
        cells = build_cells(W, H, False, B, interlace=True,
                            interlace_mode=mode, color_by=cby)
        faces = sum(len(c[1]) for c in cells)
        finite = all(all(np.isfinite(v).all() for v in c[0])
                     for c in cells)
        nd2 = nd2 and faces > 0 and finite
    ok = ok and nd2
    print("border-off ribbons non-degenerate: %s" % nd2)

    # 7. SQUARE + FLAT byte-identical to the pre-change grid path
    bi = _check_square_byte_identical()
    ok = ok and bi
    print("SQUARE+FLAT byte-identical to pre-change: %s" % bi)

    # 8. ARBITRARY TILINGS -- medial knot invariants per substrate:
    # successor bijection, orbits partition all segments into closed
    # loops, no segment spans more than one edge, one-over-one-under, and
    # over/under alternates (bipartite) or the frustration seam is handled
    print("-- tiling medial knots --")
    for sub in _TILING_SUBSTRATES:
        r = _check_tiling(sub, 4, 4, trim=False)
        good = (r['bij'] and r['partition'] and r['closes']
                and r['spans1'] and r['one_over'])
        ok = ok and good
        print("tiling %-11s : bij=%s part=%s closes=%s span1edge=%s "
              "1over1under=%s alt=%s cords=%d crossings=%d"
              % (sub, r['bij'], r['partition'], r['closes'], r['spans1'],
                 r['one_over'], r['alternates'], r['cords'],
                 r['crossings']))

    # 8b. tiling walls (procedural fault lines / random + single-cord)
    # still yield a valid closed-loop knot
    print("-- tiling barriers --")
    coords, rings = _tiling_patch('HEX', 5, 5, True)
    topo = _tiling_topology(coords, rings)
    pw = _tiling_stripe_walls(topo, 2, True, True)
    r = _check_tiling('HEX', 5, 5, trim=True, walls=pw)
    proc_ok = r['bij'] and r['partition'] and r['one_over']
    ok = ok and proc_ok
    print("HEX procedural walls=%d : valid=%s cords=%d"
          % (len(pw), proc_ok, r['cords']))
    rw, note = _tiling_resolve_walls('TRIHEX', 5, 5, True, 'RANDOM', 3,
                                     0.2, 2, True, True, True)
    n1 = _tiling_cord_count('TRIHEX', 5, 5, True, rw)
    sc_ok2 = n1 == 1
    ok = ok and sc_ok2
    print("TRIHEX random+single-cord : loops=%d ok=%s note=%r"
          % (n1, sc_ok2, note))

    # 9. tiling geometry (ribbon + tube + rope) non-degenerate
    print("-- tiling / profile geometry --")
    tg_ok = True
    for sub in ('HEX', 'TRIANGLE', 'SNUBSQUARE'):
        for prof in ('FLAT', 'TUBE', 'ROPE'):
            g = _check_tiling_geometry(sub, prof)
            tg_ok = tg_ok and g
        print("tiling geometry %-11s FLAT/TUBE/ROPE : %s" % (sub, tg_ok))
    ok = ok and tg_ok

    # 9b. TUBE / ROPE profiles on the square grid (all interlace / color
    # modes) build finite, non-degenerate geometry
    pg = _check_profile_geometry()
    ok = ok and pg
    print("square tube/rope profiles non-degenerate: %s" % pg)

    # 10. SYMMETRY-GROUP BARRIERS -- the symmetrized seed is invariant
    # under every element of its point group and still a valid knot
    print("-- symmetry-group barriers --")
    for (grp, W, H) in [('C2', 6, 6), ('C4', 6, 6), ('D2', 6, 6),
                        ('D4', 6, 6), ('C4', 7, 5), ('D4', 7, 5)]:
        good, ns, ne = _check_square_symmetry(grp, W, H)
        ok = ok and good
        print("square %-3s %dx%d : invariant+valid=%s walls=%d elems=%d"
              % (grp, W, H, good, ns, ne))
    for (sub, grp) in [('HEX', 'C6'), ('HEX', 'D6'), ('TRIANGLE', 'C6'),
                       ('TRIANGLE', 'D3'), ('TRIHEX', 'C6'),
                       ('TRIHEX', 'C3'), ('SNUBSQUARE', 'C4')]:
        good, ns, ne = _check_tiling_symmetry(sub, grp, 5, 5, trim=False)
        ok = ok and good
        print("tiling %-11s %-3s : invariant+valid=%s walls=%d elems=%d"
              % (sub, grp, good, ns, ne))

    # 11. LENTICULAR (POINTED) reflection turns: same sample count, reduce
    # to ROUNDED at the limit, finite, non-degenerate, and compose with
    # both substrates
    print("-- lenticular (pointed) turns --")
    pc_ok, dev = _check_pointed_corner()
    ok = ok and pc_ok
    print("square pointed turns : reduces-to-round + non-degenerate=%s "
          "max_dev=%.4f" % (pc_ok, dev))
    pt_ok = _check_pointed_tiling()
    ok = ok and pt_ok
    print("tiling pointed turns : round-identical@t=0 + geometry ok=%s"
          % pt_ok)

    # 12. HISTORICAL PANELS -- each named panel is a valid closed-cord knot
    print("-- historical panels --")
    for name, (W, H) in _PRESET_HINTS.items():
        good, n = _check_historical_preset(name, W, H)
        ok = ok and good
        print("preset %-12s %dx%d : valid closed-cord knot=%s cords=%d"
              % (name, W, H, good, n))

    # 13. FLAT-interlace flush cut on the medial tilings (ANGULAR): the
    # under-cord endcaps run parallel to the over-cord's edge at the
    # oblique 60/120-degree crossings
    print("-- tiling angular flush-cut --")
    for sub in ('TRIHEX', 'HEX', 'TRIANGLE'):
        good, nchk, worst = _check_tiling_flush_cut(sub)
        ok = ok and good
        print("flush-cut %-11s ANGULAR : caps-parallel=%s checked=%d "
              "worst_sin=%.4f" % (sub, good, nchk, worst))

    print("RESULT:", "OK" if ok else "BAD")
