
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

from math import gcd, hypot
import random as _random

import numpy as np

try:
    from . import pattern_common as pc
    from . import islamic_pattern_generator as isl
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc
    import islamic_pattern_generator as isl


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


def _preset_barriers(name, W, H):
    """Internal barriers for a named preset panel."""
    B = set()
    if name == 'PLAIN':
        return B
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


def _procedural_barriers(W, H, spacing, verticals, horizontals):
    """Internal walls on a regular `spacing` (in cells), optionally
    vertical and/or horizontal."""
    B = set()
    sp = max(1, int(spacing))
    if verticals:
        for k in range(1, W):
            if k % sp == 0:
                i = 2 * k
                for j in range(1, 2 * H, 2):
                    B.add((i, j))
    if horizontals:
        for k in range(1, H):
            if k % sp == 0:
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
    """(closed, control_pts, cross): the control polyline of one cord and
    its crossings as (control_index, over_sign) with +1 over, -1 under."""
    control = [(float(pos[0]), float(pos[1])) for pos, _b, _d in rec]
    cross = []
    for k, (pos, isb, d) in enumerate(rec):
        if isb:
            continue
        fam = _family(d)
        over = x.get(pos, 0) ^ fam           # cord rides over here?
        cross.append((k, 1 if over else -1))
    return control, cross


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


def _cord_cell(rec, x, width, style, subdiv, interlace, mode,
               weave_height, height, color_by, loop_index):
    """Build one merged (verts, faces, mats) cell for a single cord."""
    control, cross = _cord_signed(rec, x)
    if len(control) < 2:
        return None
    step = subdiv if style == 'SMOOTH' else 1
    path = (isl.catmull_rom(control, True, subdiv)
            if style == 'SMOOTH' else control)
    signed = [(k * step, sg) for k, sg in cross]

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
        idxs = [k * step for k, _sg in cross]
        for sp, par in _split_at(path, True, idxs):
            left, right = isl.miter_ribbon(sp, width, False)
            cv, cf = isl.band_ribbon_faces(left, right, False, height)
            if cf:
                sub_cells.append((cv, cf, [par] * len(cf)))
    elif interlace and mode == 'WOVEN':
        zoff = isl._weave_zoff(path, True, signed, weave_height)
        left, right = isl.miter_ribbon(path, width, True)
        cv, cf = isl.band_ribbon_faces_z(left, right, True, height, zoff)
        if cf:
            sub_cells.append((cv, cf, [matof(0)] * len(cf)))
    elif interlace and mode == 'FLAT':
        under = [pi for pi, sg in signed if sg < 0]
        if under:
            s, total = isl._arclen(path, True)
            cut_s = sorted(s[pi] for pi in under)
            margin = max(0.02, 0.25 * width)
            half = 0.5 * (width + margin)
            pieces = isl._cut_band(path, True, cut_s, half, s, total)
        else:
            pieces = [(path, True)]
        for sp, sp_closed in pieces:
            if len(sp) < 2:
                continue
            left, right = isl.miter_ribbon(sp, width, sp_closed)
            cv, cf = isl.band_ribbon_faces(left, right, sp_closed, height)
            if cf:
                sub_cells.append((cv, cf, [matof(0)] * len(cf)))
    else:                                     # plain flat ribbon
        left, right = isl.miter_ribbon(path, width, True)
        cv, cf = isl.band_ribbon_faces(left, right, True, height)
        if cf:
            sub_cells.append((cv, cf, [matof(0)] * len(cf)))

    if not sub_cells:
        return None
    return pc.merge_cells(sub_cells)


def build_cells(W, H, border, barriers, cord_width=0.25, style='ANGULAR',
                interlace=True, interlace_mode='FLAT', weave_height=0.06,
                color_by='CHECKER', height=0.0, backing=False, base=0.08,
                subdiv=8):
    """One merged (verts, faces, mats) cell per cord, plus an optional
    backing slab.  `cord_width` is a fraction of a cell (2 lattice
    units)."""
    cords = trace_cords(W, H, border, barriers)
    x = _solve_over(cords)
    width = max(0.02, cord_width * 2.0)
    cells = []
    all_verts = []
    for li, rec in enumerate(cords):
        cell = _cord_cell(rec, x, width, style, subdiv, interlace,
                          interlace_mode, weave_height, height,
                          color_by, li)
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


def cord_paths(W, H, border, barriers, style='ANGULAR', subdiv=8):
    """Cord centerlines (control polyline, closed) for the CURVE
    output."""
    cords = trace_cords(W, H, border, barriers)
    out = []
    for rec in cords:
        control = [(float(p[0]), float(p[1])) for p, _b, _d in rec]
        if len(control) < 2:
            continue
        path = (isl.catmull_rom(control, True, subdiv)
                if style == 'SMOOTH' else control)
        out.append((path, True))
    return out


def resolve_barriers(source, W, H, border, preset='PLAIN', wall_spacing=2,
                     walls_v=True, walls_h=True, seed=0, density=0.25,
                     single_cord=False):
    """The internal barrier set for a chosen source, applying the
    single-cord search when requested.  Returns (barriers, note)."""
    if source == 'PRESET':
        B = _preset_barriers(preset, W, H)
    elif source == 'PROCEDURAL':
        B = _procedural_barriers(W, H, wall_spacing, walls_v, walls_h)
    else:                                     # RANDOM
        B = _random_barriers(W, H, seed, density)
    note = ""
    if source == 'RANDOM' and single_cord:
        B, ok = make_single_cord(W, H, border, B)
        if not ok:
            note = "single-cord search did not reach 1 loop"
    return B, note


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
]


if _IN_BLENDER:

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

        grid_w: IntProperty(name="Grid Width", default=5, min=1, max=40)
        grid_h: IntProperty(name="Grid Height", default=5, min=1, max=40)
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
        cord_width: FloatProperty(
            name="Cord Width", default=0.25, min=0.02, max=0.6,
            description="Cord ribbon width as a fraction of a cell")
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
            barriers, note = resolve_barriers(
                self.barriers, W, H, self.border, self.preset,
                self.wall_spacing, self.walls_v, self.walls_h,
                self.seed, self.density, self.single_cord)
            if note:
                self.report({'WARNING'}, note)
            if self.output == 'CURVE':
                paths = cord_paths(W, H, self.border, barriers,
                                   self.style, self.smoothness)
                obj = _emit_curve(context, "Celtic Knot 2D", paths,
                                  operator=self)
                if obj is None:
                    self.report({'ERROR'}, "no knot generated")
                    return {'CANCELLED'}
                obj["math_art_pattern"] = True
                self.report({'INFO'}, "%dx%d  %d cords" %
                            (W, H, len(paths)))
                return {'FINISHED'}
            cells = build_cells(
                W, H, self.border, barriers, self.cord_width,
                self.style, self.interlace, self.interlace_mode,
                self.weave_height, self.color_by, self.height,
                self.backing, self.base, self.smoothness)
            obj = pc.emit(context, "Celtic Knot 2D", cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no knot generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            n_cords = count_loops(W, H, self.border, barriers)
            if obj.type == 'MESH':
                self.report({'INFO'}, "%dx%d  %d cords  V=%d F=%d" %
                            (W, H, n_cords, len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%dx%d  %d cords" %
                            (W, H, len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'grid_w')
            lay.prop(self, 'grid_h')
            lay.prop(self, 'border')
            lay.prop(self, 'barriers')
            if self.barriers == 'PRESET':
                lay.prop(self, 'preset')
            elif self.barriers == 'PROCEDURAL':
                lay.prop(self, 'wall_spacing')
                lay.prop(self, 'walls_v')
                lay.prop(self, 'walls_h')
            else:                              # RANDOM
                lay.prop(self, 'seed')
                lay.prop(self, 'density')
                lay.prop(self, 'single_cord')
            lay.prop(self, 'cord_width')
            lay.prop(self, 'style')
            if self.style == 'SMOOTH':
                lay.prop(self, 'smoothness')
            lay.prop(self, 'output')
            if self.output == 'RIBBON':
                lay.prop(self, 'interlace')
                if self.interlace:
                    lay.prop(self, 'interlace_mode')
                    if self.interlace_mode == 'WOVEN':
                        lay.prop(self, 'weave_height')
                lay.prop(self, 'color_by')
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
        signs = [sg for _k, sg in cross]
        L = len(signs)
        for a in range(L):
            if signs[a] == signs[(a + 1) % L]:
                alternates = False
        for k, sg in cross:
            pos = rec[k][0]
            visits.setdefault(pos, []).append(sg > 0)
    one_over = True
    for pos, ov in visits.items():
        if len(ov) != 2 or ov[0] == ov[1]:
            one_over = False
            break
    return alternates, one_over, len(visits)


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

    print("RESULT:", "OK" if ok else "BAD")
