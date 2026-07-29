
# Knot Carpet Generator for Blender
#
# Rob Scharein's KNOT CARPETS (KnotPlot): "an infinitely complex link
# consisting of an infinite number of unknots", each unknot with
# n-fold symmetry about its centre.  A knot carpet is knot theory
# rendered as wallpaper -- a doubly periodic field of closed,
# individually UNKNOTTED loops, each overlapping its lattice
# neighbours, the whole field woven strictly over-and-under into an
# alternating link.  No loop is knotted on its own; the interest is
# entirely in how the infinite family interlocks.
#
# THE LATTICE.  One closed loop is centred on every node of a SQUARE
# (basis (1,0),(0,1)) or TRIANGULAR (basis (1,0),(1/2,sqrt3/2))
# lattice over an Nx x Ny block plus a one-cell margin ring, so every
# interior loop has its full ring of 4 (square) or 6 (triangular)
# nearest neighbours.  Each loop is a k-fold rosette
#     r(theta) = R (1 + amp cos k theta),
# a plain circle at amp = 0, sampled as a closed polyline; R is a
# little over half the nearest-neighbour spacing (the `overlap`
# factor), so each loop overlaps each of its nearest neighbours.
#
# CROSSINGS.  Crossing points are exact segment-segment intersections
# between the sampled polylines of neighbouring loops (nearest and
# next-nearest shells, so generous overlaps stay woven).  Each
# crossing carries ONE bit: whether the lower-id loop rides over.
# Alternation along a loop -- over, under, over, under ... in cyclic
# order -- reduces to a parity (XOR) relation between consecutive
# crossings along every loop, solved globally with the same parity
# union-find the Islamic and Celtic strapwork engines use.  A planar
# diagram always admits an alternating assignment, so the carpet
# comes out a genuine alternating link; and because the two loops at
# a crossing read opposite senses of the same bit, every crossing is
# exactly one-over-one-under by construction (even if sampling noise
# ever frustrated the alternation, the weave stays consistent).
#
# The loops then flow through the shared Pattern Engine strapwork
# machinery (pattern_common / Islamic generator) -- mitered ribbons,
# the flat cut-under interlace or the true 3D weave, relief and
# backing -- so the carpet is built as continuous ribbons.
#
# References:
#   Robert G. Scharein, "Interactive Topological Drawing" (PhD
#     thesis, The University of British Columbia, 1998) -- the
#     KnotPlot program.
#   Robert G. Scharein, KnotPlot knot carpets --
#     https://knotplot.com/carpets -- periodic alternating links of
#     symmetric unknots, the construction reproduced here.
#   George Bain, "Celtic Art: The Methods of Construction" (1951) --
#     the interlace tradition of woven closed cords.
#   Peter R. Cromwell, "Celtic Knotwork: Mathematical Art"
#     (Mathematical Intelligencer 15(1), 1993) -- the topology of
#     alternating interlace.
#   Slavik V. Jablan, "Mirror curves" and "Symmetry, Ornament and
#     Modularity" (2002) -- knotwork diagrams and their alternating
#     over/under structure.

bl_info = {
    "name": "Knot Carpet",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Knot carpets -- a tileable alternating link of "
                   "interlocked unknots (ribbon, woven rope tube, or "
                   "curve) on a square or triangular lattice",
    "category": "Add Mesh",
}

from math import pi, sqrt

import numpy as np

try:
    from . import pattern_common as pc
    from . import islamic_pattern_generator as isl
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc
    import islamic_pattern_generator as isl


# --------------------------------------------------------------------
# Lattices and loop placement
# --------------------------------------------------------------------
#
# Both lattices are normalized to nearest-neighbour spacing 1.  The
# first shell holds the true nearest neighbours (whose count is the
# coordination number the self-test checks); the second shell catches
# the extra contacts a generous overlap/amplitude can create, so no
# real crossing ever goes unwoven.

def _basis(lattice):
    """(b1, b2, first_shell, second_shell) of a lattice; the shells
    are (dm, dn) offsets in lattice coordinates."""
    if lattice == 'TRIANGULAR':
        b1 = np.array([1.0, 0.0])
        b2 = np.array([0.5, 0.5 * sqrt(3.0)])
        nb1 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
        nb2 = ((1, 1), (-1, -1), (2, -1), (-2, 1), (1, -2), (-1, 2))
    else:                                          # SQUARE
        b1 = np.array([1.0, 0.0])
        b2 = np.array([0.0, 1.0])
        nb1 = ((1, 0), (-1, 0), (0, 1), (0, -1))
        nb2 = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    return b1, b2, nb1, nb2


def loop_centers(lattice, nx, ny):
    """Loop centres over an nx x ny block plus a one-cell margin ring.
    Returns (centers, mns, idmap, interior): the centre points, their
    (m, n) lattice coordinates, the (m, n) -> id map, and the id set
    of INTERIOR loops (those with a full neighbour ring)."""
    b1, b2, _nb1, _nb2 = _basis(lattice)
    centers, mns = [], []
    idmap = {}
    for n in range(-1, ny + 1):
        for m in range(-1, nx + 1):
            idmap[(m, n)] = len(centers)
            centers.append(m * b1 + n * b2)
            mns.append((m, n))
    interior = {idmap[(m, n)] for (m, n) in idmap
                if 0 <= m < nx and 0 <= n < ny}
    return centers, mns, idmap, interior


def _rosette(center, R, k, amp, samples):
    """A k-fold rosette r = R (1 + amp cos k theta) about `center` as
    an (S, 2) closed polyline (no duplicated seam point).  amp = 0 is
    a plain circle; the lobes point along theta = 0, 2pi/k, ... --
    towards the lattice neighbours for k = 4 (square) / 6
    (triangular)."""
    th = np.linspace(0.0, 2.0 * pi, samples, endpoint=False)
    r = R * (1.0 + amp * np.cos(k * th))
    return np.column_stack([center[0] + r * np.cos(th),
                            center[1] + r * np.sin(th)])


def _sample_loop(center, R, k, amp, samples, style, subdiv):
    """The rendered polyline of one loop.  ANGULAR samples the rosette
    directly; SMOOTH runs a coarser rosette through Catmull-Rom (same
    final density), so crossings found on this path index it
    directly -- no control/subdiv bookkeeping downstream."""
    if style == 'SMOOTH' and subdiv >= 2:
        nc = max(12, int(samples) // int(subdiv))
        control = [tuple(p) for p in _rosette(center, R, k, amp, nc)]
        return np.asarray(isl.catmull_rom(control, True, subdiv), float)
    return _rosette(center, R, k, amp, samples)


def neighbour_pairs(lattice, idmap):
    """(first, second): unordered id pairs (i < j) of loops one shell
    apart -- the only pairs that can cross, so only these are tested."""
    _b1, _b2, nb1, nb2 = _basis(lattice)
    first, second = set(), set()
    for (m, n), i in idmap.items():
        for shell, out in ((nb1, first), (nb2, second)):
            for dm, dn in shell:
                j = idmap.get((m + dm, n + dn))
                if j is not None and i < j:
                    out.add((i, j))
    return sorted(first), sorted(second)


# --------------------------------------------------------------------
# Crossing detection (segment-segment, vectorized)
# --------------------------------------------------------------------

def _seg_cross(P, Q, eps=1e-12):
    """All intersection points between two closed polylines P and Q
    ((S, 2) arrays; segment i runs P[i] -> P[i+1 mod S]).  Returns
    [(fp, fq)]: the FRACTIONAL arc positions of each intersection on P
    and on Q.  Half-open segment parameters [0, 1) count each
    intersection exactly once; parallel/degenerate pairs are skipped
    (robust to tangency)."""
    A, C = P, Q
    r = np.roll(P, -1, axis=0) - P                 # (S, 2)
    s = np.roll(Q, -1, axis=0) - Q                 # (T, 2)
    denom = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    CA = C[None, :, :] - A[:, None, :]             # (S, T, 2)
    tnum = CA[:, :, 0] * s[None, :, 1] - CA[:, :, 1] * s[None, :, 0]
    unum = CA[:, :, 0] * r[:, None, 1] - CA[:, :, 1] * r[:, None, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = tnum / denom
        u = unum / denom
    hit = ((np.abs(denom) > eps)
           & (t >= 0.0) & (t < 1.0) & (u >= 0.0) & (u < 1.0))
    out = []
    for i, j in zip(*np.nonzero(hit)):
        out.append((float(i + t[i, j]), float(j + u[i, j])))
    return out


def build_carpet(lattice='SQUARE', k=4, nx=3, ny=3, amp=0.10,
                 overlap=1.15, samples=192, style='ANGULAR', subdiv=6):
    """The full combinatorial carpet.  Returns a dict:
      paths     -- one (S_i, 2) closed polyline per loop
      centers, mns, idmap, interior -- the placement (see loop_centers)
      crossings -- [(key, lo, hi, f_lo, f_hi)]: unique key, the two
                   loop ids (lo < hi) and the fractional arc position
                   of the crossing on each loop's path
      nn_pairs  -- the first-shell (nearest-neighbour) id pairs
      nn_hits   -- {pair: crossing count} over the first shell
      over      -- {key: bit}, 1 = the LOWER-id loop rides over
      per_loop  -- {loop id: [(f, key, is_hi)]} sorted along the loop
      consistent-- True if every alternation constraint was satisfied
    """
    centers, mns, idmap, interior = loop_centers(lattice, nx, ny)
    R = 0.5 * float(overlap)
    paths = [_sample_loop(c, R, k, amp, samples, style, subdiv)
             for c in centers]
    first, second = neighbour_pairs(lattice, idmap)
    first_set = set(first)
    crossings = []
    nn_hits = {}
    for (i, j) in first + second:
        hits = _seg_cross(paths[i], paths[j])
        if (i, j) in first_set:
            nn_hits[(i, j)] = len(hits)
        for (fi, fj) in hits:
            crossings.append((len(crossings), i, j, fi, fj))
    over, per_loop, consistent = _solve_over(paths, crossings)
    return dict(paths=paths, centers=centers, mns=mns, idmap=idmap,
                interior=interior, crossings=crossings,
                nn_pairs=first, nn_hits=nn_hits, over=over,
                per_loop=per_loop, consistent=consistent)


# --------------------------------------------------------------------
# Over / under assignment (alternating link, parity union-find)
# --------------------------------------------------------------------

def _solve_over(paths, crossings):
    """One bit per crossing: over[key] = 1 means the LOWER-id loop
    rides over there, so "loop i is over at c" = over[c] XOR
    (i == c.hi).  Walking loop i, consecutive crossings must flip that
    sense, i.e.  o1 XOR o2 = 1 XOR h1 XOR h2  with h the is-hi flag --
    a pure parity relation fed to the union-find.  Whatever the DSU
    returns, the two loops at a crossing read opposite senses of the
    same bit, so one-over-one-under can never break; `consistent`
    reports whether alternation itself was fully satisfiable."""
    per_loop = {}
    for key, lo, hi, flo, fhi in crossings:
        per_loop.setdefault(lo, []).append((flo, key, 0))
        per_loop.setdefault(hi, []).append((fhi, key, 1))
    dsu = isl._ParityDSU()
    consistent = True
    for ent in per_loop.values():
        ent.sort()
        C = len(ent)
        if C < 2:
            continue
        for a in range(C):
            _f1, k1, h1 = ent[a]
            _f2, k2, h2 = ent[(a + 1) % C]
            if k1 == k2:
                continue
            if not dsu.union(k1, k2, 1 ^ h1 ^ h2):
                consistent = False
    over = {}
    for key, _lo, _hi, _flo, _fhi in crossings:
        dsu.add(key)
        _root, par = dsu.find(key)
        over[key] = par
    return over, per_loop, consistent


def loop_signed(carpet):
    """{loop id: [(path_index, +1 over / -1 under)]}: each loop's
    crossings as direct indices into its own sampled path, ready for
    the strapwork machinery."""
    paths = carpet['paths']
    over = carpet['over']
    out = {i: [] for i in range(len(paths))}
    for i, ent in carpet['per_loop'].items():
        S = len(paths[i])
        for f, key, hi in sorted(ent):
            ov = over[key] ^ hi
            out[i].append((int(round(f)) % S, 1 if ov else -1))
    return out


# --------------------------------------------------------------------
# Ribbon assembly (reusing the Islamic strapwork machinery)
# --------------------------------------------------------------------

_BACKING_MAT = len(pc.PALETTE_RGBA) - 1


def _checker_pieces(path, signed):
    """Split a closed path at its crossings; each arc is colored by
    the over/under sense at the crossing it leaves from (the two-tone
    CHECKER coloring).  Returns [(subpath, mat)]."""
    n = len(path)
    if not signed:
        return [(path, 0)]
    ent = sorted({a % n: sg for a, sg in signed}.items())
    idxs = [a for a, _sg in ent]
    m = len(idxs)
    out = []
    for a in range(m):
        i0, i1 = idxs[a], idxs[(a + 1) % m]
        sp = path[i0:i1 + 1] if i1 > i0 else path[i0:] + path[:i1 + 1]
        if len(sp) >= 2:
            out.append((sp, 0 if ent[a][1] > 0 else 1))
    return out


def _loop_cell(path, signed, width, interlace, mode, weave_height,
               height, color_by, loop_index):
    """Sub-cells (verts, faces, mats) for one closed loop rendered as
    a mitered ribbon.  `signed` is [(path_index, +1 over / -1 under)]
    at the loop's crossings -- direct indices into `path`."""
    if len(path) < 3:
        return []

    def matof(default):
        if color_by == 'LOOP':
            return loop_index % len(pc.PALETTE_RGBA)
        if color_by == 'UNIFORM':
            return 0
        return default

    sub_cells = []

    if color_by == 'CHECKER':
        # two-tone by over/under: cut at every crossing, color each
        # arc by the sense it leaves its crossing with
        for sp, mat in _checker_pieces(path, signed):
            left, right = isl.miter_ribbon(sp, width, False)
            cv, cf = isl.band_ribbon_faces(left, right, False, height)
            if cf:
                sub_cells.append((cv, cf, [mat] * len(cf)))
    elif interlace and mode == 'WOVEN':
        zoff = isl._weave_zoff(path, True, signed, weave_height)
        left, right = isl.miter_ribbon(path, width, True)
        cv, cf = isl.band_ribbon_faces_z(left, right, True, height,
                                         zoff)
        if cf:
            sub_cells.append((cv, cf, [matof(0)] * len(cf)))
    elif interlace and mode == 'FLAT':
        under = [ci for ci, sg in signed if sg < 0]
        if under:
            s, total = isl._arclen(path, True)
            cut_s = sorted(s[ci] for ci in under)
            # the gap must clear the over strand even at the shallow
            # crossing angles a gentle overlap produces
            margin = max(0.02, 0.75 * width)
            half = 0.5 * (width + margin)
            pieces = isl._cut_band(path, True, cut_s, half, s, total)
        else:
            pieces = [(path, True)]
        for sp, sp_closed in pieces:
            if len(sp) < 2:
                continue
            left, right = isl.miter_ribbon(sp, width, sp_closed)
            cv, cf = isl.band_ribbon_faces(left, right, sp_closed,
                                           height)
            if cf:
                sub_cells.append((cv, cf, [matof(0)] * len(cf)))
    else:                                     # plain flat ribbon
        left, right = isl.miter_ribbon(path, width, True)
        cv, cf = isl.band_ribbon_faces(left, right, True, height)
        if cf:
            sub_cells.append((cv, cf, [matof(0)] * len(cf)))

    return sub_cells


def build_cells(lattice='SQUARE', k=4, nx=3, ny=3, amp=0.10,
                overlap=1.15, samples=192, cord_width=0.12,
                style='ANGULAR', subdiv=6, interlace=True,
                interlace_mode='FLAT', weave_height=0.05,
                color_by='LOOP', height=0.0, backing=False, base=0.06):
    """One merged (verts, faces, mats) cell per loop, plus an optional
    backing slab.  `cord_width` is in loop-spacing units (the
    nearest-neighbour distance is 1)."""
    carpet = build_carpet(lattice, k, nx, ny, amp, overlap, samples,
                          style, subdiv)
    signed = loop_signed(carpet)
    width = max(0.01, float(cord_width))
    cells = []
    all_verts = []
    for i, path in enumerate(carpet['paths']):
        pl = [tuple(p) for p in path]
        sub = _loop_cell(pl, signed[i], width, interlace,
                         interlace_mode, weave_height, height,
                         color_by, i)
        if not sub:
            continue
        cell = pc.merge_cells(sub)
        if not cell[1]:
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


def loop_paths(lattice='SQUARE', k=4, nx=3, ny=3, amp=0.10,
               overlap=1.15, samples=192, style='ANGULAR', subdiv=6):
    """Loop centerlines [(points, True)] for the CURVE output (every
    loop is a single closed spline)."""
    centers, _mns, _idmap, _interior = loop_centers(lattice, nx, ny)
    R = 0.5 * float(overlap)
    return [([tuple(p) for p in _sample_loop(c, R, k, amp, samples,
                                             style, subdiv)], True)
            for c in centers]


# --------------------------------------------------------------------
# Tube assembly (round rope, woven over/under)
# --------------------------------------------------------------------

def _tube_welded(pts, radius, sides):
    """Sweep a round closed tube along the 3D centerline `pts` with
    rotation-minimizing frames, and join the rings by PROXIMITY.

    A closed RMF sweep leaves a residual twist at the seam (the frame's
    holonomy), so a naive vertex-s -> vertex-s closure connects rotated
    points and pinches the loop shut.  Two things prevent that here: the
    holonomy is measured (atan2, full range) and spread evenly along the
    loop, and then EACH ring is connected to the next by the whole-ring
    rotation that best aligns them (nearest-vertex weld), so any leftover
    fractional twist lands on the seam as a clean shift rather than a
    fold.  Returns (verts, faces) with quad faces."""
    P = np.asarray(pts, float)
    n = len(P)
    if n < 3:
        return [], []
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12

    def _reflect_step(a, b, na, ta, tb):
        """Double-reflection transport of normal `na` from a to b."""
        v1 = b - a
        c1 = float(v1 @ v1) + 1e-12
        rl = na - (2.0 / c1) * float(v1 @ na) * v1
        tl = ta - (2.0 / c1) * float(v1 @ ta) * v1
        v2 = tb - tl
        c2 = float(v2 @ v2) + 1e-12
        return rl - (2.0 / c2) * float(v2 @ rl) * v2

    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    n0 = np.cross(T[0], ref)
    n0 /= np.linalg.norm(n0) + 1e-12
    N = [n0]
    for i in range(n - 1):
        nn = _reflect_step(P[i], P[i + 1], N[-1], T[i], T[i + 1])
        N.append(nn / (np.linalg.norm(nn) + 1e-12))
    N = np.array(N)
    # holonomy: transport once more across the closing edge, compare to N[0]
    nw = _reflect_step(P[n - 1], P[0], N[n - 1], T[n - 1], T[0])
    nw /= np.linalg.norm(nw) + 1e-12
    b0 = np.cross(T[0], N[0])
    theta = np.arctan2(float(nw @ b0), float(nw @ N[0]))

    ring0 = 2.0 * pi * np.arange(sides) / sides
    ca0, sa0 = np.cos(ring0), np.sin(ring0)
    verts, bases = [], []
    for i in range(n):
        corr = -theta * i / n
        B = np.cross(T[i], N[i])
        Nc = N[i] * np.cos(corr) + B * np.sin(corr)
        Bc = -N[i] * np.sin(corr) + B * np.cos(corr)
        ring = P[i] + radius * (ca0[:, None] * Nc + sa0[:, None] * Bc)
        bases.append(len(verts))
        verts.extend(map(tuple, ring))
    faces = []
    for i in range(n):
        j = (i + 1) % n
        Ri = np.asarray(verts[bases[i]:bases[i] + sides])
        Rj = np.asarray(verts[bases[j]:bases[j] + sides])
        # whole-ring rotation of ring j that best aligns it to ring i
        shift = min(range(sides),
                    key=lambda s: float(np.sum(
                        (Ri - np.roll(Rj, -s, axis=0)) ** 2)))
        for s in range(sides):
            s2 = (s + 1) % sides
            faces.append([bases[i] + s, bases[i] + s2,
                          bases[j] + (s2 + shift) % sides,
                          bases[j] + (s + shift) % sides])
    return verts, faces


def build_tube_cells(lattice='SQUARE', k=4, nx=3, ny=3, amp=0.10,
                     overlap=1.15, samples=192, style='ANGULAR',
                     subdiv=6, tube_radius=0.04, tube_sides=10,
                     weave_height=0.06, color_by='LOOP', backing=False,
                     base=0.06):
    """One closed round tube per loop, each dipping in z at its
    crossings so the carpet reads as woven rope.  The weave amplitude
    is forced to at least clear the tube diameter, so an over strand
    never intersects the under strand it passes.  Returns one
    (verts, faces, mats) cell per loop (+ an optional backing slab)."""
    carpet = build_carpet(lattice, k, nx, ny, amp, overlap, samples,
                          style, subdiv)
    signed = loop_signed(carpet)
    tr = max(0.005, float(tube_radius))
    # the over/under lift must exceed the tube radius or the ropes touch
    lift = max(float(weave_height), 1.3 * tr)
    sides = max(3, int(tube_sides))
    cells = []
    all_verts = []
    for i, path in enumerate(carpet['paths']):
        pl2 = [(float(p[0]), float(p[1])) for p in path]
        zoff = isl._weave_zoff(pl2, True, signed[i], lift)
        path3d = [(pl2[j][0], pl2[j][1], zoff[j]) for j in range(len(pl2))]
        verts, faces = _tube_welded(path3d, tr, sides)
        if not faces:
            continue
        mat = (i % len(pc.PALETTE_RGBA)) if color_by == 'LOOP' else 0
        cells.append((verts, faces, [mat] * len(faces)))
        all_verts.extend(verts)
    if backing and all_verts:
        a = np.asarray(all_verts, float)
        lo = (a[:, 0].min(), a[:, 1].min())
        hi = (a[:, 0].max(), a[:, 1].max())
        cv, cf, cm = [], [], []
        pc.slab(cv, cf, cm, lo, hi, -tr, -tr - base, mat=_BACKING_MAT)
        cells.append((cv, cf, cm))
    return cells


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

    def _shade_smooth(obj):
        """Smooth-shade a tube object (or every mesh child, when the
        loops are separated) so the round rope reads as curved rather
        than faceted."""
        if obj is None:
            return
        if obj.type == 'MESH':
            for p in obj.data.polygons:
                p.use_smooth = True
            obj.data.update()
        for child in obj.children:
            _shade_smooth(child)

    def _emit_curve(context, name, paths, span=2.0, operator=None):
        """Build a curve object whose cyclic POLY splines are the loop
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
                sp.points[i].co = ((px - cx) * s, (py - cy) * s, 0.0,
                                   1.0)
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

    class MESH_OT_knot_carpet_add(bpy.types.Operator, AddObjectHelper):
        """Add a knot carpet (tileable alternating link of unknots)"""
        bl_idname = "mesh.knot_carpet_add"
        bl_label = "Knot Carpet"
        bl_options = {'REGISTER', 'UNDO'}

        lattice: EnumProperty(
            name="Lattice",
            items=[('SQUARE', "Square",
                    "Loops on a square lattice (4 neighbours)"),
                   ('TRIANGULAR', "Triangular",
                    "Loops on a triangular lattice (6 neighbours)")],
            default='SQUARE')
        symmetry: IntProperty(
            name="Symmetry", default=4, min=2, max=12,
            description="k-fold symmetry of each loop (rosette lobes; "
                        "4 suits the square lattice, 3 or 6 the "
                        "triangular)")
        nx: IntProperty(name="Tiles X", default=3, min=1, max=12)
        ny: IntProperty(name="Tiles Y", default=3, min=1, max=12)
        amplitude: FloatProperty(
            name="Rosette Amplitude", default=0.10, min=0.0, max=0.35,
            description="Lobe depth of the k-fold rosette; 0 = plain "
                        "circles")
        overlap: FloatProperty(
            name="Overlap", default=1.15, min=1.02, max=1.35,
            description="Loop radius as a multiple of half the "
                        "neighbour spacing (> 1 so loops interlock)")
        samples: IntProperty(
            name="Samples", default=192, min=48, max=512,
            description="Polyline samples per loop")
        cord_width: FloatProperty(
            name="Cord Width", default=0.12, min=0.02, max=0.4,
            description="Ribbon width in loop-spacing units")
        style: EnumProperty(
            name="Style",
            items=[('ANGULAR', "Angular",
                    "The sampled rosette polyline as-is"),
                   ('SMOOTH', "Smooth",
                    "Catmull-Rom resampling of a coarser rosette")],
            default='ANGULAR')
        smoothness: IntProperty(
            name="Smoothness", default=6, min=2, max=16,
            description="Spline subdivisions per control segment "
                        "(smooth)")
        interlace: BoolProperty(
            name="Interlace (Weave)", default=True,
            description="Weave the loops over and under (alternating "
                        "link)")
        interlace_mode: EnumProperty(
            name="Interlace Mode",
            items=[('FLAT', "Flat Knotwork",
                    "Break the under-loop at crossings (2D "
                    "interlace)"),
                   ('WOVEN', "Woven (3D)",
                    "Raise/dip each loop into a 3D woven surface")],
            default='FLAT')
        weave_height: FloatProperty(
            name="Weave Height", default=0.05, min=0.0, max=0.4,
            description="Z amplitude of the woven loops (woven only)")
        color_by: EnumProperty(
            name="Color By",
            items=[('UNIFORM', "Uniform", "A single material"),
                   ('LOOP', "By Loop",
                    "A distinct color per loop"),
                   ('CHECKER', "Over/Under",
                    "Two-tone by the over/under weave")],
            default='LOOP')
        output: EnumProperty(
            name="Output",
            items=[('RIBBON', "Ribbon Mesh",
                    "Filled loop ribbons (supports relief)"),
                   ('TUBE', "Tube (Rope)",
                    "Round 3D tubes woven over and under, like rope"),
                   ('CURVE', "Centerline Curves",
                    "Loop centerlines as a Blender curve object")],
            default='RIBBON')
        tube_radius: FloatProperty(
            name="Tube Radius", default=0.04, min=0.005, max=0.25,
            description="Rope radius in loop-spacing units (tube)")
        tube_sides: IntProperty(
            name="Tube Sides", default=10, min=3, max=32,
            description="Cross-section segments of each tube")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=1.0,
            description="0 = flat ribbons; > 0 extrudes the loops")
        backing: BoolProperty(
            name="Backing Slab", default=False,
            description="Add a slab behind the carpet")
        base: FloatProperty(
            name="Base Thickness", default=0.06, min=0.01, max=0.5)
        separate: BoolProperty(
            name="Separate Loops", default=False,
            description="Output each loop as its own object")

        def execute(self, context):
            if self.output == 'CURVE':
                paths = loop_paths(self.lattice, self.symmetry,
                                   self.nx, self.ny, self.amplitude,
                                   self.overlap, self.samples,
                                   self.style, self.smoothness)
                obj = _emit_curve(context, "Knot Carpet", paths,
                                  operator=self)
                if obj is None:
                    self.report({'ERROR'}, "no carpet generated")
                    return {'CANCELLED'}
                obj["math_art_pattern"] = True
                self.report({'INFO'}, "%s %dx%d  %d loops" %
                            (self.lattice, self.nx, self.ny,
                             len(paths)))
                return {'FINISHED'}
            if self.output == 'TUBE':
                cells = build_tube_cells(
                    self.lattice, self.symmetry, self.nx, self.ny,
                    self.amplitude, self.overlap, self.samples,
                    self.style, self.smoothness, self.tube_radius,
                    self.tube_sides, self.weave_height, self.color_by,
                    self.backing, self.base)
            else:
                cells = build_cells(
                self.lattice, self.symmetry, self.nx, self.ny,
                self.amplitude, self.overlap, self.samples,
                self.cord_width, self.style, self.smoothness,
                self.interlace, self.interlace_mode,
                self.weave_height, self.color_by, self.height,
                self.backing, self.base)
            obj = pc.emit(context, "Knot Carpet", cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no carpet generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if self.output == 'TUBE':
                _shade_smooth(obj)
            n_loops = len(cells) - (1 if self.backing else 0)
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s %dx%d  %d loops  V=%d F=%d"
                            % (self.lattice, self.nx, self.ny,
                               n_loops, len(obj.data.vertices),
                               len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s %dx%d  %d loops" %
                            (self.lattice, self.nx, self.ny,
                             len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'lattice')
            lay.prop(self, 'symmetry')
            lay.prop(self, 'nx')
            lay.prop(self, 'ny')
            lay.prop(self, 'amplitude')
            lay.prop(self, 'overlap')
            lay.prop(self, 'samples')
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
            elif self.output == 'TUBE':
                lay.prop(self, 'tube_radius')
                lay.prop(self, 'tube_sides')
                lay.prop(self, 'weave_height')
                lay.prop(self, 'color_by')
                lay.prop(self, 'backing')
                if self.backing:
                    lay.prop(self, 'base')
                lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.knot_carpet_add",
                             icon='MESH_CIRCLE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_knot_carpet_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_knot_carpet_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

_EXPECT_DEG = {'SQUARE': 4, 'TRIANGULAR': 6}


def _check_neighbours(carpet, lattice):
    """Every interior loop has the lattice coordination number of
    nearest neighbours, and every nearest-neighbour pair crosses at
    least twice."""
    deg = {}
    for (i, j) in carpet['nn_pairs']:
        deg[i] = deg.get(i, 0) + 1
        deg[j] = deg.get(j, 0) + 1
    want = _EXPECT_DEG[lattice]
    deg_ok = all(deg.get(i, 0) == want for i in carpet['interior'])
    min_hits = min(carpet['nn_hits'].values())
    return deg_ok, min_hits


def _check_one_over(carpet):
    """Every crossing is met by exactly two loops with opposite
    over/under senses -- must ALWAYS hold, frustrated or not."""
    over = carpet['over']
    senses = {}
    for i, ent in carpet['per_loop'].items():
        for _f, key, hi in ent:
            senses.setdefault(key, []).append(over[key] ^ hi)
    for key, _lo, _hi, _flo, _fhi in carpet['crossings']:
        sv = senses.get(key, [])
        if len(sv) != 2 or sv[0] == sv[1]:
            return False
    return True


def _check_alternation(carpet):
    """Along every interior loop, consecutive crossings alternate
    over/under (the alternating-link property)."""
    over = carpet['over']
    for i in carpet['interior']:
        ent = sorted(carpet['per_loop'].get(i, []))
        C = len(ent)
        if C < 2:
            return False
        sv = [over[key] ^ hi for _f, key, hi in ent]
        for a in range(C):
            if sv[a] == sv[(a + 1) % C]:
                return False
    return True


def _check_geometry(lattice, k, mode, color_by):
    """Built ribbons are non-empty with finite vertices."""
    cells = build_cells(lattice, k, 2, 2, amp=0.10, overlap=1.15,
                        samples=120, interlace=True,
                        interlace_mode=mode, color_by=color_by,
                        weave_height=0.05, height=0.04)
    faces = sum(len(c[1]) for c in cells)
    finite = all(all(np.isfinite(v).all() for v in c[0])
                 for c in cells)
    return faces > 0 and finite


def _check_tube(lattice, k):
    """Woven round tubes build non-empty, finite, watertight-ish
    geometry (every face a quad)."""
    cells = build_tube_cells(lattice, k, 2, 2, amp=0.10, overlap=1.15,
                             samples=120, tube_radius=0.04,
                             tube_sides=8, weave_height=0.06)
    faces = sum(len(c[1]) for c in cells)
    finite = all(all(np.isfinite(v).all() for v in c[0])
                 for c in cells)
    quads = all(len(f) == 4 for c in cells for f in c[1])
    return faces > 0 and finite and quads


def _check_periodic(lattice, nx, ny, tol=1e-9):
    """The interior loop centres are periodic under both lattice basis
    vectors (translates land on placed centres) -- the carpet tiles."""
    b1, b2, _nb1, _nb2 = _basis(lattice)
    centers, _mns, _idmap, interior = loop_centers(lattice, nx, ny)
    have = {(round(c[0], 6), round(c[1], 6)) for c in centers}
    for i in interior:
        for b in (b1, b2):
            t = centers[i] + b
            if (round(t[0], 6), round(t[1], 6)) not in have:
                return False
    return True


if __name__ == "__main__":
    ok = True

    # 1. neighbour counts and >= 2 crossings per neighbour pair
    for lattice, k in (('SQUARE', 4), ('TRIANGULAR', 6)):
        for amp in (0.0, 0.12):
            car = build_carpet(lattice, k, 3, 3, amp, 1.15, 160)
            deg_ok, min_hits = _check_neighbours(car, lattice)
            good = deg_ok and min_hits >= 2
            ok = ok and good
            print("neighbours %-10s k=%d amp=%.2f : degree=%s "
                  "min-crossings=%d" % (lattice, k, amp, deg_ok,
                                        min_hits))

            # 2. one-over-one-under at EVERY crossing (always)
            one = _check_one_over(car)
            ok = ok and one
            print("one-over   %-10s k=%d amp=%.2f : %s  (%d "
                  "crossings, consistent=%s)"
                  % (lattice, k, amp, one, len(car['crossings']),
                     car['consistent']))

    # 3. alternation along every interior loop (plain square carpet)
    car = build_carpet('SQUARE', 4, 3, 3, 0.0, 1.15, 160)
    alt = _check_alternation(car) and car['consistent']
    ok = ok and alt
    print("alternation SQUARE amp=0 : %s" % alt)

    # 4. ribbon geometry finite and non-empty across the options
    for lattice, k, mode, cby in (('SQUARE', 4, 'FLAT', 'LOOP'),
                                  ('SQUARE', 4, 'WOVEN', 'UNIFORM'),
                                  ('TRIANGULAR', 6, 'FLAT', 'CHECKER'),
                                  ('TRIANGULAR', 3, 'WOVEN', 'LOOP')):
        g = _check_geometry(lattice, k, mode, cby)
        ok = ok and g
        print("geometry %-10s k=%d %-5s %-7s : %s"
              % (lattice, k, mode, cby, g))

    # 5. tileability: interior centres periodic under the basis
    for lattice in ('SQUARE', 'TRIANGULAR'):
        per = _check_periodic(lattice, 3, 3)
        ok = ok and per
        print("periodic %-10s : %s" % (lattice, per))

    # 6. woven round tubes build valid geometry
    for lattice, k in (('SQUARE', 4), ('TRIANGULAR', 6)):
        t = _check_tube(lattice, k)
        ok = ok and t
        print("tube     %-10s k=%d : %s" % (lattice, k, t))

    print("RESULT:", "OK" if ok else "BAD")
