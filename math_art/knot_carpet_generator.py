
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
# backing -- so the carpet is built as continuous ribbons.  A third
# scaffold closes the carpet over a SPHERE as a finite woven knot
# ball (see the knot-ball section below), and a fourth generalizes
# the lattice to any uniform TILING -- one medallion per tile of a
# regular, Archimedean or Laves tiling (the tiling-carpet section).
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
                   "curve) on a square or triangular lattice, on any "
                   "regular or Archimedean tiling, or closed over a "
                   "geodesic sphere as a knot ball",
    "category": "Add Mesh",
}

from math import pi, sqrt

import numpy as np

try:
    from . import pattern_common as pc
    from . import islamic_pattern_generator as isl
    from . import tiling_generator as tg
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc
    import islamic_pattern_generator as isl
    import tiling_generator as tg


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
                     base=0.06, relax_iters=0, clearance=0.10,
                     weave_gap=0.10):
    """One closed round tube per loop, each dipping in z at its
    crossings so the carpet reads as woven rope.  The weave amplitude
    is forced to at least clear the tube diameter, so an over strand
    never intersects the under strand it passes.  With relax_iters > 0
    the tubes are swept along the tier-2 relaxed centerlines instead
    of the flat weave path; clearance and weave_gap are floored at the
    tube diameter so the relaxed ropes cannot touch.  Returns one
    (verts, faces, mats) cell per loop (+ an optional backing slab)."""
    carpet = build_carpet(lattice, k, nx, ny, amp, overlap, samples,
                          style, subdiv)
    signed = loop_signed(carpet)
    tr = max(0.005, float(tube_radius))
    # the over/under lift must exceed the tube radius or the ropes touch
    lift = max(float(weave_height), 1.3 * tr)
    sides = max(3, int(tube_sides))
    relaxed = None
    if relax_iters > 0:
        relaxed = relax_carpet(carpet, lattice, nx, ny,
                               iters=int(relax_iters),
                               clearance=max(float(clearance), 2.2 * tr),
                               weave_gap=max(float(weave_gap), 2.2 * tr))
    cells = []
    all_verts = []
    for i, path in enumerate(carpet['paths']):
        if relaxed is not None:
            path3d = [tuple(p) for p in relaxed[i]]
        else:
            pl2 = [(float(p[0]), float(p[1])) for p in path]
            zoff = isl._weave_zoff(pl2, True, signed[i], lift)
            path3d = [(pl2[j][0], pl2[j][1], zoff[j])
                      for j in range(len(pl2))]
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
# Tier-2 relaxation: the physically relaxed rope carpet
# --------------------------------------------------------------------
#
# Everything above is a FLAT diagram given fake depth: crossings are
# found in 2D, over/under is solved combinatorially, and the only 3D
# is a local z-bump at each crossing -- so at steep crossings the rope
# thins and pinches, because the drawing is not a physical link.  The
# relaxation below keeps that combinatorial diagram as the SEED
# (topology and over/under), lifts each loop to a 3D bead polygon,
# and relaxes the whole field under KnotPlot's dynamical model
# (Scharein, thesis ch. 7): knots are polygons of beads and sticks
# under two forces -- a mechanical spring between adjacent beads,
# F_m = H r^(1+alpha), and an electrical repulsion between all
# non-adjacent bead pairs, F_e = K r^-(2+beta) with beta ~ 4-6 (a
# plain 1/r^2 repulsion is deliberately avoided: it does not
# discourage self-crossing, higher exponents do) -- integrated
# inertialessly ("critically damped", F proportional to v), each bead
# moving at most d_max per step so strands cannot jump through one
# another.  The repulsion-with-cutoff below is the discrete cousin of
# the knot energies KnotPlot minimizes: Simon's minimum-distance
# energy and Buck's symmetric ("radiating tube") energy.
#
# THE TORUS.  So the relaxed patch still TILES, the carpet lives on a
# torus with period vectors nx*b1 and ny*b2: only the nx x ny
# fundamental loops (the `interior` set) are simulated, every
# inter-loop force is computed over the periodic images (the
# minimum-image convention of molecular dynamics), and the margin
# loops are reconstructed afterwards as exact lattice translates of
# their wrapped partners.  Periodicity is therefore exact by
# construction, not approximate.
#
# References:
#   Robert G. Scharein, "Interactive Topological Drawing" (PhD
#     thesis, The University of British Columbia, 1998), ch. 7 -- the
#     KnotPlot dynamical model: bead/stick forces, critical damping,
#     the d_max step bound, knot-type-preserving relaxation.
#   J. K. Simon, "Energy functions for polygonal knots", J. Knot
#     Theory Ramifications 3(3):299-320, 1994 -- the minimum-distance
#     energy.
#   G. Buck & J. Orloff, Topology Appl. 51 (1993) and 61 (1995) --
#     the symmetric ("radiating tube") energy.


def _fund_map(carpet, lattice, nx, ny):
    """{loop id: (fundamental id, 2D lattice shift)}: every patch loop
    wrapped into the nx x ny fundamental block.  All loops are the
    same rosette translated, so bead j of a margin loop is bead j of
    its fundamental partner plus the shift."""
    b1, b2, _nb1, _nb2 = _basis(lattice)
    fund = {}
    for (m, n), i in carpet['idmap'].items():
        mf, nf = m % nx, n % ny
        j = carpet['idmap'][(mf, nf)]
        fund[i] = (j, (m - mf) * b1 + (n - nf) * b2)
    return fund


def _torus_crossings(carpet, fundmap, fidx):
    """The crossing list wrapped onto the torus and deduplicated.
    Returns (cons, conflicts): cons = [((la, ba), (lb, bb))] meaning
    bead ba of fundamental loop la passes OVER bead bb of loop lb
    (indices into the fundamental loop order `fidx`); conflicts counts
    patch crossings whose wrapped duplicates disagreed on the over
    bit (dropped -- zero for a periodic alternating solution)."""
    S = len(carpet['paths'][0])
    over = carpet['over']
    cons, bad = {}, set()
    for key, lo, hi, flo, fhi in carpet['crossings']:
        a = (fidx[fundmap[lo][0]], int(round(flo)) % S)
        b = (fidx[fundmap[hi][0]], int(round(fhi)) % S)
        o, u = (a, b) if over[key] else (b, a)
        ck = (min(o, u), max(o, u))
        if ck in cons and cons[ck] != (o, u):
            bad.add(ck)
        cons.setdefault(ck, (o, u))
    for ck in bad:
        del cons[ck]
    return sorted(cons.values()), len(bad)


def relax_carpet(carpet, lattice, nx, ny, iters=120, clearance=0.10,
                 weave_gap=0.10, flatten=0.5, step=0.15, trace=None):
    """Physically relax the carpet into a 3D rope field.  Returns one
    (S, 3) centerline per patch loop (same order as carpet['paths']);
    margin loops are exact translates of their wrapped fundamental
    partners, so the result tiles under nx*b1 / ny*b2.

    Per iteration (damped Euler, all deterministic): rest-length edge
    springs + Laplacian fairing keep each loop smooth and its beads
    evenly spaced; a soft 1/r^4 repulsion with cutoff pushes beads of
    different strands (min-image over the torus) apart to about
    `clearance`; at every crossing the over bead is kept at least
    `weave_gap` above the under bead; and z is pulled toward 0 with
    strength `flatten` away from crossings so the sheet stays a
    carpet.  Each bead moves at most a fraction of `clearance` per
    step (KnotPlot's d_max), which also discourages strand
    pass-through.  If `trace` is a list, the mean bead movement of
    each iteration is appended (for convergence checks)."""
    b1, b2, _nb1, _nb2 = _basis(lattice)
    T1, T2 = nx * b1, ny * b2
    fids = sorted(carpet['interior'])
    fidx = {i: a for a, i in enumerate(fids)}
    fundmap = _fund_map(carpet, lattice, nx, ny)
    cons, _conf = _torus_crossings(carpet, fundmap, fidx)
    L = len(fids)
    S = len(carpet['paths'][0])

    # -- seed: the tier-1 weave.  Per-loop signed crossing lists come
    # from the deduped torus constraints so the seed z respects every
    # constraint exactly (dz = 2*lift >= weave_gap and clearance).
    signed_f = [{} for _ in range(L)]
    for (la, ba), (lb, bb) in cons:
        signed_f[la][ba] = +1
        signed_f[lb][bb] = -1
    lift = 0.6 * max(weave_gap, clearance)
    P = np.empty((L, S, 3))
    for i in fids:
        a = fidx[i]
        p2 = np.asarray(carpet['paths'][i], float)
        sg = sorted(signed_f[a].items())
        P[a, :, :2] = p2
        P[a, :, 2] = isl._weave_zoff([tuple(q) for q in p2], True,
                                     sg, lift)

    # -- flatten weight: small at a crossing bead, ramping to 1 within
    # `win` beads.  The floor (rather than 0) leaves a gentle restoring
    # force everywhere, so the weave cannot balloon in z -- the z-order
    # constraint below still guarantees the gap survives.
    win = max(3, S // 40)
    wfl = np.ones((L, S))
    for a in range(L):
        for b in signed_f[a]:
            for db in range(-win, win + 1):
                j = (b + db) % S
                wfl[a, j] = min(wfl[a, j], abs(db) / float(win))
    wfl = np.maximum(wfl, 0.15)

    # -- repulsion pair list over the torus: fundamental loop pairs
    # plus every periodic image whose centres come near enough to
    # interact.  For a loop against itself, shift (0,0) is the
    # intra-loop case (ring-adjacent beads excluded) and a half set of
    # image shifts covers the wrapped self-contacts.
    cen = np.array([np.mean(carpet['paths'][i], axis=0) for i in fids])
    rad = np.array([np.max(np.linalg.norm(
        np.asarray(carpet['paths'][i]) - cen[a], axis=1))
        for a, i in enumerate(fids)])
    rcut = 1.7 * clearance
    reach = rcut + 0.2
    pairs = []
    for a in range(L):
        for b in range(a, L):
            shifts = (((0, 0), (1, 0), (0, 1), (1, 1), (1, -1))
                      if a == b else
                      tuple((h1, h2) for h1 in (-1, 0, 1)
                            for h2 in (-1, 0, 1)))
            for h1, h2 in shifts:
                sv = h1 * T1 + h2 * T2
                intra = (a == b and h1 == 0 and h2 == 0)
                if intra or (np.linalg.norm(cen[a] - cen[b] - sv)
                             <= rad[a] + rad[b] + reach):
                    pairs.append((a, b, np.array([sv[0], sv[1], 0.0]),
                                  intra))
    excl = max(2, S // 32)
    idx = np.arange(S)
    ring = np.abs(idx[:, None] - idx[None, :])
    near = np.minimum(ring, S - ring) <= excl

    # crossing constraints as flat index arrays
    if cons:
        ol = np.array([c[0][0] for c in cons])
        ob = np.array([c[0][1] for c in cons])
        ul = np.array([c[1][0] for c in cons])
        ub = np.array([c[1][1] for c in cons])

    # gains: ks * step * 4 (the extremal Laplacian eigenvalue) must
    # stay below 1 or the edge springs zigzag at the d_max clamp
    # instead of settling
    ks, fair, kr, kw = 1.2, 0.25, 1.0, 3.0
    rest = np.array([np.mean(np.linalg.norm(
        np.roll(P[a], -1, 0) - P[a], axis=1)) for a in range(L)])
    dmax = 0.25 * clearance
    dmin = 0.2 * clearance
    for _it in range(int(iters)):
        F = np.zeros_like(P)
        # springs: rest-length edges (even spacing, no shrinkage)
        e = np.roll(P, -1, 1) - P
        ln = np.linalg.norm(e, axis=2, keepdims=True)
        fe = ks * (1.0 - rest[:, None, None] / (ln + 1e-12)) * e
        F += fe - np.roll(fe, 1, 1)
        # Laplacian fairing (smoothness)
        F += fair * (0.5 * (np.roll(P, -1, 1) + np.roll(P, 1, 1)) - P)
        # electrical repulsion, min-image over the torus
        for a, b, sv, intra in pairs:
            D = P[a][:, None, :] - (P[b] + sv)[None, :, :]
            d = np.linalg.norm(D, axis=2)
            if intra:
                d[near] = np.inf
            act = d < rcut
            if not act.any():
                continue
            da = np.maximum(d[act], dmin)
            mag = np.zeros_like(d)
            mag[act] = kr * (clearance / da) ** 4 \
                * (1.0 - d[act] / rcut)
            f = D * (mag / (d + 1e-12))[:, :, None]
            F[a] += f.sum(axis=1)
            if not intra:
                F[b] -= f.sum(axis=0)
        # weave z-order: over bead >= under bead + weave_gap
        if cons:
            dz = P[ol, ob, 2] - P[ul, ub, 2]
            push = kw * np.maximum(0.0, weave_gap - dz)
            np.add.at(F, (ol, ob, np.full(len(ol), 2)), 0.5 * push)
            np.add.at(F, (ul, ub, np.full(len(ul), 2)), -0.5 * push)
        # planar confinement away from crossings
        F[:, :, 2] -= flatten * wfl * P[:, :, 2]
        # damped Euler with the d_max step bound
        mv = step * F
        mlen = np.linalg.norm(mv, axis=2, keepdims=True)
        mv *= np.minimum(1.0, dmax / (mlen + 1e-12))
        P += mv
        if trace is not None:
            trace.append(float(np.mean(np.linalg.norm(mv, axis=2))))

    out = []
    for i in range(len(carpet['paths'])):
        j, sh = fundmap[i]
        Q = P[fidx[j]].copy()
        Q[:, 0] += sh[0]
        Q[:, 1] += sh[1]
        out.append(Q)
    return out


def relaxed_paths(lattice='SQUARE', k=4, nx=3, ny=3, amp=0.10,
                  overlap=1.15, samples=192, style='ANGULAR', subdiv=6,
                  iters=120, clearance=0.10, weave_gap=0.10):
    """Relaxed 3D loop centerlines [(points, True)] for the CURVE
    output (tier 2; every loop one closed spline)."""
    carpet = build_carpet(lattice, k, nx, ny, amp, overlap, samples,
                          style, subdiv)
    lines = relax_carpet(carpet, lattice, nx, ny, iters, clearance,
                         weave_gap)
    return [([tuple(p) for p in Q], True) for Q in lines]


# --------------------------------------------------------------------
# The knot ball: the carpet woven over a sphere
# --------------------------------------------------------------------
#
# The same construction with the flat lattice replaced by a GEODESIC
# SPHERE: one rosette loop is centred on every vertex of a geodesic
# icosahedron -- an icosahedron whose 20 faces are each subdivided
# into freq^2 triangles and projected to the unit sphere, the
# geodesic dome subdivision of R. Buckminster Fuller.  Each loop is
# drawn ON the sphere through the exponential map at its vertex: the
# planar rosette point at polar coordinates (rho, theta) in the
# tangent plane lands at
#     cos(rho) v + sin(rho) (cos(theta) e1 + sin(theta) e2),
# so the rosette becomes a geodesic-circle rosette.  Adjacent loops
# overlap and are woven with the SAME parity union-find as the flat
# carpet; "over" now means the larger sphere radius, and the field
# closes into a finite alternating link with no boundary -- a woven
# ball of unknots.
#
# THE TWELVE DEFECTS.  A triangular carpet gives every loop six
# neighbours, but a sphere cannot carry a uniform 6-valent
# triangulation: by Euler's polyhedron formula V - E + F = 2, a
# sphere triangulation whose vertex degrees are only 5 and 6 has
# EXACTLY twelve degree-5 vertices, whatever the frequency -- the
# icosahedral / fullerene / football pattern.  Twelve rosettes
# therefore have five neighbours instead of six; that is a theorem
# of the topology, not an artefact of the construction.
#
# CROSSINGS are found per scaffold edge by projecting both loops
# orthogonally into the tangent plane at the edge midpoint: within
# the near hemisphere the sphere is a single-valued graph over that
# plane, so a 2D polyline crossing there is a genuine crossing of
# the curves on the sphere, and the planar _seg_cross applies
# unchanged.
#
# References:
#   Robert G. Scharein, KnotPlot knot carpets --
#     https://knotplot.com/carpets (see the header above).
#   R. Buckminster Fuller, "Building construction" (the geodesic
#     dome), U.S. Patent 2,682,235 (1954) -- the geodesic
#     subdivision of the icosahedron.
#   L. Euler's polyhedron formula V - E + F = 2 -- the source of the
#     twelve 5-fold defects.

_ICO_F = ((0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
          (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
          (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
          (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1))


def _icosa_verts():
    """The 12 unit icosahedron vertices (golden-rectangle form)."""
    t = 0.5 * (1.0 + sqrt(5.0))
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
                  (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)],
                 float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _geodesic_scaffold(freq):
    """The geodesic icosahedron of frequency `freq`: every face
    subdivided into freq^2 triangles, every vertex projected to the
    unit sphere, shared vertices deduplicated.  Returns (verts,
    edges, adj): the (N, 3) unit vectors, the sorted (i, j) i < j
    edge pairs, and {vertex: sorted neighbour ids}.  N = 10 freq^2
    + 2; the twelve original icosahedron vertices keep degree 5 (the
    defects), every other vertex has degree 6."""
    base = _icosa_verts()
    verts, ids = [], {}

    def vid(p):
        p = p / (np.linalg.norm(p) + 1e-300)
        key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
        if key not in ids:
            ids[key] = len(verts)
            verts.append(p)
        return ids[key]

    edges = set()
    for (a, b, c) in _ICO_F:
        A, B, C = base[a], base[b], base[c]
        grid = {}
        for i in range(freq + 1):
            for j in range(freq + 1 - i):
                grid[(i, j)] = vid((A * (freq - i - j) + B * i
                                    + C * j) / float(freq))
        for i in range(freq):
            for j in range(freq - i):
                tris = [(grid[(i, j)], grid[(i + 1, j)],
                         grid[(i, j + 1)])]
                if i + j < freq - 1:
                    tris.append((grid[(i + 1, j)],
                                 grid[(i + 1, j + 1)],
                                 grid[(i, j + 1)]))
                for t in tris:
                    for u, w in ((t[0], t[1]), (t[1], t[2]),
                                 (t[2], t[0])):
                        edges.add((min(u, w), max(u, w)))
    adj = {i: [] for i in range(len(verts))}
    for (i, j) in sorted(edges):
        adj[i].append(j)
        adj[j].append(i)
    return np.asarray(verts), sorted(edges), adj


def _tangent_frame(v):
    """An orthonormal (e1, e2) spanning the tangent plane at unit
    vector v, e1 toward the projection of global +Z (global +X near
    the poles).  Used for the crossing-projection planes, where the
    azimuth is arbitrary."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(v @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = ref - float(ref @ v) * v
    e1 /= np.linalg.norm(e1) + 1e-12
    return e1, np.cross(v, e1)


def _sphere_loop(v, e1, e2, rho0, k, amp, samples, style, subdiv):
    """One rosette loop ON the unit sphere around scaffold vertex v:
    the planar rosette (via _sample_loop, so ANGULAR / SMOOTH both
    work) pushed through the exponential map at v -- radius becomes
    angular radius, (e1, e2) the tangent basis at v.  Returns an
    (S, 3) closed polyline of unit vectors."""
    p2 = np.asarray(_sample_loop(np.zeros(2), rho0, k, amp, samples,
                                 style, subdiv), float)
    rho = np.hypot(p2[:, 0], p2[:, 1])
    th = np.arctan2(p2[:, 1], p2[:, 0])
    return (np.cos(rho)[:, None] * v
            + np.sin(rho)[:, None] * (np.cos(th)[:, None] * e1
                                      + np.sin(th)[:, None] * e2))


def build_sphere_carpet(freq=2, k=0, amp=0.10, overlap=1.15,
                        samples=192, style='ANGULAR', subdiv=6):
    """The full combinatorial knot ball.  Same dict shape as
    build_carpet, with paths as (S, 3) polylines on the unit sphere:
      paths     -- one (S, 3) closed loop per scaffold vertex
      centers   -- the (N, 3) scaffold vertices; adj -- adjacency
      nn_pairs  -- the scaffold edges (the only pairs tested)
      nn_hits   -- {edge: crossing count}
      crossings, over, per_loop, consistent -- as build_carpet
      interior  -- ALL loop ids (a sphere has no boundary)
      spacing   -- mean neighbour chord (the unit the tube radius,
                   clearance and weave gap are scaled by)
    Each loop's angular radius is 0.5 * overlap * (the LARGEST
    angular distance to a neighbour): geodesic edges vary in length
    by ~15 percent, and the longest incident edge is the one that
    must still be overlapped, so the max (not the mean) keeps the
    planar overlap margin on every edge.

    k = 0 (AUTO, the default) gives each rosette as many lobes as
    its vertex has neighbours -- 6, and 5 at the twelve defects --
    so every lobe maximum faces a neighbour and every lobe minimum
    faces a triangle centre, the spherical analogue of the flat
    triangular carpet's k = 6.  That matters: three loops around a
    scaffold triangle all pass within a hair of its circumcentre,
    and only the aligned lobe minima keep them apart there.  An
    explicit k >= 2 is honoured at every vertex (decorative;
    misaligned lobes can crowd those near-triple points, which the
    relaxation then has to untangle)."""
    V, edges, adj = _geodesic_scaffold(int(freq))
    ang = {}
    for (i, j) in edges:
        ang[(i, j)] = float(np.arccos(np.clip(V[i] @ V[j],
                                              -1.0, 1.0)))
    rho0 = np.array([0.5 * float(overlap)
                     * max(ang[(min(i, j), max(i, j))]
                           for j in adj[i])
                     for i in range(len(V))])
    # tangent basis: e1 toward the FIRST NEIGHBOUR, so lobe maxima
    # face the neighbours and lobe minima face the triangle centres
    # -- the flat triangular carpet's orientation, which keeps three
    # loops from piling up at a scaffold triangle's centroid
    paths = []
    for i in range(len(V)):
        v = V[i]
        e1 = V[adj[i][0]] - float(V[adj[i][0]] @ v) * v
        nrm = float(np.linalg.norm(e1))
        if nrm < 1e-9:
            e1, e2 = _tangent_frame(v)
        else:
            e1 = e1 / nrm
            e2 = np.cross(v, e1)
        kv = int(k) if int(k) >= 2 else len(adj[i])
        paths.append(_sphere_loop(v, e1, e2, rho0[i], kv, amp,
                                  samples, style, subdiv))
    crossings, nn_hits = [], {}
    for (i, j) in edges:
        m = V[i] + V[j]
        m /= np.linalg.norm(m) + 1e-12
        a1, a2 = _tangent_frame(m)
        Pi = np.column_stack([paths[i] @ a1, paths[i] @ a2])
        Pj = np.column_stack([paths[j] @ a1, paths[j] @ a2])
        hits = _seg_cross(Pi, Pj)
        nn_hits[(i, j)] = len(hits)
        for (fi, fj) in hits:
            crossings.append((len(crossings), i, j, fi, fj))
    over, per_loop, consistent = _solve_over(paths, crossings)
    spacing = float(np.mean([np.linalg.norm(V[i] - V[j])
                             for (i, j) in edges]))
    return dict(paths=paths, centers=V, adj=adj, nn_pairs=edges,
                interior=set(range(len(V))), crossings=crossings,
                nn_hits=nn_hits, over=over, per_loop=per_loop,
                consistent=consistent, spacing=spacing)


def _weave_roff(P, signed, h):
    """Per-bead RADIAL offset of a sphere loop: isl._weave_zoff's
    smoothstep between crossings, but along the 3D arclength of the
    closed loop, applied by scaling each unit-sphere point to radius
    1 + roff (over = outward, under = inward)."""
    P = np.asarray(P, float)
    n = len(P)
    z = np.zeros(n)
    if not signed or h <= 0.0:
        return z
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    total = float(s[-1] + seg[-1])
    anchors = sorted((float(s[pi % n]), sg * h) for pi, sg in signed)
    fs, fz = anchors[0]
    ls, lz = anchors[-1]
    ext = [(ls - total, lz)] + anchors + [(fs + total, fz)]
    for i in range(n):
        si = s[i]
        for j in range(len(ext) - 1):
            s0, z0 = ext[j]
            s1, z1 = ext[j + 1]
            if si <= s1 or j == len(ext) - 2:
                span = s1 - s0
                u = 0.0 if span < 1e-12 else min(1.0, max(
                    0.0, (si - s0) / span))
                f = u * u * (3.0 - 2.0 * u)         # smoothstep
                z[i] = z0 + (z1 - z0) * f
                break
    return z


def build_sphere_tube_cells(freq=2, k=0, amp=0.10, overlap=1.15,
                            samples=192, style='ANGULAR', subdiv=6,
                            tube_radius=0.04, tube_sides=10,
                            weave_height=0.06, color_by='LOOP',
                            relax_iters=0, clearance=0.10,
                            weave_gap=0.10):
    """One closed round tube per sphere loop, woven radially: over
    beads swell to radius 1 + roff, under beads sink to 1 - roff,
    smooth-stepped between crossings, then swept with _tube_welded.
    tube_radius / weave_height / clearance / weave_gap are in
    loop-spacing units (the mean neighbour chord), matching the
    planar builders.  With relax_iters > 0 the tubes follow the
    sphere-constrained tier-2 relaxation instead.  Returns one
    (verts, faces, mats) cell per loop."""
    carpet = build_sphere_carpet(freq, k, amp, overlap, samples,
                                 style, subdiv)
    signed = loop_signed(carpet)
    d = carpet['spacing']
    tru = max(0.005, float(tube_radius))
    tr = tru * d
    # the radial lift must exceed the tube radius or the ropes touch
    lift = max(float(weave_height), 1.3 * tru) * d
    sides = max(3, int(tube_sides))
    relaxed = None
    if relax_iters > 0:
        relaxed = relax_sphere_carpet(
            carpet, iters=int(relax_iters),
            clearance=max(float(clearance), 2.2 * tru),
            weave_gap=max(float(weave_gap), 2.2 * tru))
    cells = []
    for i, path in enumerate(carpet['paths']):
        if relaxed is not None:
            path3d = [tuple(p) for p in relaxed[i]]
        else:
            roff = _weave_roff(path, signed[i], lift)
            path3d = [tuple(p) for p in
                      path * (1.0 + roff)[:, None]]
        verts, faces = _tube_welded(path3d, tr, sides)
        if not faces:
            continue
        mat = (i % len(pc.PALETTE_RGBA)) if color_by == 'LOOP' else 0
        cells.append((verts, faces, [mat] * len(faces)))
    return cells


def _sphere_band(path3d, width, thickness):
    """Sweep a closed flat band (a thin box cross-section) along a 3D
    loop on the sphere: the width lies TANGENT to the sphere
    (perpendicular to the loop), the thickness runs RADIALLY -- a woven
    strap on the ball.  Returns (verts, quad faces)."""
    P = np.asarray(path3d, float)
    n = len(P)
    if n < 3:
        return [], []
    hw = 0.5 * float(width)
    ht = 0.5 * max(float(thickness), 1e-4)
    verts, faces = [], []
    for i in range(n):
        p = P[i]
        tang = P[(i + 1) % n] - P[(i - 1) % n]
        tang /= np.linalg.norm(tang) + 1e-12
        rad = p / (np.linalg.norm(p) + 1e-12)
        side = np.cross(tang, rad)
        side /= np.linalg.norm(side) + 1e-12
        for sw, sr in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            verts.append(tuple(p + side * (sw * hw) + rad * (sr * ht)))
    for i in range(n):
        j = (i + 1) % n
        for c in range(4):
            faces.append([4 * i + c, 4 * i + (c + 1) % 4,
                          4 * j + (c + 1) % 4, 4 * j + c])
    return verts, faces


def build_sphere_ribbon_cells(freq=2, k=0, amp=0.10, overlap=1.15,
                              samples=192, style='ANGULAR', subdiv=6,
                              cord_width=0.12, weave_height=0.06,
                              height=0.0, color_by='LOOP', relax_iters=0,
                              clearance=0.10, weave_gap=0.10):
    """One closed woven flat BAND per sphere loop -- the ribbon form of
    the knot ball (a woven strap ball).  Same radial over/under weave as
    the tube builder, but a flat strap tangent to the sphere instead of
    a round tube.  Returns one (verts, faces, mats) cell per loop."""
    carpet = build_sphere_carpet(freq, k, amp, overlap, samples,
                                 style, subdiv)
    signed = loop_signed(carpet)
    d = carpet['spacing']
    cw = max(0.01, float(cord_width))
    width = cw * d
    thick = max(0.0, float(height)) * d
    if thick <= 0.0:
        thick = 0.12 * width
    lift = max(float(weave_height), 0.8 * cw) * d
    relaxed = None
    if relax_iters > 0:
        relaxed = relax_sphere_carpet(
            carpet, iters=int(relax_iters),
            clearance=max(float(clearance), 2.2 * cw),
            weave_gap=max(float(weave_gap), 2.2 * cw))
    cells = []
    for i, path in enumerate(carpet['paths']):
        if relaxed is not None:
            path3d = np.asarray(relaxed[i], float)
        else:
            roff = _weave_roff(path, signed[i], lift)
            path3d = path * (1.0 + roff)[:, None]
        verts, faces = _sphere_band(path3d, width, thick)
        if not faces:
            continue
        mat = (i % len(pc.PALETTE_RGBA)) if color_by == 'LOOP' else 0
        cells.append((verts, faces, [mat] * len(faces)))
    return cells


def relax_sphere_carpet(carpet, iters=120, clearance=0.10,
                        weave_gap=0.10, confine=0.5, step=0.15,
                        trace=None):
    """Physically relax the knot ball (the KnotPlot bead/stick model
    of relax_carpet) with the planar confinement replaced by a
    radial pull back to the unit sphere.  The sphere is finite and
    closed, so ALL loops relax together -- no torus, no periodic
    images.  clearance / weave_gap are in loop-spacing units (the
    mean neighbour chord).  Per iteration: rest-length edge springs
    + Laplacian fairing; soft 1/r^4 repulsion with cutoff between
    nearby strands; at every crossing the over bead is kept at least
    weave_gap FURTHER FROM the centre than the under bead (the
    radial over/under separation); and each bead's radius is pulled
    toward 1 with strength `confine`, weighted down near crossings
    -- the damped reprojection onto the sphere.  Steps are clamped
    to a fraction of clearance (d_max), so strands cannot pass
    through one another.  Returns one (S, 3) centerline per loop."""
    d = float(carpet['spacing'])
    clr = float(clearance) * d
    gap = float(weave_gap) * d
    paths = carpet['paths']
    L = len(paths)
    S = len(paths[0])
    signed = loop_signed(carpet)
    lift = 0.6 * max(gap, clr)
    P = np.empty((L, S, 3))
    for i in range(L):
        roff = _weave_roff(paths[i], signed[i], lift)
        P[i] = np.asarray(paths[i]) * (1.0 + roff)[:, None]

    # crossing constraints as flat index arrays (over / under bead)
    over = carpet['over']
    ol, ob, ul, ub = [], [], [], []
    for key, lo, hi, flo, fhi in carpet['crossings']:
        blo = int(round(flo)) % S
        bhi = int(round(fhi)) % S
        o, u = ((lo, blo), (hi, bhi)) if over[key] \
            else ((hi, bhi), (lo, blo))
        ol.append(o[0])
        ob.append(o[1])
        ul.append(u[0])
        ub.append(u[1])
    ol, ob = np.asarray(ol, int), np.asarray(ob, int)
    ul, ub = np.asarray(ul, int), np.asarray(ub, int)
    have_cons = len(ol) > 0

    # confinement weight: small at a crossing bead, ramping to 1
    # within `win` beads (as relax_carpet's flatten weight)
    win = max(3, S // 40)
    wfl = np.ones((L, S))
    for a in range(len(ol)):
        for (li, bi) in ((ol[a], ob[a]), (ul[a], ub[a])):
            for db in range(-win, win + 1):
                jj = (bi + db) % S
                wfl[li, jj] = min(wfl[li, jj], abs(db) / float(win))
    wfl = np.maximum(wfl, 0.15)

    # repulsion pair list: loop pairs whose bead clouds can interact
    cen = P.mean(axis=1)
    rad = np.array([np.max(np.linalg.norm(P[a] - cen[a], axis=1))
                    for a in range(L)])
    rcut = 1.7 * clr
    reach = rcut + 0.1 * d
    pairs = []
    for a in range(L):
        for b in range(a, L):
            if a == b or (np.linalg.norm(cen[a] - cen[b])
                          <= rad[a] + rad[b] + reach):
                pairs.append((a, b, a == b))
    excl = max(2, S // 32)
    idx = np.arange(S)
    ring = np.abs(idx[:, None] - idx[None, :])
    near = np.minimum(ring, S - ring) <= excl

    ks, fair, kr, kw = 1.2, 0.25, 1.0, 3.0
    rest = np.array([np.mean(np.linalg.norm(
        np.roll(P[a], -1, 0) - P[a], axis=1)) for a in range(L)])
    dmax = 0.25 * clr
    dmin = 0.2 * clr
    for _it in range(int(iters)):
        F = np.zeros_like(P)
        # springs: rest-length edges (even spacing, no shrinkage)
        e = np.roll(P, -1, 1) - P
        ln = np.linalg.norm(e, axis=2, keepdims=True)
        fe = ks * (1.0 - rest[:, None, None] / (ln + 1e-12)) * e
        F += fe - np.roll(fe, 1, 1)
        # Laplacian fairing (smoothness)
        F += fair * (0.5 * (np.roll(P, -1, 1) + np.roll(P, 1, 1))
                     - P)
        # electrical repulsion between nearby strands
        for a, b, intra in pairs:
            D = P[a][:, None, :] - P[b][None, :, :]
            dd = np.linalg.norm(D, axis=2)
            if intra:
                dd[near] = np.inf
            act = dd < rcut
            if not act.any():
                continue
            da = np.maximum(dd[act], dmin)
            mag = np.zeros_like(dd)
            mag[act] = kr * (clr / da) ** 4 \
                * (1.0 - dd[act] / rcut)
            f = D * (mag / (dd + 1e-12))[:, :, None]
            F[a] += f.sum(axis=1)
            if not intra:
                F[b] -= f.sum(axis=0)
        # weave order: over radius >= under radius + weave_gap,
        # pushed along each bead's own radial direction
        if have_cons:
            ro = np.linalg.norm(P[ol, ob], axis=1)
            ru = np.linalg.norm(P[ul, ub], axis=1)
            push = kw * np.maximum(0.0, gap - (ro - ru))
            np.add.at(F, (ol, ob),
                      (0.5 * push / (ro + 1e-12))[:, None]
                      * P[ol, ob])
            np.add.at(F, (ul, ub),
                      -(0.5 * push / (ru + 1e-12))[:, None]
                      * P[ul, ub])
        # spherical confinement away from crossings (radius -> 1)
        r = np.linalg.norm(P, axis=2)
        F += (confine * wfl * (1.0 - r)
              / (r + 1e-12))[:, :, None] * P
        # damped Euler with the d_max step bound
        mv = step * F
        mlen = np.linalg.norm(mv, axis=2, keepdims=True)
        mv *= np.minimum(1.0, dmax / (mlen + 1e-12))
        P += mv
        if trace is not None:
            trace.append(float(np.mean(np.linalg.norm(mv, axis=2))))
    return [P[i].copy() for i in range(L)]


def sphere_loop_paths(freq=2, k=0, amp=0.10, overlap=1.15,
                      samples=192, style='ANGULAR', subdiv=6,
                      iters=0, clearance=0.10, weave_gap=0.10):
    """Knot-ball loop centerlines [(points, True)] for the CURVE
    output: the unit-sphere loops as sampled (iters = 0) or the
    sphere-constrained tier-2 relaxation (iters > 0)."""
    carpet = build_sphere_carpet(freq, k, amp, overlap, samples,
                                 style, subdiv)
    if iters > 0:
        lines = relax_sphere_carpet(carpet, iters=int(iters),
                                    clearance=clearance,
                                    weave_gap=weave_gap)
    else:
        lines = carpet['paths']
    return [([tuple(p) for p in Q], True) for Q in lines]


# --------------------------------------------------------------------
# The tiling carpet: medallions on a uniform tiling
# --------------------------------------------------------------------
#
# The same construction with the lattice generalized to any of the
# regular, Archimedean or Laves (dual) tilings of the shared tiling
# engine: one rosette medallion is centred on every TILE (at its
# centroid), with as many lobes as the tile has edge-neighbours, the
# first lobe aimed at the first neighbour -- so each lobe maximum
# faces a neighbouring medallion across the shared edge.  On the
# square tiling this reproduces the SQUARE carpet; mixed tilings
# such as 3.6.3.6, 4.8.8 or 4.6.12 interleave medallions of
# different sizes and symmetries.  The radius follows the LONGEST
# centroid-to-neighbour distance (times the overlap factor, as on
# the sphere), so the longest incident edge is still overlapped and
# every adjacent pair of medallions interlocks; crossings are exact
# planar segment intersections, tested for EVERY pair of loops whose
# bounding circles meet (crowded tilings put vertex-neighbour
# medallions in contact too), and the whole field is woven
# alternating with the same parity union-find as the flat carpet.
# Unlike the lattice carpets the patch is FINITE -- a general
# uniform tiling has no single translation cell of medallions -- so
# the relaxation pins the boundary medallions lightly instead of
# wrapping a torus.
#
# References:
#   Robert G. Scharein, KnotPlot knot carpets --
#     https://knotplot.com/carpets (see the header above).
#   Branko Gruenbaum & G. C. Shephard, "Tilings and Patterns"
#     (W. H. Freeman, 1987) -- the classification of the uniform
#     (Archimedean) and Laves tilings the scaffolds come from.


def _tiling_scaffold(tiling_name, nx, ny):
    """The medallion scaffold of a uniform tiling patch.  Returns
    (centers, polys, adj, degree, interior): the tile centroids, the
    (M, 2) tile polygons, {tile: sorted edge-neighbour ids},
    {tile: neighbour count}, and the id set of INTERIOR tiles (every
    edge shared with another tile).  Neighbours are found through a
    shared-edge map keyed on the rounded unordered endpoint pair, so
    tiny arithmetic differences between tile constructions cannot
    split an edge in two."""
    polys = [np.asarray(p, float)
             for p in tg._build_tiling(tiling_name, nx, ny)[0]]
    centers = [p.mean(axis=0) for p in polys]
    owners = {}
    for ti, p in enumerate(polys):
        m = len(p)
        for a in range(m):
            q0 = (round(float(p[a][0]), 6),
                  round(float(p[a][1]), 6))
            q1 = (round(float(p[(a + 1) % m][0]), 6),
                  round(float(p[(a + 1) % m][1]), 6))
            if q0 == q1:
                continue
            key = (q0, q1) if q0 < q1 else (q1, q0)
            owners.setdefault(key, []).append(ti)
    adj = {i: set() for i in range(len(polys))}
    shared = {i: 0 for i in range(len(polys))}
    for own in owners.values():
        if len(own) == 2 and own[0] != own[1]:
            i, j = own
            adj[i].add(j)
            adj[j].add(i)
            shared[i] += 1
            shared[j] += 1
    adj = {i: sorted(s) for i, s in adj.items()}
    degree = {i: len(adj[i]) for i in adj}
    interior = {i for i in range(len(polys))
                if shared[i] == len(polys[i])}
    return centers, polys, adj, degree, interior


def build_tiling_carpet(tiling_name='SQUARE', nx=3, ny=3, amp=0.10,
                        overlap=1.15, samples=192, style='ANGULAR',
                        subdiv=6):
    """The full combinatorial tiling carpet.  Same dict shape as
    build_carpet, one medallion per tile:
      paths     -- one (S, 2) closed polyline per tile
      centers   -- the tile centroids; polys / adj / degree -- the
                   scaffold (see _tiling_scaffold)
      nn_pairs  -- the edge-adjacent tile pairs (i < j)
      nn_hits   -- {pair: crossing count} over those pairs
      crossings, over, per_loop, consistent -- as build_carpet
      interior  -- tiles with a full neighbour ring
      spacing   -- mean neighbour-centroid distance (the
                   loop-spacing unit widths and radii scale by)
    Each medallion's radius is 0.5 * overlap * (the LARGEST distance
    from its centroid to a neighbour centroid): mixed tilings pair
    tiles of different sizes, and the longest incident edge is the
    one that must still be overlapped, so the max (not the mean)
    keeps the interlock on every edge.  The radius is then CAPPED so
    the rosette's lobe minima clear the tile's own vertices: several
    medallions meet around every tiling vertex, and a loop that
    swallows a vertex grazes its vertex-neighbours exactly where a
    third loop passes -- a three-strand bundle whose alternating
    over/under is cyclic (rock-paper-scissors) and so cannot be
    realized by any z-ordering.  Keeping the minima inside the
    vertex circle dissolves those bundles; a floor of just over
    half the neighbour distance preserves the interlock whenever
    the cap and the overlap conflict.  The lobe count is AUTO -- as
    many lobes as the tile has neighbours, the first lobe aimed at
    the first neighbour.  Crossings are tested for every loop pair
    whose bounding circles meet (not only edge-adjacent pairs), so
    vertex-neighbour contacts in crowded tilings stay woven."""
    centers, polys, adj, degree, interior = _tiling_scaffold(
        tiling_name, nx, ny)
    T = len(centers)
    paths = []
    for i in range(T):
        c = centers[i]
        nb = adj[i]
        if nb:
            dmax = max(float(np.linalg.norm(centers[j] - c))
                       for j in nb)
            R = 0.5 * float(overlap) * dmax
            # vertex-clearing cap (see above): lobe minima R(1-amp)
            # stay inside the nearest tile vertex, floored so the
            # farthest neighbour is still overlapped
            dvert = float(np.min(np.linalg.norm(polys[i] - c,
                                                axis=1)))
            cap = 0.98 * dvert / max(0.65, 1.0 - float(amp))
            R = max(min(R, cap), 0.51 * dmax)
            kv = max(1, degree[i])
            dv = centers[nb[0]] - c
            phi = float(np.arctan2(dv[1], dv[0]))
            av = amp
        else:                       # isolated tile: a plain circle
            R = 0.5 * float(overlap) * float(np.max(
                np.linalg.norm(polys[i] - c, axis=1)))
            kv, phi, av = 2, 0.0, 0.0
        p2 = np.asarray(_sample_loop(np.zeros(2), R, kv, av,
                                     samples, style, subdiv), float)
        cp, sp = np.cos(phi), np.sin(phi)
        rot = np.array([[cp, sp], [-sp, cp]])
        paths.append(c + p2 @ rot)
    C = np.asarray(centers, float)
    rad = np.array([float(np.max(np.linalg.norm(paths[i] - C[i],
                                                axis=1)))
                    for i in range(T)])
    nn_pairs = sorted((i, j) for i in adj for j in adj[i] if i < j)
    nn_set = set(nn_pairs)
    crossings = []
    nn_hits = {p: 0 for p in nn_pairs}
    for i in range(T):
        d = np.linalg.norm(C - C[i], axis=1)
        cand = np.nonzero((np.arange(T) > i)
                          & (d < rad + rad[i] + 1e-9))[0]
        for j in cand:
            j = int(j)
            hits = _seg_cross(paths[i], paths[j])
            if (i, j) in nn_set:
                nn_hits[(i, j)] = len(hits)
            for (fi, fj) in hits:
                crossings.append((len(crossings), i, j, fi, fj))
    over, per_loop, consistent = _solve_over(paths, crossings)
    if nn_pairs:
        spacing = float(np.mean([np.linalg.norm(C[i] - C[j])
                                 for (i, j) in nn_pairs]))
    else:
        spacing = 1.0
    return dict(paths=paths, centers=C, polys=polys, adj=adj,
                degree=degree, nn_pairs=nn_pairs, nn_hits=nn_hits,
                interior=interior, crossings=crossings, over=over,
                per_loop=per_loop, consistent=consistent,
                spacing=spacing)


def build_tiling_ribbon_cells(tiling_name='SQUARE', nx=3, ny=3,
                              amp=0.10, overlap=1.15, samples=192,
                              cord_width=0.12, style='ANGULAR',
                              subdiv=6, interlace=True,
                              interlace_mode='FLAT',
                              weave_height=0.05, color_by='LOOP',
                              height=0.0, backing=False, base=0.06):
    """One merged (verts, faces, mats) cell per medallion, plus an
    optional backing slab -- the ribbon form of the tiling carpet
    (flat knotwork or 3D weave via _loop_cell, as build_cells).
    cord_width / weave_height / height / base are in loop-spacing
    units (the mean neighbour-centroid distance), matching the
    planar builders."""
    carpet = build_tiling_carpet(tiling_name, nx, ny, amp, overlap,
                                 samples, style, subdiv)
    signed = loop_signed(carpet)
    d = carpet['spacing']
    width = max(0.01, float(cord_width)) * d
    cells = []
    all_verts = []
    for i, path in enumerate(carpet['paths']):
        pl = [tuple(p) for p in path]
        sub = _loop_cell(pl, signed[i], width, interlace,
                         interlace_mode, float(weave_height) * d,
                         float(height) * d, color_by, i)
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
        pc.slab(cv, cf, cm, lo, hi, 0.0, -float(base) * d,
                mat=_BACKING_MAT)
        cells.append((cv, cf, cm))
    return cells


def build_tiling_tube_cells(tiling_name='SQUARE', nx=3, ny=3,
                            amp=0.10, overlap=1.15, samples=192,
                            style='ANGULAR', subdiv=6,
                            tube_radius=0.04, tube_sides=10,
                            weave_height=0.06, color_by='LOOP',
                            backing=False, base=0.06, relax_iters=0,
                            clearance=0.10, weave_gap=0.10):
    """One closed round tube per medallion, dipping in z at its
    crossings -- the rope form of the tiling carpet (as
    build_tube_cells).  tube_radius / weave_height / clearance /
    weave_gap are in loop-spacing units; the over/under lift is
    floored at the tube diameter so the ropes cannot touch.  With
    relax_iters > 0 the tubes follow the finite-patch tier-2
    relaxation instead of the flat weave path.  Returns one (verts,
    faces, mats) cell per medallion (+ an optional backing slab)."""
    carpet = build_tiling_carpet(tiling_name, nx, ny, amp, overlap,
                                 samples, style, subdiv)
    signed = loop_signed(carpet)
    d = carpet['spacing']
    tru = max(0.005, float(tube_radius))
    tr = tru * d
    # the over/under lift must exceed the tube radius or ropes touch
    lift = max(float(weave_height), 1.3 * tru) * d
    sides = max(3, int(tube_sides))
    relaxed = None
    if relax_iters > 0:
        relaxed = relax_tiling_carpet(
            carpet, iters=int(relax_iters),
            clearance=max(float(clearance), 2.2 * tru),
            weave_gap=max(float(weave_gap), 2.2 * tru))
    cells = []
    all_verts = []
    for i, path in enumerate(carpet['paths']):
        if relaxed is not None:
            path3d = [tuple(p) for p in relaxed[i]]
        else:
            pl2 = [(float(p[0]), float(p[1])) for p in path]
            zoff = isl._weave_zoff(pl2, True, signed[i], lift)
            path3d = [(pl2[j][0], pl2[j][1], zoff[j])
                      for j in range(len(pl2))]
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
        pc.slab(cv, cf, cm, lo, hi, -tr, -tr - float(base) * d,
                mat=_BACKING_MAT)
        cells.append((cv, cf, cm))
    return cells


def relax_tiling_carpet(carpet, iters=120, clearance=0.10,
                        weave_gap=0.10, flatten=0.5, pin=0.4,
                        step=0.15, trace=None):
    """Physically relax the tiling carpet (the KnotPlot bead/stick
    model of relax_carpet) on the FINITE patch: no torus, no
    periodic images -- all medallions relax together in plain 3D
    distance, and the BOUNDARY medallions (incomplete neighbour
    ring) are lightly pinned to their seed x, y so the patch cannot
    drift or shear while the interior settles.  clearance /
    weave_gap are in loop-spacing units.  Per iteration:
    rest-length edge springs + Laplacian fairing; soft 1/r^4
    repulsion with cutoff between nearby strands; the over bead
    kept at least weave_gap above the under bead at every crossing;
    z pulled toward 0 away from crossings (strength `flatten`); and
    each step clamped to a fraction of clearance (KnotPlot's d_max),
    so strands cannot pass through one another.  Returns one (S, 3)
    centerline per medallion."""
    d = float(carpet['spacing'])
    clr = float(clearance) * d
    gap = float(weave_gap) * d
    paths = carpet['paths']
    L = len(paths)
    S = len(paths[0])
    signed = loop_signed(carpet)
    lift = 0.6 * max(gap, clr)
    P = np.empty((L, S, 3))
    for i in range(L):
        p2 = np.asarray(paths[i], float)
        P[i, :, :2] = p2
        P[i, :, 2] = isl._weave_zoff([tuple(q) for q in p2], True,
                                     signed[i], lift)
    seed_xy = P[:, :, :2].copy()
    pinned = np.array([0.0 if i in carpet['interior'] else 1.0
                       for i in range(L)])

    # crossing constraints as flat index arrays (over / under bead)
    over = carpet['over']
    ol, ob, ul, ub = [], [], [], []
    for key, lo, hi, flo, fhi in carpet['crossings']:
        blo = int(round(flo)) % S
        bhi = int(round(fhi)) % S
        o, u = ((lo, blo), (hi, bhi)) if over[key] \
            else ((hi, bhi), (lo, blo))
        ol.append(o[0])
        ob.append(o[1])
        ul.append(u[0])
        ub.append(u[1])
    ol, ob = np.asarray(ol, int), np.asarray(ob, int)
    ul, ub = np.asarray(ul, int), np.asarray(ub, int)
    have_cons = len(ol) > 0

    # flatten weight: small at a crossing bead, ramping to 1 within
    # `win` beads (as relax_carpet's flatten weight)
    win = max(3, S // 40)
    wfl = np.ones((L, S))
    for a in range(len(ol)):
        for (li, bi) in ((ol[a], ob[a]), (ul[a], ub[a])):
            for db in range(-win, win + 1):
                jj = (bi + db) % S
                wfl[li, jj] = min(wfl[li, jj], abs(db) / float(win))
    wfl = np.maximum(wfl, 0.15)

    # repulsion pair list: loop pairs whose bead clouds can interact
    cen = P.mean(axis=1)
    rad = np.array([np.max(np.linalg.norm(P[a] - cen[a], axis=1))
                    for a in range(L)])
    rcut = 1.7 * clr
    reach = rcut + 0.1 * d
    pairs = []
    for a in range(L):
        for b in range(a, L):
            if a == b or (np.linalg.norm(cen[a] - cen[b])
                          <= rad[a] + rad[b] + reach):
                pairs.append((a, b, a == b))
    excl = max(2, S // 32)
    idx = np.arange(S)
    ring = np.abs(idx[:, None] - idx[None, :])
    near = np.minimum(ring, S - ring) <= excl

    ks, fair, kr, kw = 1.2, 0.25, 1.0, 3.0
    rest = np.array([np.mean(np.linalg.norm(
        np.roll(P[a], -1, 0) - P[a], axis=1)) for a in range(L)])
    dmax = 0.25 * clr
    dmin = 0.2 * clr
    for _it in range(int(iters)):
        F = np.zeros_like(P)
        # springs: rest-length edges (even spacing, no shrinkage)
        e = np.roll(P, -1, 1) - P
        ln = np.linalg.norm(e, axis=2, keepdims=True)
        fe = ks * (1.0 - rest[:, None, None] / (ln + 1e-12)) * e
        F += fe - np.roll(fe, 1, 1)
        # Laplacian fairing (smoothness)
        F += fair * (0.5 * (np.roll(P, -1, 1) + np.roll(P, 1, 1))
                     - P)
        # electrical repulsion between nearby strands
        for a, b, intra in pairs:
            D = P[a][:, None, :] - P[b][None, :, :]
            dd = np.linalg.norm(D, axis=2)
            if intra:
                dd[near] = np.inf
            act = dd < rcut
            if not act.any():
                continue
            da = np.maximum(dd[act], dmin)
            mag = np.zeros_like(dd)
            mag[act] = kr * (clr / da) ** 4 \
                * (1.0 - dd[act] / rcut)
            f = D * (mag / (dd + 1e-12))[:, :, None]
            F[a] += f.sum(axis=1)
            if not intra:
                F[b] -= f.sum(axis=0)
        # weave z-order: over bead >= under bead + weave_gap
        if have_cons:
            dz = P[ol, ob, 2] - P[ul, ub, 2]
            push = kw * np.maximum(0.0, gap - dz)
            np.add.at(F, (ol, ob, np.full(len(ol), 2)), 0.5 * push)
            np.add.at(F, (ul, ub, np.full(len(ul), 2)),
                      -0.5 * push)
        # planar confinement away from crossings
        F[:, :, 2] -= flatten * wfl * P[:, :, 2]
        # light boundary pinning (x, y toward the seed positions)
        F[:, :, :2] += (pin * pinned)[:, None, None] \
            * (seed_xy - P[:, :, :2])
        # damped Euler with the d_max step bound
        mv = step * F
        mlen = np.linalg.norm(mv, axis=2, keepdims=True)
        mv *= np.minimum(1.0, dmax / (mlen + 1e-12))
        P += mv
        if trace is not None:
            trace.append(float(np.mean(np.linalg.norm(mv, axis=2))))
    return [P[i].copy() for i in range(L)]


def tiling_loop_paths(tiling_name='SQUARE', nx=3, ny=3, amp=0.10,
                      overlap=1.15, samples=192, style='ANGULAR',
                      subdiv=6, iters=0, clearance=0.10,
                      weave_gap=0.10):
    """Tiling-carpet loop centerlines [(points, True)] for the CURVE
    output: the flat 2D medallions as sampled (iters = 0) or the
    finite-patch tier-2 relaxation (iters > 0)."""
    carpet = build_tiling_carpet(tiling_name, nx, ny, amp, overlap,
                                 samples, style, subdiv)
    if iters > 0:
        lines = relax_tiling_carpet(carpet, iters=int(iters),
                                    clearance=clearance,
                                    weave_gap=weave_gap)
        return [([tuple(p) for p in Q], True) for Q in lines]
    return [([tuple(p) for p in Q], True)
            for Q in carpet['paths']]


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
        centerlines (2D flat or 3D relaxed points), centered and
        scaled to the span cube."""
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
            for i, p in enumerate(pa):
                pz = p[2] if len(p) > 2 else 0.0
                sp.points[i].co = ((p[0] - cx) * s, (p[1] - cy) * s,
                                   pz * s, 1.0)
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
                    "Loops on a triangular lattice (6 neighbours)"),
                   ('SPHERE', "Sphere",
                    "Loops on a geodesic icosahedron -- a closed "
                    "woven knot ball (6 neighbours, with twelve "
                    "5-neighbour defects)"),
                   ('TILING', "Tiling",
                    "Medallion loops on a regular/Archimedean "
                    "tiling -- one woven loop per tile, lobes "
                    "following the tile's neighbour count")],
            default='SQUARE')
        sphere_freq: IntProperty(
            name="Sphere Frequency", default=2, min=1, max=6,
            description="Geodesic subdivision frequency of the "
                        "icosahedral scaffold (sphere lattice); "
                        "10 x freq^2 + 2 loops")
        tiling_name: EnumProperty(
            name="Tiling", items=tg.TILING_ITEMS, default='SQUARE',
            description="The underlying tiling (one medallion per "
                        "tile)")
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
        relax_iters: IntProperty(
            name="Relax Iterations", default=0, min=0, max=500,
            description="0 = flat tier-1 weave; > 0 physically "
                        "relaxes the carpet into 3D rope "
                        "(KnotPlot-style bead/stick dynamics on the "
                        "torus, so it still tiles)")
        rope_clearance: FloatProperty(
            name="Rope Clearance", default=0.10, min=0.02, max=0.4,
            description="Target centerline separation between "
                        "strands during relaxation (loop-spacing "
                        "units)")
        weave_gap: FloatProperty(
            name="Weave Gap", default=0.10, min=0.02, max=0.4,
            description="Minimum z separation kept between the over "
                        "and under strand at each crossing during "
                        "relaxation")
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

        def _execute_sphere(self, context):
            """The closed knot ball.  CURVE emits the loop centerlines;
            RIBBON sweeps a woven flat strap per loop; TUBE sweeps a
            woven round rope.  The rosette lobe count is AUTO (k = 0):
            each loop follows its scaffold degree, so lobes stay aligned
            with the neighbours across the twelve 5-fold defects."""
            if self.output == 'CURVE':
                paths = sphere_loop_paths(
                    self.sphere_freq, 0, self.amplitude,
                    self.overlap, self.samples, self.style,
                    self.smoothness, self.relax_iters,
                    self.rope_clearance, self.weave_gap)
                obj = _emit_curve(context, "Knot Ball", paths,
                                  operator=self)
                if obj is None:
                    self.report({'ERROR'}, "no carpet generated")
                    return {'CANCELLED'}
                obj["math_art_pattern"] = True
                self.report({'INFO'}, "SPHERE freq=%d  %d loops" %
                            (self.sphere_freq, len(paths)))
                return {'FINISHED'}
            if self.output == 'RIBBON':
                cells = build_sphere_ribbon_cells(
                    self.sphere_freq, 0, self.amplitude,
                    self.overlap, self.samples, self.style,
                    self.smoothness, self.cord_width,
                    self.weave_height, self.height, self.color_by,
                    self.relax_iters, self.rope_clearance,
                    self.weave_gap)
            else:
                cells = build_sphere_tube_cells(
                    self.sphere_freq, 0, self.amplitude,
                    self.overlap, self.samples, self.style,
                    self.smoothness, self.tube_radius, self.tube_sides,
                    self.weave_height, self.color_by, self.relax_iters,
                    self.rope_clearance, self.weave_gap)
            obj = pc.emit(context, "Knot Ball", cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no carpet generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if self.output == 'TUBE':
                _shade_smooth(obj)
            if obj.type == 'MESH':
                self.report({'INFO'},
                            "SPHERE freq=%d  %d loops  V=%d F=%d"
                            % (self.sphere_freq, len(cells),
                               len(obj.data.vertices),
                               len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "SPHERE freq=%d  %d loops" %
                            (self.sphere_freq, len(cells)))
            return {'FINISHED'}

        def _execute_tiling(self, context):
            """The tiling carpet.  CURVE emits the loop centerlines;
            RIBBON builds the interlaced ribbons; TUBE sweeps woven
            round ropes.  The rosette lobe count is AUTO: each
            medallion follows its tile's neighbour count, so lobes
            stay aimed at the adjacent medallions on every tiling."""
            if self.output == 'CURVE':
                paths = tiling_loop_paths(
                    self.tiling_name, self.nx, self.ny,
                    self.amplitude, self.overlap, self.samples,
                    self.style, self.smoothness, self.relax_iters,
                    self.rope_clearance, self.weave_gap)
                obj = _emit_curve(context, "Knot Carpet (Tiling)",
                                  paths, operator=self)
                if obj is None:
                    self.report({'ERROR'}, "no carpet generated")
                    return {'CANCELLED'}
                obj["math_art_pattern"] = True
                self.report({'INFO'}, "TILING %s %dx%d  %d loops" %
                            (self.tiling_name, self.nx, self.ny,
                             len(paths)))
                return {'FINISHED'}
            if self.output == 'RIBBON':
                cells = build_tiling_ribbon_cells(
                    self.tiling_name, self.nx, self.ny,
                    self.amplitude, self.overlap, self.samples,
                    self.cord_width, self.style, self.smoothness,
                    self.interlace, self.interlace_mode,
                    self.weave_height, self.color_by, self.height,
                    self.backing, self.base)
            else:
                cells = build_tiling_tube_cells(
                    self.tiling_name, self.nx, self.ny,
                    self.amplitude, self.overlap, self.samples,
                    self.style, self.smoothness, self.tube_radius,
                    self.tube_sides, self.weave_height,
                    self.color_by, self.backing, self.base,
                    self.relax_iters, self.rope_clearance,
                    self.weave_gap)
            obj = pc.emit(context, "Knot Carpet (Tiling)", cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no carpet generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if self.output == 'TUBE':
                _shade_smooth(obj)
            n_loops = len(cells) - (1 if self.backing else 0)
            if obj.type == 'MESH':
                self.report({'INFO'},
                            "TILING %s %dx%d  %d loops  V=%d F=%d"
                            % (self.tiling_name, self.nx, self.ny,
                               n_loops, len(obj.data.vertices),
                               len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "TILING %s %dx%d  %d loops" %
                            (self.tiling_name, self.nx, self.ny,
                             n_loops))
            return {'FINISHED'}

        def execute(self, context):
            if self.lattice == 'SPHERE':
                return self._execute_sphere(context)
            if self.lattice == 'TILING':
                return self._execute_tiling(context)
            if self.output == 'CURVE':
                if self.relax_iters > 0:
                    paths = relaxed_paths(
                        self.lattice, self.symmetry, self.nx, self.ny,
                        self.amplitude, self.overlap, self.samples,
                        self.style, self.smoothness, self.relax_iters,
                        self.rope_clearance, self.weave_gap)
                else:
                    paths = loop_paths(self.lattice, self.symmetry,
                                       self.nx, self.ny,
                                       self.amplitude, self.overlap,
                                       self.samples, self.style,
                                       self.smoothness)
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
                    self.backing, self.base, self.relax_iters,
                    self.rope_clearance, self.weave_gap)
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
            sphere = self.lattice == 'SPHERE'
            tiling = self.lattice == 'TILING'
            if sphere:
                # lobe count is AUTO on the sphere (follows the
                # scaffold degree), so symmetry is not shown
                lay.prop(self, 'sphere_freq')
            elif tiling:
                # lobe count is AUTO on a tiling too (each medallion
                # follows its tile's neighbour count)
                lay.prop(self, 'tiling_name')
                lay.prop(self, 'nx')
                lay.prop(self, 'ny')
            else:
                lay.prop(self, 'symmetry')
                lay.prop(self, 'nx')
                lay.prop(self, 'ny')
            lay.prop(self, 'amplitude')
            lay.prop(self, 'overlap')
            lay.prop(self, 'samples')
            if not (sphere or tiling):
                lay.prop(self, 'cord_width')
            lay.prop(self, 'style')
            if self.style == 'SMOOTH':
                lay.prop(self, 'smoothness')
            lay.prop(self, 'output')
            if tiling:
                if self.output == 'RIBBON':
                    lay.prop(self, 'cord_width')
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
                if self.output in ('TUBE', 'CURVE'):
                    lay.prop(self, 'relax_iters')
                    if self.relax_iters > 0:
                        lay.prop(self, 'rope_clearance')
                        lay.prop(self, 'weave_gap')
                lay.prop(self, 'align')
                return
            if sphere:
                if self.output == 'RIBBON':
                    lay.prop(self, 'cord_width')
                    lay.prop(self, 'weave_height')
                    lay.prop(self, 'height')
                    lay.prop(self, 'color_by')
                    lay.prop(self, 'separate')
                elif self.output == 'TUBE':
                    lay.prop(self, 'tube_radius')
                    lay.prop(self, 'tube_sides')
                    lay.prop(self, 'weave_height')
                    lay.prop(self, 'color_by')
                    lay.prop(self, 'separate')
                lay.prop(self, 'relax_iters')
                if self.relax_iters > 0:
                    lay.prop(self, 'rope_clearance')
                    lay.prop(self, 'weave_gap')
                lay.prop(self, 'align')
                return
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
            if self.output in ('TUBE', 'CURVE'):
                lay.prop(self, 'relax_iters')
                if self.relax_iters > 0:
                    lay.prop(self, 'rope_clearance')
                    lay.prop(self, 'weave_gap')
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


def _torus_min_sep(lines, carpet, lattice, nx, ny):
    """Minimum min-image distance between beads of DIFFERENT loops of
    the infinite carpet: fundamental loop pairs plus each loop against
    its own periodic images (direct same-loop pairs excluded).  An
    independent implementation (fractional min-image rounding), so it
    does not share code with the relaxer it checks."""
    b1, b2, _nb1, _nb2 = _basis(lattice)
    M = np.column_stack([nx * b1, ny * b2])
    Minv = np.linalg.inv(M)
    fids = sorted(carpet['interior'])
    best = np.inf
    for ai in range(len(fids)):
        for bi in range(ai, len(fids)):
            A = np.asarray(lines[fids[ai]], float)
            B = np.asarray(lines[fids[bi]], float)
            D = A[:, None, :] - B[None, :, :]
            dxy = D[..., :2]
            wrapped = dxy - np.round(dxy @ Minv.T) @ M.T
            d = np.sqrt(((wrapped) ** 2).sum(-1) + D[..., 2] ** 2)
            if ai == bi:
                # keep only pairs whose min image is a WRAPPED copy
                moved = np.abs(wrapped - dxy).max(-1) > 1e-9
                d[~moved] = np.inf
            best = min(best, float(d.min()))
    return best


def _check_relax(lattice, k, nx, ny, samples=120, iters=120,
                 clearance=0.10, weave_gap=0.10):
    """Run the tier-2 relaxation and verify: loops closed and finite;
    exact tileability (margin loops = wrapped fundamental translates,
    loop centres periodic); inter-loop separation safe (vs the tier-1
    seed); the over strand above the under strand at every crossing;
    and convergence (movement decreasing, nothing diverging).
    Returns (ok, stats dict)."""
    b1, b2, _nb1, _nb2 = _basis(lattice)
    carpet = build_carpet(lattice, k, nx, ny, 0.10, 1.15, samples)
    seed = relax_carpet(carpet, lattice, nx, ny, iters=0,
                        clearance=clearance, weave_gap=weave_gap)
    trace = []
    lines = relax_carpet(carpet, lattice, nx, ny, iters=iters,
                         clearance=clearance, weave_gap=weave_gap,
                         trace=trace)
    st = {}
    finite = all(np.isfinite(Q).all() for Q in lines)
    # closed: no bead flew off its loop (edge lengths stay bounded)
    closed = True
    for Q in lines:
        el = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
        if el.max() > 4.0 * el.mean():
            closed = False
    # tileability: every loop matches its wrapped fundamental partner
    fundmap = _fund_map(carpet, lattice, nx, ny)
    tile_err = 0.0
    for i, Q in enumerate(lines):
        j, sh = fundmap[i]
        ref = lines[j] + np.array([sh[0], sh[1], 0.0])
        tile_err = max(tile_err, float(np.abs(Q - ref).max()))
    cen_err = 0.0
    for i in range(len(lines)):
        j, sh = fundmap[i]
        d = np.mean(lines[i], 0) - np.mean(lines[j], 0) \
            - np.array([sh[0], sh[1], 0.0])
        cen_err = max(cen_err, float(np.abs(d).max()))
    st['tile_err'] = max(tile_err, cen_err)
    # separation: min inter-loop min-image distance, before vs after
    st['sep_before'] = _torus_min_sep(seed, carpet, lattice, nx, ny)
    st['sep_after'] = _torus_min_sep(lines, carpet, lattice, nx, ny)
    sep_ok = (st['sep_after'] >= 0.6 * clearance
              and (st['sep_after'] >= st['sep_before'] - 1e-9
                   or st['sep_after'] >= 0.8 * clearance))
    # z-order at every torus crossing
    fids = sorted(carpet['interior'])
    fidx = {i: a for a, i in enumerate(fids)}
    cons, conf = _torus_crossings(carpet, fundmap, fidx)
    P = np.array([lines[i] for i in fids])
    dz = np.array([P[la, ba, 2] - P[lb, bb, 2]
                   for (la, ba), (lb, bb) in cons])
    st['min_dz'] = float(dz.min())
    st['conflicts'] = conf
    z_ok = conf == 0 and st['min_dz'] >= 0.6 * weave_gap
    # convergence: finite trace, late movement well below early
    tr = np.asarray(trace)
    st['mv_early'] = float(tr[:10].mean())
    st['mv_late'] = float(tr[-10:].mean())
    conv = (np.isfinite(tr).all() and len(tr) == iters
            and st['mv_late'] < st['mv_early'])
    st['ok'] = (finite and closed and st['tile_err'] < 1e-8
                and sep_ok and z_ok and conv)
    return st['ok'], st


def _sphere_min_sep(lines):
    """Minimum 3D distance between beads of DIFFERENT loops of the
    knot ball (the sphere is finite -- no images, plain pairs)."""
    best = np.inf
    for a in range(len(lines)):
        A = np.asarray(lines[a], float)
        for b in range(a + 1, len(lines)):
            B = np.asarray(lines[b], float)
            D = A[:, None, :] - B[None, :, :]
            best = min(best, float(np.sqrt((D ** 2).sum(-1)).min()))
    return best


def _check_sphere_relax(freq, k, samples=96, iters=120,
                        clearance=0.10, weave_gap=0.10):
    """Run the sphere-constrained relaxation and verify: loops closed
    and finite; every bead stays within a tolerance band of the unit
    sphere; inter-loop separation safe (vs the radial-weave seed);
    the over strand at LARGER sphere radius than the under strand at
    every crossing; and convergence (movement decreasing).  Returns
    (ok, stats dict)."""
    carpet = build_sphere_carpet(freq, k, 0.08, 1.15, samples)
    d = carpet['spacing']
    clr = clearance * d
    gap = weave_gap * d
    seed = relax_sphere_carpet(carpet, iters=0, clearance=clearance,
                               weave_gap=weave_gap)
    trace = []
    lines = relax_sphere_carpet(carpet, iters=iters,
                                clearance=clearance,
                                weave_gap=weave_gap, trace=trace)
    st = {}
    finite = all(np.isfinite(Q).all() for Q in lines)
    closed = True
    for Q in lines:
        el = np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1)
        if el.max() > 4.0 * el.mean():
            closed = False
    # radius band: beads stay near the unit sphere
    r = np.concatenate([np.linalg.norm(Q, axis=1) for Q in lines])
    st['rdev'] = float(np.abs(r - 1.0).max())
    st['band'] = 2.0 * max(clr, gap)
    rad_ok = st['rdev'] <= st['band']
    # separation: min inter-loop distance, before vs after
    st['sep_before'] = _sphere_min_sep(seed)
    st['sep_after'] = _sphere_min_sep(lines)
    sep_ok = (st['sep_after'] >= 0.6 * clr
              and (st['sep_after'] >= st['sep_before'] - 1e-9
                   or st['sep_after'] >= 0.8 * clr))
    # radial over/under order at every crossing
    over = carpet['over']
    S = len(carpet['paths'][0])
    min_dz = np.inf
    for key, lo, hi, flo, fhi in carpet['crossings']:
        rlo = float(np.linalg.norm(lines[lo][int(round(flo)) % S]))
        rhi = float(np.linalg.norm(lines[hi][int(round(fhi)) % S]))
        dz = (rlo - rhi) if over[key] else (rhi - rlo)
        min_dz = min(min_dz, dz)
    st['min_dz'] = float(min_dz)
    st['gap'] = gap
    z_ok = st['min_dz'] >= 0.6 * gap
    # convergence: finite trace, late movement well below early
    tr = np.asarray(trace)
    st['mv_early'] = float(tr[:10].mean())
    st['mv_late'] = float(tr[-10:].mean())
    conv = (np.isfinite(tr).all() and len(tr) == iters
            and st['mv_late'] < st['mv_early'])
    st['ok'] = (finite and closed and rad_ok and sep_ok and z_ok
                and conv)
    return st['ok'], st


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

    # 7. tier-2 relaxation: closed/finite loops, exact tileability,
    #    safe separation, weave z-order, convergence
    for lattice, k, nx, ny in (('SQUARE', 4, 3, 3),
                               ('TRIANGULAR', 6, 2, 2)):
        r, st = _check_relax(lattice, k, nx, ny)
        ok = ok and r
        print("relax    %-10s k=%d %dx%d : %s  tile_err=%.2e  "
              "sep %.4f->%.4f  min_dz=%.4f (gap 0.10)  "
              "move %.2e->%.2e  conflicts=%d"
              % (lattice, k, nx, ny, r, st['tile_err'],
                 st['sep_before'], st['sep_after'], st['min_dz'],
                 st['mv_early'], st['mv_late'], st['conflicts']))

    # 8. sphere scaffold: 10 freq^2 + 2 vertices, exactly twelve of
    #    degree 5 and the rest degree 6 (Euler's twelve defects)
    for freq in (1, 2, 3):
        V, edges, adj = _geodesic_scaffold(freq)
        deg = [len(adj[i]) for i in range(len(V))]
        n5 = deg.count(5)
        n6 = deg.count(6)
        scaf = (len(V) == 10 * freq * freq + 2 and n5 == 12
                and n5 + n6 == len(V))
        ok = ok and scaf
        print("scaffold freq=%d : verts=%d edges=%d deg5=%d deg6=%d "
              ": %s" % (freq, len(V), len(edges), n5, n6, scaf))

    # 9. sphere carpet: every adjacent loop pair crosses >= 2 times,
    #    every crossing one-over-one-under (always) -- AUTO lobes
    #    (k = 0, deg(v)-fold) and one explicit-k case
    for freq, kk, amp in ((1, 0, 0.0), (1, 0, 0.12), (2, 0, 0.0),
                          (2, 0, 0.12), (2, 6, 0.10)):
        car = build_sphere_carpet(freq, kk, amp, 1.15, 160)
        min_hits = min(car['nn_hits'].values())
        one = _check_one_over(car)
        good = min_hits >= 2 and one
        ok = ok and good
        print("sphere   freq=%d k=%d amp=%.2f : min-crossings=%d "
              "one-over=%s  (%d crossings, consistent=%s)"
              % (freq, kk, amp, min_hits, one,
                 len(car['crossings']), car['consistent']))

    # 10. knot-ball tubes: finite, non-empty, all quads
    cells = build_sphere_tube_cells(2, 0, samples=120, tube_sides=8)
    faces = sum(len(c[1]) for c in cells)
    finite = all(all(np.isfinite(v).all() for v in c[0])
                 for c in cells)
    quads = all(len(f) == 4 for c in cells for f in c[1])
    t = faces > 0 and finite and quads
    ok = ok and t
    print("sphere tube freq=2 : %s  (%d loops, %d faces)"
          % (t, len(cells), faces))

    # 10b. sphere RIBBON (woven straps) builds valid quad geometry
    rcells = build_sphere_ribbon_cells(2, 0, samples=120)
    rfaces = sum(len(c[1]) for c in rcells)
    rfin = all(all(np.isfinite(v).all() for v in c[0])
               for c in rcells)
    rquad = all(len(f) == 4 for c in rcells for f in c[1])
    rb = rfaces > 0 and rfin and rquad
    ok = ok and rb
    print("sphere ribbon freq=2 : %s  (%d loops, %d faces)"
          % (rb, len(rcells), rfaces))

    # 11. sphere relaxation: beads near the sphere, safe separation,
    #     radial over/under order, convergence
    r, st = _check_sphere_relax(2, 0)
    ok = ok and r
    print("sphere relax freq=2 : %s  rdev=%.4f (band %.4f)  "
          "sep %.4f->%.4f  min_dz=%.4f (gap %.4f)  move %.2e->%.2e"
          % (r, st['rdev'], st['band'], st['sep_before'],
             st['sep_after'], st['min_dz'], st['gap'],
             st['mv_early'], st['mv_late']))

    # 12. tiling scaffold + carpet: symmetric adjacency, interior
    #     degree = side count, >= 2 crossings per adjacent medallion
    #     pair, one-over-one-under (always)
    _tiling_ids = [t[0] for t in tg.TILING_ITEMS]
    assert 'SQUARE' in _tiling_ids
    for tname in ('SQUARE', 'TRIHEX', 'TRUNCSQ'):
        centers, polys, adjt, degree, interior = _tiling_scaffold(
            tname, 4, 4)
        sym = all(i in adjt[j] for i in adjt for j in adjt[i])
        intd = all(degree[i] == len(polys[i]) for i in interior)
        car = build_tiling_carpet(tname, 4, 4, 0.10, 1.15, 160)
        min_hits = min(car['nn_hits'].values())
        one = _check_one_over(car)
        good = (sym and intd and len(interior) > 0
                and min_hits >= 2 and one)
        ok = ok and good
        print("tiling   %-8s : sym=%s interior=%d deg-ok=%s "
              "min-crossings=%d one-over=%s  (%d crossings, "
              "consistent=%s) : %s"
              % (tname, sym, len(interior), intd, min_hits, one,
                 len(car['crossings']), car['consistent'], good))

    # 13. tiling tubes (finite, non-empty, all quads) and ribbons
    #     (finite, non-empty)
    for tname in ('SQUARE', 'TRIHEX', 'TRUNCSQ'):
        tc = build_tiling_tube_cells(tname, 3, 3, samples=120,
                                     tube_sides=8)
        tf = sum(len(c[1]) for c in tc)
        tfin = all(all(np.isfinite(v).all() for v in c[0])
                   for c in tc)
        tq = all(len(f) == 4 for c in tc for f in c[1])
        rc = build_tiling_ribbon_cells(tname, 3, 3, samples=120)
        rf = sum(len(c[1]) for c in rc)
        rfin = all(all(np.isfinite(v).all() for v in c[0])
                   for c in rc)
        g = tf > 0 and tfin and tq and rf > 0 and rfin
        ok = ok and g
        print("tiling geom %-8s : tube %d faces quads=%s  "
              "ribbon %d faces : %s" % (tname, tf, tq, rf, g))

    # 14. tiling relaxation: finite, safe separation, weave z-order
    #     at every INTERIOR-pair crossing (boundary medallions have
    #     defective lobe counts and stay lightly pinned), convergence
    for tname in ('TRIHEX', 'TRUNCSQ'):
        car = build_tiling_carpet(tname, 3, 3, 0.10, 1.15, 120)
        dsp = car['spacing']
        clr = 0.10 * dsp
        gap = 0.10 * dsp
        seed = relax_tiling_carpet(car, iters=0)
        trace = []
        lines = relax_tiling_carpet(car, iters=120, trace=trace)
        fin = all(np.isfinite(Q).all() for Q in lines)
        sb = _sphere_min_sep(seed)      # generic min inter-loop sep
        sa = _sphere_min_sep(lines)
        sep_ok = (sa >= 0.6 * clr
                  and (sa >= sb - 1e-9 or sa >= 0.8 * clr))
        S = len(car['paths'][0])
        min_dz = np.inf
        for key, lo, hi, flo, fhi in car['crossings']:
            if lo not in car['interior'] \
                    or hi not in car['interior']:
                continue
            zlo = float(lines[lo][int(round(flo)) % S][2])
            zhi = float(lines[hi][int(round(fhi)) % S][2])
            dz = (zlo - zhi) if car['over'][key] else (zhi - zlo)
            min_dz = min(min_dz, dz)
        z_ok = min_dz >= 0.6 * gap
        tr = np.asarray(trace)
        conv = (np.isfinite(tr).all() and len(tr) == 120
                and tr[-10:].mean() < tr[:10].mean())
        g = fin and sep_ok and z_ok and conv
        ok = ok and g
        print("tiling relax %-8s : %s  sep %.4f->%.4f (clr %.4f)  "
              "interior min_dz=%.4f (gap %.4f)  move %.2e->%.2e"
              % (tname, g, sb, sa, clr, min_dz, gap,
                 tr[:10].mean(), tr[-10:].mean()))

    print("RESULT:", "OK" if ok else "BAD")
