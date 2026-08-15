
# Fractal Knotwork Generator for Blender
#
# An infinite (self-similar) over-under LINK woven over a self-similar
# tiling.  Take a Fathauer fractal tiling ("f-tiling") -- a patch tiled
# by similar kites that shrink generation by generation toward a
# fractal boundary -- and decorate it with the classic medial / mirror-
# curve knotwork: put one CROSSING on every tiling edge midpoint, route
# the cords around the faces, and weave them strictly one-over-one-
# under.  Because the substrate is self-similar, the resulting
# interlaced link is self-similar too: every deeper generation adds a
# finer halo of smaller cords, crossings and components around the
# previous ones -- a "fractal knotwork" in the spirit of Fathauer's
# iterated-substitution fractal knots.
#
# SUBSTRATE.  The kite f-tilings of fractal_tiling_generator (KITE_R6 /
# KITE_R8 / KITE_R12).  These are EDGE-TO-EDGE by construction at every
# finite depth: a child's long edge glues endpoint-to-endpoint onto its
# parent's short edge (child long = s * long = parent short), fan
# siblings pair long edges, and coincident tips pair short edges -- so
# unlike the general f-tiling family there are NO T-junctions between
# scales (that module's own self-test proves the full-edge pairing).
# The polygons are WELDED into a shared-vertex cell complex here, and a
# T-JUNCTION SPLITTER still runs as a verified safety net: every tile
# edge is split at any welded vertex lying in its interior, so the
# complex is a proper edge-to-edge cell complex (each interior edge
# shared by exactly two faces) before the medial graph is built.  The
# self-test asserts the validity of the welded complex and reports the
# split count (zero on these substrates).
#
# MEDIAL KNOT.  On the welded complex the cord lives on the medial
# graph, exactly as in celtic_knot_2d_generator's arbitrary-tiling
# path (Mercat's "knot on any graph" / Jablan's mirror curves): an
# unwalled interior edge is a crossing the cord passes straight
# through; a boundary edge reflects it (a U-turn).  The deterministic
# successor on directed corner-segments is a bijection, so its orbits
# partition every corner-segment into closed cords.  The over/under is
# the alternating-knot assignment solved with a parity union-find --
# each crossing carries one strand over and one under, and the sense
# alternates along every cord (mirror curves realize alternating
# links).  All of that machinery is REUSED from the Celtic module; this
# module contributes the fractal substrate, the welding / T-junction
# handling, and a SCALE-AWARE rendering.
#
# SCALE-AWARE CORDS.  On a fractal tiling the edges span many sizes
# (ratio s per generation), so a constant-width cord would swamp the
# fine tiles.  Every control point of a cord is an edge midpoint whose
# LOCAL SCALE is that edge's length; the cord's width (and its woven
# z-amplitude) follows that scale along its length, interpolated in
# log-space through the Catmull-Rom smoothing -- big cords over big
# tiles taper smoothly into fine cords over fine tiles, exactly as in
# Fathauer's fractal-knot drawings.
#
# OUTPUTS.  RIBBON -- flat interlaced tape, FLAT (under-cord broken at
# crossings) or WOVEN (cords rise and dip in z); TUBE -- a round rope
# swept with rotation-minimizing frames and welded closed (all-quad,
# smooth-shaded), always woven in z so the ropes genuinely pass over
# and under; CURVE -- the woven centerlines as a Blender curve object.
# Colors: uniform, per link component, or an over/under checker.
#
# References:
#   Robert W. Fathauer, "Fractal Knots Created by Iterative
#     Substitution", Bridges Donostia: Mathematics, Music, Art,
#     Architecture, Culture (2007), pp. 335-342 -- self-similar knots
#     and links built by iterated substitution of a knotted motif.
#   Robert W. Fathauer, "Fractal tilings based on kite- and dart-
#     shaped prototiles" (Computers & Graphics 24, 2000) -- the
#     f-tiling substrate reproduced by fractal_tiling_generator.
#   Christian Mercat, "Les entrelacs des enluminures celtes" (2005) /
#     "Knots and links from a graph" -- the medial construction of an
#     interlaced knot on ANY planar graph (one crossing per edge),
#     shared with celtic_knot_2d_generator.
#   Slavik V. Jablan, "Mirror curves" and "Symmetry, Ornament and
#     Modularity" (2002) -- mirror curves as alternating knot and link
#     diagrams on arbitrary graphs.

bl_info = {
    "name": "Fractal Knotwork",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Self-similar over-under link woven on a Fathauer "
                   "fractal tiling -- scale-aware medial knotwork",
    "category": "Add Mesh",
}

from bisect import bisect_right
from collections import Counter
from math import exp, hypot, log

import numpy as np

try:                                  # inside the math_art package
    from .patterns.polygon2d import (arclen)
    from .patterns.ribbon import (band_ribbon_faces, band_ribbon_faces_z, catmull_rom, miter_ribbon)
    from .patterns.overunder import (ParityDSU, weave_zoff)
except ImportError:                   # flat import (test runner)
    from patterns.polygon2d import (arclen)
    from patterns.ribbon import (band_ribbon_faces, band_ribbon_faces_z, catmull_rom, miter_ribbon)
    from patterns.overunder import (ParityDSU, weave_zoff)

try:                                  # inside the math_art package
    from .curve_frames import welded_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import welded_tube

try:
    from .patterns import common as pc
    from . import islamic_pattern_generator as isl
    from . import celtic_knot_2d_generator as ck
    from . import fractal_tiling_generator as ft
    from . import knot_carpet_generator as kc
    from . import spiral_tiling_generator as st
except Exception:                       # legacy single-file / CLI use
    from patterns import common as pc
    import islamic_pattern_generator as isl
    import celtic_knot_2d_generator as ck
    import fractal_tiling_generator as ft
    import knot_carpet_generator as kc
    import spiral_tiling_generator as st


# --------------------------------------------------------------------
# Welding the fractal patch into an edge-to-edge cell complex
# --------------------------------------------------------------------

def _weld(polys, nd=6):
    """Weld a list of (M, 2) polygons into (coords, rings): shared
    vertices deduplicated on a rounded key (1e-6, far below the
    smallest tile edge at any offered depth), rings as vertex-index
    lists.  The rounded coordinates are kept as the canonical
    positions so paired edges and their midpoints match exactly."""
    coords, index, rings = [], {}, []
    for p in polys:
        ring = []
        for x, y in p:
            key = (round(float(x), nd), round(float(y), nd))
            vi = index.get(key)
            if vi is None:
                vi = len(coords)
                index[key] = vi
                coords.append((key[0], key[1]))
            ring.append(vi)
        ring = [v for i, v in enumerate(ring)
                if v != ring[(i - 1) % len(ring)]]      # drop dupes
        if len(ring) >= 3:
            rings.append(ring)
    return coords, rings


def _split_t_junctions(coords, rings, tol=1e-6):
    """T-JUNCTION handling (the medial graph needs a proper cell
    complex): split every tile edge at all welded vertices lying in
    its interior, so a small tile's corner landing on a big tile's
    edge becomes a genuine shared vertex of both rings.  Only edges
    whose endpoint-pair key is UNPAIRED (used once) can host a
    T-junction -- an exactly paired edge already matches its partner
    endpoint to endpoint -- so only those are scanned.  Returns
    (rings, n_inserted); on the kite f-tilings n_inserted is zero
    (they are edge-to-edge by construction) and this is a verified
    no-op safety net."""
    use = Counter()
    for r in rings:
        n = len(r)
        for i in range(n):
            a, b = r[i], r[(i + 1) % n]
            use[(a, b) if a < b else (b, a)] += 1
    V = np.asarray(coords, float)
    insert = {}
    n_inserted = 0
    for key, c in use.items():
        if c != 1:
            continue
        ia, ib = key
        a = V[ia]
        d = V[ib] - a
        L2 = float(d @ d)
        if L2 < 1e-18:
            continue
        t = ((V - a) @ d) / L2
        perp = np.linalg.norm(V - (a + t[:, None] * d), axis=1)
        eps = tol / (L2 ** 0.5)
        hit = np.nonzero((perp <= tol) & (t > eps) & (t < 1.0 - eps))[0]
        vids = [int(v) for v in hit if v != ia and v != ib]
        if vids:
            vids.sort(key=lambda v: t[v])
            insert[key] = vids
            n_inserted += len(vids)
    if insert:
        out = []
        for r in rings:
            n = len(r)
            nr = []
            for i in range(n):
                a, b = r[i], r[(i + 1) % n]
                nr.append(a)
                seq = insert.get((a, b) if a < b else (b, a))
                if seq:
                    nr.extend(seq if a < b else seq[::-1])
            out.append(nr)
        rings = out
    return rings, n_inserted


def _substrate_polys(kind, iterations):
    """Tile polygons of the chosen substrate.  Fathauer kite f-tilings
    (KITE_*) come from fractal_tiling_generator; the SPIRAL_* log-spiral
    triangle tilings come from spiral_tiling_generator.  Both are
    self-similar, so the medial cord width -- which follows the LOCAL
    edge length -- tapers from the big outer tiles to the tiny inner
    ones."""
    if kind.startswith('SPIRAL_'):
        # log-spiral tilings of similar triangles: SELF-SIMILAR (tiles
        # shrink inward by scale-level), so the local-edge-length cord
        # width tapers the knotwork from the big outer triangles toward
        # the tiny inner ones.  `iterations` -> spiral rings, arms fixed
        # per option; the Golden family is the default in the screenshot.
        _SP = {'SPIRAL_GOLDEN': ('GOLDEN', 3, 1),
               'SPIRAL_GOLDEN_2': ('GOLDEN', 3, 2),
               'SPIRAL_EQUILATERAL': ('EQUILATERAL', 3, 5)}
        fam, sn, arms = _SP[kind]
        rings = max(2, min(int(iterations) + 2, 10))
        polys, _arms, _lvl = st.spiral_patch(fam, sn, arms, 50.0, rings)
        return polys
    polys, _types = ft.fractal_patch(kind, iterations)
    return polys


def build_topology(kind, iterations):
    """Medial-graph topology of a welded tiling patch.  Returns
    (topo, n_split): the celtic module's tiling topology dict (per-edge
    incident faces / midpoints, interior and boundary edge lists) and
    the number of T-junction splits performed (0 on the edge-to-edge
    kite substrates; the splitter is a safety net for any others)."""
    polys = _substrate_polys(kind, iterations)
    coords, rings = _weld(polys)
    rings, n_split = _split_t_junctions(coords, rings)
    rings = [ck._poly_ccw(coords, r) for r in rings]
    return ck._tiling_topology(coords, rings), n_split


def _edge_length(coords, key):
    a, b = key
    return hypot(coords[a][0] - coords[b][0],
                 coords[a][1] - coords[b][1])


# --------------------------------------------------------------------
# Cords and the alternating over/under
# --------------------------------------------------------------------

def _solve_over(cords):
    """Alternating over/under across all crossings, like the celtic
    module's tiling solve but also COUNTING parity clashes: between
    consecutive crossings on a cord the reference-strand parity must
    flip (x1 XOR x2 = 1 XOR lab1 XOR lab2), resolved with a parity
    union-find.  Returns (x, clashes); clashes == 0 means the diagram
    is perfectly alternating (mirror curves are, so this is expected
    to be 0 and the self-test reports it)."""
    dsu = ParityDSU()
    keys = set()
    clashes = 0
    for rec in cords:
        L = len(rec)
        cidx = [k for k in range(L) if not rec[k][1]]
        for _co, isr, _lab, cid in rec:
            if not isr:
                keys.add(cid)
        C = len(cidx)
        for a in range(C):
            k1 = cidx[a]
            k2 = cidx[(a + 1) % C]
            if not dsu.union(rec[k1][3], rec[k2][3],
                             1 ^ rec[k1][2] ^ rec[k2][2]):
                clashes += 1
    x = {}
    for key in sorted(keys):
        dsu.add(key)
        _r, par = dsu.find(key)
        x[key] = par
    return x, clashes


def trace_knotwork(kind, iterations):
    """The complete fractal knotwork: returns a dict with the medial
    topology, the closed cords, the over/under solution, and the
    headline counts (components, crossings, parity clashes, T-splits).
    Every cord is a closed loop; together they form the link."""
    topo, n_split = build_topology(kind, iterations)
    cords = ck._tiling_trace_cords(topo, set())
    x, clashes = _solve_over(cords)
    return {'topo': topo, 'cords': cords, 'x': x,
            'components': len(cords),
            'crossings': len(topo['interior']),
            'clashes': clashes, 'n_split': n_split}


def _cord_render(rec, coords, x, style, subdiv, cord_width):
    """Drawable data of one closed cord: (path, widths, signed).
    `path` is the (optionally Catmull-Rom smoothed) centerline through
    the edge midpoints; `widths` is the LOCAL cord width at every
    sample -- cord_width times the local edge length, interpolated in
    log-space so it tapers geometrically between generations; `signed`
    is [(sample_index, +1 over / -1 under)] at the crossings."""
    control = [(float(c[0][0]), float(c[0][1])) for c in rec]
    scales = [max(_edge_length(coords, c[3]), 1e-9) for c in rec]
    cross = [(k, 1 if (x.get(rec[k][3], 0) ^ rec[k][2]) else -1)
             for k in range(len(rec)) if not rec[k][1]]
    smoothed = style == 'SMOOTH' and len(control) >= 3 and subdiv >= 2
    step = subdiv if smoothed else 1
    if smoothed:
        path = catmull_rom(control, True, subdiv)
        lw = catmull_rom([(log(s),) for s in scales], True, subdiv)
        widths = [cord_width * exp(v[0]) for v in lw]
    else:
        path = [tuple(p) for p in control]
        widths = [cord_width * s for s in scales]
    signed = [(k * step, sg) for k, sg in cross]
    return path, widths, signed


# --------------------------------------------------------------------
# Scale-aware ribbon / tube assembly
# --------------------------------------------------------------------

def _var_ribbon(path, widths, closed):
    """Variable-width mitered ribbon: reuse the Islamic miter machinery
    at unit half-width, then scale each station's offset by its local
    half-width.  Returns (left, right) parallel to `path`."""
    lu, ru = miter_ribbon(path, 2.0, closed)     # unit offsets
    left, right = [], []
    for p, a, b, w in zip(path, lu, ru, widths):
        h = 0.5 * w
        left.append((p[0] + (a[0] - p[0]) * h, p[1] + (a[1] - p[1]) * h))
        right.append((p[0] + (b[0] - p[0]) * h, p[1] + (b[1] - p[1]) * h))
    return left, right


def _sample_at(path, widths, s, total, arc):
    """(point, width, arc) interpolated at arclength `arc` on a closed
    path with cumulative arclengths `s`."""
    arc = arc % total
    n = len(path)
    if arc >= s[-1]:
        i, j = n - 1, 0
        seg = total - s[-1]
        t = 0.0 if seg < 1e-12 else (arc - s[-1]) / seg
    else:
        i = max(0, bisect_right(s, arc) - 1)
        j = i + 1
        seg = s[j] - s[i]
        t = 0.0 if seg < 1e-12 else (arc - s[i]) / seg
    p = (path[i][0] + t * (path[j][0] - path[i][0]),
         path[i][1] + t * (path[j][1] - path[i][1]))
    w = widths[i] + t * (widths[j] - widths[i])
    return p, w, arc


def _cut_closed_var(path, widths, s, total, cuts):
    """Break a CLOSED variable-width centerline into open pieces,
    removing a gap of LOCAL half-length around each cut arclength (the
    under-passages of the FLAT interlace; the gap follows the local
    cord scale).  `cuts` is [(arc, half)].  Returns a list of
    (points, widths, arcs, closed); a cord with no cuts comes back as
    one closed piece."""
    if not cuts:
        return [(list(path), list(widths), list(s), True)]
    intervals = []
    for c, h in cuts:
        a = (c - h) % total
        b = a + 2.0 * h
        if b <= total:
            intervals.append((a, b))
        else:
            intervals.append((a, total))
            intervals.append((0.0, b - total))
    intervals.sort()
    merged = []
    for a, b in intervals:
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    kept = []
    prev = 0.0
    for a, b in merged:
        if a > prev + 1e-9:
            kept.append((prev, a))
        prev = max(prev, b)
    if prev < total - 1e-9:
        kept.append((prev, total))
    pieces = []
    for lo, hi in kept:
        pts, ws, arcs = [], [], []
        p, w, a = _sample_at(path, widths, s, total, lo)
        pts.append(p); ws.append(w); arcs.append(lo)
        for i in range(len(path)):
            if lo + 1e-9 < s[i] < hi - 1e-9:
                pts.append(path[i]); ws.append(widths[i]); arcs.append(s[i])
        p, w, a = _sample_at(path, widths, s, total, hi)
        pts.append(p); ws.append(w); arcs.append(hi)
        pieces.append((pts, ws, arcs, False))
    # join the two pieces flanking the seam (no cut covers arc 0)
    if (len(pieces) >= 2 and kept[0][0] <= 1e-9
            and kept[-1][1] >= total - 1e-9):
        lp, lw, la, _c = pieces.pop()
        fp, fw, fa, _c = pieces.pop(0)
        pieces.append((lp + fp[1:], lw + fw[1:],
                       la + [a + total for a in fa[1:]], False))
    return [pc_ for pc_ in pieces if len(pc_[0]) >= 2]


def _weave_amp(widths, cord_width, weave_height, floor=0.0):
    """Per-sample woven z amplitude: `weave_height` (a fraction of the
    local edge, like cord_width) times the local scale, floored at
    `floor` cord-widths so tubes always clear each other."""
    f = max(weave_height, floor * cord_width)
    return [f * w / cord_width for w in widths]


def _parity(cross_arcs, total, arc):
    """Number of crossings passed before `arc`, mod 2 -- the checker
    color of the cord arc containing `arc` (arcs between crossings
    alternate, tracking the over/under sense)."""
    return bisect_right(cross_arcs, arc % total) % 2


def _matof(color_by, li):
    if color_by == 'LOOP':
        return li % len(pc.PALETTE_RGBA)
    return 0


def _ribbon_cells(path, widths, signed, mode, weave_height, cord_width,
                  color_by, li):
    """Ribbon sub-cells (verts, faces, mats) for one cord.  FLAT
    breaks the cord where it passes under (gap scaled to the local
    width); WOVEN lifts and dips the whole cord in z.  CHECKER colors
    each inter-crossing arc by its alternation parity."""
    s, total = arclen(path, True)
    cross_arcs = sorted(s[m] for m, _sg in signed)

    def face_mats(arcs, nfaces, closed):
        if color_by != 'CHECKER':
            return [_matof(color_by, li)] * nfaces
        out = []
        for i in range(nfaces):
            j = i + 1
            if closed and j == len(arcs):
                mid = 0.5 * (arcs[i] + total)
            else:
                mid = 0.5 * (arcs[i] + arcs[j])
            out.append(_parity(cross_arcs, total, mid))
        return out

    cells = []
    if mode == 'WOVEN':
        amp = _weave_amp(widths, cord_width, weave_height)
        zu = weave_zoff(path, True, signed, 1.0)
        zoff = [zu[i] * amp[i] for i in range(len(path))]
        left, right = _var_ribbon(path, widths, True)
        cv, cf = band_ribbon_faces_z(left, right, True, 0.0, zoff)
        if cf:
            cells.append((cv, cf, face_mats(s, len(cf), True)))
        return cells
    # FLAT: cut the under-passages with locally scaled gaps
    cuts = [(s[m], 0.5 * 1.3 * widths[m]) for m, sg in signed if sg < 0]
    for pts, ws, arcs, closed in _cut_closed_var(path, widths, s, total,
                                                 cuts):
        left, right = _var_ribbon(pts, ws, closed)
        cv, cf = band_ribbon_faces(left, right, closed, 0.0)
        if cf:
            cells.append((cv, cf, face_mats(arcs, len(cf), closed)))
    return cells


def _tube_cell(path, widths, signed, weave_height, cord_width,
               tube_sides, color_by, li):
    """One all-quad woven TUBE cell for a closed cord: the centerline
    is lifted into the woven z (amplitude floored to clear the tube
    radius, so over and under ropes never touch), swept with the knot-
    carpet's welded rotation-minimizing tube, then each ring is
    rescaled about its center to the LOCAL cord radius -- a rope that
    tapers with the fractal scale while staying a closed quad mesh."""
    n = len(path)
    if n < 3:
        return None
    amp = _weave_amp(widths, cord_width, weave_height, floor=0.6)
    zu = weave_zoff(path, True, signed, 1.0)
    p3 = [(path[i][0], path[i][1], zu[i] * amp[i]) for i in range(n)]
    r0 = 0.5 * cord_width
    verts, faces = welded_tube(p3, r0, tube_sides)
    if not faces:
        return None
    V = np.asarray(verts, float).reshape(n, tube_sides, 3)
    P = np.asarray(p3, float)[:, None, :]
    f = (0.5 * np.asarray(widths, float) / r0)[:, None, None]
    V = P + f * (V - P)
    verts = [tuple(v) for v in V.reshape(-1, 3)]
    if color_by == 'CHECKER':
        s, total = arclen(path, True)
        cross_arcs = sorted(s[m] for m, _sg in signed)
        mats = []
        for i in range(n):
            hi = s[i + 1] if i + 1 < n else total
            par = _parity(cross_arcs, total, 0.5 * (s[i] + hi))
            mats.extend([par] * tube_sides)
    else:
        mats = [_matof(color_by, li)] * len(faces)
    return verts, [tuple(fc) for fc in faces], mats


def build_cells(kind='KITE_R6', iterations=3, cord_width=0.32,
                style='SMOOTH', subdiv=6, interlace_mode='FLAT',
                weave_height=0.25, output='RIBBON', tube_sides=12,
                color_by='LOOP'):
    """One (verts, faces, mats) cell per link component of the fractal
    knotwork.  `cord_width` and `weave_height` are fractions of the
    LOCAL tile edge, so the cords taper with the fractal scale.  TUBE
    is always woven in z (a broken round tube would need triangle end
    caps; the welded woven tube stays all-quad and reads as rope)."""
    net = trace_knotwork(kind, iterations)
    coords = net['topo']['coords']
    cells = []
    for li, rec in enumerate(net['cords']):
        path, widths, signed = _cord_render(rec, coords, net['x'],
                                            style, subdiv, cord_width)
        if len(path) < 3:
            continue
        if output == 'TUBE':
            cell = _tube_cell(path, widths, signed, weave_height,
                              cord_width, tube_sides, color_by, li)
            if cell and cell[1]:
                cells.append(cell)
        else:
            sub = _ribbon_cells(path, widths, signed, interlace_mode,
                                weave_height, cord_width, color_by, li)
            if sub:
                cell = pc.merge_cells(sub)
                if cell[1]:
                    cells.append(cell)
    return cells


def cord_paths(kind='KITE_R6', iterations=3, cord_width=0.32,
               style='SMOOTH', subdiv=6, weave_height=0.25):
    """Woven 3D centerlines [(points_xyz, closed)] for the CURVE
    output -- every component one closed spline riding its over/under
    z-offsets."""
    net = trace_knotwork(kind, iterations)
    coords = net['topo']['coords']
    out = []
    for rec in net['cords']:
        path, widths, signed = _cord_render(rec, coords, net['x'],
                                            style, subdiv, cord_width)
        if len(path) < 2:
            continue
        amp = _weave_amp(widths, cord_width, weave_height)
        zu = weave_zoff(path, True, signed, 1.0)
        out.append(([(path[i][0], path[i][1], zu[i] * amp[i])
                     for i in range(len(path))], True))
    return out


# --------------------------------------------------------------------
# Blender operator
# --------------------------------------------------------------------

SUBSTRATE_ITEMS = [
    ('KITE_R6', "Kite f-tiling (6-fold)",
     "Fathauer f-tiling of 60-degree kites (ratio 0.577): four "
     "children per tip -- the densest, most legible knotwork"),
    ('KITE_R8', "Kite f-tiling (8-fold)",
     "Fathauer f-tiling of 45-degree kites (ratio 0.414): five "
     "children per tip"),
    ('KITE_R12', "Kite f-tiling (12-fold)",
     "Fathauer f-tiling of 30-degree kites (ratio 0.268): seven "
     "children per tip -- steep scale contrast"),
    ('SPIRAL_GOLDEN', "Golden-triangle spiral",
     "Log-spiral tiling of golden triangles (self-similar): the "
     "knotwork tapers from big outer triangles to tiny inner ones "
     "(Iterations = rings)"),
    ('SPIRAL_GOLDEN_2', "Golden-triangle spiral (2-arm)",
     "Two-arm golden-triangle log-spiral -- a tapering double-spiral "
     "knotwork (Iterations = rings)"),
    ('SPIRAL_EQUILATERAL', "Equilateral spiral (5-fold)",
     "Five-arm log-spiral of equilateral triangles (plastic-number "
     "scale) -- a self-similar tapering pinwheel knotwork "
     "(Iterations = rings)"),
]


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    from bpy_extras.object_utils import AddObjectHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _shade_smooth(obj):
        if obj is not None and obj.type == 'MESH':
            for p in obj.data.polygons:
                p.use_smooth = True
            obj.data.update()

    class MESH_OT_fractal_knotwork_add(bpy.types.Operator,
                                       AddObjectHelper):
        """Add a fractal knotwork (self-similar interlaced link on a """ \
        """Fathauer fractal tiling)"""
        bl_idname = "mesh.fractal_knotwork_add"
        bl_label = "Fractal Knotwork"
        bl_options = {'REGISTER', 'UNDO'}

        substrate: EnumProperty(
            name="Substrate", items=SUBSTRATE_ITEMS, default='KITE_R6',
            description="The self-similar tiling the knotwork is woven "
                        "over (one crossing per tiling edge)")
        iterations: IntProperty(
            name="Iterations", default=3, min=0, max=6,
            description="Fractal depth: each generation adds a finer "
                        "halo of smaller cords and crossings (5+ is "
                        "heavy)")
        cord_width: FloatProperty(
            name="Cord Width", default=0.05, min=0.05, max=0.85,
            description="Cord width as a fraction of the LOCAL tile "
                        "edge (the cords taper with the fractal scale)")
        style: EnumProperty(
            name="Style",
            items=[('ANGULAR', "Angular",
                    "Straight mitered cords with sharp corners"),
                   ('SMOOTH', "Smooth",
                    "Catmull-Rom curved cords")],
            default='SMOOTH')
        smoothness: IntProperty(
            name="Smoothness", default=6, min=2, max=16,
            description="Spline subdivisions per cord segment (smooth)")
        interlace_mode: EnumProperty(
            name="Interlace",
            items=[('FLAT', "Flat Knotwork",
                    "Break the under-cord at crossings (2D interlace)"),
                   ('WOVEN', "Woven (3D)",
                    "Raise/dip each cord into a 3D woven surface")],
            default='FLAT',
            description="How the over/under weave is rendered "
                        "(TUBE always weaves in z)")
        weave_height: FloatProperty(
            name="Weave Height", default=0.07, min=0.0, max=1.0,
            description="Woven z amplitude as a fraction of the local "
                        "tile edge (woven ribbon / tube / curve)")
        output: EnumProperty(
            name="Output",
            items=[('RIBBON', "Ribbon Mesh",
                    "Flat interlaced cord ribbons"),
                   ('TUBE', "Round Tube",
                    "Woven all-quad rope tubes (smooth-shaded)"),
                   ('CURVE', "Centerline Curves",
                    "Woven cord centerlines as a curve object")],
            default='RIBBON')
        tube_sides: IntProperty(
            name="Tube Sides", default=12, min=3, max=32,
            description="Facets around the tube cross-section")
        color_by: EnumProperty(
            name="Color By",
            items=[('UNIFORM', "Uniform", "A single material"),
                   ('LOOP', "By Component",
                    "A distinct color per link component"),
                   ('CHECKER', "Over/Under",
                    "Two-tone by the alternating weave")],
            default='LOOP')
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.05, max=10.0,
            description="Overall size multiplier (1 = fit the 2 m "
                        "cube)")

        def execute(self, context):
            span = 2.0 * self.scale
            if self.output == 'CURVE':
                paths = cord_paths(self.substrate, self.iterations,
                                   self.cord_width, self.style,
                                   self.smoothness, self.weave_height)
                obj = kc._emit_curve(context, "Fractal Knotwork",
                                     paths, span=span, operator=self)
                if obj is None:
                    self.report({'ERROR'}, "no knotwork generated")
                    return {'CANCELLED'}
                obj["math_art_pattern"] = True
                self.report({'INFO'}, "%s  %d components" %
                            (self.substrate, len(paths)))
                return {'FINISHED'}
            cells = build_cells(self.substrate, self.iterations,
                                self.cord_width, self.style,
                                self.smoothness, self.interlace_mode,
                                self.weave_height, self.output,
                                self.tube_sides, self.color_by)
            obj = pc.emit(context, "Fractal Knotwork", cells,
                          separate=False, fit=True, span=span,
                          operator=self)
            if obj is None:
                self.report({'ERROR'}, "no knotwork generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if self.output == 'TUBE':
                _shade_smooth(obj)
            self.report({'INFO'}, "%s  %d components  V=%d F=%d" %
                        (self.substrate, len(cells),
                         len(obj.data.vertices),
                         len(obj.data.polygons)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'substrate')
            lay.prop(self, 'iterations')
            lay.prop(self, 'cord_width')
            lay.prop(self, 'style')
            if self.style == 'SMOOTH':
                lay.prop(self, 'smoothness')
            lay.prop(self, 'output')
            if self.output == 'RIBBON':
                lay.prop(self, 'interlace_mode')
                if self.interlace_mode == 'WOVEN':
                    lay.prop(self, 'weave_height')
            elif self.output == 'TUBE':
                lay.prop(self, 'tube_sides')
                lay.prop(self, 'weave_height')
            else:
                lay.prop(self, 'weave_height')
            if self.output != 'CURVE':
                lay.prop(self, 'color_by')
            lay.prop(self, 'scale')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.fractal_knotwork_add",
                             icon='OUTLINER_OB_CURVES')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_fractal_knotwork_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_fractal_knotwork_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _check_complex(kind, iterations):
    """The welded complex is a valid edge-to-edge cell complex: every
    edge is used by one face (boundary) or exactly two DISTINCT faces
    (interior), never more, after T-junction handling."""
    topo, n_split = build_topology(kind, iterations)
    ok = True
    for key, fs in topo['edge_faces'].items():
        if len(fs) not in (1, 2):
            ok = False
        if len(fs) == 2 and fs[0] == fs[1]:
            ok = False
    return ok, n_split, topo


def _check_cords(topo):
    """The medial successor is a bijection whose orbits partition all
    corner-segments into CLOSED cords."""
    darts = ck._tiling_all_darts(topo)
    succ = [ck._tiling_step(d, topo, set()) for d in darts]
    bij = len(set(succ)) == len(darts) and set(succ) == set(darts)
    cycles, n = ck._tiling_cycles(topo, set())
    covered = [d for cyc in cycles for d in cyc]
    part = len(covered) == n == len(darts) and len(set(covered)) == n
    closes = all(ck._tiling_step(cyc[-1], topo, set()) == cyc[0]
                 for cyc in cycles)
    return bij and part and closes


def _check_weave(cords, x):
    """Every crossing is ONE-over-ONE-under, and the over/under sense
    strictly alternates along every cord (no parity frustration)."""
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
    one_over = all(len(v) == 2 and v[0] != v[1]
                   for v in visits.values())
    return one_over, alternates, len(visits)


def _check_geometry(kind, iterations):
    """RIBBON (flat + woven) and TUBE geometry is finite, non-empty;
    the tube is ALL QUADS; and the fitted result spans the 2 m cube."""
    ok = True
    for output, mode, cby in (('RIBBON', 'FLAT', 'LOOP'),
                              ('RIBBON', 'WOVEN', 'CHECKER'),
                              ('TUBE', 'FLAT', 'LOOP')):
        cells = build_cells(kind, iterations, 0.32, 'SMOOTH', 6, mode,
                            0.25, output, 8, cby)
        faces = sum(len(c[1]) for c in cells)
        finite = all(all(np.isfinite(v).all() for v in c[0])
                     for c in cells)
        ok = ok and faces > 0 and finite
        if output == 'TUBE':
            quads = all(len(f) == 4 for c in cells for f in c[1])
            ok = ok and quads
            v, f, m = pc.merge_cells(cells)
            fitted = np.asarray(pc.center_scale(v, 2.0), float)
            span = max(float(np.ptp(fitted[:, 0])),
                       float(np.ptp(fitted[:, 1])))
            ok = (ok and abs(span - 2.0) < 1e-6
                  and float(np.abs(fitted).max()) <= 1.0 + 1e-6)
    curves = cord_paths(kind, iterations)
    ok = ok and len(curves) > 0 and all(
        np.isfinite(np.asarray(p, float)).all() for p, _c in curves)
    return ok


def _selftest():
    all_ok = True
    DEPTHS = {'KITE_R6': (1, 2, 3), 'KITE_R8': (1, 2, 3),
              'KITE_R12': (1, 2), 'SPIRAL_GOLDEN': (3,),
              'SPIRAL_GOLDEN_2': (3,), 'SPIRAL_EQUILATERAL': (3,)}
    print("-- fractal knotwork: welded complex + medial link --")
    for kind, _lab, _d in SUBSTRATE_ITEMS:
        prev_cross = prev_comp = None
        grow_ok = True
        for it in DEPTHS[kind]:
            cx_ok, n_split, topo = _check_complex(kind, it)
            # the kite f-tilings are exactly edge-to-edge (t-splits == 0);
            # the spiral-triangle patches may leave a few boundary
            # T-junctions the splitter repairs -> only require a VALID
            # complex.
            all_ok = all_ok and cx_ok and (
                n_split == 0 or kind.startswith('SPIRAL_'))
            closed_ok = _check_cords(topo)
            all_ok = all_ok and closed_ok
            net = trace_knotwork(kind, it)
            one_over, alt, ncross = _check_weave(net['cords'], net['x'])
            all_ok = all_ok and one_over
            # perfect alternation expected (mirror curves are
            # alternating); assert it whenever the parity solve found
            # no frustrated constraint, and assert none was found
            all_ok = all_ok and (net['clashes'] == 0) and alt
            if prev_cross is not None:
                grow_ok = (grow_ok and ncross > prev_cross
                           and net['components'] >= prev_comp)
            prev_cross, prev_comp = ncross, net['components']
            print("%-9s it=%d : complex=%-5s t-splits=%d closed=%-5s "
                  "components=%3d crossings=%4d 1over1under=%-5s "
                  "alternates=%-5s clashes=%d"
                  % (kind, it, cx_ok, n_split, closed_ok,
                     net['components'], ncross, one_over, alt,
                     net['clashes']))
        all_ok = all_ok and grow_ok
        print("%-9s deeper iterations add detail (self-similar "
              "growth): %s" % (kind, grow_ok))
    print("-- geometry (ribbon flat/woven, all-quad tube, 2m fit) --")
    for kind, it in (('KITE_R6', 2), ('KITE_R8', 1)):
        g = _check_geometry(kind, it)
        all_ok = all_ok and g
        print("%-9s it=%d : ribbon+tube+curve geometry ok=%s"
              % (kind, it, g))
    print("RESULT:", "OK" if all_ok else "BAD")
    assert all_ok
