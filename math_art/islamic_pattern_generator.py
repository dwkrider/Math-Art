
# Islamic Star Pattern Generator for Blender
#
# Geometric star patterns ("girih" strapwork) built by Hankin's
# "polygons in contact" (PIC) method.  An underlying tiling of regular
# polygons is laid down; on every edge, from its MIDPOINT, a pair of
# rays leaves at a fixed CONTACT ANGLE to the edge, one ray directed
# into each incident tile.  Within a tile the rays from its edges
# extend and meet -- each ray running until it reaches the next ray
# around the tile -- forming that tile's star motif.  Because the
# contact points sit at shared edge midpoints, the strapwork is
# continuous from tile to tile.  The contact angle is the primary
# knob: small angles give spiky acute stars, large angles obtuse ones.
#
# The substrate can be any of our uniform tilings (reused from
# tiling_generator): the square grid (4/8-point stars), the hexagonal
# tiling (6/12-point), the triangular tiling (3/6-point), the 4.8.8
# truncated-square tiling (the iconic 8-point star-and-cross), and the
# twelve-fold 3.12.12 and 4.6.12 tilings.
#
# Output is strapwork as flat RIBBONS (default), optionally given
# relief by extrusion to a height on an optional backing slab, in the
# modular-constructivist relief tradition of the Pattern Engine (see
# pattern_common.py).
#
# References:
#   E. H. Hankin, "The Drawing of Geometric Patterns in Saracenic Art"
#     (Memoirs of the Archaeological Survey of India, no. 15, 1925) --
#     the "polygons in contact" construction.
#   Craig S. Kaplan, "Computer Generated Islamic Star Patterns"
#     (Bridges 2000) -- the algorithmic PIC formulation reused here.
#   A. J. Lee, "Islamic Star Patterns" (Muqarnas 4, 1987).
#   Jay Bonner, "Islamic Geometric Patterns: Their Historical
#     Development and Traditional Methods of Construction" (2017).
#   Branko Grunbaum & G. C. Shephard, "Tilings and Patterns" (1987) --
#     the uniform tilings used as PIC substrates.

bl_info = {
    "name": "Islamic Star Pattern",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Patterns",
    "description": "Polygons-in-contact Islamic star patterns "
                   "(strapwork ribbons, optional relief)",
    "category": "Add Mesh",
}

from math import cos, sin, radians, hypot
import numpy as np

try:
    from . import pattern_common as pc
    from . import tiling_generator as tg
except Exception:                       # legacy single-file / CLI use
    import pattern_common as pc
    import tiling_generator as tg


# --------------------------------------------------------------------
# Substrates: reuse tiling_generator's regular-polygon tilings
# --------------------------------------------------------------------
#
# Every substrate here is one of tiling_generator's base tilings (all
# regular polygons, edge length 1, sharing whole edges), so the PIC
# construction lands clean contact points on the shared edges.

SUBSTRATE_ITEMS = [
    ('SQUARE', "Square (4.4.4.4)",
     "Square grid -> 4/8-point stars"),
    ('TRI', "Triangular (3.3.3.3.3.3)",
     "Triangular tiling -> 3/6-point stars"),
    ('HEX', "Hexagonal (6.6.6)",
     "Hexagonal tiling -> 6/12-point stars"),
    ('TRUNCSQ', "Truncated Square (4.8.8)",
     "Octagon-square tiling -> the iconic 8-point star-and-cross"),
    ('TRUNCHEX', "Truncated Hexagonal (3.12.12)",
     "Dodecagon-triangle tiling -> 12-point stars"),
    ('TRUNCTRIHEX', "Truncated Trihexagonal (4.6.12)",
     "Dodecagon-hexagon-square tiling -> 12-point stars"),
]

# Contact angle (degrees) giving the classic motif for each substrate.
DEFAULT_CONTACT = {
    'SQUARE': 30.0, 'TRI': 30.0, 'HEX': 30.0,
    'TRUNCSQ': 30.0, 'TRUNCHEX': 30.0, 'TRUNCTRIHEX': 30.0,
}

# Historical presets: (substrate, contact angle).
PRESET_ITEMS = [
    ('CUSTOM', "Custom", "Use the substrate and contact angle below"),
    ('STARCROSS8', "8-fold Star-and-Cross (4.8.8)",
     "The classic octagon star with square crosses"),
    ('TWELVE', "12-fold Star (3.12.12)",
     "Twelve-point dodecagon stars"),
    ('SIX', "6-fold Star (hexagonal)",
     "Six-point stars on the hexagonal tiling"),
    ('FOUREIGHT', "4/8 Star (square)",
     "Eight-point stars on the square grid"),
]
PRESETS = {
    'STARCROSS8': ('TRUNCSQ', 30.0),
    'TWELVE':     ('TRUNCHEX', 30.0),
    'SIX':        ('HEX', 30.0),
    'FOUREIGHT':  ('SQUARE', 30.0),
}

# Material slot per polygon side count, for the STAR_ORDER color mode.
_ORDER_MAT = {3: 0, 4: 1, 5: 2, 6: 3, 8: 4, 12: 5}
# Backing slab uses the last palette slot (a neutral gray).
_BACKING_MAT = len(pc.PALETTE_RGBA) - 1


# --------------------------------------------------------------------
# Polygons in contact (Hankin / Kaplan)
# --------------------------------------------------------------------

def _ray_param(o0, d0, o1, d1):
    """Parameter s>=0 along ray o0 + s*d0 where it crosses ray
    o1 + t*d1 with t>=0, or None.  Both rays are half-lines."""
    det = d0[0] * (-d1[1]) - (-d1[0]) * d0[1]
    if abs(det) < 1e-12:
        return None
    bx, by = o1[0] - o0[0], o1[1] - o0[1]
    s = (bx * (-d1[1]) - (-d1[0]) * by) / det
    t = (d0[0] * by - bx * d0[1]) / det
    if s <= 1e-7 or t <= 1e-7:
        return None
    return s


def star_segments(poly, angle_deg):
    """PIC strapwork segments for one convex tile (Hankin / Kaplan).

    From each edge midpoint two rays leave into the tile, each making
    `angle_deg` with the edge (symmetric about the inward normal): one
    "left" ray turning toward the next vertex and one "right" ray
    toward the previous vertex.  Each ray is truncated at its nearest
    intersection with a ray of the OPPOSITE hand -- Kaplan's standard
    construction.  A left ray of one edge and a right ray of another
    are mirror images across a polygon symmetry axis, so they always
    meet mutually on that axis, knitting a clean star: small angles
    give grazing rays and blunt motifs; larger angles cross deeper
    into acute, many-pointed stars.  Returns ((x0, y0), (x1, y1))
    segments whose union is the closed motif, every segment starting
    at an edge midpoint (the contact point that guarantees continuity
    across the shared edge).
    """
    theta = radians(angle_deg)
    ct, st = cos(theta), sin(theta)
    poly = np.asarray(poly, float)
    n = len(poly)
    c = poly.mean(axis=0)
    left, right = [], []                         # (origin, direction)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        m = 0.5 * (a + b)
        e = b - a
        e = e / (hypot(e[0], e[1]) + 1e-12)      # along-edge unit
        nrm = c - m
        nrm = nrm / (hypot(nrm[0], nrm[1]) + 1e-12)   # inward unit
        left.append((m, ct * e + st * nrm))      # turns toward vtx i+1
        right.append((m, -ct * e + st * nrm))    # turns toward vtx i
    segs = []
    for src, others in ((left, right), (right, left)):
        for o0, d0 in src:
            best = None
            for o1, d1 in others:
                s = _ray_param(o0, d0, o1, d1)
                if s is not None and (best is None or s < best):
                    best = s
            if best is None:
                continue
            tip = (o0[0] + best * d0[0], o0[1] + best * d0[1])
            segs.append((tuple(o0), tip))
    return segs


# --------------------------------------------------------------------
# Substrate + motif assembly
# --------------------------------------------------------------------

def _clip_segment(p0, p1, rect):
    """Liang-Barsky clip of segment p0->p1 to axis-aligned rectangle
    rect = (x0, y0, x1, y1).  Returns (a, b) clipped, or None."""
    x0, y0, x1, y1 = rect
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]),
                 (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if abs(p) < 1e-15:
            if q < 0.0:
                return None                       # parallel, outside
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None
            if r < t1:
                t1 = r
    if t0 > t1:
        return None
    a = (p0[0] + t0 * dx, p0[1] + t0 * dy)
    b = (p0[0] + t1 * dx, p0[1] + t1 * dy)
    if hypot(b[0] - a[0], b[1] - a[1]) < 1e-9:
        return None
    return a, b


def tile_motifs(substrate, nx, ny, contact_deg, trim=False, pad=2):
    """Return (motifs, rect): one (poly, order, segments) per substrate
    tile, where `order` is the polygon side count and `segments` are
    that tile's PIC strapwork segments (clipped to `rect` when trim).
    `rect` is the trim rectangle or None."""
    NX, NY = (nx + pad, ny + pad) if trim else (nx, ny)
    placed = list(tg._base_iter(substrate, NX, NY))
    rect = None
    if trim:
        rect = tg._trim_rect([p for p, _ in placed], nx, ny, pad)
    motifs = []
    for poly, _ti in placed:
        segs = star_segments(poly, contact_deg)
        if rect is not None:
            clipped = []
            for a, b in segs:
                cs = _clip_segment(a, b, rect)
                if cs is not None:
                    clipped.append(cs)
            segs = clipped
        if not segs:
            continue
        motifs.append((np.asarray(poly, float), len(poly), segs))
    return motifs, rect


# --------------------------------------------------------------------
# Strapwork arrangement: continuous bands across the whole patch
# --------------------------------------------------------------------
#
# The PIC construction hands us short line segments, one per ray, each
# starting at an edge midpoint (a contact point).  Rendering each
# segment as its own rectangle would leave the strapwork as disjoint
# quads.  Instead we glue the segments into an ARRANGEMENT -- endpoints
# snapped to shared nodes -- and trace the maximal "straightest-
# through" bands that weave from tile to tile: a band crosses each
# contact point straight (the two adjacent tiles' rays are collinear
# there) and bends at the star points, exactly the classic interlaced
# strapwork topology.  Each traced band is then built as ONE continuous
# mitered ribbon of shared vertices, not a pile of rectangles.

def _unit(x, y):
    L = hypot(x, y)
    if L < 1e-12:
        return (0.0, 0.0)
    return (x / L, y / L)


def _flatten_segments(motifs):
    """Flatten per-tile motifs into parallel lists (segments, orders,
    tiles): each strapwork segment with the side count and index of the
    tile whose PIC motif produced it (used for coloring)."""
    segments, orders, tiles = [], [], []
    for ti, (_poly, order, segs) in enumerate(motifs):
        for a, b in segs:
            segments.append((a, b))
            orders.append(order)
            tiles.append(ti)
    return segments, orders, tiles


class _NodeSet:
    """Snap 2D points to shared node ids within `tol` via a small
    spatial hash, so coincident contact points from neighboring tiles
    merge to one node (the seam that makes bands continuous)."""

    def __init__(self, tol=1e-6):
        self.tol = tol
        self.pts = []
        self.grid = {}

    def add(self, p):
        t = self.tol
        gx, gy = round(p[0] / t), round(p[1] / t)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in self.grid.get((gx + dx, gy + dy), ()):
                    q = self.pts[idx]
                    if abs(q[0] - p[0]) <= t and abs(q[1] - p[1]) <= t:
                        return idx
        idx = len(self.pts)
        self.pts.append((float(p[0]), float(p[1])))
        self.grid.setdefault((gx, gy), []).append(idx)
        return idx


def _pair_at_node(incident):
    """Pair a node's incident half-edges "straightest-through": match
    the two directions closest to anti-parallel first (greedy), so a
    band continues as straight as it can.  `incident` is a list of
    (seg_index, unit_direction_pointing_away_from_node).  Returns a
    dict seg_index -> partner seg_index; an unpaired segment is absent
    and terminates its band there (a flat cap)."""
    n = len(incident)
    pairs = []
    for i in range(n):
        di = incident[i][1]
        for j in range(i + 1, n):
            dj = incident[j][1]
            pairs.append((di[0] * dj[0] + di[1] * dj[1], i, j))
    pairs.sort()                                  # most anti-parallel first
    taken = [False] * n
    out = {}
    for _dot, i, j in pairs:
        if taken[i] or taken[j]:
            continue
        taken[i] = taken[j] = True
        out[incident[i][0]] = incident[j][0]
        out[incident[j][0]] = incident[i][0]
    return out


def build_arrangement(segments, tol=1e-6):
    """Snap all segment endpoints to shared nodes and pair the half-
    edges at every node.  Returns (pts, seg_nodes, pair): node
    coordinates, per-segment (na, nb) node ids (None for degenerate),
    and node -> {seg: partner_seg}."""
    ns = _NodeSet(tol)
    seg_nodes = []
    for a, b in segments:
        na, nb = ns.add(a), ns.add(b)
        seg_nodes.append(None if na == nb else (na, nb))
    incident = {}
    for si, nn in enumerate(seg_nodes):
        if nn is None:
            continue
        na, nb = nn
        pa, pb = ns.pts[na], ns.pts[nb]
        d = _unit(pb[0] - pa[0], pb[1] - pa[1])
        incident.setdefault(na, []).append((si, d))
        incident.setdefault(nb, []).append((si, (-d[0], -d[1])))
    pair = {node: _pair_at_node(lst) for node, lst in incident.items()}
    return ns.pts, seg_nodes, pair


def trace_bands(seg_nodes, pair):
    """Trace maximal bands through the arrangement.  Each band is
    (node_path, seg_path): the ordered node ids of its polyline and the
    segment ids it walks.  A closed band repeats its first node last."""
    def other(seg, node):
        a, b = seg_nodes[seg]
        return b if a == node else a

    used = [False] * len(seg_nodes)
    bands = []
    for s0 in range(len(seg_nodes)):
        if seg_nodes[s0] is None or used[s0]:
            continue
        a0, b0 = seg_nodes[s0]
        node_path, seg_path, local = [a0, b0], [s0], {s0}
        for start_node, forward in ((b0, True), (a0, False)):
            cur_seg, cur_node = s0, start_node
            while True:
                nxt = pair.get(cur_node, {}).get(cur_seg)
                if nxt is None or nxt in local:
                    break                          # dead end or closed loop
                nn = other(nxt, cur_node)
                if forward:
                    node_path.append(nn)
                    seg_path.append(nxt)
                else:
                    node_path.insert(0, nn)
                    seg_path.insert(0, nxt)
                local.add(nxt)
                cur_seg, cur_node = nxt, nn
        for s in local:
            used[s] = True
        bands.append((node_path, seg_path))
    return bands


def _miter(np_, nn, limit):
    """Miter offset direction (unit bisector) and length scale for a
    joint between offset normals `np_` (incoming) and `nn` (outgoing).
    scale = 1/cos(half-angle), clamped to +/-`limit` so a sharp turn
    cannot fire off an inverting spike."""
    mx, my = np_[0] + nn[0], np_[1] + nn[1]
    L = hypot(mx, my)
    if L < 1e-9:                                   # ~180 deg reversal
        return nn[0], nn[1], 1.0
    mx, my = mx / L, my / L
    cos_half = mx * nn[0] + my * nn[1]
    scale = limit if abs(cos_half) < 1e-6 else 1.0 / cos_half
    scale = max(-limit, min(limit, scale))
    return mx, my, scale


def miter_ribbon(points, width, closed, limit=4.0):
    """Offset a polyline left and right by width/2 with mitered joints.
    Returns (left, right): two point lists parallel to `points`, with
    interior vertices mitered and (open) ends cut flat."""
    w = 0.5 * width
    V = list(points)
    k = len(V)
    left, right = [], []
    if closed:
        d = [_unit(V[(i + 1) % k][0] - V[i][0],
                   V[(i + 1) % k][1] - V[i][1]) for i in range(k)]
        for i in range(k):
            dp, dn = d[(i - 1) % k], d[i]
            mx, my, s = _miter((-dp[1], dp[0]), (-dn[1], dn[0]), limit)
            left.append((V[i][0] + mx * w * s, V[i][1] + my * w * s))
            right.append((V[i][0] - mx * w * s, V[i][1] - my * w * s))
    else:
        d = [_unit(V[i + 1][0] - V[i][0], V[i + 1][1] - V[i][1])
             for i in range(k - 1)]
        for i in range(k):
            if i == 0:
                dn = d[0]
                mx, my, s = -dn[1], dn[0], 1.0     # flat cap
            elif i == k - 1:
                dp = d[-1]
                mx, my, s = -dp[1], dp[0], 1.0     # flat cap
            else:
                dp, dn = d[i - 1], d[i]
                mx, my, s = _miter((-dp[1], dp[0]),
                                   (-dn[1], dn[0]), limit)
            left.append((V[i][0] + mx * w * s, V[i][1] + my * w * s))
            right.append((V[i][0] - mx * w * s, V[i][1] - my * w * s))
    return left, right


def band_ribbon_faces(left, right, closed, height):
    """Build one continuous ribbon cell (verts, faces, mats-less) from
    the left/right boundaries.  Flat (height <= 0) is a single strip of
    top quads sharing vertices along the band; with relief it is a
    watertight strip -- top, bottom, and both side walls, plus flat end
    caps when open."""
    cv, cf = [], []
    m = len(left)
    if m < 2:
        return cv, cf
    relief = height > 0.0
    z_top = height if relief else 0.0

    def addv(pt, z):
        cv.append((float(pt[0]), float(pt[1]), float(z)))
        return len(cv) - 1

    TL = [addv(left[i], z_top) for i in range(m)]
    TR = [addv(right[i], z_top) for i in range(m)]
    span = m if closed else m - 1

    def nxt(i):
        return (i + 1) % m if closed else i + 1

    for i in range(span):
        j = nxt(i)
        cf.append((TL[i], TR[i], TR[j], TL[j]))
    if relief:
        BL = [addv(left[i], 0.0) for i in range(m)]
        BR = [addv(right[i], 0.0) for i in range(m)]
        for i in range(span):
            j = nxt(i)
            cf.append((BL[j], BR[j], BR[i], BL[i]))       # bottom
            cf.append((TL[j], BL[j], BL[i], TL[i]))       # left wall
            cf.append((TR[i], BR[i], BR[j], TR[j]))       # right wall
        if not closed:
            cf.append((TL[0], BL[0], BR[0], TR[0]))       # start cap
            cf.append((TR[m - 1], BR[m - 1], BL[m - 1], TL[m - 1]))  # end
    return cv, cf


def _band_kind(color_by, band_index, seg_path, orders):
    """Material index for a whole band, per color mode.  A band spans
    many tiles, so STAR_ORDER colors by the dominant star order along
    the band, TILE colors per band, and BAND gives every continuous
    ribbon its own palette color (band index modulo the palette)."""
    if color_by == 'STAR_ORDER':
        from collections import Counter
        order = Counter(orders[s] for s in seg_path).most_common(1)[0][0]
        return _ORDER_MAT.get(order, 6)
    if color_by == 'BAND':
        return band_index % len(pc.PALETTE_RGBA)  # a color per ribbon
    if color_by == 'TILE':
        return band_index % _BACKING_MAT          # keep clear of backing
    return 0                                      # UNIFORM


def _cr_point(p0, p1, p2, p3, t):
    """Uniform Catmull-Rom interpolation between p1 and p2."""
    t2, t3 = t * t, t * t * t
    return tuple(0.5 * (2.0 * p1
                        + (-p0 + p2) * t
                        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3))


def catmull_rom(points, closed, subdiv):
    """Smooth a band polyline with a uniform Catmull-Rom spline sampled
    `subdiv` times per segment.  The curve INTERPOLATES every control
    point, so the contact-point nodes shared with neighboring bands stay
    exactly anchored and the strapwork remains continuous across edges."""
    n = len(points)
    if n < 3 or subdiv < 2:
        return [tuple(p) for p in points]
    P = [np.asarray(p, float) for p in points]
    if closed:
        def get(i):
            return P[i % n]
        segs = range(n)
    else:
        def get(i):
            return P[min(max(i, 0), n - 1)]
        segs = range(n - 1)
    out = []
    for i in segs:
        p0, p1, p2, p3 = get(i - 1), get(i), get(i + 1), get(i + 2)
        for t in range(subdiv):
            out.append(_cr_point(p0, p1, p2, p3, t / subdiv))
    if not closed:
        out.append(tuple(P[-1]))                  # anchor the far end
    return out


def strapwork_bands(substrate, nx, ny, contact_deg, ribbon_width,
                    trim=False, curved=False, subdiv=8):
    """Trace the continuous strapwork bands of a patch.  Returns
    (bands, orders): each band is
    (left, right, closed, seg_path, path) -- the mitered ribbon
    boundaries, whether it is a closed loop, the segment ids it walks,
    and the centerline polyline (Catmull-Rom smoothed when `curved`)."""
    motifs, _rect = tile_motifs(substrate, nx, ny, contact_deg, trim)
    segments, orders, _tiles = _flatten_segments(motifs)
    if not segments:
        return [], orders
    pts, seg_nodes, pair = build_arrangement(segments)
    bands = []
    for node_path, seg_path in trace_bands(seg_nodes, pair):
        closed = len(node_path) >= 4 and node_path[0] == node_path[-1]
        poly = [pts[n] for n in node_path]
        if closed:
            poly = poly[:-1]
        if len(poly) < 2:
            continue
        path = catmull_rom(poly, closed, subdiv) if curved else poly
        left, right = miter_ribbon(path, ribbon_width, closed)
        bands.append((left, right, closed, seg_path, path))
    return bands, orders


def build_cells(substrate, nx, ny, contact_deg, ribbon_width,
                color_by='UNIFORM', trim=False, height=0.0,
                backing=False, base=0.08, curved=False, subdiv=8):
    """One (verts, faces, mats) cell per continuous strapwork BAND: the
    band as a single mitered ribbon (flat, or extruded to `height`;
    straight-mitered or Catmull-Rom curved).  An optional backing slab
    is appended as a final cell."""
    bands, orders = strapwork_bands(
        substrate, nx, ny, contact_deg, ribbon_width, trim,
        curved, subdiv)
    cells = []
    all_verts = []
    for bi, (left, right, closed, seg_path, _path) in enumerate(bands):
        cv, cf = band_ribbon_faces(left, right, closed, height)
        if not cf:
            continue
        kind = _band_kind(color_by, bi, seg_path, orders)
        cm = [kind] * len(cf)
        cells.append((cv, cf, cm))
        all_verts.extend(cv)
    if backing and height > 0.0 and all_verts:
        a = np.asarray(all_verts, float)
        lo = (a[:, 0].min(), a[:, 1].min())
        hi = (a[:, 0].max(), a[:, 1].max())
        cv, cf, cm = [], [], []
        pc.slab(cv, cf, cm, lo, hi, 0.0, -base, mat=_BACKING_MAT)
        cells.append((cv, cf, cm))
    return cells


def build(substrate, nx, ny, contact_deg, ribbon_width,
          color_by='UNIFORM', trim=False, height=0.0,
          backing=False, base=0.08, curved=False, subdiv=8):
    """Merged (verts, faces, mats) for one strapwork patch."""
    return pc.merge_cells(build_cells(
        substrate, nx, ny, contact_deg, ribbon_width, color_by,
        trim, height, backing, base, curved, subdiv))


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

    def _emit_curve(context, name, bands, span=2.0, operator=None):
        """Build a single Blender curve object whose POLY splines are
        the band centerlines (one spline per continuous ribbon path),
        centered and scaled to the `span` cube.  Placed via the
        operator's Align, matching the mesh path."""
        paths = [(list(path), closed)
                 for (_l, _r, closed, _sp, path) in bands
                 if len(path) >= 2]
        if not paths:
            return None
        allv = np.asarray([p for path, _c in paths for p in path], float)
        lo, hi = allv.min(axis=0), allv.max(axis=0)
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        s = span / max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '3D'
        for path, closed in paths:
            sp = cu.splines.new('POLY')
            sp.points.add(len(path) - 1)
            for i, (x, y) in enumerate(path):
                sp.points[i].co = ((x - cx) * s, (y - cy) * s, 0.0, 1.0)
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

    class MESH_OT_islamic_pattern_add(bpy.types.Operator,
                                      AddObjectHelper):
        """Add an Islamic star pattern (polygons in contact)"""
        bl_idname = "mesh.islamic_pattern_add"
        bl_label = "Islamic Star Pattern"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset", items=PRESET_ITEMS, default='STARCROSS8',
            description="Historical star pattern; Custom exposes the "
                        "substrate and contact angle")
        substrate: EnumProperty(
            name="Substrate", items=SUBSTRATE_ITEMS, default='TRUNCSQ',
            description="Underlying tiling of regular polygons")
        contact_angle: FloatProperty(
            name="Contact Angle", default=30.0, min=10.0, max=80.0,
            description="Angle (degrees) of the rays to each edge; the "
                        "primary knob -- small = spiky, large = obtuse")
        nx: IntProperty(name="Cells X", default=5, min=1, max=40)
        ny: IntProperty(name="Cells Y", default=5, min=1, max=40)
        trim: BoolProperty(
            name="Trim Boundary", default=True,
            description="Clip the strapwork to a clean central "
                        "rectangle, removing the ragged edge")
        ribbon_width: FloatProperty(
            name="Ribbon Width", default=0.14, min=0.01, max=0.6,
            description="Width of the strapwork ribbons (edge units)")
        color_by: EnumProperty(
            name="Color By",
            items=[('UNIFORM', "Uniform", "A single material"),
                   ('TILE', "By Tile",
                    "Material per substrate polygon"),
                   ('STAR_ORDER', "By Star Order",
                    "Material per polygon side count (star order)"),
                   ('BAND', "By Ribbon",
                    "A distinct color per continuous strapwork band")],
            default='STAR_ORDER')
        curved: BoolProperty(
            name="Curved Ribbons", default=False,
            description="Flow each band along a Catmull-Rom spline "
                        "through its nodes instead of straight segments")
        smoothness: IntProperty(
            name="Smoothness", default=8, min=2, max=32,
            description="Spline subdivisions per band segment "
                        "(curved ribbons only)")
        output: EnumProperty(
            name="Output",
            items=[('RIBBON', "Ribbon Mesh",
                    "Filled strapwork ribbons (supports relief)"),
                   ('CURVE', "Centerline Curves",
                    "Band centerlines as a Blender curve object")],
            default='RIBBON',
            description="Build filled ribbon meshes or the band "
                        "centerlines as curve splines")
        height: FloatProperty(
            name="Relief Height", default=0.0, min=0.0, max=2.0,
            description="0 = flat ribbons; > 0 extrudes the strapwork")
        backing: BoolProperty(
            name="Backing Slab", default=False,
            description="Add a slab behind raised strapwork (relief "
                        "only)")
        base: FloatProperty(
            name="Base Thickness", default=0.08, min=0.01, max=1.0)
        separate: BoolProperty(
            name="Separate Motifs", default=False,
            description="Output each tile's motif as its own mesh "
                        "object (parented to an empty)")

        def _resolved(self):
            """(substrate, contact angle) after applying any preset."""
            if self.preset != 'CUSTOM':
                return PRESETS[self.preset]
            return self.substrate, self.contact_angle

        def execute(self, context):
            substrate, contact = self._resolved()
            if self.output == 'CURVE':
                bands, _orders = strapwork_bands(
                    substrate, self.nx, self.ny, contact,
                    self.ribbon_width, self.trim, self.curved,
                    self.smoothness)
                obj = _emit_curve(context, "Islamic Star", bands,
                                  operator=self)
                if obj is None:
                    self.report({'ERROR'}, "no pattern generated")
                    return {'CANCELLED'}
                obj["math_art_pattern"] = True
                self.report({'INFO'}, "%s theta=%.0f  %d bands" %
                            (substrate, contact, len(bands)))
                return {'FINISHED'}
            cells = build_cells(
                substrate, self.nx, self.ny, contact,
                self.ribbon_width, self.color_by, self.trim,
                self.height, self.backing, self.base, self.curved,
                self.smoothness)
            obj = pc.emit(context, "Islamic Star", cells,
                          self.separate, fit=True, operator=self)
            if obj is None:
                self.report({'ERROR'}, "no pattern generated")
                return {'CANCELLED'}
            obj["math_art_pattern"] = True
            if obj.type == 'MESH':
                self.report({'INFO'}, "%s theta=%.0f  V=%d F=%d" %
                            (substrate, contact,
                             len(obj.data.vertices),
                             len(obj.data.polygons)))
            else:
                self.report({'INFO'}, "%s  %d motifs" %
                            (substrate, len(obj.children)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'CUSTOM':
                lay.prop(self, 'substrate')
                lay.prop(self, 'contact_angle')
            lay.prop(self, 'nx')
            lay.prop(self, 'ny')
            lay.prop(self, 'trim')
            lay.prop(self, 'ribbon_width')
            lay.prop(self, 'curved')
            if self.curved:
                lay.prop(self, 'smoothness')
            lay.prop(self, 'output')
            if self.output == 'RIBBON':
                lay.prop(self, 'color_by')
                lay.prop(self, 'height')
                if self.height > 0.0:
                    lay.prop(self, 'backing')
                    if self.backing:
                        lay.prop(self, 'base')
                lay.prop(self, 'separate')
            lay.prop(self, 'align')

    def _menu_func(self, context):
        self.layout.operator("mesh.islamic_pattern_add",
                             icon='SOLO_ON')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_islamic_pattern_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_islamic_pattern_add)


# --------------------------------------------------------------------
# Self-test (pure Python)
# --------------------------------------------------------------------

def _poly_area(pts):
    """Signed area of a 2D polygon (shoelace)."""
    a = 0.0
    n = len(pts)
    for k in range(n):
        x0, y0 = pts[k]
        x1, y1 = pts[(k + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def _patch_bands(name, contact):
    """(pts, seg_nodes, pair, traced, tiles) for a 4x4 untrimmed patch --
    the exact-contact arrangement used by the continuity tests."""
    placed = list(tg._base_iter(name, 4, 4))
    motifs = [(np.asarray(p, float), len(p), star_segments(p, contact))
              for p, _ in placed]
    segments, _orders, tiles = _flatten_segments(motifs)
    pts, seg_nodes, pair = build_arrangement(segments)
    return pts, seg_nodes, pair, trace_bands(seg_nodes, pair), tiles


def _self_test():
    ok = True
    print("substrate    tiles  segs bands maxseg xtile  cont "
          "shared  quads  miter")
    for name in ('SQUARE', 'TRI', 'HEX', 'TRUNCSQ', 'TRUNCHEX',
                 'TRUNCTRIHEX'):
        contact = DEFAULT_CONTACT[name]
        placed = list(tg._base_iter(name, 4, 4))
        pts, seg_nodes, pair, traced, tiles = _patch_bands(name, contact)
        nsegs = sum(1 for s in seg_nodes if s is not None)

        # (1) continuity: bands are multi-segment (the strapwork
        # actually continues) and at least one band weaves across a
        # tile boundary (spans >= 2 distinct substrate tiles).
        maxseg = 0
        xtile = 0
        n_multi = 0
        longest = None
        for node_path, seg_path in traced:
            nseg = len(seg_path)
            nt = len(set(tiles[s] for s in seg_path))
            maxseg = max(maxseg, nseg)
            xtile = max(xtile, nt)
            if nseg >= 2:
                n_multi += 1
            if longest is None or nseg > len(longest[1]):
                longest = (node_path, seg_path)
        cont_ok = maxseg >= 2 and xtile >= 2 and n_multi > 0

        # (2)/(3)/(4) on the longest band: ONE continuous ribbon
        # (vertices shared along the path -- 2 per station, not 4 per
        # segment); every ribbon quad non-degenerate; miters do not
        # invert (all quads keep one orientation, width stays bounded).
        shared_ok = quad_ok = miter_ok = False
        if longest is not None:
            node_path, seg_path = longest
            closed = len(node_path) >= 4 and node_path[0] == node_path[-1]
            poly = [pts[n] for n in node_path]
            if closed:
                poly = poly[:-1]
            left, right = miter_ribbon(poly, 0.14, closed)
            cv, cf = band_ribbon_faces(left, right, closed, 0.0)
            want_faces = len(poly) if closed else len(poly) - 1
            shared_ok = (len(cv) == 2 * len(poly)
                         and len(cf) == want_faces
                         and len(cv) < 4 * len(seg_path))
            areas = [_poly_area([cv[i][:2] for i in f]) for f in cf]
            quad_ok = bool(areas) and all(abs(a) > 1e-9 for a in areas)
            same_sign = all(a > 0 for a in areas) or all(a < 0
                                                         for a in areas)
            widths = [hypot(left[i][0] - right[i][0],
                            left[i][1] - right[i][1])
                      for i in range(len(poly))]
            miter_ok = (same_sign and min(widths) > 1e-6
                        and max(widths) <= 0.14 * 4.0 + 1e-6)

        good = cont_ok and shared_ok and quad_ok and miter_ok
        ok = ok and good
        print("%-11s  %5d %5d %5d %6d %5d   %-4s  %-5s  %-5s  %-5s %s" %
              (name, len(placed), nsegs, len(traced), maxseg, xtile,
               cont_ok, shared_ok, quad_ok, miter_ok,
               "OK" if good else "BAD"))

    # (5) curved spline: Catmull-Rom densifies each band yet keeps every
    # node -- especially the shared contact points -- exactly anchored,
    # so curved bands still meet across edges.
    _pts, _sn, _pr, traced, _tiles = _patch_bands('HEX', 30.0)
    anchor_err = 0.0
    grew = True
    for node_path, seg_path in traced:
        closed = len(node_path) >= 4 and node_path[0] == node_path[-1]
        poly = [_pts[n] for n in node_path]
        if closed:
            poly = poly[:-1]
        if len(poly) < 3:
            continue
        sm = catmull_rom(poly, closed, 6)
        if len(sm) <= len(poly):
            grew = False
        if not closed:
            for i, p in enumerate(poly):
                q = sm[i * 6]
                anchor_err = max(anchor_err,
                                 hypot(q[0] - p[0], q[1] - p[1]))
            anchor_err = max(anchor_err,
                             hypot(sm[-1][0] - poly[-1][0],
                                   sm[-1][1] - poly[-1][1]))
    curve_ok = grew and anchor_err < 1e-9
    ok = ok and curve_ok
    print("curved spline  anchor.err=%.2e  grew=%s  %s" %
          (anchor_err, grew, "OK" if curve_ok else "BAD"))

    # (6) color modes all build; BAND gives many distinct ribbon colors,
    # all within the palette.
    color_ok = True
    for cb in ('UNIFORM', 'TILE', 'STAR_ORDER', 'BAND'):
        cells = build_cells('TRUNCSQ', 4, 4, 25.0, 0.14, cb, trim=True)
        mats = set(m for _v, _f, cm in cells for m in cm)
        if not cells or not all(c[1] for c in cells) \
                or any(m >= len(pc.PALETTE_RGBA) for m in mats):
            color_ok = False
    band_cells = build_cells('TRUNCSQ', 4, 4, 30.0, 0.14, 'BAND',
                             trim=True)
    band_colors = len(set(m for _v, _f, cm in band_cells for m in cm))
    color_ok = color_ok and band_colors > 1
    ok = ok and color_ok
    print("color modes    band colors=%d  %s" %
          (band_colors, "OK" if color_ok else "BAD"))

    # (7) trimmed / curved / relief+backing builds are all non-empty.
    flat = build_cells('TRUNCSQ', 4, 4, 25.0, 0.14, 'STAR_ORDER',
                       trim=True)
    curved = build_cells('TRUNCSQ', 3, 3, 30.0, 0.14, 'UNIFORM',
                         trim=True, curved=True, subdiv=6)
    relief = build_cells('HEX', 3, 3, 30.0, 0.14, 'STAR_ORDER',
                         trim=True, height=0.3, backing=True)
    build_ok = all(len(c) > 0 and all(x[1] for x in c)
                   for c in (flat, curved, relief))
    ok = ok and build_ok
    print("builds         flat=%d curved=%d relief=%d  %s" %
          (len(flat), len(curved), len(relief),
           "OK" if build_ok else "BAD"))

    print("RESULT:", "OK" if ok else "BAD")
    return ok


if __name__ == "__main__":
    _self_test()
