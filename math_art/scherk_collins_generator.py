
# Scherk-Collins Sculpture Generator for Blender
#
# A re-implementation of the geometry engine of Carlo Sequin's
# "Sculpture Generator I" (1996/97) as a Blender add-on.
#
#   C. H. Sequin, "Virtual Prototyping of Scherk-Collins Saddle Rings",
#       Leonardo, Vol 30, No 2, pp 89-96, 1997.
#   C. H. Sequin, H. Meshkin, L. Downs, "Interactive Generation of
#       Scherk-Collins Sculptures", Proc. I3D '97.
#
# Only the sculpture *geometry* is reproduced (per project scope);
# materials, textures and backgrounds are left to Blender.
#
# The tower is based on the exact singly-periodic Scherk minimal surface
#   sin(z) = sinh(x) * sinh(y)
# parametrized per height-level c = sin(z) as
#   x(s) = asinh(sqrt(c) * e^( s*L)),   y(s) = asinh(sqrt(c) * e^(-s*L)),
#   L = ln(sinh(W)/sqrt(c)),   s in [-1, 1]
# which lands the curve ends exactly on the truncation planes x = W / y = W
# (W = "flange" slider; holes break open for W < asinh(1) ~ 0.881, which is
# why the original slider bottoms out at 0.7).
# Higher-order saddles ("branches" b) come from compressing the 90-degree
# wedge of that curve into a 180/b-degree wedge; consecutive storeys are
# rotated by 180/b degrees (so a closed warp=360 ring joins smoothly iff
# (twist + storeys*180/b) mod (360/b) == 0 -- verified against all demo
# files shipped with the original program).
#
# References:
#   H. F. Scherk, "Bemerkungen ueber die kleinste Flaeche innerhalb
#       gegebener Grenzen", J. reine angew. Math. (Crelle) 13, 1835
#       -- the singly-periodic saddle-tower minimal surface.
#   Sculptural form after the collaboration of sculptor Brent Collins
#       and Carlo H. Sequin (see the Leonardo and I3D papers above).

bl_info = {
    "name": "Scherk-Collins Sculpture Generator",
    "author": "Math Art project (after Carlo H. Sequin's Sculpture Generator I)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Scherk-Collins Sculpture / N-panel 'Scherk'",
    "description": "Generate Scherk-Collins saddle-chain toroid sculptures",
    "category": "Add Mesh",
}

import math
from math import sin, cos, pi, sqrt, sinh, asinh, atan2, exp, log, radians, hypot

# Matches the proportions of the original program: a storey of the exact
# Scherk tower (period pi) is displayed with height == the "height" slider
# value, at its natural aspect ratio when height = 1.5.
XY_SCALE = 1.5 / pi
WELD_EPS_BASE = 1e-5


def fit_transform(verts, global_scale=1.0, span=2.0):
    """Return (center, factor) that, applied as (v - center) * factor,
    centres `verts` on the origin and scales the largest bounding-box
    extent to `span` (the 2 m cube convention), then multiplies by
    `global_scale` -- so global_scale = 1 fits the sculpture exactly
    inside a `span`-unit cube and acts as a plain size multiplier on top
    of the normalized form.  `global_scale` already baked into `verts`
    cancels out in the span/ext ratio, so it stays a clean multiplier."""
    if not verts:
        return (0.0, 0.0, 0.0), global_scale
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    center = (0.5 * (min(xs) + max(xs)),
              0.5 * (min(ys) + max(ys)),
              0.5 * (min(zs) + max(zs)))
    ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    factor = (span / ext if ext > 1e-9 else 1.0) * global_scale
    return center, factor


# --------------------------------------------------------------------------
# Pure-python geometry core (no bpy - testable standalone)
# --------------------------------------------------------------------------

class Params:
    def __init__(self, branches=2, storeys=2, height=1.5, flange=1.5,
                 thickness=0.15, rim_bulge=1.5, twist=0.0, azimuth=0.0,
                 warp=0.0, detail=5, scale_x=1.0, scale_y=1.0, scale_z=1.0,
                 global_scale=1.0, phase=0.5, rim_round=1.0):
        self.branches = int(branches)
        self.storeys = int(storeys)
        self.height = height
        self.flange = flange
        self.thickness = thickness
        self.rim_bulge = rim_bulge
        self.twist = twist
        self.azimuth = azimuth
        self.warp = warp
        self.phase = phase
        self.rim_round = rim_round
        self.detail = int(detail)
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.global_scale = global_scale


def ring_closes(p):
    """True when the warped tower ends meet with matching profiles."""
    if p.warp <= 0 or abs(p.warp % 360.0) > 1e-4 and abs(p.warp % 360.0 - 360.0) > 1e-4:
        return False
    b = p.branches
    mismatch = (p.twist + p.storeys * 180.0 / b) % (360.0 / b)
    return mismatch < 1e-3 or (360.0 / b) - mismatch < 1e-3


def _v_sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _v_add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def _v_scale(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def _v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])
def _v_dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _v_len(a): return sqrt(_v_dot(a, a))
def _v_norm(a):
    l = _v_len(a)
    if l < 1e-12:
        return None
    return (a[0] / l, a[1] / l, a[2] / l)


def generate_sculpture(p, return_grids=False, with_uv=False):
    """Build the sculpture mesh. Returns (verts, faces) with faces as
    index tuples (already cleaned of degenerate entries).

    With with_uv=True returns (verts, faces, vert_uv): one (u, v) per
    vertex, U running around the branch cross-section and V up the tower
    (each branch a 1/branches slice of U, each storey stacking in V), so
    an image texture wraps the sheet naturally.

    With return_grids=True, stops after the mid-surface grids are built
    and returns (grids, R, m): grids maps (storey, branch) to a list of
    R+1 rows, each a list of m points or None (opened hole) -- the raw
    material for NURBS patch output (no thickness/rims)."""
    b = p.branches
    S = p.storeys
    # flange carries a +0.1 correction: the original tool's flange width
    # was 0.1 narrower for a given setting (a calibration bug), so a
    # stored flange is taken as the corrected value -- matching the demo
    # .txt parameter files directly
    W = p.flange + 0.1
    d = max(1, p.detail)
    R = 4 * d                    # rows per storey
    m = 4 * d + 1                # points per cross-section branch curve (odd)
    H_ex = S * pi                # tower height in exact-Scherk units
    H_out = S * p.height         # tower height in output units
    gs = p.global_scale
    t_out = p.thickness * XY_SCALE * gs          # wall thickness, output units
    solid = t_out > 1e-6
    warp_on = p.warp > 1e-6
    closes = ring_closes(p)
    # phase shift of the saddle chain relative to the tower ends, in
    # storey units: 0 puts a vane (flange) at each end (symmetric),
    # while a fraction cuts the ends through a hole.  It is rotationally
    # invisible on a closed ring, so it is forced to 0 there (keeping
    # every warped preset unchanged and the closure logic untouched).
    ph = 0.0 if closes else max(0.0, min(0.999,
                                         float(getattr(p, 'phase', 0.0))))
    z_base = ph * pi
    # azimuth carries a -45 deg correction: the original tool had a
    # 45-degree offset (a bug), so a stored azimuth is taken as the
    # corrected value -- matching the demo .txt parameter files directly
    az = radians(p.azimuth - 45.0)
    tw = radians(p.twist)
    wr = radians(p.warp) if warp_on else 0.0
    R_ring = (H_out / wr) if warp_on else 0.0
    sinhW = sinh(W)
    c_open = sinhW * sinhW       # levels with c > c_open have open (cut) holes

    def xform(xt, yt, z_ex):
        """template cross-section point + tower height -> final position"""
        xs = xt * XY_SCALE * gs
        ys = yt * XY_SCALE * gs
        zn = (z_ex - z_base) / H_ex
        a = az + tw * zn
        ca, sa = cos(a), sin(a)
        x1 = xs * ca - ys * sa
        y1 = xs * sa + ys * ca
        if warp_on:
            th = wr * (zn - 0.5)
            rad = (R_ring * gs + x1)
            X, Y, Z = rad * cos(th), rad * sin(th), y1
        else:
            X, Y, Z = x1, y1, (zn - 0.5) * H_out * gs
        return (X * p.scale_x, Y * p.scale_y, Z * p.scale_z)

    # cosine-spaced curve parameter, s[-1..1], midpoint exactly 0
    sig = [-cos(pi * k / (m - 1)) for k in range(m)]

    # ---- tower cells --------------------------------------------------
    # The tower is the window [ph, S + ph] of the periodic saddle chain
    # (heights in storey units), split into cells at the integer "vane"
    # (saddle) levels.  ph = 0 gives S full vane-to-vane storeys (the
    # original behaviour); a fractional ph cuts the two end cells through
    # a hole, shifting where the holes sit relative to the tower ends.
    # Each cell carries s_rot, the storey index driving its branch
    # rotation (constant within a cell because cells never cross a vane).
    t_lo = ph
    t_hi = S + ph
    bounds = [t_lo] + [k for k in range(1, S + 1)
                       if t_lo + 1e-9 < k < t_hi - 1e-9] + [t_hi]
    cells = []
    for cidx in range(len(bounds) - 1):
        a, bcell = bounds[cidx], bounds[cidx + 1]
        cells.append((a, bcell, int(a + 1e-9)))    # (t0, t1, s_rot)
    n_cells = len(cells)

    def cell_t(cidx, i):
        a, bcell, _ = cells[cidx]
        return a + (bcell - a) * (0.5 - 0.5 * cos(pi * i / R))

    # ---- per-grid position arrays -------------------------------------
    # grid identity: (cell index, branch j). rows i=0..R; row entries are
    # a list of m points or None (level inside an opened hole).
    grids = {}
    for cidx, (a, bcell, s_rot) in enumerate(cells):
        for j in range(b):
            A = (2 * j + s_rot) % (2 * b)      # wedge start vane index
            base_ang = A * pi / b
            wedge = pi / b
            rows = []
            for i in range(R + 1):
                # cosine-clustered rows: extra resolution at the cell ends
                # where the surface curvature concentrates
                t = cell_t(cidx, i)
                z_ex = t * pi
                frac = t - s_rot                 # 0..1 within the storey
                c = sin(pi * frac)
                # A single branch is an order-1 "saddle" -- geometrically
                # just a plane -- so every level uses the flat flange
                # cross-section (twist/warp still apply, as in Design 16);
                # only at a true vane level otherwise.
                if b == 1 or abs(frac - round(frac)) < 1e-9:
                    # saddle (vane) level: cross-section degenerates to the
                    # two vane segments [0, W]; interior points at origin
                    row = []
                    for k in range(m):
                        sg = sig[k]
                        if sg > 1e-12:            # wedge-start vane
                            ang = base_ang
                            rad = W * sg
                        elif sg < -1e-12:         # wedge-end vane
                            ang = base_ang + wedge
                            rad = -W * sg
                        else:
                            ang, rad = base_ang, 0.0
                        row.append(xform(rad * cos(ang), rad * sin(ang), z_ex))
                    rows.append(row)
                    continue
                if c > c_open:
                    rows.append(None)             # hole cut open at this level
                    continue
                sc = sqrt(c)
                L = log(sinhW / sc)
                row = []
                for k in range(m):
                    e = exp(sig[k] * L)
                    x = asinh(sc * e)
                    y = asinh(sc / e)
                    f = atan2(y, x) / (pi / 2)    # 0 at vane-start .. 1 at end
                    ang = base_ang + f * wedge
                    rad = hypot(x, y)
                    row.append(xform(rad * cos(ang), rad * sin(ang), z_ex))
                rows.append(row)
            grids[(cidx, j)] = rows

    if return_grids:
        return grids, R, m

    # ---- parametric normals per grid ----------------------------------
    normals = {}
    for key, rows in grids.items():
        nrm = [None] * (R + 1)
        for i in range(R + 1):
            if rows[i] is None:
                continue
            row_n = []
            im = i - 1 if (i > 0 and rows[i - 1] is not None) else i
            ip = i + 1 if (i < R and rows[i + 1] is not None) else i
            for k in range(m):
                km = max(0, k - 1)
                kp = min(m - 1, k + 1)
                du = _v_sub(rows[i][kp], rows[i][km])
                dv = _v_sub(rows[ip][k], rows[im][k]) if ip != im else (0, 0, 0)
                n = _v_norm(_v_cross(du, dv))
                row_n.append(n)
            row_n_fixed = []
            for k in range(m):
                n = row_n[k]
                if n is None:
                    for kk in range(1, m):
                        for cand in (k - kk, k + kk):
                            if 0 <= cand < m and row_n[cand] is not None:
                                n = row_n[cand]
                                break
                        if n is not None:
                            break
                if n is None:
                    n = (0.0, 0.0, 1.0)
                row_n_fixed.append(n)
            nrm[i] = row_n_fixed
        # fill rows whose neighbours were missing (isolated saddle rows)
        normals[key] = nrm

    # ---- canonical registry for boundary vertices ---------------------
    # key: rounded position -> [canonical normal, outward accumulator]
    def ckey(pt):
        return (round(pt[0], 5), round(pt[1], 5), round(pt[2], 5))

    boundary = {}     # ckey -> canonical normal
    outward = {}      # (ckey, tag) -> outward-direction accumulator
    canon_n = {}      # ckey -> canonical normal (first grid registering wins)

    def note_boundary(pt, n, o, tag=None):
        kk = ckey(pt)
        canon_n.setdefault(kk, n)
        boundary.setdefault(kk, canon_n[kk])
        ent = outward.get((kk, tag))
        if ent is None:
            outward[(kk, tag)] = list(o)
        else:
            ent[0] += o[0]
            ent[1] += o[1]
            ent[2] += o[2]

    def grid_boundary_edges(key):
        """yield (i,k1,k2_or_row) boundary descriptors:
        ('col', i1, i2, k)   vertical rim edge between valid rows i1,i2 at col k
        ('row', i, k1, k2)   cap edge along row i between cols k1,k2"""
        cidx, j = key
        rows = grids[key]
        out = []
        # side rims (the cut/flange edges) - always true boundary
        for k in (0, m - 1):
            for i in range(R):
                if rows[i] is not None and rows[i + 1] is not None:
                    out.append(('col', i, i + 1, k))
        # tower-end caps: only at the ends of an open (non-closing) tower
        if cidx == 0 and not closes:
            if rows[0] is not None:
                for k in range(m - 1):
                    out.append(('row', 0, k, k + 1))
        if cidx == n_cells - 1 and not closes:
            if rows[R] is not None:
                for k in range(m - 1):
                    out.append(('row', R, k, k + 1))
        # opened-hole gap boundaries
        for i in range(R + 1):
            if rows[i] is None:
                continue
            prev_gap = i > 0 and rows[i - 1] is None
            next_gap = i < R and rows[i + 1] is None
            if prev_gap or next_gap:
                for k in range(m - 1):
                    out.append(('row', i, k, k + 1))
        return out

    all_bedges = {}
    if solid:
        # canonical normals at storey-joint (saddle) rows: both adjacent
        # grids must offset their sheets along the *identical* vector or
        # the joint verts land epsilon-apart and fail to weld
        for key in grids:
            rows = grids[key]
            nrm = normals[key]
            for i in (0, R):
                if rows[i] is None:
                    continue
                for k in range(m):
                    canon_n.setdefault(ckey(rows[i][k]), nrm[i][k])
        for key in grids:
            rows = grids[key]
            nrm = normals[key]
            bed = grid_boundary_edges(key)
            all_bedges[key] = bed
            cap_rows = set()
            if key[0] == 0 and not closes:
                cap_rows.add(0)
            if key[0] == n_cells - 1 and not closes:
                cap_rows.add(R)
            for ed in bed:
                if ed[0] == 'col':
                    _, i1, i2, k = ed
                    for i in (i1, i2):
                        pt = rows[i][k]
                        kin = m - 2 if k == m - 1 else 1
                        o = _v_norm(_v_sub(rows[i][k], rows[i][kin]))
                        if o is None:
                            o = (0, 0, 0)
                        note_boundary(pt, nrm[i][k], o,
                                      key if i in cap_rows else None)
                else:
                    _, i, k1, k2 = ed
                    iin = i - 1 if (i > 0 and rows[i - 1] is not None) else \
                          (i + 1 if (i < R and rows[i + 1] is not None) else i)
                    for k in (k1, k2):
                        pt = rows[i][k]
                        if iin == i:
                            o = (0, 0, 0)
                        else:
                            o = _v_norm(_v_sub(rows[i][k], rows[iin][k])) or (0, 0, 0)
                        note_boundary(pt, nrm[i][k], o,
                                      key if i in cap_rows else None)

    # ---- emit vertices and faces --------------------------------------
    verts = []
    faces = []
    vert_uv = []

    def add_vert(pt, uv=(0.0, 0.0)):
        verts.append(pt)
        vert_uv.append(uv)
        return len(verts) - 1

    def uv_of(cidx, jp, i, k):
        """(u, v) for row i / column k of branch jp, cell cidx: u runs
        around the branch cross-section, v up the tower."""
        v = (cell_t(cidx, i) - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0.0
        return ((jp + k / (m - 1)) / b, v)

    def add_quad(a, bq, c2, d2):
        uniq = []
        for idx in (a, bq, c2, d2):
            if idx not in uniq:
                uniq.append(idx)
        if len(uniq) >= 3:
            faces.append(tuple(uniq))

    n_arc = 8 if d >= 4 else 4
    w_bulge = (t_out / 2.0) * (1.0 + p.rim_bulge)
    # rim roundness: 1 = the rounded bull-nose, 0 = a flat/square edge
    # (the outward arc collapses to a straight wall between the sheets)
    rim_r = max(0.0, min(1.0, float(getattr(p, 'rim_round', 1.0))))

    # canonical rim arc interiors, shared across grids so rim tubes join
    # watertight even where neighbouring grids have opposite winding
    arc_cache = {}

    def rim_arc(pt, sheet_top_idx, sheet_bot_idx, sign, cache_tag=None,
                uv=(0.0, 0.0)):
        """Full arc vertex list ordered from the (p + t/2*n_canonical) end
        to the (p - t/2*n_canonical) end. `sign` says which of this grid's
        sheet verts sits at which end. `cache_tag` scopes the shared-arc
        cache (cap-row arcs are per-grid so that the 2b leg tubes meeting
        on the tower axis stay manifold)."""
        kk = (ckey(pt), cache_tag)
        interior = arc_cache.get(kk)
        if interior is None:
            n = boundary[ckey(pt)]
            o_sum = outward.get((ckey(pt), cache_tag), (0.0, 0.0, 0.0))
            o = _v_norm(tuple(o_sum))
            if o is None:
                # undefined outward direction: straight connector
                o = (0.0, 0.0, 0.0)
            interior = []
            for a in range(1, n_arc):
                phi = pi * a / n_arc
                q = _v_add(pt, _v_scale(n, (t_out / 2.0) * cos(phi)))
                q = _v_add(q, _v_scale(o, rim_r * w_bulge * sin(phi)))
                interior.append(add_vert(q, uv))
            arc_cache[kk] = interior
        if sign >= 0:
            return [sheet_top_idx] + interior + [sheet_bot_idx]
        return [sheet_bot_idx] + interior + [sheet_top_idx]

    for key in grids:
        rows = grids[key]
        nrm = normals[key]
        bed = all_bedges.get(key, [])
        bset = set()
        for ed in bed:
            if ed[0] == 'col':
                _, i1, i2, k = ed
                bset.add((i1, k)); bset.add((i2, k))
            else:
                _, i, k1, k2 = ed
                bset.add((i, k1)); bset.add((i, k2))

        if not solid:
            vid = [None] * (R + 1)
            for i in range(R + 1):
                if rows[i] is None:
                    continue
                vid[i] = [add_vert(rows[i][k], uv_of(key[0], key[1], i, k))
                          for k in range(m)]
            for i in range(R):
                if vid[i] is None or vid[i + 1] is None:
                    continue
                for k in range(m - 1):
                    add_quad(vid[i][k], vid[i][k + 1],
                             vid[i + 1][k + 1], vid[i + 1][k])
            continue

        # solid: top + bottom sheets. Boundary verts use the canonical
        # normal (sign-matched to this grid) so rim arcs join exactly.
        top = [None] * (R + 1)
        bot = [None] * (R + 1)
        gsign = {}
        for i in range(R + 1):
            if rows[i] is None:
                continue
            trow, brow = [], []
            for k in range(m):
                pt = rows[i][k]
                n = nrm[i][k]
                cn = canon_n.get(ckey(pt)) if ((i, k) in bset or i in (0, R)) \
                    else None
                if cn is not None:
                    if _v_dot(cn, n) < 0:
                        n = _v_scale(cn, -1.0)
                        gsign[(i, k)] = -1
                    else:
                        n = cn
                        gsign[(i, k)] = 1
                off = _v_scale(n, t_out / 2.0)
                uv = uv_of(key[0], key[1], i, k)
                trow.append(add_vert(_v_add(pt, off), uv))
                brow.append(add_vert(_v_sub(pt, off), uv))
            top[i] = trow
            bot[i] = brow
        for i in range(R):
            if top[i] is None or top[i + 1] is None:
                continue
            for k in range(m - 1):
                add_quad(top[i][k], top[i][k + 1],
                         top[i + 1][k + 1], top[i + 1][k])
                add_quad(bot[i + 1][k], bot[i + 1][k + 1],
                         bot[i][k + 1], bot[i][k])
        # rim strips along true boundary edges
        for ed in bed:
            if ed[0] == 'col':
                _, i1, i2, k = ed
                va, vb = (i1, k), (i2, k)
            else:
                _, i, k1, k2 = ed
                va, vb = (i, k1), (i, k2)
            pa = rows[va[0]][va[1]]
            pb = rows[vb[0]][vb[1]]
            if ckey(pa) == ckey(pb):
                continue                     # degenerate (zero-length) edge
            s = key[0]
            cap_rows = set()
            if s == 0 and not closes:
                cap_rows.add(0)
            if s == n_cells - 1 and not closes:
                cap_rows.add(R)
            tag_a = key if va[0] in cap_rows else None
            tag_b = key if vb[0] in cap_rows else None
            arc_a = rim_arc(pa, top[va[0]][va[1]], bot[va[0]][va[1]],
                            gsign.get(va, 1), tag_a,
                            uv=uv_of(key[0], key[1], va[0], va[1]))
            arc_b = rim_arc(pb, top[vb[0]][vb[1]], bot[vb[0]][vb[1]],
                            gsign.get(vb, 1), tag_b,
                            uv=uv_of(key[0], key[1], vb[0], vb[1]))
            # arcs are listed in canonical-normal order; if the endpoints'
            # canonical normals oppose (happens where grid orientations
            # flip, e.g. next to the closure seam), pair against the
            # reversed arc -- the semicircle is index-symmetric, so this
            # joins the geometrically matching sides
            if _v_dot(boundary[ckey(pa)], boundary[ckey(pb)]) < 0:
                arc_b = arc_b[::-1]
            for a in range(n_arc):
                add_quad(arc_a[a], arc_b[a], arc_b[a + 1], arc_a[a + 1])

    if with_uv:
        return verts, faces, vert_uv
    return verts, faces


def weld_epsilon(p):
    return WELD_EPS_BASE * max(1.0, p.storeys * p.height) * p.global_scale


# --------------------------------------------------------------------------
# Spec-file I/O (the original program's save/demo format)
# --------------------------------------------------------------------------

SPEC_KEYS = {
    'branches': ('branches', int),
    'storeys': ('storeys', int),
    'stories': ('storeys', int),
    'height': ('height', float),
    'flange': ('flange', float),
    'thickness': ('thickness', float),
    'rim_bulge': ('rim_bulge', float),
    'rim_round': ('rim_round', float),
    'twist': ('twist', float),
    'azimuth': ('azimuth', float),
    'warp': ('warp', float),
    'phase': ('phase', float),
    'detail': ('detail', int),
    'scaleX': ('scale_x', float),
    'scaleY': ('scale_y', float),
    'scaleZ': ('scale_z', float),
}


def parse_spec_text(text):
    """Parse a Sculpture Generator spec/demo file; returns dict of the
    geometry parameters found (display-related fields are ignored)."""
    found = {}
    for line in text.splitlines():
        if '=' not in line:
            continue
        name, _, val = line.partition('=')
        name = name.strip()
        if name in SPEC_KEYS:
            attr, cast = SPEC_KEYS[name]
            try:
                found[attr] = cast(float(val.split()[0]))
            except (ValueError, IndexError):
                pass
    return found


def spec_text_from(p):
    """Write a spec file the original generator.exe can load (display
    fields are filled with the defaults used by its demo files)."""
    return (
        "SCHERK-COLLINS SCULPTURE\n"
        f"branches = {p.branches}\n"
        f"storeys = {p.storeys}\n"
        f"height = {p.height:g}\n"
        f"flange = {p.flange:g}\n"
        f"thickness = {p.thickness:g}\n"
        f"rim_bulge = {p.rim_bulge:g}\n"
        f"rim_round = {p.rim_round:g}\n"
        f"warp = {p.warp:g}\n"
        f"twist = {p.twist:g}\n"
        f"azimuth = {p.azimuth:g}\n"
        f"phase = {p.phase:g}\n"
        "texture_tiles = 1\n"
        f"detail = {p.detail}\n"
        f"scaleX = {p.scale_x:g}\n"
        f"scaleY = {p.scale_y:g}\n"
        f"scaleZ = {p.scale_z:g}\n"
        "texture_file = Images/env.ppm\n"
        "texture_on = 0\n"
        "environment_map_on = 0\n"
        "background_file = Images/SculptStand.rgb\n"
        "background_on = 0\n"
        "background_color = 0 0 0\n"
        "face1_color = 255 255 255\n"
        "face2_color = 255 255 255\n"
        "rim1_color = 255 255 255\n"
        "rim2_color = 255 255 255\n"
        "rim3_color = 255 255 255\n"
        "auto_rotate_on = 0\n"
        "auto_wobble_on = 0\n"
        "x_rotation = 1.000\n"
        "y_rotation = 1.000\n"
        "scale_factor = 0.362\n"
        "x_translate = 0.000\n"
        "y_translate = 0.000\n"
        "modelview_matrix = 1.00000 0.00000 0.00000 0.00000\n"
        "                   0.00000 1.00000 0.00000 0.00000\n"
        "                   0.00000 0.00000 1.00000 0.00000\n"
        "                   0.00000 0.00000 0.00000 1.00000\n"
    )


PRESETS = {
    'HEX': ("Hyperbolic Hexagon", dict(branches=2, storeys=6, height=1.2,
            flange=1.1, thickness=0.07, rim_bulge=1.06, twist=0, azimuth=45,
            warp=360)),
    'TREFOIL': ("Minimal Trefoil", dict(branches=2, storeys=3, height=1.5,
            flange=1.3, thickness=0.06, rim_bulge=1.06, twist=270, azimuth=45,
            warp=360)),
    'MONKEY': ("Monkey-Saddle Trefoil", dict(branches=3, storeys=3, height=1.75,
            flange=1.5, thickness=0.05, rim_bulge=1.0, twist=180, azimuth=0,
            warp=360)),
    'HEPTOROID': ("Heptoroid", dict(branches=4, storeys=7, height=1.5,
            flange=1.0, thickness=0.10, rim_bulge=0.0, twist=135, azimuth=0,
            warp=360)),
    'TOWER': ("Scherk Tower (straight)", dict(branches=2, storeys=5, height=1.5,
            flange=1.5, thickness=0.1, rim_bulge=1.0, twist=0, azimuth=0,
            warp=0)),
    # Presets imported from the original SculptureGenerator demo/*.txt
    # parameter files (azimuth taken directly, in the corrected
    # convention).  Provisional "Demo N" names pending final names.
    # demo5 = Hyperbolic Hexagon, demo9 = Minimal Trefoil,
    # demo19 = Heptoroid (already above, not duplicated).
    'DEMO1': ("Demo 1", dict(branches=1, storeys=4, height=1.9, flange=1.5,
            thickness=0.08, rim_bulge=1.5, warp=270, twist=885, azimuth=0,
            detail=11)),
    'DEMO2': ("Demo 2", dict(branches=1, storeys=3, height=1.7, flange=1.0,
            thickness=0.08, rim_bulge=1.5, warp=360, twist=540, azimuth=0,
            detail=10)),
    'DEMO3': ("Demo 3", dict(branches=3, storeys=5, height=1.0, flange=1.1,
            thickness=0.04, rim_bulge=1.5, warp=0, twist=-135, azimuth=0,
            detail=5)),
    'DEMO4': ("Demo 4", dict(branches=3, storeys=6, height=1.0, flange=1.1,
            thickness=0.08, rim_bulge=1.01, warp=360, twist=0, azimuth=0,
            detail=5)),
    'DEMO6': ("Demo 6", dict(branches=6, storeys=4, height=1.0, flange=0.8,
            thickness=0.02, rim_bulge=0.0, warp=360, twist=180, azimuth=45,
            detail=7)),
    'DEMO7': ("Demo 7", dict(branches=4, storeys=8, height=1.0, flange=0.8,
            thickness=0.04, rim_bulge=0.96, warp=360, twist=180, azimuth=45,
            detail=6)),
    'DEMO8': ("Demo 8", dict(branches=4, storeys=12, height=1.0, flange=1.0,
            thickness=0.06, rim_bulge=1.06, warp=210, twist=180, azimuth=45,
            detail=7)),
    'DEMO10': ("Demo 10", dict(branches=4, storeys=3, height=1.0, flange=1.0,
            thickness=0.09, rim_bulge=1.06, warp=0, twist=180, azimuth=45,
            detail=5)),
    'DEMO11': ("Demo 11", dict(branches=7, storeys=2, height=0.1, flange=2.1,
            thickness=0.0, rim_bulge=0.0, warp=0, twist=0, azimuth=0,
            detail=5)),
    'DEMO12': ("Demo 12", dict(branches=7, storeys=1, height=0.3, flange=2.6,
            thickness=0.03, rim_bulge=0.0, warp=0, twist=-75, azimuth=0,
            detail=8)),
    'DEMO13': ("Demo 13", dict(branches=5, storeys=3, height=0.3, flange=1.1,
            thickness=0.2, rim_bulge=0.55, warp=0, twist=0, azimuth=0,
            detail=6)),
    'DEMO14': ("Demo 14", dict(branches=2, storeys=3, height=1.5, flange=1.5,
            thickness=0.15, rim_bulge=1.5, warp=0, twist=0, azimuth=0,
            detail=5)),
    'DEMO15': ("Demo 15", dict(branches=2, storeys=11, height=1.5, flange=1.5,
            thickness=0.15, rim_bulge=1.5, warp=180, twist=495, azimuth=0,
            detail=5)),
    'DEMO16': ("Demo 16", dict(branches=1, storeys=3, height=1.5, flange=1.5,
            thickness=0.15, rim_bulge=1.5, warp=0, twist=-300, azimuth=0,
            detail=5)),
    'DEMO20': ("Demo 20", dict(branches=4, storeys=7, height=1.5, flange=0.9,
            thickness=0.11, rim_bulge=0.43, warp=60, twist=390, azimuth=-15,
            detail=5)),
}


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from mathutils import Matrix
    from bpy.props import (IntProperty, FloatProperty, BoolProperty,
                           PointerProperty, StringProperty, EnumProperty)
    from bpy_extras.io_utils import ImportHelper, ExportHelper
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _params_from_props(st):
        detail = st.nurbs_detail if st.output_nurbs else st.detail
        return Params(branches=st.branches, storeys=st.storeys,
                      height=st.height, flange=st.flange,
                      thickness=st.thickness, rim_bulge=st.rim_bulge,
                      twist=st.twist, azimuth=st.azimuth, warp=st.warp,
                      phase=st.phase, rim_round=st.rim_round, detail=detail,
                      scale_x=st.scale_x, scale_y=st.scale_y,
                      scale_z=st.scale_z, global_scale=st.global_scale)

    def _nurbs_patches(p):
        """Contiguous mid-surface row blocks -> NURBS control patches.
        Each block is split at the wedge bisector column so that patch
        edges coincide with single vane legs; adjacent storeys' patches
        then share identical edge control sequences and join cleanly."""
        grids, R, m = generate_sculpture(p, return_grids=True)
        mid = (m - 1) // 2
        patches = []
        for key in sorted(grids.keys()):
            block = []
            for row in grids[key] + [None]:
                if row is not None:
                    block.append(row)
                    continue
                if len(block) >= 2:
                    patches.append([r[:mid + 1] for r in block])
                    patches.append([r[mid:] for r in block])
                block = []
        return patches

    def _build_nurbs_data(obj, p):
        """Replace obj's data with a NURBS surface (one clamped patch per
        storey/branch grid; thickness and rims do not apply)."""
        su = bpy.data.curves.new("ScherkCollins", 'SURFACE')
        su.dimensions = '3D'
        old = obj.data
        obj.data = su
        if old is not None and old.users == 0:
            if isinstance(old, bpy.types.Mesh):
                bpy.data.meshes.remove(old)
            else:
                bpy.data.curves.remove(old)
        view = bpy.context.view_layer
        prev_active = view.objects.active
        view.objects.active = obj

        def _grid_fps():
            fps = []
            for sp in su.splines:
                if sp.point_count_u > 1 and sp.point_count_v > 1:
                    c0 = sp.points[0].co
                    fps.append((sp.point_count_u, sp.point_count_v,
                                round(c0[0], 4), round(c0[1], 4),
                                round(c0[2], 4)))
            return fps

        # Fit the mid-surface control net into the 2 m cube (same
        # convention as the mesh path) before laying down the splines.
        patches = _nurbs_patches(p)
        allpts = [pt for patch in patches for row in patch for pt in row]
        center, f = fit_transform(allpts, p.global_scale)
        patches = [[[((pt[0] - center[0]) * f,
                      (pt[1] - center[1]) * f,
                      (pt[2] - center[2]) * f) for pt in row]
                    for row in patch] for patch in patches]
        for patch in patches:
            prev_fps = _grid_fps()
            for sp in su.splines:
                for pt in sp.points:
                    pt.select = False
            for row in patch:
                sp = su.splines.new('NURBS')
                sp.points.add(len(row) - 1)
                flat = []
                for (x, y, z) in row:
                    flat.extend((x, y, z, 1.0))
                sp.points.foreach_set('co', flat)
                for pt in sp.points:
                    pt.select = True
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.curve.make_segment()
            bpy.ops.object.mode_set(mode='OBJECT')
            # make_segment chains rows in an arbitrary order: find the
            # newly created grid spline and rewrite its control points
            # into the intended row-major layout
            remaining = list(prev_fps)
            new_sp = None
            for sp in su.splines:
                if sp.point_count_u > 1 and sp.point_count_v > 1:
                    c0 = sp.points[0].co
                    fp = (sp.point_count_u, sp.point_count_v,
                          round(c0[0], 4), round(c0[1], 4), round(c0[2], 4))
                    if fp in remaining:
                        remaining.remove(fp)
                    else:
                        new_sp = sp
            if new_sp is not None:
                pu, pv = new_sp.point_count_u, new_sp.point_count_v
                nrows, ncols = len(patch), len(patch[0])
                flat = []
                if (pu, pv) == (nrows, ncols):
                    # storage is u-fastest: S[v][u] = patch[u][v]
                    for vv in range(pv):
                        for uu in range(pu):
                            x, y, z = patch[uu][vv]
                            flat.extend((x, y, z, 1.0))
                else:
                    for vv in range(pv):
                        for uu in range(pu):
                            x, y, z = patch[vv][uu]
                            flat.extend((x, y, z, 1.0))
                new_sp.points.foreach_set('co', flat)
            for sp in su.splines:
                if sp.point_count_u > 1 and sp.point_count_v > 1:
                    sp.order_u = min(4, sp.point_count_u)
                    sp.order_v = min(4, sp.point_count_v)
                    sp.use_endpoint_u = True
                    sp.use_endpoint_v = True
                    sp.resolution_u = 4
                    sp.resolution_v = 4
        su.update_tag()
        if prev_active is not None:
            view.objects.active = prev_active

    _PROP_COPY_KEYS = ('is_scherk', 'auto_update', 'branches', 'storeys',
                       'height', 'flange', 'thickness', 'rim_bulge',
                       'rim_round', 'twist', 'azimuth', 'warp', 'phase',
                       'detail', 'scale_x',
                       'scale_y', 'scale_z', 'global_scale', 'output_nurbs',
                       'nurbs_detail')

    def _swap_object_type(old_obj, to_surface):
        """Mesh <-> Surface object types cannot be changed in place;
        recreate the object, carrying over transform and parameters."""
        name = old_obj.name
        saved = {k: getattr(old_obj.scherk_collins, k)
                 for k in _PROP_COPY_KEYS}
        mw = old_obj.matrix_world.copy()
        colls = list(old_obj.users_collection)
        if to_surface:
            data = bpy.data.curves.new(name, 'SURFACE')
            data.dimensions = '3D'
        else:
            data = bpy.data.meshes.new(name)
        bpy.data.objects.remove(old_obj, do_unlink=True)
        new_obj = bpy.data.objects.new(name, data)
        for coll in (colls or [bpy.context.collection]):
            coll.objects.link(new_obj)
        new_obj.matrix_world = mw
        st = new_obj.scherk_collins
        st.auto_update = False
        for k in saved:
            if k != 'auto_update':
                setattr(st, k, saved[k])
        st.auto_update = saved['auto_update']
        new_obj.select_set(True)
        bpy.context.view_layer.objects.active = new_obj
        return new_obj

    def rebuild_object(obj):
        st = obj.scherk_collins
        p = _params_from_props(st)
        if st.output_nurbs != (obj.type == 'SURFACE'):
            obj = _swap_object_type(obj, st.output_nurbs)
            st = obj.scherk_collins
        if st.output_nurbs:
            _build_nurbs_data(obj, p)
            return obj
        verts, faces, vert_uv = generate_sculpture(p, with_uv=True)
        me = bpy.data.meshes.new(obj.data.name if obj.data else "ScherkCollins")
        me.from_pydata(verts, [], faces)
        me.validate(clean_customdata=True)
        bm = bmesh.new()
        bm.from_mesh(me)
        # UV map from the parametric grid, set before welding so the
        # per-loop coordinates survive remove_doubles (bmesh vert index
        # still matches the pre-weld vertex order / vert_uv here)
        if len(vert_uv) == len(me.vertices):
            bm.verts.ensure_lookup_table()
            uvl = bm.loops.layers.uv.new("UVMap")
            for f in bm.faces:
                for loop in f.loops:
                    loop[uvl].uv = vert_uv[loop.vert.index]
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld_epsilon(p))
        bmesh.ops.dissolve_degenerate(bm, dist=weld_epsilon(p) * 0.1,
                                      edges=bm.edges)
        if p.thickness * XY_SCALE * p.global_scale > 1e-6:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        # Centre at the origin and fit the whole sculpture (thickness and
        # rim beads included) into the 2 m cube, then apply Overall Scale.
        # Done after welding so remove_doubles still runs at the tuned raw
        # scale; a uniform positive scale preserves the recalculated normals.
        center, f = fit_transform(verts, p.global_scale)
        me.transform(Matrix.Diagonal((f, f, f, 1.0)) @
                     Matrix.Translation((-center[0], -center[1], -center[2])))
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        old = obj.data
        obj.data = me
        if old and old.users == 0:
            bpy.data.meshes.remove(old)
        return obj

    def _prop_update(self, context):
        if not self.auto_update or not self.is_scherk:
            return
        obj = self.id_data
        if obj is None:
            return
        if self.output_nurbs or obj.type == 'SURFACE':
            # NURBS rebuilds use edit-mode operators and may replace the
            # object; defer out of the property-update callback
            name = obj.name
            def _deferred():
                o = bpy.data.objects.get(name)
                if o is not None and o.scherk_collins.is_scherk:
                    rebuild_object(o)
                return None
            bpy.app.timers.register(_deferred, first_interval=0.05)
        else:
            rebuild_object(obj)

    class ScherkCollinsProps(bpy.types.PropertyGroup):
        is_scherk: BoolProperty(default=False, options={'HIDDEN'})
        auto_update: BoolProperty(
            name="Auto Update", default=True,
            description="Rebuild the mesh whenever a parameter changes")
        branches: IntProperty(
            name="Branches", description="Order of the saddles (# of branches)",
            default=2, min=1, max=10, update=_prop_update)
        storeys: IntProperty(
            name="Storeys", description="Number of hole/saddle storeys",
            default=2, min=1, max=16, update=_prop_update)
        height: FloatProperty(
            name="Storey Height", description="Height of one storey",
            default=1.5, min=0.1, max=5.0, update=_prop_update)
        flange: FloatProperty(
            name="Flange Width",
            description="Width of the flanges (holes break open below ~0.88)",
            default=1.5, min=0.7, max=5.0, update=_prop_update)
        thickness: FloatProperty(
            name="Thickness", description="Thickness of the vanes (0 = surface only)",
            default=0.15, min=0.0, max=0.5, update=_prop_update)
        rim_bulge: FloatProperty(
            name="Rim Bulge", description="Amount of bulge on the rim beads",
            default=1.5, min=0.0, max=4.0, update=_prop_update)
        rim_round: FloatProperty(
            name="Rim Round",
            description="Roundness of the edge: 1 = a rounded bull-nose, "
                        "0 = a flat/square edge",
            default=1.0, min=0.0, max=1.0, step=10, update=_prop_update)
        twist: FloatProperty(
            name="Twist", description="Overall axial twist (degrees)",
            default=0.0, min=-900.0, max=1080.0, step=1500, update=_prop_update)
        azimuth: FloatProperty(
            name="Azimuth", description="Turn of the profile around the tower axis (degrees)",
            default=0.0, min=-360.0, max=360.0, step=500, update=_prop_update)
        warp: FloatProperty(
            name="Warp", description="Bend of the tower towards an arch/toroid (degrees; 360 = closed ring)",
            default=0.0, min=0.0, max=1080.0, step=1000, update=_prop_update)
        phase: FloatProperty(
            name="Phase",
            description="Shift the holes along an open tower, in storeys "
                        "(0 = a flange at each end; 0.5 = a half-hole at "
                        "each end). No effect on closed rings.",
            default=0.5, min=0.0, max=0.999, step=10, update=_prop_update)
        detail: IntProperty(
            name="Detail", description="Grid detail (tessellation density)",
            default=5, min=1, max=16, update=_prop_update)
        scale_x: FloatProperty(
            name="Stretch X", default=1.0, min=0.2, max=5.0, update=_prop_update)
        scale_y: FloatProperty(
            name="Stretch Y", default=1.0, min=0.2, max=5.0, update=_prop_update)
        scale_z: FloatProperty(
            name="Stretch Z", default=1.0, min=0.2, max=5.0, update=_prop_update)
        global_scale: FloatProperty(
            name="Overall Scale", default=1.0, min=0.05, max=10.0,
            update=_prop_update)
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Output a compact NURBS surface (mid-surface only; "
                        "thickness and rim bulge do not apply)",
            update=_prop_update)
        nurbs_detail: IntProperty(
            name="NURBS Detail",
            description="Control-point density used for NURBS output "
                        "(the NURBS surface stays smooth at low values)",
            default=2, min=1, max=16, update=_prop_update)

    def _apply_param_dict(st, d):
        st.auto_update, saved = False, st.auto_update
        for k, v in d.items():
            if hasattr(st, k):
                setattr(st, k, v)
        st.auto_update = saved

    _RESET_KEYS = ('branches', 'storeys', 'height', 'flange', 'thickness',
                   'rim_bulge', 'rim_round', 'twist', 'azimuth', 'warp',
                   'phase', 'detail', 'scale_x', 'scale_y', 'scale_z',
                   'global_scale')

    def _preset_chosen(self, context):
        """Copy the chosen preset's values into the operator's own
        properties so the redo-panel sliders start from them and remain
        freely tweakable afterwards.  Choosing Default resets every
        parameter to the program defaults."""
        if self.preset == 'CUSTOM':
            for k in _RESET_KEYS:
                prop = self.bl_rna.properties.get(k)
                if prop is not None and hasattr(self, k):
                    setattr(self, k, prop.default)
        else:
            for k, v in PRESETS[self.preset][1].items():
                if hasattr(self, k):
                    setattr(self, k, v)
        self.preset_applied = True

    class MESH_OT_scherk_collins_add(bpy.types.Operator):
        """Add a Scherk-Collins sculpture (tweak all parameters live in
        the redo panel, or later in the N-panel 'Scherk' tab)"""
        bl_idname = "mesh.scherk_collins_add"
        bl_label = "Scherk-Collins Sculpture"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Preset",
            items=[('CUSTOM', "Default", "Program defaults")] +
                  [(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='CUSTOM', update=_preset_chosen)
        branches: IntProperty(name="Branches", default=2, min=1, max=10)
        storeys: IntProperty(name="Storeys", default=2, min=1, max=16)
        height: FloatProperty(name="Storey Height", default=1.5,
                              min=0.1, max=5.0)
        flange: FloatProperty(name="Flange Width", default=1.5,
                              min=0.7, max=5.0)
        thickness: FloatProperty(name="Thickness", default=0.15,
                                 min=0.0, max=0.5)
        rim_bulge: FloatProperty(name="Rim Bulge", default=1.5,
                                 min=0.0, max=4.0)
        rim_round: FloatProperty(
            name="Rim Round", default=1.0, min=0.0, max=1.0,
            description="Roundness of the edge: 1 = rounded bull-nose, "
                        "0 = flat/square edge")
        twist: FloatProperty(name="Twist", default=0.0,
                             min=-900.0, max=1080.0)
        azimuth: FloatProperty(name="Azimuth", default=0.0,
                               min=-360.0, max=360.0)
        warp: FloatProperty(name="Warp", default=0.0, min=0.0, max=1080.0)
        phase: FloatProperty(
            name="Phase", default=0.5, min=0.0, max=0.999,
            description="Shift the holes along an open tower, in storeys "
                        "(0 = flange at each end, 0.5 = half-hole at each "
                        "end); no effect on closed rings")
        detail: IntProperty(name="Detail", default=5, min=1, max=16)
        scale_x: FloatProperty(name="Stretch X", default=1.0,
                               min=0.2, max=5.0)
        scale_y: FloatProperty(name="Stretch Y", default=1.0,
                               min=0.2, max=5.0)
        scale_z: FloatProperty(name="Stretch Z", default=1.0,
                               min=0.2, max=5.0)
        global_scale: FloatProperty(name="Overall Scale", default=1.0,
                                    min=0.05, max=10.0)
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Compact NURBS surface instead of a mesh "
                        "(mid-surface only; no thickness/rims)")
        nurbs_detail: IntProperty(
            name="NURBS Detail", default=2, min=1, max=16,
            description="Control-point density used for NURBS output")
        # set once the preset values have been copied into the sliders,
        # so redo-panel tweaks are not overwritten on re-execute
        preset_applied: BoolProperty(default=False, options={'HIDDEN'})

        _PARAM_KEYS = ('branches', 'storeys', 'height', 'flange',
                       'thickness', 'rim_bulge', 'rim_round', 'twist',
                       'azimuth', 'warp', 'phase', 'detail', 'scale_x',
                       'scale_y', 'scale_z',
                       'global_scale', 'output_nurbs', 'nurbs_detail')

        def execute(self, context):
            # menu/scripted invocation sets `preset` without firing its
            # update callback -- apply it here exactly once
            if self.preset != 'CUSTOM' and not self.preset_applied:
                _preset_chosen(self, context)
            me = bpy.data.meshes.new("ScherkCollins")
            obj = bpy.data.objects.new("ScherkCollins", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            st = obj.scherk_collins
            st.is_scherk = True
            _apply_param_dict(st, {k: getattr(self, k)
                                   for k in self._PARAM_KEYS})
            rebuild_object(obj)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, "preset")
            col = lay.column(align=True)
            skip = 'detail' if self.output_nurbs else 'nurbs_detail'
            for k in self._PARAM_KEYS:
                if k != skip:
                    col.prop(self, k)

    class SCHERK_OT_regenerate(bpy.types.Operator):
        """Rebuild the sculpture mesh from its parameters"""
        bl_idname = "scherk.regenerate"
        bl_label = "Regenerate"
        bl_options = {'REGISTER', 'UNDO'}

        @classmethod
        def poll(cls, context):
            o = context.object
            return o is not None and o.scherk_collins.is_scherk

        def execute(self, context):
            rebuild_object(context.object)
            return {'FINISHED'}

    class SCHERK_OT_load_spec(bpy.types.Operator, ImportHelper):
        """Load a Sculpture Generator spec/demo file (.txt)"""
        bl_idname = "scherk.load_spec"
        bl_label = "Load Spec File"
        bl_options = {'REGISTER', 'UNDO'}
        filename_ext = ".txt"
        filter_glob: StringProperty(default="*.txt;*.param", options={'HIDDEN'})

        def execute(self, context):
            try:
                with open(self.filepath, 'r', encoding='utf-8',
                          errors='replace') as f:
                    d = parse_spec_text(f.read())
            except OSError as e:
                self.report({'ERROR'}, f"Cannot read file: {e}")
                return {'CANCELLED'}
            if not d:
                self.report({'ERROR'}, "No sculpture parameters found in file")
                return {'CANCELLED'}
            obj = context.object
            if obj is None or not obj.scherk_collins.is_scherk:
                bpy.ops.mesh.scherk_collins_add()
                obj = context.object
            _apply_param_dict(obj.scherk_collins, d)
            rebuild_object(obj)
            self.report({'INFO'}, f"Loaded {len(d)} parameters")
            return {'FINISHED'}

    class SCHERK_OT_save_spec(bpy.types.Operator, ExportHelper):
        """Save parameters as a Sculpture Generator spec file (.txt)"""
        bl_idname = "scherk.save_spec"
        bl_label = "Save Spec File"
        filename_ext = ".txt"
        filter_glob: StringProperty(default="*.txt", options={'HIDDEN'})

        @classmethod
        def poll(cls, context):
            o = context.object
            return o is not None and o.scherk_collins.is_scherk

        def execute(self, context):
            p = _params_from_props(context.object.scherk_collins)
            try:
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    f.write(spec_text_from(p))
            except OSError as e:
                self.report({'ERROR'}, f"Cannot write file: {e}")
                return {'CANCELLED'}
            return {'FINISHED'}

    class VIEW3D_PT_scherk_collins(bpy.types.Panel):
        bl_label = "Scherk-Collins"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "Math Art"
        bl_order = 20

        @classmethod
        def poll(cls, context):
            # Only for a Scherk-Collins sculpture.  The panel used to offer
            # an Add button when none was selected, which forced it to stay
            # visible in every scene; creating one belongs in Add > Mesh,
            # where every other generator's entry already is.
            o = context.object
            return o is not None and o.scherk_collins.is_scherk

        def draw(self, context):
            lay = self.layout
            obj = context.object
            st = obj.scherk_collins
            col = lay.column(align=True)
            col.use_property_split = True
            col.prop(st, "branches")
            col.prop(st, "storeys")
            col.prop(st, "height")
            col.prop(st, "flange")
            col.prop(st, "thickness")
            col.prop(st, "rim_bulge")
            col.prop(st, "rim_round")
            col.separator()
            col.prop(st, "twist")
            col.prop(st, "azimuth")
            col.prop(st, "warp")
            col.prop(st, "phase")
            col.separator()
            col.prop(st, "nurbs_detail" if st.output_nurbs else "detail")
            col.prop(st, "scale_x")
            col.prop(st, "scale_y")
            col.prop(st, "scale_z")
            col.prop(st, "global_scale")
            col.prop(st, "output_nurbs")
            if st.output_nurbs:
                col.label(text="NURBS: thickness/rims not applied",
                          icon='INFO')
            p = _params_from_props(st)
            if p.warp > 0:
                if ring_closes(p):
                    lay.label(text="Ring closes smoothly", icon='CHECKMARK')
                else:
                    b = p.branches
                    need = (-p.storeys * 180.0 / b) % (360.0 / b)
                    lay.label(text=f"Seam at ring closure (twist "
                                   f"{need:.0f} + k*{360.0 / b:.0f} closes)",
                              icon='ERROR')
            row = lay.row(align=True)
            row.prop(st, "auto_update", toggle=True)
            row.operator("scherk.regenerate", icon='FILE_REFRESH')
            row = lay.row(align=True)
            row.operator("scherk.load_spec", text="Load", icon='FILE_FOLDER')
            row.operator("scherk.save_spec", text="Save", icon='FILE_TICK')
            lay.operator_menu_enum("mesh.scherk_collins_add", "preset",
                                   text="Add Another", icon='MESH_TORUS')

    def _menu_func(self, context):
        self.layout.operator_menu_enum("mesh.scherk_collins_add", "preset",
                                       text="Scherk-Collins Sculpture",
                                       icon='MESH_TORUS')

    _classes = (ScherkCollinsProps, MESH_OT_scherk_collins_add,
                SCHERK_OT_regenerate, SCHERK_OT_load_spec,
                SCHERK_OT_save_spec, VIEW3D_PT_scherk_collins)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        bpy.types.Object.scherk_collins = PointerProperty(
            type=ScherkCollinsProps)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        del bpy.types.Object.scherk_collins
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


def _selftest():
    # standalone smoke test of the geometry core
    for name, kw in [("defaults", {}),
                     ("trefoil", PRESETS['TREFOIL'][1]),
                     ("heptoroid", PRESETS['HEPTOROID'][1]),
                     ("open arc b1", dict(branches=1, storeys=4, height=1.9,
                                          flange=1.5, thickness=0.08,
                                          rim_bulge=1.5, warp=270,
                                          twist=885, detail=6)),
                     ("open holes", dict(branches=6, storeys=4, height=1.0,
                                         flange=0.8, thickness=0.02,
                                         rim_bulge=0.0, warp=360,
                                         twist=180, azimuth=45, detail=7)),
                     ("thin sheet", dict(thickness=0.0, warp=360,
                                         storeys=4))]:
        p = Params(**kw)
        v, f = generate_sculpture(p)
        xs = [q[0] for q in v]; ys = [q[1] for q in v]; zs = [q[2] for q in v]
        print(f"{name:12s}: verts={len(v):7d} faces={len(f):7d} "
              f"closes={ring_closes(p)} "
              f"bbox=({min(xs):.2f}..{max(xs):.2f}, "
              f"{min(ys):.2f}..{max(ys):.2f}, {min(zs):.2f}..{max(zs):.2f})")
