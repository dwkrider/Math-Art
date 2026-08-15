# Scherk-Collins saddle towers: the toroidal sculptures of Brent Collins.
#
# Part of the Math Art minsurf engine (`math_art/minsurf/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# A Scherk saddle tower is the singly-periodic minimal surface Scherk
# found in 1835: a stack of saddles, each rotated from the last, with n
# arms per storey.  Bending the stack around a circle and closing it up
# gives the toroidal form Brent Collins carved in wood and Carlo Sequin
# parameterised as the `Scherk-Collins` family -- storeys, arms, the
# twist applied around the ring, and the flange thickness.
#
# References:
# - H. F. Scherk, "Bemerkungen uber die kleinste Flache innerhalb
#   gegebener Grenzen", Journal fur die reine und angewandte Mathematik
#   13, 1835, pp. 185-208 -- the singly-periodic saddle tower.
# - C. H. Sequin, "Analogies from 2D to 3D -- Exercises in Disciplined
#   Creativity", Bridges 1999, and "Scherk-Collins Sculptures",
#   describing the parameterisation of Brent Collins' wood sculptures.

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
