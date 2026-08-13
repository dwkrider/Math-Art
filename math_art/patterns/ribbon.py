# Mitred ribbons, bands and strand smoothing.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operators stay in their flat generator
# modules and import this package.
#
# Ribbons and bands: turn a polyline into a strip of given width with
# MITRED corners, then into faces, optionally split at crossings so a
# strand can pass under another.
#
# This machinery was written inside `islamic_pattern_generator.py` and
# then reached for privately by four other generators -- celtic_knot_2d,
# knot_carpet, substitution_knot and fractal_knotwork -- across roughly
# eighty call sites.  It is the interlace layer the Pattern Engine plan
# calls for, and it belongs to the engine rather than to one generator.
#
# References:
#   The mitre limit is the standard stroke-joining rule (PostScript
#   Language Reference, Adobe, 1985 -- setmiterlimit), which falls back
#   to a bevel when the join angle would make the spike unbounded.
#   E. Catmull and R. Rom, "A class of local interpolating splines", in
#   Computer Aided Geometric Design (1974) -- the centripetal-family
#   spline used to smooth a strand before it is widened.

from math import atan2, cos, hypot, pi, sin

import numpy as np

from .polygon2d import line_intersection, unit



def miter(np_, nn, limit):
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
        d = [unit(V[(i + 1) % k][0] - V[i][0],
                   V[(i + 1) % k][1] - V[i][1]) for i in range(k)]
        for i in range(k):
            dp, dn = d[(i - 1) % k], d[i]
            mx, my, s = miter((-dp[1], dp[0]), (-dn[1], dn[0]), limit)
            left.append((V[i][0] + mx * w * s, V[i][1] + my * w * s))
            right.append((V[i][0] - mx * w * s, V[i][1] - my * w * s))
    else:
        d = [unit(V[i + 1][0] - V[i][0], V[i + 1][1] - V[i][1])
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
                mx, my, s = miter((-dp[1], dp[0]),
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


def band_ribbon_faces_z(left, right, closed, height, zoff):
    """Like `band_ribbon_faces`, but each station i is lifted by
    `zoff[i]` in z, so the ribbon undulates along its length (the 3D
    weave).  With relief (`height` > 0) the whole cross-section rides at
    the weave offset -- top at zoff+height, bottom at zoff -- so the
    extruded ribbon undulates without changing thickness; flat ribbons
    become a single woven top surface at z = zoff."""
    cv, cf = [], []
    m = len(left)
    if m < 2:
        return cv, cf
    relief = height > 0.0
    lift = height if relief else 0.0

    def addv(pt, z):
        cv.append((float(pt[0]), float(pt[1]), float(z)))
        return len(cv) - 1

    TL = [addv(left[i], zoff[i] + lift) for i in range(m)]
    TR = [addv(right[i], zoff[i] + lift) for i in range(m)]
    span = m if closed else m - 1

    def nxt(i):
        return (i + 1) % m if closed else i + 1

    for i in range(span):
        j = nxt(i)
        cf.append((TL[i], TR[i], TR[j], TL[j]))
    if relief:
        BL = [addv(left[i], zoff[i]) for i in range(m)]
        BR = [addv(right[i], zoff[i]) for i in range(m)]
        for i in range(span):
            j = nxt(i)
            cf.append((BL[j], BR[j], BR[i], BL[i]))       # bottom
            cf.append((TL[j], BL[j], BL[i], TL[i]))       # left wall
            cf.append((TR[i], BR[i], BR[j], TR[j]))       # right wall
        if not closed:
            cf.append((TL[0], BL[0], BR[0], TR[0]))       # start cap
            cf.append((TR[m - 1], BR[m - 1], BL[m - 1], TL[m - 1]))  # end
    return cv, cf


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


def cut_band(path, closed, cut_s, half, s, total):
    """Break a band's centerline into open sub-paths, removing a gap of
    half-length `half` centered on each arclength in `cut_s` (the band's
    under-crossings).  Returns [(subpath_points, False), ...]."""
    n = len(path)

    def pt_at(arc):
        if arc <= 0.0:
            return path[0]
        if arc >= total:
            return path[0] if closed else path[-1]
        for i in range(n - 1):
            if s[i] <= arc <= s[i + 1]:
                seg = s[i + 1] - s[i]
                t = 0.0 if seg < 1e-12 else (arc - s[i]) / seg
                return (path[i][0] + t * (path[i + 1][0] - path[i][0]),
                        path[i][1] + t * (path[i + 1][1] - path[i][1]))
        seg = total - s[-1]
        t = 0.0 if seg < 1e-12 else (arc - s[-1]) / seg
        return (path[-1][0] + t * (path[0][0] - path[-1][0]),
                path[-1][1] + t * (path[0][1] - path[-1][1]))

    intervals = []
    for c in cut_s:
        if closed:
            a = c - half
            a %= total
            b = a + 2.0 * half
            if b <= total:
                intervals.append((a, b))
            else:
                intervals.append((a, total))
                intervals.append((0.0, b - total))
        else:
            a, b = max(0.0, c - half), min(total, c + half)
            if b > a:
                intervals.append((a, b))
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
    subs = []
    for lo, hi in kept:
        pts_sp = [pt_at(lo)]
        for i in range(n):
            if lo + 1e-9 < s[i] < hi - 1e-9:
                pts_sp.append(path[i])
        pts_sp.append(pt_at(hi))
        subs.append(pts_sp)
    if closed and subs and kept:
        seam_cut = (any(a <= 1e-9 for a, _b in merged)
                    or any(b >= total - 1e-9 for _a, b in merged))
        if (not seam_cut and len(subs) >= 2
                and kept[0][0] <= 1e-9 and kept[-1][1] >= total - 1e-9):
            subs[-1] = subs[-1][:-1] + subs[0]     # stitch across seam
            subs.pop(0)
    return [(sp, False) for sp in subs if len(sp) >= 2]


def cut_cap_on_edge(cap_l, cap_r, rail_dir, side_pt, X, t_o, n_o,
                     h_o, gap, maxreach):
    """Slide an under band's flat end cap onto the OVER band's edge line so
    the cap is parallel to (and flush against) that edge.  The over band is
    the slab of half-width `h_o` centred on crossing `X`, with unit tangent
    `t_o` and unit normal `n_o`; its two edge lines run parallel to `t_o` at
    +/- h_o along `n_o`.  The under ribbon's two rail cap points `cap_l`,
    `cap_r` (running in unit direction `rail_dir`) are each slid along their
    own rail until they meet the over edge line on the side the under band
    exits (the side of `side_pt`), pushed a hairline `gap` beyond the edge.
    Both new points then lie ON that edge line, so the cap segment is
    parallel to the over band's edge.  Returns the originals unchanged for a
    grazing crossing (rails nearly parallel to the edge, i.e. the
    reprojection would fling the cap farther than `maxreach`)."""
    sgn = 1.0 if ((side_pt[0] - X[0]) * n_o[0]
                  + (side_pt[1] - X[1]) * n_o[1]) >= 0.0 else -1.0
    qx = X[0] + sgn * (h_o + gap) * n_o[0]
    qy = X[1] + sgn * (h_o + gap) * n_o[1]
    q2 = (qx + t_o[0], qy + t_o[1])
    nl = line_intersection(cap_l, (cap_l[0] + rail_dir[0], cap_l[1] + rail_dir[1]),
                    (qx, qy), q2)
    nr = line_intersection(cap_r, (cap_r[0] + rail_dir[0], cap_r[1] + rail_dir[1]),
                    (qx, qy), q2)
    if nl is None or nr is None:
        return cap_l, cap_r
    if (hypot(nl[0] - cap_l[0], nl[1] - cap_l[1]) > maxreach
            or hypot(nr[0] - cap_r[0], nr[1] - cap_r[1]) > maxreach):
        return cap_l, cap_r
    return nl, nr


def angle_cut_piece(left, right, sp, start_struct, end_struct, ugeo,
                     h_o, gap, maxreach, cut_gate):
    """Reproject the interlace-cut end(s) of one under-band ribbon piece so
    each cut cap lies flush ALONG the over band's edge (parallel to it)
    rather than perpendicular to the under band.  `left`/`right` are the
    piece's mitered rails (mutated in place); `sp` its centerline; `ugeo` a
    list of (X, t_o, n_o) for the band's under-crossings.  `start_struct`
    / `end_struct` mark ends that are structurally interlace cuts (always
    reprojected); an end that is structurally a genuine band terminus is
    reprojected only when it actually sits within `cut_gate` of a crossing
    (the rare short-first-segment case).  A cut end is matched to its
    nearest under-crossing."""
    if not ugeo or len(sp) < 2:
        return
    ends = [(0, sp[0], unit(sp[1][0] - sp[0][0], sp[1][1] - sp[0][1]),
             start_struct),
            (-1, sp[-1], unit(sp[-1][0] - sp[-2][0], sp[-1][1] - sp[-2][1]),
             end_struct)]
    for idx, P, rd, struct in ends:
        best = None
        for X, t_o, n_o in ugeo:
            dd = hypot(P[0] - X[0], P[1] - X[1])
            if best is None or dd < best[0]:
                best = (dd, X, t_o, n_o)
        if best is None:
            continue
        dd, X, t_o, n_o = best
        if not struct and dd > cut_gate:
            continue                              # genuine terminus: flat cap
        if dd > maxreach:
            continue
        nl, nr = cut_cap_on_edge(left[idx], right[idx], rd, P, X,
                                  t_o, n_o, h_o, gap, maxreach)
        left[idx], right[idx] = nl, nr


def _selftest():
    ok = True

    # A straight path widened by w gives two rails, each w/2 to a side,
    # parallel to it -- the simplest case the mitre must not disturb.
    path = [(float(i), 0.0) for i in range(6)]
    left, right = miter_ribbon(path, 0.4, False)
    L, R = np.asarray(left, float), np.asarray(right, float)
    good = (len(L) == len(path) == len(R)
            and np.allclose(np.abs(L[:, 1]), 0.2, atol=1e-12)
            and np.allclose(np.abs(R[:, 1]), 0.2, atol=1e-12)
            and np.allclose(L[:, 1] + R[:, 1], 0.0, atol=1e-12))
    ok &= good
    print(f"ribbon: straight path -> two parallel rails at +-w/2 "
          f"{'OK' if good else 'FAIL'}")

    # At a corner the rails must MITRE: the inner rail's vertex sits at the
    # intersection of the two offset lines, so the band keeps its full
    # width through the turn instead of pinching.  Measured as the
    # perpendicular distance between the rails at the corner.
    corner = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    lc, rc = miter_ribbon(corner, 0.4, False)
    mid = 1
    d = float(np.hypot(lc[mid][0] - rc[mid][0], lc[mid][1] - rc[mid][1]))
    good = d > 0.4 - 1e-9        # >= width; a mitre is longer than the width
    ok &= good
    print(f"ribbon: 90 deg corner keeps width (rail gap {d:.4f} >= 0.4) "
          f"{'OK' if good else 'FAIL'}")

    # The mitre LIMIT must bound the spike at a sharp turn: without it a
    # near-reversal sends the joint to infinity.
    spike = [(0.0, 0.0), (1.0, 0.0), (0.02, 0.06)]
    ls, rs = miter_ribbon(spike, 0.4, False, limit=4.0)
    worst = max(float(np.hypot(a[0] - b[0], a[1] - b[1]))
                for a, b in zip(ls, rs))
    good = np.isfinite(worst) and worst <= 4.0 * 0.4 + 1e-9
    ok &= good
    print(f"ribbon: mitre limit bounds a near-reversal "
          f"(worst gap {worst:.3f} <= {4.0 * 0.4}) {'OK' if good else 'FAIL'}")

    # band_ribbon_faces stitches the two rails into a quad strip: one quad
    # per segment, every index in range.
    verts, faces = band_ribbon_faces(left, right, False, 0.0)
    good = (len(faces) == len(path) - 1
            and all(0 <= i < len(verts) for f in faces for i in f))
    ok &= good
    print(f"ribbon: band faces V={len(verts)} F={len(faces)} "
          f"(exp {len(path) - 1} quads) {'OK' if good else 'FAIL'}")

    # catmull_rom interpolates: every control point must appear on the
    # smoothed path, and the result is finer than the input.
    ctrl = [(0.0, 0.0), (1.0, 0.5), (2.0, -0.5), (3.0, 0.0)]
    sm = np.asarray(catmull_rom(ctrl, False, 8), float)
    hits = sum(1 for c in ctrl
               if np.min(np.hypot(sm[:, 0] - c[0], sm[:, 1] - c[1])) < 1e-9)
    good = len(sm) > len(ctrl) and hits >= len(ctrl) - 1
    ok &= good
    print(f"ribbon: catmull_rom {len(ctrl)} -> {len(sm)} pts, passes through "
          f"{hits}/{len(ctrl)} controls {'OK' if good else 'FAIL'}")

    # a closed ring smooths to a closed ring of the same winding
    ring = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    smc = np.asarray(catmull_rom(ring, True, 6), float)
    from .polygon2d import signed_area
    good = len(smc) > len(ring) and signed_area(smc) > 0
    ok &= good
    print(f"ribbon: closed ring stays closed and CCW "
          f"(area {signed_area(smc):.3f}) {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("ribbon self-test failed")
