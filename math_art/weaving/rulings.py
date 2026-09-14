# Weaving the two ruling families of a doubly-ruled surface.
#
# Part of the Math Art weaving engine (`math_art/weaving/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operator stays in `ruled_surface_generator`.
#
# A doubly-ruled surface carries two families of straight lines, and
# through each of its points passes exactly one line of each family.
# Draw a finite set from each and the two sets cross in a lattice -- the
# look of a stick hyperboloid or a string-art saddle.  Here every line is
# instead a flat ribbon lying in the surface, and at each crossing one
# ribbon passes over the other, so the two families interlace the way
# warp and weft do in cloth.
#
# Nothing below knows which surface it is weaving.  It is handed the two
# families as plain segments and recovers everything else from them:
#
#   * the crossings, as the pairs of segments that actually meet;
#   * the surface normal at each crossing, as the cross product of the
#     two ruling directions.  That is exact rather than an estimate: the
#     two rulings through a point both lie in the surface, so together
#     they span its tangent plane there;
#   * the over/under assignment, from an integer LEVEL on the crossings
#     that rises by one per step along either strand.  A plain weave is
#     the level's parity and an n/n twill is the level taken mod 2n.  A
#     crossing graph that admits no such level is counted and reported,
#     never silently woven wrong.
#
# So a surface gains the woven output by supplying its two families,
# each oriented consistently (every strand of a family running the same
# way across the surface -- bottom to top, say).
#
# References:
# - D. Hilbert and S. Cohn-Vossen, "Anschauliche Geometrie" (1932),
#   chapter 1 -- the doubly ruled quadrics, with the two rulings through
#   every point spanning the tangent plane there.
# - B. Grunbaum and G. C. Shephard, "Satins and Twills: An Introduction
#   to the Geometry of Fabrics," Mathematics Magazine 53 (1980),
#   139-161 -- plain weave and twills as over/under patterns on the
#   lattice of crossings of two strand families.
# - G. W. Hart, "Curved, yet Straight: Stick Hyperboloids," Bridges
#   2023 Conference Proceedings, pp. 251-258.

import math

import numpy as np


# --------------------------------------------------------------------
# crossings
# --------------------------------------------------------------------

def segment_crossings(fam_a, fam_b, rel_tol=1e-7):
    """Every place a segment of `fam_a` meets a segment of `fam_b`.

    Each family is a sequence of (p0, p1) segments.  Two lines meet when
    their closest approach is (numerically) zero and falls inside both
    segments; parallel pairs never meet.  Endpoints count, so two rulings
    that start from one point on a boundary rail cross there.

    Returns a dict of arrays, one entry per crossing:
        ia, ta      index into fam_a, and parameter along it (0..1)
        ib, tb      the same for fam_b
        point       the crossing point
        normal      unit surface normal, cross(dir_a, dir_b) normalized
        sin         sine of the angle between the two rulings
    """
    A = np.asarray(fam_a, dtype=float).reshape(-1, 2, 3)
    B = np.asarray(fam_b, dtype=float).reshape(-1, 2, 3)
    empty = dict(ia=np.zeros(0, int), ta=np.zeros(0), ib=np.zeros(0, int),
                 tb=np.zeros(0), point=np.zeros((0, 3)),
                 normal=np.zeros((0, 3)), sin=np.zeros(0))
    if len(A) == 0 or len(B) == 0:
        return empty
    A0, u = A[:, 0], A[:, 1] - A[:, 0]
    B0, v = B[:, 0], B[:, 1] - B[:, 0]
    tol = rel_tol * max(float(np.abs(A).max()), float(np.abs(B).max()),
                        1e-12)
    w0 = A0[:, None, :] - B0[None, :, :]
    a = np.einsum('ij,ij->i', u, u)[:, None]
    c = np.einsum('ij,ij->i', v, v)[None, :]
    b = u @ v.T
    d = np.einsum('ik,ijk->ij', u, w0)
    e = np.einsum('jk,ijk->ij', v, w0)
    den = a * c - b * b
    ok = (a > 0) & (c > 0) & (den > 1e-12 * a * c)
    safe = np.where(ok, den, 1.0)
    s = (b * e - c * d) / safe
    t = (a * e - b * d) / safe
    ra = tol / np.sqrt(np.where(a > 0, a, 1.0))
    rc = tol / np.sqrt(np.where(c > 0, c, 1.0))
    ok &= (s >= -ra) & (s <= 1.0 + ra) & (t >= -rc) & (t <= 1.0 + rc)
    ia, ib = np.nonzero(ok)
    if len(ia) == 0:
        return empty
    s = np.clip(s[ia, ib], 0.0, 1.0)
    t = np.clip(t[ia, ib], 0.0, 1.0)
    pa = A0[ia] + s[:, None] * u[ia]
    pb = B0[ib] + t[:, None] * v[ib]
    hit = np.linalg.norm(pa - pb, axis=1) <= 10.0 * tol
    ia, ib, s, t, pa, pb = ia[hit], ib[hit], s[hit], t[hit], pa[hit], pb[hit]
    ua = u[ia] / np.linalg.norm(u[ia], axis=1, keepdims=True)
    vb = v[ib] / np.linalg.norm(v[ib], axis=1, keepdims=True)
    n = np.cross(ua, vb)
    sn = np.linalg.norm(n, axis=1)
    return dict(ia=ia, ta=s, ib=ib, tb=t, point=(pa + pb) / 2.0,
                normal=n / sn[:, None], sin=sn)


# --------------------------------------------------------------------
# over / under
# --------------------------------------------------------------------

def weave_levels(ia, ta, ib, tb, run=1):
    """Integer level per crossing, rising by one per step along a strand.

    Consecutive crossings along any strand of either family are joined
    by an edge demanding level(next) = level(prev) + 1.  On a lattice of
    crossings -- which is what two ruling families form -- such a level
    exists, and then

        family A passes over  <=>  (level // run) is even

    alternates in runs of `run` along every strand of BOTH families at
    once: run 1 is a plain weave, run 2 a 2/2 twill.  The level is
    assigned by a breadth-first walk and every edge is then re-checked.
    A plain weave only needs the PARITY of the step, so for run 1 an edge
    counts as a conflict only when it joins two crossings of equal
    parity (a strand reversed in direction is still fine); for longer
    runs the step must be exactly +1.

    Returns (levels, conflicts).
    """
    m = len(ia)
    nbrs = [[] for _ in range(m)]
    edges = []
    for idx, par in ((np.asarray(ia), np.asarray(ta)),
                     (np.asarray(ib), np.asarray(tb))):
        if m == 0:
            break
        order = np.lexsort((par, idx))
        for p, q in zip(order[:-1], order[1:]):
            if idx[p] == idx[q]:
                nbrs[p].append((q, 1))
                nbrs[q].append((p, -1))
                edges.append((p, q))
    level = [None] * m
    # seed each connected piece at its lowest (strand, parameter)
    # crossing, so the pattern's phase does not depend on dict order
    seeds = np.lexsort((np.asarray(ta), np.asarray(ia))) if m else []
    for s in seeds:
        if level[s] is not None:
            continue
        level[s] = 0
        queue = [s]
        while queue:
            p = queue.pop()
            for q, step in nbrs[p]:
                if level[q] is None:
                    level[q] = level[p] + step
                    queue.append(q)
    level = np.asarray(level, dtype=int) if m else np.zeros(0, int)
    conflicts = 0
    for p, q in edges:
        diff = int(level[q] - level[p])
        if (diff % 2 != 1) if run <= 1 else (diff != 1):
            conflicts += 1
    return level, conflicts


# --------------------------------------------------------------------
# ribbon plan
# --------------------------------------------------------------------

def _segment_distance(p0, p1, q0, q1):
    """Shortest distance between two 3-D segments."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1e-18
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, min(1.0, max(0.0, f / e))
    else:
        c = float(d1 @ r)
        if e <= eps:
            s, t = min(1.0, max(0.0, -c / a)), 0.0
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / den)) if den > eps \
                else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = min(1.0, max(0.0, -c / a)), 0.0
            elif t > 1.0:
                s, t = min(1.0, max(0.0, (b - c) / a)), 1.0
    return float(np.linalg.norm(p0 + s * d1 - (q0 + t * d2)))


def family_gap(fam):
    """Narrowest distance between neighbouring strands of one family
    (consecutive in list order, the last wrapping to the first).  A ribbon
    wider than this would overlap its neighbour edge to edge."""
    S = np.asarray(fam, dtype=float).reshape(-1, 2, 3)
    if len(S) < 2:
        return math.inf
    return min(_segment_distance(S[i, 0], S[i, 1],
                                 S[(i + 1) % len(S), 0],
                                 S[(i + 1) % len(S), 1])
               for i in range(len(S)))


#: fraction of the span between two crossings that is kept clear of
#: both footprints, for the ribbon to change level in
_FREE = 0.15


def plan_weave(fam_a, fam_b, width=0.7, thickness=0.15, run=1):
    """Everything needed to sweep the woven ribbons, before any mesh.

    `width` is a FRACTION of the widest ribbon that weaves cleanly on
    these two families (see below), so any value up to 1 weaves without
    the ribbons touching, whatever the surface or strand count.
    `thickness` is a fraction of the ribbon width.  At each crossing the upper ribbon is lifted one thickness
    along the surface normal and the lower one sunk by the same, leaving
    a clear gap of one thickness between them.

    Around each crossing the lift is held flat over the whole footprint
    of the other ribbon -- measured along this strand, a ribbon of width
    w crossing at angle theta covers (w/2)(1 + |cos theta|)/sin theta
    either side of the centre -- and the ribbon only changes level in
    the space left between footprints.  The width limit is the largest
    that leaves a share _FREE of every span between consecutive
    crossings for that change.  Past it (width > 1) the ribbons are too
    wide to weave cleanly; the offending spans are counted in `tight`.
    """
    fam_a = [tuple(map(tuple, s)) for s in fam_a]
    fam_b = [tuple(map(tuple, s)) for s in fam_b]
    X = segment_crossings(fam_a, fam_b)
    run = max(1, int(run))
    level, conflicts = weave_levels(X['ia'], X['ta'], X['ib'], X['tb'], run)
    a_over = (level // run) % 2 == 0

    # footprint half-length of a crossing ribbon along either strand,
    # per unit of ribbon width
    sin_ = np.maximum(X['sin'], 1e-3)
    cos_ = np.sqrt(np.maximum(0.0, 1.0 - sin_ * sin_))
    per_w = 0.5 * (1.0 + cos_) / sin_

    # The widest ribbon that still weaves cleanly.  Between every pair
    # of consecutive crossings on a strand the two footprints must leave
    # a fraction _FREE of the span free for the ribbon to change level
    # in; and no ribbon may be wider than the gap to its neighbour in its
    # own family.  Oblique crossings are what bind: at 48 degrees a
    # footprint is 1.1 widths long, so a saddle whose lattice shears
    # toward its edges needs far narrower ribbons than its spacing alone
    # suggests.
    lengths = [np.linalg.norm(np.diff(np.asarray(f, dtype=float)
                                      .reshape(-1, 2, 3), axis=1)[:, 0],
                              axis=1) for f in (fam_a, fam_b)]
    limit = min(family_gap(fam_a), family_gap(fam_b))
    for (key_i, key_t), lens in zip((('ia', 'ta'), ('ib', 'tb')), lengths):
        idx, par = X[key_i], X[key_t]
        if len(idx) < 2:
            continue
        order = np.lexsort((par, idx))
        same = idx[order[:-1]] == idx[order[1:]]
        p, q = order[:-1][same], order[1:][same]
        span = (par[q] - par[p]) * lens[idx[p]]
        ok = span > 1e-12
        if np.any(ok):
            limit = min(limit, float(np.min(
                (1.0 - _FREE) * span[ok] / (per_w[p][ok] + per_w[q][ok]))))
    longest = max((float(np.max(ln)) for ln in lengths if len(ln)),
                  default=1.0)
    if not math.isfinite(limit) or limit <= 0.0:
        limit = 0.05 * longest
    w = width * limit
    th = thickness * w
    amp = th
    foot = w * per_w

    fallback = (np.mean(X['normal'], axis=0) if len(X['ia'])
                else np.array([0.0, 0.0, 1.0]))
    strands = []
    tight = 0
    for fam, key_i, key_t, sign in ((fam_a, 'ia', 'ta', 1.0),
                                    (fam_b, 'ib', 'tb', -1.0)):
        for i, (p0, p1) in enumerate(fam):
            p0 = np.asarray(p0, dtype=float)
            p1 = np.asarray(p1, dtype=float)
            L = float(np.linalg.norm(p1 - p0))
            if L < 1e-12:
                continue
            T = (p1 - p0) / L
            sel = np.nonzero(X[key_i] == i)[0]
            sel = sel[np.argsort(X[key_t][sel])]
            tk = X[key_t][sel]
            sg = np.where(a_over[sel], sign, -sign)
            hk = foot[sel] / L
            lo = tk[:-1] + hk[:-1]
            hi = tk[1:] - hk[1:]
            for j in range(len(lo)):
                if sg[j] == sg[j + 1]:
                    continue
                span = tk[j + 1] - tk[j]
                if hi[j] - lo[j] < (_FREE - 1e-6) * span:
                    tight += 1
                    mid = min(max(0.5 * (lo[j] + hi[j]),
                                  tk[j] + 0.1 * span),
                              tk[j + 1] - 0.1 * span)
                    lo[j], hi[j] = mid - 0.075 * span, mid + 0.075 * span
            Nf = fallback - float(fallback @ T) * T
            if np.linalg.norm(Nf) < 1e-9:
                Nf = np.cross(T, [1.0, 0.0, 0.0])
                if np.linalg.norm(Nf) < 1e-9:
                    Nf = np.cross(T, [0.0, 1.0, 0.0])
            strands.append(dict(p0=p0, p1=p1, T=T, L=L, t=tk, sign=sg,
                                half=hk, lo=lo, hi=hi,
                                N=X['normal'][sel],
                                fallback=Nf / np.linalg.norm(Nf),
                                crossings=sel))
    return dict(strands=strands, width=w, thickness=th, amp=amp,
                crossings=X, level=level, a_over=a_over,
                conflicts=conflicts, tight=tight, run=run, limit=limit)


def strand_section(plan, st, t):
    """Ribbon centre, side and normal vectors along one strand at
    parameters `t`: the base ruling, lifted along the surface normal by
    the woven over/under profile."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    tk, sg = st['t'], st['sign']
    n = len(tk)
    base = st['p0'][None, :] + t[:, None] * (st['p1'] - st['p0'])[None, :]
    if n == 0:
        N = np.repeat(st['fallback'][None, :], len(t), axis=0)
        d = np.zeros(len(t))
    else:
        j = np.searchsorted(tk, t, side='right') - 1
        d = np.empty(len(t))
        d[j < 0] = sg[0]
        d[j >= n - 1] = sg[-1]
        mid = (j >= 0) & (j < n - 1)
        if np.any(mid):
            jm = j[mid]
            lo, hi = st['lo'][jm], st['hi'][jm]
            u = np.clip((t[mid] - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
            f = u * u * (3.0 - 2.0 * u)
            d[mid] = sg[jm] + (sg[jm + 1] - sg[jm]) * f
        jc = np.clip(j, 0, n - 1)
        jn = np.clip(j + 1, 0, n - 1)
        span = np.where(jn > jc, tk[jn] - tk[jc], 1.0)
        u = np.where(jn > jc, np.clip((t - tk[jc]) / span, 0.0, 1.0), 0.0)
        N = (1.0 - u)[:, None] * st['N'][jc] + u[:, None] * st['N'][jn]
        N /= np.linalg.norm(N, axis=1, keepdims=True)
    C = base + (plan['amp'] * d)[:, None] * N
    S = np.cross(N, st['T'][None, :])
    return C, S, N


def _sample_ts(plan, st, steps):
    """Sample parameters: every knot, footprint edge and transition end,
    with the level-changing spans subdivided `steps` times."""
    br = [0.0, 1.0]
    br.extend(st['t'])
    br.extend(st['t'] - st['half'])
    br.extend(st['t'] + st['half'])
    br.extend(st['lo'])
    br.extend(st['hi'])
    br = np.unique(np.clip(np.asarray(br, dtype=float), 0.0, 1.0))
    br = br[np.concatenate(([True], np.diff(br) > 1e-9))]
    if len(br) < 2:
        return np.array([0.0, 1.0])
    mids = 0.5 * (br[:-1] + br[1:])
    C, _S, N = strand_section(plan, st, mids)
    base = st['p0'][None, :] + mids[:, None] * (st['p1'] - st['p0'])
    lift = np.abs(np.einsum('ij,ij->i', C - base, N)) / max(plan['amp'],
                                                             1e-300)
    out = [br[0]]
    for k in range(len(br) - 1):
        m = steps if lift[k] < 1.0 - 1e-9 else 1
        out.extend(br[k] + (br[k + 1] - br[k]) * np.arange(1, m + 1) / m)
    return np.asarray(out)


def sweep_woven_ribbons(plan, steps=6):
    """Mesh the plan: each strand a closed box-section ribbon.

    Returns (verts, faces, face_strand).  Faces wind outward.
    """
    hw, ht = 0.5 * plan['width'], 0.5 * plan['thickness']
    corners = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    verts, faces, face_strand = [], [], []
    for si, st in enumerate(plan['strands']):
        ts = _sample_ts(plan, st, steps)
        C, S, N = strand_section(plan, st, ts)
        m = len(ts)
        ring = np.stack([C + sw * hw * S + sr * ht * N
                         for sw, sr in corners], axis=1)   # (m, 4, 3)
        base = len(verts)
        verts.extend(map(tuple, ring.reshape(-1, 3)))
        for i in range(m - 1):
            for j in range(4):
                j1 = (j + 1) % 4
                faces.append([base + 4 * i + j, base + 4 * i + j1,
                              base + 4 * (i + 1) + j1,
                              base + 4 * (i + 1) + j])
        faces.append([base + 3, base + 2, base + 1, base])
        last = base + 4 * (m - 1)
        faces.append([last, last + 1, last + 2, last + 3])
        face_strand.extend([si] * (4 * (m - 1) + 2))
    return verts, faces, face_strand


def weave_rulings(fam_a, fam_b, width=0.7, thickness=0.15, run=1,
                  steps=6):
    """Woven ribbons for two ruling families: (verts, faces, plan)."""
    plan = plan_weave(fam_a, fam_b, width, thickness, run)
    verts, faces, _fs = sweep_woven_ribbons(plan, steps)
    return verts, faces, plan


def crossing_clearance(plan, samples=13, across=7):
    """Smallest clear gap between the two ribbons at any crossing, in
    ribbon thicknesses (1 is the designed gap; <= 0 means they touch).

    Measured on the swept geometry itself, and only where the ribbons
    really lie over one another.  The upper ribbon's underside is
    sampled across its whole footprint; each sample is located on the
    lower ribbon by its position in the tangent plane at the crossing
    (along the lower ruling, and across it), samples that fall outside
    the lower ribbon are dropped, and the rest are compared with the
    lower ribbon's top face at that same spot, along the normal.

    Comparing like points matters on a curved surface.  Each ribbon
    follows the rotating tangent plane along its ruling, so across a
    footprint it twists by roughly (along x across) times the surface's
    curvature -- but so does the surface, and so does the other ribbon
    at the same point.  Comparing the highest point of one footprint
    with the lowest of the other would charge that shared twist to the
    gap twice and report a saddle as colliding when it is not.
    """
    X = plan['crossings']
    if len(X['ia']) == 0:
        return math.inf
    by_cross = {}
    for st in plan['strands']:
        for k, c in enumerate(st['crossings']):
            by_cross.setdefault(int(c), []).append((st, k))
    hw, ht = 0.5 * plan['width'], 0.5 * plan['thickness']
    betas = np.linspace(-hw, hw, across)
    worst = math.inf
    for c, pair in by_cross.items():
        if len(pair) != 2:
            continue
        ups = [p for p in pair if p[0]['sign'][p[1]] > 0]
        downs = [p for p in pair if p[0]['sign'][p[1]] < 0]
        if len(ups) != 1 or len(downs) != 1:
            continue                       # both over: a solver conflict
        (so, ko), (su, ku) = ups[0], downs[0]
        P, Nc = X['point'][c], X['normal'][c]
        tc, h = so['t'][ko], so['half'][ko]
        ts = np.clip(np.linspace(tc - h, tc + h, samples), 0.0, 1.0)
        C, S, N = strand_section(plan, so, ts)
        Q = (C[:, None, :] + betas[None, :, None] * S[:, None, :]
             - ht * N[:, None, :]).reshape(-1, 3)
        # position of each underside sample in the lower ribbon's frame
        side_u = np.cross(Nc, su['T'])
        d = Q - P
        tu = su['t'][ku] + (d @ su['T']) / su['L']
        bu = d @ side_u
        keep = (tu >= 0.0) & (tu <= 1.0) & (np.abs(bu) <= hw)
        if not np.any(keep):
            continue
        Cu, Su, Nu = strand_section(plan, su, tu[keep])
        top = Cu + bu[keep, None] * Su + ht * Nu
        gap = float(np.min((Q[keep] - top) @ Nc)) / plan['thickness']
        worst = min(worst, gap)
    return worst


# --------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------

def _signed_volume(verts, faces):
    V = np.asarray(verts, dtype=float)
    vol = 0.0
    for f in faces:
        for i in range(1, len(f) - 1):
            vol += float(V[f[0]] @ np.cross(V[f[i]], V[f[i + 1]])) / 6.0
    return vol


def _selftest():
    # A flat square grid is the simplest doubly-ruled "surface": rows and
    # columns in the plane z = 0, every row crossing every column.
    n = 6
    rows = [((-0.5, float(j), 0.0), (n - 0.5, float(j), 0.0))
            for j in range(n)]
    cols = [((float(i), -0.5, 0.0), (float(i), n - 0.5, 0.0))
            for i in range(n)]
    X = segment_crossings(rows, cols)
    assert len(X['ia']) == n * n, len(X['ia'])
    assert np.allclose(X['normal'], [0.0, 0.0, 1.0]), "normal not +z"
    assert np.allclose(X['sin'], 1.0)
    # same-family strands are parallel and never meet
    assert len(segment_crossings(rows, rows)['ia']) == 0
    print(f"rulings: {n}x{n} grid -> {n * n} crossings, normals +z OK")

    plan = plan_weave(rows, cols, width=0.7, thickness=0.15, run=1)
    assert plan['conflicts'] == 0 and plan['tight'] == 0, plan['tight']
    # square crossings one unit apart: each footprint is half a width
    # either side, so the clean limit is (1 - _FREE) of the unit span
    assert abs(plan['width'] - 0.7 * (1.0 - _FREE)) < 1e-12, plan['width']
    # plain weave: a checkerboard, and so alternating along every strand
    X = plan['crossings']
    over = {(int(X['ib'][c]), int(X['ia'][c])): bool(plan['a_over'][c])
            for c in range(len(X['ia']))}
    ref = over[(0, 0)]
    assert all(over[(i, j)] == (ref == ((i + j) % 2 == 0))
               for i in range(n) for j in range(n)), "not a checkerboard"
    for st in plan['strands']:
        assert np.all(st['sign'][1:] != st['sign'][:-1]), "no alternation"
    clear = crossing_clearance(plan)
    assert clear > 0.5, clear
    print(f"rulings: plain weave is a checkerboard, alternates on every "
          f"strand, clearance {clear:.2f} thickness OK")

    # every ribbon is a closed, outward-wound box of volume w * t * L
    verts, faces, fs = sweep_woven_ribbons(plan)
    fs = np.asarray(fs)
    for si, st in enumerate(plan['strands']):
        F = [faces[k] for k in np.nonzero(fs == si)[0]]
        vol = _signed_volume(verts, F)
        want = plan['width'] * plan['thickness'] * st['L']
        assert vol > 0 and abs(vol - want) < 1e-9 * want + 1e-12, \
            (si, vol, want)
    edges = {}
    for f in faces:
        for k in range(len(f)):
            e = (f[k], f[(k + 1) % len(f)])
            edges[e] = edges.get(e, 0) + 1
    assert all(edges.get((b, a), 0) == 1 and cnt == 1
               for (a, b), cnt in edges.items()), "not closed / manifold"
    print(f"rulings: {len(plan['strands'])} ribbons closed, outward, "
          f"volume = width x thickness x length OK")

    # a 2/2 twill: interior runs along every strand have length 2
    plan2 = plan_weave(rows, cols, width=0.6, run=2)
    assert plan2['conflicts'] == 0
    for st in plan2['strands']:
        sg = list(st['sign'])
        runs, k = [], 0
        while k < len(sg):
            k2 = k
            while k2 < len(sg) and sg[k2] == sg[k]:
                k2 += 1
            runs.append(k2 - k)
            k = k2
        assert all(r == 2 for r in runs[1:-1]), runs
    assert crossing_clearance(plan2) > 0.5
    print("rulings: 2/2 twill floats over two, under two OK")

    # direction matters only beyond a plain weave: reverse every other
    # row and the parity survives, but a twill's level cannot
    flipped = [(s[1], s[0]) if j % 2 else s for j, s in enumerate(rows)]
    assert plan_weave(flipped, cols, run=1)['conflicts'] == 0
    assert plan_weave(flipped, cols, run=2)['conflicts'] > 0
    print("rulings: reversed strands keep a plain weave, flag a twill OK")

    # the width limit is exact: at 1 nothing is tight, just past it the
    # spans are flagged, and the clearance measure sees the squeeze
    assert plan_weave(rows, cols, width=1.0)['tight'] == 0
    wide = plan_weave(rows, cols, width=1.02)
    assert wide['tight'] > 0
    assert crossing_clearance(wide) < crossing_clearance(plan)
    print(f"rulings: width limit exact -- clean at 1, {wide['tight']} "
          f"tight spans at 1.02 OK")

    # a genuinely curved case: the saddle z = xy is doubly ruled by the
    # lines x = const and y = const; the crossing normal must match the
    # analytic one, (-y, -x, 1) normalized
    k = np.linspace(-1.0, 1.0, 9)
    fa = [((x, -1.0, -x), (x, 1.0, x)) for x in k]
    fb = [((-1.0, y, -y), (1.0, y, y)) for y in k]
    X = segment_crossings(fa, fb)
    assert len(X['ia']) == 81
    P = X['point']
    want = np.stack([-P[:, 1], -P[:, 0], np.ones(len(P))], axis=1)
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    assert np.max(np.abs(X['normal'] + want)) < 1e-12 or \
        np.max(np.abs(X['normal'] - want)) < 1e-12
    sp = plan_weave(fa, fb, width=0.7)
    assert sp['conflicts'] == 0 and sp['tight'] == 0
    assert crossing_clearance(sp) > 0.5
    full = plan_weave(fa, fb, width=1.0)
    assert full['tight'] == 0 and crossing_clearance(full) > 0.5
    print("rulings: saddle z = xy crossing normals exact, woven clean OK")
    print("RESULT: OK")
