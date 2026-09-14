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
# - C. Ericson, "Real-Time Collision Detection" (Morgan Kaufmann, 2005),
#   sec. 5.1.9 -- closest points of two segments, used to find rods
#   that pass through each other.
# - M. Muller, B. Heidelberger, M. Hennix and J. Ratcliff, "Position
#   Based Dynamics," Journal of Visual Communication and Image
#   Representation 18 (2007), 109-118 -- the constraint projection that
#   bends touching rods apart.
#
# The same module also separates solid rods: where two straight rods of
# a given radius would pass through each other, both are bent aside just
# enough to clear, with their ends left on their rails (see
# `separate_rods`).

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


def extend_segments(segs, amount):
    """Lengthen every segment by `amount` past BOTH of its ends, along
    its own direction; a negative amount trims it back instead, and a
    segment trimmed to nothing is dropped."""
    out = []
    for p0, p1 in segs:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 1e-12 or L + 2.0 * amount <= 1e-9 * L:
            continue
        d = (p1 - p0) / L
        out.append((tuple(float(c) for c in p0 - amount * d),
                    tuple(float(c) for c in p1 + amount * d)))
    return out


def extend_families(fam_a, fam_b, overhang):
    """Both families run on past their ends by `overhang` times the
    longest segment of either -- one absolute distance for every strand,
    so ends that sat on a common edge stay level with each other.

    On a doubly-ruled surface the extended lines still lie on the
    surface, so they go on crossing: weaving the extended families
    carries the weave out past the edge rather than just poking the
    ribbon ends through it.
    """
    if not overhang:
        return list(fam_a), list(fam_b)
    segs = np.asarray(list(fam_a) + list(fam_b),
                      dtype=float).reshape(-1, 2, 3)
    if len(segs) == 0:
        return [], []
    longest = float(np.max(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)))
    amount = overhang * longest
    return extend_segments(fam_a, amount), extend_segments(fam_b, amount)


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

def segment_distance(p0, p1, q0, q1):
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


def _pairwise_closest(P0a, Da, P0b, Db):
    """Clamped closest points between every segment of one set (starts
    P0a, spans Da) and every segment of another: arrays s, t and dist,
    each shaped (len(a), len(b)).  C. Ericson, "Real-Time Collision
    Detection" (2005), sec. 5.1.9, vectorized."""
    r = P0a[:, None, :] - P0b[None, :, :]
    a = np.einsum('ij,ij->i', Da, Da)[:, None]
    e = np.einsum('ij,ij->i', Db, Db)[None, :]
    b = Da @ Db.T
    c = np.einsum('ik,ijk->ij', Da, r)
    f = np.einsum('jk,ijk->ij', Db, r)
    den = a * e - b * b
    ok = den > 1e-12 * a * e
    a_safe = np.where(a > 0, a, 1.0)
    s = np.where(ok, np.clip((b * f - c * e) / np.where(ok, den, 1.0),
                             0.0, 1.0), 0.0)
    t = (b * s + f) / np.where(e > 0, e, 1.0)
    s = np.where(t < 0.0, np.clip(-c / a_safe, 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b - c) / a_safe, 0.0, 1.0), s)
    t = np.clip(t, 0.0, 1.0)
    gap = r + s[..., None] * Da[:, None, :] - t[..., None] * Db[None, :, :]
    return s, t, np.linalg.norm(gap, axis=-1)


def segment_contacts(segs, reach, chunk=256):
    """Every pair of segments i < j whose closest approach is under
    `reach`, found for all pairs at once (in row chunks, to bound
    memory).

    Closest points are clamped to the segments (C. Ericson, "Real-Time
    Collision Detection", 2005, sec. 5.1.9), so a pair that only nears
    at an end reports that end: s or t of exactly 0 or 1.

    Returns a dict of arrays: i, j, s, t, dist, pa, pb.
    """
    S = np.asarray(segs, dtype=float).reshape(-1, 2, 3)
    m = len(S)
    P0, D = S[:, 0], S[:, 1] - S[:, 0]
    acc = {k: [] for k in ('i', 'j', 's', 't', 'dist')}
    for lo in range(0, m, chunk):
        ii = np.arange(lo, min(m, lo + chunk))
        s, t, dist = _pairwise_closest(P0[ii], D[ii], P0, D)
        hit = (dist < reach) & (np.arange(m)[None, :] > ii[:, None])
        pi, pj = np.nonzero(hit)
        acc['i'].append(ii[pi])
        acc['j'].append(pj)
        acc['s'].append(s[pi, pj])
        acc['t'].append(t[pi, pj])
        acc['dist'].append(dist[pi, pj])
    out = {k: (np.concatenate(v) if v else np.zeros(0))
           for k, v in acc.items()}
    out['i'] = out['i'].astype(int)
    out['j'] = out['j'].astype(int)
    out['pa'] = P0[out['i']] + out['s'][:, None] * D[out['i']]
    out['pb'] = P0[out['j']] + out['t'][:, None] * D[out['j']]
    return out


# --------------------------------------------------------------------
# separating rods that pass through each other
# --------------------------------------------------------------------

#: in rod radii: the clear gap left between two rods, and the spacing of
#: the points a bent rod is carried through
_ROD_GAP = 0.25
_ROD_SPACING = 2.0


def _polyline_length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _resample(P, n):
    """n + 1 points spaced evenly by arc length along polyline P, keeping
    both ends."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    t = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)


def _coarse(P, pieces=16):
    """At most `pieces` pieces through P's own points, and how far P
    strays from them: a cheap stand-in for the proximity search, whose
    reach is widened by the stray so nothing near is missed."""
    if len(P) <= pieces + 1:
        return P, 0.0
    idx = np.unique(np.rint(np.linspace(0, len(P) - 1,
                                        pieces + 1)).astype(int))
    C = P[idx]
    stray = 0.0
    for k in range(len(idx) - 1):
        pts = P[idx[k]:idx[k + 1] + 1]
        a, d = C[k], C[k + 1] - C[k]
        dd = float(d @ d)
        tt = (np.clip(((pts - a) @ d) / dd, 0.0, 1.0) if dd > 0.0
              else np.zeros(len(pts)))
        stray = max(stray, float(np.linalg.norm(
            pts - (a + tt[:, None] * d), axis=1).max()))
    return C, stray


def _piece_distances(Pa, Pb):
    """Clamped closest points between every piece of polyline Pa and
    every piece of Pb: matrices s, t and dist."""
    return _pairwise_closest(Pa[:-1], np.diff(Pa, axis=0),
                             Pb[:-1], np.diff(Pb, axis=0))


def _joint(Pa, Pb, want):
    """If two rods share an end (within `want`), the joint point and how
    far from it they are meant to touch; otherwise None.

    Two rods leaving a joint at angle theta stay within `want` of each
    other for about want / sin(theta), so that is the stretch excused --
    not the whole pair: a curved rod that leaves a joint can come back
    and cross its partner further on, and that crossing still counts.
    """
    ends = np.linalg.norm(Pa[[0, -1]][:, None, :] - Pb[[0, -1]][None, :, :],
                          axis=-1)
    ia, ib = np.unravel_index(int(np.argmin(ends)), ends.shape)
    if ends[ia, ib] >= want:
        return None
    ta = Pa[1] - Pa[0] if ia == 0 else Pa[-2] - Pa[-1]
    tb = Pb[1] - Pb[0] if ib == 0 else Pb[-2] - Pb[-1]
    cos = abs(float(ta @ tb)) / max(float(np.linalg.norm(ta)
                                          * np.linalg.norm(tb)), 1e-300)
    sin = math.sqrt(max(0.0, 1.0 - cos * cos))
    point = 0.5 * ((Pa[0] if ia == 0 else Pa[-1])
                   + (Pb[0] if ib == 0 else Pb[-1]))
    return point, want / max(sin, math.sin(math.radians(3.0))) + want


def _outside_joint(s, t, Pa, Pb, joint):
    """Mask of the piece pairs (of a `_piece_distances` result) that are
    not both within the joint's excused stretch."""
    if joint is None:
        return np.ones(s.shape, dtype=bool)
    point, reach = joint
    qa = Pa[:-1][:, None, :] + s[..., None] * np.diff(Pa, axis=0)[:, None, :]
    qb = Pb[:-1][None, :, :] + t[..., None] * np.diff(Pb, axis=0)[None, :, :]
    return ~((np.linalg.norm(qa - point, axis=-1) < reach)
             & (np.linalg.norm(qb - point, axis=-1) < reach))


def _segment_pairs_closest(A0, A1, B0, B1):
    """Clamped closest points of segment pairs taken element by element
    -- piece A0[i]A1[i] against piece B0[i]B1[i] -- returning s, t, the
    two closest points and their distance (Ericson, as above)."""
    d1, d2, r = A1 - A0, B1 - B0, A0 - B0
    a = np.einsum('ij,ij->i', d1, d1)
    e = np.einsum('ij,ij->i', d2, d2)
    b = np.einsum('ij,ij->i', d1, d2)
    c = np.einsum('ij,ij->i', d1, r)
    f = np.einsum('ij,ij->i', d2, r)
    den = a * e - b * b
    ok = den > 1e-12 * a * e
    s = np.where(ok, np.clip((b * f - c * e) / np.where(ok, den, 1.0),
                             0.0, 1.0), 0.0)
    t = (b * s + f) / np.maximum(e, 1e-300)
    s = np.where(t < 0.0, np.clip(-c / np.maximum(a, 1e-300), 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b - c) / np.maximum(a, 1e-300), 0.0,
                                  1.0), s)
    t = np.clip(t, 0.0, 1.0)
    qa = A0 + s[:, None] * d1
    qb = B0 + t[:, None] * d2
    return s, t, qa, qb, np.linalg.norm(qa - qb, axis=1)


def separate_rods(rods, radius, gap=_ROD_GAP, spacing=_ROD_SPACING,
                  iterations=150, settle=100, smoothing=0.5,
                  straighten=0.02, reach=2.0):
    """Bend rods of `radius` aside wherever two would pass through each
    other, leaving a clear gap of `gap` radii and every rod end where it
    was.  A rod is a polyline: two points for a straight rod, more for a
    curved one.

    Position-based dynamics (Muller et al. 2007) on the rods that
    matter.  Each rod within `reach` clearances of another becomes a
    chain of points `spacing` radii apart along it, its two end points
    fixed, and every iteration

      * pushes apart each nearby pair of pieces (one from each rod) that
        sit closer than the clearance, at their closest points and along
        the line between them -- the common normal, where two pieces
        truly cross -- sharing the correction among the four piece ends
        by where along each piece the closest point falls, so a fixed
        rod end passes all of it to the other rod;
      * smooths each chain's DISPLACEMENT toward the average of its
        neighbours', so a push spreads into a gentle bend rather than a
        kink, while a curved rod keeps its own curve; and
      * draws the displacement a little back toward zero, so no bend is
        longer or deeper than the pushes require.

    A last few passes apply the pushes alone, to settle the gap.

    Pushing pieces rather than whole contacts matters on these surfaces.
    Along a fold, rods of one family graze their neighbours one after
    another and several at once; an earlier solve that gave each contact
    one push of a fixed bump shape stacked those pushes into offsets ten
    rod radii deep.  Pushed piece by piece, such a bundle settles into
    layers about a clearance deep on its own -- and the crossing lattice
    of a hyperboloid settles into its two families, one over the other.

    Rods that share an end (within a clearance) meet at a joint on a
    rail and are meant to touch next to it; only that stretch is left
    out (`_joint`).

    Returns (polylines, info).  polylines[i] is the rod as given if it
    did not move; a moved straight rod keeps just its bent points, a
    moved curved rod all of its chain.  info counts the rod pairs that
    were closer than (2 + gap/2) radii (`contacts`) and the pairs that
    touch at a joint, the pairs still closer than a rod diameter
    afterwards (`remaining`), and gives the largest offset and the
    smallest gap left between two rods away from any joint.
    """
    src = [np.asarray(r, dtype=float).reshape(-1, 3) for r in rods]
    m = len(src)
    info = dict(contacts=0, joints=0, remaining=0, max_offset=0.0,
                min_gap=math.inf)
    polys = [P.copy() for P in src]
    if m < 2 or radius <= 0.0:
        return polys, info
    L = np.array([_polyline_length(P) for P in src])
    want = (2.0 + gap) * radius
    detect = (2.0 + 0.5 * gap) * radius
    far = want * reach

    # candidate rod pairs, searched on coarse copies of the rods
    pieces, owner, stray = [], [], np.zeros(m)
    for k, P in enumerate(src):
        if L[k] < 1e-12:
            continue
        C, stray[k] = _coarse(P)
        pieces.extend(zip(C[:-1], C[1:]))
        owner.extend([k] * (len(C) - 1))
    owner = np.asarray(owner, dtype=int)
    near = segment_contacts(pieces, far + 2.0 * float(stray.max()))
    cand = set()
    for i, j, d in zip(near['i'], near['j'], near['dist']):
        a, b = int(owner[i]), int(owner[j])
        if a != b and d < far + stray[a] + stray[b]:
            cand.add((min(a, b), max(a, b)))

    pairs = []
    for a, b in sorted(cand):
        joint = _joint(src[a], src[b], want)
        s, t, dist = _piece_distances(src[a], src[b])
        ok = _outside_joint(s, t, src[a], src[b], joint)
        if joint is not None and np.any((dist < detect) & ~ok):
            info['joints'] += 1
        if not np.any(ok):
            continue
        dmin = float(dist[ok].min())
        if dmin >= far:
            continue
        pairs.append((a, b, joint))
        info['contacts'] += int(dmin < detect)
    if info['contacts'] == 0:
        return polys, info

    # every rod near another becomes a chain of points, its ends pinned
    rods_in = sorted({k for a, b, _j in pairs for k in (a, b)})
    first, count, chains, pinned = {}, {}, [], []
    total = 0
    for k in rods_in:
        n = max(8, int(math.ceil(L[k] / (spacing * radius))))
        first[k], count[k] = total, n
        total += n + 1
        chains.append(_resample(src[k], n))
        pin = np.zeros(n + 1, dtype=bool)
        pin[0] = pin[-1] = True
        pinned.append(pin)
    X = np.concatenate(chains)
    base = X.copy()
    free = (~np.concatenate(pinned)).astype(float)
    inner = np.concatenate([first[k] + np.arange(1, count[k])
                            for k in rods_in])

    def chain(k, arr):
        return arr[first[k]:first[k] + count[k] + 1]

    # the pieces of each pair that can meet, found once on the rods as
    # given: the bends stay far smaller than the search reach
    from_a, from_b = [], []
    for a, b, joint in pairs:
        Ca, Cb = chain(a, base), chain(b, base)
        s, t, dist = _piece_distances(Ca, Cb)
        ia, ib = np.nonzero((dist < far) & _outside_joint(s, t, Ca, Cb,
                                                          joint))
        from_a.append(first[a] + ia)
        from_b.append(first[b] + ib)
    SA, SB = np.concatenate(from_a), np.concatenate(from_b)

    def push():
        A0, A1, B0, B1 = X[SA], X[SA + 1], X[SB], X[SB + 1]
        s, t, qa, qb, dist = _segment_pairs_closest(A0, A1, B0, B1)
        hit = np.nonzero(dist < want)[0]
        if len(hit) == 0:
            return False
        s, t, dist = s[hit], t[hit], dist[hit]
        nrm = (qa[hit] - qb[hit]) / np.maximum(dist, 1e-300)[:, None]
        crossing = dist < 1e-9 * want
        if np.any(crossing):
            cn = np.cross(A1[hit][crossing] - A0[hit][crossing],
                          B1[hit][crossing] - B0[hit][crossing])
            nrm[crossing] = cn / np.maximum(np.linalg.norm(cn, axis=1),
                                            1e-300)[:, None]
        a0, a1, b0, b1 = SA[hit], SA[hit] + 1, SB[hit], SB[hit] + 1
        wa = (1.0 - s) ** 2 * free[a0] + s ** 2 * free[a1]
        wb = (1.0 - t) ** 2 * free[b0] + t ** 2 * free[b1]
        lam = (want - dist) / np.maximum(wa + wb, 1e-300)
        move = np.zeros_like(X)
        votes = np.zeros(len(X))
        for idx, coef in ((a0, lam * (1.0 - s) * free[a0]),
                          (a1, lam * s * free[a1]),
                          (b0, -lam * (1.0 - t) * free[b0]),
                          (b1, -lam * t * free[b1])):
            np.add.at(move, idx, coef[:, None] * nrm)
            np.add.at(votes, idx, 1.0)
        X[:] += move / np.maximum(votes, 1.0)[:, None]
        return True

    for _ in range(iterations):
        push()
        D = X - base
        D[inner] += smoothing * (0.5 * (D[inner - 1] + D[inner + 1])
                                 - D[inner])
        D[inner] *= 1.0 - straighten
        X[:] = base + D
    for _ in range(settle):
        if not push():
            break

    tol = 0.01 * radius
    for k in rods_in:
        P = chain(k, X)
        off = np.linalg.norm(P - chain(k, base), axis=1)
        if off.max() <= tol:
            continue
        if len(src[k]) == 2:
            # a straight rod: its unmoved stretches stay straight chords
            moved = off > tol
            keep = moved.copy()
            keep[1:] |= moved[:-1]
            keep[:-1] |= moved[1:]
            keep[0] = keep[-1] = True
            polys[k] = P[keep].copy()
        else:
            polys[k] = P.copy()
        info['max_offset'] = max(info['max_offset'], float(off.max()))
    for a, b, joint in pairs:
        s, t, dist = _piece_distances(polys[a], polys[b])
        ok = _outside_joint(s, t, polys[a], polys[b], joint)
        if not np.any(ok):
            continue
        dmin = float(dist[ok].min())
        info['min_gap'] = min(info['min_gap'], dmin)
        info['remaining'] += int(dmin < 2.0 * radius)
    return polys, info


def family_gap(fam):
    """Narrowest distance between neighbouring strands of one family
    (consecutive in list order, the last wrapping to the first).  A ribbon
    wider than this would overlap its neighbour edge to edge.  Strands
    may be segments or polylines."""
    S = [np.asarray(f, dtype=float).reshape(-1, 3) for f in fam]
    if len(S) < 2:
        return math.inf
    best = math.inf
    for i, A in enumerate(S):
        B = S[(i + 1) % len(S)]
        _s, _t, dist = _pairwise_closest(A[:-1], np.diff(A, axis=0),
                                         B[:-1], np.diff(B, axis=0))
        best = min(best, float(dist.min()))
    return best


def polyline_keep(P, tol):
    """Mask of the points of polyline P that must stay for it to keep
    within `tol` of the original -- every other point lies within `tol`
    of the chord between the kept points either side -- with both ends
    kept (U. Ramer 1972; D. Douglas and T. Peucker 1973)."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    keep = np.zeros(len(P), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, d = P[i], P[j] - P[i]
        dd = float(d @ d)
        mid = P[i + 1:j]
        tt = (np.clip(((mid - a) @ d) / dd, 0.0, 1.0) if dd > 0.0
              else np.zeros(len(mid)))
        dist = np.linalg.norm(mid - (a + tt[:, None] * d), axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return keep


def _strand(P):
    """A strand as the weaver keeps it: its points, the arc-length
    fraction reached at each, and its length.  Two points make a straight
    strand, more a curved one."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    L = float(cum[-1])
    return P, (cum / L if L > 0.0 else cum), L


def _strand_at(st, t):
    """Points on a strand, and its unit tangents there, at arc-length
    fractions t.  A curved strand's tangent is taken across half a piece
    either side, so a ribbon turns smoothly through the polyline's
    corners instead of snapping at each one."""
    t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), 0.0, 1.0)
    P, s = st['P'], st['s']

    def at(x):
        return np.stack([np.interp(x, s, P[:, k]) for k in range(3)], axis=1)

    base = at(t)
    if len(P) == 2:
        T = np.repeat(((P[1] - P[0]) / st['L'])[None, :], len(t), axis=0)
    else:
        h = 0.5 / (len(P) - 1)
        T = at(np.clip(t + h, 0.0, 1.0)) - at(np.clip(t - h, 0.0, 1.0))
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-300)
    return base, T


#: fraction of the span between two crossings that is kept clear of
#: both footprints, for the ribbon to change level in
_FREE = 0.15


def plan_weave(fam_a, fam_b, width=0.9, thickness=0.15, run=1,
               crossings=None, gap=None, limit_quantile=None,
               local_width=False):
    """Everything needed to sweep the woven ribbons, before any mesh.

    `width` is a FRACTION of the widest ribbon that weaves cleanly on
    these two families (see below), so any value up to 1 weaves without
    the ribbons touching, whatever the surface or strand count.
    `thickness` is a fraction of the ribbon width.  At each crossing the
    upper ribbon is lifted one thickness along the surface normal and the
    lower one sunk by the same, leaving a clear gap of one thickness
    between them.

    Strands may be straight segments or polylines.  For straight
    families the crossings are found here (`segment_crossings`).  For a
    curved family the caller passes `crossings` in the same form -- ta
    and tb as fractions of arc length -- because where two sampled curves
    meet is better computed from the surface they lie on than recovered
    from their samples; curved strands without them are refused.  `gap`
    overrides the neighbour spacing that caps the width (`family_gap`),
    for a surface whose tightest neighbours sit at a fold rather than in
    the lattice being woven.  `limit_quantile`, if given, sizes the
    ribbons to that quantile of the per-span width limits below instead
    of the tightest one, for a surface where a few crossings crowd too
    close for any visible ribbon to part between them: the spans tighter
    than that are squeezed, and counted in `tight`.

    `local_width` sizes the ribbons crossing by crossing instead: each
    crossing takes the tightest of the limits of the spans either side of
    it, along both its strands, and a ribbon's width eases from one
    crossing's to the next.  On an even lattice that is the uniform width
    again; on an uneven one -- open cells in one place, crowded ones in
    another -- the ribbons fill every cell to the same fraction, widening
    where the lattice opens out and narrowing where it crowds, instead of
    all being sized to the tightest cell anywhere.  Their thickness, and
    so the lift at every crossing, stays one value sized to the typical
    ribbon.

    Around each crossing the lift is held flat over the whole footprint
    of the other ribbon -- measured along this strand, a ribbon of width
    w crossing at angle theta covers (w/2)(1 + |cos theta|)/sin theta
    either side of the centre -- and the ribbon only changes level in
    the space left between footprints.  The width limit is the largest
    that leaves a share _FREE of every span between consecutive
    crossings for that change.  Past it (width > 1) the ribbons are too
    wide to weave cleanly; the offending spans are counted in `tight`.
    """
    strands_a = [_strand(s) for s in fam_a]
    strands_b = [_strand(s) for s in fam_b]
    if crossings is None:
        if any(len(P) != 2 for P, _s, _L in strands_a + strands_b):
            raise ValueError("curved strands need their crossings supplied")
        X = segment_crossings([P for P, _s, _L in strands_a],
                              [P for P, _s, _L in strands_b])
    else:
        X = crossings
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
    lengths = [np.array([L for _P, _s, L in fam])
               for fam in (strands_a, strands_b)]
    limit = (gap if gap is not None
             else min(family_gap(fam_a), family_gap(fam_b)))
    local = np.full(len(X['ia']), np.inf)
    span_limits = []
    for (key_i, key_t), lens in zip((('ia', 'ta'), ('ib', 'tb')), lengths):
        idx, par = X[key_i], X[key_t]
        if len(idx) < 2:
            continue
        order = np.lexsort((par, idx))
        same = idx[order[:-1]] == idx[order[1:]]
        p, q = order[:-1][same], order[1:][same]
        span = (par[q] - par[p]) * lens[idx[p]]
        ok = span > 1e-12
        lim = (1.0 - _FREE) * span[ok] / (per_w[p][ok] + per_w[q][ok])
        span_limits.append(lim)
        np.minimum.at(local, p[ok], lim)
        np.minimum.at(local, q[ok], lim)
    span_limits = (np.concatenate(span_limits) if span_limits
                   else np.zeros(0))
    if len(span_limits):
        limit = min(limit, float(
            np.quantile(span_limits, limit_quantile) if limit_quantile
            else span_limits.min()))
    longest = max((float(np.max(ln)) for ln in lengths if len(ln)),
                  default=1.0)
    if not math.isfinite(limit) or limit <= 0.0:
        limit = 0.05 * longest
    if local_width and np.any(np.isfinite(local)):
        # each crossing's own limit -- floored, so crossings that all but
        # coincide still get a ribbon rather than none
        typical = float(np.median(local[np.isfinite(local)]))
        local = np.where(np.isfinite(local), local, typical)
        w_c = width * np.maximum(local, 0.02 * typical)
    else:
        w_c = np.full(len(X['ia']), width * limit)
    if local_width and len(w_c):
        # one thickness across the weave, sized to the typical ribbon, so
        # the wide ribbons of open cells stay ribbons rather than slabs
        th_c = np.full(len(w_c), thickness * float(np.median(w_c)))
    else:
        th_c = thickness * w_c
    foot = w_c * per_w
    w = float(np.median(w_c)) if len(w_c) else width * limit
    th = thickness * w
    amp = th

    fallback = (np.mean(X['normal'], axis=0) if len(X['ia'])
                else np.array([0.0, 0.0, 1.0]))
    strands = []
    tight = 0
    for fam, key_i, key_t, sign in ((strands_a, 'ia', 'ta', 1.0),
                                    (strands_b, 'ib', 'tb', -1.0)):
        for i, (P, s, L) in enumerate(fam):
            if L < 1e-12:
                continue
            chord = P[-1] - P[0]
            if float(np.linalg.norm(chord)) < 1e-9 * L:
                # a closed strand ends where it starts: take its general
                # direction from the point halfway round instead
                chord = P[len(P) // 2] - P[0]
            T = chord / max(float(np.linalg.norm(chord)), 1e-300)
            sel = np.nonzero(X[key_i] == i)[0]
            sel = sel[np.argsort(X[key_t][sel])]
            tk = X[key_t][sel]
            sg = np.where(a_over[sel], sign, -sign)
            # A crossing's normal and which ribbon is over are one choice
            # up to sign: (N, over) and (-N, under) put the ribbons in the
            # same place.  Strand by strand, take the sign that keeps
            # consecutive normals agreeing, so where the surface folds
            # over, a ribbon turns with it rather than being flipped half
            # a turn between two crossings.
            Ns = X['normal'][sel].copy()
            for k in range(1, len(sel)):
                if float(Ns[k] @ Ns[k - 1]) < 0.0:
                    Ns[k] = -Ns[k]
                    sg[k] = -sg[k]
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
                # the crossings' normals cancel out, as they do round a
                # closed surface: any direction square to the strand will do
                Nf = np.cross(T, np.eye(3)[int(np.argmin(np.abs(T)))])
            st = dict(P=P, s=s, L=L, t=tk, sign=sg, half=hk, lo=lo, hi=hi,
                      N=Ns, fallback=Nf / np.linalg.norm(Nf),
                      crossings=sel, w=w_c[sel], th=th_c[sel])
            st['Tk'] = (_strand_at(st, tk)[1] if len(tk)
                        else np.zeros((0, 3)))
            strands.append(st)
    return dict(strands=strands, width=w, thickness=th, amp=amp,
                widths=w_c, thicknesses=th_c,
                crossings=X, level=level, a_over=a_over,
                conflicts=conflicts, tight=tight, run=run, limit=limit)


def _strand_widths(plan, st, t):
    """Ribbon width and thickness along a strand at arc-length fractions
    t: each crossing's own, eased linearly from one crossing to the next
    and held beyond the first and last."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if len(st['t']) == 0:
        return (np.full(len(t), plan['width']),
                np.full(len(t), plan['thickness']))
    return np.interp(t, st['t'], st['w']), np.interp(t, st['t'], st['th'])


def strand_section(plan, st, t):
    """Ribbon centre, side and normal vectors along one strand at
    arc-length fractions `t`: the strand, lifted along the surface normal
    by the woven over/under profile."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    tk, sg = st['t'], st['sign']
    n = len(tk)
    base, T = _strand_at(st, t)
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
    # square the normal up to the local tangent: a straight strand's
    # normals already are, a curved strand's turn between crossings
    N = N - np.einsum('ij,ij->i', N, T)[:, None] * T
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
    # the lift at a crossing is one ribbon thickness
    _w, lift = _strand_widths(plan, st, t)
    C = base + (lift * d)[:, None] * N
    S = np.cross(N, T)
    return C, S, N


def _sample_ts(plan, st, steps):
    """Sample parameters: every knot, footprint edge and transition end,
    with the level-changing spans subdivided `steps` times -- and, on a
    curved strand, the points of its polyline it needs to follow the
    curve to a tenth of the ribbon's narrowest width."""
    br = [0.0, 1.0]
    if len(st['P']) > 2:
        narrowest = float(st['w'].min()) if len(st['w']) else plan['width']
        br.extend(st['s'][polyline_keep(st['P'], 0.1 * narrowest)])
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
    base, _T = _strand_at(st, mids)
    _w, amp = _strand_widths(plan, st, mids)
    lift = (np.abs(np.einsum('ij,ij->i', C - base, N))
            / np.maximum(amp, 1e-300))
    out = [br[0]]
    for k in range(len(br) - 1):
        m = steps if lift[k] < 1.0 - 1e-9 else 1
        out.extend(br[k] + (br[k + 1] - br[k]) * np.arange(1, m + 1) / m)
    return np.asarray(out)


def sweep_woven_ribbons(plan, steps=6):
    """Mesh the plan: each strand a closed box-section ribbon.

    Returns (verts, faces, face_strand).  Faces wind outward.
    """
    corners = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    verts, faces, face_strand = [], [], []
    for si, st in enumerate(plan['strands']):
        ts = _sample_ts(plan, st, steps)
        C, S, N = strand_section(plan, st, ts)
        w_t, th_t = _strand_widths(plan, st, ts)
        hw, ht = 0.5 * w_t[:, None], 0.5 * th_t[:, None]
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


def weave_rulings(fam_a, fam_b, width=0.9, thickness=0.15, run=1,
                  steps=6, crossings=None, gap=None, limit_quantile=None,
                  local_width=False):
    """Woven ribbons for two ruling families: (verts, faces, plan).
    `crossings`, `gap`, `limit_quantile` and `local_width` are passed to
    `plan_weave`."""
    plan = plan_weave(fam_a, fam_b, width, thickness, run, crossings, gap,
                      limit_quantile, local_width)
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
    across_frac = np.linspace(-1.0, 1.0, across)
    worst = math.inf
    for c, pair in by_cross.items():
        if len(pair) != 2:
            continue
        # over is the side of the crossing's own normal each strand is
        # lifted to; a strand may carry that normal the other way round
        Nc = X['normal'][c]
        lifts = [(pr, float(pr[0]['sign'][pr[1]]
                            * (pr[0]['N'][pr[1]] @ Nc))) for pr in pair]
        ups = [pr for pr, up in lifts if up > 0.0]
        downs = [pr for pr, up in lifts if up < 0.0]
        if len(ups) != 1 or len(downs) != 1:
            continue                       # both over: a solver conflict
        (so, ko), (su, ku) = ups[0], downs[0]
        P, Nc = X['point'][c], X['normal'][c]
        tc, h = so['t'][ko], so['half'][ko]
        ts = np.clip(np.linspace(tc - h, tc + h, samples), 0.0, 1.0)
        C, S, N = strand_section(plan, so, ts)
        wo, tho = _strand_widths(plan, so, ts)
        Q = (C[:, None, :]
             + (across_frac[None, :] * 0.5 * wo[:, None])[..., None]
             * S[:, None, :]
             - (0.5 * tho)[:, None, None] * N[:, None, :]).reshape(-1, 3)
        # position of each underside sample in the lower ribbon's frame,
        # along its tangent at the crossing and across it
        Tu = su['Tk'][ku]
        side_u = np.cross(Nc, Tu)
        d = Q - P
        tu = su['t'][ku] + (d @ Tu) / su['L']
        bu = d @ side_u
        wu, _thu = _strand_widths(plan, su, tu)
        keep = (tu >= 0.0) & (tu <= 1.0) & (np.abs(bu) <= 0.5 * wu)
        if not np.any(keep):
            continue
        Cu, Su, Nu = strand_section(plan, su, tu[keep])
        _wk, thk = _strand_widths(plan, su, tu[keep])
        top = Cu + bu[keep, None] * Su + (0.5 * thk)[:, None] * Nu
        gap = (float(np.min((Q[keep] - top) @ Nc))
               / float(plan['thicknesses'][c]))
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

    # overhang: one absolute distance for every strand.  Parallel
    # families gain no crossings by growing; trimming a quarter of the
    # longest (1.5 of 6) off each end leaves the rows and columns 1..4,
    # whose 16 crossings include the exact endpoint ones.
    er, ec = extend_families(rows, cols, 0.25)
    assert all(abs(math.dist(p, q) - 9.0) < 1e-12 for p, q in er + ec)
    assert len(segment_crossings(er, ec)['ia']) == n * n
    tr, tc = extend_families(rows, cols, -0.25)
    assert len(tr) == len(tc) == n
    assert len(segment_crossings(tr, tc)['ia']) == 16
    assert extend_families(rows, cols, -0.5) == ([], [])
    assert plan_weave([], [])['strands'] == []
    print("rulings: overhang lengthens and trims every strand by one "
          "distance; trimmed-away strands drop OK")

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

    # the same saddle families given as polylines -- each straight rod cut
    # into five pieces -- with their crossings supplied, must plan the
    # same weave as the straight strands do
    fa_poly = [np.linspace(np.asarray(p0), np.asarray(p1), 6) for p0, p1 in fa]
    fb_poly = [np.linspace(np.asarray(p0), np.asarray(p1), 6) for p0, p1 in fb]
    pp = plan_weave(fa_poly, fb_poly, width=0.7,
                    crossings=segment_crossings(fa, fb))
    assert abs(pp['width'] - sp['width']) < 1e-12, (pp['width'], sp['width'])
    assert pp['conflicts'] == 0 and pp['tight'] == 0
    assert abs(crossing_clearance(pp) - crossing_clearance(sp)) < 1e-6

    # Truly curved strands: meridians and parallels of the unit sphere,
    # crossing at right angles with the outward radius as the normal.  A
    # curved family carries its crossings in, as arc-length fractions,
    # and must weave as cleanly as a straight one -- closed outward
    # ribbons of the right volume, lying on the sphere.
    th0, th1, ph0, ph1 = math.pi / 4.0, 3.0 * math.pi / 4.0, 0.0, 1.3
    phis = np.linspace(0.1, 1.2, 8)
    thetas = np.linspace(th0 + 0.1, th1 - 0.1, 7)
    tt = np.linspace(0.0, 1.0, 65)
    th_, ph_ = th0 + (th1 - th0) * tt, ph0 + (ph1 - ph0) * tt
    meridians = [np.stack([np.sin(th_) * math.cos(ph), np.sin(th_)
                           * math.sin(ph), np.cos(th_)], axis=1)
                 for ph in phis]
    parallels = [np.stack([math.sin(th) * np.cos(ph_), math.sin(th)
                           * np.sin(ph_), np.full_like(ph_, math.cos(th))],
                          axis=1) for th in thetas]
    grid_x = [(i, k, (th - th0) / (th1 - th0), (ph - ph0) / (ph1 - ph0),
               (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph),
                math.cos(th)))
              for i, ph in enumerate(phis) for k, th in enumerate(thetas)]
    SX = dict(ia=np.array([r[0] for r in grid_x]),
              ib=np.array([r[1] for r in grid_x]),
              ta=np.array([r[2] for r in grid_x]),
              tb=np.array([r[3] for r in grid_x]),
              point=np.array([r[4] for r in grid_x]),
              normal=np.array([r[4] for r in grid_x]),
              sin=np.ones(len(grid_x)))
    sph = plan_weave(meridians, parallels, crossings=SX)
    assert sph['conflicts'] == 0 and sph['tight'] == 0, (sph['conflicts'],
                                                        sph['tight'])
    sph_clear = crossing_clearance(sph)
    assert sph_clear > 0.5, sph_clear
    verts, faces, fs = sweep_woven_ribbons(sph)
    fs = np.asarray(fs)
    for si, st in enumerate(sph['strands']):
        F = [faces[k] for k in np.nonzero(fs == si)[0]]
        vol = _signed_volume(verts, F)
        want = sph['width'] * sph['thickness'] * st['L']
        assert vol > 0 and abs(vol - want) < 0.05 * want, (si, vol, want)
    radii = np.linalg.norm(np.asarray(verts), axis=1)
    assert np.max(np.abs(radii - 1.0)) < (sph['amp'] + sph['thickness']
                                          + sph['width'] ** 2)
    try:
        plan_weave(meridians, parallels)
    except ValueError:
        pass
    else:
        raise AssertionError("curved strands were woven without crossings")
    print("rulings: polyline strands plan as their straight originals; a "
          "sphere's meridians and parallels weave cleanly (clearance %.2f), "
          "closed and on the sphere; curved strands without crossings are "
          "refused OK" % sph_clear)

    # local width: rows spaced unevenly (gaps of 1, 0.5, 2 and 0.5)
    # across evenly spaced columns.  Sized crossing by crossing, the
    # ribbons must be wider where the cells are wide than where they are
    # narrow -- on an even grid they would all be equal -- while the weave
    # stays conflict-free, untight and clear, and every ribbon a closed,
    # outward box.
    ys = [0.0, 1.0, 1.5, 3.5, 4.0]
    urows = [((-0.5, y, 0.0), (5.5, y, 0.0)) for y in ys]
    ucols = [((float(i), -0.5, 0.0), (float(i), 4.5, 0.0)) for i in range(6)]
    lp = plan_weave(urows, ucols, width=0.9, local_width=True)
    assert lp['conflicts'] == 0 and lp['tight'] == 0, (lp['conflicts'],
                                                      lp['tight'])
    assert crossing_clearance(lp) > 0.5, crossing_clearance(lp)
    # square crossings, so each limit is (1 - _FREE) times the tighter of
    # the spans beside it: the bottom row sits between gaps of 1 (row and
    # column spacing alike), every other row beside a gap of 0.5
    row_y = lp['crossings']['point'][:, 1]
    want_w = np.where(np.isclose(row_y, 0.0), 1.0, 0.5) * 0.9 * (1.0 - _FREE)
    assert np.allclose(lp['widths'], want_w), (lp['widths'], want_w)
    even = plan_weave(rows, cols, width=0.9, local_width=True)
    assert np.allclose(even['widths'], plan_weave(rows, cols,
                                                  width=0.9)['width'])
    verts, faces, fs = sweep_woven_ribbons(lp)
    fs = np.asarray(fs)
    for si in range(len(lp['strands'])):
        F = [faces[k] for k in np.nonzero(fs == si)[0]]
        assert _signed_volume(verts, F) > 0.0, si
    print("rulings: local width widens ribbons where cells open out "
          "(%.2f to %.2f), and on an even grid equals the uniform width; "
          "conflict-free and clear OK"
          % (lp['widths'].min(), lp['widths'].max()))

    # ---- separating rods ----------------------------------------------
    # Judge the result on the bent geometry itself, by brute force over
    # every piece of every rod -- not through the solver's own contact
    # search, which is what is being tested.
    r_ = 0.05

    def worst_gap(polys, segs_):
        worst = math.inf
        S_ = np.asarray(segs_, dtype=float)
        for a in range(len(polys)):
            for b in range(a + 1, len(polys)):
                ends = np.linalg.norm(S_[a][:, None] - S_[b][None, :],
                                      axis=-1)
                if ends.min() < (2.0 + _ROD_GAP) * r_:
                    continue                   # a joint: meant to touch
                Pa, Pb = polys[a], polys[b]
                _s, _t, dd = _pairwise_closest(Pa[:-1], np.diff(Pa, axis=0),
                                               Pb[:-1], np.diff(Pb, axis=0))
                worst = min(worst, float(dd.min()))
        return worst

    def check(label, segs_, contacts):
        polys, inf = separate_rods(segs_, r_)
        assert inf['contacts'] == contacts, (label, inf)
        assert inf['remaining'] == 0, (label, inf)
        for P, s_ in zip(polys, segs_):         # ends never move
            assert np.allclose(P[0], s_[0]) and np.allclose(P[-1], s_[1])
        g = worst_gap(polys, segs_)
        # clear of each other, and no closer than the solver claims
        assert g >= 2.0 * r_ and g >= inf['min_gap'] - 1e-9, \
            (label, g / r_, inf['min_gap'] / r_)
        return polys, inf, g

    # an exact X: both rods give way equally, along the common normal
    ang = math.radians(60.0)
    x_rods = [((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
              ((-math.cos(ang), -math.sin(ang), 0.0),
               (math.cos(ang), math.sin(ang), 0.0))]
    polys, inf, g = check("X", x_rods, 1)
    lift = [float(np.max(np.abs(P[:, 2]))) for P in polys]
    # equal up to sampling: the two rods are cut into pieces at
    # different angles to each other, so the lifts differ in the 4th
    # significant figure -- a lopsided solve would differ by a factor
    assert abs(lift[0] - lift[1]) < 0.01 * max(lift), lift
    assert min(lift) > 0.4 * (2.0 + _ROD_GAP) * r_, lift
    print(f"rulings: an X of rods parts symmetrically along the normal, "
          f"gap {g / r_:.3f} radii OK")
    # nearly parallel, skew by less than a radius: a long overlap
    a5 = math.radians(5.0)
    skew = [((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            ((-2.0 * math.cos(a5), -2.0 * math.sin(a5), 0.03),
             (2.0 * math.cos(a5), 2.0 * math.sin(a5), 0.03))]
    _p, _i, g = check("skew", skew, 1)
    # a V sharing an end is a joint: left alone
    vee = [((0.0, 0.0, 0.0), (1.0, 0.2, 0.0)),
           ((0.0, 0.0, 0.0), (1.0, -0.2, 0.0))]
    polys, inf = separate_rods(vee, r_)
    assert inf['contacts'] == 0 and inf['joints'] == 1
    assert all(len(P) == 2 for P in polys)
    # a T: the stem's end sits on the bar's middle, so the bar gives way
    tee = [((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
           ((0.0, 0.0, 0.07), (0.0, 0.0, 1.0))]
    polys, inf, g = check("T", tee, 1)
    assert float(np.min(polys[0][:, 2])) < -0.02, "the bar did not move"
    # four rods through one point: the pushes must stack them in layers
    star = [((-math.cos(k * math.pi / 4), -math.sin(k * math.pi / 4), 0.0),
             (math.cos(k * math.pi / 4), math.sin(k * math.pi / 4), 0.0))
            for k in range(4)]
    _p, inf, g = check("star", star, 6)
    # and rods that never come near are left exactly as they were
    far = [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
           ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0))]
    polys, inf = separate_rods(far, r_)
    assert inf['contacts'] == 0 and all(len(P) == 2 for P in polys)
    print("rulings: skew, T and a four-rod star separate with every end "
          "fixed; a V joint and distant rods stay straight OK")

    # curved rods are polylines.  An arc crossing a straight bar twice:
    # both crossings clear, the ends stay put, and away from the
    # crossings the arc is still the arc
    th_ = np.linspace(0.0, math.pi, 65)
    arc = np.stack([np.cos(th_), np.sin(th_), np.zeros_like(th_)], axis=1)
    bar = np.array([(-1.0, 0.8, 0.0), (1.0, 0.8, 0.0)])
    polys, inf = separate_rods([arc, bar], r_)
    assert inf['contacts'] == 1 and inf['remaining'] == 0, inf
    assert np.allclose(polys[0][0], arc[0]) and np.allclose(polys[0][-1], arc[-1])
    assert worst_gap(polys, [(arc[0], arc[-1]), tuple(bar)]) >= 2.0 * r_
    tips = polys[0][np.abs(polys[0][:, 0]) > 0.99]
    assert np.allclose(np.hypot(tips[:, 0], tips[:, 1]), 1.0, atol=0.05 * r_)
    # a joint excuses only its own stretch: a curve that leaves the end
    # it shares with a straight rod and crosses that rod again further on
    # is still pushed clear there
    xs = np.linspace(0.0, 1.8, 91)
    wave = np.stack([xs, 0.3 * np.sin(math.pi * xs / 1.5), np.zeros_like(xs)], axis=1)
    rod = np.array([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    polys, inf = separate_rods([wave, rod], r_)
    assert inf['joints'] == 1 and inf['contacts'] == 1 and inf['remaining'] == 0, inf
    Pw, Pr = polys
    _s, _t, dd = _pairwise_closest(Pw[:-1], np.diff(Pw, axis=0), Pr[:-1], np.diff(Pr, axis=0))
    qa = Pw[:-1][:, None, :] + _s[..., None] * np.diff(Pw, axis=0)[:, None, :]
    away = np.linalg.norm(qa, axis=-1) > 0.5
    assert float(dd[away].min()) >= 2.0 * r_, float(dd[away].min()) / r_
    print("rulings: curved rods separate too -- an arc keeps its curve, and "
          "a crossing beyond a shared joint is still cleared OK")
    print("RESULT: OK")
