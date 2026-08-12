
# Polyline repair for Blender's curve bevel.
#
# Both problems here come from the same place: Blender sweeps a bevel
# profile along a POLY spline using a frame derived from the tangent,
# and a polyline produced by a turtle has two features that frame does
# not cope with -- exact 180-degree reversals (the tangent flips, so the
# frame is undefined) and sharp corners (the profile is placed in the
# BISECTING plane rather than mitred, so the tube pinches).
#
# Kept free of bpy so `tests/test_selftests.py` can check the geometry
# headlessly; `emit` re-exports both functions.

import numpy as np


def split_at_reversals(points, tol=-0.999):
    """Split a polyline wherever it doubles back on itself exactly.

    WHY THIS IS NEEDED.  Blender bevels a poly spline by carrying a frame
    along the tangent.  At a 180-degree reversal the tangent flips sign,
    so the frame is undefined and the swept profile pinches and spins --
    it shows up as little rosettes and blobs strung along the curve.

    Several classical curves genuinely retrace themselves: the Levy C
    curve has 68 exact reversals at iteration 10, and the dragon and
    Peano families have them too.  The curve is not wrong; the bevel
    simply has nothing to orient itself with.

    Splitting into separate splines at each reversal gives every piece a
    well-defined tangent.  The pieces still overlap in space -- that is
    the geometry -- but each is correctly formed instead of degenerate.

    Returns a list of index ranges (start, stop) into `points`.
    """
    n = len(points)
    if n < 3:
        return [(0, n)]
    v = np.diff(points, axis=0)
    ln = np.linalg.norm(v, axis=1)
    ln[ln < 1e-12] = 1.0
    v = v / ln[:, None]
    dots = (v[:-1] * v[1:]).sum(axis=1)
    cuts = [i + 1 for i, d in enumerate(dots) if d <= tol]
    if not cuts:
        return [(0, n)]
    spans, start = [], 0
    for c in cuts:
        if c - start >= 2:
            spans.append((start, c + 1))
        start = c
    if n - start >= 2:
        spans.append((start, n))
    return spans or [(0, n)]


def round_corners(points, widths, fraction=0.25, segments=4,
                  min_angle=10.0, closed=False):
    """Replace each sharp vertex with a short fillet.

    WHY THIS IS NEEDED.  Blender bevels a poly spline by placing the
    profile circle in the plane that BISECTS the two adjacent segments --
    it does not mitre the joint.  Projected along the direction of
    travel, that circle becomes an ellipse whose semi-axis is only

        r * cos(theta / 2)

    for a turn of theta.  So the tube is full width along every straight
    run and choked at every corner, which reads as a string of lozenges:
    fat in the middle, pinched at the ends.  Measured on the shipped
    presets at bevel depth 0.01 -- Koch turns 120 degrees and pinches to
    0.00500 (exactly cos 60), the 90-degree turns pinch to 0.00707.

    Sampling the fillet at equal angles turns each joint by exactly
    theta / segments (and half that where the arc meets the straight
    run), so the worst pinch is bounded by

        cos(theta / (2 * segments))

    -- for Koch's 120 degrees at the default four segments that is
    cos 15 = 0.966 rather than 0.5.

    The fillet cuts back `fraction` of the SHORTER adjacent segment from
    each side, so with fraction <= 0.5 two neighbouring corners can never
    eat into each other.  The arc is a true CIRCULAR fillet, tangent to
    both segments at the cut-back points: sampled at equal angles it
    divides the turn into exactly equal joints, which is what puts a
    floor under the pinch.  (A quadratic Bezier through the vertex is
    tangent in the same way but bunches its curvature -- sampled
    uniformly in its parameter it leaves a 33.7-degree joint on a
    90-degree corner instead of 22.5, and the floor is lost.)

    Returns (points, widths) as new arrays.  `fraction` <= 0 is a no-op,
    so the caller can keep mathematically exact corners on demand.
    """
    pts = np.asarray(points, dtype=float)
    wid = np.asarray(widths, dtype=float)

    # Accept plane (N,2) input as well as (N,3).  The fillet is built
    # from cross products, and `np.cross` of two 2-vectors returns a
    # SCALAR rather than a vector, so a 2-D array would otherwise fail
    # deep inside with an unhelpful axis error.  Lift, compute, and give
    # the caller back the shape it passed in.
    flat = pts.ndim == 2 and pts.shape[1] == 2
    if flat:
        pts = np.hstack([pts, np.zeros((len(pts), 1))])

    # A closed turtle path ends where it began, so the point list carries
    # the seam vertex TWICE.  Read cyclically that is a zero-length
    # segment, and it defeats the fillet exactly at the seam -- the one
    # corner a closed figure most obviously shows.  Blender re-closes the
    # spline itself from `use_cyclic_u`, so the duplicate is redundant
    # as well as harmful.
    if closed and len(pts) > 2 and np.allclose(pts[0], pts[-1], atol=1e-9):
        pts, wid = pts[:-1], wid[:-1]

    n = len(pts)
    if fraction <= 0.0 or segments < 1 or n < 3:
        return (pts[:, :2] if flat else pts), wid

    frac = min(float(fraction), 0.5)
    cos_max = np.cos(np.radians(float(min_angle)))
    k = int(segments) + 1

    # Vectorised over CORNERS, not looped.  A dense curve has as many
    # corners as segments -- the Cesaro curve at its old default had
    # 65,536 -- and a per-corner Python loop costs seconds there, which
    # is felt directly as operator latency.  Everything below is one
    # numpy expression per quantity across all corners at once.
    i = np.arange(n) if closed else np.arange(1, n - 1)
    P0, P1, P2 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]

    u, v = P1 - P0, P2 - P1
    lu = np.linalg.norm(u, axis=1)
    lv = np.linalg.norm(v, axis=1)
    ok = (lu > 1e-12) & (lv > 1e-12)
    safe_u = np.where(ok, lu, 1.0)[:, None]
    safe_v = np.where(ok, lv, 1.0)[:, None]
    u, v = u / safe_u, v / safe_v

    dot = np.clip((u * v).sum(1), -1.0, 1.0)
    axis = np.cross(u, v)
    sin_t = np.linalg.norm(axis, axis=1)
    theta = np.arctan2(sin_t, dot)
    # skip: degenerate segment, already straight, or an exact reversal
    # (the latter is `split_at_reversals`'s job, not the fillet's)
    fil = ok & (dot < cos_max) & (sin_t > 1e-9) & (theta < np.pi - 1e-6)
    if not fil.any():
        P, W = pts.copy(), wid.copy()
    else:
        axis = axis / np.where(fil, sin_t, 1.0)[:, None]
        t = frac * np.minimum(lu, lv)
        A = P1 - u * t[:, None]
        # radius of the circle tangent to both segments at the cut-backs
        rho = t / np.where(fil, np.tan(theta / 2.0), 1.0)
        centre = A + rho[:, None] * np.cross(axis, u)
        arm = A - centre
        perp = np.cross(axis, arm)

        # equal angular steps -> equal turn at every joint, which is what
        # bounds the pinch at cos(theta / (2 * segments))
        phi = theta[:, None] * (np.arange(k) / float(segments))
        arc = (centre[:, None, :]
               + arm[:, None, :] * np.cos(phi)[:, :, None]
               + perp[:, None, :] * np.sin(phi)[:, :, None])
        arc[~fil, 0] = P1[~fil]        # a skipped corner keeps its vertex

        # keep all k samples of a filleted corner, only the vertex of a
        # skipped one -- row-major order preserves the traversal
        keep = fil[:, None] | (np.arange(k)[None, :] == 0)
        P = arc.reshape(-1, 3)[keep.ravel()]
        W = np.repeat(wid[i], k).reshape(-1, k)[keep].ravel()
        if not closed:
            P = np.vstack([pts[:1], P, pts[-1:]])
            W = np.concatenate([wid[:1], W, wid[-1:]])

    # At the maximum fraction two neighbouring fillets meet exactly, so
    # one ends where the next begins and the shared point is emitted
    # twice.  A repeated point is a zero-length segment, which is the
    # very thing the bevel cannot build a frame from -- drop it.
    if len(P) > 1:
        keep = np.ones(len(P), dtype=bool)
        keep[1:] = (np.abs(np.diff(P, axis=0)) > 1e-12).any(axis=1)
        if closed and len(P) > 2 and np.allclose(P[0], P[-1], atol=1e-12):
            keep[-1] = False        # the wrap joins them; don't repeat it
        P, W = P[keep], W[keep]
    return (P[:, :2] if flat else P), W


def max_turn(points, closed=False):
    """Largest turn angle in degrees -- the quantity that governs how
    badly Blender's bevel pinches (the tube narrows to cos(turn/2))."""
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return 0.0
    d = np.diff(np.vstack([p, p[:1]]) if closed else p, axis=0)
    L = np.linalg.norm(d, axis=1)
    keep = L > 1e-12
    u = d[keep] / L[keep][:, None]
    if len(u) < 2:
        return 0.0
    pairs = np.vstack([u, u[:1]]) if closed else u
    dots = np.clip((pairs[:-1] * pairs[1:]).sum(1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).max())


def _selftest():
    # --- split_at_reversals ------------------------------------------
    # a straight line has nothing to split
    line = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    assert split_at_reversals(line) == [(0, 4)]
    # an exact retrace does
    back = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]])
    spans = split_at_reversals(back)
    assert len(spans) == 2, spans
    for a, b in spans:                       # every piece stays drawable
        assert b - a >= 2, spans

    # --- round_corners -----------------------------------------------
    corner = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0]])
    w = np.ones(3)

    # fraction 0 is an exact no-op, so exact corners stay available
    p0, w0 = round_corners(corner, w, fraction=0.0)
    assert np.allclose(p0, corner) and len(w0) == 3

    # THE POINT OF THE EXERCISE: a 90-degree turn pinches the bevel to
    # cos 45 = 0.707 of the nominal radius.  Spreading it over four
    # joints leaves cos 11.25 = 0.981.
    assert abs(max_turn(corner) - 90.0) < 1e-9
    p1, w1 = round_corners(corner, w, fraction=0.25, segments=3)
    assert max_turn(p1) < 90.0 / 3 + 1e-6, max_turn(p1)
    assert np.cos(np.radians(max_turn(p1) / 2)) > 0.96
    assert len(p1) == len(w1)

    # the fillet must not move the endpoints or leave the original span
    assert np.allclose(p1[0], corner[0]) and np.allclose(p1[-1], corner[-1])
    assert p1[:, 0].min() >= -1e-12 and p1[:, 0].max() <= 1.0 + 1e-12
    assert p1[:, 1].min() >= -1e-12 and p1[:, 1].max() <= 1.0 + 1e-12

    # more segments -> monotonically gentler joints
    prev = max_turn(corner)
    for k in (2, 4, 8, 16):
        t = max_turn(round_corners(corner, w, 0.25, k)[0])
        assert t < prev, (k, t, prev)
        prev = t

    # Adjacent corners must not eat into each other.  A tight zigzag of
    # equal-length segments at the maximum fraction is the worst case:
    # each corner claims half of each neighbouring segment, so the
    # cut-backs meet exactly and never cross.
    zig = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0], [2, 2, 0]])
    pz, wz = round_corners(zig, np.ones(len(zig)), fraction=0.5, segments=4)
    d = np.linalg.norm(np.diff(pz, axis=0), axis=1)
    assert (d > 1e-9).all(), "fillets overlapped and produced a null segment"
    assert max_turn(pz) < max_turn(zig)

    # Koch's 120-degree turn is the worst case in the shipped presets:
    # it is what pinched the tube to half width in the first place.
    koch = np.array([[0., 0, 0], [1, 0, 0],
                     [1 + np.cos(np.radians(120)), np.sin(np.radians(120)), 0]])
    assert abs(max_turn(koch) - 120.0) < 1e-9
    pk, _ = round_corners(koch, np.ones(3), 0.25, 4)
    assert np.cos(np.radians(max_turn(pk) / 2)) > 0.96, max_turn(pk)

    # It really is a circular arc, not just a smooth-looking blend: on a
    # circle sampled at equal angles every chord has the same length and
    # every interior joint the same turn.  That equality is the whole
    # reason the pinch has a floor.
    chords = np.linalg.norm(np.diff(pk[1:-1], axis=0), axis=1)
    assert np.ptp(chords) < 1e-12, chords
    d = np.diff(pk[1:-1], axis=0)
    d = d / np.linalg.norm(d, axis=1)[:, None]
    joints = np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(1), -1, 1)))
    assert np.ptp(joints) < 1e-9, joints
    assert abs(joints[0] - 120.0 / 4) < 1e-9, joints[0]

    # a nearly-straight vertex is left alone rather than resampled
    soft = np.array([[0., 0, 0], [1, 0, 0], [2, 0.01, 0]])
    ps, _ = round_corners(soft, np.ones(3), 0.25, 3, min_angle=10.0)
    assert len(ps) == 3, "a sub-threshold bend should not be filleted"

    # A CLOSED turtle path ends where it began, so its point list repeats
    # the seam vertex.  Read cyclically that is a zero-length segment,
    # and it used to leave the seam corner unfilleted at its full turn --
    # 120 degrees on the Koch snowflake, which is where the pinch was
    # most visible.
    h = np.sqrt(3) / 2                # a TRUE equilateral triangle,
    tri = np.array([[0., 0, 0], [1, 0, 0],   # so the turns are exactly
                    [.5, h, 0], [0, 0, 0]])  # 120 degrees
    pt, wt = round_corners(tri, np.ones(4), 0.25, 4, closed=True)
    assert not np.allclose(pt[0], pt[-1]), "seam point still duplicated"
    assert max_turn(pt, closed=True) < 120.0 / 4 + 1e-6, max_turn(pt, True)
    dt = np.linalg.norm(np.diff(np.vstack([pt, pt[:1]]), axis=0), axis=1)
    assert (dt > 1e-9).all(), "zero-length segment survived at the seam"

    # a nearly-straight vertex is left alone rather than resampled
    sq = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    pc, _ = round_corners(sq, np.ones(4), 0.25, 3, closed=True)
    assert len(pc) == 4 * 4, len(pc)
    assert max_turn(pc, closed=True) < 90.0 / 3 + 1e-6

    # degenerate input is tolerated
    dup = np.array([[0., 0, 0], [0, 0, 0], [1, 0, 0]])
    round_corners(dup, np.ones(3), 0.25, 3)

    # Plane (N,2) input is accepted and returns (N,2): `np.cross` of two
    # 2-vectors gives a scalar, so without the internal lift this failed
    # with an axis error rather than working.
    flat2 = np.array([[0., 0], [1, 0], [1, 1]])
    pf, wf = round_corners(flat2, np.ones(3), 0.25, 4)
    assert pf.shape[1] == 2, pf.shape
    lifted = np.hstack([flat2, np.zeros((3, 1))])
    pl, _ = round_corners(lifted, np.ones(3), 0.25, 4)
    assert np.allclose(pf, pl[:, :2]), "2-D and 3-D paths must agree"
    # and the no-op path preserves the shape too
    assert round_corners(flat2, np.ones(3), 0.0)[0].shape[1] == 2

    print("polyline: OK -- reversals split, corners filleted from 90 to "
          "22.5 degrees, endpoints fixed, adjacent fillets disjoint")
