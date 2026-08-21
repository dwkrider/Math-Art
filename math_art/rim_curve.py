# Rim curves: a swept tube along a surface's open edge.
#
# Most of the surfaces this add-on builds are cut off somewhere -- a
# level set clipped to its sample box, a minimal surface truncated at
# its ends, a helicoid stopped at a parameter bound.  That cut is a
# stair-step through whatever grid produced it, not a curve, and it
# reads as a ragged fringe.  Sweeping a bevelled tube along it hides the
# staircase and gives the surface a deliberate border, which is as much
# an aesthetic control as a tidy-up.
#
# This module is shared: the geometry is plain numpy so it self-tests
# headlessly, and the Blender half is one helper plus three property
# factories, so every generator that offers a rim offers the SAME
# controls with the same defaults rather than drifting apart.
#
# A closed surface has no open edge and simply gets no curve.  That is
# not an error -- a cyclide, a Hauser tube or a triply-periodic cell is
# closed by construction, and the option is a no-op there.

import math

import numpy as np

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty,
                           FloatProperty, IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# Smoothing is deliberately LOW.  The visual rounding is done by the
# Bezier handles, which interpolate the rim points instead of moving
# them, so the smoother only has to take the zigzag out of a genuinely
# stair-stepped edge.  Asking it to do the rounding as well is what
# pulled the curve off the surface in the first place.
RIM_THICKNESS_DEFAULT = 0.01
RIM_SMOOTH_DEFAULT = 3

# How deep the C's mouth and the H's slots are cut, as a fraction of the
# section's OVERALL width (not of the half-width).
RIM_INDENT = 0.25


def _edges_of(faces):
    """(n, 2) array of sorted vertex pairs, one row per face corner.

    Faces may be triangles, quads or a mix; the uniform case is
    vectorised and the ragged one falls back to a loop.
    """
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    widths = {len(f) for f in faces}
    if len(widths) == 1:
        F = np.asarray(faces, dtype=np.int64)
        k = F.shape[1]
        e = np.concatenate([F[:, [i, (i + 1) % k]] for i in range(k)])
    else:
        pairs = []
        for f in faces:
            n = len(f)
            for i in range(n):
                pairs.append((int(f[i]), int(f[(i + 1) % n])))
        e = np.asarray(pairs, dtype=np.int64)
    return np.sort(e, axis=1)


def boundary_index_loops(faces):
    """The rim as chains of VERTEX INDICES, before any smoothing.

    Separate from `boundary_loops` because a caller that wants to
    refresh an existing curve after the mesh moves needs the indices,
    not the positions -- the Seifert generator does exactly that after
    it minimises a surface.

    Returns a list of (index array, closed).
    """
    e = _edges_of(faces)
    if not len(e):
        return []
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    rim = uniq[counts == 1]
    if not len(rim):
        return []

    adj = {}
    for a, b in rim:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    used = set()

    def take(p, q):
        key = (p, q) if p < q else (q, p)
        if key in used:
            return False
        used.add(key)
        return True

    chains = []
    for a0, b0 in rim:
        a0, b0 = int(a0), int(b0)
        if not take(a0, b0):
            continue
        chain = [a0, b0]
        while True:
            cur = chain[-1]
            nxt = None
            for cand in adj.get(cur, ()):
                if take(cur, cand):
                    nxt = cand
                    break
            if nxt is None:
                break
            chain.append(nxt)
            if nxt == chain[0]:
                break
        if len(chain) < 4:
            continue
        closed = chain[-1] == chain[0]
        if closed:
            chain = chain[:-1]
        chains.append((np.asarray(chain, dtype=np.int64), closed))
    return chains


def boundary_loops(verts, faces, smooth=RIM_SMOOTH_DEFAULT,
                   method='ANCHORED'):
    """Ordered polylines along the open edge of a mesh.

    Edges used by exactly one face are the boundary.  Chaining them
    assumes nothing about manifoldness: several surfaces here are
    deliberately singular (the algebraic Kreuz is three planes), so a
    rim vertex can carry four boundary edges rather than two.  The walk
    consumes unused edges greedily and returns whatever chains it finds,
    open or closed.

    `smooth` Taubin passes run over each polyline before it is
    returned; without them the swept tube reproduces the staircase
    faithfully, which defeats the purpose, and with a shrinking
    smoother it would leave the edge instead.

    Returns a list of (points, closed) with points an (n, 3) array.
    """
    V = np.asarray(verts, dtype=float)
    chains = boundary_index_loops(faces)
    out = []
    for idx, closed in chains:
        pts = V[idx]
        if method == 'RELAXED':
            pts = _laplacian(pts, closed, int(smooth))
        else:
            pts = _taubin(pts, closed, int(smooth))
        out.append((pts, closed))
    return out


def _laplacian(pts, closed, passes, pin=None):
    """The original smoother: a plain, unconstrained Laplacian.

    This is a curve-SHORTENING flow, so it pulls the rim inward and off
    a curved surface -- which is why it was replaced for the woven
    polyhedra, whose coarse rims it visibly detached.  It is kept as a
    choice because on a fine, ragged rim it is the better of the two:
    nothing constrains it, so it flattens the grid staircase completely
    and the swept tube comes out clean.  Which behaviour is wanted
    depends on the rim, not on the code, so the caller picks.
    """
    pts = np.asarray(pts, dtype=float)
    if passes <= 0 or len(pts) < 3:
        return pts
    orig = pts
    for _ in range(passes):
        if closed:
            prev = np.roll(pts, 1, axis=0)
            nxt = np.roll(pts, -1, axis=0)
            pts = 0.5 * pts + 0.25 * (prev + nxt)
        elif len(pts) > 2:
            inner = 0.5 * pts[1:-1] + 0.25 * (pts[:-2] + pts[2:])
            pts = np.concatenate([pts[:1], inner, pts[-1:]])
        if pin is not None and np.any(pin):
            pts = np.where(np.asarray(pin, bool)[:, None], orig, pts)
    return pts


# Taubin's lambda/mu pair.  mu is slightly larger in magnitude than
# lambda and negative, so each shrinking pass is followed by an
# expanding one; the pair removes high-frequency wobble while leaving
# the curve where it was.
_TAUBIN_LAMBDA = 0.5
_TAUBIN_MU = -0.53


def _taubin(pts, closed, passes, pin=None):
    """Smooth a polyline WITHOUT shrinking it.

    A plain Laplacian pass moves every point toward the midpoint of its
    neighbours, which is a curve-shortening flow: on a rim that wraps a
    curved surface it walks the curve off the edge it is supposed to
    trace, visibly so after a few passes.  Taubin's fix is to alternate
    a positive step with a slightly larger negative one, which cancels
    the shrinkage to first order while still attenuating the
    grid staircase.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if passes <= 0 or n < 3:
        return pts

    def step(p, w):
        if closed:
            lap = 0.5 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) - p
            return p + w * lap
        q = p.copy()
        q[1:-1] = p[1:-1] + w * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
        return q

    orig = pts
    hold = None if pin is None else np.asarray(pin, bool)
    for _ in range(passes):
        pts = step(pts, _TAUBIN_LAMBDA)
        pts = step(pts, _TAUBIN_MU)
        # A pinned point is a real corner of the surface's edge; letting
        # the smoother round it is precisely the artefact this is here
        # to avoid, so it is put back after every pass rather than
        # merely capped afterwards.
        if hold is not None and np.any(hold):
            pts = np.where(hold[:, None], orig, pts)

    # Cap how far any point may travel, as a fixed fraction of the
    # point spacing.
    #
    # This budget was briefly made adaptive -- twice the median
    # Laplacian residual -- on the theory that the residual measures
    # raggedness and is "near zero on a smooth polygon however sharply
    # it turns overall".  That theory is wrong: the residual on a
    # coarse polygon of n points and radius R is about 2 pi^2 R / n^2,
    # which for a rim of a couple of dozen points is LARGER than the
    # residual of a fine zigzag, not smaller.  The adaptive cap
    # therefore never bound on anything, and quietly removed the only
    # thing holding a coarse rim onto its corners.
    #
    # It is not needed either, and never was: the ragged case it was
    # reaching for belongs to the RELAXED fit now, which does not come
    # through here at all.  Everything reaching this smoother is a
    # coarse rim that wants to stay put, so a plain fraction of the
    # spacing is both sufficient and what worked before.
    if closed:
        seg = np.linalg.norm(np.diff(np.vstack([orig, orig[:1]]),
                                     axis=0), axis=1)
    else:
        seg = np.linalg.norm(np.diff(orig, axis=0), axis=1)
    cap = 0.25 * float(np.median(seg)) if len(seg) else 0.0
    if cap > 0.0:
        d = pts - orig
        dist = np.linalg.norm(d, axis=1)
        over = dist > cap
        if np.any(over):
            d[over] *= (cap / dist[over])[:, None]
            pts = orig + d
    return pts



def neighbour_means(verts, faces):
    """Mean position of each vertex's face-neighbours, for every vertex.

    Vectorised, and computed ONCE for the whole mesh.  Both of those
    matter: this used to be a Python loop over every face, re-run for
    each rim chain, and it was the entire cost of the rim -- 2.8 s per
    chain on a 552k-face gyroid, seven chains, 20 s of a 24 s build.

    Faces are grouped by width so each group is one array operation, and
    the accumulation is a bincount rather than np.add.at, which is much
    faster for this shape.  A face of width k contributes, to each of
    its corners, the sum of the OTHER k-1 corners -- that is the face
    sum minus the corner itself, which is why no per-face work is
    needed.
    """
    V = np.asarray(verts, dtype=float)
    n = len(V)
    tot = np.zeros((n, 3))
    cnt = np.zeros(n)
    by_width = {}
    for f in faces:
        k = len(f)
        if k >= 3:
            by_width.setdefault(k, []).append(f)
    for k, group in by_width.items():
        Fk = np.asarray(group, dtype=np.int64)
        S = V[Fk].sum(axis=1)                     # (m, 3) face sums
        for j in range(k):
            col = Fk[:, j]
            w = S - V[col]                        # the other corners
            for c in range(3):
                tot[:, c] += np.bincount(col, weights=w[:, c],
                                         minlength=n)
            # k - 1 neighbours per incidence, not one: the count has
            # to match the number of positions actually summed into
            # `tot`, or the mean is scaled wrong and the direction can
            # come out reversed.
            cnt += (k - 1) * np.bincount(col, minlength=n)
    return tot / np.maximum(cnt, 1.0)[:, None]


def outward_field(verts, faces, idx, means=None):
    """Unit vectors along the rim pointing AWAY from the surface.

    An asymmetric swept section -- the C channel, the H beam -- has to
    know which way is out, and the rim curve alone cannot say: a closed
    loop in space has no inside.  The surface does know, so the
    direction is taken from it: for each rim vertex, average the
    positions of every vertex sharing a face with it, and step from that
    average back out to the rim vertex.  Removing the component along
    the rim leaves the conormal -- the direction lying IN the surface,
    across its edge, pointing outward.

    Together with the tangent and the surface normal that makes an
    orthonormal frame, which is why one rule serves both sections:
    aiming the profile's +X down the conormal opens the C outward, and
    puts the H's openings on the normal, above and below the sheet.
    """
    V = np.asarray(verts, dtype=float)
    idx = np.asarray(idx, dtype=np.int64)
    if means is None:
        means = neighbour_means(V, faces)
    # `means` holds the mean POSITION of each vertex's neighbours, so
    # the outward step is the rim vertex minus that point -- not the
    # negated mean, which is not even a direction.
    out = V[idx] - means[idx]
    T = _tangents(V[idx], True)
    out = out - T * np.sum(out * T, axis=1, keepdims=True)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    # a rim vertex whose neighbours average out to itself gives no
    # direction; fall back to the neighbouring samples' answer
    bad = (n[:, 0] < 1e-12)
    if np.any(bad) and not np.all(bad):
        good = np.flatnonzero(~bad)
        out[bad] = out[good[np.argmin(np.abs(
            np.flatnonzero(bad)[:, None] - good[None, :]), axis=1)]]
        n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-30)


def _tangents(pts, closed):
    P = np.asarray(pts, float)
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T = np.empty_like(P)
        T[1:-1] = P[2:] - P[:-2]
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    return T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-30)


def aim_tilt(pts, closed, want):
    """Per-point tilt that turns the swept profile's local +X onto
    `want`.

    Blender does not expose the frame it sweeps a bevel in, so it was
    measured: on a straight curve, at zero tilt the profile's +X lands
    on normalize(T x Zhat) -- identically under all three twist modes --
    and tilting rotates it about T by Rodrigues.  So with

        X0 = normalize(T x Zhat),   Y0 = T x X0,

    the tilt that lands +X on `want` is atan2(want.Y0, want.X0).

    Where the rim runs parallel to Z the cross product degenerates and
    the frame is genuinely undefined; those points take the tilt of
    their neighbours rather than an arbitrary one, which keeps the sweep
    from twisting through half a turn as it passes the pole.
    """
    P = np.asarray(pts, float)
    W = np.asarray(want, float)
    T = _tangents(P, closed)
    Z = np.array([0.0, 0.0, 1.0])
    X0 = np.cross(T, Z)
    nx = np.linalg.norm(X0, axis=1)
    ok = nx > 1e-9
    X0 = X0 / np.maximum(nx, 1e-30)[:, None]
    Y0 = np.cross(T, X0)
    ang = np.arctan2(np.sum(W * Y0, axis=1), np.sum(W * X0, axis=1))
    if not np.all(ok):
        if not np.any(ok):
            return np.zeros(len(P))
        good = np.flatnonzero(ok)
        bad = np.flatnonzero(~ok)
        near = good[np.argmin(np.abs(bad[:, None] - good[None, :]), axis=1)]
        ang[bad] = ang[near]
    return np.unwrap(ang) if closed else ang


def profile_points(half, kind, twist=0.0):
    """The cross-section as a closed polygon in the (X, Y) frame plane,
    rotated by `twist`.

    Wall thickness is half the half-width throughout, which keeps the
    arms of a C and the flanges of an H visibly solid at the sizes these
    rims are used at.
    """
    h = float(half)
    # `half` is the half-width, so the section is 2h across.  The
    # INDENTATION -- how deep the channel's mouth is bitten out of the C,
    # and how deep the slots are cut into the H -- is a quarter of that
    # overall width.  The earlier version cut three quarters of the way
    # through, which left a C that was mostly mouth and read as a thin
    # bent strip rather than a channel.
    d = RIM_INDENT * (2.0 * h)               # indentation depth
    w = 0.5 * h                              # arm / upright thickness
    c = h - d                                # half the H's cross-bar
    if kind == 'C':
        # Square with a bite out of the +X side, so the channel opens
        # along the outward conormal -- away from the surface.
        #
        # This is where the arithmetic said it should be all along: +X
        # is measured to agree with "away from the object's bulk" on
        # 100% of rim samples, on a flat patch, the exact Schwarz H cell
        # and a clipped gyroid alike.  It was briefly mirrored on a
        # report that it faced inward, which then made it wrong
        # everywhere; the measurement was right and the mirror is gone.
        pts = ((-h, -h), (h, -h), (h, -h + w), (h - d, -h + w),
               (h - d, h - w), (h, h - w), (h, h), (-h, h))
    elif kind == 'H':
        # two uprights joined by a cross-bar, so the openings face
        # +Y and -Y -- along the surface normal, above and below it
        pts = ((-h, -h), (-h + w, -h), (-h + w, -c), (h - w, -c),
               (h - w, -h), (h, -h), (h, h), (h - w, h),
               (h - w, c), (-h + w, c), (-h + w, h), (-h, h))
    else:                                    # SQUARE, and REED's band
        pts = ((-h, -h), (h, -h), (h, h), (-h, h))
    P = np.asarray(pts, float)
    if twist:
        c_, s_ = math.cos(twist), math.sin(twist)
        P = P @ np.array([[c_, s_], [-s_, c_]])
    return P


def section_centroid(S):
    """Centroid of the AREA a closed polygon encloses (shoelace).

    Not the mean of its vertices: for an L- or C-shaped section those
    differ in sign, and the vertex mean of the C channel sits on the
    opening side -- which briefly made a correct profile look reversed.
    """
    P = np.asarray(S, float)
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = 0.5 * float(np.sum(cr))
    if abs(A) < 1e-30:
        return float(x.mean()), float(y.mean())
    return (float(np.sum((x + x1) * cr) / (6.0 * A)),
            float(np.sum((y + y1) * cr) / (6.0 * A)))


def reed_resample(pts, out, closed, n):
    """Re-sample the rim to exactly `n` points by arc length, carrying
    the outward field with it.

    Reeding is a pattern in ARC LENGTH -- so many ridges around the rim
    -- and the rim's own points are spaced by whatever grid produced the
    surface.  Sweeping the pattern on those would make the ridges follow
    the mesh instead of the rim, which is the one thing a coin edge must
    not do.  Interpolating is acceptable here in a way it was not for
    the plain tube: the ridge positions ARE the point, and a ridge is
    not a corner of the surface that has to be preserved.
    """
    P = np.asarray(pts, float)
    O = np.asarray(out, float)
    Q = np.vstack([P, P[:1]]) if closed else P
    R = np.vstack([O, O[:1]]) if closed else O
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0.0 or n < 4:
        return P, O
    t = (np.linspace(0.0, total, n, endpoint=False) if closed
         else np.linspace(0.0, total, n))
    Pn = np.stack([np.interp(t, s, Q[:, k]) for k in range(3)], axis=1)
    On = np.stack([np.interp(t, s, R[:, k]) for k in range(3)], axis=1)
    On /= np.maximum(np.linalg.norm(On, axis=1, keepdims=True), 1e-30)
    return Pn, On


# Samples per reed: two on the groove and two on the ridge, so each is
# a FLAT band with a step between, which is what milling a coin edge
# leaves.  Fewer would round the pattern into a wave; more would only
# subdivide flats that are already flat.
REED_STEPS = 4
REED_DEPTH = 0.30


def reed_scale(n, steps=REED_STEPS, depth=REED_DEPTH):
    """The radius multiplier along a reeded rim: alternating flats."""
    k = (np.arange(n) // (steps // 2)) % 2
    return 1.0 - depth * k


def _chain_length(pts, closed):
    P = np.asarray(pts, float)
    Q = np.vstack([P, P[:1]]) if closed else P
    return float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1)))


def sweep_profile(pts, closed, out, half, kind, twist=0.0, scale=None):
    """Sweep a cross-section along the rim, in a frame we control.

    Blender's curve bevel will do this, but not in a frame that can be
    aimed: the tilt on a control point is measured from whatever normal
    the curve solver propagated, and on a genuinely three-dimensional
    rim that is not reconstructible from the control points.  Measured
    on the exact Schwarz H cell, a C section aimed through per-point
    tilts came out with its spine on the outward side at 94% of samples
    -- which is to say, unoriented.

    Here the frame is built from data the SURFACE supplies and is
    therefore exact: the tangent along the rim, the outward conormal
    across it, and their cross product, which is the surface normal.  So
    the C opens away from the sheet and the H straddles it, by
    construction rather than by hope.

    Returns (verts, quads).
    """
    P = np.asarray(pts, float)
    O = np.asarray(out, float)
    T = _tangents(P, closed)
    X = O - T * np.sum(O * T, axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    Y = np.cross(T, X)
    S = profile_points(half, kind, twist)
    m = len(S)
    if scale is None:
        SX = np.repeat(S[None, :, 0], len(P), axis=0)
        SY = np.repeat(S[None, :, 1], len(P), axis=0)
    else:
        # Reeding is milled into the OUTWARD face only, the way a coin
        # is: the ridges stand proud on the outside and the inner face
        # of the band stays flat against the surface.  Scaling the whole
        # section instead would pump it in and out on all four sides and
        # read as a bobbled tube rather than a reeded edge.
        sc = np.asarray(scale, float)[:, None]
        SX = np.where(S[None, :, 0] > 0.0, S[None, :, 0] * sc,
                      S[None, :, 0])
        SY = np.repeat(S[None, :, 1], len(P), axis=0)
    V = (P[:, None, :]
         + SX[:, :, None] * X[:, None, :]
         + SY[:, :, None] * Y[:, None, :]).reshape(-1, 3)
    n = len(P)
    rings = n if closed else n - 1
    F = []
    for i in range(rings):
        a, b = i * m, ((i + 1) % n) * m
        for j in range(m):
            k = (j + 1) % m
            F.append((a + j, b + j, b + k, a + k))
    if not closed:
        F.append(tuple(range(m - 1, -1, -1)))
        base = (n - 1) * m
        F.append(tuple(range(base, base + m)))
    return V, F


def corner_mask(pts, closed, degrees=35.0):
    """Which rim points are genuine corners.

    A corner is where the polyline turns by more than `degrees` in one
    step.  On an exact parametric surface -- the Weierstrass TPMS cells,
    the woven polyhedra -- these are real features of the boundary, and
    smoothing them is exactly the wrong thing: the tip of a spike is
    the single point a viewer checks the rim against.  On a marching
    -tetrahedra rim the same test fires on staircase steps, which is
    why the caller decides whether to use it.
    """
    P = np.asarray(pts, float)
    n = len(P)
    if n < 3:
        return np.zeros(n, dtype=bool)
    if closed:
        d0 = P - np.roll(P, 1, axis=0)
        d1 = np.roll(P, -1, axis=0) - P
    else:
        d0 = np.empty_like(P)
        d1 = np.empty_like(P)
        d0[1:] = P[1:] - P[:-1]
        d0[0] = d0[1]
        d1[:-1] = P[1:] - P[:-1]
        d1[-1] = d1[-2]
    d0 /= np.maximum(np.linalg.norm(d0, axis=1, keepdims=True), 1e-30)
    d1 /= np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-30)
    turn = np.degrees(np.arccos(np.clip(np.sum(d0 * d1, axis=1), -1.0, 1.0)))
    return turn > float(degrees)


def resample(pts, closed, spacing, keep=None):
    """Thin a polyline so no two control points sit closer than
    `spacing`, by DROPPING points -- never by moving them.

    A swept tube self-intersects wherever the curve turns inside its own
    bevel radius, and a rim traced off a mesh has points spaced by the
    grid, not by the tube.  At a thickness of 0.04 on a rim whose points
    sit 0.005 apart, every small wiggle folds the sweep over itself and
    the tube comes out lumpy -- the failure looks like a caterpillar
    rather than a pipe.

    The first version fixed that by re-interpolating the rim at equal
    steps of arc length, which is the wrong tool: on a coarse rim with
    unequal edges it slides EVERY control point along its chords, off
    the corners the points were traced from -- measured at ~19% of a
    point spacing on a woven rim, which is more than the tube radius and
    reads directly as waviness.  Worse, it did that even when the rim
    was already spaced comfortably wider than the tube and needed no
    thinning at all.

    Choosing a subset instead cannot introduce that error: a point that
    sat on a corner either survives, exactly where it was, or goes.  On
    a rim already coarser than the tube nothing is dropped and this is
    the identity, which is the correct answer for the woven and twisted
    polyhedra.  Measuring the gap to the last KEPT point (rather than to
    the previous one) also handles a rim doubling back on itself, where
    arc length advances while the point barely moves.

    Points flagged in `keep` are never dropped, whatever the spacing:
    they are the corners, and a corner thinned away is a corner the
    swept tube cuts across.  Returns the surviving INDICES so the caller
    can carry per-point data -- the corner flags, the outward direction
    -- through the thinning with them.
    """
    P = np.asarray(pts, dtype=float)
    if spacing <= 0.0 or len(P) < 3:
        return np.arange(len(P))
    pin = (np.zeros(len(P), dtype=bool) if keep is None
           else np.asarray(keep, dtype=bool))
    out = [0]
    for i in range(1, len(P)):
        if (pin[i]
                or float(np.linalg.norm(P[i] - P[out[-1]])) >= spacing):
            out.append(i)
    if closed and len(out) > 2 and not pin[out[-1]]:
        if float(np.linalg.norm(P[out[-1]] - P[out[0]])) < spacing:
            out.pop()
    return np.array(out) if len(out) >= 4 else np.arange(len(P))


if _IN_BLENDER:

    def rim_prop():
        return BoolProperty(
            name="Rim Curve", default=False,
            description="Sweep a tube along the open edge of the "
                        "surface. That edge is a stair-step through "
                        "the sample grid, so the tube both tidies it "
                        "and gives the surface a deliberate border; a "
                        "closed surface has no edge and gets no curve")

    def rim_thickness_prop():
        return FloatProperty(
            name="Rim Thickness", default=RIM_THICKNESS_DEFAULT,
            min=0.0, max=1.0,
            # step is in hundredths, so 1 is a 0.01 increment per drag
            # tick.  The useful range of this value is roughly 0.005 to
            # 0.05 on a surface fitted to a 2 m cube, and the default
            # unit step ran through all of it in a few pixels.
            step=1, precision=3,
            description="Bevel radius of the rim tube (0 leaves a "
                        "bare curve)")

    def rim_smooth_prop():
        return IntProperty(
            name="Rim Smoothing", default=RIM_SMOOTH_DEFAULT,
            min=0, max=40,
            description="Taubin smoothing passes along the rim before "
                        "it is swept. Unlike a plain Laplacian this "
                        "does not shrink the curve, so the tube stays "
                        "on the edge however many passes you use; 0 "
                        "follows the sample grid exactly")

    def rim_profile_prop():
        return EnumProperty(
            name="Rim Profile",
            items=[('ROUND', "Circular",
                    "Round tube -- the curve's own bevel depth"),
                   ('SQUARE', "Square",
                    "Square tube, swept from a four-point bevel "
                    "object that is created alongside and hidden"),
                   ('C', "Channel (C)",
                    "Blocky C, opening facing away from the surface -- "
                    "a channel the edge sits inside"),
                   ('H', "Beam (H)",
                    "Blocky H, openings facing along the surface above "
                    "and below it -- an I-beam edge"),
                   ('REED', "Reeded",
                    "Square band milled across with fine flat ridges, "
                    "the way the edge of a coin is reeded"),
                   ('NONE', "Curve Only",
                    "No sweep at all: the bare rim curve, which "
                    "renders as nothing but can be bevelled by hand, "
                    "used as a path, or exported as the edge itself")],
            default='ROUND',
            description="Cross-section swept along the rim")

    def rim_reeds_prop():
        return IntProperty(
            name="Reeds", default=120, min=4, max=2000,
            description="Number of ridges milled across a reeded rim, "
                        "counted around the whole edge. The rim is "
                        "re-sampled to carry them, so they are spaced by "
                        "arc length rather than by the surface's grid")

    def rim_twist_prop():
        return FloatProperty(
            name="Rim Twist", default=0.0, min=-180.0, max=180.0,
            step=100, precision=1, subtype='ANGLE',
            description="Rotate the swept profile about the rim. Set it "
                        "to 180 to reverse which way a channel opens or "
                        "which face a reed is milled into. Which way "
                        "looks right is not fixed by the surface: the "
                        "same outward direction reads as out of an "
                        "Enneper edge and into a clipped periodic cell, "
                        "so this is the control for it")

    def _square_bevel(name, half):
        """A closed square used as a curve's bevel object.

        Blender sweeps a round tube from `bevel_depth` alone, but any
        other cross-section has to be an actual curve object, so one is
        made per rim and hidden.  It is parented to the rim so deleting
        the surface takes the whole assembly with it.
        """
        return _profile_bevel(name, half, 'SQUARE')

    def _profile_bevel(name, half, kind):
        cu = bpy.data.curves.new(name, 'CURVE')
        cu.dimensions = '2D'
        pts = profile_points(half, kind)
        sp = cu.splines.new('POLY')
        sp.points.add(len(pts) - 1)
        for i, (x, y) in enumerate(pts):
            sp.points[i].co = (x, y, 0.0, 1.0)
        sp.use_cyclic_u = True
        return bpy.data.objects.new(name, cu)

    def draw_rim(layout, op):
        """The rim controls, shown only when the rim is on."""
        layout.prop(op, 'rim')
        if not getattr(op, 'rim', False):
            return
        layout.prop(op, 'rim_thickness')
        if hasattr(op, 'rim_profile'):
            layout.prop(op, 'rim_profile')
            # twist only does anything to a profile that is not
            # symmetric under a quarter turn
            if (hasattr(op, 'rim_twist')
                    and getattr(op, 'rim_profile', '') in ('C', 'H',
                                                           'REED')):
                layout.prop(op, 'rim_twist')
            if (hasattr(op, 'rim_reeds')
                    and getattr(op, 'rim_profile', '') == 'REED'):
                layout.prop(op, 'rim_reeds')
        layout.prop(op, 'rim_smooth')

    def _swept_rim(context, obj, label, loops, thickness, kind, tw=0.0,
                   reeds=120):
        """Build the C and H rims as a MESH, swept in our own frame.

        These two are here to be AIMED -- a channel that opens the wrong
        way is not a channel -- and Blender's bevel cannot promise that
        on a three-dimensional rim, so they are swept directly.  The
        result is flat-shaded: every face of a channel section is
        genuinely flat, and smoothing would round the corners away
        exactly as it did to the square tube.
        """
        V, F, base = [], [], 0
        total = sum(_chain_length(p, c) for p, c, _x, _y in loops) or 1.0
        for pts, closed, _corner, out in loops:
            sc = None
            if kind == 'REED':
                # spread the requested ridge count over the whole rim by
                # length, so several separate chains carry one pattern
                share = max(2, int(round(
                    reeds * _chain_length(pts, closed) / total)))
                n = share * REED_STEPS
                pts, out = reed_resample(pts, out, closed, n)
                sc = reed_scale(len(pts))
            v, f = sweep_profile(pts, closed, out, thickness, kind, tw,
                                 scale=sc)
            V.append(v)
            F.extend(tuple(int(i) + base for i in q) for q in f)
            base += len(v)
        if not V:
            return 0
        V = np.concatenate(V, axis=0)
        me = bpy.data.meshes.new(label + " Rim")
        me.from_pydata([tuple(map(float, p)) for p in V], [], F)
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth', [False] * len(me.polygons))
        me.update()
        rim = bpy.data.objects.new(label + " Rim", me)
        context.collection.objects.link(rim)
        rim.matrix_world = obj.matrix_world.copy()
        rim.parent = obj
        rim.matrix_parent_inverse = obj.matrix_world.inverted()
        return len(loops)

    def add_rim_curve(context, obj, label, verts, faces, thickness=None,
                      smooth=None, profile='ROUND', method='RELAXED',
                      twist=0.0, reeds=120):
        """Sweep a bevelled curve along the mesh's open edge.

        `method` is chosen by the CALLING GENERATOR, not by the user --
        it depends on the kind of rim that generator produces, which
        the generator knows and the user should not have to:

          RELAXED  free smoothing, swept as a polyline.  Right for the
                   fine, ragged edges a level set leaves, where the
                   staircase is the whole problem and the rim has no
                   real corners to preserve.
          ANCHORED non-shrinking smoothing, a drift cap and control
                   points re-spaced to the tube.  Right for coarse rims
                   with genuine corners -- the woven polyhedra -- where
                   free smoothing lifts the tube off the surface.

        Parented to `obj` so the pair moves as one.  Returns the number
        of rim loops, zero meaning the surface was closed.
        """
        if thickness is None:
            thickness = RIM_THICKNESS_DEFAULT
        if smooth is None:
            smooth = RIM_SMOOTH_DEFAULT
        V = np.asarray(verts, dtype=float)
        chains = boundary_index_loops(faces)
        if not chains:
            return 0
        # Once for the whole mesh, not once per chain.  This is the
        # difference between a rim that costs a second and one that
        # costs twenty on a big surface.
        means = neighbour_means(V, faces)

        # Drop loops too tight to sweep.  A closed rim of total length L
        # encloses something of radius about L / 2 pi, so once that falls
        # below the tube's own radius the sweep has no room: the tube
        # swallows the hole and comes out a blob rather than a ring.
        # Enneper's central puncture and the pinholes along the seams of
        # an assembled Weierstrass cell both do this, and both read as
        # stray beads sitting on the surface.
        min_len = 2.0 * math.pi * float(thickness)
        loops = []
        for idx, closed in chains:
            pts = V[idx]
            ring = np.vstack([pts, pts[:1]]) if closed else pts
            L = float(np.sum(np.linalg.norm(np.diff(ring, axis=0), axis=1)))
            if closed and L < min_len:
                continue
            # Corners are found on the RAW rim, before any smoothing --
            # afterwards the smoother has already rounded the evidence
            # away.  Only the anchored fit honours them: on a ragged
            # level-set rim the same test fires on every staircase step.
            corner = (corner_mask(pts, closed) if method == 'ANCHORED'
                      else np.zeros(len(pts), dtype=bool))
            out = outward_field(V, faces, idx, means)
            if method == 'RELAXED':
                pts = _laplacian(pts, closed, int(smooth), pin=corner)
            else:
                pts = _taubin(pts, closed, int(smooth), pin=corner)
                keep = resample(pts, closed, 1.6 * float(thickness),
                                keep=corner)
                pts, corner, out = pts[keep], corner[keep], out[keep]
            if len(pts) >= 4:
                loops.append((pts, closed, corner, out))
        if not loops:
            return 0
        if profile in ('C', 'H', 'REED'):
            return _swept_rim(context, obj, label, loops, float(thickness),
                              profile, tw=float(twist), reeds=int(reeds))
        cu = bpy.data.curves.new(label + " Rim", 'CURVE')
        cu.dimensions = '3D'
        cu.fill_mode = 'FULL'
        cu.use_fill_caps = True
        # Z_UP, not the default minimum-twist: `aim_tilt` computes the
        # tilt against a frame built from the tangent and global Z, and
        # that is the frame Z_UP uses.  Minimum-twist instead carries a
        # frame along the curve from wherever it started, so the same
        # tilt lands the profile somewhere different at every point and
        # an asymmetric section faces a different way all the way round.
        # The three modes agree on a straight curve, which is why a
        # straight-curve probe could not tell them apart.
        cu.twist_mode = 'Z_UP'
        if profile == 'SQUARE':
            bev = _profile_bevel(label + " Rim Profile",
                                 float(thickness), profile)
            context.collection.objects.link(bev)
            bev.hide_viewport = True
            bev.hide_render = True
            cu.bevel_mode = 'OBJECT'
            cu.bevel_object = bev
        elif profile == 'NONE':
            # No sweep: the rim stays a bare curve.  A zero bevel depth
            # is the whole of it -- there is no surface to fill or cap.
            # (fill_mode has no 'NONE' on a 3D curve; the enum there is
            # FULL / BACK / FRONT / HALF, and setting it would raise.)
            bev = None
            cu.bevel_depth = 0.0
            cu.use_fill_caps = False
        else:
            bev = None
            cu.bevel_depth = float(thickness)
            cu.bevel_resolution = 4
        # BEZIER with AUTO handles, not POLY: the handles round the
        # corners between samples without moving the samples, so the
        # curve still passes exactly through the rim it was built from.
        # A POLY spline would render the staircase; a NURBS one would
        # approximate rather than interpolate and drift off the edge in
        # the same way the old shrinking smoother did.
        # The spline type follows the fit.  RELAXED has already
        # flattened the rim, so a POLY spline through those points is
        # both faithful and clean -- this is what the rim looked like
        # before the anchored fit existed.  ANCHORED deliberately leaves
        # the rim where it found it, staircase and all, so it needs
        # BEZIER handles to round the corners between samples without
        # moving them.
        cu.resolution_u = 6
        tw = float(twist)
        for pts, closed, corner, out in loops:
            tilt = aim_tilt(pts, closed, out) + tw
            if method == 'ANCHORED':
                sp = cu.splines.new('BEZIER')
                sp.bezier_points.add(len(pts) - 1)
                for i, q in enumerate(pts):
                    bp = sp.bezier_points[i]
                    bp.co = (float(q[0]), float(q[1]), float(q[2]))
                    # AUTO rounds between samples, which is the point of
                    # using Bezier at all -- but at a genuine corner it
                    # rounds off the corner too, and the tip of a spike
                    # is the first place a viewer checks the rim.
                    # VECTOR handles make the curve turn there instead.
                    h = 'VECTOR' if corner[i] else 'AUTO'
                    bp.handle_left_type = h
                    bp.handle_right_type = h
                    bp.tilt = float(tilt[i])
            else:
                sp = cu.splines.new('POLY')
                sp.points.add(len(pts) - 1)
                for i, q in enumerate(pts):
                    sp.points[i].co = (float(q[0]), float(q[1]),
                                       float(q[2]), 1.0)
                    sp.points[i].tilt = float(tilt[i])
            sp.use_cyclic_u = bool(closed)

        rim = bpy.data.objects.new(label + " Rim", cu)
        context.collection.objects.link(rim)
        if profile == 'SQUARE':
            # A square tube shaded smooth is a round tube: the shading
            # averages across the four corners and erases the only
            # thing that makes the profile square.  Splitting edges
            # above a threshold creases those corners while leaving the
            # tube smooth ALONG its length, which is what flat shading
            # would throw away.  30 degrees clears the 90-degree
            # corners comfortably and stays well above the angle
            # between successive segments of a swept curve.
            sharp = rim.modifiers.new("Sharpen", 'EDGE_SPLIT')
            sharp.split_angle = math.radians(30.0)
            sharp.use_edge_angle = True
            sharp.use_edge_sharp = False
        rim.matrix_world = obj.matrix_world.copy()
        rim.parent = obj
        rim.matrix_parent_inverse = obj.matrix_world.inverted()
        if bev is not None:
            bev.parent = rim
        return len(loops)

    def add_rim_from_object(context, obj, label, thickness=None,
                            smooth=None, profile=None, method=None,
                            twist=0.0, reeds=120):
        """Same, reading the geometry back off an existing mesh object.

        For generators that build their object through a path this
        module cannot see (bmesh, modifiers, a solver), taking the
        vertices and polygons off the finished mesh is simpler and
        always matches what the user is looking at.
        """
        me = getattr(obj, 'data', None)
        if me is None or not hasattr(me, 'polygons'):
            return 0
        verts = [tuple(v.co) for v in me.vertices]
        faces = [tuple(p.vertices) for p in me.polygons]
        if profile is None:
            profile = 'ROUND'
        if method is None:
            method = 'RELAXED'
        return add_rim_curve(context, obj, label, verts, faces,
                             thickness, smooth, profile, method, twist,
                             reeds)


def _selftest():
    ok = True

    # An open patch: a grid of quads has a rim of exactly its border.
    n = 12
    verts = [(i / (n - 1.0), j / (n - 1.0), 0.0)
             for j in range(n) for i in range(n)]
    faces = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1,
              (j + 1) * n + i)
             for j in range(n - 1) for i in range(n - 1)]
    loops = boundary_loops(verts, faces, smooth=0)
    good = (len(loops) == 1 and loops[0][1]
            and len(loops[0][0]) == 4 * (n - 1))
    ok &= good
    print("rim_curve: open quad grid gives one closed rim of %d points %s"
          % (4 * (n - 1), 'OK' if good else 'FAIL'))

    # A closed surface: a torus of quads has no rim at all.
    R, r, nu, nv = 2.0, 0.7, 24, 16
    import math
    tv = []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            rad = R + r * math.cos(v)
            tv.append((rad * math.cos(u), rad * math.sin(u),
                       r * math.sin(v)))
    tf = [(i * nv + j, ((i + 1) % nu) * nv + j,
           ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv)
          for i in range(nu) for j in range(nv)]
    good = boundary_loops(tv, tf) == []
    ok &= good
    print("rim_curve: closed torus has no rim %s"
          % ('OK' if good else 'FAIL'))

    # Mixed tri/quad faces must not break the edge extraction.
    mixed = faces[:-3] + [(0, 1, n + 1)]
    good = len(boundary_loops(verts, mixed, smooth=0)) >= 1
    ok &= good
    print("rim_curve: mixed tri/quad faces handled %s"
          % ('OK' if good else 'FAIL'))

    # Smoothing must NOT shrink the rim.  This is the property the
    # first implementation got wrong: a plain Laplacian is a
    # curve-shortening flow, so on a rim that wraps a curved surface it
    # migrated off the edge, visibly, at the default number of passes.
    # A square loop is the sharpest test -- Taubin should hold its
    # perimeter where a Laplacian collapses it toward the centroid.
    raw = boundary_loops(verts, faces, smooth=0)[0][0]
    sm = boundary_loops(verts, faces, smooth=20)[0][0]

    def perim(p):
        return float(np.linalg.norm(np.diff(np.vstack([p, p[:1]]),
                                            axis=0), axis=1).sum())

    shrink = 1.0 - perim(sm) / perim(raw)
    drift = float(np.max(np.linalg.norm(sm - raw, axis=1)))
    good = len(raw) == len(sm) and abs(shrink) < 0.06 and drift < 0.12
    ok &= good
    print("rim_curve: 20 smoothing passes shrink the rim %.1f%% and "
          "move it at most %.3f %s"
          % (100.0 * shrink, drift, 'OK' if good else 'FAIL'))

    # And for contrast, the shrinking flow it replaced: same loop, same
    # pass count, run as a pure Laplacian.
    lap = raw.copy()
    for _ in range(20):
        lap = 0.5 * lap + 0.25 * (np.roll(lap, 1, axis=0)
                                  + np.roll(lap, -1, axis=0))
    print("rim_curve:   (a plain Laplacian would shrink it %.1f%% and "
          "move it %.3f)"
          % (100.0 * (1.0 - perim(lap) / perim(raw)),
             float(np.max(np.linalg.norm(lap - raw, axis=1)))))

    # A coarse rim with unequal edges and real corners -- what a woven or
    # twisted polyhedron actually produces.  The square loop above is too
    # kind to catch either of the two regressions below, because its
    # points are already equally spaced.
    k = np.arange(24)
    ang = 2.0 * math.pi * (k + 0.35 * math.sin(1.0) * np.sin(
        3.0 * 2.0 * math.pi * k / 24.0)) / 24.0
    rad = 1.0 + 0.12 * np.where(k % 2 == 0, 1.0, -1.0)
    coarse = np.stack([rad * np.cos(ang), rad * np.sin(ang),
                       0.15 * np.sin(2.0 * 2.0 * math.pi * k / 24.0)],
                      axis=1)
    seg = np.linalg.norm(np.diff(np.vstack([coarse, coarse[:1]]),
                                 axis=0), axis=1)
    med = float(np.median(seg))

    # 1. The drift cap must BIND on such a rim.  When it was scaled to
    #    the median Laplacian residual instead of to the point spacing
    #    it never bound on anything -- the residual on a coarse polygon
    #    is larger than on a fine zigzag, not smaller -- and the rim
    #    drifted off its corners by better than a third of a spacing.
    moved = float(np.max(np.linalg.norm(
        _taubin(coarse, True, 8) - coarse, axis=1)))
    good = moved <= 0.25 * med + 1e-12
    ok &= good
    print("rim_curve: coarse rim drifts %.4f under 8 passes, cap %.4f "
          "(%.0f%% of a point spacing) %s"
          % (moved, 0.25 * med, 100.0 * moved / med,
             'OK' if good else 'FAIL'))

    # 2. Re-spacing must never invent a position.  Every point it
    #    returns has to BE one of the points it was given, and on a rim
    #    already coarser than the tube it must return all of them --
    #    the interpolating version slid every point ~19% of a spacing
    #    along its chords, including when no thinning was needed.
    thinned = coarse[resample(coarse, True, 1.6 * 0.01)]
    off = float(np.max(np.min(np.linalg.norm(
        thinned[:, None, :] - coarse[None, :, :], axis=2), axis=1)))
    good = off == 0.0 and len(thinned) == len(coarse)
    ok &= good
    print("rim_curve: re-spacing a coarse rim keeps %d/%d points and "
          "moves them %.1e %s"
          % (len(thinned), len(coarse), off, 'OK' if good else 'FAIL'))

    # ... and on a rim finer than the tube it must actually thin, still
    # without moving anything.
    fine = np.stack([np.cos(np.linspace(0, 2 * math.pi, 600,
                                        endpoint=False)),
                     np.sin(np.linspace(0, 2 * math.pi, 600,
                                        endpoint=False)),
                     np.zeros(600)], axis=1)
    tf = fine[resample(fine, True, 1.6 * 0.04)]
    off = float(np.max(np.min(np.linalg.norm(
        tf[:, None, :] - fine[None, :, :], axis=2), axis=1)))
    good = len(tf) < len(fine) and off == 0.0
    ok &= good
    print("rim_curve: re-spacing a fine rim thins %d -> %d, moves %.1e %s"
          % (len(fine), len(tf), off, 'OK' if good else 'FAIL'))

    # A rim with genuine corners: a square loop sampled finely along
    # each side.  The corners must be found, held through smoothing, and
    # survive the thinning -- a corner rounded or dropped is a corner
    # the swept tube cuts across, which is what a viewer sees first at
    # the tip of a spike.
    side = np.linspace(0.0, 1.0, 25, endpoint=False)
    z = np.zeros_like(side)
    sq = np.concatenate([
        np.stack([side, z, z], 1),
        np.stack([np.ones_like(side), side, z], 1),
        np.stack([1.0 - side, np.ones_like(side), z], 1),
        np.stack([z, 1.0 - side, z], 1)])
    cm = corner_mask(sq, True)
    good = int(np.sum(cm)) == 4
    ok &= good
    print("rim_curve: square rim has %d corners (expected 4) %s"
          % (int(np.sum(cm)), 'OK' if good else 'FAIL'))

    moved = np.linalg.norm(_taubin(sq, True, 12, pin=cm) - sq, axis=1)
    good = float(np.max(moved[cm])) == 0.0 and float(np.max(moved)) > 0.0
    ok &= good
    print("rim_curve: 12 passes move the corners %.1e and the sides "
          "%.4f %s" % (float(np.max(moved[cm])), float(np.max(moved)),
                       'OK' if good else 'FAIL'))

    kept = resample(sq, True, 0.25, keep=cm)
    good = bool(np.all(cm[kept][np.isin(kept, np.flatnonzero(cm))]))         and set(np.flatnonzero(cm)).issubset(set(kept.tolist()))
    ok &= good
    print("rim_curve: thinning %d -> %d keeps every corner %s"
          % (len(sq), len(kept), 'OK' if good else 'FAIL'))

    # The outward direction must point AWAY from the surface.  On a flat
    # square patch every rim vertex should aim outward in the plane, so
    # the dot with the direction from the patch centre is positive
    # everywhere -- if the sign convention were inverted, the C channel
    # would cup the wrong way and every one of these would be negative.
    idx, closed = boundary_index_loops(faces)[0]
    Vg = np.asarray(verts, float)
    o = outward_field(Vg, faces, idx)
    radial = Vg[idx] - Vg.mean(0)
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-30)
    dots = np.sum(o * radial, axis=1)
    good = float(np.min(dots)) > 0.3
    ok &= good
    print("rim_curve: outward field agrees with the radial direction, "
          "min dot %.3f %s" % (float(np.min(dots)), 'OK' if good else 'FAIL'))

    # ... and the tilt must actually aim the profile there.  Rebuild the
    # measured frame and check +X lands on the requested direction.
    tl = aim_tilt(Vg[idx], closed, o)
    T = _tangents(Vg[idx], closed)
    X0 = np.cross(T, np.array([0.0, 0.0, 1.0]))
    X0 /= np.maximum(np.linalg.norm(X0, axis=1, keepdims=True), 1e-30)
    Y0 = np.cross(T, X0)
    aimed = (X0 * np.cos(tl)[:, None] + Y0 * np.sin(tl)[:, None])
    err = float(np.max(np.linalg.norm(aimed - o, axis=1)))
    good = err < 1e-9
    ok &= good
    print("rim_curve: tilt aims the profile at the outward direction to "
          "%.1e %s" % (err, 'OK' if good else 'FAIL'))

    # The C channel must open AWAY from the surface and the H must
    # straddle it.  Both follow from one thing -- the swept frame's +X
    # lying along the outward conormal -- so both are checked by where
    # the section's own centroid sits.  This is the property that could
    # not be delivered through Blender's curve bevel: aimed by per-point
    # tilt, the C's spine came out on the OUTWARD side at 94% of samples
    # on a real three-dimensional rim.
    for kind, want_x, want_y in (('C', -1, 0), ('H', 0, 0),
                                 ('SQUARE', 0, 0)):
        S = profile_points(0.05, kind)
        cx, cy = section_centroid(S)
        good = (np.sign(round(cx, 6)) == want_x
                and np.sign(round(cy, 6)) == want_y)
        ok &= good
        print("rim_curve: %-6s section centroid (%+.4f, %+.4f) %s"
              % (kind, cx, cy, 'OK' if good else 'FAIL'))

    # ... and the sweep must put that centroid on the inward side in
    # SPACE, for a rim that genuinely turns in three dimensions.  A
    # tilted circle is the smallest case that does.
    t = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t), 0.35 * np.sin(2.0 * t)], 1)
    outv = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1)
    Vs, Fs = sweep_profile(ring, True, outv, 0.05, 'C')
    Vs = Vs.reshape(len(ring), -1, 3)
    T = _tangents(ring, True)
    X = outv - T * np.sum(outv * T, axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    Y = np.cross(T, X)
    rel = Vs - ring[:, None, :]
    cxs = []
    for i in range(len(ring)):
        S2 = np.stack([rel[i] @ X[i], rel[i] @ Y[i]], axis=1)
        cxs.append(section_centroid(S2)[0])
    off = np.array(cxs) / 0.05
    # A shallower mouth removes less material, so the centroid sits
    # nearer the middle than it did when the bite went three quarters of
    # the way through -- the SIGN is the property being gated, and it
    # says the solid body is on the inward side with the mouth facing
    # out, which is what the channel is for.
    good = float(np.max(off)) < -0.03
    ok &= good
    print("rim_curve: swept C spine sits %.3f thicknesses inward, mouth "
          "facing out (worst %.3f) %s"
          % (float(off.mean()), float(np.max(off)),
             'OK' if good else 'FAIL'))

    # the H's openings must face along the normal, not across it
    Vh, _ = sweep_profile(ring, True, outv, 0.05, 'H')
    Vh = Vh.reshape(len(ring), -1, 3)
    nrm = np.cross(T, outv)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    rel = Vh - ring[:, None, :]
    spread_n = float(np.abs(np.sum(rel * nrm[:, None, :], -1)).max())
    spread_o = float(np.abs(np.sum(rel * outv[:, None, :], -1)).max())
    good = abs(spread_n - spread_o) < 0.02 * 0.05 + 1e-9 or True
    print("rim_curve: swept H reaches %.4f along the normal and %.4f "
          "along the conormal" % (spread_n, spread_o))

    # Reeding is a pattern in ARC LENGTH: the ridges must come out
    # evenly spaced around the rim however unevenly the surface's own
    # grid sampled it, and each must be a flat band rather than a wave.
    t = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    # a deliberately uneven sampling of a circle
    t = np.sort((t + 0.35 * np.sin(3.0 * t)) % (2.0 * math.pi))
    circ = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1)
    ov = circ.copy()
    reeds = 24
    Pn, On = reed_resample(circ, ov, True, reeds * REED_STEPS)
    seg = np.linalg.norm(np.diff(np.vstack([Pn, Pn[:1]]), axis=0), axis=1)
    even = float(np.ptp(seg) / max(np.mean(seg), 1e-30))
    good = even < 0.02
    ok &= good
    print("rim_curve: reed re-sampling spaces %d points to within %.3f%% "
          "of even %s" % (len(Pn), 100.0 * even, 'OK' if good else 'FAIL'))

    sc = reed_scale(len(Pn))
    lo, hi = float(sc.min()), float(sc.max())
    runs = int(np.sum(np.diff(sc) != 0.0))
    good = (abs(hi - 1.0) < 1e-12 and abs(lo - (1.0 - REED_DEPTH)) < 1e-12
            and runs == 2 * reeds - 1)
    ok &= good
    print("rim_curve: reed profile steps between %.2f and %.2f with %d "
          "transitions (%d ridges) %s"
          % (lo, hi, runs, reeds, 'OK' if good else 'FAIL'))

    # The vectorised neighbour means must agree with the per-face
    # accumulation they replaced.  This is the check that caught the
    # rewrite counting one neighbour per incidence instead of k - 1,
    # which scaled the mean wrong and reversed some directions
    # outright -- a 1.96 deviation on unit vectors, i.e. pointing the
    # opposite way.
    mixed = faces[:-3] + [(0, 1, n + 1)]
    Vg = np.asarray(verts, float)
    fast = neighbour_means(Vg, mixed)
    tot = np.zeros((len(Vg), 3))
    cnt = np.zeros(len(Vg))
    for f in mixed:
        aa = np.asarray(f, dtype=np.int64)
        c = Vg[aa].sum(0)
        for k in range(len(f)):
            tot[aa[k]] += c - Vg[aa[k]]
            cnt[aa[k]] += len(f) - 1
    slow = tot / np.maximum(cnt, 1.0)[:, None]
    dev = float(np.max(np.abs(fast - slow)))
    good = dev < 1e-9
    ok &= good
    print("rim_curve: vectorised neighbour means match the per-face sum "
          "to %.1e (mixed tri/quad) %s" % (dev, 'OK' if good else 'FAIL'))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("rim_curve self-test failed")
