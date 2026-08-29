# Shared mesh and domain utilities.
#
# Part of the Math Art minimal-surface engine (`math_art/minsurf/`), split
# out of the former single-file `minimal_surface_toolkit.py`.  Numpy only --
# no `bpy` -- so the whole engine imports and self-tests headlessly; the
# registered Blender operators stay in the flat `minimal_surface_toolkit.py`
# front-end.
#
# These are the shared mesh/domain utilities the rest of the engine needs:
# grid sampling on the torus, puncture masks for surface ends, connected-
# component selection, bounding-box normalization, and boundary-loop
# cleanup.  They live here rather than beside any one surface family
# because `weierstrass.py` (the Weierstrass-Enneper / Bjorling engine)
# needs them too -- reaching back into the operator module for them is what
# made the old toolkit <-> weierstrass import cycle.
#
# `_center_fit` implements the project-wide convention that a generator
# emits geometry centered on the origin and fitted to a 2 m cube.

import math
import numpy as np

TAU = 2.0 * math.pi


def _torus_grid(nu, nv):
    """Periodic (nu, nv) sample of the unit torus [0,1)^2 (endpoint-free)."""
    u = np.linspace(0.0, 1.0, nu, endpoint=False)
    v = np.linspace(0.0, 1.0, nv, endpoint=False)
    return np.meshgrid(u, v, indexing='ij')


def _puncture_mask(U, V, centers):
    """Valid where the toroidal distance to every puncture exceeds its
    radius (so all lattice translates of each end are excluded at once).
    `centers` is a list of (cu, cv, rho). Ends whose parametrization
    stretches fastest (planar, Enneper) want a larger rho so the rim sits
    where grid cells are still small -> a cleaner circular rim."""
    valid = np.ones(U.shape, dtype=bool)
    for cu, cv, rho in centers:
        du = np.abs(((U - cu + 0.5) % 1.0) - 0.5)
        dv = np.abs(((V - cv + 0.5) % 1.0) - 0.5)
        valid &= (du * du + dv * dv) > rho * rho
    return valid

def _largest_component(V, quads):
    """Keep only the face-connected component with the most faces."""
    parent = list(range(len(V)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for f in quads:
        for i in range(1, len(f)):
            union(f[0], f[i])
    from collections import Counter
    sizes = Counter(find(f[0]) for f in quads)
    if len(sizes) <= 1:
        return V, quads
    keep_root = sizes.most_common(1)[0][0]
    quads = [f for f in quads if find(f[0]) == keep_root]
    used = np.unique(np.array([i for f in quads for i in f], dtype=np.int64))
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], [tuple(int(remap[i]) for i in f) for f in quads]

def _center_fit(pts, scale, ref=None):
    """Center on the bounding-box midpoint and scale so the largest extent
    is 2.0 units (a 2 m cube), then apply `scale`. `ref` (a subset) fixes
    the box, so runaway ends don't shrink the body."""
    pts = np.asarray(pts, dtype=float)
    ref = pts if ref is None else np.asarray(ref, dtype=float)
    if len(ref) == 0:
        return pts
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    cen = 0.5 * (lo + hi)
    ext = float(np.max(hi - lo))
    s = (2.0 / ext if ext > 1e-9 else 1.0) * scale
    return (pts - cen) * s


def _inliers(pts):
    """Points within the 90th distance percentile of the median -- the
    body of a surface with ends running to infinity."""
    c = np.median(pts, axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    keep = d <= np.percentile(d, 90.0)
    return pts[keep] if keep.any() else pts

def _smooth_boundary(V, quads, iters=10, lam=0.5):
    """Relax open mesh-boundary loops in place, removing the grid
    staircase left on an end-rim cut from an axis-aligned grid, without
    disturbing interior vertices.

    Each iteration is a Taubin lambda / -lambda pair: a Laplacian step
    toward the two boundary neighbours' midpoint, then the same step
    reversed.  Per pair the gain on a loop mode of frequency w = 1-cos(omega)
    is (1 - lam*w)(1 + lam*w) = 1 - lam^2 w^2: the alternating staircase
    zigzag (the Nyquist mode, w = 2, gain 0 at lam = 0.5) is annihilated
    exactly as a plain Laplacian would, but the first-order SHRINK term of
    the Laplacian cancels.  That shrink is not cosmetic: on a smoothly
    curved rim every neighbour-chord midpoint lies on the concave side, so
    a plain Laplacian translates the whole loop inward by the chord sagitta
    (~ curvature * spacing^2 / 2) per step.  With rim-graded radial
    sampling (Enneper et al.) that pull exceeds the last-ring gap, dragging
    the boundary ring through its neighbour and folding the outermost face
    ring inside out -- one full ring of inverted normals, seen as a thin
    doubled "lip" along the rim.  The pair keeps smooth loops in place to
    second order, so an analytically sampled rim is (correctly) a near
    no-op while a clipped staircase rim still comes out clean."""
    if not quads:
        return V
    from collections import defaultdict
    count = defaultdict(int)
    for q in quads:
        for k in range(len(q)):
            a, b = q[k], q[(k + 1) % len(q)]
            count[(a, b) if a < b else (b, a)] += 1
    nbr = defaultdict(list)
    bnd = set()
    for (a, b), c in count.items():
        if c == 1:                       # boundary edge
            nbr[a].append(b)
            nbr[b].append(a)
            bnd.add(a)
            bnd.add(b)
    # keep only vertices with exactly two boundary neighbours (clean loops)
    loop = [v for v in bnd if len(nbr[v]) == 2]
    if not loop:
        return V
    V = V.copy()
    idx = np.array(loop)
    n0 = np.array([nbr[v][0] for v in loop])
    n1 = np.array([nbr[v][1] for v in loop])
    for _ in range(iters):
        target = 0.5 * (V[n0] + V[n1])
        V[idx] += lam * (target - V[idx])       # smoothing step
        target = 0.5 * (V[n0] + V[n1])
        V[idx] -= lam * (target - V[idx])       # anti-shrink step
    return V


def _circularize_outer(V, quads, min_len=8):
    """Snap each open boundary loop -- the planar end and the two catenoid
    ends of a Costa / Costa-Hoffman-Meeks surface -- to a clean circle
    about the vertical axis (constant XY radius, z kept), so every rim
    reads as a circle instead of the few-percent staircase wobble the
    radial end clip leaves behind.  Interior vertices are untouched."""
    if not quads:
        return V
    from collections import defaultdict
    cnt = defaultdict(int)
    nbr = defaultdict(list)
    for q in quads:
        L = len(q)
        for k in range(L):
            a, b = q[k], q[(k + 1) % L]
            cnt[(a, b) if a < b else (b, a)] += 1
    for (a, b), c in cnt.items():
        if c == 1:
            nbr[a].append(b)
            nbr[b].append(a)
    bnd = [v for v in nbr if len(nbr[v]) == 2]     # clean-loop vertices
    if not bnd:
        return V
    seen = set()
    loops = []
    for v in bnd:
        if v in seen:
            continue
        comp, stack = [], [v]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for w in nbr[u]:
                if w not in seen:
                    stack.append(w)
        loops.append(comp)
    V = V.copy()
    for comp in loops:
        if len(comp) < min_len:
            continue                                # skip stray fragments
        idx = np.array(comp)
        rmean = float(np.hypot(V[idx, 0], V[idx, 1]).mean())
        ang = np.arctan2(V[idx, 1], V[idx, 0])
        V[idx, 0] = rmean * np.cos(ang)
        V[idx, 1] = rmean * np.sin(ang)
        V[idx, 2] = float(V[idx, 2].mean())         # flat horizontal circle
    return V


# --------------------------------------------------------------------
# Equal-area domain resampling
# --------------------------------------------------------------------
#
# A parametric surface is sampled on a uniform (u, v) grid, but a
# uniform step in the DOMAIN is not a uniform step on the SURFACE: the
# area element |X_u x X_v| varies, so cells bunch up where the map
# contracts and stretch where it flares.  On an Enneper disk the outer
# cells are orders of magnitude larger than the inner ones.
#
# For a minimal surface this is purely a SIZE problem, not a shape one.
# The Weierstrass-Enneper representation is conformal (isothermal): the
# metric is lambda^2 (du^2 + dv^2) with a single scalar
#
#     lambda = |f| (1 + |g|^2) / 2,
#
# so the map preserves angles and scales both directions by the SAME
# factor.  Grid cells therefore come out the right SHAPE -- near-square
# -- and only the wrong SIZE.  (Measured: a Costa grid scores 0.778 on
# the isoperimetric quotient against 0.785 for a perfect square, while
# its area varies with a coefficient of variation of 0.67.)  Fixing the
# sizes is then a one-scalar problem rather than a full remesh.
#
# The fix here is inverse-transform sampling of the area density, done
# in two stages so the areas come out equal rather than merely better:
#
#   1. integrate the area element over v to get a marginal density in
#      u, and place the nu grid lines at equal quantiles of it;
#   2. for each of those u lines, take the CONDITIONAL density along v
#      and place that row's nv samples at equal quantiles of it.
#
# Stage 2 is what makes it exact.  A separable (tensor-product) grid
# can only equalize areas when the density factors as a(u)b(v); the
# conditional stage lets each row carry its own v samples, which costs
# nothing -- the grid keeps its (i, j) topology and every quad is still
# a quad -- and buys exact equality for any density.  Rows shear
# smoothly against each other, the way an equal-area map projection
# does.
#
# Sampling happens on a deliberately over-resolved grid and the result
# is interpolated with a Catmull-Rom spline, whose O(h^4) error is far
# below the resampling's own discretization error.
#
# References:
# - Karl Weierstrass (1866) and Alfred Enneper (1864), the conformal
#   representation of a minimal surface that makes lambda a single
#   scalar; see e.g. Robert Osserman, "A Survey of Minimal Surfaces",
#   Dover (1986), chapter 8.
# - The two-stage marginal/conditional construction is standard inverse
#   transform sampling; see Luc Devroye, "Non-Uniform Random Variate
#   Generation", Springer (1986), chapter 11.


def _catmull_rom(arr, t):
    """Sample `arr` (N, ...) at fractional positions `t` along axis 0."""
    n = arr.shape[0]
    if n < 2:
        return np.repeat(arr[:1], len(t), axis=0)
    i = np.clip(np.floor(t).astype(np.int64), 0, n - 2)
    f = (np.asarray(t, dtype=float) - i)
    f = f.reshape((-1,) + (1,) * (arr.ndim - 1))
    p0 = arr[np.clip(i - 1, 0, n - 1)]
    p1 = arr[i]
    p2 = arr[np.clip(i + 1, 0, n - 1)]
    p3 = arr[np.clip(i + 2, 0, n - 1)]
    return (((-0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3) * f
             + (p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3)) * f
            + (-0.5 * p0 + 0.5 * p2)) * f + p1


def _cell_areas(G):
    """Area of every cell of a (NU, NV, 3) grid, as (NU-1, NV-1)."""
    p00, p10 = G[:-1, :-1], G[1:, :-1]
    p01, p11 = G[:-1, 1:], G[1:, 1:]
    a1 = np.linalg.norm(np.cross(p10 - p00, p01 - p00), axis=-1)
    a2 = np.linalg.norm(np.cross(p10 - p11, p01 - p11), axis=-1)
    return 0.5 * (a1 + a2)


def _quantile_nodes(w, n):
    """Fractional node positions splitting weights `w` into n-1 equal
    parts. `w[k]` is the weight of the interval between nodes k and k+1,
    so the result spans [0, len(w)] and is monotone."""
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    m = len(w)
    if m == 0:
        return np.zeros(n)
    c = np.concatenate([[0.0], np.cumsum(w)])
    if c[-1] <= 0.0:                      # degenerate: nothing to equalize
        return np.linspace(0.0, float(m), n)
    # nudge to strictly increasing so np.interp cannot stall on a plateau
    # of zero-area cells (a punctured end) and collapse samples onto it
    c = c + np.arange(m + 1) * (c[-1] * 1e-12)
    return np.interp(np.linspace(0.0, c[-1], n), c, np.arange(m + 1.0))


def area_cov(G):
    """Coefficient of variation (stddev/mean) of the grid's cell areas.
    0 means every cell has exactly the same area."""
    a = _cell_areas(G).ravel()
    a = a[np.isfinite(a) & (a > 0.0)]
    if len(a) < 2:
        return 0.0
    return float(a.std() / a.mean())


def _fill_nonfinite(G, bad):
    """Replace non-finite grid points by the nearest finite neighbour,
    along v and then along u.

    Several surfaces (Costa, Chen-Gackstatter) put a pole inside the
    parameter domain, so the raw grid genuinely contains infinities.
    Those points always sit inside a puncture that gets clipped away, so
    their VALUES never reach the mesh -- but they have to be finite
    anyway, because the interpolation spline reaches two samples to
    either side and would otherwise smear NaN into cells that do
    survive."""
    for axis in (1, 0):
        n = G.shape[axis]
        shape = [1, 1]
        shape[axis] = n
        ar = np.arange(n).reshape(shape)
        good = ~bad
        fwd = np.maximum.accumulate(np.where(good, ar, -1), axis=axis)
        bwd = np.flip(np.minimum.accumulate(
            np.flip(np.where(good, ar, n), axis=axis), axis=axis), axis=axis)
        take = np.clip(np.where(fwd >= 0, fwd, bwd), 0, n - 1)
        if axis == 1:
            near = G[np.arange(G.shape[0])[:, None], take]
        else:
            near = G[take, np.arange(G.shape[1])[None, :]]
        G = np.where(bad[..., None], near, G)
        bad = ~np.isfinite(G).all(axis=-1)
        if not bad.any():
            break
    return np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)


def equal_area_resample(G, nu, nv, mask=None):
    """Resample a fine (NU, NV, 3) grid to (nu, nv, 3) with equal cell
    areas, by inverse-transform sampling of the area density.

    Returns (G2, mask2, cov_before, cov_after).  G2 is ALWAYS (nu, nv, 3)
    -- a caller that asked for a resolution gets it even when the density
    turns out to be degenerate, in which case the fallback is a plain
    uniform subsample rather than the oversampled input.

    Cells that will not survive to the mesh carry no weight: a cell is
    excluded from the density if any corner is non-finite (a pole in the
    domain) or if `mask` clips it away.  Without that, the handful of
    enormous cells on a surface's runaway ends dominate the integral, and
    equalizing chases the infinities while starving the body -- measured
    on Costa, that made the area spread four times WORSE than leaving it
    alone.
    """
    G = np.asarray(G, dtype=float)
    if G.ndim != 3 or G.shape[0] < 4 or G.shape[1] < 4:
        return G, mask, area_cov(G), area_cov(G)

    before = area_cov(G)
    bad = ~np.isfinite(G).all(axis=-1)
    if bad.any():
        G = _fill_nonfinite(G, bad)

    A = _cell_areas(G)                                   # (NU-1, NV-1)

    # a cell counts only if all four of its corners are usable
    live = np.ones(A.shape, dtype=bool)
    for corner in (bad[:-1, :-1], bad[1:, :-1], bad[:-1, 1:], bad[1:, 1:]):
        live &= ~corner
    if isinstance(mask, np.ndarray) and mask.shape == bad.shape:
        for corner in (mask[:-1, :-1], mask[1:, :-1],
                       mask[:-1, 1:], mask[1:, 1:]):
            live &= corner
    if live.any():
        A = np.where(live, A, 0.0)

    ui = _quantile_nodes(A.sum(axis=1), nu)              # (nu,)
    rows = _catmull_rom(G, ui)                           # (nu, NV, 3)

    out = np.empty((nu, nv, 3), dtype=float)
    vjs = np.empty((nu, nv), dtype=float)
    last = A.shape[0] - 1
    for k, p in enumerate(ui):
        lo = int(np.clip(np.floor(p), 0, last))
        hi = min(lo + 1, last)
        fr = float(p - lo)
        dens = (1.0 - fr) * A[lo] + fr * A[hi]           # (NV-1,)
        vj = _quantile_nodes(dens, nv)
        vjs[k] = vj
        out[k] = _catmull_rom(rows[k], vj)

    mask2 = mask
    if isinstance(mask, np.ndarray) and mask.shape == G.shape[:2]:
        # a clip mask is categorical -- resample it nearest, never blended
        iu = np.clip(np.round(ui).astype(np.int64), 0, mask.shape[0] - 1)
        iv = np.clip(np.round(vjs).astype(np.int64), 0, mask.shape[1] - 1)
        mask2 = mask[iu[:, None], iv]

    # Report over the cells that actually survive to the mesh.  Measuring
    # the clipped-away runaway ends too would drown the number the user
    # cares about in the very cells the clip exists to remove.
    return out, mask2, _cov_kept(G, mask), _cov_kept(out, mask2)


def mesh_area_cov(V, faces):
    """Area CoV over a finished (V, faces) mesh.

    The grid CoV is the right measure of the resampling itself, but it is
    NOT what ships: rim smoothing, circularization and welding all run
    afterwards and move boundary vertices.  Equalizing thins the boundary
    bands, which makes those steps bite harder, so a grid that equalized
    beautifully can still deliver a worse mesh.  Judge the mesh."""
    V = np.asarray(V, dtype=float)
    if len(V) == 0 or not faces:
        return 0.0
    areas = []
    for f in faces:
        p = V[list(f)]
        if len(p) < 3:
            continue
        a = 0.0
        for i in range(1, len(p) - 1):
            a += 0.5 * float(np.linalg.norm(
                np.cross(p[i] - p[0], p[i + 1] - p[0])))
        if np.isfinite(a) and a > 1e-14:
            areas.append(a)
    if len(areas) < 2:
        return 0.0
    areas = np.asarray(areas)
    return float(areas.std() / areas.mean())


def _cov_kept(G, mask):
    """Area CoV over cells whose four corners are finite and kept."""
    a = _cell_areas(G)
    keep = np.isfinite(a) & (a > 0.0)
    fin = ~np.isfinite(G).all(axis=-1)
    for c in (fin[:-1, :-1], fin[1:, :-1], fin[:-1, 1:], fin[1:, 1:]):
        keep &= ~c
    if isinstance(mask, np.ndarray) and mask.shape == G.shape[:2]:
        for c in (mask[:-1, :-1], mask[1:, :-1], mask[:-1, 1:], mask[1:, 1:]):
            keep &= c
    a = a[keep]
    if len(a) < 2:
        return 0.0
    return float(a.std() / a.mean())


def _selftest():
    ok = True

    # _center_fit implements the project convention: centered on the origin,
    # largest extent exactly 2.0 (a 2 m cube), times `scale`.
    pts = np.array([[3.0, 3.0, 3.0], [7.0, 5.0, 4.0], [5.0, 4.0, 3.5]])
    out = _center_fit(pts, 1.0)
    lo, hi = out.min(axis=0), out.max(axis=0)
    cen = float(np.max(np.abs(0.5 * (lo + hi))))
    ext = float(np.max(hi - lo))
    good = cen < 1e-12 and abs(ext - 2.0) < 1e-12
    ok &= good
    print(f"domain: center_fit max|c|={cen:.2e} ext={ext:.6f} "
          f"{'OK' if good else 'FAIL'}")

    # `scale` multiplies after the fit, and `ref` (a subset) fixes the box so
    # runaway ends cannot shrink the body.
    out2 = _center_fit(pts, 2.5)
    ext2 = float(np.max(out2.max(axis=0) - out2.min(axis=0)))
    far = np.vstack([pts, [[100.0, 0.0, 0.0]]])
    out3 = _center_fit(far, 1.0, ref=pts)
    ext3 = float(np.max(out3[:3].max(axis=0) - out3[:3].min(axis=0)))
    good = abs(ext2 - 5.0) < 1e-12 and abs(ext3 - 2.0) < 1e-12
    ok &= good
    print(f"domain: scale ext={ext2:.4f} (exp 5) ref-fixed body ext={ext3:.4f} "
          f"(exp 2) {'OK' if good else 'FAIL'}")

    # _torus_grid must be endpoint-free, so the periodic wrap has no seam.
    U, V = _torus_grid(8, 5)
    good = (U.shape == (8, 5) and abs(float(U.max()) - 7.0 / 8.0) < 1e-12
            and abs(float(V.max()) - 4.0 / 5.0) < 1e-12)
    ok &= good
    print(f"domain: torus_grid shape={U.shape} u_max={U.max():.4f} "
          f"v_max={V.max():.4f} {'OK' if good else 'FAIL'}")

    # _puncture_mask excludes every lattice translate of a puncture: a hole at
    # (0, 0) must also bite the far corner (0.99, 0.99), which is near (1, 1).
    U, V = _torus_grid(100, 100)
    m = _puncture_mask(U, V, [(0.0, 0.0, 0.1)])
    good = (not m[0, 0]) and (not m[99, 99]) and bool(m[50, 50])
    ok &= good
    print(f"domain: puncture_mask wraps corners "
          f"kept={int(m.sum())}/10000 {'OK' if good else 'FAIL'}")

    # _largest_component keeps the bigger island and reindexes it compactly.
    Vv = np.zeros((10, 3))
    quads = [(0, 1, 2, 3), (1, 2, 4, 5), (6, 7, 8, 9)]     # 2 faces + 1 face
    V2, q2 = _largest_component(Vv, quads)
    good = len(q2) == 2 and len(V2) == 6 and max(max(f) for f in q2) == 5
    ok &= good
    print(f"domain: largest_component faces={len(q2)} verts={len(V2)} "
          f"{'OK' if good else 'FAIL'}")

    # _smooth_boundary relaxes only the boundary loop and leaves the interior
    # alone; a square ring with one pushed-out vertex must come back in.
    n = 12
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=1)
    Vr = np.vstack([ring, [[0.0, 0.0, 0.0]]])
    fan = [(i, (i + 1) % n, n) for i in range(n)]
    Vr[0] *= 1.5                                    # kick one rim vertex out
    before = float(np.linalg.norm(Vr[0][:2]))
    Vs = _smooth_boundary(Vr.copy(), fan, iters=20, lam=0.5)
    after = float(np.linalg.norm(Vs[0][:2]))
    moved_interior = float(np.linalg.norm(Vs[n] - Vr[n]))
    good = after < before and moved_interior < 1e-12
    ok &= good
    print(f"domain: smooth_boundary r {before:.4f} -> {after:.4f}, "
          f"interior moved {moved_interior:.1e} {'OK' if good else 'FAIL'}")

    # _circularize_outer snaps a wobbly rim to an exact circle at constant z.
    Vr2 = np.vstack([ring, [[0.0, 0.0, 0.0]]])
    Vr2[:n, 0] *= 1.0 + 0.05 * np.cos(3 * ang)      # 3-fold wobble
    Vc = _circularize_outer(Vr2.copy(), fan, min_len=8)
    r = np.hypot(Vc[:n, 0], Vc[:n, 1])
    good = float(np.ptp(r)) < 1e-9 and float(np.ptp(Vc[:n, 2])) < 1e-12
    ok &= good
    print(f"domain: circularize rim radius spread={np.ptp(r):.2e} "
          f"{'OK' if good else 'FAIL'}")

    # --- equal-area resampling ---------------------------------------

    # Catmull-Rom must reproduce the samples it interpolates exactly at
    # integer positions, and stay on a straight line for collinear data.
    line = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], axis=1)
    at_nodes = _catmull_rom(line, np.arange(10.0))
    mid = _catmull_rom(line, np.array([3.5]))[0]
    good = (float(np.abs(at_nodes - line).max()) < 1e-12
            and abs(float(mid[0]) - 3.5) < 1e-12)
    ok &= good
    print(f"domain: catmull_rom nodes exact, mid={mid[0]:.4f} (exp 3.5) "
          f"{'OK' if good else 'FAIL'}")

    # A flat sheet stretched exponentially in u: cell areas span e^3 ~ 20x.
    # Equalizing must drive the coefficient of variation to ~0, and the
    # samples must follow the analytic answer.  Equal area under
    # x = e^u means equal steps in x, i.e. u_k = log(linear in e^u).
    NU = NV = 200
    u = np.linspace(0.0, 3.0, NU)
    v = np.linspace(0.0, 1.0, NV)
    UU, VV = np.meshgrid(u, v, indexing='ij')
    sheet = np.stack([np.exp(UU), VV, np.zeros_like(UU)], axis=-1)
    G2, _m, before, after = equal_area_resample(sheet, 40, 40)
    want = np.log(np.linspace(math.exp(0.0), math.exp(3.0), 40))
    err = float(np.abs(G2[:, 0, 0] - np.exp(want)).max())
    good = before > 0.5 and after < 0.02 and err < 1e-3
    ok &= good
    print(f"domain: equal_area exp-sheet CoV {before:.3f} -> {after:.4f}, "
          f"max dev from analytic {err:.2e} {'OK' if good else 'FAIL'}")

    # Non-separable density: a(u)*b(v) factors, but this one does not.
    # The conditional (per-row) stage is what handles it -- a purely
    # separable scheme would leave a large residual here.
    u = np.linspace(0.5, 2.0, NU)
    v = np.linspace(0.5, 2.0, NV)
    UU, VV = np.meshgrid(u, v, indexing='ij')
    warp = np.stack([UU, VV * (1.0 + UU), np.zeros_like(UU)], axis=-1)
    _G3, _m, before, after = equal_area_resample(warp, 40, 40)
    good = after < 0.05 and after < before / 5.0
    ok &= good
    print(f"domain: equal_area non-separable CoV {before:.3f} -> "
          f"{after:.4f} {'OK' if good else 'FAIL'}")

    # An already-uniform grid must survive untouched (idempotence).
    flat = np.stack([UU, VV, np.zeros_like(UU)], axis=-1)
    _G4, _m, b4, a4 = equal_area_resample(flat, 40, 40)
    good = b4 < 1e-9 and a4 < 1e-6
    ok &= good
    print(f"domain: equal_area uniform grid stays uniform "
          f"{b4:.1e} -> {a4:.1e} {'OK' if good else 'FAIL'}")

    # A pole in the domain must not spread NaN into the surviving cells:
    # the result must be the requested size and entirely finite.
    bad = sheet.copy()
    bad[5, 5, 0] = np.inf
    bad[80:90, 80:90, 1] = np.nan
    G5, _m, _b5, a5 = equal_area_resample(bad, 40, 40)
    good = (G5.shape == (40, 40, 3) and bool(np.isfinite(G5).all())
            and a5 < 0.05)
    ok &= good
    print(f"domain: equal_area survives poles in the domain, finite="
          f"{bool(np.isfinite(G5).all())} CoV={a5:.4f} "
          f"{'OK' if good else 'FAIL'}")

    # The output is the REQUESTED size even when the density degenerates
    # to nothing -- never the oversampled input passed through.
    dead = np.zeros((120, 120, 3))
    G7, _m, _b, _a = equal_area_resample(dead, 40, 30)
    good = G7.shape == (40, 30, 3)
    ok &= good
    print(f"domain: equal_area honours the requested size on a degenerate "
          f"grid {G7.shape} {'OK' if good else 'FAIL'}")

    # A clip mask must steer the density: cells outside it carry no
    # weight, so a runaway end cannot drag every sample out to infinity.
    blow = sheet.copy()
    blow[-3:, :, 0] *= 5000.0                      # a runaway end
    keep = np.ones(blow.shape[:2], dtype=bool)
    keep[-3:, :] = False                           # ...that gets clipped
    _G8, _m8, _b8, a8 = equal_area_resample(blow, 40, 40, mask=keep)
    _G9, _m9, _b9, a9 = equal_area_resample(blow, 40, 40)
    good = a8 < a9
    ok &= good
    print(f"domain: equal_area ignores clipped runaway cells "
          f"masked CoV {a8:.4f} < unmasked {a9:.4f} "
          f"{'OK' if good else 'FAIL'}")

    # A clip mask must be carried across as booleans, never blended.
    m0 = np.zeros(sheet.shape[:2], dtype=bool)
    m0[NU // 2:, :] = True
    _G6, m6, _b, _a = equal_area_resample(sheet, 40, 40, mask=m0)
    good = (m6 is not None and m6.dtype == bool and m6.shape == (40, 40)
            and m6.any() and not m6.all())
    ok &= good
    print(f"domain: equal_area carries a clip mask as bool "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("domain self-test failed")
