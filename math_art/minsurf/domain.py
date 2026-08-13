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
    """Relax open mesh-boundary loops in place: each boundary vertex is
    averaged toward its two boundary neighbours. Removes the grid
    staircase left on an end-rim cut from an axis-aligned grid, without
    disturbing interior vertices."""
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
        V[idx] += lam * (target - V[idx])
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

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("domain self-test failed")
