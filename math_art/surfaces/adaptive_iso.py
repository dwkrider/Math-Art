# Adaptive iso-surface extraction for implicit algebraic surfaces.
#
# Part of the Math Art surfaces engine (`math_art/surfaces/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly.
#
# WHY THIS EXISTS.  The uniform marching-tetrahedra pass in
# `minsurf.tpms` spends ~96% of its samples on cells the level set never
# enters (measured on the Endrass octic family: 4.24% of cells carry a
# sign change), and it places vertices by LINEAR interpolation of a
# polynomial that is nowhere near linear over a cell -- for a degree-8
# field ranging over [-8620, +1] that lands vertices up to a full cell
# off the true surface, which reads as staircasing and tearing around
# the nodes.  This module fixes both:
#
#   1. SPARSE REFINEMENT.  Sample a base grid, keep only the cells that
#      either straddle the level set or whose corner values are within
#      about one cell diagonal of it (|F| / |grad F|, the gradient
#      estimated from the corner differences -- sign change alone would
#      drop cells that a thin sheet or a cone vertex crosses between
#      corners), dilate by one cell for safety, and subdivide.  Repeat
#      `depth` times.  Cost then scales with the surface AREA rather
#      than the sampled volume, so each extra doubling costs a few
#      times the base pass instead of 8 times.
#
#   2. EXACT VERTEX PLACEMENT.  Every crossing edge gets its vertex
#      from bisection of the actual field along the edge (plus one
#      guarded secant step) instead of linear interpolation.  Vertices
#      then satisfy F = 0 to ~1e-9 of an edge length, which is the
#      quantity that decides whether the mesh looks like the surface.
#
# NO CRACKS, BY CONSTRUCTION.  Every retained cell is refined to the
# SAME final depth, and all samples live on one global fine lattice
# addressed by integer indices; two cells sharing a face therefore
# share its samples bitwise, and welding on lattice-edge keys (the
# same scheme as `minsurf.tpms`) makes the result watertight wherever
# the level set stays inside the refined region.  A T-junction between
# cells of different depth -- the classic octree crack -- cannot occur
# because no two extracted cells differ in depth.  The price is that
# the neighbourhood of a node is refined no deeper than the rest of
# the surface; that is deliberate: a mixed-depth extraction would need
# crack patching or dual contouring, and an ordinary double point is a
# CONE, which no sampling scheme resolves at the apex anyway -- the
# honest target is a clean cone truncated at one fine cell, which
# uniform final depth already delivers.
#
# References:
# - A. Doi and A. Koide, "An efficient method of triangulating
#   equi-valued surfaces by using tetrahedral cells", IEICE Trans.
#   Inf. & Syst. E74-D (1991) 214-224 -- marching tetrahedra.
# - J. Bloomenthal, "Polygonization of implicit surfaces", Computer
#   Aided Geometric Design 5 (1988) 341-355 -- adaptive refinement
#   over implicit fields and the crack problem.
# - T. Ju, F. Losasso, S. Schaefer and J. Warren, "Dual contouring of
#   Hermite data", SIGGRAPH 2002 -- the alternative sharp-feature
#   extractor considered and (for now) not used; see the header note.

import numpy as np


# cube corners (i,j,k offsets) and a 6-tetrahedra decomposition sharing
# the 0-6 diagonal -- identical to minsurf.tpms, restated here so this
# module is self-contained.
_CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_TETS = [(0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6),
         (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6)]
_ONE = {1: (0, (1, 2, 3)), 2: (1, (0, 2, 3)),
        4: (2, (0, 1, 3)), 8: (3, (0, 1, 2)),
        14: (0, (1, 2, 3)), 13: (1, (0, 2, 3)),
        11: (2, (0, 1, 3)), 7: (3, (0, 1, 2))}
_TWO = {3: ((0, 1), (2, 3)), 5: ((0, 2), (1, 3)), 9: ((0, 3), (1, 2)),
        6: ((1, 2), (0, 3)), 10: ((1, 3), (0, 2)), 12: ((2, 3), (0, 1))}

_BIG = 1e30

# field samples per evaluation chunk, bounding peak coordinate-array
# memory the way minsurf.tpms's slab loop does
_CHUNK = 4_000_000


def _orientation_flags():
    """Whether each (tet, sign-case) emits triangles wound against the
    field gradient, calibrated on an exact linear field (as in
    minsurf.tpms -- the flags are combinatorial)."""
    flags = {}
    cube = np.array(_CUBE, dtype=float)
    for ti, tet in enumerate(_TETS):
        P = cube[list(tet)]
        M = P[1:] - P[0]
        for cd in list(_ONE) + list(_TWO):
            f = np.where([cd >> i & 1 for i in range(4)], -1.0, 1.0)
            g = np.linalg.solve(M, f[1:] - f[0])

            def x(ci, cj):
                t = f[ci] / (f[ci] - f[cj])
                return P[ci] + t * (P[cj] - P[ci])

            if cd in _ONE:
                lone, (o0, o1, o2) = _ONE[cd]
                p0, p1, p2 = x(lone, o0), x(lone, o1), x(lone, o2)
            else:
                (n0, n1), (q0, q1) = _TWO[cd]
                p0, p1, p2 = x(n0, q0), x(n0, q1), x(n1, q1)
            n = np.cross(p1 - p0, p2 - p0)
            flags[(ti, cd)] = float(np.dot(n, g)) < 0.0
    return flags


_ORIENT = None


def _uniq(a, inverse=False):
    """Sorted unique of an int64 array, sort-based.  numpy's np.unique
    routes big integer arrays through a hash table that measured ~6x
    slower than an argsort on the 10-100M-element key arrays this
    module produces."""
    if not inverse:
        s = np.sort(a, kind='stable')
        if not len(s):
            return s
        flag = np.empty(len(s), dtype=bool)
        flag[0] = True
        np.not_equal(s[1:], s[:-1], out=flag[1:])
        return s[flag]
    order = np.argsort(a, kind='stable')
    s = a[order]
    flag = np.empty(len(s), dtype=bool)
    flag[0] = True
    np.not_equal(s[1:], s[:-1], out=flag[1:])
    inv = np.empty(len(a), dtype=np.int64)
    inv[order] = np.cumsum(flag) - 1
    return s[flag], inv


class _Grid:
    """Geometry of the global fine lattice, plus safe field access."""

    def __init__(self, field, lo, hi, dims, nudge):
        self.field = field
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.dims = dims                        # fine CELLS per axis
        self.step = tuple(float((self.hi[i] - self.lo[i]) / dims[i])
                          for i in range(3))
        self.sy = dims[2] + 1
        self.sx = (dims[1] + 1) * self.sy
        self.npt = (dims[0] + 1) * self.sx
        self.nudge = float(nudge)

    def coords(self, ids):
        """Fine-lattice point ids -> coordinates.  One formula used
        everywhere in this module, so a point evaluated twice (from two
        neighbouring cells, or during placement) lands bitwise on the
        same coordinates and therefore the same value."""
        i, rem = np.divmod(ids, self.sx)
        j, k = np.divmod(rem, self.sy)
        return (self.lo[0] + i * self.step[0],
                self.lo[1] + j * self.step[1],
                self.lo[2] + k * self.step[2])

    def raw(self, x, y, z):
        """Field values with non-finite samples pushed to +-_BIG (NaN
        reads as outside), exactly as minsurf.tpms._sample does."""
        v = np.asarray(self.field(x, y, z), dtype=float)
        bad = ~np.isfinite(v)
        if bad.any():
            v = np.where(bad, np.where(v < 0.0, -_BIG, _BIG), v)
        return v

    def sampled(self, x, y, z):
        """`raw` plus the nudge: samples landing exactly on the surface
        give degenerate crossings, so they are displaced, as the
        uniform pass does."""
        v = self.raw(x, y, z)
        return np.where(np.abs(v) < self.nudge, self.nudge, v)

    def sampled_at(self, ids, out_dtype=float):
        out = np.empty(len(ids), dtype=out_dtype)
        for a in range(0, len(ids), _CHUNK):
            b = min(a + _CHUNK, len(ids))
            x, y, z = self.coords(ids[a:b])
            out[a:b] = self.sampled(x, y, z)
        return out


def _cell_ids(grid, ci, cj, ck, step):
    """Global point id of corner 0 of cells (ci,cj,ck) whose edge is
    `step` fine lattice units, plus the 8 corner offsets."""
    base = (ci * step) * grid.sx + (cj * step) * grid.sy + (ck * step)
    offs = [(oi * step) * grid.sx + (oj * step) * grid.sy + (ok * step)
            for (oi, oj, ok) in _CUBE]
    return base, offs


def _near_mask(V, hx, hy, hz, kappa):
    """Cells whose corners sit within ~kappa cell diagonals of the
    level set: min |F| < kappa * diag * |grad F|, the gradient bounded
    from the 4 parallel corner differences per axis.  This is what
    catches thin sheets and cone (node) neighbourhoods that pass
    between the corners without flipping a sign; it needs no floor --
    a zero gradient estimate reads as 'infinitely far', and a cell
    that flat genuinely carries no detectable surface."""
    amin = np.abs(V).min(axis=0)
    gx = np.maximum.reduce([np.abs(V[1] - V[0]), np.abs(V[2] - V[3]),
                            np.abs(V[5] - V[4]),
                            np.abs(V[6] - V[7])]) / hx
    gy = np.maximum.reduce([np.abs(V[3] - V[0]), np.abs(V[2] - V[1]),
                            np.abs(V[7] - V[4]),
                            np.abs(V[6] - V[5])]) / hy
    gz = np.maximum.reduce([np.abs(V[4] - V[0]), np.abs(V[5] - V[1]),
                            np.abs(V[6] - V[2]),
                            np.abs(V[7] - V[3])]) / hz
    g = np.sqrt(gx * gx + gy * gy + gz * gz)
    diag = np.sqrt(hx * hx + hy * hy + hz * hz)
    return amin < (kappa * diag) * g


def _level0(grid, res, kappa):
    """Active cells of the base grid, from a dense float32 sample pass
    (float32 is enough for a refinement DECISION -- rounding a nonzero
    float64 to float32 never flips its sign, and every sample is
    nudged away from zero -- and it halves the footprint of the one
    pass that still touches the whole volume)."""
    rx, ry, rz = res
    n = (rx + 1, ry + 1, rz + 1)
    V = np.empty(n, dtype=np.float32)
    # evaluate in z-plane slabs, bounded by _CHUNK samples.  Coordinates
    # always come from FINE lattice indices through grid.step -- the one
    # formula shared with _refine and _place -- so a point revisited at
    # any later stage lands bitwise on the same value.
    per_plane = n[0] * n[1]
    layers = max(1, _CHUNK // per_plane)
    sub = [grid.dims[i] // res[i] for i in range(3)]
    xs = grid.lo[0] + (np.arange(n[0]) * sub[0]) * grid.step[0]
    ys = grid.lo[1] + (np.arange(n[1]) * sub[1]) * grid.step[1]
    zs = grid.lo[2] + (np.arange(n[2]) * sub[2]) * grid.step[2]
    for k0 in range(0, n[2], layers):
        k1 = min(k0 + layers, n[2])
        X, Y, Z = np.meshgrid(xs, ys, zs[k0:k1], indexing='ij')
        V[:, :, k0:k1] = grid.sampled(X, Y, Z)
    neg = V < 0.0
    any_neg = np.zeros((rx, ry, rz), dtype=bool)
    all_neg = np.ones((rx, ry, rz), dtype=bool)
    corners = []
    for oi, oj, ok in _CUBE:
        c = neg[oi:oi + rx, oj:oj + ry, ok:ok + rz]
        any_neg |= c
        all_neg &= c
        corners.append(V[oi:oi + rx, oj:oj + ry, ok:ok + rz])
    act = any_neg & ~all_neg
    del neg, any_neg, all_neg
    hx = float(sub[0] * grid.step[0])
    hy = float(sub[1] * grid.step[1])
    hz = float(sub[2] * grid.step[2])
    act |= _near_mask(np.stack([c.astype(np.float64) for c in corners]),
                      hx, hy, hz, kappa)
    ci, cj, ck = np.nonzero(act)
    return ci.astype(np.int64), cj.astype(np.int64), ck.astype(np.int64)


def _dilate(ci, cj, ck, dims_l):
    """Grow the active cell set by one cell (face neighbours), clipped
    to the level's grid."""
    outs = [np.stack([ci, cj, ck])]
    for ax in range(3):
        for d in (-1, 1):
            s = [ci, cj, ck]
            s[ax] = s[ax] + d
            outs.append(np.stack(s))
    all_ = np.concatenate(outs, axis=1)
    ok = ((all_[0] >= 0) & (all_[0] < dims_l[0])
          & (all_[1] >= 0) & (all_[1] < dims_l[1])
          & (all_[2] >= 0) & (all_[2] < dims_l[2]))
    all_ = all_[:, ok]
    flat = _uniq((all_[0] * dims_l[1] + all_[1]) * dims_l[2] + all_[2])
    fi, rem = np.divmod(flat, dims_l[1] * dims_l[2])
    fj, fk = np.divmod(rem, dims_l[2])
    return fi, fj, fk


# per-parent 3x3x3 block: the 8 children, and for each its 8 corners as
# (a, b, c) indices into the block -- computed once at import
_CHILD = [(a, b, c) for (a, b, c) in _CUBE]
_CHILD_CORNER = [[(a + oi, b + oj, c + ok) for (oi, oj, ok) in _CUBE]
                 for (a, b, c) in _CHILD]


def _refine(grid, ci, cj, ck, lvl, depth, res, kappa, final):
    """One subdivision round: evaluate the 3x3x3 point block of every
    parent cell at the child level, decide each child's activity, and
    return the active children -- plus their corner values when this is
    the final level (they feed extraction directly, so no global value
    store and no searches are ever needed).

    Points shared between neighbouring parents are evaluated once per
    parent; the duplication (~2x on a surface-hugging set) measured
    cheaper than deduplicating tens of millions of ids per level, and
    it cannot cause seams: a shared point has one global id, hence one
    coordinate triple, hence bitwise one value."""
    sub = 1 << (depth - lvl - 1)     # child cell edge, fine units
    hx = float(sub * grid.step[0])
    hy = float(sub * grid.step[1])
    hz = float(sub * grid.step[2])
    out_i, out_j, out_k, out_v = [], [], [], []
    chunk = max(1, _CHUNK // 27)
    for a0 in range(0, len(ci), chunk):
        a1 = min(a0 + chunk, len(ci))
        pi = ci[a0:a1] * 2
        pj = cj[a0:a1] * 2
        pk = ck[a0:a1] * 2
        # block point coordinates, from FINE lattice indices through
        # grid.step -- bitwise the same as grid.coords() would give
        off = np.arange(3, dtype=np.int64)
        X = grid.lo[0] + ((pi[:, None] + off) * sub) * grid.step[0]
        Y = grid.lo[1] + ((pj[:, None] + off) * sub) * grid.step[1]
        Z = grid.lo[2] + ((pk[:, None] + off) * sub) * grid.step[2]
        B = grid.sampled(X[:, :, None, None],
                         Y[:, None, :, None],
                         Z[:, None, None, :])       # (n, 3, 3, 3)
        # a field that ignores a variable (Hauser's Pipe is x^2 - z)
        # comes back under-broadcast; expand it to the full block
        if B.shape != (len(pi), 3, 3, 3):
            B = np.broadcast_to(B, (len(pi), 3, 3, 3))
        for ch, corners in enumerate(_CHILD_CORNER):
            a, b, c = _CHILD[ch]
            V = np.stack([B[:, q, r, s] for (q, r, s) in corners])
            neg = V < 0.0
            cross = neg.any(axis=0) & ~neg.all(axis=0)
            act = cross if final else \
                (cross | _near_mask(V, hx, hy, hz, kappa))
            sel = np.nonzero(act)[0]
            if not len(sel):
                continue
            out_i.append(pi[sel] + a)
            out_j.append(pj[sel] + b)
            out_k.append(pk[sel] + c)
            if final:
                out_v.append(V[:, sel])
    if not out_i:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, np.zeros((8, 0))
    return (np.concatenate(out_i), np.concatenate(out_j),
            np.concatenate(out_k),
            np.concatenate(out_v, axis=1) if final else None)


def _extract(grid, ci, cj, ck, cvals):
    """Marching tetrahedra over an explicit list of fine cells with
    known corner values; returns (ntri, 3) global lattice-edge keys or
    None.  Same tables, same orientation calibration and the same
    edge-key weld as the uniform pass in minsurf.tpms."""
    global _ORIENT
    if _ORIENT is None:
        _ORIENT = _orientation_flags()
    base, offs = _cell_ids(grid, ci, cj, ck, 1)
    npt = grid.npt
    out = []
    for ti, (a, b, c, d) in enumerate(_TETS):
        fa, fb, fc, fd = cvals[a], cvals[b], cvals[c], cvals[d]
        code = ((fa < 0).astype(np.int8) | ((fb < 0) << 1)
                | ((fc < 0) << 2) | ((fd < 0) << 3))
        tet = (base + offs[a], base + offs[b],
               base + offs[c], base + offs[d])
        present = np.bincount(code.ravel(), minlength=16)

        def edge(sel, x, y):
            ia, ib = tet[x][sel], tet[y][sel]
            return np.minimum(ia, ib) * npt + np.maximum(ia, ib)

        for cd, (lone, others) in _ONE.items():
            if not present[cd]:
                continue
            sel = np.nonzero(code == cd)[0]
            p0 = edge(sel, lone, others[0])
            p1 = edge(sel, lone, others[1])
            p2 = edge(sel, lone, others[2])
            if _ORIENT[(ti, cd)]:
                p1, p2 = p2, p1
            out.append(np.stack([p0, p1, p2], axis=1))
        for cd, ((n0, n1), (pp0, pp1)) in _TWO.items():
            if not present[cd]:
                continue
            sel = np.nonzero(code == cd)[0]
            q0 = edge(sel, n0, pp0)
            q1 = edge(sel, n0, pp1)
            q2 = edge(sel, n1, pp1)
            q3 = edge(sel, n1, pp0)
            if _ORIENT[(ti, cd)]:
                q1, q3 = q3, q1
            out.append(np.stack([q0, q1, q2], axis=1))
            out.append(np.stack([q0, q2, q3], axis=1))
    return np.concatenate(out, axis=0) if out else None


def _place(grid, keys, polish, iters):
    """One vertex per unique lattice edge.  Linear interpolation of the
    corner samples when polish is off; otherwise bisection of the
    ACTUAL field along the edge -- `iters` halvings, then one secant
    step inside the final bracket -- which pins the vertex onto F = 0
    to ~1e-9 of an edge length however nonlinear the field is.
    Bisection rather than a pure secant scheme because the bracket is
    guaranteed (edge keys exist only where the corner signs differ)
    and halving is immune to the vanishing gradients around nodes."""
    mn, mx = np.divmod(keys, grid.npt)
    ax, ay, az = grid.coords(mn)
    bx, by, bz = grid.coords(mx)
    va = grid.sampled_at(mn)
    vb = grid.sampled_at(mx)
    if not polish:
        t = va / (va - vb)
        return np.stack([ax + t * (bx - ax), ay + t * (by - ay),
                         az + t * (bz - az)], axis=-1)
    sn = va < 0.0
    ta = np.zeros(len(keys))
    tb = np.ones(len(keys))
    fa, fb = va, vb
    for _ in range(int(iters)):
        t = 0.5 * (ta + tb)
        ft = grid.raw(ax + t * (bx - ax), ay + t * (by - ay),
                      az + t * (bz - az))
        same = (ft < 0.0) == sn
        ta = np.where(same, t, ta)
        fa = np.where(same, ft, fa)
        tb = np.where(same, tb, t)
        fb = np.where(same, fb, ft)
    # one secant step across the final bracket, clamped into it; where
    # the values degenerate (fa == fb around a node) keep the midpoint
    with np.errstate(divide='ignore', invalid='ignore'):
        ts = ta - fa * (tb - ta) / (fb - fa)
    t = np.where(np.isfinite(ts) & (ts >= ta) & (ts <= tb),
                 ts, 0.5 * (ta + tb))
    return np.stack([ax + t * (bx - ax), ay + t * (by - ay),
                     az + t * (bz - az)], axis=-1)


def contour(field, box_min, box_max, res, depth=2, kappa=1.0,
            polish=True, iters=14, nudge=1e-9):
    """Extract the zero level set of `field` over the box.

    `res` is the BASE sample grid (cells per axis, int or 3-tuple):
    surface detection happens at exactly this resolution, so anything
    a uniform `res` grid would find, this finds.  Cells near the level
    set are then refined `depth` times, giving an effective resolution
    of res * 2**depth at a cost proportional to the surface area.
    Returns (verts (n,3), tris (m,3)) with triangle winding oriented
    along the field gradient, watertight wherever the level set stays
    inside the refined region (see the module header on cracks).
    """
    if np.isscalar(res):
        res = (int(res),) * 3
    res = tuple(int(t) for t in res)
    depth = int(depth)
    # edge keys are packed as min*npt + max and must fit int64
    while depth > 0 and ((res[0] << depth) + 1) * ((res[1] << depth) + 1) \
            * ((res[2] << depth) + 1) > 3_000_000_000:
        depth -= 1
    dims = tuple(r << depth for r in res)
    grid = _Grid(field, box_min, box_max, dims, nudge)

    ci, cj, ck = _level0(grid, res, kappa)
    cvals = None
    for lvl in range(depth):
        dims_l = tuple(r << lvl for r in res)
        ci, cj, ck = _dilate(ci, cj, ck, dims_l)
        ci, cj, ck, cvals = _refine(grid, ci, cj, ck, lvl, depth, res,
                                    kappa, final=(lvl == depth - 1))

    if depth == 0:
        # no refinement pass ran, so extraction values come straight
        # from the corner points of the crossing base cells
        base, offs = _cell_ids(grid, ci, cj, ck, 1)
        cvals = np.stack([grid.sampled_at(base + o) for o in offs])
        neg = cvals < 0.0
        cross = neg.any(axis=0) & ~neg.all(axis=0)
        ci, cj, ck, cvals = ci[cross], cj[cross], ck[cross], \
            cvals[:, cross]

    if not len(ci):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    keys = _extract(grid, ci, cj, ck, cvals)
    if keys is None:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    uniq, inv = _uniq(keys.ravel(), inverse=True)
    tris = inv.reshape(-1, 3)
    verts = _place(grid, uniq, polish, iters)

    # cosmetic weld of near-coincident crossings + degenerate-triangle
    # drop, exactly as the uniform pass does (watertightness is already
    # guaranteed by the edge keys; this only collapses slivers)
    eps = max(float(np.max(grid.hi) - np.min(grid.lo)), 1.0) * 1e-6
    q = np.round(verts / eps).astype(np.int64)
    _, first, back = np.unique(q, axis=0, return_index=True,
                               return_inverse=True)
    verts = verts[first]
    tris = back.ravel()[tris]
    good = ((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
            & (tris[:, 0] != tris[:, 2]))
    return verts, tris[good]


def _selftest():
    ok = True

    # 1. a sphere: closed, exact-residual vertices, right area
    f = lambda X, Y, Z: X * X + Y * Y + Z * Z - 1.0
    V, T = contour(f, (-1.5,) * 3, (1.5,) * 3, 20, depth=2)
    r = np.abs(np.linalg.norm(V, axis=1) - 1.0)
    good = len(T) > 0 and float(r.max()) < 1e-7
    ok &= good
    print("adaptive_iso: sphere vertices on the surface (max |r-1| "
          "= %.2e) %s" % (float(r.max()), 'OK' if good else 'FAIL'))

    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]],
                                T[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    good = bool((cnt == 2).all())
    ok &= good
    print("adaptive_iso: sphere mesh is closed (every edge in 2 "
          "triangles) %s" % ('OK' if good else 'FAIL'))

    a_, b_, c_ = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b_ - a_, c_ - a_),
                                axis=1).sum()
    good = abs(area - 4.0 * np.pi) < 0.05 * 4.0 * np.pi
    ok &= good
    print("adaptive_iso: sphere area %.4f vs 4pi %s"
          % (area, 'OK' if good else 'FAIL'))

    # 2. the base grid alone (depth 0, no polish) agrees with the
    # uniform pass in minsurf.tpms: same tables, same welds
    try:
        from .. import minsurf as mst
    except ImportError:
        import minsurf as mst
    V0, T0 = contour(f, (-1.5,) * 3, (1.5,) * 3, 16, depth=0,
                     polish=False)
    V1, T1 = mst.marching_tets(f, (-1.5,) * 3, (1.5,) * 3, (16,) * 3)
    good = len(V0) == len(V1) and len(T0) == len(T1)
    if good:
        d = np.abs(np.sort(V0.ravel()) - np.sort(V1.ravel()))
        good = float(d.max()) < 1e-9
    ok &= good
    print("adaptive_iso: depth-0 extraction matches marching_tets "
          "(%d/%d verts, %d/%d tris) %s"
          % (len(V0), len(V1), len(T0), len(T1),
             'OK' if good else 'FAIL'))

    # 3. a cone (one ordinary double point at the origin): the mesh
    # must pinch to within one fine cell of the apex, and its only
    # open edges must lie on the clip box
    f = lambda X, Y, Z: X * X + Y * Y - Z * Z
    V, T = contour(f, (-1.2,) * 3, (1.2,) * 3, 20, depth=2)
    fine = 2.4 / (20 * 4)
    apex = float(np.linalg.norm(V, axis=1).min())
    good = apex < fine * np.sqrt(3.0)
    ok &= good
    print("adaptive_iso: cone pinches to %.4f of the apex (fine cell "
          "%.4f) %s" % (apex, fine, 'OK' if good else 'FAIL'))

    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]],
                                T[:, [2, 0]]]), axis=1)
    eu, cnt = np.unique(e, axis=0, return_counts=True)
    rim = eu[cnt == 1]
    if len(rim):
        # each rim endpoint must sit on a box face
        on_box = np.abs(np.abs(V[rim]).max(axis=2) - 1.2) < 1e-6
        good = bool(on_box.all())
    else:
        good = True
    ok &= good
    print("adaptive_iso: cone rim lies on the clip box (%d rim edges) "
          "%s" % (len(rim), 'OK' if good else 'FAIL'))

    if not ok:
        raise AssertionError("adaptive_iso self-test failed")
    print("RESULT: OK")
