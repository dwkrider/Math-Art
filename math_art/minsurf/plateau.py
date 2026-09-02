# Plateau solver: discrete area minimization.
#
# Part of the Math Art minimal-surface engine (`math_art/minsurf/`), split
# out of the former single-file `minimal_surface_toolkit.py`.  Numpy only --
# no `bpy` -- so the whole engine imports and self-tests headlessly; the
# registered Blender operators stay in the flat `minimal_surface_toolkit.py`
# front-end.
#
# A Plateau-problem solver: pin one or two boundary curves and minimize
# area by cotangent-Laplacian iteration solved with conjugate gradients --
# a lightweight take on what Brakke's Surface Evolver does.  Includes the
# "minimal surface spanning a circle and a torus knot" construction.
#
# References:
#   J. Plateau, "Statique experimentale et theorique des liquides" (1873).
#   U. Pinkall and K. Polthier, "Computing discrete minimal surfaces and
#       their conjugates", Experimental Mathematics 2(1) (1993).
#   K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#       (1992).
#   H. Seifert, "Ueber das Geschlecht von Knoten", Math. Annalen 110
#       (1934); visualization after J. J. van Wijk and A. M. Cohen, IEEE
#       TVCG 12(4) (2006).

import math
import re
from collections import defaultdict

import numpy as np

try:
    from ..solver import cotan as _sc
    from ..solver import descent as _sd
    from ..solver import collide as _scol
except ImportError:                      # flat (path-based) headless import
    from solver import cotan as _sc
    from solver import descent as _sd
    from solver import collide as _scol

TAU = 2.0 * math.pi
SQRT3 = math.sqrt(3.0)


# ==========================================================================
# 3. Plateau solver (area minimization with pinned boundaries)
# ==========================================================================

def resample_loop(pts, m):
    """Uniform-arclength resample of a closed polyline (k,3) -> (m,3)."""
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    closed = np.vstack([pts, pts[:1]])
    t = np.linspace(0.0, total, m, endpoint=False)
    out = np.empty((m, 3))
    j = 0
    for i, ti in enumerate(t):
        while s[j + 1] < ti:
            j += 1
        f = (ti - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out[i] = closed[j] * (1 - f) + closed[j + 1] * f
    return out


def resample_indexed(poly, m):
    """`resample_loop` that also reports where each sample came from.

    Returns `(pts, arc, corner)`: `arc[i]` is the polyline segment
    sample `i` lies on and `corner[k]` is the sample nearest corner `k`.
    The plain resampler throws both away, and they are exactly what a
    datafile's `frame` command asks for -- it names its planes by
    ORIGINAL EDGE and reads distances off ORIGINAL VERTICES, neither of
    which survives a uniform-arclength resample unless carried along.
    """
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    seg = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0.0:
        raise ValueError("degenerate contour")
    t = np.linspace(0.0, total, m, endpoint=False)
    idx = np.clip(np.searchsorted(s, t, side='right') - 1, 0, n - 1)
    f = (t - s[idx]) / np.maximum(seg[idx], 1e-12)
    closed = np.vstack([poly, poly[:1]])
    pts = closed[idx] * (1.0 - f)[:, None] + closed[idx + 1] * f[:, None]
    corner = (np.round(s[:n] / total * m).astype(np.int64)) % m
    return pts, idx, corner


def align_loops(A, B):
    """Cyclic shift + optional reversal of B minimizing sum |A_i - B_i|^2."""
    m = len(A)
    best = (None, 1e30)
    for Bc in (B, B[::-1]):
        # brute-force cyclic shifts (m <= a few hundred)
        for sft in range(m):
            d = np.sum((A - np.roll(Bc, -sft, axis=0)) ** 2)
            if d < best[1]:
                best = (np.roll(Bc, -sft, axis=0), d)
    return best[0]


def _cotan_weights(V, T, mode="clamp"):
    """Per-edge cotangent weights. Returns (edges (e,2), w (e,)).

    mode="clamp" (default) clips the cotangents to [0.01, 20] -- the
    historical behaviour: positive weights keep the maximum principle,
    so the solve cannot fold a degenerate fan, at the cost of biasing
    every obtuse corner toward the uniform-weight (Tutte) solution.
    mode="mollify" uses the true cotangents built from intrinsically
    mollified edge lengths (Sharp-Crane; see solver/cotan.py): exact
    Pinkall-Polthier away from degenerate triangles, finite on them."""
    return _sc.edge_cotan_weights(V, T, mode=mode)


def minimize_area(V, T, fixed, outer_iters=30, cg_tol=1e-8, cg_iters=400,
                  uniform=False, cotan_mode="mollify", groom_every=0,
                  groom_smooth=0.25):
    """Pinkall-Polthier: repeatedly solve the cotan-Laplace equation for
    the interior vertices (boundary pinned). V modified in place.
    With uniform=True, unit weights are used instead of cotangents --
    a Tutte-style fairing solve that untangles folded regions (at the
    cost of exact minimality; follow with a cotan pass).

    cotan_mode: "mollify" (default; true cotangents from intrinsically
    mollified lengths, see solver/cotan) or "clamp" (the historical
    [0.01, 20] clip, kept for comparison).  The default changed after
    tests/bench measured mollify strictly more minimal (catenoid waist
    error 10x smaller, mean-curvature residual 6-10x smaller on the
    Seifert spans), faster (fewer CG iterations), and still embedded
    across the whole Seifert q x samples sweep.
    groom_every=k > 0 runs a mesh-grooming cycle (Delaunay flips +
    tangential vertex averaging, solver/groom) every k outer iterations;
    T is then rewritten in place (counts unchanged), so callers keeping
    a separate quad list for display are unaffected."""
    n = len(V)
    free = ~fixed
    nfree = int(np.sum(free))
    if nfree == 0:
        return V
    if groom_every:
        try:
            from ..solver import groom as _sg
        except ImportError:
            from solver import groom as _sg
    for _it in range(outer_iters):
        if groom_every and _it and _it % groom_every == 0:
            _sg.groom(V, T, fixed=fixed, smooth_lam=groom_smooth)
        E, W = _cotan_weights(V, T, mode=cotan_mode)
        if uniform:
            W = np.ones_like(W)
        deg = np.zeros(n)
        np.add.at(deg, E[:, 0], W)
        np.add.at(deg, E[:, 1], W)

        def matvec_full(Xf):
            y = deg[:, None] * Xf
            np.add.at(y, E[:, 0], -W[:, None] * Xf[E[:, 1]])
            np.add.at(y, E[:, 1], -W[:, None] * Xf[E[:, 0]])
            return y

        Xb = np.where(fixed[:, None], V, 0.0)
        b = -matvec_full(Xb)[free]

        def matvec(xf):
            full = np.zeros((n, 3))
            full[free] = xf
            return matvec_full(full)[free]

        x = V[free].copy()
        r = b - matvec(x)
        p = r.copy()
        rs = np.sum(r * r, axis=0)
        b_norm = max(np.max(np.sum(b * b, axis=0)), 1e-30)
        for _cg in range(cg_iters):
            Ap = matvec(p)
            pAp = np.sum(p * Ap, axis=0)
            alpha = rs / np.where(np.abs(pAp) > 1e-30, pAp, 1e-30)
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.sum(r * r, axis=0)
            if np.max(rs_new) < cg_tol * b_norm:
                break
            p = r + (rs_new / np.maximum(rs, 1e-30)) * p
            rs = rs_new
        move = np.max(np.linalg.norm(x - V[free], axis=1))
        V[free] = x
        if move < 1e-6 * max(1.0, np.max(np.abs(V))):
            break
    return V


def facet_volume(V, T, symmetric=False):
    """The volume Evolver's facets carry, in Evolver's own convention.

    By DEFAULT that is the `z dx dy` form -- (z1+z2+z3)/6 times twice the
    signed area of the triangle's projection on the xy plane -- and only
    under a datafile's `symmetric_content` is it the symmetric
    triple-product (the cone from the origin).  The distinction is not
    cosmetic: Brakke's `content:` integrands are the Green's-theorem
    closures for the FIRST form, which is why the two vertical planes of
    Schwarz P carry no content at all while `x = z`, a plane through the
    origin, does.  Adding those integrands to a cone term computes
    neither quantity, and for pcell it put the flat starting square 0.118
    away from a volume it in fact satisfies exactly.
    """
    V = np.asarray(V, dtype=float)
    P = V[np.asarray(T)]
    if symmetric:
        return float(np.sum(np.einsum('ij,ij->i', P[:, 0],
                                      np.cross(P[:, 1], P[:, 2]))) / 6.0)
    zbar = (P[:, 0, 2] + P[:, 1, 2] + P[:, 2, 2]) / 6.0
    twice = ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
             - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
    return float(np.sum(zbar * twice))


def facet_volume_grad(V, T, symmetric=False):
    """d(facet_volume)/dp, vertex by vertex."""
    V = np.asarray(V, dtype=float)
    T = np.asarray(T)
    g = np.zeros_like(V)
    if symmetric:
        P = V[T]
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            np.add.at(g, T[:, a], np.cross(P[:, b], P[:, c]) / 6.0)
        return g
    P = V[T]
    x1, y1 = P[:, 0, 0], P[:, 0, 1]
    x2, y2 = P[:, 1, 0], P[:, 1, 1]
    x3, y3 = P[:, 2, 0], P[:, 2, 1]
    twice = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    zbar = (P[:, 0, 2] + P[:, 1, 2] + P[:, 2, 2]) / 6.0
    # d(twice)/d(each corner), and d(zbar)/dz = 1/6 at every corner.
    dx = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)
    dy = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)
    for k in range(3):
        np.add.at(g[:, 0], T[:, k], zbar * dx[:, k])
        np.add.at(g[:, 1], T[:, k], zbar * dy[:, k])
        np.add.at(g[:, 2], T[:, k], twice / 6.0)
    return g


def cone_volume(V, T):
    """Signed volume of the cone from the origin over the mesh.

    The divergence-theorem term Evolver computes as a facet integral of
    (x/3, y/3, z/3); for a closed surface it is the enclosed volume, and
    for an open one it is that volume minus whatever the boundary caps
    contribute.
    """
    V = np.asarray(V, dtype=float)
    P = V[np.asarray(T)]
    return float(np.sum(np.einsum('ij,ij->i', P[:, 0],
                                  np.cross(P[:, 1], P[:, 2]))) / 6.0)


def cone_volume_grad(V, T):
    """d(cone_volume)/dp, vertex by vertex."""
    V = np.asarray(V, dtype=float)
    T = np.asarray(T)
    g = np.zeros_like(V)
    P = V[T]
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        np.add.at(g, T[:, a], np.cross(P[:, b], P[:, c]) / 6.0)
    return g


def content_segments(fe, arc_of, nrows):
    """[(field, rows)] -- the boundary runs that carry a content field.

    Each run is closed THROUGH its following corner.  Leaving the last
    segment out drops 1/m of every integral, which at m = 96 is a whole
    percent of the enclosed volume and quietly biases the constraint.
    """
    out = []
    order = np.nonzero(arc_of >= 0)[0]
    if not len(order):
        return out
    for eid, cons in sorted(getattr(fe, 'edge_constraints', {}).items()):
        rows = np.nonzero(arc_of == eid)[0]
        if len(rows) < 2:
            continue
        for n in cons:
            field = fe.constraint_content(n)
            if field is None:
                continue
            # The row after this arc's last, wrapping inside the loop the
            # arc belongs to, so the run reaches the corner.
            lo = order[0]
            hi = order[-1]
            nxt = rows[-1] + 1
            if nxt > hi or arc_of[nxt] < 0:
                nxt = lo
            out.append((field, np.append(rows, nxt)))
    return out


def _content_field(fe, field, P):
    """Evaluate a content integrand at points, with its gradient."""
    from .fedata import _expr, FEError
    C = np.zeros_like(P)
    G = np.zeros((len(P), 3, 3))
    base = dict(fe.params)
    names = (('x', 'x1', 'X1'), ('y', 'x2', 'X2'), ('z', 'x3', 'X3'))
    for j, p in enumerate(P):
        env = dict(base)
        for k, nm in enumerate(names):
            for a in nm:
                env[a] = float(p[k])
        try:
            C[j] = [_expr(e, env) for e in field]
        except FEError:
            return None, None
        # Central differences: the integrands are low-order polynomials,
        # so this is exact to round-off and saves writing a symbolic
        # differentiator for three expressions.
        for k, nm in enumerate(names):
            h = 1e-5 * max(1.0, abs(float(p[k])))
            up, dn = dict(env), dict(env)
            for a in nm:
                up[a] = float(p[k]) + h
                dn[a] = float(p[k]) - h
            try:
                cu = np.array([_expr(e, up) for e in field])
                cd = np.array([_expr(e, dn) for e in field])
            except FEError:
                return None, None
            G[j, :, k] = (cu - cd) / (2.0 * h)
    return C, G


def boundary_content(fe, V, arc_of, want_grad=False):
    """The line integral that closes an open body over its constraints.

    Evolver's `content:` integrand, integrated round each boundary run
    that lies on a constraint declaring one.  Arcs on a plane through the
    origin contribute nothing, which is why only some constraints carry
    the field at all.

    With `want_grad`, also returns d(content)/dp.  The gradient matters
    more than it looks: for Schwarz P the facet term starts at exactly
    zero and the whole volume IS the content, so a restore step driven by
    the facet gradient alone is pushing on something it does not control.
    """
    V = np.asarray(V, dtype=float)
    total = 0.0
    grad = np.zeros_like(V) if want_grad else None
    for field, rows in content_segments(fe, arc_of, len(V)):
        P = V[rows]
        mid = 0.5 * (P[:-1] + P[1:])
        seg = P[1:] - P[:-1]
        C, G = _content_field(fe, field, mid)
        if C is None:
            continue
        total += float(np.sum(C * seg))
        if want_grad:
            # d/dp of C(m).s with m = (a+b)/2 and s = b - a:
            #   tail  <-  0.5 G^T s - C      head  <-  0.5 G^T s + C
            gs = 0.5 * np.einsum('jki,jk->ji', G, seg)
            np.add.at(grad, rows[:-1], gs - C)
            np.add.at(grad, rows[1:], gs + C)
    return (total, grad) if want_grad else total


def fe_volume(fe, V, T, arc_of, want_grad=False):
    """The body volume the datafile fixes, Evolver's way.

    Facet term plus the content line integrals, in the convention the
    datafile actually declares.  For `pcell.fe` this returns exactly 1/12
    at the flat starting square -- which is the point: Brakke's initial
    configuration already satisfies its own volume constraint, so a
    correct implementation has nothing to restore before it starts, and
    an incorrect one begins by throwing the patch across the cell.
    """
    sym = bool(re.search(r'^\s*symmetric_content\b', fe.src, re.M | re.I))
    # `volconst` seeds the body's volume sum; leaving it out shifts the
    # target by a fixed amount, so the constraint would hold the surface
    # in the wrong place rather than not at all.
    v = facet_volume(V, T, symmetric=sym) + float(fe.body_volconst())
    if want_grad:
        c, gc = boundary_content(fe, V, arc_of, want_grad=True)
        return v + c, facet_volume_grad(V, T, symmetric=sym) + gc
    return v + boundary_content(fe, V, arc_of)


def minimize_area_at_volume(fe, V, T, fixed, slide, arc_of, target,
                            outer_iters=260, step=0.9):
    """Minimise area at FIXED VOLUME, the way Evolver does it.

    Area alone is the wrong functional for a surface whose whole boundary
    slides, and for Schwarz P demonstrably so: its constraint planes
    admit a family of flat squares at x = c whose area is c(1-c), maximal
    exactly at the datafile's start and falling to zero at either end.
    The infimum is a collapse into a corner, the true P patch is a saddle
    of area alone, and a descent simply slides past it -- which is what
    produced a patch 14% too large made of a few big panels.  Brakke's
    own comment says as much: "Surface is stabilized with a volume
    constraint, since we know the P-surface equipartitions volume".

    So the motion is Evolver's: one velocity field over all movable
    vertices, `-(gA - lambda gV)` with `lambda = <gA,gV>/<gV,gV>`, which
    is volume-preserving to first order, both gradients projected into
    the joint tangent space of each vertex's constraints BEFORE the
    vertex moves.  A backtracking line search on area follows, then a
    couple of Newton steps along `gV` to mop up second-order drift.  The
    earlier attempt moved first and projected afterwards, drove the
    restore with a gradient that did not match its own functional, and
    diverged.
    """
    V = np.array(V, dtype=float)
    T = np.asarray(T)
    hold = np.asarray(fixed, dtype=bool).copy()
    groups = group_slide_planes(slide, len(V)) if slide else []
    for rows, _p in groups:
        hold[rows] = True
    movable = ~hold
    for rows, _p in groups:
        movable[rows] = True

    def tangent(g):
        g = np.array(g, dtype=float)
        g[~movable] = 0.0
        return project_rows_tangent(g, groups) if groups else g

    def volume(X):
        return fe_volume(fe, X, T, arc_of)

    def vgrad(X):
        _v, g = fe_volume(fe, X, T, arc_of, want_grad=True)
        return tangent(g)

    def restore(X, rounds=6):
        for _ in range(rounds):
            err = target - volume(X)
            if abs(err) <= 1e-11 * max(1.0, abs(target)):
                break
            gv = vgrad(X)
            d = float(np.sum(gv * gv))
            if d <= 1e-30:
                break
            X = X + (err / d) * gv
            if groups:
                X = project_rows(X, groups)
        return X

    def area_grad(X):
        E, W = _cotan_weights(X, T)
        g = np.zeros_like(X)
        deg = np.zeros(len(X))
        d = X[E[:, 0]] - X[E[:, 1]]
        np.add.at(g, E[:, 0], W[:, None] * d)
        np.add.at(g, E[:, 1], -W[:, None] * d)
        np.add.at(deg, E[:, 0], W)
        np.add.at(deg, E[:, 1], W)
        return tangent(g / np.maximum(deg, 1e-30)[:, None])

    if groups:
        V = project_rows(V, groups)
    V = restore(V)
    best = float(mesh_area(V, T))
    for _it in range(outer_iters):
        gA = area_grad(V)
        gV = vgrad(V)
        d = float(np.sum(gV * gV))
        lam = float(np.sum(gA * gV)) / d if d > 1e-30 else 0.0
        v = gA - lam * gV
        big = float(np.max(np.linalg.norm(v, axis=1)))
        if not np.isfinite(big) or big < 1e-9:
            break
        lam_step = step
        moved = False
        while lam_step > 1e-5:
            X = V - lam_step * v
            if groups:
                X = project_rows(X, groups)
            X = restore(X)
            a = float(mesh_area(X, T))
            if np.isfinite(a) and a < best - 1e-13:
                V, best, moved = X, a, True
                break
            lam_step *= 0.5
        if not moved:
            break
    return V


def minimize_area_volume(V, T, fixed, slide, target, content=None,
                         outer_iters=90, inner=8, step=0.5):
    """Minimise area at FIXED VOLUME, with boundary arcs free on planes.

    Area alone is the wrong problem for a surface whose entire boundary
    slides: it has no minimum, and the patch simply shrinks into a corner
    of the constraint planes.  Schwarz P, Schwarz D, Neovius and Schoen's
    I-WP are all posed this way, and all four came out as flat plates
    until the volume was held.

    The volume is restored after each area step by moving along its own
    gradient, projected into the constraint planes so the boundary stays
    where the datafile put it.  `content` is a callable returning the
    boundary line integral that closes the body over those planes.
    """
    V = np.asarray(V, dtype=float).copy()
    T = np.asarray(T)
    hold = np.asarray(fixed, dtype=bool).copy()
    units = [(rows, np.asarray(vec, float),
              np.asarray(vec, float) / float(np.linalg.norm(vec)), float(off))
             for rows, vec, off in slide]
    for rows, _v, _u, _o in units:
        hold[rows] = True
    movable = ~hold
    for rows, _v, _u, _o in units:
        movable[rows] = True

    def project(X):
        for rows, vec, _u, off in units:
            X[rows] -= ((X[rows] @ vec - off)
                        / float(vec @ vec))[:, None] * vec
        return X

    def volume(X):
        v = cone_volume(X, T)
        return v + (content(X) if content else 0.0)

    span0 = float(np.max(V.max(0) - V.min(0))) or 1.0

    def restore(X):
        # Newton on the volume, but with the step CAPPED.  Near a flat
        # start the volume gradient is small and an uncapped step throws
        # the patch across the cell -- Schwarz P came out with 500 times
        # its own area that way.
        cap = 0.05 * span0
        for _ in range(60):
            err = target - volume(X)
            if abs(err) <= 1e-12 * max(1.0, abs(target)):
                break
            g = cone_volume_grad(X, T)
            g[~movable] = 0.0
            for rows, _v, unit, _o in units:
                g[rows] -= (g[rows] @ unit)[:, None] * unit
            denom = float(np.sum(g * g))
            if denom <= 1e-30:
                break
            d = (err / denom) * g
            big = float(np.max(np.linalg.norm(d, axis=1)))
            if big > cap:
                d *= cap / big
            X += d
            project(X)
        return X

    # A patch that starts flat encloses nothing, and the volume gradient
    # of a flat plate is orthogonal to it everywhere -- fine in
    # direction, but the plate has to be lifted off its own plane before
    # the Newton step has a well-conditioned quantity to work with.
    # pcell's four corners all have x = 0.5, so this is not a corner case
    # here; it is the normal one.
    V = project(V)
    if abs(volume(V) - target) > 0.25 * abs(target):
        C = V - V.mean(0)
        _u, _s, vt = np.linalg.svd(C, full_matrices=False)
        V[movable] += (0.12 * span0) * vt[2]
        V = project(V)
    V = restore(V)
    for _ in range(outer_iters):
        X = np.asarray(minimize_area(V.copy(), T, hold, outer_iters=inner),
                       dtype=float)
        E, W = _cotan_weights(X, T)
        g = np.zeros_like(X)
        deg = np.zeros(len(X))
        d = X[E[:, 0]] - X[E[:, 1]]
        np.add.at(g, E[:, 0], W[:, None] * d)
        np.add.at(g, E[:, 1], -W[:, None] * d)
        np.add.at(deg, E[:, 0], W)
        np.add.at(deg, E[:, 1], W)
        for rows, _v, unit, _o in units:
            gg = g[rows] / np.maximum(deg[rows], 1e-30)[:, None]
            gg -= (gg @ unit)[:, None] * unit
            X[rows] -= step * gg
        X = restore(project(X))
        if not np.all(np.isfinite(X)):
            break
        moved = float(np.max(np.linalg.norm(X - V, axis=1)))
        V = X
        if moved < 1e-7 * max(1.0, float(np.max(np.abs(V)))):
            break
    return V


def minimize_area_sliding(V, T, fixed, slide, outer_iters=1500, inner=10,
                          step=0.6, tol=1e-6):
    """Area minimisation with some boundary arcs free to slide on planes.

    Twenty-four of Brakke's forty-three adjoint datafiles pin only part
    of their contour and let one arc run free on a plane.  The minimal
    surface then meets that plane at a RIGHT ANGLE and the arc's shape is
    an output of the solve, not an input to it; spanning the polygon as
    if it were pinned gives a different surface -- for Schoen's batwing,
    half again too much area, on a cell that still welded into one clean
    sheet and passed every topological check.

    Freeing those rows inside the Laplace solve does not work: they are
    unknowns in three dimensions there, so the solve pulls them into the
    interior and re-projecting afterwards restores only the normal
    component, leaving the arc a little further in each round until the
    patch collapses.  (Measured: the conjugate's area ratio drifted from
    1.07 to 1.58 as that loop was given more rounds, away from the truth
    rather than toward it.)

    So the interior is solved exactly with the arc held, and the arc is
    then stepped along the IN-PLANE part of the Dirichlet gradient, which
    vanishes exactly when the surface meets the plane orthogonally.
    `slide` is a list of `(rows, normal, offset)`; the normal need not be
    a unit vector, and the plane is `normal . p = offset`.
    """
    V = np.asarray(V, dtype=float)
    if not slide:
        return minimize_area(V, T, fixed, outer_iters=outer_iters * inner)
    hold = np.asarray(fixed, dtype=bool).copy()
    groups = group_slide_planes(slide, len(V))
    for rows, _planes in groups:
        hold[rows] = True
    moving = np.unique(np.concatenate([r for r, _p in groups]))
    V = project_rows(np.array(V, dtype=float), groups)
    V = np.asarray(minimize_area(V, T, hold, outer_iters=inner), dtype=float)
    best = float(mesh_area(V, T))
    for _it in range(outer_iters):
        E, W = _cotan_weights(V, T)
        g = np.zeros_like(V)
        deg = np.zeros(len(V))
        d = V[E[:, 0]] - V[E[:, 1]]
        np.add.at(g, E[:, 0], W[:, None] * d)
        np.add.at(g, E[:, 1], -W[:, None] * d)
        np.add.at(deg, E[:, 0], W)
        np.add.at(deg, E[:, 1], W)
        g = g / np.maximum(deg, 1e-30)[:, None]
        g = project_rows_tangent(g, groups)
        worst = float(np.max(np.linalg.norm(g[moving], axis=1)))
        if not np.isfinite(worst) or \
                worst < tol * max(1.0, float(np.max(np.abs(V)))):
            break
        # Backtrack on area.  Without it this loop diverges on about a
        # third of the sliding cases -- the arc overshoots, the triangles
        # touching it invert, and the "minimal" patch comes out an order
        # of magnitude too big while still assembling into a clean cell.
        lam = step
        while lam > 1e-4:
            X = V.copy()
            X[moving] -= lam * g[moving]
            X = project_rows(X, groups)
            X = np.asarray(minimize_area(X, T, hold, outer_iters=inner),
                           dtype=float)
            a = float(mesh_area(X, T))
            if np.isfinite(a) and a <= best + 1e-12:
                V, best = X, a
                break
            lam *= 0.5
        else:
            break
    return np.asarray(minimize_area(V, T, hold, outer_iters=inner * 2),
                      dtype=float)


def minimize_area_lbfgs(V, T, fixed, outer_iters=200, m=8, step_cap=4.0,
                        cotan_mode="mollify", guard=None, tol=1e-10,
                        h0_eps=1e-3, h0_tol=1e-2, h0_iters=100):
    """Direct L-BFGS area descent over the free vertices (Laplacian H0
    seed, Armijo backtracking), the measured upgrade path over the
    Pinkall-Polthier iteration above: on the catenoid it reaches the
    same area ~2.7x faster in wall clock with a mean-curvature residual
    200-380x smaller (see research/plans/lbfgs-descent-core-plan.md).
    V is modified in place and returned.

    guard: optional collision guard (solver/collide) -- True, an options
    dict, or a MeshGuard.  The simplified-IPC barrier joins the energy
    and gradient and its conservative cap bounds every trial step, so
    the sheet cannot pass through itself no matter how long the flow
    runs.  That matters here because the least-area surface in a class
    like the Seifert spans is NOT embedded: the unguarded flow must be
    stopped by an iteration cap (_SEIFERT_MAX_ITERS) before it pinches
    through, while the guarded flow settles against its own contact
    barrier instead."""
    fixed = np.asarray(fixed, bool)
    free = ~fixed
    if not np.any(free):
        return V
    guard = _scol.make_guard(guard)
    lb = _sd.LBFGS(m=m)
    h0 = _sd.LaplacianH0(eps=h0_eps, tol=h0_tol, max_iters=h0_iters)

    def _garea(Vv):
        E, W = _cotan_weights(Vv, T, mode=cotan_mode)
        g = np.zeros_like(Vv)
        contrib = W[:, None] * (Vv[E[:, 0]] - Vv[E[:, 1]])
        np.add.at(g, E[:, 0], contrib)
        np.add.at(g, E[:, 1], -contrib)
        return g

    def _E(Vv):
        Eb = 0.0 if guard is None else guard.energy(Vv)
        return mesh_area(Vv, T) + Eb

    x_prev = None
    g_prev = None
    if guard is not None:
        guard.ensure(V, T)
    E_prev = _E(V)
    flat = 0
    for _it in range(1, outer_iters + 1):
        if guard is not None:
            # A rebuild does not change the energy VALUE (the gate is
            # exact), so E_prev from the last acceptance stays valid.
            guard.ensure(V, T)
        g = _garea(V)
        if guard is not None:
            g = g + guard.gradient(V)
        g[fixed] = 0.0
        if x_prev is not None:
            lb.push((V - x_prev).ravel(), (g - g_prev).ravel())
        h0.update(V, T, free=free)
        d = -lb.direction(g.ravel(), h0=h0).reshape(-1, 3)
        d[fixed] = 0.0
        slope = float(np.einsum('nj,nj->', g, d))
        if not (slope < 0.0):
            lb.reset()
            d = -h0(g.ravel()).reshape(-1, 3)
            d[fixed] = 0.0
            slope = float(np.einsum('nj,nj->', g, d))
            if not (slope < 0.0):
                break
        Lmean = float(np.mean(np.linalg.norm(
            V[T[:, 1]] - V[T[:, 0]], axis=1)))
        dmax = float(np.max(np.linalg.norm(d, axis=1)))
        s_max = step_cap * Lmean / max(dmax, 1e-300)
        if guard is not None:
            s_max = min(s_max, float(guard.max_step(V, d)))
        x1, s, E1, _ne = _sd.armijo_backtrack(
            _E, V, d, slope, s0=min(1.0, s_max), E0=E_prev, s_max=s_max)
        if s == 0.0:
            break
        x_prev = V.copy()
        g_prev = g
        V[:] = x1
        if abs(E1 - E_prev) < tol * max(E_prev, 1e-300):
            flat += 1
            if flat >= 3:
                break
        else:
            flat = 0
        E_prev = E1
    return V


def relax_normal_flow(V, T, fixed, iters=60, lam=0.4,
                      cotan_mode="mollify"):
    """Mean-curvature flow restricted to the surface normal: pulls a
    (slightly perturbed) net back toward the minimal surface without
    tangential sliding, so a fair control net stays fair. Only suitable
    as a polish -- it cannot perform global reorganization."""
    free = ~fixed
    n = len(V)
    for _ in range(iters):
        E, W = _cotan_weights(V, T, mode=cotan_mode)
        lap = np.zeros((n, 3))
        d = V[E[:, 1]] - V[E[:, 0]]
        np.add.at(lap, E[:, 0], W[:, None] * d)
        np.add.at(lap, E[:, 1], -W[:, None] * d)
        wsum = np.zeros(n)
        np.add.at(wsum, E[:, 0], W)
        np.add.at(wsum, E[:, 1], W)
        umb = lap / np.maximum(wsum, 1e-12)[:, None]
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        vn = np.zeros((n, 3))
        np.add.at(vn, T[:, 0], fn)
        np.add.at(vn, T[:, 1], fn)
        np.add.at(vn, T[:, 2], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        move = np.sum(umb * vn, axis=1, keepdims=True) * vn
        V[free] += lam * move[free]
    return V


def fair_grid_2d(G, iters=8, step=0.5):
    """Light Laplacian fairing of an (rows, m, 3) net, cyclic in m,
    end rows pinned. Restores row/column coherence after per-column
    resampling; follow with relax_normal_flow to restore minimality."""
    G = G.copy()
    for _ in range(iters):
        up = np.roll(G, 1, axis=0)
        dn = np.roll(G, -1, axis=0)
        up[0] = G[0]
        dn[-1] = G[-1]
        lt = np.roll(G, 1, axis=1)
        rt = np.roll(G, -1, axis=1)
        avg = (up + dn + lt + rt) / 4.0
        G[1:-1] += step * (avg[1:-1] - G[1:-1])
    return G


def fair_grid_columns(G):
    """Re-sample every column (axis 0) of an (rows, m, 3) grid uniformly
    by arc length. Points stay on their column polylines -- i.e. on the
    solved surface -- but the severe bunching/shear the area solver
    introduces (which makes a NURBS control net ring) is equalized."""
    rows, m, _ = G.shape
    out = np.empty_like(G)
    for i in range(m):
        P = G[:, i, :]
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total < 1e-12:
            out[:, i, :] = P
            continue
        t = np.linspace(0.0, total, rows)
        for a in range(3):
            out[:, i, a] = np.interp(t, s, P[:, a])
    return out


def mesh_area(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return 0.5 * float(np.sum(np.linalg.norm(n, axis=1)))


def _quads_to_tris(quads):
    T = []
    for q in quads:
        if len(q) == 4:
            T.append((q[0], q[1], q[2]))
            T.append((q[0], q[2], q[3]))
        else:
            T.append(tuple(q))
    return np.array(T, dtype=np.int64)


def build_disk_grid(boundary, rings):
    """Initial disk spanning one loop: concentric rings toward centroid."""
    m = len(boundary)
    cen = boundary.mean(axis=0)
    verts = [boundary]
    for r in range(1, rings):
        f = r / rings
        verts.append(boundary * (1 - f) + cen * f)
    V = np.concatenate(verts + [cen[None, :]], axis=0)
    quads = []
    for r in range(rings - 1):
        for i in range(m):
            i2 = (i + 1) % m
            quads.append((r * m + i, r * m + i2,
                          (r + 1) * m + i2, (r + 1) * m + i))
    last = (rings - 1) * m
    cidx = rings * m
    for i in range(m):
        quads.append((last + i, last + (i + 1) % m, cidx))
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:m] = True
    return V, quads, fixed


def build_annulus_grid(loopA, loopB, rows):
    """Initial ruled surface between two aligned loops of equal length."""
    m = len(loopA)
    verts = []
    for r in range(rows + 1):
        f = r / rows
        verts.append(loopA * (1 - f) + loopB * f)
    V = np.concatenate(verts, axis=0)
    quads = []
    for r in range(rows):
        for i in range(m):
            i2 = (i + 1) % m
            quads.append((r * m + i, r * m + i2,
                          (r + 1) * m + i2, (r + 1) * m + i))
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:m] = True
    fixed[rows * m:] = True
    return V, quads, fixed


def torus_knot(p, q, m, scale=1.0, tube=1.0):
    t = np.linspace(0, TAU, m, endpoint=False)
    r = np.cos(q * t) * tube + 2.0
    return np.stack([r * np.cos(p * t), r * np.sin(p * t),
                     -np.sin(q * t) * tube], axis=1) * scale


# The Seifert span is embedded as built, but the area flow that relaxes
# it afterwards is what can undo that: the cotangent-Laplacian solve is
# free to drag the membrane's rim collar (where the inner Seifert
# circle's ~q/2 waves of wobble die away) through the membrane behind
# it, and to pinch the handles. Three limits keep the flow inside the
# embedded regime: a floor on the azimuth samples per half period, so
# the collar is never one quad thick; a ceiling on the radial rows,
# since rows much finer than the azimuth spacing make slivers that the
# cotangent weights fold (the ceiling tightens with q, whose waves
# crowd the collar); and a cap on the relaxation passes, because the
# least-area surface in this class is not the embedded one -- run the
# flow long enough and it pinches through itself whatever the grid.
# All three were fixed by sweeping q x samples x rings x iterations and
# counting genuine edge-through-triangle crossings; a normal-flip
# detector was tried first and rejected (it both misses folds that
# creep in without a flip and fires on the harmless slivers in the
# q = 1 band).
_SEIFERT_MIN_G = 12         # azimuth samples per half period, minimum
_SEIFERT_MAX_ROWS = 48      # radial row budget, divided down as q grows
_SEIFERT_MAX_ITERS = 8      # relaxation passes the span stays embedded for


def _seifert_rows(q, rings, mA):
    """Radial rows the relaxation stays embedded for: the ceiling falls
    with q (whose waves crowd the collar) and never lets a row be finer
    than a third of the azimuth spacing."""
    return max(4, min(int(rings),
                      _SEIFERT_MAX_ROWS // ((int(q) + 3) // 2),
                      mA // 3))


def build_seifert_span_grid(q, m, rings, knot_scale=1.0, circle_radius=4.5,
                            inner_height=1.0, inner_lift=0.0, window=0.35):
    """Embedded single-sheet spanning surface between the (2, q) torus
    knot (q odd) and the round circle of radius `circle_radius`.

    An embedded *annulus* between a knotted and a round boundary cannot
    exist (its two boundary curves would be isotopic knots), so this
    builds the correct next-simplest surface: Seifert's algorithm
    (H. Seifert, 1934) applied to the round (2, q) diagram, with the
    outer Seifert disk opened out to the circle. Three families of
    structured patches share their seam vertices:

      funnel   -- annulus from the circle in to the outer smoothed
                  Seifert circle (knot arcs + saddle diagonals);
      membrane -- disk spanning the inner smoothed Seifert circle
                  across the middle;
      q bands  -- one half-twist Coons patch per diagram crossing,
                  edged by the two knot strands and the two diagonals.

    Every patch preserves azimuth and they occupy disjoint radial
    ranges (funnel outside `r_out`, membrane inside `r_in`, bands in
    between, in disjoint azimuth windows), so the result is embedded
    by construction. Genus (q - 1) / 2, two boundary loops: the exact
    torus knot (2 m-ish samples) and the exact circle. Returns
    (V, quads, fixed) with `fixed` True exactly on the two boundary
    loops, ready for minimize_area -- for at most _SEIFERT_MAX_ITERS
    passes, beyond which the area flow pulls the sheet back through
    itself (see the note on that constant).
    """
    q = int(q)
    if q < 1 or q % 2 == 0:
        raise ValueError("build_seifert_span_grid needs odd q >= 1")
    k = float(knot_scale)
    h = float(inner_height)
    R = float(circle_radius)
    # `m` is a request, not a promise: the azimuth resolution is floored
    # at _SEIFERT_MIN_G samples per half period so the relaxation cannot
    # fold the membrane collar (a coarse grid leaves it one quad thick)
    g = max(_SEIFERT_MIN_G, int(round(m / (2.0 * q))))
    mA = 2 * q * g                      # azimuth samples, once around
    # ... and `rings` likewise: rows finer than a third of the azimuth
    # spacing, or too many of them for this q, slice the sheet into
    # slivers that the flow folds
    rings = _seifert_rows(q, rings, mA)
    wn = max(2, min(g - 2, int(round(window * g))))
    ns = 2 * wn + 1                     # samples across a crossing window
    th = TAU * np.arange(mA) / mA

    def strand(which, t):
        """Knot strand 0/1 at azimuth t (i.e. tau = t/2 [+ pi])."""
        ph = q * t / 2.0 + (q * math.pi if which else 0.0)
        rho = 2.0 * k + k * math.cos(ph)
        return np.array([rho * math.cos(t), rho * math.sin(t),
                         -k * h * math.sin(ph) + inner_lift])

    centers = [g * (2 * n + 1) for n in range(q)]   # crossing indices
    in_open = np.zeros(mA, dtype=bool)  # strictly inside a window
    for c in centers:
        for d in range(1 - wn, wn):
            in_open[(c + d) % mA] = True

    cw = math.sin(q * (wn * TAU / mA) / 2.0)   # |cos(q u/2)| at edge
    r_out = 2.0 * k + k * cw            # diagonal rail radii
    r_in = 2.0 * k - k * cw

    # smoothed Seifert circles: knot arcs outside the crossing windows,
    # constant-radius diagonals (the saddle rails) across them
    Cout = np.empty((mA, 3))
    Cin = np.empty((mA, 3))
    for i in range(mA):
        A, B = strand(0, th[i]), strand(1, th[i])
        a_out = math.cos(q * th[i] / 2.0) >= 0.0
        Cout[i], Cin[i] = (A, B) if a_out else (B, A)
    for c in centers:
        s1 = 0 if math.cos(q * th[(c - wn) % mA] / 2.0) >= 0.0 else 1
        zo0 = Cout[(c - wn) % mA][2]    # strand z at the window edges
        zo1 = strand(1 - s1, (c + wn) * TAU / mA)[2]
        zi0 = Cin[(c - wn) % mA][2]
        zi1 = strand(s1, (c + wn) * TAU / mA)[2]
        for d in range(-wn, wn + 1):
            i = (c + d) % mA
            fb = (d + wn) / (2.0 * wn)
            Cout[i] = (r_out * math.cos(th[i]), r_out * math.sin(th[i]),
                       (1 - fb) * zo0 + fb * zo1)
            Cin[i] = (r_in * math.cos(th[i]), r_in * math.sin(th[i]),
                      (1 - fb) * zi0 + fb * zi1)

    circle = np.stack([R * np.cos(th), R * np.sin(th),
                       np.zeros(mA)], axis=1)
    verts = []
    fixed = []
    quads = []
    for r in range(rings + 1):          # funnel rows: circle -> Cout
        f = r / rings
        verts.append(circle * (1 - f) + Cout * f)
        fixed.append(np.ones(mA, bool) if r == 0
                     else (~in_open if r == rings
                           else np.zeros(mA, bool)))
    for r in range(rings):
        for i in range(mA):
            i2 = (i + 1) % mA
            quads.append((r * mA + i, r * mA + i2,
                          (r + 1) * mA + i2, (r + 1) * mA + i))
    fun_last = rings * mA               # first index of the Cout row
    base = (rings + 1) * mA             # first index of the membrane
    cen = Cin.mean(axis=0)
    for r in range(rings):              # membrane rows: Cin -> center
        f = r / rings
        verts.append(Cin * (1 - f) + cen * f)
        fixed.append(~in_open if r == 0 else np.zeros(mA, bool))
    verts.append(cen[None, :])
    fixed.append(np.zeros(1, bool))
    cidx = base + rings * mA
    for r in range(rings - 1):
        for i in range(mA):
            i2 = (i + 1) % mA
            quads.append((base + r * mA + i, base + r * mA + i2,
                          base + (r + 1) * mA + i2,
                          base + (r + 1) * mA + i))
    lastm = base + (rings - 1) * mA
    for i in range(mA):
        quads.append((lastm + i, lastm + (i + 1) % mA, cidx))

    Vp = np.concatenate(verts, axis=0)
    Vs = [Vp]
    Fs = [np.concatenate(fixed)]
    nxt = len(Vp)
    for c in centers:                   # one saddle band per crossing
        s1 = 0 if math.cos(q * th[(c - wn) % mA] / 2.0) >= 0.0 else 1
        S1 = np.array([strand(s1, (c - wn + x) * TAU / mA)
                       for x in range(ns)])           # rail y = 0
        S2r = np.array([strand(1 - s1, (c + wn - x) * TAU / mA)
                        for x in range(ns)])          # rail y = ns-1
        Out = Vp[[fun_last + (c - wn + y) % mA for y in range(ns)]]
        Inr = Vp[[base + (c + wn - y) % mA for y in range(ns)]]
        idx = np.empty((ns, ns), dtype=np.int64)
        for y in range(ns):
            idx[0, y] = fun_last + (c - wn + y) % mA
            idx[ns - 1, y] = base + (c + wn - y) % mA
        new_v = []
        new_f = []
        for x in range(1, ns - 1):
            u = x / (ns - 1.0)
            for y in (0, ns - 1):       # strand rails: on the knot
                idx[x, y] = nxt
                new_v.append(S1[x] if y == 0 else S2r[x])
                new_f.append(True)
                nxt += 1
            for y in range(1, ns - 1):  # Coons interior of the saddle
                v = y / (ns - 1.0)
                pt = ((1 - v) * S1[x] + v * S2r[x]
                      + (1 - u) * Out[y] + u * Inr[y]
                      - ((1 - u) * (1 - v) * S1[0]
                         + u * (1 - v) * S1[-1]
                         + (1 - u) * v * S2r[0] + u * v * S2r[-1]))
                idx[x, y] = nxt
                new_v.append(pt)
                new_f.append(False)
                nxt += 1
        Vs.append(np.array(new_v))
        Fs.append(np.array(new_f, dtype=bool))
        for x in range(ns - 1):
            for y in range(ns - 1):
                quads.append((idx[x, y], idx[x + 1, y],
                              idx[x + 1, y + 1], idx[x, y + 1]))
    return np.concatenate(Vs, axis=0), quads, np.concatenate(Fs)


def _selfx_crossings(V, T, tol=1e-7):
    """Count genuine self-intersections of a triangle mesh: edges that
    pierce the open interior of a triangle they share no vertex with
    (Moeller-Trumbore, bbox-prefiltered). O(edges x tris) per bbox
    survivor -- intended for selftest-sized meshes."""
    T = np.asarray(T)
    E = set()
    for tri in T:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]),
                     (tri[2], tri[0])):
            E.add((a, b) if a < b else (b, a))
    E = np.array(sorted(E))
    P0, P1 = V[E[:, 0]], V[E[:, 1]]
    elo, ehi = np.minimum(P0, P1), np.maximum(P0, P1)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    tlo = np.minimum(np.minimum(A, B), C)
    thi = np.maximum(np.maximum(A, B), C)
    count = 0
    for i0 in range(0, len(E), 512):
        i1 = min(i0 + 512, len(E))
        ov = np.all((elo[i0:i1, None, :] <= thi[None, :, :] + tol)
                    & (ehi[i0:i1, None, :] >= tlo[None, :, :] - tol),
                    axis=2)
        ei, ti = np.nonzero(ov)
        ei += i0
        if not len(ei):
            continue
        share = np.zeros(len(ei), dtype=bool)
        for cc in range(2):
            for dd in range(3):
                share |= (E[ei, cc] == T[ti, dd])
        ei, ti = ei[~share], ti[~share]
        if not len(ei):
            continue
        o = V[E[ei, 0]]
        d = V[E[ei, 1]] - o
        e1 = V[T[ti, 1]] - V[T[ti, 0]]
        e2 = V[T[ti, 2]] - V[T[ti, 0]]
        pv = np.cross(d, e2)
        det = np.einsum('ij,ij->i', e1, pv)
        good = np.abs(det) > 1e-14
        inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
        tv = o - V[T[ti, 0]]
        uu = np.einsum('ij,ij->i', tv, pv) * inv
        qv = np.cross(tv, e1)
        vv = np.einsum('ij,ij->i', d, qv) * inv
        tt = np.einsum('ij,ij->i', e2, qv) * inv
        count += int(np.sum(good & (uu > tol) & (vv > tol)
                            & (uu + vv < 1 - tol)
                            & (tt > tol) & (tt < 1 - tol)))
    return count


# ==========================================================================
# ==========================================================================
# The discrete conjugate (adjoint) surface
# ==========================================================================
# The capability that most of Brakke's triply-periodic collection waits
# on, and it turned out to be smaller than it looked.
#
# His `*adj.fe` datafiles were read here for a long time as needing
# constraint planes with free-sliding vertices.  They do not.  Each one
# states a PLAIN PLATEAU PROBLEM -- a closed space polygon with every
# vertex and every edge fixed (GW5adj.fe is six fixed vertices, six
# fixed edges, one hexagonal face) -- which `minimize_area` already
# solves.  The declared constraint planes describe where the CONJUGATE
# surface's boundary lands, not where the original's does.  So the only
# missing piece was the conjugate itself.
#
# Pinkall and Polthier's discrete conjugate, in the form Brakke's
# adjoint.cmd implements: walk the facets, and for a facet with unit
# normal n whose three edge vectors are taken in traversal order,
#
#     p[next] = p[this] - ( cos(b) * e[other]
#                           + sin(b) * (n x e[other]) ) / 2,
#
# with b the Bonnet angle -- b = 90 degrees is the conjugate proper.
# The positions live on EDGES, not vertices, which is the whole trick:
# the conjugate of a conforming triangulation is nonconforming, and
# forcing it onto vertices too early is what makes naive attempts fail.
# Vertices are recovered afterwards by averaging their incident edges.
#
# STATUS -- read before using this on a surface.  The diagnosis below is
# complete; what is missing is a solver, not an understanding.
#
# VERIFIED, and gated in `_selftest`:
#   * the per-facet relation closes exactly -- 5.6e-17 walking one
#     triangle 0 -> 1 -> 2 -> 0, with medial edge lengths |e|/2 at every
#     Bonnet angle, as they must be since the two terms are orthogonal;
#   * at bangle 0 the whole pipeline is the IDENTITY on a real mesh, to
#     0.0e+00 in area -- propagation and vertex recovery together;
#   * on an EXACTLY minimal mesh (a flat grid) the propagation is exact
#     at bangle 90 too: residual 1.0e-16 against an edge scale of 0.25.
#     That is the check that says the algorithm is right.
#
# WHY IT IS NOT YET USED BY ANY SHIPPED ROW.  On a relaxed surface the
# propagation picks up a residual, and the reason is now understood
# rather than suspected.  It has two independent parts:
#
#   1. A REAL PERIOD, on any domain that is not simply connected.  The
#      relation is a discrete 1-form, and integrating it around a
#      non-contractible loop returns the conjugate's period, not zero --
#      the conjugate of a catenoid is a helicoid, which does not close.
#      On an annulus the residual is 42x the edge length and is
#      concentrated exactly where the walk wraps and meets itself; on a
#      simply connected disk it falls to 0.12x.  This is the "period
#      killing" Karcher's conjugate Plateau method exists to do, and it
#      is why Brakke's datafiles carry tunable rhs parameters.
#   2. NON-INTEGRABILITY, because the surface is only approximately
#      minimal.  The closing condition around an interior vertex is
#      exactly the area gradient there -- (1/2) sum over incident facets
#      of n_T x (edge opposite v) -- which `minimize_area` drives down
#      (2.6e-2 -> 1.5e-4 -> 9.7e-6 over 0/10/600 iterations) but never
#      to zero.
#
# So a WALK is the wrong integrator: it commits to whichever facet
# reached an edge first and pushes all the inconsistency into the edges
# reached last.  Measured on a disk, its residual RISES relative to the
# edge length as the mesh refines -- 0.083 at 48x8, 0.116 at 96x16,
# 0.324 at 144x24 -- while the area converges cleanly.
#
# RESUME, and this is a solver task now.  Replace the walk with a
# least-squares integration of the 1-form: minimise
#     sum over facets, over k, |P[e_{k+1}] - P[e_k] - w_{f,k}|^2,
# a graph Laplacian on edge-adjacency, singular only in the global
# translation.  That spreads the inconsistency instead of accumulating
# it.  It was tried here with 600 Jacobi sweeps and abandoned: Jacobi
# converges far too slowly on a Laplacian of this diameter and lost the
# bangle-0 identity entirely (33% area error), which is a solver
# failure, not a formulation one.  Use conjugate gradients, and keep the
# bangle-0 identity as the acceptance test -- it is exact for the walk
# and must stay exact.
#
def _facet_edges(T):
    """Undirected edge index per facet corner, plus the edge list."""
    ids = {}
    per = np.empty((len(T), 3), dtype=np.int64)
    for f, tri in enumerate(T):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            key = (a, b) if a < b else (b, a)
            j = ids.get(key)
            if j is None:
                j = ids[key] = len(ids)
            per[f, k] = j
    return per, ids


def adjoint_mesh(V, T, bangle=90.0):
    """The conjugate as a mesh, in the form where its area is exact.

    The conjugate of a conforming triangulation is NONCONFORMING: it
    carries one position per EDGE, and adjacent triangles meet only at
    those edge midpoints, not at shared vertices.  Brakke states the
    reconstruction directly (`fe/adjoint.cmd`, `write_conjugate`): the
    facet whose three edge positions are P1, P2, P3 becomes the triangle

        P1 + P2 - P3,   -P1 + P2 + P3,   P1 - P2 + P3,

    the unique triangle whose edge midpoints are P1, P2, P3 -- check:
    the midpoint of the first two is P2, of the second two P3, of the
    third and first P1.  Each such triangle is the Bonnet rotation of
    its parent about that parent's own normal, hence CONGRUENT to it, so
    the area is preserved facet by facet at any resolution.

    Two other reconstructions are wrong in ways that look plausible, and
    both were tried here first:

      * either conforming surface `discrete_adjoint` returns.  Those
        converge to the right area under refinement but never hit it
        exactly (corner mode: 2.5373 / 2.5196 / 2.5165 against 2.5179 /
        2.5146 / 2.5136), because forcing one position per vertex is an
        approximation and Brakke calls it a "tweak" for that reason.
        Use them when a conforming mesh is REQUIRED -- reflection groups
        need seams to weld -- and this one when area is being measured.
      * the medial triangles on P1, P2, P3 themselves.  Those ARE
        isometric, but a medial triangle has a QUARTER its parent's
        area, so they measure a quarter of the surface.

    The returned mesh is deliberately unwelded -- adjacent triangles
    share midpoints, not vertices -- because that is what the conjugate
    is.  Weld it afterwards if a closed mesh is wanted, and expect the
    seams to be exactly as good as the input surface is critical.
    """
    V = np.asarray(V, dtype=float)
    T = np.asarray(T, dtype=np.int64)
    P, per = _adjoint_edge_positions(V, T, bangle)
    p1 = P[per[:, 0]]
    p2 = P[per[:, 1]]
    p3 = P[per[:, 2]]
    pts = np.concatenate([p1 + p2 - p3, -p1 + p2 + p3, p1 - p2 + p3], axis=0)
    nf = len(T)
    faces = [(f, f + nf, f + 2 * nf) for f in range(nf)]
    return pts, faces


def _boundary_vertices(T):
    d = defaultdict(int)
    for tri in T:
        for k in range(3):
            d[(int(tri[k]), int(tri[(k + 1) % 3]))] += 1
    return {v for (a, b) in d if (b, a) not in d for v in (a, b)}


def _adjoint_edge_positions(V, T, bangle=90.0):
    """Conjugate positions, one per EDGE, by least-squares integration.

    Shared by `adjoint_mesh` and `discrete_adjoint`; see the note above
    `adjoint_mesh` for why the edge form is the primary one.
    """
    V = np.asarray(V, dtype=float)
    T = np.asarray(T, dtype=np.int64)
    per, _ids = _facet_edges(T)
    ne = int(per.max()) + 1
    P = np.zeros((ne, 3))
    known = np.zeros(ne, dtype=bool)

    e = np.empty((len(T), 3, 3))
    for k in range(3):
        e[:, k] = V[T[:, (k + 1) % 3]] - V[T[:, k]]
    n = np.cross(e[:, 0], e[:, 1])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-300)

    bc = math.cos(math.radians(bangle))
    bs = math.sin(math.radians(bangle))

    # INTEGRATE the 1-form by least squares, with conjugate gradients.
    #
    # The relation P[next] - P[this] = -(bc e + bs n x e)/2 is a discrete
    # 1-form on the facet graph.  Integrating it by WALKING the mesh is
    # exact only if the form is exactly closed, which needs the surface
    # to be exactly discretely minimal -- and a relaxed surface never
    # is.  Worse, a walk commits to whichever facet reached an edge
    # first and dumps every inconsistency on the edges reached last, so
    # its residual RISES relative to the edge length as the mesh refines
    # (0.083 at 48x8, 0.116 at 96x16, 0.324 at 144x24) even while the
    # area converges.
    #
    # Least squares spreads the inconsistency instead: minimise
    #     sum over facets, over k, |P[e_{k+1}] - P[e_k] - w_{f,k}|^2.
    # The normal equations are a graph Laplacian on edge-adjacency,
    # singular only in the global translation, so CG converges on it
    # with the constant mode projected out.  Jacobi does NOT -- it was
    # tried at 600 sweeps and lost the bangle-0 identity entirely, which
    # is a solver failure rather than a formulation one.
    src = per[:, [0, 1, 2]].ravel()
    dst = per[:, [1, 2, 0]].ravel()
    W = np.empty((len(T) * 3, 3))
    for k in range(3):
        ko = (k + 2) % 3
        W[k::3] = -0.5 * (bc * e[:, ko] + bs * np.cross(n, e[:, ko]))
    # reorder to match src/dst raveling (facet-major, k within facet)
    W = np.concatenate([(-0.5 * (bc * e[:, (k + 2) % 3]
                                 + bs * np.cross(n, e[:, (k + 2) % 3])))[:, None]
                        for k in range(3)], axis=1).reshape(-1, 3)

    def _AtA(X):
        r = X[dst] - X[src]
        out = np.zeros_like(X)
        np.add.at(out, dst, r)
        np.add.at(out, src, -r)
        return out

    rhs = np.zeros((ne, 3))
    np.add.at(rhs, dst, W)
    np.add.at(rhs, src, -W)

    P = np.zeros((ne, 3))
    r = rhs - _AtA(P)
    r -= r.mean(0)                     # project out the constant mode
    d = r.copy()
    rs = float(np.sum(r * r))
    for _ in range(4000):
        if rs < 1e-24:
            break
        Ad = _AtA(d)
        denom = float(np.sum(d * Ad))
        if abs(denom) < 1e-300:
            break
        al = rs / denom
        P += al * d
        r -= al * Ad
        r -= r.mean(0)
        rs2 = float(np.sum(r * r))
        d = r + (rs2 / rs) * d
        rs = rs2

    return P, per


def discrete_adjoint(V, T, bangle=90.0, mode="corner"):
    """Conjugate of a discrete minimal surface (Pinkall-Polthier), as a
    CONFORMING mesh -- one position per vertex, same connectivity as the
    input.

    `bangle` sweeps the associate family: 0 returns the original, 90 the
    conjugate.  The honest conjugate is nonconforming (`adjoint_mesh`);
    this is the conforming approximation to it, which is what a
    reflection group needs, since reflecting a mesh whose triangles meet
    only at edge midpoints leaves no seams to weld.

    Two ways to get there, both implemented, `mode` selecting:

    "corner" (default) -- Brakke's own, from `adjoint.cmd`.  The exact
        nonconforming reconstruction gives every facet three corners
        (see `adjoint_mesh`); assign each to its vertex and AVERAGE over
        the incident facets.  Every averaged quantity is a genuine
        vertex position, so nothing systematically shrinks.

    "midpoint" -- least squares on (x_a + x_b)/2 = P_e, solved by CG.
        Very nearly as good (2.5388 / 2.5203 / 2.5168 against corner's
        2.5373 / 2.5196 / 2.5165, true areas 2.5179 / 2.5146 / 2.5136),
        and kept because it is the independent check on "corner": two
        different reconstructions agreeing to 6e-4 is worth more than
        either alone.

        Its history is a warning about blaming the model for the solver.
        This mode was once measured DIVERGING under refinement -- 2.5300
        / 2.0798 / 1.9025 -- and the midpoint operator was blamed, on the
        plausible argument that it damps the mesh's high-frequency part.
        The real cause was the 400 Jacobi sweeps then used to solve the
        normal equations: they simply had not converged, and the finer
        the mesh the further short they fell.  With CG the divergence is
        gone entirely.

    Note what is NOT an option: averaging the edge values P_e themselves.
    At bangle 0 those are the original edge midpoints, so a vertex is not
    their mean -- averaging pulls every vertex to its neighbourhood
    centroid and costs 6% of the area and 31% of the edge lengths at the
    one angle where the transform must be the identity.
    """
    V = np.asarray(V, dtype=float)
    T = np.asarray(T, dtype=np.int64)
    P, per = _adjoint_edge_positions(V, T, bangle)
    ne = len(P)

    if mode == "corner":
        p1 = P[per[:, 0]]
        p2 = P[per[:, 1]]
        p3 = P[per[:, 2]]
        # The corner-to-vertex assignment is forced by the midpoint
        # identity in `adjoint_mesh`: the pair of corners averaging to P1
        # must be the pair spanning the facet's first edge (v0, v1), and
        # so on round the triangle.
        acc = np.zeros((len(V), 3))
        cnt = np.zeros(len(V))
        for k, corner in enumerate((p1 - p2 + p3, p1 + p2 - p3,
                                    -p1 + p2 + p3)):
            np.add.at(acc, T[:, k], corner)
            np.add.at(cnt, T[:, k], 1.0)
        x = acc / np.maximum(cnt, 1.0)[:, None]
        return x - x.mean(0) + V.mean(0)

    if mode != "midpoint":
        raise ValueError("discrete_adjoint: unknown mode %r" % (mode,))

    # Least squares on (x_a + x_b)/2 = P_e.  Normal equations
    #     deg(v) x_v + sum_{w ~ v} x_w = 2 sum_{e in v} P_e,
    # singular only in the global translation, fixed by recentring.
    ends = np.empty((ne, 2), dtype=np.int64)
    for f in range(len(T)):
        for k in range(3):
            ends[int(per[f, k])] = (int(T[f, k]), int(T[f, (k + 1) % 3]))
    rhs = np.zeros((len(V), 3))
    deg = np.zeros(len(V))
    np.add.at(rhs, ends[:, 0], 2.0 * P)
    np.add.at(rhs, ends[:, 1], 2.0 * P)
    np.add.at(deg, ends[:, 0], 1.0)
    np.add.at(deg, ends[:, 1], 1.0)
    deg = np.maximum(deg, 1.0)

    def _AtA(X):
        y = deg[:, None] * X
        np.add.at(y, ends[:, 0], X[ends[:, 1]])
        np.add.at(y, ends[:, 1], X[ends[:, 0]])
        return y

    x = V - V.mean(0)
    r = rhs - _AtA(x)
    r -= r.mean(0)
    d = r.copy()
    rs = float(np.sum(r * r))
    for _ in range(2000):
        if rs <= 1e-24:
            break
        Ad = _AtA(d)
        Ad -= Ad.mean(0)
        alpha = rs / max(float(np.sum(d * Ad)), 1e-300)
        x += alpha * d
        r -= alpha * Ad
        rs_new = float(np.sum(r * r))
        d = r + (rs_new / max(rs, 1e-300)) * d
        rs = rs_new
    return x - x.mean(0) + V.mean(0)


# Schoen's ring-like surfaces (the R family), by relaxation
# ==========================================================================
# A SECOND construction route, and the only one that reaches these.
#
# Schoen's R_I (= Schwarz H), R_II and R_III are, in his own words in the
# 1970 NASA catalogue, "assembled from ring-like surfaces, each bounded
# by the opposite parallel triangles of a prism" -- (pi/3,pi/3,pi/3) for
# R_I, (pi/2,pi/4,pi/4) for R_II, (pi/2,pi/3,pi/6) for R_III.  Karcher
# (1989) section 5.2 builds the same objects as annular Plateau problems
# and calls them triangular catenoids.
#
# The exact-Weierstrass route does NOT reach R_III.  Karcher says why,
# and it is a property of the surfaces rather than a gap in anyone's
# effort: "RIII is not cut by planar symmetry lines into simply
# connected pieces.  RII has such symmetry lines but the conjugate
# contour is complicated, in particular not a Nitsche graph."  Brakke's
# Surface Evolver datafiles state the same problem the same way -- two
# fixed parallel triangles, everything between them free -- so this is
# what those datafiles are, transcribed rather than read at runtime
# (the extension cannot depend on a mirror).
#
# WHAT IS AND IS NOT CLAIMED.  This produces a DISCRETE minimal surface,
# certified by area minimisation and the Pinkall-Polthier cotan flow,
# not by integrating an exact Weierstrass representation.  It is a
# weaker claim than the `hexagonal` spec rows make and the gate reflects
# that: convergence is measured on AREA, which settles to four or five
# figures, and not on median |H|, which on a relaxed mesh measures
# triangle quality as much as it measures the surface.
#
# References:
# - A. H. Schoen, "Infinite periodic minimal surfaces without
#   self-intersections", NASA TN D-5541 (1970), Table II -- the R family
#   and the triangles bounding each.
# - H. Karcher, "The triply periodic minimal surfaces of Alan Schoen and
#   their constant mean curvature companions", manuscripta math. 64
#   (1989) 291-357, section 5.2 -- triangular catenoids, and the
#   obstruction quoted above.
# - U. Pinkall and K. Polthier, "Computing discrete minimal surfaces and
#   their conjugates", Experimental Mathematics 2(1) (1993) -- the flow
#   `minimize_area` runs.
# - K. A. Brakke, "The Surface Evolver", Experimental Mathematics 1(2)
#   (1992); datafiles RII.fe and RIII.fe.

# key -> (label, triangle vertices in the z = 0 plane, prism height).
# The triangles are Schoen's, and the heights are Brakke's `#define HT`.
# The contour need not be convex, or even simple.  Schoen's I-series is
# bounded by ROSETTES -- closed polylines that return to the origin
# several times, two petals for I-6 and I-8 and four for I-9 -- which is
# the "figure-8 wireframe" whose soap film Schoen recorded finding in
# October 1970.  `build_annulus_grid` spans any closed loop, so these
# need contour data and nothing else.
RING_SURFACES = {
    'R1': ("Schoen R-I / Schwarz H (ring form)",
           ((0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(0.75))), 0.5),
    'R2': ("Schoen R-II (ring form, genus 9)",
           ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), 0.2),
    'R3': ("Schoen R-III (ring form, genus 13)",
           ((0.0, 0.0), (math.sqrt(3.0), 0.0), (0.0, 3.0)), 0.5),
    # Two-petal rosettes.  I-6 also ships as an exact Weierstrass row, so
    # it is kept here as a CROSS-CHECK between the two routes rather than
    # as a second way to offer the same surface.
    'I6': ("Schoen I-6 (ring form, genus 5)",
           ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
            (0.0, 0.0), (0.0, -1.0), (-1.0, -1.0), (-1.0, 0.0)), 0.8),
    'I8': ("Schoen I-8 (ring form)",
           ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0),
            (0.0, 0.0), (0.0, -1.0), (-0.5, -0.5), (-1.0, 0.0)), 0.5),
    # Four petals.
    'I9': ("Schoen I-9 (ring form)",
           ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
            (0.0, 0.0), (0.0, 1.0), (-1.0, 1.0),
            (0.0, 0.0), (-1.0, 0.0), (-1.0, -1.0),
            (0.0, 0.0), (0.0, -1.0), (1.0, -1.0)), 0.5),
}


# Half-turn generators for the ring surfaces, transcribed from the
# `view_transform_generators` blocks of Brakke's datafiles.  Each is a
# 180-degree rotation about one edge of the bounding prism: three about
# the edges of the bottom triangle, three about the top ones.  Reflecting
# the annulus in these is Schwarz's principle applied to the straight
# boundary lines, and it is what turns one "triangular catenoid" into a
# piece of the periodic surface.
#
# Stored as (3x3 rotation, translation), the two halves of the 4x4 the
# datafile prints.
def _halfturn_pair(rot, tx, ty, ht):
    """The datafile's generators come in pairs: one about a bottom edge
    and the same rotation about the corresponding top edge, differing
    only by a z offset of 2*HT."""
    return [(rot, (tx, ty, 0.0)), (rot, (tx, ty, 2.0 * ht))]


def ring_generators(key):
    """The tiling generators for one ring surface, or None.

    The R family and the I family do NOT share generators, and assuming
    they do is a silent error rather than a loud one: applying the R
    half-turns to I-8 welds into 244 over-shared edges, and to I-9 into
    14 266 while halving the vertex count -- coincident duplicate sheets,
    which no topological count catches on its own.  So each key declares
    its own, transcribed from its datafile's own
    `view_transform_generators` block, and a key with none returns None
    and is not tiled.

    Note the datafile writes its 4x4 matrices ROW-major and inline, so
    "1 0 0 2*xsize  0 1 0 0  0 0 1 0  0 0 0 1" is a translation by
    (2*xsize, 0, 0), not a shear.
    """
    _label, tri, ht = RING_SURFACES[key]
    rx = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    ry = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
    ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    rotz = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    if key == 'R3':
        c = math.sqrt(3.0)
        rc = ((-0.5, -c / 2.0, 0.0), (-math.sqrt(0.75), 0.5, 0.0),
              (0.0, 0.0, -1.0))
        return (_halfturn_pair(rx, 0.0, 0.0, ht)
                + _halfturn_pair(ry, 0.0, 0.0, ht)
                + _halfturn_pair(rc, 3.0 * c / 2.0, 1.5, ht))
    if key == 'R2':
        rc = ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        return (_halfturn_pair(rx, 0.0, 0.0, ht)
                + _halfturn_pair(ry, 0.0, 0.0, ht)
                + _halfturn_pair(rc, 1.0, 1.0, ht))
    if key == 'I6':
        return [(ident, (2.0, 0.0, 0.0)), (ident, (0.0, 2.0, 0.0)),
                (rotz, (0.0, 0.0, ht))]
    if key == 'I8':
        return [(ident, (2.0, 0.0, 0.0)),
                (rotz, (1.0, 1.0, 0.0)),          # translation with twist
                (rotz, (0.0, 0.0, ht))]
    if key == 'I9':
        # I-9's third generator is the MIRROR-like ((0,1,0),(1,0,0),z),
        # not the rotation the other two use -- the datafile prints
        # "0 1 0 0   1 0 0 0" where I-6 and I-8 print "-1 0 0 0".
        swapxy = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        return [(ident, (2.0, 0.0, 0.0)), (ident, (0.0, 2.0, 0.0)),
                (swapxy, (0.0, 0.0, ht))]
    return None


def _mat(rot, t):
    M = np.eye(4)
    M[:3, :3] = np.asarray(rot, dtype=float)
    M[:3, 3] = np.asarray(t, dtype=float)
    return M


# Brakke's `view_transform_generators`, transcribed VERBATIM, and his
# cell words with them.
#
# This is the short way round, and it took far too long to take it.  The
# engine's own reflection generators are DERIVED -- classify the patch's
# boundary curves, guess which are mirrors or axes, rebuild the group --
# and for these surfaces that fails, because the group is not generated
# by reflections at all: it is translations and screws.  But these rows
# are relaxed on Brakke's OWN contours, so they already sit in his
# coordinate frame, and his matrices apply literally.  Nothing needs
# deriving.
#
# The matrices below are the datafile text with its parameters
# substituted (I-8 zsize 0.5, I-9 height 0.5, R-III HT 0.5), which are
# the same numbers `RING_SURFACES` above builds the contours with.
_R90M = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
_S3 = math.sqrt(3.0)
_RIII_A = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
_RIII_B = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
_RIII_C = ((-0.5, -_S3 / 2.0, 0.0), (-math.sqrt(0.75), 0.5, 0.0),
           (0.0, 0.0, -1.0))

RING_CELLS = {
    # I-8.fe: a x-translation, b a 1,1 translation WITH TWIST, c a screw
    # up the z axis.  `layers := { transform_expr "bac" }`.
    'I8': (lambda: {'a': _mat(np.eye(3), (2.0, 0.0, 0.0)),
                    'b': _mat(_R90M, (1.0, 1.0, 0.0)),
                    'c': _mat(_R90M, (0.0, 0.0, 0.5))},
           ('bac',)),
    # I-9.fe: two translations and a screw whose rotation part has
    # determinant -1 -- it is a rotoreflection, not a half-turn, and
    # copying the sign wrong is the difference between a clean cell and
    # 1380 over-shared edges.  `layers := { transform_expr "abc" }`.
    'I9': (lambda: {'a': _mat(np.eye(3), (2.0, 0.0, 0.0)),
                    'b': _mat(np.eye(3), (0.0, 2.0, 0.0)),
                    'c': _mat(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                               (0.0, 0.0, 1.0)), (0.0, 0.0, 0.5))},
           ('abc',)),
    # RIII.fe: six half-turns, one about each edge of the prism -- three
    # about the bottom triangle and the same three raised by 2*HT.
    # `layers` is a flat sheet, `stack12` a column.
    'R3': (lambda: {'a': _mat(_RIII_A, (0.0, 0.0, 0.0)),
                    'b': _mat(_RIII_B, (0.0, 0.0, 0.0)),
                    'c': _mat(_RIII_C, (3.0 * _S3 / 2.0, 1.5, 0.0)),
                    'd': _mat(_RIII_A, (0.0, 0.0, 1.0)),
                    'e': _mat(_RIII_B, (0.0, 0.0, 1.0)),
                    'f': _mat(_RIII_C, (3.0 * _S3 / 2.0, 1.5, 1.0))},
           ('fdfedfe', 'febfefe')),
}


def ring_cell_assemble(key, V, quads, tol=1e-4):
    """Assemble a ring surface with the datafile's own word, or None.

    Gated exactly like the reflection routes: duplicate faces,
    over-shared edges and connectedness, and a word that fails any of
    them is refused rather than shipped.
    """
    entry = RING_CELLS.get(key)
    if entry is None:
        return None
    lets, words = entry[0](), entry[1]
    V = np.asarray(V, dtype=float)
    quads = [tuple(f) for f in quads]
    for word in words:
        mats = eval_transform_expr(lets, word)
        pts = np.concatenate([V @ M[:3, :3].T + M[:3, 3] for M in mats])
        nV = len(V)
        faces = []
        for j, M in enumerate(mats):
            flip = float(np.linalg.det(M[:3, :3])) < 0.0
            for f in quads:
                g = tuple(i + j * nV for i in f)
                faces.append(g[::-1] if flip else g)
        span = float(np.max(pts.max(0) - pts.min(0))) or 1.0
        W, wf = _weld_points(pts, faces, tol * span)
        dup, over, comps = _orbit_defects(W, wf)
        if dup == 0 and over == 0 and comps == 1:
            return W, wf, len(mats), word
    return None


def ring_tile(V, quads, key, depth=2, tol=1e-7):
    """Close the group generated by `ring_generators` up to `depth`
    words and emit the welded orbit of the patch.

    The orbit is deduplicated by the transform, not by the mesh: two
    different words that give the same isometry would otherwise each
    contribute a copy, and coincident sheets are invisible to any
    topological count -- the failure that once rendered CLP-with-a-handle
    as leopard spots.
    """
    V = np.asarray(V, dtype=float)
    seen = {}
    ident = (np.eye(3), np.zeros(3))

    def keyof(R, t):
        return tuple(np.round(np.concatenate([R.ravel(), t]), 7))

    frontier = [ident]
    seen[keyof(*ident)] = ident
    raw = ring_generators(key)
    if raw is None:
        return V, [tuple(int(i) for i in q) for q in quads], 1
    gens = [(np.array(R, float), np.array(t, float)) for R, t in raw]
    for _ in range(depth):
        nxt = []
        for R0, t0 in frontier:
            for R1, t1 in gens:
                R = R1 @ R0
                t = R1 @ t0 + t1
                k = keyof(R, t)
                if k not in seen:
                    seen[k] = (R, t)
                    nxt.append((R, t))
        frontier = nxt
        if not frontier:
            break
    Vs, Qs, base = [], [], 0
    for R, t in seen.values():
        Vs.append(V @ R.T + t)
        flip = np.linalg.det(R) < 0.0
        for q in quads:
            f = tuple(int(i) + base for i in q)
            Qs.append(f[::-1] if flip else f)
        base += len(V)
    return np.concatenate(Vs, 0), Qs, len(seen)


def ring_tile_checked(V, quads, key, depth=2):
    """Tile, weld, and ACCEPT ONLY IF the result is a surface.

    The discriminator is DUPLICATE FACES, not over-shared edges, and the
    difference is the whole point.  These contours are rosettes: the
    boundary curve passes through the origin two or four times, so the
    assembled surface genuinely has lines where four sheets meet, at the
    origin and at every lattice translate of it.  Over-shared edges
    THERE are real geometry -- the same situation as the even-k
    Fischer-Koch surfaces, whose line self-intersection is a classical
    property rather than a meshing defect.  Rejecting on over-shared
    edges alone would therefore throw away correct tilings.

    Coincident duplicate COPIES are the actual failure, and they are
    what `hexagonal._assembly_ok` was written to catch after a stack of
    overlapping patches once shipped rendering as leopard spots.  They
    show up as faces sharing a centroid, which no component count and no
    edge count notices.

    Measured at depth 2, the two are cleanly separated:

        R-2, R-3, I-9   0 over-shared, 0 duplicate      -> accepted
        I-6          1390 over-shared, 368 duplicate    -> refused
        I-8            78 over-shared,   0 duplicate    -> refused

    I-6 fails on duplicates outright.  I-8 has none, but only 5% of its
    over-shared edges lie near a pinch line, so they are not the rosette
    self-touching either and its generator set is not yet right.  Both
    fall back to the single relaxed patch, which is honest: one annulus
    IS a piece of the surface, whereas a stack of overlapping copies is
    not.
    """
    try:
        VT, QT, n = ring_tile(V, quads, key, depth=depth)
    except Exception:
        return np.asarray(V, float), [tuple(int(i) for i in q)
                                      for q in quads], 1
    if n <= 1:
        return VT, QT, 1
    VT = np.asarray(VT, float)
    span = float(np.max(VT.max(0) - VT.min(0)))
    W, QQ = _weld_points(VT, QT, 1e-4 * max(span, 1e-12))

    cent = {}
    for f in QQ:
        c = tuple(np.round(W[list(f)].mean(0), 6))
        cent[c] = cent.get(c, 0) + 1
    if any(v > 1 for v in cent.values()):
        return np.asarray(V, float), [tuple(int(i) for i in q)
                                      for q in quads], 1

    ec = {}
    for f in QQ:
        m = len(f)
        for t in range(m):
            a, b = f[t], f[(t + 1) % m]
            e = (a, b) if a < b else (b, a)
            ec[e] = ec.get(e, 0) + 1
    # CONNECTEDNESS, for the same reason `hexagonal._assembly_ok` gained
    # it: I-8's depth-1 orbit welds into THREE pieces (12288 + 6144 +
    # 6144 vertices) while its single patch is one, and a surface that
    # falls apart is wrong however clean its edges are.
    nv = len(W)
    parent = list(range(nv))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for f in QQ:
        r0 = _find(int(f[0]))
        for j in range(1, len(f)):
            rj = _find(int(f[j]))
            if rj != r0:
                parent[rj] = r0
    if len({_find(int(i)) for f in QQ for i in f}) > 1:
        return np.asarray(V, float), [tuple(int(i) for i in q)
                                      for q in quads], 1

    over = [e for e, c in ec.items() if c > 2]
    if over:
        # tolerated only where they ARE the rosette's own pinch: within
        # a tenth of a cell of a lattice node of the 2x2 translations
        P = np.array([0.5 * (W[e[0]] + W[e[1]]) for e in over])
        d = np.hypot(((P[:, 0] + 1.0) % 2.0) - 1.0,
                     ((P[:, 1] + 1.0) % 2.0) - 1.0)
        if not np.all(d < 0.10):
            return np.asarray(V, float), [tuple(int(i) for i in q)
                                          for q in quads], 1
    return W, QQ, n


def _weld_points(V, faces, tol):
    """Merge vertices within `tol` and reindex."""
    keys = np.round(np.asarray(V, float) / max(tol, 1e-30)).astype(np.int64)
    uniq = {}
    remap = np.empty(len(V), dtype=np.int64)
    out = []
    for i, k in enumerate(map(tuple, keys)):
        j = uniq.get(k)
        if j is None:
            j = uniq[k] = len(out)
            out.append(V[i])
        remap[i] = j
    W = np.asarray(out, dtype=float)
    QQ = []
    for f in faces:
        g = [int(remap[i]) for i in f]
        d = [g[i] for i in range(len(g)) if g[i] != g[(i + 1) % len(g)]]
        if len(d) >= 3:
            QQ.append(tuple(d))
    return W, QQ


def group_slide_planes(slide, nrows):
    """Regroup per-plane slide entries by the ROWS they act on.

    A corner vertex constrained to two planes is one point on the LINE
    where they cross, not two independent single-plane problems.  Handled
    as the latter -- projecting onto one plane and then the other -- the
    two projections undo each other whenever the planes are oblique, and
    for these datafiles they meet at 45 degrees.  Evolver solves the
    whole set at a vertex jointly, through one small Gram system
    (`constr_proj` in its `cnstrnt.c`), and so does `project_rows` below.

    Returns `[(rows, [(vec, off), ...]), ...]`, one entry per distinct
    set of rows.
    """
    by_rows = {}
    for rows, vec, off in slide:
        key = (int(rows[0]), int(rows[-1]), len(rows))
        by_rows.setdefault(key, [np.asarray(rows), []])[1].append(
            (np.asarray(vec, dtype=float), float(off)))
    return [(v[0], v[1]) for v in by_rows.values()]


def project_rows(X, groups):
    """Put each row group on ALL of its planes at once.

    One Newton step on the joint system: with normals `n_i` and offsets
    `c_i`, solve `G a = r` where `G_ij = n_i . n_j` and `r_i = c_i -
    n_i . x`, then move by `sum a_i n_i`.  Linear constraints, so one
    step lands exactly; a couple more cost nothing and cover round-off.
    """
    for rows, planes in groups:
        if len(planes) == 1:
            vec, off = planes[0]
            X[rows] -= ((X[rows] @ vec - off)
                        / float(vec @ vec))[:, None] * vec
            continue
        N = np.array([p[0] for p in planes], dtype=float)
        c = np.array([p[1] for p in planes], dtype=float)
        G = N @ N.T
        for _ in range(3):
            r = c[None, :] - X[rows] @ N.T
            try:
                a = np.linalg.solve(G, r.T).T
            except np.linalg.LinAlgError:
                a = np.linalg.lstsq(G, r.T, rcond=None)[0].T
            X[rows] += a @ N
    return X


def project_rows_tangent(g, groups):
    """Remove from `g` the component normal to each group's planes.

    The velocity has to be tangent to EVERY constraint at that vertex
    before the vertex moves, which for two planes means along their
    intersection line -- not tangent to one and then the other.
    """
    for rows, planes in groups:
        N = np.array([p[0] for p in planes], dtype=float)
        N = N / np.linalg.norm(N, axis=1, keepdims=True)
        if len(planes) == 1:
            g[rows] -= (g[rows] @ N[0])[:, None] * N[0]
            continue
        G = N @ N.T
        try:
            a = np.linalg.solve(G, (g[rows] @ N.T).T).T
        except np.linalg.LinAlgError:
            a = np.linalg.lstsq(G, (g[rows] @ N.T).T, rcond=None)[0].T
        g[rows] -= a @ N
    return g


def fe_slide_planes(fe, arc_of, corners, nrows):
    """Boundary rows that are FREE on a plane, and the plane they run on.

    A datafile's boundary is not automatically a given contour.  An edge
    written `4  3 2 constraint 1` slides on the plane of constraint 1
    while the surface minimises, and a vertex written `1 ... constraints
    1 3` slides along the line where two planes cross.  Both are outputs
    of the solve, not inputs to it.

    Whole datafiles are built this way, not just the odd arc: Schwarz P,
    Schwarz D, Neovius and Schoen's I-WP pin NOTHING.  Each starts life
    as a flat quadrilateral -- pcell's four corners all have x = 0.5 --
    and every edge slides in its own mirror plane until the surface
    curves into shape.  Spanning that quadrilateral as if it were the
    contour returns the flat square it already is, which is exactly what
    those four cells were: a faceted plate, assembled 48 times into a
    star, passing every topological check.
    """
    slide = []
    at_corner = np.zeros(nrows, dtype=bool)
    at_corner[list(corners.values())] = True
    for eid, cons in sorted(getattr(fe, 'edge_constraints', {}).items()):
        if eid in getattr(fe, 'edge_fixed', ()):
            continue
        rows = np.nonzero((arc_of == eid) & ~at_corner)[0]
        if not len(rows):
            continue
        for n in cons:
            pl = fe.constraint_plane(n, fe.params)
            if pl is not None and pl[2] is None:
                slide.append((rows, pl[0], pl[1]))
    for vid, row in sorted(corners.items()):
        if vid in getattr(fe, 'vertex_fixed', ()):
            continue
        for n in getattr(fe, 'vertex_constraints', {}).get(vid, ()):
            pl = fe.constraint_plane(n, fe.params)
            if pl is not None and pl[2] is None:
                slide.append((np.array([row]), pl[0], pl[1]))
    return slide


def fe_grid(fe, m=96, rings=14):
    """Span the datafile's boundary, carrying its arc and corner labels.

    Returns `(V, quads, fixed, arc_of, corners, rims)`.  One boundary
    loop is a disk and two are an annulus; the labels are what let the
    constraints reach the right part of the boundary afterwards.
    """
    chains = fe.boundary_chains()
    if not chains or len(chains) > 2:
        return None
    lps, arcs, cors = [], [], []
    for vids, eids in chains:
        poly = np.asarray([fe.vertices[v] for v in vids], dtype=float)
        pts, idx, cor = resample_indexed(poly, m)
        lps.append(pts)
        arcs.append(np.asarray([eids[i] for i in idx], dtype=np.int64))
        cors.append({vids[k]: int(cor[k]) for k in range(len(vids))})
    if len(lps) == 1:
        V, quads, fixed = build_disk_grid(lps[0], rings)
        arc_of = np.full(len(V), -1, dtype=np.int64)
        arc_of[:m] = arcs[0]
        return (V, quads, fixed, arc_of, cors[0], [(0, m)])
    (la, lb), (aa, ab), (ca, cb) = lps, arcs, cors
    if float(np.mean(la[:, 2])) > float(np.mean(lb[:, 2])):
        la, lb, aa, ab, ca, cb = lb, la, ab, aa, cb, ca
    j = int(np.argmin(np.linalg.norm(lb - la[0], axis=1)))
    lb, ab = np.roll(lb, -j, axis=0), np.roll(ab, -j)
    cb = {v: (i - j) % m for v, i in cb.items()}
    V, quads, fixed = build_annulus_grid(la, lb, rings)
    nb = len(V) - m
    arc_of = np.full(len(V), -1, dtype=np.int64)
    arc_of[:m], arc_of[nb:] = aa, ab
    corners = dict(ca)
    corners.update({v: nb + i for v, i in cb.items()})
    return (V, quads, fixed, arc_of, corners, [(0, m), (nb, nb + m)])


def fe_pinned_patch(source, m=96, rings=14, iters=300):
    """Relax a NON-adjoint datafile's surface, honouring its constraints.

    The counterpart of `fe_adjoint_patch` for the files that define their
    surface directly rather than as a conjugate.  It is not simply "span
    the polygon": see `fe_slide_planes` for why four of these datafiles
    hand you a flat square and expect the solve to curve it.
    """
    from . import fedata
    fe = fedata.read(source) if isinstance(source, str) else source
    got = fe_grid(fe, m=m, rings=rings)
    if got is None:
        return None
    V, quads, fixed, arc_of, corners, rims = got
    V = np.asarray(V, dtype=float)
    fixed = np.asarray(fixed, dtype=bool)
    T = np.asarray(_quads_to_tris(quads))
    # Hold the volume, but ONLY where our volume functional can be shown
    # to be the one the datafile means.  The test is the datafile's own
    # starting configuration: Brakke sets these up already satisfying
    # their volume constraint, so if our number does not reproduce the
    # declared target there, we are computing some other quantity and
    # driving it would do more harm than leaving it alone.  pcell
    # reproduces 1/12 exactly; Neovius states its volume through a custom
    # `quantity` integrand we do not model yet, and misses, so it stays
    # on the sliding solve -- where it is already within 5% of Evolver.
    slide = fe_slide_planes(fe, arc_of, corners, len(V))
    target = fe.body_volume() if slide else None
    if target is not None:
        v0 = fe_volume(fe, V, T, arc_of)
        if abs(v0 - target) <= 1e-3 * max(1.0, abs(target)):
            V = minimize_area_at_volume(fe, V, T, fixed, slide, arc_of,
                                        float(target),
                                        outer_iters=max(1600, iters * 5))
            return (V, [tuple(f) for f in quads],
                    [np.array(V[a:b]) for a, b in rims])
    if slide:
        # Generous, because this loop exits as soon as the line search
        # stops improving.  The old cap of `iters // 6` was stopping the
        # descent while it was still moving: Schwarz P needed about 1500
        # passes to reach Evolver's answer and was being given 50, which
        # left it 14% high with visibly flat panels.  Schoen's I-WP,
        # Schwarz D and Neovius were all short in the same way.
        V = minimize_area_sliding(V.copy(), T, fixed, slide,
                                  outer_iters=max(1500, iters * 4))
    else:
        V = np.asarray(minimize_area(V.copy(), T, fixed, outer_iters=iters),
                       dtype=float)
    # The SOLVED boundary, not the datafile's.  For a file whose boundary
    # slides, the loop it starts from is not the loop it ends on -- it is
    # the flat quadrilateral the solve exists to curve -- so baking the
    # declared loops and re-spanning them at runtime threw the whole
    # solve away and handed back the flat plate again.  Schwarz P, D,
    # Neovius, I-WP and C(D) all shipped that way even after the solver
    # itself was right.
    return V, [tuple(f) for f in quads], [np.array(V[a:b]) for a, b in rims]


def fe_adjoint_patch(source, m=96, rings=14, iters=300, relax_iters=120):
    """Rebuild an ADJOINT datafile's cell the way the datafile does it.

    Roughly half of Brakke's periodic collection is written as an
    adjoint: the datafile pins a polygon, and the surface you want is the
    CONJUGATE of the disk that polygon spans.  Conjugating is only the
    first step, though, and the steps after it are what a static reading
    of the file misses.  The conjugate is minimal but arrives at an
    arbitrary position and scale, so each file carries a `frame` command
    that moves it -- reads a corner height here, a minimum there,
    translates, rescales -- and only then declares which boundary arc
    lies on which mirror plane.

    So this runs the file: span, conjugate, execute `frame`, land each
    arc on the plane `frame` assigned it, and relax the interior back.
    Reproducing the placement by inspection instead, or matching arcs to
    constraints by their order, is what left most of this collection
    coming out as disjoint sheets -- Evolver evolves all of these
    without complaint, so a refusal was always a defect here.

    Returns `(points, quads, letters, env, loops)`; `loops` is the
    conjugate's own boundary, which is what gets baked, because a
    contour is all the shipped code needs to span the patch again.
    """
    from . import fedata
    fe = fedata.read(source) if isinstance(source, str) else source
    chains = fe.boundary_chains()
    if not chains or len(chains) > 2:
        return None
    lps, arcs, cors = [], [], []
    for vids, eids in chains:
        poly = np.asarray([fe.vertices[v] for v in vids], dtype=float)
        pts, idx, cor = resample_indexed(poly, m)
        lps.append(pts)
        arcs.append(np.asarray([eids[i] for i in idx], dtype=np.int64))
        cors.append({vids[k]: int(cor[k]) for k in range(len(vids))})
    if len(lps) == 1:
        V, quads, fixed = build_disk_grid(lps[0], rings)
        arc_of = np.full(len(V), -1, dtype=np.int64)
        arc_of[:m] = arcs[0]
        corners = cors[0]
        rims = [(0, m)]
    else:
        (la, lb), (aa, ab), (ca, cb) = lps, arcs, cors
        if float(np.mean(la[:, 2])) > float(np.mean(lb[:, 2])):
            la, lb, aa, ab, ca, cb = lb, la, ab, aa, cb, ca
        j = int(np.argmin(np.linalg.norm(lb - la[0], axis=1)))
        lb, ab = np.roll(lb, -j, axis=0), np.roll(ab, -j)
        cb = {v: (i - j) % m for v, i in cb.items()}
        V, quads, fixed = build_annulus_grid(la, lb, rings)
        nb = len(V) - m
        arc_of = np.full(len(V), -1, dtype=np.int64)
        arc_of[:m], arc_of[nb:] = aa, ab
        corners = dict(ca)
        corners.update({v: nb + i for v, i in cb.items()})
        rims = [(0, m), (nb, nb + m)]
    V = np.asarray(V, dtype=float)
    fixed = np.asarray(fixed, dtype=bool)
    T = np.asarray(_quads_to_tris(quads))

    # Arcs the datafile leaves FREE on a plane are not part of the given
    # contour -- their shape comes out of the solve, and a minimal
    # surface meets such a plane at a right angle.  Pinning them instead
    # spans a different surface entirely: for Schoen's batwing that put
    # the patch half again too large in area, on a cell that still welded
    # into one clean sheet and passed every topological check.
    slide = fe_slide_planes(fe, arc_of, corners, len(V))
    if slide:
        V = minimize_area_sliding(V.copy(), T, fixed, slide,
                                  outer_iters=max(20, iters // 8))
    else:
        V = np.asarray(minimize_area(V.copy(), T, fixed, outer_iters=iters),
                       dtype=float)
    W0 = np.asarray(discrete_adjoint(V, T, bangle=90.0, mode='corner'),
                    dtype=float)

    # The discrete conjugate is fixed only up to a TRANSLATION and a
    # SIGN, and the datafiles assume Evolver's choice of both.
    #
    # The translation, because Polthier's construction propagates from a
    # seed edge that is simply declared to sit at the origin, and Evolver
    # picks that seed by mesh numbering after refinement.  Several frames
    # lean on it -- the first three triplanes normalise only `z - x` and
    # then divide by `max(z)`, which still depends on where x happens to
    # sit -- so inheriting an arbitrary origin made those patches come
    # out several times too large while still assembling cleanly.
    #
    # The sign, because a Bonnet rotation through 90 and through 270
    # degrees give point reflections of each other, both minimal and both
    # congruent to the conjugate.  The frames measure a signed length and
    # divide by it, so the wrong choice divides by a NEGATIVE number: the
    # tell for Schoen's mantas and the disphenoids, which came out
    # mirrored and mis-scaled.
    #
    # Neither has to be guessed.  By Schwarz each straight segment of the
    # pinned contour conjugates to an arc in the plane perpendicular to
    # it, and the datafile says which plane; the planes through the
    # origin are scale-free and fix the translation on their own, and of
    # the two signs the right one is the one whose boundary then actually
    # lands on the planes.
    best = None
    for sign in (1.0, -1.0):
        W = sign * W0
        _junk, econ, _junk2 = fe.run_frame(W.copy(), corners, arc_of)
        rows_a, rows_b = [], []
        for eid, cons in sorted(econ.items()):
            rows = np.nonzero(arc_of == eid)[0]
            if not len(rows):
                continue
            for n in cons:
                pl = fe.constraint_plane(n, fe.params)
                if pl is None or pl[2] is not None or abs(pl[1]) > 1e-12:
                    continue
                rows_a.append(pl[0])
                rows_b.append(-float(np.mean(W[rows] @ pl[0])))
        if rows_a:
            shift, _r, _rk, _s = np.linalg.lstsq(
                np.asarray(rows_a), np.asarray(rows_b), rcond=None)
            W = W + shift
        W, econ, env = fe.run_frame(W, corners, arc_of)
        if not np.all(np.isfinite(W)):
            # A frame divides by a quantity it measures off the
            # conjugate.  If that came out zero the placement is
            # meaningless, and going on would only launder NaNs into a
            # shipped surface.
            continue

        # Land each arc on the plane `frame` gave it.  Where the datafile
        # leaves the offset to be measured it says so by naming it, and
        # the name is bound here from the conjugate itself -- the one
        # number Evolver could not know before conjugating either.
        planes = {}
        for eid, cons in sorted(econ.items()):
            rows = np.nonzero(arc_of == eid)[0]
            if not len(rows):
                continue
            for n in cons:
                pl = fe.constraint_plane(n, env)
                if pl is None:
                    continue
                vec, const, name = pl
                if name is None:
                    off = const
                elif name in env:
                    off = float(env[name]) + const
                else:
                    off = float(np.median(W[rows] @ vec))
                    env[name] = off - const
                planes[n] = (vec, off)

        # How far the conjugate already sits from the planes it is meant
        # to meet.  This is the honest measure of whether the whole
        # reconstruction worked: the conjugate of a minimal surface is
        # minimal, and if the framing put it in the right place its
        # boundary is ALREADY on those planes.  A large residual means
        # the placement, the sign or the arc-to-plane mapping is wrong,
        # and projecting anyway would hide that by deforming a correct
        # surface into a plausible-looking wrong one.
        span = float(np.max(W.max(0) - W.min(0))) or 1.0
        resid = 0.0
        for eid, cons in sorted(econ.items()):
            rows = np.nonzero(arc_of == eid)[0]
            if not len(rows):
                continue
            for n in cons:
                if n not in planes:
                    continue
                vec, off = planes[n]
                d = np.abs(W[rows] @ vec - off) / float(np.linalg.norm(vec))
                resid = max(resid, float(np.max(d)) / span)
        if best is None or resid < best[0]:
            best = (resid, W, econ, env, planes)
    if best is None:
        return None
    resid, W, econ, env, planes = best

    if planes:
        # An arc named by two constraints belongs on the LINE where they
        # meet, and alternating the two projections converges to it.
        for _ in range(40):
            for eid, cons in sorted(econ.items()):
                rows = np.nonzero(arc_of == eid)[0]
                if not len(rows):
                    continue
                for n in cons:
                    if n not in planes:
                        continue
                    vec, off = planes[n]
                    W[rows] -= (((W[rows] @ vec) - off)
                                / float(vec @ vec))[:, None] * vec
        W = np.asarray(minimize_area(W.copy(), T, fixed,
                                     outer_iters=relax_iters), dtype=float)
    lets = dict(zip(fe.gen_names, fe.generators_with(env)))
    loops = [np.array(W[a:b], dtype=float) for a, b in rims]
    env = dict(env)
    env['_plane_residual'] = resid
    return W, [tuple(f) for f in quads], lets, env, loops


def fe_cell_patch(key, m=96, rings=16, iters=300):
    """Relax the datafile's contour, as a disk or an annulus.

    One boundary loop is a disk; two are an annulus.  The distinction is
    read off the datafile, not assumed: I-6's surface is the side wall of
    a prism, so its boundary is a bottom rosette and a top one, and
    spanning only the first gives a flat degenerate patch.
    """
    from .fecells import FE_CELLS
    spec = FE_CELLS[key]
    loops = [np.asarray(l, dtype=float) for l in spec['loops']]
    if len(loops) == 1:
        poly = loops[0]
        lp = resample_loop(np.vstack([poly, poly[:1]]), m)
        V, quads, fixed = build_disk_grid(lp, rings)
    elif len(loops) == 2:
        a, b = loops
        la = resample_loop(np.vstack([a, a[:1]]), m)
        lb = resample_loop(np.vstack([b, b[:1]]), m)
        if float(np.mean(la[:, 2])) > float(np.mean(lb[:, 2])):
            la, lb = lb, la
        j = int(np.argmin(np.linalg.norm(lb - la[0], axis=1)))
        lb = np.roll(lb, -j, axis=0)
        V, quads, fixed = build_annulus_grid(la, lb, rings)
    else:
        return None
    V = np.asarray(V, dtype=float)
    fixed = np.asarray(fixed, dtype=bool)
    T = np.asarray(_quads_to_tris(quads))
    V = np.asarray(minimize_area(V.copy(), T, fixed, outer_iters=iters),
                   dtype=float)
    return V, [tuple(f) for f in quads]


def dedupe_placements(V, mats):
    """Group elements that put THIS patch in the same place, kept once.

    Not the same question as which matrices are distinct, and the two
    part company whenever the patch is stabilised by part of its own
    group.  Fischer and Koch's S cell is the extreme case: its twelve
    generators give 380 distinct elements but far fewer distinct
    positions for the fundamental piece, so one copy per element laid
    181,870 faces exactly on top of one another -- a pile of sheets that
    every check then reported as broken.

    Two probes rather than one, because a rotation about the patch's own
    centroid moves the patch while leaving the centroid where it was,
    and a centroid-only key would merge two genuinely different copies.
    """
    V = np.asarray(V, dtype=float)
    if not len(V):
        return list(mats)
    c0 = V.mean(0)
    far = V[int(np.argmax(np.linalg.norm(V - c0, axis=1)))]
    seen, keep = set(), []
    for M in mats:
        R, t = M[:3, :3], M[:3, 3]
        k = (tuple(np.round(c0 @ R.T + t, 6))
             + tuple(np.round(far @ R.T + t, 6)))
        if k in seen:
            continue
        seen.add(k)
        keep.append(M)
    return keep


def assemble_orbit(V, quads, mats, tol=1e-4):
    """Lay the patch down under every group element and weld the result.

    A negative determinant reverses the copy's faces, so the assembled
    sheet keeps one consistent orientation instead of alternating with
    each reflection.

    The group elements are deduplicated a second time, by WHERE THEY PUT
    THIS PATCH rather than by what matrix they are.  The two are not the
    same question whenever the patch is stabilised by part of its own
    group, and for Fischer and Koch's cells they differ enormously:
    S has twelve generators whose 380 distinct elements place the patch
    in only a fraction of that many positions, so assembling one copy per
    element laid 181,870 faces exactly on top of each other.
    """
    V = np.asarray(V, dtype=float)
    mats = dedupe_placements(V, mats)
    pts = np.concatenate([V @ M[:3, :3].T + M[:3, 3] for M in mats])
    nV = len(V)
    faces = []
    for j, M in enumerate(mats):
        flip = float(np.linalg.det(M[:3, :3])) < 0.0
        for f in quads:
            g = tuple(i + j * nV for i in f)
            faces.append(g[::-1] if flip else g)
    span = float(np.max(pts.max(0) - pts.min(0))) or 1.0
    return _weld_points(pts, faces, tol * span)


# Weld tolerances to try, tightest first.
#
# The tolerance is doing two jobs at once and they pull opposite ways.
# Copies joined across an arc the transform fixes POINTWISE -- a mirror
# --- meet to machine precision, so a tight weld is correct there; copies
# joined by a translation meet arc-to-arc between samples that were laid
# down independently, and need a loose one.  A single value cannot serve
# both: at 1e-4 several of the triplane cells fused sheets that merely
# pass close through the middle of the cell, reporting edges shared by
# four faces on a surface that is perfectly sound.  So the tightest weld
# that closes is the one taken, and the gate decides whether it closed.
FE_WELD_LADDER = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)

# Datafile commands that show a whole cell, best first.
FE_WORD_PREFER = ('showcube', 'cube', 'full', 'showrhombic', 'showcubelet',
                  'showwprism', 'showgprism', 'layers', 'showlayer', 'seven',
                  'showfour', 'showsix', 'stack8', 'stack6', 'stack4',
                  'stack12')


def fe_word_order(words):
    """The datafile's cell commands, best first."""
    return ([k for k in FE_WORD_PREFER if k in words]
            + sorted(k for k in words if k not in FE_WORD_PREFER))


def fe_words_with_fallback(fe):
    """The datafile's cell words, plus the one it leaves implicit.

    Fischer and Koch's C(S) and Y cells declare their generators and no
    `transform_expr` at all: the datafile's instructions say to turn
    `transforms on`, which shows the orbit under EVERY declared
    generator.  That implicit word is the concatenation of all the
    letters, and without it those two surfaces have no cell to build.
    """
    words = dict(fe.words)
    if not words and fe.gen_names:
        words['transforms on'] = ''.join(fe.gen_names)
    return words


def fe_orbit_ok(W, wf):
    """One clean, closed, non-degenerate sheet?"""
    dup, over, comps = _orbit_defects(W, wf)
    bb = np.asarray(W).max(0) - np.asarray(W).min(0)
    return (dup == 0 and over == 0 and comps == 1
            and float(np.min(bb)) > 1e-6)


def fe_adjoint_cell(source, m=96, rings=14, iters=300):
    """Run an adjoint datafile end to end and return the cell that closes.

    Tries the file's own cell commands in turn and, for each, the weld
    ladder; the first combination that welds into a single clean sheet
    wins.  Returns None rather than a guess -- a heap of disjoint sheets
    is worse than no row at all.
    """
    from . import fedata
    fe = fedata.read(source) if isinstance(source, str) else source
    got = fe_adjoint_patch(fe, m=m, rings=rings, iters=iters)
    if got is None:
        return None
    V, quads, lets, env, loops = got
    words = fe_words_with_fallback(fe)
    for name in fe_word_order(words):
        word = words[name]
        if any(c not in lets for c in word):
            continue
        mats = eval_transform_expr(lets, word)
        if len(mats) > 2048:
            continue
        mats = dedupe_placements(V, mats)
        if not (1 < len(mats) <= 256):
            continue
        for tol in FE_WELD_LADDER:
            W, wf = assemble_orbit(V, quads, mats, tol)
            if fe_orbit_ok(W, wf):
                return {'fe': fe, 'patch': V, 'quads': quads,
                        'letters': lets, 'env': env, 'loops': loops,
                        'command': name, 'word': word, 'copies': len(mats),
                        'tol': tol, 'cell': (W, wf),
                        'bbox': np.asarray(W).max(0) - np.asarray(W).min(0)}
    return None


def fe_cell_assemble(key, V, quads, tol=None):
    """Assemble with the datafile's own generators and word.

    Gated like every other route: duplicate faces, over-shared edges and
    connectedness, and a refusal ships the bare patch rather than a pile
    of sheets.
    """
    from .fecells import FE_CELLS
    spec = FE_CELLS[key]
    lets = {k: np.asarray(v, dtype=float)
            for k, v in spec['letters'].items()}
    mats = eval_transform_expr(lets, spec['word'])
    W, wf = assemble_orbit(V, quads, mats,
                           spec.get('tol', 1e-4) if tol is None else tol)
    if not fe_orbit_ok(W, wf):
        return None
    return W, wf, len(mats)


def fe_cell_build(key, cells, res_per_cell, scale, theta):
    """TPMS_EXACT-compatible wrapper for a datafile-derived cell."""
    res = max(8, int(res_per_cell))
    V, quads = fe_cell_patch(key, m=max(48, 3 * res), rings=max(8, res // 2))
    if int(cells) > 1 if np.isscalar(cells) else True:
        got = fe_cell_assemble(key, V, quads)
        if got is not None:
            V, quads = got[0], got[1]
    V = np.asarray(V, dtype=float)
    if len(V):
        V = V - 0.5 * (V.max(0) + V.min(0))
        span = float(np.max(V.max(0) - V.min(0)))
        if span > 1e-12:
            V = V * (2.0 * scale / span)
    if theta:
        ct, st = math.cos(theta), math.sin(theta)
        R = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        V = V @ R.T
    return V, quads


def ring_build(key, cells, res_per_cell, scale, theta):
    """TPMS_EXACT-compatible wrapper: relax, then centre and fit the
    result into the 2*scale cube the rest of the catalog uses.

    `cells` selects how many words of the symmetry group to close, and
    the tiling is accepted only if it welds into a surface (see
    `ring_tile_checked`); otherwise the single relaxed patch ships.
    `theta` is accepted and ignored -- it is meaningless here because
    there is no holomorphic family to rotate, which is precisely the
    difference between this route and the exact Weierstrass rows.
    """
    m = max(48, int(round(res_per_cell)) * 2)
    V, quads, _area = ring_surface(key, m=m, rows=max(10, m // 6))
    # `cells` drives how many words of the symmetry group to close, so
    # asking for more cells genuinely extends the surface instead of
    # being ignored -- but only where the tiling verifies.
    depth = int(np.clip(int(cells) if np.isscalar(cells) else
                        max(cells), 1, 3))
    # Brakke's own word first, where the datafile gives one.  It beats
    # the derived reflection tiling outright: these patches are relaxed
    # on HIS contours, so they already sit in his coordinate frame and
    # his generator matrices apply literally, with nothing to classify
    # and nothing to guess.
    cell = ring_cell_assemble(key, V, quads) if depth > 1 else None
    if cell is not None:
        V, quads = cell[0], cell[1]
    else:
        V, quads, _n = ring_tile_checked(V, quads, key, depth=depth)
    V = np.asarray(V, dtype=float)
    lo, hi = V.min(0), V.max(0)
    V = V - 0.5 * (lo + hi)
    ext = float(np.max(hi - lo))
    if ext > 1e-12:
        V = V * (2.0 * float(scale) / ext)
    return V, [tuple(int(i) for i in q) for q in quads]


def ring_surface(key, m=120, rows=20, iters=150):
    """Relax the minimal annulus spanning two parallel congruent
    triangles.  Returns (V, quads, area)."""
    _label, tri, ht = RING_SURFACES[key]
    pts = np.array([(x, y, 0.0) for x, y in tri], dtype=float)
    loop = resample_loop(np.vstack([pts, pts[:1]]), m)
    top = loop.copy()
    top[:, 2] = ht
    V, quads, fixed = build_annulus_grid(loop, top, rows)
    T = _quads_to_tris(quads)
    V = np.asarray(minimize_area(np.asarray(V, float).copy(), T, fixed,
                                 outer_iters=iters), dtype=float)
    return V, quads, mesh_area(V, T)


# ---------------------------------------------------------------------------
# Conjugate-Plateau surfaces (Ken Brakke's `*adj.fe` datafiles)
#
# Every one of Brakke's adjoint datafiles states the SAME problem, and it
# is worth saying plainly because it looks like twenty problems:
#
#   1. a small fixed space polygon (4-8 corners) spanned by one face --
#      an ordinary pinned-boundary Plateau problem, which `minimize_area`
#      already solves;
#   2. conjugate the result (Bonnet angle 90);
#   3. each boundary arc of the CONJUGATE lies in a plane.  The plane's
#      NORMAL is exact and comes from the surface's symmetry; only its
#      offset is data, and Brakke reads that off the conjugate itself
#      (`rhs1 := avg(edge ee where original==1, avg(ee.vertex,x))`);
#   4. reflect in those planes to build the periodic surface.
#
# Revision 2 of the implementation plan said these needed "constraint
# planes with free-sliding vertices".  That is half right, and the half
# matters.  Step 1 needs nothing of the sort -- the polygon is entirely
# fixed.  The constraint planes belong to step 3, where they are not
# constraints imposed on a solve but a PROPERTY of the conjugate, then
# measured.
#
# CHECKED AGAINST EVOLVER ITSELF.  Brakke's program runs headlessly
# (`evolver64 -f script.txt GW5adj.fe`), so `GW5adj.fe` was evolved and
# conjugated by Evolver and the result compared with what this module
# builds from the same contour:
#
#     quantity          Evolver        here        differs by
#     patch area        1.551728       1.550341    0.089%
#     bbox x/y/z        1.008/1.080/1.872  1.010/1.085/1.867  0.2-0.5%
#     z-mirror gap      1.87217        1.86737     0.257%
#
# Evolver also reports rhs3 == rhs6 exactly, as this module measures
# independently.  Agreement at that level is the evidence that the route
# below is Brakke's route and not merely something that produces a
# plausible surface.
#
# Two things learned by measuring, each of which cost a rebuild:
#
# * The conjugate's boundary is planar only to ~2e-3 (the discrete
#   minimality defect), and reflected copies at that accuracy DO NOT
#   WELD -- 9 vertices merged out of 14406, against ~120 when it is
#   right.  So the boundary is PROJECTED onto its exact planes and the
#   interior re-solved against it.  The projection moves points by
#   ~2e-3 and costs 0.07% of the area.
#
# * Do NOT then let the boundary slide in those planes to re-minimize.
#   That is the obvious next idea, and it destroys the surface: the
#   free-boundary problem has no minimum, the sheet just shrinks along
#   the planes, measured here from area 2.518 down to 0.676.  Brakke
#   reaches a SADDLE (`hessian_seek`), not a minimum, and Evolver's own
#   documentation warns about precisely this.  Until a saddle-point
#   solver exists here, projection alone is the honest step.

def gw_poly(length1):
    """GW's contour at a given prism height (GW5adj.fe's `length1`).

    GW is a one-parameter FAMILY, not a single surface, and Brakke's page
    shows six members: "as a parameter is varied from small (left) to
    large (right), the surface goes from horizontal parallel sheets with
    catenoid connections to pairs of vertical sheets in a hexagonal
    layout with cross-tunnels at the junctions".  The datafile ships
    length1 = 0.5, which is his `gw.5` figure -- comparing against `gw.1`
    shows visibly larger holes, and that is a different member, not an
    error in the surface.
    """
    h = float(length1)
    return [(0.0, 0.0, 0.0),
            (SQRT3, -1.0, 0.0),
            (SQRT3, -1.0, h),
            (SQRT3 / 2.0, -1.5, h),
            (SQRT3 / 2.0, 0.5, h),
            (SQRT3 / 2.0, 0.5, 0.0)]


GW_LENGTH_DEFAULT = 0.5


def gw_params(length1=GW_LENGTH_DEFAULT):
    """Set GW's prism height, the way `clp_params` sets CLP's moduli.

    The spec table is the only channel a TPMS_EXACT row has for a shape
    parameter, so this rewrites GW's contour in place before a build.
    """
    spec = CONJUGATE_SURFACES['GW']
    spec['poly'] = gw_poly(length1)
    spec['length1'] = float(length1)
    return spec


CONJUGATE_SURFACES = {
    'GW': {
        'name': "Schoen GW (graphite-wurtzite)",
        # GW5adj.fe, data supplied to Brakke by Alan Schoen, 2 May 2008.
        # `length1` defaults to 0.5, matching his `gw.5` figure; see
        # `gw_poly` on why the family matters.
        'length1': 0.5,
        'poly': [(0.0, 0.0, 0.0),
                 (SQRT3, -1.0, 0.0),
                 (SQRT3, -1.0, 0.5),
                 (SQRT3 / 2.0, -1.5, 0.5),
                 (SQRT3 / 2.0, 0.5, 0.5),
                 (SQRT3 / 2.0, 0.5, 0.0)],
        'normals': [(SQRT3, -1.0, 0.0), (0.0, 0.0, 1.0),
                    (-SQRT3, -1.0, 0.0), (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0), (-SQRT3, -1.0, 0.0)],
        # `rhs6 := rhs3` in the datafile.  Not taken on trust: measured
        # independently here the two arcs give -0.14968 and -0.14965.
        'same': {5: 2},
        # Brakke names one generator per constraint, a..f = arcs 0..5,
        # and GW5adj.fe uses their MEASURED offsets directly -- unlike
        # the hybrid files it never translates the surface to a
        # canonical pose first, which is the proof that offsets can live
        # inside the generators.
        'letters': {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5},
        'words': ('baca', 'adacada'),   # showwprism (12), showlayer (42)
        'evolver_area': 1.551728181162,
    },
}


# Brakke's three hybrids (HRHTadj.fe, HTTRadj.fe, TRHTadj.fe).
#
# THEY ARE NOT ONE PROBLEM WITH ONE PARAMETER.  That assumption was made
# here first and it is wrong in three separate ways: the contours differ
# (HRHT puts its free corner at (-alpha,0,*), HTTR at (0,-alpha,*), TRHT
# at (-alpha/2, -(1-alpha)*sqrt(3)/2, *)), the six constraint normals
# differ, and WHICH constraint each generator letter names differs.
# Building all three from HRHT's geometry gave patch areas 0.9645 /
# 0.9338 / 1.4064 against Evolver's 0.9657 / 0.6623 / 0.5893 -- the first
# right and the other two not remotely.  Each is transcribed separately
# below.
#
# The letter -> arc maps are not guesswork either.  Brakke's generators
# are written for a pose in which some mirrors pass through the origin,
# and his comments name those as "z=0 mirror" and "y=0 mirror" without
# saying which constraint they are.  Running each datafile in Evolver and
# printing rhs1..rhs6 after `adj` settles it, because exactly those
# constraints come out zero:
#
#     HRHTadj  rhs = 2.1187  0  2.1187  0  0.8697  0
#     HTTRadj  rhs = 1.6341  0  0       0  0.5056  0
#     TRHTadj  rhs = 0       1.4858  0   0.4333  0  0
#
# so HRHT's origin mirrors are constraints 2/4/6, HTTR's are 2/3/4/6, and
# TRHT's are 1/3/5/6.  Offsets are used as measured rather than
# normalised away; GW5adj proves that works, since it never canonicalises
# its pose at all.

CONJUGATE_SURFACES['HT_HR'] = {
    'name': 'Schoen H\'-T | H"-R hybrid',
    'alpha': 0.12126744808636576,
    'poly': [(0.0, 0.0, 0.0),
             (-0.12126744808636576, 0.0, 0.0),
             (-0.12126744808636576, 0.0, 1.0),
             (-0.5, 0.0, 1.0),
             (0.0, -SQRT3 / 2.0, 1.0),
             (0.0, -SQRT3 / 2.0, 0.0)],
    'normals': [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
                (0.5, -SQRT3 / 2.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)],
    'same': {2: 0},                        # rhs3 := rhs1
    'letters': {'a': 1, 'b': 5, 'c': 3, 'd': 4, 'e': 0},
    'words': ('abcbcbc', 'abcbcbcecbcbc'),
    'evolver_area': 0.965668985,
}

CONJUGATE_SURFACES['TR_HT'] = {
    'name': 'Schoen T\'-R\' | H\'-T hybrid',
    'alpha': 0.07562891619932040,
    'poly': [(0.0, 0.0, 0.0),
             (-0.5, 0.0, 0.0),
             (-0.5, 0.0, 1.0),
             (0.0, -SQRT3 / 2.0, 1.0),
             (0.0, -0.07562891619932040, 1.0),
             (0.0, -0.07562891619932040, 0.0)],
    'normals': [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.5, -SQRT3 / 2.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)],
    'letters': {'a': 1, 'b': 3, 'c': 2, 'd': 4, 'e': 0},
    'words': ('abcbcbc', 'abcbcbcecbcbc'),
    'evolver_area': 0.662341950,
}

_TRHT_A = 0.73119569743699331
CONJUGATE_SURFACES['HR_TR'] = {
    'name': 'Schoen H"-R | T\'-R\' hybrid',
    'alpha': _TRHT_A,
    'poly': [(0.0, 0.0, 0.0),
             (0.0, 0.0, 1.0),
             (-0.5, 0.0, 1.0),
             (-0.5 * _TRHT_A, -(1.0 - _TRHT_A) * SQRT3 / 2.0, 1.0),
             (-0.5 * _TRHT_A, -(1.0 - _TRHT_A) * SQRT3 / 2.0, 0.0),
             (0.0, -SQRT3 / 2.0, 0.0)],
    'normals': [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.5, -SQRT3 / 2.0, 0.0),
                (0.0, 0.0, 1.0), (0.5, -SQRT3 / 2.0, 0.0), (0.0, 1.0, 0.0)],
    'letters': {'a': 0, 'b': 5, 'c': 2, 'd': 3, 'e': 1},
    'words': ('abcbcbc', 'abcbcbcecbcbc'),
    'evolver_area': 0.589273118,
}


def _conj_segment_ids(P, poly):
    """Label each point by the contour segment it sits on."""
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    best = np.full(len(P), np.inf)
    sid = np.zeros(len(P), dtype=np.int64)
    for k in range(n):
        a = poly[k]
        d = poly[(k + 1) % n] - a
        t = np.clip(((P - a) @ d) / max(float(d @ d), 1e-300), 0.0, 1.0)
        dist = np.linalg.norm(P - (a + t[:, None] * d), axis=1)
        hit = dist < best
        best = np.where(hit, dist, best)
        sid = np.where(hit, k, sid)
    return sid


def conjugate_patch(key, m=120, rings=20, iters=400):
    """Solve the Plateau problem, conjugate it, and land its boundary
    exactly on the mirror planes.

    Returns `(V, quads, arc_planes)` with one `(unit normal, offset)`
    per boundary arc, in the datafile's own constraint order -- the
    assembly words index generators by that order.
    """
    spec = CONJUGATE_SURFACES[key]
    poly = np.asarray(spec['poly'], dtype=float)
    loop = resample_loop(np.vstack([poly, poly[:1]]), m)
    V, quads, fixed = build_disk_grid(loop, rings)
    V = np.asarray(V, dtype=float)
    fixed = np.asarray(fixed, dtype=bool)
    T = np.asarray(_quads_to_tris(quads))
    V = np.asarray(minimize_area(V.copy(), T, fixed, outer_iters=iters),
                   dtype=float)

    W = discrete_adjoint(V, T, bangle=90.0, mode="corner")

    bnd = np.nonzero(fixed)[0]
    sid = _conj_segment_ids(V[bnd], poly)
    nrm = [np.asarray(v, float) / np.linalg.norm(np.asarray(v, float))
           for v in spec['normals']]
    off = []
    for k in range(len(nrm)):
        pts = W[bnd[sid == k]]
        off.append(float((pts @ nrm[k]).mean()) if len(pts) else 0.0)
    for a, b in spec.get('same', {}).items():
        off[a] = off[b]

    for k in range(len(nrm)):
        idx = bnd[sid == k]
        if not len(idx):
            continue
        W[idx] -= ((W[idx] @ nrm[k]) - off[k])[:, None] * nrm[k]
    W = np.asarray(minimize_area(W.copy(), T, fixed,
                                 outer_iters=max(1, iters // 2)),
                   dtype=float)

    # One plane PER ARC, in datafile order -- not deduplicated.  The
    # assembly words below index generators by arc, so collapsing
    # coincident planes here would renumber the letters.
    return W, [tuple(f) for f in quads], list(zip(nrm, off))


def _reflection(n, c):
    M = np.eye(4)
    M[:3, :3] = np.eye(3) - 2.0 * np.outer(n, n)
    M[:3, 3] = 2.0 * c * n
    return M


# Order of the transform set each surface's own cell word produces,
# measured by running Brakke's datafiles in Evolver 2.70.  Gated as an
# equality in `_selftest`, because a change here means the letter-to-arc
# mapping is wrong, not that the mesh got slightly worse.
_CONJ_CELL_COPIES = {'GW': 12, 'HT_HR': 24, 'TR_HT': 24, 'HR_TR': 24}


def _matkey(M):
    return tuple(np.round(np.asarray(M)[:3, :].ravel(), 7) + 0.0)


def eval_transform_expr(gens, word):
    """Evolver's `transform_expr`, which is how Brakke states a unit cell.

    Each letter `g` denotes the SET {I, g}, and juxtaposition means all
    ordered products, so scanning the word left to right the transform
    set doubles:  S := S union S*g.  The result is every product of a
    SUBSEQUENCE of the word, deduplicated as group elements -- which is
    what keeps it small: "bcbcbc" is 12 transforms (the dihedral group
    D6), not 2^6 = 64, and "adacada" is 42, not 128.

    Composition is left to right, so the RIGHTMOST letter acts on points
    first.  Read the word right to left and it says: reflect the patch in
    the last letter's mirror, then double that cluster in the next, and
    so on.

    Verified against Evolver 2.70 itself, which reports 16 for SSadj's
    "dcba", 24 for the hybrids' "abcbcbc" and 42 for GW's "adacada".
    """
    S = [np.eye(4)]
    seen = {_matkey(S[0])}
    for ch in word:
        G = gens[ch]
        for M in list(S):
            N = M @ G
            k = _matkey(N)
            if k not in seen:
                seen.add(k)
                S.append(N)
    return S


def conjugate_tile(V, quads, arc_planes, spec, depth=2, tol=1e-5):
    """Assemble the cell Brakke's datafile names, not a ball of copies.

    `Reflections` selects: 1 the bare patch, 2 the surface's own cell
    word, 3+ the next word it defines (a layer, or seven cells).

    This replaced a nearest-N centroid ball, and the ball was wrong twice
    over.  Its counts (12/32/72/144) missed the canonical ones -- 12, 24
    and 42 for these surfaces -- and its SHAPE was wrong regardless:
    GW's layer is a flat 6.2 x 6.5 x 1.9 hexagonal slab, which no ball
    approximates.  An earlier attempt grew the orbit breadth-first in the
    WORD metric instead, which is worse again: reflecting in two parallel
    mirrors makes a translation, so the shortest-word translation runs
    away and GW came out 6.0 x 5.4 x 16.8, a column.

    The groups are infinite, but that never has to be dealt with: a word
    of length L bounds the set at 2^L before dedup, and Brakke's longest
    is 13 letters.
    """
    V = np.asarray(V, dtype=float)
    words = spec.get('words') or ()
    if int(depth) <= 1 or not words:
        return V, list(quads), 1
    word = words[min(int(depth) - 2, len(words) - 1)]
    gens = {ch: _reflection(*arc_planes[k])
            for ch, k in spec['letters'].items()}
    mats = eval_transform_expr(gens, word)

    # A second, IMAGE-level dedup on top of the matrix one: distinct
    # group elements can still place this particular patch identically
    # when the patch is stabilised by part of the group.
    keep = []
    seen = set()
    c0 = V.mean(0)
    for M in mats:
        k = tuple(np.round(c0 @ M[:3, :3].T + M[:3, 3], 5))
        if k in seen:
            continue
        seen.add(k)
        keep.append(M)

    pts = np.concatenate([V @ M[:3, :3].T + M[:3, 3] for M in keep])
    nV = len(V)
    faces = []
    for j, M in enumerate(keep):
        flip = float(np.linalg.det(M[:3, :3])) < 0.0
        for f in quads:
            g = tuple(i + j * nV for i in f)
            faces.append(g[::-1] if flip else g)
    W, wf = _weld_points(pts, faces, tol)
    return W, wf, len(keep)


def _orbit_defects(W, wf):
    """(duplicate faces, over-shared edges, component count)."""
    dup = len(wf) - len({frozenset(f) for f in wf})
    ec = {}
    for f in wf:
        mm = len(f)
        for t in range(mm):
            x, y = f[t], f[(t + 1) % mm]
            if x == y:
                continue
            e = (x, y) if x < y else (y, x)
            ec[e] = ec.get(e, 0) + 1
    over = sum(1 for v in ec.values() if v > 2)
    parent = list(range(len(W)))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for f in wf:
        r0 = _find(f[0])
        for v in f[1:]:
            r1 = _find(v)
            if r0 != r1:
                parent[r1] = r0
    comps = len({_find(i) for i in range(len(W))})
    return dup, over, comps


def conjugate_tile_checked(V, quads, arc_planes, spec, depth=2):
    """`conjugate_tile`, but only if the assembly verifies.

    Tries the requested word FIRST and falls back through the shorter
    ones, ending at the bare patch.  It used to count upwards from depth
    1 and stop at the first failure, which silently disabled the whole
    feature the moment depth 1 came to mean "the bare patch": that
    returns one copy, the loop read one copy as a failed orbit, and every
    surface fell back before the real word was ever tried.

    The gate itself is the R family's and stays strict for the same
    reason: an assembly that fails it is not a slightly-wrong surface but
    a pile of coincident sheets, which still renders as a plausible
    picture.  Falling back is the correct answer, not a consolation
    prize.
    """
    for d in range(max(int(depth), 1), 1, -1):
        W, wf, n = conjugate_tile(V, quads, arc_planes, spec, depth=d)
        if n <= 1:
            continue
        dup, over, comps = _orbit_defects(W, wf)
        if dup == 0 and over == 0 and comps == 1:
            return W, wf, n
    return np.asarray(V, dtype=float), list(quads), 1


def conjugate_surface(key, m=120, rings=20, iters=400, depth=2):
    """Patch, orbit and patch area -- the headless entry point."""
    V, quads, arc_planes = conjugate_patch(key, m=m, rings=rings,
                                           iters=iters)
    W, wf, n = conjugate_tile_checked(V, quads, arc_planes,
                                      CONJUGATE_SURFACES[key], depth=depth)
    T = np.asarray(_quads_to_tris(quads))
    return W, wf, n, mesh_area(np.asarray(V, dtype=float), T)


def conjugate_build(key, cells, res_per_cell, scale, theta):
    """TPMS_EXACT-compatible wrapper: build, centre, and fit into the
    2*scale cube the rest of the catalog uses."""
    # The patch is a disk grid, so its quad count is m * rings and grows
    # as the SQUARE of the resolution -- then it is multiplied by the
    # copy count.  6*res by res put 15000 quads in one patch at the
    # resolution 50 the UI offers, which is 480k faces before tiling even
    # starts.  Half that in each direction is visually indistinguishable
    # here and keeps the tiled mesh usable.
    res = max(int(res_per_cell), 8)
    m = max(24, 3 * res)
    rings = max(6, res // 2)
    depth = max(1, int(cells)) if np.isscalar(cells) else 2
    V, faces, _n, _a = conjugate_surface(key, m=m, rings=rings, depth=depth)
    V = np.asarray(V, dtype=float)
    if len(V):
        V = V - 0.5 * (V.max(0) + V.min(0))
        span = float(np.max(V.max(0) - V.min(0)))
        if span > 1e-12:
            V = V * (2.0 * scale / span)
    if theta:
        ct, st = math.cos(theta), math.sin(theta)
        R = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        V = V @ R.T
    return V, faces


def _selftest():
    ok = True

    # resample_loop must lay points at uniform arclength and keep them on the
    # curve.  Sampled from a fine polygon so the chord-vs-arc discrepancy at
    # the input corners (which is real, and O(1/n^2)) stays well below the
    # non-uniformity a broken resampler would produce.
    ang = np.linspace(0.0, TAU, 512, endpoint=False)
    circ = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
    r = resample_loop(circ, 64)
    step = np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1)
    spread = float(np.ptp(step)) / float(np.mean(step))
    oncurve = abs(float(np.mean(np.hypot(r[:, 0], r[:, 1]))) - 1.0)
    good = len(r) == 64 and spread < 1e-3 and oncurve < 1e-3
    ok &= good
    print(f"plateau: resample_loop n={len(r)} step spread={spread:.2e} "
          f"|r-1|={oncurve:.2e} {'OK' if good else 'FAIL'}")

    # mesh_area on a unit square split into two triangles is exactly 1.
    Vs = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0], [0.0, 1.0, 0]])
    Ts = np.array([[0, 1, 2], [0, 2, 3]])
    a = mesh_area(Vs, Ts)
    good = abs(a - 1.0) < 1e-12
    ok &= good
    print(f"plateau: mesh_area unit square={a:.12f} {'OK' if good else 'FAIL'}")

    # The real gate: area minimization between two coaxial circles must
    # converge to the catenoid.  For radius 1 rings at z = +-0.5 the neck of
    # the minimal catenoid solves r0 cosh(0.5/r0) = 1 -> r0 = 0.8483...,
    # with area 2*pi*r0*(r0*sinh(1/r0)/... ) -- checked here structurally:
    # the solve must (a) reduce area, (b) leave the pinned rims untouched,
    # (c) pull the waist inside the initial cylinder.
    h, nring, nrow = 0.5, 48, 13
    th = np.linspace(0.0, TAU, nring, endpoint=False)
    zs = np.linspace(-h, h, nrow)
    V = np.array([[math.cos(t), math.sin(t), z] for z in zs for t in th])
    T = []
    for j in range(nrow - 1):
        for i in range(nring):
            a0 = j * nring + i
            b0 = j * nring + (i + 1) % nring
            T.append([a0, b0, a0 + nring])
            T.append([b0, b0 + nring, a0 + nring])
    T = np.array(T)
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:nring] = True
    fixed[-nring:] = True
    rim0 = V[:nring].copy()
    a0 = mesh_area(V, T)
    minimize_area(V, T, fixed, outer_iters=40)
    a1 = mesh_area(V, T)
    waist = float(np.mean(np.hypot(V[nrow // 2 * nring:(nrow // 2 + 1) * nring, 0],
                                   V[nrow // 2 * nring:(nrow // 2 + 1) * nring, 1])))
    rim_moved = float(np.max(np.abs(V[:nring] - rim0)))
    r0 = 0.8483
    good = (a1 < a0 and rim_moved < 1e-12 and abs(waist - r0) < 0.02
            and bool(np.all(np.isfinite(V))))
    ok &= good
    print(f"plateau: catenoid area {a0:.4f} -> {a1:.4f}, waist={waist:.4f} "
          f"(exp {r0:.4f}), rims pinned ({rim_moved:.1e}) "
          f"{'OK' if good else 'FAIL'}")

    # The same solve with in-loop grooming (flips + tangential
    # averaging) must still converge to the catenoid with pinned rims,
    # and the historical clamped mode must keep working as a fallback.
    for label, kw in (("groomed", dict(groom_every=4)),
                      ("clamped", dict(cotan_mode="clamp"))):
        V2 = np.array([[math.cos(t), math.sin(t), z]
                       for z in zs for t in th])
        T2 = T.copy()
        rimA = V2[:nring].copy()
        minimize_area(V2, T2, fixed, outer_iters=40, **kw)
        waist2 = float(np.mean(np.hypot(
            V2[nrow // 2 * nring:(nrow // 2 + 1) * nring, 0],
            V2[nrow // 2 * nring:(nrow // 2 + 1) * nring, 1])))
        moved2 = float(np.max(np.abs(V2[:nring] - rimA)))
        good = (abs(waist2 - r0) < 0.02 and moved2 < 1e-12
                and bool(np.all(np.isfinite(V2))))
        ok &= good
        print(f"plateau: catenoid ({label}) waist={waist2:.4f}, rims "
              f"pinned ({moved2:.1e}) {'OK' if good else 'FAIL'}")

    # The R family is certified by AREA, not by median |H|.  On a mesh
    # produced by relaxation rather than by integration, |H| measures
    # triangle quality as much as it measures the surface -- R-II's rises
    # under refinement while its area settles to five figures -- so area
    # is the honest invariant and the one gated here.
    for key in ('R1', 'R2', 'R3', 'I6', 'I8', 'I9'):
        areas = []
        for m, rows in ((96, 16), (144, 24)):
            _V, _Q, a = ring_surface(key, m=m, rows=rows)
            areas.append(a)
        drift = abs(areas[1] - areas[0]) / max(areas[1], 1e-30)
        good = drift < 0.01 and areas[1] > 1e-6
        ok &= good
        print("plateau: ring %s area %.6f -> %.6f (drift %.4f%%) %s"
              % (key, areas[0], areas[1], 100.0 * drift,
                 'OK' if good else 'FAIL'))

    # The tiling must be ACCEPTED where the generators are right and
    # REFUSED where they are not.  Both halves matter: a check that only
    # ever accepts would have passed the first version of
    # `ring_generators`, which applied the R family's half-turns to the
    # I family and welded I-9 into 14 266 over-shared edges while halving
    # its vertex count -- coincident sheets, invisible to a component
    # count.
    for key, expect_tiled in (('R2', True), ('R3', True),
                              ('I9', True), ('I8', False), ('I6', False)):
        V, Q, _a = ring_surface(key, m=96, rows=16)
        W, QQ, n = ring_tile_checked(V, Q, key, depth=2)
        ec = {}
        for f in QQ:
            mm = len(f)
            for t in range(mm):
                x, y = f[t], f[(t + 1) % mm]
                e = (x, y) if x < y else (y, x)
                ec[e] = ec.get(e, 0) + 1
        over = sum(1 for c in ec.values() if c > 2)
        tiled = n > 1
        good = (over == 0) and (tiled == expect_tiled)
        ok &= good
        print("plateau: ring %s tiling %s (%d copies, %d over-shared) %s"
              % (key, "accepted" if tiled else "refused -> patch",
                 n, over, 'OK' if good else 'FAIL'))

    # The discrete adjoint: only the two facts that ARE established.
    a3 = np.array([0., 0., 0.])
    b3 = np.array([1., 0., 0.])
    c3 = np.array([0.3, 0.9, 0.2])
    Vt = np.array([a3, b3, c3])
    Tt = np.array([[0, 1, 2]])
    ev = [b3 - a3, c3 - b3, a3 - c3]
    nn = np.cross(ev[0], ev[1])
    nn = nn / np.linalg.norm(nn)
    worst_close = 0.0
    worst_len = 0.0
    for ang in (0.0, 30.0, 90.0):
        bc = math.cos(math.radians(ang))
        bs = math.sin(math.radians(ang))

        def _st(eo, _bc=bc, _bs=bs):
            return -0.5 * (_bc * eo + _bs * np.cross(nn, eo))

        Pp = [np.zeros(3), None, None]
        Pp[1] = Pp[0] + _st(ev[2])
        Pp[2] = Pp[1] + _st(ev[0])
        worst_close = max(worst_close,
                          float(np.linalg.norm(Pp[2] + _st(ev[1]) - Pp[0])))
        for k, want in enumerate((ev[2], ev[0], ev[1])):
            got = float(np.linalg.norm(Pp[(k + 1) % 3] - Pp[k]))
            worst_len = max(worst_len,
                            abs(got - 0.5 * float(np.linalg.norm(want))))
    good = worst_close < 1e-12 and worst_len < 1e-12
    ok &= good
    print("plateau: adjoint facet relation closes %.1e, medial lengths "
          "|e|/2 to %.1e %s"
          % (worst_close, worst_len, 'OK' if good else 'FAIL'))

    # ...and bangle 0 must be the identity on a real mesh.
    tt = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    lo = np.stack([np.cos(tt), np.sin(tt), np.zeros_like(tt)], 1)
    hi = lo.copy()
    hi[:, 2] = 1.0
    Vc, qc, fx = build_annulus_grid(lo, hi, 8)
    Tc = _quads_to_tris(qc)
    Vc = np.asarray(minimize_area(np.asarray(Vc, float).copy(), Tc, fx,
                                  outer_iters=80), dtype=float)
    Wc = discrete_adjoint(Vc, Tc, bangle=0.0)
    da = abs(mesh_area(Wc, Tc) / max(mesh_area(Vc, Tc), 1e-30) - 1.0)
    good = da < 1e-9
    ok &= good
    print("plateau: adjoint at bangle 0 is the identity (area ratio "
          "off by %.1e) %s" % (da, 'OK' if good else 'FAIL'))

    # The conjugate must be ISOMETRIC, and with Brakke's reconstruction
    # that is checkable FACET BY FACET, not just in total: each triangle
    # (P1+P2-P3, -P1+P2+P3, P1-P2+P3) is the Bonnet rotation of its
    # parent about the parent's own normal, hence congruent to it.
    #
    # Two weaker reconstructions were measured first and neither is the
    # conjugate: the conforming meshes only approach the right area under
    # refinement, and the medial triangles are isometric but cover a
    # quarter of it.  Gate on the real thing.
    #
    # The two Bonnet angles are gated differently ON PURPOSE:
    #
    #   bangle 0 is the identity, so per-facet congruence must hold to
    #   ROUNDOFF (1e-11) and the total area must match exactly.  That
    #   catches any error in the reconstruction algebra itself.
    #
    #   bangle 90 propagates a 1-form that is closed only insofar as the
    #   input is discretely minimal, and `_adjoint_edge_positions` fits
    #   it by least squares, which SPREADS the closure defect over the
    #   mesh.  Per-facet congruence is therefore only ~3e-3 in the mean,
    #   while the defect cancels in the sum and the TOTAL area survives
    #   to ~2e-4.  So the total is what is gated here, and the number is
    #   a measure of how discretely minimal `minimize_area`'s output is,
    #   not of the transform.  Tightening it means polishing the input,
    #   not touching this code.
    r3v = math.sqrt(3.0)
    hexpoly = np.array([[0, 0, 0], [r3v, -1, 0], [r3v, -1, 1.0],
                        [r3v / 2, -1.5, 1.0], [r3v / 2, .5, 1.0],
                        [r3v / 2, .5, 0]], dtype=float)
    lp = resample_loop(np.vstack([hexpoly, hexpoly[:1]]), 96)
    Vh, qh, fxh = build_disk_grid(lp, 16)
    Th = np.asarray(_quads_to_tris(qh))
    Vh = np.asarray(minimize_area(np.asarray(Vh, float).copy(), Th, fxh,
                                  outer_iters=300), dtype=float)

    def _tri_areas(P, F):
        P = np.asarray(P, dtype=float)
        F = np.asarray(F)
        A, B, C = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)

    face_a = _tri_areas(Vh, Th)
    tot_a = float(face_a.sum())
    for ang, cong_tol, tot_tol in ((0.0, 1e-11, 1e-12), (90.0, None, 2e-3)):
        Pa, Fa = adjoint_mesh(Vh, Th, bangle=ang)
        new_a = _tri_areas(Pa, np.asarray(Fa))
        cong = float(np.max(np.abs(new_a - face_a) / np.maximum(face_a, 1e-300)))
        rat = float(new_a.sum()) / max(tot_a, 1e-30)
        good = abs(rat - 1.0) < tot_tol and bool(np.all(np.isfinite(Pa)))
        if cong_tol is not None:
            good = good and cong < cong_tol
        ok &= good
        print("plateau: conjugate at %2.0fdeg -- per-facet congruence %.1e, "
              "total area ratio %.9f %s"
              % (ang, cong, rat, 'OK' if good else 'FAIL'))

    # The conjugate-Plateau route, end to end: Plateau -> conjugate ->
    # project onto the mirrors -> reflect.  Three things are gated, and
    # the middle one is the whole point of the route.
    #
    #   * the patch area converges under refinement (it is a relaxation,
    #     so area is the honest invariant -- see the R family above);
    #   * the boundary lands EXACTLY on its mirror planes.  Not "nearly":
    #     at the ~2e-3 the raw conjugate achieves, reflected copies do
    #     not weld at all, so this is pass/fail rather than a tolerance;
    #   * the orbit verifies, so `conjugate_tile_checked` returns more
    #     than the bare patch.
    for key in sorted(CONJUGATE_SURFACES):
        areas = []
        planes = quads = Vc = None
        for mm, rr in ((72, 12), (96, 16)):
            Vc, quads, planes = conjugate_patch(key, m=mm, rings=rr,
                                                iters=250)
            areas.append(mesh_area(Vc, np.asarray(_quads_to_tris(quads))))
        drift = abs(areas[1] - areas[0]) / max(areas[1], 1e-30)
        bnd = _boundary_vertices(np.asarray(_quads_to_tris(quads)))
        offp = 0.0
        for i in bnd:
            offp = max(offp, min(abs(float(Vc[i] @ n) - c) for n, c in planes))
        _W, _wf, ncopy = conjugate_tile_checked(
            Vc, quads, planes, CONJUGATE_SURFACES[key], depth=2)
        # The copy count is not a soft quality measure -- it is the ORDER
        # of the transform set Brakke's own word produces, checked
        # against Evolver 2.70 running his datafile.  Anything else means
        # the letter-to-arc mapping has drifted, so it is an equality.
        want = _CONJ_CELL_COPIES[key]
        # Gated against EVOLVER, not against ourselves.  Brakke's own
        # program was run on each datafile and its patch area recorded in
        # the spec; agreement to 1% is what says the contour and the six
        # constraint normals were transcribed correctly.  This is the
        # check that would have caught building all three hybrids from
        # one contour (0.9338 and 1.4064 against 0.6623 and 0.5893).
        ev = CONJUGATE_SURFACES[key]['evolver_area']
        aerr = abs(areas[1] - ev) / ev
        good = (drift < 0.01 and offp < 1e-9 and ncopy == want
                and aerr < 0.01 and areas[1] > 1e-6)
        ok &= good
        print("plateau: conjugate %-6s area %.6f (Evolver %.6f, %.2f%%), "
              "drift %.3f%%, on-plane %.0e, %d copies (Evolver %d) %s"
              % (key, areas[1], ev, 100.0 * aerr, 100.0 * drift, offp,
                 ncopy, want, 'OK' if good else 'FAIL'))

    # The ring rows assembled by Brakke's own generator matrices and
    # words.  The copy counts are HIS -- 8 for I-8's `bac`, 8 for I-9's
    # `abc`, 24 for R-III's `fdfedfe` -- so they are gated as
    # equalities, and the gate is the usual one besides: no duplicate
    # faces, no over-shared edges, one component.
    for key, want, word in (('I8', 8, 'bac'), ('I9', 8, 'abc'),
                            ('R3', 24, 'fdfedfe')):
        Vr, Qr, _a = ring_surface(key, m=96, rows=16)
        got = ring_cell_assemble(key, Vr, Qr)
        good = got is not None and got[2] == want and got[3] == word
        if got is None:
            print("plateau: ring cell %-3s REFUSED (wanted %s, %d copies) "
                  "FAIL" % (key, word, want))
        else:
            Wc = np.asarray(got[0])
            bb = Wc.max(0) - Wc.min(0)
            print("plateau: ring cell %-3s word %-8s %2d copies (Brakke %d) "
                  "%.3f x %.3f x %.3f %s"
                  % (key, got[3], got[2], want, bb[0], bb[1], bb[2],
                     'OK' if good else 'FAIL'))
        ok &= good

    # The datafile-derived cells.  Every number here was READ from a
    # `.fe` by `fedata` rather than transcribed, so what is gated is that
    # the pipeline still reproduces the copy count the datafile's own
    # word gives, and that the result is one clean sheet.
    from .fecells import FE_CELLS
    for key in sorted(FE_CELLS):
        spec = FE_CELLS[key]
        Vf, Qf = fe_cell_patch(key, m=72, rings=12, iters=200)
        # Flatness FIRST, and at runtime rather than at bake time.  The
        # shipped cell re-spans its recorded boundary, so a cell can be
        # solved correctly and still come back a flat plate if what was
        # recorded was the contour the solve started from instead of the
        # one it ended on.  Schwarz P, Schwarz D, Neovius, I-WP and C(D)
        # all shipped exactly that way, assembling into clean faceted
        # stars that every other check here was happy with.
        _u, _sv, _vt = np.linalg.svd(np.asarray(Vf) - np.asarray(Vf).mean(0),
                                     full_matrices=False)
        flat = float(_sv[2] / max(_sv[0], 1e-30))
        if flat < 0.02:
            print("plateau: fe cell %-14s FLAT (%.4f) -- the recorded "
                  "boundary spans a plate  FAIL" % (key, flat))
            ok = False
        got = fe_cell_assemble(key, Vf, Qf)
        want = spec['copies']
        good = got is not None and got[2] == want
        if got is None:
            print("plateau: fe cell %-14s %-10r REFUSED (wanted %d) FAIL"
                  % (key, spec['word'], want))
        else:
            bb = np.asarray(got[0]).max(0) - np.asarray(got[0]).min(0)
            print("plateau: fe cell %-14s %-10r %3d copies (file %d) "
                  "%.2f x %.2f x %.2f %s"
                  % (key, spec['word'], got[2], want, bb[0], bb[1], bb[2],
                     'OK' if good else 'FAIL'))
        ok &= good

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("plateau self-test failed")
