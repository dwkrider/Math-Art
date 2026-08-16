# Gluing two flat pieces and solving for the shape they pop into.
#
# Part of the Math Art D-form engine (`math_art/dform/`).  Python and
# numpy only -- no `bpy`.
#
# WHAT DETERMINES THE SHAPE.  Demaine and O'Rourke (Geometric Folding
# Algorithms, 2007, ch. 25) prove that gluing two convex flat pieces of
# equal perimeter edge to edge is an Alexandrov gluing: the resulting
# metric sphere is realised by exactly ONE convex body, the D-form is
# that body, and it is the convex hull of its own seam curve --
# developable and smooth away from the seam.  Three things follow, and
# the solver enforces all three; the theorem then supplies uniqueness:
#
#   isometry    every edge keeps the length it had in the flat layout
#               (position-based dynamics, Muller et al. 2007)
#   convexity   locally, no flap is reflex; globally, nothing may leave
#               the convex hull of the seam ring (`_project_hull`)
#   inflation   a pressure term, annealed to zero, to leave the flat
#               pillow -- which is an isometric solution too
#
# WHY BOTH CONVEXITY TERMS.  The local flap test alone does not work.
# An accordion pleat is exactly isometric AND has zero angle defect, so
# neither the metric nor the developability check can see it, and the
# local projection is far too weak to iron one out once it forms: every
# parameter setting tried stalled with a third of the flaps folded past
# 20 degrees.  The hull projection is global and is a true projection
# onto a convex set, so it cannot be pleated.  It is also what the
# theorem actually says, rather than a consequence of it.
#
# WHERE IT STARTS decides whether any of that converges.  The obvious
# start -- two hemispheres of the right total AREA -- fails badly: for
# two discs of perimeter P the area-matched sphere has an equator of
# only P/sqrt(2), so every seam edge begins 30% short, the seam must
# stretch by nearly half, and the sheet buckles against it.  The seam
# length, not the area, is the binding constraint, so `_place_cap`
# starts from a shallow dome whose seam is exactly the right length.
#
# THE ANTI-D-FORM -- two annuli joined at their HOLES, outer edges free
# (John Sharp) -- is not convex and has no uniqueness theorem, so the
# hull machinery above is useless there and something else must select
# the shape.  What does the selecting (`settle_anti`):
#
#   seam first  on each sheet the seam's curvature splits as
#               kappa^2 = kappa_g^2 + kappa_n^2 (Darboux), so the seam's
#               space curvature is at least the LARGER of the two flat
#               hole curvatures pointwise -- and those integrate to 2*pi
#               each, so the max integrates to MORE than 2*pi and the
#               seam cannot stay planar.  `_solve_seam` solves the ring
#               alone for a closed curve carrying that floor times a
#               MARGIN (at the bare floor the two sheets pinch flat
#               against each other at every curvature crossover), and
#               the whole shape hangs off this one curve.
#   start       flat coplanar annuli -- the exactly-isometric degenerate
#               state -- carrying the solved seam wave faded inward,
#               with the innermost quarter swapped for a marched
#               developable wrap (`_collar_start`) so the crease-like
#               tilt at the seam is present from the first iteration.
#   hinge       a true rest-flat dihedral hinge (`_project_hinge`), not
#               the linearised height push: the height of the far vertex
#               over the near plane VANISHES again at a 180-degree fold,
#               so a pleat is a spurious equilibrium of the linear term.
#               The angle-based hinge is monotone all the way to the
#               fold.  Seam flaps are excluded -- the seam is a genuine
#               crease -- and the seam boundary layer is graded stiffer.
#   spectrum    the one gate pleats cannot pass: a wrinkled ring is
#               exactly as isometric as a curled one and the hinge alone
#               cannot coordinate the ring-wide rotation between them,
#               so while the shape locks in, every ring is low-pass
#               filtered azimuthally (`_ring_lowpass`), then the guard
#               is opened and isometry gets the last word.
#   no pressure the rims are free and there is no volume to inflate; the
#               turning floor on the seam (a second-neighbour chord CAP,
#               `_project_chord_cap`) replaces pressure as the term that
#               keeps the surface off the flat degenerate state.
#
# References:
#   E. Demaine, J. O'Rourke, "Geometric Folding Algorithms," Cambridge,
#       2007, ch. 25 (Alexandrov gluings).
#   M. Muller, B. Heidelberger, M. Hennix, J. Ratcliff, "Position Based
#       Dynamics," J. Vis. Commun. Image R. 18(2), 2007, pp. 109-118.
#   T. Wills, "D-Forms: 3D Forms from Two 2D Sheets," Bridges 2006;
#       J. Sharp, "D-Forms and Developable Surfaces," Bridges 2005, and
#       "D-Forms" (Tarquin, 2009) -- the anti-D-form.
#   E. Grinspun, A. Hirani, M. Desbrun, P. Schroder, "Discrete Shells,"
#       SCA 2003 -- the hinge bending energy `_project_hinge` discretises.

import numpy as np

from .sheet import (angle_defects, boundary_edges, edges_from_tris,
                    interior_edge_flaps)

# Isometry sweeps in the closing polish.  Every other pass in the loop
# can leave a little strain behind; this one is the last word, and it is
# paid ONCE per solve, so it is cheap to make generous.
_POLISH = 400


class Glued:
    """Two flat pieces welded along a seam, ready to solve.

    Carries the 3D state plus everything needed to unroll it again: each
    piece's flat layout and the global<->local index maps.
    """

    def __init__(self, V, tris, edges, rest, seam, piece, flat, maps,
                 closed=True, hole_seam=False):
        self.V = V
        self.tris = tris
        self.edges = edges
        self.rest = rest
        self.seam = seam
        self.piece = piece
        self.flat = flat            # [(V2_a, tris_a), (V2_b, tris_b)]
        self.maps = maps            # [local->global, local->global]
        self.flaps = interior_edge_flaps(tris)
        self.closed = closed
        # True when the seam is a HOLE boundary (anti-D-form): the flat
        # material then lies OUTSIDE the seam ring, which flips the sign
        # of every angle bookkeeping done against the flat layout
        self.hole_seam = hole_seam
        # vertices on a free edge (none when the surface closes up): the
        # anti-D-form's outer rims, which no constraint may pull on
        bnd = boundary_edges(tris)
        self.free = np.zeros(len(V), bool)
        if len(bnd):
            self.free[np.unique(bnd)] = True
        self.strain = float('nan')
        self.iterations = 0
        self._cache = None

    def plan(self, hull_dirs):
        """Topology-constant work, done once and reused by every sweep.

        The mesh never changes during a solve, so the edge colouring, the
        per-colour index arrays, the flap scatter indices and their
        (constant) incidence counts are all loop invariants.  `settle`
        calls `relax` eight times; recomputing these there cost more than
        the solve at high segment counts -- the greedy colouring alone is
        a Python loop over every edge.
        """
        if self._cache is not None and self._cache['dirs'] is not None \
                and len(self._cache['dirs']) == hull_dirs:
            return self._cache
        nv = len(self.V)
        ar = np.arange(3)

        groups = []
        for g in edge_colouring(self.edges, nv):
            e0 = self.edges[g, 0].copy()
            e1 = self.edges[g, 1].copy()
            # within a colour no vertex repeats, so both endpoints can be
            # written in ONE scatter instead of two
            groups.append((e0, e1, self.rest[g].copy(),
                           np.concatenate([e0, e1])))

        a, b, c, d = (self.flaps[:, 0], self.flaps[:, 1],
                      self.flaps[:, 2], self.flaps[:, 3])
        flat3 = lambda i: (i[:, None] * 3 + ar).ravel()   # noqa: E731
        cache = {
            'groups': groups,
            'valence': np.maximum(np.bincount(self.edges.ravel(),
                                              minlength=nv), 1).astype(float),
            'e0': self.edges[:, 0].copy(),
            'e1': self.edges[:, 1].copy(),
            'edge3': np.concatenate([flat3(self.edges[:, 0]),
                                     flat3(self.edges[:, 1])]),
            'flap3': np.concatenate([flat3(d), flat3(a), flat3(b), flat3(c)]),
            # each flap touches d once and a, b, c once -- V-independent
            'flap_cnt': np.maximum(
                np.bincount(np.concatenate([d, a, b, c]), minlength=nv),
                1).astype(float)[:, None],
            'tri3': np.concatenate([flat3(self.tris[:, 0]),
                                    flat3(self.tris[:, 1]),
                                    flat3(self.tris[:, 2])]),
            'dirs': _sphere_dirs(hull_dirs) if hull_dirs else None,
            'nv': nv,
        }
        self._cache = cache
        return cache

    def max_strain(self):
        d = self.V[self.edges[:, 1]] - self.V[self.edges[:, 0]]
        L = np.linalg.norm(d, axis=1)
        return float(np.max(np.abs(L / self.rest - 1.0)))

    def interior_defect(self):
        """Worst angle defect away from the seam: developability.

        Free-rim vertices are excluded along with the seam: a boundary
        vertex's angle sum is about pi by construction, so counting its
        "defect" would report every open surface as broken.
        """
        d = np.abs(angle_defects(self.V, self.tris, len(self.V)))
        mask = ~self.free
        mask[self.seam] = False
        return float(np.max(d[mask])) if mask.any() else 0.0

    def seam_defect_total(self):
        """Total angle defect carried by the seam (4*pi for a D-form)."""
        d = angle_defects(self.V, self.tris, len(self.V))
        return float(np.sum(d[self.seam]))

    def aspect(self):
        ext = self.V.max(axis=0) - self.V.min(axis=0)
        return float(ext.min() / max(ext.max(), 1e-15))


def _align_piece(V2, src, dst):
    """Move a whole flat piece by the rigid map best taking `src`->`dst`.

    Orthogonal Procrustes with the reflection allowed: piece B is glued
    face-down, so the good alignment usually IS a reflection.
    """
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    R = U @ Vt
    return (V2 - cs) @ R + cd


def glue(sheet_a, sheet_b, corr, closed=True, pop=1.0, hole_seam=False):
    """Weld the two pieces and lay them out ready to solve.

    WHERE THE SOLVE STARTS MATTERS MORE THAN HOW IT STEPS, and the
    quantity to get right is the SEAM LENGTH.  The two outlines are
    different shapes -- that is the point of a D-form -- so they cannot
    both sit at the seam unchanged; whatever compromise is made there is
    the strain the solver has to work off, and strain at the seam is
    what buckles the sheet next to it.

    So the seam starts as the average of the two boundaries, rescaled to
    the exact common perimeter: every seam edge begins at its rest
    length.  Each piece is then laid out in its OWN flat coordinates and
    coned onto that seam by `_place_cap`, which leaves the interior
    near-isometric.  Measured on an ellipse+circle pair, that start has
    zero seam strain and 2% mean strain overall, against 31% at the seam
    for the area-matched sphere it replaced.

    `pop` scales the dome height that breaks the flat-pillow symmetry
    (1 = the default shallow dome, 0 = dead flat).
    """
    A2, TA, sa, da, pa = sheet_a
    B2, TB, sb, db, pb = sheet_b
    n = len(sa)
    if len(sb) != n:
        raise ValueError("seam rings must have the same length")

    map_a = np.empty(len(A2), dtype=int)
    map_b = np.empty(len(B2), dtype=int)
    map_a[sa] = np.arange(n)
    map_b[sb[corr]] = np.arange(n)
    nxt = n
    inner_a = np.setdiff1d(np.arange(len(A2)), sa)
    map_a[inner_a] = np.arange(nxt, nxt + len(inner_a))
    nxt += len(inner_a)
    inner_b = np.setdiff1d(np.arange(len(B2)), sb)
    map_b[inner_b] = np.arange(nxt, nxt + len(inner_b))
    nxt += len(inner_b)

    # piece B's ring parameter runs against the global seam index when
    # the correspondence is the (default) reversed traversal, so recover
    # the azimuth from `corr` itself rather than assuming a direction.
    # Both pieces end up parameterised by (global seam index)/n, which is
    # what lets one displacement field drive them both.
    j0, j1 = int(corr[0]), int(corr[1] if n > 1 else corr[0])
    step = ((j1 - j0 + n // 2) % n) - n // 2
    phase = j0 / n
    par_b = (phase - pb) if step < 0 else (pb - phase)

    # B is glued face down, so bring it into A's frame by the best rigid
    # map (reflection allowed) before the two boundaries are averaged
    B2 = _align_piece(B2, B2[sb[corr]], A2[sa])

    bnd_a = A2[sa]
    bnd_b = B2[sb[corr]]
    seam2 = 0.5 * (bnd_a + bnd_b)
    from .curves import perimeter
    p_seam = perimeter(seam2)
    if p_seam > 1e-12:
        seam2 = seam2 * (perimeter(bnd_a) / p_seam)

    R = float(np.mean(np.linalg.norm(bnd_a - bnd_a.mean(axis=0), axis=1)))
    V = np.zeros((nxt, 3))
    _place_cap(V, map_a, pa, da, A2, bnd_a, seam2, +1.0, pop * 0.15 * R)
    _place_cap(V, map_b, par_b, db, B2, bnd_b, seam2, -1.0, pop * 0.15 * R)

    piece = np.zeros(nxt, dtype=int)
    piece[map_b[inner_b]] = 1

    # Whether piece B's triangles need reversing is decided by the SEAM
    # correspondence, not by a fixed convention: the default backwards
    # traversal already makes B walk the seam against A, so reversing it
    # as well would give two triangles per seam edge pointing the same
    # way.  Ask the assembled mesh instead of assuming.
    TB_g = map_b[TB]
    if not _orientable(np.concatenate([map_a[TA], TB_g], axis=0)):
        TB_g = TB_g[:, ::-1].copy()
    tris = np.concatenate([map_a[TA], TB_g], axis=0)
    # Point it outward NOW, before the flaps are cached: the convexity
    # test reads triangle normals, so a later global flip would invert
    # its sign and the solver would drive the surface concave.
    if closed:
        tris = orient_outward(V, tris)
    edges = edges_from_tris(tris)
    rest = _rest_from_flat(edges, [(A2, map_a, TA), (B2, map_b, TB)])

    g = Glued(V, tris, edges, rest, np.arange(n), piece,
              [(A2, TA), (B2, TB)], [map_a, map_b], closed=closed,
              hole_seam=hole_seam)
    # keep the polar coordinates the layout was built from: they are what
    # lets a finer mesh be warm-started off a coarser solve (`warm_start`)
    g.param = np.zeros(nxt)
    g.depth = np.zeros(nxt)
    g.param[map_a] = pa % 1.0
    g.depth[map_a] = da
    g.param[map_b] = par_b % 1.0
    g.depth[map_b] = db
    return g


def _place_cap(V, mp, param, depth, flat, bnd, seam2, sign, height):
    """Lay a piece down as a shallow dome, boundary bent onto the seam.

    The piece keeps its OWN flat coordinates and only the boundary is
    moved, onto the shared seam curve; that displacement fades inward as
    (1-depth), which is exactly the factor by which the ring-meshed
    piece already scales its boundary.  So the piece becomes the cone
    over the seam curve and the shape mismatch is spread smoothly over
    the whole sheet.  Fading it faster would confine the mismatch to a
    collar one ring deep, where -- the displacement being comparable to
    the ring spacing -- it lands as tens of percent of strain against
    the seam, which is what buckles the solve.

    `depth` runs 0 at the seam to 1 at the centre, so the dome rises
    away from the seam plane; `height` breaks the flat-pillow symmetry
    (that degenerate doubly-covered solution is isometric too, so
    something has to push the solve off it).
    """
    n = len(seam2)
    d = seam2 - bnd                       # boundary displacement
    w = (np.asarray(param) % 1.0) * n
    i0 = np.floor(w).astype(int) % n
    i1 = (i0 + 1) % n
    f = (w - np.floor(w))[:, None]
    off = (1.0 - f) * d[i0] + f * d[i1]
    fade = (1.0 - depth)[:, None]

    V[mp, :2] = flat + fade * off
    V[mp, 2] = sign * height * (1.0 - (1.0 - depth) ** 2)


def _rest_from_flat(edges, pieces):
    """Flat length of every global edge, from the piece(s) it lies in.

    The seam edges lie in BOTH pieces, and their two flat lengths are
    not identical: both boundaries are sampled uniformly in ARCLENGTH,
    so the chords differ by a curvature term of order h^2.  Neither
    piece is right, and forcing one of them would build that error into
    the seam, so a shared edge takes the mean.
    """
    total = np.zeros(len(edges))
    count = np.zeros(len(edges))
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(edges)}
    for V2, mp, T in pieces:
        for a, b in edges_from_tris(T):
            g0, g1 = int(mp[a]), int(mp[b])
            i = key.get((g0, g1) if g0 < g1 else (g1, g0))
            if i is None:
                raise ValueError("edge with no global counterpart")
            total[i] += float(np.linalg.norm(V2[a] - V2[b]))
            count[i] += 1.0
    if not np.all(count > 0):
        raise ValueError("edge with no flat pre-image")
    return total / count


def _orientable(tris):
    """True when no directed edge is used twice (consistent winding)."""
    d = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    key = d[:, 0].astype(np.int64) * (int(d.max()) + 1) + d[:, 1]
    return len(np.unique(key)) == len(key)


def _vertex_normals(V, tris, cache=None):
    """Area-weighted vertex normals (the volume gradient, up to 1/3)."""
    P = V[tris]
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]) * 0.5
    if cache is None:
        N = np.zeros_like(V)
        for k in range(3):
            np.add.at(N, tris[:, k], fn)
        return N
    f = fn.ravel()
    return _scatter3(cache['tri3'], np.concatenate([f, f, f]), cache['nv'])


def signed_volume(V, tris):
    P = V[tris]
    return float(np.sum(np.einsum('ij,ij->i', P[:, 0],
                                  np.cross(P[:, 1], P[:, 2]))) / 6.0)


def orient_outward(V, tris):
    """Flip every triangle if the closed surface is wound inward."""
    return tris[:, ::-1].copy() if signed_volume(V, tris) < 0 else tris


def edge_colouring(edges, n_vert):
    """Greedy edge colouring: no two edges of a colour share a vertex.

    Within one colour every constraint is independent, so a colour can
    be projected EXACTLY and all at once -- Gauss-Seidel with the speed
    of a vectorised Jacobi step.  It matters: averaging every vertex
    over its incident edges (plain Jacobi) damps each correction by the
    valence, and on a mesh this stiff that is the difference between
    converging and stalling at a few percent strain.
    """
    order = np.argsort(-np.bincount(edges.ravel(), minlength=n_vert)[
        edges].max(axis=1))
    used = [set() for _ in range(n_vert)]
    colour = np.zeros(len(edges), dtype=int)
    for e in order:
        a, b = int(edges[e, 0]), int(edges[e, 1])
        c = 0
        while c in used[a] or c in used[b]:
            c += 1
        colour[e] = c
        used[a].add(c)
        used[b].add(c)
    return [np.nonzero(colour == c)[0] for c in range(int(colour.max()) + 1)]


def _project_edges(V, groups, sweeps):
    for _ in range(sweeps):
        for e0, e1, rest_g, both in groups:
            d = V[e1] - V[e0]
            L = np.sqrt(np.einsum('ij,ij->i', d, d))
            f = (0.5 * (1.0 - rest_g / np.maximum(L, 1e-12)))[:, None] * d
            V[both] += np.concatenate([f, -f])
    return V


def _scatter3(idx3, vals, nv):
    """Scatter-add 3-vectors by flattened index -- bincount, not add.at.

    `np.add.at` is the obvious spelling and is an order of magnitude
    slower than `np.bincount` for exactly this pattern, which is worth
    caring about because these run inside the iteration loop.
    """
    return np.bincount(idx3, weights=vals, minlength=3 * nv).reshape(nv, 3)


def _sphere_dirs(k):
    """`k` quasi-uniform directions on the sphere (Fibonacci spiral)."""
    i = np.arange(k) + 0.5
    z = 1.0 - 2.0 * i / k
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    th = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)


def _project_hull(V, seam, dirs):
    """Clip every vertex back inside the convex hull of the SEAM ring.

    This is the Demaine-O'Rourke theorem used as a constraint rather than
    as a description: the D-form *is* the convex hull of its seam curve,
    so no point of it may lie outside that hull.

    The hull is never built.  A convex body is the intersection of its
    supporting halfspaces, so sampling directions u and taking
    h(u) = max over the seam of <p,u> gives an outer approximation of the
    hull as {x : <x,u> <= h(u)} -- and clipping into a halfspace is an
    exact projection onto a convex set, which is what makes this stable
    where the local flap term stalls.  Building the hull properly costs
    O(n^4) in the repo's scipy-free `polyhedra.hull`; this costs one
    (m x k) matmul and converges to the same body as k grows.

    Only the excess along each vertex's WORST direction is removed, which
    is the standard alternating projection: correcting every violated
    halfspace at once overshoots at the edges where several meet.  Seam
    vertices generate h(u), so they are already inside and never move.

    Sampling leaves slack: the approximation is an outer one, loose by
    about R*theta^2/2 where theta ~ sqrt(4*pi/k), so a vertex may sit
    ~2% of R outside the true hull at k = 256.  That is the same order
    as the solver's own strain tolerance, and dropping to 128 doubles it
    -- which is what sets the floor on `hull_dirs`.
    """
    h = (V[seam] @ dirs.T).max(axis=0)
    excess = V @ dirs.T - h
    k = np.argmax(excess, axis=1)
    worst = excess[np.arange(len(V)), k]
    hit = worst > 0.0
    if not np.any(hit):
        return V
    V[hit] -= worst[hit][:, None] * dirs[k[hit]]
    return V


def _project_flaps(V, flaps, cache, bending, convexity):
    """Bending stiffness and the local convexity push, in one pass.

    Both terms are the same measurement -- the height h of the flap's
    opposite vertex over the plane of its neighbour triangle -- and
    differ only in which way they act.  Bending is symmetric, a
    linearised dihedral spring with rest angle pi: paper resists being
    bent, and that resistance is what stops a real D-form from
    crumpling, since the isometry constraint cannot supply it (a
    wrinkled sheet is exactly as isometric as a smooth one).  Convexity
    is one-sided, pushing only where the flap is reflex -- the local
    half of Alexandrov's theorem.  Computing the normals once for both
    halves the cost of the most expensive term in the loop.
    """
    if len(flaps) == 0 or (bending <= 0.0 and convexity <= 0.0):
        return V
    a, b, c, d = flaps[:, 0], flaps[:, 1], flaps[:, 2], flaps[:, 3]
    n = np.cross(V[b] - V[a], V[c] - V[a])
    n = n / np.maximum(np.linalg.norm(n, axis=1), 1e-15)[:, None]
    h = np.einsum('ij,ij->i', n, V[d] - V[a])
    amp = bending * h + convexity * np.maximum(h, 0.0)
    step = amp[:, None] * n
    third = (step / 6.0).ravel()
    vals = np.concatenate([(-0.5 * step).ravel(), third, third, third])
    V += _scatter3(cache['flap3'], vals, cache['nv']) / cache['flap_cnt']
    return V


def _fair(V, cache, weight):
    """Umbrella smoothing; the isometry pass undoes the shrinkage."""
    if weight <= 0.0:
        return V
    e0, e1 = cache['e0'], cache['e1']
    acc = _scatter3(cache['edge3'],
                    np.concatenate([V[e1].ravel(), V[e0].ravel()]),
                    cache['nv'])
    V += weight * (acc / cache['valence'][:, None] - V)
    return V


def relax(g, iterations=400, sweeps=3, convexity=0.5, bending=0.15,
          pressure=0.06, fairing=0.0, tol=1.5e-3, closed=None,
          hull_dirs=256, finish=False):
    """Settle the surface: isometry, bending stiffness, convexity.

    Starting from the sphere there is no symmetry left to break, so
    `pressure` defaults to OFF -- the shape is held by the constraints
    the theorem names, not by a force.  (Inflation is also what wrinkles
    it: pushing every vertex outward along its normal is impossible for
    an inextensible sheet, so it buckles, which is why a real paper bag
    creases and a real D-form does not.)  `bending` is the paper's own
    stiffness, and it is what keeps the seam a smooth space curve
    instead of a zigzag.  Set `convexity` to 0 for the anti-D-form,
    which is not convex and has no uniqueness theorem behind it;
    `fairing` then supplies the smoothness that convexity was providing.

    `closed` defaults to whatever `glue` built.  On an OPEN surface the
    pressure term is masked off the free rim: area-weighted normals are
    one-sided there, so inflating them peels the boundary outward into a
    flare instead of lifting the collar.
    """
    V, rest = g.V, g.rest
    if closed is None:
        closed = g.closed
    cache = g.plan(hull_dirs if convexity > 0.0 else 0)
    groups = cache['groups']
    mean_len = float(np.mean(rest))
    drive = (~g.free)[:, None] if not closed else 1.0
    dirs = cache['dirs'] if convexity > 0.0 else None

    for it in range(iterations):
        if pressure > 0.0:
            N = _vertex_normals(V, g.tris, cache)
            s = float(np.max(np.linalg.norm(N, axis=1)))
            if s > 1e-15:
                V += (pressure * mean_len / s) * N * drive
        _project_edges(V, groups, sweeps)
        if dirs is not None:
            _project_hull(V, g.seam, dirs)
        _project_flaps(V, g.flaps, cache, bending, convexity)
        _fair(V, cache, fairing)
        _project_edges(V, groups, 1)
        g.iterations = it + 1
        # `max_strain` is a full pass over the edges, so do not pay for
        # it when the caller has switched the early exit off
        if tol > 0.0 and it > 20 and g.max_strain() < tol:
            break
    if finish:
        # finish on the constraint that has to hold exactly: bending,
        # convexity and fairing are the passes that leave strain behind
        _project_edges(V, groups, _POLISH)

    g.strain = g.max_strain()
    g.V = V - 0.5 * (V.max(axis=0) + V.min(axis=0))
    return g


def _rings(g, which):
    """One piece's vertices, grouped into its constant-depth rings.

    The sheet is ring-meshed, so a piece is already a stack of closed
    curves at fixed depth -- a structured (depth, param) grid, which is
    what makes resampling it onto a different resolution easy.  The seam
    is the depth-0 ring of BOTH pieces.
    """
    sel = (g.piece == which)
    sel[g.seam] = True
    idx = np.nonzero(sel)[0]
    dep = np.round(g.depth[idx], 9)
    rings = []
    for dv in np.unique(dep):
        m = idx[dep == dv]
        order = np.argsort(g.param[m] % 1.0)
        rings.append((float(dv), m[order], (g.param[m] % 1.0)[order]))
    return rings


def _eval_ring(P, params, t):
    """Periodic linear interpolation of a closed ring at parameters `t`."""
    t = np.asarray(t) % 1.0
    j = np.searchsorted(params, t, side='right') - 1
    j0 = j % len(params)
    j1 = (j + 1) % len(params)
    span = (params[j1] - params[j0]) % 1.0
    span = np.where(span < 1e-12, 1.0, span)
    f = ((t - params[j0]) % 1.0) / span
    return P[j0] + f[:, None] * (P[j1] - P[j0])


def warm_start(fine, coarse):
    """Put a fine mesh onto an already-solved coarse surface.

    Convergence gets much harder as the mesh refines -- corrections
    travel about one ring per sweep, so a mesh twice as wide needs
    several times the iterations, and at 240 segments even 3000 of them
    left the seam carrying only 85% of its 4*pi.  Solving small first
    and resampling up sidesteps that: the fine mesh starts on the right
    shape and only has to sharpen it.

    Both meshes carry the same (piece, depth, param) coordinates, and
    each piece is a stack of closed rings in them, so this is a bilinear
    resample -- interpolate the two bracketing coarse rings at the fine
    vertex's param, then blend by depth.
    """
    for which in (0, 1):
        cr = _rings(coarse, which)
        if len(cr) < 2:
            continue
        cd = np.array([r[0] for r in cr])
        sel = (fine.piece == which)
        sel[fine.seam] = True
        idx = np.nonzero(sel)[0]
        t = fine.param[idx] % 1.0
        d = fine.depth[idx]
        k = np.clip(np.searchsorted(cd, d, side='right') - 1, 0, len(cd) - 2)
        w = (d - cd[k]) / np.maximum(cd[k + 1] - cd[k], 1e-12)
        out = np.empty((len(idx), 3))
        for kk in np.unique(k):
            m = k == kk
            lo = _eval_ring(coarse.V[cr[kk][1]], cr[kk][2], t[m])
            hi = _eval_ring(coarse.V[cr[kk + 1][1]], cr[kk + 1][2], t[m])
            out[m] = lo + w[m][:, None] * (hi - lo)
        fine.V[idx] = out
    return fine


def settle(g, iterations=900, pressure=0.4, convexity=0.5, bending=0.2,
           fairing=0.15, hull_dirs=256, closed=None, sweeps=3):
    """Solve a glued pair with the three-stage schedule.

    The stages exist because the terms that FIND the shape and the terms
    that FINISH it are not the same:

    1. inflate -- pressure annealed to zero over six steps, with fairing
       on.  The start is deliberately a shallow dome (see `_place_cap`),
       and the flat pillow it sits near is itself an isometric solution,
       so without a drive the sheet simply stays there.  Annealing means
       the pressure has left by the time the shape is found, so the
       result is held by the metric and not by a force.
    2. settle -- constraints only, moderate bending.
    3. tighten -- bending dropped an order of magnitude so the isometry
       and convexity projections have the last word.

    THE WARM-UP PRESSURE HAS TO BE GENEROUS, and this is the parameter
    that decides whether a hard pair converges at all.  Forms that end
    up fat (a join offset that makes the two outlines disagree strongly)
    need a lot of inflation to reach; too little strands them
    half-inflated, and the wrinkles of that state then survive every
    later stage.  Adding iterations does NOT rescue it -- it is a wrong
    attractor, not a slow one, and 6000 iterations came out worse than
    900.  Across seven curve pairs, dropping the warm-up from 0.4 to 0.2
    took the worst interior defect from 0.022 to 0.199 rad.

    Measured over those seven pairs at 900 iterations: max strain 0.005,
    worst interior angle defect 0.022 rad, seam defect within 2% of 4*pi.
    """
    stages = 6
    warm = max(stages, int(0.4 * iterations))
    per = max(1, warm // stages)
    total = 0
    for k in range(stages):
        relax(g, iterations=per, sweeps=sweeps, bending=bending,
              pressure=pressure * (1.0 - k / stages), convexity=convexity,
              fairing=fairing, tol=0.0, hull_dirs=hull_dirs, closed=closed)
        total += g.iterations
    rest = max(2, iterations - per * stages)
    relax(g, iterations=rest // 2, sweeps=sweeps, bending=0.5 * bending,
          pressure=0.0, convexity=convexity, fairing=0.0, tol=0.0,
          hull_dirs=hull_dirs, closed=closed)
    total += g.iterations
    # only the LAST stage pays for the 60-sweep isometry polish; running
    # it at the end of each stage just fed the next stage's pressure
    relax(g, iterations=rest - rest // 2, sweeps=sweeps,
          bending=0.1 * bending, pressure=0.0, convexity=convexity,
          fairing=0.0, tol=0.0, hull_dirs=hull_dirs, closed=closed,
          finish=True)
    g.iterations += total
    return g


def polish(g, iterations=400, convexity=0.5, bending=0.2, hull_dirs=256,
           closed=None, sweeps=3):
    """Settle a warm-started mesh: no inflation, it is already inflated.

    The pressure stages of `settle` exist only to escape the flat pillow.
    A mesh resampled off a solved coarse surface starts nowhere near it,
    so re-inflating would just push the shape back out of the answer.
    """
    half = max(1, iterations // 2)
    relax(g, iterations=half, sweeps=sweeps, bending=bending, pressure=0.0,
          convexity=convexity, fairing=0.0, tol=0.0, hull_dirs=hull_dirs,
          closed=closed)
    first = g.iterations
    relax(g, iterations=max(1, iterations - half), sweeps=sweeps,
          bending=0.1 * bending, pressure=0.0, convexity=convexity,
          fairing=0.0, tol=0.0, hull_dirs=hull_dirs, closed=closed,
          finish=True)
    g.iterations += first
    return g


# --------------------------------------------------------------------
# The anti-D-form (John Sharp): two annuli glued along their HOLE
# boundaries, outer rims free.  See the header for why every piece of
# this machinery exists.

def _normalize_rows(M, eps=1e-15):
    return M / np.maximum(np.linalg.norm(M, axis=1), eps)[:, None]


def fold_angles(V, flaps, exclude=None):
    """Signed deviation-from-flat angle across each interior edge (rad).

    0 is flat, +/-pi is folded back onto itself.  Unlike the height of
    the far vertex over the near plane -- which vanishes AGAIN at a full
    fold -- this is monotone over the whole range, so it can see a pleat.
    An accordion pleat is exactly isometric and has zero angle defect,
    so this distribution is the only gate that catches one.
    """
    f = flaps if exclude is None else flaps[~exclude]
    if len(f) == 0:
        return np.zeros(0)
    _, _, psi, _ = _hinge_state(V, f)
    return psi


def _hinge_state(V, flaps):
    """Per-flap frame and fold angle: (uh, nh, psi, rho).

    For flap (a, b, c, d): eh along the edge, nh the (a,b,c) normal, uh
    in-plane toward c.  d sits at (q, h) in the (uh, nh) plane; flat is
    q < 0, h = 0, and psi = atan2(h, -q) is the deviation from flat.
    """
    a, b, c, d = flaps[:, 0], flaps[:, 1], flaps[:, 2], flaps[:, 3]
    e = V[b] - V[a]
    eh = _normalize_rows(e)
    cp = V[c] - V[a]
    nh = _normalize_rows(np.cross(e, cp))
    uh = _normalize_rows(cp - np.einsum('ij,ij->i', cp, eh)[:, None] * eh)
    w = V[d] - V[a]
    h = np.einsum('ij,ij->i', w, nh)
    q = np.einsum('ij,ij->i', w, uh)
    psi = np.arctan2(h, -q)
    return uh, nh, psi, np.hypot(h, q)


def _project_hinge(V, hc, k):
    """Rest-flat dihedral hinge on the cached (non-seam) flaps.

    The step rotates d about the edge toward flat: direction
    sin(psi)*uh + cos(psi)*nh (the tangent of that rotation), magnitude
    k * rho * psi -- which reduces to the linearised height push
    k * h for small angles but, crucially, does NOT die away at a
    180-degree fold the way the height does.  Reactions on a, b, c keep
    the pattern (and hence the momentum bookkeeping) of
    `_project_flaps`.  A per-flap weight in the cache grades the
    stiffness (used to stiffen the seam boundary layer, where the tilt
    transition otherwise collapses into a one-ring crease).
    """
    if k <= 0.0 or hc is None or len(hc['flaps']) == 0:
        return V
    uh, nh, psi, rho = _hinge_state(V, hc['flaps'])
    amp = k * hc.get('weight', 1.0) * rho * psi
    step = amp[:, None] * (np.sin(psi)[:, None] * uh
                           + np.cos(psi)[:, None] * nh)
    third = (step / 6.0).ravel()
    vals = np.concatenate([(-0.5 * step).ravel(), third, third, third])
    V += _scatter3(hc['flap3'], vals, hc['nv']) / hc['cnt']
    return V


def _hinge_cache(g, cache):
    """Scatter arrays for the hinge pass, seam flaps excluded.

    The seam of an anti-D-form is a crease: wherever the two hole
    curvatures differ the sheets MUST meet at an angle there (their
    normal curvatures differ), so flattening the seam flaps would fight
    the very curvature the form is made of.
    """
    n = len(g.seam)
    f = g.flaps
    keep = ~((f[:, 0] < n) & (f[:, 1] < n))
    f = f[keep]
    ar = np.arange(3)
    flat3 = lambda i: (i[:, None] * 3 + ar).ravel()   # noqa: E731
    a, b, c, d = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
    # stiffen the seam boundary layer: the tilt transition wants to
    # collapse into a one-ring crease there, and a stronger hinge is
    # what spreads it back over the layer's natural width
    dep = 0.5 * (g.depth[a] + g.depth[b])
    return {
        'flaps': f,
        'flap3': np.concatenate([flat3(d), flat3(a), flat3(b), flat3(c)]),
        'cnt': np.maximum(np.bincount(np.concatenate([d, a, b, c]),
                                      minlength=cache['nv']),
                          1).astype(float)[:, None],
        'nv': cache['nv'],
        'weight': 1.0 + 3.0 * np.maximum(0.0, 1.0 - dep / 0.3),
    }


def _seam_targets(g):
    """Per-vertex flat turning of each piece's seam ring, and the floor.

    Returns (tau_a, tau_b, h): the absolute turning angle of each hole
    boundary in its own flat layout, walked in GLOBAL seam order, and
    the seam edge rest lengths.  max(tau_a, tau_b) is the pointwise
    lower bound on the 3D seam's turning -- the Darboux split
    kappa^2 = kappa_g^2 + kappa_n^2 on each sheet.
    """
    from .curves import turning_angles
    n = len(g.seam)
    taus = []
    for (V2, _T), mp in zip(g.flat, g.maps):
        S = np.asarray(V2, dtype=float)[np.argsort(mp)[:n]]
        taus.append(np.abs(turning_angles(S)))
    key = {(int(a), int(b)): float(r)
           for (a, b), r in zip(g.edges, g.rest)}
    h = np.array([key[(j, j + 1)] for j in range(n - 1)] + [key[(0, n - 1)]])
    return taus[0], taus[1], h


def _tau_required(tau_a, tau_b, p=8.0):
    """Smooth pointwise upper bound on both flat turnings.

    A p-norm rather than the hard max: at a crossover of the two flat
    curvatures the max has a corner, and the sheet that is exactly at
    its bound there has tilt sqrt(1 - (tau_X/tau)^2) -- whose derivative
    is INFINITE at the bound.  The surface normal field then kinks, and
    the ruling construction in `_collar_start` differentiates that
    field, so the corner becomes an order-one ruling error.  The p-norm
    stays strictly above both inputs and is smooth, at the price of a
    few percent of extra seam curvature.
    """
    return (tau_a ** p + tau_b ** p) ** (1.0 / p)


def _chord_targets(h, tau):
    """Second-neighbour distance that realises turning `tau` at vertex i.

    With seam edges h[i-1], h[i] meeting at interior angle pi - tau, the
    law of cosines gives the chord; turning >= tau is chord <= this, so
    the turning floor is a chord CAP -- a pure distance constraint.
    """
    hp = np.roll(h, 1)
    return np.sqrt(hp * hp + h * h + 2.0 * hp * h * np.cos(tau))


def _project_chord_cap(V, ip, inx, D, strength=1.0):
    """Inequality projection: pull i-1, i+1 together where the chord
    exceeds its cap (i.e. where the seam turns less than the floor).

    `strength` < 1 softens it: the floor is a necessary condition for a
    SMOOTH sheet, but a discrete seam vertex may legitimately prefer a
    local crease instead, and holding the floor hard there just parks a
    few percent of strain in the boundary layer forever.
    """
    d = V[inx] - V[ip]
    L = np.linalg.norm(d, axis=1)
    over = L > D
    if not np.any(over):
        return V
    corr = np.zeros(len(D))
    corr[over] = strength * 0.5 * (1.0 - D[over] / L[over])
    step = corr[:, None] * d
    nv = len(V)
    upd = (_scatter3((ip[:, None] * 3 + np.arange(3)).ravel(),
                     step.ravel(), nv)
           - _scatter3((inx[:, None] * 3 + np.arange(3)).ravel(),
                       step.ravel(), nv))
    cnt = np.maximum(np.bincount(np.concatenate([ip[over], inx[over]]),
                                 minlength=nv), 1).astype(float)[:, None]
    V += upd / cnt
    return V


def _polygon_turning_total(G):
    e = np.roll(G, -1, axis=0) - G
    u = _normalize_rows(np.roll(e, 1, axis=0))
    v = _normalize_rows(e)
    return float(np.sum(np.arctan2(
        np.linalg.norm(np.cross(u, v), axis=1),
        np.einsum('ij,ij->i', u, v))))


def _solve_seam(xy, h, tau, iterations=12000, fair=0.06, omega=1.5):
    """The 3D seam ring of an anti-D-form, solved on its own.

    The whole shape hangs on this curve, and it is a tiny problem (the
    ring alone), so it is solved first: a closed polygon with the given
    edge lengths whose turning at vertex i is EXACTLY tau[i] -- the
    least curved curve the two sheets admit -- smoothed toward the
    least wavy torsion distribution by a light fairing term.

    Since sum(tau) > 2*pi the curve cannot be planar; the planar input
    ring is seeded with a z-wave whose wavenumber is the dominant
    harmonic of the turning-excess distribution (an ellipse-against-
    circle mismatch is 2-lobed, a triangle-against-circle one 3-lobed)
    and whose amplitude is bisected so the seeded total turning already
    matches sum(tau).
    """
    n = len(h)
    excess = float(np.sum(tau)) - 2.0 * np.pi
    if excess < 1e-8:       # identical holes: the flat state is exact
        return np.concatenate([xy, np.zeros((n, 1))], axis=1)

    spec = np.abs(np.fft.rfft(tau - tau.mean()))
    hi = min(len(spec) - 1, 8)
    k = 2 + int(np.argmax(spec[2:hi + 1])) if hi >= 2 else 2
    t = np.arange(n) / n

    def with_wave(amp):
        z = amp * np.cos(2.0 * np.pi * k * t)
        return np.concatenate([xy, z[:, None]], axis=1)

    want = float(np.sum(tau))
    lo, hi_amp = 0.0, 2.0 * float(np.mean(np.linalg.norm(
        xy - xy.mean(axis=0), axis=1)))
    for _ in range(40):
        mid = 0.5 * (lo + hi_amp)
        if _polygon_turning_total(with_wave(mid)) < want:
            lo = mid
        else:
            hi_amp = mid
    G = with_wave(0.5 * (lo + hi_amp))

    # Gauss-Seidel with conflict-free groups, NOT averaged Jacobi: the
    # averaged version stalls with turning errors of ~30% of the target,
    # and any vertex whose turning falls BELOW the floor is a point
    # where no smooth sheet can exist -- the whole collar then kinks
    # there no matter what the surface solver does.  Exact sequential
    # projections converge this small system to ~1e-3.
    D = _chord_targets(h, tau)

    def chain_groups(touch):
        groups = []
        for i in range(n):
            for members, verts in groups:
                if not (touch(i) & verts):
                    members.append(i)
                    verts |= touch(i)
                    break
            else:
                groups.append(([i], set(touch(i))))
        return [np.asarray(mem) for mem, _ in groups]

    eg = chain_groups(lambda i: {i, (i + 1) % n})
    cg = chain_groups(lambda i: {(i - 1) % n, (i + 1) % n})
    for it in range(iterations):
        for gi in eg:
            j0, j1 = gi, (gi + 1) % n
            d = G[j1] - G[j0]
            L = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
            f = (omega * 0.5 * (1.0 - h[gi] / L))[:, None] * d
            G[j0] += f
            G[j1] -= f
        for gi in cg:
            j0, j1 = (gi - 1) % n, (gi + 1) % n
            d = G[j1] - G[j0]
            L = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
            f = (omega * 0.5 * (1.0 - D[gi] / L))[:, None] * d
            G[j0] += f
            G[j1] -= f
        # fairing selects the smooth torsion distribution -- ANNEALED to
        # zero, or its shrinkage leaves a permanent percent-level fight
        # with the constraints and neither ever wins
        w = fair * max(0.0, 1.0 - it / max(1, int(0.6 * iterations)))
        if w > 0.0:
            G += w * (0.5 * (np.roll(G, 1, axis=0)
                             + np.roll(G, -1, axis=0)) - G)
    return G


def _march_ring(P1, P2, r1, r2, ext):
    """Place a ring of vertices at exact distances from two placed ones.

    Each new vertex must sit at distance r1 from P1 and r2 from P2 --
    the two flat lengths it will be asked to keep anyway -- which pins
    it to a circle; `ext` (the smooth straight-line continuation of its
    column) picks the point on that circle.  This is discrete
    developable marching: the isometry is exact on the edges to the
    placed ring, and the branch choice is what keeps the sheet from
    creasing at every step.
    """
    w = P2 - P1
    d = np.maximum(np.linalg.norm(w, axis=1), 1e-12)
    wh = w / d[:, None]
    a = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    rho = np.sqrt(np.maximum(r1 * r1 - a * a, 0.0))
    C = P1 + a[:, None] * wh
    q = ext - C
    q = q - np.einsum('ij,ij->i', q, wh)[:, None] * wh
    qn = np.linalg.norm(q, axis=1)
    # a vertex whose continuation is degenerate falls back to the circle
    # centre; the relax sorts out anything that rare
    q = np.where(qn[:, None] > 1e-12, q / np.maximum(qn, 1e-12)[:, None],
                 np.zeros_like(q))
    return C + rho[:, None] * q


def _collar_start(g, G, tau_a, tau_b, tau_curve=None):
    """Wrap both annuli onto the solved seam by developable marching.

    At each seam vertex the curve frame is (t, m, B): tangent, curvature
    direction, and their cross product.  A sheet whose hole turns by
    tau_X there must tilt so the geodesic part of the seam's bending is
    exactly tau_X: with cos(phi) = -tau_X/tau the material direction
    leaves the seam along cos(phi) m - sin(phi) B, and the two pieces
    take OPPOSITE signs of sin(phi) -- which is what separates the
    sheets into the two flaring funnels of Sharp's photographs.

    Two dead ends are worth recording.  A first-order wrap (flat offset
    carried in the rotating seam frame) fails because the annulus is
    several hole radii wide: at that distance the frame rotation costs
    ~300% strain.  The exact envelope developable (rulings n x n')
    fails more subtly: the tilt field varies on the scale of the hole's
    curvature lobes, which drags the rulings nearly tangent to the seam
    (median transversality 0.17 measured) and its regression curve into
    the sheet, so the "exact" map folds over itself.  Marching ring by
    ring (`_march_ring`) sidesteps both: each ring is placed at exact
    flat distances from the previous one, the straight continuation of
    each column selects the smooth branch, and the tilt field is only
    ever used to seed ring 1, where its roughness is a one-ring error.
    """
    n = len(g.seam)
    V = g.V
    e = np.roll(G, -1, axis=0) - G
    u = _normalize_rows(np.roll(e, 1, axis=0))
    v = _normalize_rows(e)
    t3 = _normalize_rows(u + v)
    m = _normalize_rows(v - u)
    B = _normalize_rows(np.cross(t3, m))
    # the turning the seam curve was actually SOLVED at -- when the seam
    # carries a margin above the floor, the tilts must be computed
    # against that richer curvature or the wrap fights the curve
    tau_req = (_tau_required(tau_a, tau_b) if tau_curve is None
               else tau_curve)

    for which, sgn, tau_X in ((0, 1.0, tau_a), (1, -1.0, tau_b)):
        V2, _T = g.flat[which]
        V2 = np.asarray(V2, dtype=float)
        mp = g.maps[which]
        c = np.clip(tau_X / np.maximum(tau_req, 1e-9), 0.0, 1.0)
        s = np.sqrt(np.maximum(1.0 - c * c, 0.0))
        mat = -c[:, None] * m - (sgn * s)[:, None] * B    # into the sheet

        # the annulus mesh is rings of n, seam first: local k*n+j is
        # ring k, and the glue phase shift maps local j to global seam
        # index round(param*n) -- march in GLOBAL index order so ring 0
        # is the solved seam as-is
        nloc = len(V2)
        K = nloc // n - 1
        perm = np.empty(n, dtype=int)          # global j -> local j
        perm[np.rint(np.asarray(
            [g.param[mp[j]] for j in range(n)]) * n).astype(int) % n] = \
            np.arange(n)
        flat_r = [V2[k * n + perm] for k in range(K + 1)]

        # the second trilateration distance should be the mesh's own
        # QUAD DIAGONAL, not merely some nearby chord -- strain is
        # measured on mesh edges, so an uncontrolled diagonal is where
        # it all ends up.  Which global direction the diagonal runs
        # depends on whether the glue reversed this piece's traversal,
        # so ask the edge set rather than assume.
        eset = {(int(a), int(b)) for a, b in g.edges}
        ga = int(mp[1 * n + perm[0]])
        fwd = int(mp[0 * n + perm[1]])
        roll_dir = -1 if ((min(ga, fwd), max(ga, fwd)) in eset) else 1

        # the axial sense this sheet leaves the seam plane on
        ctr = G.mean(axis=0)
        _U, _S, Vt = np.linalg.svd(G - ctr, full_matrices=False)
        ax = Vt[2] * np.sign(np.mean(mat @ Vt[2]) or 1.0)

        prev3 = G
        for k in range(1, K + 1):
            r1 = np.linalg.norm(flat_r[k] - flat_r[k - 1], axis=1)
            r2 = np.linalg.norm(flat_r[k] - np.roll(flat_r[k - 1],
                                                    roll_dir, axis=0),
                                axis=1)
            # continuation with the ring-length budget built in: scale
            # the previous ring about its centroid so its length equals
            # this ring's FLAT length (a homothety scales length
            # exactly, and grows the seam wave's amplitude with radius,
            # which is what a real frill does), then spend whatever is
            # left of the column step moving axially off the seam
            # plane.  Straight-line continuation instead makes every
            # sheet a shade too conical, the rings come up ~20% short,
            # and the relax converts that deficit into wrinkles.
            rest = np.linalg.norm(np.roll(flat_r[k], -1, axis=0)
                                  - flat_r[k], axis=1)
            cen = prev3.mean(axis=0)
            L_prev = float(np.sum(np.linalg.norm(
                np.roll(prev3, -1, axis=0) - prev3, axis=1)))
            alpha = float(np.sum(rest)) / max(L_prev, 1e-12)
            radial = (alpha - 1.0) * (prev3 - cen)
            beta = np.sqrt(np.maximum(
                r1 * r1 - np.einsum('ij,ij->i', radial, radial),
                (0.2 * r1) ** 2))
            ext = prev3 + radial + beta[:, None] * ax
            cur3 = _march_ring(prev3, np.roll(prev3, roll_dir, axis=0),
                               r1, r2, ext)
            # the trilateration pins each vertex to its circle but the
            # ring's OWN edge lengths are whatever the branch choice
            # left; alternate projecting them with snapping back onto
            # the circles -- both projections are exact, so this is a
            # clean alternating projection and a dozen rounds suffice
            rest = np.linalg.norm(np.roll(flat_r[k], -1, axis=0)
                                  - flat_r[k], axis=1)
            for _ in range(12):
                d = np.roll(cur3, -1, axis=0) - cur3
                L = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
                f = (0.5 * (1.0 - rest / L))[:, None] * d
                cur3 += 0.5 * (f - np.roll(f, 1, axis=0))
                cur3 = _march_ring(prev3, np.roll(prev3, roll_dir, axis=0),
                                   r1, r2, cur3)
            V[mp[k * n + perm]] = cur3
            prev3 = cur3
    V[:n] = G
    return V


def _piece_rings(g):
    """Each piece's rings as vertex-id arrays in global param order,
    seam ring excluded (it is pinned or handled by its own machinery)."""
    n = len(g.seam)
    rings = []
    dep = np.round(g.depth, 9)
    for which in (0, 1):
        sel = (g.piece == which)
        sel[:n] = False
        idx = np.nonzero(sel)[0]
        for dv in np.unique(dep[idx]):
            ring = idx[dep[idx] == dv]
            rings.append(ring[np.argsort(g.param[ring] % 1.0)])
    return rings


def _ring_lowpass(V, rings, kcut):
    """Keep only azimuthal harmonics <= kcut of each ring's positions.

    Wrinkles are the failure mode of every relaxation tried here: a
    wrinkled ring is exactly as isometric as a curled one, and the local
    hinge cannot coordinate the ring-wide rotation that trades one for
    the other, so PBD walks into the wrinkle and stays.  The collar's
    true shape is low-harmonic (the seam wave plus a few overtones);
    filtering the rings during the growth phase simply denies the
    wrinkle its degrees of freedom until the smooth shape has locked in.
    """
    for ring in rings:
        F = np.fft.rfft(V[ring], axis=0)
        F[kcut + 1:] = 0.0
        V[ring] = np.fft.irfft(F, n=len(ring), axis=0)
    return V


def settle_anti(g, iterations=3400, bending=0.15, margin=1.25, sweeps=3,
                seam_iters=12000, sheet_gap=0.15):
    """Solve an anti-D-form: the seam ring first, then the collar.

    No pressure, no convexity, no hull: the seam's turning floor is what
    keeps the surface off the flat degenerate state, and the rest-flat
    hinge selects the smooth collar out of the many isometric immersions
    (there is no uniqueness theorem to lean on here).  The outer rims
    are genuinely free -- nothing in the loop touches them except the
    isometry sweeps and the hinge.

    `margin` is the factor by which the solved seam curvature exceeds
    the floor, and it is NOT cosmetic: a seam that rides the floor
    exactly has, at every crossover of the two hole curvatures, BOTH
    sheets lying in its osculating plane -- pinch-folded flat against
    each other -- and the collar kinks there no matter how the surface
    is relaxed.  The margin keeps both sheets strictly tilted apart
    along the whole seam.

    The schedule is pin-then-release with a spectral guard: the seam is
    held at the solved curve while the sheets wrap it (their shape is
    found under a low-harmonic filter, which denies wrinkles their
    degrees of freedom -- a wrinkled ring is exactly as isometric as a
    curled one, so nothing else prevents them), then the cap replaces
    the pin and the filter is opened up so the isometry projections
    have the last word.
    """
    n = len(g.seam)
    V = g.V
    tau_a, tau_b, h = _seam_targets(g)
    # the floor enforced by the cap is the honest hard max -- an
    # inequality may be slack, so the margined curve satisfies it
    tau_star = np.maximum(tau_a, tau_b)
    tau_curve = margin * _tau_required(tau_a, tau_b)
    G = _solve_seam(np.asarray(V[g.seam][:, :2]), h, tau_curve,
                    iterations=seam_iters)

    # hybrid start: flat far field carrying the seam wave faded inward,
    # blended over the innermost quarter with the marched (tilt-aware)
    # near field.  `sheet_gap` cants the two far fields to OPPOSITE
    # sides of the seam plane (in units of the hole's mean radius): the
    # sheets are joined only at the seam, so their asymptotic planes
    # need not coincide, and coincident planes are where most of the
    # sheet-through-sheet crossings come from.
    V[:, 2] = 0.0
    F = V.copy()
    S0 = F[:n].copy()
    anchor = np.rint(g.param * n).astype(int) % n
    F += (1.0 - g.depth)[:, None] * (G[anchor] - S0[anchor])
    R = float(np.mean(np.linalg.norm(S0 - S0.mean(axis=0), axis=1)))
    side = np.where(g.piece == 0, 1.0, -1.0)
    F[:, 2] += sheet_gap * R * side * g.depth
    _collar_start(g, G, tau_a, tau_b, tau_curve=tau_curve)
    w = np.clip(1.0 - g.depth / 0.25, 0.0, 1.0)[:, None]
    V[:] = w * V + (1.0 - w) * F

    cache = g.plan(0)
    hc = _hinge_cache(g, cache)
    idx = np.arange(n)
    ip, inx = (idx - 1) % n, (idx + 1) % n
    D = _chord_targets(h, tau_star)
    rings = _piece_rings(g)

    # (fraction of budget, azimuthal harmonic cut, hinge, seam pinned)
    phases = [(0.24, 6, bending, True),
              (0.15, 10, bending, True),
              (0.12, 16, bending, False),
              (0.09, 24, bending / 3.0, False)]
    g.iterations = 0
    for frac, kc, bend, pin in phases:
        for _ in range(max(1, int(frac * iterations))):
            _project_edges(V, cache['groups'], sweeps)
            if pin:
                V[:n] = G
            else:
                for _ in range(4):
                    _project_chord_cap(V, ip, inx, D)
            _project_hinge(V, hc, bend)
            _ring_lowpass(V, rings, kc)
            g.iterations += 1
    # polish: isometry and the floor, nothing else, and MOST of the
    # budget -- edges + cap is a clean pair of projections that grinds
    # the boundary-layer strain down monotonically (and pulls the seam
    # total back toward its -4*pi) long after the shape has stopped
    # changing.  (Softening the cap instead was tried and is WORSE: the
    # seam sags below the floor broadly and loses ~5% of its 4*pi.)
    # The tail is adaptive: the shaping trajectory is chaotic at the
    # +-0.5% level run to run, so the polish keeps going -- within a 3x
    # budget -- until the strain target is actually met.
    _anti_polish_tail(g, cache, ip, inx, D, sweeps,
                      max(1, int(0.40 * iterations)))
    g.strain = g.max_strain()
    g.V = V - 0.5 * (V.max(axis=0) + V.min(axis=0))
    return g


def _anti_polish_tail(g, cache, ip, inx, D, sweeps, base, tol=0.017):
    V = g.V
    done = 0
    while done < base or (done < 3 * base and g.max_strain() > tol):
        for _ in range(50):
            _project_edges(V, cache['groups'], sweeps)
            for _ in range(4):
                _project_chord_cap(V, ip, inx, D)
        done += 50
        g.iterations += 50
    return g


def polish_anti(g, iterations=1400, bending=0.15, sweeps=3):
    """Settle a warm-started anti-D-form mesh: release phases only.

    A mesh resampled off a solved coarse collar already has the shape;
    it needs the fine boundary layer settled and the isometry
    sharpened, so this is `settle_anti` minus the seam solve, the
    start, and the pinned phases.
    """
    n = len(g.seam)
    V = g.V
    tau_a, tau_b, h = _seam_targets(g)
    tau_star = np.maximum(tau_a, tau_b)
    cache = g.plan(0)
    hc = _hinge_cache(g, cache)
    idx = np.arange(n)
    ip, inx = (idx - 1) % n, (idx + 1) % n
    D = _chord_targets(h, tau_star)
    rings = _piece_rings(g)
    g.iterations = 0
    for frac, kc, bend in [(0.35, 16, bending), (0.20, 24, bending / 3.0)]:
        for _ in range(max(1, int(frac * iterations))):
            _project_edges(V, cache['groups'], sweeps)
            for _ in range(4):
                _project_chord_cap(V, ip, inx, D)
            _project_hinge(V, hc, bend)
            _ring_lowpass(V, rings, kc)
            g.iterations += 1
    _anti_polish_tail(g, cache, ip, inx, D, sweeps,
                      max(1, int(0.45 * iterations)))
    g.strain = g.max_strain()
    g.V = V - 0.5 * (V.max(axis=0) + V.min(axis=0))
    return g


def _seg_tri_hits(P0, P1, A, B, C, eps=1e-12):
    """Moller-Trumbore, vectorised: does segment P0-P1 cross triangle ABC."""
    d = P1 - P0
    e1 = B - A
    e2 = C - A
    pvec = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, pvec)
    ok = np.abs(det) > eps
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = P0 - A
    u = np.einsum('ij,ij->i', tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = np.einsum('ij,ij->i', d, qvec) * inv
    t = np.einsum('ij,ij->i', e2, qvec) * inv
    return (ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
            & (t >= 0.0) & (t <= 1.0))


def sheet_crossings(g):
    """Triangle pairs of piece A x piece B that actually intersect.

    Self-intersection is a REAL risk on an anti-D-form collar -- the
    two sheets are tangent along the seam and hover near each other
    beyond it, and nothing in the solver knows about contact -- so it
    is measured honestly rather than assumed away: centroid prefilter,
    then edge-against-triangle tests both ways (a non-coplanar
    triangle pair intersects iff some edge of one crosses the other).
    Triangles touching the seam are excluded: the sheets legitimately
    SHARE the seam line.
    """
    V, T = g.V, g.tris
    n = len(g.seam)
    which = g.piece[T].max(axis=1)
    off_seam = ~np.any(T < n, axis=1)
    TA = T[(which == 0) & off_seam]
    TB = T[(which == 1) & off_seam]
    if len(TA) == 0 or len(TB) == 0:
        return 0
    ca = V[TA].mean(axis=1)
    cb = V[TB].mean(axis=1)
    ra = np.max(np.linalg.norm(V[TA] - ca[:, None, :], axis=2), axis=1)
    rb = np.max(np.linalg.norm(V[TB] - cb[:, None, :], axis=2), axis=1)
    d2 = np.sum((ca[:, None, :] - cb[None, :, :]) ** 2, axis=2)
    lim = (ra[:, None] + rb[None, :]) ** 2
    ia, ib = np.nonzero(d2 <= lim)
    if len(ia) == 0:
        return 0
    A1 = V[TA[ia]]
    B1 = V[TB[ib]]
    hit = np.zeros(len(ia), dtype=bool)
    for k in range(3):
        hit |= _seg_tri_hits(A1[:, k], A1[:, (k + 1) % 3],
                             B1[:, 0], B1[:, 1], B1[:, 2])
        hit |= _seg_tri_hits(B1[:, k], B1[:, (k + 1) % 3],
                             A1[:, 0], A1[:, 1], A1[:, 2])
    return int(np.count_nonzero(hit))


def _selftest():
    ok = True
    from .curves import (curve_points, match_perimeter, resample_arclength,
                         seam_correspondence)
    from .sheet import disc_mesh

    def build(kind_a, kw_a, kind_b, kw_b, n=64, off=0.25, **rel):
        A = resample_arclength(curve_points(kind_a, **kw_a), n)
        B = resample_arclength(curve_points(kind_b, **kw_b), n)
        B, _ = match_perimeter(A, B)
        g = glue(disc_mesh(A), disc_mesh(B), seam_correspondence(n, off))
        return settle(g, **rel)

    g = build('ELLIPSE', dict(aspect=0.6), 'ELLIPSE', dict(aspect=1.0),
              n=64, off=0.25)

    # isometry: the whole point.  Every edge must still have its flat
    # length, or the "paper" stretched and the piece will not fold up.
    good = g.strain < 0.01
    ok &= good
    print(f"solve: isometric to the flat pieces (max strain "
          f"{g.strain:.4f} < 0.01) {'OK' if good else 'FAIL'}")

    # it popped: a converged D-form is genuinely 3D, not the degenerate
    # flat pillow that also satisfies isometry
    asp = g.aspect()
    good = asp > 0.10
    ok &= good
    print(f"solve: popped, bbox aspect {asp:.3f} (>0.10) "
          f"{'OK' if good else 'FAIL'}")

    # developable: curvature only on the seam.  Interior vertices keep
    # the flat angle sum, so their defect is zero to solver tolerance.
    d = g.interior_defect()
    good = d < 0.05
    ok &= good
    print(f"solve: interior angle defect {d:.4f} rad (<0.05) "
          f"{'OK' if good else 'FAIL'}")

    # and the seam carries all of it: Gauss-Bonnet says 4*pi on a sphere
    tot = g.seam_defect_total()
    good = abs(tot - 4 * np.pi) < 0.02 * 4 * np.pi
    ok &= good
    print(f"solve: seam defect total {tot:.4f} (4pi = {4*np.pi:.4f}) "
          f"{'OK' if good else 'FAIL'}")

    # convex, as Demaine-O'Rourke require.  Gated on the BULK of the
    # flaps, not the single worst one: a PBD solution keeps a handful of
    # isolated dimples of a tenth of an edge (~6 degrees) that do not
    # shrink with resolution, while their COUNT does -- measured 18% of
    # flaps reflex at n=48 falling to 1.8% at n=128.  A max over several
    # thousand flaps is an extreme-value statistic, so it is the wrong
    # thing to assert on; the fraction and the 99th percentile are the
    # numbers that actually track whether the surface is convex.
    a, b, c, dd = (g.flaps[:, 0], g.flaps[:, 1], g.flaps[:, 2],
                   g.flaps[:, 3])
    nrm = np.cross(g.V[b] - g.V[a], g.V[c] - g.V[a])
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-15)[:, None]
    viol = np.einsum('ij,ij->i', nrm, g.V[dd] - g.V[a])
    viol /= float(np.mean(g.rest))
    p99 = float(np.percentile(viol, 99))
    frac = float(np.mean(viol > 0.05))
    good = p99 < 0.20 and frac < 0.15
    ok &= good
    print(f"solve: convex (p99 reflex {p99:.3f} of an edge, "
          f"{100*frac:.1f}% of flaps) {'OK' if good else 'FAIL'}")

    # closed and consistently oriented outward
    vol = signed_volume(g.V, g.tris)
    from collections import Counter
    cnt = Counter()
    for t in g.tris:
        for i in range(3):
            cnt[(int(t[i]), int(t[(i + 1) % 3]))] += 1
    manifold = all(v == 1 for v in cnt.values()) and all(
        cnt.get((b_, a_), 0) == 1 for (a_, b_) in cnt)
    good = manifold and vol > 0
    ok &= good
    print(f"solve: closed, oriented outward (V={vol:.4f}) "
          f"{'OK' if good else 'FAIL'}")

    # The join point genuinely changes the shape -- this is THE artistic
    # parameter, so a solver that ignored it would be worthless.
    #
    # The pair has to share no symmetry for this to test anything, and
    # picking it carelessly tests nothing: against a CIRCLE, moving the
    # join is just a rotation of the circle, so every offset gives the
    # same D-form; against a rounded TRIANGLE, offsets 1/3 apart do.
    # Equal volumes are the correct answer in both cases rather than a
    # dead parameter.  An egg has no rotational symmetry at all, so
    # against an ellipse the quarter turn is a real change -- and it is
    # a big one, roughly doubling the enclosed volume.
    ea, kwa = 'ELLIPSE', dict(aspect=0.6)
    eb, kwb = 'EGG', dict(aspect=0.75, egg=0.4)
    g1 = build(ea, kwa, eb, kwb, n=60, off=0.0)
    g2 = build(ea, kwa, eb, kwb, n=60, off=0.25)
    v1, v2 = signed_volume(g1.V, g1.tris), signed_volume(g2.V, g2.tris)
    good = abs(v1 - v2) / max(abs(v1), 1e-9) > 0.2
    ok &= good
    print(f"solve: join offset changes the form (V {v1:.4f} -> {v2:.4f}) "
          f"{'OK' if good else 'FAIL'}")

    # ----------------------------------------------------------- ANTI
    from .curves import ensure_ccw
    from .sheet import annulus_mesh

    def build_anti(outer_a, outer_b, hole_a, hole_b, n, off):
        OA = resample_arclength(ensure_ccw(curve_points(*outer_a[:1],
                                                        **outer_a[1])), n)
        OB = resample_arclength(ensure_ccw(curve_points(*outer_b[:1],
                                                        **outer_b[1])), n)
        HA = resample_arclength(ensure_ccw(curve_points(*hole_a[:1],
                                                        **hole_a[1])),
                                n) * 0.4
        HB = resample_arclength(ensure_ccw(curve_points(*hole_b[:1],
                                                        **hole_b[1])),
                                n) * 0.4
        HB, _ = match_perimeter(HA, HB)
        return settle_anti(glue(annulus_mesh(OA, HA), annulus_mesh(OB, HB),
                                seam_correspondence(n, off), closed=False,
                                hole_seam=True))

    cases = [
        ('ellipse holes 0.6/1.0',
         build_anti(('ELLIPSE', dict(aspect=0.9)),
                    ('ELLIPSE', dict(aspect=0.9)),
                    ('ELLIPSE', dict(aspect=0.6)),
                    ('ELLIPSE', dict(aspect=1.0)), 64, 0.3)),
        ('triangle hole vs circle',
         build_anti(('ELLIPSE', dict(aspect=1.0)),
                    ('SUPERELLIPSE', dict(aspect=1.0, super_n=3.0)),
                    ('ROUNDED_POLYGON', dict(sides=3, corner=0.35)),
                    ('ELLIPSE', dict(aspect=1.0)), 64, 0.0)),
    ]
    for label, ga in cases:
        n = len(ga.seam)

        # isometric: the collar is still paper
        good = ga.strain < 0.02
        ok &= good
        print(f"anti [{label}]: strain {ga.strain:.4f} (<0.02) "
              f"{'OK' if good else 'FAIL'}")

        # genuinely 3D: the flat doubly-covered annulus is isometric
        # too, and this is the gate that catches falling back onto it
        asp = ga.aspect()
        good = asp > 0.12
        ok &= good
        print(f"anti [{label}]: popped, aspect {asp:.3f} (>0.12) "
              f"{'OK' if good else 'FAIL'}")

        # developable away from the seam
        dd = ga.interior_defect()
        good = dd < 0.08
        ok &= good
        print(f"anti [{label}]: interior defect {dd:.4f} (<0.08) "
              f"{'OK' if good else 'FAIL'}")

        # Gauss-Bonnet: the seam carries -4*pi (the two free rims'
        # boundary turning balances it on the chi = 0 collar)
        tot = ga.seam_defect_total()
        good = abs(tot + 4 * np.pi) < 0.05 * 4 * np.pi
        ok &= good
        print(f"anti [{label}]: seam defect {tot:.3f} (-4pi = "
              f"{-4 * np.pi:.3f}) {'OK' if good else 'FAIL'}")

        # NOT pleated.  Strain and defect cannot see a pleat (an
        # accordion is exactly isometric with zero defect), so the gate
        # is the fold-angle distribution across non-seam interior
        # edges; the seam itself is a genuine crease and is exempt.
        f = ga.flaps
        exs = (f[:, 0] < n) & (f[:, 1] < n)
        deg = np.degrees(np.abs(fold_angles(ga.V, f, exclude=exs)))
        p90 = float(np.percentile(deg, 90))
        p99 = float(np.percentile(deg, 99))
        frac = float(np.mean(deg > 30.0))
        good = p99 < 65.0 and float(deg.max()) < 135.0 and frac < 0.08
        ok &= good
        print(f"anti [{label}]: folds p90 {p90:.1f} p99 {p99:.1f} max "
              f"{deg.max():.1f} deg, {100 * frac:.1f}% over 30 "
              f"(p99<65, max<135, <8%) {'OK' if good else 'FAIL'}")

        # the honest number, reported rather than gated: the two
        # zero-thickness sheets do cross in places (real paper rests in
        # contact instead); a smooth collar keeps this modest
        print(f"anti [{label}]: sheet crossings {sheet_crossings(ga)} "
              f"triangle pairs (reported)")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("dform.solve self-test failed")
