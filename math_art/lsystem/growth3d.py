
# Growth algorithms that are not grammars.
#
# Everything in `core.py` rewrites a string and then reads it as turtle
# commands: form is decided by the productions, and the environment is
# not consulted.  The four algorithms here decide form the other way
# round -- the structure grows into a space and its shape is a
# CONSEQUENCE of that space.  They belong beside the L-systems because
# they answer the same artistic brief and skin through the same
# emitters, not because they are the same mathematics.
#
# COLONIZE (Runions, Lane and Prusinkiewicz, 2007).  Scatter attractors
# in an envelope; every node grows toward the mean direction of the
# attractors within its influence radius; attractors within the kill
# radius are removed.  The striking result is that excurrent form (a
# single dominant trunk, like a spruce) and decurrent form (a spreading
# crown, like an oak) EMERGE FROM THE ENVELOPE ALONE -- no habit
# parameter, no apical-dominance dial.  Change the shape the attractors
# occupy and the tree changes character.
#
# SELFORG (Palubicki et al., SIGGRAPH 2009).  Buds compete for LIGHT.
# Shadow is propagated down a voxel grid as s += a * b**(-q), each
# metamer darkening the cone below it; a bud's exposure is what is left.
# Resource then flows by the extended Borchert-Honda model: at every
# branch point one parameter lambda splits vigour between the main axis
# and the lateral,
#
#     v_main = v * lambda * Qm / (lambda * Qm + (1 - lambda) * Ql)
#
# and that single dial sweeps the whole excurrent-to-decurrent range.
# It is the same range COLONIZE gets from the envelope, reached from the
# opposite direction, which is why both are worth having.
#
# DLA.  A random walker sticks where it first touches the cluster.
# Stickiness below 1 lets it slide along and pack tighter, taking the
# form from wispy dendrite to compact moss.
#
# PYTHAGORAS (Bosman, 1942).  Squares on the legs of a right triangle,
# recursively.  Relief and rolled-3-D variants.
#
# All four return the same (nodes, parents, radii) triple, so the
# emitters and the Strahler styling operator do not need to know which
# one produced a structure.
#
# References:
# - Adam Runions, Brendan Lane and Przemyslaw Prusinkiewicz, "Modeling
#   trees with a space colonization algorithm", Eurographics Workshop on
#   Natural Phenomena, 2007.
# - Wojciech Palubicki, Kipp Horel, Steven Longay, Adam Runions, Brendan
#   Lane, Radomir Mech and Przemyslaw Prusinkiewicz, "Self-organizing
#   tree models for image synthesis", ACM TOG 28(3), SIGGRAPH 2009.
# - Rolf Borchert and Hisao Honda, "Control of development in the
#   bifurcating branch system of Tabebuia rosea", Botanical Gazette 145,
#   1984 -- the resource-allocation model lambda comes from.
# - Cecil D. Murray, "The physiological principle of minimum work...",
#   J. General Physiology 9, 1926 -- the pipe-model exponent.
# - T. A. Witten and L. M. Sander, "Diffusion-limited aggregation, a
#   kinetic critical phenomenon", Physical Review Letters 47, 1981.
# - Albert E. Bosman, "Het wondere onderzoekingsveld der vlakke
#   meetkunde", 1942 -- the Pythagoras tree.

import numpy as np

#: Murray's exponent for the pipe model.  2.0 conserves cross-sectional
#: area; measured trees run nearer 2.5, which is the value the 2007
#: paper uses and the one that looks right on a skinned trunk.
MURRAY = 2.5


# ---------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------

ENVELOPES = ("SPHERE", "CROWN", "CONE", "BOX", "CYLINDER", "MUSHROOM")


def envelope_points(kind="CROWN", n=800, seed=0, height=2.0,
                    radius=0.9, base=0.0):
    """Attractors filling a crown shape.

    The envelope is the whole habit control for COLONIZE: a cone gives
    an excurrent spruce, a broad sphere a decurrent oak.  Nothing else
    in the algorithm knows about tree form.

    `base` lifts the crown off the ground, so the structure grows a bare
    trunk up into it.  Without that the root sits inside the widest part
    of the envelope and the tree forks from the soil, which washes out
    the very difference the envelope is supposed to make.
    """
    rng = np.random.default_rng(int(seed))
    out = []
    h, r = float(height), float(radius)
    while len(out) < n:
        p = rng.uniform([-r, -r, 0.0], [r, r, h], size=(max(n, 64), 3))
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        t = z / h
        rad = np.hypot(x, y)
        if kind == "SPHERE":
            c = np.array([0.0, 0.0, h * 0.5])
            keep = np.linalg.norm(p - c, axis=1) <= min(r, h * 0.5)
        elif kind == "CONE":
            keep = rad <= r * (1.0 - t)
        elif kind == "CYLINDER":
            keep = rad <= r
        elif kind == "BOX":
            keep = (np.abs(x) <= r) & (np.abs(y) <= r)
        elif kind == "MUSHROOM":
            keep = rad <= r * np.sqrt(np.clip(t, 0.0, 1.0))
        else:                                    # CROWN: an ovoid
            keep = (rad / r) ** 2 + ((t - 0.55) / 0.45) ** 2 <= 1.0
        out.extend(p[keep].tolist())
    A = np.asarray(out[:n], dtype=float)
    A[:, 2] += float(base)
    return A


# ---------------------------------------------------------------------
# COLONIZE
# ---------------------------------------------------------------------

def colonize(attractors, root=(0.0, 0.0, 0.0), step=0.06, influence=0.55,
             kill=0.12, max_iters=400, gravity=0.0, up=(0, 0, 1)):
    """Space colonization.  Returns (nodes, parents).

    `gravity` biases every step along `up`: positive for a tree reaching
    for light, negative for the mycelial variant, which is the same
    algorithm growing downward and outward.
    """
    A = np.asarray(attractors, dtype=float).copy()
    nodes = [np.asarray(root, dtype=float)]
    parents = [-1]
    u = np.asarray(up, dtype=float)
    u = u / (np.linalg.norm(u) or 1.0)

    # A crown lifted clear of the root is normal -- that gap is what
    # gives a tree its bare trunk.  But if no attractor is within the
    # influence radius, nothing can see the root and the algorithm
    # simply never starts, returning a single node.  Grow a leader
    # toward the nearest attractor until the crown is in reach, which is
    # what a seedling does.
    if len(A):
        guard = int(max_iters)
        while guard > 0:
            d = np.linalg.norm(A - nodes[-1], axis=1)
            j = int(np.argmin(d))
            if d[j] <= influence:
                break
            v = A[j] - nodes[-1]
            n = np.linalg.norm(v)
            if n < 1e-12:
                break
            nodes.append(nodes[-1] + (v / n) * float(step))
            parents.append(len(nodes) - 2)
            guard -= 1

    for _ in range(int(max_iters)):
        if not len(A):
            break
        N = np.asarray(nodes)
        d = np.linalg.norm(A[:, None, :] - N[None, :, :], axis=2)
        near = np.argmin(d, axis=1)
        ok = d[np.arange(len(A)), near] <= influence
        if not ok.any():
            break
        pull = {}
        for ai in np.flatnonzero(ok):
            pull.setdefault(int(near[ai]), []).append(ai)

        grew = False
        for ni, ais in pull.items():
            v = A[ais] - nodes[ni]
            L = np.linalg.norm(v, axis=1)
            v = v[L > 1e-12] / L[L > 1e-12][:, None]
            if not len(v):
                continue
            dirn = v.sum(axis=0) + u * float(gravity) * len(v)
            n = np.linalg.norm(dirn)
            if n < 1e-12:
                continue
            nodes.append(nodes[ni] + (dirn / n) * float(step))
            parents.append(ni)
            grew = True
        if not grew:
            break
        N = np.asarray(nodes)
        dmin = np.linalg.norm(A[:, None, :] - N[None, :, :], axis=2).min(1)
        A = A[dmin > float(kill)]
    return np.asarray(nodes), np.asarray(parents)


# ---------------------------------------------------------------------
# SELFORG -- self-organizing trees
# ---------------------------------------------------------------------

def shadow_grid(nodes, res=24, a=0.6, b=1.8, q=2, extent=None):
    """Voxel shadow: each node darkens a widening cone beneath it.

    s += a * b**(-q) with q the depth below the caster -- so a metamer's
    shade weakens with distance but spreads.  A bud's light exposure is
    what survives, and that competition is what makes the model
    self-organizing rather than merely recursive.
    """
    P = np.asarray(nodes, dtype=float)
    if not len(P):
        return np.zeros((res, res, res)), (np.zeros(3), 1.0)
    lo = P.min(axis=0) - 0.3 if extent is None else extent[0]
    hi = P.max(axis=0) + 0.3 if extent is None else extent[1]
    span = float(np.max(hi - lo)) or 1.0
    grid = np.zeros((res, res, res))
    idx = np.clip(((P - lo) / span * (res - 1)).astype(int), 0, res - 1)
    for (i, j, k) in idx:
        for depth in range(1, int(q) + 2):
            kk = k - depth
            if kk < 0:
                break
            w = int(depth)
            i0, i1 = max(i - w, 0), min(i + w + 1, res)
            j0, j1 = max(j - w, 0), min(j + w + 1, res)
            grid[i0:i1, j0:j1, kk] += float(a) * float(b) ** (-float(depth))
    return grid, (lo, span)


def _light(grid, frame, pts, res):
    lo, span = frame
    idx = np.clip(((np.asarray(pts) - lo) / span * (res - 1)).astype(int),
                  0, res - 1)
    s = grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    return np.clip(1.0 - s, 0.0, 1.0)


def selforg(steps=14, lam=0.5, root=(0.0, 0.0, 0.0), step=0.12,
            angle=42.0, res=20, seed=0, shed=0.0, alpha=1.0):
    """Self-organizing tree.  Returns (nodes, parents).

    `lam` is the Borchert-Honda split.  Above 0.5 the main axis is
    favoured and the tree is EXCURRENT -- a single dominant leader, a
    conifer.  Below 0.5 the laterals win and it is DECURRENT -- a
    spreading, forked crown.  One dial, the whole range.
    """
    rng = np.random.default_rng(int(seed))
    nodes = [np.asarray(root, dtype=float)]
    parents = [-1]
    dirs = [np.array([0.0, 0.0, 1.0])]
    buds = [0]
    vig = {0: float(alpha) * 4.0}

    for _s in range(int(steps)):
        if not buds:
            break
        grid, frame = shadow_grid(nodes, res=res)
        Q = _light(grid, frame, [nodes[b] for b in buds], res)

        new_buds, new_vig = [], {}
        for bi, b in enumerate(buds):
            v = vig.get(b, 0.0) * (0.35 + 0.65 * float(Q[bi]))
            if v < 0.12:
                continue
            h = dirs[b]
            # main axis continues; one lateral departs by `angle`
            axis = np.array([0.0, 0.0, 1.0])
            side = np.cross(h, axis)
            if np.linalg.norm(side) < 1e-9:
                side = np.array([1.0, 0.0, 0.0])
            side = side / np.linalg.norm(side)
            roll = rng.uniform(0, 2 * np.pi)
            side = (side * np.cos(roll)
                    + np.cross(h, side) * np.sin(roll))
            th = np.radians(float(angle))
            lat = h * np.cos(th) + side * np.sin(th)
            lat = lat / (np.linalg.norm(lat) or 1.0)

            # Borchert-Honda: split v between main and lateral by lam,
            # weighted by the light each direction is heading into
            qm = max(float(Q[bi]), 1e-6)
            ql = max(float(Q[bi]) * 0.85, 1e-6)
            den = lam * qm + (1.0 - lam) * ql
            vm = v * (lam * qm) / den
            vl = v * ((1.0 - lam) * ql) / den

            for dirn, vv in ((h, vm), (lat, vl)):
                if vv < 0.12 or vv < float(shed):
                    continue
                nodes.append(nodes[b] + dirn * float(step))
                parents.append(b)
                dirs.append(dirn)
                k = len(nodes) - 1
                new_buds.append(k)
                new_vig[k] = vv * 0.92
        buds, vig = new_buds, new_vig
    return np.asarray(nodes), np.asarray(parents)


# ---------------------------------------------------------------------
# DLA
# ---------------------------------------------------------------------

SEEDS = ("POINT", "LINE", "RING", "DISK", "SPHERE")


def dla(count=900, dim=3, seed_kind="POINT", stickiness=1.0, size=48,
        seed=0, max_walk=4000):
    """Diffusion-limited aggregation on a lattice.

    A walker released from far away sticks where it first touches the
    cluster.  `stickiness` under 1 lets it refuse and keep walking, so
    it penetrates further before attaching -- wispy dendrite at 1.0,
    compact moss well below.
    """
    rng = np.random.default_rng(int(seed))
    dim = 3 if int(dim) >= 3 else 2
    half = size // 2
    occupied = set()

    def put(p):
        occupied.add(tuple(int(c) for c in p))

    if seed_kind == "LINE":
        for x in range(-half, half + 1):
            put((x,) + (0,) * (dim - 1))
    elif seed_kind == "RING":
        for a in np.linspace(0, 2 * np.pi, 4 * half, endpoint=False):
            p = [int(half * 0.6 * np.cos(a)), int(half * 0.6 * np.sin(a))]
            put(p + [0] * (dim - 2))
    elif seed_kind == "DISK":
        for x in range(-half // 2, half // 2 + 1):
            for y in range(-half // 2, half // 2 + 1):
                if x * x + y * y <= (half // 2) ** 2:
                    put([x, y] + [0] * (dim - 2))
    elif seed_kind == "SPHERE":
        for _ in range(400):
            v = rng.normal(size=dim)
            v = v / (np.linalg.norm(v) or 1.0) * (half * 0.5)
            put(v.astype(int))
    else:
        put((0,) * dim)

    offs = []
    for ax in range(dim):
        for s in (-1, 1):
            o = [0] * dim
            o[ax] = s
            offs.append(tuple(o))

    radius = 3.0
    for _ in range(int(count)):
        v = rng.normal(size=dim)
        v = v / (np.linalg.norm(v) or 1.0)
        p = np.round(v * radius).astype(int)
        for _w in range(int(max_walk)):
            touching = any(tuple(p + np.array(o)) in occupied for o in offs)
            if touching and rng.random() <= float(stickiness):
                put(p)
                radius = max(radius, np.linalg.norm(p) + 3.0)
                break
            p = p + np.array(offs[int(rng.integers(len(offs)))])
            if np.linalg.norm(p) > radius * 2.5 + 6:
                break
        if radius > half - 2:
            break
    pts = np.asarray(sorted(occupied), dtype=float)
    if dim == 2:
        pts = np.hstack([pts, np.zeros((len(pts), 1))])
    return pts


# ---------------------------------------------------------------------
# PYTHAGORAS
# ---------------------------------------------------------------------

def pythagoras(depth=10, alpha=45.0, roll=0.0, jitter=0.0, seed=0,
               size=1.0):
    """The Pythagoras tree: squares on the legs of a right triangle.

    Returns (quads, depths) where each quad is 4 points in 3-D.  `roll`
    turns the classic relief into a genuinely spatial figure by rotating
    each generation out of its parent's plane.
    """
    rng = np.random.default_rng(int(seed))
    quads, depths = [], []

    def emit(origin, ex, ey, d):
        if d > int(depth):
            return
        p0 = origin
        p1 = origin + ex
        p2 = origin + ex + ey
        p3 = origin + ey
        quads.append(np.array([p0, p1, p2, p3]))
        depths.append(d)
        if d == int(depth):
            return
        a = np.radians(float(alpha)
                       + (rng.uniform(-jitter, jitter) if jitter else 0.0))
        ca, sa = np.cos(a), np.sin(a)
        # the two child squares sit on the legs of the right triangle
        # raised on the top edge
        left_len = ca
        right_len = sa
        top = p3
        exl = (ex * ca + ey * sa) * left_len
        eyl = (-ex * sa + ey * ca) * left_len
        emit(top, exl, eyl, d + 1)
        apex = top + exl
        exr = (ex * (-sa) + ey * ca) * right_len
        eyr = (-ex * ca - ey * sa) * right_len
        emit(apex, exr, -eyr, d + 1)

    ex = np.array([float(size), 0.0, 0.0])
    ey = np.array([0.0, 0.0, float(size)])
    if roll:
        r = np.radians(float(roll))
        ey = np.array([0.0, np.sin(r) * size, np.cos(r) * size])
    emit(np.zeros(3), ex, ey, 0)
    return quads, np.asarray(depths)


# ---------------------------------------------------------------------
# Shared skinning
# ---------------------------------------------------------------------

def pipe_radii(parents, base=1.0, exponent=MURRAY):
    """Pipe-model radius per node: r^n = r1^n + r2^n up the tree.

    With `exponent` 2 this conserves cross-sectional area; measured
    trees run nearer 2.5, which is what the space-colonization paper
    uses and what looks right once the skeleton is skinned.
    """
    par = np.asarray(parents)
    n = len(par)
    kids = [[] for _ in range(n)]
    for i, p in enumerate(par):
        if p >= 0:
            kids[int(p)].append(i)
    r = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if not kids[i]:
            r[i] = float(base)
        else:
            r[i] = sum(r[c] ** float(exponent)
                       for c in kids[i]) ** (1.0 / float(exponent))
    return r


def fit(points, size=2.0):
    """Centre and scale into the 2 m cube."""
    P = np.asarray(points, dtype=float)
    if not len(P):
        return P
    lo, hi = P.min(axis=0), P.max(axis=0)
    ext = float((hi - lo).max())
    P = P - (lo + hi) / 2.0
    return P * (float(size) / ext) if ext > 1e-12 else P


def _selftest():
    # --- envelopes ------------------------------------------------------
    for kind in ENVELOPES:
        A = envelope_points(kind, n=300, seed=1)
        assert len(A) == 300, kind
        assert np.all(np.isfinite(A)), kind
        assert float(A[:, 2].min()) >= -1e-9, kind
    # a cone really does narrow with height and a cylinder does not
    def top_width(kind):
        A = envelope_points(kind, n=1500, seed=2)
        hi = A[A[:, 2] > 0.75 * A[:, 2].max()]
        return float(np.hypot(hi[:, 0], hi[:, 1]).max()) if len(hi) else 0.0
    assert top_width("CONE") < top_width("CYLINDER")

    # --- COLONIZE -------------------------------------------------------
    # Habit must come from the ENVELOPE, not from a parameter: the same
    # call on a cone and on a broad crown has to give measurably
    # different form.  That is the paper's headline claim.
    def build(kind):
        A = envelope_points(kind, n=700, seed=3, base=0.5)
        return colonize(A, step=0.07, influence=0.7, kill=0.14,
                        max_iters=200)
    nc, pc = build("CONE")
    nb, pb = build("MUSHROOM")
    assert len(nc) > 30 and len(nb) > 30, (len(nc), len(nb))
    assert len(pc) == len(nc) and len(pb) == len(nb)
    assert (pc[1:] >= 0).all(), "every node but the root needs a parent"
    for N in (nc, nb):
        assert np.all(np.isfinite(N))
    # An excurrent tree tapers to a leader; a decurrent one is broadest
    # where it is tallest.  Overall slenderness does NOT separate them --
    # both envelopes reach the same maximum radius, just at different
    # heights -- so measure the crown width near the TOP, which is what
    # the habit actually looks like.
    def top_span(N):
        hi = N[N[:, 2] > N[:, 2].min()
               + 0.8 * (N[:, 2].max() - N[:, 2].min())]
        return (float(np.hypot(hi[:, 0], hi[:, 1]).max())
                if len(hi) else 0.0)
    slender = top_span
    assert top_span(nc) < top_span(nb), (top_span(nc), top_span(nb))

    # gravity biases the growth direction
    A = envelope_points("CROWN", n=500, seed=4)
    up = colonize(A, gravity=0.8, max_iters=120)[0]
    dn = colonize(A, gravity=-0.8, max_iters=120)[0]
    assert float(up[:, 2].mean()) > float(dn[:, 2].mean())

    # --- SELFORG --------------------------------------------------------
    # lambda sweeps excurrent to decurrent.  Excurrent means a tall
    # narrow crown dominated by one leader; decurrent a wide low one.
    shapes = {}
    for lam in (0.2, 0.5, 0.8):
        N, P = selforg(steps=9, lam=lam, seed=5)
        assert len(N) > 5, lam
        assert len(P) == len(N)
        assert np.all(np.isfinite(N)), lam
        w = float(np.hypot(N[:, 0], N[:, 1]).max()) or 1e-9
        shapes[lam] = float(N[:, 2].max()) / w
    assert shapes[0.8] > shapes[0.2], shapes

    # the shadow grid must actually darken beneath a caster
    g, fr = shadow_grid(np.array([[0.0, 0.0, 1.0]]), res=12)
    assert float(g.max()) > 0.0
    assert float(g.sum()) > 0.0

    # --- DLA ------------------------------------------------------------
    for kind in SEEDS:
        pts = dla(count=120, dim=3, seed_kind=kind, size=28, seed=6)
        assert len(pts) > 10, kind
        assert np.all(np.isfinite(pts)), kind
        assert pts.shape[1] == 3, kind
    # 2-D stays planar
    flat = dla(count=150, dim=2, size=28, seed=7)
    assert abs(float(flat[:, 2].max() - flat[:, 2].min())) < 1e-9
    # low stickiness packs tighter: same walkers, smaller radius of
    # gyration, which is the whole reason the knob exists
    def gyration(s):
        p = dla(count=250, dim=2, stickiness=s, size=40, seed=8)
        return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))
    assert gyration(0.15) < gyration(1.0), (gyration(0.15), gyration(1.0))

    # --- PYTHAGORAS -----------------------------------------------------
    for d in (4, 7, 10):
        q, dep = pythagoras(depth=d)
        assert len(q) == 2 ** (d + 1) - 1, (d, len(q))
        assert dep.max() == d
        for quad in q[:20]:
            assert quad.shape == (4, 3)
            # every square must be a genuine square: equal sides
            s = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i]))
                 for i in range(4)]
            assert max(s) - min(s) < 1e-9 * max(max(s), 1.0), s
    # squares shrink by cos/sin of the angle, so a 45-degree tree halves
    # its area every level
    q45, d45 = pythagoras(depth=6, alpha=45.0)
    def area(quad):
        return float(np.linalg.norm(quad[1] - quad[0])) ** 2
    a0 = np.mean([area(q) for q, dd in zip(q45, d45) if dd == 0])
    a1 = np.mean([area(q) for q, dd in zip(q45, d45) if dd == 1])
    assert abs(a1 / a0 - 0.5) < 1e-9, (a0, a1)
    # roll takes it out of the plane
    flatq, _d = pythagoras(depth=5, roll=0.0)
    rolled, _d = pythagoras(depth=5, roll=35.0)
    fy = np.vstack(flatq)[:, 1]
    ry = np.vstack(rolled)[:, 1]
    assert abs(float(fy.max() - fy.min())) < 1e-9
    assert float(ry.max() - ry.min()) > 0.1

    # --- pipe model -----------------------------------------------------
    N, P = colonize(envelope_points("CROWN", n=400, seed=9), max_iters=120)
    r = pipe_radii(P, exponent=2.0)
    assert len(r) == len(N)
    assert r[0] == r.max(), "the trunk must be the thickest"
    kids = [[] for _ in range(len(P))]
    for i, p in enumerate(P):
        if p >= 0:
            kids[int(p)].append(i)
    forks = [i for i, k in enumerate(kids) if len(k) >= 2]
    assert forks, "a colonized tree should fork"
    for i in forks[:20]:
        assert abs(r[i] ** 2 - sum(r[c] ** 2 for c in kids[i])) < 1e-9
    # a higher Murray exponent gives a relatively slimmer trunk
    assert pipe_radii(P, exponent=3.0)[0] < r[0]

    # --- fitting --------------------------------------------------------
    F = fit(N, 2.0)
    assert abs(float((F.max(0) - F.min(0)).max()) - 2.0) < 1e-9

    print(f"growth3d: OK -- habit emerges from the envelope (crown "
          f"width aloft: cone {slender(nc):.2f} vs mushroom "
          f"{slender(nb):.2f}), "
          f"lambda sweeps excurrent/decurrent {shapes}, DLA stickiness "
          f"packs tighter, Pythagoras squares are square and halve in "
          f"area, pipe model conserves area at {len(forks)} forks")
