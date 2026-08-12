
# Leaf venation by space colonization (Runions et al., 2005).
#
# THE IDEA.  Veins are not drawn; they are competed for.  Scatter auxin
# sources through the blade, and let every vein node grow toward the
# sources that are closest to IT.  A source is removed once a vein
# reaches within a kill radius.  The branching pattern, the spacing, the
# way secondary veins fill the gaps left by primaries -- none of it is
# specified.  It falls out of the competition for sources.
#
# OPEN vs CLOSED VENATION is the substantive choice, and it is one line
# of the algorithm:
#
#   * OPEN (dichotomous) -- each source influences only the SINGLE
#     nearest vein node.  Veins never rejoin, so the result is a tree.
#     Ginkgo, and most conifer and monocot venation.
#   * CLOSED (reticulate) -- each source influences every node in its
#     RELATIVE NEIGHBOURHOOD: node v qualifies when no other node w is
#     closer to both, i.e. d(s,v) < max(d(s,w), d(v,w)) for all w.  Two
#     veins can then be drawn to the same source and MEET, closing a
#     loop.  That is where areoles come from, and it is why a dicot leaf
#     is a network rather than a tree.
#
# The relative-neighbourhood test is O(S * N^2) if written naively,
# which is hopeless at a few thousand nodes.  It is only ever applied to
# nodes already within the influence radius of the source, which is a
# handful, so the cost collapses.
#
# References:
# - Adam Runions, Martin Fuhrer, Brendan Lane, Pavol Federl, Anne-Gaelle
#   Rolland-Lagan and Przemyslaw Prusinkiewicz, "Modeling and
#   visualization of leaf venation patterns", ACM Transactions on
#   Graphics 24(3), SIGGRAPH 2005, pp. 702-711.
# - Adam Runions, Brendan Lane and Przemyslaw Prusinkiewicz, "Modeling
#   trees with a space colonization algorithm", Eurographics Workshop on
#   Natural Phenomena, 2007 -- the same algorithm applied to branches.
# - Gabriel Toussaint, "The relative neighbourhood graph of a finite
#   planar set", Pattern Recognition 12, 1980 -- the neighbourhood test
#   the closed model uses.
# - Cecil D. Murray, "The physiological principle of minimum work
#   applied to the angle of branching of arteries", J. General
#   Physiology 9, 1926 -- the vein-width law.

import numpy as np

OPEN, CLOSED = "OPEN", "CLOSED"

#: How many generations of a tip's own lineage are ineligible
#: for anastomosis.  A vein is always within a step or two of
#: what it just grew; fusing with that is self-intersection,
#: not reticulation.
_ANCESTRY = 6


def inside(outline, pts):
    """Point-in-polygon by ray casting, vectorised over `pts`."""
    P = np.asarray(outline, dtype=float)
    Q = np.asarray(pts, dtype=float)
    x, y = Q[:, 0], Q[:, 1]
    x0, y0 = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    hit = np.zeros(len(Q), dtype=bool)
    for a, b, c, d in zip(x0, y0, x1, y1):
        cross = ((b > y[:, None]).ravel() != (d > y))
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = (c - a) * (y - b) / np.where(d != b, d - b, 1e-30) + a
        hit ^= cross & (x < xint)
    return hit


def sample_sources(outline, n, seed=0, inset=0.0):
    """`n` auxin sources scattered uniformly INSIDE the blade.

    Rejection sampling against the outline's bounding box.  Uniform in
    area is the right distribution: the sources stand for a uniform
    demand for vascular supply, and biasing them would prejudge the
    pattern the algorithm is supposed to discover.
    """
    P = np.asarray(outline, dtype=float)
    lo, hi = P.min(axis=0), P.max(axis=0)
    if inset:
        c = 0.5 * (lo + hi)
        P = c + (P - c) * (1.0 - float(inset))
        lo, hi = P.min(axis=0), P.max(axis=0)
    rng = np.random.default_rng(int(seed))
    out = np.zeros((0, 2))
    tries = 0
    while len(out) < n and tries < 200:
        cand = rng.uniform(lo, hi, size=(max(n * 3, 64), 2))
        keep = cand[inside(P, cand)]
        out = np.vstack([out, keep])
        tries += 1
    return out[:n]


def grow(outline, sources=None, n_sources=500, mode=OPEN, step=0.02,
         kill=0.035, influence=0.35, max_iters=600, seed=0, root=None,
         merge=None):
    """Space colonization.  Returns (nodes, edges, parents).

    `step` is how far a node advances per iteration, `kill` the radius
    at which a source is consumed, `influence` how far a source can
    reach.  The three together set the vein density: `kill` is the one
    that decides how fine the network is, because it is what stops
    veins crowding.

    In CLOSED mode a tip that arrives within `merge` of an existing vein
    ANASTOMOSES with it -- an edge is added instead of a node.  That
    fusion is what actually closes the network: the relative-
    neighbourhood rule alone lets two veins pursue the same source, but
    without letting them fuse on arrival the result is still a tree, and
    a closed venation model that never closes is just an open one with
    extra work.
    """
    P = np.asarray(outline, dtype=float)
    if sources is None:
        sources = sample_sources(P, int(n_sources), seed=seed)
    S = np.asarray(sources, dtype=float).copy()

    if root is None:
        # the petiole: the lowest point of the blade
        root = P[np.argmin(P[:, 1])]
    nodes = [np.asarray(root, dtype=float)]
    parents = [-1]
    edges = []
    seen = set()
    # 2x the step reticulates convincingly without the network
    # degenerating; areoles scale with it (3, 26, 90, 155 loops
    # at 1.4, 2.0, 2.5, 3.0 times the step on the test blade).
    merge = float(step) * 2.0 if merge is None else float(merge)

    for _it in range(int(max_iters)):
        if not len(S):
            break
        N = np.asarray(nodes)
        # distance from every source to every node
        d = np.linalg.norm(S[:, None, :] - N[None, :, :], axis=2)

        attract = {}
        if mode == OPEN:
            near = np.argmin(d, axis=1)
            ok = d[np.arange(len(S)), near] <= influence
            for si in np.flatnonzero(ok):
                attract.setdefault(int(near[si]), []).append(si)
        else:
            # CLOSED: every node in the source's relative neighbourhood.
            # Only nodes already within the influence radius can qualify,
            # which is what keeps the O(N^2) test affordable.
            for si in range(len(S)):
                cand = np.flatnonzero(d[si] <= influence)
                if not len(cand):
                    continue
                C = N[cand]
                dsv = d[si][cand]
                # d(v,w) between candidates
                dvw = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
                # v qualifies unless some w is closer to both
                worse = np.maximum(dsv[None, :], dvw) < dsv[:, None]
                np.fill_diagonal(worse, False)
                for k in np.flatnonzero(~worse.any(axis=1)):
                    attract.setdefault(int(cand[k]), []).append(si)

        if not attract:
            break

        grew = False
        for ni, sis in attract.items():
            v = S[sis] - nodes[ni]
            L = np.linalg.norm(v, axis=1)
            v = v[L > 1e-12] / L[L > 1e-12][:, None]
            if not len(v):
                continue
            dirn = v.sum(axis=0)
            n = np.linalg.norm(dirn)
            if n < 1e-12:
                continue
            new = nodes[ni] + (dirn / n) * float(step)
            if not inside(P, new[None, :])[0]:
                continue
            if mode == CLOSED:
                # Exclude the tip's own recent ancestry.  A growing vein
                # is always within a step or two of the nodes it just
                # laid down, so without this the tip fuses with itself,
                # stops advancing, and the whole network stalls -- at a
                # merge radius of 2*step that collapsed 251 nodes to 74.
                # Anastomosis has to mean meeting ANOTHER vein.
                anc, a = set(), ni
                for _ in range(_ANCESTRY):
                    if a < 0:
                        break
                    anc.add(a)
                    a = int(parents[a])
                dd = np.linalg.norm(np.asarray(nodes) - new, axis=1)
                cand = [int(c) for c in np.flatnonzero(dd <= merge)
                        if c not in anc]
                fused = False
                for j in sorted(cand, key=lambda c: dd[c]):
                    key = (min(ni, j), max(ni, j))
                    if key not in seen:
                        seen.add(key)
                        edges.append((ni, j))
                        grew = fused = True
                        break
                # Only skip growth when a fusion actually happened.
                # Skipping whenever a CANDIDATE existed stalls the tip
                # forever against an edge it has already made -- which
                # is what pinned the network at 74 nodes.
                if fused:
                    continue
            nodes.append(new)
            parents.append(ni)
            k = len(nodes) - 1
            seen.add((min(ni, k), max(ni, k)))
            edges.append((ni, k))
            grew = True
        if not grew:
            break

        # consume sources the veins have reached
        N = np.asarray(nodes)
        d2 = np.linalg.norm(S[:, None, :] - N[None, :, :], axis=2).min(axis=1)
        S = S[d2 > float(kill)]

    return np.asarray(nodes), edges, np.asarray(parents)


def vein_widths(nodes, edges, parents, base=1.0, exponent=2.0):
    """Murray's law up the tree: w_parent^n = sum of w_child^n.

    Applied to the edge list by counting how much of the network each
    edge carries.  With exponent 2 the cross-sectional AREA is conserved
    at a fork, which is the same rule the Strahler styling uses and the
    reason the veins taper convincingly without a taper parameter.
    """
    n = len(nodes)
    kids = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if p >= 0:
            kids[p].append(i)
    w = np.zeros(n)
    order = sorted(range(n), key=lambda i: -i)     # children before parents
    for i in order:
        if not kids[i]:
            w[i] = float(base)
        else:
            w[i] = sum(w[c] ** float(exponent)
                       for c in kids[i]) ** (1.0 / float(exponent))
    return w


def has_cycle(nodes, edges):
    """Does the vein network contain a loop?

    The whole point of closed venation is that it does; open venation
    must not.  Union-find over the edge list.
    """
    par = list(range(len(nodes)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        par[ra] = rb
    return False


def _selftest():
    # A blade to grow into: a plain convex-ish leaf outline.
    th = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    blade = np.stack([0.35 * np.sin(th), 0.5 + 0.5 * np.cos(th + np.pi)],
                     axis=1)

    # --- point-in-polygon ---------------------------------------------
    assert inside(blade, np.array([[0.0, 0.5]]))[0]
    assert not inside(blade, np.array([[5.0, 5.0]]))[0]
    pts = sample_sources(blade, 200, seed=1)
    assert len(pts) == 200
    assert inside(blade, pts).all(), "sources must lie inside the blade"

    # --- open venation is a TREE ---------------------------------------
    nodes, edges, parents = grow(blade, n_sources=250, mode=OPEN,
                                 step=0.03, kill=0.06, influence=0.4,
                                 max_iters=200, seed=2)
    assert len(nodes) > 20, len(nodes)
    assert len(edges) == len(nodes) - 1, (len(edges), len(nodes))
    assert not has_cycle(nodes, edges), "open venation must not loop"
    assert inside(blade, nodes[1:]).all(), "veins must stay in the blade"
    # it must actually spread, not sit at the petiole
    assert float(nodes[:, 1].max() - nodes[:, 1].min()) > 0.3

    # --- closed venation makes LOOPS ------------------------------------
    # This is the substantive difference between the two models, so it is
    # the thing worth asserting: the relative-neighbourhood rule lets two
    # veins be drawn to one source and meet.
    n2, e2, p2 = grow(blade, n_sources=250, mode=CLOSED,
                      step=0.03, kill=0.06, influence=0.4,
                      max_iters=200, seed=2)
    assert len(n2) > 20, len(n2)
    assert inside(blade, n2[1:]).all()
    # THE defining property: closed venation forms areoles.  Without
    # anastomosis on arrival the relative-neighbourhood rule alone still
    # yields a tree, which would make the two modes differ only in
    # density -- so this is the assertion that the model is really the
    # closed one.
    assert has_cycle(n2, e2), "closed venation must form loops"
    assert len(e2) > len(n2) - 1, (len(e2), len(n2))
    # and the number of independent loops is edges - nodes + 1
    loops = len(e2) - len(n2) + 1
    assert loops >= 10, loops
    # Fusing must not COST the network its growth: a tip that stalls
    # against an edge it already made pins the whole pattern (it held at
    # 74 nodes against 250).  Closed venation should reach comparable
    # size to open, not a fraction of it.
    assert len(n2) > 0.6 * len(nodes), (len(n2), len(nodes))
    # more merging means more areoles, monotonically
    prev = -1
    for f in (1.4, 2.0, 2.5, 3.0):
        nn, ee, _p = grow(blade, n_sources=250, mode=CLOSED, step=0.03,
                          kill=0.06, influence=0.4, max_iters=200,
                          seed=2, merge=0.03 * f)
        lp = len(ee) - len(nn) + 1
        assert lp > prev, (f, lp, prev)
        prev = lp

    # --- Murray's law ---------------------------------------------------
    w = vein_widths(nodes, edges, parents, exponent=2.0)
    assert len(w) == len(nodes)
    assert w[0] == w.max(), "the petiole must be the thickest vein"
    assert w.min() > 0.0
    # at a genuine fork, area is conserved
    kids = [[] for _ in range(len(nodes))]
    for i, p in enumerate(parents):
        if p >= 0:
            kids[p].append(i)
    forks = [i for i, k in enumerate(kids) if len(k) >= 2]
    assert forks, "an open venation pattern should branch"
    for i in forks[:20]:
        got = w[i] ** 2
        want = sum(w[c] ** 2 for c in kids[i])
        assert abs(got - want) < 1e-9, (i, got, want)

    # a higher exponent makes the trunk relatively thicker
    w3 = vein_widths(nodes, edges, parents, exponent=3.0)
    assert w3[0] < w[0], (w3[0], w[0])

    # --- determinism ----------------------------------------------------
    a = grow(blade, n_sources=120, seed=7, max_iters=60)[0]
    b = grow(blade, n_sources=120, seed=7, max_iters=60)[0]
    assert np.allclose(a, b), "same seed must give the same venation"
    c = grow(blade, n_sources=120, seed=8, max_iters=60)[0]
    assert a.shape != c.shape or not np.allclose(a, c)

    print(f"venation: OK -- sources stay inside the blade, open venation "
          f"is a tree ({len(nodes)} nodes, {len(edges)} edges, no cycle), "
          f"closed reaches {len(n2)} with {loops} areoles, Murray's "
          f"law conserves area at {len(forks)} forks")
