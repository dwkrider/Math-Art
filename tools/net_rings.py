"""Candidate natural-tiling faces from a net's own rings.

Not every RCSR net has a 3dt input file, and a few that do carry it in
an incompatible setting (jsa) or with extra ring orbits the tiling does
not use (ocu).  For those nets the faces of the natural tiling must be
computed from the net itself.  The literature says how: the faces are
the net's small STRONG RINGS -- cycles that are not sums (symmetric
differences) of any number of smaller cycles -- and the natural tiling
is built from them subject to Blatov's rules (a)-(e).

This module computes, for a net in the exact integer-24ths form of
`tools/tile_pearce.py`, the strong-ring orbits up to the face sizes the
net's published tiling signature names.  It does NOT decide which
orbits are the tiling's faces: `tile_pearce.tiles_of` already tries
subsets of seed orbits and gates every attempt on the published
signature, so this module only has to propose a small, correct
candidate set.

Method, all exact integer arithmetic on the 1/24 grid:

  1. build a finite chunk of the net ((blocks+1)^3 cells of vertices
     with their edges);
  2. enumerate simple cycles up to the maximum face size, anchored at
     the central cell's vertices, pruned by BFS distance-to-return;
     deduplicate cycles into translation classes (shapes);
  3. classify each shape strong or weak by ascending-size Gaussian
     elimination over GF(2): a cycle is strong exactly when it is not
     in the span of all strictly smaller cycles (all translates of all
     smaller shapes that fit in the chunk).  The test instance is the
     most central translate, so the span is not starved at the chunk
     boundary;
  4. group the strong shapes into space-group orbits and return one
     representative per orbit, sizes restricted to the published
     signature's face sizes.

References:
- R. Goetzke, H.-J. Klein, "Properties and efficient algorithmic
  determination of different classes of rings in finite and infinite
  polyhedral networks", Journal of Non-Crystalline Solids 127, 1991,
  pp. 215-220 (ring / strong-ring definitions).
- V. A. Blatov, O. Delgado-Friedrichs, M. O'Keeffe, D. M. Proserpio,
  "Three-periodic nets and tilings: natural tilings for nets", Acta
  Crystallographica A63, 2007, pp. 418-425 (natural tilings; faces are
  essential rings; the TOPOS algorithm this follows in outline).
"""
import collections
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "math_art"))
import pearce_net as pn

GRID = 24

#: hard caps -- a net that outgrows them is reported, not guessed at
MAX_SHAPES = 20000
MAX_PATHS = 4000000


def chunk_vertices(sites, edgeset, blocks=3):
    """(vertex set, adjacency) for (blocks+1)^3 cells of the net."""
    lim = blocks * GRID
    P = set()
    for s in sites:
        for i, j, k in itertools.product(range(blocks + 1), repeat=3):
            p = (s[0] + GRID * i, s[1] + GRID * j, s[2] + GRID * k)
            if all(0 <= c <= lim for c in p):
                P.add(p)
    offs = collections.defaultdict(list)
    for s, d in edgeset:
        offs[s].append(d)
    adj = {}
    for p in P:
        key = tuple(c % GRID for c in p)
        nbrs = []
        for d in offs[key]:
            q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
            if q in P:
                nbrs.append(q)
        adj[p] = nbrs
    return P, adj


def _bfs_dist(adj, src, cap):
    dist = {src: 0}
    frontier = [src]
    d = 0
    while frontier and d < cap:
        d += 1
        nxt = []
        for p in frontier:
            for q in adj[p]:
                if q not in dist:
                    dist[q] = d
                    nxt.append(q)
        frontier = nxt
    return dist


def _shape_key(cycle):
    """Translation-normalized canonical form of a cycle."""
    lo = [min(v[i] for v in cycle) for i in range(3)]
    shift = [-(lo[i] // GRID) * GRID for i in range(3)]
    f = [tuple(v[i] + shift[i] for i in range(3)) for v in cycle]
    return pn.canonical_ring(f)


def cycle_shapes(sites, edgeset, max_len, blocks=3):
    """All translation classes of simple cycles up to max_len, or None.

    Cycles are anchored at the central cell's vertices; every cycle of
    the infinite net has a translate through the central cell, so no
    class is missed.  Returns {size: [cycle-as-vertex-tuple, ...]} or
    None if the enumeration outgrows the caps."""
    P, adj = chunk_vertices(sites, edgeset, blocks)
    mid = blocks // 2
    anchors = sorted(p for p in P
                     if all(mid * GRID <= c < (mid + 1) * GRID for c in p))
    shapes = {}
    budget = [MAX_PATHS]

    def dfs(v0, path, seen, dist):
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        cur = path[-1]
        room = max_len - len(path)
        for q in adj[cur]:
            if q == v0 and len(path) >= 3:
                key = _shape_key(path)
                if key not in shapes:
                    shapes[key] = tuple(path)
                    if len(shapes) > MAX_SHAPES:
                        return False
                continue
            if q in seen or room < 1:
                continue
            dq = dist.get(q)
            if dq is None or dq > room - 0:
                continue
            seen.add(q)
            path.append(q)
            if not dfs(v0, path, seen, dist):
                return False
            path.pop()
            seen.remove(q)
        return True

    for v0 in anchors:
        dist = _bfs_dist(adj, v0, max_len)
        if not dfs(v0, [v0], {v0}, dist):
            return None
    out = collections.defaultdict(list)
    for cyc in shapes.values():
        out[len(cyc)].append(cyc)
    return dict(out)


def _edge_bits(cycle, eindex):
    """A cycle as a GF(2) vector (int bitmask) over chunk edges."""
    m = 0
    n = len(cycle)
    for k in range(n):
        a, b = cycle[k], cycle[(k + 1) % n]
        e = (a, b) if a < b else (b, a)
        i = eindex.get(e)
        if i is None:
            return None
        m |= 1 << i
    return m


def _translates(cycle, blocks):
    """Every translate of the cycle inside the chunk."""
    lim = blocks * GRID
    lo = [min(v[i] for v in cycle) for i in range(3)]
    hi = [max(v[i] for v in cycle) for i in range(3)]
    out = []
    for i in range(-(lo[0] // GRID), (lim - hi[0]) // GRID + 1):
        for j in range(-(lo[1] // GRID), (lim - hi[1]) // GRID + 1):
            for k in range(-(lo[2] // GRID), (lim - hi[2]) // GRID + 1):
                t = (GRID * i, GRID * j, GRID * k)
                out.append(tuple((v[0] + t[0], v[1] + t[1], v[2] + t[2])
                                 for v in cycle))
    return out


def _central_translate(cycle, blocks):
    """The translate whose centroid is nearest the chunk center."""
    lim = blocks * GRID
    best, best_d = None, None
    for c in _translates(cycle, blocks):
        cen = [sum(v[i] for v in c) / len(c) for i in range(3)]
        d = sum((cen[i] - lim / 2.0) ** 2 for i in range(3))
        if best_d is None or d < best_d:
            best, best_d = c, d
    return best


def strong_shapes(sites, edgeset, max_len, blocks=3):
    """Strong-ring translation classes up to max_len, or None on cap.

    Ascending-size GF(2) elimination: a size-s cycle is strong iff its
    most central translate is not in the span of all translates of all
    strictly smaller cycles.  All size-s translates then join the basis
    before size s+1 is examined."""
    got = cycle_shapes(sites, edgeset, max_len, blocks)
    if got is None:
        return None
    P, adj = chunk_vertices(sites, edgeset, blocks)
    eindex = {}
    for p in P:
        for q in adj[p]:
            e = (p, q) if p < q else (q, p)
            if e not in eindex:
                eindex[e] = len(eindex)

    pivots = {}                          # highest set bit -> row

    def reduce(m):
        while m:
            h = m.bit_length() - 1
            row = pivots.get(h)
            if row is None:
                return m
            m ^= row
        return 0

    def insert(m):
        m = reduce(m)
        if m:
            pivots[m.bit_length() - 1] = m

    strong = {}
    for size in sorted(got):
        keep = []
        for cyc in got[size]:
            probe = _central_translate(cyc, blocks)
            m = _edge_bits(probe, eindex)
            if m is None:
                continue
            if reduce(m):
                keep.append(cyc)
        if keep:
            strong[size] = keep
        for cyc in got[size]:
            for t in _translates(cyc, blocks):
                m = _edge_bits(t, eindex)
                if m is not None:
                    insert(m)
    return strong


def orbit_reps(cycles, gops):
    """One representative per space-group orbit of the given cycles."""
    seen = {}
    for cyc in cycles:
        key = min(_shape_key([tuple(sum(R[i][k] * v[k] for k in range(3))
                                    + 2 * t[i] for i in range(3))
                              for v in cyc])
                  for R, t in gops)
        seen.setdefault(key, cyc)
    return [list(c) for c in seen.values()]


def ring_seeds(net, face_sizes, blocks=3):
    """Candidate seed faces for tile_pearce, or (None, why).

    `net` is tile_pearce.build_net24's (sites, edgeset, step, gops);
    `face_sizes` the face sizes of the net's published tiling
    signature.  Returns one representative cycle per strong-ring orbit
    whose size occurs in the signature."""
    sites, edgeset, _step, gops = net
    max_len = max(face_sizes)
    strong = strong_shapes(sites, edgeset, max_len, blocks)
    if strong is None:
        return None, "ring enumeration outgrew its caps"
    picked = [c for size in sorted(strong) if size in face_sizes
              for c in strong[size]]
    if not picked:
        return None, "no strong rings at the signature's face sizes"
    reps = orbit_reps(picked, gops)
    return reps, "%d strong-ring orbit(s), sizes %s" % (
        len(reps), sorted({len(r) for r in reps}))


def _selftest():
    """srs: the natural tiling's faces are its 10-rings (one orbit)."""
    import rcsr_nets
    import tile_pearce as tp
    recs = {r["name"]: r for r in rcsr_nets.parse_cgd()
            if r["name"] in ("srs", "dia")}
    for name, want_sizes, want_orbits in (("dia", (6,), 1),
                                          ("srs", (10,), 1)):
        net, why = tp.build_net24(recs[name])
        assert net is not None, (name, why)
        reps, note = ring_seeds(net, set(want_sizes))
        assert reps is not None, (name, note)
        assert len(reps) == want_orbits, (name, note)
        assert {len(r) for r in reps} == set(want_sizes), (name, note)
    print("net_rings: OK (dia 6-ring and srs 10-ring orbits recovered)")
