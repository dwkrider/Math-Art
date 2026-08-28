"""The wurtzite (lonsdaleite / RCSR `lon`) net and its saddle cells.

Pearce's Universal Node is cubic, and every net in `math_art/pearce_net.py`
lives on the integer eighth-of-a-cubic-cell lattice.  The wurtzite solids
of Table 8.1 do not: a single wurtzite CELL is fine in cubic coordinates
(all its branches are tetrahedral, which the <111> class supplies), but
the PACKING needs wurtzite's hexagonal ABAB stacking, whose lattice is
not a sublattice of the cubic grid.  That is why entry 6 builds correctly
and then fails to tile in `pearce_tiling`.

This script builds the net in its own hexagonal basis and finds its
cells, as the groundwork for giving those solids a real packing.

VERIFIED HERE (run it):
  * bond length is exactly (3/8)c and EVERY bond angle is 109.4712
    degrees, so the net is lonsdaleite and not some near-miss;
  * a 3-faced cell with V=8, E=9, chi=2 and valence {3: 2, 2: 6} --
    Pearce's entry 6, the Wurtzite trihedron, in its natural net;
  * a 5-faced cell with V=12, E=15 and valence {3: 6, 2: 6} -- Pearce's
    entry 25, the Wurtzite pentahedron, which is currently unresolved.

AND THE REASON A SINGLE CELL WILL NEVER TILE THIS NET.  The 3-faced
closures come in three volume classes in the exact ratio 1 : 3.5 : 8,
and none divides the hexagonal cell a whole number of times (13.5,
3.857, 1.688).  Two COMBINATIONS do close it exactly:

    2 small + 1 mid + 1 large = 1.414214  (the cell volume, to 6e-6)
    3 small + 3 mid           = 1.414214  (to 1.6e-5)

So wurtzite is a multi-cell space filling -- Table 8.2 territory -- and
matches the note on DaveMakesStuff's "Saddle Polyhedra" models that the
wurtzite net gives two polyhedra which tile "separately or in the
Wurtzite Combined form".

References:
- Peter Pearce, "Structure in Nature is a Strategy for Design", The MIT
  Press, 1978, ch. 8 -- entries 6 and 25 of Table 8.1, and Table 8.2's
  space filling systems.
- M. O'Keeffe, M. A. Peskov, S. J. Ramsden & O. M. Yaghi, "The
  Reticular Chemistry Structure Resource (RCSR) Database of, and
  Symbols for, Crystal Nets", Accounts of Chemical Research 41(12),
  2008, pp. 1782-1789 -- the `lon` net and the standard net symbols.
"""

import collections
import itertools

import numpy as np

A = 1.0
C = A * np.sqrt(8.0 / 3.0)
A1 = np.array([A, 0.0, 0.0])
A2 = np.array([-A / 2.0, A * np.sqrt(3.0) / 2.0, 0.0])
A3 = np.array([0.0, 0.0, C])

#: lonsdaleite: four sites per hexagonal cell, ABAB stacking.  In
#: twenty-fourths of the cell these are integral (8, 16, 9, 21, 12), so
#: the net can be carried in exact integer coordinates if wanted.
FRAC = ((1.0 / 3, 2.0 / 3, 0.0), (2.0 / 3, 1.0 / 3, 0.5),
        (1.0 / 3, 2.0 / 3, 3.0 / 8), (2.0 / 3, 1.0 / 3, 7.0 / 8))


def chunk(n=3):
    """Net sites over an n^3 block of hexagonal cells, in Cartesian."""
    P = []
    for i, j, k in itertools.product(range(n), repeat=3):
        for f in FRAC:
            u, v, w = f[0] + i, f[1] + j, f[2] + k
            P.append(u * A1 + v * A2 + w * A3)
    return np.asarray(P)


def adjacency(P, tol=1.05):
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(D, 1e9)
    bond = float(D.min())
    return {i: list(np.where(D[i] < bond * tol)[0])
            for i in range(len(P))}, bond


def _canon(cyc):
    k = cyc.index(min(cyc))
    c1 = tuple(cyc[k:] + cyc[:k])
    return min(c1, tuple([c1[0]] + list(reversed(c1[1:]))))


def rings(adj, length):
    out = set()
    for a0 in adj:
        def dfs(path, seen):
            last = path[-1]
            if len(path) == length:
                if a0 in adj[last]:
                    out.add(_canon(path))
                return
            for q in adj[last]:
                if q not in seen and q >= a0:
                    dfs(path + [q], seen | {q})
        dfs([a0], {a0})
    return out


def ring_edges(cyc):
    n = len(cyc)
    return [tuple(sorted((cyc[i], cyc[(i + 1) % n]))) for i in range(n)]


def closures(pool, nfaces):
    """Closed surfaces of `nfaces` rings, grown by matching edges."""
    by_edge = collections.defaultdict(list)
    for r in pool:
        for e in ring_edges(r):
            by_edge[e].append(r)
    found = set()
    for seed in sorted(pool):
        res = []

        def grow(chosen, counts):
            if res:
                return
            deficient = [e for e, c in counts.items() if c == 1]
            if not deficient:
                if len(chosen) == nfaces:
                    res.append(frozenset(chosen))
                return
            if len(chosen) >= nfaces:
                return
            e = min(deficient)
            for r in by_edge.get(e, ()):
                if r in chosen:
                    continue
                es = ring_edges(r)
                if any(counts.get(x, 0) >= 2 for x in es):
                    continue
                for x in es:
                    counts[x] = counts.get(x, 0) + 1
                chosen.add(r)
                grow(chosen, counts)
                chosen.discard(r)
                for x in es:
                    counts[x] -= 1
                    if counts[x] == 0:
                        del counts[x]

        c = {}
        for x in ring_edges(seed):
            c[x] = 1
        grow({seed}, c)
        if res:
            found.add(res[0])
    return found


def cell_volume(P, cell):
    total = 0.0
    for r in cell:
        loop = P[list(r)]
        c = loop.mean(axis=0)
        n = len(r)
        for k in range(n):
            total += float(np.dot(c, np.cross(loop[k],
                                              loop[(k + 1) % n]))) / 6.0
    return abs(total)


def report():
    P = chunk(3)
    adj, bond = adjacency(P)
    print("bond %.6f   (3/8)c %.6f" % (bond, 3.0 / 8 * C))
    assert abs(bond - 3.0 / 8 * C) < 1e-9

    inner = [i for i in adj if len(adj[i]) == 4]
    angles = set()
    for i in inner[:40]:
        for a, b in itertools.combinations(adj[i], 2):
            u, v = P[a] - P[i], P[b] - P[i]
            cs = float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
            angles.add(round(np.degrees(np.arccos(np.clip(cs, -1, 1))), 3))
    print("bond angles:", sorted(angles))
    assert angles == {109.471}, angles

    R6 = rings(adj, 6)
    print("6-rings:", len(R6))
    for nf in (3, 5):
        cells = closures(R6, nf)
        if not cells:
            continue
        g = sorted(cells, key=lambda s: sorted(map(sorted, s)))[0]
        vs = sorted({v for r in g for v in r})
        E = len({e for r in g for e in ring_edges(r)})
        deg = collections.Counter()
        for r in g:
            for e in ring_edges(r):
                deg[e[0]] += 1
                deg[e[1]] += 1
        vh = collections.Counter(v // 2 for v in deg.values())
        print("F=%d  V=%d E=%d chi=%d  valence=%s"
              % (nf, len(vs), E, len(vs) - E + nf, dict(sorted(vh.items()))))

    cells = closures(R6, 3)
    vols = collections.Counter(round(cell_volume(P, g), 5) for g in cells)
    hexcell = abs(float(np.dot(A1, np.cross(A2, A3))))
    print("\nhexagonal cell volume %.6f" % hexcell)
    for v, k in sorted(vols.items()):
        print("  3-faced cell volume %.5f  x%-3d  cells per hex cell %.3f"
              % (v, k, hexcell / v))
    vs = sorted(vols)
    if len(vs) >= 3:
        s, m, l = vs[0], vs[1], vs[2]
        print("  volume ratios 1 : %.4f : %.4f" % (m / s, l / s))
        for a in range(6):
            for b in range(6):
                for c in range(6):
                    if a + b + c == 0:
                        continue
                    if abs(a * s + b * m + c * l - hexcell) < 2e-4:
                        print("  %d small + %d mid + %d large fills the cell"
                              % (a, b, c))


if __name__ == "__main__":
    report()
