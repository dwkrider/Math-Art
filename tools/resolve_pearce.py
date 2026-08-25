"""Search the Universal Node net for the solids of Pearce's Table 8.1.

Pearce prints no coordinates, so every saddle polyhedron has to be
FOUND: for each row, pool the net's circuits at that row's face sizes,
grow closed complexes with exactly that face count, and keep one only
if its node valences, branch classes, face inventory and symmetry axes
all agree.  Writes resolved.pkl for tools/emit_pearce_data.py.

Run from this directory; it is a data-derivation tool, not part of the
shipped extension.
"""
import sys, pickle, collections, time, json
MA = r"C:\Users\dkrid\Projects\2026_07_21_Math_Art\.claude\worktrees\spidrons\math_art"
sys.path.insert(0, MA)
import pearce_net as pn
import pearce_table as pt


def ring_edges(cyc):
    n = len(cyc)
    return [pn.edge_key(cyc[i], cyc[(i + 1) % n]) for i in range(n)]


def local_rings(idx, adj, centre, radius, L, cap=200000):
    ball = {centre}
    frontier = {centre}
    for _ in range(radius):
        nxt = set()
        for p in frontier:
            nxt.update(adj[p])
        ball |= nxt
        frontier = nxt
    sub = {p: [q for q in adj[p] if q in ball] for p in ball}
    order = {p: i for i, p in enumerate(sorted(ball))}
    rings = set()
    spent = [0]
    for A in sorted(ball):
        ia = order[A]

        def dfs(path, seen):
            if spent[0] > cap:
                return
            spent[0] += 1
            last = path[-1]
            if len(path) == L:
                if A in sub[last]:
                    rings.add(pn.canonical_ring([idx[p] for p in path]))
                return
            for q in sub[last]:
                if q not in seen and order[q] > ia:
                    dfs(path + [q], seen | {q})

        dfs([A], {A})
    return rings


def grow_exact(pool, nfaces, seed, want_sizes, budget=60000):
    """Closed surfaces of exactly nfaces whose face-size multiset is
    `want_sizes`."""
    by_edge = {}
    for r in pool:
        for e in ring_edges(r):
            by_edge.setdefault(e, []).append(r)
    out = []
    spent = [0]

    def grow(chosen, counts):
        if out or spent[0] > budget:
            return
        spent[0] += 1
        deficient = [e for e, c in counts.items() if c == 1]
        if not deficient:
            if len(chosen) == nfaces:
                sizes = collections.Counter(len(r) for r in chosen)
                if sizes == want_sizes:
                    out.append(tuple(sorted(chosen)))
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
            cur = collections.Counter(len(x) for x in chosen)
            cur[len(r)] += 1
            if any(cur[k] > want_sizes.get(k, 0) for k in cur):
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
    return out


def cell_sig(V, F):
    v, e, f, chi = pn.euler(V, F)
    hist, _ = pn.valence_histogram(F)
    bt = pn.branch_totals(V, F)
    faces = collections.Counter()
    for cyc in F:
        loop = [V[i] for i in cyc]
        faces[(len(cyc), pn.face_symmetry_label(loop),
               pn.face_plane_class(loop))] += 1
    pts = [V[i] for i in range(len(V))]
    return dict(V=v, E=e, F=f, chi=chi, hist=hist, branches=bt,
                faces=faces, axes=pn.axis_counts(pts))


def row_sig(r):
    hist = {}
    for z, c in tuple(r['primary']) + tuple(r['secondary']):
        hist[z] = hist.get(z, 0) + c
    faces = collections.Counter()
    for fd in r['faces']:
        faces[(fd['n'], fd['symmetry'], fd['plane'])] += fd['count']
    return dict(V=r['nodes_total'], E=r['branches_total'],
                F=r['faces_total'], chi=2, hist=hist,
                branches=dict(r['branches']), faces=faces,
                axes=tuple(r['axes']))


def full_match(cs, rs):
    return (cs['V'] == rs['V'] and cs['E'] == rs['E'] and cs['F'] == rs['F']
            and cs['hist'] == rs['hist'] and cs['branches'] == rs['branches']
            and cs['faces'] == rs['faces'] and cs['axes'] == rs['axes'])


def core_match(cs, rs):
    """Topology + branches + face sizes, ignoring the derived symmetry
    labels -- a weaker but still highly specific agreement."""
    csz = collections.Counter({k[0]: v for k, v in cs['faces'].items()})
    rsz = collections.Counter()
    for k, v in rs['faces'].items():
        rsz[k[0]] += v
    return (cs['V'] == rs['V'] and cs['E'] == rs['E'] and cs['F'] == rs['F']
            and cs['hist'] == rs['hist']
            and cs['branches'] == rs['branches'] and csz == rsz)


MAX_NGON = 12
MAX_FACES = 26
MIXED_RADIUS = 2
MIXED_RING_CAP = 400000
MIXED_BUDGET = 200000

NETS = [('NBO', 3, 3), ('SRS', 3, 5), ('DIAMOND', 2, 3), ('BCC', 2, 3),
        ('FCC', 2, 2), ('SC', 2, 2)]

chunks = {}
for net, n, rad in NETS:
    Vi, idx, adj = pn.net_chunk(net, n)
    full = max(len(a) for a in adj.values())
    mid = (4 * n, 4 * n, 4 * n)
    centre, best = None, None
    for p in Vi:
        if len(adj[p]) != full:
            continue
        d = sum((a - b) ** 2 for a, b in zip(p, mid))
        if best is None or d < best:
            best, centre = d, p
    chunks[net] = (Vi, idx, adj, centre, rad)

ringcache = {}


def rings_for(net, L):
    k = (net, L)
    if k not in ringcache:
        Vi, idx, adj, centre, rad = chunks[net]
        ringcache[k] = local_rings(idx, adj, centre, rad, L)
    return ringcache[k]


resolved = {}
t_start = time.time()
for r in pt.TABLE:
    num = r['number']
    want = collections.Counter()
    for fd in r['faces']:
        want[fd['n']] += fd['count']
    if max(want) > MAX_NGON or r['faces_total'] > MAX_FACES:
        print("#%-3d %-40s SKIP (too large: F=%d, max n=%d)"
              % (num, r['name'], r['faces_total'], max(want)), flush=True)
        continue
    rs = row_sig(r)
    hit = None
    for net, n, rad in NETS:
        try:
            pool = set()
            for L in want:
                pool |= rings_for(net, L)
        except Exception:
            continue
        if not pool:
            continue
        Vi = chunks[net][0]
        for s in sorted(pool):
            if len(s) not in want:
                continue
            got = grow_exact(pool, r['faces_total'], s, want)
            for g in got:
                V, F = pn.compact(Vi, g)
                try:
                    F = pn.orient_faces(V, F)
                    cs = cell_sig(V, F)
                except Exception:
                    continue
                if cs['chi'] != 2:
                    continue
                if full_match(cs, rs):
                    hit = ('FULL', net, V, F)
                    break
                if core_match(cs, rs) and hit is None:
                    hit = ('CORE', net, V, F)
            if hit and hit[0] == 'FULL':
                break
        if hit and hit[0] == 'FULL':
            break
    if hit is None or hit[0] != 'FULL':
        # Stage 2: the classical nets each carry ONE branch class, so a
        # row mixing classes cannot be found in any of them.  Build the
        # Universal Net restricted to exactly the classes this row uses
        # -- which is what keeps it searchable.
        for kinds in pn.kinds_for_row(dict(r['branches'])):
            try:
                Vi, idx, adj = pn.mixed_net(kinds, n=2)
            except Exception:
                continue
            if not Vi:
                continue
            full = max((len(a) for a in adj.values()), default=0)
            mid = (8, 8, 8)
            centre, bestd = None, None
            for p in Vi:
                if len(adj[p]) != full:
                    continue
                d = sum((a - b) ** 2 for a, b in zip(p, mid))
                if bestd is None or d < bestd:
                    bestd, centre = d, p
            if centre is None:
                continue
            pool = set()
            for L in want:
                pool |= local_rings(idx, adj, centre, MIXED_RADIUS, L,
                                    cap=MIXED_RING_CAP)
            if not pool:
                continue
            for s_ in sorted(pool):
                if len(s_) not in want:
                    continue
                for g in grow_exact(pool, r['faces_total'], s_, want,
                                    budget=MIXED_BUDGET):
                    V, F = pn.compact(Vi, g)
                    try:
                        F = pn.orient_faces(V, F)
                        cs = cell_sig(V, F)
                    except Exception:
                        continue
                    if cs['chi'] != 2:
                        continue
                    if full_match(cs, rs):
                        hit = ('FULL', 'MIXED%s' % (kinds,), V, F)
                        break
                    if core_match(cs, rs) and hit is None:
                        hit = ('CORE', 'MIXED%s' % (kinds,), V, F)
                if hit and hit[0] == 'FULL':
                    break
            if hit and hit[0] == 'FULL':
                break

    if hit:
        resolved[num] = hit
        print("#%-3d %-40s %s via %s (V=%d F=%d)"
              % (num, r['name'], hit[0], hit[1], len(hit[2]), len(hit[3])),
              flush=True)
    else:
        print("#%-3d %-40s --" % (num, r['name']), flush=True)
    with open("resolved.pkl", "wb") as _fh:
        pickle.dump(resolved, _fh)

print("\nresolved %d of 53 in %.0fs" % (len(resolved), time.time() - t_start))
print("FULL:", sorted(n for n, h in resolved.items() if h[0] == 'FULL'))
print("CORE:", sorted(n for n, h in resolved.items() if h[0] == 'CORE'))
with open("resolved.pkl", "wb") as fh:
    pickle.dump(resolved, fh)
