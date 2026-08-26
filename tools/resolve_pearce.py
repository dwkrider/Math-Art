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

# Register every RCSR net that passes its own coordination gate, then
# search the hand-built nets first (they are the classical ones and
# resolve fastest) followed by whatever RCSR adds.
try:
    import rcsr_nets as _rcsr
    _added = _rcsr.register(pn.NETS)
except Exception as _exc:                       # mirror absent
    _added = []
    print("RCSR nets unavailable: %s" % _exc)
print("RCSR nets registered: %d %s" % (len(_added), _added))

NETS = ([('NBO', 3, 3), ('SRS', 3, 5), ('DIAMOND', 2, 3), ('BCC', 2, 3),
         ('FCC', 2, 2), ('SC', 2, 2)]
        + [(n, 2, 2) for n in _added])

def net_classes(name, n):
    """Which branch classes a net actually supplies.

    Table 8.1 states the branch classes of every solid, and a solid's
    edges ARE net edges -- so a net that cannot supply a row's classes
    can never contain that solid, and searching it is wasted work.
    This is the filter that makes the sweep affordable: without it every
    row is tried against every net at every class/modulus combination,
    which measured ~7 minutes per row and never finished the table."""
    Vi, idx, adj = pn.net_chunk(name, n)
    full = max((len(a) for a in adj.values()), default=0)
    out = set()
    for p in Vi:
        if len(adj[p]) != full:
            continue
        for q in adj[p]:
            c = pn.branch_class(tuple(q[i] - p[i] for i in range(3)))
            if c:
                out.add(c)
        if out:
            break
    return out


NET_CLASSES = {}
for _n, _b, _r in NETS:
    try:
        NET_CLASSES[_n] = net_classes(_n, _b)
    except Exception:
        NET_CLASSES[_n] = set()
print("net branch classes: %s"
      % {k: sorted(v) for k, v in sorted(NET_CLASSES.items())})


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
    _t0 = time.time()
    rs = row_sig(r)
    hit = None
    need = {c for c in ('100', '110', '111') if r['branches'].get(c, 0)}
    # nets that can supply this row's classes, closest match first
    usable = [(net, n, rad) for net, n, rad in NETS
              if need <= NET_CLASSES.get(net, set())]
    usable.sort(key=lambda t: len(NET_CLASSES.get(t[0], set()) - need))
    if not usable:
        print("#%-3d %-40s -- (no net supplies %s)"
              % (num, r['name'], sorted(need)), flush=True)
    for net, n, rad in usable:
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
        print("#%-3d %-40s %s via %s (V=%d F=%d)  %.0fs"
              % (num, r['name'], hit[0], hit[1], len(hit[2]), len(hit[3]),
                 time.time() - _t0), flush=True)
    else:
        print("#%-3d %-40s --  %.0fs"
              % (num, r['name'], time.time() - _t0), flush=True)
    with open("resolved.pkl", "wb") as _fh:
        pickle.dump(resolved, _fh)

print("\nresolved %d of 53 in %.0fs" % (len(resolved), time.time() - t_start))
print("FULL:", sorted(n for n, h in resolved.items() if h[0] == 'FULL'))
print("CORE:", sorted(n for n, h in resolved.items() if h[0] == 'CORE'))
with open("resolved.pkl", "wb") as fh:
    pickle.dump(resolved, fh)
