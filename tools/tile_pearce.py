"""Derive Pearce saddle polyhedra as NATURAL-TILING tiles, not by search.

The literature solves the interstitial-domain problem: a saddle
polyhedron of a periodic net is a TILE of the net's natural tiling
(Blatov, Delgado-Friedrichs, O'Keeffe & Proserpio, "Three-periodic nets
and tilings: natural tilings for nets", Acta Cryst. A63 (2007) 418-425;
the idea goes back to Schoen's NASA note TN D-5541 (1970), which built
saddle polyhedra as vertex domains of nets, and to Pearce himself).
RCSR publishes, for every net, the input files of Delgado-Friedrichs'
tiling program 3dt: ONE REPRESENTATIVE FACE per face orbit, as vertex
coordinates.  Those files are the answer key: the full face set of the
natural tiling is the space-group orbit of the seed faces, and the
tiles are the closed chambers the faces bound.  No circuit search, no
exponential growth -- the tiles fall out deterministically.

Pipeline, per candidate net (a net is a candidate when its cell is
cubic, its published tiling signature contains a tile whose face sizes
match a Table 8.1 row, and 3dt seed faces exist for it):

  1. the net itself from `tools/rcsr_nets.build` -- exact site and
     edge orbits under the net's ACTUAL space group
     (`tools/spacegroups.py`), gated on RCSR's stated coordination and
     on every edge being a Universal Node branch.  Nets whose sites
     leave Pearce's eighth grid but stay on the 1/24 grid are built at
     24 units per cell instead: the branch moduli are properties of
     the VECTORS, not of the cell, so a 24th-grid tile is simply a
     Pearce solid in a cell three lattice units across;
  2. the face orbit of the 3dt seeds under the same space-group ops,
     in exact integer 24ths (with an origin-shift search to absorb
     origin-setting differences between the .cgd and the 3dt file);
  3. tiles by dihedral wedge closure: around every edge, sort the
     incident faces angularly; consecutive faces bound a wedge of
     exactly one tile; union-find on (face, side) closes each shell;
  4. THE SIGNATURE GUARD: the set of distinct tile symbols extracted
     must equal the tile symbols of the net's published signature in
     research/data/rcsr/3dall.txt ([6^8], [4^2.8^4], ...).  A net
     whose extraction does not reproduce the published tiling is
     dropped entirely -- this catches any symmetry mishap;
  5. every surviving tile is matched against every row of Table 8.1
     with the same signature logic tools/resolve_pearce.py uses, and
     hits are written to tiles_resolved.pkl for emit_pearce_data.py to
     merge.  Eighth-grid hits carry their RCSR net name (via
     rcsr_nets.ALIASES for nets the hand-built table already has);
     24th-grid hits carry the (class, modulus) kinds of their own
     edges, the same MIXED-net form the resolver's stage 2 emits, so
     the existing emit machinery ships them without new plumbing.
     Every tile still passes the emit gate unchanged -- this tool
     proposes, the gate disposes.

Run from tools/.  It does NOT touch resolved.pkl (a sweep may own it).
"""
import collections
import itertools
import os
import pickle
import re
import sys
import zipfile
from math import atan2, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "math_art"))
import pearce_net as pn
import pearce_table as pt
import rcsr_nets
import spacegroups as sg

ZIP = os.path.join(os.path.dirname(rcsr_nets.CGD), "3dtInputFiles.zip")

GRID = 24           # all arithmetic in exact integer 24ths of the cell
MAX_FACES = 20000   # skip nets whose face chunk outgrows this


# -------------------------------------------------------- RCSR metadata

def parse_3dall_sigs(path=rcsr_nets.DALL):
    """name -> published tiling signature string."""
    recs, cur = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.rstrip("\n")
        if s.strip() == "start":
            cur = []
            recs.append(cur)
        elif cur is not None:
            cur.append(s)
    out = {}
    for r in recs:
        if len(r) < 5:
            continue
        name = r[1].strip()
        sig = next((s.strip() for s in r if re.search(r'\[\d', s)), None)
        out[name] = sig
    return out


def sig_units(sig):
    """The distinct tile symbols of a published signature string."""
    return set(re.findall(r'\[[^\]]*\]', sig or ""))


def _symbol(counts):
    parts = []
    for n in sorted(counts):
        parts.append("%d" % n if counts[n] == 1
                     else "%d^%d" % (n, counts[n]))
    return "[" + ".".join(parts) + "]"


def tile_symbol(F):
    """RCSR-style face symbol [M^m.N^n...] for a tile's face list."""
    return _symbol(collections.Counter(len(f) for f in F))


def row_symbol(r):
    cnt = collections.Counter()
    for fd in r['faces']:
        cnt[fd['n']] += fd['count']
    return _symbol(cnt)


# ----------------------------------------------------------------- seeds

def seed_faces(zf, name):
    """The 3dt seed faces for a net, in integer 24ths, or None.

    Only an exact-stem file counts: 'bcu-x-3dt.cgd' is the net bcu-x,
    not bcu, and RCSR's -a/-b/... extensions are DIFFERENT nets."""
    try:
        text = zf.read("3dtInputFiles/%s-3dt.cgd" % name).decode(
            "utf-8", "replace")
    except KeyError:
        return None
    faces, cur = [], None
    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        if parts[0].upper() == "FACES":
            if cur:
                faces.append(cur)
            cur = []
            continue
        if cur is not None and len(parts) == 3:
            try:
                v = [float(x) for x in parts]
            except ValueError:
                continue
            p = rcsr_nets._to24(v)
            if p is None:
                return None              # not on the 1/24 grid
            cur.append(p)
    if cur:
        faces.append(cur)
    faces = [f for f in faces if len(f) >= 3]
    return faces or None


# ------------------------------------------------------------- net build

def build_net24(rec):
    """(sites, edgeset) in 24ths, plus the grid step, or (None, why).

    sites   : set of positions mod 24;
    edgeset : set of (site, offset) pairs, both directions;
    step    : 3 when the net lives on Pearce's eighth grid (sites all
              on multiples of 3), else 1 (a genuine 24th-grid net,
              whose tiles are Pearce solids in a triple-size cell).
    Same gates as rcsr_nets.build: exact space-group orbits, RCSR's
    stated coordination, every edge a Universal Node branch -- with
    branch legality judged at the scale the net actually has."""
    if not rcsr_nets.is_cubic(rec) or not rec["nodes"] or not rec["edges"]:
        return None, "not cubic, or no nodes/edges"
    try:
        gops = sg.ops(rec["group"])
    except KeyError:
        return None, "group %s not in the Hall table" % rec["group"]

    cn_of = {}
    for _id, cn, frac in rec["nodes"]:
        if len(frac) != 3:
            return None, "node without 3 coordinates"
        v = rcsr_nets._to24(frac)
        if v is None:
            return None, "site not on the 1/24 grid"
        for op in gops:
            q = tuple(c % GRID for c in rcsr_nets._apply24(op, v))
            prev = cn_of.get(q)
            if prev is not None and prev != cn:
                return None, "two orbits collide"
            cn_of[q] = cn
    if not cn_of:
        return None, "no sites"

    edgeset = set()
    for a, b in rec["edges"]:
        A, B = rcsr_nets._to24(a), rcsr_nets._to24(b)
        if A is None or B is None:
            return None, "edge endpoint not on the 1/24 grid"
        for op in gops:
            Ai = rcsr_nets._apply24(op, A)
            Bi = rcsr_nets._apply24(op, B)
            for P, Q in ((Ai, Bi), (Bi, Ai)):
                s = tuple(c % GRID for c in P)
                if s not in cn_of:
                    return None, "EDGE endpoint %s is not a site" % (s,)
                d = tuple(q - p for q, p in zip(Q, P))
                if d == (0, 0, 0):
                    return None, "zero edge"
                edgeset.add((s, d))
    deg = collections.Counter(s for s, _d in edgeset)
    for s, cn in cn_of.items():
        if deg[s] != cn:
            return None, ("node %s has degree %d, RCSR says %d"
                          % (s, deg[s], cn))

    # scale: eighth-grid nets have every site and offset on
    # multiples of 3; branch legality is judged after that division
    on8 = (all(all(c % 3 == 0 for c in s) for s in cn_of)
           and all(all(c % 3 == 0 for c in d) for _s, d in edgeset))
    step = 3 if on8 else 1
    for d in {d for _s, d in edgeset}:
        try:
            pn.branch_kind(tuple(c // step for c in d))
        except Exception:
            return None, ("edge %s is not a Universal Node branch "
                          "at step %d" % (d, step))
    return (set(cn_of), edgeset, step, gops), None


# --------------------------------------------------------- face orbit

def _seed_shift(seeds, sites, edgeset):
    """An origin shift landing every seed on the net, or None.

    The 3dt file and the cgd sometimes use different origin settings
    of the same space group, so the seed coordinates can miss the site
    set by a constant offset.  Candidate shifts are the differences
    between a site and the first seed vertex; a shift is accepted only
    if every shifted seed vertex is a site and every seed edge is a
    net edge.  A wrong-but-consistent shift is caught later by the
    signature guard."""
    def fits(t):
        for f in seeds:
            n = len(f)
            for k in range(n):
                v = f[k]
                s = tuple((v[i] + t[i]) % GRID for i in range(3))
                if s not in sites:
                    return False
                d = tuple(f[(k + 1) % n][i] - v[i] for i in range(3))
                if (s, d) not in edgeset:
                    return False
        return True
    v0 = seeds[0][0]
    for anchor in sorted(sites):
        t = tuple((anchor[i] - v0[i]) % GRID for i in range(3))
        if fits(t):
            return t
    return None


def face_orbit(seeds, gops):
    """Orbit of the seed faces under the group, one representative per
    lattice class, as tuples of 24th-grid vertex tuples."""
    reps = {}
    for face in seeds:
        for op in gops:
            f = [rcsr_nets._apply24(op, v) for v in face]
            lo = [min(v[i] for v in f) for i in range(3)]
            shift = [-(lo[i] // GRID) * GRID for i in range(3)]
            f = [tuple(v[i] + shift[i] for i in range(3)) for v in f]
            key = pn.canonical_ring(f)
            reps.setdefault(key, tuple(f))
    return sorted(reps.values())


# --------------------------------------------------------- cell extraction

def _unit(v):
    n = sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def extract_cells(faces):
    """Closed tiles bounded by the given faces.

    `faces` is a list of vertex-coordinate rings.  Around each edge the
    incident faces are sorted by angle; the wedge between angular
    neighbours belongs to exactly one tile, which glues (face, side)
    pairs together.  Union-find over those pairs yields the tiles."""
    normals = [pn.newell_normal(f) for f in faces]
    cents = [tuple(sum(v[i] for v in f) / len(f) for i in range(3))
             for f in faces]
    by_edge = collections.defaultdict(list)
    for fi, f in enumerate(faces):
        n = len(f)
        for k in range(n):
            a, b = f[k], f[(k + 1) % n]
            e = (a, b) if a < b else (b, a)
            by_edge[e].append(fi)

    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent.setdefault(rx, rx)
            parent[rx] = ry

    for (a, b), inc in by_edge.items():
        if len(inc) < 2:
            continue                    # hull edge of the chunk
        u = _unit(_sub(b, a))
        mid = tuple((a[i] + b[i]) / 2.0 for i in range(3))
        ref = None
        entries = []
        for fi in inc:
            w = _sub(cents[fi], mid)
            w = tuple(w[i] - _dot(w, u) * u[i] for i in range(3))
            if _dot(w, w) < 1e-12:
                # centroid on the edge axis: fall back to the mean of
                # the ring vertices adjacent to the edge
                f = faces[fi]
                ia, ib = f.index(a), f.index(b)
                nb = [f[(ia - 1) % len(f)], f[(ia + 1) % len(f)],
                      f[(ib - 1) % len(f)], f[(ib + 1) % len(f)]]
                nb = [q for q in nb if q not in (a, b)]
                w = _sub(tuple(sum(q[i] for q in nb) / len(nb)
                               for i in range(3)), mid)
                w = tuple(w[i] - _dot(w, u) * u[i] for i in range(3))
            w = _unit(w)
            if ref is None:
                ref = w
                ref2 = _unit(_cross(u, ref))
            theta = atan2(_dot(w, ref2), _dot(w, ref))
            tdir = _cross(u, w)         # direction of increasing theta
            s = 1 if _dot(normals[fi], tdir) > 0 else -1
            entries.append((theta, fi, s))
        entries.sort()
        k = len(entries)
        for i in range(k):
            _th1, f1, s1 = entries[i]
            _th2, f2, s2 = entries[(i + 1) % k]
            # wedge between f1 (increasing-theta side) and f2
            # (decreasing-theta side)
            union((f1, s1), (f2, -s2))

    cells = collections.defaultdict(set)
    for fi in range(len(faces)):
        for s in (1, -1):
            cells[find((fi, s))].add((fi, s))
    return list(cells.values())


# ------------------------------------------------------------- validation

def compact_cell(faces, members):
    """(V, F) with local indices for one union-find class, or None."""
    used = sorted({fi for fi, _s in members})
    if len(used) != len(members):
        return None                     # a face used from both sides
    verts = sorted({v for fi in used for v in faces[fi]})
    index = {v: i for i, v in enumerate(verts)}
    F = tuple(tuple(index[v] for v in faces[fi]) for fi in used)
    return list(verts), F


def valid_cell(V, F):
    ext = max(max(v[i] for v in V) - min(v[i] for v in V) for i in range(3))
    if ext > (5 * GRID) // 2:
        return False                    # the chunk hull, not a tile
    try:
        if not pn.is_closed_surface(F):
            return False
        _v, _e, _f, chi = pn.euler(V, F)
        if chi != 2:
            return False
        return pn.orientation_consistent(pn.orient_faces(V, F))
    except Exception:
        return False


def congruence_key(V, F):
    d2 = sorted(sum((a[i] - b[i]) ** 2 for i in range(3))
                for a, b in itertools.combinations(V, 2))
    return (len(V), tuple(sorted(len(f) for f in F)), tuple(d2))


# ---------------------------------------------------------- row matching
# The same signature logic resolve_pearce.py matches with.  Copied, not
# imported: importing resolve_pearce would start a sweep.

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
    csz = collections.Counter({k[0]: v for k, v in cs['faces'].items()})
    rsz = collections.Counter()
    for k, v in rs['faces'].items():
        rsz[k[0]] += v
    return (cs['V'] == rs['V'] and cs['E'] == rs['E'] and cs['F'] == rs['F']
            and cs['hist'] == rs['hist']
            and cs['branches'] == rs['branches'] and csz == rsz)


# ------------------------------------------------------------------ main

def tiles_of(name, net, seeds, published_units):
    """All distinct tile types (in 24ths), or (None, why)."""
    sites, edgeset, _step, gops = net
    shift = _seed_shift(seeds, sites, edgeset)
    if shift is None:
        return None, "no origin shift lands the seeds on the net"
    if shift != (0, 0, 0):
        seeds = [[tuple(v[i] + shift[i] for i in range(3)) for v in f]
                 for f in seeds]
    reps = face_orbit(seeds, gops)
    blocks = 3
    lim = blocks * GRID
    faces = []
    for f in reps:
        for i, j, k in itertools.product(range(blocks + 1), repeat=3):
            t = (GRID * i, GRID * j, GRID * k)
            g = tuple((v[0] + t[0], v[1] + t[1], v[2] + t[2]) for v in f)
            if all(0 <= v[d] <= lim for v in g for d in range(3)):
                faces.append(g)
    faces = [tuple(f) for f in sorted(set(pn.canonical_ring(f)
                                          for f in faces))]
    if len(faces) > MAX_FACES:
        return None, "chunk too large (%d faces)" % len(faces)
    cells = extract_cells(faces)
    out = {}
    for members in cells:
        got = compact_cell(faces, members)
        if got is None:
            continue
        V, F = got
        if not valid_cell(V, F):
            continue
        out.setdefault(congruence_key(V, F), (V, F))
    tiles = list(out.values())
    got_units = {tile_symbol(F) for _V, F in tiles}
    if got_units != published_units:
        return None, ("signature guard: extracted %s != published %s"
                      % (sorted(got_units), sorted(published_units)))
    return tiles, "%d reps, %d ops, %d faces" % (len(reps), len(gops),
                                                 len(faces))


def reduce_tile(V, F):
    """A tile in its Pearce coordinates, or None.

    Eighth-grid tiles (all coordinates multiples of 3) divide down to
    integer eighths.  Genuine 24th-grid tiles stay as they are: their
    24th-units are read as eighths of a cell three lattice units
    across, which is legitimate because the branch moduli belong to
    the vectors.  Either way every edge must be a Universal Node
    branch at the final scale, or the tile is not a Pearce solid."""
    if all(all(c % 3 == 0 for c in v) for v in V):
        V = [tuple(c // 3 for c in v) for v in V]
    for cyc in F:
        n = len(cyc)
        for k in range(n):
            d = tuple(V[cyc[(k + 1) % n]][i] - V[cyc[k]][i]
                      for i in range(3))
            try:
                pn.branch_kind(d)
            except Exception:
                return None
    return V


def main():
    rows = {r['number']: r for r in pt.TABLE}
    sigs = {n: row_sig(r) for n, r in rows.items()}
    want_units = {}
    for n, r in rows.items():
        want_units.setdefault(row_symbol(r), []).append(n)

    published = parse_3dall_sigs()
    zf = zipfile.ZipFile(ZIP)
    have_seeds = {m.group(1)
                  for nm in zf.namelist()
                  for m in [re.match(r'3dtInputFiles/(.+)-3dt\.cgd$', nm)]
                  if m}
    recs = {r["name"]: r for r in rcsr_nets.parse_cgd()}

    cands = []
    for name, sig in published.items():
        units = sig_units(sig)
        hit_rows = sorted({n for u in units for n in want_units.get(u, ())})
        if not hit_rows:
            continue
        if name not in have_seeds or name not in recs:
            continue
        if not rcsr_nets.is_cubic(recs[name]):
            continue
        cands.append((name, hit_rows, units))
    cands.sort()
    print("candidate nets: %d" % len(cands))

    hits = {}
    per_row = collections.defaultdict(list)
    unmatched = collections.Counter()
    fails = collections.Counter()
    for name, hit_rows, units in cands:
        net, why = build_net24(recs[name])
        if net is None:
            fails[why.split(" (")[0].split(" has ")[0]] += 1
            continue
        seeds = seed_faces(zf, name)
        if seeds is None:
            fails["seeds not on the 1/24 grid"] += 1
            continue
        tiles, note = tiles_of(name, net, seeds, units)
        if tiles is None:
            fails[note.split(":")[0]] += 1
            print("%-8s FAIL %s" % (name, note))
            continue
        step = net[2]
        print("%-8s rows %s (step %d): %s -> %d tile type(s)"
              % (name, hit_rows, step, note, len(tiles)))
        for V24, F in tiles:
            V = reduce_tile(V24, F)
            if V is None:
                print("         tile %s: edges not Universal Node branches"
                      % tile_symbol(F))
                continue
            try:
                F2 = pn.orient_faces(V, F)
                cs = cell_sig(V, F2)
            except Exception as exc:
                print("         tile %s: signature failed (%s)"
                      % (tile_symbol(F), exc))
                continue
            best = None
            for num, rs in sigs.items():
                if full_match(cs, rs):
                    best = ('FULL', num)
                    break
                if core_match(cs, rs) and best is None:
                    best = ('CORE', num)
            if best is None:
                sym = tile_symbol(F2)
                unmatched[sym] += 1
                for num in want_units.get(sym, ()):
                    rs = sigs[num]
                    diffs = [k for k in ('V', 'E', 'F', 'hist', 'branches',
                                         'faces', 'axes') if cs[k] != rs[k]]
                    print("         tile %s V=%d E=%d vs #%d %s: differs "
                          "in %s" % (sym, cs['V'], cs['E'], num,
                                     rows[num]['name'], ",".join(diffs)))
                    if diffs and diffs[0] in ('hist', 'branches'):
                        print("           cell %s=%r row %s=%r"
                              % (diffs[0], cs[diffs[0]],
                                 diffs[0], rs[diffs[0]]))
                continue
            kind, num = best
            # eighth-grid tiles ride their RCSR net (through the alias
            # table where the hand-built NETS already has it); genuine
            # 24th-grid tiles ride the MIXED-net kinds of their own
            # edges, the form the resolver's stage 2 already emits.
            if all(all(c % 3 == 0 for c in v) for v in V24):
                key = name.upper()
                netid = rcsr_nets.ALIASES.get(key, key)
            else:
                kinds = set()
                for cyc in F2:
                    n = len(cyc)
                    for k in range(n):
                        d = tuple(V[cyc[(k + 1) % n]][i] - V[cyc[k]][i]
                                  for i in range(3))
                        kinds.add(pn.branch_kind(d))
                netid = tuple(sorted(kinds))
            print("         tile %s V=%d: #%d %s (%s) via %s"
                  % (tile_symbol(F2), cs['V'], num, rows[num]['name'],
                     kind, netid))
            per_row[num].append((kind, name))
            old = hits.get(num)
            if old is None or (old[0] == 'CORE' and kind == 'FULL'):
                hits[num] = (kind, netid, V, F2)

    with open(os.path.join(HERE, "tiles_resolved.pkl"), "wb") as fh:
        pickle.dump(hits, fh)
    print("\nrows hit by tiles: %s" % sorted(hits))
    print("FULL: %s" % sorted(n for n, h in hits.items() if h[0] == 'FULL'))
    print("CORE: %s" % sorted(n for n, h in hits.items() if h[0] == 'CORE'))
    print("per-row sources: %s" % {n: v for n, v in sorted(per_row.items())})
    print("tile symbols matching no row: %s" % dict(unmatched))
    print("net failures: %s" % dict(fails))
    print("wrote tiles_resolved.pkl")


if __name__ == "__main__":
    main()
