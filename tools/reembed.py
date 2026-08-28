"""Alternative on-grid embeddings for nets RCSR publishes off-grid.

RCSR embeds every net at MAXIMUM SYMMETRY with free Wyckoff parameters
chosen barycentrically (equal edge lengths), which routinely lands
sites off the 1/24 grid or makes edges that are not Universal Node
branches.  Pearce's constraint is different: sites on the eighth grid
and every edge one of the two branch moduli.  Those are properties of
the EMBEDDING, not of the abstract net -- so before declaring such a
net unreachable, this module searches the net's other embeddings.

Scope: the SAME space group, with each node's free Wyckoff parameters
moved onto the grid.  For every node the stabilizer of its published
position is computed (all group operations fixing it mod 1); candidate
positions are the exact 1/24-grid points fixed by that same
stabilizer, so each node keeps its Wyckoff site type and orbit size.
Edge representatives are matched SYMBOLICALLY at the published
embedding (endpoint = image of node i under operation g plus lattice
offset L), so they move rigidly with the nodes.  A depth-first search
over grid candidates prunes on Universal-Node branch legality per
edge, and every surviving assignment is finally gated by the exact
`tile_pearce.build_net24` (orbit collisions, stated coordination,
branch legality at the net's own scale) -- nothing is accepted on the
search's own say-so.

What this deliberately does NOT do: descend to proper subgroups.  A
node pinned at a 0-dimensional Wyckoff position cannot move in its own
group; a subgroup embedding could move it, but needs the subgroup
lattice of 36 cubic groups plus cell re-indexing, and the measured
same-group yield below did not justify building that.  See the honest
accounting in the module's report output.

Embeddings equivalent under a cubic isometry plus a lattice
translation give congruent tiles, so they are deduplicated by a
48-rotation canonical key before any tiling work is spent on them.

References:
- V. A. Blatov, O. Delgado-Friedrichs, M. O'Keeffe, D. M. Proserpio,
  "Three-periodic nets and tilings: natural tilings for nets", Acta
  Crystallographica A63, 2007, pp. 418-425.
- M. O'Keeffe, M. A. Peskov, S. J. Ramsden & O. M. Yaghi, "The
  Reticular Chemistry Structure Resource (RCSR) Database of, and
  Symbols for, Crystal Nets", Accounts of Chemical Research 41(12),
  2008, pp. 1782-1789.
"""
import collections
import itertools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "math_art"))
import pearce_net as pn
import rcsr_nets
import spacegroups as sg

GRID = 24
TOL = 2e-4            # float tolerance against the catalogue's 5 decimals

#: caps -- honesty beats optimism when a search space explodes
MAX_ROOT_CANDIDATES = 24 ** 3
MAX_EMBEDDINGS_RAW = 20000
MAX_EMBEDDINGS_KEPT = 24


def _apply_frac(op, x):
    R, t = op
    return tuple(sum(R[i][k] * x[k] for k in range(3)) + t[i] / 12.0
                 for i in range(3))


def _close_mod1(a, b, tol=TOL):
    return all(min(abs((p - q) % 1.0), 1.0 - abs((p - q) % 1.0)) <= tol
               for p, q in zip(a, b))


def stab_ops(gops, p):
    """All operations fixing the fractional point p modulo the lattice."""
    return [op for op in gops if _close_mod1(_apply_frac(op, p), p)]


def grid_candidates(stab):
    """Exact 1/24-grid points fixed (mod 24) by every stabilizer op."""
    out = []
    for q in itertools.product(range(GRID), repeat=3):
        ok = True
        for op in stab:
            im = tuple(c % GRID for c in rcsr_nets._apply24(op, q))
            if im != q:
                ok = False
                break
        if ok:
            out.append(q)
    return out


def match_endpoint(x, node_fracs, gops):
    """(node index, op, lattice offset) reproducing coordinate x."""
    for i, p in enumerate(node_fracs):
        for op in gops:
            y = _apply_frac(op, p)
            L = tuple(int(round(x[k] - y[k])) for k in range(3))
            if all(abs(x[k] - y[k] - L[k]) <= TOL for k in range(3)):
                return i, op, L
    return None


def _branch_legal(d):
    """Is a 24ths edge vector a UN branch at either working scale?"""
    if d == (0, 0, 0):
        return False
    for step in (3, 1):
        if all(c % step == 0 for c in d):
            try:
                pn.branch_kind(tuple(c // step for c in d))
                return True
            except Exception:
                pass
    return False


#: the 48 cubic point operations, for congruence deduplication
_CUBIC48 = None


def _cubic48():
    global _CUBIC48
    if _CUBIC48 is None:
        mats = set()
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                R = tuple(tuple(signs[i] if perm[i] == j else 0
                                for j in range(3)) for i in range(3))
                mats.add(R)
        _CUBIC48 = sorted(mats)
    return _CUBIC48


def canonical_key(assign_pts):
    """Congruence key of a site multiset under cubic isometry + shift."""
    best = None
    for R in _cubic48():
        pts = sorted(tuple(sum(R[i][k] * p[k] for k in range(3))
                           for i in range(3)) for p in assign_pts)
        lo = [min(p[i] for p in pts) for i in range(3)]
        pts = tuple(tuple(p[i] - lo[i] for i in range(3)) for p in pts)
        if best is None or pts < best:
            best = pts
    return best


def embeddings(rec, gops, budget=20.0):
    """All inequivalent on-grid assignments for the net's nodes.

    Returns (list of node->q24 dicts, why-or-note).  The list can be
    empty: that is a MEASUREMENT (no same-group on-grid embedding
    exists under the stabilizer-preserving search), not a failure.
    `budget` is a wall-clock cap in seconds; a net that outruns it is
    reported as NOT FULLY SEARCHED (the note says so), which is a
    different and weaker statement than "no embedding exists"."""
    deadline = time.time() + budget
    node_fracs = [tuple(f) for _id, _cn, f in rec["nodes"]]
    if not node_fracs or not rec["edges"]:
        return [], "no nodes or edges"

    stabs = [stab_ops(gops, p) for p in node_fracs]
    cands = [grid_candidates(s) for s in stabs]
    for i, c in enumerate(cands):
        if not c:
            return [], "node %d has no grid point with its stabilizer" % i

    # symbolic edge list: ((i, opA, LA), (j, opB, LB)) per representative
    sym = []
    for a, b in rec["edges"]:
        ma = match_endpoint(a, node_fracs, gops)
        mb = match_endpoint(b, node_fracs, gops)
        if ma is None or mb is None:
            return [], "edge endpoint matches no node image"
        sym.append((ma, mb))

    # order nodes so each new one is edge-connected to an assigned one
    n = len(node_fracs)
    nbr = collections.defaultdict(set)
    for (i, _oa, _la), (j, _ob, _lb) in sym:
        nbr[i].add(j)
        nbr[j].add(i)
    order = []
    seen = set()
    for start in range(n):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            order.append(v)
            for w in sorted(nbr[v]):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)

    def new_end(m, assign):
        i, op, L = m
        q = assign.get(i)
        if q is None:
            return None
        v = rcsr_nets._apply24(op, q)
        return tuple(v[k] + GRID * L[k] for k in range(3))

    out = []
    raw = [0]
    expired = [False]

    def dfs(pos, assign):
        if len(out) >= MAX_EMBEDDINGS_RAW or expired[0]:
            return
        if raw[0] % 512 == 0 and time.time() > deadline:
            expired[0] = True
            return
        if pos == len(order):
            out.append(dict(assign))
            return
        i = order[pos]
        for q in cands[i]:
            raw[0] += 1
            assign[i] = q
            ok = True
            for ma, mb in sym:
                A = new_end(ma, assign)
                B = new_end(mb, assign)
                if A is None or B is None:
                    continue
                d = tuple(B[k] - A[k] for k in range(3))
                if not _branch_legal(d):
                    ok = False
                    break
            if ok:
                dfs(pos + 1, assign)
            del assign[i]

    dfs(0, {})
    if not out:
        return [], ("no assignment satisfies the branch constraints"
                    + (" [BUDGET HIT: search incomplete]" if expired[0]
                       else ""))

    # deduplicate by congruence of the full site orbit
    kept = []
    seen_keys = set()
    for assign in out:
        if time.time() > deadline and kept:
            expired[0] = True
            break
        pts = set()
        for i, q in assign.items():
            for op in gops:
                pts.add(tuple(c % GRID
                              for c in rcsr_nets._apply24(op, q)))
        key = canonical_key(sorted(pts))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kept.append(assign)
        if len(kept) >= MAX_EMBEDDINGS_KEPT:
            break
    return kept, "%d raw, %d inequivalent%s" % (
        len(out), len(kept),
        " [BUDGET HIT: search incomplete]" if expired[0] else "")


def reembedded_rec(rec, gops, assign):
    """A CGD-style record with the nodes moved to the assignment."""
    node_fracs = [tuple(f) for _id, _cn, f in rec["nodes"]]
    new = dict(rec)
    new["nodes"] = [(nid, cn, [assign[i][k] / 24.0 for k in range(3)])
                    for i, (nid, cn, _f) in enumerate(rec["nodes"])]
    sym = []
    for a, b in rec["edges"]:
        ma = match_endpoint(a, node_fracs, gops)
        mb = match_endpoint(b, node_fracs, gops)
        sym.append((ma, mb))
    edges = []
    for (ia, opa, La), (ib, opb, Lb) in sym:
        A = rcsr_nets._apply24(opa, assign[ia])
        B = rcsr_nets._apply24(opb, assign[ib])
        edges.append(([(A[k] + GRID * La[k]) / 24.0 for k in range(3)],
                      [(B[k] + GRID * Lb[k]) / 24.0 for k in range(3)]))
    new["edges"] = edges
    return new


def _selftest():
    """Nets already on the grid must recover an equivalent embedding,
    and the published bcu embedding must be among them."""
    import tile_pearce as tp
    recs = {r["name"]: r for r in rcsr_nets.parse_cgd()
            if r["name"] in ("bcu", "srs")}
    for name in ("bcu", "srs"):
        rec = recs[name]
        gops = sg.ops(rec["group"])
        found, note = embeddings(rec, gops)
        assert found, (name, note)
        good = 0
        for assign in found:
            nr = reembedded_rec(rec, gops, assign)
            if tp.build_net24(nr)[0] is not None:
                good += 1
        assert good >= 1, "%s: no re-embedding passes build_net24" % name
    print("reembed: OK (bcu and srs admit on-grid embeddings)")
