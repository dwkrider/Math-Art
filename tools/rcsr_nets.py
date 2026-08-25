"""Build Pearce-compatible nets from the local RCSR mirror.

`research/data/rcsr/downloads/RCSRnets-2019-06-01.cgd` is the Systre
form of the whole catalogue: per net a NAME, a space GROUP, a CELL and
the NODE positions with their coordination number.  That last number is
what makes this safe to automate -- a net is only accepted here if the
orbit we generate gives every node exactly the coordination RCSR states.
Nothing is taken on trust.

SCOPE.  Only CUBIC nets are converted, because Pearce's Universal Node
is cubic: its branches are the 26 directions of a cubic lattice, and a
solid of Table 8.1 cannot live anywhere else.  A hexagonal net such as
wurtzite (`lon`) is a genuinely different lattice -- see
`tools/wurtzite_net.py` for what that costs.

Each accepted net is emitted in integer EIGHTHS of the conventional
cubic cell, the coordinate system `math_art/pearce_net.py` uses, and is
rejected unless every one of its edges is a Universal Node branch at
one of Pearce's two moduli.  So what comes out is not merely "an RCSR
net", it is an RCSR net that Pearce's system can actually express.

References:
- M. O'Keeffe, M. A. Peskov, S. J. Ramsden & O. M. Yaghi, "The
  Reticular Chemistry Structure Resource (RCSR) Database of, and
  Symbols for, Crystal Nets", Accounts of Chemical Research 41(12),
  2008, pp. 1782-1789.
"""

import collections
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
#: `research/` is gitignored, so in a feature worktree it exists only
#: in the main checkout -- look there too rather than failing.
_REL = os.path.join("research", "data", "rcsr", "downloads",
                    "RCSRnets-2019-06-01.cgd")
_CANDIDATES = [
    os.path.join(ROOT, _REL),
    os.path.join(ROOT, "..", "..", "..", _REL),
    os.path.join("C:/Users/dkrid/Projects/2026_07_21_Math_Art", _REL),
]
CGD = next((os.path.normpath(p) for p in _CANDIDATES
            if os.path.exists(os.path.normpath(p))),
           os.path.normpath(_CANDIDATES[0]))

sys.path.insert(0, os.path.join(ROOT, "math_art"))


def parse_cgd(path=CGD):
    """Every CRYSTAL record: name, group, cell, nodes, edges."""
    recs = []
    cur = None
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.upper()
        if low == "CRYSTAL":
            cur = dict(name=None, group=None, cell=None, nodes=[], edges=[])
        elif low == "END":
            if cur and cur["name"]:
                recs.append(cur)
            cur = None
        elif cur is not None:
            parts = line.split()
            key = parts[0].upper()
            if key == "NAME":
                cur["name"] = parts[1]
            elif key == "GROUP":
                cur["group"] = parts[1]
            elif key == "CELL":
                cur["cell"] = [float(x) for x in parts[1:]]
            elif key == "NODE":
                # node ids are not always integers ("NODE q 4 ..."), so
                # take the id as text and the coordination as the first
                # integer after it
                nums = parts[2:]
                try:
                    cn = int(nums[0])
                    coords = [float(x) for x in nums[1:]]
                except (ValueError, IndexError):
                    continue
                cur["nodes"].append((parts[1], cn, coords))
            elif key == "EDGE":
                v = [float(x) for x in parts[1:]]
                if len(v) == 6:
                    cur["edges"].append((v[:3], v[3:]))
    return recs


def _cubic_ops():
    """The 48 operations of m-3m, as signed coordinate permutations."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for sgn in itertools.product((1, -1), repeat=3):
            ops.append((perm, sgn))
    return ops


def _centrings(group):
    """Lattice centring translations implied by the group symbol."""
    g = group.strip().upper()
    if g.startswith("I"):
        return [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
    if g.startswith("F"):
        return [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5),
                (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    return [(0.0, 0.0, 0.0)]


def orbit(frac, group, tol=1e-6):
    """Fractional orbit of a site under the cubic point group + centring."""
    out = set()
    for perm, sgn in _cubic_ops():
        p = tuple(sgn[i] * frac[perm[i]] for i in range(3))
        for t in _centrings(group):
            q = tuple((p[i] + t[i]) % 1.0 for i in range(3))
            q = tuple(0.0 if abs(x) < tol or abs(x - 1.0) < tol else x
                      for x in q)
            out.add(tuple(round(x, 6) for x in q))
    return sorted(out)


def is_cubic(rec):
    c = rec["cell"]
    return (c is not None and len(c) == 6
            and abs(c[0] - c[1]) < 1e-6 and abs(c[1] - c[2]) < 1e-6
            and all(abs(a - 90.0) < 1e-6 for a in c[3:]))


def build(rec, blocks=3, eighths=8):
    """The net in integer eighths, or None if it cannot be expressed.

    Returns (sites, adjacency, info).  Rejects the net unless every
    node has the coordination RCSR states AND every edge is a Universal
    Node branch."""
    import pearce_net as pn

    if not is_cubic(rec) or not rec["nodes"]:
        return None
    sites = []
    want_cn = {}
    for _idx, cn, frac in rec["nodes"]:
        if len(frac) != 3:
            return None
        for q in orbit(frac, rec["group"]):
            sites.append(q)
            want_cn[q] = cn
    if not sites:
        return None

    # to integer eighths of the conventional cell
    def to8(q):
        v = []
        for x in q:
            y = x * eighths
            n = int(round(y))
            if abs(y - n) > 1e-6:
                return None
            v.append(n)
        return tuple(v)

    base = []
    cn_of = {}
    for q in sites:
        p = to8(q)
        if p is None:
            return None                  # site not on the eighth grid
        base.append(p)
        cn_of[p] = want_cn[q]
    base = sorted(set(base))

    # expand over a block and connect by nearest distance
    P = []
    for i, j, k in itertools.product(range(blocks + 1), repeat=3):
        for b in base:
            P.append((b[0] + eighths * i, b[1] + eighths * j,
                      b[2] + eighths * k))
    P = sorted(set(P))
    A = np.asarray(P, float)
    D = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    np.fill_diagonal(D, 1e18)
    d0 = float(D.min())
    adj = {}
    for i, p in enumerate(P):
        adj[p] = [P[j] for j in np.where(D[i] < d0 * 1.001)[0]]

    # gate 1: interior coordination matches RCSR's cn.  The window has
    # to be a real interval -- an earlier revision computed lo == hi and
    # so found no interior sites at all, silently rejecting every net
    # whose lattice is offset from the cell corner (nbo among them).
    lo, hi = eighths, eighths * (blocks - 1)
    interior = [p for p in P if all(lo <= c <= hi for c in p)]
    if not interior:
        return None
    for p in interior:
        want = cn_of.get(tuple(c % eighths for c in p))
        if want is None:
            want = cn_of.get(tuple(int(c) % eighths for c in p))
        if want is not None and len(adj[p]) != want:
            return None

    # gate 2: every edge must be a Universal Node branch
    for p in interior:
        for q in adj[p]:
            v = tuple(q[i] - p[i] for i in range(3))
            try:
                pn.branch_kind(v)
            except Exception:
                return None

    cls = collections.Counter()
    for p in interior:
        for q in adj[p]:
            cls[pn.branch_class(tuple(q[i] - p[i] for i in range(3)))] += 1
    return base, tuple(sorted({tuple(v[i] - base[0][i] for i in range(3))
                               for v in ()})), dict(
        name=rec["name"], group=rec["group"], cn=sorted(set(cn_of.values())),
        classes=dict(cls), sites=len(base), interior=len(interior))


def survey(limit=None, verbose=True):
    recs = [r for r in parse_cgd() if is_cubic(r)]
    good = []
    for r in recs[:limit]:
        try:
            res = build(r)
        except Exception:
            res = None
        if res is None:
            continue
        base, _n, info = res
        good.append((r["name"], base, info))
        if verbose:
            print("%-8s %-10s sites=%-3d cn=%-8s classes=%s"
                  % (info["name"], info["group"], info["sites"],
                     info["cn"], info["classes"]))
    return good


if __name__ == "__main__":
    recs = parse_cgd()
    print("CGD records: %d   (3-periodic cubic: %d)"
          % (len(recs), sum(1 for r in recs if is_cubic(r))))
    print()
    good = survey()
    print()
    print("nets expressible in Pearce's Universal Node: %d" % len(good))
