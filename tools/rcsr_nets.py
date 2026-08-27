"""Build Pearce-compatible nets from the local RCSR mirror.

`research/data/rcsr/downloads/RCSRnets-2019-06-01.cgd` is the Systre
form of the whole catalogue: per net a NAME, a space GROUP, a CELL and
the NODE positions with their coordination number, plus one
representative EDGE per edge orbit.  A net is only accepted here if the
orbit we generate gives every node exactly the coordination RCSR
states.  Nothing is taken on trust.

Site orbits and edge orbits are generated with the PROPER operations of
each net's space group (`tools/spacegroups.py`), not a point-group
approximation.  An earlier revision applied the full m-3m point group
plus a centring guess, which is correct only for centrosymmetric
holohedral groups: chiral groups (srs in I4132) and glide/screw groups
(dia in Fd-3m:2) over-generated and were rejected by the coordination
gate, which silently cost every net outside the 15 holohedral
survivors.  The operations are validated offline against the Wyckoff
multiplicities RCSR states in `3dall.txt` (`wyckoff_check()` below).

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
All arithmetic is exact: sites live on the 1/24 grid (eighths for the
cell, twelfths for the group translations), so acceptance never hinges
on floating-point tolerance.

References:
- M. O'Keeffe, M. A. Peskov, S. J. Ramsden & O. M. Yaghi, "The
  Reticular Chemistry Structure Resource (RCSR) Database of, and
  Symbols for, Crystal Nets", Accounts of Chemical Research 41(12),
  2008, pp. 1782-1789.
- S. R. Hall, "Space-group notation with an explicit origin", Acta
  Crystallographica A37, 1981, pp. 517-525.
"""

import collections
import itertools
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
#: `research/` is gitignored, so in a feature worktree it exists only
#: in the main checkout -- look there too rather than failing.


def _find(rel):
    cands = [os.path.join(ROOT, rel),
             os.path.join(ROOT, "..", "..", "..", rel),
             os.path.join("C:/Users/dkrid/Projects/2026_07_21_Math_Art", rel)]
    return next((os.path.normpath(p) for p in cands
                 if os.path.exists(os.path.normpath(p))),
                os.path.normpath(cands[0]))


CGD = _find(os.path.join("research", "data", "rcsr", "downloads",
                         "RCSRnets-2019-06-01.cgd"))
DALL = _find(os.path.join("research", "data", "rcsr", "3dall.txt"))

sys.path.insert(0, os.path.join(ROOT, "math_art"))
sys.path.insert(0, HERE)

import spacegroups as sg


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


def orbit(frac, group):
    """Fractional orbit of a site under its actual space group."""
    return sg.orbit(frac, group)


def is_cubic(rec):
    c = rec["cell"]
    return (c is not None and len(c) == 6
            and abs(c[0] - c[1]) < 1e-6 and abs(c[1] - c[2]) < 1e-6
            and all(abs(a - 90.0) < 1e-6 for a in c[3:]))


def _to24(frac):
    """Signed integer 24ths of the cell, or None if off that grid.

    24 is the least grid holding both Pearce's eighths and the group
    translations' twelfths, so on it every operation is exact integer
    arithmetic.  The tolerance (5e-4 of a 24th, i.e. 2e-5 of the cell)
    accepts the catalogue's 5-decimal quantisation of exact 24ths and
    nothing else."""
    out = []
    for x in frac:
        y = x * 24.0
        n = int(round(y))
        if abs(y - n) > 5e-4 * 24.0:
            return None
        out.append(n)
    return tuple(out)


def _apply24(op, v24):
    """Exact op application in 24ths (translations are twelfths)."""
    R, t = op
    return tuple(sum(R[i][k] * v24[k] for k in range(3)) + 2 * t[i]
                 for i in range(3))


def build(rec, blocks=3, eighths=8):
    """The net in integer eighths, or None if it cannot be expressed.

    Returns (sites, offsets, info).  Gates, in order:

    A. every node coordinate and edge endpoint lies on the 1/24 grid,
       and every site of every orbit on the 1/8 grid (Pearce's);
    B. expanding the EDGE representatives under the group gives every
       site EXACTLY the coordination RCSR states -- wrong operations
       cannot pass this, because an over- or under-generated orbit
       leaves endpoints unmatched or degrees wrong;
    C. the exported form reproduces that same adjacency.  The compact
       export is base sites plus one GLOBAL offset list (what the
       classical `pearce_net.NETS` entries use); where a global list
       would create edges between sites that are not bonded, the net
       is exported with PER-SITE offsets instead (a dict, which
       `pearce_net.net_chunk` also consumes) rather than shipped wrong;
    D. every edge is a Universal Node branch at one of Pearce's two
       moduli (`pearce_net.branch_kind`)."""
    import pearce_net as pn

    if not is_cubic(rec) or not rec["nodes"] or not rec["edges"]:
        return None
    try:
        gops = sg.ops(rec["group"])
    except KeyError:
        return None

    # ---- gate A: site orbits, exact, on the eighth grid -------------
    cn_of = {}                              # site (24ths, mod 24) -> cn
    for _id, cn, frac in rec["nodes"]:
        if len(frac) != 3:
            return None
        v = _to24(frac)
        if v is None:
            return None
        for op in gops:
            q = tuple(c % 24 for c in _apply24(op, v))
            if any(c % 3 for c in q):
                return None                 # orbit leaves the 1/8 grid
            prev = cn_of.get(q)
            if prev is not None and prev != cn:
                return None                 # two orbits collide
            cn_of[q] = cn

    # ---- gate B: exact adjacency from the EDGE representatives ------
    per = {s: set() for s in cn_of}         # site -> offset vectors
    for a, b in rec["edges"]:
        A = _to24(a)
        B = _to24(b)
        if A is None or B is None:
            return None
        for op in gops:
            Ai = _apply24(op, A)
            Bi = _apply24(op, B)
            for P, Q in ((Ai, Bi), (Bi, Ai)):
                s = tuple(c % 24 for c in P)
                if s not in per:
                    return None             # endpoint is not a site
                d = tuple(q - p for q, p in zip(Q, P))
                if d == (0, 0, 0) or any(c % 3 for c in d):
                    return None
                per[s].add(tuple(c // 3 for c in d))
    for s, cn in cn_of.items():
        if len(per[s]) != cn:
            return None

    base = sorted(tuple(c // 3 for c in s) for s in cn_of)
    per8 = {tuple(c // 3 for c in s): offs for s, offs in per.items()}
    cn8 = {tuple(c // 3 for c in s): cn for s, cn in cn_of.items()}
    offs = tuple(sorted(set().union(*per8.values())))

    # ---- gate D first (cheap): every offset is a branch -------------
    for d in offs:
        try:
            pn.branch_kind(d)
        except Exception:
            return None

    # ---- gate C: the exported form reproduces the net ---------------
    # With a GLOBAL offset list, net_chunk applies every offset at
    # every site and keeps the hits -- an offset belonging to one site
    # class can accidentally connect another.  Nets where the compact
    # global list is faithful export it; the rest export per-site
    # offsets, and either way the exported adjacency is validated over
    # a block, not assumed.
    P = set()
    for i, j, k in itertools.product(range(blocks + 1), repeat=3):
        for b in base:
            P.add((b[0] + eighths * i, b[1] + eighths * j,
                   b[2] + eighths * k))
    lo, hi = eighths, eighths * (blocks - 1)
    interior = [p for p in sorted(P) if all(lo <= c <= hi for c in p)]
    if not interior:
        return None
    cls = collections.Counter()
    global_ok = True
    for p in interior:
        key = tuple(c % eighths for c in p)
        want = {tuple(p[i] + d[i] for i in range(3)) for d in per8[key]}
        if len(want) != cn8[key] or not want <= P:
            return None                     # exported net loses edges
        if global_ok:
            got = {q for q in (tuple(p[i] + d[i] for i in range(3))
                               for d in offs) if q in P}
            global_ok = got == want
        for d in per8[key]:
            cls[pn.branch_class(d)] += 1

    export = offs if global_ok else {b: tuple(sorted(per8[b]))
                                     for b in base}
    return base, export, dict(
        name=rec["name"], group=rec["group"], cn=sorted(set(cn8.values())),
        classes=dict(cls), sites=len(base), interior=len(interior),
        export='global' if global_ok else 'per-site')


#: RCSR names for nets the hand-built table already carries under
#: another key -- registering them again would make every downstream
#: sweep search the same net twice.
ALIASES = {'DIA': 'DIAMOND', 'PCU': 'SC', 'BCU': 'BCC', 'FCU': 'FCC'}


def register(into, limit=None):
    """Add every accepted RCSR net to a pearce_net-style NETS dict.

    Returns the names added.  Existing entries are never overwritten --
    the hand-built srs and diamond nets are verified and stay -- and
    RCSR spellings of nets already present (ALIASES) are skipped."""
    added = []
    for name, base, nbrs, _info in survey_full(limit=limit, verbose=False):
        key = name.upper()
        if key in into or ALIASES.get(key) in into:
            continue
        into[key] = (tuple(base),
                     nbrs if isinstance(nbrs, dict) else tuple(nbrs))
        added.append(key)
    return added


def survey_full(limit=None, verbose=True):
    recs = [r for r in parse_cgd() if is_cubic(r)]
    out = []
    for r in recs[:limit]:
        try:
            res = build(r)
        except Exception:
            res = None
        if res is None:
            continue
        base, nbrs, info = res
        out.append((r["name"], base, nbrs, info))
        if verbose:
            print("%-8s %-10s sites=%-3d cn=%-8s export=%-8s classes=%s"
                  % (info["name"], info["group"], info["sites"],
                     info["cn"], info["export"], info["classes"]))
    return out


def survey(limit=None, verbose=True):
    good = []
    for name, base, _nbrs, info in survey_full(limit=limit, verbose=False):
        good.append((name, base, info))
        if verbose:
            print("%-8s %-10s sites=%-3d cn=%-8s classes=%s"
                  % (info["name"], info["group"], info["sites"],
                     info["cn"], info["classes"]))
    return good


# --------------------------------------------------------------------
# offline validation of the symmetry operations themselves
# --------------------------------------------------------------------

def parse_3dall(path=DALL):
    """Per net: (space-group number, [(cn, Wyckoff multiplicity), ...]).

    `3dall.txt` states each vertex's Wyckoff multiplicity -- data
    independent of the CGD coordinates, so orbit sizes generated from
    the Hall table can be checked against it."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    recs = {}
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() != "start":
            i += 1
            continue
        name = lines[i + 2].strip()
        j, sgline = i + 3, None
        while j < n and j < i + 200:
            m = re.match(r"^\s*(\S+)\s+(\d{1,3})\s*$", lines[j])
            if (m and 1 <= int(m.group(2)) <= 230
                    and any(c.isalpha() for c in m.group(1))):
                parts = lines[j + 1].split()
                if len(parts) == 6:
                    try:
                        [float(x) for x in parts]
                        sgline = j
                        break
                    except ValueError:
                        pass
            j += 1
        if sgline is None:
            i += 1
            continue
        sgnum = int(re.match(r"^\s*\S+\s+(\d{1,3})\s*$",
                             lines[sgline]).group(1))
        k = sgline + 2
        try:
            nv = int(lines[k].strip())
        except ValueError:
            i += 1
            continue
        verts, ok = [], True
        k += 1
        for _ in range(nv):
            try:
                cn = int(lines[k].split()[1])
                mult = int(lines[k + 3].split()[0])
            except Exception:
                ok = False
                break
            verts.append((cn, mult))
            k += 6
        if ok and verts:
            recs[name] = (sgnum, verts)
        i = k
    return recs


def wyckoff_check(verbose=True):
    """Compare every generated orbit size against RCSR's multiplicity.

    Runs over the WHOLE catalogue (all crystal systems), so it
    exercises every Hall-table entry the data can reach.  Returns
    (checked, mismatches); mismatches include the handful of nets whose
    CGD embedding genuinely sits on higher symmetry than the abstract
    net (their vertices coincide geometrically -- the coordination gate
    rejects those anyway)."""
    dall = parse_3dall()
    ok, bad = 0, []
    for r in parse_cgd():
        g = r["group"]
        if not g or not g[0].isupper() or r["name"] not in dall:
            continue
        _num, verts = dall[r["name"]]
        if len(verts) != len(r["nodes"]):
            continue                        # differently factored listing
        try:
            gops = sg.ops(g)
        except KeyError:
            continue
        got = sorted(len(sg.orbit(f, group_ops=gops))
                     for _, _, f in r["nodes"])
        want = sorted(m for _, m in verts)
        if got == want:
            ok += 1
        else:
            bad.append((r["name"], g, got, want))
    if verbose:
        print("wyckoff_check: %d nets agree, %d mismatch" % (ok, len(bad)))
        for name, g, got, want in bad:
            print("  %-10s %-10s orbit sizes %s, RCSR says %s"
                  % (name, g, got, want))
    return ok, bad


# --------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------

def _match_up_to_translation(got, want, eighths=8):
    """Are two site sets the same net modulo a lattice translation?"""
    got = sorted(got)
    want = sorted(want)
    if len(got) != len(want):
        return False
    for target in want:
        shift = tuple((target[i] - got[0][i]) % eighths for i in range(3))
        moved = sorted(tuple((p[i] + shift[i]) % eighths for i in range(3))
                       for p in got)
        if moved == want:
            return True
    return False


def _selftest():
    import pearce_net as pn

    sg._selftest()
    recs = {r["name"]: r for r in parse_cgd() if r["name"] in ("srs", "dia")}
    assert set(recs) == {"srs", "dia"}, "srs/dia missing from CGD"

    base, offs, info = build(recs["srs"])
    assert info["cn"] == [3], info
    assert len(base) == 8, base
    assert _match_up_to_translation(base, pn.SRS_BASE8), (base, pn.SRS_BASE8)
    assert set(offs) == set(pn.SRS_NBR8), (offs, pn.SRS_NBR8)

    base, offs, info = build(recs["dia"])
    assert info["cn"] == [4], info
    assert len(base) == 8, base
    assert _match_up_to_translation(base, pn.DIAMOND_BASE8), (
        base, pn.DIAMOND_BASE8)
    assert set(offs) == set(pn.DIAMOND_NBR8), (offs, pn.DIAMOND_NBR8)
    print("rcsr_nets: OK (srs and dia reconstruct the hand-built nets)")


if __name__ == "__main__":
    _selftest()
    recs = parse_cgd()
    print("CGD records: %d   (3-periodic cubic: %d)"
          % (len(recs), sum(1 for r in recs if is_cubic(r))))
    print()
    good = survey()
    print()
    print("nets expressible in Pearce's Universal Node: %d" % len(good))
    wyckoff_check()
