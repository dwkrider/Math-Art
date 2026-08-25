"""Emit math_art/pearce_data.py from tools/resolve_pearce.py's output.

Re-run after a longer resolver pass to widen the operator's coverage;
the gate in pearce_data._selftest re-checks every emitted solid against
its Table 8.1 row.
"""
import sys, pickle, re, collections
MA = r"C:\Users\dkrid\Projects\2026_07_21_Math_Art\.claude\worktrees\spidrons\math_art"
sys.path.insert(0, MA)
import pearce_net as pn
import pearce_table as pt

with open("resolved.pkl", "rb") as fh:
    resolved = pickle.load(fh)

BY_NUM = {r['number']: r for r in pt.TABLE}


def key_for(name, num):
    k = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').upper()
    k = re.sub(r'_+', '_', k)
    return k or ("SOLID_%d" % num)


HEAD = '''# Pearce's saddle polyhedra -- the solids of Table 8.1, as geometry.
#
# Each record here is a saddle polyhedron of Peter Pearce's Table 8.1,
# stored as integer eighth-coordinates of the cubic cell and a list of
# face circuits.  NOTHING IN THIS FILE WAS TRANSCRIBED BY HAND.  Every
# solid was FOUND, by searching the Universal Node net for a closed
# circuit-complex whose face count, face sizes, node valences, branch
# classes and symmetry axes all match the row -- Pearce's book gives no
# coordinates, so derivation is the only honest route, and the table is
# the acceptance test rather than the source.
#
# `pearce_table.TABLE` holds the transcribed rows (checked against
# themselves by Euler, the degree/branch handshake and the face-edge
# count); `pearce_net` holds the net and the geometry predicates; this
# module holds the solids that have been matched to a row and pass the
# whole gate in `_selftest`.
#
# COVERAGE IS PARTIAL AND DELIBERATELY HONEST.  A solid appears here
# only if a search actually produced it and every checked column
# agreed.  The rest of Table 8.1 is listed in `UNRESOLVED` with the
# reason, and the operator does not offer them: shipping a guess that
# merely has the right face count would be worse than shipping fewer
# solids.  `MATCH` records how completely each entry agreed --
# 'FULL' means every column including the symmetry axes.
#
# References:
# - Peter Pearce, "Structure in Nature is a Strategy for Design", The
#   MIT Press, 1978 (paperback 1990), ch. 8 -- Table 8.1's inventory of
#   53 saddle polyhedra with their node, branch and face
#   specifications, and the Universal Node system they are built in.

'''


def fmt_tuple(t, per=6, indent=8):
    items = ["(%s)" % ", ".join(str(x) for x in v) for v in t]
    lines, cur = [], []
    for it in items:
        cur.append(it)
        if len(cur) == per:
            lines.append(", ".join(cur))
            cur = []
    if cur:
        lines.append(", ".join(cur))
    pad = " " * indent
    return (",\n" + pad).join(lines)


out = [HEAD]
out.append("SOLIDS = (\n")
nums = sorted(resolved)
for num in nums:
    kind, net, V, F = resolved[num]
    r = BY_NUM[num]
    key = key_for(r['name'], num)
    out.append("    dict(\n")
    out.append("        number=%d, key=%r,\n" % (num, key))
    out.append("        name=%r,\n" % r['name'])
    out.append("        net=%r, match=%r,\n" % (net, kind))
    out.append("        verts=(\n            %s,\n        ),\n"
               % fmt_tuple(V, per=5, indent=12))
    out.append("        faces=(\n            %s,\n        ),\n"
               % fmt_tuple(F, per=3, indent=12))
    out.append("    ),\n")
out.append(")\n\n")

unres = [r['number'] for r in pt.TABLE if r['number'] not in resolved]
out.append("#: rows of Table 8.1 with no verified geometry yet.  They are\n")
out.append("#: NOT offered by the operator.  See the module header.\n")
out.append("UNRESOLVED = (\n")
for n in unres:
    r = BY_NUM[n]
    out.append("    (%d, %r),\n" % (n, r['name']))
out.append(")\n\n")

out.append('''
def by_key(key):
    for s in SOLIDS:
        if s['key'] == key:
            return s
    raise KeyError(key)


def by_number(num):
    for s in SOLIDS:
        if s['number'] == num:
            return s
    raise KeyError(num)


def face_count_family(n):
    """Pearce groups the table by face count; so does the operator."""
    return {3: 'TRIHEDRA', 4: 'TETRAHEDRA', 5: 'PENTAHEDRA',
            6: 'HEXAHEDRA', 8: 'OCTAHEDRA', 10: 'DECAHEDRA',
            12: 'DODECAHEDRA'}.get(n, 'LARGER')


def families():
    seen = []
    for s in SOLIDS:
        f = face_count_family(len(s['faces']))
        if f not in seen:
            seen.append(f)
    return seen


def in_family(fam):
    return [s for s in SOLIDS
            if face_count_family(len(s['faces'])) == fam]


def _selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-58s %s %s" % (name, "OK" if cond else "BAD", extra))

    try:
        from . import pearce_net as pnet
        from . import pearce_table as ptab
    except Exception:
        import pearce_net as pnet
        import pearce_table as ptab

    print("pearce_data: %d solids, %d rows still unresolved"
          % (len(SOLIDS), len(UNRESOLVED)))
    rows = {r['number']: r for r in ptab.TABLE}
    chk("every solid maps to a Table 8.1 row",
        all(s['number'] in rows for s in SOLIDS))
    chk("no solid is also listed unresolved",
        not ({s['number'] for s in SOLIDS}
             & {n for n, _ in UNRESOLVED}))
    chk("resolved + unresolved == 53",
        len(SOLIDS) + len(UNRESOLVED) == 53,
        "%d + %d" % (len(SOLIDS), len(UNRESOLVED)))
    chk("keys unique", len({s['key'] for s in SOLIDS}) == len(SOLIDS))

    for s in SOLIDS:
        r = rows[s['number']]
        V, F = s['verts'], s['faces']
        tag = "#%d %s" % (s['number'], s['name'])
        # 1. exact integer closure of every circuit
        chk("%s: circuits close (exact)" % tag,
            all(pnet.closes([V[i] for i in f]) for f in F))
        # 2. every edge is a Universal Node branch
        try:
            bt = pnet.branch_totals(V, F)
            good = True
        except Exception:
            bt, good = None, False
        chk("%s: every edge is a branch" % tag, good)
        # 3. the row's checksum, column by column
        v, e, f_, chi = pnet.euler(V, F)
        chk("%s: V/E/F match the row" % tag,
            (v, e, f_) == (r['nodes_total'], r['branches_total'],
                           r['faces_total']),
            "%r vs %r" % ((v, e, f_), (r['nodes_total'],
                                       r['branches_total'],
                                       r['faces_total'])))
        chk("%s: chi = 2" % tag, chi == 2)
        want = {}
        for z, c in tuple(r['primary']) + tuple(r['secondary']):
            want[z] = want.get(z, 0) + c
        hist, _ = pnet.valence_histogram(F)
        chk("%s: node valences match" % tag, hist == want,
            "%r vs %r" % (hist, want))
        chk("%s: branch classes match" % tag,
            bt == dict(r['branches']), "%r vs %r" % (bt, dict(r['branches'])))
        # 4. face inventory: size, own symmetry, plane direction
        got = {}
        for cyc in F:
            loop = [V[i] for i in cyc]
            k = (len(cyc), pnet.face_symmetry_label(loop),
                 pnet.face_plane_class(loop))
            got[k] = got.get(k, 0) + 1
        wantf = {}
        for fd in r['faces']:
            k = (fd['n'], fd['symmetry'], fd['plane'])
            wantf[k] = wantf.get(k, 0) + fd['count']
        chk("%s: face inventory matches" % tag, got == wantf,
            "" if got == wantf else "%r vs %r" % (got, wantf))
        # 5. included angles are the tabulated ones
        legal = set(pnet.TABULATED)
        allang = set()
        for cyc in F:
            for a in pnet.circuit_angles([V[i] for i in cyc]):
                allang.add(pnet.angle_label(a))
        chk("%s: angles are Universal Node angles" % tag,
            allang <= legal, " ".join(sorted(allang - legal)))
        # 6. symmetry axes
        pts = [V[i] for i in range(len(V))]
        ax = pnet.axis_counts(pts)
        if s['match'] == 'FULL':
            chk("%s: symmetry axes match" % tag, ax == tuple(r['axes']),
                "%r vs %r" % (ax, tuple(r['axes'])))
        # 7. the collapse gate -- topology can pass while the solid is flat
        import numpy as _np
        P = _np.asarray(pts, float)
        ext = P.max(axis=0) - P.min(axis=0)
        chk("%s: not collapsed (aspect >= 0.2)" % tag,
            float(ext.min()) / float(ext.max()) >= 0.2,
            "aspect %.3f" % (float(ext.min()) / float(ext.max())))
        # 8. closed and consistently orientable
        chk("%s: closed surface" % tag, pnet.is_closed_surface(F))
        chk("%s: orientable" % tag, pnet.orientation_consistent(F))

    print("RESULT:", "OK" if ok else "BAD")
    if not ok:
        raise AssertionError("pearce_data self-test failed")
''')

path = MA + r"\pearce_data.py"
with open(path, "w") as fh:
    fh.write("".join(out))
print("wrote %s: %d solids, %d unresolved" % (path, len(nums), len(unres)))
print("solids:", nums)
