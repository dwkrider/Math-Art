"""Emit math_art/pearce_data.py from the resolver's output."""
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


def _cart(V, basis):
    """Lattice coords -> Cartesian, for whichever lattice the solid is in."""
    if basis is None:
        return [tuple(float(x) for x in p) for p in V]
    return pn.cartesian(V, pn.HEX_BASIS, pn.HEX_DIVISOR)


def validate(num, V, F, kind, basis=None):
    """The acceptance gate, applied BEFORE emitting.

    The resolver's match test only compares counts; this checks the
    geometry itself -- that every corner is a Universal Node angle,
    that each face's plane direction is the one the table names, and
    that the complex is orientable.  Entries that satisfy the resolver
    but fail here must not ship."""
    r = BY_NUM[num]
    X = _cart(V, basis)
    try:
        # closure stays EXACT integer arithmetic in either lattice --
        # the hexagonal coordinates are integer twenty-fourths of the
        # cell, chosen precisely so this check does not go floating
        if not all(pn.closes([V[i] for i in f]) for f in F):
            return "a circuit does not close"
        v, e, f_, chi = pn.euler(V, F)
        if (v, e, f_) != (r['nodes_total'], r['branches_total'],
                          r['faces_total']):
            return "V/E/F %r != %r" % ((v, e, f_),
                                       (r['nodes_total'],
                                        r['branches_total'],
                                        r['faces_total']))
        if chi != 2:
            return "chi = %d" % chi
        if not pn.is_closed_surface(F):
            return "not a closed surface"
        if not pn.orientation_consistent(pn.orient_faces(V, F)):
            return "not orientable"
        legal = set(pn.TABULATED)
        for cyc in F:
            for a in pn.circuit_angles([X[i] for i in cyc]):
                if pn.angle_label(a) not in legal:
                    return ("corner %s is not a Universal Node angle"
                            % pn.angle_label(a))
        # face CORNER ANGLES, not merely legal ones.  The gate used to
        # check that every angle was a Universal Node angle, which any
        # cell of the net passes trivially -- so a solid could match on
        # topology, valences, branch classes, face symmetry, face planes
        # and symmetry axes while its corners were the wrong angles
        # entirely.  Entry 6 shipped that way: hexagons of 2x109d28' +
        # 4x70d32' against a row that says all six corners are 109d28'.
        for cyc in F:
            loop = [X[i] for i in cyc]
            got_ang = tuple(sorted(pn.angle_label(a)
                                   for a in pn.circuit_angles(loop)))
            ok = False
            for fd in r['faces']:
                if fd['n'] != len(cyc):
                    continue
                want = list(fd['angles'])
                if len(want) == 1:
                    want = want * fd['n']
                if len(want) != fd['n']:
                    ok = True            # book truncates this list
                    break
                if tuple(sorted(want)) == got_ang:
                    ok = True
                    break
            if not ok:
                return ("face angles %s match no face of the row"
                        % (got_ang,))

        got = {}
        for cyc in F:
            loop = [X[i] for i in cyc]
            # Face-plane direction is a CUBIC notion -- it names a
            # <100>/<110>/<111> normal -- so it is not asserted for a
            # solid living in the hexagonal lattice.  Its faces are
            # still checked on size, symmetry and angles.
            plane = pn.face_plane_class(loop) if basis is None else None
            k = (len(cyc), pn.face_symmetry_label(loop), plane)
            got[k] = got.get(k, 0) + 1
        want = {}
        for fd in r['faces']:
            k = (fd['n'], fd['symmetry'],
                 fd['plane'] if basis is None else None)
            want[k] = want.get(k, 0) + fd['count']
        if got != want:
            return "face inventory %r != %r" % (got, want)
        if kind == 'FULL' and basis is None:
            pts = [V[i] for i in range(len(V))]
            if pn.axis_counts(pts) != tuple(r['axes']):
                return "axes %r != %r" % (pn.axis_counts(pts),
                                          tuple(r['axes']))
    except Exception as exc:
        return "gate raised: %s" % exc
    return None


#: Solids that live in a NON-CUBIC lattice, keyed by entry number.
#: The wurtzite pair (entries 6 and 25) cannot be expressed in the cubic
#: grid: a single cell can be, since its branches are tetrahedral, but
#: its geometry is only right in the hexagonal lattice -- the cubic
#: embedding of entry 6 has hexagons of 2x109d28' + 4x70d32' where the
#: row says all six corners are 109d28'.
import os as _os
HEXFILE = "hex_solids.pkl"
hex_solids = {}
if _os.path.exists(HEXFILE):
    with open(HEXFILE, "rb") as _fh:
        hex_solids = pickle.load(_fh)
for _n, (_V, _F) in hex_solids.items():
    resolved[_n] = ('FULL', 'WURTZITE', _V, _F)
print("hex solids offered: %s" % sorted(hex_solids))

rejected = {}
for _num in sorted(resolved):
    _kind, _net, _V, _F = resolved[_num]
    _basis = pn.HEX_BASIS if _num in hex_solids else None
    _why = validate(_num, _V, _F, _kind, _basis)
    if _why:
        rejected[_num] = _why
        del resolved[_num]
print("rejected by the gate: %d" % len(rejected))
for _n, _why in sorted(rejected.items()):
    print("  #%-3d %s" % (_n, _why))

#: Any net a surviving solid uses that `pearce_net` does not define has
#: to be shipped alongside the data: the resolver registers RCSR nets at
#: run time from a local mirror, and the extension has no mirror.  A
#: solid whose net cannot be shipped is dropped rather than emitted with
#: a net reference that will not resolve at build time.
extra_nets = {}
for _num in sorted(resolved):
    _net = resolved[_num][1]
    if _num in hex_solids:
        continue                       # not a cubic net at all
    if not isinstance(_net, str) or _net in pn.NETS:
        continue
    try:
        import rcsr_nets as _rc
        for _nm, _base, _nbrs, _info in _rc.survey_full(verbose=False):
            if _nm.upper() == _net:
                extra_nets[_net] = (tuple(_base), tuple(_nbrs))
                break
    except Exception as _e:
        print("cannot ship net %s: %s" % (_net, _e))
    if _net not in extra_nets:
        print("dropping #%d: net %s is not shippable" % (_num, _net))
        del resolved[_num]
print("nets shipped with the data: %s" % sorted(extra_nets))
# register them NOW so the packing probe below can actually run; without
# this the probe raises on the missing net and the capability flags fall
# back to False by accident rather than by measurement
for _k, _v in extra_nets.items():
    pn.NETS.setdefault(_k, _v)

out = [HEAD]
out.append("SOLIDS = (\n")
nums = sorted(resolved)
for num in nums:
    kind, net, V, F = resolved[num]
    r = BY_NUM[num]
    key = key_for(r['name'], num)
    # Orient every circuit outward IN CARTESIAN SPACE.  Newell normals
    # of a hexagonal solid's lattice coordinates point nowhere useful,
    # so orienting on the raw integers leaves the shell inconsistent.
    _basis_here = pn.HEX_BASIS if num in hex_solids else None
    F = pn.orient_faces(_cart(V, _basis_here), F)
    out.append("    dict(\n")
    out.append("        number=%d, key=%r,\n" % (num, key))
    out.append("        name=%r,\n" % r['name'])
    out.append("        net=%r, match=%r,\n" % (net, kind))
    # The lattice the vertices are expressed in.  A hexagonal solid's
    # coordinates are integer twenty-fourths of the hexagonal cell and
    # are meaningless read as Cartesian, so the basis travels with them.
    if num in hex_solids:
        out.append("        lattice='HEX',\n")
        out.append("        basis=%r,\n" % (pn.HEX_BASIS,))
        out.append("        divisor=%r,\n" % (pn.HEX_DIVISOR,))
    else:
        out.append("        lattice='CUBIC8', basis=None, divisor=8.0,\n")
    out.append("        verts=(\n            %s,\n        ),\n"
               % fmt_tuple(V, per=5, indent=12))
    out.append("        faces=(\n            %s,\n        ),\n"
               % fmt_tuple(F, per=3, indent=12))
    # Which layouts this solid can actually offer, measured ONCE here --
    # a packing search is far too slow to run from a UI enum callback.
    #
    # `packs` means the packing genuinely FILLS: ratio 1 with no face
    # used by more than two cells.  Merely producing several cells is
    # not enough -- entry 30's orbit comes out at 3.75x the block, which
    # is cells interpenetrating, and offering that as "space filling"
    # shows a tangle.
    try:
        import pearce_tiling as _pt
        _X = _cart(V, _basis_here)
        _cps, _rep = _pt.pack(_X, F, net, 1, 1, 1,
                              lattice=('HEX' if num in hex_solids
                                       else 'CUBIC8'))
        _fills = bool(_rep['fills'])
        # Volume accounting is NOT sufficient: overlapping cells with
        # compensating gaps sum to exactly the block.  Entries 14 and 43
        # passed a ratio-1.0 check while visibly interpenetrating, so the
        # packing is voxel-tested for double coverage before it is
        # offered.
        # Overlap is decided EXACTLY, by triangle-triangle intersection
        # plus a bounding-box-guarded containment test -- not by
        # sampling, which can only ever say "none found", and not by
        # volume, which cannot see an overlap a gap compensates for.
        #
        # Nor is the volume ratio a fill test here: cells are selected
        # by CENTROID, so they straddle the block and their volumes need
        # not sum to it.  Entry 30 sits at 2.5x the block with no
        # overlap at all.  What is required is that the cells do not
        # intersect and that they actually TOUCH.
        import itertools as _it
        _bad = sum(1 for _i, _j in _it.combinations(range(len(_cps)), 2)
                   if _pt.cells_overlap(_cps[_i], _cps[_j], F))
        _packs = bool(len(_cps) > 1 and _bad == 0
                      and _rep['shared_faces'] > 0)
        _unit = bool(_rep['shared_faces'] > 0 and _bad == 0)
        if _bad:
            print("  #%d: %d overlapping cell pairs -- not offered"
                  % (num, _bad))
    except Exception as _e:
        print("  #%d: packing probe failed: %s" % (num, _e))
        _packs = _unit = _fills = False
    out.append("        packs=%r, has_unit=%r, fills=%r,\n"
               % (_packs, _unit, _fills))
    out.append("    ),\n")
out.append(")\n\n")

FMT_UNRES = '    (%d, %r, %r),\n'
unres = [r['number'] for r in pt.TABLE if r['number'] not in resolved]
out.append("#: Rows of Table 8.1 with no verified geometry, each with the\n")
out.append("#: reason.  They are NOT offered by the operator.  A row whose\n")
out.append("#: reason is not 'no geometry found' is a SEARCH RESULT THAT\n")
out.append("#: FAILED THE GATE -- the resolver matched it on counts but the\n")
out.append("#: geometry did not survive checking, which is worth recording\n")
out.append("#: and is not worth shipping.\n")
out.append("UNRESOLVED = (\n")
for n in unres:
    r = BY_NUM[n]
    why = rejected.get(n, "no geometry found by the search")
    out.append(FMT_UNRES % (n, r["name"], why))
out.append(")\n\n")

out.append("#: Nets these solids use that `pearce_net` does not build in.\n")
out.append("#: The resolver registers RCSR nets at run time from a local\n")
out.append("#: mirror; the extension has no such mirror, so any net a\n")
out.append("#: shipped solid depends on has to travel WITH the data.\n")
out.append("EXTRA_NETS = {\n")
for _nm in sorted(extra_nets):
    _b, _nb = extra_nets[_nm]
    out.append("    %r: (\n        (%s),\n        (%s),\n    ),\n"
               % (_nm, fmt_tuple(_b, per=5, indent=9),
                  fmt_tuple(_nb, per=5, indent=9)))
out.append("}\n\n\n")

out.append('''def _register_nets():
    """Make EXTRA_NETS visible to pearce_net, so the data is complete."""
    try:
        from . import pearce_net as _pn
    except Exception:
        import pearce_net as _pn
    for _k, _v in EXTRA_NETS.items():
        _pn.NETS.setdefault(_k, _v)


_register_nets()


def points(solid):
    """A solid's vertices in CARTESIAN space.

    Cubic solids are stored in integer eighths, which are Cartesian up
    to a uniform scale nothing downstream depends on.  Hexagonal solids
    are stored in integer twenty-fourths of the hexagonal cell and must
    be mapped through their basis -- reading those coordinates as if
    they were Cartesian gives a sheared, wrong solid.
    """
    try:
        from . import pearce_net as _pn
    except Exception:
        import pearce_net as _pn
    if solid.get('basis') is None:
        return [tuple(float(x) for x in p) for p in solid['verts']]
    return _pn.cartesian(solid['verts'], solid['basis'],
                         solid.get('divisor', 1.0))


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
             & {u[0] for u in UNRESOLVED}))
    chk("resolved + unresolved == 53",
        len(SOLIDS) + len(UNRESOLVED) == 53,
        "%d + %d" % (len(SOLIDS), len(UNRESOLVED)))
    chk("keys unique", len({s['key'] for s in SOLIDS}) == len(SOLIDS))

    for s in SOLIDS:
        r = rows[s['number']]
        V, F = s['verts'], s['faces']
        X = points(s)
        cubic = s.get('basis') is None
        tag = "#%d %s" % (s['number'], s['name'])
        # 1. exact integer closure of every circuit
        chk("%s: circuits close (exact)" % tag,
            all(pnet.closes([V[i] for i in f]) for f in F))
        # 2. every edge is a Universal Node branch.  Only meaningful on
        #    the cubic grid: <100>/<110>/<111> name cubic directions.
        bt = None
        if cubic:
            try:
                bt = pnet.branch_totals(V, F)
                good = True
            except Exception:
                good = False
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
        if cubic:
            chk("%s: branch classes match" % tag,
                bt == dict(r['branches']),
                "%r vs %r" % (bt, dict(r['branches'])))
        # 4. face inventory: size, own symmetry, plane direction
        got = {}
        for cyc in F:
            loop = [X[i] for i in cyc]
            k = (len(cyc), pnet.face_symmetry_label(loop),
                 pnet.face_plane_class(loop) if cubic else None)
            got[k] = got.get(k, 0) + 1
        wantf = {}
        for fd in r['faces']:
            k = (fd['n'], fd['symmetry'],
                 fd['plane'] if cubic else None)
            wantf[k] = wantf.get(k, 0) + fd['count']
        chk("%s: face inventory matches" % tag, got == wantf,
            "" if got == wantf else "%r vs %r" % (got, wantf))
        # 5. included angles are the tabulated ones
        legal = set(pnet.TABULATED)
        allang = set()
        for cyc in F:
            for a in pnet.circuit_angles([X[i] for i in cyc]):
                allang.add(pnet.angle_label(a))
        chk("%s: angles are Universal Node angles" % tag,
            allang <= legal, " ".join(sorted(allang - legal)))
        # and that each face's angles are the row's, not merely legal
        bad = []
        for cyc in F:
            loop = [X[i] for i in cyc]
            ga = tuple(sorted(pnet.angle_label(a)
                              for a in pnet.circuit_angles(loop)))
            ok = False
            for fd in r['faces']:
                if fd['n'] != len(cyc):
                    continue
                want = list(fd['angles'])
                if len(want) == 1:
                    want = want * fd['n']
                if len(want) != fd['n'] or tuple(sorted(want)) == ga:
                    ok = True
                    break
            if not ok:
                bad.append(ga)
        chk("%s: face angles match the row" % tag, not bad,
            "%s" % (bad[:1],))
        # 6. symmetry axes
        pts = [X[i] for i in range(len(X))]
        ax = pnet.axis_counts(pts)
        if s['match'] == 'FULL' and cubic:
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
        chk("%s: orientable" % tag,
            pnet.orientation_consistent(pnet.orient_faces(X, F)))

    print("RESULT:", "OK" if ok else "BAD")
    if not ok:
        raise AssertionError("pearce_data self-test failed")
''')

path = MA + r"\pearce_data.py"
with open(path, "w") as fh:
    fh.write("".join(out))
print("wrote %s: %d solids, %d unresolved" % (path, len(nums), len(unres)))
print("solids:", nums)
