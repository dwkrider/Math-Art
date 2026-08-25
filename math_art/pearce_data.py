# Pearce's saddle polyhedra -- the solids of Table 8.1, as geometry.
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

SOLIDS = (
    dict(
        number=1, key='DECATRIHEDRON',
        name='Decatrihedron',
        net='SRS', match='FULL',
        verts=(
            (7, 13, 11), (5, 11, 15), (5, 13, 13), (7, 13, 19), (7, 11, 17),
            (11, 15, 13), (13, 11, 15), (13, 13, 13), (11, 9, 15), (9, 15, 11),
            (9, 9, 17), (9, 15, 19), (11, 17, 15), (9, 17, 17),
        ),
        faces=(
            (0, 2, 1, 4, 3, 11, 13, 12, 5, 9), (9, 5, 7, 6, 8, 10, 4, 1, 2, 0), (3, 4, 10, 8, 6, 7, 5, 12, 13, 11),
        ),
    ),
    dict(
        number=6, key='WURTZITE_TRIHEDRON',
        name='Wurtzite trihedron',
        net='BCC', match='FULL',
        verts=(
            (4, 4, 4), (4, 4, 12), (4, 12, 4), (0, 8, 8), (12, 4, 4),
            (8, 0, 8), (8, 8, 0), (8, 8, 8),
        ),
        faces=(
            (0, 3, 1, 7, 2, 6), (5, 4, 7, 1, 3, 0), (6, 2, 7, 4, 5, 0),
        ),
    ),
    dict(
        number=11, key='DIAMOND_TETRAHEDRON',
        name='Diamond tetrahedron',
        net='DIAMOND', match='FULL',
        verts=(
            (6, 6, 2), (6, 2, 6), (2, 6, 6), (4, 4, 8), (6, 6, 10),
            (4, 8, 4), (6, 10, 6), (8, 4, 4), (10, 6, 6), (8, 8, 8),
        ),
        faces=(
            (7, 1, 3, 2, 5, 0), (0, 5, 6, 9, 8, 7), (7, 8, 9, 4, 3, 1),
            (2, 3, 4, 9, 6, 5),
        ),
    ),
    dict(
        number=12, key='BCC_TETRAHEDRON',
        name='bcc tetrahedron',
        net='BCC', match='FULL',
        verts=(
            (4, 4, 4), (0, 0, 8), (4, 4, 12), (0, 8, 8), (8, 0, 8),
            (8, 8, 8),
        ),
        faces=(
            (0, 1, 2, 3), (4, 2, 1, 0), (0, 3, 2, 5),
            (5, 2, 4, 0),
        ),
    ),
    dict(
        number=14, key='FCC_ORTHORHOMBIC_TETRAHEDRON',
        name='fcc orthorhombic tetrahedron',
        net='FCC', match='FULL',
        verts=(
            (4, 4, 0), (0, 4, 4), (4, 4, 8), (0, 8, 0), (4, 8, 4),
            (0, 8, 8),
        ),
        faces=(
            (4, 2, 1, 0), (0, 1, 3, 4), (1, 2, 4, 5),
            (5, 4, 3, 1),
        ),
    ),
)

#: Rows of Table 8.1 with no verified geometry, each with the
#: reason.  They are NOT offered by the operator.  A row whose
#: reason is not 'no geometry found' is a SEARCH RESULT THAT
#: FAILED THE GATE -- the resolver matched it on counts but the
#: geometry did not survive checking, which is worth recording
#: and is not worth shipping.
UNRESOLVED = (
    (2, 'Universal trihedron (enantiomorphic)', 'not orientable'),
    (3, 'Trirectangular trihedron', 'no geometry found by the search'),
    (4, 'Digonal trihedron (enantiomorphic)', 'not orientable'),
    (5, 'Trigonal trihedron', "face inventory {(4, 'MIRROR', None): 3} != {(4, 'MIRROR', '111'): 3}"),
    (7, 'Delta trihedron', 'no geometry found by the search'),
    (8, 'bcc trihedron', "face inventory {(4, 'MIRROR', None): 3} != {(4, 'MIRROR', '110'): 3}"),
    (9, 'Rectangular trihedron (enantiomorphic)', 'not orientable'),
    (10, 'Double rectangular trihedron', 'no geometry found by the search'),
    (13, 'fcc tetragonal tetrahedron', 'no geometry found by the search'),
    (15, 'Universal tetrahedron', 'no geometry found by the search'),
    (16, 'Digonal tetrahedron', 'no geometry found by the search'),
    (17, 'Truncated orthorhombic tetrahedron', 'no geometry found by the search'),
    (18, 'Digonal hemisaddle tetrahedron', 'no geometry found by the search'),
    (19, 'Wurtzite nodal tetrahedron', 'no geometry found by the search'),
    (20, 'bcc orthorhombic tetrahedron', 'no geometry found by the search'),
    (21, 'Rectangular orthorhombic tetrahedron', 'no geometry found by the search'),
    (22, 'Hemisaddle digonal disphenoid', 'no geometry found by the search'),
    (23, 'Double delta tetrahedron', 'no geometry found by the search'),
    (24, 'Trigonal pentahedron', 'no geometry found by the search'),
    (25, 'Wurtzite pentahedron', 'no geometry found by the search'),
    (26, 'Digonal pentahedron', 'no geometry found by the search'),
    (27, 'Triangular hexahedron', 'no geometry found by the search'),
    (28, 'Cubical saddle hexahedron', 'no geometry found by the search'),
    (29, 'Saddle cube', 'no geometry found by the search'),
    (30, 'Truncated tetragonal tetrahedron', 'no geometry found by the search'),
    (31, 'Universal hexahedron', 'no geometry found by the search'),
    (32, 'Augmented universal hexahedron', 'no geometry found by the search'),
    (33, 'Bioctagonal hexahedron', 'no geometry found by the search'),
    (34, 'Bidodecagonal hexahedron', 'no geometry found by the search'),
    (35, 'Tetragonal saddle hexahedron', 'no geometry found by the search'),
    (36, 'Fissioned tetragonal saddle hexahedron', 'no geometry found by the search'),
    (37, 'Tetragonal octagonal hexahedron', 'no geometry found by the search'),
    (38, 'Trigonal hexahedron', 'no geometry found by the search'),
    (39, 'Trapezoidal trigonal hexahedron', 'no geometry found by the search'),
    (40, 'bcc octahedron', 'no geometry found by the search'),
    (41, 'fcc saddle octahedron', 'no geometry found by the search'),
    (42, 'Tetragonal pentagonal octahedron', 'no geometry found by the search'),
    (43, 'Tetrahedral decahedron', 'no geometry found by the search'),
    (44, 'Saddle dodecahedron', 'no geometry found by the search'),
    (45, 'Blunted saddle dodecahedron', 'no geometry found by the search'),
    (46, 'Truncated tetrahedral decahedron', 'no geometry found by the search'),
    (47, 'bcc saddle cuboctahedron', 'no geometry found by the search'),
    (48, 'Fissioned bcc saddle cuboctahedron', 'no geometry found by the search'),
    (49, 'fcc saddle cuboctahedron', 'no geometry found by the search'),
    (50, 'Truncated fcc saddle cuboctahedron', 'no geometry found by the search'),
    (51, 'Saddle cube dodecahedron', 'no geometry found by the search'),
    (52, 'Truncated saddle dodecahedron', 'no geometry found by the search'),
    (53, 'Universal cuboctadodecahedron', 'no geometry found by the search'),
)


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
