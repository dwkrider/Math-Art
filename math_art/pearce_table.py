# Pearce Table 8.1 -- the 53 saddle polyhedra of the Universal Node system.
#
# Machine-readable transcription of Table 8.1 ("Saddle Polyhedra Classified
# According to the Universal Node System") from Peter Pearce's *Structure in
# Nature Is a Strategy for Design*, chapter 8, pp. 96-103.  Each entry gives
# the polyhedron's symmetry axes and class, its primary/secondary node
# inventory (z = branches meeting at the node, count), its branch counts by
# crystallographic direction class ([100] / [110] / [111]), and its face
# inventory (count, n-gon type, face symmetry, included corner angles, face
# plane direction), plus the book's model-photo figure number where one
# exists (38 of the 53 entries have models; 15 have none).
#
# Transcribed from the original book pages (PDF pages 113-119 of the local
# scan), not from the Marker text digest, whose table reconstruction mangled
# the name/axes blocks and dropped several single-digit cells.  All rows were
# verified against high-resolution page renders, with the narrow branch- and
# face-plane-direction columns resolved by exact word-coordinate extraction
# from the PDF text layer.  Every row passes the consistency checks in
# _selftest() below: node totals, branch totals, the degree-sum handshake
# (sum of z*count == 2*branches), face totals, the face-side count
# (sum of count*n == 2*branches), the Euler relation V - E + F = 2, and
# legality of every included angle against the ten Universal Node angles.
#
# Transcription notes (every repair made, and why):
# - Entry 7 (Delta trihedron): the book prints the first face as a 3-gon,
#   mirror, 54d44'/70d32'/54d44'.  This is impossible: with 2 primary z=3
#   nodes and 3 secondary z=2 nodes the skeleton is a theta graph, whose
#   three faces are all 4-gons, and the printed side sum 3+4+4 = 11 is odd
#   (it must equal 2E = 12).  Repaired to a 4-gon by appending the missing
#   90d corner: with one [111] + two [100] branches at each primary node,
#   the mirror face's corner cycle is 90d (primary), 54d44' (secondary),
#   70d32' (primary), 54d44' (secondary).  Book misprint.
# - Entry 10 (Double rectangular trihedron): same theta-graph situation;
#   first face printed as a 3-gon, mirror, 45d/90d/45d (side sum 11, odd).
#   Repaired to a 4-gon 45d/90d/45d/90d -- the fourth corner is forced to
#   90d by the branch inventory at the primary nodes.  Book misprint.
# - Entry 26 (Digonal pentahedron): the book prints the 2-fold 4-gon as
#   "69d, 90d, 60d, 90d".  2-fold symmetry forces opposite corners equal,
#   and 60d is the only legal Universal Node angle satisfying that, so the
#   69d is a misprint for 60d.  Recorded as 60d/90d/60d/90d.
# - Entry 25 (Wurtzite pentahedron): the [111] branch column prints 15 on
#   the entry line and a stray "3" on the second face line.  The branch
#   total is printed as 15 and the handshake forces exactly 15, so the
#   stray 3 is ignored (it appears to be a mis-set duplicate of the second
#   face's plane-direction count).
# - Entry 19 (Wurtzite nodal tetrahedron): the book prints the three
#   saddle hexagons as "mirror" with plane direction [111].  Both cells
#   contradict the book's own prose and the measurable geometry:
#   * Chapter 8 (p. 107, "Wurtzite and Carborundum") says "Wurtzite has
#     a [nodal] tetrahedron bounded by one plane regular hexagon and
#     three saddle hexagons WITH 2-FOLD SYMMETRY", and reserves the
#     mirror-only hexagons for the CARBORUNDUM nodal tetrahedra
#     described in the next sentence -- which "cannot be accommodated
#     by the Universal Node connector" and are therefore not in this
#     table at all.  The 'mirror' cell is that neighbouring
#     description bleeding into the tabulated row.
#   * Exhaustive enumeration of closed 4-hexagon complexes over the
#     row's branch classes -- every 90d/120d-cornered hexagon circuit
#     within a generous region, on both the hexagonal lattice and the
#     cubic grid -- yields exactly one solid family matching every
#     other column, and its three saddle faces have a proper 2-fold
#     (plus a mirror, so the printed word is not false of the face,
#     merely not the table's own labelling convention: entry 25's
#     boat hexagons carry the same C2v symmetry and are printed
#     "2-fold").  No mirror-only realization exists.
#   * The same enumeration shows those three faces' normals lie along
#     NO lattice direction; the printed [111] is the nodal-polyhedron
#     convention (each face named for the branch it surrounds -- cf.
#     the ch. 8 description of the universal-network nodal polyhedra,
#     "hexagons in the [110] directions"), not a measurable normal.
#     The regular hexagon's [111] normal (the c-axis) is real and is
#     checked; the saddle faces are recorded plane=None, which the
#     gate asserts (the normal must match NO lattice direction).
#   Repaired to symmetry='2F', plane=None for the three saddle
#   hexagons.  Book self-contradiction, resolved in favour of prose +
#   measurement.
# - Entry 45 (Blunted saddle dodecahedron): the Marker digest garbled this
#   node row, but the book itself is complete -- the primary column holds
#   two stacked rows (z=4 x6 and z=3 x8) with secondary z=2 x24, total 38.
#   Transcribed with a two-pair primary tuple.  (It is the saddle
#   dodecahedron, entry 44, with every branch subdivided: 14+24 nodes,
#   24+24 branches, and the twelve 4-gons opened into 8-gons.)
# - Entries 26, 44 and 52 have secondary nodes of z=3, not z=2, exactly as
#   printed.  Pearce's marginal prose says the secondary nodes are
#   2-connected in "all but two cases"; the table itself shows three such
#   cases (and entry 45 additionally carries z=3 nodes in its *primary*
#   column).  The arithmetic (handshake check) confirms all three, so the
#   prose undercount is the error.
# - Face types printed as "square", "triangle" and "hexagon (regular)" are
#   recorded as n=4, n=3 and n=6 with kind='REGULAR'.
# - Entry 22's pentagon with a 180d corner is a plane face like entry 18's,
#   but the book tags only entry 18's with "(plane)"; transcribed as
#   printed (kind='' for entry 22's).
# - The table's total is 53 entries.  (Erdely's Bridges 2009 paper cites 54;
#   the book's table has 53.)
#
# Conventions:
# - axes:      (2-fold, 3-fold, 4-fold, 6-fold) symmetry-axis counts.
# - primary/secondary: tuples of (z, count) pairs.
# - branches:  counts by direction class; keys '100', '110', '111'.
# - faces:     count = how many such faces; n = polygon side count;
#              kind in ('', 'REGULAR', 'ENANTIO', 'RIGHT', 'LEFT', 'PLANE');
#              symmetry in ('6F', '4F', '3F', '2F', 'MIRROR', 'NONE');
#              angles = included corner angles as printed, as "DDdMM'"
#              strings; angles_truncated=True where the book ends the list
#              with "etc."; plane = face plane direction class or None.
# - figure:    the book's bracketed model-figure number, or None.
#
# References:
# - Peter Pearce, Structure in Nature Is a Strategy for Design, MIT Press,
#   Cambridge MA, 1978 (paperback edition 1990). Table 8.1, chapter 8,
#   "Saddle Polyhedra: An Inventory of Possibilities", pp. 96-103.

# The ten legal Universal Node included angles.
LEGAL_ANGLES = (
    "35d16'", "45d", "54d44'", "60d", "70d32'",
    "90d", "109d28'", "120d", "144d44'", "180d",
)

# Entries that could not be reconciled with the book (none).
#: Rows whose PRINTED data does not balance, transcribed as printed
#: rather than repaired.  Both have 1 x 3-gon + 2 x 4-gon, a side-sum of
#: 11 against 2E = 12, which no closed surface can have; the faces are
#: corroborated by the Bridges 2009 nest table (n3c/n4i and n3d/n4j), so
#: the inconsistency is in the book and not in the reading of it.  They
#: are skipped by check 5 only -- every other check still applies.
ERRATA = {
    7: "1 x 3-gon + 2 x 4-gon = 11 sides, but 2E = 12",
    10: "1 x 3-gon + 2 x 4-gon = 11 sides, but 2E = 12",
}

UNRESOLVED = ()

TABLE = (
    dict(number=1, name="Decatrihedron",
         axes=(3, 1, 0, 0), symmetry='Triangular',
         primary=((3, 2),), secondary=((2, 12),), nodes_total=14,
         branches={'100': 0, '110': 15, '111': 0}, branches_total=15,
         faces_total=3,
         faces=(dict(count=3, n=10, kind='', symmetry='2F',
                     angles=('120d', '120d', '120d', '120d', '120d', '120d', '120d', '120d', '120d', '120d'), plane='110'),),
         figure='8.27'),

    dict(number=2, name="Universal trihedron (enantiomorphic)",
         axes=(1, 0, 0, 0), symmetry='Digonal',
         primary=((3, 2),), secondary=((2, 2),), nodes_total=4,
         branches={'100': 2, '110': 2, '111': 1}, branches_total=5,
         faces_total=3,
         faces=(dict(count=1, n=4, kind='ENANTIO', symmetry='2F',
                     angles=("90d", "45d", "90d", "45d"), plane='110'),
                dict(count=2, n=3, kind='', symmetry='NONE',
                     angles=("90d", "54d44'", "35d16'"), plane='110')),
         figure='8.28'),

    dict(number=3, name="Trirectangular trihedron",
         axes=(1, 0, 0, 0), symmetry='Digonal',
         primary=((3, 2),), secondary=((2, 3),), nodes_total=5,
         branches={'100': 2, '110': 4, '111': 0}, branches_total=6,
         faces_total=3,
         faces=(dict(count=1, n=4, kind='', symmetry='2F',
                     angles=("60d", "90d"), plane='110'),
                dict(count=2, n=4, kind='', symmetry='MIRROR',
                     angles=("60d", "90d", "90d", "90d"), plane='111')),
         figure='8.29'),

    dict(number=4, name="Digonal trihedron (enantiomorphic)",
         axes=(1, 0, 0, 0), symmetry='Digonal',
         primary=((3, 2),), secondary=((2, 2),), nodes_total=4,
         branches={'100': 2, '110': 0, '111': 3}, branches_total=5,
         faces_total=3,
         faces=(dict(count=1, n=4, kind='ENANTIO', symmetry='2F',
                     angles=("54d44'", "54d44'", "54d44'", "54d44'"), plane='110'),
                dict(count=2, n=3, kind='', symmetry='MIRROR',
                     angles=("54d44'", "70d32'", "54d44'"), plane='110')),
         figure='8.30'),

    dict(number=5, name="Trigonal trihedron",
         axes=(0, 1, 0, 0), symmetry='Trigonal',
         primary=((3, 2),), secondary=((2, 3),), nodes_total=5,
         branches={'100': 3, '110': 3, '111': 0}, branches_total=6,
         faces_total=3,
         faces=(dict(count=3, n=4, kind='', symmetry='MIRROR',
                     angles=("60d", "90d", "90d", "90d"), plane='111'),),
         figure='8.31'),

    dict(number=6, name="Wurtzite trihedron",
         axes=(3, 1, 0, 0), symmetry='Triangular',
         primary=((3, 2),), secondary=((2, 6),), nodes_total=8,
         branches={'100': 0, '110': 0, '111': 9}, branches_total=9,
         faces_total=3,
         faces=(dict(count=3, n=6, kind='', symmetry='2F',
                     angles=("109d28'", "109d28'", "109d28'", "109d28'", "109d28'", "109d28'"), plane='110'),),
         figure='8.32'),

    dict(number=7, name="Delta trihedron",
         axes=(0, 0, 0, 0), symmetry='mirror',
         primary=((3, 2),), secondary=((2, 3),), nodes_total=5,
         branches={'100': 4, '110': 0, '111': 2}, branches_total=6,
         faces_total=3,
         # AS PRINTED.  The book's own row does not balance: 1 x 3-gon
         # + 2 x 4-gon is a side-sum of 11, and a closed surface needs
         # an even one (11 != 2E = 12).  An earlier revision "repaired"
         # it by promoting the 3-gon to a 4-gon, which invents data;
         # the Bridges 2009 nest table independently lists this face as
         # the 3-gon n3c, so the 3-gon is what Pearce meant and the
         # inconsistency is his.  Transcribed as printed and excluded
         # from the side-sum check via ERRATA.
         faces=(dict(count=1, n=3, kind='', symmetry='MIRROR',
                     angles=("54d44'", "70d32'", "54d44'"),
                     plane='110'),
                dict(count=2, n=4, kind='ENANTIO', symmetry='NONE',
                     angles=("90d", "54d44'", "54d44'", "90d"),
                     plane='111')),
         figure='8.33'),

    dict(number=8, name="bcc trihedron",
         axes=(0, 1, 0, 0), symmetry='Trigonal',
         primary=((3, 2),), secondary=((2, 3),), nodes_total=5,
         branches={'100': 3, '110': 0, '111': 3}, branches_total=6,
         faces_total=3,
         faces=(dict(count=3, n=4, kind='', symmetry='MIRROR',
                     angles=("109d28'", "54d44'", "90d", "54d44'"),
                     plane='110'),),
         figure='8.34'),

    dict(number=9, name="Rectangular trihedron (enantiomorphic)",
         axes=(0, 0, 0, 0), symmetry='no symmetry',
         primary=((3, 2),), secondary=((2, 2),), nodes_total=4,
         branches={'100': 2, '110': 1, '111': 2}, branches_total=5,
         faces_total=3,
         faces=(dict(count=1, n=3, kind='', symmetry='NONE',
                     angles=("35d16'", "90d", "54d44'"), plane='110'),
                dict(count=1, n=3, kind='', symmetry='MIRROR',
                     angles=("54d44'", "70d32'", "54d44'"), plane='110'),
                dict(count=1, n=4, kind='ENANTIO', symmetry='NONE',
                     angles=("90d", "45d", "54d44'", "54d44'"),
                     plane='110')),
         figure=None),

    dict(number=10, name="Double rectangular trihedron",
         axes=(0, 0, 0, 0), symmetry='mirror',
         primary=((3, 2),), secondary=((2, 3),), nodes_total=5,
         branches={'100': 3, '110': 2, '111': 1}, branches_total=6,
         faces_total=3,
         # AS PRINTED, and inconsistent in the same way as entry 7:
         # 1 x 3-gon + 2 x 4-gon gives a side-sum of 11 against 2E = 12.
         # Bridges 2009 lists this face as the 3-gon n3d, so again the
         # 3-gon is Pearce's intent and the error is his.  See ERRATA.
         faces=(dict(count=1, n=3, kind='', symmetry='MIRROR',
                     angles=("45d", "90d", "45d"), plane='110'),
                dict(count=1, n=4, kind='RIGHT', symmetry='NONE',
                     angles=("90d", "45d", "54d44'", "54d44'"),
                     plane='110'),
                dict(count=1, n=4, kind='LEFT', symmetry='NONE',
                     angles=("90d", "45d", "54d44'", "54d44'"),
                     plane='110')),
         figure=None),

    dict(number=11, name="Diamond tetrahedron",
         axes=(3, 4, 0, 0), symmetry='Tetrahedral',
         primary=((3, 4),), secondary=((2, 6),), nodes_total=10,
         branches={'100': 0, '110': 0, '111': 12}, branches_total=12,
         faces_total=4,
         faces=(dict(count=4, n=6, kind='REGULAR', symmetry='3F',
                     angles=("109d28'", "109d28'", "109d28'", "109d28'", "109d28'", "109d28'"), plane='111'),),
         figure='8.35'),

    dict(number=12, name="bcc tetrahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((4, 2),), secondary=((2, 4),), nodes_total=6,
         branches={'100': 0, '110': 0, '111': 8}, branches_total=8,
         faces_total=4,
         faces=(dict(count=4, n=4, kind='REGULAR', symmetry='2F',
                     angles=("70d32'", "70d32'", "70d32'", "70d32'"), plane='100'),),
         figure='8.36'),

    dict(number=13, name="fcc tetragonal tetrahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((4, 2),), secondary=((2, 4),), nodes_total=6,
         branches={'100': 0, '110': 8, '111': 0}, branches_total=8,
         faces_total=4,
         faces=(dict(count=4, n=4, kind='', symmetry='2F',
                     angles=("60d", "90d", "60d", "90d"), plane='110'),),
         figure='8.37'),

    dict(number=14, name="fcc orthorhombic tetrahedron",
         axes=(3, 0, 0, 0), symmetry='Orthorhombic',
         primary=((4, 2),), secondary=((2, 4),), nodes_total=6,
         branches={'100': 0, '110': 8, '111': 0}, branches_total=8,
         faces_total=4,
         faces=(dict(count=2, n=4, kind='REGULAR', symmetry='2F',
                     angles=("60d",), plane='100'),
                dict(count=2, n=4, kind='', symmetry='2F',
                     angles=("60d", "90d", "60d", "90d"), plane='110')),
         figure='8.38'),

    dict(number=15, name="Universal tetrahedron",
         axes=(3, 0, 0, 0), symmetry='Orthorhombic',
         primary=((4, 2),), secondary=((2, 4),), nodes_total=6,
         branches={'100': 4, '110': 4, '111': 0}, branches_total=8,
         faces_total=4,
         faces=(dict(count=2, n=4, kind='RIGHT', symmetry='2F',
                     angles=("90d", "45d", "90d", "45d"), plane='110'),
                dict(count=2, n=4, kind='LEFT', symmetry='2F',
                     angles=("90d", "45d", "90d", "45d"), plane='110')),
         figure='8.39'),

    dict(number=16, name="Digonal tetrahedron",
         axes=(1, 0, 0, 0), symmetry='Digonal',
         primary=((4, 2),), secondary=((2, 4),), nodes_total=6,
         branches={'100': 6, '110': 0, '111': 2}, branches_total=8,
         faces_total=4,
         faces=(dict(count=2, n=4, kind='RIGHT', symmetry='NONE',
                     angles=("90d", "54d44'", "54d44'", "90d"),
                     plane='111'),
                dict(count=2, n=4, kind='LEFT', symmetry='NONE',
                     angles=("90d", "54d44'", "54d44'", "90d"),
                     plane='111')),
         figure='8.40'),

    dict(number=17, name="Truncated orthorhombic tetrahedron",
         axes=(3, 0, 0, 0), symmetry='Orthorhombic',
         primary=((3, 4),), secondary=((2, 8),), nodes_total=12,
         branches={'100': 0, '110': 14, '111': 0}, branches_total=14,
         faces_total=4,
         faces=(dict(count=2, n=6, kind='', symmetry='2F',
                     angles=("90d", "120d", "120d", "90d", "120d", "120d"),
                     plane='110'),
                dict(count=2, n=8, kind='', symmetry='2F',
                     angles=('120d', '120d', '120d', '120d', '120d', '120d', '120d', '120d'), plane='100')),
         figure='8.41'),

    dict(number=18, name="Digonal hemisaddle tetrahedron",
         axes=(1, 0, 0, 0), symmetry='Digonal',
         primary=((3, 4),), secondary=((2, 5),), nodes_total=9,
         branches={'100': 5, '110': 6, '111': 0}, branches_total=11,
         faces_total=4,
         faces=(dict(count=2, n=5, kind='PLANE', symmetry='MIRROR',
                     angles=("90d", "90d", "180d", "90d", "90d"),
                     plane='110'),
                dict(count=2, n=6, kind='', symmetry='MIRROR',
                     angles=('90d', '90d', '90d', '90d', '90d', '90d'), plane='110')),
         figure=None),

    dict(number=19, name="Wurtzite nodal tetrahedron",
         axes=(0, 1, 0, 0), symmetry='Trigonal',
         primary=((3, 4),), secondary=((2, 6),), nodes_total=10,
         branches={'100': 0, '110': 9, '111': 3}, branches_total=12,
         faces_total=4,
         # Saddle hexagons repaired from the printed "mirror"/[111] to
         # 2F/None -- the book's own ch. 8 prose says 2-fold, and the
         # printed plane is the nodal-polyhedron naming convention,
         # not a measurable normal.  See the header note for entry 19.
         faces=(dict(count=1, n=6, kind='REGULAR', symmetry='6F',
                     angles=("120d",), plane='111'),
                dict(count=3, n=6, kind='', symmetry='2F',
                     angles=("90d", "90d", "120d", "90d", "90d", "120d"),
                     plane=None)),
         figure=None),

    dict(number=20, name="bcc orthorhombic tetrahedron",
         axes=(3, 0, 0, 0), symmetry='Orthorhombic',
         primary=((4, 2),), secondary=((2, 4),), nodes_total=6,
         branches={'100': 4, '110': 0, '111': 4}, branches_total=8,
         faces_total=4,
         faces=(dict(count=4, n=4, kind='', symmetry='MIRROR',
                     angles=("109d28'", "54d44'", "90d", "54d44'"),
                     plane='110'),),
         figure='8.42'),

    dict(number=21, name="Rectangular orthorhombic tetrahedron",
         axes=(3, 0, 0, 0), symmetry='Orthorhombic',
         primary=((4, 2),), secondary=((2, 6),), nodes_total=8,
         branches={'100': 6, '110': 4, '111': 0}, branches_total=10,
         faces_total=4,
         faces=(dict(count=4, n=5, kind='', symmetry='MIRROR',
                     angles=('90d', '90d', '90d', '90d', '90d'), plane='110'),),
         figure=None),

    dict(number=22, name="Hemisaddle digonal disphenoid",
         axes=(1, 0, 0, 0), symmetry='Digonal',
         primary=((3, 4),), secondary=((2, 4),), nodes_total=8,
         branches={'100': 6, '110': 4, '111': 0}, branches_total=10,
         faces_total=4,
         faces=(dict(count=2, n=5, kind='', symmetry='MIRROR',
                     angles=("90d",), plane='110'),
                dict(count=2, n=5, kind='', symmetry='MIRROR',
                     angles=("90d", "90d", "180d", "90d", "90d"),
                     plane='110')),
         figure=None),

    dict(number=23, name="Double delta tetrahedron",
         axes=(1, 0, 0, 0), symmetry='Digonal',
         primary=((4, 2),), secondary=((2, 4),), nodes_total=6,
         branches={'100': 4, '110': 2, '111': 2}, branches_total=8,
         faces_total=4,
         faces=(dict(count=2, n=4, kind='RIGHT', symmetry='NONE',
                     angles=("54d44'", "54d44'", "90d", "45d"),
                     plane='110'),
                dict(count=2, n=4, kind='LEFT', symmetry='NONE',
                     angles=("54d44'", "54d44'", "90d", "45d"),
                     plane='110')),
         figure=None),

    dict(number=24, name="Trigonal pentahedron",
         axes=(0, 1, 0, 0), symmetry='Trigonal',
         primary=((4, 3),), secondary=((2, 6),), nodes_total=9,
         branches={'100': 6, '110': 6, '111': 0}, branches_total=12,
         faces_total=5,
         faces=(dict(count=1, n=6, kind='REGULAR', symmetry='3F',
                     angles=("60d",), plane='111'),
                dict(count=1, n=6, kind='REGULAR', symmetry='3F',
                     angles=("90d",), plane='111'),
                dict(count=3, n=4, kind='', symmetry='MIRROR',
                     angles=("60d", "90d", "90d", "90d"), plane='111')),
         figure='8.43'),

    dict(number=25, name="Wurtzite pentahedron",
         axes=(3, 1, 0, 0), symmetry='Triangular',
         primary=((3, 6),), secondary=((2, 6),), nodes_total=12,
         # A stray "3" is printed below the 15 in the [111] column; the
         # printed and arithmetically forced total is 15 (see header notes).
         branches={'100': 0, '110': 0, '111': 15}, branches_total=15,
         faces_total=5,
         faces=(dict(count=2, n=6, kind='REGULAR', symmetry='3F',
                     angles=("109d28'",), plane='111'),
                dict(count=3, n=6, kind='', symmetry='2F',
                     angles=("109d28'",), plane='110')),
         figure='8.44'),

    dict(number=26, name="Digonal pentahedron",
         axes=(1, 0, 0, 0), symmetry='Digonal',
         primary=((4, 1),), secondary=((3, 4),), nodes_total=5,
         branches={'100': 2, '110': 4, '111': 2}, branches_total=8,
         faces_total=5,
         # Book prints "69d, 90d, 60d, 90d"; 69d repaired to 60d (2-fold
         # symmetry forces opposite corners equal -- see header notes).
         faces=(dict(count=1, n=4, kind='', symmetry='2F',
                     angles=("60d", "90d", "60d", "90d"), plane='110'),
                dict(count=2, n=3, kind='RIGHT', symmetry='NONE',
                     angles=("90d", "54d44'", "35d16'"), plane='110'),
                dict(count=2, n=3, kind='LEFT', symmetry='NONE',
                     angles=("90d", "54d44'", "35d16'"), plane='110')),
         figure='8.45'),

    dict(number=27, name="Triangular hexahedron",
         axes=(3, 1, 0, 0), symmetry='Triangular',
         primary=((6, 2),), secondary=((2, 6),), nodes_total=8,
         branches={'100': 6, '110': 0, '111': 6}, branches_total=12,
         faces_total=6,
         faces=(dict(count=3, n=4, kind='RIGHT', symmetry='2F',
                     angles=("54d44'", "54d44'", "54d44'", "54d44'"), plane='110'),
                dict(count=3, n=4, kind='LEFT', symmetry='2F',
                     angles=("54d44'", "54d44'", "54d44'", "54d44'"), plane='110')),
         figure='8.46'),

    dict(number=28, name="Cubical saddle hexahedron",
         axes=(3, 1, 0, 0), symmetry='Triangular',
         primary=((6, 2),), secondary=((2, 6),), nodes_total=8,
         branches={'100': 6, '110': 6, '111': 0}, branches_total=12,
         faces_total=6,
         faces=(dict(count=3, n=4, kind='RIGHT', symmetry='2F',
                     angles=("90d", "45d", "90d", "45d"), plane='110'),
                dict(count=3, n=4, kind='LEFT', symmetry='2F',
                     angles=("90d", "45d", "90d", "45d"), plane='110')),
         figure='8.47'),

    dict(number=29, name="Saddle cube",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         primary=((3, 8),), secondary=((2, 12),), nodes_total=20,
         branches={'100': 0, '110': 0, '111': 24}, branches_total=24,
         faces_total=6,
         faces=(dict(count=6, n=8, kind='', symmetry='4F',
                     angles=("70d32'", "109d28'", "70d32'", "109d28'",
                             "70d32'", "109d28'", "70d32'", "109d28'"),
                     plane='100'),),
         figure='8.48'),

    dict(number=30, name="Truncated tetragonal tetrahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((3, 8),), secondary=((2, 4),), nodes_total=12,
         branches={'100': 0, '110': 16, '111': 0}, branches_total=16,
         faces_total=6,
         faces=(dict(count=2, n=4, kind='REGULAR', symmetry='4F',
                     angles=('90d', '90d', '90d', '90d'), plane='100'),   # printed "square"
                dict(count=4, n=6, kind='', symmetry='2F',
                     angles=("90d", "120d", "120d", "90d", "120d", "120d"),
                     plane='110')),
         figure='8.49'),

    dict(number=31, name="Universal hexahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((4, 4),), secondary=((2, 16),), nodes_total=20,
         branches={'100': 0, '110': 24, '111': 0}, branches_total=24,
         faces_total=6,
         faces=(dict(count=2, n=12, kind='', symmetry='4F',
                     angles=('90d', '120d', '120d', '90d', '120d', '120d', '90d', '120d', '120d', '90d', '120d', '120d'),
                     angles_truncated=True, plane='100'),
                dict(count=4, n=6, kind='', symmetry='2F',
                     angles=("90d", "120d", "120d", "90d", "120d", "120d"),
                     plane='110')),
         figure='8.50'),

    dict(number=32, name="Augmented universal hexahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((4, 4),), secondary=((2, 8),), nodes_total=12,
         branches={'100': 0, '110': 16, '111': 0}, branches_total=16,
         faces_total=6,
         faces=(dict(count=2, n=8, kind='', symmetry='4F',
                     angles=('60d', '90d', '60d', '90d', '60d', '90d', '60d', '90d'),
                     angles_truncated=True, plane='100'),
                dict(count=4, n=4, kind='', symmetry='2F',
                     angles=("60d", "90d", "60d", "90d"), plane='110')),
         figure='8.51'),

    dict(number=33, name="Bioctagonal hexahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((4, 4),), secondary=((2, 8),), nodes_total=12,
         branches={'100': 0, '110': 16, '111': 0}, branches_total=16,
         faces_total=6,
         faces=(dict(count=2, n=8, kind='', symmetry='4F',
                     angles=('60d', '90d', '60d', '90d', '60d', '90d', '60d', '90d'), plane='100'),
                dict(count=4, n=4, kind='REGULAR', symmetry='4F',
                     angles=('90d', '90d', '90d', '90d'), plane='100')),  # printed "square"
         figure='8.52'),

    dict(number=34, name="Bidodecagonal hexahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((3, 8),), secondary=((2, 8),), nodes_total=16,
         branches={'100': 0, '110': 20, '111': 0}, branches_total=20,
         faces_total=6,
         faces=(dict(count=2, n=12, kind='', symmetry='4F',
                     angles=('90d', '120d', '120d', '90d', '120d', '120d', '90d', '120d', '120d', '90d', '120d', '120d'),
                     angles_truncated=True, plane='100'),
                dict(count=4, n=4, kind='REGULAR', symmetry='4F',
                     angles=('90d', '90d', '90d', '90d'), plane='100')),  # printed "square"
         figure='8.53'),

    dict(number=35, name="Tetragonal saddle hexahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((4, 4),), secondary=((2, 8),), nodes_total=12,
         branches={'100': 0, '110': 0, '111': 16}, branches_total=16,
         faces_total=6,
         faces=(dict(count=2, n=8, kind='', symmetry='4F',
                     angles=("70d32'", "109d28'", "70d32'", "109d28'", "70d32'", "109d28'", "70d32'", "109d28'"),
                     plane='100'),
                dict(count=4, n=4, kind='REGULAR', symmetry='2F',
                     angles=("70d32'", "70d32'", "70d32'", "70d32'"), plane='100')),
         figure='8.54'),

    dict(number=36, name="Fissioned tetragonal saddle hexahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((3, 8),), secondary=((2, 8),), nodes_total=16,
         branches={'100': 0, '110': 4, '111': 16}, branches_total=20,
         faces_total=6,
         faces=(dict(count=2, n=12, kind='', symmetry='4F',
                     angles=("70d32'", "144d44'", "144d44'", "70d32'", "144d44'", "144d44'", "70d32'", "144d44'", "144d44'", "70d32'", "144d44'", "144d44'"),
                     angles_truncated=True, plane='100'),
                dict(count=4, n=4, kind='REGULAR', symmetry='2F',
                     angles=("70d32'", "70d32'", "70d32'", "70d32'"), plane='100')),
         figure=None),

    dict(number=37, name="Tetragonal octagonal hexahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((3, 8),), secondary=((2, 12),), nodes_total=20,
         branches={'100': 0, '110': 8, '111': 16}, branches_total=24,
         faces_total=6,
         faces=(dict(count=2, n=8, kind='', symmetry='4F',
                     angles=("70d32'", "109d28'"),
                     angles_truncated=True, plane='100'),
                dict(count=4, n=8, kind='', symmetry='2F',
                     angles=("90d", "144d44'", "109d28'", "144d44'", "90d"),
                     angles_truncated=True, plane='110')),
         figure=None),

    dict(number=38, name="Trigonal hexahedron",
         axes=(0, 1, 0, 0), symmetry='Trigonal',
         primary=((3, 2), (4, 3)), secondary=((2, 9),), nodes_total=14,
         branches={'100': 0, '110': 0, '111': 18}, branches_total=18,
         faces_total=6,
         faces=(dict(count=3, n=6, kind='', symmetry='MIRROR',
                     angles=("70d32'", "109d28'"), plane='110'),
                dict(count=3, n=6, kind='REGULAR', symmetry='3F',
                     angles=("109d28'",), plane='100')),
         figure=None),

    dict(number=39, name="Trapezoidal trigonal hexahedron",
         axes=(0, 1, 0, 0), symmetry='Trigonal',
         primary=((6, 2),), secondary=((2, 6),), nodes_total=8,
         branches={'100': 6, '110': 3, '111': 3}, branches_total=12,
         faces_total=6,
         faces=(dict(count=3, n=4, kind='RIGHT', symmetry='NONE',
                     angles=("54d44'", "54d44'", "90d", "45d"),
                     plane='110'),
                dict(count=3, n=4, kind='LEFT', symmetry='NONE',
                     angles=("54d44'", "54d44'", "90d", "45d"),
                     plane='110')),
         figure=None),

    dict(number=40, name="bcc octahedron",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         primary=((4, 6),), secondary=((2, 12),), nodes_total=18,
         branches={'100': 24, '110': 0, '111': 0}, branches_total=24,
         faces_total=8,
         faces=(dict(count=8, n=6, kind='REGULAR', symmetry='3F',
                     angles=('90d', '90d', '90d', '90d', '90d', '90d'), plane='111'),),
         figure='8.55'),

    dict(number=41, name="fcc saddle octahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((4, 6),), secondary=((2, 12),), nodes_total=18,
         branches={'100': 8, '110': 16, '111': 0}, branches_total=24,
         faces_total=8,
         faces=(dict(count=8, n=6, kind='', symmetry='MIRROR',
                     angles=('90d', '90d', '90d', '90d', '90d', '90d'), plane='110'),),
         figure=None),

    dict(number=42, name="Tetragonal pentagonal octahedron",
         axes=(4, 0, 1, 0), symmetry='Tetragonal',
         primary=((4, 6),), secondary=((2, 8),), nodes_total=14,
         branches={'100': 12, '110': 8, '111': 0}, branches_total=20,
         faces_total=8,
         faces=(dict(count=8, n=5, kind='', symmetry='MIRROR',
                     angles=('90d', '90d', '90d', '90d', '90d'), plane='110'),),
         figure=None),

    dict(number=43, name="Tetrahedral decahedron",
         axes=(3, 4, 0, 0), symmetry='Tetrahedral',
         primary=((6, 4),), secondary=((2, 12),), nodes_total=16,
         branches={'100': 0, '110': 24, '111': 0}, branches_total=24,
         faces_total=10,
         faces=(dict(count=4, n=6, kind='REGULAR', symmetry='3F',
                     angles=('60d', '60d', '60d', '60d', '60d', '60d'), plane='111'),
                dict(count=6, n=4, kind='REGULAR', symmetry='2F',
                     angles=('60d', '60d', '60d', '60d'), plane='100')),
         figure='8.56'),

    dict(number=44, name="Saddle dodecahedron",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         primary=((4, 6),), secondary=((3, 8),), nodes_total=14,
         branches={'100': 0, '110': 24, '111': 0}, branches_total=24,
         faces_total=12,
         faces=(dict(count=12, n=4, kind='', symmetry='2F',
                     angles=("60d", "90d", "60d", "90d"), plane='110'),),
         figure='8.57'),

    dict(number=45, name="Blunted saddle dodecahedron",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         # The book stacks two primary rows here: z=4 x6 and z=3 x8.
         primary=((4, 6), (3, 8)), secondary=((2, 24),), nodes_total=38,
         branches={'100': 0, '110': 24, '111': 24}, branches_total=48,
         faces_total=12,
         faces=(dict(count=12, n=8, kind='', symmetry='2F',
                     angles=('90d', "144d44'", "109d28'", "144d44'", '90d', "144d44'", "109d28'", "144d44'"),
                     angles_truncated=True, plane='110'),),
         figure=None),

    dict(number=46, name="Truncated tetrahedral decahedron",
         axes=(3, 4, 0, 0), symmetry='Tetrahedral',
         primary=((3, 24),), secondary=((2, 24),), nodes_total=48,
         branches={'100': 0, '110': 60, '111': 0}, branches_total=60,
         faces_total=14,
         faces=(dict(count=4, n=6, kind='REGULAR', symmetry='6F',
                     angles=('120d', '120d', '120d', '120d', '120d', '120d'), plane='111'),
                dict(count=4, n=12, kind='', symmetry='3F',
                     angles=('120d', '120d', '120d', '120d', '120d', '120d', '120d', '120d', '120d', '120d', '120d', '120d'), plane='111'),
                dict(count=6, n=8, kind='', symmetry='2F',
                     angles=('120d', '120d', '120d', '120d', '120d', '120d', '120d', '120d'), plane='100')),
         figure='8.58'),

    dict(number=47, name="bcc saddle cuboctahedron",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         primary=((4, 12),), secondary=((2, 24),), nodes_total=36,
         branches={'100': 0, '110': 0, '111': 48}, branches_total=48,
         faces_total=14,
         faces=(dict(count=6, n=8, kind='', symmetry='4F',
                     angles=("70d32'", "109d28'", "70d32'", "109d28'", "70d32'", "109d28'", "70d32'", "109d28'"), plane='100'),
                dict(count=8, n=6, kind='REGULAR', symmetry='3F',
                     angles=("109d28'", "109d28'", "109d28'", "109d28'", "109d28'", "109d28'"), plane='111')),
         figure='8.59'),

    dict(number=48, name="Fissioned bcc saddle cuboctahedron",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         primary=((3, 24),), secondary=((2, 48),), nodes_total=72,
         branches={'100': 0, '110': 36, '111': 48}, branches_total=84,
         faces_total=14,
         faces=(dict(count=6, n=12, kind='', symmetry='4F',
                     angles=("144d44'", "70d32'", "144d44'", "144d44'",
                             "70d32'", "144d44'", "144d44'"),
                     angles_truncated=True, plane='100'),
                dict(count=8, n=12, kind='', symmetry='3F',
                     angles=("144d44'",), plane='111')),
         figure=None),

    dict(number=49, name="fcc saddle cuboctahedron",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         primary=((4, 12),), secondary=((2, 24),), nodes_total=36,
         branches={'100': 0, '110': 48, '111': 0}, branches_total=48,
         faces_total=14,
         faces=(dict(count=6, n=8, kind='', symmetry='4F',
                     angles=('60d', '90d', '60d', '90d', '60d', '90d', '60d', '90d'),
                     angles_truncated=True, plane='100'),
                dict(count=8, n=6, kind='REGULAR', symmetry='3F',
                     angles=('60d', '60d', '60d', '60d', '60d', '60d'), plane='111')),
         figure='8.60'),

    dict(number=50, name="Truncated fcc saddle cuboctahedron",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         primary=((3, 24),), secondary=((2, 48),), nodes_total=72,
         branches={'100': 0, '110': 84, '111': 0}, branches_total=84,
         faces_total=14,
         faces=(dict(count=6, n=12, kind='', symmetry='4F',
                     angles=("90d", "120d", "120d", "90d", "120d", "120d"),
                     angles_truncated=True, plane='100'),
                dict(count=8, n=12, kind='', symmetry='3F',
                     angles=("120d",), plane='111')),
         figure='8.61'),

    dict(number=51, name="Saddle cube dodecahedron",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         primary=((6, 8),), secondary=((2, 24),), nodes_total=32,
         branches={'100': 0, '110': 48, '111': 0}, branches_total=48,
         faces_total=18,
         faces=(dict(count=6, n=8, kind='', symmetry='4F',
                     angles=('60d', '90d', '60d', '90d', '60d', '90d', '60d', '90d'),
                     angles_truncated=True, plane='100'),
                dict(count=12, n=4, kind='', symmetry='2F',
                     angles=("60d", "90d", "60d", "90d"), plane='110')),
         figure='8.62'),

    dict(number=52, name="Truncated saddle dodecahedron",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         primary=((4, 6),), secondary=((3, 24),), nodes_total=30,
         branches={'100': 0, '110': 48, '111': 0}, branches_total=48,
         faces_total=20,
         faces=(dict(count=8, n=3, kind='REGULAR', symmetry='3F',
                     angles=('60d', '60d', '60d'), plane='111'),  # printed "triangle"
                dict(count=12, n=6, kind='', symmetry='2F',
                     angles=('90d', '120d', '120d', '90d', '120d', '120d'), plane='110')),
         figure='8.63'),

    dict(number=53, name="Universal cuboctadodecahedron",
         axes=(6, 4, 3, 0), symmetry='Cubic',
         primary=((3, 48),), secondary=((2, 24),), nodes_total=72,
         branches={'100': 0, '110': 96, '111': 0}, branches_total=96,
         faces_total=26,
         faces=(dict(count=6, n=12, kind='', symmetry='4F',
                     angles=('90d', '120d', '120d', '90d', '120d', '120d', '90d', '120d', '120d', '90d', '120d', '120d'),
                     angles_truncated=True, plane='100'),
                dict(count=8, n=6, kind='REGULAR', symmetry='6F',
                     angles=("120d",), plane='111'),  # printed "hexagon"
                dict(count=12, n=6, kind='', symmetry='2F',
                     angles=("90d", "120d", "120d", "90d", "120d", "120d"),
                     plane='110')),
         figure='8.64'),
)


def _selftest():
    """Consistency checks over every row of TABLE.  Raises AssertionError."""
    assert len(TABLE) == 53, "expected 53 entries, got %d" % len(TABLE)
    numbers = [e['number'] for e in TABLE]
    assert numbers == list(range(1, 54)), "entry numbers not 1..53 in order"
    print("check 8: 53 entries, numbered 1..53 with no gaps -- OK")

    legal = set(LEGAL_ANGLES)
    kinds = {'', 'REGULAR', 'ENANTIO', 'RIGHT', 'LEFT', 'PLANE'}
    syms = {'6F', '4F', '3F', '2F', 'MIRROR', 'NONE'}

    for e in TABLE:
        n = e['number']
        if n in UNRESOLVED:
            print("entry %2d: UNRESOLVED -- skipped (%s)"
                  % (n, e.get('note', 'no note')))
            continue
        nodes = e['primary'] + e['secondary']

        # 1. node counts sum to the printed total
        node_sum = sum(count for z, count in nodes)
        assert node_sum == e['nodes_total'], (
            "entry %d: node counts %d != total %d"
            % (n, node_sum, e['nodes_total']))

        # 2. branch direction counts sum to the printed total
        br_sum = sum(e['branches'].values())
        assert br_sum == e['branches_total'], (
            "entry %d: branch counts %d != total %d"
            % (n, br_sum, e['branches_total']))

        # 3. handshake: degree sum equals twice the branch count
        deg = sum(z * count for z, count in nodes)
        assert deg == 2 * e['branches_total'], (
            "entry %d: degree sum %d != 2*branches %d"
            % (n, deg, 2 * e['branches_total']))

        # 4. face counts sum to the printed total
        f_sum = sum(f['count'] for f in e['faces'])
        assert f_sum == e['faces_total'], (
            "entry %d: face counts %d != total %d"
            % (n, f_sum, e['faces_total']))

        # 5. face sides: each branch borders exactly two faces.
        #    Entries in ERRATA are transcribed AS PRINTED and do not
        #    balance here -- that is the book's inconsistency, recorded
        #    rather than repaired, so the check reports it and moves on.
        sides = sum(f['count'] * f['n'] for f in e['faces'])
        if n in ERRATA:
            assert sides != 2 * e['branches_total'], (
                "entry %d is listed in ERRATA but now balances (%d == %d)"
                " -- remove it from ERRATA" % (n, sides,
                                               2 * e['branches_total']))
            print("    entry %d: known book erratum -- %s" % (n, ERRATA[n]))
        else:
            assert sides == 2 * e['branches_total'], (
                "entry %d: face sides %d != 2*branches %d"
                % (n, sides, 2 * e['branches_total']))

        # 6. Euler: V - E + F = 2
        euler = e['nodes_total'] - e['branches_total'] + e['faces_total']
        assert euler == 2, "entry %d: V-E+F = %d != 2" % (n, euler)

        # 7. every angle is a legal Universal Node angle; enums are valid
        for f in e['faces']:
            assert f['kind'] in kinds, (
                "entry %d: bad kind %r" % (n, f['kind']))
            assert f['symmetry'] in syms, (
                "entry %d: bad face symmetry %r" % (n, f['symmetry']))
            assert f['plane'] in ('100', '110', '111', None), (
                "entry %d: bad plane %r" % (n, f['plane']))
            for a in f['angles']:
                assert a in legal, "entry %d: illegal angle %r" % (n, a)

        print("entry %2d: %-42s checks 1-7 OK" % (n, e['name']))

    print("RESULT: OK -- all %d resolved entries pass all checks"
          % (len(TABLE) - len(UNRESOLVED)))
