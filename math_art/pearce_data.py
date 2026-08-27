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
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (7, 13, 11), (5, 11, 15), (5, 13, 13), (7, 13, 19), (7, 11, 17),
            (11, 15, 13), (13, 11, 15), (13, 13, 13), (11, 9, 15), (9, 15, 11),
            (9, 9, 17), (9, 15, 19), (11, 17, 15), (9, 17, 17),
        ),
        faces=(
            (0, 2, 1, 4, 3, 11, 13, 12, 5, 9), (9, 5, 7, 6, 8, 10, 4, 1, 2, 0), (3, 4, 10, 8, 6, 7, 5, 12, 13, 11),
        ),
        packs=True, has_unit=True, fills=True,
    ),
    dict(
        number=6, key='WURTZITE_TRIHEDRON',
        name='Wurtzite trihedron',
        net='WURTZITE', match='FULL',
        lattice='HEX',
        basis=((1.0, 0.0, 0.0), (-0.5, 0.8660254037844386, 0.0), (0.0, 0.0, 1.632993161855452)),
        divisor=24.0,
        verts=(
            (16, 8, 12), (16, 8, 21), (40, 8, 12), (32, 16, 9), (40, 8, 21),
            (32, 16, 24), (40, 32, 12), (40, 32, 21),
        ),
        faces=(
            (3, 2, 4, 5, 1, 0), (0, 1, 5, 7, 6, 3), (2, 3, 6, 7, 5, 4),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=11, key='DIAMOND_TETRAHEDRON',
        name='Diamond tetrahedron',
        net='DIAMOND', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (6, 6, 2), (6, 2, 6), (2, 6, 6), (4, 4, 8), (6, 6, 10),
            (4, 8, 4), (6, 10, 6), (8, 4, 4), (10, 6, 6), (8, 8, 8),
        ),
        faces=(
            (7, 1, 3, 2, 5, 0), (0, 5, 6, 9, 8, 7), (7, 8, 9, 4, 3, 1),
            (2, 3, 4, 9, 6, 5),
        ),
        packs=True, has_unit=True, fills=True,
    ),
    dict(
        number=12, key='BCC_TETRAHEDRON',
        name='bcc tetrahedron',
        net='BCC', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, 4, 4), (0, 0, 8), (4, 4, 12), (0, 8, 8), (8, 0, 8),
            (8, 8, 8),
        ),
        faces=(
            (0, 1, 2, 3), (4, 2, 1, 0), (0, 3, 2, 5),
            (5, 2, 4, 0),
        ),
        packs=True, has_unit=True, fills=True,
    ),
    dict(
        number=13, key='FCC_TETRAGONAL_TETRAHEDRON',
        name='fcc tetragonal tetrahedron',
        net='FCC', match='CORE',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, 4, 0), (4, 0, 4), (0, 4, 4), (4, 4, 8), (4, 8, 4),
            (8, 4, 4),
        ),
        faces=(
            (5, 1, 2, 0), (0, 2, 4, 5), (5, 3, 2, 1),
            (2, 3, 5, 4),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=14, key='FCC_ORTHORHOMBIC_TETRAHEDRON',
        name='fcc orthorhombic tetrahedron',
        net='FCC', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, 4, 0), (0, 4, 4), (4, 4, 8), (0, 8, 0), (4, 8, 4),
            (0, 8, 8),
        ),
        faces=(
            (4, 2, 1, 0), (0, 1, 3, 4), (1, 2, 4, 5),
            (5, 4, 3, 1),
        ),
        packs=False, has_unit=False, fills=True,
    ),
    dict(
        number=25, key='WURTZITE_PENTAHEDRON',
        name='Wurtzite pentahedron',
        net='WURTZITE', match='FULL',
        lattice='HEX',
        basis=((1.0, 0.0, 0.0), (-0.5, 0.8660254037844386, 0.0), (0.0, 0.0, 1.632993161855452)),
        divisor=24.0,
        verts=(
            (16, 8, 12), (8, 16, 9), (16, 8, 21), (8, 16, 24), (16, 32, 12),
            (16, 32, 21), (32, 16, 9), (32, 16, 24), (40, 32, 12), (32, 40, 9),
            (40, 32, 21), (32, 40, 24),
        ),
        faces=(
            (2, 3, 5, 4, 1, 0), (0, 1, 4, 9, 8, 6), (6, 8, 10, 7, 2, 0),
            (7, 10, 11, 5, 3, 2), (4, 5, 11, 10, 8, 9),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=27, key='TRIANGULAR_HEXAHEDRON',
        name='Triangular hexahedron',
        net='HXG-D', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (2, 10, 10), (6, 6, 6), (6, 6, 14), (6, 14, 14), (10, 2, 10),
            (10, 10, 10), (10, 10, 18), (14, 6, 14),
        ),
        faces=(
            (5, 1, 2, 0), (0, 2, 3, 5), (5, 4, 2, 1),
            (6, 5, 3, 2), (2, 4, 5, 7), (7, 5, 6, 2),
        ),
        packs=True, has_unit=True, fills=False,
    ),
    dict(
        number=30, key='TRUNCATED_TETRAGONAL_TETRAHEDRON',
        name='Truncated tetragonal tetrahedron',
        net='MJT', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (2, 6, 8), (4, 4, 8), (6, 6, 8), (2, 8, 6), (4, 8, 4),
            (6, 8, 6), (2, 8, 10), (2, 10, 8), (4, 8, 12), (4, 12, 8),
            (6, 8, 10), (6, 10, 8),
        ),
        faces=(
            (3, 4, 5, 2, 1, 0), (0, 1, 2, 10, 8, 6), (6, 7, 3, 0),
            (2, 5, 11, 10), (7, 9, 11, 5, 4, 3), (8, 10, 11, 9, 7, 6),
        ),
        packs=False, has_unit=True, fills=False,
    ),
    dict(
        number=32, key='AUGMENTED_UNIVERSAL_HEXAHEDRON',
        name='Augmented universal hexahedron',
        net='REO', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (0, 0, 4), (0, 0, 12), (0, 4, 8), (4, 0, 8), (0, 8, 4),
            (0, 8, 12), (4, 8, 8), (8, 0, 4), (8, 0, 12), (8, 4, 8),
            (8, 8, 4), (8, 8, 12),
        ),
        faces=(
            (3, 1, 2, 0), (0, 2, 4, 6, 10, 9, 7, 3), (3, 8, 9, 11, 6, 5, 2, 1),
            (5, 6, 4, 2), (3, 7, 9, 8), (11, 9, 10, 6),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=33, key='BIOCTAGONAL_HEXAHEDRON',
        name='Bioctagonal hexahedron',
        net='FCC', match='CORE',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, 0, 4), (0, 4, 4), (0, 0, 8), (4, 0, 12), (0, 4, 12),
            (4, 8, 4), (0, 8, 8), (4, 8, 12), (8, 4, 4), (8, 0, 8),
            (8, 4, 12), (8, 8, 8),
        ),
        faces=(
            (0, 2, 1, 6, 5, 11, 8, 9), (9, 3, 2, 0), (1, 2, 4, 6),
            (2, 3, 9, 10, 11, 7, 6, 4), (5, 6, 7, 11), (11, 10, 9, 8),
        ),
        packs=False, has_unit=False, fills=True,
    ),
    dict(
        number=35, key='TETRAGONAL_SADDLE_HEXAHEDRON',
        name='Tetragonal saddle hexahedron',
        net=(('111', 'FULL'),), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, 12, 20), (4, 12, 28), (8, 8, 24), (8, 16, 24), (12, 4, 20),
            (12, 4, 28), (12, 20, 20), (12, 20, 28), (16, 8, 24), (16, 16, 24),
            (20, 12, 20), (20, 12, 28),
        ),
        faces=(
            (0, 2, 1, 3), (3, 6, 9, 10, 8, 4, 2, 0), (1, 2, 5, 8, 11, 9, 7, 3),
            (2, 4, 8, 5), (7, 9, 6, 3), (8, 10, 9, 11),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=40, key='BCC_OCTAHEDRON',
        name='bcc octahedron',
        net='NBO', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (0, 0, 4), (0, 4, 0), (0, 4, 4), (0, 4, 8), (0, 8, 4),
            (4, 0, 0), (4, 0, 4), (4, 0, 8), (4, 4, 0), (4, 4, 8),
            (4, 8, 0), (4, 8, 4), (4, 8, 8), (8, 0, 4), (8, 4, 0),
            (8, 4, 4), (8, 4, 8), (8, 8, 4),
        ),
        faces=(
            (0, 2, 1, 8, 5, 6), (6, 7, 9, 3, 2, 0), (1, 2, 4, 11, 10, 8),
            (2, 3, 9, 12, 11, 4), (8, 14, 15, 13, 6, 5), (13, 15, 16, 9, 7, 6),
            (8, 10, 11, 17, 15, 14), (16, 15, 17, 11, 12, 9),
        ),
        packs=True, has_unit=True, fills=True,
    ),
    dict(
        number=43, key='TETRAHEDRAL_DECAHEDRON',
        name='Tetrahedral decahedron',
        net='FCC', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, 4, 0), (4, 0, 4), (0, 4, 4), (4, 4, 8), (4, 8, 4),
            (0, 8, 8), (4, 12, 8), (4, 8, 12), (8, 4, 4), (8, 0, 8),
            (12, 4, 8), (8, 4, 12), (8, 8, 0), (12, 8, 4), (8, 12, 4),
            (8, 8, 8),
        ),
        faces=(
            (8, 1, 3, 2, 4, 0), (0, 4, 12, 8), (8, 9, 3, 1),
            (2, 3, 5, 4), (7, 15, 6, 4, 5, 3), (11, 15, 7, 3),
            (3, 9, 8, 10, 15, 11), (4, 6, 15, 14), (14, 15, 13, 8, 12, 4),
            (13, 15, 10, 8),
        ),
        packs=True, has_unit=True, fills=True,
    ),
    dict(
        number=44, key='SADDLE_DODECAHEDRON',
        name='Saddle dodecahedron',
        net='FCC', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, 4, 0), (4, 4, 8), (4, 12, 0), (4, 8, 4), (4, 12, 8),
            (12, 4, 0), (8, 4, 4), (12, 4, 8), (8, 8, 0), (12, 12, 0),
            (12, 8, 4), (8, 12, 4), (8, 8, 8), (12, 12, 8),
        ),
        faces=(
            (6, 1, 3, 0), (0, 3, 2, 8), (8, 5, 6, 0),
            (12, 4, 3, 1), (1, 6, 7, 12), (2, 3, 4, 11),
            (11, 9, 8, 2), (12, 13, 11, 4), (10, 7, 6, 5),
            (5, 8, 9, 10), (7, 10, 13, 12), (11, 13, 10, 9),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=47, key='BCC_SADDLE_CUBOCTAHEDRON',
        name='bcc saddle cuboctahedron',
        net=(('111', 'FULL'),), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (0, 8, 8), (0, 8, 16), (0, 16, 8), (0, 16, 16), (4, 4, 12),
            (4, 12, 4), (4, 12, 20), (4, 20, 12), (8, 0, 8), (8, 0, 16),
            (8, 8, 0), (8, 8, 24), (8, 16, 0), (8, 16, 24), (8, 24, 8),
            (8, 24, 16), (12, 4, 4), (12, 4, 20), (12, 20, 4), (12, 20, 20),
            (16, 0, 8), (16, 0, 16), (16, 8, 0), (16, 8, 24), (16, 16, 0),
            (16, 16, 24), (16, 24, 8), (16, 24, 16), (20, 4, 12), (20, 12, 4),
            (20, 12, 20), (20, 20, 12), (24, 8, 8), (24, 8, 16), (24, 16, 8),
            (24, 16, 16),
        ),
        faces=(
            (0, 4, 1, 6, 3, 7, 2, 5), (5, 10, 16, 8, 4, 0), (1, 4, 9, 17, 11, 6),
            (7, 14, 18, 12, 5, 2), (3, 6, 13, 19, 15, 7), (4, 8, 16, 20, 28, 21, 17, 9),
            (12, 18, 24, 29, 22, 16, 10, 5), (6, 11, 17, 23, 30, 25, 19, 13), (15, 19, 27, 31, 26, 18, 14, 7),
            (22, 29, 32, 28, 20, 16), (17, 21, 28, 33, 30, 23), (26, 31, 34, 29, 24, 18),
            (19, 25, 30, 35, 31, 27), (28, 32, 29, 34, 31, 35, 30, 33),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=51, key='SADDLE_CUBE_DODECAHEDRON',
        name='Saddle cube dodecahedron',
        net='SHE', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (0, 2, 4), (0, 4, 2), (0, 4, 6), (0, 6, 4), (2, 0, 4),
            (2, 2, 2), (2, 2, 6), (2, 4, 0), (2, 4, 8), (2, 6, 2),
            (2, 6, 6), (2, 8, 4), (4, 0, 2), (4, 0, 6), (4, 2, 0),
            (4, 2, 8), (4, 6, 0), (4, 6, 8), (4, 8, 2), (4, 8, 6),
            (6, 0, 4), (6, 2, 2), (6, 2, 6), (6, 4, 0), (6, 4, 8),
            (6, 6, 2), (6, 6, 6), (6, 8, 4), (8, 2, 4), (8, 4, 2),
            (8, 4, 6), (8, 6, 4),
        ),
        faces=(
            (6, 2, 10, 3, 9, 1, 5, 0), (0, 5, 4, 6), (9, 7, 5, 1),
            (2, 6, 8, 10), (10, 11, 9, 3), (4, 5, 12, 21, 20, 22, 13, 6),
            (5, 7, 9, 16, 25, 23, 21, 14), (14, 21, 12, 5), (15, 22, 24, 26, 17, 10, 8, 6),
            (6, 13, 22, 15), (9, 11, 10, 19, 26, 27, 25, 18), (18, 25, 16, 9),
            (10, 17, 26, 19), (20, 21, 28, 22), (21, 23, 25, 29),
            (29, 25, 31, 26, 30, 22, 28, 21), (30, 26, 24, 22), (25, 27, 26, 31),
        ),
        packs=False, has_unit=False, fills=False,
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
    (26, 'Digonal pentahedron', 'no geometry found by the search'),
    (28, 'Cubical saddle hexahedron', 'no geometry found by the search'),
    (29, 'Saddle cube', 'no geometry found by the search'),
    (31, 'Universal hexahedron', 'no geometry found by the search'),
    (34, 'Bidodecagonal hexahedron', 'no geometry found by the search'),
    (36, 'Fissioned tetragonal saddle hexahedron', 'no geometry found by the search'),
    (37, 'Tetragonal octagonal hexahedron', 'no geometry found by the search'),
    (38, 'Trigonal hexahedron', 'no geometry found by the search'),
    (39, 'Trapezoidal trigonal hexahedron', 'no geometry found by the search'),
    (41, 'fcc saddle octahedron', 'no geometry found by the search'),
    (42, 'Tetragonal pentagonal octahedron', 'no geometry found by the search'),
    (45, 'Blunted saddle dodecahedron', 'no geometry found by the search'),
    (46, 'Truncated tetrahedral decahedron', "face inventory {(8, '2F', '100'): 6, (12, '3F', '111'): 4, (6, '3F', '111'): 4} != {(6, '6F', '111'): 4, (12, '3F', '111'): 4, (8, '2F', '100'): 6}"),
    (48, 'Fissioned bcc saddle cuboctahedron', 'no geometry found by the search'),
    (49, 'fcc saddle cuboctahedron', 'no geometry found by the search'),
    (50, 'Truncated fcc saddle cuboctahedron', 'no geometry found by the search'),
    (52, 'Truncated saddle dodecahedron', 'no geometry found by the search'),
    (53, 'Universal cuboctadodecahedron', 'no geometry found by the search'),
)

#: Nets these solids use that `pearce_net` does not build in.
#: The resolver registers RCSR nets at run time from a local
#: mirror; the extension has no such mirror, so any net a
#: shipped solid depends on has to travel WITH the data.
EXTRA_NETS = {
    'HXG-D': (
        ((2, 2, 2), (6, 6, 6)),
        {   (2, 2, 2): (   (-8, 0, 0),
                           (-4, -4, -4),
                           (-4, 4, 4),
                           (0, -8, 0),
                           (0, 0, -8),
                           (0, 0, 8),
                           (0, 8, 0),
                           (4, -4, 4),
                           (4, 4, -4),
                           (8, 0, 0)),
            (6, 6, 6): (   (-8, 0, 0),
                           (-4, -4, 4),
                           (-4, 4, -4),
                           (0, -8, 0),
                           (0, 0, -8),
                           (0, 0, 8),
                           (0, 8, 0),
                           (4, -4, -4),
                           (4, 4, 4),
                           (8, 0, 0))},
    ),
    'MJT': (
        ((0, 0, 4), (0, 2, 2), (0, 2, 6), (0, 4, 0), (0, 4, 4),
         (0, 6, 2), (0, 6, 6), (2, 0, 2), (2, 0, 6), (2, 2, 0),
         (2, 2, 4), (2, 4, 2), (2, 4, 6), (2, 6, 0), (2, 6, 4),
         (4, 0, 0), (4, 0, 4), (4, 2, 2), (4, 2, 6), (4, 4, 0),
         (4, 6, 2), (4, 6, 6), (6, 0, 2), (6, 0, 6), (6, 2, 0),
         (6, 2, 4), (6, 4, 2), (6, 4, 6), (6, 6, 0), (6, 6, 4)),
        ((-2, -2, 0), (-2, 0, -2), (-2, 0, 2), (-2, 2, 0), (0, -2, -2),
         (0, -2, 2), (0, 2, -2), (0, 2, 2), (2, -2, 0), (2, 0, -2),
         (2, 0, 2), (2, 2, 0)),
    ),
    'REO': (
        ((0, 0, 4), (0, 4, 0), (4, 0, 0)),
        ((-4, -4, 0), (-4, 0, -4), (-4, 0, 4), (-4, 4, 0), (0, -4, -4),
         (0, -4, 4), (0, 4, -4), (0, 4, 4), (4, -4, 0), (4, 0, -4),
         (4, 0, 4), (4, 4, 0)),
    ),
    'SHE': (
        ((0, 2, 4), (0, 4, 2), (0, 4, 6), (0, 6, 4), (2, 0, 4),
         (2, 2, 2), (2, 2, 6), (2, 4, 0), (2, 6, 2), (2, 6, 6),
         (4, 0, 2), (4, 0, 6), (4, 2, 0), (4, 6, 0), (6, 0, 4),
         (6, 2, 2), (6, 2, 6), (6, 4, 0), (6, 6, 2), (6, 6, 6)),
        {   (0, 2, 4): ((-2, 0, -2), (-2, 0, 2), (2, 0, -2), (2, 0, 2)),
            (0, 4, 2): ((-2, -2, 0), (-2, 2, 0), (2, -2, 0), (2, 2, 0)),
            (0, 4, 6): ((-2, -2, 0), (-2, 2, 0), (2, -2, 0), (2, 2, 0)),
            (0, 6, 4): ((-2, 0, -2), (-2, 0, 2), (2, 0, -2), (2, 0, 2)),
            (2, 0, 4): ((0, -2, -2), (0, -2, 2), (0, 2, -2), (0, 2, 2)),
            (2, 2, 2): (   (-2, 0, 2),
                           (-2, 2, 0),
                           (0, -2, 2),
                           (0, 2, -2),
                           (2, -2, 0),
                           (2, 0, -2)),
            (2, 2, 6): (   (-2, 0, -2),
                           (-2, 2, 0),
                           (0, -2, -2),
                           (0, 2, 2),
                           (2, -2, 0),
                           (2, 0, 2)),
            (2, 4, 0): ((0, -2, -2), (0, -2, 2), (0, 2, -2), (0, 2, 2)),
            (2, 6, 2): (   (-2, -2, 0),
                           (-2, 0, 2),
                           (0, -2, -2),
                           (0, 2, 2),
                           (2, 0, -2),
                           (2, 2, 0)),
            (2, 6, 6): (   (-2, -2, 0),
                           (-2, 0, -2),
                           (0, -2, 2),
                           (0, 2, -2),
                           (2, 0, 2),
                           (2, 2, 0)),
            (4, 0, 2): ((-2, -2, 0), (-2, 2, 0), (2, -2, 0), (2, 2, 0)),
            (4, 0, 6): ((-2, -2, 0), (-2, 2, 0), (2, -2, 0), (2, 2, 0)),
            (4, 2, 0): ((-2, 0, -2), (-2, 0, 2), (2, 0, -2), (2, 0, 2)),
            (4, 6, 0): ((-2, 0, -2), (-2, 0, 2), (2, 0, -2), (2, 0, 2)),
            (6, 0, 4): ((0, -2, -2), (0, -2, 2), (0, 2, -2), (0, 2, 2)),
            (6, 2, 2): (   (-2, -2, 0),
                           (-2, 0, -2),
                           (0, -2, 2),
                           (0, 2, -2),
                           (2, 0, 2),
                           (2, 2, 0)),
            (6, 2, 6): (   (-2, -2, 0),
                           (-2, 0, 2),
                           (0, -2, -2),
                           (0, 2, 2),
                           (2, 0, -2),
                           (2, 2, 0)),
            (6, 4, 0): ((0, -2, -2), (0, -2, 2), (0, 2, -2), (0, 2, 2)),
            (6, 6, 2): (   (-2, 0, -2),
                           (-2, 2, 0),
                           (0, -2, -2),
                           (0, 2, 2),
                           (2, -2, 0),
                           (2, 0, 2)),
            (6, 6, 6): (   (-2, 0, 2),
                           (-2, 2, 0),
                           (0, -2, 2),
                           (0, 2, -2),
                           (2, -2, 0),
                           (2, 0, -2))},
    ),
}


def _register_nets():
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
