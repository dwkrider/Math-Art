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
        number=17, key='TRUNCATED_ORTHORHOMBIC_TETRAHEDRON',
        name='Truncated orthorhombic tetrahedron',
        net=(('110', 'FULL'),), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (8, 8, 4), (8, 4, 8), (4, 0, 8), (0, 0, 4), (4, 0, 0),
            (8, 4, 0), (-4, 4, 4), (-4, 8, 0), (0, 12, 0), (4, 12, 4),
            (0, 12, 8), (-4, 8, 8),
        ),
        faces=(
            (0, 1, 2, 3, 4, 5), (5, 4, 3, 6, 7, 8, 9, 0), (6, 3, 2, 1, 0, 9, 10, 11),
            (11, 10, 9, 8, 7, 6),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=19, key='WURTZITE_NODAL_TETRAHEDRON',
        name='Wurtzite nodal tetrahedron',
        net='WURTZITE', match='FULL',
        lattice='HEX',
        basis=((1.0, 0.0, 0.0), (-0.5, 0.8660254037844386, 0.0), (0.0, 0.0, 1.632993161855452)),
        divisor=24.0,
        verts=(
            (32, 16, 0), (32, 40, 0), (8, 40, 0), (-16, 16, 0), (-16, -8, 0),
            (8, -8, 0), (32, 16, 9), (8, 40, 9), (-16, -8, 9), (8, 16, 9),
        ),
        faces=(
            (5, 4, 3, 2, 1, 0), (0, 1, 2, 7, 9, 6), (2, 3, 4, 8, 9, 7),
            (4, 5, 0, 6, 9, 8),
        ),
        packs=False, has_unit=False, fills=False,
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
        packs=True, has_unit=True, fills=False,
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
        number=28, key='CUBICAL_SADDLE_HEXAHEDRON',
        name='Cubical saddle hexahedron',
        net='SHE-D', match='CORE',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (0, 0, 0), (0, 0, 4), (0, 4, 0), (0, 4, 4), (4, 0, 0),
            (4, 0, 4), (4, 4, 0), (4, 4, 4),
        ),
        faces=(
            (0, 1, 7, 3), (5, 7, 1, 0), (3, 7, 2, 0),
            (0, 2, 7, 6), (0, 4, 7, 5), (6, 7, 4, 0),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=29, key='SADDLE_CUBE',
        name='Saddle cube',
        net=(('111', 'FULL'),), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, 4, 12), (4, 12, 4), (4, 12, 20), (4, 20, 12), (8, 8, 8),
            (8, 8, 16), (8, 16, 8), (8, 16, 16), (12, 4, 4), (12, 4, 20),
            (12, 20, 4), (12, 20, 20), (16, 8, 8), (16, 8, 16), (16, 16, 8),
            (16, 16, 16), (20, 4, 12), (20, 12, 4), (20, 12, 20), (20, 20, 12),
        ),
        faces=(
            (5, 2, 7, 3, 6, 1, 4, 0), (0, 4, 8, 12, 16, 13, 9, 5), (6, 10, 14, 17, 12, 8, 4, 1),
            (2, 5, 9, 13, 18, 15, 11, 7), (7, 11, 15, 19, 14, 10, 6, 3), (17, 14, 19, 15, 18, 13, 16, 12),
        ),
        packs=False, has_unit=False, fills=False,
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
        number=31, key='UNIVERSAL_HEXAHEDRON',
        name='Universal hexahedron',
        net=(('110', 'FULL'),), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, -4, 8), (0, -4, 12), (-4, 0, 12), (-4, 4, 8), (-4, 0, 4),
            (0, -4, 4), (-4, 8, 4), (0, 12, 4), (4, 12, 8), (8, 12, 4),
            (12, 8, 4), (12, 4, 8), (12, 0, 4), (8, -4, 4), (8, -4, 12),
            (12, 0, 12), (12, 8, 12), (8, 12, 12), (0, 12, 12), (-4, 8, 12),
        ),
        faces=(
            (0, 1, 2, 3, 4, 5), (5, 4, 3, 6, 7, 8, 9, 10, 11, 12, 13, 0), (0, 14, 15, 11, 16, 17, 8, 18, 19, 3, 2, 1),
            (19, 18, 8, 7, 6, 3), (0, 13, 12, 11, 15, 14), (17, 16, 11, 10, 9, 8),
        ),
        packs=False, has_unit=False, fills=False,
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
        number=34, key='BIDODECAGONAL_HEXAHEDRON',
        name='Bidodecagonal hexahedron',
        net=(('110', 'FULL'), ('110', 'HALF')), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (6, 8, 24), (6, 12, 20), (6, 12, 28), (6, 16, 24), (8, 6, 24),
            (8, 18, 24), (12, 6, 20), (12, 6, 28), (12, 18, 20), (12, 18, 28),
            (16, 6, 24), (16, 18, 24), (18, 8, 24), (18, 12, 20), (18, 12, 28),
            (18, 16, 24),
        ),
        faces=(
            (2, 3, 1, 0), (0, 1, 3, 5, 8, 11, 15, 13, 12, 10, 6, 4), (4, 7, 10, 12, 14, 15, 11, 9, 5, 3, 2, 0),
            (4, 6, 10, 7), (9, 11, 8, 5), (12, 13, 15, 14),
        ),
        packs=False, has_unit=False, fills=False,
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
        number=37, key='TETRAGONAL_OCTAGONAL_HEXAHEDRON',
        name='Tetragonal octagonal hexahedron',
        net=(('110', 'FULL'), ('111', 'FULL')), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (0, 12, 24), (4, 12, 20), (4, 12, 28), (8, 8, 16), (8, 8, 32),
            (8, 16, 16), (8, 16, 32), (12, 0, 24), (12, 4, 20), (12, 4, 28),
            (12, 20, 20), (12, 20, 28), (12, 24, 24), (16, 8, 16), (16, 8, 32),
            (16, 16, 16), (16, 16, 32), (20, 12, 20), (20, 12, 28), (24, 12, 24),
        ),
        faces=(
            (0, 1, 3, 8, 7, 9, 4, 2), (2, 6, 11, 12, 10, 5, 1, 0), (5, 10, 15, 17, 13, 8, 3, 1),
            (2, 4, 9, 14, 18, 16, 11, 6), (7, 8, 13, 17, 19, 18, 14, 9), (10, 12, 11, 16, 18, 19, 17, 15),
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
        number=45, key='BLUNTED_SADDLE_DODECAHEDRON',
        name='Blunted saddle dodecahedron',
        net=(('110', 'FULL'), ('111', 'FULL')), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (12, 20, 20), (12, 20, 28), (12, 24, 24), (12, 28, 20), (12, 28, 28),
            (16, 16, 16), (16, 16, 32), (16, 32, 16), (16, 32, 32), (20, 12, 20),
            (20, 12, 28), (20, 20, 12), (20, 20, 36), (20, 28, 12), (20, 28, 36),
            (20, 36, 20), (20, 36, 28), (24, 12, 24), (24, 24, 12), (24, 24, 36),
            (24, 36, 24), (28, 12, 20), (28, 12, 28), (28, 20, 12), (28, 20, 36),
            (28, 28, 12), (28, 28, 36), (28, 36, 20), (28, 36, 28), (32, 16, 16),
            (32, 16, 32), (32, 32, 16), (32, 32, 32), (36, 20, 20), (36, 20, 28),
            (36, 24, 24), (36, 28, 20), (36, 28, 28),
        ),
        faces=(
            (5, 9, 17, 10, 6, 1, 2, 0), (0, 2, 3, 7, 13, 18, 11, 5), (6, 12, 19, 14, 8, 4, 2, 1),
            (4, 8, 16, 20, 15, 7, 3, 2), (11, 18, 23, 29, 21, 17, 9, 5), (6, 10, 17, 22, 30, 24, 19, 12),
            (15, 20, 27, 31, 25, 18, 13, 7), (8, 14, 19, 26, 32, 28, 20, 16), (17, 21, 29, 33, 35, 34, 30, 22),
            (25, 31, 36, 35, 33, 29, 23, 18), (19, 24, 30, 34, 35, 37, 32, 26), (28, 32, 37, 35, 36, 31, 27, 20),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=46, key='TRUNCATED_TETRAHEDRAL_DECAHEDRON',
        name='Truncated tetrahedral decahedron',
        net=(('110', 'FULL'),), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, 12, 20), (4, 16, 24), (4, 20, 12), (4, 24, 16), (8, 12, 16),
            (8, 16, 12), (8, 20, 24), (8, 24, 20), (12, 4, 20), (12, 8, 16),
            (12, 16, 8), (12, 16, 32), (12, 20, 4), (12, 20, 28), (12, 28, 20),
            (12, 32, 16), (16, 4, 24), (16, 8, 12), (16, 12, 8), (16, 12, 32),
            (16, 24, 4), (16, 24, 28), (16, 28, 24), (16, 32, 12), (20, 4, 12),
            (20, 8, 24), (20, 12, 4), (20, 12, 28), (20, 24, 8), (20, 24, 32),
            (20, 28, 12), (20, 32, 24), (24, 4, 16), (24, 8, 20), (24, 16, 4),
            (24, 16, 28), (24, 20, 8), (24, 20, 32), (24, 28, 16), (24, 32, 20),
            (28, 12, 20), (28, 16, 24), (28, 20, 12), (28, 24, 16), (32, 12, 16),
            (32, 16, 12), (32, 20, 24), (32, 24, 20),
        ),
        faces=(
            (0, 1, 6, 7, 3, 2, 5, 4), (4, 9, 8, 16, 25, 27, 19, 11, 13, 6, 1, 0), (2, 3, 7, 14, 15, 23, 30, 28, 20, 12, 10, 5),
            (4, 5, 10, 18, 17, 9), (13, 21, 22, 14, 7, 6), (8, 9, 17, 24, 32, 33, 25, 16),
            (10, 12, 20, 28, 36, 34, 26, 18), (19, 27, 35, 37, 29, 21, 13, 11), (22, 31, 39, 38, 30, 23, 15, 14),
            (17, 18, 26, 34, 36, 42, 45, 44, 40, 33, 32, 24), (29, 37, 35, 41, 46, 47, 43, 38, 39, 31, 22, 21), (33, 40, 41, 35, 27, 25),
            (28, 30, 38, 43, 42, 36), (44, 45, 42, 43, 47, 46, 41, 40),
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
        number=48, key='FISSIONED_BCC_SADDLE_CUBOCTAHEDRON',
        name='Fissioned bcc saddle cuboctahedron',
        net=(('110', 'FULL'), ('111', 'FULL')), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (0, 4, 8), (0, 4, 16), (0, 8, 4), (0, 8, 20), (0, 16, 4),
            (0, 16, 20), (0, 20, 8), (0, 20, 16), (4, 0, 8), (4, 0, 16),
            (4, 8, 0), (4, 8, 12), (4, 8, 24), (4, 12, 8), (4, 12, 16),
            (4, 16, 0), (4, 16, 12), (4, 16, 24), (4, 24, 8), (4, 24, 16),
            (8, 0, 4), (8, 0, 20), (8, 4, 0), (8, 4, 12), (8, 4, 24),
            (8, 12, 4), (8, 12, 20), (8, 20, 0), (8, 20, 12), (8, 20, 24),
            (8, 24, 4), (8, 24, 20), (12, 4, 8), (12, 4, 16), (12, 8, 4),
            (12, 8, 20), (12, 16, 4), (12, 16, 20), (12, 20, 8), (12, 20, 16),
            (16, 0, 4), (16, 0, 20), (16, 4, 0), (16, 4, 12), (16, 4, 24),
            (16, 12, 4), (16, 12, 20), (16, 20, 0), (16, 20, 12), (16, 20, 24),
            (16, 24, 4), (16, 24, 20), (20, 0, 8), (20, 0, 16), (20, 8, 0),
            (20, 8, 12), (20, 8, 24), (20, 12, 8), (20, 12, 16), (20, 16, 0),
            (20, 16, 12), (20, 16, 24), (20, 24, 8), (20, 24, 16), (24, 4, 8),
            (24, 4, 16), (24, 8, 4), (24, 8, 20), (24, 16, 4), (24, 16, 20),
            (24, 20, 8), (24, 20, 16),
        ),
        faces=(
            (11, 1, 3, 14, 5, 7, 16, 6, 4, 13, 2, 0), (0, 2, 13, 25, 10, 22, 34, 32, 20, 8, 23, 11), (11, 23, 9, 21, 33, 35, 24, 12, 26, 14, 3, 1),
            (4, 6, 16, 28, 18, 30, 38, 36, 27, 15, 25, 13), (14, 26, 17, 29, 37, 39, 31, 19, 28, 16, 7, 5), (8, 20, 32, 40, 52, 43, 53, 41, 33, 21, 9, 23),
            (25, 15, 27, 36, 47, 59, 45, 54, 42, 34, 22, 10), (12, 24, 35, 44, 56, 46, 61, 49, 37, 29, 17, 26), (18, 28, 19, 31, 39, 51, 63, 48, 62, 50, 38, 30),
            (32, 34, 42, 54, 45, 57, 66, 64, 55, 43, 52, 40), (41, 53, 43, 55, 65, 67, 58, 46, 56, 44, 35, 33), (36, 38, 50, 62, 48, 60, 70, 68, 57, 45, 59, 47),
            (49, 61, 46, 58, 69, 71, 60, 48, 63, 51, 39, 37), (55, 64, 66, 57, 68, 70, 60, 71, 69, 58, 67, 65),
        ),
        packs=False, has_unit=False, fills=False,
    ),
    dict(
        number=50, key='TRUNCATED_FCC_SADDLE_CUBOCTAHEDRON',
        name='Truncated fcc saddle cuboctahedron',
        net=(('110', 'FULL'), ('110', 'HALF')), match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (0, 6, 8), (0, 6, 16), (0, 8, 6), (0, 8, 18), (0, 16, 6),
            (0, 16, 18), (0, 18, 8), (0, 18, 16), (4, 6, 12), (4, 12, 6),
            (4, 12, 18), (4, 18, 12), (6, 0, 8), (6, 0, 16), (6, 4, 12),
            (6, 8, 0), (6, 8, 24), (6, 12, 4), (6, 12, 20), (6, 16, 0),
            (6, 16, 24), (6, 20, 12), (6, 24, 8), (6, 24, 16), (8, 0, 6),
            (8, 0, 18), (8, 6, 0), (8, 6, 24), (8, 18, 0), (8, 18, 24),
            (8, 24, 6), (8, 24, 18), (12, 4, 6), (12, 4, 18), (12, 6, 4),
            (12, 6, 20), (12, 18, 4), (12, 18, 20), (12, 20, 6), (12, 20, 18),
            (16, 0, 6), (16, 0, 18), (16, 6, 0), (16, 6, 24), (16, 18, 0),
            (16, 18, 24), (16, 24, 6), (16, 24, 18), (18, 0, 8), (18, 0, 16),
            (18, 4, 12), (18, 8, 0), (18, 8, 24), (18, 12, 4), (18, 12, 20),
            (18, 16, 0), (18, 16, 24), (18, 20, 12), (18, 24, 8), (18, 24, 16),
            (20, 6, 12), (20, 12, 6), (20, 12, 18), (20, 18, 12), (24, 6, 8),
            (24, 6, 16), (24, 8, 6), (24, 8, 18), (24, 16, 6), (24, 16, 18),
            (24, 18, 8), (24, 18, 16),
        ),
        faces=(
            (8, 1, 3, 10, 5, 7, 11, 6, 4, 9, 2, 0), (0, 2, 9, 17, 15, 26, 34, 32, 24, 12, 14, 8), (8, 14, 13, 25, 33, 35, 27, 16, 18, 10, 3, 1),
            (4, 6, 11, 21, 22, 30, 38, 36, 28, 19, 17, 9), (10, 18, 20, 29, 37, 39, 31, 23, 21, 11, 7, 5), (24, 32, 40, 48, 50, 49, 41, 33, 25, 13, 14, 12),
            (15, 17, 19, 28, 36, 44, 55, 53, 51, 42, 34, 26), (27, 35, 43, 52, 54, 56, 45, 37, 29, 20, 18, 16), (23, 31, 39, 47, 59, 57, 58, 46, 38, 30, 22, 21),
            (32, 34, 42, 51, 53, 61, 66, 64, 60, 50, 48, 40), (41, 49, 50, 60, 65, 67, 62, 54, 52, 43, 35, 33), (36, 38, 46, 58, 57, 63, 70, 68, 61, 53, 55, 44),
            (45, 56, 54, 62, 69, 71, 63, 57, 59, 47, 39, 37), (60, 64, 66, 61, 68, 70, 63, 71, 69, 62, 67, 65),
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
    dict(
        number=52, key='TRUNCATED_SADDLE_DODECAHEDRON',
        name='Truncated saddle dodecahedron',
        net='XAE', match='FULL',
        lattice='CUBIC8', basis=None, divisor=8.0,
        verts=(
            (4, 6, 6), (4, 6, 10), (4, 8, 8), (4, 10, 6), (4, 10, 10),
            (6, 4, 6), (6, 4, 10), (6, 6, 4), (6, 6, 12), (6, 10, 4),
            (6, 10, 12), (6, 12, 6), (6, 12, 10), (8, 4, 8), (8, 8, 4),
            (8, 8, 12), (8, 12, 8), (10, 4, 6), (10, 4, 10), (10, 6, 4),
            (10, 6, 12), (10, 10, 4), (10, 10, 12), (10, 12, 6), (10, 12, 10),
            (12, 6, 6), (12, 6, 10), (12, 8, 8), (12, 10, 6), (12, 10, 10),
        ),
        faces=(
            (5, 13, 6, 1, 2, 0), (0, 2, 3, 9, 14, 7), (7, 5, 0),
            (8, 15, 10, 4, 2, 1), (1, 6, 8), (4, 12, 16, 11, 3, 2),
            (11, 9, 3), (4, 10, 12), (5, 7, 14, 19, 17, 13),
            (13, 18, 20, 15, 8, 6), (9, 11, 16, 23, 21, 14), (15, 22, 24, 16, 12, 10),
            (13, 17, 25, 27, 26, 18), (21, 28, 27, 25, 19, 14), (15, 20, 26, 27, 29, 22),
            (24, 29, 27, 28, 23, 16), (17, 19, 25), (26, 20, 18),
            (21, 23, 28), (29, 24, 22),
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
    (18, 'Digonal hemisaddle tetrahedron', 'no geometry found by the search'),
    (20, 'bcc orthorhombic tetrahedron', 'no geometry found by the search'),
    (21, 'Rectangular orthorhombic tetrahedron', 'no geometry found by the search'),
    (22, 'Hemisaddle digonal disphenoid', 'no geometry found by the search'),
    (23, 'Double delta tetrahedron', 'no geometry found by the search'),
    (24, 'Trigonal pentahedron', 'no geometry found by the search'),
    (26, 'Digonal pentahedron', 'no geometry found by the search'),
    (36, 'Fissioned tetragonal saddle hexahedron', 'face angles ("70d32\'", "70d32\'", "70d32\'", "70d32\'", \'90d\', \'90d\', \'90d\', \'90d\', \'90d\', \'90d\', \'90d\', \'90d\') match no face of the row'),
    (38, 'Trigonal hexahedron', 'no geometry found by the search'),
    (39, 'Trapezoidal trigonal hexahedron', 'no geometry found by the search'),
    (41, 'fcc saddle octahedron', 'no geometry found by the search'),
    (42, 'Tetragonal pentagonal octahedron', 'no geometry found by the search'),
    (49, 'fcc saddle cuboctahedron', 'no geometry found by the search'),
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
    'SHE-D': (
        ((0, 0, 0), (0, 0, 4), (0, 4, 0), (0, 4, 4), (4, 0, 0),
         (4, 0, 4), (4, 4, 0), (4, 4, 4)),
        {   (0, 0, 0): (   (-4, -4, 0),
                           (-4, 0, -4),
                           (-4, 0, 0),
                           (-4, 0, 4),
                           (-4, 4, 0),
                           (0, -4, -4),
                           (0, -4, 0),
                           (0, -4, 4),
                           (0, 0, -4),
                           (0, 0, 4),
                           (0, 4, -4),
                           (0, 4, 0),
                           (0, 4, 4),
                           (4, -4, 0),
                           (4, 0, -4),
                           (4, 0, 0),
                           (4, 0, 4),
                           (4, 4, 0)),
            (0, 0, 4): (   (-4, -4, 0),
                           (-4, 4, 0),
                           (0, 0, -4),
                           (0, 0, 4),
                           (4, -4, 0),
                           (4, 4, 0)),
            (0, 4, 0): (   (-4, 0, -4),
                           (-4, 0, 4),
                           (0, -4, 0),
                           (0, 4, 0),
                           (4, 0, -4),
                           (4, 0, 4)),
            (0, 4, 4): (   (-4, 0, 0),
                           (0, -4, -4),
                           (0, -4, 4),
                           (0, 4, -4),
                           (0, 4, 4),
                           (4, 0, 0)),
            (4, 0, 0): (   (-4, 0, 0),
                           (0, -4, -4),
                           (0, -4, 4),
                           (0, 4, -4),
                           (0, 4, 4),
                           (4, 0, 0)),
            (4, 0, 4): (   (-4, 0, -4),
                           (-4, 0, 4),
                           (0, -4, 0),
                           (0, 4, 0),
                           (4, 0, -4),
                           (4, 0, 4)),
            (4, 4, 0): (   (-4, -4, 0),
                           (-4, 4, 0),
                           (0, 0, -4),
                           (0, 0, 4),
                           (4, -4, 0),
                           (4, 4, 0)),
            (4, 4, 4): (   (-4, -4, 0),
                           (-4, 0, -4),
                           (-4, 0, 0),
                           (-4, 0, 4),
                           (-4, 4, 0),
                           (0, -4, -4),
                           (0, -4, 0),
                           (0, -4, 4),
                           (0, 0, -4),
                           (0, 0, 4),
                           (0, 4, -4),
                           (0, 4, 0),
                           (0, 4, 4),
                           (4, -4, 0),
                           (4, 0, -4),
                           (4, 0, 0),
                           (4, 0, 4),
                           (4, 4, 0))},
    ),
    'XAE': (
        ((0, 0, 4), (0, 4, 0), (2, 2, 4), (2, 4, 2), (2, 4, 6),
         (2, 6, 4), (4, 0, 0), (4, 2, 2), (4, 2, 6), (4, 6, 2),
         (4, 6, 6), (6, 2, 4), (6, 4, 2), (6, 4, 6), (6, 6, 4)),
        ((-2, -2, 0), (-2, 0, -2), (-2, 0, 2), (-2, 2, 0), (0, -2, -2),
         (0, -2, 2), (0, 2, -2), (0, 2, 2), (2, -2, 0), (2, 0, -2),
         (2, 0, 2), (2, 2, 0)),
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
        lat = 'CUBIC8' if s.get('basis') is None else s['lattice']
        tag = "#%d %s" % (s['number'], s['name'])
        # 1. exact integer closure of every circuit
        chk("%s: circuits close (exact)" % tag,
            all(pnet.closes([V[i] for i in f]) for f in F))
        # 2. every edge is a branch of the solid's OWN lattice -- the
        #    cubic Universal Node classes, or the hexagonal system of
        #    pearce_net section 6b.  No lattice is exempt.
        try:
            bt = pnet.branch_totals(V, F, lattice=lat)
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
            bt == dict(r['branches']),
            "%r vs %r" % (bt, dict(r['branches'])))
        # 4. face inventory: size, own symmetry, plane direction --
        #    the plane checked in the solid's own lattice frame
        got = {}
        for cyc in F:
            loop = [X[i] for i in cyc]
            k = (len(cyc), pnet.face_symmetry_label(loop),
                 pnet.face_plane_class(loop, lattice=lat))
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
        # 6. symmetry axes, in the solid's own lattice frame -- no
        #    longer skipped for hexagonal solids
        pts = [X[i] for i in range(len(X))]
        ax = pnet.axis_counts(pts, lattice=lat)
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
        chk("%s: orientable" % tag,
            pnet.orientation_consistent(pnet.orient_faces(X, F)))

    print("RESULT:", "OK" if ok else "BAD")
    if not ok:
        raise AssertionError("pearce_data self-test failed")
