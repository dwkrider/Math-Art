# Menger-type sponges and Sierpinski solids by recursive subdivision.
#
# Part of the Math Art IFS engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Each of these is a digit rule: subdivide the cube into 3^3 (or 2^3)
# subcells and keep those whose index vector satisfies a predicate.  The
# Menger sponge keeps cells with at most one coordinate equal to 1 (so
# the centre of a face, and the body centre, are removed); the Vicsek
# cross keeps the opposite.  Writing the rule as a predicate on the digit
# vector, rather than as a hand-listed cell set, is what makes the whole
# family one function.
#
# The seed solids come from `polyhedra.seeds`: the tetrahedron and
# octahedron here are the same ones the rest of the project uses, and
# were three verbatim copies of one table before.
#
# References:
# - K. Menger, "Allgemeine Raume und Cartesische Raume II", Proceedings
#   of the Amsterdam Academy 29, 1926, pp. 1125-1128.
# - W. Sierpinski, "Sur une courbe cantorienne dont tout point est un
#   point de ramification", Comptes Rendus 160, 1915, pp. 302-305.
# - T. Vicsek, "Fractal models for diffusion controlled aggregation",
#   Journal of Physics A 16, 1983, L647.

import numpy as np

from .voxel import _FACE_DIRS

try:
    from ..polyhedra.seeds import seed_poly
except ImportError:  # flat import outside the package
    from polyhedra.seeds import seed_poly

# The seed solids are the project's canonical ones, from
# `polyhedra.seeds` -- this module carried a verbatim second copy of the
# same vertices and winding.
_TETRA, _TETRA_F = seed_poly("TETRA")
_OCTA, _OCTA_F = seed_poly("OCTA")

def _grid_rule(kind):
    """Which of the 3x3x3 (or 3x3x1) subcells survive one step."""
    keep = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                mid = (x == 1, y == 1, z == 1)
                if kind == 'MENGER':
                    ok = sum(mid) < 2
                elif kind == 'VICSEK':
                    ok = sum(mid) >= 2
                elif kind == 'CARPET':
                    ok = z == 0 and not (mid[0] and mid[1])
                elif kind == 'MOSELY':
                    ok = sum(mid) >= 1          # keep all but 8 corners
                elif kind == 'MOSELYL':
                    ok = 1 <= sum(mid) <= 2     # also drop body centre
                elif kind == 'CANTOR':
                    ok = sum(mid) == 0          # keep only the corners
                else:
                    raise ValueError(kind)
                if ok:
                    keep.append((x, y, z))
    return keep


def sponge_cells(kind, level):
    """Occupied unit cells of the level-n grid sponge, as integer
    coordinates in a 3^n grid."""
    rule = _grid_rule(kind)
    cells = [(0, 0, 0)]
    for _ in range(level):
        cells = [(3 * cx + dx, 3 * cy + dy, 3 * cz + dz)
                 for (cx, cy, cz) in cells
                 for (dx, dy, dz) in rule]
    return cells


def build_grid_sponge(kind, level, size=2.0):
    """Watertight exterior surface of a grid sponge: only faces
    between a solid cell and an empty neighbour are emitted, with
    shared vertices."""
    cells = sponge_cells(kind, level)
    occ = set(cells)
    n = 3 ** level
    s = size / n
    off = size / 2.0
    zoff = (size / 2.0 if kind != 'CARPET'
            else s / 2.0)          # carpet: one cell thick, centred
    verts = []
    vid = {}
    faces = []

    def vertex(ix, iy, iz):
        key = (ix, iy, iz)
        i = vid.get(key)
        if i is None:
            i = len(verts)
            vid[key] = i
            verts.append((ix * s - off, iy * s - off, iz * s - zoff))
        return i

    for (cx, cy, cz) in cells:
        for (d, corners) in _FACE_DIRS:
            if (cx + d[0], cy + d[1], cz + d[2]) in occ:
                continue
            faces.append([vertex(cx + a, cy + b, cz + c)
                          for (a, b, c) in corners])
    return verts, faces










def build_corner_sponge(kind, level, size=2.0):
    """Sierpinski tetrahedron / octahedron: recursive half-scale
    copies at the vertices, meeting at points. One closed solid per
    smallest copy."""
    base_v, base_f = ((_TETRA, _TETRA_F) if kind == 'TETRA'
                      else (_OCTA, _OCTA_F))
    centres = [(0.0, 0.0, 0.0)]
    sc = size / 2.0
    for _ in range(level):
        sc /= 2.0
        centres = [(c[0] + v[0] * sc, c[1] + v[1] * sc,
                    c[2] + v[2] * sc)
                   for c in centres for v in base_v]
    verts = []
    faces = []
    for c in centres:
        base = len(verts)
        for v in base_v:
            verts.append((c[0] + v[0] * sc, c[1] + v[1] * sc,
                          c[2] + v[2] * sc))
        for f in base_f:
            faces.append([base + i for i in f])
    return verts, faces


GRID_KINDS = ('MENGER', 'VICSEK', 'CARPET', 'MOSELY', 'MOSELYL',
              'CANTOR')


MAX_LEVEL = {'MENGER': 4, 'VICSEK': 5, 'CARPET': 6,
             'MOSELY': 3, 'MOSELYL': 3, 'CANTOR': 4,
             'TETRA': 6, 'OCTA': 5}


def build_sponge(kind='MENGER', level=3, size=2.0):
    level = min(level, MAX_LEVEL[kind])
    if kind in GRID_KINDS:
        return build_grid_sponge(kind, level, size)
    return build_corner_sponge(kind, level, size)
