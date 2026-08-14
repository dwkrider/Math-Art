# Space-filling cell complexes: honeycombs and their spiral blocks.
#
# Part of the Math Art IFS engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Polyhedra that tile three-space without gaps: the cube, the truncated
# octahedron (the Kelvin cell, the only Archimedean solid that tiles on
# its own), the rhombic dodecahedron (the Voronoi cell of the FCC
# lattice), and the tetrahedron-octahedron honeycomb.  Each is built from
# its defining half-space planes rather than from a vertex table, so the
# faces come out planar by construction.
#
# References:
# - Lord Kelvin (W. Thomson), "On the division of space with minimum
#   partitional area", Philosophical Magazine 24, 1887, pp. 503-514.
# - H. S. M. Coxeter, "Regular Polytopes", 3rd ed., Dover, 1973.
# - J. H. Conway, H. Burgiel and C. Goodman-Strauss, "The Symmetries of
#   Things", A K Peters, 2008 -- the honeycomb classification.

import itertools

import numpy as np


def _plane_faces(verts, planes):
    """One convex face per plane (normal, offset): the vertices
    with v . normal == offset, wound counterclockwise seen from
    outside (all plane normals point away from the cell centre)."""
    V = np.asarray(verts, float)
    faces = []
    for nrm, off in planes:
        n = np.asarray(nrm, float)
        sel = np.where(np.abs(V @ n - off) < 1e-9)[0]
        nn = n / np.linalg.norm(n)
        a = (np.array((1.0, 0.0, 0.0)) if abs(nn[0]) < 0.9
             else np.array((0.0, 1.0, 0.0)))
        u = np.cross(nn, a)
        u /= np.linalg.norm(u)
        v = np.cross(nn, u)            # (u, v, nn) right-handed
        d = V[sel] - V[sel].mean(axis=0)
        ang = np.arctan2(d @ v, d @ u)
        faces.append([int(i) for i in sel[np.argsort(ang)]])
    return faces


def _tet_faces(V):
    """Outward-wound faces of the tetrahedron with vertices V."""
    c = V.mean(axis=0)
    faces = []
    for tri in itertools.combinations(range(4), 3):
        i, j, k = tri
        n = np.cross(V[j] - V[i], V[k] - V[i])
        if np.dot(n, V[i] - c) < 0:
            tri = (i, k, j)
        faces.append(list(tri))
    return faces


def _axis_planes(off):
    return [((s, 0.0, 0.0), off) for s in (-1.0, 1.0)] + \
           [((0.0, s, 0.0), off) for s in (-1.0, 1.0)] + \
           [((0.0, 0.0, s), off) for s in (-1.0, 1.0)]


def _diag_planes(off):
    return [(s, off)
            for s in itertools.product((-1.0, 1.0), repeat=3)]


# unit cube about its centre
_CUBE_V = np.array([(x - .5, y - .5, z - .5)
                    for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                   float)


_CUBE_F = _plane_faces(_CUBE_V, _axis_planes(0.5))


# regular octahedron, vertices on the axes (edge sqrt2)
_OCT_V = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                   (0, 0, 1), (0, 0, -1)], float)


_OCT_F = _plane_faces(_OCT_V, _diag_planes(1.0))


# octet tetrahedra: the even-parity corners of a unit cube, about
# the cube centre; odd cube parity gives the mirror orientation
_TET_V = (np.array([(-.5, -.5, -.5), (.5, .5, -.5),
                    (.5, -.5, .5), (-.5, .5, .5)]),)


_TET_V = _TET_V + (-_TET_V[0],)


_TET_F = (_tet_faces(_TET_V[0]), _tet_faces(_TET_V[1]))


# truncated octahedron: all permutations of (0, +-1, +-2); the
# BCC Voronoi cell, 6 squares + 8 hexagons, width 4 along an axis
_TO_V = np.array(sorted({p for s1 in (-1, 1) for s2 in (-2, 2)
                         for p in itertools.permutations(
                             (0, s1, s2))}), float)


_TO_F = _plane_faces(_TO_V, _axis_planes(2.0) + _diag_planes(3.0))


# rhombic dodecahedron: the FCC Voronoi cell, 12 rhombi normal to
# the (1,1,0)-type neighbour directions, width 2 along an axis
_RD_V = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1)]
                 + [(x * .5, y * .5, z * .5)
                    for x in (-1, 1) for y in (-1, 1)
                    for z in (-1, 1)], float)


_RD_P = ([((a, b, 0.0), 1.0)
          for a in (-1.0, 1.0) for b in (-1.0, 1.0)]
         + [((a, 0.0, b), 1.0)
            for a in (-1.0, 1.0) for b in (-1.0, 1.0)]
         + [((0.0, a, b), 1.0)
            for a in (-1.0, 1.0) for b in (-1.0, 1.0)])


_RD_F = _plane_faces(_RD_V, _RD_P)


_AXIS_TAG = {(1.0, 0.0, 0.0): 0, (-1.0, 0.0, 0.0): 1,
             (0.0, 1.0, 0.0): 2, (0.0, -1.0, 0.0): 3,
             (0.0, 0.0, 1.0): 4, (0.0, 0.0, -1.0): 5}


def _is_cube_vert(v):
    return all(abs(abs(c) - 0.5) < 1e-9 for c in v)


def _obtet_cells():
    """The 24 tetragonal disphenoids of one rhombic dodecahedron
    (centred at the origin) as (centroid_offset, local_verts, faces,
    orientation_tag) about the RD centre."""
    out = []
    for f in _RD_F:
        vs = [_RD_V[i] for i in f]
        axis = [v for v in vs if not _is_cube_vert(v)]
        cube = [v for v in vs if _is_cube_vert(v)]
        for a in axis:                    # two disphenoids per face
            T = np.array([(0.0, 0.0, 0.0), a, cube[0], cube[1]])
            tc = T.mean(axis=0)
            tag = _AXIS_TAG[tuple(a)]
            out.append((tc, T - tc, _tet_faces(T - tc), tag))
    return out


_OBTET_CELLS = _obtet_cells()


# ------------------------------------------------------------------
# rhombic spirallohedra (Russell Towle): cells cut from the polar
# zonohedron construction, spiralling bundles of rhombi
# ------------------------------------------------------------------
_SPIRAL_ARMS = {'SPIRAL3': 3, 'SPIRAL4': 4}


_SPIRAL_CACHE = {}


def spiral_n(kind, segments):
    """Usable polar-star size: segments rounded down to a multiple
    of the arm count (at least two rhombi per arm turn)."""
    arms = _SPIRAL_ARMS[kind]
    return max(2 * arms, (segments // arms) * arms)


def _face_key(P, nd=6):
    return frozenset(map(tuple, np.round(P, nd)))


def _tiling_basis(V, F, vol):
    """Basis of the translation lattice of a face-to-face
    translational tiling by the cell (V, F): for each face find the
    translations carrying it onto an opposite face of the same
    cell, then pick three that span volume vol.  Raises if the cell
    does not tile that way."""
    data = []
    for f in F:
        P = V[list(f)]
        n = np.zeros(3)
        for i in range(1, len(f) - 1):
            n += np.cross(P[i] - P[0], P[i + 1] - P[0])
        data.append((P.mean(axis=0), n, P))
    trans = []
    for c, n, P in data:
        found = False
        for c2, n2, P2 in data:
            if np.linalg.norm(n + n2) > 1e-6:
                continue
            t = c - c2
            if np.linalg.norm(t) < 1e-9:
                continue
            if _face_key(P - t) == _face_key(P2):
                found = True
                if not any(np.linalg.norm(t - u) < 1e-6
                           for u in trans):
                    trans.append(t)
        if not found:
            raise ValueError("cell has a face with no translation "
                             "partner: not a translational tiler")
    for tri in itertools.combinations(trans, 3):
        B = np.array(tri)
        if abs(abs(np.linalg.det(B)) - vol) < 1e-6 * max(1.0, vol):
            if all(np.allclose(s := np.linalg.solve(B.T, t),
                               np.round(s), atol=1e-6)
                   for t in trans):
                return B
    raise ValueError("no lattice basis matches the cell volume")


def _spiral_data(kind, segments=12, pitch=55.0):
    """(local_verts, faces, basis, height, volume) of the
    S(n, n/arms) cell, centred on the midpoint of its two apexes;
    cached per (kind, n, pitch)."""
    n = spiral_n(kind, segments)
    key = (kind, n, round(pitch, 4))
    if key in _SPIRAL_CACHE:
        return _SPIRAL_CACHE[key]
    try:
        from .zonohedra_generator import (polar_star,
                                          make_polar_zonohedron)
    except ImportError:
        from zonohedra_generator import (polar_star,
                                         make_polar_zonohedron)
    w = n // _SPIRAL_ARMS[kind]
    verts, faces = make_polar_zonohedron(polar_star(n, pitch), 1, w)
    V = np.asarray(verts, float)
    vol = _mesh_volume(V, faces)
    if vol < 0:                        # wind the faces outward
        faces = [list(reversed(f)) for f in faces]
        vol = -vol
    B = _tiling_basis(V, faces, vol)
    centre = (V[0] + V[1]) / 2.0       # the two apexes
    height = abs(V[1][2] - V[0][2])
    out = (V - centre, faces, B, height, vol)
    _SPIRAL_CACHE[key] = out
    return out


def build_block(kind, nx, ny, nz, spiral_segments=12,
                spiral_pitch=55.0):
    """Cells of the block in canonical lattice coordinates.

    Returns (cells, pitch): cells is a list of tuples
    (centroid, local_verts, faces, tag) with local_verts about the
    centroid, tag 0 for the primary cell (octahedra for OCTET) and
    1 for the secondary (tetrahedra); pitch is the canonical length
    the Cell Size property maps to (one lattice step per axis)."""
    P = itertools.product
    cells = []
    if kind == 'CUBIC':
        for i, j, k in P(range(nx), range(ny), range(nz)):
            cells.append((np.array((i + .5, j + .5, k + .5)),
                          _CUBE_V, _CUBE_F, 0))
        return cells, 1.0
    if kind == 'OCTET':
        # one tetrahedron per unit cube of an nx x ny x nz box
        # (orientation by cube parity); octahedra at the odd-parity
        # integer points strictly inside, so the block stays boxy
        for i, j, k in P(range(nx), range(ny), range(nz)):
            p = (i + j + k) % 2
            cells.append((np.array((i + .5, j + .5, k + .5)),
                          _TET_V[p], _TET_F[p], 1))
        for x, y, z in P(range(1, nx), range(1, ny), range(1, nz)):
            if (x + y + z) % 2:
                cells.append((np.array((x, y, z), float),
                              _OCT_V, _OCT_F, 0))
        return cells, 1.0
    if kind == 'TRUNCOCT':
        # BCC: a primary grid plus a second grid at the cube centres
        for i, j, k in P(range(nx), range(ny), range(nz)):
            cells.append((4.0 * np.array((i, j, k)),
                          _TO_V, _TO_F, 0))
        for i, j, k in P(range(nx - 1), range(ny - 1),
                         range(nz - 1)):
            cells.append((4.0 * np.array((i, j, k)) + 2.0,
                          _TO_V, _TO_F, 0))
        return cells, 4.0
    if kind == 'RHOMBDODEC':
        # FCC: even-parity integer points of a box, pitch 2
        for x, y, z in P(range(2 * nx - 1), range(2 * ny - 1),
                         range(2 * nz - 1)):
            if (x + y + z) % 2 == 0:
                cells.append((np.array((x, y, z), float),
                              _RD_V, _RD_F, 0))
        return cells, 2.0
    if kind == 'OBTET':
        # each FCC rhombic dodecahedron split into 24 disphenoids
        for x, y, z in P(range(2 * nx - 1), range(2 * ny - 1),
                         range(2 * nz - 1)):
            if (x + y + z) % 2 == 0:
                c = np.array((x, y, z), float)
                for tc, V, F, tag in _OBTET_CELLS:
                    cells.append((c + tc, V, F, tag))
        return cells, 2.0
    if kind in _SPIRAL_ARMS:
        # translates over the derived tiling lattice; tag by
        # lattice parity for the optional two-tone coloring
        V, F, B, height, vol = _spiral_data(kind, spiral_segments,
                                            spiral_pitch)
        for i, j, k in P(range(nx), range(ny), range(nz)):
            cells.append((i * B[0] + j * B[1] + k * B[2],
                          V, F, (i + j + k) % 2))
        return cells, height
    raise ValueError(kind)


def build_mesh(kind='OCTET', nx=3, ny=3, nz=2, gap=0.92, size=1.0,
               spiral_segments=12, spiral_pitch=55.0):
    """Whole block as (verts, faces, face_tags), centred at the
    origin; gap scales every cell about its own centroid and size
    scales one lattice step to that length."""
    cells, pitch = build_block(kind, nx, ny, nz, spiral_segments,
                               spiral_pitch)
    s = size / pitch
    lo = np.full(3, np.inf)
    hi = -lo
    for c, V, F, t in cells:
        lo = np.minimum(lo, c + V.min(axis=0))
        hi = np.maximum(hi, c + V.max(axis=0))
    mid = (lo + hi) / 2.0
    verts, faces, tags = [], [], []
    for c, V, F, t in cells:
        base = len(verts)
        verts.extend(map(tuple, (c - mid + gap * V) * s))
        faces.extend([base + i for i in f] for f in F)
        tags.extend([t] * len(F))
    return verts, faces, tags


_CELL_VOL = {'CUBIC': {0: 1.0},               # per canonical cell
             'OCTET': {0: 4.0 / 3.0, 1: 1.0 / 3.0},
             'TRUNCOCT': {0: 32.0},
             'RHOMBDODEC': {0: 2.0},
             'OBTET': {t: 1.0 / 12.0 for t in range(6)}}


def block_volume(kind, nx, ny, nz, size=1.0, spiral_segments=12,
                 spiral_pitch=55.0):
    """Analytic total volume of all cells at gap = 1."""
    cells, pitch = build_block(kind, nx, ny, nz, spiral_segments,
                               spiral_pitch)
    s3 = (size / pitch) ** 3
    if kind in _SPIRAL_ARMS:
        vol = _spiral_data(kind, spiral_segments, spiral_pitch)[4]
        return len(cells) * vol * s3
    return sum(_CELL_VOL[kind][t] for *_, t in cells) * s3


def _mesh_volume(verts, faces):
    """Signed volume of a from_pydata-style mesh (outward faces)."""
    V = np.asarray(verts, float)
    tot = 0.0
    for f in faces:
        for i in range(1, len(f) - 1):
            tot += np.linalg.det(V[[f[0], f[i], f[i + 1]]])
    return tot / 6.0
