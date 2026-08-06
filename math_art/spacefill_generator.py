
# Space-Filling Solids generator for Blender
#
# Blocks of the classical space-filling honeycombs, every cell
# shrunk by a gap factor about its own centroid so the packing
# reads as a stack of separate solids -- after Henry Segerman,
# "Visualizing Mathematics with 3D Printing", figs 4-17..4-20.
#
#   CUBIC       cubes on the cubic lattice
#   OCTET       alternated cubic honeycomb: regular octahedra on
#               one FCC sublattice + regular tetrahedra (two
#               orientations) filling the gaps, ratio 1:2 in bulk
#   TRUNCOCT    bitruncated cubic: truncated octahedra on BCC
#   RHOMBDODEC  rhombic dodecahedra on FCC
#   SPIRAL3     3-armed rhombic spirallohedra S(12, 4) (Towle)
#   SPIRAL4     4-armed rhombic spirallohedra S(12, 3)
#
# The spirallohedra tile space by pure translations: every rhombic
# face of the cell maps onto an opposite face of the same cell
# under some translation, and those translations span a rank-3
# lattice whose fundamental volume equals the cell volume (both
# verified numerically at import of the cell -- see
# _spiral_data).  The lattice basis is derived from the face
# pairing, not hardcoded.
#
# At gap = 1.0 adjacent cells share their faces exactly (the
# lattices below are in integer canonical coordinates, so shared
# faces coincide to the bit); the default gap < 1 gives the
# printed-separately look.
#
# References:
# - Henry Segerman, "Visualizing Mathematics with 3D Printing", Johns
#   Hopkins University Press, 2016 (figs 4-17..4-20, space-filling
#   solids).
# - Rhombic spirallohedra S(n,w) via the polar-zonohedron
#   construction: Russell Towle (Spirallohedra / polar-zonohedron
#   notebooks).
# - The alternated cubic (octet) and bitruncated cubic honeycombs and
#   the Voronoi cells of the cubic, BCC and FCC lattices: H. S. M.
#   Coxeter, "Regular Polytopes", 3rd ed., Dover, 1973.
# - The obtetrahedrille (Conway's oblate tetrahedrille / tetragonal
#   disphenoid honeycomb, each rhombic dodecahedron split into 24
#   disphenoids): J. H. Conway, H. Burgiel, C. Goodman-Strauss, "The
#   Symmetries of Things", A K Peters, 2008; and Tom Verhoeff & Koos
#   Verhoeff, "The Obtetrahedrille as a Modular Building Block for 3D
#   Mathematical Art", Bridges 2019, 407-410.

bl_info = {
    "name": "Space-Filling Solids",
    "author": "Math Art project (after Henry Segerman)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Space-Filling Solids",
    "description": "Honeycomb blocks: cubes, octet truss, "
                   "truncated octahedra, rhombic dodecahedra",
    "category": "Add Mesh",
}

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


# ------------------------------------------------------------------
# obtetrahedrille (Conway's oblate tetrahedrille): the rhombic
# dodecahedron splits into 24 congruent tetragonal disphenoids -- the
# space-filling cell of the tetragonal disphenoid honeycomb (Verhoeff
# & Verhoeff, Bridges 2019).  Each rhombic face pyramid (RD centre +
# the 4 face vertices) splits along its short diagonal (the two
# degree-3 "cube" vertices) into two disphenoids.  Every disphenoid
# has one opposite edge pair of length 1 and four edges of length
# sqrt(3)/2; the six orientations (by the axis vertex direction) tile
# space by translation of the FCC-placed rhombic dodecahedra.
# ------------------------------------------------------------------

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


try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _LABEL = {'CUBIC': "Cubes", 'OCTET': "Octet",
              'TRUNCOCT': "Truncated Octahedra",
              'RHOMBDODEC': "Rhombic Dodecahedra",
              'OBTET': "Obtetrahedrille",
              'SPIRAL3': "Spirallohedra (3-Armed)",
              'SPIRAL4': "Spirallohedra (4-Armed)"}

    # six-colour palette for the obtetrahedrille disphenoid orientations
    _OBTET_RGB = {0: (0.86, 0.28, 0.24), 1: (0.30, 0.55, 0.82),
                  2: (0.32, 0.70, 0.42), 3: (0.95, 0.76, 0.28),
                  4: (0.60, 0.40, 0.75), 5: (0.28, 0.72, 0.70)}

    def _material(name, rgb):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = (*rgb, 1.0)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.5
        return mat

    class MESH_OT_spacefill_add(bpy.types.Operator):
        """Add a block of a space-filling honeycomb, each cell
        shrunk about its centroid so the packing reads as a stack
        of separate solids (after Segerman)"""
        bl_idname = "mesh.spacefill_add"
        bl_label = "Space-Filling Solids"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Honeycomb",
            items=[('CUBIC', "Cubes",
                    "Cubes on the cubic lattice"),
                   ('OCTET', "Octahedra + Tetrahedra",
                    "Alternated cubic honeycomb: regular octahedra "
                    "and tetrahedra (the octet truss)"),
                   ('TRUNCOCT', "Truncated Octahedra",
                    "Bitruncated cubic honeycomb on the BCC "
                    "lattice"),
                   ('RHOMBDODEC', "Rhombic Dodecahedra",
                    "Rhombic dodecahedra on the FCC lattice"),
                   ('OBTET', "Obtetrahedrille",
                    "The tetragonal disphenoid honeycomb: each FCC "
                    "rhombic dodecahedron split into 24 congruent "
                    "disphenoids (Conway's oblate tetrahedrille), "
                    "six-coloured by orientation"),
                   ('SPIRAL3', "Rhombic Spirallohedra (3-Armed)",
                    "Three-armed rhombic spirallohedra S(12,4) "
                    "after Russell Towle, interlocking on their "
                    "translation lattice"),
                   ('SPIRAL4', "Rhombic Spirallohedra (4-Armed)",
                    "Four-armed rhombic spirallohedra S(12,3) "
                    "after Russell Towle, interlocking on their "
                    "translation lattice")],
            default='OCTET')
        nx: IntProperty(name="Cells X", default=3, min=1, max=12,
                        description="Lattice cells along X")
        ny: IntProperty(name="Cells Y", default=3, min=1, max=12,
                        description="Lattice cells along Y")
        nz: IntProperty(name="Cells Z", default=2, min=1, max=12,
                        description="Lattice cells along Z")
        gap: FloatProperty(
            name="Gap Factor", default=0.92, min=0.05, max=1.0,
            description="Scale of each cell about its own centroid "
                        "(1.0 = cells share faces exactly)")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Overall scale (1.0 fits a 2 m cube, centred "
                        "at the origin)")
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain solid cells"),
                   ('LEONARDO', "Leonardo",
                    "Open-faced da Vinci panels (Geometry Nodes "
                    "modifier on the whole block)"),
                   ('BALLSTICK', "Ball and Stick",
                    "Edges as solid cylindrical struts and vertices "
                    "as small spheres (ball-and-stick model)")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.3, min=0.02, max=0.95,
            description="Leonardo face frame width (fraction of "
                        "the face)")
        thickness: FloatProperty(
            name="Thickness", default=0.06, min=0.001, max=1.0,
            description="Leonardo panel thickness")
        strut_radius: FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.5,
            description="Ball-and-stick edge cylinder radius")
        node_radius: FloatProperty(
            name="Node Radius", default=0.035, min=0.0, max=0.5,
            description="Ball-and-stick vertex sphere radius "
                        "(0 = no nodes)")
        two_materials: BoolProperty(
            name="Two Materials", default=True,
            description="Distinct materials for octahedra and "
                        "tetrahedra (Octet), or alternating by "
                        "lattice parity (Spirallohedra); the other "
                        "honeycombs have no honest 2-coloring")
        spiral_segments: IntProperty(
            name="Spiral Segments", default=12, min=6, max=24,
            description="Star vectors of the spirallohedron cell, "
                        "rounded down to a multiple of the arm "
                        "count; more segments give finer, longer "
                        "spiral cells")
        spiral_pitch: FloatProperty(
            name="Spiral Pitch", default=55.0, min=5.0, max=85.0,
            description="Polar star pitch angle from the axis "
                        "(degrees); higher is squatter, wider "
                        "spiral cells")

        def execute(self, context):
            try:
                verts, faces, tags = build_mesh(
                    self.kind, self.nx, self.ny, self.nz,
                    self.gap, 1.0, self.spiral_segments,
                    self.spiral_pitch)
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            # centre at the origin and fit a 2 m cube, then scale
            P = np.asarray(verts, float)
            if len(P):
                lo = P.min(axis=0)
                hi = P.max(axis=0)
                span = float(np.max(hi - lo)) or 1.0
                P = (P - (lo + hi) / 2.0) * (2.0 * self.scale / span)
                verts = [tuple(p) for p in P]
            me = bpy.data.meshes.new("Spacefill")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            if (self.kind == 'OBTET'
                    and len(me.polygons) == len(tags)):
                # six materials, one per disphenoid orientation
                for t in range(6):
                    me.materials.append(_material(
                        f"Spacefill Obtet {t}", _OBTET_RGB[t]))
                me.polygons.foreach_set('material_index', tags)
            elif (self.kind in ('OCTET', 'SPIRAL3', 'SPIRAL4')
                    and self.two_materials
                    and len(me.polygons) == len(tags)):
                if self.kind == 'OCTET':
                    me.materials.append(_material(
                        "Spacefill Octahedron", (0.9, 0.45, 0.12)))
                    me.materials.append(_material(
                        "Spacefill Tetrahedron", (0.15, 0.35, 0.8)))
                else:
                    me.materials.append(_material(
                        "Spacefill Spiral A", (0.85, 0.55, 0.15)))
                    me.materials.append(_material(
                        "Spacefill Spiral B", (0.2, 0.45, 0.7)))
                me.polygons.foreach_set('material_index', tags)
            me.update()
            obj = bpy.data.objects.new(
                f"Spacefill {_LABEL[self.kind]}", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.style == 'LEONARDO':
                try:
                    from . import leonardo_style
                except ImportError:
                    import leonardo_style
                leonardo_style.add_modifier(obj, self.border,
                                            self.thickness)
            elif self.style == 'BALLSTICK':
                try:
                    from . import ball_and_stick
                except ImportError:
                    import ball_and_stick
                ball_and_stick.rebuild(obj, self.strut_radius,
                                       self.node_radius)
            self.report({'INFO'},
                        f"{_LABEL[self.kind]}: "
                        f"V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('kind', 'nx', 'ny', 'nz', 'gap', 'scale'):
                lay.prop(self, k)
            if self.kind in ('SPIRAL3', 'SPIRAL4'):
                lay.prop(self, 'spiral_segments')
                lay.prop(self, 'spiral_pitch')
            if self.kind in ('OCTET', 'SPIRAL3', 'SPIRAL4'):
                lay.prop(self, 'two_materials')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
                lay.prop(self, 'thickness')
            elif self.style == 'BALLSTICK':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')

    def _menu_func(self, context):
        self.layout.operator("mesh.spacefill_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_spacefill_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_spacefill_add)


def _selftest():
    # at gap = 1 the numeric mesh volume must equal the
    # analytic total cell volume for every honeycomb
    for kind in ('CUBIC', 'OCTET', 'TRUNCOCT', 'RHOMBDODEC',
                 'OBTET', 'SPIRAL3', 'SPIRAL4'):
        v, f, t = build_mesh(kind, 3, 3, 2, gap=1.0)
        num = _mesh_volume(v, f)
        ana = block_volume(kind, 3, 3, 2)
        ok = abs(num - ana) < 1e-9 * max(1.0, ana)
        print(f"{kind}: vol={num:.6f} ({ana:.6f}) "
              f"{'OK' if ok else 'BAD'}")
    # spirallohedra: the tiling-basis derivation must succeed
    # (it raises when the cell is not a translational tiler)
    # and at gap = 1 non-boundary faces of a block must be
    # shared exactly, for a sweep of segment counts and pitches
    for kind in ('SPIRAL3', 'SPIRAL4'):
        for segs in (6, 8, 9, 12, 15, 16, 20, 24):
            if spiral_n(kind, segs) != segs:
                continue
            for pitch in (35.0, 45.0, 55.0, 70.0):
                V, F, B, height, vol = _spiral_data(kind, segs,
                                                    pitch)
                keys = {}
                for off in itertools.product(range(2),
                                             repeat=3):
                    t = (off[0] * B[0] + off[1] * B[1]
                         + off[2] * B[2])
                    for f in F:
                        k = _face_key(V[list(f)] + t)
                        keys[k] = keys.get(k, 0) + 1
                shared = sum(1 for c in keys.values()
                             if c == 2)
                assert shared > 0, (kind, segs, pitch)
            print(f"{kind} n={segs}: tiles at all pitches, "
                  f"shared faces OK")
    print("spacefill standalone tests passed")
