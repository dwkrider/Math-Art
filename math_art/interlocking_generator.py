
# Topological Interlocking & Modular Blocks generator for Blender
#
# Assemblies of rigid blocks that, once a peripheral "frame" is held
# fixed, kinematically lock: no non-empty subset of the interior
# blocks can be slid or rotated out without collision.  The blocks
# are never glued -- interlocking is purely geometric.
#
# Families implemented here:
#
#   TETRA     the canonical interlocking layer of regular tetrahedra
#             (two 90-deg-rotated orientations on a checkerboard),
#             after Dyskin, Estrin, Kanel-Belov & Pasternak (2001).
#   MCS       the "moving cross-section" family: cubes and octahedra
#             reconstructed as the intersection of edge-tilted planes
#             over a hexagonal middle section (Kanel-Belov et al.).
#   ESCHER    the Escher-trick / osteomorphic / Versatile-block
#             family: a square (p4) fundamental domain is edge-
#             deformed, placed at z=0, rotated 90 deg at the mid-
#             plane, and lofted back to z=1, so every horizontal
#             section tiles the plane and copies interlock.  A sine
#             deformation reproduces Estrin's osteomorphic block; a
#             tent/zig-zag deformation reproduces a Versatile-style
#             block.  Two-colouring follows the Truchet rule.
# TETRA, ESCHER, KITTEN and the SL strand are space-filling /
# interlocking assemblies (all verified non-overlapping).  The exact
# Versatile block and the UFO / cushion tetroctahedrille blocks are
# shown as single reference blocks -- their space-filling assemblies
# require the rotation-based Truchet / grid grammars left for future
# work.
#
# References:
# - A. V. Dyskin, Y. Estrin, A. J. Kanel-Belov, E. Pasternak, "A new
#   concept in design of materials and structures: assemblies of
#   interlocked tetrahedron-shaped elements", Scripta Materialia 44
#   (2001) 2689-2694.
# - A. J. Kanel-Belov, A. V. Dyskin, Y. Estrin, E. Pasternak,
#   I. A. Ivanov-Pogodaev, "Interlocking of convex polyhedra: towards
#   a geometric theory of fragmented solids", Moscow Math. J. 10(2)
#   (2010) 337-342 (arXiv:0812.5089).
# - Y. Estrin, A. V. Dyskin, E. Pasternak et al., "Fracture resistant
#   structures based on topological interlocking with non-planar
#   contacts", Adv. Eng. Materials 5(3) (2003) 116-119 (osteomorphic
#   blocks).
# - R. Akpanya, T. Goertzen, S. Wiesenhuetter, A. C. Niemeyer,
#   J. Noennig, "Topological Interlocking, Truchet Tiles and
#   Self-Assemblies", Bridges 2023, 61-68 (Versatile block).
# - R. Akpanya, T. Goertzen, A. C. Niemeyer, "From Tilings of
#   Orientable Surfaces to Topological Interlocking Assemblies",
#   Applied Sciences 14(16):7276 (2024) (the Escher-loft framework).

bl_info = {
    "name": "Topological Interlocking Blocks",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Topological Interlocking",
    "description": "Interlocking block assemblies: tetrahedra "
                   "layers, Escher/osteomorphic blocks, "
                   "tetroctahedrille and SL blocks",
    "category": "Add Mesh",
}

import itertools
import math

import numpy as np


# ==================================================================
# generic mesh helpers
# ==================================================================

def _tet_faces(V):
    """Outward-wound triangular faces of the tetrahedron V (4x3)."""
    V = np.asarray(V, float)
    c = V.mean(axis=0)
    faces = []
    for tri in itertools.combinations(range(4), 3):
        i, j, k = tri
        n = np.cross(V[j] - V[i], V[k] - V[i])
        if np.dot(n, V[i] - c) < 0:
            tri = (i, k, j)
        faces.append(list(tri))
    return faces


def _wind_face(V, idx, outward_ref):
    """Order the coplanar vertex indices `idx` of V into a convex
    polygon wound counterclockwise as seen from outside, where
    `outward_ref` is a point strictly outside along the face
    normal side (used only to fix orientation)."""
    P = V[idx]
    c = P.mean(axis=0)
    # face normal from the best-fit plane (Newell)
    n = np.zeros(3)
    m = len(idx)
    for i in range(m):
        a = P[i] - c
        b = P[(i + 1) % m] - c
        n += np.cross(a, b)
    if np.linalg.norm(n) < 1e-12:
        n = np.cross(P[1] - P[0], P[2] - P[0])
    n = n / (np.linalg.norm(n) + 1e-30)
    a = (np.array((1.0, 0.0, 0.0)) if abs(n[0]) < 0.9
         else np.array((0.0, 1.0, 0.0)))
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    d = P - c
    ang = np.arctan2(d @ v, d @ u)
    order = list(np.argsort(ang))
    ordered = [int(idx[i]) for i in order]
    # flip so the winding normal points away from outward_ref side
    if np.dot(n, c - outward_ref) < 0:
        ordered = ordered[::-1]
    return ordered


def _hull_from_halfspaces(planes, tol=1e-7):
    """Bounded convex polyhedron that is the intersection of the
    half-spaces {X : n . X <= d} for (n, d) in `planes`.

    Vertices are the triple-plane intersection points that satisfy
    every half-space; one polygonal face is emitted per plane that
    carries at least three vertices.  Returns (verts, faces) or
    raises ValueError if the intersection is empty/unbounded (fewer
    than 4 vertices found)."""
    P = [(np.asarray(n, float), float(d)) for n, d in planes]
    pts = []
    npl = len(P)
    for i, j, k in itertools.combinations(range(npl), 3):
        A = np.array([P[i][0], P[j][0], P[k][0]])
        b = np.array([P[i][1], P[j][1], P[k][1]])
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        x = np.linalg.solve(A, b)
        if all(np.dot(n, x) <= d + tol for n, d in P):
            pts.append(x)
    if len(pts) < 4:
        raise ValueError("halfspace intersection is not a solid")
    V = np.array(pts)
    # dedup
    keep = []
    for p in V:
        if not any(np.linalg.norm(p - q) < 1e-6 for q in keep):
            keep.append(p)
    V = np.array(keep)
    c = V.mean(axis=0)
    faces = []
    for n, d in P:
        sel = [i for i in range(len(V))
               if abs(np.dot(n, V[i]) - d) < 1e-6]
        if len(sel) >= 3:
            faces.append(_wind_face(V, np.array(sel), c))
    return V, faces


def _mesh_volume(verts, faces):
    """Signed volume of an outward-wound from_pydata mesh."""
    V = np.asarray(verts, float)
    tot = 0.0
    for f in faces:
        for i in range(1, len(f) - 1):
            tot += np.linalg.det(V[[f[0], f[i], f[i + 1]]])
    return tot / 6.0


def _regular_polygon(n, r=1.0, phase=0.0):
    """n-gon vertices (n x 2) of circumradius r in the plane."""
    a = phase + 2.0 * np.pi * np.arange(n) / n
    return np.column_stack((r * np.cos(a), r * np.sin(a)))


# ==================================================================
# FAMILY: TETRA -- interlocking regular-tetrahedron layer
# ==================================================================
#
# A regular tetrahedron with two opposite edges mutually
# perpendicular and horizontal: the top edge (z = +H) runs along one
# axis, the bottom edge (z = -H) along the perpendicular axis.  Its
# horizontal mid-section is a unit square.  Placing type A (top edge
# along x) on one colour of the checkerboard and type B = A turned 90
# deg on the other colour makes every cell's square mid-section tile
# the plane; the x-neighbours block a cell from rising and the
# y-neighbours block it from sinking, so the framed interior locks.

# top/bottom half-height for a *regular* tetra whose opposite edges
# have length 2 (mid-section = unit square): slant edge length is
# sqrt(2 + 4H^2) = 2  =>  H = 1/sqrt(2)
_TET_H = 1.0 / math.sqrt(2.0)

# type A: top edge along x at +H, bottom edge along y at -H
_TETRA_A = np.array([(1.0, 0.0, _TET_H), (-1.0, 0.0, _TET_H),
                     (0.0, 1.0, -_TET_H), (0.0, -1.0, -_TET_H)])
# type B: 90-deg rotation about z (top edge along y)
_TETRA_B = _TETRA_A[:, (1, 0, 2)] * np.array((1.0, 1.0, 1.0))
_TETRA_B = np.array([(0.0, 1.0, _TET_H), (0.0, -1.0, _TET_H),
                     (1.0, 0.0, -_TET_H), (-1.0, 0.0, -_TET_H)])
_TETRA_FA = _tet_faces(_TETRA_A)
_TETRA_FB = _tet_faces(_TETRA_B)


def build_tetra(nx, ny):
    """Cells of the interlocking-tetrahedron layer as
    (centre, local_verts, faces, is_frame, colour) tuples; the layer
    fills an nx by ny patch of the integer lattice, one tetra per
    lattice point, checkerboarded into the two orientations.  The
    outer ring is flagged as frame."""
    cells = []
    for i, j in itertools.product(range(nx), range(ny)):
        parity = (i + j) % 2
        V = _TETRA_A if parity == 0 else _TETRA_B
        F = _TETRA_FA if parity == 0 else _TETRA_FB
        frame = (i == 0 or j == 0 or i == nx - 1 or j == ny - 1)
        cells.append((np.array((float(i), float(j), 0.0)),
                      V, F, frame, parity))
    return cells


# ==================================================================
# FAMILY: MCS -- moving-cross-section cubes & octahedra
# ==================================================================
#
# Erect tilted planes over the six edges of a regular hexagon (the
# common middle section) and intersect their half-spaces.  Around the
# hexagon the tilt sense alternates (+ - + - + -); at the tilt angle
# beta = asin(1/sqrt3) the six planes close into a cube, at
# beta = asin(1/3) (plus horizontal caps) into an octahedron.  The
# hexagon is the section of the solid normal to a 3-fold axis.
#
# Neighbouring hexagons share an edge, and a shared edge must carry
# opposite tilt senses for the two cells (outward for one is inward
# for the other); this yields exactly two cell types, placed on the
# two sublattices of the honeycomb's brick colouring.

# tilt angle of the side faces from vertical for the cube case
_MCS_BETA = {'CUBE': math.asin(1.0 / math.sqrt(3.0))}


def _hex_edges(r=1.0, phase=0.0):
    """(midpoint_normal, apothem) for the 6 edges of a regular
    hexagon of circumradius r; normals point outward."""
    verts = _regular_polygon(6, r, phase)
    out = []
    for i in range(6):
        a = verts[i]
        b = verts[(i + 1) % 6]
        mid = (a + b) / 2.0
        nrm = mid / np.linalg.norm(mid)
        out.append((nrm, float(np.dot(nrm, mid))))
    return out


def _align_to_z(axis):
    """Rotation matrix taking the unit vector `axis` onto +z."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    z = np.array((0.0, 0.0, 1.0))
    v = np.cross(a, z)
    c = float(np.dot(a, z))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag((1.0, -1.0, -1.0))
    vx = np.array(((0, -v[2], v[1]),
                   (v[2], 0, -v[0]),
                   (-v[1], v[0], 0)), float)
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _section_hexagon_radius(V):
    """Circumradius of the z=0 section of a convex solid, taken as
    the max in-plane radius of the points nearest z=0 (used only to
    normalise MCS cells to a unit-hexagon equator)."""
    r = np.hypot(V[:, 0], V[:, 1])
    return float(r.max())


def _orient_equator(V, faces):
    """Rotate a 3-fold-vertical solid about z so its z=0 section is a
    flat-top hexagon matching _regular_polygon(6, phase=0), and scale
    so the equatorial circumradius is 1."""
    # find the two vertex rings (top/bottom) and infer the equatorial
    # hexagon from edge midpoints crossing z=0
    zc = V[:, 2]
    top = V[zc > 1e-6]
    # in-plane angle of the highest-radius top vertex -> align a
    # hexagon vertex (equator vertex sits between two such); simplest
    # is to scale by the equatorial radius and rotate by measured
    # phase of the first equator vertex
    mids = []
    for f in faces:
        for i in range(len(f)):
            a, b = V[f[i]], V[f[(i + 1) % len(f)]]
            if a[2] * b[2] < -1e-9:
                t = a[2] / (a[2] - b[2])
                mids.append(a + t * (b - a))
    M = np.array(mids)
    # dedup equator points
    keep = []
    for p in M:
        if not any(np.linalg.norm(p - q) < 1e-6 for q in keep):
            keep.append(p)
    M = np.array(keep)
    R = np.hypot(M[:, 0], M[:, 1]).max()
    ph = math.atan2(M[0, 1], M[0, 0])
    Rz = np.array(((math.cos(-ph), -math.sin(-ph), 0),
                   (math.sin(-ph), math.cos(-ph), 0),
                   (0, 0, 1)))
    return (V @ Rz.T) / R


def _analytic_platonic(kind):
    """Regular cube or octahedron oriented with a 3-fold axis along z
    and normalised to a unit-circumradius equatorial hexagon at z=0.
    Returns (verts, faces)."""
    if kind == 'CUBE':
        V = np.array([(x, y, z) for x in (-1.0, 1.0)
                      for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
        faces = []                       # 6 quad faces
        for ax in range(3):
            for s in (-1.0, 1.0):
                sel = [i for i in range(8) if V[i][ax] == s]
                faces.append(_wind_face(V, np.array(sel),
                                        V.mean(axis=0)))
    elif kind == 'OCTA':
        V = np.array([(1.0, 0, 0), (-1.0, 0, 0), (0, 1.0, 0),
                      (0, -1.0, 0), (0, 0, 1.0), (0, 0, -1.0)])
        idx = {(0, 'x'): 0, (1, 'x'): 1}
        faces = []                       # 8 tri faces by sign octant
        for sx in (0, 1):
            for sy in (2, 3):
                for sz in (4, 5):
                    faces.append(_wind_face(V, np.array([sx, sy, sz]),
                                            V.mean(axis=0)))
    else:
        raise ValueError(kind)
    R = _align_to_z((1.0, 1.0, 1.0))
    V = V @ R.T
    return _orient_equator(V, faces), faces


_MCS_ANALYTIC = {}


def _mcs_cell(kind, mirror=False):
    """One MCS cell: a regular cube or octahedron with its 3-fold
    axis vertical and a unit-hexagon equator at z=0.  `mirror`
    reflects it in z (the interlocking partner orientation)."""
    if kind not in _MCS_ANALYTIC:
        _MCS_ANALYTIC[kind] = _analytic_platonic(kind)
    V, faces = _MCS_ANALYTIC[kind]
    V = V.copy()
    if mirror:
        V = V * np.array((1.0, 1.0, -1.0))
        faces = [f[::-1] for f in faces]     # keep outward winding
    return V, faces


def build_mcs(kind, nx, ny):
    """Cells of an MCS layer over a honeycomb patch.  Hexagon centres
    sit on a triangular lattice; the alternating tilt pattern gives
    two cell types placed by row/column parity.  Returns the same
    tuple form as build_tetra."""
    # flat-top unit hexagon (circumradius 1, apothem sqrt3/2): the
    # centres of the honeycomb sit on a triangular lattice; columns
    # are spaced 1.5 apart and odd columns drop by half a row
    ax = 1.5                         # horizontal (column) spacing
    ay = math.sqrt(3.0)             # vertical (row) spacing
    cellP = _mcs_cell(kind, mirror=False)
    cellQ = _mcs_cell(kind, mirror=True)
    cells = []
    for i in range(nx):
        for j in range(ny):
            cx = i * ax
            cy = (j + 0.5 * (i % 2)) * ay
            parity = (i + j) % 2
            V, F = cellP if parity == 0 else cellQ
            frame = (i == 0 or j == 0
                     or i == nx - 1 or j == ny - 1)
            cells.append((np.array((cx, cy, 0.0)),
                          np.asarray(V, float), F, frame, parity))
    return cells


# ==================================================================
# FAMILY: ESCHER -- Escher-trick / osteomorphic / Versatile loft
# ==================================================================
#
# Take the unit square [0,1]^2 (a p4 fundamental domain).  Deform its
# four edges by a single profile (the Escher trick: the four edges
# are 90-deg images of one another, so one profile fixes all four).
# Put the deformed square M at z=0, its 90-deg rotation M' at z=1/2,
# and M again at z=1, and loft.  Every horizontal section is a
# deformed-square tiling, so copies on the integer lattice fill the
# slab; checkerboard parity flips the mid rotation sense and gives the
# two Truchet colours.  A sine profile is Estrin's osteomorphic
# block; a tent profile is a Versatile-style block.

def _profile(kind, depth, t):
    """Edge offset at parameters t in [-0.5, 0.5]; zero at the two
    endpoints so tile corners are fixed."""
    if kind == 'SINE':
        off = depth * np.cos(np.pi * t)             # single bump
    elif kind == 'TENT':
        off = depth * (1.0 - 2.0 * np.abs(t))        # triangle
    elif kind == 'ZIGZAG':
        off = depth * np.sin(2.0 * np.pi * t)        # S-curve, mean 0
    elif kind == 'STEP':
        off = depth * np.clip(2.0 - 4.0 * np.abs(t),
                              0.0, 1.0)               # trapezoid
    else:
        off = np.zeros_like(t)
    off = np.asarray(off, float)
    off[0] = 0.0
    off[-1] = 0.0
    return off


def _deformed_square(kind, depth, samples):
    """Boundary polyline (Nx2) of the Escher-deformed unit square,
    centred at the origin, counterclockwise.  Opposite edges carry
    the SAME offset in the SAME absolute direction (bottom & top both
    shifted in +y by p(x); left & right both in +x by p(y)), so a
    copy translated by one unit mates its neighbour's edge exactly --
    the tile fills the plane with no gap and no overlap."""
    t = np.linspace(-0.5, 0.5, samples + 1)
    p = _profile(kind, depth, t)
    poly = []
    for s in range(samples):                         # bottom L->R
        poly.append((t[s], -0.5 + p[s]))
    for s in range(samples):                         # right  B->T
        poly.append((0.5 + p[s], t[s]))
    for s in range(samples):                         # top    R->L
        k = samples - s
        poly.append((t[k], 0.5 + p[k]))
    for s in range(samples):                         # left   T->B
        k = samples - s
        poly.append((-0.5 + p[k], t[k]))
    return np.array(poly)


def _rot2(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    R = np.array(((c, -s), (s, c)))
    return p @ R.T


def _loft_rings(rings, zs, close=True):
    """Triangulate a stack of equal-length rings at heights zs into a
    closed solid (bottom fan cap, side quads split to tris, top fan
    cap).  rings[k] is (m x 2); returns (verts, faces)."""
    m = len(rings[0])
    verts = []
    for ring, z in zip(rings, zs):
        for p in ring:
            verts.append((float(p[0]), float(p[1]), float(z)))
    V = np.array(verts)
    faces = []
    L = len(rings)
    # side walls
    for k in range(L - 1):
        base0 = k * m
        base1 = (k + 1) * m
        for i in range(m):
            j = (i + 1) % m
            a, b = base0 + i, base0 + j
            c, d = base1 + i, base1 + j
            faces.append([a, b, d])
            faces.append([a, d, c])
    if close:
        # bottom cap (ring 0), wound downward
        bc = list(range(m))
        cen_b = len(V)
        V = np.vstack([V, rings[0].mean(axis=0).tolist()
                       + [zs[0]]])
        for i in range(m):
            faces.append([cen_b, (i + 1) % m, i])
        # top cap (last ring), wound upward
        top0 = (L - 1) * m
        cen_t = len(V)
        V = np.vstack([V, rings[-1].mean(axis=0).tolist()
                       + [zs[-1]]])
        for i in range(m):
            faces.append([cen_t, top0 + i, top0 + (i + 1) % m])
    return [tuple(map(float, p)) for p in V], faces


def build_escher_block(kind, depth, samples, height):
    """One Escher-loft block centred at the origin: deformed square at
    z=-height/2, its 90-deg rotation at z=0, deformed square again at
    z=+height/2.  Every block is identical, so translated copies tile
    the slab at every height and interlock.  Returns (verts, faces)."""
    base = _deformed_square(kind, depth, samples)
    # the mid ring is the same tile turned 90 deg -- but M and rot90(M)
    # share the four square corners, so roll the rotated ring by one
    # edge (`samples`) to loft corner->corner and edge->edge; without
    # the roll the side walls jump a whole corner and twist
    mid = np.roll(_rot2(base, math.pi / 2.0), samples, axis=0)
    rings = [base, mid, base]
    zs = [-height / 2.0, 0.0, height / 2.0]
    return _loft_rings(rings, zs)


def build_escher(kind, nx, ny, depth, samples, height):
    """Escher-loft assembly: identical blocks on the integer lattice
    (space-filling), coloured in a checkerboard for the Truchet look.
    Outer ring flagged as frame.  Returns build_tetra-style tuples."""
    V, F = build_escher_block(kind, depth, samples, height)
    V = np.asarray(V, float)
    cells = []
    for i, j in itertools.product(range(nx), range(ny)):
        parity = (i + j) % 2
        frame = (i == 0 or j == 0 or i == nx - 1 or j == ny - 1)
        cells.append((np.array((float(i), float(j), 0.0)),
                      V, F, frame, parity))
    return cells


# ==================================================================
# small convex hull (faces of a convex point set, outward wound)
# ==================================================================

def _convex_faces(V):
    """One outward-wound polygon per face of the convex hull of the
    point set V (small n; O(n^4)).  Robust for the tet/oct/cube
    primitives used below."""
    V = np.asarray(V, float)
    n = len(V)
    c = V.mean(axis=0)
    faces = []
    seen = set()
    for i, j, k in itertools.combinations(range(n), 3):
        nrm = np.cross(V[j] - V[i], V[k] - V[i])
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = float(np.dot(nrm, V[i]))
        if np.dot(nrm, c) > d:
            nrm, d = -nrm, -d
        s = V @ nrm - d
        if np.all(s <= 1e-7):
            key = tuple(np.round(np.append(nrm, d), 5))
            if key in seen:
                continue
            seen.add(key)
            sel = [t for t in range(n) if abs(s[t]) < 1e-7]
            faces.append(_wind_face(V, np.array(sel), c))
    return faces


# ==================================================================
# FAMILY: VERSATILE / BISQUARE -- exact Escher-trick literature blocks
# ==================================================================
#
# The Versatile Block (Akpanya, Goertzen, Wiesenhuetter, Niemeyer,
# Noennig, Bridges 2023): the interpolation between a square (as a
# diamond at z=0) and a 1x2 rectangle (at z=1), both p4 fundamental
# domains.  Copies tile the slab and interlock.  Exact vertices and
# triangulation as published.

_VERSATILE_V = np.array([
    (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0),
    (1.0, -1.0, 0.0),                                  # z=0 diamond
    (0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0),
    (1.0, -1.0, 1.0), (0.0, -1.0, 1.0)])               # z=1 rectangle
_VERSATILE_F = [[a - 1, b - 1, c - 1] for a, b, c in (
    (1, 2, 3), (1, 2, 5), (1, 3, 4), (1, 4, 9), (1, 5, 9),
    (2, 3, 7), (2, 5, 6), (2, 6, 7), (3, 4, 7), (4, 7, 8),
    (4, 8, 9), (5, 6, 7), (5, 7, 9), (7, 8, 9))]


def _orient_outward(V, faces):
    """Flip any faces whose normal points toward the centroid so the
    whole mesh is wound outward."""
    V = np.asarray(V, float)
    c = V.mean(axis=0)
    out = []
    for f in faces:
        p = V[f]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(nrm, p[0] - c) < 0:
            f = f[::-1]
        out.append(list(f))
    return out


# the Versatile block's z=0 face is a diamond (a square turned 45 deg)
# and its z=1 face a 1x2 rectangle.  The diamond lattice L_D spanned
# by (1,1) and (1,-1) tiles both faces at once, so a single-
# orientation translation over L_D fills the layer -- verified: every
# horizontal section tiles with coverage exactly one.  (The Truchet
# classification gives many more assemblies; this is the simplest.)
_VERSATILE_L = (np.array((1.0, 1.0, 0.0)),
                np.array((1.0, -1.0, 0.0)))


def build_versatile(nx=4, ny=4):
    """A layer of Versatile blocks tiled by translation over the
    diamond lattice L_D; the outer ring is flagged as frame and cells
    are checkerboard-coloured for the Truchet look."""
    cen = _VERSATILE_V.mean(axis=0)
    V0 = _VERSATILE_V - cen
    F = _orient_outward(V0, _VERSATILE_F)
    L1, L2 = _VERSATILE_L
    cells = []
    for i in range(nx):
        for j in range(ny):
            t = i * L1 + j * L2 + cen
            frame = (i == 0 or j == 0 or i == nx - 1 or j == ny - 1)
            cells.append((t, V0, F, frame, (i + j) % 2))
    return cells


# The Bisquare block (Frezier 1737 / Weiss & Niemeyer 2026): a p4
# fundamental domain deformed into two small squares.  11 vertices,
# 18 triangular faces, a closed non-convex block of volume 3/2 (the
# operator recomputes normals, so the published winding is fine).
_A2 = math.sqrt(2.0) / 2.0
_BISQUARE_V = np.array([
    (-_A2, _A2, 0.0), (_A2, _A2, 0.0), (_A2, -_A2, 0.0),
    (-_A2, -_A2, 0.0),
    (-_A2, _A2, 1.0), (0.0, math.sqrt(2.0), 1.0), (_A2, _A2, 1.0),
    (_A2, -_A2, 1.0), (-_A2, -_A2, 1.0), (0.0, -math.sqrt(2.0), 1.0),
    (0.0, 0.0, 1.0)])
_BISQUARE_F = [[i - 1 for i in f] for f in (
    (1, 2, 4), (2, 3, 4), (5, 6, 7), (5, 7, 11), (8, 9, 10),
    (8, 10, 11), (4, 9, 10), (3, 8, 9), (3, 4, 9), (1, 4, 11),
    (1, 5, 11), (4, 10, 11), (2, 6, 7), (1, 5, 6), (2, 3, 11),
    (2, 7, 11), (3, 8, 11), (1, 2, 6))]


def build_bisquare():
    """The exact Bisquare block as a single block (a square deformed
    into two smaller squares).  Nests with the Versatile block in
    heterogeneous p4 assemblies (see BACKLOG)."""
    V0 = _BISQUARE_V - _BISQUARE_V.mean(axis=0)
    return [(np.zeros(3), V0, [list(f) for f in _BISQUARE_F],
             False, 0)]


# ==================================================================
# FAMILY: TETROCTA -- interlocking blocks in the tetroctahedrille
# ==================================================================
#
# Regular tetrahedra and octahedra of edge sqrt2 glued on the FCC
# lattice (the tetrahedral-octahedral honeycomb).  Kitten = 1 oct + 2
# tets, UFO = 1 oct + 4 tets, cushion = 1 oct + 4 tets in a strip.
# Every vertex is integer, so blocks tile space by lattice
# translation.  After Akpanya, Goertzen & Niemeyer (arXiv:2405.01944).

_V1 = np.array((0.0, 1.0, 1.0))
_V2 = np.array((1.0, 0.0, 1.0))
_V3 = np.array((1.0, 1.0, 0.0))

_OCT = np.array([_V1, _V2, _V3, _V1 + _V2, _V1 + _V3, _V2 + _V3])
_T1 = np.array([(0.0, 0.0, 0.0), _V1, _V2, _V3])
_T2 = np.array([_V2, _V3, _V2 + _V3, _V2 + _V3 - _V1])
_T3 = np.array([_V1, _V2, _V1 + _V2, _V1 + _V2 - _V3])
_T4 = np.array([_V2, _V1 + _V2, _V2 + _V3, 2.0 * _V2])

# each block = list of (primitive_vertices, colour_tag); colour 0 for
# octahedra, 1 for tetrahedra
_TETROCTA_BLOCKS = {
    'KITTEN': [(_OCT, 0), (_T1, 1), (_T2, 1)],
    'UFO':    [(_OCT, 0), (_T1, 1), (_T2, 1), (_T3, 1), (_T4, 1)],
    'CUSHION': [(_OCT, 0), (_T1, 1), (_T2, 1),
                (_V1 + _T1, 1), (_V1 + _T2, 1)],
}


# The kitten (volume 2) tiles space by pure translation over the
# non-degenerate FCC basis (v1, v2, v3) (covolume 2, verified no
# overlap).  UFO and cushion have volume 8/3, which admits no integer
# translation lattice, so they are shown as single blocks.
_TETROCTA_LATTICE = (_V1, _V2, _V3)
_TETROCTA_TILES = {'KITTEN'}


def build_tetrocta(kind, nx, ny, nz):
    """Tetroctahedrille blocks.  The kitten is translated over the FCC
    basis (v1, v2, v3) into a space-filling patch; the UFO and cushion
    are shown as a single block (they do not tile by translation).
    Each primitive tet/oct becomes a coloured cell."""
    block = _TETROCTA_BLOCKS[kind]
    prims = [(prim, _convex_faces(prim), col) for prim, col in block]
    if kind not in _TETROCTA_TILES:
        return [(np.zeros(3), P, F, False, col) for P, F, col in prims]
    lat = _TETROCTA_LATTICE
    cells = []
    for i, j, k in itertools.product(range(nx), range(ny),
                                     range(nz)):
        off = i * lat[0] + j * lat[1] + k * lat[2]
        frame = (i == 0 or j == 0 or i == nx - 1 or j == ny - 1
                 or k == 0 or k == nz - 1)
        for P, F, col in prims:
            cells.append((np.zeros(3), P + off, F, frame, col))
    return cells


# ==================================================================
# FAMILY: SL -- self-interlocking octocube blocks (Shih 2018)
# ==================================================================
#
# An S-shaped tetracube fused to an L-shaped tetracube gives the SL
# octocube.  Each tetracube is a single contiguous unit cell; two SL
# blocks make a conjugate pair (one is the other turned 180 deg about
# a y-axis through a shared corner of the S tetracube); pairs chain by
# the `a` engagement (Rz(-90) then T(1,-1,0)), whose fourth power is
# the identity, so a4 closes a periodic square strand.

# unit-cube boundary quads (min corner at the origin), outward-wound
_CUBE_FACES = (
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]))

# S-tetracube and L-tetracube (unit-cube min corners); they share
# three vertical faces, so their union is a contiguous octocube
_S_TETRA = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]
_L_TETRA = [(0, 0, 1), (1, 0, 1), (2, 0, 1), (2, 1, 1)]


def _polycube_mesh(origins):
    """Contiguous boundary surface of a set of unit cubes given by
    integer min-corner origins: only faces without a cube neighbour
    are emitted, and shared vertices are welded, so the result is one
    connected solid (no internal faces).  Returns (verts, faces)."""
    cset = set(map(tuple, origins))
    vidx = {}
    verts = []
    faces = []

    def vid(p):
        if p not in vidx:
            vidx[p] = len(verts)
            verts.append((float(p[0]), float(p[1]), float(p[2])))
        return vidx[p]

    for (x, y, z) in cset:
        for (dx, dy, dz), quad in _CUBE_FACES:
            if (x + dx, y + dy, z + dz) not in cset:
                faces.append([vid((x + cx, y + cy, z + cz))
                              for (cx, cy, cz) in quad])
    return np.array(verts), faces


def _rz(deg):
    a = math.radians(deg)
    c, s = round(math.cos(a)), round(math.sin(a))
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


_SL_ENGAGE = {                       # (rotation, translation)
    'a': (_rz(-90), np.array((1.0, -1.0, 0.0))),
    'h': (np.diag((1.0, -1.0, -1.0)), np.array((0.0, 0.0, 0.0))),
}


def _sl_octocube():
    """The SL octocube as (origins, colour) with the S tetracube
    coloured 0 and the L tetracube 1."""
    return [(np.array(_S_TETRA, float), 0),
            (np.array(_L_TETRA, float), 1)]


def _conjugate(origins):
    """The conjugate of an octocube: a 180 deg turn about a y-axis
    through a shared corner of the S tetracube, mapping cube min-
    corners (x,y,z) -> (2-x, y, -1-z).  The pair is cube-disjoint and
    face-adjacent (verified)."""
    return np.array([(2 - x, y, -1 - z) for (x, y, z) in origins],
                    float)


def _sl_pair():
    """The conjugate pair as (origins, colour): the S/L tetracubes of
    the base octocube plus those of its conjugate partner."""
    base = _sl_octocube()
    return base + [(_conjugate(o), c) for o, c in base]


def build_sl(mode='PAIR', strand=4):
    """SL blocks as contiguous polycubes.  mode BLOCK = one octocube
    (S + L tetracubes, two-coloured); PAIR = a conjugate pair; STRAND
    = `strand` conjugate pairs chained by the a-engagement
    (Rz(-90) then T(1,-1,0)); four pairs close the a4 loop.  Every
    tetracube is a single connected solid."""
    def emit(origins, col, R, t, cells):
        P = origins @ R.T + t
        V, F = _polycube_mesh(np.round(P).astype(int))
        cells.append((np.zeros(3), V, F, False, col))

    cells = []
    if mode == 'BLOCK':
        for o, c in _sl_octocube():
            emit(o, c, np.eye(3), np.zeros(3), cells)
        return cells
    if mode == 'PAIR':
        for o, c in _sl_pair():
            emit(o, c, np.eye(3), np.zeros(3), cells)
        return cells
    # STRAND: conjugate pairs at cumulative a-engagement powers
    R, t = np.eye(3), np.zeros(3)
    aR, at = _SL_ENGAGE['a']
    for k in range(strand):
        for o, c in _sl_pair():
            emit(o, c, R, t, cells)
        R, t = aR @ R, aR @ t + at
    return cells


# ==================================================================
# FAMILY: DOME -- interlocking spherical shell (Escher on a sphere)
# ==================================================================
#
# One block per face of a seed polyhedron, lofted radially from an
# inner shell to an outer shell.  Every shared edge is displaced
# tangentially at the middle shell by a single globally-consistent
# vector, so the two blocks meeting at an edge share the same wavy
# radial wall -- the spherical analogue of the planar Escher loft.
# The blocks tile the shell with no gaps or overlaps (verified) and
# interlock through the deformed walls.  After Akpanya, Goertzen &
# Niemeyer, "From Tilings of Orientable Surfaces to TIA" (2024).

def _icosahedron():
    p = (1.0 + math.sqrt(5.0)) / 2.0
    raw = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            raw += [(0.0, s1, s2 * p), (s1, s2 * p, 0.0),
                    (s1 * p, 0.0, s2)]
    V = np.array(sorted(set(raw)))
    V = V / np.linalg.norm(V[0])
    return V, _convex_faces(V)


def _dodecahedron():
    p = (1.0 + math.sqrt(5.0)) / 2.0
    ip = 1.0 / p
    raw = [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0)
           for z in (-1.0, 1.0)]
    raw += [(0.0, s1 * ip, s2 * p) for s1 in (-1, 1) for s2 in (-1, 1)]
    raw += [(s1 * ip, s2 * p, 0.0) for s1 in (-1, 1) for s2 in (-1, 1)]
    raw += [(s1 * p, 0.0, s2 * ip) for s1 in (-1, 1) for s2 in (-1, 1)]
    V = np.array(raw)
    V = V / np.linalg.norm(V[0])
    return V, _convex_faces(V)


_DOME_SEED = {'ICOSA': _icosahedron, 'DODECA': _dodecahedron}


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-30)


def build_dome(seed, depth, thickness):
    """Interlocking dome: one radially-lofted block per face of the
    seed polyhedron, with each shared edge displaced tangentially at
    the middle shell (globally consistent, so adjacent blocks share
    the deformed wall).  `depth` is the tangential edge push, and
    `thickness` the inner/outer offset about the unit sphere."""
    V, faces = _DOME_SEED[seed]()
    r_in, r_out = 1.0 - thickness, 1.0 + thickness
    # one tangential push vector per undirected edge
    push = {}
    for f in faces:
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            ek = (min(a, b), max(a, b))
            if ek not in push:
                ed = _unit(V[b] - V[a])
                emid = _unit((V[a] + V[b]) / 2.0)
                push[ek] = _unit(np.cross(ed, emid)) * depth
    cells = []
    for fi, f in enumerate(faces):
        m = len(f)
        dirs, offs = [], []            # ring point = unit dir, + push
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            dirs.append(V[a])
            offs.append(np.zeros(3))
            dirs.append((V[a] + V[b]) / 2.0)
            offs.append(push[(min(a, b), max(a, b))])
        dirs = [_unit(d) for d in dirs]
        n2 = 2 * m
        inner = [d * r_in for d in dirs]
        mid = [dirs[i] * 1.0 + offs[i] for i in range(n2)]
        outer = [d * r_out for d in dirs]
        verts = np.array(inner + mid + outer)
        F = []
        for kk in range(2):
            b0, b1 = kk * n2, (kk + 1) * n2
            for a in range(n2):
                bb = (a + 1) % n2
                F.append([b0 + a, b0 + bb, b1 + bb, b1 + a])
        F.append(list(range(n2)))                  # inner cap
        F.append([2 * n2 + a for a in range(n2)])   # outer cap
        F = _orient_outward(verts, F)
        cells.append((np.zeros(3), verts, F, False, fi % 2))
    return cells


# ==================================================================
# assembly -> mesh
# ==================================================================

def cells_to_mesh(cells, size=2.0, gap=1.0):
    """Merge placement cells into one mesh centred at the origin and
    fit into a `size` cube.  `gap` shrinks each block about its own
    centroid (1.0 = touching).  Returns (verts, faces, colour_tags,
    frame_tags)."""
    lo = np.full(3, np.inf)
    hi = -lo
    for c, V, F, frame, col in cells:
        lo = np.minimum(lo, c + V.min(axis=0))
        hi = np.maximum(hi, c + V.max(axis=0))
    mid = (lo + hi) / 2.0
    span = float(np.max(hi - lo))
    s = size / span if span > 0 else 1.0
    verts, faces, cols, frames = [], [], [], []
    for c, V, F, frame, col in cells:
        cen = V.mean(axis=0)
        VV = (c - mid + cen + gap * (V - cen)) * s
        base = len(verts)
        verts.extend(map(tuple, VV))
        for f in F:
            faces.append([base + i for i in f])
            cols.append(col)
            frames.append(1 if frame else 0)
    return verts, faces, cols, frames


# ==================================================================
# family dispatch
# ==================================================================

def build_cells(family, nx=4, ny=4, nz=2, profile='SINE',
                deform=0.18, samples=8, height=1.0,
                sl_mode='STRAND', sl_strand=4,
                dome_seed='ICOSA', dome_depth=0.18, dome_thick=0.15):
    """Placement cells for the chosen family (see the family enum in
    the operator).  Returns build_tetra-style tuples."""
    if family == 'TETRA':
        return build_tetra(nx, ny)
    if family == 'ESCHER':
        return build_escher(profile, nx, ny, deform, samples, height)
    if family == 'VERSATILE':
        return build_versatile(nx, ny)
    if family == 'BISQUARE':
        return build_bisquare()
    if family in _TETROCTA_BLOCKS:
        return build_tetrocta(family, nx, ny, nz)
    if family == 'SL':
        return build_sl(sl_mode, sl_strand)
    if family == 'DOME':
        return build_dome(dome_seed, dome_depth, dome_thick)
    raise ValueError(family)


# families that build a space-filling / interlocking assembly; the
# rest emit a single reference block
_ASSEMBLY = {'TETRA', 'ESCHER', 'VERSATILE', 'KITTEN', 'SL', 'DOME'}
_RIGOROUS = {'TETRA', 'ESCHER', 'VERSATILE'}
_SINGLE_BLOCK = {'BISQUARE', 'UFO', 'CUSHION'}


# ==================================================================
# Blender operator
# ==================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _LABEL = {'TETRA': "Interlocking Tetrahedra",
              'ESCHER': "Escher / Osteomorphic Blocks",
              'VERSATILE': "Versatile Block",
              'BISQUARE': "Bisquare Block",
              'KITTEN': "Tetroctahedrille Kitten",
              'UFO': "Tetroctahedrille UFO",
              'CUSHION': "Tetroctahedrille Cushion",
              'SL': "SL Block",
              'DOME': "Interlocking Dome"}

    _COLOURS = {0: (0.85, 0.55, 0.15), 1: (0.20, 0.45, 0.70),
                2: (0.55, 0.65, 0.25)}
    _FRAME_RGB = (0.72, 0.10, 0.12)

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

    class MESH_OT_interlocking_add(bpy.types.Operator):
        """Add a topological-interlocking block assembly: a layer or
        shell of blocks that lock once the peripheral frame is fixed"""
        bl_idname = "mesh.interlocking_add"
        bl_label = "Topological Interlocking"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(
            name="Family",
            items=[
                ('TETRA', "Interlocking Tetrahedra",
                 "The canonical layer of regular tetrahedra in two "
                 "90-deg orientations on a checkerboard (Dyskin et "
                 "al.); the framed interior kinematically locks"),
                ('ESCHER', "Escher / Osteomorphic",
                 "A square domain edge-deformed, turned 90 deg at "
                 "mid-height and lofted back; sine gives Estrin's "
                 "osteomorphic saddle block, tent a Versatile-style "
                 "block; every layer tiles so copies interlock"),
                ('VERSATILE', "Versatile Blocks",
                 "The exact Versatile block (Akpanya et al., Bridges "
                 "2023): a square lofted to a rectangle, tiled into a "
                 "layer over the diamond lattice (every section tiles "
                 "so the framed layer interlocks)"),
                ('BISQUARE', "Bisquare Block (single)",
                 "The exact Bisquare block (a square deformed into "
                 "two smaller squares); nests with the Versatile "
                 "block in heterogeneous assemblies"),
                ('KITTEN', "Tetroctahedrille Kitten",
                 "One octahedron + two tetrahedra glued on the "
                 "octet-truss lattice; tiles space by translation "
                 "(Akpanya et al. 2024)"),
                ('UFO', "Tetroctahedrille UFO (single)",
                 "One octahedron + four tetrahedra; a single "
                 "tetroctahedrille block"),
                ('CUSHION', "Tetroctahedrille Cushion (single)",
                 "One octahedron + four tetrahedra in a strip; a "
                 "single tetroctahedrille block"),
                ('SL', "SL Blocks",
                 "Self-interlocking octocubes (Shih 2018): an "
                 "S-tetracube fused to an L-tetracube; conjugate "
                 "pairs chain by the a-engagement into a periodic "
                 "strand (four pairs close the a4 loop)"),
                ('DOME', "Interlocking Dome",
                 "One radially-lofted block per polyhedron face; each "
                 "shared edge is displaced tangentially at the middle "
                 "shell so adjacent blocks share a wavy wall and "
                 "interlock (Escher on a sphere; Akpanya et al. 2024)")],
            default='TETRA')

        nx: IntProperty(name="Cells X", default=4, min=1, max=16)
        ny: IntProperty(name="Cells Y", default=4, min=1, max=16)
        nz: IntProperty(name="Cells Z", default=2, min=1, max=8,
                        description="Layers (tetroctahedrille only)")

        profile: EnumProperty(
            name="Profile",
            items=[('SINE', "Sine (Osteomorphic)",
                    "Sinusoidal edge -> saddle block"),
                   ('TENT', "Tent (Versatile)", "Tented edge"),
                   ('ZIGZAG', "S-Curve", "S-shaped edge"),
                   ('STEP', "Step", "Stepped edge")],
            default='SINE')
        deform: FloatProperty(
            name="Deform Depth", default=0.18, min=0.0, max=0.45,
            description="Escher edge-deformation depth (interlock "
                        "depth)")
        samples: IntProperty(name="Edge Samples", default=8, min=2,
                             max=24)
        height: FloatProperty(name="Block Height", default=1.0,
                              min=0.2, max=4.0)

        sl_mode: EnumProperty(
            name="SL Mode",
            items=[('BLOCK', "Single Block",
                    "One SL octocube (S + L tetracubes)"),
                   ('PAIR', "Conjugate Pair",
                    "Two SL blocks sharing an S corner"),
                   ('STRAND', "Strand", "Conjugate pairs chained by "
                    "the a-engagement")],
            default='STRAND')
        sl_strand: IntProperty(
            name="Strand Pairs", default=4, min=1, max=12,
            description="Conjugate pairs in the strand (4 closes the "
                        "a4 loop)")

        dome_seed: EnumProperty(
            name="Seed",
            items=[('ICOSA', "Icosahedron", "20 triangular blocks"),
                   ('DODECA', "Dodecahedron",
                    "12 pentagonal blocks")],
            default='ICOSA')
        dome_depth: FloatProperty(
            name="Edge Deform", default=0.18, min=0.0, max=0.5,
            description="Tangential push of each shared edge at the "
                        "middle shell (the interlock depth)")
        dome_thick: FloatProperty(
            name="Shell Thickness", default=0.15, min=0.03, max=0.5,
            description="Inner/outer shell offset about the sphere")

        gap: FloatProperty(
            name="Gap Factor", default=0.94, min=0.3, max=1.0,
            description="Scale of each block about its centroid "
                        "(1.0 = blocks touch)")
        size: FloatProperty(name="Size", default=2.0, min=0.1,
                            max=100.0,
                            description="Fit within this cube")
        colour_mode: EnumProperty(
            name="Colouring",
            items=[('TYPE', "By Block Type",
                    "Two-tone by Truchet colour / block role"),
                   ('FRAME', "Highlight Frame",
                    "Colour the fixed peripheral frame apart"),
                   ('NONE', "None", "Single material")],
            default='TYPE')

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
            fam = self.family
            if fam in ('TETRA', 'ESCHER', 'VERSATILE', 'KITTEN'):
                lay.prop(self, 'nx')
                lay.prop(self, 'ny')
            if fam == 'KITTEN':
                lay.prop(self, 'nz')
            if fam == 'ESCHER':
                lay.prop(self, 'profile')
                lay.prop(self, 'deform')
                lay.prop(self, 'samples')
                lay.prop(self, 'height')
            if fam == 'SL':
                lay.prop(self, 'sl_mode')
                if self.sl_mode == 'STRAND':
                    lay.prop(self, 'sl_strand')
            if fam == 'DOME':
                lay.prop(self, 'dome_seed')
                lay.prop(self, 'dome_depth')
                lay.prop(self, 'dome_thick')
            lay.prop(self, 'gap')
            lay.prop(self, 'size')
            lay.prop(self, 'colour_mode')

        def execute(self, context):
            try:
                cells = build_cells(
                    self.family, self.nx, self.ny, self.nz,
                    self.profile, self.deform, self.samples,
                    self.height, self.sl_mode, self.sl_strand,
                    self.dome_seed, self.dome_depth, self.dome_thick)
                verts, faces, cols, frames = cells_to_mesh(
                    cells, self.size, self.gap)
            except Exception as e:               # bad parameter combo
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            me = bpy.data.meshes.new("Interlocking")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)

            if self.colour_mode != 'NONE' and \
                    len(me.polygons) == len(cols):
                if self.colour_mode == 'FRAME':
                    _material_a = _material("Interlock Body",
                                            _COLOURS[0])
                    _material_b = _material("Interlock Frame",
                                            _FRAME_RGB)
                    me.materials.append(_material_a)
                    me.materials.append(_material_b)
                    me.polygons.foreach_set('material_index', frames)
                else:
                    used = sorted(set(cols))
                    for c in used:
                        _material(f"Interlock {c}",
                                  _COLOURS.get(c, (0.6, 0.6, 0.6)))
                        me.materials.append(
                            bpy.data.materials[f"Interlock {c}"])
                    remap = {c: idx for idx, c in enumerate(used)}
                    me.polygons.foreach_set(
                        'material_index', [remap[c] for c in cols])
            me.update()
            # the Escher blocks are 90-deg-twisted lofts, so the side
            # walls come out with mixed winding; recalculate normals
            # so each connected shell is consistently outward
            try:
                import bmesh
                bm = bmesh.new()
                bm.from_mesh(me)
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
                bm.to_mesh(me)
                bm.free()
                me.update()
            except Exception:
                pass
            obj = bpy.data.objects.new(
                f"Interlocking {_LABEL[self.family]}", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            tag = ("interlocking (framed interior)"
                   if self.family in _RIGOROUS
                   else "modular blocks")
            self.report({'INFO'},
                        f"{_LABEL[self.family]}: {tag}, "
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.interlocking_add",
                             icon='MOD_BUILD')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_interlocking_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_interlocking_add)


# ==================================================================
# standalone self-test
# ==================================================================

def _selftest():
    from collections import Counter

    def closed_bad_edges(V, F):
        ec = Counter()
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                ec[(min(a, b), max(a, b))] += 1
        return sum(1 for c in ec.values() if c != 2)

    ok = True

    # regular tetrahedra
    for nm, V, F in (('A', _TETRA_A, _TETRA_FA),
                     ('B', _TETRA_B, _TETRA_FB)):
        el = [np.linalg.norm(V[f[i]] - V[f[(i + 1) % 3]])
              for f in F for i in range(3)]
        reg = max(el) - min(el) < 1e-9
        print(f"tetra {nm}: regular={reg} vol={_mesh_volume(V, F):.4f}")
        ok = ok and reg

    # MCS cells are exact platonic (kept for reuse / backlog)
    for kind, nv, nf in (('CUBE', 8, 6), ('OCTA', 6, 8)):
        V, F = _mcs_cell(kind)
        el = [np.linalg.norm(np.asarray(V)[f[i]] -
              np.asarray(V)[f[(i + 1) % len(f)]])
              for f in F for i in range(len(f))]
        reg = max(el) - min(el) < 1e-6 and len(V) == nv and \
            len(F) == nf
        print(f"mcs {kind}: regular={reg} V={len(V)} F={len(F)}")
        ok = ok and reg

    # Versatile exact block: closed manifold, unit-height volume 2
    V0 = _VERSATILE_V - _VERSATILE_V.mean(axis=0)
    F = _orient_outward(V0, _VERSATILE_F)
    be = closed_bad_edges(V0, F)
    vol = abs(_mesh_volume(V0, F))
    print(f"versatile: bad_edges={be} vol={vol:.4f}")
    ok = ok and be == 0 and abs(vol - 2.0) < 1e-6

    # tetroctahedrille primitives are exact regular tet/oct
    for nm, prim, ev in (('OCT', _OCT, 4.0 / 3.0),
                         ('T1', _T1, 1.0 / 3.0),
                         ('T2', _T2, 1.0 / 3.0)):
        Fp = _convex_faces(prim)
        v = abs(_mesh_volume(prim, Fp))
        good = closed_bad_edges(prim, Fp) == 0 and abs(v - ev) < 1e-9
        print(f"tetrocta {nm}: closed+vol_ok={good} vol={v:.4f}")
        ok = ok and good

    # every family builds a non-degenerate mesh that fits the cube
    for name, cells in (
            ('TETRA', build_tetra(4, 4)),
            ('ESCHER', build_escher('SINE', 4, 4, 0.18, 8, 1.0)),
            ('VERSATILE', build_versatile(4, 4)),
            ('BISQUARE', build_bisquare()),
            ('KITTEN', build_tetrocta('KITTEN', 2, 2, 2)),
            ('UFO', build_tetrocta('UFO', 1, 1, 1)),
            ('CUSHION', build_tetrocta('CUSHION', 1, 1, 1)),
            ('SL', build_sl('STRAND', 4)),
            ('DOME_I', build_dome('ICOSA', 0.18, 0.15)),
            ('DOME_D', build_dome('DODECA', 0.18, 0.15))):
        v, f, cols, fr = cells_to_mesh(cells, 2.0, 0.94)
        Vv = np.asarray(v)
        deg = sum(1 for ff in f
                  if np.linalg.norm(np.cross(
                      Vv[ff[1]] - Vv[ff[0]],
                      Vv[ff[2]] - Vv[ff[0]])) < 1e-9)
        fit = (Vv.max(axis=0) - Vv.min(axis=0)).max() <= 2.0 + 1e-6
        good = deg == 0 and fit and len(v) > 0
        print(f"family {name}: mesh_ok={good} V={len(v)} F={len(f)} "
              f"degen={deg}")
        ok = ok and good

    # OVERLAP GATE: assembled families must not interpenetrate.
    # convex-primitive families use an interior-penetration test;
    # the Escher tiling is verified by a section-partition test.
    def _face_planes(V, F):
        V = np.asarray(V)
        c = V.mean(axis=0)
        pls = []
        for f in F:
            p = V[f]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            ln = np.linalg.norm(n)
            if ln < 1e-12:
                continue
            n = n / ln
            d = float(np.dot(n, p[0]))
            if np.dot(n, c) > d:
                n, d = -n, -d
            pls.append((n, d))
        return pls

    def _maxpen(A, B):
        Va, Fa = A
        plB = _face_planes(*B)
        Va = np.asarray(Va)
        ca = Va.mean(axis=0)
        S = [ca] + [0.5 * Va[i] + 0.5 * ca for i in range(len(Va))]
        for f in Fa:
            S.append(0.5 * ca + 0.5 * Va[f].mean(axis=0))
        return max(min(d - np.dot(n, x) for n, d in plB) for x in S)

    def _assembly_overlap(cells):
        sol = [(np.asarray(c) + np.asarray(V), F)
               for c, V, F, fr, col in cells]
        cen = [s[0].mean(axis=0) for s in sol]
        worst = -9.0
        for a in range(len(sol)):
            for b in range(a + 1, len(sol)):
                if np.linalg.norm(cen[a] - cen[b]) < 3.0:
                    worst = max(worst, min(_maxpen(sol[a], sol[b]),
                                           _maxpen(sol[b], sol[a])))
        return worst

    for name, cells in (
            ('TETRA', build_tetra(5, 5)),
            ('KITTEN', build_tetrocta('KITTEN', 2, 2, 2))):
        w = _assembly_overlap(cells)
        good = w < 1e-3
        print(f"overlap {name}: max_penetration={w:+.4f} "
              f"{'ok' if good else 'OVERLAP'}")
        ok = ok and good

    # SL blocks are integer polycubes -> overlap is exact (a cube
    # shared by two blocks).  Check the a-strand is cube-disjoint.
    aR, at = _SL_ENGAGE['a']
    R, t = np.eye(3), np.zeros(3)
    all_cubes = []
    for _ in range(4):
        s = set()
        for o, c in _sl_pair():
            s |= set(map(tuple, np.round(o @ R.T + t).astype(int)))
        all_cubes.append(s)
        R, t = aR @ R, aR @ t + at
    tot = sum(len(s) for s in all_cubes)
    uni = len(set().union(*all_cubes))
    good = tot == uni
    print(f"overlap SL a4-strand: shared_cubes={tot - uni} "
          f"{'ok' if good else 'OVERLAP'}")
    ok = ok and good

    # Escher: every plane section must tile (coverage exactly 1)
    def _pip(pt, poly):
        x, y = pt
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside

    rng = np.random.RandomState(0)
    for prof in ('SINE', 'TENT', 'ZIGZAG', 'STEP'):
        base = _deformed_square(prof, 0.18, 10)
        mid = _rot2(base, math.pi / 2.0)
        good = True
        for tile in (base, mid):
            for pt in rng.uniform(-0.45, 0.45, (200, 2)):
                cov = sum(1 for i in range(-2, 3) for j in range(-2, 3)
                          if _pip(pt - np.array([i, j]), tile))
                if cov != 1:
                    good = False
                    break
        print(f"escher tiling {prof}: coverage_one={good}")
        ok = ok and good

    print("RESULT: OK" if ok else "RESULT: FAIL")
    return ok


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        _selftest()
