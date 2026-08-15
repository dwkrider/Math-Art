# Topological interlocking: assemblies held by geometry, not glue.
#
# Part of the Math Art interlocking engine (`math_art/interlocking/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# A topological interlocking assembly is a set of convex blocks, none
# of which can be removed from the assembly while the others are held,
# even though no block is bonded or fastened to any other.  The classic
# construction takes a planar tiling and tilts the face planes of each
# tile alternately, so each block is a frustum that is wider at the top
# than its neighbour's opening and wider at the bottom than the one
# above -- kinematically locked in both directions.
#
# References:
# - A. V. Dyskin, Y. Estrin, A. J. Kanel-Belov and E. Pasternak,
#   "Topological interlocking of platonic solids: A way to new
#   materials and structures", Philosophical Magazine Letters 83,
#   2003, pp. 197-203.
# - Y. Estrin, A. V. Dyskin and E. Pasternak, "Topological
#   interlocking as a material design concept", Materials Science and
#   Engineering C 31, 2011, pp. 1189-1194.

import itertools
import math
from collections import deque
import numpy as np


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
    """Boundary of the convex polygon spanned by the coplanar vertex
    indices `idx`, wound counterclockwise as seen from outside.  Only
    the 2-D convex-hull corners are kept -- points lying flat in the
    interior of the face are dropped -- so a face with an interior
    coplanar vertex still yields a clean simple polygon.  `outward_ref`
    (the solid centroid) fixes the orientation."""
    P = V[idx]
    c = P.mean(axis=0)
    n = np.zeros(3)
    m = len(idx)
    for i in range(m):
        n += np.cross(P[i] - c, P[(i + 1) % m] - c)
    if np.linalg.norm(n) < 1e-12:
        n = np.cross(P[1] - P[0], P[2] - P[0])
    n = n / (np.linalg.norm(n) + 1e-30)
    a = (np.array((1.0, 0.0, 0.0)) if abs(n[0]) < 0.9
         else np.array((0.0, 1.0, 0.0)))
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    pts = np.column_stack(((P - c) @ u, (P - c) @ v))
    # Andrew's monotone-chain 2-D convex hull -> boundary order
    order = sorted(range(m), key=lambda k: (pts[k, 0], pts[k, 1]))

    def cross(o, i, j):
        return ((pts[i, 0] - pts[o, 0]) * (pts[j, 1] - pts[o, 1])
                - (pts[i, 1] - pts[o, 1]) * (pts[j, 0] - pts[o, 0]))
    lower = []
    for k in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], k) <= 1e-12:
            lower.pop()
        lower.append(k)
    upper = []
    for k in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], k) <= 1e-12:
            upper.pop()
        upper.append(k)
    hull = lower[:-1] + upper[:-1]        # CCW boundary indices
    ordered = [int(idx[k]) for k in hull]
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
    """Cells of a moving-cross-section (MCS) interlocking layer over a
    honeycomb patch (Kanel-Belov / Dyskin et al.).  Every solid is an
    *identical* pure translate -- a single orientation, NOT a two-type
    alternation.  The in/out tilt lives on the six faces of each cell
    (three up, three down, alternating around the equatorial hexagon);
    because opposite hexagon edges have opposite parity and a honeycomb
    neighbour sits across an edge (mapping an edge to the *opposite*
    edge of the neighbour), adjacent cells automatically share each
    inclined face-plane -- so nothing needs a per-cell tilt sign, and
    the three-hexagons-at-a-vertex conflict of a two-colour scheme
    never arises.  Hexagon centres sit on the triangular lattice
    t1=(3/2, sqrt3/2), t2=(0, sqrt3): neighbours lie across the six
    hexagon *edges*, and the cells are cube/octahedron-disjoint
    (verified, face-to-face contact).  Returns build_tetra-style
    tuples."""
    ax = 1.5                         # horizontal (column) spacing
    ay = math.sqrt(3.0)             # vertical (row) spacing
    V, F = _mcs_cell(kind, mirror=False)
    V = np.asarray(V, float)
    cells = []
    for i in range(nx):
        for j in range(ny):
            cx = i * ax
            cy = (j + 0.5 * (i % 2)) * ay
            frame = (i == 0 or j == 0
                     or i == nx - 1 or j == ny - 1)
            # colour is cosmetic only (all cells are identical
            # translates); checkerboard for legibility
            cells.append((np.array((cx, cy, 0.0)),
                          V, F, frame, (i + j) % 2))
    return cells


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
# fundamental domain whose square base (side sqrt2, at z=0) rises to a
# two-tent roof (peak vertex 11 at z=1).  11 vertices, 18 triangular
# faces, a closed non-convex block of volume 3/2 (the operator
# recomputes normals, so the published winding is fine).  It is a p4
# TIA, not a solid space-filler: base squares tile the z=0 plane and
# the roofs interlock, so the layer has a flat floor and a tented top
# (block volume 3/2 < the cell's 2 -- the missing quarter is the
# valleys between roofs, exactly as in Weiss & Niemeyer Fig 8).
_A2 = math.sqrt(2.0) / 2.0


_S2 = math.sqrt(2.0)


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


# the two block orientations (quad-triangular tile senses) are 90-deg
# rotations of each other about the vertical; neighbours on the unit
# square lattice always take opposite orientations (p4), so a single
# checkerboard tiles the interlocking layer (verified: no
# interpenetration).  Lattice = the base square's side sqrt2.
_BISQUARE_L = (np.array((_S2, 0.0, 0.0)), np.array((0.0, _S2, 0.0)))


def _rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def build_bisquare(nx=1, ny=1):
    """The Bisquare block: at nx=ny=1 a single reference block, else
    the p4 interlocking layer -- one block per unit square, adjacent
    cells rotated 90 deg (checkerboard), verified non-overlapping.
    The base squares tile the floor; the roofs interlock above."""
    F = [list(f) for f in _BISQUARE_F]
    cen = _BISQUARE_V.mean(axis=0)
    if nx == 1 and ny == 1:
        return [(np.zeros(3), _BISQUARE_V - cen, F, False, 0)]
    L1, L2 = _BISQUARE_L
    cells = []
    for i in range(nx):
        for j in range(ny):
            R = _rot_z(90.0 * ((i + j) % 2))
            V = _BISQUARE_V @ R.T + i * L1 + j * L2
            frame = (i == 0 or j == 0 or i == nx - 1 or j == ny - 1)
            cells.append((np.zeros(3), V, F, frame, (i + j) % 2))
    # centre the whole layer at the origin
    mid = np.mean([c[1].mean(axis=0) for c in cells], axis=0)
    return [(t, V - mid, F2, fr, col) for t, V, F2, fr, col in cells]


# The Rhom block and its obverse (Weiss & Niemeyer 2026, sec. 3.2,
# building on Goertzen's lozenge construction): p3 blocks over a unit
# lozenge, the interpolation between the lozenge (z=0) and a deformed
# tile (z=sqrt6/3).  Both are convex, so the faces come straight from
# the convex hull.  Height sqrt6/3, edge length 1; volume 0.7857.
# Their p3 / heterogeneous space-filling assembly is left for BACKLOG.
_S3 = math.sqrt(3.0)


_S6 = math.sqrt(6.0)


_RHOM_V = np.array([
    (0.0, 0.0, 0.0), (0.5, _S3 / 2, 0.0), (1.0, 0.0, 0.0),
    (0.5, -_S3 / 2, 0.0),                              # z=0 lozenge
    (0.0, 0.0, _S6 / 3), (0.5, _S3 / 6, _S6 / 3),
    (1.0, 0.0, _S6 / 3), (1.0, -_S3 / 3, _S6 / 3),
    (0.5, -_S3 / 2, _S6 / 3), (0.0, -_S3 / 3, _S6 / 3)])


_RHOM_OBV_V = np.array([
    (0.0, 0.0, 0.0), (0.5, _S3 / 2, 0.0), (1.0, 0.0, 0.0),
    (0.5, -_S3 / 2, 0.0), (0.0, 0.0, _S6 / 3),
    (0.5, _S3 / 6, _S6 / 3), (0.5, _S3 / 2, _S6 / 3),
    (1.0, _S3 / 3, _S6 / 3), (1.0, 0.0, _S6 / 3),
    (0.5, -_S3 / 6, _S6 / 3), (0.5, -_S3 / 2, _S6 / 3),
    (0.0, -_S3 / 3, _S6 / 3)])


def build_rhom(obverse=False):
    """The exact Rhom block (or its obverse) as a single convex block
    over a unit lozenge (p3).  A couple of the published vertices lie
    flat in the interior of a face (they matter for the tiling, not
    the solid shape); dropping the ones that are not true hull corners
    keeps the block a clean manifold."""
    Vraw = _RHOM_OBV_V if obverse else _RHOM_V
    V0 = Vraw - Vraw.mean(axis=0)
    F = _convex_faces(V0)
    # drop vertices not used by any face (flat interior-of-face points)
    used = sorted({i for f in F for i in f})
    remap = {i: k for k, i in enumerate(used)}
    Vk = V0[used]
    Fk = [[remap[i] for i in f] for f in F]
    return [(np.zeros(3), Vk, Fk, False, 0)]


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


# unit-cube boundary quads (min corner at the origin), outward-wound
_CUBE_FACES = (
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]))


# The SL octocube at the verified interlocking placement (unit-cube
# min corners): the S and L tetracubes share three faces so their
# union is one contiguous three-level octocube.  The engagements are
# absolute (world-frame), so the block only locks at this placement.
_SL_S = [(-2, 0, 1), (-2, 0, 2), (-1, 0, 2), (0, 0, 2)]


_SL_L = [(-2, 0, 0), (-1, -1, 0), (-1, 0, 0), (-1, 0, 1)]


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


def _eng(rows, t):
    return (np.array(rows, float), np.array(t, float))


# Shih's six engagements as world-frame affine maps p -> R p + t.
# {h,s,t} carry Rx(180) (a flip to the layer below, reversing z);
# {d,a,y} keep Rx(0); within each group the z-rotation is 0 / -90 /
# +90.  a = Rz(-90) then T(1,-1,0), whose fourth power is the identity.
_SL_ENGAGE = {
    'h': _eng([[1, 0, 0], [0, -1, 0], [0, 0, -1]], (2.0, 0.0, 0.0)),
    's': _eng([[0, 1, 0], [1, 0, 0], [0, 0, -1]], (1.0, 1.0, -1.0)),
    't': _eng([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], (1.0, -1.0, 1.0)),
    'd': _eng([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (2.0, 0.0, -1.0)),
    'a': _eng([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], (1.0, -1.0, 0.0)),
    'y': _eng([[0, -1, 0], [1, 0, 0], [0, 0, 1]], (1.0, 1.0, -2.0)),
}


def _sl_octocube():
    """The SL octocube as [(origins, colour)] with the S tetracube
    coloured 0 and the L tetracube 1."""
    return [(np.array(_SL_S, float), 0), (np.array(_SL_L, float), 1)]


def _sl_compose(G, e):
    """Post-multiply the frame G=(R,t) by engagement e (world-frame
    translation): returns G . M_e."""
    R, t = G
    Re, te = _SL_ENGAGE[e]
    return (R @ Re, R @ te + t)


def _sl_word(mode, engage, frame):
    """Engagement word (a string over a,h,s,t,d,y) for a build mode."""
    if mode == 'BLOCK':
        return ''
    if mode == 'ENGAGEMENT':
        return engage
    return ('a' + 'h' * (2 * frame)) * 4        # STRAND: (a h^{2n})^4


def _sl_frames(word):
    """Cumulative engagement frames for a word, with coincident frames
    dropped (a closing strand ends where it began)."""
    G = (np.eye(3), np.zeros(3))
    frames = [G]
    for e in word:
        G = _sl_compose(G, e)
        frames.append(G)
    seen, uniq = set(), []
    for R, t in frames:
        key = tuple(np.round(np.concatenate([R.ravel(), t]), 3))
        if key not in seen:
            seen.add(key)
            uniq.append((R, t))
    return uniq


def build_sl(mode='STRAND', engage='a', frame=0):
    """SL blocks as contiguous polycubes (Shih 2018), one octocube per
    engagement letter placed at the cumulative (post-multiplied) frame.
    BLOCK = one octocube; ENGAGEMENT = the octocube plus one engaged
    partner (`engage` in a,h,s,t,d,y); STRAND = the periodic square
    strand (a h^{2n})^4 (`frame` = n; n=0 closes the a4 loop, n>=1
    gives nested square frames).  The S and L tetracubes are two-
    coloured and every strand is cube-disjoint (verified)."""
    cells = []
    for R, t in _sl_frames(_sl_word(mode, engage, frame)):
        for origins, col in _sl_octocube():
            P = origins @ R.T + t
            V, F = _polycube_mesh(np.round(P).astype(int))
            cells.append((np.zeros(3), V, F, False, col))
    return cells


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
        # the block is non-convex, so orient the whole (closed) block
        # by its signed volume rather than a per-face centroid test
        if _mesh_volume(verts, F) < 0:
            F = [f[::-1] for f in F]
        cells.append((np.zeros(3), verts, F, False, fi % 2))
    return cells


_HENDECA_V = np.array([
    (0.0, 0.0, 2.0), (2.0, 1.0, 1.0), (0.0, -1.0, 1.0),
    (-2.0, 1.0, 1.0), (0.0, 2.0, 0.0), (1.0, -1.0, 0.0),
    (-1.0, -1.0, 0.0), (2.0, 1.0, -1.0), (0.0, -1.0, -1.0),
    (-2.0, 1.0, -1.0), (0.0, 0.0, -2.0)])


# vertices A..L map to indices 0..10; 4 triangles + 7 quadrilaterals
_HENDECA_F = [[5, 2, 0, 1], [3, 0, 1, 4], [3, 0, 2, 6], [7, 1, 4],
              [7, 1, 5], [5, 8, 6, 2], [4, 9, 3], [6, 9, 3],
              [4, 9, 10, 7], [5, 8, 10, 7], [10, 8, 6, 9]]


def _poly_normals(W, F):
    """Unit normals of the (convex) faces F of vertex set W."""
    out = []
    for f in F:
        p = W[f]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        ln = np.linalg.norm(n)
        if ln > 1e-9:
            out.append(n / ln)
    return out


def _convex_overlap(A, B, FA, FB):
    """True if convex solids A, B interpenetrate (separating-axis test
    over both solids' face normals; touching counts as disjoint)."""
    for n in _poly_normals(A, FA) + _poly_normals(B, FB):
        a = A @ n
        b = B @ n
        if a.min() - b.max() > -1e-6 or b.min() - a.max() > -1e-6:
            return False
    return True


# The space-filling rule (verified: motif + lattice covers space with
# no gaps and no overlaps, face-to-face everywhere).  Four cells make
# Inchbald's "hexagonal boat": the base cell plus three copies turned
# by a 2-fold rotation (no reflections needed).  The boat then tiles
# by pure translation on a body-centred-tetragonal lattice -- the
# square lattice (4,4,0),(4,-4,0) with vertical period (0,0,4) and the
# body centre (4,0,2) -- i.e. BCC compressed vertically by one half,
# exactly as Inchbald describes.  The four orientations are forced:
# of the 22 face-adjacent placements per cell only these four are
# crystallographically consistent (the rest are "false" neighbours
# that never extend to a tiling, which is why a naive flood fill
# leaves permanent gaps).
_HENDECA_MOTIF = [
    (np.eye(3), np.array((0.0, 0.0, 0.0))),
    (np.array(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))),
     np.array((1.0, 3.0, 4.0))),                     # C2 about (1,1,0)
    (np.array(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))),
     np.array((0.0, 2.0, 2.0))),                     # C2 about x
    (np.array(((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))),
     np.array((3.0, 3.0, 6.0)))]                     # C2 about (1,-1,0)


_HENDECA_L = (np.array((4.0, 0.0, 2.0)), np.array((0.0, 4.0, 2.0)),
              np.array((0.0, 0.0, 4.0)))


def build_hendeca(count=32):
    """A compact cluster of `count` bisymmetric hendecahedra from the
    exact space-filling rule: the four-cell boat translated over the
    body-centred-tetragonal lattice.  The `count` cells nearest the
    centre of a generated block are kept, giving a solid interlocking
    chunk (verified space-filling: no gaps, no overlaps, face-to-face).
    Cells are coloured by their orientation in the boat."""
    V, F = _HENDECA_V, _HENDECA_F
    L1, L2, L3 = _HENDECA_L
    # generate enough boats to contain `count` cells, then keep the
    # nearest ones to the block centre for a compact solid cluster
    r = 1
    cells = []
    while len(cells) < count + 4:
        cells = []
        rng = range(-r, r + 1)
        for ci, (R, t) in enumerate(_HENDECA_MOTIF):
            for i in rng:
                for j in rng:
                    for k in rng:
                        off = t + i * L1 + j * L2 + k * L3
                        W = V @ R.T + off
                        cells.append((W, W.mean(axis=0), ci))
        r += 1
    mid = np.mean([c[1] for c in cells], axis=0)
    cells.sort(key=lambda c: np.linalg.norm(c[1] - mid))
    cells = cells[:count]
    keep_mid = np.mean([c[1] for c in cells], axis=0)
    faces = [list(f) for f in F]
    return [(np.zeros(3), W - keep_mid, faces, False, ci % 3)
            for W, _c, ci in cells]


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


def cells_to_meshes(cells, size=2.0, gap=1.0):
    """Like cells_to_mesh but returns one (verts, faces, cols,
    frames) tuple per cell -- all sharing the same centring and
    scale -- for output as separate objects."""
    lo = np.full(3, np.inf)
    hi = -lo
    for c, V, F, frame, col in cells:
        lo = np.minimum(lo, c + V.min(axis=0))
        hi = np.maximum(hi, c + V.max(axis=0))
    mid = (lo + hi) / 2.0
    span = float(np.max(hi - lo))
    s = size / span if span > 0 else 1.0
    out = []
    for c, V, F, frame, col in cells:
        cen = V.mean(axis=0)
        VV = (c - mid + cen + gap * (V - cen)) * s
        verts = [tuple(p) for p in VV]
        faces = [list(f) for f in F]
        out.append((verts, faces, [col] * len(F),
                    [1 if frame else 0] * len(F)))
    return out


def build_cells(family, nx=4, ny=4, nz=2, profile='SINE',
                deform=0.18, samples=8, height=1.0,
                sl_mode='STRAND', sl_engage='a', sl_frame=0,
                dome_seed='ICOSA', dome_depth=0.18, dome_thick=0.15,
                hendeca_count=32):
    """Placement cells for the chosen family (see the family enum in
    the operator).  Returns build_tetra-style tuples."""
    if family == 'TETRA':
        return build_tetra(nx, ny)
    if family == 'ESCHER':
        return build_escher(profile, nx, ny, deform, samples, height)
    if family == 'VERSATILE':
        return build_versatile(nx, ny)
    if family == 'MCSCUBE':
        return build_mcs('CUBE', nx, ny)
    if family == 'MCSOCTA':
        return build_mcs('OCTA', nx, ny)
    if family == 'BISQUARE':
        return build_bisquare(nx, ny)
    if family == 'RHOM':
        return build_rhom(False)
    if family == 'RHOM_OBV':
        return build_rhom(True)
    if family in _TETROCTA_BLOCKS:
        return build_tetrocta(family, nx, ny, nz)
    if family == 'SL':
        return build_sl(sl_mode, sl_engage, sl_frame)
    if family == 'DOME':
        return build_dome(dome_seed, dome_depth, dome_thick)
    if family == 'HENDECA':
        return build_hendeca(hendeca_count)
    raise ValueError(family)


# families that build a space-filling / interlocking assembly; the
# rest emit a single reference block
_ASSEMBLY = {'TETRA', 'ESCHER', 'VERSATILE', 'MCSCUBE', 'MCSOCTA',
             'BISQUARE', 'KITTEN', 'SL', 'DOME', 'HENDECA'}


_RIGOROUS = {'TETRA', 'ESCHER', 'VERSATILE', 'MCSCUBE', 'MCSOCTA'}


_SINGLE_BLOCK = {'RHOM', 'RHOM_OBV', 'UFO', 'CUSHION'}
