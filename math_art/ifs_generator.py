
# Iterated Function System Generator for Blender
#
# Attractors of iterated function systems in three dimensions, in two
# families that need quite different machinery.
#
# RADIX -- self-affine lattice tiles.  An expanding integer matrix
# M (all eigenvalues of modulus > 1) together with a digit set
# D subset of Z^3 that is a COMPLETE RESIDUE SYSTEM for Z^3 / M Z^3
# (so |D| = |det M| = C) determines a unique compact set T with
#
#     M T = T + D,   equivalently   T = union over d in D of M^-1 (T+d)
#
# and that T tiles R^3 by the lattice Z^3.  The level-k approximation
# is exact rather than sampled: iterate the integer point set
# S_0 = {0}, S_(j+1) = D + M S_j (which has exactly C^k points after k
# steps, distinctness guaranteed by the residue condition), mesh those
# unit cubes with an exterior-face walker, and apply the single linear
# map M^-k.  A linear map preserves watertightness, so the result is a
# closed surface of volume exactly 1 whatever the level.
#
#   ABC          the Thuswaldner-Zhang normal form for collinear digit
#                sets: M is the companion matrix of
#                lambda^3 + A lambda^2 + B lambda + C with digits
#                j*e1, j = 0 .. C-1.  Such a tile is a closed 3-ball
#                when 1 = A <= B < C AND it has 14 neighbours
#                (Thuswaldner-Zhang Thm 1.1) -- the neighbour count is
#                a hypothesis, not a consequence, and their Remark 1.4
#                conjectures the opposite without it.  Their Remark 1.3
#                turns the count into arithmetic, which this module
#                evaluates and reports per tile.
#   TWINDRAGON   Bandt's seven three-dimensional twindragons: |det| = 2,
#                two digits, characteristic polynomial
#                lambda^3 - a lambda^2 - b lambda - 2 for the seven
#                (a, b) pairs that give distinct tiles.
#   CUBE         M = 2I with the eight digits {0,1}^3 -- the unit cube,
#                the degenerate case, and the base for the gaskets.
#
# A `holes` count drops the last h digits at every level, giving
# (C-h)^k cells instead of C^k: the same "gasket" semantics the sibling
# 2-D Fractal Rep-Tile generator uses, and the way to get Sierpinski-
# like sets out of a tile family.
#
# IFS -- general affine attractors.  Contractive maps w_i(x) = A_i x +
# b_i (all with largest singular value < 1) have, by Hutchinson's
# theorem, a unique compact attractor.  Three ways to render it:
#
#   SOLIDS   deterministic: apply every map k times to a seed solid and
#            emit the m^k transformed copies.  Exact, and the copies
#            meet at points for the Sierpinski sets.
#   VOXEL    chaos game into a voxel grid, then the same exterior-face
#            walker -- watertight and blocky, the printable option.
#   ISO      chaos game into a density grid, blurred and contoured by
#            marching tetrahedra -- smooth and cloud-like.
#
# Note on anisotropy: in three dimensions the eigenvalues of M
# generally have DIFFERENT moduli, so M^-k contracts unevenly and the
# level-k approximation is genuinely thin in the slow direction.  That
# is not a defect -- it is what "self-affine" rather than "self-similar"
# means.  (A self-affine tile is conjugate to a self-similar one if and
# only if all eigenvalues of M share a modulus.)
#
# Geometry only; materials and rendering are left to Blender.
#
# References:
# - C. Bandt, Mai The Duy and M. Mesing, "Three-Dimensional Fractals",
#   The Mathematical Intelligencer 32, 2010, pp. 12-18.
#   doi:10.1007/s00283-009-9110-6
# - C. Bandt, "Self-similar sets 5. Integer matrices and fractal
#   tilings of R^n", Proceedings of the American Mathematical Society
#   112, 1991, pp. 549-562 -- the integer-matrix plus residue-digit-set
#   theorem behind every radix tile here.
# - C. Bandt, "Combinatorial topology of three-dimensional self-affine
#   tiles", arXiv:1002.0710, 2010 -- the seven twindragon cases.
# - J. M. Thuswaldner and S.-Q. Zhang, "On self-affine tiles that are
#   homeomorphic to a ball", arXiv:2107.12076 -- the ABC normal form.
# - J. E. Hutchinson, "Fractals and self similarity", Indiana
#   University Mathematics Journal 30, 1981 -- existence and uniqueness
#   of the attractor of a contractive IFS.
# - M. F. Barnsley, "Fractals Everywhere", 2nd ed., Academic Press,
#   1993 -- the chaos game, and the fern whose (two-dimensional) maps
#   are offered here embedded in the z = 0 plane.

bl_info = {
    "name": "Iterated Function System Generator",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Iterated Function System",
    "description": "Self-affine lattice tiles from an expanding integer "
                   "matrix, and affine IFS attractors in two and three "
                   "dimensions",
    "category": "Add Mesh",
}

import math

import numpy as np


def _toolkit():
    """The sibling `minsurf` engine package supplies marching_tets."""
    try:
        from . import minsurf as mst
    except ImportError:
        import minsurf as mst
    return mst


# ==========================================================================
# Radix tiles: expanding integer matrix + complete residue digit set
# ==========================================================================

def companion(c2, c1, c0):
    """The companion matrix of lambda^3 + c2 lambda^2 + c1 lambda + c0,
    whose determinant is -c0."""
    return np.array([[0, 0, -c0],
                     [1, 0, -c1],
                     [0, 1, -c2]], dtype=np.int64)


def is_expanding(M, tol=1e-9):
    """Every eigenvalue of modulus > 1 -- the condition that makes the
    radix representation converge."""
    return bool(np.min(np.abs(np.linalg.eigvals(
        np.asarray(M, dtype=float)))) > 1.0 + tol)


def is_residue_system(M, digits, tol=1e-9):
    """True when no two digits are congruent mod M Z^3, i.e. the digit
    set is a complete residue system for Z^3 / M Z^3.  This is exactly
    what makes every radix string denote a distinct lattice point, and
    hence what makes |S_k| = C^k."""
    Mi = np.linalg.inv(np.asarray(M, dtype=float))
    D = np.asarray(digits, dtype=float)
    for a in range(len(D)):
        for b in range(a + 1, len(D)):
            v = Mi @ (D[a] - D[b])
            if np.all(np.abs(v - np.round(v)) < tol):
                return False
    return True


def radix_points(M, digits, level, holes=0):
    """The level-k integer point set S_k defined by
    S_0 = {0}, S_(j+1) = D + M S_j.

    With `holes` > 0 the last `holes` digits are dropped at every
    level, giving (C-h)^k points -- a gasket inside the tile."""
    M = np.asarray(M, dtype=np.int64)
    D = np.asarray(digits, dtype=np.int64)
    if holes:
        D = D[:max(1, len(D) - int(holes))]
    S = np.zeros((1, 3), dtype=np.int64)
    for _ in range(int(level)):
        # S <- D + M S, as one broadcast; int64 throughout because the
        # coordinates grow like ||M||^k and would overflow int32
        MS = S @ M.T
        S = (MS[:, None, :] + D[None, :, :]).reshape(-1, 3)
    return S


_FACE_DIRS = (
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
)


def edge_stats(faces):
    """(boundary, non_manifold) edge counts.

    A boundary edge (one incident face) means the surface has a hole --
    always a bug here.  An edge with FOUR incident faces means two cubes
    of the set meet only along that edge; the surface still encloses its
    solid, but it is not a manifold there.  That happens for real in
    these families -- Sierpinski-like sets touch at edges and corners by
    construction -- so it is reported, not treated as an error."""
    if not len(faces):
        return 0, 0
    e = []
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            e.append((a, b) if a < b else (b, a))
    arr = np.asarray(e, dtype=np.int64)
    _, counts = np.unique(arr, axis=0, return_counts=True)
    return (int(np.sum(counts == 1)), int(np.sum(counts > 2)))


def blur_density(dens):
    """Separable 3-tap blur with ZERO padding, and an empty shell left
    around the grid.

    np.roll would wrap, bleeding density from one face of the box to
    the opposite one; and unless the outermost layer is empty the
    contour runs into the sample box and comes out as an open surface
    with a boundary, whose normals then mean nothing."""
    out = dens.astype(float)
    for axis in (0, 1, 2):
        lo = np.zeros_like(out)
        hi = np.zeros_like(out)
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(1, None)
        sl_hi[axis] = slice(None, -1)
        lo[tuple(sl_hi)] = out[tuple(sl_lo)]
        hi[tuple(sl_lo)] = out[tuple(sl_hi)]
        out = (out + lo + hi) / 3.0
    out[0, :, :] = out[-1, :, :] = 0.0
    out[:, 0, :] = out[:, -1, :] = 0.0
    out[:, :, 0] = out[:, :, -1] = 0.0
    return out


def orient_outward(verts, faces):
    """Reverse every face when the mesh is inside-out.

    The divergence theorem gives the enclosed volume as
    sum over faces of (1/6) v0 . ((v1-v0) x (v2-v0)); a closed surface
    whose normals point outward has it positive.  An orientation-
    REVERSING transform -- and det M is negative for every ABC tile, so
    M^-k reverses at odd levels -- silently turns a solid inside out
    without changing a single vertex."""
    V = np.asarray(verts, dtype=float)
    tot = 0.0
    for f in faces:
        f = list(f)
        for i in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[i]], V[f[i + 1]]
            tot += float(np.dot(a, np.cross(b - a, c - a)))
    if tot < 0.0:
        return [tuple(reversed(tuple(f))) for f in faces]
    return [tuple(f) for f in faces]


def fill_pinholes(cells, res, need=5):
    """Fill empty cells that have `need` or more of their six face
    neighbours occupied.

    A finite sample leaves isolated gaps inside a solid region; they
    are sampling noise, and each one is a void a slicer would try to
    print around.  The threshold is deliberately conservative -- five
    of six means the cell is all but enclosed -- so this closes
    pinholes without dilating the set or bridging a genuine thin gap
    (which would inflate the volume, the trap with rasterising these
    tiles)."""
    g = np.zeros((res, res, res), dtype=bool)
    g[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    n = np.zeros((res, res, res), dtype=np.int8)
    for axis in (0, 1, 2):
        for shift in (1, -1):
            n += np.roll(g, shift, axis=axis)
    g |= (~g) & (n >= int(need))
    return np.argwhere(g).astype(np.int64)


def _packer(pts, pad=2):
    """A collision-free integer key for lattice points, or None when
    the bounding box is too large to pack into an int64."""
    lo = pts.min(axis=0) - pad
    span = (pts.max(axis=0) + pad) - lo + 1
    if int(span[0]) * int(span[1]) * int(span[2]) > (1 << 62):
        return None
    return lo, span.astype(np.int64)


def _pack(pts, lo, span):
    q = pts - lo
    return (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]


def voxel_surface(cells):
    """Watertight exterior surface of a set of unit cubes on the
    integer lattice: only faces between an occupied cell and an empty
    neighbour are emitted, with shared vertices.  Returns integer
    vertices and an (n, 4) array of quad faces.

    Same walker the Fractal Sponge generator uses, but vectorised --
    the occupancy and vertex-welding lookups are sorted-key searches
    rather than Python dict hits, because these sets run to hundreds of
    thousands of cells."""
    cells = np.unique(np.asarray(cells, dtype=np.int64), axis=0)
    if not len(cells):
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 4),
                                                          dtype=np.int64)
    pk = _packer(cells)
    if pk is None:                     # pathological span: sparse path
        return _voxel_surface_slow(cells)
    lo, span = pk
    keys = np.sort(_pack(cells, lo, span))

    quads = []
    for (d, corners) in _FACE_DIRS:
        nb = cells + np.asarray(d, dtype=np.int64)
        nk = _pack(nb, lo, span)
        pos = np.searchsorted(keys, nk)
        pos_c = np.clip(pos, 0, len(keys) - 1)
        occupied = keys[pos_c] == nk
        free = cells[~occupied]
        if not len(free):
            continue
        quads.append(np.stack(
            [free + np.asarray(c, dtype=np.int64) for c in corners],
            axis=1))
    if not quads:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 4),
                                                          dtype=np.int64)
    Q = np.concatenate(quads, axis=0)              # (nf, 4, 3)
    flat = Q.reshape(-1, 3)
    verts, inv = np.unique(flat, axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 4)


def _voxel_surface_slow(cells):
    """Dict-based fallback for point sets whose bounding box will not
    pack into an int64 key."""
    occ = set(map(tuple, cells.tolist()))
    verts, vid, faces = [], {}, []

    def vertex(key):
        i = vid.get(key)
        if i is None:
            i = len(verts)
            vid[key] = i
            verts.append(key)
        return i

    for (cx, cy, cz) in sorted(occ):
        for (d, corners) in _FACE_DIRS:
            if (cx + d[0], cy + d[1], cz + d[2]) in occ:
                continue
            faces.append([vertex((cx + a, cy + b, cz + c))
                          for (a, b, c) in corners])
    return (np.asarray(verts, dtype=np.int64),
            np.asarray(faces, dtype=np.int64))


# --- what the tile actually is, in closed form -----------------------------

def tile_support_bbox(M, digits, terms=400, tol=1e-15):
    """Exact axis-aligned bounding box of the attractor of
    w_d(x) = M^-1 (x + d).

    Every point of the attractor is a convergent radix series
    sum_(j>=1) M^-j d_j with each d_j free, so the support function
    along an axis separates term by term:

        max coordinate = sum_(j>=1) max over d of (M^-j d)

    and likewise the min.  That gives the true extent in closed form,
    which is the only honest yardstick for how far a level-k
    approximation still has to go."""
    M = np.asarray(M, dtype=float)
    D = np.asarray(digits, dtype=float)
    Mi = np.linalg.inv(M)
    P = Mi.copy()
    lo = np.zeros(3)
    hi = np.zeros(3)
    for _ in range(int(terms)):
        W = D @ P.T
        hi += W.max(axis=0)
        lo += W.min(axis=0)
        if float(np.max(np.abs(W))) < tol:
            break
        P = P @ Mi
    return lo, hi


# --- presets --------------------------------------------------------------
# (key, label, M, digits).  Verified expanding with a complete residue
# system by the self-test, not by assertion here.

def _abc(A, B, C):
    return companion(A, B, C), np.array([(j, 0, 0) for j in range(C)],
                                        dtype=np.int64)


def _twindragon(a, b):
    # characteristic polynomial lambda^3 - a lambda^2 - b lambda - 2
    return (companion(-a, -b, -2),
            np.array([(0, 0, 0), (1, 0, 0)], dtype=np.int64))


# The cube's digits are ordered so that dropping them from the END
# always leaves an affinely three-dimensional set: a naive i,j,k
# ordering puts all four i = 0 digits last, so a four-hole gasket
# collapses to a flat sheet.  This order takes the first removals off a
# space diagonal instead, and keeps rank 3 down to four digits (whose
# attractor is a tetrahedral gasket).
# The first four are one of the cube's two inscribed tetrahedra, so a
# four-hole gasket is the Sierpinski tetrahedron rather than a slab.
_CUBE_DIGITS = np.array([(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1),
                         (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                        dtype=np.int64)

RADIX_PRESETS = {
    # ABC (1,1,2) is deliberately absent: it satisfies the normal form
    # 1 <= A <= B < C, so it IS an ABC tile, but it is also a
    # twindragon (two collinear digits, |det| = 2) and is congruent to
    # case C -- diag(-1,1,-1) conjugates one matrix to minus the other
    # -- so it would be twindragon C shipped twice, turned around.
    'ABC_123': ("ABC tile (1,2,3)", _abc(1, 2, 3), (1, 2, 3)),
    'ABC_124': ("ABC tile (1,2,4)", _abc(1, 2, 4), (1, 2, 4)),
    'ABC_128': ("ABC tile (1,2,8) - self-similar", _abc(1, 2, 8),
                (1, 2, 8)),
    'ABC_136': ("ABC tile (1,3,6)", _abc(1, 3, 6), (1, 3, 6)),
    'ABC_134': ("ABC tile (1,3,4)", _abc(1, 3, 4), (1, 3, 4)),
    'ABC_223': ("ABC tile (2,2,3)", _abc(2, 2, 3), (2, 2, 3)),
    # case A is Bandt's own non-fractal example: in this lattice basis
    # the tile is exactly the unit cube
    'TWIN_A': ("Twindragon A (0,0) - non-fractal (a cube)",
               _twindragon(0, 0), 'A'),
    'TWIN_B': ("Twindragon B (-1,1)", _twindragon(-1, 1), 'B'),
    'TWIN_C': ("Twindragon C (1,-1)", _twindragon(1, -1), 'C'),
    'TWIN_D': ("Twindragon D (0,1)", _twindragon(0, 1), 'D'),
    'TWIN_E': ("Twindragon E (2,-2)", _twindragon(2, -2), 'E'),
    'TWIN_F': ("Twindragon F (1,0)", _twindragon(1, 0), 'F'),
    'TWIN_G': ("Twindragon G (0,2)", _twindragon(0, 2), 'G'),
    'CUBE': ("Cube (2I, 8 digits)",
             (2 * np.eye(3, dtype=np.int64), _CUBE_DIGITS),
             'CUBE'),
}


def attractor_rank(M, digits, terms=16):
    """Dimension of the affine hull of the attractor of
    w_d(x) = M^-1 (x + d).

    NOT the rank of the digit set: every point of the attractor is
    sum_(j>=1) M^-j d_j, so the hull is the span of
    {M^-j (d - d_0) : j >= 1}, the smallest M^-1-invariant subspace
    containing the digit differences.  The twindragons have only two,
    collinear, digits and are still solidly three-dimensional, because
    M^-1 turns that one direction through the other two."""
    M = np.asarray(M, dtype=float)
    D = np.asarray(digits, dtype=float)
    if len(D) < 2:
        return 0
    V = D[1:] - D[0]
    Mi = np.linalg.inv(M)
    P = Mi.copy()
    rows = []
    for _ in range(int(terms)):
        W = V @ P.T
        n = np.linalg.norm(W, axis=1, keepdims=True)
        # normalise: M^-j shrinks geometrically, and un-normalised rows
        # would fall under the rank tolerance and be miscounted
        rows.append(W / np.maximum(n, 1e-300))
        P = P @ Mi
    return int(np.linalg.matrix_rank(np.vstack(rows), tol=1e-8))


def abc_has_14_neighbours(A, B, C):
    """Thuswaldner-Zhang Remark 1.3: for a collinear-digit tile with
    1 <= A <= B < C, T has exactly 14 neighbours iff

        1 <= A < B < C  and  B >= 2A-1, C >= 2(B-A)+2,   or
        1 <= A < B < C  and  B <  2A-1, C >= A+B-2.

    Note both branches need A < B strictly, so A = B never gives 14."""
    A, B, C = int(A), int(B), int(C)
    if not (1 <= A < B < C):
        return False
    if B >= 2 * A - 1:
        return C >= 2 * (B - A) + 2
    return C >= A + B - 2


def abc_is_ball(A, B, C):
    """Thuswaldner-Zhang Theorem 1.1: a 3-dimensional self-affine tile
    with collinear digit set whose characteristic polynomial satisfies
    1 = A <= B < C, AND which has 14 neighbours, is a 3-ball."""
    return int(A) == 1 and abc_has_14_neighbours(A, B, C)


# Bandt, "Combinatorial topology of three-dimensional self-affine
# tiles", Prop 6.4 (neighbours, uncountable boundary sets, faces) and
# Prop 7.2 / Examples 6.3, 10.2, 12.4 (point neighbours, topology).
#
# Cases F and G are deliberately omitted.  Bandt states 48 and 76
# neighbours for them but also says "we shall provide no details for
# the complicated twindragons F and G", and warns in his neighbour
# algorithm that "for F and G there are rare outliers on the thin
# fibres, and an exact estimate is needed" -- so those two numbers are
# not repeated here as though they were settled.
TWINDRAGON_FACTS = {
    'A': (26, 18, 6, 8, "not a fractal: the unique self-similar 3-D "
                        "lattice tile with two pieces"),
    'B': (18, 14, 14, 4, "truncated-octahedron face pattern; ball "
                         "conjectured, not proved"),
    'C': (20, 12, 12, 8, "rhombic face pattern; interior proved "
                         "connected; ball conjectured"),
    'D': (34, 14, 14, 12, "interior not simply connected, so NOT a "
                          "ball"),
    'E': (34, 32, 12, 2, "interior not simply connected, so NOT a "
                         "ball"),
}


def radix_topology(meta):
    """A one-line, citable statement of what is known about a preset's
    topology -- or nothing, when the papers do not settle it."""
    if isinstance(meta, tuple) and len(meta) == 3:
        A, B, C = meta
        if abc_is_ball(A, B, C):
            return (f"proved a 3-ball with 14 neighbours "
                    f"(Thuswaldner-Zhang Thm 1.1); truncated-octahedron "
                    f"CW structure, 24 vertices / 36 edges / 14 faces")
        if abc_has_14_neighbours(A, B, C):
            return ("14 neighbours, so its faces are 2-balls meeting in "
                    "1-balls (Thuswaldner-Zhang Prop 2.6); the ball "
                    "theorem needs A = 1, so it does not apply")
        return ("more than 14 neighbours: the ball theorem does not "
                "apply, and Thuswaldner-Zhang Remark 1.4 conjectures "
                "such tiles are not balls")
    if meta in TWINDRAGON_FACTS:
        nb, unc, faces, pts, note = TWINDRAGON_FACTS[meta]
        return (f"{nb} neighbours ({faces} faces, {pts} point "
                f"neighbours) -- {note} [Bandt, Prop 6.4/7.2]")
    if meta == 'CUBE':
        return "the unit cube: 6 faces, 12 edge and 8 point neighbours"
    return ""


def max_holes(M, digits):
    """Most digits that can be dropped from the end while the attractor
    still fills three dimensions.  Dropping past this point collapses
    it to a sheet, a line or a point."""
    D = np.asarray(digits, dtype=np.int64)
    for h in range(len(D) - 1, -1, -1):
        keep = D[:len(D) - h]
        if len(keep) >= 2 and attractor_rank(M, keep) == 3:
            return h
    return 0

MAX_CELLS = 300000


def max_level(ndigits):
    """Largest level whose point count stays inside the cell budget."""
    if ndigits < 2:
        return 1
    return max(1, int(math.floor(math.log(MAX_CELLS)
                                 / math.log(ndigits))))


def default_level(ndigits):
    """A level landing in the 30k-300k cell band, where these tiles
    read as their limit shape without costing a minute."""
    if ndigits < 2:
        return 1
    return max(1, int(round(math.log(60000) / math.log(ndigits))))


def build_radix(preset='ABC_124', level=0, holes=0, custom=None,
                output='VOXEL', resolution=128, points=800000, seed=0,
                largest_only=False, scale=1.0):
    """Mesh a self-affine radix tile.  Returns (verts, faces, info).

    Three renderings, and the choice matters:

    VOXEL / SMOOTH sample the ATTRACTOR.  Because T tiles R^3 by Z^3,
    the invariant measure of w_d(x) = M^-1 (x + d) with equal weights
    is Lebesgue measure restricted to T, so the chaos game samples the
    tile uniformly and a voxel grid over its exact bounding box
    recovers the solid.  This is what the published pictures of these
    tiles are.

    EXACT is the level-k union of cubes: S_k has exactly C^k points and
    the body has volume exactly 1.  It is the mathematically exact
    object -- but M^-k maps the unit cube to a parallelepiped whose
    aspect ratio grows like (max|lambda| / min|lambda|)^k, so beyond a
    few levels the cells are plates or needles and the surface reads as
    a laminate rather than a solid.  Raising the level improves the
    shape and worsens the lamination at the same time, so this mode is
    offered with its aspect ratio and its shortfall reported, not as
    the default."""
    if preset == 'CUSTOM' and custom is not None:
        M, D = custom
    else:
        M, D = RADIX_PRESETS[preset][1]
    M = np.asarray(M, dtype=np.int64)
    D = np.asarray(D, dtype=np.int64)
    if not is_expanding(M):
        raise ValueError("the matrix is not expanding (it needs every "
                         "eigenvalue of modulus > 1)")
    if not is_residue_system(M, D):
        raise ValueError("the digits are not a complete residue system "
                         "for Z^3 / M Z^3 (two of them are congruent)")
    C = len(D)
    hmax = max_holes(M, D)
    want_holes = int(max(0, int(holes)))
    holes = min(want_holes, hmax)
    kept = C - holes
    Dk = D[:kept]

    detM = abs(int(round(np.linalg.det(M.astype(float)))))
    ev = np.linalg.eigvals(M.astype(float))
    lo_t, hi_t = tile_support_bbox(M, Dk)
    true_span = hi_t - lo_t
    meta = (RADIX_PRESETS[preset][2]
            if preset in RADIX_PRESETS and len(RADIX_PRESETS[preset]) > 2
            else None)
    info = {'digits': C, 'kept': kept, 'det': detM, 'eigenvalues': ev,
            'topology': radix_topology(meta),
            'holes': holes, 'holes_clamped': want_holes > hmax,
            'max_holes': hmax, 'true_span': true_span,
            'output': output}

    if output == 'EXACT':
        lvl = int(level) if level > 0 else default_level(kept)
        lvl = max(1, min(lvl, max_level(kept)))
        S = radix_points(M, D, lvl, holes)
        verts_i, faces = voxel_surface(S)
        if not len(faces):
            raise ValueError("the tile came out empty")
        A = np.linalg.inv(np.linalg.matrix_power(M.astype(float), lvl))
        verts = verts_i.astype(float) @ A.T
        sv = np.linalg.svd(A, compute_uv=False)
        # det M is negative for every ABC tile, so M^-k turns the
        # cubes inside out at odd levels
        faces = orient_outward(verts, _as_quads(faces))
        span = verts.max(axis=0) - verts.min(axis=0)
        info.update({
            'level': lvl, 'cells': len(S),
            'volume': len(S) / float(detM) ** lvl,
            'cell_aspect': float(sv[0] / max(sv[-1], 1e-300)),
            'fidelity': float(np.min(span / np.maximum(true_span,
                                                       1e-12)))})
        return center_fit(verts, scale), faces, info

    # --- attractor sampling -------------------------------------------
    Mi = np.linalg.inv(M.astype(float))
    maps = [(Mi, Mi @ d.astype(float), 1.0) for d in Dk]
    if not spectrally_contractive(maps):
        raise ValueError("M^-1 is not contractive -- the matrix must "
                         "be expanding")
    # the transient must outlast ||M^-n||, which decays only like
    # min|lambda|^-n -- 0.94^n for twindragon G, so 20 steps is nowhere
    # near enough to forget the starting point
    P = chaos_game(maps, points=points, seed=seed, transient=300)
    res = int(resolution)
    pad = 0.02 * float(np.max(true_span))
    lo = lo_t - pad
    s = (float(np.max(true_span)) + 2.0 * pad) / res
    idx = np.floor((P - lo) / s).astype(np.int64)
    idx = idx[np.all((idx >= 0) & (idx < res), axis=1)]
    if not len(idx):
        raise ValueError("the attractor sample fell outside its own "
                         "bounding box")

    if output == 'VOXEL':
        cells = fill_pinholes(np.unique(idx, axis=0), res)
        verts_i, faces = voxel_surface(cells)
        verts = lo + verts_i.astype(float) * s
        span = verts.max(axis=0) - verts.min(axis=0)
        info.update({'cells': len(cells), 'points': len(P),
                     'resolution': res,
                     'fidelity': float(np.min(
                         span / np.maximum(true_span, 1e-12)))})
        return center_fit(verts, scale), _as_quads(faces), info

    # SMOOTH: density grid -> blur -> marching tetrahedra
    dens = np.zeros((res, res, res), dtype=float)
    np.add.at(dens, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    dens = blur_density(dens)
    occupied = dens[dens > 0]
    t = float(np.quantile(occupied, 0.02)) if len(occupied) else 0.0
    hi = lo + s * res

    def field(X, Y, Z):
        ix = np.clip(((X - lo[0]) / s).astype(np.int64), 0, res - 1)
        iy = np.clip(((Y - lo[1]) / s).astype(np.int64), 0, res - 1)
        iz = np.clip(((Z - lo[2]) / s).astype(np.int64), 0, res - 1)
        return t - dens[ix, iy, iz]

    mst = _toolkit()
    verts, tris = mst.marching_tets(field, lo, hi, (res, res, res))
    if not len(tris):
        raise ValueError("the contour came out empty -- try more "
                         "points or a lower resolution")
    if largest_only:
        verts, tris = keep_largest(verts, tris)
    span = verts.max(axis=0) - verts.min(axis=0)
    info.update({'points': len(P), 'resolution': res, 'tris': len(tris),
                 'fidelity': float(np.min(
                     span / np.maximum(true_span, 1e-12)))})
    return (center_fit(verts, scale),
            orient_outward(verts, [tuple(int(i) for i in f)
                                   for f in tris]), info)


def _as_quads(faces):
    """The vectorised walker returns an (n, 4) array; Blender wants a
    list of tuples."""
    return [tuple(int(i) for i in f) for f in np.asarray(faces)]


# ==========================================================================
# General affine IFS
# ==========================================================================

_TETRA = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
                  dtype=float)
# wound so the normals face outward: a signed-volume check in the
# self-test keeps them that way
_TETRA_F = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
_OCTA = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1)], dtype=float)
_OCTA_F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
           (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
_CUBE_V = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1)
                    for z in (-1, 1)], dtype=float)
_CUBE_F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
           (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]

SEEDS = {'CUBE': (_CUBE_V, _CUBE_F), 'TETRA': (_TETRA, _TETRA_F),
         'OCTA': (_OCTA, _OCTA_F)}


def _uniform(vs, s):
    """Maps x -> s x + (1-s) v for each v: the classic 'shrink toward
    each vertex' system whose attractor is the Sierpinski set."""
    return [(s * np.eye(3), (1.0 - s) * np.asarray(v, dtype=float),
             1.0) for v in vs]


def _rot(axis, theta):
    """Rodrigues rotation matrix about `axis` by `theta`."""
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)
    K = np.array([[0.0, -u[2], u[1]],
                  [u[2], 0.0, -u[0]],
                  [-u[1], u[0], 0.0]])
    return (np.eye(3) + math.sin(theta) * K
            + (1.0 - math.cos(theta)) * (K @ K))


def _bmm_sierpinski(sides=3, ratio=2.0 / 3.0):
    """Bandt, Mai The Duy and Mesing's three-dimensional modification of
    Sierpinski's triangle (their Figure 7).

    Their construction verbatim: the fixed points c_i sit symmetrically
    about 0 in the x1,x2-plane with c_1 = (1,0,0), and each f_i is the
    homothety of ratio r toward c_i composed with a 90 degree rotation
    about the axis [0, c_i] --

        f_1(x1,x2,x3) = (r x1 + 1 - r,  -r x3,  r x2)
        f_2 = t f_1 t^-1,   f_3 = t^-1 f_1 t

    with t the 120 degree rotation in the x1,x2-plane.  At r = 1/2 the
    attractor is a Cantor set; at r = 2/3 the three pieces meet in a
    Cantor set on a vertical segment through 0, which is the figure in
    the paper.  That 2/3 comes from altitudes of an isosceles triangle
    meeting in ratio 1:2, so it is derived for the TRIANGLE only -- the
    paper notes the construction applies to every n-gon with n >= 3 but
    gives no ratio for them, which is why `sides` and `ratio` are both
    exposed rather than tied together.

    Dimension (triangle, r = 2/3): log 3 / log(3/2) = 2.7095..."""
    n = max(3, int(sides))
    r = float(ratio)
    A1 = r * np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0]])
    b1 = np.array([1.0 - r, 0.0, 0.0])
    out = []
    for k in range(n):
        t = _rot((0.0, 0.0, 1.0), 2.0 * math.pi * k / n)
        out.append((t @ A1 @ t.T, t @ b1, 1.0))
    return out


# the regular tetrahedron on alternate vertices of the unit cube, in
# the paper's own labelling: c3, c4, c1, c2
_BMM_TETRA_V = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])


def _bmm_tetrahedron(ratio=0.6):
    """The same paper's modified fractal tetrahedron (their Figure 8).

    Each f_i is the homothety toward vertex c_i composed with a
    rotation about the altitude from c_i onto the opposite face.  A
    120 degree turn there would be a symmetry of the ordinary fractal
    tetrahedron and give nothing new, so they take 60 degrees --
    "or, equivalently, around 180 degrees", the two differing by that
    symmetry.  The ratio r = 3/5 is the value at which the images of
    the basic tetrahedron meet along an edge; they derive it from
    |c2' - c3| = |c2' - c1|, giving t = 4/5 and |c2' - c1| =
    (3/5) sqrt 2 against |c2 - c1| = sqrt 2.

    The pieces meet the faces of the convex hull in Koch curves.
    Dimension: log 4 / log(5/3) = 2.7138..."""
    r = float(ratio)
    centre = _BMM_TETRA_V.mean(axis=0)
    out = []
    for c in _BMM_TETRA_V:
        R = _rot(centre - c, math.pi)
        out.append((r * R, c - r * (R @ c), 1.0))
    return out


def _bmm_cube(ratio=0.625):
    """The same paper's modified cube (their Figure 9).

    The cube is the self-similar set of eight homotheties of factor 1/2
    centred at its vertices; combining those with a 180 degree rotation
    about the corresponding space diagonal and raising the factor to
    5/8 gives this fractal, whose pieces touch in single points.

    No dimension is quoted, and none should be: 8 (5/8)^3 = 1.953 > 1,
    so the pieces overlap in measure and Moran's formula does not
    apply.  Being centrally symmetric, this one coincides with its own
    reverse -- the paper says so, and the self-test checks it."""
    r = float(ratio)
    out = []
    for v in _CUBE_V:
        R = _rot(v, math.pi)
        out.append((r * R, v - r * (R @ v), 1.0))
    return out


def _menger_maps():
    cells = [c for c in
             ((i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
              for k in (-1, 0, 1))
             if sum(1 for t in c if t == 0) <= 1]
    return [(np.eye(3) / 3.0, (2.0 / 3.0) * np.asarray(c, dtype=float),
             1.0) for c in cells]


def _fern_maps():
    """Barnsley's fern -- a TWO-dimensional system, embedded in the
    z = 0 plane.  Its four maps are published (Barnsley, "Fractals
    Everywhere"); there is no authoritative three-dimensional fern, so
    none is invented here."""
    raw = [((0.00, 0.00, 0.00, 0.16), (0.0, 0.00), 0.01),
           ((0.85, 0.04, -0.04, 0.85), (0.0, 1.60), 0.85),
           ((0.20, -0.26, 0.23, 0.22), (0.0, 1.60), 0.07),
           ((-0.15, 0.28, 0.26, 0.24), (0.0, 0.44), 0.07)]
    # embedded in the xz-plane, not xy, so the fern stands upright in
    # Blender's z-up world instead of lying flat on the ground
    out = []
    for (a, b, c, d), (e, f), p in raw:
        A = np.array([[a, 0.0, b], [0.0, 0.0, 0.0], [c, 0.0, d]])
        out.append((A, np.array([e, 0.0, f]), p))
    return out


def _planar(maps2d):
    """Lift a list of 2-D maps ((a,b,c,d), (e,f), p) into the xz-plane,
    so a planar system stands upright in Blender's z-up world."""
    out = []
    for (a, b, c, d), (e, f), pr in maps2d:
        A = np.array([[a, 0.0, b], [0.0, 0.0, 0.0], [c, 0.0, d]])
        out.append((A, np.array([e, 0.0, f]), pr))
    return out


def _sierpinski2d():
    """Three half-scale maps to the corners of a triangle."""
    v = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3.0) / 2.0)]
    return _planar([((0.5, 0.0, 0.0, 0.5), (0.5 * x, 0.5 * y), 1.0)
                    for (x, y) in v])


def _dragon2d():
    """Heighway dragon: two similarities of ratio 1/sqrt2 at +-45
    degrees.  Its attractor is the boundary curve's filled region."""
    return _planar([((0.5, -0.5, 0.5, 0.5), (0.0, 0.0), 1.0),
                    ((-0.5, -0.5, 0.5, -0.5), (1.0, 0.0), 1.0)])


def _levy2d():
    """Levy C curve: two similarities of ratio 1/sqrt2."""
    return _planar([((0.5, -0.5, 0.5, 0.5), (0.0, 0.0), 1.0),
                    ((0.5, 0.5, -0.5, 0.5), (0.5, 0.5), 1.0)])


def _koch2d():
    """Koch curve as four third-scale maps, two of them turned by
    +-60 degrees."""
    t = math.radians(60.0)
    ct, st = math.cos(t) / 3.0, math.sin(t) / 3.0
    return _planar([
        ((1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0), (0.0, 0.0), 1.0),
        ((ct, -st, st, ct), (1.0 / 3.0, 0.0), 1.0),
        ((ct, st, -st, ct), (0.5, math.sqrt(3.0) / 6.0), 1.0),
        ((1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0), (2.0 / 3.0, 0.0), 1.0)])


# (label, map factory, dimension).  The dimension is declared rather
# than guessed because it decides which renderings even make sense: a
# planar system has no solid image, and a volume grid over it wastes
# almost all of its cells on empty space.
IFS_PRESETS = {
    'SIERP_TETRA': ("Sierpinski Tetrahedron",
                    lambda: _uniform(_TETRA, 0.5), 3),
    'SIERP_OCTA': ("Sierpinski Octahedron",
                   lambda: _uniform(_OCTA, 0.5), 3),
    'SIERP_CUBE': ("Cantor Dust (cube corners)",
                   lambda: _uniform(_CUBE_V, 1.0 / 3.0), 3),
    'MENGER': ("Menger Sponge", _menger_maps, 3),
    'BMM_SIERP': ("Sierpinski Triangle in 3D (Bandt et al.)",
                  _bmm_sierpinski, 3),
    'BMM_TETRA': ("Modified Fractal Tetrahedron (Bandt et al.)",
                  _bmm_tetrahedron, 3),
    'BMM_CUBE': ("Modified Cube (Bandt et al.)", _bmm_cube, 3),
    'FERN2D': ("Barnsley Fern (2-D)", _fern_maps, 2),
    'SIERP_TRI': ("Sierpinski Triangle (2-D)", _sierpinski2d, 2),
    'DRAGON': ("Heighway Dragon (2-D)", _dragon2d, 2),
    'LEVY': ("Levy C Curve (2-D)", _levy2d, 2),
    'KOCH': ("Koch Curve (2-D)", _koch2d, 2),
}


def plane_frame(points, tol=1e-6):
    """Detect a planar point set and return (centre, axes, flat).

    `axes` are the two in-plane principal directions followed by the
    normal.  Whether a system is planar is measured, not assumed, so a
    custom map set that happens to be flat gets the same treatment as
    the shipped 2-D families."""
    P = np.asarray(points, dtype=float)
    c = P.mean(axis=0)
    X = P - c
    # the 3x3 covariance, not an SVD of the whole cloud: for a few
    # hundred thousand points the full SVD would try to allocate a
    # square matrix of that side
    w, V = np.linalg.eigh(X.T @ X)
    order = np.argsort(w)[::-1]              # widest spread first
    axes = V[:, order].T
    sv = np.sqrt(np.maximum(w[order], 0.0))
    flat = bool(sv[2] <= tol * max(sv[0], 1e-300))
    return c, axes, flat


def plane_relief(points, resolution=512, thickness_cells=1):
    """Mesh a planar point set as a watertight slab one cell thick.

    A planar attractor deserves a plane's worth of resolution: a
    512 x 512 grid is a quarter of a million cells, where a 512^3
    volume grid is out of reach and would leave all but a sliver of it
    empty.  The slab is closed, so it prints, and Solidify will give it
    real depth."""
    P = np.asarray(points, dtype=float)
    c, axes, _flat = plane_frame(P)
    u = (P - c) @ axes[0]
    v = (P - c) @ axes[1]
    res = max(8, int(resolution))
    lo_u, lo_v = float(u.min()), float(v.min())
    s = max(float(u.max()) - lo_u, float(v.max()) - lo_v) / res
    if s <= 0.0:
        raise ValueError("the attractor collapsed to a point")
    iu = np.clip(np.floor((u - lo_u) / s).astype(np.int64), 0, res)
    iv = np.clip(np.floor((v - lo_v) / s).astype(np.int64), 0, res)
    nk = max(1, int(thickness_cells))
    cells = np.unique(np.stack([iu, iv], axis=1), axis=0)
    cells = np.concatenate(
        [np.column_stack([cells, np.full(len(cells), k)])
         for k in range(nk)], axis=0)
    verts_i, faces = voxel_surface(cells)
    # back into three dimensions, centred on the plane
    xyz = (c
           + np.outer(lo_u + verts_i[:, 0] * s, axes[0])
           + np.outer(lo_v + verts_i[:, 1] * s, axes[1])
           + np.outer((verts_i[:, 2] - 0.5 * nk) * s, axes[2]))
    return xyz, faces, len(cells)


def format_maps(maps, prec=8):
    """Render a map list back into the text the Maps field shows --
    the inverse of parse_maps, so a preset can be loaded into the field
    and edited from there."""
    out = []
    for A, b, p in maps:
        lin = " ".join(f"{v:.{prec}g}" for v in np.asarray(A).ravel())
        tr = " ".join(f"{v:.{prec}g}" for v in np.asarray(b).ravel())
        out.append(f"{lin} | {tr} | {p:.{prec}g}")
    return "; ".join(out)


# Facts quoted from Bandt, Mai The Duy and Mesing, "Three-Dimensional
# Fractals", Math. Intelligencer 32(3), 2010, and shown in the
# operator's status line.
IFS_FACTS = {
    'SIERP_TETRA': "the fractal tetrahedron; A. G. Bell built it from "
                   "kites in 1903, some years before Sierpinski. "
                   "Dimension 2",
    'SIERP_OCTA': "the fractal octahedron: 8 faces and NO interior -- "
                  "a deflated balloon. Neighbouring faces meet in an "
                  "ordinary Euclidean triangle; face dimension "
                  "log 6 / log 2 = 2.585",
    'MENGER': "dimension log 20 / log 3 = 2.727, with three neighbour "
              "types (face, edge and point)",
    'BMM_SIERP': "3 pieces at ratio 2/3, each turned 90 degrees about "
                 "its own axis; the pieces meet in a Cantor set on a "
                 "vertical segment. Dimension log 3 / log(3/2) = "
                 "2.710 [Bandt et al. 2010, Fig. 7]",
    'BMM_TETRA': "4 pieces at ratio 3/5, each turned 180 degrees about "
                 "its altitude; the pieces meet the faces of the hull "
                 "in Koch curves. Dimension log 4 / log(5/3) = 2.714 "
                 "[Fig. 8]",
    'BMM_CUBE': "8 pieces at ratio 5/8, each turned 180 degrees about "
                "a space diagonal; they touch in single points. "
                "Centrally symmetric, so it is its own reverse. The "
                "pieces overlap in measure, so no dimension formula "
                "applies [Fig. 9]",
}


def parse_maps(spec):
    """Parse custom affine maps, one per line or semicolon:

        a b c d e f g h i | tx ty tz | p

    with the nine numbers the row-major 3x3 linear part, the three the
    translation and the optional p the chaos-game probability."""
    out = []
    for chunk in str(spec).replace('\n', ';').split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split('|')]
        if len(parts) < 2:
            raise ValueError(f"map {chunk!r} needs a '|' between the "
                             f"matrix and the translation")
        try:
            lin = [float(v) for v in parts[0].split()]
            tr = [float(v) for v in parts[1].split()]
            p = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
        except ValueError:
            raise ValueError(f"cannot read the numbers in map {chunk!r}")
        if len(lin) != 9:
            raise ValueError(f"map {chunk!r} needs 9 matrix entries, "
                             f"got {len(lin)}")
        if len(tr) != 3:
            raise ValueError(f"map {chunk!r} needs 3 translation "
                             f"entries, got {len(tr)}")
        out.append((np.asarray(lin).reshape(3, 3), np.asarray(tr), p))
    if not out:
        raise ValueError("the map specification is empty")
    return out


def contractive(maps, tol=1e-9):
    """Every map a contraction in the Euclidean metric, by largest
    singular value.  This is the strict condition, and it is the right
    one for the deterministic solid-copies path, where a map that
    stretches in some direction would make the copies grow."""
    return all(float(np.linalg.svd(A, compute_uv=False)[0]) < 1.0 - tol
               for A, _, _ in maps)


def spectrally_contractive(maps, tol=1e-9):
    """Every map eventually contracting, by spectral radius.

    A radix tile's own maps M^-1 (x + d) are contractions in SOME
    metric but usually not in the Euclidean one -- sigma_max(M^-1) runs
    to 1.46 for these presets while the spectral radius stays near 0.7.
    Testing singular values there would reject the very systems this
    module is built on."""
    return all(float(np.max(np.abs(np.linalg.eigvals(A)))) < 1.0 - tol
               for A, _, _ in maps)


def chaos_game(maps, points=400000, seed=0, walkers=4096,
               transient=20):
    """Sample the attractor.  Rather than one long sequential orbit,
    many walkers advance together and each step is a handful of
    vectorised affine maps -- the same measure, a hundred times faster
    in Python."""
    m = len(maps)
    probs = np.array([max(p, 0.0) for _, _, p in maps], dtype=float)
    probs = (probs / probs.sum() if probs.sum() > 0
             else np.full(m, 1.0 / m))
    rng = np.random.default_rng(int(seed))
    n = max(64, int(walkers))
    steps = max(1, int(math.ceil(points / n)) + int(transient))
    X = rng.normal(size=(n, 3)) * 0.1
    keep = []
    for t in range(steps):
        idx = rng.choice(m, size=n, p=probs)
        Y = np.empty_like(X)
        for i, (A, b, _) in enumerate(maps):
            sel = idx == i
            if np.any(sel):
                Y[sel] = X[sel] @ A.T + b
        X = Y
        if t >= transient:
            keep.append(X.copy())
    P = np.vstack(keep)
    return P[:int(points)] if len(P) > points else P


def _occupied_cells(P, res, cover=0.98):
    """Bin points into a res^3 grid over their own bounding box and
    return (cells, counts, lo, cell_size).  A robust box is used: the
    outermost `1 - cover` of the points, which for a chaos game are
    stragglers still converging, would otherwise stretch the grid."""
    lo = np.quantile(P, (1.0 - cover) / 2.0, axis=0)
    hi = np.quantile(P, 1.0 - (1.0 - cover) / 2.0, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    s = float(np.max(span)) / int(res)
    lo = 0.5 * (lo + hi) - 0.5 * s * int(res)
    idx = np.floor((P - lo) / s).astype(np.int64)
    idx = idx[np.all((idx >= 0) & (idx < int(res)), axis=1)]
    cells, counts = np.unique(idx, axis=0, return_counts=True)
    return cells, counts, lo, s


def build_ifs(preset='SIERP_TETRA', output='SOLIDS', maps=None,
              depth=5, seed_solid='TETRA', points=400000,
              resolution=128, plane_resolution=512, cover=0.90, seed=0,
              min_count=1, largest_only=False, reverse=False,
              scale=1.0):
    """Mesh a general affine IFS attractor, in two or three dimensions.
    Returns (verts, faces, info).

    A planar system is detected from the attractor itself and meshed as
    a slab one cell thick at plane resolution, which is far finer than
    any volume grid could be."""
    if maps is None:
        if preset == 'CUSTOM':
            raise ValueError("custom mode needs a map specification")
        maps = IFS_PRESETS[preset][1]()
    if not maps:
        raise ValueError("no maps given")
    if reverse:
        # Bandt, Mai The Duy and Mesing call these the "reverse
        # fractals": replacing every f_i by -f_i leaves the neighbour
        # maps f_i^-1 f_j unchanged, so the dimension and the number of
        # boundary types are the same while the shape is quite
        # different.
        maps = [(-np.asarray(A), -np.asarray(b), p) for A, b, p in maps]
    if not contractive(maps):
        raise ValueError("every map must be a contraction (largest "
                         "singular value < 1)")

    m = len(maps)

    # Is this system planar?  Measure it rather than trusting the
    # preset label, so a custom flat map set is handled too.
    probe = chaos_game(maps, points=4000, seed=1, transient=200)
    _c, _axes, planar = plane_frame(probe)
    if planar:
        if output == 'SOLIDS':
            raise ValueError(
                "this system is planar, so solid copies would flatten "
                "the seed solid to a plate. It is drawn as a relief "
                "instead -- choose the Relief or Smooth Contour output")
        P = chaos_game(maps, points=points, seed=seed, transient=300)
        verts, faces, ncell = plane_relief(P, plane_resolution)
        info = {'points': len(P), 'cells': ncell, 'maps': m,
                'planar': True, 'resolution': int(plane_resolution)}
        return (center_fit(verts, scale),
                orient_outward(verts, _as_quads(faces)), info)

    if output == 'RELIEF':
        raise ValueError("the Relief output is for planar systems; "
                         "this one is three-dimensional, so use "
                         "Voxels, Smooth Contour or Solid Copies")

    if output == 'SOLIDS':
        # A singular map has no solid image: it squashes the seed flat.
        # The Barnsley fern is the case that matters here -- it is a
        # TWO-dimensional system embedded in z = 0, so all four of its
        # maps have a zero third row and column, and solid copies of it
        # come out as a scatter of loose plates rather than a fern.
        flat = [i for i, (A, _, _) in enumerate(maps)
                if abs(float(np.linalg.det(A))) < 1e-12]
        if flat:
            raise ValueError(
                f"solid copies need invertible maps, but "
                f"{len(flat)} of {m} are singular (they flatten the "
                f"seed solid to a plate). Use the Voxels or Smooth "
                f"Contour output for this system")
        d = max(1, int(depth))
        while m ** d > MAX_CELLS and d > 1:
            d -= 1
        sv, sf = SEEDS[seed_solid]
        # compose all m^d words, then place one seed copy per word
        A = [np.eye(3)]
        b = [np.zeros(3)]
        for _ in range(d):
            A2, b2 = [], []
            for Ai, bi in zip(A, b):
                for (Am, bm, _p) in maps:
                    A2.append(Am @ Ai)
                    b2.append(Am @ bi + bm)
            A, b = A2, b2
        verts, faces = [], []
        for Ai, bi in zip(A, b):
            base = len(verts)
            for v in sv:
                verts.append(Ai @ v + bi)
            # a map with negative determinant reflects, which reverses
            # the seed's winding along with it
            wind = (list(sf) if np.linalg.det(Ai) >= 0.0
                    else [tuple(reversed(tuple(f))) for f in sf])
            for f in wind:
                faces.append([base + i for i in f])
        verts = np.asarray(verts, dtype=float)
        info = {'copies': len(A), 'depth': d, 'maps': m}
        return center_fit(verts, scale), faces, info

    P = chaos_game(maps, points=points, seed=seed)
    if output == 'VOXEL':
        cells, counts, lo, s = _occupied_cells(P, resolution)
        cells = cells[counts >= max(1, int(min_count))]
        if not len(cells):
            raise ValueError("no cell met the minimum point count")
        verts_i, faces = voxel_surface(cells)
        verts = lo + verts_i.astype(float) * s
        info = {'points': len(P), 'cells': len(cells), 'maps': m}
        return center_fit(verts, scale), _as_quads(faces), info

    # ISO: density grid, blurred, contoured
    res = int(resolution)
    cells, counts, lo, s = _occupied_cells(P, res)
    dens = np.zeros((res, res, res), dtype=float)
    dens[cells[:, 0], cells[:, 1], cells[:, 2]] = counts
    dens = blur_density(dens)
    flat = dens.ravel()
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order])
    if cum[-1] <= 0.0:
        raise ValueError("the density grid came out empty")
    # contour enclosing `cover` of the sampled mass
    t = float(flat[order[min(int(np.searchsorted(cum,
                                                 cover * cum[-1])),
                             len(order) - 1)]])
    hi = lo + s * res

    def field(X, Y, Z):
        ix = np.clip(((X - lo[0]) / s).astype(np.int64), 0, res - 1)
        iy = np.clip(((Y - lo[1]) / s).astype(np.int64), 0, res - 1)
        iz = np.clip(((Z - lo[2]) / s).astype(np.int64), 0, res - 1)
        return t - dens[ix, iy, iz]

    mst = _toolkit()
    verts, tris = mst.marching_tets(field, lo, hi, (res, res, res))
    if not len(tris):
        raise ValueError("the contour came out empty -- try a larger "
                         "cover or more points")
    if largest_only:
        verts, tris = keep_largest(verts, tris)
    info = {'points': len(P), 'level': t, 'maps': m,
            'tris': len(tris)}
    return (center_fit(verts, scale),
            orient_outward(verts, [tuple(int(i) for i in f)
                                   for f in tris]), info)


def keep_largest(verts, tris):
    """Biggest connected piece only -- a chaos game on a coarse grid
    leaves speckle."""
    parent = list(range(len(verts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for t in tris:
        a = find(int(t[0]))
        for i in (1, 2):
            b = find(int(t[i]))
            if a != b:
                parent[b] = a
    lab = np.array([find(int(t[0])) for t in tris])
    vals, counts = np.unique(lab, return_counts=True)
    keep = tris[lab == vals[int(np.argmax(counts))]]
    used = np.unique(keep)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[keep]


def center_fit(verts, scale=1.0):
    """Centre on the bounding box and fit the largest extent to a 2 m
    cube (the project-wide convention), then apply `scale`."""
    verts = np.asarray(verts, dtype=float)
    if not len(verts):
        return verts
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ext = float((hi - lo).max())
    return (verts - 0.5 * (lo + hi)) * (2.0 / ext
                                        if ext > 1e-9 else 1.0) * scale


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty, StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _new_object(context, name, verts, faces, smooth=False):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        me.polygons.foreach_set('use_smooth',
                                [smooth] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    # Blender hands dynamic enum items back to C, which does not keep
    # the Python strings alive; anything returned from an items
    # callback has to be held in a module-level cache or the labels
    # turn to garbage.
    _ENUM_CACHE = {}

    # Selecting a system loads its maps into the field; editing the
    # field flips the system to Custom so the edit is what gets built.
    # The flag stops the two callbacks from calling each other.
    _SYNC = {'busy': False}

    def _on_system(self, context):
        if _SYNC['busy'] or self.ifs_preset == 'CUSTOM':
            return
        entry = IFS_PRESETS.get(self.ifs_preset)
        if entry is None:
            return
        _SYNC['busy'] = True
        try:
            self.maps = format_maps(entry[1]())
        finally:
            _SYNC['busy'] = False

    def _on_maps(self, context):
        if _SYNC['busy'] or self.ifs_preset == 'CUSTOM':
            return
        _SYNC['busy'] = True
        try:
            self.ifs_preset = 'CUSTOM'
        finally:
            _SYNC['busy'] = False

    def _system_items(self, context):
        dim = int(getattr(self, 'dimension', '3'))
        items = [(k, v[0], v[0]) for k, v in IFS_PRESETS.items()
                 if v[2] == dim]
        items.append(('CUSTOM', "Custom", "Use the maps field below"))
        _ENUM_CACHE[f'sys{dim}'] = items
        return items

    def _output_items(self, context):
        dim = int(getattr(self, 'dimension', '3'))
        if dim == 2:
            # a planar attractor has no solid image, and a volume grid
            # over it would be almost entirely empty
            items = [('RELIEF', "Relief",
                      "A watertight slab one cell thick, meshed at "
                      "plane resolution")]
        else:
            items = [('SOLIDS', "Solid Copies",
                      "Deterministic: one seed solid per word"),
                     ('VOXEL', "Voxels",
                      "Chaos game binned into a watertight voxel grid"),
                     ('ISO', "Smooth Contour",
                      "Chaos game contoured by marching tetrahedra")]
        _ENUM_CACHE[f'out{dim}'] = items
        return items

    class MESH_OT_ifs_add(bpy.types.Operator):
        """Add a three-dimensional self-affine tile or the attractor of
        an affine iterated function system"""
        bl_idname = "mesh.ifs_add"
        bl_label = "Iterated Function System"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            items=[('IFS', "IFS Attractor", "The attractor of a set of "
                                            "contractive affine maps"),
                   ('RADIX', "Self-Affine Tile", "A lattice tile from "
                                                 "an expanding integer "
                                                 "matrix and a residue "
                                                 "digit set")],
            default='IFS')
        preset: EnumProperty(
            name="Tile",
            items=[(k, v[0], v[0]) for k, v in RADIX_PRESETS.items()],
            default='ABC_124')
        tile_output: EnumProperty(
            name="Tile Output",
            items=[('VOXEL', "Voxels", "Sample the attractor and mesh "
                                       "it as a watertight voxel "
                                       "solid"),
                   ('SMOOTH', "Smooth Contour", "Sample the attractor "
                                                "and contour it with "
                                                "marching tetrahedra"),
                   ('EXACT', "Exact Level-k Cubes", "The exact union "
                                                    "of C^k cells, "
                                                    "volume exactly 1 "
                                                    "-- but the cells "
                                                    "flatten into "
                                                    "plates as the "
                                                    "level rises")],
            default='VOXEL')
        level: IntProperty(
            name="Level", default=0, min=0, max=24,
            description="Exact mode only: radix depth; 0 picks a level "
                        "landing in the 30k-300k cell band")
        holes: IntProperty(
            name="Holes", default=0, min=0, max=6,
            description="Drop this many digits at every level, turning "
                        "the tile into a gasket")
        dimension: EnumProperty(
            name="Dimension",
            items=[('3', "3D", "Systems whose attractor fills three "
                               "dimensions"),
                   ('2', "2D", "Planar systems, meshed as a relief")],
            default='3')
        # both of these are filtered by the dimension above, so they
        # take an items callback rather than a fixed list; a dynamic
        # enum cannot carry a default, so the first entry is it
        ifs_preset: EnumProperty(name="System", items=_system_items,
                                 update=_on_system)
        output: EnumProperty(name="Output", items=_output_items)
        seed_solid: EnumProperty(
            name="Seed Solid",
            items=[('TETRA', "Tetrahedron", ""), ('CUBE', "Cube", ""),
                   ('OCTA', "Octahedron", "")],
            default='TETRA')
        depth: IntProperty(
            name="Depth", default=5, min=1, max=12,
            description="Solid-copies depth; the count is maps^depth "
                        "and is capped automatically")
        points: IntProperty(
            name="Points", default=400000, min=10000, max=5000000,
            description="Chaos-game sample count")
        resolution: IntProperty(
            name="Resolution", default=128, min=16, max=256,
            description="Voxel / density grid resolution per axis")
        plane_resolution: IntProperty(
            name="Plane Resolution", default=512, min=32, max=2048,
            description="In-plane grid resolution for a planar "
                        "system; a plane affords far more of it than a "
                        "volume can")
        cover: FloatProperty(
            name="Cover", default=0.90, min=0.1, max=0.999,
            description="Smooth contour: the fraction of the sampled "
                        "mass the surface encloses")
        min_count: IntProperty(
            name="Min Points per Cell", default=1, min=1, max=200,
            description="Voxel mode: cells with fewer points than this "
                        "are left empty")
        maps: StringProperty(
            name="Maps", update=_on_maps,
            default="0.5 0 0 0 0.5 0 0 0 0.5 | 0.5 0.5 0.5 | 1; "
                    "0.5 0 0 0 0.5 0 0 0 0.5 | -0.5 -0.5 0.5 | 1; "
                    "0.5 0 0 0 0.5 0 0 0 0.5 | 0.5 -0.5 -0.5 | 1; "
                    "0.5 0 0 0 0.5 0 0 0 0.5 | -0.5 0.5 -0.5 | 1",
            description="Custom affine maps: nine matrix entries | "
                        "three translations | probability, one map per "
                        "semicolon")
        seed: IntProperty(
            name="Seed", default=0, min=0, max=99999,
            description="Chaos-game random seed; the same seed always "
                        "gives the same mesh")
        poly_sides: IntProperty(
            name="Polygon Sides", default=3, min=3, max=10,
            description="The n-gon the Sierpinski-in-3D construction "
                        "is built on; the paper derives its ratio for "
                        "the triangle and notes the construction "
                        "applies to every n >= 3")
        poly_ratio: FloatProperty(
            name="Polygon Ratio", default=2.0 / 3.0, min=0.35,
            max=0.95,
            description="Contraction ratio toward each vertex; 2/3 is "
                        "the triangle value at which the pieces meet, "
                        "and 1/2 would give a Cantor set")
        reverse: BoolProperty(
            name="Reverse", default=False,
            description="Replace every map f by -f: the neighbour maps "
                        "are unchanged, so the dimension and the "
                        "boundary structure survive, but the shape "
                        "does not")
        largest_only: BoolProperty(
            name="Largest Piece Only", default=False,
            description="Smooth contour: discard all but the biggest "
                        "connected piece")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0)
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this "
                        "thickness")
        smooth: BoolProperty(name="Smooth Shading", default=False)

        def execute(self, context):
            try:
                if self.mode == 'RADIX':
                    verts, faces, info = build_radix(
                        preset=self.preset, level=self.level,
                        holes=self.holes, output=self.tile_output,
                        resolution=self.resolution, points=self.points,
                        seed=self.seed,
                        largest_only=self.largest_only,
                        scale=self.scale)
                    label = RADIX_PRESETS[self.preset][0]
                    if info['holes']:
                        label += f" gasket -{info['holes']}"
                else:
                    if (self.ifs_preset != 'CUSTOM'
                            and self.ifs_preset in IFS_PRESETS
                            and IFS_PRESETS[self.ifs_preset][2]
                            != int(self.dimension)):
                        raise ValueError(
                            f"{IFS_PRESETS[self.ifs_preset][0]} is a "
                            f"{IFS_PRESETS[self.ifs_preset][2]}D "
                            f"system; switch the Dimension to match")
                    if self.ifs_preset == 'CUSTOM':
                        mp = parse_maps(self.maps)
                    elif self.ifs_preset == 'BMM_SIERP':
                        mp = _bmm_sierpinski(self.poly_sides,
                                             self.poly_ratio)
                    else:
                        mp = None
                    verts, faces, info = build_ifs(
                        preset=self.ifs_preset, output=self.output,
                        maps=mp, depth=self.depth,
                        seed_solid=self.seed_solid, points=self.points,
                        resolution=self.resolution,
                        plane_resolution=self.plane_resolution,
                        cover=self.cover,
                        seed=self.seed, min_count=self.min_count,
                        largest_only=self.largest_only,
                        reverse=self.reverse,
                        scale=self.scale)
                    label = (IFS_PRESETS[self.ifs_preset][0]
                             if self.ifs_preset != 'CUSTOM'
                             else "Custom IFS")
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

            # edge-to-edge contact is intrinsic to these families, but
            # a slicer will choke on it, so say so -- only when the
            # count is small enough that the check is cheap
            if len(faces) <= 200000 and self.output != 'ISO':
                nb, nm = edge_stats(faces)
                if nb:
                    self.report({'WARNING'},
                                f"{label}: {nb} boundary edges -- the "
                                f"surface is not closed")
                elif nm:
                    self.report({'WARNING'},
                                f"{label}: closed, but {nm} edges have "
                                f"cells meeting only edge to edge; "
                                f"non-manifold, so thicken it before "
                                f"printing")

            obj = _new_object(context, label, verts, faces,
                              smooth=self.smooth)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            me = obj.data
            if self.mode == 'RADIX':
                if info.get('topology'):
                    self.report({'INFO'},
                                f"{label}: {info['topology']}")
                if info.get('holes_clamped'):
                    self.report({'WARNING'},
                                f"{label}: dropping more than "
                                f"{info['max_holes']} digits leaves "
                                f"them coplanar, which would collapse "
                                f"the tile to a sheet -- clamped")
                fid = info.get('fidelity', 1.0)
                if self.tile_output == 'EXACT':
                    asp = info.get('cell_aspect', 1.0)
                    if asp > 20.0:
                        self.report(
                            {'WARNING'},
                            f"{label}: at level {info['level']} the "
                            f"cells are {asp:.0f}:1 slivers, so this "
                            f"reads as a laminate rather than a solid "
                            f"-- use the Voxels or Smooth output")
                    if fid < 0.95:
                        self.report(
                            {'WARNING'},
                            f"{label}: the level-{info['level']} body "
                            f"reaches only {100 * fid:.0f}% of the "
                            f"true tile's extent on its thinnest axis")
                    self.report(
                        {'INFO'},
                        f"{label}: level {info['level']}, "
                        f"{info['cells']} cells, volume "
                        f"{info['volume']:.4f}, cell aspect "
                        f"{info['cell_aspect']:.0f}:1, "
                        f"{len(me.vertices)} verts")
                else:
                    self.report(
                        {'INFO'},
                        f"{label}: {info.get('points', 0)} attractor "
                        f"samples at resolution "
                        f"{info.get('resolution', 0)}, "
                        f"{100 * fid:.0f}% of the true extent, "
                        f"{len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces")
            else:
                note = IFS_FACTS.get(self.ifs_preset)
                if note:
                    self.report({'INFO'}, f"{label}: {note}")
                if self.output == 'SOLIDS':
                    extra = f"{info.get('copies', 0)} copies"
                elif info.get('planar'):
                    extra = (f"planar relief, {info.get('cells', 0)} "
                             f"cells at {info.get('resolution', 0)}^2")
                else:
                    extra = f"{info.get('points', 0)} points"
                self.report({'INFO'},
                            f"{label}: {extra}, {len(me.vertices)} "
                            f"verts, {len(me.polygons)} faces")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'RADIX':
                lay.prop(self, 'preset')
                lay.prop(self, 'tile_output')
                lay.prop(self, 'holes')
                if self.tile_output == 'EXACT':
                    lay.prop(self, 'level')
                else:
                    lay.prop(self, 'resolution')
                    lay.prop(self, 'points')
                    lay.prop(self, 'seed')
                    if self.tile_output == 'SMOOTH':
                        lay.prop(self, 'largest_only')
            else:
                lay.prop(self, 'dimension', expand=True)
                lay.prop(self, 'ifs_preset')
                if self.ifs_preset == 'BMM_SIERP':
                    lay.prop(self, 'poly_sides')
                    lay.prop(self, 'poly_ratio')
                lay.prop(self, 'reverse')
                lay.prop(self, 'maps')
                lay.prop(self, 'output')
                if self.output == 'SOLIDS':
                    lay.prop(self, 'seed_solid')
                    lay.prop(self, 'depth')
                else:
                    lay.prop(self, 'points')
                    lay.prop(self, 'seed')
                    if self.output == 'RELIEF':
                        lay.prop(self, 'plane_resolution')
                    else:
                        lay.prop(self, 'resolution')
                        if self.output == 'VOXEL':
                            lay.prop(self, 'min_count')
                        else:
                            lay.prop(self, 'cover')
                            lay.prop(self, 'largest_only')
            for k in ('scale', 'thickness', 'smooth'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.ifs_add", icon='MOD_REMESH')

    _classes = (MESH_OT_ifs_add,)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


# ==========================================================================
# Standalone numeric self-test
# ==========================================================================

def _signed_volume(verts, faces):
    """Enclosed volume by the divergence theorem; positive when the
    normals point outward."""
    V = np.asarray(verts, dtype=float)
    tot = 0.0
    for f in faces:
        f = list(f)
        for i in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[i]], V[f[i + 1]]
            tot += float(np.dot(a, np.cross(b - a, c - a))) / 6.0
    return tot


def _selftest():
    # ---- 0. normals point outward, everywhere ------------------------
    # Three separate ways to get this wrong, all of them silent: a seed
    # solid wound the wrong way; an orientation-REVERSING transform
    # (det M is negative for every ABC tile, so M^-k flips at odd
    # levels); and an isosurface left open, whose normals then mean
    # nothing at all.  The divergence theorem catches all three.
    for name, (sv, sf) in SEEDS.items():
        vol = _signed_volume(sv, sf)
        if vol <= 0.0:
            raise AssertionError(
                f"the {name} seed solid is wound inside out "
                f"(signed volume {vol:+.4f})")
    print(f"seed solids: {', '.join(sorted(SEEDS))} all wound outward")

    checks = [
        ("radix exact, odd level",
         lambda: build_radix(preset='ABC_124', level=3, output='EXACT')),
        ("radix exact, even level",
         lambda: build_radix(preset='ABC_124', level=4, output='EXACT')),
        ("radix voxels",
         lambda: build_radix(preset='ABC_124', output='VOXEL',
                             resolution=48, points=120000)),
        ("radix smooth",
         lambda: build_radix(preset='ABC_223', output='SMOOTH',
                             resolution=48, points=150000)),
        ("ifs solid tetrahedra",
         lambda: build_ifs(preset='SIERP_TETRA', output='SOLIDS',
                           depth=3)),
        ("ifs solid cubes",
         lambda: build_ifs(preset='MENGER', output='SOLIDS',
                           seed_solid='CUBE', depth=1)),
        ("ifs solid octahedra",
         lambda: build_ifs(preset='SIERP_OCTA', output='SOLIDS',
                           seed_solid='OCTA', depth=2)),
        ("ifs voxels",
         lambda: build_ifs(preset='SIERP_TETRA', output='VOXEL',
                           points=80000, resolution=40)),
        ("ifs smooth",
         lambda: build_ifs(preset='SIERP_TETRA', output='ISO',
                           points=150000, resolution=48)),
        ("planar relief",
         lambda: build_ifs(preset='SIERP_TRI', output='RELIEF',
                           points=150000, plane_resolution=192)),
    ]
    for name, fn in checks:
        V, F, info = fn()
        nb, nm = edge_stats(F)
        if nb:
            raise AssertionError(
                f"{name}: {nb} boundary edges -- the surface is open, "
                f"so its normals are meaningless")
        vol = _signed_volume(V, F)
        if vol <= 0.0:
            raise AssertionError(
                f"{name}: signed volume {vol:+.4f} -- the mesh is "
                f"inside out")
        print(f"{name:24s}: closed, signed volume {vol:+.4f}")

    # ---- 1. every preset really is a radix system -------------------
    for key, (label, (M, D), _meta) in RADIX_PRESETS.items():
        if not is_expanding(M):
            raise AssertionError(f"{label}: matrix is not expanding, "
                                 f"eigenvalues "
                                 f"{np.linalg.eigvals(M.astype(float))}")
        if not is_residue_system(M, D):
            raise AssertionError(f"{label}: digits are not a complete "
                                 f"residue system")
        det = abs(int(round(np.linalg.det(M.astype(float)))))
        if det != len(D):
            raise AssertionError(f"{label}: |det M| = {det} but there "
                                 f"are {len(D)} digits")
    print(f"{len(RADIX_PRESETS)} radix presets: expanding, "
          f"|det M| = |D|, digits a complete residue system")

    # ---- 2. |S_k| = C^k exactly (this is what the residue condition
    #         buys, and the check that catches a wrong digit set) -----
    for key, (label, (M, D), _meta) in RADIX_PRESETS.items():
        C = len(D)
        kmax = min(6, max_level(C))
        for k in range(1, kmax + 1):
            S = radix_points(M, D, k)
            uniq = len(np.unique(S, axis=0))
            if uniq != C ** k or len(S) != C ** k:
                raise AssertionError(
                    f"{label}: level {k} has {uniq} distinct points of "
                    f"{len(S)}, expected {C ** k}")
    print("radix point sets: |S_k| = C^k distinct points, all presets")

    # gaskets drop exactly the digits asked for
    M, D = RADIX_PRESETS['CUBE'][1]
    for h in (1, 2, 4):
        for k in (1, 2, 3):
            S = radix_points(M, D, k, holes=h)
            if len(np.unique(S, axis=0)) != (8 - h) ** k:
                raise AssertionError(
                    f"cube gasket -{h}: level {k} has "
                    f"{len(np.unique(S, axis=0))} points, expected "
                    f"{(8 - h) ** k}")
    print("gaskets: (C-h)^k cells at every level")

    # ---- 3. the exact body: volume exactly 1, and watertight --------
    for key in ('ABC_124', 'TWIN_A', 'TWIN_G', 'CUBE'):
        label = RADIX_PRESETS[key][0]
        V, F, info = build_radix(preset=key, level=4, output='EXACT')
        if abs(info['volume'] - 1.0) > 1e-12:
            raise AssertionError(
                f"{label}: level-4 volume {info['volume']} != 1")
        nb, nm = edge_stats(F)
        if nb:
            raise AssertionError(
                f"{label}: {nb} boundary edges -- the surface has a "
                f"hole in it")
        ext = float((V.max(axis=0) - V.min(axis=0)).max())
        if abs(ext - 2.0) > 1e-6:
            raise AssertionError(f"{label}: {ext:.4f} across, expected "
                                 f"a 2 m fit")
        note = (f", {nm} non-manifold edges (cubes touching edge to "
                f"edge)" if nm else "")
        print(f"{label:24s}: level {info['level']}, {info['cells']:6d} "
              f"cells, volume {info['volume']:.6f}, closed{note}")

    # ---- 3a. the papers' own arithmetic and tables -------------------
    # Thuswaldner-Zhang Remark 1.3 reduced to arithmetic, checked
    # against the neighbour counts computed for these tiles: only
    # (1,2,4) of the originally shipped six had 14 neighbours.
    for (A, B, C), want in (((1, 1, 2), False), ((1, 1, 3), False),
                            ((1, 2, 3), False), ((1, 2, 4), True),
                            ((1, 3, 4), False), ((2, 2, 3), False),
                            ((1, 2, 5), True), ((1, 2, 8), True),
                            ((1, 3, 6), True), ((1, 4, 8), True),
                            ((2, 3, 4), True)):
        if abc_has_14_neighbours(A, B, C) != want:
            raise AssertionError(
                f"ABC ({A},{B},{C}): Remark 1.3 gives "
                f"{abc_has_14_neighbours(A, B, C)}, expected {want}")
    # the ball theorem additionally needs A = 1
    if abc_is_ball(2, 3, 4):
        raise AssertionError("ABC (2,3,4) has 14 neighbours but A = 2, "
                             "so Theorem 1.1 must not claim it")
    if not abc_is_ball(1, 2, 4):
        raise AssertionError("ABC (1,2,4) is the paper's own worked "
                             "example of a ball")
    balls = [k for k, v in RADIX_PRESETS.items()
             if isinstance(v[2], tuple) and abc_is_ball(*v[2])]
    print(f"Thuswaldner-Zhang Remark 1.3 reproduced; proven 3-balls "
          f"shipped: {', '.join(sorted(balls))}")

    # (1,2,8) factors as (x+2)(x^2-x+4), so all three eigenvalues have
    # modulus exactly 2 -- by Bandt Prop 2.2 that makes it conjugate to
    # a SELF-SIMILAR tile, and its level-k cells distort only
    # polynomially instead of exponentially
    M, D = RADIX_PRESETS['ABC_128'][1]
    ev = np.abs(np.linalg.eigvals(M.astype(float)))
    if float(ev.max() / ev.min()) > 1.0 + 1e-9:
        raise AssertionError(
            f"ABC (1,2,8) should have equal eigenvalue moduli, got "
            f"{np.sort(ev)}")
    asp = []
    for k in (4, 12):
        Ak = np.linalg.inv(np.linalg.matrix_power(M.astype(float), k))
        sv = np.linalg.svd(Ak, compute_uv=False)
        asp.append(float(sv[0] / sv[-1]))
    if asp[1] > 4.0 * asp[0]:
        raise AssertionError(
            f"ABC (1,2,8) cells went {asp[0]:.1f} -> {asp[1]:.1f} from "
            f"level 4 to 12; equal moduli should keep that slow")
    print(f"ABC (1,2,8): eigenvalue moduli all 2, cell aspect "
          f"{asp[0]:.1f} -> {asp[1]:.1f} over levels 4-12")

    # every preset's topology note must come from the tables, not thin
    # air -- and F and G must stay silent, since Bandt gives no details
    for key, v in RADIX_PRESETS.items():
        note = radix_topology(v[2])
        if key in ('TWIN_F', 'TWIN_G'):
            if note:
                raise AssertionError(
                    f"{v[0]}: Bandt provides no details for F and G, so "
                    f"nothing should be claimed")
        elif not note:
            raise AssertionError(f"{v[0]}: no topology note")
    print("topology notes present for every preset but F and G")

    # ---- 3b. the closed-form bounding box -----------------------------
    # the cube tile is [0,1]^3 exactly, which pins the support series
    M, D = RADIX_PRESETS['CUBE'][1]
    lo, hi = tile_support_bbox(M, D)
    if (float(np.max(np.abs(lo))) > 1e-12
            or float(np.max(np.abs(hi - 1.0))) > 1e-12):
        raise AssertionError(f"the cube tile's support box came out "
                             f"{lo} .. {hi}, expected [0,1]^3")
    # Twindragon A is Bandt's non-fractal case, and in this lattice
    # basis its tile is exactly the unit cube: M [0,1]^3 is the box
    # spanned by Me1 = e2, Me2 = e3, Me3 = 2e1, i.e. [0,2]x[0,1]x[0,1],
    # which is precisely [0,1]^3 union ([0,1]^3 + e1) = T + D.  Support
    # box [0,1]^3 together with volume 1 pins it down.
    M, D = RADIX_PRESETS['TWIN_A'][1]
    lo, hi = tile_support_bbox(M, D)
    if (float(np.max(np.abs(lo))) > 1e-12
            or float(np.max(np.abs(hi - 1.0))) > 1e-12):
        raise AssertionError(f"twindragon A's support box came out "
                             f"{lo} .. {hi}; it should be the unit "
                             f"cube")
    print("support series: cube and twindragon A both exactly [0,1]^3 "
          "(A is Bandt's non-fractal case)")

    # ---- 3c. gasket digit sets stay three-dimensional -----------------
    # dropping digits off the end must never leave them coplanar: the
    # cube's naive i,j,k order put all four i = 0 digits last, so a
    # four-hole gasket collapsed to a flat sheet
    for key, (label, (M, D), _meta) in RADIX_PRESETS.items():
        hmax = max_holes(M, D)
        for h in range(0, hmax + 1):
            keep = D[:len(D) - h]
            rk = attractor_rank(M, keep)
            if rk != 3:
                raise AssertionError(
                    f"{label}: {h} holes leaves an attractor of "
                    f"dimension {rk} -- it would collapse")
        if hmax + 1 <= len(D) - 2:
            over = D[:len(D) - (hmax + 1)]
            if attractor_rank(M, over) == 3:
                raise AssertionError(
                    f"{label}: {hmax + 1} holes still fills three "
                    f"dimensions, so the clamp is too tight")
    hmax_cube = max_holes(*RADIX_PRESETS['CUBE'][1])
    if hmax_cube < 4:
        raise AssertionError(
            f"the cube should support a 4-hole gasket, got {hmax_cube}")
    V, F, info = build_radix(preset='CUBE', holes=4, output='VOXEL',
                             resolution=48, points=120000)
    span = V.max(axis=0) - V.min(axis=0)
    if float(span.max() / max(span.min(), 1e-12)) > 4.0:
        raise AssertionError(
            f"the 4-hole cube gasket has aspect "
            f"{span.max() / span.min():.1f} -- it collapsed")
    print(f"gasket digit sets: rank 3 throughout, cube takes up to "
          f"{hmax_cube} holes")

    # ---- 3d. sampled tiles reach their true extent -------------------
    # this is the check the old per-step metric could not make: compare
    # against the closed-form support box, not against the last level
    # The tips of these tiles are reached only by rare addresses along
    # thin fibres -- Bandt says as much of cases F and G -- so no finite
    # sample reaches 100%.  What must hold is that the body stays
    # INSIDE the true tile and gets closer as the sample grows.
    for key in ('ABC_124', 'ABC_134', 'TWIN_A', 'TWIN_D', 'TWIN_G',
                'CUBE'):
        label = RADIX_PRESETS[key][0]
        V, F, info = build_radix(preset=key, output='VOXEL',
                                 resolution=64, points=200000)
        fid = info['fidelity']
        if fid > 1.05:
            raise AssertionError(
                f"{label}: the sampled body is {100 * fid:.0f}% of the "
                f"true extent -- it cannot exceed the tile")
        if fid < 0.75:
            raise AssertionError(
                f"{label}: the sampled tile reaches only "
                f"{100 * fid:.0f}% of its true extent")
        nb, nm = edge_stats(F)
        if nb:
            raise AssertionError(f"{label}: {nb} boundary edges in the "
                                 f"sampled tile")
        print(f"{label:34s}: sampled to {100 * fid:5.1f}% of the true "
              f"extent, {info['cells']:6d} cells, closed")

    # more samples must get closer -- the check that the shortfall is
    # sampling and not a wrong bounding box
    lean = build_radix(preset='ABC_134', output='VOXEL', resolution=64,
                       points=60000)[2]['fidelity']
    rich = build_radix(preset='ABC_134', output='VOXEL', resolution=64,
                       points=600000)[2]['fidelity']
    if rich <= lean:
        raise AssertionError(
            f"ABC (1,3,4): fidelity went {lean:.3f} -> {rich:.3f} as "
            f"the sample grew tenfold; it should improve")
    print(f"sampling converges: ABC (1,3,4) {100 * lean:.1f}% -> "
          f"{100 * rich:.1f}% on a tenfold sample")

    # ---- 4. the exact level-k body, measured against the truth -------
    # The old check compared each level with the previous one, which is
    # far too weak: the remaining error is the tail of a geometric
    # series with ratio 1/min|lambda|, so for twindragon G a per-step
    # change of 9% hides a body that has reached barely a quarter of
    # the real tile.  Measure against the closed-form support box, and
    # assert the two things that actually matter -- the level-k body
    # must approach the tile from INSIDE and must improve with k.
    for key in ('ABC_124', 'TWIN_B', 'TWIN_G', 'CUBE'):
        label = RADIX_PRESETS[key][0]
        M, D = RADIX_PRESETS[key][1]
        true_span = np.subtract(*reversed(tile_support_bbox(M, D)))
        top = min(10, max_level(len(D)))
        fids = []
        for k in (3, top):
            S = radix_points(M, D, k)
            A = np.linalg.inv(np.linalg.matrix_power(M.astype(float),
                                                     k))
            P = S.astype(float) @ A.T
            span = P.max(axis=0) - P.min(axis=0)
            if float(np.max(span / true_span)) > 1.0 + 1e-9:
                raise AssertionError(
                    f"{label}: the level-{k} body sticks out past the "
                    f"true tile ({span} vs {true_span}) -- M^-k is "
                    f"wrong")
            fids.append(float(np.min(span / true_span)))
        if fids[-1] <= fids[0]:
            raise AssertionError(
                f"{label}: fidelity went {fids[0]:.3f} -> {fids[-1]:.3f} "
                f"as the level rose; it should improve")
        A = np.linalg.inv(np.linalg.matrix_power(M.astype(float), top))
        sv = np.linalg.svd(A, compute_uv=False)
        print(f"{label:34s}: level {top:2d} reaches "
              f"{100 * fids[-1]:5.1f}% of the true extent, cells "
              f"{sv[0] / sv[-1]:8.0f}:1")

    # the aspect blow-up is real and must be REPORTED, not hidden: at
    # its own default level twindragon G's cells are thousands to one
    V, F, info = build_radix(preset='TWIN_G', output='EXACT')
    if info['cell_aspect'] < 100.0:
        raise AssertionError(
            f"twindragon G cells came out {info['cell_aspect']:.0f}:1; "
            f"the laminate warning would never fire")
    if info['fidelity'] > 0.9:
        raise AssertionError(
            f"twindragon G exact mode reached {info['fidelity']:.2f} "
            f"of the true extent; the shortfall warning would never "
            f"fire")
    print(f"exact mode on twindragon G: cells "
          f"{info['cell_aspect']:.0f}:1 and only "
          f"{100 * info['fidelity']:.0f}% of the extent -- both "
          f"correctly flagged")

    # ---- 5. IFS maps are contractions -------------------------------
    for key, (label, fn, dim) in IFS_PRESETS.items():
        mp = fn()
        if dim == 2:
            # the fern's first map is singular (it flattens to the
            # stem), which is contractive but not invertible
            if not all(float(np.linalg.svd(A, compute_uv=False)[0])
                       < 1.0 for A, _, _ in mp):
                raise AssertionError(f"{label}: not every map is a "
                                     f"contraction")
        elif not contractive(mp):
            raise AssertionError(f"{label}: not every map is a "
                                 f"contraction")
    print(f"{len(IFS_PRESETS)} IFS presets: every map a contraction")

    # ---- 5b. the Bandt-Mai-Mesing constructions ---------------------
    # Each is a homothety toward a fixed point composed with a proper
    # rotation, so the linear part must be exactly ratio x orthogonal
    # with determinant +1 -- a reflection would be a different fractal.
    for key, r, count in (('BMM_SIERP', 2.0 / 3.0, 3),
                          ('BMM_TETRA', 0.6, 4),
                          ('BMM_CUBE', 0.625, 8)):
        mp = IFS_PRESETS[key][1]()
        if len(mp) != count:
            raise AssertionError(
                f"{key}: {len(mp)} maps, the paper gives {count}")
        for A, b, _p in mp:
            R = np.asarray(A) / r
            if not np.allclose(R @ R.T, np.eye(3), atol=1e-12):
                raise AssertionError(f"{key}: the linear part is not "
                                     f"{r} times an orthogonal matrix")
            if abs(float(np.linalg.det(R)) - 1.0) > 1e-12:
                raise AssertionError(
                    f"{key}: det {np.linalg.det(R):+.3f} -- a rotation "
                    f"must be proper, a reflection is a different set")
        # each map must fix its own centre
        for A, b, _p in mp:
            fix = np.linalg.solve(np.eye(3) - np.asarray(A), b)
            if not np.all(np.isfinite(fix)):
                raise AssertionError(f"{key}: a map has no fixed point")
    print("Bandt-Mai-Mesing maps: proper rotations at ratios 2/3, 3/5, "
          "5/8 with the stated piece counts")

    # the triangle case must reproduce the paper's f_1 verbatim:
    #   f_1(x1,x2,x3) = (r x1 + 1 - r, -r x3, r x2)
    r = 2.0 / 3.0
    A1, b1, _ = _bmm_sierpinski(3, r)[0]
    x = np.array([0.3, -0.7, 0.2])
    want = np.array([r * x[0] + 1 - r, -r * x[2], r * x[1]])
    if not np.allclose(A1 @ x + b1, want, atol=1e-12):
        raise AssertionError(f"the Sierpinski-in-3D f_1 gives "
                             f"{A1 @ x + b1}, the paper gives {want}")
    print("Sierpinski-in-3D f_1 matches the paper's formula exactly")

    # Each construction is built around a 3-fold rotation, so its
    # attractor has to be invariant under that rotation -- a strong
    # check on the axes and the composition order, which a formula
    # comparison of f_1 alone would not catch.
    def _cloud_sig(P, n=28):
        P = np.asarray(P, dtype=float)
        P = (P - P.mean(axis=0))
        P = P / max(float(np.abs(P).max()), 1e-12)
        idx = np.clip(((P + 1.0) * 0.5 * n).astype(int), 0, n - 1)
        grid = np.zeros((n, n, n), dtype=bool)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return grid

    def _cloud_overlap(a, b):
        return float((a & b).sum()) / max(int((a | b).sum()), 1)

    tetra_axis = _BMM_TETRA_V.mean(axis=0) - _BMM_TETRA_V[0]
    for key, axis in (('BMM_SIERP', (0.0, 0.0, 1.0)),
                      ('BMM_TETRA', tetra_axis),
                      ('BMM_CUBE', (1.0, 1.0, 1.0))):
        P = chaos_game(IFS_PRESETS[key][1](), points=400000, seed=2,
                       transient=300)
        R = _rot(axis, 2.0 * math.pi / 3.0)
        turned = _cloud_overlap(_cloud_sig(P), _cloud_sig(P @ R.T))
        if turned < 0.85:
            raise AssertionError(
                f"{key} should be invariant under a 120 degree turn "
                f"about {np.round(np.asarray(axis, float), 2)}, but "
                f"the overlap is only {turned:.2f}")
        # a turn that is NOT a symmetry has to score much lower, or the
        # test would pass on any blob
        Rc = _rot(axis, math.radians(50.0))
        control = _cloud_overlap(_cloud_sig(P), _cloud_sig(P @ Rc.T))
        if control > 0.75 * turned:
            raise AssertionError(
                f"{key}: a 50 degree turn scores {control:.2f} against "
                f"{turned:.2f} for the real symmetry -- the test is "
                f"not discriminating")
    print("Bandt-Mai-Mesing attractors: all three invariant under "
          "their own 3-fold rotation")

    # "When A is centrally symmetric, as in Figures 9 and 5, it
    # coincides with its reverse."  The modified cube is that case; the
    # modified tetrahedron is not (the paper's Figure 1 IS the reverse
    # of its Figure 8, and looks quite different).  Compare occupancy,
    # not bounding boxes -- a set and its mirror share their extents.
    def _occupancy(V, n=24):
        idx = np.clip(((np.asarray(V) + 1.0) * 0.5 * n).astype(int),
                      0, n - 1)
        grid = np.zeros((n, n, n), dtype=bool)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return grid

    def _overlap(a, b):
        return float((a & b).sum()) / max(int((a | b).sum()), 1)

    same = {}
    for key in ('BMM_CUBE', 'BMM_TETRA', 'BMM_SIERP'):
        fwd = build_ifs(preset=key, output='VOXEL', points=300000,
                        resolution=48, seed=5)[0]
        rev = build_ifs(preset=key, output='VOXEL', points=300000,
                        resolution=48, seed=5, reverse=True)[0]
        same[key] = _overlap(_occupancy(fwd), _occupancy(rev))
    if same['BMM_CUBE'] < 0.6:
        raise AssertionError(
            f"the modified cube is centrally symmetric, so it should "
            f"coincide with its reverse, but they overlap only "
            f"{same['BMM_CUBE']:.2f}")
    for key in ('BMM_TETRA', 'BMM_SIERP'):
        if same[key] > 0.5 * same['BMM_CUBE']:
            raise AssertionError(
                f"{key} is not centrally symmetric, so its reverse "
                f"should differ, but they overlap {same[key]:.2f} "
                f"against the cube's {same['BMM_CUBE']:.2f}")
    print(f"reverse fractals: cube overlaps its reverse "
          f"{same['BMM_CUBE']:.2f} (centrally symmetric), tetrahedron "
          f"{same['BMM_TETRA']:.2f} and triangle "
          f"{same['BMM_SIERP']:.2f} (not)")

    # ---- 6. deterministic solid copies ------------------------------
    V, F, info = build_ifs(preset='SIERP_TETRA', output='SOLIDS',
                           depth=5)
    if info['copies'] != 4 ** 5:
        raise AssertionError(f"Sierpinski tetrahedron depth 5 made "
                             f"{info['copies']} copies, expected "
                             f"{4 ** 5}")
    if len(F) != 4 ** 5 * 4:
        raise AssertionError(f"expected {4 ** 5 * 4} faces, got "
                             f"{len(F)}")
    print(f"Sierpinski tetrahedron: {info['copies']} copies, "
          f"{len(F)} faces")

    V, F, info = build_ifs(preset='MENGER', output='SOLIDS',
                           seed_solid='CUBE', depth=3)
    if info['copies'] != 20 ** 3:
        raise AssertionError(f"Menger depth 3 made {info['copies']} "
                             f"copies, expected {20 ** 3}")
    print(f"Menger sponge (solid copies): {info['copies']} copies")

    # ---- 7. chaos game: determinism and watertight voxels -----------
    a = build_ifs(preset='SIERP_TETRA', output='VOXEL', points=60000,
                  resolution=48, seed=7)
    b = build_ifs(preset='SIERP_TETRA', output='VOXEL', points=60000,
                  resolution=48, seed=7)
    if (not np.array_equal(a[0], b[0])
            or not np.array_equal(np.asarray(a[1]), np.asarray(b[1]))):
        raise AssertionError("the chaos game is not reproducible from "
                             "its seed")
    c = build_ifs(preset='SIERP_TETRA', output='VOXEL', points=60000,
                  resolution=48, seed=8)
    if np.array_equal(a[0], c[0]):
        raise AssertionError("two different seeds gave an identical "
                             "mesh")
    nb, nm = edge_stats(a[1])
    if nb:
        raise AssertionError(f"voxel attractor: {nb} boundary edges")
    print(f"chaos game: reproducible from the seed, {len(a[1])} voxel "
          f"faces, closed ({nm} non-manifold edges)")

    # the attractor stays inside a bounded region
    P = chaos_game(IFS_PRESETS['SIERP_TETRA'][1](), points=20000,
                   seed=3)
    if float(np.max(np.abs(P))) > 3.0:
        raise AssertionError("the Sierpinski attractor escaped its "
                             "bounding box")

    # ---- 8. smooth contour ------------------------------------------
    V, F, info = build_ifs(preset='SIERP_TETRA', output='ISO',
                           points=200000, resolution=64, cover=0.9,
                           largest_only=False)
    if not len(F) or not np.all(np.isfinite(V)):
        raise AssertionError("the smooth contour came out empty or "
                             "non-finite")
    ext = float((V.max(axis=0) - V.min(axis=0)).max())
    if abs(ext - 2.0) > 1e-6:
        raise AssertionError(f"smooth contour is {ext:.4f} across, "
                             f"expected a 2 m fit")
    print(f"smooth contour: {len(F)} tris, level {info['level']:.1f}")

    # ---- 9. custom map parser ---------------------------------------
    mp = parse_maps("0.5 0 0 0 0.5 0 0 0 0.5 | 1 2 3 | 0.4; "
                    "0.5 0 0 0 0.5 0 0 0 0.5 | 0 0 0")
    if len(mp) != 2:
        raise AssertionError(f"parser made {len(mp)} maps, expected 2")
    if not np.allclose(mp[0][1], [1, 2, 3]) or mp[0][2] != 0.4:
        raise AssertionError("first custom map parsed wrongly")
    if mp[1][2] != 1.0:
        raise AssertionError("a missing probability should default "
                             "to 1")
    for bad in ("", "0.5 0 0 | 1 2 3", "1 2 3 4 5 6 7 8 9 | 1 2",
                "a b c d e f g h i | 1 2 3"):
        try:
            parse_maps(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_maps({bad!r}) should have raised")

    # ---- 10. planar systems, detected and meshed as reliefs ---------
    # A two-dimensional system has singular maps in R^3, so solid
    # copies would flatten the seed to a plate.  Planarity is MEASURED
    # from the attractor, not read off the preset label, so a custom
    # flat map set is handled the same way.
    planar = [k for k, v in IFS_PRESETS.items() if v[2] == 2]
    if not planar:
        raise AssertionError("no planar presets to test")
    for key in planar:
        label = IFS_PRESETS[key][0]
        mp = IFS_PRESETS[key][1]()
        if any(abs(float(np.linalg.det(A))) > 1e-12 for A, _, _ in mp):
            raise AssertionError(f"{label}: a planar system's maps "
                                 f"should be singular in 3-D")
        try:
            build_ifs(preset=key, output='SOLIDS', depth=3)
        except ValueError as e:
            if 'planar' not in str(e):
                raise AssertionError(
                    f"{label} was refused for solid copies, but "
                    f"unhelpfully: {e}")
        else:
            raise AssertionError(
                f"solid copies of {label} should have been refused")
        V, F, info = build_ifs(preset=key, output='RELIEF',
                               points=200000, plane_resolution=256)
        if not info.get('planar'):
            raise AssertionError(f"{label} was not detected as planar")
        nb, nm = edge_stats(F)
        if nb:
            raise AssertionError(f"{label}: {nb} boundary edges -- the "
                                 f"relief slab is not closed")
        if _signed_volume(V, F) <= 0.0:
            raise AssertionError(f"{label}: the relief is inside out")
        span = V.max(axis=0) - V.min(axis=0)
        thin = float(np.min(span))
        wide = float(np.max(span))
        if thin > 0.05 * wide:
            raise AssertionError(
                f"{label} should be flat, but its thinnest axis spans "
                f"{thin:.3f} against {wide:.3f}")
        # embedded in the xz-plane, so y is the thin one
        if float(np.argmin(span)) != 1:
            raise AssertionError(
                f"{label} should be flat in y (upright in a z-up "
                f"world), but the thin axis is {int(np.argmin(span))}")
        print(f"{label:28s}: planar, closed relief, "
              f"{info['cells']:6d} cells, {thin / wide:.4f} thick")

    # a three-dimensional system must refuse the relief output
    try:
        build_ifs(preset='SIERP_TETRA', output='RELIEF')
    except ValueError as e:
        if 'planar' not in str(e):
            raise AssertionError(f"unhelpful refusal: {e}")
    else:
        raise AssertionError("a relief of a 3-D system should have "
                             "been refused")
    print("relief output refused for three-dimensional systems")

    # ---- 11. the Maps field round-trips every preset ----------------
    # Selecting a system loads its maps into the editable field, so
    # format_maps has to be an exact inverse of parse_maps or a preset
    # would quietly change the moment it was displayed.
    for key, (label, fn, dim) in IFS_PRESETS.items():
        mp = fn()
        back = parse_maps(format_maps(mp))
        if len(back) != len(mp):
            raise AssertionError(
                f"{label}: the field round-tripped {len(mp)} maps into "
                f"{len(back)}")
        for (A, b, pr), (A2, b2, pr2) in zip(mp, back):
            if (not np.allclose(A, A2, atol=1e-8)
                    or not np.allclose(b, b2, atol=1e-8)
                    or abs(pr - pr2) > 1e-8):
                raise AssertionError(
                    f"{label}: a map changed on its way through the "
                    f"Maps field")
    print(f"Maps field: all {len(IFS_PRESETS)} presets round-trip "
          f"exactly")

    # a non-contractive system must be refused, not silently diverge
    try:
        build_ifs(maps=parse_maps("2 0 0 0 2 0 0 0 2 | 0 0 0"),
                  output='SOLIDS', depth=2)
    except ValueError:
        pass
    else:
        raise AssertionError("an expanding map set should have been "
                             "refused")

    # and so must a digit set that is not a residue system
    try:
        build_radix(preset='CUSTOM',
                    custom=(companion(1, 2, 4),
                            np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0),
                                      (4, 0, 0)], dtype=np.int64)))
    except ValueError:
        pass
    else:
        raise AssertionError("digits 0 and 4 are congruent mod M; the "
                             "build should have been refused")
    print("parsers and validity checks reject what they should")

    print("RESULT: OK")
