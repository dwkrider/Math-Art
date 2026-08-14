# Self-affine radix tiles: expanding integer matrix + digit set.
#
# Part of the Math Art IFS engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# An expanding integer matrix M (every eigenvalue of modulus > 1)
# together with a digit set D subset of Z^3 that is a COMPLETE RESIDUE
# SYSTEM for Z^3 / M Z^3 (so |D| = |det M| = C) determines a unique
# compact set T with
#
#     M T = T + D,   equivalently   T = union over d in D of M^-1 (T+d)
#
# and that T tiles R^3 by the lattice Z^3.  The level-k approximation is
# EXACT rather than sampled: iterate the integer point set S_0 = {0},
# S_(j+1) = D + M S_j -- which has exactly C^k points after k steps,
# distinctness guaranteed by the residue condition -- mesh those unit
# cubes, and apply the single linear map M^-k.  A linear map preserves
# watertightness, so the result is a closed surface of volume exactly 1
# whatever the level.
#
# In three dimensions the eigenvalues of M generally have DIFFERENT
# moduli, so M^-k contracts unevenly and the level-k approximation is
# genuinely thin in the slow direction.  That is not a defect -- it is
# what "self-affine" rather than "self-similar" means.  (A self-affine
# tile is conjugate to a self-similar one exactly when all eigenvalues
# of M share a modulus.)
#
# References:
# - C. Bandt, "Self-similar sets 5. Integer matrices and fractal tilings
#   of R^n", Proceedings of the American Mathematical Society 112, 1991,
#   pp. 549-562 -- the integer-matrix plus residue-digit-set theorem
#   behind every tile here.
# - C. Bandt, "Combinatorial topology of three-dimensional self-affine
#   tiles", arXiv:1002.0710, 2010 -- the seven twindragon cases.
# - J. M. Thuswaldner and S.-Q. Zhang, "On self-affine tiles that are
#   homeomorphic to a ball", arXiv:2107.12076 -- the ABC normal form,
#   the 14-neighbour hypothesis of Thm 1.1 and the arithmetic of
#   Remark 1.3.

import math

import numpy as np

from .affine import chaos_game, spectrally_contractive
from .voxel import (MAX_CELLS, _as_quads, _toolkit, blur_density, center_fit,
                    fill_pinholes, keep_largest, orient_outward,
                    voxel_surface)


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


def _selftest():
    """Bandt's hypotheses, and the exactness the radix construction
    claims: C^k distinct points at level k, and volume exactly 1."""
    ok = True

    # Bandt's theorem needs BOTH hypotheses: M expanding, and D a
    # complete residue system for Z^3 / M Z^3 with |D| = |det M|.
    bad = []
    for key, (label, (M, D), _meta) in RADIX_PRESETS.items():
        if not is_expanding(M):
            bad.append(f"{key}:not expanding")
        elif not is_residue_system(M, D):
            bad.append(f"{key}:not a residue system")
        elif len(D) != abs(int(round(np.linalg.det(M.astype(float))))):
            bad.append(f"{key}:|D|!=|det M|")
    good = not bad
    ok &= good
    print(f"radix: all {len(RADIX_PRESETS)} presets expanding with a complete "
          f"residue system {'OK' if good else 'FAIL ' + ','.join(bad)}")

    # The residue condition is exactly what makes the level-k point set
    # have C^k DISTINCT points -- that is the "exact, not sampled" claim.
    bad = []
    for key, (label, (M, D), _meta) in RADIX_PRESETS.items():
        C = len(D)
        for k in (2, 3):
            S = radix_points(M, D, k)
            if len(S) != C ** k or len(set(map(tuple, S))) != C ** k:
                bad.append(f"{key}@{k}:{len(S)}!={C**k}")
    good = not bad
    ok &= good
    print(f"radix: |S_k| = C^k distinct points, every preset "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # A digit set that is NOT a residue system must be refused: digits 0
    # and 4 are congruent mod the companion matrix of x^3+x^2+2x+4.
    try:
        build_radix(preset='CUSTOM',
                    custom=(companion(1, 2, 4),
                            np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0),
                                      (4, 0, 0)], dtype=np.int64)))
    except ValueError:
        good = True
    else:
        good = False
    ok &= good
    print(f"radix: a non-residue digit set is refused "
          f"{'OK' if good else 'FAIL'}")

    # The tile tiles R^3 by Z^3, so it has volume exactly 1 -- and the
    # level-k mesh is that tile under a single linear map M^-k, which
    # preserves the property.  Checked before the fit to the 2 m cube,
    # via the unscaled voxel cell count: C^k cells of volume |det M|^-k.
    bad = []
    for key in ('CUBE', 'TWIN_A', 'ABC_123'):
        M, D = RADIX_PRESETS[key][1]
        C, detM = len(D), abs(float(np.linalg.det(M.astype(float))))
        for k in (2, 3):
            vol = len(radix_points(M, D, k)) / detM ** k
            if abs(vol - 1.0) > 1e-9:
                bad.append(f"{key}@{k}:{vol:.6f}")
    good = not bad
    ok &= good
    print(f"radix: tile volume exactly 1 at every level "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # Gaskets: dropping the last h digits leaves (C-h)^k cells.
    M, D = RADIX_PRESETS['CUBE'][1]
    bad = []
    for h in (1, 2, 4):
        for k in (2, 3):
            n = len(radix_points(M, D[:len(D) - h], k))
            if n != (len(D) - h) ** k:
                bad.append(f"h={h},k={k}:{n}")
    good = not bad
    ok &= good
    print(f"radix: gaskets have (C-h)^k cells "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # Thuswaldner-Zhang: the ABC tile is a closed 3-ball when
    # 1 = A <= B < C AND it has 14 neighbours.  The neighbour count is a
    # hypothesis, not a consequence -- their Remark 1.4 conjectures the
    # opposite without it -- so both halves are checked.
    good = abc_is_ball(1, 2, 4) and not abc_is_ball(2, 3, 4)
    ok &= good
    print(f"radix: Thuswaldner-Zhang ball criterion needs both hypotheses "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("radix self-test failed")
