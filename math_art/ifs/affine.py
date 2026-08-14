# Affine iterated function systems and the chaos game.
#
# Part of the Math Art IFS engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# A finite set of contractive affine maps w_i(x) = A_i x + b_i has, by
# Hutchinson's theorem, a unique non-empty compact attractor satisfying
# K = union of w_i(K).  Two ways to realise it: deterministically, by
# applying every map k times to a seed solid (exact, m^k copies), or by
# the chaos game -- iterate a single point under maps drawn at random
# with the given probabilities, discarding a burn-in, and the orbit fills
# the attractor.
#
# Contractivity is checked by the largest SINGULAR value, not the
# spectral radius: a matrix can have both eigenvalues inside the unit
# disc and still expand a vector on a single step, so the spectral test
# admits systems whose chaos game visibly diverges.  `contractive` is
# the strict per-step test and `spectrally_contractive` the weaker
# eventual one, used where only the limit matters.
#
# References:
# - J. E. Hutchinson, "Fractals and self similarity", Indiana University
#   Mathematics Journal 30, 1981, pp. 713-747 -- existence and
#   uniqueness of the attractor.
# - M. F. Barnsley, "Fractals Everywhere", 2nd ed., Academic Press,
#   1993 -- the chaos game, and the fern whose two-dimensional maps are
#   offered here embedded in the z = 0 plane.
# - C. Bandt, Mai The Duy and M. Mesing, "Three-Dimensional Fractals",
#   The Mathematical Intelligencer 32, 2010, pp. 12-18.
#   doi:10.1007/s00283-009-9110-6 -- the BMM_* map families.

import math

import numpy as np

from .voxel import (MAX_CELLS, _as_quads, _occupied_cells, _signed_volume,
                    _toolkit,
                    blur_density, center_fit, keep_largest, orient_outward,
                    voxel_surface)


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


def _selftest():
    """Hutchinson's hypotheses and the chaos game's contract."""
    ok = True

    # Every preset must be a genuine contraction -- by the largest
    # SINGULAR value, not the spectral radius.  A matrix can have both
    # eigenvalues inside the unit disc and still expand a vector on a
    # single step, and such a system's chaos game visibly diverges.
    bad = []
    for key, (label, fn, dim) in IFS_PRESETS.items():
        maps = fn()
        s = max(float(np.linalg.svd(np.asarray(A, float), compute_uv=False)[0])
                for A, _b, _p in maps)
        if not contractive(maps) or s >= 1.0:
            bad.append(f"{key}:{s:.3f}")
    good = not bad
    ok &= good
    print(f"affine: all {len(IFS_PRESETS)} presets contractive by singular "
          f"value {'OK' if good else 'FAIL ' + ','.join(bad)}")

    # The strict test must be at least as strong as the spectral one:
    # anything contractive is spectrally contractive, never the reverse.
    shear = [(np.array([[0.5, 4.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
              np.zeros(3), 1.0)]
    good = spectrally_contractive(shear) and not contractive(shear)
    ok &= good
    print(f"affine: a shear with rho=0.5 but sigma>1 is spectrally but not "
          f"strictly contractive {'OK' if good else 'FAIL'}")

    # The chaos game is seeded, so it must be reproducible -- and a
    # different seed must actually give a different orbit.
    mp = IFS_PRESETS['SIERP_TETRA'][1]()
    a = chaos_game(mp, points=4000, seed=11)
    b = chaos_game(mp, points=4000, seed=11)
    c = chaos_game(mp, points=4000, seed=12)
    good = np.array_equal(a, b) and not np.array_equal(a, c)
    ok &= good
    print(f"affine: chaos game reproducible from its seed, and seed-sensitive "
          f"{'OK' if good else 'FAIL'}")

    # Moran's theorem: for a self-similar IFS of m maps of common ratio
    # r satisfying the open set condition, the attractor has Hausdorff
    # (and box-counting) dimension log m / log(1/r).  Box-counting the
    # chaos-game orbit must recover it -- a test that fails loudly if the
    # maps, the ratios or the orbit are wrong, unlike a bounding-box
    # check that any contraction passes.
    #
    # Only the SEPARATED gaskets are checked.  LEVY and DRAGON have
    # overlapping pieces, so the open set condition fails and log m /
    # log(1/r) = 2 is merely an upper bound the finite orbit does not
    # attain (it measures about 1.78).  Asserting 2 there would be
    # asserting something false.
    def _box_dim(P, scales=(8, 16, 32, 64)):
        lo, hi = P.min(axis=0), P.max(axis=0)
        span = np.maximum(hi - lo, 1e-12)
        counts = []
        for r in scales:
            idx = np.clip(((P - lo) / span * r).astype(int), 0, r - 1)
            counts.append(len(set(map(tuple, idx))))
        return float(np.polyfit(np.log(np.asarray(scales, float)),
                                np.log(np.asarray(counts, float)), 1)[0])

    bad = []
    for key in ('SIERP_TETRA', 'MENGER', 'SIERP_TRI', 'KOCH'):
        maps = IFS_PRESETS[key][1]()
        r = float(np.linalg.svd(np.asarray(maps[0][0], float),
                                compute_uv=False)[0])
        want = math.log(len(maps)) / math.log(1.0 / r)
        got = _box_dim(chaos_game(maps, points=400000, seed=5))
        if abs(got - want) > 0.15:
            bad.append(f"{key}:{got:.2f}!={want:.2f}")
    good = not bad
    ok &= good
    print(f"affine: box dimension matches log m / log(1/r) on the separated "
          f"gaskets {'OK' if good else 'FAIL ' + ','.join(bad)}")

    # parse_maps and format_maps must round-trip every preset.
    bad = []
    for key, (label, fn, dim) in IFS_PRESETS.items():
        mp2 = parse_maps(format_maps(fn()))
        if len(mp2) != len(fn()):
            bad.append(key)
            continue
        for (A1, b1, p1), (A2, b2, p2) in zip(fn(), mp2):
            if (not np.allclose(A1, A2, atol=1e-9)
                    or not np.allclose(b1, b2, atol=1e-9)):
                bad.append(key)
                break
    good = not bad
    ok &= good
    print(f"affine: all {len(IFS_PRESETS)} presets round-trip through the "
          f"Maps field {'OK' if good else 'FAIL ' + ','.join(sorted(set(bad)))}")

    # Every seed solid must be wound outward, or the emitted copies are
    # inside out.
    bad = [n for n, (sv, sf) in SEEDS.items()
           if _signed_volume(np.asarray(sv, float), sf) <= 0]
    good = not bad
    ok &= good
    print(f"affine: seed solids {', '.join(sorted(SEEDS))} all wound outward "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("affine self-test failed")
