# Closed-form soft cells -- the analytic members of the soft-cell family.
#
# Part of the Math Art soft-cell engine (`math_art/softcell/`).  Numpy only --
# no `bpy` -- so the engine imports and self-tests headlessly; the registered
# Blender operator stays in the flat `soft_cell_generator.py` front-end.
#
# A SOFT CELL fills space without gaps or overlaps while carrying the minimal
# possible number of sharp corners.  In three dimensions that minimum is ZERO:
# a cell can tile space with no corners at all, every boundary point lying on
# some smooth curve of the boundary.  The cells built here are the ones for
# which the literature gives an explicit formula, so they need no solver:
#
#   SADDLE    the square prism |x|,|y| <= 1 cut above and below by the two
#             saddle surfaces z = sqrt(1-x^2) - sqrt(1-y^2) +/- 1.  Corner
#             count zero.  Tiles by the translations 2x, 2y, 2z.  This is the
#             earliest explicit corner-free space-filler in the literature
#             (Domokos, G. Horvath and Regos 2023, Figure 4) and the simplest
#             to reason about: the cap's slope dz/dx = -x/sqrt(1-x^2) diverges
#             as x -> +/-1, so the cap meets the vertical side wall TANGENTIALLY
#             rather than in an edge.  That tangency is the whole trick.
#
#   TRIPRISM  the soft triangular prism of the Domokos-Goriely-Horvath-Regos
#             supplementary information, section 2B.1.  A closed "leading
#             curve" Gamma runs around the three walls of a triangular prism,
#             rising and falling as a sine in the wall's own coordinates; a
#             quadratic Bezier sweep carries it to the axis to make the cap.
#
#   HEXPRISM  the same construction on a regular hexagon (SI section 2B.3),
#             built from the two prototype arcs gamma' and gamma'' fitted
#             alternately to the six edges.
#
# All three are SOFT Z-CELLS: a prism segmented by a smooth manifold and its
# own z-translate.  That is also why they are the right thing to build first --
# they exercise the replication, gap-scaling and watertightness machinery
# against shapes whose volume and lattice are known exactly.
#
# References:
# - G. Domokos, A. G. Horvath and K. Regos, "A two-vertex theorem for normal
#   tilings", Aequationes Mathematicae 97(1):185-197 (2023).  Figure 4 gives
#   the saddle-cut square prism, the first monohedral tiling of space with no
#   sharp corners at all.
# - G. Domokos, A. Goriely, A. G. Horvath and K. Regos, "Soft cells and the
#   geometry of seashells", PNAS Nexus 3(9):pgae311 (2024).  Supplementary
#   Information section 2 gives the analytic prism cells reproduced here, and
#   section 3A the general definition of a z-cell.
#   https://doi.org/10.1093/pnasnexus/pgae311

import math

import numpy as np

TAU = 2.0 * math.pi

# The three analytic cells, in the order they appear in the operator's enum.
KINDS = ('SADDLE', 'TRIPRISM', 'HEXPRISM')

_SQRT3 = math.sqrt(3.0)


# ------------------------------------------------------------------
# SADDLE -- the saddle-cut square prism
# ------------------------------------------------------------------

def _saddle_height(x, y):
    """The cutting surface z = sqrt(1-x^2) - sqrt(1-y^2).

    Clipped inside the square so the sqrt stays real at the walls, where
    its argument is exactly zero and its slope is infinite.
    """
    return (np.sqrt(np.maximum(0.0, 1.0 - x * x))
            - np.sqrt(np.maximum(0.0, 1.0 - y * y)))


def _saddle_cell(n):
    """Mesh the saddle cell as top cap + bottom cap + four side walls.

    `n` is the number of quads along each side of the square.  Samples are
    graded toward the walls with a sine spacing, because the cap turns
    vertical there and a uniform grid renders the tangency as a visible
    crease.
    """
    # sine grading: dense near +/-1 where |dz/dx| blows up
    t = np.linspace(-1.0, 1.0, n + 1)
    s = np.sin(t * math.pi / 2.0)

    X, Y = np.meshgrid(s, s, indexing='ij')
    Z = _saddle_height(X, Y)

    verts = []
    faces = []

    def grid_block(zoff, flip):
        """Append one cap and return its index grid."""
        base = len(verts)
        idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1) + base
        for i in range(n + 1):
            for j in range(n + 1):
                verts.append((X[i, j], Y[i, j], Z[i, j] + zoff))
        for i in range(n):
            for j in range(n):
                q = (idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1])
                faces.append(q[::-1] if flip else q)
        return idx

    top = grid_block(+1.0, False)
    bot = grid_block(-1.0, True)

    # four side walls, each a quad strip joining the bottom rim to the top
    for a, b, flip in ((bot[0, :], top[0, :], True),
                       (bot[n, :], top[n, :], False),
                       (bot[:, 0], top[:, 0], False),
                       (bot[:, n], top[:, n], True)):
        for k in range(n):
            q = (a[k], a[k + 1], b[k + 1], b[k])
            faces.append(q[::-1] if flip else q)

    basis = np.array([[2.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0],
                      [0.0, 0.0, 2.0]])
    return np.array(verts, float), faces, basis, 8.0


# ------------------------------------------------------------------
# TRIPRISM / HEXPRISM -- the supplementary-information prism cells
# ------------------------------------------------------------------

def _tri_leading_curve(m):
    """Gamma = gamma1 U gamma2 U gamma3 of SI equations (6)-(8).

    The three straight prism edges rise through A(-1,-sqrt3,0), B(2,0,0),
    C(-1,sqrt3,0).  Each gamma_i is a closed-form two-branch curve lying on
    one vertical wall; their union is a single closed zigzag around the
    prism, which the Bezier sweep then carries to the axis.
    """
    def branch(phi, lo):
        c, s = np.cos(phi), np.sin(phi)
        z = _SQRT3 / 2.0 * s
        if lo:      # 0 <= phi <= pi
            return np.stack([0.75 * c + 1.25,
                             _SQRT3 / 4.0 * c - _SQRT3 / 4.0,
                             z], axis=-1)
        return np.stack([-0.75 * c - 0.25,
                         -_SQRT3 / 4.0 * c - 3.0 * _SQRT3 / 4.0,
                         z], axis=-1)

    # gamma1 on wall B->A; gamma2 and gamma3 are its images under the
    # 120-degree rotations that carry one wall onto the next.
    lo = np.linspace(0.0, math.pi, m, endpoint=False)
    hi = np.linspace(math.pi, TAU, m, endpoint=False)
    g1 = np.concatenate([branch(lo, True), branch(hi, False)], axis=0)

    def rot_z(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    r120 = rot_z(TAU / 3.0)
    r240 = r120 @ r120
    # gamma1 runs B -> A (over the midpoint of that wall).  The rotation by
    # 120 degrees sends A -> B -> C, so rot120(gamma1) runs C -> B and
    # rot240(gamma1) runs A -> C.  The closed curve is therefore
    # B -> A -> C -> B, i.e. g1 then rot240 then rot120 -- NOT g1, rot120,
    # rot240, which jumps from A to C and leaves the sweep undefined.
    g_ac = g1 @ r240.T
    g_cb = g1 @ r120.T
    G = np.concatenate([g1, g_ac, g_cb], axis=0)
    step = np.abs(np.diff(np.vstack([G, G[:1]]), axis=0)).max()
    assert step < 0.5, f"triangular leading curve is discontinuous ({step})"
    return G


def _hex_leading_curve(m):
    """The hexagonal leading curve of SI equations (11)-(12).

    The prototype gamma' is fitted to the 1st, 3rd and 5th edges of a
    regular hexagon and gamma'' to the 2nd, 4th and 6th, so consecutive
    edges carry opposite-going arcs -- the alternation that makes the
    faces meet tangentially rather than in a corner.
    """
    # A prototype spans exactly one edge: writing gamma' with phi = pi - t,
    # its along-edge coordinate is u(t) = (sqrt3/2)(1 - cos t), which runs
    # 0 -> sqrt3 (the edge length, equal to the circumradius) as t: 0 -> pi,
    # while z(t) = +/- (sqrt3/2) sin t arcs up or down.  Both du/dt and the
    # horizontal speed vanish at t = 0 and t = pi, so every arc leaves and
    # meets a hexagon corner VERTICALLY -- consecutive arcs then have
    # antiparallel half-tangents there and their union is smooth, which is
    # exactly the softness condition.
    R = _SQRT3
    V = np.array([[R * math.cos(k * TAU / 6.0),
                   R * math.sin(k * TAU / 6.0), 0.0] for k in range(6)])

    t = np.linspace(0.0, math.pi, m, endpoint=False)
    u = (R / 2.0) * (1.0 - np.cos(t))
    w = (R / 2.0) * np.sin(t)

    pieces = []
    for k in range(6):
        p0, p1 = V[k], V[(k + 1) % 6]
        e = (p1 - p0) / np.linalg.norm(p1 - p0)
        sign = 1.0 if k % 2 == 0 else -1.0
        arc = (p0[None, :] + u[:, None] * e[None, :])
        arc[:, 2] = sign * w
        pieces.append(arc)
    G = np.concatenate(pieces, axis=0)
    step = np.abs(np.diff(np.vstack([G, G[:1]]), axis=0)).max()
    assert step < 0.5, f"hexagonal leading curve is discontinuous ({step})"
    return G


def _prism_cell(kind, m, rings, height=1.0):
    """Sweep a leading curve to the axis, stack two copies, wall them in.

    The cap is the quadratic-Bezier surface of SI equation (9),

        r_T(phi, psi) = ((1-psi) G1, (1-psi) G2, (1-psi)^2 G3),

    whose z-component is quadratic in psi where x and y are linear.  That
    is what makes the cap smooth at the apex: the surface arrives there
    with a horizontal tangent plane instead of a cone point.
    """
    G = (_tri_leading_curve(m) if kind == 'TRIPRISM'
         else _hex_leading_curve(m))
    n = len(G)

    verts = []
    faces = []

    def cap(zoff, lower):
        """One Bezier cap: `rings` concentric rings plus a single apex.

        The apex is ONE vertex, not a collapsed ring -- sweeping to psi = 1
        makes every point of the last ring coincide, and emitting that as a
        band of quads leaves a skirt of degenerate faces that breaks the
        orientation census.  Windings are chosen so the bottom cap, the top
        cap and the wall each use every shared directed edge exactly once
        in opposite senses; `_check_closed` enforces it.
        """
        base = len(verts)
        for r in range(rings):
            f = 1.0 - r / rings
            for k in range(n):
                verts.append((f * G[k, 0], f * G[k, 1],
                              f * f * G[k, 2] + zoff))
        apex = len(verts)
        verts.append((0.0, 0.0, zoff))

        def ring(r, k):
            return base + r * n + k

        for r in range(rings - 1):
            for k in range(n):
                k2 = (k + 1) % n
                if lower:
                    faces.append((ring(r, k2), ring(r, k),
                                  ring(r + 1, k), ring(r + 1, k2)))
                else:
                    faces.append((ring(r, k), ring(r, k2),
                                  ring(r + 1, k2), ring(r + 1, k)))
        last = rings - 1
        for k in range(n):
            k2 = (k + 1) % n
            if lower:
                faces.append((ring(last, k2), ring(last, k), apex))
            else:
                faces.append((ring(last, k), ring(last, k2), apex))
        return base

    bot = cap(0.0, True)
    top = cap(height, False)

    # wall: complements both rims -- bottom rim k -> k2 here against the
    # bottom cap's k2 -> k, top rim k2 -> k here against the top cap's k -> k2
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((bot + k, bot + k2, top + k2, top + k))

    if kind == 'TRIPRISM':
        # equilateral triangle of circumradius 2, so side a = 2*sqrt(3)
        # and area 3*sqrt(3).  A triangle tiles the plane by translation
        # only in 180-degree-rotated PAIRS, so the translational unit cell
        # holds TWO prisms and det(basis) = 2 * cell volume.  That factor
        # is deliberate, not a slip; `_cells_per_lattice_cell` records it.
        a = 2.0 * _SQRT3
        area = 3.0 * _SQRT3
        basis = np.array([[a, 0.0, 0.0],
                          [a / 2.0, a * _SQRT3 / 2.0, 0.0],
                          [0.0, 0.0, height]])
    else:
        # regular hexagon of circumradius R = sqrt(3): area (3*sqrt(3)/2) R^2,
        # and adjacent hexagon centres are 2 * apothem = R*sqrt(3) = 3 apart.
        R = _SQRT3
        area = 1.5 * _SQRT3 * R * R
        d = R * _SQRT3
        basis = np.array([[d, 0.0, 0.0],
                          [d / 2.0, d * _SQRT3 / 2.0, 0.0],
                          [0.0, 0.0, height]])
    return np.array(verts, float), faces, basis, area * height


# How many cells fit in one translational lattice cell.  One for the cells
# that tile by translation alone; two for the triangular prism, whose
# partner is a 180-degree rotate.
_cells_per_lattice_cell = {'SADDLE': 1, 'TRIPRISM': 2, 'HEXPRISM': 1}


# ------------------------------------------------------------------
# public entry point
# ------------------------------------------------------------------

def build(kind, resolution=24, rings=8):
    """Return (verts, faces, lattice_basis, exact_volume) for one cell.

    `verts` is an (N,3) float array, `faces` a list of index tuples,
    `lattice_basis` the 3x3 matrix whose rows translate the cell onto its
    neighbours, and `exact_volume` the analytically known cell volume --
    which `_selftest` compares against the meshed volume.
    """
    if kind == 'SADDLE':
        return _saddle_cell(max(4, resolution))
    if kind in ('TRIPRISM', 'HEXPRISM'):
        return _prism_cell(kind, max(6, resolution), max(2, rings))
    raise ValueError(f"unknown analytic cell {kind!r}")


def check_closed(V, faces):
    """Raise unless the mesh is closed AND consistently oriented.

    The test is the directed-edge census: in a closed orientable surface
    every directed edge (a, b) occurs exactly once, its reverse supplied by
    the neighbouring face.  A duplicated directed edge means two faces wind
    the same way across a shared edge; a missing reverse means a hole.  The
    divergence-theorem volume is meaningless without this, and will happily
    return a plausible-looking wrong number instead of failing -- which is
    exactly how a 1/3-of-the-truth volume can slip through unnoticed.
    """
    seen = {}
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            if a == b:
                raise AssertionError(f"degenerate edge {a}->{b}")
            seen[(a, b)] = seen.get((a, b), 0) + 1
    dup = [k for k, c in seen.items() if c != 1]
    missing = [k for k in seen if (k[1], k[0]) not in seen]
    if dup or missing:
        raise AssertionError(
            f"not closed/oriented: {len(dup)} repeated directed edges, "
            f"{len(missing)} without a reverse partner")
    return len(seen) // 2


def mesh_volume(V, faces):
    """Signed volume of a closed polygon-soup mesh, by the divergence
    theorem over a fan triangulation of each face."""
    tot = 0.0
    for f in faces:
        p0 = V[f[0]]
        for i in range(1, len(f) - 1):
            a = V[f[i]] - p0
            b = V[f[i + 1]] - p0
            tot += float(np.dot(p0, np.cross(a, b)))
    return abs(tot) / 6.0


def _selftest():
    # SADDLE: the height between the two caps is exactly 2 everywhere, so
    # the volume is exactly the base area (4) times 2, independent of the
    # mesh resolution -- an exact target, not a converging one.
    for n in (8, 16, 32):
        V, F, B, vol = build('SADDLE', resolution=n)
        num = mesh_volume(V, F)
        assert abs(num - 8.0) < 1e-9, (n, num)
        assert abs(vol - 8.0) < 1e-12
        # the cap is z = f(x,y) +/- 1, so every vertical pair differs by 2
        assert abs(float(np.linalg.det(B)) - 8.0) < 1e-12
    print("SADDLE: volume exactly 8 at n = 8, 16, 32; lattice det 8  OK")

    # the saddle cap must arrive at the wall vertically -- that tangency is
    # what makes the cell corner-free.  Check the slope grows without bound.
    prev = 0.0
    for x in (0.9, 0.99, 0.999, 0.9999):
        slope = abs(-x / math.sqrt(1.0 - x * x))
        assert slope > prev
        prev = slope
    assert prev > 50.0
    print(f"SADDLE: cap slope at the wall diverges (|dz/dx| -> {prev:.1f})  OK")

    # Prisms: the volume is base area times height by Cavalieri, because the
    # top cap is exactly the bottom cap translated.  It comes out EXACT at
    # every resolution, not merely convergent, because each arc's xy-track
    # lies on the straight wall it spans -- so the sampled footprint polygon
    # is the base polygon itself, with extra collinear vertices.  Asserting
    # exactness (rather than a loose tolerance) is what makes this test able
    # to catch a mis-ordered leading curve: an earlier draft concatenated the
    # three wall arcs discontinuously and produced a clean-looking but wrong
    # volume, which a 1e-3 tolerance would have waved through.
    for kind, exact in (('TRIPRISM', 3.0 * _SQRT3),
                        ('HEXPRISM', 1.5 * _SQRT3 * 3.0)):
        for m in (12, 24, 48):
            for rg in (3, 6, 10):
                V, F, B, vol = build(kind, resolution=m, rings=rg)
                num = mesh_volume(V, F)
                assert abs(num - exact) < 1e-9 * exact, (kind, m, rg, num)
                assert abs(vol - exact) < 1e-9, (kind, vol, exact)
        print(f"{kind}: volume exactly {exact:.6f} at every "
              f"resolution x ring count  OK")

    # the lattice must account for exactly the cell volume (times the number
    # of cells per translational unit).  This is the check that would catch a
    # basis written down from a half-remembered picture rather than derived.
    for kind in KINDS:
        V, F, B, vol = build(kind, resolution=24, rings=6)
        det = abs(float(np.linalg.det(B)))
        want = vol * _cells_per_lattice_cell[kind]
        assert abs(det - want) < 1e-9 * want, (kind, det, want)
        print(f"{kind}: lattice det {det:.5f} = {_cells_per_lattice_cell[kind]}"
              f" x cell volume  OK")

    # every analytic cell must be closed AND consistently oriented, at a
    # spread of resolutions
    for kind in KINDS:
        for m in (8, 12, 24):
            V, F, B, vol = build(kind, resolution=m, rings=4)
            ne = check_closed(V, F)
        print(f"{kind}: closed and consistently oriented, {ne} edges  OK")

    print("softcell.analytic standalone tests passed")
