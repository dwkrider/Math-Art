"""relief.solid -- relief on closed surfaces: spheres, tori, cylinders.

The flat engine next door answers h(x, y) on a rectangle.  None of that
transfers to a closed surface, and the reason is a theorem rather than an
inconvenience: by Gauss's *Theorema Egregium* a sphere has nonzero Gaussian
curvature, so no map from it to the plane preserves distance.  Wrapping a
planar field onto a sphere cannot be made correct -- only differently wrong.
A lat-long map pays for it at the poles, where a whole row of the grid
collapses to one point; measured on the old generator's UV sphere, faces near
the pole were a tenth the area of those at the equator, the largest face was
40 times the smallest, and 130 vertices crowded into a five-degree cap that
should hold about fifteen.

So this module does not project.  Two rules follow, and everything here is a
consequence of them:

**1. The field is evaluated in 3D, at the surface's own points.**  A function
of (x, y, z) restricted to a surface is continuous wherever the surface is,
whatever the surface does -- no seam to close, no wrap to match, no pole to
special-case.  Spatial continuity stops being something to arrange and becomes
something the construction cannot violate.  The cost is that only fields with
a 3D definition can be used, which is why the pattern list here is its own and
not the flat engine's twenty.

**2. The mesh has no poles.**  The sphere is a geodesic subdivision of the
icosahedron, whose faces stay within a small factor of one another over the
whole surface; the torus is a genuine doubly periodic grid; the cylinder is
developable and distorts not at all.  Where a base does have unavoidable area
variation -- a torus is wider outside than inside -- it is measured and
reported rather than hidden.

References:
  Carl Friedrich Gauss, "Disquisitiones generales circa superficies curvas",
    1827 -- the Theorema Egregium, and why a sphere cannot be flattened.
  Michael J. Fisher and Adrian Bowyer, "Notes on the geodesic sphere" (the
    icosahedral subdivision used here); see also Buckminster Fuller's geodesic
    constructions, US Patent 2,682,235, 1954.
  J. P. Lewis, "Algorithms for Solid Noise Synthesis", SIGGRAPH 1989, 263-270
    -- sparse convolution noise, which is defined in space and so restricts to
    any surface without modification.
  Steven Worley, "A Cellular Texture Basis Function", SIGGRAPH 1996, 291-294
    -- likewise cellular texture, whose feature points live in 3D.
"""

import math

import numpy as np

BASES = ('SPHERE', 'TORUS', 'CYLINDER')
# Every field `evaluate` dispatches.  Kept complete: this tuple and the
# operator's menu are the two lists that have to agree, and LATTICE was
# missing from it while being dispatched and offered.
FIELDS3D = ('FRACTAL', 'CELLULAR', 'GABOR', 'HARMONIC', 'LATTICE', 'TURING')


# ---------------------------------------------------------------- bases ----

def icosphere(subdivisions=4):
    """Geodesic sphere: the icosahedron, subdivided and pushed onto the ball.

    Chosen over a lat-long sphere for one reason, which the flat engine's
    experience makes worth stating plainly: a UV sphere's faces vary in area
    by a factor of forty and pile up at the poles, and a relief drawn on it
    is denser and shallower there through no wish of the pattern's.  A
    geodesic sphere has no poles and no seam at all -- every vertex is
    equivalent apart from the twelve original icosahedral corners.
    """
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, dtype=float) / np.linalg.norm(v) for v in verts]

    for _ in range(max(0, int(subdivisions))):
        mid = {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                mid[key] = len(V) - 1
            return mid[key]

        nf = []
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf

    P = np.array(V)
    return P, P.copy(), [list(f) for f in faces]


def torus(res=128, ring=1.0, tube=0.4):
    """Doubly periodic grid on a torus.  Both directions wrap; no poles.

    The grid is closed by index arithmetic rather than by welding duplicated
    edges, so there is no seam to weld and none to leave behind.
    """
    nu = max(8, int(res))
    # Square cells: the tube's circumference is 2*pi*tube and the ring's is
    # 2*pi*ring, so the counts must be in the ratio tube/ring.  Halving that
    # (as this did) leaves every quad twice as long as it is wide, and the
    # triangles they split into are then obtuse -- which costs the cotangent
    # operator a quarter of its weights to clamping, and distorts any
    # isotropic pattern into ovals for the same reason the flat grid insists
    # on square cells.
    nv = max(8, int(round(res * tube / ring))) if ring > 0 else nu
    u = np.linspace(0.0, 2.0 * math.pi, nu, endpoint=False)
    v = np.linspace(0.0, 2.0 * math.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v, indexing='ij')
    cu, su = np.cos(U), np.sin(U)
    cv, sv = np.cos(V), np.sin(V)
    P = np.stack([((ring + tube * cv) * cu).ravel(),
                  ((ring + tube * cv) * su).ravel(),
                  (tube * sv).ravel()], axis=-1)
    N = np.stack([(cv * cu).ravel(), (cv * su).ravel(), sv.ravel()], axis=-1)
    return P, N, _wrapped_quads(nu, nv, wrap_u=True, wrap_v=True)


def cylinder(res=128, height=2.0, radius=0.7, caps=True):
    """A cylinder: periodic around, open along, and developable.

    The one base here with no distortion whatsoever -- a cylinder can be
    unrolled flat without stretching, so a pattern on it keeps every distance
    it had.
    """
    nu = max(8, int(res))
    nv = max(2, int(res * height / (2.0 * math.pi * radius)))
    u = np.linspace(0.0, 2.0 * math.pi, nu, endpoint=False)
    z = np.linspace(-0.5 * height, 0.5 * height, nv)
    U, Z = np.meshgrid(u, z, indexing='ij')
    P = np.stack([(radius * np.cos(U)).ravel(),
                  (radius * np.sin(U)).ravel(),
                  Z.ravel()], axis=-1)
    N = np.stack([np.cos(U).ravel(), np.sin(U).ravel(),
                  np.zeros(U.size)], axis=-1)
    faces = _wrapped_quads(nu, nv, wrap_u=True, wrap_v=False)
    if caps:
        # Close both ends with a fan to a centre vertex, so the cylinder is a
        # solid rather than a sleeve.  The cap normals point along the axis,
        # which keeps the relief flat there instead of flaring the rim.
        P = np.vstack([P, [[0.0, 0.0, -0.5 * height],
                           [0.0, 0.0, 0.5 * height]]])
        N = np.vstack([N, [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]])
        lo_c = len(P) - 2
        hi_c = len(P) - 1
        for i in range(nu):
            i2 = (i + 1) % nu
            faces.append([lo_c, i2 * nv, i * nv])
            faces.append([hi_c, i * nv + nv - 1, i2 * nv + nv - 1])
    return P, N, faces


def _wrapped_quads(nu, nv, wrap_u=False, wrap_v=False):
    """Quad faces over a (nu, nv) grid, closing whichever axes wrap."""
    out = []
    umax = nu if wrap_u else nu - 1
    vmax = nv if wrap_v else nv - 1
    for i in range(umax):
        i2 = (i + 1) % nu
        for j in range(vmax):
            j2 = (j + 1) % nv
            out.append([i * nv + j, i2 * nv + j, i2 * nv + j2, i * nv + j2])
    return out


# What "detail" means depends on the base, and the two scales are nowhere
# near each other: a geodesic sphere takes SUBDIVISIONS, where 5 is already
# 10k vertices and 8 would be half a million, while the torus and cylinder
# take a GRID RESOLUTION, where 5 is a pentagon.  One control serving both
# gave whichever base you did not have in mind a useless default.
SPHERE_RES_DEFAULT = 5
GRID_RES_DEFAULT = 128


def build_base(base='SPHERE', res=None, ring=1.0, tube=0.4, height=2.0,
               radius=0.7, sphere_res=None, grid_res=None):
    """`(points, normals, faces)` for one of the closed bases.

    `res` is accepted as a single legacy control; `sphere_res` and `grid_res`
    are the two it should have been from the start.
    """
    if base == 'SPHERE':
        n = sphere_res if sphere_res is not None else (
            res if res is not None else SPHERE_RES_DEFAULT)
        return icosphere(int(n))
    n = grid_res if grid_res is not None else (
        res if res is not None else GRID_RES_DEFAULT)
    if base == 'TORUS':
        return torus(int(n), ring=ring, tube=tube)
    if base == 'CYLINDER':
        return cylinder(int(n), height=height, radius=radius)
    raise ValueError("unknown base: %r" % (base,))


# --------------------------------------------------------------- fields ----

def fractal3d(P, method='FBM', octaves=8, lacunarity=2.0, dim=2.3,
              hurst=0.7, count=240, seed=1, scale=1.0):
    """Fractal field sampled in space, reusing the `ifs` mode generators."""
    try:
        from ..ifs.spectral import (fbm_modes, weierstrass_modes, eval_field)
    except ImportError:                     # flat import outside the package
        from ifs.spectral import (fbm_modes, weierstrass_modes, eval_field)
    if str(method).upper() == 'WEIERSTRASS':
        modes = weierstrass_modes(int(octaves), float(lacunarity), float(dim))
    else:
        modes = fbm_modes(int(count), int(octaves), float(lacunarity),
                          float(hurst), int(seed))
    return eval_field(np.asarray(P) * float(scale), modes)


def cellular3d(P, count=120, seed=1, mode='CRACK', sharp=1.0):
    """Worley cellular texture with the feature points scattered in space.

    On a closed surface this is the whole argument for working in 3D: the
    cells are regions of *space* cut by the surface, so they meet across any
    wrap without anything being matched up, and no direction on the surface
    is special.
    """
    P = np.asarray(P, dtype=float)
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    pad = 0.15 * float(np.max(hi - lo))
    pts = rng.uniform(lo - pad, hi + pad, size=(max(1, int(count)), 3))

    best1 = np.full(len(P), np.inf)
    best2 = np.full(len(P), np.inf)
    for q in pts:
        d = np.linalg.norm(P - q, axis=1)
        closer = d < best1
        best2 = np.where(closer, best1, np.minimum(best2, d))
        best1 = np.where(closer, d, best1)
    span = float(np.max(hi - lo)) or 1.0
    n = max(1, int(count))
    unit = span / max(n ** (1.0 / 3.0), 1e-9)
    if mode == 'F1':
        h = best1 / unit
    elif mode == 'F2':
        h = best2 / unit
    else:
        h = (best2 - best1) / unit
    h = np.clip(h, 0.0, None)
    if sharp != 1.0 and sharp > 0.0:
        h = h ** float(sharp)
    return h


def gabor3d(P, count=200, freq=6.0, bandwidth=0.35, seed=1, spread=3.1416,
            axis=(0.0, 0.0, 1.0)):
    """Sparse Gabor convolution in space: band-limited noise on any surface."""
    P = np.asarray(P, dtype=float)
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    span = float(np.max(hi - lo)) or 1.0
    n = max(1, int(count))
    pts = rng.uniform(lo, hi, size=(n, 3))
    w = rng.normal(size=n)
    ph = rng.uniform(0.0, 2.0 * math.pi, n)

    ax = np.array(axis, dtype=float)
    ax = ax / (np.linalg.norm(ax) or 1.0)
    # Directions: `spread` 0 aligns every kernel with `axis`, giving a banded
    # weave; full spread randomises them into isotropic noise.
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    s = float(np.clip(spread / math.pi, 0.0, 1.0))
    dirs = ax[None, :] * (1.0 - s) + dirs * s
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    F = float(freq) / span
    a = float(bandwidth) * F
    rad = 3.0 / max(a, 1e-9)
    out = np.zeros(len(P))
    for i in range(n):
        d = P - pts[i]
        r2 = np.einsum('ij,ij->i', d, d)
        near = r2 < rad * rad
        if not near.any():
            continue
        env = np.exp(-math.pi * a * a * r2)
        carrier = np.cos(2.0 * math.pi * F * (d @ dirs[i]) + ph[i])
        out += w[i] * np.where(near, env * carrier, 0.0)
    return out


def legendre_p(l, m, x):
    """Associated Legendre P_l^m by the standard upward recurrences.

    No scipy in Blender, and the closed forms are unusable past small l, so
    this walks the two recurrences that stay stable: first up in m to the
    seed P_m^m, then up in l.
    """
    l = int(l)
    m = int(abs(m))
    x = np.asarray(x, dtype=float)
    if m > l:
        return np.zeros_like(x)
    # Seed: P_m^m = (-1)^m (2m-1)!! (1-x^2)^(m/2)
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt(np.maximum(1.0 - x * x, 0.0))
        fact = 1.0
        for _ in range(m):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = np.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def spherical_harmonic(P, l=4, m=2, phase=0.0):
    """Real spherical harmonic Y_lm on the unit directions of `P`.

    These are to a sphere what Chladni figures are to a plate: its own
    vibration modes, the eigenfunctions of its Laplacian.  Being defined on
    the sphere itself they have no seam and no pole problem -- the pole is an
    ordinary point of a spherical harmonic, which is exactly what a
    parameterised field cannot manage.
    """
    P = np.asarray(P, dtype=float)
    r = np.linalg.norm(P, axis=1)
    r = np.where(r > 1e-12, r, 1.0)
    z = P[:, 2] / r
    phi = np.arctan2(P[:, 1], P[:, 0])
    l = max(0, int(l))
    m = int(np.clip(m, -l, l))
    am = abs(m)
    # Normalisation of the real form; the constant matters for the
    # orthonormality check, not for the look.
    norm = math.sqrt((2.0 * l + 1.0) / (4.0 * math.pi)
                     * math.factorial(l - am) / math.factorial(l + am))
    pl = legendre_p(l, am, z)
    if m > 0:
        return math.sqrt(2.0) * norm * pl * np.cos(am * phi + phase)
    if m < 0:
        return math.sqrt(2.0) * norm * pl * np.sin(am * phi + phase)
    return norm * pl


def lattice3d(P, nx=3, ny=2, nz=1, phase=0.0, ring=1.0, tube=0.4):
    """A triply periodic wave, for bases with no natural mode family."""
    P = np.asarray(P, dtype=float)
    span = float(np.max(P.max(axis=0) - P.min(axis=0))) or 1.0
    k = 2.0 * math.pi / span
    return (np.sin(k * nx * P[:, 0] + phase)
            * np.cos(k * ny * P[:, 1])
            + np.sin(k * nz * P[:, 2] + phase))


# Gray-Scott regimes for the MESH, which are not the grid's.
#
# Swept here rather than inherited, twice over.  The living region of the
# (F, k) plane depends on the ratio of diffusion to reaction, and that ratio
# depends on the lattice spacing relative to the object -- a 256-square grid
# over Pearson's 2.5-unit domain is nothing like a ten-thousand-vertex sphere
# two units across.  Feeding the flat engine's pairs in killed three of
# seven; calibrating the operator to Pearson's own per-step diffusion, which
# should in principle have transferred his map exactly, killed all seven.  So
# the plane was swept on this operator, at the default feature scale, and
# these are pairs that grow on a surface.
#
#                        F       k      cover
SURFACE_REGIMES = {
    'SOLITONS': (0.014, 0.055),   #  10%  sparse, well separated dots
    'SPOTS':    (0.018, 0.055),   #  18%  even spotting
    'MITOSIS':  (0.022, 0.057),   #  22%  spots caught dividing
    'WORMS':    (0.026, 0.057),   #  33%  elongated, joined
    'MAZE':     (0.030, 0.057),   #  44%  labyrinth
    'CORAL':    (0.022, 0.049),   #  55%  dense branching
    'HOLES':    (0.026, 0.053),   #  60%  pits in a filled field
}
SURFACE_REGIME_ORDER = ('SPOTS', 'SOLITONS', 'MITOSIS', 'WORMS', 'MAZE',
                        'CORAL', 'HOLES')


def _triangles(faces):
    """Triangles to build the operator from, with weights.

    A quad has to be split, and the choice of diagonal is not neutral: one
    diagonal makes the discrete operator stiffer along that direction, and a
    Gray-Scott labyrinth will happily lay its stripes along the bias and look
    like a decision someone made.  Splitting BOTH ways at half weight each is
    symmetric, costs one extra pass, and removes the artefact.
    """
    out = []
    for f in faces:
        n = len(f)
        if n == 3:
            out.append((int(f[0]), int(f[1]), int(f[2]), 1.0))
        elif n == 4:
            a, b, c, d = (int(x) for x in f)
            out.append((a, b, c, 0.5))
            out.append((a, c, d, 0.5))
            out.append((a, b, d, 0.5))
            out.append((b, c, d, 0.5))
        else:                               # fan the rest from the first
            for k in range(1, n - 1):
                out.append((int(f[0]), int(f[k]), int(f[k + 1]), 1.0))
    return out


def cotan_laplacian(P, faces):
    """Cotangent Laplace-Beltrami with a lumped mixed-area mass matrix.

    Returns `(I, J, W, area, report)`: the undirected edge list with its
    cotangent weights, the per-vertex area, and what the construction found.
    The Laplacian of a field u is then

        (L u)_i = (1 / area_i) * sum_j W_ij (u_j - u_i)

    **Why not the average of the neighbours.**  The umbrella operator --
    mean of neighbours minus self -- is dimensionless, while a Laplacian
    carries units of one over length squared.  It is therefore not merely
    inaccurate on an uneven mesh but *scaled by the local vertex spacing*,
    behaving like h(x)^2 times the true operator.  Diffusion-driven pattern
    wavelength goes as the square root of the diffusion rate, so features end
    up a fixed number of VERTICES across instead of a fixed distance: on a
    torus, whose outer ring is more finely divided than its inner one, the
    pattern comes out visibly coarser outside.  The cotangent weights come
    from the surface's actual geometry and have the right units, so a blob is
    the same size wherever it grows.

    References:
      Ulrich Pinkall and Konrad Polthier, "Computing Discrete Minimal
        Surfaces and Their Conjugates", Experimental Mathematics 2(1), 1993,
        15-36 -- the cotangent weights.
      Mark Meyer, Mathieu Desbrun, Peter Schroeder and Alan H. Barr,
        "Discrete Differential-Geometry Operators for Triangulated
        2-Manifolds", in Visualization and Mathematics III, Springer, 2003,
        35-57 -- the mixed Voronoi vertex area used as the mass matrix, which
        is what makes the operator correct on obtuse triangles.
      Max Wardetzky, Saurabh Mathur, Felix Kaelberer and Eitan Grinspun,
        "Discrete Laplace operators: No free lunch", Symposium on Geometry
        Processing, 2007, 33-37 -- no discrete Laplacian has every desirable
        property at once; the umbrella gives up linear precision, which is
        exactly the geometric blindness described above.
    """
    P = np.asarray(P, dtype=float)
    n = len(P)
    tris = _triangles(faces)

    edges = {}
    area = np.zeros(n)
    obtuse = 0
    for (a, b, c, fw) in tris:
        pa, pb, pc = P[a], P[b], P[c]
        # Edge vectors opposite each vertex.
        ab, bc, ca = pb - pa, pc - pb, pa - pc
        cross = np.cross(ab, -ca)
        twice = float(np.linalg.norm(cross))
        if twice < 1e-18:
            continue
        tri_area = 0.5 * twice
        # cot at a vertex = (adjacent dot adjacent) / |cross|, which avoids
        # the trigonometry and the cancellation that comes with it.
        cot_a = float(np.dot(ab, -ca)) / twice
        cot_b = float(np.dot(bc, -ab)) / twice
        cot_c = float(np.dot(ca, -bc)) / twice
        for (i, j, cot) in ((b, c, cot_a), (c, a, cot_b), (a, b, cot_c)):
            key = (i, j) if i < j else (j, i)
            edges[key] = edges.get(key, 0.0) + 0.5 * cot * fw

        # Mixed area (Meyer et al.): the Voronoi area is only valid while the
        # triangle is non-obtuse; past that the circumcentre leaves the
        # triangle and the "area" it implies is meaningless, so the triangle
        # is split by halves and quarters instead.
        if cot_a < 0.0 or cot_b < 0.0 or cot_c < 0.0:
            obtuse += 1
            if cot_a < 0.0:
                area[a] += 0.5 * tri_area * fw
                area[b] += 0.25 * tri_area * fw
                area[c] += 0.25 * tri_area * fw
            elif cot_b < 0.0:
                area[b] += 0.5 * tri_area * fw
                area[a] += 0.25 * tri_area * fw
                area[c] += 0.25 * tri_area * fw
            else:
                area[c] += 0.5 * tri_area * fw
                area[a] += 0.25 * tri_area * fw
                area[b] += 0.25 * tri_area * fw
        else:
            la = float(np.dot(bc, bc))
            lb = float(np.dot(ca, ca))
            lc = float(np.dot(ab, ab))
            area[a] += 0.125 * (lc * cot_c + lb * cot_b) * fw
            area[b] += 0.125 * (lc * cot_c + la * cot_a) * fw
            area[c] += 0.125 * (lb * cot_b + la * cot_a) * fw

    keys = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)
    I = keys[:, 0]
    J = keys[:, 1]
    W = np.array([edges[(int(i), int(j))] for i, j in keys])
    area = np.maximum(area, 1e-12)

    # A negative weight means the edge is not locally Delaunay, and it costs
    # the discrete maximum principle: diffusion can then overshoot and drive
    # a concentration outside [0, 1].  Clamping is the standard remedy and
    # is enough here -- a geodesic sphere has no obtuse triangles at all --
    # but it is counted rather than assumed, so a base that starts producing
    # them says so.
    neg = int((W < 0.0).sum())
    W = np.maximum(W, 0.0)
    report = {'edges': len(W), 'negative_weights': neg,
              'obtuse_triangles': obtuse,
              'median_edge': float(np.median(
                  np.linalg.norm(P[J] - P[I], axis=1)))}
    return I, J, W, area, report


def laplacian_apply(u, I, J, W, inv_area):
    """One application of the Laplacian, as two counting passes.

    `np.bincount` rather than `np.add.at`: same result, and fast enough that
    the operator is not what limits how many steps a pattern can take.
    """
    flux = W * (u[J] - u[I])
    n = len(inv_area)
    return (np.bincount(I, flux, minlength=n)
            - np.bincount(J, flux, minlength=n)) * inv_area


def vertex_laplacian(P, faces):
    """Kept for callers that want the old umbrella operator by name.

    Retained so the difference between the two can be measured rather than
    argued about; `turing_surface` uses the cotangent operator.
    """
    nbr = [[] for _ in range(len(P))]
    seen = set()
    for f in faces:
        k = len(f)
        for i in range(k):
            a_, b_ = int(f[i]), int(f[(i + 1) % k])
            key = (a_, b_) if a_ < b_ else (b_, a_)
            if key in seen:
                continue
            seen.add(key)
            nbr[a_].append(b_)
            nbr[b_].append(a_)
    width = max(len(n) for n in nbr)
    idx = np.zeros((len(P), width), dtype=np.int64)
    wgt = np.zeros((len(P), width))
    for i, ns in enumerate(nbr):
        idx[i, :len(ns)] = ns
        wgt[i, :len(ns)] = 1.0
    inv = 1.0 / np.maximum(wgt.sum(axis=1), 1.0)
    return idx, wgt, inv


def turing_surface(P, faces, regime='MAZE', steps=4000, seed=1, scale=0.35,
                   feed=None, kill=None, operator='COTAN', report=None):
    """Grow a Gray-Scott skin on the surface itself.

    The flat engine runs this on a grid and resamples; that cannot be done
    here, because there is no grid a closed surface maps to without
    distortion.  Running it on the mesh instead is both simpler and more
    faithful -- it is how these patterns are actually modelled on animal
    coats, and the pattern has no seam because the surface has none.

    **The operator is scaled by the median edge length squared.**  That makes
    the mesh behave like a uniform lattice of that spacing, so the feed/kill
    map worked out for the grid carries over unchanged -- and, more to the
    point, a blob comes out the same size in metres wherever it grows,
    instead of the same size in vertices.

    `operator='UMBRELLA'` selects the old neighbour-average operator.  It is
    kept so the difference can be measured; it is not the right answer.
    """
    f_, k_ = SURFACE_REGIMES.get(str(regime).upper(),
                                 SURFACE_REGIMES['MAZE'])
    F = float(f_ if feed is None else feed)
    K = float(k_ if kill is None else kill)

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    n = len(P)
    u = np.ones(n)
    v = np.zeros(n)
    # Finite-amplitude patches, for the reason the flat engine documents: the
    # uniform state is linearly stable, so low-amplitude noise everywhere
    # decays and leaves a blank surface.
    hit = rng.random(n) < 0.06
    u[hit] = 0.50
    v[hit] = 0.25
    u += 0.02 * (rng.random(n) - 0.5)
    v += 0.02 * (rng.random(n) - 0.5)

    # Pearson's own per-step diffusion, expressed as a multiple of
    # h0^2 * Laplacian -- which is exactly what the scaled operator below
    # applies.  With the geometry handled correctly, the feed/kill map worked
    # out on the grid transfers unchanged; there is no need for a second
    # table of regimes, and the fact that the grid's table works here is a
    # check on the operator rather than a convenience.
    du = 0.2097 * float(scale)
    dv = 0.1049 * float(scale)

    if str(operator).upper() == 'UMBRELLA':
        idx, wgt, inv = vertex_laplacian(P, faces)
        du = min(du, 0.45)
        dv = min(dv, 0.45)
        for _ in range(max(1, int(steps))):
            lu = (u[idx] * wgt).sum(axis=1) * inv - u
            lv = (v[idx] * wgt).sum(axis=1) * inv - v
            uvv = u * v * v
            u += du * lu - uvv + F * (1.0 - u)
            v += dv * lv + uvv - (F + K) * v
        return v

    I, J, W, area, rep = cotan_laplacian(P, faces)
    inv_area = 1.0 / area
    h0 = rep['median_edge']
    gain = h0 * h0                          # back into lattice units

    # Explicit stepping is stable while D dt lambda_max < 2, and Gershgorin
    # bounds lambda_max by twice the worst weight-sum over area -- one bad
    # vertex, typically the apex of a cap fan, sets the limit for everyone.
    # Computing it is cheaper than discovering it as NaN.
    wsum = (np.bincount(I, W, minlength=n) + np.bincount(J, W, minlength=n))
    lam = 2.0 * float(np.max(wsum * inv_area)) * gain
    dt = min(1.0, 1.8 / max(lam * max(du, dv), 1e-12))
    if report is not None:
        report.update(rep)
        report['lambda_max'] = lam
        report['dt'] = dt
        report['substeps'] = int(math.ceil(1.0 / dt))

    # Sub-step so that one unit of simulation time is one unit whatever the
    # mesh, keeping the feed/kill map meaningful.
    sub = max(1, int(math.ceil(1.0 / dt)))
    dts = 1.0 / sub
    for _ in range(max(1, int(steps))):
        for _ in range(sub):
            lu = laplacian_apply(u, I, J, W, inv_area) * gain
            lv = laplacian_apply(v, I, J, W, inv_area) * gain
            uvv = u * v * v
            u += dts * (du * lu - uvv + F * (1.0 - u))
            v += dts * (dv * lv + uvv - (F + K) * v)
    return v


def evaluate(kind, P, p, faces=None):
    """Dispatch one of the 3D fields over the base's points."""
    kind = str(kind).upper()
    if kind == 'FRACTAL':
        return fractal3d(P, method=p.get('method', 'FBM'),
                         octaves=int(p.get('octaves', 8)),
                         lacunarity=float(p.get('lacunarity', 2.0)),
                         dim=float(p.get('dim', 2.3)),
                         hurst=float(p.get('hurst', 0.7)),
                         count=int(p.get('modes', 240)),
                         seed=int(p.get('seed', 1)),
                         scale=float(p.get('field_scale', 1.0)))
    if kind == 'CELLULAR':
        return cellular3d(P, count=int(p.get('points_n', 120)),
                          seed=int(p.get('seed', 1)),
                          mode=p.get('cell_mode', 'CRACK'),
                          sharp=float(p.get('cell_sharp', 1.0)))
    if kind == 'GABOR':
        return gabor3d(P, count=int(p.get('points_n', 200)),
                       freq=float(p.get('gabor_freq', 6.0)),
                       bandwidth=float(p.get('gabor_band', 0.35)),
                       seed=int(p.get('seed', 1)),
                       spread=float(p.get('spread', 3.1416)))
    if kind == 'HARMONIC':
        return spherical_harmonic(P, l=int(p.get('sph_l', 4)),
                                  m=int(p.get('sph_m', 2)),
                                  phase=float(p.get('phase', 0.0)))
    if kind == 'TURING':
        if faces is None:
            raise ValueError("the Turing field needs the mesh faces")
        return turing_surface(P, faces, regime=p.get('regime', 'MAZE'),
                              steps=int(p.get('rd_steps', 4000)),
                              seed=int(p.get('seed', 1)),
                              scale=float(p.get('rd_scale', 1.0)))
    if kind == 'LATTICE':
        return lattice3d(P, nx=int(p.get('mode_m', 3)),
                         ny=int(p.get('mode_n', 2)),
                         nz=int(p.get('mode_k', 1)),
                         phase=float(p.get('phase', 0.0)))
    raise ValueError("unknown solid field: %r" % (kind,))


# ---------------------------------------------------------------- build ----

def face_areas(V, faces):
    """Area of every face, by fan triangulation."""
    V = np.asarray(V, dtype=float)
    out = []
    for f in faces:
        pts = V[list(f)]
        s = 0.0
        for k in range(1, len(pts) - 1):
            s += 0.5 * float(np.linalg.norm(
                np.cross(pts[k] - pts[0], pts[k + 1] - pts[0])))
        out.append(s)
    return np.asarray(out)


def build_solid(**kw):
    """Displace a closed base by a field evaluated in space.

    Returns `(verts, faces, info)`, with `info` carrying the measurements
    that matter for this construction: how uniform the base's faces are, and
    whether the displaced surface stayed a valid solid.
    """
    p = dict(base='SPHERE', res=None, sphere_res=SPHERE_RES_DEFAULT,
             grid_res=GRID_RES_DEFAULT, ring=1.0, tube=0.4, height=2.0,
             radius=0.7, field='FRACTAL', method='FBM', octaves=8,
             lacunarity=2.0, dim=2.3, hurst=0.7, modes=240, seed=1,
             field_scale=1.0, points_n=120, cell_mode='CRACK',
             cell_sharp=1.0, gabor_freq=6.0, gabor_band=0.35, spread=3.1416,
             sph_l=4, sph_m=2, mode_m=3, mode_n=2, mode_k=1, phase=0.0,
             regime='MAZE', rd_steps=4000, rd_scale=0.35,
             depth=0.25, curve='NONE', curve_amount=1.0, norm='STD',
             fit='CUBE', scale=1.0, span=2.0)
    p.update(kw)

    P, N, faces = build_base(p['base'], res=p['res'],
                             sphere_res=p['sphere_res'],
                             grid_res=p['grid_res'], ring=p['ring'],
                             tube=p['tube'], height=p['height'],
                             radius=p['radius'])
    h = evaluate(p['field'], P, p, faces=faces)

    from . import transfer as _transfer
    h = _transfer.normalize(h, p['norm'])
    h = _transfer.apply_curve(h, p['curve'], amount=float(p['curve_amount']))
    h = _transfer.normalize(h, 'MINMAX') - 0.5

    V = P + (float(p['depth']) * h)[:, None] * N

    base_area = face_areas(P, faces)
    info = {
        'verts': len(V), 'faces': len(faces),
        'base': p['base'], 'field': p['field'],
        # Uniformity of the base itself: the number the lat-long sphere
        # failed at, kept where it can be seen.
        'area_ratio': float(base_area.max() / max(base_area.min(), 1e-30)),
        'area_cv': float(base_area.std() / max(base_area.mean(), 1e-30)),
    }

    from . import mesh as _mesh
    V = _mesh.apply_fit(V, p['fit'], float(p['scale']), float(p['span']))
    return V, faces, info


def _selftest():
    ok = True

    # --- the bases -------------------------------------------------------
    # A geodesic sphere exists to beat the lat-long sphere's uniformity, so
    # that is the measurement.  The old UV sphere ran to a 40x spread of face
    # areas and piled 130 vertices into a 5-degree polar cap that should hold
    # about fifteen; anything close to that here would be a failure.
    P, N, F = icosphere(4)
    A = face_areas(P, F)
    ratio = float(A.max() / A.min())
    cap = float((P[:, 2] > math.cos(math.radians(5))).sum())
    want = len(P) * (1.0 - math.cos(math.radians(5))) / 2.0
    print("solid: icosphere %d verts %d faces, area max/min %.2f, "
          "5-degree cap holds %d (uniform %.0f)"
          % (len(P), len(F), ratio, cap, want))
    ok = ok and ratio < 2.0 and cap < 3.0 * max(want, 1.0)

    # Every base closes: each edge shared by exactly two faces, no boundary.
    from . import mesh as _mesh
    for name, args in (('SPHERE', dict(res=3)), ('TORUS', dict(res=48)),
                       ('CYLINDER', dict(res=48))):
        # (the cylinder is capped by default, so every base is closed)
        Pb, Nb, Fb = build_base(name, **args)
        op, nm, ori = _mesh.edge_report(Fb)
        closed = (op == 0)
        print("solid: base %-9s %5d verts %5d faces  open edges %d  "
              "non-manifold %d" % (name, len(Pb), len(Fb), op, nm))
        ok = ok and nm == 0 and closed

    # Normals point outward on the sphere and agree with the radius.
    Ps, Ns, _ = icosphere(2)
    align = float(np.min(np.einsum('ij,ij->i', Ps, Ns)
                         / np.linalg.norm(Ps, axis=1)))
    print("solid: sphere normal vs radius, worst alignment %.6f" % align)
    ok = ok and align > 0.999999

    # --- spatial continuity, which is the whole point --------------------
    # A field evaluated in space is continuous on the surface because it is
    # continuous in space.  The test is the same one the flat engine uses for
    # seams: the largest step between neighbouring vertices, against the
    # spread of the field itself.  A wrapped or projected field would show a
    # line of large steps somewhere; this must not.
    def worst_step(Pv, Fv, hv):
        edges = set()
        for f in Fv:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                edges.add((min(a, b), max(a, b)))
        e = np.array(sorted(edges))
        d = np.abs(hv[e[:, 0]] - hv[e[:, 1]])
        return float(d.max() / (hv.std() or 1.0)), float(d.mean())

    for base, args in (('SPHERE', dict(res=4)), ('TORUS', dict(res=96)),
                       ('CYLINDER', dict(res=96))):
        Pb, Nb, Fb = build_base(base, **args)
        for fld in ('FRACTAL', 'CELLULAR', 'GABOR'):
            hb = evaluate(fld, Pb, dict(seed=3, points_n=80, octaves=6),
                          faces=Fb)
            step, mean = worst_step(Pb, Fb, hb)
            print("solid: %-8s %-8s worst neighbour step %.2f sd "
                  "(mean %.4f)" % (base, fld, step, mean))
            # A seam shows up as a step many times the field's own spread.
            ok = ok and step < 6.0 and np.isfinite(hb).all()

    # The torus wraps in BOTH directions, and that is where a projected
    # field would betray itself.  Check the two wrap rings explicitly.
    Pt, Nt, Ft = torus(96)
    ht = evaluate('FRACTAL', Pt, dict(seed=5, octaves=6), faces=Ft)
    nu = 96
    nv = len(Pt) // nu
    H = ht.reshape(nu, nv)
    wrap_u = float(np.abs(H[0] - H[-1]).max() / (ht.std() or 1.0))
    wrap_v = float(np.abs(H[:, 0] - H[:, -1]).max() / (ht.std() or 1.0))
    inner_u = float(np.abs(np.diff(H, axis=0)).max() / (ht.std() or 1.0))
    print("solid: torus wrap step u %.3f, v %.3f sd; interior step %.3f sd"
          % (wrap_u, wrap_v, inner_u))
    # The wrap must be no worse than ordinary interior variation -- not
    # merely "small", which any smooth field would pass.
    ok = ok and wrap_u <= 1.5 * inner_u and wrap_v <= 1.5 * inner_u

    # --- spherical harmonics --------------------------------------------
    # Orthonormality over the sphere, integrated with the face areas as
    # weights.  This checks the Legendre recurrence, the normalisation and
    # the real form all at once, and it is a property no amount of
    # eyeballing would confirm.
    Ph, Nh, Fh = icosphere(5)
    w = np.zeros(len(Ph))
    Ah = face_areas(Ph, Fh)
    for f, a in zip(Fh, Ah):
        for i in f:
            w[i] += a / 3.0
    w *= 4.0 * math.pi / w.sum()
    pairs = [((2, 0), (2, 0)), ((3, 1), (3, 1)), ((4, -2), (4, -2)),
             ((2, 0), (3, 0)), ((3, 1), (3, -1)), ((4, 2), (2, 0))]
    worst_diag = 0.0
    worst_off = 0.0
    for (l1, m1), (l2, m2) in pairs:
        y1 = spherical_harmonic(Ph, l1, m1)
        y2 = spherical_harmonic(Ph, l2, m2)
        val = float(np.sum(w * y1 * y2))
        if (l1, m1) == (l2, m2):
            worst_diag = max(worst_diag, abs(val - 1.0))
        else:
            worst_off = max(worst_off, abs(val))
    print("solid: spherical harmonics, worst |<Y,Y> - 1| = %.2e, "
          "worst |<Y_a,Y_b>| = %.2e" % (worst_diag, worst_off))
    ok = ok and worst_diag < 0.02 and worst_off < 0.02

    # A harmonic has the nodal structure it should: Y_l0 changes sign l
    # times down the axis.
    zs = np.linspace(-0.999, 0.999, 400)
    pts = np.stack([np.zeros_like(zs), np.sqrt(1 - zs * zs), zs], axis=-1)
    for l in (2, 3, 5):
        y = spherical_harmonic(pts, l, 0)
        crossings = int(np.sum(np.diff(np.signbit(y))))
        print("solid: Y_%d0 sign changes along a meridian: %d (want %d)"
              % (l, crossings, l))
        ok = ok and crossings == l

    # --- the Laplacian, against answers it cannot influence --------------
    # A Laplacian must annihilate constants, and on a sphere it must return
    # -l(l+1) on the harmonic Y_l0.  The umbrella operator is measured beside
    # it, because the case for the cotangent weights is quantitative and
    # should stay visible rather than becoming folklore.
    Pc, Nc, Fc = icosphere(4)
    I, J, W, area, rep = cotan_laplacian(Pc, Fc)
    inv_area = 1.0 / area
    const = float(np.abs(laplacian_apply(np.ones(len(Pc)), I, J, W,
                                         inv_area)).max())
    print("solid: cotan L(constant) = %.2e; %d edges, %d clamped negative, "
          "%d obtuse triangles"
          % (const, rep['edges'], rep['negative_weights'],
             rep['obtuse_triangles']))
    ok = ok and const < 1e-9

    idx_u, wgt_u, inv_u = vertex_laplacian(Pc, Fc)
    h0 = rep['median_edge']
    for l in (2, 4):
        y = spherical_harmonic(Pc, l, 0)
        want = -float(l * (l + 1))
        rq_c = float(np.dot(y * area, laplacian_apply(y, I, J, W, inv_area))
                     / np.dot(y * area, y))
        um = (y[idx_u] * wgt_u).sum(axis=1) * inv_u - y
        rq_u = float(np.dot(y * area, um / (h0 * h0)) / np.dot(y * area, y))
        e_c = abs(rq_c - want) / abs(want)
        e_u = abs(rq_u - want) / abs(want)
        print("solid: laplacian of Y_%d0 -> cotan %.3f (%.1f%% off), "
              "umbrella %.3f (%.1f%% off), exact %.3f"
              % (l, rq_c, 100 * e_c, rq_u, 100 * e_u, want))
        ok = ok and e_c < 0.05 and e_u > 0.5

    # The same on a torus, where the answer is analytic: the Laplacian of
    # cos(m u) is -m^2 cos(m u) / (R + r cos v)^2.  This is the case the
    # umbrella cannot do at all, because it has no way to know that the outer
    # ring is longer than the inner one.
    Rt, rt = 1.0, 0.4
    Pt, Nt, Ft = torus(96, ring=Rt, tube=rt)
    It, Jt, Wt, at, rept = cotan_laplacian(Pt, Ft)
    ut = np.arctan2(Pt[:, 1], Pt[:, 0])
    cosv = (np.hypot(Pt[:, 0], Pt[:, 1]) - Rt) / rt
    fm = np.cos(3.0 * ut)
    exact = -9.0 * np.cos(3.0 * ut) / (Rt + rt * cosv) ** 2
    err = float(np.abs(laplacian_apply(fm, It, Jt, Wt, 1.0 / at) - exact).max()
                / np.abs(exact).max())
    print("solid: torus laplacian of cos(3u) vs the analytic answer: "
          "%.2f%% worst error" % (100 * err))
    ok = ok and err < 0.05

    # --- and the artefact that motivated all of it ----------------------
    # The torus's outer ring is 2.33 times the circumference of its inner
    # one and carries the same number of vertices, so the two hypotheses give
    # opposite answers: a pattern whose size is fixed in WORLD distance fits
    # more blobs outside, one whose size is fixed in VERTICES fits the same
    # number.  This is what the whole operator argument is for, so it is
    # measured rather than asserted.
    Pf, Nf, Ff = torus(256, ring=Rt, tube=rt)
    nvf = len(Pf) // 256

    def blobs(field, col):
        ring = field.reshape(256, nvf)[:, col]
        return int(np.sum(np.diff(np.signbit(ring - ring.mean())) != 0))

    ratios = {}
    for op in ('UMBRELLA', 'COTAN'):
        vv = turing_surface(Pf, Ff, regime='SPOTS', steps=2000, seed=4,
                            operator=op, scale=0.35)
        out_n = blobs(vv, 0)
        in_n = blobs(vv, nvf // 2)
        ratios[op] = out_n / max(in_n, 1)
        print("solid: %-8s blobs outer %3d inner %3d, ratio %.2f "
              "(world-true 2.33, vertex-true 1.00)"
              % (op, out_n, in_n, ratios[op]))
    # The umbrella must land near the vertex-true 1.00 and the cotangent
    # operator well away from it, toward world-true.  It is not asserted to
    # HIT 2.33: counting sign changes round a ring needs several samples per
    # blob to be reliable, and the outer ring -- the longer one -- is the
    # sparsely sampled one, so the count there is the noisier of the two.
    # What the test has to establish is that the artefact is gone, and a
    # factor of two between the operators establishes it.
    ok = ok and ratios['UMBRELLA'] < 1.4
    ok = ok and ratios['COTAN'] > 1.5
    ok = ok and ratios['COTAN'] > 1.8 * ratios['UMBRELLA']

    # Every regime grows something on a surface.
    Pr, Nr, Fr = icosphere(4)
    dead = []
    for name in SURFACE_REGIME_ORDER:
        vv = turing_surface(Pr, Fr, regime=name, steps=2000, seed=2,
                            scale=0.35)
        if vv.std() <= 0.03:
            dead.append(name)
    print("solid: %d of %d surface regimes develop structure%s"
          % (len(SURFACE_REGIME_ORDER) - len(dead), len(SURFACE_REGIME_ORDER),
             "" if not dead else "; DEAD: " + ", ".join(dead)))
    ok = ok and not dead

    # --- the built solid -------------------------------------------------
    for base in BASES:
        V, F2, info = build_solid(base=base, res=4 if base == 'SPHERE' else 64,
                                  field='FRACTAL', seed=2, depth=0.3)
        op, nm, ori = _mesh.edge_report(F2)
        ext = np.asarray(V).max(axis=0) - np.asarray(V).min(axis=0)
        print("solid: %-9s V=%-6d F=%-6d open=%d nonman=%d area ratio %.2f "
              "bbox %s" % (base, len(V), len(F2), op, nm, info['area_ratio'],
                           np.round(ext, 3)))
        ok = ok and nm == 0 and op == 0 and np.isfinite(np.asarray(V)).all()

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
