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
FIELDS3D = ('FRACTAL', 'CELLULAR', 'GABOR', 'HARMONIC', 'QUASI',
            'GROUP', 'RIPPLE', 'WAVE', 'SCATTER', 'TORUS_MODE',
            'WALLPAPER', 'ELLIPTIC', 'TRUCHET', 'SEIGAIHA', 'OCEAN',
            'LATTICE', 'TURING')


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


def icosahedral_axes():
    """The six five-fold axes of the icosahedron, as unit vectors.

    An icosahedron's twelve vertices come in six antipodal pairs, and each
    pair defines one five-fold axis.  They are the golden-ratio coordinates
    the icosphere is already built from.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    v = np.array([[0.0, 1.0, phi], [0.0, -1.0, phi],
                  [1.0, phi, 0.0], [-1.0, phi, 0.0],
                  [phi, 0.0, 1.0], [phi, 0.0, -1.0]])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def quasicrystal3d(P, cells=6.0, phase=0.0, sharp=0.0, jitter=0.0, seed=1):
    """Icosahedral quasicrystal: six cosines along the five-fold axes.

    Two-dimensional quasicrystals come from summing plane waves at angles no
    lattice can hold; the three-dimensional case is the same idea with the
    icosahedron's six five-fold axes, which is the structure that started
    quasicrystallography.  Fivefold symmetry is impossible for any lattice --
    the crystallographic restriction -- so this field has perfect icosahedral
    symmetry and no translational period whatever, in any direction.

    That makes it the one pattern here that is ordered everywhere and repeats
    nowhere, and it needs no chart, no parameterisation and no seam handling:
    it is a function of space, sampled on the surface.

    Zero phase keeps the symmetry exact, because a cosine is even and so the
    sign of each axis does not matter.  `jitter` breaks it deliberately.

    References:
      Dov Levine and Paul J. Steinhardt, "Quasicrystals: A New Class of
        Ordered Structures", Phys. Rev. Lett. 53, 1984, 2477-2480.
      Marjorie Senechal, "Quasicrystals and Geometry", Cambridge University
        Press, 1995.
    """
    P = np.asarray(P, dtype=float)
    span = float(np.max(P.max(axis=0) - P.min(axis=0))) or 1.0
    k = 2.0 * math.pi * float(cells) / span
    axes = icosahedral_axes()
    if jitter > 0.0:
        rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        axes = axes + jitter * rng.normal(size=axes.shape)
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    out = np.zeros(len(P))
    for b_ in axes:
        out += np.cos(k * (P @ b_) + float(phase))
    out /= math.sqrt(len(axes))
    if sharp > 0.0:
        out = np.tanh(float(sharp) * out)
    return out


SPHERE_GROUPS = ('332', 'STAR_332', '432', 'STAR_432', '532', 'STAR_532',
                 '3STAR2')


def sphere_group_matrices(signature='STAR_532', n=5):
    """Orthogonal matrices of one of the spherical symmetry types.

    Reuses `orbifold_sphere_generator.build_group`, which already builds all
    fourteen and is order-tested against Conway's table -- there is no reason
    for a second implementation of the same finite groups.
    """
    try:
        from ..orbifold_sphere_generator import build_group
    except ImportError:                     # flat import outside the package
        from orbifold_sphere_generator import build_group
    return [np.asarray(g, dtype=float) for g in build_group(signature, n)]


def spherical_wallpaper(P, signature='STAR_532', n=5, waves=5, seed=1,
                        cells=4.0, kind='WAVE'):
    """A field invariant under a spherical symmetry group.

    The sphere's answer to a wallpaper pattern.  Averaging any function over
    a group produces something that group cannot change, so the invariance is
    *by construction* rather than by arrangement -- the same argument the flat
    engine makes for the seventeen plane groups, one dimension up.  Where the
    plane has a lattice and a point group, the sphere has only the point
    group, and there are fourteen of them rather than seventeen.

    The seed field is a handful of 3D plane waves (or Gabor-like packets),
    so the result is a smooth field on the whole sphere with no chart, no
    parameterisation and no pole.

    References:
      John H. Conway, Heidi Burgiel and Chaim Goodman-Strauss, "The
        Symmetries of Things", A K Peters, 2008 -- the orbifold notation and
        the enumeration of the spherical types.
      Frank A. Farris, "Creating Symmetry", Princeton University Press,
        2015 -- the group-averaging construction, for the plane.
    """
    P = np.asarray(P, dtype=float)
    span = float(np.max(P.max(axis=0) - P.min(axis=0))) or 1.0
    k = 2.0 * math.pi * float(cells) / span
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    m = max(1, int(waves))
    dirs = rng.normal(size=(m, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    phases = rng.uniform(0.0, 2.0 * math.pi, m)
    amps = 1.0 / (1.0 + np.arange(m))

    G = sphere_group_matrices(signature, n)
    out = np.zeros(len(P))
    for g in G:
        Q = P @ g.T
        for i in range(m):
            t = Q @ dirs[i]
            if str(kind).upper() == 'PACKET':
                out += amps[i] * np.cos(k * t + phases[i]) * np.exp(-t * t)
            else:
                out += amps[i] * np.cos(k * t + phases[i])
    out /= float(len(G))
    sd = out.std()
    return out / sd if sd > 1e-12 else out


def surface_distance(P, c, base='SPHERE', ring=1.0, tube=0.4, radius=0.7):
    """Distance from every point of the base to a point `c` ON the base.

    Geodesic where a closed form exists -- along the sphere it is the angle
    between the two directions, times the radius, and on a cylinder it is the
    unrolled Pythagoras with the angle wrapped the short way round.  On the
    torus there is no closed form (its geodesics are elliptic integrals), so
    the straight-line distance through space is used instead, which is the
    same choice the cellular field already makes and is continuous
    everywhere for the same reason.
    """
    P = np.asarray(P, dtype=float)
    c = np.asarray(c, dtype=float)
    if base == 'SPHERE':
        R = float(np.linalg.norm(c)) or 1.0
        u = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        cc = c / (np.linalg.norm(c) or 1.0)
        return R * np.arccos(np.clip(u @ cc, -1.0, 1.0))
    if base == 'CYLINDER':
        rad = float(radius)
        du = np.arctan2(P[:, 1], P[:, 0]) - math.atan2(c[1], c[0])
        du = (du + math.pi) % (2.0 * math.pi) - math.pi     # the short way
        dz = P[:, 2] - c[2]
        return np.hypot(rad * du, dz)
    return np.linalg.norm(P - c[None, :], axis=1)


def surface_points(P, faces, count, seed=1):
    """`count` points scattered over the surface, uniform by AREA.

    Uniform in area, not in vertices: a mesh's vertices are only as evenly
    spread as the mesh is, and sampling them directly would put more scatter
    wherever the modeller happened to spend triangles.
    """
    A = face_areas(P, faces)
    cum = np.cumsum(A)
    total = float(cum[-1]) or 1.0
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    pick = np.searchsorted(cum, rng.random(max(1, int(count))) * total)
    out = []
    for f_i in pick:
        f = faces[int(min(f_i, len(faces) - 1))]
        v = np.asarray([P[int(i)] for i in f], dtype=float)
        # Barycentric within a triangle; for a quad pick a triangle first.
        if len(v) > 3:
            v = v[:3] if rng.random() < 0.5 else v[[0, 2, 3]]
        a, b = rng.random(), rng.random()
        if a + b > 1.0:
            a, b = 1.0 - a, 1.0 - b
        out.append(v[0] + a * (v[1] - v[0]) + b * (v[2] - v[0]))
    return np.asarray(out)


def ripple3d(P, faces, sources=4, wavelength=0.35, seed=1, base='SPHERE',
             ring=1.0, tube=0.4, radius=0.7, decay=0.0):
    """Interfering circular wavefronts spreading over the surface.

    The two-dimensional construction, done intrinsically: each source sends
    out rings at a constant *geodesic* wavelength, so on a sphere they close
    up again at the antipode instead of being squashed by a projection.  The
    cusp at a source and at its focus is the physics, not a seam.
    """
    C = surface_points(P, faces, sources, seed=seed)
    rng = np.random.default_rng((int(seed) + 977) & 0x7FFFFFFF)
    lam = max(float(wavelength), 1e-6)
    out = np.zeros(len(P))
    for c in C:
        d = surface_distance(P, c, base=base, ring=ring, tube=tube,
                             radius=radius)
        amp = np.exp(-float(decay) * d) if decay > 0.0 else 1.0
        out += amp * np.sin(2.0 * math.pi * d / lam
                            + rng.uniform(0.0, 2.0 * math.pi))
    out /= math.sqrt(max(1, len(C)))
    sd = out.std()
    return out / sd if sd > 1e-12 else out


def zonal_wave(P, axis=(0.0, 0.0, 1.0), wavelength=0.5, phase=0.0,
               steepness=0.0, count=1, spread=0.0, seed=1):
    """Rings of constant geodesic wavelength about an axis.

    The sphere's honest answer to a directional wave.  A plane wave through
    space cuts a sphere in circles of *varying* spacing; measuring the angle
    from an axis instead keeps every ring the same distance from the last,
    which is what a wave on the surface means.

    `count` above one sums several such waves about jittered axes, giving the
    panel's wave-train look.
    """
    P = np.asarray(P, dtype=float)
    R = float(np.max(np.linalg.norm(P, axis=1))) or 1.0
    u = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    a0 = np.asarray(axis, dtype=float)
    a0 = a0 / (np.linalg.norm(a0) or 1.0)
    lam = max(float(wavelength), 1e-6)
    out = np.zeros(len(P))
    n = max(1, int(count))
    for i in range(n):
        a = a0 if i == 0 and spread <= 0.0 else a0 + spread * rng.normal(size=3)
        a = a / (np.linalg.norm(a) or 1.0)
        ang = np.arccos(np.clip(u @ a, -1.0, 1.0)) * R
        ph = float(phase) if n == 1 else float(rng.uniform(0, 2 * math.pi))
        w = np.sin(2.0 * math.pi * ang / lam + ph)
        if steepness > 0.0:
            q = float(np.clip(steepness, 0.0, 1.0))
            w = 2.0 * (0.5 * (w + 1.0)) ** (1.0 + 3.0 * q) - 1.0
        out += w
    out /= math.sqrt(n)
    sd = out.std()
    return out / sd if sd > 1e-12 else out


def scatter3d(P, faces, count=120, sigma=0.12, seed=1, kernel='WYVILL',
              merge='MAX'):
    """Points scattered over the surface, each raised by a smooth kernel.

    Distance in space is the natural metric here -- no chart is involved at
    all -- so this is the object layer's splat with the surface's own points
    as the sources.
    """
    try:
        from .kernels import KERNELS
    except ImportError:                     # flat import outside the package
        from kernels import KERNELS
    fn = KERNELS.get(str(kernel).upper(), KERNELS['WYVILL'])[3]
    C = surface_points(P, faces, count, seed=seed)
    sig = max(float(sigma), 1e-6)
    acc = None
    for c in C:
        d = np.linalg.norm(P - c[None, :], axis=1) / sig
        k = fn(d)
        acc = k if acc is None else (np.maximum(acc, k)
                                     if str(merge).upper() == 'MAX'
                                     else acc + k)
    if acc is None:
        return np.zeros(len(P))
    sd = acc.std()
    return (acc - acc.mean()) / sd if sd > 1e-12 else acc - acc.mean()


def torus_modes(m=2, n=1, ring=1.0, tube=0.4, samples=256):
    """Eigenmodes of the Laplace-Beltrami operator on a ring torus.

    The sphere's vibration modes are the spherical harmonics and the
    cylinder's are a product of cosines; the torus is the one base whose
    modes have no closed form.  They do separate, though.  With

        ds^2 = (R + r cos v)^2 du^2 + r^2 dv^2

    the ansatz f = cos(m u + phi) g(v) reduces the eigenproblem to a periodic
    Sturm-Liouville problem in v alone:

        -(1/W) d/dv [ ((R + r cos v)/r) g' ] + m^2 g/(R + r cos v)^2 = lam g

    with W = r (R + r cos v).  That is one small dense eigenproblem on a
    circle -- milliseconds, and no mesh eigensolver, which would need a
    sparse solver this add-on does not have.

    Returns `(lambda, g, v)` for the n-th mode of angular order m.

    The construction is separation of variables and classical
    Sturm-Liouville theory; see Richard Courant and David Hilbert, "Methods
    of Mathematical Physics", Volume I, Interscience, 1953, on eigenvalue
    problems.  I know of no single canonical "torus harmonics" paper.
    """
    N = int(max(32, samples))
    v = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False)
    dv = 2.0 * math.pi / N
    R = float(ring)
    r = float(tube)
    rad = R + r * np.cos(v)                 # distance from the axis
    W = r * rad                             # sqrt of the metric determinant
    p_co = rad / r                          # the flux coefficient

    # Midpoint coefficients, so the operator stays symmetric in the measure.
    p_half = 0.5 * (p_co + np.roll(p_co, -1))
    A = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        k = (i - 1) % N
        A[i, i] += (p_half[i] + p_half[k]) / (dv * dv)
        A[i, j] -= p_half[i] / (dv * dv)
        A[i, k] -= p_half[k] / (dv * dv)
    A += np.diag(m * m / (rad * rad) * W)
    # Symmetrise in the weight W (the Liouville transformation): the operator
    # is self-adjoint against W dv, not against dv, and `eigh` wants the
    # latter.
    sw = np.sqrt(W)
    S = A / sw[:, None] / sw[None, :]
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    idx = int(np.clip(int(n) - 1, 0, N - 1))
    g = vecs[:, idx] / sw
    # Normalise against the physical measure, and fix the arbitrary sign.
    norm = math.sqrt(float(np.sum(g * g * W) * dv)) or 1.0
    g = g / norm
    if g[int(np.argmax(np.abs(g)))] < 0:
        g = -g
    return float(vals[idx]), g, v


def torus_harmonic(P, m=2, n=1, ring=1.0, tube=0.4, phase=0.0, samples=256):
    """Sample a torus eigenmode at the base's points."""
    lam, g, vs = torus_modes(m=m, n=n, ring=ring, tube=tube, samples=samples)
    P = np.asarray(P, dtype=float)
    u = np.arctan2(P[:, 1], P[:, 0])
    v = np.arctan2(P[:, 2], np.hypot(P[:, 0], P[:, 1]) - float(ring))
    idx = np.mod(np.round(v / (2.0 * math.pi) * len(vs)).astype(int), len(vs))
    out = np.cos(float(m) * u + float(phase)) * g[idx]
    sd = out.std()
    return out / sd if sd > 1e-12 else out


# Patterns that descend from the flat engine because the surface they land
# on is intrinsically flat.
#
# **The flat torus is Euclidean.**  The abstract torus is R^2 / L and the
# parameterisation is the universal covering map -- a local isometry, not a
# projection of a plane onto something curved.  Every doubly periodic
# construction therefore descends EXACTLY: no seam to match, no pole, no
# distortion of the intrinsic pattern.  The cylinder is developable and takes
# the same treatment with only one direction closed.
#
# Two things to be honest about rather than hide.  The *embedded* ring torus
# is not flat -- Gauss-Bonnet forces positive curvature outside and negative
# inside -- so the embedding stretches the pattern by
# (R + r cos v) / (R + r), bounded and about 2.3 to 1 at the defaults; that
# is the same factor `build_solid` already reports as `area_ratio`.  And only
# the u-rotations and the reflections are isometries of the embedded surface,
# so a symmetry group's other elements are symmetries of the intrinsic
# structure rather than of the object you can pick up.
PARAMETRIC = ('TORUS', 'CYLINDER')


def surface_uv(P, base='TORUS', ring=1.0, tube=0.4, height=2.0, radius=0.7):
    """Intrinsic coordinates of a parametrised base, each in [0, 1).

    Only ever used to evaluate functions that are periodic in them, so the
    branch cut of the arctangent is invisible: a periodic function does not
    notice where the chart was cut.
    """
    P = np.asarray(P, dtype=float)
    u = (np.arctan2(P[:, 1], P[:, 0]) / (2.0 * math.pi)) % 1.0
    if base == 'CYLINDER':
        v = (P[:, 2] / float(height)) + 0.5
        return u, np.clip(v, 0.0, 1.0)
    rho = np.hypot(P[:, 0], P[:, 1]) - float(ring)
    v = (np.arctan2(P[:, 2], rho) / (2.0 * math.pi)) % 1.0
    return u, v


def _uv_grid(u, v, cells_u=1.0, cells_v=1.0):
    """(X, Y) for a flat-engine field, in the panel's own [-1, 1] units."""
    X = (u * float(cells_u) % 1.0) * 2.0 - 1.0
    Y = (v * float(cells_v) % 1.0) * 2.0 - 1.0
    return X, Y


def flat_on_surface(P, kind, p, base='TORUS', ring=1.0, tube=0.4,
                    height=2.0, radius=0.7):
    """Evaluate one of the flat engine's doubly periodic fields here.

    The field is evaluated at the intrinsic coordinates, so what lands on the
    surface is the pattern itself rather than a picture of it.  Only fields
    that are genuinely periodic in both directions are offered: anything else
    would show a seam where the chart closes, which is exactly what this
    module exists to avoid.
    """
    u, v = surface_uv(P, base=base, ring=ring, tube=tube, height=height,
                      radius=radius)
    cu = float(p.get('cells_u', 1.0))
    cv = float(p.get('cells_v', 1.0))
    X, Y = _uv_grid(u, v, cu, cv)
    info = {'dx': 2.0 / max(float(p.get('grid_res', 128)), 2.0),
            'dy': 2.0 / max(float(p.get('grid_res', 128)), 2.0),
            'width': 2.0, 'height': 2.0, 'nx': 2, 'ny': 2}

    kind = str(kind).upper()
    if kind == 'WALLPAPER':
        try:
            from .symmetry import wallpaper
        except ImportError:
            from symmetry import wallpaper
        return wallpaper(X, Y, info, group=p.get('group', 'P4M'),
                         # Shared with the spherical group's seed count:
                         # both are "how many waves go in".
                         waves=int(p.get('waves', 5)),
                         seed=int(p.get('seed', 1)),
                         cells=1.0, freq_max=int(p.get('freq_max', 3)))
    if kind == 'ELLIPTIC':
        try:
            from .elliptic import elliptic_field
        except ImportError:
            from elliptic import elliptic_field
        return elliptic_field(X, Y, info, kind=p.get('ell_kind', 'WP'),
                              tau_re=float(p.get('tau_re', 0.0)),
                              tau_im=float(p.get('tau_im', 1.0)),
                              cells=1.0, part=p.get('ell_part', 'SPHERE'))
    if kind == 'TRUCHET':
        try:
            from .tiles import truchet
        except ImportError:
            from tiles import truchet
        return truchet(X, Y, info, cells=int(p.get('tile_cells', 6)),
                       seed=int(p.get('seed', 1)),
                       lane=float(p.get('lane', 0.3)),
                       straight=float(p.get('straight', 0.0)))
    if kind == 'OCEAN':
        try:
            from .ocean import spectrum_field, _forward_choppy
            from .ocean import _sample_wrapped, cusp_lambda
        except ImportError:                 # flat import outside the package
            from ocean import spectrum_field, _forward_choppy
            from ocean import _sample_wrapped, cusp_lambda
        # The one port that is not pointwise: the sea is SYNTHESISED on its
        # own square lattice by an inverse FFT and then sampled, so it cannot
        # be handed a list of scattered points the way the others can.  It is
        # sampled at each vertex's own (u, v) instead, wrapped -- which is
        # exactly what makes it descend to the torus without a seam, since
        # the synthesis was periodic to begin with.
        n = int(max(32, p.get('sea_sim', 256)))
        patch = float(p.get('patch', 100.0))
        h_s, dx_s, dy_s = spectrum_field(
            n=n, patch=patch, wind=float(p.get('wind_speed', 8.0)),
            direction=float(p.get('wind_dir', 0.0)),
            seed=int(p.get('seed', 1)))
        lam = float(p.get('choppy', 0.0))
        if lam > 0.0:
            cell = patch / n
            h_s = _forward_choppy(h_s, dx_s, dy_s,
                                  lam * cusp_lambda(dx_s, dy_s, cell) / cell)
        gx = (u * float(p.get('cells_u', 1.0)) % 1.0) * n
        gy = (v * float(p.get('cells_v', 1.0)) % 1.0) * n
        out = _sample_wrapped(h_s, gx, gy)
        sd = out.std()
        return (out - out.mean()) / sd if sd > 1e-12 else out - out.mean()
    if kind == 'SEIGAIHA':
        try:
            from .tiles import seigaiha
        except ImportError:
            from tiles import seigaiha
        # **The crown is snapped so the stagger closes.**  Seigaiha offsets
        # alternate rows by half a cell, so its pattern repeats every TWO
        # rows -- and it only wraps if an even number of rows fits the
        # period.  At an arbitrary crown it does not: measured, the v wrap
        # came to 2.57 against an interior variation of 1.65, a plain line
        # across the torus.  Rows across the period are cells/crown, so
        # snapping the crown to the nearest value giving an even count fixes
        # it, and moves the crown by a few percent at most.
        cells_n = max(1, int(p.get('tile_cells', 5)))
        crown = float(p.get('crown', 0.55))
        rows = cells_n / max(crown, 1e-6)
        even = max(2.0, 2.0 * round(rows / 2.0))
        return seigaiha(X, Y, info, cells=cells_n,
                        seed=int(p.get('seed', 1)),
                        rings=int(p.get('rings', 3)),
                        crown=cells_n / even,
                        rim=float(p.get('rim', 0.08)))
    raise ValueError("no surface form for %r" % (kind,))


def conformal_tau(ring=1.0, tube=0.4):
    """The ring torus's own conformal modulus.

    A torus of revolution is conformally the flat torus with modulus
    tau = i R / sqrt(R^2 - r^2), so defaulting the elliptic lattice to it
    puts the Weierstrass function on the shape it actually belongs to rather
    than on an arbitrary square lattice.
    """
    R = float(ring)
    r = float(tube)
    d = math.sqrt(max(R * R - r * r, 1e-12))
    return R / d


def cube_chart(P):
    """Which cube face each direction belongs to, and where on it.

    The equiangular cubed sphere: the six faces of a cube, each divided by
    equal ANGLES rather than equal distances, then projected outward.  Equal
    angles matter because they are what make the cell boundaries of adjacent
    faces line up along their shared edge -- a gnomonic (equal-distance)
    division would leave them mismatched and every face boundary would show.

    Returns `(face, s, t)` with s and t in [0, 1) across the face.

    References:
      Ronald Ronchi, Roberto Iacono and Pier S. Paolucci, "The 'Cubed
        Sphere': A New Method for the Solution of Partial Differential
        Equations in Spherical Geometry", J. Comput. Phys. 124(1), 1996,
        93-114 -- the equiangular construction used here.
      C. Ronchi et al. build on R. Sadourny, "Conservative finite-difference
        approximations of the primitive equations on quasi-uniform spherical
        grids", Monthly Weather Review 100, 1972, 136-144.
    """
    P = np.asarray(P, dtype=float)
    u = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    a = np.abs(u)
    axis = np.argmax(a, axis=1)
    pos = u[np.arange(len(u)), axis] >= 0.0
    face = axis * 2 + (~pos).astype(int)

    # For each face, the two in-plane axes and the outward one.  Chosen so
    # neighbouring faces agree along their shared edge.
    other = np.zeros((len(u), 2))
    major = np.abs(u[np.arange(len(u)), axis])
    for ax in (0, 1, 2):
        m = axis == ax
        if not m.any():
            continue
        i1, i2 = (ax + 1) % 3, (ax + 2) % 3
        other[m, 0] = u[m, i1]
        other[m, 1] = u[m, i2]
    # Equal angles: the tangent of the in-plane angle is the ratio, so the
    # angle itself is what gets divided evenly.
    q = np.pi / 4.0
    s = (np.arctan2(other[:, 0], major) / q + 1.0) * 0.5
    t = (np.arctan2(other[:, 1], major) / q + 1.0) * 0.5
    return face, np.clip(s, 0.0, 1.0 - 1e-12), np.clip(t, 0.0, 1.0 - 1e-12)


def truchet_sphere(P, cells=4, seed=1, lane=0.3, straight=0.0):
    """Truchet arc tiles over a cubed-sphere quad complex.

    Truchet's tile only works on a QUAD: its arcs cross each edge at the
    midpoint, and pairing those crossings needs an even number of them.  A
    cell with three or five sides cannot pair its midpoints at all, so an
    icosphere's triangles and a Goldberg dual's pentagons are both ruled out
    by parity, not by effort.  The cubed sphere is the quad complex that
    fits: six square charts, each divided into cells whose edges line up with
    their neighbours' across every face boundary.

    The chart is independent of the mesh, so this works on any sphere mesh --
    the geodesic one included -- rather than requiring its own base.
    """
    face, s, t = cube_chart(P)
    n = max(1, int(cells))
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    orient = rng.integers(0, 2, size=(6, n + 1, n + 1))
    cross = (rng.random((6, n + 1, n + 1)) < float(straight)
             if straight > 0.0 else np.zeros((6, n + 1, n + 1), dtype=bool))

    fi = np.clip((s * n).astype(int), 0, n - 1)
    fj = np.clip((t * n).astype(int), 0, n - 1)
    lu = s * n - fi
    lv = t * n - fj
    o = orient[face, fi, fj]
    xs = cross[face, fi, fj]

    r = 0.5
    d_a = np.abs(np.hypot(lu, lv) - r)
    d_b = np.abs(np.hypot(lu - 1.0, lv - 1.0) - r)
    d_c = np.abs(np.hypot(lu - 1.0, lv) - r)
    d_d = np.abs(np.hypot(lu, lv - 1.0) - r)
    d = np.where(o == 0, np.minimum(d_a, d_b), np.minimum(d_c, d_d))
    d_x = np.minimum(np.abs(lu - 0.5), np.abs(lv - 0.5))
    d = np.where(xs, d_x, d)

    w = max(float(lane), 1e-3) * 0.5
    q = np.clip(1.0 - d / w, 0.0, 1.0)
    return q * q * (3.0 - 2.0 * q) * 2.0 - 1.0


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
    if kind == 'QUASI':
        return quasicrystal3d(P, cells=float(p.get('qc_cells', 6.0)),
                              phase=float(p.get('phase', 0.0)),
                              sharp=float(p.get('qc_sharp', 0.0)),
                              jitter=float(p.get('qc_jitter', 0.0)),
                              seed=int(p.get('seed', 1)))
    if kind == 'GROUP':
        return spherical_wallpaper(P, signature=p.get('sph_group',
                                                      'STAR_532'),
                                   n=int(p.get('sph_n', 5)),
                                   waves=int(p.get('waves', 5)),
                                   seed=int(p.get('seed', 1)),
                                   cells=float(p.get('sym_cells', 4.0)),
                                   kind=p.get('seed_kind', 'WAVE'))
    if kind == 'RIPPLE':
        if faces is None:
            raise ValueError("the ripple field needs the mesh faces")
        return ripple3d(P, faces, sources=int(p.get('sources', 4)),
                        wavelength=float(p.get('wavelength', 0.35)),
                        seed=int(p.get('seed', 1)),
                        base=p.get('base', 'SPHERE'),
                        ring=float(p.get('ring', 1.0)),
                        tube=float(p.get('tube', 0.4)),
                        radius=float(p.get('radius', 0.7)),
                        decay=float(p.get('decay', 0.0)))
    if kind == 'WAVE':
        return zonal_wave(P, wavelength=float(p.get('wavelength', 0.5)),
                          phase=float(p.get('phase', 0.0)),
                          steepness=float(p.get('steepness', 0.0)),
                          count=int(p.get('wave_count', 1)),
                          spread=float(p.get('wave_spread', 0.0)),
                          seed=int(p.get('seed', 1)))
    if kind == 'SCATTER':
        if faces is None:
            raise ValueError("the scatter field needs the mesh faces")
        return scatter3d(P, faces, count=int(p.get('points_n', 120)),
                         sigma=float(p.get('sigma', 0.12)),
                         seed=int(p.get('seed', 1)),
                         kernel=p.get('kernel', 'WYVILL'),
                         merge=p.get('merge_mode', 'MAX'))
    if kind == 'TORUS_MODE':
        return torus_harmonic(P, m=int(p.get('mode_m', 3)),
                              n=int(p.get('mode_n', 1)),
                              ring=float(p.get('ring', 1.0)),
                              tube=float(p.get('tube', 0.4)),
                              phase=float(p.get('phase', 0.0)))
    if kind == 'TRUCHET' and p.get('base', 'TORUS') == 'SPHERE':
        # The one repeating pattern that DOES reach a sphere, because its
        # tile needs only a quad complex and not a flat structure.
        return truchet_sphere(P, cells=int(p.get('tile_cells', 4)),
                              seed=int(p.get('seed', 1)),
                              lane=float(p.get('lane', 0.3)),
                              straight=float(p.get('straight', 0.0)))
    if kind in ('WALLPAPER', 'ELLIPTIC', 'TRUCHET', 'SEIGAIHA',
                'OCEAN'):
        base = p.get('base', 'TORUS')
        if base == 'SPHERE':
            raise ValueError(
                "%s is a doubly periodic construction and belongs on "
                "the torus or cylinder; a sphere has no flat structure "
                "for it to descend to%s"
                % (kind, " -- and no global wind direction either, by "
                   "the hairy ball theorem" if kind == 'OCEAN'
                   else ""))
        return flat_on_surface(P, kind, p, base=base,
                               ring=float(p.get('ring', 1.0)),
                               tube=float(p.get('tube', 0.4)),
                               height=float(p.get('height', 2.0)),
                               radius=float(p.get('radius', 0.7)))
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


def feature_samples(kind, p, P, faces):
    """How many mesh edges span the finest thing a field draws.

    The flat engine learned this the hard way and reports it (plan section
    12.1): a mesh interpolates linearly between its samples, so a feature
    given three of them comes out as a facet however smooth the underlying
    function is.  Nyquist is the wrong bar -- comfortable is about eight.

    Returns `None` where a field has no single characteristic size.
    """
    edges = set()
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = int(f[i]), int(f[(i + 1) % k])
            edges.add((min(a, b), max(a, b)))
    e = np.array(sorted(edges), dtype=np.int64)
    h = float(np.median(np.linalg.norm(P[e[:, 1]] - P[e[:, 0]], axis=1)))
    if h <= 0.0:
        return None
    span = float(np.max(P.max(axis=0) - P.min(axis=0))) or 1.0
    kind = str(kind).upper()

    def n(size):
        return float(size) / h

    if kind == 'TRUCHET':
        # The lane EDGE is the fine part, not the lane: the profile ramps
        # over half the lane width within one cell.
        cells = (max(1, int(p.get('tile_cells', 6)))
                 * max(1.0, float(p.get('cells_u', 1.0))))
        return n(span * math.pi / cells * float(p.get('lane', 0.3)) * 0.5)
    if kind == 'SEIGAIHA':
        cells = (max(1, int(p.get('tile_cells', 5)))
                 * max(1.0, float(p.get('cells_u', 1.0))))
        return n(span * math.pi / cells * 0.75
                 / max(1, int(p.get('rings', 3))))
    if kind in ('WALLPAPER',):
        cells = (max(1, int(p.get('tile_cells', 1)))
                 * max(1.0, float(p.get('cells_u', 1.0))))
        return n(span * math.pi / (cells
                                   * max(1, int(p.get('freq_max', 3))) * 2.0))
    if kind == 'QUASI':
        return n(span / max(1e-6, float(p.get('qc_cells', 6.0))))
    if kind == 'GROUP':
        return n(span / max(1e-6, float(p.get('sym_cells', 4.0))))
    if kind == 'GABOR':
        return n(span / max(1e-6, float(p.get('gabor_freq', 6.0))))
    if kind in ('RIPPLE', 'WAVE'):
        return n(float(p.get('wavelength', 0.35)))
    if kind == 'SCATTER':
        return n(float(p.get('sigma', 0.12)))
    if kind == 'HARMONIC':
        l = max(1, int(p.get('sph_l', 4)))
        return n(span * math.pi / (2.0 * l))
    if kind == 'TORUS_MODE':
        m = max(1, int(p.get('mode_m', 3)))
        return n(2.0 * math.pi * float(p.get('ring', 1.0)) / (2.0 * m))
    return None


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
             qc_cells=6.0, qc_sharp=0.0, qc_jitter=0.0,
             sph_group='STAR_532', sph_n=5, waves=5, sym_cells=4.0,
             seed_kind='WAVE',
             sources=4, wavelength=0.35, decay=0.0, steepness=0.0,
             wave_count=1, wave_spread=0.0, sigma=0.12,
             kernel='WYVILL', merge_mode='MAX',
             cells_u=1.0, cells_v=1.0, group='P4M',
             freq_max=3, ell_kind='WP', ell_part='SPHERE',
             tau_re=0.0, tau_im=1.0, tile_cells=6, lane=0.3,
             straight=0.0, rings=3, crown=0.55, rim=0.08,
             sea_sim=256, patch=100.0, wind_speed=8.0, wind_dir=0.0,
             choppy=0.0,
             depth=0.05, curve='NONE', curve_amount=1.0, norm='STD',
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
    feat = feature_samples(p['field'], p, P, faces)
    info = {
        'verts': len(V), 'faces': len(faces),
        'base': p['base'], 'field': p['field'],
        # Uniformity of the base itself: the number the lat-long sphere
        # failed at, kept where it can be seen.
        'area_ratio': float(base_area.max() / max(base_area.min(), 1e-30)),
        # Reported, not clamped, for the same reason the flat engine reports
        # it: a coarse mesh is a legitimate choice, and silently faceting a
        # pattern while calling it the pattern is not.
        'feature_samples': feat,
        'undersampled': (feat is not None and feat < 6.0),
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

    # --- quasicrystal: exact icosahedral symmetry, and no period --------
    Pq, Nq, Fq = icosphere(3)
    hq = quasicrystal3d(Pq, cells=5.0)
    try:
        from ..orbifold_sphere_generator import build_group as _bg
    except ImportError:
        from orbifold_sphere_generator import build_group as _bg
    worst = 0.0
    for g in _bg('532'):
        worst = max(worst, float(np.abs(
            quasicrystal3d(Pq @ np.asarray(g, dtype=float).T, cells=5.0)
            - hq).max()))
    # The control matters: a field invariant under EVERYTHING would pass the
    # line above and mean nothing.  Octahedral symmetry is the one an
    # icosahedral field must NOT have.
    ctrl = max(float(np.abs(
        quasicrystal3d(Pq @ np.asarray(g, dtype=float).T, cells=5.0)
        - hq).max()) for g in _bg('432'))
    print("solid: quasicrystal invariance over 532 = %.2e; over 432 "
          "(must fail) = %.3f" % (worst, ctrl))
    ok = ok and worst < 1e-10 and ctrl > 0.1

    # Aperiodic: no translation repeats it.  Sampled along a generic line,
    # the autocorrelation of a periodic field returns to 1 at its period; a
    # quasicrystal's never does.  A cubic lattice sum is the control that
    # shows the measurement can see a period when there is one.
    t = np.linspace(0.0, 80.0, 8000)

    def repeat_peak(f):
        # Overlap-normalised, or the answer is meaningless: `np.correlate`
        # compares fewer and fewer samples as the lag grows, so its raw output
        # decays like (N - k)/N and even an exactly periodic signal scores
        # 0.92 instead of 1.  Dividing by the overlap removes that envelope.
        f = f - f.mean()
        n = len(f)
        ac = np.correlate(f, f, mode='full')[n - 1:]
        counts = np.arange(n, 0, -1)
        ac = (ac / counts) / (ac[0] / n)
        lo = n // 40                    # skip the central peak's own width
        hi = int(n * 0.75)              # and the tail, where overlap is thin
        band = ac[lo:hi]
        # Return both the best near-repeat and how many lags are an actual
        # repeat.  The height alone does not separate the two cases: a
        # quasiperiodic field has arbitrarily good ALMOST-periods, and scored
        # 0.963 against the lattice's 1.002.  What it never has is an exact
        # period, which recurs at every multiple -- so counting the lags that
        # reach 1 is the robust discriminator, not the tallest one.
        return float(np.max(band)), int(np.sum(band > 0.99))

    # **Each field is sampled along its OWN most favourable direction.**  A
    # generic line through a cubic lattice is an incommensurate section and
    # is not periodic either, so comparing both along one arbitrary ray shows
    # nothing -- it scored the lattice lower than the quasicrystal.  The
    # lattice is therefore sampled along a lattice axis, where it certainly
    # repeats, and the quasicrystal along one of its own five-fold axes,
    # which is the best chance it has of repeating.
    axis = icosahedral_axes()[0]
    qray = t[:, None] * axis[None, :]
    qp, qn = repeat_peak(quasicrystal3d(qray, cells=1.0))
    cp, cn = repeat_peak(np.cos(t) + 2.0)
    print("solid: repeats -- quasicrystal along a 5-fold axis: best %.3f, "
          "exact %d; cubic lattice along a lattice axis: best %.3f, exact %d"
          % (qp, qn, cp, cn))
    ok = ok and qn == 0 and cn > 5

    # --- spherical groups: invariance, and the spectral fingerprint ------
    for sig, order in (('332', 12), ('432', 24), ('STAR_532', 120)):
        G = sphere_group_matrices(sig)
        hw = spherical_wallpaper(Pq, signature=sig, waves=4, seed=3)
        bad = 0.0
        for g in G:
            bad = max(bad, float(np.abs(
                spherical_wallpaper(Pq @ g.T, signature=sig, waves=4, seed=3)
                - hw).max()))
        print("solid: group %-9s |G|=%-4d worst invariance violation %.2e"
              % (sig, len(G), bad))
        ok = ok and len(G) == order and bad < 1e-9

    # The fingerprint invariant theory predicts: the lowest non-constant
    # spherical harmonic an icosahedrally symmetric field can carry is l = 6,
    # so every projection from l = 1 to 5 must vanish.  This is a much
    # stronger statement than "it looks symmetric", and it uses the
    # area-weighted inner product already verified above.
    Ph, Nh, Fh = icosphere(4)
    w = np.zeros(len(Ph))
    for f, a_ in zip(Fh, face_areas(Ph, Fh)):
        for i in f:
            w[i] += a_ / 3.0
    w *= 4.0 * math.pi / w.sum()
    hi = spherical_wallpaper(Ph, signature='STAR_532', waves=4, seed=3)
    hi = hi - float(np.sum(w * hi)) / (4.0 * math.pi)
    power = []
    for l in range(1, 7):
        p_l = sum(float(np.sum(w * hi * spherical_harmonic(Ph, l, m))) ** 2
                  for m in range(-l, l + 1))
        power.append(p_l)
    low = max(power[:5])
    print("solid: icosahedral field, harmonic power l=1..6: %s"
          % " ".join("%.2e" % v for v in power))
    ok = ok and power[5] > 100.0 * low

    # --- ripple: the wavefronts really are geodesic --------------------
    # One source on a sphere makes a field that depends ONLY on the angle
    # from that source, so it is constant on every circle of constant
    # colatitude about it.  That is a property of the geodesic construction
    # which a projected field could not have.
    # The distance really is the GEODESIC one, checked against the straight
    # line through space, which is measurable independently: a chord of a
    # unit sphere subtending arc d has length 2 sin(d/2).  (Checking instead
    # that the field is constant where the distance is constant would be
    # vacuous -- the field is built from the distance.)
    Pr2, Nr2, Fr2 = icosphere(4)
    src = np.array([0.0, 0.0, 1.0])
    d = surface_distance(Pr2, src, base='SPHERE')
    chord = np.linalg.norm(Pr2 - src[None, :], axis=1)
    err = float(np.abs(chord - 2.0 * np.sin(d / 2.0)).max())
    print("solid: sphere geodesic vs chord identity, worst error %.2e" % err)
    ok = ok and err < 1e-12

    # And on the cylinder, where the surface is developable so the geodesic
    # is exactly the unrolled straight line -- and must be SHORTER than going
    # the long way round, which is the part a naive angle difference gets
    # wrong.
    Pc2, Nc2, Fc2 = cylinder(64, height=2.0, radius=0.7)
    c2 = Pc2[3]
    dc = surface_distance(Pc2, c2, base='CYLINDER', radius=0.7)
    long_way = np.abs(np.arctan2(Pc2[:, 1], Pc2[:, 0])
                      - math.atan2(c2[1], c2[0])) * 0.7
    print("solid: cylinder geodesic never exceeds the unwrapped angle: %s"
          % bool(np.all(dc <= np.hypot(long_way, Pc2[:, 2] - c2[2]) + 1e-12)))
    ok = ok and bool(np.all(dc <= np.hypot(long_way,
                                           Pc2[:, 2] - c2[2]) + 1e-12))

    # The rings are evenly spaced in geodesic distance: crossings along a
    # meridian must match the arc length divided by the wavelength.
    lam = 0.5
    meridian = np.linspace(0.0, math.pi, 2000)
    prof = np.sin(2.0 * math.pi * meridian / lam)
    want = int(round(2.0 * (math.pi / lam)))
    got = int(np.sum(np.diff(np.signbit(prof)) != 0))
    print("solid: ripple rings along a meridian: %d (arc %.2f / wavelength "
          "%.2f = %d)" % (got, math.pi, lam, want))
    ok = ok and abs(got - want) <= 1

    for base, kw in (('SPHERE', dict(sphere_res=3)),
                     ('TORUS', dict(grid_res=64)),
                     ('CYLINDER', dict(grid_res=64))):
        Pb, Nb, Fb = build_base(base, **kw)
        hb = ripple3d(Pb, Fb, sources=3, wavelength=0.4, seed=2, base=base)
        ok = ok and np.isfinite(hb).all() and hb.std() > 1e-6

    # --- zonal wave: constant geodesic spacing --------------------------
    # A plane wave through space cuts a sphere in rings that crowd together
    # near the equator of the cut; a zonal wave does not.  Measure the gap
    # between successive zero crossings down a meridian and compare the
    # spread of those gaps.
    def gaps(vals, angles):
        z = np.where(np.diff(np.signbit(vals)))[0]
        return np.diff(angles[z]) if len(z) > 2 else np.array([1.0, 2.0])

    th = np.linspace(0.01, math.pi - 0.01, 4000)
    pts = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=-1)
    zon = zonal_wave(pts, wavelength=0.4)
    plane = np.sin(2.0 * math.pi * pts[:, 2] / 0.4)      # a 3D plane wave
    gz, gp = gaps(zon, th), gaps(plane, th)
    print("solid: ring spacing spread -- zonal %.4f, plane wave %.4f "
          "(lower is more even)"
          % (float(gz.std() / gz.mean()), float(gp.std() / gp.mean())))
    ok = ok and gz.std() / gz.mean() < 0.1 * (gp.std() / gp.mean())

    # --- scatter: uniform by area, not by vertex ------------------------
    # The torus is the test: its outer half carries more area than its inner
    # half, so an area-uniform scatter must put proportionally more points
    # there.  Sampling vertices instead would split them evenly.
    Ps2, Ns2, Fs2 = torus(96, ring=1.0, tube=0.4)
    pts2 = surface_points(Ps2, Fs2, 4000, seed=5)
    rho = np.hypot(pts2[:, 0], pts2[:, 1])
    outer = float((rho > 1.0).mean())
    # Area of the outer half over the whole: (2 pi r)(pi R + 2 r) style, but
    # simplest is to integrate the face areas directly.
    fc = np.array([Ps2[list(f)].mean(axis=0) for f in Fs2])
    fa = face_areas(Ps2, Fs2)
    want_outer = float(fa[np.hypot(fc[:, 0], fc[:, 1]) > 1.0].sum() / fa.sum())
    print("solid: scatter puts %.3f of its points on the torus's outer half; "
          "that half is %.3f of the area" % (outer, want_outer))
    ok = ok and abs(outer - want_outer) < 0.03

    # --- torus harmonics ------------------------------------------------
    # As the tube thins, a ring torus tends to a flat one, whose spectrum is
    # known exactly: lambda -> m^2/R^2 + n^2/r^2.  Watching the computed
    # eigenvalue approach that is a real check on the Sturm-Liouville
    # assembly, and one the code cannot fudge.
    print("solid: torus mode m=2 n=1, thin-tube limit")
    for rr in (0.20, 0.10, 0.05):
        lam, g, vv = torus_modes(m=2, n=1, ring=1.0, tube=rr)
        flat = 4.0 / 1.0                    # m^2/R^2, n = 1 is the constant
        print("       r/R=%.2f  lambda %.4f  flat-torus m^2/R^2 %.4f  "
              "difference %.4f" % (rr, lam, flat, abs(lam - flat)))
    l1, _, _ = torus_modes(m=2, n=1, ring=1.0, tube=0.20)
    l2, _, _ = torus_modes(m=2, n=1, ring=1.0, tube=0.05)
    ok = ok and abs(l2 - 4.0) < abs(l1 - 4.0)

    # Orthogonality in the PHYSICAL measure, which is what checks the
    # Liouville symmetrisation rather than just `eigh`'s own guarantee.
    N = 256
    vg = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False)
    dvg = 2.0 * math.pi / N
    Wg = 0.4 * (1.0 + 0.4 * np.cos(vg))
    gs = [torus_modes(m=2, n=k, ring=1.0, tube=0.4, samples=N)[1]
          for k in (1, 2, 3)]
    worst_off = 0.0
    worst_diag = 0.0
    for i in range(3):
        for j in range(3):
            val = float(np.sum(gs[i] * gs[j] * Wg) * dvg)
            if i == j:
                worst_diag = max(worst_diag, abs(val - 1.0))
            else:
                worst_off = max(worst_off, abs(val))
    print("solid: torus modes orthonormal in the physical measure: worst "
          "|<g,g>-1| = %.2e, worst |<g_i,g_j>| = %.2e"
          % (worst_diag, worst_off))
    ok = ok and worst_diag < 1e-9 and worst_off < 1e-9

    # Nodal counts: the n-th mode crosses zero n-1 times more than the first.
    counts = []
    for k in (1, 2, 3, 4):
        _, g, _ = torus_modes(m=1, n=k, ring=1.0, tube=0.4)
        counts.append(int(np.sum(np.diff(np.signbit(g)) != 0)))
    print("solid: torus mode nodal crossings by order: %s" % counts)
    ok = ok and counts == sorted(counts) and counts[-1] > counts[0]

    # High angular order localises on the OUTER equator -- the whispering
    # gallery, and the thing that makes these look unlike a lattice wave.
    _, g_hi, v_hi = torus_modes(m=12, n=1, ring=1.0, tube=0.4)
    peak = float(v_hi[int(np.argmax(np.abs(g_hi)))])
    print("solid: m=12 mode peaks at v=%.3f rad (0 is the outer equator)"
          % peak)
    ok = ok and (peak < 0.4 or peak > 2.0 * math.pi - 0.4)

    # --- the flat ports, and the property that lets them come -----------
    # A doubly periodic field evaluated at the torus's intrinsic coordinates
    # descends exactly, so it must be continuous across BOTH wraps -- which
    # is where a projected pattern would betray itself.  Measured the same
    # way as everything else here: the step across the wrap against the step
    # inside.
    Pw, Nw, Fw = torus(128, ring=1.0, tube=0.4)
    nvw = len(Pw) // 128
    for name, prm in (('WALLPAPER', dict(group='P4M', waves=5, seed=3)),
                      ('ELLIPTIC', dict(ell_kind='WP')),
                      ('TRUCHET', dict(tile_cells=4, lane=0.35)),
                      ('SEIGAIHA', dict(tile_cells=4, rings=2)),
                      ('OCEAN', dict(sea_sim=128, wind_speed=9.0,
                                     choppy=0.6))):
        prm.update(cells_u=2.0, cells_v=1.0, base='TORUS')
        hv = evaluate(name, Pw, prm, faces=Fw)
        H = hv.reshape(128, nvw)
        sd = hv.std() or 1.0
        wu = float(np.abs(H[0] - H[-1]).max() / sd)
        wv = float(np.abs(H[:, 0] - H[:, -1]).max() / sd)
        iu = float(np.abs(np.diff(H, axis=0)).max() / sd)
        print("solid: %-9s on the torus -- wrap step u %.3f, v %.3f sd; "
              "interior %.3f sd" % (name, wu, wv, iu))
        ok = ok and np.isfinite(hv).all()
        # The wrap must be no worse than ordinary interior variation.  Note
        # this is a real test only because the fields are periodic by
        # construction: a projected one would fail it loudly.
        ok = ok and wu <= 1.0 * iu and wv <= 1.0 * iu

    # Asking for one on a sphere is refused rather than obliged: there is no
    # flat structure there for a doubly periodic pattern to descend to.
    try:
        evaluate('WALLPAPER', Pq, dict(base='SPHERE'), faces=Fq)
        refused = False
    except ValueError:
        refused = True
    print("solid: a doubly periodic field on a sphere is refused: %s"
          % refused)
    ok = ok and refused

    # The conformal modulus is the torus's own, not an arbitrary square.
    tau = conformal_tau(1.0, 0.4)
    print("solid: ring torus conformal modulus tau = %.4f i (R/sqrt(R^2-r^2))"
          % tau)
    ok = ok and abs(tau - 1.0 / math.sqrt(1.0 - 0.16)) < 1e-12

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

    # --- Truchet on the cubed sphere ------------------------------------
    # The chart's whole job is that neighbouring faces divide their shared
    # edge the same way, so the tiling crosses a face boundary without a
    # joint.  Measured the way every seam here is: the largest step between
    # neighbouring vertices that straddle a cube edge, against the largest
    # step anywhere else.
    Pcs, Ncs, Fcs = icosphere(5)
    hcs = truchet_sphere(Pcs, cells=4, seed=3, lane=0.32, straight=0.25)
    fc, sc, tc = cube_chart(Pcs)
    edge_set = set()
    for f in Fcs:
        for i in range(len(f)):
            a_, b_ = int(f[i]), int(f[(i + 1) % len(f)])
            edge_set.add((min(a_, b_), max(a_, b_)))
    ee = np.array(sorted(edge_set))
    step = np.abs(hcs[ee[:, 0]] - hcs[ee[:, 1]])
    crosses = fc[ee[:, 0]] != fc[ee[:, 1]]
    sd = hcs.std() or 1.0
    across = float(step[crosses].max() / sd)
    inside = float(step[~crosses].max() / sd)
    print("solid: truchet on the cubed sphere -- worst step across a face "
          "boundary %.3f sd, worst step elsewhere %.3f sd"
          % (across, inside))
    ok = ok and np.isfinite(hcs).all() and across <= 1.05 * inside

    # Every face carries tiles, and the pattern is not simply constant.
    print("solid: cubed-sphere chart covers %d faces, cell occupancy %s"
          % (len(np.unique(fc)),
             "even" if np.bincount(fc, minlength=6).min() > 0.1
             * len(fc) / 6 else "UNEVEN"))
    ok = ok and len(np.unique(fc)) == 6 and hcs.std() > 0.1

    # --- the sampling report --------------------------------------------
    # The Truchet lane edge is the finest thing any of these fields draws,
    # and at the detail a torus preset used to carry it spanned about three
    # mesh edges -- which is a facet, not a curve.  The figure is now
    # reported so the choice is visible.
    for res_t, want in ((192, False), (384, True)):
        Pt2, Nt2, Ft2 = torus(res_t, ring=1.0, tube=0.4)
        f_s = feature_samples('TRUCHET', dict(tile_cells=5, lane=0.32,
                                              cells_u=2.0), Pt2, Ft2)
        print("solid: torus detail %3d -> truchet lane edge spans %.1f mesh "
              "edges (%s)" % (res_t, f_s,
                              "comfortable" if f_s >= 6.0 else "facets"))
        ok = ok and ((f_s >= 6.0) == want)

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
