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
FIELDS3D = ('FRACTAL', 'CELLULAR', 'GABOR', 'HARMONIC')


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
    nv = max(8, int(res * tube / ring / 2.0)) if ring > 0 else nu
    nv = max(8, nv)
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


def build_base(base='SPHERE', res=4, ring=1.0, tube=0.4, height=2.0,
               radius=0.7):
    """`(points, normals, faces)` for one of the closed bases."""
    if base == 'SPHERE':
        return icosphere(int(res))
    if base == 'TORUS':
        return torus(int(res), ring=ring, tube=tube)
    if base == 'CYLINDER':
        return cylinder(int(res), height=height, radius=radius)
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


def evaluate(kind, P, p):
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
    p = dict(base='SPHERE', res=4, ring=1.0, tube=0.4, height=2.0,
             radius=0.7, field='FRACTAL', method='FBM', octaves=8,
             lacunarity=2.0, dim=2.3, hurst=0.7, modes=240, seed=1,
             field_scale=1.0, points_n=120, cell_mode='CRACK',
             cell_sharp=1.0, gabor_freq=6.0, gabor_band=0.35, spread=3.1416,
             sph_l=4, sph_m=2, mode_m=3, mode_n=2, mode_k=1, phase=0.0,
             depth=0.25, curve='NONE', curve_amount=1.0, norm='STD',
             fit='CUBE', scale=1.0, span=2.0)
    p.update(kw)

    P, N, faces = build_base(p['base'], res=p['res'], ring=p['ring'],
                             tube=p['tube'], height=p['height'],
                             radius=p['radius'])
    h = evaluate(p['field'], P, p)

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
            hb = evaluate(fld, Pb, dict(seed=3, points_n=80, octaves=6))
            step, mean = worst_step(Pb, Fb, hb)
            print("solid: %-8s %-8s worst neighbour step %.2f sd "
                  "(mean %.4f)" % (base, fld, step, mean))
            # A seam shows up as a step many times the field's own spread.
            ok = ok and step < 6.0 and np.isfinite(hb).all()

    # The torus wraps in BOTH directions, and that is where a projected
    # field would betray itself.  Check the two wrap rings explicitly.
    Pt, Nt, Ft = torus(96)
    ht = evaluate('FRACTAL', Pt, dict(seed=5, octaves=6))
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
