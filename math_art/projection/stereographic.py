# Stereographic projection, and the shadows a curve casts under it.
#
# Part of the Math Art projection engine (`math_art/projection/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Stereographic projection maps the sphere minus a point onto the plane
# from the point directly opposite.  It is CONFORMAL -- it preserves
# angles exactly while distorting area without bound -- and it carries
# every circle on the sphere to a circle or a line in the plane, which
# is what makes it the projection worth building sculpture around.
#
# References:
# - Hipparchus and Ptolemy used it for star charts; the conformality is
#   due to E. Halley, "An easie Demonstration of the Analogy of the
#   Logarithmick Tangents...", Phil. Trans. 19, 1695.
# - D. Hilbert and S. Cohn-Vossen, "Geometry and the Imagination",
#   Chelsea, 1952, section 34 -- circles to circles.

import math
from math import pi, sqrt
import numpy as np


_PHI = (1.0 + sqrt(5.0)) / 2.0


def _valid_pq(p, q):
    """Clamp (p, q) to a spherical tiling: 1/p + 1/q > 1/2. p = 2
    gives the hosohedron (q meridians); q is kept >= 3 so the
    dihedron (a free-floating equator band) cannot occur."""
    p = max(2, min(p, 5))
    if p == 2:
        return p, max(3, q)
    return p, max(3, min(q, {3: 5, 4: 3, 5: 3}[p]))


def _tiling_edges(p, q):
    """Edges of the spherical {p,q} tiling as pairs of unit
    vectors, plus the common edge arc length (radians)."""
    if p == 2:                        # hosohedron: q meridians,
        n = np.array((0.0, 0.0, 1.0))  # split at the equator so no
        s = -n                         # arc joins antipodal points
        eq = [np.array((math.cos(2 * pi * k / q),
                        math.sin(2 * pi * k / q), 0.0))
              for k in range(q)]
        return ([(n, e) for e in eq] + [(e, s) for e in eq],
                pi / 2)
    if (p, q) == (3, 3):
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    elif (p, q) == (3, 4):
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
    elif (p, q) == (4, 3):
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
    elif (p, q) == (3, 5):
        V = [v for a in (-1.0, 1.0) for b in (-_PHI, _PHI)
             for v in ((0, a, b), (a, b, 0), (b, 0, a))]
    else:                             # (5, 3) dodecahedron
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
        V += [v for a in (-1 / _PHI, 1 / _PHI)
              for b in (-_PHI, _PHI)
              for v in ((0, a, b), (a, b, 0), (b, 0, a))]
    V = [np.asarray(v, float) / np.linalg.norm(v) for v in V]
    # edges = vertex pairs at the minimal chord distance
    dmin = min(np.linalg.norm(a - b)
               for i, a in enumerate(V) for b in V[i + 1:])
    edges = [(a, b) for i, a in enumerate(V) for b in V[i + 1:]
             if np.linalg.norm(a - b) < dmin * 1.001]
    length = math.acos(max(-1.0, min(1.0, float(
        edges[0][0] @ edges[0][1]))))
    return edges, length


def _arc_dist_min(D, edges):
    """Minimal angular distance from each unit vector in D (n,3)
    to any of the given great-circle arcs."""
    best = np.full(len(D), pi)
    for a, b in edges:
        n = np.cross(a, b)
        n /= np.linalg.norm(n)
        ta = np.cross(n, a)           # tangent at a, towards b
        tb = np.cross(b, n)           # tangent at b, towards a
        inside = (D @ ta >= 0.0) & (D @ tb >= 0.0)
        s = np.abs(np.arcsin(np.clip(D @ n, -1.0, 1.0)))
        da = np.arccos(np.clip(D @ a, -1.0, 1.0))
        db = np.arccos(np.clip(D @ b, -1.0, 1.0))
        best = np.minimum(best,
                          np.where(inside, s, np.minimum(da, db)))
    return best


def _fill_divisions(edges, lo, hi, step):
    """Sorted division points of [lo, hi] containing every edge in
    `edges`, with each gap filled uniformly at roughly `step`."""
    pts = sorted({lo, hi, *(e for e in edges if lo < e < hi)})
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil((b - a) / step - 1e-9)))
        out.extend(a + (b - a) * k / n for k in range(n))
    out.append(hi)
    return out


def _thicken(verts_outer, verts_inner, faces):
    """Closed solid from an open pattern surface: outer faces +
    reversed inner faces + side walls on every boundary edge."""
    n = len(verts_outer)
    ecnt = {}
    for f in faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            key = (a, b) if a < b else (b, a)
            ecnt[key] = None if key in ecnt else (a, b)
    out = [list(f) for f in faces]
    out += [[v + n for v in reversed(f)] for f in faces]
    out += [[e[1], e[0], e[0] + n, e[1] + n]
            for e in ecnt.values() if e is not None]
    return list(verts_outer) + list(verts_inner), out


def _grid_shell(R, t, width, spacing, Rmax, res):
    """Exact square-grid shell: the plane is meshed on a Cartesian
    grid whose division lines lie exactly on the strip edges, so
    every cell is entirely material or entirely hole and the hole
    boundaries are exact (no staircase).  The domain extends well
    past the pattern disc as solid cells (their image crowds the
    north pole), and the last gap is closed with a fan to the
    pole."""
    s = spacing * R
    hw = 0.5 * width * s
    step = 2.0 * Rmax / max(res, 8)
    kmax = int(math.floor((Rmax + s) / s))
    edges = []
    for k in range(-kmax, kmax + 1):
        edges += [k * s - hw, k * s + hw]
    A = 8.0 * Rmax
    div = _fill_divisions(edges, -Rmax, Rmax, step)
    ext = []
    v = Rmax
    while v < A:
        v *= 1.6
        ext.append(min(v, A))
    xs = [-e for e in reversed(ext)] + div + ext
    ys = list(xs)
    nx, ny = len(xs), len(ys)

    def on_strip(a, b):
        m = 0.5 * (a + b)
        return abs(m - s * round(m / s)) < hw

    verts_o = []
    verts_i = []
    C = np.array((0.0, 0.0, R))

    def emit(x, y):
        r = math.hypot(x, y)
        th = 2.0 * math.atan2(2.0 * R, r) if r > 1e-12 else pi
        ph = math.atan2(y, x)
        d = np.array((math.sin(th) * math.cos(ph),
                      math.sin(th) * math.sin(ph), math.cos(th)))
        verts_o.append(tuple(C + (R + 0.5 * t) * d))
        verts_i.append(tuple(C + (R - 0.5 * t) * d))
        return len(verts_o) - 1

    vid = [[emit(x, y) for y in ys] for x in xs]
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            cx = 0.5 * (xs[i] + xs[i + 1])
            cy = 0.5 * (ys[j] + ys[j + 1])
            if not (on_strip(xs[i], xs[i + 1])
                    or on_strip(ys[j], ys[j + 1])):
                # part of a hole square: open it only when the
                # WHOLE square sits inside the pattern disc, so
                # rim holes are complete squares or nothing
                hx = s * (math.floor(cx / s) + 0.5)
                hy = s * (math.floor(cy / s) + 0.5)
                half = 0.5 * s - hw
                far = math.hypot(abs(hx) + half, abs(hy) + half)
                if far < Rmax:
                    continue                 # open hole cell
            faces.append([vid[i][j], vid[i + 1][j],
                          vid[i + 1][j + 1], vid[i][j + 1]])
    # close the far square boundary with a fan to the north pole
    pole = emit(0.0, 0.0)
    verts_o[pole] = (0.0, 0.0, 2.0 * R + 0.5 * t)
    verts_i[pole] = (0.0, 0.0, 2.0 * R - 0.5 * t)
    ring = ([vid[i][0] for i in range(nx)]
            + [vid[nx - 1][j] for j in range(1, ny)]
            + [vid[i][ny - 1] for i in range(nx - 2, -1, -1)]
            + [vid[0][j] for j in range(ny - 2, 0, -1)])
    for k in range(len(ring)):
        faces.append([pole, ring[k], ring[(k + 1) % len(ring)]])
    # orient outward (positive volume about the sphere centre)
    Vo = np.asarray(verts_o)
    vol = 0.0
    for f in faces[:200]:
        P = Vo[f] - C
        for k in range(1, len(f) - 1):
            vol += float(np.dot(P[0], np.cross(P[k], P[k + 1])))
    if vol < 0:
        faces = [list(reversed(f)) for f in faces]
    return _thicken(verts_o, verts_i, faces)


def _field_shell(R, t, thetas, phis, gfun, has_north=True):
    """Marching-squares shell over the lat-long grid: gfun(TH, PH)
    is a signed field (material < 0).  Cells fully material become
    quads; mixed cells are clipped at zero crossings interpolated
    along the grid edges, so pattern boundaries are smooth curves
    limited only by the resolution, never staircased."""
    F = len(thetas) - 1
    L = len(phis)
    C = np.array((0.0, 0.0, R))
    verts_o = []
    verts_i = []
    thof = []
    phof = []
    gof = []

    def emit(th, ph, g):
        d = np.array((math.sin(th) * math.cos(ph),
                      math.sin(th) * math.sin(ph), math.cos(th)))
        verts_o.append(C + (R + 0.5 * t) * d)
        verts_i.append(C + (R - 0.5 * t) * d)
        thof.append(th)
        phof.append(ph)
        gof.append(g)
        return len(verts_o) - 1

    def geval(th, ph):
        g = float(gfun(np.array([[th]]), np.array([[ph]]))[0, 0])
        return g if g != 0.0 else 1e-12

    lo = 1 if has_north else 0
    TH, PH = np.meshgrid(thetas[lo:F], phis, indexing='ij')
    G = gfun(TH, PH)
    G[G == 0.0] = 1e-12
    rid = {}
    for ii, i in enumerate(range(lo, F)):
        for j in range(L):
            rid[(i, j)] = emit(thetas[i], phis[j], G[ii, j])
    north = emit(0.0, 0.0, geval(1e-9, 0.0)) if has_north else None
    south = emit(pi, 0.0, geval(pi, 0.0))

    def vid(i, j):
        if has_north and i == 0:
            return north
        if i == F:
            return south
        return rid[(i, j % L)]

    xing = {}

    def crossing(a, b):
        key = (a, b) if a < b else (b, a)
        if key in xing:
            return xing[key]
        ga, gb = gof[a], gof[b]
        u = ga / (ga - gb)
        pha, phb = phof[a], phof[b]
        if a in (north, south):
            pha = phb
        if b in (north, south):
            phb = pha
        dph = (phb - pha + pi) % (2.0 * pi) - pi
        idx = emit(thof[a] + u * (thof[b] - thof[a]),
                   pha + u * dph, 0.0)
        xing[key] = idx
        return idx

    faces = []
    for i in range(F):
        for j in range(L):
            cs = []
            for v in (vid(i, j), vid(i + 1, j),
                      vid(i + 1, j + 1), vid(i, j + 1)):
                if v not in cs:
                    cs.append(v)
            if len(cs) < 3:
                continue
            gs = [gof[c] for c in cs]
            if all(g < 0 for g in gs):
                faces.append(cs)
                continue
            if not any(g < 0 for g in gs):
                continue
            poly = []
            n = len(cs)
            for k in range(n):
                if gs[k] < 0:
                    poly.append(cs[k])
                if (gs[k] < 0) != (gs[(k + 1) % n] < 0):
                    poly.append(crossing(cs[k], cs[(k + 1) % n]))
            if len(poly) >= 3:
                faces.append(poly)
    # orient outward
    Vo = np.asarray(verts_o)
    vol = 0.0
    for f in faces[:200]:
        P = Vo[f] - C
        for k in range(1, len(f) - 1):
            vol += float(np.dot(P[0], np.cross(P[k], P[k + 1])))
    if vol < 0:
        faces = [list(reversed(f)) for f in faces]
    return _thicken([tuple(v) for v in verts_o],
                    [tuple(v) for v in verts_i], faces)


def _plane_marching_shell(R, t, Rmax, Rrim, field, step, bowl):
    """Marching-squares shell for plane-defined patterns, meshed on
    a uniform Cartesian grid in the plane so features are equally
    resolved everywhere (a polar grid starves small features far
    from the centre).  `field(x, y)` is the signed pattern field
    (material < 0); the solid band r in [Rmax, Rrim] and, for
    bowls, the open trim at Rrim are folded into the field so the
    boundaries come out as smooth level sets."""
    A = Rrim if bowl else Rmax
    n = max(8, int(math.ceil(2.0 * A / step)))
    n = min(n, 360)
    xs = np.linspace(-A, A, n + 1)
    if not bowl:                      # solid skirt out to the cap
        ext = []
        v = A
        while v < 8.0 * Rmax:
            v *= 1.6
            ext.append(min(v, 8.0 * Rmax))
        xs = np.concatenate([-np.array(ext[::-1]), xs,
                             np.array(ext)])
    ys = xs.copy()
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    RP = np.hypot(X, Y)
    G = np.minimum(field(X, Y), Rmax - RP)
    if bowl:
        G = np.maximum(G, RP - Rrim)
    G[G == 0.0] = 1e-12

    verts_o = []
    verts_i = []
    C = np.array((0.0, 0.0, R))

    def emit_xy(x, y):
        r = math.hypot(x, y)
        th = 2.0 * math.atan2(2.0 * R, r) if r > 1e-12 else pi
        ph = math.atan2(y, x)
        d = np.array((math.sin(th) * math.cos(ph),
                      math.sin(th) * math.sin(ph), math.cos(th)))
        verts_o.append(tuple(C + (R + 0.5 * t) * d))
        verts_i.append(tuple(C + (R - 0.5 * t) * d))
        return len(verts_o) - 1

    vid = {}

    def gv(i, j):
        if (i, j) not in vid:
            vid[(i, j)] = emit_xy(xs[i], ys[j])
        return vid[(i, j)]

    xing = {}

    def crossing(a, b):
        key = (a, b) if a < b else (b, a)
        if key in xing:
            return xing[key]
        (ia, ja), (ib, jb) = a, b
        ga, gb = G[ia, ja], G[ib, jb]
        u = ga / (ga - gb)
        idx = emit_xy(xs[ia] + u * (xs[ib] - xs[ia]),
                      ys[ja] + u * (ys[jb] - ys[ja]))
        xing[key] = idx
        return idx

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            cs = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            gs = [G[c] for c in cs]
            if all(g < 0 for g in gs):
                faces.append([gv(*c) for c in cs])
                continue
            if not any(g < 0 for g in gs):
                continue
            poly = []
            for k in range(4):
                if gs[k] < 0:
                    poly.append(gv(*cs[k]))
                if (gs[k] < 0) != (gs[(k + 1) % 4] < 0):
                    poly.append(crossing(cs[k], cs[(k + 1) % 4]))
            if len(poly) >= 3:
                faces.append(poly)
    if not bowl:
        # close the far square boundary with a fan to the pole
        pole = emit_xy(0.0, 0.0)
        verts_o[pole] = (0.0, 0.0, 2.0 * R + 0.5 * t)
        verts_i[pole] = (0.0, 0.0, 2.0 * R - 0.5 * t)
        ring = ([gv(i, 0) for i in range(nx)]
                + [gv(nx - 1, j) for j in range(1, ny)]
                + [gv(i, ny - 1) for i in range(nx - 2, -1, -1)]
                + [gv(0, j) for j in range(ny - 2, 0, -1)])
        for k in range(len(ring)):
            faces.append([pole, ring[k],
                          ring[(k + 1) % len(ring)]])
    Vo = np.asarray(verts_o)
    vol = 0.0
    for f in faces[:200]:
        P = Vo[f] - C
        for k in range(1, len(f) - 1):
            vol += float(np.dot(P[0], np.cross(P[k], P[k + 1])))
    if vol < 0:
        faces = [list(reversed(f)) for f in faces]
    return _thicken(verts_o, verts_i, faces)


def _flower_field(x, y, s, petals, width):
    """Signed field of the flower lattice: petal-shaped holes
    (ellipses radiating from hexagonal lattice points); material
    (< 0) is the web between petals."""
    r0 = 0.10 * s
    plen = 0.38 * s
    pw = min(0.9, max(0.1, width)) * plen
    hy = s * sqrt(3.0) / 2.0
    jj = np.round(y / hy)
    emin = np.full(x.shape, np.inf)
    for dj in (-1.0, 0.0, 1.0):
        j = jj + dj
        off = 0.5 * s * np.mod(j, 2.0)
        ii = np.round((x - off) / s)
        for di in (-1.0, 0.0, 1.0):
            cx = (ii + di) * s + off
            cy = j * hy
            ux = x - cx
            uy = y - cy
            for k in range(petals):
                a = 2.0 * pi * k / petals
                ca, sa = math.cos(a), math.sin(a)
                ual = ux * ca + uy * sa - (r0 + 0.5 * plen)
                upe = -ux * sa + uy * ca
                e = ((ual / (0.5 * plen)) ** 2
                     + (upe / (0.5 * pw)) ** 2)
                emin = np.minimum(emin, e)
    return 1.0 - emin


def build_shell(pattern='GRID', radius=1.0, thickness=0.05,
                width=0.35, extent=3.0, spacing=0.6, rings=4,
                rays=8, p=4, q=3, gores=8, res=48, bowl=False,
                petals=8):
    """Return (verts, faces) of the watertight perforated shell.
    'spacing' and 'extent' are in units of the sphere radius;
    'width' is the strip width as a fraction of the pattern cell
    (grid spacing / ring gap / edge length / lune width)."""
    R = radius
    t = min(thickness, R)
    Rmax = extent * R
    if pattern == 'GRID' and not bowl:
        # exact Cartesian construction: division lines on the
        # strip edges, no staircase (bowls go through the field
        # path below, which trims cleanly at the rim circle)
        return _grid_shell(R, t, width, spacing, Rmax, res)
    th_min = 2.0 * math.atan2(2.0 * R, Rmax)
    Rrim = 1.15 * Rmax                # bowl rim band outer radius
    th_top = 2.0 * math.atan2(2.0 * R, Rrim) if bowl else 0.0
    n_cap = max(2, res // 10) if bowl else max(4, res // 6)
    L = max(12, 3 * res)              # longitudes

    if pattern in ('TILING', 'FLOWER') or (pattern == 'GRID'
                                           and bowl):
        # marching-squares field path: smooth clipped boundaries
        phis = 2.0 * pi * np.arange(L) / L
        if pattern == 'TILING':
            if bowl:
                thetas = np.linspace(th_top, pi, res + n_cap + 1)
            else:
                thetas = np.concatenate([
                    np.linspace(0.0, th_min, n_cap + 1)[:-1],
                    np.linspace(th_min, pi, res + 1)])
            p, q = _valid_pq(p, q)
            edges, elen = _tiling_edges(p, q)
            hw = 0.5 * width * elen

            def gfun(TH, PH):
                D = np.stack((np.sin(TH) * np.cos(PH),
                              np.sin(TH) * np.sin(PH),
                              np.cos(TH)), axis=-1).reshape(-1, 3)
                gp = _arc_dist_min(D, edges).reshape(TH.shape) - hw
                return np.minimum(gp, TH - th_min)
        else:
            # plane-defined fields march on a Cartesian grid with
            # a feature-sized step, so the pattern is equally
            # resolved everywhere in the plane
            s = spacing * R
            if pattern == 'GRID':
                ghw = 0.5 * width * s
                half = 0.5 * s - ghw

                def pfield(x, y):
                    dx = np.abs(x - s * np.round(x / s))
                    dy = np.abs(y - s * np.round(y / s))
                    gp = np.minimum(dx, dy) - ghw
                    # rim holes are whole squares or nothing
                    hx = s * (np.floor(x / s) + 0.5)
                    hy = s * (np.floor(y / s) + 0.5)
                    far = np.hypot(np.abs(hx) + half,
                                   np.abs(hy) + half)
                    return np.where((gp > 0) & (far >= Rmax),
                                    -ghw, gp)
                step = min(2.0 * Rmax / res, s / 12.0)
            else:                     # FLOWER

                def pfield(x, y):
                    return _flower_field(x, y, s, petals, width)
                step = min(2.0 * Rmax / res, s / 16.0)
            return _plane_marching_shell(R, t, Rmax, Rrim, pfield,
                                         step, bowl)
        return _field_shell(R, t, thetas, phis, gfun,
                            has_north=not bowl)

    # longitude divisions: aligned to the pattern's angular edges
    # (ray / lune boundaries) so those edges are exact
    pedges = []
    if pattern == 'POLAR':
        nr = max(1, rays)
        ha = 0.5 * width * (2.0 * pi / nr)
        for k in range(nr):
            pedges += [(2.0 * pi * k / nr - ha) % (2.0 * pi),
                       (2.0 * pi * k / nr + ha) % (2.0 * pi)]
    elif pattern == 'BEACHBALL':
        wc = np.clip(width, 0.05, 0.95)
        for k in range(gores):
            pedges += [2.0 * pi * k / gores,
                       2.0 * pi * (k + wc) / gores]
    phis = np.array(_fill_divisions(sorted(set(pedges)), 0.0,
                                    2.0 * pi, 2.0 * pi / L)[:-1])
    L = len(phis)
    phic = np.empty(L)
    phic[:-1] = 0.5 * (phis[:-1] + phis[1:])
    phic[-1] = 0.5 * (phis[-1] + 2.0 * pi + phis[0])

    # ring latitudes: cap / rim-band rows, then pattern rows
    # (uniform in the plane radius r for POLAR -- aligned to its
    # ring edges); ends at the south pole th = pi
    th_cap = np.linspace(th_top, th_min, n_cap + 1)
    if pattern == 'POLAR':
        dr = Rmax / (rings + 1)
        hw = 0.5 * width * dr
        redges = [2.0 * hw]
        for k in range(1, rings + 2):
            redges += [k * dr - hw, k * dr + hw]
        rs = np.array(_fill_divisions(redges, 0.0, Rmax,
                                      Rmax / res)[::-1][1:])
        th_pat = 2.0 * np.arctan2(2.0 * R, rs)
    else:                             # BEACHBALL
        th_pat = np.linspace(th_min, pi, res + 1)[1:]
    thetas = np.concatenate([th_cap, th_pat])
    F = len(thetas) - 1               # face rows

    # face-centre coordinates and their plane projection
    thc = 0.5 * (thetas[:-1] + thetas[1:])
    TH, PH = np.meshgrid(thc, phic, indexing='ij')
    rp = 2.0 * R / np.tan(0.5 * TH)

    if pattern == 'POLAR':
        # concentric rings every dr (k = 0 doubles as a hub disc,
        # keeping the converging rays joined) plus radial rays
        dr = Rmax / (rings + 1)
        hw = 0.5 * width * dr
        nr = max(1, rays)
        ha = 0.5 * width * (2.0 * pi / nr)
        u = PH * nr / (2.0 * pi)
        angd = np.abs(u - np.round(u)) * (2.0 * pi / nr)
        M = ((np.abs(rp - dr * np.round(rp / dr)) < hw) |
             (angd < ha) | (rp < 2.0 * hw))
    else:                             # BEACHBALL
        # n solid lunes, each 'width' of its 2 pi / n slot; a small
        # solid cap at the south pole keeps the gore tips joined
        u = (PH * gores / (2.0 * pi)) % 1.0
        M = u < np.clip(width, 0.05, 0.95)
        M |= TH > 0.9 * pi

    M[:n_cap] = True                  # solid cap / rim band
    if M[-1].any() and not M[-1].all():
        M[-1, :] = True               # pole fans must be uniform

    # fill "checkerboard" vertices (material only on one diagonal)
    # -- they would put four side walls on one vertical edge
    for _ in range(64):
        Nr, Sr = M[:-1], M[1:]        # views: writes reach M
        NW = np.roll(Nr, 1, axis=1)
        SW = np.roll(Sr, 1, axis=1)
        bad_a = NW & Sr & ~Nr & ~SW
        bad_b = Nr & SW & ~NW & ~Sr
        if not (bad_a.any() or bad_b.any()):
            break
        Nr[bad_a] = True
        Sr[bad_b] = True

    # grid vertex ids: (north pole unless bowl), rings, south pole
    has_north = not bowl
    lo = 1 if has_north else 0
    Vg = (2 if has_north else 1) + (F - lo) * L

    def vid(i, j):
        if has_north and i == 0:
            return 0
        if i == F:
            return Vg - 1
        return (1 if has_north else 0) + (i - lo) * L + (j % L)

    verts = np.empty((2 * Vg, 3))     # outer 0..Vg-1, inner rest

    def setv(idx, th, ph):
        d = np.array((math.sin(th) * math.cos(ph),
                      math.sin(th) * math.sin(ph), math.cos(th)))
        c = np.array((0.0, 0.0, R))
        verts[idx] = c + (R + 0.5 * t) * d
        verts[idx + Vg] = c + (R - 0.5 * t) * d

    if has_north:
        setv(0, thetas[0], 0.0)
    setv(Vg - 1, thetas[F], 0.0)
    for i in range(lo, F):
        for j in range(L):
            setv(vid(i, j), thetas[i], phis[j])

    # outer faces of the kept region + boundary edge census
    faces = []
    ecnt = {}                         # (lo,hi) -> directed rep or
    for i in range(F):                # None once seen twice
        for j in range(L):
            if not M[i, j]:
                continue
            f = []                    # pole rows collapse to tris
            for v in (vid(i, j), vid(i + 1, j),
                      vid(i + 1, j + 1), vid(i, j + 1)):
                if v not in f:
                    f.append(v)
            faces.append(f)
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                key = (a, b) if a < b else (b, a)
                ecnt[key] = None if key in ecnt else (a, b)
    # inner surface (reversed) + side walls on boundary edges
    faces += [[v + Vg for v in reversed(f)] for f in faces]
    faces += [[e[1], e[0], e[0] + Vg, e[1] + Vg]
              for e in ecnt.values() if e is not None]
    return [tuple(v) for v in verts], faces
