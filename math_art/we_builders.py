
# Weierstrass-Enneper / Bjorling integration engine for the Math Art
# minimal-surface catalog.
#
# Numpy-only (no bpy): the generic machinery that turns a small data
# "spec" (Gauss map g, height differential dh, a domain, a couple of
# parameter hooks) into a surface builder matching the toolkit's
# PARAMETRIC / MESH_PARAM contracts.  The data tables themselves live
# in minimal_surface_zoo.py; the meshing pipeline, elliptic-function
# engine and operators stay in minimal_surface_toolkit.py.
#
#   X = Re [ e^{i theta} * Int (phi1, phi2, phi3) dz ],
#   phi1 = (1/2)(1/g - g) h,  phi2 = (i/2)(1/g + g) h,  phi3 = h,
#   where dh = h(z) dz  (Weierstrass 1866, Enneper 1864).
#
# Bjorling's problem (Schwarz's formula, 1890):
#   X(w) = Re [ e^{i theta} ( c(w) - i Int_{w0}^{w} n x c' dw~ ) ]
# solved numerically by evaluating the (analytic) curve and normal at
# complex arguments and integrating column-wise from the real axis.
#
# References:
#   K. Weierstrass (1866), A. Enneper (1864); H. A. Schwarz,
#   Gesammelte Mathematische Abhandlungen (1890) for the Bjorling
#   formula; U. Dierkes, S. Hildebrandt, F. Sauvigny, "Minimal
#   Surfaces" (2010); M. Weber, "Classical Minimal Surfaces in
#   Euclidean Space by Examples" and https://minimalsurfaces.blog/
#   for the modern data-driven presentation this engine follows.

import math
import numpy as np

TAU = 2.0 * math.pi


# --------------------------------------------------------------------------
# Toolkit access (lazy -- resolves the toolkit -> zoo -> we_builders
# circular import; by the time any builder here runs, the toolkit module
# object is fully initialized)
# --------------------------------------------------------------------------

def _toolkit():
    import sys
    for name in ('math_art.minimal_surface_toolkit',
                 'minimal_surface_toolkit'):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, '_center_fit'):
            return m
    m = sys.modules.get('__main__')
    if (m is not None and hasattr(m, '_center_fit')
            and hasattr(m, '_Lattice')):
        return m                       # toolkit run as a script
    try:
        from . import minimal_surface_toolkit as tk
    except ImportError:
        import minimal_surface_toolkit as tk
    return tk


def _ev(x, p):
    """Evaluate a spec field: plain value or callable(p)."""
    return x(p) if callable(x) else x


# --------------------------------------------------------------------------
# Shared numeric helpers
# --------------------------------------------------------------------------

def period_integral(f, center, rx, ry, n=4096):
    """Contour integral of f around the ellipse center + rx cos t
    + i ry sin t (midpoint rule -- spectrally accurate for analytic f)."""
    t = (np.arange(n) + 0.5) * (TAU / n)
    z = center + rx * np.cos(t) + 1j * ry * np.sin(t)
    dz = (-rx * np.sin(t) + 1j * ry * np.cos(t)) * (TAU / n)
    return np.sum(f(z) * dz)


_SOLVE_CACHE = {}


def solve_scalar(residual, lo, hi, tol=1e-13, iters=200, key=None):
    """1-parameter period solver: bisection root of `residual` on
    [lo, hi] (residual must change sign). Cached by `key`."""
    if key is not None and key in _SOLVE_CACHE:
        return _SOLVE_CACHE[key]
    flo, fhi = residual(lo), residual(hi)
    if flo == 0.0:
        x = lo
    elif fhi == 0.0:
        x = hi
    else:
        if flo * fhi > 0:
            raise ValueError("solve_scalar: no sign change on bracket")
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            fm = residual(mid)
            if fm == 0.0 or (hi - lo) < tol:
                break
            if flo * fm < 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        x = 0.5 * (lo + hi)
    if key is not None:
        _SOLVE_CACHE[key] = x
    return x


def _phi_fn(spec, p, theta):
    """Callable z -> (nu, nv, 3) complex phi-stack from the spec's
    Weierstrass data ('phi' triple, or 'g' + 'dh')."""
    rot = np.exp(1j * theta)
    if 'phi' in spec:
        fn = spec['phi']

        def phi(z):
            f1, f2, f3 = fn(z, p)
            return np.stack(np.broadcast_arrays(f1, f2, f3),
                            axis=-1) * rot
        return phi
    g, dh = spec['g'], spec['dh']

    def phi(z):
        gv = g(z, p)
        hv = dh(z, p)
        f1 = 0.5 * (1.0 / gv - gv) * hv
        f2 = 0.5j * (1.0 / gv + gv) * hv
        return np.stack(np.broadcast_arrays(f1, f2, hv), axis=-1) * rot
    return phi


# --------------------------------------------------------------------------
# Domain integrators
# --------------------------------------------------------------------------

def _we_disk(spec, p, nu, nv, theta):
    """Radial-ray integration on a disk/annulus domain
    ('disk', r_in, r_out).  Rays get a half-step angular offset so none
    lands exactly on a puncture (a la the Jorge-Meeks k-noid); an
    annulus (r_in > 0) is stitched with a base-ring arc integral, which
    closes because the engine's rows have vanishing real periods."""
    d = spec['domain']
    r0 = max(float(_ev(d[1], p)), 0.0)
    r1 = float(_ev(d[2], p))
    dth = TAU / nv
    off = 0.5 * dth if spec.get('offset_rays', True) else 0.0
    v = off + np.arange(nv) * dth
    ra = max(r0, 1e-3)
    # radial node distribution.  Enneper-type ends grow like a power of
    # the radius, so a linear-in-r grid starves the fast-growing rim with
    # a few huge facets; 'radial_grade' clusters nodes toward the
    # end(s) instead ('rim' -> toward r_out, 'both' -> a cosine/Chebyshev
    # grid dense at r_in and r_out for two-ended annuli).
    grade = spec.get('radial_grade')
    s = np.linspace(0.0, 1.0, nu)
    if grade == 'rim':
        s = 1.0 - (1.0 - s) ** 2
    elif grade == 'both':
        s = 0.5 - 0.5 * np.cos(math.pi * s)
    u = ra + (r1 - ra) * s
    R, TH = np.meshgrid(u, v, indexing='ij')
    z = R * np.exp(1j * TH)
    if 'Xexact' in spec:
        # closed-form immersion (no radial quadrature): the antiderivative
        # is known analytically, so evaluate it straight on the grid --
        # avoids the near-pole capping/folding that tears wing ends.
        with np.errstate(divide='ignore', invalid='ignore'):
            xx, yy, zz = spec['Xexact'](z, p, theta)
        X = np.stack(np.broadcast_arrays(xx, yy, zz), axis=-1).astype(float)
        mask = np.isfinite(X).all(axis=-1)
        punct = _ev(spec.get('mask_punctures'), p) if spec.get(
            'mask_punctures') else None
        if punct:
            for zc, rho in punct:
                mask &= np.abs(z - zc) > rho
            if spec.get('clip_punctures'):
                # marching-squares-style boundary CLIP: instead of dropping
                # whole grid quads that straddle a puncture circle (which
                # leaves a one-quad staircase along every wing rim), pull the
                # first ring of just-inside grid vertices radially out onto
                # the circle |z - zc| = rho and re-evaluate the immersion
                # there, so the cut lands exactly on the mask boundary and
                # the wing edge reads as a clean smooth curve.
                zc_a = np.array([zc for zc, _ in punct])
                rho_a = np.array([rho for _, rho in punct])
                keep = mask
                nb = np.zeros_like(keep)
                nb[:-1] |= keep[1:]
                nb[1:] |= keep[:-1]
                nb |= np.roll(keep, 1, axis=1)
                nb |= np.roll(keep, -1, axis=1)
                ring = (~keep) & nb          # inside a puncture, but adjacent
                ii, jj = np.nonzero(ring)    # to a kept vertex -> snap it out
                if len(ii):
                    zr = z[ii, jj]
                    k = np.abs(zr[:, None] - zc_a[None, :]).argmin(axis=1)
                    d = zr - zc_a[k]
                    znew = zc_a[k] + rho_a[k] * d / np.abs(d)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        nx, ny, nz = spec['Xexact'](znew, p, theta)
                    good = np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
                    gi, gj = ii[good], jj[good]
                    X[gi, gj, 0] = nx[good]
                    X[gi, gj, 1] = ny[good]
                    X[gi, gj, 2] = nz[good]
                    mask = mask.copy()
                    mask[gi, gj] = True
        return X[..., 0], X[..., 1], X[..., 2], False, True, mask
    phi = _phi_fn(spec, p, theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = phi(z)
    ez = np.exp(1j * TH)[..., None]
    Xr = np.real(F * ez)                       # dz = e^{i th} dr
    Xr = np.where(np.isfinite(Xr), Xr, 0.0)
    # loose cap: kill only true numerical garbage from rays that graze
    # a pole, without flattening tall catenoid/planar ends
    cap = 400.0 * float(np.median(np.abs(Xr))) or 1.0
    Xr = np.clip(Xr, -cap, cap)
    dr = np.diff(R, axis=0)[..., None]
    X = np.concatenate(
        [np.zeros((1, nv, 3)),
         np.cumsum(0.5 * (Xr[1:] + Xr[:-1]) * dr, axis=0)], axis=0)
    if u[0] > 2e-3:
        # annulus: connect the ray base points along the inner circle
        zi = u[0] * np.exp(1j * v)
        with np.errstate(divide='ignore', invalid='ignore'):
            Fi = phi(zi)
        arc = np.real(Fi * (1j * zi)[:, None])   # dz = i z dth
        arc = np.where(np.isfinite(arc), arc, 0.0)
        A = np.zeros((nv, 3))
        A[1:] = np.cumsum(0.5 * (arc[1:] + arc[:-1]) * dth, axis=0)
        X = X + A[None, :, :]
    mask = None
    punct = spec.get('mask_punctures')
    if punct:
        valid = np.ones(z.shape, dtype=bool)
        for zc, rho in _ev(punct, p):
            valid &= np.abs(z - zc) > rho
        mask = valid
    clip = spec.get('clip', True)
    tail = mask if mask is not None else clip
    return X[..., 0], X[..., 1], X[..., 2], False, True, tail


def _we_rect(spec, p, nu, nv, theta):
    """2-D cumulative-trapezoid integration on a rectangle
    ('rect', u0, u1, v0, v1): base row along v0, columns upward."""
    d = spec['domain']
    u0, u1 = float(_ev(d[1], p)), float(_ev(d[2], p))
    v0, v1 = float(_ev(d[3], p)), float(_ev(d[4], p))
    u = np.linspace(u0, u1, nu)
    v = np.linspace(v0, v1, nv)
    U, V = np.meshgrid(u, v, indexing='ij')
    z = U + 1j * V
    phi = _phi_fn(spec, p, theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = phi(z)
    F = np.where(np.isfinite(F), F, 0.0)
    du = (u1 - u0) / max(nu - 1, 1)
    dv = (v1 - v0) / max(nv - 1, 1)
    base = np.zeros((nu, 3), dtype=complex)
    base[1:] = np.cumsum(0.5 * (F[1:, 0, :] + F[:-1, 0, :]) * du, axis=0)
    col = np.zeros((nu, nv, 3), dtype=complex)
    col[:, 1:, :] = np.cumsum(
        0.5 * (F[:, 1:, :] + F[:, :-1, :]) * (1j * dv), axis=1)
    X = np.real(base[:, None, :] + col)
    return (X[..., 0], X[..., 1], X[..., 2], False, False,
            spec.get('clip', False))


def _we_torus(spec, p, nu, nv, theta):
    """Closed-form antiderivative on a torus C / <2w1, 2w1 tau>
    ('torus', w1, tau): spec['X'](z, p, L) -> (x, y, z) real arrays.
    Punctures are removed with the toolkit's toroidal mask.  A spec
    may unwrap one direction ('torus_wrap') and span several copies
    ('copies') -- used by singly periodic surfaces whose deck
    translation is a genuine space translation."""
    tk = _toolkit()
    d = spec['domain']
    w1 = float(_ev(d[1], p))
    tau = _ev(d[2], p)
    L = tk._Lattice(w1, tau) if (w1, tau) != (0.5, 1j) else tk._SQUARE
    wrap_u, wrap_v = spec.get('torus_wrap', (True, True))
    m = int(_ev(spec.get('copies', 1), p))
    if wrap_u and wrap_v and m == 1:
        U, V = tk._torus_grid(nu, nv)
    else:
        u = (np.linspace(0.0, m, nu, endpoint=not wrap_u) if not wrap_u
             else np.linspace(0.0, m, nu, endpoint=False))
        v = np.linspace(0.0, 1.0, nv, endpoint=not wrap_v) if not wrap_v \
            else np.linspace(0.0, 1.0, nv, endpoint=False)
        U, V = np.meshgrid(u, v, indexing='ij')
    z = 2.0 * w1 * (U + tau * V)
    with np.errstate(divide='ignore', invalid='ignore'):
        x, y, zc = spec['X'](z, p, L)
    mask = tk._puncture_mask(U % 1.0, V % 1.0, _ev(spec['punctures'], p))
    return x, y, zc, wrap_u, wrap_v, mask


# --- half-plane patch + dihedral tiling (Costa-Hoffman-Meeks family) ------

def halfplane_patch(phi, seed, rot, n_in, n_out, nv, r_out=12.0,
                    grade=None):
    """Integrate a phi-stack over the closed upper half plane -> one
    fundamental patch (nr, nv, 3).  Grid: Chebyshev in theta (clustered
    at the ends theta = 0, pi), graded in r (clustered at the branch
    point r = 0 and the ends r = 1).  `seed(z0)` supplies the analytic
    integral of phi over the first spine cell 0 -> i r1 (the integrand
    is singular at the branch point).  `rot` is the dihedral order the
    patch will be tiled to (drives the default r-grading)."""
    nv = nv if nv % 2 else nv + 1              # odd: a column hits pi/2
    j = np.arange(nv)
    th = (math.pi / 2) * (1 - np.cos(math.pi * j / (nv - 1)))
    s = np.linspace(0.0, 1.0, n_in + 1)[1:]
    gexp = grade if grade is not None else 0.5 * (rot)
    r_in = ((1 - np.cos(math.pi * s)) / 2) ** gexp
    r_o = np.exp(np.linspace(0.0, math.log(r_out), n_out + 1))[1:]
    r = np.concatenate([[0.0], r_in, r_o])
    nr = len(r)
    R, TH = np.meshgrid(r, th, indexing='ij')
    Z = R * np.exp(1j * TH)
    Z[0, :] = 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        F = phi(Z)                             # (nr, nv, 3) complex
    jm = (nv - 1) // 2                         # spine column theta = pi/2
    X = np.zeros((nr, nv, 3), dtype=complex)
    z0 = 1j * r[1]
    sd = np.asarray(seed(z0), dtype=complex)
    # spine: cumulative along r at column jm (dz = i dr)
    dr = np.diff(r)[:, None]
    sp_inc = 0.5 * (F[1:, jm, :] + F[:-1, jm, :]) * (1j * dr)
    spine = np.concatenate([[np.zeros(3)], [sd],
                            sd + np.cumsum(sp_inc[1:], axis=0)], axis=0)
    X[:, jm, :] = spine
    # arcs: cumulative along theta per row (dz = i z dtheta), from jm out
    gfac = F * (1j * Z)[..., None]
    dth = np.diff(th)
    trap = np.zeros((nr, nv, 3), dtype=complex)
    trap[:, 1:, :] = 0.5 * (gfac[:, 1:, :] + gfac[:, :-1, :]) \
        * dth[None, :, None]
    # zero non-finite increments (masked end nodes theta = 0, pi at
    # z = +-1) so one bad cell cannot poison a whole row's cumsum
    trap = np.where(np.isfinite(trap), trap, 0.0)
    C = np.cumsum(trap, axis=1)
    X = X[:, jm, :][:, None, :] + (C - C[:, jm, :][:, None, :])
    return Z, np.real(X)


def tile_dihedral(Z, Xr, rot, punctures, radius, scale,
                  circularize=True):
    """Snap the patch's symmetry seams, tile into 2*rot rigid copies
    (rot rotations x mirror), weld, trim the ends, smooth the rims,
    center and fit the 2 m cube.  Returns (V, quads, uv) where uv is
    the per-vertex fundamental-domain parametrization (repeated per
    copy -- a tiled surface has no global conformal chart).
    Generalizes the Costa-Hoffman-Meeks assembly; assumes the standard
    hyperelliptic cut structure along [0, 1] and (-inf, -1]."""
    tk = _toolkit()
    nr, nvp, _ = Xr.shape
    Rabs = np.abs(Z)
    ang = -math.pi / rot
    uvec = np.array([math.cos(ang), math.sin(ang)])
    # snap the two boundary columns onto their symmetry planes
    for jcol, is0 in ((0, True), (nvp - 1, False)):
        xy = Xr[:, jcol, :2]
        bank = (Rabs[:, jcol] < 1.0) if is0 else (Rabs[:, jcol] > 1.0)
        proj = (xy @ uvec)[:, None] * uvec[None, :]
        flat = np.stack([xy[:, 0], np.zeros_like(xy[:, 1])], axis=-1)
        Xr[:, jcol, :2] = np.where(bank[:, None], proj, flat)
    Xr[0, :, :] = 0.0                          # center vertex
    valid = np.ones(Z.shape, dtype=bool)
    for zc, rho in punctures:
        valid &= np.abs(Z - zc) > rho
    valid[0, :] = True
    Xr = np.where(np.isfinite(Xr), Xr, 0.0)
    Xr[~valid] = 0.0
    V0 = Xr.reshape(-1, 3)
    iu, ju = np.meshgrid(np.arange(nr), np.arange(nvp), indexing='ij')
    uv0 = np.stack([iu / max(nr - 1, 1), ju / max(nvp - 1, 1)],
                   axis=-1).reshape(-1, 2)
    vv = valid.reshape(-1)
    ii, jj = np.meshgrid(np.arange(nr - 1), np.arange(nvp - 1),
                         indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    q0 = np.stack([ii * nvp + jj, ii * nvp + jj + 1,
                   (ii + 1) * nvp + jj + 1, (ii + 1) * nvp + jj], axis=1)
    q0 = q0[np.all(vv[q0], axis=1)]
    # tile: rot rotations, each with its y-mirror -> 2*rot copies
    M = np.diag([1.0, -1.0, 1.0])
    Vparts, Fparts, base = [], [], 0
    for jrot in range(rot):
        a = TAU * jrot / rot
        Rj = np.array([[math.cos(a), -math.sin(a), 0.0],
                       [math.sin(a), math.cos(a), 0.0],
                       [0.0, 0.0, 1.0]])
        for mir in (False, True):
            T = Rj @ (M if mir else np.eye(3))
            Vparts.append(V0 @ T.T)
            qf = (q0[:, ::-1] if mir else q0) + base
            Fparts.append(qf)
            base += len(V0)
    V = np.concatenate(Vparts, axis=0)
    uvcat = np.concatenate([uv0] * (2 * rot), axis=0)
    faces = np.concatenate(Fparts, axis=0)
    # weld (quantize + unique); tight tolerance -- the seam snap already
    # makes shared vertices coincide to machine epsilon
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    keyq = np.round(V / (1e-7 * max(diag, 1.0))).astype(np.int64)
    _, inv = np.unique(keyq, axis=0, return_inverse=True)
    inv = inv.ravel()
    Vw = np.zeros((int(inv.max()) + 1, 3))
    Vw[inv] = V
    uvw = np.zeros((len(Vw), 2))
    uvw[inv] = uvcat
    faces = inv[faces]
    # collapse welded seam quads to triangles; drop true degenerates
    flist = []
    for f in faces:
        g = [int(f[0])]
        for t in range(1, 4):
            if int(f[t]) != g[-1]:
                g.append(int(f[t]))
        if len(g) >= 3 and g[0] != g[-1] and len(set(g)) == len(g):
            flist.append(tuple(g))
    # object-space radius clip to trim the (infinite) ends
    cen = np.median(Vw, axis=0)
    rad = np.linalg.norm(Vw - cen, axis=1)
    thr = float(np.percentile(rad, 93.0))
    flist = [f for f in flist if all(rad[i] <= thr for i in f)]
    used = np.unique(np.array([i for f in flist for i in f],
                              dtype=np.int64))
    remap = np.full(len(Vw), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    Vf = Vw[used]
    uvf = uvw[used]
    quads = [tuple(int(remap[i]) for i in f) for f in flist]
    # keep the main body only (the clip can shear off small islands);
    # ride uv along by stacking it into the vertex array
    Vu, quads = tk._largest_component(np.hstack([Vf, uvf]), quads)
    Vf, uvf = Vu[:, :3], Vu[:, 3:]
    Vf = tk._smooth_boundary(Vf, quads)
    if circularize:
        Vf = tk._circularize_outer(Vf, quads)
    Vf = tk._center_fit(Vf, scale, Vf)
    return Vf, quads, uvf


def _we_halfplane(spec, p, nu, nv, radius, scale, theta):
    rot = int(_ev(spec['symmetry']['rot'], p))
    phi = _phi_fn(spec, p, theta)
    n_in = max(70, int(1.4 * nu))
    n_out = max(30, int(0.6 * nu))
    r_out = float(_ev(spec['domain'][1], p)) if len(spec['domain']) > 1 \
        else 12.0
    grade = _ev(spec['r_grade'], p) if 'r_grade' in spec else None
    Z, Xr = halfplane_patch(phi, lambda z0: spec['seed'](p, z0), rot,
                            n_in, n_out, nv, r_out=r_out, grade=grade)
    punct = _ev(spec['hp_punctures'], p) if 'hp_punctures' in spec else []
    punct = [(zc, rho / max(radius / 1.2, 0.4)) for zc, rho in punct]
    return tile_dihedral(Z, Xr, rot, punct, radius, scale,
                         circularize=spec.get('circularize', True))


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def we_surface(spec, nu, nv, order, radius, scale=None, theta=0.0):
    """Build a surface from a Weierstrass spec.  Grid domains return
    the PARAMETRIC tuple (x, y, z, wrap_u, wrap_v, clip|mask); the
    halfplane domain returns a finished (V, quads, uv) mesh."""
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    if 'solve' in spec:
        p = spec['solve'](p) or p
    kind = spec['domain'][0]
    if theta and kind in ('torus', 'halfplane'):
        raise ValueError("associate family needs a simply connected "
                         "domain (disk/rect/Bjorling)")
    if kind == 'disk':
        return _we_disk(spec, p, nu, nv, theta)
    if kind == 'rect':
        return _we_rect(spec, p, nu, nv, theta)
    if kind == 'torus':
        return _we_torus(spec, p, nu, nv, theta)
    if kind == 'halfplane':
        return _we_halfplane(spec, p, nu, nv, radius, scale, theta)
    raise ValueError(f"unknown domain {kind!r}")


def _bj_curve_fns(spec, p):
    c = spec['curve']
    h = 1e-5

    def c1(w):
        return tuple((a - b) / (2 * h) for a, b in
                     zip(c(w + h, p), c(w - h, p)))

    def c2(w):
        cp, c0, cm = c(w + h, p), c(w, p), c(w - h, p)
        return tuple((a - 2 * b + d) / (h * h) for a, b, d in
                     zip(cp, c0, cm))
    return (lambda w: c(w, p)), c1, c2


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def bjorling_surface(spec, nu, nv, order, radius, theta=0.0):
    """Schwarz's solution of the Bjorling problem for an analytic seed
    curve + unit normal field: the surface strip containing the curve
    with the prescribed surface normal along it.  Integrates n x c'
    vertically from the real axis per grid column (the strip is simply
    connected, so no period machinery is needed); theta sweeps the
    associate family."""
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    t0, t1 = _ev(spec['t_range'], p)
    closed = spec.get('closed', False)
    vh = _ev(spec.get('v_half', 0.6), p) if not callable(
        spec.get('v_half', 0.6)) else spec['v_half'](p)
    nv = nv if nv % 2 else nv + 1              # middle row on the axis
    wrap_u = bool(closed) and abs(theta) < 1e-12
    u = np.linspace(t0, t1, nu, endpoint=not wrap_u)
    v = np.linspace(-vh, vh, nv)
    U, V = np.meshgrid(u, v, indexing='ij')
    W = U + 1j * V
    cfn, c1fn, c2fn = _bj_curve_fns(spec, p)
    C = cfn(W)
    Cp = c1fn(W)
    nrm = spec.get('normal')
    if nrm is None or nrm == 'frenet':
        # principal normal, analytically continued (principal-branch
        # square roots -- valid for the thin strips we build)
        Cpp = c2fn(W)
        dot_pp = sum(a * a for a in Cp)
        T = tuple(a / np.sqrt(dot_pp) for a in Cp)
        proj = sum(a * b for a, b in zip(Cpp, T))
        Nn = tuple(a - proj * b for a, b in zip(Cpp, T))
        nlen = np.sqrt(sum(a * a for a in Nn))
        N = tuple(a / nlen for a in Nn)
    else:
        N = nrm(W, p)
    Q = np.stack(np.broadcast_arrays(*_cross(N, Cp)), axis=-1)
    Q = np.where(np.isfinite(Q), Q, 0.0)
    # real-axis leg (real on the axis, so it only matters off-axis)
    jm = (nv - 1) // 2
    du = (t1 - t0) / max(len(u) - 1, 1) if not wrap_u \
        else (t1 - t0) / len(u)
    A = np.zeros((len(u), 3), dtype=complex)
    qa = Q[:, jm, :]
    A[1:] = np.cumsum(0.5 * (qa[1:] + qa[:-1]) * du, axis=0)
    # vertical legs from the axis, both directions
    dv = (2.0 * vh) / max(nv - 1, 1)
    G = np.zeros((len(u), nv, 3), dtype=complex)
    up = np.cumsum(0.5 * (Q[:, jm:-1, :] + Q[:, jm + 1:, :]) * dv,
                   axis=1)
    G[:, jm + 1:, :] = up
    dn = np.cumsum(0.5 * (Q[:, jm:0:-1, :] + Q[:, jm - 1::-1, :]) * dv,
                   axis=1)
    G[:, jm - 1::-1, :] = -dn
    Cs = np.stack(np.broadcast_arrays(*C), axis=-1)
    F = Cs - 1j * A[:, None, :] + G            # int n x c' folded in
    X = np.real(np.exp(1j * theta) * F)
    return (X[..., 0], X[..., 1], X[..., 2], wrap_u, False,
            spec.get('clip', False))


# --------------------------------------------------------------------------
# Saddle-tower stacking (singly periodic Scherk / Karcher towers)
# --------------------------------------------------------------------------

def _rot_z(ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _open_boundary(V, quads):
    """Vertex indices on open (once-used) mesh edges, plus the median
    edge length -- used to size the seam-weld tolerance."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for q in quads:
        m = len(q)
        for k in range(m):
            a, b = q[k], q[(k + 1) % m]
            cnt[(a, b) if a < b else (b, a)] += 1
    bset = {v for (a, b), c in cnt.items() if c == 1 for v in (a, b)}
    lens = [float(np.linalg.norm(V[a] - V[b])) for (a, b) in cnt]
    med = float(np.median(lens)) if lens else 1.0
    return np.array(sorted(bset), dtype=np.int64), med


def we_saddle_tower(spec, nu, nv, order, radius, scale, theta, storeys):
    """Stack `storeys` copies of one saddle unit into a genuinely periodic
    tower.

    Each fundamental domain is a single 2n-winged saddle from the exact
    log-sum immersion (_tower_X) on the punctured disk.  Consecutive
    storeys are related by the surface's deck isometry -- a *screw motion*:
    a vertical rise T = pi/(2n) composed with a rotation by pi/n about the
    axis.  The screw (not a pure translation) is the true deck map: the two
    rims of one disk unit are related by a roto-reflection, not a z-shift,
    so no pure translation registers them -- whereas rot(pi/n) permutes the
    2n vertical wing-walls onto themselves, keeping every wall one flat,
    uncrossed vertical half-plane running the whole height (its azimuth is
    independent of z), and puts consecutive storeys in disjoint z-slabs so
    the tower stays embedded.  Storey-s top boundary and storey-(s+1) bottom
    boundary are the same seam curve, so a nearest-neighbour match between
    the two storeys' boundary rings welds the joins watertight."""
    tk = _toolkit()
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    # Karcher unequal-wing tower: the angle knob (theta) is the alpha modulus,
    # not a Bonnet rotation; fold it into the params so the end positions /
    # residues / puncture mask all pick it up, and build a single fundamental
    # domain (the unequal-wing unit has no screw deck isometry to stack -- see
    # the SADDLE_TOWER_A note in minimal_surface_zoo.py).
    if spec.get('alpha_from_theta'):
        p = dict(p, alpha=theta)
        theta = 0.0
        storeys = 1
    n = int(p['n'])
    S = max(1, int(storeys))
    rb = spec.get('res_boost')
    if rb:
        nu = max(3, int(round(nu * rb[0])))
        nv = max(3, int(round(nv * rb[1])))
    nv = max(2 * n, int(round(nv / (2 * n))) * (2 * n))   # whole wings
    # sample one fundamental domain (exact immersion + puncture mask)
    x, y, z, _, _, mask = _we_disk(spec, p, nu, nv, theta)
    Vg = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    vm = np.asarray(mask).reshape(-1)
    gu = np.arange(nu) / max(nu - 1, 1)
    gv = np.arange(nv) / nv
    UVg = np.stack(np.meshgrid(gu, gv, indexing='ij'), axis=-1).reshape(-1, 2)
    N = nu * nv

    def vid(i, j):
        return i * nv + j
    quads0 = []
    for i in range(nu - 1):
        for j in range(nv):                       # wrap in v
            j2 = (j + 1) % nv
            f = (vid(i, j), vid(i + 1, j), vid(i + 1, j2), vid(i, j2))
            if vm[f[0]] and vm[f[1]] and vm[f[2]] and vm[f[3]]:
                quads0.append(f)
    used = (np.unique(np.array(quads0).ravel()) if quads0
            else np.array([], dtype=np.int64))
    remap = np.full(N, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    V0 = Vg[used]
    UV0 = UVg[used]
    quads0 = [tuple(int(remap[i]) for i in q) for q in quads0]
    M = len(V0)

    T = math.pi / (2 * n)                          # one storey's height
    Rm = _rot_z(math.pi / n)

    if S > 1 and M:
        # Stack S copies under the surface's deck isometry -- the screw
        # motion screw(P) = P @ Rm.T + T*zhat (rotate pi/n about the axis,
        # rise T = pi/(2n)) -- and weld the shared seam.
        #
        # Why the screw (and not a pure vertical translation): the two rims
        # of one disk unit are related by a roto-reflection, not a pure
        # z-shift (domain map z -> e^{i pi/n} z acts on the surface as
        # rot * diag(1,1,-1) with ZERO z-shift), so no pure translation
        # tiles the unit -- its rims never register.  The screw is the true
        # deck isometry: rot(pi/n) permutes the 2n vertical wing-walls onto
        # themselves so every wall stays one flat, uncrossed vertical
        # half-plane running the whole height (its azimuth is independent of
        # z), while consecutive storeys occupy disjoint z-slabs
        # [sT - T/2, sT + T/2] and so cannot self-intersect.
        #
        # Seam weld: storey-s TOP boundary and storey-(s+1) BOTTOM boundary
        # are the *same space curve* (top rim = screw(bottom rim)), only
        # sampled at slightly offset grid points, so a nearest-neighbour
        # match between the two storeys' boundary rings welds it watertight.
        # The tolerance is a fraction of the slab height T: the only other
        # inter-storey approach is a full slab away (>= T), so wing side
        # edges -- free, and in disjoint slabs -- are never falsely merged.
        bnd, med = _open_boundary(V0, quads0)
        Vparts, Rp = [], np.eye(3)
        for s in range(S):
            Vparts.append(V0 @ Rp.T + np.array([0.0, 0.0, s * T]))
            Rp = Rp @ Rm.T                        # (Rm^{s+1}).T
        Vcat = np.concatenate(Vparts, axis=0)
        UVcat = np.concatenate([UV0] * S, axis=0)
        parent = np.arange(S * M)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        tol = min(0.45 * T, max(3.0 * med, 0.15 * T))
        for s in range(S - 1):
            A = Vcat[s * M + bnd]                 # storey s boundary ring
            B = Vcat[(s + 1) * M + bnd]           # storey s+1 boundary ring
            d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
            jmin = d.argmin(axis=1)
            dmin = d[np.arange(len(bnd)), jmin]
            for a in np.nonzero(dmin < tol)[0]:
                ra = find(s * M + int(bnd[a]))
                rb = find((s + 1) * M + int(bnd[jmin[a]]))
                if ra != rb:
                    parent[ra] = rb
        roots = np.array([find(a) for a in range(S * M)])
        uniq, inv = np.unique(roots, return_inverse=True)
        nuq = len(uniq)
        Vf = np.zeros((nuq, 3))
        UVf = np.zeros((nuq, 2))
        cnt = np.zeros(nuq)
        np.add.at(Vf, inv, Vcat)
        np.add.at(UVf, inv, UVcat)
        np.add.at(cnt, inv, 1)
        Vf /= cnt[:, None]
        UVf /= cnt[:, None]
        quads = []
        for s in range(S):
            for q in quads0:
                fq = tuple(int(inv[s * M + i]) for i in q)
                if len(set(fq)) >= 3:
                    quads.append(fq)
    else:
        Vf, quads, UVf = V0, quads0, UV0

    # the wing rims are cut exactly on the puncture circles by the
    # marching-squares clip in _we_disk (clip_punctures), so they arrive
    # already clean; a short boundary relaxation just evens out the sampling.
    Vf = tk._smooth_boundary(Vf, quads, iters=6)
    Vf = tk._center_fit(Vf, scale, Vf)
    return Vf, quads, UVf


def make_entry(key, spec):
    """PARAMETRIC/MESH_PARAM builder closure for a WE spec."""
    if spec.get('mesher'):
        # bespoke finished-mesh builder (higher-genus Chen-Gackstatter,
        # periodic Callahan-Hoffman-Meeks): the spec supplies the whole
        # mesher, the engine only wires it in.  A spec that declares a
        # storeys_label is periodic -- its mesher takes the storey count.
        def build(nu, nv, order, radius, scale, theta=0.0, storeys=1):
            if spec.get('storeys_label'):
                return spec['mesher'](spec, nu, nv, order, radius, scale,
                                      theta, storeys)
            return spec['mesher'](spec, nu, nv, order, radius, scale, theta)
        build.finished_mesh = True
    elif spec.get('tower'):
        def build(nu, nv, order, radius, scale, theta=0.0, storeys=1):
            return we_saddle_tower(spec, nu, nv, order, radius, scale,
                                   theta, storeys)
        build.finished_mesh = True
    elif spec['domain'][0] == 'halfplane':
        def build(nu, nv, order, radius, scale, theta=0.0, storeys=1):
            return we_surface(spec, nu, nv, order, radius, scale, theta)
        build.finished_mesh = True
    else:
        if spec.get('associate') and spec['domain'][0] not in \
                ('disk', 'rect'):
            raise ValueError(f"{key}: associate family requires a "
                             "simply connected domain")

        def build(nu, nv, order, radius, theta=0.0):
            return we_surface(spec, nu, nv, order, radius, None, theta)
        build.finished_mesh = False
    build.spec = spec
    build.__name__ = f"we_{key.lower()}"
    return build


def make_bjorling_entry(key, spec):
    def build(nu, nv, order, radius, theta=0.0):
        return bjorling_surface(spec, nu, nv, order, radius, theta)
    build.spec = spec
    build.finished_mesh = False
    build.__name__ = f"bj_{key.lower()}"
    return build


# ==========================================================================
# Schwarz P / Gyroid / Schwarz D -- the exact genus-3 associate (Bonnet)
# family from ONE Weierstrass representation with a single angle theta
# ==========================================================================
# Unlike the nodal marching-tets TPMS (mesh.tpms_add), which have no
# associate parameter, this is the *exact* Enneper-Weierstrass immersion
#     X_theta = Re[ e^{i theta} Int (om1, om2, om3) dz ]
# on the branched sphere.  A single angle theta continuously morphs the
# whole iconic family:  theta = 0  -> Schwarz P,  theta ~ 38.0148 deg ->
# Gyroid,  theta = 90 deg -> Schwarz D.  P and D are conjugate; the gyroid
# is the embedded associate discovered by A. Schoen (1970).
#
# Weierstrass data (Schwarz P and D share ONE algebraic Gauss map; branch
# points {0, +-1, +-3, inf}), as harvested from M. Weber's repository
# (research/msblog_harvest/triply_periodic.json):
#     om1 = -1 / ( sqrt(z) sqrt(z-1) sqrt(z+3) )
#     om2 = -2 / ( sqrt(z+3) sqrt(z+1) sqrt(z-1) sqrt(z-3) )
#     om3 = -i / ( sqrt(z) sqrt(z+1) sqrt(z-3) )
# These satisfy om1^2 + om2^2 + om3^2 = 0 (conformal minimal immersion,
# verified < 1e-16 by the self-test).
#
# Fundamental domain + tiling scheme.  The upper half z-plane (all six
# branch points lie on the real axis) is one fundamental domain -- a
# hyperbolic hexagon.  It is mapped conformally to the unit w-disk by the
# Cayley transform  w = (z-i)/(z+i)  (base point z=i -> w=0), and the WE
# 1-forms are integrated radially over that disk.  Its six rim arcs (the
# images of the six real intervals between consecutive branch points) are the
# boundary edges of the fundamental surface patch, and by the Schwarz
# reflection principle each edge is a symmetry element that continues the
# surface across itself.  WHICH isometry it is depends on the edge geometry,
# and this is read off each edge directly (an SVD of the sampled edge curve),
# so no theta-tracking is needed -- every special angle lives in its own fixed
# symmetry frame:
#   * theta = 0 (Schwarz P): all six edges are PLANAR geodesics lying in the
#     cube's coordinate mirror planes, so the continuation is a plane
#     REFLECTION.  Reflecting the patch across the six coordinate planes is
#     drift-free (the maps are exact and compose to exact translations), and
#     fills one cubic cell watertight -- the classic rounded-cube "P" network.
#   * theta = 90 deg (Schwarz D, P's conjugate): the six edges are STRAIGHT
#     lines, so the continuation is a 180-degree ROTATION about the line; the
#     2-fold reassembly fills the (larger) D cell.
#   * the gyroid (theta ~ 38.0148 deg): the edges are SKEW curves (it is
#     chiral -- no straight line and no mirror plane), so only 2-fold rotations
#     apply.  Its cell over-covers under the six edge-2-folds and is not
#     reassembled watertight here; the exact fundamental piece is built instead
#     (see the pgd_build / _pgd_tile_cell docstrings).
# Every generator's linear part is snapped to the exact 48-element cube group
# O_h so the generated space group closes without composition drift; a
# breadth-first orbit of the patch, welded on the shared symmetry elements,
# fills one cubic cell, and pgd_build arrays that cell on the verified cubic
# period (a ~ 4.31 for P, 4.69 for D), which is independent of theta up to the
# associate family's overall scale.
#
# References:
#   H. A. Schwarz, "Gesammelte Mathematische Abhandlungen" (1890) -- the P
#     and D surfaces and the reflection principle;
#   A. H. Schoen, "Infinite periodic minimal surfaces without
#     self-intersections", NASA TN D-5541 (1970) -- the gyroid as the
#     theta ~ 38.0148-degree associate of P;
#   A. Weyhaupt, "Deformations of the gyroid and lidinoid minimal
#     surfaces", Pacific J. Math. 235 (2008) 137-171;
#   H. Karcher, K. Polthier, "Construction of triply periodic minimal
#     surfaces", Phil. Trans. R. Soc. Lond. A 354 (1996) 2077-2104 -- the
#     reflection-group assembly of the P and D cells;
#   M. Weber, https://minimalsurfaces.blog/ (triply periodic) -- the
#     explicit g/dh data and the "Associate rPD" notebook this follows.

import itertools as _itertools

# branch points ordered by their w-disk rim angle (the vertices of the
# fundamental hexagon); edge i runs between vertex i and vertex i+1
_PGD_ORDER = ('inf', '-3', '-1', '0', '1', '3')
_PGD_ZBR = {'inf': None, '-3': -3.0, '-1': -1.0, '0': 0.0, '1': 1.0,
            '3': 3.0}


def _pgd_wbp(key):
    """w-disk rim position of a branch point (Cayley image of z)."""
    z = _PGD_ZBR[key]
    return 1.0 + 0j if z is None else (z - 1j) / (z + 1j)


def _pgd_angs():
    return {k: math.atan2(_pgd_wbp(k).imag, _pgd_wbp(k).real) % TAU
            for k in _PGD_ORDER}


def _pgd_om(z):
    """The three Weierstrass coordinate 1-forms (om1, om2, om3)."""
    z = np.asarray(z, dtype=complex)

    def s(a):
        return np.sqrt(z - a)
    om1 = -1.0 / (s(0.0) * s(1.0) * s(-3.0))
    om2 = -2.0 / (s(-3.0) * s(-1.0) * s(1.0) * s(3.0))
    om3 = -1j / (s(0.0) * s(-1.0) * s(3.0))
    return om1, om2, om3


def _pgd_z_of_w(w):
    return 1j * (1.0 + w) / (1.0 - w)


def _pgd_dzdw(w):
    return 2j / (1.0 - w) ** 2


def _pgd_forms_w(w, theta):
    """e^{i theta} * (om1, om2, om3) * dz/dw, evaluated on the w-disk."""
    z = _pgd_z_of_w(w)
    o1, o2, o3 = _pgd_om(z)
    J = _pgd_dzdw(w)
    return np.stack([o1 * J, o2 * J, o3 * J], axis=-1) * np.exp(1j * theta)


def _pgd_good_nv(nv):
    """Round nv up to a multiple of 4 so no offset ray lands on the 0/90/
    180/270-degree branch points (a ray through a branch point integrates
    a 1/sqrt singularity and blows the patch up)."""
    nv = int(nv)
    return nv + ((-nv) % 4) if nv % 4 else nv


def pgd_patch(theta, nu=64, nv=160, rmax=1.0, mask_eps=0.05):
    """One fundamental surface patch: the WE immersion integrated radially
    over the Cayley w-disk.  Returns (V (nu*nv, 3), faces, valid-mask,
    w-grid).  Small disks of radius `mask_eps` around the six rim branch
    points are dropped so the (parametrization-)singular hexagon corners
    become small clean holes rather than spikes."""
    nv = _pgd_good_nv(nv)
    s = np.linspace(0.0, 1.0, nu)
    r = rmax * (1.0 - (1.0 - s) ** 1.7)          # graded dense toward rim
    v = (np.arange(nv) + 0.5) * (TAU / nv)       # offset rays
    R, TH = np.meshgrid(r, v, indexing='ij')
    w = R * np.exp(1j * TH)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = _pgd_forms_w(w, theta) * np.exp(1j * TH)[..., None]
    F = np.where(np.isfinite(F), F, 0.0)
    dr = np.diff(R, axis=0)[..., None]
    X = np.concatenate([np.zeros((1, nv, 3)),
                        np.cumsum(0.5 * (F[1:] + F[:-1]) * dr, axis=0)],
                       axis=0)
    X = np.real(X)
    bw = np.array([_pgd_wbp(k) / abs(_pgd_wbp(k)) for k in _PGD_ORDER])
    valid = np.ones(w.shape, dtype=bool)
    for b in bw:
        valid &= np.abs(w - b) > mask_eps
    V = X.reshape(-1, 3)
    vm = valid.reshape(-1)
    faces = []
    for i in range(nu - 1):
        for j in range(nv):
            j2 = (j + 1) % nv
            if i == 0:
                f = (j, nv + j, nv + j2)         # center fan
            else:
                f = (i * nv + j, (i + 1) * nv + j,
                     (i + 1) * nv + j2, i * nv + j2)
            if vm[f[0]] and vm[f[1]] and vm[f[2]] and (len(f) == 3
                                                       or vm[f[3]]):
                faces.append(f)
    return V, faces, valid, w


def _pgd_grid_pts(theta, nu=60, nv=160):
    """Raw immersion sampled on a fixed w-disk grid (no masking) -- a
    parametrization-stable array for the Bonnet-morph continuity check."""
    nv = _pgd_good_nv(nv)
    s = np.linspace(0.0, 1.0, nu)
    r = 0.985 * (1.0 - (1.0 - s) ** 1.7)
    v = (np.arange(nv) + 0.5) * (TAU / nv)
    R, TH = np.meshgrid(r, v, indexing='ij')
    w = R * np.exp(1j * TH)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = _pgd_forms_w(w, theta) * np.exp(1j * TH)[..., None]
    F = np.where(np.isfinite(F), F, 0.0)
    dr = np.diff(R, axis=0)[..., None]
    X = np.concatenate([np.zeros((1, nv, 3)),
                        np.cumsum(0.5 * (F[1:] + F[:-1]) * dr, axis=0)],
                       axis=0)
    return np.real(X)


def _pgd_cube_rotations():
    """The 24 proper rotations of the cube (signed permutation matrices,
    det = +1) -- the point group O shared by the whole P/Gyroid/D family."""
    mats = []
    for perm in _itertools.permutations(range(3)):
        for sg in _itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = sg[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                mats.append(M)
    return np.array(mats)


_PGD_CUBE = _pgd_cube_rotations()


def _pgd_snap_rot(M):
    """Nearest exact cube rotation to M (kills composition drift)."""
    return _PGD_CUBE[int(np.argmin(
        np.abs(_PGD_CUBE - M).reshape(24, -1).max(axis=1)))]


def _pgd_edge_curve(theta, i, m=48, rr=0.9993):
    """Sample boundary edge i of the fundamental hexagon (the rim arc
    between two consecutive branch vertices), avoiding the singular tips."""
    angs = _pgd_angs()
    a, b = _PGD_ORDER[i], _PGD_ORDER[(i + 1) % 6]
    lo = angs[a]
    span = (angs[b] - lo) % TAU
    ts = lo + np.linspace(0.06, 0.94, m) * span
    s = np.linspace(0.0, 1.0, 2600)
    rr_ = rr * (1.0 - (1.0 - s) ** 2.0)
    dr = np.diff(rr_)[:, None]
    pts = []
    for t in ts:
        w = rr_ * np.exp(1j * t)
        with np.errstate(divide='ignore', invalid='ignore'):
            F = _pgd_forms_w(w, theta) * np.exp(1j * t)
        F = np.where(np.isfinite(F), F, 0.0)
        pts.append(np.real(np.sum(0.5 * (F[1:] + F[:-1]) * dr, axis=0)))
    return np.array(pts)


def _pgd_fit_twofold(C):
    """Proper 180-degree rotation R(x) = M x + b mapping the edge curve C to
    its own reversal (the surface's 2-fold rotation about that edge)."""
    Q = C[::-1]

    def resid(d):
        d = d / np.linalg.norm(d)
        M = 2.0 * np.outer(d, d) - np.eye(3)
        b = (Q - C @ M.T).mean(axis=0)
        return (float(np.sqrt(np.mean(np.sum((C @ M.T + b - Q) ** 2, 1)))),
                M, b)
    grid = [(math.sin(t) * math.cos(p), math.sin(t) * math.sin(p),
             math.cos(t))
            for t in np.linspace(0, math.pi, 13)
            for p in np.linspace(0, TAU, 25)]
    d0 = np.array(min(grid, key=lambda g: resid(np.array(g))[0]))
    step = 0.2
    for _ in range(80):
        r0 = resid(d0)[0]
        improved = False
        for e in np.eye(3):
            for sgn in (step, -step):
                dt = d0 + sgn * e
                if resid(dt)[0] < r0:
                    d0 = dt / np.linalg.norm(dt)
                    r0 = resid(dt)[0]
                    improved = True
        if not improved:
            step *= 0.5
        if step < 1e-4:
            break
    r, M, b = resid(d0)
    return M, b, r


def pgd_gluings(theta):
    """The six edge 2-fold rotations (M, b), linear parts snapped to exact
    cube rotations so the generated space group closes without drift.

    At exactly theta = 0 (P) and theta = pi/2 (D) the boundary edges are
    straight lines, so the 2-fold *axis* fit is degenerate (any axis in the
    plane perpendicular to the line reverses the segment) and would snap to
    an arbitrary cube rotation.  The correct rotation is the continuous
    limit, so the axis is fit at an angle nudged just inside the open
    interval (0, pi/2) -- where the edge has curved enough to pin the axis
    -- while the translation is refit at the true theta so the patch and
    its lattice stay exact."""
    tf = min(max(theta, 0.035), 0.5 * math.pi - 0.035)   # axis-fit angle
    gens = []
    for i in range(6):
        Cf = _pgd_edge_curve(tf, i)
        M, _b, _r = _pgd_fit_twofold(Cf)
        Ms = _pgd_snap_rot(M)
        C = _pgd_edge_curve(theta, i)
        b = (C[::-1] - C @ Ms.T).mean(axis=0)    # refit translation at theta
        gens.append((Ms, b))
    return gens


def _pgd_compose(g, h):
    return (g[0] @ h[0], g[0] @ h[1] + g[1])


def pgd_lattice(gens, maxlen=4):
    """Pure-translation lattice vectors of the tiling: words in the six
    generators (up to `maxlen` letters) whose linear part is the identity
    cube rotation.  P and the gyroid expose the cubic period in 2-letter
    words; the conjugate D surface (whose opposite edges compose to
    near-identity) needs longer words.  A real period has norm ~ 4; the
    small (< ~0.5) near-identity words are translation-fit drift and are
    filtered out.  Returned deduplicated, shortest first."""
    allg = []
    for M, b in gens:
        allg.append((M, b))
        allg.append((M.T, -M.T @ b))
    trans = []
    frontier = [(np.eye(3), np.zeros(3))]
    for _ in range(maxlen):
        nxt = []
        for g in frontier:
            for h in allg:
                gh = _pgd_compose(g, h)
                nxt.append(gh)
                if np.abs(gh[0] - np.eye(3)).max() < 1e-6:
                    n = np.linalg.norm(gh[1])
                    if 2.5 < n < 7.0:
                        trans.append(gh[1])
        frontier = nxt
    if not trans:
        return np.zeros((0, 3))
    trans = np.array(trans)
    trans = trans[np.argsort(np.linalg.norm(trans, axis=1))]
    keep = []
    for t in trans:
        if not any(np.linalg.norm(t - k) < 0.2 for k in keep):
            keep.append(t)
    return np.array(keep)


def _pgd_frames(gens, box, cap=600):
    """Breadth-first orbit of the identity frame under the six 2-fold
    generators (+ inverses), bounded to translations within `box` of the
    origin.  Cube-snapped rotations make the dedup key exact."""
    allg = []
    for M, b in gens:
        allg.append((M, b))
        allg.append((M.T, -M.T @ b))
    I = (np.eye(3), np.zeros(3))

    def rid(M):
        return int(np.argmin(np.abs(_PGD_CUBE - M).reshape(24, -1).max(1)))

    def key(g):
        return (rid(g[0]), tuple(np.round(g[1], 1)))
    frames = [I]
    seen = {key(I)}
    queue = [I]
    while queue:
        g = queue.pop()
        for h in allg:
            g2 = _pgd_compose(g, h)
            k = key(g2)
            if k in seen:
                continue
            if np.max(np.abs(g2[1])) > box:      # cen0 ~ 0, so t is centroid
                continue
            seen.add(k)
            frames.append(g2)
            queue.append(g2)
            if len(frames) >= cap:
                return frames
    return frames


def _pgd_smooth(V, tris, iters=6, lam=0.5):
    """A few Laplacian relaxation sweeps over the triangle mesh -- evens out
    the radial-grid sampling and the small seams left where cube-snapped
    neighbour patches meet, without moving the surface off itself."""
    if not tris or iters <= 0:
        return V
    T = np.asarray(tris, dtype=np.int64)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]],
                        T[:, [1, 0]], T[:, [2, 1]], T[:, [0, 2]]], axis=0)
    n = len(V)
    deg = np.zeros(n)
    np.add.at(deg, e[:, 0], 1.0)
    deg = np.maximum(deg, 1.0)
    V = V.copy()
    for _ in range(iters):
        acc = np.zeros_like(V)
        np.add.at(acc, e[:, 0], V[e[:, 1]])
        V += lam * (acc / deg[:, None] - V)
    return V


def _pgd_weld_tris(Vc, faces_per, nframes, nV, tol):
    """Concatenate `nframes` transformed copies of one patch (already in
    Vc) and weld coincident vertices (absolute tolerance `tol`, chosen a
    little under half a grid cell so adjacent patch seams merge but the
    patch's own grid is preserved); return welded V and triangle list."""
    q = np.round(Vc / tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    Vw = np.zeros((int(inv.max()) + 1, 3))
    Vw[inv] = Vc
    tris = []
    for fr in range(nframes):
        base = fr * nV
        for f in faces_per:
            idx = [int(inv[base + i]) for i in f]
            if len(f) == 3:
                if len(set(idx)) == 3:
                    tris.append(tuple(idx))
            else:
                a, b, c, d = idx
                if len({a, b, c}) == 3:
                    tris.append((a, b, c))
                if len({a, c, d}) == 3:
                    tris.append((a, c, d))
    return Vw, tris


def _pgd_patch_tris(theta, nu, nv, mask_eps=0.045):
    """The exact fundamental surface patch as a welded triangle mesh:
    integrate over the Cayley w-disk, weld the coincident centre-fan
    vertices, triangulate, and relax the sampling.  Small clean holes are
    left at the six (parametrization-singular) hexagon corners."""
    V0, faces, valid, w = pgd_patch(theta, nu=nu, nv=nv, rmax=1.0,
                                    mask_eps=mask_eps)
    diag = float(np.linalg.norm(V0.max(0) - V0.min(0))) or 1.0
    Vw, tris = _pgd_weld_tris(V0, faces, 1, len(V0), 1e-5 * diag)
    used = np.unique(np.array(tris, dtype=np.int64).ravel())
    remap = np.full(len(Vw), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    Vf = Vw[used]
    tris = [(int(remap[x]), int(remap[y]), int(remap[z]))
            for (x, y, z) in tris]
    Vf = _pgd_smooth(Vf, tris, iters=4, lam=0.5)
    return Vf, tris


# --- watertight space-group tiling of one unit cell (P and D) --------------
# The fundamental patch tiles space by the Schwarz reflection principle applied
# to its six hexagon edges.  Which isometry continues the surface across an
# edge depends on the edge's geometry (verified per edge by an SVD of the edge
# curve, so no theta-tracking is needed -- each special angle is handled in its
# own fixed symmetry frame):
#   * a PLANAR-but-curved edge  -> mirror reflection in the plane it lies in
#     (Schwarz P at theta = 0: all six edges are planar geodesics lying in the
#     cube's coordinate mirror planes -- so P assembles by pure coordinate-plane
#     reflections, which compose EXACTLY with no drift);
#   * a STRAIGHT edge           -> 180-degree rotation about the line (Schwarz D
#     at theta = 90 deg: the six edges are straight cube-diagonal lines);
#   * a SKEW edge               -> 180-degree rotation about the fitted 2-fold
#     axis (the chiral gyroid -- no mirror planes exist, so it never reflects).
# The linear part of every generator is snapped to the exact 48-element cube
# group O_h so the generated space group closes without composition drift; the
# breadth-first orbit of the patch, welded on the shared symmetry elements,
# fills one cubic cell, which pgd_build then arrays by the (verified) cubic
# period.  See Schwarz (1890) for the reflection principle and Karcher-Polthier,
# "Construction of triply periodic minimal surfaces", Phil. Trans. R. Soc. A
# 354 (1996) 2077-2104, for the reflection-group assembly of P and D.

def _pgd_cube48():
    """The 48 signed permutation matrices -- the full cube point group O_h
    (proper rotations det +1 and improper/mirror operations det -1)."""
    mats = []
    for perm in _itertools.permutations(range(3)):
        for sg in _itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = sg[i]
            mats.append(M)
    return np.array(mats)


_PGD_CUBE48 = _pgd_cube48()


def _pgd_snap48(M):
    """Nearest exact O_h operation to M (kills composition drift)."""
    return _PGD_CUBE48[int(np.argmin(
        np.abs(_PGD_CUBE48 - M).reshape(48, -1).max(axis=1)))]


def _pgd_rid48(M):
    return int(np.argmin(np.abs(_PGD_CUBE48 - M).reshape(48, -1).max(axis=1)))


def _pgd_arclen_mid(C):
    """Arc-length midpoint of a polyline -- a fixed point of the edge's 2-fold
    (it lies on the rotation axis) / a point of the edge's mirror plane."""
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    half = s[-1] / 2.0
    j = max(0, min(int(np.searchsorted(s, half)) - 1, len(C) - 2))
    f = (half - s[j]) / max(s[j + 1] - s[j], 1e-12)
    return C[j] * (1 - f) + C[j + 1] * f


def _pgd_edge_ops(theta):
    """The six edge-continuation isometries (M, b, kind): mirror for a planar
    edge, 2-fold for a straight/skew one.  M is snapped to O_h and the offset
    b = (I - M) q is taken from the edge's arc-length midpoint q (which lies on
    the mirror plane / rotation axis), giving an accurate, mutually consistent
    generator set."""
    tf = min(max(theta, 0.05), 0.5 * math.pi - 0.05)   # 2-fold axis-fit angle
    ops = []
    for i in range(6):
        C = _pgd_edge_curve(theta, i, m=400)
        _, s, vt = np.linalg.svd(C - C.mean(0), full_matrices=False)
        sr = s / s[0]
        q = _pgd_arclen_mid(C)
        if sr[2] < 0.003 and sr[1] > 0.02:             # planar curve -> mirror
            n = vt[2]
            M = np.eye(3) - 2.0 * np.outer(n, n)
            kind = 'mirror'
        else:                                          # straight/skew -> 2fold
            Cf = _pgd_edge_curve(tf, i, m=96)
            M, _b, _r = _pgd_fit_twofold(Cf)
            kind = 'twofold'
        Ms = _pgd_snap48(M)
        ops.append((Ms, (np.eye(3) - Ms) @ q, kind))
    return ops


def _pgd_lattice_basis(lat):
    """Three shortest linearly independent lattice vectors (basis rows)."""
    order = np.argsort(np.linalg.norm(lat, axis=1))
    basis = []
    for i in order:
        v = lat[i]
        if not basis or np.linalg.matrix_rank(
                np.vstack(basis + [v]), tol=1e-2) > len(basis):
            basis.append(v)
        if len(basis) == 3:
            break
    return np.array(basis)


def _pgd_coset_reps(gens, B, tol=0.25):
    """Canonical coset translation tau_R (reduced mod the lattice L) for each
    cube-group element reached -- lets composed frames be snapped back onto the
    exact space group {tau_R + L n}, so the orbit stays drift-free and finite
    even for the 2-fold (D / gyroid) assemblies."""
    Binv = np.linalg.inv(B.T)

    def redmod(t):
        c = Binv @ t
        return B.T @ (c - np.round(c))
    tau = {_pgd_rid48(np.eye(3)): np.zeros(3)}
    queue = [(np.eye(3), np.zeros(3))]
    guard = 0
    while queue and guard < 5000:
        guard += 1
        g = queue.pop(0)
        for M, b in gens:
            M2 = g[0] @ M
            t2 = g[0] @ b + g[1]
            r = _pgd_rid48(M2)
            tr = redmod(t2)
            if r in tau and np.linalg.norm(redmod(tr - tau[r])) < tol:
                continue
            if r in tau and np.linalg.norm(tr) >= np.linalg.norm(tau[r]):
                continue
            tau[r] = tr
            queue.append((_PGD_CUBE48[r], tr))
    return tau


def _pgd_tile_cell(theta, nu, nv, mask_eps=0.02, smooth=3):
    """Assemble one watertight filled unit cell of the (periodic) surface at
    `theta` by orbiting the fundamental patch under its six edge isometries and
    welding on the shared symmetry elements.  Returns (V, tris, a) in natural
    (un-fit) coordinates with a the cubic period, or None if the angle is not
    one of the cleanly-tileable members (only P and D reassemble watertight)."""
    ops = _pgd_edge_ops(theta)
    kinds = [k for (_M, _b, k) in ops]
    # gyroid / generic angles: the 2-fold reassembly over-covers and cannot be
    # made watertight here -> caller falls back to the fundamental piece.
    is_P = all(k == 'mirror' for k in kinds)
    is_D = all(k == 'twofold' for k in kinds) and \
        abs(theta - 0.5 * math.pi) < 0.02
    if not (is_P or is_D):
        return None
    Vp, tris_p = _pgd_patch_tris(theta, nu, nv, mask_eps=mask_eps)
    diag = float(np.linalg.norm(Vp.max(0) - Vp.min(0))) or 1.0
    cen0 = Vp.mean(0)
    gens = [(M, b) for (M, b, k) in ops]
    lat = pgd_lattice(gens)
    a = float(np.min(np.linalg.norm(lat, axis=1))) if len(lat) else 4.4
    B = _pgd_lattice_basis(lat) if len(lat) >= 3 else None
    tau = _pgd_coset_reps(gens, B) if B is not None else None
    Binv = np.linalg.inv(B.T) if B is not None else None

    def snap_frame(M2, b2):
        if tau is None:
            return b2
        tr = tau.get(_pgd_rid48(M2))
        if tr is None:
            return b2
        return tr + B.T @ np.round(Binv @ (b2 - tr))

    box = 0.5 * a + 0.05 * a
    # interior sample points for the interpenetration test (2-fold orbits of the
    # over-large patch would otherwise pile copies onto the same sheet)
    rin = np.linalg.norm(Vp - cen0, axis=1)
    Vint = Vp[np.argsort(rin)[:max(24, len(Vp) // 5)]]
    ohit = 0.06 * diag
    ghash = {}

    def _gk(p):
        return (int(round(p[0] / ohit)), int(round(p[1] / ohit)),
                int(round(p[2] / ohit)))

    def add_pts(P):
        for p in P:
            ghash.setdefault(_gk(p), []).append(p)

    def overlap_frac(P):
        hit = 0
        for p in P:
            gk = _gk(p)
            f = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for q in ghash.get((gk[0] + dx, gk[1] + dy,
                                            gk[2] + dz), ()):
                            if (abs(q[0] - p[0]) < ohit
                                    and abs(q[1] - p[1]) < ohit
                                    and abs(q[2] - p[2]) < ohit):
                                f = True
                                break
                        if f:
                            break
                    if f:
                        break
            hit += f
        return hit / len(P)

    I = (np.eye(3), np.zeros(3), +1)
    frames = [I]
    add_pts(Vint)
    dedup = max(0.12 * a, 0.15)
    seen = {tuple(np.round(cen0 / dedup).astype(int))}
    queue = [I]
    while queue and len(frames) < 400:
        (M, b, o) = queue.pop(0)
        for (Mg, bg) in gens:
            M2 = Mg @ M
            b2 = snap_frame(M2, Mg @ b + bg)
            c = M2 @ cen0 + b2
            if np.max(np.abs(c)) > box:
                continue
            key = tuple(np.round(c / dedup).astype(int))
            if key in seen:
                continue
            seen.add(key)
            cand = Vint @ M2.T + b2
            if overlap_frac(cand) > 0.4:               # would interpenetrate
                continue
            o2 = o * int(round(np.linalg.det(Mg)))
            frames.append((M2, b2, o2))
            add_pts(cand)
            queue.append((M2, b2, o2))
    # assemble with consistent winding (mirror copies flip orientation)
    Vparts, Tp, base = [], [], 0
    for (M, b, o) in frames:
        Vparts.append(Vp @ M.T + b)
        for (x, y, z) in tris_p:
            Tp.append((x + base, z + base, y + base) if o < 0
                      else (x + base, y + base, z + base))
        base += len(Vp)
    Vc = np.concatenate(Vparts, axis=0)
    tol = 0.9 * diag / nu
    q = np.round(Vc / tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    Vw = np.zeros((int(inv.max()) + 1, 3))
    Vw[inv] = Vc
    tris = []
    for (x, y, z) in Tp:
        t = (int(inv[x]), int(inv[y]), int(inv[z]))
        if len(set(t)) == 3:
            tris.append(t)
    Vw = _pgd_smooth(Vw, tris, iters=smooth, lam=0.4)
    return Vw, tris, a


def pgd_build(cells, res, scale, theta):
    """Build the exact Schwarz P / Gyroid / Schwarz D associate surface at
    Bonnet angle `theta`, centered and fit to a 2 m cube (times `scale`).
    Returns (V (n,3) float, tris list).

    Two regimes, chosen by the angle:

    * P (theta ~ 0) and D (theta ~ 90 deg) are assembled into a *watertight
      filled unit cell* by the Schwarz reflection principle -- P by reflecting
      the fundamental patch across its six coordinate mirror planes, D by
      180-degree rotations about its six straight edge lines (see
      `_pgd_tile_cell`).  `cells` > 1 arrays that whole cell on the verified
      cubic period, so the recognizable P / D network fills the lattice.
      `cells` may be an int (symmetric block) or a (cx, cy, cz) triple for
      independent per-axis counts.

    * The Gyroid (theta ~ 38.0148 deg) and every generic (non-periodic)
      intermediate angle keep the exact Weierstrass *fundamental piece* -- the
      image of one translational fundamental domain -- which morphs
      continuously and correctly through the whole family.  Only P / Gyroid / D
      are truly triply periodic, and the chiral gyroid (no mirror planes) has
      no drift-free reflection assembly: its 2-fold reassembly over-covers and
      cannot be closed watertight in this scheme, so -- following the honest-
      scope rule -- the clean fundamental piece is built rather than a torn
      multi-cell approximation.  (`cells` is ignored for these angles; the
      single piece is returned.)"""
    if isinstance(cells, (tuple, list)):
        cx, cy, cz = (int(max(1, c)) for c in (list(cells) + [1, 1, 1])[:3])
    else:
        cx = cy = cz = max(1, int(cells))
    nu = max(24, int(round(res)))
    nv = _pgd_good_nv(max(120, int(round(res * 2.4))))
    tiled = _pgd_tile_cell(theta, nu, nv)
    if tiled is not None:
        Vc, tris_c, a = tiled
        if cx > 1 or cy > 1 or cz > 1:
            oxs = (np.arange(cx) - 0.5 * (cx - 1)) * a
            oys = (np.arange(cy) - 0.5 * (cy - 1)) * a
            ozs = (np.arange(cz) - 0.5 * (cz - 1)) * a
            Vparts, Tparts, base = [], [], 0
            for ox in oxs:
                for oy in oys:
                    for oz in ozs:
                        Vparts.append(Vc + np.array([ox, oy, oz]))
                        Tparts.extend((x + base, y + base, z + base)
                                      for (x, y, z) in tris_c)
                        base += len(Vc)
            V, tris = np.concatenate(Vparts, axis=0), Tparts
        else:
            V, tris = Vc, tris_c
    else:
        # gyroid / generic angle: honest single fundamental piece
        V, tris = _pgd_patch_tris(theta, nu, nv)
    lo, hi = V.min(0), V.max(0)
    cen = 0.5 * (lo + hi)
    ext = float(np.max(hi - lo)) or 1.0
    V = (V - cen) * (2.0 / ext) * float(scale)
    return V, tris


# ==========================================================================
# Higher-genus Chen-Gackstatter surfaces (genus 2, 4, 5)
# ==========================================================================
# Complete minimal surfaces of genus g with ONE Enneper-type end of winding
# 3: the higher-genus continuation of the Chen-Gackstatter torus.  They live
# on the hyperelliptic curve  y^2 = z * prod_i (z^2 - r_i^2)  with the
# unified Weierstrass data
#     g = rho * y / D(z),      dh = dz          (so x3 = Re z),
# where D collects half of the branch factors (for genus 2 the z sits in the
# DENOMINATOR of g^2, for genus 4/5 in the numerator) and the real constants
# r_i, rho solve the period problem (values below verified numerically:
# the null identity |phi1^2 + phi2^2 + phi3^2| ~ 1e-16 and all periods of
# the double cover close).
#
# Symmetry group: D2d of order 8, the SAME for every genus:
#   * z -> -z acts on the immersion as the rotoreflection S4 about the
#     vertical axis, (x1, x2, x3) -> (x2, -x1, -x3)  (g -> i g, dh -> -dh);
#   * S4^2 = C2 is the hyperelliptic sheet swap (y -> -y);
#   * z -> conj(z) and z -> -conj(z) are antiholomorphic: two vertical
#     sigma_d mirror planes x = cx, y = cy (the images of the real-axis
#     branch intervals) and two horizontal 2-fold axes at x3 = 0 (the
#     images of the imaginary axis);
#   * the S4 axis threads every branch-point image at (cx, cy, z_branch).
#
# Meshing scheme (the part that makes the assembly watertight): ONE
# 1/8 Coxeter fundamental domain -- the quarter  {Re z >= 0}  of the upper
# half plane, i.e. the half  {Im w <= 0}  of the Cayley w-disk
# (w = (z-i)/(z+i)) -- whose boundary lies ON the symmetry elements:
# the real-w diameter maps onto a horizontal 2-fold axis (a straight line
# through (cx, cy, 0) at 45 degrees to the mirror planes), and the rim
# semicircle maps into the two mirror planes, split at the branch points.
# The WE forms are integrated radially on one continuous sqrt branch, each
# boundary vertex is snapped exactly onto its symmetry element (line /
# plane / axis point), the patch is orbited under the full 8-element D2d
# group and the copies weld by coincidence -- boundary vertices only, so
# interior verts can never fuse.  Face winding flips exactly for the four
# antiholomorphic copies (mirrors and 2-fold axes), NOT by det: S4 has
# det -1 but is holomorphic (z -> -z) and keeps its winding.  A spherical
# clip about (cx, cy, 0) trims the one flaring end; with the trim radius
# past the outermost handles the result is exactly Euler characteristic
# chi = 1 - 2g (one boundary loop), edge-manifold and connected -- gated
# by the self-tests below.
#
# References:
#   C. C. Chen, F. Gackstatter, "Elliptische und hyperelliptische
#     Funktionen und vollstaendige Minimalflaechen vom Enneperschen Typ",
#     Math. Ann. 259 (1982) -- the genus-1 and genus-2 surfaces;
#   E. C. Thayer, "Higher-genus Chen-Gackstatter surfaces and the
#     Weierstrass representation for surfaces of infinite genus",
#     Experiment. Math. 4 (1995) -- the genus >= 2 family;
#   H. Karcher, "Construction of minimal surfaces", Univ. of Tokyo lecture
#     notes (1989) -- the symmetry/period method;
#   M. Weber, https://minimalsurfaces.blog/ (higher-genus Chen-Gackstatter
#     pages) -- the numerical data this implementation follows.

_CGH_DATA = {
    2: dict(roots=(1.0, 1.7126826390981942),
            Dfac=(('z',), ('sq', 1.7126826390981942)),
            rho=None),                # solved from the [0,1] period ratio
    4: dict(roots=(1.0, 1.81645934660556296, 3.11436011061010598,
                   3.77509108812262628),
            Dfac=(('sq', 1.0), ('sq', 3.11436011061010598)),
            rho=0.580558059350863508),
    5: dict(roots=(1.0, 2.19951977246661467, 3.04734348507243302,
                   4.58374227188035909, 5.28690084560405004),
            Dfac=(('sq', 1.0), ('sq', 3.04734348507243302),
                  ('sq', 5.28690084560405004)),
            rho=1.97502242055676724),
}

_CGH_RHO_CACHE = {}


def _cgh_forms(genus):
    """(Pfun, Dfun, rho, roots) for the genus: y^2 = P(z), g = rho*y/D."""
    d = _CGH_DATA[genus]
    roots = d['roots']

    def Pfun(z):
        z = np.asarray(z, dtype=complex)
        out = z.copy()
        for r in roots:
            out = out * (z ** 2 - r ** 2)
        return out

    def Dfun(z):
        z = np.asarray(z, dtype=complex)
        out = np.ones_like(z)
        for f in d['Dfac']:
            out = out * (z if f[0] == 'z' else (z ** 2 - f[1] ** 2))
        return out

    rho = d['rho']
    if rho is None:                    # genus 2: scalar period ratio on [0,1]
        if genus not in _CGH_RHO_CACHE:
            zz = np.linspace(1e-9, 1 - 1e-9, 400000)
            y = np.sqrt(Pfun(zz))
            _trapz = getattr(np, 'trapezoid', None) or np.trapz
            _CGH_RHO_CACHE[genus] = math.sqrt(
                _trapz((y / (zz ** 2 - 1.0)).real, zz)
                / _trapz(((zz ** 2 - 1.0) / y).real, zz))
        rho = _CGH_RHO_CACHE[genus]
    return Pfun, Dfun, rho, roots


def _cgh_octant(genus, nu, arcn):
    """Mesh the 1/8 fundamental domain (half of the Cayley disk,
    Im w <= 0) by radial integration from w = 0 (z = i).  Returns
    (V, faces, boundary-classification dict, per-vertex uv)."""
    Pfun, Dfun, rho, roots = _cgh_forms(genus)
    bps = [0.0] + sorted(roots)        # finite branch points, z = 0 first
    ang = [np.angle((b - 1j) / (b + 1j)) % TAU for b in bps]
    ang = [a + TAU if a < math.pi - 1e-12 else a for a in ang]
    ang.append(TAU)                    # z = inf at w = 1
    # angular grid with the exact branch angles as shared nodes
    th = []
    for k in range(len(ang) - 1):
        n_k = max(8, int(round(arcn * (ang[k + 1] - ang[k])
                               / (math.pi / 4))))
        seg = np.linspace(ang[k], ang[k + 1], n_k + 1)
        th.extend(seg[:-1] if k < len(ang) - 2 else seg)
    th = np.array(th)
    bcol = [int(np.argmin(np.abs(th - a))) for a in ang[:-1]]
    nv = len(th)
    s = np.linspace(0.0, 1.0, nu)
    r = 1.0 - (1.0 - s) ** 1.7         # radially graded toward the rim
    R, TH = np.meshgrid(r, th, indexing='ij')
    w = R * np.exp(1j * TH)
    # one continuous branch of sqrt(P(z(w))) along the radial rays
    P = Pfun(1j * (1.0 + w) / (1.0 - w))
    pang = np.unwrap(np.angle(P), axis=0)
    pang = pang - pang[:1, :] + np.unwrap(pang[0])[None, :]
    ycont = np.sqrt(np.abs(P)) * np.exp(0.5j * pang)
    z = 1j * (1.0 + w) / (1.0 - w)
    g = rho * ycont / Dfun(z)
    J = 2j / (1.0 - w) ** 2            # dz/dw
    with np.errstate(divide='ignore', invalid='ignore'):
        F = np.stack([0.5 * (1.0 / g - g) * J,
                      0.5j * (1.0 / g + g) * J,
                      np.broadcast_to(J, g.shape)], axis=-1) \
            * np.exp(1j * TH)[..., None]
    F = np.where(np.isfinite(F), F, 0.0)
    dr = np.diff(R, axis=0)[..., None]
    X = np.concatenate([np.zeros((1, nv, 3)),
                        np.cumsum(0.5 * (F[1:] + F[:-1]) * dr, axis=0)],
                       axis=0)
    X = np.real(X)

    def vid(i, j):                     # row 0 collapses to one center vert
        return 0 if i == 0 else (i - 1) * nv + j + 1

    V = np.concatenate([X[:1, 0, :], X[1:].reshape(-1, 3)], axis=0)
    UV = np.zeros((len(V), 2))
    UV[0] = (0.5, 0.0)
    UV[1:, 0] = np.tile((th - math.pi) / math.pi, nu - 1)
    UV[1:, 1] = np.repeat(r[1:], nv)
    faces = []
    for j in range(nv - 1):
        faces.append((0, vid(1, j), vid(1, j + 1)))
    for i in range(1, nu - 1):
        for j in range(nv - 1):
            faces.append((vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)))
    b = {'seam': np.array([0] + [vid(i, j) for i in range(1, nu - 1)
                                 for j in (0, nv - 1)], dtype=np.int64),
         'rim': {}, 'branch': {}, 'branch_z': {}}
    for k in range(len(ang) - 1):
        j0 = bcol[k]
        j1 = bcol[k + 1] if k + 1 < len(bcol) else nv - 1
        b['rim'][k] = np.array([vid(nu - 1, j) for j in range(j0 + 1, j1)],
                               dtype=np.int64)
    for k in range(len(bcol)):
        b['branch'][k] = vid(nu - 1, bcol[k])
        b['branch_z'][k] = bps[k]
    # rim ring triples (rim vert, its inner neighbor, next inner) -- used
    # to keep the snapped rim clear of the last interior row
    b['ring'] = np.array([[vid(nu - 1, j), vid(nu - 2, j), vid(nu - 3, j)]
                          for j in range(nv)], dtype=np.int64)
    return V, faces, b, UV


def _cgh_snap(V, b):
    """Snap every boundary vertex exactly onto its symmetry element.
    The mirror-plane offsets cx, cy are read off the rim arcs themselves
    (each arc's near-constant coordinate); returns the axis point q."""
    arc_ax, arc_val = {}, {}
    for k, idx in b['rim'].items():
        if len(idx) == 0:
            continue
        sx = float(np.median(np.abs(V[idx, 0] - np.median(V[idx, 0]))))
        sy = float(np.median(np.abs(V[idx, 1] - np.median(V[idx, 1]))))
        ax = 0 if sx < sy else 1
        arc_ax[k] = ax
        arc_val[k] = float(np.median(V[idx, ax]))
    cx = float(np.median([v for k, v in arc_val.items() if arc_ax[k] == 0]))
    cy = float(np.median([v for k, v in arc_val.items() if arc_ax[k] == 1]))
    q = np.array([cx, cy, 0.0])
    for k, idx in b['rim'].items():
        if len(idx):
            V[idx, arc_ax[k]] = q[arc_ax[k]]
    # the imaginary axis maps onto the horizontal 2-fold line through q at
    # 45 degrees to the mirror planes; project the seam onto the best of
    # the two candidate directions
    sv = b['seam']
    Pq = V[sv] - q
    best = None
    for sgn in (1.0, -1.0):
        d = np.array([1.0, sgn, 0.0]) / math.sqrt(2.0)
        t = Pq @ d
        res = float(np.median(np.linalg.norm(
            Pq - t[:, None] * d[None, :], axis=1)))
        if best is None or res < best[0]:
            best = (res, d)
    d = best[1]
    t = (V[sv] - q) @ d
    V[sv] = q + t[:, None] * d[None, :]
    # branch-point images: exactly on the S4 axis at height z_branch
    for k, vidx in b['branch'].items():
        V[vidx] = np.array([q[0], q[1], b['branch_z'][k]])
    return V, q


def _cgh_frames(q):
    """The 8 affine isometries of D2d about q.  Frames 0..3 are the
    holomorphic copies e, S4, C2, S4^3 (winding kept); frames 4..7 are
    the antiholomorphic mirror / 2-fold copies (winding reversed)."""
    S4 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    SX = np.diag([-1.0, 1.0, 1.0])
    mats = []
    M = np.eye(3)
    for _ in range(4):
        mats.append(M)
        M = S4 @ M
    mats += [SX @ Mk for Mk in mats]
    return [(Mk, q - Mk @ q) for Mk in mats]


def cg_higher_assemble(genus, nu, arcn, Rend):
    """Watertight D2d assembly: 8 snapped copies of the 1/8 domain,
    boundary-coincidence weld, spherical end trim.  Returns
    (V, faces, uv) of the largest component."""
    V0, faces0, b, UV0 = _cgh_octant(genus, nu, arcn)
    V0, q = _cgh_snap(V0.copy(), b)
    # keep the last interior row clear of the snapped rim: the plane snap
    # can land a rim vertex arbitrarily close to its inward neighbor, and
    # a later mesh-level weld (the operator's remove-doubles) would pinch
    # the sheets there.  Interior verts are free, so back the neighbor off
    # to the midpoint of its own inward edge.
    vr, v1, v2 = b['ring'][:, 0], b['ring'][:, 1], b['ring'][:, 2]
    close = np.linalg.norm(V0[v1] - V0[vr], axis=1) < 2e-4
    if np.any(close):
        V0[v1[close]] = 0.5 * (V0[vr[close]] + V0[v2[close]])
    bmask0 = np.zeros(len(V0), dtype=bool)
    bmask0[b['seam']] = True
    for idx in b['rim'].values():
        bmask0[idx] = True
    bmask0[list(b['branch'].values())] = True
    keep = np.linalg.norm(V0 - q, axis=1) <= Rend
    faces0 = [f for f in faces0 if all(keep[i] for i in f)]
    used = sorted(set(i for f in faces0 for i in f))
    rmv = {v: k for k, v in enumerate(used)}
    V0, UV0, bmask0 = V0[used], UV0[used], bmask0[used]
    faces0 = [tuple(rmv[i] for i in f) for f in faces0]
    nV = len(V0)
    frames = _cgh_frames(q)
    Vp, Fp = [], []
    for fr, (M, bb) in enumerate(frames):
        Vp.append(V0 @ M.T + bb)
        rev = fr >= 4                  # antiholomorphic copies flip winding
        for f in faces0:
            ff = tuple(int(x) + fr * nV for x in f)
            Fp.append(ff[::-1] if rev else ff)
    V = np.concatenate(Vp)
    UVall = np.tile(UV0, (len(frames), 1))
    N = len(V)
    # coincidence weld restricted to the snapped boundary verts
    tol = 1e-6
    parent = np.arange(N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bidx = np.nonzero(np.tile(bmask0, len(frames)))[0]
    key = np.floor(V[bidx] / tol + 0.5).astype(np.int64)
    H = {}
    for t, i in enumerate(bidx):
        H.setdefault((int(key[t, 0]), int(key[t, 1]), int(key[t, 2])),
                     []).append(int(i))
    for t, i in enumerate(bidx):
        k0 = key[t]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in H.get((int(k0[0]) + dx, int(k0[1]) + dy,
                                    int(k0[2]) + dz), ()):
                        if j > i and np.linalg.norm(V[j] - V[i]) < tol:
                            ra, rb = find(int(i)), find(j)
                            if ra != rb:
                                parent[ra] = rb
    rt = np.array([find(a) for a in range(N)])
    uniq, inv = np.unique(rt, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    UVw = np.zeros((len(uniq), 2))
    cnt = np.zeros(len(uniq))
    np.add.at(Vw, inv, V)
    np.add.at(UVw, inv, UVall)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    UVw /= cnt[:, None]
    F = []
    for f in Fp:
        gg = [int(inv[i]) for i in f]
        h = [gg[0]]
        for t in range(1, len(gg)):
            if gg[t] != h[-1]:
                h.append(gg[t])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    # largest face-connected component (drops any stray trim islands)
    parent2 = np.arange(len(Vw))

    def find2(a):
        while parent2[a] != a:
            parent2[a] = parent2[parent2[a]]
            a = parent2[a]
        return a

    for f in F:
        for i in range(1, len(f)):
            ra, rb = find2(f[0]), find2(f[i])
            if ra != rb:
                parent2[ra] = rb
    from collections import Counter
    sizes = Counter(find2(f[0]) for f in F)
    root = sizes.most_common(1)[0][0]
    F = [f for f in F if find2(f[0]) == root]
    used = sorted(set(i for f in F for i in f))
    rmv = {v: k for k, v in enumerate(used)}
    return (Vw[used], [tuple(rmv[i] for i in f) for f in F], UVw[used])


def cg_higher_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """MESH_PARAM builder: finished (V, quads, uv), fit to the 2 m cube."""
    p = spec['p_from'](order, radius)
    genus = p['genus']
    pnu = int(np.clip(nu * 1.5, 80, 280))
    arcn = int(np.clip(nv * 0.65, 28, 110))
    R0 = {2: 5.5, 4: 6.0, 5: 8.0}[genus]
    lo, hi = {2: (3.2, 10.0), 4: (4.2, 10.0), 5: (6.2, 12.0)}[genus]
    Rend = float(np.clip(R0 * radius / 1.2, lo, hi))
    V, quads, uv = cg_higher_assemble(genus, pnu, arcn, Rend)
    tk = _toolkit()
    V = tk._smooth_boundary(V, quads, iters=6)
    V = tk._center_fit(V, scale, V)
    return V, quads, uv


# ==========================================================================
# Translation-invariant genus-one helicoid ("helicoid with a handle")
# ==========================================================================
# The singly periodic minimal surface asymptotic to a helicoid whose
# quotient by its vertical translation is a rhombic torus minus two
# helicoidal ends -- i.e. a helicoid that carries ONE handle per period.
# It was the key existence step toward the (non-periodic) genus-one
# helicoid.
#
# Weierstrass data on the rhombic torus C/<1, tau>, written with the
# Jacobi theta function theta_11 (data as in Weber's notebook,
# harvested in research/msblog_harvest/singly_periodic.json under
# translation_invariant_helicoid_with_handle):
#
#   G(z)  = rho1 e^{i pi (b - 2z + 2 tau + b tau)}
#           theta(z + (b-2)c) theta(z - (1+b)c)
#           / ( theta(z + (b-1)c) theta(z - b c) ),      c = (1+tau)/2
#   dh    = theta(z + (b-2)c) theta(z - b c)
#           / ( theta(z + (b-1)c) theta(z - (1+b)c) ) dz / dhper
#
# Both are elliptic (fully periodic) on the torus; the four theta
# points on the diagonal are the two helicoidal ends (parameters 1-b
# and 1+b, where dh has simple poles) and the two points of vertical
# normal (parameters b and 2-b) that make the handle.  The solved
# period problem (all constants harvested verbatim, no re-solving
# here): tau = e^{i alpha0 deg}, plus rho1, b, dhper, and the domain
# constants r0, a0.  With them the lattice cycle z -> z+1 integrates
# to the exact vertical translation (0, 0, 2) and the cycle z -> z+tau
# closes to ~1e-7 (the self-test checks both).
#
# The parameter domain is the conformal half-strip R x (0, pi): w maps
# to the torus by  z = tst(tr(e^w)) (1+tau)/2, where tr is a real
# Moebius map and tst the Schwarz-Christoffel rectangle map
# F(arcsin s | 1/r0^2) / (2 K(1/r0^2)) + 1/2, evaluated for complex s
# via Carlson's R_F.  The strip covers half the torus; the immersed
# sheet contains the vertical z-axis and horizontal rulings at integer
# heights, and one fundamental cell is the sheet plus its 180-degree
# rotation about the z axis, welded along those lines; cells stack by
# (0, 0, 2).  A stack of S cells has genus exactly S (one handle per
# period; the self-test verifies chi = 2 - 2S - loops).
#
# References:
#   D. Hoffman, H. Karcher, F. Wei, "Adding handles to the helicoid",
#     Bull. Amer. Math. Soc. 29 (1993), 77-84.
#   D. Hoffman, H. Karcher, F. Wei, "The singly periodic genus-one
#     helicoid", Comment. Math. Helv. 74 (1999), 248-279.
#   D. Hoffman, M. Weber, M. Wolf, "An embedded genus-one helicoid",
#     Ann. of Math. 169 (2009), 347-448 (the non-periodic limit).
#   M. Weber, "The translation invariant helicoid with handle",
#     https://minimalsurfaces.blog/ (notebook data, 1996).
#   B. C. Carlson, "Numerical computation of real or complex elliptic
#     integrals", Numer. Algorithms 10 (1995), 13-98 (R_F).

# harvested constants (verbatim from the notebook; do not re-solve)
_G1H_ALPHA0 = 70.7083362972048057                 # degrees
_G1H_TAU = complex(np.exp(1j * np.pi * _G1H_ALPHA0 / 180.0))
_G1H_B = 0.629065098323904514
_G1H_RHO1 = 108.369522264594063 - 62.8417365006266681j
_G1H_DHPER = 0.386191090012370175 - 0.169838749468014027j
_G1H_R0 = 2.43050611112724901
_G1H_A0 = -0.409955776251214221
# x-position of the normal-symmetry point of the strip (the two sums
# solve tr(e^x) = 1 on y = 0 and tr(e^x) = -r0 on y = pi)
_G1H_SYM = -0.359811577777830482 - 0.528287934072206422
# strip x of the four rectangle corners (branch points of the domain
# map): tr(e^x) = 1, r0 on y = 0 and -1, -r0 on y = pi
_G1H_XA = -0.3598115777778303
_G1H_XB = 0.6834249528850583
_G1H_XC = -1.5715244647350952
_G1H_XD = -0.5282879340722066
_G1H_EPS = 1e-7      # inset from the strip boundary: keeps the Carlson
#                      arguments off their branch cut (negative reals)


def genus1helicoid_theta11(z, tau, nterms=30):
    """Jacobi theta_11 (odd theta), theta_11(z, tau) = 2 sum_{n>=0}
    (-1)^n q^{(n+1/2)^2} sin((2n+1) pi z), q = e^{i pi tau}.  z is
    first reduced modulo the lattice <1, tau> and the exact
    quasi-periodicity factors are applied, so the truncated series
    converges fast for any argument."""
    z = np.asarray(z, dtype=complex)
    m = np.round(z.imag / tau.imag)
    z1 = z - m * tau
    n = np.round(z1.real)
    z2 = z1 - n
    fac = ((-1.0) ** (m + n)
           * np.exp(-1j * np.pi * m * m * tau - 2j * np.pi * m * z2))
    q = np.exp(1j * np.pi * tau)
    s = np.zeros_like(z2)
    for k in range(nterms):
        s = s + ((-1.0) ** k * q ** ((k + 0.5) ** 2)
                 * np.sin((2 * k + 1) * np.pi * z2))
    return 2.0 * fac * s


def _g1h_rf(x, y, z, iters=26):
    """Carlson symmetric elliptic integral R_F for complex arguments
    off the negative real axis (duplication iteration + the standard
    5th-order tail; Carlson 1995)."""
    x = np.asarray(x, dtype=complex).copy()
    y = np.asarray(y, dtype=complex).copy()
    z = np.asarray(z, dtype=complex).copy()
    for _ in range(iters):
        sx, sy, sz = np.sqrt(x), np.sqrt(y), np.sqrt(z)
        lam = sx * sy + sy * sz + sz * sx
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
    A = (x + y + z) / 3.0
    X = 1.0 - x / A
    Y = 1.0 - y / A
    Z = -(X + Y)
    E2 = X * Y - Z * Z
    E3 = X * Y * Z
    return (1.0 - E2 / 10.0 + E3 / 14.0 + E2 * E2 / 24.0
            - 3.0 * E2 * E3 / 44.0) / np.sqrt(A)


def _g1h_ellf(z, msq):
    """Incomplete elliptic integral F(arcsin z | m), analytically
    continued to the upper half plane: F = z R_F(1-z^2, 1-m z^2, 1)."""
    z = np.asarray(z, dtype=complex)
    return z * _g1h_rf(1.0 - z * z, 1.0 - msq * z * z, np.ones_like(z))


_G1H_M = 1.0 / (_G1H_R0 * _G1H_R0)
_G1H_QUOT = float(2.0 * _g1h_ellf(np.array(1.0 - 1e-15 + 0j),
                                  _G1H_M).real)          # 2 K(m)


def _g1h_tst(s):
    """Schwarz-Christoffel map: upper half plane -> rectangle
    [0,1] x [0, h] (the notebook's tst)."""
    return _g1h_ellf(s, _G1H_M) / _G1H_QUOT + 0.5


def _g1h_map(w):
    """Half-strip coordinate w = x + iy (0 < y < pi) -> torus coord."""
    ew = np.exp(np.asarray(w, dtype=complex))
    s = (-_G1H_A0 - _G1H_R0 * ew) / (-1.0 + _G1H_A0 * ew)
    return _g1h_tst(s) * 0.5 * (1.0 + _G1H_TAU)


def _g1h_omega(z):
    """The three Weierstrass 1-forms (om1, om2, om3) as functions of
    the torus coordinate (values w.r.t. dz), om3 normalized by the
    harvested dhper so the z -> z+1 cycle translates by (0, 0, 2)."""
    c = 0.5 * (1.0 + _G1H_TAU)
    th = genus1helicoid_theta11
    t1 = th(z + (_G1H_B - 2.0) * c, _G1H_TAU)
    t2 = th(z - (1.0 + _G1H_B) * c, _G1H_TAU)
    t3 = th(z + (_G1H_B - 1.0) * c, _G1H_TAU)
    t4 = th(z - _G1H_B * c, _G1H_TAU)
    e = np.exp(1j * np.pi * (_G1H_B - 2.0 * z + 2.0 * _G1H_TAU
                             + _G1H_B * _G1H_TAU))
    G = _G1H_RHO1 * e * t1 * t2 / (t3 * t4)
    o3 = (t1 * t4) / (t3 * t2) / _G1H_DHPER
    o1 = 0.5 * (1.0 / G - G) * o3
    o2 = 0.5j * (1.0 / G + G) * o3
    return o1, o2, o3


def _g1h_path_int(za, zb, n=20001):
    """Integral of (om1, om2, om3) along the straight segment za->zb."""
    t = np.linspace(0.0, 1.0, n)
    path = za + (zb - za) * t
    o1, o2, o3 = _g1h_omega(path)
    dz = np.diff(path)
    return np.array([np.sum(0.5 * (o[1:] + o[:-1]) * dz)
                     for o in (o1, o2, o3)])


def _g1h_graded(lo, hi, n, specials, w=0.2):
    """n samples on [lo, hi] clustered near each special value."""
    t = np.linspace(0.0, 1.0, n)
    x = lo + (hi - lo) * t
    sp = np.asarray(specials)
    for _ in range(3):
        d = np.min(np.abs(x[:, None] - sp[None, :]), axis=1)
        wgt = 1.0 / (w + d)
        cdf = np.concatenate([[0.0],
                              np.cumsum(0.5 * (wgt[1:] + wgt[:-1])
                                        * np.diff(x))])
        cdf /= cdf[-1]
        x = np.interp(t, cdf, x)
    return x


_G1H_SHEET_CACHE = {}


def genus1helicoid_sheet(r1=-2.5, nu=131, nv=53, K=10):
    """Immersed fundamental sheet over the half-strip
    [r1, SYM - r1] x [0, pi]: returns (xs, ys, X) with X (nu', nv, 3)
    real.  The x-grid is symmetric about SYM/2 and contains the four
    corner x-values exactly, so the sheet's straight boundary arcs
    (the z-axis segment and the horizontal rulings) land sample-exact
    and every weld of the assembly is vertex-to-vertex.  Cumulative
    trapezoid integration along grid lines with each interval
    subdivided K times (resolves the sqrt branch corners)."""
    ck = (round(r1, 6), nu, nv, K)
    if ck in _G1H_SHEET_CACHE:
        return _G1H_SHEET_CACHE[ck]
    x_hi = _G1H_SYM - r1
    corners = (_G1H_XA, _G1H_XB, _G1H_XC, _G1H_XD)
    spec = sorted(set(list(corners)
                      + [_G1H_SYM - c for c in corners]))
    xs = _g1h_graded(r1, x_hi, nu, spec)
    xs = np.unique(np.round(np.concatenate(
        [xs, _G1H_SYM - xs, spec, [_G1H_SYM - s for s in spec]]), 12))
    t = np.linspace(0.0, 1.0, nv)
    ys = _G1H_EPS + (np.pi - 2 * _G1H_EPS) * (0.5 - 0.5
                                              * np.cos(np.pi * t))
    nu2 = len(xs)
    j0 = nv // 2
    i0 = int(np.argmin(np.abs(xs - _G1H_SYM / 2.0)))

    def seg(wa, wb):
        tt = np.linspace(0.0, 1.0, K + 1)
        W = wa[:, None] + (wb - wa)[:, None] * tt[None, :]
        Z = _g1h_map(W)
        o1, o2, o3 = _g1h_omega(Z)
        O = np.stack([o1, o2, o3], axis=-1)
        dZ = np.diff(Z, axis=1)
        return np.sum(0.5 * (O[:, 1:] + O[:, :-1]) * dZ[..., None],
                      axis=1)

    F = np.zeros((nu2, nv, 3), complex)
    row = np.concatenate([np.zeros((1, 3), complex),
                          np.cumsum(seg(xs[:-1] + 1j * ys[j0],
                                        xs[1:] + 1j * ys[j0]), axis=0)])
    F[:, j0] = row - row[i0]
    for j in range(j0 + 1, nv):
        F[:, j] = F[:, j - 1] + seg(xs + 1j * ys[j - 1], xs + 1j * ys[j])
    for j in range(j0 - 1, -1, -1):
        F[:, j] = F[:, j + 1] - seg(xs + 1j * ys[j], xs + 1j * ys[j + 1])
    # base point: integrate from the lattice point 1 through tau/2 so
    # the surface's vertical line is exactly the z axis (the notebook's
    # w0 offset)
    C = _g1h_path_int(1.0 + 0.0j, _G1H_TAU / 2.0) \
        + _g1h_path_int(_G1H_TAU / 2.0,
                        complex(_g1h_map(xs[i0] + 1j * ys[j0])))
    out = (xs, ys, np.real(F + C[None, None, :]))
    _G1H_SHEET_CACHE[ck] = out
    return out


def _g1h_weld_pairs(V, quads, pairs):
    """Weld the given exact vertex-index pairs (union-find; merged
    positions averaged).  Returns (V', quads', vertex_map)."""
    n = len(V)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(a) for a in range(n)])
    uniq, first, inv = np.unique(roots, return_index=True,
                                 return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(Vw, inv, V)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    qw = []
    seen = set()
    for q in quads:
        f = tuple(int(inv[i]) for i in q)
        if len(set(f)) >= 3 and frozenset(f) not in seen:
            seen.add(frozenset(f))
            qw.append(f)
    return Vw, qw, first


def genus1helicoid_assemble(storeys=1, r1=-2.5, nu=131, nv=53, K=10):
    """Finished (V, quads, uv): `storeys` translational cells, each the
    strip sheet plus its 180-degree rotation about the z axis, stacked
    by (0, 0, 2) and welded along the shared straight arcs (the axis
    segments and the horizontal rulings).  Genus = storeys.

    Every weld is an exact grid-index correspondence -- the symmetric
    x-grid makes the partner of sample i the sample nu-1-i (x maps to
    SYM - x) -- so no positional tolerance is involved and dense grid
    regions can never over-merge:
      * axis, y=0 edge  (x in [XA, XB], z in [-2, -1] of the cell):
        sheet <-> rotated sheet at the SAME i (the axis is pointwise
        fixed by the 180-degree rotation);
      * axis, y=pi edge (x in [XC, XD], z in [-1, 0]): likewise;
      * in-cell ruling z = -1: E0 arc x < XA of one sheet <-> E1 arc
        x > XD of the other, i <-> nu-1-i;
      * cell-to-cell rulings: the E1 arc x < XC (z = 0) of cell k
        <-> the E0 arc x > XB (z = -2) of cell k+1's other sheet."""
    xs, ys, X = genus1helicoid_sheet(r1, nu, nv, K)
    nu2, nv2 = X.shape[:2]
    U, Vv = np.meshgrid((xs - xs[0]) / (xs[-1] - xs[0]), ys / np.pi,
                        indexing='ij')
    uv0 = np.stack([U, Vv], axis=-1).reshape(-1, 2)
    Rz = np.array([-1.0, -1.0, 1.0])
    zoff = -(storeys - 1)                      # center the stack
    sheets, flips = [], []
    for s in range(storeys):
        off = np.array([0.0, 0.0, 2.0 * s + zoff])
        sheets.append(X + off)
        flips.append(False)
        sheets.append(X * Rz + off)
        flips.append(True)
    V = np.concatenate([S.reshape(-1, 3) for S in sheets], axis=0)
    uv = np.concatenate([uv0] * len(sheets), axis=0)
    quads = []
    for k, fl in enumerate(flips):
        b = k * nu2 * nv2
        for i in range(nu2 - 1):
            for j in range(nv2 - 1):
                a = b + i * nv2 + j
                c = b + (i + 1) * nv2 + j
                q = (a, c, c + 1, a + 1)
                # the rotated sheets get reversed winding so the welded
                # surface is consistently oriented
                quads.append(q[::-1] if fl else q)

    iA = int(np.argmin(np.abs(xs - _G1H_XA)))
    iB = int(np.argmin(np.abs(xs - _G1H_XB)))
    iC = int(np.argmin(np.abs(xs - _G1H_XC)))
    iD = int(np.argmin(np.abs(xs - _G1H_XD)))

    def gid(sheet, i, j):
        return sheet * nu2 * nv2 + i * nv2 + j

    pairs = []
    for kk in range(storeys):
        p, r = 2 * kk, 2 * kk + 1
        for i in range(iA, iB + 1):            # axis segment on E0
            pairs.append((gid(p, i, 0), gid(r, i, 0)))
        for i in range(iC, iD + 1):            # axis segment on E1
            pairs.append((gid(p, i, nv2 - 1), gid(r, i, nv2 - 1)))
        for i in range(0, iA + 1):             # in-cell ruling z = -1
            pairs.append((gid(p, i, 0), gid(r, nu2 - 1 - i, nv2 - 1)))
            pairs.append((gid(r, i, 0), gid(p, nu2 - 1 - i, nv2 - 1)))
        if kk + 1 < storeys:                   # cell-to-cell rulings
            p2, r2 = 2 * (kk + 1), 2 * (kk + 1) + 1
            for i in range(0, iC + 1):
                pairs.append((gid(p, i, nv2 - 1),
                              gid(r2, nu2 - 1 - i, 0)))
                pairs.append((gid(r, i, nv2 - 1),
                              gid(p2, nu2 - 1 - i, 0)))
    Vw, qw, first = _g1h_weld_pairs(V, quads, pairs)
    return Vw, qw, uv[first]


def genus1helicoid_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """MESH_PARAM builder: finished (V, quads, uv), fit to the 2 m
    cube.  order = number of translational periods (= handles);
    radius sets how far the two helicoidal ends flare (the strip
    truncation)."""
    p = spec['p_from'](order, radius)
    storeys = p['storeys']
    r1 = -(1.7 + 0.8 * float(np.clip(radius / 1.2, 0.6, 2.0)))
    pnu = int(np.clip(nu * 1.8, 90, 240))
    pnv = int(np.clip(nv * 0.85, 36, 96))
    V, quads, uv = genus1helicoid_assemble(storeys, r1, pnu, pnv)
    tk = _toolkit()
    V = tk._center_fit(V, scale, V)
    return V, quads, uv


# ==========================================================================
# Symmetrized Chen-Gackstatter towers (k-fold symmetry, D_kd assembly)
# ==========================================================================
# The k-fold-symmetric continuations of the Chen-Gackstatter surface: for
# each symmetry order k these are complete minimal surfaces with ONE
# Enneper-type end and dihedral-antiprismatic symmetry D_kd (order 4k),
# living on the k-fold cyclic cover of the sphere branched over the real
# branch values {0, +-r_1, ..., +-r_m, inf}.  Unified Weierstrass data
# (e = (k-1)/k throughout, dh = dz so x3 = Re z):
#   * genus k-1 tower:    g = rho z^(-e) (1 - z^2)^e
#     with the CLOSED-FORM Lopez-Ros constant (a Gamma-quotient; equal to
#     Weber's  1/Sqrt[4^e G(3/2-e/2) G(1+e/2) / (G(1-e/2) G((3+e)/2))]):
#       rho^2 = Int_0^1 |g/rho|^-1 dz / Int_0^1 |g/rho| dz
#             = G((1+e)/2) G(1-e) G((3+e)/2)
#               / ( G((3-e)/2) G((1-e)/2) G(1+e) );
#     k = 2 is the classical Chen-Gackstatter torus, k = 4 gives GENUS 3
#     (the genus the D2d normalization of cg_higher_* cannot reach).
#   * genus 2(k-1) tower: g = rho z^e (1-z^2)^(-e) (1-(z/a)^2)^e
#   * genus 3(k-1) tower: g = rho z^e (1-z^2)^(-e) (1-(z/a)^2)^e
#                             (1-(z/b)^2)^(-e)
#     where a (resp. a, b) solve the 1-D (2-D) period problem -- numeric
#     values from Weber's notebooks (see _SYMMCG_G2N / _SYMMCG_G3K); rho
#     then follows from the same per-segment period ratio, and the
#     surviving per-segment consistency (every real segment must yield
#     the SAME rho) is the self-test's validation of those constants.
#
# Symmetry / meshing scheme (generalizes the D2d = D_2d octant assembly
# of cg_higher_* from order 8 to order 4k): the QUARTER z-plane
# {Re z >= 0, Im z >= 0} is one 1/(4k) fundamental piece of the surface.
#   * the real intervals between consecutive branch values map to planar
#     geodesics in VERTICAL mirror planes through the axis; crossing the
#     branch value r_j rotates the plane by -pi b_j (the branch of
#     (1-(z/r_j)^2)^(b_j) continued from Im z > 0), so segment s lies in
#     the plane with polar angle psi_s = -pi sum_{j<s} b_j;
#   * the imaginary axis maps to a horizontal 2-fold axis in x3 = 0 at
#     polar angle pi (1 + a0) / 2  (a0 = the z-exponent of g);
#   * every branch value r_j (and z = 0) maps ONTO the vertical axis at
#     height r_j -- the branch images (0, 0, r_j) thread the axis;
#   * the group is generated by the rotoreflection
#     S = Rz(pi (1 + a0)) diag(1, 1, -1)  with  S^2 = Rz(2 pi / k) (the
#     deck rotation g -> e^{2 pi i / k} g) and the mirror
#     M = diag(1, -1, 1) (z -> conj z): 2k holomorphic frames S^j keep
#     the face winding, 2k antiholomorphic frames S^j M reverse it.  For
#     k = 2 (a0 = -1/2) S is exactly the S4 rotoreflection of the D2d
#     octant assembler.
# The WE forms are integrated along radial rays with one continuous
# branch of g (per-factor angle unwrapping; the theta = 0 ray uses the
# exact piecewise-constant branch with exact offsets to the branch
# values, so the algebraic singularities of |g|^{+-1} at 0 and r_j are
# integrated on strongly graded subgrids in the substitution variable,
# where the integrand is smooth).  Boundary vertices are snapped exactly
# onto their symmetry elements, the piece is trimmed by a sphere about
# the origin past the outermost branch images, orbited under all 4k
# frames and welded by boundary coincidence -- the result is exactly
# Euler characteristic chi = 1 - 2 genus (one boundary loop), gated by
# the self-tests below.
#
# References:
#   C. C. Chen, F. Gackstatter, "Elliptische und hyperelliptische
#     Funktionen und vollstaendige Minimalflaechen vom Enneperschen
#     Typ", Math. Ann. 259 (1982) -- the k = 2 surface;
#   H. Karcher, "Construction of minimal surfaces", Univ. of Tokyo
#     lecture notes (1989) -- the symmetrization method and the k-fold
#     towers;
#   E. C. Thayer, "Higher-genus Chen-Gackstatter surfaces and the
#     Weierstrass representation for surfaces of infinite genus",
#     Experiment. Math. 4 (1995);
#   M. Weber, https://minimalsurfaces.blog/ (repository,
#     "Symmetrized Chen-Gackstatter" pages) -- the data tables and the
#     closed-form rho this implementation follows.

# Period solutions k -> (a, rho): branch value a and the Lopez-Ros
# constant rho in THIS normalization (g2n factor (1-(z/a)^2)^e; Weber's
# notebooks use (a^2-z^2)^e, i.e. rho_nb = rho / a^{2e}).  Seeded from
# the values harvested off minimalsurfaces.blog and re-solved here to
# full double precision (secant on the per-segment period-ratio
# compatibility; the seeds agree to their published ~1e-4 precision,
# and symmcg_rho_solve in the self-test re-derives rho from scratch).
_SYMMCG_G2N = {3: (1.7168384042293399, 1.422067467716003),
               4: (1.7196230274467104, 1.4905575097998258),
               7: (1.7240865072638545, 1.5865591121925635),
               12: (1.727085736152026, 1.644449176251574)}
# k -> (a, b, rho): the 2-D period problem (2-D Newton on the three-way
# per-segment compatibility, seeded from Weber's (a, b) values); rho has
# no closed form -- the stored value is the common per-segment ratio
_SYMMCG_G3K = {2: (2.328309968067604, 3.1051967096086575,
                   1.228240310564583),
               3: (2.3294796517339766, 3.11368463663408,
                   1.3120297016852307),
               4: (2.3303049378429166, 3.1194657181873082,
                   1.3548397561984624),
               5: (2.330902276725775, 3.123554235422156,
                   1.380698302281986),
               7: (2.3316997357092446, 3.1288985424668323,
                   1.4103169367659762)}

_SYMMCG_RHO_CACHE = {}


def _symmcg_data(tower, k):
    """(a0, [(root, b), ...], rho) with g = rho z^a0 prod (1-(z/root)^2)^b;
    rho None means 'solve numerically' (see symmcg_rho_solve)."""
    e = (k - 1.0) / k
    G = math.gamma
    if tower == 'gn':
        rho = math.sqrt(G((1 + e) / 2) * G(1 - e) * G((3 + e) / 2)
                        / (G((3 - e) / 2) * G((1 - e) / 2) * G(1 + e)))
        return -e, [(1.0, e)], rho
    if tower == 'g2n':
        a, rho = _SYMMCG_G2N[k]
        return e, [(1.0, -e), (a, e)], rho
    if tower == 'g3k':
        a, b, rho = _SYMMCG_G3K[k]
        return e, [(1.0, -e), (a, e), (b, -e)], rho
    raise KeyError(f"symmcg tower {tower!r}")


def _symmcg_abs_logG(a0, factors, r, dL, dR, segA, segB):
    """log |g / rho| on a real segment; dL / dR are EXACT offsets to the
    segment endpoints (no cancellation at the branch values)."""
    with np.errstate(divide='ignore'):
        out = a0 * np.log(np.abs(r))
        for (rt, b) in factors:
            if abs(rt - segA) < 1e-12:
                t = np.abs(dL) / rt
            elif abs(rt - segB) < 1e-12:
                t = np.abs(dR) / rt
            else:
                t = np.abs(1.0 - r / rt)
            out = out + b * (np.log(t) + np.log1p(r / rt))
    return out


def _symmcg_seg_int(logf, A, B, eL, eR, n=3000):
    """Integral of exp(logf(r, dL, dR)) over [A, B] with known algebraic
    endpoint exponents |eL|, |eR| (0 = regular end): midpoint split, each
    half integrated in the graded variable u (r = end +- D u^p with
    p = 2/(1-e), where the integrand is smooth) by trapezoid."""
    total = 0.0
    M = 0.5 * (A + B)
    for lo, hi, ee, from_left in ((A, M, eL, True), (M, B, eR, False)):
        D = hi - lo
        p = 2.0 / max(1e-9, 1.0 - min(ee, 0.95)) if ee > 0 else 1.0
        u = np.linspace(0.0, 1.0, n + 1)
        off = D * u ** p
        if from_left:
            r = lo + off
            dL, dR = off + (lo - A), (B - lo) - off
        else:
            r = hi - off
            dL, dR = (hi - A) - off, off + (B - hi)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            f = np.exp(logf(r, dL, dR)) * D * p * u ** (p - 1.0)
        f = np.where(np.isfinite(f), f, 0.0)
        total += float(np.trapezoid(f, u) if hasattr(np, 'trapezoid')
                       else np.trapz(f, u))
    return total


def symmcg_rho_solve(tower, k):
    """Per-segment rho estimates: on every real segment between
    consecutive branch values the period condition reads
    rho^2 = Int |g/rho|^-1 / Int |g/rho|; all segments must agree (that
    IS the period problem -- the spread validates the (a, b) data).
    Returns (rho_from_segment_0, [estimates...])."""
    key = (tower, k)
    if key in _SYMMCG_RHO_CACHE:
        return _SYMMCG_RHO_CACHE[key]
    a0, factors, _ = _symmcg_data(tower, k)
    bounds = [0.0] + [rt for rt, _ in factors]
    est = []
    for s in range(len(bounds) - 1):
        A, B = bounds[s], bounds[s + 1]
        eL = abs(a0) if s == 0 else abs(factors[s - 1][1])
        eR = abs(factors[s][1])
        Ig = _symmcg_seg_int(
            lambda r, dL, dR: _symmcg_abs_logG(a0, factors, r, dL, dR,
                                               A, B), A, B, eL, eR)
        Iq = _symmcg_seg_int(
            lambda r, dL, dR: -_symmcg_abs_logG(a0, factors, r, dL, dR,
                                                A, B), A, B, eL, eR)
        est.append(math.sqrt(Iq / Ig))
    _SYMMCG_RHO_CACHE[key] = (est[0], est)
    return _SYMMCG_RHO_CACHE[key]


def _symmcg_radial_grids(roots, rmax, nu, e_by_end, msub=8, msing=100,
                         pmesh=1.7):
    """Coarse mesh nodes + fine integration nodes with per-interval
    trapezoid weights: inside the graded blocks adjacent to an algebraic
    singularity the weights are the u-substitution Jacobian weights (the
    integrand is smooth in u), elsewhere plain dr/2.  Returns
    (r_coarse, r_fine, coarse_idx, sing_id, sing_off, w_lo, w_hi); the
    ray integral is cumsum(w_lo f[:-1] + w_hi f[1:])."""
    bounds = [0.0] + list(roots) + [rmax]
    nseg = len(bounds) - 1
    lens = np.diff(np.array(bounds))
    w = np.sqrt(lens)
    counts = np.maximum(10, np.round(nu * w / w.sum()).astype(int))
    r_c, r_f, sid, soff = [0.0], [0.0], [-1], [0.0]
    cidx = [0]
    w_lo, w_hi = [], []

    def root_id(x):
        for i, rt in enumerate(roots):
            if abs(x - rt) < 1e-12:
                return i
        return -1

    for s in range(nseg):
        A, B = bounds[s], bounds[s + 1]
        n = int(counts[s])
        last = s == nseg - 1
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        if last:
            tt = t ** pmesh              # dense only at the branch end
        else:                            # dense toward both branch ends
            tt = np.where(t < 0.5, 0.5 * (2 * t) ** pmesh,
                          1.0 - 0.5 * (2 * (1 - t)) ** pmesh)
        nodes = A + (B - A) * tt
        nodes[-1] = B
        prev = A
        for ci in range(n):
            hi = nodes[ci]
            singL = (ci == 0 and A in e_by_end)
            singR = (ci == n - 1 and not last and B in e_by_end)
            if singL:                    # graded from the left end A
                p = 2.0 / max(1e-9, 1.0 - min(e_by_end[A], 0.95))
                ms = int(msing * max(1.0, p / 8.0))   # denser as e -> 1
                D = hi - A
                u = np.linspace(0.0, 1.0, ms + 1)
                off = D * u ** p
                J = D * p * u ** (p - 1.0)
                du = 1.0 / ms
                rf = (A + off)[1:]
                rf[-1] = hi
                sid_i = np.full(len(rf), root_id(A))
                soff_i = off[1:]
                w_lo.extend((0.5 * du * J[:-1]).tolist())
                w_hi.extend((0.5 * du * J[1:]).tolist())
            elif singR:                  # graded toward the right end B
                p = 2.0 / max(1e-9, 1.0 - min(e_by_end[B], 0.95))
                ms = int(msing * max(1.0, p / 8.0))
                D = B - prev
                v = np.linspace(0.0, 1.0, ms + 1)
                offB = D * (1.0 - v) ** p
                J = D * p * (1.0 - v) ** (p - 1.0)
                dv = 1.0 / ms
                rf = (B - offB)[1:]
                rf[-1] = B
                sid_i = np.full(len(rf), root_id(B))
                soff_i = -offB[1:]
                w_lo.extend((0.5 * dv * J[:-1]).tolist())
                w_hi.extend((0.5 * dv * J[1:]).tolist())
            else:
                # inner cell: if its segment half ends at an algebraic
                # singularity, place the subnodes log-uniform in the
                # distance d to that end and integrate in the log
                # variable (f d varies like d^{1-e} there -- nearly
                # flat), else plain uniform trapezoid
                t = np.linspace(0.0, 1.0, msub + 1)
                left_half = (prev - A) <= (B - hi)
                if left_half and A in e_by_end and prev > A:
                    dlo, dhi = prev - A, hi - A
                    d = dlo * (dhi / dlo) ** t
                    rf_full = A + d
                    J = d * math.log(dhi / dlo)
                elif (not left_half and not last and B in e_by_end
                      and hi < B):
                    dtop, dbot = B - prev, B - hi
                    d = dtop * (dbot / dtop) ** t
                    rf_full = B - d
                    J = d * abs(math.log(dbot / dtop))
                else:
                    rf_full = prev + (hi - prev) * t
                    J = None
                rf = rf_full[1:].copy()
                rf[-1] = hi
                sid_i = np.full(len(rf), -1)
                soff_i = np.zeros(len(rf))
                if J is None:
                    h = 0.5 * (hi - prev) / msub
                    w_lo.extend([h] * msub)
                    w_hi.extend([h] * msub)
                else:
                    dt = 1.0 / msub
                    w_lo.extend((0.5 * dt * J[:-1]).tolist())
                    w_hi.extend((0.5 * dt * J[1:]).tolist())
            r_f.extend(rf.tolist())
            sid.extend(sid_i.tolist())
            soff.extend(soff_i.tolist())
            cidx.append(len(r_f) - 1)
            r_c.append(hi)
            prev = hi
    return (np.array(r_c), np.array(r_f), np.array(cidx),
            np.array(sid, dtype=int), np.array(soff),
            np.array(w_lo), np.array(w_hi))


def _symmcg_phi_ray(a0, factors, rho, r_f, sid, soff, theta):
    """(nf, 3) complex integrand phi e^{i theta} along one radial ray
    with a continuous branch of g (per-factor angle unwrap; the theta = 0
    ray uses the exact piecewise branch: angle 0 before each branch
    value, -pi after -- the limit from Im z > 0 -- and exact offsets for
    |1 - r/root|).  Non-finite entries (the branch nodes) are zeroed;
    their quadrature weight is negligible on the graded grid."""
    nf = len(r_f)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        lmag = math.log(rho) + a0 * np.log(r_f)
        ang = a0 * theta * np.ones(nf)
        if theta == 0.0:
            for fi, (rt, b) in enumerate(factors):
                t = np.abs(1.0 - r_f / rt)
                m = sid == fi
                t[m] = np.abs(soff[m]) / rt
                lmag = lmag + b * (np.log(t) + np.log1p(r_f / rt))
                # which side of the root: for the graded-block nodes the
                # stored offset sign is EXACT (their r may round onto the
                # root itself -- r = rt + 1e-40 == rt in doubles -- and a
                # plain r > rt test would put them on the wrong branch)
                side = r_f > rt
                side[m] = soff[m] > 0
                ang = ang + b * np.where(side, -math.pi, 0.0)
        else:
            z = r_f * np.exp(1j * theta)
            for (rt, b) in factors:
                P = 1.0 - (z / rt) ** 2
                lmag = lmag + b * np.log(np.abs(P))
                ang = ang + b * np.unwrap(np.angle(P))
        g = np.exp(lmag + 1j * ang)
        ph = np.stack([0.5 * (1.0 / g - g),
                       0.5j * (1.0 / g + g),
                       np.ones_like(g)], axis=-1) * np.exp(1j * theta)
    return np.where(np.isfinite(ph), ph, 0.0)


def _symmcg_probe_rmax(a0, factors, rho, target):
    """Parameter radius at which the image reaches `target` (probed on
    the mid ray theta = pi/4), so the sphere trim never runs dry."""
    r = np.linspace(0.0, 80.0, 200001)
    ph = _symmcg_phi_ray(a0, factors, rho, r, np.full(len(r), -1),
                         np.zeros(len(r)), 0.25 * math.pi)
    dr = np.diff(r)[:, None]
    F = np.concatenate([np.zeros((1, 3), complex),
                        np.cumsum(0.5 * (ph[1:] + ph[:-1]) * dr, axis=0)],
                       axis=0)
    rad = np.linalg.norm(np.real(F), axis=1)
    ok = np.nonzero(rad >= target)[0]
    return float(r[ok[0]]) if len(ok) else 80.0


def _symmcg_rend0(tower, k):
    """Default sphere-trim radius (raw units): past the outermost branch
    images (heights 1 / a / b) AND past the handle lobes (which widen
    with k -- at gn k = 8 they reach r ~ 2.5), with room for the flare."""
    return {'gn': 2.6, 'g2n': 3.9, 'g3k': 5.6}[tower]


def _symmcg_piece(tower, k, nu, nv, Rend):
    """Mesh the 1/(4k) fundamental piece (quarter plane, radial rays).
    Returns (V, faces, boundary-classification, per-vertex uv, diag)."""
    a0, factors, rho = _symmcg_data(tower, k)
    if rho is None:
        rho = symmcg_rho_solve(tower, k)[0]
    roots = [rt for rt, _ in factors]
    e_by_end = {0.0: abs(a0)}
    for rt, b in factors:
        e_by_end[rt] = abs(b)
    rmax = _symmcg_probe_rmax(a0, factors, rho, 1.45 * Rend)
    r_c, r_f, cidx, sid, soff, w_lo, w_hi = _symmcg_radial_grids(
        roots, rmax, nu, e_by_end)
    th = np.linspace(0.0, 0.5 * math.pi, nv)
    F = np.zeros((len(r_c), nv, 3))
    for j, t in enumerate(th):
        ph = _symmcg_phi_ray(a0, factors, rho, r_f, sid, soff, float(t))
        contrib = w_lo[:, None] * ph[:-1] + w_hi[:, None] * ph[1:]
        Ff = np.concatenate([np.zeros((1, 3), complex),
                             np.cumsum(contrib, axis=0)], axis=0)
        F[:, j, :] = np.real(Ff[cidx])
    ncr = len(r_c)

    def vid(i, j):                       # row 0 collapses to the center
        return 0 if i == 0 else (i - 1) * nv + j + 1

    V = np.concatenate([F[:1, 0, :], F[1:].reshape(-1, 3)], axis=0)
    UV = np.zeros((len(V), 2))
    UV[1:, 0] = np.tile(th / (0.5 * math.pi), ncr - 1)
    UV[1:, 1] = np.repeat(r_c[1:] / r_c[-1], nv)
    faces = []
    for j in range(nv - 1):
        faces.append((0, vid(1, j), vid(1, j + 1)))
    for i in range(1, ncr - 1):
        for j in range(nv - 1):
            faces.append((vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)))
    root_ci = [int(np.argmin(np.abs(r_c - rt))) for rt in roots]
    b = {'seg': {}, 'branch': {}, 'branch_z': {}, 'center': 0}
    lastci = 0
    for s in range(len(roots) + 1):
        hi_ci = root_ci[s] if s < len(roots) else ncr - 1
        stop = hi_ci + (1 if s == len(roots) else 0)
        b['seg'][s] = np.array([vid(i, 0)
                                for i in range(lastci + 1, stop)],
                               dtype=np.int64)
        if s < len(roots):
            b['branch'][s] = vid(hi_ci, 0)
            b['branch_z'][s] = roots[s]
        lastci = hi_ci
    b['seam'] = np.array([vid(i, nv - 1) for i in range(1, ncr)],
                         dtype=np.int64)
    psis = [0.0]                         # mirror-plane angle per segment
    for (rt, bb) in factors:
        psis.append(psis[-1] - math.pi * bb)
    b['psi'] = psis
    b['psi_seam'] = 0.5 * math.pi * (1.0 + a0)
    return V, faces, b, UV, {'rmax': rmax, 'rho': rho, 'a0': a0}


def _symmcg_snap(V, b):
    """Snap every boundary vertex exactly onto its symmetry element.
    Returns the PRE-snap deviations -- the numeric period-closure
    residuals (they must be small; the snap only removes quadrature
    noise, it cannot fix wrong period data)."""
    res = {}
    for s, idx in b['seg'].items():
        if len(idx) == 0:
            continue
        psi = b['psi'][s]
        u = np.array([math.cos(psi), math.sin(psi)])
        xy = V[idx][:, :2]
        t = xy @ u
        perp = xy - t[:, None] * u[None, :]
        res[f'seg{s}'] = float(np.max(np.linalg.norm(perp, axis=1)))
        V[idx, 0] = t * u[0]
        V[idx, 1] = t * u[1]
    for s, vi in b['branch'].items():
        res[f'branch{s}'] = float(np.hypot(V[vi, 0], V[vi, 1]))
        V[vi] = (0.0, 0.0, b['branch_z'][s])
    u = np.array([math.cos(b['psi_seam']), math.sin(b['psi_seam'])])
    idx = b['seam']
    xy = V[idx][:, :2]
    t = xy @ u
    perp = xy - t[:, None] * u[None, :]
    res['seam'] = float(np.max(np.hypot(np.linalg.norm(perp, axis=1),
                                        V[idx, 2])))
    V[idx, 0] = t * u[0]
    V[idx, 1] = t * u[1]
    V[idx, 2] = 0.0
    V[b['center']] = (0.0, 0.0, 0.0)
    return V, res


def _symmcg_frames(k, a0):
    """The 4k isometries of D_kd: 2k holomorphic frames S^j (winding
    kept -- S is a rotoreflection but holomorphic, like S4 in the D2d
    assembler) and 2k antiholomorphic frames S^j M (winding reversed)."""
    ang = math.pi * (1.0 + a0)
    c, s = math.cos(ang), math.sin(ang)
    S = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, -1.0]])
    My = np.diag([1.0, -1.0, 1.0])
    holo = []
    M = np.eye(3)
    for _ in range(2 * k):
        holo.append(M)
        M = S @ M
    return holo + [Mk @ My for Mk in holo]


def symmcg_assemble(tower, k, nu, nv, Rend=None):
    """Watertight D_kd assembly: snap, sphere-trim, orbit the 1/(4k)
    piece under all 4k frames, weld by boundary coincidence.  Returns
    (V, faces, uv, diag) of the largest component; diag carries the
    period residuals."""
    if Rend is None:
        Rend = _symmcg_rend0(tower, k)
    V0, faces0, b, UV0, diag = _symmcg_piece(tower, k, nu, nv, Rend)
    V0, res = _symmcg_snap(V0.copy(), b)
    diag['res'] = res
    bmask = np.zeros(len(V0), dtype=bool)
    for idx in b['seg'].values():
        bmask[idx] = True
    bmask[b['seam']] = True
    bmask[list(b['branch'].values())] = True
    bmask[b['center']] = True
    keep = np.linalg.norm(V0, axis=1) <= Rend
    faces0 = [f for f in faces0 if all(keep[i] for i in f)]
    used = sorted(set(i for f in faces0 for i in f))
    rmv = {v: i for i, v in enumerate(used)}
    V0, UV0, bmask = V0[used], UV0[used], bmask[used]
    faces0 = [tuple(rmv[i] for i in f) for f in faces0]
    nV = len(V0)
    frames = _symmcg_frames(k, diag['a0'])
    Vp, Fp = [], []
    for fr, M in enumerate(frames):
        Vp.append(V0 @ M.T)
        rev = fr >= 2 * k                # antiholomorphic copies flip
        for f in faces0:
            ff = tuple(int(x) + fr * nV for x in f)
            Fp.append(ff[::-1] if rev else ff)
    V = np.concatenate(Vp)
    UVall = np.tile(UV0, (len(frames), 1))
    N = len(V)
    tol = 1e-6                           # coincidence weld on boundary
    parent = np.arange(N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bidx = np.nonzero(np.tile(bmask, len(frames)))[0]
    key = np.floor(V[bidx] / tol + 0.5).astype(np.int64)
    H = {}
    for t, i in enumerate(bidx):
        H.setdefault((int(key[t, 0]), int(key[t, 1]), int(key[t, 2])),
                     []).append(int(i))
    for t, i in enumerate(bidx):
        k0 = key[t]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in H.get((int(k0[0]) + dx, int(k0[1]) + dy,
                                    int(k0[2]) + dz), ()):
                        if j > i and np.linalg.norm(V[j] - V[i]) < tol:
                            ra, rb = find(int(i)), find(j)
                            if ra != rb:
                                parent[ra] = rb
    rts = np.array([find(a) for a in range(N)])
    uniq, inv = np.unique(rts, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    UVw = np.zeros((len(uniq), 2))
    cnt = np.zeros(len(uniq))
    np.add.at(Vw, inv, V)
    np.add.at(UVw, inv, UVall)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    UVw /= cnt[:, None]
    F = []
    for f in Fp:
        gg = [int(inv[i]) for i in f]
        h = [gg[0]]
        for t in range(1, len(gg)):
            if gg[t] != h[-1]:
                h.append(gg[t])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    parent2 = np.arange(len(Vw))         # largest face-connected comp

    def find2(a):
        while parent2[a] != a:
            parent2[a] = parent2[parent2[a]]
            a = parent2[a]
        return a

    for f in F:
        for i in range(1, len(f)):
            ra, rb = find2(f[0]), find2(f[i])
            if ra != rb:
                parent2[ra] = rb
    from collections import Counter
    sizes = Counter(find2(f[0]) for f in F)
    root = sizes.most_common(1)[0][0]
    F = [f for f in F if find2(f[0]) == root]
    used = sorted(set(i for f in F for i in f))
    rmv = {v: i for i, v in enumerate(used)}
    return (Vw[used], [tuple(rmv[i] for i in f) for f in F],
            UVw[used], diag)


def symmcg_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """MESH_PARAM builder: finished (V, quads, uv) fit to the 2 m cube.
    The sphere trim must clear the handle lobes or the largest-component
    filter would keep only the disconnected end flare, so the build
    verifies the exact Euler characteristic and widens the trim until
    the full tower survives (chi = 1 - 2 genus gates it honestly)."""
    p = spec['p_from'](order, radius)
    tower, k = p['tower'], p['k']
    genus = {'gn': k - 1, 'g2n': 2 * (k - 1), 'g3k': 3 * (k - 1)}[tower]
    pnu = int(np.clip(nu * 1.4, 70, 200))
    pnv = int(np.clip(nv * 0.45, 16, 40))
    R0 = _symmcg_rend0(tower, k)
    Rend = float(np.clip(R0 * radius / 1.2, 0.7 * R0, 2.2 * R0))
    for _ in range(4):
        V, quads, uv, _diag = symmcg_assemble(tower, k, pnu, pnv, Rend)
        edges = set()
        for f in quads:
            m = len(f)
            for t in range(m):
                a, b = f[t], f[(t + 1) % m]
                edges.add((a, b) if a < b else (b, a))
        if len(V) - len(edges) + len(quads) == 1 - 2 * genus:
            break
        Rend *= 1.22                   # trim cut a handle: widen and retry
    tk = _toolkit()
    V = tk._smooth_boundary(V, quads, iters=6)
    V = tk._center_fit(V, scale, V)
    return V, quads, uv


# ==========================================================================
# Callahan-Hoffman-Meeks singly periodic surface (k = 1) -- chm_* helpers
# ==========================================================================
# The singly periodic analog of the Costa surface: an embedded minimal
# surface invariant under a vertical translation, with TWO horizontal
# planar ends per translational period (so infinitely many ends in all).
# The quotient by the translation has genus 2k + 1 = 3 (the CHM theorem;
# this is the k = 1 member, Weber's CHM-(1,1) notebook), so one period
# meshes at exactly chi = -6 with the two end punctures -- the mesh gate
# below MEASURES that.  (The harvest metadata's "genus 2 / chi = -4"
# annotation is off by one handle; the built surface, the notebook and
# the theorem all agree on genus 3.)  The surface is also invariant
# under the half-period screw T_{2d} rot90 and the vertical glide
# T_{2d} swap(x, y); neither is a pure translation, so the primitive
# translation is the full 4d (verified numerically: swap(x, y) alone is
# NOT a symmetry -- probe points map ~0.1 off the surface).
#
# Weierstrass data (M. Weber's CHM-(1,1) notebook; branch values on the
# real axis at 0, +-1, +-a):
#     G    = sqrt(x) (x^2-a^2)^{3/4} / (rho (x^2-1)^{1/4})
#     phi1 = 1 / (sqrt(x) (x^2-1)^{1/4} (x^2-a^2)^{5/4})
#     phi2 = sqrt(x) (x^2-a^2)^{1/4} / (x^2-1)^{3/4}
#     om1  = (rho phi1 - phi2/rho) / 2,   om2 = i (rho phi1 + phi2/rho) / 2
#     om3  = 1 / (sqrt(x^2-1) sqrt(x^2-a^2))
# with the two solved period constants (harvested verbatim from the
# notebook's FindRoot; rho independently re-verified here to 1e-12 from
# its closure condition rho^2 = int_0^1|phi2| / int_0^1|phi1|):
#     a   = 1.31922381870184635
#     rho = 1.22370848596689185
# om1^2 + om2^2 + om3^2 = 0 holds exactly (checked in the self-test).
#
# Fundamental patch.  Following the notebook's substitution, the patch is
# integrated on the strip chart  w = u + i t,  z(w) = sqrt(a^2 + e^w)
# (so z^2 - a^2 = e^w exactly), u in [umin, umax], t in [0, pi], with
# umax = log(a sqrt(a^2-1)).  On it every principal fractional power is
# single-valued (z stays in the first quadrant, z^2-1 and z^2-a^2 in the
# closed upper half plane), so no branch cuts are crossed.  The patch
# boundary consists of symmetry curves only:
#   t = 0            planar geodesic in the vertical mirror plane y = 0
#   t = pi, u > xa   planar geodesic in the vertical mirror plane x = 0
#   t = pi, u < xa   STRAIGHT line, direction (1,1,0), at height -d
#                    (xa = log(a^2-1); z=1 maps to the axis point (0,0,-d))
#   u = umax         planar geodesic in the horizontal mirror plane z = 0
#   u = umin         the end trim: u -> -inf is the planar end at z = a
#                    (asymptotic height exactly -d; the flare radius grows
#                    as ~0.87 e^{-u/4}, so umin sizes the trimmed end disk)
# with d = 0.7288482 (the measured half-spacing; the real-axis elliptic
# integral int_0^1 dx/sqrt((1-x^2)(a^2-x^2)) equals 2d to 7 digits).
#
# Assembly.  16 isometries tile ONE translational period from the patch:
#     {E, sigma_h} x {E, Mx} x {E, My} x {E, R11}
# where sigma_h reflects across the horizontal plane z = 0, Mx / My across
# the vertical planes x = 0 / y = 0, and R11 is the 180-degree rotation
# about the straight line {x = y, z = -d}.  The period translation is
# (0, 0, 4d); `storeys` stacked periods weld vertex-exactly because the
# chunk's top boundary sigma_h Mx^b My^c R11 (arc) and the next chunk's
# bottom boundary T_{4d} Mx^b My^c R11 (arc) evaluate to bitwise-identical
# coordinates.  Boundary rows are snapped exactly onto their symmetry
# elements before tiling, so every seam welds by float equality.  (The
# harvested constants close the period problem to ~1.2e-4 absolute -- the
# residual of the notebook's FindRoot tolerance; the snap absorbs it.)
#
# References:
#   M. J. Callahan, D. Hoffman, W. H. Meeks III, "Embedded minimal
#     surfaces with an infinite number of ends", Invent. Math. 96 (1989)
#     459-505 -- the CHM_k family (this is k = 2);
#   D. Hoffman, W. H. Meeks III, "Minimal surfaces based on the catenoid",
#     Amer. Math. Monthly 97 (1990) -- exposition;
#   M. Weber, https://minimalsurfaces.blog/ (Callahan-Hoffman-Meeks
#     surfaces) and the CHM-(1,1) notebook -- the g / dh data, the strip
#     substitution and the solved constants
#     (research/msblog_harvest/singly_periodic.json).
# --------------------------------------------------------------------------

_CHMP_A = 1.31922381870184635        # branch point a (harvested)
_CHMP_RHO = 1.22370848596689185      # balance constant rho (harvested)
_CHMP_XA = math.log(_CHMP_A * _CHMP_A - 1.0)     # z = 1 at (xa, pi)
_CHMP_X0 = math.log(_CHMP_A * _CHMP_A)           # z = 0 at (x0, pi)
_CHMP_UMAX = 0.5 * (_CHMP_XA + _CHMP_X0)         # the arc edge


def chm_periodic_forms(w):
    """(om1, om2, om3) pulled back to the strip chart (times dz/dw =
    e^w / 2z).  With z^2 - a^2 = e^w the fractional powers of z^2 - a^2
    become exact exponentials, so the planar end u -> -inf is free of
    branch issues; the only strip singularity is the integrable
    (z^2-1)^{-3/4} corner at (xa, pi), the branch point z = 1."""
    a, rho = _CHMP_A, _CHMP_RHO
    ew = np.exp(w)
    z = np.sqrt(a * a + ew)                    # first quadrant (principal)
    zm1 = a * a - 1.0 + ew                     # z^2 - 1, closed UHP
    q1 = np.exp(-0.25 * w) / (2.0 * z ** 1.5 * zm1 ** 0.25)
    q2 = np.exp(1.25 * w) / (2.0 * np.sqrt(z) * zm1 ** 0.75)
    om3 = np.exp(0.5 * w) / (2.0 * z * np.sqrt(zm1))
    om1 = 0.5 * (rho * q1 - q2 / rho)
    om2 = 0.5j * (rho * q1 + q2 / rho)
    return om1, om2, om3


def _chmp_cluster(x0, h0, hmin=1e-10, ratio=0.2):
    """Geometric offsets h0*ratio^k down to hmin (largest first)."""
    offs = []
    h = h0
    while h > hmin:
        offs.append(h)
        h *= ratio
    return np.array(offs)


def _chmp_ugrid(nu, umin):
    """Ascending u grid: flare-uniform toward the end (uniform in the
    end's conformal radius e^{-u/4}), a uniform mid band, geometric
    clusters into the singular corner u = xa from BOTH sides (the
    (z^2-1)^{-3/4} branch point demands geometrically graded cells for
    the compound Gauss-Legendre row integrals to converge), and a
    uniform tail to the arc at umax.  xa is an exact grid node."""
    xa, umax = _CHMP_XA, _CHMP_UMAX
    nE = max(14, int(0.40 * nu))
    nM = max(6, int(0.16 * nu))
    nR = max(6, int(0.18 * nu))
    uf = -4.0 * np.log(np.linspace(math.exp(-0.25 * umin),
                                   math.exp(-0.25 * (xa - 0.8)), nE))
    um = np.linspace(xa - 0.8, xa - 0.16, nM + 1)[1:]
    cl = _chmp_cluster(xa, 0.16 * 0.2)
    left = xa - cl                              # ascending toward xa
    right = xa + cl[::-1]                       # ascending away from xa
    h0r = min(0.16, 0.45 * (umax - xa))
    ur = np.linspace(xa + h0r, umax, nR)
    u = np.concatenate([uf, um, left, [xa], right,
                        np.array([xa + h0r * 0.5]), ur])
    u = np.unique(u)
    return u[np.concatenate([[True], np.diff(u) > 1e-13])]


def _chmp_tgrid(nt):
    """t grid on [0, pi]: Chebyshev-clustered at both edges plus a
    geometric cluster into t = pi (the corner rows)."""
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0.0, 1.0, max(24, nt))))
    extra = math.pi - _chmp_cluster(0.0, 0.08, ratio=0.25)
    t = np.unique(np.concatenate([base, extra]))
    return t[np.concatenate([[True], np.diff(t) > 1e-13])]


_CHMP_GL = np.polynomial.legendre.leggauss(8)


def chm_periodic_patch(u, t):
    """Integrate the immersion over the strip grid.  Compound 8-point
    Gauss-Legendre per grid cell (nodes are strictly interior, so the
    singular corner node (xa, pi) is never evaluated): down/up the arc
    column u = umax from the base (umax, ~pi/2), then leftward along
    every row.  Returns X real (nu, nt, 3)."""
    xg, wg = _CHMP_GL
    nu, nt = len(u), len(t)
    du = np.diff(u)
    umid = 0.5 * (u[1:] + u[:-1])
    W = (umid[:, None, None] + 0.5 * du[:, None, None] * xg[None, None, :]
         + 1j * t[None, :, None])
    o1, o2, o3 = chm_periodic_forms(W)
    incU = np.stack(
        [np.sum(o1 * wg, axis=-1), np.sum(o2 * wg, axis=-1),
         np.sum(o3 * wg, axis=-1)], axis=-1) * (0.5 * du)[:, None, None]
    dt = np.diff(t)
    tmid = 0.5 * (t[1:] + t[:-1])
    Wc = u[-1] + 1j * (tmid[:, None] + 0.5 * dt[:, None] * xg[None, :])
    c1, c2, c3 = chm_periodic_forms(Wc)
    incC = np.stack(
        [np.sum(c1 * wg, axis=-1), np.sum(c2 * wg, axis=-1),
         np.sum(c3 * wg, axis=-1)], axis=-1) * (0.5j * dt)[:, None]
    Fc = np.zeros((nt, 3), complex)
    Fc[1:] = np.cumsum(incC, axis=0)
    Fc -= Fc[int(np.argmin(np.abs(t - 0.5 * math.pi)))]
    # F(i, j) = Fc(j) - sum_{cells k >= i} incU(k, j)
    S = np.zeros((nu, nt, 3), complex)
    S[:-1] = np.cumsum(incU[::-1], axis=0)[::-1]
    return np.real(Fc[None, :, :] - S)


def chm_periodic_snap(X, u):
    """Normalize and snap the patch boundary exactly onto its symmetry
    elements: y = 0 mirror (t = 0 row), x = 0 mirror (t = pi row past
    xa), z = 0 horizontal mirror (arc column), the straight line
    {x = y, z = -d} (t = pi row before xa) and the axis point (0,0,-d)
    at the z = 1 corner.  Returns (X, iL, d)."""
    iL = int(np.searchsorted(u, _CHMP_XA))
    X = X.copy()
    X[..., 1] -= np.median(X[:, 0, 1])            # y = 0 mirror
    X[..., 0] -= np.median(X[iL:, -1, 0])         # x = 0 mirror
    X[..., 2] -= np.median(X[-1, :, 2])           # z = 0 mirror (arc)
    X[:, 0, 1] = 0.0
    X[iL:, -1, 0] = 0.0
    X[-1, :, 2] = 0.0
    zL = float(np.median(X[:iL + 1, -1, 2]))
    d = -zL
    X[:iL + 1, -1, 2] = zL                        # the straight line...
    m = 0.5 * (X[:iL + 1, -1, 0] + X[:iL + 1, -1, 1])
    X[:iL + 1, -1, 0] = m                         # ...x = y exactly
    X[:iL + 1, -1, 1] = m
    X[iL, -1, :] = (0.0, 0.0, zL)                 # z = 1 -> axis point
    return X, iL, d


def _chmp_frames(d, periods):
    """The 16 * periods assembly isometries (M, tvec, parity), indexed
    (((sh*2 + bx)*2 + cy)*2 + e)*periods + k:  v -> sigma_h^sh Mx^bx
    My^cy R11^e v + (0, 0, 4dk).  parity drives face winding."""
    R11 = (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0]]), np.array([0.0, 0.0, -2.0 * d]))
    out = []
    for sh in (0, 1):
        for bx in (0, 1):
            for cy in (0, 1):
                D = np.diag([-1.0 if bx else 1.0, -1.0 if cy else 1.0,
                             -1.0 if sh else 1.0])
                for e in (0, 1):
                    if e:
                        M = D @ R11[0]
                        tv = D @ R11[1]
                    else:
                        M = D.copy()
                        tv = np.zeros(3)
                    # every generator is a Schwarz continuation (z -> zbar
                    # on the parameter domain, anti-conformal), so each --
                    # including the PROPER rotation R11 -- reverses the
                    # surface orientation: winding flips with the total
                    # generator count, not with det M.
                    par = (-1.0) ** (sh + bx + cy + e)
                    for k in range(periods):
                        out.append((M, tv + np.array(
                            [0.0, 0.0, 4.0 * d * k]), par))
    return out


def chm_periodic_seams(iL, nu, nt, periods):
    """The COMBINATORIAL weld: every seam pairs frame (sh, bx, cy, e, k)
    with one partner frame along one boundary index set, with identical
    within-patch indices (the gluing isometry fixes the seam pointwise).
    Derived from the generator algebra (R11 My = Mx R11, R11 Mx = My R11,
    R11 sigma_h = T_{-4d} sigma_h R11):
      t = 0 row   (y = 0 mirror):   toggle cy if e = 0 else bx
      x row       (x = 0 mirror):   toggle bx if e = 0 else cy
      L1 row      (2-fold line):    toggle e
      arc column  (z = 0 mirror):   toggle sh; same k if e = 0, else the
                  NEIGHBOR chunk k-1 (sh = 0) / k+1 (sh = 1) -- the
                  period interface (open at the stack's outer cuts).
    Returns [(patch-index array, frame A, frame B), ...]."""
    S = periods

    def fidx(sh, bx, cy, e, k):
        return (((sh * 2 + bx) * 2 + cy) * 2 + e) * S + k

    t0 = np.arange(nu) * nt
    xr = np.arange(iL, nu) * nt + (nt - 1)
    l1 = np.arange(0, iL + 1) * nt + (nt - 1)
    ac = (nu - 1) * nt + np.arange(nt)
    out = []
    for sh in (0, 1):
        for bx in (0, 1):
            for cy in (0, 1):
                for e in (0, 1):
                    for k in range(S):
                        me = fidx(sh, bx, cy, e, k)
                        out.append((t0, me, fidx(sh, bx, 1 - cy, e, k)
                                    if e == 0 else
                                    fidx(sh, 1 - bx, cy, e, k)))
                        out.append((xr, me, fidx(sh, 1 - bx, cy, e, k)
                                    if e == 0 else
                                    fidx(sh, bx, 1 - cy, e, k)))
                        out.append((l1, me, fidx(sh, bx, cy, 1 - e, k)))
                        k2 = k if e == 0 else (k - 1 if sh == 0 else k + 1)
                        if 0 <= k2 < S:
                            out.append((ac, me,
                                        fidx(1 - sh, bx, cy, e, k2)))
    return out


def chm_periodic_assemble(u, t, X, iL, d, periods):
    """Tile the snapped patch under the 16 * periods isometries and weld
    every seam COMBINATORIALLY (chm_periodic_seams knows each edge's
    partner frame exactly, so no floating-point coincidence matching is
    involved).  Returns (V, faces, uv) of the largest component -- one
    translational period per storey, the 2*periods planar-end rims and
    the 2 outer horizontal cuts left as clean open boundaries."""
    nu, nt = len(u), len(t)
    V0 = X.reshape(-1, 3)
    UV0 = np.stack(np.meshgrid(
        (u - u[0]) / (u[-1] - u[0]), t / math.pi, indexing='ij'),
        axis=-1).reshape(-1, 2)
    q0 = []
    for i in range(nu - 1):
        for j in range(nt - 1):
            q0.append((i * nt + j, i * nt + j + 1,
                       (i + 1) * nt + j + 1, (i + 1) * nt + j))
    frames = _chmp_frames(d, periods)
    nV = len(V0)
    Vp, Fp = [], []
    for fr, (M, tv, par) in enumerate(frames):
        Vp.append(V0 @ M.T + tv)
        off = fr * nV
        if par < 0:
            Fp.extend(tuple(off + i for i in f[::-1]) for f in q0)
        else:
            Fp.extend(tuple(off + i for i in f) for f in q0)
    V = np.concatenate(Vp, axis=0)
    UVall = np.tile(UV0, (len(frames), 1))
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for idx, fA, fB in chm_periodic_seams(iL, nu, nt, periods):
        if fA >= fB:                   # each seam appears from both sides
            continue
        for i in idx:
            ra, rb = find(fA * nV + int(i)), find(fB * nV + int(i))
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(a) for a in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    UVw = np.zeros((len(uniq), 2))
    cnt = np.zeros(len(uniq))
    np.add.at(Vw, inv, V)
    np.add.at(UVw, inv, UVall)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    UVw /= cnt[:, None]
    F = []
    for f in Fp:
        g = [int(inv[i]) for i in f]
        h = [g[0]]
        for s in range(1, len(g)):
            if g[s] != h[-1]:
                h.append(g[s])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    tk = _toolkit()
    Vu, F = tk._largest_component(np.hstack([Vw, UVw]), F)
    return Vu[:, :3], F, Vu[:, 3:]


def chm_periodic_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                      storeys=1):
    """MESH_PARAM builder: finished (V, faces, uv) of `storeys` stacked
    translational periods, fit to the 2 m cube.  radius (via p_from's
    umin) sizes the trimmed planar-end disks; order and theta unused."""
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    umin = float(p.get('umin', -5.0))
    S = int(np.clip(storeys, 1, 6))
    ug = _chmp_ugrid(int(np.clip(nu, 24, 220)), umin)
    tg = _chmp_tgrid(int(np.clip(int(0.85 * nv), 20, 170)))
    X = chm_periodic_patch(ug, tg)
    X, iL, d = chm_periodic_snap(X, ug)
    V, F, uv = chm_periodic_assemble(ug, tg, X, iL, d, S)
    tk = _toolkit()
    V = tk._center_fit(V, scale, V)
    return V, F, uv


# ==========================================================================
# Catenoid-Enneper (genus 2/3/4) and Costa-Wohlgemuth / Wohlgemuth
# higher-genus surfaces  (cwce_* block)
# ==========================================================================
# Two families of complete finite-total-curvature minimal surfaces on
# hyperelliptic / cyclic-cover curves, meshed watertight from ONE
# fundamental piece orbited under the full point group (the same
# snap-onto-symmetry-element-then-weld scheme as the cg_higher_* and
# symmcg_* assemblers above).
#
# CATENOID-ENNEPER, genus g = 2, 3, 4 (one catenoid + one Enneper end):
#   on the hyperelliptic curve y^2 = z (z - v_1) ... (z - v_2g) with
#     phi1 = G dh = rho z^{-1/2} prod (z - v_i)^{+-1/2},
#     phi2 = dh/G = (1/rho) z^{-3/2} prod (z - v_i)^{-+1/2} (z+1)^2,
#     dh   = (1 + z)/z dz
#   (x = Re Int (phi2 - phi1)/2, y = Re Int i(phi2 + phi1)/2,
#   x3 = Re Int dh).  The catenoid end is the branch point z = 0 (dh has
#   residue 2 on the double cover), the Enneper end is z = infinity; dh
#   vanishes at the regular point z = -1 (the neck).  The 2g + 1 real
#   constants (rho, v_2..v_2g; v_1 normalized to 1) solve the 2g-period
#   problem of the notebook: Re Int_{v_s}^{v_s+1} omega = 0 alternating
#   omega_2 / omega_1 across consecutive branch values (the planar
#   symmetry arcs on either side must lie in the SAME mirror plane) plus
#   one condition tying the negative real axis into the x-mirror.  All
#   constants in _CWCE_CE were verified/re-solved here (residuals
#   ~5e-9, quadrature-stable; g4's were solved from the notebook's
#   Newton seeds -- the notebook stores none).  Symmetry: two orthogonal vertical mirror planes intersecting
#   in a vertical axis that threads every branch-point image; group
#   {1, Mx, My, Rz(pi)} of order 4 (Rz(pi) = the hyperelliptic sheet
#   swap).  Fundamental piece: the upper-half-plane annulus
#   rmin <= |z| <= rmax; the positive real segments map to planar
#   geodesics alternating between the two mirror planes, the negative
#   real axis into the x-mirror, and the two rims are the catenoid /
#   Enneper end trims (chi = 2 - 2g - 2 exactly, gated below).
#
# COSTA-WOHLGEMUTH, genus 2(k-1), and WOHLGEMUTH SECOND SURFACE,
# genus 3(k-1)  (4 ends: 2 catenoidal + 2 planar):
#   on the k-fold cyclic cover of the sphere branched over
#   {+-1, +-a, +-b} (resp. {+-1, +-a, +-b, +-c}), with
#     phi1 = (z-1)^{-1+1/k} (z+1)^{1-1/k} (z-a)^{1+1/k} (z+a)^{-1-1/k}
#            (z-b)^{-1-1/k} (z+b)^{-1+1/k}   [second surface: extra
#            (z-c)^{1-1/k} (z+c)^{-1+1/k} and the a/b roles reshuffled
#            as in _cwce_cw_data],
#     phi2 = the exponent-negated mirror partner (phi1 phi2 = dh^2),
#     dh   = dz / ((z-b)(z+b)).
#   Catenoidal ends at z = +-b (dh poles, totally ramified), planar ends
#   at z = +-a, finite branch images (handles) at z = +-1 (and +-c).
#   k = 2 gives Wohlgemuth's genus-2 (Costa-Wohlgemuth) and genus-3
#   surfaces -- the first complete embedded minimal surfaces with four
#   ends.  Period problem: (a, b) [resp. (a, b, c)] solve the conditions
#   that the (a,b) symmetry arc's vertical plane and the (0,a) arc's
#   plane pass through the axis (plus, for the second surface, the z=c
#   branch image landing on the axis); the constants in _CWCE_CW /
#   _CWCE_W2 are the notebook literals re-verified here to ~1e-13 for
#   every k.  Symmetry: prismatic D_kh of order 4k -- k-fold vertical
#   axis through the +-1 (and +-c) branch images, k vertical mirror
#   planes at angles j pi/k, one HORIZONTAL mirror plane at the height
#   h0 of the z = 0 image (the image of the imaginary axis is a planar
#   geodesic in that plane).  Fundamental piece: the conformal strip
#   chart z = sqrt(moeb(e^w)), w in [xa, xb] x [0, pi] with
#   moeb(u) = (rho_m b^2 u + a^2)/(rho_m u + 1), rho_m = a^2/b^2 (the
#   notebook's chart of the quarter plane): the planar end opens at the
#   left strip end, the catenoidal end at the right, and the marked
#   points z = 0, 1, c, infinity sit at exact grid columns on the top
#   edge where the integrand's algebraic singularities get graded
#   quadrature blocks with exact node offsets.  4k copies weld to
#   chi = 2 - 2 genus - 4 exactly (four open end rims), gated below.
#
# The Wohlgemuth second-surface family for k > 2 is NOT shipped: the
# blog notebook's k = 3 / k = 4 constants do not satisfy the closure
# conditions (residuals ~3e-2 under the same quadrature that confirms
# every other member at ~1e-13), and re-solving from them stalls --
# see BACKLOG.md.
#
# References:
#   C. J. Costa, "Example of a complete minimal immersion in R^3 of
#     genus one and three embedded ends", Bol. Soc. Bras. Mat. 15
#     (1984) 47-54 -- the ancestor construction;
#   M. Wohlgemuth, "Higher genus minimal surfaces by growing handles
#     out of a catenoid", Manuscripta Math. 70 (1991) 397-428, Bonn
#     dissertation (1993), and "Minimal surfaces of higher genus with
#     finite total curvature", Arch. Rational Mech. Anal. 137 (1997)
#     1-25 -- the genus-2 and genus-3 four-ended embedded surfaces;
#   C. C. Chen, F. Gackstatter, "Elliptische und hyperelliptische
#     Funktionen und vollstaendige Minimalflaechen vom Enneperschen
#     Typ", Math. Ann. 259 (1982) -- the Enneper-end heritage of the
#     catenoid-Enneper family;
#   H. Karcher, "Construction of minimal surfaces", Univ. of Tokyo
#     lecture notes (1989) -- the symmetry / period method;
#   M. Weber, https://minimalsurfaces.blog/ -- repository pages
#     "Costa-Wohlgemuth surfaces", "Wohlgemuth's second surface"
#     (higher-symmetry notebook by Ramazan Yol) and "Catenoid-Enneper
#     surfaces of genus g", whose Mathematica notebooks supplied the
#     Weierstrass data, period conditions and constants followed here.

# Solved period constants.  CE: genus -> (rho, (v1..v2g)) with v1 = 1;
# g2/g3 re-verified notebook literals, g4 solved here from the
# notebook's Newton seeds (residuals ~1e-13, re-checked by the
# self-test through cwce_ce_residual).
_CWCE_CE = {
    2: (2.441868735003271,
        (1.0, 1.5899997953673903, 4.2856159413955295,
         5.235200094406395)),
    3: (2.709609741854014,
        (1.0, 1.619390029065051, 3.9554260271562294, 4.993548107702124,
         7.829353305141648, 8.864263970017321)),
    4: (2.9455445424953384,
        (1.0, 1.6412114092203107, 3.8014391806809074, 4.89812108352522,
         7.320461951519022, 8.47252262184776, 11.478960661548184,
         12.542173771958389)),
}
# Costa-Wohlgemuth: k -> (a, b), genus 2(k-1) -- notebook literals
# re-verified/refined to ~1e-13 for every k
_CWCE_CW = {
    2: (0.5936584192549862, 0.8240789231207696),
    3: (0.6342850763231503, 0.8412721089595467),
    4: (0.674940760134315, 0.8583521760688522),
    5: (0.7084980441058044, 0.8722674706663541),
    7: (0.7583644125949724, 0.8928687061979),
}
# Wohlgemuth second surface: k -> (a, b, c), genus 3(k-1); only the
# k = 2 (genus 3) period problem closes -- see the header note
_CWCE_W2 = {
    2: (0.6518984510521686, 0.9141432157734375, 3.0832637626461463),
}

_CWCE_GL_CACHE = {}


def _cwce_gl(n):
    if n not in _CWCE_GL_CACHE:
        _CWCE_GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _CWCE_GL_CACHE[n]


def cwce_path_int(fn, waypoints, e_end=(0.0, 0.0), n=400):
    """Gauss-Legendre integral of fn(z) dz along the polyline
    `waypoints`; e_end = (e_first, e_last) are algebraic singularity
    exponents |z - z_end|^-e at the path ends (graded u^p
    substitution there makes the transformed integrand smooth)."""
    x, w = _cwce_gl(n)
    u = 0.5 * (x + 1.0)
    wu = 0.5 * w
    total = 0.0 + 0.0j
    m = len(waypoints) - 1
    for i in range(m):
        A, B = complex(waypoints[i]), complex(waypoints[i + 1])
        eL = e_end[0] if i == 0 else 0.0
        eR = e_end[1] if i == m - 1 else 0.0
        if eL > 0.0 and eR > 0.0:
            Mid = 0.5 * (A + B)
            total += cwce_path_int(fn, [A, Mid], (eL, 0.0), n)
            total += cwce_path_int(fn, [Mid, B], (0.0, eR), n)
            continue
        if eR > 0.0:
            A, B = B, A
            eL, sgn = eR, -1.0
        else:
            sgn = 1.0
        if eL > 0.0:
            p = 2.0 / max(1e-9, 1.0 - min(eL, 0.95)) + 1.0
            t = u ** p
            J = p * u ** (p - 1.0)
        else:
            t, J = u, np.ones_like(u)
        z = A + t * (B - A)
        f = fn(z) * (B - A) * J
        f = np.where(np.isfinite(f), f, 0.0)
        total += sgn * np.sum(f * wu)
    return total


def _cwce_make_phi(factors, pref=1.0):
    """z -> pref * prod (z - v)^p on Im z >= 0 (principal branches --
    continuous there, matching continuation from the upper halfplane)."""
    def phi(z):
        z = np.asarray(z, dtype=complex)
        out = np.full(z.shape, pref, dtype=complex)
        for (v, p) in factors:
            out = out * np.exp(p * np.log(z - v))
        return out
    return phi


def cwce_ce_phis(genus):
    """(fac1, fac2, rho, roots) for Catenoid-Enneper genus g."""
    rho, roots = _CWCE_CE[genus]
    f1 = [(0.0, -0.5)]
    f2 = [(0.0, -1.5)]
    for i, v in enumerate(roots):
        s = 0.5 if i % 2 == 0 else -0.5
        f1.append((v, s))
        f2.append((v, -s))
    return f1, f2, rho, list(roots)


def cwce_ce_residual(genus, n=400):
    """The notebook's 2g period conditions at the stored constants
    (all must vanish -- the self-test's honesty gate on _CWCE_CE)."""
    f1, f2, rho, v = cwce_ce_phis(genus)
    phi1 = _cwce_make_phi(f1, rho)
    phi2r = _cwce_make_phi(f2, 1.0 / rho)

    def om1(z):
        return 0.5 * (phi2r(z) * (1.0 + z) ** 2 - phi1(z))

    def om2(z):
        return 0.5j * (phi2r(z) * (1.0 + z) ** 2 + phi1(z))

    res = []
    for s in range(2 * genus - 1):
        A, B = v[s], v[s + 1]
        om = om2 if s % 2 == 0 else om1
        res.append(np.real(cwce_path_int(om, [A, 1j, B], (0.5, 0.5), n)))
    res.append(np.real(cwce_path_int(
        om1, [-0.5, 1j, 0.5 * (v[0] + v[1])], (0.0, 0.0), n)))
    return np.array(res)


def cwce_cw_residual(fam, k, n=400):
    """Period-condition residuals at the stored (a, b[, c]) constants:
    the (a,b) arc's mirror plane and the (0,a) arc's plane must pass
    through the vertical axis (and z = c must land on it)."""
    a, b, c, exps, marks = _cwce_cw_data(fam, k)
    f1 = [(p, q1) for p, (q1, q2) in exps.items()]
    f2 = [(p, q2) for p, (q1, q2) in exps.items()]
    phi1 = _cwce_make_phi(f1)
    phi2 = _cwce_make_phi(f2)

    def om1(z):
        return 0.5 * (phi2(z) - phi1(z))

    def om2(z):
        return 0.5j * (phi1(z) + phi2(z))

    e1 = 1.0 - 1.0 / k
    up = 0.35j
    m = 0.5 * (a + b)
    if fam == 'cw':
        r1 = np.real(cwce_path_int(om2, [1.0, 1.0 + up, m + up, m],
                                   (e1, 0.0), n))
        I1 = np.real(cwce_path_int(om1, [1.0, 1.0 + up, up, 0.0],
                                   (e1, 0.0), n))
        I2 = np.real(cwce_path_int(om2, [1.0, 1.0 + up, up, 0.0],
                                   (e1, 0.0), n))
        al = math.pi / k
        return np.array([r1, -I1 * math.sin(al) + I2 * math.cos(al)])
    r1 = np.real(cwce_path_int(om2, [0.0, 0.3 + up, 1.0 + up, 1.0],
                               (0.0, e1), n))
    r2 = np.real(cwce_path_int(om2, [0.0, 0.3 + up, c + up, c],
                               (0.0, e1), n))
    I1 = np.real(cwce_path_int(om1, [1.0, 1.0 + up, m + up, m],
                               (e1, 0.0), n))
    I2 = np.real(cwce_path_int(om2, [1.0, 1.0 + up, m + up, m],
                               (e1, 0.0), n))
    bet = 0.5 * math.pi - math.pi / k
    return np.array([r1, r2, I1 * math.cos(bet) + I2 * math.sin(bet)])


def cwce_grids(bounds, sing, nu, msub=6, msing=80, pmesh=1.7):
    """Coarse mesh nodes + fine trapezoid nodes/weights on
    [bounds[0], bounds[-1]]; interior bounds are exact coarse nodes,
    entries of `sing` (node -> exponent e of an |x - node|^-e
    integrable singularity) get graded u^p quadrature blocks with
    EXACT node offsets (sid/soff).  Returns (x_c, x_f, cidx, sid,
    soff, w_lo, w_hi); a cumulative line integral is
    cumsum(w_lo f[:-1] + w_hi f[1:]) sampled at cidx."""
    nseg = len(bounds) - 1
    lens = np.diff(np.array(bounds, dtype=float))
    wseg = np.sqrt(lens)
    counts = np.maximum(6, np.round(nu * wseg / wseg.sum()).astype(int))
    keys = sorted(sing)

    def sing_id(x):
        for i, s in enumerate(keys):
            if abs(x - s) < 1e-12:
                return i
        return -1

    x_c, x_f = [bounds[0]], [bounds[0]]
    sid, soff = [-1], [0.0]
    cidx = [0]
    w_lo, w_hi = [], []
    for s in range(nseg):
        A, B = bounds[s], bounds[s + 1]
        n = int(counts[s])
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        tt = np.where(t < 0.5, 0.5 * (2 * t) ** pmesh,
                      1.0 - 0.5 * (2 * (1 - t)) ** pmesh)
        nodes = A + (B - A) * tt
        nodes[-1] = B
        prev = A
        for ci in range(n):
            hi = nodes[ci]
            singL = ci == 0 and A in sing
            singR = ci == n - 1 and B in sing
            if singL:
                p = 2.0 / max(1e-9, 1.0 - min(sing[A], 0.95)) + 1.0
                D = hi - A
                u = np.linspace(0.0, 1.0, msing + 1)
                off = D * u ** p
                J = D * p * u ** (p - 1.0)
                du = 1.0 / msing
                xf = (A + off)[1:]
                xf[-1] = hi
                sid_i = np.full(len(xf), sing_id(A))
                soff_i = off[1:]
                w_lo.extend((0.5 * du * J[:-1]).tolist())
                w_hi.extend((0.5 * du * J[1:]).tolist())
            elif singR:
                p = 2.0 / max(1e-9, 1.0 - min(sing[B], 0.95)) + 1.0
                D = B - prev
                v = np.linspace(0.0, 1.0, msing + 1)
                offB = D * (1.0 - v) ** p
                J = D * p * (1.0 - v) ** (p - 1.0)
                dv = 1.0 / msing
                xf = (B - offB)[1:]
                xf[-1] = B
                sid_i = np.full(len(xf), sing_id(B))
                soff_i = -offB[1:]
                w_lo.extend((0.5 * dv * J[:-1]).tolist())
                w_hi.extend((0.5 * dv * J[1:]).tolist())
            else:
                tloc = np.linspace(0.0, 1.0, msub + 1)
                left_half = (prev - A) <= (B - hi)
                if left_half and A in sing and prev > A:
                    dlo, dhi = prev - A, hi - A
                    d = dlo * (dhi / dlo) ** tloc
                    xf_full = A + d
                    J = d * math.log(dhi / dlo)
                elif not left_half and B in sing and hi < B:
                    dtop, dbot = B - prev, B - hi
                    d = dtop * (dbot / dtop) ** tloc
                    xf_full = B - d
                    J = d * abs(math.log(dbot / dtop))
                else:
                    xf_full = prev + (hi - prev) * tloc
                    J = None
                xf = xf_full[1:].copy()
                xf[-1] = hi
                sid_i = np.full(len(xf), -1)
                soff_i = np.zeros(len(xf))
                if J is None:
                    h = 0.5 * (hi - prev) / msub
                    w_lo.extend([h] * msub)
                    w_hi.extend([h] * msub)
                else:
                    dt = 1.0 / msub
                    w_lo.extend((0.5 * dt * J[:-1]).tolist())
                    w_hi.extend((0.5 * dt * J[1:]).tolist())
            x_f.extend(xf.tolist())
            sid.extend(sid_i.tolist())
            soff.extend(soff_i.tolist())
            cidx.append(len(x_f) - 1)
            x_c.append(hi)
            prev = hi
    return (np.array(x_c), np.array(x_f), np.array(cidx),
            np.array(sid, dtype=int), np.array(soff),
            np.array(w_lo), np.array(w_hi))


def cwce_weld_frames(V0, faces0, UV0, bmask, mats, revs, tol=1e-6):
    """Orbit the snapped piece under the affine frames [(M, t)], weld
    boundary vertices by coincidence (frames with revs[i] True reverse
    the face winding), drop degenerate faces, and keep the largest
    face-connected component.  Returns (V, faces, uv)."""
    nV = len(V0)
    Vp, Fp = [], []
    for fr, (M, t) in enumerate(mats):
        Vp.append(V0 @ np.asarray(M).T + t)
        for f in faces0:
            ff = tuple(int(x) + fr * nV for x in f)
            Fp.append(ff[::-1] if revs[fr] else ff)
    V = np.concatenate(Vp)
    UVall = np.tile(UV0, (len(mats), 1))
    N = len(V)
    parent = np.arange(N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bidx = np.nonzero(np.tile(bmask, len(mats)))[0]
    key = np.floor(V[bidx] / tol + 0.5).astype(np.int64)
    H = {}
    for t2, i in enumerate(bidx):
        H.setdefault((int(key[t2, 0]), int(key[t2, 1]),
                      int(key[t2, 2])), []).append(int(i))
    for t2, i in enumerate(bidx):
        k0 = key[t2]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in H.get((int(k0[0]) + dx, int(k0[1]) + dy,
                                    int(k0[2]) + dz), ()):
                        if j > i and np.linalg.norm(V[j] - V[i]) < tol:
                            ra, rb = find(int(i)), find(j)
                            if ra != rb:
                                parent[ra] = rb
    rts = np.array([find(a) for a in range(N)])
    uniq, inv = np.unique(rts, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    UVw = np.zeros((len(uniq), 2))
    cnt = np.zeros(len(uniq))
    np.add.at(Vw, inv, V)
    np.add.at(UVw, inv, UVall)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    UVw /= cnt[:, None]
    F = []
    for f in Fp:
        gg = [int(inv[i]) for i in f]
        h = [gg[0]]
        for t2 in range(1, len(gg)):
            if gg[t2] != h[-1]:
                h.append(gg[t2])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    parent2 = np.arange(len(Vw))

    def find2(a):
        while parent2[a] != a:
            parent2[a] = parent2[parent2[a]]
            a = parent2[a]
        return a

    for f in F:
        for i in range(1, len(f)):
            ra, rb = find2(f[0]), find2(f[i])
            if ra != rb:
                parent2[ra] = rb
    from collections import Counter
    sizes = Counter(find2(f[0]) for f in F)
    root = sizes.most_common(1)[0][0]
    F = [f for f in F if find2(f[0]) == root]
    used = sorted(set(i for f in F for i in f))
    rmv = {v: i for i, v in enumerate(used)}
    return Vw[used], [tuple(rmv[i] for i in f) for f in F], UVw[used]


# ---- Catenoid-Enneper piece / assembly -----------------------------------

def cwce_ce_omega(genus, r, theta, sid, soff, roots):
    """(n, 3) complex (om1, om2, om3) at z = r e^{i theta} on one
    radial ray; sid/soff give exact offsets to the branch values on
    the theta = 0 ray (its algebraic singularities are integrated in
    graded variables where the integrand is smooth)."""
    f1, f2, rho, _ = cwce_ce_phis(genus)
    n = len(r)
    z = r * np.exp(1j * theta)
    out = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for fac, pref in ((f1, rho), (f2, 1.0 / rho)):
            lmag = np.full(n, math.log(pref))
            ang = np.zeros(n)
            for (v, p) in fac:
                if theta == 0.0:
                    t = np.abs(r - v)
                    if v in roots:
                        m = sid == roots.index(v)
                        t[m] = np.abs(soff[m])
                        left = np.array(r < v)
                        left[m] = soff[m] < 0
                    else:
                        left = r < v
                    lmag = lmag + p * np.log(t)
                    ang = ang + p * np.where(left, math.pi, 0.0)
                else:
                    d = z - v
                    lmag = lmag + p * np.log(np.abs(d))
                    ang = ang + p * np.angle(d)
            out.append(np.exp(lmag + 1j * ang))
        phi1, phi2 = out
        phi2 = phi2 * (1.0 + z) ** 2
        phi3 = (1.0 + z) / z
        om = np.stack([0.5 * (phi2 - phi1), 0.5j * (phi2 + phi1),
                       phi3], axis=-1)
    return np.where(np.isfinite(om), om, 0.0)


def cwce_ce_piece(genus, nu, nv, rmin, rmax):
    """Mesh the upper-half-plane annulus fundamental piece: stitch arc
    at a mid radius plus radial-ray integration in both directions.
    Returns (V, faces, boundary classification, per-vertex uv)."""
    f1, f2, rho, roots = cwce_ce_phis(genus)
    bounds = [rmin] + roots + [rmax]
    sing = {v: 0.5 for v in roots}
    r_c, r_f, cidx, sid, soff, w_lo, w_hi = cwce_grids(bounds, sing, nu)
    th = np.linspace(0.0, math.pi, nv)
    rB = math.sqrt(roots[0] * roots[-1])
    ib = int(np.argmin(np.abs(r_c - rB)))
    marc = 12
    thf = np.linspace(0.0, math.pi, (nv - 1) * marc + 1)
    zf = float(r_c[ib]) * np.exp(1j * thf)
    lm1 = np.full(len(thf), math.log(rho))
    an1 = np.zeros(len(thf))
    lm2 = np.full(len(thf), -math.log(rho))
    an2 = np.zeros(len(thf))
    for (v, p) in f1:
        d = zf - v
        lm1 += p * np.log(np.abs(d))
        an1 += p * np.angle(d)
    for (v, p) in f2:
        d = zf - v
        lm2 += p * np.log(np.abs(d))
        an2 += p * np.angle(d)
    phi1 = np.exp(lm1 + 1j * an1)
    phi2 = np.exp(lm2 + 1j * an2) * (1.0 + zf) ** 2
    phi3 = (1.0 + zf) / zf
    oma = np.stack([0.5 * (phi2 - phi1), 0.5j * (phi2 + phi1), phi3],
                   axis=-1)
    dzf = np.diff(zf)[:, None]
    Aarc = np.concatenate(
        [np.zeros((1, 3), complex),
         np.cumsum(0.5 * (oma[1:] + oma[:-1]) * dzf, axis=0)],
        axis=0)[::marc]
    ncr = len(r_c)
    X = np.zeros((ncr, nv, 3))
    for j, t in enumerate(th):
        om = cwce_ce_omega(genus, r_f, float(t), sid, soff, roots)
        contrib = (w_lo[:, None] * om[:-1] + w_hi[:, None] * om[1:]) \
            * np.exp(1j * t)
        F = np.concatenate([np.zeros((1, 3), complex),
                            np.cumsum(contrib, axis=0)], axis=0)
        F = F - F[cidx[ib]]
        X[:, j, :] = np.real(Aarc[j][None, :] + F[cidx])
    V = X.reshape(-1, 3)

    def vid(i, j):
        return i * nv + j

    UV = np.zeros((len(V), 2))
    UV[:, 0] = np.tile(th / math.pi, ncr)
    UV[:, 1] = np.repeat(np.linspace(0.0, 1.0, ncr), nv)
    faces = []
    for i in range(ncr - 1):
        for j in range(nv - 1):
            faces.append((vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)))
    root_ci = [int(np.argmin(np.abs(r_c - v))) for v in roots]
    b = {'seg': {}, 'branch': {}}
    lastci = 0
    for s in range(len(roots) + 1):
        hi_ci = root_ci[s] if s < len(roots) else ncr - 1
        i0 = lastci if s == 0 else lastci + 1
        stop = hi_ci + (1 if s == len(roots) else 0)
        b['seg'][s] = np.array([vid(i, 0) for i in range(i0, stop)],
                               dtype=np.int64)
        if s < len(roots):
            b['branch'][s] = vid(hi_ci, 0)
        lastci = hi_ci
    b['neg'] = np.array([vid(i, nv - 1) for i in range(ncr)],
                        dtype=np.int64)
    return V, faces, b, UV


def cwce_ce_snap(V, b):
    """Snap the boundary onto the exact symmetry elements: even real
    segments into the y = cy mirror, odd segments and the negative
    axis into x = cx, branch images onto the axis (cx, cy).  Returns
    (V, (cx, cy), pre-snap residuals -- the numeric period closure)."""
    ys, xs = [], []
    for s, idx in b['seg'].items():
        if len(idx):
            (ys if s % 2 == 0 else xs).append(
                V[idx, 1] if s % 2 == 0 else V[idx, 0])
    xs.append(V[b['neg'], 0])
    cy = float(np.median(np.concatenate(ys)))
    cx = float(np.median(np.concatenate(xs)))
    res = {}
    for s, idx in b['seg'].items():
        if len(idx) == 0:
            continue
        ax = 1 if s % 2 == 0 else 0
        cv = cy if s % 2 == 0 else cx
        res[f'seg{s}'] = float(np.max(np.abs(V[idx, ax] - cv)))
        V[idx, ax] = cv
    res['neg'] = float(np.max(np.abs(V[b['neg'], 0] - cx)))
    V[b['neg'], 0] = cx
    for s, vi in b['branch'].items():
        res[f'branch{s}'] = float(np.hypot(V[vi, 0] - cx,
                                           V[vi, 1] - cy))
        V[vi, 0], V[vi, 1] = cx, cy
    return V, (cx, cy), res


def cwce_ce_assemble(genus, nu, nv, rmin=None, rmax=None):
    """Watertight order-4 assembly of the Catenoid-Enneper surface:
    4 snapped copies of the half-plane piece welded along the mirror
    curves; the two rims stay open (catenoid / Enneper end trims).
    Returns (V, faces, uv, diag)."""
    rho, roots = _CWCE_CE[genus]
    if rmin is None:
        rmin = roots[0] / {2: 10.0, 3: 40.0, 4: 60.0}[genus]
    if rmax is None:
        rmax = {2: 1.5, 3: 1.12, 4: 1.08}[genus] * roots[-1]
    V0, faces0, b, UV0 = cwce_ce_piece(genus, nu, nv, rmin, rmax)
    V0, (cx, cy), res = cwce_ce_snap(V0.copy(), b)
    bmask = np.zeros(len(V0), dtype=bool)
    for idx in b['seg'].values():
        bmask[idx] = True
    bmask[list(b['branch'].values())] = True
    bmask[b['neg']] = True
    I3 = np.eye(3)
    My = np.diag([1.0, -1.0, 1.0])
    Mx = np.diag([-1.0, 1.0, 1.0])
    R2 = np.diag([-1.0, -1.0, 1.0])
    mats = [(I3, np.zeros(3)),
            (R2, np.array([2 * cx, 2 * cy, 0.0])),
            (My, np.array([0.0, 2 * cy, 0.0])),
            (Mx, np.array([2 * cx, 0.0, 0.0]))]
    revs = [False, False, True, True]
    V, F, UV = cwce_weld_frames(V0, faces0, UV0, bmask, mats, revs)
    return V, F, UV, {'res': res}


def cwce_ce_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """MESH_PARAM builder: finished (V, quads, uv) fit to the 2 m cube.
    The radius slider scales both end trims (how far the catenoid
    funnel and the Enneper flare reach)."""
    p = spec['p_from'](order, radius)
    genus = p['genus']
    rho, roots = _CWCE_CE[genus]
    t = float(np.clip(radius / 1.2, 0.55, 1.8))
    rmin = roots[0] / ({2: 10.0, 3: 40.0, 4: 60.0}[genus] ** t)
    rmax = roots[-1] * (1.0 + ({2: 0.5, 3: 0.12, 4: 0.08}[genus]) * t)
    pnu = int(np.clip(nu * 1.6, 90, 260))
    pnv = int(np.clip(nv * 0.55, 21, 49))
    V, quads, uv, _diag = cwce_ce_assemble(genus, pnu, pnv, rmin, rmax)
    tk = _toolkit()
    V = tk._smooth_boundary(V, quads, iters=6)
    V = tk._center_fit(V, scale, V)
    return V, quads, uv


# ---- Costa-Wohlgemuth / Wohlgemuth piece / assembly ----------------------

def _cwce_expm1c(w):
    """exp(w) - 1 for complex w, accurate near w = 0."""
    w = np.asarray(w, dtype=complex)
    small = np.abs(w) < 1e-5
    out = np.exp(w) - 1.0
    ws = w[small]
    out[small] = ws * (1.0 + ws * (0.5 + ws / 6.0))
    return out


def _cwce_cw_data(fam, k):
    """(a, b, c, exps, marks): exps maps branch value p -> exponents
    (in phi1, in phi2) of the factor (z - p); marks are the strip
    x-positions of the top-edge marked points z = 0, infinity, 1, c."""
    if fam == 'cw':
        a, b = _CWCE_CW[k]
        c = None
    else:
        a, b, c = _CWCE_W2[k]
    e1 = 1.0 - 1.0 / k
    ea = 1.0 + 1.0 / k
    exps = {1.0: (-e1, e1), -1.0: (e1, -e1),
            a: (ea, -ea), -a: (-ea, ea),
            b: (-1.0 - 1.0 / k, -1.0 + 1.0 / k),
            -b: (-1.0 + 1.0 / k, -1.0 - 1.0 / k)}
    if fam == 'w2':
        exps[c] = (e1, -e1)
        exps[-c] = (-e1, e1)
    marks = {'x0': 0.0,
             'xinf': math.log(b * b / (a * a)),
             'x1': math.log(b * b * (1 - a * a)
                            / (a * a * (1 - b * b)))}
    if fam == 'w2':
        rho = a * a / (b * b)
        marks['xc'] = math.log((c * c - a * a)
                               / (rho * (c * c - b * b)))
    return a, b, c, exps, marks


def cwce_cw_omega_row(fam, k, dat, x_f, y, sid, soff, skeys):
    """(nf, 3) complex (om1, om2, om3) * dz/dw along the strip row
    w = x + iy under the chart z = sqrt(moeb(e^w)).  Every pair
    difference z^2 - p^2 is evaluated through a cancellation-free
    Moebius identity (exact expm1 forms), so the rows next to the
    marked points z = 0, 1, c, infinity stay at full precision;
    sid/soff carry exact offsets to the singular marks."""
    a, b, c, exps, marks = dat
    rho = a * a / (b * b)
    w = x_f + 1j * y
    with np.errstate(divide='ignore', invalid='ignore',
                     over='ignore'):
        D = -_cwce_expm1c(w + math.log(rho) - 1j * math.pi)
        Nz = -rho * b * b * _cwce_expm1c(w - 1j * math.pi)
        z = np.sqrt(Nz / D)
        flip = (z.imag < 0) | ((z.imag == 0) & (z.real < 0))
        z = np.where(flip, -z, z)
        if y in (0.0, math.pi):
            z = np.where(np.abs(z.imag) < 1e-30 * np.abs(z.real),
                         z.real + 0j, z)
        u = np.exp(w)
        x1 = marks['x1']
        d1 = x_f - x1
        if 'x1' in skeys:
            m = sid == skeys.index('x1')
            d1 = d1.copy()
            d1[m] = soff[m]
        pairs = {1.0: rho * (b * b - 1.0) * (-math.exp(x1))
                 * _cwce_expm1c(d1 + 1j * (y - math.pi)) / D,
                 a: rho * u * (b * b - a * a) / D,
                 b: (a * a - b * b) / D}
        if fam == 'w2':
            xc = marks['xc']
            dc = x_f - xc
            if 'xc' in skeys:
                m = sid == skeys.index('xc')
                dc = dc.copy()
                dc[m] = soff[m]
            pairs[c] = rho * (b * b - c * c) * (-math.exp(xc)) \
                * _cwce_expm1c(dc + 1j * (y - math.pi)) / D
        lm1 = np.zeros(len(x_f))
        an1 = np.zeros(len(x_f))
        lm2 = np.zeros(len(x_f))
        an2 = np.zeros(len(x_f))
        for p, (q1, q2) in exps.items():
            fac = pairs[p] / (z + p) if p > 0 else z - p
            if y in (0.0, math.pi):
                fac = np.where(np.abs(fac.imag)
                               < 1e-25 * np.abs(fac.real),
                               fac.real + 0j, fac)
            L = np.log(np.abs(fac))
            A = np.angle(fac)
            lm1 = lm1 + q1 * L
            an1 = an1 + q1 * A
            lm2 = lm2 + q2 * L
            an2 = an2 + q2 * A
        phi1 = np.exp(lm1 + 1j * an1)
        phi2 = np.exp(lm2 + 1j * an2)
        dh = 1.0 / pairs[b]
        dzdw = rho * (b * b - a * a) / (D * D) * u / (2.0 * z)
        om = np.stack([0.5 * (phi2 - phi1), 0.5j * (phi1 + phi2), dh],
                      axis=-1) * dzdw[:, None]
    return np.where(np.isfinite(om), om, 0.0)


def cwce_cw_piece(fam, k, nu, nv, xa, xb):
    """Mesh the strip fundamental piece [xa, xb] x [0, pi]: row-wise
    integration rebased at a generic column plus one column stitch.
    Returns (V, faces, boundary classification, per-vertex uv)."""
    dat = _cwce_cw_data(fam, k)
    marks = dat[4]
    e1 = 1.0 - 1.0 / k
    mk = sorted(marks.items(), key=lambda t: t[1])
    bounds = [xa] + [v for _, v in mk] + [xb]
    sing = {v: (e1 if name in ('x1', 'xc') else 0.5)
            for name, v in mk}
    x_c, x_f, cidx, sid_x, soff, w_lo, w_hi = cwce_grids(
        bounds, sing, nu, msub=6, msing=80)
    skeys = [name for name, v in mk]
    ys = np.linspace(0.0, math.pi, nv)
    ncx = len(x_c)
    icB = int(np.argmin(np.abs(x_c - 0.5 * (marks['x1'] + xb))))
    marc = 10
    yf = np.linspace(0.0, math.pi, (nv - 1) * marc + 1)
    omc = np.stack([cwce_cw_omega_row(
        fam, k, dat, np.array([x_c[icB]]), float(yf[t]),
        np.array([-1]), np.array([0.0]), skeys)[0]
        for t in range(len(yf))])
    dyf = np.diff(yf)[:, None]
    Acol = np.concatenate(
        [np.zeros((1, 3), complex),
         np.cumsum(0.5 * (omc[1:] + omc[:-1]) * (1j * dyf), axis=0)],
        axis=0)[::marc]
    X = np.zeros((ncx, nv, 3))
    for j, y in enumerate(ys):
        om = cwce_cw_omega_row(fam, k, dat, x_f, float(y), sid_x,
                               soff, skeys)
        contrib = w_lo[:, None] * om[:-1] + w_hi[:, None] * om[1:]
        F = np.concatenate([np.zeros((1, 3), complex),
                            np.cumsum(contrib, axis=0)], axis=0)
        F = F - F[cidx[icB]]
        X[:, j, :] = np.real(Acol[j][None, :] + F[cidx])
    V = X.reshape(-1, 3)

    def vid(i, j):
        return i * nv + j

    UV = np.zeros((len(V), 2))
    UV[:, 0] = np.tile(ys / math.pi, ncx)
    UV[:, 1] = np.repeat(np.linspace(0.0, 1.0, ncx), nv)
    faces = []
    for i in range(ncx - 1):
        for j in range(nv - 1):
            faces.append((vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)))
    mark_ci = {name: int(np.argmin(np.abs(x_c - v)))
               for name, v in marks.items()}
    b_cls = {'bottom': np.array([vid(i, 0) for i in range(ncx)],
                                dtype=np.int64)}
    edges = [0] + [mark_ci[name] for name in skeys] + [ncx - 1]
    segs = []
    for s in range(len(edges) - 1):
        i0, i1 = edges[s], edges[s + 1]
        first = i0 if s == 0 else i0 + 1
        last = i1 + (1 if s == len(edges) - 2 else 0)
        segs.append(np.array([vid(i, nv - 1)
                              for i in range(first, last)],
                             dtype=np.int64))
    b_cls['top_segs'] = segs
    b_cls['top_names'] = skeys
    b_cls['mark_vid'] = {name: vid(ci, nv - 1)
                         for name, ci in mark_ci.items()}
    return V, faces, b_cls, UV


def cwce_cw_snap(fam, k, V, b_cls):
    """Center on the z = 1 branch image (the origin of the D_kh
    frame), then snap every boundary curve onto its exact symmetry
    element: vertical mirror planes through the axis at the family's
    angles (0 / pi/k, measured off the Weierstrass phases), the
    horizontal mirror plane at the z = 0 image height h0, and the
    branch images onto the axis.  Returns (V, h0, residuals)."""
    res = {}
    V = V - V[b_cls['mark_vid']['x1']]
    if fam == 'cw':
        seg_angle = [math.pi / k, None, 0.0, math.pi / k]
        bottom_angle = 0.0
        ang0 = math.pi / k
    else:
        seg_angle = [0.0, None, 0.0, math.pi / k, 0.0]
        bottom_angle = math.pi / k
        ang0 = 0.0
    h0 = float(V[b_cls['mark_vid']['x0'], 2])
    for s, idx in enumerate(b_cls['top_segs']):
        if len(idx) == 0:
            continue
        ang = seg_angle[s]
        if ang is None:                    # horizontal-mirror curve
            res[f'top{s}'] = float(np.max(np.abs(V[idx, 2] - h0)))
            V[idx, 2] = h0
        else:
            uvec = np.array([math.cos(ang), math.sin(ang)])
            xy = V[idx][:, :2]
            t = xy @ uvec
            perp = xy - t[:, None] * uvec[None, :]
            res[f'top{s}'] = float(np.max(np.linalg.norm(perp,
                                                        axis=1)))
            V[idx, 0] = t * uvec[0]
            V[idx, 1] = t * uvec[1]
    idx = b_cls['bottom']
    uvec = np.array([math.cos(bottom_angle), math.sin(bottom_angle)])
    xy = V[idx][:, :2]
    t = xy @ uvec
    perp = xy - t[:, None] * uvec[None, :]
    res['bottom'] = float(np.max(np.linalg.norm(perp, axis=1)))
    V[idx, 0] = t * uvec[0]
    V[idx, 1] = t * uvec[1]
    mv = b_cls['mark_vid']
    res['b_x1'] = float(np.hypot(V[mv['x1'], 0], V[mv['x1'], 1]))
    V[mv['x1'], 0] = V[mv['x1'], 1] = 0.0
    if 'xc' in mv:
        res['b_xc'] = float(np.hypot(V[mv['xc'], 0], V[mv['xc'], 1]))
        V[mv['xc'], 0] = V[mv['xc'], 1] = 0.0
    for name, ang in (('x0', ang0), ('xinf', 0.0)):
        vi = mv[name]
        uvec = np.array([math.cos(ang), math.sin(ang)])
        t = float(V[vi, :2] @ uvec)
        res[f'b_{name}'] = max(
            float(np.linalg.norm(V[vi, :2] - t * uvec)),
            abs(float(V[vi, 2] - h0)))
        V[vi, 0], V[vi, 1] = t * uvec[0], t * uvec[1]
        V[vi, 2] = h0
    return V, h0, res


def cwce_cw_assemble(fam, k, nu, nv, xa=None, xb=None):
    """Watertight D_kh assembly (4k frames: k rotations x vertical
    mirror x horizontal mirror at h0); the four end rims stay open.
    Returns (V, faces, uv, diag)."""
    dat = _cwce_cw_data(fam, k)
    marks = dat[4]
    if xa is None:
        xa = -1.4
    if xb is None:
        xb = marks['x1'] + 2.6
    V0, faces0, b_cls, UV0 = cwce_cw_piece(fam, k, nu, nv, xa, xb)
    V0, h0, res = cwce_cw_snap(fam, k, V0.copy(), b_cls)
    bmask = np.zeros(len(V0), dtype=bool)
    bmask[b_cls['bottom']] = True
    for idx in b_cls['top_segs']:
        bmask[idx] = True
    bmask[list(b_cls['mark_vid'].values())] = True
    mats, revs = [], []
    Mzh = np.diag([1.0, 1.0, -1.0])
    My = np.diag([1.0, -1.0, 1.0])
    for j in range(k):
        cj, sj = math.cos(TAU * j / k), math.sin(TAU * j / k)
        R = np.array([[cj, -sj, 0.0], [sj, cj, 0.0],
                      [0.0, 0.0, 1.0]])
        for m in (0, 1):
            for h in (0, 1):
                M = R @ (My if m else np.eye(3)) \
                    @ (Mzh if h else np.eye(3))
                mats.append((M, np.array([0.0, 0.0, 2.0 * h0 * h])))
                revs.append((m + h) % 2 == 1)
    V, F, UV = cwce_weld_frames(V0, faces0, UV0, bmask, mats, revs)
    return V, F, UV, {'res': res, 'h0': h0}


def cwce_cw_mesh(spec, nu, nv, order, radius, scale, theta=0.0):
    """MESH_PARAM builder: finished (V, quads, uv) fit to the 2 m
    cube.  The radius slider slides the strip trims (planar-end reach
    on the left, catenoid depth on the right)."""
    p = spec['p_from'](order, radius)
    fam, k = p['fam'], p['k']
    marks = _cwce_cw_data(fam, k)[4]
    t = float(np.clip(radius / 1.2, 0.7, 1.5))
    xa = -1.4 * t
    xb = marks['x1'] + 2.6 * t
    pnu = int(np.clip(nu * 1.8, 110, 300))
    pnv = int(np.clip(nv * 0.42, 17, 37))
    V, quads, uv, _diag = cwce_cw_assemble(fam, k, pnu, pnv, xa, xb)
    tk = _toolkit()
    V = tk._smooth_boundary(V, quads, iters=6)
    V = tk._center_fit(V, scale, V)
    return V, quads, uv


# ==========================================================================
# Doubly periodic hyperelliptic tiler -- Karcher-Meeks-Rosenberg + Wei
# ==========================================================================
# A reusable engine for the classical four-ended doubly periodic minimal
# surfaces whose Weierstrass data lives on a hyperelliptic curve with
# reciprocal-paired REAL branch points and dh = dz/z (the "KMR form"):
#
#     g(z) = prod_i (z - p_i)^(e_i/2),   e_i = +-1,  p_i real,
#     dh   = dz/z,
#
# plus the KMR-3 member, whose Gauss map is the Mobius map
# (z + eps)/(z - eps) with eps = e^{i Phi} and whose dh is the
# holomorphic differential dz/w of the elliptic curve
# w^2 = z^4 - 2 cos(2 Alpha) z^2 + 1 (branch points on the unit circle).
#
# Structure exploited (and verified by the self-tests):
#   * In exponential coordinates z = e^w the quotient surface is an
#     annulus; the engine integrates ONE conformal patch over the strip
#     u in [log rmin, 0], v in [0, pi] (the upper-half-plane half of the
#     annulus) with a graded, branch-aligned grid and a refined
#     trapezoid rule anchored on the smooth v = pi/2 midline.
#   * The patch boundary v = 0 / v = pi (the real z axis) decomposes at
#     the branch points into planar symmetry arcs lying in VERTICAL
#     MIRROR PLANES x1 = const / x2 = const -- alternating with the
#     parity of the branch points passed -- or (KMR-3) into straight
#     lines on the surface.  Each arc is snapped exactly onto its plane
#     or line; the mismatch of wall constants that must coincide IS the
#     period problem, measured and gated as a residual (for Wei's
#     surface it reproduces the notebook's FindRoot condition
#     Im int_a^b (G + 1/G) dz/z = 0).
#   * The whole doubly periodic surface is the orbit of that one patch
#     under the group generated by the boundary isometries -- diagonal
#     sign maps (reflections in the coordinate mirror planes, 180-degree
#     rotations about straight lines in the surface) -- together with
#     the two lattice translations they compose to:
#         T1 = (2 dx, 0, 0),  T2 = (0, 2 pi, 0)      (KMR-1/2, Wei), or
#         T2 = (0, 2 c2, 2 c3)                       (KMR-3, tilted).
#     Copies weld seam-exactly because shared arcs are snapped onto the
#     fixed sets of the gluing isometries (no loose float matching).
#
# Every gluing map is a Schwarz reflection (z -> zbar on the parameter
# domain, anti-conformal), so every generator -- including the proper
# 180-degree rotations -- reverses the surface orientation; face
# windings flip with the generator parity, keeping the welded mesh
# consistently oriented.
#
# References:
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988) 83-114 (the doubly
#     periodic Scherk deformations; toroidal saddle towers);
#   W. H. Meeks III, H. Rosenberg, "The global theory of doubly
#     periodic minimal surfaces", Invent. Math. 97 (1989) 351-379;
#   J. Perez, M. M. Rodriguez, M. Traizet, "The classification of
#     doubly periodic minimal tori with parallel ends", J. Differential
#     Geom. 69 (2005) 523-577 (the standard examples are named "KMR
#     surfaces" here);
#   F. Wei, "Some existence and uniqueness theorems for doubly periodic
#     minimal surfaces", Invent. Math. 109 (1992) 113-136 (the genus-2
#     surface: a KMR/Scherk surface with an added handle);
#   M. Weber, https://minimalsurfaces.blog/ repository, doubly periodic
#     section (KMR-2, KMR-3 and "Doubly Wei (g=2)" notebooks -- the
#     Weierstrass data, solved period constants and reflection
#     assemblies this engine follows).

# Wei genus-2 family: (b, a) pairs with a solved from b by the period
# condition Im int_a^b (G + 1/G) dz/z = 0 (the notebook's FindRoot
# values, verified against the same integral by the self-test below).
DPERIODIC_WEI_SAMPLES = (
    (0.30, 0.23640853975826828),
    (0.35, 0.18932453473239355),
    (0.40, 0.14531351679837468),
    (0.50, 0.07082564803712697),
    (0.60, 0.022673092012506006),
    (0.65, 0.009769033395931668),
    (0.70, 0.0030808434547059727),
)


def _dperiodic_graded(a, b, n):
    """n+1 cosine-graded nodes on [a, b], clustered at both endpoints
    (the branch points sit at interval ends, where the integrand has
    its inverse-square-root singularities)."""
    t = np.linspace(0.0, 1.0, max(2, int(n)) + 1)
    return a + (b - a) * (0.5 - 0.5 * np.cos(math.pi * t))


def _dperiodic_refine(x, m):
    """Insert m-1 intermediate nodes per interval (keeps the original
    nodes bit-exact at indices ::m, so a fine integration grid can be
    subsampled back onto the mesh grid)."""
    x = np.asarray(x, dtype=float)
    if m <= 1:
        return x
    seg = [x[:-1] + (x[1:] - x[:-1]) * (k / m) for k in range(m)]
    return np.append(np.stack(seg, axis=1).reshape(-1), x[-1])


def _dperiodic_ugrid(splits, total):
    """Concatenated per-interval graded u grid through the exact split
    values (branch-point logs).  Returns (u, split_indices)."""
    splits = list(splits)
    L = np.diff(np.asarray(splits))
    w = np.maximum(L, 1e-9) ** 0.6
    counts = [max(8, int(round(total * wi / w.sum()))) for wi in w]
    u = [np.array([splits[0]])]
    idx = [0]
    for k, n in enumerate(counts):
        seg = _dperiodic_graded(splits[k], splits[k + 1], n)
        u.append(seg[1:])
        idx.append(idx[-1] + n)
    return np.concatenate(u), idx


def _dperiodic_vgrid(nv):
    """Cosine-graded v grid on [0, pi] (dense at both boundary lines),
    even interval count so v = pi/2 is an exact node."""
    n = max(16, int(nv))
    n += n % 2
    t = np.linspace(0.0, 1.0, n + 1)
    return math.pi * (0.5 - 0.5 * np.cos(math.pi * t))


def _dperiodic_integrate(F, ug, vg):
    """Cumulative-trapezoid antiderivative of the (nu, nv, 3) integrand
    F (in w = u + iv coordinates, dz/dw folded in) on the nonuniform
    grid: base row along the smooth v = pi/2 midline, columns up and
    down from it.  Returns Re of the antiderivative (nu, nv, 3)."""
    F = np.where(np.isfinite(F), F, 0.0)
    jm = int(np.argmin(np.abs(vg - 0.5 * math.pi)))
    du = np.diff(ug)[:, None]
    dv = np.diff(vg)
    base = np.zeros((len(ug), 3), dtype=complex)
    row = F[:, jm, :]
    base[1:] = np.cumsum(0.5 * (row[1:] + row[:-1]) * du, axis=0)
    col = np.zeros(F.shape, dtype=complex)
    up = np.cumsum(0.5 * (F[:, jm:-1, :] + F[:, jm + 1:, :])
                   * (1j * dv[jm:])[None, :, None], axis=1)
    col[:, jm + 1:, :] = up
    Frev = F[:, jm::-1, :]
    dvrev = dv[:jm][::-1]
    dn = np.cumsum(0.5 * (Frev[:, :-1, :] + Frev[:, 1:, :])
                   * (-1j * dvrev)[None, :, None], axis=1)
    col[:, jm - 1::-1, :] = dn
    return np.real(base[:, None, :] + col)


def _dperiodic_cont_sqrt(Q, jm):
    """Continuous branch of sqrt(Q) on a grid over a simply connected
    domain with no interior zeros: principal sqrt with the sign
    propagated by continuity from the (smooth) row jm outward."""
    s = np.sqrt(Q)
    t = s.copy()
    r = s[:, jm]
    step = np.where(np.abs(r[1:] - r[:-1]) > np.abs(r[1:] + r[:-1]),
                    -1.0, 1.0)
    sgn = np.concatenate([[1.0], np.cumprod(step)])
    t[:, jm] = r * sgn
    for rng in (range(jm + 1, Q.shape[1]), range(jm - 1, -1, -1)):
        prev = jm
        for j in rng:
            sj = s[:, j]
            tp = t[:, prev]
            fj = np.where(np.abs(sj - tp) > np.abs(sj + tp), -1.0, 1.0)
            t[:, j] = sj * fj
            prev = j
    return t


def _dperiodic_sqrtprod(z, factors):
    """g(z) = prod (z - p)^(e/2), principal branches per factor (single
    valued and holomorphic on the upper half plane for real p)."""
    g = np.ones_like(z)
    for p, e in factors:
        s = np.sqrt(z - p)
        g = g * s if e > 0 else g / s
    return g


def _dperiodic_F_exp(factors):
    """Integrand closure for the exp-coordinate KMR form: z = e^w,
    dh = dz/z; returns (omega1, omega2, omega3) * dz/dw on the w grid.
    With dh = dz/z the third component is exactly 1, so x3 = Re w = u
    exactly -- the horizontal-mirror edge u = 0 needs no snapping."""
    def F(W, jm):
        z = np.exp(W)
        with np.errstate(divide='ignore', invalid='ignore'):
            g = _dperiodic_sqrtprod(z, factors)
            f1 = 0.5 * (1.0 / g - g)
            f2 = 0.5j * (1.0 / g + g)
        one = np.ones_like(f1)
        return np.stack(np.broadcast_arrays(f1, f2, one), axis=-1)
    return F


def _dperiodic_F_kmr3(alpha, phi):
    """Integrand closure for KMR-3: parameter x = e^w on the half
    annulus, z = e^{i phi} (i - x)/(x + i) (which maps the real x axis
    onto the unit circle where dh's four branch points +-e^{+-i alpha}
    live), G = (z + eps)/(z - eps) with eps = e^{i phi}, and
    dh = i dz / sqrt(z^4 - 2 cos(2 alpha) z^2 + 1) taken with a
    continuity-tracked branch."""
    eps = np.exp(1j * phi)
    c2a = math.cos(2.0 * alpha)

    def F(W, jm):
        x = np.exp(W)
        z = eps * (1j - x) / (x + 1j)
        dzdw = eps * (-2j) / (x + 1j) ** 2 * x       # dz/dx * dx/dw
        Q = z ** 4 - 2.0 * c2a * z ** 2 + 1.0
        w = _dperiodic_cont_sqrt(Q, jm)
        with np.errstate(divide='ignore', invalid='ignore'):
            h = 1j / w * dzdw                        # dh * dz/dw ... = h dw
            g = (z + eps) / (z - eps)
            f1 = 0.5 * (1.0 / g - g) * h
            f2 = 0.5j * (1.0 / g + g) * h
        return np.stack(np.broadcast_arrays(f1, f2, h), axis=-1)
    return F


def _dperiodic_arcs_exp(factors, rmin):
    """Boundary-arc tables for the exp-coordinate KMR form.  The v = 0
    edge is the positive real axis, the v = pi edge the negative one;
    crossing the real axis at x flips the sign of every factor with
    p > x, so an arc is a planar symmetry curve in a vertical plane
    x2 = const when that count is EVEN (g real there) and x1 = const
    when it is ODD (g imaginary).  Returns (usplits, arcs) with arcs =
    [(edge, k0, k1, axis, cluster)], edge 0/1 = v = 0/pi, k0..k1 the
    split-interval index range, axis the constant coordinate, cluster
    '0' (wall through the origin) or 'far' (the wall whose doubled
    offset is the lattice vector)."""
    pos = sorted(p for p, e in factors if 0 < p < 1 and p > rmin)
    neg = sorted(-p for p, e in factors if -1 < p < 0 and -p > rmin)
    usplits = sorted({math.log(rmin), 0.0}
                     | {math.log(p) for p in pos}
                     | {math.log(t) for t in neg})
    kof = {round(s, 12): k for k, s in enumerate(usplits)}
    arcs = []
    stops_pos = [rmin] + pos + [1.0]
    for lo, hi in zip(stops_pos[:-1], stops_pos[1:]):
        x = math.sqrt(lo * hi)
        cnt = sum(1 for p, e in factors if p > x)
        axis = 1 if cnt % 2 == 0 else 0
        arcs.append((0, kof[round(math.log(lo), 12)],
                     kof[round(math.log(hi), 12)], axis, '0'))
    stops_neg = [rmin] + neg + [1.0]
    for lo, hi in zip(stops_neg[:-1], stops_neg[1:]):
        t = math.sqrt(lo * hi)
        cnt = sum(1 for p, e in factors if p > -t)
        axis = 1 if cnt % 2 == 0 else 0
        arcs.append((1, kof[round(math.log(lo), 12)],
                     kof[round(math.log(hi), 12)], axis,
                     '0' if axis == 1 else 'far'))
    return usplits, arcs


def _dperiodic_arc_slice(X, edge, i0, i1):
    """View of the arc's vertex rows: edge 0 -> v = 0 row, 1 -> v = pi."""
    j = 0 if edge == 0 else X.shape[1] - 1
    return X[i0:i1 + 1, j, :]


def _dperiodic_arc_med(X, edge, i0, i1, axis):
    """Median of one coordinate along an arc, endpoints excluded (the
    corner samples sit next to the branch singularities)."""
    seg = _dperiodic_arc_slice(X, edge, i0, i1)
    core = seg[1:-1] if len(seg) > 4 else seg
    return float(np.median(core[:, axis]))


def dperiodic_patch(key, p, nu, nv, refine=3):
    """Integrate, wall-align and snap ONE conformal patch of the doubly
    periodic surface `key` ('kmr2' | 'wei' | 'kmr3').  Returns a dict:
    X (nu, nv, 3) vertex grid, ug/vg parameter grids, ops (list of
    (sign3, parity) diagonal isometries whose orbit of the patch is one
    translational fundamental cell), T1/T2 lattice vectors, res
    (residual diagnostics -- the period problem's closure errors), and
    arc bookkeeping."""
    if key == 'wei':
        b, a = p['b'], p['a']
        r = a / b
        rmin = min(p.get('rmin', 0.05), 0.45 * min(a, r))
        factors = ((a, -1), (b, 1), (1.0 / b, -1), (1.0 / a, 1),
                   (-r, 1), (-1.0 / r, -1))
        usplits, arcs = _dperiodic_arcs_exp(factors, rmin)
        Ffn = _dperiodic_F_exp(factors)
    elif key == 'kmr2':
        a = p['a']
        rmin = min(p.get('rmin', 0.05), 0.45 * a)
        factors = ((a, -1), (1.0 / a, 1), (-a, 1), (-1.0 / a, -1))
        usplits, arcs = _dperiodic_arcs_exp(factors, rmin)
        Ffn = _dperiodic_F_exp(factors)
    elif key == 'kmr3':
        alpha = p.get('alpha', 0.5 * math.pi - 0.125 * math.pi)
        phi = p.get('phi', 0.25 * math.pi)
        xmin = p.get('xmin', 0.05)
        t1 = math.tan(0.5 * (alpha - phi))          # v=0 junctions
        t2 = math.tan(0.5 * (alpha + phi))          # v=pi junctions: t2, 1/t1
        xmin = min(xmin, 0.45 * t1)
        usplits = sorted({math.log(xmin), math.log(t1), -math.log(t2),
                          math.log(t2), -math.log(t1), -math.log(xmin)})
        kof = {round(s, 12): k for k, s in enumerate(usplits)}
        # (edge, k0, k1, kind, cluster): planes are x1 = const walls,
        # 'line' arcs are straight lines parallel to x1 (x2, x3 const)
        arcs = [
            (0, kof[round(math.log(xmin), 12)],
             kof[round(math.log(t1), 12)], 0, 'far'),
            (0, kof[round(math.log(t1), 12)],
             kof[round(-math.log(t2), 12)], 'line', '0'),
            (0, kof[round(-math.log(t2), 12)],
             kof[round(-math.log(xmin), 12)], 0, '0'),
            (1, kof[round(math.log(xmin), 12)],
             kof[round(math.log(t2), 12)], 0, '0'),
            (1, kof[round(math.log(t2), 12)],
             kof[round(-math.log(t1), 12)], 'line', 'far'),
            (1, kof[round(-math.log(t1), 12)],
             kof[round(-math.log(xmin), 12)], 0, 'far'),
        ]
        Ffn = _dperiodic_F_kmr3(alpha, phi)
    else:
        raise ValueError(f"dperiodic_patch: unknown key {key!r}")

    ug, uidx = _dperiodic_ugrid(usplits, max(24, int(nu)))
    vg = _dperiodic_vgrid(max(24, int(nv)))
    m = max(1, int(refine))
    uf = _dperiodic_refine(ug, m)
    vf = _dperiodic_refine(vg, m)
    jmf = int(np.argmin(np.abs(vf - 0.5 * math.pi)))
    W = uf[:, None] + 1j * vf[None, :]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        F = Ffn(W, jmf)
    Xf = _dperiodic_integrate(F, uf, vf)
    X = Xf[::m, ::m, :].copy()
    kidx = {k: uidx[k] for k in range(len(usplits))}
    arcsg = [(e, kidx[k0], kidx[k1], ax, cl) for (e, k0, k1, ax, cl)
             in arcs]

    res = {}
    if key in ('kmr2', 'wei'):
        # exact height: with dh = dz/z, x3 = u identically (measured
        # residual is pure quadrature error -- gate it, then use u)
        res['x3=u'] = float(np.max(np.abs(
            X[..., 2] - (X[0, 0, 2] - ug[0] + ug[:, None]))))
        X[..., 2] = ug[:, None]
        # align: v=0 x1 wall and (first) v=0 x2 wall through the origin
        s1 = [_dperiodic_arc_med(X, e, i0, i1, 0)
              for (e, i0, i1, ax, cl) in arcsg if ax == 0 and cl == '0']
        s2 = [_dperiodic_arc_med(X, e, i0, i1, 1)
              for (e, i0, i1, ax, cl) in arcsg if ax == 1 and e == 0]
        X[..., 0] -= s1[0]
        X[..., 1] -= s2[0]
        # wall constants + period residuals: every x2 wall must sit at
        # a multiple of pi (0 on v=0 -- Wei's FindRoot condition -- and
        # +-pi on v=pi); the 'far' x1 walls must agree on one offset dx
        far = []
        for (e, i0, i1, ax, cl) in arcsg:
            if ax == 'line':
                continue
            med = _dperiodic_arc_med(X, e, i0, i1, ax)
            if ax == 1:
                tgt = math.pi * round(med / math.pi)
                res[f'x2wall@{e}:{i0}'] = abs(med - tgt)
            elif cl == 'far':
                far.append(med)
        dx = float(np.mean(far))
        if len(far) > 1:
            res['x1walls'] = float(np.ptp(far))
        # snap every arc exactly onto its wall plane
        vpi_sign = 0.0
        for (e, i0, i1, ax, cl) in arcsg:
            seg = _dperiodic_arc_slice(X, e, i0, i1)
            if ax == 1:
                med = float(np.median(seg[1:-1, 1] if len(seg) > 4
                                      else seg[:, 1]))
                tgt = math.pi * round(med / math.pi)
                seg[:, 1] = tgt
                if e == 1:
                    vpi_sign = math.copysign(1.0, tgt if tgt else 1.0)
            else:
                seg[:, 0] = 0.0 if cl == '0' else dx
        ops = [((sx, sy, sz), sx * sy * sz)
               for sx in (1.0, -1.0) for sy in (1.0, -1.0)
               for sz in (1.0, -1.0)]
        T1 = np.array([2.0 * dx, 0.0, 0.0])
        T2 = np.array([0.0, 2.0 * math.pi * (vpi_sign or 1.0), 0.0])
    else:                                            # kmr3
        # classify residuals BEFORE alignment: constancy of the declared
        # coordinates along each arc (plane arcs: x1; line arcs: x2, x3)
        for (e, i0, i1, ax, cl) in arcsg:
            seg = _dperiodic_arc_slice(X, e, i0, i1)
            core = seg[1:-1] if len(seg) > 4 else seg
            if ax == 'line':
                res[f'line@{e}:{i0}'] = float(
                    max(np.ptp(core[:, 1]), np.ptp(core[:, 2])))
            else:
                res[f'plane@{e}:{i0}'] = float(np.ptp(core[:, 0]))
        # align: v=0 '0' plane wall -> x1 = 0; v=0 line -> x2 = x3 = 0
        for (e, i0, i1, ax, cl) in arcsg:
            if e == 0 and ax == 0 and cl == '0':
                X[..., 0] -= _dperiodic_arc_med(X, e, i0, i1, 0)
        for (e, i0, i1, ax, cl) in arcsg:
            if e == 0 and ax == 'line':
                X[..., 1] -= _dperiodic_arc_med(X, e, i0, i1, 1)
                X[..., 2] -= _dperiodic_arc_med(X, e, i0, i1, 2)
        # walls: dx from the v=0 'far' plane; the v=pi planes must land
        # on the SAME two walls (their offsets are the period residuals)
        dx = [_dperiodic_arc_med(X, e, i0, i1, 0)
              for (e, i0, i1, ax, cl) in arcsg
              if e == 0 and ax == 0 and cl == 'far'][0]
        c23 = [( _dperiodic_arc_med(X, e, i0, i1, 1),
                 _dperiodic_arc_med(X, e, i0, i1, 2))
               for (e, i0, i1, ax, cl) in arcsg
               if e == 1 and ax == 'line'][0]
        for (e, i0, i1, ax, cl) in arcsg:
            if e == 1 and ax == 0:
                med = _dperiodic_arc_med(X, e, i0, i1, 0)
                tgt = 0.0 if cl == '0' else dx
                res[f'x1wall@{e}:{i0}'] = abs(med - tgt)
        # snap
        for (e, i0, i1, ax, cl) in arcsg:
            seg = _dperiodic_arc_slice(X, e, i0, i1)
            if ax == 'line':
                seg[:, 1], seg[:, 2] = ((0.0, 0.0) if cl == '0'
                                        else (c23[0], c23[1]))
            else:
                seg[:, 0] = 0.0 if cl == '0' else dx
        # orbit: mirror x1 = 0 and the 180-degree rotation about the
        # x1 axis (both Schwarz continuations -> parity -1 each)
        ops = [((1.0, 1.0, 1.0), 1.0), ((-1.0, 1.0, 1.0), -1.0),
               ((1.0, -1.0, -1.0), -1.0), ((-1.0, -1.0, -1.0), 1.0)]
        T1 = np.array([2.0 * dx, 0.0, 0.0])
        T2 = np.array([0.0, 2.0 * c23[0], 2.0 * c23[1]])
    return {'X': X, 'ug': ug, 'vg': vg, 'arcs': arcsg, 'ops': ops,
            'T1': T1, 'T2': T2, 'res': res, 'key': key,
            'ends_edge': 'both' if key == 'kmr3' else 'low'}


def _dperiodic_weld(V, quads, UV, decimals=6):
    """Weld coincident vertices by exact rounded-coordinate keys (the
    assembly snaps every shared arc onto the fixed set of its gluing
    isometry, so seam partners agree to machine precision -- this is a
    hash join on those exact positions, not a loose tolerance match)."""
    key = np.round(V * 10.0 ** decimals).astype(np.int64)
    view = np.ascontiguousarray(key).view(
        np.dtype((np.void, key.dtype.itemsize * key.shape[1])))
    _, first, inv = np.unique(view.ravel(), return_index=True,
                              return_inverse=True)
    inv = inv.ravel()
    Vw = V[first]
    UVw = UV[first] if UV is not None else None
    out = []
    for f in inv[np.asarray(quads, dtype=np.int64)]:
        h = [int(f[0])]
        for s in f[1:]:
            if int(s) != h[-1]:
                h.append(int(s))
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            out.append(tuple(h))
    return Vw, out, UVw


def dperiodic_assemble(P, cells=(1, 1)):
    """Tile the patch orbit over cells[0] x cells[1] lattice cells at
    the TRUE period vectors and weld all seams.  Returns (V, quads,
    per-vertex UV)."""
    X = P['X']
    nu, nv = X.shape[:2]
    Vp = X.reshape(-1, 3)
    gu = np.arange(nu) / max(nu - 1, 1)
    gv = np.arange(nv) / max(nv - 1, 1)
    UVp = np.stack(np.meshgrid(gu, gv, indexing='ij'),
                   axis=-1).reshape(-1, 2)
    ii, jj = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1),
                         indexing='ij')
    b = (ii * nv + jj).ravel()
    q0 = np.stack([b, b + nv, b + nv + 1, b + 1], axis=1)
    cu, cv = int(max(1, cells[0])), int(max(1, cells[1]))
    Vs, Qs, UVs = [], [], []
    off0 = 0
    for icell in range(cu):
        for jcell in range(cv):
            t = ((icell - 0.5 * (cu - 1)) * P['T1']
                 + (jcell - 0.5 * (cv - 1)) * P['T2'])
            for sgn, parity in P['ops']:
                Vs.append(Vp * np.asarray(sgn)[None, :] + t[None, :])
                UVs.append(UVp)
                Qs.append((q0 if parity > 0 else q0[:, ::-1]) + off0)
                off0 += len(Vp)
    V = np.concatenate(Vs, axis=0)
    UV = np.concatenate(UVs, axis=0)
    Q = np.concatenate(Qs, axis=0)
    return _dperiodic_weld(V, Q, UV)


def dperiodic_quotient(P):
    """Topology of the translational quotient: assemble ONE cell, then
    identify its opposite lattice walls (vertices matched under +-T1,
    +-T2 by the same exact-coordinate hash).  Returns (chi, boundary
    loop count, nonmanifold edge count, oriented, components) --
    ends stay as boundary rims, so chi must equal 2 - 2 genus - ends."""
    V, quads, _ = dperiodic_assemble(P, (1, 1))
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    key = np.round(V * 1e6).astype(np.int64)
    hkey = {tuple(k): i for i, k in enumerate(key)}
    for T in (P['T1'], P['T2'], -P['T1'], -P['T2']):
        kt = np.round((V + T[None, :]) * 1e6).astype(np.int64)
        for i, k in enumerate(kt):
            j = hkey.get(tuple(k))
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    root = np.array([find(i) for i in range(len(V))])
    ec, dc = {}, {}
    for f in quads:
        mlen = len(f)
        for k in range(mlen):
            a2, b2 = int(root[f[k]]), int(root[f[(k + 1) % mlen]])
            e = (a2, b2) if a2 < b2 else (b2, a2)
            ec[e] = ec.get(e, 0) + 1
            dc[(a2, b2)] = dc.get((a2, b2), 0) + 1
    nvq = len({int(r) for f in quads for r in root[list(f)]})
    chi = nvq - len(ec) + len(quads)
    nonman = sum(1 for c in ec.values() if c > 2)
    orient = all(c == 1 for c in dc.values())
    bed = [e for e, c in ec.items() if c == 1]
    par = {}

    def bfind(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a2, b2 in bed:
        ra, rb = bfind(a2), bfind(b2)
        if ra != rb:
            par[ra] = rb
    loops = len({bfind(a2) for a2, b2 in bed})
    parc = {}

    def cfind(x):
        parc.setdefault(x, x)
        while parc[x] != x:
            parc[x] = parc[parc[x]]
            x = parc[x]
        return x

    for f in quads:
        for i in range(1, len(f)):
            ra, rb = cfind(int(root[f[0]])), cfind(int(root[f[i]]))
            if ra != rb:
                parc[ra] = rb
    ncomp = len({cfind(int(root[i2])) for f in quads for i2 in f})
    return chi, loops, nonman, orient, ncomp


_DPERIODIC_CACHE = {}


def dperiodic_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                   cells=(1, 1)):
    """Finished-mesh builder for the doubly periodic rows (toolkit
    MESH_PARAM / cells2d contract): patch -> orbit -> cells[0] x
    cells[1] lattice tiling -> largest component -> 2 m fit.  theta is
    unused (no associate family on the tiled surfaces)."""
    tk = _toolkit()
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    key = spec['dp_key']
    if isinstance(cells, (int, float)):
        cells = (int(cells), 1)
    cu = int(np.clip(cells[0], 1, 8))
    cv = int(np.clip(cells[1] if len(cells) > 1 else 1, 1, 8))
    ck = (key, tuple(sorted(p.items())), int(nu), int(nv))
    P = _DPERIODIC_CACHE.get(ck)
    if P is None:
        P = dperiodic_patch(key, p, int(np.clip(nu, 24, 220)),
                            int(np.clip(nv, 24, 220)))
        if len(_DPERIODIC_CACHE) > 12:
            _DPERIODIC_CACHE.clear()
        _DPERIODIC_CACHE[ck] = P
    V, quads, UV = dperiodic_assemble(P, (cu, cv))
    Vu, quads = tk._largest_component(np.hstack([V, UV]), quads)
    V, UV = Vu[:, :3], Vu[:, 3:]
    V = tk._center_fit(V, scale, V)
    return V, quads, UV


def dperiodic_wei_residual(b, a, n=320001):
    """Wei's period condition, exactly as the notebook's FindRoot test:
    Im int_a^b (G + 1/G) dz/z along the real segment [a, b] (the cut),
    with the same principal-branch factors as the surface.  The cosine
    substitution regularizes the endpoint inverse-square-root
    singularities; the remaining O(h^2) trapezoid error is Richardson-
    extrapolated away (verified: the harvested constants then close to
    ~1e-6, and the h^2 tail alone was the earlier 1e-3 'residual')."""
    r = a / b
    factors = ((a, -1), (b, 1), (1.0 / b, -1), (1.0 / a, 1),
               (-r, 1), (-1.0 / r, -1))
    sa, sb = math.log(a), math.log(b)

    def quad(m):
        # integrate in s = log x (dz/z = ds exactly, so the 1/z pole
        # near the small branch point a never amplifies the error),
        # cosine-graded against the endpoint branch singularities
        t = np.linspace(0.0, 1.0, m)
        s = sa + (sb - sa) * (0.5 - 0.5 * np.cos(math.pi * t))
        x = np.exp(s) + 0j
        with np.errstate(divide='ignore', invalid='ignore'):
            g = _dperiodic_sqrtprod(x, factors)
            f = g + 1.0 / g
        f = np.where(np.isfinite(f), f, 0.0)
        ds = np.diff(s)
        return float(np.imag(np.sum(0.5 * (f[1:] + f[:-1]) * ds)))

    r2 = quad(n)
    r1 = quad(n // 2 + 1)
    return (4.0 * r2 - r1) / 3.0


# ==========================================================================
# Doubly periodic long tail -- Karcher-Scherk, RTW, Wei tower, Connor
# ==========================================================================
# The remaining classical/experimental four-or-more-ended doubly periodic
# minimal surfaces whose Weierstrass data is hyperelliptic with REAL
# branch points and dh = dz/z, generalizing the KMR/Wei tiler above to
# the three assembly styles the source notebooks use:
#
#   * 'half8' -- reciprocal-paired branch points with opposite square-
#     root exponents, so |g| = 1 on the unit circle: the u = 0 edge of
#     the exp-strip is a horizontal planar symmetry curve and the patch
#     orbit is the 8 diagonal sign maps (exactly the engine above).
#     Members: F. Wei's higher-genus (1,3)/(1,4)/(2,3)/(1,6) towers and
#     the Rossman-Thayer-Wohlgemuth M1+ / M1+- surfaces.
#   * 'ks'    -- the Karcher-Scherk surfaces with handles: branch
#     points at +-1 put arg g = const on the unit circle, which is a
#     horizontal STRAIGHT LINE (direction (1,1,0)) in the surface (the
#     classical doubly periodic Scherk contains these diagonal lines at
#     mid-level).  Orbit = closure of {180-degree rotation about that
#     line, mirrors x1 = 0 and x2 = 0} (8 ops); lattice fixed at
#     (2 pi, 0, 0) x (0, 2 pi, 0) by the dh = dz/z normalization, so
#     BOTH x1 and x2 walls must land on multiples of pi -- that is this
#     family's period problem, solved in the notebooks by FindRoot on
#     segment integrals of omega1/omega2 between branch points.
#   * 'full4' -- no circle symmetry at all (Connor's experimental
#     surfaces, branch sets not reciprocal-invariant in pairs with
#     mixed exponents): integrate the WHOLE annulus rlo < |z| < rhi
#     (upper half), orbit = {id, mirror x1, mirror x2, both} only,
#     ends truncated at both edges.
#
# Every member ships with the literal solved period constants recovered
# from the repository notebooks (FindRoot outputs / family tables); the
# self-tests re-measure the wall residuals, the 2-D lattice, and the
# quotient topology chi = 2 - 2 genus - ends per fundamental domain.
#
# References:
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988) 83-114 (Scherk surfaces
#     with handles; the genus 2..4 doubly periodic tower);
#   F. Wei, "Some existence and uniqueness theorems for doubly periodic
#     minimal surfaces", Invent. Math. 109 (1992) 113-136, and the
#     higher-genus (m,n) Wei family after M. Weber's repository;
#   W. Rossman, E. C. Thayer, M. Wohlgemuth, "Embedded, doubly periodic
#     minimal surfaces", Experiment. Math. 9 (2000) 197-219 (the M1+
#     and M1+- families);
#   P. Connor, "A note on special polynomials and minimal surfaces"
#     (experimental doubly periodic surfaces of genus 3, 2018-2019;
#     numerically established -- see also P. Connor, M. Weber, "The
#     construction of doubly periodic minimal surfaces via balance
#     equations", Amer. J. Math. 134 (2012) 1275-1301);
#   M. Weber, https://minimalsurfaces.blog/ repository, doubly periodic
#     section (Karcher-Scherk g=2/3, exotic g=3, RTW, higher-genus Wei
#     and Connor pages 78-85 notebooks: Weierstrass data, solved
#     constants and reflection assemblies followed here).

def _dptail_recip(vals):
    """[(p, e)] -> reciprocal-completed factor list [(p, e), (1/p, -e)]."""
    out = []
    for p, e in vals:
        out.append((p, e))
        out.append((1.0 / p, -e))
    return tuple(out)


def _dptail_members():
    """The harvested member catalog: key -> dict(factors, style, genus,
    label).  Constants are the notebooks' solved period parameters."""
    M = {}

    # --- Karcher-Scherk with handles (style 'ks') ----------------------
    a, b = 0.1253991455944982, 0.25068716043696654
    M['ksg2'] = {
        'style': 'ks', 'genus': 2,
        'label': "Karcher-Scherk genus 2",
        'factors': ((-1.0, 1), (-1.0 / a, 1), (-a, 1),
                    (1.0, -1), (-1.0 / b, -1), (-b, -1))}
    a, b, c = (0.06861386286581515, 0.11033686517910986,
               0.4744615458765442)
    M['ksg3'] = {
        'style': 'ks', 'genus': 3,
        'label': "Karcher-Scherk genus 3",
        'factors': ((-1.0 / a, 1), (-a, 1), (-1.0 / c, 1), (-c, 1),
                    (1.0, -1), (-1.0, -1), (-1.0 / b, -1), (-b, -1))}
    a, b, c = (0.0029790137955450864, 0.21361044126340875,
               0.0038598089630300695)
    M['ksg3x'] = {
        'style': 'ks', 'genus': 3,
        'label': "Karcher-Scherk genus 3 (exotic)",
        'factors': ((1.0, 1), (-1.0, 1), (-1.0 / a, 1), (-a, 1),
                    (c, -1), (1.0 / c, -1), (-1.0 / b, -1), (-b, -1))}

    # --- Rossman-Thayer-Wohlgemuth (style 'half8') ---------------------
    a = 0.26                       # M1+: b solved from a (self-test)
    b = DPTAIL_RTWMP_B
    M['rtwmp'] = {
        'style': 'half8', 'genus': 2,
        'label': "RTW M1+ (genus 2)",
        'factors': ((1.0 / a, 1), (1.0 / b, 1), (-a * b, 1),
                    (a, -1), (b, -1), (-1.0 / (a * b), -1))}
    for i, (b, c, d) in enumerate((
            (0.7348870529374415, -0.6951258227510726, -0.03),
            (0.6855592258248246, -0.5572870898881896, -0.1))):
        M[f'rtwm1pm_{i}'] = {
            'style': 'half8', 'genus': 3,
            'label': f"RTW M1+- (genus 3, d={d})",
            'factors': _dptail_recip(((b, 1), (c, 1), (d, 1),
                                      (1.0 / (b * c * d), 1)))}

    # --- Wei higher-genus towers (style 'half8') -----------------------
    # (1,3): (c; a, b) rows, r = a c / b
    for i, (c, a, b) in enumerate((
            (0.1, 0.008055067127584473, 0.028060852552439276),
            (0.2, 0.016531255568679768, 0.054948614240354035),
            (0.5, 0.04830868160434303, 0.11731752208030022),
            (0.8, 0.09242500104957037, 0.13493687204523117))):
        r = a * c / b
        M[f'wei13_{i}'] = {
            'style': 'half8', 'genus': 3,
            'label': f"Wei (1,3) genus 3, c={c}",
            'factors': _dptail_recip(((a, -1), (b, 1), (c, -1),
                                      (-r, 1)))}
    # (1,4): (d; a, b, c) rows, r = a c / (b d)
    for i, (d, a, b, c) in enumerate((
            (0.5, 0.06504976295629919, 0.07123688388229928,
             0.451792255249194),
            (0.7, 0.008088938096668071, 0.01971002640051247,
             0.07514495599787499))):
        r = a * c / (b * d)
        M[f'wei14_{i}'] = {
            'style': 'half8', 'genus': 4,
            'label': f"Wei (1,4) genus 4, d={d}",
            'factors': _dptail_recip(((a, -1), (b, 1), (c, -1),
                                      (d, 1), (-r, 1)))}
    # (2,3): a fixed 1e-4; (b, c, d) solved, r = a c / (b d)
    a, b, c, d = (1e-4, 0.04459799657015391, 0.5189369915935217,
                  0.00011126886913811259)
    r = a * c / (b * d)
    M['wei23'] = {
        'style': 'half8', 'genus': 4,
        'label': "Wei (2,3) genus 4",
        'factors': _dptail_recip(((1.0 / a, 1), (b, 1), (1.0 / c, 1),
                                  (-d, 1), (-r, 1)))}
    # (1,6): d fixed 0.02; (a, b, c, e, h) solved, r = a c e / (b d h)
    a, b, c = (0.0004555021778972459, 0.0008261557506571832,
               0.0033952038946453533)
    d, e, h = 0.009809755189394816, 0.02, 0.8260536205512837
    r = a * c * e / (b * d * h)
    M['wei16'] = {
        'style': 'half8', 'genus': 6,
        'label': "Wei (1,6) genus 6",
        'factors': _dptail_recip(((1.0 / a, 1), (b, 1), (1.0 / c, 1),
                                  (d, 1), (1.0 / e, 1), (h, 1),
                                  (-r, 1)))}

    # --- Connor experimental surfaces (style 'full4') ------------------
    # (a, c, d) Newton-refined on the notebook's three conditions at
    # fixed b, e (the published continuation tuple mixes solve stages
    # and misses closure by ~1e-2; refined residual < 1e-12)
    a, b, c = 0.014359861472204319, 0.8490182629508395, 2.39270674308568
    d, e = 31.950515854168334, -0.01
    f = a * b * d / (c * e)
    M['conn_asym'] = {
        'style': 'full4', 'genus': 2,
        'label': "Connor asymmetric (genus 2)",
        'factors': ((a, 1), (b, 1), (d, 1),
                    (c, -1), (e, -1), (f, -1))}
    a, b, c, d = (0.015, 8.45438228656637, 30.30978961482249,
                  81.61817271788935)
    e, h, i2 = (-0.026253034486005255, -0.49160372140098835,
                -1.2857374677056)
    j = b * d * e * h / (a * c * i2)
    M['conn78'] = {
        'style': 'full4', 'genus': 3,
        'label': "Connor page 78 (genus 3)",
        'factors': ((b, 1), (d, 1), (e, 1), (h, 1),
                    (a, -1), (c, -1), (i2, -1), (j, -1))}
    a, b, c, d = (0.00572883889017319, 0.05805739403536119,
                  0.221960415838744, 306.0)
    M['conn80'] = {
        'style': 'full4', 'genus': 3,
        'label': "Connor page 80 (genus 3)",
        'factors': ((-1.0 / a, 1), (a, 1), (-1.0 / b, 1), (b, 1),
                    (-1.0 / c, -1), (c, -1), (-1.0 / d, -1), (d, -1))}
    a, b, c, d = (0.01, 0.0624509711969649, 0.28200180374974865,
                  -0.005219595739646119)
    M['conn82'] = {
        'style': 'full4', 'genus': 3,
        'label': "Connor page 82 (genus 3)",
        'factors': ((a, 1), (b, 1), (1.0 / a, 1), (1.0 / b, 1),
                    (c, -1), (d, -1), (1.0 / c, -1), (1.0 / d, -1))}
    a, b, c, d = (0.0026378211134974183, 0.490458496891309, 1.0,
                  5478.680983554296)
    e, h, i2 = (16486.601223173104, 162703.99452674662,
                -0.0018594407009749243)
    j = c * d * h * i2 / (a * b * e)
    M['conn84'] = {
        'style': 'full4', 'genus': 3,
        'label': "Connor page 84 (genus 3)",
        'factors': ((c, 1), (d, 1), (h, 1), (i2, 1),
                    (a, -1), (b, -1), (e, -1), (j, -1))}
    a, b, c, d = (0.00006188567963069017, 0.0055,
                  0.012221191957186576, 10628.77254193301)
    M['conn85'] = {
        'style': 'full4', 'genus': 3,
        'label': "Connor page 85 (genus 3)",
        'factors': ((a, -1), (b, 1), (c, -1), (d, 1),
                    (-1.0 / a, -1), (-1.0 / b, 1), (-1.0 / c, -1),
                    (-1.0 / d, 1))}
    return M


def dptail_rtwmp_residual(b, a=0.26, n=420):
    """RTW M1+ period condition, as its notebook's SolvePeriod: the
    real part of int i (G + 1/G) dz/z from -a b through i to b must
    vanish; G has reciprocal-paired branch points driven by (a, b)."""
    factors = ((1.0 / a, 1), (1.0 / b, 1), (-a * b, 1),
               (a, -1), (b, -1), (-1.0 / (a * b), -1))

    def fn(z):
        g = _dperiodic_sqrtprod(np.asarray(z, dtype=complex), factors)
        return 1j * (g + 1.0 / g) / z
    return float(np.real(cwce_path_int(fn, [-a * b, 1j, b],
                                       e_end=(0.5, 0.5), n=n)))


# b for RTW M1+ at a = 0.26, solved from dptail_rtwmp_residual by
# bisection at import-verification time (value checked in the
# self-test against the same residual; the notebook only published the
# solve call, not the solved value).
DPTAIL_RTWMP_B = 0.2759530541264884

DPTAIL_MEMBERS = None            # built lazily (needs DPTAIL_RTWMP_B)


def dptail_member(key):
    global DPTAIL_MEMBERS
    if DPTAIL_MEMBERS is None:
        DPTAIL_MEMBERS = _dptail_members()
    return DPTAIL_MEMBERS[key]


def _dptail_arcs(factors, ulo, uhi):
    """Boundary-arc tables for an exp-strip u in [ulo, uhi] whose real
    z-axis edges decompose at the branch-point logs.  Same parity rule
    as _dperiodic_arcs_exp, but over an arbitrary radial window (the
    full annulus of the 'full4' style, or the [rmin, 1] half of the
    'half8'/'ks' styles).  Returns (usplits, arcs) with arcs entries
    (edge, k0, k1, axis) -- axis 0 = x1 wall, 1 = x2 wall."""
    rlo, rhi = math.exp(ulo), math.exp(uhi)
    pos = sorted(p for p, e in factors if rlo < p < rhi)
    neg = sorted(-p for p, e in factors if rlo < -p < rhi)
    usplits = sorted({ulo, uhi}
                     | {math.log(p) for p in pos}
                     | {math.log(t) for t in neg})
    kof = {round(s, 12): k for k, s in enumerate(usplits)}
    arcs = []
    for lo, hi in zip([rlo] + pos, pos + [rhi]):
        x = math.sqrt(lo * hi)
        cnt = sum(1 for p, e in factors if p > x)
        arcs.append((0, kof[round(math.log(lo), 12)],
                     kof[round(math.log(hi), 12)],
                     1 if cnt % 2 == 0 else 0))
    for lo, hi in zip([rlo] + neg, neg + [rhi]):
        t = math.sqrt(lo * hi)
        cnt = sum(1 for p, e in factors if p > -t)
        arcs.append((1, kof[round(math.log(lo), 12)],
                     kof[round(math.log(hi), 12)],
                     1 if cnt % 2 == 0 else 0))
    return usplits, arcs


def dptail_patch(key, p, nu, nv, refine=3):
    """Integrate, wall-align and snap ONE conformal patch of the
    doubly periodic tail member `key`.  Returns the same dict contract
    as dperiodic_patch, with 'ops' generalized to (M 3x3, t 3, parity)
    isometries consumed by dptail_assemble/dptail_quotient."""
    mem = dptail_member(key)
    factors, style = mem['factors'], mem['style']
    mags = sorted(abs(q) for q, e in factors)
    if style == 'full4':
        f = float(np.clip(p.get('trim', 8.0), 1.5, 200.0))
        ulo = math.log(mags[0] / f)
        uhi = math.log(mags[-1] * f)
    else:
        rmin = min(p.get('rmin', 0.05), 0.45 * mags[0])
        ulo, uhi = math.log(rmin), 0.0
    usplits, arcs = _dptail_arcs(factors, ulo, uhi)
    Ffn = _dperiodic_F_exp(factors)

    ug, uidx = _dperiodic_ugrid(usplits, max(24, int(nu)))
    vg = _dperiodic_vgrid(max(24, int(nv)))
    m = max(1, int(refine))
    uf = _dperiodic_refine(ug, m)
    vf = _dperiodic_refine(vg, m)
    jmf = int(np.argmin(np.abs(vf - 0.5 * math.pi)))
    W = uf[:, None] + 1j * vf[None, :]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        F = Ffn(W, jmf)
    Xf = _dperiodic_integrate(F, uf, vf)
    X = Xf[::m, ::m, :].copy()
    kidx = {k: uidx[k] for k in range(len(usplits))}
    arcsg = [(e, kidx[k0], kidx[k1], ax) for (e, k0, k1, ax) in arcs]

    res = {}
    # exact height: with dh = dz/z, x3 = u identically
    res['x3=u'] = float(np.max(np.abs(
        X[..., 2] - (X[0, 0, 2] - ug[0] + ug[:, None]))))
    X[..., 2] = ug[:, None]

    def med(e, i0, i1, ax):
        return _dperiodic_arc_med(X, e, i0, i1, ax)

    # base alignment: first x1 wall and first x2 wall -> 0
    w1 = [(e, i0, i1) for (e, i0, i1, ax) in arcsg if ax == 0]
    w2 = [(e, i0, i1) for (e, i0, i1, ax) in arcsg if ax == 1]
    if w1:
        X[..., 0] -= med(*w1[0], 0)
    if w2:
        X[..., 1] -= med(*w2[0], 1)

    # wall targets: x2 walls sit on multiples of pi (dh = dz/z fixes
    # the y-period at 2 pi); x1 walls sit on multiples of pi for 'ks'
    # (square 2 pi lattice) or on {0, +-dx} otherwise, dx measured
    m2 = [med(e, i0, i1, 1) for (e, i0, i1) in w2]
    for (e, i0, i1), v in zip(w2, m2):
        res[f'x2wall@{e}:{i0}'] = abs(v - math.pi * round(v / math.pi))
    m1 = [med(e, i0, i1, 0) for (e, i0, i1) in w1]
    if style == 'ks':
        dx = math.pi
        for (e, i0, i1), v in zip(w1, m1):
            res[f'x1wall@{e}:{i0}'] = abs(v - math.pi
                                          * round(v / math.pi))
    else:
        dx = max(abs(v) for v in m1) if m1 else math.pi
        if dx < 1e-9:
            dx = math.pi
        for (e, i0, i1), v in zip(w1, m1):
            res[f'x1wall@{e}:{i0}'] = abs(v - dx * round(v / dx))

    # snap every wall arc exactly onto its target plane
    for (e, i0, i1, ax) in arcsg:
        seg = _dperiodic_arc_slice(X, e, i0, i1)
        core = seg[1:-1] if len(seg) > 4 else seg
        v = float(np.median(core[:, ax]))
        step = math.pi if (ax == 1 or style == 'ks') else dx
        seg[:, ax] = step * round(v / step)

    # ops + lattice per style
    if style == 'half8':
        ops = [(np.diag([sx, sy, sz]), np.zeros(3), sx * sy * sz)
               for sx in (1.0, -1.0) for sy in (1.0, -1.0)
               for sz in (1.0, -1.0)]
        T1 = np.array([2.0 * dx, 0.0, 0.0])
        T2 = np.array([0.0, 2.0 * math.pi, 0.0])
    elif style == 'full4':
        ops = [(np.diag([sx, sy, 1.0]), np.zeros(3), sx * sy)
               for sx in (1.0, -1.0) for sy in (1.0, -1.0)]
        T1 = np.array([2.0 * dx, 0.0, 0.0])
        T2 = np.array([0.0, 2.0 * math.pi, 0.0])
    else:                                            # ks
        # u = 0 edge is a straight horizontal line along (1, 1, 0) or
        # (1, -1, 0): snap the constant diagonal combination
        line = X[-1, :, :]
        d1 = float(np.ptp(line[1:-1, 0] - line[1:-1, 1]))
        d2 = float(np.ptp(line[1:-1, 0] + line[1:-1, 1]))
        sgn = 1.0 if d1 <= d2 else -1.0            # x1 - sgn*x2 const
        res['line'] = min(d1, d2)
        delta = float(np.median(line[1:-1, 0] - sgn * line[1:-1, 1]))
        # the line offset must land on the pi wall lattice for the
        # reflection group to close -- its deviation IS a period
        # residual; snap it exactly
        tgt = math.pi * round(delta / math.pi)
        res['line_off'] = abs(delta - tgt)
        delta = tgt
        s = line[:, 0] + sgn * line[:, 1]
        line[:, 0] = 0.5 * (s + delta)
        line[:, 1] = sgn * 0.5 * (s - delta)
        line[:, 2] = 0.0
        # the u = 0 corners sit exactly ON the circle branch points
        # z = +-1, where the quadrature is unreliable: replace them by
        # the exact intersection of the line with the adjacent
        # (already snapped) wall plane
        nlast = X.shape[0] - 1
        for (e, i0, i1, ax) in arcsg:
            if i1 != nlast:
                continue
            j = 0 if e == 0 else X.shape[1] - 1
            wall = X[i1 - 1, j, ax] if i1 > i0 else X[i0, j, ax]
            wall = math.pi * round(float(wall) / math.pi)
            if ax == 0:
                X[nlast, j, 0] = wall
                X[nlast, j, 1] = sgn * (wall - delta)
            else:
                X[nlast, j, 1] = wall
                X[nlast, j, 0] = delta + sgn * wall
            X[nlast, j, 2] = 0.0
        RL = np.array([[0.0, sgn, 0.0], [sgn, 0.0, 0.0],
                       [0.0, 0.0, -1.0]])
        tL = np.array([delta, -sgn * delta, 0.0])
        gens = [(np.diag([-1.0, 1.0, 1.0]), np.zeros(3), -1.0),
                (np.diag([1.0, -1.0, 1.0]), np.zeros(3), -1.0),
                (RL, tL, -1.0)]
        T1 = np.array([2.0 * math.pi, 0.0, 0.0])
        T2 = np.array([0.0, 2.0 * math.pi, 0.0])
        ops = _dptail_close_ops(gens, T1, T2)
    return {'X': X, 'ug': ug, 'vg': vg, 'arcs': arcsg, 'ops': ops,
            'T1': T1, 'T2': T2, 'res': res, 'key': key,
            'style': style}


def _dptail_close_ops(gens, T1, T2, cap=40):
    """Close the generator list into a finite op set modulo the
    lattice {T1, T2} (translations reduced componentwise into the
    centered fundamental cell).  Ops are (M, t, parity)."""
    L = np.stack([T1[:2], T2[:2]])

    def reduce_t(t):
        # canonical representative in the HALF-OPEN centered cell:
        # components landing at +half-period (within tolerance) wrap
        # to -half-period so +-pi never split into distinct keys
        t = t.copy()
        c = np.linalg.solve(L.T, t[:2])
        c -= np.round(c)
        c[c > 0.5 - 1e-7] -= 1.0
        t[:2] = L.T @ c
        return t

    def key_of(M, t):
        return (tuple(np.round(M, 6).ravel()),
                tuple(np.round(t, 6)))

    ops = {key_of(np.eye(3), np.zeros(3)):
           (np.eye(3), np.zeros(3), 1.0)}
    frontier = list(ops.values())
    while frontier and len(ops) < cap:
        nxt = []
        for (M, t, par) in frontier:
            for (Mg, tg, pg) in gens:
                M2 = Mg @ M
                t2 = reduce_t(Mg @ t + tg)
                k = key_of(M2, t2)
                if k not in ops:
                    ops[k] = (M2, t2, par * pg)
                    nxt.append(ops[k])
        frontier = nxt
    return list(ops.values())


def dptail_assemble(P, cells=(1, 1)):
    """Tile the patch orbit over cells[0] x cells[1] lattice cells at
    the true period vectors and weld all seams (general (M, t, parity)
    ops).  Returns (V, quads, per-vertex UV)."""
    X = P['X']
    nu, nv = X.shape[:2]
    Vp = X.reshape(-1, 3)
    gu = np.arange(nu) / max(nu - 1, 1)
    gv = np.arange(nv) / max(nv - 1, 1)
    UVp = np.stack(np.meshgrid(gu, gv, indexing='ij'),
                   axis=-1).reshape(-1, 2)
    ii, jj = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1),
                         indexing='ij')
    b = (ii * nv + jj).ravel()
    q0 = np.stack([b, b + nv, b + nv + 1, b + 1], axis=1)
    cu, cv = int(max(1, cells[0])), int(max(1, cells[1]))
    Vs, Qs, UVs = [], [], []
    off = 0
    for ic in range(cu):
        for jc in range(cv):
            t0 = ((ic - 0.5 * (cu - 1)) * P['T1']
                  + (jc - 0.5 * (cv - 1)) * P['T2'])
            for (M, t, parity) in P['ops']:
                Vs.append(Vp @ M.T + (t + t0)[None, :])
                UVs.append(UVp)
                Qs.append((q0 if parity > 0 else q0[:, ::-1]) + off)
                off += len(Vp)
    V = np.concatenate(Vs, axis=0)
    UV = np.concatenate(UVs, axis=0)
    Q = np.concatenate(Qs, axis=0)
    return _dperiodic_weld(V, Q, UV)


def dptail_quotient(P):
    """Topology of the translational quotient (same contract as
    dperiodic_quotient): chi, boundary loops, nonmanifold edge count,
    oriented, components -- ends stay as boundary rims."""
    V, quads, _ = dptail_assemble(P, (1, 1))
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    key = np.round(V * 1e6).astype(np.int64)
    hkey = {tuple(k): i for i, k in enumerate(key)}
    # identify under ALL 8 neighbor translations: general (M, t) ops
    # can place seam partners a composite T1 +- T2 step apart
    shifts = [i1 * P['T1'] + i2 * P['T2']
              for i1 in (-1, 0, 1) for i2 in (-1, 0, 1)
              if (i1, i2) != (0, 0)]
    for T in shifts:
        kt = np.round((V + T[None, :]) * 1e6).astype(np.int64)
        for i, k in enumerate(kt):
            j = hkey.get(tuple(k))
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    root = np.array([find(i) for i in range(len(V))])
    ec, dc = {}, {}
    for f in quads:
        mlen = len(f)
        for k in range(mlen):
            a2, b2 = int(root[f[k]]), int(root[f[(k + 1) % mlen]])
            e = (a2, b2) if a2 < b2 else (b2, a2)
            ec[e] = ec.get(e, 0) + 1
            dc[(a2, b2)] = dc.get((a2, b2), 0) + 1
    nvq = len({int(r) for f in quads for r in root[list(f)]})
    chi = nvq - len(ec) + len(quads)
    nonman = sum(1 for c in ec.values() if c > 2)
    orient = all(c == 1 for c in dc.values())
    bed = [e for e, c in ec.items() if c == 1]
    par = {}

    def bfind(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a2, b2 in bed:
        ra, rb = bfind(a2), bfind(b2)
        if ra != rb:
            par[ra] = rb
    loops = len({bfind(a2) for a2, b2 in bed})
    parc = {}

    def cfind(x):
        parc.setdefault(x, x)
        while parc[x] != x:
            parc[x] = parc[parc[x]]
            x = parc[x]
        return x

    for f in quads:
        for i2 in range(1, len(f)):
            ra, rb = cfind(int(root[f[0]])), cfind(int(root[f[i2]]))
            if ra != rb:
                parc[ra] = rb
    ncomp = len({cfind(int(root[i2])) for f in quads for i2 in f})
    return chi, loops, nonman, orient, ncomp


_DPTAIL_CACHE = {}


def dptail_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                cells=(1, 1)):
    """Finished-mesh builder for the doubly periodic tail rows
    (toolkit MESH_PARAM / cells2d contract)."""
    tk = _toolkit()
    p = spec['p_from'](order, radius) if 'p_from' in spec else {}
    key = p.pop('dpt_key')
    mem = dptail_member(key)
    nsp = max(6, len(mem['factors']))
    nuw = int(np.clip(nu * nsp / 6.0, 32, 320))
    if isinstance(cells, (int, float)):
        cells = (int(cells), 1)
    cu = int(np.clip(cells[0], 1, 8))
    cv = int(np.clip(cells[1] if len(cells) > 1 else 1, 1, 8))
    ck = (key, tuple(sorted(p.items())), int(nuw), int(nv))
    P = _DPTAIL_CACHE.get(ck)
    if P is None:
        P = dptail_patch(key, p, nuw, int(np.clip(nv, 24, 220)))
        if len(_DPTAIL_CACHE) > 10:
            _DPTAIL_CACHE.clear()
        _DPTAIL_CACHE[ck] = P
    V, quads, UV = dptail_assemble(P, (cu, cv))
    Vu, quads = tk._largest_component(np.hstack([V, UV]), quads)
    V, UV = Vu[:, :3], Vu[:, 3:]
    V = tk._center_fit(V, scale, V)
    return V, quads, UV


# ==========================================================================
# Singly periodic long tail (sptail_* block): six-ended Scherk towers,
# alternating fence of half-catenoids, fence of catenoids, helicoidal
# Karcher-Scherk (twisted saddle tower), translation-invariant Enneper
# surfaces
# ==========================================================================
# A family of Karcher-style singly periodic minimal surfaces, all meshed
# by the same scheme: integrate the Weierstrass 1-forms over ONE
# fundamental patch (compound Gauss-Legendre per grid cell, so singular
# corner nodes are never evaluated), snap the patch boundary exactly onto
# its symmetry elements (straight lines / mirror planes measured from the
# integrand's real/imaginary structure on the domain boundary), orbit the
# patch under the surface's isometry group plus `storeys` translation /
# screw copies, and weld seam-exactly.  Fractional powers are always
# taken in the LIMIT FROM THE DOMAIN INTERIOR (boundary points get a tiny
# upper-half-plane nudge), so no branch cut is ever crossed -- evaluating
# exactly ON a cut is where numpy's principal branch silently picks the
# wrong side.
#
# Period problems.  Every shipped member closes its period problem with a
# machine-checkable residual:
#   * six-ended Scherk (g = 0): rho and the branch point a are CLOSED
#     FORMS in the end-angle phi (rho = sqrt((1-cos phi)/(1+cos phi)),
#     a = 4 rho^4/(1+rho^2)^2); the residue of om2 at z = 0 gives the
#     translation dy = -(pi/4)(1+rho^2)/rho, verified against the
#     integrated patch.
#   * alternating fence of half-catenoids (g = 1): rho = 1/sqrt(a)
#     closes the horizontal period -- the patch's two vertical-mirror
#     segments land in the SAME plane (measured ~1e-15).
#   * helicoidal Karcher-Scherk: one real parameter R solved here by
#     bisection on the GEOMETRIC closure residual (the k-fold axis point
#     of the w = 0 line pair must coincide with that of the w = infinity
#     pair); the vertical rise then reproduces pi R^2/(1+R^4) to 1e-14.
#   * fence of catenoids: rho = sqrt(a) is the Lopez-Ros normalization
#     that makes the horizontal arc |z| = sqrt(a) a genuine planar
#     geodesic: |G| = 1 identically on the arc (an exact algebraic
#     identity at rho = sqrt(a), checked in the tests), so the
#     sigma_h reflection is a true symmetry; a is a free neck modulus.
#     MEASURED topology: quotient chi = -2 with 2 ends -- genus 1 per
#     period (a chain of catenoids joined neck to neck; the harvest's
#     "genus 0" annotation is off by the handle, as its metadata has
#     been before).
#   * periodic Enneper / translation-invariant Enneper with three
#     annular ends: g = z resp. g = (1/sqrt2)(1-z^2)/z with dh = dz --
#     closed-form antiderivatives, no constants at all.
#
# Topology is MEASURED, not assumed: the self-tests check the Euler
# characteristic of the welded stacks (chi(S+1) - chi(S) must equal the
# quotient chi = 2 - 2 genus - #ends) and/or of the translation-wrapped
# quotient, plus edge-manifoldness, orientability and connectivity.
# Note on the three-annular-end Enneper: its harvest metadata says
# genus 1 ("translation invariant torus"), but the surface built from
# the notebook's own rational data g = (1/sqrt2)(1-z^2)/z, dh = dz is
# parametrized by the four-punctured SPHERE (both g and dh are rational
# in z, so the quotient is genus 0 with 1 Enneper + 3 annular ends,
# chi = -2) -- which is exactly what the measured Euler characteristic
# gives.  This limit member is shipped with its measured genus.
#
# References:
#   H. Karcher, "Embedded minimal surfaces derived from Scherk's
#     examples", Manuscripta Math. 62 (1988) 83-114 -- saddle towers,
#     their helicoidal (screw-motion) deformations, and the fence of
#     catenoids;
#   H. F. Scherk (1835) and A. Enneper (1864) -- the classical data the
#     towers and the periodic Enneper interpolate;
#   M. Weber, https://minimalsurfaces.blog/ -- the harvested notebooks
#     (6-ended Scherk g=0; Alternating Fence of Half-Catenoids, 2024;
#     Helicoidal Karcher-Scherk; Fence of Catenoids; Periodic Enneper;
#     Enneper-Scherk; Translation-Invariant Torus with One Enneper and
#     Three Annular Ends, notebook by Ramazan Yol, 2024) --
#     research/msblog_harvest/singly_periodic.json.
# --------------------------------------------------------------------------

_SPTAIL_GL = np.polynomial.legendre.leggauss(8)


def sptail_cluster(h0, hmin=2e-7, ratio=0.3):
    """Geometric offsets h0 * ratio^k down to hmin (largest first)."""
    offs = []
    h = h0
    while h > hmin:
        offs.append(h)
        h *= ratio
    return np.array(offs)


def sptail_nudge(z):
    """Upper-half-plane limit for boundary evaluation: points exactly on
    the real axis get a tiny positive imaginary part, so every principal
    fractional power / log is continuous with the domain interior."""
    return np.where(np.imag(z) == 0.0, z + 1e-14j * (1.0 + np.abs(z)), z)


def sptail_polar_patch(om, r, t):
    """Integrate the 1-form triple om(z) over the polar grid (r_j, t_m)
    in the upper half plane.  Spine: the column nearest t = pi/2,
    cumulative in r; rows: arcs at fixed r outward from the spine.
    Compound GL-8 per cell -- nodes are strictly interior, so singular
    boundary nodes (branch points on the real axis) are never hit."""
    xg, wg = _SPTAIL_GL
    nr, nt = len(r), len(t)
    jm = int(np.argmin(np.abs(t - 0.5 * math.pi)))
    dr = np.diff(r)
    rmid = 0.5 * (r[1:] + r[:-1])
    zs = (rmid[:, None] + 0.5 * dr[:, None] * xg[None, :]) \
        * np.exp(1j * t[jm])
    o = om(zs)
    incS = np.stack([np.sum(c * wg, axis=-1) for c in o], axis=-1) \
        * (0.5 * np.exp(1j * t[jm]) * dr)[:, None]
    S = np.zeros((nr, 3), complex)
    S[1:] = np.cumsum(incS, axis=0)
    dt = np.diff(t)
    tmid = 0.5 * (t[1:] + t[:-1])
    Z = r[:, None, None] * np.exp(
        1j * (tmid[None, :, None] + 0.5 * dt[None, :, None] * xg))
    o = om(Z)
    incA = np.stack(
        [np.sum(c * (1j * Z) * wg, axis=-1) for c in o], axis=-1) \
        * (0.5 * dt)[None, :, None]
    C = np.zeros((nr, nt, 3), complex)
    C[:, 1:, :] = np.cumsum(incA, axis=1)
    C = C - C[:, jm, :][:, None, :]
    return np.real(S[:, None, :] + C)


def sptail_rect_patch(om, x, y):
    """Same scheme on a rectangular strip grid (x_i, y_j): spine down
    the column nearest x = 0 (dz = i dy), rows along x."""
    xg, wg = _SPTAIL_GL
    nx, ny = len(x), len(y)
    jm = int(np.argmin(np.abs(x)))
    dy = np.diff(y)
    ymid = 0.5 * (y[1:] + y[:-1])
    zs = x[jm] + 1j * (ymid[:, None] + 0.5 * dy[:, None] * xg[None, :])
    o = om(zs)
    incS = np.stack([np.sum(c * wg, axis=-1) for c in o], axis=-1) \
        * (0.5j * dy)[:, None]
    S = np.zeros((ny, 3), complex)
    S[1:] = np.cumsum(incS, axis=0)
    dx = np.diff(x)
    xmid = 0.5 * (x[1:] + x[:-1])
    Z = (xmid[:, None, None] + 0.5 * dx[:, None, None] * xg[None, None, :]) \
        + 1j * y[None, :, None]
    o = om(Z)
    incR = np.stack([np.sum(c * wg, axis=-1) for c in o], axis=-1) \
        * (0.5 * dx)[:, None, None]
    C = np.zeros((nx, ny, 3), complex)
    C[1:] = np.cumsum(incR, axis=0)
    C = C - C[jm][None, :, :]
    return np.real(S[None, :, :] + C)


def sptail_grid_quads(nr, nt, valid=None):
    """Grid quads (i, j) -> (i, j+1) -> ...; drop any touching an
    invalid (masked) vertex."""
    q = []
    for i in range(nr - 1):
        for j in range(nt - 1):
            f = (i * nt + j, i * nt + j + 1,
                 (i + 1) * nt + j + 1, (i + 1) * nt + j)
            if valid is None or (valid[f[0]] and valid[f[1]]
                                 and valid[f[2]] and valid[f[3]]):
                q.append(f)
    return q


def sptail_orbit_weld(V0, UV0, quads0, frames, tol):
    """Tile V0 under `frames` [(M, tvec, parity), ...] and weld
    coincident vertices (two offset quantization passes, so a pair
    straddling a rounding boundary is still caught).  Carries the
    per-vertex uv chart; drops vertices no face uses.  Seam vertices are
    bitwise-exact after the callers' boundary snapping, so the weld
    tolerance only needs to absorb float rounding."""
    Vp, Fp = [], []
    n0 = len(V0)
    for i, (M, tv, par) in enumerate(frames):
        Vp.append(V0 @ M.T + tv)
        off = i * n0
        if par < 0:
            Fp.extend(tuple(off + k for k in f[::-1]) for f in quads0)
        else:
            Fp.extend(tuple(off + k for k in f) for f in quads0)
    V = np.concatenate(Vp, axis=0)
    UV = np.tile(UV0, (len(frames), 1))
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    for shift in (0.0, 0.5):
        key = np.round(V / tol + shift).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        first = {}
        for i in range(len(V)):
            b = int(inv[i])
            if b in first:
                ra, rb = find(i), find(first[b])
                if ra != rb:
                    parent[ra] = rb
            else:
                first[b] = i
    roots = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    nw = len(uniq)
    Vw = np.zeros((nw, 3))
    UVw = np.zeros((nw, 2))
    cnt = np.zeros(nw)
    np.add.at(Vw, inv, V)
    np.add.at(UVw, inv, UV)
    np.add.at(cnt, inv, 1)
    Vw /= cnt[:, None]
    UVw /= cnt[:, None]
    F = []
    for f in Fp:
        g = [int(inv[i]) for i in f]
        h = [g[0]]
        for s in range(1, len(g)):
            if g[s] != h[-1]:
                h.append(g[s])
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            F.append(tuple(h))
    used = sorted({i for f in F for i in f})
    remap = np.full(nw, -1, dtype=np.int64)
    remap[np.array(used, dtype=np.int64)] = np.arange(len(used))
    return (Vw[np.array(used, dtype=np.int64)],
            [tuple(int(remap[i]) for i in f) for f in F],
            UVw[np.array(used, dtype=np.int64)])


def sptail_topology(V, F):
    """(chi, nonmanifold_edges, oriented, boundary_loops, components)."""
    ec, dc = {}, {}
    for f in F:
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            e = (a, b) if a < b else (b, a)
            ec[e] = ec.get(e, 0) + 1
            dc[(a, b)] = dc.get((a, b), 0) + 1
    chi = len(V) - len(ec) + len(F)
    nonman = sum(1 for c in ec.values() if c > 2)
    orient = all(c == 1 for c in dc.values())
    bed = [e for e, c in ec.items() if c == 1]
    par = {}

    def bf(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in bed:
        ra, rb = bf(a), bf(b)
        if ra != rb:
            par[ra] = rb
    loops = len({bf(a) for a, b in bed})
    parc = list(range(len(V)))

    def cf(a):
        while parc[a] != a:
            parc[a] = parc[parc[a]]
            a = parc[a]
        return a
    for f in F:
        for i in range(1, len(f)):
            ra, rb = cf(f[0]), cf(f[i])
            if ra != rb:
                parc[ra] = rb
    ncomp = len({cf(f[0]) for f in F})
    return chi, nonman, orient, loops, ncomp


def sptail_quotient(V, F, Tvec, span, kmax=5):
    """Weld open-boundary vertices under v -> v + k Tvec (k = 1..kmax):
    the translation-wrapped quotient of a ONE-period stack.  Test-only
    (measures the true per-period chi and end-rim count)."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for f in F:
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            cnt[(a, b) if a < b else (b, a)] += 1
    bnd = sorted({v for e, c in cnt.items() if c == 1 for v in e})
    B = V[bnd]
    tol = 1e-6 * span
    key = {tuple(np.round(B[i] / tol).astype(np.int64)): bnd[i]
           for i in range(len(bnd))}
    parent = list(range(len(V)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for k in range(1, kmax + 1):
        for i in range(len(bnd)):
            j = key.get(tuple(np.round(
                (B[i] + k * Tvec) / tol).astype(np.int64)))
            if j is not None:
                ra, rb = find(bnd[i]), find(j)
                if ra != rb:
                    parent[ra] = rb
    roots = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(roots, return_inverse=True)
    Vq = V[uniq]
    Fq = []
    for f in F:
        g = [int(inv[i]) for i in f]
        h = [g[0]]
        for s in g[1:]:
            if s != h[-1]:
                h.append(s)
        if len(h) >= 3 and h[0] != h[-1] and len(set(h)) == len(h):
            Fq.append(tuple(h))
    return Vq, Fq


def _sptail_grid_uv(nr, nt):
    gu = np.arange(nr) / max(nr - 1, 1)
    gv = np.arange(nt) / max(nt - 1, 1)
    return np.stack(np.meshgrid(gu, gv, indexing='ij'),
                    axis=-1).reshape(-1, 2)


# ---- six-ended Scherk tower (genus 0, 6 ends per period) -----------------

def sptail_six_om(phid):
    """Closed-form data for end angle phid (degrees, 0 < phid < 90):
    rho, a, and the 1-form triple."""
    ph = math.radians(phid)
    rho = math.sqrt((1.0 - math.cos(ph)) / (1.0 + math.cos(ph)))
    a = 4.0 * rho ** 4 / (1.0 + rho ** 2) ** 2

    def om(z):
        z = sptail_nudge(z)
        s_za = np.sqrt(z - a)
        s_z1 = np.sqrt(z - 1.0)
        p1 = rho / z / s_za * s_z1
        p2 = 1.0 / rho / s_za / s_z1
        return (-0.5 * (p1 - p2), 0.5j * (p1 + p2),
                1.0 / (np.sqrt(z) * s_za))
    return rho, a, om


def sptail_six_build(phid=30.0, nu=48, nt=32, storeys=1, rmin=1e-4,
                     rmax=30.0):
    """One 8-frame period block ({E, R_a} x {E, Mx} x {E, My}) per
    storey, translation (0, 4 dy, 0).  Returns (V, F, uv, diag)."""
    rho, a, om = sptail_six_om(phid)
    rE = np.exp(np.linspace(math.log(rmin), math.log(a * 0.8),
                            max(8, int(0.22 * nu))))
    cl = sptail_cluster(0.15 * a, hmin=2e-6 * a)
    rC = np.linspace(a * 1.4, 0.9, max(8, int(0.25 * nu)))
    cl1 = sptail_cluster(0.08, hmin=2e-6)
    rO = np.exp(np.linspace(0.0, math.log(rmax),
                            max(8, int(0.30 * nu))))[1:]
    r = np.unique(np.concatenate(
        [rE, a - cl, [a], a + cl[::-1], rC, 1.0 - cl1, [1.0],
         1.0 + cl1[::-1], rO]))
    r = r[np.concatenate([[True], np.diff(r) > 1e-13])]
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0, 1, 2 * (nt // 2) + 1)))
    ex = sptail_cluster(0.06, hmin=1e-7, ratio=0.25)
    t = np.unique(np.concatenate([base, ex, math.pi - ex]))
    t = t[np.concatenate([[True], np.diff(t) > 1e-13])]
    X = sptail_polar_patch(om, r, t)
    i_a = int(np.searchsorted(r, a))
    i_1 = int(np.searchsorted(r, 1.0))
    diag = {
        'line_dy': float(np.ptp(X[:i_a + 1, 0, 1])),
        'line_dz': float(np.ptp(X[:i_a + 1, 0, 2])),
        'mx_dx': float(np.ptp(X[i_a:i_1 + 1, 0, 0])),
        'my_dy': float(np.ptp(X[i_1:, 0, 1])),
        'pi_dy': float(np.ptp(X[:, -1, 1]))}
    X = X.copy()
    X[..., 0] -= np.median(X[i_a:i_1 + 1, 0, 0])
    X[..., 1] -= np.median(X[i_1:, 0, 1])
    X[..., 2] -= np.median(X[:i_a + 1, 0, 2])
    ya = float(np.median(X[:i_a + 1, 0, 1]))
    ypi = float(np.median(X[:, -1, 1]))
    dy_theory = -(math.pi / 4) * (1.0 + rho * rho) / rho
    diag['dy_vs_residue'] = abs(ya - dy_theory)
    diag['pi_vs_2dy'] = abs(ypi - 2.0 * ya)
    X[:i_a + 1, 0, 1] = ya
    X[:i_a + 1, 0, 2] = 0.0
    X[i_a:i_1 + 1, 0, 0] = 0.0
    X[i_1:, 0, 1] = 0.0
    X[:, -1, 1] = 2.0 * ya
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(r), len(t))
    T = np.array([0.0, 4.0 * ya, 0.0])
    frames = []
    for e in (0, 1):
        for bx in (0, 1):
            for cy in (0, 1):
                M = np.eye(3)
                tv = np.zeros(3)
                if e:                      # R_a: (x, 2 ya - y, -z)
                    M = M @ np.diag([1.0, -1.0, -1.0])
                    tv = np.array([0.0, 2.0 * ya, 0.0])
                if bx:
                    M = np.diag([-1.0, 1.0, 1.0]) @ M
                    tv = np.array([-1.0, 1.0, 1.0]) * tv
                if cy:
                    M = np.diag([1.0, -1.0, 1.0]) @ M
                    tv = np.array([1.0, -1.0, 1.0]) * tv
                par = (-1.0) ** (e + bx + cy)
                for s in range(storeys):
                    frames.append((M, tv + s * T, par))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(t)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def sptail_six_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                    storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sptail_six_build(
        p['phid'], nu=int(np.clip(int(0.75 * nu), 24, 120)),
        nt=int(np.clip(int(0.55 * nv), 20, 80)), storeys=S,
        rmax=p['rmax'], rmin=1.0 / p['rmax'])
    tk = _toolkit()
    V = tk._center_fit(V, scale, V)
    return V, F, uv


# ---- alternating fence of half-catenoids (genus 1, 2 ends) ---------------

def sptail_fencealt_om(a):
    rho = 1.0 / math.sqrt(a)

    def om(z):
        z = sptail_nudge(z)
        p1 = rho * (z + 1.0 / a) ** -0.5 * z ** -1.5 * (z - a) ** 0.5
        p2 = (1.0 / rho) * (z + 1.0 / a) ** 0.5 * z ** -0.5 \
            * (z - a) ** -0.5
        return (0.5 * (p2 - p1), 0.5j * (p2 + p1), 1.0 / z)
    return om


def sptail_fencealt_build(a=2.0, nu=48, nt=32, storeys=1, rmin=0.05):
    """{E, Mx} x {E, My} per storey, translation (0, 2 dy, 0).  The
    period problem is closed by rho = 1/sqrt(a): the two vertical-mirror
    segments (theta = 0 past a, theta = pi past -1/a) land in the SAME
    plane -- measured in diag['period_x']."""
    om = sptail_fencealt_om(a)
    rmax = 1.0 / rmin
    b1, b2 = 1.0 / a, a
    cl1 = sptail_cluster(0.10 * b1, hmin=1e-6 * b1)
    cl2 = sptail_cluster(0.10 * (b2 - b1), hmin=1e-6)
    n3 = max(8, int(0.30 * nu))
    rA = np.exp(np.linspace(math.log(rmin), math.log(b1 * 0.85), n3))
    rB = np.exp(np.linspace(math.log(b1 * 1.15), math.log(b2 * 0.85),
                            n3))
    rC = np.exp(np.linspace(math.log(b2 * 1.15), math.log(rmax), n3))
    r = np.unique(np.concatenate(
        [rA, b1 - cl1, [b1], b1 + cl1[::-1], rB, b2 - cl2, [b2],
         b2 + cl2[::-1], rC]))
    r = r[np.concatenate([[True], np.diff(r) > 1e-13])]
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0, 1, 2 * (nt // 2) + 1)))
    ex = sptail_cluster(0.06, hmin=1e-7, ratio=0.25)
    t = np.unique(np.concatenate([base, ex, math.pi - ex]))
    t = t[np.concatenate([[True], np.diff(t) > 1e-13])]
    X = sptail_polar_patch(om, r, t)
    i1 = int(np.searchsorted(r, b1))
    i2 = int(np.searchsorted(r, b2))
    xa = float(np.median(X[:i2 + 1, 0, 0]))
    x2 = float(np.median(X[i1:, -1, 0]))
    diag = {
        'xa_ptp': float(np.ptp(X[:i2 + 1, 0, 0])),
        'y0_ptp': float(np.ptp(X[i2:, 0, 1])),
        'y1_ptp': float(np.ptp(X[:i1 + 1, -1, 1])),
        'x2_ptp': float(np.ptp(X[i1:, -1, 0])),
        'period_x': abs(xa - x2)}
    X = X.copy()
    X[..., 0] -= xa
    X[..., 1] -= np.median(X[i2:, 0, 1])
    dy = float(np.median(X[:i1 + 1, -1, 1]))
    X[:i2 + 1, 0, 0] = 0.0
    X[i2:, 0, 1] = 0.0
    X[:i1 + 1, -1, 1] = dy
    X[i1:, -1, 0] = 0.0
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(r), len(t))
    T = np.array([0.0, 2.0 * dy, 0.0])
    frames = []
    for bx in (0, 1):
        for cy in (0, 1):
            M = np.diag([-1.0 if bx else 1.0, -1.0 if cy else 1.0, 1.0])
            par = (-1.0) ** (bx + cy)
            for s in range(storeys):
                frames.append((M, s * T, par))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(t)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def sptail_fencealt_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                         storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sptail_fencealt_build(
        p['a'], nu=int(np.clip(int(0.8 * nu), 24, 130)),
        nt=int(np.clip(int(0.55 * nv), 20, 80)), storeys=S,
        rmin=p['rmin'])
    tk = _toolkit()
    V = tk._center_fit(V, scale, V)
    return V, F, uv


# ---- fence of catenoids (measured genus 1, 2 ends per period) ------------

def sptail_fencecat_om(a):
    rho = math.sqrt(a)

    def om(z):
        z = sptail_nudge(z)
        p1 = z ** -0.5 * (z - 1.0) ** -0.5 * (z - a) ** 0.5
        p2 = z ** -1.5 * (z - 1.0) ** 0.5 * (z - a) ** -0.5
        return (-0.5 * (rho * p1 - p2 / rho),
                0.5j * (rho * p1 + p2 / rho), 1.0 / z)
    return om


def sptail_fencecat_build(a=0.2, nu=48, nt=32, storeys=1, r1=6.0):
    """Patch: upper half annulus a/r1 <= |z| <= sqrt(a); boundary =
    y-mirror (0, a), TWO x-mirrors ((a, sqrt a) and (-sqrt a, 0)) whose
    plane gap is half the fence translation, the horizontal mirror arc
    |z| = sqrt(a) (height log(a)/2), and the catenoid end trim at a/r1.
    Frames {E, MxB} x {E, My} x {E, sigma_h} per storey, horizontal
    translation T = (2(xB - xA), 0, 0)."""
    om = sptail_fencecat_om(a)
    sa = math.sqrt(a)
    rlo = a / r1
    cl = sptail_cluster(0.15 * a, hmin=2e-6 * a)
    rA = np.exp(np.linspace(math.log(rlo), math.log(a * 0.85),
                            max(8, int(0.45 * nu))))
    rB = np.linspace(a * 1.15, sa, max(8, int(0.35 * nu)))
    r = np.unique(np.concatenate(
        [rA, a - cl, [a], a + cl[::-1], rB, [sa]]))
    r = r[np.concatenate([[True], np.diff(r) > 1e-13])]
    r = r[r <= sa + 1e-12]
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0, 1, 2 * (nt // 2) + 1)))
    ex = sptail_cluster(0.06, hmin=1e-7, ratio=0.25)
    t = np.unique(np.concatenate([base, ex, math.pi - ex]))
    t = t[np.concatenate([[True], np.diff(t) > 1e-13])]
    X = sptail_polar_patch(om, r, t)
    i_a = int(np.searchsorted(r, a))
    xA = float(np.median(X[i_a:, 0, 0]))       # (a, sqrt a) mirror
    xB = float(np.median(X[:, -1, 0]))         # (-sqrt a, 0) mirror
    diag = {
        'y_ptp': float(np.ptp(X[:i_a + 1, 0, 1])),
        'xA_ptp': float(np.ptp(X[i_a:, 0, 0])),
        'xB_ptp': float(np.ptp(X[:, -1, 0])),
        'arc_z_ptp': float(np.ptp(X[-1, :, 2]))}
    X = X.copy()
    X[..., 0] -= xA
    X[..., 1] -= np.median(X[:i_a + 1, 0, 1])
    X[..., 2] -= np.median(X[-1, :, 2])
    xB = float(np.median(X[:, -1, 0]))
    X[i_a:, 0, 0] = 0.0
    X[:i_a + 1, 0, 1] = 0.0
    X[:, -1, 0] = xB
    X[-1, :, 2] = 0.0
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(r), len(t))
    T = np.array([2.0 * xB, 0.0, 0.0])
    diag['T'] = T
    frames = []
    for bx in (0, 1):
        for cy in (0, 1):
            for sh in (0, 1):
                M = np.diag([-1.0 if bx else 1.0,
                             -1.0 if cy else 1.0,
                             -1.0 if sh else 1.0])
                tv = np.array([2.0 * xB if bx else 0.0, 0.0, 0.0])
                par = (-1.0) ** (bx + cy + sh)
                for s in range(storeys):
                    frames.append((M, tv + s * T, par))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(t)),
                                 q0, frames, 1e-9 * span)
    diag['span'] = span
    return V, F, uv, diag


def sptail_fencecat_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                         storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 8))
    V, F, uv, _ = sptail_fencecat_build(
        p['a'], nu=int(np.clip(int(0.8 * nu), 24, 130)),
        nt=int(np.clip(int(0.6 * nv), 20, 90)), storeys=S, r1=p['r1'])
    tk = _toolkit()
    V = tk._center_fit(V, scale, V)
    return V, F, uv


# ---- helicoidal Karcher-Scherk (twisted saddle tower) --------------------

def sptail_hks_om(k, a, R):
    def om(w):
        p1 = 0.5 * R ** 2 * (R ** 2 - w) ** (-1 + a) \
            * w ** (1 - 1.0 / k) * (1 + R ** 2 * w) ** (-1 - a)
        p2 = -0.5 * R ** 2 * (R ** 2 - w) ** (-1 - a) \
            * w ** (-1 + 1.0 / k) * (1 + R ** 2 * w) ** (-1 + a)
        dh = 0.5j / (w * (1.0 / R ** 2 - R ** 2 - 1.0 / w + w))
        return (-0.5 * (p1 - p2), 0.5j * (p2 + p1), dh)
    return om


def _sptail_hks_patch(k, a, R, nu, nt, xmax):
    om = sptail_hks_om(k, a, R)
    v1 = -math.log(R * R)

    def omz(zeta):
        e = np.exp(zeta)
        w = (e + R ** 2) / (1.0 - e * R ** 2)
        w = sptail_nudge(w)
        dw = e * (1.0 + R ** 4) / (1.0 - e * R ** 2) ** 2
        return tuple(c * dw for c in om(w))

    cl = sptail_cluster(0.25, hmin=3e-6, ratio=0.35)
    nseg = max(6, int(nu / 4))
    xs = [np.linspace(-xmax, -v1 - 0.3, nseg), -v1 - cl, [-v1],
          -v1 + cl[::-1], np.linspace(-v1 + 0.3, -0.02,
                                      max(4, nseg // 2)), [0.0],
          np.linspace(0.02, v1 - 0.3, max(4, nseg // 2)),
          v1 - cl, [v1], v1 + cl[::-1], np.linspace(v1 + 0.3, xmax,
                                                    nseg)]
    x = np.unique(np.concatenate([np.atleast_1d(np.asarray(s, float))
                                  for s in xs]))
    x = x[np.concatenate([[True], np.diff(x) > 1e-12])]
    base = 0.5 * math.pi * (1.0 - np.cos(np.pi * np.linspace(
        0, 1, 2 * (nt // 2) + 1)))
    ex = sptail_cluster(0.05, hmin=1e-6, ratio=0.3)
    y = np.unique(np.concatenate([base, ex, math.pi - ex]))
    y = y[np.concatenate([[True], np.diff(y) > 1e-12])]
    X = sptail_rect_patch(omz, x, y)
    iv1 = int(np.searchsorted(x, -v1))
    ip1 = int(np.searchsorted(x, v1))
    return X, x, y, iv1, ip1


def _sptail_line_fit(S):
    """2-D line through the xy-projection of a snapped boundary row
    (drop the sixth nearest the singular corner)."""
    n = len(S)
    cut = max(2, n // 6)
    P = S[:, :2]
    p = np.mean(P, axis=0)
    _, _, vt = np.linalg.svd(P - p, full_matrices=False)
    return p, vt[0]


def _sptail_isect(p1, u1, p2, u2):
    A = np.array([u1, -u2]).T
    tt = np.linalg.solve(A, p2 - p1)
    return p1 + tt[0] * u1


def sptail_hks_axis_residual(k, a, R, nu=32, nt=24, xmax=4.0):
    """Geometric period residual: the k-fold axis point fixed by the two
    w = 0 boundary lines minus the one fixed by the two w = infinity
    lines (y component; the residual is 1-D)."""
    X, x, y, iv1, ip1 = _sptail_hks_patch(k, a, R, nu, nt, xmax)

    def li(S, from_end):
        n = len(S)
        cut = max(2, n // 6)
        return _sptail_line_fit(S[:-cut] if from_end else S[cut:])
    q1 = _sptail_isect(*li(X[:iv1 + 1, -1, :], True),
                       *li(X[iv1:, -1, :], False))
    q2 = _sptail_isect(*li(X[:ip1 + 1, 0, :], True),
                       *li(X[ip1:, 0, :], False))
    return q2 - q1


_SPTAIL_HKS_R = {}


def sptail_hks_solveR(k, a):
    """Bisection on the geometric closure residual; cached."""
    key = (int(k), round(float(a), 8))
    if key in _SPTAIL_HKS_R:
        return _SPTAIL_HKS_R[key]
    lo, hi = 0.45, 0.985
    flo = sptail_hks_axis_residual(k, a, lo)[1]
    fhi = sptail_hks_axis_residual(k, a, hi)[1]
    if flo * fhi > 0:
        raise ValueError(f"hks solve: no bracket for k={k} a={a}")
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        fm = sptail_hks_axis_residual(k, a, mid)[1]
        if flo * fm < 0:
            hi = mid
        else:
            lo, flo = mid, fm
        if hi - lo < 1e-12:
            break
    R = 0.5 * (lo + hi)
    _SPTAIL_HKS_R[key] = R
    return R


def sptail_hks_build(k=4, a=0.1, nu=48, nt=36, storeys=1, xmax=4.0,
                     R=None):
    """{E, R_A} x k rotations x storey screws (twist -2 a pi, rise
    trans per period).  Returns (V, F, uv, diag)."""
    if R is None:
        R = sptail_hks_solveR(k, a)
    X, x, y, iv1, ip1 = _sptail_hks_patch(k, a, R, nu, nt, xmax)

    def li(S, from_end):
        n = len(S)
        cut = max(2, n // 6)
        return _sptail_line_fit(S[:-cut] if from_end else S[cut:])
    pA, uA = li(X[:iv1 + 1, -1, :], True)
    q1 = _sptail_isect(pA, uA, *li(X[iv1:, -1, :], False))
    q2 = _sptail_isect(*li(X[:ip1 + 1, 0, :], True),
                       *li(X[ip1:, 0, :], False))
    diag = {'axis_misalign': float(np.linalg.norm(q1 - q2)), 'R': R}
    zA = float(X[0, -1, 2])
    X = X.copy()
    X[..., 0] -= q1[0]
    X[..., 1] -= q1[1]
    X[..., 2] -= zA
    alA = math.atan2(uA[1], uA[0])
    ca, sa_ = math.cos(-alA), math.sin(-alA)
    X[..., :2] = X[..., :2] @ np.array([[ca, -sa_], [sa_, ca]]).T

    def snap_line(rows, beta, zval):
        u = np.array([math.cos(beta), math.sin(beta)])
        d = rows[:, :2] @ u
        rows[:, 0] = d * u[0]
        rows[:, 1] = d * u[1]
        rows[:, 2] = zval
    h = float(np.median(X[:ip1 + 1, 0, 2]))
    snap_line(X[:iv1 + 1, -1, :], 0.0, 0.0)
    snap_line(X[iv1:, -1, :], -math.pi / k, 0.0)
    snap_line(X[:ip1 + 1, 0, :], -a * math.pi, h)
    snap_line(X[ip1:, 0, :], -math.pi / k - a * math.pi, h)
    trans = 2.0 * h
    diag['trans'] = trans
    diag['trans_vs_theory'] = abs(abs(trans)
                                  - math.pi * R * R / (1.0 + R ** 4))
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(len(x), len(y))
    RA = np.diag([1.0, -1.0, -1.0])
    frames = []
    for j in range(storeys):
        for i in range(k):
            phi = -2.0 * a * math.pi * j - TAU * i / k
            c, s = math.cos(phi), math.sin(phi)
            Mz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            for e in (0, 1):
                M = Mz @ (RA if e else np.eye(3))
                frames.append((M, np.array([0.0, 0.0, j * trans]),
                               (-1.0) ** e))
    span = float(np.linalg.norm(V0.max(0) - V0.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(x), len(y)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = np.array([0.0, 0.0, trans])
    diag['span'] = span
    return V, F, uv, diag


def sptail_hks_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                    storeys=1):
    """The associate-angle knob is the TWIST modulus (like the alpha
    saddle tower): twist fraction f = theta/(pi/2) in [0, 1] maps to
    the exponent a = (0.12 + 0.78 f)/k; theta = 0 keeps a gentle
    default twist."""
    p = spec['p_from'](order, radius)
    k = p['k']
    f = float(np.clip(theta / (0.5 * math.pi), 0.0, 1.0))
    a = (0.12 + 0.78 * f) / k
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sptail_hks_build(
        k, a, nu=int(np.clip(int(0.7 * nu), 24, 110)),
        nt=int(np.clip(int(0.55 * nv), 18, 80)), storeys=S,
        xmax=p['xmax'])
    tk = _toolkit()
    V = tk._center_fit(V, scale, V)
    return V, F, uv


# ---- translation-invariant Enneper with three annular ends ---------------

_SPTAIL_SQ2 = math.sqrt(2.0)
_SPTAIL_E3A_Y1 = math.pi / (2.0 * _SPTAIL_SQ2)


def sptail_e3a_f(z):
    """Closed-form immersion: g = (1/sqrt2)(1-z^2)/z, dh = dz."""
    z = z + 1e-13j * (1.0 + np.abs(z))     # interior branch limit
    l1 = np.log(z)
    l2 = np.log(1.0 - z * z)
    x = np.real((z * z - 2.0 * l1 - 2.0 * l2) / (4.0 * _SPTAIL_SQ2))
    y = np.imag(z * z - 2.0 * l1 + 2.0 * l2) / (4.0 * _SPTAIL_SQ2)
    return np.stack(np.broadcast_arrays(x, y, np.real(z)), axis=-1)


def sptail_e3a_build(nu=56, nt=44, storeys=1, rmin=0.03, Rt=2.2,
                     eps=0.10):
    """Quarter-disk chart; {E, R_L} x {E, My} per storey, translation
    (0, -pi/sqrt2, 0).  R_L is the 2-fold rotation about the straight
    line {y = -Y1/2, z = 0} (the imaginary-axis image); the mirrors
    y = 0 / y = -Y1 sit on the real segments (0,1) / (1, Rt).  Genus 0
    with 1 Enneper + 3 annular ends per period (measured; see the
    block header note on the harvest's genus-1 annotation)."""
    Y1 = _SPTAIL_E3A_Y1
    cl = sptail_cluster(0.15, hmin=2e-3, ratio=0.45)
    r1 = np.exp(np.linspace(math.log(rmin), math.log(1.0 - cl[0] - 0.05),
                            max(8, int(0.4 * nu))))
    r2 = np.exp(np.linspace(math.log(1.0 + cl[0] + 0.05), math.log(Rt),
                            max(8, int(0.35 * nu))))
    r = np.unique(np.concatenate([r1, 1.0 - cl, [1.0], 1.0 + cl[::-1],
                                  r2]))
    th = 0.5 * math.pi * (0.5 - 0.5 * np.cos(np.pi * np.linspace(
        0, 1, nt)))
    R_, T_ = np.meshgrid(r, th, indexing='ij')
    Z = R_ * np.exp(1j * T_)
    with np.errstate(divide='ignore', invalid='ignore'):
        X = sptail_e3a_f(Z)
    mask = (np.abs(Z - 1.0) > eps) & np.isfinite(X).all(axis=-1)
    i1 = int(np.searchsorted(r, 1.0))
    diag = {
        'y01_ptp': float(np.ptp(X[:i1, 0, 1][mask[:i1, 0]])),
        'y1R_vs_const': abs(float(np.median(
            X[i1:, 0, 1][mask[i1:, 0]])) + Y1),
        'line_y_vs_const': abs(float(np.median(X[:, -1, 1])) + 0.5 * Y1),
        'line_z_ptp': float(np.ptp(X[:, -1, 2]))}
    X[:, -1, 1] = -0.5 * Y1
    X[:, -1, 2] = 0.0
    X[:i1, 0, 1] = 0.0
    X[i1:, 0, 1] = -Y1
    X[~mask] = 0.0
    V0 = X.reshape(-1, 3)
    vm = mask.reshape(-1)
    q0 = sptail_grid_quads(len(r), len(th), vm)
    T = np.array([0.0, -2.0 * Y1, 0.0])
    frames = []
    for e in (0, 1):
        for b in (0, 1):
            M = np.eye(3)
            tv = np.zeros(3)
            if e:                          # R_L: (x, -Y1 - y, -z)
                M = M @ np.diag([1.0, -1.0, -1.0])
                tv = np.array([0.0, -Y1, 0.0])
            if b:                          # My (y = 0)
                M = np.diag([1.0, -1.0, 1.0]) @ M
                tv = np.array([1.0, -1.0, 1.0]) * tv
            par = (-1.0) ** (e + b)
            for s in range(storeys):
                frames.append((M, tv + s * T, par))
    vu = V0[vm]
    span = float(np.linalg.norm(vu.max(0) - vu.min(0)))
    V, F, uv = sptail_orbit_weld(V0, _sptail_grid_uv(len(r), len(th)),
                                 q0, frames, 1e-9 * span)
    diag['T'] = T
    diag['span'] = span
    return V, F, uv, diag


def sptail_e3a_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                    storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 6))
    V, F, uv, _ = sptail_e3a_build(
        nu=int(np.clip(nu, 30, 140)), nt=int(np.clip(int(0.8 * nv),
                                                     24, 110)),
        storeys=S, Rt=p['Rt'])
    tk = _toolkit()
    V = tk._smooth_boundary(V, F, iters=4)
    V = tk._center_fit(V, scale, V)
    return V, F, uv


# ---- periodic (translation-invariant) Enneper ----------------------------

def sptail_penneper_X(u, v):
    """Closed-form immersion on the log chart z = e^(u+iv):
    X1 = u/2 - e^(2u) cos(2v)/4, X2 = -v/2 - e^(2u) sin(2v)/4,
    X3 = e^u cos v.  One turn v -> v + 2 pi is the exact translation
    (0, -pi, 0)."""
    e2 = np.exp(2.0 * u)
    return np.stack(np.broadcast_arrays(
        0.5 * u - 0.25 * e2 * np.cos(2.0 * v),
        -0.5 * v - 0.25 * e2 * np.sin(2.0 * v),
        np.exp(u) * np.cos(v)), axis=-1)


def sptail_penneper_build(nu=64, nt=48, storeys=2, rmin=0.05,
                          rmax=1.3):
    """`storeys` full turns of the universal cover -- the mesh is one
    continuous grid (each turn joins the next exactly)."""
    S = max(1, int(storeys))
    u = np.linspace(math.log(rmin), math.log(rmax), nu)
    ntt = max(24, nt)                      # columns per turn (exact)
    nv = S * ntt + 1
    v = np.linspace(-S * math.pi, S * math.pi, nv)
    U, V_ = np.meshgrid(u, v, indexing='ij')
    X = sptail_penneper_X(U, V_)
    V0 = X.reshape(-1, 3)
    q0 = sptail_grid_quads(nu, nv)
    uv = _sptail_grid_uv(nu, nv)
    return V0, q0, uv, {'T': np.array([0.0, -math.pi, 0.0]),
                        'ncol_per_turn': int(round(
                            TAU / (v[1] - v[0])))}


def sptail_penneper_mesh(spec, nu, nv, order, radius, scale, theta=0.0,
                         storeys=1):
    p = spec['p_from'](order, radius)
    S = int(np.clip(storeys, 1, 5))
    V, F, uv, _ = sptail_penneper_build(
        nu=int(np.clip(nu, 24, 160)),
        nt=int(np.clip(int(0.9 * nv), 24, 100)),
        storeys=S, rmax=p['rmax'])
    tk = _toolkit()
    V = tk._center_fit(V, scale, V)
    return V, F, uv


# --------------------------------------------------------------------------
# Extension plumbing (no Blender UI of its own; the toolkit owns it)
# --------------------------------------------------------------------------

ADD_MENU = True


def register():
    pass


def unregister():
    pass


if __name__ == "__main__":
    # engine self-tests (numpy only)
    ok = True
    # period integral: winding integral of 1/z
    pi1 = period_integral(lambda z: 1.0 / z, 0.0, 1.0, 0.5)
    e = abs(pi1 - TAU * 1j)
    print(f"period_integral 1/z: {pi1:.6f} err={e:.2e} "
          f"{'OK' if e < 1e-10 else 'FAIL'}")
    ok &= e < 1e-10
    # solve_scalar: root of cos on [1, 2]
    r = solve_scalar(math.cos, 1.0, 2.0)
    e = abs(r - math.pi / 2)
    print(f"solve_scalar cos: {r:.12f} err={e:.2e} "
          f"{'OK' if e < 1e-10 else 'FAIL'}")
    ok &= e < 1e-10
    # disk engine vs the closed-form Enneper antiderivative
    spec = {'g': lambda z, p: z, 'dh': lambda z, p: z,
            'domain': ('disk', 0.0, 1.2), 'offset_rays': False,
            'clip': False}
    x, y, z, wu, wv, _ = we_surface(spec, 220, 48, 1, 1.2)
    u = np.linspace(1e-3, 1.2, 220)
    v = np.arange(48) * (TAU / 48)
    R, TH = np.meshgrid(u, v, indexing='ij')
    zz = R * np.exp(1j * TH)
    xe = np.real(0.5 * (zz - zz ** 3 / 3.0))
    ye = np.real(0.5j * (zz + zz ** 3 / 3.0))
    ze = np.real(0.5 * zz ** 2)
    err = max(np.max(np.abs(x - xe)), np.max(np.abs(y - ye)),
              np.max(np.abs(z - ze)))
    print(f"disk engine vs Enneper closed form: err={err:.2e} "
          f"{'OK' if err < 2e-3 else 'FAIL'}")
    ok &= err < 2e-3
    # rect engine on the same data (path independence check)
    specr = {'g': lambda z, p: z, 'dh': lambda z, p: z,
             'domain': ('rect', 0.1, 1.0, 0.1, 0.9)}
    x, y, z, _, _, _ = we_surface(specr, 200, 200, 1, 1.0)
    u = np.linspace(0.1, 1.0, 200)
    v = np.linspace(0.1, 0.9, 200)
    U, V = np.meshgrid(u, v, indexing='ij')
    zz = U + 1j * V
    F0 = 0.1 + 0.1j
    xe = np.real(0.5 * (zz - zz ** 3 / 3.0) - 0.5 * (F0 - F0 ** 3 / 3.0))
    err = np.max(np.abs(x - xe))
    print(f"rect engine vs closed form: err={err:.2e} "
          f"{'OK' if err < 2e-3 else 'FAIL'}")
    ok &= err < 2e-3
    # Bjorling: cycloid seed must reproduce the Catalan surface
    bj = {'curve': lambda w, p: (w - np.sin(w), 1.0 - np.cos(w),
                                 0.0 * w),
          'normal': lambda w, p: (np.cos(w / 2), -np.sin(w / 2),
                                  0.0 * w),
          't_range': (-math.pi, 3 * math.pi), 'v_half': 1.2}
    x, y, z, _, _, _ = bjorling_surface(bj, 160, 161, 1, 1.2)
    u = np.linspace(-math.pi, 3 * math.pi, 160)
    v = np.linspace(-1.2, 1.2, 161)
    U, V = np.meshgrid(u, v, indexing='ij')
    xe = U - np.sin(U) * np.cosh(V)
    ye = 1 - np.cos(U) * np.cosh(V)
    ze = 4 * np.sin(U / 2) * np.sinh(V / 2)
    err = max(np.max(np.abs(x - xe)), np.max(np.abs(y - ye)),
              np.max(np.abs(z - ze)))
    print(f"Bjorling cycloid == Catalan: err={err:.2e} "
          f"{'OK' if err < 2e-3 else 'FAIL'}")
    ok &= err < 2e-3
    # Bjorling seed row: v = 0 must reproduce the curve exactly
    jm = 161 // 2
    err = max(np.max(np.abs(x[:, jm] - (u - np.sin(u)))),
              np.max(np.abs(y[:, jm] - (1 - np.cos(u)))),
              np.max(np.abs(z[:, jm])))
    print(f"Bjorling seed row: err={err:.2e} "
          f"{'OK' if err < 1e-9 else 'FAIL'}")
    ok &= err < 1e-9

    # ---- exact P / Gyroid / D associate family (Bonnet angle) --------------
    # (1) the three coordinate 1-forms are a NULL (conformal minimal) triple
    zt = np.array([0.3 + 0.7j, -0.5 + 1.2j, 2.0 + 0.4j, -2.5 + 0.9j,
                   0.1 + 3.0j, 1.4 + 2.2j])
    o1, o2, o3 = _pgd_om(zt)
    nullerr = float(np.max(np.abs(o1 ** 2 + o2 ** 2 + o3 ** 2)))
    print(f"P/G/D null identity |om1^2+om2^2+om3^2|={nullerr:.2e} "
          f"{'OK' if nullerr < 1e-12 else 'FAIL'}")
    ok &= nullerr < 1e-12
    # (2) every boundary edge is a proper 180-degree 2-fold axis, all theta,
    #     and its snapped linear part is one of the 24 cube rotations
    for thd in (0.0, 38.0148, 90.0):
        th = math.radians(thd)
        worst_ang, worst_res, worst_snap = 0.0, 0.0, 0.0
        for i in range(6):
            C = _pgd_edge_curve(th, i)
            M, b, res = _pgd_fit_twofold(C)
            ang = math.degrees(math.acos(
                max(-1.0, min(1.0, (np.trace(M) - 1.0) / 2.0))))
            span = float(np.linalg.norm(C.max(0) - C.min(0)))
            worst_ang = max(worst_ang, abs(ang - 180.0))
            worst_res = max(worst_res, res / max(span, 1e-9))
            worst_snap = max(worst_snap,
                             np.abs(_pgd_snap_rot(M) - M).max())
        good = worst_ang < 3.0 and worst_res < 0.02 and worst_snap < 0.35
        ok &= good
        print(f"P/G/D 2-fold edges theta={thd:7.3f}deg: "
              f"max|angle-180|={worst_ang:.2f} max_res={worst_res:.4f} "
              f"snap<{worst_snap:.2f} {'OK' if good else 'FAIL'}")
    # (3) the tiling closes into a cubic lattice, independent of theta
    for thd in (0.0, 38.0148, 90.0):
        gens = pgd_gluings(math.radians(thd))
        lat = pgd_lattice(gens)
        norms = np.sort(np.linalg.norm(lat, axis=1)) if len(lat) else \
            np.array([0.0])
        a = float(norms.min()) if len(lat) else 0.0
        good = len(lat) >= 3 and 2.5 < a < 6.0
        ok &= good
        print(f"P/G/D lattice theta={thd:7.3f}deg: a_min={a:.3f} "
              f"n={len(lat)} {'OK' if good else 'FAIL'}")
    # (4) the build fits the 2 m cube and is finite across the whole morph;
    #     P (0 deg) and D (90 deg) assemble a recognizable *filled cell* --
    #     one connected component, near-manifold (few non-manifold edges
    #     relative to its size) -- while the gyroid / intermediate angles keep
    #     the exact fundamental piece (strictly edge-manifold, no edge used
    #     more than twice).

    def _edge_stats(T):
        ec = {}
        for (x0, x1, x2) in T:
            for a2, b2 in ((x0, x1), (x1, x2), (x2, x0)):
                e = (a2, b2) if a2 < b2 else (b2, a2)
                ec[e] = ec.get(e, 0) + 1
        return sum(1 for c in ec.values() if c > 2)

    def _ncomp(nV, T):
        parent = list(range(nV))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for (x, y, z) in T:
            for u, v in ((x, y), (y, z)):
                ra, rb = find(u), find(v)
                if ra != rb:
                    parent[ra] = rb
        used = set(i for f in T for i in f)
        return len({find(i) for i in used})

    for thd in (0.0, 19.0, 38.0148, 64.0, 90.0):
        V, T = pgd_build(1, 44, 1.0, math.radians(thd))
        finite = bool(np.all(np.isfinite(V)))
        lo, hi = V.min(0), V.max(0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        nonman = _edge_stats(T)
        tiled = thd in (0.0, 90.0)
        if tiled:                                  # filled P / D cell
            comps = _ncomp(len(V), T)
            good = (finite and len(T) > 4000 and cen < 1e-6
                    and abs(ext - 2.0) < 1e-6 and comps == 1
                    and nonman < 0.12 * len(T))
            tag = f"cell comps={comps}"
        else:                                      # exact fundamental piece
            good = (finite and len(T) > 500 and cen < 1e-6
                    and abs(ext - 2.0) < 1e-6 and nonman == 0)
            tag = "piece"
        ok &= good
        print(f"P/G/D {tag} theta={thd:7.3f}deg: {len(V):6d}v {len(T):6d}t "
              f"fit[|c|={cen:.1e} ext={ext:.4f}] nonman={nonman} "
              f"{'OK' if good else 'FAIL'}")
    # (5) the Bonnet morph is continuous (a small angle step gives a bounded,
    #     non-degenerate change) and the three iconic members are distinct
    base = _pgd_grid_pts(0.0)
    d1 = float(np.nanmax(np.abs(_pgd_grid_pts(0.03) - base)))
    d2 = float(np.nanmax(np.abs(_pgd_grid_pts(0.06) - base)))
    cont = (np.isfinite(d1) and np.isfinite(d2)
            and 1e-4 < d1 < 1.0 and d2 < 3.0 * d1 + 1e-6)
    gpat = _pgd_grid_pts(math.radians(38.0148))
    dpat = _pgd_grid_pts(math.radians(90.0))
    distinct = (float(np.nanmax(np.abs(gpat - base))) > 0.05
                and float(np.nanmax(np.abs(dpat - base))) > 0.05)
    good = cont and distinct
    ok &= good
    print(f"P/G/D Bonnet morph: d(.03)={d1:.2e} d(.06)={d2:.2e} "
          f"distinct={distinct} {'OK' if good else 'FAIL'}")

    # ---- higher-genus Chen-Gackstatter: watertight D2d assembly ------------
    # gates: exact chi = 1 - 2g, edge-manifold, ONE boundary loop (the
    # trimmed Enneper end), one component, globally consistent winding
    for gg, nu_t, arcn_t, R_t in ((2, 100, 44, 5.5), (4, 100, 40, 6.0),
                                  (5, 100, 36, 8.0)):
        Vg, Fg, _uv = cg_higher_assemble(gg, nu_t, arcn_t, R_t)
        ecc, dcc = {}, {}
        for f in Fg:
            m = len(f)
            for kk in range(m):
                a2, b2 = f[kk], f[(kk + 1) % m]
                e2 = (a2, b2) if a2 < b2 else (b2, a2)
                ecc[e2] = ecc.get(e2, 0) + 1
                dcc[(a2, b2)] = dcc.get((a2, b2), 0) + 1
        chi = len(Vg) - len(ecc) + len(Fg)
        nonman = sum(1 for c in ecc.values() if c > 2)
        bed = [e2 for e2, c in ecc.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a2, b2 in bed:
            ra, rb = bfind(a2), bfind(b2)
            if ra != rb:
                par[ra] = rb
        loops = len({bfind(a2) for a2, b2 in bed})
        orient = all(c == 1 for c in dcc.values())
        good = (chi == 1 - 2 * gg and nonman == 0 and loops == 1
                and orient and bool(np.all(np.isfinite(Vg))))
        ok &= good
        print(f"CG higher genus {gg}: {len(Vg):6d}v {len(Fg):6d}f "
              f"chi={chi} (want {1 - 2 * gg}) nonman={nonman} "
              f"loops={loops} orient={orient} {'OK' if good else 'FAIL'}")

    # ---- genus-one helicoid: theta engine, harvested closure, topology ----
    # (1) theta_11 identities: odd, quasi-periodic under z+1 and z+tau
    zt = np.array([0.13 + 0.21j, -0.4 + 0.7j, 0.9 - 0.3j, 2.3 + 1.9j])
    tt = _G1H_TAU
    t0 = genus1helicoid_theta11(zt, tt)
    # relative errors (theta grows like e^{pi Im(z)^2 / Im tau}, so the
    # identities are compared against the values' own magnitude)
    e1 = float(np.max(np.abs(genus1helicoid_theta11(zt + 1.0, tt) + t0)
                      / np.abs(t0)))
    e2 = float(np.max(np.abs(
        genus1helicoid_theta11(zt + tt, tt)
        + np.exp(-1j * np.pi * tt - 2j * np.pi * zt) * t0)
        / np.abs(genus1helicoid_theta11(zt + tt, tt))))
    e3 = float(np.max(np.abs(genus1helicoid_theta11(-zt, tt) + t0)
                      / np.abs(t0)))
    good = max(e1, e2, e3) < 1e-10
    ok &= good
    print(f"g1-helicoid theta11: |z+1|={e1:.2e} |z+tau|={e2:.2e} "
          f"odd={e3:.2e} {'OK' if good else 'FAIL'}")
    # (2) the domain map reproduces the notebook's solved constants:
    # tst(a0) = 1 - b0 and the r0 root residual
    ra = abs(complex(_g1h_tst(np.array(_G1H_A0 + 0j))) - (1.0 - _G1H_B))
    v = complex(_g1h_tst(np.array(-_G1H_R0 + 1e-12j)))
    rb = abs((0.5 * (1.0 + tt) * (1.0 + v) - tt).imag)
    good = ra < 1e-6 and rb < 1e-6
    ok &= good
    print(f"g1-helicoid domain: |tst(a0)-(1-b0)|={ra:.2e} "
          f"r0-residual={rb:.2e} {'OK' if good else 'FAIL'}")
    # (3) period closure with the harvested constants: the z -> z+1
    # cycle is the exact vertical translation (0, 0, 2); z -> z+tau
    # closes (both to the FindRoot precision of the constants)
    z0 = 0.311 + 0.077j
    PA = _g1h_path_int(z0, z0 + 1.0, 8001)
    PB = _g1h_path_int(z0, z0 + tt, 8001)
    eh = max(abs(PA[0].real), abs(PA[1].real),
             abs(PB[0].real), abs(PB[1].real))
    ev = abs(PA[2].real - 2.0)
    eb = abs(PB[2].real)
    good = eh < 1e-6 and ev < 1e-6 and eb < 1e-5
    ok &= good
    print(f"g1-helicoid periods: horiz={eh:.2e} |A_v-2|={ev:.2e} "
          f"B_v={eb:.2e} {'OK' if good else 'FAIL'}")
    # (4) the sheet contains the z axis and the horizontal rulings
    xs_t, ys_t, Xs = genus1helicoid_sheet(-2.5, 111, 45)
    max_ax = (xs_t > _G1H_XA + 1e-6) & (xs_t < _G1H_XB - 1e-6)
    axdev = float(np.max(np.abs(Xs[max_ax, 0, :2])))
    rul = Xs[xs_t < _G1H_XA - 1e-6, 0]
    rdev = max(float(np.max(np.abs(rul[:, 0]))),
               float(np.max(np.abs(rul[:, 2] + 1.0))))
    # weld-correspondence residuals (the assembly merges these pairs)
    Rzv = np.array([-1.0, -1.0, 1.0])
    ax_r = float(np.max(np.linalg.norm(
        Xs[max_ax, 0] - Rzv * Xs[max_ax, 0], axis=1)))
    mA = xs_t <= _G1H_XA + 1e-6
    rl_r = float(np.max(np.linalg.norm(
        Xs[mA, 0] - Rzv * Xs[::-1][mA, -1], axis=1)))
    mC = xs_t <= _G1H_XC + 1e-6
    cr_r = float(np.max(np.linalg.norm(
        Xs[mC, -1] - (Rzv * Xs[::-1][mC, 0] + np.array([0, 0, 2.0])),
        axis=1)))
    wres = max(ax_r, rl_r, cr_r)
    good = axdev < 5e-3 and rdev < 5e-3 and wres < 2e-3
    ok &= good
    print(f"g1-helicoid lines: axis-dev={axdev:.2e} "
          f"ruling-dev={rdev:.2e} weld-residual={wres:.2e} "
          f"{'OK' if good else 'FAIL'}")
    # (5) topology: a stack of S cells is one connected, manifold,
    # consistently oriented surface of genus exactly S (one handle per
    # translational period): chi = 2 - 2S - boundary_loops
    for S in (1, 2):
        Vg, Fg, uvg = genus1helicoid_assemble(S, -2.5, 111, 45)
        ecc, dcc = {}, {}
        for f in Fg:
            m = len(f)
            for kk in range(m):
                a2, b2 = f[kk], f[(kk + 1) % m]
                e2k = (a2, b2) if a2 < b2 else (b2, a2)
                ecc[e2k] = ecc.get(e2k, 0) + 1
                dcc[(a2, b2)] = dcc.get((a2, b2), 0) + 1
        chi = len(Vg) - len(ecc) + len(Fg)
        nonman = sum(1 for c in ecc.values() if c > 2)
        orient = all(c == 1 for c in dcc.values())
        bed = [e2k for e2k, c in ecc.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a2, b2 in bed:
            ra2, rb2 = bfind(a2), bfind(b2)
            if ra2 != rb2:
                par[ra2] = rb2
        loops = len({bfind(a2) for a2, b2 in bed})
        parc = list(range(len(Vg)))

        def cfind(a2):
            while parc[a2] != a2:
                parc[a2] = parc[parc[a2]]
                a2 = parc[a2]
            return a2

        for f in Fg:
            for i2 in range(1, len(f)):
                ra2, rb2 = cfind(f[0]), cfind(f[i2])
                if ra2 != rb2:
                    parc[ra2] = rb2
        ncomp = len({cfind(i2) for f in Fg for i2 in f})
        genus = (2 - chi - loops) / 2.0
        uv_ok = bool(np.all(np.isfinite(uvg))) and len(uvg) == len(Vg)
        good = (genus == S and nonman == 0 and orient and ncomp == 1
                and uv_ok and bool(np.all(np.isfinite(Vg))))
        ok &= good
        print(f"g1-helicoid S={S}: {len(Vg):6d}v {len(Fg):6d}f chi={chi} "
              f"loops={loops} genus={genus:.0f} (want {S}) "
              f"nonman={nonman} orient={orient} ncomp={ncomp} "
              f"{'OK' if good else 'FAIL'}")

    # ---- symmetrized Chen-Gackstatter towers (k-fold D_kd assembly) --------
    # (a) period data: the gn rho quadrature must reproduce the closed
    #     Gamma form; g2n / g3k per-segment rho estimates must agree
    #     (that consistency IS the harvested-period validation)
    for kk in (2, 4, 6):
        _SYMMCG_RHO_CACHE.pop(('gn', kk), None)
        a0t, ft, rho_cf = _symmcg_data('gn', kk)
        est0, _ests = symmcg_rho_solve('gn', kk)
        e2 = abs(est0 - rho_cf)
        good = e2 < 1e-6
        ok &= good
        print(f"symmcg gn rho k={kk}: quad={est0:.9f} "
              f"closed={rho_cf:.9f} err={e2:.1e} "
              f"{'OK' if good else 'FAIL'}")
    for tower, kk in (('g2n', 3), ('g2n', 4), ('g3k', 2), ('g3k', 3)):
        a0t, ft, rho_t = _symmcg_data(tower, kk)
        est0, ests = symmcg_rho_solve(tower, kk)
        spread = max(ests) - min(ests)
        good = spread < 5e-5 and (rho_t is None or abs(est0 - rho_t) < 1e-4)
        ok &= good
        tgt = f" harvested={rho_t:.6f}" if rho_t is not None else ""
        print(f"symmcg {tower} rho k={kk}: per-seg "
              f"{['%.6f' % x for x in ests]}{tgt} spread={spread:.1e} "
              f"{'OK' if good else 'FAIL'}")
    # (b) watertight assembly: exact chi = 1 - 2g (one trimmed end),
    #     edge-manifold, ONE boundary loop, oriented, and small pre-snap
    #     period residuals at the branch images
    for tower, kk, gen in (('gn', 2, 1), ('gn', 3, 2), ('gn', 4, 3),
                           ('gn', 6, 5), ('g2n', 3, 4), ('g3k', 2, 3)):
        Vg, Fg, _uv, diag = symmcg_assemble(tower, kk, 90, 24)
        ecc, dcc = {}, {}
        for f in Fg:
            m = len(f)
            for t in range(m):
                a2, b2 = f[t], f[(t + 1) % m]
                e3 = (a2, b2) if a2 < b2 else (b2, a2)
                ecc[e3] = ecc.get(e3, 0) + 1
                dcc[(a2, b2)] = dcc.get((a2, b2), 0) + 1
        chi = len(Vg) - len(ecc) + len(Fg)
        nonman = sum(1 for c in ecc.values() if c > 2)
        bed = [e3 for e3, c in ecc.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a2, b2 in bed:
            ra, rb = bfind(a2), bfind(b2)
            if ra != rb:
                par[ra] = rb
        loops = len({bfind(a2) for a2, b2 in bed})
        orient = all(c == 1 for c in dcc.values())
        pres = max(v for kk2, v in diag['res'].items()
                   if kk2.startswith('branch'))
        good = (chi == 1 - 2 * gen and nonman == 0 and loops == 1
                and orient and pres < 5e-4
                and bool(np.all(np.isfinite(Vg))))
        ok &= good
        print(f"symmcg {tower} k={kk} (genus {gen}): {len(Vg):6d}v "
              f"{len(Fg):6d}f chi={chi} (want {1 - 2 * gen}) "
              f"nonman={nonman} loops={loops} orient={orient} "
              f"period_res={pres:.1e} {'OK' if good else 'FAIL'}")
    # ---- singly periodic Callahan-Hoffman-Meeks (k = 2) --------------------
    # (1) the pulled-back 1-forms are a null triple (exact minimality)
    rng = np.random.default_rng(7)
    wt = (rng.uniform(-6.0, _CHMP_UMAX, 300)
          + 1j * rng.uniform(1e-3, math.pi - 1e-3, 300))
    o1, o2, o3 = chm_periodic_forms(wt)
    nullerr = float(np.max(np.abs(o1 ** 2 + o2 ** 2 + o3 ** 2)))
    print(f"CHM periodic null |sum om^2|={nullerr:.2e} "
          f"{'OK' if nullerr < 1e-12 else 'FAIL'}")
    ok &= nullerr < 1e-12
    # (2) patch boundary structure BEFORE snapping: the t = 0 / t = pi
    # rows are planar geodesics (constant y / x), the arc column is a
    # horizontal planar geodesic, the pre-xa t = pi row is a straight
    # (1,1,0) line at constant height, and the half-spacing d matches
    # the real-axis elliptic integral (= 2d).  Residual tolerances sit
    # at the harvested constants' own closure precision (~1e-3).
    ug = _chmp_ugrid(90, -5.0)
    tg = _chmp_tgrid(64)
    Xp = chm_periodic_patch(ug, tg)
    iLt = int(np.searchsorted(ug, _CHMP_XA))
    ry = float(np.ptp(Xp[:, 0, 1]))
    rx = float(np.ptp(Xp[iLt:, -1, 0]))
    rz = float(np.ptp(Xp[-1, :, 2]))
    rl = float(np.ptp(Xp[:iLt + 1, -1, 0] - Xp[:iLt + 1, -1, 1]))
    rlz = float(np.ptp(Xp[:iLt + 1, -1, 2]))
    dmeas = float(np.median(Xp[-1, :, 2]) - np.median(Xp[:iLt + 1, -1, 2]))
    th_e = np.linspace(0.0, 0.5 * math.pi, 4001)
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    ell = _trapz(1.0 / np.sqrt(_CHMP_A ** 2 - np.sin(th_e) ** 2), th_e)
    good = (ry < 1e-3 and rx < 1e-3 and rz < 1e-6 and rl < 1e-3
            and rlz < 1e-3 and abs(2.0 * dmeas - ell) < 1e-3)
    ok &= good
    print(f"CHM periodic patch: dy={ry:.1e} dx={rx:.1e} dz_arc={rz:.1e} "
          f"line[d(x-y)={rl:.1e} dz={rlz:.1e}] d={dmeas:.6f} "
          f"(elliptic/2={0.5 * ell:.6f}) {'OK' if good else 'FAIL'}")
    # (3) assembled periods: edge-manifold, oriented, one component,
    # chi = -6 S (the quotient by the 4d translation is genus 3 = 2k+1
    # with two planar ends -- measured here, and matching the CHM theorem
    # for k = 1), 2 S end rims + 2 outer horizontal cut loops
    for S in (1, 2):
        Xs, iLs, dS = chm_periodic_snap(Xp, ug)
        Vg, Fg, uvg = chm_periodic_assemble(ug, tg, Xs, iLs, dS, S)
        ecc, dcc = {}, {}
        for f in Fg:
            m = len(f)
            for kk in range(m):
                a2, b2 = f[kk], f[(kk + 1) % m]
                e2 = (a2, b2) if a2 < b2 else (b2, a2)
                ecc[e2] = ecc.get(e2, 0) + 1
                dcc[(a2, b2)] = dcc.get((a2, b2), 0) + 1
        chi = len(Vg) - len(ecc) + len(Fg)
        nonman = sum(1 for c in ecc.values() if c > 2)
        bed = [e2 for e2, c in ecc.items() if c == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a2, b2 in bed:
            ra, rb = bfind(a2), bfind(b2)
            if ra != rb:
                par[ra] = rb
        loops = len({bfind(a2) for a2, b2 in bed})
        orient = all(c == 1 for c in dcc.values())
        good = (chi == -6 * S and nonman == 0 and loops == 2 * S + 2
                and orient and bool(np.all(np.isfinite(Vg)))
                and bool(np.all(np.isfinite(uvg))))
        ok &= good
        print(f"CHM periodic S={S}: {len(Vg):6d}v {len(Fg):6d}f "
              f"chi={chi} (want {-6 * S}) nonman={nonman} "
              f"loops={loops} (want {2 * S + 2}) orient={orient} "
              f"{'OK' if good else 'FAIL'}")
    # ---- Catenoid-Enneper / Costa-Wohlgemuth / Wohlgemuth (cwce block) -----
    # (1) the stored period constants close the period problem: every
    #     notebook condition vanishes under independent quadrature
    for gg in (2, 3, 4):
        r = float(np.max(np.abs(cwce_ce_residual(gg))))
        good = r < 1e-6
        ok &= good
        print(f"catenoid-enneper g={gg} period residual: {r:.1e} "
              f"{'OK' if good else 'FAIL'}")
    for fam, kk in (('cw', 2), ('cw', 3), ('cw', 4), ('cw', 5),
                    ('cw', 7), ('w2', 2)):
        r = float(np.max(np.abs(cwce_cw_residual(fam, kk))))
        good = r < 1e-8
        ok &= good
        print(f"{'costa-wohlgemuth' if fam == 'cw' else 'wohlgemuth-2'} "
              f"k={kk} period residual: {r:.1e} "
              f"{'OK' if good else 'FAIL'}")
    # (2) null identity: phi1 phi2 = dh^2 makes sum omega_i^2 vanish
    #     identically -- checked on the actual meshing integrands
    rt = np.linspace(0.31, 4.7, 40)
    omn = cwce_ce_omega(3, rt, 0.71, np.full(40, -1), np.zeros(40),
                        list(_CWCE_CE[3][1]))
    nerr = float(np.max(np.abs(np.sum(omn ** 2, axis=-1))))
    good = nerr < 1e-12
    ok &= good
    print(f"catenoid-enneper null |sum om^2|={nerr:.1e} "
          f"{'OK' if good else 'FAIL'}")
    datn = _cwce_cw_data('w2', 2)
    omn = cwce_cw_omega_row('w2', 2, datn, np.linspace(-0.9, 2.9, 40),
                            1.1, np.full(40, -1), np.zeros(40),
                            ['x0', 'xinf', 'xc', 'x1'])
    nerr = float(np.max(np.abs(np.sum(omn ** 2, axis=-1))))
    good = nerr < 1e-12
    ok &= good
    print(f"wohlgemuth null |sum om^2|={nerr:.1e} "
          f"{'OK' if good else 'FAIL'}")
    # (3) watertight assemblies: exact chi = 2 - 2g - (open end rims),
    #     edge-manifold, oriented, one component, small pre-snap period
    #     residuals (the snap only removes quadrature noise)
    for fam, kk, gen, nrim in (('ce', 2, 2, 2), ('ce', 3, 3, 2),
                               ('ce', 4, 4, 2), ('cw', 2, 2, 4),
                               ('cw', 3, 4, 4), ('cw', 5, 8, 4),
                               ('w2', 2, 3, 4)):
        if fam == 'ce':
            Vg, Fg, _uv, diag = cwce_ce_assemble(kk, 100, 29)
        else:
            Vg, Fg, _uv, diag = cwce_cw_assemble(fam, kk, 110, 21)
        ecc, dcc = {}, {}
        for f in Fg:
            m = len(f)
            for tq in range(m):
                a2, b2 = f[tq], f[(tq + 1) % m]
                e2 = (a2, b2) if a2 < b2 else (b2, a2)
                ecc[e2] = ecc.get(e2, 0) + 1
                dcc[(a2, b2)] = dcc.get((a2, b2), 0) + 1
        chi = len(Vg) - len(ecc) + len(Fg)
        nonman = sum(1 for cq in ecc.values() if cq > 2)
        bed = [e2 for e2, cq in ecc.items() if cq == 1]
        par = {}

        def bfind(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for a2, b2 in bed:
            ra, rb = bfind(a2), bfind(b2)
            if ra != rb:
                par[ra] = rb
        loops = len({bfind(a2) for a2, b2 in bed})
        orient = all(cq == 1 for cq in dcc.values())
        parc = list(range(len(Vg)))

        def cfind(a2):
            while parc[a2] != a2:
                parc[a2] = parc[parc[a2]]
                a2 = parc[a2]
            return a2

        for f in Fg:
            for i2 in range(1, len(f)):
                ra, rb = cfind(f[0]), cfind(f[i2])
                if ra != rb:
                    parc[ra] = rb
        ncomp = len({cfind(f[0]) for f in Fg})
        pres = max(diag['res'].values())
        want = 2 - 2 * gen - nrim
        good = (chi == want and nonman == 0 and loops == nrim
                and orient and ncomp == 1 and pres < 5e-3
                and bool(np.all(np.isfinite(Vg))))
        ok &= good
        nm = {'ce': 'catenoid-enneper', 'cw': 'costa-wohlgemuth',
              'w2': 'wohlgemuth-2'}[fam]
        print(f"{nm} {'g' if fam == 'ce' else 'k'}={kk} "
              f"(genus {gen}): {len(Vg):6d}v {len(Fg):6d}f chi={chi} "
              f"(want {want}) nonman={nonman} loops={loops} "
              f"(want {nrim}) orient={orient} ncomp={ncomp} "
              f"period_res={pres:.1e} {'OK' if good else 'FAIL'}")
    # ---- doubly periodic hyperelliptic tiler (KMR + Wei) -------------------
    # (1) Wei's period problem: the harvested FindRoot constants close the
    #     notebook's own residual Im int_a^b (G + 1/G) dz/z, and the
    #     residual moves off-solution (the gate is not vacuous)
    for bb, aa in ((0.3, 0.23640853975826828), (0.5, 0.07082564803712697),
                   (0.7, 0.0030808434547059727)):
        rres = abs(dperiodic_wei_residual(bb, aa))
        roff = abs(dperiodic_wei_residual(bb, 0.8 * aa))
        good = rres < 1e-5 and roff > 50 * max(rres, 1e-9)
        ok &= good
        print(f"dperiodic Wei period b={bb}: |res|={rres:.2e} "
              f"off-solution={roff:.2e} {'OK' if good else 'FAIL'}")
    # (2) per member: wall/period closure residuals small; a genuinely 2-D
    #     lattice; quotient topology chi = 2 - 2g - 4 with exactly 4
    #     Scherk end rims, edge-manifold, oriented, one component; and a
    #     2 x 2 lattice tiling stays ONE connected manifold block
    for key, p, gen in (('kmr2', {'a': 0.4, 'rmin': 0.05}, 1),
                        ('kmr2', {'a': 0.55, 'rmin': 0.05}, 1),
                        ('wei', {'b': 0.3, 'a': 0.23640853975826828,
                                 'rmin': 0.05}, 2),
                        ('wei', {'b': 0.5, 'a': 0.07082564803712697,
                                 'rmin': 0.05}, 2),
                        ('kmr3', {'xmin': 0.05}, 1)):
        P = dperiodic_patch(key, p, 48, 48)
        wres = max(v for kk2, v in P['res'].items() if kk2 != 'x3=u')
        lat = float(np.linalg.norm(np.cross(P['T1'], P['T2'])))
        chi, loops, nonman, orient, ncomp = dperiodic_quotient(P)
        V2, Q2, UV2 = dperiodic_assemble(P, (2, 2))
        par2 = list(range(len(V2)))

        def find2(a2):
            while par2[a2] != a2:
                par2[a2] = par2[par2[a2]]
                a2 = par2[a2]
            return a2

        ec2 = {}
        for f in Q2:
            m2 = len(f)
            for t2 in range(m2):
                a2, b2 = f[t2], f[(t2 + 1) % m2]
                e2 = (a2, b2) if a2 < b2 else (b2, a2)
                ec2[e2] = ec2.get(e2, 0) + 1
                ra2, rb2 = find2(a2), find2(b2)
                if ra2 != rb2:
                    par2[ra2] = rb2
        ncomp2 = len({find2(i2) for f in Q2 for i2 in f})
        nonman2 = sum(1 for c in ec2.values() if c > 2)
        chi_want = 2 - 2 * gen - 4
        good = (wres < 5e-3 and lat > 1.0
                and chi == chi_want and loops == 4 and nonman == 0
                and orient and ncomp == 1 and ncomp2 == 1 and nonman2 == 0
                and bool(np.all(np.isfinite(V2)))
                and bool(np.all(np.isfinite(UV2))))
        ok &= good
        print(f"dperiodic {key:5s} (genus {gen}): wall_res={wres:.1e} "
              f"|T1xT2|={lat:.2f} chi={chi} (want {chi_want}) ends={loops} "
              f"nonman={nonman} orient={orient} 2x2[comp={ncomp2} "
              f"nonman={nonman2}] {'OK' if good else 'FAIL'}")
    # (3) KMR-3 external cross-check: the measured lattice reproduces the
    #     notebook's printed translation constants (disx, disy, disz)
    P3 = dperiodic_patch('kmr3', {'xmin': 0.01}, 60, 48)
    e1 = abs(abs(P3['T1'][0]) - 2.0 * 2.6417540147391194)
    e2 = abs(abs(P3['T2'][1]) - 2.0 * 1.2779360827691162)
    e3 = abs(abs(P3['T2'][2]) - 2.0 * 2.4000944407384024)
    good = max(e1, e2, e3) < 5e-3
    ok &= good
    print(f"dperiodic KMR-3 lattice vs notebook: |dT|=({e1:.1e},"
          f"{e2:.1e},{e3:.1e}) {'OK' if good else 'FAIL'}")
    # ---- doubly periodic long tail (KS / RTW / Wei tower / Connor) --------
    # (1) RTW M1+ period solve: the re-solved b closes the notebook's own
    #     SolvePeriod condition, and the residual moves off-solution
    rres = abs(dptail_rtwmp_residual(DPTAIL_RTWMP_B))
    roff = abs(dptail_rtwmp_residual(0.9 * DPTAIL_RTWMP_B))
    good = rres < 1e-9 and roff > 1e3 * max(rres, 1e-12)
    ok &= good
    print(f"dptail RTW M1+ period b={DPTAIL_RTWMP_B:.6f}: |res|={rres:.2e} "
          f"off-solution={roff:.2e} {'OK' if good else 'FAIL'}")
    # (2) every shipped member: wall/period residuals small, x3 = u exact,
    #     a genuinely 2-D lattice, quotient topology chi = 2 - 2g - 4 with
    #     exactly 4 Scherk end rims, edge-manifold, oriented, connected;
    #     and a 2 x 2 lattice tiling stays ONE connected manifold block
    dpt_ship = ('ksg2', 'ksg3', 'ksg3x',
                'rtwmp', 'rtwm1pm_0', 'rtwm1pm_1',
                'wei13_0', 'wei13_1', 'wei13_2', 'wei13_3',
                'wei14_0', 'wei14_1', 'wei23', 'wei16',
                'conn_asym', 'conn78', 'conn80', 'conn82',
                'conn84', 'conn85')
    for key in dpt_ship:
        mem = dptail_member(key)
        gen = mem['genus']
        P = dptail_patch(key, {}, 56, 40)
        wres = max(v for k2, v in P['res'].items() if k2 != 'x3=u')
        lat = float(np.linalg.norm(np.cross(P['T1'], P['T2'])))
        chi, loops, nonman, orient, ncomp = dptail_quotient(P)
        V2, Q2, UV2 = dptail_assemble(P, (2, 2))
        par2 = list(range(len(V2)))

        def find2(a2):
            while par2[a2] != a2:
                par2[a2] = par2[par2[a2]]
                a2 = par2[a2]
            return a2

        ec2 = {}
        for f in Q2:
            m2 = len(f)
            for t2 in range(m2):
                a2, b2 = f[t2], f[(t2 + 1) % m2]
                e2 = (a2, b2) if a2 < b2 else (b2, a2)
                ec2[e2] = ec2.get(e2, 0) + 1
                ra2, rb2 = find2(a2), find2(b2)
                if ra2 != rb2:
                    par2[ra2] = rb2
        ncomp2 = len({find2(i2) for f in Q2 for i2 in f})
        nonman2 = sum(1 for c in ec2.values() if c > 2)
        chi_want = 2 - 2 * gen - 4
        good = (wres < 5e-3 and P['res']['x3=u'] < 1e-9 and lat > 1.0
                and chi == chi_want and loops == 4 and nonman == 0
                and orient and ncomp == 1 and ncomp2 == 1
                and nonman2 == 0 and bool(np.all(np.isfinite(V2)))
                and bool(np.all(np.isfinite(UV2))))
        ok &= good
        print(f"dptail {key:10s} ({mem['style']:5s} genus {gen}): "
              f"wall_res={wres:.1e} |T1xT2|={lat:.2f} chi={chi} "
              f"(want {chi_want}) ends={loops} nonman={nonman} "
              f"orient={orient} 2x2[comp={ncomp2} nonman={nonman2}] "
              f"{'OK' if good else 'FAIL'}")
    print("dptail deferred (see BACKLOG.md): ksg1 ksg4 dp_catenoid "
          "lubeck_batista chm_g3")
    # ---- singly periodic long tail (sptail block) --------------------------
    # (1) null identity phi1 phi2 = dh^2 <=> sum om_i^2 = 0, on the actual
    #     meshing integrands of every family
    rng2 = np.random.default_rng(11)
    zt2 = rng2.uniform(0.05, 3.0, 200) + 1j * rng2.uniform(1e-3, 3.0, 200)
    for nm2, omf in (('six-scherk', sptail_six_om(30.0)[2]),
                     ('alt-fence', sptail_fencealt_om(2.0)),
                     ('fence-cat', sptail_fencecat_om(0.2)),
                     ('helicoidal-KS', sptail_hks_om(4, 0.1, 0.8))):
        o1, o2, o3 = omf(zt2)
        nerr = float(np.max(np.abs(o1 * o1 + o2 * o2 + o3 * o3)))
        good = nerr < 1e-12
        ok &= good
        print(f"sptail null {nm2}: |sum om^2|={nerr:.1e} "
              f"{'OK' if good else 'FAIL'}")
    # (2) six-ended Scherk: boundary symmetry residuals pre-snap, the
    #     translation vs the om2 residue closed form, and measured
    #     chi(S+1) - chi(S) = -4 = 2 - 2*0 - 6 (genus 0, 6 ends/period)
    for phid in (30.0, 60.0):
        V1, F1, _u, dg = sptail_six_build(phid, storeys=1)
        pres = max(dg['line_dy'], dg['line_dz'], dg['mx_dx'],
                   dg['my_dy'], dg['pi_dy'])
        V2, F2, _u2, _d2 = sptail_six_build(phid, storeys=2)
        c1 = sptail_topology(V1, F1)
        c2 = sptail_topology(V2, F2)
        good = (pres < 1e-3 and dg['dy_vs_residue'] < 1e-6
                and dg['pi_vs_2dy'] < 1e-6
                and c2[0] - c1[0] == -4
                and c1[1] == 0 and c2[1] == 0 and c1[2] and c2[2]
                and c1[4] == 1 and c2[4] == 1
                and bool(np.all(np.isfinite(V2))))
        ok &= good
        print(f"sptail six-scherk phi={phid:.0f}: presnap={pres:.1e} "
              f"dy_res={dg['dy_vs_residue']:.1e} chi {c1[0]}->{c2[0]} "
              f"(dchi -4) nonman={c2[1]} orient={c2[2]} "
              f"{'OK' if good else 'FAIL'}")
    # (3) alternating fence: rho = 1/sqrt(a) closes the horizontal period
    #     (the two vertical mirrors coincide); quotient chi = -2, 2 ends
    #     (genus 1); stacks stay manifold/oriented/connected
    V1, F1, _u, dg = sptail_fencealt_build(2.0, storeys=1)
    V2, F2, _u2, _d2 = sptail_fencealt_build(2.0, storeys=2)
    c1 = sptail_topology(V1, F1)
    c2 = sptail_topology(V2, F2)
    Vq, Fq = sptail_quotient(V1, F1, dg['T'], dg['span'])
    cq = sptail_topology(Vq, Fq)
    good = (dg['period_x'] < 1e-9 and c2[0] - c1[0] == -2
            and cq[0] == -2 and cq[3] == 2 and cq[1] == 0 and cq[2]
            and c2[1] == 0 and c2[2] and c2[4] == 1)
    ok &= good
    print(f"sptail alt-fence a=2: period_x={dg['period_x']:.1e} "
          f"chi {c1[0]}->{c2[0]} (dchi -2) quotient chi={cq[0]} "
          f"ends={cq[3]} (want -2, 2) {'OK' if good else 'FAIL'}")
    # (4) fence of catenoids: |G| = 1 identically on the arc (the
    #     Lopez-Ros closure rho = sqrt(a)); quotient chi = -2 with 2
    #     ends (genus 1 per period, measured)
    aa = 0.2
    tha = np.linspace(0.01, math.pi - 0.01, 400)
    za = math.sqrt(aa) * np.exp(1j * tha)
    Ga = math.sqrt(aa) * np.sqrt(za - 1.0) / (np.sqrt(za)
                                              * np.sqrt(za - aa))
    gerr = float(np.max(np.abs(np.abs(Ga) - 1.0)))
    V1, F1, _u, dg = sptail_fencecat_build(aa, storeys=1)
    V2, F2, _u2, _d2 = sptail_fencecat_build(aa, storeys=2)
    c1 = sptail_topology(V1, F1)
    c2 = sptail_topology(V2, F2)
    Vq, Fq = sptail_quotient(V1, F1, dg['T'], dg['span'])
    cq = sptail_topology(Vq, Fq)
    pres = max(dg['y_ptp'], dg['xA_ptp'], dg['xB_ptp'],
               dg['arc_z_ptp'])
    good = (gerr < 1e-12 and pres < 5e-3 and c2[0] - c1[0] == -2
            and cq[0] == -2 and cq[3] == 2 and cq[1] == 0 and cq[2]
            and c2[1] == 0 and c2[2] and c2[4] == 1)
    ok &= good
    print(f"sptail fence-cat a={aa}: max||G|-1|={gerr:.1e} "
          f"presnap={pres:.1e} chi {c1[0]}->{c2[0]} quotient "
          f"chi={cq[0]} ends={cq[3]} (want -2, 2) "
          f"{'OK' if good else 'FAIL'}")
    # (5) helicoidal Karcher-Scherk: the geometric period solve closes
    #     the k-fold axis (and is not vacuous -- off-solution R misses),
    #     the measured screw rise reproduces pi R^2/(1+R^4), and
    #     chi(S+1) - chi(S) = 2 - 2k (genus 0, 2k ends per period)
    for kk, av in ((2, 0.2), (4, 0.1)):
        Rs = sptail_hks_solveR(kk, av)
        res_on = float(np.linalg.norm(
            sptail_hks_axis_residual(kk, av, Rs)))
        res_off = float(np.linalg.norm(
            sptail_hks_axis_residual(kk, av, 0.97 * Rs)))
        V1, F1, _u, dg = sptail_hks_build(kk, av, storeys=1, R=Rs)
        V2, F2, _u2, _d2 = sptail_hks_build(kk, av, storeys=2, R=Rs)
        c1 = sptail_topology(V1, F1)
        c2 = sptail_topology(V2, F2)
        good = (res_on < 1e-6 and res_off > 100 * max(res_on, 1e-9)
                and dg['trans_vs_theory'] < 1e-8
                and c2[0] - c1[0] == 2 - 2 * kk
                and c2[1] == 0 and c2[2] and c2[4] == 1
                and bool(np.all(np.isfinite(V2))))
        ok &= good
        print(f"sptail helicoidal-KS k={kk} a={av}: R={Rs:.8f} "
              f"axis_res={res_on:.1e} (off {res_off:.1e}) "
              f"rise_vs_theory={dg['trans_vs_theory']:.1e} "
              f"chi {c1[0]}->{c2[0]} (dchi {2 - 2 * kk}) "
              f"{'OK' if good else 'FAIL'}")
    # (6) translation-invariant Enneper, 3 annular ends: the closed-form
    #     boundary constants (mirrors at y = 0 / -Y1, line at -Y1/2),
    #     quotient chi = -2 with 4 end rims (MEASURED genus 0 -- see the
    #     block note on the harvest's genus-1 annotation)
    V1, F1, _u, dg = sptail_e3a_build(storeys=1)
    V2, F2, _u2, _d2 = sptail_e3a_build(storeys=2)
    c1 = sptail_topology(V1, F1)
    c2 = sptail_topology(V2, F2)
    Vq, Fq = sptail_quotient(V1, F1, dg['T'], dg['span'])
    cq = sptail_topology(Vq, Fq)
    pres = max(dg['y01_ptp'], dg['y1R_vs_const'],
               dg['line_y_vs_const'], dg['line_z_ptp'])
    good = (pres < 1e-9 and c2[0] - c1[0] == -2
            and cq[0] == -2 and cq[3] == 4 and cq[1] == 0 and cq[2]
            and c2[1] == 0 and c2[2] and c2[4] == 1)
    ok &= good
    print(f"sptail enneper-3ann: const_res={pres:.1e} chi "
          f"{c1[0]}->{c2[0]} (dchi -2) quotient chi={cq[0]} "
          f"ends={cq[3]} (want -2, 4) {'OK' if good else 'FAIL'}")
    # (7) periodic Enneper: one turn of the built grid is the EXACT
    #     translation (0, -pi, 0); the wrapped quotient is a cylinder
    #     (chi = 0, 2 rims -- genus 0, 2 ends)
    V2t, q2g, _u2, dg2 = sptail_penneper_build(storeys=2)
    nturn = dg2['ncol_per_turn']
    nvv = len(V2t) // 64
    G1 = V2t.reshape(64, nvv, 3)
    dT = G1[:, nturn:, :] - G1[:, :-nturn, :]
    terr = float(np.max(np.abs(dT - dg2['T'][None, None, :])))
    V1, q1g, _u, dg = sptail_penneper_build(storeys=1)
    span1 = float(np.linalg.norm(V1.max(0) - V1.min(0)))
    Vq, Fq = sptail_quotient(V1, q1g, dg['T'], span1, kmax=1)
    cq = sptail_topology(Vq, Fq)
    good = (terr < 1e-12 and cq[0] == 0 and cq[3] == 2
            and cq[1] == 0 and cq[2] and cq[4] == 1)
    ok &= good
    print(f"sptail periodic-enneper: |T - (0,-pi,0)|={terr:.1e} "
          f"quotient chi={cq[0]} ends={cq[3]} (want 0, 2) "
          f"{'OK' if good else 'FAIL'}")

    print("\nRESULT:", "ALL OK" if ok else "FAILURES in we_builders")
