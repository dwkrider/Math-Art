
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
        # bespoke finished-mesh builder (higher-genus Chen-Gackstatter):
        # the spec supplies the whole mesher, the engine only wires it in
        def build(nu, nv, order, radius, scale, theta=0.0, storeys=1):
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
    print("\nRESULT:", "ALL OK" if ok else "FAILURES in we_builders")
