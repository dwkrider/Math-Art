
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
        punct = spec.get('mask_punctures')
        if punct:
            for zc, rho in _ev(punct, p):
                mask &= np.abs(z - zc) > rho
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
    a vertical translation by half the vertical monodromy, T = pi/(2n)
    (the period pi/n picked up by dh = phi3 around one end, halved because
    the disk sees only a half-turn around each logarithmic end), composed
    with a rotation by pi/n about the vertical axis.  The rotation carries
    the 2n vertical wing-walls onto themselves, so each storey's wing tops
    meet the next storey's wing bottoms and the walls read as continuous
    vertical planes.  The shared wing rims are welded (union-find over the
    screw-matched boundary vertices) so the joins carry no seam."""
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

    T = math.pi / (2 * n)
    Rm = _rot_z(math.pi / n)

    if S > 1 and M:
        # seam: storey-s top rim coincides with storey-(s+1) bottom rim,
        # i.e. V0[i] ~= screw(V0[j]),  screw(P) = P @ Rm.T + T*zhat.
        bnd, med = _open_boundary(V0, quads0)
        Vb = V0[bnd]
        scr = Vb @ Rm.T + np.array([0.0, 0.0, T])
        d = np.linalg.norm(Vb[:, None, :] - scr[None, :, :], axis=2)
        jmin = d.argmin(axis=1)
        dmin = d[np.arange(len(bnd)), jmin]
        tol = 0.5 * med
        pairs = [(int(bnd[a]), int(bnd[jmin[a]]))
                 for a in range(len(bnd)) if dmin[a] < tol]
        parent = list(range(S * M))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for s in range(S - 1):
            for (it, jb) in pairs:
                ra, rb2 = find(s * M + it), find((s + 1) * M + jb)
                if ra != rb2:
                    parent[ra] = rb2
        Vparts, Rp = [], np.eye(3)
        for s in range(S):
            Vparts.append(V0 @ Rp.T + np.array([0.0, 0.0, s * T]))
            Rp = Rp @ Rm.T                        # (Rm^{s+1}).T
        Vcat = np.concatenate(Vparts, axis=0)
        UVcat = np.concatenate([UV0] * S, axis=0)
        roots = np.array([find(a) for a in range(S * M)])
        uniq = np.unique(roots)
        rmap = np.full(S * M, -1, dtype=np.int64)
        rmap[uniq] = np.arange(len(uniq))
        ri = rmap[roots]
        Vf = np.zeros((len(uniq), 3))
        UVf = np.zeros((len(uniq), 2))
        cnt = np.zeros(len(uniq))
        np.add.at(Vf, ri, Vcat)
        np.add.at(UVf, ri, UVcat)
        np.add.at(cnt, ri, 1)
        Vf /= cnt[:, None]
        UVf /= cnt[:, None]
        quads = []
        for s in range(S):
            for q in quads0:
                fq = tuple(int(ri[s * M + i]) for i in q)
                if len(set(fq)) >= 3:
                    quads.append(fq)
    else:
        Vf, quads, UVf = V0, quads0, UV0

    # the wing/end rims are cut from the puncture mask, so they start as a
    # one-quad staircase; the radial-graded sampling (radial_grade='rim')
    # already shrinks each step near the ends, and a longer boundary
    # relaxation than the default finishes them into smooth wing edges.
    Vf = tk._smooth_boundary(Vf, quads, iters=20)
    Vf = tk._center_fit(Vf, scale, Vf)
    return Vf, quads, UVf


def make_entry(key, spec):
    """PARAMETRIC/MESH_PARAM builder closure for a WE spec."""
    if spec.get('tower'):
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
    print("\nRESULT:", "ALL OK" if ok else "FAILURES in we_builders")
