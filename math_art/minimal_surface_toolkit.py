
# Minimal Surface Toolkit for Blender
#
# Three generators in one add-on:
#
# 1. Classic parametric minimal surfaces (Enneper & higher orders,
#    catenoid, helicoid, Henneberg, Catalan, Bour, Richmond, Scherk's
#    doubly-periodic graph) -- after Juergen Meier's gallery
#    (3d-meier.de, tut25).
#
# 2. Triply-periodic minimal surfaces via their standard nodal
#    (level-set) approximations, meshed by marching tetrahedra:
#    Schwarz P & D, Schoen's Gyroid / I-WP / F-RD, Neovius, Lidinoid,
#    Split P, plus the singly-periodic Scherk tower. Inventory after
#    Ken Brakke's periodic-surface pages (kenbrakke.com/evolver).
#
# 3. A Plateau-problem solver -- a lightweight, in-Blender take on what
#    Brakke's Surface Evolver does: pin one or two boundary curves and
#    minimize surface area (Pinkall-Polthier cotangent-Laplacian
#    iteration solved with conjugate gradients). Includes the classic
#    "minimal surface between a circle and a torus knot" construction
#    (trefoil by default).
#
# Geometry only; materials and rendering are left to Blender.

bl_info = {
    "name": "Minimal Surface Toolkit",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Minimal Surfaces / N-panel 'Minimal Surfaces'",
    "description": "Parametric & triply-periodic minimal surfaces, and a "
                   "Plateau solver spanning minimal surfaces on curves",
    "category": "Add Mesh",
}

import math
import numpy as np

TAU = 2.0 * math.pi


# ==========================================================================
# Weierstrass elliptic-function engine (Jacobi-theta series, numpy only)
# ==========================================================================
# Provides Weierstrass P, P' and zeta for a lattice given by half-periods,
# via the Jacobi theta functions (DLMF 23.6 for the elliptic functions,
# DLMF 20.2 for the theta q-series). The nome q = exp(i*pi*tau) is small
# for the lattices we use, so ~a dozen terms of each series reach 1e-15.
#
# Costa and Chen-Gackstatter both live on the square (lemniscatic) torus:
#   periods 1, i ; half-periods w1 = 1/2, w3 = i/2 ; tau = i ; q = e^-pi ;
#   g2 = Gamma(1/4)^8 / (16 pi^2) = 189.0727... , g3 = 0 ,
#   e1 = P(1/2) = 6.87519... , and (this lattice) g2 = 4 e1^2.

_THETA_TERMS = 16   # q^((n+.5)^2) underflows long before this for our q


def _theta1_series(xi, q):
    """theta1(xi) and its first three xi-derivatives (t0..t3), where xi is
    a complex ndarray. DLMF 20.2.1 differentiated term by term."""
    n = np.arange(_THETA_TERMS)
    a = ((-1.0) ** n) * q ** ((n + 0.5) ** 2)          # (terms,)
    k = (2 * n + 1).astype(float)
    ang = np.multiply.outer(np.asarray(xi, dtype=complex), k)
    s, c = np.sin(ang), np.cos(ang)
    t0 = 2.0 * np.sum(a * s, axis=-1)
    t1 = 2.0 * np.sum(a * k * c, axis=-1)
    t2 = -2.0 * np.sum(a * k ** 2 * s, axis=-1)
    t3 = -2.0 * np.sum(a * k ** 3 * c, axis=-1)
    return t0, t1, t2, t3


class _Lattice:
    """Weierstrass P, P', zeta on the lattice with real half-period w1 and
    ratio tau = w3/w1 (Im tau > 0). All methods are vectorized over z."""

    def __init__(self, w1, tau):
        self.w1 = float(w1)
        self.q = np.exp(1j * math.pi * tau)
        self.c = math.pi / (2.0 * self.w1)             # dxi/dz
        # quasi-period eta1 = zeta(w1)  (DLMF 23.6.8), from theta1 at 0
        n = np.arange(_THETA_TERMS)
        a = ((-1.0) ** n) * self.q ** ((n + 0.5) ** 2)
        k = (2 * n + 1).astype(float)
        t1_0 = 2.0 * np.sum(a * k)                      # theta1'(0)
        t3_0 = -2.0 * np.sum(a * k ** 3)               # theta1'''(0)
        self.eta1 = -(math.pi ** 2 / (12.0 * self.w1)) * (t3_0 / t1_0)

    def zeta(self, z):
        z = np.asarray(z, dtype=complex)
        t0, t1, _, _ = _theta1_series(self.c * z, self.q)
        return (self.eta1 / self.w1) * z + self.c * (t1 / t0)

    def wp(self, z):
        z = np.asarray(z, dtype=complex)
        t0, t1, t2, _ = _theta1_series(self.c * z, self.q)
        r1 = t1 / t0
        return -self.eta1 / self.w1 - self.c ** 2 * (t2 / t0 - r1 ** 2)

    def wp_prime(self, z):
        z = np.asarray(z, dtype=complex)
        t0, t1, t2, t3 = _theta1_series(self.c * z, self.q)
        r1 = t1 / t0
        return -self.c ** 3 * (t3 / t0 - 3.0 * r1 * (t2 / t0) + 2.0 * r1 ** 3)


# The square torus shared by Costa and Chen-Gackstatter.
_SQUARE = _Lattice(0.5, 1j)


# ==========================================================================
# 1. Parametric classic surfaces
# ==========================================================================
# Each entry: (label, builder(params) -> (V (n,3) ndarray, quads list,
# wrap_u, wrap_v)); u along axis 0, v along axis 1 of the grid.

def _grid(nu, nv, u0, u1, v0, v1):
    u = np.linspace(u0, u1, nu)
    v = np.linspace(v0, v1, nv)
    return np.meshgrid(u, v, indexing='ij')


def _enneper(nu, nv, order, radius, theta=0.0):
    U, V = _grid(nu, nv, 1e-4, radius, 0.0, TAU)
    z = U * np.exp(1j * V)
    n = order
    x = np.real(z - z ** (2 * n + 1) / (2 * n + 1))
    y = -np.imag(z + z ** (2 * n + 1) / (2 * n + 1))
    w = (2.0 / (n + 1)) * np.real(z ** (n + 1))
    return x, y, w, False, True


def _catenoid(nu, nv, order, radius, theta=0.0):
    c = 1.0
    U, V = _grid(nu, nv, 0.0, TAU, -radius, radius)
    x = c * np.cosh(V / c) * np.cos(U)
    y = c * np.cosh(V / c) * np.sin(U)
    return x, y, V, True, False


def _helicoid(nu, nv, order, radius, theta=0.0):
    turns = max(1, order)
    U, V = _grid(nu, nv, -turns * math.pi, turns * math.pi, -radius, radius)
    return V * np.cos(U), V * np.sin(U), 0.6 * U, False, False


def _henneberg(nu, nv, order, radius, theta=0.0):
    U, V = _grid(nu, nv, 1e-3, 0.4 * radius, 0.0, TAU)
    x = 2 * np.sinh(U) * np.cos(V) - (2.0 / 3.0) * np.sinh(3 * U) * np.cos(3 * V)
    y = 2 * np.sinh(U) * np.sin(V) + (2.0 / 3.0) * np.sinh(3 * U) * np.sin(3 * V)
    w = 2 * np.cosh(2 * U) * np.cos(2 * V)
    return x, y, w, False, True


def _catalan(nu, nv, order, radius, theta=0.0):
    U, V = _grid(nu, nv, -math.pi, 3 * math.pi, -radius, radius)
    x = U - np.sin(U) * np.cosh(V)
    y = 1 - np.cos(U) * np.cosh(V)
    w = 4 * np.sin(U / 2) * np.sinh(V / 2)
    return x, y, w, False, False


def _bour(nu, nv, order, radius, theta=0.0):
    U, V = _grid(nu, nv, 1e-3, radius, 0.0, 2 * TAU)   # double cover closes it
    x = U * np.cos(V) - 0.5 * U ** 2 * np.cos(2 * V)
    y = -U * np.sin(V) - 0.5 * U ** 2 * np.sin(2 * V)
    w = (4.0 / 3.0) * U ** 1.5 * np.cos(1.5 * V)
    return x, y, w, False, True


def _richmond(nu, nv, order, radius, theta=0.0):
    U, V = _grid(nu, nv, 0.25, radius + 0.25, 0.0, TAU)
    z = U * np.exp(1j * V)
    x = np.real(-1.0 / (2 * z) - z ** 3 / 6.0)
    y = np.imag(-1.0 / (2 * z) + z ** 3 / 6.0)
    w = np.real(z)
    return x, y, w, False, True


def _scherk_graph(nu, nv, order, radius, theta=0.0):
    lim = 0.47 * math.pi
    U, V = _grid(nu, nv, -lim, lim, -lim, lim)
    w = np.log(np.cos(U) / np.cos(V))
    return U, V, w, False, False


# --- Weierstrass-based surfaces (Costa, Chen-Gackstatter) ------------------
# Costa and Chen-Gackstatter live on a torus that closes up (eta1 = pi/2),
# so they are meshed periodically in both directions with small disks
# removed around the ends; the 6th return value is then a boolean validity
# mask and the only mesh boundaries are the clean circular end rims. The
# k-noid instead stops at a modest domain radius (a smooth parameter-curve
# boundary), so it needs no clipping.

def _torus_grid(nu, nv):
    """Periodic (nu, nv) sample of the unit torus [0,1)^2 (endpoint-free)."""
    u = np.linspace(0.0, 1.0, nu, endpoint=False)
    v = np.linspace(0.0, 1.0, nv, endpoint=False)
    return np.meshgrid(u, v, indexing='ij')


def _puncture_mask(U, V, centers):
    """Valid where the toroidal distance to every puncture exceeds its
    radius (so all lattice translates of each end are excluded at once).
    `centers` is a list of (cu, cv, rho). Ends whose parametrization
    stretches fastest (planar, Enneper) want a larger rho so the rim sits
    where grid cells are still small -> a cleaner circular rim."""
    valid = np.ones(U.shape, dtype=bool)
    for cu, cv, rho in centers:
        du = np.abs(((U - cu + 0.5) % 1.0) - 0.5)
        dv = np.abs(((V - cv + 0.5) % 1.0) - 0.5)
        valid &= (du * du + dv * dv) > rho * rho
    return valid


def _costa(nu, nv, order, radius, theta=0.0):
    """Costa's minimal surface -- genus 1, three ends, on the square torus.
    Gray/Nylander closed form (constant offsets dropped; re-centered by the
    mesher). Meshed periodically with the planar end (0,0) and the two
    catenoid ends (1/2,0), (0,1/2) removed. `order`/`theta` unused;
    `radius` scales the end-rim disk size (smaller -> ends reach further)."""
    L = _SQUARE
    e1 = L.wp(0.5).real
    U, V = _torus_grid(nu, nv)
    z = U + 1j * V
    ze, z1, z3 = L.zeta(z), L.zeta(z - 0.5), L.zeta(z - 0.5j)
    P = L.wp(z)
    a = math.pi / (2.0 * e1)
    x = 0.5 * np.real(-ze + math.pi * U + a * (z1 - z3))
    y = 0.5 * np.real(-1j * ze + math.pi * V - a * (1j * z1 - 1j * z3))
    zc = (math.sqrt(2.0 * math.pi) / 4.0) * np.log(
        np.abs((P - e1) / (P + e1)))
    s = max(radius / 1.2, 0.4)
    mask = _puncture_mask(U, V, [(0.0, 0.0, 0.20 / s),      # planar end
                                 (0.5, 0.0, 0.11 / s),      # catenoid end
                                 (0.0, 0.5, 0.11 / s)])     # catenoid end
    return x, y, zc, True, True, mask


def _chen_gackstatter(nu, nv, order, radius, theta=0.0):
    """Chen-Gackstatter -- genus 1 with a single Enneper (order-3) end,
    total curvature -8 pi, on the square torus. Meshed periodically with a
    disk removed around the lone end at w = 0. `order`/`theta` unused;
    `radius` scales the end-rim disk size."""
    L = _SQUARE
    g2 = 4.0 * L.wp(0.5).real ** 2
    U, V = _torus_grid(nu, nv)
    w = U + 1j * V
    ze, P, Pp = L.zeta(w), L.wp(w), L.wp_prime(w)
    x = np.real(math.pi * w - ze - (math.pi / g2) * Pp)
    y = np.imag(math.pi * w + ze - (math.pi / g2) * Pp)
    zc = math.sqrt(6.0 * math.pi / g2) * np.real(P)
    mask = _puncture_mask(U, V, [(0.0, 0.0, 0.26 / max(radius / 1.2, 0.4))])
    return x, y, zc, True, True, mask


def _knoid(nu, nv, order, radius, theta=0.0):
    """Jorge-Meeks k-noid: genus 0 with `order` catenoid ends (n >= 3),
    the n-fold-symmetric generalization of the catenoid. WE data
    g = z^(n-1), dh = z^(n-1)/(z^n-1)^2 dz. The n ends sit at the n-th
    roots of unity on |z| = 1; the surface runs from the bottom point
    z = 0 across the equator to the top point z = infinity, so the domain
    spans BOTH sides of |z| = 1. Built by radial WE integration from z = 0
    (a pole-free base point), with disks around the ends removed.
    `radius` scales the top cap's reach past the equator; `theta` unused."""
    n = int(max(3, min(order, 12)))
    r_out = 1.35 + 0.5 * min(max(radius / 1.2, 0.0), 1.6)   # spans past 1
    u = np.linspace(1e-3, r_out, nu)
    # offset theta by half a step so no radial ray lands exactly on an end
    # (a root of unity), which would otherwise integrate through a pole
    dth = TAU / nv
    v = np.linspace(0.5 * dth, TAU + 0.5 * dth, nv, endpoint=False)
    R, TH = np.meshgrid(u, v, indexing='ij')
    z = R * np.exp(1j * TH)
    den = (z ** n - 1.0) ** 2
    zm = z ** (n - 1)
    f1 = 0.5 * (1.0 - z ** (2 * (n - 1))) / den        # phi1 integrand / dz
    f2 = 0.5j * (1.0 + z ** (2 * (n - 1))) / den
    f3 = zm / den
    ez = np.exp(1j * TH)                               # dz = ez dr (radial)
    Xr = np.stack([np.real(f * ez) for f in (f1, f2, f3)], axis=-1)
    # Radial rays from z ~ 0 (all columns share that base point, so no
    # angular term is needed). A loose cap only kills true numerical
    # garbage from a ray grazing a pole; it is high enough not to flatten
    # the (tall, near-vertical) catenoid ends. The ends are trimmed by the
    # object-space radius clip in build_parametric.
    cap = 400.0 * float(np.median(np.abs(Xr)))
    Xr = np.clip(Xr, -cap, cap)
    dr = np.diff(R, axis=0)[..., None]
    XYZ = np.concatenate(
        [np.zeros((1, nv, 3)),
         np.cumsum(0.5 * (Xr[1:] + Xr[:-1]) * dr, axis=0)], axis=0)
    return XYZ[..., 0], XYZ[..., 1], XYZ[..., 2], False, True, True


def _cathel(nu, nv, order, radius, theta=0.0):
    """Catenoid<->helicoid associate (Bonnet) family. theta = 0 is the
    catenoid, theta = pi/2 the helicoid; every intermediate value is a
    complete minimal surface isometric to both. `order`/`radius` set the
    vertical (u) extent."""
    h = 0.9 + 0.4 * max(radius, 0.2)
    U, V = _grid(nu, nv, -h, h, 0.0, TAU)
    ct, st = math.cos(theta), math.sin(theta)
    cu, su = np.cosh(U), np.sinh(U)
    x = ct * cu * np.cos(V) + st * su * np.sin(V)
    y = ct * cu * np.sin(V) - st * su * np.cos(V)
    z = ct * U + st * V
    return x, y, z, False, True


# --- Costa-Hoffman-Meeks (genus k configurable) ---------------------------
# The genus-k generalization of Costa: genus k, three ends (two catenoid at
# z = +-1, one planar at z = infinity), dihedral symmetry of order 4(k+1),
# on the hyperelliptic curve w^(k+1) = z^k (z^2 - 1). Built by numeric
# Weierstrass-Enneper integration of ONE fundamental patch (the image of the
# closed upper half z-plane) which is then rigidly tiled into 2(k+1) copies
# (k+1 rotations about the vertical axis, each with its mirror) and welded --
# the deck transformation (z,w) -> (z, e^(2 pi i/(k+1)) w) is exactly a
# rotation by 2 pi/(k+1). Reduces to the ordinary Costa surface at k = 1.
# (Plan/derivation cross-checked with fable-5; modulus table verified.)

_CHM_C_CACHE = {}


def _chm_w(z, k):
    """Single-valued branch of (z^k (z^2-1))^(1/(k+1)) on the upper half
    plane (cuts along [0,1] and (-inf,-1]); DLMF-style principal logs."""
    return np.exp((k * np.log(z) + np.log(z - 1.0) + np.log(z + 1.0))
                  / (k + 1))


def _chm_modulus(k, n=4096):
    """Real modulus c fixing the end balance, from the period ratio on an
    ellipse around the cut [0,1]: c^2 = -Im(A)/Im(B) with A = oint
    w/(z^2-1) dz, B = oint 1/(w(z^2-1)) dz. (A, B come out imaginary.)"""
    if k in _CHM_C_CACHE:
        return _CHM_C_CACHE[k]
    t = (np.arange(n) + 0.5) * (TAU / n)
    z = 0.5 + 0.75 * np.cos(t) + 0.35j * np.sin(t)
    dz = (-0.75 * np.sin(t) + 0.35j * np.cos(t)) * (TAU / n)
    w = _chm_w(z, k)
    A = np.sum(w / (z * z - 1.0) * dz)
    B = np.sum(1.0 / (w * (z * z - 1.0)) * dz)
    c = math.sqrt(-A.imag / B.imag)
    _CHM_C_CACHE[k] = c
    return c


def _chm_patch(k, c, n_in, n_out, nv, r_out=12.0):
    """Integrate the WE 1-forms over the closed upper half plane -> one
    fundamental patch, as a real (nr, nv, 3) array. Grid: Chebyshev in
    theta (clustered at the ends theta = 0, pi), graded in r (clustered at
    the branch point r=0 and the ends r=1)."""
    nv = nv if nv % 2 else nv + 1                  # odd: a column hits pi/2
    j = np.arange(nv)
    th = (math.pi / 2) * (1 - np.cos(math.pi * j / (nv - 1)))
    s = np.linspace(0.0, 1.0, n_in + 1)[1:]
    r_in = ((1 - np.cos(math.pi * s)) / 2) ** ((k + 1) / 2)
    r_o = np.exp(np.linspace(0.0, math.log(r_out), n_out + 1))[1:]
    r = np.concatenate([[0.0], r_in, r_o])
    nr = len(r)
    R, TH = np.meshgrid(r, th, indexing='ij')
    Z = R * np.exp(1j * TH)
    Z[0, :] = 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        W = _chm_w(Z, k)
        d = Z * Z - 1.0
        F = np.stack([0.5 * (W - c * c / W) / d,
                      0.5j * (W + c * c / W) / d,
                      c / d], axis=-1)             # (nr, nv, 3)
    jm = (nv - 1) // 2                             # spine column theta=pi/2
    X = np.zeros((nr, nv, 3), dtype=complex)
    # analytic first spine cell 0 -> z0 = i r[1] (integrand ~ z^-(k/(k+1)))
    z0 = 1j * r[1]
    I1 = (c * c / 2) * np.exp(-1j * math.pi / (k + 1)) * (k + 1) \
        * z0 ** (1.0 / (k + 1))
    seed = np.array([-I1, -1j * I1, -c * z0])
    # spine: cumulative along r at column jm (dz = i dr)
    dr = np.diff(r)[:, None]
    sp_inc = 0.5 * (F[1:, jm, :] + F[:-1, jm, :]) * (1j * dr)
    spine = np.concatenate([[np.zeros(3)], [seed],
                            seed + np.cumsum(sp_inc[1:], axis=0)], axis=0)
    X[:, jm, :] = spine
    # arcs: cumulative along theta per row (dz = i z dtheta), from jm out
    gfac = F * (1j * Z)[..., None]
    dth = np.diff(th)
    trap = np.zeros((nr, nv, 3), dtype=complex)
    trap[:, 1:, :] = 0.5 * (gfac[:, 1:, :] + gfac[:, :-1, :]) \
        * dth[None, :, None]
    # Zero non-finite increments (they occur only at the masked end nodes
    # theta = 0, pi where z = +-1). Otherwise the single cumsum below lets
    # one bad increment before the spine column poison the whole row.
    trap = np.where(np.isfinite(trap), trap, 0.0)
    C = np.cumsum(trap, axis=1)
    X = X[:, jm, :][:, None, :] + (C - C[:, jm, :][:, None, :])
    return Z, np.real(X)


def _build_chm(nu, nv, order, radius, scale, theta=0.0):
    """Assemble the full genus-`order` Costa-Hoffman-Meeks mesh: integrate
    the patch, snap its symmetry seams, tile into 2(k+1) rigid copies, weld,
    trim the ends, smooth the rims, center and fit a 2 m cube."""
    k = int(max(1, min(order, 6)))
    c = _chm_modulus(k)
    n_in = max(70, int(1.4 * nu))
    n_out = max(30, int(0.6 * nu))
    Z, Xr = _chm_patch(k, c, n_in, n_out, nv)
    nr, nvp, _ = Xr.shape
    jm = (nvp - 1) // 2
    Rabs = np.abs(Z)
    ang = -math.pi / (k + 1)
    uvec = np.array([math.cos(ang), math.sin(ang)])
    # Step 4: snap the two boundary columns onto their symmetry planes
    for jcol, is0 in ((0, True), (nvp - 1, False)):
        xy = Xr[:, jcol, :2]
        bank = (Rabs[:, jcol] < 1.0) if is0 else (Rabs[:, jcol] > 1.0)
        proj = (xy @ uvec)[:, None] * uvec[None, :]         # onto bank line
        flat = np.stack([xy[:, 0], np.zeros_like(xy[:, 1])], axis=-1)  # y=0
        Xr[:, jcol, :2] = np.where(bank[:, None], proj, flat)
    Xr[0, :, :] = 0.0                                       # center vertex
    rho = 0.09 / max(radius / 1.2, 0.4)
    valid = (np.abs(Z - 1.0) > rho) & (np.abs(Z + 1.0) > rho)
    valid[0, :] = True
    Xr = np.where(np.isfinite(Xr), Xr, 0.0)
    Xr[~valid] = 0.0                                        # kill near-pole
    V0 = Xr.reshape(-1, 3)
    vv = valid.reshape(-1)
    ii, jj = np.meshgrid(np.arange(nr - 1), np.arange(nvp - 1), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    q0 = np.stack([ii * nvp + jj, ii * nvp + jj + 1,
                   (ii + 1) * nvp + jj + 1, (ii + 1) * nvp + jj], axis=1)
    q0 = q0[np.all(vv[q0], axis=1)]
    # tile: k+1 rotations, each with its y-mirror -> 2(k+1) copies
    M = np.diag([1.0, -1.0, 1.0])
    Vparts, Fparts, base = [], [], 0
    for jrot in range(k + 1):
        a = TAU * jrot / (k + 1)
        Rj = np.array([[math.cos(a), -math.sin(a), 0.0],
                       [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        for mir in (False, True):
            T = Rj @ (M if mir else np.eye(3))
            Vparts.append(V0 @ T.T)
            qf = (q0[:, ::-1] if mir else q0) + base       # flip mirror wind
            Fparts.append(qf)
            base += len(V0)
    V = np.concatenate(Vparts, axis=0)
    faces = np.concatenate(Fparts, axis=0)
    # weld (quantize + unique). Tolerance is tight: the seam snap already
    # makes shared vertices coincide to machine epsilon, so a loose weld
    # only risks fusing distinct sheets near the dense central saddle
    # (which silently adds handles / breaks the topology).
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    keyq = np.round(V / (1e-7 * max(diag, 1.0))).astype(np.int64)
    _, inv = np.unique(keyq, axis=0, return_inverse=True)
    inv = inv.ravel()
    Vw = np.zeros((int(inv.max()) + 1, 3))
    Vw[inv] = V
    faces = inv[faces]
    # Collapse welded seam faces: a quad whose two adjacent corners merged
    # becomes a TRIANGLE (dropping it instead would tear the seam and
    # disconnect the surface); genuine degenerates are discarded.
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
    used = np.unique(np.array([i for f in flist for i in f], dtype=np.int64))
    remap = np.full(len(Vw), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    Vf = Vw[used]
    quads = [tuple(int(remap[i]) for i in f) for f in flist]
    # the radius clip can shear off small islands; keep the main body only
    Vf, quads = _largest_component(Vf, quads)
    Vf = _smooth_boundary(Vf, quads)
    Vf = _center_fit(Vf, scale, Vf)
    return Vf, quads


def _largest_component(V, quads):
    """Keep only the face-connected component with the most faces."""
    parent = list(range(len(V)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for f in quads:
        for i in range(1, len(f)):
            union(f[0], f[i])
    from collections import Counter
    sizes = Counter(find(f[0]) for f in quads)
    if len(sizes) <= 1:
        return V, quads
    keep_root = sizes.most_common(1)[0][0]
    quads = [f for f in quads if find(f[0]) == keep_root]
    used = np.unique(np.array([i for f in quads for i in f], dtype=np.int64))
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], [tuple(int(remap[i]) for i in f) for f in quads]


PARAMETRIC = {
    'ENNEPER': ("Enneper", _enneper),
    'CATENOID': ("Catenoid", _catenoid),
    'HELICOID': ("Helicoid", _helicoid),
    'HENNEBERG': ("Henneberg", _henneberg),
    'CATALAN': ("Catalan", _catalan),
    'BOUR': ("Bour", _bour),
    'RICHMOND': ("Richmond", _richmond),
    'SCHERK1': ("Scherk (doubly periodic)", _scherk_graph),
    'COSTA': ("Costa (genus 1)", _costa),
    'COSTA_HM': ("Costa-Hoffman-Meeks (genus k)", _build_chm),
    'CHEN_GACK': ("Chen-Gackstatter", _chen_gackstatter),
    'KNOID': ("Jorge-Meeks k-noid", _knoid),
    'CATHEL': ("Catenoid-Helicoid (associate)", _cathel),
}

# surfaces built as a finished (V, quads) mesh rather than a parameter grid
MESH_PARAM = {'COSTA_HM': _build_chm}
# surfaces whose `order` selects a discrete count rather than an Enneper
# order / helicoid turns (drives the operator UI label)
COUNT_PARAM = {'KNOID': "Ends (n)", 'COSTA_HM': "Genus (k)"}
# surfaces that use the associate-family angle
ANGLE_PARAM = {'CATHEL'}


def _raw_grid(kind, nu, nv, order, radius, theta):
    """Raw (unnormalized) surface grid. Returns (G (nu,nv,3), wrap_u,
    wrap_v, clip) where clip requests end-face trimming by the mesher."""
    out = PARAMETRIC[kind][1](nu, nv, order, radius, theta)
    x, y, w, wrap_u, wrap_v = out[:5]
    clip = out[5] if len(out) > 5 else False
    return np.stack([x, y, w], axis=-1), wrap_u, wrap_v, clip


def _center_fit(pts, scale, ref=None):
    """Center on the bounding-box midpoint and scale so the largest extent
    is 2.0 units (a 2 m cube), then apply `scale`. `ref` (a subset) fixes
    the box, so runaway ends don't shrink the body."""
    pts = np.asarray(pts, dtype=float)
    ref = pts if ref is None else np.asarray(ref, dtype=float)
    if len(ref) == 0:
        return pts
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    cen = 0.5 * (lo + hi)
    ext = float(np.max(hi - lo))
    s = (2.0 / ext if ext > 1e-9 else 1.0) * scale
    return (pts - cen) * s


def _inliers(pts):
    """Points within the 90th distance percentile of the median -- the
    body of a surface with ends running to infinity."""
    c = np.median(pts, axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    keep = d <= np.percentile(d, 90.0)
    return pts[keep] if keep.any() else pts


def _smooth_boundary(V, quads, iters=10, lam=0.5):
    """Relax open mesh-boundary loops in place: each boundary vertex is
    averaged toward its two boundary neighbours. Removes the grid
    staircase left on an end-rim cut from an axis-aligned grid, without
    disturbing interior vertices."""
    if not quads:
        return V
    from collections import defaultdict
    count = defaultdict(int)
    for q in quads:
        for k in range(len(q)):
            a, b = q[k], q[(k + 1) % len(q)]
            count[(a, b) if a < b else (b, a)] += 1
    nbr = defaultdict(list)
    bnd = set()
    for (a, b), c in count.items():
        if c == 1:                       # boundary edge
            nbr[a].append(b)
            nbr[b].append(a)
            bnd.add(a)
            bnd.add(b)
    # keep only vertices with exactly two boundary neighbours (clean loops)
    loop = [v for v in bnd if len(nbr[v]) == 2]
    if not loop:
        return V
    V = V.copy()
    idx = np.array(loop)
    n0 = np.array([nbr[v][0] for v in loop])
    n1 = np.array([nbr[v][1] for v in loop])
    for _ in range(iters):
        target = 0.5 * (V[n0] + V[n1])
        V[idx] += lam * (target - V[idx])
    return V


def build_parametric_grid(kind, nu, nv, order, radius, scale, theta=0.0):
    """(nu, nv, 3) point grid plus wrap flags, centered and fit to a 2 m
    cube. (Used for NURBS output; end clipping is a mesh-only operation.)"""
    if kind in MESH_PARAM:
        raise ValueError(f"{kind} has no NURBS/grid form; use mesh output")
    G, wrap_u, wrap_v, clip = _raw_grid(kind, nu, nv, order, radius, theta)
    flat = G.reshape(-1, 3)
    if isinstance(clip, np.ndarray):
        ref = flat[clip.reshape(-1)]
    elif clip:
        ref = _inliers(flat)
    else:
        ref = flat
    G = _center_fit(flat, scale, ref).reshape(nu, nv, 3)
    return G, wrap_u, wrap_v


def build_parametric(kind, nu, nv, order, radius, scale, theta=0.0):
    if kind in MESH_PARAM:
        return MESH_PARAM[kind](nu, nv, order, radius, scale, theta)
    G, wrap_u, wrap_v, clip = _raw_grid(kind, nu, nv, order, radius, theta)
    V = G.reshape(-1, 3)
    valid = clip.reshape(-1) if isinstance(clip, np.ndarray) else None

    def vid(i, j):
        return i * nv + j

    quads = []
    for i in range(nu if wrap_u else nu - 1):
        i2 = (i + 1) % nu
        for j in range(nv if wrap_v else nv - 1):
            j2 = (j + 1) % nv
            f = (vid(i, j), vid(i2, j), vid(i2, j2), vid(i, j2))
            if valid is None or (valid[f[0]] and valid[f[1]]
                                 and valid[f[2]] and valid[f[3]]):
                quads.append(f)

    if valid is not None:
        # parameter-space clip: boundaries are the clean end-rim circles
        ref = None
    elif clip and quads:
        # object-space radius clip for ends that run to infinity: drop a
        # face if any corner lies past the radius cutoff (clean rounded
        # rims), plus any face bridging across an excluded end.
        c = np.median(V, axis=0)
        rad = np.linalg.norm(V - c, axis=1)
        keepv = rad <= float(np.percentile(rad, 90.0))
        q = np.array(quads)
        q = q[np.all(keepv[q], axis=1)]
        P = V[q]
        maxlen = np.max(np.linalg.norm(P - P[:, [1, 2, 3, 0], :], axis=2),
                        axis=1)
        quads = [tuple(int(i) for i in t)
                 for t in q[maxlen <= 4.0 * float(np.median(maxlen))]]
        ref = None
    else:
        ref = _inliers(V) if clip else V

    if ref is None:
        # compact to referenced vertices only (drops loose end points)
        used = (np.unique(np.array(quads).ravel()) if quads
                else np.array([], dtype=int))
        remap = np.full(len(V), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        V = V[used]
        quads = [tuple(int(remap[i]) for i in qd) for qd in quads]
        V = _smooth_boundary(V, quads)   # smooth staircase/clip end rims
        ref = V

    V = _center_fit(V, scale, ref)
    return V, quads


# ==========================================================================
# 2. Triply-periodic minimal surfaces (nodal approximations)
# ==========================================================================

def _f_p(x, y, z):
    return np.cos(x) + np.cos(y) + np.cos(z)

def _f_d(x, y, z):
    return (np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(y) * np.sin(z))

def _f_g(x, y, z):
    return (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z)
            + np.sin(z) * np.cos(x))

def _f_neovius(x, y, z):
    return (3 * (np.cos(x) + np.cos(y) + np.cos(z))
            + 4 * np.cos(x) * np.cos(y) * np.cos(z))

def _f_iwp(x, y, z):
    return (2 * (np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z)
                 + np.cos(z) * np.cos(x))
            - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)))

def _f_frd(x, y, z):
    return (4 * np.cos(x) * np.cos(y) * np.cos(z)
            - (np.cos(2 * x) * np.cos(2 * y) + np.cos(2 * y) * np.cos(2 * z)
               + np.cos(2 * z) * np.cos(2 * x)))

def _f_lidinoid(x, y, z):
    return (0.5 * (np.sin(2 * x) * np.cos(y) * np.sin(z)
                   + np.sin(2 * y) * np.cos(z) * np.sin(x)
                   + np.sin(2 * z) * np.cos(x) * np.sin(y)
                   - np.cos(2 * x) * np.cos(2 * y)
                   - np.cos(2 * y) * np.cos(2 * z)
                   - np.cos(2 * z) * np.cos(2 * x)) + 0.15)

def _f_splitp(x, y, z):
    return (1.1 * (np.sin(2 * x) * np.sin(z) * np.cos(y)
                   + np.sin(2 * y) * np.sin(x) * np.cos(z)
                   + np.sin(2 * z) * np.sin(y) * np.cos(x))
            - 0.2 * (np.cos(2 * x) * np.cos(2 * y)
                     + np.cos(2 * y) * np.cos(2 * z)
                     + np.cos(2 * z) * np.cos(2 * x))
            - 0.4 * (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)))

def _f_scherk_tower(x, y, z):
    return np.sin(z) - np.sinh(x) * np.sinh(y)

TPMS = {
    'P': ("Schwarz P", _f_p, True),
    'D': ("Schwarz D", _f_d, True),
    'G': ("Gyroid", _f_g, True),
    'NEOVIUS': ("Neovius", _f_neovius, True),
    'IWP': ("Schoen I-WP", _f_iwp, True),
    'FRD': ("Schoen F-RD", _f_frd, True),
    'LIDINOID': ("Lidinoid", _f_lidinoid, True),
    'SPLITP': ("Split P", _f_splitp, True),
    'SCHERKT': ("Scherk Tower (singly periodic)", _f_scherk_tower, False),
}

# cube corners (i,j,k offsets) and a 6-tetrahedra decomposition sharing
# the 0-6 diagonal
_CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_TETS = [(0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6),
         (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6)]
# sign-pattern cases for one tetrahedron: bit set = corner value < 0
_ONE = {1: (0, (1, 2, 3)), 2: (1, (0, 2, 3)),
        4: (2, (0, 1, 3)), 8: (3, (0, 1, 2)),
        14: (0, (1, 2, 3)), 13: (1, (0, 2, 3)),
        11: (2, (0, 1, 3)), 7: (3, (0, 1, 2))}
_TWO = {3: ((0, 1), (2, 3)), 5: ((0, 2), (1, 3)), 9: ((0, 3), (1, 2)),
        6: ((1, 2), (0, 3)), 10: ((1, 3), (0, 2)), 12: ((2, 3), (0, 1))}


def _orientation_flags():
    """Whether each (tet, sign-case) emits triangles wound against
    the field gradient. Calibrated on an exact linear field, where
    the crossing polygon is exactly perpendicular to the gradient --
    so the flags are combinatorial and immune to sliver triangles."""
    flags = {}
    cube = np.array(_CUBE, dtype=float)
    for ti, tet in enumerate(_TETS):
        P = cube[list(tet)]                       # (4,3) corners
        M = P[1:] - P[0]                          # for gradient solve
        for cd in list(_ONE) + list(_TWO):
            f = np.where([cd >> i & 1 for i in range(4)], -1.0, 1.0)
            g = np.linalg.solve(M, f[1:] - f[0])  # exact gradient

            def x(ci, cj):
                t = f[ci] / (f[ci] - f[cj])
                return P[ci] + t * (P[cj] - P[ci])

            if cd in _ONE:
                lone, (o0, o1, o2) = _ONE[cd]
                p0, p1, p2 = x(lone, o0), x(lone, o1), x(lone, o2)
            else:
                (n0, n1), (q0, q1) = _TWO[cd]
                p0, p1, p2 = x(n0, q0), x(n0, q1), x(n1, q1)
            n = np.cross(p1 - p0, p2 - p0)
            flags[(ti, cd)] = float(np.dot(n, g)) < 0.0
    return flags


_ORIENT = None


def marching_tets(field, box_min, box_max, res):
    """Extract the zero level set of `field` on a res[0]xres[1]xres[2]
    sample grid over the box. Returns (verts (n,3), tris (m,3)) with
    triangle winding oriented along the field gradient."""
    nx, ny, nz = (r + 1 for r in res)
    xs = np.linspace(box_min[0], box_max[0], nx)
    ys = np.linspace(box_min[1], box_max[1], ny)
    zs = np.linspace(box_min[2], box_max[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    vals = field(X, Y, Z).ravel()
    # samples landing exactly on the surface (e.g. Schwarz P at the
    # lattice points) produce degenerate crossings; nudge them off
    vals = np.where(np.abs(vals) < 1e-9, 1e-9, vals)
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    ii, jj, kk = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1),
                             np.arange(nz - 1), indexing='ij')
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()

    def flat(i, j, k):
        return (i * ny + j) * nz + k

    corner = [flat(ii + o[0], jj + o[1], kk + o[2]) for o in _CUBE]

    global _ORIENT
    if _ORIENT is None:
        _ORIENT = _orientation_flags()

    tri_pts = []          # list of (3, ntri, 3) blocks
    for ti, (a, b, c, d) in enumerate(_TETS):
        A, B, C, D = corner[a], corner[b], corner[c], corner[d]
        fa, fb, fc, fd = vals[A], vals[B], vals[C], vals[D]
        code = ((fa < 0).astype(np.int8) | ((fb < 0) << 1)
                | ((fc < 0) << 2) | ((fd < 0) << 3))
        tet = np.stack([A, B, C, D], axis=0)

        def interp(sel, ci, cj):
            ia, ib = tet[ci][sel], tet[cj][sel]
            va, vb = vals[ia], vals[ib]
            t = va / (va - vb)
            return pos[ia] + t[:, None] * (pos[ib] - pos[ia])

        for cd, (lone, others) in _ONE.items():
            sel = np.nonzero(code == cd)[0]
            if len(sel) == 0:
                continue
            p0 = interp(sel, lone, others[0])
            p1 = interp(sel, lone, others[1])
            p2 = interp(sel, lone, others[2])
            if _ORIENT[(ti, cd)]:
                p1, p2 = p2, p1
            tri_pts.append(np.stack([p0, p1, p2], axis=1))
        for cd, ((n0, n1), (pp0, pp1)) in _TWO.items():
            sel = np.nonzero(code == cd)[0]
            if len(sel) == 0:
                continue
            q0 = interp(sel, n0, pp0)
            q1 = interp(sel, n0, pp1)
            q2 = interp(sel, n1, pp1)
            q3 = interp(sel, n1, pp0)
            if _ORIENT[(ti, cd)]:
                q1, q3 = q3, q1
            tri_pts.append(np.stack([q0, q1, q2], axis=1))
            tri_pts.append(np.stack([q0, q2, q3], axis=1))

    if not tri_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    tris_xyz = np.concatenate(tri_pts, axis=0)      # (ntri, 3, 3)

    # weld: quantize and unique
    flatv = tris_xyz.reshape(-1, 3)
    eps = max(np.max(box_max) - np.min(box_min), 1.0) * 1e-6
    keys = np.round(flatv / eps).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_index=False,
                          return_inverse=True)
    order = np.zeros(len(uniq), dtype=np.int64)
    order[inv] = np.arange(len(flatv))              # a representative
    verts = flatv[order]
    tris = inv.reshape(-1, 3)
    good = ((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
            & (tris[:, 0] != tris[:, 2]))
    tris = tris[good]

    return verts, tris


def build_tpms(kind, cells, res_per_cell, scale):
    label, field, triply = TPMS[kind]
    if triply:
        span = cells * TAU
        box_min = (-span / 2, -span / 2, -span / 2)
        box_max = (span / 2, span / 2, span / 2)
        res = (cells * res_per_cell,) * 3
    else:  # Scherk tower: periodic in z only, clip x/y
        w = 2.2
        box_min = (-w, -w, -cells * math.pi)
        box_max = (w, w, cells * math.pi)
        rxy = int(res_per_cell * 1.4)
        res = (rxy, rxy, cells * res_per_cell)
    verts, tris = marching_tets(field, box_min, box_max, res)
    s = scale / TAU  # one period -> `scale` Blender units
    return verts * s, tris


# ==========================================================================
# 3. Plateau solver (area minimization with pinned boundaries)
# ==========================================================================

def resample_loop(pts, m):
    """Uniform-arclength resample of a closed polyline (k,3) -> (m,3)."""
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    closed = np.vstack([pts, pts[:1]])
    t = np.linspace(0.0, total, m, endpoint=False)
    out = np.empty((m, 3))
    j = 0
    for i, ti in enumerate(t):
        while s[j + 1] < ti:
            j += 1
        f = (ti - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out[i] = closed[j] * (1 - f) + closed[j + 1] * f
    return out


def align_loops(A, B):
    """Cyclic shift + optional reversal of B minimizing sum |A_i - B_i|^2."""
    m = len(A)
    best = (None, 1e30)
    for Bc in (B, B[::-1]):
        # brute-force cyclic shifts (m <= a few hundred)
        for sft in range(m):
            d = np.sum((A - np.roll(Bc, -sft, axis=0)) ** 2)
            if d < best[1]:
                best = (np.roll(Bc, -sft, axis=0), d)
    return best[0]


def _cotan_weights(V, T):
    """Per-edge cotangent weights. Returns (edges (e,2), w (e,))."""
    ijk = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    E = []
    W = []
    for (a, b, c) in ijk:
        u = V[T[:, a]] - V[T[:, c]]
        v = V[T[:, b]] - V[T[:, c]]
        cross = np.cross(u, v)
        denom = np.linalg.norm(cross, axis=1)
        cot = np.einsum('ij,ij->i', u, v) / np.maximum(denom, 1e-12)
        # clamp to positive: keeps the system positive-definite (maximum
        # principle), so the CG solve cannot blow up on degenerate fans
        cot = np.clip(cot, 0.01, 20.0)
        E.append(np.stack([T[:, a], T[:, b]], axis=1))
        W.append(0.5 * cot)
    return np.concatenate(E), np.concatenate(W)


def minimize_area(V, T, fixed, outer_iters=30, cg_tol=1e-8, cg_iters=400,
                  uniform=False):
    """Pinkall-Polthier: repeatedly solve the cotan-Laplace equation for
    the interior vertices (boundary pinned). V modified in place.
    With uniform=True, unit weights are used instead of cotangents --
    a Tutte-style fairing solve that untangles folded regions (at the
    cost of exact minimality; follow with a cotan pass)."""
    n = len(V)
    free = ~fixed
    nfree = int(np.sum(free))
    if nfree == 0:
        return V
    for _ in range(outer_iters):
        E, W = _cotan_weights(V, T)
        if uniform:
            W = np.ones_like(W)
        deg = np.zeros(n)
        np.add.at(deg, E[:, 0], W)
        np.add.at(deg, E[:, 1], W)

        def matvec_full(Xf):
            y = deg[:, None] * Xf
            np.add.at(y, E[:, 0], -W[:, None] * Xf[E[:, 1]])
            np.add.at(y, E[:, 1], -W[:, None] * Xf[E[:, 0]])
            return y

        Xb = np.where(fixed[:, None], V, 0.0)
        b = -matvec_full(Xb)[free]

        def matvec(xf):
            full = np.zeros((n, 3))
            full[free] = xf
            return matvec_full(full)[free]

        x = V[free].copy()
        r = b - matvec(x)
        p = r.copy()
        rs = np.sum(r * r, axis=0)
        b_norm = max(np.max(np.sum(b * b, axis=0)), 1e-30)
        for _cg in range(cg_iters):
            Ap = matvec(p)
            pAp = np.sum(p * Ap, axis=0)
            alpha = rs / np.where(np.abs(pAp) > 1e-30, pAp, 1e-30)
            x += alpha * p
            r -= alpha * Ap
            rs_new = np.sum(r * r, axis=0)
            if np.max(rs_new) < cg_tol * b_norm:
                break
            p = r + (rs_new / np.maximum(rs, 1e-30)) * p
            rs = rs_new
        move = np.max(np.linalg.norm(x - V[free], axis=1))
        V[free] = x
        if move < 1e-6 * max(1.0, np.max(np.abs(V))):
            break
    return V


def relax_normal_flow(V, T, fixed, iters=60, lam=0.4):
    """Mean-curvature flow restricted to the surface normal: pulls a
    (slightly perturbed) net back toward the minimal surface without
    tangential sliding, so a fair control net stays fair. Only suitable
    as a polish -- it cannot perform global reorganization."""
    free = ~fixed
    n = len(V)
    for _ in range(iters):
        E, W = _cotan_weights(V, T)
        lap = np.zeros((n, 3))
        d = V[E[:, 1]] - V[E[:, 0]]
        np.add.at(lap, E[:, 0], W[:, None] * d)
        np.add.at(lap, E[:, 1], -W[:, None] * d)
        wsum = np.zeros(n)
        np.add.at(wsum, E[:, 0], W)
        np.add.at(wsum, E[:, 1], W)
        umb = lap / np.maximum(wsum, 1e-12)[:, None]
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        vn = np.zeros((n, 3))
        np.add.at(vn, T[:, 0], fn)
        np.add.at(vn, T[:, 1], fn)
        np.add.at(vn, T[:, 2], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        move = np.sum(umb * vn, axis=1, keepdims=True) * vn
        V[free] += lam * move[free]
    return V


def fair_grid_2d(G, iters=8, step=0.5):
    """Light Laplacian fairing of an (rows, m, 3) net, cyclic in m,
    end rows pinned. Restores row/column coherence after per-column
    resampling; follow with relax_normal_flow to restore minimality."""
    G = G.copy()
    for _ in range(iters):
        up = np.roll(G, 1, axis=0)
        dn = np.roll(G, -1, axis=0)
        up[0] = G[0]
        dn[-1] = G[-1]
        lt = np.roll(G, 1, axis=1)
        rt = np.roll(G, -1, axis=1)
        avg = (up + dn + lt + rt) / 4.0
        G[1:-1] += step * (avg[1:-1] - G[1:-1])
    return G


def fair_grid_columns(G):
    """Re-sample every column (axis 0) of an (rows, m, 3) grid uniformly
    by arc length. Points stay on their column polylines -- i.e. on the
    solved surface -- but the severe bunching/shear the area solver
    introduces (which makes a NURBS control net ring) is equalized."""
    rows, m, _ = G.shape
    out = np.empty_like(G)
    for i in range(m):
        P = G[:, i, :]
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total < 1e-12:
            out[:, i, :] = P
            continue
        t = np.linspace(0.0, total, rows)
        for a in range(3):
            out[:, i, a] = np.interp(t, s, P[:, a])
    return out


def mesh_area(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return 0.5 * float(np.sum(np.linalg.norm(n, axis=1)))


def _quads_to_tris(quads):
    T = []
    for q in quads:
        if len(q) == 4:
            T.append((q[0], q[1], q[2]))
            T.append((q[0], q[2], q[3]))
        else:
            T.append(tuple(q))
    return np.array(T, dtype=np.int64)


def build_disk_grid(boundary, rings):
    """Initial disk spanning one loop: concentric rings toward centroid."""
    m = len(boundary)
    cen = boundary.mean(axis=0)
    verts = [boundary]
    for r in range(1, rings):
        f = r / rings
        verts.append(boundary * (1 - f) + cen * f)
    V = np.concatenate(verts + [cen[None, :]], axis=0)
    quads = []
    for r in range(rings - 1):
        for i in range(m):
            i2 = (i + 1) % m
            quads.append((r * m + i, r * m + i2,
                          (r + 1) * m + i2, (r + 1) * m + i))
    last = (rings - 1) * m
    cidx = rings * m
    for i in range(m):
        quads.append((last + i, last + (i + 1) % m, cidx))
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:m] = True
    return V, quads, fixed


def build_annulus_grid(loopA, loopB, rows):
    """Initial ruled surface between two aligned loops of equal length."""
    m = len(loopA)
    verts = []
    for r in range(rows + 1):
        f = r / rows
        verts.append(loopA * (1 - f) + loopB * f)
    V = np.concatenate(verts, axis=0)
    quads = []
    for r in range(rows):
        for i in range(m):
            i2 = (i + 1) % m
            quads.append((r * m + i, r * m + i2,
                          (r + 1) * m + i2, (r + 1) * m + i))
    fixed = np.zeros(len(V), dtype=bool)
    fixed[:m] = True
    fixed[rows * m:] = True
    return V, quads, fixed


def torus_knot(p, q, m, scale=1.0, tube=1.0):
    t = np.linspace(0, TAU, m, endpoint=False)
    r = np.cos(q * t) * tube + 2.0
    return np.stack([r * np.cos(p * t), r * np.sin(p * t),
                     -np.sin(q * t) * tube], axis=1) * scale


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _nurbs_grid_object(context, name, grid, cyclic_u=False,
                           cyclic_v=False, order=4):
        """Create a NURBS surface object from an (nu, nv, 3) control grid.
        `u` runs across rows, `v` along each row. Grids must be built
        row-by-row as separate splines and merged with make_segment (the
        only scriptable way to produce a surface grid in Blender)."""
        grid = np.asarray(grid)
        nu, nv = grid.shape[:2]
        su = bpy.data.curves.new(name, 'SURFACE')
        su.dimensions = '3D'
        for i in range(nu):
            sp = su.splines.new('NURBS')
            sp.points.add(nv - 1)
            flat = np.concatenate(
                [grid[i], np.ones((nv, 1))], axis=1).ravel()
            sp.points.foreach_set('co', flat)
        obj = bpy.data.objects.new(name, su)
        context.collection.objects.link(obj)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.curve.select_all(action='SELECT')
        bpy.ops.curve.make_segment()
        bpy.ops.object.mode_set(mode='OBJECT')
        sp = su.splines[0]
        # make_segment chains the selected splines in an arbitrary
        # (geometry-dependent) order, so only the resulting GRID
        # STRUCTURE is trustworthy. Rewrite every control point into the
        # intended layout (storage is u-fastest: flat = v*pu + u).
        pu, pv = sp.point_count_u, sp.point_count_v
        if (pu, pv) == (nu, nv):
            ordered = grid.transpose(1, 0, 2)   # S[v][u] = grid[u][v]
            cu, cv = cyclic_u, cyclic_v
        else:                                   # (pu, pv) == (nv, nu)
            ordered = grid                      # S[v][u] = grid[v][u]
            cu, cv = cyclic_v, cyclic_u
        flat = np.concatenate(
            [ordered.reshape(-1, 3), np.ones((pu * pv, 1))],
            axis=1).ravel()
        sp.points.foreach_set('co', flat)
        sp.use_cyclic_u = cu
        sp.use_cyclic_v = cv
        sp.use_endpoint_u = not cu
        sp.use_endpoint_v = not cv
        sp.order_u = min(order, pu)
        sp.order_v = min(order, pv)
        sp.resolution_u = 6
        sp.resolution_v = 6
        su.update_tag()
        obj.location = context.scene.cursor.location
        return obj

    def _new_object(context, name, verts, faces, weld=0.0, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        if weld > 0:
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
            bm.to_mesh(me)
            bm.free()
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        obj = bpy.data.objects.new(name, me)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj

    def _extract_loop(obj, depsgraph):
        """Ordered closed polyline (world space) from a curve or mesh
        object. Returns ndarray (k,3) or raises ValueError."""
        ev = obj.evaluated_get(depsgraph)
        me = ev.to_mesh()
        try:
            if len(me.vertices) == 0:
                raise ValueError(f"{obj.name}: no geometry")
            adj = {}
            for e in me.edges:
                a, b = e.vertices
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            if not adj or any(len(v) != 2 for v in adj.values()):
                raise ValueError(
                    f"{obj.name}: must be a single closed loop "
                    f"(every point joining exactly 2 segments)")
            start = next(iter(adj))
            loop = [start]
            prev, cur = None, start
            while True:
                nxt = [v for v in adj[cur] if v != prev]
                if not nxt:
                    raise ValueError(f"{obj.name}: open curve")
                prev, cur = cur, nxt[0]
                if cur == start:
                    break
                loop.append(cur)
                if len(loop) > len(me.vertices):
                    raise ValueError(f"{obj.name}: not a single loop")
            mw = np.array(obj.matrix_world)
            pts = np.array([me.vertices[i].co[:] for i in loop])
            pts = pts @ mw[:3, :3].T + mw[:3, 3]
            return pts
        finally:
            ev.to_mesh_clear()

    class MESH_OT_parametric_minimal_add(bpy.types.Operator):
        """Add a classic parametric minimal surface"""
        bl_idname = "mesh.parametric_minimal_add"
        bl_label = "Classic Minimal Surface"
        bl_options = {'REGISTER', 'UNDO'}

        surface: EnumProperty(
            name="Surface",
            items=[(k, v[0], v[0]) for k, v in PARAMETRIC.items()],
            default='ENNEPER')
        output: EnumProperty(
            name="Output",
            items=[('MESH', "Mesh", "Dense polygon mesh"),
                   ('NURBS', "NURBS", "Compact NURBS surface patch "
                                      "(control grid = Resolution U x V)")],
            default='MESH')
        res_u: IntProperty(name="Resolution U", default=48, min=8, max=512)
        res_v: IntProperty(name="Resolution V", default=48, min=8, max=512)
        ctrl_u: IntProperty(
            name="Control Points U", default=24, min=6, max=128,
            description="NURBS control grid size in U")
        ctrl_v: IntProperty(
            name="Control Points V", default=24, min=6, max=128,
            description="NURBS control grid size in V")
        order: IntProperty(
            name="Order / Count", default=1, min=1, max=12,
            description="Enneper order; helicoid half-turns; Jorge-Meeks "
                        "end count n (>= 3); ignored for the rest")
        radius: FloatProperty(
            name="Domain Radius", default=1.2, min=0.2, max=4.0,
            description="Extent of the parameter domain (for the k-noid, "
                        "how close the disk reaches its ends)")
        assoc_angle: FloatProperty(
            name="Associate Angle", default=0.0,
            min=0.0, max=math.pi / 2.0, subtype='ANGLE',
            description="Bonnet associate family: 0 = catenoid, "
                        "pi/2 = helicoid")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Multiplier on the normalized size (1.0 = a 2 m "
                        "cube, centered on the origin)")

        def execute(self, context):
            label = PARAMETRIC[self.surface][0]
            theta = (self.assoc_angle if self.surface in ANGLE_PARAM
                     else 0.0)
            # some surfaces are assembled meshes with no NURBS/grid form
            if self.output == 'NURBS' and self.surface not in MESH_PARAM:
                G, wrap_u, wrap_v = build_parametric_grid(
                    self.surface, self.ctrl_u, self.ctrl_v,
                    self.order, self.radius, self.scale, theta)
                if wrap_u:          # drop duplicated periodic endpoint
                    G = G[:-1]
                if wrap_v:
                    G = G[:, :-1]
                _nurbs_grid_object(context, label, G,
                                   cyclic_u=wrap_u, cyclic_v=wrap_v)
            else:
                V, quads = build_parametric(self.surface, self.res_u,
                                            self.res_v, self.order,
                                            self.radius, self.scale, theta)
                _new_object(context, label, V, quads,
                            weld=1e-5 * max(1.0, self.scale))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'surface')
            mesh_only = self.surface in MESH_PARAM
            if not mesh_only:
                lay.prop(self, 'output')
            if self.output == 'NURBS' and not mesh_only:
                lay.prop(self, 'ctrl_u')
                lay.prop(self, 'ctrl_v')
            else:
                lay.prop(self, 'res_u')
                lay.prop(self, 'res_v')
            if self.surface in COUNT_PARAM:
                lay.prop(self, 'order', text=COUNT_PARAM[self.surface])
            elif self.surface not in ANGLE_PARAM:
                lay.prop(self, 'order')
            if self.surface in ANGLE_PARAM:
                lay.prop(self, 'assoc_angle')
            lay.prop(self, 'radius')
            lay.prop(self, 'scale')

    class MESH_OT_tpms_add(bpy.types.Operator):
        """Add a triply-periodic minimal surface (nodal approximation)"""
        bl_idname = "mesh.tpms_add"
        bl_label = "Periodic Minimal Surface (TPMS)"
        bl_options = {'REGISTER', 'UNDO'}

        surface: EnumProperty(
            name="Surface",
            items=[(k, v[0], v[0]) for k, v in TPMS.items()],
            default='G')
        cells: IntProperty(
            name="Cells", default=1, min=1, max=4,
            description="Number of unit cells per axis")
        resolution: IntProperty(
            name="Resolution / Cell", default=28, min=8, max=80,
            description="Sample grid resolution per unit cell")
        cell_size: FloatProperty(
            name="Cell Size", default=2.0, min=0.1, max=100.0,
            description="Edge length of one unit cell in Blender units")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this thickness")

        def execute(self, context):
            verts, tris = build_tpms(self.surface, self.cells,
                                     self.resolution, self.cell_size)
            if len(tris) == 0:
                self.report({'ERROR'}, "Empty level set")
                return {'CANCELLED'}
            label = TPMS[self.surface][0]
            obj = _new_object(context, label, verts, tris)
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('surface', 'cells', 'resolution', 'cell_size',
                      'thickness'):
                lay.prop(self, k)

    class OBJECT_OT_minimal_span(bpy.types.Operator):
        """Span a minimal surface across the selected curve (1 object:
        disk) or between two selected curves (2 objects: annulus)"""
        bl_idname = "object.minimal_span"
        bl_label = "Span Minimal Surface"
        bl_options = {'REGISTER', 'UNDO'}

        samples: IntProperty(
            name="Boundary Samples", default=128, min=16, max=512)
        rings: IntProperty(
            name="Interior Rings", default=24, min=3, max=128)
        iterations: IntProperty(
            name="Solver Iterations", default=40, min=1, max=200)
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Emit a compact NURBS surface (control grid = the "
                        "solver grid) instead of a dense mesh. Where the "
                        "surface curls tightly (e.g. near a knot) the NURBS "
                        "may ripple; raise rings/samples or use mesh output")

        @classmethod
        def poll(cls, context):
            sel = [o for o in context.selected_objects
                   if o.type in ('CURVE', 'MESH')]
            return len(sel) in (1, 2)

        def execute(self, context):
            deps = context.evaluated_depsgraph_get()
            sel = [o for o in context.selected_objects
                   if o.type in ('CURVE', 'MESH')]
            try:
                loops = [_extract_loop(o, deps) for o in sel]
            except ValueError as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            m = self.samples
            if len(loops) == 1:
                A = resample_loop(loops[0], m)
                V, quads, fixed = build_disk_grid(A, self.rings)
                rows = self.rings   # + center point row
            else:
                A = resample_loop(loops[0], m)
                B = align_loops(A, resample_loop(loops[1], m))
                V, quads, fixed = build_annulus_grid(A, B, self.rings)
                rows = self.rings + 1
            T = _quads_to_tris(quads)
            minimize_area(V, T, fixed, outer_iters=self.iterations)
            if self.output_nurbs:
                G = fair_grid_columns(V[:rows * m].reshape(rows, m, 3))
                G = fair_grid_2d(G)
                V[:rows * m] = G.reshape(-1, 3)
                relax_normal_flow(V, T, fixed)
                G = V[:rows * m].reshape(rows, m, 3)
                if len(loops) == 1:   # close the pole with the center point
                    cen = np.tile(V[-1], (1, m, 1))
                    G = np.concatenate([G, cen], axis=0)
                obj = _nurbs_grid_object(context, "MinimalSpan", G,
                                         cyclic_u=False, cyclic_v=True)
            else:
                obj = _new_object(context, "MinimalSpan", V, quads)
            obj.location = (0, 0, 0)   # loops are in world space
            self.report({'INFO'},
                        f"area = {mesh_area(V, T):.4f}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('samples', 'rings', 'iterations', 'output_nurbs'):
                lay.prop(self, k)

    class MESH_OT_knot_span_add(bpy.types.Operator):
        """Minimal surface between a circle and a (p,q) torus knot
        (trefoil by default) -- the classic Plateau demonstration"""
        bl_idname = "mesh.minimal_knot_span_add"
        bl_label = "Knot to Knot Surface"
        bl_options = {'REGISTER', 'UNDO'}

        p: IntProperty(name="Knot p", default=2, min=1, max=8)
        q: IntProperty(
            name="Knot q", default=3, min=0, max=9,
            description="q of the inner torus knot; 0 degenerates "
                        "it to a flat circle (radius 3 x Knot "
                        "Scale) wound p times")
        circle_radius: FloatProperty(
            name="Circle Radius", default=4.5, min=1.0, max=20.0)
        knot_scale: FloatProperty(
            name="Knot Scale", default=1.0, min=0.1, max=5.0)
        inner_height: FloatProperty(
            name="Inner Height", default=1.0, min=0.0, max=5.0,
            description="Scale of the inner boundary's vertical "
                        "oscillation, independent of its radius "
                        "(0 flattens it into a wavy-radius ring)")
        inner_lift: FloatProperty(
            name="Inner Lift", default=0.0, min=-10.0, max=10.0,
            description="Shift the inner boundary up or down; with "
                        "two circles this makes a catenoid-style "
                        "span")
        inner_rotation: FloatProperty(
            name="Inner Rotation", default=0.0,
            min=-2.0 * math.pi, max=2.0 * math.pi, subtype='ANGLE',
            description="Rotate the inner boundary about the "
                        "vertical axis relative to the outer one, "
                        "twisting the ruling between them")
        samples: IntProperty(
            name="Boundary Samples", default=96, min=32, max=512)
        rings: IntProperty(name="Interior Rings", default=16, min=4, max=128)
        iterations: IntProperty(
            name="Solver Iterations", default=2, min=1, max=200)
        output_nurbs: BoolProperty(
            name="NURBS Output", default=False,
            description="Emit a compact NURBS surface (control grid = the "
                        "solver grid) instead of a dense mesh. Where the "
                        "surface curls tightly (e.g. near a knot) the NURBS "
                        "may ripple; raise rings/samples or use mesh output")
        outer_q: IntProperty(
            name="Outer Knot q", default=0, min=0, max=9,
            description="The outer boundary as a (p, q) torus knot; "
                        "0 keeps the flat round circle (a circle is "
                        "the degenerate q = 0 knot)")
        outer_p: IntProperty(
            name="Outer Knot p", default=0, min=0, max=8,
            description="p of the outer boundary; 0 matches the "
                        "inner knot's p (which keeps the ruling "
                        "lined up)")
        outer_scale: FloatProperty(
            name="Outer Knot Scale", default=2.0, min=0.1, max=10.0,
            description="Scale of the outer torus knot (its radii "
                        "are ~1-3 x this)")
        outer_height: FloatProperty(
            name="Outer Height", default=1.0, min=0.0, max=5.0,
            description="Scale of the outer knot's vertical "
                        "oscillation, independent of its radius "
                        "(0 flattens it into a wavy-radius ring)")
        split_sheets: BoolProperty(
            name="Split Sheets", default=False,
            description="The span winds its outer boundary p times "
                        "and the sheets pass through one another; "
                        "this outputs p separate one-winding sheet "
                        "objects instead (they share their seam "
                        "edges, so together they still form the "
                        "whole span)")

        def execute(self, context):
            m = self.samples
            if self.split_sheets and self.p > 1:
                m = max(self.p * 8, (m // self.p) * self.p)
            knot = torus_knot(self.p, self.q, m, scale=self.knot_scale)
            knot[:, 2] *= self.inner_height
            knot[:, 2] += self.inner_lift
            if self.inner_rotation != 0.0:
                ca = math.cos(self.inner_rotation)
                sa = math.sin(self.inner_rotation)
                knot[:, :2] = np.stack(
                    [knot[:, 0] * ca - knot[:, 1] * sa,
                     knot[:, 0] * sa + knot[:, 1] * ca], axis=1)
            t = np.linspace(0, TAU, m, endpoint=False)
            po = self.outer_p or self.p
            if self.outer_q > 0:
                # outer boundary: another torus knot
                circ = torus_knot(po, self.outer_q, m,
                                  scale=self.outer_scale)
                circ[:, 2] *= self.outer_height
            else:
                # circle wound p times so the ruling lines up
                # (outer_p is ignored while the outer boundary is
                # a circle -- it is hidden in the UI then, and a
                # stale value must not change the winding)
                circ = np.stack(
                    [self.circle_radius * np.cos(self.p * t),
                     self.circle_radius * np.sin(self.p * t),
                     np.zeros(m)], axis=1)
            V, quads, fixed = build_annulus_grid(knot, circ, self.rings)
            T = _quads_to_tris(quads)
            minimize_area(V, T, fixed, outer_iters=self.iterations)
            # center on the origin and fit within a 2 m cube
            V = _center_fit(V, 1.0)
            name = (f"Knot({self.p},{self.q})Span" if self.outer_q == 0
                    else f"Knot({self.p},{self.q})-"
                         f"({po},{self.outer_q})Span")
            if self.split_sheets and self.p > 1:
                # one object per winding: columns [k w, (k+1) w] of
                # the solver grid, seam columns shared between
                # neighboring sheets
                w = m // self.p
                R = self.rings + 1
                if self.output_nurbs:
                    Gs = fair_grid_columns(V.reshape(R, m, 3))
                    Gs = fair_grid_2d(Gs)
                    V2 = Gs.reshape(-1, 3).copy()
                    relax_normal_flow(V2, T, fixed)
                    Gs = V2.reshape(R, m, 3)
                else:
                    Gs = V.reshape(R, m, 3)
                made = []
                for k in range(self.p):
                    cols = [(k * w + j) % m for j in range(w + 1)]
                    nm = f"{name} Sheet {k + 1}of{self.p}"
                    if self.output_nurbs:
                        made.append(_nurbs_grid_object(
                            context, nm, Gs[:, cols],
                            cyclic_u=False, cyclic_v=False))
                    else:
                        Vk = Gs[:, cols, :].reshape(-1, 3)
                        qk = [(r * (w + 1) + j,
                               r * (w + 1) + j + 1,
                               (r + 1) * (w + 1) + j + 1,
                               (r + 1) * (w + 1) + j)
                              for r in range(self.rings)
                              for j in range(w)]
                        made.append(_new_object(context, nm, Vk, qk))
                for o in made:
                    o.select_set(True)
                self.report({'INFO'},
                            f"{self.p} sheets, area = "
                            f"{mesh_area(V, T):.4f}")
                return {'FINISHED'}
            if self.output_nurbs:
                G = fair_grid_columns(V.reshape(self.rings + 1, m, 3))
                G = fair_grid_2d(G)
                V = G.reshape(-1, 3).copy()
                relax_normal_flow(V, T, fixed)
                G = V.reshape(self.rings + 1, m, 3)
                obj = _nurbs_grid_object(context, name, G,
                                         cyclic_u=False, cyclic_v=True)
            else:
                obj = _new_object(context, name, V, quads)
            self.report({'INFO'}, f"area = {mesh_area(V, T):.4f}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            for k in ('p', 'q', 'knot_scale', 'inner_height',
                      'inner_lift', 'inner_rotation', 'outer_q'):
                lay.prop(self, k)
            if self.outer_q > 0:
                lay.prop(self, 'outer_p')
                lay.prop(self, 'outer_scale')
                lay.prop(self, 'outer_height')
            else:
                lay.prop(self, 'circle_radius')
            for k in ('samples', 'rings', 'iterations',
                      'output_nurbs'):
                lay.prop(self, k)
            if self.p > 1:
                lay.prop(self, 'split_sheets')

    class VIEW3D_PT_minimal_surfaces(bpy.types.Panel):
        bl_label = "Minimal Surfaces"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = "Minimal Surfaces"

        def draw(self, context):
            lay = self.layout
            col = lay.column(align=True)
            col.operator("mesh.parametric_minimal_add", icon='SURFACE_NSPHERE')
            col.operator("mesh.tpms_add", icon='MESH_ICOSPHERE')
            col.separator()
            col.operator("mesh.minimal_knot_span_add", icon='MESH_TORUS')
            col.label(text="Select 1-2 closed curves, then:")
            col.operator("object.minimal_span", icon='OUTLINER_OB_SURFACE')

    class VIEW3D_MT_minimal_add(bpy.types.Menu):
        bl_idname = "VIEW3D_MT_minimal_add"
        bl_label = "Minimal Surfaces"

        def draw(self, context):
            lay = self.layout
            lay.operator("mesh.parametric_minimal_add")
            lay.operator("mesh.tpms_add")
            lay.operator("mesh.minimal_knot_span_add")
            lay.operator("object.minimal_span")

    def _menu_func(self, context):
        self.layout.menu("VIEW3D_MT_minimal_add", icon='SURFACE_DATA')

    _classes = (MESH_OT_parametric_minimal_add, MESH_OT_tpms_add,
                OBJECT_OT_minimal_span, MESH_OT_knot_span_add,
                VIEW3D_PT_minimal_surfaces, VIEW3D_MT_minimal_add)

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        # standalone smoke tests of the numeric core
        ok = True
        # Weierstrass engine invariants on the square torus
        L = _SQUARE
        e1 = L.wp(0.5).real
        zt = np.array([0.2 + 0.3j, 0.37 + 0.11j, 0.6 + 0.44j])
        resid = np.max(np.abs(L.wp_prime(zt) ** 2
                              - (4 * L.wp(zt) ** 3 - 4 * e1 ** 2 * L.wp(zt))))
        print(f"weierstrass: e1={e1:.5f} (exp 6.87519) g2={4*e1**2:.4f} "
              f"(exp 189.0727) |P'^2-(4P^3-g2 P)|={resid:.2e} "
              f"{'OK' if abs(e1-6.87519) < 1e-3 and resid < 1e-8 else 'FAIL'}")
        ok &= abs(e1 - 6.87519) < 1e-3 and resid < 1e-8
        for kind in PARAMETRIC:
            th = math.pi / 4 if kind in ANGLE_PARAM else 0.0
            n = 5 if kind == 'KNOID' else 1
            V, Q = build_parametric(kind, 60, 60, n, 1.2, 1.0, th)
            finite = bool(np.all(np.isfinite(V)))
            lo, hi = V.min(0), V.max(0)
            cen = float(np.max(np.abs(0.5 * (lo + hi))))
            ext = float(np.max(hi - lo))
            good = (finite and len(Q) > 100 and cen < 1e-6
                    and abs(ext - 2.0) < 1e-6)
            ok &= good
            print(f"parametric {kind:10s}: {len(V):5d} verts {len(Q):5d} "
                  f"quads  fit[max|c|={cen:.1e} ext={ext:.4f}] "
                  f"{'OK' if good else 'FAIL'}")
        # Costa-Hoffman-Meeks: modulus table + Euler characteristic gate
        # (genus k, 3 ends removed -> chi = 2 - 2k - 3 = -(2k+1))
        cref = {1: 0.955978, 2: 0.988070, 3: 0.995117, 4: 0.997535}
        cok = all(abs(_chm_modulus(kk) - cref[kk]) < 1e-5 for kk in cref)
        print(f"CHM modulus: "
              + " ".join(f"c({kk})={_chm_modulus(kk):.5f}" for kk in cref)
              + f"  {'OK' if cok else 'FAIL'}")
        ok &= cok
        for kk in (1, 2, 3):
            Vc, Qc = build_parametric('COSTA_HM', 48, 48, kk, 1.2, 1.0)
            ec = {}
            for f in Qc:
                for t in range(len(f)):
                    a, b = f[t], f[(t + 1) % len(f)]
                    e = (a, b) if a < b else (b, a)
                    ec[e] = ec.get(e, 0) + 1
            chi = len(Vc) - len(ec) + len(Qc)
            want = -(2 * kk + 1)
            good = chi == want and np.all(np.isfinite(Vc))
            ok &= good
            print(f"CHM genus {kk}: verts={len(Vc)} faces={len(Qc)} "
                  f"chi={chi} (want {want}) {'OK' if good else 'FAIL'}")
        # k-noid vs closed-form trinoid (n=3): both are minimal with 3
        # ends; compare 3-fold symmetry of the numeric build
        Vn, _ = build_parametric('KNOID', 72, 72, 3, 0.9, 1.0)
        ang = np.arctan2(Vn[:, 1], Vn[:, 0])
        print(f"k-noid n=3: verts={len(Vn)} z-range="
              f"[{Vn[:,2].min():.3f},{Vn[:,2].max():.3f}] "
              f"{'OK' if np.all(np.isfinite(Vn)) else 'FAIL'}")
        for kind in TPMS:
            V, T = build_tpms(kind, 1, 20, 2.0)
            print(f"tpms {kind:10s}: {len(V):6d} verts {len(T):6d} tris")
        # flat-disk validation: minimal surface on a planar circle is flat
        t = np.linspace(0, TAU, 96, endpoint=False)
        circle = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
        V, quads, fixed = build_disk_grid(circle, 12)
        V[~fixed] += np.random.default_rng(1).normal(0, 0.1, V[~fixed].shape)
        T = _quads_to_tris(quads)
        minimize_area(V, T, fixed)
        print("flat disk: max|z| =", float(np.max(np.abs(V[:, 2]))),
              " area =", mesh_area(V, T), " (pi =", math.pi, ")")
        # catenoid validation: two rings radius 1 at z = +/-0.4
        z0 = 0.4
        ringA = np.stack([np.cos(t), np.sin(t), np.full_like(t, z0)], axis=1)
        ringB = np.stack([np.cos(t), np.sin(t), np.full_like(t, -z0)], axis=1)
        V, quads, fixed = build_annulus_grid(ringA, ringB, 20)
        T = _quads_to_tris(quads)
        minimize_area(V, T, fixed)
        waist = np.min(np.linalg.norm(V[:, :2], axis=1))
        print("catenoid: waist =", float(waist), "(analytic ~0.9098)")
        print("\nRESULT:", "ALL OK" if ok else "FAILURES in parametric core")
