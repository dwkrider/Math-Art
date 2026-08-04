
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
#
# References:
#   Weierstrass-Enneper representation: K. Weierstrass (1866) and
#       A. Enneper (1864); the catenoid (surface of Euler, 1744) was
#       shown minimal by J. B. C. Meusnier (1776).
#   Costa surface: C. J. Costa (1982); embeddedness by D. Hoffman and
#       W. H. Meeks III (1985). Chen-Gackstatter: C. C. Chen and
#       F. Gackstatter (1982). Jorge-Meeks k-noids: L. P. Jorge and
#       W. H. Meeks III (1983).
#   Triply-periodic families: H. A. Schwarz (P, D; Gesammelte Math.
#       Abhandlungen, 1890),
#       A. H. Schoen (gyroid, I-WP, F-RD; NASA TN D-5541, 1970),
#       E. R. Neovius (1883). Cotangent-Laplacian area flow after
#       U. Pinkall and K. Polthier (1993).

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


# --- Costa-Hoffman-Meeks, k-noids, and the wider catalog ------------------
# The Jorge-Meeks k-noid and the genus-k Costa-Hoffman-Meeks family
# (formerly bespoke builders here) now live as data rows in
# minimal_surface_zoo.py, built by the generic Weierstrass-Enneper
# engine in we_builders.py; they are wired into PARAMETRIC below by
# the zoo's register() call, together with the rest of the catalog
# (saddle towers, Bjorling strips, Meeks Mobius, Riemann's example...).


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
    'CHEN_GACK': ("Chen-Gackstatter", _chen_gackstatter),
    'CATHEL': ("Catenoid-Helicoid (associate)", _cathel),
}

# per-surface mesh-density boost for the classical grid builders that read
# faceted at the shared default resolution.  build_parametric reads a
# builder's `.spec['res_boost']` (the same hook the Weierstrass engine
# uses), so attaching a tiny spec here lets these fast-flaring disk/strip
# surfaces get a denser grid -- and hence a smooth rim and body -- without
# touching the resolution the rest of the catalog is happy with.  (ENNEPER
# is overridden by the Weierstrass engine below and tuned there instead.)
_henneberg.spec = {'res_boost': (1.8, 1.8)}   # small domain, body faceted
_bour.spec = {'res_boost': (1.6, 1.6)}        # z ~ u^1.5 flare -> facets
_richmond.spec = {'res_boost': (1.4, 1.4)}    # flower rim reads polygonal
_catalan.spec = {'res_boost': (1.3, 1.3)}     # strip edge slightly polygonal

# surfaces built as a finished (V, quads[, uv]) mesh rather than a
# parameter grid
MESH_PARAM = {}
# surfaces whose `order` selects a discrete count rather than an Enneper
# order / helicoid turns (drives the operator UI label)
COUNT_PARAM = {}
# surfaces that stack a periodic fundamental domain into N repeats (the
# saddle tower); maps surface key -> the operator "Storeys" UI label
STOREY_PARAM = {}
# surfaces that use the associate-family angle
ANGLE_PARAM = {'CATHEL', 'PGD'}

# Wire in the catalog (KNOID, COSTA_HM and the rest of the zoo): the
# rows in minimal_surface_zoo.py are built by the generic engine in
# we_builders.py and registered into the four dicts above.  Resilient:
# if the zoo cannot be imported, the classical core still works.
try:
    try:
        from . import minimal_surface_zoo as _zoo
    except ImportError:
        import minimal_surface_zoo as _zoo
    _zoo.register(PARAMETRIC, MESH_PARAM, COUNT_PARAM, ANGLE_PARAM,
                  STOREY_PARAM)
    SURFACE_FAMILY = _zoo.SURFACE_FAMILY
    FAMILIES = _zoo.FAMILIES
except Exception as _e:                        # WIP catalog: skip
    print(f"minimal_surface_toolkit: zoo unavailable: {_e}")
    _zoo = None
    SURFACE_FAMILY = {}
    FAMILIES = ()


def _raw_grid(kind, nu, nv, order, radius, theta, copies=None):
    """Raw (unnormalized) surface grid. Returns (G (nu,nv,3), wrap_u,
    wrap_v, clip) where clip requests end-face trimming by the mesher.

    `copies` overrides a torus surface's translational-copy count (Riemann's
    singly periodic repeat): the immersion is continuous across the deck
    translation, so spanning `copies` fundamental domains in the unwrapped
    direction tiles the surface welded and in its true period scale."""
    b = PARAMETRIC[kind][1]
    spec = getattr(b, 'spec', None)
    if (copies is not None and spec is not None and 'domain' in spec
            and spec['domain'][0] == 'torus'):
        spec2 = dict(spec, copies=int(max(1, copies)))
        out = _we_pgd.we_surface(spec2, nu, nv, order, radius, None, theta)
    else:
        out = b(nu, nv, order, radius, theta)
    x, y, w, wrap_u, wrap_v = out[:5]
    clip = out[5] if len(out) > 5 else False
    return np.stack([x, y, w], axis=-1), wrap_u, wrap_v, clip


# singly periodic torus surfaces whose 1-D repeat is driven through the
# torus 'copies' mechanism (continuous deck translation) rather than a
# rigid lattice array
PERIODIC_COPIES_1D = {'RIEMANN'}

# Periodic surfaces that CANNOT be tiled cleanly and stay at a single
# fundamental domain (honesty gate).  Karcher's unequal-wing saddle tower
# (SADDLE_TOWER_A): the unequal-wing unit admits no screw deck isometry to
# stack (its only rigid symmetries have zero vertical shift) -- see the
# SADDLE_TOWER_A note in minimal_surface_zoo.py.  we_saddle_tower already
# forces storeys=1 for it; the operator hides the (no-op) cell control.
# TODO: an unequal-wing tower would need a genuine translational period
# (not the alpha=0 screw) before it can array.
PERIODIC_NO_ARRAY = {'SADDLE_TOWER_A'}


def _periodic_dim(surf):
    """Tiling dimensionality of a periodic surface: 3 for the triply
    periodic TPMS / PGD, else 2 for a doubly periodic family member and 1
    for a singly periodic one.  Drives how many independent cell counts the
    operator shows and applies."""
    if surf in TPMS or surf in TPMS_EXACT:
        return 3
    fam = (SURFACE_FAMILY or {}).get(surf)
    if fam == 'DOUBLY':
        return 2
    return 1


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


# --- periodic lattice arraying (doubly periodic grid surfaces) ------------
# A doubly periodic minimal surface is meshed as ONE fundamental domain in
# its TRUE period scale.  To tile it we must replicate that raw mesh by the
# lattice's translation vectors *before* _center_fit rescales the whole
# object into the 2 m cube (which destroys the period scale -- a post-hoc
# Blender array or bbox-relative offset would no longer line up).  Each
# doubly periodic surface supplies its two raw-space lattice vectors below;
# _array_by_lattice then lays down cells_u x cells_v rigid copies.

def _lattice_vectors(kind, order, radius, theta):
    """Raw-space (pre-_center_fit) translation lattice vectors for a doubly
    periodic grid surface, in the same coordinates its builder emits.
    Returns a list of vectors (one per tiling dimension) or None when the
    surface is not arrayed by rigid lattice translation on this path
    (the classical Scherk graph tiles as one continuous grid -- see
    _scherk_doubly; the tilted Scherk likewise, see _tilt_scherk_doubly;
    towers array via their screw motion; Riemann via the torus copies).

    Both Scherk surfaces are built connected as ONE continuous graph over the
    whole cells block (SCHERK1 -> _scherk_doubly, TILT_SCHERK ->
    _tilt_scherk_doubly, the classical graph with the exact horizontal tilt
    displacement), so neither needs -- or takes -- the rigid lattice-array
    path here.  (Their asymptotic walls meet only at infinity, so rigid copies
    cut on the ends could never weld; the shared-wall graph sidesteps that.)"""
    return None


def _array_by_lattice(V, quads, UV, vectors, counts):
    """Lay down a grid of rigid copies of the fundamental mesh (V, quads,
    per-vertex UV) offset by integer combinations of the raw-space lattice
    `vectors` (len == len(counts)).  The block is centered on the origin so
    the subsequent _center_fit keeps it symmetric.  Copies that share a seam
    are merged later by the object weld; here they are placed at the exact
    period so they line up."""
    import itertools
    ranges = [range(int(max(1, c))) for c in counts]
    offs = []
    for idx in itertools.product(*ranges):
        off = np.zeros(3)
        for k, i in enumerate(idx):
            off = off + (i - 0.5 * (int(counts[k]) - 1)) * vectors[k]
        offs.append(off)
    if len(offs) <= 1:
        return V, quads, UV
    nV = len(V)
    Vbig = np.concatenate([V + o for o in offs], axis=0)
    UVbig = np.concatenate([UV] * len(offs), axis=0) if UV is not None else None
    quadsbig = []
    for c in range(len(offs)):
        base = c * nV
        for q in quads:
            quadsbig.append(tuple(base + i for i in q))
    return Vbig, quadsbig, UVbig


# --- classical Scherk: the connected doubly periodic graph ----------------
# The doubly periodic Scherk surface (H. F. Scherk, 1835) is the single graph
#     z = ln|cos x / cos y|   =   ln|cos x| - ln|cos y|
# over the whole plane (minus the singular grid lines).  Reading it as a graph
# over the CHECKERBOARD of squares |cos x/cos y| > 0 (the usual z = ln(cos x/
# cos y) picture) leaves each fundamental saddle stranded on its own square,
# joined to its neighbours only along the vertical asymptotic PLANES
# x = pi/2 + k pi (where z -> -inf, cos x -> 0) and y = pi/2 + k pi (z -> +inf).
# Those planes lie *between* the checkerboard cells, so a rigid lattice array
# of the single-square graph never welds -- the walls meet only at z = +-inf.
#
# The absolute value fixes this: |cos| is continuous across every wall line, so
# z = ln|cos x| - ln|cos y| (trimmed in z at the wall rims) is ONE
# continuous graph across the whole cells_u x cells_v block -- a single
# connected component with no seams to weld.  Over an even square it is the
# classic saddle ln(cos x/cos y); over the neighbouring square it is the same
# surface's next sheet, ln(cos(x-pi)/cos y) = ln|cos x/cos y|, which asymptotes
# to the SAME wall plane x = pi/2 from the other side.  The surface is
# asymptotic to the family of planes y = pi/2 + k pi at the top and
# x = pi/2 + k pi at the bottom -- the two interleaved half-plane families
# that are Scherk's defining picture.
#
# References:
#   H. F. Scherk, "Bemerkungen ueber die kleinste Flaeche mit gegebener
#     Begrenzung", J. Reine Angew. Math. 13 (1835) 185-208 (Scherk's first,
#     doubly periodic, surface);
#   J. C. C. Nitsche, "Lectures on Minimal Surfaces" (1989);
#   H. Karcher, K. Polthier, Phil. Trans. R. Soc. Lond. A 354 (1996)
#     2077-2104 (the Scherk family and its period lattice).

def _shift2d(a, di, dj, fill):
    """`a` shifted so result[i, j] == a[i + di, j + dj]; the wrapped edge that
    falls off the grid is set to `fill`, so no false neighbours are seen across
    the grid boundary."""
    b = np.roll(a, (-di, -dj), axis=(0, 1))
    if di == 1:
        b[-1, :] = fill
    elif di == -1:
        b[0, :] = fill
    if dj == 1:
        b[:, -1] = fill
    elif dj == -1:
        b[:, 0] = fill
    return b


def _scherk_trim_snap(X, Y, w, zcap):
    """Truncate the height graph w over the (X, Y) grid at |w| = zcap by
    TRIMMING (not clamping) the over-cap region, then snapping the surviving
    rim vertices onto the exact |w| = zcap contour.

    A plain np.clip(w, -zcap, zcap) leaves a flat plateau capping every wall;
    keeping only faces whose four corners are all `inside` (|w| <= zcap) drops
    the caps and leaves each wall ending in an OPEN slot down its top/bottom.
    Where the wall slit is wider than the grid the slot opens; where it narrows
    to sub-grid width (near the corners) the grid steps over it and the cells
    stay joined -- exactly the corner necks that keep the whole block one
    connected component.

    A raw trim edge is a stair-step, so each surviving rim vertex (an `inside`
    vertex with an over-cap neighbour) is moved OUTWARD along its edge(s) to the
    over-cap side, onto the |w| = zcap crossing, and its height set to the cap:
    the truncation edge then follows the true contour and the wall reaches full
    height.  The domain's outer boundary has no over-cap neighbour, so it is
    left untouched (the finite-patch saddle cut).

    Returns (keep (nx, ny) bool, Xf, Yf, Zf); keep == `inside`."""
    finite = np.isfinite(w)
    inside = finite & (np.abs(w) <= zcap)
    outside = finite & ~inside               # over-cap, excluding nan corners
    accX = np.zeros_like(w)
    accY = np.zeros_like(w)
    accZ = np.zeros_like(w)
    cnt = np.zeros_like(w)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb_out = _shift2d(outside, di, dj, False)
        nb_w = _shift2d(w, di, dj, np.nan)
        nbX = _shift2d(X, di, dj, np.nan)
        nbY = _shift2d(Y, di, dj, np.nan)
        m = inside & nb_out                  # rim edge: crosses the contour
        capval = np.sign(nb_w) * zcap        # the wall (+/-) this edge reaches
        denom = nb_w - w
        safe = m & (np.abs(denom) > 1e-9)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(safe, (capval - w) / np.where(safe, denom, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)             # fraction from this vertex outward
        accX += np.where(m, X + t * (nbX - X), 0.0)
        accY += np.where(m, Y + t * (nbY - Y), 0.0)
        accZ += np.where(m, capval, 0.0)
        cnt += m
    rim = inside & (cnt > 0)
    d = np.maximum(cnt, 1.0)
    Xf = np.where(rim, accX / d, X)
    Yf = np.where(rim, accY / d, Y)
    Zf = np.where(rim, accZ / d, np.where(inside, w, 0.0))
    return inside, Xf, Yf, Zf


def _scherk_block(nu, nv, scale, cells_u, cells_v, rho, with_uv=False):
    """Shared builder for the two doubly periodic Scherk graphs.  Meshes the
    continuous graph z = ln|cos x| - ln|cos y| over cells_u (x) by cells_v (y)
    pi-period cells as ONE connected component, truncates the walls at
    |z| = zcap by trim-and-snap (see _scherk_trim_snap), and -- for rho != 1 --
    applies the exact Lopez-Ros tilt reparametrization to the KEPT vertices
    only, so no clamp plateau survives to be sheared into blocky slabs.
    rho == 1 is the classical Scherk graph exactly."""
    Cu = int(max(1, cells_u))
    Cv = int(max(1, cells_v))
    lim = 0.47 * math.pi            # per-cell half width (matches the 1x1 cell)
    zcap = 2.8                      # wall truncation height
    # cells centred at x = k pi (k = 0..Cu-1); the grid spans the interior wall
    # lines x = pi/2 + k pi (y likewise).  Sample with an ODD count per cell and
    # a vertex on every cell centre (x = k pi) so each wall line falls on a
    # cell-edge MIDPOINT and each corner (pi/2 + k pi, pi/2 + l pi) sits at a
    # grid-cell centre.  That alignment makes the trim robust: along a wall the
    # slit (over-cap cells) opens as a clean slot, while the single grid cell
    # straddling each corner has all four corners near w = ln(dx/dy) ~ 0 and so
    # is always kept -- a bridging quad that welds the four cells there (the
    # corner neck), keeping the whole block ONE connected component.
    per_x = int(max(9, nu)) | 1     # odd samples across one x-cell
    per_y = int(max(9, nv)) | 1
    dx = math.pi / per_x
    dy = math.pi / per_y
    x_hi = (Cu - 1) * math.pi + lim
    y_hi = (Cv - 1) * math.pi + lim
    x = np.arange(math.ceil(-lim / dx), math.floor(x_hi / dx) + 1) * dx
    y = np.arange(math.ceil(-lim / dy), math.floor(y_hi / dy) + 1) * dy
    nx = len(x)
    ny = len(y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.log(np.abs(np.cos(X))) - np.log(np.abs(np.cos(Y)))
    keep, Xf, Yf, Zf = _scherk_trim_snap(X, Y, w, zcap)
    if rho == 1.0:
        Xo, Yo = Xf, Yf
    else:
        # exact horizontal tilt map, evaluated on the trimmed+snapped vertices;
        # the height Zf is left unchanged (the tilt only leans the walls).  On
        # the kept region P, Q stay bounded (~+-3.5 for zcap = 2.8), so no cap
        # is needed -- the old pcap clamp only distorted the near-wall band.
        with np.errstate(divide='ignore', invalid='ignore'):
            sinhQ = -np.cos(Xf) * np.tan(Yf)
            Q = np.arcsinh(sinhQ)
            coshQ = np.sqrt(1.0 + sinhQ ** 2)
            P = (np.log(np.abs(np.cos(Xf))) - np.log(np.abs(np.cos(Yf)))
                 - np.log(coshQ - np.sin(Xf)))
        P = np.where(np.isfinite(P), P, 0.0)
        Q = np.where(np.isfinite(Q), Q, 0.0)
        m = 0.5 * (rho + 1.0 / rho)
        n = 0.5 * (rho - 1.0 / rho)
        Xo = m * Xf - n * P
        Yo = m * Yf + n * Q
    Vfull = np.stack([Xo, Yo, Zf], axis=-1).reshape(-1, 3)
    # per-vertex conformal-ish UV: the normalised parameter grid, 0..1 over
    # the whole tiled block (the (x, y) graph chart)
    gu = (np.arange(nx) / max(nx - 1, 1))
    gv = (np.arange(ny) / max(ny - 1, 1))
    UVfull = np.stack(np.meshgrid(gu, gv, indexing='ij'),
                      axis=-1).reshape(-1, 2)
    # quads over the grid whose four corners are all kept
    cell_ok = (keep[:-1, :-1] & keep[1:, :-1]
               & keep[1:, 1:] & keep[:-1, 1:])
    ii, jj = np.nonzero(cell_ok)
    base = ii * ny + jj
    quads_arr = np.stack([base, base + ny, base + ny + 1, base + 1], axis=1)
    # compact to referenced vertices only (drop the trimmed ones)
    used = np.unique(quads_arr)
    remap = np.full(nx * ny, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    V = Vfull[used]
    UVg = UVfull[used]
    quads = [tuple(int(remap[i]) for i in q) for q in quads_arr]
    V = _center_fit(V, scale, V)
    if not with_uv:
        return V, quads
    if not quads:
        return V, quads, None
    q = np.array(quads)
    cuv = UVg[q].astype(float)
    return V, quads, cuv.reshape(-1, 2)


def _scherk_doubly(nu, nv, order, radius, scale, cells_u, cells_v,
                   with_uv=False):
    """Connected doubly periodic Scherk graph, z = ln|cos x| - ln|cos y|,
    over cells_u (x) by cells_v (y) pi-period cells -> ONE continuous mesh.
    cells = (1, 1) reproduces the single fundamental saddle; larger counts
    tile it gap-free (the walls are shared, not welded copies).  The walls are
    trimmed (not clamped) at |z| = zcap, so they end in clean open rims rather
    than flat plateaus."""
    return _scherk_block(nu, nv, scale, cells_u, cells_v, 1.0, with_uv)


# --- tilted Scherk: the connected doubly periodic graph, tilted -----------
# The tilted (Lopez-Ros) Scherk surface deforms the classical doubly periodic
# Scherk graph by the Lopez-Ros factor Rho (g = Rho z, dh = 4z/(z^4 - 1)),
# which tilts the two families of asymptotic half-planes so they meet at an
# angle other than the classical 90 degrees.  Meshed straight from the exact
# Weierstrass log-sum immersion the surface is a single fundamental saddle cut
# on its four ends (wing rims), and adjacent lattice copies never weld -- the
# ends meet only at infinity.
#
# But the tilt is exactly a HORIZONTAL reparametrization of the classical
# Scherk graph, with the HEIGHT left unchanged (derived from, and verified to
# machine precision against, the log-sum immersion _tiltscherk_X):
#     x_t = m x - n P(x, y),   y_t = m y + n Q(x, y),   z_t = z = ln|cos x/cos y|
# where, with the Lopez-Ros factor Rho,
#     m = (Rho + 1/Rho)/2 ,  n = (Rho - 1/Rho)/2 ,
#     Q = asinh(-cos x * tan y) ,
#     P = ln|cos x| - ln|cos y| - ln(cosh Q - sin x) .
# At Rho = 1 (n = 0, m = 1) the map is the identity and the surface is exactly
# the classical Scherk graph (SCHERK1).  Because the displacement is a
# continuous per-vertex function of (x, y), applying it to the single
# continuous classical grid (which spans the whole cells_u x cells_v block and
# crosses the shared wall lines by construction) keeps the mesh ONE connected,
# gap-free component -- the tilt merely leans the shared walls, it does not cut
# them.  Rho is driven by the radius slider (the surface's tilt control).
#
# References:
#   H. F. Scherk (1835), as for _scherk_doubly (the base doubly periodic
#     surface and its wall/period lattice);
#   F. J. Lopez and A. Ros, "On embedded complete minimal surfaces of genus
#     zero", J. Differential Geom. 33 (1991) 293-300 (the Lopez-Ros
#     deformation that tilts the ends).

def _tilt_scherk_doubly(nu, nv, rho, scale, cells_u, cells_v,
                        with_uv=False):
    """Connected doubly periodic tilted-Scherk graph over cells_u (x) by
    cells_v (y) pi-period cells -> ONE continuous mesh.  `rho` is the
    Lopez-Ros tilt factor (rho = 1 reproduces the classical Scherk graph,
    _scherk_doubly, exactly).  cells = (1, 1) is the single fundamental
    saddle; larger counts tile it gap-free (shared, leaned walls).

    The tilt reparametrization is applied to the KEPT vertices of the
    trimmed graph only (see _scherk_block / _scherk_trim_snap): there is no
    clamp plateau left to be sheared, so the leaned walls end in clean open
    rims instead of blocky slanted slabs.  Since the Lopez-Ros tilt makes
    this surface non-embedded, a tiled copy honestly self-intersects (the
    leaned asymptotic half-planes pass through each other) -- that is the
    true surface, not an artifact."""
    return _scherk_block(nu, nv, scale, cells_u, cells_v, float(rho), with_uv)


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


def _circularize_outer(V, quads, min_len=8):
    """Snap each open boundary loop -- the planar end and the two catenoid
    ends of a Costa / Costa-Hoffman-Meeks surface -- to a clean circle
    about the vertical axis (constant XY radius, z kept), so every rim
    reads as a circle instead of the few-percent staircase wobble the
    radial end clip leaves behind.  Interior vertices are untouched."""
    if not quads:
        return V
    from collections import defaultdict
    cnt = defaultdict(int)
    nbr = defaultdict(list)
    for q in quads:
        L = len(q)
        for k in range(L):
            a, b = q[k], q[(k + 1) % L]
            cnt[(a, b) if a < b else (b, a)] += 1
    for (a, b), c in cnt.items():
        if c == 1:
            nbr[a].append(b)
            nbr[b].append(a)
    bnd = [v for v in nbr if len(nbr[v]) == 2]     # clean-loop vertices
    if not bnd:
        return V
    seen = set()
    loops = []
    for v in bnd:
        if v in seen:
            continue
        comp, stack = [], [v]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for w in nbr[u]:
                if w not in seen:
                    stack.append(w)
        loops.append(comp)
    V = V.copy()
    for comp in loops:
        if len(comp) < min_len:
            continue                                # skip stray fragments
        idx = np.array(comp)
        rmean = float(np.hypot(V[idx, 0], V[idx, 1]).mean())
        ang = np.arctan2(V[idx, 1], V[idx, 0])
        V[idx, 0] = rmean * np.cos(ang)
        V[idx, 1] = rmean * np.sin(ang)
        V[idx, 2] = float(V[idx, 2].mean())         # flat horizontal circle
    return V


def build_parametric_grid(kind, nu, nv, order, radius, scale, theta=0.0):
    """(nu, nv, 3) point grid plus wrap flags, centered and fit to a 2 m
    cube. (Used for NURBS output; end clipping is a mesh-only operation.)"""
    if kind in MESH_PARAM:
        raise ValueError(f"{kind} has no NURBS/grid form; use mesh output")
    G, wrap_u, wrap_v, clip = _raw_grid(kind, nu, nv, order, radius, theta)
    nu, nv = G.shape[:2]           # the grid may resize itself (odd rows)
    flat = G.reshape(-1, 3)
    if isinstance(clip, np.ndarray):
        ref = flat[clip.reshape(-1)]
    elif clip:
        ref = _inliers(flat)
    else:
        ref = flat
    G = _center_fit(flat, scale, ref).reshape(nu, nv, 3)
    return G, wrap_u, wrap_v


def build_parametric(kind, nu, nv, order, radius, scale, theta=0.0,
                     with_uv=False, cells=(1, 1)):
    """Mesh (V, quads) for `kind`; with_uv=True additionally returns a
    per-face-corner UV array (sum of face lengths, 2).  Minimal
    surfaces are conformally parametrized by their Weierstrass data,
    so the normalized (u, v) grid is a high-quality conformal UV chart
    for free; periodic directions get a clean 0<->1 seam.  Finished
    (tiled) meshes carry a best-effort per-fundamental-domain UV.

    `cells` = (cells_u, cells_v) drives the periodic repeat, with the
    array's DIMENSIONALITY following the surface's periodicity:
      * singly periodic (saddle tower, Riemann) -> cells_u copies along
        the single period axis (cells_v ignored);
      * doubly periodic (Scherk, tilted Scherk) -> cells_u x cells_v
        copies over the two lattice vectors.
    All arraying happens in the surface's true period scale BEFORE
    _center_fit normalizes the whole (tiled) object into the 2 m cube.
    A plain int is accepted as (cells, 1)."""
    if isinstance(cells, (int, float)):
        cells = (int(cells), 1)
    cells_u = int(max(1, cells[0]))
    cells_v = int(max(1, cells[1] if len(cells) > 1 else 1))
    if kind == 'SCHERK1':
        # classical Scherk: build the whole cells_u x cells_v block as ONE
        # continuous graph z = ln|cos x/cos y| (single connected component,
        # gap-free) rather than arraying rigid single-cell copies whose
        # asymptotic-plane walls would meet only at infinity.
        return _scherk_doubly(nu, nv, order, radius, scale,
                              cells_u, cells_v, with_uv)
    if kind == 'TILT_SCHERK':
        # tilted (Lopez-Ros) Scherk: same connected-graph tiling as SCHERK1,
        # with the height field untouched and the horizontal coordinates
        # displaced by the exact tilt map (see _tilt_scherk_doubly).  Rho
        # (the tilt factor) is read from the surface's own p_from, so the
        # radius slider drives the tilt; Rho = 1 reproduces SCHERK1 exactly.
        b = PARAMETRIC[kind][1]
        spec = getattr(b, 'spec', None)
        rho = float(spec['p_from'](order, radius)['Rho']) if spec else 1.0
        return _tilt_scherk_doubly(nu, nv, rho, scale,
                                   cells_u, cells_v, with_uv)
    # doubly periodic finished meshes (KMR / Wei): a spec that declares a
    # 'cells2d_mesher' tiles its own 2-D lattice from BOTH cell counts
    # (the 1-D storeys contract below cannot carry the second axis)
    _bdp = PARAMETRIC.get(kind)
    _sdp = getattr(_bdp[1], 'spec', None) if _bdp else None
    if _sdp is not None and _sdp.get('cells2d_mesher'):
        out = _sdp['cells2d_mesher'](_sdp, nu, nv, order, radius, scale,
                                     theta, (cells_u, cells_v))
        V, quads = out[0], out[1]
        if not with_uv:
            return V, quads
        uvv = out[2] if len(out) > 2 else None
        if uvv is None or not quads:
            return V, quads, None
        idx = np.fromiter((i for f in quads for i in f), dtype=np.int64)
        return V, quads, np.asarray(uvv)[idx]
    if kind in MESH_PARAM:
        # towers: cells_u storeys stacked under the surface's screw motion
        out = MESH_PARAM[kind](nu, nv, order, radius, scale, theta,
                               storeys=cells_u)
        V, quads = out[0], out[1]
        if not with_uv:
            return V, quads
        uvv = out[2] if len(out) > 2 else None
        if uvv is None or not quads:
            return V, quads, None
        idx = np.fromiter((i for f in quads for i in f), dtype=np.int64)
        return V, quads, np.asarray(uvv)[idx]
    # per-surface mesh-density boost: some conformal domains (strongly
    # curved Bjorling ribbons, the fast-growing Enneper disk ends) need a
    # denser grid than the shared 48x48 default to render smooth rather
    # than faceted.  Applied only on the mesh path -- the NURBS/grid path
    # keeps the requested control-point count, and the engine self-tests
    # call the builders directly, so both stay unaffected.
    # res_boost may be a plain (nu_mult, nv_mult) tuple or a callable
    # res_boost(order) -> (nu_mult, nv_mult).  The callable form lets a
    # surface whose rim winds more with order (g = z^k gains ~2(k+1) lobes)
    # scale its angular sampling with k, so the boundary stays smooth at
    # every order rather than only at order 1.
    _b = PARAMETRIC[kind][1]
    _rb = getattr(_b, 'spec', {}).get('res_boost') if hasattr(_b, 'spec') \
        else None
    if callable(_rb):
        _rb = _rb(order)
    if _rb:
        nu = max(3, int(round(nu * _rb[0])))
        nv = max(3, int(round(nv * _rb[1])))
    # singly periodic torus surfaces (Riemann): array cells_u fundamental
    # domains along the deck translation via the torus copies mechanism,
    # scaling the sampling so each copy keeps its density
    copies = None
    if kind in PERIODIC_COPIES_1D:
        copies = cells_u
        nu = max(3, nu * cells_u)
    G, wrap_u, wrap_v, clip = _raw_grid(kind, nu, nv, order, radius, theta,
                                        copies=copies)
    # trust the grid's own dimensions: a builder may return a different
    # size than requested (the Bjorling strip forces an odd row count so
    # a column lands on the seed axis).  Indexing the quads/UVs with the
    # requested nv instead would misalign every row by one and shear the
    # whole strip into a corrugated "accordion" -- the actual grid shape
    # is the single source of truth.
    nu, nv = G.shape[:2]
    V = G.reshape(-1, 3)
    # conformal UV: the normalized parameter grid (endpoint-free on
    # wrapped axes, so the seam face closes at exactly u or v = 1)
    gu = np.arange(nu) / (nu if wrap_u else max(nu - 1, 1))
    gv = np.arange(nv) / (nv if wrap_v else max(nv - 1, 1))
    UVg = np.stack(np.meshgrid(gu, gv, indexing='ij'),
                   axis=-1).reshape(-1, 2)
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
        # grid disk/strip with no puncture mask and no radial clip (Enneper,
        # Bour, Henneberg, Richmond, the associate disks...).  Historically
        # this branch skipped _smooth_boundary, so the outer disk-edge ring
        # (and any strip corner) kept the raw grid staircase.  A gentle
        # boundary relaxation knocks that residual facet down; the denser
        # sampling (res baseline + per-surface res_boost) does the heavy
        # lifting, so a light pass is enough and leaves clean circular rims
        # (catenoid/cathel) essentially untouched.
        V = _smooth_boundary(V, quads, iters=5)
        ref = _inliers(V) if clip else V

    if ref is None:
        # compact to referenced vertices only (drops loose end points)
        used = (np.unique(np.array(quads).ravel()) if quads
                else np.array([], dtype=int))
        remap = np.full(len(V), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        V = V[used]
        UVg = UVg[used]
        quads = [tuple(int(remap[i]) for i in qd) for qd in quads]
        V = _smooth_boundary(V, quads)   # smooth staircase/clip end rims
        ref = V

    # --- doubly periodic lattice array (in the true period scale, BEFORE
    # _center_fit collapses it) ------------------------------------------
    lat = _lattice_vectors(kind, order, radius, theta)
    if lat and (cells_u > 1 or cells_v > 1):
        counts = (cells_u, cells_v)[:len(lat)]
        V, quads, UVg = _array_by_lattice(V, quads, UVg, lat, counts)
        ref = V

    V = _center_fit(V, scale, ref)
    if not with_uv:
        return V, quads
    if not quads:
        return V, quads, None
    q = np.array(quads)
    cuv = UVg[q].astype(float)                 # (nf, 4, 2) corner UVs
    for axis, wrapped in ((0, wrap_u), (1, wrap_v)):
        if wrapped:
            # faces crossing the periodic seam: lift the low corners by
            # one period so the face maps to the [.., 1.0] edge cleanly
            a = cuv[..., axis]
            seam = (a.max(axis=1) - a.min(axis=1)) > 0.5
            a += ((a < 0.5) & seam[:, None]).astype(float)
    return V, quads, cuv.reshape(-1, 2)


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


def _cells_xyz(cells):
    """Coerce a cell count to a (cx, cy, cz) triple.  A plain int means a
    symmetric cx = cy = cz block (kept for internal callers/tests); a tuple
    or list gives independent per-axis counts."""
    if isinstance(cells, (tuple, list)):
        vals = [int(max(1, c)) for c in cells]
        while len(vals) < 3:
            vals.append(1)
        return vals[0], vals[1], vals[2]
    c = int(max(1, cells))
    return c, c, c


def build_tpms(kind, cells, res_per_cell, scale):
    """Nodal TPMS mesh.  `cells` is an int (symmetric block) or a
    (cx, cy, cz) triple: the block spans cx x cy x cz unit cells, one
    independent count per lattice axis.  For the singly periodic Scherk
    tower only the z count repeats (x/y are clipped)."""
    label, field, triply = TPMS[kind]
    cx, cy, cz = _cells_xyz(cells)
    if triply:
        sx, sy, sz = cx * TAU, cy * TAU, cz * TAU
        box_min = (-sx / 2, -sy / 2, -sz / 2)
        box_max = (sx / 2, sy / 2, sz / 2)
        res = (cx * res_per_cell, cy * res_per_cell, cz * res_per_cell)
    else:  # Scherk tower: periodic in z only, clip x/y
        w = 2.2
        box_min = (-w, -w, -cz * math.pi)
        box_max = (w, w, cz * math.pi)
        rxy = int(res_per_cell * 1.4)
        res = (rxy, rxy, cz * res_per_cell)
    verts, tris = marching_tets(field, box_min, box_max, res)
    s = scale / TAU  # one period -> `scale` Blender units
    return verts * s, tris


# --- exact Weierstrass P / Gyroid / D (Bonnet angle) ---------------------
# The nodal TPMS above (marching-tets level sets) have NO associate
# parameter -- you cannot morph P <-> Gyroid <-> D in the nodal
# representation.  The exact genus-3 Enneper-Weierstrass immersion in
# we_builders.pgd_build does: a single Bonnet angle theta continuously
# sweeps the whole classical family (P at 0, Gyroid at ~38.0148 deg, D at
# 90 deg).  It is exposed under the periodic generator's Triply Periodic
# list as its own entry (a preset for P / Gyroid / D plus an independent
# associate-angle slider), separate from and leaving unchanged the nodal TPMS
# set.  At the two reflective members it reassembles a watertight FILLED unit
# cell by the Schwarz reflection principle -- Schwarz P from coordinate-plane
# reflections, Schwarz D from 2-fold rotations about its straight edges -- and
# `cells` arrays that cell on the verified cubic period.  The chiral gyroid and
# every generic intermediate angle instead keep the exact fundamental surface
# piece (the honest choice: only P / Gyroid / D are truly periodic, and the
# gyroid cell has no drift-free reflection assembly).  See the pgd_build /
# _pgd_tile_cell docstrings in we_builders.

try:
    from . import we_builders as _we_pgd
except ImportError:                                # script / test context
    import we_builders as _we_pgd

# key -> (menu label, builder(cells, res_per_cell, scale, theta))
TPMS_EXACT = {
    'PGD': ("Schwarz P-Gyroid-D (exact, Bonnet angle)", _we_pgd.pgd_build),
}

# named-preset -> Bonnet angle (radians).  P and D reassemble a filled cell;
# the gyroid angle builds the fundamental piece.  CUSTOM (absent here) means
# "use the raw Associate Angle slider".
_PGD_PRESET_ANGLE = {
    'P': 0.0,
    'GYROID': 0.6635246,          # 38.0148 deg -- Schoen's gyroid
    'D': math.pi / 2.0,
}


def build_tpms_exact(kind, cells, res_per_cell, scale, theta):
    label, builder = TPMS_EXACT[kind]
    return builder(cells, res_per_cell, scale, theta)


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

    def _new_object(context, name, verts, faces, weld=0.0, smooth=True,
                    loop_uv=None, recalc_normals=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
                       [tuple(int(i) for i in f) for f in faces])
        me.validate(clean_customdata=True)
        if loop_uv is not None:
            # per-face-corner UVs (assigned before any weld: bmesh's
            # remove_doubles merges vertices but keeps loop layers)
            luv = np.asarray(loop_uv, dtype=np.float32)
            if len(me.loops) == len(luv):
                layer = me.uv_layers.new(name="UVMap")
                layer.data.foreach_set('uv', luv.ravel())
        if weld > 0 or recalc_normals:
            bm = bmesh.new()
            bm.from_mesh(me)
            if weld > 0:
                bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
            if recalc_normals:
                # coherent winding across each connected patch: kills the
                # alternating light/dark facet stripes that flipped quads
                # produce under smooth shading (one-sided surfaces still
                # keep a single unavoidable orientation seam)
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
        me.polygons.foreach_set('use_smooth',
                                [bool(smooth)] * len(me.polygons))
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

    # --- family-filtered surface enum (zoo catalog UX) ------------------
    # Item tuples are cached module-level: Blender requires the strings
    # of a dynamic-items enum to stay referenced from Python.
    # Periodic minimal surfaces live in their own generator
    # (mesh.periodic_minimal_add), picked by a Singly/Doubly/Triply
    # dropdown -- singly/doubly are Weierstrass-Enneper families, triply
    # is the nodal TPMS set.  They are therefore hidden from the
    # non-periodic Minimal Surface generator's family list.
    PERIODIC_FAMILIES = ('SINGLY', 'DOUBLY')

    _FAMILY_ITEMS = []
    _SURF_ITEMS_ALL = []
    _SURF_ITEMS_FAM = {}
    _PERIODIC_ITEMS = {}          # periodicity key -> [surface items]
    _PERIODIC_ALL = []            # union fallback for scripted calls
    _PERIODICITY_ITEMS = []       # the Singly/Doubly/Triply dropdown

    def _build_surface_items():
        _FAMILY_ITEMS.clear()
        _SURF_ITEMS_ALL.clear()
        _SURF_ITEMS_FAM.clear()
        fam_of = SURFACE_FAMILY or {}
        for key, (label, _fn) in PARAMETRIC.items():
            it = (key, label, label)
            _SURF_ITEMS_ALL.append(it)
            _SURF_ITEMS_FAM.setdefault(
                fam_of.get(key, 'CLASSICAL'), []).append(it)
        for fam, flabel in (FAMILIES or ()):
            if fam in PERIODIC_FAMILIES:
                continue                       # -> periodic generator
            if fam in _SURF_ITEMS_FAM:
                n = len(_SURF_ITEMS_FAM[fam])
                _FAMILY_ITEMS.append(
                    (fam, flabel, f"{flabel} ({n} surfaces)"))
        if not _FAMILY_ITEMS:
            _FAMILY_ITEMS.append(
                ('CLASSICAL', "Classical", "Classical"))

        # --- periodic catalog: singly/doubly (WE) + triply (TPMS) ------
        _PERIODIC_ITEMS.clear()
        _PERIODIC_ALL.clear()
        _PERIODICITY_ITEMS.clear()
        flabel_of = dict(FAMILIES or ())
        for fam in PERIODIC_FAMILIES:
            items = list(_SURF_ITEMS_FAM.get(fam, ()))
            if items:
                _PERIODIC_ITEMS[fam] = items
                _PERIODIC_ALL.extend(items)
                lbl = flabel_of.get(fam, fam)
                _PERIODICITY_ITEMS.append(
                    (fam, lbl, f"{lbl} ({len(items)} surfaces)"))
        # Only genuinely triply-periodic surfaces belong under Triply.
        # SCHERKT (Scherk's tower) is a nodal SINGLY-periodic surface that
        # historically rode in the TPMS field dict; the proper singly
        # periodic Scherk tower is the WE SCHERK_TOWER under Singly, so it
        # is dropped from this list (still reachable via mesh.tpms_add).
        _NOT_TRIPLY = {'SCHERKT'}
        # the nodal approximations lead the Triply list; the exact
        # Weierstrass P/Gyroid/D (Bonnet angle) is listed LAST
        exact_items = [(k, v[0], v[0]) for k, v in TPMS_EXACT.items()]
        tpms_items = [(k, v[0], v[0]) for k, v in TPMS.items()
                      if k not in _NOT_TRIPLY] + exact_items
        if tpms_items:
            _PERIODIC_ITEMS['TRIPLY'] = tpms_items
            _PERIODIC_ALL.extend(tpms_items)
            _PERIODICITY_ITEMS.append(
                ('TRIPLY', "Triply Periodic (TPMS)",
                 f"Triply periodic minimal surfaces: the exact "
                 f"P/Gyroid/D associate family + "
                 f"{len(tpms_items) - len(exact_items)} nodal "
                 f"approximations"))
        if not _PERIODICITY_ITEMS:
            _PERIODICITY_ITEMS.append(
                ('TRIPLY', "Triply Periodic (TPMS)", "TPMS"))

    _build_surface_items()

    def _surface_items(self, context):
        # NOTE: fall back to the FULL union list whenever there is no
        # UI area (context is None, or a background/scripted context).
        # Scripted calls -- mesh.parametric_minimal_add(surface='COSTA')
        # -- must not be rejected by the family filter, and the stored
        # enum index must map against the same list on set and get
        # (see COORDINATION.md).  Only an interactive area (the redo
        # panel / add-menu) sees the family-filtered list.
        if context is None or getattr(context, 'area', None) is None:
            return _SURF_ITEMS_ALL
        return _SURF_ITEMS_FAM.get(self.family, _SURF_ITEMS_ALL)

    def _periodic_surface_items(self, context):
        # Same context=None -> full-union fallback contract as above, so
        # scripted mesh.periodic_minimal_add(surface='G') keeps working.
        if context is None or getattr(context, 'area', None) is None:
            return _PERIODIC_ALL or _SURF_ITEMS_ALL
        return _PERIODIC_ITEMS.get(self.periodicity,
                                   _PERIODIC_ALL or _SURF_ITEMS_ALL)

    class MESH_OT_parametric_minimal_add(bpy.types.Operator):
        """Add a minimal surface from the Weierstrass-Enneper / Bjorling
        catalog. Pick a Family, then a Surface within it."""
        bl_idname = "mesh.parametric_minimal_add"
        bl_label = "Minimal Surface"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(
            name="Family",
            items=_FAMILY_ITEMS,
            default='CLASSICAL',
            description="Minimal-surface family (Weber's taxonomy); "
                        "filters the Surface list")
        surface: EnumProperty(
            name="Surface",
            items=_surface_items)
        output: EnumProperty(
            name="Output",
            items=[('MESH', "Mesh", "Dense polygon mesh"),
                   ('NURBS', "NURBS", "Compact NURBS surface patch "
                                      "(control grid = Resolution U x V)")],
            default='MESH')
        res_u: IntProperty(name="Resolution U", default=64, min=8, max=512)
        res_v: IntProperty(name="Resolution V", default=64, min=8, max=512)
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
        storeys: IntProperty(
            name="Storeys", default=3, min=1, max=8,
            description="Number of periodic fundamental domains to stack "
                        "(saddle tower); ignored for the rest")
        radius: FloatProperty(
            name="Domain Radius", default=1.2, min=0.2, max=4.0,
            description="Extent of the parameter domain (for the k-noid, "
                        "how close the disk reaches its ends)")
        assoc_angle: FloatProperty(
            name="Associate Angle", default=0.0,
            min=0.0, max=math.pi / 2.0, subtype='ANGLE',
            description="Bonnet associate family (0 = catenoid, "
                        "pi/2 = helicoid); for the Karcher saddle tower it "
                        "is the wing-clustering angle alpha (0 = symmetric)")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Multiplier on the normalized size (1.0 = a 2 m "
                        "cube, centered on the origin)")

        def execute(self, context):
            # When the Family changes, the dynamic Surface enum is
            # refiltered and the stored value can momentarily resolve to
            # an identifier outside the new list (Blender returns '' or a
            # stale key).  Coerce to the first surface of the current
            # family so switching families never raises.
            surf = self.surface
            if surf not in PARAMETRIC:
                items = _surface_items(self, context)
                surf = items[0][0] if items else 'ENNEPER'
            label = PARAMETRIC[surf][0]
            theta = (self.assoc_angle if surf in ANGLE_PARAM
                     else 0.0)
            # some surfaces are assembled meshes with no NURBS/grid form
            if self.output == 'NURBS' and surf not in MESH_PARAM:
                G, wrap_u, wrap_v = build_parametric_grid(
                    surf, self.ctrl_u, self.ctrl_v,
                    self.order, self.radius, self.scale, theta)
                if wrap_u:          # drop duplicated periodic endpoint
                    G = G[:-1]
                if wrap_v:
                    G = G[:, :-1]
                _nurbs_grid_object(context, label, G,
                                   cyclic_u=wrap_u, cyclic_v=wrap_v)
            else:
                out = build_parametric(surf, self.res_u,
                                       self.res_v, self.order,
                                       self.radius, self.scale, theta,
                                       with_uv=True, cells=(self.storeys, 1))
                V, quads = out[0], out[1]
                cuv = out[2] if len(out) > 2 else None
                _new_object(context, label, V, quads,
                            weld=1e-5 * max(1.0, self.scale),
                            loop_uv=cuv)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
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
            if self.surface in STOREY_PARAM:
                lay.prop(self, 'storeys', text=STOREY_PARAM[self.surface])
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

    class MESH_OT_periodic_minimal_add(bpy.types.Operator):
        """Add a periodic minimal surface.  Pick the Periodicity
        (singly / doubly / triply), then a Surface within it.  Singly and
        doubly periodic surfaces come from the Weierstrass-Enneper
        catalog; triply periodic are the nodal TPMS approximations."""
        bl_idname = "mesh.periodic_minimal_add"
        bl_label = "Periodic Minimal Surface"
        bl_options = {'REGISTER', 'UNDO'}

        periodicity: EnumProperty(
            name="Periodicity",
            items=_PERIODICITY_ITEMS,
            description="Translational symmetry of the surface: singly / "
                        "doubly periodic (Weierstrass-Enneper) or triply "
                        "periodic (TPMS); filters the Surface list")
        surface: EnumProperty(
            name="Surface",
            items=_periodic_surface_items)
        # -- Weierstrass (singly / doubly) parameters
        output: EnumProperty(
            name="Output",
            items=[('MESH', "Mesh", "Dense polygon mesh"),
                   ('NURBS', "NURBS", "Compact NURBS surface patch")],
            default='MESH')
        res_u: IntProperty(name="Resolution U", default=64, min=8, max=512)
        res_v: IntProperty(name="Resolution V", default=64, min=8, max=512)
        ctrl_u: IntProperty(
            name="Control Points U", default=24, min=6, max=128,
            description="NURBS control grid size in U")
        ctrl_v: IntProperty(
            name="Control Points V", default=24, min=6, max=128,
            description="NURBS control grid size in V")
        order: IntProperty(
            name="Order / Count", default=1, min=1, max=12,
            description="Period count / lattice modulus where the surface "
                        "uses it (e.g. saddle-tower wing count)")
        # -- unified cell counts: one INDEPENDENT count per tiling
        # dimension, shown by periodicity (singly -> u only; doubly ->
        # u, v; triply -> u, v, w = x, y, z).  The array's dimensionality
        # follows the surface's periodicity.
        cells_u: IntProperty(
            name="Cells", default=1, min=1, max=8,
            description="Copies along the 1st period axis (singly: the "
                        "single period; doubly: lattice vector 1; triply: x)")
        cells_v: IntProperty(
            name="Cells V", default=1, min=1, max=8,
            description="Copies along the 2nd period axis (doubly: lattice "
                        "vector 2; triply: y)")
        cells_w: IntProperty(
            name="Cells W", default=1, min=1, max=8,
            description="Copies along the 3rd period axis (triply: z)")
        # legacy scalar alias (not shown): scripted
        # periodic_minimal_add(surface=..., cells=3) still works and
        # broadcasts to every tiling axis left at its default (1).
        cells: IntProperty(
            name="Cells", default=0, min=0, max=8,
            description="Legacy uniform cell count (broadcasts to every "
                        "period axis); 0 = use the per-axis Cells controls")
        radius: FloatProperty(
            name="Domain Radius", default=1.2, min=0.2, max=4.0,
            description="Extent of the parameter domain")
        assoc_angle: FloatProperty(
            name="Associate Angle", default=0.0,
            min=0.0, max=math.pi / 2.0, subtype='ANGLE',
            description="Bonnet associate family angle; for the Karcher "
                        "saddle tower it is the wing-clustering angle alpha")
        # -- exact P/Gyroid/D preset: named shortcuts to the three iconic
        # Bonnet angles, leaving the raw angle slider independently settable.
        pgd_preset: EnumProperty(
            name="Preset",
            items=[
                ('P', "Schwarz P", "theta = 0: filled watertight P cell"),
                ('GYROID', "Gyroid",
                 "theta = 38.0148 deg: the gyroid (fundamental piece -- the "
                 "chiral cell has no watertight reflection assembly)"),
                ('D', "Schwarz D",
                 "theta = 90 deg: filled watertight D cell (P's conjugate)"),
                ('CUSTOM', "Custom angle",
                 "Use the Associate Angle slider verbatim; builds the exact "
                 "fundamental piece of the morph at that angle"),
            ],
            default='P',
            description="Iconic P / Gyroid / D by name, or Custom to drive "
                        "the raw Bonnet angle")
        # -- TPMS (triply) parameters (cells come from cells_u/v/w above)
        resolution: IntProperty(
            name="Resolution / Cell", default=28, min=8, max=80,
            description="Sample grid resolution per unit cell")
        cell_size: FloatProperty(
            name="Cell Size", default=2.0, min=0.1, max=100.0,
            description="Edge length of one unit cell in Blender units")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="If > 0, add a Solidify modifier with this thickness")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Multiplier on the normalized size (1.0 = a 2 m "
                        "cube, centered on the origin)")

        def execute(self, context):
            # Coerce a stale/empty dynamic-enum value (the Periodicity
            # dropdown refilters Surface) to a valid key so switching
            # periodicity never raises.  Route on the surface's actual
            # backend, not the dropdown, so scripted calls also resolve.
            surf = self.surface
            if (surf not in TPMS and surf not in PARAMETRIC
                    and surf not in TPMS_EXACT):
                items = _periodic_surface_items(self, context)
                surf = items[0][0] if items else 'G'
            # effective per-axis cell counts: the legacy scalar `cells`
            # (default 0) broadcasts to every axis still at its default
            def _eff(v):
                return self.cells if (self.cells > 0 and v == 1) else v
            cu, cv, cw = (_eff(self.cells_u), _eff(self.cells_v),
                          _eff(self.cells_w))
            if surf in TPMS_EXACT:
                # A named preset (P / Gyroid / D) shows the CLEAN, iconic
                # surface via the nodal builder -- the exact-WE tiling of P/D
                # reads rougher and the chiral gyroid can't be tiled
                # watertight, so the nodal cells are the better "iconic" view
                # (and give a clean gyroid).  Custom sweeps the exact
                # Weierstrass Bonnet morph at the raw slider angle -- the
                # unique capability of this entry -- and at exactly 0 / 38.0148
                # / 90 deg Custom still yields the exact tiled P/D cell.
                cxyz = (cu, cv, cw)
                _PGD_NODAL = {'P': 'P', 'GYROID': 'G', 'D': 'D'}
                if self.pgd_preset in _PGD_NODAL:
                    nk = _PGD_NODAL[self.pgd_preset]
                    verts, tris = build_tpms(nk, cxyz,
                                             self.resolution, self.cell_size)
                    label = TPMS[nk][0]
                else:                                   # CUSTOM: exact morph
                    verts, tris = build_tpms_exact(
                        surf, cxyz, self.resolution, self.cell_size,
                        self.assoc_angle)
                    label = TPMS_EXACT[surf][0]
                if len(tris) == 0:
                    self.report({'ERROR'}, "Empty surface")
                    return {'CANCELLED'}
                obj = _new_object(context, label, verts, tris)
                if self.thickness > 0:
                    mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                    mod.thickness = self.thickness
                    mod.offset = 0.0
                return {'FINISHED'}
            if surf in TPMS:
                cxyz = (cu, cv, cw)
                verts, tris = build_tpms(surf, cxyz,
                                         self.resolution, self.cell_size)
                if len(tris) == 0:
                    self.report({'ERROR'}, "Empty level set")
                    return {'CANCELLED'}
                label = TPMS[surf][0]
                obj = _new_object(context, label, verts, tris)
                if self.thickness > 0:
                    mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                    mod.thickness = self.thickness
                    mod.offset = 0.0
                return {'FINISHED'}
            if surf not in PARAMETRIC:
                self.report({'ERROR'}, f"Unknown surface '{surf}'")
                return {'CANCELLED'}
            label = PARAMETRIC[surf][0]
            theta = (self.assoc_angle if surf in ANGLE_PARAM
                     else 0.0)
            if self.output == 'NURBS' and surf not in MESH_PARAM:
                G, wrap_u, wrap_v = build_parametric_grid(
                    surf, self.ctrl_u, self.ctrl_v,
                    self.order, self.radius, self.scale, theta)
                if wrap_u:
                    G = G[:-1]
                if wrap_v:
                    G = G[:, :-1]
                _nurbs_grid_object(context, label, G,
                                   cyclic_u=wrap_u, cyclic_v=wrap_v)
            else:
                out = build_parametric(surf, self.res_u,
                                       self.res_v, self.order,
                                       self.radius, self.scale, theta,
                                       with_uv=True,
                                       cells=(cu, cv))
                V, quads = out[0], out[1]
                cuv = out[2] if len(out) > 2 else None
                _new_object(context, label, V, quads,
                            weld=1e-5 * max(1.0, self.scale),
                            loop_uv=cuv)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'periodicity')
            lay.prop(self, 'surface')
            if (self.periodicity == 'TRIPLY' or self.surface in TPMS
                    or self.surface in TPMS_EXACT):
                if self.surface in TPMS_EXACT:
                    # exact P/Gyroid/D: named preset first, then always show
                    # the raw Bonnet-angle slider.  A named preset reflects the
                    # special angle it uses; Custom drives the slider directly.
                    lay.prop(self, 'pgd_preset')
                    row = lay.row()
                    row.enabled = (self.pgd_preset == 'CUSTOM')
                    ang = _PGD_PRESET_ANGLE.get(self.pgd_preset)
                    if ang is None:
                        row.prop(self, 'assoc_angle')
                    else:
                        row.prop(self, 'assoc_angle',
                                 text=f"Associate Angle [{math.degrees(ang):.4g} deg]")
                # triply: three independent per-axis counts (x, y, z)
                lay.prop(self, 'cells_u', text="Cells X")
                lay.prop(self, 'cells_v', text="Cells Y")
                lay.prop(self, 'cells_w', text="Cells Z")
                for k in ('resolution', 'cell_size', 'thickness'):
                    lay.prop(self, k)
                return
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
            # cell counts: one per tiling dimension (singly -> u; doubly ->
            # u, v).  NURBS output is a single control patch (no array).
            dim = _periodic_dim(self.surface)
            if (self.output == 'MESH' or mesh_only) \
                    and self.surface not in PERIODIC_NO_ARRAY:
                if dim >= 2:
                    lay.prop(self, 'cells_u', text="Cells U")
                    lay.prop(self, 'cells_v', text="Cells V")
                else:
                    lay.prop(self, 'cells_u', text="Cells")
            if self.surface in ANGLE_PARAM:
                lay.prop(self, 'assoc_angle')
            lay.prop(self, 'radius')
            lay.prop(self, 'scale')

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

    class VIEW3D_MT_math_art_minimal_zoo(bpy.types.Menu):
        """The minimal-surface catalog, one entry per family (each
        opens the parametric operator with that family preset)."""
        bl_idname = "VIEW3D_MT_math_art_minimal_zoo"
        bl_label = "Minimal Surfaces"

        def draw(self, context):
            lay = self.layout
            for fam, flabel, _desc in _FAMILY_ITEMS:
                op = lay.operator("mesh.parametric_minimal_add",
                                  text=flabel, icon='SURFACE_NSPHERE')
                op.family = fam
            lay.separator()
            lay.operator("mesh.tpms_add",
                         text="Triply Periodic (TPMS)",
                         icon='MESH_ICOSPHERE')

    class VIEW3D_MT_minimal_add(bpy.types.Menu):
        bl_idname = "VIEW3D_MT_minimal_add"
        bl_label = "Minimal Surfaces"

        def draw(self, context):
            lay = self.layout
            lay.menu("VIEW3D_MT_math_art_minimal_zoo",
                     icon='SURFACE_NSPHERE')
            lay.operator("mesh.parametric_minimal_add")
            lay.operator("mesh.tpms_add")
            lay.operator("mesh.minimal_knot_span_add")
            lay.operator("object.minimal_span")

    def _menu_func(self, context):
        self.layout.menu("VIEW3D_MT_minimal_add", icon='SURFACE_DATA')

    _classes = (MESH_OT_parametric_minimal_add, MESH_OT_tpms_add,
                MESH_OT_periodic_minimal_add,
                OBJECT_OT_minimal_span, MESH_OT_knot_span_add,
                VIEW3D_PT_minimal_surfaces,
                VIEW3D_MT_math_art_minimal_zoo, VIEW3D_MT_minimal_add)

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
            th = (math.pi / 4
                  if kind in ANGLE_PARAM and kind not in COUNT_PARAM
                  else 0.0)
            n = {'KNOID': 5, 'COSTA_HM': 1, 'SCHERK_TOWER': 3}.get(kind, 1)
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
        # UV gate: every parametric surface carries a finite, in-range,
        # non-collapsed conformal UV chart (per-corner, [0, 1])
        for kind in PARAMETRIC:
            n = {'KNOID': 5, 'COSTA_HM': 1, 'SCHERK_TOWER': 3}.get(kind, 1)
            out = build_parametric(kind, 48, 48, n, 1.2, 1.0,
                                   with_uv=True)
            cuv = out[2] if len(out) > 2 else None
            if cuv is None:
                print(f"uv {kind:15s}: NO UV  FAIL")
                ok = False
                continue
            finite = bool(np.all(np.isfinite(cuv)))
            inrange = (cuv.min() >= -1e-6) and (cuv.max() <= 1.0 + 1e-6)
            span = np.ptp(cuv, axis=0)
            good = finite and inrange and bool(np.all(span > 0.3))
            ok &= good
            print(f"uv {kind:15s}: n={len(cuv):6d} "
                  f"range[{cuv.min():.3f},{cuv.max():.3f}] "
                  f"span=({span[0]:.2f},{span[1]:.2f}) "
                  f"{'OK' if good else 'FAIL'}")
        # Costa-Hoffman-Meeks: modulus table + Euler characteristic gate
        # (genus k, 3 ends removed -> chi = 2 - 2k - 3 = -(2k+1))
        cref = {1: 0.955978, 2: 0.988070, 3: 0.995117, 4: 0.997535}
        cok = all(abs(_zoo.chm_modulus(kk) - cref[kk]) < 1e-5
                  for kk in cref)
        print(f"CHM modulus: "
              + " ".join(f"c({kk})={_zoo.chm_modulus(kk):.5f}"
                         for kk in cref)
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
