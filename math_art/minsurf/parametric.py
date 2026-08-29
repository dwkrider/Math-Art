# Classic parametric minimal surfaces and the meshing pipeline.
#
# Part of the Math Art minimal-surface engine (`math_art/minsurf/`), split
# out of the former single-file `minimal_surface_toolkit.py`.  Numpy only --
# no `bpy` -- so the whole engine imports and self-tests headlessly; the
# registered Blender operators stay in the flat `minimal_surface_toolkit.py`
# front-end.
#
# The classic parametric minimal surfaces (Enneper and higher orders,
# catenoid, helicoid, Henneberg, Catalan, Bour, Richmond, Scherk's doubly
# periodic graph, Costa, Chen-Gackstatter) and the registries that the
# catalog in `zoo.py` extends, plus the meshing pipeline `build_parametric`.
#
# References:
#   K. Weierstrass (1866); A. Enneper (1864); the catenoid (Euler, 1744)
#       shown minimal by J. B. C. Meusnier (1776).
#   C. J. Costa (1982); embeddedness by D. Hoffman and W. H. Meeks III
#       (1985).  Chen-Gackstatter: C. C. Chen and F. Gackstatter (1982).
#   H. F. Scherk (1835).  Classical gallery presentation after Juergen
#       Meier (3d-meier.de, tut25).

import math
import numpy as np

try:
    from .. import geom_cache as _geom_cache
except ImportError:  # flat import outside the package
    import geom_cache as _geom_cache

TAU = 2.0 * math.pi

from . import weierstrass as _we
from .domain import (_center_fit, _circularize_outer, _inliers,
                     _largest_component, _puncture_mask, _smooth_boundary,
                     _torus_grid, area_cov, equal_area_resample,
                     mesh_area_cov)

# How finely the domain is sampled before equal-area resampling measures
# the area element on it.  The measurement is only as good as the grid it
# is taken on, so this is deliberately well above any usable output
# resolution; `build_parametric` also honours 4x the request, whichever
# is larger, and caps at that.
_EQ_AREA_FINE = 256

# (cov_before, cov_after) of the last equal-area resample, so the
# operator can report what it actually achieved instead of claiming
# success blind.  None when the last build did not resample.
LAST_EQ_AREA_COV = None
from .elliptic import _SQUARE, _Lattice
from .tpms import TPMS, TPMS_EXACT


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


def _costa_xyz(U, V):
    """Costa immersion (Gray/Nylander closed form) at torus coordinates
    (U, V); vectorized, poles at the three end punctures come out
    non-finite and must be masked by the caller."""
    L = _SQUARE
    e1 = L.wp(0.5).real
    z = U + 1j * V
    with np.errstate(divide='ignore', invalid='ignore'):
        ze, z1, z3 = L.zeta(z), L.zeta(z - 0.5), L.zeta(z - 0.5j)
        P = L.wp(z)
        a = math.pi / (2.0 * e1)
        x = 0.5 * np.real(-ze + math.pi * U + a * (z1 - z3))
        y = 0.5 * np.real(-1j * ze + math.pi * V
                          - a * (1j * z1 - 1j * z3))
        zc = (math.sqrt(2.0 * math.pi) / 4.0) * np.log(
            np.abs((P - e1) / (P + e1)))
    return x, y, zc


def _wrap_half(d):
    """Signed toroidal offset in (-1/2, 1/2]."""
    return ((d + 0.5) % 1.0) - 0.5


def _costa(nu, nv, order, radius, theta=0.0):
    """Costa's minimal surface -- genus 1, three ends, on the square torus.
    Gray/Nylander closed form (constant offsets dropped; re-centered by the
    mesher). Meshed periodically with the planar end (0,0) and the two
    catenoid ends (1/2,0), (0,1/2) removed. `order`/`theta` unused;
    `radius` scales the end-rim disk size (smaller -> ends reach further).

    End rims are cut on the ANALYTIC parameter circles, not on the raw
    grid staircase: every kept node bordering a puncture is pulled (in
    parameter space, along its ray from the end) onto the exact circle
    and the immersion re-evaluated there, so each rim lies on the true
    surface along a smooth analytic curve.  This kills the old rims'
    grid-staircase wobble outright -- the visible 12-lobe "ripple" of
    the planar rim was purely the staircase cut sampled through the
    flaring end's large cells (the analytic circle's image is round to
    ~0.2%), and the catenoid rims land exactly on their flat (z-spread
    ~0.002) closed-form cut curves.  Those catenoid cut curves are
    genuinely ~25% oval in image radius at this depth: the end's 2-fold
    deviation from its asymptotic catenoid decays like 1/r^2, and no
    rounder on-surface cut exists here (the iso-z contour is MORE oval,
    and the constant-image-radius curve trades the ovality for a large
    height wave).  The presentation fix for that -- the same one the
    Costa-Hoffman-Meeks assembly uses -- is the mesher's final
    _circularize_outer snap, requested via spec['circularize_ends']."""
    U, V = _torus_grid(nu, nv)
    x, y, zc = _costa_xyz(U, V)
    s = max(radius / 1.2, 0.4)
    ends = [(0.0, 0.0, 0.20 / s),      # planar end
            (0.5, 0.0, 0.11 / s),      # catenoid end
            (0.0, 0.5, 0.11 / s)]      # catenoid end
    mask = _puncture_mask(U, V, ends)
    # rim snap: kept nodes with a masked 8-neighbour move onto the circle
    inv = ~mask
    nb = np.zeros_like(inv)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            nb |= np.roll(np.roll(inv, di, axis=0), dj, axis=1)
    rim = mask & nb
    if rim.any():
        band = 2.5 / max(min(nu, nv), 1)
        for cu, cv, rho0 in ends:
            du = _wrap_half(U - cu)
            dv = _wrap_half(V - cv)
            d = np.hypot(du, dv)
            sel = rim & (d <= rho0 + band) & (d > 0)
            if not sel.any():
                continue
            f = rho0 / d[sel]
            Us = cu + f * du[sel]
            Vs = cv + f * dv[sel]
            xs, ys, zs = _costa_xyz(Us, Vs)
            x[sel], y[sel], zc[sel] = xs, ys, zs
    return x, y, zc, True, True, mask


_costa.spec = {'circularize_ends': True}


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
# zoo.py, built by the generic Weierstrass-Enneper
# engine in weierstrass.py; they are wired into PARAMETRIC below by
# the zoo's register() call, together with the rest of the catalog
# (saddle towers, Bjorling strips, Meeks Mobius, Riemann's example...).



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
# surfaces that are a single fixed family member: the order/count slider
# has no effect, so the operator hides it (a visible dead control reads
# as a bug)
ORDERLESS = set()
# surfaces whose order/count slider only supports a sub-range of the
# shared property: surface key -> (lo, hi) in SLIDER units.  The
# operator snaps the slider into this range, so the control always
# shows the value actually built -- a slider that silently substitutes
# a different value is worse than one that is absent or correctly
# bounded (the k-noid-with-Enneper-ends "Ends (k)" slider used to
# build k = 3 for order 1, 2 and 3 without saying so)
ORDER_RANGE = {}

# Wire in the catalog (KNOID, COSTA_HM and the rest of the zoo): the
# rows in zoo.py are built by the generic engine in weierstrass.py and
# registered into the four dicts above.  Resilient: if the zoo cannot be
# imported, the classical core still works.
#
# This is the one edge in the package that points "forward": zoo imports
# weierstrass, which imports domain -- none of which import parametric at
# module level, so the graph stays acyclic.  (zoo reaches back for
# build_parametric only inside its own _selftest, where this module is
# long since initialized.)
try:
    from . import zoo as _zoo
    _zoo.register(PARAMETRIC, MESH_PARAM, COUNT_PARAM, ANGLE_PARAM,
                  STOREY_PARAM, ORDERLESS, ORDER_RANGE)
    SURFACE_FAMILY = _zoo.SURFACE_FAMILY
    FAMILIES = _zoo.FAMILIES
except Exception as _e:                        # WIP catalog: skip
    print(f"minsurf.parametric: zoo unavailable: {_e}")
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
    # stale-guard: the engine publishes per-build masks (the bell-mouth
    # no-smooth zones) through a module global; clear it before every
    # build so a non-disk builder can never inherit the previous one's
    _we.LAST_PROTECT = None
    if (copies is not None and spec is not None and 'domain' in spec
            and spec['domain'][0] == 'torus'):
        spec2 = dict(spec, copies=int(max(1, copies)))
        out = _we.we_surface(spec2, nu, nv, order, radius, None, theta)
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
# SADDLE_TOWER_A note in zoo.py.  we_saddle_tower already
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



def build_parametric_grid(kind, nu, nv, order, radius, scale, theta=0.0,
                          equal_areas=False):
    """(nu, nv, 3) point grid plus wrap flags, centered and fit to a 2 m
    cube. (Used for NURBS output; end clipping is a mesh-only operation.)

    `equal_areas` spaces the control grid by equal surface area rather
    than equal parameter, which is usually what you want of a NURBS
    control net too -- it puts control points where the surface actually
    is instead of bunching them where the parametrization contracts."""
    if kind in MESH_PARAM:
        raise ValueError(f"{kind} has no NURBS/grid form; use mesh output")
    global LAST_EQ_AREA_COV
    LAST_EQ_AREA_COV = None
    if equal_areas:
        fu = max(nu, min(4 * nu, _EQ_AREA_FINE))
        fv = max(nv, min(4 * nv, _EQ_AREA_FINE))
        Gf, wrap_u, wrap_v, clipf = _raw_grid(kind, fu, fv, order, radius,
                                              theta)
        G, clip, cov0, cov1 = equal_area_resample(Gf, nu, nv, clipf)
        if cov1 >= cov0:                       # do no harm -- see below
            G, wrap_u, wrap_v, clip = _raw_grid(kind, nu, nv, order,
                                                radius, theta)
            LAST_EQ_AREA_COV = (cov0, cov0, False)
        else:
            LAST_EQ_AREA_COV = (cov0, cov1, True)
    else:
        G, wrap_u, wrap_v, clip = _raw_grid(kind, nu, nv, order, radius,
                                            theta)
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


@_geom_cache.memoise(version=10)  # v10: COSTA_HM seed-sign fix (the
                                  # fold strips down every copy seam);
                                  # COSTA end rims cut on the analytic
                                  # circles and circularized flat
def build_parametric(kind, nu, nv, order, radius, scale, theta=0.0,
                     with_uv=False, cells=(1, 1), equal_areas=False):
    """Mesh (V, quads) for `kind` -- see `_build_parametric` for the full
    contract.  This wrapper owns the equal-area DECISION.

    Judging the resampling on the grid alone is not good enough: the rim
    smoothing, circularization and welding that run afterwards move
    boundary vertices, and equalizing thins the boundary bands so those
    steps bite harder.  Measured on Catalan, a grid that equalized from
    0.632 to 0.033 still delivered a mesh that was WORSE than the plain
    one (0.708 against 0.631).  So when the option is on, build the
    surface both ways and ship whichever mesh actually has the more even
    faces.  The second build only happens for an opt-in flag."""
    if not equal_areas:
        return _build_parametric(kind, nu, nv, order, radius, scale, theta,
                                 with_uv, cells, equal_areas=False)

    global LAST_EQ_AREA_COV
    out_eq = _build_parametric(kind, nu, nv, order, radius, scale, theta,
                               with_uv, cells, equal_areas=True)
    grid = LAST_EQ_AREA_COV
    if grid is None or not grid[2]:
        # never reached the grid seam, or already stood down there
        return out_eq
    out_pl = _build_parametric(kind, nu, nv, order, radius, scale, theta,
                               with_uv, cells, equal_areas=False)
    cov_eq = mesh_area_cov(out_eq[0], out_eq[1])
    cov_pl = mesh_area_cov(out_pl[0], out_pl[1])
    if cov_eq < cov_pl:
        LAST_EQ_AREA_COV = (cov_pl, cov_eq, True)
        return out_eq
    LAST_EQ_AREA_COV = (cov_pl, cov_pl, False)
    return out_pl


def _build_parametric(kind, nu, nv, order, radius, scale, theta=0.0,
                      with_uv=False, cells=(1, 1), equal_areas=False):
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
    # Cleared HERE, not at the grid seam below: several surfaces return
    # early (the Scherk graphs, the finished 2-D lattice meshes, the
    # towers) and never reach it, and a stale value from the previous
    # build would have the operator report an equalization that this one
    # never performed.  None means "the option did not apply".
    global LAST_EQ_AREA_COV
    LAST_EQ_AREA_COV = None
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
    if equal_areas:
        # Sample the domain far more densely than asked, measure the area
        # element on that fine grid, then lay the requested nu x nv lines
        # at equal quantiles of it (see domain.equal_area_resample).  The
        # oversampling is what makes the measured density trustworthy.
        fu = max(nu, min(4 * nu, _EQ_AREA_FINE))
        fv = max(nv, min(4 * nv, _EQ_AREA_FINE))
        Gf, wrap_u, wrap_v, clipf = _raw_grid(kind, fu, fv, order, radius,
                                              theta, copies=copies)
        G, clip, cov0, cov1 = equal_area_resample(Gf, nu, nv, clipf)
        if cov1 >= cov0:
            # Do no harm.  A surface whose live domain is not simply a
            # rectangle -- Costa punctures its ends at INTERIOR points --
            # can end up worse: concentrating samples in the live region
            # leaves the cells that bridge a hole enormous.  Equalizing is
            # an improvement or it does not happen.
            G, wrap_u, wrap_v, clip = _raw_grid(kind, nu, nv, order,
                                                radius, theta,
                                                copies=copies)
            LAST_EQ_AREA_COV = (cov0, cov0, False)
        else:
            LAST_EQ_AREA_COV = (cov0, cov1, True)
    else:
        G, wrap_u, wrap_v, clip = _raw_grid(kind, nu, nv, order, radius,
                                            theta, copies=copies)
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
        # Bour, Henneberg, Richmond, the associate disks...).  On this
        # branch the boundary vertices are EXACT samples of the immersion
        # along the domain edge -- there is no clip staircase to relax, so
        # do NOT smooth here: any curve smoothing displaces a genuinely
        # curved rim off the true surface by ~ curvature * spacing^2, and
        # wherever the transverse mesh spacing is finer than that (the
        # rim-graded Enneper disk, the tightly wound helicoid strip edge,
        # Henneberg's inner ring) the boundary ring gets dragged through
        # its neighbour ring, folding the outermost face ring inside out --
        # one full ring of inverted normals, seen as a thin doubled "lip"
        # along the rim.  Rim smoothness comes from sampling density (res
        # baseline + per-surface res_boost), not from moving exact points.
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
        # smooth staircase/clip end rims -- EXCEPT the bell-mouth rims the
        # engine cut on exact conformal circles (LAST_PROTECT): those are
        # analytically placed, and smoothing them folds the adjacent
        # steep-flare sliver quads inside out (measured on the k-noid
        # family: every smoothing-induced flipped edge sat on a mouth ring)
        prot = getattr(_we, 'LAST_PROTECT', None)
        protv = None
        if (prot is not None and prot.shape == (nu, nv)
                and not equal_areas and len(used)):
            protv = prot.reshape(-1)[used]
        V = _smooth_boundary(V, quads, protect=protv)
        # opt-in end-rim circularization (Costa): snap each end rim to a
        # flat circle, the same presentation tile_dihedral gives the
        # Costa-Hoffman-Meeks ends.  Only for surfaces whose spec asks:
        # forcing a flat horizontal circle is right for planar/catenoid
        # ends and wrong for almost everything else (e.g. Enneper rims).
        if getattr(_b, 'spec', {}).get('circularize_ends'):
            V = _circularize_outer(V, quads)
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


def _selftest():
    ok = True

    # Registry integrity: every row is (label, callable), and the catalog in
    # zoo.py has been wired in on import.
    bad = [k for k, v in PARAMETRIC.items()
           if not (isinstance(v, tuple) and len(v) == 2
                   and isinstance(v[0], str) and callable(v[1]))]
    good = not bad and len(PARAMETRIC) > 50
    ok &= good
    print(f"parametric: {len(PARAMETRIC)} registered surfaces, "
          f"{len(MESH_PARAM)} mesh-only, malformed={bad if bad else 0} "
          f"{'OK' if good else 'FAIL'}")

    # The classical core must survive independently of the catalog.
    for key in ('ENNEPER', 'CATENOID', 'HELICOID', 'HENNEBERG', 'CATALAN',
                'BOUR', 'RICHMOND', 'SCHERK1', 'COSTA', 'CHEN_GACK'):
        good = key in PARAMETRIC
        ok &= good
        if not good:
            print(f"parametric: classical {key} MISSING  FAIL")

    # Every build is finite, non-degenerate, centred, and exactly 2 units
    # across -- the project's 2 m cube convention, enforced by _center_fit.
    for key in ('ENNEPER', 'CATENOID', 'HELICOID', 'BOUR', 'COSTA',
                'CHEN_GACK', 'SCHERK1'):
        V, Q = build_parametric(key, 40, 40, 1, 1.0, 1.0)
        lo, hi = V.min(axis=0), V.max(axis=0)
        cen = float(np.max(np.abs(0.5 * (lo + hi))))
        ext = float(np.max(hi - lo))
        g = (bool(np.all(np.isfinite(V))) and len(Q) > 100
             and cen < 1e-6 and abs(ext - 2.0) < 1e-6)
        ok &= g
        if not g:
            print(f"parametric: {key:10s} V={len(V)} Q={len(Q)} "
                  f"|c|={cen:.1e} ext={ext:.6f} FAIL")
    print(f"parametric: 7 classical surfaces fit the 2 m cube "
          f"{'OK' if ok else 'FAIL'}")

    # _center_fit normalizes each surface by one unknown uniform factor, so
    # the closed forms below are checked after recovering that single factor
    # by a 1-D fit.  Recovering it (rather than assuming a domain constant)
    # keeps these tests honest if a builder's parameter range is ever
    # retuned -- what is asserted is the SHAPE, to near machine precision.
    def _fit_scale(f, lo, hi, iters=60):
        """Scale k in [lo, hi] minimizing f(k); unimodal by construction."""
        for _ in range(iters):
            a = lo + (hi - lo) / 3.0
            b = hi - (hi - lo) / 3.0
            if f(a) < f(b):
                hi = b
            else:
                lo = a
        return 0.5 * (lo + hi)

    # The catenoid is a surface of revolution whose profile is a catenary:
    # every vertex satisfies r = s cosh(z / s) for the waist radius s.
    # Checked on the interior only, and to 5e-3 rather than to machine
    # precision, because the mesher deliberately relaxes open boundary loops
    # (_smooth_boundary) -- that pulls each rim in by a fraction of a percent
    # and shifts the normalization slightly.  A wrong profile formula would
    # miss by O(1), so this still discriminates.
    V, _ = build_parametric('CATENOID', 80, 80, 1, 1.0, 1.0)
    r = np.hypot(V[:, 0], V[:, 1])
    z = V[:, 2]
    inner = np.abs(z) < 0.98 * float(np.abs(z).max())

    def cat_res(s):
        return float(np.max(np.abs(r[inner] - s * np.cosh(z[inner] / s))))

    r0 = float(r.min())
    s = _fit_scale(cat_res, 0.5 * r0, 1.5 * r0)
    resid = cat_res(s)
    good = s > 1e-6 and resid < 5e-3
    ok &= good
    print(f"parametric: catenoid waist={s:.6f} "
          f"max|r - s cosh(z/s)|={resid:.2e} {'OK' if good else 'FAIL'}")

    # The defining property, checked directly: a minimal surface has zero
    # mean curvature everywhere.  Discrete H from the cotangent Laplacian,
    # |dx| / (4A), over interior vertices.  The MEDIAN is the statistic --
    # several of these surfaces carry genuine mesh degeneracies at poles and
    # branch points where a per-vertex H blows up harmlessly.  For scale: a
    # unit sphere at this size would read H = 1, three orders up from these.
    def _median_H(Vv, quads):
        T = []
        for f in quads:
            for k in range(1, len(f) - 1):
                T.append((f[0], f[k], f[k + 1]))
        n = len(Vv)
        Lx = np.zeros((n, 3))
        A = np.zeros(n)
        cnt = {}
        for t in T:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = (a, b) if a < b else (b, a)
                cnt[key] = cnt.get(key, 0) + 1
        bnd = set()
        for key, c in cnt.items():
            if c == 1:
                bnd.update(key)
        for t in T:
            p = Vv[list(t)]
            for i in range(3):
                j, k = (i + 1) % 3, (i + 2) % 3
                u, v = p[j] - p[i], p[k] - p[i]
                cr = float(np.linalg.norm(np.cross(u, v)))
                if cr < 1e-14:
                    continue
                cot = float(np.dot(u, v)) / cr
                Lx[t[j]] += cot * (p[k] - p[j])
                Lx[t[k]] += cot * (p[j] - p[k])
                A[t[i]] += cr / 6.0
        idx = np.array([i for i in range(n) if i not in bnd and A[i] > 1e-12])
        if not len(idx):
            return float('nan'), 0
        H = np.linalg.norm(Lx[idx], axis=1) / (4.0 * A[idx])
        return float(np.median(H)), len(idx)

    for key in ('CATENOID', 'ENNEPER', 'HELICOID', 'SCHERK1', 'COSTA',
                'CHEN_GACK', 'BOUR'):
        Vv, Q = build_parametric(key, 60, 60, 1, 1.0, 1.0)
        h, ni = _median_H(Vv, Q)
        g = ni > 500 and h == h and h < 0.02
        ok &= g
        print(f"parametric: {key:10s} median |H|={h:.5f} over {ni:5d} "
              f"interior verts {'OK' if g else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("parametric self-test failed")
