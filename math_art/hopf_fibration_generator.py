
# Hopf Fibration generator for Blender.
#
# The Hopf fibration is the map h: S^3 -> S^2 whose fibres are great
# circles of the 3-sphere; any two distinct fibres are linked exactly
# once (a Hopf link).  This generator draws a chosen set of fibres,
# stereographically projected from S^3 into R^3, where every fibre
# becomes a circle (a Villarceau circle) -- except the one fibre
# through the projection pole, which becomes the straight central
# axis.  Fibres over a circle of latitude on S^2 fill a torus of
# revolution; sweeping the latitude fills space with nested, pairwise
# linked tori (the canonical "Niles Johnson" picture).
#
# Construction (per base point p = (sin b cos l, sin b sin l, cos b)
# on S^2, colatitude b and longitude l): the fibre h^{-1}(p) is the
# Clifford circle
#     z0 = cos(b/2) e^{i(t + l)},   z1 = sin(b/2) e^{i t},   t in [0,2pi)
# i.e. the S^3 point (Re z0, Im z0, Re z1, Im z1).  One checks
# h(z0,z1) = (2 z0 conj(z1), |z0|^2 - |z1|^2) = p for every t, so the
# whole circle sits over p.  A (p,q) generalisation winds the two
# phases at integer rates -- z0 = cos(b/2) e^{i(P t + l)},
# z1 = sin(b/2) e^{i Q t} -- laying a (P,Q) torus knot/link on the
# same Clifford torus (P=Q=1 is the classical fibre).  Stereographic
# projection from the north pole N=(0,0,0,1) is
#     sigma(x1,x2,x3,x4) = (x1,x2,x3) / (1 - x4).
# Colour convention (after Niles Johnson): a fibre is coloured by its
# base point on S^2 -- hue from the longitude, value from the
# latitude -- so nearby fibres share a colour.
#
# WHAT THIS GENERATOR IS NOT.  `mesh.willmore_add` also says
# "Willmore", and the two do opposite things.  That one DESCENDS the
# bending energy and lands on its MINIMISER -- the Clifford torus at
# 2 pi^2.  The Elastica and Constrained presets here CONSTRUCT
# critical points that are not minimisers, which a descent slides
# straight off and so can never produce.  Between them they span the
# picture: one finds the minimum, the other draws the saddles around
# it.
#
# References:
# - Heinz Hopf, "Ueber die Abbildungen der dreidimensionalen Sphaere
#   auf die Kugelflaeche", Math. Ann. 104 (1931), 637-665 (the
#   fibration and its linking invariant).
# - D. W. Lyons, "An Elementary Introduction to the Hopf Fibration",
#   Math. Mag. 76 (2003), 87-98.
# - N. Johnson, "Visualization of the Hopf fibration" (2011),
#   https://nilesjohnson.net/hopf.html (the colour/latitude-torus
#   rendering imitated here).
# - Y. Villarceau (1848): a torus of revolution carries two extra
#   circles through each point, the "Villarceau circles" -- exactly
#   the Hopf fibres of a stereographically projected Clifford torus.
# - Ulrich Pinkall, "Hopf tori in S^3", Invent. Math. 81 (1985),
#   379-386 (the Hopf torus over a curve on S^2, and the theorem that
#   it is Willmore exactly over an elastic curve).
# - Joel Langer and David A. Singer, "The total squared curvature of
#   closed curves", J. Differential Geom. 20 (1984), 1-22 (closed
#   elasticae on S^2 in closed form, and their monodromy).
# - Fernando C. Marques and Andre Neves, "Min-max theory and the
#   Willmore conjecture", Ann. of Math. 179 (2014), 683-782 (the
#   Clifford torus really is the minimiser at 2 pi^2).

bl_info = {
    "name": "Hopf Fibration",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Hopf Fibration",
    "description": "Fibres of the Hopf fibration of S^3 as "
                   "stereographically projected circles (Villarceau "
                   "circles), coloured by base point on S^2",
    "category": "Add Curve",
}

import math
from math import gcd, pi, sqrt

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

_PHI = (1.0 + sqrt(5.0)) / 2.0

# A fixed, generic rotation of S^2 applied to every base point so that
# none of the standard base sets (octahedron / cube vertices, latitude
# poles) lands exactly on the south pole, whose fibre would project to
# an infinite line and blow up the scene.  The angles are arbitrary
# and irrational-looking on purpose.
_TILT = (0.3178, 0.2114, 0.1291)


def _rot_matrix(ax, ay, az):
    import numpy as np
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# --------------------------------------------------------------------------
# Kernel: fibre lift to S^3 and stereographic projection to R^3
# --------------------------------------------------------------------------

def fiber_s3(base, samples, P=1, Q=1):
    """The fibre over unit 3-vector `base` = (a,b,c) on S^2, sampled
    at `samples` angles, as an (samples, 4) array on S^3.  With P=Q=1
    this is the Hopf fibre (a great circle); general (P,Q) winds the
    two complex phases at those integer rates, giving a (P,Q) torus
    curve on the same Clifford torus."""
    import numpy as np
    a, b, c = base
    beta = math.acos(max(-1.0, min(1.0, c)))   # colatitude in [0, pi]
    lam = math.atan2(b, a)                      # longitude
    ch, sh = math.cos(beta / 2.0), math.sin(beta / 2.0)
    t = np.linspace(0.0, 2.0 * pi, samples, endpoint=False)
    p0 = P * t + lam
    p1 = Q * t
    return np.stack([ch * np.cos(p0), ch * np.sin(p0),
                     sh * np.cos(p1), sh * np.sin(p1)], axis=1)


def stereographic(X):
    """Stereographic projection S^3 -> R^3 from the north pole
    N = (0,0,0,1): (x1,x2,x3,x4) -> (x1,x2,x3)/(1 - x4).  Points near
    N are pushed far out; the caller keeps base points off the south
    pole so no fibre actually reaches N."""
    import numpy as np
    denom = 1.0 - X[:, 3]
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    return X[:, :3] / denom[:, None]


def project_fiber(base, samples, P=1, Q=1):
    """A single fibre as an (samples, 3) projected polyline."""
    return stereographic(fiber_s3(base, samples, P, Q))


# --------------------------------------------------------------------------
# Base-point providers on S^2 (each returns a list of unit 3-vectors)
# --------------------------------------------------------------------------

def _normalize(v):
    import numpy as np
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def latitudes(n_lat, n_fiber, lat_min_deg=20.0, lat_max_deg=160.0):
    """`n_lat` circles of latitude (colatitude spread between
    lat_min and lat_max degrees), each carrying `n_fiber` fibres
    equally spaced in longitude.  This is the nested-tori picture."""
    import numpy as np
    pts = []
    if n_lat == 1:
        betas = [0.5 * (lat_min_deg + lat_max_deg) * pi / 180.0]
    else:
        betas = np.linspace(lat_min_deg, lat_max_deg, n_lat) * pi / 180.0
    for beta in betas:
        for k in range(n_fiber):
            lam = 2.0 * pi * k / n_fiber
            pts.append((math.sin(beta) * math.cos(lam),
                        math.sin(beta) * math.sin(lam),
                        math.cos(beta)))
    return pts


def flower(n_fiber, ring_deg=35.0):
    """Fibres over a single small circle at colatitude `ring_deg`:
    the projected circles interleave into a flower-like linked
    bundle."""
    return latitudes(1, n_fiber, ring_deg, ring_deg)


def _platonic(kind):
    import numpy as np
    if kind == 'TETRA':
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    elif kind == 'OCTA':
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
             (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif kind == 'CUBE':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
    elif kind == 'ICOSA':
        p = _PHI
        V = []
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                V += [(0, s1, s2 * p), (s1, s2 * p, 0),
                      (s2 * p, 0, s1)]
    elif kind == 'DODECA':
        p = _PHI
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                V += [(0, s1 / p, s2 * p), (s1 / p, s2 * p, 0),
                      (s2 * p, 0, s1 / p)]
    else:
        raise ValueError(kind)
    return [tuple(_normalize(v)) for v in V]


def fibonacci(n):
    """`n` base points spread over S^2 by the golden-angle spiral --
    a quasi-uniform ball of linked fibres."""
    pts = []
    ga = pi * (3.0 - sqrt(5.0))            # golden angle
    for k in range(n):
        z = 1.0 - (2.0 * k + 1.0) / n      # in (-1, 1)
        r = sqrt(max(0.0, 1.0 - z * z))
        phi = ga * k
        pts.append((r * math.cos(phi), r * math.sin(phi), z))
    return pts


def great_circle(n_fiber, tilt_deg=55.0):
    """Fibres over a great circle of S^2 tilted `tilt_deg` from the
    equator: a single, maximally spread band of linked fibres (its
    two extreme fibres bound a Hopf band)."""
    import numpy as np
    pts = []
    a = tilt_deg * pi / 180.0
    for k in range(n_fiber):
        u = 2.0 * pi * k / n_fiber
        # equator circle rotated about the x-axis by `a`
        x = math.cos(u)
        y = math.sin(u) * math.cos(a)
        z = math.sin(u) * math.sin(a)
        pts.append((x, y, z))
    return pts


_PROVIDERS = {
    'LATITUDES': "Nested tori (circles of latitude)",
    'FLOWER': "Flower (one small circle of base points)",
    'GREATCIRCLE': "Great-circle band",
    'FIBONACCI': "Fibonacci sphere (quasi-uniform)",
    'TETRA': "Tetrahedron vertices",
    'OCTA': "Octahedron vertices",
    'CUBE': "Cube vertices",
    'ICOSA': "Icosahedron vertices",
    'DODECA': "Dodecahedron vertices",
}


def base_points(preset, n_lat, n_fiber, lat_min, lat_max):
    """Dispatch to a base-point provider; returns a list of unit
    3-vectors on S^2 (before the generic tilt is applied)."""
    if preset == 'LATITUDES':
        return latitudes(n_lat, n_fiber, lat_min, lat_max)
    if preset == 'FLOWER':
        return flower(n_fiber, 0.5 * (lat_min + lat_max))
    if preset == 'GREATCIRCLE':
        return great_circle(n_fiber, 0.5 * (lat_min + lat_max))
    if preset == 'FIBONACCI':
        return fibonacci(max(1, n_fiber))
    return _platonic(preset)


# --------------------------------------------------------------------------
# Assembly: project every fibre, tilt/centre/scale to fit the unit cube
# --------------------------------------------------------------------------

def build_fibers(preset='LATITUDES', n_lat=6, n_fiber=24, samples=160,
                 P=1, Q=1, lat_min=20.0, lat_max=160.0,
                 fit_radius=1.0, max_radius=12.0):
    """Return (fibers, bases): `fibers` is a list of (samples, 3)
    projected polylines and `bases` the matching tilted base points
    (for colouring).  The whole set is centred at the origin and
    uniformly scaled so its 95th-percentile point radius is
    `fit_radius`; any fibre whose points blow past `max_radius`
    (a base point near the south pole) is dropped."""
    import numpy as np
    R = _rot_matrix(*_TILT)
    raw = base_points(preset, n_lat, n_fiber, lat_min, lat_max)
    bases = [tuple(R @ np.asarray(b)) for b in raw]

    polylines = [project_fiber(b, samples, P, Q) for b in bases]

    # drop near-line fibres, then centre + scale on what remains
    kept = [(p, b) for p, b in zip(polylines, bases)
            if np.isfinite(p).all()
            and np.linalg.norm(p, axis=1).max() < max_radius]
    if not kept:
        return [], []
    fibers = [p for p, _ in kept]
    bases = [b for _, b in kept]

    allpts = np.concatenate(fibers, axis=0)
    center = 0.5 * (allpts.max(0) + allpts.min(0))
    rad = np.linalg.norm(allpts - center, axis=1)
    ref = np.percentile(rad, 95.0)
    scale = (fit_radius / ref) if ref > 1e-9 else 1.0
    fibers = [(p - center) * scale for p in fibers]
    return fibers, bases


def _fiber_color(base):
    """Standard sphere colouring of a base point: hue from longitude,
    value from latitude (north pole light, south pole dark)."""
    import colorsys
    a, b, c = base
    hue = (math.atan2(b, a) / (2.0 * pi)) % 1.0
    val = 0.35 + 0.6 * (0.5 * (c + 1.0))       # c in [-1,1] -> val
    return colorsys.hsv_to_rgb(hue, 0.72, val)


# --------------------------------------------------------------------------
# Pinkall Hopf tori: the preimage of a closed curve on S^2
# --------------------------------------------------------------------------
# Ulrich Pinkall, "Hopf tori in S^3", Invent. Math. 81 (1985) 379-386:
# the full preimage h^{-1}(gamma) of a closed curve gamma on S^2 is an
# (immersed) torus in S^3 -- a "Hopf torus".  It is intrinsically flat,
# its mean curvature equals the geodesic curvature of gamma, and its
# Willmore energy is pi * integral_gamma (1 + kappa^2) ds, so minimising
# surface bending reduces to an elastic-curve problem for gamma.  We
# build the torus as the orbit of a lift Gamma(s) of gamma under the
# Hopf circle action e^{i psi}*(z0,z1); the loop closes because a curve
# winding the sphere returns e^{i psi} to itself.  The closure carries a
# twist equal to half the spherical area A enclosed by gamma (the
# holonomy), which we report for reference.

def gamma_curve(preset, n, colat_deg, lobes, amp_deg, ecc,
                el_m=1, el_n=3, cw=None):
    """A closed curve on S^2 as `n` unit 3-vectors (endpoint
    excluded).  colat_deg sets the mean colatitude; `lobes`/`amp_deg`
    drive the wavy m-fold curve; `ecc` squashes the ellipse;
    `el_m`/`el_n` are the monodromy fraction of the ELASTICA preset."""
    import numpy as np
    b0 = colat_deg * pi / 180.0
    amp = amp_deg * pi / 180.0
    s = np.linspace(0.0, 2.0 * pi, n, endpoint=False)
    if preset == 'ELASTICA':
        return spherical_elastica(el_m, el_n, n)
    if preset == 'CONSTRAINED':
        # Heller's constrained elastic curves, in closed form on a
        # rhombic Weierstrass lattice.  The mathematics lives in its own
        # module because it is large; the OPERATOR is this one, because
        # a constrained Willmore torus is a Hopf torus over a
        # constrained elastic curve -- the same construction as the
        # ELASTICA preset with the Lagrange multiplier switched on.
        from .constrained_willmore_generator import (heller_curve,
                                                     resolve_params)
        kw = dict(cw or {})
        t, m_wind, branch, x0f = resolve_params(
            kw.get('lobes', 3), kw.get('winding', 1),
            kw.get('family', 'WILLMORE'), kw.get('shape', 0.55),
            kw.get('branch', 'UPPER'), kw.get('phase', 0.5),
            kw.get('high_wrap', False))
        # heller_curve returns (points, info); gamma_curve's contract is
        # just the points, so the diagnostics are dropped here
        return heller_curve(t, kw.get('lobes', 3), m_wind, branch, x0f,
                            n)[0]
    if preset == 'CIRCLE':
        beta = np.full_like(s, b0)
        lam = s
    elif preset == 'WAVY':
        beta = np.clip(b0 + amp * np.cos(lobes * s), 0.06, pi - 0.06)
        lam = s
    elif preset == 'TREFOIL':
        # (2, lobes) winding: colatitude modulated as longitude winds
        beta = np.clip(b0 + amp * np.cos(lobes * s), 0.06, pi - 0.06)
        lam = 2.0 * s
    elif preset == 'BAND':
        # OPEN arc of a meridian -> Hopf band (an annulus whose two
        # boundary fibres form a Hopf link). Endpoints included.
        lo = max(0.06, b0 - amp)
        hi = min(pi - 0.06, b0 + amp)
        beta = np.linspace(lo, hi, n)
        lam = np.zeros_like(beta)
    elif preset == 'ELLIPSE':
        a = math.sin(b0)
        bb = math.sin(b0) * max(0.05, 1.0 - ecc)
        c = math.cos(b0)
        V = np.stack([a * np.cos(s), bb * np.sin(s),
                      np.full_like(s, c)], axis=1)
        return [tuple(v / np.linalg.norm(v)) for v in V]
    else:
        beta = np.full_like(s, b0)
        lam = s
    return [(math.sin(bt) * math.cos(lm), math.sin(bt) * math.sin(lm),
             math.cos(bt)) for bt, lm in zip(beta, lam)]


# --------------------------------------------------------------------------
# Spherical elastica -> Willmore tori
# --------------------------------------------------------------------------
# Pinkall's theorem (Sect. 4 of the 1985 paper): the Hopf torus over gamma
# is a WILLMORE surface -- a critical point of the bending energy -- if and
# only if gamma is a critical point of the elastic energy
#   F^lambda(gamma) = closed-integral (kappa^2 + lambda) ds  with lambda = 1.
# Langer and Singer (J. Diff. Geom. 20 (1984), eq. (1.2)) show such a curve
# on a surface of Gauss curvature G obeys
#   2 kappa_ss + kappa^3 + 2 kappa G - lambda kappa = 0 ,
# which on the unit sphere with lambda = 1 is  2 k_ss + k^3 + k = 0 , and
# their Table (2.7)(a) integrates it: on a 2-manifold the torsion constant
# vanishes, so the general solution collapses to the single "wavelike" arc
#   kappa(s) = sqrt(a) cn(r s, p) ,  p^2 = a/(2a+2) ,  r = sqrt(2a+2)/2 ,
# one branch per maximum squared curvature a > 0 (p^2 < 1/2 automatically).
#
# Closure is a rotation condition, not a length condition.  kappa has
# period P = 4 K(p)/r, and over one period the spherical Frenet frame is
# carried by a fixed rotation (the MONODROMY) about the axis of the Killing
# field that Langer-Singer attach to an elastica.  The curve closes after n
# periods exactly when that rotation has order dividing n, i.e. when its
# angle is theta = 2 pi m / n.  theta(a) is measured here directly from the
# integrated frame; it decreases monotonically from 2 pi (2 - sqrt 2) at
# a -> 0 to 0 as a -> infinity, so each admissible m/n has exactly one a and
# bisection cannot miss it.  Hence the admissible monodromies are exactly
# the fractions m/n in (0, 2 - sqrt 2); nothing outside closes.
#
# That upper endpoint is not fitted, it is forced.  As a -> 0 the curve
# degenerates to a great circle while the period of kappa tends to
# 4 K(0)/r = 2 pi sqrt 2, so one "lobe" runs sqrt 2 times around a closed
# geodesic and leaves a residual frame rotation of 2 pi (sqrt 2 - 1);
# measured with the axis as oriented here that reads 2 pi (2 - sqrt 2).
# Note this window is for lambda = 1 -- Pinkall's functional -- and so
# differs from the familiar (1/2, 1/sqrt 2) window of the FREE (lambda = 0)
# spherical elasticae; the two families are not the same curves.
#
# Working with the monodromy ANGLE rather than the unwrapped longitude
# matters: at a = 1 the curve passes exactly through the pole of its own
# axis, where an azimuth is undefined and any winding-number bookkeeping
# breaks down.  The rotation angle is smooth straight through that point.

_ELASTICA_LAMBDA = 1.0          # Pinkall's functional is F^{lambda=1}
# sup of the monodromy angle, at a -> 0:  2 pi (2 - sqrt 2)
ELASTICA_RATIO_MAX = 2.0 - sqrt(2.0)


def _elastica_kappa(a, s):
    """Curvature kappa(s) and its s-derivative for the spherical elastica
    with maximum squared curvature `a`, plus the period of kappa."""
    import numpy as np
    from .minsurf.elliptic import ellipk, jacobi_sncndn
    p2 = a / (2.0 * a + 2.0)
    r = 0.5 * sqrt(2.0 * a + 2.0)
    sn, cn, dn = jacobi_sncndn(r * np.asarray(s, dtype=float), p2)
    root_a = sqrt(a)
    return root_a * cn, -root_a * r * sn * dn, 4.0 * ellipk(p2) / r


def _elastica_frames(a, n_per, nstep):
    """Integrate the spherical Frenet system
        gamma' = T ,   T' = kappa (gamma x T) - gamma
    for `n_per` periods of kappa with RK4, reprojecting onto S^2 each step.
    Returns (gamma samples, final frame, initial frame, axis, period).

    kappa is sampled on a HALF-step grid so the two midpoint stages get
    the exact kappa(s + h/2) rather than the average of its endpoints.
    That distinction is what makes this RK4 rather than an O(h^2) scheme:
    with the averaged midpoint the monodromy angle converges only like
    h^2, and the root-find below then needs ~40x more steps for the same
    accuracy."""
    import numpy as np
    P = _elastica_kappa(a, 0.0)[2]
    total = n_per * P
    h = total / nstep
    s = np.linspace(0.0, total, 2 * nstep + 1)   # endpoints AND midpoints
    kap, kaps, _ = _elastica_kappa(a, s)

    g = np.array([0.0, 0.0, 1.0])
    t = np.array([1.0, 0.0, 0.0])
    # Langer-Singer's Killing field J = -2k gamma - 2k_s T + (k^2-lambda) U
    # is constant along the curve; its direction is the axis of the
    # monodromy (|J| = a + 1, so it never degenerates).
    axis = (-2.0 * kap[0] * g - 2.0 * kaps[0] * t
            + (kap[0] ** 2 - _ELASTICA_LAMBDA) * np.cross(g, t))
    axis = axis / np.linalg.norm(axis)
    M0 = np.stack([g, t, np.cross(g, t)], axis=1)

    def dv(g, t, k):
        return t, k * np.cross(g, t) - g

    pts = [tuple(g)]
    for i in range(nstep):
        k0, km, k1 = kap[2 * i], kap[2 * i + 1], kap[2 * i + 2]
        a1, b1 = dv(g, t, k0)
        a2, b2 = dv(g + 0.5 * h * a1, t + 0.5 * h * b1, km)
        a3, b3 = dv(g + 0.5 * h * a2, t + 0.5 * h * b2, km)
        a4, b4 = dv(g + h * a3, t + h * b3, k1)
        g = g + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        t = t + (h / 6.0) * (b1 + 2 * b2 + 2 * b3 + b4)
        g = g / np.linalg.norm(g)
        t = t - np.dot(t, g) * g
        t = t / np.linalg.norm(t)
        pts.append(tuple(g))
    M1 = np.stack([g, t, np.cross(g, t)], axis=1)
    return pts, M1, M0, axis, P


def _monodromy_angle(a, nstep=200):
    """Rotation angle in [0, 2pi) carried by one period of kappa.

    200 RK4 steps put the angle within ~4e-7 of its converged value,
    which is far finer than the bisection needs and keeps the root-find
    to a few hundredths of a second."""
    import numpy as np
    _, M1, M0, axis, _ = _elastica_frames(a, 1, nstep)
    R = M1 @ M0.T
    v = np.array([1.0, 0.0, 0.0]) - axis[0] * axis
    if np.linalg.norm(v) < 1e-8:
        v = np.array([0.0, 1.0, 0.0]) - axis[1] * axis
    v = v / np.linalg.norm(v)
    Rv = R @ v
    return math.atan2(float(np.dot(axis, np.cross(v, Rv))),
                      float(np.dot(v, Rv))) % (2.0 * pi)


_ELASTICA_A_CACHE = {}


def elastica_a_for(m, n, tol=1e-12):
    """The maximum squared curvature `a` whose elastica closes after `n`
    periods with monodromy angle 2 pi m / n.  theta(a) is monotone
    decreasing onto (0, 2 pi (2 - sqrt 2)), so plain bisection converges;
    raises ValueError when m/n is outside that range.

    The bisection costs ~80 RK4 sweeps, so results are memoised: the
    redo panel re-runs execute() on every slider drag and the shape
    parameter depends only on (m, n)."""
    cached = _ELASTICA_A_CACHE.get((m, n))
    if cached is not None:
        return cached
    target = 2.0 * pi * m / n
    if not (0.0 < m / n < ELASTICA_RATIO_MAX):
        raise ValueError(
            f"m/n = {m}/{n} = {m / n:.4f} is outside the admissible range "
            f"(0, 2-sqrt(2)) = (0, {ELASTICA_RATIO_MAX:.4f}); no closed "
            f"spherical elastica has this monodromy")
    lo, hi = 1e-9, 1.0                     # theta(lo) > target
    while _monodromy_angle(hi) > target:   # push hi until theta(hi) < target
        hi *= 4.0
        if hi > 1e12:
            raise ValueError("elastica bracket failed")
    for _ in range(80):
        mid = sqrt(lo * hi)                # geometric: `a` spans many decades
        if _monodromy_angle(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * hi:
            break
    a = sqrt(lo * hi)
    _ELASTICA_A_CACHE[(m, n)] = a
    return a


def _clear_of_poles(pts, n_cand=4000):
    """Rotate a closed spherical curve so that BOTH poles (0,0,+-1) are
    as far from it as possible, and return the rotated copy.

    This is not cosmetic.  `build_hopf_torus` lifts the base point at
    colatitude beta to z0 = cos(beta/2) e^{i lam}, z1 = sin(beta/2), and
    that section of the Hopf bundle degenerates at both ends of the
    axis:

      * beta = pi lifts to the fibre {(0, 0, cos psi, sin psi)}, which
        runs straight through the projection centre N = (0,0,0,1) of
        `stereographic`.  A curve touching the south pole therefore has
        a fibre escaping to infinity, and one merely passing near it
        gives a torus thousands of units across whose useful part
        collapses to a sliver once the mesh is normalised.
      * beta = 0 leaves lam undefined.  The fibre is still the right
        circle, but the ring's starting phase swings wildly from one
        curve sample to the next, shearing the quads into slivers and
        wrecking the discrete mean curvature even though the surface
        itself is fine.

    An elastica generically DOES sweep the whole sphere (it starts at
    (0,0,1) and its lobes reach the far side), so this must be handled
    rather than assumed away.  A rotation of S^2 lifts to an isometry of
    S^3, and stereographic projection is conformal, so it changes
    neither the Willmore energy nor the conformal type of the torus --
    only which conformal representative in R^3 we draw.

    The optimum is found by max-min search over a golden-angle grid:
    take the candidate AXIS whose smallest angle to the curve -- measured
    to the line +-p, so both poles count -- is largest, then map it to z."""
    import numpy as np
    G = np.asarray(pts, dtype=float)
    sub = G[:: max(1, len(G) // 400)]
    C = np.asarray(fibonacci(n_cand), dtype=float)
    worst_dot = np.abs(C @ sub.T).max(axis=1)   # cos of the SMALLEST angle
    best = C[int(np.argmin(worst_dot))]         # smallest max-cos = farthest
    tgt = np.array([0.0, 0.0, 1.0])
    v = np.cross(best, tgt)
    c = float(np.dot(best, tgt))
    if np.linalg.norm(v) < 1e-12:            # already (anti)aligned
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0.0, -v[2], v[1]],
                      [v[2], 0.0, -v[0]],
                      [-v[1], v[0], 0.0]])
        R = np.eye(3) + K + K @ K / (1.0 + c)
    return G @ R.T


def spherical_elastica(m, n, samples):
    """Closed elastic curve on S^2 with monodromy 2 pi m / n, sampled at
    `samples` points (endpoint excluded).  Its Hopf torus is a Willmore
    surface (Pinkall Sect. 4); the curve has `n` lobes.

    The curve is returned rotated clear of both poles (see
    `_clear_of_poles`) so that its Hopf torus projects to a bounded
    region of R^3 on a well-conditioned quad grid."""
    import numpy as np
    a = elastica_a_for(m, n)
    pts, _, _, _, _ = _elastica_frames(a, n, samples)
    return _clear_of_poles(np.asarray(pts[:-1], dtype=float))


def _geodesic_curvature(gamma_pts):
    """Geodesic curvature kappa(s) and arclength spacing of a closed curve
    on the unit sphere, sampled at `gamma_pts`.

    For a unit-speed spherical curve the Frenet relation on S^2 is
        gamma'' = kappa * (gamma x gamma') - gamma ,
    so kappa = <gamma'', gamma x gamma'>.  The samples are generally NOT
    equally spaced in arclength (a wavy curve bunches up near its
    turning points), so the first and second derivatives use the
    non-uniform centred stencils rather than the equal-spacing ones --
    with equal-spacing formulas the error does not vanish under
    refinement and the reported energy converges to the wrong number.

    Returns (kappa, ds, L): per-node curvature, per-node arclength weight
    (the trapezoid dual of the edge lengths) and the total length."""
    import numpy as np
    G = np.asarray(gamma_pts, dtype=float)
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    nxt, prv = np.roll(G, -1, axis=0), np.roll(G, 1, axis=0)

    # Edge lengths as great-circle arcs; h_minus/h_plus straddle a node.
    e_plus = np.arccos(np.clip(np.sum(G * nxt, axis=1), -1.0, 1.0))
    e_minus = np.roll(e_plus, 1)
    hp, hm = e_plus[:, None], e_minus[:, None]
    hs = hm + hp

    d1 = (-hp / (hm * hs)) * prv + ((hp - hm) / (hm * hp)) * G \
        + (hm / (hp * hs)) * nxt
    d2 = 2.0 * (prv / (hm * hs) - G / (hm * hp) + nxt / (hp * hs))

    # Normalise the tangent: the stencil is O(h^2)-accurate, not exactly
    # unit, and kappa is sensitive to that on coarse samplings.
    t = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    kappa = np.sum(d2 * np.cross(G, t), axis=1)
    ds = 0.5 * (e_minus + e_plus)
    return kappa, ds, float(np.sum(e_plus))


def willmore_energy(gamma_pts):
    """Pinkall's Willmore energy of the Hopf torus over the closed
    spherical curve `gamma_pts`:

        W(M) = pi * closed-integral (1 + kappa^2) ds

    (Pinkall, Invent. Math. 81 (1985), Sect. 4).  Returns (W, L), the
    energy and the curve's length.

    Reference values: a GREAT circle gives the Clifford torus, kappa = 0
    and L = 2 pi, hence W = 2 pi^2 = 19.7392, the conjectured (and since
    Marques-Neves 2014, proven) minimum over all immersed tori.  A
    latitude circle at colatitude beta has kappa = cot beta and
    L = 2 pi sin beta, so W = 2 pi^2 / sin beta -- which is the exact
    closed form the self-test checks against."""
    kappa, ds, L = _geodesic_curvature(gamma_pts)
    import numpy as np
    return float(np.pi * np.sum((1.0 + kappa ** 2) * ds)), L


def _enclosed_area(gamma_pts):
    """Signed spherical area enclosed by the closed curve, via the
    solid-angle line integral A = closed-integral (1 - cos beta) dlam."""
    import numpy as np
    G = np.asarray(gamma_pts)
    beta = np.arccos(np.clip(G[:, 2], -1.0, 1.0))
    lam = np.arctan2(G[:, 1], G[:, 0])
    dlam = np.angle(np.exp(1j * (np.roll(lam, -1) - lam)))  # wrapped
    return float(np.sum((1.0 - np.cos(beta)) * dlam))


def build_hopf_torus(gamma_pts, m_psi, closed=True, fit_radius=1.0,
                     max_radius=40.0):
    """Mesh the Hopf torus h^{-1}(gamma): each of the N curve samples
    contributes a fibre circle of `m_psi` points (the Hopf orbit of a
    lift), joined into a quad grid.  `closed` wraps the curve direction
    into a torus; `closed=False` leaves an open annulus (a Hopf band).
    Returns (verts, faces, area) with verts centred and scaled to the
    unit cube."""
    import numpy as np
    N, M = len(gamma_pts), m_psi
    psi = np.linspace(0.0, 2.0 * pi, M, endpoint=False)
    cp, sp = np.cos(psi), np.sin(psi)
    rings = []
    for g in gamma_pts:
        x, y, z = g
        beta = math.acos(max(-1.0, min(1.0, z)))
        lam = math.atan2(y, x)
        cb, sb = math.cos(beta / 2.0), math.sin(beta / 2.0)
        z0r, z0i = cb * math.cos(lam), cb * math.sin(lam)
        z1r, z1i = sb, 0.0
        X = np.stack([cp * z0r - sp * z0i, sp * z0r + cp * z0i,
                      cp * z1r - sp * z1i, sp * z1r + cp * z1i], axis=1)
        rings.append(stereographic(X))
    verts = np.concatenate(rings, axis=0)
    center = 0.5 * (verts.max(0) + verts.min(0))
    rad = np.linalg.norm(verts - center, axis=1)
    ref = np.percentile(rad, 97.0)
    scale = (fit_radius / ref) if ref > 1e-9 else 1.0
    verts = (verts - center) * scale
    faces = []
    rows = N if closed else N - 1
    for i in range(rows):
        i1 = (i + 1) % N
        for j in range(M):
            j1 = (j + 1) % M
            faces.append([i * M + j, i * M + j1,
                          i1 * M + j1, i1 * M + j])
    return verts, faces, _enclosed_area(gamma_pts)


if _IN_BLENDER:

    class CURVE_OT_hopf_fibration_add(bpy.types.Operator):
        """Add fibres of the Hopf fibration of S^3, stereographically
        projected to R^3 as circles and coloured by base point"""
        bl_idname = "curve.hopf_fibration_add"
        bl_label = "Hopf Fibration"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Base Points",
            items=[(k, k.title(), v) for k, v in _PROVIDERS.items()],
            default='LATITUDES')
        n_lat: IntProperty(
            name="Latitudes / Rings", default=6, min=1, max=48,
            description="Circles of latitude (LATITUDES), or the "
                        "point budget for the Fibonacci sphere")
        n_fiber: IntProperty(
            name="Fibres", default=24, min=1, max=200,
            description="Fibres per latitude (LATITUDES), or the total "
                        "number of fibres for FLOWER / GREATCIRCLE / "
                        "FIBONACCI presets")
        samples: IntProperty(
            name="Samples", default=160, min=24, max=1024,
            description="Points per fibre")
        wind_p: IntProperty(
            name="Wind P", default=1, min=1, max=12,
            description="Winding rate of the first phase; (P,Q) != "
                        "(1,1) lays a (P,Q) torus knot/link on each "
                        "fibre's Clifford torus")
        wind_q: IntProperty(
            name="Wind Q", default=1, min=1, max=12,
            description="Winding rate of the second phase")
        lat_min: FloatProperty(
            name="Colat Min", default=20.0, min=1.0, max=179.0,
            description="Smallest colatitude (deg) for latitude / "
                        "flower / great-circle presets")
        lat_max: FloatProperty(
            name="Colat Max", default=160.0, min=1.0, max=179.0,
            description="Largest colatitude (deg)")
        output: EnumProperty(
            name="Output",
            items=[('BEZIER', "Bezier Curve", "auto-smoothed"),
                   ('POLY', "Poly Curve", ""),
                   ('NURBS', "NURBS Curve", ""),
                   ('MESH', "Mesh Tube", "swept tube mesh")],
            default='MESH')
        radius: FloatProperty(
            name="Tube Radius", default=0.02, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius")
        resolution: IntProperty(name="Bevel Resolution", default=4,
                                min=1, max=16)
        tube_sides: IntProperty(name="Tube Sides", default=8,
                                min=3, max=32)
        color_fibers: BoolProperty(
            name="Colour by Base Point", default=True,
            description="One material per fibre, hue from longitude "
                        "and value from latitude of its base point")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            import numpy as np
            fibers, bases = build_fibers(
                self.preset, self.n_lat, self.n_fiber, self.samples,
                self.wind_p, self.wind_q, self.lat_min, self.lat_max)
            if not fibers:
                self.report({'ERROR'}, "No fibres produced")
                return {'CANCELLED'}
            fibers = [np.asarray(f) * self.scale for f in fibers]
            d = len(fibers)
            name = f"Hopf Fibration ({self.preset.title()})"
            color = self.color_fibers

            if self.output == 'MESH':
                try:
                    from .knots import closed_tube as tube
                except ImportError:
                    from knots import closed_tube as tube
                verts, faces, midx = [], [], []
                for k, Pl in enumerate(fibers):
                    v, f = tube(Pl, self.radius, self.tube_sides)
                    base = len(verts)
                    verts.extend(v)
                    faces.extend([[base + i for i in face]
                                  for face in f])
                    midx.extend([k] * len(f))
                me = bpy.data.meshes.new(name)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                me.polygons.foreach_set(
                    'use_smooth', [True] * len(me.polygons))
                if color and len(me.polygons) == len(midx):
                    me.polygons.foreach_set('material_index', midx)
                me.update()
                obj = bpy.data.objects.new(name, me)
            else:
                cu = bpy.data.curves.new(name, 'CURVE')
                cu.dimensions = '3D'
                for Pl in fibers:
                    if self.output == 'BEZIER':
                        sp = cu.splines.new('BEZIER')
                        sp.bezier_points.add(len(Pl) - 1)
                        for i, pnt in enumerate(Pl):
                            bp = sp.bezier_points[i]
                            bp.co = pnt
                            bp.handle_left_type = 'AUTO'
                            bp.handle_right_type = 'AUTO'
                    else:
                        sp = cu.splines.new(self.output)
                        sp.points.add(len(Pl) - 1)
                        for i, pnt in enumerate(Pl):
                            sp.points[i].co = (pnt[0], pnt[1],
                                               pnt[2], 1.0)
                        if self.output == 'NURBS':
                            sp.order_u = 4
                    sp.use_cyclic_u = True
                cu.bevel_depth = self.radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)

            if color:
                for k in range(d):
                    rgb = _fiber_color(bases[k])
                    mat = bpy.data.materials.new(f"{name} F{k + 1}")
                    mat.diffuse_color = (*rgb, 1.0)
                    mat.use_nodes = True
                    node = mat.node_tree.nodes.get("Principled BSDF")
                    if node:
                        node.inputs["Base Color"].default_value = \
                            (*rgb, 1.0)
                    obj.data.materials.append(mat)
                if self.output != 'MESH':
                    for k, sp in enumerate(obj.data.splines):
                        sp.material_index = k

            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report(
                {'INFO'},
                f"{name}: {d} fibres, {self.samples} samples each"
                + ("" if (self.wind_p, self.wind_q) == (1, 1)
                   else f", ({self.wind_p},{self.wind_q}) winding"))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'LATITUDES':
                lay.prop(self, 'n_lat')
            if self.preset in ('LATITUDES', 'FLOWER', 'GREATCIRCLE',
                               'FIBONACCI'):
                lay.prop(self, 'n_fiber')
            lay.prop(self, 'samples')
            row = lay.row(align=True)
            row.prop(self, 'wind_p')
            row.prop(self, 'wind_q')
            if self.preset in ('LATITUDES', 'FLOWER', 'GREATCIRCLE'):
                lay.prop(self, 'lat_min')
                lay.prop(self, 'lat_max')
            lay.prop(self, 'output')
            lay.prop(self, 'radius')
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            elif self.radius > 0:
                lay.prop(self, 'resolution')
            lay.prop(self, 'color_fibers')
            lay.prop(self, 'scale')

    class MESH_OT_hopf_torus_add(bpy.types.Operator):
        """Add a Pinkall Hopf torus: the preimage h^{-1}(gamma) of a
        closed curve gamma on S^2, stereographically projected to R^3"""
        bl_idname = "mesh.hopf_torus_add"
        bl_label = "Hopf Torus"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Curve on S^2",
            items=[('CIRCLE', "Circle", "latitude circle -> torus "
                    "of revolution filled by Villarceau circles"),
                   ('WAVY', "Wavy (m-lobed)", "m-fold undulating "
                    "closed curve -> m-fold symmetric Hopf torus"),
                   ('ELLIPSE', "Ellipse", "squashed loop"),
                   ('TREFOIL', "Trefoil-like", "doubly-wound curve"),
                   ('ELASTICA', "Elastica (Willmore)", "closed "
                    "elastic curve on S^2 -> a WILLMORE torus: a "
                    "critical point of the Willmore energy"),
                   ('CONSTRAINED', "Constrained Elastica",
                    "constrained elastic curve on S^2 -> a CONSTRAINED "
                    "WILLMORE torus (Heller): the same construction as "
                    "Elastica with the area multiplier switched on"),
                   ('BAND', "Hopf Band", "open meridian arc -> "
                    "annulus whose two boundary fibres are a Hopf "
                    "link (a Seifert surface for it)")],
            default='WAVY')
        n_curve: IntProperty(
            name="Curve Samples", default=200, min=12, max=2000,
            description="Samples along gamma (the torus meridians)")
        m_psi: IntProperty(
            name="Fibre Samples", default=64, min=6, max=512,
            description="Samples around each Hopf fibre (the torus "
                        "longitudes)")
        colat: FloatProperty(
            name="Mean Colatitude", default=90.0, min=10.0, max=170.0,
            description="Mean colatitude of gamma on S^2 (deg)")
        lobes: IntProperty(
            name="Lobes", default=3, min=1, max=12,
            description="Number of lobes (WAVY / TREFOIL)")
        amp: FloatProperty(
            name="Amplitude", default=35.0, min=0.0, max=80.0,
            description="Lobe amplitude in colatitude (deg)")
        ecc: FloatProperty(
            name="Ellipse Squash", default=0.5, min=0.0, max=0.95,
            description="Ellipse eccentricity (ELLIPSE)")
        elastica_m: IntProperty(
            name="Winding", default=1, min=1, max=20,
            description="Numerator of the elastica's monodromy m/n: "
                        "one lobe carries the frame 2 pi m/n around "
                        "the axis, so the closed curve accumulates m "
                        "full turns")
        elastica_n: IntProperty(
            name="Lobes", default=3, min=2, max=40,
            description="Denominator of the monodromy m/n: the curve "
                        "has n lobes.  m/n must lie in (0, 2-sqrt 2)")
        cw_family: EnumProperty(
            name="Family",
            items=[('WILLMORE', "Willmore",
                    "shape solved so the torus is WILLMORE (mu = -G/2) "
                    "-- the same surfaces the Elastica preset builds, "
                    "by an independent route"),
                   ('FREE', "Free elastica",
                    "shape solved so the curve is a FREE elastica "
                    "(mu = 0)"),
                   ('ELASTIC', "Elastic (custom shape)",
                    "elastic curve (lambda = 0) at a chosen lattice "
                    "shape"),
                   ('CONSTRAINED', "Constrained elastic",
                    "genuinely constrained (lambda != 0): the phase "
                    "slides Heller's isospectral family")],
            default='WILLMORE')
        cw_lobes: IntProperty(
            name="Lobes", default=3, min=2, max=24,
            description="Lobe count n of the constrained elastic curve")
        cw_winding: IntProperty(
            name="Winding", default=1, min=1, max=23,
            description="Winding w, coprime to the lobe count")
        cw_shape: FloatProperty(
            name="Shape", default=0.55, min=0.05, max=3.0,
            description="Lattice shape parameter (custom families only)")
        cw_branch: EnumProperty(
            name="Branch",
            items=[('UPPER', "Upper (gentler)", "the gentler root"),
                   ('LOWER', "Lower (curlier)", "the curlier root")],
            default='UPPER')
        cw_phase: FloatProperty(
            name="Phase", default=0.5, min=0.02, max=0.98,
            description="Phase x0 as a fraction of the imaginary "
                        "half-period; 0.5 is lambda = 0, off-centre "
                        "sweeps genuinely constrained curves")
        cw_high_wrap: BoolProperty(
            name="High Wrap", default=False,
            description="Use the high-wrap branch m = 2n + w")
        shade_smooth: BoolProperty(name="Shade Smooth", default=True)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            import numpy as np
            try:
                gamma = gamma_curve(
                    self.preset, self.n_curve, self.colat, self.lobes,
                    self.amp, self.ecc, self.elastica_m, self.elastica_n,
                    dict(lobes=self.cw_lobes, winding=self.cw_winding,
                         family=self.cw_family, shape=self.cw_shape,
                         branch=self.cw_branch, phase=self.cw_phase,
                         high_wrap=self.cw_high_wrap))
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            verts, faces, area = build_hopf_torus(
                gamma, self.m_psi, closed=(self.preset != 'BAND'))
            verts = (np.asarray(verts) * self.scale).tolist()
            if self.preset == 'ELASTICA':
                name = (f"Willmore Torus ({self.elastica_m}-"
                        f"{self.elastica_n})")
            elif self.preset == 'CONSTRAINED':
                name = (f"Constrained Willmore Torus "
                        f"({self.cw_lobes}-{self.cw_winding})")
            else:
                name = f"Hopf Torus ({self.preset.title()})"
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            if self.shade_smooth:
                me.polygons.foreach_set(
                    'use_smooth', [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            msg = (f"{name}: {len(verts)} verts, enclosed area "
                   f"{area:.3f} sr, closure twist {area / 2.0:.3f} rad")
            if self.preset != 'BAND':
                energy, length = willmore_energy(gamma)
                msg += (f", Willmore energy {energy:.4f} = "
                        f"{energy / (2.0 * pi * pi):.4f} x 2pi^2"
                        f" (gamma length {length:.3f})")
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'n_curve')
            lay.prop(self, 'm_psi')
            if self.preset not in ('ELASTICA', 'CONSTRAINED'):
                lay.prop(self, 'colat')
            if self.preset == 'CONSTRAINED':
                lay.prop(self, 'cw_family')
                lay.prop(self, 'cw_lobes')
                lay.prop(self, 'cw_winding')
                if self.cw_family in ('ELASTIC', 'CONSTRAINED'):
                    lay.prop(self, 'cw_shape')
                    lay.prop(self, 'cw_branch')
                if self.cw_family == 'CONSTRAINED':
                    lay.prop(self, 'cw_phase')
                lay.prop(self, 'cw_high_wrap')
            if self.preset == 'ELASTICA':
                lay.prop(self, 'elastica_m')
                lay.prop(self, 'elastica_n')
                ratio = self.elastica_m / max(self.elastica_n, 1)
                if not 0.0 < ratio < ELASTICA_RATIO_MAX:
                    lay.label(text="m/n must be < 0.5858 (2-sqrt 2)",
                              icon='ERROR')
            if self.preset in ('WAVY', 'TREFOIL'):
                lay.prop(self, 'lobes')
            if self.preset in ('WAVY', 'TREFOIL', 'BAND'):
                lay.prop(self, 'amp')
            if self.preset == 'ELLIPSE':
                lay.prop(self, 'ecc')
            lay.prop(self, 'shade_smooth')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.hopf_fibration_add",
                             icon='FORCE_MAGNETIC')
        self.layout.operator("mesh.hopf_torus_add",
                             icon='MESH_TORUS')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(CURVE_OT_hopf_fibration_add)
        bpy.utils.register_class(MESH_OT_hopf_torus_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_hopf_torus_add)
        bpy.utils.unregister_class(CURVE_OT_hopf_fibration_add)


def _selftest():
    import numpy as np
    ok_all = True

    # 1) lifted fibre lies on S^3 and its Hopf image is constant (= base)
    def hopf(X):
        a, b, c, d = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        z0z1_re = 2.0 * (a * c + b * d)          # Re(2 z0 conj z1)
        z0z1_im = 2.0 * (b * c - a * d)          # Im(2 z0 conj z1)
        return np.stack([z0z1_re, z0z1_im,
                         a * a + b * b - c * c - d * d], axis=1)

    for base in [(0.3, 0.5, 0.8), (0.0, 0.0, 1.0), (-0.6, 0.4, -0.7)]:
        b = _normalize(base)
        X = fiber_s3(b, 128)
        on_s3 = abs(np.linalg.norm(X, axis=1) - 1.0).max()
        img = hopf(X)
        img_dev = np.linalg.norm(img - img.mean(0), axis=1).max()
        base_err = np.linalg.norm(img.mean(0) - b)
        ok = on_s3 < 1e-12 and img_dev < 1e-9 and base_err < 1e-9
        ok_all = ok_all and ok
        print(f"base={tuple(round(x,2) for x in b)}: |S3 dev|="
              f"{on_s3:.1e} img const={img_dev:.1e} "
              f"base err={base_err:.1e} {'OK' if ok else 'BAD'}")

    # 2) a projected fibre is a genuine circle: coplanar, and an
    #    algebraic circle fit in that plane leaves ~0 radial residual.
    #    (Stereographic projection spaces the samples non-uniformly, so
    #    the circle's centre is NOT the point centroid -- fit it.)
    fib = project_fiber(_normalize((0.4, 0.2, 0.6)), 200)
    ctr = fib.mean(0)
    U, S, Vt = np.linalg.svd(fib - ctr)
    coplanar = S[2] / S[0]                        # ~0 if planar
    u = (fib - ctr) @ Vt[0]                       # in-plane coords
    v = (fib - ctr) @ Vt[1]
    A = np.stack([2 * u, 2 * v, np.ones_like(u)], axis=1)
    sol, *_ = np.linalg.lstsq(A, u * u + v * v, rcond=None)
    cu_, cv_, cc = sol
    r = np.sqrt(np.maximum(0.0, (u - cu_) ** 2 + (v - cv_) ** 2))
    circ = (r.max() - r.min()) / r.mean()
    ok = coplanar < 1e-6 and circ < 1e-6
    ok_all = ok_all and ok
    print(f"projected fibre: coplanarity={coplanar:.1e} "
          f"circularity={circ:.1e} {'OK' if ok else 'BAD'}")

    # 3) every base-point provider yields unit vectors; assembly builds
    for preset in _PROVIDERS:
        pts = base_points(preset, 4, 12, 20.0, 160.0)
        unit = max(abs(np.linalg.norm(p) - 1.0) for p in pts)
        fibers, bases = build_fibers(preset, 3, 12, 60)
        ok = unit < 1e-9 and len(fibers) > 0 and len(fibers) == len(bases)
        ok_all = ok_all and ok
        print(f"{preset}: {len(pts)} base pts unit dev={unit:.1e} "
              f"-> {len(fibers)} fibres {'OK' if ok else 'BAD'}")

    # 4) Hopf torus: curve is unit-norm, un-projected orbit lies on
    #    S^3, mesh has N*M verts / N*M quads, all finite; the enclosed
    #    area of a latitude circle matches 2*pi*(1 - cos beta).
    for preset in ('CIRCLE', 'WAVY', 'ELLIPSE', 'TREFOIL'):
        g = gamma_curve(preset, 120, 90.0, 3, 35.0, 0.5)
        gunit = max(abs(np.linalg.norm(p) - 1.0) for p in g)
        V, F, area = build_hopf_torus(g, 40)
        V = np.asarray(V)
        ok = (gunit < 1e-9 and len(V) == 120 * 40
              and len(F) == 120 * 40 and np.isfinite(V).all())
        ok_all = ok_all and ok
        print(f"torus {preset}: curve unit dev={gunit:.1e} "
              f"verts={len(V)} faces={len(F)} area={area:.3f} "
              f"{'OK' if ok else 'BAD'}")
    # Hopf band: open arc -> annulus with (N-1)*M faces
    gb = gamma_curve('BAND', 100, 90.0, 1, 60.0, 0.0)
    Vb, Fb, _ = build_hopf_torus(gb, 40, closed=False)
    ok = len(Vb) == 100 * 40 and len(Fb) == 99 * 40
    ok_all = ok_all and ok
    print(f"hopf band: verts={len(Vb)} faces={len(Fb)} "
          f"(want {99 * 40}) {'OK' if ok else 'BAD'}")

    # latitude-circle enclosed area check (beta = 60 deg)
    gc = gamma_curve('CIRCLE', 400, 60.0, 1, 0.0, 0.0)
    a_num = _enclosed_area(gc)
    a_exp = 2.0 * pi * (1.0 - math.cos(60.0 * pi / 180.0))
    ok = abs(abs(a_num) - a_exp) < 1e-3
    ok_all = ok_all and ok
    print(f"circle area: num={abs(a_num):.4f} exp={a_exp:.4f} "
          f"{'OK' if ok else 'BAD'}")

    # 5) Willmore energy of the Hopf torus over gamma.  Two exact closed
    #    forms are available: a GREAT circle gives the Clifford torus at
    #    W = 2 pi^2, and a latitude circle at colatitude beta gives
    #    W = 2 pi^2 / sin beta.  (Pinkall 1985, Sect. 4.)
    two_pi2 = 2.0 * pi * pi
    for beta, tol in ((90.0, 1e-9), (60.0, 2e-3), (35.0, 2e-2)):
        g = gamma_curve('CIRCLE', 600, beta, 1, 0.0, 0.0)
        w, L = willmore_energy(g)
        w_exp = two_pi2 / math.sin(beta * pi / 180.0)
        L_exp = 2.0 * pi * math.sin(beta * pi / 180.0)
        ok = abs(w - w_exp) < tol * w_exp and abs(L - L_exp) < 1e-4
        ok_all = ok_all and ok
        print(f"willmore circle beta={beta:5.1f}: W={w:.6f} "
              f"exp={w_exp:.6f} L={L:.5f} {'OK' if ok else 'BAD'}")

    # 6) The spherical elastica closed form.  kappa(s) = sqrt(a) cn(r s)
    #    must solve the Euler-Lagrange equation of Pinkall's functional
    #    F^{lambda} = closed-integral (kappa^2 + lambda) ds on S^2,
    #        2 kappa_ss + kappa^3 + 2 kappa G - lambda kappa = 0 ,
    #    with Gauss curvature G = 1 and lambda = 1, i.e.
    #        2 kappa_ss + kappa^3 + kappa = 0
    #    (Langer & Singer, J. London Math. Soc. 30 (1984), Sect. 1-2).
    for a in (0.6, 3.5, 12.8):
        h = 1e-4
        s = np.array([0.37, 1.1, 2.9, 4.4])
        k0, _, _ = _elastica_kappa(a, s)
        kp, _, _ = _elastica_kappa(a, s + h)
        km, _, _ = _elastica_kappa(a, s - h)
        kss = (kp - 2.0 * k0 + km) / (h * h)
        resid = np.abs(2.0 * kss + k0 ** 3 + k0).max()
        ok = resid < 1e-5 * max(1.0, abs(a) ** 1.5)
        ok_all = ok_all and ok
        print(f"elastica ODE a={a:6.2f}: max|2k_ss+k^3+k|={resid:.2e} "
              f"{'OK' if ok else 'BAD'}")

    # 7) Closed elasticae: the curve closes up, its monodromy is the
    #    requested 2 pi m / n, the geodesic curvature recovered from the
    #    integrated frame matches the closed form, and the resulting
    #    Willmore energy strictly exceeds the Clifford value 2 pi^2
    #    (the Clifford torus is the unique minimum, so every other
    #    Willmore Hopf torus must sit above it).
    for (m, n) in ((1, 3), (2, 5)):
        a = elastica_a_for(m, n)
        raw, _, _, _, _ = _elastica_frames(a, n, 800)
        closure = float(np.linalg.norm(np.asarray(raw[-1])
                                       - np.asarray(raw[0])))
        g = spherical_elastica(m, n, 800)
        theta = _monodromy_angle(a)
        d_theta = abs(theta - 2.0 * pi * m / n)
        kap, _, _ = _geodesic_curvature(g)
        k_max = float(np.abs(kap).max())
        d_k = abs(k_max - sqrt(a))
        w, L = willmore_energy(g)
        ok = (closure < 1e-4 and d_theta < 1e-6
              and d_k < 1e-3 * max(1.0, sqrt(a)) and w > two_pi2)
        ok_all = ok_all and ok
        print(f"elastica {m}/{n}: a={a:8.4f} closure={closure:.1e} "
              f"d(theta)={d_theta:.1e} max|k|={k_max:.4f} "
              f"(exp {sqrt(a):.4f}) W={w / two_pi2:.4f}x2pi^2 "
              f"{'OK' if ok else 'BAD'}")

    # 8) The admissible monodromy window is (0, 2 - sqrt 2): outside it
    #    no closed spherical elastica exists and the solver must say so
    #    rather than return a bogus curve.
    bad = 0
    for (m, n) in ((3, 5), (1, 1), (5, 8)):
        try:
            elastica_a_for(m, n)
        except ValueError:
            bad += 1
    ok = bad == 3
    ok_all = ok_all and ok
    print(f"elastica domain: {bad}/3 out-of-range m/n rejected "
          f"{'OK' if ok else 'BAD'}")

    # 9) The whole pipeline: the ELASTICA preset builds a closed torus.
    ge = gamma_curve('ELASTICA', 240, 90.0, 3, 35.0, 0.5, 1, 3)
    Ve, Fe, area_e = build_hopf_torus(ge, 48)
    Ve = np.asarray(Ve)
    ok = (len(Ve) == 240 * 48 and len(Fe) == 240 * 48
          and np.isfinite(Ve).all())
    ok_all = ok_all and ok
    print(f"torus ELASTICA: verts={len(Ve)} faces={len(Fe)} "
          f"area={area_e:.3f} {'OK' if ok else 'BAD'}")

    # 10) Pinkall's isoperimetric inequality (14) for a closed curve on
    #     S^2 bounding area A with length L:  L^2 - 4 pi A + A^2 >= 0,
    #     with equality exactly for a circle.  (The OCR of the 1985 scan
    #     renders the last term -A^2; +A^2 is what the page image says.)
    #     Every generated gamma must satisfy it, and the CIRCLE preset
    #     must sit on the equality.
    for label, gam in (("circle", gamma_curve('CIRCLE', 800, 55.0,
                                              1, 0.0, 0.0)),
                       ("wavy", gamma_curve('WAVY', 800, 90.0,
                                            3, 35.0, 0.0)),
                       ("elastica 1/3", spherical_elastica(1, 3, 800)),
                       ("elastica 2/5", spherical_elastica(2, 5, 800))):
        L = willmore_energy(gam)[1]
        A = abs(_enclosed_area(gam))
        # relative, because the circle sits ON the equality: there the
        # sign of the defect is decided by the O(h^2) polygon error in L
        # and A, not by the inequality.
        defect = (L * L - 4.0 * pi * A + A * A) / (L * L)
        tight = abs(defect) < 2e-3
        ok = defect > -2e-3 and (tight == (label == "circle"))
        ok_all = ok_all and ok
        print(f"isoperimetric {label:13s}: (L^2-4piA+A^2)/L^2 = "
              f"{defect:9.6f} ({'equality' if tight else 'strict'}) "
              f"{'OK' if ok else 'BAD'}")

    # 11) Pinkall's Proposition 1: the Hopf torus over gamma is a FLAT
    #     torus in S^3, conformally C / Gamma with the lattice
    #         Gamma = < (2 pi, 0), (A/2, L/2) > ,
    #     A the spherical area gamma encloses and L its length.
    #
    #     Both halves are checked on the S^3 torus itself, before the
    #     stereographic projection -- projection is conformal, so it
    #     preserves the conformal type but wrecks the metric, and this is
    #     a statement about the metric.
    #
    #     The halving is not a typo.  The Hopf fibration is a Riemannian
    #     submersion onto a sphere of radius 1/2, while `gamma_curve`
    #     returns curves on the UNIT sphere, so a curve of length L here
    #     lifts horizontally to length L/2 -- which is exactly where
    #     Pinkall's L/2 and A/2 come from.  The lattice therefore has
    #     covolume 2 pi * L/2 = pi L, and that is the sharpest scalar
    #     consequence available: the S^3 area of the torus must be pi L.
    for label, gam in (("circle b=90", gamma_curve('CIRCLE', 600, 90.0,
                                                   1, 0.0, 0.0)),
                       ("circle b=55", gamma_curve('CIRCLE', 600, 55.0,
                                                   1, 0.0, 0.0)),
                       ("wavy", gamma_curve('WAVY', 600, 90.0, 3,
                                            35.0, 0.0)),
                       ("elastica 1/3", spherical_elastica(1, 3, 600))):
        G = np.asarray(gam, dtype=float)
        L = willmore_energy(G)[1]
        M = 96
        psi = np.linspace(0.0, 2.0 * pi, M, endpoint=False)
        cp, sp = np.cos(psi), np.sin(psi)
        lift = np.empty((len(G), M, 4))
        for i, g in enumerate(G):
            x, y, z = g
            beta = math.acos(max(-1.0, min(1.0, z)))
            lam = math.atan2(y, x)
            cb, sb = math.cos(beta / 2.0), math.sin(beta / 2.0)
            z0r, z0i = cb * math.cos(lam), cb * math.sin(lam)
            lift[i] = np.stack([cp * z0r - sp * z0i, sp * z0r + cp * z0i,
                                cp * sb, sp * sb], axis=1)
        # metric in the (s, psi) grid, with s the arclength of gamma
        du = np.roll(lift, -1, axis=0) - lift
        dv = np.roll(lift, -1, axis=1) - lift
        E = np.einsum('ijk,ijk->ij', du, du)
        F = np.einsum('ijk,ijk->ij', du, dv)
        Gm = np.einsum('ijk,ijk->ij', dv, dv)
        area = float(np.sum(np.sqrt(np.maximum(E * Gm - F * F, 0.0))))
        want = pi * L
        rel = abs(area - want) / want
        ok = rel < 5e-3
        ok_all = ok_all and ok
        print(f"Pinkall lattice {label:13s}: S^3 area {area:.5f} vs "
              f"covolume pi L = {want:.5f} (L={L:.4f}) rel {rel:.1e} "
              f"{'OK' if ok else 'BAD'}")

    # 12) THE acceptance gate.  Everything above tests gamma; this tests
    #     the SURFACE, and against an independent implementation.
    #     math_art.solver.willmore carries the discrete Willmore energy
    #     int H^2 dA and its exact analytic first variation.  Pinkall's
    #     theorem says the Hopf torus over a closed elastica is a
    #     critical point of that functional, so the NORMAL component of
    #     the gradient must vanish -- and since the discrete gradient
    #     carries O(h^2) truncation error, the real signature is that it
    #     vanishes UNDER REFINEMENT.  A non-elastic gamma (WAVY) is the
    #     control: its residual must plateau instead.
    #     (Only the normal component is meaningful: the discrete energy
    #     is not invariant under tangential vertex motion, so tangential
    #     gradient components are a property of the mesh, not the shape.)
    from .solver.willmore import willmore_gradient, vertex_area_data

    def _criticality(gamma, m_psi):
        verts, quads, _ = build_hopf_torus(gamma, m_psi)
        V = np.asarray(verts, dtype=float)
        T = np.empty((2 * len(quads), 3), dtype=np.int64)
        Q = np.asarray(quads, dtype=np.int64)
        T[0::2] = Q[:, [0, 1, 2]]
        T[1::2] = Q[:, [0, 2, 3]]
        E, grad = willmore_gradient(V, T)
        g_area, av, nvec = vertex_area_data(V, T)
        av = np.maximum(av, 1e-300)
        nl = np.maximum(np.linalg.norm(nvec, axis=1), 1e-300)
        gn = np.einsum('ij,ij->i', grad, nvec / nl[:, None])
        # discrete L^2 norm of delta(int H^2 dA) / delta n
        return E, float(np.sqrt(np.sum(gn * gn / av)))

    for label, gam in (
            ("elastica 1/3", lambda k: spherical_elastica(1, 3, k)),
            ("circle (Clifford)",
             lambda k: gamma_curve('CIRCLE', k, 90.0, 1, 0.0, 0.0)),
            ("wavy (control)",
             lambda k: gamma_curve('WAVY', k, 90.0, 3, 35.0, 0.0))):
        E1, r1 = _criticality(gam(200), 56)
        E2, r2 = _criticality(gam(400), 112)
        rate = math.log2(r1 / r2) if r2 > 0 else 99.0
        critical = rate > 1.2                # residual -> 0 like h^2
        want = label != "wavy (control)"
        ok = critical == want
        ok_all = ok_all and ok
        print(f"willmore criticality {label:18s}: |dW/dn| {r1:.3e} -> "
              f"{r2:.3e} rate {rate:+.2f} -> "
              f"{'CRITICAL' if critical else 'not critical'} "
              f"{'OK' if ok else 'BAD'}")
        # the mesh energy must also agree with the curve integral
        w_curve = willmore_energy(gam(400))[0]
        rel = abs(E2 - w_curve) / w_curve
        ok = rel < 0.02
        ok_all = ok_all and ok
        print(f"   int H^2 dA = {E2:.4f} vs pi*int(1+k^2)ds = "
              f"{w_curve:.4f}  rel {rel:.2e} {'OK' if ok else 'BAD'}")

    assert ok_all
    print("hopf fibration standalone tests passed")
