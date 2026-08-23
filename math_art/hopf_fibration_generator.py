
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

def fiber_s3(base, samples, P=1, Q=1, chirality='RIGHT'):
    """The fibre over unit 3-vector `base` = (a,b,c) on S^2, sampled
    at `samples` angles, as an (samples, 4) array on S^3.  With P=Q=1
    this is the Hopf fibre (a great circle); general (P,Q) winds the
    two complex phases at those integer rates, giving a (P,Q) torus
    curve on the same Clifford torus.  `chirality='LEFT'` conjugates the
    second phase (z1 -> conj z1), giving the MIRROR fibration -- the
    other Villarceau ruling of the same tori."""
    import numpy as np
    a, b, c = base
    beta = math.acos(max(-1.0, min(1.0, c)))   # colatitude in [0, pi]
    lam = math.atan2(b, a)                      # longitude
    ch, sh = math.cos(beta / 2.0), math.sin(beta / 2.0)
    t = np.linspace(0.0, 2.0 * pi, samples, endpoint=False)
    p0 = P * t + lam
    p1 = Q * t
    sgn = -1.0 if chirality == 'LEFT' else 1.0
    return np.stack([ch * np.cos(p0), ch * np.sin(p0),
                     sh * np.cos(p1), sgn * sh * np.sin(p1)], axis=1)


def _quat_left(phi):
    """Unit quaternion (cos phi, sin phi, 0, 0) -- one plane of a
    left-isoclinic (Clifford) rotation of S^3."""
    return (math.cos(phi), math.sin(phi), 0.0, 0.0)


def _s3_rotate(X, q):
    """Left quaternion-multiply every S^3 point (row of X) by q.  A left
    Clifford rotation commutes with the Hopf action, so it descends to a
    rotation of the base S^2 -- spinning the whole fibre family through
    each other (the 'flow' / cyclide-morph motion)."""
    import numpy as np
    w, i, j, k = q
    a, b, c, d = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.stack([w * a - i * b - j * c - k * d,
                     w * b + i * a + j * d - k * c,
                     w * c - i * d + j * a + k * b,
                     w * d + i * c - j * b + k * a], axis=1)


def _so4(a1, a2, a3):
    """A double (isoclinic-ish) rotation of R^4: rotate the (x1,x2) plane
    by a1, the (x3,x4) plane by a2, and the (x1,x3) plane by a3.  Applied
    before stereographic projection it morphs a Hopf torus through the
    ring / horn / spindle tori and Dupin cyclides."""
    import numpy as np
    def rot(i, j, ang):
        M = np.eye(4)
        c, s = math.cos(ang), math.sin(ang)
        M[i, i] = c; M[i, j] = -s; M[j, i] = s; M[j, j] = c
        return M
    return rot(0, 1, a1) @ rot(2, 3, a2) @ rot(0, 2, a3)


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


def spherical_cap(n, colat_min_deg, colat_max_deg):
    """`n` base points spiral-filling the spherical zone between
    colatitudes `colat_min` and `colat_max` (a Fibonacci spiral inside
    a cap/band): the Segerman-print composition -- a tight twisted core
    opening into ever-larger loops."""
    zmax = math.cos(colat_min_deg * pi / 180.0)
    zmin = math.cos(colat_max_deg * pi / 180.0)
    ga = pi * (3.0 - sqrt(5.0))
    pts = []
    for k in range(max(1, n)):
        z = zmin + (zmax - zmin) * (k + 0.5) / max(1, n)
        r = sqrt(max(0.0, 1.0 - z * z))
        ph = ga * k
        pts.append((r * math.cos(ph), r * math.sin(ph), z))
    return pts


def loxodrome(n, colat_min_deg, colat_max_deg, turns):
    """`n` base points along a rhumb line (loxodrome): colatitude sweeps
    the band while longitude winds `turns` times -- a nested spiral of
    Villarceau circles (mwalczyk's mode)."""
    pts = []
    for k in range(max(1, n)):
        f = (k + 0.5) / max(1, n)
        beta = (colat_min_deg + (colat_max_deg - colat_min_deg) * f) * pi / 180.0
        lam = turns * 2.0 * pi * f
        pts.append((math.sin(beta) * math.cos(lam),
                    math.sin(beta) * math.sin(lam), math.cos(beta)))
    return pts


def random_sphere(n, seed):
    """`n` base points drawn uniformly at random on S^2 (fixed `seed`
    so the redo panel is reproducible)."""
    import numpy as np
    rng = np.random.default_rng(int(seed))
    v = rng.standard_normal((max(1, n), 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return [tuple(x) for x in v]


def curl(n, colat_deg, lobes, amp_deg):
    """`n` base points on a rose curve beta = b0 + a cos(k s) -- a single
    floral closed ring (mwalczyk's 'curl')."""
    pts = []
    for k in range(max(1, n)):
        s = 2.0 * pi * k / max(1, n)
        beta = min(pi - 0.06, max(0.06,
                   colat_deg * pi / 180.0
                   + amp_deg * pi / 180.0 * math.cos(lobes * s)))
        pts.append((math.sin(beta) * math.cos(s),
                    math.sin(beta) * math.sin(s), math.cos(beta)))
    return pts


_PROVIDERS = {
    'LATITUDES': "Nested tori (circles of latitude)",
    'FLOWER': "Flower (one small circle of base points)",
    'GREATCIRCLE': "Great-circle band",
    'CAP': "Spherical cap / band spiral (Segerman patch)",
    'LOXODROME': "Loxodrome (rhumb-line spiral)",
    'CURL': "Curl (floral rose ring)",
    'FIBONACCI': "Fibonacci sphere (quasi-uniform)",
    'RANDOM': "Random points on S^2",
    'TETRA': "Tetrahedron vertices",
    'OCTA': "Octahedron vertices",
    'CUBE': "Cube vertices",
    'ICOSA': "Icosahedron vertices",
    'DODECA': "Dodecahedron vertices",
}

# presets whose base points form continuous closed ring(s) -- these can
# be skinned into surfaces (SURFACE output) and carry lat_min/lat_max
_RING_PRESETS = ('LATITUDES', 'FLOWER', 'GREATCIRCLE', 'CAP',
                 'LOXODROME', 'CURL')


def base_points(preset, n_lat, n_fiber, lat_min, lat_max, extra=None):
    """Dispatch to a base-point provider; returns a list of unit
    3-vectors on S^2 (before the generic tilt is applied).  `extra`
    carries the few preset-specific knobs (turns, seed, curl lobes)."""
    ex = extra or {}
    if preset == 'LATITUDES':
        return latitudes(n_lat, n_fiber, lat_min, lat_max)
    if preset == 'FLOWER':
        return flower(n_fiber, 0.5 * (lat_min + lat_max))
    if preset == 'GREATCIRCLE':
        return great_circle(n_fiber, 0.5 * (lat_min + lat_max))
    if preset == 'CAP':
        return spherical_cap(n_fiber, lat_min, lat_max)
    if preset == 'LOXODROME':
        return loxodrome(n_fiber, lat_min, lat_max, ex.get('turns', 5.0))
    if preset == 'CURL':
        return curl(n_fiber, 0.5 * (lat_min + lat_max),
                    ex.get('curl_lobes', 5), ex.get('curl_amp', 25.0))
    if preset == 'FIBONACCI':
        return fibonacci(max(1, n_fiber))
    if preset == 'RANDOM':
        return random_sphere(n_fiber, ex.get('seed', 0))
    return _platonic(preset)


def base_rings(preset, n_lat, n_fiber, lat_min, lat_max, extra=None):
    """The base points grouped into closed rings (for SURFACE output),
    or None when the preset has no ring adjacency to skin."""
    import numpy as np
    if preset == 'LATITUDES':
        rings = []
        if n_lat == 1:
            betas = [0.5 * (lat_min + lat_max) * pi / 180.0]
        else:
            betas = np.linspace(lat_min, lat_max, n_lat) * pi / 180.0
        for beta in betas:
            rings.append([(math.sin(beta) * math.cos(2.0 * pi * k / n_fiber),
                           math.sin(beta) * math.sin(2.0 * pi * k / n_fiber),
                           math.cos(beta)) for k in range(n_fiber)])
        return rings
    if preset in ('FLOWER', 'GREATCIRCLE', 'CAP', 'LOXODROME', 'CURL'):
        return [base_points(preset, n_lat, n_fiber, lat_min, lat_max, extra)]
    return None


# --------------------------------------------------------------------------
# Assembly: project every fibre, tilt/centre/scale to fit the unit cube
# --------------------------------------------------------------------------

def _longest_run(mask):
    """Indices of the longest contiguous True run in a cyclic boolean
    mask (used to keep the in-range arc of a near-axis fibre)."""
    import numpy as np
    n = len(mask)
    if mask.all():
        return np.arange(n)
    start = int(np.argmin(mask))               # first False
    order = np.roll(np.arange(n), -start)
    m = mask[order]
    best = (0, 0)
    i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return order[best[0]:best[1]]


def build_fibers(preset='LATITUDES', n_lat=6, n_fiber=24, samples=160,
                 P=1, Q=1, lat_min=20.0, lat_max=160.0,
                 fit_radius=1.0, max_radius=12.0,
                 sphere_euler=None, s3_rot=0.0, chirality='RIGHT',
                 include_axis=False, extra=None, return_stats=False):
    """Return (fibers, bases): `fibers` is a list of (n, 3) projected
    polylines and `bases` the matching (rotated) base points for
    colouring.  Centred at the origin and uniformly scaled so the
    95th-percentile point radius is `fit_radius`.

    `sphere_euler` orients S^2 (defaults to the built-in tilt); `s3_rot`
    (degrees) applies a left Clifford rotation of S^3 before projecting
    (the flow / morph); `chirality` in RIGHT/LEFT/BOTH selects the
    Villarceau ruling(s).  A fibre exceeding `max_radius` (base near the
    south pole) is dropped, unless `include_axis`, when its in-range arc
    is kept as an OPEN polyline (the axis fibre as a clipped line).

    With `return_stats` the return is (fibers, bases, closed, stats)
    where `closed[i]` says whether fibre i is a full loop and `stats`
    reports how many fibres were dropped."""
    import numpy as np
    R = _rot_matrix(*(sphere_euler if sphere_euler is not None else _TILT))
    raw = base_points(preset, n_lat, n_fiber, lat_min, lat_max, extra)
    based = [tuple(R @ np.asarray(b)) for b in raw]
    q = _quat_left(math.radians(s3_rot)) if s3_rot else None
    chis = ('RIGHT', 'LEFT') if chirality == 'BOTH' else (chirality,)

    fibers, bases, closed, dropped = [], [], [], 0
    for b in based:
        for chi in chis:
            X = fiber_s3(b, samples, P, Q, chi)
            if q is not None:
                X = _s3_rotate(X, q)
            p = stereographic(X)
            if not np.isfinite(p).all():
                dropped += 1
                continue
            r = np.linalg.norm(p, axis=1)
            if r.max() < max_radius:
                fibers.append(p)
                bases.append(b)
                closed.append(True)
            elif include_axis:
                run = _longest_run(r < max_radius)
                if len(run) >= 2:
                    fibers.append(p[run])
                    bases.append(b)
                    closed.append(False)
                else:
                    dropped += 1
            else:
                dropped += 1

    if not fibers:
        return ([], [], [], {'dropped': dropped}) if return_stats else ([], [])

    allpts = np.concatenate(fibers, axis=0)
    center = 0.5 * (allpts.max(0) + allpts.min(0))
    rad = np.linalg.norm(allpts - center, axis=1)
    ref = np.percentile(rad, 95.0)
    scale = (fit_radius / ref) if ref > 1e-9 else 1.0
    fibers = [(p - center) * scale for p in fibers]
    if return_stats:
        return fibers, bases, closed, {'dropped': dropped}
    return fibers, bases


def _fiber_color(base):
    """Standard sphere colouring of a base point: hue from longitude,
    value from latitude (north pole light, south pole dark)."""
    import colorsys
    a, b, c = base
    hue = (math.atan2(b, a) / (2.0 * pi)) % 1.0
    val = 0.35 + 0.6 * (0.5 * (c + 1.0))       # c in [-1,1] -> val
    return colorsys.hsv_to_rgb(hue, 0.72, val)


def _palette_rgb(base, style='RAINBOW', mono=(0.27, 0.86, 0.80)):
    """Colour a base point under a chosen palette.  All three read the
    base point on S^2 -- hue from longitude, lightness from latitude --
    but with different aesthetics: RAINBOW (saturated, black-bg render),
    PASTEL (soft two-axis ramp, the 3D-print look), MONO (one hue,
    lightness by latitude)."""
    import colorsys
    a, b, c = base
    hue = (math.atan2(b, a) / (2.0 * pi)) % 1.0
    lat = 0.5 * (c + 1.0)                        # 1 north .. 0 south
    if style == 'PASTEL':
        return colorsys.hsv_to_rgb(hue, 0.40, 0.72 + 0.24 * lat)
    if style == 'MONO':
        h, s, _v = colorsys.rgb_to_hsv(*mono)
        return colorsys.hsv_to_rgb(h, s, 0.42 + 0.5 * lat)
    return colorsys.hsv_to_rgb(hue, 0.85, 0.35 + 0.55 * lat)


def _param_rgb(t01, style='RAINBOW', mono=(0.27, 0.86, 0.80)):
    """PARAMETER colouring: hue swept by position `t01` along the fibre,
    revealing the flow direction."""
    import colorsys
    if style == 'MONO':
        h, s, _v = colorsys.rgb_to_hsv(*mono)
        return colorsys.hsv_to_rgb(h, s, 0.40 + 0.5 * t01)
    sat = 0.42 if style == 'PASTEL' else 0.85
    return colorsys.hsv_to_rgb(t01 % 1.0, sat, 0.9)


# --------------------------------------------------------------------------
# Small geometry primitives (beads, markers, control sphere)
# --------------------------------------------------------------------------

def _unit3(p):
    m = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) or 1.0
    return (p[0] / m, p[1] / m, p[2] / m)


def _icosphere(radius=1.0, subdiv=1):
    """(verts, faces) of an icosphere, verts as tuples."""
    t = _PHI
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
         (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    v = [list(_unit3(p)) for p in v]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    for _ in range(subdiv):
        mid = {}
        nf = []

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in mid:
                m = _unit3(((v[i][0] + v[j][0]) / 2.0,
                            (v[i][1] + v[j][1]) / 2.0,
                            (v[i][2] + v[j][2]) / 2.0))
                mid[key] = len(v)
                v.append(list(m))
            return mid[key]
        for a, b, c in f:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return [(x * radius, y * radius, z * radius) for x, y, z in v], f


def _uv_sphere(radius=1.0, rings=14, segs=22):
    verts, faces = [], []
    for i in range(rings + 1):
        th = pi * i / rings
        for j in range(segs):
            ph = 2.0 * pi * j / segs
            verts.append((radius * math.sin(th) * math.cos(ph),
                          radius * math.sin(th) * math.sin(ph),
                          radius * math.cos(th)))
    for i in range(rings):
        for j in range(segs):
            j1 = (j + 1) % segs
            a = i * segs + j
            b = i * segs + j1
            c = (i + 1) * segs + j1
            d = (i + 1) * segs + j
            faces.append([a, b, c, d])
    return verts, faces


def _cone(radius=1.0, height=2.0, seg=8):
    """A cone with apex at +z*height, base ring at z=0, as (verts, faces)."""
    verts = [(0.0, 0.0, height)]
    for j in range(seg):
        ph = 2.0 * pi * j / seg
        verts.append((radius * math.cos(ph), radius * math.sin(ph), 0.0))
    faces = [[0, 1 + j, 1 + (j + 1) % seg] for j in range(seg)]
    faces.append([1 + j for j in range(seg)][::-1])       # base cap
    return verts, faces


def _resample_closed(P, count):
    """Arc-length-uniform resample of a closed polyline P to `count`
    points (returns an (count,3) array)."""
    import numpy as np
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-12:
        return np.repeat(P[:1], count, axis=0)
    targ = np.linspace(0.0, total, count, endpoint=False)
    out = np.empty((count, 3))
    Pw = np.vstack([P, P[:1]])
    for k, tv in enumerate(targ):
        i = int(np.searchsorted(s, tv) - 1)
        i = max(0, min(i, len(seg) - 1))
        f = (tv - s[i]) / (seg[i] if seg[i] > 1e-12 else 1.0)
        out[k] = Pw[i] * (1 - f) + Pw[i + 1] * f
    return out


def _place(template_v, template_f, xform, base_v, base_f):
    """Append a transformed copy of a template mesh to (base_v, base_f).
    `xform` maps a template vertex (tuple) to a world tuple."""
    off = len(base_v)
    base_v.extend(xform(p) for p in template_v)
    base_f.extend([[off + i for i in fc] for fc in template_f])


def _closed_tube(P, radius, sides):
    """A swept tube along a CLOSED polyline P, with parallel-transported
    frames whose closure holonomy is distributed so the seam matches.
    Vertex layout is ring-major: vertex i*sides + j is sample i, side j
    -- so the caller can colour by fibre parameter i."""
    import numpy as np
    P = np.asarray(P, float)
    m = len(P)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, T[0])) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    N = np.empty_like(P)
    N[0] = np.cross(T[0], ref)
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, m):
        v = N[i - 1] - T[i] * np.dot(N[i - 1], T[i])
        N[i] = v / (np.linalg.norm(v) or 1.0)
    v = N[m - 1] - T[0] * np.dot(N[m - 1], T[0])
    v /= (np.linalg.norm(v) or 1.0)
    B0 = np.cross(T[0], N[0])
    ang = math.atan2(float(np.dot(v, B0)), float(np.dot(v, N[0])))
    verts, faces = [], []
    for i in range(m):
        corr = -ang * i / m
        B = np.cross(T[i], N[i])
        for j in range(sides):
            a = 2.0 * pi * j / sides + corr
            verts.append(tuple(P[i] + radius
                               * (math.cos(a) * N[i] + math.sin(a) * B)))
    for i in range(m):
        i1 = (i + 1) % m
        for j in range(sides):
            j1 = (j + 1) % sides
            faces.append([i * sides + j, i * sides + j1,
                          i1 * sides + j1, i1 * sides + j])
    return verts, faces


def _open_tube(P, radius, sides):
    """A swept tube along an OPEN polyline P (no end-to-start closure),
    with parallel-transported frames -- for the clipped axis fibre."""
    import numpy as np
    P = np.asarray(P, float)
    m = len(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    N = np.cross(T[0], ref)
    if np.linalg.norm(N) < 1e-6:
        N = np.cross(T[0], np.array([1.0, 0.0, 0.0]))
    N /= np.linalg.norm(N)
    frames = [N]
    for i in range(1, m):
        v = frames[-1] - T[i] * np.dot(frames[-1], T[i])
        nn = np.linalg.norm(v)
        frames.append(v / nn if nn > 1e-9 else frames[-1])
    verts, faces = [], []
    for i in range(m):
        B = np.cross(T[i], frames[i])
        for j in range(sides):
            a = 2.0 * pi * j / sides
            verts.append(tuple(P[i] + radius
                               * (math.cos(a) * frames[i]
                                  + math.sin(a) * B)))
    for i in range(m - 1):
        for j in range(sides):
            j1 = (j + 1) % sides
            faces.append([i * sides + j, i * sides + j1,
                          (i + 1) * sides + j1, (i + 1) * sides + j])
    return verts, faces


# --------------------------------------------------------------------------
# Fibre surface: skin the fibres over each continuous base ring into the
# nested tori (the "surface" companion to the fibre bundle)
# --------------------------------------------------------------------------

def build_fiber_surface(rings, samples, P=1, Q=1, sphere_euler=None,
                        s3_rot=0.0, chirality='RIGHT', fit_radius=1.0):
    """Skin the fibres over each closed base ring into a quad torus.
    `rings` is a list of closed rings of base points (from
    `base_rings`).  Returns (verts, faces, vbase) with one base point per
    vertex (for colouring); the whole set is centred and scaled together
    so the nested tori share a frame."""
    import numpy as np
    R = _rot_matrix(*(sphere_euler if sphere_euler is not None else _TILT))
    q = _quat_left(math.radians(s3_rot)) if s3_rot else None
    chi = 'RIGHT' if chirality == 'BOTH' else chirality
    verts, faces, vbase = [], [], []
    for ring in rings:
        base_off = len(verts)
        Nr = len(ring)
        for b in ring:
            bb = tuple(R @ np.asarray(b))
            X = fiber_s3(bb, samples, P, Q, chi)
            if q is not None:
                X = _s3_rotate(X, q)
            pts = stereographic(X)
            verts.extend(pts.tolist())
            vbase.extend([bb] * samples)
        for i in range(Nr):
            i1 = (i + 1) % Nr
            for j in range(samples):
                j1 = (j + 1) % samples
                faces.append([base_off + i * samples + j,
                              base_off + i * samples + j1,
                              base_off + i1 * samples + j1,
                              base_off + i1 * samples + j])
    V = np.asarray(verts)
    center = 0.5 * (V.max(0) + V.min(0))
    rad = np.linalg.norm(V - center, axis=1)
    ref = np.percentile(rad, 95.0)
    scale = (fit_radius / ref) if ref > 1e-9 else 1.0
    V = (V - center) * scale
    return V.tolist(), faces, vbase


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
                     max_radius=40.0, R4=None):
    """Mesh the Hopf torus h^{-1}(gamma): each of the N curve samples
    contributes a fibre circle of `m_psi` points (the Hopf orbit of a
    lift), joined into a quad grid.  `closed` wraps the curve direction
    into a torus; `closed=False` leaves an open annulus (a Hopf band).
    An optional `R4` (4x4 SO(4) matrix) rotates the torus in S^3 before
    projecting -- morphing it through the ring/horn/spindle tori and
    Dupin cyclides.  Returns (verts, faces, area) with verts centred and
    scaled to the unit cube."""
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
        if R4 is not None:
            X = X @ R4.T
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

    def _color_material(name, emission):
        """A single material whose Base Colour (and, if `emission` > 0,
        its emission) is driven by the mesh's `hopf_color` attribute."""
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        vc = nt.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = "hopf_color"
        if bsdf:
            nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
            if emission > 0.0:
                if "Emission Color" in bsdf.inputs:
                    nt.links.new(vc.outputs["Color"],
                                 bsdf.inputs["Emission Color"])
                if "Emission Strength" in bsdf.inputs:
                    bsdf.inputs["Emission Strength"].default_value = emission
        return mat

    def _finish(context, obj):
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location
        return obj

    class CURVE_OT_hopf_fibration_add(bpy.types.Operator):
        """Add fibres of the Hopf fibration of S^3, stereographically
        projected to R^3 -- as tubes, a nested-tori surface, or beads --
        coloured by base point, with a control sphere and S^3 motion"""
        bl_idname = "curve.hopf_fibration_add"
        bl_label = "Hopf Fibration"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Base Points",
            items=[(k, k.title(), v) for k, v in _PROVIDERS.items()],
            description="Set of base points on S^2 whose fibres are drawn",
            default='LATITUDES')
        n_lat: IntProperty(
            name="Latitudes / Rings", default=6, min=1, max=48,
            description="Circles of latitude (LATITUDES)")
        n_fiber: IntProperty(
            name="Fibres", default=24, min=1, max=400,
            description="Fibres per latitude (LATITUDES), or the total "
                        "number of fibres for the other point sets")
        samples: IntProperty(
            name="Samples", default=160, min=12, max=1024,
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
            description="Smallest colatitude (deg) of the base band")
        lat_max: FloatProperty(
            name="Colat Max", default=160.0, min=1.0, max=179.0,
            description="Largest colatitude (deg) of the base band")
        turns: FloatProperty(
            name="Turns", default=5.0, min=0.5, max=40.0,
            description="Longitude turns of the loxodrome spiral")
        seed: IntProperty(
            name="Seed", default=0, min=0, max=9999,
            description="Random seed for the RANDOM base set")
        curl_lobes: IntProperty(
            name="Curl Lobes", default=5, min=2, max=16,
            description="Petal count of the CURL rose ring")
        curl_amp: FloatProperty(
            name="Curl Amplitude", default=25.0, min=0.0, max=80.0,
            description="Colatitude swing of the CURL rose (deg)")
        chirality: EnumProperty(
            name="Chirality",
            items=[('RIGHT', "Right", "the right-handed fibration"),
                   ('LEFT', "Left", "the mirror (left) fibration -- the "
                    "other Villarceau ruling"),
                   ('BOTH', "Both", "both rulings woven together")],
            description="Which Villarceau ruling(s) to draw",
            default='RIGHT')
        output: EnumProperty(
            name="Output",
            items=[('BEZIER', "Bezier Curve", "auto-smoothed curve"),
                   ('POLY', "Poly Curve", "polyline curve"),
                   ('NURBS', "NURBS Curve", "NURBS curve"),
                   ('MESH', "Mesh Tube", "swept tube mesh per fibre"),
                   ('SURFACE', "Surface", "skin the fibres over each "
                    "continuous base ring into the nested torus "
                    "surface(s)"),
                   ('BEADS', "Beads", "a string of spheres along each "
                    "fibre (the ball-and-stick look)")],
            description="How the fibres are realised",
            default='MESH')
        radius: FloatProperty(
            name="Tube Radius", default=0.02, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius")
        resolution: IntProperty(name="Bevel Resolution", default=4,
                                min=1, max=16,
                                description="Smoothness of the round bevel "
                                            "on curve output")
        tube_sides: IntProperty(name="Tube Sides", default=8,
                                min=3, max=32,
                                description="Sides around each swept tube")
        bead_count: IntProperty(
            name="Beads / Fibre", default=48, min=4, max=400,
            description="Spheres placed along each fibre (Beads output)")
        bead_radius: FloatProperty(
            name="Bead Radius", default=0.03, min=0.001, max=0.5,
            step=1, precision=3,
            description="Radius of each bead (Beads output)")
        markers: IntProperty(
            name="Markers / Fibre", default=0, min=0, max=24,
            description="Cone glyphs riding each fibre, pointing along "
                        "it (0 = none) -- the base-point markers")
        marker_size: FloatProperty(
            name="Marker Size", default=0.05, min=0.005, max=0.5,
            step=1, precision=3, description="Size of the marker cones")
        color_fibers: BoolProperty(
            name="Colour by Base Point", default=True,
            description="Colour each fibre from its base point on S^2")
        color_style: EnumProperty(
            name="Palette",
            items=[('RAINBOW', "Rainbow", "saturated hue by longitude "
                    "(black-background render)"),
                   ('PASTEL', "Pastel", "soft two-axis ramp (the 3D-"
                    "print look)"),
                   ('MONO', "Mono", "one hue, lightness by latitude"),
                   ('PARAMETER', "Parameter", "hue swept along each "
                    "fibre, showing the flow direction")],
            description="Colour aesthetic (all keyed to the base point)",
            default='RAINBOW')
        mono_color: bpy.props.FloatVectorProperty(
            name="Mono Colour", subtype='COLOR', size=3,
            min=0.0, max=1.0, default=(0.27, 0.86, 0.80),
            description="Base hue for the Mono palette")
        emission: FloatProperty(
            name="Glow", default=0.0, min=0.0, max=20.0,
            description="Emission strength (0 = matte; raise for the "
                        "glowing-on-black look)")
        sphere_euler: bpy.props.FloatVectorProperty(
            name="Sphere Rotation", subtype='EULER', size=3,
            default=_TILT,
            description="Orientation of the base sphere S^2 (default is "
                        "a gentle tilt keeping fibres off the axis)")
        s3_rot: FloatProperty(
            name="S3 Flow", default=0.0, min=-360.0, max=360.0,
            description="Left Clifford rotation of S^3 before projecting "
                        "(deg) -- keyframe it to spin the whole family")
        include_axis: BoolProperty(
            name="Keep Axis Fibres", default=False,
            description="Keep near-pole fibres as clipped open lines "
                        "(the axis fibre) instead of dropping them")
        show_base_sphere: BoolProperty(
            name="Control Sphere", default=False,
            description="Add a small S^2 beside the fibres with a dot "
                        "per base point in its fibre's colour")
        sphere_size: FloatProperty(
            name="Sphere Size", default=0.5, min=0.05, max=3.0,
            description="Radius of the control sphere (scene units)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the fibre set")

        # ---- colour helpers -------------------------------------------
        def _rgb(self, base):
            return _palette_rgb(base, self.color_style, tuple(self.mono_color))

        def _fiber_vcolors(self, base, n_ring, sides_or_none):
            """rgba per vertex for one fibre.  PARAMETER varies along the
            fibre; the others are constant at the base colour."""
            if self.color_style == 'PARAMETER':
                cols = []
                for i in range(n_ring):
                    rgb = _param_rgb(i / max(1, n_ring), 'RAINBOW',
                                     tuple(self.mono_color))
                    reps = sides_or_none if sides_or_none else 1
                    cols.extend([(*rgb, 1.0)] * reps)
                return cols
            rgb = self._rgb(base)
            per = (sides_or_none or 1) * n_ring
            return [(*rgb, 1.0)] * per

        # ---- output builders ------------------------------------------
        def _build_tubes(self, fibers, bases, closed):
            verts, faces, vcol = [], [], []
            for Pl, b, cl in zip(fibers, bases, closed):
                if cl:
                    v, f = _closed_tube(Pl, self.radius, self.tube_sides)
                else:
                    v, f = _open_tube(Pl, self.radius, self.tube_sides)
                off = len(verts)
                verts.extend(v)
                faces.extend([[off + i for i in fc] for fc in f])
                vcol.extend(self._fiber_vcolors(b, len(Pl), self.tube_sides))
            return verts, faces, vcol

        def _build_beads(self, fibers, bases):
            import numpy as np
            iv, ifc = _icosphere(self.bead_radius, 1)
            verts, faces, vcol = [], [], []
            for Pl, b in zip(fibers, bases):
                pts = _resample_closed(Pl, self.bead_count)
                for k, c in enumerate(pts):
                    if self.color_style == 'PARAMETER':
                        rgb = _param_rgb(k / max(1, self.bead_count),
                                         'RAINBOW', tuple(self.mono_color))
                    else:
                        rgb = self._rgb(b)
                    off = len(verts)
                    verts.extend((c[0] + p[0], c[1] + p[1], c[2] + p[2])
                                 for p in iv)
                    faces.extend([[off + i for i in fc] for fc in ifc])
                    vcol.extend([(*rgb, 1.0)] * len(iv))
            return verts, faces, vcol

        def _add_markers(self, fibers, bases, verts, faces, vcol):
            import numpy as np
            cv, cf = _cone(self.marker_size * 0.5, self.marker_size, 8)
            for Pl, b in zip(fibers, bases):
                P = np.asarray(Pl)
                m = len(P)
                rgb = self._rgb(b)
                for s in range(self.markers):
                    i = int(round(s * m / self.markers)) % m
                    t = P[(i + 1) % m] - P[i - 1]
                    nt = np.linalg.norm(t)
                    if nt < 1e-9:
                        continue
                    t = t / nt
                    up = np.array([0.0, 0.0, 1.0])
                    if abs(np.dot(up, t)) > 0.9:
                        up = np.array([1.0, 0.0, 0.0])
                    nx = np.cross(up, t)
                    nx /= np.linalg.norm(nx)
                    ny = np.cross(t, nx)
                    off = len(verts)
                    for p in cv:
                        w = P[i] + p[0] * nx + p[1] * ny + p[2] * t
                        verts.append((w[0], w[1], w[2]))
                    faces.extend([[off + i2 for i2 in fc] for fc in cf])
                    vcol.extend([(*rgb, 1.0)] * len(cv))

        def _control_sphere(self, context, bases, parent, extent):
            import numpy as np
            r = self.sphere_size
            cx = extent + r + 0.2 * max(extent, 1.0)
            sv, sf = _uv_sphere(r, 14, 22)
            verts = [(x + cx, y, z) for x, y, z in sv]
            faces = [list(f) for f in sf]
            vcol = [(0.16, 0.17, 0.2, 1.0)] * len(sv)
            dv, dfc = _icosphere(0.06 * r + 0.01, 1)
            for b in bases:
                rgb = self._rgb(b)
                c = (b[0] * r * 1.02 + cx, b[1] * r * 1.02, b[2] * r * 1.02)
                off = len(verts)
                verts.extend((c[0] + p[0], c[1] + p[1], c[2] + p[2])
                             for p in dv)
                faces.extend([[off + i for i in fc] for fc in dfc])
                vcol.extend([(*rgb, 1.0)] * len(dv))
            me = bpy.data.meshes.new("Control Sphere")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            self._write_colors(me, vcol)
            me.update()
            obj = bpy.data.objects.new("Control Sphere", me)
            obj.data.materials.append(
                _color_material("Hopf Control Sphere", 0.0))
            _finish(context, obj)
            obj.parent = parent

        def _write_colors(self, me, vcol):
            import numpy as np
            if not vcol or len(vcol) != len(me.vertices):
                return
            attr = me.color_attributes.new("hopf_color", 'FLOAT_COLOR',
                                            'POINT')
            flat = np.asarray(vcol, dtype=np.float32).ravel()
            attr.data.foreach_set("color", flat)

        def execute(self, context):
            import numpy as np
            se = tuple(self.sphere_euler)
            ex = dict(turns=self.turns, seed=self.seed,
                      curl_lobes=self.curl_lobes, curl_amp=self.curl_amp)
            name = f"Hopf Fibration ({self.preset.title()})"

            # -------- SURFACE: skin fibres over continuous base rings ---
            if self.output == 'SURFACE':
                rings = base_rings(self.preset, self.n_lat, self.n_fiber,
                                   self.lat_min, self.lat_max, ex)
                if rings is None:
                    self.report({'WARNING'},
                                "SURFACE needs a ring preset "
                                "(Latitudes/Flower/Cap/...); drawing tubes")
                    self.output = 'MESH'
                else:
                    verts, faces, vbase = build_fiber_surface(
                        rings, self.samples, self.wind_p, self.wind_q,
                        se, self.s3_rot, self.chirality)
                    verts = [(x * self.scale, y * self.scale, z * self.scale)
                             for x, y, z in verts]
                    me = bpy.data.meshes.new(name)
                    me.from_pydata(verts, [], faces)
                    me.validate(clean_customdata=True)
                    me.polygons.foreach_set('use_smooth',
                                            [True] * len(me.polygons))
                    if self.color_fibers:
                        self._write_colors(me, [(*self._rgb(b), 1.0)
                                                for b in vbase])
                    me.update()
                    obj = bpy.data.objects.new(name, me)
                    if self.color_fibers:
                        obj.data.materials.append(
                            _color_material(name, self.emission))
                    _finish(context, obj)
                    self._select(context, obj)
                    if self.show_base_sphere:
                        flat = np.asarray(verts)
                        extent = float(np.abs(flat).max()) if len(flat) else 1.0
                        self._control_sphere(
                            context, [tuple(_rot_matrix(*se) @ np.asarray(b))
                                      for b in rings[0]], obj, extent)
                    self.report({'INFO'},
                                f"{name}: surface, {len(rings)} ring(s), "
                                f"{len(verts)} verts")
                    return {'FINISHED'}

            # -------- fibre bundle (tubes / beads / curves) -------------
            fibers, bases, closed, stats = build_fibers(
                self.preset, self.n_lat, self.n_fiber, self.samples,
                self.wind_p, self.wind_q, self.lat_min, self.lat_max,
                sphere_euler=se, s3_rot=self.s3_rot,
                chirality=self.chirality, include_axis=self.include_axis,
                extra=ex, return_stats=True)
            if not fibers:
                self.report({'ERROR'}, "No fibres produced")
                return {'CANCELLED'}
            fibers = [np.asarray(f) * self.scale for f in fibers]
            d = len(fibers)

            if self.output in ('MESH', 'BEADS'):
                if self.output == 'BEADS':
                    verts, faces, vcol = self._build_beads(fibers, bases)
                else:
                    verts, faces, vcol = self._build_tubes(fibers, bases,
                                                           closed)
                if self.markers > 0:
                    self._add_markers(fibers, bases, verts, faces, vcol)
                me = bpy.data.meshes.new(name)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
                if self.color_fibers:
                    self._write_colors(me, vcol)
                me.update()
                obj = bpy.data.objects.new(name, me)
                if self.color_fibers:
                    obj.data.materials.append(
                        _color_material(name, self.emission))
            else:
                cu = bpy.data.curves.new(name, 'CURVE')
                cu.dimensions = '3D'
                for Pl, cl in zip(fibers, closed):
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
                            sp.points[i].co = (pnt[0], pnt[1], pnt[2], 1.0)
                        if self.output == 'NURBS':
                            sp.order_u = 4
                    sp.use_cyclic_u = cl
                cu.bevel_depth = self.radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)
                if self.color_fibers:
                    for k in range(d):
                        rgb = self._rgb(bases[k])
                        mat = bpy.data.materials.new(f"{name} F{k + 1}")
                        mat.diffuse_color = (*rgb, 1.0)
                        mat.use_nodes = True
                        node = mat.node_tree.nodes.get("Principled BSDF")
                        if node:
                            node.inputs["Base Color"].default_value = \
                                (*rgb, 1.0)
                        obj.data.materials.append(mat)
                    for k, sp in enumerate(obj.data.splines):
                        sp.material_index = k

            _finish(context, obj)
            self._select(context, obj)
            if self.show_base_sphere:
                extent = max((float(np.abs(f).max()) for f in fibers),
                             default=1.0)
                self._control_sphere(context, bases, obj, extent)

            drp = stats.get('dropped', 0)
            self.report(
                {'INFO'},
                f"{name}: {d} fibres, {self.samples} samples"
                + ("" if (self.wind_p, self.wind_q) == (1, 1)
                   else f", ({self.wind_p},{self.wind_q}) winding")
                + (f", {drp} near-axis dropped" if drp else "")
                + (f", chirality {self.chirality.lower()}"
                   if self.chirality != 'RIGHT' else ""))
            return {'FINISHED'}

        def _select(self, context, obj):
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'LATITUDES':
                lay.prop(self, 'n_lat')
            if self.preset not in ('TETRA', 'OCTA', 'CUBE', 'ICOSA',
                                   'DODECA'):
                lay.prop(self, 'n_fiber')
            if self.preset == 'LOXODROME':
                lay.prop(self, 'turns')
            if self.preset == 'CURL':
                lay.prop(self, 'curl_lobes')
                lay.prop(self, 'curl_amp')
            if self.preset == 'RANDOM':
                lay.prop(self, 'seed')
            lay.prop(self, 'samples')
            row = lay.row(align=True)
            row.prop(self, 'wind_p')
            row.prop(self, 'wind_q')
            if self.preset in _RING_PRESETS:
                lay.prop(self, 'lat_min')
                lay.prop(self, 'lat_max')
            lay.prop(self, 'chirality')

            lay.separator()
            lay.prop(self, 'output')
            if self.output in ('BEZIER', 'POLY', 'NURBS'):
                lay.prop(self, 'radius')
                if self.radius > 0:
                    lay.prop(self, 'resolution')
            elif self.output == 'MESH':
                lay.prop(self, 'radius')
                lay.prop(self, 'tube_sides')
            elif self.output == 'BEADS':
                lay.prop(self, 'bead_count')
                lay.prop(self, 'bead_radius')
            if self.output != 'SURFACE':
                lay.prop(self, 'markers')
                if self.markers > 0:
                    lay.prop(self, 'marker_size')

            lay.separator()
            lay.prop(self, 'color_fibers')
            if self.color_fibers:
                lay.prop(self, 'color_style')
                if self.color_style == 'MONO':
                    lay.prop(self, 'mono_color')
                lay.prop(self, 'emission')

            lay.separator()
            lay.prop(self, 'sphere_euler')
            lay.prop(self, 's3_rot')
            if self.output != 'SURFACE':
                lay.prop(self, 'include_axis')
            lay.prop(self, 'show_base_sphere')
            if self.show_base_sphere:
                lay.prop(self, 'sphere_size')
            lay.prop(self, 'scale')

    class MESH_OT_hopf_torus_add(bpy.types.Operator):
        """Add a Pinkall Hopf torus: the preimage h^{-1}(gamma) of a
        closed curve gamma on S^2, stereographically projected to R^3"""
        bl_idname = "mesh.hopf_torus_add"
        bl_label = "Hopf Torus"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Curve on S^2",
            description="Closed curve on S^2 whose Hopf preimage is built",
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
            description="How the constrained elastic curve's shape is "
                        "solved (Constrained Elastica preset)",
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
            description="Which root of the shape equation to use: the "
                        "gentler or the curlier curve",
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
        so4_a1: FloatProperty(
            name="SO(4) x1x2", default=0.0, min=-180.0, max=180.0,
            description="Rotate the (x1,x2) plane of S^3 before "
                        "projecting -- morphs the torus (ring/horn/"
                        "spindle, Dupin cyclides)")
        so4_a2: FloatProperty(
            name="SO(4) x3x4", default=0.0, min=-180.0, max=180.0,
            description="Rotate the (x3,x4) plane of S^3 before "
                        "projecting")
        so4_a3: FloatProperty(
            name="SO(4) x1x3", default=0.0, min=-180.0, max=180.0,
            description="Rotate the (x1,x3) plane of S^3 before "
                        "projecting")
        shade_smooth: BoolProperty(
            name="Shade Smooth", default=True,
            description="Smooth-shade the torus surface")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size of the torus")

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
            R4 = None
            if any((self.so4_a1, self.so4_a2, self.so4_a3)):
                R4 = _so4(math.radians(self.so4_a1),
                          math.radians(self.so4_a2),
                          math.radians(self.so4_a3))
            verts, faces, area = build_hopf_torus(
                gamma, self.m_psi, closed=(self.preset != 'BAND'), R4=R4)
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
                # Pinkall's flat torus is C / <(2pi,0),(A/2,L/2)>; its
                # conformal modulus is tau = (A/2 + i L/2) / (2 pi).
                tau_re = (area / 2.0) / (2.0 * pi)
                tau_im = (length / 2.0) / (2.0 * pi)
                msg += (f", Willmore energy {energy:.4f} = "
                        f"{energy / (2.0 * pi * pi):.4f} x 2pi^2"
                        f" (gamma length {length:.3f}), conformal "
                        f"tau = {tau_re:.3f} + {tau_im:.3f} i")
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
            lay.separator()
            col = lay.column(align=True)
            col.prop(self, 'so4_a1')
            col.prop(self, 'so4_a2')
            col.prop(self, 'so4_a3')
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

    # 13) Phase 1/2 additions: chirality, S^3 rotation, SO(4), the new
    #     base-point providers, the fibre-surface skinner, and the small
    #     geometry primitives.
    b = _normalize((0.4, 0.2, 0.7))
    XR = fiber_s3(b, 64, 1, 1, 'RIGHT')
    XL = fiber_s3(b, 64, 1, 1, 'LEFT')
    chi_ok = (abs(np.linalg.norm(XL, axis=1) - 1.0).max() < 1e-12
              and np.abs(XR - XL).max() > 1e-6
              and np.allclose(XL[:, 3], -XR[:, 3]))
    ok_all = ok_all and chi_ok
    print(f"chirality: LEFT on S^3 & mirrors RIGHT "
          f"{'OK' if chi_ok else 'BAD'}")

    Xr = _s3_rotate(XR, _quat_left(0.7))
    rot_ok = abs(np.linalg.norm(Xr, axis=1) - 1.0).max() < 1e-12
    R4 = _so4(0.5, -0.9, 0.3)
    so4_ok = np.allclose(R4 @ R4.T, np.eye(4), atol=1e-12)
    ok_all = ok_all and rot_ok and so4_ok
    print(f"S^3 rot keeps |x|=1 {'OK' if rot_ok else 'BAD'}; "
          f"SO(4) orthogonal {'OK' if so4_ok else 'BAD'}")

    for prov in ('CAP', 'LOXODROME', 'CURL', 'RANDOM'):
        pts = base_points(prov, 3, 20, 20.0, 160.0,
                          dict(turns=4.0, seed=1, curl_lobes=5,
                               curl_amp=25.0))
        unit = max(abs(np.linalg.norm(p) - 1.0) for p in pts)
        ok = len(pts) == 20 and unit < 1e-9
        ok_all = ok_all and ok
        print(f"provider {prov:10s}: {len(pts)} pts unit dev={unit:.1e} "
              f"{'OK' if ok else 'BAD'}")

    rings = base_rings('LATITUDES', 4, 16, 20.0, 160.0)
    Vs, Fs, vb = build_fiber_surface(rings, 48)
    surf_ok = (len(Vs) == 4 * 16 * 48 and len(Fs) == 4 * 16 * 48
               and len(vb) == len(Vs) and np.isfinite(np.asarray(Vs)).all())
    ok_all = ok_all and surf_ok
    print(f"fibre surface: verts={len(Vs)} faces={len(Fs)} "
          f"{'OK' if surf_ok else 'BAD'}")

    iv, ifc = _icosphere(0.1, 1)
    cvv, cff = _cone(0.5, 1.0, 8)
    P = project_fiber(_normalize((0.3, 0.1, 0.6)), 40)
    ctv, ctf = _closed_tube(P, 0.05, 6)
    otv, otf = _open_tube(P[:20], 0.05, 6)
    prim_ok = (len(iv) == 42 and len(ctv) == 40 * 6 and len(ctf) == 40 * 6
               and len(otv) == 20 * 6 and len(otf) == 19 * 6
               and len(cvv) == 9)
    pal_ok = all(0.0 <= c <= 1.0 for st in ('RAINBOW', 'PASTEL', 'MONO')
                 for c in _palette_rgb(b, st))
    ok_all = ok_all and prim_ok and pal_ok
    print(f"primitives {'OK' if prim_ok else 'BAD'}; "
          f"palettes {'OK' if pal_ok else 'BAD'}")

    assert ok_all
    print("hopf fibration standalone tests passed")
