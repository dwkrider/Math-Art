# Phyllotaxis: the golden-angle spiral arrangement of florets.
#
# Part of the Math Art lsystem engine (`math_art/lsystem/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# Vogel's model places floret k at angle k * 137.508 degrees and radius
# proportional to sqrt(k), which reproduces the parastichy spirals whose
# counts are consecutive Fibonacci numbers.
#
# References:
# - H. Vogel, "A better way to construct the sunflower head",
#   Mathematical Biosciences 44, 1979, pp. 179-189.

import math
from math import sqrt, sin, cos, pi, radians
import numpy as np

try:
    from ..surfaces.primitives import icosphere as _icosphere_shared
except ImportError:  # flat import outside the package
    from surfaces.primitives import icosphere as _icosphere_shared


# The golden angle in degrees: 360 * (2 - phi) = 360 / phi^2.
PHI = (1.0 + sqrt(5.0)) / 2.0


GOLDEN_ANGLE_DEG = 360.0 * (2.0 - PHI)      # 137.50776405003785...


def phyllotaxis_points(n, form='DISK', divergence=GOLDEN_ANGLE_DEG,
                       dome_height=0.8, chirality='RIGHT',
                       crest_waves=3.0, crest_amp=0.35):
    """Golden-angle floret centers and surface normals.

    Returns (P, Nrm, rho) where P is an (n, 3) array of floret centers,
    Nrm the (n, 3) unit outward surface normals, and rho an (n,)
    normalized radial/latitude parameter in [0, 1] (0 at the center or
    top, 1 at the rim) used for radial coloring and size grading.

    form: 'DISK' (flat sunflower), 'DOME' (paraboloid cap), 'CONE',
    'SPHERE' (equal-area spherical Fibonacci lattice) or 'CREST'
    (corrugated, fasciated ridges).
    """
    n = max(1, int(n))
    i = np.arange(n, dtype=float)
    frac = (i + 0.5) / n
    chir = -1.0 if chirality == 'LEFT' else 1.0
    th = chir * radians(float(divergence)) * i
    ct, st = np.cos(th), np.sin(th)
    H = float(dome_height)

    if form == 'SPHERE':
        z = 1.0 - 2.0 * frac                 # +1 (top) .. -1 (bottom)
        rc = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        P = np.column_stack([rc * ct, rc * st, z])
        Nrm = P.copy()                       # already unit on the sphere
        rho = np.arccos(np.clip(z, -1.0, 1.0)) / pi
        return P, Nrm, rho

    rr = np.sqrt(frac)                        # ~ 0 .. 1, Vogel radius
    x, y = rr * ct, rr * st
    rmax = float(rr.max()) or 1.0

    if form == 'DISK':
        z = np.zeros(n)
        fx = np.zeros(n)
        fy = np.zeros(n)
    elif form == 'DOME':                      # paraboloid z = H(1 - r^2)
        z = H * (1.0 - rr * rr)
        fx = -2.0 * H * x
        fy = -2.0 * H * y
    elif form == 'CONE':                      # z = H(1 - r)
        z = H * (1.0 - rr)
        safe = np.where(rr > 1e-9, rr, 1.0)
        fx = np.where(rr > 1e-9, -H * x / safe, 0.0)
        fy = np.where(rr > 1e-9, -H * y / safe, 0.0)
    elif form == 'CREST':                     # dome + parallel ridges
        w = crest_waves * pi
        z = H * (1.0 - rr * rr) + crest_amp * np.cos(w * x)
        fx = -2.0 * H * x - crest_amp * w * np.sin(w * x)
        fy = -2.0 * H * y
    else:
        raise ValueError("unknown phyllotaxis form %r" % form)

    P = np.column_stack([x, y, z])
    # surface z = f(x, y): upward unit normal is (-f_x, -f_y, 1)/|.|
    Nrm = np.column_stack([-fx, -fy, np.ones(n)])
    Nrm /= np.linalg.norm(Nrm, axis=1, keepdims=True)
    rho = rr / rmax
    return P, Nrm, rho


def nearest_neighbor_dist(P):
    """Per-point distance to the closest other point, in chunks so the
    pairwise matrix never fully materializes.  A single point returns
    a distance of 1.0."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 2:
        return np.ones(n)
    out = np.empty(n)
    B = 256
    for s in range(0, n, B):
        e = min(n, s + B)
        d = np.linalg.norm(P[s:e, None, :] - P[None, :, :], axis=2)
        for k in range(s, e):
            d[k - s, k] = np.inf
        out[s:e] = d.min(axis=1)
    return out


def _icosphere(subdiv=0):
    """Unit icosphere: the icosahedron subdivided `subdiv` times.
    See `surfaces.primitives`.  Same surface, standard vertex order.
    """
    return _icosphere_shared(subdiv, 'per_level')


def _bump_template(subdiv, flatten):
    """Oblate spheroid: an icosphere squashed along the local normal so
    it reads as a seed sitting tangent to the surface."""
    V, F = _icosphere(subdiv)
    V = V.copy()
    V[:, 2] *= flatten                        # flatten along the normal
    return V, [tuple(f) for f in F], True     # smooth-shaded


def _spike_template(seg, spike_ratio):
    """Cone (areole spike): apex out along the normal, base ring in the
    tangent plane, plus a base fan so it is a closed solid."""
    seg = max(3, int(seg))
    V = [(cos(2 * pi * k / seg), sin(2 * pi * k / seg), 0.0)
         for k in range(seg)]
    apex = len(V)
    V.append((0.0, 0.0, float(spike_ratio)))
    base = len(V)
    V.append((0.0, 0.0, 0.0))
    F = []
    for k in range(seg):
        k2 = (k + 1) % seg
        F.append((k, k2, apex))               # side
        F.append((k2, k, base))               # base cap
    return np.array(V, float), F, False       # flat-shaded facets


def _disc_template(seg):
    """Flat disc floret in the tangent plane, lifted a hair off the
    surface so it does not z-fight."""
    seg = max(3, int(seg))
    V = [(cos(2 * pi * k / seg), sin(2 * pi * k / seg), 0.02)
         for k in range(seg)]
    ctr = len(V)
    V.append((0.0, 0.0, 0.02))
    F = [(k, (k + 1) % seg, ctr) for k in range(seg)]
    return np.array(V, float), F, False


def _frames(Nrm):
    """Two unit tangents (t1, t2) per normal, forming a right-handed
    frame (t1, t2, n)."""
    N = np.asarray(Nrm, float)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(N), 1))
    par = np.abs(N[:, 2]) > 0.9               # normal near vertical
    ref[par] = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(N, ref)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(N, t1)
    return t1, t2


def point_classes(n, rho, color_by='PARASTICHY', parastichy=13, npal=8):
    """Per-point color class in 0..npal-1: by parastichy residue
    (i mod k), by radial ring, or uniform (all 0)."""
    k = max(1, int(parastichy))
    if color_by == 'PARASTICHY':
        return (np.arange(n) % k) % npal
    if color_by == 'RING':
        return np.floor(np.asarray(rho) * (npal - 1e-9)).astype(int)
    return np.zeros(n, dtype=int)


def build_points(n=500, form='DISK', divergence=GOLDEN_ANGLE_DEG,
                 dome_height=0.8, chirality='RIGHT', crest_waves=3.0,
                 crest_amp=0.35, fill=0.9, size_grade=0.0,
                 color_by='PARASTICHY', parastichy=13):
    """Just the floret placement, for the points-only output: returns
    (P, Nrm, size, cls) -- centers, unit surface normals, per-floret
    size, and color class -- so the user can instance their own object
    at each point (orienting by the normal, scaling by the size)."""
    P, Nrm, rho = phyllotaxis_points(n, form, divergence, dome_height,
                                     chirality, crest_waves, crest_amp)
    spacing = nearest_neighbor_dist(P)
    grade = np.clip(1.0 - float(size_grade) * (1.0 - rho), 0.05, 1.0)
    size = float(fill) * spacing * grade
    cls = point_classes(len(P), rho, color_by, parastichy)
    return P, Nrm, size, cls


def build_phyllotaxis(n=500, form='DISK', floret='BUMP',
                      divergence=GOLDEN_ANGLE_DEG, dome_height=0.8,
                      chirality='RIGHT', crest_waves=3.0, crest_amp=0.35,
                      fill=0.9, size_grade=0.0, spike_ratio=2.2,
                      flatten=0.55, subdiv=1, seg=12,
                      color_by='PARASTICHY', parastichy=13):
    """Assemble the whole florets mesh.  Returns (verts, faces, tags,
    smooth) where verts is a list of (x, y, z), faces a list of index
    tuples, tags a per-face material index and smooth a bool for the
    shading of the floret solid."""
    P, Nrm, rho = phyllotaxis_points(n, form, divergence, dome_height,
                                     chirality, crest_waves, crest_amp)
    t1, t2 = _frames(Nrm)
    spacing = nearest_neighbor_dist(P)
    grade = np.clip(1.0 - float(size_grade) * (1.0 - rho), 0.05, 1.0)
    size = float(fill) * spacing * grade

    if floret == 'SPIKE':
        Vloc, Floc, smooth = _spike_template(seg, spike_ratio)
    elif floret == 'DISC':
        Vloc, Floc, smooth = _disc_template(seg)
    else:
        Vloc, Floc, smooth = _bump_template(subdiv, flatten)
    Vloc = np.asarray(Vloc, float)
    m = len(Vloc)

    # per-point residue class for coloring
    cls = point_classes(n, rho, color_by, parastichy)

    verts = []
    faces = []
    tags = []
    for idx in range(n):
        base = len(verts)
        s = size[idx]
        # world = P + s*(vx*t1 + vy*t2 + vz*n)
        world = (P[idx]
                 + s * (Vloc[:, 0:1] * t1[idx]
                        + Vloc[:, 1:2] * t2[idx]
                        + Vloc[:, 2:3] * Nrm[idx]))
        verts.extend((float(v[0]), float(v[1]), float(v[2]))
                     for v in world)
        c = int(cls[idx])
        for f in Floc:
            faces.append(tuple(base + j for j in f))
            tags.append(c)
    return verts, faces, tags, smooth
