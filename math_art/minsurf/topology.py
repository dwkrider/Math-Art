# Topological surfaces: Klein bottles, cross-caps, genus-g handlebodies.
#
# Part of the Math Art minsurf engine (`math_art/minsurf/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#

import math
import numpy as np


TAU = 2.0 * math.pi


def edge_face_counts(faces):
    """{sorted edge tuple: number of incident faces}."""
    cnt = {}
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    return cnt


def euler_characteristic(nverts, faces):
    return nverts - len(edge_face_counts(faces)) + len(faces)


def build_klein_bottle(nu, nv):
    """The iconic bottle-shaped Klein immersion (the standard smooth
    closed-form parametrization, u in [0, pi], v in [0, 2pi]). The
    u = pi rim coincides with the u = 0 rim under v -> pi - v.  The
    seam is left SPLIT (coincident duplicate vertices, no index
    gluing): welding it makes the winding flip there, and averaged
    smooth normals then degenerate into a dark shading crease.  Split,
    each side shades smoothly and the renderer's double-sided normal
    flip hides the join.  Cut along that rim the surface is an
    orientable cylinder, so chi is still 0."""
    nv += nv % 2
    u = math.pi * np.arange(nu + 1)[:, None] / nu
    v = TAU * np.arange(nv)[None, :] / nv
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)
    x = (-2.0 / 15.0) * cu * (3 * cv - 30 * su + 90 * cu ** 4 * su
                              - 60 * cu ** 6 * su + 5 * cu * cv * su)
    y = (-1.0 / 15.0) * su * (3 * cv - 3 * cu ** 2 * cv
                              - 48 * cu ** 4 * cv + 48 * cu ** 6 * cv
                              - 60 * su + 5 * cu * cv * su
                              - 5 * cu ** 3 * cv * su
                              - 80 * cu ** 5 * cv * su
                              + 80 * cu ** 7 * cv * su)
    z = (2.0 / 15.0) * sv * (3 + 5 * cu * su)
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            j2 = (j + 1) % nv
            faces.append((i * nv + j, i * nv + j2,
                          (i + 1) * nv + j2, (i + 1) * nv + j))
    return V, faces


def build_klein_figure8(nu, nv, radius=2.0):
    """Figure-8 (twisted-torus) Klein immersion: the cross-section is a
    figure-8 that makes a half-turn per revolution. The u = 2pi seam
    coincides with u = 0 under v -> -v but is left SPLIT (coincident
    duplicate vertices) -- see build_klein_bottle for why.  v samples
    sit at half-steps so no column lands on the figure-8 crossing
    point."""
    u = TAU * np.arange(nu + 1)[:, None] / nu
    v = TAU * (np.arange(nv)[None, :] + 0.5) / nv
    c2, s2 = np.cos(u / 2), np.sin(u / 2)
    sv, s2v = np.sin(v), np.sin(2 * v)
    r = radius + c2 * sv - s2 * s2v
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = s2 * sv + c2 * s2v
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            j2 = (j + 1) % nv
            faces.append((i * nv + j, i * nv + j2,
                          (i + 1) * nv + j2, (i + 1) * nv + j))
    return V, faces


def build_sudanese_mobius(nu, nv):
    """Lawson's minimal Mobius band in S^3, stereographically projected
    to R^3.  The t = 0 and t = pi seam rows coincide in space (with a
    flip v -> pi - v), so -- exactly as for the Klein bottles above --
    the grid is left SPLIT there rather than index-glued: welding flips
    the winding and averaged smooth normals then form a dark crease.
    Cut open along the seam the mesh is a disk (chi 1) whose two ends
    meet on the boundary circle."""
    R2 = math.sqrt(2.0)
    t = math.pi * np.arange(nu + 1)[:, None] / nu       # around
    v = math.pi * np.arange(nv + 1)[None, :] / nv        # across
    ct, st = np.cos(t), np.sin(t)
    cv, sv = np.cos(v), np.sin(v)
    x1 = ct * cv
    x2 = st * cv
    x3 = np.cos(2 * t) * sv
    x4 = np.sin(2 * t) * sv
    s = 1.0 + (x1 + x3) / R2                              # = 1 - x . p
    x = x2 / s
    y = x4 / s
    z = (x1 - x3) / (R2 * s)
    V = np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(-1, 3)
    stride = nv + 1
    faces = []
    for i in range(nu):
        for j in range(nv):
            faces.append((i * stride + j, i * stride + j + 1,
                          (i + 1) * stride + j + 1, (i + 1) * stride + j))
    return V, faces


def _rp2_quotient(nu, nv, fn, theta_offset):
    """Mesh a hemisphere parametrization fn(theta, phi) -> (x, y, z),
    phi in (0, pi/2], with the phi = pi/2 pole collapsed to one vertex
    and the phi = 0 equator glued to itself by theta -> theta + pi
    (the RP^2 quotient). nu must be even. Returns (verts, faces) with
    Euler characteristic 1 by construction."""
    half = nu // 2
    th = TAU * (np.arange(nu) + theta_offset) / nu
    x, y, z = fn(th[:1], math.pi / 2)
    verts = [np.array([x[0], y[0], z[0]])]
    for k in range(1, nv):
        ph = (math.pi / 2) * (1.0 - k / (nv - 1))
        m = nu if k < nv - 1 else half
        x, y, z = fn(th[:m], ph)
        verts.extend(np.stack(np.broadcast_arrays(x, y, z), axis=-1))

    def rid(k, j):
        j %= nu
        if k < nv - 1:
            return 1 + (k - 1) * nu + j
        return 1 + (nv - 2) * nu + (j % half)

    faces = [(0, rid(1, j + 1), rid(1, j)) for j in range(nu)]
    for k in range(1, nv - 1):
        for j in range(nu):
            faces.append((rid(k, j), rid(k, j + 1),
                          rid(k + 1, j + 1), rid(k + 1, j)))
    return np.array(verts), faces


def _crosscap_pt(th, ph):
    """Standard cross-cap immersion of RP^2; antipodes
    of the sphere (th, ph latitude) map to the same point."""
    st, ct = np.sin(th), np.cos(th)
    return (0.5 * st * np.sin(2 * ph) + 0 * th,
            0.5 * np.sin(2 * th) * np.cos(ph) ** 2,
            0.5 * np.cos(2 * th) * np.cos(ph) ** 2)


def _roman_pt(th, ph):
    """Steiner's Roman surface: the sphere mapped through
    (x, y, z) -> (yz, zx, xy)."""
    cp, sp = np.cos(ph), np.sin(ph)
    return (np.sin(th) * cp * sp,
            np.cos(th) * cp * sp,
            np.sin(th) * np.cos(th) * cp * cp)


def build_crosscap(nu, nv):
    nu += nu % 2               # theta -> theta + pi must be a grid map
    return _rp2_quotient(nu, nv, _crosscap_pt, 0.5)


def build_roman(nu, nv):
    nu += (-nu) % 4            # quarter-offset grid: need 4 | nu
    return _rp2_quotient(nu, nv, _roman_pt, 0.25)


def build_boy(ntheta, nrings):
    """Boy's surface via the Bryant-Kusner parametrization on the unit
    disk (polar grid), with the boundary circle glued antipodally
    (z ~ -z on |z| = 1) by vertex index. The three poles of the
    denominator inside the disk are the planar ends of the underlying
    minimal surface; they invert to the triple point at the origin, and
    samples landing on them are nudged off."""
    ntheta += ntheta % 2
    half = ntheta // 2
    th = TAU * (np.arange(ntheta) + 0.5) / ntheta
    s5 = math.sqrt(5.0)

    def bk(zc):
        w = zc ** 6 + s5 * zc ** 3 - 1
        bad = np.abs(w) < 1e-7
        if np.any(bad):
            zc = np.where(bad, zc * 1.01, zc)
            w = zc ** 6 + s5 * zc ** 3 - 1
        g1 = -1.5 * (zc * (1 - zc ** 4) / w).imag
        g2 = -1.5 * (zc * (1 + zc ** 4) / w).real
        g3 = ((1 + zc ** 6) / w).imag - 0.5
        s = g1 * g1 + g2 * g2 + g3 * g3
        return np.stack([g1 / s, g2 / s, g3 / s], axis=-1)

    verts = [bk(np.zeros(1, dtype=complex))[0]]
    for k in range(1, nrings + 1):
        r = k / nrings
        m = ntheta if k < nrings else half
        verts.extend(bk(r * np.exp(1j * th[:m])))

    def rid(k, j):
        j %= ntheta
        if k < nrings:
            return 1 + (k - 1) * ntheta + j
        return 1 + (nrings - 1) * ntheta + (j % half)

    faces = [(0, rid(1, j + 1), rid(1, j)) for j in range(ntheta)]
    for k in range(1, nrings):
        for j in range(ntheta):
            faces.append((rid(k, j), rid(k, j + 1),
                          rid(k + 1, j + 1), rid(k + 1, j)))
    return np.array(verts), faces


_GENUS_R = 1.0          # circle radius


_GENUS_SPACING = 1.4    # center spacing (< 2r: adjacent circles overlap)


_GENUS_LEVEL = 0.015    # eps: below the lens-core peak for g = 1..5


_GENUS_ZK = 2.0         # z^2 coefficient: slab half-height <= ~0.42


def build_genus(genus, cell=0.125):
    # This module IS inside `minsurf`, so it imports the package it lives
    # in -- one dot up, not one dot across.
    try:
        from . import parametric, plateau, weierstrass, zoo   # noqa: F401
        from .. import minsurf as mst
    except ImportError:
        import minsurf as mst
    g = genus
    r, d = _GENUS_R, _GENUS_SPACING
    cs = [(i - g / 2.0) * d for i in range(g + 1)]

    def field(x, y, z):
        q = np.ones_like(x)
        for c in cs:
            rho2 = (x - c) ** 2 + y ** 2
            q = q * (rho2 - r * r) / (rho2 + r * r)
        return q + _GENUS_ZK * z * z - _GENUS_LEVEL

    m = 0.6
    bmin = (cs[0] - r - m, -r - m, -0.55)
    bmax = (cs[-1] + r + m, r + m, 0.55)
    res = tuple(max(8, int(round((bmax[i] - bmin[i]) / cell)))
                for i in range(3))
    return mst.marching_tets(field, bmin, bmax, res)


def build_twist_strip(half_twists, segments, width=0.6, thick=0.18,
                      ridge=False, radius=1.5):
    """Sweep a rectangular cross-section (optionally with a raised
    center-line ridge on both wide faces, as in Segerman fig 6-1)
    around a circle, turning it by n*pi over one revolution. The
    cross-section point list is symmetric under a half-turn (index
    shift k/2), so for odd n the seam closes with an index shift and
    the result is a single watertight solid -- printable directly."""
    n = half_twists
    m = max(segments, 8 * max(abs(n), 1))
    w2, t2 = width / 2.0, thick / 2.0
    if ridge:
        bw, bh = 0.16 * width, 0.7 * thick
        prof = [(w2, -t2), (w2, t2), (bw, t2), (0.0, t2 + bh),
                (-bw, t2), (-w2, t2), (-w2, -t2), (-bw, -t2),
                (0.0, -t2 - bh), (bw, -t2)]
    else:
        prof = [(w2, -t2), (w2, t2), (-w2, t2), (-w2, -t2)]
    k = len(prof)
    shift = (n % 2) * (k // 2)
    verts = []
    for j in range(m):
        t = TAU * j / m
        al = 0.5 * n * t
        ca, sa = math.cos(al), math.sin(al)
        ct, st = math.cos(t), math.sin(t)
        for (a, b) in prof:
            ar = a * ca - b * sa       # rotate in the (radial, z) plane
            br = a * sa + b * ca
            verts.append(((radius + ar) * ct, (radius + ar) * st, br))
    faces = []
    for j in range(m):
        j2 = (j + 1) % m
        s = shift if j == m - 1 else 0
        for i in range(k):
            i2 = (i + 1) % k
            faces.append((j * k + i, j * k + i2,
                          j2 * k + (i2 + s) % k, j2 * k + (i + s) % k))
    return np.array(verts), faces
