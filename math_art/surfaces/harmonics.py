# Spherical harmonics as radial displacement.
#
# Part of the Math Art surfaces engine (`math_art/surfaces/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# The spherical harmonics Y_l^m are the eigenfunctions of the
# Laplace-Beltrami operator on the sphere -- the natural vibration modes
# of a spherical membrane, and a complete basis for functions on it.
# Displacing the sphere radially by one of them gives the familiar
# lobed forms, with l nodal circles and m nodal meridians.
#
# References:
# - P.-S. Laplace, "Theorie des attractions des spheroides", 1782.
# - E. W. Hobson, "The Theory of Spherical and Ellipsoidal Harmonics",
#   Cambridge, 1931.

import math
import numpy as np


TAU = 2.0 * math.pi


def assoc_legendre(l, m, x):
    """Associated Legendre function P_l^m(x) on an array x in [-1, 1],
    Condon-Shortley phase included.  Requires 0 <= m <= l."""
    l, m = int(l), int(abs(m))
    if m > l:
        raise ValueError(f"assoc_legendre needs |m| <= l, got l={l}, m={m}")
    x = np.asarray(x, dtype=float)
    pmm = np.ones_like(x)
    if m > 0:
        # (1-x^2)^(m/2), clamped: rounding can push |x| a hair over 1
        somx2 = np.sqrt(np.maximum(0.0, 1.0 - x * x))
        fact = 1.0
        for _ in range(m):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = pmmp1
    for ll in range(m + 2, l + 1):
        pll = (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pll


def sph_harm_norm(l, m):
    """The real-harmonic normalisation sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!),
    computed through lgamma so large l cannot overflow."""
    l, m = int(l), int(abs(m))
    return math.exp(0.5 * (math.log(2.0 * l + 1.0) - math.log(4.0 * math.pi)
                           + math.lgamma(l - m + 1.0)
                           - math.lgamma(l + m + 1.0)))


def real_sph_harm(l, m, theta, phi):
    """Real spherical harmonic Y_l^m evaluated on arrays of the polar
    angle theta in [0, pi] (colatitude, 0 at +z) and the azimuth phi.

    Sign convention: m > 0 takes the cosine (cos m phi) partner, m < 0
    the sine partner, and both carry the sqrt(2) that keeps the real
    pair orthonormal.  This is the standard real basis used for atomic
    orbitals: (l, m) = (1, 0) is p_z, (1, 1) is p_x, (1, -1) is p_y."""
    l, m = int(l), int(m)
    am = abs(m)
    if am > l:
        raise ValueError(f"real_sph_harm needs |m| <= l, got l={l}, m={m}")
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    p = assoc_legendre(l, am, np.cos(theta))
    n = sph_harm_norm(l, am)
    if m == 0:
        return n * p
    if m > 0:
        return math.sqrt(2.0) * n * p * np.cos(am * phi)
    return math.sqrt(2.0) * n * p * np.sin(am * phi)


def max_abs_harmonic(l, m, samples=512):
    """max |Y_l^m| over the sphere -- the scale the OFFSET form needs in
    order to guarantee a positive (hence star-shaped) radius."""
    th = np.linspace(0.0, math.pi, samples)
    p = assoc_legendre(l, abs(m), np.cos(th))
    n = sph_harm_norm(l, abs(m))
    amp = n * float(np.max(np.abs(p)))
    return amp if m == 0 else math.sqrt(2.0) * amp


def bourke_radius(mm, theta, phi):
    """r = sin(m0 phi)^m1 + cos(m2 phi)^m3 + sin(m4 theta)^m5
           + cos(m6 theta)^m7
    with phi the polar angle in [0, pi] and theta the azimuth in
    [0, 2 pi], following Bourke's own convention.

    The exponents MUST be Python ints: numpy raises a negative float
    base to a float power as NaN, which is the classic way to get an
    empty mesh out of this family."""
    m0, m1, m2, m3, m4, m5, m6, m7 = (int(v) for v in mm)
    return (np.sin(m0 * phi) ** m1 + np.cos(m2 * phi) ** m3
            + np.sin(m4 * theta) ** m5 + np.cos(m6 * theta) ** m7)


# Project-chosen parameter tuples (Bourke publishes renders, not named
# sets).  Each is (label, (m0..m7)).
BOURKE_PRESETS = [
    ('B1', "Bourke 4-1-4-1", (4, 1, 4, 1, 4, 1, 4, 1)),
    ('B2', "Bourke 2-1-2-1", (2, 1, 2, 1, 2, 1, 2, 1)),
    ('B3', "Bourke 1-2-2-2", (1, 2, 2, 2, 4, 2, 3, 2)),
    ('B4', "Bourke 3-2-2-3", (3, 2, 2, 3, 3, 2, 2, 3)),
    ('B5', "Bourke 5-1-3-1", (5, 1, 3, 1, 5, 1, 3, 1)),
    ('B6', "Bourke 2-3-4-1", (2, 3, 4, 1, 2, 3, 4, 1)),
]


def build_radial_surface(rfun, nu=128, nv=256):
    """Mesh r = rfun(theta_polar, phi_azimuth) as a closed sphere-topology
    surface.  The two poles collapse to single vertices and the azimuth
    seam is glued by index, so the result has chi = 2 and no boundary.

    Returns (verts (n,3), faces list, face_param (m,2)) where face_param
    holds the (theta, phi) of each face centre -- the SIGNED form uses it
    to look up the sign of Y without re-deriving it from geometry."""
    nu, nv = max(4, int(nu)), max(6, int(nv))
    th = math.pi * np.arange(nu + 1) / nu            # 0 .. pi
    ph = TAU * np.arange(nv) / nv                    # 0 .. 2pi (wrapped)
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    R = np.asarray(rfun(TH, PH), dtype=float)
    if R.shape != TH.shape:
        R = np.broadcast_to(R, TH.shape)
    X = R * np.sin(TH) * np.cos(PH)
    Y = R * np.sin(TH) * np.sin(PH)
    Z = R * np.cos(TH)

    # poles: every column of row 0 (and row nu) is the same point, so
    # collapse each to one vertex -- welding by index, never by
    # coordinate proximity
    verts = [(0.0, 0.0, float(np.mean(Z[0])))]
    idx = np.zeros((nu + 1, nv), dtype=np.int64)
    idx[0, :] = 0
    for i in range(1, nu):
        base = len(verts)
        for j in range(nv):
            verts.append((float(X[i, j]), float(Y[i, j]), float(Z[i, j])))
        idx[i, :] = base + np.arange(nv)
    south = len(verts)
    verts.append((0.0, 0.0, float(np.mean(Z[nu]))))
    idx[nu, :] = south

    faces, fparam = [], []

    def mid(i0, i1, j0, j1):
        return (0.5 * (th[i0] + th[i1]),
                0.5 * (ph[j0] + ph[j0] + TAU / nv))

    for j in range(nv):
        j2 = (j + 1) % nv
        faces.append((0, idx[1, j2], idx[1, j]))
        fparam.append(mid(0, 1, j, j2))
    for i in range(1, nu - 1):
        for j in range(nv):
            j2 = (j + 1) % nv
            faces.append((idx[i, j], idx[i, j2],
                          idx[i + 1, j2], idx[i + 1, j]))
            fparam.append(mid(i, i + 1, j, j2))
    for j in range(nv):
        j2 = (j + 1) % nv
        faces.append((south, idx[nu - 1, j], idx[nu - 1, j2]))
        fparam.append(mid(nu - 1, nu, j, j2))

    return np.asarray(verts, dtype=float), faces, np.asarray(fparam)


def _drop_faces(verts, faces, extra, keep):
    """Keep the faces flagged by `keep`, dropping orphaned vertices.
    `extra` is a per-face array carried along."""
    faces = [f for f, k in zip(faces, keep) if k]
    extra = [e for e, k in zip(extra, keep) if k]
    used = sorted({i for f in faces for i in f})
    remap = {o: n for n, o in enumerate(used)}
    return (verts[used], [tuple(remap[i] for i in f) for f in faces],
            np.asarray(extra))


def center_fit(verts, scale=1.0):
    """Centre on the bounding-box midpoint and fit the largest extent to
    a 2 m cube, then apply `scale` (the project-wide convention)."""
    if not len(verts):
        return verts
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ext = float((hi - lo).max())
    out = (verts - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)
    return out * scale


FORM_ITEMS = [
    ('OFFSET', "Offset Sphere", "r = r0 + a Y_l^m: a smoothly deformed "
                                "sphere, always embedded"),
    ('ABS', "Absolute (lobes)", "r = |Y_l^m|: the classic lobed balloon "
                                "(pinched at the nodal circles)"),
    ('SIGNED', "Signed lobes", "r = |Y_l^m| with the lobes separated by "
                               "the sign of Y"),
    ('BOURKE', "Bourke Family", "The eight-integer trigonometric "
                                "harmonic family"),
]


def build_spherical_harmonic(form='OFFSET', l=3, m=2, nu=128, nv=256,
                             r0=1.0, amp=0.6, eps=0.02, mm=None,
                             abs_radius=False, split_lobes=False,
                             scale=1.0):
    """Build one spherical-harmonic surface.  Returns
    (verts, faces, face_sign) with face_sign +1/-1 per face for the
    SIGNED form (all +1 otherwise)."""
    l = max(0, int(l))
    m = int(np.clip(int(m), -l, l))

    if form == 'BOURKE':
        mm = tuple(mm) if mm else BOURKE_PRESETS[0][2]

        def rfun(theta, phi):
            # Bourke's phi is the POLAR angle and his theta the azimuth;
            # our grid hands them over in (polar, azimuth) order, so his
            # (theta, phi) is our (phi, theta).
            r = bourke_radius(mm, phi, theta)
            return np.abs(r) if abs_radius else r
    elif form == 'OFFSET':
        def rfun(theta, phi):
            return r0 + amp * real_sph_harm(l, m, theta, phi)
    else:
        def rfun(theta, phi):
            return np.abs(real_sph_harm(l, m, theta, phi)) + eps

    verts, faces, fparam = build_radial_surface(rfun, nu, nv)

    sign = np.ones(len(faces), dtype=int)
    if form == 'SIGNED':
        y = real_sph_harm(l, m, fparam[:, 0], fparam[:, 1])
        sign = np.where(y >= 0.0, 1, -1)
        if split_lobes:
            # drop the band of faces straddling a nodal line so the
            # lobes come apart into separate loose parts
            keep = np.abs(y) > 0.02 * max(max_abs_harmonic(l, m), 1e-9)
            verts, faces, fparam = _drop_faces(verts, faces, fparam, keep)
            y = real_sph_harm(l, m, fparam[:, 0], fparam[:, 1])
            sign = np.where(y >= 0.0, 1, -1)

    return center_fit(verts, scale), faces, sign
