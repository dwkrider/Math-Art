# Escape-time fractals: the Mandelbulb, quaternion Julia and Mandelbox.
#
# Part of the Math Art IFS engine (`math_art/ifs/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# An escape-time fractal is the set of points whose orbit under a fixed
# iteration stays bounded.  Sampling that set on a grid gives a scalar
# field -- how long the orbit survives -- which is contoured to a
# surface.  The Mandelbulb raises a point to a power in spherical
# coordinates, multiplying the polar and azimuthal angles by n; the
# Mandelbox alternates box folds with a spherical fold.
#
# References:
# - D. White and P. Nylander, the "Mandelbulb" triplex power formula,
#   2009 -- described in the Skytopia and FractalForums write-ups.
# - T. Lowe, "Mandelbox", 2010 -- the box-fold/sphere-fold iteration.
# - A. Norton, "Generation and display of geometric fractals in 3-D",
#   SIGGRAPH 1982 -- quaternion Julia sets.

import math

import numpy as np

from .voxel import _toolkit


def _mandelbulb_field(power, iters, bailout, cell):
    def field(X, Y, Z):
        shape = X.shape
        cx, cy, cz = X.ravel(), Y.ravel(), Z.ravel()
        zx = np.zeros_like(cx)
        zy = np.zeros_like(cy)
        zz = np.zeros_like(cz)
        dr = np.ones_like(cx)
        escaped = np.zeros(cx.shape, dtype=bool)
        for _ in range(iters):
            r = np.sqrt(zx * zx + zy * zy + zz * zz)
            escaped |= r > bailout
            active = ~escaped
            rr = np.clip(r, 1e-9, None)
            theta = np.arccos(np.clip(zz / rr, -1.0, 1.0))
            phi = np.arctan2(zy, zx)
            zr = rr ** power
            new_dr = power * rr ** (power - 1.0) * dr + 1.0
            t, p = theta * power, phi * power
            sin_t = np.sin(t)
            nx = zr * sin_t * np.cos(p) + cx
            ny = zr * sin_t * np.sin(p) + cy
            nz = zr * np.cos(t) + cz
            zx = np.where(active, nx, zx)
            zy = np.where(active, ny, zy)
            zz = np.where(active, nz, zz)
            dr = np.where(active, new_dr, dr)
        r = np.sqrt(zx * zx + zy * zy + zz * zz)
        rr = np.clip(r, 1e-9, None)
        de = 0.5 * np.log(rr) * rr / np.maximum(dr, 1e-9)
        f = np.where(escaped, de, -0.5 * cell)
        return f.reshape(shape)
    return field


def _quat_mul(a, b):
    """Hamilton product of two quaternion arrays (each a 4-tuple of
    grids), returned as a 4-tuple of grids."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0)


def _julia_field(c, w_slice, iters, bailout, cell):
    c0, c1, c2, c3 = c
    def field(X, Y, Z):
        shape = X.shape
        z0 = X.ravel().copy()
        z1 = Y.ravel().copy()
        z2 = Z.ravel().copy()
        z3 = np.full_like(z0, w_slice)
        dr = np.ones_like(z0)
        escaped = np.zeros(z0.shape, dtype=bool)
        for _ in range(iters):
            r = np.sqrt(z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3)
            escaped |= r > bailout
            active = ~escaped
            new_dr = 2.0 * r * dr
            s0, s1, s2, s3 = _quat_mul((z0, z1, z2, z3),
                                       (z0, z1, z2, z3))
            z0 = np.where(active, s0 + c0, z0)
            z1 = np.where(active, s1 + c1, z1)
            z2 = np.where(active, s2 + c2, z2)
            z3 = np.where(active, s3 + c3, z3)
            dr = np.where(active, new_dr, dr)
        r = np.sqrt(z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3)
        rr = np.clip(r, 1e-9, None)
        de = 0.5 * np.log(rr) * rr / np.maximum(dr, 1e-9)
        f = np.where(escaped, de, -0.5 * cell)
        return f.reshape(shape)
    return field


def _mandelbox_field(scale, min_r, fixed_r, iters, bailout, cell):
    min_r2, fixed_r2 = min_r * min_r, fixed_r * fixed_r
    def field(X, Y, Z):
        shape = X.shape
        cx, cy, cz = X.ravel(), Y.ravel(), Z.ravel()
        zx, zy, zz = cx.copy(), cy.copy(), cz.copy()
        dr = np.ones_like(cx)
        escaped = np.zeros(cx.shape, dtype=bool)
        for _ in range(iters):
            active = ~escaped

            def box_fold(v):
                v = np.where(v > 1.0, 2.0 - v, v)
                return np.where(v < -1.0, -2.0 - v, v)

            bx, by, bz = box_fold(zx), box_fold(zy), box_fold(zz)
            m2 = bx * bx + by * by + bz * bz
            factor = np.where(m2 < min_r2, fixed_r2 / min_r2,
                              np.where(m2 < fixed_r2,
                                       fixed_r2 / np.clip(m2, 1e-9, None),
                                       1.0))
            nx = scale * bx * factor + cx
            ny = scale * by * factor + cy
            nz = scale * bz * factor + cz
            new_dr = dr * factor * abs(scale) + 1.0
            zx = np.where(active, nx, zx)
            zy = np.where(active, ny, zy)
            zz = np.where(active, nz, zz)
            dr = np.where(active, new_dr, dr)
            r = np.sqrt(zx * zx + zy * zy + zz * zz)
            escaped |= r > bailout
        r = np.sqrt(zx * zx + zy * zy + zz * zz)
        de = r / np.maximum(np.abs(dr), 1e-9)
        f = np.where(escaped, de, -0.5 * cell)
        return f.reshape(shape)
    return field


# ==========================================================================
# Presets and driver
# ==========================================================================
# (label, box half-extent, default iterations, bailout)
PRESETS = {
    'MANDELBULB': ("Mandelbulb", 1.25, 10, 2.0),
    'JULIA': ("Quaternion Julia", 1.6, 12, 4.0),
    'MANDELBOX': ("Mandelbox", 6.0, 11, 6.0),
}


# a fixed quaternion constant that gives a well-known, richly detailed
# Julia set (drawn on the w = 0 slice)
_JULIA_C = (-0.2, 0.6, 0.2, 0.2)


def _largest_component(verts, tris):
    """Keep only the triangles of the largest edge-connected component
    (union-find over the mesh), dropping isolated specks. Returns
    remapped (verts, tris)."""
    if len(tris) == 0:
        return verts, tris
    n = len(verts)
    parent = np.arange(n)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:      # path compression
            parent[a], a = root, parent[a]
        return root

    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        ra, rb, rc = find(a), find(b), find(c)
        if ra != rb:
            parent[rb] = ra
        if ra != rc:
            parent[rc] = ra
    roots = np.array([find(i) for i in range(n)])
    labels, counts = np.unique(roots, return_counts=True)
    keep_root = labels[np.argmax(counts)]
    vkeep = roots == keep_root
    tkeep = vkeep[tris[:, 0]]
    tris = tris[tkeep]
    used = np.unique(tris)
    remap = np.full(n, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[tris]


def _field_for(kind, res, power, iters, julia_w):
    label, r, def_iters, bailout = PRESETS[kind]
    it = iters if iters > 0 else def_iters
    cell = 2.0 * r / res
    if kind == 'MANDELBULB':
        return _mandelbulb_field(power, it, bailout, cell), r
    if kind == 'JULIA':
        c = (_JULIA_C[0], _JULIA_C[1], _JULIA_C[2], _JULIA_C[3])
        return _julia_field(c, julia_w, it, bailout, cell), r
    return _mandelbox_field(2.0, 0.5, 1.0, it, bailout, cell), r


def build_escape_fractal(kind='MANDELBULB', res=96, power=8.0,
                         iters=0, julia_w=0.0, scale=1.0,
                         largest_only=True):
    """Mesh the boundary of an escape-time fractal. Returns
    (verts, tris) centred on the origin and fit to a 2 m cube."""
    field, r = _field_for(kind, res, power, iters, julia_w)
    mst = _toolkit()
    verts, tris = mst.marching_tets(field, (-r, -r, -r), (r, r, r),
                                    (res, res, res))
    if largest_only and len(tris):
        verts, tris = _largest_component(verts, tris)
    if len(verts):
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        ext = float((hi - lo).max())
        verts = (verts - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9
                                             else 1.0)
    return verts * scale, tris
