
# Pattern Engine -- shared foundation
#
# A subsystem of generators that produce 2D patterns carrying genuine
# 3D structure: a unit cell (a motif or a saddle module) replicated by
# a symmetry across the plane and given relief, in the modular-
# constructivist tradition of Erwin Hauer and Norman Carlberg.
#
# This module is the Phase-0 foundation shared by every pattern
# generator.  It is pure Python + numpy (no bpy at import time) so the
# math is unit-testable outside Blender; the small Blender-facing
# helpers live behind an `if _IN_BLENDER:` guard at the bottom.
#
# It provides:
#   * the Conway-Thurston ORBIFOLD SIGNATURE parser -- cost ("magic
#     theorem" sum) and the geometry router that sends a signature to
#     the spherical / Euclidean / hyperbolic world;
#   * 2D isometry primitives (rotation, reflection, glide, lattice
#     translation) and a wallpaper-group table built from them;
#   * the common PLACED-TILES interchange format every tiling backend
#     emits, and the replicate/clip helpers of the Layer-3 wrapper;
#   * Mode-A relief builders -- turn 2D polygons or line segments into
#     watertight prisms on a backing slab -- plus centre/scale to the
#     2 m cube the rest of the extension uses;
#   * a Blender object builder shared by the operators.
#
# References:
#   John H. Conway, Heidi Burgiel & Chaim Goodman-Strauss, "The
#     Symmetries of Things" (2008) -- the orbifold signature notation and
#     the "magic theorem" cost accounting used throughout the engine.
#   E. S. Fedorov (1891) -- the classification of the 17 wallpaper groups
#     that the Euclidean table encodes.
#   Erwin Hauer and Norman Carlberg -- the modular-constructivist relief
#     tradition this engine builds toward.

from math import cos, sin, pi, hypot, gcd            # noqa: F401
import numpy as np


# --------------------------------------------------------------------
# Orbifold signatures (Conway-Thurston)
# --------------------------------------------------------------------
#
# A signature is a string of: digits n (order-n gyration point), '*'
# (a mirror boundary), 'x'/'×' (a glide cross), 'o' (a torus handle).
# The orbifold cost of each feature (Conway's "magic theorem"):
#
#     o                       -> 2
#     * or x                  -> 1
#     gyration digit n        -> (n-1)/n
#     digit n after a '*'     -> (n-1)/(2n)   (a corner reflector)
#
# The costs sum to exactly 2 for a Euclidean (wallpaper/frieze) group;
# < 2 is spherical (a finite point group), > 2 is hyperbolic.

def orbifold_cost(sig):
    """Sum of feature costs of an orbifold signature string."""
    total = 0.0
    star = False
    for ch in sig:
        if ch in 'oO':
            total += 2.0
        elif ch == '*':
            total += 1.0
            star = True
        elif ch in 'xX×':
            total += 1.0
        elif ch.isdigit():
            n = int(ch)
            total += (n - 1) / (2 * n) if star else (n - 1) / n
        elif ch.isspace():
            continue
        else:
            raise ValueError("bad orbifold symbol %r" % ch)
    return total


def geometry_of(sig, tol=1e-9):
    """Route a signature to 'SPHERICAL', 'EUCLIDEAN' or 'HYPERBOLIC'
    by its orbifold cost -- the unifying decision behind the sphere,
    plane and hyperbolic pattern generators."""
    c = orbifold_cost(sig)
    if c < 2.0 - tol:
        return 'SPHERICAL'
    if c > 2.0 + tol:
        return 'HYPERBOLIC'
    return 'EUCLIDEAN'


# The 17 Euclidean wallpaper groups, orbifold signature -> IUC name.
WALLPAPER_NAMES = {
    'o': 'p1', '2222': 'p2', '**': 'pm', 'xx': 'pg', '*x': 'cm',
    '*2222': 'pmm', '22*': 'pmg', '22x': 'pgg', '2*22': 'cmm',
    '442': 'p4', '*442': 'p4m', '4*2': 'p4g',
    '333': 'p3', '*333': 'p3m1', '3*3': 'p31m',
    '632': 'p6', '*632': 'p6m',
}

# canonical display order and the inverse (IUC name -> signature),
# shared by the wallpaper and layer generators
IUC_ORDER = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg',
             'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m',
             'p6', 'p6m']
SIG_OF = {v: k for k, v in WALLPAPER_NAMES.items()}


# --------------------------------------------------------------------
# 2D isometries as 3x3 homogeneous matrices acting on [x, y, 1]
# --------------------------------------------------------------------

def I():
    return np.eye(3)


def T(tx, ty):
    """Translation."""
    return np.array([[1.0, 0.0, tx],
                     [0.0, 1.0, ty],
                     [0.0, 0.0, 1.0]])


def Rot(theta, cx=0.0, cy=0.0):
    """Rotation by theta about (cx, cy)."""
    c, s = cos(theta), sin(theta)
    m = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return T(cx, cy) @ m @ T(-cx, -cy)


def Mir(angle, cx=0.0, cy=0.0):
    """Reflection across the line through (cx, cy) at `angle`."""
    c, s = cos(2 * angle), sin(2 * angle)
    m = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, 1.0]])
    return T(cx, cy) @ m @ T(-cx, -cy)


def Glide(angle, dist, cx=0.0, cy=0.0):
    """Reflect across the line at `angle` through (cx, cy), then
    translate `dist` along that line."""
    return T(dist * cos(angle), dist * sin(angle)) @ Mir(angle, cx, cy)


def apply(M, pts):
    """Apply a 3x3 homogeneous transform to an (N, 2) point array."""
    pts = np.asarray(pts, dtype=float)
    hom = np.column_stack([pts, np.ones(len(pts))])
    return (hom @ M.T)[:, :2]


# --------------------------------------------------------------------
# Wallpaper groups: lattice basis + coset representatives
# --------------------------------------------------------------------
#
# Each group is (b1, b2, cosets): the lattice is spanned by b1, b2 and
# the finite `cosets` list holds one affine representative per point-
# group element (glides carry their half-lattice translation).  The
# full symmetry set over an Nx x Ny block is every lattice translation
# composed with every coset.

_SQ = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
_HEX = (np.array([1.0, 0.0]), np.array([0.5, 0.5 * 3 ** 0.5]))


def _dihedral(n, cx, cy):
    """The 2n elements of the dihedral point group D_n about (cx, cy)
    (n rotations + n mirrors through the centre)."""
    out = [Rot(2 * pi * k / n, cx, cy) for k in range(n)]
    out += [Mir(pi * k / n, cx, cy) for k in range(n)]
    return out


def _cyclic(n, cx, cy):
    return [Rot(2 * pi * k / n, cx, cy) for k in range(n)]


def wallpaper_group(sig):
    """Return (b1, b2, cosets) for a wallpaper signature.  Rotation
    centres/mirrors are placed at the origin (cell corner)."""
    sig = sig.replace('×', 'x')
    b1, b2 = _SQ
    if sig == 'o':
        cos_ = [I()]
    elif sig == '2222':
        cos_ = _cyclic(2, 0, 0)
    elif sig == '**':
        cos_ = [I(), Mir(pi / 2)]
    elif sig == 'xx':
        cos_ = [I(), Glide(pi / 2, 0.5)]
    elif sig == '*x':                        # cm
        # a mirror on a centred (rhombic) lattice; the centring
        # translation supplies the parallel glide
        b1, b2 = np.array([1.0, 1.0]), np.array([1.0, -1.0])
        cos_ = [I(), Mir(pi / 2)]
    elif sig == '*2222':
        cos_ = [I(), Rot(pi), Mir(0.0), Mir(pi / 2)]
    elif sig == '22*':                       # pmg
        # vertical mirrors sit at x = 1/4, off the 2-fold centres
        cos_ = [I(), Rot(pi), Mir(pi / 2, 0.25, 0.0), Glide(0.0, 0.5)]
    elif sig == '22x':                       # pgg
        # glide axes offset 1/4 from the 2-fold centres
        cos_ = [I(), Rot(pi),
                Glide(0.0, 0.5, 0.0, 0.25),
                Glide(pi / 2, 0.5, 0.25, 0.0)]
    elif sig == '2*22':                      # cmm
        b1, b2 = np.array([1.0, 1.0]), np.array([1.0, -1.0])
        cos_ = [I(), Rot(pi), Mir(0.0), Mir(pi / 2)]
    elif sig == '442':
        cos_ = _cyclic(4, 0, 0)
    elif sig == '*442':
        cos_ = _dihedral(4, 0, 0)
    elif sig == '4*2':                        # p4g
        # diagonal mirrors run through the 2-fold centres at (1/2, 0)
        cos_ = _cyclic(4, 0, 0) + [Rot(k * pi / 2) @ Mir(pi / 4, 0.5, 0.0)
                                   for k in range(4)]
    elif sig == '333':
        b1, b2 = _HEX
        cos_ = _cyclic(3, 0, 0)
    elif sig == '*333':
        # p3m1: every 3-fold centre lies on a mirror -> mirrors at
        # 30/90/150 (through the hexagonal lattice's deep holes)
        b1, b2 = _HEX
        cos_ = _cyclic(3, 0, 0) + [Mir(pi / 6), Mir(pi / 2),
                                   Mir(5 * pi / 6)]
    elif sig == '3*3':
        # p31m: the deep-hole 3-fold centres are OFF the mirrors ->
        # mirrors at 0/60/120 (along the lattice directions)
        b1, b2 = _HEX
        cos_ = _dihedral(3, 0, 0)
    elif sig == '632':
        b1, b2 = _HEX
        cos_ = _cyclic(6, 0, 0)
    elif sig == '*632':
        b1, b2 = _HEX
        cos_ = _dihedral(6, 0, 0)
    else:
        raise ValueError("unknown wallpaper signature %r" % sig)
    return b1, b2, cos_


def group_closes(sig, tol=1e-6):
    """Regression guard: verify a wallpaper group's coset table is a
    real group -- every product of two cosets lands back on
    lattice x cosets, and no two cosets coincide modulo the lattice
    (a duplicated or mis-placed coset both fail this).  This is what
    catches the p3m1/p31m/cm/pmg/pgg/p4g class of errors that a
    count-only self-test misses."""
    b1, b2, cosets = wallpaper_group(sig)
    b1, b2 = np.asarray(b1, float), np.asarray(b2, float)
    refs = [T(m * b1[0] + n * b2[0], m * b1[1] + n * b2[1]) @ c
            for m in range(-2, 3) for n in range(-2, 3) for c in cosets]

    def on_lattice(M):
        return any(np.allclose(M, R, atol=tol) for R in refs)

    for g in cosets:                          # closure
        for h in cosets:
            if not on_lattice(g @ h):
                return False
    for i in range(len(cosets)):              # distinct modulo lattice
        for j in range(i + 1, len(cosets)):
            for m in (-1, 0, 1):
                for n in (-1, 0, 1):
                    t = T(m * b1[0] + n * b2[0], m * b1[1] + n * b2[1])
                    if np.allclose(cosets[i], t @ cosets[j], atol=tol):
                        return False
    return True


def wallpaper_isometries(sig, nx, ny):
    """All isometries of a wallpaper group over an nx x ny run of the
    lattice, as a list of 3x3 matrices."""
    b1, b2, cosets = wallpaper_group(sig)
    out = []
    for i in range(nx):
        for j in range(ny):
            t = T(i * b1[0] + j * b2[0], i * b1[1] + j * b2[1])
            for g in cosets:
                out.append(t @ g)
    return out


# --------------------------------------------------------------------
# Frieze groups (7): 1D strip patterns
# --------------------------------------------------------------------
#
# The lattice is the single translation (1, 0); operations use the
# same origin-based convention as the wallpaper groups.

FRIEZE_ORDER = ['p1', 'p11g', 'p1m1', 'p11m', 'p2', 'p2mg', 'p2mm']
FRIEZE_NAMES = {                     # key -> (label, orbifold)
    'p1':   ('Hop', 'oo'),
    'p11g': ('Step (glide)', 'ox'),
    'p1m1': ('Sidle (vertical mirror)', '*oo'),
    'p11m': ('Jump (horizontal mirror)', 'o*'),
    'p2':   ('Spinning hop (180)', '22o'),
    'p2mg': ('Spinning sidle', '2*o'),
    'p2mm': ('Spinning jump', '*22o'),
}


def frieze_group(name):
    """Coset representatives of a frieze group (lattice = the 1D
    translation along x)."""
    if name == 'p1':
        return [I()]
    if name == 'p11g':
        return [I(), Glide(0.0, 0.5)]                 # glide along x
    if name == 'p1m1':
        return [I(), Mir(pi / 2)]                     # vertical mirror
    if name == 'p11m':
        return [I(), Mir(0.0)]                        # horizontal mirror
    if name == 'p2':
        return [I(), Rot(pi)]
    if name == 'p2mg':
        return [I(), Rot(pi), Mir(pi / 2, 0.25, 0.0), Glide(0.0, 0.5)]
    if name == 'p2mm':
        return [I(), Rot(pi), Mir(0.0), Mir(pi / 2)]
    raise ValueError("unknown frieze group %r" % name)


# --------------------------------------------------------------------
# Motif library + color classification (shared by the pattern ops)
# --------------------------------------------------------------------

def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _comma():
    """A chiral paisley: a tapering ribbon swept along an arc with a
    round head, so its handedness reveals reflections and glides."""
    t = np.linspace(0.0, 1.0, 22)
    a0, a1 = np.deg2rad(205.0), np.deg2rad(-25.0)
    ang = a0 + (a1 - a0) * t
    R, cx, cy = 0.24, 0.52, 0.50
    spine = np.column_stack([cx + R * np.cos(ang), cy + R * np.sin(ang)])
    half = 0.135 * (1.0 - t) ** 0.7 + 0.004
    d = np.gradient(spine, axis=0)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    left = spine + nrm * half[:, None]
    right = spine - nrm * half[:, None]
    c0, h0 = spine[0], half[0]
    base = np.arctan2(right[0, 1] - c0[1], right[0, 0] - c0[0])
    cap = np.array([[c0[0] + h0 * np.cos(base + np.pi * s),
                     c0[1] + h0 * np.sin(base + np.pi * s)]
                    for s in np.linspace(0.0, 1.0, 9)])
    if np.dot(cap[len(cap) // 2] - c0, -d[0]) < 0:
        cap = np.array([[c0[0] + h0 * np.cos(base - np.pi * s),
                         c0[1] + h0 * np.sin(base - np.pi * s)]
                        for s in np.linspace(0.0, 1.0, 9)])
    return [np.vstack([left, right[::-1], cap])]


MOTIFS = ('ARROW', 'F', 'L', 'COMMA', 'ZIG', 'TRIANGLE')


def motif(kind):
    """One or more asymmetric polygons in the unit cell [0, 1]^2.  All
    are chiral so reflections and glides in the group stay visible."""
    if kind == 'F':
        return [_rect(0.30, 0.18, 0.42, 0.82),
                _rect(0.42, 0.70, 0.72, 0.82),
                _rect(0.42, 0.46, 0.64, 0.58)]
    if kind == 'L':
        return [_rect(0.32, 0.20, 0.48, 0.80),
                _rect(0.48, 0.20, 0.74, 0.36)]
    if kind == 'ARROW':
        return [np.array([[0.22, 0.44], [0.58, 0.44], [0.58, 0.30],
                          [0.82, 0.52], [0.58, 0.66], [0.58, 0.56],
                          [0.22, 0.56]])]
    if kind == 'COMMA':
        return _comma()
    if kind == 'ZIG':
        return [_rect(0.26, 0.66, 0.72, 0.80),
                _rect(0.26, 0.20, 0.72, 0.34),
                _rect(0.40, 0.20, 0.56, 0.80)]
    return [np.array([[0.25, 0.2], [0.78, 0.32], [0.35, 0.8]])]  # TRI


def iso_type(g):
    """Classify a coset isometry: 0 identity, 1 rotation, 2 reflection,
    3 glide."""
    A = g[:2, :2]
    if A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0] > 0.0:
        return 0 if np.allclose(A, np.eye(2), atol=1e-6) else 1
    return 2 if np.allclose(g @ g, np.eye(3), atol=1e-6) else 3


def kind_of(color_by, M, g, cell, gi):
    """Material index for one replicated copy, per color mode."""
    if color_by == 'OP':
        return iso_type(g)
    if color_by == 'HAND':
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        return 0 if det > 0 else 1
    if color_by == 'CELL':
        return cell % len(PALETTE_RGBA)      # wrap: avoid a material per cell
    return gi                                            # COPY


# a 12-way categorical palette (shared by every generator's coloring);
# the Blender PALETTE below aliases this so there is one source of truth
PALETTE_RGBA = [(0.85, 0.30, 0.24, 1.0), (0.20, 0.45, 0.70, 1.0),
                (0.95, 0.77, 0.30, 1.0), (0.35, 0.65, 0.38, 1.0),
                (0.60, 0.35, 0.68, 1.0), (0.90, 0.55, 0.22, 1.0),
                (0.30, 0.72, 0.72, 1.0), (0.82, 0.47, 0.60, 1.0),
                (0.52, 0.55, 0.25, 1.0), (0.45, 0.40, 0.75, 1.0),
                (0.72, 0.60, 0.45, 1.0), (0.42, 0.42, 0.47, 1.0)]


# --------------------------------------------------------------------
# Placed-tiles interchange format
# --------------------------------------------------------------------

class Tiling:
    """The common output of every tiling backend.

    * polys    -- list of (M, 2) arrays: filled tile polygons
    * segments -- list of ((x0, y0), (x1, y1)) motif line segments
    * types    -- parallel to polys: an int per tile for coloring
    """

    def __init__(self):
        self.polys = []
        self.segments = []
        self.types = []

    def add_poly(self, pts, kind=0):
        self.polys.append(np.asarray(pts, dtype=float))
        self.types.append(int(kind))

    def add_segment(self, a, b):
        self.segments.append((tuple(a), tuple(b)))

    def replicate(self, isometries):
        """Return a new Tiling with polys and segments imaged under
        every transform in `isometries`."""
        out = Tiling()
        for M in isometries:
            for p, k in zip(self.polys, self.types):
                out.add_poly(apply(M, p), k)
            for a, b in self.segments:
                q = apply(M, [a, b])
                out.add_segment(q[0], q[1])
        return out

    def bbox(self):
        pts = []
        pts += [p for p in self.polys]
        if self.segments:
            pts.append(np.array([c for s in self.segments for c in s]))
        if not pts:
            return np.zeros(2), np.zeros(2)
        allp = np.vstack(pts)
        return allp.min(axis=0), allp.max(axis=0)

    def clip(self, lo, hi, margin=0.51):
        """Keep polys/segments whose centroid lies in [lo, hi]
        (padded by `margin` cells so partial tiles at the frame stay)."""
        out = Tiling()
        lo = np.asarray(lo) - margin
        hi = np.asarray(hi) + margin
        for p, k in zip(self.polys, self.types):
            c = p.mean(axis=0)
            if np.all(c >= lo) and np.all(c <= hi):
                out.add_poly(p, k)
        for a, b in self.segments:
            c = 0.5 * (np.array(a) + np.array(b))
            if np.all(c >= lo) and np.all(c <= hi):
                out.add_segment(a, b)
        return out


# --------------------------------------------------------------------
# Mode A -- relief geometry (2D polygons/segments -> 3D prisms)
# --------------------------------------------------------------------

def ribbon_polys(segments, width):
    """Turn line segments into rectangular ribbon polygons of the
    given width (strapwork).  Overlaps at joints are intentional --
    the shells union cleanly for print or render."""
    w = 0.5 * width
    polys = []
    for (x0, y0), (x1, y1) in segments:
        dx, dy = x1 - x0, y1 - y0
        L = hypot(dx, dy)
        if L < 1e-9:
            continue
        nx_, ny_ = -dy / L * w, dx / L * w
        polys.append(np.array([[x0 + nx_, y0 + ny_],
                               [x1 + nx_, y1 + ny_],
                               [x1 - nx_, y1 - ny_],
                               [x0 - nx_, y0 - ny_]]))
    return polys


def prisms(verts, faces, mats, polys2d, z_top, z_bot, mat=0):
    """Append watertight prisms (one per 2D polygon) between z_bot and
    z_top.  Top n-gon, reversed bottom n-gon, side quads."""
    for poly in polys2d:
        n = len(poly)
        if n < 3:
            continue
        b0 = len(verts)
        for x, y in poly:
            verts.append((float(x), float(y), z_top))
        for x, y in poly:
            verts.append((float(x), float(y), z_bot))
        faces.append(tuple(b0 + k for k in range(n)))
        faces.append(tuple(b0 + n + k for k in reversed(range(n))))
        for k in range(n):
            k2 = (k + 1) % n
            faces.append((b0 + k, b0 + n + k, b0 + n + k2, b0 + k2))
        mats.extend([mat] * (n + 2))


def slab(verts, faces, mats, lo, hi, z_top, z_bot, mat=1):
    """A single box backing slab spanning the xy rectangle lo..hi."""
    x0, y0 = lo
    x1, y1 = hi
    prisms(verts, faces, mats,
           [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])],
           z_top, z_bot, mat)


def center_scale(verts, span=2.0):
    """Centre a vertex list at the origin and scale so its larger in-
    plane extent spans `span` (the 2 m cube convention)."""
    if not verts:
        return verts
    a = np.asarray(verts, dtype=float)
    lo, hi = a.min(axis=0), a.max(axis=0)
    c = 0.5 * (lo + hi)
    ext = max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
    s = span / ext
    a = (a - c) * s
    return [tuple(v) for v in a]


def center_xy(verts):
    """Centre a vertex list on the origin in X and Y (leaving Z), so
    the object's origin sits at the pattern centre and the object's
    transform can then place/orient it anywhere."""
    if not verts:
        return verts
    a = np.asarray(verts, dtype=float)
    cx = 0.5 * (a[:, 0].min() + a[:, 0].max())
    cy = 0.5 * (a[:, 1].min() + a[:, 1].max())
    a[:, 0] -= cx
    a[:, 1] -= cy
    return [tuple(v) for v in a]


# --------------------------------------------------------------------
# Cells: per-unit pieces, merged into one mesh or emitted separately
# --------------------------------------------------------------------
#
# Every pattern generator produces a list of "cells": one
# (verts, faces, mats) piece per replicated unit (motif copy or tile).
# merge_cells() concatenates them into a single mesh; emit() (Blender)
# either merges or builds one child object per cell so each can be
# edited individually.

def merge_cells(cells):
    """Concatenate per-cell (verts, faces, mats) pieces into a single
    (verts, faces, mats), offsetting the face indices of each cell."""
    verts, faces, mats = [], [], []
    for cv, cf, cm in cells:
        b0 = len(verts)
        verts.extend(cv)
        for f in cf:
            faces.append(tuple(b0 + idx for idx in f))
        mats.extend(cm)
    return verts, faces, mats


def _global_transform(verts, fit, span):
    """(cx, cy, cz, s) mapping the pattern into place: with fit it is
    centred and scaled to `span` (matching center_scale); without fit
    it is only centred in XY (matching center_xy)."""
    a = np.asarray(verts, dtype=float)
    lo, hi = a.min(axis=0), a.max(axis=0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    if fit:
        cz = 0.5 * (lo[2] + hi[2])
        s = span / max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
    else:
        cz, s = 0.0, 1.0
    return cx, cy, cz, s


def _apply_transform(verts, p):
    cx, cy, cz, s = p
    return [((x - cx) * s, (y - cy) * s, (z - cz) * s)
            for x, y, z in verts]


# --------------------------------------------------------------------
# Blender object builder (shared by the pattern operators)
# --------------------------------------------------------------------

try:
    import bpy
    import bmesh
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    PALETTE = PALETTE_RGBA               # one source of truth (above)

    def _material(name, color):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = color
            mat.use_nodes = True
            node = mat.node_tree.nodes.get("Principled BSDF")
            if node:
                node.inputs[0].default_value = color
        return mat

    def build_object(context, name, verts, faces, mats,
                     palette=None, span=2.0, fit=True, operator=None):
        """Build the mesh, assign per-face materials from a small
        palette, recalc normals, and add it to the scene.  With
        fit=True the result is centred and scaled to the `span` cube;
        with fit=False the vertices keep their original size but are
        still centred in XY so the object origin is the pattern centre.

        The pattern is built in local space; if `operator` is an
        AddObjectHelper it is placed via object_data_add, honouring the
        operator's Align (World / 3D Cursor / View), Location and
        Rotation -- so the pattern can land on any plane and be moved
        or reoriented afterwards.  Returns the object (or None if
        empty)."""
        if not faces:
            return None
        verts = center_scale(verts, span) if fit else center_xy(verts)
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        cols = palette or PALETTE
        nmat = (max(mats) + 1) if mats else 1
        for i in range(nmat):
            me.materials.append(_material(
                "%s %d" % (name, i), cols[i % len(cols)]))
        if mats:
            me.polygons.foreach_set('material_index', mats)
        me.validate(clean_customdata=True)
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        me.update()
        if operator is not None:
            from bpy_extras import object_utils
            obj = object_utils.object_data_add(context, me,
                                               operator=operator)
        else:
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return obj

    def _palette_material(i, cols=None):
        """A shared palette material by color index, reused across all
        cells and generators (so separate cells don't spawn duplicate
        material datablocks)."""
        cols = cols or PALETTE
        return _material("Math Art Pattern %d" % i, cols[i % len(cols)])

    def emit(context, name, cells, separate=False, fit=True, span=2.0,
             operator=None):
        """Turn per-cell pieces into scene geometry.  separate=False
        builds one merged object (via build_object); separate=True
        builds one child object per cell under a parent empty, so each
        cell can be edited on its own.  A single global centre/scale is
        applied so every cell keeps its relative place, and the parent
        follows the operator's Align."""
        cells = [c for c in cells if c[1]]           # drop empty cells
        if not cells:
            return None
        if not separate:
            v, f, m = merge_cells(cells)
            return build_object(context, name, v, f, m, fit=fit,
                                span=span, operator=operator)
        allv = [vv for c in cells for vv in c[0]]
        p = _global_transform(allv, fit, span)
        parent = bpy.data.objects.new(name, None)    # an empty
        context.collection.objects.link(parent)
        if operator is not None:
            from bpy_extras import object_utils
            parent.matrix_world = object_utils.add_object_align_init(
                context, operator)
        else:
            parent.location = context.scene.cursor.location
        for idx, (cv, cf, cm) in enumerate(cells):
            me = bpy.data.meshes.new("%s Cell %d" % (name, idx))
            me.from_pydata(_apply_transform(cv, p), [], cf)
            for i in range((max(cm) + 1) if cm else 1):
                me.materials.append(_palette_material(i))
            if cm:
                me.polygons.foreach_set('material_index', cm)
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            me.update()
            obj = bpy.data.objects.new(me.name, me)
            obj["math_art_pattern"] = True
            context.collection.objects.link(obj)
            obj.parent = parent
        for o in context.selected_objects:
            o.select_set(False)
        parent.select_set(True)
        context.view_layer.objects.active = parent
        return parent

    # Not a generator: contributes no menu entry / operator.
    ADD_MENU = False

    def register():
        pass

    def unregister():
        pass


if __name__ == "__main__":
    # Self-test of the orbifold router: all 17 wallpaper signatures
    # must be Euclidean (cost 2), sample point groups spherical, and a
    # {7,3,2} triangle group hyperbolic.
    bad = [s for s in WALLPAPER_NAMES if geometry_of(s) != 'EUCLIDEAN']
    sph = geometry_of('532') == 'SPHERICAL'          # icosahedral
    hyp = geometry_of('732') == 'HYPERBOLIC'
    grp_ok = all(len(wallpaper_isometries(s, 2, 2)) > 0
                 for s in WALLPAPER_NAMES)
    frieze_ok = all(len(frieze_group(n)) > 0 for n in FRIEZE_ORDER)
    motif_ok = all(len(motif(k)) > 0 for k in MOTIFS)
    not_closed = [WALLPAPER_NAMES[s] for s in WALLPAPER_NAMES
                  if not group_closes(s)]
    print("wallpaper Euclidean:", not bad, "| bad:", bad)
    print("532 spherical:", sph, "| 732 hyperbolic:", hyp)
    print("groups build:", grp_ok, "| frieze:", frieze_ok,
          "| motifs:", motif_ok)
    print("group closure:", not not_closed, "| not closed:", not_closed)
    print("RESULT:", "OK" if (not bad and sph and hyp and grp_ok
                              and frieze_ok and motif_ok
                              and not not_closed) else "BAD")
