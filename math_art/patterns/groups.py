# Wallpaper and frieze groups.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`), split out of
# the former single-file `pattern_common.py`.  Python + numpy only -- no
# `bpy` -- so the engine imports and self-tests headlessly; the registered
# operators stay in their flat generator modules and import this package.
#
# The 17 wallpaper groups and the 7 frieze groups, each as a lattice
# basis plus the coset representatives that generate it from the
# translation subgroup.
#
# References:
#   E. S. Fedorov (1891); George Polya (1924) -- the plane groups.
#   Conway-Burgiel-Goodman-Strauss (2008) -- the orbifold names used as
#     the primary keys here.

from math import cos, sin, pi, hypot, gcd            # noqa: F401
import numpy as np
from .isometry import Glide, I, Mir, Rot, T, apply
from .orbifold import WALLPAPER_NAMES, orbifold_cost


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
    return [M for M, _w in wallpaper_isometries_tagged(sig, nx, ny)]


def wallpaper_isometries_tagged(sig, nx, ny):
    """As `wallpaper_isometries`, but each matrix is paired with the GROUP
    WORD that produced it: `(lattice_i, lattice_j, coset_index)`.

    The word is what turns a pile of matrices back into a group.  Two
    things need it and neither can be done without it:

      * COLOUR SYMMETRY.  Colouring by a homomorphism G -> S_n (a perfect
        colouring, in the Conway-Burgiel-Goodman-Strauss sense) needs to
        know WHICH group element placed each copy.  Colouring by cell
        index cannot express it -- the colours would not permute under the
        group action.
      * The placements layer of the tiling interchange format, which
        records (prototile, transform, word) rather than baked geometry.

    The loop below already computed the word and threw it away; this
    returns it.  `wallpaper_isometries` is unchanged and still returns
    bare matrices, so nothing that exists today has to move.
    """
    b1, b2, cosets = wallpaper_group(sig)
    out = []
    for i in range(nx):
        for j in range(ny):
            t = T(i * b1[0] + j * b2[0], i * b1[1] + j * b2[1])
            for k, g in enumerate(cosets):
                out.append((t @ g, (i, j, k)))
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


def _selftest():
    ok = True
    # Every one of the 17 groups must actually CLOSE: composing any two
    # coset representatives must land back in the group, modulo the
    # lattice.  This is the real test of the table -- a wrong generator
    # still produces a pattern, just not that group's pattern.
    bad = [s for s in WALLPAPER_NAMES if not group_closes(s)]
    good = not bad
    ok &= good
    print(f"groups: all 17 wallpaper groups close under composition "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # The number of cosets is the order of the point group.  Checked
    # against the classical values for the groups whose point group is
    # unambiguous: p1 trivial, p3 = C3, p4 = C4, p6 = C6, p4m = D4,
    # p6m = D6, pmm = D2.  wallpaper_group returns (b1, b2, cosets).
    orders = {'o': 1, '333': 3, '442': 4, '632': 6,
              '*442': 8, '*632': 12, '*2222': 4}
    bad = []
    for s_, want in orders.items():
        n = len(wallpaper_group(s_)[2])
        if n != want:
            bad.append(f"{s_}:{n}!={want}")
    # and every point group must have one of the orders the
    # crystallographic restriction permits: the cyclic C1..C6 (n = 1, 2,
    # 3, 4, 6 -- no 5-fold) and the dihedral D1..D6 (twice those).  So
    # 8 is legal (D4, as in p4m and p4g) even though it does not divide 12.
    legal = {1, 2, 3, 4, 6, 8, 12}
    for s_ in WALLPAPER_NAMES:
        n = len(wallpaper_group(s_)[2])
        if n not in legal:
            bad.append(f"{s_}:order {n}")
    good = not bad
    ok &= good
    print(f"groups: point-group orders match the classical values "
          f"{'OK' if good else 'FAIL ' + ','.join(bad[:4])}")

    # replicating over a 2x2 patch yields cosets * 4 isometries, all finite
    bad = []
    for s in WALLPAPER_NAMES:
        iso = wallpaper_isometries(s, 2, 2)
        if not iso or not all(np.all(np.isfinite(m)) for m in iso):
            bad.append(s)
    good = not bad
    ok &= good
    print(f"groups: wallpaper_isometries finite over a 2x2 patch "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # the 7 friezes are present and produce finite generators
    bad = [n for n in FRIEZE_ORDER
           if not all(np.all(np.isfinite(m)) for m in frieze_group(n))]
    good = not bad and len(FRIEZE_ORDER) == 7 and len(FRIEZE_NAMES) == 7
    ok &= good
    print(f"groups: all 7 frieze groups build "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("groups self-test failed")
