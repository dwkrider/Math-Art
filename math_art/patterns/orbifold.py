# Orbifold signatures and the magic theorem.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`), split out of
# the former single-file `pattern_common.py`.  Python + numpy only -- no
# `bpy` -- so the engine imports and self-tests headlessly; the registered
# operators stay in their flat generator modules and import this package.
#
# The Conway-Thurston orbifold signature: a single string that names a
# symmetry group in any of the three geometries.  Its "cost" (the magic
# theorem sum) routes it -- below 2 is spherical, exactly 2 Euclidean,
# above 2 hyperbolic.  One grammar covers the 17 wallpaper groups, the 7
# friezes, the spherical point groups and the hyperbolic tilings.
#
# References:
#   John H. Conway, Heidi Burgiel and Chaim Goodman-Strauss, "The
#     Symmetries of Things" (2008) -- the notation and the magic theorem.
#   E. S. Fedorov (1891) -- the classification of the 17 plane groups
#     the Euclidean table encodes.

from math import cos, sin, pi, hypot, gcd            # noqa: F401
import numpy as np


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
# Spherical signatures
# --------------------------------------------------------------------
# `orbifold_sphere_generator` predates this parser and names its point
# groups with ad-hoc tokens -- 'STAR_NN', 'NX', '2STAR_N' and so on --
# because a signature like `*22n` cannot be spelled in a Blender enum
# identifier.  They ARE orbifold signatures; only the spelling differs.
#
# This table is the translation, so a real signature can route through
# `geometry_of` and reach the spherical builder.  The seven infinite
# families take the order n; the seven exceptional groups do not.
#
# Reference: Conway, Burgiel and Goodman-Strauss, "The Symmetries of
# Things" (2008), chapters 3-4 -- the spherical orbifolds and their
# correspondence with the Schoenflies point groups noted below.

#: The seven infinite families, as EXPLICIT patterns.  Order matters:
#: '226' is the group 22n with n = 6 (dihedral D6), not the family nn --
#: so the patterns carrying a literal '22' must be tried first.  And 'nn'
#: means the SAME order twice, hence the backreference; '55' is C5, while
#: '54' is not a signature at all.
_SPHERICAL_PATTERNS = [
    (r'^\*22(\d+)$', 'STAR_22N', 'D{n}h'),
    (r'^22(\d+)$',    '22N',      'D{n}'),
    (r'^2\*(\d+)$',   '2STAR_N',  'D{n}d'),
    (r'^\*(\d+)\1$',  'STAR_NN',  'C{n}v'),
    (r'^(\d+)\*$',    'N_STAR',   'C{n}h'),
    (r'^(\d+)x$',     'NX',       'S{n2}'),
    (r'^(\d+)\1$',    'NN',       'C{n}'),
]

#: the seven exceptional (polyhedral) spherical groups
SPHERICAL_EXCEPTIONAL = {
    '332':  ('332',       'T'),
    '*332': ('STAR_332',  'Td'),
    '3*2':  ('3STAR_2',   'Th'),
    '432':  ('432',       'O'),
    '*432': ('STAR_432',  'Oh'),
    '532':  ('532',       'I'),
    '*532': ('STAR_532',  'Ih'),
}


def spherical_token(sig):
    """Signature -> (generator token, order n, Schoenflies name).

    Accepts the exceptional groups verbatim ('*532') and the infinite
    families with a concrete order ('*226' is the *22n family at n = 6,
    i.e. D6h).  Returns None when `sig` is not a spherical signature.
    """
    import re
    sig = sig.replace('×', 'x')
    if sig in SPHERICAL_EXCEPTIONAL:
        tok, name = SPHERICAL_EXCEPTIONAL[sig]
        return tok, None, name
    for pat, tok, name in _SPHERICAL_PATTERNS:
        m = re.match(pat, sig)
        if m:
            n = int(m.group(1))
            # S2n: the rotoreflection group's Schoenflies index is 2n
            return tok, n, name.format(n=n, n2=2 * n)
    return None


def _selftest():
    ok = True
    # THE MAGIC THEOREM: every one of the 17 wallpaper signatures must
    # cost exactly 2.  That is the whole classification -- cost 2 is the
    # Euclidean plane, and there are precisely 17 signatures that reach it.
    bad = [s for s in WALLPAPER_NAMES if abs(orbifold_cost(s) - 2.0) > 1e-9]
    good = not bad and len(WALLPAPER_NAMES) == 17
    ok &= good
    print(f"orbifold: all {len(WALLPAPER_NAMES)} wallpaper signatures cost 2 "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # and the router agrees: cost < 2 spherical, = 2 Euclidean, > 2 hyperbolic
    bad = [s for s in WALLPAPER_NAMES if geometry_of(s) != 'EUCLIDEAN']
    cases = [('532', 'SPHERICAL'), ('*532', 'SPHERICAL'), ('332', 'SPHERICAL'),
             ('732', 'HYPERBOLIC'), ('*732', 'HYPERBOLIC'),
             ('2222', 'EUCLIDEAN')]
    bad += [f"{s}->{geometry_of(s)}" for s, want in cases
            if geometry_of(s) != want]
    good = not bad
    ok &= good
    print(f"orbifold: geometry router on 17 + 6 probes "
          f"{'OK' if good else 'FAIL ' + ','.join(bad[:4])}")

    # the IUC names round-trip through SIG_OF, and the ordering is complete
    good = (all(SIG_OF[WALLPAPER_NAMES[s]] == s for s in WALLPAPER_NAMES)
            and sorted(IUC_ORDER) == sorted(WALLPAPER_NAMES.values())
            and len(IUC_ORDER) == 17)
    ok &= good
    print(f"orbifold: signature <-> IUC name round-trip "
          f"{'OK' if good else 'FAIL'}")


    # Spherical routing: a real signature must reach the right point
    # group.  The traps this pins down, all of which I got wrong first:
    #   '226' is the 22n family at n=6 (D6), NOT the nn family at n=226;
    #   'nn' means the SAME order twice, so '55' is C5 while '54' is not
    #   a signature at all; and 'nx' is the rotoreflection group S(2n),
    #   so '8x' is S16, not S8.
    cases = {'*532': ('STAR_532', None, 'Ih'), '332': ('332', None, 'T'),
             '3*2': ('3STAR_2', None, 'Th'), '*226': ('STAR_22N', 6, 'D6h'),
             '226': ('22N', 6, 'D6'), '2*3': ('2STAR_N', 3, 'D3d'),
             '55': ('NN', 5, 'C5'), '*44': ('STAR_NN', 4, 'C4v'),
             '8x': ('NX', 8, 'S16'), '5*': ('N_STAR', 5, 'C5h')}
    bad = [f"{k}->{spherical_token(k)}" for k, v in cases.items()
           if spherical_token(k) != v]
    good = not bad
    ok &= good
    print(f"orbifold: spherical routing on {len(cases)} signatures "
          f"{'OK' if good else 'FAIL ' + ', '.join(bad[:3])}")

    # non-signatures must be refused, not silently mapped
    bad = [s_ for s_ in ('54', '*54', '', 'xyz', '2') if spherical_token(s_)]
    good = not bad
    ok &= good
    print(f"orbifold: non-signatures refused "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # and every routed signature really is spherical by the cost rule
    bad = [k for k in cases if geometry_of(k) != 'SPHERICAL']
    good = not bad
    ok &= good
    print(f"orbifold: every routed signature costs < 2 "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("orbifold self-test failed")
