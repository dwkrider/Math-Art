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

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("orbifold self-test failed")
