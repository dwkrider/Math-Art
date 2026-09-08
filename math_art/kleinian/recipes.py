# Parameterised families of two-generator Kleinian groups.
#
# GRANDMA'S RECIPE builds a pair of Moebius maps from their traces, so the whole
# deformation space of once-punctured-torus groups is three complex numbers.
# Given t_a and t_b, take
#
#     t_ab = ( t_a t_b - sqrt( t_a^2 t_b^2 - 4 (t_a^2 + t_b^2) ) ) / 2
#     z_0  = ( (t_ab - 2) t_b ) / ( t_b t_ab - 2 t_a + 2 i t_ab )
#
#     a = [ t_a/2                              (t_a t_ab - 2 t_b + 4i)
#                                              / ((2 t_ab + 4) z_0)      ]
#         [ (t_a t_ab - 2 t_b - 4i) z_0                                  ]
#         [   / (2 t_ab - 4)                   t_a/2                     ]
#
#     b = [ (t_b - 2i)/2   t_b/2 ;   t_b/2   (t_b + 2i)/2 ]
#
# The choice of t_ab is exactly what forces the commutator to be parabolic:
#
#     tr[a,b] = t_a^2 + t_b^2 + t_ab^2 - t_a t_b t_ab - 2
#
# so tr[a,b] = -2 is the condition t_a^2 + t_b^2 + t_ab^2 = t_a t_b t_ab, with
# NO constant term.  (Rescaling t = 3x turns that into x^2 + y^2 + z^2 = 3xyz,
# the Markov equation -- which is why punctured-torus groups and Markov triples
# share a tree.)  A parabolic commutator is what pinches the quotient surface
# into a once-punctured torus.
#
# THE MASKIT SLICE is the sub-family with b parabolic, parameterised by a single
# complex mu:
#
#     a(z) = mu + 1/z        b(z) = z + 2
#
# with tr a = -i mu and tr b = 2 identically.  Groups in the slice need
# Im(mu) > 1.
#
# Two parameter choices worth knowing: t_a = 1.91 + 0.05i, t_b = 3 gives a wild
# quasifuchsian Jordan curve; t_a = t_b = 2 makes every generator parabolic too
# and the limit set closes into a gasket of tangent circles.
#
# References:
# - David Mumford, Caroline Series and David Wright, "Indra's Pearls: The Vision
#   of Felix Klein", Cambridge University Press, 2002 (Box 21, Grandma's special
#   parabolic commutator groups; Box 23; chapter 9 for the Maskit slice).
# - Bernard Maskit, "Kleinian Groups", Springer Grundlehren 287, 1988.

import cmath

from . import mobius
from .mobius import mat, mul, inv, trace

GRANDMA = 'GRANDMA'
MASKIT = 'MASKIT'
FAMILIES = (GRANDMA, MASKIT)

# Parameters that give the two characteristic pictures.
PRESETS = {
    'QUASIFUCHSIAN': (1.91 + 0.05j, 3.0 + 0.0j),
    'GASKET': (2.0 + 0.0j, 2.0 + 0.0j),
    'SPIRAL': (1.87 + 0.1j, 1.87 - 0.1j),
    'NEAR_FUCHSIAN': (1.95 + 0.02j, 3.0 + 0.0j),
}


def grandma(t_a, t_b):
    """Grandma's recipe: two generators from two traces.

    Returns (a, b, t_ab).  The commutator [a,b] is parabolic by construction."""
    t_a = complex(t_a)
    t_b = complex(t_b)
    disc = cmath.sqrt(t_a * t_a * t_b * t_b - 4.0 * (t_a * t_a + t_b * t_b))
    t_ab = (t_a * t_b - disc) / 2.0
    den = t_b * t_ab - 2.0 * t_a + 2j * t_ab
    if abs(den) < 1e-300:
        raise ValueError("degenerate traces for Grandma's recipe")
    z0 = ((t_ab - 2.0) * t_b) / den
    a = mat(t_a / 2.0,
            (t_a * t_ab - 2.0 * t_b + 4j) / ((2.0 * t_ab + 4.0) * z0),
            (t_a * t_ab - 2.0 * t_b - 4j) * z0 / (2.0 * t_ab - 4.0),
            t_a / 2.0)
    b = mat((t_b - 2j) / 2.0, t_b / 2.0,
            t_b / 2.0, (t_b + 2j) / 2.0)
    return a, b, t_ab


def maskit(mu):
    """The Maskit slice: a(z) = mu + 1/z, b(z) = z + 2."""
    mu = complex(mu)
    a = mat(-1j * mu, -1j, -1j, 0.0)
    b = mat(1.0, 2.0, 0.0, 1.0)
    return a, b


def generators(family=GRANDMA, t_a=1.91 + 0.05j, t_b=3.0, mu=2.0j):
    """The four generators in the CYCLIC ORDER a, B, A, b.

    The order matters: the depth-first search in `limitset` walks branches by
    stepping around this cycle, and with the wrong order it still avoids
    cancellation but no longer traces the limit set continuously."""
    if family == MASKIT:
        a, b = maskit(mu)
    elif family == GRANDMA:
        a, b, _ = grandma(t_a, t_b)
    else:
        raise ValueError("unknown family %r" % (family,))
    return [a, inv(b), inv(a), b]          # a, B, A, b


def commutator_trace(a, b):
    return trace(mul(mul(a, b), mul(inv(a), inv(b))))


def markov_residual(t_a, t_b, t_ab):
    """t_a^2 + t_b^2 + t_ab^2 - t_a t_b t_ab; zero exactly when tr[a,b] = -2."""
    return t_a * t_a + t_b * t_b + t_ab * t_ab - t_a * t_b * t_ab


def _selftest():
    # 1. Grandma's recipe: unimodular generators with the prescribed traces,
    #    and a PARABOLIC COMMUTATOR -- the defining condition
    for name, (ta, tb) in PRESETS.items():
        a, b, t_ab = grandma(ta, tb)
        assert abs(mobius.det(a) - 1.0) < 1e-9, (name, mobius.det(a))
        assert abs(mobius.det(b) - 1.0) < 1e-9, (name, mobius.det(b))
        assert abs(trace(a) - ta) < 1e-9, (name, trace(a), ta)
        assert abs(trace(b) - tb) < 1e-9, (name, trace(b), tb)
        assert abs(trace(mul(a, b)) - t_ab) < 1e-9, name
        ct = commutator_trace(a, b)
        assert abs(ct + 2.0) < 1e-9, \
            "%s: tr[a,b] must be -2, got %r" % (name, ct)
        # and the identity with NO constant term
        assert abs(markov_residual(ta, tb, t_ab)) < 1e-9, \
            "%s: t_a^2+t_b^2+t_ab^2 = t_a t_b t_ab must hold" % name
        # the form WITH a -2 is the common error: it equals -2, not 0
        assert abs(markov_residual(ta, tb, t_ab) - 2.0 - (-2.0)) < 1e-9

    # 2. the Markov rescaling t = 3x
    for (x, y, z) in ((1, 1, 1), (1, 1, 2), (1, 2, 5), (1, 5, 13), (2, 5, 29)):
        assert abs(x * x + y * y + z * z - 3 * x * y * z) < 1e-12
        assert abs(markov_residual(3.0 * x, 3.0 * y, 3.0 * z)) < 1e-9

    # 3. Maskit: tr a = -i mu, tr b = 2, and a is parabolic exactly at mu = 2i
    a, b = maskit(2j)
    assert abs(mobius.det(a) - 1.0) < 1e-12
    assert abs(trace(b) - 2.0) < 1e-12
    assert abs(trace(a) - 2.0) < 1e-12, trace(a)
    assert mobius.is_parabolic(a), "a must be parabolic at mu = 2i"
    a2, _ = maskit(2.0)
    assert abs(trace(a2) + 2j) < 1e-12
    assert not mobius.is_parabolic(a2), \
        "mu = 2 gives tr = -2i, modulus 2 but not parabolic"

    # 4. the generator list is in the cyclic order a, B, A, b, so that index
    #    i+2 (mod 4) is always the inverse of index i
    gens = generators(GRANDMA, *PRESETS['QUASIFUCHSIAN'])
    assert len(gens) == 4
    for i in range(4):
        prod = mul(gens[i], gens[(i + 2) % 4])
        assert all(abs(x - y) < 1e-9 for x, y in zip(prod, mobius.IDENT)), \
            "gens[%d] and gens[%d] must be inverses" % (i, (i + 2) % 4)

    print("kleinian.recipes: Grandma's recipe gives det 1, prescribed traces "
          "and tr[a,b] = -2; Markov identity has no constant term; Maskit "
          "parabolic at mu = 2i; generators cyclically ordered. RESULT: OK")
