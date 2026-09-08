# Moebius transformations as 2x2 complex matrices.
#
# A Moebius map z -> (a z + b) / (c z + d) is represented by its matrix
# [a b; c d], normalised to determinant 1 so that composition is matrix
# multiplication and the trace classifies the map:
#
#     |tr| < 2 (real)   elliptic     a rotation about two fixed points
#     tr = +/-2         parabolic    one fixed point; the boundary case
#     otherwise         loxodromic   attracting and repelling fixed points
#
# PARABOLIC MEANS tr = +/-2 EXACTLY, NOT |tr| = 2.  Traces here are complex, so
# testing abs(abs(tr) - 2) < eps silently accepts non-parabolic elements: the
# Maskit parameter mu = 2 gives tr = -2i, whose modulus is 2 and which is not
# parabolic at all.  The test is min(|tr - 2|, |tr + 2|).
#
# References:
# - David Mumford, Caroline Series and David Wright, "Indra's Pearls: The Vision
#   of Felix Klein", Cambridge University Press, 2002 (Box 7, matrix algebra and
#   Moebius maps; Box 9, classification by trace).
# - Alan F. Beardon, "The Geometry of Discrete Groups", Springer GTM 91, 1983.

import cmath

ELLIPTIC = 'ELLIPTIC'
PARABOLIC = 'PARABOLIC'
LOXODROMIC = 'LOXODROMIC'
IDENTITY = 'IDENTITY'


def mat(a, b, c, d):
    return (complex(a), complex(b), complex(c), complex(d))


IDENT = mat(1, 0, 0, 1)


def mul(m, n):
    a, b, c, d = m
    e, f, g, h = n
    return (a * e + b * g, a * f + b * h,
            c * e + d * g, c * f + d * h)


def inv(m):
    """Inverse, assuming determinant 1 (the adjugate)."""
    a, b, c, d = m
    return (d, -b, -c, a)


def det(m):
    a, b, c, d = m
    return a * d - b * c


def trace(m):
    return m[0] + m[3]


def normalise(m):
    """Rescale to determinant 1."""
    D = det(m)
    if abs(D) < 1e-300:
        raise ValueError("singular matrix")
    s = cmath.sqrt(D)
    return (m[0] / s, m[1] / s, m[2] / s, m[3] / s)


def apply(m, z):
    """Act on a point of the Riemann sphere (inf is not represented)."""
    a, b, c, d = m
    den = c * z + d
    if abs(den) < 1e-300:
        return complex(float('inf'), 0.0)
    return (a * z + b) / den


def word(gens, letters):
    """Product of generators named by index."""
    m = IDENT
    for i in letters:
        m = mul(m, gens[i])
    return m


def is_parabolic(m, eps=1e-9):
    """tr = +/-2 exactly -- NOT |tr| = 2; see the module header."""
    t = trace(m)
    return min(abs(t - 2.0), abs(t + 2.0)) < eps


def classify(m, eps=1e-9):
    t = trace(m)
    if abs(m[1]) < eps and abs(m[2]) < eps and abs(m[0] - m[3]) < eps:
        return IDENTITY
    if min(abs(t - 2.0), abs(t + 2.0)) < eps:
        return PARABOLIC
    if abs(t.imag) < eps and abs(t.real) < 2.0:
        return ELLIPTIC
    return LOXODROMIC


def fixed_points(m):
    """The one or two fixed points of a Moebius map."""
    a, b, c, d = m
    if abs(c) < 1e-14:
        if abs(a - d) < 1e-14:
            return ()
        return (b / (d - a),)
    disc = cmath.sqrt((a - d) * (a - d) + 4.0 * b * c)
    return ((a - d + disc) / (2.0 * c), (a - d - disc) / (2.0 * c))


def attracting_fixed_point(m):
    """The fixed point orbits converge to (either one, if parabolic)."""
    fp = fixed_points(m)
    if not fp:
        return complex(0.0, 0.0)
    if len(fp) == 1:
        return fp[0]
    a, b, c, d = m
    best = fp[0]
    best_scale = None
    for z in fp:
        den = c * z + d
        if abs(den) < 1e-300:
            continue
        scale = abs(1.0 / (den * den))          # |derivative| at the fixed point
        if best_scale is None or scale < best_scale:
            best_scale, best = scale, z
    return best


def _selftest():
    import math

    # 1. group laws
    m = normalise(mat(2, 1, 1, 1))
    assert abs(det(m) - 1.0) < 1e-12
    assert all(abs(x - y) < 1e-12 for x, y in zip(mul(m, inv(m)), IDENT))
    n = normalise(mat(0, -1, 1, 0))
    assert all(abs(x - y) < 1e-12
               for x, y in zip(mul(mul(m, n), inv(n)), m))

    # 2. classification, including the trap the header warns about
    assert classify(mat(1, 1, 0, 1)) == PARABOLIC
    assert classify(mat(-1, 1, 0, -1)) == PARABOLIC
    rot = mat(cmath.exp(0.3j), 0, 0, cmath.exp(-0.3j))
    assert classify(rot) == ELLIPTIC
    assert classify(mat(2, 0, 0, 0.5)) == LOXODROMIC
    trap = mat(-1j, 0, 0, -1j)                  # trace -2i: |tr| = 2 but NOT parabolic
    assert abs(abs(trace(trap)) - 2.0) < 1e-12
    assert not is_parabolic(trap), "|tr| = 2 must not be read as parabolic"

    # 3. fixed points really are fixed
    for m2 in (normalise(mat(2, 1, 1, 1)), normalise(mat(3, 1, 2, 1))):
        for z in fixed_points(m2):
            assert abs(apply(m2, z) - z) < 1e-9, (m2, z)

    # 4. the attracting fixed point attracts.  Use a map with BOTH fixed points
    #    finite: for a diagonal map the attractor is infinity, which this
    #    representation cannot express.
    m3 = normalise(mat(3, 1, 1, 1))
    assert classify(m3) == LOXODROMIC
    assert len(fixed_points(m3)) == 2
    p = attracting_fixed_point(m3)
    z = complex(0.7, 0.3)
    for _ in range(200):
        z = apply(m3, z)
    assert abs(z - p) < 1e-6, (z, p)

    # 5. words compose in order
    gens = [normalise(mat(1, 1, 0, 1)), normalise(mat(1, 0, 1, 1))]
    w = word(gens, [0, 1, 0])
    assert all(abs(x - y) < 1e-12
               for x, y in zip(w, mul(mul(gens[0], gens[1]), gens[0])))

    print("kleinian.mobius: group laws hold, classification separates "
          "parabolic from |tr|=2, fixed points fixed and attracting. "
          "RESULT: OK")
