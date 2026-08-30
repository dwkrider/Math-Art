"""Weierstrass data: storing it, and checking it against the shipped code.

WHY MEASURING H IS THE WRONG CHECK HERE.  The Weierstrass-Enneper
representation produces a minimal surface from ANY holomorphic pair
(g, dh):

    X(z) = Re integral ( (1/2)(1/g - g), (i/2)(1/g + g), 1 ) dh

Minimality is a theorem about the construction, not a property of the
particular data.  So integrating a record's g and dh and finding H = 0
would test the integrator and say nothing whatever about whether the
record holds the RIGHT g and dh.  Every wrong transcription would pass.

WHAT IS WORTH CHECKING, therefore, is the data itself -- against the
shipped implementation, exactly as the implicit block is checked against
its own.  `math_art/minsurf/zoo.py` carries `g` and `dh` as callables of
(z, p) on 16 rows, so a stored expression can be sampled against them
over the complex plane and must agree.

A second, independent check comes for free once g is stored: the degree
of the Gauss map is the winding number of g around the domain boundary,
and for a complete minimal surface of finite total curvature the total
curvature is -4*pi times that degree.  Where a record states its total
curvature, the two must agree -- and that is a genuine cross-check
between two facts stored for different reasons.
"""

import cmath
import math

from . import expr


def sample_ring(n, radius, phase=0.017):
    """`n` points on a circle of the given radius, off the real axis.

    The phase offset keeps samples away from the positive real axis,
    where a branch cut of a fractional power would otherwise be
    evaluated exactly ON the cut -- a trap already recorded in this
    repo's minimal-surface notes, where numpy's principal branch picks
    the wrong side.
    """
    return [radius * cmath.exp(1j * (2.0 * math.pi * k / n + phase))
            for k in range(n)]


def compare(text, oracle, params=None, radii=(0.45, 0.7, 0.95),
            n=48, allow_scalar=False, rel_tol=1e-8):
    """Does the stored expression reproduce the shipped callable?

    `oracle` takes a complex z and returns a complex value.  With
    `allow_scalar`, agreement up to one constant multiple is accepted --
    right for dh, whose scaling merely scales the surface, and wrong for
    g, which IS the Gauss map and is not free to be rescaled.
    """
    mine, theirs = [], []
    for r in radii:
        for z in sample_ring(n, r):
            try:
                a = expr.evaluate_complex(text, dict(params or {}, z=z))
            except expr.ExprError as exc:
                return False, "expression does not evaluate: %s" % exc
            try:
                b = complex(oracle(z))
            except Exception as exc:                  # noqa: BLE001
                return False, "oracle failed at z=%s: %s" % (z, exc)
            if not (cmath.isfinite(a) and cmath.isfinite(b)):
                continue
            mine.append(a)
            theirs.append(b)

    if len(mine) < 12:
        return False, "too few finite samples (%d)" % len(mine)

    scale = 1.0 + 0j
    if allow_scalar:
        i0 = max(range(len(theirs)), key=lambda k: abs(theirs[k]))
        if abs(theirs[i0]) < 1e-12:
            return False, "oracle is ~0 everywhere sampled"
        scale = mine[i0] / theirs[i0]

    mag = max(abs(b * scale) for b in theirs) or 1.0
    worst = max(abs(a - scale * b) for a, b in zip(mine, theirs))
    if worst <= rel_tol * mag:
        return True, ("matches over %d complex samples (worst %.3g%s)"
                      % (len(mine), worst,
                         ", scale %.4g" % abs(scale) if allow_scalar else ""))
    return False, ("disagrees: worst residual %.3g over %d samples (scale %.3g)"
                   % (worst, len(mine), mag))


def gauss_degree(text, params=None, radius=0.9, n=720):
    """Degree of the Gauss map: the winding number of g about the origin.

    Counted by accumulating the argument of g along a circle, which is
    the standard argument-principle count of zeros minus poles inside.
    Returns None when the accumulation does not land near an integer --
    which happens when a zero or pole sits ON the contour, and is
    reported rather than rounded away.
    """
    prev = None
    total = 0.0
    for z in sample_ring(n, radius) + [sample_ring(n, radius)[0]]:
        try:
            w = expr.evaluate_complex(text, dict(params or {}, z=z))
        except expr.ExprError:
            return None
        if w == 0 or not cmath.isfinite(w):
            return None
        a = cmath.phase(w)
        if prev is not None:
            d = a - prev
            while d > math.pi:
                d -= 2.0 * math.pi
            while d < -math.pi:
                d += 2.0 * math.pi
            total += d
        prev = a
    deg = total / (2.0 * math.pi)
    if abs(deg - round(deg)) > 0.02:
        return None
    return int(round(deg))


def total_curvature_from_degree(deg):
    """-4*pi*deg, the total curvature of a complete minimal surface of
    finite total curvature whose Gauss map has this degree."""
    return None if deg is None else -4.0 * math.pi * abs(deg)


def _selftest():
    """Complex oracle and winding number; raises on failure."""
    # the oracle must ACCEPT a correct transcription...
    ok, detail = compare("z**k", lambda z: z ** 3, {"k": 3})
    assert ok, detail

    # ...and REJECT a wrong exponent, which is the silent failure mode
    ok, _ = compare("z**k", lambda z: z ** 2, {"k": 3})
    assert not ok, "a wrong exponent must not pass"

    # dh may differ by a constant multiple (it scales the surface);
    # g may NOT (it is the Gauss map)
    ok, _ = compare("2.0*z**2", lambda z: z ** 2, allow_scalar=True)
    assert ok, "dh must be allowed a scalar"
    ok, _ = compare("2.0*z**2", lambda z: z ** 2, allow_scalar=False)
    assert not ok, "g must NOT be allowed a scalar -- rescaling the Gauss " \
                   "map is a different surface, not the same one"

    # winding numbers
    assert gauss_degree("z") == 1
    assert gauss_degree("z**3") == 3
    assert gauss_degree("1/z**2") == -2
    assert gauss_degree("z**2 + 0.001") == 2

    # Enneper: g = z, so degree 1 and total curvature -4*pi -- the value
    # the record independently carries as a classical closed form
    tc = total_curvature_from_degree(gauss_degree("z"))
    assert abs(tc + 4 * math.pi) < 1e-9, tc

    # a constant g has no winding, and that must read as 0, not as None
    assert gauss_degree("1") == 0

    print("RESULT: OK  (surfdb.weierstrass)")
