# Turning gallery equation notation into the exact language -- and
# PROVING the result is the same polynomial.
#
# math_art/surfaces/algebraic.py carries `HAUSER_EQUATION`, 63 equations
# in the notation the gallery prints them in:
#
#     "x^2+y^2z^3 = z^4"
#
# That is not the exact language: `^` is not exponentiation, `y^2z^3` has
# an implied multiplication, and there is a right-hand side.  Normalising
# it is a transcription step, and transcription is exactly the operation
# this database is most afraid of -- the contributor notes and the
# record schema both
# say so, because a mistyped coefficient does not error, it silently
# yields a DIFFERENT surface with the wrong node count.
#
# THE ORACLE.  We do not have to trust the conversion, because the
# shipped implementation is right there: `PRESETS[key][1]` is a callable
# that evaluates the same polynomial.  So every converted string is
# checked against it on a few hundred pseudo-random points, and a
# conversion that does not agree is DISCARDED in favour of a null.  The
# database never carries an equation it has not reproduced numerically
# against an independent implementation.
#
# The comparison is up to a nonzero scalar multiple: a level set is
# unchanged by scaling F, and the shipped functions are written in
# whatever normalisation was convenient (some move the RHS across, some
# negate).  So the test is that F_ours = c * F_theirs for one constant c
# across ALL sample points -- which is a much stronger statement than
# agreeing at any one of them.

import math
import random
import re

from . import expr

# Names a defining equation may reference beyond the record's parameters.
COORDS = ("x", "y", "z")


def normalise(text):
    """Gallery notation -> the exact language.

    Handles: `^` -> `**`, implied multiplication between a variable or
    closing paren and a following variable/number/open paren, and moving
    a right-hand side across the `=`.

    Returns a string, which the caller MUST then verify against an
    oracle.  This function is deliberately not trusted on its own.
    """
    s = text.strip()

    # right-hand side across the equals sign
    if "=" in s:
        lhs, rhs = s.split("=", 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        if rhs in ("0", "0.0", ""):
            s = lhs
        else:
            s = "(%s) - (%s)" % (lhs, rhs)

    # fractions written as a/b in coefficient position are already fine

    # implied multiplication.  Insert `*` between:
    #   digit or var or ')'   followed by   var or '('
    #   ')'                   followed by   digit
    # Done before `^` -> `**` so the exponent digits are not caught.
    def insert_mults(t):
        out = []
        prev = ""
        i = 0
        while i < len(t):
            c = t[i]
            if prev and c in "xyz(" and (prev.isdigit() or prev in "xyz)"):
                # do not split a multi-digit number or a function name
                out.append("*")
            out.append(c)
            if not c.isspace():
                prev = c
            i += 1
        return "".join(out)

    s = insert_mults(s)
    s = s.replace("^", "**")
    s = re.sub(r"\s+", "", s)
    return s


def _sample_points(n, extent, rng):
    return [(rng.uniform(-extent, extent),
             rng.uniform(-extent, extent),
             rng.uniform(-extent, extent)) for _ in range(n)]


def verify_against(poly, oracle, params=None, n=240, extent=1.7, seed=20260827,
                   rel_tol=1e-9):
    """Is `poly` the same polynomial as `oracle`, up to a scalar multiple?

    `oracle` is a callable taking numpy arrays (x, y, z) -- the signature
    the shipped `PRESETS[key][1]` functions use -- or a plain scalar
    function.  Returns (ok, detail).

    The test: pick the sample point where |oracle| is largest, take the
    ratio there as the candidate constant c, then require
    |ours - c*theirs| <= rel_tol * scale at EVERY point.  Agreement at
    one point is a coincidence; agreement at 240 with a single shared
    constant is the same polynomial.
    """
    rng = random.Random(seed)
    pts = _sample_points(n, extent, rng)

    try:
        ours = [expr.evaluate(poly, dict(zip(COORDS, p), **(params or {})))
                for p in pts]
    except expr.ExprError as exc:
        return False, "converted expression does not evaluate: %s" % exc

    theirs = []
    try:
        import numpy as np
        for p in pts:
            v = oracle(np.array([p[0]]), np.array([p[1]]), np.array([p[2]]))
            theirs.append(float(np.asarray(v).ravel()[0]))
    except Exception as exc:                      # noqa: BLE001 - oracle is foreign
        return False, "oracle failed: %s" % exc

    finite = [(a, b) for a, b in zip(ours, theirs)
              if math.isfinite(a) and math.isfinite(b)]
    if len(finite) < n // 3:
        return False, "too few finite samples (%d of %d)" % (len(finite), n)

    # candidate scalar from the largest-magnitude oracle sample
    a0, b0 = max(finite, key=lambda ab: abs(ab[1]))
    if abs(b0) < 1e-12:
        return False, "oracle is ~0 everywhere sampled; cannot fix the scale"
    c = a0 / b0
    if abs(c) < 1e-12:
        return False, "converted polynomial is ~0 where the oracle is not"

    scale = max(abs(b * c) for _, b in finite)
    worst = 0.0
    for a, b in finite:
        worst = max(worst, abs(a - c * b))
    if worst <= rel_tol * max(scale, 1.0):
        return True, "matches oracle x %.6g over %d points (worst %.3g)" % (
            c, len(finite), worst)
    return False, ("disagrees with oracle: worst residual %.3g over %d points "
                   "(scale %.3g, c = %.6g)" % (worst, len(finite), scale, c))


def convert_verified(text, oracle, params=None, **kw):
    """normalise() then verify_against(). Returns (poly_or_None, detail).

    A None means the record stores `polynomial: null` with the detail as
    its note -- honest, and strictly more useful than a guess.
    """
    try:
        poly = normalise(text)
    except Exception as exc:                       # noqa: BLE001
        return None, "normalisation failed: %s" % exc
    ok, detail = verify_against(poly, oracle, params=params, **kw)
    return (poly if ok else None), detail


def _selftest():
    """Numeric self-test; raises on failure."""
    import numpy as np

    assert normalise("x^2+y^2z^3 = z^4") == "(x**2+y**2*z**3)-(z**4)"
    assert normalise("x^2 + y^2 + z^2 = 1") == "(x**2+y**2+z**2)-(1)"
    assert normalise("xyz = 0") == "x*y*z"
    assert normalise("x^3y + xz^3 + y^3z + z^3 + 7z^2 + 5z=0") == \
        "x**3*y+x*z**3+y**3*z+z**3+7*z**2+5*z"
    # a coefficient must not be broken apart
    assert normalise("100(x^2+y^2)") == "100*(x**2+y**2)"
    # 2 sqrt(2) style coefficients keep their function name intact
    assert "sqrt" in normalise("sqrt(2)*x^2 = y")

    # the oracle test must ACCEPT a correct conversion...
    def truth(x, y, z):
        return x ** 2 + y ** 2 * z ** 3 - z ** 4
    poly, detail = convert_verified("x^2+y^2z^3 = z^4", truth)
    assert poly is not None, detail

    # ...accept it up to a scalar multiple (the shipped functions are
    # written in whatever normalisation was convenient)...
    def truth_negated(x, y, z):
        return -3.0 * (x ** 2 + y ** 2 * z ** 3 - z ** 4)
    poly, detail = convert_verified("x^2+y^2z^3 = z^4", truth_negated)
    assert poly is not None, detail

    # ...and REJECT a wrong one.  This is the whole point: a mistyped
    # exponent is silent otherwise.
    def wrong(x, y, z):
        return x ** 2 + y ** 2 * z ** 2 - z ** 4
    poly, detail = convert_verified("x^2+y^2z^3 = z^4", wrong)
    assert poly is None, "a mistyped exponent must not pass: %s" % detail
    assert "disagrees" in detail

    # a scalar multiple is NOT enough on its own -- agreement has to hold
    # at every point with ONE constant
    def scaled_plus_junk(x, y, z):
        return 2.0 * (x ** 2 + y ** 2 * z ** 3 - z ** 4) + 0.1 * x
    poly, _ = convert_verified("x^2+y^2z^3 = z^4", scaled_plus_junk)
    assert poly is None, "a perturbed oracle must not pass"

    print("RESULT: OK  (surfdb.polynomial)")
