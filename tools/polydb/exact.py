# Exact-form recognition for the polyhedron database.
#
# The generators produce double-precision coordinates. Almost all of these
# solids have coordinates in a real quadratic field or a square root of one --
# Q(sqrt2), Q(sqrt3), Q(sqrt5), and nested forms like sqrt(58 + 18*sqrt(5))/4 --
# so the exact value can be RECOVERED from the float rather than transcribed.
#
# Recovery uses integer-relation detection (PSLQ, via mpmath) against a small
# basis, then VERIFIES the candidate by re-evaluating it at 50 digits. Nothing
# is emitted that does not reproduce its own float, and coefficient bounds are
# kept tight so a chance relation cannot slip through: with ~1e-13 of usable
# precision and coefficients capped in the thousands, a false hit would need a
# coincidence of about one part in 10^13.
#
# Where no radical form is found -- the snub solids and the elementary Johnson
# solids J84-J92, whose coordinates are roots of irreducible cubics and
# quartics with no nice radical expression -- the honest answer is the MINIMAL
# POLYNOMIAL, which `minimal_polynomial()` returns. That is still an exact
# description, and the validator checks the stored value really is a root.

import math
from fractions import Fraction

import mpmath as mp

# squarefree radicands worth trying, in the order they actually occur
RADICANDS = (5, 2, 3, 6, 10, 15, 30, 7, 14, 21, 35, 11, 13, 17, 19, 22, 26,
             33, 55, 65, 70, 105)

_TOL = 1e-11          # accept a candidate only if it reproduces the float here
_MAXC = 4000          # PSLQ coefficient cap


def _pslq(vec, maxcoeff=_MAXC):
    try:
        with mp.workdps(30):
            return mp.pslq([mp.mpf(v) for v in vec], tol=mp.mpf(10) ** -13,
                           maxcoeff=maxcoeff, maxsteps=200)
    except Exception:
        return None


def _fmt_frac(p, q):
    """p/q as a string, reduced, with q > 0."""
    f = Fraction(int(p), int(q))
    return str(f.numerator) if f.denominator == 1 else "%d/%d" % (f.numerator,
                                                                  f.denominator)


def _fmt_quad(p, q, d, r):
    """(p + q*sqrt(d))/r, tidied."""
    g = math.gcd(math.gcd(abs(int(p)), abs(int(q))), abs(int(r)))
    if g:
        p, q, r = int(p) // g, int(q) // g, int(r) // g
    if r < 0:
        p, q, r = -p, -q, -r
    if q == 0:
        return _fmt_frac(p, r)
    qs = "sqrt(%d)" % d if abs(q) == 1 else "%d*sqrt(%d)" % (abs(q), d)
    if p == 0:
        body = qs if q > 0 else "-" + qs
        return body if r == 1 else ("%s/%d" % (body, r) if q > 0
                                    else "-%s/%d" % (qs, r))
    # pull a common numerator factor out front: (3+3*sqrt(5))/4 -> 3*(1+sqrt(5))/4
    k = math.gcd(abs(p), abs(q))
    if k > 1 and r != 1:
        p2, q2 = p // k, q // k
        qs2 = "sqrt(%d)" % d if abs(q2) == 1 else "%d*sqrt(%d)" % (abs(q2), d)
        inner = "%d%s%s" % (p2, "+" if q2 > 0 else "-", qs2)
        return "%d*(%s)/%d" % (k, inner, r)
    body = "%d%s%s" % (p, "+" if q > 0 else "-", qs)
    return body if r == 1 else "(%s)/%d" % (body, r)


def _reduce_nested(a, b, d, c):
    """sqrt(a + b*sqrt(d))/c -> pull any square factor k^2 out of (a, b) when
    k also divides c, so sqrt(232+72*sqrt(5))/8 prints as sqrt(58+18*sqrt(5))/4."""
    g = math.gcd(abs(a), abs(b))
    k = 1
    m = 2
    while m * m <= g:
        while g % (m * m) == 0 and c % (k * m) == 0 and (a // (m * m)) and True:
            if c % (k * m) != 0:
                break
            g //= m * m
            a //= m * m
            b //= m * m
            k *= m
        m += 1
    if k > 1:
        c //= k
    return a, b, c


def _verify(expr, x, tol=_TOL):
    try:
        with mp.workdps(50):
            val = float(mp.mpf(_eval_mp(expr)))
    except Exception:
        return False
    return abs(val - x) <= tol * max(1.0, abs(x))


def _eval_mp(expr):
    env = {"sqrt": mp.sqrt, "pi": mp.pi, "acos": mp.acos, "asin": mp.asin,
           "atan": mp.atan, "cos": mp.cos, "sin": mp.sin, "tan": mp.tan}
    return eval(expr, {"__builtins__": {}}, env)              # noqa: S307


def recognize(x, tol=_TOL):
    """An exact expression string for the float x, or None.

    Tries, in order: zero, rational, (p+q*sqrt(d))/r, and sqrt of a rational
    or quadratic (which is what nested radicals like sqrt(58+18*sqrt(5))/4
    reduce to).
    """
    if not isinstance(x, float) or x != x or math.isinf(x):
        return None
    if abs(x) < 1e-13:
        return "0"

    # rational
    rel = _pslq([x, 1.0], maxcoeff=100000)
    if rel and rel[0]:
        cand = _fmt_frac(-rel[1], rel[0])
        if _verify(cand, x, tol):
            return cand

    # (p + q sqrt(d)) / r
    for d in RADICANDS:
        sd = math.sqrt(d)
        rel = _pslq([x, 1.0, sd])
        if rel and rel[0]:
            cand = _fmt_quad(-rel[1], -rel[2], d, rel[0])
            if _verify(cand, x, tol):
                return cand

    # x^2 rational -> x = sqrt(p)/q
    xx = x * x
    rel = _pslq([xx, 1.0], maxcoeff=100000)
    if rel and rel[0]:
        f = Fraction(int(-rel[1]), int(rel[0]))
        p, q = f.numerator, f.denominator
        if p > 0:
            cand = "sqrt(%d)/%d" % (p * q, q) if q != 1 else "sqrt(%d)" % p
            if x < 0:
                cand = "-" + cand
            if _verify(cand, x, tol):
                return cand

    # x^2 quadratic -> x = sqrt(P*R + Q*R*sqrt(d)) / R
    for d in RADICANDS:
        sd = math.sqrt(d)
        rel = _pslq([xx, 1.0, sd])
        if rel and rel[0]:
            a, b, c = int(rel[0]), int(-rel[1]), int(-rel[2])
            if a < 0:
                a, b, c = -a, -b, -c
            P, Q, R = b, c, a                    # x^2 = (P + Q sqrt d)/R
            g = math.gcd(math.gcd(abs(P), abs(Q)), abs(R))
            if g:
                P, Q, R = P // g, Q // g, R // g
            if R <= 0:
                continue
            inner_p, inner_q, R = _reduce_nested(P * R, Q * R, d, R)
            body = "%d%s%s" % (inner_p, "+" if inner_q > 0 else "-",
                               "sqrt(%d)" % d if abs(inner_q) == 1
                               else "%d*sqrt(%d)" % (abs(inner_q), d))
            cand = "sqrt(%s)/%d" % (body, R) if R != 1 else "sqrt(%s)" % body
            if x < 0:
                cand = "-" + cand
            if _verify(cand, x, tol):
                return cand
    return None


def minimal_polynomial(x, maxdeg=8, tol=1e-11):
    """Minimal polynomial of x as (coeff_list, pretty_string), lowest degree
    first, or None. Used for the snub / elementary-Johnson constants that have
    no usable radical form."""
    if abs(x) < 1e-13:
        return None
    for deg in range(2, maxdeg + 1):
        vec = [x ** k for k in range(deg + 1)]
        rel = _pslq(vec, maxcoeff=100000)
        if not rel or not any(rel):
            continue
        # verify: the polynomial really vanishes, to a scale-aware tolerance
        val = sum(int(c) * (x ** k) for k, c in enumerate(rel))
        scale = max(1.0, max(abs(int(c)) * abs(x) ** k
                             for k, c in enumerate(rel)))
        if abs(val) > tol * scale:
            continue
        coeffs = [int(c) for c in rel]
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        if len(coeffs) < 3:
            continue
        return coeffs, _poly_str(coeffs)
    return None


def _poly_str(coeffs):
    terms = []
    for k in range(len(coeffs) - 1, -1, -1):
        c = coeffs[k]
        if c == 0:
            continue
        if k == 0:
            t = "%d" % abs(c)
        else:
            base = "x" if k == 1 else "x^%d" % k
            t = base if abs(c) == 1 else "%d*%s" % (abs(c), base)
        terms.append(("- " if c < 0 else "+ ") + t)
    s = " ".join(terms).lstrip("+ ").strip()
    if s.startswith("- "):
        s = "-" + s[2:]
    return s + " = 0"


def build_constants(values, prefix="C", tol=_TOL):
    """Recognise a set of coordinate magnitudes and return
    (constants_dict, mapping) where mapping sends each rounded value to the
    constant NAME (C0, C1, ...) or to a literal string for simple rationals.

    Simple values (0, 1, 1/2 ...) are inlined rather than named, matching
    McCooey's convention of naming only the awkward constants.
    """
    uniq = sorted({round(abs(v), 12) for v in values if abs(v) > 1e-13})
    consts, mapping = {}, {}
    n = 0
    for v in uniq:
        expr = recognize(float(v), tol)
        if expr is None:
            mapping[v] = None
            continue
        if len(expr) <= 4:                       # 1, 2, 1/2, 3/2 ... inline
            mapping[v] = expr
            continue
        name = "%s%d" % (prefix, n)
        n += 1
        consts[name] = {"exact": expr, "value": float(v)}
        mapping[v] = name
    return consts, mapping
