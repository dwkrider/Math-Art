# The boundary of the Maskit slice.
#
# A limit set is a picture of ONE group.  The more interesting picture is the
# deformation space itself: the set of parameters mu for which the group is
# discrete and free.  Its boundary is a fractal curve decorated with CUSPS at
# every rational p/q, where the simple closed curve of slope p/q on the
# punctured torus becomes parabolic.
#
# THE TRACE RECURSION (Box 24).  Building the word w_{p/q} and multiplying it
# out is hopeless -- the trace polynomials reach enormous degree (the book takes
# q past 100,000).  Instead the trace is evaluated AT A POINT by walking the
# Stern-Brocot tree, keeping Farey neighbours p1/q1 and p2/q2 closing in on
# p/q and updating the three traces by the mediant rule
#
#     tr_{uv} <- tr_u tr_{uv} - tr_v      (stepping left)
#     tr_{uv} <- tr_v tr_{uv} - tr_u      (stepping right)
#
# THE SIGN IS A PROPERTY OF THE LIFT, NOT OF THE WORD.  Box 25 solves
# T_{p/q}(mu) - 2 = 0 at EVERY p/q: under the lift fixed by tr a = -i mu and
# tr b = 2, every boundary cusp satisfies tr = +2, at every word length.  A rule
# of "+/-2 by parity of word length" is simply wrong.  Negating a's lift
# multiplies tr w_{p/q} by (-1)^q, and that is the only thing that flips it.
#
# NEWTON NEEDS A DERIVATIVE (Box 27), and finite-differencing a polynomial of
# that degree is hopeless too, so the pair (T, dT/dmu) is carried through the
# recursion together -- the product rule applies term by term.
#
# THE NEXT FAREY FRACTION (Box 28) comes from the continued-fraction expansion
# of p/q, which is the same walk as Box 24 with the left- and right-runs each
# collapsed into a single Farey addition.
#
# Verified fixtures, both the book's own values: mu(0/1) = 2i, and
# mu(3/10) = (1 + sqrt(11) i)/2 ~ 0.5 + 1.658312i.  Note also that membership
# requires Im(mu) > 1, which bounds any search.
#
# References:
# - David Mumford, Caroline Series and David Wright, "Indra's Pearls: The Vision
#   of Felix Klein", Cambridge University Press, 2002 (Box 24, trace recursion;
#   Boxes 25-28, boundary tracing, the Newton solver and the next Farey
#   fraction; Project 9.2 for the Im(mu) > 1 bound).
# - David J. Wright, "Searching for the cusp", in "Spaces of Kleinian Groups",
#   London Mathematical Society Lecture Note Series 329, Cambridge University
#   Press, 2006, pp. 301-336.
# - Bernard Maskit, "Kleinian Groups", Springer Grundlehren 287, 1988.

import cmath
from fractions import Fraction


def _traces(mu):
    """(tr a, tr B, tr aB) for the Maskit lift a(z) = mu + 1/z, b(z) = z + 2."""
    return (-1j * mu, complex(2.0), -1j * (mu - 2.0))


def _dtraces():
    """Derivatives of those traces with respect to mu."""
    return (-1j, complex(0.0), -1j)


def trace_poly(p, q, tr_a, tr_B, tr_aB):
    """tr w_{p/q}, evaluated at a point by the Box 24 mediant recursion."""
    if (p, q) == (0, 1):
        return tr_a
    if (p, q) == (1, 0):
        return tr_B
    p1, q1, p2, q2, p3, q3 = 0, 1, 1, 0, 1, 1
    tr_u, tr_v, tr_uv = tr_a, tr_B, tr_aB
    target = Fraction(p, q)
    guard = 0
    while Fraction(p3, q3) != target:
        guard += 1
        if guard > 100000:
            raise ValueError("trace recursion did not reach %d/%d" % (p, q))
        if target < Fraction(p3, q3):
            p2, q2 = p3, q3
            p3, q3 = p1 + p3, q1 + q3
            tr_u, tr_v, tr_uv = tr_u, tr_uv, tr_u * tr_uv - tr_v
        else:
            p1, q1 = p3, q3
            p3, q3 = p3 + p2, q3 + q2
            tr_u, tr_v, tr_uv = tr_uv, tr_v, tr_v * tr_uv - tr_u
    return tr_uv


def trace_and_rate(p, q, mu):
    """(tr w_{p/q}(mu), d/dmu tr w_{p/q}(mu)).

    The derivative is carried alongside the value through the same recursion --
    the product rule applies term by term -- because finite-differencing a
    polynomial of this degree is numerically hopeless."""
    tr_a, tr_B, tr_aB = _traces(mu)
    da, dB, dab = _dtraces()
    if (p, q) == (0, 1):
        return tr_a, da
    if (p, q) == (1, 0):
        return tr_B, dB
    p1, q1, p2, q2, p3, q3 = 0, 1, 1, 0, 1, 1
    u, v, uv = tr_a, tr_B, tr_aB
    du, dv, duv = da, dB, dab
    target = Fraction(p, q)
    guard = 0
    while Fraction(p3, q3) != target:
        guard += 1
        if guard > 100000:
            raise ValueError("trace recursion did not reach %d/%d" % (p, q))
        if target < Fraction(p3, q3):
            p2, q2 = p3, q3
            p3, q3 = p1 + p3, q1 + q3
            nuv = u * uv - v
            dnuv = du * uv + u * duv - dv
            u, v, uv = u, uv, nuv
            du, dv, duv = du, duv, dnuv
        else:
            p1, q1 = p3, q3
            p3, q3 = p3 + p2, q3 + q2
            nuv = v * uv - u
            dnuv = dv * uv + v * duv - du
            u, v, uv = uv, v, nuv
            du, dv, duv = duv, dv, dnuv
    return uv, duv


def next_farey(p, q, denom):
    """The next fraction after p/q in the Farey sequence of order `denom`.

    Box 28: read off the continued fraction of p/q, collapsing each run of left
    or right steps into one Farey addition."""
    p1, q1 = 0, 1
    p2, q2 = 1, 0
    r, s = p, q
    sign = -1
    guard = 0
    while s != 0 and guard < 10000:
        guard += 1
        a = r // s
        r, s = s, r - a * s
        p1, q1, p2, q2 = p2, q2, a * p2 + p1, a * q2 + q1
        sign = -sign
    k = (denom - sign * q1) // q
    return (k * p + sign * p1, k * q + sign * q1)


def newton_cusp(p, q, mu0, max_iter=60, val_eps=1e-13, prm_eps=1e-14,
                require_in_slice=True):
    """Solve tr w_{p/q}(mu) = 2 by Newton, from the starting guess mu0.

    Always +2 -- see the module header.  Returns None if it does not converge,
    or if it lands on a SPURIOUS ROOT.

    The trace polynomials have very high degree and many roots, only one of
    which is the cusp; a start even moderately far off converges to one of the
    others.  For 3/10, for instance, a start at 0.4 + 1.6i runs to
    0.5 + i*sqrt(3)/2 instead of the true cusp 0.5 + i*sqrt(11)/2.  Groups in
    the slice need Im(mu) > 1 (Indra's Pearls, Project 9.2), so that bound is
    applied as a filter rather than trusting whatever Newton returns.  This is
    why `boundary()` seeds each solve from the previous cusp."""
    mu = complex(mu0)
    root = None
    for _ in range(max_iter):
        val, rate = trace_and_rate(p, q, mu)
        f = val - 2.0
        if abs(rate) < 1e-300:
            return None
        step = f / rate
        nxt = mu - step
        if abs(f) <= val_eps and abs(step) <= prm_eps:
            root = nxt
            break
        mu = nxt
    if root is None:
        val, _ = trace_and_rate(p, q, mu)
        root = mu if abs(val - 2.0) < 1e-8 else None
    if root is None:
        return None
    if require_in_slice and root.imag <= 1.0:
        return None
    return root


def boundary(denom=20, mu0=2j):
    """Trace the Maskit slice boundary: the cusps p/q with q <= denom.

    Box 25: start at the known 0/1 cusp (mu = 2i) and walk the Farey sequence,
    each cusp seeding Newton for the next.  Returns [(p, q, mu), ...]."""
    out = [(0, 1, complex(mu0))]
    p, q = 0, 1
    mu = complex(mu0)
    guard = 0
    while guard < 100000:
        guard += 1
        p, q = next_farey(p, q, denom)
        if q == 0 or Fraction(p, q) > 1:
            break
        root = newton_cusp(p, q, mu)
        if root is None:
            continue
        mu = root
        out.append((p, q, mu))
        if Fraction(p, q) == 1:
            break
    return out


def _selftest():
    import math

    # 1. the book's own fixtures, both at +2 exactly
    tr = trace_poly(0, 1, *_traces(2j))
    assert abs(tr - 2.0) < 1e-12, tr
    mu310 = (1.0 + math.sqrt(11.0) * 1j) / 2.0
    tr2 = trace_poly(3, 10, *_traces(mu310))
    assert abs(tr2 - 2.0) < 1e-9, tr2

    # 2. tr b is identically 2, at every mu
    for mu in (2j, 0.3 + 1.7j, -1.0 + 2.5j):
        assert abs(trace_poly(1, 0, *_traces(mu)) - 2.0) < 1e-12

    # 3. THE SIGN IS NOT SET BY WORD-LENGTH PARITY.  Every cusp found below
    #    satisfies +2, at both odd and even p+q.
    cusps = boundary(denom=8)
    assert len(cusps) > 5, len(cusps)
    for (p, q, mu) in cusps:
        val = trace_poly(p, q, *_traces(mu))
        assert abs(val - 2.0) < 1e-7, \
            "cusp %d/%d should satisfy tr = +2, got %r" % (p, q, val)

    # 4. every cusp respects the Im(mu) > 1 necessary condition
    for (p, q, mu) in cusps:
        assert mu.imag > 1.0 - 1e-9, \
            "cusp %d/%d has Im(mu) = %.6f, outside the slice" % (p, q, mu.imag)

    # 5. the analytic derivative agrees with a finite difference
    for (p, q) in ((1, 2), (2, 5), (3, 10)):
        mu = 0.2 + 1.9j
        _, rate = trace_and_rate(p, q, mu)
        h = 1e-6
        fd = (trace_poly(p, q, *_traces(mu + h))
              - trace_poly(p, q, *_traces(mu - h))) / (2 * h)
        assert abs(rate - fd) < 1e-4 * max(1.0, abs(rate)), (p, q, rate, fd)

    # 6. the Farey walk is strictly increasing and stays in lowest terms
    seq = [(p, q) for (p, q, _) in cusps]
    vals = [Fraction(p, q) for (p, q) in seq]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), seq
    for (p, q) in seq:
        assert math.gcd(p, q) == 1, (p, q)

    # 7. Newton finds the 3/10 cusp from a nearby start, to the book's value
    got = newton_cusp(3, 10, 0.5 + 1.66j)
    assert got is not None and abs(got - mu310) < 1e-9, (got, mu310)

    # 8. and a start that is only moderately off runs to a SPURIOUS root --
    #    0.5 + i*sqrt(3)/2 rather than 0.5 + i*sqrt(11)/2 -- which the
    #    Im(mu) > 1 filter is there to reject.  This is exactly why the
    #    boundary walk seeds each solve from the previous cusp.
    stray = newton_cusp(3, 10, 0.4 + 1.6j, require_in_slice=False)
    assert stray is not None and abs(stray - mu310) > 0.5, stray
    assert abs(stray - (0.5 + math.sqrt(3.0) / 2.0 * 1j)) < 1e-6, stray
    assert newton_cusp(3, 10, 0.4 + 1.6j) is None, \
        "the spurious root must be filtered out"

    print("kleinian.slice: trace recursion matches the book at 0/1 and 3/10, "
          "every cusp satisfies tr = +2 with Im(mu) > 1, analytic derivative "
          "agrees with finite differences, %d cusps traced. RESULT: OK"
          % len(cusps))
