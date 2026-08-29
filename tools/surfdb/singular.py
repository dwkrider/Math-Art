"""Counting the singular points of an implicit surface.

WHY THIS IS THE CHECK THAT MATTERS for the record-nodal family. Those
surfaces are interesting for exactly one reason: they carry the largest
known number of ordinary double points for their degree. A mistyped
coefficient does not error and does not look wrong -- it produces a
smooth, plausible surface with a DIFFERENT node count, destroying the
only property anyone cared about. Nothing else in this toolchain would
notice.

A singular point of F = 0 satisfies

    F = 0   and   grad F = 0

simultaneously -- four equations in three unknowns, so the system is
overdetermined and a generic surface has no solutions at all. That
overdetermination is what makes the count meaningful.

Solved by minimising

    G = F^2 + |grad F|^2

from many random seeds. G >= 0 with equality exactly at a singular
point, so any converged zero is one. Distinct minima are deduped on
position, and only points inside the clip region are counted -- Barth's
sextic has 65 nodes of which 15 lie at infinity, so a finite count is
expected to fall short of the published total and the shortfall is
reported rather than hidden.

This is a NUMERICAL count. It can miss a node whose basin no seed
reaches, so a count BELOW the published one is weak evidence, while a
count ABOVE it means the published figure or the transcription is wrong.
The asymmetry is stated in the result rather than glossed.
"""

import math
import random

from . import expr


def _fn(poly, params):
    env = dict(params or {})

    def f(x, y, z):
        env["x"], env["y"], env["z"] = x, y, z
        return expr.evaluate(poly, env)
    return f


def _grad(f, p, h=1e-5):
    x, y, z = p
    return ((f(x + h, y, z) - f(x - h, y, z)) / (2 * h),
            (f(x, y + h, z) - f(x, y - h, z)) / (2 * h),
            (f(x, y, z + h) - f(x, y, z - h)) / (2 * h))


def _G(f, p):
    """F^2 + |grad F|^2 -- zero exactly at a singular point."""
    g = _grad(f, p)
    v = f(*p)
    return v * v + g[0] ** 2 + g[1] ** 2 + g[2] ** 2


def _descend(f, p, steps=400, h=1e-4):
    """Gradient descent on G, with a shrinking step."""
    cur = list(p)
    val = _G(f, cur)
    step = 0.05
    for _ in range(steps):
        g = []
        for i in range(3):
            q = list(cur)
            q[i] += h
            a = _G(f, q)
            q[i] -= 2 * h
            b = _G(f, q)
            g.append((a - b) / (2 * h))
        n = math.sqrt(sum(t * t for t in g))
        if n < 1e-18 or not math.isfinite(n):
            break
        moved = False
        for _try in range(12):
            cand = [cur[i] - step * g[i] / n for i in range(3)]
            try:
                cv = _G(f, cand)
            except (expr.ExprError, ValueError, OverflowError,
                    ZeroDivisionError):
                step *= 0.5
                continue
            if math.isfinite(cv) and cv < val:
                cur, val = cand, cv
                moved = True
                break
            step *= 0.5
        if not moved or step < 1e-12:
            break
    return tuple(cur), val


def count(poly, params=None, extent=1.6, seeds=900, tol=1e-7,
          merge=1e-3, seed=20260828):
    """Count distinct singular points inside a ball of radius `extent`.

    Returns {"count", "points", "seeds", "note"}.
    """
    rng = random.Random(seed)
    try:
        f = _fn(poly, params)
    except expr.ExprError as exc:
        return {"count": None, "points": [], "seeds": 0,
                "note": "expression does not evaluate: %s" % exc}

    found = []
    for _ in range(seeds):
        p0 = (rng.uniform(-extent, extent), rng.uniform(-extent, extent),
              rng.uniform(-extent, extent))
        try:
            p, v = _descend(f, p0)
        except (expr.ExprError, ValueError, OverflowError, ZeroDivisionError):
            continue
        if not math.isfinite(v) or v > tol:
            continue
        if max(abs(t) for t in p) > extent:
            continue
        if any(math.dist(p, q) < merge for q in found):
            continue
        found.append(p)

    return {
        "count": len(found),
        "points": found,
        "seeds": seeds,
        "note": ("numerical: a count BELOW the published one may mean a "
                 "node whose basin no seed reached, or nodes at infinity; "
                 "a count ABOVE it means the published figure or the "
                 "transcription is wrong"),
    }


def check(poly, published, params=None, extent=1.6, **kw):
    """Compare a numerical count against a published one.

    Returns (verdict, detail) with verdict True / False / None, where
    None means 'consistent but not conclusive' -- the usual outcome when
    some of the singular points lie at infinity.
    """
    res = count(poly, params, extent=extent, **kw)
    got = res["count"]
    if got is None:
        return None, res["note"]
    if got > published:
        return False, ("found %d singular points inside the clip but only "
                       "%d are published -- the transcription or the "
                       "published figure is wrong" % (got, published))
    if got == published:
        return True, "found all %d published singular points" % got
    return None, ("found %d of %d published singular points inside the clip; "
                  "the rest may lie at infinity or outside it, so this is "
                  "consistent but not conclusive" % (got, published))


def _selftest():
    """Counts on surfaces whose singularities are known exactly.

    SEED COUNTS ARE DELIBERATELY LOW HERE.  Each descent is a few
    thousand expression evaluations, so this test is minutes long, not
    seconds, and a self-test nobody can afford to run is a self-test
    that stops being run.  350 seeds is the smallest budget at which
    Cayley's four nodes are still found every time and the mistyped
    variant still shows a different count -- which is the discrimination
    the test exists for.  The VALIDATOR is where the seed count should
    be raised (--node-seeds), because there it is bounded by the handful
    of records that publish a count.
    """
    # a smooth sphere has NO singular points
    r = count("x**2 + y**2 + z**2 - 1", seeds=300)
    assert r["count"] == 0, r

    # a cone has exactly one: its apex at the origin
    r = count("x**2 + y**2 - z**2", seeds=400, extent=1.2)
    assert r["count"] == 1, r
    assert max(abs(t) for t in r["points"][0]) < 1e-2, r["points"]

    # two crossing planes meet in a LINE of singular points, so the count
    # is not meaningful there -- it must not claim a small number
    r = count("x*y", seeds=300, extent=1.0)
    assert r["count"] > 3, \
        "a singular curve must not be reported as a few isolated points"

    # Cayley's nodal cubic: FOUR nodes, the maximum for a cubic
    cayley = "4*(x**2 + y**2 + z**2) + 16*x*y*z - 1"
    r = count(cayley, seeds=350, extent=1.1)
    assert r["count"] == 4, "Cayley must show 4 nodes, got %d" % r["count"]
    ok, detail = check(cayley, 4, extent=1.1, seeds=350)
    assert ok is True, detail

    # and a MISTYPED Cayley must not still show four -- this is the whole
    # point of the check
    wrong = "4*(x**2 + y**2 + z**2) + 15*x*y*z - 1"
    r2 = count(wrong, seeds=350, extent=1.1)
    assert r2["count"] != 4, \
        "a wrong coefficient must change the node count, got %d" % r2["count"]

    # over-counting must be reported as a failure, not a pass
    ok, detail = check(cayley, 2, extent=1.1, seeds=350)
    assert ok is False and "wrong" in detail, detail

    # under-counting is inconclusive, not a failure: nodes at infinity
    ok, detail = check(cayley, 65, extent=1.1, seeds=200)
    assert ok is None and "not conclusive" in detail, detail

    print("RESULT: OK  (surfdb.singular)")
