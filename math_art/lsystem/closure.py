
# When does a turtle path close, and when are two paths the same figure?
#
# Every mode of `turtle_curve_generator` needs to answer one of these.
# Spirolaterals need to know how many blocks to draw before the figure
# repeats; the teragon and word modes need to know whether the result is
# a closed island (fillable) or an open arc; and the spirolateral
# reversal enumeration needs a way to tell 2^n variants apart without
# shipping thousands of near-duplicates.
#
# Abelson and diSessa answer all three with total turning.
#
# THE CLOSED-PATH THEOREM.  Total turning is a topological invariant of a
# closed path, and for any closed path it is an integer multiple of 360
# degrees.  That integer is the turning number.  It is the termination
# rule for the whole POLY family: accumulate turning, stop at a multiple
# of 360.
#
# THE MINIMUM-TURN CONVENTION.  Total turning is only well defined once
# each vertex is assigned the MINIMUM turn that aligns the turtle for the
# next segment -- a value in (-180, 180].  Without it, `LEFT 90` and
# `RIGHT 270` draw the same picture but accumulate different totals, and
# the closure test becomes meaningless.  This is why `minimum_turn` is
# applied everywhere rather than raw turn values being summed.
#
# WHITNEY-GRAUSTEIN.  Two closed plane paths can be deformed into one
# another exactly when they have the same total turning.  That makes the
# turning number a computable invariant for de-duplicating a family of
# generated figures -- the honest answer to "are these two spirolaterals
# the same figure?".
#
# SCISSORS.  A crossing where two segments pass through each other flips
# a vertex from LEFT 180 to RIGHT 180 and so changes total turning by
# exactly 360 degrees.  Not a deformation; a predictable jump.  Detected
# rather than silently absorbed, because it is the one way two figures
# that look related can have different invariants.
#
# References:
# - Harold Abelson and Andrea diSessa, "Turtle Geometry: The Computer as
#   a Medium for Exploring Mathematics", MIT Press, 1981 -- ch. 1-2 (POLY
#   and total turning) and ch. 4 (the Closed-Path Theorem, the
#   minimum-turn convention, Whitney-Graustein, scissors).
# - Hassler Whitney, "On regular closed curves in the plane", Compositio
#   Mathematica 4, 1937, pp. 276-284.
# - Werner Graustein, unpublished; the theorem is universally cited as
#   Whitney-Graustein.

import numpy as np

#: Angles closer than this to a multiple of 360 count as closed.  A
#: teragon at iteration 6 accumulates thousands of turns, so the
#: tolerance has to absorb accumulated floating-point error without
#: being loose enough to call an open path closed.
CLOSE_TOL = 1e-6


def minimum_turn(deg):
    """Reduce a turn to the equivalent value in (-180, 180].

    This is the convention the Closed-Path Theorem is stated in.  Note
    the half-open interval: exactly +180 stays +180 rather than becoming
    -180, so a straight reversal has a definite sign.
    """
    a = np.asarray(deg, dtype=float)
    out = -((-a + 180.0) % 360.0 - 180.0)
    # `(-a + 180) % 360 - 180` maps 180 to -180; put it back.
    out = np.where(np.isclose(out, -180.0), 180.0, out)
    return float(out) if np.isscalar(deg) or out.ndim == 0 else out


def total_turning(turns):
    """Sum of the minimum turns, in degrees."""
    if len(turns) == 0:
        return 0.0
    return float(np.sum(minimum_turn(np.asarray(turns, dtype=float))))


def turning_number(turns):
    """Total turning divided by 360 -- an integer for a closed path.

    Returned as a float so an open path is still describable; use
    `closes` to decide whether it is meaningful.
    """
    return total_turning(turns) / 360.0


def closes(turns):
    """Closed-Path Theorem test: is total turning a multiple of 360?

    NOTE this is necessary, not sufficient -- it says the turtle ends up
    facing its original heading, which every closed path does, but an
    open path can too (a `Z` shape returns to its heading without
    returning to its start).  `closes_geometrically` is the full test;
    this one is the cheap invariant, and the one the theorem is about.
    """
    t = total_turning(turns)
    return abs(t / 360.0 - round(t / 360.0)) < CLOSE_TOL


def closes_geometrically(points, tol=1e-6):
    """Does the path actually return to its starting point?"""
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return False
    return bool(np.linalg.norm(p[-1] - p[0]) < tol * max(
        1.0, float(np.abs(p).max())))


def turns_of(points):
    """Recover the minimum turn at each interior vertex of a polyline.

    Degenerate (zero-length) segments are dropped rather than producing
    a NaN turn, so a path that pauses still yields a usable sequence.
    """
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return np.zeros(0)
    d = np.diff(p, axis=0)
    L = np.linalg.norm(d, axis=1)
    d = d[L > 1e-12]
    if len(d) < 2:
        return np.zeros(0)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return minimum_turn(np.diff(ang))


def blocks_to_close(n, q):
    """Spirolateral closure: how many blocks of `n` segments are needed.

    For a turn of 360/q degrees repeated through blocks of `n` segments,
    the figure closes after `q / gcd(n, q)` blocks.  Each block advances
    the heading by `n * 360 / q`, so the heading returns to its start
    after the least `k` with `k * n` divisible by `q`, which is
    `q / gcd(n, q)`.
    """
    from math import gcd
    n, q = int(n), int(q)
    if n <= 0 or q <= 0:
        return 1
    return q // gcd(n, q)


def spirolateral_closes(n, q):
    """Does an order-n spirolateral at 360/q degrees close, or spiral?

    Heading closure is NOT enough here.  After `q / gcd(n, q)` blocks the
    turtle always faces its original direction -- but if `q` divides `n`
    the heading returns after a SINGLE block, so every block applies the
    same net displacement and the successive blocks march off in a
    straight line instead of cancelling.  That is the spiral the name
    refers to: an order-4 spirolateral at 90 degrees never closes, while
    orders 1, 2, 3, 5, 6, 7 all do.

    So: closes exactly when `n` is not a multiple of `q`.
    """
    n, q = int(n), int(q)
    return q > 0 and (n % q) != 0


def scissor_crossings(turns, tol=1e-9):
    """Indices of vertices that are exact reversals.

    These are the `scissors` vertices: passing two segments through one
    another flips such a vertex between +180 and -180 and moves total
    turning by exactly 360, so a figure containing them is not
    deformable into its mirror without that jump.
    """
    t = np.asarray(turns, dtype=float)
    return np.where(np.abs(np.abs(minimum_turn(t)) - 180.0) < tol)[0]


def dedupe_by_turning(figures, key=None):
    """Keep one representative of each Whitney-Graustein class.

    `figures` is any iterable; `key(f)` must give its turn sequence
    (default: the figure IS the turn sequence).  Two closed paths with
    the same total turning are deformable into one another, so shipping
    both is shipping the same figure twice -- which is what makes the
    2**n spirolateral reversal enumeration tractable.

    Returns (representatives, classes) where `classes` maps the turning
    number to every index that produced it.
    """
    key = key or (lambda f: f)
    reps, classes = [], {}
    for i, f in enumerate(figures):
        t = turning_number(key(f))
        k = int(round(t)) if closes(key(f)) else None
        k = k if k is not None else round(t, 6)
        if k not in classes:
            classes[k] = []
            reps.append(f)
        classes[k].append(i)
    return reps, classes


def _selftest():
    # --- the minimum-turn convention ---------------------------------
    # This is the whole reason the convention exists: LEFT 90 and
    # RIGHT 270 draw the same picture, so they must reduce to the same
    # turn or total turning is not well defined.
    assert abs(minimum_turn(90.0) - 90.0) < 1e-12
    assert abs(minimum_turn(-270.0) - 90.0) < 1e-12
    assert abs(minimum_turn(450.0) - 90.0) < 1e-12
    # +180 must stay +180, not flip to -180: the sign of a reversal has
    # to be definite for the scissors argument to mean anything.
    assert abs(minimum_turn(180.0) - 180.0) < 1e-12
    assert abs(minimum_turn(-180.0) - 180.0) < 1e-12
    assert abs(minimum_turn(0.0)) < 1e-12
    v = minimum_turn(np.array([90.0, -270.0, 450.0, 359.0]))
    assert np.allclose(v[:3], 90.0) and abs(v[3] + 1.0) < 1e-12

    # --- the Closed-Path Theorem -------------------------------------
    # A convex n-gon turns 360 exactly once.
    for n in (3, 4, 5, 6, 12):
        turns = [360.0 / n] * n
        assert closes(turns), n
        assert abs(turning_number(turns) - 1.0) < 1e-9, n

    # A pentagram closes too, but with turning number 2 -- and that
    # difference is precisely Whitney-Graustein: it is NOT deformable
    # into a pentagon, even though both are closed five-segment paths.
    star = [144.0] * 5
    assert closes(star)
    assert abs(turning_number(star) - 2.0) < 1e-9, turning_number(star)
    assert turning_number(star) != turning_number([72.0] * 5)

    # An open path generally does not close ...
    assert not closes([90.0, 90.0, 90.0])
    # ... but heading closure is necessary, not sufficient: a Z returns
    # to its original heading without returning to its start.
    z = [90.0, -90.0]
    assert closes(z), "the Z regains its heading"
    assert not closes_geometrically([[0, 0, 0], [1, 0, 0],
                                     [1, 1, 0], [2, 1, 0]])

    # --- recovering turns from geometry ------------------------------
    square = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    t = turns_of(square)
    assert len(t) == 3 and np.allclose(t, 90.0), t
    assert closes_geometrically(square)
    # a degenerate repeated point must not produce a NaN turn
    dup = np.array([[0., 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    assert np.all(np.isfinite(turns_of(dup)))

    # --- spirolateral closure ----------------------------------------
    # q/gcd(n,q) blocks.  n=3 at 90 degrees (q=4) needs 4 blocks;
    # n=4 at 90 needs just 1; n=2 at 90 needs 2.
    assert blocks_to_close(3, 4) == 4
    assert blocks_to_close(4, 4) == 1
    assert blocks_to_close(2, 4) == 2
    assert blocks_to_close(5, 6) == 6
    assert blocks_to_close(6, 4) == 2

    # and the prediction must match the actual heading arithmetic
    for n in range(1, 9):
        for q in (3, 4, 5, 6, 8, 12):
            k = blocks_to_close(n, q)
            assert abs(((k * n * (360.0 / q)) % 360.0)) < 1e-9, (n, q, k)

    # Heading closure is not geometric closure: when q divides n the
    # heading resets every single block, so the blocks translate instead
    # of cancelling and the figure spirals.
    assert not spirolateral_closes(4, 4)
    assert not spirolateral_closes(8, 4)
    assert all(spirolateral_closes(n, 4) for n in (1, 2, 3, 5, 6, 7))
    assert not spirolateral_closes(6, 3) and spirolateral_closes(4, 3)

    # --- scissors ----------------------------------------------------
    assert list(scissor_crossings([90.0, 180.0, -90.0])) == [1]
    assert list(scissor_crossings([90.0, -180.0])) == [1]
    assert len(scissor_crossings([90.0, 90.0])) == 0

    # --- Whitney-Graustein de-duplication ----------------------------
    figs = [[90.0] * 4,          # square, turning 1
            [-90.0] * 4,         # square the other way, turning -1
            [72.0] * 5,          # pentagon, turning 1  -> same class
            [144.0] * 5]         # pentagram, turning 2
    reps, classes = dedupe_by_turning(figs)
    assert len(reps) == 3, len(reps)
    assert classes[1] == [0, 2], classes      # square and pentagon agree
    assert classes[-1] == [1] and classes[2] == [3]

    print("closure: OK -- minimum-turn convention, Closed-Path Theorem "
          "(pentagon 1 vs pentagram 2), q/gcd(n,q) closure verified "
          "against heading arithmetic, scissors, Whitney-Graustein dedupe")
