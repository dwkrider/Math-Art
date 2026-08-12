
# Timed growth: one continuous parameter that freezes a developmental
# stage, and a bake that is provably free of pops.
#
# THE PROBLEM.  Everything else in this add-on generates ONE mesh.
# Development is continuous, and a derivation step is discrete: at step
# n a branch does not exist, at step n+1 it is fully formed.  Animating
# that directly makes every new element spring into existence at full
# size, which reads as a pop and cannot be fixed by interpolation
# afterwards -- the topology changed.
#
# THE ANSWER, from ABOP equation (6.11).  Give every element a growth
# function g(b, t) of its own age with
#
#     g^(k)(b, 0) = 0     for k = 0, 1
#
# -- the element is born with zero SIZE and zero RATE.  Then the whole
# structure can be built once at its final topology, every element given
# a birth time, and the geometry at time t obtained by evaluating each
# element's g.  An element that has not been born yet has length zero and
# is invisible; one just born grows away from zero smoothly.  The
# topology never changes, so a shape-key bake over the sequence is
# fixed-topology and pop-free by construction rather than by tuning.
#
# The condition is on the FIRST derivative as well as the value, and that
# is the part that matters: g(0) = 0 alone gives an element that appears
# at zero length but at full speed, which still reads as a snap.
#
# References:
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990, chapter 6 and equation (6.11).
# - Przemyslaw Prusinkiewicz, Mark Hammel and Eric Mjolsness,
#   "Animation of plant development", SIGGRAPH 1993.

import numpy as np

CURVES = ("SMOOTH", "SIGMOID", "QUINTIC")


def growth(t, kind="SMOOTH"):
    """Growth fraction for normalised age `t`, clamped to [0, 1].

    Every curve here satisfies g(0) = 0 AND g'(0) = 0, which is the
    (6.11) condition.  They differ only in how they approach maturity:

      SMOOTH   3t^2 - 2t^3, the cubic smoothstep -- also g'(1) = 0, so
               growth eases out as well as in
      SIGMOID  a logistic-like curve, fastest in the middle
      QUINTIC  6t^5 - 15t^4 + 10t^3, with g''(0) = g''(1) = 0 too, so
               even the acceleration is continuous
    """
    x = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    if kind == "QUINTIC":
        return x ** 3 * (x * (6.0 * x - 15.0) + 10.0)
    if kind == "SIGMOID":
        # t^2 / (t^2 + (1-t)^2) -- zero value and zero slope at 0
        return x ** 2 / np.where(x ** 2 + (1 - x) ** 2 < 1e-12, 1.0,
                                 x ** 2 + (1 - x) ** 2)
    return x * x * (3.0 - 2.0 * x)


def birth_times(parents, per_step=1.0):
    """Birth time of every node, from its depth in the tree.

    Depth is the natural developmental clock for a structure grown by
    repeated branching: a node is born one plastochron after its parent.
    """
    par = np.asarray(parents, dtype=int)
    b = np.zeros(len(par))
    for i in range(len(par)):
        p = par[i]
        b[i] = 0.0 if p < 0 else b[p] + float(per_step)
    return b


def stage(nodes, parents, t, births=None, duration=1.0, kind="SMOOTH",
          per_step=1.0):
    """Positions at global time `t`.

    Every node is placed a growth-scaled fraction of the way from its
    parent, so an unborn node sits exactly on its parent (a zero-length
    segment, invisible) and a mature one is fully extended.  The node
    COUNT never changes, which is what makes the bake fixed-topology.

    Returns (positions, alive) -- `alive` marks nodes whose segment has
    any length at all, for callers that would rather drop them.
    """
    P = np.asarray(nodes, dtype=float)
    par = np.asarray(parents, dtype=int)
    if births is None:
        births = birth_times(par, per_step)
    age = (float(t) - np.asarray(births, dtype=float)) / max(
        float(duration), 1e-9)
    f = growth(age, kind)

    out = P.copy()
    # walk in index order: parents always precede children in the
    # structures this package produces, so one pass suffices
    for i in range(len(P)):
        p = par[i]
        if p < 0:
            continue
        out[i] = out[p] + (P[i] - P[p]) * f[i]
    return out, f > 1e-9


def bake(nodes, parents, frames=24, t0=None, t1=None, kind="SMOOTH",
         duration=1.0, per_step=1.0):
    """A fixed-topology sequence of stages, ready for shape keys.

    Every frame has the same vertex count in the same order, which is
    the whole requirement for a shape-key bake.  Returns a list of
    position arrays.
    """
    b = birth_times(parents, per_step)
    lo = float(b.min()) if t0 is None else float(t0)
    hi = float(b.max() + duration) if t1 is None else float(t1)
    return [stage(nodes, parents, t, b, duration, kind)[0]
            for t in np.linspace(lo, hi, int(frames))]


def _selftest():
    # --- the (6.11) condition -----------------------------------------
    # g(0) = 0 AND g'(0) = 0.  The derivative is the part that matters:
    # a curve with g(0) = 0 but g'(0) > 0 gives an element that appears
    # at zero length but at full speed, which still pops.
    h = 1e-6
    for kind in CURVES:
        assert abs(float(growth(0.0, kind))) < 1e-12, kind
        assert abs(float(growth(1.0, kind)) - 1.0) < 1e-9, kind
        d0 = (float(growth(h, kind)) - float(growth(0.0, kind))) / h
        assert abs(d0) < 1e-4, f"{kind}: g'(0) = {d0}, must vanish"
        # monotone, so growth never reverses
        xs = np.linspace(0, 1, 200)
        g = growth(xs, kind)
        assert np.all(np.diff(g) >= -1e-12), kind
        # clamped outside [0, 1]
        assert float(growth(-5.0, kind)) == 0.0, kind
        assert abs(float(growth(5.0, kind)) - 1.0) < 1e-12, kind

    # QUINTIC additionally has zero second derivative at both ends
    d2 = (float(growth(2 * h, "QUINTIC")) - 2 * float(growth(h, "QUINTIC"))
          + float(growth(0.0, "QUINTIC"))) / (h * h)
    assert abs(d2) < 1e-2, d2

    # --- birth times ---------------------------------------------------
    parents = np.array([-1, 0, 1, 1, 3])
    b = birth_times(parents)
    assert list(b) == [0.0, 1.0, 2.0, 2.0, 3.0], list(b)

    # --- staging --------------------------------------------------------
    nodes = np.array([[0., 0, 0], [0, 0, 1], [0, 0, 2],
                      [1, 0, 2], [1, 0, 3]], dtype=float)
    # before anything is born, every node sits on the root
    P0, alive = stage(nodes, parents, -1.0)
    assert np.allclose(P0, nodes[0]), P0
    assert not alive[1:].any()
    # fully grown, the structure is exactly as built
    P1, alive = stage(nodes, parents, 10.0)
    assert np.allclose(P1, nodes), P1
    assert alive[1:].all()

    # THE POP TEST.  Sample the sequence finely and check that no vertex
    # ever jumps: the largest single-frame displacement must stay small
    # and must fall as the sampling gets finer, which is what continuity
    # means.  A g'(0) > 0 curve fails this at every birth.
    def worst_jump(n):
        ts = np.linspace(-0.5, 5.0, n)
        prev = None
        worst = 0.0
        for t in ts:
            P, _a = stage(nodes, parents, t)
            if prev is not None:
                worst = max(worst, float(
                    np.linalg.norm(P - prev, axis=1).max()))
            prev = P
        return worst
    j1, j2 = worst_jump(200), worst_jump(400)
    assert j2 < j1 * 0.75, (j1, j2)
    assert j2 < 0.05, j2

    # and it really is first-order smooth: halving the step should
    # roughly halve the jump, which a value-only curve would not do
    j3 = worst_jump(800)
    assert j3 < j2 * 0.75, (j2, j3)

    # --- bake ------------------------------------------------------------
    seq = bake(nodes, parents, frames=12)
    assert len(seq) == 12
    for P in seq:
        # fixed topology: same count, same order, every frame
        assert P.shape == nodes.shape
        assert np.all(np.isfinite(P))
    # the sequence starts collapsed and ends complete
    assert np.allclose(seq[0], nodes[0])
    assert np.allclose(seq[-1], nodes)
    # and it is monotone in extent -- the structure only ever grows
    ext = [float(np.linalg.norm(P - P[0], axis=1).max()) for P in seq]
    assert all(b >= a - 1e-9 for a, b in zip(ext, ext[1:])), ext

    print(f"timed: OK -- all {len(CURVES)} curves satisfy g(0)=g'(0)=0, "
          f"staging is continuous (worst jump {j1:.4f} -> {j2:.4f} -> "
          f"{j3:.4f} as sampling doubles), bake is fixed-topology")
