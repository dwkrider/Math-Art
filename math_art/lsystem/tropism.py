
# Tropism and branch-width laws -- shared by every branching generator.
#
# Two small pieces of mathematics that buy more realism per line than
# anything else in the plant-modelling literature, kept here as plain
# functions so `fractal_tree_generator`, the L-system turtle, and the
# growth generators can all use one implementation.
#
# TROPISM.  A turtle's rotations are all RELATIVE to its current frame,
# but real branches respond to ABSOLUTE directions -- bending up toward
# light, or down under gravity.  The standard correction rotates the
# heading slightly toward a fixed tropism vector T after every drawn
# segment, by
#
#     alpha = e * |H x T|
#
# with `e` the axis's susceptibility to bending.  The formula has a
# physical reading rather than being a fudge: with T a force applied to
# the free end of segment H, the torque about its base is H x T, so the
# bend is proportional to the torque.  Two consequences fall out of it
# and are worth knowing before using it:
#
#   * a branch already parallel to T does not bend at all (the cross
#     product vanishes) -- a vertical leader under gravity stays
#     straight, which is correct, not a bug;
#   * the bend is strongest at right angles to T, so horizontal limbs
#     droop the most.
#
# Published values of `e` run from about 0.14 to 0.40.
#
# WIDTH.  The classical rule is da Vinci's: the combined
# cross-sectional AREA of the branches above a fork equals the area
# below it.  Murray's law generalises the exponent, and measurements of
# real trees put it between 2 and 3, so the exponent is a parameter here
# rather than a constant:
#
#     w_parent^n = w_child1^n + w_child2^n + ...
#
# For k equally sized children that gives w_child = w_parent * k^(-1/n),
# i.e. 1/sqrt(2) = 0.7071 for a binary fork under da Vinci's n = 2.
#
# The point of exposing `n` is that it DECOUPLES width from length.  A
# generator that shrinks width and length by the same ratio can only
# make self-similar trees; separating them is what lets a stout trunk
# carry long thin limbs, or the reverse.
#
# References:
# - Przemyslaw Prusinkiewicz, Aristid Lindenmayer and James Hanan,
#   "Developmental Models of Herbaceous Plants for Computer Imagery
#   Purposes", SIGGRAPH '88, Computer Graphics 22(4), pp. 141-150 --
#   the tropism formula and its torque justification.
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990, chapter 2 and Table 2.3 (values
#   of the elasticity e).
# - Leonardo da Vinci, Notebooks (the area-conservation rule).
# - Cecil D. Murray, "The physiological principle of minimum work
#   applied to the angle of branching of arteries", J. General
#   Physiology 9, 1926 (the cube-law form).
# - Adam Runions, Brendan Lane and Przemyslaw Prusinkiewicz, "Modeling
#   Trees with a Space Colonization Algorithm", Eurographics Workshop on
#   Natural Phenomena, 2007 -- the pipe-model radii r^n = r1^n + r2^n
#   used with n around 2.5.
# - Jason Weber and Joseph Penn, "Creation and rendering of realistic
#   trees", SIGGRAPH '95 -- `RatioPower`, the same free exponent under
#   another name.

import math

import numpy as np

DA_VINCI = 2.0          # area conservation
MURRAY = 3.0            # minimum-work flow
PIPE_MODEL = 2.5        # the value Runions et al. use for trees


def bend(H, L, U, T, elasticity):
    """Rotate the frame (H, L, U) toward `T` by alpha = e |H x T|.

    Returns a new (H, L, U).  A zero cross product -- H parallel or
    antiparallel to T -- leaves the frame untouched, which is the
    physically correct no-torque case.
    """
    H = np.asarray(H, dtype=float)
    L = np.asarray(L, dtype=float)
    U = np.asarray(U, dtype=float)
    if not elasticity:
        return H, L, U
    T = np.asarray(T, dtype=float)
    m = float(np.linalg.norm(T))
    if m < 1e-12:
        return H, L, U
    T = T / m
    axis = np.cross(H, T)
    mag = float(np.linalg.norm(axis))
    if mag < 1e-12:
        return H, L, U
    alpha = float(elasticity) * mag
    axis = axis / mag
    return (_rot(H, axis, alpha), _rot(L, axis, alpha), _rot(U, axis, alpha))


def _rot(v, axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1 - c)


def width_ratio(n_children, exponent=DA_VINCI):
    """Child/parent width for `n_children` EQUAL children.

    w_child = w_parent * k^(-1/n).  With da Vinci's n = 2 this is 0.7071
    for a binary fork and 0.5774 for a ternary one.
    """
    k = max(int(n_children), 1)
    n = max(float(exponent), 1e-6)
    return float(k ** (-1.0 / n))


def parent_width(child_widths, exponent=DA_VINCI):
    """Solve w_parent from the children: (sum w_i^n)^(1/n).

    This is the direction that matters when widths are assigned from the
    tips inward -- the pipe model -- because a parent's thickness is then
    determined by everything it carries rather than by its depth.
    """
    n = max(float(exponent), 1e-6)
    tot = sum(max(float(w), 0.0) ** n for w in child_widths)
    return float(tot ** (1.0 / n)) if tot > 0.0 else 0.0


def split_widths(w_parent, weights, exponent=DA_VINCI):
    """Distribute a parent's width among children in the given
    proportions while conserving `w^n`.

    `weights` need not sum to anything in particular; they are the
    relative vigour of each child, so an asymmetric fork (a dominant
    leader plus a small lateral) is expressible.
    """
    n = max(float(exponent), 1e-6)
    ws = [max(float(x), 0.0) for x in weights]
    tot = sum(w ** n for w in ws)
    if tot <= 0.0:
        return [0.0] * len(ws)
    scale = float(w_parent) / (tot ** (1.0 / n))
    return [w * scale for w in ws]


def taper(widths, exponent=DA_VINCI):
    """Convenience: `parent_width` over an iterable, for readability at
    call sites that are accumulating up a tree."""
    return parent_width(list(widths), exponent)


def elasticity_for(name="default"):
    """Published elasticity values, by rough habit (ABOP Table 2.3)."""
    table = {"weak": 0.14, "default": 0.22, "moderate": 0.27,
             "strong": 0.40}
    return table.get(str(name).lower(), 0.22)


def _selftest():
    # --- the width law -----------------------------------------------
    assert abs(width_ratio(2) - 0.70710678) < 1e-7, width_ratio(2)
    assert abs(width_ratio(3) - 0.57735027) < 1e-7, width_ratio(3)
    assert abs(width_ratio(2, MURRAY) - 2 ** (-1 / 3)) < 1e-12
    assert abs(width_ratio(1) - 1.0) < 1e-12, "one child inherits the width"

    # area really is conserved for da Vinci
    w = 1.0
    kids = [w * width_ratio(2)] * 2
    assert abs(sum(k ** 2 for k in kids) - w ** 2) < 1e-12
    # and volume-flow for Murray
    kids3 = [w * width_ratio(2, MURRAY)] * 2
    assert abs(sum(k ** 3 for k in kids3) - w ** 3) < 1e-12

    # the two directions are inverses
    assert abs(parent_width([0.70710678] * 2) - 1.0) < 1e-7
    assert abs(parent_width([1.0, 1.0], MURRAY) - 2 ** (1 / 3)) < 1e-12

    # asymmetric splits conserve too
    got = split_widths(1.0, [3.0, 1.0])
    assert abs(sum(g ** 2 for g in got) - 1.0) < 1e-12
    assert got[0] > got[1], "the more vigorous child must be thicker"

    # --- tropism ------------------------------------------------------
    H = np.array([1.0, 0.0, 0.0])
    L = np.array([0.0, 1.0, 0.0])
    U = np.array([0.0, 0.0, 1.0])
    # a horizontal branch under gravity must droop
    H2, L2, U2 = bend(H, L, U, (0, 0, -1), 0.3)
    assert H2[2] < -1e-6, H2
    assert abs(np.linalg.norm(H2) - 1.0) < 1e-9, "frame must stay unit"
    assert abs(float(np.dot(H2, L2))) < 1e-9, "frame must stay orthogonal"
    assert np.allclose(np.cross(H2, L2), U2, atol=1e-9)

    # the no-torque case: heading parallel to T does not bend
    Hp = np.array([0.0, 0.0, -1.0])
    Hp2, _, _ = bend(Hp, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                     (0, 0, -1), 0.3)
    assert np.allclose(Hp, Hp2, atol=1e-12), \
        "H parallel to T gives zero torque, hence no bend"

    # bending is monotone in elasticity, and zero elasticity is a no-op
    d1 = bend(H, L, U, (0, 0, -1), 0.1)[0][2]
    d2 = bend(H, L, U, (0, 0, -1), 0.3)[0][2]
    assert d2 < d1 < 0.0, (d1, d2)
    assert np.allclose(bend(H, L, U, (0, 0, -1), 0.0)[0], H)

    # a repeated bend converges toward T rather than overshooting past it
    h, l, u = H.copy(), L.copy(), U.copy()
    for _ in range(200):
        h, l, u = bend(h, l, u, (0, 0, -1), 0.2)
    assert h[2] < -0.99, f"should approach straight down, got {h}"

    assert elasticity_for("strong") == 0.40

    print("tropism: OK -- da Vinci/Murray width conservation both "
          "directions, asymmetric splits, tropism torque + frame "
          "orthonormality + no-torque case + convergence")
