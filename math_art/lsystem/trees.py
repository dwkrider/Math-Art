# Recursive branching trees.
#
# Part of the Math Art lsystem engine (`math_art/lsystem/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#

import math
import numpy as np

try:
    from ..surfaces.primitives import icosphere as _icosphere_shared
except ImportError:  # flat import outside the package
    from surfaces.primitives import icosphere as _icosphere_shared


# The old cap of 8000 segments was low enough to block the book's own
# figures: a ternary tree at depth 8 needs 13,120 segments, so ABOP's
# Figures 2.8b and 2.8c could not be reproduced at all.  Raised, with the
# clamp still reported so a runaway request is visible rather than silent.
MAX_SEGMENTS = 200_000


GOLDEN_ANGLE = math.radians(137.507764)


MOBILE_TWIST = math.radians(60.0)


# Honda's plates, as (theta1, theta2, R1, R2, alpha, N).  Kept in the
# lsystem package so the L-system presets and this generator's native
# mode read the same verified numbers; imported lazily so this module
# still works standalone.
HONDA_PLATES = {
    'PLATE_I': (16.7, -33.3, 0.85, 0.85, 137.5, 9),
    'PLATE_II': (0.0, -45.0, 0.90, 0.80, 137.5, 9),
    'PLATE_III': (16.7, -33.3, 0.90, 0.79, 137.5, 9),
    'PLATE_IV': (0.0, -45.0, 0.90, 0.70, 137.5, 9),
    'PLATE_V': (15.0, -30.0, 0.90, 0.75, 137.5, 9),
    'PLATE_VI_CONIC': (0.0, -45.0, 0.90, 0.60, 137.5, 9),
    'PLATE_VI_FLAT': (0.0, -45.0, 0.70, 0.90, 137.5, 9),
}


def honda_segment_count(depth):
    """Honda bifurcates, so the count is the binary-tree total."""
    return 2 ** (depth + 1) - 1


def build_honda(depth, theta1, theta2, r1, r2, alpha=137.5,
                trunk_len=1.0, width_exponent=2.0):
    """Honda's tree-like body: repeated bifurcation from four parameters.

    Implements the end-point formula of Honda (1971) directly.  Given a
    mother branch from P_A to P_B, write

        u = xB - xA,  v = yB - yA,  w = zB - zA,  L = sqrt(u^2+v^2+w^2)

    then a daughter with contraction R and angle theta ends at

        x = xB + R ( u cos t - L v sin t / sqrt(u^2+v^2) )
        y = yB + R ( v cos t + L u sin t / sqrt(u^2+v^2) )
        z = zB + R   w cos t

    The daughter is the mother direction tilted by theta toward the
    HORIZONTAL unit vector (-v, u, 0)/sqrt(u^2+v^2), which is normal to
    the vertical plane through the mother -- Honda's assumption (d), that
    the branching plane contains the mother branch and has the mother
    direction as its steepest-gradient line.  Its magnitude is exactly
    R*L, so assumption (c) holds.

    Two degenerate cases are handled as Honda handles them: a vertical
    mother (the trunk) forks in the x-z plane, and a zero branching angle
    uses the divergence angle `alpha` about the mother instead.  The
    signs of theta1 and theta2 alternate at every branching, as he
    specifies.

    Returns (segments, tips) in the same shape as `build_tree`, with
    radii from the da Vinci / Murray law at `width_exponent`.
    """
    segs, tips = [], []
    t1, t2 = math.radians(theta1), math.radians(theta2)
    ar = math.radians(alpha)
    inv = 1.0 / max(float(width_exponent), 1e-6)

    def daughter(pa, pb, R, t, az):
        """Honda's end-point formula, with his two degenerate cases."""
        u, v, w = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
        ln = math.sqrt(u * u + v * v + w * w)
        if ln < 1e-12:
            return pb
        h = math.sqrt(u * u + v * v)
        ct, st = math.cos(t), math.sin(t)
        if h < 1e-9:
            # A vertical mother has no horizontal normal, so the formula
            # is undefined.  Honda forks the trunk in a plane chosen by
            # the divergence angle instead; `az` carries that roll.
            sign = 1.0 if w >= 0.0 else -1.0
            return (pb[0] + R * ln * st * math.cos(az),
                    pb[1] + R * ln * st * math.sin(az),
                    pb[2] + R * ln * ct * sign)
        return (pb[0] + R * (u * ct - ln * v * st / h),
                pb[1] + R * (v * ct + ln * u * st / h),
                pb[2] + R * w * ct)

    def rec(pa, pb, level, width, az):
        # Widths follow the da Vinci / Murray law, split in proportion to
        # each child's contraction ratio, so the two children of an
        # unequal fork differ in thickness the way apical dominance
        # implies -- and w^n is conserved across the fork.
        kids = ((r1, t1 if level % 2 == 0 else -t1),
                (r2, t2 if level % 2 == 0 else -t2))
        share = [max(rr, 1e-9) for rr, _t in kids]
        tot = sum(x ** float(width_exponent) for x in share)
        scale = width / (tot ** inv) if tot > 0.0 else 0.0
        child_w = [s * scale for s in share]

        segs.append((tuple(map(float, pa)), tuple(map(float, pb)),
                     width, max(child_w) if level < depth else width * 0.5))
        if level >= depth:
            d = np.array(pb, dtype=float) - np.array(pa, dtype=float)
            n = float(np.linalg.norm(d))
            tips.append((tuple(map(float, pb)),
                         tuple(map(float, d / n if n > 1e-12 else d))))
            return
        for k, (R, t) in enumerate(kids):
            # A zero branching angle is Honda's other special case: the
            # daughter is placed by the divergence angle about the mother
            # rather than by a tilt.  The two daughters of the vertical
            # trunk go on opposite sides, hence the + k*pi.
            roll = az + k * math.pi
            q = daughter(pa, pb, R, t, roll)
            rec(pb, q, level + 1, child_w[k], az + ar)

    p0 = (0.0, 0.0, 0.0)
    p1 = (0.0, 0.0, float(trunk_len))
    rec(p0, p1, 0, 1.0, 0.0)
    return segs, tips


def tree_segment_count(arity, depth):
    """Segments in a full tree: trunk + all branch generations,
    sum_{g=0}^{depth} arity^g."""
    return (arity ** (depth + 1) - 1) // (arity - 1)


def mobile_segment_count(arity, levels):
    """Strings + spreader arms in a full mobile (weights excluded)."""
    strings = tree_segment_count(arity, levels)      # incl. final drops
    arms = strings - 1                               # arity^1 .. ^levels
    return strings + arms


def _basis(d):
    """Right-handed orthonormal (u, v) perpendicular to unit vector
    d, deterministically referenced to the world axes."""
    a = (np.array((0.0, 0.0, 1.0)) if abs(d[2]) < 0.999
         else np.array((1.0, 0.0, 0.0)))
    u = np.cross(a, d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


def build_tree(arity, depth, branch_angle, ratio, twist, trunk_len):
    """Recursive n-ary tree from the origin, trunk along +Z.

    Returns (segments, tips): segments as (p0, p1, r0, r1) tuples
    with relative radii ratio^level, tips as (position, direction)
    of the arity^depth final branch ends."""
    segs, tips = [], []
    sa, ca = math.sin(branch_angle), math.cos(branch_angle)

    def rec(p, d, length, level):
        q = p + d * length
        segs.append((tuple(map(float, p)), tuple(map(float, q)),
                     ratio ** level, ratio ** (level + 1)))
        if level >= depth:
            tips.append((tuple(map(float, q)),
                         tuple(map(float, d))))
            return
        u, v = _basis(d)
        g = level + 1
        for k in range(arity):
            az = 2.0 * math.pi * k / arity + g * twist
            cd = sa * (math.cos(az) * u + math.sin(az) * v) + ca * d
            cd /= np.linalg.norm(cd)
            rec(q, cd, length * ratio, g)

    rec(np.zeros(3), np.array((0.0, 0.0, 1.0)), float(trunk_len), 0)
    return segs, tips


def build_mobile(arity, levels, twist, trunk_len, ratio,
                 string_r=0.3, bar_r=1.0):
    """Hanging mobile from the origin, growing downward.

    Each hang point drops a vertical string of length trunk_len *
    ratio^g to a horizontal arity-armed spreader (arm length the
    same), generation g's arms rotated by g * twist in azimuth; the
    final tips drop a half-length string ending at a weight hang
    point.

    Returns (segments, weights): segments as (p0, p1, r0, r1) with
    relative radii, weights as (hang_point, spreader_z) -- the weight
    sphere is to be centred just below hang_point, and spreader_z is
    the height of the bar it hangs from (always above)."""
    segs, weights = [], []

    def rec(p, g):
        s = ratio ** g
        if g == levels:
            end = (p[0], p[1], p[2] - 0.5 * trunk_len * s)
            segs.append((p, end, string_r * s, string_r * s))
            weights.append((end, p[2]))
            return
        c = (p[0], p[1], p[2] - trunk_len * s)
        segs.append((p, c, string_r * s, string_r * s))
        arm = trunk_len * s
        for k in range(arity):
            az = 2.0 * math.pi * k / arity + g * twist
            tip = (c[0] + arm * math.cos(az),
                   c[1] + arm * math.sin(az), c[2])
            segs.append((c, tip, bar_r * s, 0.7 * bar_r * s))
            rec(tip, g + 1)

    rec((0.0, 0.0, 0.0), 0)
    return segs, weights


def icosphere(subdiv=0):
    """Unit icosphere: the icosahedron subdivided `subdiv` times.
    The shared builder is in `surfaces.primitives`.
    """
    return _icosphere_shared(subdiv, 'once')


_ICO = None


def _ico():
    global _ICO
    if _ICO is None:
        _ICO = icosphere(1)
    return _ICO
