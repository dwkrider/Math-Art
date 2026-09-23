# Chair44 (R44): the three-dimensional aperiodic monotile -- engine
#
# Pure geometry, no Blender.  The Blender layer is the CHAIR44 kind of
# `spacefill_generator`.
#
# THE SOLID.  Take a 2x2x2 block of unit cubes and remove one corner
# cube; the seven that remain are the CARRIER P, the "chair"
#
#     P = union of a + [0,1]^3 over a in {0,1}^3 \ {(1,1,1)},
#
# a rep-tile of volume 7 whose 24 exposed unit squares are its PANELS
# (21 on the outside -- the -x, -y and -z faces are full 2x2 grids while
# +x, +y and +z have only three each -- plus the three inset faces of
# the notch, at x = 1, y = 1 and z = 1).  Each panel carries eight tiny
# square pyramids, its FEATURES, at the in-panel offsets
#
#     (+-1/8, +-1/4)  and  (+-1/4, +-1/8)
#
# measured along the panel's two named axes.  Every feature has base
# side 1/50 (half-width eta = 1/100) and signed height a/10000 along the
# outward normal, where a is one of +-1 .. +-12: positive is a bump,
# negative a dent of the same depth.  192 features in all; the table is
# PANELS below, transcribed from Figure 3 of the paper.  Bump and dent
# volumes cancel exactly (each of the twelve magnitudes is carried by
# eight positive and eight negative marks), so the solid still has
# volume exactly 7.
#
# WHY THE FEATURES.  The bare chair tiles space periodically.  The
# pyramids distinguish arrangements its flat panels cannot: two features
# mate only as a protrusion into a recess of the SAME magnitude, and
# since each of the twelve magnitudes belongs to exactly one signed
# group, a shared panel matches only when the two profiles agree centre
# by centre.  The heights are not decorative -- they are forced.  Take
# the 21 face contacts occurring inside the eight-child dissection
# below, close that set under refinement (it closes at 30), and impose
# a_u = -a_v at every pair of marks the contacts bring into coincidence:
# the resulting 372 equations split the 192 marks into exactly twelve
# balanced components, and numbering those 1..12 is the table.
#
# THE SUBSTITUTION.  The chair is a rep-8 rep-tile: eight rotated copies
# form the doubled chair 2P.  Writing a frame as an axis permutation p
# with a sign vector s, acting by G(x)_i = s_i x_{p_i}, the eight child
# poses are CHILDREN below -- every frame a proper rotation, their 56
# unit cells partitioning 2P exactly.  Refinement acts on poses by
#
#     (G, t)  ->  { (G H, 2t + G u) : (H, u) a child pose },
#
# so iterating from one chair gives patches of 8, 64, 512, ... chairs
# filling 2P, 4P, 8P, ...  Contacts between neighbouring chairs in such
# a patch are drawn from ATLAS44, the 44 relative poses the features
# permit out of the 2,388 that the bare carrier allows.
#
# WHAT IS AND IS NOT ESTABLISHED.  Tsiokos presents Chair44 as a
# strongly aperiodic monotile -- a single solid admitting tilings of
# space, none of them periodic, none with a symmetry of infinite order,
# all homochiral.  That is presented as a PROOF SUBMISSION and is not
# refereed: its Lean development is kernel-checked only modulo a named
# compiler hook per machine-decided theorem, and the written geometric
# lemmas stand as exposition.  Nothing in this module depends on the
# aperiodicity theorem.  What it builds -- the solid, the eight-child
# dissection, and patches grown by refinement -- is finite, exact and
# checked here in `_selftest`: the panel table is re-derived from the
# carrier, the 56 child cells are shown to partition 2P, all 384
# coincident feature sites inside the dissection are shown to mate bump
# to dent at equal magnitude, every contact of a depth-2 patch is shown
# to lie in ATLAS44, and the tile mesh is shown to close with the
# published V = 2138, E = 6408, F = 4272 and volume 7.
#
# THE ARROW MARKING.  Goodman-Strauss re-draws the same rule as one
# flat arrow per panel in three colours -- blue meets blue, green
# meets red -- which is far easier to read than 192 pyramids
# 1/10000 of a cell tall.  That re-drawing is reproduced here, and it
# is checked rather than assumed: `_selftest` derives the colour
# classes' behaviour from ATLAS44 and then shows that the arrow rule,
# applied to the same 2,388 candidate poses the paper enumerates,
# admits exactly the 44-contact atlas once reflected copies are set
# aside.  (Allowing reflections it admits 60; the extra 16 are all
# improper, and a physical solid cannot be reflected.)  So the arrows
# are not an illustration of the rule -- for unreflected tiles they
# are the rule.
#
# References:
# - Ioannis Tsiokos, "A Strongly Aperiodic Monotile in Three
#   Dimensions", arXiv:2609.19214 (2026) -- the Chair44 (R44) solid,
#   its 24-panel / 192-feature recipe, the eight-child chair
#   substitution and the 44-contact atlas.  Presented as a proof
#   submission; the aperiodicity theorem is not yet refereed.
# - Chaim Goodman-Strauss, "Notes on a strongly aperiodic monotile in
#   E^3", arXiv:2609.24779 (2026) -- the three-colour arrow marking
#   drawn here, the reading of Chair44 as a marked three-dimensional
#   L-tile, and a short Berger-style proof of its aperiodicity.
# - Joshua E. S. Socolar and Joan M. Taylor, "Forcing nonperiodicity
#   with a single tile", Math. Intelligencer 34(1):18-28 (2012) -- the
#   question this solid answers, and the Schmitt-Conway-Danzer biprism
#   whose screw motions it avoids.
# - David Smith, Joseph Samuel Myers, Craig S. Kaplan and Chaim
#   Goodman-Strauss, "An aperiodic monotile", Comb. Theory 4(1):6
#   (2024) -- the planar einstein whose unique-hierarchy argument this
#   construction follows.

import itertools
from fractions import Fraction as F

# ------------------------------------------------------------------
# the eight in-panel feature offsets, in the reading order of Figure 3:
# top-left, top-right, upper-left, upper-right, lower-left,
# lower-right, bottom-left, bottom-right.  The top and bottom pairs sit
# at |u| = 1/8 and |v| = 1/4; the two middle pairs the other way round.
# ------------------------------------------------------------------
OFFSETS = ((F(-1, 8), F(1, 4)), (F(1, 8), F(1, 4)),
           (F(-1, 4), F(1, 8)), (F(1, 4), F(1, 8)),
           (F(-1, 4), F(-1, 8)), (F(1, 4), F(-1, 8)),
           (F(-1, 8), F(-1, 4)), (F(1, 8), F(-1, 4)))

# ------------------------------------------------------------------
# The panel recipe (Figure 3).  Key: (normal axis, normal sign, plane
# coordinate, centre along the first named axis, centre along the
# second).  The named axes are the two non-normal coordinates in
# increasing order, so (y, z) for an x-normal, (x, z) for a y-normal
# and (x, y) for a z-normal -- exactly the diagram headings.  Value:
# (panel id, the eight coefficients in OFFSETS order).
#
# Panels 4, 14 and 23 are the inset faces of the notch and sit at
# plane 1; every other panel sits at 0 or 2.
# ------------------------------------------------------------------
PANELS = {
    (0, -1, 0, F(1, 2), F(1, 2)): (0, (2, 4, 6, 8, 5, 7, 1, 3)),
    (0, -1, 0, F(1, 2), F(3, 2)): (1, (-1, -3, -5, -7, -6, -8, -2, -4)),
    (0, -1, 0, F(3, 2), F(1, 2)): (2, (8, 7, 4, 3, 2, 1, 6, 5)),
    (0, -1, 0, F(3, 2), F(3, 2)): (3, (10, 12, -11, -12, -9, -10, 9, 11)),
    (0, 1, 1, F(3, 2), F(3, 2)): (4, (-2, -4, -6, -8, -5, -7, -1, -3)),
    (0, 1, 2, F(1, 2), F(1, 2)): (5, (-11, -9, 10, 9, 12, 11, -12, -10)),
    (0, 1, 2, F(1, 2), F(3, 2)): (6, (-5, -6, -1, -2, -3, -4, -7, -8)),
    (0, 1, 2, F(3, 2), F(1, 2)): (7, (4, 2, 8, 6, 7, 5, 3, 1)),
    (1, -1, 0, F(1, 2), F(1, 2)): (8, (-2, -4, -6, -8, -5, -7, -1, -3)),
    (1, -1, 0, F(1, 2), F(3, 2)): (9, (1, 3, 5, 7, 6, 8, 2, 4)),
    (1, 1, 2, F(1, 2), F(1, 2)): (10, (11, 9, -10, -9, -12, -11, 12, 10)),
    (1, 1, 2, F(1, 2), F(3, 2)): (11, (5, 6, 1, 2, 3, 4, 7, 8)),
    (1, -1, 0, F(3, 2), F(1, 2)): (12, (-8, -7, -4, -3, -2, -1, -6, -5)),
    (1, -1, 0, F(3, 2), F(3, 2)): (13, (-10, -12, 11, 12, 9, 10, -9, -11)),
    (1, 1, 1, F(3, 2), F(3, 2)): (14, (2, 4, 6, 8, 5, 7, 1, 3)),
    (1, 1, 2, F(3, 2), F(1, 2)): (15, (-4, -2, -8, -6, -7, -5, -3, -1)),
    (2, -1, 0, F(1, 2), F(1, 2)): (16, (11, 9, -10, -9, -12, -11, 12, 10)),
    (2, 1, 2, F(1, 2), F(1, 2)): (17, (-11, -9, 10, 9, 12, 11, -12, -10)),
    (2, -1, 0, F(1, 2), F(3, 2)): (18, (-1, -3, -5, -7, -6, -8, -2, -4)),
    (2, 1, 2, F(1, 2), F(3, 2)): (19, (-5, -6, -1, -2, -3, -4, -7, -8)),
    (2, -1, 0, F(3, 2), F(1, 2)): (20, (8, 7, 4, 3, 2, 1, 6, 5)),
    (2, 1, 2, F(3, 2), F(1, 2)): (21, (4, 2, 8, 6, 7, 5, 3, 1)),
    (2, -1, 0, F(3, 2), F(3, 2)): (22, (10, 12, -11, -12, -9, -10, 9, 11)),
    (2, 1, 1, F(3, 2), F(3, 2)): (23, (-11, -9, 10, 9, 12, 11, -12, -10)),
}

# ------------------------------------------------------------------
# The eight child poses of the doubled chair (Table 1): permutation,
# signs, translation, in the coordinates of 2P.  Every frame is a
# proper rotation and the 56 cells partition 2P exactly.
# ------------------------------------------------------------------
CHILDREN = (
    ("000", (0, 1, 2), (1, 1, 1), (0, 0, 0)),
    ("001", (1, 0, 2), (1, 1, -1), (0, 0, 4)),
    ("010", (0, 2, 1), (1, -1, 1), (0, 4, 0)),
    ("011", (2, 0, 1), (1, -1, -1), (0, 4, 4)),
    ("100", (2, 1, 0), (-1, 1, 1), (4, 0, 0)),
    ("101", (1, 2, 0), (-1, 1, -1), (4, 0, 4)),
    ("110", (0, 1, 2), (-1, -1, 1), (4, 4, 0)),
    ("central", (0, 1, 2), (1, 1, 1), (1, 1, 1)),
)

# ------------------------------------------------------------------
# The 44-contact atlas (Figure 7): the relative poses a neighbour may
# take against a root chair in the identity pose.  Used here only as a
# check on the patches this module grows.
# ------------------------------------------------------------------
ATLAS44 = (
    ((0, 1, 2), (-1, -1, 1), (2, 2, -2)),
    ((0, 1, 2), (-1, -1, 1), (2, 2, 2)),
    ((0, 1, 2), (-1, -1, 1), (3, 3, -1)),
    ((0, 1, 2), (-1, -1, 1), (3, 3, 1)),
    ((0, 1, 2), (-1, 1, -1), (2, -2, 2)),
    ((0, 1, 2), (-1, 1, -1), (2, 2, 2)),
    ((0, 1, 2), (1, -1, -1), (-2, 2, 2)),
    ((0, 1, 2), (1, -1, -1), (2, 2, 2)),
    ((0, 1, 2), (1, 1, 1), (-1, -1, -1)),
    ((0, 1, 2), (1, 1, 1), (1, 1, 1)),
    ((0, 2, 1), (-1, 1, 1), (4, 0, 0)),
    ((0, 2, 1), (1, -1, 1), (-1, 3, -1)),
    ((0, 2, 1), (1, -1, 1), (0, 4, 0)),
    ((0, 2, 1), (1, 1, -1), (0, 0, 4)),
    ((0, 2, 1), (1, 1, -1), (1, 1, 3)),
    ((1, 0, 2), (-1, 1, 1), (0, 0, 0)),
    ((1, 0, 2), (-1, 1, 1), (4, 0, 0)),
    ((1, 0, 2), (1, -1, 1), (0, 0, 0)),
    ((1, 0, 2), (1, -1, 1), (0, 4, 0)),
    ((1, 0, 2), (1, 1, -1), (-1, -1, 3)),
    ((1, 0, 2), (1, 1, -1), (0, 0, 0)),
    ((1, 0, 2), (1, 1, -1), (0, 0, 4)),
    ((1, 0, 2), (1, 1, -1), (1, 1, 3)),
    ((1, 2, 0), (-1, -1, 1), (2, 2, -2)),
    ((1, 2, 0), (-1, -1, 1), (2, 2, 2)),
    ((1, 2, 0), (-1, -1, 1), (3, 3, 1)),
    ((1, 2, 0), (-1, 1, -1), (2, -2, 2)),
    ((1, 2, 0), (-1, 1, -1), (2, 2, 2)),
    ((1, 2, 0), (-1, 1, -1), (3, -1, 3)),
    ((1, 2, 0), (1, -1, -1), (-2, 2, 2)),
    ((1, 2, 0), (1, -1, -1), (2, 2, 2)),
    ((2, 0, 1), (-1, -1, 1), (2, 2, -2)),
    ((2, 0, 1), (-1, -1, 1), (2, 2, 2)),
    ((2, 0, 1), (-1, -1, 1), (3, 3, 1)),
    ((2, 0, 1), (-1, 1, -1), (2, -2, 2)),
    ((2, 0, 1), (-1, 1, -1), (2, 2, 2)),
    ((2, 0, 1), (1, -1, -1), (-2, 2, 2)),
    ((2, 0, 1), (1, -1, -1), (-1, 3, 3)),
    ((2, 0, 1), (1, -1, -1), (2, 2, 2)),
    ((2, 1, 0), (-1, 1, 1), (3, -1, -1)),
    ((2, 1, 0), (-1, 1, 1), (4, 0, 0)),
    ((2, 1, 0), (1, -1, 1), (0, 4, 0)),
    ((2, 1, 0), (1, 1, -1), (0, 0, 4)),
    ((2, 1, 0), (1, 1, -1), (1, 1, 3)),
)

# exact geometry of the published solid
ETA_TRUE = F(1, 100)        # feature base half-width
HEIGHT_TRUE = F(1, 10000)   # unit of signed height
ETA_SHOWN = F(5, 100)       # the paper's own figure convention, bases x5

# the carrier's volume centroid: 4 cells at 1/2 and 3 at 3/2 per axis
CENTROID = (F(13, 14), F(13, 14), F(13, 14))


# ------------------------------------------------------------------
# The arrow markings (Goodman-Strauss).
#
# The 192 pyramids are the physical rule, but they are a poor picture
# of it.  Goodman-Strauss re-draws the same rule as ONE flat arrow per
# panel, in three colours: blue meets blue, green meets red.  The
# structure behind it is that each panel owns exactly one "special"
# vertex -- one of the seven corners of the 2x2x2 cube that survive on
# the chair, or the concave socket at the cube's centre -- and the
# three panels meeting at each of those eight vertices carry one
# marking of each colour.  The arrow lies along the panel's diagonal
# and points at that vertex, so two panels match when their arrows
# coincide and their colours are compatible.
#
# This is not a decoration: `_selftest` checks that the arrow rule,
# restricted to unreflected copies, admits exactly the 44-contact
# atlas -- the same 44 out of the same 2,388 candidates.  (Allowing
# reflections it admits 60, the extra 16 all improper; physical tiles
# cannot be reflected, which is the case Goodman-Strauss makes.)
# ------------------------------------------------------------------

ARROW_COLORS = ("BLUE", "GREEN", "RED")

# panel id -> arrow colour index.  Blue is forced (it is the
# self-matching class); which of the other two is called green and
# which red is a free relabelling, since the rule is symmetric in
# them.
PANEL_COLOR = {
    0: 2, 1: 1, 2: 2, 3: 0, 4: 1, 5: 0, 6: 1, 7: 2,
    8: 1, 9: 2, 10: 0, 11: 2, 12: 1, 13: 0, 14: 2, 15: 1,
    16: 0, 17: 0, 18: 1, 19: 1, 20: 2, 21: 2, 22: 0, 23: 0,
}

# the eight special vertices: the seven surviving corners of the
# 2-cube, plus the concave socket at its centre
SPECIAL_VERTICES = tuple(
    [v for v in itertools.product((0, 2), repeat=3) if v != (2, 2, 2)]
    + [(1, 1, 1)])


def panel_home(key):
    """The one special vertex among a panel's four corners."""
    ax, _sg, plane, u0, v0 = key
    o0, o1 = (i for i in range(3) if i != ax)
    found = []
    for du in (F(-1, 2), F(1, 2)):
        for dv in (F(-1, 2), F(1, 2)):
            p = [None, None, None]
            p[ax] = F(plane)
            p[o0] = u0 + du
            p[o1] = v0 + dv
            t = tuple(int(x) for x in p)
            if t in SPECIAL_VERTICES:
                found.append((t, (du, dv)))
    if len(found) != 1:
        raise AssertionError(f"panel {key} owns {len(found)} vertices")
    return found[0]


# arrow shape in panel coordinates, along the diagonal toward the home
# vertex: a shaft with a barbed head, sized to sit inside the unit
# panel with room to spare.
_A_TAIL, _A_TIP = -0.50, 0.62     # along the diagonal, from the centre
_A_HEAD, _A_BARB = 0.34, 0.20     # head length, barb setback from tip
_A_HALF, _A_SHAFT = 0.17, 0.072   # head half-width, shaft half-width
_A_SINK, _A_RISE = -0.005, 0.014  # below / above the panel surface


def arrow_polygon(su, sv):
    """The arrow outline in panel (u, v), pointing at (su, sv)/2.

    Returned counterclockwise, as seven points.
    """
    r = 0.7071067811865476
    du, dv = su * r, sv * r          # unit vector along the diagonal
    nu, nv = -dv, du                 # and its perpendicular

    def at(along, across):
        return (du * along + nu * across, dv * along + nv * across)

    pts = [at(_A_TAIL, _A_SHAFT),
           at(_A_TIP - _A_BARB, _A_SHAFT),
           at(_A_TIP - _A_HEAD, _A_HALF),
           at(_A_TIP, 0.0),
           at(_A_TIP - _A_HEAD, -_A_HALF),
           at(_A_TIP - _A_BARB, -_A_SHAFT),
           at(_A_TAIL, -_A_SHAFT)]
    area = sum(pts[i][0] * pts[(i + 1) % 7][1]
               - pts[(i + 1) % 7][0] * pts[i][1] for i in range(7)) / 2
    return pts[::-1] if area < 0 else pts


# The outline is concave at the two barb junctions, so it is cut into
# convex pieces by hand rather than fanned: the shaft quad, the tip,
# and the two barbs.  Indices are into arrow_polygon's seven points.
_ARROW_PIECES = ((0, 1, 5, 6), (2, 3, 4), (1, 2, 4), (1, 4, 5))


# ------------------------------------------------------------------
# frames
# ------------------------------------------------------------------

def frame(p, s):
    """The matrix of G(x)_i = s_i * x_{p_i}, as rows."""
    M = [[0, 0, 0] for _ in range(3)]
    for i in range(3):
        M[i][p[i]] = s[i]
    return tuple(tuple(r) for r in M)


def mat_mul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def mat_apply(M, v):
    return tuple(M[i][0] * v[0] + M[i][1] * v[1] + M[i][2] * v[2]
                 for i in range(3))


def mat_transpose(M):
    return tuple(tuple(M[j][i] for j in range(3)) for i in range(3))


def det(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def proper_frames():
    """The 24 proper signed permutation matrices, in a fixed order."""
    out = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            M = frame(p, s)
            if det(M) == 1:
                out.append(M)
    return tuple(out)


PROPER_FRAMES = proper_frames()
_FRAME_INDEX = {M: i for i, M in enumerate(PROPER_FRAMES)}


def frame_index(M):
    """Which of the 24 proper cubic rotations a registered pose uses."""
    return _FRAME_INDEX[tuple(tuple(r) for r in M)]


# ------------------------------------------------------------------
# carrier, panels, features
# ------------------------------------------------------------------

def carrier_cells():
    """The seven unit cells of the chair, by their lower corners."""
    return tuple(a for a in itertools.product((0, 1), repeat=3)
                 if a != (1, 1, 1))


def panels_from_carrier():
    """Derive the 24 exposed panels from the carrier geometry alone.

    This is the cross-check on the transcribed table: a panel centre
    typed wrongly into PANELS shows up here as a key mismatch.
    """
    cells = set(carrier_cells())
    out = set()
    for a in cells:
        for ax in range(3):
            for sg in (-1, 1):
                nb = list(a)
                nb[ax] += sg
                if tuple(nb) in cells:
                    continue
                o0, o1 = (i for i in range(3) if i != ax)
                plane = a[ax] + (1 if sg == 1 else 0)
                out.add((ax, sg, F(plane),
                         F(2 * a[o0] + 1, 2), F(2 * a[o1] + 1, 2)))
    return out


def features(height=HEIGHT_TRUE):
    """The 192 features, as (centre, outward normal, coefficient).

    The centre is the pyramid's base centre in exact eighths; the
    signed height along the outward normal is coefficient * height.
    """
    out = []
    for (ax, sg, plane, u0, v0), (_pid, coeffs) in PANELS.items():
        o0, o1 = (i for i in range(3) if i != ax)
        n = [0, 0, 0]
        n[ax] = sg
        for (du, dv), a in zip(OFFSETS, coeffs):
            c = [None, None, None]
            c[ax] = F(plane)
            c[o0] = u0 + du
            c[o1] = v0 + dv
            out.append((tuple(c), tuple(n), a))
    return out


# ------------------------------------------------------------------
# the tile mesh
# ------------------------------------------------------------------

def _panel_axes(ax):
    """The two named in-panel axes, and whether their cross product
    points along +e_ax.  (e_y x e_z = +e_x, e_x x e_z = -e_y,
    e_x x e_y = +e_z.)"""
    o0, o1 = (i for i in range(3) if i != ax)
    return o0, o1, (1 if ax != 1 else -1)


def bare_tile_mesh():
    """The carrier alone: 24 unit-square panels, outward-oriented.

    Not the featured mesh with its pyramids flattened -- that would
    still carry the whole subdivision grid.  This is the cheap tile
    for deep patches.
    """
    index = {}
    verts = []
    faces = []

    def vid(pt):
        i = index.get(pt)
        if i is None:
            i = index[pt] = len(verts)
            verts.append(pt)
        return i

    for (ax, sg, plane, u0, v0) in sorted(PANELS, key=lambda k: PANELS[k][0]):
        o0, o1, orient = _panel_axes(ax)
        corners = ((u0 - F(1, 2), v0 - F(1, 2)), (u0 + F(1, 2), v0 - F(1, 2)),
                   (u0 + F(1, 2), v0 + F(1, 2)), (u0 - F(1, 2), v0 + F(1, 2)))
        quad = []
        for (u, v) in corners:
            p = [None, None, None]
            p[ax] = F(plane)
            p[o0] = u
            p[o1] = v
            quad.append(vid(tuple(p)))
        if orient != sg:
            quad.reverse()
        faces.append(tuple(quad))
    return verts, faces


def arrow_tile_mesh():
    """The bare carrier, plus Goodman-Strauss's arrow marking per panel.

    The body is exactly `bare_tile_mesh` -- 24 flat panels, volume 7.
    Each panel then carries one arrow as a thin closed slab lying along
    the panel diagonal and pointing at the panel's special vertex,
    sunk slightly into the surface so its underside never z-fights
    with the panel.

    Returns (verts, faces, colors) with `colors` parallel to `faces`:
    None for a body panel, else an index into ARROW_COLORS.
    """
    index = {}
    verts = []
    faces = []
    colors = []

    def vid(pt):
        pt = tuple(float(x) for x in pt)
        i = index.get(pt)
        if i is None:
            i = index[pt] = len(verts)
            verts.append(pt)
        return i

    body_v, body_f = bare_tile_mesh()
    remap = [vid(q) for q in body_v]
    for f in body_f:
        faces.append(tuple(remap[i] for i in f))
        colors.append(None)

    for key in sorted(PANELS, key=lambda k: PANELS[k][0]):
        ax, sg, plane, u0, v0 = key
        pid, _coeffs = PANELS[key]
        o0, o1, orient = _panel_axes(ax)
        flip = (orient != sg)
        _home, (du, dv) = panel_home(key)
        poly = arrow_polygon(1 if du > 0 else -1, 1 if dv > 0 else -1)

        def pt(uv, lift):
            q = [None, None, None]
            q[ax] = float(plane) + sg * lift
            q[o0] = float(u0) + uv[0]
            q[o1] = float(v0) + uv[1]
            return vid(tuple(q))

        lo = [pt(q, _A_SINK) for q in poly]
        hi = [pt(q, _A_RISE) for q in poly]
        col = PANEL_COLOR[pid]
        for piece in _ARROW_PIECES:
            top = [hi[i] for i in piece]
            bot = [lo[i] for i in piece][::-1]
            faces.append(tuple(top[::-1] if flip else top))
            colors.append(col)
            faces.append(tuple(bot[::-1] if flip else bot))
            colors.append(col)
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            side = [lo[i], lo[j], hi[j], hi[i]]
            faces.append(tuple(side[::-1] if flip else side))
            colors.append(col)
    return verts, faces, colors


def arrow_area():
    """Plan area of one arrow marking (both diagonals give the same)."""
    pts = arrow_polygon(1, 1)
    n = len(pts)
    return abs(sum(pts[i][0] * pts[(i + 1) % n][1]
                   - pts[(i + 1) % n][0] * pts[i][1]
                   for i in range(n))) / 2


def tile_mesh(eta=ETA_TRUE, height=HEIGHT_TRUE):
    """One copy of Q, in exact rational coordinates.

    Each panel is cut by the lines through every feature base edge --
    u in {+-1/8 +- eta, +-1/4 +- eta} and the panel edges +-1/2, and
    likewise for v -- giving a 9x9 grid of 81 cells.  The eight cells
    centred on the feature offsets become pyramids (base quad replaced
    by four triangles meeting at the apex, at signed height
    coefficient * height along the outward normal); the other 73 stay
    flat quads.  That is 24 * (73 quads + 32 triangles) = 1752 quads
    and 768 triangles, which is the published 4272 triangles once the
    quads are split.

    Adjacent panels agree on the cut coordinates along their shared
    edge -- every panel centre is at a half-integer and the pattern is
    symmetric about it -- so welding on the exact coordinate closes the
    surface, and it closes on the published 2,138 vertices.

    Returns (verts, faces) with verts exact and faces a mix of quads
    and triangles, all oriented outward.
    """
    if not 0 < eta < F(1, 16):
        raise ValueError("feature half-width must satisfy 0 < eta < 1/16 "
                         "(the bases are 1/8 apart)")
    cuts = []
    for base in (F(-1, 4), F(-1, 8), F(1, 8), F(1, 4)):
        cuts.extend((base - eta, base + eta))
    cuts = [F(-1, 2)] + sorted(cuts) + [F(1, 2)]
    # index of the low edge of each feature's base interval
    low = {base - eta: i for i, base in
           ((cuts.index(b - eta), b) for b in
            (F(-1, 4), F(-1, 8), F(1, 8), F(1, 4)))}
    slot = {b: cuts.index(b - eta) for b in
            (F(-1, 4), F(-1, 8), F(1, 8), F(1, 4))}
    del low

    index = {}
    verts = []
    faces = []

    def vid(pt):
        i = index.get(pt)
        if i is None:
            i = index[pt] = len(verts)
            verts.append(pt)
        return i

    for key in sorted(PANELS, key=lambda k: PANELS[k][0]):
        ax, sg, plane, u0, v0 = key
        _pid, coeffs = PANELS[key]
        o0, o1, orient = _panel_axes(ax)
        flip = (orient != sg)

        def point(u, v, lift=0):
            p = [None, None, None]
            p[ax] = F(plane) + sg * lift
            p[o0] = u0 + u
            p[o1] = v0 + v
            return vid(tuple(p))

        pyramids = {}
        for (du, dv), a in zip(OFFSETS, coeffs):
            pyramids[(slot[du], slot[dv])] = a

        for i in range(9):
            for j in range(9):
                u_lo, u_hi = cuts[i], cuts[i + 1]
                v_lo, v_hi = cuts[j], cuts[j + 1]
                quad = [point(u_lo, v_lo), point(u_hi, v_lo),
                        point(u_hi, v_hi), point(u_lo, v_hi)]
                if flip:
                    quad.reverse()
                a = pyramids.get((i, j))
                if a is None:
                    faces.append(tuple(quad))
                    continue
                apex = point((u_lo + u_hi) / 2, (v_lo + v_hi) / 2,
                             a * height)
                for k in range(4):
                    faces.append((quad[k], quad[(k + 1) % 4], apex))
    return verts, faces


def mesh_volume(verts, faces):
    """Exact signed volume of a closed, outward-oriented mesh."""
    tot = F(0)
    for f in faces:
        a = verts[f[0]]
        for k in range(1, len(f) - 1):
            b, c = verts[f[k]], verts[f[k + 1]]
            tot += (a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return tot / 6


def triangulate(faces):
    """Fan-split every face into triangles."""
    out = []
    for f in faces:
        for k in range(1, len(f) - 1):
            out.append((f[0], f[k], f[k + 1]))
    return out


# ------------------------------------------------------------------
# the substitution
# ------------------------------------------------------------------

def refine(pose):
    """The eight children of one pose (G, t)."""
    G, t = pose
    out = []
    for _name, p, s, u in CHILDREN:
        H = frame(p, s)
        Gu = mat_apply(G, u)
        out.append((mat_mul(G, H),
                    tuple(2 * t[i] + Gu[i] for i in range(3))))
    return out


def patch(depth):
    """A substitution patch of 8**depth chairs.

    Returns [(frame, integer translation, group)], where `group` is the
    index of the child taken at the FIRST refinement -- so at depth n
    the patch splits into eight groups of 8**(n-1) chairs, and those
    eight groups are the level-(n-1) supertiles: at depth 2 they are
    the eight 2-scaled chairs of the paper's Figure 6.  At depth 0 the
    single native chair has group 0.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    identity = frame((0, 1, 2), (1, 1, 1))
    if depth == 0:
        return [(identity, (0, 0, 0), 0)]
    tagged = [(child, g) for g, child in enumerate(refine((identity,
                                                          (0, 0, 0))))]
    for _ in range(depth - 1):
        tagged = [(kid, g) for pose, g in tagged for kid in refine(pose)]
    return [(G, t, g) for (G, t), g in tagged]


def pose_cells(G, t):
    """The seven unit cells a posed chair occupies."""
    out = []
    for a in carrier_cells():
        corners = [mat_apply(G, (a[0] + dx, a[1] + dy, a[2] + dz))
                   for dx, dy, dz in itertools.product((0, 1), repeat=3)]
        out.append(tuple(min(c[i] for c in corners) + t[i]
                         for i in range(3)))
    return out


def _contact_spans():
    """For every legal contact, the sum of the two chairs' centroid
    distances to their shared panel plane.

    This is what sets how much room a shrink-about-own-centroid gap
    opens between two mated panels: the clearance is
    (1 - gap) * span.  The smallest span belongs to the notch, whose
    three panels sit only 1/14 from the centroid -- but a notch panel
    always mates with an OUTER panel of its neighbour (13/14 or 15/14
    away), never with another notch, so the minimum is 1, not 2/14.
    """
    root = frame((0, 1, 2), (1, 1, 1))
    root_cells = set(pose_cells(root, (0, 0, 0)))
    spans = set()
    for p, s, off in ATLAS44:
        G = frame(p, s)
        nb_centroid = tuple(mat_apply(G, CENTROID)[i] + off[i]
                            for i in range(3))
        for b in pose_cells(G, off):
            for ax in range(3):
                for sg in (-1, 1):
                    a = list(b)
                    a[ax] += sg
                    if tuple(a) not in root_cells:
                        continue
                    plane = F(max(a[ax], b[ax]))
                    spans.add(abs(plane - CENTROID[ax])
                              + abs(plane - nb_centroid[ax]))
    return spans


MIN_CONTACT_SPAN = min(_contact_spans())


def max_relief(gap):
    """Largest height exaggeration that keeps the pyramids clear.

    At gap < 1 every chair shrinks about its own centroid, so a bump
    and its matching dent slide apart laterally and each bump ends up
    facing flat panel -- or, in the worst case, another bump coming
    the other way.  Requiring both of the tallest bumps to fit in the
    clearance gives this bound.  It is a bound on a cosmetic overlap,
    not on correctness: the chairs themselves never intersect, because
    the carrier is star-shaped about its centroid.
    """
    if gap >= 1.0:
        return float('inf')
    clearance = (1.0 - gap) * float(MIN_CONTACT_SPAN)
    return clearance * 10000.0 / (2.0 * gap * 12.0)


def contacts(poses):
    """Relative poses of every face-adjacent pair in a patch.

    Yields (i, j, relative frame, relative translation) with the pair
    normalized so that tile i sits in the identity pose.
    """
    owner = {}
    for idx, (G, t, *_rest) in enumerate(poses):
        for cell in pose_cells(G, t):
            owner[cell] = idx
    seen = set()
    for cell, i in owner.items():
        for ax in range(3):
            for sg in (-1, 1):
                nb = list(cell)
                nb[ax] += sg
                j = owner.get(tuple(nb))
                if j is None or j == i or (i, j) in seen:
                    continue
                seen.add((i, j))
                Gi, ti = poses[i][0], poses[i][1]
                Gj, tj = poses[j][0], poses[j][1]
                Ri = mat_transpose(Gi)
                d = tuple(tj[k] - ti[k] for k in range(3))
                yield i, j, mat_mul(Ri, Gj), mat_apply(Ri, d)


# ------------------------------------------------------------------
# self-test
# ------------------------------------------------------------------

def _check(ok, label):
    print(f"  {'OK  ' if ok else 'BAD '} {label}")
    return bool(ok)


def _selftest():
    good = True

    # 1. the transcribed panel table's keys are forced by the carrier
    good &= _check(panels_from_carrier() == set(PANELS),
                   "24 panels derived from the carrier match the table")

    # 2. the paper's worked example.  This is the pin on the in-panel
    #    LAYOUT: the 384/384 mating check below catches a mirrored or
    #    transposed reading of Figure 3, but NOT a half-turn of every
    #    panel, because a half-turn about the panel centre commutes
    #    with every child frame.  Panel 13 is the paper's own example.
    want = {((F(11, 8), F(0), F(5, 4)), (0, -1, 0), -9),
            ((F(5, 4), F(0), F(11, 8)), (0, -1, 0), 9)}
    p13 = [f for f in features()
           if f[0][1] == 0 and f[0][0] > 1 and f[0][2] > 1]
    good &= _check(want <= set(p13),
                   "panel 13: -9 at (-1/8,-1/4), +9 at (-1/4,-1/8)")

    # 3. the 192 features are distinct, in eighths, and balanced
    feats = features()
    centres = [c for c, _n, _a in feats]
    mags = {}
    for _c, _n, a in feats:
        mags.setdefault(abs(a), [0, 0])[0 if a > 0 else 1] += 1
    good &= _check(len(feats) == 192 and len(set(centres)) == 192,
                   "192 features, all centres distinct")
    good &= _check(all(F(x).denominator in (1, 2, 4, 8)
                       for c in centres for x in c),
                   "every feature centre lies in (1/8)Z^3")
    good &= _check(sorted(mags) == list(range(1, 13))
                   and all(v == [8, 8] for v in mags.values()),
                   "magnitudes 1..12, each 8 bumps and 8 dents")

    # 4. the eight child poses are proper and partition 2P
    good &= _check(all(det(frame(p, s)) == 1 for _n, p, s, _u in CHILDREN),
                   "all eight child frames are proper rotations")
    cells = {}
    for name, p, s, u in CHILDREN:
        for cell in pose_cells(frame(p, s), u):
            cells.setdefault(cell, []).append(name)
    two_p = {c for c in itertools.product(range(4), repeat=3)
             if not (c[0] >= 2 and c[1] >= 2 and c[2] >= 2)}
    good &= _check(len(cells) == 56
                   and all(len(v) == 1 for v in cells.values())
                   and set(cells) == two_p,
                   "the 56 child cells partition 2P exactly")

    # 5. every coincident feature site inside the dissection mates
    #    bump-to-dent at equal magnitude
    world = {}
    for _name, p, s, u in CHILDREN:
        G = frame(p, s)
        for c, n, a in feats:
            wc = tuple(mat_apply(G, c)[i] + u[i] for i in range(3))
            world.setdefault(wc, []).append((mat_apply(G, n), a))
    shared = [v for v in world.values() if len(v) > 1]
    paired = [v for v in shared if len(v) == 2
              and tuple(-x for x in v[0][0]) == v[1][0]
              and v[0][1] == -v[1][1]]
    good &= _check(len(shared) == 384 and len(paired) == 384,
                   f"{len(paired)}/{len(shared)} internal feature sites "
                   "mate, none with a third")

    # 6. the tile mesh closes, with the published counts and volume
    for label, eta, h in (("exact", ETA_TRUE, HEIGHT_TRUE),
                          ("shown", ETA_SHOWN, 80 * HEIGHT_TRUE)):
        verts, faces = tile_mesh(eta, h)
        tris = triangulate(faces)
        edges = {}
        for f in tris:
            for k in range(3):
                e = (f[k], f[(k + 1) % 3])
                edges[tuple(sorted(e))] = edges.get(tuple(sorted(e)), 0) + 1
        V, E, Fc = len(verts), len(edges), len(tris)
        vol = mesh_volume(verts, faces)
        good &= _check(
            (V, E, Fc) == (2138, 6408, 4272)
            and all(c == 2 for c in edges.values())
            and V - E + Fc == 2 and vol == 7,
            f"tile ({label}): V={V} E={E} F={Fc} chi={V - E + Fc} "
            f"volume={vol}")

    # 7. the bare carrier tile
    bv, bf = bare_tile_mesh()
    good &= _check(len(bf) == 24 and mesh_volume(bv, bf) == 7,
                   f"bare tile: {len(bf)} panels, volume "
                   f"{mesh_volume(bv, bf)}")

    # 8. a depth-2 patch is 64 registered chairs filling 4P
    poses = patch(2)
    pcells = {}
    for G, t, _g in poses:
        for cell in pose_cells(G, t):
            pcells.setdefault(cell, 0)
            pcells[cell] += 1
    four_p = {c for c in itertools.product(range(8), repeat=3)
              if not (c[0] >= 4 and c[1] >= 4 and c[2] >= 4)}
    good &= _check(len(poses) == 64
                   and all(det(G) == 1 for G, _t, _g in poses)
                   and all(isinstance(x, int) for _G, t, _g in poses
                           for x in t)
                   and set(pcells) == four_p
                   and all(v == 1 for v in pcells.values()),
                   f"depth-2 patch: {len(poses)} chairs, "
                   f"{len(pcells)} disjoint cells = 4P")
    groups = {}
    for _G, _t, g in poses:
        groups[g] = groups.get(g, 0) + 1
    good &= _check(sorted(groups) == list(range(8))
                   and all(v == 8 for v in groups.values()),
                   "the 64 chairs split into eight supertiles of eight")

    # 9. every contact in the patch is one of the 44 legal poses
    atlas = {(frame(p, s), off) for p, s, off in ATLAS44}
    seen = {(G, t) for _i, _j, G, t in contacts(poses)}
    good &= _check(seen <= atlas,
                   f"all {len(seen)} distinct contacts of the patch lie "
                   f"in the 44-contact atlas")

    # 10. the arrow markings: one special vertex per panel, one arrow
    #     of each colour at each of the eight special vertices
    owners = {}
    for key in PANELS:
        pid, _co = PANELS[key]
        v, _duv = panel_home(key)
        owners.setdefault(v, []).append(PANEL_COLOR[pid])
    good &= _check(len(owners) == 8
                   and all(sorted(v) == [0, 1, 2] for v in owners.values()),
                   "each of the 8 special vertices carries one blue, "
                   "one green and one red arrow")

    # 11. over the atlas, mated panels point their arrows at the same
    #     vertex of the shared square and carry compatible colours
    def posed_panels(G, t):
        out = []
        for key in PANELS:
            ax, sg, plane, u0, v0 = key
            pid, _co = PANELS[key]
            o0, o1 = (i for i in range(3) if i != ax)
            ctr = [None, None, None]
            ctr[ax] = F(plane)
            ctr[o0] = u0
            ctr[o1] = v0
            n = [0, 0, 0]
            n[ax] = sg
            home, _duv = panel_home(key)
            out.append((
                tuple(mat_apply(G, tuple(ctr))[i] + t[i] for i in range(3)),
                mat_apply(G, tuple(n)),
                pid,
                tuple(mat_apply(G, home)[i] + t[i] for i in range(3))))
        return out

    ident = frame((0, 1, 2), (1, 1, 1))
    site = {(wc, wn): (pid, wh)
            for wc, wn, pid, wh in posed_panels(ident, (0, 0, 0))}
    # blue meets blue; green meets red
    legal_pair = {(0, 0), (1, 2), (2, 1)}

    def arrows_agree(G, t):
        """True when every panel this pose mates carries a matching
        arrow.  Returns (ok, how many panels were mated)."""
        met = 0
        for wc, wn, pid, wh in posed_panels(G, t):
            hit = site.get((wc, tuple(-x for x in wn)))
            if hit is None:
                continue
            met += 1
            rid, rh = hit
            if rh != wh or (PANEL_COLOR[rid], PANEL_COLOR[pid]) not in legal_pair:
                return False, met
        return True, met

    mated = agree = 0
    for pp, ss, off in ATLAS44:
        ok, met = arrows_agree(frame(pp, ss), off)
        mated += met
        agree += met if ok else 0
    good &= _check(mated == 135 and agree == 135,
                   f"{agree}/{mated} mated panels across the atlas agree "
                   "on arrow vertex and colour")

    # 12. the arrow rule is not merely necessary: over the same 2,388
    #     candidate poses the paper enumerates, it admits exactly the
    #     44-contact atlas once reflected copies are excluded.  (With
    #     reflections it admits 60; the extra 16 are all improper, and
    #     a physical tile cannot be reflected.)
    root_cells = set(pose_cells(ident, (0, 0, 0)))
    shell = set()
    for a in root_cells:
        for ax in range(3):
            for sg in (-1, 1):
                b = list(a)
                b[ax] += sg
                if tuple(b) not in root_cells:
                    shell.add(tuple(b))
    candidates = set()
    all_frames = [frame(pp, ss) for pp in itertools.permutations(range(3))
                  for ss in itertools.product((1, -1), repeat=3)]
    for M in all_frames:
        own = pose_cells(M, (0, 0, 0))
        for cell in shell:
            for o in own:
                t = tuple(cell[i] - o[i] for i in range(3))
                cells = set(pose_cells(M, t))
                if cells & root_cells:
                    continue
                if any(sum(abs(x[i] - y[i]) for i in range(3)) == 1
                       for x in cells for y in root_cells):
                    candidates.add((M, t))
    good &= _check(len(candidates) == 2388,
                   f"{len(candidates)} candidate touching poses in the 48 "
                   "signed frames (the paper's 2,388)")
    admitted = {(M, t) for M, t in candidates if arrows_agree(M, t)[0]}
    proper = {(M, t) for M, t in admitted if det(M) == 1}
    atlas = {(frame(pp, ss), off) for pp, ss, off in ATLAS44}
    good &= _check(proper == atlas and len(admitted) == 60
                   and all(det(M) == -1 for M, t in admitted - atlas),
                   f"arrow rule admits {len(admitted)} poses, the {len(proper)} "
                   "proper ones exactly the 44-contact atlas")

    # 13. the arrow tile closes, with the volume its slabs imply
    av, af, acol = arrow_tile_mesh()
    aedges = {}
    for f in af:
        for i in range(len(f)):
            k = tuple(sorted((f[i], f[(i + 1) % len(f)])))
            aedges[k] = aedges.get(k, 0) + 1
    want_vol = 7 + 24 * arrow_area() * (_A_RISE - _A_SINK)
    got_vol = float(mesh_volume(av, af))
    good &= _check(len(af) == 384
                   and sum(1 for c in acol if c is not None) == 360
                   and all(n == 2 for n in aedges.values())
                   and abs(got_vol - want_vol) < 1e-9,
                   f"arrow tile: {len(af)} faces, closed, volume "
                   f"{got_vol:.6f} = 7 + 24 slabs")

    # 14. the exaggerated preset stays clear of the gap it is drawn in
    good &= _check(MIN_CONTACT_SPAN == 1,
                   f"tightest legal contact spans {MIN_CONTACT_SPAN} "
                   "(a notch panel against an outer one)")
    for gap, relief in ((0.92, 30.0), (0.95, 18.0), (0.97, 10.0)):
        good &= _check(relief <= max_relief(gap),
                       f"gap {gap}: relief {relief} within the "
                       f"bound {max_relief(gap):.1f}")

    print("chair44 standalone tests "
          + ("passed" if good else "FAILED"))
    if not good:
        raise AssertionError("chair44 self-test failed")
