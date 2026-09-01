# The classical crease patterns, parametrically.
#
# Part of the Math Art crease engine (`math_art/crease/`).  Python and
# numpy only -- no `bpy`.
#
# WHY THESE EXIST WHEN THE ADD-ON DOES NOT DESIGN CREASE PATTERNS.  The
# scope rule is about *substitutability*: a named model or a published
# tessellation can be downloaded, so it is imported, not generated.  The
# patterns here are the opposite case -- folk mathematics with no author
# to credit and no copyright to infringe, the origami equivalent of
# shipping a cube rather than making you import one.  Without them the
# add-on can open every crease pattern in the world and produce none,
# which is not a usable tool.
#
# All of them return a `fold_io.Frame` in the FLAT state: 2-D
# coordinates, assignments, and (where the folded state is known in
# closed form) a fold angle per crease.  Folding is `rigid.py`'s job.
#
# THE MIURA PARAMETERISATION, since everything else here is easier.  The
# flat pattern is a tessellation of congruent parallelograms with sides
# `a` (horizontal) and `b` (slanted), acute angle `alpha`:
#
#     h = b sin(alpha)        row height
#     s = b cos(alpha)        horizontal shift between successive rows
#     v[i][j] = (j*a + (i mod 2)*s,  i*h)
#
# so the horizontal creases are straight lines and the vertical ones
# zigzag.  At an interior vertex the four sectors come out
# theta, theta, pi-theta, pi-theta with theta = atan2(h, s), whose
# alternating sum vanishes for EVERY alpha -- which is exactly why the
# Miura family has a free panel angle.
#
# This flat state agrees with Schenk and Guest's folded-state formulas
# evaluated at their theta = 0, which is what makes their closed form
# usable as an oracle for the solver (see `rigid.py`).
#
# References:
#   K. Miura, "Method of Packaging and Deployment of Large Membranes in
#       Space," Inst. of Space and Astronautical Science report 618,
#       1985 -- the Miura-ori.
#   M. Schenk, S. D. Guest, "Geometry of Miura-folded metamaterials,"
#       PNAS 110(9), 2013 -- the closed-form folded state.
#   Y. Chen, H. Feng, J. Ma, R. Peng, Z. You, "Symmetric waterbomb
#       origami," Proc. R. Soc. A 472, 2016 -- the waterbomb tessellation.
#   E. D. Demaine, M. L. Demaine, V. Hart, G. N. Price, T. Tachi,
#       "(Non)existence of Pleated Folds: How Paper Folds Between
#       Creases," Graphs and Combinatorics 27(3), 2011 -- the pleated
#       hypar has no proper folding with planar facets, which is why the
#       HYPAR pattern here is emitted but flagged.
#   T. C. Hull, "Origametry" (Cambridge, 2020), ch. 5-6 -- the local
#       conditions every pattern here is checked against.

import collections

import numpy as np

from .fold_io import (ASSIGNMENTS, BOUNDARY, MOUNTAIN, UNASSIGNED, VALLEY,
                      Frame)

PATTERNS = ("MIURA", "ACCORDION", "WATERBOMB", "YOSHIMURA", "HYPAR",
            "MONKEY", "KRESLING", "KRESZIG", "RESCH",
            "TWIST")


class _Builder:
    """Accumulates vertices and edges with de-duplicated coordinates."""

    def __init__(self, tol=1e-9):
        self._key = {}
        self.verts = []
        self.edges = []
        self.assign = []
        self.tol = tol

    def v(self, x, y):
        k = (round(float(x) / self.tol), round(float(y) / self.tol))
        if k not in self._key:
            self._key[k] = len(self.verts)
            self.verts.append((float(x), float(y)))
        return self._key[k]

    def e(self, i, j, kind):
        """Add an edge; return True if it was new (so a seed lines up)."""
        if i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        for n, (p, q) in enumerate(self.edges):
            if (p, q) == (a, b):
                # A boundary claim wins over an interior one: the rim of
                # the sheet is the rim however many cells touch it.
                if kind == BOUNDARY:
                    self.assign[n] = BOUNDARY
                return False
        self.edges.append((a, b))
        self.assign.append(kind)
        return True

    def frame(self, title, angles=None):
        fr = Frame(
            verts=np.array(self.verts, dtype=float),
            edges=np.array(self.edges, dtype=np.int64).reshape(-1, 2),
            assignment=np.array(self.assign, dtype="<U1"),
            fold_angle=None if angles is None else np.asarray(angles, float),
            faces=None,
            face_orders=None,
            meta={"frame_title": title},
        )
        return fr


def miura(rows=4, cols=6, panel_a=1.0, panel_b=1.0, alpha=np.deg2rad(60.0)):
    """The Miura-ori: a rigid-foldable, flat-foldable parallelogram mesh.

    `rows` and `cols` count PANELS, so the vertex grid is one larger in
    each direction.  `alpha` is the parallelogram's acute angle; the
    pattern is flat-foldable for any value in (0, pi/2).
    """
    if not (0.0 < alpha < np.pi / 2):
        raise ValueError("miura: alpha must lie strictly in (0, pi/2)")
    h = panel_b * np.sin(alpha)
    s = panel_b * np.cos(alpha)

    B = _Builder()
    idx = [[B.v(j * panel_a + (i % 2) * s, i * h) for j in range(cols + 1)]
           for i in range(rows + 1)]

    # THE ASSIGNMENT, read off the closed-form folded state rather than
    # guessed.  Evaluating Schenk and Guest's p(i, j) and measuring the
    # dihedral angle across each crease gives:
    #
    #   straight row lines   MOUNTAIN where (i + j) is ODD -- so the sign
    #                        flips ALONG a row, segment by segment, not
    #                        merely from one row to the next
    #   zigzag columns       VALLEY where j is odd, constant down the
    #                        whole zigzag
    #
    # Maekawa then comes out at every interior vertex: the two row-line
    # segments meeting there have consecutive (i + j) and so disagree
    # (1-1), while the two zigzag segments share a column and so agree
    # (2 of a kind), giving 3-1.
    #
    # This matters beyond validity.  An assignment can satisfy Maekawa
    # and Kawasaki everywhere and still not be the Miura: making the row
    # lines depend on i alone passes both checks, but its infinitesimal
    # fold mode has NO support on the zigzag creases, so a solver started
    # from flat slides onto a degenerate branch -- every straight line
    # folding like an accordion while the zigzags stay dead flat.  The
    # local conditions are necessary, not sufficient, and this is what
    # that costs in practice.
    #
    # THE ROW-LINE PARITY WAS ONCE THE OTHER WAY ROUND, and every check
    # then in place passed anyway, which is why `_selftest` now measures
    # the labels against the closed form directly.  Flipping that one
    # family preserves the 1-1 at each vertex, so Maekawa still held;
    # Kawasaki never looks at assignments at all; and the folded geometry
    # still matched Schenk and Guest because the solver's corrector pulled
    # it back onto the true branch despite being seeded the wrong way
    # along the row creases.  The only visible symptom was in the
    # viewport -- valleys drawn in mountain red and vice versa, on the
    # shallower of the two families.
    # THE SEED, and why the assignment alone is not enough.  A crease
    # pattern's mountain/valley labels give the SIGN of each fold angle
    # but say nothing about relative MAGNITUDE, and the two Miura crease
    # families do not fold at the same rate.  Measuring the closed-form
    # folded state as the fold parameter goes to zero gives
    #
    #     |row-line angle| / |zigzag angle|  ->  cos(alpha)
    #
    # (0.5000 at alpha = 60 degrees, 0.7071 at 45).  The solver needs
    # that ratio to leave the flat state along the Miura mode instead of
    # the accordion one, so it is recorded here as a seed DIRECTION --
    # relative magnitudes with the right signs, not actual angles.
    seed = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            # straight row line: mountain where (i + j) is odd
            if j < cols:
                on_rim = i in (0, rows)
                kind = BOUNDARY if on_rim else (
                    MOUNTAIN if (i + j) % 2 == 1 else VALLEY)
                if B.e(idx[i][j], idx[i][j + 1], kind):
                    seed.append(0.0 if on_rim else
                                (-1.0 if kind == MOUNTAIN else 1.0)
                                * np.cos(alpha))
            # zigzag column: depends on j alone
            if i < rows:
                on_rim = j in (0, cols)
                kind = BOUNDARY if on_rim else (
                    VALLEY if j % 2 else MOUNTAIN)
                if B.e(idx[i][j], idx[i + 1][j], kind):
                    seed.append(0.0 if on_rim else
                                (-1.0 if kind == MOUNTAIN else 1.0))
    fr = B.frame("Miura-ori")
    fr.meta["fold_seed"] = seed
    return fr


def accordion(count=8, length=2.0, spacing=0.25):
    """A simple parallel pleat: the degenerate case, and a useful probe.

    Every crease is parallel, so there are no interior vertices at all
    and the flat-foldability conditions are vacuous.  That makes it the
    right first test for anything that folds: a solver that cannot fold
    an accordion cannot fold anything.
    """
    B = _Builder()
    w = count * spacing
    for k in range(count + 1):
        x = k * spacing - 0.5 * w
        lo, hi = B.v(x, -0.5 * length), B.v(x, 0.5 * length)
        kind = BOUNDARY if k in (0, count) else (
            MOUNTAIN if k % 2 else VALLEY)
        B.e(lo, hi, kind)
    for y in (-0.5 * length, 0.5 * length):
        for k in range(count):
            B.e(B.v(k * spacing - 0.5 * w, y),
                B.v((k + 1) * spacing - 0.5 * w, y), BOUNDARY)
    return B.frame("Accordion Pleat")


def waterbomb(rows=3, cols=4, cell=1.0):
    """The waterbomb ("magic ball") tessellation.

    Each cell is a square split by both diagonals and by its horizontal
    mid-line, giving degree-6 interior vertices.  Chen et al. analyse the
    symmetric case and its folded tube; here only the flat pattern is
    produced.
    """
    # THE ASSIGNMENT, and why it is not symmetric.  Three kinds of
    # interior vertex have to satisfy Maekawa at once, and they pull in
    # different directions:
    #
    #   cell centre  degree 6 -- four diagonals and two mid-line halves.
    #                Chen et al. call this the six-crease base and record
    #                it as TWO mountain, FOUR valley, so the diagonals are
    #                valley and the mid-line is mountain.
    #   edge midpoint  degree 4 -- the two mid-line halves (both mountain,
    #                from the rule above) plus the vertical above and
    #                below.  Two mountains already, so the verticals must
    #                split 1-1 to reach 3-1.
    #   grid corner  degree 8 -- four diagonals (all valley), two
    #                horizontals, and the vertical above and below.  With
    #                the verticals split, horizontals both mountain gives
    #                3 mountain against 5 valley, and |M - V| = 2.
    #
    # So the verticals alternate by HALF-ROW: the lower half of a cell's
    # side is mountain and the upper half valley.  Making both halves
    # alike -- the obvious first guess -- leaves every edge midpoint at
    # 2-2 and every corner at 4-4, which is what the earlier version of
    # this function did and what the self-test below now catches.
    B = _Builder()
    half = 0.5 * cell
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell, r * cell
            x1, y1 = x0 + cell, y0 + cell
            xm, ym = x0 + half, y0 + half
            a, b = B.v(x0, y0), B.v(x1, y0)
            d, e = B.v(x0, y1), B.v(x1, y1)
            ml, mr = B.v(x0, ym), B.v(x1, ym)
            ctr = B.v(xm, ym)

            # horizontal grid lines: mountain
            B.e(a, b, BOUNDARY if r == 0 else MOUNTAIN)
            B.e(d, e, BOUNDARY if r == rows - 1 else MOUNTAIN)
            # vertical sides, split at the midpoint: lower mountain,
            # upper valley, so the two halves meeting at a grid corner
            # always disagree
            B.e(a, ml, BOUNDARY if c == 0 else MOUNTAIN)
            B.e(ml, d, BOUNDARY if c == 0 else VALLEY)
            B.e(b, mr, BOUNDARY if c == cols - 1 else MOUNTAIN)
            B.e(mr, e, BOUNDARY if c == cols - 1 else VALLEY)
            # the horizontal mid-line, and the four half-diagonals
            B.e(ml, ctr, MOUNTAIN)
            B.e(ctr, mr, MOUNTAIN)
            for corner in (a, b, d, e):
                B.e(corner, ctr, VALLEY)
    return B.frame("Waterbomb Tessellation")


def yoshimura(rows=4, cols=6, cell=1.0, height=0.8):
    """The Yoshimura (diamond) pattern: the buckle pattern of a cylinder.

    A triangulated grid whose rows are offset by half a cell, so every
    interior vertex has degree 6.  Widely used for deployable tubes.
    """
    B = _Builder()
    idx = []
    for i in range(rows + 1):
        off = 0.5 * cell * (i % 2)
        idx.append([B.v(j * cell + off, i * height) for j in range(cols + 1)])

    for i in range(rows + 1):
        for j in range(cols + 1):
            if j < cols:
                B.e(idx[i][j], idx[i][j + 1],
                    BOUNDARY if i in (0, rows) else VALLEY)
            if i < rows:
                # two slanted creases per cell make the diamonds
                B.e(idx[i][j], idx[i + 1][j],
                    BOUNDARY if j in (0, cols) else MOUNTAIN)
                # THE SECOND DIAGONAL, and which neighbour it may reach.
                # Row i sits at x = j*cell + 0.5*cell*(i % 2), so the only
                # row-(i+1) vertices within half a cell of (i, j) are
                # (i+1, j) -- the edge above -- and (i+1, j-1) when i is
                # EVEN, (i+1, j+1) when i is ODD.  These two branches were
                # once the other way round, which reaches a vertex three
                # half-cells away: still a triangulation, still crossing
                # nothing, but slivers instead of diamonds.
                if (i % 2) and j < cols:
                    B.e(idx[i][j], idx[i + 1][j + 1], MOUNTAIN)
                elif not (i % 2) and j < cols:
                    B.e(idx[i][j + 1], idx[i + 1][j], MOUNTAIN)
    return B.frame("Yoshimura Diamond")


def hypar(rings=6, sides=4, radius=1.0):
    """Concentric-polygon pleats plus diagonals: the pleated hypar.

    Emitted with a warning in its metadata, because the straight-crease
    hypar provably has NO proper folding with planar facets (Demaine,
    Demaine, Hart, Price and Tachi, 2011, Corollary 14).  Real ones bend
    between the creases.  The same paper's Theorem 19 does construct
    proper foldings of TRIANGULATED hypars, which is what a solver that
    triangulates will actually fold -- faceted, not smooth.
    """
    B = _Builder()
    cen = B.v(0.0, 0.0)
    ring_idx = []
    for k in range(1, rings + 1):
        r = radius * k / rings
        ring_idx.append([B.v(r * np.cos(2 * np.pi * t / sides + np.pi / 4),
                             r * np.sin(2 * np.pi * t / sides + np.pi / 4))
                         for t in range(sides)])

    for k, ring in enumerate(ring_idx):
        outer = (k == rings - 1)
        for t in range(sides):
            u = ring[t]
            w = ring[(t + 1) % sides]
            B.e(u, w, BOUNDARY if outer else
                (MOUNTAIN if k % 2 else VALLEY))
            # THE DIAGONALS ALTERNATE ALONG THE RADIUS, not by sector.
            #
            # This shipped the other way round -- constant along each
            # radial line, alternating from one sector to the next -- and
            # the result was not a hypar: the concentric pleats folded by
            # exactly 0.00 degrees for the square case, so only the
            # radial creases moved and the sheet coned instead of
            # saddling.
            #
            # The correct rule was read off an independent reference,
            # Origami Simulator's own `assets/Origami/hypar.svg`.  Its
            # mountain and valley strokes carry equal counts of
            # 0/45/90/135-degree segments, and the diagonal segments run
            # M at r=81, V at 90, M at 99, V at 108 ... -- alternating
            # with the RADIUS, at the same pitch as the concentric
            # polygons they cross.  With that assignment the pleats fold
            # and the rim alternates up and down, which is the saddle.
            lower = cen if k == 0 else ring_idx[k - 1][t]
            B.e(lower, u, MOUNTAIN if k % 2 else VALLEY)
            # THE CELLS MUST BE TRIANGLES, or this does not fold at all.
            #
            # Between two rings each cell is a trapezoid, and a rigid
            # solver holds a quad panel flat -- so with quads the pleats
            # have nothing to bend with and the sheet merely cones,
            # every ring crease sitting at almost zero while the whole
            # thing stays a shallow dome.  That is the state this
            # shipped in.
            #
            # It is also exactly what the theory predicts: the pleated
            # hypar has NO proper folding with planar facets (Demaine,
            # Demaine, Hart, Price and Tachi 2011, Corollary 14), and
            # the same paper's Theorem 19 constructs proper foldings of
            # the TRIANGULATED hypar.  So the triangulation is not a
            # meshing convenience, it is the thing that makes the
            # object foldable, and the header has said so all along.
            #
            # The diagonal is UNASSIGNED, not flat: it is not a designed
            # crease -- real paper takes this up by bending smoothly --
            # so the solver must be free to choose its angle.  Marking
            # it F would freeze it and put us back where we started.
            if k:
                B.e(ring_idx[k - 1][(t + 1) % sides], u, UNASSIGNED)
    return B.frame("Pleated Hypar")


# --------------------------------------------------------------------
# EVERY PATTERN HERE HAS AN ORACLE.  That is a rule, not a summary.
# --------------------------------------------------------------------
# The Miura is checked against Schenk and Guest's closed form; the
# Yoshimura must curl; the hypar must saddle while its triangular case
# stays flat; the Kresling must close, twist by rows * phi and reproduce
# Liu's worked example; the zigzag stacking must close and NOT twist;
# Resch must reproduce the two vertex figures measured off Ghassaei's
# reference file.
#
# The rule exists because this module has shipped three patterns that
# passed Maekawa and Kawasaki and were still not the pattern they were
# named for -- two Kreslings and a Resch, each assembled from a verbal
# description.  Local flat-foldability conditions are necessary and
# nowhere near sufficient, and a self-consistent metric will happily
# confirm a wrong construction.  Only an independent statement about the
# folded object settles it: a published closed form, a reference crease
# pattern, or a behaviour the pattern is named for.
#
# Two patterns without one were removed rather than carried: EGGBOX,
# whose positive Poisson ratio was never demonstrated, and CHEVRON,
# which was only ever a control.


def _kresling_module(n, alpha, a):
    """The deployed Kresling module: radius, cell, twist and height.

    Everything follows from the number of sides and the panel angle.
    Liu's closure condition -- that the fully folded strip closes into a
    REGULAR n-gon -- fixes the cell, and in the reference generator's own
    parameters it reduces to `b/a = sin(alpha - pi/n) / sin(pi/n)`.
    Returns None when no deployed state exists.
    """
    eta = np.pi / n
    gamma = alpha - eta                       # the valley diagonal's slope
    if gamma <= 1e-9 or alpha >= 0.5 * np.pi:
        return None
    h = a * np.sin(gamma) * np.sin(alpha) / np.sin(eta)
    off = h / np.tan(gamma) - a
    th = 2.0 * eta
    R = a / (2.0 * np.sin(eta))

    # The module has both rings regular and coaxial, so only the twist
    # `phi` and the height `H` are free.  The mountain side joins bottom
    # j to top j and the valley diagonal joins bottom j to top j+1:
    #
    #     side^2 = 2R^2 (1 - cos phi)        + H^2
    #     diag^2 = 2R^2 (1 - cos(theta+phi)) + H^2
    #
    # Subtracting kills H and leaves a single sine, whose TWO roots are
    # the Kresling's bistability -- one is the flat-packed diaphragm, the
    # other the deployed tube.  Take the taller.
    val = np.sin(eta) * (1.0 + 2.0 * off / a)
    if abs(val) > 1.0:
        return None
    best = None
    for phi in (np.arcsin(val) - eta, np.pi - np.arcsin(val) - eta):
        H2 = off * off + h * h - 2.0 * R * R * (1.0 - np.cos(phi))
        if H2 > 1e-9 and (best is None or H2 > best[1] ** 2):
            best = (phi, np.sqrt(H2))
    return None if best is None else (R, h, off, best[0], best[1])


def _kresling_alpha(n, alpha, a, floor=0.05):
    """Clamp the panel angle into the range that actually deploys.

    Below a lower bound that depends on `n`, the two roots above collide
    and the only state is the flat diaphragm -- the pattern is a valid
    sheet that can never stand up.  Rather than raise on a slider value,
    find the bound by bisection and clamp to just inside it, so every
    setting of Panel Angle gives a tube.
    """
    top = 0.5 * np.pi - 1e-3
    alpha = float(min(max(alpha, np.pi / n + 1e-3), top))
    ok = _kresling_module(n, alpha, a)
    if ok is not None and ok[4] > floor * a:
        return alpha
    lo, hi = np.pi / n, top
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m = _kresling_module(n, mid, a)
        if m is not None and m[4] > floor * a:
            hi = mid
        else:
            lo = mid
    return min(top, hi + 1e-3)


def kresling(rows=3, cols=6, alpha=np.deg2rad(72.0), radius=1.0,
             stacking='UNIFORM'):
    """The Kresling pattern: a tube that TWISTS as it deploys.

    A strip of inclined parallelograms, each cut by its LONG diagonal.
    Roll the strip up and join the ends and it becomes a cylinder whose
    top ring is rotated relative to its bottom -- the twist-buckling
    pattern a thin-walled tube adopts by itself under torsion, which is
    how Biruta Kresling found it rather than designed it.

    `cols` is the number of sides around the tube, `rows` the number of
    stacked bands, and `alpha` the panel angle: the lean of the
    parallelogram's side away from the horizontal.

    `stacking` is how the bands are piled up, and Kresling gives both:
    a belt "may operate alone or be multiple, either in left-hand or
    right-hand version, OR BE INCLINED ALTERNATELY TO LEFT AND RIGHT".

      UNIFORM    every band leans the same way, so the sheet is one
                 sheared lattice with every diagonal parallel and the
                 band twists add up down the tube.  This is what the
                 twist-buckling experiment produces, and what the
                 published generators draw.
      ALTERNATE  mirror every other band.  The side creases zigzag, the
                 diagonals mirror into Vs, and the band twists cancel in
                 pairs -- so the tube pumps along its axis instead of
                 turning.  Kresling calls the result "similar to a
                 Miura-ori"; it is the chevron-looking picture that
                 usually comes up under the pattern's name.  Small `alpha` gives
    the strongly inclined, elongated cells of a deep twist; near a right
    angle the lean vanishes and the tube stops turning.

    THE CELL IS NOT FREE.  Its proportions come from the fully folded
    state, where the strip closes into a REGULAR n-gon -- the flat
    diaphragm perpendicular to the axis that Kresling describes the
    buckled cylinder collapsing into.  Writing `eta = pi/n` for the
    half-angle of that polygon and `gamma = alpha - eta` for the
    diagonal's slope, Liu's construction gives

        h = a sin(gamma) sin(alpha) / sin(eta)          cell height
        offset = h / tan(gamma) - a                     lean per band

    which in the reference generator's parameters is just
    `b/a = sin(alpha - pi/n)/sin(pi/n)`.  Choosing `a`, `b` and `alpha`
    independently -- as one naturally would -- gives a strip that folds
    but never closes; this is the one-parameter family that does.

    ASSIGNMENTS AND ANGLES ARE MEASURED ON THE DEPLOYED MODULE, not
    assigned by parity, and they come out as both sources describe: the
    parallelogram outline MOUNTAIN, the long diagonal VALLEY.  That
    agreement is the check that the geometry is right, because it is not
    something the construction was told to produce.

    WHAT THIS REPLACED, since the failure is instructive.  The first
    version guessed the layout (Yoshimura offsets, one diagonal family
    deleted) and never closed at all.  The second derived a flat sheet
    isometric to a tube, which folded beautifully to zero strain and was
    still not a Kresling: it split each cell on the SHORT diagonal, so
    the deployed module was a convex antiprism whose creases all bend the
    same way, where a Kresling tucks its diagonals in.  A pattern can be
    foldable, isometric and self-consistent and still be the wrong
    pattern; only the sources settle it.

    References:
      B. Kresling, "Natural twist buckling in shells: from the
          hawkmoth's bellows to the deployable Kresling-pattern and
          cylindrical Miura-ori," Proc. IASS-IACM 2008, Ithaca -- the
          buckling experiment, and the description of the pattern as
          "inclined and elongated parallelograms (mountain-folds),
          divided on their long diagonal by a valley-fold".
      Y. Liu, "Kresling origami structure: Geometric design principles
          and application research of foldable paper cups," CMSDA 2024,
          ACM -- the closure condition used here.
      S. D. Guest, S. Pellegrino, "The folding of triangulated
          cylinders," ASME J. Applied Mechanics 61, 1994 -- the
          kinematics of the triangulated cylinder.
      G. W. Hunt, I. Ario, "Twist buckling and the foldable cylinder: an
          exercise in origami," Int. J. Non-Linear Mechanics 40, 2004.
    """
    n = max(3, int(cols))
    a = 2.0 * radius * np.sin(np.pi / n)
    alpha = _kresling_alpha(n, float(alpha), a)
    mod = _kresling_module(n, alpha, a)
    if mod is None:                            # unreachable after clamping
        raise ValueError("kresling: no deployed module for these settings")
    R, h, off, phi, H = mod

    # ONE SIGN PER BAND: +1 leans right, -1 leans left.  The lean,
    # which diagonal is the long one, the winding and the tube's twist
    # all follow from this list, so the two stackings stay one
    # construction rather than becoming two that can drift apart.
    alt = str(stacking).upper() == 'ALTERNATE'
    sgn = [(-1.0 if (alt and i % 2) else 1.0) for i in range(rows)]
    lean = [0.0]
    for i in range(rows):
        lean.append(lean[-1] + sgn[i] * off)

    B = _Builder()
    idx = [[B.v(j * a + lean[i], i * h) for j in range(n + 1)]
           for i in range(rows + 1)]
    faces = []
    for i in range(rows + 1):
        for j in range(n + 1):
            if j < n:
                B.e(idx[i][j], idx[i][j + 1],
                    BOUNDARY if i in (0, rows) else MOUNTAIN)
            if i < rows:
                B.e(idx[i][j], idx[i + 1][j],
                    BOUNDARY if j in (0, n) else MOUNTAIN)
                if j < n:
                    # THE LONG DIAGONAL of this band's cell, which
                    # flips with the lean.  The short one gives a convex
                    # antiprism instead -- it folds perfectly well and
                    # is not this pattern.  Both windings below are
                    # counter-clockwise, which is what makes the
                    # measured dihedral signs mean anything.
                    if sgn[i] > 0:
                        B.e(idx[i][j], idx[i + 1][j + 1], VALLEY)
                        faces.append([idx[i][j], idx[i][j + 1],
                                      idx[i + 1][j + 1]])
                        faces.append([idx[i][j], idx[i + 1][j + 1],
                                      idx[i + 1][j]])
                    else:
                        B.e(idx[i][j + 1], idx[i + 1][j], VALLEY)
                        faces.append([idx[i][j], idx[i][j + 1],
                                      idx[i + 1][j]])
                        faces.append([idx[i][j + 1], idx[i + 1][j + 1],
                                      idx[i + 1][j]])

    fr = B.frame("Kresling")
    fr.faces = faces
    fr.fold_angle = np.full(fr.n_edges, np.nan)

    th = 2.0 * np.pi / n
    turn = [0.0]
    for i in range(rows):
        turn.append(turn[-1] + sgn[i] * phi)
    tube = np.zeros((fr.n_verts, 3))
    for i in range(rows + 1):
        for j in range(n + 1):
            ang = th * j + turn[i]
            tube[idx[i][j]] = (R * np.cos(ang), R * np.sin(ang), H * i)

    # MEASURED WITH THE SOLVER'S OWN FUNCTION, deliberately, rather than
    # with a private copy of the dihedral formula: the two would be free
    # to disagree by a sign, and a target that is exactly negated drives
    # the sheet the wrong way while looking entirely reasonable.
    from .compliant import CompliantFolder
    cf = CompliantFolder(fr)
    ang = cf.fold_angles(tube)
    where = {(min(int(p), int(q)), max(int(p), int(q))): k
             for k, (p, q) in enumerate(fr.edges)}
    for m, (p, q) in enumerate(cf.cr_edge):
        k = where[(min(int(p), int(q)), max(int(p), int(q)))]
        fr.fold_angle[k] = ang[m]
        if fr.assignment[k] in (MOUNTAIN, VALLEY):
            fr.assignment[k] = VALLEY if ang[m] > 0.0 else MOUNTAIN

    # The same angles steer the rigid solver.  Its continuation leaves
    # the flat state along `fold_seed`, and the flat state is a
    # bifurcation point where several branches meet -- so the relative
    # magnitudes are not a refinement, they choose the branch.
    fr.meta["fold_seed"] = np.nan_to_num(fr.fold_angle, nan=0.0)
    fr.meta["kresling"] = (R, h, off, phi, H, alpha)
    fr.meta["kresling_signs"] = tuple(sgn)
    return fr


def kresling_zigzag(rows=4, cols=6, alpha=np.deg2rad(72.0), radius=1.0):
    """Kresling with its bands inclined alternately left and right.

    The same construction as `kresling` and sharing all of its geometry
    -- see that docstring -- mirrored band by band.  What changes is the
    mechanism: the per-band twists cancel in pairs, so the tube extends
    and contracts along its axis WITHOUT turning, where the uniform
    stacking turns as it deploys.

    Kresling notes that alternating belts give "a derived pattern
    similar to a Miura-ori", and this is the zigzag picture that usually
    appears under the pattern's name -- which is why it is worth having
    both: they look nothing alike flat, and they do different things.

    References:
      B. Kresling, "Natural twist buckling in shells," Proc. IASS-IACM
          2008, Ithaca, sec. 4 -- belts "inclined alternately to left
          and right", and the cylindrical Miura-ori derived from them.
    """
    return kresling(rows=rows, cols=cols, alpha=alpha, radius=radius,
                    stacking='ALTERNATE')


#: The Resch molecule, in units of the short mountain `u`.  Lattice
#: neighbour directions are 30, 90, ... 330 degrees, and a hub points a
#: MOUNTAIN at three of them and a VALLEY at the other three.
_RESCH_DIRS = np.deg2rad(np.array([30.0, 90.0, 150.0, 210.0, 270.0,
                                   330.0]))
_RESCH_UNIT = np.stack([np.cos(_RESCH_DIRS), np.sin(_RESCH_DIRS)], axis=1)
_RESCH_MOUNT = (1, 3, 5)                       # 90, 210, 330 degrees


def square_twist(rows=1, cols=1, cell=1.0,
                 alpha=np.deg2rad(60.0)):
    """The square twist: a square that rotates as four pleats close.

    A central square with a pleat running off each of its sides.  Folding
    the pleats swings the square round and hides the surplus paper
    underneath -- the twist the whole family of twist tessellations is
    named for.

    `alpha` is the twist angle, strictly between 0 and 90 degrees; it
    sets how far the inner square is rotated against the four outer ones
    and so how far it turns when folded.  `rows` and `cols` count
    TWISTS, tiling the molecule across the sheet.

    TRANSCRIBED FROM THE PUBLISHED GENERATOR, not reconstructed.  The
    source is OrigamiSimulator's own `SquareTwist.pde`, and the result is
    checked edge for edge against the crease pattern it produces
    (`assets/Origami/singlesquaretwist.svg`): 24 segments, 8 valley, 4
    mountain, 12 boundary, all matching to 1e-3 of the sheet.  Writing
    `d = cell/sqrt(2)` for the inner square's circumradius -- so the
    inner square has the SAME side as the four outer ones -- its corners
    are

        P1 = d (sin(alpha - pi/4), -cos(alpha - pi/4))
        P2 = d (sin(alpha + pi/4), -cos(alpha + pi/4))

    together with -P1 and -P2, the whole cell then being turned by
    -alpha/2.  From each corner one crease leaves along an axis and one
    along the other, and the four outer squares hang off those.

    THE COLOURS IN THE SOURCE ARE THE TRAP.  `SquareTwist.pde` defines
    `setMountainColor()` to paint BLUE and `setValleyColor()` to paint
    RED -- the reverse of this project's convention and of the usual one.
    Following the FUNCTION NAMES inverts every crease in the pattern;
    following the COLOURS, which is what the published figure shows,
    gives what is built here: the inner square all VALLEY, and each of
    its corners sending out one further valley and one mountain.  That
    is 3 valleys against 1 mountain at every vertex, so Maekawa holds
    with room to spare, and Kawasaki holds at all four.

    A PREVIOUS VERSION OF THIS WAS THE OTHER SQUARE TWIST.  It came from
    edemaine/fold's `squaretwist.fold`, whose frame is titled "Rigidly
    folded square twist" -- a DIFFERENT and less familiar member of the
    family, whose inner square carries two mountains and two valleys and
    whose outline is a square rather than a pinwheel.  It was faithful to
    its own reference and was not the pattern anybody means by "the
    square twist".  Matching a reference is not enough; it has to be the
    right reference.

    THE TILING.  One cell is a twelve-sided pinwheel, and translating
    it by `2*ex + (P2 - P1)` -- two square widths along, plus one side of
    the twist square -- sets it exactly against its neighbour; the
    90-degree rotation of that closes the lattice.  Inside a patch every
    rim segment gets drawn twice, once by each of the two cells meeting
    along it, and the paper is continuous there rather than creased, so
    those are dropped and only the segments seen once survive as the
    patch's outer edge.  That is why the source's tessellation draws no
    rim at all.

    References:
      A. Ghassaei, Origami Simulator,
          `CreasePatternScripts/SquareTwist/SquareTwist.pde` and
          `assets/Origami/singlesquaretwist.svg` -- the construction
          transcribed here and the figure it is checked against.
      J. L. Silverberg, J.-H. Na, A. A. Evans, B. Liu, T. C. Hull,
          C. D. Santangelo, R. J. Lang, R. C. Hayward, I. Cohen,
          "Origami structures with a critical transition to bistability
          arising from hidden degrees of freedom," Nature Materials 14,
          2015 -- the square twist's bistability.
      T. C. Hull, "Origametry" (Cambridge, 2020), ch. 5-6.
    """
    a = float(cell)
    alpha = float(np.clip(alpha, np.deg2rad(5.0), np.deg2rad(85.0)))
    rows, cols = max(1, int(rows)), max(1, int(cols))
    d = a / np.sqrt(2.0)
    P1 = np.array([d * np.sin(alpha - np.pi / 4),
                   -d * np.cos(alpha - np.pi / 4)])
    P2 = np.array([d * np.sin(alpha + np.pi / 4),
                   -d * np.cos(alpha + np.pi / 4)])
    ex = np.array([a, 0.0])
    ey = np.array([0.0, a])

    # Processing draws y DOWNWARDS and wraps the cell in rotate(-alpha/2).
    # Undo both, so the sheet comes out in ordinary maths coordinates the
    # same way up as the published figure.
    c, s = np.cos(-alpha / 2), np.sin(-alpha / 2)

    def place(p):
        x, y = float(p[0]), float(p[1])
        return np.array([c * x - s * y, -(s * x + c * y)])

    # THE TILING LATTICE, chosen by measurement.  Translating a cell by
    # 2*ex + 2*P2 and by the 90-degree rotation of that sets neighbouring
    # twists against each other so their crease ends MEET: a 3x3 patch
    # merges twelve vertices and not one vertex in it fails Maekawa.
    #
    # The obvious denser candidate does not work, which is worth
    # recording.  2*ex + (P2 - P1) has determinant equal to the cell's
    # own area, so it tiles with neither gap nor overlap -- and where the
    # crease ends meet, a valley from one cell runs head-on into a
    # mountain from the next.  A degree-2 vertex needs both creases the
    # same to fold flat, so that lattice fails Maekawa at 24 vertices
    # despite fitting perfectly.  Fitting the paper is not the same as
    # fitting the creases.
    L1 = place(2.0 * ex + 2.0 * P2)
    L2 = np.array([-L1[1], L1[0]])

    B = _Builder(tol=1e-7)
    RIM = ((P1 - ex - ey, P1 - ex), (P1 - ex, -P2 - ex),
           (-P2 - ex, -P2 - ex + ey),
           (P1 - ey, P1 - ex - ey), (P1 - ey, P2 - ey),
           (P2 - ey, P2 + ex - ey),
           (P2 + ex - ey, P2 + ex), (P2 + ex, -P1 + ex),
           (-P1 + ex, -P1 + ex + ey),
           (-P1 + ex + ey, -P1 + ey), (-P1 + ey, -P2 + ey),
           (-P2 + ey, -P2 - ex + ey))

    # The rim is the outline of ONE cell, so inside a patch every rim
    # segment is drawn twice -- once by each of the two cells that meet
    # along it -- and the paper is continuous across it.  Count them
    # first and keep only the ones seen once; those are the patch's real
    # outer edge.  (The tessellation in the source draws no rim at all,
    # for exactly this reason.)
    seen = {}
    for i in range(rows):
        for j in range(cols):
            o = i * L1 + j * L2
            for p, q in RIM:
                u, v = place(p) + o, place(q) + o
                k = tuple(sorted((tuple(np.round(u, 6)),
                                  tuple(np.round(v, 6)))))
                seen[k] = seen.get(k, 0) + 1

    for i in range(rows):
        for j in range(cols):
            o = i * L1 + j * L2

            def v(p):
                q = place(p) + o
                return B.v(q[0], q[1])

            # The inner square: EVERY side a valley.  This is the single
            # fact that most obviously separates this pattern from the
            # rigidly foldable one, and the first thing to check against
            # a picture.
            for x, y in ((P1, P2), (P1, -P2), (-P1, P2), (-P1, -P2)):
                B.e(v(x), v(y), VALLEY)
            # One more valley and one mountain out of each corner, 3-1.
            for p, val, mnt in ((P1, -ey, -ex), (-P1, ey, ex),
                                (P2, ex, -ey), (-P2, -ex, ey)):
                B.e(v(p), v(p + val), VALLEY)
                B.e(v(p), v(p + mnt), MOUNTAIN)
            for p, q in RIM:
                pu, qu = place(p) + o, place(q) + o
                k = tuple(sorted((tuple(np.round(pu, 6)),
                                  tuple(np.round(qu, 6)))))
                if seen[k] == 1:
                    B.e(v(p), v(q), BOUNDARY)

    fr = B.frame("Square Twist")

    # TARGETS, because the default ones are nowhere near deep enough.
    # An unlabelled crease is driven to one radian, and at that depth the
    # twist square turns 5 degrees -- the pattern looks like a slightly
    # dented sheet rather than a twist.  This pattern is flat-foldable,
    # so every crease genuinely wants +-180; 170 keeps the folded state
    # off the degenerate flat packet while still turning the square
    # through about 35 degrees.  Fold Amount then scales all of them
    # together, which is what makes that slider mean something here.
    want = np.deg2rad(170.0)
    fr.fold_angle = np.array(
        [want if c == VALLEY else (-want if c == MOUNTAIN else np.nan)
         for c in fr.assignment], dtype=float)
    return fr


def resch(rows=3, cols=3, cell=1.0):
    """Ron Resch's triangular tessellation.

    A triangular grid whose every vertex is INFLATED into a small
    triangular face, with straight creases running from it to describe
    triangular faces everywhere else.  Folding tucks the surplus
    material behind the sheet and leaves a stiff faceted surface that
    takes curvature both ways, which is what made it interesting to
    architecture and later to the freeform-origami literature.

    `rows` and `cols` count HUBS -- inflated vertices -- so the sheet
    grows by one molecule per step rather than one crease.

    THE MOLECULE, read off Ghassaei's reference crease pattern rather
    than reconstructed from a description.  There are exactly two
    interior vertex figures, and between them they fix everything:

      HUB, degree 6   mountains of length u at 90, 210, 330 degrees,
                      valleys of length 2u at 30, 150, 270
      TIP, degree 6   where a hub's short mountain ends: the valley
                      carries straight on for 2u to the NEXT hub, and
                      four mountains of sqrt(3) u leave at 0, 60, 120
                      and 180 degrees from it

    So hubs sit on a triangular lattice of spacing 3u, and every lattice
    edge carries one tip -- 1u from the hub that points a mountain at
    it, 2u from the one that points a valley.  That is consistent all
    the way across precisely because each hub alternates three mountains
    and three valleys around itself, so the two ends of every edge
    always disagree about which it is.  The sqrt(3) mountains join tip
    to tip and are what triangulate the sheet; in the reference they are
    drawn as single strokes of 2 sqrt(3) that pass straight THROUGH an
    intermediate tip, which is why that file also contains a handful of
    sqrt(3) halves where a stroke happened to stop early.

    IT DOES NOT FOLD FLAT, AND THAT IS THE POINT.  A hub carries three
    mountains and three valleys, so |M - V| = 0 and Maekawa fails there
    by construction.  Resch's pattern is not a flat packet; it collapses
    to a stiff tucked sheet, and the surplus material disappearing
    behind that sheet is the whole trick.  Reporting Maekawa failures at
    every hub is therefore correct behaviour, not a defect to fix.

    WHAT THIS REPLACED.  The first version was built from a verbal
    description -- a triangular grid with a counter-rotated triangle
    inscribed in each cell -- and every one of its 31 interior vertices
    had ODD degree, which no folded state of any kind can have.  It
    curled and stretched instead of folding (41 per cent out of plane,
    2.3 per cent edge strain) because it was not a foldable pattern at
    all.  The lesson is the same one the Kresling taught twice: a
    pattern assembled from a description is a guess, and only reference
    geometry settles it.

    References:
      R. D. Resch, "The topological design of sculptural and
          architectural systems," AFIPS Conference Proceedings 42, 1973
          -- the tessellation and its architectural use.
      T. Tachi, "Designing Freeform Origami Tessellations by
          Generalizing Resch's Patterns," ASME J. Mech. Des. 135, 2013.
      E. Vouga et al. / A. Ghassaei, Origami Simulator reference crease
          patterns, `assets/Tessellations/reschTriTessellation.svg` --
          the file the molecule above was measured from.
      A. Koerner, "Computing Curved-Folded Tessellations through
          Straight-Folding Approximation," SimAUD 2015 -- Resch's
          "inflated vertices", and why a pattern whose creases all
          converge on a point instead cannot be folded with straight
          creases at all.
    """
    u = float(cell)
    e1 = 3.0 * u * _RESCH_UNIT[0]
    e2 = 3.0 * u * _RESCH_UNIT[1]

    B = _Builder(tol=1e-7)
    hub = {}
    for i in range(int(rows) + 1):
        for j in range(int(cols) + 1):
            p = i * e1 + j * e2
            hub[(i, j)] = (B.v(p[0], p[1]), p)

    # A tip belongs to the hub that points a mountain at it, so indexing
    # them by (hub, direction) rather than by position keeps the two
    # halves of each lattice edge from being built twice.
    tip = {}
    for (i, j), (hv, hp) in hub.items():
        for k in _RESCH_MOUNT:
            tp = hp + u * _RESCH_UNIT[k]
            tv = B.v(tp[0], tp[1])
            tip[(i, j, k)] = (tv, tp)
            B.e(hv, tv, MOUNTAIN)
            far = hp + 3.0 * u * _RESCH_UNIT[k]
            for kk, (fv, fp) in hub.items():
                if np.linalg.norm(fp - far) < 1e-7 * max(1.0, u):
                    B.e(tv, fv, VALLEY)
                    break

    where = {(round(float(p[0]) / u, 5), round(float(p[1]) / u, 5)): v
             for v, p in tip.values()}
    for (i, j, k), (tv, tp) in tip.items():
        for s in (-1.0, 1.0):
            d = _RESCH_DIRS[k] + s * np.pi / 6.0
            q = tp + np.sqrt(3.0) * u * np.array([np.cos(d), np.sin(d)])
            other = where.get((round(float(q[0]) / u, 5),
                               round(float(q[1]) / u, 5)))
            if other is not None:
                B.e(tv, other, MOUNTAIN)

    fr = B.frame("Resch Triangular")

    # The rim is whatever ended up with one face beside it.  Marking it
    # by geometry rather than by index keeps the patch honest at any
    # size: a crease with paper on one side only is not a crease.
    from .graph import build_faces
    fr.faces = build_faces(fr.verts, fr.edges)
    seen = {}
    for f in fr.faces:
        for t in range(len(f)):
            a, b = int(f[t]), int(f[(t + 1) % len(f)])
            seen[(min(a, b), max(a, b))] = seen.get((min(a, b), max(a, b)),
                                                    0) + 1
    for e, (a, b) in enumerate(fr.edges):
        if seen.get((min(int(a), int(b)), max(int(a), int(b))), 0) < 2:
            fr.assignment[e] = BOUNDARY
    return fr


def monkey_saddle(rings=8, radius=1.0):
    """The six-sector pleated hypar: a monkey saddle.

    Same construction as `hypar`, six sectors instead of four, and a
    genuinely different surface rather than a cosmetic variant: the
    four-sector hypar folds to an ordinary saddle, two rim corners up
    and two down alternating, while this one comes out **3-fold
    periodic**.

    A CAVEAT ON THE NAME, which is the user's and is kept, but which the
    folded result does not fully earn. A true monkey saddle
    (z = r^3 cos 3*theta) has three ups and three downs alternating
    around the rim. Measured here the rim is +0.39, -0.24, -0.29 twice
    over -- 3-fold, but two up and four down. Whether a 3-up/3-down
    branch of this pattern also exists is open: it has many degrees of
    freedom and the continuation settles on one of them. Do not describe
    the output as the textbook monkey saddle without re-measuring.

    References:
      E. D. Demaine, M. L. Demaine, A. Lubiw, "Polyhedral Sculptures
          with Hyperbolic Paraboloids," Bridges 1999 -- the generalized
          hypar crease pattern for non-square polygons, of which this is
          the hexagonal case.
    """
    return hypar(rings=rings, sides=6, radius=radius)


_MAKERS = {
    "MIURA": miura,
    "MONKEY": monkey_saddle,
    "KRESLING": kresling,
    "KRESZIG": kresling_zigzag,
    "RESCH": resch,
    "TWIST": square_twist,
    "ACCORDION": accordion,
    "WATERBOMB": waterbomb,
    "YOSHIMURA": yoshimura,
    "HYPAR": hypar,
}


def build(pattern, **kw):
    """Dispatch by name, passing only the keywords the maker accepts."""
    try:
        fn = _MAKERS[pattern]
    except KeyError:
        raise ValueError(f"unknown pattern {pattern!r}; "
                         f"expected one of {', '.join(PATTERNS)}")
    import inspect
    ok = set(inspect.signature(fn).parameters)
    return fn(**{k: v for k, v in kw.items() if k in ok})


def _selftest():
    from .graph import build_faces
    from .validate import validate

    # --- Miura: the flat-foldability conditions hold for every alpha --
    for deg in (35.0, 60.0, 75.0):
        fr = miura(rows=4, cols=4, alpha=np.deg2rad(deg))
        fr.faces = build_faces(fr.verts, fr.edges)
        assert fr.n_faces == 16, (deg, fr.n_faces)
        rep = validate(fr)
        assert rep.checked and rep.n_interior == 9, rep.summary()
        assert rep, f"alpha={deg}: {rep.summary()}"

    # --- the LABELS themselves, against the closed-form folded state ---
    #
    # Maekawa and Kawasaki above cannot do this job: the row-line parity
    # was once inverted and both still passed, because flipping a whole
    # family preserves the 1-1 split at each vertex and Kawasaki never
    # reads assignments at all.  So measure the real thing -- build
    # Schenk and Guest's folded Miura, orient each panel by its CCW
    # winding in the FLAT pattern, and ask which way the paper actually
    # goes at every crease.
    def _mv_oracle(R, C, gv, th=1.0, av=1.0, bv=1.0):
        H = av * np.sin(th) * np.sin(gv)
        ct, tg = np.cos(th), np.tan(gv)
        Sc = bv * ct * tg / np.sqrt(1 + ct * ct * tg * tg)
        L = av * np.sqrt(1 - np.sin(th) ** 2 * np.sin(gv) ** 2)
        Vv = bv / np.sqrt(1 + ct * ct * tg * tg)
        P = np.array([[i * Sc,
                       j * L + (Vv / 2) * (1 - (-1) ** i),
                       (H / 2) * (1 - (-1) ** j)]
                      for i in range(R + 1) for j in range(C + 1)])
        FL = np.array([[j * av + (i % 2) * bv * np.cos(gv), i * bv * np.sin(gv)]
                       for i in range(R + 1) for j in range(C + 1)])

        def vid(i, j):
            return i * (C + 1) + j

        quads = []
        for i in range(R):
            for j in range(C):
                q = [vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)]
                p = FL[q]
                cz = ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) -
                      (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]))
                quads.append(q[::-1] if cz < 0 else q)

        ef = {}
        for fi, q in enumerate(quads):
            for t in range(4):
                a2, b2 = q[t], q[(t + 1) % 4]
                ef.setdefault((min(a2, b2), max(a2, b2)), []).append(fi)

        def nrm(q):
            p = P[q]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            return n / np.linalg.norm(n)

        out = {}
        for key, fl in ef.items():
            if len(fl) != 2:
                out[key] = BOUNDARY
                continue
            nv = nrm(quads[fl[0]]) + nrm(quads[fl[1]])
            nv = nv / np.linalg.norm(nv)
            far = [w for q in (quads[fl[0]], quads[fl[1]])
                   for w in q if w not in key]
            d = P[far].mean(0) - P[list(key)].mean(0)
            out[key] = VALLEY if float(d @ nv) > 0 else MOUNTAIN
        return out

    for (R, C, gd) in ((4, 6, 60.0), (3, 4, 45.0), (4, 4, 75.0)):
        gv = np.deg2rad(gd)
        fr = miura(rows=R, cols=C, alpha=gv)
        truth = _mv_oracle(R, C, gv)
        # map builder vertices back to (i, j) by position
        pos = {}
        for i in range(R + 1):
            for j in range(C + 1):
                w = (j * 1.0 + (i % 2) * np.cos(gv), i * np.sin(gv))
                hit = np.nonzero((np.abs(fr.verts[:, 0] - w[0]) < 1e-9) &
                                 (np.abs(fr.verts[:, 1] - w[1]) < 1e-9))[0]
                pos[int(hit[0])] = i * (C + 1) + j
        bad = []
        for k, (a2, b2) in enumerate(fr.edges):
            got = str(fr.assignment[k])
            if got == BOUNDARY:
                continue
            key = (min(pos[int(a2)], pos[int(b2)]),
                   max(pos[int(a2)], pos[int(b2)]))
            want = truth.get(key)
            if want is not None and want != BOUNDARY and want != got:
                bad.append((k, got, want))
        assert not bad, (
            f"Miura {R}x{C} alpha={gd}: {len(bad)} crease(s) labelled the "
            f"wrong way round vs the closed form, e.g. {bad[:5]}")

    # the flat pattern agrees with Schenk & Guest at their theta = 0
    a, b, g = 1.3, 0.9, np.deg2rad(55.0)
    fr = miura(rows=2, cols=2, panel_a=a, panel_b=b, alpha=g)
    S, L, V = b * np.sin(g), a, b * np.cos(g)
    for i in range(3):
        for j in range(3):
            want = (j * L + (V / 2) * (1 - (-1) ** i), i * S)
            got = fr.verts[[k for k, p in enumerate(fr.verts)
                            if abs(p[0] - want[0]) < 1e-9
                            and abs(p[1] - want[1]) < 1e-9]]
            assert len(got) == 1, f"no Schenk vertex at i={i} j={j}"

    # --- accordion: no interior vertices, so nothing to violate ------
    ac = accordion(count=6)
    ac.faces = build_faces(ac.verts, ac.edges)
    assert ac.n_faces == 6
    rep = validate(ac)
    assert rep.n_interior == 0 and rep

    # --- waterbomb and yoshimura: faces recover, degrees are even ----
    # Full validity, not just "no odd degrees": three kinds of interior
    # vertex (centre, edge midpoint, grid corner) must satisfy Maekawa
    # simultaneously, and a lenient check here let a 2-2 assignment ship.
    for rc in ((2, 2), (3, 4), (2, 5)):
        wb = waterbomb(rows=rc[0], cols=rc[1])
        wb.faces = build_faces(wb.verts, wb.edges)
        assert wb.n_faces == rc[0] * rc[1] * 6, wb.n_faces
        rep = validate(wb)
        assert rep.checked and rep.n_interior > 0
        assert rep, f"waterbomb {rc}: {rep.summary()}"
    # and the six-crease base really is 2 mountain / 4 valley at a centre
    wb = waterbomb(rows=2, cols=2)
    centre = [i for i, p in enumerate(wb.verts)
              if abs(p[0] - 0.5) < 1e-9 and abs(p[1] - 0.5) < 1e-9][0]
    inc = [k for k, (a, b) in enumerate(wb.edges) if centre in (a, b)]
    kinds = [str(wb.assignment[k]) for k in inc]
    assert len(inc) == 6, inc
    assert kinds.count("M") == 2 and kinds.count("V") == 4, kinds

    # --- yoshimura: the diamond grid's SHAPE, not just its validity ---
    #
    # `n_faces > 0` and Maekawa/Kawasaki were the whole test here, and
    # all three passed while the two second-diagonal branches were
    # swapped -- which reaches a vertex three half-cells away instead of
    # one.  That still triangulates the sheet, still crosses no edges and
    # still satisfies every local condition; it just makes slivers rather
    # than diamonds, and the result does not curl when folded.  So pin
    # the geometry: an offset grid admits exactly two edge vectors.
    for rc in ((2, 3), (4, 6), (5, 4)):
        cell, hgt = 1.0, 0.8
        ym = yoshimura(rows=rc[0], cols=rc[1], cell=cell, height=hgt)
        ym.faces = build_faces(ym.verts, ym.edges)
        assert ym.n_faces > 0
        assert validate(ym), f"yoshimura {rc}: {validate(ym).summary()}"
        assert all(len(f) == 3 for f in ym.faces), "yoshimura is triangulated"

        want = {(cell, 0.0), (cell / 2, hgt)}
        for k, (a, b) in enumerate(ym.edges):
            d = ym.verts[b] - ym.verts[a]
            got = (round(abs(float(d[0])), 9), round(abs(float(d[1])), 9))
            assert got in want, (
                f"yoshimura {rc}: edge {k} spans {got}; an offset grid can "
                f"only have {sorted(want)} -- this is the wrong neighbour")

        # every INTERIOR vertex of a diamond grid meets six creases
        deg = {}
        for a, b in ym.edges:
            deg[int(a)] = deg.get(int(a), 0) + 1
            deg[int(b)] = deg.get(int(b), 0) + 1
        rim = set()
        for k, (a, b) in enumerate(ym.edges):
            if str(ym.assignment[k]) == BOUNDARY:
                rim.add(int(a))
                rim.add(int(b))
        inner = [v for v in range(ym.n_verts) if v not in rim]
        assert inner and all(deg[v] == 6 for v in inner), (
            f"yoshimura {rc}: interior degrees "
            f"{sorted({deg[v] for v in inner})}, expected all 6")

    # --- hypar: every cell must be a TRIANGLE ------------------------
    #
    # Not a meshing detail -- it is what makes the object foldable at
    # all.  A rigid solver holds a quad panel flat, so with trapezoidal
    # cells the concentric pleats cannot move: measured, every ring
    # crease sat at exactly 0.00 degrees while only the radial creases
    # folded, and the "hypar" came out a shallow cone.  Demaine et al.
    # 2011 says the same thing from the other side: no proper folding
    # with planar facets (Cor. 14), proper foldings of the TRIANGULATED
    # hypar (Thm 19).
    for rings, sides in ((3, 4), (4, 6), (5, 4)):
        hp = hypar(rings=rings, sides=sides)
        hp.faces = build_faces(hp.verts, hp.edges)
        assert hp.n_faces > 0
        bad = [f for f in hp.faces if len(f) != 3]
        assert not bad, (
            f"hypar {rings}x{sides}: {len(bad)} non-triangular cell(s), "
            f"sizes {sorted({len(f) for f in bad})} -- a rigid solver "
            f"holds those flat and the pleats will not fold")
    hp = hypar(rings=3, sides=4)
    hp.faces = build_faces(hp.verts, hp.edges)

    # --- Kresling: the sources' pattern, not merely a folding tube ---
    # THE HISTORY IS THE REASON THESE ARE HERE.  Two earlier Kreslings
    # passed everything asked of them and were both wrong.  The first
    # guessed the layout and never closed.  The second was isometric to a
    # tube and folded to ZERO strain -- and split each cell on the short
    # diagonal, making a convex antiprism whose creases all bend the same
    # way, where Kresling's own sentence says the parallelograms are
    # mountains "divided on their long diagonal by a valley-fold".  So
    # the checks below test the sources' claims, not self-consistency.
    from .compliant import CompliantFolder

    # (0) LIU'S WORKED EXAMPLE, reproduced.  The paper designs a strip
    # with a = 3.2 cm, h = 5.4 cm and n = 8, and reports gamma = 44.4 and
    # beta = 113.1 degrees.  The closed form used here has to land on the
    # same numbers, which is the cleanest possible check that solving
    # eq (3) for the cell height rather than for s is the same identity
    # and not a convenient rearrangement of it.
    _n, _a, _h = 8, 3.2, 5.4
    _eta = np.pi / _n
    _gamma = np.deg2rad(44.4)
    assert abs(_a * np.sin(_gamma) * np.sin(_gamma + _eta) / np.sin(_eta)
               - _h) < 0.02, "kresling: closed form misses Liu's example"
    assert abs(np.degrees(np.pi - _gamma - _eta) - 113.1) < 0.1

    for rows, n, ad in ((2, 5, 70.0), (3, 6, 72.0), (2, 8, 65.0)):
        fr = kresling(rows=rows, cols=n, alpha=np.deg2rad(ad))
        R, h, off, phi, H, alpha = fr.meta["kresling"]
        eta = np.pi / n
        a = 2.0 * R * np.sin(eta)

        # (1) THE CLOSURE CONDITION.  b/a = sin(alpha-eta)/sin(eta) is
        # what makes the folded strip close into a regular n-gon; a
        # freely chosen cell folds but never closes.
        b = np.hypot(off, h)
        want = np.sin(alpha - eta) / np.sin(eta)
        assert abs(b / a - want) < 1e-9, (
            f"kresling {rows}x{n}: b/a is {b / a:.6f} where closure needs "
            f"{want:.6f} -- this strip cannot shut into a regular polygon")

        # (2) THE LONG DIAGONAL, not the short one.
        assert (a + off) ** 2 > (a - off) ** 2 and off > 0.0, (
            f"kresling {rows}x{n}: cell leans {off:+.3f}, so the diagonal "
            f"drawn is not the long one")

        # (3) THE ASSIGNMENT BOTH SOURCES STATE: outline mountain, long
        # diagonal valley.  Measured on the deployed module, never
        # assigned, so agreement is evidence about the geometry.
        pos = {(round(float(p[0]), 9), round(float(p[1]), 9)): k
               for k, p in enumerate(fr.verts)}
        node = [[pos[(round(j * a + i * off, 9), round(i * h, 9))]
                 for j in range(n + 1)] for i in range(rows + 1)]
        diag = {(min(node[i][j], node[i + 1][j + 1]),
                 max(node[i][j], node[i + 1][j + 1]))
                for i in range(rows) for j in range(n)}
        bad = []
        for k, (p, q) in enumerate(fr.edges):
            code = str(fr.assignment[k])
            if code == BOUNDARY:
                continue
            is_d = (min(int(p), int(q)), max(int(p), int(q))) in diag
            if code != (VALLEY if is_d else MOUNTAIN):
                bad.append(("diagonal" if is_d else "outline", code))
        assert not bad, (
            f"kresling {rows}x{n}: {len(bad)} crease(s) disagree with the "
            f"sources -- the outline must measure MOUNTAIN and the long "
            f"diagonal VALLEY, got e.g. {bad[:4]}")

        drv = [k for k, c in enumerate(fr.assignment)
               if c in (MOUNTAIN, VALLEY)]
        assert drv and not np.isnan(fr.fold_angle[drv]).any(), (
            f"kresling {rows}x{n}: a driven crease has no measured angle")

    # (4) IT DEPLOYS.  One fold, at the smallest size that shows it: the
    # strip must contract, the two cut edges must meet, and the top ring
    # must turn by rows * phi -- the closed form predicting the folded
    # twist is the strongest single check available here, because the
    # solver knows nothing about that number.
    rows, n = 2, 5
    fr = kresling(rows=rows, cols=n, alpha=np.deg2rad(70.0))
    R, h, off, phi, H, alpha = fr.meta["kresling"]
    a = 2.0 * R * np.sin(np.pi / n)
    pos = {(round(float(p[0]), 9), round(float(p[1]), 9)): k
           for k, p in enumerate(fr.verts)}
    node = [[pos[(round(j * a + i * off, 9), round(i * h, 9))]
             for j in range(n + 1)] for i in range(rows + 1)]

    w0 = float(np.ptp(fr.verts[:, 0]))
    cf = CompliantFolder(fr)
    cf.run(drive=1.0)
    P = cf.pos
    Q = P - P.mean(0)
    wide = max(float(np.ptp(Q @ u))
               for u in np.linalg.svd(Q, full_matrices=False)[2])
    assert wide < 0.7 * w0, (
        f"kresling does not curl: widest extent went {w0:.2f} -> "
        f"{wide:.2f}, where a tube must contract")

    seam = float(np.mean([np.linalg.norm(P[node[i][0]] - P[node[i][n]])
                          for i in range(rows + 1)])) / a
    assert seam < 0.15, (
        f"kresling does not close: its two cut edges finish {seam:.2f} "
        f"ring-edges apart, so the tube is still a strip")

    bot = P[[node[0][j] for j in range(n)]]
    top = P[[node[rows][j] for j in range(n)]]
    cb, ct = bot.mean(0), top.mean(0)
    ax = ct - cb
    ax = ax / np.linalg.norm(ax)
    u = bot[0] - cb
    u -= (u @ ax) * ax
    w = top[0] - ct
    w -= (w @ ax) * ax
    turn = np.degrees(np.arctan2(np.cross(u, w) @ ax, u @ w))
    want = np.degrees(rows * phi)
    assert abs(abs(turn) - want) < 6.0, (
        f"kresling twists {abs(turn):.1f} deg where its closed form says "
        f"{want:.1f}. Without the twist this is not a Kresling")

    assert float(np.abs(cf.edge_strain()).max()) < 0.02, (
        "kresling reached its tube by stretching, not by folding")

    # (5) THE ZIGZAG STACKING IS A DIFFERENT MECHANISM, not a different
    # drawing.  Mirroring alternate bands cancels their twists in pairs,
    # so an even stack must close into a tube that does NOT turn -- and
    # that contrast with the check above is the whole reason both are
    # shipped.  The assignment must survive the mirroring too: a band
    # leaning the other way has its long diagonal the other way, and
    # getting that wrong puts short diagonals in every second band.
    fz = kresling_zigzag(rows=2, cols=5, alpha=np.deg2rad(70.0))
    Rz, hz, offz, phiz, Hz, alz = fz.meta["kresling"]
    assert fz.meta["kresling_signs"] == (1.0, -1.0), \
        "kresling_zigzag: bands are not alternating"
    az = 2.0 * Rz * np.sin(np.pi / 5)
    lean = [0.0]
    for sg in fz.meta["kresling_signs"]:
        lean.append(lean[-1] + sg * offz)
    posz = {(round(float(p[0]), 9), round(float(p[1]), 9)): k
            for k, p in enumerate(fz.verts)}
    nz = [[posz[(round(j * az + lean[i], 9), round(i * hz, 9))]
           for j in range(6)] for i in range(3)]
    for k, c in enumerate(fz.assignment):
        assert c in (MOUNTAIN, VALLEY, BOUNDARY), (k, c)

    cz = CompliantFolder(fz)
    cz.run(drive=1.0)
    Pz = cz.pos
    seamz = float(np.mean([np.linalg.norm(Pz[nz[i][0]] - Pz[nz[i][5]])
                           for i in range(3)])) / az
    assert seamz < 0.15, (
        f"kresling_zigzag does not close: seam {seamz:.2f} ring-edges")
    b0 = Pz[[nz[0][j] for j in range(5)]]
    t0 = Pz[[nz[2][j] for j in range(5)]]
    c0, c1 = b0.mean(0), t0.mean(0)
    axz = c1 - c0
    axz = axz / np.linalg.norm(axz)
    uz = b0[0] - c0
    uz -= (uz @ axz) * axz
    wz = t0[0] - c1
    wz -= (wz @ axz) * axz
    netz = abs(np.degrees(np.arctan2(np.cross(uz, wz) @ axz, uz @ wz)))
    assert netz < 5.0, (
        f"kresling_zigzag nets {netz:.1f} deg of twist over an even stack "
        f"where the mirrored bands must cancel to zero -- if it turns, "
        f"the bands are not actually mirrored")
    assert float(np.abs(cz.edge_strain()).max()) < 0.02, (
        "kresling_zigzag reached its tube by stretching")

    # --- Resch: the reference molecule, vertex figure by figure ----
    # THE PATTERN BEFORE THIS ONE WAS BUILT FROM A DESCRIPTION and every
    # one of its 31 interior vertices had ODD degree -- which no folded
    # state of any kind can have, so it curled and stretched (41 per cent
    # out of plane, 2.3 per cent strain) instead of folding.  These check
    # the two figures measured off Ghassaei's reference crease pattern,
    # because that is the only thing that distinguishes this from another
    # plausible-looking triangular grid.
    fr = resch(rows=3, cols=3)
    assert all(len(f) == 3 for f in fr.faces), (
        "resch: Resch's creases describe TRIANGULAR faces; sizes "
        f"{sorted({len(f) for f in fr.faces})}")

    inc = {}
    for k, (p, q) in enumerate(fr.edges):
        for a, b in ((int(p), int(q)), (int(q), int(p))):
            d = fr.verts[b] - fr.verts[a]
            inc.setdefault(a, []).append(
                (str(fr.assignment[k]),
                 round(float(np.degrees(np.arctan2(d[1], d[0]))) % 360),
                 round(float(np.linalg.norm(d)), 2)))

    # A hub: three short mountains and three long valleys, alternating.
    hub = tuple(sorted([(MOUNTAIN, 90, 1.0), (MOUNTAIN, 210, 1.0),
                        (MOUNTAIN, 330, 1.0), (VALLEY, 30, 2.0),
                        (VALLEY, 150, 2.0), (VALLEY, 270, 2.0)]))
    # A tip: the hub's mountain arriving, the valley carrying on to the
    # next hub, and four sqrt(3) mountains out to the neighbouring tips.
    s3 = round(float(np.sqrt(3.0)), 2)
    tip = tuple(sorted([(MOUNTAIN, 270, 1.0), (VALLEY, 90, 2.0),
                        (MOUNTAIN, 0, s3), (MOUNTAIN, 60, s3),
                        (MOUNTAIN, 120, s3), (MOUNTAIN, 180, s3)]))
    figs = {tuple(sorted(v)) for v in inc.values()}
    assert hub in figs, (
        "resch: no vertex has the reference HUB figure -- three "
        "mountains of u at 90/210/330 and three valleys of 2u at "
        "30/150/270")
    assert tip in figs, (
        "resch: no vertex has the reference TIP figure -- a mountain of "
        "u back to its hub, a valley of 2u on to the next, and four "
        "mountains of sqrt(3) u to the neighbouring tips")

    # NOT FLAT-FOLDABLE, and that is the pattern rather than a defect: a
    # hub is 3 mountains and 3 valleys, so |M - V| = 0 and Maekawa fails
    # there by construction.  Asserted so nobody "fixes" it later.
    rep = validate(fr)
    assert not rep and rep.n_interior, (
        "resch should FAIL Maekawa at its hubs; if it passes, the hubs "
        "are no longer 3 mountains and 3 valleys and this is not Resch")

    # --- Square twist: the published crease pattern ------------------
    # THE FIRST VERSION OF THIS WAS A DIFFERENT SQUARE TWIST -- the
    # "rigidly folded" one from edemaine/fold, whose inner square carries
    # two mountains and two valleys and whose outline is a square rather
    # than a pinwheel.  It matched its own reference exactly and was
    # still not what anybody means by the square twist.  Matching a
    # reference is not enough; it has to be the right reference.  So the
    # check here is against the FAMILIAR pattern, and the single fact
    # that separates the two is first: every side of the inner square is
    # a valley.
    fr = square_twist()
    assert fr.n_verts == 16 and fr.n_edges == 24, (
        f"square twist: {fr.n_verts} verts, {fr.n_edges} edges; the "
        f"reference figure has 16 and 24")
    kinds = collections.Counter(str(a) for a in fr.assignment)
    assert kinds == collections.Counter({VALLEY: 8, MOUNTAIN: 4,
                                         BOUNDARY: 12}), (
        f"square twist: {dict(kinds)}; singlesquaretwist.svg has 8 blue, "
        f"4 red and 12 black")

    # The inner square's corners are the only vertices carrying four
    # creases, so the creases joining two of them are its sides.
    deg = collections.Counter()
    for a, b in fr.edges:
        deg[int(a)] += 1
        deg[int(b)] += 1
    corner = {v for v, n in deg.items() if n >= 4}
    assert len(corner) == 4, (
        f"square twist: {len(corner)} vertices of degree 4, wanted the "
        f"inner square's four corners")
    sides = [str(fr.assignment[k]) for k, (a, b) in enumerate(fr.edges)
             if int(a) in corner and int(b) in corner]
    assert len(sides) == 4 and set(sides) == {VALLEY}, (
        f"square twist: the inner square reads {sides}.  Every side of "
        f"it is a VALLEY -- a mountain there means this is the rigidly "
        f"foldable variant, which is a different pattern")

    for v in corner:
        got = [str(fr.assignment[k]) for k, (a, b) in enumerate(fr.edges)
               if v in (int(a), int(b))]
        assert sorted(got) == sorted([VALLEY] * 3 + [MOUNTAIN]), (
            f"square twist: a corner reads {sorted(got)}, wanted three "
            f"valleys and one mountain")

    fr.faces = build_faces(fr.verts, fr.edges)
    assert len(fr.faces) == 9, (
        f"square twist: {len(fr.faces)} faces; the cell is four squares, "
        f"four pleats and the twist square")
    rep = validate(fr)
    assert rep and rep.n_interior == 4, (
        f"square twist should be locally flat-foldable at all four "
        f"interior vertices: {rep.summary()}")

    # THE TILING HAS TO STAY FOLDABLE TOO, and that is not automatic:
    # the denser lattice fits the paper perfectly and still runs a valley
    # into a mountain where two cells meet.  Four interior vertices per
    # twist, all passing, is the check.
    for r, cl in ((2, 2), (3, 2)):
        ft = square_twist(rows=r, cols=cl)
        ft.faces = build_faces(ft.verts, ft.edges)
        rt = validate(ft)
        assert rt and rt.n_interior == 4 * r * cl, (
            f"square twist {r}x{cl}: {rt.summary()}; expected "
            f"{4 * r * cl} interior vertices, all flat-foldable")

    # The twist angle is a real parameter, not decoration: the whole
    # family from a shallow twist to a deep one has to stay foldable.
    for deg_ in (35.0, 50.0, 75.0):
        f2 = square_twist(alpha=np.deg2rad(deg_))
        f2.faces = build_faces(f2.verts, f2.edges)
        r2 = validate(f2)
        assert r2 and r2.n_interior == 4, (
            f"square twist at alpha={deg_}: {r2.summary()}")

    # --- dispatch ----------------------------------------------------
    for name in PATTERNS:
        fr = build(name)
        assert fr.n_edges > 0 and fr.is_flat, name
        # every edge carries one of the five legal assignments.  U is
        # among them and is not a defect: the hypar's triangulation
        # diagonals are deliberately unassigned, because real paper
        # takes those up by bending and the solver must be free to pick
        # the angle.  (This assertion used to allow only M/V/B, which
        # contradicted its own comment and would have rejected that.)
        assert set(fr.assignment.tolist()) <= set(ASSIGNMENTS), name
    try:
        build("NOPE")
    except ValueError as exc:
        assert "unknown pattern" in str(exc)
    else:
        raise AssertionError("unknown pattern should raise")

    print("RESULT: OK  crease.patterns")


# NOTE: no __main__ guard -- tests/test_selftests.py discovers and runs
# _selftest() headlessly (see CLAUDE.md).
