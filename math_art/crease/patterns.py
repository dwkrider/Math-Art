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

import numpy as np

from .fold_io import BOUNDARY, MOUNTAIN, VALLEY, Frame

PATTERNS = ("MIURA", "ACCORDION", "WATERBOMB", "YOSHIMURA", "HYPAR")


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
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        for n, (p, q) in enumerate(self.edges):
            if (p, q) == (a, b):
                # A boundary claim wins over an interior one: the rim of
                # the sheet is the rim however many cells touch it.
                if kind == BOUNDARY:
                    self.assign[n] = BOUNDARY
                return
        self.edges.append((a, b))
        self.assign.append(kind)

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

    for i in range(rows + 1):
        for j in range(cols + 1):
            # horizontal run: straight lines, alternating M/V by row
            if j < cols:
                kind = BOUNDARY if i in (0, rows) else (
                    MOUNTAIN if i % 2 else VALLEY)
                B.e(idx[i][j], idx[i][j + 1], kind)
            # Vertical zigzag: alternates by ROW index, so the crease
            # entering a vertex from below and the one leaving above
            # always disagree.  That is what makes Maekawa come out: the
            # two horizontal creases at a vertex lie on one straight
            # line and so share an assignment (2 of a kind), and the two
            # vertical ones split 1-1, giving 3-1 and |M - V| = 2.
            # Assigning both verticals alike gives 2-2 and fails at
            # every interior vertex -- which is what the self-test below
            # exists to catch.
            if i < rows:
                kind = BOUNDARY if j in (0, cols) else (
                    MOUNTAIN if i % 2 else VALLEY)
                B.e(idx[i][j], idx[i + 1][j], kind)
    return B.frame("Miura-ori")


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
                if (i % 2) and j < cols:
                    B.e(idx[i][j + 1], idx[i + 1][j], MOUNTAIN)
                elif not (i % 2) and j < cols:
                    B.e(idx[i][j], idx[i + 1][j + 1], MOUNTAIN)
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
            # the diagonals: radial creases from the centre outwards
            lower = cen if k == 0 else ring_idx[k - 1][t]
            B.e(lower, u, MOUNTAIN if t % 2 else VALLEY)
    return B.frame("Pleated Hypar")


_MAKERS = {
    "MIURA": miura,
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

    for rc in ((2, 3), (4, 6)):
        ym = yoshimura(rows=rc[0], cols=rc[1])
        ym.faces = build_faces(ym.verts, ym.edges)
        assert ym.n_faces > 0
        assert validate(ym), f"yoshimura {rc}: {validate(ym).summary()}"

    # --- hypar builds, and is honest about not folding rigidly -------
    hp = hypar(rings=3, sides=4)
    hp.faces = build_faces(hp.verts, hp.edges)
    assert hp.n_faces > 0

    # --- dispatch ----------------------------------------------------
    for name in PATTERNS:
        fr = build(name)
        assert fr.n_edges > 0 and fr.is_flat, name
        # every edge carries one of the five legal assignments
        assert set(fr.assignment.tolist()) <= {"M", "V", "B"}, name
    try:
        build("NOPE")
    except ValueError as exc:
        assert "unknown pattern" in str(exc)
    else:
        raise AssertionError("unknown pattern should raise")

    print("RESULT: OK  crease.patterns")


# NOTE: no __main__ guard -- tests/test_selftests.py discovers and runs
# _selftest() headlessly (see CLAUDE.md).
