
# Edge rewriting: the Koch construction, paper folding, and FASS curves.
#
# WHY THIS IS NOT THE SAME AS `core.py`.  The 2003 course notes state the
# distinction precisely, and it is the reason `turtle_curve_generator`
# has two rewriting modes rather than one:
#
#   * In the KOCH CONSTRUCTION (this module), a straight segment is
#     replaced by a set of lines whose position, orientation and scale
#     are determined BY THE SEGMENT BEING REPLACED.  Substitution is
#     geometric and local: each edge is an independent similarity.
#
#   * In PLANT-STYLE REWRITING (`core.py` + `turtle.py`), the position
#     and orientation of a module are determined by the whole chain of
#     modules from the base of the structure to that module.  When an
#     internode bends, every subtended branch rotates and displaces with
#     it, preserving connectivity.
#
# "Development is simulated as a parallel application of productions,
# followed by a sequential connection of the child structures."  The same
# production under the two composition rules gives different figures --
# which is why the paper shows them side by side.  They are not unified
# here, deliberately.
#
# A generator is given as a polyline in the unit frame: it starts at
# (0,0), ends at (1,0), and every iteration maps it onto each existing
# segment by the similarity that carries (0,0)->(1,0) onto that segment.
# A per-edge SIGN flips the generator across the segment, which is
# Mandelbrot's signed-teragon notation and the difference between the
# Koch snowflake and its antisnowflake.
#
# References:
# - Helge von Koch, "Sur une courbe continue sans tangente...", Arkiv
#   for Matematik 1, 1904.
# - Ernesto Cesaro, "Remarques sur la courbe de von Koch", Atti della
#   R. Accademia della Scienze fisiche e matematiche di Napoli 12, 1906
#   -- the free bump angle.
# - Paul Levy, "Les courbes planes ou gauches et les surfaces composees
#   de parties semblables au tout", J. Ecole Polytechnique, 1938.
# - Benoit Mandelbrot, "The Fractal Geometry of Nature", W. H. Freeman,
#   1982, chapters 6-7 -- teragons, signed generators, islands.
# - Przemyslaw Prusinkiewicz, Aristid Lindenmayer and F. David Fracchia,
#   "Synthesis of Space-Filling Curves on the Square Grid", in Fractals
#   in the Fundamental and Applied Sciences, North-Holland, 1991 -- the
#   FASS definition and the connecting-pattern tables.
# - Chandler Davis and Donald Knuth, "Number Representations and Dragon
#   Curves", J. Recreational Mathematics 3, 1970 -- the paperfolding
#   sequence.
# - Przemyslaw Prusinkiewicz et al., "L-systems: from the Theory to
#   Visual Models of Plants", SIGGRAPH 2003 course notes, section 3.

import numpy as np

from .closure import closes_geometrically, minimum_turn, turns_of


# ---------------------------------------------------------------------
# Generators, as unit-frame polylines from (0,0) to (1,0)
# ---------------------------------------------------------------------

def from_turns(turns, lengths=None, close_to_unit=True):
    """Build a unit-frame generator from a turn/length sequence.

    `turns[i]` is the turn in degrees applied BEFORE segment i.  The
    result is rescaled and rotated so it runs from (0,0) to (1,0), which
    is what makes it usable as a Koch generator on any edge.
    """
    turns = list(turns)
    n = len(turns)
    lengths = [1.0] * n if lengths is None else list(lengths)
    if len(lengths) != n:
        raise ValueError("turns and lengths must agree in length")
    pts, head, pos = [np.zeros(2)], 0.0, np.zeros(2)
    for t, L in zip(turns, lengths):
        head += float(t)
        r = np.radians(head)
        pos = pos + L * np.array([np.cos(r), np.sin(r)])
        pts.append(pos.copy())
    P = np.array(pts)
    if close_to_unit:
        end = P[-1] - P[0]
        d = float(np.linalg.norm(end))
        if d < 1e-12:
            raise ValueError("generator returns to its start; it cannot "
                             "be normalised onto a unit edge")
        a = np.arctan2(end[1], end[0])
        c, s = np.cos(-a), np.sin(-a)
        R = np.array([[c, -s], [s, c]])
        P = (P - P[0]).dot(R.T) / d
    return P


def koch_generator(alpha=60.0, parts=4):
    """The classic bump: `_/\\_` with a free apex angle.

    `alpha` is Cesaro's free parameter.  60 degrees gives von Koch's
    original snowflake edge; 85 gives the Cesaro curve; approaching 90
    degenerates the bump into a spike.
    """
    if parts != 4:
        raise ValueError("the bump generator has four parts")
    # Segment lengths so the whole thing spans exactly 1.
    a = np.radians(alpha)
    L = 1.0 / (2.0 + 2.0 * np.cos(a))
    return from_turns([0.0, alpha, -2.0 * alpha, alpha], [L, L, L, L])


#: Named signed-teragon generators.  Each entry is (turns, lengths,
#: default initiator sides, default sign pattern).
TERAGONS = {
    "KOCH":        ([0, 60, -120, 60], None, 3, +1),
    "ANTIKOCH":    ([0, 60, -120, 60], None, 3, -1),
    "CESARO":      ([0, 85, -170, 85], None, 4, +1),
    "QUADKOCH":    ([0, 90, -90, -90, 90, 0], None, 4, +1),
    "MINKOWSKI":   ([0, 90, -90, -90, -90, 90, 90, 0], None, 4, +1),
    "LEVY":        ([45, -90], None, 2, +1),
    "KOCH_SQUARE": ([0, 90, -90, -90, 90, 0], None, 4, -1),
    "SEVEN":       ([0, 60, -120, 60], None, 7, +1),
    "ELEVEN":      ([0, 60, -120, 60], None, 11, +1),
}


def polygon(n, radius=1.0, rotate=0.0):
    """Regular n-gon initiator, closed (last point repeats the first)."""
    n = max(int(n), 2)
    if n == 2:                      # a bare segment, for open teragons
        return np.array([[0.0, 0.0], [1.0, 0.0]])
    a = np.radians(rotate) + np.arange(n + 1) * (2 * np.pi / n)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)


def teragon(initiator, generator, iters, signs=None):
    """Apply a Koch generator to every edge, `iters` times.

    Each edge is replaced by the similarity image of `generator` that
    carries (0,0)->(1,0) onto that edge.  `signs` is +1 (bump outward,
    i.e. to the left of travel) or -1 (inward) per iteration, or a single
    value used throughout -- Mandelbrot's signed teragons, and the only
    difference between the Koch snowflake and the antisnowflake.
    """
    P = np.asarray(initiator, dtype=float)[:, :2].copy()
    G = np.asarray(generator, dtype=float)[:, :2]
    iters = max(int(iters), 0)
    if isinstance(signs, (int, float)) or signs is None:
        signs = [1 if signs is None else signs] * iters
    signs = list(signs) + [signs[-1] if signs else 1] * (iters - len(signs))

    for k in range(iters):
        sg = 1.0 if signs[k] >= 0 else -1.0
        # The initiator runs counter-clockwise, so its interior is to the
        # LEFT of travel and a generator bumping left would eat into it.
        # Negate so that +1 means OUTWARD (snowflake) and -1 INWARD
        # (antisnowflake), which is the sense Mandelbrot's signs carry.
        Gk = G * np.array([1.0, -sg])
        a, b = P[:-1], P[1:]
        d = b - a
        # the similarity carrying (0,0)->(1,0) onto (a->b) is
        # z -> a + (dx, dy) applied as a complex multiply
        gx, gy = Gk[:, 0], Gk[:, 1]
        # broadcast: (edges, gen_points, 2)
        x = (a[:, None, 0] + d[:, None, 0] * gx[None, :]
             - d[:, None, 1] * gy[None, :])
        y = (a[:, None, 1] + d[:, None, 1] * gx[None, :]
             + d[:, None, 0] * gy[None, :])
        new = np.stack([x, y], axis=2)
        # drop each edge's duplicated start point except the very first
        P = np.vstack([new[0, :1], new[:, 1:].reshape(-1, 2)])
    return P


def teragon_preset(name, iters=4, alpha=None, sides=None, sign=None):
    """A named teragon, with the free parameters overridable."""
    if name not in TERAGONS:
        raise KeyError(f"unknown teragon {name!r}")
    turns, lengths, dsides, dsign = TERAGONS[name]
    if alpha is not None and len(turns) == 4:
        turns = [0.0, alpha, -2.0 * alpha, alpha]
    gen = from_turns(turns, lengths)
    init = polygon(dsides if sides is None else sides)
    return teragon(init, gen, iters, dsign if sign is None else sign)


# ---------------------------------------------------------------------
# Paper folding
# ---------------------------------------------------------------------

def fold_sequence(level, base=2):
    """The regular paperfolding (dragon) sequence of a given level.

    Fold a strip in half `level` times, then unfold every crease to the
    same angle: the sequence of crease directions is the paperfolding
    sequence.  For base 2 the i-th crease (1-indexed) is a left turn iff
    the bit above the lowest set bit of i is 0 -- Davis and Knuth's
    characterisation.

    `base` 3 gives the ternary folding sequence behind the terdragon.
    """
    level = max(int(level), 0)
    n = base ** level - 1
    if n <= 0:
        return np.zeros(0, dtype=int)
    i = np.arange(1, n + 1)
    if base == 2:
        low = i & (-i)                      # lowest set bit
        return np.where((i & (low << 1)) == 0, 1, -1)
    # general base: strip trailing zeros, then look at the next digit
    q = i.copy()
    while True:
        m = (q % base) == 0
        if not m.any():
            break
        q = np.where(m, q // base, q)
    return np.where((q % base) == 1, 1, -1)


def fold_curve(level, angle=90.0, base=2, fraction=1.0):
    """Points of a folded curve, with a CONTINUOUS unfolding parameter.

    `fraction` in [0, 1] scales the creases created by the newest fold
    only, so sweeping it from 0 to 1 shows the curve unfolding from the
    previous generation into this one.  That makes the parameter
    keyframeable, which a discrete iteration count is not.

    At fraction 0 the result is exactly the previous level; at 1 it is
    exactly this level.
    """
    level = max(int(level), 0)
    seq = fold_sequence(level, base)
    if len(seq) == 0:
        return np.array([[0.0, 0.0], [1.0, 0.0]])
    # Creases belonging to the newest fold are those at odd positions
    # (1-indexed): every earlier level's creases sit at even positions.
    idx = np.arange(1, len(seq) + 1)
    newest = (idx % base) != 0
    scale = np.where(newest, float(np.clip(fraction, 0.0, 1.0)), 1.0)
    turns = seq * float(angle) * scale

    head = np.concatenate([[0.0], np.cumsum(turns)])
    r = np.radians(head)
    step = np.stack([np.cos(r), np.sin(r)], axis=1)
    pts = np.vstack([np.zeros(2), np.cumsum(step, axis=0)])
    return pts


#: Named folding curves: (base, angle).
FOLDS = {
    "DRAGON":    (2, 90.0),
    "TWINDRAGON": (2, 90.0),
    "LEVY_FOLD": (2, 45.0),
    "TERDRAGON": (3, 120.0),
    "FUDGEFLAKE": (3, 120.0),
}


# ---------------------------------------------------------------------
# FASS curves
# ---------------------------------------------------------------------
#
# FASS = space-Filling, self-Avoiding, Simple and self-Similar.  The 1991
# paper derives them from tables of "connecting patterns" on a grid; what
# is shipped here is the resulting GRAMMARS for the classical members on
# the square, triangular and hexagonal grids, not the search that
# produces them.  The paper's full table (43 patterns at n = 5) is a
# generative procedure and remains future work -- see BACKLOG.

FASS = {
    "HILBERT": ("square", 90.0, "X",
                {"X": "+YF-XFX-FY+", "Y": "-XF+YFY+FX-"}),
    "PEANO": ("square", 90.0, "X",
              {"X": "XFYFX+F+YFXFY-F-XFYFX",
               "Y": "YFXFY-F-XFYFX+F+YFXFY"}),
    "MOORE": ("square", 90.0, "LFL+F+LFL",
              {"L": "-RF+LFL+FR-", "R": "+LF-RFR-FL+"}),
    "E_CURVE": ("square", 90.0, "X",
                {"X": "FX+FX+FXFY-FY-", "Y": "+FX+FXFY-FY-FY"}),
    "GOSPER": ("triangular", 60.0, "A",
               {"A": "A-B--B+A++AA+B-", "B": "+A-BB--B-A++A+B"}),
    "SIERPINSKI_FASS": ("triangular", 60.0, "F",
                        {"F": "F-G+F+G-F", "G": "GG"}),
}


def fass_word(name, iters):
    """Expand a FASS grammar to its turtle word (pure string rewriting)."""
    if name not in FASS:
        raise KeyError(f"unknown FASS curve {name!r}")
    _grid, _angle, axiom, rules = FASS[name]
    s = axiom
    for _ in range(max(int(iters), 0)):
        s = "".join(rules.get(c, c) for c in s)
    return s


def word_to_points(word, angle, draw="F", step=1.0):
    """Read a flat turtle word as a 2-D polyline."""
    head, pos, pts = 0.0, np.zeros(2), [np.zeros(2)]
    for c in word:
        if c == '+':
            head += angle
        elif c == '-':
            head -= angle
        elif c in draw:
            r = np.radians(head)
            pos = pos + step * np.array([np.cos(r), np.sin(r)])
            pts.append(pos.copy())
    return np.array(pts)


def fass_points(name, iters):
    _grid, angle, _axiom, _rules = FASS[name]
    draw = "F" if name not in ("GOSPER", "SIERPINSKI_FASS") else "ABFG"
    return word_to_points(fass_word(name, iters), angle, draw)


def to3d(points, plane="XY"):
    """Lift a 2-D point array into the XY (default) or XZ plane."""
    P = np.asarray(points, dtype=float)
    z = np.zeros((len(P), 1))
    if plane == "XZ":
        return np.hstack([P[:, :1], z, P[:, 1:2]])
    return np.hstack([P[:, :2], z])


def _selftest():
    # --- generator normalisation -------------------------------------
    # Whatever the turns, a generator must span exactly (0,0)->(1,0),
    # because that is what makes it applicable to an arbitrary edge.
    for turns in ([0, 60, -120, 60], [0, 90, -90, -90, 90, 0], [45, -90]):
        G = from_turns(turns)
        assert np.allclose(G[0], [0, 0], atol=1e-12), turns
        assert np.allclose(G[-1], [1, 0], atol=1e-12), turns
    # a generator that returns to its start cannot be normalised
    try:
        from_turns([120, 120, 120])
        raise AssertionError("closed generator should have been rejected")
    except ValueError:
        pass

    # --- the Koch construction ---------------------------------------
    # Segment count multiplies by the generator's part count each pass.
    G = from_turns([0, 60, -120, 60])
    for it in range(4):
        P = teragon(polygon(3), G, it)
        assert len(P) - 1 == 3 * 4 ** it, (it, len(P))
        # the initiator is closed, so every teragon over it is too
        assert closes_geometrically(P), it

    # The snowflake's perimeter grows by exactly 4/3 per iteration --
    # the defining property, and a real check on the similarity map.
    per = []
    for it in range(4):
        P = teragon(polygon(3), G, it)
        per.append(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()))
    for a, b in zip(per, per[1:]):
        assert abs(b / a - 4.0 / 3.0) < 1e-9, (a, b)

    # Sign flips the bump inward: the antisnowflake encloses LESS area
    # than its initiator while the snowflake encloses more.
    def area(P):
        x, y = P[:, 0], P[:, 1]
        return 0.5 * abs(float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])))
    base = area(polygon(3))
    assert area(teragon(polygon(3), G, 3, signs=+1)) > base
    assert area(teragon(polygon(3), G, 3, signs=-1)) < base
    # The snowflake's area after n passes is exactly
    #     A_n / A_0 = 1 + (3/5)(1 - (4/9)^n)
    # rising to 8/5 in the limit.  Matching the CLOSED FORM at every n,
    # rather than just the limit, pins the similarity map down completely
    # -- a wrong scale or a wrong bump angle would still converge to
    # something, just not to this.
    for it in range(8):
        got = area(teragon(polygon(3), G, it, signs=+1)) / base
        want = 1.0 + 0.6 * (1.0 - (4.0 / 9.0) ** it)
        assert abs(got - want) < 1e-9, (it, got, want)

    # Cesaro's free apex angle really is free, and 60 reproduces Koch.
    assert np.allclose(koch_generator(60.0), from_turns([0, 60, -120, 60]),
                       atol=1e-12)
    assert not np.allclose(koch_generator(85.0), koch_generator(60.0))

    # --- paper folding -----------------------------------------------
    # The dragon's crease sequence starts L L R L L R R ...
    s = fold_sequence(4, 2)
    assert len(s) == 15, len(s)
    assert list(s[:7]) == [1, 1, -1, 1, 1, -1, -1], list(s[:7])
    # It is self-similar: even positions reproduce the previous level.
    assert list(s[1::2]) == list(fold_sequence(3, 2)), "not self-similar"

    # The dragon has exactly 2**level segments, all unit steps on the
    # integer lattice.  NOTE it is NOT self-avoiding: the Heighway dragon
    # touches itself without ever crossing -- that is what lets copies of
    # it tile the plane -- so a distinct-point count is the wrong test
    # here, though it is exactly the right one for a FASS curve below.
    P = fold_curve(10, 90.0)
    assert len(P) - 1 == 2 ** 10, len(P)
    d = np.diff(P, axis=0)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0), "steps not unit"
    assert np.allclose(P, np.round(P), atol=1e-9), "dragon left the lattice"
    # it touches itself, and that is a fact worth asserting rather than
    # discovering later
    uniq = len({(round(x, 6), round(y, 6)) for x, y in P})
    assert uniq < len(P), "the dragon is expected to touch itself"

    # Continuous unfolding: fraction 0 must give the PREVIOUS level
    # exactly, and 1 this level -- that is what makes it keyframeable.
    # At fraction 0 the newest creases relax to ZERO turn, so each new
    # pair of segments becomes collinear and what remains -- the non-zero
    # turns -- is exactly the previous level's crease sequence.  (The
    # point count stays at this level's; the SHAPE reverts.)
    a = fold_curve(4, 90.0, fraction=0.0)
    ta = turns_of(to3d(a))
    nz = ta[np.abs(ta) > 1e-9]
    assert np.allclose(nz, fold_sequence(3, 2) * 90.0), (nz,)
    # and at fraction 1 every crease is fully folded
    t1 = turns_of(to3d(fold_curve(4, 90.0, fraction=1.0)))
    assert np.allclose(np.abs(t1), 90.0), "level 4 should be fully folded"
    # and intermediate fractions must actually differ from both ends
    mid = fold_curve(4, 90.0, fraction=0.5)
    assert not np.allclose(mid, fold_curve(4, 90.0, fraction=1.0))
    assert not np.allclose(mid, a)

    # ternary folding gives the terdragon's 3x growth
    assert len(fold_sequence(3, 3)) == 26

    # --- FASS ---------------------------------------------------------
    # Each of these fills its grid exactly: node count and bounding box.
    for name, n, want, ext in (("HILBERT", 3, 4 ** 3, (7, 7)),
                               ("PEANO", 2, 9 ** 2, (8, 8)),
                               ("MOORE", 2, 4 ** 3, (7, 7))):
        P = fass_points(name, n)
        uniq = len({(round(x, 6), round(y, 6)) for x, y in P})
        got = tuple(int(round(v)) for v in (P.max(0) - P.min(0)))
        assert uniq == want, f"{name}: {uniq} nodes, want {want}"
        assert got == ext, f"{name}: extent {got}, want {ext}"

    # a FASS curve must be self-avoiding, which is the S in the acronym
    P = fass_points("HILBERT", 4)
    assert len({(round(x, 6), round(y, 6)) for x, y in P}) == len(P)

    # --- lifting ------------------------------------------------------
    P3 = to3d(np.array([[1.0, 2.0]]))
    assert P3.shape == (1, 3) and abs(P3[0, 2]) < 1e-12
    assert abs(to3d(np.array([[1.0, 2.0]]), "XZ")[0, 1]) < 1e-12

    print("koch: OK -- generators normalise to the unit edge, snowflake "
          "area matches 1+(3/5)(1-(4/9)^n) exactly, signed teragons "
          "swap area, dragon on the lattice and self-similar, "
          "continuous unfold, FASS curves fill their grids exactly")
