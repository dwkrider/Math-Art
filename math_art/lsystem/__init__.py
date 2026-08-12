"""lsystem -- Lindenmayer systems: rewriting, turtle interpretation, presets.

A parametric, stochastic, context-sensitive L-system engine with the full
20-symbol Appendix-C turtle, plus a library of published grammars.

Layout follows `math_art/seifert/`: this package is the Blender-FREE
engine, and the registered operator (`l_system_generator.py`) stays a
flat module that imports it.  `emit` is the only submodule that touches
bpy and is deliberately NOT imported here, so that `import lsystem` works
headlessly and `tests/test_selftests.py` can exercise the numerics.

Typical use:

    from math_art import lsystem
    g = lsystem.parse(lsystem.PRESETS["PLANT"])
    out = lsystem.interpret(g.derive(5), angle=g.angle, draw=g.draw)
    out.bbox_fit(2.0)

References for the mathematics are in each submodule's header; the
principal ones are Lindenmayer (1968); Prusinkiewicz & Lindenmayer, "The
Algorithmic Beauty of Plants" (1990); Prusinkiewicz, Hanan, Hammel &
Mech, SIGGRAPH 2003 course notes chapter 2-1; Rozenberg & Salomaa, "The
Mathematical Theory of L Systems" (1980); and Honda (1971).
"""

from ._data_curves import CURVES
from ._data_plants import PLANTS
from ._data_trees import (HONDA_KEYS, TABLE1_KEYS, TREES, honda_params,
                          table1_params)
from .core import (BRANCH_CLOSE, BRANCH_OPEN, Grammar, GrammarError, Module,
                   Param, Production, parse, tokenize, word_str)
from .expr import Expr, ExprError, compile_expr
from .polyline import max_turn, round_corners, split_at_reversals
from .growth import (EXPONENTIAL, POLYNOMIAL, UNKNOWN, classify, counts,
                     describe, growth_matrix, recurrence, safe_iterations,
                     sequence)
from .turtle import (DEFAULT_ANGLE, DEFAULT_STEP, DEFAULT_WIDTH, Strand,
                     TurtleOutput, interpret, stack_series)

#: Every shipped grammar, keyed by preset id.
PRESETS = {}
PRESETS.update(CURVES)
PRESETS.update(PLANTS)
PRESETS.update(TREES)

#: Preset ids grouped for a UI enum.
GROUPS = (
    ("Curves", tuple(CURVES)),
    ("Plants", tuple(PLANTS)),
    ("Trees", tuple(TREES)),
)


def preset(key, params=None):
    """Parse a shipped preset into a Grammar.

    `params` overrides the defaults of any `#param` it declares, e.g.
    `preset("TUNABLE", {"arity": 3})`.
    """
    return parse(PRESETS[key], params)


def title(key):
    """The human-readable name a preset declares with `#title`."""
    src = PRESETS.get(key, "")
    for line in src.splitlines():
        line = line.strip()
        if line.lower().startswith("#title"):
            return line.split(None, 1)[1].strip()
    return key.replace("_", " ").title()


def build(key_or_grammar, iters=None, seed=None, size=2.0, scale=1.0,
          width=None, tropism=None, elasticity=0.0, budget=None,
          params=None):
    """Preset id (or Grammar) -> a TurtleOutput fitted to the 2 m cube.

    Honours the grammar's own `#angle`, `#draw`, `#step` and `#iters`
    defaults, so a caller that just wants the shipped look passes only
    the key.
    """
    g = (key_or_grammar if isinstance(key_or_grammar, Grammar)
         else preset(key_or_grammar, params))
    n = g.iters if iters is None else int(iters)
    if n is None:
        n = 4
    if g.max_iters is not None:
        n = min(n, g.max_iters)
    kw = {}
    if budget is not None:
        kw["budget"] = budget
    word = g.derive(n, seed=seed, **kw)
    out = interpret(word,
                    angle=g.angle if g.angle is not None else DEFAULT_ANGLE,
                    step=g.step if g.step is not None else DEFAULT_STEP,
                    width=DEFAULT_WIDTH if width is None else width,
                    tropism=tropism, elasticity=elasticity,
                    draw=g.draw or None)
    return out.bbox_fit(size, scale)


__all__ = [
    "BRANCH_CLOSE", "BRANCH_OPEN", "CURVES", "DEFAULT_ANGLE", "DEFAULT_STEP",
    "DEFAULT_WIDTH", "EXPONENTIAL", "GROUPS", "Grammar", "GrammarError",
    "HONDA_KEYS", "Module", "PLANTS", "POLYNOMIAL", "PRESETS", "Production",
    "Param", "Strand", "TABLE1_KEYS", "TREES", "TurtleOutput", "UNKNOWN",
    "Expr",
    "ExprError", "build", "classify", "compile_expr", "counts", "describe",
    "growth_matrix", "honda_params", "interpret", "max_turn", "parse",
    "preset", "round_corners", "split_at_reversals",
    "recurrence", "safe_iterations", "sequence", "stack_series",
    "table1_params", "title",
    "tokenize", "word_str",
]


def _selftest():
    """Every shipped grammar must parse, derive and draw something.

    This is the regression fixture the plan asks for: a preset that stops
    parsing, stops producing geometry, or silently explodes in size is
    caught here rather than in Blender.
    """
    import numpy as np

    bad = []
    for key in sorted(PRESETS):
        try:
            g = preset(key)
        except GrammarError as exc:
            bad.append(f"{key}: parse failed: {exc}")
            continue
        if not g.productions:
            bad.append(f"{key}: no productions")
        n = min(g.iters or 3, 4)          # keep the sweep quick
        try:
            out = build(g, iters=n)
        except GrammarError as exc:
            bad.append(f"{key}: derive failed: {exc}")
            continue
        pts = out.points()
        if not len(pts):
            bad.append(f"{key}: produced no geometry at n={n}")
            continue
        if not np.all(np.isfinite(pts)):
            bad.append(f"{key}: non-finite coordinates")
        ext = (pts.max(axis=0) - pts.min(axis=0)).max()
        if abs(ext - 2.0) > 1e-6:
            bad.append(f"{key}: bbox {ext:.4f} != 2.0")
        # A preset that collapses to a LINE draws a bare rod and reads as
        # empty in the viewport.  `ANABAENA` did: 0.02 x 2.0 x 0.02.  That
        # is not wrong -- a filament really is one-dimensional -- but such
        # a grammar has to say so with `#develop`, so the host draws the
        # generations as rows instead of one uninformative rod.
        spread = (pts.max(axis=0) - pts.min(axis=0))
        if int((spread > 1e-6).sum()) < 2 and not g.develop:
            bad.append(f"{key}: collapses to a line (extent "
                       f"{tuple(round(float(v), 4) for v in spread)}) and "
                       f"does not declare #develop")
    if bad:
        raise AssertionError("BAD presets:\n  " + "\n  ".join(bad))

    # --- the developmental series ------------------------------------
    g = preset("ANABAENA")
    assert g.develop, "ANABAENA must ask for the developmental series"
    # Lindenmayer's filament: a -> ab, b -> a, so the cell count is the
    # Fibonacci sequence.  This is the check that the grammar is really
    # his and not a lookalike.
    cells = [len([m for m in g.derive(k) if m.sym == 'F']) for k in range(8)]
    assert cells == [1, 2, 3, 5, 8, 13, 21, 34], cells

    outs = [interpret(g.derive(k), angle=g.angle, draw=g.draw or None)
            for k in range(g.iters + 1)]
    one = outs[-1]
    lone = one.points().max(0) - one.points().min(0)
    assert int((lone > 1e-6).sum()) == 1, \
        f"a single generation of a filament should be a line, got {lone}"

    ser = stack_series(outs)
    ser.bbox_fit(2.0)
    e = ser.points().max(0) - ser.points().min(0)
    assert int((e > 1e-6).sum()) == 2, f"the series should be planar: {e}"
    # ... and not a sliver: the stack is laid out about as wide as it is
    # long, which is what the row-pitch rule exists to guarantee
    assert 0.4 < e[0] / e[1] < 2.5, f"series aspect {e[0] / e[1]:.3f}"
    # one row per generation, evenly spaced and non-overlapping
    xs = sorted({round(float(s.points[0][0]), 6) for s in ser.strands})
    assert len(xs) == len(outs), f"{len(xs)} rows for {len(outs)} generations"
    gaps = np.diff(xs)
    assert np.ptp(gaps) < 1e-9, f"rows unevenly spaced: {gaps}"

    # stacking nothing, or a single generation, must not explode
    assert not stack_series([]).strands
    assert len(stack_series([outs[0]]).strands) == len(outs[0].strands)

    # the seven grammars the previous generator shipped must survive
    for key in ("KOCH", "LEVY", "DRAGON", "ARROWHEAD", "GOSPER",
                "PLANT", "BUSH3D"):
        assert key in PRESETS, f"legacy preset {key} disappeared"

    # multi-character module names really are single modules
    g = preset("FLOWER")
    syms = {mod.sym for mod in g.axiom}
    assert "plant" in syms, syms

    # a parametric preset is correctly classified as unpredictable
    assert classify(preset("MESOTONIC")) == UNKNOWN
    # ... while a plain D0L curve is predictable and exponential
    assert classify(preset("KOCH")) == EXPONENTIAL

    # --- lattice-filling curves must actually fill their lattice ------
    # Each of these visits every node of a grid exactly once.  Checking
    # the node COUNT and the bounding box catches a wrong turn sequence,
    # which otherwise just looks like an interesting tangle: the shipped
    # Peano grammar revisited nodes at n=1, and the 3-D Hilbert had the
    # wrong roll handedness and wandered out of its cube.
    lattice = {
        "HILBERT":   lambda n: (4 ** n, (2 ** n - 1, 2 ** n - 1, 0)),
        "MOORE":     lambda n: (4 ** (n + 1),
                                (2 ** (n + 1) - 1, 2 ** (n + 1) - 1, 0)),
        "PEANO":     lambda n: (9 ** n, (3 ** n - 1, 3 ** n - 1, 0)),
        "HILBERT3D": lambda n: (8 ** n, (2 ** n - 1,) * 3),
    }
    for key, want in lattice.items():
        g = preset(key)
        for n in (1, 2, 3):
            out = interpret(g.derive(n), angle=g.angle, draw=g.draw or None)
            pts = np.vstack([s.points for s in out.strands])
            uniq = len(np.unique(np.round(pts, 6), axis=0))
            ext = tuple(int(round(x)) for x in (pts.max(0) - pts.min(0)))
            n_want, ext_want = want(n)
            assert uniq == n_want, \
                f"BAD {key} n={n}: {uniq} distinct nodes, expected {n_want}"
            assert ext == tuple(ext_want), \
                f"BAD {key} n={n}: extent {ext}, expected {tuple(ext_want)}"

    # --- planar grammars must lie flat in XY -------------------------
    # A 2-D fractal standing on edge in XZ is wrong for a Blender
    # workflow: it should read from the default top view.
    for key in ("KOCH", "LEVY", "DRAGON", "HILBERT", "PEANO", "GOSPER",
                "PLANT", "ARROWHEAD", "SIERPINSKI", "TERDRAGON"):
        g = preset(key)
        out = interpret(g.derive(min(g.iters or 3, 3)),
                        angle=g.angle, draw=g.draw or None)
        pts = np.vstack([s.points for s in out.strands])
        ext = pts.max(0) - pts.min(0)
        assert ext[2] < 1e-9, f"BAD {key} is not flat in XY (extent {ext})"
        assert ext[0] > 1e-6 and ext[1] > 1e-6, \
            f"BAD {key} collapsed to a line (extent {ext})"

    # ... and spatial ones must genuinely leave the plane
    for key in ("BUSH3D", "HILBERT3D", "HONDA_II"):
        g = preset(key)
        out = interpret(g.derive(min(g.iters or 3, 3)),
                        angle=g.angle, draw=g.draw or None)
        pts = np.vstack([s.points for s in out.strands])
        ext = pts.max(0) - pts.min(0)
        assert (ext > 1e-6).all(), f"BAD {key} is flat (extent {ext})"

    # --- a preset's OWN default must actually be reachable -----------
    # The sweep above caps every preset at n=4 to stay quick, which
    # means the iteration count a preset actually ships with was never
    # exercised.  `LEAF_POLY` declared `#iters 12` -- 3.7 MILLION
    # modules, past the operator's default budget, so selecting it
    # failed outright at its own default; and `CESARO` declared 8, which
    # is 4x-per-step growth given the budget of a 2x-per-step curve and
    # cost seconds of turtle work.  Deriving (without interpreting) is
    # cheap even at six figures, so every default is checked here.
    OPERATOR_BUDGET = 300_000          # l_system_generator's own default
    for key in sorted(PRESETS):
        g = preset(key)
        n = g.iters
        if n is None:
            continue
        try:
            word = g.derive(n, budget=OPERATOR_BUDGET)
        except GrammarError as exc:
            bad.append(f"{key}: its own default #iters {n} is unreachable "
                       f"within the operator's {OPERATOR_BUDGET:,}-module "
                       f"budget: {exc}")
            continue
        # A preset whose default is a large fraction of the budget will
        # be slow enough to feel broken in the redo panel, even though
        # it technically succeeds.
        if len(word) > OPERATOR_BUDGET // 2:
            bad.append(f"{key}: default #iters {n} yields {len(word):,} "
                       f"modules -- too slow for the redo panel")
    if bad:
        raise AssertionError("BAD preset defaults:\n  " + "\n  ".join(bad))

    # --- grammar-declared free parameters ----------------------------
    # `#param` is what lets a grammar expose its own knobs to a UI
    # without the operator knowing anything about that grammar.
    g = preset("TUNABLE")
    by = {q.name: q for q in g.params}
    assert set(by) == {"ratio", "spread", "taper", "arity"}, sorted(by)
    assert by["ratio"].label == "Contraction ratio"
    assert (by["ratio"].lo, by["ratio"].hi, by["ratio"].step) == (0.3, 0.95, 0.01)
    assert by["arity"].is_choice and not by["ratio"].is_choice
    assert [c[0] for c in by["arity"].choices] == \
        ["Binary", "Ternary", "Whorl of four"]

    # a declared range is enforced, not merely advertised
    assert preset("TUNABLE", {"ratio": 99.0}).constants["ratio"] == 0.95
    assert preset("TUNABLE", {"ratio": -5.0}).constants["ratio"] == 0.3
    # ... and so is the declared increment
    assert abs(preset("TUNABLE", {"ratio": 0.7149}).constants["ratio"]
               - 0.71) < 1e-9
    # a choice snaps to the nearest DECLARED value, never an arbitrary one
    assert preset("TUNABLE", {"arity": 2.6}).constants["arity"] == 3.0
    assert preset("TUNABLE", {"arity": 99}).constants["arity"] == 4.0

    # the override must actually reach the geometry: arity selects
    # between three productions, so the strand count has to change
    counts = [len(build("TUNABLE", iters=5, params={"arity": a}).strands)
              for a in (2, 3, 4)]
    assert counts[0] < counts[1] < counts[2], counts

    # an unparameterised grammar simply reports none
    assert preset("KOCH").params == ()

    # --- curves that retrace themselves are detected -----------------
    # The Levy C curve genuinely doubles back; the emitter must split
    # there or Blender's bevel produces spinning, pinched tube.
    from .turtle import DEFAULT_ANGLE  # noqa: F401  (keeps the import list honest)
    g = preset("LEVY")
    out = interpret(g.derive(10), angle=g.angle, draw=g.draw or None)
    s = out.strands[0]
    v = np.diff(s.points, axis=0)
    v = v / np.linalg.norm(v, axis=1)[:, None]
    rev = int(((v[:-1] * v[1:]).sum(axis=1) <= -0.999).sum())
    assert rev > 0, "expected the Levy C curve to retrace itself"

    print(f"lsystem: OK -- {len(PRESETS)} presets parse, derive, draw and "
          f"fit the 2 m cube; legacy 7 intact")
