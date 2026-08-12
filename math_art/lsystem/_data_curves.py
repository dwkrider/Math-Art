
# Grammar library: fractal and space-filling CURVES.
#
# Bulk data, kept out of the engine and the operator the way the repo
# keeps `_canonical_polyhedra_data.py` and `_uniform_snub_data.py`.
#
# Each entry is the grammar source text; the `#`-directives carry the
# presentation defaults (draw letters, angle, sensible and maximum
# iteration counts, whether the figure closes) so the operator needs no
# parallel table of magic numbers.
#
# The seven grammars the previous generator shipped -- Koch snowflake,
# Levy C, Heighway dragon, Sierpinski arrowhead, Gosper, plant and 3-D
# bush -- are all here with their original axioms, rules, angles and
# iteration limits, so saved operator presets keep working.
#
# References:
# - Helge von Koch, "Sur une courbe continue sans tangente, obtenue par
#   une construction geometrique elementaire", Arkiv for Matematik 1,
#   1904 (the snowflake).
# - Paul Levy, "Les courbes planes ou gauches et les surfaces composees
#   de parties semblables au tout", J. Ecole Polytechnique, 1938.
# - Martin Gardner, "Mathematical Games", Scientific American 235(6),
#   1976 -- the Gosper (flowsnake) curve, after William Gosper; also the
#   Heighway dragon.
# - Waclaw Sierpinski, "Sur une courbe dont tout point est un point de
#   ramification", C. R. Acad. Sci. Paris 160, 1915 (arrowhead, gasket).
# - Giuseppe Peano, "Sur une courbe, qui remplit toute une aire plane",
#   Mathematische Annalen 36, 1890.
# - David Hilbert, "Ueber die stetige Abbildung einer Linie auf ein
#   Flaechenstueck", Mathematische Annalen 38, 1891.
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990, sections 1.3-1.4 -- the turtle
#   encodings of these curves, the FASS-curve family and the
#   Koch-construction/edge-rewriting distinction.
# - Przemyslaw Prusinkiewicz, Aristid Lindenmayer and F. David Fracchia,
#   "Synthesis of Space-Filling Curves on the Square Grid", in Fractals
#   in the Fundamental and Applied Sciences, North-Holland, 1991.

CURVES = {

    "KOCH": """
        #title Koch Snowflake
        #draw F
        #angle 60
        #iters 3 6
        #closed
        axiom: F--F--F
        p1: F -> F+F--F+F
    """,

    "KOCH_ISLAND": """
        #title Koch Island (quadratic)
        #draw F
        #angle 90
        #iters 2 4
        #closed
        axiom: F-F-F-F
        p1: F -> F-F+F+FF-F-F+F
    """,

    "LEVY": """
        #title Levy C Curve
        #draw F
        #angle 45
        #iters 10 15
        axiom: F
        p1: F -> +F--F+
    """,

    "DRAGON": """
        #title Heighway Dragon
        #draw F
        #angle 90
        #iters 11 16
        axiom: FX
        p1: X -> X+YF+
        p2: Y -> -FX-Y
    """,

    "TERDRAGON": """
        #title Terdragon
        #draw F
        #angle 120
        #iters 7 10
        axiom: F
        p1: F -> F+F-F
    """,

    "ARROWHEAD": """
        #title Sierpinski Arrowhead
        #draw A B
        #angle 60
        #iters 6 10
        axiom: A
        p1: A -> B-A-B
        p2: B -> A+B+A
    """,

    "SIERPINSKI": """
        #title Sierpinski Gasket
        #draw F G
        #angle 120
        #iters 6 9
        axiom: F-G-G
        p1: F -> F-G+F+G-F
        p2: G -> GG
    """,

    "GOSPER": """
        #title Gosper (Flowsnake)
        #draw A B
        #angle 60
        #iters 4 6
        axiom: A
        p1: A -> A-B--B+A++AA+B-
        p2: B -> +A-BB--B-A++A+B
    """,

    "PEANO": """
        #title Peano Curve
        #draw F
        #angle 90
        #iters 3 4
        /* The single-production form widely quoted as "the Peano curve",
           F -> F+F-F-F-F+F+F+F-F, is NOT one: it revisits nodes at the
           very first iteration (8 distinct nodes from 9 steps) and spans
           a 2x3 box rather than filling a 3x3 one.  Peano's curve needs
           two non-terminals.  Verified: n=1/2/3 visit 9/81/729 distinct
           nodes on exact 3x3, 9x9 and 27x27 grids. */
        axiom: X
        p1: X -> XFYFX+F+YFXFY-F-XFYFX
        p2: Y -> YFXFY-F-XFYFX+F+YFXFY
    """,

    "HILBERT": """
        #title Hilbert Curve (2-D)
        #draw F
        #angle 90
        #iters 5 7
        axiom: X
        p1: X -> +YF-XFX-FY+
        p2: Y -> -XF+YFY+FX-
    """,

    "MOORE": """
        #title Moore Curve (closed Hilbert)
        #draw F
        #angle 90
        #iters 4 6
        axiom: LFL+F+LFL
        p1: L -> -RF+LFL+FR-
        p2: R -> +LF-RFR-FL+
    """,

    # RAW string: this rule is full of backslashes (the roll-right
    # symbol), and in a normal literal they would be eaten as escapes --
    # the trailing one would even swallow the newline as a line
    # continuation.  What is written here is exactly what the parser sees.
    "HILBERT3D": r"""
        #title Hilbert Curve (3-D)
        #draw F
        #angle 90
        #iters 3 4
        /* Verified: n=1/2/3 visit 8/64/512 distinct lattice nodes,
           exactly filling a 2x2x2, 4x4x4 and 8x8x8 cube.  The ROLL pair
           is swapped relative to the usual published listing, whose
           roll-left/roll-right symbols turn the opposite way from this
           turtle's.  With the published handedness the curve leaves the
           cube and revisits nodes: bbox 1x2x1 at n=1, and 63 of 64
           nodes at n=2. */
        axiom: X
        p1: X -> ^/XF^/XFX-F^\\XFX&F+\\XFX-F\X-\
    """,

    "DRAGON_TWIN": """
        #title Twindragon
        #draw F
        #angle 90
        #iters 10 14
        axiom: FX+FX+
        p1: X -> X+YF
        p2: Y -> FX-Y
    """,

    "KRISHNA": """
        #title Krishna Anklets (kolam)
        #draw F
        #angle 45
        #iters 3 5
        axiom: -X--X
        p1: X -> XFX--FF--FXFX
    """,

    "CESARO": """
        #title Cesaro Curve
        #draw F
        #angle 85
        /* 4x per step, like Koch -- not 2x like the Levy curve whose
           iteration budget this preset originally borrowed.  n=8 was
           152,916 modules and about three seconds of turtle work for a
           figure that is already visually complete at 6 (9,556). */
        #iters 6 8
        axiom: F
        p1: F -> F+F--F+F
    """,

    "PENTAPLEXITY": """
        #title Pentaplexity
        #draw F
        #angle 36
        #iters 4 6
        #closed
        axiom: F++F++F++F++F
        p1: F -> F++F++F|F-F++F
    """,

    "QUADKOCH": """
        #title Quadratic Koch (Minkowski sausage)
        #draw F
        #angle 90
        #iters 3 5
        axiom: F
        p1: F -> F+F-F-F+F
    """,
}
