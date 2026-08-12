
# Grammar library: PLANTS.
#
# These are the grammars the previous generator could not express at all.
# The bracketed 2-D plants were within reach of a per-character rewriter,
# but everything below them -- parametric internodes, stochastic
# variation, tapering with `!`, filled leaves with `{ . }`, signal
# propagation with context, shedding with `%` -- was not.
#
# ABOP figure numbers are given where a grammar is transcribed from the
# book, so a rendering can be checked against the printed plate.
#
# References:
# - Przemyslaw Prusinkiewicz and Aristid Lindenmayer, "The Algorithmic
#   Beauty of Plants", Springer, 1990 -- Figure 1.24 (the six bracketed
#   plants), Figure 1.25 (the 3-D bush), Figure 1.26 (the flowering
#   plant), section 1.7 (stochastic branching) and section 1.10
#   (parametric).
# - Przemyslaw Prusinkiewicz, Aristid Lindenmayer and James Hanan,
#   "Developmental Models of Herbaceous Plants for Computer Imagery
#   Purposes", SIGGRAPH '88 -- the signal model whose sign
#   Delta = un - vm decides acropetal versus basipetal flowering, the
#   raceme and the cordate leaf.
# - Przemyslaw Prusinkiewicz, Jim Hanan, Mark Hammel and Radomir Mech,
#   "L-systems: from the Theory to Visual Models of Plants", SIGGRAPH
#   2003 course notes, chapter 2-1 -- sections 6.2 (the compound leaf),
#   6.4 (the mesotonic structure) and 6.5 (shedding with %).
# - Aristid Lindenmayer, "Mathematical models for cellular interaction
#   in development", J. Theoretical Biology 18, 1968 -- the Anabaena
#   filament.

PLANTS = {
    # The reference grammar for `#param`: every form of the directive in
    # one place -- a plain range, a range with an explicit increment, and
    # a choice list whose options carry values.  The declared parameters
    # are ordinary constants, so this grammar is still valid, and still
    # self-describing, outside Blender.
    "TUNABLE": """
        #title Tunable Tree (parameter demo)
        #draw F
        #angle 30
        #iters 7 11
        #param ratio  0.72  0.30 0.95 0.01  Contraction ratio
        #param spread 30    5    80   1     Branch angle
        #param taper  0.72  0.40 0.95 0.01  Width taper
        #param arity  2 (Binary=2|Ternary=3|Whorl of four=4) Branches per node
        axiom: !(1)A(1)
        p1: A(w) : arity < 3 -> F(w)[+(spread)!(w*taper)A(w*ratio)][-(spread)!(w*taper)A(w*ratio)]
        p2: A(w) : arity == 3 -> F(w)[+(spread)!(w*taper)A(w*ratio)][-(spread)!(w*taper)A(w*ratio)][&(spread)!(w*taper)A(w*ratio)]
        p3: A(w) : arity > 3 -> F(w)[+(spread)!(w*taper)A(w*ratio)][-(spread)!(w*taper)A(w*ratio)][&(spread)!(w*taper)A(w*ratio)][^(spread)!(w*taper)A(w*ratio)]
    """,


    # --- ABOP Figure 1.24: the six classic bracketed plants ----------

    "PLANT": """
        #title Plant (ABOP fig 1.24f)
        #draw F
        #angle 25
        #iters 5 8
        axiom: X
        p1: X -> F+[[X]-X]-F[-FX]+X
        p2: F -> FF
    """,

    "PLANT_A": """
        #title Bracketed Plant a (ABOP fig 1.24a)
        #draw F
        #angle 25.7
        #iters 5 7
        axiom: F
        p1: F -> F[+F]F[-F]F
    """,

    "PLANT_B": """
        #title Bracketed Plant b (ABOP fig 1.24b)
        #draw F
        #angle 20
        #iters 5 7
        axiom: F
        p1: F -> F[+F]F[-F][F]
    """,

    "PLANT_C": """
        #title Bracketed Plant c (ABOP fig 1.24c)
        #draw F
        #angle 22.5
        #iters 4 6
        axiom: F
        p1: F -> FF-[-F+F+F]+[+F-F-F]
    """,

    "PLANT_D": """
        #title Bracketed Plant d (ABOP fig 1.24d)
        #draw F
        #angle 20
        #iters 7 9
        axiom: X
        p1: X -> F[+X]F[-X]+X
        p2: F -> FF
    """,

    "PLANT_E": """
        #title Bracketed Plant e (ABOP fig 1.24e)
        #draw F
        #angle 25.7
        #iters 7 9
        axiom: X
        p1: X -> F[+X][-X]FX
        p2: F -> FF
    """,

    # --- three dimensions --------------------------------------------

    "BUSH3D": """
        #title 3-D Bush (ABOP fig 1.25)
        #draw F
        #angle 25
        #iters 5 7
        axiom: A
        p1: A -> F[&A]////[&A]////[&A]////
    """,

    "BUSH_TAPERED": """
        #title 3-D Bush, tapered and colour-graded
        #draw F
        #angle 22.5
        #iters 5 7
        axiom: !(1)'(0)A
        p1: A -> !F[&'A]////[&'A]////[&'A]
        p2: F -> F
    """,

    "HILBERT_BUSH": """
        #title Ternary 3-D Tree (ABOP fig 1.27)
        #draw F
        #angle 18.95
        #iters 6 8
        #define d1 94.74
        #define d2 132.63
        #define a  18.95
        axiom: !(1)F(200)/(45)A
        p1: A -> !(0.7)F(50)[&(a)F(50)A]/(d1)[&(a)F(50)A]/(d2)[&(a)F(50)A]
        p2: F(l) -> F(l*1.109)
        p3: !(w) -> !(w*1.732)
    """,

    # --- parametric: what the old engine provably could not do -------

    "MESOTONIC": """
        #title Mesotonic Structure (course-2003 sec 6.4)
        #draw F
        #angle 45
        #iters 10 16
        axiom: FA(0)
        p1: A(v)         -> [-FB(v)][+FB(v)]FA(v+1)
        p2: B(v) : v > 0 -> FB(v-1)
    """,

    "SHEDDING": """
        #title Shedding Crown (course-2003 sec 6.5)
        #draw F
        #angle 30
        #iters 12 20
        axiom: A
        p1: A       -> F(1)[-X(3)B][+X(3)B]A
        p2: B       -> F(1)B
        p3: X(d) : d > 0  -> X(d-1)
        p4: X(d) : d == 0 -> U%
        p5: U       -> F(0.3)
    """,

    "PARAMETRIC_TREE": """
        #title Parametric Tree (continuous taper)
        #draw F
        #angle 30
        #iters 8 12
        #define R 0.72
        #define A 32
        axiom: !(0.3)F(1)X(1)
        p1: X(s) : s > 0.02 -> [+(A)!(s*R)F(s*R)X(s*R)][-(A)!(s*R)F(s*R)X(s*R)]
    """,

    # --- stochastic: many individuals of one species ------------------

    "STOCHASTIC_BUSH": """
        #title Stochastic Bush (ABOP sec 1.7)
        #draw F
        #angle 22.5
        #iters 5 7
        axiom: F
        p1: F -> F[+F]F[-F]F : 0.34
        p2: F -> F[+F]F      : 0.33
        p3: F -> F[-F]F      : 0.33
    """,

    # --- context-sensitive: a signal travelling up the axis ----------

    "ANABAENA": """
        #title Anabaena catenula (Lindenmayer 1968)
        #draw F
        #angle 0
        #iters 8 12
        #mode parallel
        #develop
        /* Lindenmayer's original organism, and the paper this whole
           subject starts from.  The filament carries two cell types: a
           LONG cell divides into a long and a short one, and a short
           cell simply grows into a long one --

               a -> a b        b -> a

           which is why the cell count runs 1, 2, 3, 5, 8, 13 ...  The
           parametric form below carries the cell length as the argument
           of F, so the two types are drawn at their real relative sizes
           (and at different widths via `!`) rather than as an
           undifferentiated chain.

           `#develop` is essential here, not decoration: a filament is
           ONE-dimensional, so a single generation can only ever be a
           bare rod -- 0.02 x 2.0 x 0.02, which reads as nothing at all
           in the viewport.  Stacking the generations as rows is how
           Lindenmayer drew it and how ABOP Fig. 1.28 shows it, and it
           is the only rendering in which the division pattern is
           visible. */
        axiom: !(0.09)F(1)
        p1: F(x) : x > 0.7  -> !(0.09)F(1)!(0.055)F(0.6)
        p2: F(x) : x <= 0.7 -> !(0.09)F(1)
    """,

    "SIGNAL": """
        #title Acropetal Signal (SIGGRAPH '88)
        #draw I J
        #angle 45
        #iters 12 24
        #ignore + - /
        #develop
        /* A context-sensitive signal walking up a filament: the rule
           `J < I -> J` turns the cell just above a J into a J, one cell
           per step.  The POINT is the position of the signal over time,
           so a single generation is both a bare rod and a picture of
           nothing in particular -- like ANABAENA it needs the
           generations drawn as rows, which is exactly how the SIGGRAPH
           '88 figure presents it. */
        axiom: JIIIIIIII
        p1: J < I -> J
    """,

    # --- filled organs, needing { . } --------------------------------

    "LEAF_POLY": """
        #title Filled Leaf Blade (ABOP sec 5.2)
        #draw F G
        #angle 60
        /* Ternary branching: 3x per step, so n=12 is 3.7M modules --
           past the operator's default budget, i.e. the preset used to
           fail at its own declared default.  n=8 is 45,923 and draws in
           a third of a second. */
        #iters 8 10
        axiom: {A}
        p1: A -> [+G.A][-G.A]G.A
    """,

    "FLOWER": """
        #title Flowering Plant (ABOP fig 1.26)
        #draw F
        #angle 18
        #iters 5 7
        module plant, internode, seg, leaf, flower, pedicel, wedge
        axiom: plant
        p1: plant   -> internode + [ plant + flower ] - - // [ - - leaf ] internode [ + + leaf ] - - plant flower
        p2: internode -> F seg [ // & & leaf ] [ // ^ ^ leaf ] F seg
        p3: seg     -> seg F seg
        p4: leaf    -> [ ' { + F - FF - F + | + F - FF - F } ]
        p5: flower  -> [ & & & pedicel ' / wedge //// wedge //// wedge //// wedge //// wedge ]
        p6: pedicel -> FF
        p7: wedge   -> [ ' ^ F ] [ { & & & & - F + F | - F + F } ]
    """,
}
