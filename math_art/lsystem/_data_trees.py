
# Grammar library: TREES, from published parameter tables.
#
# Two tables are reproduced here, both as PARAMETRIC grammars, because
# neither can be written as a non-parametric one.
#
# 1. The nine structures of Table 1 in the SIGGRAPH 2003 course notes.
#    One production generates all nine; only the constants differ:
#
#        omega : A(100, w0)
#        p1    : A(s,w) : s >= min
#                -> !(w) F(s)
#                   [ +(a1) /(phi1) A(s*r1, w*q^e) ]
#                   [ +(a2) /(phi2) A(s*r2, w*(1-q)^e) ]
#
#    The constants are worth reading rather than merely copying:
#      r1, r2      contraction of branch length per level
#      a1, a2      branching angles off the parent
#      phi1, phi2  roll about the parent axis before branching
#      w0          initial stem width
#      q           how width is split between the two children
#      e           the width exponent.  At e = 0.5 the combined
#                  cross-sectional AREA of the children equals the
#                  parent's -- da Vinci's rule -- so `e` is exactly the
#                  dial the plan wants decoupled from the length ratio.
#      min         prune branches shorter than this
#      n           derivation depth
#
# 2. Honda's own plates.  Honda (1971) approximates a tree by repeated
#    bifurcation with branching angles theta1, theta2 and length ratios
#    R1, R2, plus a divergence angle alpha used when a branching angle is
#    zero; he alternates the SIGNS of theta1 and theta2 at every
#    branching.  His six plates each fix the whole parameter set, and
#    each demonstrates one effect:
#
#      I    axiality lost      theta1 16.7, theta2 -33.3, R 0.85/0.85
#      II   axis preserved     0, -45, 0.9/0.8, alpha 137.5
#      III  cone -> flat crown theta1 swept at constant branching angle
#      IV   width/stretch      theta2 swept -20..-80
#      V    at fixed axiality  theta1:theta2 ratio held
#      VI   apical dominance   R1,R2 swept -> conic versus flat
#
#    Honda's branching plane contains the mother branch and is
#    perpendicular to the vertical plane through it -- equivalently, its
#    steepest-gradient line is the mother direction.  `$` (rotate to
#    vertical) is exactly that re-levelling, which is why it appears in
#    every production below.
#
# References:
# - Hisao Honda, "Description of the Form of Trees by the Parameters of
#   the Tree-like Body: Effects of the Branching Angle and the Branch
#   Length on the Shape of the Tree-like Body", J. Theoretical Biology
#   31, 1971, pp. 331-338 -- assumptions (a)-(e), the end-point formula
#   and Plates I-VI.
# - Przemyslaw Prusinkiewicz, Jim Hanan, Mark Hammel and Radomir Mech,
#   "L-systems: from the Theory to Visual Models of Plants", SIGGRAPH
#   2003 course notes, chapter 2-1, section 6.3 and Table 1.
# - Akiko Nakamasu and Takumi Higaki, "Theoretical models for branch
#   formation in plants", J. Plant Research 132, 2019, pp. 325-333 --
#   Figure 1e-j runs Honda's I-model at divergence 90/137.5/90 degrees,
#   branch angle 45, with and without length contraction at ratio 0.8.
# - Leonardo da Vinci, Notebooks (the area-conservation rule behind e).

# (r1, r2, a1, a2, phi1, phi2, w0, q, e, min, n)
_TABLE1 = {
    "T1_A": (0.75, 0.77, 35, -35, 0, 0, 30, 0.50, 0.40, 0.0, 10),
    "T1_B": (0.65, 0.71, 27, -68, 0, 0, 20, 0.53, 0.50, 1.7, 12),
    "T1_C": (0.50, 0.85, 25, -15, 180, 0, 20, 0.45, 0.50, 0.5, 9),
    "T1_D": (0.60, 0.85, 25, -15, 180, 180, 20, 0.45, 0.50, 0.0, 10),
    "T1_E": (0.58, 0.83, 30, 15, 0, 180, 20, 0.40, 0.50, 1.0, 11),
    "T1_F": (0.92, 0.37, 0, 60, 180, 0, 2, 0.50, 0.00, 0.5, 15),
    "T1_G": (0.80, 0.80, 30, -30, 137, 137, 30, 0.50, 0.50, 0.0, 10),
    "T1_H": (0.95, 0.75, 5, -30, -90, 90, 40, 0.60, 0.45, 25.0, 12),
    "T1_I": (0.55, 0.95, -5, 30, 137, 137, 5, 0.40, 0.00, 5.0, 12),
}

_TABLE1_TITLES = {
    "T1_A": "Course Table 1a (broad, near-symmetric)",
    "T1_B": "Course Table 1b (weeping, unequal angles)",
    "T1_C": "Course Table 1c (flattened, 180 roll)",
    "T1_D": "Course Table 1d (planar fan)",
    "T1_E": "Course Table 1e (alternating fan)",
    "T1_F": "Course Table 1f (whip-like, thin stem)",
    "T1_G": "Course Table 1g (golden-angle roll)",
    "T1_H": "Course Table 1h (long-limbed, pruned)",
    "T1_I": "Course Table 1i (sparse, strongly pruned)",
}


def _table1_grammar(key):
    r1, r2, a1, a2, phi1, phi2, w0, q, e, mn, n = _TABLE1[key]
    return f"""
        #title {_TABLE1_TITLES[key]}
        #draw F
        #angle 30
        #iters {n} {min(n + 3, 16)}
        #define r1 {r1}
        #define r2 {r2}
        #define a1 {a1}
        #define a2 {a2}
        #define phi1 {phi1}
        #define phi2 {phi2}
        #define w0 {w0}
        #define q {q}
        #define e {e}
        #define mn {mn}
        axiom: A(100, w0)
        p1: A(s,w) : s >= mn -> !(w)F(s)[+(a1)/(phi1)A(s*r1, w*q^e)][+(a2)/(phi2)A(s*r2, w*(1-q)^e)]
    """


# (theta1, theta2, R1, R2, alpha, N, title)
_HONDA = {
    "HONDA_I": (16.7, -33.3, 0.85, 0.85, 137.5, 9,
                "Honda Plate I (main axis lost)"),
    "HONDA_II": (0.0, -45.0, 0.90, 0.80, 137.5, 9,
                 "Honda Plate II (axis preserved)"),
    "HONDA_III": (16.7, -33.3, 0.90, 0.79, 137.5, 9,
                  "Honda Plate III (cone to flat crown)"),
    "HONDA_IV": (0.0, -45.0, 0.90, 0.70, 137.5, 9,
                 "Honda Plate IV (branching angle sweep)"),
    "HONDA_V": (15.0, -30.0, 0.90, 0.75, 137.5, 9,
                "Honda Plate V (constant axiality ratio)"),
    "HONDA_VI_CONIC": (0.0, -45.0, 0.90, 0.60, 137.5, 9,
                       "Honda Plate VI conic (strong apical dominance)"),
    "HONDA_VI_FLAT": (0.0, -45.0, 0.70, 0.90, 137.5, 9,
                      "Honda Plate VI flat (weak apical dominance)"),
}


def _honda_grammar(key):
    t1, t2, r1, r2, alpha, n, title = _HONDA[key]
    # `d` carries the sign, flipped every generation, because Honda
    # substitutes -theta1, -theta2 at the next branching.  `$` re-levels
    # the frame so the branching plane stays perpendicular to the
    # vertical plane through the mother branch, per his assumption (d).
    return f"""
        #title {title}
        #draw F
        #angle 30
        #iters {n} {n + 3}
        #define t1 {t1}
        #define t2 {t2}
        #define R1 {r1}
        #define R2 {r2}
        #define alpha {alpha}
        axiom: !(1)F(1)A(1,1)
        p1: A(s,d) : s > 0.01 -> $/(alpha)[+(t1*d)!(s*R1)F(s*R1)A(s*R1,0-d)][+(t2*d)!(s*R2)F(s*R2)A(s*R2,0-d)]
    """


# Nakamasu & Higaki (2019) Figure 1e-j: Honda's I-model at three
# phyllotaxes, with and without length contraction at ratio 0.8.
_NH2019 = {
    "NH_BIFURCATE": (90.0, 45.0, 1.00, "2019 fig 1e: bifurcation, equal lengths"),
    "NH_BIFURCATE_R": (90.0, 45.0, 0.80, "2019 fig 1h: bifurcation, ratio 0.8"),
    "NH_ALTERNATE": (137.5, 45.0, 1.00, "2019 fig 1f: alternate phyllotaxis"),
    "NH_ALTERNATE_R": (137.5, 45.0, 0.80, "2019 fig 1i: alternate, ratio 0.8"),
    "NH_OPPOSITE": (90.0, 45.0, 1.00, "2019 fig 1g: opposite phyllotaxis"),
    "NH_OPPOSITE_R": (90.0, 45.0, 0.80, "2019 fig 1j: opposite, ratio 0.8"),
}


def _nh_grammar(key):
    div, br, ratio, title = _NH2019[key]
    depth = 7 if ratio < 1.0 else 6
    return f"""
        #title {title}
        #draw F
        #angle {br}
        #iters {depth} {depth + 2}
        #define div {div}
        #define br {br}
        #define R {ratio}
        axiom: F(1)A(1)
        p1: A(s) : s > 0.02 -> $/(div)[+(br)F(s*R)A(s*R)][-(br)F(s*R)A(s*R)]
    """


TREES = {}
for _k in _TABLE1:
    TREES[_k] = _table1_grammar(_k)
for _k in _HONDA:
    TREES[_k] = _honda_grammar(_k)
for _k in _NH2019:
    TREES[_k] = _nh_grammar(_k)


def honda_params(key):
    """(theta1, theta2, R1, R2, alpha, N) for a Honda plate -- exposed so
    `fractal_tree_generator` can drive its native Honda mode from the
    same verified numbers rather than a second copy of the table."""
    t1, t2, r1, r2, alpha, n, _title = _HONDA[key]
    return t1, t2, r1, r2, alpha, n


def table1_params(key):
    """The 11-tuple of course-notes Table 1 row `key`."""
    return _TABLE1[key]


HONDA_KEYS = tuple(_HONDA)
TABLE1_KEYS = tuple(_TABLE1)
