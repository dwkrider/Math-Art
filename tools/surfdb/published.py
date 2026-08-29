"""Independently published invariants, keyed by slug.

Contract, shared by ferreol.py and vmm.py:

    records()  -> {slug: spec}   (unused here -- no new records)
    ids()      -> {slug: {...}}  (unused here -- no new cross-references)

published.py exports the third hook:

    invariants_for(slug) -> {...} deep-merged onto the record, carrying
      INDEPENDENTLY PUBLISHED values (genus per cell, quotient genus,
      end counts and types, total curvature, node counts, closed-form
      area/volume) that the drive stage can then compare against what
      the generator actually produces.  Every value below was read from
      the literature or from a local mirror of it -- NEVER measured from
      this repository's own build, because a value measured from the
      code cannot catch a bug in the code.

Sources actually consulted (all quotes verbatim from these):

  [WEBER]  M. Weber, minimalsurfaces.blog, "The repository" -- local
           mirror at S:/data/math_art/references/websites/minsurf/book/
           (repository/chNNN_*.md and posts/chNNN_*.md).  Cited below
           as (minsurf chNNN).
  [SCHOEN] A. Schoen, schoengeometry.com TPMS pages -- local mirror at
           S:/data/math_art/references/websites/schoen/book/main/
           ch006_e_tpms.md: "P, D, and G are each of genus three, which
           is the minimum possible genus for a triply-periodic minimal
           surface"; I-WP "of genus 4".
  [MC]     R. Ferreol, mathcurve.com -- local mirror at
           S:/data/math_art/references/websites/mathcurve/book/surfaces/.
  [BLvS]   S. Breske, O. Labs, D. van Straten, "Real Line Arrangements
           and Surfaces with Many Real Nodes" (2005) -- converted copy in
           research/papers/algebraic-surfaces/; its mu(d) table gives the
           maximum known node counts 4, 16, 31, 65, 99, 168, 216, 345,
           425, 600 for degrees 3..12 (exact, upper == lower, for d <= 6).
  [GE]     J. Garcia-Escudero, "A Construction of Algebraic Surfaces
           with Many Real Nodes" (arXiv:1107.3401, 2011) -- converted
           copy alongside [BLvS]; independently restates the Barth 65 /
           345, Labs 99, Endrass 168 and Sarti 600 records.
  [IMAG]   IMAGINARY (imaginary.org) -- local mirror; ch934:
           "Eugenio Giuseppe Togliatti proved in 1937, that there is a
           surface of degree 5 - a Quintic - with exactly 31
           singularities, which was world record at that time."
  [CHM89]  M. Callahan, D. Hoffman, W. H. Meeks III, "Embedded minimal
           surfaces with an infinite number of ends", Invent. Math. 96
           (1989) 459-505: M_k/T has genus 2k+1 and two ends.
  [WEY08]  A. G. Weyhaupt, "Deformations of the gyroid and Lidinoid
           minimal surfaces", Pacific J. Math. 235 (2008) 137-171:
           "The gyroid and Lidinoid are triply periodic minimal surfaces
           of genus three ... the unique embedded members of the
           associate families of the Schwarz P and H surfaces."

Where a published value DISAGREES with another source or with the
shipped implementation, the disagreement is recorded as a
`cross_checked` entry with `agrees: false` rather than silently picking
a side -- the same standard data/polyhedra applies to its two McCooey
disagreements.

Citation plumbing: `deep_merge` replaces lists wholesale, so this module
never blindly writes `provenance.sources`.  Where curation.facts_for()
supplies the record's sources (curation runs immediately before this
module in surfdb_build.curate(), so at merge time the record's sources
ARE curation's list), the citation is appended to that list; everywhere
else it becomes a `provenance.cross_checked` entry, and measure objects
and singularity entries carry their citations in their own `note`.
"""

import copy
import glob
import os

from . import curation, expr


def records():
    return {}


def ids():
    return {}


# ---------------------------------------------------------------------------
# helpers

def _m(exact, note):
    """A measure object whose value is evaluated FROM its exact form,
    so the validator's exact-vs-value gate cannot disagree by
    construction."""
    return {"exact": exact, "value": expr.evaluate(exact),
            "source": "classical", "note": note}


def _ck(source, detail, agrees=True):
    return {"source": source, "agrees": agrees, "detail": detail}


# ---------------------------------------------------------------------------
# The invariant table: slug -> fragment deep-merged onto the record.
#
# Conventions (matching the existing records):
#   * TPMS: topology.genus_per_cell = genus of the quotient by the
#     translation lattice of the ORIENTED surface (Schwarz P is 3, not
#     the 7 of a conventional cubic cell of I-WP); chi_per_cell = 2-2g.
#   * Singly/doubly periodic: topology.genus = genus of the quotient by
#     the period(s), topology.ends = ends OF THE QUOTIENT (the pattern
#     riemann-minimal-example and the record names already use).
#   * Complete finite-total-curvature: euler_characteristic is that of
#     the punctured surface, chi = 2 - 2g - #ends (the convention the
#     costa-surface record already uses: -3 = 2 - 2 - 3).

INVARIANTS = {
    # ---------------- triply periodic (genus per lattice cell) ----------
    "schwarz-p": {
        "topology": {"genus_per_cell": 3, "euler_characteristic_per_cell": -4},
    },
    "schwarz-d": {
        "topology": {"genus_per_cell": 3, "euler_characteristic_per_cell": -4},
    },
    "gyroid": {
        "topology": {"genus_per_cell": 3, "euler_characteristic_per_cell": -4},
    },
    "neovius-surface": {
        "topology": {"genus_per_cell": 9, "euler_characteristic_per_cell": -16},
    },
    "h-exact": {
        "topology": {"genus_per_cell": 3, "euler_characteristic_per_cell": -4},
    },
    "clp-exact": {
        "topology": {"genus_per_cell": 3, "euler_characteristic_per_cell": -4},
    },
    "clp-handle-exact": {
        "topology": {"genus_per_cell": 4, "euler_characteristic_per_cell": -6},
    },
    "iwp-surface": {
        "topology": {"genus_per_cell": 4, "euler_characteristic_per_cell": -6},
    },
    "frd-surface": {
        "topology": {"genus_per_cell": 6, "euler_characteristic_per_cell": -10},
    },
    "lidinoid": {
        "topology": {"genus_per_cell": 3, "euler_characteristic_per_cell": -4},
    },
    "lidinoid-exact": {
        "topology": {"genus_per_cell": 3, "euler_characteristic_per_cell": -4},
    },
    "schoen-i6": {
        "topology": {"genus_per_cell": 5, "euler_characteristic_per_cell": -8},
    },
    "schoen-frd-r": {
        "topology": {"genus_per_cell": 5, "euler_characteristic_per_cell": -8},
    },
    "triply-periodic-costa": {
        "topology": {"genus_per_cell": 4, "euler_characteristic_per_cell": -6},
        "embedding": {"quality": "embedded"},
    },

    # ---------------- classical / finite total curvature ---------------
    "costa-surface": {
        "curvature": {"total_curvature": {
            "note": "degree-3 Gauss map; C. J. Costa, 'Classification of "
                    "complete minimal surfaces in R^3 with total curvature "
                    "12pi', Invent. Math. 105 (1991) 273-303 -- the value is "
                    "in the paper's own title (sign convention: integral of "
                    "K dA = -12pi)."}},
    },
    "catenoid": {
        "curvature": {"total_curvature": {
            "note": "degree-1 Gauss map; M. Weber (minsurf ch189): 'Besides "
                    "the catenoid it is the only complete minimal surface of "
                    "finite total curvature -4pi' (said of Enneper's "
                    "surface)."}},
    },
    "enneper-surface": {
        "curvature": {"total_curvature": {
            "note": "M. Weber (minsurf ch189): 'Besides the catenoid it is "
                    "the only complete minimal surface of finite total "
                    "curvature -4pi.'  Single end of winding number 3, not "
                    "embedded."}},
    },
    "chen-gackstatter": {
        "curvature": {"total_curvature": {
            "note": "M. Weber (minsurf ch058): 'The Chen-Gackstatter surface "
                    "of genus g has finite total curvature -4pi(g+1), the "
                    "largest possible value for its genus'; g = 1 here."}},
    },
    "lopez-minimal-klein-bottle": {
        "curvature": {"total_curvature": _m(
            "-8*pi",
            "F. Lopez, 'A Complete Minimal Klein Bottle in R^3', Duke Math. "
            "J. 71 (1993) 23-30.  M. Weber (minsurf ch063): 'It has a single "
            "Enneper end and total curvature -8pi. Lopez proved later that "
            "it is the only minimal Klein bottle of total curvature -8pi.'")},
        "topology": {"orientable": False,
                     "ends": [{"type": "enneper", "count": 1,
                               "embedded": False}]},
    },
    "sphere-catenoid-enneper-end": {
        "curvature": {"total_curvature": _m(
            "-8*pi",
            "M. Weber (minsurf ch195): 'These are complete minimal surfaces "
            "of finite total curvature -8pi and were completely classified "
            "by Francisco Lopez.'  F. Lopez, Trans. Amer. Math. Soc. 334 "
            "(1992) 49-73.")},
        "topology": {"genus": 0,
                     "ends": [{"type": "catenoidal", "count": 1},
                              {"type": "enneper", "count": 1}]},
    },
    "meeks-mobius-strip": {
        "curvature": {"total_curvature": _m(
            "-6*pi",
            "W. H. Meeks III, 'The classification of complete minimal "
            "surfaces in R^3 with total curvature greater than -8pi', Duke "
            "Math. J. 48 (1981) 523-535 -- the complete minimal Moebius "
            "strip with total curvature -6pi; unique up to rigid motion by "
            "J. L. Barbosa and A. G. Colares (1986).")},
        "topology": {"orientable": False},
    },
    "double-enneper": {
        "topology": {"genus": 0,
                     "ends": [{"type": "enneper", "count": 2,
                               "embedded": False}]},
    },
    "costa-wohlgemuth-4-ends": {
        "topology": {"genus": 2, "euler_characteristic": -6,
                     "ends": [{"type": "catenoidal", "count": 2},
                              {"type": "planar", "count": 2}]},
    },
    "wohlgemuth-second-surface-genus-3": {
        "topology": {"genus": 3, "euler_characteristic": -8,
                     "ends": [{"type": "catenoidal", "count": 2},
                              {"type": "planar", "count": 2}]},
    },
    "henneberg-surface": {
        "topology": {"orientable": False,
                     "ends": [{"type": "enneper", "count": 1}]},
    },
    "kusner-projective-plane-p-planar-ends": {
        "topology": {"orientable": False},
        "embedding": {"quality": "immersed"},
    },

    # ---------------- singly periodic (quotient) -----------------------
    "callahan-hoffman-meeks-singly-periodic": {
        "topology": {"genus": 3,
                     "ends": [{"type": "planar", "count": 2}]},
        "embedding": {"quality": "embedded"},
    },
    "costa-scherk-tower-genus-1": {
        "topology": {"genus": 1,
                     "ends": [{"type": "annular", "count": 6}]},
    },
    "six-ended-scherk-tower": {
        "topology": {"genus": 0,
                     "ends": [{"type": "annular", "count": 6}]},
    },
    "six-ended-scherk-tower-genus-1": {
        "topology": {"genus": 1,
                     "ends": [{"type": "annular", "count": 6}]},
    },
    "eight-ended-scherk-tower-genus-2": {
        "topology": {"genus": 2,
                     "ends": [{"type": "annular", "count": 8}]},
    },
    "scherk-saddle-tower": {
        "topology": {"genus": 0},
    },
    "saddle-tower-karcher-unequal-wings": {
        "topology": {"genus": 0},
    },
    "helicoidal-karcher-scherk-twisted-tower": {
        "topology": {"genus": 0},
    },
    "helicoid-with-handle-genus-1": {
        "topology": {"genus": 1},
        "embedding": {"quality": "embedded"},
    },
    "catenoid-tower-with-handle-genus-2": {
        "topology": {"genus": 2},
    },
    "catenoid-tower-with-2-handles-genus-3": {
        "topology": {"genus": 3},
    },
    "alternating-fence-of-half-catenoids": {
        "topology": {"genus": 1,
                     "ends": [{"type": "catenoidal", "count": 2}]},
    },
    "translation-invariant-costa": {
        "topology": {"genus": 1,
                     "ends": [{"type": "annular", "count": 2},
                              {"type": "catenoidal", "count": 2}]},
    },
    "periodic-enneper": {
        "curvature": {"total_curvature": _m(
            "-4*pi",
            "Per translational quotient.  M. Weber (minsurf ch134): 'The "
            "surface belongs to the class of translation invariant complete "
            "minimal surfaces which have total curvature -4pi in the "
            "quotient.'  D. Freese, M. Weber, J. Geom. 108 (2017) "
            "743-762.")},
        "topology": {"ends": [{"type": "enneper", "count": 1},
                              {"type": "annular", "count": 1}]},
    },
    "torus-with-2-enneper-2-annular-ends": {
        "topology": {"genus": 1,
                     "ends": [{"type": "annular", "count": 2},
                              {"type": "enneper", "count": 2}]},
    },
    "torus-with-catenoid-2-annular-ends": {
        "topology": {"genus": 1,
                     "ends": [{"type": "catenoidal", "count": 1},
                              {"type": "annular", "count": 2}]},
    },
    "dasilva-batista-surface-genus-2": {
        "topology": {"genus": 2,
                     "ends": [{"type": "annular", "count": 8}]},
    },

    # ---------------- doubly periodic (quotient) -----------------------
    "scherk-doubly-periodic": {
        "topology": {"genus": 0,
                     "ends": [{"type": "annular", "count": 4}]},
        "embedding": {"quality": "embedded"},
    },
    "karcher-meeks-rosenberg-doubly-periodic": {
        "topology": {"genus": 1,
                     "ends": [{"type": "annular", "count": 4}]},
        "embedding": {"quality": "embedded"},
    },
    "karcher-meeks-rosenberg-kmr-3-tilted-ends": {
        "topology": {"genus": 1,
                     "ends": [{"type": "annular", "count": 4}]},
    },
    "wei-doubly-periodic-genus-2": {
        "topology": {"genus": 2,
                     "ends": [{"type": "annular", "count": 4}]},
        "embedding": {"quality": "embedded"},
    },

    # ---------------- algebraic node counts ----------------------------
    "togliatti-quintic": {
        "embedding": {
            "quality": "singular",
            "is_record": True,
            "record_for": "maximum nodes, degree 5",
            "singularities": [{
                "type": "node (A1)", "count": 31,
                "note": "IMAGINARY (imaginary.org, local mirror ch934): "
                        "'Eugenio Giuseppe Togliatti proved in 1937, that "
                        "there is a surface of degree 5 - a Quintic - with "
                        "exactly 31 singularities, which was world record at "
                        "that time.'  Breske-Labs-van Straten (2005), mu(5) "
                        "table: upper and lower bounds both 31, so the count "
                        "is exact (optimality is Beauville's).",
            }],
        },
    },
    "boys-surface": {
        "embedding": {"singularities": [{
            "type": "triple point", "count": 1,
            "note": "Self-intersection triple point of the immersion (not a "
                    "surface singularity).  mathcurve.com, Boy surface "
                    "(mirror ch1216): the three tangents at the triple "
                    "point are pairwise orthogonal; the self-intersection "
                    "curve meets itself at the single triple point.",
        }]},
    },

    # ---------------- closed-form metrics ------------------------------
    "pseudosphere": {
        "metrics": {
            "area": _m(
                "4*pi",
                "Complete two-horned pseudosphere (full tractrix revolved), "
                "scale parameter a = 1: area 4*pi*a^2, equal to the sphere "
                "of the same radius (classical, Huygens 1693; mathcurve "
                "pseudosphere page, mirror ch1210: 'area: 4pi a^2').  A "
                "single tractroid horn has half this, 2*pi."),
            "volume_enclosed": _m(
                "2*pi/3",
                "Both horns, a = 1: (2/3)*pi*a^3, half the ball of the same "
                "radius (mathcurve mirror ch1210: 'Volume: 2/3 pi a^3').  "
                "One horn encloses pi/3."),
        },
    },
    "oloid": {
        "metrics": {
            "normalization": "unit_radius",
            "area": _m(
                "4*pi",
                "H. Dirnboeck, H. Stachel, 'The development of the oloid', "
                "J. Geom. Graphics 1 (1997) 105-118: the oloid's surface "
                "area equals that of the sphere with the same generating- "
                "circle radius r, A = 4*pi*r^2; r = 1 here."),
            "measures_note": "Area for generating-circle radius 1 "
                             "(Dirnboeck-Stachel 1997).  The enclosed "
                             "volume has no comparably simple closed form "
                             "and is not recorded here.",
        },
    },

    # The one surface added by the algebraic transcription pass that had
    # NO curated record to merge into, and so would otherwise have been
    # the only member of the many-nodal octic family carrying an
    # implementation and no published count to check it against.  The
    # count is the only reason the surface is interesting, so a record
    # without it is a record of nothing.
    "van-straten-octic-124-nodes": {
        "discovered_by": "Duco van Straten",
        "embedding": {
            "quality": "singular",
            "singularities": [
                {"type": "node (A1)", "count": 124,
                 "note": "Per the mirrored vstrconstr page.  Of these, "
                         "96 are REAL and affine, at closed-form "
                         "octagon-trigonometric positions "
                         "(rho in {tan(pi/8), 1, 1+sqrt(2), ...}); the "
                         "remainder are complex or at infinity.  A "
                         "finite numerical count is therefore expected "
                         "to stop at 96, and that shortfall is "
                         "consistent rather than a discrepancy."}]},
        "symmetry": {"kind": "point", "periodicity_rank": 0,
                     "schoenflies": "D8h", "verified_by": "curated"},
        "definition": {"degree": 8},
        "relations": {"member_of": "van-straten-dihedral-series"},
        "provenance": {"sources": [
            "D. van Straten's D8 x Z2-symmetric octic construction, as "
            "mirrored on O. Labs's Algebraic Surface Homepage "
            "(vstrconstr page), which prints the equation in full.",
            "The octics survey on the same site: Miyaoka's bound for an "
            "octic is 174 and the best known count is 168, which is the "
            "scale this 124 sits on.",
        ]},
    },
}


# ---------------------------------------------------------------------------
# Citations to APPEND to provenance.sources -- applied only when
# curation.facts_for() supplies the sources list (see module docstring);
# otherwise they fall through to cross_checked entries.

CITATIONS = {
    "schwarz-p": [
        "A. Schoen, schoengeometry.com TPMS pages (mirror: schoen ch006): "
        "'P, D, and G are each of genus three, which is the minimum "
        "possible genus for a triply-periodic minimal surface' -- "
        "independent genus-per-cell oracle.",
    ],
    "schwarz-d": [
        "M. Weber, minimalsurfaces.blog, 'Schwarz D Surface' (mirror: "
        "minsurf ch308): 'The period lattice is the face-centered cubic "
        "lattice, and its quotient is a compact Riemann surface of genus "
        "3' -- independent genus-per-cell oracle.",
    ],
    "gyroid": [
        "A. G. Weyhaupt, 'Deformations of the gyroid and Lidinoid minimal "
        "surfaces', Pacific J. Math. 235 (2008) 137-171: genus three, no "
        "straight lines or planar symmetry curves; the unique embedded "
        "member of the associate family of Schwarz P.",
    ],
    "neovius-surface": [
        "M. Weber, minimalsurfaces.blog, 'Neovius Surface' (mirror: "
        "minsurf ch275): the X-pieces 'extend to the edges of a cube, "
        "making it a genus 9 surface' -- independent genus-per-cell "
        "oracle.",
    ],
    "lidinoid": [
        "A. G. Weyhaupt, 'Deformations of the gyroid and Lidinoid minimal "
        "surfaces', Pacific J. Math. 235 (2008) 137-171: 'The gyroid and "
        "Lidinoid are triply periodic minimal surfaces of genus three "
        "... the unique embedded members of the associate families of the "
        "Schwarz P and H surfaces.'",
    ],
    "costa-surface": [
        "C. J. Costa, 'Classification of complete minimal surfaces in R^3 "
        "with total curvature 12pi', Invent. Math. 105 (1991) 273-303 -- "
        "independent confirmation of the recorded -12pi.",
    ],
    "catenoid": [
        "M. Weber, minimalsurfaces.blog (mirror: minsurf ch189, ch202, "
        "ch222): total curvature -4pi; 'the catenoid is a 2-Noid (in "
        "fact, the only one)'.",
    ],
    "enneper-surface": [
        "A. Enneper, 'Weitere Bemerkungen ueber asymptotische Linien', "
        "Nachr. Koenigl. Ges. Wiss. Goettingen 15 (1871); M. Weber "
        "(mirror: minsurf ch189): total curvature -4pi, single end of "
        "winding number 3, not embedded.",
    ],
    "chen-gackstatter": [
        "C. C. Chen, F. Gackstatter, Math. Ann. 259 (1982) 359-369; M. "
        "Weber (mirror: minsurf ch058): total curvature -4pi(g+1) for the "
        "genus-g member, 'the largest possible value for its genus'.",
    ],
    "scherk-saddle-tower": [
        "H. Karcher, 'Embedded minimal surfaces derived from Scherk's "
        "examples', Manuscripta Math. 62 (1988) 83-114 -- 'translation "
        "invariant minimal surface with genus 0 in the quotient' (as "
        "quoted at minsurf ch140).",
        "J. Perez, M. Traizet, 'The classification of singly periodic "
        "minimal surfaces with genus zero and Scherk type ends', Trans. "
        "Amer. Math. Soc. 359 (2007) 965-990.",
    ],
    "riemann-minimal-example": [
        "B. Riemann, 'Ueber die Flaeche vom kleinsten Inhalt bei "
        "gegebener Begrenzung', Abh. Koenigl. Ges. Wiss. Goettingen 13 "
        "(1867) 3-52.",
        "W. H. Meeks III, J. Perez, A. Ros, 'Properly embedded minimal "
        "planar domains', Ann. of Math. 181 (2015) 473-546 -- any "
        "properly embedded minimal surface of genus 0 is the plane, "
        "catenoid, helicoid, or a Riemann example.",
    ],
    "scherk-doubly-periodic": [
        "H. F. Scherk, 'Bemerkungen ueber die kleinste Flaeche innerhalb "
        "gegebener Grenzen', J. Reine Angew. Math. 13 (1835) 185-208.",
        "H. Lazard-Holly, W. H. Meeks III, 'Classification of "
        "doubly-periodic minimal surfaces of genus zero', Invent. Math. "
        "143 (2001) 1-27; M. Weber (mirror: minsurf ch046): 'The quotient "
        "surface by its translational symmetries is a 4-punctured "
        "sphere.'",
    ],
    "jorge-meeks-k-noid": [
        "L. P. Jorge, W. H. Meeks III, 'The topology of complete minimal "
        "surfaces of finite total Gaussian curvature', Topology 22 (1983) "
        "203-221; M. Weber (mirror: minsurf ch222): 'A k-Noid is a "
        "minimal surface with k catenoidal ends.'",
    ],
    "kummer-quartic": [
        "mathcurve.com, Kummer surface (mirror ch1294): 16 ordinary "
        "singular points at the tetrahedral vertices; Breske-Labs-van "
        "Straten (2005) mu(4) = 16 (exact) -- independent node-count "
        "oracle.",
    ],
    "cayley-nodal-cubic": [
        "Breske-Labs-van Straten, 'Real Line Arrangements and Surfaces "
        "with Many Real Nodes' (2005), mu(3) = 4 (exact) -- independent "
        "node-count oracle.",
    ],
    "barth-sextic": [
        "Breske-Labs-van Straten (2005) mu(6) table and J. "
        "Garcia-Escudero (arXiv:1107.3401, 2011): Barth's 65-nodal "
        "sextic -- independent confirmations of the 65.",
    ],
    "barth-decic": [
        "Breske-Labs-van Straten (2005) and J. Garcia-Escudero "
        "(arXiv:1107.3401, 2011): Barth's 345-nodal decic -- independent "
        "confirmations of the 345.",
    ],
    "labs-septic": [
        "O. Labs, 'A septic with 99 real nodes' (2004; converted copy in "
        "research/papers/algebraic-surfaces/) and Breske-Labs-van Straten "
        "(2005) mu(7) >= 99 -- independent confirmations of the 99.",
    ],
    "endrass-octic": [
        "S. Endrass, 'A projective surface of degree eight with 168 "
        "nodes' (1995; converted copy in research/papers/algebraic-"
        "surfaces/) and Breske-Labs-van Straten (2005) mu(8) >= 168 -- "
        "independent confirmations of the 168.",
    ],
    "boys-surface": [
        "mathcurve.com, Boy surface (mirror ch1216): single triple point "
        "with pairwise-orthogonal tangents; self-intersection curve of "
        "three loops through it.",
    ],
    "pseudosphere": [
        "mathcurve.com, pseudosphere (mirror ch1210): 'Volume: 2/3 pi "
        "a^3; area: 4 pi a^2' -- the classical Huygens values.",
    ],
    "oloid": [
        "H. Dirnboeck, H. Stachel, 'The development of the oloid', "
        "J. Geom. Graphics 1 (1997) 105-118 -- area 4*pi*r^2.",
    ],
    "henneberg-surface": [
        "E. L. Henneberg, 'Ueber diejenige Minimalflaeche, welche die "
        "Neil'sche Parabel zur ebenen geodaetischen Linie hat' (1876); "
        "M. Weber (mirror: minsurf posts ch384): parametrizes a "
        "projective plane, 'the puncture at 0 represents an Enneper "
        "end'.",
    ],
}


# ---------------------------------------------------------------------------
# cross_checked entries: citations for records whose sources list this
# module cannot safely extend, plus every DISAGREEMENT found while
# compiling the table.

CROSS_CHECKS = {
    # -- TPMS ------------------------------------------------------------
    "h-exact": [_ck(
        "minimalsurfaces.blog (minsurf ch315, Schwarz H)",
        "'Like his other examples, this surface has genus 3 when divided "
        "by its translational symmetries.'  H. A. Schwarz, Gesammelte "
        "Mathematische Abhandlungen (1890).")],
    "clp-exact": [_ck(
        "minimalsurfaces.blog (minsurf ch297, Schwarz CLP)",
        "Genus 3 per translational quotient; 'the conjugate surfaces are "
        "also triply periodic and of genus 3.'")],
    "clp-handle-exact": [_ck(
        "minimalsurfaces.blog (minsurf ch264, CLP with handle)",
        "'This gives a 1-parameter family of triply periodic surfaces of "
        "genus 4.'")],
    "iwp-surface": [_ck(
        "minimalsurfaces.blog (minsurf ch284, Schoen I-WP)",
        "'[Schoen's] I-WP surface from 1970 is a triply periodic minimal "
        "surface of genus 4. ... After identifying opposite faces of a "
        "cubical cell the surface has genus 7, but half of such a cell "
        "already constitutes a translational fundamental domain, bringing "
        "the genus down to 4' -- the stored 4 is per translational "
        "fundamental domain, NOT per conventional cubic cell.  Also A. "
        "Schoen, schoengeometry.com (mirror schoen ch006): 'a fourth "
        "surface (I-WP) of genus 4'.")],
    "frd-surface": [_ck(
        "minimalsurfaces.blog (minsurf ch281, Schoen F-RD)",
        "'[The] FR-D surface extends handles towards the faces of a "
        "rhombic dodecahedron (or, the edges of a cube) and hence has "
        "genus 6.'  A. H. Schoen, NASA TN D-5541 (1970).")],
    "lidinoid-exact": [_ck(
        "A. G. Weyhaupt, Pacific J. Math. 235 (2008) 137-171",
        "'The gyroid and Lidinoid are triply periodic minimal surfaces of "
        "genus three ... the unique embedded members of the associate "
        "families of the Schwarz P and H surfaces.'  S. Lidin, S. "
        "Larsson, J. Chem. Soc. Faraday Trans. 86 (1990).")],
    "schoen-i6": [_ck(
        "minimalsurfaces.blog (minsurf ch285, Schoen I6)",
        "'It extends triply periodically to a surface of genus 5.'")],
    "schoen-frd-r": [_ck(
        "minimalsurfaces.blog (minsurf ch290, Schoen's unnamed surface "
        "#12, F-RD-r)",
        "'This genus 5 surface can be interpreted as a twisted version of "
        "the oS deformation.'")],
    "triply-periodic-costa": [_ck(
        "minimalsurfaces.blog (minsurf ch333, triply periodic "
        "Costa-Scherk)",
        "'A 1-parameter family of embedded triply periodic minimal "
        "surfaces of genus 4 with horizontal straight lines and vertical "
        "symmetry planes.'  D. Freese, M. Weber, A. T. Yerger, R. Yol, "
        "'Two New Embedded Triply Periodic Minimal Surfaces of Genus 4' "
        "(arXiv).")],

    # -- classical / FTC -------------------------------------------------
    "chen-gackstatter-higher-genus": [_ck(
        "minimalsurfaces.blog (minsurf ch058) + K. Sato, Tohoku Math. J. "
        "48 (1996) 229-246; M. Weber, M. Wolf; E. Thayer, Experiment. "
        "Math. 4 (1995)",
        "Published family law for the drive stage: 'The Chen-Gackstatter "
        "surface of genus g has finite total curvature -4pi(g+1), the "
        "largest possible value for its genus', with 'one Enneper end of "
        "winding number 3' -- per-specimen total curvature must follow "
        "-4pi(g+1).")],
    "double-enneper": [_ck(
        "minimalsurfaces.blog (minsurf ch176, Double Enneper)",
        "'We have a sphere with two Enneper ends' (Karcher's Tokyo "
        "Notes); not embedded.")],
    "costa-wohlgemuth-4-ends": [_ck(
        "M. Wohlgemuth, Arch. Rational Mech. Anal. 137 (1997) 1-25 "
        "(minsurf ch053)",
        "'a 4-ended embedded minimal surface of genus 2 ... It has two "
        "catenoidal and two planar ends' (CSSCFF); NOTE 'an embeddedness "
        "proof is still lacking' per the same page.")],
    "wohlgemuth-second-surface-genus-3": [_ck(
        "M. Wohlgemuth, Dissertation Bonn 1993; Arch. Rational Mech. "
        "Anal. 137 (1997) 1-25 (minsurf posts ch418)",
        "'Wohlgemuth was able to add a handle to his 4-ended "
        "Costa-Wohlgemuth surface of genus 2, increasing the genus to 3'; "
        "end types inherited from CSSCFF (2 catenoidal + 2 planar).")],
    "lopez-minimal-klein-bottle": [_ck(
        "F. Lopez, Duke Math. J. 71 (1993) 23-30 (minsurf ch063)",
        "'It has a single Enneper end and total curvature -8pi ... the "
        "only minimal Klein bottle of total curvature -8pi.'")],
    "sphere-catenoid-enneper-end": [_ck(
        "F. Lopez, Trans. Amer. Math. Soc. 334 (1992) 49-73 (minsurf "
        "ch195)",
        "'Complete minimal surfaces of finite total curvature -8pi', one "
        "catenoidal + one Enneper end on a sphere.")],
    "meeks-mobius-strip": [_ck(
        "W. H. Meeks III, Duke Math. J. 48 (1981) 523-535",
        "The complete minimal Moebius strip with total curvature -6pi; "
        "by the same classification every complete minimal surface with "
        "total curvature > -8pi is the plane, catenoid, Enneper surface "
        "or this Moebius strip.  Uniqueness: Barbosa-Colares 1986.")],
    "kusner-projective-plane-p-planar-ends": [_ck(
        "R. Kusner, Bull. Amer. Math. Soc. 17 (1987) 291-295 (minsurf "
        "ch065)",
        "'immersed minimal spheres with an even number 2n of planar ends. "
        "If n is odd ... an immersion of the projective plane with n "
        "planar ends.'")],
    "finite-riemann-plane-2-catenoids": [_ck(
        "D. Hoffman, H. Karcher, 'Complete embedded minimal surfaces of "
        "finite total curvature' (1995) (minsurf ch190)",
        "'These surfaces are never embedded. This is an illustration of "
        "the Lopez-Ros theorem which asserts that a complete, embedded "
        "minimal sphere of finite total curvature must be the plane or "
        "catenoid.'")],
    "symmetrized-finite-riemann-2m-catenoids": [_ck(
        "minimalsurfaces.blog (minsurf ch228)",
        "'This symmetrization of the Finite Riemann minimal surface has "
        "one planar end and 2n catenoidal ends'; 'By the Lopez-Ros "
        "theorem, they can never be embedded.'")],
    "symmetrized-chen-gackstatter-k-fold-genus-k-1": [_ck(
        "minimalsurfaces.blog (minsurf ch224)",
        "Published family law: 'Their symmetries are generated by "
        "reflections at n vertical planes and rotations about n "
        "horizontal lines. The genus is n-1, and there is only one "
        "end.'")],

    # -- singly periodic -------------------------------------------------
    "callahan-hoffman-meeks-singly-periodic": [
        _ck("M. Callahan, D. Hoffman, W. H. Meeks III, Invent. Math. 96 "
            "(1989) 459-505",
            "Primary source: M_k/T has genus 2k+1 and two ends (planar); "
            "k = 1 gives quotient genus 3.  The recorded genus 3 follows "
            "the paper."),
        _ck("minimalsurfaces.blog (minsurf ch128)",
            "DISAGREES with the primary source: the page says 'When "
            "divided by its translational symmetry, it has two planar "
            "ends and genus 2', but Invent. Math. 96 (1989) gives "
            "M_1/T genus 3 (2k+1, k=1), and the shipped implementation "
            "independently measures quotient genus 3 (chi = -6 with 2 "
            "ends).  The blog page appears to be off by one.",
            agrees=False),
    ],
    "callahan-hoffman-meeks-chm-1-2-genus-4": [_ck(
        "minimalsurfaces.blog (minsurf posts ch372, higher-genus CHM)",
        "UNRESOLVED between three statements, so no genus is asserted "
        "here: (a) the page's convention 'denote by CHM_r,s a translation "
        "invariant surface with 2r ends of genus s+1 in the quotient' "
        "makes CHM-(1,2) genus 3 with 2 ends; (b) the shipped "
        "implementation (math_art/minsurf/zoo.py) measures quotient genus "
        "4 with 2 planar ends (chi = -8); (c) Invent. Math. 96 (1989) "
        "covers only the dihedral family M_k (genus 2k+1), of which "
        "CHM-(1,2) is not a member.  Note the same page convention gives "
        "CHM-(1,1) genus 2, which contradicts the 1989 paper's genus 3 "
        "for M_1 -- so the blog's genus convention appears shifted by "
        "one, and the record's name '(genus 4)' agrees with the "
        "measurement.  Flagged for resolution against Weber's 'CHM 1999' "
        "manuscript.",
        agrees=False)],
    "screw-motion-chm-tower": [_ck(
        "minimalsurfaces.blog (minsurf ch156)",
        "'The translation invariant Callahan-Hoffman-Meeks Surfaces can "
        "be deformed into screw motion invariant surfaces' -- twist angle "
        "in (0, pi/2]; helicoidal parking-garage limit via the Traizet "
        "balance equation.  No quotient invariants are stated on the "
        "page.")],
    "costa-scherk-tower-genus-1": [_ck(
        "minimalsurfaces.blog (minsurf ch130, Costa-Scherk)",
        "'This 1-parameter family is a simple example of a 6-ended Scherk "
        "surface of genus 1 with Costa saddles' -- ends are annular "
        "Scherk-type in the quotient.")],
    "six-ended-scherk-tower": [_ck(
        "minimalsurfaces.blog (minsurf ch125, 6-ended Scherk g=0)",
        "'The embedded singly periodic Scherk surfaces of genus 0' with "
        "six annular ends; embedded examples shown for end angles 30 and "
        "89 degrees, 'increasing the angle to 90 and higher creates "
        "non-embedded surfaces'.")],
    "six-ended-scherk-tower-genus-1": [_ck(
        "minimalsurfaces.blog (minsurf ch126); K. Li, PhD thesis, "
        "Bloomington 2012",
        "'a 1-parameter subfamily (with some symmetry) of 6-ended "
        "surfaces of genus 1' with annular Scherk ends.")],
    "eight-ended-scherk-tower-genus-2": [_ck(
        "minimalsurfaces.blog (minsurf ch127, 8-ended Scherk g=2)",
        "'The surfaces on this page are translation invariant of genus 2 "
        "and have the coordinate planes as symmetry planes'; 8 Scherk "
        "(annular) ends per the page title.")],
    "saddle-tower-karcher-unequal-wings": [_ck(
        "H. Karcher, Manuscripta Math. 62 (1988); J. Perez, M. Traizet, "
        "Trans. Amer. Math. Soc. 359 (2007) 965-990 (minsurf ch140)",
        "'Hermann Karcher's Scherk Towers are translation invariant "
        "minimal surface with genus 0 in the quotient'; Perez-Traizet: "
        "'the only singly periodic minimal surfaces of genus 0 with "
        "annular ends.'")],
    "helicoidal-karcher-scherk-twisted-tower": [_ck(
        "H. Karcher, Manuscripta Math. 62 (1988) (minsurf ch132)",
        "Screw-motion invariant Karcher-Scherk surfaces: 'multivalued on "
        "the 2k-punctured quotient spheres' -- genus 0 in the quotient "
        "with 2k ends; twist limit 1/k.")],
    "helicoid-with-handle-genus-1": [_ck(
        "D. Hoffman, H. Karcher, F. Wei (1996), arXiv math/9605222 "
        "(minsurf posts ch409)",
        "'an embedded, translation invariant minimal surface asymptotic "
        "to the helicoid that has genus 1 in the quotient' (helicoidal "
        "ends).")],
    "catenoid-tower-with-handle-genus-2": [_ck(
        "minimalsurfaces.blog (minsurf ch162)",
        "'Here we add a simple handle to [the fence of catenoids], "
        "increasing the genus of the quotient surface to 2.'")],
    "catenoid-tower-with-2-handles-genus-3": [_ck(
        "minimalsurfaces.blog (minsurf ch163)",
        "'The quotient surfaces of this 1-parameter family have genus 3, "
        "and the period problem becomes 2-dimensional.'")],
    "alternating-fence-of-half-catenoids": [_ck(
        "minimalsurfaces.blog (minsurf posts ch406)",
        "'As a translation invariant surface, it has genus one and two "
        "catenoidal ends in the quotient.'")],
    "translation-invariant-costa": [_ck(
        "minimalsurfaces.blog (minsurf ch171)",
        "'It has genus 1 in the quotient, two annular and two catenoidal "
        "ends.'")],
    "periodic-enneper": [_ck(
        "D. Freese, M. Weber, J. Geom. 108 (2017) 743-762 (minsurf "
        "ch134)",
        "'a translation invariant surface with one periodic Enneper end "
        "and one annular flat end. ... It is not embedded'; total "
        "curvature -4pi in the quotient.")],
    "torus-with-2-enneper-2-annular-ends": [_ck(
        "minimalsurfaces.blog (minsurf posts ch413)",
        "'a translation invariant surface with two parallel annular ends "
        "and two periodic Enneper ends. The quotient surface has genus "
        "1.'")],
    "torus-with-catenoid-2-annular-ends": [_ck(
        "minimalsurfaces.blog (minsurf posts ch401)",
        "One catenoidal and two annular ends per the page title ('Singly "
        "Periodic Surfaces with One Catenoid and Two Annular Ends'); "
        "genus 1 per the page's own resource naming "
        "('Annular2Catenoidal1Genus1'); 'the limit of k-Noids of genus k "
        "for k to infinity'.")],
    "dasilva-batista-surface-genus-2": [_ck(
        "M. F. da Silva, V. Ramos Batista, 'Scherk Saddle Towers of "
        "Genus Two in R^3' (minsurf ch131)",
        "'This translation invariant surface ... features 8 annular ends "
        "and has genus 2.'  NOT the same surface as Lubeck-Batista "
        "(doubly periodic, genus 3, four annular ends -- minsurf "
        "ch038).")],
    "fence-of-catenoids-karcher": [_ck(
        "minimalsurfaces.blog (minsurf ch170); R. Schoen's two-ends "
        "theorem",
        "Karcher's Fence of Catenoids (Tokyo notes).  'Every complete, "
        "properly immersed minimal surface with two catenoidal ends is "
        "the catenoid' explains why the non-periodic handle version "
        "cannot close up.  The page states no quotient invariants.")],

    # -- doubly periodic -------------------------------------------------
    "tilted-scherk-doubly-periodic": [_ck(
        "minimalsurfaces.blog (minsurf ch048, Tilted Scherk)",
        "'The Lopez-Ros deformation tilts the ends of the doubly periodic "
        "Scherk surface ... the surfaces are no longer embedded' -- "
        "confirms the record's non-embedded-by-design claim.")],
    "karcher-meeks-rosenberg-doubly-periodic": [_ck(
        "H. Karcher (1988); W. H. Meeks III, H. Rosenberg, Invent. Math. "
        "97 (1989) 351-379; J. Perez, M. M. Rodriguez, M. Traizet, J. "
        "Diff. Geom. 69 (2005) 523-577 (minsurf ch035)",
        "'complete, embedded doubly periodic minimal surfaces with two "
        "top and two bottom parallel annular ends in the quotient'; "
        "Perez-Rodriguez-Traizet: 'the only embedded doubly periodic "
        "minimal surfaces with parallel ends of genus one.'")],
    "karcher-meeks-rosenberg-kmr-3-tilted-ends": [_ck(
        "minimalsurfaces.blog (minsurf ch035, KMR case 3)",
        "The tilted-ends member of the 3-parameter KMR family (genus 1, "
        "four parallel annular ends in the quotient); 'the surfaces "
        "resemble a doubly periodic version of Riemann's singly periodic "
        "surface.'")],
    "wei-doubly-periodic-genus-2": [_ck(
        "F. Wei, Invent. Math. 109 (1992) 113-136 (minsurf ch052)",
        "'It was the first genus 2 embedded doubly periodic minimal "
        "surface, adding a handle to one of the Karcher-Meeks-Rosenberg "
        "examples'; 4-ended (parallel annular ends inherited from "
        "KMR).")],
    "wei-higher-genus-tower-doubly-periodic": [_ck(
        "P. Connor, M. Weber, 'The construction of doubly periodic "
        "minimal surfaces via balance equations' (minsurf ch033)",
        "Family law, genus varies by member: 'Traizet's method yields "
        "solutions for all types (1,n) of genus n'; the (2,3) member has "
        "genus 4.  No single genus is asserted because the shipped row "
        "carries members of different genus.")],
    "karcher-scherk-with-handles-doubly-periodic": [_ck(
        "M. Weber, M. Wolf, 'Handle Addition for doubly-periodic Scherk "
        "Surfaces' (minsurf ch032/ch047)",
        "Genus-one member: page title 'The Doubly Periodic Karcher-Scherk "
        "Surface of Genus One' (annular ends); higher-genus members of "
        "genus 2, 3, 4 are shown at ch032.  No single genus is asserted "
        "because the shipped row carries members of different genus.")],
    "connor-experimental-doubly-periodic": [_ck(
        "P. Connor, PhD thesis 2009 (minsurf ch016/ch023)",
        "Numerically established doubly periodic surfaces: the "
        "experimental genus-3 series (pages 78-85) and an asymmetric "
        "genus-2 surface.  'All this has only been established only "
        "numerically.'  No single genus is asserted because the shipped "
        "row carries members of different genus.")],
    "rossman-thayer-wohlgemuth-doubly-periodic": [_ck(
        "W. Rossman, E. C. Thayer, M. Wohlgemuth, 'Embedded, Doubly "
        "Periodic Minimal Surfaces', Experimental Math. 9 (2000) 197-219 "
        "(minsurf ch040/ch045)",
        "M1+ has quotient genus 2 ('a surface of genus 2 they dub M1+, "
        "different from Fusheng Wei's genus 2 surface'); the M1+- member "
        "is 'a doubly periodic genus 3 surface with 4 parallel annular "
        "ends'.  No single genus is asserted because the shipped row "
        "carries both members.")],

    # -- algebraic -------------------------------------------------------
    "sarti-dodecic": [_ck(
        "A. Sarti, 'Pencils of symmetric surfaces in P^3', J. Algebra "
        "246 (2001); Breske-Labs-van Straten (2005) mu(12) >= 600; J. "
        "Garcia-Escudero (arXiv:1107.3401, 2011)",
        "'Sarti's degree-12 surface with mu_R = 600' -- confirms the "
        "recorded 600 nodes from two independent compilations.")],
    "togliatti-quintic": [_ck(
        "IMAGINARY, imaginary.org (mirror ch934); Breske-Labs-van "
        "Straten (2005)",
        "'Togliatti proved in 1937, that there is a surface of degree 5 "
        "- a Quintic - with exactly 31 singularities, which was world "
        "record at that time'; mu(5) = 31 is exact (both bounds) in the "
        "Breske-Labs-van Straten table.")],
}


# ---------------------------------------------------------------------------

def invariants_for(slug):
    inv = copy.deepcopy(INVARIANTS.get(slug, {}))
    cites = CITATIONS.get(slug) or []
    checks = list(CROSS_CHECKS.get(slug) or [])
    if cites:
        cur = ((curation.facts_for(slug) or {}).get("provenance") or {}) \
            .get("sources") or []
        if cur:
            # curate() applies curation immediately before this module,
            # so the record's sources ARE this list: extending it here is
            # a true append, never a clobber.
            merged = list(cur) + [c for c in cites if c not in cur]
            inv.setdefault("provenance", {})["sources"] = merged
        else:
            # no curated sources to extend -- carry the citations as
            # cross-check entries instead of overwriting stage-built
            # sources this module cannot see.
            checks = checks + [_ck(c, "citation for the published "
                                      "invariants on this record")
                               for c in cites]
    if checks:
        inv.setdefault("provenance", {})["cross_checked"] = \
            copy.deepcopy(checks)
    return inv


# ---------------------------------------------------------------------------

_END_TYPES = {"planar", "catenoidal", "helicoidal", "enneper", "annular",
              "flat", "cylindrical"}


def _iter_measures(node):
    if isinstance(node, dict):
        if "exact" in node and "value" in node:
            yield node
        for v in node.values():
            yield from _iter_measures(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_measures(v)


def _selftest():
    n_measures = 0
    slugs = set(INVARIANTS) | set(CITATIONS) | set(CROSS_CHECKS)

    # 1. every slug keys an actual record
    root = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "data", "surfaces", "surfaces"))
    if os.path.isdir(root):
        on_disk = {os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(root, "*", "*.json"))}
        missing = sorted(s for s in slugs if s not in on_disk)
        assert not missing, "slugs with no record: %s" % missing
    else:
        print("  (data/surfaces/surfaces absent -- slug-existence check "
              "skipped; run after surfdb_build)")

    for slug in sorted(slugs):
        frag = invariants_for(slug)
        assert frag, slug

        # 2. every emitted measure passes the validator's own gate
        for meas in _iter_measures(frag):
            ok, detail = expr.check_measure(meas)
            assert ok, "%s: bad measure %r (%s)" % (slug, meas, detail)
            n_measures += 1

        topo = frag.get("topology") or {}

        # 3. ends entries are schema-shaped
        for end in topo.get("ends") or []:
            assert end.get("type") in _END_TYPES, (slug, end)
            assert isinstance(end.get("count"), int) and end["count"] > 0, \
                (slug, end)

        # 4. Euler characteristics agree with the genus they accompany
        g = topo.get("genus_per_cell")
        e = topo.get("euler_characteristic_per_cell")
        if g is not None or e is not None:
            assert g is not None and e is not None and e == 2 - 2 * g, \
                "%s: chi_per_cell %r != 2-2g for g=%r" % (slug, e, g)
        g = topo.get("genus")
        chi = topo.get("euler_characteristic")
        if isinstance(g, int) and isinstance(chi, int):
            ends = sum(x["count"] for x in topo.get("ends") or [])
            assert chi == 2 - 2 * g - ends, \
                "%s: chi %r != 2-2g-ends (g=%r, ends=%r)" % \
                (slug, chi, g, ends)

        # 5. cross-check entries are schema-shaped
        for ck in (frag.get("provenance") or {}).get("cross_checked") or []:
            assert isinstance(ck.get("source"), str) and ck["source"], \
                (slug, ck)
            assert isinstance(ck.get("agrees"), bool), (slug, ck)

        # 6. singularity entries are schema-shaped
        for s in (frag.get("embedding") or {}).get("singularities") or []:
            assert isinstance(s.get("type"), str) and s["type"], (slug, s)
            assert isinstance(s.get("count"), int) and s["count"] > 0, \
                (slug, s)

    n_dis = sum(1 for entries in CROSS_CHECKS.values()
                for ck in entries if not ck["agrees"])
    print("RESULT: OK  (surfdb.published: %d slugs, %d measures checked, "
          "%d cross-check entries, %d recorded disagreements, "
          "%d records, %d id sets)"
          % (len(slugs), n_measures,
             sum(len(v) for v in CROSS_CHECKS.values()),
             n_dis, len(records()), len(ids())))
