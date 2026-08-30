"""Cross-references and gap records from Oliver Labs's Algebraic Surface
Homepage (algebraicsurface.net), mirrored locally as a Wayback rescue.

Source: ``S:/data/math_art/references/websites/algsurf/`` -- captures up to
2017-10-01 (the live domain now serves an unrelated MO-Labs splash page).
53 chapters under ``book/``; ``_mirror/`` holds the untouched site including
the original Singular (``.sin``/``.out``) computer-algebra sources.  Ids are
the mirror chapter stems, e.g. ``ch002_sept99nodes_constr``; a stem resolves
as ``book/<part>/<stem>.md`` and the chapter numbering is global across
parts, so a bare stem is unambiguous.

Contract, shared with ferreol.py and vmm.py:

  records() -> {slug: spec}   new records, spec shaped as in tail.py.
    Every entry carries a real citation (the mirrored page, plus the paper
    or Singular script the page itself cites) and a stated reason it is
    unimplemented.
  ids()     -> {slug: {"algsurf": "chNNN_stem"}}
    For surfaces that ALREADY have a record.  Each id was verified by
    reading the page and matching its content (not merely its title)
    against the record's mathematics.

Equations are deliberately NOT transcribed (no polynomial / x /
gauss_map), even though the mirrored Singular sources hold them: those
fields are guarded by numerical oracles that can only run against a
shipped implementation, and an unverified transcription would silently
define a different surface.  The sources are a citation, not a
transcription target.

What the site holds, and how it was triaged:

  * septics/ (5 chapters) -- the Labs septic, degree 7 with 99 nodes
    (arXiv math.AG/0409348).  Already recorded (labs-septic); gets an id.
    The site's mainframe also announces A SEXTIC WITH 35 CUSPS, published
    only as a Singular proof script (book/data/sextic35cusps_all.sin):
    a NEW record, and the singularities are CUSPS (A2), not nodes.
  * octics/ (6) -- the D8 x Z2 story: Endrass's 5-parameter family
    (generic member 112 nodes), its 120/128/136-nodal members, Endrass's
    160-nodal octic, van Straten's 165-nodal octic, and the 168-nodal
    record holder (already recorded as endrass-octic).  Plus van
    Straten's dihedral D_d modification of Chmutov's construction
    (84 nodes at degree 7, 124 at degree 8).  The octics survey page
    (ch006) is the citation for every count.
  * adcal/ (27) -- the 2002 advent calendar.  Of its 24 days, 7 show
    surfaces the database already holds (ids attached), 7 carry enough
    data for NEW records (132-A5 surface, modified Chmutov octic,
    swallowtail, 16-triple-point septic, Stagnaro quintic, Hyde sextic,
    Humbert sextic, Segre's 64-line quartic -- 8 counting day 24), 2 are
    extra views of surfaces recorded from the octics part, and the rest
    are decoration (a shape Labs liked, a degree-21 joke, a circus tent,
    a photo page, an empty page, a software page).
  * series/ (4) and models/ (3) -- frameset shells with no mathematical
    content in the capture (the real series content lived at
    oliverlabs.net/series, not mirrored).  Skipped.
  * other/ (8) -- the mu(d) table page (ch046) is the site's summary of
    known nodal bounds; used as a cross-check citation, no new records.

Known site-internal discrepancy, preserved rather than resolved: the
octics construction page (ch011) says van Straten's dihedral septic has
84 nodes; advent day 7 (ch022), showing the same construction, says 81
real A1 singularities.  Both statements are recorded verbatim in the
van-straten-dihedral-series record.
"""

import os

ALGSURF_ROOT = "S:/data/math_art/references/websites/algsurf"
ALGSURF_BOOK = os.path.join(ALGSURF_ROOT, "book")
ALGSURF_PARTS = ("septics", "octics", "adcal", "series", "models", "other")


def _find_sextic35():
    """Locate Labs's 35-cusp Singular proof script in the mirror.

    Worth resolving rather than hardcoding: this one file is the whole
    evidential basis for the `labs-sextic-35-cusps` record's TERMINAL
    verdict -- its elimination is what gives 884736 s^6 - 8640 s^3 + 25,
    whose negative discriminant proves the 35-cusp member is not real and
    so can never be built. If the citation silently dangles, a future
    reader has a strong claim resting on a file nobody can open.

    A 2026-08 refresh of the mirror moved it out of `book/data/` (where
    the converted chapters keep their attachments) into `_mirror/` (the
    untouched site capture), so both are searched, newest layout first.
    """
    for rel in (("_mirror", "sextic35cusps_all.sin"),
                ("book", "data", "sextic35cusps_all.sin")):
        p = os.path.join(ALGSURF_ROOT, *rel)
        if os.path.exists(p):
            return p
    return os.path.join(ALGSURF_ROOT, "_mirror", "sextic35cusps_all.sin")


SEXTIC35_SIN = _find_sextic35()


def _src(stem, title):
    return ("O. Labs, '%s', The Algebraic Surface Homepage, "
            "algebraicsurface.net (Wayback rescue, captures to 2017-10-01; "
            "local mirror: algsurf/book/.../%s.md)." % (title, stem))


LABS_ARXIV = ("O. Labs, 'A septic with 99 real nodes', arXiv:math.AG/0409348 "
              "(2004); Ph.D. thesis under D. van Straten, Mainz.")

ALGSURF_BLOCKED = (
    "Not built.  The equation is deliberately NOT transcribed -- even "
    "though the mirrored site publishes its construction (and, for some "
    "surfaces, its Singular sources) -- because the numerical oracle that "
    "guards the shipped algebraic block can only verify an equation "
    "against an implementation, and none exists here; an unverified "
    "transcription would silently define a different surface.")
ALGSURF_RESUME = (
    "Transcribe from the mirrored page (and its Singular source where "
    "present) into math_art/surfaces/algebraic.py; the build's oracle "
    "check then confirms the stored form against that implementation.")

_ALG = {"curvature": {"condition": "none"}, "tradition": ["classical"]}


def _deep(*layers):
    """Small dict merge so the shared extra templates stay immutable."""
    out = {}
    for layer in layers:
        for k, v in layer.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep(out[k], v)
            elif isinstance(v, list):
                out[k] = list(v)
            else:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
# ids() -- cross-references for records that already exist.
#
# Verified against page content, not titles alone:
#   * labs-septic -> ch002_sept99nodes_constr: the construction page
#     (Rohn's E - Q^2 globalization, Barth-style symmetric planes, finite
#     prime fields, the three Singular proof scripts).  ch001_septics is
#     the overview stating 99 <= mu(7) <= 104.
#   * endrass-octic -> ch005_endroctconstr: gives the full 5-parameter
#     D8 x Z2 family and the parameter values for exactly 168 nodes.
#     (Advent day 8 shows the same surface; one id slot, the construction
#     page wins.)
#   * barth-sextic -> ch020_no_02: "W. Barth's sextic with 65 nodes (the
#     maximum for sextics) together with the planes and the sphere used
#     to construct it" -- matches the record's 65-node count.
#   * kummer-quartic -> ch021_no_03: "a quartic with 16 double points".
#   * sarti-dodecic -> ch023_no_05: "A. Sarti's surface of degree 12 with
#     600 nodes" -- matches the record's 600.
#   * fresnel-wave-surface -> ch028_no_10: quartic derived from an
#     ellipsoid; the page adds that of its 16 nodes only 12 are real
#     (the record carries no singularity count, so nothing conflicts).
#   * clebsch-diagonal-cubic -> ch040_no_22: "The Clebsch Cubic Surface
#     containing 10 Eckardt points".
#
# Deliberately NOT attached: advent day 4 ("The cyclide - a surface of
# degree 4", with a 1-dimensional real component) to dupin-cyclide or
# darboux-cyclide -- the page names no construction, and both records
# are quartic cyclides, so the match cannot be verified beyond the title.
# A wrong cross-reference is worse than none.
# ---------------------------------------------------------------------------

IDS = {
    "labs-septic": "ch002_sept99nodes_constr",
    # NOTE: the mirror RE-CHAPTERS WHOLESALE when algsurf is
    # re-downloaded -- a 2026-08 refresh renumbered 20 of these 24
    # ids at once (ch024_no_02 -> ch020_no_02, and so on).  Only the
    # chNNN prefix moves; the slug after it is stable, which is what
    # makes the drift recoverable AND safe: because the id carries
    # the slug, a renumber fails `_selftest`'s resolve check loudly
    # instead of silently pointing at a DIFFERENT surface.  To repair,
    # re-resolve every id by its slug against the mirror rather than
    # patching them one at a time.
    "endrass-octic": "ch005_endroctconstr",
    "barth-sextic": "ch020_no_02",
    "kummer-quartic": "ch021_no_03",
    "sarti-dodecic": "ch023_no_05",
    "fresnel-wave-surface": "ch028_no_10",
    "clebsch-diagonal-cubic": "ch040_no_22",
}


def ids():
    return {slug: {"algsurf": stem} for slug, stem in IDS.items()}


# ---------------------------------------------------------------------------
# records() -- named surfaces with no record at all.
# ---------------------------------------------------------------------------

def _septic_sextic_records():
    """The mainframe's headline beyond the septic: the 35-cusp sextic."""
    return {
        "labs-sextic-35-cusps": {
            "name": "Labs Sextic (35 cusps)", "family": "algebraic",
            "mode": "implicit",
            # NOT the generic "equation not transcribed" block. This one is
            # TERMINAL, and worth stating precisely, because it is the kind
            # of obstruction that looks like a transcription failure and is
            # not.
            #
            # s is NOT an auxiliary quantity -- it is a coefficient of the
            # surface.  With f = p - q^3 and q = s(x^2+y^2) + t z^2 + u wz
            # + v w^2, the sextic's coefficients are polynomials in s, so a
            # non-real s propagates straight into them.  Eliminating t and u
            # from the mirrored Singular script's ideal (a lex Groebner
            # basis over sextic35cusps_all.sin lines 165-171) gives the
            # paper's equation (3),
            #
            #     884736 s^6 - 8640 s^3 + 25 = 0,
            #
            # i.e. 2^15*3^3 s^6 - 2^6*3^3*5 s^3 + 5^2, a quadratic in s^3 of
            # discriminant 8640^2 - 4*884736*25 = -13,824,000 < 0.  No real
            # root, so no real s, so S_35 exists only over C.  Labs draws
            # the same conclusion and extends it to the WHOLE four-parameter
            # family, which closes the obvious escape route ("maybe some
            # other member is real"):
            #
            #   "Note that the coefficients of the surface S_35 are not
            #    real. In fact, the ideal sl_{f,3} does not contain any real
            #    point, because equation (3) does not have any real root. In
            #    particular, it is not possible to use the software surf to
            #    draw an image of this sextic. This also holds for the more
            #    general family f_{s,t,u,v} because of equation (2)."
            #       -- arXiv:math/0502520, section 1, after Corollary 2.
            #
            # WHY THIS LOOKS WRONG AT FIRST, and is not.  Labs sextics ARE
            # rendered all over the place, which makes "cannot be drawn"
            # sound absurd.  Those pictures are of DIFFERENT surfaces: the
            # same paper's Theorem 3 sextic (30 real cusps + 10 real nodes,
            # its Figure 1), Labs's 99-node septic, and Barth's 65-node
            # sextic.  None of them is S_35.  The contrast with the septic
            # is the tell: there the analogous condition 7a^3 + 7a + 1 = 0
            # HAS a real root (a = -0.14010685...) and every singularity is
            # real, which is exactly why that one is everywhere and this one
            # is nowhere.
            #
            # SCOPE, precisely.  What is proved is that no member OF THIS
            # FAMILY with 35 cusps is real, and that 35 <= mu_A2(6) <= 37
            # over C.  It is NOT proved that no real sextic with 35 cusps
            # exists at all -- that is open.  Do not strengthen this.
            "blocked_by":
                "Not built, and not buildable by this add-on: the 35-cusp "
                "member of the family has NON-REAL coefficients, so its real "
                "solution set is not a surface (at most a curve) and there "
                "is nothing to mesh.  The Singular script's elimination "
                "gives the paper's equation (3), "
                "884736*s^6 - 8640*s^3 + 25 = 0, for the parameter s that "
                "enters the surface's own coefficients; it is a quadratic "
                "in s^3 of discriminant "
                "8640^2 - 4*884736*25 = -13824000 < 0 and so has no real "
                "root.  Labs states the conclusion himself -- 'the "
                "coefficients of the surface S_35 are not real ... it is "
                "not possible to use the software surf to draw an image of "
                "this sextic' -- and extends it to the whole four-parameter "
                "family.  Terminal, not a missing transcription.  NOTE the "
                "scope: no member OF THIS FAMILY with 35 cusps is real; "
                "whether some other real sextic attains 35 cusps is open.",
            "resume":
                "Nothing to resume for the 35-cusp surface itself.  The "
                "buildable substitute from the same paper is its Theorem 3 "
                "sextic -- the one its Figure 1 actually shows -- which "
                "carries 30 REAL cusps and 10 real nodes at "
                "s0 = 5**(1/3)/12, t0 = 4*s0, u = 0 (its nodes lie at "
                "infinity).  Source: O. Labs, 'A Sextic with 35 Cusps', "
                "arXiv:math/0502520, Theorem 3; and the mirrored Singular "
                "source sextic35cusps_all.sin in algsurf/book/data/.",
            "sources": [
                _src("ch018_mainframe",
                     "Algebraic Surface Homepage (announcing 'A Sextic "
                     "with 35 Cusps')"),
                "O. Labs, 'A Sextic with 35 Cusps', Singular proof script "
                "sextic35cusps_all.sin (local mirror: algsurf/book/data/), "
                "which computes the parameters and proves ('theorem 1') "
                "that the surface has exactly 35 cusps and no other "
                "singularities.  The script refers to an accompanying "
                "article for the family f_{s,t,u,v}."],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch018_mainframe"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "cusp (A2)", "count": 35,
                         "note": "CUSPS, not nodes: A2 singularities, per "
                                 "the Singular script's own statement "
                                 "('35 cusps (i.e., A_2-singularities)') "
                                 "and its no-other-singularities proof."}]},
                "definition": {"note":
                    "Degree-6 surface with 35 cusps, built Rohn-style as "
                    "P - Q^3 = 0: P a product of six symmetrically chosen "
                    "planes, Q a quadric with parameters (s,t,u,v).  The "
                    "family generically carries 30 cusps; the mirrored "
                    "Singular script computes parameter values raising "
                    "that to 35 and proves the count."}}),
        },
    }


def _octic_records():
    """The D8 x Z2 octics between the generic 112 and the record 168,
    plus van Straten's dihedral series.  The 168-nodal member already
    has a record (endrass-octic); these are the intermediate
    constructions the site documents alongside it."""
    OCTICS_LIST = _src("ch006_octics",
                       "Octics (the survey of known nodal octic counts: "
                       "Miyaoka bound 174, best known 168)")
    ENDRASS_THESIS = ("St. Endrass, Ph.D. thesis (1996), as cited "
                      "throughout the mirrored octics pages for the "
                      "D8 x Z2 construction.")
    return {
        "van-straten-octic": {
            "name": "Van Straten Octic (165 nodes)", "family": "algebraic",
            "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED, "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch004_duco165",
                     "Van Straten's D8 x Z2-symmetric Octic with 165 "
                     "Nodes"),
                OCTICS_LIST, ENDRASS_THESIS],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch004_duco165"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "node (A1)", "count": 165,
                         "note": "Per the mirrored duco165 page and the "
                                 "octics survey list ('165: D. van "
                                 "Straten (1995?)'); the highest known "
                                 "nodal octic count below Endrass's "
                                 "168."}]},
                "definition": {"note":
                    "Member of Endrass's 5-parameter D8 x Z2 octic family "
                    "(see endrass-octic-family) found by D. van Straten: "
                    "the parameters are those of Endrass's 160-nodal "
                    "octic except b = 1.  The site notes its earlier "
                    "publication venue as 'the printout on his office "
                    "door'."}}),
        },
        "endrass-octic-160": {
            "name": "Endrass Octic (160 nodes)", "family": "algebraic",
            "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED, "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch008_stephan160",
                     "St. Endrass's 160-nodal D8 x Z2-symmetric Octic "
                     "(with the explicit parameter values)"),
                _src("ch037_no_19",
                     "Advent calendar 2002, No. 19 ('Besides constructing "
                     "the octics with 168 nodes, St. Endrass found this "
                     "160-nodal octic with dihedral symmetry, which is "
                     "not so well-known')"),
                OCTICS_LIST, ENDRASS_THESIS],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch008_stephan160"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "node (A1)", "count": 160,
                         "note": "Per the mirrored stephan160 page; the "
                                 "octics survey also lists independent "
                                 "160-nodal octics by Gallarati (1957) "
                                 "and -- doubted by van Straten -- Kreis "
                                 "(1956)."}]},
                "definition": {"note":
                    "Endrass's second D8 x Z2 octic: parameter values in "
                    "his 5-parameter family (b = 4 and four derived "
                    "expressions in b and sqrt(2), printed on the "
                    "mirrored page) giving exactly 160 nodes.  Distinct "
                    "from, and found alongside, his 168-nodal record "
                    "surface."}}),
        },
        "endrass-octic-family": {
            "name": "Endrass Octic Family (D8 x Z2)", "family": "algebraic",
            "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED, "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch005_endroctconstr",
                     "St. Endrass's Construction of D8 x Z2-symmetric "
                     "Octics (the full 5-parameter equation)"),
                _src("ch007_stephan120128136",
                     "120-/128-/136-nodal D8 x Z2-symmetric Octics"),
                OCTICS_LIST, ENDRASS_THESIS],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch005_endroctconstr"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "node (A1)", "count": None,
                         "note": "Varies over the family: 112 for the "
                                 "generic member, 120/128/136 for one "
                                 "generic parameter (c/b/d resp.), 160, "
                                 "165 and 168 at special values.  The "
                                 "construction page adds that every "
                                 "multiple of 4 from 112 to 152 seems "
                                 "attainable, while 156- and 164-nodal "
                                 "members do not seem to exist in this "
                                 "family."}]},
                "definition": {
                    "note":
                        "Endrass's 5-parameter family of octics with "
                        "D8 x Z2 symmetry, combining two constructions of "
                        "B. Segre: a product of eight planes (four "
                        "symmetric plane-pair factors) minus the square "
                        "of a degree-4 form in x^2+y^2, z and w, with "
                        "complex parameters a, b, c, d, e.  The named "
                        "members are promoted to their own records (see "
                        "specimens).",
                    "parameters": [
                        {"name": "a", "domain": "complex"},
                        {"name": "b", "domain": "complex"},
                        {"name": "c", "domain": "complex"},
                        {"name": "d", "domain": "complex"},
                        {"name": "e", "domain": "complex"}]},
                "specimens": [
                    {"label": "generic member",
                     "note": "112 nodes, at the pairwise intersections "
                             "of the plane pairs with the doubled "
                             "quartic."},
                    {"label": "generic c", "note": "120 nodes."},
                    {"label": "generic b", "note": "128 nodes."},
                    {"label": "generic d", "note": "136 nodes."},
                    {"label": "Endrass 160-nodal octic",
                     "slug": "endrass-octic-160"},
                    {"label": "van Straten 165-nodal octic",
                     "slug": "van-straten-octic"},
                    {"label": "Endrass 168-nodal octic (the record)",
                     "slug": "endrass-octic"}]}),
        },
        "van-straten-dihedral-series": {
            "name": "Van Straten Dihedral Series", "family": "algebraic",
            "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED, "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch009_vstrconstr",
                     "Van Straten's Construction of D_d-symmetric "
                     "Surfaces of Degree d with many Nodes"),
                _src("ch025_no_07",
                     "Advent calendar 2002, No. 7 (the septic member)"),
                OCTICS_LIST,
                "S. Chmutov, J. Algebraic Geom. 1 (1992) 191-196 (the "
                "construction being modified; cited by the mirrored "
                "pages)."],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch009_vstrconstr"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "node (A1)", "count": None,
                         "note": "Varies with the degree d; the page "
                                 "leaves the general count open ('has "
                                 "exactly ??? nodes').  All nodes are "
                                 "real."}]},
                "definition": {
                    "note":
                        "Van Straten's modification of Chmutov's nodal "
                        "series: replace the folding polynomial by the "
                        "regular d-gon polynomial R_d(x,y), giving the "
                        "D_d-symmetric surface R_d(x,y) + "
                        "lambda*(T_d(z)+1) = 0 with lambda the critical "
                        "value of R_d and T_d the Chebyshev polynomial "
                        "with critical values 0 and 1.  Trades some of "
                        "Chmutov's (partly complex) nodes for fewer, ALL "
                        "REAL ones.",
                    "parameters": [
                        {"name": "d", "domain": "integers >= 3",
                         "integer": True,
                         "note": "degree, and the dihedral symmetry "
                                 "order D_d"}]},
                "specimens": [
                    {"label": "d = 7 (septic)", "parameters": {"d": 7},
                     "note": "84 nodes per the construction page; the "
                             "advent-calendar rendering of the same "
                             "construction says 81 real A1 "
                             "singularities.  The site-internal "
                             "discrepancy is preserved here, not "
                             "resolved."},
                    {"label": "d = 8 (octic)", "parameters": {"d": 8},
                     "note": "124 nodes; the octics survey lists it as "
                             "'124: D. v. Straten, a modification of "
                             "Chmutov's construction'."}]}),
        },
    }


def _adcal_records():
    """Advent calendar 2002 days carrying enough data for a record.

    The other days: 2/3/4/5/8/10/22 show surfaces already recorded
    (ids above, except day 4's unverifiable cyclide), 7 and 19 are extra
    views of the octics-part records, and 11 (a septic Labs liked), 13
    (a degree-21 joke), 15 (a cut of the Sarti dodecic), 20 (an unnamed
    sextic), 21 (empty page), 23 (photos of plaster models) are
    decoration with no defining data.  Day 12 shows an Enriques surface
    from Endrass's surf script but states nothing about it -- not even
    the degree -- so no record is emitted for it."""
    return {
        "surface-with-132-a5-singularities": {
            "name": "Surface with 132 A5 Singularities",
            "family": "algebraic", "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED + (
                "  The page states the singularity count but not the "
                "degree; the series it belongs to lived at "
                "oliverlabs.net/series, which the capture did not "
                "preserve."),
            "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch019_no_01", "Advent calendar 2002, No. 1"),
                "Cover image of the SuSE Linux 8.1 distribution, per the "
                "page (Labs made the SuSE cover surfaces from version "
                "7.1 on)."],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch019_no_01"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "A5", "count": 132,
                         "note": "12*11 = 132 A5 singularities, per the "
                                 "page; higher A_j points, not nodes or "
                                 "cusps."}]},
                "definition": {"note":
                    "Part of a series of surfaces in P^3 with many A_j "
                    "singularities (the page gives the count 12*11 = 132 "
                    "for j = 5 but not the degree)."}}),
        },
        "modified-chmutov-octic": {
            "name": "Modified Chmutov Octic (144 real nodes)",
            "family": "algebraic", "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED, "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch024_no_06", "Advent calendar 2002, No. 6 (with "
                     "the defining Chebyshev form)"),
                _src("ch006_octics",
                     "Octics (listing it as '144: (2002?), modified "
                     "Chmutov surface')"),
                "V. I. Arnold et al., Singularities of Differentiable "
                "Maps, Vol. II, Birkhaeuser (1988), p. 419 (cited by the "
                "page for Chmutov's original series)."],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch024_no_06"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "node (A1)", "count": 144,
                         "note": "All 144 nodes are REAL -- the point of "
                                 "the modification: Chmutov's original "
                                 "octic has 154 nodes, but complex "
                                 "ones."}]},
                "definition": {"note":
                    "The j = 1, d = 8, n = 3 member of a series of "
                    "degree-d hypersurfaces in P^n modifying Chmutov's: "
                    "the sum over the coordinates of the (j+1)-th powers "
                    "of the Chebyshev polynomials T_{d/2} (critical "
                    "values +1/-1) set equal to 1, giving real A_j "
                    "singularities."}}),
        },
        "swallowtail": {
            "name": "Swallowtail", "family": "algebraic",
            "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED + (
                "  The page shows the surface with no equation and the "
                "bare caption 'The so-called swallowtail'; the "
                "identification with the standard discriminant surface "
                "rests on that established name."),
            "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch027_no_09", "Advent calendar 2002, No. 9")],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch027_no_09"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "cuspidal edge", "count": None},
                        {"type": "swallowtail point", "count": 1,
                         "note": "where the cuspidal edge and the "
                                 "self-intersection curve meet."}]},
                "definition": {"note":
                    "The swallowtail of singularity theory: the "
                    "discriminant surface of the general quartic "
                    "polynomial in one variable (parameter space "
                    "coordinates where the quartic has a repeated root), "
                    "one of the elementary catastrophes."}}),
        },
        "septic-with-16-triple-points": {
            "name": "Septic with 16 Triple Points", "family": "algebraic",
            "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED, "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch032_no_14", "Advent calendar 2002, No. 14"),
                "S. Endrass, U. Persson, J. Stevens, 'Surfaces with "
                "Triple Points', arXiv:math.AG/0010163 (linked from the "
                "page as the reference)."],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch032_no_14"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "triple point", "count": 16,
                         "note": "Triple points, a heavier singularity "
                                 "than a node; per the page and the "
                                 "Endrass-Persson-Stevens paper it "
                                 "cites."}]},
                "definition": {"note":
                    "A degree-7 surface with 16 triple points, rendered "
                    "by Labs from Endrass's surf script; the "
                    "Endrass-Persson-Stevens paper on surfaces with "
                    "triple points is the page's reference."}}),
        },
        "stagnaro-enriques-quintic": {
            "name": "Stagnaro Enriques Quintic (4 tacnodes)",
            "family": "algebraic", "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED, "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch034_no_16", "Advent calendar 2002, No. 16"),
                "E. Stagnaro's construction; descriptions and equations "
                "communicated to O. Labs by W. Barth, per the page."],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch034_no_16"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "tacnode", "count": 4,
                         "note": "The parameters can be chosen so the "
                                 "four tacnodes are of type 0, 1 or 2; "
                                 "the page shows all three variants."}]},
                "definition": {"note":
                    "One of Stagnaro's Enriques quintics: a degree-5 "
                    "surface with 4 tacnodes, in a family whose "
                    "parameters set the tacnode type (0, 1 or 2)."}}),
        },
        "hyde-sextic": {
            "name": "Hyde Sextic", "family": "algebraic",
            "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED, "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch035_no_17", "Advent calendar 2002, No. 17"),
                "Hyde, Ann. of Math., 2nd series, vol. 2 (1901), and "
                "Ann. of Math. 4 (1888) (the page's references for the "
                "screw-theory construction); description and equation "
                "communicated to O. Labs by W. Barth."],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch035_no_17"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "cuspidal double curve", "count": None,
                         "note": "The surface is singular ALONG A CURVE, "
                                 "not merely at points."},
                        {"type": "singular point", "count": 6,
                         "note": "Only 2 visible in the real picture, "
                                 "per the page."}]},
                "definition": {"note":
                    "Hyde's 1901 sextic, touched by the axes of all "
                    "screws reciprocal to three given screws (classical "
                    "screw theory); singular along a cuspidal double "
                    "curve and at 6 points."}}),
        },
        "humbert-sextic": {
            "name": "Humbert Sextic", "family": "algebraic",
            "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED, "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch036_no_18", "Advent calendar 2002, No. 18"),
                "G. Humbert, 'Sur une surface du sixieme ordre liee aux "
                "fonctions abeliennes de genre 3', J. de Mathem., 2e "
                "serie, t. III (1896); collected papers vol. 2, "
                "p. 269-296 (the page's reference).  Description and "
                "equation communicated to O. Labs by W. Barth."],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch036_no_18"},
                "embedding": {
                    "quality": "singular",
                    "singularities": [
                        {"type": "node (A1)", "count": 16,
                         "note": "The 16 nodes of the Kummer quartic "
                                 "used to construct it."},
                        {"type": "triple point", "count": 1,
                         "note": "At the point (1:1:1:1)."}]},
                "definition": {"note":
                    "Humbert's 1896 sextic, tied to abelian functions of "
                    "genus 3 and built from a Kummer quartic: it "
                    "inherits the Kummer surface's 16 nodes and adds a "
                    "triple point at (1:1:1:1)."}}),
        },
        "segre-quartic-64-lines": {
            "name": "Segre Quartic (64 lines)", "family": "algebraic",
            "mode": "implicit",
            "blocked_by": ALGSURF_BLOCKED, "resume": ALGSURF_RESUME,
            "sources": [
                _src("ch042_no_24", "Advent calendar 2002, No. 24")],
            "extra": _deep(_ALG, {
                "ids": {"algsurf": "ch042_no_24"},
                "embedding": {"quality": "embedded"},
                "definition": {"note":
                    "B. Segre proved a quartic surface can contain at "
                    "most 64 lines and gave this example attaining the "
                    "bound; the distinction is the line count, not "
                    "singularities (the surface is smooth), and most of "
                    "the 64 lines are complex, so they are invisible in "
                    "the real picture."}}),
        },
    }


def records():
    out = {}
    for table in (_septic_sextic_records(), _octic_records(),
                  _adcal_records()):
        for slug, spec in table.items():
            assert slug not in out, "duplicate algsurf-record slug %r" % slug
            out[slug] = spec
    return out


def invariants_for(slug):
    return {}


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _resolve(stem):
    """True if a chapter stem resolves to a page in the local mirror."""
    return any(os.path.isfile(os.path.join(ALGSURF_BOOK, part, stem + ".md"))
               for part in ALGSURF_PARTS)


def _selftest():
    recs = records()
    idtab = ids()

    # -- records: shape ----------------------------------------------------
    seen = set()
    for slug, spec in recs.items():
        assert slug == slug.lower() and " " not in slug and "_" not in slug, \
            slug
        assert slug not in seen, "duplicate algsurf slug %r" % slug
        seen.add(slug)
        assert spec.get("name"), slug
        assert spec.get("family") == "algebraic", (slug, spec.get("family"))
        assert spec.get("mode") == "implicit", (slug, spec.get("mode"))
        assert spec.get("blocked_by"), \
            "%s is absent without a stated reason" % slug
        assert spec.get("resume"), "%s has no resume pointer" % slug
        assert spec.get("sources"), "%s cites nothing" % slug
        d = (spec.get("extra") or {}).get("definition") or {}
        for banned in ("polynomial", "x", "gauss_map"):
            assert not d.get(banned), \
                "%s must not carry an untranscribed %s" % (slug, banned)

    # -- records vs ids: disjoint (ids() is for EXISTING records) ----------
    overlap = set(recs) & set(idtab)
    assert not overlap, "slugs in both records() and ids(): %r" % overlap

    # -- the cusp/node distinction is actually recorded --------------------
    def sing(slug):
        return {e["type"]: e.get("count")
                for e in recs[slug]["extra"]["embedding"]["singularities"]}

    s35 = sing("labs-sextic-35-cusps")
    assert s35 == {"cusp (A2)": 35}, s35  # cusps, and ONLY cusps
    assert sing("van-straten-octic") == {"node (A1)": 165}
    assert sing("endrass-octic-160") == {"node (A1)": 160}
    assert sing("modified-chmutov-octic") == {"node (A1)": 144}
    assert sing("septic-with-16-triple-points") == {"triple point": 16}
    assert sing("humbert-sextic") == {"node (A1)": 16, "triple point": 1}
    assert sing("stagnaro-enriques-quintic") == {"tacnode": 4}
    assert sing("surface-with-132-a5-singularities") == {"A5": 132}

    # -- family records carry parameters and specimens ---------------------
    for slug in ("endrass-octic-family", "van-straten-dihedral-series"):
        extra = recs[slug]["extra"]
        assert extra["definition"]["parameters"], slug
        assert extra["specimens"], slug

    # -- every id resolves against the local mirror ------------------------
    stems = [(slug, stem) for slug, entry in idtab.items()
             for key, stem in entry.items()
             if key == "algsurf" or _fail(slug, key)]
    for slug, spec in recs.items():
        for key, stem in ((spec.get("extra") or {}).get("ids") or {}).items():
            assert key == "algsurf", (slug, key)
            stems.append((slug, stem))
    if os.path.isdir(ALGSURF_BOOK):
        for slug, stem in stems:
            assert _resolve(stem), \
                "%s: algsurf id %r does not resolve in the mirror" % (
                    slug, stem)
        assert os.path.isfile(SEXTIC35_SIN), \
            "the 35-cusp sextic's Singular source is missing from the mirror"
        resolution = "all %d ids resolve" % len(stems)
    else:
        resolution = ("mirror not mounted; %d ids NOT resolution-checked"
                      % len(stems))

    print("RESULT: OK  (surfdb.algsurf, %d records, %d id sets on existing "
          "records; %s)" % (len(recs), len(idtab), resolution))


def _fail(slug, key):
    raise AssertionError("%s: unexpected id key %r" % (slug, key))
