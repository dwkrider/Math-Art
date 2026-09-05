# Literature facts, by slug.
#
# The build DERIVES what the code knows -- labels, family, enum keys,
# parameter ranges, clip regions, which operator reaches a surface.  It
# cannot derive what only a library knows: who found the surface, when,
# what its symmetry group is, how many ends it has, what the sources are.
# That is this file.
#
# Two rules govern everything here.
#
# 1. NEVER FABRICATE.  A missing fact is `null`; a fact this file is
#    unsure of is absent, not guessed.  data/polyhedra sets
#    `vertices_exact: null` for the snubs rather than inventing radicals,
#    and the same discipline applies to a discovery year or a space group.
#
# 2. POLYNOMIALS ARE PROPOSALS, NOT ASSERTIONS.  Every entry in
#    `POLYNOMIAL` below is checked numerically against the shipped
#    implementation in math_art/surfaces/algebraic.py before it reaches a
#    record (see surfdb/polynomial.py).  A proposal that disagrees is
#    DISCARDED and the record carries a null.  This is what makes it safe
#    to write an equation down at all: a mistyped coefficient does not
#    error, it silently produces a different surface, so nothing is
#    trusted that has not been reproduced against an independent
#    implementation.

# ---------------------------------------------------------------------------
# Candidate closed forms for implicit surfaces whose shipped definition
# lives in a Python function rather than a string.  VERIFIED AT BUILD
# TIME against that function; discarded on disagreement.
# ---------------------------------------------------------------------------

POLYNOMIAL = {
    "monkey-saddle": "x**3 - 3*x*y**2 - z",
    "ding-dong-surface": "x**2 + y**2 - z**2 + z**3",
    "cayley-nodal-cubic": "4*(x**2 + y**2 + z**2) + 16*x*y*z - 1",
    "tangle-cube": "x**4 - 5*x**2 + y**4 - 5*y**2 + z**4 - 5*z**2 + 11.8",
    # NOTE: an earlier candidate here was (2x^2 + y^2 + z^2 - 1)^3
    # - (1/10) x^2 z^3 - y^2 z^3, which is a DIFFERENT heart surface that
    # circulates widely. The oracle check rejected it. This is the form
    # math_art ships and that Taubin gives.
    "taubin-heart":
        "(x**2 + (9.0/4.0)*y**2 + z**2 - 1)**3"
        " - x**2*z**3 - (9.0/80.0)*y**2*z**3",
    "barth-sextic":
        "4*(phi**2*x**2 - y**2)*(phi**2*y**2 - z**2)*(phi**2*z**2 - x**2)"
        " - (1 + 2*phi)*(x**2 + y**2 + z**2 - 1)**2",
    "clebsch-diagonal-cubic":
        "81*(x**3 + y**3 + z**3)"
        " - 189*(x**2*y + x**2*z + y**2*x + y**2*z + z**2*x + z**2*y)"
        " + 54*x*y*z + 126*(x*y + x*z + y*z)"
        " - 9*(x**2 + y**2 + z**2) - 9*(x + y + z) + 1",
    "titeica-surface": "x*y*z - 1",
    # NOTE: an earlier candidate added a "+ 3*x*y**2" term, conflating this
    # with the monkey saddle. Rejected by the oracle. Cartan's umbrella is
    # the cone over an Agnesi cubic: z(x^2 + y^2) = x^3.
    "cartans-umbrella": "z*(x**2 + y**2) - x**3",
    # The general Darboux cyclide
    #   (x^2+y^2+z^2)^2 + (x^2+y^2+z^2)(ax+by+cz) + P_2 = 0
    # at the operator's default member: (a,b,c) = (0.9, 0, 0) and P_2
    # derived from the ring (R, r) = (1, 0.45).  The ring is left in
    # symbolically rather than folded into decimals so the derivation
    # stays readable -- with a = 0 these coefficients ARE the torus,
    # which is the identity the module's self-test gates on.
    "darboux-cyclide":
        "(x**2 + y**2 + z**2)**2"
        " + (x**2 + y**2 + z**2)*(0.9*x)"
        " + (2*(1.0**2 - 0.45**2) - 4*1.0**2)*(x**2 + y**2)"
        " + 2*(1.0**2 - 0.45**2)*z**2"
        " + (1.0**2 - 0.45**2)**2",
}

# ---------------------------------------------------------------------------
# Per-slug literature facts.  Deep-merged onto the derived skeleton, so a
# key absent here simply leaves the derived value (or a null) in place.
# ---------------------------------------------------------------------------

FACTS = {

    # ---------------- classical minimal surfaces ----------------

    "catenoid": {
        "discovered_by": "Leonhard Euler; shown minimal by Jean Baptiste Meusnier",
        "year": 1744,
        "alternate_names": ["Catenoid of revolution"],
        "primary_family": "minimal",
        "tradition": ["classical", "physical"],
        "ids": {"mathworld": "Catenoid", "mathcurve": "ch1198_catenoid_2",
                "vmm": "Catenoid", "wikipedia": "Catenoid"},
        "topology": {"genus": 0, "orientable": True, "one_sided": False,
                     "compact": False, "complete": True,
                     "boundary_components": 0, "euler_characteristic": 0,
                     "finite_total_curvature": True,
                     "ends": [{"type": "catenoidal", "count": 2,
                               "embedded": True}]},
        "curvature": {"condition": "minimal",
                      "mean": {"exact": "0", "value": 0.0, "source": "classical"},
                      "total_curvature": {"exact": "-4*pi",
                                          "value": -12.566370614359172,
                                          "source": "classical",
                                          "note": "degree-1 Gauss map"}},
        "symmetry": {"kind": "continuous", "periodicity_rank": 0,
                     "continuous": "SO(2) about the axis, plus reflection in the waist plane",
                     "point_residual": {"schoenflies": "D_inf_h"},
                     "order": "infinite", "chiral": False,
                     "verified_by": "curated"},
        "embedding": {"quality": "embedded", "self_intersecting": False},
        "metrics": {"normalization": "unit_waist"},
        "relations": {"conjugate": "helicoid",
                      "associate_family": "catenoid-helicoid-associate-family",
                      "associate_angle_degrees": 0.0,
                      "limit_of": ["delaunay-surface"]},
        "provenance": {
            "sources": [
                "L. Euler, Methodus inveniendi lineas curvas maximi minimive "
                "proprietate gaudentes (1744) -- the catenary of revolution as "
                "the minimal surface of revolution.",
                "J. B. Meusnier, 'Memoire sur la courbure des surfaces', "
                "Memoires des savans etrangers 10 (1785, read 1776) -- "
                "identified the catenoid and helicoid as minimal.",
            ]},
    },

    "helicoid": {
        "discovered_by": "Jean Baptiste Meusnier",
        "year": 1776,
        "primary_family": "minimal",
        "tradition": ["classical"],
        "ids": {"mathworld": "Helicoid", "mathcurve": "ch1279_helicoid_2",
                "vmm": "Helicoid", "wikipedia": "Helicoid"},
        "topology": {"genus": 0, "orientable": True, "compact": False,
                     "complete": True, "boundary_components": 0,
                     "finite_total_curvature": False,
                     "ends": [{"type": "helicoidal", "count": 1,
                               "embedded": True}]},
        "curvature": {"condition": "minimal",
                      "mean": {"exact": "0", "value": 0.0, "source": "classical"}},
        "symmetry": {"kind": "continuous", "periodicity_rank": 1,
                     "continuous": "one-parameter screw motion about the axis",
                     "order": "infinite", "chiral": True,
                     "verified_by": "curated"},
        "embedding": {"quality": "embedded"},
        "relations": {"conjugate": "catenoid",
                      "associate_family": "catenoid-helicoid-associate-family",
                      "associate_angle_degrees": 90.0},
        "provenance": {"sources": [
            "J. B. Meusnier, 'Memoire sur la courbure des surfaces', "
            "Memoires des savans etrangers 10 (1785, read 1776).",
            "E. Catalan, 'Sur les surfaces reglees dont l'aire est un "
            "minimum', J. Math. Pures Appl. 7 (1842) -- the helicoid is the "
            "only ruled minimal surface besides the plane."]},
    },

    "catenoid-helicoid-associate-family": {
        "name": "Catenoid-Helicoid Associate Family",
        "discovered_by": "Ossian Bonnet",
        "year": 1853,
        "primary_family": "minimal",
        "tradition": ["classical"],
        "topology": {"genus": 0, "orientable": True, "compact": False,
                     "complete": True},
        "curvature": {"condition": "minimal"},
        "symmetry": {"kind": "continuous", "periodicity_rank": 0},
        "embedding": {"quality": "varies"},
        "specimens": [
            {"label": "catenoid (theta = 0)", "parameters": {"theta_degrees": 0},
             "slug": "catenoid", "embedding": {"quality": "embedded"}},
            {"label": "helicoid (theta = 90)", "parameters": {"theta_degrees": 90},
             "slug": "helicoid", "embedding": {"quality": "embedded"}},
            {"label": "intermediate associate", "parameters": {"theta_degrees": 45},
             "embedding": {"quality": "immersed"}},
        ],
        "provenance": {"sources": [
            "O. Bonnet, 'Memoire sur la theorie generale des surfaces', "
            "J. Ecole Polytechnique 19 (1853) -- the associate family.",
            "H. A. Schwarz, Gesammelte Mathematische Abhandlungen (1890)."]},
        "notes": {"caveats": [
            "The members are ISOMETRIC but not congruent, and only theta = 0 "
            "and theta = 90 are embedded; the intermediates self-intersect."]},
    },

    "enneper-surface": {
        "discovered_by": "Alfred Enneper", "year": 1864,
        "primary_family": "minimal", "tradition": ["classical"],
        "ids": {"mathworld": "EnnepersMinimalSurface",
                "mathcurve": "ch1264_enneper_2", "vmm": "Enneper"},
        "topology": {"genus": 0, "orientable": True, "compact": False,
                     "complete": True, "finite_total_curvature": True,
                     "ends": [{"type": "enneper", "count": 1, "order": 1,
                               "embedded": False}]},
        "curvature": {"condition": "minimal",
                      "total_curvature": {"exact": "-4*pi",
                                          "value": -12.566370614359172,
                                          "source": "classical"}},
        "embedding": {"quality": "self-intersecting"},
        "provenance": {"sources": [
            "A. Enneper, 'Analytisch-geometrische Untersuchungen', "
            "Zeitschrift fur Mathematik und Physik 9 (1864)."]},
        "notes": {"caveats": [
            "Complete and of finite total curvature, but NOT embedded: the "
            "surface crosses itself beyond radius 1 in the standard "
            "parametrisation."]},
    },

    "scherk-doubly-periodic": {
        "name": "Scherk's Doubly Periodic Surface",
        "discovered_by": "Heinrich Ferdinand Scherk", "year": 1835,
        "primary_family": "minimal-periodic", "tradition": ["classical"],
        "ids": {"mathworld": "ScherksMinimalSurfaces",
                "mathcurve": "ch1335_scherk_2"},
        "curvature": {"condition": "minimal"},
        "symmetry": {"kind": "layer", "periodicity_rank": 2},
        "embedding": {"quality": "embedded"},
        "provenance": {"sources": [
            "H. F. Scherk, 'Bemerkungen uber die kleinste Flache innerhalb "
            "gegebener Grenzen', J. reine angew. Math. 13 (1835)."]},
        "notes": {"caveats": [
            "The first minimal surfaces found after the catenoid and helicoid. "
            "The doubly periodic one is the graph of z = log(cos y / cos x)."]},
    },

    "costa-surface": {
        "name": "Costa's Surface",
        "discovered_by": "Celso Jose Costa; embeddedness proved by "
                         "David Hoffman and William Meeks III",
        "year": 1982,
        "primary_family": "minimal", "tradition": ["classical"],
        "ids": {"mathworld": "CostaMinimalSurface", "mathcurve": "ch1242_costa_2",
                "vmm": "Costa", "wikipedia": "Costa's minimal surface"},
        "topology": {"genus": 1, "orientable": True, "compact": False,
                     "complete": True, "finite_total_curvature": True,
                     "euler_characteristic": -3,
                     "ends": [{"type": "planar", "count": 1, "embedded": True},
                              {"type": "catenoidal", "count": 2, "embedded": True}]},
        "curvature": {"condition": "minimal",
                      "total_curvature": {"exact": "-12*pi",
                                          "value": -37.69911184307752,
                                          "source": "classical",
                                          "note": "degree-3 Gauss map"}},
        "embedding": {"quality": "embedded"},
        "relations": {"generalizes": [], "specialises": ["costa-hoffman-meeks"]},
        "provenance": {"sources": [
            "C. J. Costa, 'Example of a complete minimal immersion in R^3 of "
            "genus one and three embedded ends', Bol. Soc. Bras. Mat. 15 (1984).",
            "D. Hoffman and W. H. Meeks III, 'A complete embedded minimal "
            "surface in R^3 with genus one and three ends', J. Diff. Geom. 21 "
            "(1985) -- the proof that Costa's example is embedded."]},
        "notes": {"caveats": [
            "The first new embedded complete minimal surface of finite total "
            "curvature found since the 19th century; it broke the long-held "
            "belief that the plane, catenoid and helicoid were the only ones."]},
    },

    "costa-hoffman-meeks": {
        "name": "Costa-Hoffman-Meeks Surface",
        "discovered_by": "David Hoffman and William Meeks III", "year": 1990,
        "primary_family": "minimal", "tradition": ["classical"],
        "topology": {"genus": "varies", "orientable": True, "compact": False,
                     "complete": True, "finite_total_curvature": True,
                     "euler_characteristic": "varies",
                     "ends": [{"type": "planar", "count": 1, "embedded": True},
                              {"type": "catenoidal", "count": 2, "embedded": True}]},
        "curvature": {"condition": "minimal"},
        "embedding": {"quality": "embedded"},
        "specimens": [
            {"label": "genus 1 (Costa)", "parameters": {"k": 1},
             "slug": "costa-surface",
             "topology": {"genus": 1, "euler_characteristic": -3}},
            {"label": "genus 2", "parameters": {"k": 2},
             "topology": {"genus": 2, "euler_characteristic": -5}},
            {"label": "genus 3", "parameters": {"k": 3},
             "topology": {"genus": 3, "euler_characteristic": -7}},
        ],
        "relations": {"generalizes": ["costa-surface"]},
        "provenance": {"sources": [
            "D. Hoffman and W. H. Meeks III, 'Embedded minimal surfaces of "
            "finite topology', Ann. of Math. 131 (1990)."]},
    },

    "henneberg-surface": {
        "discovered_by": "Lebrecht Henneberg", "year": 1875,
        "primary_family": "minimal", "tradition": ["classical"],
        "ids": {"mathworld": "HennebergsMinimalSurface",
                "mathcurve": "ch1284_henneberg_2"},
        "topology": {"orientable": False, "compact": False, "complete": False},
        "curvature": {"condition": "minimal"},
        "embedding": {"quality": "self-intersecting", "branch_points": 2},
        "provenance": {"sources": [
            "L. Henneberg, 'Uber salche Minimalflachen, welche eine "
            "vorgeschriebene ebene Curve zur geodatischen Linie haben' (1875)."]},
        "notes": {"caveats": [
            "The classical example of a NON-ORIENTABLE minimal surface; it is "
            "a minimal immersion of the projective plane minus a point, and "
            "carries two branch points."]},
    },

    "catalan-surface": {
        "name": "Catalan's Minimal Surface",
        "discovered_by": "Eugene Charles Catalan", "year": 1855,
        "primary_family": "minimal", "tradition": ["classical"],
        "ids": {"mathworld": "CatalanMinimalSurface",
                "mathcurve": "ch1222_minimale_catalan_2"},
        "curvature": {"condition": "minimal"},
        "topology": {"orientable": True, "compact": False, "complete": True},
        "embedding": {"quality": "self-intersecting"},
        "provenance": {"sources": [
            "E. Catalan, 'Memoire sur les surfaces dont les rayons de courbure "
            "en chaque point sont egaux et de signes contraires', "
            "C. R. Acad. Sci. Paris 41 (1855).",
            "E. Schwarz's Bjorling construction realises it from a cycloid."]},
        "notes": {"caveats": [
            "Contains a cycloid as a geodesic -- which is exactly the Bjorling "
            "data that generates it, so the parametric row and the Bjorling "
            "row in math_art are two constructions of ONE surface."]},
    },

    "bour-surface": {
        "name": "Bour's Minimal Surface",
        "discovered_by": "Edmond Bour", "year": 1862,
        "primary_family": "minimal", "tradition": ["classical"],
        "ids": {"mathworld": "BoursMinimalSurface", "mathcurve": "ch1214_bour_2"},
        "curvature": {"condition": "minimal"},
        "embedding": {"quality": "self-intersecting"},
        "provenance": {"sources": [
            "E. Bour, 'Theorie de la deformation des surfaces', "
            "J. Ecole Imperiale Polytechnique 22 (1862)."]},
    },

    "richmond-surface": {
        "discovered_by": "Herbert William Richmond", "year": 1904,
        "primary_family": "minimal", "tradition": ["classical"],
        "ids": {"mathcurve": "ch1329_richmond_2"},
        "curvature": {"condition": "minimal"},
        "topology": {"genus": 0, "complete": True, "orientable": True,
                     "finite_total_curvature": True,
                     "ends": [{"type": "planar", "count": 1, "embedded": True},
                              {"type": "enneper", "count": 1, "order": 1,
                               "embedded": False}]},
        "embedding": {"quality": "self-intersecting"},
        "provenance": {"sources": [
            "H. W. Richmond, 'On the simplest algebraic minimal curves, and "
            "the derived real minimal surfaces', Proc. Cambridge Phil. Soc. "
            "19 (1904)."]},
    },

    "chen-gackstatter": {
        "name": "Chen-Gackstatter Surface",
        "discovered_by": "Chi Cheng Chen and Fritz Gackstatter", "year": 1982,
        "primary_family": "minimal", "tradition": ["classical"],
        "topology": {"genus": 1, "orientable": True, "complete": True,
                     "finite_total_curvature": True,
                     "ends": [{"type": "enneper", "count": 1, "order": 1,
                               "embedded": False}]},
        "curvature": {"condition": "minimal",
                      "total_curvature": {"exact": "-8*pi",
                                          "value": -25.132741228718345,
                                          "source": "classical"}},
        "embedding": {"quality": "self-intersecting"},
        "provenance": {"sources": [
            "C. C. Chen and F. Gackstatter, 'Elliptische und hyperelliptische "
            "Funktionen und vollstandige Minimalflachen vom Enneperschen Typ', "
            "Math. Ann. 259 (1982)."]},
    },

    "riemann-minimal-example": {
        "name": "Riemann's Minimal Example",
        "discovered_by": "Bernhard Riemann", "year": 1867,
        "primary_family": "minimal-periodic", "tradition": ["classical"],
        "curvature": {"condition": "minimal"},
        "symmetry": {"kind": "rod", "periodicity_rank": 1},
        "embedding": {"quality": "embedded"},
        "topology": {"genus": 0, "orientable": True, "complete": True},
        "provenance": {"sources": [
            "B. Riemann, 'Uber die Flache vom kleinsten Inhalt bei gegebener "
            "Begrenzung', Abh. Konigl. Ges. Wiss. Gottingen 13 (1867), "
            "posthumous."]},
        "notes": {"caveats": [
            "Foliated by circles and lines in parallel planes; Meeks, Perez "
            "and Ros later proved it is the only such properly embedded "
            "singly periodic example of genus zero."]},
    },

    "scherk-saddle-tower": {
        "name": "Scherk Saddle Tower",
        "discovered_by": "Heinrich Ferdinand Scherk; the k-wing family is "
                         "due to Hermann Karcher",
        "year": 1835,
        "primary_family": "minimal-periodic", "tradition": ["classical"],
        "curvature": {"condition": "minimal"},
        "symmetry": {"kind": "rod", "periodicity_rank": 1},
        "embedding": {"quality": "embedded"},
        "provenance": {"sources": [
            "H. F. Scherk, J. reine angew. Math. 13 (1835) -- the singly "
            "periodic surface.",
            "H. Karcher, 'Embedded minimal surfaces derived from Scherk's "
            "examples', Manuscripta Math. 62 (1988) -- the saddle towers."]},
    },

    "jorge-meeks-k-noid": {
        "name": "Jorge-Meeks k-noid",
        "discovered_by": "Luquesio Jorge and William Meeks III", "year": 1983,
        "primary_family": "minimal", "tradition": ["classical"],
        "curvature": {"condition": "minimal"},
        "topology": {"genus": 0, "orientable": True, "complete": True,
                     "finite_total_curvature": True},
        # NOT "varies": embeddedness as a function of k is a real theorem
        # this file does not have to hand, and `varies` would oblige every
        # specimen to resolve it. `immersed` is the weaker claim that is
        # certainly true, with the uncertainty stated in the caveat rather
        # than dressed up as a classification.
        "embedding": {"quality": "immersed"},
        "specimens": [
            {"label": "k = 2 (the catenoid)", "parameters": {"k": 2},
             "slug": "catenoid"},
            {"label": "k = 3 (trinoid)", "parameters": {"k": 3}},
            {"label": "k = 4", "parameters": {"k": 4}},
        ],
        "provenance": {"sources": [
            "L. P. Jorge and W. H. Meeks III, 'The topology of complete "
            "minimal surfaces of finite total Gaussian curvature', "
            "Topology 22 (1983)."]},
        "notes": {"caveats": [
            "k catenoidal ends in symmetric position; k = 2 degenerates to the "
            "catenoid. Which k give EMBEDDED surfaces is not asserted here."]},
    },

    # ---------------- triply periodic ----------------

    "schwarz-p": {
        "name": "Schwarz P Surface",
        "alternate_names": ["Primitive surface", "Schwarz primitive surface"],
        "discovered_by": "Hermann Amandus Schwarz", "year": 1865,
        "primary_family": "minimal-periodic",
        "tradition": ["classical", "crystallographic"],
        "ids": {"mathworld": "SchwarzsMinimalSurface",
                "mathcurve": "ch1336_schwarz_2", "epinet": "pcu"},
        "curvature": {"condition": "minimal"},
        "topology": {"genus_per_cell": 3, "euler_characteristic_per_cell": -4,
                     "orientable": True, "compact": False, "complete": True,
                     "finite_total_curvature": False},
        "symmetry": {"kind": "space", "periodicity_rank": 3,
                     "hermann_mauguin": "Im-3m", "ita_number": 229,
                     "lattice": "primitive cubic", "chiral": False,
                     "verified_by": "curated"},
        "embedding": {"quality": "embedded"},
        "relations": {"associate_family": "pgd-associate-family",
                      "associate_angle_degrees": 0.0,
                      "conjugate": "schwarz-d",
                      "dual_labyrinth_graph": "pcu"},
        "metrics": {"normalization": "unit_cell"},
        "provenance": {"sources": [
            "H. A. Schwarz, Gesammelte Mathematische Abhandlungen, Springer "
            "(1890) -- the P and D surfaces."]},
    },

    "schwarz-d": {
        "name": "Schwarz D Surface",
        "alternate_names": ["Diamond surface", "F surface"],
        "discovered_by": "Hermann Amandus Schwarz", "year": 1865,
        "primary_family": "minimal-periodic",
        "tradition": ["classical", "crystallographic"],
        "ids": {"mathcurve": "ch1336_schwarz_2", "epinet": "dia"},
        "curvature": {"condition": "minimal"},
        "topology": {"genus_per_cell": 3, "euler_characteristic_per_cell": -4,
                     "orientable": True, "compact": False, "complete": True},
        "symmetry": {"kind": "space", "periodicity_rank": 3,
                     "hermann_mauguin": "Pn-3m", "ita_number": 224,
                     "lattice": "diamond", "chiral": False,
                     "verified_by": "curated"},
        "embedding": {"quality": "embedded"},
        "relations": {"associate_family": "pgd-associate-family",
                      "associate_angle_degrees": 90.0,
                      "conjugate": "schwarz-p",
                      "dual_labyrinth_graph": "dia"},
        "metrics": {"normalization": "unit_cell"},
        "provenance": {"sources": [
            "H. A. Schwarz, Gesammelte Mathematische Abhandlungen (1890)."]},
        "notes": {"known_issue":
                  "The exact (Weierstrass) tiling is ~6% non-manifold in the "
                  "shipped build; recorded in research/minimal_surfaces_status.md."},
    },

    "gyroid": {
        # The registry row is keyed 'G' and labelled "G surface"; the
        # literature name is the Gyroid, and the cross-check flagged the
        # mismatch against Ferreol's chapter title.
        "name": "Gyroid",
        "discovered_by": "Alan H. Schoen", "year": 1970,
        "primary_family": "minimal-periodic",
        "tradition": ["classical", "crystallographic"],
        "ids": {"mathworld": "Gyroid", "mathcurve": "ch1277_gyroide_2",
                "epinet": "srs", "wikipedia": "Gyroid"},
        "curvature": {"condition": "minimal"},
        "topology": {"genus_per_cell": 3, "euler_characteristic_per_cell": -4,
                     "orientable": True, "compact": False, "complete": True},
        "symmetry": {"kind": "space", "periodicity_rank": 3,
                     "hermann_mauguin": "Ia-3d", "ita_number": 230,
                     "lattice": "body-centred cubic", "chiral": True,
                     "verified_by": "curated"},
        "embedding": {"quality": "embedded"},
        "relations": {"associate_family": "pgd-associate-family",
                      "associate_angle_degrees": 38.014772,
                      "dual_labyrinth_graph": "srs"},
        "metrics": {"normalization": "unit_cell"},
        "provenance": {"sources": [
            "A. H. Schoen, 'Infinite periodic minimal surfaces without "
            "self-intersections', NASA Technical Note TN D-5541 (1970).",
            "H. Karcher, 'The triply periodic minimal surfaces of Alan Schoen "
            "and their constant mean curvature companions', Manuscripta Math. "
            "64 (1989) -- the first proof that the gyroid is embedded."]},
        "notes": {
            "known_issue":
                "The exact chiral cell cannot currently be tiled watertight in "
                "the shipped build; recorded in "
                "research/minimal_surfaces_status.md.",
            "caveats": [
                "Contains no straight lines and no mirror symmetries, which is "
                "what makes it chiral and why Schoen found it last of the "
                "three associates."]},
    },

    "pgd-associate-family": {
        "name": "P-Gyroid-D Associate Family",
        "discovered_by": "Hermann Amandus Schwarz (P, D); "
                         "Alan H. Schoen (the gyroid member)",
        "year": 1970,
        "primary_family": "minimal-periodic",
        "tradition": ["classical", "crystallographic"],
        "curvature": {"condition": "minimal"},
        "topology": {"genus_per_cell": 3, "orientable": True, "compact": False,
                     "complete": True},
        "symmetry": {"kind": "space", "periodicity_rank": 3,
                     "hermann_mauguin": "varies", "chiral": "varies"},
        "embedding": {"quality": "varies"},
        "metrics": {"normalization": "unit_cell"},
        "specimens": [
            {"label": "Schwarz P (theta = 0)", "slug": "schwarz-p",
             "parameters": {"theta_degrees": 0.0},
             "embedding": {"quality": "embedded"},
             "symmetry": {"hermann_mauguin": "Im-3m", "chiral": False}},
            {"label": "gyroid (theta = 38.0148)", "slug": "gyroid",
             "parameters": {"theta_degrees": 38.014772},
             "embedding": {"quality": "embedded"},
             "symmetry": {"hermann_mauguin": "Ia-3d", "chiral": True}},
            {"label": "Schwarz D (theta = 90)", "slug": "schwarz-d",
             "parameters": {"theta_degrees": 90.0},
             "embedding": {"quality": "embedded"},
             "symmetry": {"hermann_mauguin": "Pn-3m", "chiral": False}},
            {"label": "generic associate", "parameters": {"theta_degrees": 20.0},
             "embedding": {"quality": "self-intersecting"},
             "symmetry": {"hermann_mauguin": None, "chiral": True}},
        ],
        "provenance": {"sources": [
            "A. H. Schoen, NASA TN D-5541 (1970).",
            "O. Bonnet (1853) for the associate family in general."]},
        "notes": {"caveats": [
            "The three named members are the only EMBEDDED ones; a generic "
            "associate angle self-intersects, which is why only three of a "
            "continuous family have names."]},
    },

    "neovius-surface": {
        "discovered_by": "Edvard Rudolf Neovius", "year": 1883,
        "primary_family": "minimal-periodic",
        "tradition": ["classical", "crystallographic"],
        "ids": {"mathcurve": "ch1307_neovius_2"},
        "curvature": {"condition": "minimal"},
        "topology": {"genus_per_cell": 9, "orientable": True, "compact": False,
                     "complete": True},
        "symmetry": {"kind": "space", "periodicity_rank": 3,
                     "hermann_mauguin": "Im-3m", "ita_number": 229,
                     "chiral": False, "verified_by": "curated"},
        "embedding": {"quality": "embedded"},
        "metrics": {"normalization": "unit_cell"},
        "provenance": {"sources": [
            "E. R. Neovius, 'Bestimmung zweier speciellen periodischen "
            "Minimalflachen', Helsingfors (1883)."]},
    },

    "lidinoid": {
        "discovered_by": "Sven Lidin", "year": 1990,
        "primary_family": "minimal-periodic", "tradition": ["crystallographic"],
        "curvature": {"condition": "minimal"},
        "symmetry": {"kind": "space", "periodicity_rank": 3, "chiral": True,
                     "verified_by": "curated"},
        "embedding": {"quality": "embedded"},
        "metrics": {"normalization": "unit_cell"},
        "provenance": {"sources": [
            "S. Lidin and S. Larsson, 'Bonnet transformation of infinite "
            "periodic minimal surfaces with hexagonal symmetry', "
            "J. Chem. Soc. Faraday Trans. 86 (1990)."]},
        "notes": {"caveats": [
            "The hexagonal analogue of the gyroid: an associate of the H "
            "family, in the same way the gyroid is an associate of P and D."]},
    },

    # ---------------- CMC ----------------

    "delaunay-surface": {
        "name": "Delaunay Surface",
        "discovered_by": "Charles-Eugene Delaunay", "year": 1841,
        "primary_family": "cmc", "tradition": ["classical"],
        "ids": {"mathcurve": "ch1252_delaunay_4"},
        "curvature": {"condition": "cmc"},
        "topology": {"orientable": True, "compact": False, "complete": True},
        "symmetry": {"kind": "continuous", "periodicity_rank": 1,
                     "continuous": "SO(2) about the axis; periodic along it"},
        "embedding": {"quality": "varies"},
        "metrics": {"normalization": "unit_waist"},
        "specimens": [
            {"label": "sphere chain", "slug": "sphere-chain",
             "embedding": {"quality": "immersed"}},
            {"label": "unduloid", "slug": "unduloid",
             "embedding": {"quality": "embedded"}},
            {"label": "cylinder", "slug": "circular-cylinder",
             "embedding": {"quality": "embedded"}},
            {"label": "nodoid", "slug": "nodoid",
             "embedding": {"quality": "self-intersecting"}},
            {"label": "catenoid (H = 0)", "slug": "catenoid",
             "embedding": {"quality": "embedded"}},
        ],
        "provenance": {"sources": [
            "C. Delaunay, 'Sur la surface de revolution dont la courbure "
            "moyenne est constante', J. Math. Pures Appl. 6 (1841)."]},
        "notes": {"caveats": [
            "Delaunay's theorem: the surfaces of revolution of constant mean "
            "curvature are exactly the plane, sphere, cylinder, catenoid, "
            "unduloid and nodoid -- and their profiles are the ROULETTES of "
            "the conics, which is why the family is one object."]},
    },

    "unduloid": {
        "discovered_by": "Charles-Eugene Delaunay", "year": 1841,
        "primary_family": "cmc", "tradition": ["classical", "physical"],
        "ids": {"mathworld": "Unduloid"},
        "curvature": {"condition": "cmc"},
        "topology": {"genus": 0, "orientable": True, "compact": False,
                     "complete": True},
        "symmetry": {"kind": "continuous", "periodicity_rank": 1},
        "embedding": {"quality": "embedded"},
        "relations": {"member_of": "delaunay-surface",
                      "degenerates_to": ["circular-cylinder", "sphere-chain"]},
        "provenance": {"sources": [
            "C. Delaunay, J. Math. Pures Appl. 6 (1841) -- the roulette of an "
            "ellipse."]},
    },

    "nodoid": {
        "discovered_by": "Charles-Eugene Delaunay", "year": 1841,
        "primary_family": "cmc", "tradition": ["classical"],
        "curvature": {"condition": "cmc"},
        "topology": {"orientable": True, "compact": False, "complete": True},
        "symmetry": {"kind": "continuous", "periodicity_rank": 1},
        "embedding": {"quality": "self-intersecting"},
        "relations": {"member_of": "delaunay-surface"},
        "provenance": {"sources": [
            "C. Delaunay, J. Math. Pures Appl. 6 (1841) -- the roulette of a "
            "hyperbola."]},
    },

    "wente-torus": {
        "discovered_by": "Henry C. Wente", "year": 1986,
        "primary_family": "cmc", "tradition": ["classical"],
        "curvature": {"condition": "cmc"},
        "topology": {"genus": 1, "orientable": True, "compact": True,
                     "euler_characteristic": 0},
        "embedding": {"quality": "immersed"},
        "provenance": {"sources": [
            "H. C. Wente, 'Counterexample to a conjecture of H. Hopf', "
            "Pacific J. Math. 121 (1986).",
            "R. Walter, 'Explicit examples to the H-problem of Heinz Hopf', "
            "Geom. Dedicata 23 (1987) -- the elementary parametrisation used "
            "here (supplement 6.B')."]},
        "notes": {"caveats": [
            "The counterexample to Hopf's conjecture that the only closed CMC "
            "surface is the round sphere. Immersed, not embedded -- Alexandrov "
            "had already proved no EMBEDDED counterexample can exist."]},
    },

    # ---------------- constant curvature ----------------

    "pseudosphere": {
        "alternate_names": ["Tractricoid", "Beltrami's pseudosphere"],
        "discovered_by": "Eugenio Beltrami", "year": 1868,
        "primary_family": "constant-curvature", "tradition": ["classical",
                                                              "physical-model"],
        "ids": {"mathworld": "Pseudosphere", "mathcurve": "ch1210_pseudosphere_2",
                "wikipedia": "Pseudosphere"},
        "curvature": {"condition": "k-const-negative",
                      "gaussian": {"exact": "-1", "value": -1.0,
                                   "source": "classical"}},
        "topology": {"orientable": True, "compact": False, "complete": False,
                     "boundary_components": 1},
        "symmetry": {"kind": "continuous", "periodicity_rank": 0,
                     "continuous": "SO(2) about the axis"},
        "embedding": {"quality": "embedded"},
        "metrics": {"normalization": "unit_scale_parameter"},
        "provenance": {"sources": [
            "E. Beltrami, 'Saggio di interpretazione della geometria non-"
            "euclidea', Giornale di Matematiche 6 (1868)."]},
        "notes": {"caveats": [
            "A local model of the hyperbolic plane, not a global one: Hilbert "
            "(1901) proved no complete surface of constant negative curvature "
            "can be immersed in R^3, so the pseudosphere necessarily has an "
            "edge."]},
    },

    "dini-surface": {
        "discovered_by": "Ulisse Dini", "year": 1865,
        "primary_family": "constant-curvature", "tradition": ["classical"],
        "ids": {"mathworld": "DinisSurface", "mathcurve": "ch1255_dini_2"},
        "curvature": {"condition": "k-const-negative",
                      "gaussian": {"exact": "-1", "value": -1.0,
                                   "source": "classical"}},
        "symmetry": {"kind": "continuous", "periodicity_rank": 1,
                     "continuous": "screw motion about the axis"},
        "embedding": {"quality": "embedded"},
        "provenance": {"sources": [
            "U. Dini, 'Sopra le funzioni di una variabile complessa' and the "
            "helicoidal surfaces of constant curvature (1865)."]},
        "notes": {"caveats": [
            "The helicoidal (twisted) pseudosphere: applying a screw motion to "
            "the tractrix gives a whole one-parameter family of K = -1 "
            "surfaces, of which the pseudosphere is the untwisted member."]},
    },

    "kuen-surface": {
        "discovered_by": "Theodor Kuen", "year": 1884,
        "primary_family": "constant-curvature",
        "tradition": ["classical", "physical-model"],
        "ids": {"mathworld": "KuenSurface", "mathcurve": "ch1293_kuen_2"},
        "curvature": {"condition": "k-const-negative",
                      "gaussian": {"exact": "-1", "value": -1.0,
                                   "source": "classical"}},
        "embedding": {"quality": "self-intersecting"},
        "provenance": {"sources": [
            "T. Kuen, 'Ueber Flachen von constantem Krummungsmass', "
            "Sitzungsber. Bayer. Akad. Wiss. (1884)."]},
        "notes": {"caveats": [
            "A Backlund transform of the pseudosphere; a staple of the "
            "Gottingen plaster-model collection."]},
    },

    "breather-surface": {
        "discovered_by": "known from the sine-Gordon breather solution",
        "primary_family": "constant-curvature", "tradition": ["classical"],
        "curvature": {"condition": "k-const-negative",
                      "gaussian": {"exact": "-1", "value": -1.0,
                                   "source": "classical"}},
        "embedding": {"quality": "self-intersecting"},
        "provenance": {"sources": [
            "A. I. Bobenko, 'Surfaces in terms of 2 by 2 matrices: old and new "
            "integrable cases', in Harmonic Maps and Integrable Systems (1994) "
            "-- Sym's formula for pseudospherical surfaces from sine-Gordon "
            "solutions."]},
        "notes": {"caveats": [
            "A soliton/antisoliton bound state of the sine-Gordon equation, "
            "rendered as a K = -1 surface. Melko-Sterling 1993 does NOT contain "
            "a closed-form breather parametrisation, despite being widely cited "
            "for one -- 'breather' there classifies branch points of the "
            "spectral curve."]},
    },

    "amsler-surface": {
        "discovered_by": "Marc-Henri Amsler", "year": 1955,
        "primary_family": "constant-curvature", "tradition": ["classical"],
        "curvature": {"condition": "k-const-negative",
                      "gaussian": {"exact": "-1", "value": -1.0,
                                   "source": "classical"}},
        "embedding": {"quality": "self-intersecting"},
        "provenance": {"sources": [
            "M.-H. Amsler, 'Des surfaces a courbure negative constante dans "
            "l'espace a trois dimensions et de leurs singularites', "
            "Math. Ann. 130 (1955)."]},
        "notes": {"caveats": [
            "The pseudospherical surface containing two intersecting straight "
            "lines; it necessarily carries cuspidal edges, which is why the "
            "shipped patch stops short of them."]},
    },

    "sieverts-surface": {
        "name": "Sievert's Surface",
        "discovered_by": "Heinrich Sievert", "year": 1886,
        "primary_family": "constant-curvature",
        "tradition": ["classical", "physical-model"],
        "ids": {"mathworld": "SievertsSurface", "mathcurve": "ch1339_sievert_2"},
        "curvature": {"condition": "k-const-positive",
                      "gaussian": {"exact": "1", "value": 1.0,
                                   "source": "classical"}},
        "embedding": {"quality": "self-intersecting"},
        "provenance": {"sources": [
            "H. Sievert, Uber die Zentralflachen der Enneperschen Flachen "
            "konstanten Krummungsmasses, dissertation, Tubingen (1886)."]},
        "notes": {"caveats": [
            "The K = +1 counterpart of Enneper's constant-curvature surfaces; "
            "unlike the sphere it is not closed."]},
    },

    "minding-surface": {
        "name": "Minding Surface",
        "discovered_by": "Ferdinand Minding", "year": 1839,
        "primary_family": "constant-curvature", "tradition": ["classical"],
        "curvature": {"condition": "k-const-negative",
                      "gaussian": {"exact": "-1", "value": -1.0,
                                   "source": "classical"}},
        "symmetry": {"kind": "continuous", "periodicity_rank": 0,
                     "continuous": "SO(2) about the axis"},
        "embedding": {"quality": "embedded"},
        "specimens": [
            {"label": "bulge", "parameters": {"kind": "bulge"}},
            {"label": "spindle", "parameters": {"kind": "spindle"}},
        ],
        "provenance": {"sources": [
            "F. Minding, 'Wie sich entscheiden lasst, ob zwei gegebene krumme "
            "Flachen auf einander abwickelbar sind oder nicht', "
            "J. reine angew. Math. 19 (1839)."]},
        "notes": {"caveats": [
            "Minding classified the K = -1 surfaces of revolution into three "
            "types: the pseudosphere (tractricoid), the bulge and the spindle. "
            "The meridian quadrature converges only like h^(3/2) at the rim, "
            "where z' = sqrt(1 - f'^2) vanishes like a square root."]},
    },

    "k-positive-revolution": {
        "name": "Constant Positive Curvature Surface of Revolution",
        "discovered_by": "Ferdinand Minding", "year": 1839,
        "primary_family": "constant-curvature", "tradition": ["classical"],
        "curvature": {"condition": "k-const-positive",
                      "gaussian": {"exact": "1", "value": 1.0,
                                   "source": "classical"}},
        "symmetry": {"kind": "continuous", "periodicity_rank": 0},
        "embedding": {"quality": "embedded"},
        "specimens": [
            {"label": "sphere (the closed member)", "slug": "sphere"},
            {"label": "spindle"},
            {"label": "bulge"},
        ],
        "provenance": {"sources": [
            "F. Minding, J. reine angew. Math. 19 (1839) -- the K = +1 "
            "classification, mirroring the K = -1 one."]},
        "notes": {"caveats": [
            "Only the sphere is closed and smooth; the spindle has two conical "
            "points and the bulge has an edge."]},
    },

    # ---------------- topological ----------------

    # The object every "Calabi-Yau manifold" picture actually shows.
    # A Calabi-Yau threefold has six real dimensions; what is drawn is a
    # two-real-dimensional cross-section of one, projected R^4 -> R^3.
    # Saying so is the whole point of the record.
    "calabi-yau-cross-section": {
        "discovered_by": "Andrew J. Hanson",
        "year": 1994,
        "alternate_names": ["Fermat surface cross-section",
                            "Calabi-Yau quintic cross-section",
                            "Milnor fibre of a Brieskorn singularity"],
        "primary_family": "topological",
        "tradition": ["gallery", "sculptural"],
        "definition": {
            "mode": "parametric",
            "fidelity": "exact",
            "note":
                "z1^p + z2^q = 1 in C^2 = R^4, projected to R^3. For the "
                "quintic threefold z0^5 + ... + z4^5 = 0 in CP^4, fixing "
                "two coordinates and normalising leaves z1^5 + z2^5 = 1: "
                "a complex curve, so a real 2-manifold. Hanson's charts "
                "are z1 = exp(2 pi i k1/p) u1^(2/p), z2 = exp(2 pi i "
                "k2/q) u2^(2/q) with u1 = cosh(xi + i theta) and "
                "u2 = -i sinh(xi + i theta), over 0 <= theta <= pi/2 and "
                "|xi| <= xi_max; u1^2 + u2^2 = 1 makes the defining "
                "equation an identity. NO SINGLE x,y,z CHART IS STORED, "
                "and not for the usual reason: the surface is p*q charts "
                "related by roots of unity, not one, and the projection "
                "(Re z1, Re z2, cos a Im z1 + sin a Im z2) depends on a "
                "chosen angle a. The shipped implementation is "
                "authoritative and checks its own residual "
                "max |z1^p + z2^q - 1| at build time.",
        },
        "curvature": {"condition": "none"},
        "topology": {
            # All three vary with (p, q) and are resolved on every
            # specimen: chi = 1 - (p-1)(q-1), gcd(p,q) boundary circles,
            # genus ((p-1)(q-1) - gcd(p,q) + 1)/2.  These are the Milnor
            # fibre's invariants, and the generator's self-test measures
            # them off the welded mesh rather than asserting them.
            "euler_characteristic": "varies",
            "genus": "varies",
            "boundary_components": "varies",
            "orientable": True,
            "one_sided": False,
            "compact": False,
            "complete": False,
        },
        "symmetry": {"kind": "point", "periodicity_rank": 0},
        "embedding": {
            "quality": "self-intersecting",
            "self_intersecting": True,
        },
        "metrics": {
            "measures_note":
                "Clip-dependent, so left null. The affine curve is "
                "unbounded -- it runs out to gcd(p,q) points at "
                "projective infinity -- and the mesh is a truncation at "
                "a chosen |xi| <= xi_max, so its area is a property of "
                "the cut, not of the surface.",
        },
        "specimens": [
            {"label": "quintic cross-section (n = 5)",
             "parameters": {"family": "FERMAT", "degree": 5},
             "topology": {"euler_characteristic": -15, "genus": 6,
                          "boundary_components": 5}},
            {"label": "cubic cross-section (n = 3)",
             "parameters": {"family": "FERMAT", "degree": 3},
             "topology": {"euler_characteristic": -3, "genus": 1,
                          "boundary_components": 3}},
            {"label": "conic cross-section (n = 2)",
             "parameters": {"family": "FERMAT", "degree": 2},
             "topology": {"euler_characteristic": 0, "genus": 0,
                          "boundary_components": 2}},
            {"label": "trefoil Milnor fibre (p, q) = (2, 3)",
             "parameters": {"family": "BRIESKORN", "power_p": 2,
                            "power_q": 3},
             "topology": {"euler_characteristic": -1, "genus": 1,
                          "boundary_components": 1}},
        ],
        "notes": {"caveats": [
            "It is NOT a Calabi-Yau manifold. A Calabi-Yau threefold is "
            "six real dimensions; this is a two-real-dimensional "
            "cross-section of one. Every popular treatment blurs this.",
            "The self-intersections belong to the R^4 -> R^3 projection, "
            "not to the surface, and they move when the projection angle "
            "is turned. In R^4 the surface is embedded.",
            "As a complex curve it is the Milnor fibre of the "
            "Brieskorn-Pham singularity z1^p + z2^q, so its boundary is "
            "the (p, q) torus link and it is a genus-minimising Seifert "
            "surface of that link -- but not the same object as the "
            "`seifert-surface` record, which is built from braid words "
            "and comes out embedded.",
        ]},
    },

    "klein-bottle": {
        "discovered_by": "Felix Klein", "year": 1882,
        "primary_family": "topological", "tradition": ["classical",
                                                       "physical-model"],
        "ids": {"mathworld": "KleinBottle", "mathcurve": "ch1215_klein_4",
                "wikipedia": "Klein bottle"},
        "curvature": {"condition": "none"},
        "topology": {"euler_characteristic": 0, "orientable": False,
                     "one_sided": True, "compact": True,
                     "boundary_components": 0, "non_orientable_genus": 2},
        "symmetry": {"kind": "point", "periodicity_rank": 0},
        "embedding": {"quality": "immersed", "self_intersecting": True},
        "provenance": {"sources": [
            "F. Klein, 'Ueber Riemann's Theorie der algebraischen Functionen "
            "und ihrer Integrale' (1882)."]},
        "notes": {"known_issue":
                  "FOUND BY THE DRIVE STAGE. The shipped mesh has the right "
                  "Euler characteristic (chi = 0, as the module header "
                  "claims) but is NOT closed: 96 edges bound only one face, "
                  "forming 2 open loops. The combinatorial identification "
                  "closes one direction of the parameter grid and leaves the "
                  "other seamed. chi being correct anyway is a coincidence -- "
                  "an open cylinder also has chi = 0 -- which is exactly why "
                  "an Euler check alone would not have caught this."},
        "notes": {"caveats": [
            "Cannot be embedded in R^3 -- any realisation here self-"
            "intersects. It embeds in R^4. The self-intersection is a property "
            "of the ambient space, not a defect of the model."]},
    },

    "boys-surface": {
        "name": "Boy's Surface",
        "discovered_by": "Werner Boy", "year": 1901,
        "primary_family": "topological", "tradition": ["classical",
                                                       "physical-model",
                                                       "sculptural"],
        "ids": {"mathworld": "BoySurface", "mathcurve": "ch1216_boy_2",
                "wikipedia": "Boy's surface"},
        "curvature": {"condition": "none"},
        "topology": {"euler_characteristic": 1, "orientable": False,
                     "one_sided": True, "compact": True,
                     "boundary_components": 0, "non_orientable_genus": 1},
        "embedding": {"quality": "immersed", "self_intersecting": True},
        "symmetry": {"kind": "point", "periodicity_rank": 0,
                     "schoenflies": "C3v", "orbifold": "*33", "order": 6},
        "provenance": {"sources": [
            "W. Boy, 'Uber die Curvatura integra und die Topologie "
            "geschlossener Flachen', Math. Ann. 57 (1903).",
            "F. Apery, Models of the Real Projective Plane (1987) -- the "
            "explicit parametrisation.",
            "R. Bryant and R. Kusner, for the Willmore-minimising immersion."]},
        "notes": {"caveats": [
            "An immersion of the real projective plane with NO branch points -- "
            "which is what makes it remarkable; Boy found it against Hilbert's "
            "expectation that none existed."]},
    },

    "roman-surface": {
        "name": "Roman Surface",
        "alternate_names": ["Steiner surface", "Steiner's Roman surface"],
        "discovered_by": "Jakob Steiner", "year": 1844,
        "primary_family": "topological", "tradition": ["classical",
                                                       "physical-model"],
        "ids": {"mathworld": "RomanSurface", "mathcurve": "ch1331_romaine_2"},
        "curvature": {"condition": "none"},
        "topology": {"euler_characteristic": 1, "orientable": False,
                     "one_sided": True, "compact": True,
                     "non_orientable_genus": 1},
        "embedding": {"quality": "singular",
                      "singularities": [{"type": "double line", "count": 3},
                                        {"type": "pinch point", "count": 6},
                                        {"type": "triple point", "count": 1}]},
        "symmetry": {"kind": "point", "periodicity_rank": 0,
                     "schoenflies": "Td", "orbifold": "*332", "order": 24},
        "provenance": {"sources": [
            "J. Steiner, discovered in Rome in 1844; published posthumously by "
            "K. Weierstrass (1863)."]},
        "notes": {"caveats": [
            "A quartic mapping of the projective plane with three double lines "
            "meeting at a triple point -- unlike Boy's surface it is NOT an "
            "immersion, because of the six pinch points."]},
    },

    "cross-cap": {
        "primary_family": "topological", "tradition": ["classical"],
        "ids": {"mathworld": "Cross-Cap"},
        "curvature": {"condition": "none"},
        "topology": {"euler_characteristic": 1, "orientable": False,
                     "one_sided": True, "compact": True,
                     "non_orientable_genus": 1},
        "embedding": {"quality": "singular",
                      "singularities": [{"type": "pinch point", "count": 2}]},
        "provenance": {"sources": [
            "D. Hilbert and S. Cohn-Vossen, Anschauliche Geometrie (1932) -- "
            "Geometry and the Imagination, the standard exposition."]},
    },

    "morin-surface": {
        "discovered_by": "Bernard Morin", "year": 1978,
        "primary_family": "topological", "tradition": ["classical", "sculptural"],
        "ids": {"mathcurve": "ch1306_morin_2"},
        "curvature": {"condition": "none"},
        "topology": {"orientable": True, "compact": True,
                     "euler_characteristic": 2},
        "embedding": {"quality": "immersed", "self_intersecting": True},
        "provenance": {"sources": [
            "B. Morin's halfway model for the sphere eversion; see "
            "G. Francis, A Topological Picturebook (1987)."]},
        "notes": {"caveats": [
            "The halfway model of the sphere eversion: an immersed sphere that "
            "is isotopic to its own inside-out image, which is what makes the "
            "eversion possible. Morin was blind from the age of six."]},
    },

    "non-orientable-genus-k": {
        "name": "Non-Orientable Genus-k Surface",
        "primary_family": "topological", "tradition": ["classical"],
        "curvature": {"condition": "none"},
        "topology": {"orientable": False, "one_sided": True, "compact": True,
                     "euler_characteristic": "varies",
                     "non_orientable_genus": "varies"},
        "embedding": {"quality": "immersed", "self_intersecting": True},
        "specimens": [
            {"label": "N_1 = projective plane", "parameters": {"k": 1},
             "topology": {"euler_characteristic": 1, "non_orientable_genus": 1}},
            {"label": "N_2 = Klein bottle", "parameters": {"k": 2},
             "slug": "klein-bottle",
             "topology": {"euler_characteristic": 0, "non_orientable_genus": 2}},
            {"label": "N_3 = Dyck's surface", "parameters": {"k": 3},
             "topology": {"euler_characteristic": -1, "non_orientable_genus": 3}},
        ],
        "provenance": {"sources": [
            "The classification theorem of closed surfaces; W. von Dyck, "
            "'Beitrage zur Analysis situs', Math. Ann. 32 (1888) for N_3."]},
        "notes": {"caveats": [
            "One parameterised row subsumes Dyck's surface as the k = 3 case, "
            "which is why no separate Dyck record exists."]},
    },

    "genus-g-surface": {
        "name": "Genus-g Surface",
        "primary_family": "topological", "tradition": ["classical"],
        "curvature": {"condition": "none"},
        "topology": {"orientable": True, "compact": True,
                     "euler_characteristic": "varies", "genus": "varies",
                     "boundary_components": 0},
        "embedding": {"quality": "embedded"},
        "specimens": [
            {"label": "sphere (g = 0)", "parameters": {"g": 0}, "slug": "sphere",
             "topology": {"genus": 0, "euler_characteristic": 2}},
            {"label": "torus (g = 1)", "parameters": {"g": 1}, "slug": "torus",
             "topology": {"genus": 1, "euler_characteristic": 0}},
            {"label": "double torus (g = 2)", "parameters": {"g": 2},
             "topology": {"genus": 2, "euler_characteristic": -2}},
            {"label": "triple torus (g = 3)", "parameters": {"g": 3},
             "topology": {"genus": 3, "euler_characteristic": -4}},
        ],
        "provenance": {"sources": [
            "The classification theorem of closed orientable surfaces."]},
    },

    # ---------------- algebraic ----------------

    "clebsch-diagonal-cubic": {
        "discovered_by": "Alfred Clebsch", "year": 1871,
        "primary_family": "algebraic", "tradition": ["classical",
                                                     "physical-model"],
        "ids": {"mathworld": "ClebschDiagonalCubic",
                "mathcurve": "ch1231_clebsch_2"},
        "curvature": {"condition": "none"},
        "embedding": {"quality": "embedded", "singularities": []},
        # NOT Td. The Clebsch surface's celebrated S5 symmetry -- the one
        # that permutes its 27 lines -- is PROJECTIVE, acting on the five
        # homogeneous coordinates of the symmetric form. In this affine
        # embedding the Euclidean point group is only the S3 permuting
        # x, y, z, whose 3-fold axis is the body diagonal (1,1,1). The
        # symbolic check caught the overclaim: the polynomial is odd under
        # the sign flips Td contains.
        "symmetry": {"kind": "point", "periodicity_rank": 0,
                     "schoenflies": "C3v", "generator_set": "C3v_diag",
                     "order": 6, "orbifold": "*33",
                     "verified_by": "symbolic"},
        "topology": {"compact": True, "orientable": True},
        "metrics": {"normalization": "published"},
        "provenance": {"sources": [
            "A. Clebsch, 'Ueber die Flachen vierter Ordnung, welche eine "
            "Doppelcurve zweiten Grades besitzen', J. reine angew. Math. 69 "
            "(1868); the diagonal surface.",
            "A. Cayley and G. Salmon (1849) for the 27 lines on a cubic."]},
        "notes": {"caveats": [
            "The unique smooth cubic surface on which all 27 lines are REAL. "
            "Its ten Eckardt points are where three of the lines meet."]},
    },

    "cayley-nodal-cubic": {
        "discovered_by": "Arthur Cayley", "year": 1869,
        "primary_family": "algebraic", "tradition": ["classical",
                                                     "physical-model"],
        "ids": {"mathworld": "CayleyCubic", "mathcurve": "ch1225_cayley_2"},
        "curvature": {"condition": "none"},
        "embedding": {"quality": "singular", "is_record": True,
                      "record_for": "maximum nodes, degree 3",
                      "singularities": [{"type": "node (A1)", "count": 4}]},
        "symmetry": {"kind": "point", "periodicity_rank": 0,
                     "schoenflies": "Td", "order": 24, "verified_by": "curated"},
        "provenance": {"sources": [
            "A. Cayley, 'A Memoir on Cubic Surfaces', Phil. Trans. Roy. Soc. "
            "159 (1869)."]},
        "notes": {"caveats": [
            "Four nodes is the maximum for a cubic surface, so this is the "
            "degree-3 member of the record-nodal family that continues through "
            "Kummer, Barth, Labs, Endrass and Sarti."]},
    },

    "kummer-quartic": {
        "discovered_by": "Ernst Eduard Kummer", "year": 1864,
        "primary_family": "algebraic", "tradition": ["classical",
                                                     "physical-model"],
        "ids": {"mathworld": "KummerSurface", "mathcurve": "ch1294_kummer_2"},
        "curvature": {"condition": "none"},
        "embedding": {"quality": "singular", "is_record": True,
                      "record_for": "maximum nodes, degree 4",
                      "singularities": [{"type": "node (A1)", "count": 16}]},
        # Tetrahedral, but in the EDGE frame: math_art writes the four
        # tangent planes as 1 -+ z -+ sqrt2 x and 1 + z +- sqrt2 y, whose
        # normals point along (+-sqrt2, 0, -1) and (0, +-sqrt2, 1). Those
        # four normals have pairwise dot product -1/3, so the tetrahedron
        # is regular -- but its 3-fold axes are the normals themselves,
        # not the coordinate diagonals, so the standard-frame Td
        # generators do not apply. The mesh-symmetry check reported this
        # surface as asymmetric until the frame was named.
        "symmetry": {"kind": "point", "periodicity_rank": 0,
                     "schoenflies": "Td", "generator_set": "Td_edge",
                     "order": 24, "verified_by": "symbolic"},
        "provenance": {"sources": [
            "E. E. Kummer, 'Uber die Flachen vierten Grades mit sechzehn "
            "singularen Punkten', Monatsber. Akad. Wiss. Berlin (1864)."]},
        "notes": {"caveats": [
            "Sixteen nodes is the maximum for a quartic. The Kummer surface is "
            "a K3 surface -- the named specimen this database records, as "
            "opposed to the K3 CLASS, which is not a renderable object."]},
    },

    "barth-sextic": {
        "discovered_by": "Wolf Barth", "year": 1996,
        "primary_family": "algebraic", "tradition": ["classical"],
        "ids": {"mathworld": "BarthSextic", "mathcurve": "ch1209_barth_2"},
        "curvature": {"condition": "none"},
        "embedding": {"quality": "singular", "is_record": True,
                      "record_for": "maximum nodes, degree 6",
                      "singularities": [
                          {"type": "node (A1)", "count": 65,
                           "note": "50 in the finite part, 15 at infinity"}]},
        "symmetry": {"kind": "point", "periodicity_rank": 0,
                     "schoenflies": "Ih", "orbifold": "*532", "coxeter": "[5,3]",
                     "order": 120},
        "provenance": {"sources": [
            "W. Barth, 'Two projective surfaces with many nodes admitting the "
            "symmetries of the icosahedron', J. Algebraic Geometry 5 (1996)."]},
    },

    "barth-decic": {
        "discovered_by": "Wolf Barth", "year": 1996,
        "primary_family": "algebraic", "tradition": ["classical"],
        "ids": {"mathworld": "BarthDecic"},
        "curvature": {"condition": "none"},
        "embedding": {"quality": "singular", "is_record": True,
                      "record_for": "maximum known nodes, degree 10",
                      "singularities": [{"type": "node (A1)", "count": 345}]},
        "symmetry": {"kind": "point", "periodicity_rank": 0,
                     "schoenflies": "Ih", "orbifold": "*532", "order": 120},
        "provenance": {"sources": [
            "W. Barth, J. Algebraic Geometry 5 (1996) -- same work as the "
            "sextic."]},
    },

    "labs-septic": {
        "discovered_by": "Oliver Labs", "year": 2004,
        "primary_family": "algebraic", "tradition": ["classical"],
        "ids": {"mathworld": "LabsSeptic"},
        "curvature": {"condition": "none"},
        "embedding": {"quality": "singular", "is_record": True,
                      "record_for": "maximum known nodes, degree 7",
                      "singularities": [{"type": "node (A1)", "count": 99}]},
        "provenance": {"sources": [
            "O. Labs, 'A Septic with 99 real Nodes', Rend. Sem. Mat. Univ. "
            "Padova 116 (2006)."]},
        "notes": {"caveats": [
            "The coefficients are roots of an auxiliary cubic, so the equation "
            "cannot be reconstructed -- it must be transcribed from the source."]},
    },

    "endrass-octic": {
        "name": "Endrass Octic",
        "discovered_by": "Stephan Endrass", "year": 1995,
        "primary_family": "algebraic", "tradition": ["classical"],
        "ids": {"mathworld": "EndrassOctic"},
        "curvature": {"condition": "none"},
        "embedding": {"quality": "singular", "is_record": True,
                      "record_for": "maximum known nodes, degree 8",
                      "singularities": [{"type": "node (A1)", "count": 168}]},
        "provenance": {"sources": [
            "S. Endrass, 'A projective surface of degree eight with 168 "
            "nodes', J. Algebraic Geom. 6 (1997)."]},
        "notes": {"caveats": [
            "168 is the best known LOWER bound for degree 8; the upper bound "
            "is 174, so whether it is optimal is open."]},
    },

    "monkey-saddle": {
        "primary_family": "algebraic", "tradition": ["classical"],
        "ids": {"mathworld": "MonkeySaddle", "wikipedia": "Monkey saddle"},
        "curvature": {"condition": "none"},
        "topology": {"compact": False, "orientable": True},
        "embedding": {"quality": "embedded"},
        "provenance": {"sources": [
            "The standard example of a degenerate critical point of higher "
            "order; z = x^3 - 3xy^2 is the real part of z^3."]},
        "notes": {"caveats": [
            "Named for having three descending regions -- two for the legs and "
            "one for the tail."]},
    },

    # ---------------- quadrics ----------------

    "sphere": {
        "primary_family": "quadric", "tradition": ["classical"],
        "ids": {"mathworld": "Sphere", "mathcurve": "ch1257_sphere_2"},
        "curvature": {"condition": "k-const-positive",
                      "gaussian": {"exact": "1", "value": 1.0,
                                   "source": "classical"},
                      "mean": {"exact": "1", "value": 1.0, "source": "classical",
                               "note": "at unit radius, with the inward normal"},
                      "total_curvature": {"exact": "4*pi",
                                          "value": 12.566370614359172,
                                          "source": "classical",
                                          "note": "Gauss-Bonnet: 2*pi*chi"},
                      "willmore_energy": {"exact": "4*pi",
                                          "value": 12.566370614359172,
                                          "source": "classical",
                                          "note": "the Willmore minimum"}},
        "topology": {"genus": 0, "euler_characteristic": 2, "orientable": True,
                     "one_sided": False, "compact": True,
                     "boundary_components": 0},
        "symmetry": {"kind": "continuous", "periodicity_rank": 0,
                     "continuous": "O(3) -- the full orthogonal group",
                     "order": "infinite", "chiral": False},
        "embedding": {"quality": "embedded"},
        "metrics": {"normalization": "unit_radius",
                    "area": {"exact": "4*pi", "value": 12.566370614359172,
                             "source": "classical"},
                    "volume_enclosed": {"exact": "4*pi/3",
                                        "value": 4.1887902047863905,
                                        "source": "classical"},
                    "isoperimetric_quotient": {"exact": "1", "value": 1.0,
                                               "source": "classical"}},
        "provenance": {"sources": ["Classical."]},
        "notes": {"caveats": [
            "Simultaneously the K = +1 model, the CMC model, the Willmore "
            "minimiser and the isoperimetric extremum; the curvature facet "
            "records the strongest condition only (k-const-positive), with the "
            "rest in also_satisfies."]},
    },

    "torus": {
        "primary_family": "revolution", "tradition": ["classical"],
        "ids": {"mathworld": "Torus", "mathcurve": "ch1291_tore_2"},
        "curvature": {"condition": "none",
                      "total_curvature": {"exact": "0", "value": 0.0,
                                          "source": "classical",
                                          "note": "Gauss-Bonnet: chi = 0"}},
        "topology": {"genus": 1, "euler_characteristic": 0, "orientable": True,
                     "compact": True, "boundary_components": 0},
        "symmetry": {"kind": "continuous", "periodicity_rank": 0,
                     "continuous": "SO(2) about the axis, with reflections"},
        "embedding": {"quality": "varies"},
        "metrics": {"normalization": "unit_radius"},
        "specimens": [
            {"label": "ring torus (R > r)", "parameters": {"regime": "ring"},
             "embedding": {"quality": "embedded"}},
            {"label": "horn torus (R = r)", "parameters": {"regime": "horn"},
             "embedding": {"quality": "singular"}},
            {"label": "spindle torus (R < r)", "parameters": {"regime": "spindle"},
             "embedding": {"quality": "self-intersecting"}},
        ],
        "relations": {"inverse_of": "dupin-cyclide"},
        "provenance": {"sources": ["Classical."]},
    },

    "hyperboloid-one-sheet": {
        "name": "Hyperboloid of One Sheet",
        "primary_family": "quadric", "tradition": ["classical", "architectural"],
        "ids": {"mathworld": "One-SheetedHyperboloid",
                "mathcurve": "ch1286_hyperboloid1_2"},
        "curvature": {"condition": "none"},
        "topology": {"compact": False, "orientable": True, "genus": 0},
        "symmetry": {"kind": "continuous", "periodicity_rank": 0,
                     "continuous": "SO(2) about the axis"},
        "embedding": {"quality": "embedded"},
        "provenance": {"sources": ["Classical; one of the two doubly ruled "
                                   "quadrics."]},
        "notes": {"caveats": [
            "DOUBLY ruled: two distinct families of straight lines lie on it, "
            "which is what makes cooling towers and Shukhov's lattice towers "
            "buildable from straight members."]},
    },

    "hyperbolic-paraboloid": {
        "primary_family": "quadric", "tradition": ["classical", "architectural"],
        "ids": {"mathworld": "HyperbolicParaboloid",
                "mathcurve": "ch1285_paraboloidhyperbolic_2"},
        "curvature": {"condition": "none"},
        "topology": {"compact": False, "orientable": True},
        "embedding": {"quality": "embedded"},
        "provenance": {"sources": [
            "Classical; the second doubly ruled quadric. Felix Candela's shell "
            "structures are the architectural realisation."]},
    },

    # ---------------- cyclides, ruled, misc ----------------

    "dupin-cyclide": {
        "name": "Dupin Cyclide",
        "discovered_by": "Charles Dupin", "year": 1822,
        "primary_family": "cyclide", "tradition": ["classical",
                                                   "physical-model"],
        "ids": {"mathworld": "Cyclide", "mathcurve": "ch1245_cyclidededupin_2"},
        "curvature": {"condition": "none"},
        "topology": {"compact": True, "orientable": True, "genus": 1},
        "embedding": {"quality": "varies"},
        "metrics": {"normalization": "unit_radius"},
        "specimens": [
            {"label": "ring", "embedding": {"quality": "embedded"},
             "topology": {"genus": 1}},
            {"label": "horn", "embedding": {"quality": "singular"},
             "topology": {"genus": 1}},
            {"label": "spindle", "embedding": {"quality": "self-intersecting"},
             "topology": {"genus": 1}},
        ],
        "relations": {"inverse_of": "torus"},
        "provenance": {"sources": [
            "C. Dupin, Applications de geometrie et de mechanique (1822).",
            "J. Schrott and B. Odehnal, 'Ortho-circles of Dupin cyclides', "
            "J. Geometry and Graphics 10 (2006)."]},
        "notes": {"caveats": [
            "The classical surface all of whose curvature lines are CIRCLES. "
            "Obtained by inverting a torus, cylinder or cone in a sphere, which "
            "is why the three types are regimes of one formula rather than "
            "three surfaces."]},
    },

    "whitney-umbrella": {
        "discovered_by": "Hassler Whitney", "year": 1943,
        "primary_family": "ruled", "tradition": ["classical"],
        "ids": {"mathworld": "WhitneyUmbrella", "mathcurve": "ch1355_whitney_2"},
        "curvature": {"condition": "none"},
        "embedding": {"quality": "singular",
                      "singularities": [{"type": "pinch point", "count": 1},
                                        {"type": "double line", "count": 1}]},
        "provenance": {"sources": [
            "H. Whitney, 'The general type of singularity of a set of 2n-1 "
            "smooth functions of n variables', Duke Math. J. 10 (1943)."]},
        "notes": {"caveats": [
            "The canonical pinch-point singularity, and the reason it appears "
            "twice in math_art: once parametrically as a ruled surface and once "
            "implicitly as x^2 - y^2 z = 0 in the Hauser gallery."]},
    },

    "right-conoid": {
        "name": "Right Conoid",
        "primary_family": "ruled", "tradition": ["classical"],
        "ids": {"mathworld": "RightConoid", "mathcurve": "ch1239_conoid_2"},
        "curvature": {"condition": "none"},
        "embedding": {"quality": "varies"},
        "specimens": [
            {"label": "Plucker's cylindroid", "embedding": {"quality": "self-intersecting"}},
            {"label": "n-fold conoid", "embedding": {"quality": "self-intersecting"}},
            {"label": "Wallis' conical edge", "embedding": {"quality": "self-intersecting"}},
            {"label": "Zindler's conoid", "embedding": {"quality": "self-intersecting"}},
        ],
        "provenance": {"sources": [
            "J. Plucker, Neue Geometrie des Raumes (1868) -- the cylindroid.",
            "J. Wallis, for the conical edge.",
            "K. Zindler, for the conoid z(x^2 - y^2) = 2axy."]},
    },

    "oloid": {
        "discovered_by": "Paul Schatz", "year": 1929,
        "primary_family": "ruled", "tradition": ["classical", "sculptural"],
        "ids": {"mathworld": "Oloid", "wikipedia": "Oloid"},
        "curvature": {"condition": "flat"},
        "topology": {"compact": True, "orientable": True, "genus": 0},
        "embedding": {"quality": "embedded"},
        "provenance": {"sources": [
            "P. Schatz, Rhythmusforschung und Technik (1975); the oloid dates "
            "from his 1929 work on the invertible cube."]},
        "notes": {"caveats": [
            "The convex hull of two perpendicular circles each passing through "
            "the other's centre. DEVELOPABLE -- it can be unrolled flat -- "
            "which is why it is filed under `ruled` rather than `misc`, and it "
            "rolls with its whole surface touching the ground."]},
    },

    "schwarz-lantern": {
        "name": "Schwarz's Lantern",
        "discovered_by": "Hermann Amandus Schwarz", "year": 1880,
        "primary_family": "discrete", "tradition": ["classical"],
        # No mathcurve id: the mirror has no Schwarz-lantern chapter, and
        # ch1340_sinus_2 -- which an earlier revision pointed at -- is
        # Ferreol's SINE SURFACE. The cross-check caught it by comparing
        # the page title against the record name.
        "curvature": {"condition": "none"},
        "topology": {"compact": True, "orientable": True, "genus": 0,
                     "boundary_components": 2},
        "embedding": {"quality": "embedded"},
        "provenance": {"sources": [
            "H. A. Schwarz, 'Sur une definition erronee de l'aire d'une surface "
            "courbe', Gesammelte Mathematische Abhandlungen (1890)."]},
        "notes": {"caveats": [
            "THE cautionary counterexample: a polyhedral approximation to a "
            "cylinder whose area DIVERGES as the mesh is refined. Any generator "
            "that measures a discrete surface and assumes convergence is wrong "
            "in exactly this way."]},
    },

    "gabriels-horn": {
        "name": "Gabriel's Horn",
        "alternate_names": ["Torricelli's trumpet"],
        "discovered_by": "Evangelista Torricelli", "year": 1643,
        "primary_family": "revolution", "tradition": ["classical"],
        "ids": {"mathworld": "GabrielsHorn", "mathcurve": "ch1269_gabriel_2"},
        "curvature": {"condition": "none"},
        "topology": {"compact": False, "orientable": True},
        "embedding": {"quality": "embedded"},
        "metrics": {"area": None, "measures_note":
                    "Surface area is INFINITE while the enclosed volume is "
                    "finite (pi) -- which is the whole point of the object, so "
                    "a null with this note is the honest record."},
        "provenance": {"sources": [
            "E. Torricelli, De solido hyperbolico acuto (1643)."]},
    },

    "zoll-surface": {
        "name": "Zoll's Surface",
        "discovered_by": "Otto Zoll", "year": 1903,
        "primary_family": "revolution", "tradition": ["classical"],
        "ids": {"mathworld": "ZollSurface"},
        "curvature": {"condition": "none"},
        "topology": {"compact": True, "orientable": True, "genus": 0,
                     "euler_characteristic": 2},
        "embedding": {"quality": "embedded"},
        "provenance": {"sources": [
            "O. Zoll, 'Uber Flachen mit Scharen geschlossener geodatischer "
            "Linien', Math. Ann. 57 (1903).",
            "J. Tannery (1892) for the earlier, non-smooth example."]},
        "notes": {"caveats": [
            "Smooth, not the round sphere, and yet EVERY geodesic closes up "
            "with the same length -- which for a long time the sphere was "
            "believed to be the only surface to do."]},
    },

    "tannery-pear": {
        "name": "Tannery's Pear",
        "discovered_by": "Jules Tannery", "year": 1892,
        "primary_family": "revolution", "tradition": ["classical"],
        "ids": {"mathcurve": "ch1316_tannery_2"},
        "curvature": {"condition": "none"},
        "topology": {"compact": True, "orientable": True, "genus": 0},
        "embedding": {"quality": "singular",
                      "singularities": [{"type": "conical point", "count": 2}]},
        "specimens": [
            {"label": "pear (half the figure eight)"},
            {"label": "hourglass (the whole figure eight)"},
        ],
        "relations": {"generalizes": [], "specialises": ["zoll-surface"]},
        "provenance": {"sources": [
            "J. Tannery (1892); the surface of revolution of half a Gerono "
            "lemniscate."]},
        "notes": {"caveats": [
            "All geodesics close, like Zoll's surface, but Tannery's is NOT "
            "smooth -- both tips are conical points, and repairing exactly that "
            "is what Zoll's surface achieves."]},
    },
}


def facts_for(slug):
    """Curated facts for `slug`, or an empty dict.

    Includes the Evolver cell definition where one has been read out of
    a datafile -- see `_evolver_facts` below.
    """
    out = dict(FACTS.get(slug, {}))
    cell = _evolver_facts(slug)
    if cell:
        defn = dict(out.get('definition') or {})
        defn.update(cell['definition'])
        out['definition'] = defn
    extra = _definition_facts(slug)
    if extra:
        defn = dict(out.get('definition') or {})
        defn.update(extra)
        out['definition'] = defn
    return out


# What DEFINES a minimal surface, alongside the Evolver cell below.
#
# An algebraic record carries its polynomial; `gyroid.json` carried
# "mode: nodal, level 0, cubic lattice" and never said which nodal
# surface, though the equation is in `minsurf/tpms.py`.  Both kinds of
# defining data are now generated into `surface_defs.py` by
# `tools/extract_definitions.py` -- READ OUT of the shipped builders, not
# transcribed into them, per the rule `barth-decic` states: an unverified
# transcription silently defines a different surface.
_DEF_INDEX = None


def _definition_index():
    """slug -> the defining data extracted for its generator row."""
    global _DEF_INDEX
    if _DEF_INDEX is not None:
        return _DEF_INDEX
    _DEF_INDEX = {}
    try:
        from .surface_defs import NODAL_POLYNOMIAL, WEIERSTRASS
        from . import mapping as _map
    except ImportError:
        return _DEF_INDEX
    # The tables are keyed by generator ROW; records are keyed by slug,
    # and `mapping` is where that correspondence is already declared.
    row_to_slug = {}
    # PROMOTE too, and its values are (slug, family) tuples rather than
    # bare slugs -- which is where Schwarz P, D and the gyroid live,
    # promoted out of the associate family into records of their own.
    # Reading only ALIAS and MERGE found twenty surfaces and missed
    # exactly the three best-known ones.
    for table in (getattr(_map, 'ALIAS', {}), getattr(_map, 'MERGE', {}),
                  getattr(_map, 'PROMOTE', {}), getattr(_map, 'SPECIMEN', {})):
        for ref, target in table.items():
            if ':' not in ref:
                continue
            slug = target[0] if isinstance(target, (tuple, list)) else target
            if isinstance(slug, str):
                row_to_slug.setdefault(ref.split(':', 1)[1], slug)
    for key, expr in NODAL_POLYNOMIAL.items():
        slug = row_to_slug.get(key)
        if slug:
            _DEF_INDEX.setdefault(slug, {})['nodal_polynomial'] = expr
    for key, data in WEIERSTRASS.items():
        slug = row_to_slug.get(key)
        if slug:
            _DEF_INDEX.setdefault(slug, {})['weierstrass'] = data
    return _DEF_INDEX


def _definition_facts(slug):
    return dict(_definition_index().get(slug, {}))


# Evolver cell definitions, generated by tools/bake_fe_cells.py from the
# datafile mirror.  Merged here so a surface's contour, symmetry
# generators and cell word live in its RECORD, which is where the
# definition of a surface belongs -- the datafiles are not in this repo.
def _evolver_facts(slug):
    try:
        from . import fecells as _fe
    except ImportError:
        return {}
    return _fe.facts_for(slug)

def polynomial_for(slug):
    """Candidate closed form for `slug`, or None. MUST be verified."""
    return POLYNOMIAL.get(slug)


def _selftest():
    """Structural checks on the curation table; raises on failure."""
    from . import views

    for slug, rec in FACTS.items():
        assert slug == slug.lower(), slug
        assert " " not in slug, slug

        # a record that states a curvature condition must use a legal one
        cond = (rec.get("curvature") or {}).get("condition")
        if cond is not None:
            assert cond in views.CURVATURE_PRECEDENCE, (slug, cond)

        # every curated record must cite something real
        prov = rec.get("provenance") or {}
        if prov:
            assert prov.get("sources"), "%s has a provenance block with no sources" % slug

        # periodicity rank is 0..3 where given
        rank = (rec.get("symmetry") or {}).get("periodicity_rank")
        if rank is not None:
            assert rank in (0, 1, 2, 3), (slug, rank)

        # a 'varies' field must be resolved on every specimen -- the rule
        # that stops 'varies' being used to dodge stating a value
        emb = (rec.get("embedding") or {}).get("quality")
        if emb == "varies":
            specs = rec.get("specimens")
            assert specs, "%s marks embedding 'varies' but lists no specimens" % slug
            for sp in specs:
                assert (sp.get("embedding") or {}).get("quality"), \
                    "%s specimen %r does not resolve embedding.quality" % (
                        slug, sp.get("label"))

        # specimen slugs, where given, must look like slugs
        for sp in rec.get("specimens") or []:
            assert sp.get("label"), slug
            if sp.get("slug"):
                assert sp["slug"] == sp["slug"].lower(), (slug, sp["slug"])

    # polynomials must at least PARSE in the exact language; whether they
    # are the right polynomial is decided at build time against the oracle
    from . import expr
    for slug, poly in POLYNOMIAL.items():
        names = expr.free_names(poly)
        assert names <= {"x", "y", "z"}, (slug, names)

    print("RESULT: OK  (surfdb.curation, %d curated slugs, %d candidate "
          "polynomials)" % (len(FACTS), len(POLYNOMIAL)))


# ---------------------------------------------------------------------------
# The join to data/polyhedra/: which polyhedron each surface is the smooth
# or continuous counterpart of.
#
# Only edges that RESOLVE are listed.  The most interesting join in the
# whole design does NOT resolve and is recorded as a known limitation
# rather than forced: the Petrie-Coxeter apeirohedra (mucube,
# muoctahedron, mutetrahedron) are the polygonal ancestors of the
# triply-periodic minimal surfaces -- research/taxonomy.md says so
# directly -- but data/polyhedra/ catalogues FINITE polyhedra only, so
# `mucube` is not a slug there and pointing at it would produce a
# dangling reference. See data/surfaces/README.md.
# ---------------------------------------------------------------------------

POLYHEDRAL_ANALOGUE = {
    # Goursat level sets: the smooth surface carrying a regular solid's
    # symmetry, in the literal sense that the family is DEFINED as the
    # polynomials invariant under that solid's group.
    "rounded-dodecahedron": "dodecahedron",
    "rounded-icosahedron": "icosahedron",
    "rounded-tetrahedron-0-lt-k-lt-4": "tetrahedron",
    "tetrahedral-cubic-k-eq-0": "tetrahedron",
    "tetrahedral-cubic-k-lt-0": "tetrahedron",
    "tetrahedral-cubic-k-gt-4": "tetrahedron",
    "triconic-tetrahedral-quartic": "tetrahedron",
    "cube-diagonal-quartic": "cube",
    "cube-edge-quartic": "cube",
    "cube-median-quartic": "cube",
    "octahedron-edge-quartic": "octahedron",
    "cuboctahedral-quartic": "cuboctahedron",
    "cuboctahedral-triangle-quartic": "cuboctahedron",
    "quartic-with-12-cuboctahedral-nodes": "cuboctahedron",
    "icosidodecahedral-pentagon-sextic": "icosidodecahedron",
    "icosidodecahedral-triangle-sextic": "icosidodecahedron",
    "six-icosidodecahedral-planes": "icosidodecahedron",
    "barth-sextic-icosahedral-frame": "icosahedron",

    # Record-nodal surfaces whose node arrangement carries the solid's
    # symmetry.
    "barth-sextic": "icosahedron",
    "barth-decic": "icosahedron",
    "cayley-nodal-cubic": "tetrahedron",
    "kummer-quartic": "tetrahedron",
    "clebsch-diagonal-cubic": "tetrahedron",
    "sarti-dodecic": "icosahedron",

    # Schwarz's lantern is a POLYHEDRON inscribed in a cylinder -- the
    # counterexample that cost the subject its first definition of surface
    # area. Its polyhedral nature is the whole point, so the analogue runs
    # the other way from the rest of this table.
    "schwarz-lantern": "hexagonal-prism",

    # Steinmetz solids are boolean intersections of cylinders on a
    # polyhedron's axis set; the family record names the simplest.
    "steinmetz-solid": "cube",
}


def polyhedral_analogue(slug):
    """The `data/polyhedra` slug this surface is the counterpart of."""
    return POLYHEDRAL_ANALOGUE.get(slug)


# Two records carry a `known_issue` written from what the drive stage
# actually observed rather than from reading the code. Both are findings
# ABOUT math_art, which is the point of driving the generators at all.
FACTS["scherkt-surface"] = {
    "notes": {"known_issue":
              "FOUND BY THE DRIVE STAGE. 'SCHERKT' is a row in "
              "math_art/minsurf/tpms.py's TPMS registry and builds fine "
              "through minsurf.build_tpms, but NO operator offers it: "
              "mesh.periodic_minimal_add's surface enum omits it under "
              "every one of its five periodicity settings (SINGLY, DOUBLY, "
              "TRIPLY, EXACT, EXACT_FAMILY). The surface is reachable from "
              "the engine and not from the user interface."},
}
