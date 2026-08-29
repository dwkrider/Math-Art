"""The Ferreol pass: closing the gap against the mathcurve surfaces mirror.

The mirror (S:/data/math_art/references/websites/mathcurve/book/surfaces/)
holds 181 chapters of Robert Ferreol's *Encyclopedie des formes
mathematiques remarquables*.  Before this pass 64 of them resolved from a
record's `ids.mathcurve`; the remaining 117 were read one by one and
sorted into exactly three piles:

  * `ids()`     -- 19 chapters describe a surface that already has a
                   record and only lacked the cross-reference.  Every id
                   below was verified by opening the mirror page and
                   checking that the chapter really is that surface (the
                   page BODY, not just the H1 -- two H1s in the mirror
                   are conversion artifacts, see the notes).
  * `records()` -- 14 chapters name a surface with no record at all.
                   Per the house rule these carry NO transcribed
                   equations: an unverified coefficient does not error,
                   it silently defines a different surface, and the
                   numerical oracle that guards the shipped algebraic
                   block cannot guard a surface with no implementation.
                   Definitions are stated in words; the equations stay in
                   the cited chapter.
  * SKIPPED     -- 84 chapters, each with its stated reason.  The big
                   groups: concept/notion pages (an asymptotic line is
                   not an object), classification classes (the scope rule
                   that keeps Enriques out keeps "ruled surface" out),
                   function-parameterised families with no canonical
                   member, higher-dimensional objects, index pages -- and,
                   importantly, EIGHT chapters whose surface is already
                   implemented in math_art as a mode, specimen or special
                   case of an existing record (Zindler, Plucker, Wallis,
                   the superellipsoid, the Clifford torus, the circled
                   and developable helicoids, the Goursat family).  A
                   records() entry is hard-coded `implemented: false`, so
                   writing one for an implemented surface would misstate
                   the ledger in the flattering-to-fix direction; whether
                   those eight deserve promotion to their own records is
                   an integrator decision, recorded here rather than made
                   here.

Six correct ids below draw an identity flag from crosscheck.py.  Five
share no title word with their record's name because of Ferreol's
filing (Riemann's minimal surface under "Skew catenoid", the
Jorge-Meeks k-noids under "Trinoid", Bouligand's pillow under
"Cushion", the canal surface under "Tube", the genus-g surface under
"n-holed torus"); the sixth is the associate family under "Minimal
helicoid", where the checker's hyphen collapsing turns
"Catenoid-Helicoid" into one word that no longer contains "helicoid".
Each flag is the honest trace of a filing or normalisation quirk, not
a wrong id -- every page's body names the record's surface explicitly.
(A seventh flag, on the new `coil` record, traces a mirror H1 typo,
"Coild"; see that record's note.)
"""

# --------------------------------------------------------------------
# citation helper
# --------------------------------------------------------------------

_MIRROR = "S:/data/math_art/references/websites/mathcurve/book/surfaces"


def _mc(title, stem):
    return ("R. Ferreol, \"%s\", Encyclopedie des formes mathematiques "
            "remarquables, mathcurve.com; local mirror chapter %s "
            "(%s/%s.md)." % (title, stem, _MIRROR, stem))


# --------------------------------------------------------------------
# 1. Cross-references for records that already exist
# --------------------------------------------------------------------
# slug of an EXISTING record -> resolved chapter stem.  Every stem was
# read off the mirror and its page checked against the record.

IDS_EXISTING = {
    # title matches record name directly
    "spheroid": "ch1195_ellipsoidrevol_2",          # "Spheroid"
    "cross-cap": "ch1243_bonnetcroise_2",           # "Cross-cap"
    "darboux-surface": "ch1251_darboux_2",          # "Darboux surface";
    #   math_art/helical_surface_generator.py's own reference block cites
    #   this very chapter ("surface de Darboux") for its DARBOUX mode.
    "fresnel-elasticity-surface": "ch1262_elasticite_2",
    #   H1 "Elasticity surface"; body header "FRESNEL'S ELASTICITY
    #   SURFACE" with the record's quartic.  Distinct from the WAVE
    #   surface, which gets a new record below.
    "focal-surface": "ch1267_focale_2",             # "Focal" (of a surface)
    "helico-conical-surface": "ch1278_helicoconique_2",
    "oloid": "ch1311_orthobicycle_2",               # "Oloid"
    "monkey-saddle": "ch1334_selle_2",              # "Monkey saddle"
    "seifert-surface": "ch1338_seifert_2",          # "Seifert surface"
    "sine-torus": "ch1351_toredeklein_2",           # "Sine torus"
    "darboux-cyclide": "ch1194_cyclid_2",
    #   "Cyclide" -- the page opens "A Darboux cyclide" and defines the
    #   general (Darboux) cyclide; the Dupin chapter is separate and
    #   already resolved on dupin-cyclide.
    "cassinian-surface-3-poles": "ch1220_casinienne3d_2",
    #   "Cassinian surface", the n-pole family prod MA_i = b^n of which
    #   the record's 3-pole surface is the shipped member; the n = 2
    #   case has its own chapter, already resolved on cassini-surface.
    "catenoid-helicoid-associate-family": "ch1282_helicoidminimal_2",
    #   "Minimal helicoid" -- Scherk's 1834 "helcats": the one-parameter
    #   cos(alpha)/sin(alpha) interpolation between helicoid and
    #   catenoid, i.e. exactly the record's associate family.
    "finite-riemann-plane-2-catenoids": "ch1330_weber_2",
    #   "Riemann finite minimal surface" (Hoffman-Karcher 1993, the
    #   Lopez-Ros construction); stem says "weber" but the page is this
    #   surface.

    # correct ids whose chapter TITLE shares no word with the record
    # name -- each verified from the page body; crosscheck will log an
    # identity disagree, which is the honest trace of Ferreol's filing.
    "canal-surface": "ch1217_tube_2",
    #   "Tube" -- the page's own synonym line: "canal surface, channel
    #   surface"; mathcurve has no separate canal chapter.
    "riemann-minimal-example": "ch1223_catenoidgauche_2",
    #   "Skew catenoid" -- body header "SKEW CATENOID, RIEMANN'S MINIMAL
    #   SURFACE", studied by Riemann in 1860: the periodic minimal
    #   surface foliated by circles that the record implements.
    "genus-g-surface": "ch1261_tn_4",
    #   "n-holed torus" -- the OBJECT page for the genus-n orientable
    #   surface.  Deliberately not ch1271_genre_4, which defines the
    #   genus as a concept (see the note in sources.py).
    "jorge-meeks-k-noid": "ch1309_trinoide_2",
    #   "Trinoid, n-noid" -- "surface studied by Jorge and Meeks in
    #   1983"; the record is that family.  (math_art/trinoid.py is the
    #   CMC-1 Bryant trinoid, a different surface -- not this page.)
    "bouligands-pillow": "ch1318_coussin_2",
    #   "Cushion" -- body header "BOULIGAND'S PILLOW", Bouligand 1928,
    #   with the record's surface.
}

# --------------------------------------------------------------------
# 2. New records: named surfaces with no record at all
# --------------------------------------------------------------------

_NO_EQ = (
    "Not built.  The defining equation is deliberately NOT transcribed "
    "here: an unverified coefficient does not error, it silently defines "
    "a different surface, and the numerical oracle that guards the "
    "shipped algebraic block cannot guard a surface with no "
    "implementation to check against."
)

_RESUME_ALGEBRAIC = (
    "Transcribe the Cartesian equation from the cited mirror chapter "
    "into math_art/surfaces/algebraic.py; the build's oracle check then "
    "confirms it against that implementation."
)

_RESUME_RULED = (
    "math_art/ruled_surface_generator.py already carries the cone and "
    "conoid machinery (build_conoid and the CONOID variant table); add "
    "a row and take the parametrization from the cited mirror chapter."
)

_RESUME_REVOLUTION = (
    "A meridian-profile surface-of-revolution row; the radial builders "
    "in math_art/curiosity_surface_generator.py (build_radial, "
    "build_fresnel) are the pattern to follow, and the parametrization "
    "is in the cited mirror chapter."
)

_RESUME_QUADRIC = (
    "The quadric block is the database's biggest honest gap (10 of 13 "
    "quadrics unimplemented -- data/surfaces/README.md, Coverage); "
    "implement together with the other quadrics of revolution rather "
    "than piecemeal."
)


def records():
    """slug -> spec dict, in the shape surfdb_build.MISSING expects."""
    out = {}

    # -- algebraic ---------------------------------------------------
    out["fresnel-wave-surface"] = {
        "name": "Fresnel's Wave Surface",
        "family": "algebraic", "mode": "implicit",
        "blocked_by": _NO_EQ,
        "resume": _RESUME_ALGEBRAIC,
        "sources": [
            _mc("Fresnel's wave surface", "ch1201_ondes_2"),
            "A. Fresnel (1821), per the chapter; the chapter also points "
            "to Jules Richard's thesis and [Loria 3d p. 197].",
        ],
        "extra": {
            "curvature": {"condition": "none"},
            "discovered_by": "Augustin Fresnel",
            "year": 1821,
            "tradition": ["classical", "physical"],
            "definition": {"note":
                "The quartic locus of light-wavefront propagation in a "
                "biaxial anisotropic crystal: two sheets, with four "
                "conical points that gave conical refraction.  Distinct "
                "from Fresnel's ELASTICITY surface, which is already "
                "implemented."},
        },
    }

    out["mobius-surface"] = {
        "name": "M\u00f6bius Surface",
        "family": "algebraic", "mode": "parametric",
        "blocked_by": _NO_EQ,
        "resume": _RESUME_ALGEBRAIC,
        "sources": [_mc("Mobius surface", "ch1302_mobiussurface_2")],
        "extra": {
            "curvature": {"condition": "none"},
            "embedding": {"quality": "self-intersecting"},
            "tradition": ["classical"],
            "definition": {"note":
                "The rotoidal algebraic model of the Mobius strip: a "
                "segment carried around a circle at half the rotation "
                "rate sweeps a cubic surface that contains the classic "
                "half-twist band.  Ferreol keeps it as its own chapter, "
                "separate from the Mobius-strip chapter; the strip's "
                "other concrete models already carry records "
                "(twisted-strip, meeks-mobius-strip, "
                "sudanese-mobius-band, bjorling-twisted-band)."},
        },
    }

    out["sine-surface"] = {
        "name": "Sine Surface",
        "family": "algebraic", "mode": "parametric",
        "blocked_by": _NO_EQ,
        "resume": _RESUME_ALGEBRAIC,
        "sources": [
            _mc("Sine surface", "ch1340_sinus_2"),
            "A. Gray, \"Modern Differential Geometry of Curves and "
            "Surfaces with Mathematica\", 2nd ed., CRC Press (1997) -- "
            "the chapter credits Gray with the name.",
        ],
        "extra": {
            "curvature": {"condition": "none"},
            "embedding": {"quality": "self-intersecting"},
            "year": 1997,
            "tradition": ["classical", "gallery"],
            "definition": {"note":
                "The surface (sin u, sin v, sin(u+v)): a sextic that is "
                "a union of ellipses in three ways, with pinch points "
                "where the sheets meet.  NOT the surface Ferreol's "
                "lantern-hunting almost mislabeled: the note in "
                "sources.py records that this chapter is not Schwarz's "
                "lantern."},
        },
    }

    # -- quadrics of revolution --------------------------------------
    out["cone-of-revolution"] = {
        "name": "Cone of Revolution",
        "family": "quadric", "mode": "implicit",
        "blocked_by": "Not built -- like most of the quadric family. "
                      "Blender's own cone primitive is not a math_art "
                      "construction and is not counted as coverage.",
        "resume": _RESUME_QUADRIC,
        "sources": [_mc("Cone of revolution", "ch1229_conederevolution_2")],
        "extra": {
            "curvature": {"condition": "none"},
            "alternate_names": ["Right circular cone"],
            "tradition": ["classical"],
            "definition": {"note":
                "Revolution of a line secant to an axis about that "
                "axis; the special case of the elliptic cone (whose "
                "record already resolves ch1235) with circular "
                "cross-section."},
        },
    }

    out["paraboloid-of-revolution"] = {
        "name": "Paraboloid of Revolution",
        "family": "quadric", "mode": "implicit",
        "blocked_by": "Not built -- like most of the quadric family "
                      "(the elliptic paraboloid record is also "
                      "unimplemented).",
        "resume": _RESUME_QUADRIC,
        "sources": [_mc("Paraboloid of revolution",
                        "ch1314_paraboloidrevolution_2")],
        "extra": {
            "curvature": {"condition": "none"},
            "alternate_names": ["Circular paraboloid"],
            "tradition": ["classical"],
            "definition": {"note":
                "Revolution of a parabola about its axis: the mirror "
                "shape that focuses a parallel beam to a point; the "
                "circular special case of the elliptic paraboloid."},
        },
    }

    # -- ruled cones and conoids -------------------------------------
    out["helicoidal-cone"] = {
        "name": "Helicoidal Cone",
        "family": "ruled", "mode": "parametric",
        "blocked_by": "Not built.  math_art's HELICAL_CONE (the compound "
                      "helical cone / Solomonic column, a fluted conical "
                      "envelope) is a different surface despite the "
                      "similar name.",
        "resume": _RESUME_RULED,
        "sources": [_mc("Helicoidal cone (Cone helicoidal)",
                        "ch1064_conehelicoidal")],
        "extra": {
            "curvature": {"condition": "none"},
            "tradition": ["classical"],
            "definition": {"note":
                "The cone joining a fixed apex to a circular helix: "
                "rulings (a u cos v, a u sin v, b u v).  NOTE: the "
                "mirror file's H1 reads \"Elliptic cone\" -- a "
                "conversion artifact; the page body is headed "
                "HELICOIDAL CONE and describes this surface, so the "
                "identity flag crosscheck records against this id is a "
                "defect of the mirror's title line, not of the id."},
        },
    }

    out["sinusoidal-cone"] = {
        "name": "Sinusoidal Cone",
        "family": "ruled", "mode": "parametric",
        "blocked_by": "Not built.",
        "resume": _RESUME_RULED,
        "sources": [_mc("Sinusoidal cone", "ch1236_conesinusoidal_2")],
        "extra": {
            "curvature": {"condition": "none"},
            "tradition": ["classical"],
            "definition": {"note":
                "The cone with apex at the origin over the closed "
                "spherical sinusoid: cylindrical equation "
                "z = k rho cos(n theta).  The n = 1/2 member has a "
                "Viviani-curve directrix.  Distinct from Plucker's "
                "conoid z = a cos(n theta), whose rulings do not pass "
                "through one point and which already ships as a "
                "right-conoid specimen."},
        },
    }

    out["parabolic-conoid"] = {
        "name": "Parabolic Conoid",
        "family": "ruled", "mode": "parametric",
        "blocked_by": "Not built.  The shipped conoid modes (Plucker, "
                      "n-fold, Wallis, Zindler, Whitney) do not include "
                      "a parabolic directrix.",
        "resume": _RESUME_RULED,
        "sources": [_mc("Parabolic conoid", "ch1240_conoide_parabolique_2")],
        "extra": {
            "curvature": {"condition": "none"},
            "tradition": ["classical", "architectural"],
            "definition": {"note":
                "The conoid whose curved directrix is a parabola; the "
                "right case with the parabola's axis parallel to the "
                "directrix line degenerates to Whitney's umbrella "
                "(which has its own record).  Used in roof shells."},
        },
    }

    # -- surfaces of revolution --------------------------------------
    out["solid-of-maximal-attraction"] = {
        "name": "Solid of Maximal Attraction",
        "family": "revolution", "mode": "parametric",
        "blocked_by": "Not built.",
        "resume": _RESUME_REVOLUTION,
        "sources": [
            _mc("Solid of maximal attraction", "ch1205_attraction_2"),
            "The chapter credits the Marquis de Saint-Jacques (1750) "
            "and C. F. Gauss (1830).",
        ],
        "extra": {
            "curvature": {"condition": "none"},
            "year": 1750,
            "tradition": ["classical", "physical"],
            "definition": {"note":
                "The shape of the homogeneous solid that maximises "
                "gravitational attraction at a boundary point: a "
                "surface of revolution with spherical equation "
                "r = a sqrt(sin(latitude))."},
        },
    }

    out["second-tractroid"] = {
        "name": "Second Tractroid",
        "family": "revolution", "mode": "parametric",
        "blocked_by": "Not built.",
        "resume": _RESUME_REVOLUTION,
        "sources": [
            _mc("Tractroid 2", "ch1352_tractroid_2"),
            "Proposed by Ludovic Schwob, per the chapter.",
        ],
        "extra": {
            "curvature": {"condition": "none"},
            "alternate_names": ["Tractroid 2"],
            "tradition": ["classical"],
            "definition": {"note":
                "Revolution of the tractrix about the axis "
                "PERPENDICULAR to its asymptote -- the companion to the "
                "pseudosphere (revolution about the asymptote), which "
                "already resolves ch1210.  Unlike the pseudosphere it "
                "is not of constant curvature; its volume involves "
                "Catalan's constant."},
        },
    }

    out["revolution-of-the-catenary"] = {
        "name": "Revolution of the Catenary",
        "family": "revolution", "mode": "parametric",
        "blocked_by": "Not built.",
        "resume": _RESUME_REVOLUTION,
        "sources": [_mc("Revolution of the catenary (Alysseid)",
                        "ch1366_alysseid_2")],
        "extra": {
            "curvature": {"condition": "none"},
            "alternate_names": ["Alysseid"],
            "tradition": ["classical"],
            "definition": {"note":
                "Revolution of the catenary about its AXIS of symmetry "
                "(z = a cosh(rho / a)) -- the companion to the catenoid, "
                "which rotates the catenary about its base and already "
                "resolves ch1198.  Not minimal."},
        },
    }

    out["revolution-of-the-sinusoid"] = {
        "name": "Revolution of the Sinusoid",
        "family": "revolution", "mode": "parametric",
        "blocked_by": "Not built.",
        "resume": _RESUME_REVOLUTION,
        "sources": [_mc("Revolution of the sinusoid", "ch1328_revolsin_2")],
        "extra": {
            "curvature": {"condition": "none"},
            "year": 2012,
            "tradition": ["classical"],
            "definition": {"note":
                "Revolution of the sinusoid x = a cos(z/b) about Oz: a "
                "string of onion-dome beads meeting in cusp circles "
                "where the meridian crosses the axis; the chapter dates "
                "its study to 2012."},
        },
    }

    out["rotation-surface-with-proportional-curvatures"] = {
        "name": "Rotation Surface with Proportional Curvatures",
        "family": "revolution", "mode": "parametric",
        "blocked_by": "Not built.",
        "resume": _RESUME_REVOLUTION,
        "sources": [
            _mc("Rotation surface with proportional curvatures",
                "ch1365_revolpropor_2"),
            "W. Kuhnel, \"Differential Geometry: Curves - Surfaces - "
            "Manifolds\", p. 95, where the family is worked (the "
            "chapter's own reference, dated 1999).",
        ],
        "extra": {
            # the defining relation kappa_1 = k * kappa_2 is exactly a
            # Weingarten relation, and it is what the page states
            "curvature": {"condition": "weingarten"},
            "year": 1999,
            "tradition": ["classical"],
            "definition": {"note":
                "The surfaces of revolution whose principal curvatures "
                "are in a constant ratio k: meridian rho = a cos^k(t). "
                "k = 1 gives the sphere, k = -1 a minimal surface "
                "(catenoid); the four sign/size regimes of k give "
                "spindles, domes, trumpets and annular waists."},
        },
    }

    # -- swept -------------------------------------------------------
    out["coil"] = {
        "name": "Coil",
        "family": "swept", "mode": "parametric",
        "blocked_by": "Not built as a named surface.  The shipped "
                      "Darboux sweep (helical_surface_generator, DARBOUX "
                      "mode with helical motion of a circle) produces "
                      "circled helicoids, but its generatrix rides in "
                      "the meridian plane, not perpendicular to the "
                      "motion, so the tube-around-a-helix proper has no "
                      "operator.",
        "resume": "A tube of constant radius swept along a circular "
                  "helix; either extend the Darboux generatrix "
                  "orientation or reuse the tube machinery; the "
                  "parametrization is in the cited mirror chapter.",
        "sources": [_mc("Coil (Serpentin)", "ch1233_serpentin_2")],
        "extra": {
            "curvature": {"condition": "none"},
            "alternate_names": ["Helical tube", "Spring",
                                "Toroidal helicoid"],
            "tradition": ["classical"],
            "definition": {"note":
                "The tube whose bore is a circular helix -- the spring "
                "or serpentine: h > 0 right-handed, h = 0 degenerates "
                "to the torus, h < 0 left-handed.  NOTE: the mirror "
                "file's H1 reads \"Coild\", a typo carried over from "
                "conversion; the page body is headed COIL."},
        },
    }

    return out


# --------------------------------------------------------------------
# ids for the NEW records above (curate() applies this table to every
# record, including the MISSING ones, so the new records resolve too)
# --------------------------------------------------------------------

IDS_NEW = {
    "fresnel-wave-surface": "ch1201_ondes_2",
    "mobius-surface": "ch1302_mobiussurface_2",
    "sine-surface": "ch1340_sinus_2",
    "cone-of-revolution": "ch1229_conederevolution_2",
    "paraboloid-of-revolution": "ch1314_paraboloidrevolution_2",
    "helicoidal-cone": "ch1064_conehelicoidal",     # H1 artifact, see note
    "sinusoidal-cone": "ch1236_conesinusoidal_2",
    "parabolic-conoid": "ch1240_conoide_parabolique_2",
    "solid-of-maximal-attraction": "ch1205_attraction_2",
    "second-tractroid": "ch1352_tractroid_2",
    "revolution-of-the-catenary": "ch1366_alysseid_2",
    "revolution-of-the-sinusoid": "ch1328_revolsin_2",
    "rotation-surface-with-proportional-curvatures": "ch1365_revolpropor_2",
    "coil": "ch1233_serpentin_2",                   # H1 "Coild" typo
}


def ids():
    out = {}
    for slug, stem in IDS_EXISTING.items():
        out[slug] = {"mathcurve": stem}
    for slug, stem in IDS_NEW.items():
        out[slug] = {"mathcurve": stem}
    return out


def invariants_for(slug):
    return {}


# --------------------------------------------------------------------
# 3. The skip ledger: 84 chapters deliberately NOT curated, with reasons
# --------------------------------------------------------------------
# A ledger that hides what it declined is not a ledger.  Grouped by the
# rule that excludes each; every stem here was read, not guessed.

SKIPPED = {
    # -- index / annex pages (not chapters about one surface) ---------
    "ch1016_superficies": "alphabetical index page (French mirror copy)",
    "ch1017_superfice": "alphabetical index page (French mirror copy)",
    "ch1018_surfaces_2": "the English alphabetical index page",
    "ch1367_cubique_reglee_demo_2":
        "untranslated proof annex to the skew ruled cubic chapter "
        "(classification of ruled cubics), not a surface page",
    "ch1368_double_sept_2":
        "proof annex to the Clebsch chapter: nonexistence of a "
        "'double seven' line configuration; not a surface",
    "ch1256_doublesix_2":
        "Schlafli's double six: a configuration of 12 lines (annex to "
        "the Clebsch chapter), not a surface",

    # -- concept / notion pages (out of scope: not objects) -----------
    "ch1189_betti_2": "concept: Betti number of a surface",
    "ch1203_asymptotic_4": "concept: asymptotic lines of a surface",
    "ch1204_planasymptotic_2": "concept: asymptotic plane of a ruled "
                               "surface's generatrix",
    "ch1227_eulerpoincare_2": "concept: Euler characteristic",
    "ch1237_conicpoint_2": "concept: conical point of a surface",
    "ch1259_indicatricededupin_2": "concept: the Dupin indicatrix",
    "ch1271_genre_4": "concept: genus (the OBJECT page ch1261_tn_4 now "
                      "resolves from genus-g-surface instead)",
    "ch1319_pince_2": "concept: pinch point",
    "ch1322_meplat_2": "concept: planar point",
    "ch1354_ombilic_2": "concept: umbilic",
    "ch1341_lisse_4": "concept: smoothness of a surface",
    "ch1345_surface_2": "concept: what a (topological) surface is",
    "ch1297_variete_2": "concept: manifold",
    "ch1298_espace_2": "concept: 3-dimensional manifold",
    "ch1344_sommeconnexe_2": "operation: connected sum of two surfaces",
    "ch1200_applicable_2": "concept: local isometry (applicability)",
    "ch1206_axoid_2": "concept: the axoid of a rigid motion",
    "ch1312_unilatere_2": "concept: orientability / one-sidedness",
    "ch1265_retournable_2": "property: surfaces invariant under a "
                            "half-turn",
    "ch1346_symetrierotation_2": "property: rotational symmetry",
    "ch1348_tendue_2": "property: tightness (minimal total absolute "
                       "curvature)",
    "ch1347_tripleorthog_2": "concept: triple orthogonal systems",
    "ch1350_toretopologic_2": "concept: the topological torus as a "
                              "homeomorphism class; the ring torus "
                              "already resolves ch1291",

    # -- derived-surface METHOD pages (operations, not objects) -------
    "ch1191_enveloppe_6": "method: envelope of a family of surfaces",
    "ch1292_inverse_4": "method: inversion of a surface",
    "ch1317_podaire_4": "method: pedal of a surface",
    "ch1324_polaire_4": "method: reciprocal polar",
    "ch1305_orthoptic_4": "method: orthoptic surface of a surface",
    "ch1315_parallele_6": "method: parallel (offset) surfaces",
    "ch1224_caustic_4": "method: caustic of a surface (the FOCAL page, "
                        "its evolute analogue, now resolves from "
                        "focal-surface)",
    "ch1190_equidistance_4": "method: equidistance surface of two "
                             "surfaces",
    "ch1310_normalie_2": "method: normal surface along a curve",

    # -- classification classes (the rule that keeps Enriques/K3 out) -
    "ch1197_algebricsu_2": "class: algebraic surfaces",
    "ch1244_cubic_4": "class: cubic surfaces (named members ship)",
    "ch1326_quartic_4": "class: quartic surfaces",
    "ch1325_quadric_2": "class: quadrics (the classification page; "
                        "individual quadrics have records)",
    "ch1327_rationnelle_5": "class: rational surfaces",
    "ch1333_reglee_2": "class: ruled surfaces",
    "ch1253_developpable_2": "class: developable surfaces",
    "ch1299_revolution_2": "class: surfaces of revolution",
    "ch1301_minimale_2": "class: minimal surfaces",
    "ch1247_cylindre_2": "class: general cylinders",
    "ch1234_cone_2": "class: general cones",
    "ch1221_catalan_5": "class: Catalan (plane-directrix ruled) "
                        "surfaces -- not Catalan's minimal surface, "
                        "which already resolves ch1222",
    "ch1228_cerclee_2": "class: circled surfaces",
    "ch1199_anallagmaticsu_2": "class: inversion-invariant surfaces",
    "ch1212_bispheric_2": "class: bispherical algebraic surfaces",
    "ch1296_liouville_2": "class: Liouville metrics",
    "ch1357_weingarten_2": "class: Weingarten surfaces",
    "ch1313_ovoid_2": "class: ovoids (defined by convexity, no "
                      "defining equation)",
    "ch1272_geoid_2": "the geoid: a two-sentence geophysics stub with "
                      "no mathematical definition to record",
    "ch1276_guthrie_2": "Guthrie's solid: a combinatorial configuration "
                        "of 2n parallelepipeds (map-colouring), not a "
                        "surface",

    # -- function-parameterised families with no canonical member -----
    "ch1246_cyclotomique_2": "family parameterised by an arbitrary "
                             "directrix function (cyclotomic surfaces)",
    "ch1303_moulure_2": "family parameterised by two arbitrary curves "
                        "(molding surfaces)",
    "ch1304_monge_2": "family parameterised by arbitrary curves (Monge "
                      "surfaces)",
    "ch1332_rotoide_2": "family parameterised by arbitrary spine and "
                        "generatrix (rotoids / generalized helicoids)",
    "ch1353_translation_2": "family: translation surfaces z=f(x)+g(y) "
                            "and generalisations",
    "ch1343_regleedirectricerecti_2": "family: ruled surfaces with a "
                                      "straight directrix",
    "ch1283_helicoidregle_2": "family: ruled helicoids (the shipped "
                              "ruled HELICOID mode covers the right "
                              "and oblique cases)",
    "ch1211_beziersu_2": "CAD construction: Bezier surfaces from "
                         "arbitrary control nets",
    "ch1241_patchcoons_2": "CAD construction: the Coons patch from "
                           "arbitrary boundary arcs",
    "ch1364_developablepli_2": "study of folded developables "
                               "(Fuchs-Tabachnikov), a construction on "
                               "arbitrary developables",

    # -- not surfaces in R^3 ------------------------------------------
    "ch1288_s3_2": "the 3-sphere: a 3-manifold",
    "ch1289_sn_2": "the n-sphere",
    "ch1290_tndim_2": "the n-dimensional torus",
    "ch1268_fortunatus_2": "Fortunatus's purse: needs a fourth "
                           "dimension, per the page itself",
    "ch1356_veronese_2":
        "the Veronese surface lives in R^5; this repo realises it only "
        "through its 3D shadows, recorded as steiner-surface (which "
        "already resolves ch1342).  The dormant veronese-surface alias "
        "in sources.py is left for the day the R^5 object gets a "
        "projection-family record of its own -- an integrator call.",
    "ch1321_planprojectif_2":
        "the abstract real projective plane; its concrete immersions "
        "(Boy, Roman, cross-cap) each carry a record already",
    "ch1208_mobius_2":
        "the Mobius strip as abstract object; its concrete models "
        "carry records (twisted-strip n=1, meeks-mobius-strip, "
        "sudanese-mobius-band, bjorling-twisted-band) and the rotoidal "
        "algebraic model is the new mobius-surface record (ch1302)",

    # -- already implemented as a mode / specimen / special case ------
    # A records() entry is hard-coded implemented:false, so writing one
    # for these would misstate the ledger; promotion to their own
    # records is an integrator decision.
    "ch1359_zindler_2": "Zindler's conoid: shipped as the ZINDLER "
                        "specimen of the right-conoid record "
                        "(ruled_surface_generator)",
    "ch1323_plucker_2": "Plucker's conoid: shipped as the PLUCKER "
                        "cylindroid / n-fold conoid specimens of the "
                        "right-conoid record",
    "ch1238_coinconic_2": "the conocuneus of Wallis: shipped as the "
                          "WALLIS conical-edge specimen of the "
                          "right-conoid record",
    "ch1295_lame_4": "the Lame surface (superellipsoid): shipped as "
                     "the SUPERELLIPSOID mode of supershape_generator, "
                     "carried by the supershape record",
    "ch1232_clifford_2": "Clifford's torus: produced by the Hopf "
                         "fibration generator's great-circle case (its "
                         "Willmore solver's own reference minimiser); "
                         "carried by the hopf-torus record",
    "ch1281_helicoiddeveloppable_2":
        "the developable helicoid: the shipped TANGENT_DEV mode IS the "
        "tangent developable of a helix, and its record "
        "(tangent-developable) already resolves the general chapter "
        "ch1254; one mathcurve id per record",
    "ch1280_helicoidcercle_2":
        "the circled helicoid: produced by the shipped Darboux sweep "
        "(DARBOUX mode, helical motion of a circle generatrix), "
        "carried by the darboux-surface record (now id ch1251)",
    "ch1274_goursat_4":
        "the Goursat octahedral quartic family: implemented across "
        "many member records (cube-edge/diagonal/median quartics, "
        "cuboctahedral quartics, rounded solids...); a single "
        "unimplemented family record would contradict them, and the "
        "dormant goursat-surface alias in sources.py waits on an "
        "integrator-owned family record",

    # -- duplicate coverage of an already-resolved subject ------------
    "ch1273_helicoiddroit_2":
        "the right helicoid: the helicoid record already resolves the "
        "main helicoid chapter ch1279; one mathcurve id per record",
}


# --------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------

def _selftest():
    """Structural checks on the tables; raises on failure."""
    import re

    from . import sources

    _FAMILIES = {"minimal", "minimal-periodic", "cmc", "constant-curvature",
                 "algebraic", "quadric", "ruled", "revolution", "swept",
                 "cyclide", "topological", "spectral", "discrete", "derived",
                 "physical", "misc"}
    _MODES = {"implicit", "parametric", "weierstrass", "nodal",
              "variational", "swept", "derived", "discrete"}

    recs = records()
    assert len(recs) == 14, len(recs)
    for slug, spec in recs.items():
        assert re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", slug), slug
        assert spec.get("name"), slug
        assert spec.get("family") in _FAMILIES, (slug, spec.get("family"))
        assert spec.get("mode") in _MODES, (slug, spec.get("mode"))
        assert spec.get("blocked_by"), \
            "%s is absent without a stated reason" % slug
        assert spec.get("resume"), "%s has no resume pointer" % slug
        assert spec.get("sources"), "%s cites nothing" % slug
        # the mirror chapter must be cited by stem
        assert any(IDS_NEW[slug] in s for s in spec["sources"]), \
            "%s does not cite its own chapter" % slug
        # no record may claim a defining datum -- that is the point
        d = (spec.get("extra") or {}).get("definition") or {}
        assert not d.get("polynomial"), \
            "%s must not carry an untranscribed polynomial" % slug
        for k in ("x", "y", "z", "gauss_map"):
            assert not d.get(k), \
                "%s must not carry an untranscribed chart" % slug
        assert d.get("note"), "%s has no definition note" % slug
        assert slug in IDS_NEW, "%s has no chapter id" % slug

    table = ids()
    assert len(table) == len(IDS_EXISTING) + len(IDS_NEW) == 33, len(table)
    assert not set(IDS_EXISTING) & set(IDS_NEW)

    stems_used = []
    for slug, entry in table.items():
        assert set(entry) == {"mathcurve"}, slug
        stem = entry["mathcurve"]
        assert re.match(r"^ch\d+_", stem), (slug, stem)
        stems_used.append(stem)
    assert len(stems_used) == len(set(stems_used)), \
        "two records claim the same chapter"

    # no stem may collide with one already claimed via the alias table,
    # and none may be one this module itself declines.  The alias table
    # carries DORMANT entries -- slugs with no record (trinoid, and the
    # goursat/veronese/zindler slugs discussed in SKIPPED); the trinoid
    # alias names the same chapter this module resolves onto the
    # jorge-meeks-k-noid record, which IS that family, so it is exempt.
    _dormant = {"trinoid", "goursat-surface", "veronese-surface",
                "zindler-conoid"}
    alias_stems = {st for sl, st in sources.MATHCURVE_ALIAS.items()
                   if sl not in _dormant}
    clash = set(stems_used) & alias_stems
    assert not clash, "stem already claimed by sources.py alias: %r" % clash
    clash = set(stems_used) & set(SKIPPED)
    assert not clash, "stem both used and skipped: %r" % clash
    for stem in SKIPPED:
        assert re.match(r"^ch\d+_", stem), stem

    # against the real mirror if it is mounted: every id and every skip
    # must RESOLVE -- a cross-reference to a page the reader does not
    # have is exactly the failure mode this module exists to prevent
    if sources.mirror_present():
        pool = set(sources.stems())
        bad = [s for s in stems_used if s not in pool]
        assert not bad, "id points at a stem not in the mirror: %r" % bad[:5]
        bad = [s for s in SKIPPED if s not in pool]
        assert not bad, "skip names a stem not in the mirror: %r" % bad[:5]

    print("RESULT: OK  (surfdb.ferreol, %d records, %d id sets: %d for "
          "existing records, %d for new; %d chapters skipped with reasons)"
          % (len(recs), len(table), len(IDS_EXISTING), len(IDS_NEW),
             len(SKIPPED)))
