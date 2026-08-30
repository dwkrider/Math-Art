"""Cross-references and gap records from two mirrored surface sources.

Sources (both mirrored locally, so every id resolves offline):

  * The 3DXM Virtual Math Museum, ``vmm/book/surface/`` -- 94 exhibit
    pages (the ``conformal/``, ``mathart/`` and ``other/`` sections hold
    conformal maps, artist galleries and site notes, not surfaces).
    Ids are the mirror chapter stems, e.g. ``ch010_wavy_enneper``.
  * Matthias Weber's minimal-surface repository, minimalsurfaces.blog,
    ``minsurf/book/`` -- 335 repository pages and 84 posts.  Chapter
    numbering is global across ``repository/`` (ch001-ch335) and
    ``posts/`` (ch336-ch419), so a bare stem is unambiguous; the
    selftest resolves each stem against both directories.

Contract, shared by ferreol.py and vmm.py:

  records()  -> {slug: spec}, spec in the shape tools/surfdb/tail.py uses:
      {"name", "family", "mode", "blocked_by", "resume", "sources",
       "extra": {...deep-merged onto the record...}}
    Used for surfaces with NO record yet.  Every entry carries a real
    citation and a stated reason it is unimplemented.  Three entries are
    NON-EXISTENCE records (Weber keeps a section of constructions that
    were tried and ruled out): their `blocked_by` says the surface is
    proved -- or, where only numerical evidence exists, believed -- not
    to exist, which is a stronger statement than "not built".

  ids()      -> {slug: {"vmm": "chNNN_stem", "msblog": "chNNN_stem"}}
    Used for surfaces that ALREADY have a record.  Each id was verified
    by opening the page and matching its content (not merely its title)
    against the record's mathematics; a wrong cross-reference is worse
    than none.  New records carry their ids in their own spec `extra`.

Equations are deliberately NOT transcribed (no polynomial / x /
gauss_map): those fields are guarded by numerical oracles elsewhere, and
an unverified transcription would silently define a different surface.
"""

import os

VMM_SURFACE_DIR = "S:/data/math_art/references/websites/vmm/book/surface"
MS_BOOK_DIRS = (
    "S:/data/math_art/references/websites/minsurf/book/repository",
    "S:/data/math_art/references/websites/minsurf/book/posts",
)


def _vmm_src(stem, title):
    return ("The 3DXM Consortium, '%s', Virtual Math Museum, "
            "virtualmathmuseum.org (local mirror: vmm/book/surface/%s.md)."
            % (title, stem))


def _ms_src(stem, title):
    return ("M. Weber, '%s', minimalsurfaces.blog (local mirror: "
            "minsurf/book/.../%s.md)." % (title, stem))


KARCHER_TOKYO = (
    "H. Karcher, 'Construction of minimal surfaces', Surveys in Geometry, "
    "Univ. of Tokyo, 1989, and Lecture Notes No. 12, SFB 256, Bonn (1989) "
    "1-96 (the 'Tokyo notes' the mirrored pages cite for their formulas).")
KARCHER_1988 = (
    "H. Karcher, 'Embedded minimal surfaces derived from Scherk's "
    "examples', Manuscripta Math. 62 (1988) 83-114.")
SCHOEN_NASA = (
    "A. H. Schoen, 'Infinite periodic minimal surfaces without "
    "self-intersections', NASA Technical Note TN D-5541 (1970).")

MS_BLOCKED = (
    "Documented as a named surface on Matthias Weber's minimal-surface "
    "repository (mirrored locally, including its Mathematica notebook and "
    "POV-Ray parameter sweeps); no construction has been built in math_art.")
MS_RESUME = (
    "The local mirror recovered the page's Mathematica notebook (193 "
    "notebooks; see the mirror's catalog.md); transcribe its Weierstrass "
    "data into math_art/minsurf/ along the zoo route -- "
    "research/minimal_surfaces_status.md is the worked path.")
VMM_BLOCKED = (
    "Named exhibit in the 3DXM Virtual Math Museum (mirrored locally); no "
    "construction has been built in math_art.")
VMM_RESUME = (
    "The mirrored VMM page links the exhibit's 3D-XplorMath documentation "
    "PDF; reconstruct the Weierstrass data from it (or from Karcher's "
    "cited Tokyo notes) and follow the zoo route in math_art/minsurf/; "
    "see research/minimal_surfaces_status.md.")

_MIN = {"curvature": {"condition": "minimal"},
        "topology": {"complete": True, "compact": False},
        "tradition": ["classical"]}
_ROD = {"curvature": {"condition": "minimal"},
        "symmetry": {"kind": "rod", "periodicity_rank": 1},
        "topology": {"complete": True, "compact": False},
        "metrics": {"normalization": "unit_cell"},
        "tradition": ["classical"]}
_LAYER = {"curvature": {"condition": "minimal"},
          "symmetry": {"kind": "layer", "periodicity_rank": 2},
          "topology": {"complete": True, "compact": False},
          "metrics": {"normalization": "unit_cell"},
          "tradition": ["classical"]}
_SPACE = {"curvature": {"condition": "minimal"},
          "symmetry": {"kind": "space", "periodicity_rank": 3},
          "topology": {"complete": True, "compact": False},
          "metrics": {"normalization": "unit_cell"},
          "tradition": ["crystallographic"]}


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
# Verified against page content, not titles alone.  Notes on the less
# obvious identifications:
#   * ch011_planar_enneper -> richmond-surface AND richmond-generalized:
#     the page ("fitting plane with Enneper surface", rational Weierstrass
#     data, higher-order dihedral variants) is the planar-Enneper family,
#     i.e. Richmond's surfaces; the base member and the g = z^k family
#     both have records, so the one family page serves both.
#   * ch018_catenoid_chain -> alternating-fence-of-half-catenoids: the
#     page's surface has Gauss map c*sin(z), half-catenoids growing
#     alternately up and down out of a plane -- Weber's "alternating
#     fence of half-catenoids" (his posts page ch406 is the same object).
#   * ch019_inverted_boy -> kusner-projective-plane-p-planar-ends: the
#     page's minimal surface is parametrized by the sphere with antipodal
#     points identified and has three planar ends -- Kusner's projective
#     plane with p = 3, whose inversion is Boy's surface.  (The VMM
#     'Kusner Surface' page ch020 is the EVEN case, an immersed sphere
#     with 8 planar ends -- a new record below, not this one.)
#   * ch023_catenoid_fence -> fence-of-catenoids-karcher (rectangular
#     tori, designed handle, Karcher-Hoffman); msblog ch170 confirms
#     "which Hermann Karcher calls a Fence of Catenoids".
#   * ch028_scherk_w_handle -> karcher-scherk-with-handles-doubly-
#     periodic: the VMM page is the genus-one member (KWH formulas) of
#     the doubly periodic Karcher-Scherk-with-handles family that the
#     record covers (msblog ch032 is the family page).
#   * ch032_schwarz_pd_family serves both schwarz-p and schwarz-d: it is
#     the museum's only page for either surface (the 2-parameter PD
#     family whose members they are).
#   * ch052_hyperbolic_k1_sor -> minding-surface: the record's generator
#     (math_art.hyperbolic_surface_generator) builds both Minding types;
#     the hyperbolic-type page is attached, the conic-type page ch051 is
#     the same record's other regime and is noted here rather than
#     attached (one id slot).
#   * ch056_two_soliton -> multi-soliton-pseudospherical: the record
#     covers the 2-, 3- and 4-soliton surfaces (VMM pages ch056, ch058,
#     ch059) and the breather+soliton page ch057; the first is attached.
#   * ch065_paraboloid -> elliptic-paraboloid: VMM's rotational
#     paraboloid is the circular special case of the record.
#   * ch137 (Scherk's singly periodic) -> scherk-saddle-tower: the
#     record is Karcher's saddle-tower family, whose 4-ended member IS
#     Scherk's singly periodic surface; VMM's family page is ch015.
#   * ch170_translation_invariant_catenoid -> fence-of-catenoids-karcher:
#     the page's outward-handle surface "which Karcher calls a Fence of
#     Catenoids"; its inward-handle NON-existent variant is the new
#     catenoid-with-handle record below.
#   * ch259_box_symmetry_type_4 -> triply-periodic-costa: Weber's
#     Simoes-Batista post links exactly this page as "Batista's triply
#     periodic Costa surface".
#   * ch282 'Schoen H"-R' -> weber-h2r, ch288 "Schoen T'-R'" ->
#     weber-trr: the tail records' names are ASCII spellings of the same
#     surfaces.
#   * ch286_schoen_r2 -> schoen-rii: the page's 45-45-90 triangular
#     cylinders match the record's "isosceles right triangular prism
#     cell".
#   * ch290 "Schoen's Unnamed Surface 12 (F-RD(r))" -> schoen-frd-r.
#   * ch372 (CHM of higher genus) -> callahan-hoffman-meeks-chm-1-2-
#     genus-4: the post's family CHM(1,k) contains the record's CHM(1,2).
#   * ch409 -> helicoid-with-handle-genus-1: the record's operator key
#     GENUS1_HELICOID sits in the SINGLY-periodic family, i.e. it is the
#     Hoffman-Karcher-Wei translation-invariant helicoid with handle,
#     which is this posts page.
# ---------------------------------------------------------------------------

VMM_IDS = {
    "catenoid-helicoid-associate-family": "ch003_helicoid_catenoid",
    "scherk-doubly-periodic": "ch004_scherk",
    "henneberg-surface": "ch005_henneberg",
    "catalan-surface": "ch006_catalan",
    "riemann-minimal-example": "ch008_riemann",
    "double-enneper": "ch009_double_enneper",
    "richmond-surface": "ch011_planar_enneper",
    "richmond-generalized-g-eq-z-k": "ch011_planar_enneper",
    "sphere-catenoid-enneper-end": "ch012_catenoid_enneper",
    "scherk-saddle-tower": "ch015_saddle_tower",
    "helicoidal-karcher-scherk-twisted-tower": "ch016_twisted_scherk",
    "alternating-fence-of-half-catenoids": "ch018_catenoid_chain",
    "kusner-projective-plane-p-planar-ends": "ch019_inverted_boy",
    "chen-gackstatter": "ch021_chen_gackstatter",
    "fence-of-catenoids-karcher": "ch023_catenoid_fence",
    "karcher-scherk-with-handles-doubly-periodic": "ch028_scherk_w_handle",
    "costa-hoffman-meeks": "ch029_costa_h_m",
    "h-exact": "ch030_schwarz_h_family",
    "lidinoid": "ch031_lidinoid",
    "schwarz-p": "ch032_schwarz_pd_family",
    "schwarz-d": "ch032_schwarz_pd_family",
    "gyroid": "ch033_gyroid",
    "neovius-surface": "ch034_neovius",
    "cross-cap": "ch042_cross_cap",
    "boys-surface": "ch044_boys_apery",
    "steiner-surface": "ch046_steiner",
    "klein-bottle": "ch049_klein_bottle",
    "pseudosphere": "ch050_pseudosphere",
    "minding-surface": "ch052_hyperbolic_k1_sor",
    "dini-surface": "ch053_dini",
    "breather-surface": "ch054_breather",
    "kuen-surface": "ch055_kuen",
    "multi-soliton-pseudospherical": "ch056_two_soliton",
    "k-positive-revolution": "ch060_k1_sor",
    "sieverts-surface": "ch061_sievert_enneper",
    "spherical-helicoid": "ch062_spherical_helicoid",
    "unduloid": "ch063_unduloid",
    "ellipsoid": "ch064_ellipsoid",
    "elliptic-paraboloid": "ch065_paraboloid",
    "hyperbolic-paraboloid": "ch066_hyperbolic_paraboloid",
    "hyperboloid-one-sheet": "ch067_hyperboloid1",
    "hyperboloid-two-sheets": "ch068_hyperboloid2",
    "sphere": "ch069_sphere",
    "torus": "ch070_torus",
    "dupin-cyclide": "ch071_cyclide",
    "bianchi-pinkall-flat-torus": "ch072_bianchi_pinkall_tori",
    "hopf-torus": "ch073_hopf_fibered",
    "constant-width-solid": "ch075_constant_width_surface",
    "whitney-umbrella": "ch076_whitney_umbrella",
    "right-conoid": "ch077_right_conoid",
    "monkey-saddle": "ch079_monkey_saddle",
    "cayley-nodal-cubic": "ch081_cayley_cubic",
    "clebsch-diagonal-cubic": "ch082_clebsch_cubic",
    "kummer-quartic": "ch083_kummer",
    "barth-sextic": "ch084_barth_sextic",
    "pretzel": "ch085_pretzel",
    "pilz-surface": "ch086_pilz",
    "bretzel2": "ch087_bretzel2",
    "bretzel5": "ch088_bretzel5",
    "orthocircles": "ch089_orthocircles",
    "deco-cube": "ch090_decocube",
    "join-of-two-tori": "ch092_join_2_tori",
}

MS_IDS = {
    # classical / spheres
    "catenoid": "ch202_the_catenoid",
    "helicoid": "ch155_the_helicoid",
    "enneper-surface": "ch189_enneper_surface",
    "catalan-surface": "ch129_catalans_minimal_surface",
    "henneberg-surface": "ch384_henneberg_surface",
    "double-enneper": "ch176_double_enneper",
    "finite-riemann-plane-2-catenoids": "ch190_finite_riemann",
    "sphere-catenoid-enneper-end":
        "ch195_sphere_with_one_catenoid_and_one_enneper_end",
    "lopez-sphere-2-ends-of-index-2": "ch196_sphere_with_two_ends_of_index_2",
    "enneper-with-n-catenoids": "ch211_enneper_with_n_catenoids",
    "jorge-meeks-k-noid": "ch222_jorge_meeks_k_noids",
    "prismatic-k-noid": "ch193_prismatic_k_noids",
    "pyramidal-k-noid": "ch194_pyramidal_k_noids",
    "bipyramidal-k-noid": "ch175_bipyramidal_k_noids",
    "antiprismatic-k-noid-full-family": "ch174_antiprismatic_k_noids",
    "k-noid-with-enneper-ends": "ch192_k_noids_with_enneper_ends",
    "kusner-projective-plane-p-planar-ends":
        "ch065_kusners_spheres_with_planar_ends",
    "lopez-minimal-klein-bottle": "ch063_f_lopez_1_ended_klein_bottle",
    # tori / higher genus
    "costa-surface": "ch243_the_costa_surface",
    "chen-gackstatter": "ch234_chen_gackstatter_surface",
    "chen-gackstatter-higher-genus":
        "ch058_higher_genus_chen_gackstatter_surfaces",
    "costa-wohlgemuth-4-ends": "ch053_costa_wohlgemuth_surfaces",
    "wohlgemuth-second-surface-genus-3": "ch418_wohlgemuths_second_surface",
    "catenoid-enneper-higher-genus":
        "ch377_catenoid_enneper_surfaces_of_genus_g",
    "torus-with-catenoid-2-annular-ends":
        "ch401_singly_periodic_torus_with_one_catenoid_and_two",
    "torus-with-2-enneper-2-annular-ends":
        "ch413_translation_invariant_torus_with_2_enneper_and_2",
    "translation-invariant-enneper-3-annular-ends":
        "ch412_translation_invariant_torus_with_1_enneper_and_3",
    # symmetrizations
    "symmetrized-double-enneper": "ch199_symmetrized_double_enneper",
    "symmetrized-finite-riemann-2m-catenoids":
        "ch228_symmetrized_finite_riemann",
    "symmetrized-chen-gackstatter-k-fold-genus-k-1":
        "ch224_symmetrized_chen_gackstatter",
    "symmetrized-chen-gackstatter-2-level-genus-2-k-1":
        "ch223_symmetrized_chen_gackstatter_g2n",
    "symmetrized-chen-gackstatter-3-level-genus-3-k-1":
        "ch404_symmetrized_chen_gackstatter_g3k",
    # singly periodic
    "riemann-minimal-example": "ch135_riemanns_singly_periodic_surface",
    "scherk-saddle-tower": "ch137_scherks_singly_periodic_surface",
    "scherk-enneper": "ch136_scherk_enneper",
    "periodic-enneper": "ch134_periodic_enneper_surface",
    "six-ended-scherk-tower": "ch125_6_ended_scherk_g0",
    "six-ended-scherk-tower-genus-1": "ch126_6_ended_scherk_g1",
    "eight-ended-scherk-tower-genus-2": "ch127_8_ended_scherk_g2",
    "costa-scherk-tower-genus-1": "ch130_costa_scherk_surface",
    "dasilva-batista-surface-genus-2": "ch131_dasilva_batista_surface",
    "helicoidal-karcher-scherk-twisted-tower":
        "ch132_helicoidal_karcher_scherk_surfaces",
    "callahan-hoffman-meeks-singly-periodic":
        "ch128_callahan_hoffman_meeks_surfaces",
    "callahan-hoffman-meeks-chm-1-2-genus-4":
        "ch372_callahan_hoffman_meeks_surfaces_of_higher_genus",
    "screw-motion-chm-tower":
        "ch156_the_screw_motion_invariant_callahan_hoffman_meek",
    "translation-invariant-costa": "ch171_translation_invariant_costa_i",
    "helicoid-with-handle-genus-1":
        "ch409_the_translation_invariant_helicoid_with_handle",
    "catenoid-tower-with-handle-genus-2":
        "ch162_translation_invariant_catenoid_with_a_handle",
    "catenoid-tower-with-2-handles-genus-3":
        "ch163_translation_invariant_catenoid_with_two_handles",
    "fence-of-catenoids-karcher": "ch170_translation_invariant_catenoid",
    "alternating-fence-of-half-catenoids":
        "ch406_the_alternating_fence_of_half_catenoids",
    "fischer-koch-tower-translation-invariant":
        "ch411_translation_invariant_fischer_koch_surfaces",
    "fischer-koch-freese-twisted": "ch369_fischer_koch_freese_surfaces",
    # doubly periodic
    "scherk-doubly-periodic": "ch046_scherks_doubly_periodic_surface",
    "tilted-scherk-doubly-periodic": "ch048_tilted_scherk",
    "karcher-scherk-with-handles-doubly-periodic":
        "ch032_doubly_periodic_karcher_scherk_surfaces_of_highe",
    "karcher-meeks-rosenberg-doubly-periodic":
        "ch035_karcher_meeks_rosenberg_surfaces",
    "wei-doubly-periodic-genus-2":
        "ch052_weis_doubly_periodic_surface_of_genus_2",
    "wei-higher-genus-tower-doubly-periodic": "ch033_higher_genus_wei_surfaces",
    "rossman-thayer-wohlgemuth-doubly-periodic":
        "ch040_rossman_thayer_wohlgemuth_m",
    "connor-experimental-doubly-periodic":
        "ch023_connors_experimental_surfaces_of_genus_3",
    # triply periodic
    "gyroid": "ch271_gyroid",
    "lidinoid": "ch274_lidinoid",
    "neovius-surface": "ch275_neovius_surface",
    "schwarz-p": "ch317_schwarz_p_surface",
    "schwarz-d": "ch308_schwarz_d_surface",
    "h-exact": "ch315_schwarz_h_surfaces",
    "clp-exact": "ch297_schwarz_clp_surfaces",
    "clp-handle-exact": "ch264_clp_with_handle",
    "iwp-surface": "ch284_schoen_i_wp",
    "frd-surface": "ch281_schoen_f_rd",
    "schoen-i6": "ch285_schoen_i6",
    "schoen-frd-r": "ch290_schoens_unnamed_surface_12_f_rd_r",
    "schoen-rii": "ch286_schoen_r2",
    "weber-h2r": "ch282_schoen_h_r",
    "weber-trr": "ch288_schoen_t_r",
    "weber-rpd": "ch279_rpd_deformation",
    "triply-periodic-costa": "ch259_box_symmetry_type_4",
    # Bjorling surfaces (planar seed curves; name and construction match)
    "bjorling-archimedean-spiral": "ch003_archimedean_spiral",
    "bjorling-logarithmic-spiral": "ch011_logarithmic_spiral",
}


def ids():
    out = {}
    for slug, stem in VMM_IDS.items():
        out.setdefault(slug, {})["vmm"] = stem
    for slug, stem in MS_IDS.items():
        out.setdefault(slug, {})["msblog"] = stem
    return out


# ---------------------------------------------------------------------------
# records() -- named surfaces with no record at all.
# (slug, name, family, mode, blocked_by, resume, sources, extra)
# ---------------------------------------------------------------------------

def _vmm_minimal_records():
    """Named VMM minimal-surface exhibits absent from the database."""
    K = [KARCHER_TOKYO]
    return {
        "wavy-enneper": {
            "name": "Wavy Enneper", "family": "minimal",
            "mode": "weierstrass",
            "blocked_by": VMM_BLOCKED, "resume": VMM_RESUME,
            "sources": [_vmm_src("ch010_wavy_enneper",
                                 "Wavy Enneper Surface")] + K,
            "extra": _deep(_MIN, {"ids": {"vmm": "ch010_wavy_enneper"},
                "definition": {"note":
                    "Enneper surface modified so waves develop around its "
                    "boundary and can travel around it: a finite total "
                    "curvature minimal immersion of the once-punctured "
                    "sphere with k-fold symmetry (the exhibit shows the "
                    "3-fold case), from Karcher's Tokyo notes."}}),
        },
        "four-noid-two-symmetry-planes": {
            "name": "4-Noid with Two Symmetry Planes", "family": "minimal",
            "mode": "weierstrass",
            "blocked_by": VMM_BLOCKED, "resume": VMM_RESUME,
            "sources": [
                _vmm_src("ch013_symmetric_4noid", "Symmetric 4-noid"),
                _vmm_src("ch014_skew_4noid", "Skew 4-noid"),
                _ms_src("ch173_4_noids_with_two_symmetry_planes",
                        "4-Noids with Two Symmetry Planes"),
                KARCHER_TOKYO + " (pages 30ff)."],
            "extra": _deep(_MIN, {
                "ids": {"vmm": "ch013_symmetric_4noid",
                        "msblog": "ch173_4_noids_with_two_symmetry_planes"},
                "definition": {"note":
                    "Karcher's 2-parameter family of 4-noids (4-punctured "
                    "spheres, four catenoidal ends) with two orthogonal "
                    "symmetry planes; the parameters control the position "
                    "of the ends and their relative growth rates, and the "
                    "Jorge-Meeks 4-noid is the intersection of the two "
                    "1-parameter sections that 3DXM exhibits separately as "
                    "'Symmetric 4-noid' (opposite end pairs of different "
                    "size) and 'Skew 4-noid' (varying angle between end "
                    "pairs, morphing toward two catenoids joined by a "
                    "handle -- the family that convinced Hoffman that "
                    "designed handles were promising)."}}),
        },
        "karcher-je-saddle-tower": {
            "name": "Karcher JE Saddle Tower", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": VMM_BLOCKED, "resume": VMM_RESUME,
            "sources": [_vmm_src("ch026_karcher_je_st",
                                 "Karcher JE Saddle Tower"), KARCHER_1988],
            "extra": _deep(_LAYER, {"ids": {"vmm": "ch026_karcher_je_st"},
                "definition": {"note":
                    "Doubly periodic embedded minimal surfaces parametrized "
                    "by rectangular tori, from Karcher's 1988 Scherk-example "
                    "constructions; the shape can be viewed as a doubly "
                    "periodic version of Riemann's minimal surface."}}),
        },
        "karcher-jd-saddle-tower": {
            "name": "Karcher JD Saddle Tower", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": VMM_BLOCKED, "resume": VMM_RESUME,
            "sources": [_vmm_src("ch027_karcher_jd_st",
                                 "Karcher JD Saddle tower"), KARCHER_1988],
            "extra": _deep(_LAYER, {"ids": {"vmm": "ch027_karcher_jd_st"},
                "definition": {"note":
                    "Doubly periodic embedded conjugate pair of minimal "
                    "surfaces suggested by the Scherk saddle towers; unlike "
                    "the JE family the towers are separated by a planar "
                    "symmetry line, and by a rare coincidence the conjugate "
                    "family consists of the same surfaces with the "
                    "parameter running backwards.  Morphing parameter: edge "
                    "length ratio of the rectangular torus domain."}}),
        },
        "catenoid-field": {
            "name": "Catenoid Field", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": VMM_BLOCKED, "resume": VMM_RESUME,
            "sources": [_vmm_src("ch025_catenoid_field",
                                 "Catenoid Field")] + K,
            "extra": _deep(_LAYER, {"ids": {"vmm": "ch025_catenoid_field"},
                "definition": {"note":
                    "Doubly periodic field of half-catenoids growing "
                    "alternately up and down: parametrized by a twice-"
                    "punctured rectangular torus with Gauss map g(z) = "
                    "c*Je(z), Je a Jacobi-type elliptic function with two "
                    "simple poles and zeros.  Weber's repository page "
                    "'Doubly Periodic Catenoids' (mirror ch025) describes "
                    "a closely related doubly periodic catenoid "
                    "arrangement; whether it is the same family has NOT "
                    "been established here, so that page is deliberately "
                    "not cross-referenced."}}),
        },
        "kusner-sphere-2n-planar-ends": {
            "name": "Kusner Sphere (2n planar ends)", "family": "minimal",
            "mode": "weierstrass",
            "blocked_by": VMM_BLOCKED, "resume": VMM_RESUME,
            "sources": [
                _vmm_src("ch020_kusner_ds", "Kusner Surface"),
                _ms_src("ch065_kusners_spheres_with_planar_ends",
                        "Kusner's Spheres with Planar Ends"),
                "R. Kusner, 'Conformal geometry and complete minimal "
                "surfaces', Bull. Amer. Math. Soc. 17 (1987) 291-295."],
            "extra": _deep(_MIN, {"ids": {"vmm": "ch020_kusner_ds"},
                "embedding": {"quality": "immersed"},
                "definition": {"note":
                    "Kusner's immersed minimal spheres with an even number "
                    "2n of planar ends (the VMM exhibit shows n = 4: a "
                    "sphere with 8 punctures, 4 per hemisphere, and 4 "
                    "straight lines on the surface).  For n odd the "
                    "immersion factors through the antipodal map and gives "
                    "the projective-plane family, which has its own record "
                    "(kusner-projective-plane-p-planar-ends)."}}),
        },
        "fujimori-weber": {
            "name": "Fujimori-Weber Surface", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": VMM_BLOCKED + (
                "  The exhibit page shows the basic family in images only, "
                "with no prose; the identification below rests on the "
                "exhibit's own attribution."),
            "resume": VMM_RESUME,
            "sources": [
                _vmm_src("ch038_fujimori_weber", "Fujimori-Weber"),
                "S. Fujimori and M. Weber, 'Triply periodic minimal "
                "surfaces bounded by vertical symmetry planes', "
                "Manuscripta Math. 129 (2009) 29-53."],
            "extra": _deep(_SPACE, {"ids": {"vmm": "ch038_fujimori_weber"},
                "definition": {"note":
                    "3DXM's exhibit of the triply periodic minimal "
                    "surfaces constructed by Shoichi Fujimori and Matthias "
                    "Weber, shown as a morphing basic family."}}),
        },
    }


def _vmm_tpms_records():
    """Schoen families exhibited by 3DXM but absent from the database
    (the database's Schoen set covers R-II/R-III/I-6/8/9/GW/F-RD(r)/
    I-WP(r)/Hybrid-1/Manta/Batwing, plus H''-R and T'-R' as tail
    records -- none of these)."""
    return {
        "schoen-s-s": {
            "name": "Schoen S'-S'' Surface", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": VMM_BLOCKED, "resume": MS_RESUME,
            "sources": [_vmm_src("ch035_schoen_ss", "A Schoen S-S Family"),
                        _ms_src("ch287_schoen_s_s", "Schoen S'-S''"),
                        SCHOEN_NASA],
            "extra": _deep(_SPACE, {
                "ids": {"vmm": "ch035_schoen_ss",
                        "msblog": "ch287_schoen_s_s"},
                "definition": {"note":
                    "Genus-4 (per cell) member of Schoen's 1970 NASA "
                    "report, in a 1-parameter family; one nodal limit is "
                    "two planes joined per cell by one large and four "
                    "small catenoidal necks, the other limit is 8-ended "
                    "singly periodic Scherk surfaces.  The name derives "
                    "from the two skeletal graphs of the complement: "
                    "square grids at different sizes."}}),
        },
        "schoen-h-t": {
            "name": "Schoen H'-T Surface", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": VMM_BLOCKED, "resume": MS_RESUME,
            "sources": [_vmm_src("ch036_schoen_ht_hex",
                                 "A Schoen HT Hexagonal Family"),
                        _ms_src("ch283_schoen_h_t", "Schoen H'-T"),
                        SCHOEN_NASA],
            "extra": _deep(_SPACE, {
                "ids": {"vmm": "ch036_schoen_ht_hex",
                        "msblog": "ch283_schoen_h_t"},
                "definition": {"note":
                    "Schoen's hexagonal H'-T family from the 1970 NASA "
                    "report, exhibited by 3DXM as a morphing hexagonal "
                    "family with its fundamental region marked."}}),
        },
        "schoen-tw": {
            "name": "Schoen TW Family Surface", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": VMM_BLOCKED + (
                "  The exhibit gives images only, and which entry of "
                "Schoen's NASA catalogue 'TW' denotes has NOT been "
                "identified here -- resolving that naming is part of the "
                "remaining work."),
            "resume": VMM_RESUME,
            "sources": [_vmm_src("ch037_schoen_tw", "A Schoen TW Family"),
                        SCHOEN_NASA],
            "extra": _deep(_SPACE, {"ids": {"vmm": "ch037_schoen_tw"},
                "definition": {"note":
                    "A hexagonal triply periodic minimal surface family "
                    "that 3DXM exhibits as 'A Schoen TW Family', "
                    "attributed by the museum to Alan Schoen."}}),
        },
        "schoen-c-h": {
            "name": "Schoen C(H) Surface", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [_ms_src("ch280_schoen_ch", "Schoen C(H)"),
                        SCHOEN_NASA],
            "extra": _deep(_SPACE, {"ids": {"msblog": "ch280_schoen_ch"},
                "definition": {"note":
                    "Schoen's Complementary H surface: same pattern of "
                    "straight lines and reflectional symmetries as the "
                    "Schwarz H surface but with extra handles raising the "
                    "genus per cell to 7; a 1-parameter family limiting in "
                    "3-fold dihedral Callahan-Hoffman-Meeks surfaces one "
                    "way and a nodal configuration the other."}}),
        },
    }


def _vmm_other_records():
    """Non-minimal VMM exhibits absent from the database."""
    return {
        "moebius-strip": {
            "name": "Moebius Strip", "family": "topological",
            "mode": "parametric",
            "blocked_by":
                "Not built as its own generator: the database holds the "
                "minimal Meeks Moebius strip, the Sudanese band in S3, a "
                "Bjorling twisted band and a solid twisted strip, but the "
                "plain ruled Moebius band -- the canonical one-sided "
                "surface -- has no record or operator of its own.",
            "resume":
                "One ruled-band chart (the standard half-twist "
                "parametrization on the VMM page) in "
                "math_art/topological_surface_generator.py alongside its "
                "existing non-orientable surfaces.",
            "sources": [_vmm_src("ch043_moebius_strip", "Moebius Strip")],
            "extra": {
                "ids": {"vmm": "ch043_moebius_strip"},
                "topology": {"orientable": False, "one_sided": True,
                             "compact": True, "boundary_components": 1},
                "tradition": ["classical"],
                "definition": {"note":
                    "The classical one-sided band (Moebius and Listing, "
                    "1858): a strip closed up after a half twist, here as "
                    "the standard ruled parametrization the VMM exhibit "
                    "prints.  The twist sign gives the two chiralities; "
                    "per house convention that is one record with a "
                    "handedness, not two."}},
        },
        "clifford-torus": {
            "name": "Clifford Torus", "family": "topological",
            "mode": "parametric",
            "blocked_by":
                "Not built: the Hopf-fibration generator renders Hopf tori "
                "and the Bianchi-Pinkall record covers the flat-torus "
                "family in S3, but the Clifford torus itself -- the "
                "square flat torus S1(1/sqrt(2)) x S1(1/sqrt(2)) in S3 -- "
                "has no record.",
            "resume":
                "Stereographic projection of the explicit S3 chart; "
                "math_art/hopf_fibration_generator.py already carries the "
                "needed projection machinery.",
            "sources": [_vmm_src("ch048_clifford_torus", "Clifford Torus")],
            "extra": {
                "ids": {"vmm": "ch048_clifford_torus"},
                "topology": {"genus": 1, "orientable": True, "compact": True,
                             "boundary_components": 0,
                             "euler_characteristic": 0},
                "tradition": ["classical"],
                "definition": {"note":
                    "The torus S1(1/sqrt(2)) x S1(1/sqrt(2)) in R4, lying "
                    "in the unit 3-sphere; intrinsically flat there, and "
                    "stereographic projection carries it to a torus of "
                    "revolution in R3 (the VMM exhibit morphs the "
                    "projection centre, including centres on the torus "
                    "where the image becomes a plane with a handle).  The "
                    "flatness lives in S3; the R3 image is not flat, so "
                    "no curvature condition is claimed for the projected "
                    "surface."}},
        },
        "norm-one-family": {
            "name": "Norm One Family", "family": "misc",
            "mode": "implicit",
            "blocked_by":
                "Not built: the supershape record covers the Gielis "
                "superformula, which is a different family; the plain "
                "p-norm unit spheres have no record or operator.",
            "resume":
                "A one-parameter implicit |x|^p + |y|^p + |z|^p = 1 in "
                "math_art/surfaces/ (marching-cubes route used by the "
                "algebraic block); p is real, so it is not an algebraic "
                "family record.",
            "sources": [_vmm_src("ch080_norm_one_family", "Norm One Family")],
            "extra": {
                "ids": {"vmm": "ch080_norm_one_family"},
                "topology": {"compact": True, "orientable": True},
                "tradition": ["classical"],
                "definition": {
                    "note":
                        "The unit spheres of the p-norms on R3 (the "
                        "equal-exponent Lame / superellipsoid family): "
                        "|x|^p + |y|^p + |z|^p = 1, octahedron at p = 1, "
                        "sphere at p = 2, cube as p -> infinity, convex "
                        "for p >= 1.",
                    "parameters": [
                        {"name": "p", "domain": "(0, inf)", "default": 4,
                         "note": "norm exponent"}]}},
        },
        "deco-tetrahedron": {
            "name": "Deco-Tetrahedron", "family": "algebraic",
            "mode": "implicit",
            "blocked_by":
                "Not built.  The level function (squared distance from "
                "the four face circumcircles of a regular tetrahedron) is "
                "deliberately NOT transcribed: the shipped deco-cube's "
                "circle-distance construction is guarded by a numerical "
                "oracle, and this variant has no implementation to check "
                "a transcription against.",
            "resume":
                "Implement beside DECO_CUBE in "
                "math_art/surfaces/algebraic.py -- same "
                "distance-from-circles machinery with the four "
                "tetrahedral face circumcircles -- and let the build's "
                "oracle verify any stored form.",
            "sources": [_vmm_src("ch091_deco_tetahedron", "Deco-Tetahedron")],
            "extra": {
                "ids": {"vmm": "ch091_deco_tetahedron"},
                "topology": {"compact": True},
                "tradition": ["gallery"],
                "definition": {"note":
                    "Level set of the squared-distance function from the "
                    "four circles through the vertex triples of a regular "
                    "tetrahedron's faces -- the tetrahedral sibling of "
                    "the shipped Deco-Cube (distance from the cube's six "
                    "face-inscribed circles)."}},
        },
    }


def _ms_records():
    """Named surfaces from Weber's repository/posts with no record."""
    return {
        "breiner-kleene-surface": {
            "name": "Breiner-Kleene Surface", "family": "minimal",
            "mode": "parametric",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch005_breiner_kleene_surface",
                        "Breiner-Kleene Surface"),
                "C. Breiner and S. Kleene, 'Logarithmically spiraling "
                "helicoids', arXiv:1404.6996 (2014)."],
            "extra": _deep(_MIN, {
                "ids": {"msblog": "ch005_breiner_kleene_surface"},
                "definition": {"note":
                    "Embedded minimal surfaces invariant under a screw "
                    "motion composed with a homothety: bent helicoids "
                    "following a logarithmically spiraling helix, embedded "
                    "in an invariant tube around it (Breiner-Kleene "
                    "2014)."}}),
        },
        "costa-hoffman-karcher-meeks-tori": {
            "name": "Costa-Hoffman-Karcher-Meeks Tori", "family": "minimal",
            "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch241_hoffman_karcher_tori",
                        "Costa-Hoffman-Karcher-Meeks Tori"),
                "D. Hoffman and W. H. Meeks III, 'Properties of properly "
                "embedded minimal surfaces of finite topology', Bull. "
                "Amer. Math. Soc. 17 (1987) 296-300.",
                "C. J. Costa, 'Classification of complete minimal "
                "surfaces in R3 with total curvature 12pi', Invent. Math. "
                "105 (1991) 273-303.",
                "D. Hoffman and H. Karcher, 'Complete embedded minimal "
                "surfaces of finite total curvature', Geometry V, "
                "Encyclopaedia Math. Sci. 90, Springer (1997) 5-93."],
            "extra": _deep(_MIN, {
                "ids": {"msblog": "ch241_hoffman_karcher_tori"},
                "definition": {"note":
                    "The 1-parameter family of embedded minimal tori "
                    "obtained by deforming the Costa surface's planar "
                    "middle end into a catenoidal end; by Costa's "
                    "classification these are the ONLY embedded 3-ended "
                    "minimal tori of finite total curvature.  Distinct "
                    "from the genus-varying Costa-Hoffman-Meeks record, "
                    "which fixes the planar middle end and raises the "
                    "genus."}}),
        },
        "weber-wolf-genus-3-5-ends": {
            "name": "Weber-Wolf Surface (genus 3, 5 ends)",
            "family": "minimal", "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch415_weber_wolf_surface_of_genus_3_with_5_ends",
                        "Weber-Wolf Surface of genus 3 with 5 ends"),
                _ms_src("ch416_weber_wolf_surface_of_genus_4_with_5_ends",
                        "Weber-Wolf Surface of Genus 4 with 5 Ends")],
            "extra": _deep(_MIN, {
                "ids": {"msblog":
                        "ch415_weber_wolf_surface_of_genus_3_with_5_ends"},
                "definition": {"note":
                    "Weber and Wolf's surface with two catenoidal and "
                    "three planar ends realizing the borderline case g = 3 "
                    "of the Hoffman-Meeks conjecture (at most g + 2 ends "
                    "for an embedded finite-total-curvature surface of "
                    "genus g); the planar levels connect through Costa "
                    "saddles.  The page states similar surfaces exist for "
                    "all odd genera; a companion page shows a genus-4 "
                    "version."}}),
        },
        "kapouleas-surfaces": {
            "name": "Kapouleas Surface", "family": "minimal",
            "mode": "variational",
            "blocked_by": MS_BLOCKED + (
                "  The page notes all period problems here are solved "
                "numerically, with no simple existence proof for the "
                "rendered examples."),
            "resume": MS_RESUME,
            "sources": [
                _ms_src("ch387_kapouleas_surfaces", "Kapouleas Surfaces"),
                "N. Kapouleas, 'Complete embedded minimal surfaces of "
                "finite total curvature', J. Differential Geom. 47 (1997) "
                "95-169."],
            "extra": _deep(_MIN, {
                "ids": {"msblog": "ch387_kapouleas_surfaces"},
                "definition": {"note":
                    "Kapouleas' finite-total-curvature embedded minimal "
                    "surfaces with arbitrarily many ends, built by "
                    "desingularizing the circles of intersection of "
                    "coaxial catenoids and planes with bent singly "
                    "periodic Scherk surfaces."}}),
        },
        "hackman-surfaces": {
            "name": "Hackman Surface (toroidal 1-noid)", "family":
            "minimal-periodic", "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch382_hackman_surfaces", "Hackman Surfaces"),
                "M. Hackman, PhD thesis (as credited by the mirrored "
                "page for the construction and the classification "
                "remarks)."],
            "extra": _deep(_ROD, {
                "ids": {"msblog": "ch382_hackman_surfaces"},
                "definition": {"note":
                    "Toroidal 1-noids: quotients of screw-motion invariant "
                    "singly periodic minimal surfaces with a single "
                    "catenoidal end in the quotient (the maximum principle "
                    "forbids 1-noids in R3 proper).  Hackman proved one "
                    "exists on every conformal type of torus; the "
                    "alternating fence of half-catenoids is the simplest "
                    "member of the wider picture."}}),
        },
        "lopez-martin-slab-surface": {
            "name": "Lopez-Martin Slab Surface", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch399_lopez_martin_slab_surface",
                        "Lopez-Martin Slab Surface"),
                "F. J. Lopez and F. Martin, 'Minimal surfaces in a wedge "
                "of a slab' (the paper the mirrored page presents)."],
            "extra": _deep(_ROD, {
                "ids": {"msblog": "ch399_lopez_martin_slab_surface"},
                "embedding": {"quality": "immersed"},
                "topology": {"orientable": False},
                "definition": {"note":
                    "Translation invariant minimal surface with planar "
                    "ends, neither embedded (self-intersection along the "
                    "z-axis only) nor orientable; solves a Plateau problem "
                    "in a wedge of a slab, and arises from the translation "
                    "invariant helicoid-with-handle family when its "
                    "vertical period condition is left unsolved."}}),
        },
        "scherks-fourth-surface": {
            "name": "Scherk's Fourth Surface", "family": "minimal-periodic",
            "mode": "implicit",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch341_scherks_fourth_surface",
                        "Scherk's Fourth Surface"),
                "H. F. Scherk, 'Bemerkungen ueber die kleinste Flaeche "
                "innerhalb gegebener Grenzen', J. Reine Angew. Math. 13 "
                "(1835) 185-208 (equation 20)."],
            "extra": _deep(_ROD, {
                "ids": {"msblog": "ch341_scherks_fourth_surface"},
                "embedding": {"quality": "singular"},
                "definition": {"note":
                    "The least-cited of Scherk's 1835 surfaces, given "
                    "implicitly as his equation 20: singly periodic with "
                    "two annular and two helicoidal ends, and singular at "
                    "the two points where the horizontal symmetry curve "
                    "meets the straight line shared by the helicoidal "
                    "ends.  Weber recovers its Enneper-Weierstrass "
                    "representation via the Schwarz-Bjorling formula on "
                    "the x = pi level symmetry curve."}}),
        },
        "half-twisted-scherk": {
            "name": "Half-Twisted Scherk Surface",
            "family": "minimal-periodic", "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch152_the_half_twisted_scherk_surface",
                        "The Half-Twisted Scherk Surface")],
            "extra": _deep(_ROD, {
                "ids": {"msblog": "ch152_the_half_twisted_scherk_surface"},
                "definition": {"note":
                    "Weber's singly periodic surface with two annular and "
                    "two helicoidal ends -- the simpler surface with the "
                    "same end types as Scherk's fourth surface, which the "
                    "Scherk IV post says he found accidentally."}}),
        },
        "toroidal-karcher-scherk": {
            "name": "Toroidal Karcher-Scherk Surface",
            "family": "minimal-periodic", "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch157_toroidal_karcher_scherk_surfaces",
                        "Toroidal Karcher-Scherk Surfaces"), KARCHER_TOKYO],
            "extra": _deep(_ROD, {
                "ids": {"msblog": "ch157_toroidal_karcher_scherk_surfaces"},
                "definition": {"note":
                    "Vertical handles added to Karcher-Scherk saddle "
                    "towers with at least 6 ends, first mentioned in "
                    "Karcher's Tokyo notes; the mirrored page shows "
                    "5-ended members with triangular-prism symmetry in a "
                    "1-parameter family of end angles.  Distinct from the "
                    "genus-1 Costa-Scherk tower record, which needs a "
                    "half-period phase shift."}}),
        },
        "plane-with-catenoids": {
            "name": "Plane with Catenoids (doubly periodic)",
            "family": "minimal-periodic", "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch039_plane_with_catenoids", "Plane with Catenoids")],
            "extra": _deep(_LAYER, {
                "ids": {"msblog": "ch039_plane_with_catenoids"},
                "definition": {"note":
                    "The simplest doubly periodic minimal surfaces with "
                    "only catenoidal ends in the quotient: ends arranged "
                    "in a square lattice with limiting normals "
                    "perpendicular to the periodicity plane, growth rates "
                    "adjustable by the Lopez-Ros factor, and no period "
                    "problem to solve.  The repository's singly periodic "
                    "'translation invariant plane with catenoidal ends' "
                    "(mirror ch172) is the 1-periodic sibling, also "
                    "unrecorded."}}),
        },
        "lubeck-batista-surface": {
            "name": "Lubeck-Batista Surface", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch038_lubeck_batista_surface",
                        "Lubeck-Batista Surface"),
                "K. R. M. Lubeck and V. Ramos Batista, 'The doubly "
                "periodic Scherk-Costa surfaces', Journal of Mathematics "
                "Research 6 (2014) 77-90."],
            "extra": _deep(_LAYER, {
                "ids": {"msblog": "ch038_lubeck_batista_surface"},
                "definition": {"note":
                    "Doubly periodic 1-parameter family (2011) with genus "
                    "3 and four annular ends in the quotient: layers "
                    "joined by Costa saddles keeping the Costa surface's "
                    "straight lines, limiting in the Callahan-Hoffman-"
                    "Meeks surface -- a doubly periodic version of it."}}),
        },
        "simoes-batista-surface": {
            "name": "Simoes-Batista Surface", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch408_the_simoes_batista_surface",
                        "The Simoes-Batista Surface"),
                "P. Simoes and V. Ramos Batista, arXiv:0806.3088 (2008), "
                "the paper the mirrored page links."],
            "extra": _deep(_SPACE, {
                "ids": {"msblog": "ch408_the_simoes_batista_surface"},
                "definition": {"note":
                    "Genus-7 triply periodic 1-parameter family with "
                    "rectangular period lattice, obtained by adding a "
                    "handle to Batista's triply periodic Costa surface; "
                    "limits are the singly periodic CHM(1,3) surface and "
                    "the doubly periodic Scherk surface."}}),
        },
        "triply-periodic-horgan": {
            "name": "Triply Periodic Horgan Surface",
            "family": "minimal-periodic", "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch263_catenoid_scherk",
                        "Catenoid-Scherk Limits -- aka Triply Periodic "
                        "Horgan Surface")],
            "extra": _deep(_SPACE, {
                "ids": {"msblog": "ch263_catenoid_scherk"},
                "definition": {"note":
                    "Genus-5 (per cell) triply periodic 1-parameter "
                    "family with vertical symmetry planes over a square "
                    "grid and diagonal horizontal lines, limiting in "
                    "noded planes and in doubly periodic Karcher-Scherk "
                    "surfaces.  Its nodal-limit neck configuration is "
                    "exactly that of the (non-existent) finite Horgan "
                    "surface -- yet this periodic surface EXISTS, with a "
                    "1-dimensional period problem solved by an extremal-"
                    "length argument Weber presents as a picture proof."}}),
        },
        "wei-triply-periodic-genus-4": {
            "name": "Wei Triply Periodic Surface (genus 4)",
            "family": "minimal-periodic", "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch335_weis_triply_periodic_surface_of_genus_4",
                        "Wei's Triply Periodic Surface of Genus 4"),
                "F. Wei, 'Some existence and uniqueness theorems for "
                "doubly periodic minimal surfaces', Invent. Math. 109 "
                "(1992) 113-136."],
            "extra": _deep(_SPACE, {
                "ids": {"msblog":
                        "ch335_weis_triply_periodic_surface_of_genus_4"},
                "definition": {"note":
                    "Wei's 2-parameter family of genus-4 triply periodic "
                    "minimal surfaces from his 1992 doubly periodic "
                    "paper; his doubly periodic genus-2 surfaces (which "
                    "have their own record) arise as limits, as do "
                    "vertical planes over a rhombic tiling desingularized "
                    "by singly periodic Scherk surfaces."}}),
        },
        "stessmann-surface": {
            "name": "Stessmann Surface", "family": "minimal-periodic",
            "mode": "weierstrass",
            "blocked_by": MS_BLOCKED, "resume": MS_RESUME,
            "sources": [
                _ms_src("ch330_stesmanns_surface", "Stessmann's Surface"),
                "B. Stessmann, 'Periodische Minimalflaechen', Math. Z. 38 "
                "(1934) 417-442."],
            "extra": _deep(_SPACE, {
                "ids": {"msblog": "ch330_stesmanns_surface"},
                "embedding": {"quality": "self-intersecting"},
                "definition": {"note":
                    "Stessmann's 1934 Plateau solution for one of the six "
                    "Schoenflies quadrilaterals whose edge rotations "
                    "generate a discrete group (Schwarz had solved the "
                    "three most symmetric cases); extending it gives a "
                    "triply periodic, non-embedded surface that Alan "
                    "Schoen observed is the conjugate of his I-WP "
                    "surface, predating it by 40 years."}}),
        },
    }


def _nonexistent_records():
    """Weber's 'Non-Existent Surfaces' section: constructions tried and
    ruled out.  These records exist so the search is not repeated; the
    wording distinguishes PROVED non-existence from numerical evidence.
    """
    return {
        "horgan-surface": {
            "name": "Horgan Surface", "family": "minimal",
            "mode": "weierstrass",
            "blocked_by":
                "PROVED NOT TO EXIST -- not merely unbuilt.  Weber proved "
                "that the proposed finite-total-curvature minimal surface "
                "cannot close its periods, so no such surface exists; the "
                "record is a negative result, kept so the construction is "
                "not attempted again.",
            "resume":
                "Nothing to build.  If the family is ever revisited, note "
                "that the same neck configuration DOES exist as a triply "
                "periodic surface (see the triply-periodic-horgan "
                "record).",
            "sources": [
                _ms_src("ch062_the_horgan_surface", "The Horgan Surface"),
                _ms_src("ch061_non_existent_surfaces",
                        "Non-Existent Surfaces (the section index that "
                        "lists it as a failed construction)"),
                "M. Weber, 'On the Horgan minimal non-surface', Calc. "
                "Var. Partial Differential Equations 7 (1998) 373-379."],
            "extra": _deep(_MIN, {
                "ids": {"msblog": "ch062_the_horgan_surface"},
                "definition": {"note":
                    "A plausible-looking finite total curvature minimal "
                    "surface (two planes joined by catenoidal necks in "
                    "the pattern later realized triply periodically) "
                    "whose period problem admits no solution."}}),
        },
        "catenoid-with-handle": {
            "name": "Catenoid with Handle", "family": "minimal",
            "mode": "weierstrass",
            "blocked_by":
                "PROVED NOT TO EXIST -- not merely unbuilt.  A complete, "
                "properly immersed minimal surface with two catenoidal "
                "ends is the catenoid (R. Schoen), so the catenoid with "
                "an inward-grown handle necessarily leaves a gap; Weber "
                "files it under Non-Existent Surfaces.",
            "resume":
                "Nothing to build as a finite surface.  Growing the "
                "handle outward instead yields the periodic fence of "
                "catenoids, which already has a record "
                "(fence-of-catenoids-karcher).",
            "sources": [
                _ms_src("ch170_translation_invariant_catenoid",
                        "Translation Invariant Catenoid (which documents "
                        "the failed inward-handle construction)"),
                _ms_src("ch061_non_existent_surfaces",
                        "Non-Existent Surfaces (section index)"),
                "R. Schoen, 'Uniqueness, symmetry, and embeddedness of "
                "minimal surfaces', J. Differential Geom. 18 (1983) "
                "791-809."],
            "extra": _deep(_MIN, {
                "ids": {"msblog": "ch170_translation_invariant_catenoid"},
                "definition": {"note":
                    "The attempted genus-1 catenoid: two catenoidal ends "
                    "plus one handle.  Ruled out by Schoen's "
                    "characterization of the catenoid."}}),
        },
        "dihedralized-wohlgemuth-with-handle": {
            "name": "Dihedralized Wohlgemuth with Handle",
            "family": "minimal", "mode": "weierstrass",
            "blocked_by":
                "BELIEVED NOT TO EXIST on numerical evidence -- a weaker "
                "statement than a proof, and a stronger one than "
                "'unbuilt'.  Weber reports that raising the dihedral "
                "symmetry of Wohlgemuth's second surface (genus 3) to "
                "k = 3 fails: at k = 2.999 the numerically solved periods "
                "do not appear to close, and for larger k the period "
                "problem could not be solved at all.  Weber files it "
                "under Non-Existent Surfaces.",
            "resume":
                "A proof either way would settle it; the mirrored page "
                "poses the underlying free-boundary deformation question "
                "explicitly.  The existing k = 2 surface has its own "
                "record (wohlgemuth-second-surface-genus-3).",
            "sources": [
                _ms_src("ch418_wohlgemuths_second_surface",
                        "Wohlgemuth's Second Surface (which documents the "
                        "failed dihedralization)"),
                _ms_src("ch061_non_existent_surfaces",
                        "Non-Existent Surfaces (section index)"),
                "M. Wohlgemuth, 'Minimal surfaces of higher genus with "
                "finite total curvature', Arch. Rational Mech. Anal. 137 "
                "(1997) 1-25 (the existing k = 2 surface)."],
            "extra": _deep(_MIN, {
                "ids": {"msblog": "ch418_wohlgemuths_second_surface"},
                "definition": {"note":
                    "The attempted k >= 3 dihedral symmetrization of "
                    "Wohlgemuth's genus-3 handle surface.  Its failure "
                    "questions the folklore that a surface existing for "
                    "some dihedral order k exists for all larger k."}}),
        },
    }


def records():
    out = {}
    for table in (_vmm_minimal_records(), _vmm_tpms_records(),
                  _vmm_other_records(), _ms_records(),
                  _nonexistent_records()):
        for slug, spec in table.items():
            assert slug not in out, "duplicate vmm-record slug %r" % slug
            out[slug] = spec
    return out


def invariants_for(slug):
    return {}


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _resolve(key, stem):
    """True if a chapter stem resolves to a page in the local mirror the
    key names -- vmm ids must resolve in the VMM mirror and msblog ids in
    the minsurf mirror, not merely somewhere."""
    if key == "vmm":
        dirs = (VMM_SURFACE_DIR,)
    else:
        dirs = MS_BOOK_DIRS
    return any(os.path.isfile(os.path.join(d, stem + ".md")) for d in dirs)


def _selftest():
    recs = records()
    idtab = ids()

    # -- records: shape ----------------------------------------------------
    for slug, spec in recs.items():
        assert slug == slug.lower() and " " not in slug and "_" not in slug, \
            slug
        assert spec.get("name"), slug
        assert spec.get("family") in (
            "minimal", "minimal-periodic", "topological", "algebraic",
            "misc"), (slug, spec.get("family"))
        assert spec.get("mode") in (
            "implicit", "parametric", "weierstrass", "nodal", "variational",
            "swept", "derived", "discrete"), (slug, spec.get("mode"))
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

    # -- non-existence records say so explicitly ---------------------------
    for slug in ("horgan-surface", "catenoid-with-handle"):
        assert "PROVED NOT TO EXIST" in recs[slug]["blocked_by"], slug
    assert ("BELIEVED NOT TO EXIST"
            in recs["dihedralized-wohlgemuth-with-handle"]["blocked_by"])

    # -- every id resolves against the local mirror ------------------------
    mirror_up = (os.path.isdir(VMM_SURFACE_DIR)
                 and all(os.path.isdir(d) for d in MS_BOOK_DIRS))
    stems = []
    for slug, entry in idtab.items():
        for key, stem in entry.items():
            assert key in ("vmm", "msblog"), (slug, key)
            stems.append((slug, key, stem))
    for slug, spec in recs.items():
        for key, stem in ((spec.get("extra") or {}).get("ids") or {}).items():
            assert key in ("vmm", "msblog"), (slug, key)
            stems.append((slug, key, stem))
    if mirror_up:
        for slug, key, stem in stems:
            assert _resolve(key, stem), \
                "%s: %s id %r does not resolve in the mirror" % (
                    slug, key, stem)
        resolution = "all %d ids resolve" % len(stems)
    else:
        resolution = ("mirror not mounted; %d ids NOT resolution-checked"
                      % len(stems))

    n_vmm = len(VMM_IDS)
    n_ms = len(MS_IDS)
    print("RESULT: OK  (surfdb.vmm, %d records, %d id sets: %d vmm + %d "
          "msblog on existing records; %s)"
          % (len(recs), len(idtab), n_vmm, n_ms, resolution))
