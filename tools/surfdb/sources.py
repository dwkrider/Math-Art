# Resolving cross-reference IDs against the local mirrors.
#
# `S:/data/math_art/references/websites/` holds converted mirrors of the
# sources this database is audited against -- Ferreol's encyclopedia (181
# surface chapters), the 3DXM Virtual Math Museum, and Matthias Weber's
# minimal-surface blog.  When a record carries `ids.mathcurve`, it should
# RESOLVE: a reader with the mirror can open the page.
#
# Matching is deliberately conservative.  Ferreol's chapter stems are
# French -- `ch1291_tore_2` is the torus, `ch1230_cylindrederevolution_2`
# the cylinder of revolution -- so English name matching finds only part
# of the set, and a fuzzy match would confidently produce wrong IDs.  A
# wrong cross-reference is worse than an absent one: it sends a reader to
# the wrong page and looks authoritative doing it.
#
# So: an explicit alias table for the ones that matter, plus a strict
# containment match for the rest, and nothing else.

import os
import re

MIRROR = os.path.join("S:", os.sep, "data", "math_art", "references",
                      "websites")
SURFACES = os.path.join(MIRROR, "mathcurve", "book", "surfaces")

# slug -> Ferreol chapter stem, where the French title does not contain
# the English name.  Every entry here was read off the mirror's own
# directory listing, not guessed.
MATHCURVE_ALIAS = {
    "torus": "ch1291_tore_2",
    "circular-cylinder": "ch1230_cylindrederevolution_2",
    "elliptic-cylinder": "ch1248_cylindreelliptique_2",
    "hyperbolic-cylinder": "ch1249_cylindrehyperbolic_2",
    "parabolic-cylinder": "ch1250_cylindreparabolic_2",
    "elliptic-cone": "ch1235_coneelliptique_2",
    "sphere": "ch1257_sphere_2",
    "plane": "ch1320_plan_2",
    "ellipsoid": "ch1207_ellipsoid_2",
    "elliptic-paraboloid": "ch1263_paraboloidelliptic_2",
    "hyperbolic-paraboloid": "ch1285_paraboloidhyperbolic_2",
    "hyperboloid-one-sheet": "ch1286_hyperboloid1_2",
    "hyperboloid-two-sheets": "ch1287_hyperboloid2_2",
    "pseudosphere": "ch1210_pseudosphere_2",
    "dini-surface": "ch1255_dini_2",
    "kuen-surface": "ch1293_kuen_2",
    "catenoid": "ch1198_catenoid_2",
    "helicoid": "ch1279_helicoid_2",
    "enneper-surface": "ch1264_enneper_2",
    "costa-surface": "ch1242_costa_2",
    "henneberg-surface": "ch1284_henneberg_2",
    "catalan-surface": "ch1222_minimale_catalan_2",
    "bour-surface": "ch1214_bour_2",
    "richmond-surface": "ch1329_richmond_2",
    "scherk-doubly-periodic": "ch1335_scherk_2",
    "schwarz-p": "ch1336_schwarz_2",
    "schwarz-d": "ch1336_schwarz_2",
    "gyroid": "ch1277_gyroide_2",
    "neovius-surface": "ch1307_neovius_2",
    "klein-bottle": "ch1215_klein_4",
    "boys-surface": "ch1216_boy_2",
    "roman-surface": "ch1331_romaine_2",
    "morin-surface": "ch1306_morin_2",
    "dupin-cyclide": "ch1245_cyclidededupin_2",
    "whitney-umbrella": "ch1355_whitney_2",
    "right-conoid": "ch1239_conoid_2",
    "gabriels-horn": "ch1269_gabriel_2",
    "tannery-pear": "ch1316_tannery_2",
    "barth-sextic": "ch1209_barth_2",
    "cayley-nodal-cubic": "ch1225_cayley_2",
    "clebsch-diagonal-cubic": "ch1231_clebsch_2",
    "kummer-quartic": "ch1294_kummer_2",
    "cartans-umbrella": "ch1218_cartan_2",
    "titeica-surface": "ch1349_titeica_2",
    "henrici-cubic": "ch1363_henrici_2",
    "bohemian-dome": "ch1213_boheme_2",
    "astroidal-ellipsoid": "ch1202_astroidal_2",
    "bouguer-dome": "ch1360_bouguer_2",
    "neiloid": "ch1308_neiloide_2",
    "hanging-drop-of-water": "ch1258_gouttedeau_2",
    # NOT "ch1340_sinus_2": that chapter is Ferreol's SINE SURFACE, not
    # the lantern. The mirror has no Schwarz-lantern chapter at all, so
    # the record carries no mathcurve id rather than a wrong one -- the
    # cross-check caught this by comparing the page title with the
    # record name, which is exactly what that comparison is for.
    "sieverts-surface": "ch1339_sievert_2",
    "gaudi-surface": "ch1270_gaudi_2",
    "guimard-surface": "ch1275_guimard_2",
    "milk-carton-surface": "ch1300_berlingot_2",
    "skew-ruled-cubic": "ch1226_cubique_reglee_2",
    "tangent-developable": "ch1254_devellopabledestangentes_2",
    "delaunay-surface": "ch1252_delaunay_4",
    "trinoid": "ch1309_trinoide_2",
    "jeeners-flower": "ch1266_jeener_2",
    "steiner-surface": "ch1342_steiner_2",
    "seashell": "ch1337_coquillage_2",
    "egg-box": "ch1192_boiteaoeufs_2",
    "surface-of-constant-slope": "ch1193_talus_2",
    "dyck-surface": "ch1260_dyck_2",
    "cassini-surface": "ch1219_cassini_4",
    # NOT ch1271_genre_4: that chapter defines the GENUS of a surface as
    # a concept; it is not a page about the genus-g surface as an object.
    # A cross-reference should resolve to the object.
    "veronese-surface": "ch1356_veronese_2",
    "willmore-surface": "ch1358_willmore_2",
    "zindler-conoid": "ch1359_zindler_2",
    "goursat-surface": "ch1274_goursat_4",
}

_stems_cache = None


def stems():
    """Every Ferreol surface-chapter stem in the mirror, or []."""
    global _stems_cache
    if _stems_cache is None:
        try:
            _stems_cache = sorted(
                f[:-3] for f in os.listdir(SURFACES) if f.endswith(".md"))
        except OSError:
            _stems_cache = []
    return _stems_cache


def _norm(text):
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def mathcurve_id(slug, name, available=None):
    """Ferreol chapter stem for a record, or None.

    Alias first; otherwise a STRICT match -- the record's normalised name
    must appear whole inside the stem's descriptive part. No fuzz.
    """
    if slug in MATHCURVE_ALIAS:
        stem = MATHCURVE_ALIAS[slug]
        pool = available if available is not None else stems()
        return stem if (not pool or stem in pool) else None
    pool = available if available is not None else stems()
    if not pool:
        return None
    key = _norm(name)
    if len(key) < 5:
        return None
    hits = [s for s in pool if key in _norm(s.split("_", 1)[-1])]
    return hits[0] if len(hits) == 1 else None


def mirror_present():
    return bool(stems())


def _selftest():
    """Checks the alias table's shape and the matcher's strictness."""
    pool = ["ch1198_catenoid_2", "ch1291_tore_2", "ch1279_helicoid_2",
            "ch1216_boy_2", "ch1264_enneper_2"]

    assert mathcurve_id("catenoid", "Catenoid", pool) == "ch1198_catenoid_2"
    assert mathcurve_id("torus", "Torus", pool) == "ch1291_tore_2", \
        "the French stem must come from the alias table, not from matching"

    # a name with no stem must give None, not a near miss
    assert mathcurve_id("gyroid", "Gyroid", pool) is None

    # strict containment only
    assert mathcurve_id("helicoid", "Helicoid", pool) == "ch1279_helicoid_2"
    assert mathcurve_id("xyz", "Qq", pool) is None

    # an alias naming a stem absent from the pool resolves to None rather
    # than pointing at a page the reader does not have
    assert mathcurve_id("kuen-surface", "Kuen Surface", pool) is None

    # table hygiene
    for slug, stem in MATHCURVE_ALIAS.items():
        assert slug == slug.lower() and " " not in slug, slug
        assert re.match(r"^ch\d+_", stem), (slug, stem)

    # against the real mirror if it is mounted
    if mirror_present():
        pool = stems()
        bad = [(s, st) for s, st in MATHCURVE_ALIAS.items() if st not in pool]
        assert not bad, "alias points at a stem not in the mirror: %r" % bad[:5]

    print("RESULT: OK  (surfdb.sources, %d aliases, mirror %s)"
          % (len(MATHCURVE_ALIAS), "present" if mirror_present() else "absent"))
