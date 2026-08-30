# The row -> record mapping: the curated core of the build.
#
# THE PROBLEM THIS SOLVES.  The generator registries in math_art/ are
# keyed per OPERATOR ROW.  Identity in this database is per SURFACE.
# Those are not the same thing, and the difference is not marginal:
#
#   * Catalan's minimal surface IS the Bjorling surface of a cycloid.
#     Both ship, as `CATALAN` and `BJ_CYCLOID`, and the second row's own
#     label says "(Catalan)".  One surface, two rows.
#   * The Whitney umbrella ships twice -- parametrically in
#     ruled_surface_generator and implicitly as the Hauser row in
#     surfaces/algebraic.  One surface, two rows, two DEFINITION MODES.
#   * The helicoid ships in minsurf and again in ruled_surface_generator;
#     the catenoid ships in minsurf and again as delaunay_generator's
#     H -> 0 member.
#   * Ring, horn and spindle Dupin cyclides are three rows and three
#     REGIMES OF ONE FORMULA -- one family record with three specimens.
#   * Schwarz P, the gyroid and Schwarz D are three points of one Bonnet
#     associate family, but each has its own name, its own literature and
#     its own space group, so each is PROMOTED to a record of its own.
#
# A build that emitted one record per row would double-count, and a build
# that merged on name similarity would guess.  So the dispositions are
# DECLARED here, applied before any record is emitted, and auditable:
# `python tools/surfdb_build.py --mapping-report` prints every row and
# what happened to it.
#
# data/polyhedra hit the mild version of this ("the square prism IS the
# cube... `emit` now raises rather than overwriting").  Here it is an
# order of magnitude more frequent, which is why it gets its own module
# instead of a few special cases inside the builder.
#
# FOUR DISPOSITIONS
#   emit      this row becomes a record                        (the default)
#   merge     another construction of an existing record: append to its
#             `construction[]`, and if the mode differs, to its
#             `alternate_definitions[]`
#   specimen  a distinguished member of a family record
#   promote   a family member with its own identity
#
# Rows are addressed as "<source>:<key>", where <source> names the
# registry (see SOURCES in surfdb_build.py).

# ---------------------------------------------------------------------------
# MERGE -- "<source>:<key>" -> slug of the record this row is another
# construction of.  Only asserted where identity is a THEOREM or where the
# shipped code itself states it; anything merely suspected goes in
# SUSPECTED_SAME below instead, and stays two records.
# ---------------------------------------------------------------------------

MERGE = {
    # The box-symmetry series is ONE template selected by a sign vector,
    # so its three shipped members are three constructions of the same
    # catalogued family rather than three surfaces.  MERGE, not ALIAS:
    # an alias is one-to-one and three rows aliased to one slug collide.
    "tpms_exact:BOX_1001": "weber-pqr-series",
    "tpms_exact:BOX_1010": "weber-pqr-series",
    "tpms_exact:BOX_1011": "weber-pqr-series",

    # Catalan's minimal surface is the Bjorling surface of a cycloid.
    # The zoo row's own label reads "Bjorling: Cycloid (Catalan)", so the
    # identity is asserted by the shipped code, not inferred by us.
    "minsurf:BJ_CYCLOID": "catalan-surface",

    # The Whitney umbrella, parametric and implicit.  Two definition
    # modes for one surface -- exactly the case `alternate_definitions`
    # exists for.
    "algebraic:WHITNEY": "whitney-umbrella",

    # The helicoid is a minimal surface and a ruled surface; it ships as
    # both.  Filed under `minimal/` (the stronger claim), reachable from
    # two operators.
    "ruled:HELICOID": "helicoid",

    # The catenoid is the H -> 0 member of the Delaunay roulette family
    # and ships from that operator too.
    "delaunay:CATENOID": "catenoid",

    # A Delaunay "cylinder" is the circular cylinder, which is a quadric.
    "delaunay:CYLINDER": "circular-cylinder",

    # The hyperbolic helicoid appears in the swept family and as a named
    # helical surface; one surface.
    "ruled:TWIST_STRIP": "twisted-strip",

    # --- surfaces the algebraic module has now BUILT that the curated
    # tables already described as unbuilt ----------------------------------
    #
    # These five were curated from the Algebraic Surface Homepage and the
    # Encyclopedia with `blocked_by: "equation deliberately NOT
    # transcribed"`. The equations have since been transcribed from the
    # primary sources and shipped, so the row and the curated record are
    # the same surface and must be ONE record.
    #
    # Without these entries the build emits a second record per surface
    # under a slightly different slug -- "modified-chmutov-octic-144-nodes"
    # beside "modified-chmutov-octic" -- and the pair is worse than either
    # alone: the curated half keeps its published node count and goes on
    # claiming the surface is unbuilt, while the built half carries the
    # verified polynomial and no node count, so neither record can be
    # checked against the other and the coverage figure counts one surface
    # twice. The published invariant and the implementation have to meet
    # in a single record or the invariant cannot gate the implementation,
    # which is the entire reason the counts are stored.
    "algebraic:ENDRASS_160": "endrass-octic-160",
    "algebraic:MOD_CHMUTOV": "modified-chmutov-octic",
    "algebraic:VAN_STRATEN_165": "van-straten-octic",
    "algebraic:MOBIUS_SURFACE": "mobius-surface",
    "algebraic:NORDSTRAND_WEIRD": "nordstrand-weird-surface",
    # VAN_STRATEN_124 is deliberately NOT here: the curated tables carry
    # no 124-node record for it to merge into (only the open-ended
    # "van-straten-dihedral-series"), so it is a genuinely new record.
}

# ---------------------------------------------------------------------------
# SPECIMEN -- "<source>:<key>" -> (family slug, specimen label).
# The row is a distinguished member of a parameterised family, not a
# surface in its own right.  Ring / horn / spindle are regimes of ONE
# cyclide formula; Minding's bulge and spindle are two regimes of one
# K = -1 profile; the K = +1 sphere / spindle / bulge likewise.
# ---------------------------------------------------------------------------

SPECIMEN = {
    "curiosity:CYCLIDE_RING":     ("dupin-cyclide", "ring"),
    "curiosity:CYCLIDE_HORN":     ("dupin-cyclide", "horn"),
    "curiosity:CYCLIDE_SPINDLE":  ("dupin-cyclide", "spindle"),

    "curiosity:TANNERY_PEAR":      ("tannery-pear", "pear (half the figure eight)"),
    "curiosity:TANNERY_HOURGLASS": ("tannery-pear", "hourglass (the whole figure eight)"),

    "hyperbolic:MINDING_BULGE":   ("minding-surface", "bulge"),
    "hyperbolic:MINDING_SPINDLE": ("minding-surface", "spindle"),

    "spherical:SPHERE":   ("k-positive-revolution", "sphere"),
    "spherical:SPINDLE":  ("k-positive-revolution", "spindle"),
    "spherical:BULGE":    ("k-positive-revolution", "bulge"),

    "ruled:PLUCKER": ("right-conoid", "Plucker's cylindroid"),
    "ruled:NFOLD":   ("right-conoid", "n-fold conoid"),
    "ruled:WALLIS":  ("right-conoid", "Wallis' conical edge"),
    "ruled:ZINDLER": ("right-conoid", "Zindler's conoid"),

    # NOTE: these keys were verified against the modules' actual tables
    # (tools/surfdb/registry.py reads them without importing bpy). Guessed
    # spellings -- CUBOCTA6, DODEC10, REULEAUX_TET -- do not exist, and the
    # build raises on a specimen whose row never appears rather than
    # silently dropping it.
    "steinmetz:BICYLINDER": ("steinmetz-solid", "bicylinder"),
    "steinmetz:TRICYLINDER": ("steinmetz-solid", "tricylinder"),
    "steinmetz:CUBE4":      ("steinmetz-solid", "four cube diagonals"),
    "steinmetz:CUBOCT6":    ("steinmetz-solid", "six cuboctahedral axes"),
    "steinmetz:ICOSA6":     ("steinmetz-solid", "six icosahedral axes"),
    "steinmetz:DODECA10":   ("steinmetz-solid", "ten dodecahedral axes"),
    "steinmetz:TRUNCOCT12": ("steinmetz-solid", "twelve truncated-octahedral axes"),
    "steinmetz:EQUIDOMOID": ("steinmetz-solid", "equidomoid"),

    "constwidth:MEISSNER":   ("constant-width-solid", "Meissner tetrahedron"),
    "constwidth:REVOLUTION": ("constant-width-solid", "Reuleaux solid of revolution"),
    "constwidth:REULEAUX":   ("constant-width-solid", "Reuleaux tetrahedron"),

    # The lobed rows on mesh.delaunay_surface_add are bubbleton PRESETS --
    # a bubbleton with n lobes grafted on -- not distinct surfaces.
    "delaunay:CYL2":       ("bubbleton", "cylinder, 2 lobes"),
    "delaunay:CYL3":       ("bubbleton", "cylinder, 3 lobes"),
    "delaunay:CYL5":       ("bubbleton", "cylinder, 5 lobes"),
    "delaunay:UND2":       ("bubbleton", "unduloid, 2 lobes"),
    "delaunay:UND3":       ("bubbleton", "unduloid, 3 lobes"),
    "delaunay:NOD2":       ("bubbleton", "nodoid, 2 lobes"),
    "delaunay:TWIST":      ("bubbleton", "twizzler (3 lobes, 2 covers)"),
    "delaunay:DOUBLE_UND": ("bubbleton", "double bubbleton on an unduloid"),
    "delaunay:SPLASH":     ("bubbleton", "colliding bubbles (2 and 3)"),
}

# ---------------------------------------------------------------------------
# PROMOTE -- "<source>:<key>" -> (slug, family slug).
# The row is a member of a parameterised family that has earned its own
# record, by the SS6.1 rule: its own name, its own literature and its own
# symmetry group.  Schwarz P, the gyroid and Schwarz D have three
# different space groups (Im-3m, Ia-3d, Pn-3m) and three separate bodies
# of work, so they are three records plus one family record -- not three
# specimens of the Bonnet family.
#
# The same rule promotes the unduloid and the nodoid out of the Delaunay
# family, which is what makes `cmc/` records-plus-family rather than a
# flat list.
# ---------------------------------------------------------------------------

PROMOTE = {
    "tpms:P":  ("schwarz-p", "pgd-associate-family"),
    "tpms:G":  ("gyroid", "pgd-associate-family"),
    "tpms:D":  ("schwarz-d", "pgd-associate-family"),

    "delaunay:UNDULOID": ("unduloid", "delaunay-surface"),
    "delaunay:NODOID":   ("nodoid", "delaunay-surface"),
    "delaunay:SPHERE":   ("sphere-chain", "delaunay-surface"),
}

# ---------------------------------------------------------------------------
# ALIAS -- "<source>:<key>" -> slug, where the natural slug from the row's
# label is not the name the literature uses.  Purely cosmetic; no identity
# claim.
# ---------------------------------------------------------------------------

ALIAS = {
    # The classical rows carry bare labels ("Enneper", "Costa (genus 1)"),
    # which slugify to bare stems. The literature names carry the word
    # "surface", and the curation table is keyed that way, so these are
    # aliased rather than left to drift apart -- the build reports any
    # curated slug that never gets emitted, which is how the mismatch was
    # found.
    "minsurf:ENNEPER": "enneper-surface",
    "minsurf:COSTA": "costa-surface",
    "minsurf:BOUR": "bour-surface",
    "minsurf:HENNEBERG": "henneberg-surface",
    "minsurf:RICHMOND": "richmond-surface",
    "minsurf:RIEMANN": "riemann-minimal-example",
    "minsurf:KNOID": "jorge-meeks-k-noid",
    "minsurf:CATHEL": "catenoid-helicoid-associate-family",
    "minsurf:SCHERK1": "scherk-doubly-periodic",
    "minsurf:SCHERK_TOWER": "scherk-saddle-tower",
    "minsurf:COSTA_HM": "costa-hoffman-meeks",
    "minsurf:CHEN_GACK": "chen-gackstatter",
    "minsurf:MEEKS_MOBIUS": "meeks-mobius-strip",
    "minsurf:BJ_CIRCLE": "bjorling-twisted-band",
    "tpms_exact:PGD": "pgd-associate-family",
    # The exact-Weierstrass rows added on the minimal-periodic branch.
    # These MUST live here rather than being hand-edited into the record
    # files: `data/surfaces` is regenerated from the registries, so a
    # hand-flipped record is reverted the moment anyone rebuilds -- which
    # is exactly what happened once, silently, in a merge.
    "tpms_exact:H": "h-exact",
    "tpms_exact:CLP": "clp-exact",
    "tpms_exact:CLP_HANDLE": "clp-handle-exact",
    "tpms_exact:LIDINOID": "lidinoid-exact",
    "tpms_exact:RPD": "rpd-exact",
    "tpms_exact:HT": "schoen-h-t",
    "tpms_exact:SS": "schoen-s-s",
    "tpms_exact:H2R": "weber-h2r",
    "tpms_exact:TR": "weber-trr",
    "tpms_exact:STESSMANN": "stessmann-surface",
    "tpms_exact:RII": "schoen-rii",
    "tpms_exact:CH": "schoen-c-h",
    "tpms_exact:I6": "schoen-i6",
    "tpms_exact:FRD_EXACT": "weber-fr-d",
    "tpms_exact:FRDR": "schoen-frd-r",
    "tpms_exact:TRIPLY_COSTA": "triply-periodic-costa",
    "tpms_exact:SIMOES_BATISTA": "simoes-batista-surface",
    "tpms_exact:R3_RING": "schoen-riii",
    "tpms_exact:I8_RING": "schoen-i8",
    "tpms_exact:I9_RING": "schoen-i9",
    "tpms:LIDINOID": "lidinoid",
    "tpms:NEOVIUS": "neovius-surface",
    "tpms:FK_CS": "fischer-koch-cs",
    "tpms:FK_CY": "fischer-koch-cy",
    "algebraic:BARTH": "barth-sextic",
    "algebraic:BARTH_DECIC": "barth-decic",
    "algebraic:ENDRASS": "endrass-octic",
    "algebraic:LABS": "labs-septic",
    "algebraic:CAYLEY": "cayley-nodal-cubic",
    "algebraic:CLEBSCH": "clebsch-diagonal-cubic",
    "algebraic:KUMMER": "kummer-quartic",
    "algebraic:CHMUTOV": "chmutov-sextic",
    "algebraic:TOGLIATTI": "togliatti-quintic",
    "algebraic:MONKEY": "monkey-saddle",
    "algebraic:HEART": "taubin-heart",
    "algebraic:TANGLE": "tangle-cube",
    "algebraic:DINGDONG": "ding-dong-surface",
    "curiosity:ZOLL": "zoll-surface",
    "curiosity:SCHWARZ_LANTERN": "schwarz-lantern",
    "curiosity:GABRIEL": "gabriels-horn",
    "topological:KLEIN": "klein-bottle",
    "topological:KLEIN8": "klein-bottle-figure-eight",
    "topological:BOY": "boys-surface",
    "topological:MORIN": "morin-surface",
    "topological:ROMAN": "roman-surface",
    "topological:CROSSCAP": "cross-cap",
    "topological:SUDANESE": "sudanese-mobius-band",
    "topological:STEINER": "steiner-surface",
    "topological:NONORIENT": "non-orientable-genus-k",
    "topological:GENUS": "genus-g-surface",
    "topological:TWIST_STRIP": "twisted-strip",
    "hyperbolic:PSEUDOSPHERE": "pseudosphere",
    "hyperbolic:DINI": "dini-surface",
    "hyperbolic:KUEN": "kuen-surface",
    "hyperbolic:BREATHER": "breather-surface",
    "hyperbolic:AMSLER": "amsler-surface",
    "spherical:SIEVERT": "sieverts-surface",
    "ruled:WHITNEY": "whitney-umbrella",
    "ruled:CONOID": "right-conoid",
    "ruled:HYPERBOLOID": "hyperboloid-one-sheet",
    "ruled:HYPAR": "hyperbolic-paraboloid",
    "ruled:TANGENT_DEV": "tangent-developable",
    "ruled:GAUDI": "gaudi-surface",
    "ruled:GUIMARD": "guimard-surface",
    "ruled:MILK_CARTON": "milk-carton-surface",
    "ruled:RULED_CUBIC": "skew-ruled-cubic",
    "ruled:CONSTANT_SLOPE": "surface-of-constant-slope",
}

# ---------------------------------------------------------------------------
# SUSPECTED_SAME -- pairs believed identical but NOT confirmed.
# These stay as separate records, carrying `relations.same_surface_as`
# with confidence "suspected" and the check that would decide it.  This
# is the mechanism that keeps the database from either guessing a merge
# or silently shipping a duplicate.
# ---------------------------------------------------------------------------

SUSPECTED_SAME = [
    # SETTLED 2026-08-28 by measurement, which is what this mechanism is
    # for: the claim was recorded rather than guessed, then checked.
    #
    # Meshing each nodal level set over a common one-cell block at
    # resolution 64 and counting V - E + F gives
    #
    #     Schwarz P   chi =   -4          Fischer-Koch C(S)  chi = -128
    #     Schwarz D   chi =  -10          Fischer-Koch C(Y)  chi =  -15
    #
    # Euler characteristic scales roughly linearly with the number of
    # cells in the block, so even allowing for the fact that C(S)'s nodal
    # function has half the period of P's -- its leading terms are
    # cos 2x, cos 2y, cos 2z -- an eight-fold cell count would predict
    # chi near -32, not -128. A factor of four beyond that is not a cell
    # convention; the surfaces differ.
    #
    # The claim is therefore REFUTED for what this repo ships. The likely
    # explanation is that Brakke indexes the Fischer-Koch family
    # differently from the sources the shipped rows follow (Koch & Fischer
    # 1988 for C(S), von Schnering & Nesper 1991 for C(Y)); the remark may
    # be true of surfaces those names denote elsewhere. That caveat is
    # recorded rather than resolved, because resolving it would mean
    # transcribing Brakke's own definitions.
    {
        "a": "fischer-koch-cs", "b": "schwarz-p",
        "confidence": "refuted",
        "check": "Measured: chi over a common one-cell block at resolution "
                 "64 is -128 for C(S) against -4 for P. Not the same "
                 "surface, and not explainable by the half-period cell.",
        "source": "K. Brakke's Surface Evolver periodic-surface collection "
                  "lists Fischer-Koch C(S) as 'later recognised as the P "
                  "surface'; the shipped row follows Koch & Fischer (1988) "
                  "and does not agree with that identification.",
    },
    {
        "a": "fischer-koch-cy", "b": "schwarz-d",
        "confidence": "refuted",
        "check": "Measured: chi over a common one-cell block at resolution "
                 "64 is -15 for C(Y) against -10 for D.",
        "source": "Brakke, ibid., for C(Y) and the D surface; the shipped "
                  "row follows von Schnering & Nesper (1991).",
    },
    {
        "a": "bjorling-twisted-band", "b": "meeks-mobius-strip",
        "confidence": "suspected",
        "check": "the Bjorling surface of a circle with a uniformly rotating "
                 "normal is a minimal Mobius band; whether it is congruent to "
                 "Meeks' is not asserted by either row's label, so it is not "
                 "merged. Compare total curvature (Meeks' is -6*pi) and the "
                 "boundary-curve knot type.",
        "source": "row labels in math_art/minsurf/zoo.py: 'Bjorling: Twisted "
                  "Band (Mobius)' and 'Meeks Mobius Strip'.",
    },
]


def disposition(source, key):
    """What happens to registry row `source:key`.

    Returns (kind, payload) where kind is one of
    'merge' | 'specimen' | 'promote' | 'emit'.
    """
    ref = "%s:%s" % (source, key)
    if ref in MERGE:
        return "merge", MERGE[ref]
    if ref in SPECIMEN:
        return "specimen", SPECIMEN[ref]
    if ref in PROMOTE:
        return "promote", PROMOTE[ref]
    return "emit", ALIAS.get(ref)


def _selftest():
    """Consistency checks on the mapping table itself; raises on failure."""
    # A row must have at most ONE disposition.  Overlapping tables are how
    # a mapping silently starts guessing.
    seen = {}
    for table, kind in ((MERGE, "merge"), (SPECIMEN, "specimen"),
                        (PROMOTE, "promote")):
        for ref in table:
            if ref in seen:
                raise AssertionError(
                    "row %r has two dispositions: %s and %s"
                    % (ref, seen[ref], kind))
            seen[ref] = kind

    # ALIAS is cosmetic and must not overlap a real disposition -- an alias
    # on a merged row would name a record that is never emitted.
    for ref in ALIAS:
        if ref in seen:
            raise AssertionError(
                "row %r is aliased AND %s; an alias renames an emitted "
                "record, so the two cannot both apply" % (ref, seen[ref]))

    # every reference is "<source>:<KEY>"
    for ref in list(seen) + list(ALIAS):
        if ref.count(":") != 1 or not ref.split(":")[1]:
            raise AssertionError("malformed row reference %r" % ref)

    # specimen and promote payloads are well-formed pairs
    for ref, val in SPECIMEN.items():
        assert isinstance(val, tuple) and len(val) == 2, ref
    for ref, val in PROMOTE.items():
        assert isinstance(val, tuple) and len(val) == 2, ref

    # a promoted row must not also be claimed as a specimen of its family
    for ref, (slug, fam) in PROMOTE.items():
        assert slug != fam, "%s promotes to its own family slug" % ref

    # suspected-same entries never name the same record twice, and must
    # carry the check that would decide them
    for ent in SUSPECTED_SAME:
        assert ent["a"] != ent["b"], ent
        assert ent.get("check"), "a suspected identity without a check is a rumour"
        assert ent.get("source"), ent

    assert disposition("algebraic", "WHITNEY") == ("merge", "whitney-umbrella")
    assert disposition("curiosity", "CYCLIDE_HORN")[0] == "specimen"
    assert disposition("tpms", "G") == ("promote", ("gyroid", "pgd-associate-family"))
    assert disposition("algebraic", "CALYX") == ("emit", None)
    assert disposition("algebraic", "BARTH") == ("emit", "barth-sextic")

    print("RESULT: OK  (surfdb.mapping, %d merges, %d specimens, %d promotions)"
          % (len(MERGE), len(SPECIMEN), len(PROMOTE)))
