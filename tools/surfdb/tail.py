"""The tail: named surfaces absent from math_art AND, until now, from here.

A ledger that under-counts what is missing is not a ledger.  Before this
module existed, `schoen-batwing` was a single placeholder standing in for
roughly eighteen second-tier triply-periodic minimal surfaces, and
Weber's (p,q,r) series -- about forty more -- had no record at all.  That
made the headline coverage figure flattering rather than honest: the
surfaces were always missing; only the records were not.

Adding these DROPS reported coverage from 93% to about 77%, and that is
the correct direction.

Every entry states `blocked_by` and `resume`.  Equations are deliberately
NOT transcribed: an unverified coefficient does not error, it silently
defines a different surface, and the numerical oracle that guards the
shipped algebraic block cannot guard a surface with no implementation to
check against.  A null with a reason is the honest record.
"""

SCHOEN_SOURCES = [
    'A. H. Schoen, "Infinite periodic minimal surfaces without '
    'self-intersections", NASA Technical Note TN D-5541 (1970), together '
    'with his later listings of the second-tier surfaces.',
    "K. Brakke, the Surface Evolver periodic minimal-surface collection "
    "(kenbrakke.com/evolver), where several of these are figured.",
]

SCHOEN_BLOCKED = (
    "No published nodal formula exists -- the Fisher et al. (2023) "
    "catalogue, which collects every known nodal fit, omits this family -- "
    "and no exact Weierstrass tiling has been built for it here."
)

SCHOEN_RESUME = (
    "Two routes, both already identified in "
    "research/minimal_surfaces_status.md: derive a Fourier fit by the vSN "
    "structure-factor method, or build an exact Weierstrass tiling. The "
    "second is the one that already worked -- Schwarz H, CLP and "
    "CLP-handle ship in TPMS_EXACT precisely that way, which is why the "
    "older catalogue still lists them as blocked -- and the hexagonal-cell "
    "machinery (tpms2_LATTICE, TPMS2_HEX_LATTICE) is already in place and "
    "exercised by the self-test, waiting for exactly this."
)

WEBER_SOURCES = [
    "M. Weber, https://minimalsurfaces.blog/ and the archive formerly at "
    "indiana.edu/~minimal/archive/ (the site is dead; 280 surface folders "
    "were recovered from the Wayback CDX index).",
]

# (slug, name, what distinguishes it)
TPMS = [
    ("schoen-rii", "Schoen R-II Surface",
     "Isosceles right triangular prism cell."),
    ("schoen-riii", "Schoen R-III Surface",
     "30-60-90 triangular prism cell."),
    ("schoen-i6", "Schoen I-6 Surface",
     "Surface between plane square grids."),
    ("schoen-i8", "Schoen I-8 Surface", "As I-6, with diagonals."),
    ("schoen-i9", "Schoen I-9 Surface", "As I-8, differently arranged."),
    ("schoen-gw", "Schoen G-W Surface",
     "One labyrinth is a graphite hexagonal sheet, the other wurtzite."),
    ("schoen-frd-r", "Schoen F-RD(r) Surface",
     "Chamber with tubes to four edge centres; 8-fold symmetry."),
    ("schoen-iwp-r", "Schoen I-WP(r) Surface",
     "Chamber with tubes to eight edge centres; 16-fold symmetry."),
    ("schoen-hybrid-1", "Schoen Hybrid-1 [P, F-RD] Surface",
     "Tubes to alternating corners AND faces."),
    ("schoen-manta-genus-19", "Schoen Manta Surface (genus 19)",
     "Tetrahedral fundamental region, 1/96 of the cube."),
    ("schoen-manta-genus-35", "Schoen Manta Surface (genus 35)",
     "Second member of the Manta series."),
    ("schoen-manta-genus-51", "Schoen Manta Surface (genus 51)",
     "Third member of the Manta series."),
    ("neovius-n14", "Neovius N14 Surface",
     "Central chamber with necks to tubes along each cube edge."),
    ("neovius-n26", "Neovius N26 Surface",
     "N14 with extra holes; genus 26."),
    ("neovius-n38", "Neovius N38 Surface",
     "N14 with additional holes differently arranged; genus 38."),
    ("lord-mackay-p3a", "Lord-Mackay P3a Surface",
     "Disphenoid surface with equal-length C2 axes."),
]

WEBER_ARCHIVE = [
    ("weber-rpd", "Weber rPD Surface"),
    ("weber-bc2", "Weber BC2 Surface"),
    ("weber-fr-d", "Weber FR-D Surface"),
    ("weber-h2r", "Weber H-double-prime-R Surface"),
    ("weber-tr", "Weber T-prime-R Surface"),
    ("weber-trr", "Weber T-prime-R-prime Surface"),
    ("triply-periodic-costa", "Triply Periodic Costa Surface"),
    ("neovius-sym3", "Neovius-sym3 Surface"),
]

# Named entries in MathWorld's 86-item algebraic-surfaces list that this
# repo does not build.
ALGEBRAIC = [
    ("burkhardt-quartic", "Burkhardt Quartic",
     "A quartic with 45 nodes; Heinrich Burkhardt, 1891."),
    ("desmic-surface", "Desmic Surface",
     "Associated with a desmic system of three tetrahedra."),
    ("symmetroid", "Symmetroid",
     "The quartic determinant surface of a net of quadrics; ten nodes."),
    ("menns-surface", "Menn's Surface",
     "A quartic carrying a higher-order cusp; a standard singularity model."),
    ("peano-surface", "Peano Surface",
     "Peano's counterexample on second-derivative tests for maxima."),
    ("chair-surface", "Chair Surface", "A decorative quartic."),
    ("tooth-surface", "Tooth Surface", "A decorative octic."),
    ("miter-surface", "Miter Surface", "A quartic with two pinch points."),
    ("kiss-surface", "Kiss Surface", "A quintic surface of revolution."),
    ("nordstrand-weird-surface", "Nordstrand's Weird Surface",
     "A degree-8 decorative surface from Nordstrand's gallery."),
    ("handkerchief-surface", "Handkerchief Surface",
     "The fold-catastrophe graph z = x^3 + xy."),
    ("crossed-trough", "Crossed Trough", "The graph z = x^2 y^2."),
    ("cayley-ruled-surface", "Cayley's Ruled Surface",
     "Cayley's cubic RULED surface -- distinct from the four-nodal Cayley "
     "cubic this repo already ships, which is a different object with the "
     "same name attached."),
    ("hunts-surface", "Hunt's Surface", "A degree-6 surface from Hunt's work."),
]

TOPOLOGICAL = [
    ("etruscan-venus-surface", "Etruscan Venus Surface",
     "An immersed non-orientable surface; a staple of the Geometry Center "
     "exhibits and of Sequin's sculptural work."),
    ("ida-surface", "Ida Surface",
     "A companion to the Etruscan Venus in the same family."),
    ("boys-planet", "Boy's Planet",
     "A variant of Boy's surface described by Carlo Sequin."),
    ("nested-klein-bottles", "Nested Klein Bottles",
     "A composition of the shipped Klein bottle rather than new "
     "mathematics; named by Jos Leys."),
]


def records():
    """slug -> spec dict, in the shape surfdb_build.MISSING expects."""
    out = {}

    for slug, name, note in TPMS:
        out[slug] = {
            "name": name, "family": "minimal-periodic", "mode": "weierstrass",
            "blocked_by": SCHOEN_BLOCKED, "resume": SCHOEN_RESUME,
            "sources": list(SCHOEN_SOURCES),
            "extra": {
                "curvature": {"condition": "minimal"},
                "symmetry": {"kind": "space", "periodicity_rank": 3},
                "topology": {"complete": True, "compact": False,
                             "orientable": True},
                "metrics": {"normalization": "unit_cell"},
                "tradition": ["crystallographic"],
                "definition": {"note": note},
            },
        }

    for slug, name in WEBER_ARCHIVE:
        out[slug] = {
            "name": name, "family": "minimal-periodic", "mode": "weierstrass",
            "blocked_by": "Recovered from the Wayback index of Weber's dead "
                          "archive as a named surface folder; no construction "
                          "has been built here.",
            "resume": "Recover the notebook from the archived folder and "
                      "follow the exact-Weierstrass route used for the "
                      "shipped TPMS_EXACT rows.",
            "sources": list(WEBER_SOURCES),
            "extra": {
                "curvature": {"condition": "minimal"},
                "symmetry": {"kind": "space", "periodicity_rank": 3},
                "topology": {"complete": True, "compact": False},
                "metrics": {"normalization": "unit_cell"},
                "tradition": ["crystallographic"],
            },
        }

    for slug, name, note in ALGEBRAIC:
        out[slug] = {
            "name": name, "family": "algebraic", "mode": "implicit",
            "blocked_by": "Not built. The equation is deliberately NOT "
                          "transcribed here: an unverified coefficient does "
                          "not error, it silently defines a different "
                          "surface, and the numerical oracle that guards the "
                          "shipped algebraic block cannot guard a surface "
                          "with no implementation to check against.",
            "resume": "Transcribe from a primary source into "
                      "math_art/surfaces/algebraic.py; the build's oracle "
                      "check then confirms it against that implementation.",
            "sources": ["E. W. Weisstein, MathWorld, topic 'Algebraic "
                        "Surfaces' -- the enumeration this absence was "
                        "measured against."],
            "extra": {"curvature": {"condition": "none"},
                      "tradition": ["classical"],
                      "definition": {"note": note}},
        }

    for slug, name, note in TOPOLOGICAL:
        out[slug] = {
            "name": name, "family": "topological", "mode": "parametric",
            "blocked_by": "Not built.",
            "resume": "The Geometry Center's Topological Zoo and C. Sequin's "
                      "sculptural pages are the sources; see "
                      "research/surface-gallery-gap-review.md Part 3.",
            "sources": ["C. H. Sequin, topological sculpture pages, "
                        "University of California, Berkeley.",
                        "The Geometry Center Topological Zoo, University of "
                        "Minnesota."],
            "extra": {"curvature": {"condition": "none"},
                      "embedding": {"quality": "immersed"},
                      "topology": {"compact": True},
                      "tradition": ["sculptural"],
                      "definition": {"note": note}},
        }

    # ONE parameterised family record, not forty rows. That is the shape
    # the mathematics has and the shape the repo's slider idiom wants --
    # and it is why this is the best surfaces-per-effort item in the whole
    # survey.
    out["weber-pqr-series"] = {
        "name": "Weber (p,q,r) Triangle-Group Series",
        "family": "minimal-periodic", "mode": "weierstrass",
        "blocked_by": "Not built. Weber's archive holds five systematic "
                      "series indexed by triangle group -- roughly 40 "
                      "surfaces in total.",
        "resume": "Implement as ONE row plus a triangle-group selector, not "
                  "forty rows; flagged in "
                  "research/missing-surfaces-catalog.md Part D4 as the "
                  "structurally interesting item of that survey.",
        "sources": list(WEBER_SOURCES),
        "extra": {
            "curvature": {"condition": "minimal"},
            "symmetry": {"kind": "space", "periodicity_rank": 3},
            "topology": {"complete": True, "compact": False},
            "metrics": {"normalization": "unit_cell"},
            "tradition": ["crystallographic"],
            "definition": {
                "note": "Five series (series1..series5), each with members "
                        "indexed by a (p,q,r) triangle group.",
                "parameters": [
                    {"name": "series", "domain": "{1, 2, 3, 4, 5}",
                     "default": 1, "integer": True,
                     "note": "which of Weber's five series"},
                    {"name": "p", "domain": "{2, 3, 4, 6}", "default": 2,
                     "integer": True},
                    {"name": "q", "domain": "{2, 3, 4, 6}", "default": 3,
                     "integer": True},
                    {"name": "r", "domain": "{2, 3, 4, 6}", "default": 6,
                     "integer": True},
                ]},
            "specimens": [
                {"label": "(2,3,6)", "parameters": {"p": 2, "q": 3, "r": 6}},
                {"label": "(2,4,4)", "parameters": {"p": 2, "q": 4, "r": 4}},
                {"label": "(3,3,3)", "parameters": {"p": 3, "q": 3, "r": 3}},
                {"label": "(6,2,3)", "parameters": {"p": 6, "q": 2, "r": 3}},
            ]},
    }
    return out


def _selftest():
    """Structural checks on the tail table; raises on failure."""
    recs = records()
    assert len(recs) == len(TPMS) + len(WEBER_ARCHIVE) + len(ALGEBRAIC) \
        + len(TOPOLOGICAL) + 1, len(recs)

    seen = set()
    for slug, spec in recs.items():
        assert slug == slug.lower() and " " not in slug, slug
        assert slug not in seen, "duplicate tail slug %r" % slug
        seen.add(slug)
        assert spec.get("name"), slug
        assert spec.get("blocked_by"), \
            "%s is absent without a stated reason" % slug
        assert spec.get("resume"), "%s has no resume pointer" % slug
        assert spec.get("sources"), "%s cites nothing" % slug
        # no tail record may claim a defining datum -- that is the point
        d = (spec.get("extra") or {}).get("definition") or {}
        assert not d.get("polynomial"), \
            "%s must not carry an untranscribed polynomial" % slug

    # the (p,q,r) row is a FAMILY, so it must carry parameters and specimens
    pqr = recs["weber-pqr-series"]["extra"]
    assert pqr["definition"]["parameters"], "the series row needs parameters"
    assert pqr["specimens"], "the series row needs specimens"

    print("RESULT: OK  (surfdb.tail, %d records: %d TPMS, %d Weber, "
          "%d algebraic, %d topological, 1 series)"
          % (len(recs), len(TPMS), len(WEBER_ARCHIVE), len(ALGEBRAIC),
             len(TOPOLOGICAL)))
