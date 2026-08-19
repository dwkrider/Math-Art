# The named-cut registry.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python only -- no
# `bpy`.
#
# A cut is either LITERAL -- a published tier table, entered from its
# source -- or PARAMETRIC, a constructor that solves the tiers from
# proportions.  Both arrive here as a `CutDesign`, so everything
# downstream is indifferent to which it was.
#
# Facet counts are deliberately NOT recorded in the entries.  A facet
# count is a property of the built solid, not of the design text: a tier
# whose facets are entirely cut away by a later tier contributes nothing,
# which is exactly the sort of error a hand-copied count would hide.  The
# self-tests build every entry and check the count that comes out.
#
# LICENSING.  Only designs whose source permits redistribution belong
# here.  The Standard Round Brilliant below is from Strickland's GemCad
# manual, which prints it as the format's worked example.  Third-party
# design libraries -- FacetDiagrams.org, Datavue II -- are published under
# per-designer permissions and must be IMPORTED AT RUNTIME by the user,
# never vendored into this repository.
#
# References:
#   Robert W. Strickland, "GemCad for Windows Version 1.0 User's Guide",
#     GemSoft Enterprises, 2002 -- the Standard Round Brilliant listing.

import inspect
from typing import NamedTuple

try:
    from .brilliant import (old_european, portuguese, round_brilliant,
                            single_cut)
    from .design import SRB_GEMCAD, CutDesign
    from .mixed import barion, princess, radiant
    from .rose import antwerp_rose, dutch_rose
    from .step import asscher, baguette, emerald, polygon_step
except ImportError:                     # flat import outside the package
    from brilliant import (old_european, portuguese, round_brilliant,
                           single_cut)
    from design import SRB_GEMCAD, CutDesign
    from mixed import barion, princess, radiant
    from rose import antwerp_rose, dutch_rose
    from step import asscher, baguette, emerald, polygon_step


class CutEntry(NamedTuple):
    """One catalogue row: how to name it, group it, and build it."""

    label: str
    family: str
    source: object          # a CutDesign, or (constructor, default kwargs)
    description: str = ""


CUTS = {
    "SRB_GEMCAD": CutEntry(
        label="Standard Round Brilliant",
        family="brilliant",
        source=SRB_GEMCAD,
        description="The 16-sided-outline round brilliant printed as the "
                    "worked example in Strickland's GemCad manual; cut "
                    "for a refractive index of 1.54"),
    "ROUND_BRILLIANT": CutEntry(
        label="Round Brilliant",
        family="brilliant",
        source=(round_brilliant, {}),
        description="The 57/58-facet standard round brilliant solved from "
                    "its proportions -- table, crown and pavilion angles, "
                    "girdle, star and lower-half lengths"),
    "TOLKOWSKY": CutEntry(
        label="Tolkowsky Brilliant",
        family="brilliant",
        source=(round_brilliant, dict(table=0.53, crown_angle=34.5,
                                      pavilion_angle=40.75, girdle_pct=0.02,
                                      star_len=0.55, lower_girdle_len=0.75,
                                      name="Tolkowsky Brilliant")),
        description="Marcel Tolkowsky's 1919 proportions: 53% table, 34.5 "
                    "degree crown, 40.75 degree pavilion -- the calculated "
                    "ideal the modern brilliant descends from"),
    "SINGLE_CUT": CutEntry(
        label="Single (Eight) Cut",
        family="brilliant",
        source=(single_cut, {}),
        description="17/18 facets: table, 8 bezels, 8 pavilion mains. The "
                    "melee cut, where a full brilliant's facets would be "
                    "smaller than the eye can resolve"),
    "OLD_EUROPEAN": CutEntry(
        label="Old European Cut",
        family="brilliant",
        source=(old_european, {}),
        description="The 58-facet brilliant cut for candlelight: small "
                    "table, tall crown, steep pavilion and a large open "
                    "culet"),
    "PORTUGUESE": CutEntry(
        label="Portuguese Cut",
        family="brilliant",
        source=(portuguese, {}),
        description="161 facets in many rows, each offset half a step from "
                    "the one below; sixteen pavilion planes converge on "
                    "the culet"),
    "DUTCH_ROSE": CutEntry(
        label="Dutch Rose",
        family="rose",
        source=(dutch_rose, {}),
        description="A flat base under a domed crown of 24 triangular "
                    "facets, and no pavilion at all -- the cut that "
                    "preceded the brilliant"),
    "ANTWERP_ROSE": CutEntry(
        label="Antwerp Rose",
        family="rose",
        source=(antwerp_rose, {}),
        description="A 6-fold rose with a shallower dome and 12 facets"),
    # --- deliberately BAD proportions -------------------------------
    # Each is the ideal brilliant with ONE parameter moved far enough to
    # produce a named defect.  They are in the catalogue rather than
    # hidden behind a debug flag because being able to cut a fish-eye on
    # purpose, and see why it happens, is the point of a teaching tool.
    "FISH_EYE": CutEntry(
        label="Fish-eye (defect demo)",
        family="teaching",
        source=(round_brilliant, dict(pavilion_angle=38.0, table=0.68,
                                      name="Fish-eye Brilliant")),
        description="A shallow pavilion under a large table: the girdle "
                    "reflects in the table as a grey ring (IDC 4.2.1)"),
    "WINDOWED": CutEntry(
        label="Windowed (defect demo)",
        family="teaching",
        source=(round_brilliant, dict(pavilion_angle=36.0, table=0.62,
                                      name="Windowed Brilliant")),
        description="Too shallow to hold light by total internal "
                    "reflection: you see straight through the stone"),
    "NAILHEAD": CutEntry(
        label="Nail-head (defect demo)",
        family="teaching",
        source=(round_brilliant, dict(pavilion_angle=45.0, table=0.60,
                                      name="Nail-head Brilliant")),
        description="A pavilion near 45 degrees makes the stone a "
                    "retroreflector: the centre goes dark (Sasian 2003)"),
    "EMERALD": CutEntry(
        label="Emerald Cut",
        family="step",
        source=(emerald, {}),
        description="Cut-corner rectangle with concentric rows of step "
                    "facets; the pavilion converges on a keel rather than "
                    "a point. Outline after Strickland's worked diagram"),
    "ASSCHER": CutEntry(
        label="Asscher (Square Emerald)",
        family="step",
        source=(asscher, {}),
        description="A square step cut with a deep crown and broad cut "
                    "corners; its pavilion ends in a point, not a keel"),
    "BAGUETTE": CutEntry(
        label="Baguette",
        family="step",
        source=(baguette, {}),
        description="A long, plain step cut: one row a side, square "
                    "corners -- the classic side stone"),
    "HEXAGON_STEP": CutEntry(
        label="Hexagon Step Cut",
        family="step",
        source=(polygon_step, dict(sides=6, rows=2,
                                   name="Hexagon Step Cut")),
        description="A regular-polygon step cut"),
    "OCTAGON_STEP": CutEntry(
        label="Octagon Step Cut",
        family="step",
        source=(polygon_step, dict(sides=8, rows=2,
                                   name="Octagon Step Cut")),
        description="A regular-polygon step cut"),
    # --- mixed cuts: step crown over brilliant pavilion --------------
    # Unlike the step family, a mixed cut's default proportions are
    # chosen so no plane is spent: the crowns' cut corners survive to
    # the table, and every pavilion tier binds.  They therefore stay
    # OUT of the may_drop set below -- a dropped plane in a default
    # mixed cut is a regression, not step-cut business as usual.
    "PRINCESS": CutEntry(
        label="Princess Cut",
        family="mixed",
        source=(princess, {}),
        description="Sharp-cornered square with rows of V-shaped chevron "
                    "facets running down the pavilion diagonals (Ambar & "
                    "Itzkowitz, c. 1980); more chevron rows give finer "
                    "scintillation, fewer give bolder flashes"),
    "BARION": CutEntry(
        label="Barion Cut",
        family="mixed",
        source=(barion, {}),
        description="Basil Watermeyer's 1971 mixed cut: cut-corner square, "
                    "step crown, brilliant pavilion -- with the signature "
                    "crescent 'moon' facets hung just below the girdle"),
    "RADIANT": CutEntry(
        label="Radiant Cut",
        family="mixed",
        source=(radiant, {}),
        description="Henry Grossbard's 1977 cut-corner rectangle: the "
                    "barion idea carried onto a length-to-width ratio, a "
                    "brilliant pavilion giving a square-ish stone a round "
                    "brilliant's light return"),
}

# Display order for the operator's preset menu, grouped by family.
FAMILIES = ("brilliant", "step", "mixed", "rose", "cabochon",
            "fantasy", "teaching")

# Cuts pinned to the head of their family, ahead of the alphabetical
# remainder.  The round brilliant is the cut every other one is described
# against, so it leads regardless of where its key happens to sort.
FIRST = ("ROUND_BRILLIANT",)


def get_cut(key, **params):
    """Build the named cut as a `CutDesign`.

    Extra keyword arguments are passed to a parametric cut's constructor
    and are an error for a literal one, rather than being ignored.
    """
    try:
        entry = CUTS[key]
    except KeyError:
        raise KeyError(f"no cut named {key!r}; known cuts: "
                       f"{', '.join(sorted(CUTS))}") from None
    src = entry.source
    # `CutDesign` is itself a NamedTuple, so a bare `isinstance(src, tuple)`
    # matches a literal design as readily as a parametric (ctor, kwargs)
    # pair; test for the design first.
    if not isinstance(src, CutDesign) and isinstance(src, tuple):
        ctor, defaults = src
        kw = dict(defaults)
        kw.update(params)
        return ctor(**kw)
    if params:
        raise TypeError(f"{key!r} is a fixed published design and takes no "
                        f"parameters; got {', '.join(sorted(params))}")
    return src


def accepted_params(key):
    """The parameter names the named cut's constructor will accept.

    A literal design accepts none.  The families take different
    parameters -- a rose has no table and a step cut takes a LIST of row
    angles rather than one -- so a caller driving this from a fixed set
    of controls needs to know which of them apply.
    """
    src = CUTS[key].source
    if isinstance(src, CutDesign) or not isinstance(src, tuple):
        return frozenset()
    return frozenset(inspect.signature(src[0]).parameters)


def cut_items():
    """`(key, label, description)` triples, grouped by family, for a UI."""
    out = []
    for fam in FAMILIES:
        keys = [k for k in FIRST if k in CUTS and CUTS[k].family == fam]
        keys += [k for k in sorted(CUTS)
                 if CUTS[k].family == fam and k not in FIRST]
        for key in keys:
            out.append((key, CUTS[key].label, CUTS[key].description))
    for key in sorted(CUTS):            # anything in an unlisted family
        if CUTS[key].family not in FAMILIES:
            out.append((key, CUTS[key].label, CUTS[key].description))
    return out


def _selftest():
    try:
        from . import design as _design
        from . import facets as _facets
    except ImportError:
        import design as _design
        import facets as _facets

    ok = True

    good = bool(CUTS) and all(isinstance(e, CutEntry) for e in CUTS.values())
    ok &= good
    print(f"gems.catalogue: {len(CUTS)} cut(s) registered "
          f"{'OK' if good else 'malformed entries'}")

    # every entry must build, close, and be convex -- the count is
    # measured, never asserted from the entry
    # A DROPPED plane -- one that ends up carrying no facet -- means
    # different things in different families.  In a brilliant every tier
    # is placed through a meetpoint, so a dropped plane is a mistake.  In
    # a step cut it is expected: inset a chamfered rectangle far enough
    # and the corner chamfer is consumed, which is why an emerald cut's
    # corner facets stop before the keel.  So the check is per family.
    want = {"SRB_GEMCAD": 73, "TOLKOWSKY": 73, "ROUND_BRILLIANT": 73}
    for key in sorted(CUTS):
        D = get_cut(key)
        N, d, _ = _design.planes(D)
        P = _facets.intersect_halfspaces(N, d)
        chk = _facets.polytope_checks(P, N, d)
        may_drop = CUTS[key].family in ("step",)
        good = chk["closed"] and chk["convex"] \
            and (may_drop or not P.dropped)
        ok &= good
        print(f"gems.catalogue: {key} builds closed and convex with "
              f"{len(P.faces)} facets"
              + (f", {len(P.dropped)} plane(s) spent" if P.dropped
                 else ", none cut away")
              + f" {'OK' if good else 'did not build'}")
        if key in want:
            good = len(P.faces) == want[key]
            ok &= good
            print(f"gems.catalogue: {key} has the expected "
                  f"{want[key]} facets {'OK' if good else 'count moved'}")

    # every entry survives a text round trip
    try:
        from . import asc as _asc
    except ImportError:
        import asc as _asc
    # A COMPUTED design cannot survive the file format bit-for-bit: .ASC
    # carries six decimals of angle and eight of distance, so a solved
    # tier is rounded on the way out.  What must survive is everything
    # that changes the stone -- the gear, the symmetry, the name, the
    # index lists exactly, and the geometry to the format's precision.
    worst = None
    for key in sorted(CUTS):
        D = get_cut(key)
        R = _asc.parse_asc(_asc.write_asc(D))
        same = (R.gear == D.gear and R.fold == D.fold
                and R.mirror == D.mirror and R.name == D.name
                and len(R.tiers) == len(D.tiers)
                and all(a.indices == b.indices and a.name == b.name
                        and abs(a.angle_deg - b.angle_deg) < 5e-7
                        and abs(a.distance - b.distance) < 5e-9
                        for a, b in zip(R.tiers, D.tiers)))
        if not same:
            worst = key
            break
    good = worst is None
    ok &= good
    print(f"gems.catalogue: every entry survives write/parse to the "
          f"format's precision {'OK' if good else f'{worst} did not'}")

    # cut_items is grouped by FAMILY and only then alphabetical, so it is
    # not a sorted list of keys; what it promises is every cut exactly
    # once, with families kept together.
    keys = [k for k, _, _ in cut_items()]
    fams = [CUTS[k].family for k in keys]
    grouped = fams == sorted(fams, key=FAMILIES.index)
    good = sorted(keys) == sorted(CUTS) and len(keys) == len(CUTS) and grouped
    ok &= good
    print(f"gems.catalogue: cut_items lists every cut once, grouped by "
          f"family ({', '.join(dict.fromkeys(fams))}) "
          f"{'OK' if good else 'menu and registry disagree'}")

    good = keys[0] == "ROUND_BRILLIANT"
    ok &= good
    print(f"gems.catalogue: the round brilliant heads the list, not "
          f"whatever sorts first (got {keys[0]}) "
          f"{'OK' if good else 'wrong'}")
    good = "parametric" not in CUTS["ROUND_BRILLIANT"].label.lower()
    ok &= good
    print(f"gems.catalogue: it is called '{CUTS['ROUND_BRILLIANT'].label}' "
          f"{'OK' if good else 'still carries an implementation detail'}")

    raised = False
    try:
        get_cut("SRB_GEMCAD", table=0.6)
    except TypeError:
        raised = True
    ok &= raised
    print(f"gems.catalogue: parameters on a fixed design are refused, "
          f"not ignored {'OK' if raised else 'silently dropped'}")

    raised = False
    try:
        get_cut("NO_SUCH_CUT")
    except KeyError:
        raised = True
    ok &= raised
    print(f"gems.catalogue: an unknown cut name raises "
          f"{'OK' if raised else 'returned something'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.catalogue self-test failed")
