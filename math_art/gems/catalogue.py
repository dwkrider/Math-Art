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

from typing import NamedTuple

try:
    from .brilliant import round_brilliant
    from .design import SRB_GEMCAD, CutDesign
except ImportError:                     # flat import outside the package
    from brilliant import round_brilliant
    from design import SRB_GEMCAD, CutDesign


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
        label="Round Brilliant (parametric)",
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
}

# Display order for the operator's preset menu, grouped by family.
FAMILIES = ("brilliant", "step", "mixed", "rose", "cabochon", "fantasy")


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


def cut_items():
    """`(key, label, description)` triples, grouped by family, for a UI."""
    out = []
    for fam in FAMILIES:
        for key in sorted(CUTS):
            if CUTS[key].family == fam:
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
    want = {"SRB_GEMCAD": 73, "TOLKOWSKY": 73, "ROUND_BRILLIANT": 73}
    for key in sorted(CUTS):
        D = get_cut(key)
        N, d, _ = _design.planes(D)
        P = _facets.intersect_halfspaces(N, d)
        chk = _facets.polytope_checks(P, N, d)
        good = chk["closed"] and chk["convex"] and not P.dropped
        ok &= good
        print(f"gems.catalogue: {key} builds closed and convex with "
              f"{len(P.faces)} facets, none cut away "
              f"{'OK' if good else 'did not build'}")
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

    good = [k for k, _, _ in cut_items()] == sorted(CUTS)
    ok &= good
    print(f"gems.catalogue: cut_items lists every cut "
          f"{'OK' if good else 'menu and registry disagree'}")

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
