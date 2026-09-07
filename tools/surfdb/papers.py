"""Records and invariants taken from converted PAPERS, not from websites.

Kept separate from `algsurf.py` because the provenance is different in
kind. The website is a secondary source that cites these papers; the
papers are primary, and where the two differ the paper wins.

Both papers here were reached FROM Labs's Algebraic Surface Homepage,
which cites them by arXiv identifier only -- and both converted digests
record an attribution trap that the citing context creates:

  * arXiv:math/0010163 "Surfaces with triple points" is by ENDRASS,
    PERSSON and STEVENS. A first pass guessed van Straten from the
    citing context; he is thanked in the acknowledgements for joining
    the search, but he is not an author.
  * arXiv:math/0210440 "A note on octic hypersurfaces with many nodes"
    is by MARCO KUHNEL. A first pass guessed Endrass. Kuhnel thanks
    Oliver Labs in the introduction for pointing out references, which
    is presumably how the two names came to sit together.

Both traps are recorded in the records' provenance, because a
misattributed citation is worse than none: it looks authoritative and it
propagates.
"""

RECORDS = {

    "sextic-with-9-triple-points": {
        "name": "Sextic with 9 Triple Points",
        "family": "algebraic", "mode": "implicit",
        "blocked_by": "Not built. The paper classifies these rather than "
                      "displaying one equation, and no equation is invented "
                      "here.",
        "resume": "S. Endrass, U. Persson and J. Stevens, 'Surfaces with "
                  "triple points', arXiv:math/0010163 (2000), sections 3-4; "
                  "converted at research/papers/algebraic-surfaces/"
                  "endrass-persson-stevens-2000-surfaces-triple-points/.",
        "sources": [
            "S. Endrass, U. Persson and J. Stevens, 'Surfaces with triple "
            "points', arXiv:math/0010163v1 [math.AG] (2000), 37 pp.",
        ],
        "extra": {
            "discovered_by": "Stephan Endrass, Ulf Persson and Jan Stevens",
            "year": 2000,
            "curvature": {"condition": "none"},
            "tradition": ["classical"],
            "definition": {
                "degree": 6,
                "note": "A sextic in P^3 carrying nine ordinary triple "
                        "points. The paper's search began by looking for a "
                        "sextic with ELEVEN triple points, which would have "
                        "been very interesting; eleven turns out to be a "
                        "priori impossible, and nine is what exists.",
            },
            "embedding": {
                "quality": "singular",
                "singularities": [{"type": "ordinary triple point",
                                   "count": 9}],
            },
            "notes": {"caveats": [
                "An ordinary TRIPLE point is not a node. Triple points are "
                "not simple singularities in Du Val's sense: allowing them "
                "changes the surface's invariants and can drop its Kodaira "
                "dimension, whereas rational double points change nothing "
                "about the conditions of adjunction. That is why they are "
                "typed separately here.",
                "Eleven triple points on a sextic are proved impossible in "
                "the same paper -- a negative result worth recording "
                "alongside the positive one.",
            ]},
        },
    },
}

# Enrichment for records that already exist.
INVARIANTS = {
    "septic-with-16-triple-points": {
        "discovered_by": "Stephan Endrass, Ulf Persson and Jan Stevens",
        "year": 2000,
        "symmetry": {"kind": "point", "periodicity_rank": 0,
                     "schoenflies": "Td", "order": 24,
                     "verified_by": "curated"},
        "embedding": {
            "quality": "singular",
            "singularities": [{"type": "ordinary triple point", "count": 16}],
        },
        "definition": {
            "degree": 7,
            "note": "A one-parameter family of S_4-symmetric septics; the "
                    "general element carries sixteen ordinary triple points. "
                    "Imposing the symmetry is what reduces the problem enough "
                    "to solve (paper, Theorem 5.1).",
        },
        "provenance": {
            "sources": [
                "S. Endrass, U. Persson and J. Stevens, 'Surfaces with triple "
                "points', arXiv:math/0010163v1 [math.AG] (2000), section 5 "
                "and Theorem 5.1 -- 'A septic with 16 ordinary triple "
                "points'.",
            ],
        },
        "notes": {"caveats": [
            "ATTRIBUTION: the citing website gives only the arXiv number, and "
            "the authors are easily misread as van Straten, who is thanked in "
            "the acknowledgements for joining the search but is not an "
            "author.",
        ]},
    },
}


def records():
    return dict(RECORDS)


def ids():
    return {}


def invariants_for(slug):
    return INVARIANTS.get(slug, {})


def _selftest():
    """Shape checks; raises on failure."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    for slug, spec in RECORDS.items():
        assert slug == slug.lower() and " " not in slug, slug
        assert spec.get("blocked_by") and spec.get("resume"), slug
        assert spec.get("sources"), slug
        d = (spec.get("extra") or {}).get("definition") or {}
        assert not d.get("polynomial"), \
            "%s must not carry an untranscribed equation" % slug
        # every singularity entry must name its TYPE -- the distinction
        # between a node, a cusp and a triple point is the whole point
        for s in ((spec.get("extra") or {}).get("embedding") or {}).get(
                "singularities", []):
            assert s.get("type"), slug
            assert s.get("count"), slug

    for slug, inv in INVARIANTS.items():
        for s in (inv.get("embedding") or {}).get("singularities", []):
            assert s.get("type") and s.get("count"), slug

    # the converted papers this module rests on must actually be present
    for name in ("endrass-persson-stevens-2000-surfaces-triple-points",):
        p = os.path.join(root, "research", "papers", "algebraic-surfaces", name)
        if os.path.isdir(p):
            assert any(f.endswith(".md") for f in os.listdir(p)), \
                "%s has no converted markdown" % name

    print("RESULT: OK  (surfdb.papers, %d records, %d enrichments)"
          % (len(RECORDS), len(INVARIANTS)))
