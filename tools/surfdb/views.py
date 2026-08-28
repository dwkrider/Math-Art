# Generated fields -- the "computed views" of the surface taxonomy.
#
# The taxonomy has FIVE curated facets (definition.mode, curvature.condition,
# symmetry.periodicity_rank, embedding.quality, tradition).  Everything
# else that looks like a facet is derived from those plus the topology
# block, and is generated here rather than hand-written:
#
#   families[]                 the denormalised tag array
#   curvature.also_satisfies   the weaker conditions implied by the stronger
#   topology.class             the classification-theorem label
#   fabrication                all but self_supporting
#
# tools/surfdb_validate.py regenerates each of these and compares, so a
# hand-edited generated field is an ERROR.  An earlier draft of the design
# let families[] be a free tag space; three records in, it had already
# accreted tags from no declared vocabulary ("tpms", "chiral", "closed")
# and would have drifted.

# The precedence chain for curvature.condition.  These values are NOT
# disjoint -- every minimal, CMC, flat and constant-K surface is also a
# Weingarten surface -- so a record stores the STRONGEST condition and
# `also_satisfies` carries the rest.  Without this the facet is simply
# ill-defined.
CURVATURE_PRECEDENCE = [
    "minimal",
    "cmc",
    "cmc1-bryant",
    "k-const-negative",
    "k-const-positive",
    "flat",
    "willmore",
    "constrained-willmore",
    "weingarten",
    "none",
]

# Which weaker conditions each condition actually implies.  Not simply
# "everything below it in the list": a minimal surface is Weingarten but
# is not flat, and CMC does not imply constant K.
IMPLIES = {
    "minimal": ["weingarten"],
    "cmc": ["weingarten"],
    "cmc1-bryant": ["cmc", "weingarten"],
    "k-const-negative": ["weingarten"],
    "k-const-positive": ["weingarten"],
    "flat": ["weingarten"],
    "willmore": [],
    "constrained-willmore": [],
    "weingarten": [],
    "none": [],
}

PERIODICITY_TAG = {
    0: "aperiodic",
    1: "singly-periodic",
    2: "doubly-periodic",
    3: "triply-periodic",
}


def also_satisfies(condition):
    """The weaker curvature conditions implied by `condition`."""
    return list(IMPLIES.get(condition, []))


def topology_class(topo):
    """The classification-theorem label, from the topology block.

    Returns None when the block carries too little to decide -- an
    honest null rather than a guess.
    """
    if not topo:
        return None
    compact = topo.get("compact")
    complete = topo.get("complete")
    boundary = topo.get("boundary_components")
    orientable = topo.get("orientable")
    ftc = topo.get("finite_total_curvature")

    if isinstance(boundary, int) and boundary > 0:
        return "compact-with-boundary" if compact else "with-boundary"
    if compact:
        if orientable is True:
            return "closed-orientable"
        if orientable is False:
            return "closed-nonorientable"
        return None
    if complete:
        if ftc is True:
            return "complete-finite-total-curvature"
        if ftc is False:
            return "complete-infinite-total-curvature"
        return None
    if compact is False:
        return "noncompact"
    return None


def fabrication(record):
    """The computed fabrication view.

    `self_supporting` is the one field nothing can check, so it is a
    curated override and is passed through untouched.
    """
    prior = (record.get("fabrication") or {})
    curv = (record.get("curvature") or {}).get("condition")
    emb = (record.get("embedding") or {}).get("quality")
    topo = record.get("topology") or {}

    boundary = topo.get("boundary_components")
    one_sided = topo.get("one_sided")
    compact = topo.get("compact")

    developable = (curv == "flat")
    printable = (emb == "embedded" and compact is True)
    thicken = bool(one_sided is True
                   or (isinstance(boundary, int) and boundary > 0)
                   or compact is False)

    out = {
        "developable": developable,
        "printable_watertight": printable,
        "requires_thickening": thicken,
        # a surface you can slice into stacked planar sections: needs to be
        # bounded, and self-intersection makes the section ambiguous
        "sliceable": bool(compact is True and emb in ("embedded", "immersed")),
    }
    if "self_supporting" in prior:
        out["self_supporting"] = prior["self_supporting"]
    return out


def families(record):
    """The generated `families` tag array.

    One tag per curated facet, plus the derived topology and periodicity
    labels, plus the primary family.  Sorted, so the comparison the
    validator makes is order-independent.
    """
    tags = set()

    pf = record.get("primary_family")
    if pf:
        tags.add(pf)

    dfn = record.get("definition") or {}
    mode = dfn.get("mode")
    if mode:
        tags.add("def-" + mode)

    curv = (record.get("curvature") or {}).get("condition")
    if curv and curv != "none":
        tags.add(curv)

    sym = record.get("symmetry") or {}
    rank = sym.get("periodicity_rank")
    if isinstance(rank, int):
        tags.add(PERIODICITY_TAG[rank])
    if sym.get("chiral") is True:
        tags.add("chiral")

    emb = (record.get("embedding") or {}).get("quality")
    if isinstance(emb, str) and emb != "varies":
        tags.add(emb)
    if (record.get("embedding") or {}).get("is_record"):
        tags.add("record-holder")

    topo = record.get("topology") or {}
    cls = topo.get("class") or topology_class(topo)
    if cls:
        tags.add(cls)
    if topo.get("orientable") is False:
        tags.add("non-orientable")
    if topo.get("compact") is True:
        tags.add("closed")

    for t in record.get("tradition") or []:
        tags.add("tradition-" + t)

    # an approximation exists alongside the exact definition
    for d in [dfn] + list(record.get("alternate_definitions") or []):
        if d.get("fidelity") == "approximation":
            tags.add("has-approximation")

    if any(c.get("implemented") for c in record.get("construction") or []):
        tags.add("implemented")
    else:
        tags.add("not-implemented")

    return sorted(tags)


def apply_all(record):
    """Regenerate every computed field on `record`, in place. Returns it."""
    curv = record.setdefault("curvature", {"condition": "none"})
    curv["also_satisfies"] = also_satisfies(curv.get("condition"))

    topo = record.get("topology")
    if topo is not None:
        cls = topology_class(topo)
        if cls is not None:
            topo["class"] = cls
        elif "class" in topo:
            topo["class"] = None

    fab = fabrication(record)
    if fab:
        record["fabrication"] = fab

    record["families"] = families(record)
    return record


def diff_generated(record):
    """What `apply_all` would change. Empty dict means the record is clean.

    This is the validator's check that nobody hand-edited a generated
    field.
    """
    import copy
    before = {
        "families": list(record.get("families") or []),
        "also_satisfies": list((record.get("curvature") or {}).get("also_satisfies") or []),
        "class": (record.get("topology") or {}).get("class"),
        "fabrication": copy.deepcopy(record.get("fabrication")),
    }
    after_rec = apply_all(copy.deepcopy(record))
    after = {
        "families": list(after_rec.get("families") or []),
        "also_satisfies": list((after_rec.get("curvature") or {}).get("also_satisfies") or []),
        "class": (after_rec.get("topology") or {}).get("class"),
        "fabrication": after_rec.get("fabrication"),
    }
    out = {}
    for k in before:
        if before[k] != after[k]:
            out[k] = {"stored": before[k], "regenerated": after[k]}
    return out


def _selftest():
    """Numeric self-test; raises on failure."""
    assert also_satisfies("minimal") == ["weingarten"]
    assert also_satisfies("cmc1-bryant") == ["cmc", "weingarten"]
    assert also_satisfies("none") == []
    # a minimal surface is Weingarten but is NOT flat -- the chain is an
    # implication table, not a prefix of the precedence list
    assert "flat" not in also_satisfies("minimal")

    assert topology_class({"compact": True, "orientable": True}) == "closed-orientable"
    assert topology_class({"compact": True, "orientable": False}) == "closed-nonorientable"
    assert topology_class({"compact": False, "complete": True,
                           "finite_total_curvature": True}) == "complete-finite-total-curvature"
    assert topology_class({"boundary_components": 2, "compact": True}) == "compact-with-boundary"
    assert topology_class({}) is None, "too little information must give null, not a guess"

    cat = {
        "primary_family": "minimal",
        "definition": {"mode": "parametric", "fidelity": "exact"},
        "curvature": {"condition": "minimal"},
        "symmetry": {"kind": "continuous", "periodicity_rank": 0},
        "embedding": {"quality": "embedded"},
        "topology": {"compact": False, "complete": True,
                     "finite_total_curvature": True, "orientable": True},
        "tradition": ["classical"],
        "construction": [{"implemented": True}],
    }
    apply_all(cat)
    fams = cat["families"]
    for want in ("minimal", "def-parametric", "aperiodic", "embedded",
                 "complete-finite-total-curvature", "tradition-classical",
                 "implemented"):
        assert want in fams, "missing generated tag %r in %r" % (want, fams)
    assert cat["curvature"]["also_satisfies"] == ["weingarten"]
    assert cat["fabrication"]["developable"] is False
    assert cat["fabrication"]["requires_thickening"] is True, \
        "a complete non-compact surface has no volume to print"

    # regenerating an already-clean record must be a no-op
    assert diff_generated(cat) == {}

    # and a hand-edited generated field must be caught
    dirty = dict(cat)
    dirty["families"] = ["nonsense"]
    assert "families" in diff_generated(dirty)

    gyroid = {
        "primary_family": "minimal-periodic",
        "definition": {"mode": "weierstrass", "fidelity": "exact"},
        "alternate_definitions": [{"mode": "nodal", "fidelity": "approximation",
                                   "approximates": 0}],
        "curvature": {"condition": "minimal"},
        "symmetry": {"kind": "space", "periodicity_rank": 3, "chiral": True},
        "embedding": {"quality": "embedded"},
        "topology": {"compact": False, "complete": True,
                     "finite_total_curvature": False, "orientable": True},
        "construction": [{"implemented": True}],
    }
    apply_all(gyroid)
    assert "triply-periodic" in gyroid["families"]
    assert "chiral" in gyroid["families"]
    assert "has-approximation" in gyroid["families"]

    print("RESULT: OK  (surfdb.views)")
