"""Lay out the periodic table of polyhedra.

    python tools/build_periodic_table.py

Writes web/periodic-table.json: which solids are in the table, which
column each sits in, and what the columns mean.

WHY THIS IS BUILT AND NOT COMPUTED IN THE BROWSER.  The classification
lives in `symmetry.transitivity` and `symmetry.orbits`, which are in the
per-solid records and NOT in index.json. The page would have to fetch
127 records to place 127 tiles. It is a small, stable table derived from
a database that changes rarely, so it is derived once here and the gate
checks it has not gone stale.

WHICH SOLIDS, AND WHY NOT A SELECTION
-------------------------------------
A periodic table's authority is completeness: every element, no
omissions, and the gaps mean something. So this is not "the best 127
polyhedra". It is the union of classifications that are provably closed:

    5   Platonic        the regular convex solids
    13  Archimedean     the semiregular convex solids
    92  Johnson         the remaining convex solids with regular faces.
                        Listed by Johnson (1966), proved complete by
                        Zalgaller (1969)
    13  Catalan         the duals of the Archimedeans
    4   Kepler-Poinsot  the regular star polyhedra, complete by Cauchy
                        (1813)
    ---
    127

The prisms and antiprisms are left out for the reason a table cannot
have an infinite column: both families run to infinity.

WHAT THE COLUMNS ARE
--------------------
The number of FACE ORBITS -- how many classes of face the symmetry group
has, i.e. how many genuinely different kinds of face the solid has. One
orbit means every face is equivalent to every other.

This is the closest honest analogue of a chemical group. A column is a
statement about structure that every solid in it shares, it is
populated for all 127, and it orders the table by regularity: the
further right, the fewer distinct kinds of face, the more regular the
solid.

THE NOBLE GASES are the last column: the nine solids that are transitive
on vertices, edges AND faces at once -- the five Platonic solids and the
four Kepler-Poinsot stars. Nothing about them can be made more uniform;
they are closed in exactly the sense a filled electron shell is. They
are pulled out of the one-orbit column, where they would otherwise sit
among the Catalans, because being face-transitive is a weaker property
than being regular and the difference is the whole point of the column.

WHAT THIS IS NOT.  Face count is not an atomic number: it does not
identify a solid uniquely, and no property recurs at intervals of it. A
gap here predicts nothing, where Mendeleev's predicted gallium. This is
a table in the sense of a chart -- a complete set, systematically
arranged -- rather than of a periodic law.
"""
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(PROJ, "data", "polyhedra")
OUT = os.path.join(PROJ, "web", "periodic-table.json")

# The closed classifications the table is the union of.
FAMILIES = ("platonic", "archimedean", "catalan", "johnson", "kepler-poinsot")

# Point groups as periods, least to most symmetric. A solid's row within
# its column is decided by this and then by size, so reading down a
# column goes from the lopsided to the highly symmetric.
PERIODS = [
    ("Cs", "Cyclic"), ("C", "Cyclic"), ("S", "Rotoreflection"),
    ("D", "Dihedral"), ("T", "Tetrahedral"), ("O", "Octahedral"),
    ("I", "Icosahedral"),
]
PERIOD_RANK = {"Cyclic": 0, "Rotoreflection": 1, "Dihedral": 2,
               "Tetrahedral": 3, "Octahedral": 4, "Icosahedral": 5}


def period_of(schoenflies):
    s = schoenflies or "?"
    if s.startswith("I"):
        return "Icosahedral"
    if s.startswith("O"):
        return "Octahedral"
    if s.startswith("T"):
        return "Tetrahedral"
    if s.startswith("D"):
        return "Dihedral"
    if s.startswith("S"):
        return "Rotoreflection"
    return "Cyclic"


def kind_of(fams):
    for tag, label in (("platonic", "Platonic"),
                       ("archimedean", "Archimedean"),
                       ("catalan", "Catalan"),
                       ("kepler-poinsot", "Kepler-Poinsot"),
                       ("johnson", "Johnson")):
        if tag in fams:
            return label
    return "Other"

# --------------------------------------------------------------------
# THE COLUMN SCHEMES.
#
# What forms a column is a question with several defensible answers, so
# the reader picks. Each scheme below groups all 127 -- none of them is a
# filter, and none leaves a solid unplaced -- and each is derived from a
# field the records populate for every one of them.
#
# `noble` names the column, if any, that plays the part the noble gases
# do: the closed case, where nothing about the solid can be made more
# uniform in the terms the scheme is grouping by. It is a real claim, so
# it is only set where it is true. Grouping by symmetry ends with the
# icosahedral solids, which are the most symmetric but are not closed in
# that sense, and that scheme sets it to None.
# --------------------------------------------------------------------

TAIL = 8


def _capped(n):
    """Orbit counts run to 31 with a long thin tail. Cap it, or the table
    ends in a run of columns holding one solid each."""
    return str(min(n, TAIL)) if n else "?"


def _orbit_labels(word):
    return {
        "1": ("1 %s type" % word,
              "Every %s equivalent to every other." % word),
        str(TAIL): ("%d+ %s types" % (TAIL, word),
                    "The least regular solids here, gathered so the table "
                    "does not trail off into columns of one."),
    }


def _band(n):
    for hi in (6, 12, 20, 32, 60, 92):
        if n <= hi:
            return str(hi)
    return "many"


SCHEMES = [
    {
        "key": "face-types",
        "label": "Kinds of face",
        "note": "How many classes the symmetry group divides the faces "
                "into. Falls left to right, so the table runs from the "
                "lopsided to the perfectly uniform.",
        "column": lambda r: "regular" if r["regular"] else _capped(r["orbits"]),
        "order": [str(i) for i in range(TAIL, 0, -1)] + ["regular"],
        "labels": dict(_orbit_labels("face"), **{
            "regular": ("Regular", "Transitive on vertices, edges and faces "
                        "at once -- nothing about them can be made more "
                        "uniform. The five Platonic solids and the four "
                        "Kepler-Poinsot stars.")}),
        "noble": "regular",
    },
    {
        "key": "vertex-types",
        "label": "Kinds of vertex",
        "note": "The dual reading: how many classes of vertex. The "
                "Archimedean solids have exactly one, which is what "
                "'vertex-transitive' means and what defines them.",
        "column": lambda r: ("regular" if r["regular"]
                             else _capped(r["vertex_orbits"])),
        "order": [str(i) for i in range(TAIL, 0, -1)] + ["regular"],
        "labels": dict(_orbit_labels("vertex"), **{
            "regular": ("Regular", "Transitive on vertices, edges and faces "
                        "at once.")}),
        "noble": "regular",
    },
    {
        "key": "symmetry",
        "label": "Symmetry group",
        "note": "The point group, from least to most symmetric. Reading "
                "across a row compares solids of similar complexity built "
                "on different symmetries.",
        "column": lambda r: r["period"],
        "order": ["Cyclic", "Rotoreflection", "Dihedral", "Tetrahedral",
                  "Octahedral", "Icosahedral"],
        "labels": {
            "Cyclic": ("Cyclic", "A single rotation axis, with or without "
                       "mirrors."),
            "Dihedral": ("Dihedral", "A principal axis with two-fold axes "
                         "across it."),
            "Tetrahedral": ("Tetrahedral", "The symmetry of the tetrahedron."),
            "Octahedral": ("Octahedral", "The symmetry of the cube and "
                           "octahedron."),
            "Icosahedral": ("Icosahedral", "The symmetry of the dodecahedron "
                            "and icosahedron -- the largest a polyhedron can "
                            "have."),
        },
        # The icosahedral solids are the most symmetric, but "most" is not
        # "closed": nothing stops a solid having icosahedral symmetry and
        # still being irregular in every other way.
        "noble": None,
    },
    {
        "key": "classification",
        "label": "Classification",
        "note": "Which closed set each solid belongs to, ordered so the "
                "regular ones come last. Compare with the other schemes: "
                "the Johnson solids spread across every face-type column, "
                "which is the table earning its keep. No column is "
                "highlighted here because regularity is split between the "
                "last two: the Platonic solids and the Kepler-Poinsot "
                "stars are equally regular, one convex and one not.",
        "column": lambda r: r["kind"],
        "order": ["Johnson", "Catalan", "Archimedean", "Kepler-Poinsot",
                  "Platonic"],
        "labels": {
            "Johnson": ("Johnson", "The 92 convex solids with regular faces "
                        "that are neither Platonic nor Archimedean."),
            "Catalan": ("Catalan", "The duals of the Archimedean solids: "
                        "face-transitive, but not vertex-transitive."),
            "Archimedean": ("Archimedean", "Vertex-transitive, with more "
                            "than one kind of regular face."),
            "Kepler-Poinsot": ("Kepler-Poinsot", "The four regular star "
                               "polyhedra."),
            "Platonic": ("Platonic", "The five regular convex solids."),
        },
        # NOT "Platonic". The four Kepler-Poinsot solids are just as
        # fully transitive and sit in their own column, so no single
        # column here is the closed one -- and saying otherwise is a
        # claim the gate checks and rejects.
        "noble": None,
    },
    {
        "key": "faces",
        "label": "Number of faces",
        "note": "The nearest thing here to an atomic number -- an "
                "ordering, though not an identifier: many solids share a "
                "face count, and nothing recurs at intervals of it.",
        "column": lambda r: _band(r["faces"]),
        "order": ["6", "12", "20", "32", "60", "92", "many"],
        "labels": {
            "6": ("Up to 6", ""), "12": ("7-12", ""), "20": ("13-20", ""),
            "32": ("21-32", ""), "60": ("33-60", ""), "92": ("61-92", ""),
            "many": ("More than 92", ""),
        },
        "noble": None,
    },
]


def main():
    with open(os.path.join(DB, "index.json"), encoding="utf-8") as fh:
        index = json.load(fh)["entries"]

    rows = []
    for e in index:
        fams = set(e.get("families") or [])
        if not fams.intersection(FAMILIES):
            continue
        with open(os.path.join(DB, e["path"]), encoding="utf-8") as fh:
            rec = json.load(fh)
        sym = rec.get("symmetry") or {}
        comb = rec.get("combinatorics") or {}
        tr = sym.get("transitivity") or {}
        orb = sym.get("orbits") or {}
        regular = bool(tr.get("vertex") and tr.get("edge") and tr.get("face"))
        rows.append({
            "slug": e["slug"],
            "name": e["name"],
            "kind": kind_of(fams),
            "faces": e["counts"]["faces"],
            "vertices": e["counts"]["vertices"],
            "edges": e["counts"]["edges"],
            "schoenflies": sym.get("schoenflies"),
            "order": sym.get("order"),
            "period": period_of(sym.get("schoenflies")),
            "orbits": orb.get("faces"),
            "vertex_orbits": orb.get("vertices"),
            "edge_orbits": orb.get("edges"),
            "regular": regular,
            "chiral": bool(sym.get("chiral")),
            "self_dual": bool(comb.get("self_dual")),
            "convex": bool(comb.get("convex")),
            "transitivity": [bool(tr.get("vertex")), bool(tr.get("edge")),
                             bool(tr.get("face"))],
        })

    for field in ("orbits", "vertex_orbits", "edge_orbits"):
        missing = [r["slug"] for r in rows if r[field] is None]
        if missing:
            sys.stderr.write("no %s for: %s\n" % (field, ", ".join(missing)))
            return 1

    # Within a column: least symmetric first, then smallest, so a column
    # reads from lopsided to highly symmetric whichever scheme is showing.
    rows.sort(key=lambda r: (PERIOD_RANK.get(r["period"], 0), r["faces"],
                             r["name"]))

    schemes = []
    for sc in SCHEMES:
        for r in rows:
            r.setdefault("columns", {})[sc["key"]] = sc["column"](r)
        counts = {}
        for r in rows:
            k = r["columns"][sc["key"]]
            counts[k] = counts.get(k, 0) + 1
        cols = []
        for key in sc["order"]:
            if key not in counts:
                continue
            label, note = sc["labels"].get(
                key, ("%s %s" % (key, sc["label"].lower()), ""))
            cols.append({"key": key, "label": label, "note": note,
                         "count": counts[key]})
        placed = sum(c["count"] for c in cols)
        if placed != len(rows):
            sys.stderr.write(
                "scheme %r places %d of %d solids -- a column key is missing "
                "from its order\n" % (sc["key"], placed, len(rows)))
            return 1
        schemes.append({"key": sc["key"], "label": sc["label"],
                        "note": sc["note"], "noble": sc["noble"],
                        "columns": cols})

    table = {
        "generated_from": "data/polyhedra",
        "families": list(FAMILIES),
        "default_scheme": SCHEMES[0]["key"],
        "schemes": schemes,
        "solids": rows,
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(table, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    print("periodic table: %d solids, %d column schemes -> %s"
          % (len(rows), len(schemes), os.path.relpath(OUT, PROJ)))
    for s in schemes:
        print("   %-18s %s"
              % (s["label"],
                 "  ".join("%s:%d" % (c["label"], c["count"])
                           for c in s["columns"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
