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
        tr = sym.get("transitivity") or {}
        orb = (sym.get("orbits") or {}).get("faces")
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
            "orbits": orb,
            "regular": regular,
            "convex": bool(rec.get("combinatorics", {}).get("convex")),
            "transitivity": [bool(tr.get("vertex")), bool(tr.get("edge")),
                             bool(tr.get("face"))],
        })

    missing = [r["slug"] for r in rows if r["orbits"] is None]
    if missing:
        sys.stderr.write("no face-orbit count for: %s\n" % ", ".join(missing))
        return 1

    # Columns: one per face-orbit count, with the long thin tail gathered
    # so the table does not end in a run of columns holding one solid
    # each, and the regular solids lifted into a column of their own.
    TAIL = 8

    def column_of(r):
        if r["regular"]:
            return "regular"
        return str(min(r["orbits"], TAIL))

    for r in rows:
        r["column"] = column_of(r)

    order = [str(i) for i in range(TAIL, 0, -1)] + ["regular"]
    labels = {
        "regular": ("Regular", "Transitive on vertices, edges and faces at "
                    "once -- nothing about them can be made more uniform. "
                    "The five Platonic solids and the four Kepler-Poinsot "
                    "stars."),
        "1": ("1 face type", "Every face equivalent to every other: the "
              "isohedral solids, which is what the Catalans are."),
        str(TAIL): ("%d+ face types" % TAIL,
                    "The least regular solids in the table, gathered so the "
                    "table does not trail off into columns of one."),
    }
    columns = []
    for key in order:
        members = [r for r in rows if r["column"] == key]
        if not members:
            continue
        label, note = labels.get(
            key, ("%s face types" % key,
                  "Solids whose faces fall into %s symmetry classes." % key))
        columns.append({"key": key, "label": label, "note": note,
                        "count": len(members)})

    # Within a column: least symmetric first, then smallest first, so a
    # column reads from lopsided to highly symmetric and the eye can
    # compare across at a similar height.
    rows.sort(key=lambda r: (PERIOD_RANK.get(r["period"], 0), r["faces"],
                             r["name"]))
    for r in rows:
        r.pop("column", None)

    table = {
        "generated_from": "data/polyhedra",
        "families": list(FAMILIES),
        "columns": columns,
        "solids": [dict(r, column=column_of(r)) for r in rows],
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(table, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    print("periodic table: %d solids in %d columns -> %s"
          % (len(rows), len(columns), os.path.relpath(OUT, PROJ)))
    for c in columns:
        print("   %-16s %3d" % (c["label"], c["count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
