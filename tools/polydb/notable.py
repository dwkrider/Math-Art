# The notable polyhedra: famous individual solids belonging to no family.
#
# Duerer's, Jessen's, Schonhardt's and Bilinski's solids, Escher's stellated
# rhombic dodecahedron, Brehm's 9-vertex model of Boy's surface, Bricard's and
# Steffen's flexible polyhedra, Klein's regular map, and the rest of
# `math_art.other_polyhedra_generator`.  Every other polyhedron operator in
# the add-on had records in `data/polyhedra/`; this one did not, so these
# solids were invisible to anything reading the database.
#
# Two things here are worth knowing before changing anything.
#
# References are LIFTED from the generator module's own header rather than
# restated, so a citation cannot drift from the code implementing the
# mathematics.  The table below only records which of those references
# belongs to which solid.  Where the header records no attribution the entry
# says exactly that; it does not invent one.
#
# `planarize()` is a rounding repair and NOT canonicalization.  Several of
# these solids are given as decimal coordinate tables and so sit ~1e-7 out of
# plane -- invisible in a picture, but the database certifies planarity at
# 1e-9 and would rightly reject them.

import math
import os


def references(root):
    """Pull the `# - ` reference block out of the generator's header.

    Same shape docs/scaffold_pages.py reads: an entry starts at a `-`
    marker and continuation lines are indented three spaces with no marker.
    """
    path = os.path.join(root, "math_art", "other_polyhedra_generator.py")
    with open(path, encoding="utf-8") as fh:
        head = fh.read().split("bl_info")[0]
    refs, cur = [], None
    for line in head.splitlines():
        if not line.startswith("#"):
            continue
        body = line[1:]
        if body.strip().startswith("- "):
            if cur:
                refs.append(" ".join(cur.split()))
            cur = body.strip()[2:]
        elif cur is not None and body.startswith("   ") and body.strip():
            cur += " " + body.strip()
        elif cur is not None and not body.strip():
            refs.append(" ".join(cur.split()))
            cur = None
    if cur:
        refs.append(" ".join(cur.split()))
    return refs


# Which header reference belongs to which solid, by index into references().
REFS = {
    "BREHM_BOY": [0],
    "BRICARD": [1, 2, 3, 4],
    "STEFFEN": [5, 4, 2],
    "TRIACONTAHEXA": [6],
    "ESCHER": [7],
    "ASSOCIAHEDRON": [8],
    "MAP64": [9, 15],
    "MAP46": [9, 15],
    "HEPTDODEC_A": [9],
    "HEPTDODEC_C": [9],
    "ECHIDNAHEDRON": [10],
    "SCHONHARDT": [11],
    "JESSEN": [12],
    "DURER": [13],
    "BILINSKI": [14],
    "KLEIN": [15],
}

NO_ATTRIBUTION = ("No discoverer is recorded for this solid in the source; "
                  "constructed by math_art.other_polyhedra_generator.")

# Wikipedia titles, only where an article genuinely exists under that name.
WIKIPEDIA = {
    "DURER": "Dürer graph",
    "JESSEN": "Jessen's icosahedron",
    "BILINSKI": "Bilinski dodecahedron",
    "SCHONHARDT": "Schönhardt polyhedron",
    "ASSOCIAHEDRON": "Associahedron",
    "ESCHER": "Escher's solid",
    "TETRATED": "Tetrated dodecahedron",
    "KLEIN": "Klein quartic",
    "BRICARD": "Bricard octahedron",
    "STEFFEN": "Flexible polyhedron",
}


def _plane_normal(pts):
    """Newell normal, which is correct for concave and skew faces alike."""
    n = [0.0, 0.0, 0.0]
    m = len(pts)
    for i in range(m):
        p, q = pts[i], pts[(i + 1) % m]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    ln = math.sqrt(sum(t * t for t in n))
    return None if ln < 1e-15 else [t / ln for t in n]


def planarize(V, F, iters=400, tol=1e-13):
    """Flatten faces that are planar only to the precision they were typed at.

    Each face is projected onto its own best-fit plane and every vertex moves
    to the average of the projections asking for it, repeated to convergence.
    The corrections are the same order as the input error, so the published
    coordinates stay the ones the source intended -- unlike canonicalization,
    which would also pull the edges onto a common sphere and change the solid
    into a different one.
    """
    P = [list(map(float, v)) for v in V]
    for _ in range(iters):
        acc = [[0.0, 0.0, 0.0] for _ in P]
        cnt = [0] * len(P)
        worst = 0.0
        for f in F:
            pts = [P[i] for i in f]
            n = _plane_normal(pts)
            if n is None:
                continue
            c = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
            d = sum(n[k] * c[k] for k in range(3))
            for i in f:
                dev = sum(n[k] * P[i][k] for k in range(3)) - d
                worst = max(worst, abs(dev))
                for k in range(3):
                    acc[i][k] += P[i][k] - dev * n[k]
                cnt[i] += 1
        for i in range(len(P)):
            if cnt[i]:
                P[i] = [acc[i][k] / cnt[i] for k in range(3)]
        if worst < tol:
            break
    return [tuple(p) for p in P]


def meta_for(kind, name, refs):
    """The curated half of a record; assemble() computes the rest."""
    cites = [refs[i] for i in REFS.get(kind, []) if i < len(refs)]
    return {
        "slug": None, "name": name,
        "families": ["notable"],
        "ids": {"uniform": None, "wenninger": None, "coxeter_clm": None,
                "mccooey": None, "johnson": None, "netlib": None,
                "bowers": None, "wikipedia": WIKIPEDIA.get(kind),
                "wolfram": None},
        "notation": {"schlafli": None, "wythoff": None,
                     "coxeter_diagram": None, "conway": None,
                     "vertex_configuration": [], "face_configuration": None},
        # orientable and convex are overwritten from the geometry by
        # assemble(): Brehm's model is one-sided and most of the rest are
        # non-convex, and neither fact is worth asserting twice.
        "orientable": True, "density": None, "convex": False, "dual": None,
        "orientation": ("as produced by the construction, centred at the "
                        "centroid"),
        "construction": {"generator": "math_art.other_polyhedra_generator",
                         "operator_id": "mesh.notable_polyhedron_add",
                         "conway_from": None, "wythoff_from": None},
        "coordinates": "derived: via math_art.other_polyhedra_generator",
        "sources": cites or [NO_ATTRIBUTION],
    }
