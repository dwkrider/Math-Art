"""Dump the TPMS engine's meshes as JSON, for the browser port's
parity test.

    python tools/tpms_reference.py [out.json]

With no argument it writes to stdout, which is how
`tests/web/test_tpms.mjs` runs it: the reference is then always the
engine as it stands on this checkout, so a change to the nodal
formulas or to the mesher in `math_art/minsurf/tpms.py` fails the site
gate until `web/js/tpms-math.js` follows.

WHAT IS COMPARED, AND WHY NOT INDICES. Both sides run the same
marching tetrahedra over the same grid and weld on the same lattice
edges, so they find the same crossings; but the engine compacts its
vertex list through numpy's lexicographic unique and the port compacts
in the order triangles reach it, so the two number their vertices
differently. Numbering is not geometry. The dump therefore carries the
field values (which pin the formulas exactly), and the meshes in a
canonical form: every triangle written as its three corner positions,
rounded and sorted, with the triangles themselves sorted. Two meshes
with the same canonical form are the same surface, however they were
indexed.
"""
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "math_art"))

import numpy as np                                            # noqa: E402
from minsurf.tpms import (TPMS, build_tpms, clip_to_sphere,     # noqa: E402
                          marching_tets)
sys.path.insert(0, os.path.join(PROJ, "math_art"))
from rim_curve import boundary_index_loops, boundary_loops     # noqa: E402

# Field values are compared on a grid that avoids the lattice points,
# where several of these fields sit exactly on zero.
SAMPLES = [(0.3, -1.1, 2.4), (1.0, 1.0, 1.0), (-2.2, 0.7, 0.15),
           (0.0, 0.0, 0.0), (3.0, -3.0, 1.5), (0.77, 2.71, -1.41)]

# One cell each, at a resolution small enough to dump and large enough
# to have real structure. The block cases check the tiling and its
# seam weld.
# Geometry is dumped in full for a representative handful -- the three
# classical surfaces, the two awkward ones, and the tiling cases --
# and as counts and area for the rest. The fields themselves are
# compared exactly at fixed points, so a mistyped formula cannot hide
# behind a matching triangle count.
FULL = [("P", 1, 10), ("G", 1, 10), ("D", 1, 10), ("NEOVIUS", 1, 8),
        ("FK_S", 1, 8), ("CD", 1, 8),
        ("G", 2, 6), ("G", (2, 1, 3), 6)]
COUNTS = [(k, 1, 8) for k in
          ("IWP", "FRD", "LIDINOID", "SPLITP", "FK_CS", "FK_Y", "FK_PMY",
           "FK_CPMY", "FK_CY", "CG", "GPRIME", "DPRIME", "KSURF", "CI2Y",
           "FRD2", "OCTO")]
CASES = FULL + COUNTS

# Level offsets sweep each field's companion family, so they are worth
# pinning on their own.
OFFSET_CASES = [("G", 0.4, 8), ("P", -0.6, 8), ("IWP", 0.8, 8)]


def canonical(verts, tris, nd=6):
    """A mesh as a sorted list of corner triples, each ROTATED to start
    at its smallest corner.

    Rotating rather than sorting is the point: sorting the three
    corners would lose the winding, and the winding is what says which
    way the surface faces.  Rotation removes only the arbitrary choice
    of which corner the mesher happened to write first.
    """
    V = np.asarray(verts, float).round(nd) + 0.0      # kill -0.0
    out = []
    for t in np.asarray(tris):
        c = [tuple(V[i]) for i in t]
        k = min(range(3), key=lambda i: c[i])
        out.append([list(c[(k + i) % 3]) for i in range(3)])
    out.sort()
    return out


def area(verts, tris):
    V = np.asarray(verts, float)
    T = np.asarray(tris)
    if not len(T):
        return 0.0
    a = V[T[:, 1]] - V[T[:, 0]]
    b = V[T[:, 2]] - V[T[:, 0]]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def main():
    out = {"fields": {}, "cases": [], "offsets": [], "labels": {}}

    for kind, (label, field, triply) in TPMS.items():
        out["labels"][kind] = [label, bool(triply)]
        vals = []
        for (x, y, z) in SAMPLES:
            v = field(np.array([x]), np.array([y]), np.array([z]))
            vals.append(float(np.asarray(v).ravel()[0]))
        out["fields"][kind] = vals

    for kind, cells, res in CASES:
        verts, tris = build_tpms(kind, cells, res, 2.0)
        row = {
            "kind": kind,
            "cells": list(cells) if isinstance(cells, tuple) else cells,
            "res": res,
            "nverts": int(len(verts)),
            # The engine can leave a vertex that only degenerate
            # triangles referenced -- it drops the slivers but keeps
            # the point. The port compacts those away, so what the two
            # can be held to is the number of vertices a triangle
            # actually uses.
            "nused": int(len(np.unique(tris))) if len(tris) else 0,
            "ntris": int(len(tris)),
            "area": area(verts, tris),
        }
        if (kind, cells, res) in FULL:
            row["canonical"] = canonical(verts, tris)
        out["cases"].append(row)

    for kind, off, res in OFFSET_CASES:
        verts, tris = build_tpms(kind, 1, res, 2.0, offset=off)
        out["offsets"].append({
            "kind": kind, "offset": off, "res": res,
            "nverts": int(len(verts)),
            "nused": int(len(np.unique(tris))) if len(tris) else 0,
            "ntris": int(len(tris)),
            "area": area(verts, tris),
            "canonical": canonical(verts, tris),
        })

    # Clipping to a ball, and the rim it opens: the cut edge has to
    # lie ON the sphere, and the rim walk has to find the same chains.
    out["clips"] = []
    for kind, cells, res, frac in [("G", 1, 10, 0.75), ("G", 2, 8, 0.8),
                                   ("P", 1, 10, 0.6), ("IWP", 1, 8, 0.9)]:
        verts, tris = build_tpms(kind, cells, res, 2.0)
        V = np.asarray(verts, float)
        half = 0.5 * float(np.max(V.max(0) - V.min(0)))
        r = frac * half
        cv, cf = clip_to_sphere(V, [tuple(int(i) for i in t) for t in tris], r)
        CV = np.asarray(cv, float)
        loops = boundary_loops(CV, cf)
        out["clips"].append({
            "kind": kind, "cells": cells, "res": res, "frac": frac,
            "radius": float(r),
            "nverts": int(len(CV)),
            "nfaces": int(len(cf)),
            "sizes": {str(k): int(v) for k, v in
                      zip(*np.unique([len(f) for f in cf],
                                     return_counts=True))},
            "maxr": float(np.max(np.linalg.norm(CV, axis=1))) if len(CV) else 0.0,
            "loops": [{"n": int(len(pts)), "closed": bool(cl),
                       "first": [float(x) for x in pts[0]],
                       "length": float(np.linalg.norm(
                           np.diff(np.vstack([pts, pts[:1]]) if cl else pts,
                                   axis=0), axis=1).sum())}
                      for pts, cl in loops],
        })

    text = json.dumps(out)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote %s (%.1f MB)" % (sys.argv[1], len(text) / 1e6))
    else:
        sys.stdout.write(text)


main()
