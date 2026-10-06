"""Dump the Seifert engine's output as JSON, for the browser port's
parity test.

    python tools/seifert_reference.py [out.json]

With no argument it writes to stdout, which is how
`tests/web/test_seifert.mjs` runs it: the reference is then always the
engine as it stands on this checkout, so a change to the braid parser,
the state tracing or the disk/band geometry in `math_art/seifert/`
fails the site gate until `web/js/seifert-math.js` follows.

WHAT IS COMPARED, AND ONE THING THAT DELIBERATELY IS NOT. The
combinatorics are pinned exactly -- bands, feet, anchors, Euler
characteristic, boundary count, orientability, genus, the traversal
reversals -- and so is the geometry, vertex by vertex and face by face.

The DISK ORDER is handed to the port rather than compared with it.
`spatial_order` sorts the state circles by the Fiedler vector of the
state graph, and an eigenvector has no inherent sign: flipping it
reverses the stack, which reflects the whole surface. Worse, when the
state graph is symmetric the second eigenvalue is REPEATED and the
vector is not defined even up to sign -- any vector in the eigenspace
will do, and which one comes back is whatever LAPACK computed. The two
cases are told apart here by the eigenvalue gap and reported, so the
distinction is visible rather than papered over. Comparing geometry
built from two different eigensolvers would be comparing the solvers;
handing the port the engine's own order makes the mesh comparison
about the mesh.
"""
import json
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "math_art"))

from seifert.braid import Braid, torus_knot                       # noqa: E402
from seifert.build import (BAND_HALF_TURNS, SurfaceParams,        # noqa: E402
                           spanning_surface)
from seifert.states import (minimal_crosscap_state, seifert_state,  # noqa: E402
                            state_data, turnback_state)
from seifert.subdivide import catmull_clark                       # noqa: E402
from seifert.relax import RelaxParams, relax                      # noqa: E402
from seifert.fair import (cotangent_laplacian, minimal_surface,   # noqa: E402
                          smooth_boundary)

WORDS = ["AAA", "AbAb", "A5", "AABacBc", "aBaBa", "ABABAB", "A3", "AAAAA",
         "AbCb", "ABaB", "ABABABAB", "AA"]

#: A coarser mesh than the page's default, so the dump stays a sane size
#: while still exercising every stage of the construction.
PARAMS = dict(samples_per_sector=8, band_samples=8, radial_rings=1)

GEOMETRY = [("AAA", "seifert", 0), ("AAA", "turnback", 0),
            ("AbAb", "seifert", 0), ("A5", "seifert", 0),
            ("AbCb", "seifert", 0), ("aBaBa", "seifert", 0),
            ("ABABAB", "seifert", 0),
            ("AAA", "seifert", 1), ("AbAb", "seifert", 1),
            ("AAA", "seifert", 2)]


def fiedler_vector(data):
    """The vector spatial_order sorts by, so the port can be checked
    against the PROPERTY rather than against a permutation."""
    n = data.n_circles
    if n < 3:
        return None
    L = np.zeros((n, n))
    for a, _, b, _, _ in data.bands:
        if a == b:
            continue
        L[a, b] -= 1
        L[b, a] -= 1
        L[a, a] += 1
        L[b, b] += 1
    vals, vecs = np.linalg.eigh(L)
    return [float(x) for x in vecs[:, 1]]


def fiedler_gap(data):
    """How far the second eigenvalue is from the third.

    Zero means the Fiedler vector is not determined at all, so the
    stacking order is whatever the eigensolver chose.
    """
    n = data.n_circles
    if n < 3:
        # None, not inf: json.dumps writes Infinity, which is not JSON
        # and which JSON.parse refuses. Fewer than three circles means
        # there is no Fiedler vector to be ambiguous about.
        return None
    L = np.zeros((n, n))
    for a, _, b, _, _ in data.bands:
        if a == b:
            continue
        L[a, b] -= 1
        L[b, a] -= 1
        L[a, a] += 1
        L[b, b] += 1
    vals = np.linalg.eigvalsh(L)
    return float(vals[2] - vals[1])


def build(word, which, levels):
    """The pipeline state_surface runs, with the order exposed."""
    b = Braid(word)
    st = seifert_state(b) if which == "seifert" else turnback_state(b)
    data = state_data(b, st)
    order = data.spatial_order()
    height = {c: lvl for lvl, c in enumerate(order)}
    flip = data.rotation_reversals()
    feet = [0] * data.n_circles
    for circle, level in height.items():
        feet[level] = data.feet[circle]

    def place(circle, foot):
        return (data.feet[circle] - 1 - foot) if flip[circle] else foot

    wanted = (data.euler_characteristic, data.n_boundaries, data.orientable)
    params = SurfaceParams(**PARAMS)
    for extra in (0, 1):
        turns = BAND_HALF_TURNS[data.twisted] + extra
        mesh = spanning_surface(
            feet,
            [(height[a], place(a, fa), height[bb], place(bb, fb), turns * sg)
             for a, fa, bb, fb, sg in data.bands],
            params)
        info = mesh.info()
        if (info.euler_characteristic, info.n_boundaries,
                info.orientable) == wanted:
            break
    else:
        raise SystemExit("no twist parity realises %s/%s" % (word, which))
    if levels:
        mesh = catmull_clark(mesh, levels)
    return mesh, order


def main():
    out = {"params": PARAMS, "cases": [], "geometry": [], "torus": []}

    for w in WORDS:
        b = Braid(w)
        row = {
            "word": w, "strands": b.strands, "signed": b.signed_word(),
            "n_disks": b.n_disks, "n_bands": b.n_bands,
            "n_components": b.n_components,
            "chi": b.euler_characteristic, "genus": b.genus,
            "permutation": b.permutation(), "states": [],
        }
        states = [("seifert", seifert_state(b)), ("turnback", turnback_state(b))]
        if b.n_bands <= 8:
            try:
                states.append(("mincrosscap", minimal_crosscap_state(b).state))
            except ValueError:
                pass
        for name, st in states:
            d = state_data(b, st)
            row["states"].append({
                "name": name,
                "bits": [bool(x) for x in d.state],
                "n_circles": d.n_circles,
                "bands": [list(x) for x in d.bands],
                "feet": list(d.feet),
                "anchors": [[float(a), float(c)] for a, c in d.anchors],
                "chi": d.euler_characteristic,
                "n_boundaries": d.n_boundaries,
                "orientable": bool(d.orientable),
                "genus": d.genus,
                "crosscap": d.crosscap_number,
                "twisted": bool(d.twisted),
                "reversals": [bool(x) for x in d.rotation_reversals()],
                "order": [int(i) for i in d.spatial_order()],
                "fiedler_gap": fiedler_gap(d),
                "fiedler": fiedler_vector(d),
            })
        out["cases"].append(row)

    for word, which, levels in GEOMETRY:
        mesh, order = build(word, which, levels)
        info = mesh.info()
        V = np.asarray(mesh.vertices, dtype=float)
        out["geometry"].append({
            "word": word, "which": which, "levels": levels,
            "order": [int(i) for i in order],
            "n_vertices": int(len(V)), "n_faces": int(len(mesh.faces)),
            "info": {"chi": info.euler_characteristic,
                     "boundaries": info.n_boundaries,
                     "orientable": bool(info.orientable),
                     "genus": info.genus,
                     "euler_genus": info.euler_genus,
                     "n_edges": info.n_edges},
            "area": float(mesh.area()),
            "vertices": ([[round(float(x), 9) for x in v] for v in V]
                         if len(V) <= 4000 else None),
            "faces": ([list(map(int, f)) for f in mesh.faces]
                      if len(V) <= 4000 else None),
            "loop_lengths": sorted(len(l) for l in mesh.boundary_loops()),
        })

    # The finishing stages. minimal_surface is asked for the DAMPED
    # JACOBI path (mollify=False) -- the default mollified path solves
    # the raw-cotangent system with a preconditioned CG from
    # math_art.solver, which the browser has no reason to carry. The
    # Jacobi path is the one the port implements, so it is the one
    # pinned here.
    out["finish"] = []
    for word, which, relax_steps, rim_steps, levels, fair_steps, strength in [
            ("AAA", "seifert", 20, 0, 0, 0, 2.0),
            ("AAA", "seifert", 0, 0, 0, 4, 2.0),
            ("AAA", "seifert", 0, 8, 0, 0, 2.0),
            ("AAA", "seifert", 10, 4, 1, 3, 2.0),
            ("AbAb", "seifert", 15, 0, 0, 3, 2.0),
            ("AAA", "turnback", 12, 0, 0, 3, 2.0),
            ("AbCb", "seifert", 10, 0, 0, 2, 5.0)]:
        mesh, order = build(word, which, 0)
        if relax_steps:
            mesh = relax(mesh, RelaxParams(), iterations=relax_steps)
        if rim_steps:
            mesh = smooth_boundary(mesh, iterations=rim_steps)
        if levels:
            mesh = catmull_clark(mesh, levels)
        if fair_steps:
            mesh = minimal_surface(mesh, strength=strength,
                                   iterations=fair_steps, mollify=False)
        V = np.asarray(mesh.vertices, dtype=float)
        info = mesh.info()
        out["finish"].append({
            "word": word, "which": which, "order": [int(i) for i in order],
            "relax_steps": relax_steps, "rim_steps": rim_steps,
            "levels": levels, "fair_steps": fair_steps, "strength": strength,
            "n_vertices": int(len(V)),
            "area": float(mesh.area()),
            "chi": info.euler_characteristic,
            "boundaries": info.n_boundaries,
            "orientable": bool(info.orientable),
            "centroid": [float(x) for x in V.mean(axis=0)],
            "vertices": [[round(float(x), 9) for x in v] for v in V],
        })

    # The Laplacian itself, on the raw mesh: a weight that disagrees
    # would move every faired vertex a little, which an area check
    # would not localise.
    mesh, _order = build("AAA", "seifert", 0)
    lap = cotangent_laplacian(mesh)
    out["laplacian"] = {
        "word": "AAA", "n": int(lap.n),
        "n_edges": int(len(lap.src)),
        "weight_sum": float(lap.weight.sum()),
        "degree_sum": float(lap.degree.sum()),
        "degree_head": [float(x) for x in lap.degree[:24]],
    }

    for p, q in ((2, 3), (2, 5), (3, 4), (2, 4), (3, 5), (4, 3)):
        b = torus_knot(p, q)
        out["torus"].append({"p": p, "q": q, "word": b.word,
                             "strands": b.strands,
                             "n_components": b.n_components,
                             "genus": b.genus})

    text = json.dumps(out)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote %s (%.1f MB)" % (sys.argv[1], len(text) / 1e6))
    else:
        sys.stdout.write(text)


main()
