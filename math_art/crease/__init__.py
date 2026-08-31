"""crease -- the crease-pattern substrate: FOLD in, checked graph out.

Math Art does not DESIGN crease patterns.  It folds, thickens,
materialises and renders the ones you bring it, and the way you bring
them is FOLD -- the JSON interchange format that ORIPA, Oriedita,
Origami Simulator, Flat-Folder, Rabbit Ear and Tachi's tools all read
and write.  Consuming that one format is what lets this package inherit
the whole ecosystem's design capability at the cost of one reader.

This package is the mathematics and the file handling, pure Python +
numpy and importable without Blender; `math_art/fold_generator.py` is
the operator over it.

    fold_io.py    the FOLD format: frames, inheritance, M/V/F/U/B,
                  fold angles (degrees on disk, radians in memory)
    graph.py      the plane graph: face recovery by half-edge walk,
                  counter-clockwise vertex rings, triangulation
    validate.py   developability, Maekawa, Kawasaki-Justin -- reported
                  per vertex, never silently repaired
    layers.py     the stacking a file states, and nothing more; computing
                  an order is NP-hard and belongs to Flat-Folder
    patterns.py   the classical crease patterns, parametrically
    rigid.py      rigid folding: Newton in fold-angle space, then the
                  breadth-first walk that places the paper

`load_pattern` is the ordinary entry point: read a file, recover faces
if the file omitted them, and check it.

WHAT THIS PACKAGE WILL NOT DO.  Assigning mountain and valley to make a
pattern fold, and ordering the layers of a flat folded state, are both
NP-hard (Bern and Hayes 1996).  Rigid foldability is NP-hard too
(Akitaya et al. 2018).  So nothing here promises to make an arbitrary
pattern foldable: it reports what is true of the pattern you have, and
the solvers built on top fold what folds.

References:
  E. D. Demaine, J. S. Ku, R. J. Lang, "A New File Standard to Represent
      Folded Structures," 26th Fall Workshop on Computational Geometry /
      CG:YRF, 2016 -- the FOLD specification.
  T. C. Hull, "The Combinatorics of Flat Folds: a Survey," Origami^3
      (A K Peters, 2002); and "Origametry: Mathematical Methods in Paper
      Folding" (Cambridge, 2020) -- the local flat-foldability theory.
  T. Kawasaki, "On the relation between mountain-creases and
      valley-creases of a flat origami," 1st Int. Meeting of Origami
      Science and Technology, 1989.
  M. Bern, B. Hayes, "The Complexity of Flat Origami," SODA 1996.
  E. D. Demaine, J. O'Rourke, "Geometric Folding Algorithms" (Cambridge,
      2007), ch. 11-14.
"""

from . import (compliant, corrugate, fold_io, graph, layers, oripa_io,
               patterns, rigid, svg_io, validate)
from .fold_io import (ASSIGNMENTS, BOUNDARY, CREASES, FLAT, MOUNTAIN,
                      UNASSIGNED, VALLEY, FoldError, Frame, read_fold,
                      write_fold)
from .graph import GraphError, build_faces, triangulate, vertex_rings
from .oripa_io import OripaError, read_cp, read_opx
from .svg_io import SvgError, read_svg
from .layers import LayerOrder, read_layer_order
from .validate import Report, validate as check

__all__ = [
    "ASSIGNMENTS", "BOUNDARY", "CREASES", "FLAT", "MOUNTAIN",
    "UNASSIGNED", "VALLEY",
    "FoldError", "Frame", "GraphError", "LayerOrder", "Report",
    "build_faces", "check", "load_pattern", "read_fold",
    "read_layer_order", "triangulate", "vertex_rings", "write_fold",
    "patterns", "rigid", "svg_io", "read_svg", "SvgError",
    "oripa_io", "read_cp", "read_opx", "OripaError", "compliant",
    "corrugate",
]


def load_pattern(path_or_text, frame=0, recover_faces=True, validate_it=True):
    """Read a FOLD pattern and return (frame, report, order).

    `recover_faces` fills in `faces_vertices` by walking the plane graph
    when the file omits it -- which is common, because editors work in
    lines.  It is explicit rather than automatic so that a caller who
    only wants the raw file gets the raw file.

    `report` is None when validation was skipped, and otherwise a
    `validate.Report` whose `.checked` says whether the frame was even
    eligible (a folded 3-D state is not).
    """
    # DISPATCH ON THE FILE, not on a separate operator.  A user with a
    # crease pattern does not care which of the two interchange formats
    # it happens to be in, and Origami Simulator -- which reads both --
    # sets that expectation.  `stats` is None for FOLD, and the SVG
    # importer's report otherwise.
    stats = None
    low = path_or_text.lower() if isinstance(path_or_text, str) else ""
    head = path_or_text.lstrip()[:200] if isinstance(path_or_text, str) else ""
    if low.endswith(".svg") or head.startswith("<svg") or "<svg" in head:
        fr, stats = svg_io.read_svg(path_or_text)
    elif low.endswith(".opx") or "<java" in head:
        fr, stats = oripa_io.read_opx(path_or_text)
    elif low.endswith(".cp"):
        fr, stats = oripa_io.read_cp(path_or_text)
    else:
        fr, _frames = read_fold(path_or_text, frame=frame)
    if recover_faces and fr.faces is None and fr.verts is not None \
            and fr.edges is not None and fr.is_flat:
        fr.faces = build_faces(fr.verts, fr.edges)
    report = validate.validate(fr) if validate_it else None
    order = read_layer_order(fr)
    if stats is not None:
        fr.meta["import_stats"] = stats
    return fr, report, order


def _selftest():
    import json

    import numpy as np

    # The package-level story, end to end: a 2x2 grid of squares with a
    # boundary, no faces listed, folded nowhere.
    xs, ys = np.meshgrid(np.arange(3.0), np.arange(3.0))
    V = np.stack([xs.ravel(), ys.ravel()], axis=1)
    # The only interior vertex is the centre (index 4).  Give it three
    # mountains and one valley so Maekawa's |M - V| = 2 holds; every
    # edge on the rim is boundary.  Assignments are stated per edge
    # rather than derived from a formula, because a formula that quietly
    # produces 2M+2V is exactly the bug this self-test exists to catch.
    interior = {(3, 4): "M", (4, 5): "M", (1, 4): "M", (4, 7): "V"}
    E, A = [], []
    for r in range(3):
        for c in range(3):
            v = r * 3 + c
            if c < 2:
                E.append([v, v + 1])
                A.append(interior.get((v, v + 1), "B"))
            if r < 2:
                E.append([v, v + 3])
                A.append(interior.get((v, v + 3), "B"))
    assert sum(1 for a in A if a in ("M", "V")) == 4

    text = json.dumps({
        "file_spec": 1.1,
        "vertices_coords": V.tolist(),
        "edges_vertices": E,
        "edges_assignment": A,
    })

    fr, rep, order = load_pattern(text)
    # faces were absent in the file and recovered from the graph
    assert fr.n_faces == 4, fr.n_faces
    # the single interior vertex is degree 4 and every sector is a right
    # angle, so Kawasaki holds; the MV counts above give |M-V| = 2
    assert rep.checked and rep.n_interior == 1
    assert rep, rep.summary()
    assert not order          # this file states no layer relations

    # round trip preserves what we recovered
    again, _ = read_fold(write_fold(fr))
    assert again.n_faces == 4
    assert np.array_equal(again.assignment, fr.assignment)

    # a pattern that is not flat is passed through unjudged
    folded = json.dumps({
        "vertices_coords": [[0, 0, 0], [1, 0, 0], [0, 1, 1]],
        "edges_vertices": [[0, 1], [1, 2], [2, 0]],
        "edges_assignment": ["B", "B", "B"],
    })
    fr2, rep2, _ = load_pattern(folded)
    assert fr2.faces is None          # not recovered: recovery needs flat
    assert not rep2.checked

    print("RESULT: OK  crease")


# NOTE: no __main__ guard -- tests/test_selftests.py discovers and runs
# _selftest() headlessly (see CLAUDE.md).
