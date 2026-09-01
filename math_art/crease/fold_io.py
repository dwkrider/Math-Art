# Reading and writing the FOLD interchange format.
#
# Part of the Math Art crease engine (`math_art/crease/`).  Python and
# numpy only -- no `bpy` -- so it imports and self-tests headlessly.
#
# WHY FOLD AND NOT AN INVENTED FORMAT.  Every editor and solver in the
# computational-origami ecosystem reads and writes FOLD: ORIPA, Oriedita,
# Origami Simulator, Flat-Folder, Rabbit Ear, Tachi's tools.  Consuming
# it is what lets this add-on inherit all of their design capability at
# the cost of one importer, which is the whole architectural bet: Math
# Art does not design crease patterns, it folds, thickens, materialises
# and renders the ones you bring it.
#
# WHAT THE FORMAT IS.  A JSON object.  Geometry lives in flat parallel
# arrays under an `A_B` naming convention -- `edges_vertices` is a
# property of `edges`, so element i of every `edges_*` array describes
# edge i.  The pieces this module cares about:
#
#     vertices_coords    per vertex, 1..n coordinates (2-D OR 3-D)
#     edges_vertices     per edge, the pair of vertex indices
#     edges_assignment   per edge, one of M V F U B
#     edges_foldAngle    per edge, signed degrees
#     faces_vertices     per face, CCW vertex indices
#     faceOrders         [i, j, k] layer relations between coplanar faces
#
# NOTHING IS MANDATORY, not even `vertices_coords`.  A frame carrying
# coordinates is an embedded folded structure; one without is abstract.
# This module therefore returns what is present and never invents the
# rest -- recovering `faces_vertices` from the edge graph is `graph.py`'s
# job, and it is called explicitly, not silently.
#
# FRAMES AND INHERITANCE.  A file may carry extra frames in
# `file_frames`.  A frame with `frame_inherit: true` and `frame_parent: p`
# takes everything from frame p and overrides only the keys it states.
# That is how a folding MOTION is stored: one flat key frame plus a
# sequence of partial frames.  `read_fold` resolves inheritance eagerly,
# so every returned frame is self-contained.
#
# ANGLE SIGN CONVENTION.  FOLD writes fold angles in DEGREES, positive
# for valley and negative for mountain, 0 for flat, +/-180 for a fully
# folded crease.  This module converts to radians on read and back to
# degrees on write, so everything above `fold_io` works in radians.
#
# References:
#   E. D. Demaine, J. S. Ku, R. J. Lang, "A New File Standard to
#       Represent Folded Structures," abstract, 26th Fall Workshop on
#       Computational Geometry / CG:YRF, 2016 -- the FOLD specification.
#   E. D. Demaine, J. O'Rourke, "Geometric Folding Algorithms"
#       (Cambridge, 2007) -- the layer-order relation `faceOrders`
#       encodes.

import json

import numpy as np

# The five edge assignments FOLD defines.
MOUNTAIN = "M"
VALLEY = "V"
FLAT = "F"
UNASSIGNED = "U"
BOUNDARY = "B"
ASSIGNMENTS = (MOUNTAIN, VALLEY, FLAT, UNASSIGNED, BOUNDARY)

# Assignments that are creases proper -- the ones that carry a fold
# angle and that Maekawa counts.  F is a crease that happens to be flat,
# U is undecided, B is the sheet's edge and folds nothing.
CREASES = (MOUNTAIN, VALLEY)

_GEOMETRY_KEYS = (
    "vertices_coords", "vertices_vertices", "vertices_faces",
    "edges_vertices", "edges_assignment", "edges_foldAngle",
    "edges_length",
    "faces_vertices", "faces_edges", "faceOrders",
)


class FoldError(ValueError):
    """A FOLD file that cannot be understood.

    Raised with a message naming the offending key, because the common
    case is a hand-edited or half-exported file and the user needs to
    know which field to look at -- not a traceback from numpy.
    """


class Frame:
    """One self-contained frame of a FOLD file.

    Inheritance is already resolved: whatever a frame inherited from its
    parent is present here directly.  Absent properties are None rather
    than an empty array, because in FOLD "no faces listed" and "zero
    faces" are different statements.
    """

    __slots__ = ("verts", "edges", "assignment", "fold_angle",
                 "faces", "face_orders", "meta", "dim")

    def __init__(self, verts=None, edges=None, assignment=None,
                 fold_angle=None, faces=None, face_orders=None,
                 meta=None):
        self.verts = verts                  # (n, dim) float or None
        self.edges = edges                  # (m, 2) int or None
        self.assignment = assignment        # (m,) '<U1' or None
        self.fold_angle = fold_angle        # (m,) float RADIANS or None
        self.faces = faces                  # list[list[int]] or None
        self.face_orders = face_orders      # list[(i, j, k)] or None
        self.meta = meta or {}
        self.dim = 0 if verts is None else int(verts.shape[1])

    # -- convenience ------------------------------------------------
    @property
    def n_verts(self):
        if self.verts is not None:
            return int(len(self.verts))
        if self.edges is not None and len(self.edges):
            return int(self.edges.max()) + 1
        return 0

    @property
    def n_edges(self):
        return 0 if self.edges is None else int(len(self.edges))

    @property
    def n_faces(self):
        return 0 if self.faces is None else len(self.faces)

    @property
    def is_flat(self):
        """True when the frame is a crease pattern, not a folded state.

        A 2-D frame is flat by construction.  A 3-D frame is flat if
        every z is (numerically) zero, which is how many exporters write
        an unfolded sheet.
        """
        if self.verts is None or not len(self.verts):
            return False
        if self.dim <= 2:
            return True
        return bool(np.allclose(self.verts[:, 2:], 0.0, atol=1e-9))

    def crease_mask(self):
        """Boolean mask of edges that are M or V."""
        if self.assignment is None:
            return np.zeros(self.n_edges, dtype=bool)
        return np.isin(self.assignment, np.array(CREASES))

    def copy(self):
        return Frame(
            None if self.verts is None else self.verts.copy(),
            None if self.edges is None else self.edges.copy(),
            None if self.assignment is None else self.assignment.copy(),
            None if self.fold_angle is None else self.fold_angle.copy(),
            None if self.faces is None else [list(f) for f in self.faces],
            None if self.face_orders is None else list(self.face_orders),
            dict(self.meta),
        )

    def __repr__(self):
        return (f"Frame({self.n_verts}v {self.n_edges}e {self.n_faces}f "
                f"dim={self.dim}{' flat' if self.is_flat else ''})")


# ----------------------------------------------------------------- read

def _as_int_pairs(raw, key):
    try:
        arr = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise FoldError(f"{key}: not an array of index pairs ({exc})")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise FoldError(f"{key}: expected pairs, got shape {arr.shape}")
    return arr


def _as_coords(raw, key):
    # Rows may be ragged in a malformed file; numpy would build an
    # object array and fail much later with something unreadable.
    try:
        widths = {len(r) for r in raw}
    except TypeError as exc:
        raise FoldError(f"{key}: not a list of coordinate lists ({exc})")
    if not widths:
        return np.zeros((0, 2), dtype=float)
    if len(widths) != 1:
        raise FoldError(
            f"{key}: mixed coordinate dimensions {sorted(widths)}; "
            "every vertex must have the same number of coordinates")
    dim = widths.pop()
    if dim < 1 or dim > 3:
        raise FoldError(f"{key}: {dim}-D coordinates are not supported")
    return np.asarray(raw, dtype=float)


def _resolve_inherit(frames, index, _seen=None):
    """Return frame `index` with its inherited keys filled in.

    FOLD lets a frame inherit from a parent and override selected keys.
    Chains are legal; cycles are not, and a cycle in a hand-edited file
    would otherwise hang here, so it is detected explicitly.
    """
    _seen = _seen or set()
    if index in _seen:
        raise FoldError(f"frame_parent cycle through frame {index}")
    raw = dict(frames[index])
    if not raw.get("frame_inherit"):
        return raw
    parent = raw.get("frame_parent")
    if parent is None:
        raise FoldError(f"frame {index}: frame_inherit set with no frame_parent")
    if not (0 <= parent < len(frames)):
        raise FoldError(f"frame {index}: frame_parent {parent} out of range")
    merged = _resolve_inherit(frames, parent, _seen | {index})
    merged.update(raw)
    return merged


def _frame_from_dict(raw, file_meta):
    verts = edges = assign = angle = None
    faces = orders = None

    if "vertices_coords" in raw:
        verts = _as_coords(raw["vertices_coords"], "vertices_coords")

    if "edges_vertices" in raw:
        edges = _as_int_pairs(raw["edges_vertices"], "edges_vertices")
        if verts is not None and len(edges) and edges.max() >= len(verts):
            raise FoldError(
                f"edges_vertices: index {int(edges.max())} but only "
                f"{len(verts)} vertices")
        if len(edges) and (edges[:, 0] == edges[:, 1]).any():
            raise FoldError("edges_vertices: an edge joins a vertex to itself")

    if "edges_assignment" in raw:
        assign = np.asarray(
            [str(a).strip().upper()[:1] if a else UNASSIGNED
             for a in raw["edges_assignment"]], dtype="<U1")
        bad = sorted(set(assign.tolist()) - set(ASSIGNMENTS))
        if bad:
            raise FoldError(
                f"edges_assignment: unknown assignment(s) {bad}; "
                f"FOLD defines {' '.join(ASSIGNMENTS)}")
        if edges is not None and len(assign) != len(edges):
            raise FoldError(
                f"edges_assignment: {len(assign)} entries for "
                f"{len(edges)} edges")

    if "edges_foldAngle" in raw:
        deg = np.asarray(
            [np.nan if a is None else float(a)
             for a in raw["edges_foldAngle"]], dtype=float)
        if edges is not None and len(deg) != len(edges):
            raise FoldError(
                f"edges_foldAngle: {len(deg)} entries for {len(edges)} edges")
        angle = np.deg2rad(deg)

    if "faces_vertices" in raw:
        faces = [list(int(i) for i in f) for f in raw["faces_vertices"]]
        if verts is not None:
            for fi, f in enumerate(faces):
                if f and max(f) >= len(verts):
                    raise FoldError(
                        f"faces_vertices[{fi}]: index {max(f)} but only "
                        f"{len(verts)} vertices")

    if "faceOrders" in raw:
        orders = []
        for t in raw["faceOrders"]:
            if len(t) != 3:
                raise FoldError("faceOrders: entries must be [i, j, k] triples")
            i, j, k = t
            orders.append((int(i), int(j), None if k is None else int(k)))

    meta = dict(file_meta)
    meta.update({k: v for k, v in raw.items()
                 if k not in _GEOMETRY_KEYS})
    return Frame(verts, edges, assign, angle, faces, orders, meta)


def read_fold(path_or_text, frame=0):
    """Read a FOLD file and return (frame, all_frames).

    `path_or_text` is a filesystem path or the JSON text itself.
    `frame` selects which frame to return first; every frame in the
    returned list has already had `frame_inherit` resolved, so callers
    never have to walk parents.
    """
    text = path_or_text
    looks_like_json = isinstance(text, str) and text.lstrip()[:1] == "{"
    if not looks_like_json:
        # A string that is neither JSON nor a readable file is the common
        # shape of a mistake here, and an OSError traceback names the
        # wrong problem -- so translate it.
        try:
            with open(path_or_text, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            raise FoldError(
                f"not valid JSON, and not a readable file: {exc}")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FoldError(f"not valid JSON: {exc}")
    if not isinstance(doc, dict):
        raise FoldError("top level of a FOLD file must be an object")

    file_meta = {k: v for k, v in doc.items()
                 if k.startswith("file_") and k != "file_frames"}

    # The top-level object is itself frame 0; file_frames holds the rest.
    raws = [doc] + list(doc.get("file_frames", []))
    resolved = [_resolve_inherit(raws, i) for i in range(len(raws))]
    frames = [_frame_from_dict(r, file_meta) for r in resolved]

    if not (0 <= frame < len(frames)):
        raise FoldError(f"frame {frame} requested, file has {len(frames)}")
    return frames[frame], frames


# ---------------------------------------------------------------- write

def write_fold(frames, path=None, creator="Math Art", author="",
               classes=("creasePattern",)):
    """Serialise one Frame, or a sequence of them, to FOLD.

    A sequence longer than one is written as frame 0 plus `file_frames`,
    which is how FOLD stores a folding motion.  Returns the JSON text,
    and also writes it to `path` when given.
    """
    if isinstance(frames, Frame):
        frames = [frames]
    frames = list(frames)
    if not frames:
        raise FoldError("nothing to write: no frames")

    def body(fr):
        out = {}
        if fr.verts is not None:
            out["vertices_coords"] = [[float(c) for c in v] for v in fr.verts]
        if fr.edges is not None:
            out["edges_vertices"] = [[int(a), int(b)] for a, b in fr.edges]
        if fr.assignment is not None:
            out["edges_assignment"] = [str(a) for a in fr.assignment]
        if fr.fold_angle is not None:
            deg = np.rad2deg(fr.fold_angle)
            out["edges_foldAngle"] = [None if np.isnan(d) else float(d)
                                      for d in deg]
        if fr.faces is not None:
            out["faces_vertices"] = [[int(i) for i in f] for f in fr.faces]
        if fr.face_orders:
            out["faceOrders"] = [[int(i), int(j),
                                  None if k is None else int(k)]
                                 for i, j, k in fr.face_orders]
        return out

    doc = {
        "file_spec": 1.1,
        "file_creator": creator,
        "file_classes": list(classes),
    }
    if author:
        doc["file_author"] = author
    doc.update(body(frames[0]))
    for key in ("frame_title", "frame_classes", "frame_attributes"):
        if key in frames[0].meta:
            doc[key] = frames[0].meta[key]
    if len(frames) > 1:
        doc["file_frames"] = [body(fr) for fr in frames[1:]]

    text = json.dumps(doc, indent=1)
    if path is not None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def _selftest():
    # --- a single degree-4 flat-foldable vertex ---------------------
    # Four creases at 0, 90, 180, 270 degrees: Kawasaki holds
    # (90 + 90 = 180) and 3 mountains against 1 valley gives |M-V| = 2.
    doc = {
        "file_spec": 1.1,
        "file_creator": "selftest",
        "frame_title": "plus vertex",
        "vertices_coords": [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
        "edges_vertices": [[0, 1], [0, 2], [0, 3], [0, 4]],
        "edges_assignment": ["M", "M", "M", "V"],
        "edges_foldAngle": [-180, -180, -180, 180],
    }
    fr, frames = read_fold(json.dumps(doc))
    assert len(frames) == 1
    assert fr.n_verts == 5 and fr.n_edges == 4
    assert fr.dim == 2 and fr.is_flat
    assert fr.meta["frame_title"] == "plus vertex"
    # degrees on disk, radians in memory
    assert np.isclose(fr.fold_angle[0], -np.pi)
    assert np.isclose(fr.fold_angle[3], +np.pi)
    assert fr.crease_mask().sum() == 4

    # --- round trip -------------------------------------------------
    again, _ = read_fold(write_fold(fr))
    assert np.allclose(again.verts, fr.verts)
    assert np.array_equal(again.edges, fr.edges)
    assert np.array_equal(again.assignment, fr.assignment)
    assert np.allclose(again.fold_angle, fr.fold_angle)

    # --- frame inheritance ------------------------------------------
    # Frame 1 inherits the sheet and overrides only the fold angles:
    # this is the shape a folding motion takes on disk.
    doc2 = dict(doc)
    doc2["file_frames"] = [{
        "frame_inherit": True,
        "frame_parent": 0,
        "edges_foldAngle": [-90, -90, -90, 90],
    }]
    f0, all_frames = read_fold(json.dumps(doc2), frame=0)
    assert len(all_frames) == 2
    f1 = all_frames[1]
    # inherited
    assert np.array_equal(f1.edges, f0.edges)
    assert np.allclose(f1.verts, f0.verts)
    # overridden
    assert np.isclose(f1.fold_angle[0], -np.pi / 2)
    assert np.isclose(f0.fold_angle[0], -np.pi)

    # a motion round-trips as frame 0 + file_frames
    two, _ = read_fold(write_fold([f0, f1]))
    assert len(read_fold(write_fold([f0, f1]))[1]) == 2

    # --- 3-D frames, and is_flat --------------------------------------
    folded = {
        "vertices_coords": [[0, 0, 0], [1, 0, 0], [0, 1, 0.5]],
        "edges_vertices": [[0, 1], [1, 2], [2, 0]],
        "edges_assignment": ["B", "B", "B"],
    }
    f3, _ = read_fold(json.dumps(folded))
    assert f3.dim == 3 and not f3.is_flat
    assert f3.crease_mask().sum() == 0        # boundary is not a crease

    flat3 = dict(folded)
    flat3["vertices_coords"] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    assert read_fold(json.dumps(flat3))[0].is_flat

    # --- absent properties stay absent --------------------------------
    bare, _ = read_fold('{"edges_vertices": [[0, 1]]}')
    assert bare.verts is None            # unembedded, not "zero vertices"
    assert bare.faces is None
    assert bare.n_verts == 2             # inferred from the edge graph

    # --- malformed input fails with a useful message ------------------
    for text, needle in (
        ('{"edges_vertices": [[0, 1]], "edges_assignment": ["X"]}',
         "edges_assignment"),
        ('{"vertices_coords": [[0, 0], [0, 0, 0]]}', "mixed coordinate"),
        ('{"vertices_coords": [[0, 0]], "edges_vertices": [[0, 5]]}',
         "edges_vertices"),
        ('{"edges_vertices": [[2, 2]]}', "vertex to itself"),
        ('not json at all', "valid JSON"),
        ('{"file_frames": [{"frame_inherit": true, "frame_parent": 1}]}',
         "cycle"),
    ):
        try:
            read_fold(text)
        except FoldError as exc:
            assert needle in str(exc), f"wrong message for {text!r}: {exc}"
        else:
            raise AssertionError(f"should have raised: {text!r}")

    print("RESULT: OK  crease.fold_io")


# NOTE: no __main__ guard -- tests/test_selftests.py discovers and runs
# _selftest() headlessly (see CLAUDE.md).
