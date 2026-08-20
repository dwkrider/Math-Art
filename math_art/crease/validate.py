# Flat-foldability conditions, reported and never repaired.
#
# Part of the Math Art crease engine (`math_art/crease/`).  Python and
# numpy only -- no `bpy`.
#
# THE THREE LOCAL CONDITIONS, at an interior vertex of a crease pattern:
#
#   developability   the sector angles about the vertex sum to 2*pi.
#                    A crease pattern is drawn on flat paper, so this is
#                    not a folding condition at all -- it is a statement
#                    that the drawing IS flat.  It fails when a file has
#                    been edited in 3-D, or exported from a folded state.
#
#   Maekawa          |M - V| = 2.  Equivalently the degree is even, which
#                    is why an odd-degree vertex can never fold flat.
#
#   Kawasaki-Justin  the alternating sum of sector angles vanishes:
#                    a1 - a2 + a3 - ... - a2n = 0, i.e. alternate angles
#                    sum to pi each.
#
# Maekawa and Kawasaki are NECESSARY and, for a single vertex, together
# SUFFICIENT.  They are NOT sufficient for a multi-vertex pattern: a
# pattern all of whose vertices pass can still fail globally because the
# layers cannot be ordered without self-intersection.  Deciding that is
# NP-hard (Bern and Hayes), so this module does not attempt it and says
# so; `layers.py` reads an order when a file supplies one, and anything
# more is delegated to Flat-Folder.
#
# WHY REPORT AND NOT REPAIR.  The MV assignment is the user's design.  A
# "fix" that flips a crease to satisfy Maekawa changes what the model IS.
# Every check therefore returns the offending vertices so the operator
# can select them, and changes nothing.
#
# References:
#   J. Maekawa -- the |M - V| = 2 count; first recorded in K. Kasahara,
#       T. Takahama, "Origami for the Connoisseur" (Japan Publications,
#       1987).  There is no primary Maekawa paper to cite.
#   T. Kawasaki, "On the relation between mountain-creases and
#       valley-creases of a flat origami," Proc. 1st Int. Meeting of
#       Origami Science and Technology, 1989.  Independently J. Justin.
#   T. C. Hull, "The Combinatorics of Flat Folds: a Survey," Origami^3
#       (A K Peters, 2002) -- the survey these checks follow, including
#       why local conditions do not settle the global question.
#   T. C. Hull, "Origametry: Mathematical Methods in Paper Folding"
#       (Cambridge, 2020), ch. 5-6.
#   M. Bern, B. Hayes, "The Complexity of Flat Origami," SODA 1996 --
#       MV assignment and layer ordering are both NP-hard.

import numpy as np

from .fold_io import BOUNDARY, MOUNTAIN, VALLEY
from .graph import vertex_rings

TWO_PI = 2.0 * np.pi


class Report:
    """The outcome of validating one frame.

    Truthy when nothing failed.  `issues` is a list of (kind, vertex,
    detail) so an operator can group them, and `vertices(kind)` gives the
    indices to select in the viewport.
    """

    __slots__ = ("issues", "n_interior", "n_boundary", "checked")

    KINDS = ("developability", "maekawa", "kawasaki", "odd_degree")

    def __init__(self):
        self.issues = []
        self.n_interior = 0
        self.n_boundary = 0
        self.checked = True

    def add(self, kind, vertex, detail):
        self.issues.append((kind, int(vertex), detail))

    def vertices(self, kind=None):
        return sorted({v for k, v, _ in self.issues
                       if kind is None or k == kind})

    def count(self, kind):
        return sum(1 for k, _, _ in self.issues if k == kind)

    def __bool__(self):
        return not self.issues

    def summary(self):
        if not self.issues:
            return (f"flat-foldable locally: {self.n_interior} interior "
                    f"vertices pass Maekawa, Kawasaki and developability")
        parts = []
        for kind in self.KINDS:
            n = self.count(kind)
            if n:
                parts.append(f"{n} {kind.replace('_', ' ')}")
        return (f"{len(self.vertices())} of {self.n_interior} interior "
                f"vertices fail: " + ", ".join(parts))

    def __repr__(self):
        return f"Report({self.summary()})"


def sector_angles(verts, ring, v):
    """Angles between consecutive neighbours, counter-clockwise.

    `ring` must already be in CCW order (see graph.vertex_rings).  The
    returned array has one entry per sector, so len == len(ring).
    """
    xy = np.asarray(verts, dtype=float)[:, :2]
    d = xy[np.asarray(ring, dtype=np.int64)] - xy[v]
    ang = np.arctan2(d[:, 1], d[:, 0])
    gaps = np.diff(np.concatenate([ang, ang[:1] + TWO_PI]))
    return np.mod(gaps, TWO_PI)


def kawasaki_defect(angles):
    """The alternating sum a1 - a2 + a3 - ... for an even-degree vertex.

    Zero exactly when Kawasaki-Justin holds.  Returned rather than
    thresholded so a caller can show how far off a vertex is -- a nearly
    satisfied pattern is usually a rounding artefact in the file, and a
    wildly unsatisfied one is a different design.
    """
    a = np.asarray(angles, dtype=float)
    sign = np.where(np.arange(len(a)) % 2 == 0, 1.0, -1.0)
    return float(np.dot(sign, a))


def validate(frame, angle_tol=1e-6, require_flat=True):
    """Check every interior vertex of a crease pattern.

    `angle_tol` is in radians; the default is about 0.00006 degrees,
    tight enough to catch a real design error and loose enough to ignore
    the rounding in a file written with six decimal places.

    Boundary vertices are counted but not tested: the conditions are
    statements about a full turn of paper, and a boundary vertex has
    less than one.
    """
    rep = Report()
    if frame.verts is None or frame.edges is None or not frame.n_edges:
        rep.checked = False
        return rep
    if require_flat and not frame.is_flat:
        rep.checked = False
        return rep

    n = frame.n_verts
    rings = vertex_rings(n, frame.edges, frame.verts)

    # A vertex is on the boundary if any incident edge is assigned B.
    on_boundary = np.zeros(n, dtype=bool)
    if frame.assignment is not None:
        b = frame.assignment == BOUNDARY
        if b.any():
            on_boundary[frame.edges[b].ravel()] = True

    assign_of = {}
    if frame.assignment is not None:
        for (a, c), s in zip(frame.edges, frame.assignment):
            assign_of[(int(a), int(c))] = str(s)
            assign_of[(int(c), int(a))] = str(s)

    for v in range(n):
        ring = rings[v]
        if len(ring) < 2:
            continue
        if on_boundary[v]:
            rep.n_boundary += 1
            continue
        rep.n_interior += 1

        angles = sector_angles(frame.verts, ring, v)

        # -- developability: the drawing is flat -----------------------
        total = float(angles.sum())
        if abs(total - TWO_PI) > angle_tol:
            rep.add("developability", v,
                    f"sector angles sum to {np.rad2deg(total):.4f} deg, "
                    f"not 360")
            # Kawasaki is meaningless on a non-flat vertex, so stop here.
            continue

        deg = len(ring)
        if deg % 2:
            # Odd degree cannot satisfy Maekawa, and the alternating sum
            # is not even well defined -- report it as its own kind
            # rather than as two derived failures.
            rep.add("odd_degree", v,
                    f"degree {deg} is odd, so the vertex cannot fold flat")
            continue

        # -- Maekawa: |M - V| = 2 --------------------------------------
        if frame.assignment is not None:
            m = sum(1 for u in ring
                    if assign_of.get((v, int(u))) == MOUNTAIN)
            val = sum(1 for u in ring
                      if assign_of.get((v, int(u))) == VALLEY)
            if m + val == deg and abs(m - val) != 2:
                rep.add("maekawa", v,
                        f"{m} mountain, {val} valley: |M-V| = "
                        f"{abs(m - val)}, not 2")

        # -- Kawasaki-Justin: alternating sum vanishes -----------------
        defect = kawasaki_defect(angles)
        if abs(defect) > angle_tol:
            rep.add("kawasaki", v,
                    f"alternating angle sum is {np.rad2deg(defect):.4f} deg, "
                    f"not 0")

    return rep


def _selftest():
    from .fold_io import read_fold
    import json

    def frame_of(coords, edges, assign):
        return read_fold(json.dumps({
            "vertices_coords": coords,
            "edges_vertices": edges,
            "edges_assignment": assign,
        }))[0]

    # --- the textbook flat-foldable degree-4 vertex ------------------
    # Creases at 0/90/180/270 about the origin, plus a boundary square
    # so the outer vertices are recognised as boundary.
    sq = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
    plus_e = [[0, 1], [0, 2], [0, 3], [0, 4],
              [1, 2], [2, 3], [3, 4], [4, 1]]
    plus_a = ["M", "M", "M", "V", "B", "B", "B", "B"]
    fr = frame_of(sq, plus_e, plus_a)
    rep = validate(fr)
    assert rep, rep.summary()
    assert rep.n_interior == 1 and rep.n_boundary == 4
    assert "flat-foldable locally" in rep.summary()

    # sector angles are four right angles, alternating sum zero
    rings = vertex_rings(fr.n_verts, fr.edges, fr.verts)
    ang = sector_angles(fr.verts, rings[0], 0)
    assert np.allclose(ang, np.pi / 2)
    assert abs(kawasaki_defect(ang)) < 1e-12

    # --- Maekawa violated: all four mountains -----------------------
    bad_m = frame_of(sq, plus_e, ["M", "M", "M", "M", "B", "B", "B", "B"])
    rep = validate(bad_m)
    assert not rep
    assert rep.count("maekawa") == 1
    assert rep.vertices("maekawa") == [0]
    assert "|M-V| = 4" in rep.issues[0][2]

    # --- Kawasaki violated: skew one crease -------------------------
    skew = [[0, 0], [1, 0], [0.3, 1], [-1, 0], [0, -1]]
    bad_k = frame_of(skew, plus_e, plus_a)
    rep = validate(bad_k)
    assert not rep and rep.count("kawasaki") == 1
    # ... and the defect is reported, not just a boolean
    assert "alternating angle sum" in rep.issues[0][2]

    # --- odd degree is its own diagnosis ----------------------------
    tri = [[0, 0], [1, 0], [-0.5, 0.87], [-0.5, -0.87]]
    tri_e = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [3, 1]]
    tri_a = ["M", "M", "V", "B", "B", "B"]
    rep = validate(frame_of(tri, tri_e, tri_a))
    assert rep.count("odd_degree") == 1
    # and it does not also emit a Maekawa complaint about the same vertex
    assert rep.count("maekawa") == 0

    # --- a folded (non-flat) frame is not judged --------------------
    folded = read_fold(json.dumps({
        "vertices_coords": [[0, 0, 0], [1, 0, 0], [0, 1, 0.4], [-1, 0, 0]],
        "edges_vertices": [[0, 1], [0, 2], [0, 3]],
        "edges_assignment": ["M", "V", "M"],
    }))[0]
    rep = validate(folded)
    assert not rep.checked           # reported as "not checked", not "passes"
    assert bool(rep) is True         # no issues, because nothing was tested

    # --- a Miura vertex passes --------------------------------------
    # The standard Miura interior vertex: one straight ridge through the
    # vertex (creases at 0 and pi) plus two creases placed symmetrically
    # about it, so the sectors run a, pi-a, pi-a, a.  Then
    # a + (pi-a) = pi for each alternating pair and Kawasaki holds for
    # ANY a -- which is exactly why the Miura family has a free panel
    # angle.  Note the order matters: sectors a, pi-a, a, pi-a would NOT
    # satisfy Kawasaki unless a = pi/2.
    for deg in (55.0, 70.0, 84.0):
        a = np.deg2rad(deg)
        dirs = [0.0, a, np.pi, 2 * np.pi - a]
        miura_v = [[0, 0]] + [[np.cos(t), np.sin(t)] for t in dirs]
        sectors = sector_angles(np.array(miura_v, dtype=float),
                                [1, 2, 3, 4], 0)
        assert np.allclose(np.sort(sectors),
                           np.sort([a, np.pi - a, np.pi - a, a]))
        assert abs(kawasaki_defect(sectors)) < 1e-12
        rep = validate(frame_of(miura_v, plus_e, plus_a))
        assert rep, f"panel angle {deg}: {rep.summary()}"

    print("RESULT: OK  crease.validate")


# NOTE: no __main__ guard -- tests/test_selftests.py discovers and runs
# _selftest() headlessly (see CLAUDE.md).
