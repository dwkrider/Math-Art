# Layer order, as far as a file states it -- and no further.
#
# Part of the Math Art crease engine (`math_art/crease/`).  Python and
# numpy only -- no `bpy`.
#
# DELIBERATELY NOT A SOLVER.  Deciding a valid layer ordering for a flat
# folded state is NP-hard: Bern and Hayes proved that finding an overlap
# order for the flaps is NP-hard even when a valid mountain/valley
# assignment is handed to you.  Flat-Folder already solves it well, is
# MIT-licensed, and speaks FOLD, so the sane division of labour is:
#
#     this module      reads `faceOrders` when a file supplies them,
#                      turns them into a usable partial order, and
#                      reports whether that order is consistent
#     Flat-Folder      computes an order when a file does not have one
#
# The round trip is the product: export the crease pattern, run it
# through Flat-Folder, import the result with its `faceOrders` filled in.
#
# WHAT `faceOrders` MEANS.  Each entry is a triple [i, j, k] describing
# two faces that overlap in the folded state:
#
#     k = +1   face j lies ABOVE face i, relative to i's normal
#     k = -1   face j lies BELOW face i
#     k =  0   neither (the pair is unordered)
#     k = null unknown
#
# The relation is relative to face i's ORIENTATION, so it flips with the
# face normal -- which is why the spec stores both [i, j, k] and often
# the redundant [j, i, -k].  This module normalises to a single direction
# so a caller never has to reason about which way round a pair was
# written.
#
# TRANSITIVITY.  The spec notes that not every pair need be listed: a
# reader should infer a total order where the stated pairs imply one.
# So the stated relations are a DAG, and stacking is a topological sort
# of it.  A cycle means the file describes paper passing through itself,
# which is a real and useful thing to detect.
#
# References:
#   E. D. Demaine, J. S. Ku, R. J. Lang, "A New File Standard to
#       Represent Folded Structures," 2016 -- the `faceOrders` semantics.
#   M. Bern, B. Hayes, "The Complexity of Flat Origami," SODA 1996 --
#       why this module does not compute an order.
#   J. S. Ku, Flat-Folder -- the constraint-satisfaction solver this
#       module defers to (taco-taco, taco-tortilla, tortilla-tortilla
#       and transitivity constraints).

import numpy as np


class LayerOrder:
    """The stacking relations a frame states, normalised.

    `above[(i, j)] is True` means face j is above face i.  Pairs the
    file left at k = 0 or null are absent: unordered is not the same as
    "either way round", and pretending otherwise is how a renderer ends
    up drawing a stack that the file never claimed.
    """

    __slots__ = ("pairs", "n_faces", "unordered", "unknown")

    def __init__(self, n_faces):
        self.n_faces = int(n_faces)
        self.pairs = {}          # (i, j) -> True when j is above i
        self.unordered = []      # pairs explicitly stated as k = 0
        self.unknown = []        # pairs explicitly stated as k = null

    def __len__(self):
        return len(self.pairs)

    def __bool__(self):
        return bool(self.pairs)

    def is_above(self, i, j):
        """True if j is stated above i, False if below, None if unstated."""
        if (i, j) in self.pairs:
            return self.pairs[(i, j)]
        if (j, i) in self.pairs:
            return not self.pairs[(j, i)]
        return None

    def successors(self):
        """Adjacency of the 'is below' DAG: lower face -> upper faces."""
        adj = {i: set() for i in range(self.n_faces)}
        for (i, j), j_above in self.pairs.items():
            lo, hi = (i, j) if j_above else (j, i)
            adj.setdefault(lo, set()).add(hi)
        return adj

    def stacking(self):
        """A bottom-to-top order consistent with every stated relation.

        Returns None if the relations contain a cycle -- i.e. the file
        describes an impossible stack.  Faces with no stated relation
        keep their index order, which makes the result deterministic and
        so safe to use for rendering.
        """
        adj = self.successors()
        indeg = {i: 0 for i in range(self.n_faces)}
        for lo, his in adj.items():
            for hi in his:
                indeg[hi] += 1
        ready = sorted(i for i, d in indeg.items() if d == 0)
        out = []
        while ready:
            v = ready.pop(0)
            out.append(v)
            for w in sorted(adj.get(v, ())):
                indeg[w] -= 1
                if indeg[w] == 0:
                    ready.append(w)
            ready.sort()
        return out if len(out) == self.n_faces else None

    def cycle_faces(self):
        """The faces involved in a cyclic stacking, or [] if consistent."""
        order = self.stacking()
        if order is not None:
            return []
        placed = set(self.stacking_prefix())
        return sorted(set(range(self.n_faces)) - placed)

    def stacking_prefix(self):
        """The part of the stack that can be ordered before a cycle bites."""
        adj = self.successors()
        indeg = {i: 0 for i in range(self.n_faces)}
        for lo, his in adj.items():
            for hi in his:
                indeg[hi] += 1
        ready = sorted(i for i, d in indeg.items() if d == 0)
        out = []
        while ready:
            v = ready.pop(0)
            out.append(v)
            for w in sorted(adj.get(v, ())):
                indeg[w] -= 1
                if indeg[w] == 0:
                    ready.append(w)
            ready.sort()
        return out

    def __repr__(self):
        state = "consistent" if self.stacking() is not None else "CYCLIC"
        return (f"LayerOrder({len(self.pairs)} relations over "
                f"{self.n_faces} faces, {state})")


def read_layer_order(frame):
    """Extract the layer relations a frame states.

    Contradictions between a pair written both ways round are reported
    by raising, because silently keeping one of them would hide a real
    inconsistency in the file.
    """
    order = LayerOrder(frame.n_faces)
    if not frame.face_orders:
        return order
    for i, j, k in frame.face_orders:
        if k is None:
            order.unknown.append((i, j))
            continue
        if k == 0:
            order.unordered.append((i, j))
            continue
        j_above = k > 0
        prior = order.is_above(i, j)
        if prior is not None and prior != j_above:
            raise ValueError(
                f"faceOrders contradicts itself for faces {i} and {j}: "
                "stated both above and below")
        order.pairs[(i, j)] = j_above
    return order


def _selftest():
    from .fold_io import read_fold
    import json

    def frame_with(orders, n_faces=3):
        faces = [[0, 1, 2]] * n_faces
        return read_fold(json.dumps({
            "vertices_coords": [[0, 0], [1, 0], [0, 1]],
            "edges_vertices": [[0, 1], [1, 2], [2, 0]],
            "faces_vertices": faces,
            "faceOrders": orders,
        }))[0]

    # --- a simple three-layer stack: 0 under 1 under 2 --------------
    lo = read_layer_order(frame_with([[0, 1, 1], [1, 2, 1]]))
    assert len(lo) == 2
    assert lo.is_above(0, 1) is True
    assert lo.is_above(1, 0) is False      # normalised, both directions work
    assert lo.is_above(0, 2) is None       # not stated, and not invented
    assert lo.stacking() == [0, 1, 2]
    assert lo.cycle_faces() == []

    # --- the redundant reverse statement is accepted ----------------
    lo = read_layer_order(frame_with([[0, 1, 1], [1, 0, -1]]))
    assert lo.is_above(0, 1) is True

    # --- a contradiction is refused ---------------------------------
    try:
        read_layer_order(frame_with([[0, 1, 1], [1, 0, 1]]))
    except ValueError as exc:
        assert "contradicts itself" in str(exc)
    else:
        raise AssertionError("contradictory faceOrders should raise")

    # --- k = 0 and null are kept apart from ordered pairs -----------
    lo = read_layer_order(frame_with([[0, 1, 0], [1, 2, None]]))
    assert len(lo) == 0
    assert lo.unordered == [(0, 1)] and lo.unknown == [(1, 2)]
    assert lo.stacking() == [0, 1, 2]      # nothing stated: index order

    # --- a cycle is detected, and the guilty faces named ------------
    lo = read_layer_order(frame_with([[0, 1, 1], [1, 2, 1], [2, 0, 1]]))
    assert lo.stacking() is None
    assert lo.cycle_faces() == [0, 1, 2]
    assert "CYCLIC" in repr(lo)

    # --- no faceOrders at all -> empty, falsy, still orderable ------
    bare = read_fold(json.dumps({
        "vertices_coords": [[0, 0], [1, 0], [0, 1]],
        "edges_vertices": [[0, 1], [1, 2], [2, 0]],
        "faces_vertices": [[0, 1, 2]],
    }))[0]
    lo = read_layer_order(bare)
    assert not lo and lo.stacking() == [0]

    print("RESULT: OK  crease.layers")


# NOTE: no __main__ guard -- tests/test_selftests.py discovers and runs
# _selftest() headlessly (see CLAUDE.md).
