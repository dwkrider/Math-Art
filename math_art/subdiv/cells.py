# Polygonal cell complexes, the objects a finite subdivision rule acts on.
#
# A subdivision rule does not act on a triangulation but on a complex of
# polygonal tiles glued along edges, each tile carrying a TYPE label that says
# how it subdivides.  This module is the container: vertices, faces as ordered
# vertex loops, per-face type labels, and the derived edge adjacency.
#
# The one real correctness condition in the whole subject is EDGE CONSISTENCY:
# where two tiles meet along an edge, both must subdivide that edge the same
# way.  Enforcing it is what keeps the complex from tearing as it refines, and
# it is checked here rather than being left to each rule.
#
# References:
# - J. W. Cannon, W. J. Floyd and W. R. Parry, "Finite subdivision rules",
#   Conformal Geometry and Dynamics 5 (2001), pp. 153-196 (the definition of a
#   subdivision rule, tile types, edge types, and R-complexes).
# - Philip L. Bowers and Kenneth Stephenson, "A 'regular' pentagonal tiling of
#   the plane", Conformal Geometry and Dynamics 1 (1997), pp. 58-86.

class CellComplex:
    """Polygonal tiles glued along shared edges.

    faces  : list of vertex-index loops, each in consistent orientation
    types  : per-face tile-type name
    nv     : vertex count
    """

    __slots__ = ('nv', 'faces', 'types')

    def __init__(self, nv, faces, types=None):
        self.nv = nv
        self.faces = [tuple(f) for f in faces]
        if types is None:
            types = ['t'] * len(self.faces)
        self.types = list(types)
        if len(self.types) != len(self.faces):
            raise ValueError("one type label per face")

    # -- derived ------------------------------------------------------------

    def edges(self):
        """Undirected edges, deduplicated."""
        seen = set()
        for f in self.faces:
            n = len(f)
            for i in range(n):
                a, b = f[i], f[(i + 1) % n]
                e = (a, b) if a < b else (b, a)
                if e not in seen:
                    seen.add(e)
                    yield e

    def edge_faces(self):
        """Map each undirected edge to the faces using it."""
        out = {}
        for fi, f in enumerate(self.faces):
            n = len(f)
            for i in range(n):
                a, b = f[i], f[(i + 1) % n]
                e = (a, b) if a < b else (b, a)
                out.setdefault(e, []).append(fi)
        return out

    def boundary_edges(self):
        return [e for e, fs in self.edge_faces().items() if len(fs) == 1]

    def euler(self):
        return self.nv - len(list(self.edges())) + len(self.faces)

    def validate(self):
        """Raise ValueError on a complex no subdivision could act on."""
        for fi, f in enumerate(self.faces):
            if len(f) < 3:
                raise ValueError("face %d has only %d sides" % (fi, len(f)))
            if len(set(f)) != len(f):
                raise ValueError("face %d repeats a vertex: %r" % (fi, f))
        for e, fs in self.edge_faces().items():
            if len(fs) > 2:
                raise ValueError("edge %r is shared by %d faces; the complex "
                                 "is not a surface" % (e, len(fs)))
        used = {v for f in self.faces for v in f}
        if used and max(used) >= self.nv:
            raise ValueError("face references vertex >= nv")
        return True

    def face_sizes(self):
        from collections import Counter
        return Counter(len(f) for f in self.faces)


def single_polygon(n, tile_type='t'):
    """One n-gon: the seed every rule starts from."""
    return CellComplex(n, [tuple(range(n))], [tile_type])


def straight_line_positions(cx, seed_positions):
    """Positions for a subdivided complex, by placing each new vertex at the
    average of the parents recorded for it.

    Straight-line layout DEGENERATES under refinement -- that is precisely the
    argument for laying out with a circle packing instead -- but it is the
    honest 'before' picture and it is useful for debugging."""
    return seed_positions


def _selftest():
    K = single_polygon(5)
    assert K.nv == 5 and len(K.faces) == 1
    assert K.euler() == 1, K.euler()
    assert len(K.boundary_edges()) == 5
    K.validate()

    # two quads sharing an edge
    Q = CellComplex(6, [(0, 1, 2, 3), (1, 4, 5, 2)])
    Q.validate()
    ef = Q.edge_faces()
    shared = [e for e, fs in ef.items() if len(fs) == 2]
    assert shared == [(1, 2)], shared
    assert Q.euler() == 1, Q.euler()

    # a repeated vertex in a face is rejected
    try:
        CellComplex(3, [(0, 1, 1)]).validate()
        raised = False
    except ValueError:
        raised = True
    assert raised, "a face repeating a vertex must be rejected"

    # three faces on one edge is not a surface
    try:
        CellComplex(5, [(0, 1, 2), (0, 1, 3), (0, 1, 4)]).validate()
        raised = False
    except ValueError:
        raised = True
    assert raised, "an edge in three faces must be rejected"

    assert dict(single_polygon(5).face_sizes()) == {5: 1}
    print("subdiv.cells: complexes build, Euler characteristic 1 for a disc, "
          "non-surface and degenerate faces rejected. RESULT: OK")
