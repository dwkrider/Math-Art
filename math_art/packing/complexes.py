# The combinatorics a circle packing is computed on: an oriented triangulated
# complex, stored as a "flower" per vertex.
#
# A packing needs surprisingly little combinatorial data.  For every vertex it
# needs the cyclic order of its neighbours -- the FLOWER -- because the angle
# sum is accumulated over consecutive pairs around that cycle.  An interior
# vertex has a closed flower (the cycle wraps); a boundary vertex has an open
# one (a fan, whose first and last petals are not joined).  That distinction is
# the single most common source of bugs: treating an open flower as closed
# silently places boundary circles on top of one another.
#
# Flowers are derived from an oriented face list.  For an oriented triangle
# (v, a, b) the neighbour a is followed by b in the flower at v, so collecting
# the successor relation over all faces and walking it recovers each cyclic
# order.  A vertex whose walk closes up is interior; one whose walk runs out is
# on the boundary.
#
# EXISTENCE.  Collins-Stephenson's Lemma 2.2 says the angle sum of a closed
# flower of k petals runs from k*pi (as the radius tends to 0) down to 0, so a
# target angle sum A is attainable only when 0 < A < k*pi.  For the usual aim of
# 2*pi this means an interior vertex needs at least THREE petals; a degree-2
# interior vertex has no solution at all and is rejected when the complex is
# built rather than being handed to a solver that cannot converge.
#
# References:
# - Charles R. Collins and Kenneth Stephenson, "A circle packing algorithm",
#   Computational Geometry: Theory and Applications 25 (2003), pp. 233-256
#   (Lemma 2.2, and the flower formulation used throughout).
# - Kenneth Stephenson, "Introduction to Circle Packing: The Theory of Discrete
#   Analytic Functions", Cambridge University Press, 2005.

import math


class PackingComplex:
    """An oriented triangulated disc or sphere, as flowers.

    Attributes
    ----------
    nv        : vertex count
    faces     : list of oriented (i, j, k) triples
    flowers   : per vertex, the neighbours in cyclic order
    interior  : set of vertex indices whose flower closes up
    boundary  : boundary vertices in order around the outer cycle
    """

    __slots__ = ('nv', 'faces', 'flowers', 'interior', 'boundary')

    def __init__(self, nv, faces):
        self.nv = nv
        self.faces = [tuple(f) for f in faces]
        self.flowers = [[] for _ in range(nv)]
        self.interior = set()
        self.boundary = []
        self._build()

    # -- construction -------------------------------------------------------

    def _build(self):
        succ = [dict() for _ in range(self.nv)]
        pred = [dict() for _ in range(self.nv)]
        for (a, b, c) in self.faces:
            for (v, x, y) in ((a, b, c), (b, c, a), (c, a, b)):
                succ[v][x] = y
                pred[v][y] = x
        for v in range(self.nv):
            sv = succ[v]
            if not sv:
                continue
            start = None
            for x in sv:
                if x not in pred[v]:
                    start = x
                    break
            if start is None:                     # closed flower: interior
                start = next(iter(sv))
                flower = [start]
                cur = sv[start]
                while cur != start:
                    flower.append(cur)
                    cur = sv.get(cur)
                    if cur is None:               # inconsistent orientation
                        break
                self.flowers[v] = flower
                self.interior.add(v)
            else:                                 # open flower: boundary
                flower = [start]
                cur = sv.get(start)
                guard = 0
                while cur is not None and guard < self.nv + 2:
                    flower.append(cur)
                    cur = sv.get(cur)
                    guard += 1
                self.flowers[v] = flower
        self._order_boundary()

    def _order_boundary(self):
        """Walk the boundary vertices into a single oriented cycle."""
        bset = [v for v in range(self.nv)
                if v not in self.interior and self.flowers[v]]
        if not bset:
            self.boundary = []
            return
        # for a boundary vertex the last petal is the next boundary vertex
        nxt = {v: self.flowers[v][-1] for v in bset}
        start = bset[0]
        order = [start]
        cur = nxt.get(start)
        while cur is not None and cur != start and len(order) <= len(bset):
            order.append(cur)
            cur = nxt.get(cur)
        self.boundary = order if len(order) == len(bset) else bset

    # -- queries ------------------------------------------------------------

    def degree(self, v):
        return len(self.flowers[v])

    def is_interior(self, v):
        return v in self.interior

    def edges(self):
        seen = set()
        for v, fl in enumerate(self.flowers):
            for u in fl:
                e = (v, u) if v < u else (u, v)
                if e not in seen:
                    seen.add(e)
                    yield e

    def validate(self, aim=None):
        """Raise ValueError on combinatorics no solver could pack.

        With `aim` given (default 2*pi) an interior vertex must have enough
        petals for the target angle sum to be attainable: 0 < aim < k*pi."""
        if aim is None:
            aim = 2.0 * math.pi
        for v in self.interior:
            k = self.degree(v)
            if k * math.pi <= aim:
                raise ValueError(
                    "interior vertex %d has degree %d; an angle sum of %.4f "
                    "needs degree > %.2f (Collins-Stephenson Lemma 2.2)"
                    % (v, k, aim, aim / math.pi))
        for v in range(self.nv):
            if not self.flowers[v]:
                raise ValueError("vertex %d belongs to no face" % v)
        return True


# --------------------------------------------------------------------------
# Stock combinatorics
# --------------------------------------------------------------------------

def hex_disc(rings):
    """Hexagonal triangulated disc: `rings` rings of vertices around a centre."""
    dirs = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
    index = {}
    coords = []
    for q in range(-rings, rings + 1):
        lo = max(-rings, -q - rings)
        hi = min(rings, -q + rings)
        for r in range(lo, hi + 1):
            index[(q, r)] = len(coords)
            coords.append((q, r))
    faces = []
    for (q, r) in coords:
        v = index[(q, r)]
        a = index.get((q + 1, r))
        b = index.get((q, r + 1))
        c = index.get((q + 1, r - 1))
        if a is not None and b is not None:
            faces.append((v, a, b))
        if a is not None and c is not None:
            faces.append((v, c, a))
    K = PackingComplex(len(coords), faces)
    xy = [(q + r * 0.5, r * math.sqrt(3.0) * 0.5) for (q, r) in coords]
    return K, xy


def square_grid(nx, ny):
    """Triangulated rectangular grid, (nx+1) by (ny+1) vertices."""
    idx = lambda i, j: j * (nx + 1) + i
    faces = []
    for j in range(ny):
        for i in range(nx):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i + 1, j + 1), idx(i, j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    K = PackingComplex((nx + 1) * (ny + 1), faces)
    xy = [(i, j) for j in range(ny + 1) for i in range(nx + 1)]
    return K, xy


def tetrahedron():
    """The smallest triangulated sphere."""
    faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    return PackingComplex(4, faces)


def octahedron():
    faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),
             (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4)]
    return PackingComplex(6, faces)


def from_faces(faces):
    """Build a complex from an oriented triangle list, renumbering vertices."""
    verts = sorted({v for f in faces for v in f})
    remap = {v: i for i, v in enumerate(verts)}
    return PackingComplex(len(verts),
                          [tuple(remap[v] for v in f) for f in faces])


def _selftest():
    """Structural checks; raises AssertionError on failure."""
    # 1. hex disc: counts, interior/boundary split, Euler characteristic
    for rings, nv in ((1, 7), (2, 19), (3, 37), (5, 91)):
        K, xy = hex_disc(rings)
        assert K.nv == nv, (rings, K.nv, nv)
        assert len(xy) == nv
        nb = len(K.boundary)
        assert nb == 6 * rings, (rings, nb)
        assert len(K.interior) == nv - nb
        ne = len(list(K.edges()))
        chi = K.nv - ne + len(K.faces)
        assert chi == 1, "a disc must have Euler characteristic 1, got %d" % chi

    # 2. every interior flower is a genuine cycle of distinct neighbours
    K, _ = hex_disc(3)
    for v in K.interior:
        fl = K.flowers[v]
        assert len(fl) == len(set(fl)), "repeated petal at %d" % v
        assert len(fl) == 6, "hex interior degree should be 6, got %d" % len(fl)

    # 3. boundary walks form one closed cycle
    assert len(K.boundary) == 18 and len(set(K.boundary)) == 18

    # 4. adjacency is symmetric
    for v, fl in enumerate(K.flowers):
        for u in fl:
            assert v in K.flowers[u], "asymmetric adjacency %d-%d" % (v, u)

    # 5. spheres: no boundary, Euler characteristic 2
    for K2 in (tetrahedron(), octahedron()):
        assert not K2.boundary and len(K2.interior) == K2.nv
        ne = len(list(K2.edges()))
        assert K2.nv - ne + len(K2.faces) == 2

    # 6. validate() rejects a degree-2 interior vertex
    K3, _ = hex_disc(2)
    K3.validate()
    bad = PackingComplex(4, [(0, 1, 2), (0, 2, 1)])
    try:
        bad.validate()
        raised = False
    except ValueError:
        raised = True
    assert raised, "validate() must reject an interior vertex of degree 2"

    # 7. square grid
    K4, xy4 = square_grid(4, 3)
    assert K4.nv == 20 and len(xy4) == 20
    ne = len(list(K4.edges()))
    assert K4.nv - ne + len(K4.faces) == 1

    print("packing.complexes: hex/grid/sphere combinatorics build, flowers "
          "cyclic and symmetric, degree-2 interior rejected. RESULT: OK")
