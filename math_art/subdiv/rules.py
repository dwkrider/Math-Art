# Finite subdivision rules.
#
# A finite subdivision rule says: every tile of each TYPE is replaced by a fixed
# pattern of smaller tiles, and every edge of each type is split a fixed way.
# Iterating produces a nested sequence of complexes whose limit is the tiling.
# Cannon, Floyd and Parry give the general definition (a subdivision complex, a
# subdivision of it, and a subdivision map that is a homeomorphism on each open
# cell); the rules below are concrete instances of it.
#
# THE PENTAGONAL RULE is the one that matters.  Bowers and Stephenson describe
# it two ways.  The first is "reflect a pentagon across each of its five edges
# and paste at the corners"; the second, equivalent and far easier to implement,
# is a direct subdivision of one pentagon into SIX.  Its combinatorics, with
# corners c0..c4:
#
#     * five edge midpoints m_i, one on each edge (c_i, c_{i+1})
#     * five new interior vertices p_i
#     * a central pentagon (p0, p1, p2, p3, p4)
#     * five corner pentagons, the i-th being (c_i, m_i, p_i, p_{i-1}, m_{i-1})
#
# which is 15 vertices, 20 edges and 6 faces, so V - E + F = 1 as a disc must be.
# Iterating gives 6^n pentagons and the nested family K_0 < K_1 < K_2 < ...
#
# EDGE CONSISTENCY is the correctness condition: an edge shared by two tiles is
# split once, and both tiles must see the same midpoint.  This is handled by
# interning midpoints in a global edge -> vertex table, so the rules below
# cannot tear the complex even when tiles of different types meet.
#
# The tiles are pentagons COMBINATORIALLY.  There is no straight-line
# realisation in which they are all regular -- that is exactly why the geometry
# has to come from a circle packing (see `triangulate.py`).
#
# References:
# - J. W. Cannon, W. J. Floyd and W. R. Parry, "Finite subdivision rules",
#   Conformal Geometry and Dynamics 5 (2001), pp. 153-196.
# - Philip L. Bowers and Kenneth Stephenson, "A 'regular' pentagonal tiling of
#   the plane", Conformal Geometry and Dynamics 1 (1997), pp. 58-86 (the
#   pentagonal rule, its conformal realisation, and the proof that the tiling
#   is parabolic -- it tiles the plane, not the disc).

from .cells import CellComplex, single_polygon

PENTAGONAL = 'PENTAGONAL'
BARYCENTRIC = 'BARYCENTRIC'
QUAD = 'QUAD'
TRIANGLE_QUAD = 'TRIANGLE_QUAD'

RULES = (PENTAGONAL, BARYCENTRIC, QUAD, TRIANGLE_QUAD)

RULE_SEEDS = {
    PENTAGONAL: 5,
    BARYCENTRIC: 3,
    QUAD: 4,
    TRIANGLE_QUAD: 3,
}

RULE_FACTOR = {
    PENTAGONAL: 6,
    BARYCENTRIC: 6,
    QUAD: 4,
    TRIANGLE_QUAD: 3,
}


class _Mint:
    """Interns new vertices so shared edges get one shared midpoint."""

    def __init__(self, nv):
        self.nv = nv
        self.edge_mid = {}
        self.face_centre = {}

    def midpoint(self, a, b):
        key = (a, b) if a < b else (b, a)
        v = self.edge_mid.get(key)
        if v is None:
            v = self.nv
            self.nv += 1
            self.edge_mid[key] = v
        return v

    def centre(self, fi):
        v = self.face_centre.get(fi)
        if v is None:
            v = self.nv
            self.nv += 1
            self.face_centre[fi] = v
        return v

    def fresh(self):
        v = self.nv
        self.nv += 1
        return v


def subdivide(K, rule):
    """One level of `rule` applied to every tile of K."""
    if rule == PENTAGONAL:
        return _sub_pentagonal(K)
    if rule == BARYCENTRIC:
        return _sub_barycentric(K)
    if rule == QUAD:
        return _sub_quad(K)
    if rule == TRIANGLE_QUAD:
        return _sub_triangle_quad(K)
    raise ValueError("unknown subdivision rule %r" % (rule,))


def _sub_pentagonal(K):
    """Pentagon -> six pentagons (Bowers-Stephenson)."""
    mint = _Mint(K.nv)
    faces = []
    types = []
    for fi, f in enumerate(K.faces):
        n = len(f)
        if n != 5:
            raise ValueError("the pentagonal rule needs pentagons; face %d has "
                             "%d sides" % (fi, n))
        m = [mint.midpoint(f[i], f[(i + 1) % n]) for i in range(n)]
        p = [mint.fresh() for _ in range(n)]
        faces.append(tuple(p))                       # the central pentagon
        types.append('t')
        for i in range(n):
            faces.append((f[i], m[i], p[i], p[i - 1], m[i - 1]))
            types.append('t')
    return CellComplex(mint.nv, faces, types)


def _sub_barycentric(K):
    """Triangle -> six triangles, via edge midpoints and the centroid.

    Included because it is the honest counter-example: iterate it with a
    straight-line layout and the tiles degenerate into slivers, which is the
    argument for packing-based layout."""
    mint = _Mint(K.nv)
    faces = []
    types = []
    for fi, f in enumerate(K.faces):
        n = len(f)
        if n != 3:
            raise ValueError("the barycentric rule needs triangles; face %d "
                             "has %d sides" % (fi, n))
        c = mint.centre(fi)
        m = [mint.midpoint(f[i], f[(i + 1) % n]) for i in range(n)]
        for i in range(n):
            faces.append((f[i], m[i], c))
            faces.append((m[i], f[(i + 1) % n], c))
            types.extend(('t', 't'))
    return CellComplex(mint.nv, faces, types)


def _sub_quad(K):
    """Quadrilateral -> four quadrilaterals."""
    mint = _Mint(K.nv)
    faces = []
    types = []
    for fi, f in enumerate(K.faces):
        n = len(f)
        if n != 4:
            raise ValueError("the quad rule needs quadrilaterals; face %d has "
                             "%d sides" % (fi, n))
        c = mint.centre(fi)
        m = [mint.midpoint(f[i], f[(i + 1) % n]) for i in range(n)]
        for i in range(n):
            faces.append((f[i], m[i], c, m[i - 1]))
            types.append('t')
    return CellComplex(mint.nv, faces, types)


def _sub_triangle_quad(K):
    """Triangle -> three quadrilaterals, and quadrilateral -> four.

    A genuinely two-type rule: after one level every tile is a quadrilateral, so
    it exercises the type machinery that a single-type rule never touches."""
    mint = _Mint(K.nv)
    faces = []
    types = []
    for fi, f in enumerate(K.faces):
        n = len(f)
        if n not in (3, 4):
            raise ValueError("face %d has %d sides; this rule takes triangles "
                             "and quadrilaterals" % (fi, n))
        c = mint.centre(fi)
        m = [mint.midpoint(f[i], f[(i + 1) % n]) for i in range(n)]
        for i in range(n):
            faces.append((f[i], m[i], c, m[i - 1]))
            types.append('q')
    return CellComplex(mint.nv, faces, types)


def build(rule, depth, seed=None):
    """Seed tile, subdivided `depth` times."""
    if seed is None:
        seed = single_polygon(RULE_SEEDS[rule])
    K = seed
    for _ in range(max(0, depth)):
        K = subdivide(K, rule)
    return K


def check_edge_consistency(K):
    """Every interior edge must be seen by exactly two faces, in opposite
    directions -- the condition that keeps a subdivided complex from tearing."""
    seen = {}
    for f in K.faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            seen.setdefault(key, []).append((a, b))
    for key, dirs in seen.items():
        if len(dirs) == 2 and dirs[0] == dirs[1]:
            raise ValueError("edge %r traversed the same way by both faces; "
                             "orientation is inconsistent" % (key,))
        if len(dirs) > 2:
            raise ValueError("edge %r shared by %d faces" % (key, len(dirs)))
    return True


def _selftest():
    # 1. the pentagonal rule: 6^n pentagons, disc at every level, and the
    #    level-1 complex has exactly the 15/20/6 counts of the paper
    K = build(PENTAGONAL, 1)
    assert len(K.faces) == 6, len(K.faces)
    assert K.nv == 15, K.nv
    assert len(list(K.edges())) == 20, len(list(K.edges()))
    assert K.euler() == 1, K.euler()
    assert dict(K.face_sizes()) == {5: 6}, K.face_sizes()

    for depth in range(4):
        Kn = build(PENTAGONAL, depth)
        assert len(Kn.faces) == 6 ** depth, (depth, len(Kn.faces))
        assert dict(Kn.face_sizes()) == {5: 6 ** depth}, \
            "every tile must stay a pentagon at depth %d" % depth
        assert Kn.euler() == 1, (depth, Kn.euler())
        Kn.validate()
        check_edge_consistency(Kn)

    # 2. midpoints are SHARED: subdividing two pentagons glued along an edge
    #    must not duplicate that edge's midpoint
    glued = CellComplex(8, [(0, 1, 2, 3, 4), (1, 5, 6, 7, 2)])
    glued.validate()
    G1 = subdivide(glued, PENTAGONAL)
    assert len(G1.faces) == 12
    G1.validate()
    check_edge_consistency(G1)
    interior = [e for e, fs in G1.edge_faces().items() if len(fs) == 2]
    assert interior, "the shared edge must survive subdivision as shared edges"
    assert G1.euler() == 1, G1.euler()

    # 3. barycentric: 6^n triangles
    for depth in range(3):
        Kb = build(BARYCENTRIC, depth)
        assert len(Kb.faces) == 6 ** depth
        assert dict(Kb.face_sizes()) == {3: 6 ** depth}
        assert Kb.euler() == 1
        check_edge_consistency(Kb)

    # 4. quad: 4^n quads
    for depth in range(4):
        Kq = build(QUAD, depth)
        assert len(Kq.faces) == 4 ** depth
        assert dict(Kq.face_sizes()) == {4: 4 ** depth}
        assert Kq.euler() == 1

    # 5. the two-type rule turns a triangle into quads and keeps them quads
    Kt = build(TRIANGLE_QUAD, 1)
    assert dict(Kt.face_sizes()) == {4: 3}, Kt.face_sizes()
    Kt2 = subdivide(Kt, TRIANGLE_QUAD)
    assert dict(Kt2.face_sizes()) == {4: 12}, Kt2.face_sizes()
    assert Kt2.euler() == 1
    check_edge_consistency(Kt2)

    # 6. a rule applied to the wrong tile is refused, not silently mangled
    try:
        subdivide(single_polygon(4), PENTAGONAL)
        raised = False
    except ValueError:
        raised = True
    assert raised, "the pentagonal rule must refuse a quadrilateral"

    print("subdiv.rules: pentagonal gives 6^n pentagons with V-E+F=1 and the "
          "paper's 15/20/6 at level 1, midpoints shared across tiles, "
          "barycentric/quad/two-type rules consistent. RESULT: OK")
