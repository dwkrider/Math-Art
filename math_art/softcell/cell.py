# The (e2) cell and its soft relatives -- first- and second-order data.
#
# Part of the Math Art soft-cell engine (`math_art/softcell/`).  Numpy only --
# no `bpy` -- so the engine imports and self-tests headlessly.
#
# The subject is the Dirichlet-Voronoi cell of the body-centred cubic lattice:
# the TRUNCATED OCTAHEDRON, six squares and eight hexagons, called the (e2)
# cell in the soft-cell literature.  Softening it does not move a single
# vertex.  What changes is the DIRECTION each edge leaves its vertex in: bend
# the edges so that, at every node, two of the incoming half-tangents are
# exactly antiparallel, and the corner disappears -- the two edges now form
# one smooth curve through the node.
#
# The remarkable part, and the reason this module is short, is that the whole
# family is controlled by ONE unit vector.  Fix the half-tangent `a` of a
# single edge at a single node; the other three at that node follow by the
# linear maps T_b, T_c, T_d below, and the rest of the cell follows by its own
# symmetry group.  So the entire space of soft cells over this lattice is a
# 2-sphere -- two Euler angles -- and the named cells of the literature are
# marked points on it:
#
#   (e2)    the polyhedron itself, softness 0        a = (1, 0, 0)
#   (f2)    standard soft cell,    softness 0.331
#   (g2)    non-standard,          softness 0.333    Schwarz P class
#   (h2)    standard,              softness 0.464
#   (i2)    non-standard,          softness 0.474    Schwarz D class
#   Kelvin  the dry-foam cell, edges meeting at the Plateau angle
#   PD      the shortest path between the two Schwarz cells; not soft
#
# "Standard" means all half-tangents at a node are collinear; "non-standard"
# that they are antiparallel only in pairs.  (g2) and (i2) are second-order
# equivalent to the Voronoi cells of the Schwarz P and Schwarz D minimal
# surfaces, which is how a foam cell and a minimal surface end up in the same
# one-parameter family.
#
# References:
# - G. Domokos, A. Goriely, A. G. Horvath and K. Regos, "Soft cells, Kelvin's
#   foam and the minimal surfaces of Schwarz", arXiv:2412.04491 (2025).
#   Table 1 gives the 24 nodes reproduced here, equations (2)-(3) the
#   transformation matrices, and Tables 2-3 the seven named cells.
#   https://arxiv.org/abs/2412.04491
# - G. Domokos, A. Goriely, A. G. Horvath and K. Regos, "Soft cells and the
#   geometry of seashells", PNAS Nexus 3(9):pgae311 (2024) -- the edge-bending
#   algorithm and the softness measure.
# - H. S. M. Coxeter, "Regular skew polyhedra in three and four dimensions,
#   and their topological analogues", Proc. London Math. Soc. s2-43:33-62
#   (1938) -- the regular map {6,4|4}, whose vertices the Schwarz P and D unit
#   cells carry, and which is "all the hexagons of the net of truncated
#   octahedra".
# - W. Thomson (Lord Kelvin), "On the division of space with minimum
#   partitional area", Phil. Mag. 24(151):503-514 (1887) -- the Kelvin cell.

import itertools
import math

import numpy as np

_R2 = math.sqrt(2.0)

# ------------------------------------------------------------------
# First order: the 24 nodes of the (e2) cell (Table 1), unit edge.
# ------------------------------------------------------------------

NODES = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    [-0.5, -0.5, 1.0 / _R2], [1.5, -0.5, 1.0 / _R2],
    [1.5, 1.5, 1.0 / _R2], [-0.5, 1.5, 1.0 / _R2],
    [-1.0, 0.0, 2.0 / _R2], [0.0, -1.0, 2.0 / _R2],
    [1.0, -1.0, 2.0 / _R2], [2.0, 0.0, 2.0 / _R2],
    [2.0, 1.0, 2.0 / _R2], [1.0, 2.0, 2.0 / _R2],
    [0.0, 2.0, 2.0 / _R2], [-1.0, 1.0, 2.0 / _R2],
    [-0.5, -0.5, 3.0 / _R2], [1.5, -0.5, 3.0 / _R2],
    [1.5, 1.5, 3.0 / _R2], [-0.5, 1.5, 3.0 / _R2],
    [0.0, 0.0, 4.0 / _R2], [1.0, 0.0, 4.0 / _R2],
    [1.0, 1.0, 4.0 / _R2], [0.0, 1.0, 4.0 / _R2],
])

CENTRE = NODES.mean(axis=0)

# Face normals quoted in the paper, used only as a cross-check.
U_QUAD = np.array([0.0, 0.0, 1.0])
U_HEX1 = np.array([0.0, 2.0 / math.sqrt(6.0), _R2 / math.sqrt(6.0)])
U_HEX2 = np.array([2.0 / math.sqrt(6.0), 0.0, _R2 / math.sqrt(6.0)])

EXACT_VOLUME = 8.0 * _R2        # truncated octahedron of unit edge


def _edges():
    """Unit-distance pairs.  A truncated octahedron of unit edge has 36."""
    out = []
    for i in range(len(NODES)):
        for j in range(i + 1, len(NODES)):
            if abs(np.linalg.norm(NODES[i] - NODES[j]) - 1.0) < 1e-9:
                out.append((i, j))
    return out


EDGES = _edges()

NEIGHBOURS = {i: [] for i in range(len(NODES))}
for _i, _j in EDGES:
    NEIGHBOURS[_i].append(_j)
    NEIGHBOURS[_j].append(_i)


def _faces():
    """Derive the 14 face cycles rather than transcribing them.

    The paper lists face membership but not cycle order -- its hexagon
    "(1,2,6,10,11,5)" is not a walk, since nodes 6 and 10 are sqrt(3) apart.
    So the faces are found here from the geometry: every plane through a
    node and two of its neighbours that leaves all 24 nodes on one side is a
    face plane, and the nodes on it are then ordered by angle about their
    centroid.  Deriving beats transcribing -- `_selftest` re-checks the
    counts, planarity and volume, so a slip cannot survive silently.
    """
    planes = []
    for i in range(len(NODES)):
        for j, k in itertools.combinations(NEIGHBOURS[i], 2):
            p, q, r = NODES[i], NODES[j], NODES[k]
            nrm = np.cross(q - p, r - p)
            ln = np.linalg.norm(nrm)
            if ln < 1e-12:
                continue
            nrm = nrm / ln
            d = float(nrm @ p)
            s = NODES @ nrm - d
            if s.max() < 1e-9 or s.min() > -1e-9:      # all on one side
                if s.max() < 1e-9:                     # outward = +n
                    nrm, d = -nrm, -d
                key = tuple(np.round(np.append(nrm, d), 9))
                if key not in [pl[0] for pl in planes]:
                    planes.append((key, nrm, d))

    faces = []
    for _key, nrm, d in planes:
        on = [t for t in range(len(NODES))
              if abs(float(NODES[t] @ nrm) - d) < 1e-9]
        pts = NODES[on]
        cen = pts.mean(axis=0)
        u = pts[0] - cen
        u = u / np.linalg.norm(u)
        v = np.cross(nrm, u)
        ang = np.arctan2((pts - cen) @ v, (pts - cen) @ u)
        order = np.argsort(ang)
        faces.append(tuple(on[t] for t in order))
    return faces


FACES = _faces()

# ------------------------------------------------------------------
# Second order: one unit vector generates the whole tangent field.
# ------------------------------------------------------------------

T_B = np.array([[0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0]])

T_C = np.array([[-0.5, -0.5, 1.0 / _R2],
                [-0.5, -0.5, -1.0 / _R2],
                [1.0 / _R2, -1.0 / _R2, 0.0]])

# T_d1 for octahedral symmetry (Theorem 1), T_d2 for tetrahedral (Theorem 2)
T_D1 = np.array([[-0.5, -0.5, 1.0 / _R2],
                 [-0.5, -0.5, -1.0 / _R2],
                 [-1.0 / _R2, 1.0 / _R2, 0.0]])

T_D2 = np.array([[-0.5, -0.5, -1.0 / _R2],
                 [-0.5, -0.5, 1.0 / _R2],
                 [-1.0 / _R2, 1.0 / _R2, 0.0]])

# The polyhedral edge directions at node 0, in the slot order (a, b, c).
# Node 0 = (0,0,0) has neighbours 1, 3 and 4.
SLOT_DIRS = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [-0.5, -0.5, 1.0 / _R2]])

# name -> (colatitude phi, azimuth theta, symmetry, published softness)
# Tables 2 and 3.  `None` softness means the paper does not quote one.
PRESETS = {
    'E2':     (math.pi / 2.0, 0.0, 'OCTAHEDRAL', 0.0),
    'F2':     (math.pi / 2.0, math.pi / 4.0, 'OCTAHEDRAL', 0.331),
    'G2':     (math.pi / 2.0, -math.pi / 4.0, 'OCTAHEDRAL', 0.333),
    'H2':     (math.pi / 4.0, -math.pi / 4.0, 'TETRAHEDRAL', 0.464),
    'I2':     (math.acos(1.0 / math.sqrt(6.0)),
               math.atan2(1.0, 3.0), 'TETRAHEDRAL', 0.474),
    'KELVIN': (math.pi / 2.0, math.atan2(2.0 * _R2 - 3.0, 1.0),
               'OCTAHEDRAL', None),
    'PD':     (math.acos(1.0 / math.sqrt(14.0)),
               math.atan2(-1.0, 5.0), 'TETRAHEDRAL', None),
}

# Cartesian values as printed in Tables 2 and 3, for the self-test to
# confirm the angle convention rather than assume it.
PRESET_CARTESIAN = {
    'E2': (1.0, 0.0, 0.0),
    'F2': (1.0 / _R2, 1.0 / _R2, 0.0),
    'G2': (1.0 / _R2, -1.0 / _R2, 0.0),
    'H2': (0.5, -0.5, 1.0 / _R2),
    'I2': (math.sqrt(3.0) / 2.0, math.sqrt(3.0) / 6.0,
           1.0 / math.sqrt(6.0)),
    'KELVIN': (1.0 / math.sqrt(1.0 + (2.0 * _R2 - 3.0) ** 2),
               (2.0 * _R2 - 3.0) / math.sqrt(1.0 + (2.0 * _R2 - 3.0) ** 2),
               0.0),
    'PD': (5.0 / math.sqrt(28.0), -1.0 / math.sqrt(28.0),
           _R2 / math.sqrt(28.0)),
}


def direction(phi, theta):
    """The morphospace point: colatitude phi from z, azimuth theta from x."""
    return np.array([math.sin(phi) * math.cos(theta),
                     math.sin(phi) * math.sin(theta),
                     math.cos(phi)])


def nodal_set(a, symmetry='TETRAHEDRAL'):
    """The four half-tangents (a, b, c, d) at a node, from `a` alone."""
    td = T_D1 if symmetry == 'OCTAHEDRAL' else T_D2
    return np.array([a, T_B @ a, T_C @ a, td @ a])


def _node_permutation(R):
    """The permutation R induces on the nodes, or None if it is not a
    symmetry of the cell."""
    P = (NODES - CENTRE) @ R.T + CENTRE
    perm = []
    for p in P:
        d = np.linalg.norm(NODES - p, axis=1)
        j = int(np.argmin(d))
        if d[j] > 1e-9:
            return None
        perm.append(j)
    return perm


def _full_cell_group():
    """The symmetry group of the (e2) cell, DERIVED from its geometry.

    Note this is *not* the closure of {T_b, T_c, T_d}: those three matrices
    exist to build the four half-tangents at one node out of `a`, and they
    generate only a group of order 8.  The symmetry group is found instead
    by mapping the frame (node 0, two of its neighbours) onto every other
    node-and-neighbour-pair, keeping the maps that are orthogonal and
    permute the node set.  For a truncated octahedron that yields the
    octahedral group, order 48.
    """
    o = NODES[0] - CENTRE
    nb = NEIGHBOURS[0]
    A = np.array([o, NODES[nb[0]] - CENTRE, NODES[nb[1]] - CENTRE]).T
    Ainv = np.linalg.inv(A)
    out = []
    for v in range(len(NODES)):
        for w1, w2 in itertools.permutations(NEIGHBOURS[v], 2):
            B = np.array([NODES[v] - CENTRE, NODES[w1] - CENTRE,
                          NODES[w2] - CENTRE]).T
            R = B @ Ainv
            if not np.allclose(R @ R.T, np.eye(3), atol=1e-9):
                continue
            if any(np.allclose(R, H, atol=1e-9) for H, _ in out):
                continue
            perm = _node_permutation(R)
            if perm is not None:
                out.append((R, perm))
    return out


def cell_group(symmetry='TETRAHEDRAL'):
    """The symmetry group used to propagate the tangent field.

    OCTAHEDRAL is the full group of the cell (order 48).  Each node then
    has a stabiliser of order 2 -- the reflection swapping two of its three
    edges -- so the decoration is only consistent when those two slots
    carry the same half-tangent.  That extra condition is precisely why the
    paper finds only TWO soft cells with the full symmetry but FOUR with
    tetrahedral symmetry.

    TETRAHEDRAL is the rotation subgroup (det +1, order 24).  It acts
    SIMPLY TRANSITIVELY on the 24 nodes -- no non-identity rotation fixes a
    vertex of a truncated octahedron, because its 3-fold axes run through
    hexagon centres, not vertices -- so the decoration is always
    consistent, whatever direction `a` is given.
    """
    G = _full_cell_group()
    if symmetry == 'OCTAHEDRAL':
        return G
    return [(R, p) for R, p in G if np.linalg.det(R) > 0.0]


def tangent_field(a, symmetry='TETRAHEDRAL'):
    """Outgoing unit half-tangent for every directed cell edge.

    Returns {(v, w): unit vector}, the direction the edge v->w leaves v in
    after softening.  Node 0 carries the prescribed (a, b, c) on its three
    slots; every other node is reached by a symmetry of the cell, and the
    slot decoration is carried along with it.

    Raises if the assignment is not well defined -- that is, if some node's
    stabiliser maps a slot onto another slot carrying a different vector.
    That can genuinely happen for a Custom direction under the full
    octahedral group, and it is a real constraint on the morphospace, not a
    numerical wobble, so it is surfaced rather than averaged away.
    """
    abcd = nodal_set(np.asarray(a, float), symmetry)
    soft0 = abcd[:3]                       # this cell sees a, b, c
    field = {}
    for R, perm in cell_group(symmetry):
        v = perm[0]
        dirs = SLOT_DIRS @ R.T
        vecs = soft0 @ R.T
        for s in range(3):
            w_dir = dirs[s]
            w = None
            for cand in NEIGHBOURS[v]:
                u = NODES[cand] - NODES[v]
                if np.linalg.norm(u / np.linalg.norm(u) - w_dir) < 1e-7:
                    w = cand
                    break
            if w is None:
                continue
            prev = field.get((v, w))
            if prev is not None and not np.allclose(prev, vecs[s], atol=1e-7):
                raise ValueError(
                    "half-tangent field is not well defined at node "
                    f"{v} under {symmetry.lower()} symmetry")
            field[(v, w)] = vecs[s]
    return field


# ------------------------------------------------------------------
# Lattice
# ------------------------------------------------------------------

def face_translations():
    """The 14 vectors carrying the cell onto its face neighbours."""
    return np.array([2.0 * (NODES[list(f)].mean(axis=0) - CENTRE)
                     for f in FACES])


def lattice_basis():
    """A rank-3 basis picked from the face translations.

    Chosen greedily by length, so the result is the short bcc basis rather
    than whichever three happen to come first.
    """
    T = face_translations()
    T = T[np.argsort(np.linalg.norm(T, axis=1))]
    basis = []
    for t in T:
        trial = basis + [t]
        if np.linalg.matrix_rank(np.array(trial), tol=1e-9) == len(trial):
            basis = trial
            if len(basis) == 3:
                break
    return np.array(basis)


def _selftest():
    n = len(NODES)
    assert n == 24, n
    assert len(EDGES) == 36, len(EDGES)
    assert len(FACES) == 14, len(FACES)
    squares = sum(1 for f in FACES if len(f) == 4)
    hexes = sum(1 for f in FACES if len(f) == 6)
    assert (squares, hexes) == (6, 8), (squares, hexes)
    assert n - len(EDGES) + len(FACES) == 2
    print(f"(e2): {n} nodes, {len(EDGES)} unit edges, {squares} squares + "
          f"{hexes} hexagons, Euler 2  OK")

    # every derived face is planar and the volume is the exact 8*sqrt(2)
    for f in FACES:
        P = NODES[list(f)]
        nrm = np.cross(P[1] - P[0], P[2] - P[0])
        nrm = nrm / np.linalg.norm(nrm)
        assert np.abs((P - P[0]) @ nrm).max() < 1e-9
    vol = 0.0
    for f in FACES:
        P = NODES[list(f)] - CENTRE
        for i in range(1, len(f) - 1):
            vol += abs(float(np.dot(P[0], np.cross(P[i], P[i + 1])))) / 6.0
    assert abs(vol - EXACT_VOLUME) < 1e-9, (vol, EXACT_VOLUME)
    print(f"(e2): faces planar, volume {vol:.6f} = 8*sqrt(2)  OK")

    # the paper's quoted face normals must appear among the derived ones
    outward = []
    for f in FACES:
        c = NODES[list(f)].mean(axis=0) - CENTRE
        outward.append(c / np.linalg.norm(c))
    for name, u in (("u_quad", U_QUAD), ("u_hex1", U_HEX1),
                    ("u_hex2", U_HEX2)):
        hit = any(np.allclose(np.abs(o), np.abs(u), atol=1e-9)
                  for o in outward)
        assert hit, name
    print("(e2): published face normals u_quad/u_hex1/u_hex2 all present  OK")

    # the matrices are orthogonal and the groups have the stated orders
    for name, M in (("T_b", T_B), ("T_c", T_C),
                    ("T_d1", T_D1), ("T_d2", T_D2)):
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-12), name
    GO = cell_group('OCTAHEDRAL')
    GT = cell_group('TETRAHEDRAL')
    assert len(GO) == 48, len(GO)
    assert len(GT) == 24, len(GT)
    # the tetrahedral (rotation) subgroup must act simply transitively on
    # the nodes: only the identity may fix one
    fixed = [sum(1 for i, j in enumerate(p) if i == j) for _R, p in GT]
    assert sorted(fixed)[-1] == 24 and sorted(fixed)[-2] == 0, sorted(fixed)[-3:]
    assert len({tuple(p) for _R, p in GT}) == 24
    print("(e2): T matrices orthogonal; cell group 48, rotation subgroup 24 "
          "acting simply transitively on the nodes  OK")

    # the angle convention must reproduce every Cartesian triple in the
    # published tables -- this is the check that pins (phi, theta)
    for name, (phi, theta, sym, _s) in PRESETS.items():
        got = direction(phi, theta)
        want = np.array(PRESET_CARTESIAN[name])
        assert abs(np.linalg.norm(want) - 1.0) < 1e-12, name
        assert np.allclose(got, want, atol=1e-12), (name, got, want)
    print("(e2): all 7 presets reproduce their published [x,y,z] to 1e-12  OK")

    # the softening equations, exactly as the paper states them
    def prods(name):
        phi, theta, sym, _s = PRESETS[name]
        a, b, c, d = nodal_set(direction(phi, theta), sym)
        return {'ab': a @ b, 'ac': a @ c, 'ad': a @ d,
                'bc': b @ c, 'bd': b @ d, 'cd': c @ d}

    p = prods('F2')
    for k in ('ac', 'ad', 'bc', 'bd'):
        assert abs(p[k] + 1.0) < 1e-12, ('F2', k, p[k])
    p = prods('G2')
    assert abs(p['ab'] + 1.0) < 1e-12 and abs(p['cd'] + 1.0) < 1e-12, p
    p = prods('H2')
    assert abs(p['ab'] + 1.0) < 1e-12 and abs(p['cd'] + 1.0) < 1e-12, p
    p = prods('I2')
    assert abs(p['ad'] + 1.0) < 1e-12 and abs(p['bc'] + 1.0) < 1e-12, p
    p = prods('KELVIN')
    for k, v in p.items():
        assert abs(v + 1.0 / 3.0) < 1e-9, ('KELVIN', k, v)
    p = prods('PD')
    assert abs(p['ab'] + 3.0 / 7.0) < 1e-9, p
    assert abs(p['ac'] - 1.0 / 7.0) < 1e-9, p
    assert abs(p['bc'] + 5.0 / 7.0) < 1e-9, p
    print("(e2): softening products hold exactly -- f2/g2/h2/i2 antiparallel "
          "pairs, Kelvin all -1/3, PD -3/7 1/7 -5/7  OK")

    # the tangent field must be well defined, unit, and defined on every
    # directed edge of the cell
    for name, (phi, theta, sym, _s) in PRESETS.items():
        fld = tangent_field(direction(phi, theta), sym)
        assert len(fld) == 2 * len(EDGES), (name, len(fld))
        for v, vec in fld.items():
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-9, (name, v)
    print("(e2): tangent field well defined on all 72 directed edges "
          "for every preset  OK")

    # (e2) itself must reproduce the polyhedron: every tangent along its
    # own straight edge
    fld = tangent_field(direction(*PRESETS['E2'][:2]), 'OCTAHEDRAL')
    worst = 0.0
    for (v, w), t in fld.items():
        u = NODES[w] - NODES[v]
        u = u / np.linalg.norm(u)
        worst = max(worst, float(np.linalg.norm(t - u)))
    assert worst < 1e-9, worst
    print(f"(e2): the (e2) preset is the polyhedron exactly "
          f"(max tangent deviation {worst:.1e})  OK")

    # lattice: rank 3, and its determinant is the cell volume
    B = lattice_basis()
    det = abs(float(np.linalg.det(B)))
    assert abs(det - EXACT_VOLUME) < 1e-9, (det, EXACT_VOLUME)
    print(f"(e2): lattice determinant {det:.6f} = cell volume  OK")

    print("softcell.cell standalone tests passed")
