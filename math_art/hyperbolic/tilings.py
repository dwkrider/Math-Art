# Tilings of the hyperbolic plane, in the Poincare disc.
#
# Part of the Math Art hyperbolic engine (`math_art/hyperbolic/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# In the hyperbolic plane the angle sum of a triangle is less than pi,
# and {p,q} tilings exist for every 1/p + 1/q < 1/2 -- infinitely many,
# where the sphere allows five and the plane three.  The Poincare disc
# model is conformal: hyperbolic lines are circular arcs meeting the
# boundary at right angles, and reflection in such a line is inversion
# in that circle, which is how the tiling is generated.
#
# References:
# - H. Poincare, "Theorie des groupes fuchsiens", Acta Mathematica 1,
#   1882 -- the disc model.
# - H. S. M. Coxeter, "Crystal symmetry and its generalizations", 1957,
#   and the Coxeter-Escher correspondence behind the Circle Limit prints.
# - D. Dunham, "Hyperbolic symmetry", Computers & Mathematics with
#   Applications 12B, 1986 -- the replication algorithm.

from math import cos, sin, pi, sqrt, acosh
from collections import deque
import numpy as np

try:
    from .. import tiling_generator as tg
except ImportError:  # flat import outside the package
    import tiling_generator as tg



_J = np.diag([1.0, 1.0, -1.0])


# Ringed Coxeter nodes for each Wythoff form (mirror indices 0,1,2 on
# the linear diagram  node0 -p- node1 -q- node2,  branch 0-2 = r).
FORM_RINGS = {
    'REGULAR': frozenset({0}),
    'TRUNCATED': frozenset({0, 1}),
    'RECTIFIED': frozenset({1}),
    'BITRUNCATED': frozenset({1, 2}),
    'CANTELLATED': frozenset({0, 2}),
    'OMNITRUNCATED': frozenset({0, 1, 2}),
}


def _mdot(a, b):
    return a[0] * b[0] + a[1] * b[1] - a[2] * b[2]


def _normalize_h(v):
    """Scale a timelike vector onto the upper sheet <x,x> = -1, t > 0."""
    nn = -_mdot(v, v)
    if nn <= 1e-14:
        raise ValueError("point is not timelike (non-hyperbolic seed)")
    v = v / sqrt(nn)
    return -v if v[2] < 0.0 else v


def _to_klein(v):
    """Hyperboloid -> Klein disk: (x, y, t) -> (x, y) / t."""
    return np.array([v[0] / v[2], v[1] / v[2]])


def _to_poincare(v):
    """Hyperboloid -> Poincare disk: (x, y, t) -> (x, y) / (1 + t)."""
    d = 1.0 + v[2]
    return np.array([v[0] / d, v[1] / d])


def _to_hemisphere(v):
    """Hyperboloid -> upper unit hemisphere: Klein coordinates lifted
    to z = sqrt(1 - |K|^2).  Returns (point, outward unit normal); on
    the unit sphere the point is its own normal."""
    kx, ky = v[0] / v[2], v[1] / v[2]
    z = sqrt(max(0.0, 1.0 - kx * kx - ky * ky))
    p = np.array([kx, ky, z])
    return p, p


def _cayley(z):
    """Poincare disk -> upper half plane, 0 -> i, 1 -> 0."""
    return 1j * (1.0 - z) / (1.0 + z)


def _pseudosphere_point(x, y):
    """Upper half plane (x, y >= 1) -> tractricoid point and outward
    unit normal via the isometry u = arccosh(y), v = x: sech u = 1/y,
    tanh u = sqrt(y^2-1)/y.  (Derivation: the tractricoid metric
    tanh^2 u du^2 + sech^2 u dv^2 matches the UHP metric under
    y = cosh u, x = v.)"""
    y = max(y, 1.0)
    se = 1.0 / y
    th = sqrt(y * y - 1.0) / y
    cx, sx = cos(x), sin(x)
    S = np.array([se * cx, se * sx, acosh(y) - th])
    N = np.array([th * cx, th * sx, se])
    return S, N


def _pseudo_setup(corner_b):
    """Moebius shift centring the pseudosphere view: carry the q-fold
    corner B to x = pi, y = 2 on the fat part of the horn."""
    d = _to_poincare(corner_b)
    w0 = _cayley(complex(d[0], d[1]))
    return w0.real, 2.0 / w0.imag


def _to_pseudosphere(v, shift, mag, y_cap):
    """Hyperboloid -> tractricoid, or None if the point falls outside
    the drawn strip (one seam period 0 <= x <= 2pi, 1 <= y <= y_cap)."""
    d = _to_poincare(v)
    w = _cayley(complex(d[0], d[1]))
    w = (w - shift) * mag + pi
    x, y = w.real, w.imag
    if y < 1.0 or y > y_cap or x < 0.0 or x > 2.0 * pi:
        return None
    return _pseudosphere_point(x, y)


def fundamental_domain(p, q, r=2):
    """Mirror normals and fundamental triangle of the (p, q, r)
    reflection group.  Returns (normals, corners) with normals the
    three unit spacelike mirror normals and corners rows [A, B, C]:
    A the order-p corner (mirrors 0,1), B order-q (mirrors 1,2), C
    order-r (mirrors 0,2); each corner has <c,c> = -1, t > 0."""
    if 1.0 / p + 1.0 / q + 1.0 / r >= 1.0 - 1e-12:
        raise ValueError("(%d,%d,%d) is not hyperbolic "
                         "(need 1/p + 1/q + 1/r < 1)" % (p, q, r))
    sp, cp = sin(pi / p), cos(pi / p)
    cq, cr = cos(pi / q), cos(pi / r)
    n0 = np.array([0.0, 1.0, 0.0])              # <n0,n1> = -cos(pi/p)
    n1 = np.array([sp, -cp, 0.0])
    y2 = -cr                                    # <n0,n2> = -cos(pi/r)
    x2 = (-cq - cp * cr) / sp                   # <n1,n2> = -cos(pi/q)
    z2sq = x2 * x2 + y2 * y2 - 1.0              # <n2,n2> = 1
    if z2sq <= 1e-12:
        raise ValueError("(%d,%d,%d) is not hyperbolic "
                         "(need 1/p + 1/q + 1/r < 1)" % (p, q, r))
    n2 = np.array([x2, y2, sqrt(z2sq)])
    normals = [n0, n1, n2]
    A = np.array([0.0, 0.0, 1.0])               # mirrors 0, 1
    B = _normalize_h(_J @ np.cross(n1, n2))     # mirrors 1, 2
    C = _normalize_h(_J @ np.cross(n0, n2))     # mirrors 0, 2
    return normals, np.array([A, B, C])


def _reflections(normals):
    """Reflection matrices x -> x - 2<x,n>n for each mirror normal."""
    return [np.eye(3) - 2.0 * np.outer(n, _J @ n) for n in normals]


def enumerate_group(refl, corners, depth, max_tiles):
    """BFS over reflection words.  Returns the list of distinct group
    elements (3x3 matrices), pruned by the rounded ambient centroid of
    the image of the fundamental triangle."""
    cen0 = corners.mean(axis=0)
    elems, seen = [], set()
    queue = deque([(np.eye(3), 0)])
    while queue and len(elems) < max_tiles:
        M, d = queue.popleft()
        key = tuple(np.round(M @ cen0, 6))
        if key in seen:
            continue
        seen.add(key)
        elems.append(M)
        if d < depth:
            for R in refl:
                queue.append((M @ R, d + 1))
    return elems


def _dihedral(Ri, Rj):
    """The finite dihedral subgroup generated by two reflections whose
    product has finite order (the stabilizer of the corner where the
    two mirrors meet)."""
    elems, seen = [], set()
    queue = deque([np.eye(3)])
    while queue:
        M = queue.popleft()
        key = tuple(np.round(M.flatten(), 6))
        if key in seen:
            continue
        seen.add(key)
        elems.append(M)
        for R in (Ri, Rj):
            queue.append(M @ R)
        if len(elems) > 500:                    # safety valve
            break
    return elems


def compute_seed(normals, rings):
    """The Wythoff seed for a ring set: the point on every inactive
    mirror and equidistant from the active ones.  With m_i = J n_i we
    have <s, n_i> = s . m_i, so the constraints are two linear
    equations s . w = 0; the seed is their null direction renormalized
    to the hyperboloid."""
    m = [_J @ n for n in normals]
    unringed = [i for i in range(3) if i not in rings]
    ringed = [i for i in range(3) if i in rings]
    rows = [m[i] for i in unringed]                     # on inactive mirrors
    rows += [m[a] - m[b] for a, b in zip(ringed, ringed[1:])]  # equal dist
    rows = np.array(rows)                               # exactly (2, 3)
    return _normalize_h(np.cross(rows[0], rows[1]))


# Corner index -> (mirror pair, order-symbol slot 0=p 1=q 2=r).
_CORNERS = [((0, 1), 0), ((1, 2), 1), ((0, 2), 2)]


def _order_ring(pts):
    """Order hyperboloid points cyclically by angle about their Klein
    centroid (each face is convex about its corner, so this is a simple
    polygon)."""
    k = np.array([_to_klein(p) for p in pts])
    c = k.mean(axis=0)
    order = np.argsort(np.arctan2(k[:, 1] - c[1], k[:, 0] - c[0]))
    return [pts[i] for i in order]


def _local_faces(normals, corners, refl, seed, rings, orders):
    """For each face-producing corner, the complete local face: the
    seed's orbit under that corner's dihedral stabilizer, deduped and
    cyclically ordered.  Returns list over corners of (points, nsides)
    or None (that corner is a vertex, or the face is a degenerate
    2-gon)."""
    out = []
    for (mi, mj), slot in _CORNERS:
        active = len({mi, mj} & rings)
        if active == 0:                         # this corner is a vertex
            out.append(None)
            continue
        pts, keys = [], set()
        for M in _dihedral(refl[mi], refl[mj]):
            v = _normalize_h(M @ seed)
            key = tuple(np.round(_to_klein(v), 7))
            if key not in keys:
                keys.add(key)
                pts.append(v)
        if len(pts) < 3:                        # 2-gon: just an edge
            out.append(None)
            continue
        out.append((_order_ring(pts), len(pts)))
    return out


def uniform_faces(p, q, r, form, depth, max_tiles):
    """All faces of the uniform tiling: a list of (ordered hyperboloid
    corner points, side count).  Each face is emitted once, keyed by
    corner type and projected centre."""
    normals, corners = fundamental_domain(p, q, r)
    refl = _reflections(normals)
    seed = compute_seed(normals, FORM_RINGS[form])
    orders = (p, q, r)
    local = _local_faces(normals, corners, refl, seed,
                         FORM_RINGS[form], orders)
    group = enumerate_group(refl, corners, depth, max_tiles)

    faces, seen = [], set()
    for g in group:
        parity = 0 if np.linalg.det(g) > 0.0 else 1
        for cidx in range(3):
            if local[cidx] is None:
                continue
            base_pts, nsides = local[cidx]
            fp = [_normalize_h(g @ v) for v in base_pts]
            centre = np.mean([_to_klein(v) for v in fp], axis=0)
            key = (cidx,) + tuple(np.round(centre, 6))
            if key in seen:
                continue
            seen.add(key)
            faces.append((fp, nsides, parity))
    return faces


def snub_faces(p, q, r, depth, max_tiles):
    """Faces of the chiral snub sr{p,q} (snub of the (p,q,r) tiling),
    built as the alternation of the omnitruncation.

    The vertices are the orbit of the omnitruncation seed (the
    incentre) under the ROTATION subgroup [p,q,r]+ only -- the
    orientation-preserving elements (linear determinant > 0), so no
    mirror image ever appears and the tiling is chiral.  Faces are all
    orbits of the seed under that subgroup:

      * a p-gon around every order-p centre (the seed cycled by the
        rotation rho_A about corner A),
      * a q-gon around every order-q centre (rho_B about B),
      * an r-gon around every order-r centre (rho_C about C, dropped
        when r = 2 where it degenerates to an edge),
      * the snub TRIANGLES (s, rho_A^-1 s, rho_B s): their three sides
        are images of a p-gon edge, a q-gon edge and an r-gon edge, so
        each triangle plugs the gap left where an omnitruncation vertex
        was deleted by the alternation.

    Uses the incentre seed, so the p/q/r-gons are regular but the snub
    triangles are not exactly equilateral: this is the standard
    alternation snub (a valid, coverage-clean chiral tiling), not the
    fully edge-uniform snub point.  types = side count."""
    normals, corners = fundamental_domain(p, q, r)
    refl = _reflections(normals)
    R0, R1, R2 = refl
    rot_a = R0 @ R1                        # order p about corner A
    rot_b = R1 @ R2                        # order q about corner B
    rot_c = R2 @ R0                        # order r about corner C
    seed = compute_seed(normals, FORM_RINGS['OMNITRUNCATED'])

    templates = []                         # (hyperboloid points, nsides)
    templates.append(([_normalize_h(np.linalg.matrix_power(rot_a, k)
                                    @ seed) for k in range(p)], p))
    templates.append(([_normalize_h(np.linalg.matrix_power(rot_b, k)
                                    @ seed) for k in range(q)], q))
    if r >= 3:                             # r = 2 -> degenerate digon
        templates.append(([_normalize_h(np.linalg.matrix_power(rot_c, k)
                                        @ seed) for k in range(r)], r))
    rot_a_inv = np.linalg.matrix_power(rot_a, p - 1)
    tri = [seed, _normalize_h(rot_a_inv @ seed),
           _normalize_h(rot_b @ seed)]
    templates.append((tri, 3))
    templates = [(_order_ring(pts), n) for pts, n in templates]

    group = enumerate_group(refl, corners, depth, max_tiles)
    rotations = [g for g in group if np.linalg.det(g) > 0.0]

    faces, seen = [], set()
    for g in rotations:
        for tidx, (base_pts, nsides) in enumerate(templates):
            fp = [_normalize_h(g @ v) for v in base_pts]
            centre = np.mean([_to_klein(v) for v in fp], axis=0)
            key = (tidx,) + tuple(np.round(centre, 6))
            if key in seen:
                continue
            seen.add(key)
            faces.append((fp, nsides, 0))       # all rotations: parity 0
    return faces


def _subdivide_edge(P, Q, samples):
    """Sample the hyperboloid geodesic from P toward Q (excluding Q):
    convex combinations renormalized onto the sheet."""
    out = []
    for s in range(samples):
        t = s / float(samples)
        out.append(_normalize_h((1.0 - t) * P + t * Q))
    return out


def _canonicalize_corners(faces, eps=1e-6):
    """Snap every face corner shared by adjacent faces to ONE identical
    hyperboloid vector, so the curved-surface models are watertight by
    construction.

    Each face's corners are computed via its own reflection word, so a
    corner physically shared by two faces gets slightly different float
    coordinates in each.  When the geodesic edges are then subdivided,
    the two faces sample from mismatched endpoints and their shared edge
    splits into two nearly-coincident polylines -> visible cracks on the
    hemisphere / pseudosphere.

    Here we build one representative vector per corner, keyed on its
    Klein position, matched with a small spatial tolerance (robust to
    the float noise and to rounding-boundary straddle).  Both faces then
    subdivide from the SAME endpoints and emit bit-identical edge points,
    so adjacent tiles join exactly."""
    cell = 2.0 * eps
    grid = {}                                   # (ix, iy) -> [(kx, ky, v)]

    def canon(v):
        kx, ky = _to_klein(v)
        gx, gy = int(np.floor(kx / cell)), int(np.floor(ky / cell))
        for dx in (-1, 0, 1):                   # scan the 3x3 neighborhood
            for dy in (-1, 0, 1):
                for bx, by, bv in grid.get((gx + dx, gy + dy), ()):
                    if abs(bx - kx) <= eps and abs(by - ky) <= eps:
                        return bv
        grid.setdefault((gx, gy), []).append((kx, ky, v))
        return v

    return [([canon(v) for v in fp], nsides, parity)
            for fp, nsides, parity in faces]


def tiling_faces(p, q, r, form, depth, max_tiles):
    """Dispatch: the hyperboloid face list for any Wythoff form."""
    if form == 'SNUB':
        return snub_faces(p, q, r, depth, max_tiles)
    return uniform_faces(p, q, r, form, depth, max_tiles)


def _face_boundary(fp, model, edge_samples):
    """Ordered hyperboloid points around a face boundary: just the
    corners for the straight-edged Klein model, geodesic-subdivided
    for every curved model."""
    if model == 'KLEIN':
        return list(fp)
    out = []
    m = len(fp)
    for k in range(m):
        out += _subdivide_edge(fp[k], fp[(k + 1) % m], edge_samples)
    return out


def build_uniform(p, q, r, form, model='POINCARE', depth=12,
                  max_tiles=8000, edge_samples=8):
    """Build (polys, sides, parities) for a FLAT disk model (KLEIN or
    POINCARE).

    polys    -- list of (N, 2) float arrays, one flat disk polygon per
                face (straight-edged in KLEIN, geodesic-subdivided in
                POINCARE)
    sides    -- parallel list of side counts (the face's true polygon)
    parities -- parallel list of 0/1 reflection-word parities
    """
    faces = tiling_faces(p, q, r, form, depth, max_tiles)
    colors = _parity_colors(faces)
    proj = _to_klein if model == 'KLEIN' else _to_poincare
    polys, sides, parities = [], [], []
    for i, (fp, nsides, _par) in enumerate(faces):
        ring = _face_boundary(fp, model, edge_samples)
        polys.append(np.array([proj(v) for v in ring]))
        sides.append(int(nsides))
        parities.append(int(colors[i]))
    return polys, sides, parities


def _parity_colors(faces):
    """The classic alternating two-tone as a proper face 2-coloring:
    build the face-adjacency graph (two faces are adjacent when they
    share an edge = two corner vertices) and 2-color it by BFS, each
    component rooted at the face nearest the origin with the root taking
    its reflection-word (det) parity.  This flips color across every
    shared edge, so it is a clean checkerboard exactly when the tiling
    is bipartite (q even); odd q forces an odd cycle and the BFS leaves
    a frustration seam.  Returns a 0/1 list parallel to faces."""
    from collections import Counter, deque as _deque
    keyed, vtof = [], {}
    for idx, (fp, _n, _p) in enumerate(faces):
        ks = [tuple(np.round(_to_klein(v), 5)) for v in fp]
        keyed.append(ks)
        for k in ks:
            vtof.setdefault(k, []).append(idx)
    adj = [set() for _ in faces]
    for idx, ks in enumerate(keyed):
        cnt = Counter()
        for k in ks:
            for j in vtof[k]:
                if j != idx:
                    cnt[j] += 1
        for j, c in cnt.items():
            if c >= 2:
                adj[idx].add(j)
                adj[j].add(idx)
    cent = [np.mean([_to_klein(v) for v in fp], axis=0)
            for fp, _n, _p in faces]
    order = sorted(range(len(faces)),
                   key=lambda i: cent[i][0] ** 2 + cent[i][1] ** 2)
    color = [-1] * len(faces)
    for root in order:
        if color[root] != -1:
            continue
        color[root] = faces[root][2]            # det parity at the root
        queue = _deque([root])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if color[w] == -1:
                    color[w] = color[u] ^ 1
                    queue.append(w)
    return color


def _kind_for(nsides, parity, color_by):
    """Material index for a face, matching cells_from_polys: by side
    count, by tile type (also the side count here), uniform, or the
    reflection-word parity."""
    if color_by == 'UNIFORM':
        return 0
    if color_by == 'PARITY':
        return int(parity)
    if color_by == 'TYPE':
        return int(nsides)
    return tg._SIDE_MAT.get(int(nsides), 6)


def _fan_cell(ring, normals, valid, centre, cnormal, cvalid, kind,
              thickness):
    """One 3D tile cell on a curved surface: a fan of triangles from the
    face centre out to its subdivided boundary ring.  Only fan wedges
    whose centre and both rim points are in range are kept, so a tile
    straddling the pseudosphere seam / rim / cusp contributes just its
    visible part (like the old grid clipping) instead of vanishing
    whole.  With thickness > 0 a fully in-range tile also gets an inner
    shell (offset along the surface normals) and side walls."""
    if not cvalid:
        return None
    n = len(ring)
    verts = [tuple(centre)]                         # index 0 = centre
    ridx = {}

    def rid(i):
        if i not in ridx:
            ridx[i] = len(verts)
            verts.append(tuple(ring[i]))
        return ridx[i]

    faces, mats = [], []
    for i in range(n):
        j = (i + 1) % n
        if valid[i] and valid[j]:
            faces.append((0, rid(i), rid(j)))
            mats.append(kind)
    if not faces:
        return None
    if thickness > 0.0 and all(valid):             # clean solid shell
        m = len(verts)                              # = 1 + n
        verts.append(tuple(centre - cnormal * thickness))
        for i in range(n):
            verts.append(tuple(ring[i] - normals[i] * thickness))
        for i in range(n):                          # inner shell, reversed
            j = (i + 1) % n
            faces.append((m, m + 1 + j, m + 1 + i))
            mats.append(kind)
        for i in range(n):                          # side walls
            j = (i + 1) % n
            faces.append((1 + i, 1 + j, m + 1 + j, m + 1 + i))
            mats.append(kind)
    return verts, faces, mats


# Radial subdivision of a curved tile: target curved-surface segment
# length and a per-tile cap.  Small (rim) tiles get a single fan (K = 1,
# identical to the old behavior); only the large near-centre tiles, whose
# flat fan would chord far below the sphere, get extra interior rings.
_RADIAL_STEP = 0.18


_MAX_RADIAL = 6


def _surface_cell(ring_h, centre_h, project, cpt, cnm, bpts, bnms,
                  kind, thickness):
    """One curved tile as a polar grid: concentric radial rings from the
    face centre out to its subdivided boundary, EVERY vertex projected
    onto the surface so the tile follows the curved surface instead of
    chording flat across it.

    A single flat fan (the old cell) chords straight from the projected
    centre to the rim, so a large tile sinks well below the hemisphere
    and meets its neighbors at a deep concave crease -- the dark
    'cracks', worst in the mid-latitudes where the tiles are biggest.
    Sampling interior rings on the surface removes that dip: near a
    shared edge both tiles hug the sphere, so they now meet smoothly.
    The outermost ring is exactly the shared subdivided edge (bpts), so
    adjacent tiles still join edge-for-edge.  With thickness > 0 the tile
    is a closed solid: the whole grid is offset inward along the surface
    normals and closed by perimeter walls."""
    n = len(ring_h)
    dmax = max((float(np.linalg.norm(p - cpt)) for p in bpts), default=0.0)
    K = int(min(_MAX_RADIAL, max(1, np.ceil(dmax / _RADIAL_STEP))))

    # rows 0 (centre) .. K (boundary); interior rows sampled on the
    # geodesic from the centre out to each boundary point, then projected.
    pt_rows = [[cpt]]
    nm_rows = [[cnm]]
    for a in range(1, K):
        t = a / float(K)
        rp, rn = [], []
        for v in ring_h:
            p, nn = project(_normalize_h((1.0 - t) * centre_h + t * v))
            rp.append(p)
            rn.append(nn)
        pt_rows.append(rp)
        nm_rows.append(rn)
    pt_rows.append(list(bpts))
    nm_rows.append(list(bnms))

    def vid(a, i):
        return 0 if a == 0 else 1 + (a - 1) * n + i

    verts = [tuple(cpt)]
    for a in range(1, K + 1):
        for i in range(n):
            verts.append(tuple(pt_rows[a][i]))

    faces, mats = [], []
    for a in range(K):                              # top surface
        for i in range(n):
            j = (i + 1) % n
            faces.append((vid(a, i), vid(a + 1, i), vid(a + 1, j)))
            mats.append(kind)
            if a > 0:                               # the centre ring is a fan
                faces.append((vid(a, i), vid(a + 1, j), vid(a, j)))
                mats.append(kind)

    if thickness > 0.0:                             # closed solid shell
        V = len(verts)
        verts.append(tuple(cpt - cnm * thickness))
        for a in range(1, K + 1):
            for i in range(n):
                verts.append(tuple(pt_rows[a][i]
                                   - nm_rows[a][i] * thickness))

        def bid(a, i):
            return V + (0 if a == 0 else 1 + (a - 1) * n + i)

        for a in range(K):                          # inner shell, reversed
            for i in range(n):
                j = (i + 1) % n
                faces.append((bid(a, i), bid(a + 1, j), bid(a + 1, i)))
                mats.append(kind)
                if a > 0:
                    faces.append((bid(a, i), bid(a, j), bid(a + 1, j)))
                    mats.append(kind)
        for i in range(n):                          # perimeter side walls
            j = (i + 1) % n
            faces.append((vid(K, i), vid(K, j), bid(K, j), bid(K, i)))
            mats.append(kind)
    return verts, faces, mats


def _slab_cell(radius, base, kind, seg=128):
    """Closed backing cylinder for the disk plaque, top at z = 0."""
    verts, faces = [], []
    for z in (0.0, -base):
        for k in range(seg):
            a = 2.0 * pi * k / seg
            verts.append((radius * cos(a), radius * sin(a), z))
    faces.append(tuple(range(seg)))
    faces.append(tuple(reversed(range(seg, 2 * seg))))
    for k in range(seg):
        k2 = (k + 1) % seg
        faces.append((k, seg + k, seg + k2, k2))
    return verts, faces, [kind] * len(faces)


def build_surface_cells(p, q, r, form, model, depth, max_tiles,
                        color_by='SIDES', thickness=0.0, y_cap=6.0,
                        edge_samples=8, hide_off=False):
    """Per-tile 3D cells for a curved surface model (HEMISPHERE or
    PSEUDOSPHERE).  Each face is projected onto the surface and fanned
    from its centre; PSEUDOSPHERE tiles crossing the seam / rim / cusp
    are dropped.  With hide_off (Parity mode only) the 'off' parity
    class is skipped."""
    normals, corners = fundamental_domain(p, q, r)
    faces = tiling_faces(p, q, r, form, depth, max_tiles)
    # Snap shared corners to identical vectors BEFORE the geodesic edge
    # subdivision, so adjacent tiles sample the same edge and join with
    # no cracks on the curved surface.
    faces = _canonicalize_corners(faces)
    colors = _parity_colors(faces) if color_by == 'PARITY' else None
    if model == 'PSEUDOSPHERE':
        shift, mag = _pseudo_setup(corners[1])

        def project(v):
            return _to_pseudosphere(v, shift, mag, y_cap)
    else:                                          # HEMISPHERE
        def project(v):
            return _to_hemisphere(v)

    _zero = np.zeros(3)
    cells = []
    for i, (fp, nsides, parity) in enumerate(faces):
        if colors is not None:
            parity = colors[i]
        if hide_off and colors is not None and parity != 0:
            continue
        ring = _face_boundary(fp, model, edge_samples)
        pn = [project(v) for v in ring]
        valid = [r_ is not None for r_ in pn]
        pts = [r_[0] if r_ else _zero for r_ in pn]
        nrm = [r_[1] if r_ else _zero for r_ in pn]
        centre_h = _normalize_h(np.sum(fp, axis=0))
        cres = project(centre_h)
        kind = _kind_for(nsides, parity, color_by)
        if model == 'HEMISPHERE' and cres is not None:
            # Polar-grid the tile so it hugs the sphere instead of
            # chording flat across it; this is what closes the
            # mid-latitude cracks between large tiles.
            cell = _surface_cell(ring, centre_h, project, cres[0],
                                 cres[1], pts, nrm, kind, thickness)
        else:
            # PSEUDOSPHERE (and any off-surface centre): the original
            # single-fan cell, with its seam / rim / cusp clipping.
            cell = _fan_cell(
                pts, nrm, valid,
                cres[0] if cres else _zero, cres[1] if cres else _zero,
                cres is not None, kind, thickness)
        if cell is not None:
            cells.append(cell)
    return cells


FORM_ITEMS = [
    ('REGULAR', "Regular {p,q}",
     "The regular tiling: p-gons, q around each vertex"),
    ('TRUNCATED', "Truncated t{p,q}",
     "Truncation: 2p-gons and q-gons"),
    ('RECTIFIED', "Rectified r{p,q}",
     "Quasiregular: p-gons and q-gons alternating at each vertex"),
    ('BITRUNCATED', "Bitruncated t{q,p}",
     "Truncation of the dual: p-gons and 2q-gons"),
    ('CANTELLATED', "Cantellated rr{p,q}",
     "p-gons, q-gons and 2r-gons (squares when r = 2)"),
    ('OMNITRUNCATED', "Omnitruncated tr{p,q}",
     "2p-gons, 2q-gons and 2r-gons"),
    ('SNUB', "Snub sr{p,q}",
     "Chiral snub: p-gons, q-gons and snub triangles, no mirrors"),
]
