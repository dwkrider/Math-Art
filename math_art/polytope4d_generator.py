
# 4D Regular Polytope Generator for Blender
#
# The six regular convex 4-polytopes -- 5-cell, tesseract (8-cell),
# 16-cell, 24-cell, 600-cell and 120-cell -- rendered as edge
# frameworks in 3D. Edges are either
#
#   STRAIGHT: a perspective projection from 4-space (large distance
#             approaches orthographic, small distance approaches a
#             Schlegel diagram), or
#   CURVED:   vertices are placed on the unit 3-sphere, edges follow
#             great-circle arcs, and the whole picture is mapped by
#             stereographic projection -- every edge becomes a circular
#             arc, the classic "hyperbolic-looking" rendering.
#
# Struts can taper with the local projection scale (near features fat,
# far features thin), and vertices can be capped with spheres.
#
# References:
# - The six regular convex 4-polytopes: Ludwig Schlafli (c. 1852).
# - H. S. M. Coxeter, "Regular Polytopes".
# - Schlegel diagrams: Victor Schlegel (1883).

bl_info = {
    "name": "4D Polytopes",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > 4D Polytope",
    "description": "Tesseract, 120-cell and friends, straight or "
                   "stereographically curved edges",
    "category": "Add Mesh",
}

import math
import itertools
from math import cos, sin, pi, sqrt

PHI = (1 + 5 ** 0.5) / 2


# --------------------------------------------------------------------------
# Vertex constructions (standard coordinates)
# --------------------------------------------------------------------------

def _perms_even(coords):
    out = set()
    for p in itertools.permutations(range(4)):
        parity = 0
        q = list(p)
        for i in range(4):
            while q[i] != i:
                j = q[i]
                q[i], q[j] = q[j], q[i]
                parity += 1
        if parity % 2 == 0:
            out.add(tuple(coords[i] for i in p))
    return out


def _perms_all(coords):
    return set(itertools.permutations(coords))


def _signs(base):
    out = set()
    for v in base:
        idx = [i for i in range(4) if abs(v[i]) > 1e-12]
        for signs in itertools.product((1, -1), repeat=len(idx)):
            w = list(v)
            for k, i in enumerate(idx):
                w[i] = abs(w[i]) * signs[k]
            out.add(tuple(w))
    return out


def polytope_vertices(kind):
    if kind == 'CELL5':
        s5 = sqrt(5)
        V = [(1, 1, 1, -1 / s5), (1, -1, -1, -1 / s5),
             (-1, 1, -1, -1 / s5), (-1, -1, 1, -1 / s5),
             (0, 0, 0, s5 - 1 / s5)]
    elif kind == 'CELL8':
        V = list(itertools.product((-1, 1), repeat=4))
    elif kind == 'CELL16':
        V = list(_signs(_perms_all((1, 0, 0, 0))))
    elif kind == 'CELL24':
        V = list(_signs(_perms_all((1, 1, 0, 0))))
    elif kind == 'CELL600':
        V = set()
        V |= _signs({(0.5, 0.5, 0.5, 0.5)})
        V |= _signs(_perms_all((1, 0, 0, 0)))
        V |= _signs(_perms_even((PHI / 2, 0.5, 1 / (2 * PHI), 0)))
        V = list(V)
    elif kind == 'CELL120':
        s5 = sqrt(5)
        V = set()
        V |= _signs(_perms_all((2, 2, 0, 0)))
        V |= _signs(_perms_all((s5, 1, 1, 1)))
        V |= _signs(_perms_all((PHI, PHI, PHI, PHI ** -2)))
        V |= _signs(_perms_all((PHI ** 2, PHI ** -1, PHI ** -1, PHI ** -1)))
        V |= _signs(_perms_even((PHI ** 2, PHI ** -2, 1, 0)))
        V |= _signs(_perms_even((s5, PHI ** -1, PHI, 0)))
        V |= _signs(_perms_even((2, 1, PHI, PHI ** -1)))
        V = list(V)
    else:
        raise ValueError(kind)
    # normalise onto the unit 3-sphere (vertex-transitive: equal norms)
    out = []
    for v in V:
        n = sqrt(sum(x * x for x in v))
        out.append(tuple(x / n for x in v))
    return out


def _in_flat(u, v, w, x, tol=1e-6):
    """Is x in the 2-flat of R^4 through u, v, w?"""
    a = [v[k] - u[k] for k in range(4)]
    b = [w[k] - u[k] for k in range(4)]
    c = [x[k] - u[k] for k in range(4)]
    # Gram-Schmidt: remove span(a, b) from c
    la = math.sqrt(sum(t * t for t in a)) or 1.0
    a = [t / la for t in a]
    d = sum(b[k] * a[k] for k in range(4))
    b = [b[k] - d * a[k] for k in range(4)]
    lb = math.sqrt(sum(t * t for t in b)) or 1.0
    b = [t / lb for t in b]
    for e in (a, b):
        d = sum(c[k] * e[k] for k in range(4))
        c = [c[k] - d * e[k] for k in range(4)]
    return math.sqrt(sum(t * t for t in c)) < tol


_FACE_CACHE = {}

# every regular 4-polytope has one face type; filtering on its size
# rejects other planar edge cycles (equators, vertex figures)
_FACE_SIZE = {'CELL5': 3, 'CELL8': 4, 'CELL16': 3, 'CELL24': 3,
              'CELL120': 5, 'CELL600': 3}


def polytope_faces(kind, V, E, cache_key=None):
    """The 2D faces (polygon vertex cycles) of the polytope: walks
    along edges staying inside a common 2-flat. cache_key names the
    vertex ordering (defaults to kind; dual orderings pass their
    own key so they do not collide with the primal cache)."""
    ckey = cache_key or kind
    if ckey in _FACE_CACHE:
        return _FACE_CACHE[ckey]
    want = _FACE_SIZE[kind]
    from collections import defaultdict
    adj = defaultdict(set)
    for i, j in E:
        adj[i].add(j)
        adj[j].add(i)
    seen = set()
    faces = []
    for u in sorted(adj):
        for v in adj[u]:
            for w in adj[v]:
                if w == u:
                    continue
                cyc = [u, v, w]
                closed = False
                while len(cyc) <= want:
                    prev, cur = cyc[-2], cyc[-1]
                    nxt = None
                    for x in adj[cur]:
                        if x != prev and _in_flat(V[u], V[v], V[w],
                                                  V[x]):
                            nxt = x
                            break
                    if nxt is None:
                        break
                    if nxt == u:
                        closed = True
                        break
                    cyc.append(nxt)
                if closed and len(cyc) == want:
                    key = frozenset(cyc)
                    if key not in seen:
                        seen.add(key)
                        faces.append(cyc)
    _FACE_CACHE[ckey] = faces
    return faces


def polytope_edges(V):
    """Edges = vertex pairs at the minimal nonzero distance."""
    n = len(V)
    d2min = None
    for i in range(n):
        for j in range(i + 1, n):
            d2 = sum((V[i][k] - V[j][k]) ** 2 for k in range(4))
            if d2 > 1e-9 and (d2min is None or d2 < d2min):
                d2min = d2
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d2 = sum((V[i][k] - V[j][k]) ** 2 for k in range(4))
            if abs(d2 - d2min) < 1e-6:
                edges.append((i, j))
    return edges


COUNTS = {'CELL5': (5, 10), 'CELL8': (16, 32), 'CELL16': (8, 24),
          'CELL24': (24, 96), 'CELL600': (120, 720),
          'CELL120': (600, 1200)}


# --------------------------------------------------------------------------
# Duals, equatorial cutaway, Hopf rings (after Segerman, Visualizing
# Mathematics with 3D Printing, figs 3-20/22/23, 3-25, 3-29)
# --------------------------------------------------------------------------

DUAL_KIND = {'CELL5': 'CELL5', 'CELL8': 'CELL16', 'CELL16': 'CELL8',
             'CELL24': 'CELL24', 'CELL120': 'CELL600',
             'CELL600': 'CELL120'}

# vertices per (3D) cell of each polytope
_CELL_SIZE = {'CELL5': 4, 'CELL8': 8, 'CELL16': 4, 'CELL24': 6,
              'CELL120': 20, 'CELL600': 4}

HALF_TOL = 1e-6


def dual_vertices(kind):
    """Unit-sphere vertices of the dual polytope, oriented so every
    dual vertex points at a cell center of polytope_vertices(kind).
    For 8<->16 and 120<->600 the standard coordinate sets are already
    mutually dual-aligned (checked numerically); the 5-cell dual is
    the antipodal copy and the 24-cell dual is the rotated copy with
    16-cell + tesseract vertices. Returns (verts, dual_kind, key)
    where key names the vertex ordering for the face cache."""
    if kind == 'CELL5':
        V = [tuple(-x for x in v) for v in polytope_vertices('CELL5')]
        return V, 'CELL5', 'CELL5_DUAL'
    if kind == 'CELL24':
        S = set()
        S |= _signs(_perms_all((1, 0, 0, 0)))
        S |= _signs({(0.5, 0.5, 0.5, 0.5)})
        V = []
        for v in S:
            n = sqrt(sum(x * x for x in v))
            V.append(tuple(x / n for x in v))
        return V, 'CELL24', 'CELL24_DUAL'
    dk = DUAL_KIND[kind]
    return polytope_vertices(dk), dk, dk


def _cell_inradius(kind):
    """Distance from the origin to a cell center, circumradius 1:
    the norm of the centroid of one cell's vertices (the cell is the
    _CELL_SIZE[kind] vertices nearest a dual vertex direction)."""
    V = polytope_vertices(kind)
    d = dual_vertices(kind)[0][0]
    k = _CELL_SIZE[kind]
    idx = sorted(range(len(V)),
                 key=lambda i: -sum(V[i][t] * d[t]
                                    for t in range(4)))[:k]
    c = [sum(V[i][t] for i in idx) / k for t in range(4)]
    return sqrt(sum(x * x for x in c))


def _half_filter(V, E, F2, tol=HALF_TOL):
    """Equatorial cutaway: keep vertices with w <= tol (equator
    included) plus the edges/faces whose vertices all survive, and
    reindex. Applied in the polytope's own coordinates, before any
    rotation or projection."""
    keep = [i for i, v in enumerate(V) if v[3] <= tol]
    remap = {i: n for n, i in enumerate(keep)}
    V2 = [V[i] for i in keep]
    E2 = [(remap[i], remap[j]) for (i, j) in E
          if i in remap and j in remap]
    F22 = None
    if F2 is not None:
        F22 = [[remap[i] for i in cyc] for cyc in F2
               if all(i in remap for i in cyc)]
    return V2, E2, F22


def _qmul(a, b):
    """Quaternion product; component 0 is the real part."""
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def _qorder(q, maxn=12):
    """Multiplicative order of a unit quaternion (0 if > maxn)."""
    p = q
    for k in range(1, maxn + 1):
        if (abs(p[0] - 1.0) < 1e-9 and abs(p[1]) < 1e-9
                and abs(p[2]) < 1e-9 and abs(p[3]) < 1e-9):
            return k
        p = _qmul(p, q)
    return 0


def _order10_generator(centers):
    """An order-10 element of the binary icosahedral group 2I."""
    g = (PHI / 2, 0.5, 1 / (2 * PHI), 0.0)
    if _qorder(g) == 10:
        return g
    for c in sorted(centers):            # deterministic fallback
        if _qorder(c) == 10:
            return c
    raise RuntimeError("no order-10 element found in 2I")


def hopf_ring_cosets():
    """The 120 cell centers of the 120-cell -- equivalently the
    600-cell's vertices, the unit icosians 2I -- partitioned into 12
    Hopf rings of 10 cells: the left cosets h<g> of a cyclic
    subgroup of order 10. Returns (centers, rings); each ring lists
    10 center indices in chain order, so consecutive centers (with
    wraparound) are adjacent cells at the minimal center distance.
    Ring 0 passes through the identity quaternion and the rest are
    sorted farthest-from-ring-0 first, so two rings give the classic
    pair of interlocked orthogonal rings (fig 3-29)."""
    centers = polytope_vertices('CELL600')
    g = _order10_generator(centers)
    gpow = [(1.0, 0.0, 0.0, 0.0)]
    for _ in range(9):
        gpow.append(_qmul(gpow[-1], g))

    def nearest(q):
        bi, bd = 0, -2.0
        for i, c in enumerate(centers):
            d = sum(q[t] * c[t] for t in range(4))
            if d > bd:
                bd, bi = d, i
        return bi, bd

    assigned = [False] * len(centers)
    rings = []
    seeds = sorted(range(len(centers)),
                   key=lambda i: tuple(-x for x in centers[i]))
    for s in seeds:
        if assigned[s]:
            continue
        ring = []
        for gp in gpow:
            i, d = nearest(_qmul(centers[s], gp))
            if d < 1.0 - 1e-9 or assigned[i]:
                raise RuntimeError("coset walk left the group")
            assigned[i] = True
            ring.append(i)
        rings.append(ring)
    ring0 = rings[0]                     # contains (1, 0, 0, 0)

    def mind2(ring):
        return min(sum((centers[i][t] - centers[j][t]) ** 2
                       for t in range(4))
                   for i in ring for j in ring0)

    rest = sorted(rings[1:], key=lambda r: (-mind2(r),
                                            tuple(sorted(r))))
    return centers, [ring0] + rest


def ring_cell_points(n_rings, cell_scale, half=False):
    """4D vertex sets of the shrunken dodecahedral cells of the
    first n_rings Hopf rings of the 120-cell. Each cell is the 20
    polytope vertices nearest its center, scaled toward their 4D
    centroid. Returns a list of (ring_index, [20 4D points])."""
    V = polytope_vertices('CELL120')
    centers, rings = hopf_ring_cosets()
    out = []
    for ri in range(min(n_rings, len(rings))):
        for ci in rings[ri]:
            c = centers[ci]
            if half and c[3] > HALF_TOL:
                continue
            idx = sorted(range(len(V)),
                         key=lambda i: -sum(V[i][t] * c[t]
                                            for t in range(4)))[:20]
            cen = [sum(V[i][t] for i in idx) / 20 for t in range(4)]
            out.append((ri, [tuple(cen[t]
                                   + (V[i][t] - cen[t]) * cell_scale
                                   for t in range(4)) for i in idx]))
    return out


# --------------------------------------------------------------------------
# 4D rotation and projection
# --------------------------------------------------------------------------

def rotate4(V, xw, yw, zw, xy):
    """Rotations in the XW, YW, ZW and XY planes (degrees)."""
    out = [list(v) for v in V]
    for (a, b, ang) in ((0, 3, xw), (1, 3, yw), (2, 3, zw), (0, 1, xy)):
        t = math.radians(ang)
        if abs(t) < 1e-12:
            continue
        c, s = cos(t), sin(t)
        for v in out:
            va, vb = v[a], v[b]
            v[a] = va * c - vb * s
            v[b] = va * s + vb * c
    return [tuple(v) for v in out]


def _slerp4(a, b, t):
    d = max(-1.0, min(1.0, sum(a[k] * b[k] for k in range(4))))
    om = math.acos(d)
    if om < 1e-9:
        return a
    sa = sin((1 - t) * om) / sin(om)
    sb = sin(t * om) / sin(om)
    return tuple(a[k] * sa + b[k] * sb for k in range(4))


def project_point(v, dist):
    """(x,y,z,w) -> R^3 by central projection from (0,0,0,dist).
    With dist = 1 and v on the unit 3-sphere this is the exact
    stereographic projection (injective, arcs -> circles).
    Returns (point3, local_scale)."""
    denom = max(dist - v[3], 0.02)
    s = 1.0 / denom
    return (v[0] * s, v[1] * s, v[2] * s), s


def _pole_angles(V):
    """XW/YW angles of the deterministic extra 4D rotation that
    moves every vertex away from the stereographic pole (w ~ 1), or
    (0, 0) if none is needed. Exposing the angles lets the same
    rotation be applied to auxiliary point sets (dual framework,
    ring cells) so a compound stays aligned."""
    def max_w(vs):
        return max(v[3] for v in vs)
    if max_w(V) < 0.93:
        return 0.0, 0.0
    best = (1.7, 1.1)
    best_w = 2.0
    for k in range(1, 80):
        ang = (k * 1.7, k * 1.1)
        w = max_w(rotate4(V, ang[0], ang[1], 0.0, 0.0))
        if w < best_w:
            best_w = w
            best = ang
        if w < 0.9:
            break
    return best


def clear_pole(V):
    """If any vertex sits near the stereographic pole (w ~ 1), apply a
    deterministic extra 4D rotation that moves every vertex away from
    it (otherwise that vertex would project to infinity)."""
    a, b = _pole_angles(V)
    if a == 0.0 and b == 0.0:
        return V, False
    return rotate4(V, a, b, 0.0, 0.0), True


# --------------------------------------------------------------------------
# Strut / sphere mesh helpers
# --------------------------------------------------------------------------

def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _unit3(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return (v[0] / l, v[1] / l, v[2] / l)


def add_strut(verts, faces, pts, radii, sides):
    """Closed tube along a polyline with per-point radii."""
    n = len(pts)
    rings = []
    prev_n = None
    for i in range(n):
        if i == 0:
            t = _unit3(_sub3(pts[1], pts[0]))
        elif i == n - 1:
            t = _unit3(_sub3(pts[-1], pts[-2]))
        else:
            t = _unit3(_sub3(pts[i + 1], pts[i - 1]))
        if prev_n is None:
            ref = (0.0, 0.0, 1.0) if abs(t[2]) < 0.9 else (1.0, 0.0, 0.0)
            u = _unit3(_cross3(t, ref))
        else:
            u = _unit3(_cross3(t, _cross3(prev_n, t)))
            # re-project previous normal for a stable frame
            d = sum(prev_n[k] * t[k] for k in range(3))
            u = _unit3(tuple(prev_n[k] - d * t[k] for k in range(3)))
        w = _cross3(t, u)
        prev_n = u
        ring = []
        for s in range(sides):
            a = 2 * pi * s / sides
            ring.append(len(verts))
            verts.append(tuple(pts[i][k]
                               + radii[i] * (cos(a) * u[k] + sin(a) * w[k])
                               for k in range(3)))
        rings.append(ring)
    for i in range(n - 1):
        r0, r1 = rings[i], rings[i + 1]
        for s in range(sides):
            s2 = (s + 1) % sides
            faces.append([r0[s], r0[s2], r1[s2], r1[s]])
    faces.append(list(reversed(rings[0])))
    faces.append(list(rings[-1]))


def add_sphere(verts, faces, center, radius, seg=8, rings=6):
    base = len(verts)
    verts.append((center[0], center[1], center[2] + radius))
    for r in range(1, rings):
        th = pi * r / rings
        for s in range(seg):
            a = 2 * pi * s / seg
            verts.append((center[0] + radius * sin(th) * cos(a),
                          center[1] + radius * sin(th) * sin(a),
                          center[2] + radius * cos(th)))
    verts.append((center[0], center[1], center[2] - radius))
    last = len(verts) - 1
    ring0 = lambda r: base + 1 + (r - 1) * seg
    for s in range(seg):
        s2 = (s + 1) % seg
        faces.append([base, ring0(1) + s2, ring0(1) + s])
    for r in range(1, rings - 1):
        for s in range(seg):
            s2 = (s + 1) % seg
            faces.append([ring0(r) + s, ring0(r) + s2,
                          ring0(r + 1) + s2, ring0(r + 1) + s])
    for s in range(seg):
        s2 = (s + 1) % seg
        faces.append([last, ring0(rings - 1) + s, ring0(rings - 1) + s2])


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def _newell(poly):
    n = [0.0, 0.0, 0.0]
    m = len(poly)
    for i in range(m):
        p, q = poly[i], poly[(i + 1) % m]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    ln = math.sqrt(sum(t * t for t in n)) or 1.0
    return [t / ln for t in n]


def _leonardo_panels(F2, proj, origin, border, panel_thickness,
                     taper, scale):
    """Mitered da Vinci panels for the projected 2D faces. Every
    polytope vertex is offset once, along the average normal of all
    panels meeting there (even-thickness corrected), so the inner
    boundaries of adjacent panels share their vertices: joints along
    edges and at corners are exact. Rim walls are emitted once per
    polytope edge. Windings are consistent by construction (outer
    face CCW seen from outside)."""
    faces_o = []                          # (cyc, poly, n) oriented
    vsum = {}
    vcnt = {}
    for cyc in F2:
        poly = [proj[i][0] for i in cyc]
        n = _newell(poly)
        c = [sum(p[k] for p in poly) / len(poly) for k in range(3)]
        if sum(n[k] * (c[k] - origin[k]) for k in range(3)) < 0:
            cyc = list(reversed(cyc))
            poly = list(reversed(poly))
            n = [-t for t in n]
        faces_o.append((cyc, poly, n))
        for i in cyc:
            s = vsum.setdefault(i, [0.0, 0.0, 0.0])
            for k in range(3):
                s[k] += n[k]
            vcnt[i] = vcnt.get(i, 0) + 1
    # shared per-vertex inward offset (mitre): least-squares solve of
    # m . n_f = thickness over all panels at the vertex -- exact at
    # corners where three planes meet, balanced elsewhere, and the
    # offset point stays inside the local face wedge (no flaps)
    vnormals = {}
    for (cyc, poly, n) in faces_o:
        for i in cyc:
            vnormals.setdefault(i, []).append(n)
    voff = {}
    for i, ns in vnormals.items():
        th = panel_thickness * scale * (proj[i][1] if taper else 1.0)
        M = [[0.0] * 3 for _ in range(3)]
        b = [0.0, 0.0, 0.0]
        for n in ns:
            for r in range(3):
                b[r] += n[r] * th
                for c in range(3):
                    M[r][c] += n[r] * n[c]
        lam = 1e-6 * (M[0][0] + M[1][1] + M[2][2] + 1e-12)
        for r in range(3):
            M[r][r] += lam
        det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
               - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
               + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
        if abs(det) < 1e-12:
            s = vsum[i]
            ln = math.sqrt(sum(t * t for t in s)) or 1.0
            voff[i] = [s[k] / ln * th for k in range(3)]
            continue
        m = []
        for c in range(3):
            Mc = [row[:] for row in M]
            for r in range(3):
                Mc[r][c] = b[r]
            dc = (Mc[0][0] * (Mc[1][1] * Mc[2][2]
                              - Mc[1][2] * Mc[2][1])
                  - Mc[0][1] * (Mc[1][0] * Mc[2][2]
                                - Mc[1][2] * Mc[2][0])
                  + Mc[0][2] * (Mc[1][0] * Mc[2][1]
                                - Mc[1][1] * Mc[2][0]))
            m.append(dc / det)
        # guard against runaway mitres at very shallow wedges
        ln = math.sqrt(sum(t * t for t in m))
        cap = 3.0 * th
        if ln > cap:
            m = [t * cap / ln for t in m]
        voff[i] = m
    verts = []
    faces = []

    def _tri_n(a, b, c):
        u = [verts[b][k] - verts[a][k] for k in range(3)]
        v = [verts[c][k] - verts[a][k] for k in range(3)]
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
        ln = math.sqrt(sum(t * t for t in n)) or 1.0
        return [t / ln for t in n]

    def quad_tri(i0, i1, i2, i3):
        """Split a (possibly non-planar) quad along the diagonal
        whose two triangles fold the least, keeping the winding."""
        fold_a = sum(x * y for x, y in zip(_tri_n(i0, i1, i2),
                                           _tri_n(i0, i2, i3)))
        fold_b = sum(x * y for x, y in zip(_tri_n(i0, i1, i3),
                                           _tri_n(i1, i2, i3)))
        if fold_a >= fold_b:
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])
        else:
            faces.append([i0, i1, i3])
            faces.append([i1, i2, i3])

    OUT = {}
    INN = {}
    for i in vsum:
        OUT[i] = len(verts)
        verts.append(proj[i][0])
        INN[i] = len(verts)
        verts.append(tuple(proj[i][0][k] - voff[i][k]
                           for k in range(3)))
    rim_done = set()
    for (cyc, poly, n) in faces_o:
        m = len(cyc)
        c = [sum(p[k] for p in poly) / m for k in range(3)]
        avg = sum(proj[i][1] for i in cyc) / m
        th_f = panel_thickness * scale * (avg if taper else 1.0)
        HO = len(verts)
        for p in poly:
            verts.append(tuple(c[k] + (p[k] - c[k]) * (1 - border)
                               for k in range(3)))
        # the hole ring is not shared with any neighbour, so it can
        # follow the face normal exactly: planar hole walls
        HI = len(verts)
        for p in poly:
            verts.append(tuple(c[k] + (p[k] - c[k]) * (1 - border)
                               - n[k] * th_f for k in range(3)))
        for i in range(m):
            j = (i + 1) % m
            a, b = cyc[i], cyc[j]
            faces.append([OUT[a], OUT[b], HO + j, HO + i])
            quad_tri(HI + i, HI + j, INN[b], INN[a])
            faces.append([HO + j, HO + i, HI + i, HI + j])
            key = (min(a, b), max(a, b))
            if key not in rim_done:
                rim_done.add(key)
                quad_tri(OUT[b], OUT[a], INN[a], INN[b])
    return verts, faces


def build_polytope_ex(kind='CELL8', style='CURVED', proj_dist=1.05,
                      rot_xw=0.0, rot_yw=0.0, rot_zw=0.0, rot_xy=0.0,
                      arc_segments=12, radius=0.03, sides=6,
                      taper=True, vertex_spheres=True,
                      sphere_factor=1.6, scale=1.0, render='EDGES',
                      border=0.35, panel_thickness=0.03, half=False,
                      dual_compound=False, rings=0,
                      ring_cell_scale=0.9, rings_only=False):
    """Superset of build_polytope: equatorial cutaway ('half'), the
    primal + dual compound, and solid Hopf rings of the 120-cell
    (rings need bmesh, so they only build inside Blender).
    Returns (verts, faces, face_mat, stats): face_mat gives one
    material-slot index per face (0 primal, 1 dual if present, then
    one slot per ring); stats is a dict of element counts."""
    systems = []                        # (V4, E, F2) per framework
    V4 = polytope_vertices(kind)
    E = polytope_edges(V4)
    F2 = (polytope_faces(kind, V4, E) if render == 'LEONARDO'
          else None)
    if half:
        V4, E, F2 = _half_filter(V4, E, F2)
    systems.append((V4, E, F2))
    if dual_compound:
        D4, dkind, dkey = dual_vertices(kind)
        ED = polytope_edges(D4)
        FD = (polytope_faces(dkind, D4, ED, cache_key=dkey)
              if render == 'LEONARDO' else None)
        if style != 'CURVED':
            # dual vertices sit at the primal's cell centers; in
            # curved mode both frameworks live on the unit 3-sphere
            r = _cell_inradius(kind)
            D4 = [tuple(x * r for x in v) for v in D4]
        if half:
            D4, ED, FD = _half_filter(D4, ED, FD)
        systems.append((D4, ED, FD))
    cells = []
    if rings > 0 and kind == 'CELL120' and _IN_BLENDER:
        cells = ring_cell_points(rings, ring_cell_scale, half)
    if rings_only and cells:
        # the book's fig 3-29 look: rings of dodecahedra standing
        # alone, without the 1200-strut edge framework around them
        systems = []
    # the same 4D rotation for every point set, so compounds and
    # rings stay aligned with the primal framework
    systems = [(rotate4(V, rot_xw, rot_yw, rot_zw, rot_xy), Es, Fs)
               for (V, Es, Fs) in systems]
    cells = [(ri, rotate4(P, rot_xw, rot_yw, rot_zw, rot_xy))
             for (ri, P) in cells]
    if style == 'CURVED':
        # exact stereographic projection: pole ON the unit 3-sphere
        dist = 1.0
        allv = ([v for (V, _e, _f) in systems for v in V]
                or [p for (_ri, P) in cells for p in P])
        pa, pb = _pole_angles(allv)
        if pa != 0.0 or pb != 0.0:
            systems = [(rotate4(V, pa, pb, 0.0, 0.0), Es, Fs)
                       for (V, Es, Fs) in systems]
            cells = [(ri, rotate4(P, pa, pb, 0.0, 0.0))
                     for (ri, P) in cells]
    else:
        dist = max(proj_dist, 1.001)
    verts = []
    faces = []
    edges = []                              # only the Wireframe render
    face_mat = []
    for mi, (V, Es, Fs) in enumerate(systems):
        proj = {}
        for i, v in enumerate(V):
            p, s = project_point(v, dist)
            proj[i] = (tuple(c * scale for c in p), s)
        nf0 = len(faces)
        if render == 'LEONARDO':
            # flat panels per 2D face: stereographic projection maps
            # the circle through a face's vertices to a circle in
            # R^3, so every projected face is planar in both styles
            nvp = len(V)
            O = tuple(sum(proj[i][0][k] for i in range(nvp)) / nvp
                      for k in range(3))
            pv, pf = _leonardo_panels(Fs, proj, O, border,
                                      panel_thickness, taper,
                                      scale)
            base = len(verts)
            verts.extend(pv)
            faces.extend([[base + i for i in f] for f in pf])
        else:
            for (i, j) in Es:
                if style == 'CURVED':
                    pts = []
                    scls = []
                    for k in range(arc_segments + 1):
                        t = k / arc_segments
                        q = _slerp4(V[i], V[j], t)
                        p, s = project_point(q, dist)
                        pts.append(tuple(c * scale for c in p))
                        scls.append(s)
                else:
                    pts = [proj[i][0], proj[j][0]]
                    scls = [proj[i][1], proj[j][1]]
                    if arc_segments > 1:
                        # subdivide straight edges too (for tapering)
                        a, b = pts
                        sa, sb = scls
                        pts = [tuple(a[k]
                                     + (b[k] - a[k]) * t / arc_segments
                                     for k in range(3))
                               for t in range(arc_segments + 1)]
                        scls = [sa + (sb - sa) * t / arc_segments
                                for t in range(arc_segments + 1)]
                if render == 'WIREFRAME':
                    # bare polyline along the (possibly curved) edge
                    base = len(verts)
                    verts.extend(pts)
                    for a in range(len(pts) - 1):
                        edges.append((base + a, base + a + 1))
                    continue
                if taper:
                    radii = [radius * s * scale for s in scls]
                else:
                    radii = [radius * scale] * len(pts)
                add_strut(verts, faces, pts, radii, sides)
            if render == 'BALLSTICK':
                for i in range(len(V)):
                    p, s = proj[i]
                    r = (radius * sphere_factor
                         * (s if taper else 1.0) * scale)
                    add_sphere(verts, faces, p, r)
        face_mat.extend([mi] * (len(faces) - nf0))
    ring_base = len(systems)
    n_cells = 0
    for (ri, P) in cells:
        pts = [tuple(c * scale for c in project_point(p, dist)[0])
               for p in P]
        hv, hf = _hull_geometry(pts)
        base = len(verts)
        verts.extend(hv)
        faces.extend([[base + i for i in f] for f in hf])
        face_mat.extend([ring_base + ri] * len(hf))
        n_cells += 1
    stats = {'nv': (len(systems[0][0]) if systems else 0),
             'ne': (len(systems[0][1]) if systems else 0),
             'dual_nv': (len(systems[1][0])
                         if dual_compound and len(systems) > 1 else 0),
             'dual_ne': (len(systems[1][1])
                         if dual_compound and len(systems) > 1 else 0),
             'n_systems': len(systems), 'n_cells': n_cells,
             'n_rings': (min(rings, 12) if cells else 0),
             'wire_edges': edges}
    return verts, faces, face_mat, stats


def build_polytope(kind='CELL8', style='CURVED', proj_dist=1.05,
                   rot_xw=0.0, rot_yw=0.0, rot_zw=0.0, rot_xy=0.0,
                   arc_segments=12, radius=0.03, sides=6, taper=True,
                   vertex_spheres=True, sphere_factor=1.6, scale=1.0,
                   render='EDGES', border=0.35, panel_thickness=0.03):
    """Original interface, kept for compatibility: the plain single
    framework with no cutaway, compound or rings."""
    verts, faces, _mat, st = build_polytope_ex(
        kind, style, proj_dist, rot_xw, rot_yw, rot_zw, rot_xy,
        arc_segments, radius, sides, taper, vertex_spheres,
        sphere_factor, scale, render, border, panel_thickness)
    return verts, faces, st['nv'], st['ne']


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (FloatProperty, EnumProperty, IntProperty,
                           BoolProperty)
    import bmesh
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _hull_geometry(pts):
        """Convex hull (verts, faces) of a 3D point cloud via bmesh.
        Each projected dodecahedral ring cell is convex (central
        projection of a convex cell from a point beyond it), so its
        hull is exactly the solid cell."""
        bm = bmesh.new()
        for p in pts:
            bm.verts.new(p)
        bmesh.ops.convex_hull(bm, input=bm.verts[:])
        bm.verts.index_update()
        used = sorted({v.index for f in bm.faces for v in f.verts})
        remap = {vi: n for n, vi in enumerate(used)}
        bm.verts.ensure_lookup_table()
        hv = [tuple(bm.verts[vi].co) for vi in used]
        hf = [[remap[v.index] for v in f.verts] for f in bm.faces]
        bm.free()
        return hv, hf

    def _make_material(name, rgba):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = rgba
        mat.use_nodes = True
        node = mat.node_tree.nodes.get('Principled BSDF')
        if node is not None:
            node.inputs['Base Color'].default_value = rgba
        return mat

    def _ring_color(ri):
        """Distinct hue per Hopf ring (12 around the color wheel)."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb((ri / 12.0) % 1.0, 0.75, 0.9)
        return (r, g, b, 1.0)

    class MESH_OT_polytope4d_add(bpy.types.Operator):
        """Regular 4-polytope edge framework, projected to 3D with
        straight or stereographically curved edges"""
        bl_idname = "mesh.polytope4d_add"
        bl_label = "4D Polytope"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Polytope",
            items=[('CELL5', "5-cell", "4-simplex: 5 vertices, 10 edges"),
                   ('CELL8', "Tesseract (8-cell)", "16 vertices, 32 edges"),
                   ('CELL16', "16-cell", "8 vertices, 24 edges"),
                   ('CELL24', "24-cell", "24 vertices, 96 edges"),
                   ('CELL120', "120-cell", "600 vertices, 1200 edges"),
                   ('CELL600', "600-cell", "120 vertices, 720 edges")],
            default='CELL8')
        style: EnumProperty(
            name="Edges",
            items=[('CURVED', "Curved (stereographic)",
                    "Vertices on the 3-sphere, edges as great-circle "
                    "arcs, stereographically projected: circular arcs"),
                   ('STRAIGHT', "Straight (perspective)",
                    "Direct 4D perspective projection; small distance "
                    "approaches a Schlegel diagram")],
            default='CURVED')
        proj_dist: FloatProperty(
            name="Projection Distance", default=1.05, min=1.001, max=10.0,
            description="Eye distance along w for STRAIGHT edges (near 1 "
                        "= Schlegel diagram; for the 5/16/600-cell a "
                        "vertex sits at w=1, so rotate a little or back "
                        "off the distance). Curved mode always uses the "
                        "exact stereographic projection")
        rot_xw: FloatProperty(name="Rotate XW", default=0.0,
                              min=-180.0, max=180.0)
        rot_yw: FloatProperty(name="Rotate YW", default=0.0,
                              min=-180.0, max=180.0)
        rot_zw: FloatProperty(name="Rotate ZW", default=0.0,
                              min=-180.0, max=180.0)
        rot_xy: FloatProperty(name="Rotate XY", default=0.0,
                              min=-180.0, max=180.0)
        arc_segments: IntProperty(
            name="Arc Segments", default=12, min=1, max=48,
            description="Samples per edge (curved edges and tapering)")
        radius: FloatProperty(name="Strut Radius", default=0.03,
                              min=0.002, max=0.5, step=1, precision=3)
        sides: IntProperty(name="Strut Sides", default=6, min=3, max=16)
        taper: BoolProperty(
            name="Taper With Projection", default=True,
            description="Scale strut thickness by the local projection "
                        "factor (near-the-pole features fatter)")
        vertex_spheres: BoolProperty(name="Vertex Spheres", default=True)
        sphere_factor: FloatProperty(name="Sphere Size", default=1.6,
                                     min=1.0, max=4.0)
        render: EnumProperty(
            name="Style",
            items=[('EDGES', "Struts",
                    "Solid tubes along the projected edges (no vertex "
                    "spheres) -- the same edge-strut style as the "
                    "other polyhedron generators, following the curved "
                    "stereographic arcs"),
                   ('BALLSTICK', "Ball and Stick",
                    "Struts along the projected edges with a sphere "
                    "at every vertex (ball-and-stick model); the "
                    "struts still follow the curved stereographic "
                    "arcs"),
                   ('WIREFRAME', "Wireframe",
                    "Projected edges as a bare wireframe of polylines "
                    "(no solid struts or spheres)"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "A flat open panel per 2D face of the polytope "
                    "(projected faces are planar in both edge "
                    "styles, since stereographic projection maps "
                    "the circle through a face's vertices to a "
                    "circle)")],
            default='EDGES')
        border: FloatProperty(
            name="Border", default=0.35, min=0.02, max=0.95,
            description="Leonardo panel frame width (fraction of "
                        "the face)")
        panel_thickness: FloatProperty(
            name="Panel Thickness", default=0.03, min=0.002, max=0.5,
            step=1, precision=3)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)
        half: BoolProperty(
            name="Half (Cutaway)", default=False,
            description="Keep only the elements on one side of the "
                        "equatorial hyperplane (w <= 0, equator "
                        "included) before projection -- Segerman's "
                        "legible 'half of a 24/120/600-cell' models")
        dual_compound: BoolProperty(
            name="Dual Compound", default=False,
            description="Also build the dual polytope (5<->5, "
                        "8<->16, 24<->24, 120<->600), its vertices "
                        "at this polytope's cell centers, in a "
                        "second material slot")
        rings: IntProperty(
            name="Hopf Rings", default=0, min=0, max=12,
            description="120-cell only: render N of the 12 rings of "
                        "10 dodecahedral cells that partition the "
                        "120-cell along Hopf fibers, as solid "
                        "shrunken cells with one material per ring")
        ring_cell_scale: FloatProperty(
            name="Ring Cell Scale", default=0.9, min=0.1, max=1.0,
            description="Shrink factor of each ring cell toward its "
                        "4D centroid (gaps make the rings legible)")
        rings_only: BoolProperty(
            name="Rings Only", default=False,
            description="Drop the edge framework and show just the "
                        "rings of solid cells (fig 3-29 style; also "
                        "the printable version)")

        def execute(self, context):
            verts, faces, face_mat, st = build_polytope_ex(
                self.kind, self.style, self.proj_dist, self.rot_xw,
                self.rot_yw, self.rot_zw, self.rot_xy,
                self.arc_segments, self.radius, self.sides, self.taper,
                self.vertex_spheres, self.sphere_factor, self.scale,
                self.render, self.border, self.panel_thickness,
                self.half, self.dual_compound, self.rings,
                self.ring_cell_scale, self.rings_only)
            # center on the origin and fit within a 2 m cube (times the
            # Scale property), so the framework fills the cube by default
            if verts:
                xs = [v[0] for v in verts]
                ys = [v[1] for v in verts]
                zs = [v[2] for v in verts]
                cen = (0.5 * (min(xs) + max(xs)),
                       0.5 * (min(ys) + max(ys)),
                       0.5 * (min(zs) + max(zs)))
                ext = max(max(xs) - min(xs), max(ys) - min(ys),
                          max(zs) - min(zs))
                s = (2.0 * self.scale / ext) if ext > 1e-9 else 1.0
                verts = [((v[0] - cen[0]) * s, (v[1] - cen[1]) * s,
                          (v[2] - cen[2]) * s) for v in verts]
            me = bpy.data.meshes.new("Polytope4D")
            me.from_pydata(verts, st.get('wire_edges', []), faces)
            n_rings = st['n_rings']
            if st['n_systems'] > 1 or n_rings > 0:
                mats = []
                if st['n_systems'] >= 1:
                    mats.append(_make_material(
                        "Polytope4D Primal", (0.75, 0.78, 0.85, 1.0)))
                if st['n_systems'] > 1:
                    mats.append(_make_material(
                        "Polytope4D Dual", (0.9, 0.45, 0.15, 1.0)))
                for ri in range(n_rings):
                    mats.append(_make_material(
                        "Polytope4D Ring %d" % (ri + 1),
                        _ring_color(ri)))
                for mat in mats:
                    me.materials.append(mat)
                me.polygons.foreach_set('material_index', face_mat)
            me.validate(clean_customdata=True)
            # Leonardo panels are wound consistently by construction
            # (the mitred joints share rim walls between panels, so
            # a normal recalc would be unreliable there); struts
            # shade smooth, flat panels and ring cells must stay flat
            smooth = self.render != 'LEONARDO'
            if n_rings > 0 and smooth:
                rb = st['n_systems']
                flags = [m < rb for m in face_mat]
                me.polygons.foreach_set('use_smooth',
                                        flags[:len(me.polygons)])
            else:
                me.polygons.foreach_set(
                    'use_smooth', [smooth] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Polytope4D", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if st['n_systems']:
                msg = f"{st['nv']} vertices, {st['ne']} edges"
                if self.half:
                    msg += " (half)"
            else:
                msg = "rings only"
            if st['n_systems'] > 1:
                msg += (f" + dual {st['dual_nv']} vertices, "
                        f"{st['dual_ne']} edges")
            if n_rings > 0:
                msg += (f" + {n_rings} Hopf rings, "
                        f"{st['n_cells']} cells")
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            lay.prop(self, 'style')
            if self.style == 'STRAIGHT':
                lay.prop(self, 'proj_dist')
            col = lay.column(align=True)
            for k in ('rot_xw', 'rot_yw', 'rot_zw', 'rot_xy'):
                col.prop(self, k)
            lay.prop(self, 'render')
            if self.render == 'LEONARDO':
                for k in ('border', 'panel_thickness', 'taper',
                          'scale'):
                    lay.prop(self, k)
            elif self.render == 'WIREFRAME':
                # bare polylines: only arc smoothness and overall scale
                lay.prop(self, 'arc_segments')
                lay.prop(self, 'scale')
            else:
                # EDGES ("Struts") = tubes only; BALLSTICK adds spheres.
                # The vertex_spheres toggle is retired -- the style now
                # decides -- so it is never drawn.
                keys = ['arc_segments', 'radius', 'sides', 'taper']
                if self.render == 'BALLSTICK':
                    keys.append('sphere_factor')
                keys.append('scale')
                for k in keys:
                    lay.prop(self, k)
            lay.separator()
            col = lay.column(align=True)
            col.label(text="Cutaway, Compound & Rings")
            col.prop(self, 'half')
            col.prop(self, 'dual_compound')
            row = col.row(align=True)
            row.enabled = (self.kind == 'CELL120')
            row.prop(self, 'rings')
            if self.kind == 'CELL120' and self.rings > 0:
                col.prop(self, 'ring_cell_scale')
                col.prop(self, 'rings_only')

    def _menu_func(self, context):
        self.layout.operator("mesh.polytope4d_add", icon='MESH_CUBE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_polytope4d_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_polytope4d_add)


def _selftest():
    bad = []
    for kind, (env, ene) in COUNTS.items():
        V = polytope_vertices(kind)
        E = polytope_edges(V)
        ok = (len(V), len(E)) == (env, ene)
        print(f"{kind:8s}: V={len(V):4d} E={len(E):5d}  "
              f"expect {env},{ene}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(kind)
    assert not bad
