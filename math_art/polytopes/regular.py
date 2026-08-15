# Regular 4-polytopes, and their projection into three dimensions.
#
# Part of the Math Art polytopes engine (`math_art/polytopes/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# There are exactly six regular convex 4-polytopes -- the 5-cell,
# 8-cell (tesseract), 16-cell, 24-cell, 120-cell and 600-cell -- two
# more than the number of Platonic solids, and the only dimension above
# four has just three.  The 24-cell is the odd one out: self-dual, with
# no three-dimensional analogue at all.
#
# Seeing one means projecting R^4 to R^3, either orthographically or by
# stereographic projection from the 3-sphere, and rotating in a PLANE
# rather than about an axis -- in four dimensions rotations fix a plane,
# so there are two independent angles.
#
# References:
# - L. Schlafli, "Theorie der vielfachen Kontinuitat" (written 1852,
#   published 1901) -- the classification of regular polytopes.
# - H. S. M. Coxeter, "Regular Polytopes", 3rd ed., Dover, 1973 --
#   chapters 7-8 and the Schlafli symbols used here.
# - J. H. Conway and D. A. Smith, "On Quaternions and Octonions", 2003
#   -- the quaternion description of rotations of the 3-sphere.

import math
import itertools
from math import cos, sin, pi, sqrt


PHI = (1 + 5 ** 0.5) / 2


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




