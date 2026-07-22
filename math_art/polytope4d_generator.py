
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

bl_info = {
    "name": "4D Polytopes",
    "author": "David Krider (Math Art project)",
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


def polytope_faces(kind, V, E):
    """The 2D faces (polygon vertex cycles) of the polytope: walks
    along edges staying inside a common 2-flat."""
    if kind in _FACE_CACHE:
        return _FACE_CACHE[kind]
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
    _FACE_CACHE[kind] = faces
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


def clear_pole(V):
    """If any vertex sits near the stereographic pole (w ~ 1), apply a
    deterministic extra 4D rotation that moves every vertex away from
    it (otherwise that vertex would project to infinity)."""
    def max_w(vs):
        return max(v[3] for v in vs)
    if max_w(V) < 0.93:
        return V, False
    best = None
    best_w = 2.0
    for k in range(1, 80):
        cand = rotate4(V, k * 1.7, k * 1.1, 0.0, 0.0)
        w = max_w(cand)
        if w < best_w:
            best_w = w
            best = cand
        if w < 0.9:
            break
    return best, True


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


def build_polytope(kind='CELL8', style='CURVED', proj_dist=1.05,
                   rot_xw=0.0, rot_yw=0.0, rot_zw=0.0, rot_xy=0.0,
                   arc_segments=12, radius=0.03, sides=6, taper=True,
                   vertex_spheres=True, sphere_factor=1.6, scale=1.0,
                   render='EDGES', border=0.35, panel_thickness=0.03):
    V4 = polytope_vertices(kind)
    E = polytope_edges(V4)
    F2 = (polytope_faces(kind, V4, E) if render == 'LEONARDO'
          else None)
    V4 = rotate4(V4, rot_xw, rot_yw, rot_zw, rot_xy)
    if style == 'CURVED':
        # exact stereographic projection: pole ON the unit 3-sphere
        dist = 1.0
        V4, _nudged = clear_pole(V4)
    else:
        dist = max(proj_dist, 1.001)
    verts = []
    faces = []
    proj = {}
    for i, v in enumerate(V4):
        p, s = project_point(v, dist)
        proj[i] = (tuple(c * scale for c in p), s)
    if render == 'LEONARDO':
        # flat panels per 2D face: stereographic projection maps the
        # circle through a face's vertices to a circle in R^3, so
        # every projected face is planar in both styles
        nvp = len(V4)
        O = tuple(sum(proj[i][0][k] for i in range(nvp)) / nvp
                  for k in range(3))
        verts, faces = _leonardo_panels(F2, proj, O, border,
                                        panel_thickness, taper,
                                        scale)
        return verts, faces, len(V4), len(E)
    for (i, j) in E:
        if style == 'CURVED':
            pts = []
            scls = []
            for k in range(arc_segments + 1):
                t = k / arc_segments
                q = _slerp4(V4[i], V4[j], t)
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
                pts = [tuple(a[k] + (b[k] - a[k]) * t / arc_segments
                             for k in range(3))
                       for t in range(arc_segments + 1)]
                scls = [sa + (sb - sa) * t / arc_segments
                        for t in range(arc_segments + 1)]
        if taper:
            radii = [radius * s * scale for s in scls]
        else:
            radii = [radius * scale] * len(pts)
        add_strut(verts, faces, pts, radii, sides)
    if vertex_spheres:
        for i in range(len(V4)):
            p, s = proj[i]
            r = radius * sphere_factor * (s if taper else 1.0) * scale
            add_sphere(verts, faces, p, r)
    return verts, faces, len(V4), len(E)


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
            items=[('EDGES', "Edge Struts",
                    "Struts along the projected edges"),
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

        def execute(self, context):
            verts, faces, nv, ne = build_polytope(
                self.kind, self.style, self.proj_dist, self.rot_xw,
                self.rot_yw, self.rot_zw, self.rot_xy,
                self.arc_segments, self.radius, self.sides, self.taper,
                self.vertex_spheres, self.sphere_factor, self.scale,
                self.render, self.border, self.panel_thickness)
            me = bpy.data.meshes.new("Polytope4D")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            # Leonardo panels are wound consistently by construction
            # (the mitred joints share rim walls between panels, so
            # a normal recalc would be unreliable there); struts
            # shade smooth, flat panels must stay flat
            me.polygons.foreach_set(
                'use_smooth',
                [self.render != 'LEONARDO'] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Polytope4D", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{nv} vertices, {ne} edges")
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
            else:
                for k in ('arc_segments', 'radius', 'sides', 'taper',
                          'vertex_spheres', 'sphere_factor', 'scale'):
                    lay.prop(self, k)

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


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        for kind, (env, ene) in COUNTS.items():
            V = polytope_vertices(kind)
            E = polytope_edges(V)
            ok = (len(V), len(E)) == (env, ene)
            print(f"{kind:8s}: V={len(V):4d} E={len(E):5d}  "
                  f"expect {env},{ene}  {'OK' if ok else 'MISMATCH'}")
