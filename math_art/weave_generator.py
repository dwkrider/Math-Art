
# Polyhedral Weave Generator for Blender
#
# A Blender take on Antiprism's `poly_weave`, including its weave
# PATTERN language. The seed polyhedron's flag (barycentric) subdivision
# is walked by a pattern program; the visited pattern points become
# closed weave strands, swept as flat ribbons or as round rope tubes
# (optionally relaxed so the ropes round their clasps and keep clear
# of each other).
#
# Pattern syntax (after poly_weave -p):
#   [C|L]                 curve (smoothed) or linear path   (default C)
#   bV,bE,bF              barycentric start point in each flag triangle
#                         over (Vertex, Edge centre, Face centre)
#                         (default 0,1,0 = edge centre)
#   :up[,side[,along]]    intermediate point on each step; repeatable.
#                         up = radial offset, side = sideways offset,
#                         along = position along step (default: spaced)
#   steps                 V E F  pivot about vertex/edge/face corner
#                         v e f  cross to the flag on the other side
#                         R      reverse pivot direction
#                         -      stay on the current flag
#   t[l|r|b]              start circuits from left/right/both flags
#
# Examples: "FEV" (default weave), "1,1,1V" (rings around vertices),
# "1,1,1F" (rings around faces), "1,1,1FFE" (squares on a cube),
# "1,1,1::::::FEV" (curved segments), "0,1,0:0.1FEV" (raised weave).
#
# References:
#   - Adrian Rossiter, Antiprism and its `poly_weave` program and
#     pattern language (https://www.antiprism.com) -- the reference
#     implementation this follows.  The over-under plain/twill weave
#     over a surface is an otherwise classical basketry construction.

bl_info = {
    "name": "Polyhedral Weave Generator",
    "author": "Math Art project (after Antiprism's poly_weave)",
    "version": (1, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Polyhedral Weave",
    "description": "Woven-strand spheres from polyhedra (pattern language)",
    "category": "Add Mesh",
}

import math
import re
from math import cos

try:                                    # round-tube sweep (shared helper)
    from . import knot_carpet_generator as kcg
except Exception:                       # legacy single-file / CLI use
    import knot_carpet_generator as kcg

PHI = (1 + 5 ** 0.5) / 2


# ---- seeds ---------------------------------------------------------------

def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def _icosa_faces(V):
    n = len(V)
    emin = None
    d2 = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = sum((V[i][k] - V[j][k]) ** 2 for k in range(3))
            d2[(i, j)] = d
            emin = d if emin is None else min(emin, d)
    adj = {i: set() for i in range(n)}
    for (i, j), d in d2.items():
        if d < emin * 1.2:
            adj[i].add(j)
            adj[j].add(i)
    fs = set()
    for i in range(n):
        for j in adj[i]:
            for k in adj[i] & adj[j]:
                fs.add(tuple(sorted((i, j, k))))
    faces = []
    for f in fs:
        a, b, c = (V[i] for i in f)
        nx = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
              (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
              (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        cen = [(a[k] + b[k] + c[k]) / 3 for k in range(3)]
        if sum(nx[k] * cen[k] for k in range(3)) < 0:
            faces.append([f[0], f[2], f[1]])
        else:
            faces.append(list(f))
    return faces


def seed_poly(kind):
    if kind == 'TETRA':
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        F = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    elif kind == 'OCTA':
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
        F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    elif kind == 'CUBE':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    elif kind == 'ICOSA':
        V = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = _icosa_faces(V)
    elif kind == 'DODECA':
        IV = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                IV += [(0, a, b), (a, b, 0), (b, 0, a)]
        IF = _icosa_faces(IV)
        V = [tuple(sum(IV[i][k] for i in f) / 3 for k in range(3))
             for f in IF]
        F = []
        for vi in range(len(IV)):
            adj = [fi for fi, f in enumerate(IF) if vi in f]
            nrm = _unit(IV[vi])
            u0 = V[adj[0]]
            u = [u0[k] - sum(u0[j] * nrm[j] for j in range(3)) * nrm[k]
                 for k in range(3)]
            w = (nrm[1] * u[2] - nrm[2] * u[1],
                 nrm[2] * u[0] - nrm[0] * u[2],
                 nrm[0] * u[1] - nrm[1] * u[0])
            def ang(fi):
                d = V[fi]
                return math.atan2(sum(d[k] * w[k] for k in range(3)),
                                  sum(d[k] * u[k] for k in range(3)))
            F.append(sorted(adj, key=ang))
    else:
        raise ValueError(kind)
    return [_unit(v) for v in V], [list(f) for f in F]


def geodesic(V, F, freq):
    if freq <= 1:
        return V, F
    verts = list(V)
    key = {}
    faces = []

    def vid(p):
        k = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if k not in key:
            key[k] = len(verts)
            verts.append(p)
        return key[k]

    for f in F:
        A, B, C = (V[i] for i in f)
        grid = {}
        for i in range(freq + 1):
            for j in range(freq + 1 - i):
                k = freq - i - j
                grid[(i, j)] = vid(_unit(tuple(
                    (i * A[c] + j * B[c] + k * C[c]) / freq
                    for c in range(3))))
        for i in range(freq):
            for j in range(freq - i):
                faces.append([grid[(i, j)], grid[(i + 1, j)],
                              grid[(i, j + 1)]])
                if j < freq - i - 1:
                    faces.append([grid[(i + 1, j)], grid[(i + 1, j + 1)],
                                  grid[(i, j + 1)]])
    return verts, faces


# ---- pattern parsing -----------------------------------------------------

class Pattern:
    def __init__(self):
        self.curve = True
        self.bary = (0.0, 1.0, 0.0)
        self.inters = []          # list of (up, side, along-or-None)
        self.steps = []           # tokens from V E F v e f R -
        self.start = 'l'


def parse_pattern(text):
    p = Pattern()
    s = text.strip().replace(' ', '')
    if not s:
        s = 'FEV'
    if s[0] in 'CL':
        p.curve = s[0] == 'C'
        s = s[1:]
    m = re.match(r'^(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)', s)
    if m:
        p.bary = tuple(float(m.group(i)) for i in (1, 2, 3))
        if sum(p.bary) <= 0:
            raise ValueError("barycentric start must have positive sum")
        s = s[m.end():]
    while s.startswith(':'):
        s = s[1:]
        m = re.match(r'^(-?[\d.]*)(?:,(-?[\d.]+))?(?:,(-?[\d.]+))?', s)
        up = float(m.group(1)) if m.group(1) not in ('', '-', None) else 0.0
        side = float(m.group(2)) if m.group(2) else 0.0
        along = float(m.group(3)) if m.group(3) else None
        p.inters.append((up, side, along))
        s = s[m.end():]
    if s.endswith(('tl', 'tr', 'tb')):
        p.start = s[-1]
        s = s[:-2]
    for ch in s:
        if ch in 'VEFvefR-':
            p.steps.append(ch)
        else:
            raise ValueError(f"bad pattern step {ch!r}")
    if not any(c in 'VEFvef' for c in p.steps):
        raise ValueError("pattern needs at least one step from V E F v e f")
    return p


# ---- flag machinery ------------------------------------------------------
# flag = (face index, corner index i, side s); s=0: edge (f[i], f[i+1]),
# s=1: edge (f[i-1], f[i]). The three adjacencies (cross to the flag
# sharing everything except one element):
#   'v' change vertex, 'e' change edge, 'f' change face.

def build_flags(V, F):
    e2fd = {}
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            e2fd[(f[i], f[(i + 1) % m])] = (fi, i)
    return e2fd


def flag_cross(F, e2fd, flag, tok):
    fi, i, s = flag
    f = F[fi]
    m = len(f)
    if tok == 'v':
        return (fi, (i + 1) % m, 1) if s == 0 else (fi, (i - 1) % m, 0)
    if tok == 'e':
        return (fi, i, 1 - s)
    # 'f': cross the edge to the neighbouring face, keep vertex and edge
    if s == 0:
        a, b = f[i], f[(i + 1) % m]
    else:
        a, b = f[(i - 1) % m], f[i]
    gj = e2fd.get((b, a))
    if gj is None:
        raise ValueError("open surface: cannot cross a boundary edge")
    g, j = gj
    # in face g the directed edge is (b, a); flag vertex must stay f[i]
    if s == 0:      # vertex a = start of (a,b): in g it is end of (b,a)
        return (g, (j + 1) % len(F[g]), 1)
    return (g, j, 0)


# A rotation click about an element is two elementary crossings; the
# order sets the rotation sense (R reverses it). V's default sense is
# chosen opposite to F's so that the classic FEV pattern advances as a
# weave strand instead of closing on itself.
_CLICK = {'F': ('e', 'v'), 'E': ('v', 'f'), 'V': ('e', 'f')}


def flag_step(F, e2fd, flag, tok, dirn):
    """One pattern step. Uppercase pivots are a full rotation click
    about the element: two flag crossings, landing on a flag of the
    same handedness."""
    if tok == '-':
        return flag, dirn
    if tok == 'R':
        return flag, -dirn
    if tok in 'vef':
        return flag_cross(F, e2fd, flag, tok), dirn
    a, b = _CLICK[tok]
    if dirn < 0:
        a, b = b, a
    mid = flag_cross(F, e2fd, flag, a)
    return flag_cross(F, e2fd, mid, b), dirn


def flag_point(V, F, flag, bary):
    fi, i, s = flag
    f = F[fi]
    m = len(f)
    vp = V[f[i]]
    if s == 0:
        a, b = f[i], f[(i + 1) % m]
    else:
        a, b = f[(i - 1) % m], f[i]
    ep = tuple((V[a][k] + V[b][k]) / 2 for k in range(3))
    fp = tuple(sum(V[j][k] for j in f) / m for k in range(3))
    bV, bE, bF = bary
    tot = bV + bE + bF
    return tuple((bV * vp[k] + bE * ep[k] + bF * fp[k]) / tot
                 for k in range(3))


def weave_circuits(V, F, pat):
    """Run the pattern over all start flags. Returns a list of circuits,
    each a list of 3D points (pattern points with intermediates)."""
    e2fd = build_flags(V, F)
    flags = []
    for fi, f in enumerate(F):
        for i in range(len(f)):
            flags.append((fi, i, 0))
            flags.append((fi, i, 1))
    if pat.start == 'l':
        starts = [fl for fl in flags if fl[2] == 0]
    elif pat.start == 'r':
        starts = [fl for fl in flags if fl[2] == 1]
    else:
        starts = flags
    visited = set()
    circuits = []
    L = len(pat.steps)
    for f0 in starts:
        state0 = (f0, 0, 1)
        if state0 in visited:
            continue
        pts = []
        flag, dirn = f0, 1
        phase = 0
        guard = 0
        while True:
            st = (flag, phase, dirn)
            if st in visited and pts:
                break
            visited.add(st)
            prev_pt = flag_point(V, F, flag, pat.bary)
            flag2, dirn2 = flag_step(F, e2fd, flag, pat.steps[phase], dirn)
            next_pt = flag_point(V, F, flag2, pat.bary)
            pts.append(prev_pt)
            if pat.inters:
                K = len(pat.inters)
                seg = tuple(next_pt[k] - prev_pt[k] for k in range(3))
                for kk, (up, side, along) in enumerate(pat.inters):
                    t = along if along is not None else (kk + 1) / (K + 1)
                    base = tuple(prev_pt[k] + seg[k] * t for k in range(3))
                    rad = _unit(base)
                    sd = _unit((seg[1] * rad[2] - seg[2] * rad[1],
                                seg[2] * rad[0] - seg[0] * rad[2],
                                seg[0] * rad[1] - seg[1] * rad[0])) \
                        if any(abs(x) > 1e-12 for x in seg) else (0, 0, 0)
                    pts.append(tuple(base[k] + rad[k] * up + sd[k] * side
                                     for k in range(3)))
            flag, dirn = flag2, dirn2
            phase = (phase + 1) % L
            guard += 1
            if guard > 100 * len(flags) * L:
                raise RuntimeError("pattern circuit failed to close")
        # drop consecutive duplicates
        clean = []
        for q in pts:
            if not clean or math.dist(q, clean[-1]) > 1e-9:
                clean.append(q)
        if len(clean) >= 3:
            circuits.append(clean)
    # drop geometric duplicates (the same circuit reached from a
    # different flag state, e.g. traversed in the opposite direction)
    seen_sets = set()
    unique = []
    for c in circuits:
        key = frozenset((round(p[0], 6), round(p[1], 6), round(p[2], 6))
                        for p in c)
        if key not in seen_sets:
            seen_sets.add(key)
            unique.append(c)
    return unique


# ---- strand sweeps -------------------------------------------------------

def _strand_path(pts, amplitude, subdiv, smooth_rounds, curve):
    """Centerline for one strand: the circuit's pattern points with the
    alternating radial over/under undulation, subdivided and (for
    curved patterns) Laplacian-smoothed."""
    L = len(pts)
    path = []
    for i in range(L):
        p0, p1 = pts[i], pts[(i + 1) % L]
        r0 = 1.0 + amplitude * (1 if i % 2 == 0 else -1)
        r1 = 1.0 + amplitude * (1 if (i + 1) % 2 == 0 else -1)
        nseg = subdiv if curve else 1
        for s in range(nseg):
            t = s / nseg
            base = tuple(p0[k] + (p1[k] - p0[k]) * t for k in range(3))
            if amplitude > 0:
                r = r0 + (r1 - r0) * (1 - cos(math.pi * t)) / 2
                ln = math.sqrt(sum(x * x for x in base)) or 1.0
                base = tuple(x / ln * (ln * r) for x in base)
            path.append(base)
    if curve:
        for _ in range(smooth_rounds):
            n = len(path)
            new = []
            for i in range(n):
                a, b, c = path[i - 1], path[i], path[(i + 1) % n]
                new.append(tuple((a[k] + 2 * b[k] + c[k]) / 4
                                 for k in range(3)))
            path = new
    return path


def sweep_ribbons(circuits, width, thickness, amplitude, subdiv,
                  smooth_rounds, scale, curve=True):
    verts = []
    faces = []
    face_strand = []          # strand (circuit) index per face
    for si, pts in enumerate(circuits):
        path = _strand_path(pts, amplitude, subdiv, smooth_rounds, curve)
        n = len(path)
        base_i = len(verts)
        for i in range(n):
            p = path[i]
            nxt = path[(i + 1) % n]
            prv = path[i - 1]
            tang = _unit(tuple(nxt[k] - prv[k] for k in range(3)))
            rad = _unit(p)
            side = _unit((tang[1] * rad[2] - tang[2] * rad[1],
                          tang[2] * rad[0] - tang[0] * rad[2],
                          tang[0] * rad[1] - tang[1] * rad[0]))
            for (sw, sr) in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                verts.append(tuple(
                    (p[k] + side[k] * sw * width / 2
                     + rad[k] * sr * thickness / 2) * scale
                    for k in range(3)))
        for i in range(n):
            i2 = (i + 1) % n
            for j in range(4):
                faces.append([base_i + 4 * i + j,
                              base_i + 4 * i + (j + 1) % 4,
                              base_i + 4 * i2 + (j + 1) % 4,
                              base_i + 4 * i2 + j])
            face_strand.extend([si] * 4)
    return verts, faces, face_strand


def build_weave(kind='CUBE', freq=1, pattern='FEV', width=0.10,
                thickness=0.03, amplitude=0.05, subdiv=6,
                smooth_rounds=2, scale=1.0):
    V, F = seed_poly(kind)
    if kind in ('TETRA', 'OCTA', 'ICOSA'):
        V, F = geodesic(V, F, freq)
    pat = parse_pattern(pattern)
    circuits = weave_circuits(V, F, pat)
    verts, faces, face_strand = sweep_ribbons(
        circuits, width, thickness, amplitude, subdiv, smooth_rounds,
        scale, curve=pat.curve)
    return verts, faces, len(circuits), face_strand


# ---- rope (round tube) sweep ---------------------------------------------

def relax_strands(paths, tube_radius, iters, clearance=0.0):
    """Damped relaxation of all strand centerlines together, so round
    tubes of radius `tube_radius` round their clasps and stop
    interpenetrating.  Every bead is pulled toward its ORIGINAL
    (amplitude-undulated) radius -- so the over/under separation
    survives -- faired lightly along its strand, held near its rest
    edge lengths, and repelled from beads of other strands (and from
    far-away beads of its own strand) closer than `clearance`.
    Jacobi-style (all displacements accumulated, then applied),
    damped, with a per-step clamp: deterministic and stable."""
    if iters <= 0 or not paths:
        return paths
    if clearance <= 0:
        clearance = 2.2 * tube_radius
    P = [[list(p) for p in path] for path in paths]
    r_tgt = [[math.sqrt(sum(x * x for x in p)) or 1.0 for p in path]
             for path in P]
    rest = [[math.dist(path[i], path[(i + 1) % len(path)])
             for i in range(len(path))] for path in P]
    cutoff = 1.5 * clearance
    inv = 1.0 / cutoff
    K_RAD, K_LAP, K_SPR, K_REP = 0.5, 0.2, 0.4, 0.5
    DAMP = 0.5
    clamp = 0.5 * tube_radius
    for _ in range(iters):
        grid = {}                       # spatial hash, cell = cutoff
        for s, path in enumerate(P):
            for i, p in enumerate(path):
                c = (math.floor(p[0] * inv), math.floor(p[1] * inv),
                     math.floor(p[2] * inv))
                grid.setdefault(c, []).append((s, i, p))
        disp = [[[0.0, 0.0, 0.0] for _ in path] for path in P]
        for s, path in enumerate(P):
            n = len(path)
            for i, p in enumerate(path):
                d = disp[s][i]
                a, b = path[i - 1], path[(i + 1) % n]
                # light fairing toward the neighbour midpoint
                for k in range(3):
                    d[k] += K_LAP * ((a[k] + b[k]) / 2 - p[k])
                # rest-length springs along the strand
                for q, rl in ((a, rest[s][i - 1]), (b, rest[s][i])):
                    dq = math.dist(p, q)
                    if dq > 1e-12:
                        w = K_SPR * (dq - rl) / dq
                        for k in range(3):
                            d[k] += w * (q[k] - p[k])
                # pull toward the undulated target radius (the weave)
                r = math.sqrt(sum(x * x for x in p)) or 1.0
                w = K_RAD * (r_tgt[s][i] - r) / r
                for k in range(3):
                    d[k] += w * p[k]
                # repulsion from nearby beads of other strands (and
                # non-local beads of this one); each pair is visited
                # from both sides, hence the half weight
                cx, cy, cz = (math.floor(p[0] * inv),
                              math.floor(p[1] * inv),
                              math.floor(p[2] * inv))
                for gx in (cx - 1, cx, cx + 1):
                    for gy in (cy - 1, cy, cy + 1):
                        for gz in (cz - 1, cz, cz + 1):
                            for t, j, q in grid.get((gx, gy, gz), ()):
                                if t == s:
                                    gap = abs(i - j)
                                    if min(gap, n - gap) < 6:
                                        continue
                                dq = math.dist(p, q)
                                if 1e-12 < dq < clearance:
                                    w = 0.5 * K_REP \
                                        * (clearance - dq) / dq
                                    for k in range(3):
                                        d[k] += w * (p[k] - q[k])
        for s, path in enumerate(P):
            for i, p in enumerate(path):
                step = [DAMP * x for x in disp[s][i]]
                ln = math.sqrt(sum(x * x for x in step))
                if ln > clamp:
                    step = [x * clamp / ln for x in step]
                for k in range(3):
                    p[k] += step[k]
    return [[tuple(p) for p in path] for path in P]


def sweep_tubes(circuits, tube_radius, tube_sides, amplitude, subdiv,
                smooth_rounds, scale, curve=True, relax_iters=0,
                clearance=0.0):
    """Sweep each woven strand as a closed round tube (rope) along the
    same centerline `sweep_ribbons` uses, via the proximity-welded
    rotation-minimizing tube of the knot-carpet module.  With
    relax_iters > 0 the centerlines are first relaxed together (see
    relax_strands)."""
    paths = [_strand_path(pts, amplitude, subdiv, smooth_rounds, curve)
             for pts in circuits]
    if relax_iters > 0:
        paths = relax_strands(paths, tube_radius, relax_iters, clearance)
    sides = max(3, int(tube_sides))
    verts = []
    faces = []
    face_strand = []          # strand (circuit) index per face
    for si, path in enumerate(paths):
        tv, tf = kcg._tube_welded(path, tube_radius, sides)
        if not tf:
            continue
        base = len(verts)
        verts.extend(tuple(float(c) * scale for c in v) for v in tv)
        faces.extend([base + j for j in f] for f in tf)
        face_strand.extend([si] * len(tf))
    return verts, faces, face_strand


def build_weave_tubes(kind='CUBE', freq=1, pattern='FEV',
                      tube_radius=0.03, tube_sides=10, amplitude=0.05,
                      subdiv=6, smooth_rounds=2, scale=1.0,
                      relax_iters=0, clearance=0.0):
    V, F = seed_poly(kind)
    if kind in ('TETRA', 'OCTA', 'ICOSA'):
        V, F = geodesic(V, F, freq)
    pat = parse_pattern(pattern)
    circuits = weave_circuits(V, F, pat)
    verts, faces, face_strand = sweep_tubes(
        circuits, tube_radius, tube_sides, amplitude, subdiv,
        smooth_rounds, scale, curve=pat.curve,
        relax_iters=relax_iters, clearance=clearance)
    return verts, faces, len(circuits), face_strand


# ---- Blender layer -------------------------------------------------------

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    PATTERNS = [
        ('CUSTOM', "Custom", ""),
        ('vfe', "Classic Weave (vfe)",
         "straight-ahead weave: cube 4 hexagons, icosa 6 decagons"),
        ('FEV', "Corner Weave (FEV)", "circuits turning around vertices"),
        ('1,1,1V', "Vertex Rings (1,1,1V)", "circuits around vertices"),
        ('1,1,1F', "Face Rings (1,1,1F)", "circuits around faces"),
        ('1,1,1FFE', "Face Pairs (1,1,1FFE)", "squares on a cube"),
        ('1,1,1::::::FEV', "Curved (1,1,1::::::FEV)",
         "six intermediate points"),
        ('0,1,0:0.12FEV', "Raised Weave (0,1,0:0.12FEV)",
         "intermediate point lifted radially"),
        ('0,1,0:0.1,0.06:-0.1,0.06FEV', "Wavy (up/side intermediates)",
         ""),
        ('1,3,1EV', "Knots (1,3,1EV)", "small rings at each edge"),
    ]

    class MESH_OT_weave_add(bpy.types.Operator):
        """Add a woven-strand model driven by a poly_weave-style
        pattern walked over the seed polyhedron's flag triangles"""
        bl_idname = "mesh.poly_weave_add"
        bl_label = "Polyhedral Weave"
        bl_options = {'REGISTER', 'UNDO'}

        def _pattern_chosen(self, context):
            if self.pattern_preset != 'CUSTOM':
                self.pattern = self.pattern_preset

        kind: EnumProperty(
            name="Seed",
            items=[('CUBE', "Cube", ""), ('ICOSA', "Icosahedron", ""),
                   ('OCTA', "Octahedron", ""), ('TETRA', "Tetrahedron", ""),
                   ('DODECA', "Dodecahedron", "")],
            default='CUBE')
        freq: IntProperty(
            name="Geodesic Frequency", default=1, min=1, max=6,
            description="Subdivision of triangular seeds")
        pattern_preset: EnumProperty(name="Pattern Preset", items=PATTERNS,
                                     default='vfe',
                                     update=_pattern_chosen)
        pattern: StringProperty(
            name="Pattern", default="vfe",
            description="[C|L][bV,bE,bF][:up,side,along...]steps[tl|tr|tb]"
                        " -- steps from V E F v e f R -")
        output: EnumProperty(
            name="Output",
            items=[('RIBBON', "Ribbon", "Flat rectangular strands"),
                   ('TUBE', "Tube (Rope)",
                    "Round tubes -- the weave as rope with rounded "
                    "clasps")],
            default='RIBBON')
        width: FloatProperty(name="Strand Width", default=0.10,
                             min=0.005, max=0.5)
        thickness: FloatProperty(name="Strand Thickness", default=0.03,
                                 min=0.002, max=0.2)
        tube_radius: FloatProperty(name="Tube Radius", default=0.03,
                                   min=0.005, max=0.2)
        tube_sides: IntProperty(name="Tube Sides", default=10,
                                min=3, max=32)
        relax_iters: IntProperty(
            name="Relax Iterations", default=0, min=0, max=500,
            description="Relax the strand centerlines together so the "
                        "ropes round their clasps and keep clear of "
                        "each other (0 = off)")
        rope_clearance: FloatProperty(
            name="Rope Clearance", default=0.0, min=0.0, max=0.5,
            description="Target centre-to-centre strand clearance when "
                        "relaxing (0 = auto, 2.2 x tube radius)")
        amplitude: FloatProperty(
            name="Weave Amplitude", default=0.05, min=0.0, max=0.3,
            description="Alternating radial offset at pattern points "
                        "(set 0 when the pattern's 'up' values weave)")
        subdiv: IntProperty(name="Path Subdivision", default=6,
                            min=1, max=24)
        smooth_rounds: IntProperty(name="Smoothing", default=2,
                                   min=0, max=10)
        coloring: EnumProperty(
            name="Coloring",
            items=[('STRAND', "Per Strand",
                    "One material per woven strand (view with "
                    "Material Preview or Solid shading set to "
                    "Material color)"),
                   ('NONE', "None", "No materials")],
            default='STRAND')
        scale: FloatProperty(name="Radius", default=1.0, min=0.01,
                             max=100.0)

        _PALETTE = [(0.90, 0.36, 0.23), (0.27, 0.52, 0.79),
                    (0.95, 0.77, 0.29), (0.30, 0.69, 0.42),
                    (0.62, 0.40, 0.75), (0.25, 0.72, 0.72),
                    (0.91, 0.56, 0.71), (0.55, 0.60, 0.29),
                    (0.45, 0.45, 0.85), (0.80, 0.50, 0.30),
                    (0.35, 0.60, 0.55), (0.75, 0.35, 0.45)]

        @classmethod
        def _material_for(cls, i):
            name = f"Weave Strand {i + 1}"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                rgb = cls._PALETTE[i % len(cls._PALETTE)]
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb,
                                                               1.0)
                    bsdf.inputs["Roughness"].default_value = 0.45
            return mat

        def execute(self, context):
            try:
                if self.output == 'TUBE':
                    verts, faces, ns, face_strand = build_weave_tubes(
                        self.kind, self.freq, self.pattern,
                        self.tube_radius, self.tube_sides,
                        self.amplitude, self.subdiv,
                        self.smooth_rounds, self.scale,
                        self.relax_iters, self.rope_clearance)
                else:
                    verts, faces, ns, face_strand = build_weave(
                        self.kind, self.freq, self.pattern, self.width,
                        self.thickness, self.amplitude, self.subdiv,
                        self.smooth_rounds, self.scale)
            except (ValueError, RuntimeError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            me = bpy.data.meshes.new("Weave")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            # fit (roughly) within a 2 x scale cube at the origin
            lo = [min(v.co[k] for v in me.vertices)
                  for k in range(3)]
            hi = [max(v.co[k] for v in me.vertices)
                  for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0
                       for k in range(3)) or 1.0
            f = self.scale / half
            for v in me.vertices:
                v.co = [(v.co[k] - (lo[k] + hi[k]) / 2.0) * f
                        for k in range(3)]
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            if len(me.polygons) == len(face_strand):
                attr = me.attributes.new("strand_index", 'INT',
                                         'FACE')
                attr.data.foreach_set('value', face_strand)
                if self.coloring == 'STRAND':
                    for i in range(ns):
                        me.materials.append(self._material_for(i))
                    me.polygons.foreach_set('material_index',
                                            face_strand)
            me.update()
            obj = bpy.data.objects.new("Weave", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, f"{ns} strands")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            if self.kind in ('TETRA', 'OCTA', 'ICOSA'):
                lay.prop(self, 'freq')
            lay.prop(self, 'pattern_preset')
            lay.prop(self, 'pattern')
            lay.prop(self, 'output')
            if self.output == 'TUBE':
                keys = ('tube_radius', 'tube_sides', 'relax_iters',
                        'rope_clearance')
            else:
                keys = ('width', 'thickness')
            for k in keys + ('amplitude', 'subdiv', 'smooth_rounds',
                             'coloring', 'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.poly_weave_add", icon='MOD_LATTICE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_weave_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_weave_add)


def _selftest():
    for kind, freq, pattern in (
            ('CUBE', 1, 'FEV'), ('OCTA', 1, 'FEV'),
            ('ICOSA', 1, 'FEV'), ('ICOSA', 2, 'FEV'),
            ('CUBE', 1, '1,1,1V'), ('CUBE', 1, '1,1,1F'),
            ('CUBE', 1, '1,1,1FFE'), ('DODECA', 1, 'FEV'),
            ('ICOSA', 2, '1,1,1::::::FEV'),
            ('CUBE', 1, '0,1,0:0.12FEV'),
            ('CUBE', 1, '1,3,1EV')):
        V, F = seed_poly(kind)
        if kind in ('TETRA', 'OCTA', 'ICOSA'):
            V, F = geodesic(V, F, freq)
        pat = parse_pattern(pattern)
        cir = weave_circuits(V, F, pat)
        lens = sorted(set(len(c) for c in cir))
        print(f"{kind} f{freq} '{pattern}': strands={len(cir)} "
              f"point-counts={lens}")

    # tube (rope) output
    for kind, freq, relax in (('CUBE', 1, 0), ('CUBE', 1, 40),
                              ('ICOSA', 1, 0), ('ICOSA', 1, 40)):
        verts, faces, ns, fs = build_weave_tubes(
            kind, freq, 'FEV', tube_radius=0.03, tube_sides=8,
            amplitude=0.05, subdiv=4, relax_iters=relax)
        assert verts and faces, "empty tube geometry"
        assert all(len(f) == 4 for f in faces), "non-quad tube face"
        assert len(fs) == len(faces), "face_strand length mismatch"
        assert all(math.isfinite(c) for v in verts for c in v), \
            "non-finite tube vertex"
        print(f"{kind} f{freq} TUBE relax={relax}: strands={ns} "
              f"verts={len(verts)} faces={len(faces)}")

    # relaxation separates the strand centerlines
    def _min_inter(paths):
        best = None
        for a in range(len(paths)):
            for b in range(a + 1, len(paths)):
                for p in paths[a]:
                    for q in paths[b]:
                        d = math.dist(p, q)
                        if best is None or d < best:
                            best = d
        return best

    tr = 0.03
    clr = 2.2 * tr
    V, F = seed_poly('ICOSA')
    cir = weave_circuits(V, F, parse_pattern('FEV'))
    paths0 = [_strand_path(c, 0.05, 4, 2, True) for c in cir]
    paths1 = relax_strands(paths0, tr, 80)
    assert all(math.isfinite(c) for p in paths1 for q in p
               for c in q), "non-finite relaxed bead"
    d0, d1 = _min_inter(paths0), _min_inter(paths1)
    print(f"ICOSA f1 relax: min inter-strand dist "
          f"{d0:.4f} -> {d1:.4f} (target clearance {clr:.4f})")
    assert d1 >= 0.5 * clr or d1 >= d0, \
        "relaxation did not keep strands apart"
    print("RESULT: OK")
