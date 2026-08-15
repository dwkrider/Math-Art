# Weaving strands over a polyhedron's edges or faces.
#
# Part of the Math Art weaving engine (`math_art/weaving/`).  Python + numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly;
# the registered operators stay in their flat generator modules.
#
# A polyhedral weave routes bands along the edge graph so that each
# crossing alternates over and under.  Whether a consistent alternation
# EXISTS is a parity question on the crossing graph, which is why the
# solver lives in `patterns.overunder` and is shared with the flat
# interlace engines.

import math
import re
from math import cos

try:                                  # inside the math_art package
    from ..curve_frames import welded_tube
except ImportError:                   # flat import (test runner)
    from curve_frames import welded_tube
try:                                  # inside the math_art package
    from ..polyhedra.seeds import icosa_faces as _icosa_faces
    from ..polyhedra.seeds import seed_poly as _shared_seed
except ImportError:                   # flat import (test runner)
    from polyhedra.seeds import icosa_faces as _icosa_faces
    from polyhedra.seeds import seed_poly as _shared_seed


PHI = (1 + 5 ** 0.5) / 2


def seed_poly(kind):
    """Platonic seed normalised to unit circumradius.

    This module and two others built their seeds this way, while
    three others used the raw coordinates.  The two sets are
    EXACTLY related by that scale (verified vertex for vertex on
    all five solids, with identical face lists), so the shared
    table carries both behind its `unit` flag and this wrapper
    keeps the call sites here reading as before.
    """
    return _shared_seed(kind, unit=True)


def _unit(v):
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


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
        tv, tf = welded_tube(path, tube_radius, sides)
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
