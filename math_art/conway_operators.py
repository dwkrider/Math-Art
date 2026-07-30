
# Conway Polyhedron Notation for Blender
#
# Apply Conway/Hart operator strings ("dkC", "taD", "k3sT", ...) to seed
# polyhedra, in the spirit of Antiprism's `conway` program and George
# Hart's notation.
#
# Seeds:      T C O D I  (Platonics),  Pn prism, An antiprism, Yn pyramid
# Primitive:  d dual, a ambo, k kis (kN: only N-gon faces), g gyro,
#             c chamfer, r reflect, p propellor
# Derived:    t truncate (=dkd, tN=dkNd), j join (=da), e expand (=aa),
#             o ortho (=jj), b bevel (=ta), m meta (=kj), s snub (=dg),
#             n needle (=kd), z zip (=dk)
#
# Optional geometry post-processing: spherize, or George Hart-style
# canonicalization (edges tangent to the unit sphere, planar faces).
#
# References:
# - Conway polyhedron notation: John H. Conway; extended and
#   popularized by George W. Hart ("Conway Notation for Polyhedra").
# - Canonicalization algorithm: George W. Hart, "Calculating
#   Canonical Polyhedra", Mathematica in Education and Research
#   6(3), 1997, pp. 5-10.
# - Antiprism (Adrian Rossiter), the `conway` program.

bl_info = {
    "name": "Conway Polyhedron Operators",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Conway Polyhedron",
    "description": "Conway notation polyhedra (dual, kis, ambo, gyro, ...)",
    "category": "Add Mesh",
}

import math
import re

try:
    import numpy as np
except ImportError:
    np = None

PHI = (1 + 5 ** 0.5) / 2


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------

def _seed(sym, n):
    if sym == 'T':
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        F = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    elif sym == 'C':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    elif sym == 'O':
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
             (0, 0, 1), (0, 0, -1)]
        F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    elif sym == 'I':
        V = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = _hull_faces(V)
    elif sym == 'D':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        for a in (-1 / PHI, 1 / PHI):
            for b in (-PHI, PHI):
                V += [(0, a, b), (a, b, 0), (b, 0, a)]
        F = _hull_faces(V)
    elif sym == 'P':          # n-prism
        V = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n), z)
             for z in (-1, 1) for i in range(n)]
        F = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
        F += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    elif sym == 'A':          # n-antiprism
        V = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n),
              -0.5) for i in range(n)]
        V += [(math.cos(2 * math.pi * (i + 0.5) / n),
               math.sin(2 * math.pi * (i + 0.5) / n), 0.5) for i in range(n)]
        F = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
        for i in range(n):
            F.append((i, (i + 1) % n, n + i))
            F.append(((i + 1) % n, n + (i + 1) % n, n + i))
    elif sym == 'Y':          # n-pyramid
        V = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n),
              0.0) for i in range(n)] + [(0.0, 0.0, 1.0)]
        F = [tuple(range(n - 1, -1, -1))]
        F += [(i, (i + 1) % n, n) for i in range(n)]
    else:
        raise ValueError(f"unknown seed {sym!r}")
    return [list(map(float, v)) for v in V], [list(f) for f in F]


def _hull_faces(V):
    """Faces of the convex hull of V, merged into maximal planar faces.
    Small inputs only (seed construction)."""
    import itertools
    n = len(V)
    planes = []
    for combo in itertools.combinations(range(n), 3):
        a, b, c = (V[i] for i in combo)
        nx = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
              (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
              (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        ln = math.sqrt(sum(x * x for x in nx))
        if ln < 1e-9:
            continue
        nx = tuple(x / ln for x in nx)
        d = sum(nx[i] * a[i] for i in range(3))
        if d < 0:
            nx = tuple(-x for x in nx)
            d = -d
        key = (round(nx[0], 6), round(nx[1], 6), round(nx[2], 6), round(d, 6))
        if all(sum(nx[i] * V[j][i] for i in range(3)) <= d + 1e-8
               for j in range(n)):
            planes.append((key, nx, d))
    seen = set()
    faces = []
    for key, nx, d in planes:
        if key in seen:
            continue
        seen.add(key)
        onv = [j for j in range(n)
               if abs(sum(nx[i] * V[j][i] for i in range(3)) - d) < 1e-7]
        cen = [sum(V[j][i] for j in onv) / len(onv) for i in range(3)]
        ux = [V[onv[0]][i] - cen[i] for i in range(3)]
        uy = [nx[1] * ux[2] - nx[2] * ux[1], nx[2] * ux[0] - nx[0] * ux[2],
              nx[0] * ux[1] - nx[1] * ux[0]]
        def ang(j):
            dx = [V[j][i] - cen[i] for i in range(3)]
            return math.atan2(sum(dx[i] * uy[i] for i in range(3)),
                              sum(dx[i] * ux[i] for i in range(3)))
        faces.append(sorted(onv, key=ang))
    return faces


# --------------------------------------------------------------------------
# Mesh helpers
# --------------------------------------------------------------------------

def _centroid(V, f):
    return [sum(V[i][c] for i in f) / len(f) for c in range(3)]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _newell(V, f):
    n = [0.0, 0.0, 0.0]
    for i in range(len(f)):
        p = V[f[i]]
        q = V[f[(i + 1) % len(f)]]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return n


def orient_outward(V, F):
    """Flip faces whose Newell normal points toward the centroid.
    Valid for star-shaped (in particular convex-ish) solids."""
    cen = [sum(v[c] for v in V) / len(V) for c in range(3)]
    for f in F:
        n = _newell(V, f)
        fc = _centroid(V, f)
        d = _sub(fc, cen)
        if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] < 0:
            f.reverse()
    return V, F


def _edge_face_maps(F):
    """directed edge (a,b) -> face index; and next-vertex-in-face map."""
    e2f = {}
    nxt = {}
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            e2f[(a, b)] = fi
            nxt[(fi, a)] = b
    return e2f, nxt


def _faces_around_vertex(v, start_face, e2f, nxt):
    """Ordered cycle of faces around vertex v."""
    out = []
    f = start_face
    while True:
        out.append(f)
        w = nxt[(f, v)]
        f = e2f[(w, v)]
        if f == start_face:
            break
        if len(out) > 1000:
            raise ValueError("bad topology walking around vertex")
    return out


# --------------------------------------------------------------------------
# Primitive operators
# --------------------------------------------------------------------------

def op_dual(V, F):
    e2f, nxt = _edge_face_maps(F)
    v2f = {}
    for fi, f in enumerate(F):
        for v in f:
            v2f.setdefault(v, fi)
    NV = [_centroid(V, f) for f in F]
    NF = []
    for v in range(len(V)):
        NF.append(_faces_around_vertex(v, v2f[v], e2f, nxt))
    return NV, NF


def op_ambo(V, F):
    e2f, nxt = _edge_face_maps(F)
    eid = {}
    NV = []
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            k = (min(a, b), max(a, b))
            if k not in eid:
                eid[k] = len(NV)
                NV.append([(V[a][c] + V[b][c]) / 2 for c in range(3)])
    NF = []
    for f in F:
        m = len(f)
        NF.append([eid[(min(f[i], f[(i + 1) % m]),
                        max(f[i], f[(i + 1) % m]))] for i in range(m)])
    v2f = {}
    for fi, f in enumerate(F):
        for v in f:
            v2f.setdefault(v, fi)
    for v in range(len(V)):
        cyc = _faces_around_vertex(v, v2f[v], e2f, nxt)
        ring = []
        for fi in cyc:
            w = nxt[(fi, v)]
            ring.append(eid[(min(v, w), max(v, w))])
        NF.append(ring[::-1])
    return NV, NF


def op_kis(V, F, only_n=0, height=0.25):
    NV = [list(v) for v in V]
    NF = []
    for f in F:
        if only_n and len(f) != only_n:
            NF.append(list(f))
            continue
        c = _centroid(V, f)
        n = _newell(V, f)
        ln = math.sqrt(sum(x * x for x in n)) or 1.0
        el = sum(math.dist(V[f[i]], V[f[(i + 1) % len(f)]])
                 for i in range(len(f))) / len(f)
        apex = [c[j] + n[j] / ln * height * el for j in range(3)]
        ai = len(NV)
        NV.append(apex)
        for i in range(len(f)):
            NF.append([f[i], f[(i + 1) % len(f)], ai])
    return NV, NF


def op_gyro(V, F):
    NV = [list(v) for v in V]
    cid = {}
    for fi, f in enumerate(F):
        cid[fi] = len(NV)
        NV.append(_centroid(V, f))
    pid = {}
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            if (a, b) not in pid:
                pid[(a, b)] = len(NV)
                NV.append([V[a][c] + (V[b][c] - V[a][c]) / 3
                           for c in range(3)])
    NF = []
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            v1, v2, v3 = f[i], f[(i + 1) % m], f[(i + 2) % m]
            NF.append([cid[fi], pid[(v1, v2)], pid[(v2, v1)], v2,
                       pid[(v2, v3)]])
    return NV, NF


def op_propellor(V, F):
    """Hart's propellor: each n-gon becomes a smaller rotated n-gon
    surrounded by n quadrilaterals (chiral; dp = pd, pa = ap)."""
    NV = [list(v) for v in V]
    pid = {}
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            if (a, b) not in pid:
                pid[(a, b)] = len(NV)
                NV.append([V[a][c] + (V[b][c] - V[a][c]) / 3
                           for c in range(3)])
    NF = []
    for fi, f in enumerate(F):
        m = len(f)
        NF.append([pid[(f[i], f[(i + 1) % m])] for i in range(m)])
        for i in range(m):
            v1, v2, v3 = f[i], f[(i + 1) % m], f[(i + 2) % m]
            NF.append([pid[(v1, v2)], pid[(v2, v1)], v2, pid[(v2, v3)]])
    return NV, NF


def op_chamfer(V, F, t=0.35):
    e2f, nxt = _edge_face_maps(F)
    NV = [list(v) for v in V]
    sid = {}
    for fi, f in enumerate(F):
        c = _centroid(V, f)
        for v in f:
            sid[(fi, v)] = len(NV)
            NV.append([V[v][j] + (c[j] - V[v][j]) * t for j in range(3)])
    NF = []
    for fi, f in enumerate(F):
        NF.append([sid[(fi, v)] for v in f])
    done = set()
    for fi, f in enumerate(F):
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            k = (min(a, b), max(a, b))
            if k in done:
                continue
            done.add(k)
            fL = e2f[(a, b)]
            fR = e2f[(b, a)]
            NF.append([a, sid[(fR, a)], sid[(fR, b)], b,
                       sid[(fL, b)], sid[(fL, a)]])
    return NV, NF


def op_reflect(V, F):
    NV = [[-v[0], v[1], v[2]] for v in V]
    NF = [list(reversed(f)) for f in F]
    return NV, NF


# --------------------------------------------------------------------------
# Notation
# --------------------------------------------------------------------------

DERIVED = {'t': 'dk{n}d', 'j': 'da', 'e': 'aa', 'o': 'dada', 'b': 'dk{n}da',
           'm': 'k{n}da', 's': 'dg', 'n': 'k{n}d', 'z': 'dk{n}'}
PRIMS = set('dakgcrp')
SEEDS = set('TCODIPAY')


def parse_conway(text):
    """Parse a Conway string into (ops_right_to_left, seed, seed_n).
    ops is a list of (op_char, n) with n=0 meaning unfiltered."""
    text = text.strip().replace(' ', '')
    m = re.fullmatch(r'([a-z0-9]*)([TCODIPAY])(\d*)', text)
    if not m:
        raise ValueError(
            "cannot parse: expected operators then a seed, e.g. 'dkC', "
            "'taD', 'k3sT', 'eP5'")
    opstr, seed, seed_n = m.group(1), m.group(2), m.group(3)
    n = int(seed_n) if seed_n else 0
    if seed in 'PAY' and n < 3:
        raise ValueError(f"seed {seed} needs a number >= 3, e.g. {seed}5")
    toks = re.findall(r'([a-z])(\d*)', opstr)
    # expand derived operators (right-to-left application order)
    prim = []
    for ch, dig in toks:
        nn = int(dig) if dig else 0
        if ch in PRIMS:
            if nn and ch != 'k':
                raise ValueError(f"only k (and t) take a number, not {ch}")
            prim.append((ch, nn))
        elif ch in DERIVED:
            sub = DERIVED[ch].format(n='')
            subtoks = re.findall(r'([a-z])', sub)
            for sch in subtoks:
                prim.append((sch, nn if sch == 'k' else 0))
        else:
            raise ValueError(f"unknown operator {ch!r}")
    return prim, seed, n


def apply_conway(text, kis_height=0.25, chamfer_t=0.35):
    prim, seed, n = parse_conway(text)
    V, F = _seed(seed, n)
    V, F = orient_outward(V, F)
    for ch, nn in reversed(prim):
        if ch == 'd':
            V, F = op_dual(V, F)
        elif ch == 'a':
            V, F = op_ambo(V, F)
        elif ch == 'k':
            V, F = op_kis(V, F, only_n=nn, height=kis_height)
        elif ch == 'g':
            V, F = op_gyro(V, F)
        elif ch == 'c':
            V, F = op_chamfer(V, F, t=chamfer_t)
        elif ch == 'p':
            V, F = op_propellor(V, F)
        elif ch == 'r':
            V, F = op_reflect(V, F)
        V, F = orient_outward(V, F)
    return V, F


# --------------------------------------------------------------------------
# Geometry post-processing
# --------------------------------------------------------------------------

def spherize(V, F, iters=0):
    c = [sum(v[i] for v in V) / len(V) for i in range(3)]
    out = []
    for v in V:
        d = _sub(v, c)
        ln = math.sqrt(sum(x * x for x in d)) or 1.0
        out.append([d[i] / ln for i in range(3)])
    return out


def canonicalize(V, F, iters=200, lam_t=0.3, lam_p=0.5):
    """George Hart's canonicalization: iterate edge-tangency to the unit
    sphere, recentering on the edge tangency points, and face
    planarization, until converged (or iters)."""
    if np is None:
        return V
    P = np.array(V, dtype=np.float64)
    P -= P.mean(axis=0)
    P /= np.mean(np.linalg.norm(P, axis=1))
    edges = set()
    for f in F:
        m = len(f)
        for i in range(m):
            a, b = f[i], f[(i + 1) % m]
            edges.add((min(a, b), max(a, b)))
    E = np.array(sorted(edges))
    Fi = [np.array(f) for f in F]
    for it in range(iters):
        prev = P.copy()
        A = P[E[:, 0]]
        B = P[E[:, 1]]
        d = B - A
        t = -np.einsum('ij,ij->i', A, d) / np.maximum(
            np.einsum('ij,ij->i', d, d), 1e-12)
        t = np.clip(t, 0.0, 1.0)
        C = A + t[:, None] * d
        # recentre on the edge tangency points (Hart), then push edges
        # to tangency with the unit sphere
        P -= C.mean(axis=0)
        C -= C.mean(axis=0)
        cl = np.linalg.norm(C, axis=1, keepdims=True)
        corr = C / np.maximum(cl, 1e-9) * (1.0 - cl)
        adj = np.zeros_like(P)
        cnt = np.zeros(len(P))
        np.add.at(adj, E[:, 0], corr)
        np.add.at(adj, E[:, 1], corr)
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
        P += lam_t * adj / np.maximum(cnt, 1)[:, None]
        for f in Fi:
            Q = P[f]
            c = Q.mean(axis=0)
            Qc = Q - c
            nrm = np.zeros(3)
            for i in range(len(f)):
                p = Qc[i]
                q = Qc[(i + 1) % len(f)]
                nrm += np.cross(p, q)
            ln = np.linalg.norm(nrm)
            if ln < 1e-12:
                continue
            nrm /= ln
            dist = Qc @ nrm
            P[f] -= lam_p * dist[:, None] * nrm
        if np.max(np.linalg.norm(P - prev, axis=1)) < 1e-7:
            break
    return [list(map(float, p)) for p in P]


# --------------------------------------------------------------------------
# Named catalog
# --------------------------------------------------------------------------
#
# Curated named solids that are reachable from the operators above plus
# Hart canonicalization, mirroring several of the "computed" categories in
# David I. McCooey's Visual Polyhedra (dmccooey.com/polyhedra): the
# Archimedean-Catalan hulls (Conway join `j`), the propellor solids (`p`),
# the truncated (`t`) and rectified (`a`) Archimedean solids, the chamfered
# solids (`c`), and the dipyramids/trapezohedra (duals of the uniform
# prisms/antiprisms).  Every entry is a pure Conway construction on an exact
# seed followed by canonicalization -- no coordinate data is copied; the
# site is used only as a naming/verification reference.
#
# Each entry is (notation, category, name).

CATALOG = [
    # Archimedean-Catalan hulls: convex hull of an Archimedean solid and its
    # Catalan dual == the Conway join of the Archimedean solid.
    ('jtT', 'Hull', "Joined Truncated Tetrahedron"),
    ('jaC', 'Hull', "Joined Cuboctahedron"),
    ('jtO', 'Hull', "Joined Truncated Octahedron"),
    ('jtC', 'Hull', "Joined Truncated Cube"),
    ('jeC', 'Hull', "Joined Rhombicuboctahedron"),
    ('jsC', 'Hull', "Joined Snub Cube (laevo)"),
    ('rjsC', 'Hull', "Joined Snub Cube (dextro)"),
    ('jaD', 'Hull', "Joined Icosidodecahedron"),
    ('jbC', 'Hull', "Joined Truncated Cuboctahedron"),
    ('jtI', 'Hull', "Joined Truncated Icosahedron"),
    ('jtD', 'Hull', "Joined Truncated Dodecahedron"),
    ('jeD', 'Hull', "Joined Rhombicosidodecahedron"),
    ('jsD', 'Hull', "Joined Snub Dodecahedron (laevo)"),
    ('rjsD', 'Hull', "Joined Snub Dodecahedron (dextro)"),
    ('jbD', 'Hull', "Joined Truncated Icosidodecahedron"),
    # Propellor solids (George Hart's propellor operator).
    ('pT', 'Propellor', "Propello Tetrahedron"),
    ('pC', 'Propellor', "Propello Cube"),
    ('pO', 'Propellor', "Propello Octahedron"),
    ('pD', 'Propellor', "Propello Dodecahedron"),
    ('pI', 'Propellor', "Propello Icosahedron"),
    ('ptO', 'Propellor', "Propello Truncated Octahedron"),
    ('pkC', 'Propellor', "Propello Tetrakis Hexahedron"),
    ('psC', 'Propellor', "Propello Snub Cube"),
    ('prgC', 'Propellor', "Propello Pentagonal Icositetrahedron"),
    ('pbC', 'Propellor', "Propello Truncated Cuboctahedron"),
    ('pmC', 'Propellor', "Propello Disdyakis Dodecahedron"),
    ('ptI', 'Propellor', "Propello Truncated Icosahedron"),
    ('pkD', 'Propellor', "Propello Pentakis Dodecahedron"),
    ('pbD', 'Propellor', "Propello Truncated Icosidodecahedron"),
    ('pmD', 'Propellor', "Propello Disdyakis Triacontahedron"),
    # Truncated Archimedean solids (truncate all vertices, then canonicalize).
    ('ttT', 'Truncated Archimedean', "Truncated Truncated Tetrahedron"),
    ('ttO', 'Truncated Archimedean', "Truncated Truncated Octahedron"),
    ('ttC', 'Truncated Archimedean', "Truncated Truncated Cube"),
    ('teC', 'Truncated Archimedean', "Truncated Rhombicuboctahedron"),
    ('tsC', 'Truncated Archimedean', "Truncated Snub Cube"),
    ('tbC', 'Truncated Archimedean', "Truncated Truncated Cuboctahedron"),
    ('ttI', 'Truncated Archimedean', "Truncated Truncated Icosahedron"),
    ('ttD', 'Truncated Archimedean', "Truncated Truncated Dodecahedron"),
    ('teD', 'Truncated Archimedean', "Truncated Rhombicosidodecahedron"),
    ('tsD', 'Truncated Archimedean', "Truncated Snub Dodecahedron"),
    ('tbD', 'Truncated Archimedean', "Truncated Truncated Icosidodecahedron"),
    # Rectified Archimedean solids (ambo, then canonicalize).
    ('atT', 'Rectified Archimedean', "Rectified Truncated Tetrahedron"),
    ('atO', 'Rectified Archimedean', "Rectified Truncated Octahedron"),
    ('atC', 'Rectified Archimedean', "Rectified Truncated Cube"),
    ('aeC', 'Rectified Archimedean', "Rectified Rhombicuboctahedron"),
    ('asC', 'Rectified Archimedean', "Rectified Snub Cube"),
    ('abC', 'Rectified Archimedean', "Rectified Truncated Cuboctahedron"),
    ('atI', 'Rectified Archimedean', "Rectified Truncated Icosahedron"),
    ('atD', 'Rectified Archimedean', "Rectified Truncated Dodecahedron"),
    ('aeD', 'Rectified Archimedean', "Rectified Rhombicosidodecahedron"),
    ('asD', 'Rectified Archimedean', "Rectified Snub Dodecahedron"),
    ('abD', 'Rectified Archimedean', "Rectified Truncated Icosidodecahedron"),
    # Chamfered solids (chamfer, then canonicalize).
    ('cT', 'Chamfered', "Chamfered Tetrahedron"),
    ('cO', 'Chamfered', "Chamfered Octahedron"),
    ('cC', 'Chamfered', "Chamfered Cube"),
    ('cI', 'Chamfered', "Chamfered Icosahedron"),
    ('cD', 'Chamfered', "Chamfered Dodecahedron"),
    ('ctI', 'Chamfered', "Chamfered Truncated Icosahedron"),
    # Dipyramids and trapezohedra (duals of the uniform prisms/antiprisms).
    ('dP3', 'Dipyramid / Trapezohedron', "Triangular Dipyramid"),
    ('dP4', 'Dipyramid / Trapezohedron', "Square Dipyramid (Octahedron)"),
    ('dP5', 'Dipyramid / Trapezohedron', "Pentagonal Dipyramid"),
    ('dP6', 'Dipyramid / Trapezohedron', "Hexagonal Dipyramid"),
    ('dP7', 'Dipyramid / Trapezohedron', "Heptagonal Dipyramid"),
    ('dP8', 'Dipyramid / Trapezohedron', "Octagonal Dipyramid"),
    ('dA3', 'Dipyramid / Trapezohedron', "Trigonal Trapezohedron (Cube)"),
    ('dA4', 'Dipyramid / Trapezohedron', "Tetragonal Trapezohedron"),
    ('dA5', 'Dipyramid / Trapezohedron', "Pentagonal Trapezohedron"),
    ('dA6', 'Dipyramid / Trapezohedron', "Hexagonal Trapezohedron"),
    ('dA7', 'Dipyramid / Trapezohedron', "Heptagonal Trapezohedron"),
    ('dA8', 'Dipyramid / Trapezohedron', "Octagonal Trapezohedron"),
]


# --------------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------------

try:
    import bpy
    from bpy.props import (StringProperty, EnumProperty, IntProperty,
                           FloatProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    EXAMPLES = [
        ('CUSTOM', "Custom", ""),
        ('tI', "Truncated Icosahedron (tI)", "the football"),
        ('sC', "Snub Cube (sC)", ""),
        ('eD', "Rhombicosidodecahedron (eD)", ""),
        ('bT', "Bevelled Tetrahedron (bT)", ""),
        ('gD', "Pentagonal Hexecontahedron (gD)", ""),
        ('cC', "Chamfered Cube (cC)", ""),
        ('pC', "Propellor Cube (pC)", "chiral"),
        ('pkD', "Propellor Pentakis (pkD)", ""),
        ('dkt5daD', "Ornate (dkt5daD)", ""),
        ('kD', "Pentakis Dodecahedron (kD)", ""),
        ('tkD', "Truncated Pentakis (tkD)", ""),
        ('ccO', "Twice-Chamfered Octahedron (ccO)", ""),
    ]

    class MESH_OT_conway_add(bpy.types.Operator):
        """Build a polyhedron from Conway notation (e.g. dkC, taD, k3sT).
        Seeds: T C O D I, Pn, An, Yn; ops: d a k g c r t j e o b m s n z"""
        bl_idname = "mesh.conway_add"
        bl_label = "Conway Polyhedron"
        bl_options = {'REGISTER', 'UNDO'}

        def _example_chosen(self, context):
            if self.example != 'CUSTOM':
                self.notation = self.example

        example: EnumProperty(name="Example", items=EXAMPLES,
                              default='tI', update=_example_chosen)
        notation: StringProperty(
            name="Notation", default="tI",
            description="Operators then seed, applied right to left")
        post: EnumProperty(
            name="Geometry",
            items=[('CANON', "Canonical",
                    "Edges tangent to sphere, planar faces (Hart)"),
                   ('SPHERE', "Spherized", "Project vertices to a sphere"),
                   ('RAW', "Raw", "Whatever the operators produce")],
            default='CANON')
        iterations: IntProperty(name="Canonical Iterations", default=200,
                                min=5, max=2000)
        kis_height: FloatProperty(name="Kis Height", default=0.25,
                                  min=-1.0, max=2.0)
        coloring: EnumProperty(
            name="Coloring",
            items=[('SIDES', "Colored (by face sides)",
                    "One material per face size, as in Hart's 'colored' "
                    "display (view with Material Preview or Solid "
                    "shading set to Material color)"),
                   ('NONE', "None", "No materials")],
            default='SIDES')
        uv_map: BoolProperty(
            name="Spherical UV Map", default=True,
            description="Smooth equirectangular UVs projected from the "
                        "centre (seam-corrected per face)")
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", "Plain closed polyhedron"),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo Style "
                    "Geometry Nodes modifier (Border and Thickness "
                    "stay editable on the modifier)"),
                   ('WIRE', "Wireframe",
                    "Struts along the edges (Wireframe modifier)")],
            default='SOLID')
        border: FloatProperty(
            name="Border", default=0.3, min=0.02, max=0.95,
            description="Leonardo face frame width (fraction of "
                        "the face)")
        thickness: FloatProperty(
            name="Thickness", default=0.05, min=0.001, max=1.0,
            description="Panel / strut thickness for the Leonardo "
                        "and Wireframe styles")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        # Hart-style palette per face size; golden-angle HSV fallback
        _PALETTE = {3: (0.90, 0.36, 0.23), 4: (0.27, 0.52, 0.79),
                    5: (0.30, 0.69, 0.42), 6: (0.95, 0.77, 0.29),
                    7: (0.62, 0.40, 0.75), 8: (0.25, 0.72, 0.72),
                    9: (0.91, 0.56, 0.71), 10: (0.55, 0.60, 0.29),
                    12: (0.52, 0.45, 0.40)}

        @classmethod
        def _material_for(cls, n):
            name = f"Conway {n}-gon"
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name)
                if n in cls._PALETTE:
                    rgb = cls._PALETTE[n]
                else:
                    import colorsys
                    rgb = colorsys.hsv_to_rgb((n * 0.618034) % 1.0,
                                              0.55, 0.8)
                mat.diffuse_color = (*rgb, 1.0)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.55
            return mat

        def execute(self, context):
            try:
                V, F = apply_conway(self.notation,
                                    kis_height=self.kis_height)
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            if self.post == 'SPHERE':
                V = spherize(V, F)
            elif self.post == 'CANON':
                V = canonicalize(V, F, iters=self.iterations)
            V, F = orient_outward(V, [list(f) for f in F])
            me = bpy.data.meshes.new(f"Conway {self.notation}")
            me.from_pydata([tuple(c * self.scale for c in v) for v in V],
                           [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            if self.coloring == 'SIDES' and len(me.polygons) == len(F):
                sides = sorted({len(f) for f in F})
                slot = {n: i for i, n in enumerate(sides)}
                for n in sides:
                    me.materials.append(self._material_for(n))
                me.polygons.foreach_set(
                    'material_index', [slot[len(f)] for f in F])
            # face metadata: number of sides, usable in shaders (the
            # Attribute node) and Geometry Nodes
            if len(me.polygons) == len(F):
                attr = me.attributes.new("ngon_sides", 'INT', 'FACE')
                attr.data.foreach_set('value', [len(f) for f in F])
            if self.uv_map:
                uvl = me.uv_layers.new(name="UVMap")
                two_pi = 2 * math.pi
                for poly in me.polygons:
                    uvs = []
                    for li in poly.loop_indices:
                        vi = me.loops[li].vertex_index
                        x, y, z = me.vertices[vi].co
                        r = math.sqrt(x * x + y * y + z * z) or 1.0
                        u = math.atan2(y, x) / two_pi + 0.5
                        vv = math.asin(max(-1.0, min(1.0, z / r))) \
                            / math.pi + 0.5
                        uvs.append([u, vv, abs(z / r) > 0.999])
                    us = [u for u, vv, pole in uvs if not pole]
                    if us:                               # seam wrap
                        ref = max(us)
                        for q in uvs:
                            if not q[2] and ref - q[0] > 0.5:
                                q[0] += 1.0
                        us = [q[0] for q in uvs if not q[2]]
                    if us:                               # poles: average u
                        um = sum(us) / len(us)
                        for q in uvs:
                            if q[2]:
                                q[0] = um
                    for li, q in zip(poly.loop_indices, uvs):
                        uvl.data[li].uv = (q[0], q[1])
            me.update()
            obj = bpy.data.objects.new(f"Conway {self.notation}", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if self.style == 'LEONARDO':
                try:
                    from . import leonardo_style
                except ImportError:
                    import leonardo_style
                leonardo_style.add_modifier(obj, self.border,
                                            self.thickness)
            elif self.style == 'WIRE':
                mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
                mod.thickness = self.thickness
                mod.use_even_offset = False
            self.report({'INFO'},
                        f"{self.notation}: V={len(V)} F={len(F)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'example')
            lay.prop(self, 'notation')
            lay.prop(self, 'post')
            if self.post == 'CANON':
                lay.prop(self, 'iterations')
            lay.prop(self, 'kis_height')
            lay.prop(self, 'coloring')
            lay.prop(self, 'uv_map')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style != 'SOLID':
                lay.prop(self, 'thickness')
            lay.prop(self, 'scale')

    _CAT_LABELS = {
        'Hull': "Archimedean-Catalan Hulls",
        'Propellor': "Propellor Solids",
        'Truncated Archimedean': "Truncated Archimedean",
        'Rectified Archimedean': "Rectified Archimedean",
        'Chamfered': "Chamfered Solids",
        'Dipyramid / Trapezohedron': "Dipyramids & Trapezohedra",
    }

    def _catalog_category_items():
        seen = []
        for _n, cat, _name in CATALOG:
            if cat not in seen:
                seen.append(cat)
        return [(cat, _CAT_LABELS.get(cat, cat), "") for cat in seen]

    _CAT_ITEM_CACHE = {}

    def _catalog_solid_items(self, context):
        # constituents of the currently selected family (cached so Blender
        # keeps the item strings alive)
        cat = self.category
        if cat not in _CAT_ITEM_CACHE:
            _CAT_ITEM_CACHE[cat] = [(n, name, "Conway: " + n)
                                    for n, c, name in CATALOG if c == cat]
        return _CAT_ITEM_CACHE[cat]

    class MESH_OT_conway_catalog_add(bpy.types.Operator):
        """Build a named polyhedron from the curated catalog: Archimedean-
        Catalan hulls, propellor solids, truncated/rectified Archimedean
        solids, chamfered solids, and dipyramids/trapezohedra.  Each is a
        Conway construction on an exact seed, then canonicalized."""
        bl_idname = "mesh.conway_catalog_add"
        bl_label = "Polyhedron Catalog (Hulls, Propellor, ...)"
        bl_options = {'REGISTER', 'UNDO'}

        category: EnumProperty(name="Family", items=_catalog_category_items())
        solid: EnumProperty(name="Solid", items=_catalog_solid_items)
        iterations: IntProperty(name="Canonical Iterations", default=200,
                                min=5, max=2000)
        coloring: EnumProperty(
            name="Coloring",
            items=[('SIDES', "Colored (by face sides)", ""),
                   ('NONE', "None", "")],
            default='SIDES')
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'category')
            lay.prop(self, 'solid')
            lay.prop(self, 'iterations')
            lay.prop(self, 'coloring')
            lay.prop(self, 'scale')

        def execute(self, context):
            notation = self.solid
            try:
                V, F = apply_conway(notation)
            except (ValueError, KeyError) as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
            V = canonicalize(V, F, iters=self.iterations)
            V, F = orient_outward(V, [list(f) for f in F])
            label = dict((n, nm) for n, _, nm in CATALOG).get(notation,
                                                              notation)
            me = bpy.data.meshes.new(label)
            me.from_pydata([tuple(c * self.scale for c in v) for v in V],
                           [], [tuple(f) for f in F])
            me.validate(clean_customdata=True)
            if self.coloring == 'SIDES' and len(me.polygons) == len(F):
                sides = sorted({len(f) for f in F})
                slot = {n: i for i, n in enumerate(sides)}
                for n in sides:
                    me.materials.append(
                        MESH_OT_conway_add._material_for(n))
                me.polygons.foreach_set(
                    'material_index', [slot[len(f)] for f in F])
            if len(me.polygons) == len(F):
                attr = me.attributes.new("ngon_sides", 'INT', 'FACE')
                attr.data.foreach_set('value', [len(f) for f in F])
            me.update()
            obj = bpy.data.objects.new(label, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{label}: V={len(V)} F={len(F)}")
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.conway_add", icon='MESH_ICOSPHERE')
        self.layout.operator("mesh.conway_catalog_add", icon='MESH_ICOSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_conway_add)
        bpy.utils.register_class(MESH_OT_conway_catalog_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_conway_catalog_add)
        bpy.utils.unregister_class(MESH_OT_conway_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        def euler(V, F):
            E = set()
            for f in F:
                for i in range(len(f)):
                    a, b = f[i], f[(i + 1) % len(f)]
                    E.add((min(a, b), max(a, b)))
            return len(V) - len(E) + len(F)
        for s, expect in [("C", (8, 6)), ("dC", (6, 8)), ("aC", (12, 14)),
                          ("tC", (24, 14)), ("tI", (60, 32)),
                          ("sC", (24, 38)), ("gC", (38, 24)),
                          ("cC", (32, 18)), ("eD", (60, 62)),
                          ("jC", (14, 12)), ("mC", (26, 48)),
                          ("bC", (48, 26)), ("kD", (32, 60)),
                          ("pC", (32, 30)), ("pT", (16, 16)),
                          ("pD", (80, 72)), ("dpC", (30, 32)),
                          ("pdC", (30, 32)),
                          ("P6", (12, 8)), ("A5", (10, 12)),
                          ("dA5", (12, 10)), ("Y4", (5, 5))]:
            V, F = apply_conway(s)
            ok = (len(V), len(F)) == expect and euler(V, F) == 2
            print(f"{s:6s}: V={len(V):3d} F={len(F):3d} chi=2:"
                  f"{euler(V, F) == 2}  expect {expect} "
                  f"{'OK' if ok else 'MISMATCH'}")
        if np is not None:
            V, F = apply_conway("sD")
            V2 = canonicalize(V, F, iters=40)
            print(f"sD canonicalized: V={len(V2)} F={len(F)} "
                  f"chi2={euler(V2, F) == 2}")
        bad = 0
        for notation, cat, name in CATALOG:
            V, F = apply_conway(notation)
            if euler(V, F) != 2:
                print(f"CATALOG MISMATCH {notation} ({name}) chi="
                      f"{euler(V, F)}")
                bad += 1
        print(f"catalog: {len(CATALOG)} named solids, "
              f"{'all chi=2 OK' if bad == 0 else str(bad) + ' BAD'}")
