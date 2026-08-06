"""
general_stellation.py -- stellation engine generalized to arbitrary convex
seed polyhedra (pure Python + numpy; no scipy).

Generalizes math_art/stellation_engine.py (icosahedron-only) by:
  * planes n_i . x = d_i with per-plane distances (seeds need not be
    isohedral: the cuboctahedron has two face-distance classes),
  * a full-dimensionality (rank-3) check on candidate cells -- degenerate
    flat regions occur in general symmetric arrangements,
  * convex-hull facet cycling instead of a plain angle sort (a triple-
    intersection point can lie strictly inside a facet for general seeds),
  * the seed's own symmetry group, computed from the vertex set, acting
    EXACTLY on cells via plane permutations (no floating centroid lookup).

Same core strategy as the icosahedron engine: enumerate the bounded cells
of the face-plane arrangement by sign-vector BFS with a big bounding cube
for the boundedness test, group cells into symmetry orbits ("shells"), and
build the outward boundary of any union of orbits.

API:
  stellations_of(seed)   seed = name | (V, F) | V-array  ->  StellationEngine
  engine.build(cell_code)  cell_code = 'all' | iterable of shell labels
                           and/or frozensets of cell sign-vectors -> (V, F)
  named_presets(seed_name) -> [(key, title, code, note), ...]  (verified)
  build_named(seed_name, key) -> (V, F)  convenience for the presets
Built-in seeds (SEEDS): 'icosahedron', 'dodecahedron', 'cuboctahedron',
'rhombic_triacontahedron' (constructed as the polar dual of the
icosidodecahedron).  Custom convex seeds: pass a vertex array (origin
must be interior; at most 60 face planes).

Verification (via _selftest()):  for every seed the self-test
checks that the core cell reproduces the seed exactly, and validates the
classical named stellations -- the icosahedron against Du Val's shell
tally / echidnahedron / great icosahedron / the tetrahedra+octahedra
compounds, the dodecahedron against its three stellations, the
cuboctahedron against the cube, octahedron and cube-octahedron compound
(Wenninger 43), and the rhombic triacontahedron against the compound of
five cubes plus the medial and great rhombic triacontahedra (the latter
two derived independently as polar duals of the dodecadodecahedron and
great icosidodecahedron and mapped to cells by winding number).

References (mathematics implemented here):
  * H. S. M. Coxeter, P. Du Val, H. T. Flather, J. F. Petrie,
    "The Fifty-Nine Icosahedra", Univ. of Toronto Studies (1938).
  * M. Wenninger, "Polyhedron Models", CUP (1971) -- stellations of the
    cuboctahedron (models 43-46) and of the dodecahedron (models 19-22).
  * M. Wenninger, "Dual Models", CUP (1983) -- medial / great rhombic
    triacontahedra as duals of uniform star polyhedra.
  * H. M. Cundy, A. P. Rollett, "Mathematical Models", OUP (1961) --
    the compound of five cubes in the rhombic triacontahedron.
"""

import math
import random
from collections import deque, defaultdict, Counter

import numpy as np

PHI = (1.0 + 5.0 ** 0.5) / 2.0
_TOL = 1e-7          # sign-classification tolerance (matches icosa engine)
_PTROUND = 6         # arrangement-point dedup rounding (matches icosa engine)
_BIG = 50.0          # bounding-cube half-size for the boundedness test


# --------------------------------------------------------------------------
# seed library
# --------------------------------------------------------------------------
def _cyc(p):
    x, y, z = p
    return [(x, y, z), (z, x, y), (y, z, x)]


def _sign_spread(p):
    out = set()
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                out.add((sx * p[0], sy * p[1], sz * p[2]))
    return out


def _vertset(patterns):
    pts = set()
    for p in patterns:
        for c in _cyc(p):
            pts |= _sign_spread(c)
    return np.array(sorted(pts), float)


def _icosahedron_V():
    return _vertset([(0.0, 1.0, PHI)])                      # 12


def _dodecahedron_V():
    return _vertset([(1.0, 1.0, 1.0), (0.0, 1.0 / PHI, PHI)])   # 20


def _cuboctahedron_V():
    return _vertset([(1.0, 1.0, 0.0)])                      # 12


def _icosidodecahedron_V():
    return _vertset([(0.0, 0.0, PHI),
                     (0.5, PHI / 2.0, PHI * PHI / 2.0)])    # 30


def hull_planes(V, tol=1e-7):
    """Supporting face planes (n unit, n.x = d, d > 0) of a convex vertex
    set with the origin interior.  Brute force over vertex triples."""
    V = np.asarray(V, float)
    n = len(V)
    planes = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                nv = np.cross(V[j] - V[i], V[k] - V[i])
                L = np.linalg.norm(nv)
                if L < 1e-9:
                    continue
                nv = nv / L
                d = float(nv @ V[i])
                s = V @ nv - d
                if s.max() <= tol:
                    pass
                elif s.min() >= -tol:
                    nv, d = -nv, -d
                else:
                    continue
                for (n2, d2) in planes:
                    if abs(d - d2) < 1e-6 and np.linalg.norm(nv - n2) < 1e-6:
                        break
                else:
                    planes.append((nv.copy(), d))
    N = np.array([p[0] for p in planes])
    D = np.array([p[1] for p in planes])
    return N, D


def _rhombic_triacontahedron_V():
    """RT as the polar dual of the icosidodecahedron (vertex of dual per
    face plane: n / d)."""
    W = _icosidodecahedron_V()
    N, D = hull_planes(W)
    return N / D[:, None]


_SEED_BUILDERS = {
    'icosahedron': _icosahedron_V,
    'dodecahedron': _dodecahedron_V,
    'cuboctahedron': _cuboctahedron_V,
    'rhombic_triacontahedron': _rhombic_triacontahedron_V,
}


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _frame(v0, u0):
    e1 = v0 / np.linalg.norm(v0)
    w = u0 - (u0 @ e1) * e1
    e2 = w / np.linalg.norm(w)
    e3 = np.cross(e1, e2)
    return np.array([e1, e2, e3])


def _facet_cycle(pts, outward, eps=1e-9):
    """Boundary cycle (index list into pts) of the convex facet spanned by
    pts, wound CCW about `outward`.  Keeps collinear boundary points (they
    are shared with neighbouring faces -- dropping them would break the
    edge-matching / watertightness of the union boundary); drops strictly
    interior points.  Andrew's monotone chain with a collinear-keeping pop
    condition."""
    n = outward / np.linalg.norm(outward)
    a = np.array([1.0, 0.0, 0.0])
    if abs(n[0]) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)                    # (u, v, n) right-handed
    P2 = np.stack([pts @ u, pts @ v], axis=1)
    m = len(P2)
    order = sorted(range(m), key=lambda i: (P2[i, 0], P2[i, 1]))

    def cr(o, a_, b_):
        return ((P2[a_, 0] - P2[o, 0]) * (P2[b_, 1] - P2[o, 1])
                - (P2[a_, 1] - P2[o, 1]) * (P2[b_, 0] - P2[o, 0]))

    lower = []
    for i in order:
        while len(lower) >= 2 and cr(lower[-2], lower[-1], i) < -eps:
            lower.pop()
        lower.append(i)
    upper = []
    for i in reversed(order):
        while len(upper) >= 2 and cr(upper[-2], upper[-1], i) < -eps:
            upper.pop()
        upper.append(i)
    cyc = lower[:-1] + upper[:-1]
    if len(cyc) < 3:
        return None
    # orientation sanity: signed area must be positive (CCW about outward)
    area = 0.0
    for t in range(len(cyc)):
        x1, y1 = P2[cyc[t]]
        x2, y2 = P2[cyc[(t + 1) % len(cyc)]]
        area += x1 * y2 - x2 * y1
    if area < 0:
        cyc.reverse()
    return cyc


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------
class StellationEngine(object):

    def __init__(self, V, name='custom', big=_BIG, verbose=False):
        V = np.asarray(V, float)
        N, D = hull_planes(V)
        if len(N) > 60:
            raise ValueError('too many face planes (%d > 60)' % len(N))
        scale = 1.0 / D.min()             # normalize: min face distance = 1
        self.name = name
        self.seedV = V * scale
        self.N = N
        self.D = D * scale
        self.m = len(N)
        self.big = big
        self.verbose = verbose
        self.seedF = self._seed_faces()
        self._points()
        self._enumerate()
        self._symmetry()
        self._orbits()

    # ---- seed faces (vertex rings per plane) -----------------------------
    def _seed_faces(self):
        F = []
        for i in range(self.m):
            on = np.nonzero(np.abs(self.seedV @ self.N[i] - self.D[i])
                            < 1e-6)[0]
            if len(on) < 3:
                raise ValueError('face plane with <3 vertices')
            cyc = _facet_cycle(self.seedV[on], self.N[i])
            F.append([int(on[t]) for t in cyc])
        return F

    # ---- arrangement points ----------------------------------------------
    def _points(self):
        eqs = [(self.N[i], self.D[i]) for i in range(self.m)]
        for ax in range(3):
            for s in (1.0, -1.0):
                nv = np.zeros(3)
                nv[ax] = s
                eqs.append((nv, self.big))
        pts = {}
        me = len(eqs)
        A = np.empty((3, 3))
        b = np.empty(3)
        for i in range(me):
            for j in range(i + 1, me):
                for k in range(j + 1, me):
                    A[0], A[1], A[2] = eqs[i][0], eqs[j][0], eqs[k][0]
                    det = np.linalg.det(A)
                    if abs(det) < 1e-9:
                        continue
                    b[0], b[1], b[2] = eqs[i][1], eqs[j][1], eqs[k][1]
                    p = np.linalg.solve(A, b)
                    if np.max(np.abs(p)) > self.big + 1e-6:
                        continue
                    key = tuple(np.round(p, _PTROUND))
                    if key not in pts:
                        pts[key] = p
        self.P = np.array(list(pts.values()))
        dots = self.P @ self.N.T - self.D
        SIG = np.zeros(dots.shape, np.int8)
        SIG[dots > _TOL] = 1
        SIG[dots < -_TOL] = -1
        self.SIG = SIG
        self.oncube = (np.max(np.abs(self.P), axis=1) > self.big - 1e-6)
        pos = np.zeros(len(self.P), np.uint64)
        nz = np.zeros(len(self.P), np.uint64)
        for i in range(self.m):
            bit = np.uint64(1 << i)
            pos |= np.where(SIG[:, i] == 1, bit, np.uint64(0))
            nz |= np.where(SIG[:, i] != 0, bit, np.uint64(0))
        self.posbits = pos
        self.nzbits = nz

    def _region(self, sbits):
        mask = ((self.posbits ^ sbits) & self.nzbits) == 0
        return np.nonzero(mask)[0]

    # ---- bounded-cell enumeration (sign-vector BFS) ----------------------
    def _enumerate(self):
        m = self.m
        core = tuple([-1] * m)
        seen = {core}
        dq = deque([core])
        cells = {}
        guard = 0
        while dq:
            guard += 1
            if guard > 2000000:
                raise RuntimeError('BFS runaway')
            s = dq.popleft()
            sbits = np.uint64(sum(1 << i for i in range(m) if s[i] == 1))
            vi = self._region(sbits)
            if len(vi) < 4:
                continue
            unbounded = bool(self.oncube[vi].any())
            onc = (self.SIG[vi] == 0).sum(axis=0)
            fplanes = np.nonzero(onc >= 3)[0]
            if not unbounded:
                pts = self.P[vi]
                c = pts.mean(0)
                sv = np.linalg.svd(pts - c, compute_uv=False)
                if sv[2] > 1e-6:            # full-dimensional only
                    cells[s] = {
                        'signs': s,
                        'vi': vi,
                        'fplanes': fplanes,
                        'centroid': c,
                        'power': sum(1 for x in s if x == 1),
                        'radius': float(np.linalg.norm(c)),
                    }
            for i in fplanes:               # cross each touching plane
                ns = list(s)
                ns[i] = -ns[i]
                ns = tuple(ns)
                if ns not in seen:
                    seen.add(ns)
                    dq.append(ns)
        self.cells = cells
        self.n_searched = len(seen)

    # ---- symmetry group of the seed (from the vertex set) ----------------
    def _symmetry(self):
        V = self.seedV
        n = len(V)
        r = np.linalg.norm(V, axis=1)
        classes = defaultdict(list)
        for i in range(n):
            classes[round(float(r[i]), 6)].append(i)
        base = min(classes.values(), key=len)
        v0 = V[base[0]]
        dist = np.linalg.norm(V - v0, axis=1)
        j0 = None
        for j in np.argsort(dist):
            if dist[j] > 1e-6 and np.linalg.norm(np.cross(v0, V[j])) > 1e-6:
                j0 = int(j)
                break
        u0 = V[j0]
        F0 = _frame(v0, u0)
        r0 = float(np.linalg.norm(v0))
        r1 = float(np.linalg.norm(u0))
        dot0 = float(v0 @ u0)
        rots, imps = {}, {}
        for ai in range(n):
            a = V[ai]
            if abs(np.linalg.norm(a) - r0) > 1e-6:
                continue
            for bi in range(n):
                bv = V[bi]
                if abs(np.linalg.norm(bv) - r1) > 1e-6:
                    continue
                if abs(float(a @ bv) - dot0) > 1e-6:
                    continue
                if np.linalg.norm(np.cross(a, bv)) < 1e-6:
                    continue
                F1 = _frame(a, bv)
                for hand in (1.0, -1.0):
                    F1h = np.array([F1[0], F1[1], hand * F1[2]])
                    R = F1h.T @ F0
                    RV = V @ R.T
                    d2 = ((RV[:, None, :] - V[None, :, :]) ** 2).sum(-1)
                    if d2.min(axis=1).max() > 1e-12:
                        continue
                    key = tuple(np.round(R, 6).ravel())
                    if np.linalg.det(R) > 0:
                        rots[key] = R
                    else:
                        imps[key] = R
        self.rotations = list(rots.values())
        self.impropers = list(imps.values())

    def _plane_perm(self, R):
        RN = self.N @ R.T
        d2 = ((RN[:, None, :] - self.N[None, :, :]) ** 2).sum(-1)
        perm = d2.argmin(axis=1)
        if (d2[np.arange(self.m), perm] > 1e-12).any():
            raise RuntimeError('symmetry does not permute the face planes')
        if (np.abs(self.D[perm] - self.D) > 1e-6).any():
            raise RuntimeError('symmetry mismatches plane distances')
        if len(set(perm.tolist())) != self.m:
            raise RuntimeError('plane permutation not a bijection')
        return perm

    # ---- cell orbits via exact plane permutations ------------------------
    def _orbits(self):
        cl = list(self.cells.keys())
        nc = len(cl)
        idx = {s: i for i, s in enumerate(cl)}
        C = np.array(cl, np.int8)                     # ncells x m
        self.rot_perms = [self._plane_perm(R) for R in self.rotations]
        self.imp_perms = [self._plane_perm(R) for R in self.impropers]

        def orbits_under(perms):
            parent = list(range(nc))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for perm in perms:
                inv = np.empty(self.m, int)
                inv[perm] = np.arange(self.m)
                mapped = C[:, inv]                    # row i -> image signs
                for i in range(nc):
                    j = idx.get(tuple(mapped[i]))
                    if j is None:
                        raise RuntimeError('cell image missing: arrangement '
                                           'not symmetric?')
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
            groups = defaultdict(list)
            for i in range(nc):
                groups[find(i)].append(cl[i])
            return list(groups.values())

        full = orbits_under(self.rot_perms + self.imp_perms)
        rot = orbits_under(self.rot_perms)
        shells = []
        for o in full:
            pw = self.cells[o[0]]['power']
            mr = float(np.mean([self.cells[s]['radius'] for s in o]))
            shells.append({'power': pw, 'size': len(o), 'radius': mr,
                           'cells': frozenset(o)})
        shells.sort(key=lambda d: (d['power'], d['radius']))
        rot_orbs = [frozenset(o) for o in rot]
        for k, sh in enumerate(shells):
            sh['hands'] = [o for o in rot_orbs if o & sh['cells']]
            sh['chiral'] = len(sh['hands']) == 2
            sh['label'] = 'a' if (sh['power'] == 0 and sh['size'] == 1) \
                else 's%02d' % k
        self.shells = shells
        self.rot_orbits = rot_orbs
        self.labels = [sh['label'] for sh in shells]
        self.shell_by_label = {sh['label']: sh for sh in shells}

    # ---- cell-code handling ----------------------------------------------
    def cells_of_code(self, cell_code):
        if isinstance(cell_code, str):
            cell_code = [cell_code]
        def _is_cell(x):
            return (isinstance(x, tuple) and len(x) == self.m
                    and all(v in (-1, 1) for v in x))

        filled = set()
        for item in cell_code:
            if isinstance(item, str):
                if item in ('all', 'final'):
                    filled |= set(self.cells.keys())
                else:
                    filled |= self.shell_by_label[item]['cells']
            elif _is_cell(item):
                filled.add(item)
            else:
                filled |= set(item)
        return filled

    # ---- boundary surface of a union of cells ----------------------------
    def build(self, cell_code, scale=True):
        filled = self.cells_of_code(cell_code)
        polys = []
        for s in filled:
            cell = self.cells[s]
            vi = cell['vi']
            sig = self.SIG[vi]
            for i in cell['fplanes']:
                ns = list(s)
                ns[i] = -ns[i]
                if tuple(ns) in filled:
                    continue                          # interior face
                on = vi[sig[:, i] == 0]
                if len(on) < 3:
                    continue
                outward = -s[i] * self.N[i]
                cyc = _facet_cycle(self.P[on], outward)
                if cyc is not None:
                    polys.append(self.P[on][cyc])
        vmap = {}
        Vout = []
        Fout = []
        for poly in polys:
            idxs = []
            for p in poly:
                key = tuple(np.round(p, 6))
                if key not in vmap:
                    vmap[key] = len(Vout)
                    Vout.append((float(p[0]), float(p[1]), float(p[2])))
                idxs.append(vmap[key])
            clean = [idxs[0]]
            for j in idxs[1:]:
                if j != clean[-1]:
                    clean.append(j)
            while len(clean) >= 2 and clean[0] == clean[-1]:
                clean.pop()
            if len(clean) >= 3:
                Fout.append(clean)
        if scale and Vout:
            mx = max(max(abs(c) for c in v) for v in Vout)
            if mx > 0:
                Vout = [(x / mx, y / mx, z / mx) for (x, y, z) in Vout]
        return Vout, Fout

    # ---- exact convex volume of one cell ---------------------------------
    def cell_volume(self, s):
        cell = self.cells[s]
        vi = cell['vi']
        pts = self.P[vi]
        c0 = pts.mean(0)
        vol = 0.0
        for i in cell['fplanes']:
            on = pts[self.SIG[vi, i] == 0]
            outward = -s[i] * self.N[i]
            cyc = _facet_cycle(on, outward)
            if cyc is None:
                continue
            ring = on[cyc]
            for t in range(1, len(ring) - 1):
                vol += float(np.linalg.det(np.stack(
                    [ring[0] - c0, ring[t] - c0, ring[t + 1] - c0]))) / 6.0
        return abs(vol)

    # ---- cells fully inside a convex region (for named compounds) --------
    def cells_inside_convex(self, N2, D2, tol=1e-6):
        N2 = np.asarray(N2, float)
        D2 = np.asarray(D2, float)
        sel = set()
        for s, cell in self.cells.items():
            pts = self.P[cell['vi']]
            if ((pts @ N2.T) - D2 <= tol).all():
                sel.add(s)
        return sel

    def is_orbit_union(self, cellset):
        """True if the cell set is an exact union of full-symmetry orbits."""
        for sh in self.shells:
            inter = sh['cells'] & cellset
            if inter and inter != sh['cells']:
                return False
        return True

    def labels_of_cells(self, cellset):
        """Shell-label list covering the cell set exactly, or None if the
        set is not a union of full-symmetry orbits."""
        labs = []
        left = set(cellset)
        for sh in self.shells:
            inter = sh['cells'] & left
            if inter:
                if inter != sh['cells']:
                    return None
                labs.append(sh['label'])
                left -= sh['cells']
        return labs if not left else None

    # ---- report ----------------------------------------------------------
    def alias(self, label):
        """Classical alias of a generic shell label where known (Du Val's
        letters for the icosahedron); otherwise the label itself."""
        sh = self.shell_by_label.get(label)
        if sh is not None and self.name == 'icosahedron':
            return _DUVAL_BY_POWER_SIZE.get((sh['power'], sh['size']), label)
        return label

    def orbit_table(self):
        lines = ['  label  power  size  mean-radius  chiral',
                 '  -----  -----  ----  -----------  ------']
        for sh in self.shells:
            nm = sh['label']
            al = self.alias(nm)
            if al != nm:
                nm = '%s=%s' % (nm, al)
            lines.append('  %-6s %5d  %4d  %11.4f  %s'
                         % (nm, sh['power'], sh['size'],
                            sh['radius'],
                            'yes (2x%d)' % len(sh['hands'][0])
                            if sh['chiral'] else 'no'))
        lines.append('  total cells: %d   orbits: %d'
                     % (len(self.cells), len(self.shells)))
        return '\n'.join(lines)


# Du Val's classical shell letters for the icosahedron, keyed by
# (power, orbit size) -- the assignment is unambiguous.
_DUVAL_BY_POWER_SIZE = {
    (1, 20): 'b', (2, 30): 'c', (3, 60): 'd',
    (4, 20): 'e1', (4, 60): 'e2', (5, 120): 'f1', (5, 12): 'f2',
    (6, 30): 'g1', (6, 60): 'g2', (7, 60): 'g3',
}


# --------------------------------------------------------------------------
# constructor / cache
# --------------------------------------------------------------------------
_ENGINE_CACHE = {}


def stellations_of(seed, verbose=False):
    """Return a StellationEngine for a convex seed.

    seed: a name from the built-in library ('icosahedron', 'dodecahedron',
    'cuboctahedron', 'rhombic_triacontahedron'), a (V, F) pair, or a bare
    vertex array (faces are recomputed from the hull either way)."""
    if isinstance(seed, str):
        key = seed.lower()
        if key not in _ENGINE_CACHE:
            if key not in _SEED_BUILDERS:
                raise KeyError('unknown seed %r (have %s)'
                               % (seed, sorted(_SEED_BUILDERS)))
            _ENGINE_CACHE[key] = StellationEngine(
                _SEED_BUILDERS[key](), name=key, verbose=verbose)
        return _ENGINE_CACHE[key]
    if isinstance(seed, tuple) and len(seed) == 2:
        return StellationEngine(np.asarray(seed[0], float), name='custom',
                                verbose=verbose)
    return StellationEngine(np.asarray(seed, float), name='custom',
                            verbose=verbose)


# --------------------------------------------------------------------------
# surface verification helpers (ported from the icosahedron engine)
# --------------------------------------------------------------------------
def surface_stats(V, F):
    edges = Counter()
    for f in F:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            edges[(min(a, b), max(a, b))] += 1
    E = len(edges)
    two = all(v == 2 for v in edges.values())
    mult = dict(Counter(edges.values()))
    chi = len(V) - E + len(F)
    rs = [math.sqrt(x * x + y * y + z * z) for (x, y, z) in V] or [0.0]
    nan = any(not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z))
              for (x, y, z) in V)
    return {'V': len(V), 'E': E, 'F': len(F), 'chi': chi,
            'closed_2': two, 'edge_mult': mult,
            'rmin': min(rs), 'rmax': max(rs), 'nan': nan}


def signed_volume(V, F):
    """Signed volume via origin tetrahedra with fan-triangulated faces.
    Positive for a closed, outward-wound boundary."""
    vol = 0.0
    for f in F:
        v0 = V[f[0]]
        for t in range(1, len(f) - 1):
            a, b_, c = v0, V[f[t]], V[f[t + 1]]
            vol += (a[0] * (b_[1] * c[2] - b_[2] * c[1])
                    - a[1] * (b_[0] * c[2] - b_[2] * c[0])
                    + a[2] * (b_[0] * c[1] - b_[1] * c[0])) / 6.0
    return vol


def convex_hull_vertex_count(V, ndirs=800, tol=1e-6):
    """Number of convex-hull vertices of V: a point counts iff it is the
    UNIQUE maximizer of some sampled direction (plateau ties -- points on
    hull faces/edges -- are discarded).  Directions: every point's own
    direction plus random ones."""
    A = np.asarray(V, float)
    rnd = random.Random(0)
    dirs = []
    for v in A:
        L = np.linalg.norm(v)
        if L > 1e-12:
            dirs.append(v / L)
    for _ in range(ndirs):
        d = np.array([rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 1)])
        dirs.append(d / np.linalg.norm(d))
    D = np.array(dirs)
    M = D @ A.T                                  # ndirs x npts
    mx = M.max(axis=1)
    near = M > (mx[:, None] - tol)
    cnt = near.sum(axis=1)
    hull = set()
    for k in np.nonzero(cnt == 1)[0]:
        hull.add(int(M[k].argmax()))
    return len(hull)


def vertex_sets_match(V1, V2, tol=1e-6):
    """Greedy nearest matching of two point sets; True iff bijective
    within tol."""
    if len(V1) != len(V2):
        return False
    A = np.asarray(V1, float)
    B = np.asarray(V2, float)
    used = np.zeros(len(B), bool)
    for p in A:
        d2 = ((B - p) ** 2).sum(1)
        d2[used] = 1e18
        j = int(d2.argmin())
        if d2[j] > tol * tol:
            return False
        used[j] = True
    return True


def orthogonal_plane_cubes(N, D, tol=1e-6):
    """Inscribed cubes among the planes: sets {+-a,+-b,+-c} of mutually
    orthogonal plane pairs with equal distances (e.g. the five cubes of
    the rhombic triacontahedron)."""
    N = np.asarray(N, float)
    D = np.asarray(D, float)
    m = len(N)
    opp = {}
    for i in range(m):
        d2 = ((N + N[i]) ** 2).sum(1)
        j = int(d2.argmin())
        if d2[j] < 1e-12 and abs(D[j] - D[i]) < tol:
            opp[i] = j
    cubes = set()
    for i in range(m):
        for j in range(i + 1, m):
            if abs(float(N[i] @ N[j])) > tol or abs(D[i] - D[j]) > tol:
                continue
            for k in range(j + 1, m):
                if (abs(float(N[i] @ N[k])) > tol
                        or abs(float(N[j] @ N[k])) > tol
                        or abs(D[i] - D[k]) > tol):
                    continue
                if i in opp and j in opp and k in opp:
                    cubes.add(frozenset({i, j, k, opp[i], opp[j], opp[k]}))
    return [sorted(c) for c in cubes]


# --------------------------------------------------------------------------
# star-polyhedron duals for the rhombic triacontahedron's named stellations
# --------------------------------------------------------------------------
# The medial RT (dual of the dodecadodecahedron {5,5/2}) and the great RT
# (dual of the great icosidodecahedron {3,5/2}) have all 30 faces in the RT's
# face planes, so both are stellations of the RT.  We construct each dual
# from scratch (ring analysis of the icosidodecahedron's vertex set), then
# select the arrangement cells lying inside the self-intersecting surface by
# winding number.  This is an independent construction, so agreement with a
# shell union of the engine is a genuine verification.

def _icosidodeca_rings():
    W = _icosidodecahedron_V()
    R = float(np.linalg.norm(W[0]))
    N, D = hull_planes(W)
    pent, tri = [], []
    for i in range(len(N)):
        cnt = int((np.abs(W @ N[i] - D[i]) < 1e-6).sum())
        (pent if cnt == 5 else tri).append(i)
    if len(pent) != 12 or len(tri) != 20:
        raise RuntimeError('icosidodecahedron face classification failed')
    hA = float(D[pent[0]])                # outer pentagon ring (5 verts)
    u0 = N[pent[0]]
    hs5 = sorted({round(float(v), 6) for v in W @ u0 if v > 1e-6},
                 reverse=True)
    hB = hs5[1]                           # inner 5-ring: pentagram planes
    t0 = N[tri[0]]
    h1 = float(D[tri[0]])                 # icosidodeca's own triangle ring
    hs3 = sorted({round(float(v), 6) for v in W @ t0 if v > 1e-6},
                 reverse=True)
    deep3 = [h for h in hs3
             if int((np.abs(W @ t0 - h) < 1e-6).sum()) == 3
             and h < h1 - 1e-6][0]        # deep 3-ring: retrograde triangles
    return W, R, N, D, pent, tri, hA, hB, deep3


def rt_star_dual(kind):
    """Oriented rhombic faces (list of 4-point numpy arrays) of the medial
    ('medial') or great ('great') rhombic triacontahedron, in the same
    coordinates as stellations_of('rhombic_triacontahedron')."""
    W, R, N, D, pent, tri, hA, hB, deep3 = _icosidodeca_rings()
    faces = []                            # (axis, height, type)
    if kind == 'medial':                  # dual dodecadodecahedron {5,5/2}
        for i in pent:
            faces.append((N[i], hA, 0))   # pentagons
            faces.append((N[i], hB, 1))   # pentagrams
    elif kind == 'great':                 # dual great icosidodeca {3,5/2}
        for i in tri:
            faces.append((N[i], deep3, 0))   # retrograde triangles
        for i in pent:
            faces.append((N[i], hB, 1))      # pentagrams
    else:
        raise ValueError(kind)
    P = np.array([(R / h) * u for (u, h, _) in faces])   # dual vertices
    quads = []
    for w in W:                           # one rhombus per original vertex
        inc = [k for k, (u, h, _) in enumerate(faces)
               if abs(float(w @ u) - h) < 1e-6]
        if len(inc) != 4:
            raise RuntimeError('star vertex on %d faces' % len(inc))
        # quasiregular vertex figure: the two type-0 faces sit opposite each
        # other in the cycle, likewise the two type-1 faces -> alternate.
        t0s = [k for k in inc if faces[k][2] == 0]
        t1s = [k for k in inc if faces[k][2] == 1]
        if len(t0s) != 2 or len(t1s) != 2:
            raise RuntimeError('vertex figure does not alternate')
        pts = P[[t0s[0], t1s[0], t0s[1], t1s[1]]]
        n = w / np.linalg.norm(w)
        d = pts @ n
        if np.abs(d - d[0]).max() > 1e-6:
            raise RuntimeError('dual face not planar')
        quads.append(pts)
    return _orient_polys(quads)


def _orient_polys(polys):
    """Flip faces of a closed surface so every shared edge is traversed in
    opposite directions (consistent global orientation); asserts closure."""
    def vkey(p):
        return tuple(np.round(p, 6))

    edge2face = defaultdict(list)
    for f, q in enumerate(polys):
        for t in range(len(q)):
            k = frozenset([vkey(q[t]), vkey(q[(t + 1) % len(q)])])
            edge2face[k].append(f)
    for k, fs in edge2face.items():
        if len(fs) != 2:
            raise RuntimeError('surface edge in %d faces' % len(fs))
    flipped = [False] * len(polys)

    def directed(f):
        q = polys[f][::-1] if flipped[f] else polys[f]
        return {(vkey(q[t]), vkey(q[(t + 1) % len(q)]))
                for t in range(len(q))}

    seen = {0}
    stack = [0]
    while stack:
        f = stack.pop()
        df = directed(f)
        for t in range(len(polys[f])):
            q = polys[f]
            k = frozenset([vkey(q[t]), vkey(q[(t + 1) % len(q)])])
            for g in edge2face[k]:
                if g != f and g not in seen:
                    if df & directed(g):
                        flipped[g] = True
                    seen.add(g)
                    stack.append(g)
    out = [polys[f][::-1] if flipped[f] else polys[f]
           for f in range(len(polys))]
    dirs = Counter()
    for q in out:
        for t in range(len(q)):
            dirs[(vkey(q[t]), vkey(q[(t + 1) % len(q)]))] += 1
    if any(v != 1 for v in dirs.values()):
        raise RuntimeError('could not orient surface consistently')
    return out


def winding_numbers(points, polys):
    """Winding number of a closed, consistently oriented surface about each
    query point (Van Oosterom-Strackee signed solid angles, fan-triangulated
    faces).  Vectorized over the points."""
    P = np.asarray(points, float)
    if P.ndim == 1:
        P = P[None, :]
    tot = np.zeros(len(P))
    for q in polys:
        for t in range(1, len(q) - 1):
            A = q[0] - P
            B = q[t] - P
            C = q[t + 1] - P
            la = np.linalg.norm(A, axis=1)
            lb = np.linalg.norm(B, axis=1)
            lc = np.linalg.norm(C, axis=1)
            num = np.einsum('ij,ij->i', A, np.cross(B, C))
            den = (la * lb * lc
                   + np.einsum('ij,ij->i', A, B) * lc
                   + np.einsum('ij,ij->i', B, C) * la
                   + np.einsum('ij,ij->i', C, A) * lb)
            tot += 2.0 * np.arctan2(num, den)
    return tot / (4.0 * np.pi)


def cells_by_winding(engine, polys):
    """(cellset, density): cells of the engine's arrangement lying inside
    the closed surface (|winding| >= 1/2 at the cell centroid), plus the
    surface's density = |winding at the origin|."""
    cent = np.array([engine.cells[s]['centroid'] for s in engine.cells])
    w = winding_numbers(cent, polys)
    wr = np.round(np.abs(w)).astype(int)
    if np.abs(np.abs(w) - wr).max() > 1e-6:
        raise RuntimeError('ambiguous winding number at a cell centroid')
    keys = list(engine.cells.keys())
    sel = {keys[i] for i in range(len(keys)) if wr[i] > 0}
    density = int(round(abs(float(winding_numbers(np.zeros(3), polys)[0]))))
    return sel, density


def five_cubes_cells(engine):
    """Cells of the compound of five cubes inscribed in the RT (cube faces
    lie in the RT's 30 face planes).  Returns (cellset, cube_plane_lists)."""
    cubes = orthogonal_plane_cubes(engine.N, engine.D)
    if len(cubes) != 5:
        raise RuntimeError('expected 5 inscribed cubes, found %d'
                           % len(cubes))
    sel = set()
    for c in cubes:
        sel |= engine.cells_inside_convex(engine.N[c], engine.D[c])
    return sel, cubes


# --------------------------------------------------------------------------
# named presets (verified by _selftest())
# --------------------------------------------------------------------------
# code entries: shell labels, 'all', or ('hand', shell_label, hand_index)
# for one hand of a chiral shell.  Shell labels are the engine's generic
# ones (deterministic: sorted by power, then mean radius).
SEEDS = ('icosahedron', 'dodecahedron', 'cuboctahedron',
         'rhombic_triacontahedron')

NAMED_PRESETS = {
    'icosahedron': [
        ('core', 'Icosahedron', ['a'], 'Platonic seed; Crennell 1'),
        ('first', 'First stellation (small triambic icosahedron)',
         ['a', 's01'], 'Crennell 2'),
        ('five_octahedra', 'Compound of five octahedra',
         ['a', 's01', 's02'], 'Du Val C; Crennell 3'),
        ('ten_tetrahedra', 'Compound of ten tetrahedra',
         ['a', 's01', 's02', 's03', 's04', 's05', 's07'],
         'Du Val Ef1; Crennell 22'),
        ('five_tetrahedra', 'Compound of five tetrahedra (chiral)',
         ['a', 's01', 's02', 's03', 's04', 's05', ('hand', 's07', 0)],
         'Du Val Ef1 with one hand of f1; Crennell 47'),
        ('great', 'Great icosahedron',
         ['a', 's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08',
          's09'], 'Du Val G; Crennell 7'),
        ('final', 'Final stellation (echidnahedron)', ['all'],
         'Du Val H; Crennell 8'),
    ],
    'dodecahedron': [
        ('core', 'Dodecahedron', ['a'], 'Platonic seed'),
        ('small_stellated', 'Small stellated dodecahedron {5/2,5}',
         ['a', 's01'], 'Kepler; Wenninger 20'),
        ('great_dodecahedron', 'Great dodecahedron {5,5/2}',
         ['a', 's01', 's02'], 'Poinsot; Wenninger 21'),
        ('great_stellated', 'Great stellated dodecahedron {5/2,3} (final)',
         ['all'], 'Kepler; Wenninger 22'),
    ],
    'cuboctahedron': [
        ('core', 'Cuboctahedron', ['a'], 'Archimedean seed'),
        ('octahedron', 'Octahedron (square-face pyramids)', ['a', 's01'],
         'first-shell stellation over the 6 square faces'),
        ('cube', 'Cube (triangle-face pyramids)', ['a', 's02'],
         'first-shell stellation over the 8 triangle faces'),
        ('cube_octahedron', 'Compound of cube and octahedron',
         ['a', 's01', 's02'], 'Wenninger 43'),
        ('final', 'Final stellation of the cuboctahedron', ['all'], ''),
    ],
    'rhombic_triacontahedron': [
        ('core', 'Rhombic triacontahedron', ['a'],
         'Catalan seed (dual icosidodecahedron)'),
        ('five_cubes', 'Compound of five cubes',
         ['a', 's01', 's02', 's03', 's04', 's05', 's06'],
         'classical RT stellation (Cundy & Rollett)'),
        ('medial_rt', 'Medial rhombic triacontahedron',
         ['a', 's01', 's02', 's04', 's05', 's07'],
         'dual dodecadodecahedron; density 3'),
        ('great_rt', 'Great rhombic triacontahedron',
         ['a', 's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08',
          's09', 's10', 's11', 's13', 's14', 's15', 's18'],
         'dual great icosidodecahedron; density 7'),
        ('final', 'Final stellation of the RT', ['all'], ''),
    ],
}


def named_presets(seed_name):
    """Verified named stellations of a built-in seed:
    [(key, title, code, note), ...]."""
    return list(NAMED_PRESETS[seed_name.lower()])


def expand_code(engine, code):
    """Resolve preset code entries to what engine.build accepts (turn
    ('hand', label, k) into that hand's frozenset of cells)."""
    out = []
    for item in code:
        if (isinstance(item, tuple) and len(item) == 3
                and item[0] == 'hand'):
            out.append(engine.shell_by_label[item[1]]['hands'][item[2]])
        else:
            out.append(item)
    return out


def build_named(seed_name, key, scale=True):
    """Build a named preset -> (V, F)."""
    eng = stellations_of(seed_name)
    for k, title, code, note in named_presets(seed_name):
        if k == key:
            return eng.build(expand_code(eng, code), scale=scale)
    raise KeyError('no preset %r for %s' % (key, seed_name))


# --------------------------------------------------------------------------
# verification transcript
# --------------------------------------------------------------------------
def _radius_classes(V, ndig=4):
    return sorted(Counter(round(math.sqrt(x * x + y * y + z * z), ndig)
                          for (x, y, z) in V).items())


def _outer_dirs_match(V, dirs, tol=1e-5):
    """Count of outermost vertices of V whose direction matches one of the
    given unit directions."""
    A = np.asarray(V, float)
    r = np.linalg.norm(A, axis=1)
    outer = A[r > r.max() - 1e-6]
    dn = outer / np.linalg.norm(outer, axis=1)[:, None]
    return len(outer), sum(1 for d in dn
                           if np.min(((dirs - d) ** 2).sum(1)) < tol * tol)


class _Checker(object):
    def __init__(self):
        self.fails = []

    def __call__(self, ok, msg):
        print('  [%s] %s' % ('PASS' if ok else 'FAIL', msg))
        if not ok:
            self.fails.append(msg)


def _build_stats(eng, code):
    V, F = eng.build(expand_code(eng, code), scale=False)
    return V, F, surface_stats(V, F)


def _seed_header(eng):
    print('=' * 74)
    print('SEED: %s' % eng.name)
    print('=' * 74)
    print('face planes: %d   bounded cells: %d   shells (orbits): %d'
          % (eng.m, len(eng.cells), len(eng.shells)))
    print('symmetry: %d rotations + %d improper (full group order %d)'
          % (len(eng.rotations), len(eng.impropers),
             len(eng.rotations) + len(eng.impropers)))
    print(eng.orbit_table())
    print()


def _core_anchor(eng, ck, expect_vef):
    V, F, st = _build_stats(eng, ['a'])
    ok = (st['V'], st['E'], st['F']) == expect_vef and st['chi'] == 2 \
        and st['closed_2'] and vertex_sets_match(V, eng.seedV) \
        and convex_hull_vertex_count(V) == expect_vef[0]
    ck(ok, 'core cell rebuilds the seed exactly: V%d E%d F%d chi=%d '
       'closed=%s vertex-set match=%s'
       % (st['V'], st['E'], st['F'], st['chi'], st['closed_2'],
          vertex_sets_match(V, eng.seedV)))


def _preset_sweep(eng, seed_name, ck):
    print('  named presets:')
    for key, title, code, note in named_presets(seed_name):
        V, F, st = _build_stats(eng, code)
        ok = st['closed_2'] and st['chi'] == 2 and not st['nan'] \
            and signed_volume(V, F) > 0
        ck(ok, '%-18s %-44s V%-4d E%-4d F%-4d chi=%d closed=%s vol=%.4f'
           % (key, title[:44], st['V'], st['E'], st['F'], st['chi'],
              st['closed_2'], signed_volume(V, F)))


def _verify_icosahedron(ck):
    eng = stellations_of('icosahedron')
    _seed_header(eng)
    ck(len(eng.cells) == 473, '473 bounded cells (Du Val/Maeder count): %d'
       % len(eng.cells))
    tally = {'b': 20, 'c': 30, 'd': 60, 'e1': 20, 'e2': 60, 'f1': 120,
             'f2': 12, 'g1': 30, 'g2': 60, 'g3': 60}
    got = {eng.alias(sh['label']): sh['size'] for sh in eng.shells
           if sh['label'] != 'a'}
    ck(got == tally, 'shell tally matches Du Val a..g3: %s' % (got == tally))
    chir = [eng.alias(sh['label']) for sh in eng.shells if sh['chiral']]
    ck(chir == ['f1'], 'f1 is the unique chiral shell (2 hands of 60): %s'
       % chir)
    ck(len(eng.rotations) == 60 and len(eng.impropers) == 60,
       'icosahedral symmetry group 60+60')
    _core_anchor(eng, ck, (12, 30, 20))

    V, F, st = _build_stats(eng, ['a', 's01'])
    ck((st['V'], st['E'], st['F']) == (32, 90, 60) and st['closed_2'],
       'first stellation: V32 E90 F60 (triakis-like hexecontahedron)')

    V, F, st = _build_stats(eng, ['a', 's01', 's02'])
    ck(convex_hull_vertex_count(V) == 30 and st['closed_2'],
       'compound of five octahedra: 30 hull vertices -> %d'
       % convex_hull_vertex_count(V))

    code10 = next(c for k, t, c, n in NAMED_PRESETS['icosahedron']
                  if k == 'ten_tetrahedra')
    n10 = len(eng.cells_of_code(expand_code(eng, code10)))
    V, F, st = _build_stats(eng, code10)
    ck(convex_hull_vertex_count(V) == 20 and n10 == 311,
       'compound of ten tetrahedra: 20 hull vertices, 311 cells -> %d, %d'
       % (convex_hull_vertex_count(V), n10))

    code5 = next(c for k, t, c, n in NAMED_PRESETS['icosahedron']
                 if k == 'five_tetrahedra')
    n5 = len(eng.cells_of_code(expand_code(eng, code5)))
    V, F, st = _build_stats(eng, code5)
    ck(convex_hull_vertex_count(V) == 20 and n10 - n5 == 60,
       'compound of five tetrahedra (one hand of f1): 20 hull vertices, '
       '60 fewer cells -> %d, %d' % (convex_hull_vertex_count(V), n5))

    codeG = next(c for k, t, c, n in NAMED_PRESETS['icosahedron']
                 if k == 'great')
    V, F, st = _build_stats(eng, codeG)
    sv = eng.seedV / np.linalg.norm(eng.seedV, axis=1)[:, None]
    no, nm = _outer_dirs_match(V, sv)
    ck(convex_hull_vertex_count(V) == 12 and no == 12 and nm == 12,
       'great icosahedron: 12 outer vertices on the seed 5-fold axes '
       '-> hull=%d outer=%d matched=%d' % (convex_hull_vertex_count(V),
                                           no, nm))

    V, F, st = _build_stats(eng, ['all'])
    rc = _radius_classes(V, 3)
    counts = [c for _, c in rc]
    ratios = [r / rc[0][0] for r, _ in rc]
    ck((st['V'], st['E'], st['F']) == (92, 270, 180) and st['closed_2']
       and counts == [20, 12, 60]
       and abs(ratios[1] - 1.7769) < 2e-3 and abs(ratios[2] - 3.5066) < 2e-3
       and convex_hull_vertex_count(V) == 60,
       'echidnahedron: V92 E270 F180, vertex shells 20/12/60 at '
       '1:%.4f:%.4f, 60 hull tips' % (ratios[1], ratios[2]))
    _preset_sweep(eng, 'icosahedron', ck)
    print()


def _verify_dodecahedron(ck):
    eng = stellations_of('dodecahedron')
    _seed_header(eng)
    sizes = [sh['size'] for sh in eng.shells]
    ck(len(eng.cells) == 63 and sizes == [1, 12, 30, 20],
       '63 bounded cells in shells 1/12/30/20: %s' % sizes)
    ck(len(eng.rotations) == 60 and len(eng.impropers) == 60,
       'icosahedral symmetry group 60+60')
    _core_anchor(eng, ck, (20, 30, 12))
    Vc, Fc, stc = _build_stats(eng, ['a'])
    vol_core = signed_volume(Vc, Fc)

    fn = eng.N                                   # face normals = 5-fold axes
    V, F, st = _build_stats(eng, ['a', 's01'])
    no, nm = _outer_dirs_match(V, fn)
    vol_ssd = signed_volume(V, F)
    ck((st['V'], st['E'], st['F']) == (32, 90, 60) and st['closed_2']
       and convex_hull_vertex_count(V) == 12 and no == 12 and nm == 12
       and abs(vol_ssd / vol_core - 5.0 ** 0.5) < 1e-6,
       'small stellated dodecahedron: V32 E90 F60, 12 spikes on the face '
       'axes, vol = sqrt(5) x core (%.6f)' % (vol_ssd / vol_core))

    V, F, st = _build_stats(eng, ['a', 's01', 's02'])
    no, nm = _outer_dirs_match(V, fn)
    vol_gd = signed_volume(V, F)
    ck((st['V'], st['E'], st['F']) == (32, 90, 60) and st['closed_2']
       and convex_hull_vertex_count(V) == 12 and no == 12 and nm == 12
       and abs(vol_gd / vol_ssd - PHI) < 1e-6,
       'great dodecahedron: V32 E90 F60, 12 icosahedral vertices '
       '(same arrangement as SSD tips), vol = phi x SSD (%.6f)'
       % (vol_gd / vol_ssd))

    sv = eng.seedV / np.linalg.norm(eng.seedV, axis=1)[:, None]
    V, F, st = _build_stats(eng, ['all'])
    no, nm = _outer_dirs_match(V, sv)
    ck((st['V'], st['E'], st['F']) == (32, 90, 60) and st['closed_2']
       and convex_hull_vertex_count(V) == 20 and no == 20 and nm == 20,
       'great stellated dodecahedron (final): V32 E90 F60, 20 spikes on '
       'the seed 3-fold vertex axes')
    _preset_sweep(eng, 'dodecahedron', ck)
    print()


def _verify_cuboctahedron(ck):
    eng = stellations_of('cuboctahedron')
    _seed_header(eng)
    ck(len(eng.cells) == 119 and len(eng.shells) == 8,
       '119 bounded cells in 8 orbits: %d / %d'
       % (len(eng.cells), len(eng.shells)))
    ck(len(eng.rotations) == 24 and len(eng.impropers) == 24,
       'octahedral symmetry group 24+24')
    _core_anchor(eng, ck, (12, 24, 14))

    V, F, st = _build_stats(eng, ['a', 's01'])
    vol = signed_volume(V, F)
    ck(convex_hull_vertex_count(V) == 6 and st['closed_2']
       and abs(vol - 32.0 / 3.0) < 1e-6,
       'octahedron (pyramids on the 6 squares): 6 hull vertices, '
       'vol = 32/3 (%.6f)' % vol)

    V, F, st = _build_stats(eng, ['a', 's02'])
    vol = signed_volume(V, F)
    cube8 = np.array([v for v in V
                      if abs(np.linalg.norm(v) - 3.0 ** 0.5) < 1e-6])
    ck(convex_hull_vertex_count(V) == 8 and st['closed_2']
       and abs(vol - 8.0) < 1e-6 and len(cube8) == 8
       and vertex_sets_match(cube8, np.array(
           [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1)
            for sz in (-1, 1)], float)),
       'cube (pyramids on the 8 triangles): 8 hull vertices at (+-1,+-1,'
       '+-1), vol = 8 (%.6f)' % vol)

    V, F, st = _build_stats(eng, ['a', 's01', 's02'])
    vol = signed_volume(V, F)
    ck(convex_hull_vertex_count(V) == 14 and st['closed_2']
       and (st['V'], st['E'], st['F']) == (26, 72, 48)
       and abs(vol - 12.0) < 1e-6,
       'compound of cube and octahedron (Wenninger 43): V26 E72 F48, '
       '14 hull vertices, vol = 12 exactly (%.6f)' % vol)

    V, F, st = _build_stats(eng, ['all'])
    vol = signed_volume(V, F)
    ck(st['closed_2'] and st['chi'] == 2 and abs(vol - 64.0) < 1e-6,
       'final stellation: V%d E%d F%d closed, vol = 64 exactly (%.6f)'
       % (st['V'], st['E'], st['F'], vol))
    _preset_sweep(eng, 'cuboctahedron', ck)
    print()


def _verify_rt(ck):
    eng = stellations_of('rhombic_triacontahedron')
    _seed_header(eng)
    ck(len(eng.cells) == 2003, 'bounded cells: %d' % len(eng.cells))
    ck(len(eng.rotations) == 60 and len(eng.impropers) == 60,
       'icosahedral symmetry group 60+60')
    _core_anchor(eng, ck, (32, 60, 30))

    # --- compound of five cubes ---------------------------------------
    sel, cubes = five_cubes_cells(eng)
    labs = eng.labels_of_cells(sel)
    want = next(c for k, t, c, n in NAMED_PRESETS['rhombic_triacontahedron']
                if k == 'five_cubes')
    vol1 = sum(eng.cell_volume(s) for s in eng.cells_inside_convex(
        eng.N[cubes[0]], eng.D[cubes[0]]))
    V, F, st = _build_stats(eng, want)
    sv = eng.seedV / np.linalg.norm(eng.seedV, axis=1)[:, None]
    no, nm = _outer_dirs_match(V, sv)
    ck(len(cubes) == 5 and abs(vol1 - 8.0) < 1e-6 and labs == want
       and st['closed_2'] and convex_hull_vertex_count(V) == 20
       and no == 20 and nm == 20,
       'compound of five cubes: 5 orthogonal plane-sextets, cube volume '
       '8 exactly, orbit union %s, 20 hull vertices on the RT 3-fold '
       'vertex axes' % labs)

    # --- medial / great RT via independent dual construction ----------
    for kind, key, dens_want, hull_want, vef in (
            ('medial', 'medial_rt', 3, 12, (182, 480, 300)),
            ('great', 'great_rt', 7, 20, (422, 1020, 600))):
        quads = rt_star_dual(kind)
        sel, dens = cells_by_winding(eng, quads)
        labs = eng.labels_of_cells(sel)
        want = next(c for k, t, c, n in
                    NAMED_PRESETS['rhombic_triacontahedron'] if k == key)
        V, F, st = _build_stats(eng, want)
        hull = convex_hull_vertex_count(V)
        # spike tips of the built solid = outermost dual vertices
        dv = np.array(sorted({tuple(np.round(p, 6))
                              for q in quads for p in q}))
        rd = np.linalg.norm(dv, axis=1)
        tips = dv[rd > rd.max() - 1e-6]
        A = np.array(V)
        rv = np.linalg.norm(A, axis=1)
        built_tips = A[rv > rv.max() - 1e-6]
        tips_ok = (abs(rv.max() - rd.max()) < 1e-5
                   and len(built_tips) == len(tips)
                   and vertex_sets_match(built_tips, tips, tol=1e-5))
        ck(dens == dens_want and labs == want and st['closed_2']
           and st['chi'] == 2 and hull == hull_want
           and (st['V'], st['E'], st['F']) == vef and tips_ok,
           '%s RT (dual, density %d): winding cells = orbit union %s;'
           % (kind, dens, labs))
        print('         built V%d E%d F%d chi=%d closed=%s hull=%d, '
              '%d spike tips coincide with the dual\'s outer vertices'
              % (st['V'], st['E'], st['F'], st['chi'], st['closed_2'],
                 hull, len(tips)))

    V, F, st = _build_stats(eng, ['all'])
    ck(st['closed_2'] and st['chi'] == 2,
       'final stellation of the RT: V%d E%d F%d closed'
       % (st['V'], st['E'], st['F']))
    _preset_sweep(eng, 'rhombic_triacontahedron', ck)
    print()


def _self_test():
    print('GENERAL STELLATION ENGINE -- verification transcript')
    print()
    ck = _Checker()
    _verify_icosahedron(ck)
    _verify_dodecahedron(ck)
    _verify_cuboctahedron(ck)
    _verify_rt(ck)
    print('=' * 74)
    if ck.fails:
        print('RESULT: FAIL (%d)' % len(ck.fails))
        for m in ck.fails:
            print('  - ' + m)
    else:
        print('RESULT: OK')
    print('=' * 74)
    return not ck.fails


# --------------------------------------------------------------------------
# Blender operator -- Seed + Stellation dropdowns.  The named-preset enum is
# static (no arrangement enumeration to populate the UI); the ~1s cell
# enumeration for a seed runs lazily on the first build and is cached.
# --------------------------------------------------------------------------
try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _SEED_ITEMS = [(s, s.replace('_', ' ').title(), "") for s in SEEDS]
    _PRESET_CACHE = {}

    def _stell_items(self, context):
        seed = self.seed or SEEDS[0]
        if seed not in _PRESET_CACHE:
            _PRESET_CACHE[seed] = [(k, title, note)
                                   for k, title, code, note
                                   in named_presets(seed)]
        return _PRESET_CACHE[seed]

    def _seed_update(self, context):
        ids = [it[0] for it in _stell_items(self, context)]
        if ids and self.stellation not in ids:
            self.stellation = ids[0]

    class MESH_OT_general_stellation_add(bpy.types.Operator):
        """Add a stellation of a seed polyhedron (icosahedron, dodecahedron,
        cuboctahedron, or rhombic triacontahedron) -- built from the bounded
        cells of the seed's face-plane arrangement, grouped into symmetry
        shells"""
        bl_idname = "mesh.general_stellation_add"
        bl_label = "General Stellation"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(name="Seed", items=_SEED_ITEMS,
                           update=_seed_update)
        stellation: EnumProperty(name="Stellation", items=_stell_items)
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", ""),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo modifier"),
                   ('WIRE', "Struts", "Wireframe modifier"),
                   ('BALLSTICK', "Ball and Stick",
                    "Edges as solid cylindrical struts and vertices "
                    "as small spheres (ball-and-stick model)"),
                   ('WIREFRAME', "Wireframe",
                    "Mesh edges only, displayed as a wireframe")],
            default='SOLID')
        border: FloatProperty(name="Border", default=0.3, min=0.02, max=0.95,
                              description="Leonardo face frame width")
        thickness: FloatProperty(name="Thickness", default=0.05, min=0.001,
                                 max=1.0, description="Panel/strut thickness")
        strut_radius: FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.5,
            description="Ball-and-stick edge cylinder radius")
        node_radius: FloatProperty(
            name="Node Radius", default=0.035, min=0.0, max=0.5,
            description="Ball-and-stick vertex sphere radius "
                        "(0 = no nodes)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def execute(self, context):
            seed = self.seed or SEEDS[0]
            ids = [it[0] for it in _stell_items(self, context)]
            key = self.stellation if self.stellation in ids \
                else (ids[0] if ids else 'core')
            V, F = build_named(seed, key)
            title = next((t for k, t, c, n in named_presets(seed)
                          if k == key), key)
            me = bpy.data.meshes.new(title)
            me.from_pydata([tuple(c * self.scale for c in v) for v in V],
                           [], [list(f) for f in F])
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new(title, me)
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
                leonardo_style.add_modifier(obj, self.border, self.thickness)
            elif self.style == 'WIRE':
                mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
                mod.thickness = self.thickness
                mod.use_even_offset = False
            elif self.style == 'BALLSTICK':
                try:
                    from . import ball_and_stick
                except ImportError:
                    import ball_and_stick
                ball_and_stick.rebuild(obj, self.strut_radius,
                                       self.node_radius)
            elif self.style == 'WIREFRAME':
                obj.display_type = 'WIRE'
            self.report({'INFO'}, "%s: V=%d F=%d" % (title, len(V), len(F)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'seed')
            lay.prop(self, 'stellation')
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style in ('LEONARDO', 'WIRE'):
                lay.prop(self, 'thickness')
            if self.style == 'BALLSTICK':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.general_stellation_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_general_stellation_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_general_stellation_add)
