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
'rhombic_triacontahedron', 'rhombic_dodecahedron' and
'triakis_tetrahedron' (the last three constructed as polar duals of the
icosidodecahedron, cuboctahedron and truncated tetrahedron), plus
'dodecahedron_tetrahedral' -- the dodecahedron with its symmetry
RESTRICTED to the tetrahedral subgroup, which is what makes the
tetrahedral stellations possible: under its own icosahedral group the
twelve face planes form a single orbit and no merely-tetrahedral
stellation can be selected.  Custom convex seeds: pass a vertex array
(origin must be interior; at most 60 face planes).

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

import itertools
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


def _rhombic_dodecahedron_V():
    """RD as the polar dual of the cuboctahedron -- the same trick the RT
    uses, one symmetry down."""
    W = _cuboctahedron_V()
    N, D = hull_planes(W)
    return N / D[:, None]


def _triakis_tetrahedron_V():
    """The simplest Archimedean dual: the polar dual of the truncated
    tetrahedron."""
    a = 1.0 / 3.0
    W = []
    for sx, sy, sz in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        for perm in ((a, sy * 1.0, sz * 1.0), (sx * 1.0, a, sz * 1.0),
                     (sx * 1.0, sy * 1.0, a)):
            W.append([perm[0] * sx if abs(perm[0]) == a else perm[0],
                      perm[1], perm[2]])
    # truncated tetrahedron: all permutations of (+-1, +-1, +-3) with an
    # even number of minus signs, scaled
    W = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                if sx * sy * sz < 0:
                    continue
                for perm in range(3):
                    v = [sx * 1.0, sy * 1.0, sz * 3.0]
                    v = v[-perm:] + v[:-perm] if perm else v
                    W.append([v[0] * (1 if perm == 0 else 1), v[1], v[2]])
    W = np.asarray(W, float)
    N, D = hull_planes(W)
    return N / D[:, None]


_SEED_BUILDERS = {
    'icosahedron': _icosahedron_V,
    'dodecahedron': _dodecahedron_V,
    'cuboctahedron': _cuboctahedron_V,
    'rhombic_triacontahedron': _rhombic_triacontahedron_V,
    'rhombic_dodecahedron': _rhombic_dodecahedron_V,
    'triakis_tetrahedron': _triakis_tetrahedron_V,
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

    def __init__(self, V, name='custom', big=_BIG, verbose=False,
                 subgroup=None):
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
        if subgroup:
            self._restrict(subgroup)
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

    def _restrict(self, subgroup):
        """Keep only the symmetries in a named SUBGROUP of the seed's own
        group, so the cells fall into finer orbits.

        The tetrahedral stellations of the dodecahedron need exactly this:
        `_symmetry` always finds the FULL group from the vertex set, which
        for the dodecahedron is icosahedral, and under it the twelve face
        planes form one orbit -- there is then no way to make a stellation
        that is merely tetrahedral.  Restricting to the tetrahedral
        subgroup splits that orbit and the finer stellations appear.

        A rotation belongs to the tetrahedral subgroup when it maps the
        chosen tetrahedron's four vertex directions onto themselves.
        """
        if subgroup not in ('tetrahedral', 'chiral_tetrahedral'):
            raise ValueError('unknown subgroup %r' % (subgroup,))
        T = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        T = T / np.linalg.norm(T[0])

        def keeps(R):
            RT = T @ R.T
            d2 = ((RT[:, None, :] - T[None, :, :]) ** 2).sum(-1)
            return bool(d2.min(axis=1).max() < 1e-9)

        self.rotations = [R for R in self.rotations if keeps(R)]
        self.impropers = ([] if subgroup == 'chiral_tetrahedral'
                          else [R for R in self.impropers if keeps(R)])
        if not self.rotations:
            raise ValueError('the subgroup is trivial for this seed')

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
            # Sorted by their smallest sign-vector, so which enantiomorph
            # is hand 0 is a property of the arrangement and not of the
            # order a dict happened to iterate in.  Codes naming a hand
            # ('f1a') must mean the same solid on every run.
            sh['hands'] = sorted((o for o in rot_orbs if o & sh['cells']),
                                 key=min)
            sh['chiral'] = len(sh['hands']) == 2
            sh['label'] = 'a' if (sh['power'] == 0 and sh['size'] == 1) \
                else 's%02d' % k
        self.shells = shells
        self.rot_orbits = rot_orbs
        self.labels = [sh['label'] for sh in shells]
        self.shell_by_label = {sh['label']: sh for sh in shells}

        # Classical names, where the seed has any.  The generic labels stay
        # canonical -- stored codes and presets are written in them -- and
        # the classical ones are an accepted INPUT spelling plus what the UI
        # displays.  Keyed on (power, size) rather than on position, because
        # the two orderings disagree: Du Val's e1 is the 20-cell shell at
        # power 4 and e2 the 60-cell one, while this engine sorts that power
        # by radius and gets them the other way round.  Same for f1/f2.
        self.classical = {}
        if self.name == 'icosahedron':
            seen = {}
            for sh in shells:
                key = (sh['power'], sh['size'])
                if key in seen:
                    raise RuntimeError(
                        'cannot apply classical labels to %s: shells %r and '
                        '%r share (power, size) = %r, so the naming would be '
                        'arbitrary' % (self.name, seen[key], sh['label'], key))
                seen[key] = sh['label']
                cl = _DUVAL_BY_POWER_SIZE.get(key)
                if cl is not None:
                    sh['classical'] = cl
                    self.classical[cl] = sh['label']
            self.classical['a'] = 'a'
        for sh in shells:
            sh.setdefault('classical', None)

    # ---- support ---------------------------------------------------------
    def support(self):
        """Which cells each cell RESTS ON, as {cell: frozenset(cells)}.

        Pawley's rule for a non-reentrant ("fully supported") stellation
        is that "for a volume to be included in a stellation all those
        volumes on the lower courses on which it rests must also be
        included", and he draws the relation as a stack of bricks.  It
        does not have to be read off that drawing: it is geometry.  Two
        cells share a facet exactly when their sign vectors differ in ONE
        plane, and the one on the inner side of that plane is the lower.
        `power` counts the planes a cell lies outside of, so the lower
        cell is simply the one whose power is smaller.

        Reading it from the diagram is in fact unreliable -- Pawley draws
        bricks A, C, J and N BROKEN so that a three-dimensional stack
        fits on the page, so horizontal overlap there is not the resting
        relation.  Computing it sidesteps that entirely.
        """
        if getattr(self, '_support', None) is not None:
            return self._support
        keys = list(self.cells.keys())
        idx = {s: i for i, s in enumerate(keys)}
        A = np.array(keys, np.int8)
        pw = np.array([self.cells[s]['power'] for s in keys])
        out = {}
        for i, s in enumerate(keys):
            diff = (A != A[i]).sum(axis=1)
            nb = np.flatnonzero((diff == 1) & (pw < pw[i]))
            out[s] = frozenset(keys[j] for j in nb)
        self._support = out
        return out

    def is_fully_supported(self, cellset):
        """True when every cell present also has everything it rests on."""
        sup = self.support()
        cs = set(cellset)
        return all(sup[s] <= cs for s in cs)

    def support_closure(self, cellset):
        """Add whatever is needed to make a cell set fully supported."""
        sup = self.support()
        out = set(cellset)
        frontier = list(out)
        while frontier:
            s = frontier.pop()
            for t in sup[s]:
                if t not in out:
                    out.add(t)
                    frontier.append(t)
        return frozenset(out)

    def shell_support(self):
        """The same relation lifted to shells: {label: set(labels)}."""
        sup = self.support()
        owner = {}
        for sh in self.shells:
            for s in sh['cells']:
                owner[s] = sh['label']
        out = {sh['label']: set() for sh in self.shells}
        for s, below in sup.items():
            for t in below:
                if owner[t] != owner[s]:
                    out[owner[s]].add(owner[t])
        return out

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
                    filled |= self.cells_of_token(item)
            elif _is_cell(item):
                filled.add(item)
            else:
                filled |= set(item)
        return filled

    def cells_of_token(self, token):
        """Cells named by one code token.

        Accepted, in this order: a canonical shell label ('s07'); a
        classical label where the seed has one ('f1'); either of those with
        a trailing 'a'/'b' naming ONE HAND of a chiral shell ('f1a').

        Exact labels are tried before the hand suffix is peeled, so 'b' is
        Du Val's shell b and never hand-b of a shell called ''.
        """
        sh = self.shell_by_label.get(token)
        if sh is None and token in self.classical:
            sh = self.shell_by_label[self.classical[token]]
        if sh is not None:
            return set(sh['cells'])

        if len(token) > 1 and token[-1] in 'ab':
            base, hand = token[:-1], 'ab'.index(token[-1])
            sh = self.shell_by_label.get(base)
            if sh is None and base in self.classical:
                sh = self.shell_by_label[self.classical[base]]
            if sh is not None:
                if not sh['chiral']:
                    raise KeyError(
                        'shell %r is not chiral, so %r names nothing; it has '
                        'a single hand' % (base, token))
                return set(sh['hands'][hand])

        # Name each shell once, by its classical name where it has one --
        # listing 'a' twice because it is both canonical and classical helps
        # nobody find their typo.  Classical names are listed in their own
        # order, not this engine's: it sorts each power by radius and so
        # would print 'e2 e1 f2 f1', which reads as a mistake to anyone who
        # knows the notation.
        if self.classical:
            known = [lb for lb in DUVAL_LABELS if lb in self.classical]
        else:
            known = [sh['classical'] or sh['label'] for sh in self.shells]
        raise KeyError('unknown shell %r for seed %r; known: %s'
                       % (token, self.name, ' '.join(known)))

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

DUVAL_LABELS = ['a', 'b', 'c', 'd', 'e1', 'e2', 'f1', 'f2', 'g1', 'g2', 'g3']

# Du Val's capitals abbreviate a run of shells outward from the core.
_DUVAL_CAP = {
    'A': ['a'],
    'B': ['a', 'b'],
    'C': ['a', 'b', 'c'],
    'D': ['a', 'b', 'c', 'd'],
    'E': ['a', 'b', 'c', 'd', 'e1', 'e2'],
    'F': ['a', 'b', 'c', 'd', 'e1', 'e2', 'f1', 'f2'],
    # great icosahedron: 12 outer vertices at the icosahedron shell
    'G': ['a', 'b', 'c', 'd', 'e1', 'e2', 'f1', 'f2', 'g1', 'g2'],
    # final stellation / echidnahedron: every cell
    'H': ['a', 'b', 'c', 'd', 'e1', 'e2', 'f1', 'f2', 'g1', 'g2', 'g3'],
}


def expand_duval(code):
    """Expand a Du Val string like 'De1f1g1' or 'Ef1' into a shell list."""
    shells = []
    i = 0
    if code and code[0].isupper():
        shells += _DUVAL_CAP[code[0]]
        i = 1
    while i < len(code):
        ch = code[i]
        if ch.isalpha():
            tok = ch
            if i + 1 < len(code) and code[i + 1].isdigit():
                tok += code[i + 1]
                i += 1
            shells.append(tok)
        i += 1
    out = []
    for s in shells:
        if s in DUVAL_LABELS and s not in out:
            out.append(s)
    return out


# Crennell index 1..59 -> Du Val cell string (the standard published
# cross-reference).  1..32 are reflexible; 33..59 are chiral, and are the
# same cell sets with only ONE HAND of the chiral shell f1 kept.
_CRENNELL_STR = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H',
    9: 'e1', 10: 'f1', 11: 'g1', 12: 'e1f1', 13: 'e1f1g1', 14: 'f1g1',
    15: 'e2', 16: 'f2', 17: 'g2', 18: 'e2f2', 19: 'e2f2g2', 20: 'f2g2',
    21: 'De1', 22: 'Ef1', 23: 'Fg1', 24: 'De1f1', 25: 'De1f1g1',
    26: 'Ef1g1', 27: 'De2', 28: 'Ef2', 29: 'Fg2', 30: 'De2f2',
    31: 'De2f2g2', 32: 'Ef2g2',
    33: 'f1', 34: 'e1f1', 35: 'De1f1', 36: 'f1g1', 37: 'e1f1g1',
    38: 'De1f1g1', 39: 'f1g2', 40: 'e1f1g2', 41: 'De1f1g2', 42: 'f1f2g2',
    43: 'e1f1f2g2', 44: 'De1f1f2g2', 45: 'e2f1', 46: 'De2f1', 47: 'Ef1',
    48: 'e2f1g1', 49: 'De2f1g1', 50: 'Ef1g1', 51: 'e2f1f2', 52: 'De2f1f2',
    53: 'Ef1f2', 54: 'e2f1f2g1', 55: 'De2f1f2g1', 56: 'Ef1f2g1',
    57: 'e2f1f2g2', 58: 'De2f1f2g2', 59: 'Ef1f2g2',
}

CRENNELL = {k: expand_duval(v) for k, v in _CRENNELL_STR.items()}
CRENNELL_REFLEXIBLE = frozenset(range(1, 33))
CRENNELL_CHIRAL = frozenset(range(33, 60))


def crennell_code(k, hand=0):
    """Cell code for Crennell index k of the icosahedron.

    Reflexible indices (1..32) are a plain list of Du Val labels.  Chiral
    ones (33..59) keep a single hand of f1, spelled 'f1a' / 'f1b'; which
    hand is which is fixed by the canonical ordering in `_orbits`, and the
    two are mirror images, so either is a genuine member of the pair.
    """
    if k not in CRENNELL:
        raise KeyError('Crennell index %r is not in 1..59' % (k,))
    code = list(CRENNELL[k])
    if k in CRENNELL_CHIRAL and 'f1' in code:
        code = [c for c in code if c != 'f1'] + ['f1' + 'ab'[hand]]
    return code


def crennell_title(k):
    """Human label for Crennell index k: the Du Val string, plus the
    common name where the figure has one."""
    named = {1: 'Icosahedron', 2: 'First stellation (small triambic)',
             3: 'Compound of five octahedra', 4: 'Third stellation',
             6: 'Second stellation', 7: 'Great icosahedron',
             8: 'Final stellation (echidnahedron)',
             22: 'Compound of ten tetrahedra',
             26: 'Excavated dodecahedron',
             30: 'Great triambic icosahedron',
             47: 'Compound of five tetrahedra'}
    s = '%d. %s' % (k, _CRENNELL_STR[k])
    if k in named:
        s += ' -- ' + named[k]
    return s


# --------------------------------------------------------------------------
# constructor / cache
# --------------------------------------------------------------------------
_ENGINE_CACHE = {}


# Seeds that are a named seed PLUS a symmetry restriction.  The
# tetrahedral stellations of the dodecahedron are the reason the
# subgroup machinery exists: under the dodecahedron's own icosahedral
# group its twelve face planes form a single orbit, and only the
# tetrahedral subgroup splits them finely enough for these to appear.
# --------------------------------------------------------------------------
# Smith's units for the triakis tetrahedron
# --------------------------------------------------------------------------
# A. G. Smith, "Stellations of the triakis tetrahedron", Math. Gazette 49
# (1965), 135-143, divides the arrangement into nine units A..I and gives
# five rules for combining them.  His units are this engine's shells, one
# for one, matched by size and by radius order:
#
#   A=a(1)  B=s01  C=s02(12)  D=s03(6)  E=s04(4)  F=s05  G=s06(12)
#   H=s07(12)  I=s08
#
# The two CHIRAL units confirm it independently: Smith marks exactly F
# and I "left and right", and the engine's only two chiral shells are
# s05 and s08, both 24 = 2x12.
#
# One number needs care.  Smith's table totals 99 and the engine reports
# 107 cells, which looks like a discrepancy and is not: his column counts
# UNITS, the engine counts CELLS, and his unit B is 4 units of 3 cells.
# Both come to 1+12+12+6+4+24+12+12+24 = 107.
SMITH_UNITS = {'A': 'a', 'B': 's01', 'C': 's02', 'D': 's03', 'E': 's04',
               'F': 's05', 'G': 's06', 'H': 's07', 'I': 's08'}

#: Smith's combination rules, as predicates on a set of unit letters
SMITH_RULES = (
    ('C or D implies B', lambda s: not (s & {'C', 'D'}) or 'B' in s),
    ('G or H implies F', lambda s: not (s & {'G', 'H'}) or 'F' in s),
    ('E implies C', lambda s: 'E' not in s or 'C' in s),
    ('F implies D', lambda s: 'F' not in s or 'D' in s),
    ('I implies G or H', lambda s: 'I' not in s or bool(s & {'G', 'H'})),
)

#: the six Smith illustrates, "such that the external surface of each
#: solid is as completely as possible covered in the next"
SMITH_MAIN_LINE = ('A', 'AB', 'ABCD', 'ABCDEF', 'ABCDEFGH', 'ABCDEFGHI')


def smith_stellations():
    """Every unit set satisfying Smith's rules -- his 28."""
    out = []
    letters = 'BCDEFGHI'
    for r in range(len(letters) + 1):
        for c in itertools.combinations(letters, r):
            s = set(c) | {'A'}
            if all(fn(s) for _n, fn in SMITH_RULES):
                out.append(''.join(sorted(s)))
    return out


def smith_code(units):
    """Smith's unit letters -> a code this engine's build() accepts."""
    return [SMITH_UNITS[u] for u in sorted(set(units))]


# --------------------------------------------------------------------------
# Pawley's volume letters for the rhombic triacontahedron
# --------------------------------------------------------------------------
# G. S. Pawley, "The 227 Triacontahedra", Geometriae Dedicata 4 (1975),
# 221-232, labels the elementary volumes of the RT's plane arrangement
# A..Z plus the Scandinavian AE, OE, AA, and stacks them as bricks in his
# figure 2 with the innermost at the bottom.  The engine finds 29 shells
# for the same seed and Pawley uses 29 letters, so the two decompositions
# should correspond -- and they do, anchored by a fact that cannot be
# coincidence:
#
#   Pawley names NINE volumes that do not span a plane of symmetry --
#   A, C, E, G, J, L, M, S, X -- and the engine finds exactly NINE
#   chiral shells.  Ordering his stack outward and the engine's shells by
#   radius, ALL NINE chiral positions land on each other.
#
# Reading his stack from the centre out (his AA is the core):
#
#   AA | OE | AE | Y Z | W X | T U V | R S | Q O P | N L M | K J H I |
#   G E F | C D B | A
#
# against the engine's a, s01 .. s28.  Within a row the chiral flag fixes
# the assignment where it can: s06=X, s11=S, s20=J, s27=C, s28=A are
# forced outright, and {s15,s16}={L,M}, {s22,s24}={E,G} are fixed as
# pairs with their non-chiral row-mates s17=N, s23=F.
#
# This is what makes Pawley's noble pair reachable.  He states that
# "Suw and A(bcdek) are the two isohedral-isogonal polyhedra", i.e. the
# two NOBLE stellations of the RT -- Hart's K and 2B.
PAWLEY_ROWS = [
    ('a', ['AA']),
    ('s01', ['OE']),
    ('s02', ['AE']),
    ('s03 s04', ['Y', 'Z']),
    ('s05 s06', ['W', 'X']),
    ('s07 s08 s09', ['T', 'U', 'V']),
    ('s10 s11', ['R', 'S']),
    ('s12 s13 s14', ['Q', 'O', 'P']),
    ('s15 s16 s17', ['L', 'M', 'N']),
    ('s18 s19 s20 s21', ['K', 'H', 'I', 'J']),
    ('s22 s23 s24', ['E', 'F', 'G']),
    ('s25 s26 s27', ['D', 'B', 'C']),
    ('s28', ['A']),
]

#: the nine volumes Pawley says do not span a mirror plane
PAWLEY_CHIRAL = ('A', 'C', 'E', 'G', 'J', 'L', 'M', 'S', 'X')

#: the shells those nine correspond to, forced by the chiral flags
PAWLEY_CHIRAL_SHELLS = ('s06', 's11', 's15', 's16', 's20', 's22', 's24',
                        's27', 's28')


_SEED_SUBGROUPS = {
    'dodecahedron_tetrahedral': ('dodecahedron', 'tetrahedral'),
}


def stellations_of(seed, verbose=False, subgroup=None):
    """Return a StellationEngine for a convex seed.

    seed: a name from the built-in library ('icosahedron', 'dodecahedron',
    'cuboctahedron', 'rhombic_triacontahedron'), a (V, F) pair, or a bare
    vertex array (faces are recomputed from the hull either way)."""
    if isinstance(seed, str) and seed.lower() in _SEED_SUBGROUPS:
        seed, subgroup = _SEED_SUBGROUPS[seed.lower()]
    if isinstance(seed, str):
        key = seed.lower() if not subgroup else '%s@%s' % (seed.lower(),
                                                           subgroup)
        if key not in _ENGINE_CACHE:
            if seed.lower() not in _SEED_BUILDERS:
                raise KeyError('unknown seed %r (have %s)'
                               % (seed, sorted(_SEED_BUILDERS)))
            _ENGINE_CACHE[key] = StellationEngine(
                _SEED_BUILDERS[seed.lower()](), name=key, verbose=verbose,
                subgroup=subgroup)
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
#
# Two of the icosahedron's stellations are NOBLE -- isohedral and
# isogonal at once, the property that singles out the nine regulars --
# and Coxeter, Du Val, Flather & Petrie's letters name them directly:
# D = a+b+c+d and H = every cell.  Because the engine's own orbit table
# prints its generic labels against Du Val's (s01=b, s02=c, s03=d, ...),
# those two need no shell hunting; they are read off it.
#
# The other two noble stellations Hart lists are of the rhombic
# triacontahedron and are named K and 2B in MESSER's notation, which is a
# different enumeration from the shells this engine finds.  Without that
# paper there is no way to say which cells they are, so they are absent
# rather than guessed -- see BACKLOG.md.
SEEDS = ('icosahedron', 'dodecahedron', 'dodecahedron_tetrahedral',
         'cuboctahedron', 'rhombic_dodecahedron', 'triakis_tetrahedron',
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
        # Both noble stellations have 60 true vertices, each on three of
        # the twenty face planes, so each dual is a 60-TRIANGLE faceting
        # of the dodecahedron -- and the dodecahedral vertex set carries
        # exactly two of those, one per stellation.  They are built by
        # `mesh.noble_faceting_add`, and `noble_faceting_generator.
        # noble_dual_index` re-derives which is which rather than
        # trusting an enumeration order.  Note the dual cannot be had by
        # reciprocating what `build` emits: that is the VISIBLE
        # BOUNDARY, whose corners include every intersection point of
        # the surface and not just the polyhedron's own vertices.
        ('duval_d', 'Du Val D (noble stellation)',
         ['a', 's01', 's02', 's03'],
         'noble -- isohedral AND isogonal; Du Val D = a+b+c+d; '
         'dual is Dodecahedron faceting 2'),
        ('great', 'Great icosahedron',
         ['a', 's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08',
          's09'], 'Du Val G; Crennell 7'),
        ('final', 'Final stellation (echidnahedron)', ['all'],
         'Du Val H; Crennell 8; the other noble icosahedral stellation; '
         'dual is Dodecahedron faceting 5'),
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
    'rhombic_dodecahedron': [
        ('core', 'Rhombic dodecahedron', ['a'], 'Catalan seed'),
        ('first', "First stellation (Escher's solid)", ['a', 's01'],
         "the stellated rhombic dodecahedron of Escher's Waterfall"),
        ('second', 'Second stellation', ['a', 's01', 's02'], ''),
        ('final', 'Third (final) stellation', ['all'], ''),
    ],
    'triakis_tetrahedron': [
        ('core', 'Triakis tetrahedron', ['a'],
         'the simplest Archimedean dual'),
        ('first', 'First stellation', ['a', 's01'], ''),
        ('second', 'Second stellation', ['a', 's01', 's02'], ''),
        ('third', 'Third stellation', ['a', 's01', 's02', 's03'], ''),
        ('fourth', 'Fourth stellation',
         ['a', 's01', 's02', 's03', 's04', 's05'],
         'the chiral shell s05 enters here'),
        ('fifth', 'Fifth stellation',
         ['a', 's01', 's02', 's03', 's04', 's05', 's06'], ''),
        # Smith's fifth main-line solid, ABCDEFGH -- the cumulative
        # sequence above skips it because it adds s06 and s07 together
        ('smith_gh', 'Smith ABCDEFGH (main line)',
         ['a', 's01', 's02', 's03', 's04', 's05', 's06', 's07'],
         'fifth of Smith 1965 main-line sequence'),
        ('final', 'Final stellation', ['all'], ''),
    ],
    'dodecahedron_tetrahedral': [
        ('core', 'Dodecahedron', ['a'],
         'Platonic seed, under the tetrahedral subgroup'),
        ('first', 'First tetrahedral stellation', ['a', 's01'], ''),
        # There was a 'second' here, ['a','s01','s02'].  It built a solid
        # geometrically IDENTICAL to 'first' -- s02 lies inside the hull
        # of a+s01, so adding it moves no visible face.  Only three
        # distinct closed stellations exist for this seed at shell
        # granularity (core, first, final); Hart's 39 are cut finer than
        # whole shells and need his diagrams.  _verify_presets_distinct()
        # now fails on any repeat of this.
        ('final', 'Final stellation', ['all'], ''),
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
        # Pawley 1975 names "Suw and A(bcdek)" as the two isohedral-
        # isogonal -- i.e. NOBLE -- stellations of the RT, which are the
        # forms Hart calls K and 2B.  Taking the computed support closure
        # of each (see StellationEngine.support) gives:
        #   Suw       -> S U V W X Y Z AE OE AA, ten shells
        #   A(bcdek)  -> every shell, i.e. the final stellation below
        # so only the first needs a new entry.
        # Both have 120 true vertices, each on three of the thirty face
        # planes, so each dual is a 120-TRIANGLE faceting of the
        # icosidodecahedron.  Unlike the icosahedral pair that is not
        # forced by counting -- the icosidodecahedron carries four such
        # -- so `noble_faceting_generator.noble_dual_index` matches them
        # face-set for face-set.
        ('noble_suw', 'Noble stellation Suw (Pawley)',
         ['a', 's01', 's02', 's03', 's04', 's05', 's06', 's08', 's09',
          's11'],
         'isohedral and isogonal; Pawley 1975, Hart K; dual is '
         'Icosidodecahedron faceting 6'),
        ('final', 'Final stellation of the RT', ['all'],
         'also Pawley A(bcdek), Hart 2B; dual is Icosidodecahedron '
         'faceting 9'),
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


def _verify_presets_close(ck):
    """Every named preset must be a genuine stellation: a closed surface
    with every edge in exactly two faces.

    Not every set of shells is one.  Taking the dodecahedron's size-6
    tetrahedral shell WITHOUT the shell beneath it gives chi = 8 and six
    edges with four faces round them -- cells touching along edges rather
    than a supported solid.  This check is what caught that, and it is
    the check any new preset has to pass.
    """
    print('named presets: closed surfaces')
    for seed in SEEDS:
        for key, title, code, _note in named_presets(seed):
            V, F = build_named(seed, key)
            st = surface_stats(V, F)
            ck(st['chi'] == 2 and st['closed_2'] and not st['nan'],
               '%s/%s closed (chi=%d, edges %s)'
               % (seed, key, st['chi'], sorted(st['edge_mult'])))
    print()


def _verify_presets_distinct(ck):
    """No two presets of a seed may build the same solid.

    Closing is necessary but not sufficient: a shell lying INSIDE the
    hull of the shells beneath it closes perfectly well and moves no
    visible face, so the preset is a duplicate entry in the menu that
    silently does nothing.  `dodecahedron_tetrahedral`'s 'second' was
    exactly that -- identical to 'first' -- and shipped for some time,
    because every check being run asked whether a preset was a valid
    surface and none asked whether it was a NEW one.

    The signature has to cover FACES as well as vertices, and both of the
    obvious cheaper choices give false alarms:

    * V/E/F counts alone -- `dodecahedron_tetrahedral`'s 'first' and
      'final' both have 32 vertices and 60 faces and are different
      solids;
    * vertex radii alone -- the small stellated dodecahedron and the
      great dodecahedron are built on the SAME twelve icosahedral
      vertices and differ only in how the faces run, as do the five- and
      ten-tetrahedra compounds.

    So compare the vertex radius spectrum together with the face centroid
    spectrum, which separates solids that share a vertex set.
    """
    print('named presets: pairwise distinct')
    for seed in SEEDS:
        seen = {}
        for key, _title, _code, _note in named_presets(seed):
            V, F = build_named(seed, key)

            def _rad(p):
                return round(math.sqrt(sum(c * c for c in p)), 6)
            cen = []
            for f in F:
                m = [sum(V[i][k] for i in f) / len(f) for k in range(3)]
                cen.append((len(f), _rad(m)))
            sig = (tuple(sorted(_rad(v) for v in V)), tuple(sorted(cen)))
            ck(sig not in seen,
               '%s/%s distinct from %s' % (seed, key, seen.get(sig, '-')))
            seen[sig] = key
    print()


def _verify_smith_correspondence(ck):
    """Smith's nine units are this engine's nine triakis shells.

    Two independent checks.  First the CHIRALITY: Smith marks exactly two
    units "left and right", F and I, and the engine finds exactly two
    chiral shells, which must be the ones his letters map to.  Second the
    COUNT: applying his five combination rules to subsets of his units
    must reproduce his own total of 28, and must contain the six he
    illustrates.
    """
    print('Smith 1965 correspondence (triakis tetrahedron)')
    eng = stellations_of('triakis_tetrahedron')
    by_label = {sh['label']: sh for sh in eng.shells}
    ck(len(eng.shells) == len(SMITH_UNITS),
       'nine units, nine shells (%d vs %d)'
       % (len(SMITH_UNITS), len(eng.shells)))
    chiral = {sh['label'] for sh in eng.shells if sh['chiral']}
    want = {SMITH_UNITS['F'], SMITH_UNITS['I']}
    ck(chiral == want,
       "Smith's two 'left and right' units F, I are the chiral shells "
       '%s (got %s)' % (sorted(want), sorted(chiral)))
    sizes = {u: by_label[l]['size'] for u, l in SMITH_UNITS.items()}
    ck(sizes['C'] == 12 and sizes['D'] == 6 and sizes['E'] == 4,
       "Smith's C, D, E sizes 12, 6, 4 match (got %d, %d, %d)"
       % (sizes['C'], sizes['D'], sizes['E']))
    ck(sum(sizes.values()) == 107,
       'cells total 107 both ways (got %d)' % sum(sizes.values()))
    found = smith_stellations()
    ck(len(found) == 28,
       "Smith's rules give his 28 stellations (got %d)" % len(found))
    ck(all(m in found for m in SMITH_MAIN_LINE),
       'and include all six of his main-line solids')
    print()


def _verify_support(ck):
    """The computed support relation reproduces Pawley's brick stack.

    Support is geometry, not a picture: two cells share a facet exactly
    when their sign vectors differ in one plane, and the lower is the one
    with the smaller power.  Lifting that to shells must reproduce the
    layering of Pawley's figure 2 -- and it does, row for row, ending
    with his outermost volume A resting on exactly B, C and D.

    This matters because reading the figure directly is unreliable: he
    draws bricks A, C, J and N BROKEN to fit a 3-D stack on the page, so
    horizontal overlap there is not the resting relation.
    """
    print('support relation (rhombic triacontahedron)')
    eng = stellations_of('rhombic_triacontahedron')
    ss = eng.shell_support()
    letter = {}
    for row, vols in PAWLEY_ROWS:
        for sh, v in zip(row.split(), vols):
            letter[sh] = v
    inv = {v: k for k, v in letter.items()}
    ck(ss[inv['AA']] == set(), 'the core rests on nothing')
    ck(ss[inv['OE']] == {inv['AA']}, 'OE rests on the core')
    ck(ss[inv['A']] == {inv['B'], inv['C'], inv['D']},
       "Pawley's outermost volume A rests on exactly B, C, D")
    # every shell except the core must rest on something strictly lower
    bad = [l for l in eng.labels if l != 'a' and not ss[l]]
    ck(not bad, 'every shell above the core rests on something (%s)' % bad)

    # and the two noble stellations Pawley names must close up
    for key in ('noble_suw', 'final'):
        V, F = build_named('rhombic_triacontahedron', key)
        st = surface_stats(V, F)
        ck(st['closed_2'] and st['chi'] == 2,
           'rhombic_triacontahedron/%s closes (chi=%d)' % (key, st['chi']))
    # A(bcdek) closes back to the whole cell set, which is the final
    # stellation -- so Pawley's second noble form is the one already
    # shipped, and only Suw needed adding
    cl = eng.support_closure(set().union(
        *[eng.shell_by_label[inv[v]]['cells']
          for v in ('A', 'B', 'C', 'D', 'E', 'K')]))
    ck(len(cl) == len(eng.cells),
       'A(bcdek) supports out to the final stellation (%d of %d cells)'
       % (len(cl), len(eng.cells)))
    print()


def _verify_pawley_correspondence(ck):
    """The engine's RT shells line up with Pawley's volume letters.

    The evidence is the chirality pattern.  Pawley names nine volumes
    that do not span a plane of symmetry; the engine independently finds
    nine chiral shells; and laying his brick stack against the engine's
    radius ordering puts all nine on each other.  Nine coincidences in a
    row is a correspondence, so this test pins it down and will fail if
    either side is ever renumbered.
    """
    print('Pawley 1975 correspondence (rhombic triacontahedron)')
    eng = stellations_of('rhombic_triacontahedron')
    got = tuple(sh['label'] for sh in eng.shells if sh['chiral'])
    ck(len(eng.shells) == 29,
       'engine finds 29 shells, Pawley uses 29 volume letters (got %d)'
       % len(eng.shells))
    ck(len(PAWLEY_CHIRAL) == 9, 'Pawley names nine non-mirror volumes')
    ck(got == PAWLEY_CHIRAL_SHELLS,
       'the nine chiral shells are %s' % (', '.join(PAWLEY_CHIRAL_SHELLS)))
    # the row table must name every shell exactly once
    named = [s for row, _v in PAWLEY_ROWS for s in row.split()]
    ck(len(named) == len(set(named)) == 29,
       'the row table names all 29 shells once (%d)' % len(named))
    letters = [v for _r, vs in PAWLEY_ROWS for v in vs]
    ck(len(letters) == len(set(letters)) == 29,
       'and all 29 of Pawley volume letters once (%d)' % len(letters))
    # every chiral shell must sit in a row whose letters include a
    # chiral volume, and vice versa
    ok = True
    for row, vs in PAWLEY_ROWS:
        shells = row.split()
        nchir = sum(1 for s in shells
                    if s in PAWLEY_CHIRAL_SHELLS)
        nvol = sum(1 for v in vs if v in PAWLEY_CHIRAL)
        if nchir != nvol:
            ok = False
    ck(ok, 'each row has as many chiral shells as chiral volumes')
    print()


def _self_test():
    print('GENERAL STELLATION ENGINE -- verification transcript')
    print()
    ck = _Checker()
    _verify_icosahedron(ck)
    _verify_dodecahedron(ck)
    _verify_cuboctahedron(ck)
    _verify_rt(ck)
    _verify_presets_close(ck)
    _verify_presets_distinct(ck)
    _verify_smith_correspondence(ck)
    _verify_pawley_correspondence(ck)
    _verify_support(ck)
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
# Blender operator -- the ONE stellation generator.
#
# This operator is the merge of what used to be two: "Icosahedron Stellation"
# (the fifty-nine, out of stellation_engine.py) and "General Stellation" (all
# seven seeds, out of this module).  They overlapped on the icosahedron with
# different engines and different notation, which meant two of everything --
# two doc pages, two icons, two places to add any new feature -- for one idea.
#
# The merge keeps this module's ENGINE (only it handles non-isohedral seeds,
# where a face plane's distance from the centre varies) and the icosahedron
# operator's IDENTITY and UI (only it had the Crennell index and the shell
# toggles).  Keeping `mesh.icosahedron_stellation_add` as the bl_idname, with
# `solid`'s item ids and default unchanged and `seed` defaulting to the
# icosahedron, means a .blend built with the old operator reopens and
# rebuilds identically.  `mesh.general_stellation_add` survives as a
# deprecated shim so its objects stay live-editable too.
#
# Shell toggles are a fixed bank of properties drawn with per-seed labels:
# Blender fixes the property set at class-definition time, but `text=`
# overrides the label at draw time, so one bank serves seeds with 4 shells
# and seeds with 29.  The first eleven keep their classical names (`sh_e1`,
# `sh_f1`, ...) so stored settings from the old operator still land on the
# shell they named.  Past the end of the bank the Cell Code field is the
# escape hatch -- the bank is a UI limit, not a limit on what can be built.
# --------------------------------------------------------------------------
try:
    import bpy
    from bpy.props import (EnumProperty, FloatProperty, BoolProperty,
                           StringProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _SEED_ITEMS = [(s, s.replace('_', ' ').title(), "") for s in SEEDS]
    _PRESET_CACHE = {}
    _ENGINE_CACHE = {}

    # The toggle bank.  The first eleven carry Du Val's names, which is what
    # the icosahedron operator called them before the merge; the rest are
    # positional.  29 covers every seed shipped -- the rhombic
    # triacontahedron is the widest, at 29 shells.
    _SLOT_PROPS = (['sh_' + lb for lb in DUVAL_LABELS]
                   + ['sh_%02d' % i for i in range(11, 29)])

    _FAMOUS = {
        1: "Icosahedron", 2: "First stellation (small triambic)",
        3: "Compound of 5 octahedra", 4: "Third stellation",
        6: "Second stellation", 7: "Great icosahedron",
        8: "Final stellation (echidnahedron)",
        22: "Compound of 10 tetrahedra", 26: "Excavated dodecahedron",
        30: "Medial triambic icosahedron",
        47: "Compound of 5 tetrahedra (chiral)",
    }

    def _crennell_items():
        items = []
        for k in range(1, 60):
            nm = _FAMOUS.get(k, "Du Val %s" % _CRENNELL_STR[k])
            kind = "reflexible" if k in CRENNELL_REFLEXIBLE else "chiral"
            items.append((str(k), "%2d. %s" % (k, nm),
                          "Du Val %s (%s)" % (_CRENNELL_STR[k], kind)))
        items.append(('CUSTOM', "Custom (shell toggles)",
                      "Choose Du Val shells a..g3 by hand"))
        return items

    # Static, so `default='8'` still works and the item ids ('1'..'59',
    # 'CUSTOM') are exactly the ones already stored in users' files.
    _ITEMS = _crennell_items()

    def _engine_for(seed):
        """Cached engine for a seed.  Enumerating the arrangement costs
        about a second, so it happens once, lazily, and never during a draw
        that does not need shell labels."""
        if seed not in _ENGINE_CACHE:
            _ENGINE_CACHE[seed] = stellations_of(seed)
        return _ENGINE_CACHE[seed]

    def _slot_labels(seed):
        """Shell labels for the toggle bank, in the order drawn.

        The icosahedron is drawn in Du Val's order (a b c d e1 e2 f1 f2 g1
        g2 g3), which is NOT this engine's internal order: it sorts each
        power by radius and so puts e2 before e1, and f2 before f1.  Slot i
        has to mean the shell whose name is printed on it.
        """
        eng = _engine_for(seed)
        if seed == 'icosahedron':
            return list(DUVAL_LABELS)
        return list(eng.labels)

    def _preset_items(self, context):
        seed = self.seed or SEEDS[0]
        if seed not in _PRESET_CACHE:
            _PRESET_CACHE[seed] = [(k, title, note)
                                   for k, title, code, note
                                   in named_presets(seed)]
        return _PRESET_CACHE[seed]

    def _seed_update(self, context):
        ids = [it[0] for it in _preset_items(self, context)]
        if ids and self.preset not in ids:
            self.preset = ids[0]

    try:
        from .styles import net_style as _net_style
    except ImportError:
        from styles import net_style as _net_style

    class MESH_OT_icosahedron_stellation_add(bpy.types.Operator,
                                             _net_style.NetStyleProps):
        """Add a stellation of a seed polyhedron -- the solid whose faces lie
        in the seed's own face planes.  Any of the 59 icosahedra (Coxeter/
        Du Val/Flather/Petrie) by Crennell index, a named stellation of
        another seed, or a cell set chosen by hand"""
        bl_idname = "mesh.icosahedron_stellation_add"
        bl_label = "Stellation"
        bl_options = {'REGISTER', 'UNDO'}

        seed: EnumProperty(
            name="Seed", items=_SEED_ITEMS, default='icosahedron',
            update=_seed_update,
            description="Base polyhedron whose face-plane arrangement is "
                        "stellated. Its own symmetry decides which cells are "
                        "interchangeable, and so what counts as a shell")
        mode: EnumProperty(
            name="Select by",
            items=[('PRESET', "Named",
                    "Pick a published stellation by name"),
                   ('CUSTOM', "Shells",
                    "Switch individual cell shells on and off"),
                   ('CODE', "Cell code",
                    "Type a shell code, e.g. Du Val 'a b c d e1' -- the "
                    "notation the literature uses")],
            default='PRESET',
            description="How to choose which cells are solid")
        solid: EnumProperty(
            name="Stellation", items=_ITEMS, default='8',
            description="Which stellation to build, by Crennell index 1-59, "
                        "or Custom to pick Du Val shells by hand")
        preset: EnumProperty(
            name="Stellation", items=_preset_items,
            description="Which named stellation of the seed to build")

        _SH_DESC = ("Fill this cell shell when building a custom stellation "
                    "(inner shell first)")
        sh_a: BoolProperty(name="a (core)", default=True, description=_SH_DESC)
        sh_b: BoolProperty(name="b", default=True, description=_SH_DESC)
        sh_c: BoolProperty(name="c", default=False, description=_SH_DESC)
        sh_d: BoolProperty(name="d", default=False, description=_SH_DESC)
        sh_e1: BoolProperty(name="e1", default=False, description=_SH_DESC)
        sh_e2: BoolProperty(name="e2", default=False, description=_SH_DESC)
        sh_f1: BoolProperty(name="f1 (chiral)", default=False,
                            description=_SH_DESC)
        sh_f2: BoolProperty(name="f2", default=False, description=_SH_DESC)
        sh_g1: BoolProperty(name="g1", default=False, description=_SH_DESC)
        sh_g2: BoolProperty(name="g2", default=False, description=_SH_DESC)
        sh_g3: BoolProperty(name="g3 (outer)", default=False,
                            description=_SH_DESC)
        sh_11: BoolProperty(name="11", default=False, description=_SH_DESC)
        sh_12: BoolProperty(name="12", default=False, description=_SH_DESC)
        sh_13: BoolProperty(name="13", default=False, description=_SH_DESC)
        sh_14: BoolProperty(name="14", default=False, description=_SH_DESC)
        sh_15: BoolProperty(name="15", default=False, description=_SH_DESC)
        sh_16: BoolProperty(name="16", default=False, description=_SH_DESC)
        sh_17: BoolProperty(name="17", default=False, description=_SH_DESC)
        sh_18: BoolProperty(name="18", default=False, description=_SH_DESC)
        sh_19: BoolProperty(name="19", default=False, description=_SH_DESC)
        sh_20: BoolProperty(name="20", default=False, description=_SH_DESC)
        sh_21: BoolProperty(name="21", default=False, description=_SH_DESC)
        sh_22: BoolProperty(name="22", default=False, description=_SH_DESC)
        sh_23: BoolProperty(name="23", default=False, description=_SH_DESC)
        sh_24: BoolProperty(name="24", default=False, description=_SH_DESC)
        sh_25: BoolProperty(name="25", default=False, description=_SH_DESC)
        sh_26: BoolProperty(name="26", default=False, description=_SH_DESC)
        sh_27: BoolProperty(name="27", default=False, description=_SH_DESC)
        sh_28: BoolProperty(name="28", default=False, description=_SH_DESC)

        hand: EnumProperty(
            name="Chirality",
            items=[('BOTH', "Both hands",
                    "Fill the whole shell, giving a reflexible solid"),
                   ('A', "One hand", "Keep one enantiomorph only"),
                   ('B', "Other hand", "Keep the mirror enantiomorph")],
            default='BOTH',
            description="A chiral shell splits into a mirror-image pair of "
                        "half-shells. Keeping one gives a chiral stellation "
                        "-- this is what separates Crennell 33-59 from their "
                        "reflexible namesakes")
        cell_code: StringProperty(
            name="Cell Code", default="a b c d e1",
            description="Shell labels separated by spaces. Du Val names work "
                        "for the icosahedron; add 'a' or 'b' to a chiral "
                        "shell to keep one hand, e.g. 'f1a'")

        style: EnumProperty(
            name="Style",
            description="How the stellation is rendered",
            items=[('SOLID', "Solid", ""),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo modifier"),
                   ('WIRE', "Struts", "Wireframe modifier"),
                   ('BALLSTICK', "Ball and Stick",
                    "Edges as solid cylindrical struts and vertices "
                    "as small spheres (ball-and-stick model)"),
                   ('WIREFRAME', "Wireframe",
                    "Mesh edges only, displayed as a wireframe"),
                   _net_style.net_enum_item()],
            default='SOLID')
        border: FloatProperty(name="Border", default=0.06, min=0.005, max=1.0,
                              description="Leonardo face frame width")
        thickness: FloatProperty(name="Thickness", default=0.05, min=0.001,
                                 max=1.0, description="Panel/strut thickness")
        strut_radius: FloatProperty(
            name="Strut Radius", default=0.02, min=0.001, max=0.5,
            description="Ball-and-stick edge cylinder radius")
        node_radius: FloatProperty(
            name="Node Radius", default=0.035, min=0.0, max=0.5,
            description="Ball-and-stick vertex sphere radius (0 = no nodes)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0,
                             description="Overall size of the result")

        # ---- resolving the cell code -------------------------------------
        def _effective_mode(self):
            """`solid == 'CUSTOM'` is how the pre-merge operator spelled
            shell mode, so a file saved that way still opens on shells."""
            if (self.mode == 'PRESET' and self.seed == 'icosahedron'
                    and self.solid == 'CUSTOM'):
                return 'CUSTOM'
            return self.mode

        def _shell_of(self, eng, token):
            sh = eng.shell_by_label.get(token)
            if sh is None and token in eng.classical:
                sh = eng.shell_by_label[eng.classical[token]]
            return sh

        def _apply_hand(self, eng, code):
            """Replace any chiral shell in `code` with one of its hands."""
            if self.hand == 'BOTH':
                return code
            idx = 0 if self.hand == 'A' else 1
            out = []
            for tok in code:
                sh = self._shell_of(eng, tok) if isinstance(tok, str) else None
                if sh is not None and sh['chiral']:
                    out.append(sh['hands'][idx])
                else:
                    out.append(tok)
            return out

        def _resolve(self, context):
            """-> (cell_code, title).  Raises ValueError/KeyError whose
            message is meant to be shown to the user."""
            seed = self.seed or SEEDS[0]
            eng = _engine_for(seed)
            mode = self._effective_mode()

            if mode == 'CODE':
                toks = self.cell_code.replace(',', ' ').split()
                if not toks:
                    raise ValueError("the cell code is empty")
                for t in toks:
                    eng.cells_of_token(t)       # validates; raises KeyError
                return self._apply_hand(eng, toks), 'Stellation'

            if mode == 'CUSTOM':
                labels = _slot_labels(seed)
                on = [lb for lb, pr in zip(labels, _SLOT_PROPS)
                      if getattr(self, pr)]
                if not on:
                    raise ValueError("no cell shells are switched on")
                return self._apply_hand(eng, on), 'Stellation'

            if seed == 'icosahedron':
                k = int(self.solid)
                return (crennell_code(k, hand=(1 if self.hand == 'B' else 0)),
                        crennell_title(k))

            ids = [it[0] for it in _preset_items(self, context)]
            key = self.preset if self.preset in ids \
                else (ids[0] if ids else 'core')
            code = next((c for kk, t, c, n in named_presets(seed)
                         if kk == key), ['a'])
            title = next((t for kk, t, c, n in named_presets(seed)
                          if kk == key), key)
            return expand_code(eng, code), title

        def execute(self, context):
            seed = self.seed or SEEDS[0]
            try:
                eng = _engine_for(seed)
                code, title = self._resolve(context)
                V, F = eng.build(code)
            except (ValueError, KeyError) as exc:
                self.report({'ERROR'}, str(exc).strip("'"))
                return {'CANCELLED'}
            if not F:
                self.report({'ERROR'},
                            "that cell set encloses nothing -- switch on at "
                            "least one shell that reaches the surface")
                return {'CANCELLED'}

            Vs = [tuple(c * self.scale for c in v) for v in V]
            Fl = [list(f) for f in F]
            if self.style == 'NET':
                return _net_style.emit_net_from_operator(
                    self, context, Vs, Fl, title)
            me = bpy.data.meshes.new(title)
            me.from_pydata(Vs, [], Fl)
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
                    from .styles import ball_and_stick
                except ImportError:
                    from styles import ball_and_stick
                ball_and_stick.rebuild(obj, self.strut_radius,
                                       self.node_radius)
            elif self.style == 'WIREFRAME':
                obj.display_type = 'WIRE'

            # Say which of how many: 59 is a property of the icosahedron
            # PLUS Miller's rules, not a universal constant, so the number
            # only means anything next to the thing it counts.
            if seed == 'icosahedron' and self._effective_mode() == 'PRESET':
                self.report({'INFO'}, "%s (%s of 59): V=%d F=%d"
                            % (title, self.solid, len(V), len(F)))
            else:
                self.report({'INFO'}, "%s [%s]: V=%d F=%d"
                            % (title, seed.replace('_', ' '), len(V), len(F)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'seed')
            lay.prop(self, 'mode')
            mode = self._effective_mode()

            if mode == 'PRESET':
                lay.prop(self, 'solid' if self.seed == 'icosahedron'
                         else 'preset')
            elif mode == 'CUSTOM':
                labels = _slot_labels(self.seed)
                box = lay.box()
                box.use_property_split = False
                box.label(text="Cell shells (inner -> outer):")
                grid = box.grid_flow(row_major=True, columns=4,
                                     even_columns=True, align=True)
                for lb, pr in zip(labels, _SLOT_PROPS):
                    grid.prop(self, pr, text=lb)
                if len(labels) > len(_SLOT_PROPS):
                    box.label(text="%d more shells -- use Cell code"
                                   % (len(labels) - len(_SLOT_PROPS)),
                              icon='INFO')
            else:
                lay.prop(self, 'cell_code')

            # Only offered once the arrangement is known to have a chiral
            # shell; drawing it always would promise a choice most seeds
            # cannot honour.
            eng = _ENGINE_CACHE.get(self.seed)
            if eng is not None and any(sh['chiral'] for sh in eng.shells):
                lay.prop(self, 'hand')

            if self.style == 'NET':
                _net_style.draw_net_props(lay, self)
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style in ('LEONARDO', 'WIRE'):
                lay.prop(self, 'thickness')
            if self.style == 'BALLSTICK':
                lay.prop(self, 'strut_radius')
                lay.prop(self, 'node_radius')
            lay.prop(self, 'scale')

    class MESH_OT_general_stellation_add(bpy.types.Operator):
        """Deprecated: use Stellation.  Kept registered so objects built with
        the pre-merge General Stellation operator stay live-editable; it
        forwards to the merged operator.  Remove after one release"""
        bl_idname = "mesh.general_stellation_add"
        bl_label = "General Stellation (deprecated)"
        bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

        seed: EnumProperty(name="Seed", items=_SEED_ITEMS, default=SEEDS[0],
                           description="Base polyhedron to stellate")
        stellation: EnumProperty(name="Stellation", items=_preset_items,
                                 description="Which named stellation")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0,
                             description="Overall size of the result")

        def execute(self, context):
            return bpy.ops.mesh.icosahedron_stellation_add(
                seed=self.seed, mode='PRESET', preset=self.stellation,
                scale=self.scale)

    def _menu_func(self, context):
        self.layout.operator("mesh.icosahedron_stellation_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_icosahedron_stellation_add)
        bpy.utils.register_class(MESH_OT_general_stellation_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_general_stellation_add)
        bpy.utils.unregister_class(MESH_OT_icosahedron_stellation_add)
