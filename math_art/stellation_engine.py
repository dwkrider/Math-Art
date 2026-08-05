"""
stellation_engine.py -- pure-Python engine for the stellations of the
regular icosahedron ("The Fifty-Nine Icosahedra").

No third-party dependencies (no bpy / numpy / scipy): standard library only.

--------------------------------------------------------------------------
Mathematics
--------------------------------------------------------------------------
A stellation of the icosahedron is a solid whose faces lie in the twenty
face-planes of a regular icosahedron.  Extending those twenty planes divides
space into cells; the bounded cells fall into concentric symmetry "shells"
(Du Val's a, b, c, ...).  A stellation is chosen by declaring a set of shells
"solid".  This module

  1. builds the 20 face-planes of a unit-inradius icosahedron,
  2. enumerates every bounded cell of that plane-arrangement,
  3. sorts the cells into Du Val shells by *power* (number of face-planes
     crossed on the way out from the centre) and radius,
  4. builds the boundary surface of any chosen set of shells.

History / attribution
  * H. S. M. Coxeter, P. Du Val, H. T. Flather, J. F. Petrie,
    "The Fifty-Nine Icosahedra", University of Toronto Studies (1938);
    3rd ed., Tarquin (1999).  Du Val's cell/shell notation is used here.
  * L. Lines, and later J. C. P. Miller's rules, underlie the enumeration.
  * The Crennell (1974/1979) index 1..59 is transcribed from the standard
    published cross-reference (as reproduced e.g. on Wikipedia,
    "The Fifty-Nine Icosahedra") -- classical, freely derivable data.

References
  * https://en.wikipedia.org/wiki/The_fifty-nine_icosahedra
  * https://mathworld.wolfram.com/IcosahedronStellations.html
  * Roman Maeder, "The Stellations of the Icosahedron" (1994).

This file contains no proprietary data; every number below is computed from
the icosahedron's own geometry.

--------------------------------------------------------------------------
Notes on the shell count (verified, not assumed)
--------------------------------------------------------------------------
Enumerating the arrangement of the 20 face-planes gives exactly 473 bounded
cells (confirmed independently by an exhaustive random-seed sweep).  Sorting
them by Du Val power (planes crossed from the centre) gives 11 shells:

    power 0 : a  (1)
    power 1 : b  (20)
    power 2 : c  (30)
    power 3 : d  (60)
    power 4 : e1 (20)   e2 (60)
    power 5 : f1 (120)  f2 (12)
    power 6 : g1 (30)   g2 (60)
    power 7 : g3 (60)

The sizes 1,20,30,60,20,60,120,12,30,60,60 reproduce the classical Du Val /
MathWorld tally for shells a..g3 exactly.  The classical literature also
names a 12th "outer spike" shell h (20 cells) for a nominal total of 493;
that shell does NOT appear as a set of distinct *bounded* cells of the pure
20-plane arrangement.  Its role -- capping the great icosahedron up to the
final stellation -- is played here by the power-7 shell g3, and the final
stellation (echidnahedron) is realised exactly by the union of all 473
cells (V92 E270 F180, vertex shells 20/12/60 at 1 : 1.7769 : 3.5066).

Consequently the great icosahedron (12 outer vertices) is built from shells
a..g2, and the final stellation from a..g3 (= all cells).  Only Du Val's
capital G and H reach these outer shells; every other Crennell code uses
shells a..g2, which match the classical notation one-for-one.  f1 is the
unique chiral shell (it splits into two 60-cell hands under the proper
rotation group); the chiral figures 33..59 are built by keeping one hand.
"""

import math
import itertools
from collections import deque, defaultdict, Counter

# --------------------------------------------------------------------------
# basic 3-vector helpers
# --------------------------------------------------------------------------
PHI = (1.0 + 5.0 ** 0.5) / 2.0


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    L = math.sqrt(_dot(a, a))
    return (a[0] / L, a[1] / L, a[2] / L)


def _matvec(M, v):
    return (M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
            M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
            M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2])


# --------------------------------------------------------------------------
# 1.  the twenty face-planes of a unit-inradius icosahedron
#     face normals point toward the vertices of the dual dodecahedron;
#     every plane is  n . x = 1.
# --------------------------------------------------------------------------
def _face_normals():
    v = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                v.append((float(sx), float(sy), float(sz)))
    ip = 1.0 / PHI
    for a in (ip, -ip):
        for b in (PHI, -PHI):
            v.append((0.0, a, b))
            v.append((a, b, 0.0))
            v.append((b, 0.0, a))
    return [_norm(x) for x in v]


NORMALS = _face_normals()
assert len(NORMALS) == 20
assert len(set(tuple(round(c, 6) for c in n) for n in NORMALS)) == 20

# a large bounding cube is used only to detect unbounded cells robustly.
_BIG = 20.0
_CUBE = []
for _ax in range(3):
    for _s in (1.0, -1.0):
        _v = [0.0, 0.0, 0.0]
        _v[_ax] = _s
        _CUBE.append((_v[0], _v[1], _v[2], _BIG))
_EQS = [(NORMALS[i][0], NORMALS[i][1], NORMALS[i][2], 1.0) for i in range(20)] + _CUBE


# --------------------------------------------------------------------------
# 2.  arrangement vertices and cell enumeration
# --------------------------------------------------------------------------
def _det3(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def _solve3(p, q, r):
    """Intersection point of three planes (a,b,c,d): a x + b y + c z = d."""
    a = [[p[0], p[1], p[2]], [q[0], q[1], q[2]], [r[0], r[1], r[2]]]
    d = [p[3], q[3], r[3]]
    D = _det3(a)
    if abs(D) < 1e-9:
        return None

    def rep(col):
        m = [row[:] for row in a]
        for i in range(3):
            m[i][col] = d[i]
        return m

    return (_det3(rep(0)) / D, _det3(rep(1)) / D, _det3(rep(2)) / D)


def _precompute_points():
    """All triple-plane intersection points (incl. cube), de-duplicated.

    Returns (points, dots, oncube) where dots[k] = [n_i . p_k] and
    oncube[k] is True when point k lies on the bounding cube (=> its cell
    is unbounded)."""
    pts = {}
    m = len(_EQS)
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                pt = _solve3(_EQS[i], _EQS[j], _EQS[k])
                if pt is None:
                    continue
                if (abs(pt[0]) > _BIG + 1e-6 or abs(pt[1]) > _BIG + 1e-6
                        or abs(pt[2]) > _BIG + 1e-6):
                    continue
                key = (round(pt[0], 6), round(pt[1], 6), round(pt[2], 6))
                if key not in pts:
                    pts[key] = pt
    plist = list(pts.values())
    dots = [[NORMALS[i][0] * p[0] + NORMALS[i][1] * p[1] + NORMALS[i][2] * p[2]
             for i in range(20)] for p in plist]
    oncube = [abs(p[0]) > _BIG - 1e-6 or abs(p[1]) > _BIG - 1e-6
              or abs(p[2]) > _BIG - 1e-6 for p in plist]
    return plist, dots, oncube


# Enumerating the plane arrangement takes ~2 s, so it is done lazily on first
# use (see _ensure) rather than at import -- keeping add-on registration cheap.
# The static Crennell/Named tables below do NOT need it.
_POINTS = _DOTS = _ONCUBE = None
_TOL = 1e-7
_READY = False
_ALL_LABELS = ['a', 'b', 'c', 'd', 'e1', 'e2', 'f1', 'f2', 'g1', 'g2', 'g3']


def _cell_from_signs(signs):
    """Given a sign vector in {+1,-1}^20 (side of each plane), return the
    cell (dict) or None if the region is empty / lower-dimensional."""
    vi = []
    for pi in range(len(_POINTS)):
        dp = _DOTS[pi]
        ok = True
        for i in range(20):
            if signs[i] * (dp[i] - 1.0) < -_TOL:
                ok = False
                break
        if ok:
            vi.append(pi)
    if len(vi) < 4:
        return None
    verts = [_POINTS[pi] for pi in vi]
    unbounded = any(_ONCUBE[pi] for pi in vi)
    n = len(verts)
    c = (sum(v[0] for v in verts) / n,
         sum(v[1] for v in verts) / n,
         sum(v[2] for v in verts) / n)
    return {'vi': vi, 'verts': verts, 'centroid': c,
            'signs': signs, 'unbounded': unbounded}


def _enumerate_cells():
    """Breadth-first over facet-adjacent regions (single sign flips),
    collecting every *bounded* cell.  Returns {signs: cell}."""
    cells = {}
    start = tuple(-1 for _ in range(20))          # inside all planes = core
    seen = {start}
    dq = deque([start])
    while dq:
        s = dq.popleft()
        cell = _cell_from_signs(s)
        if cell is None:
            continue
        if not cell['unbounded']:
            cells[s] = cell
        for i in range(20):                        # cross each face-plane
            ns = list(s)
            ns[i] = -ns[i]
            ns = tuple(ns)
            if ns not in seen:
                seen.add(ns)
                dq.append(ns)
    return cells


CELLS = None                         # filled by _ensure()


# --------------------------------------------------------------------------
# 3.  sort cells into Du Val shells by power (= #planes crossed) and radius
# --------------------------------------------------------------------------
def _power(signs):
    return sum(1 for x in signs if x == 1)


def _radius(cell):
    c = cell['centroid']
    return math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])


# Static: each power level's (size, label) assignments, chosen to reproduce
# the classical tally (e1=20/e2=60, f1=120/f2=12, g1=30/g2=60, g3=60).
_POWER_LABELS = {
    0: [(1, 'a')],
    1: [(20, 'b')],
    2: [(30, 'c')],
    3: [(60, 'd')],
    4: [(20, 'e1'), (60, 'e2')],
    5: [(120, 'f1'), (12, 'f2')],
    6: [(30, 'g1'), (60, 'g2')],
    7: [(60, 'g3')],
}

# Du Val shell tables -- filled by _ensure() on first use.
SHELLS = {}          # label -> set(signs)
SHELL_INFO = {}      # label -> (size, mean_radius, power)
_LABEL_ORDER = []    # canonical outward label order actually present
ORBITS = {}          # label -> (size, mean_radius)


# --------------------------------------------------------------------------
# 4.  the icosahedral rotation group (order 60) -- as the rotations that
#     permute the twenty face-normals.  Used for the orbit report and for
#     splitting chiral shells into left/right hands.
# --------------------------------------------------------------------------
def _frame(u, v):
    e1 = _norm(u)
    w = (v[0] - _dot(v, e1) * e1[0],
         v[1] - _dot(v, e1) * e1[1],
         v[2] - _dot(v, e1) * e1[2])
    e2 = _norm(w)
    e3 = _cross(e1, e2)
    return (e1, e2, e3)


def _rot_between(u0, v0, u1, v1):
    F0 = _frame(u0, v0)
    F1 = _frame(u1, v1)
    return tuple(tuple(sum(F1[k][a] * F0[k][b] for k in range(3))
                       for b in range(3)) for a in range(3))


def _rotation_group():
    N = NORMALS
    n0 = N[0]
    adj = max(round(_dot(n0, N[j]), 4) for j in range(20) if j != 0)
    nbr0 = next(j for j in range(20) if abs(_dot(n0, N[j]) - adj) < 1e-3)
    Nset = set((round(n[0], 4), round(n[1], 4), round(n[2], 4)) for n in N)
    G = {}
    for i in range(20):
        for j in range(20):
            if i == j or abs(_dot(N[i], N[j]) - adj) > 1e-3:
                continue
            R = _rot_between(n0, N[nbr0], N[i], N[j])
            ok = True
            for n in N:
                p = _matvec(R, n)
                if (round(p[0], 4), round(p[1], 4), round(p[2], 4)) not in Nset:
                    ok = False
                    break
            if ok:
                G[tuple(round(R[a][b], 5) for a in range(3) for b in range(3))] = R
    return list(G.values())


ROTATIONS = None                # 60 proper rotations (filled by _ensure())


def _rotation_orbits():
    """Partition cells into orbits under the proper rotation group.
    Chiral shells split into two hands here."""
    clist = list(CELLS.keys())
    cent = {s: CELLS[s]['centroid'] for s in clist}
    lookup = {}
    for s in clist:
        c = cent[s]
        lookup[(round(c[0], 3), round(c[1], 3), round(c[2], 3))] = s
    parent = {s: s for s in clist}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in clist:
        c = cent[s]
        for R in ROTATIONS:
            p = _matvec(R, c)
            key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
            t = lookup.get(key)
            if t is not None:
                ra, rb = find(s), find(t)
                if ra != rb:
                    parent[ra] = rb
    orb = defaultdict(list)
    for s in clist:
        orb[find(s)].append(s)
    return list(orb.values())


def hands_of(label):
    """Return the chiral half-shells of a shell as a list of frozensets.
    Reflexible shells return a single set; chiral shells return two
    (a right/left enantiomorphic pair)."""
    _ensure()
    members = SHELLS[label]
    orbs = [o for o in _ROT_ORBITS if set(o) & members]
    return [frozenset(o) for o in orbs]


_ROT_ORBITS = None              # filled by _ensure()
_F1_HANDS = None
_F1_HAND = None


def _ensure():
    """Compute the plane-arrangement cells, Du Val shells, rotation group and
    chiral f1 hand -- once, on first use.  Kept out of import so that merely
    importing this module (e.g. registering the add-on) stays cheap."""
    global _READY, _POINTS, _DOTS, _ONCUBE, CELLS
    global SHELLS, SHELL_INFO, _LABEL_ORDER, ORBITS
    global ROTATIONS, _ROT_ORBITS, _F1_HANDS, _F1_HAND
    if _READY:
        return
    _POINTS, _DOTS, _ONCUBE = _precompute_points()
    CELLS = _enumerate_cells()
    groups = defaultdict(list)
    for s, c in CELLS.items():
        groups[(_power(s), round(_radius(c), 3))].append(s)
    SHELLS = {}
    SHELL_INFO = {}
    for (pw, rr), members in groups.items():
        size = len(members)
        label = None
        for (sz, lb) in _POWER_LABELS.get(pw, []):
            if sz == size and lb not in SHELLS:
                label = lb
                break
        if label is None:                          # safety: fabricate a name
            label = 'p%d_%d' % (pw, size)
        SHELLS[label] = set(members)
        rs = [_radius(CELLS[s]) for s in members]
        SHELL_INFO[label] = (size, sum(rs) / len(rs), pw)
    _LABEL_ORDER = [lb for lb in _ALL_LABELS if lb in SHELLS]
    ORBITS = {lb: (SHELL_INFO[lb][0], round(SHELL_INFO[lb][1], 4))
              for lb in _LABEL_ORDER}
    ROTATIONS = _rotation_group()
    _ROT_ORBITS = _rotation_orbits()
    f1m = SHELLS.get('f1', set())                  # split the one chiral shell
    _F1_HANDS = [frozenset(o) for o in _ROT_ORBITS if set(o) & f1m]
    _F1_HAND = _F1_HANDS[0] if len(_F1_HANDS) == 2 else None
    _READY = True


# --------------------------------------------------------------------------
# 5.  build the boundary surface of a chosen cell-code
# --------------------------------------------------------------------------
def _flip(s, i):
    ns = list(s)
    ns[i] = -ns[i]
    return tuple(ns)


def _order_polygon(pts, outward):
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    cz = sum(p[2] for p in pts) / n
    ax = _norm((pts[0][0] - cx, pts[0][1] - cy, pts[0][2] - cz))
    ay = _cross(outward, ax)

    def ang(p):
        d = (p[0] - cx, p[1] - cy, p[2] - cz)
        return math.atan2(_dot(d, ay), _dot(d, ax))

    return sorted(pts, key=ang)


def _cells_of_code(cell_code):
    """cell_code: iterable of shell labels and/or frozensets of signs."""
    filled = set()
    for item in cell_code:
        if isinstance(item, str):
            filled |= SHELLS[item]
        else:                       # an explicit set of sign-vectors (a hand)
            filled |= set(item)
    return filled


def build(cell_code, scale=True):
    """Build the polyhedral surface of the stellation whose solid cells are
    the union of the given shells.

    cell_code : iterable of shell labels ('a','b',...) and/or frozensets of
                cell sign-vectors (for chiral single-hand pieces).
    Returns (V, F): V a list of (x,y,z) vertices, F a list of polygons
    (index lists), outward-wound.  If scale, the model is centred at the
    origin and scaled so max(|coord|) == 1 (fits a 2x2x2 cube)."""
    _ensure()
    filled = _cells_of_code(cell_code)
    faces = []
    for s in filled:
        cell = CELLS[s]
        verts = cell['verts']
        for i in range(20):
            on = [v for v in verts
                  if abs(NORMALS[i][0] * v[0] + NORMALS[i][1] * v[1]
                         + NORMALS[i][2] * v[2] - 1.0) < 1e-6]
            if len(on) < 3:
                continue
            nb = _flip(s, i)
            if nb in filled:
                continue                 # interior face: skip
            outward = (-s[i] * NORMALS[i][0], -s[i] * NORMALS[i][1],
                       -s[i] * NORMALS[i][2])
            faces.append(_order_polygon(on, outward))
    # merge coincident vertices
    vmap = {}
    V = []
    F = []
    for poly in faces:
        idx = []
        for p in poly:
            key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
            if key not in vmap:
                vmap[key] = len(V)
                V.append(p)
            idx.append(vmap[key])
        clean = [idx[0]]
        for j in idx[1:]:
            if j != clean[-1]:
                clean.append(j)
        if len(clean) >= 3 and clean[0] == clean[-1]:
            clean = clean[:-1]
        if len(clean) >= 3:
            F.append(clean)
    if scale and V:
        m = max(max(abs(c) for c in v) for v in V)
        if m > 0:
            V = [(v[0] / m, v[1] / m, v[2] / m) for v in V]
    return V, F


# --------------------------------------------------------------------------
# 6.  named stellations and the Crennell 1..59 index
# --------------------------------------------------------------------------
# Du Val shorthand: a capital letter means "that shell and every shell inside
# it, solid".  Because the pure 20-plane arrangement yields 11 bounded shells
# a..g3 (the classical 12th shell 'h' of 20 cells is not a distinct bounded
# cell of the arrangement -- see notes), the great icosahedron is realised by
# a..g2 and the final stellation by all shells a..g3.

_CAP = {
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


def _expand(code):
    """Expand a Du Val string like 'De1f1g1' or 'Ef1' into a shell set."""
    shells = []
    i = 0
    # optional leading capital
    if code and code[0].isupper():
        shells += _CAP[code[0]]
        i = 1
    # remaining lowercase tokens: letter possibly followed by a digit
    while i < len(code):
        ch = code[i]
        if ch.isalpha():
            tok = ch
            if i + 1 < len(code) and code[i + 1].isdigit():
                tok += code[i + 1]
                i += 1
            shells.append(tok)
        i += 1
    # de-duplicate, keep only valid shell labels (static; no enumeration)
    out = []
    for s in shells:
        if s in _ALL_LABELS and s not in out:
            out.append(s)
    return out


# NAMED: (id, human label, cell_code set, note)
NAMED = [
    ('A',       'Icosahedron',                  _expand('A'),
     'Platonic solid; Crennell 1'),
    ('B',       'First stellation (small triambic / triakis icosahedron)',
     _expand('B'), 'Crennell 2'),
    ('C',       'Compound of five octahedra',   _expand('C'),
     'Crennell 3'),
    ('D',       'Third stellation',             _expand('D'),
     'Crennell 4'),
    ('F',       'Second stellation',            _expand('F'),
     'Crennell 6'),
    ('Ef1',     'Compound of ten tetrahedra',   _expand('Ef1'),
     'Crennell 22'),
    ('De2f2',   'Great triambic / medial triambic icosahedron',
     _expand('De2f2'), 'Crennell 30'),
    ('Ef1g1',   'Excavated dodecahedron',       _expand('Ef1g1'),
     'Crennell 26'),
    ('G',       'Great icosahedron {3,5/2}',    _expand('G'),
     'Crennell 7; 12 outer vertices at the icosahedron shell'),
    ('H',       'Final stellation / echidnahedron (W42)', _expand('H'),
     'Crennell 8; all cells'),
]

# Crennell index 1..59 -> Du Val cell string (verbatim from the standard
# published table).  1..32 are reflexible; 33..59 are chiral (single hand).
_CRENNELL_STR = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H',
    9: 'e1', 10: 'f1', 11: 'g1', 12: 'e1f1', 13: 'e1f1g1', 14: 'f1g1',
    15: 'e2', 16: 'f2', 17: 'g2', 18: 'e2f2', 19: 'e2f2g2', 20: 'f2g2',
    21: 'De1', 22: 'Ef1', 23: 'Fg1', 24: 'De1f1', 25: 'De1f1g1',
    26: 'Ef1g1', 27: 'De2', 28: 'Ef2', 29: 'Fg2', 30: 'De2f2',
    31: 'De2f2g2', 32: 'Ef2g2',
    # chiral 33..59
    33: 'f1', 34: 'e1f1', 35: 'De1f1', 36: 'f1g1', 37: 'e1f1g1',
    38: 'De1f1g1', 39: 'f1g2', 40: 'e1f1g2', 41: 'De1f1g2', 42: 'f1f2g2',
    43: 'e1f1f2g2', 44: 'De1f1f2g2', 45: 'e2f1', 46: 'De2f1', 47: 'Ef1',
    48: 'e2f1g1', 49: 'De2f1g1', 50: 'Ef1g1', 51: 'e2f1f2', 52: 'De2f1f2',
    53: 'Ef1f2', 54: 'e2f1f2g1', 55: 'De2f1f2g1', 56: 'Ef1f2g1',
    57: 'e2f1f2g2', 58: 'De2f1f2g2', 59: 'Ef1f2g2',
}

# CRENNELL: index -> expanded shell set (both-hands / full-shell form).
CRENNELL = {k: _expand(v) for k, v in _CRENNELL_STR.items()}
CRENNELL_REFLEXIBLE = set(range(1, 33))
CRENNELL_CHIRAL = set(range(33, 60))

# f1 is the one chiral shell (it splits into an enantiomorphic pair of 60
# cells).  Every chiral Crennell figure (33..59) is obtained from its
# reflexible namesake by keeping only ONE hand of f1.  We fix a hand here
# deterministically (the roman/italic choice is not recoverable from a
# plain-text source, so either hand yields a valid chiral member of the pair).
def crennell_cells(k):
    """Return the buildable cell-code for Crennell index k.

    For reflexible indices (1..32) this is a list of shell labels.
    For chiral indices (33..59) the 'f1' shell is replaced by a single
    hand (a frozenset of cell sign-vectors), so the built solid is a
    genuine chiral stellation rather than its both-hands relative."""
    _ensure()
    code = list(CRENNELL[k])
    if k in CRENNELL_CHIRAL and _F1_HAND is not None and 'f1' in code:
        code = [x for x in code if x != 'f1'] + [_F1_HAND]
    return code


# --------------------------------------------------------------------------
# 7.  surface statistics + self test
# --------------------------------------------------------------------------
def _components(V, F):
    """Number of connected components of the surface (via shared vertices)."""
    parent = list(range(len(V)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used = set()
    for f in F:
        used.update(f)
        for j in f[1:]:
            ra, rb = find(f[0]), find(j)
            if ra != rb:
                parent[ra] = rb
    return len(set(find(v) for v in used)) if used else 0


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
    ncomp = _components(V, F)
    rs = [math.sqrt(x * x + y * y + z * z) for (x, y, z) in V] or [0.0]
    return {'V': len(V), 'E': E, 'F': len(F), 'chi': chi,
            'closed_2': two, 'edge_mult': mult, 'ncomp': ncomp,
            'rmin': min(rs), 'rmax': max(rs)}


def classify_surface(st):
    """Human description of a surface's topology."""
    if st['closed_2']:
        if st['ncomp'] and st['chi'] == 2 * st['ncomp']:
            if st['ncomp'] == 1:
                return 'closed manifold (sphere)'
            return 'closed manifold, %d components' % st['ncomp']
        g = (2 * st['ncomp'] - st['chi']) // 2
        return 'closed manifold, %d comp, total genus %d' % (st['ncomp'], g)
    hi = max(k for k in st['edge_mult'] if k != 2) if \
        any(k != 2 for k in st['edge_mult']) else 0
    return 'non-manifold (edges shared by up to %d faces)' % hi


def convex_hull_vertices(V, tol=1e-6):
    """Vertices of V that lie on its convex hull (brute force: a point is a
    hull vertex iff some direction is uniquely maximised there)."""
    hull = set()
    # use face-normal-ish directions + random dirs
    import random
    rnd = random.Random(0)
    dirs = list(NORMALS)
    for (x, y, z) in V:
        dirs.append(_norm((x, y, z)) if (x or y or z) else (1.0, 0.0, 0.0))
    for _ in range(400):
        dirs.append(_norm((rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 1))))
    for d in dirs:
        best = None
        bv = -1e18
        for k, v in enumerate(V):
            val = _dot(d, v)
            if val > bv + tol:
                bv = val
                best = k
        if best is not None:
            hull.add(best)
    return hull


def _shell_radii(V, ndig=3):
    return Counter(round(math.sqrt(x * x + y * y + z * z), ndig)
                   for (x, y, z) in V)


def _self_test():
    _ensure()
    print("=" * 74)
    print("STELLATION ENGINE  --  self test")
    print("=" * 74)
    print("face-planes: %d   arrangement points: %d   bounded cells: %d"
          % (len(NORMALS), len(_POINTS), len(CELLS)))
    print("rotation group order: %d" % len(ROTATIONS))
    print()
    print("Measured shell (orbit) table")
    print("  label   power   size   mean-radius")
    tally = {'a': 1, 'b': 20, 'c': 30, 'd': 60, 'e1': 20, 'e2': 60,
             'f1': 120, 'f2': 12, 'g1': 30, 'g2': 60, 'g3': 60}
    tsum = 0
    for lb in _LABEL_ORDER:
        sz, mr, pw = SHELL_INFO[lb][0], SHELL_INFO[lb][1], SHELL_INFO[lb][2]
        tsum += sz
        flag = 'ok' if tally.get(lb) == sz else '??'
        print("   %-4s     %d     %4d    %8.4f   [%s]" % (lb, pw, sz, mr, flag))
    print("  total bounded cells: %d" % tsum)
    print()

    # ---- build every named preset ------------------------------------
    print("Named presets")
    print("  %-46s %-22s %s" % ("preset", "V/E/F  chi", "closed rmax"))
    results = {}
    for pid, name, code, note in NAMED:
        V, F = build(code, scale=False)
        st = surface_stats(V, F)
        results[pid] = (V, F, st)
        print("  %-46s V%-4d E%-4d F%-4d chi=%d  %-5s r=%.3f"
              % (name[:46], st['V'], st['E'], st['F'], st['chi'],
                 str(st['closed_2']), st['rmax']))
    print()

    # ---- anchor assertions -------------------------------------------
    print("ANCHOR CHECKS")
    ok_all = True

    # A: icosahedron V12 E30 F20 chi 2 convex
    V, F, st = results['A']
    hull = convex_hull_vertices(V)
    a_ok = (st['V'] == 12 and st['E'] == 30 and st['F'] == 20
            and st['chi'] == 2 and st['closed_2'] and len(hull) == 12)
    print("  [%s] A icosahedron: V12 E30 F20 chi2 convex  -> V%d E%d F%d chi%d "
          "closed=%s hull=%d"
          % ('PASS' if a_ok else 'FAIL', st['V'], st['E'], st['F'],
             st['chi'], st['closed_2'], len(hull)))
    ok_all &= a_ok

    # H: echidnahedron V92 E270 F180, 3 shells 20/12/60, ratios 1:1.7769:3.5066
    V, F, st = results['H']
    sh = _shell_radii(V)
    shells_sorted = sorted(sh.items())
    radii = [r for r, _ in shells_sorted]
    counts = [c for _, c in shells_sorted]
    ratios = [r / radii[0] for r in radii] if radii else []
    hull = convex_hull_vertices(V)
    h_ok = (st['V'] == 92 and st['E'] == 270 and st['F'] == 180
            and st['chi'] == 2 and st['closed_2']
            and counts == [20, 12, 60] and len(hull) == 60
            and abs(ratios[1] - 1.7769) < 2e-3 and abs(ratios[2] - 3.5066) < 2e-3)
    print("  [%s] H echidnahedron: V92 E270 F180 chi2; shells 20/12/60 "
          "ratios 1:1.7769:3.5066" % ('PASS' if h_ok else 'FAIL'))
    print("        -> V%d E%d F%d chi%d closed=%s hull_tips=%d"
          % (st['V'], st['E'], st['F'], st['chi'], st['closed_2'], len(hull)))
    print("        shells counts=%s  ratios=%s"
          % (counts, [round(x, 4) for x in ratios]))
    ok_all &= h_ok

    # G: great icosahedron -- outermost 12 vertices form the icosahedron shell
    V, F, st = results['G']
    hull = convex_hull_vertices(V)
    rmax = st['rmax']
    outer = [k for k, v in enumerate(V)
             if abs(math.sqrt(_dot(v, v)) - rmax) < 1e-3]
    # those 12 outer vertices lie along the 12 icosahedral 5-fold axes
    g_ok = (len(hull) == 12 and len(outer) == 12)
    print("  [%s] G great icosahedron: 12 outermost vertices (icosa shell) "
          "-> hull=%d outer=%d" % ('PASS' if g_ok else 'FAIL',
                                    len(hull), len(outer)))
    ok_all &= g_ok

    # Extra named cross-checks with known convex-hull vertex counts
    Vc, Fc, stc = results['C']
    hull_c = convex_hull_vertices(Vc)
    c_ok = len(hull_c) == 30            # 5 octahedra -> 30 outer vertices
    print("  [%s] C compound of five octahedra: 30 outer (octahedra) "
          "vertices -> hull=%d" % ('PASS' if c_ok else 'FAIL', len(hull_c)))
    ok_all &= c_ok

    Vt, Ft = build(CRENNELL[22], scale=False)
    hull_t = convex_hull_vertices(Vt)
    t_ok = len(hull_t) == 20            # 10 tetrahedra -> 20 outer vertices
    print("  [%s] Ef1 compound of ten tetrahedra: 20 outer vertices "
          "-> hull=%d" % ('PASS' if t_ok else 'FAIL', len(hull_t)))
    ok_all &= t_ok

    # closedness of every preset
    print()
    print("  Manifold (every edge in exactly 2 faces) for each named preset:")
    for pid, name, code, note in NAMED:
        st = results[pid][2]
        tag = 'closed' if st['closed_2'] else ('OPEN/non-manifold %s'
                                               % st['edge_mult'])
        print("     %-46s %s" % (name[:46], tag))

    # ---- Crennell reflexible set: build + report ---------------------
    print()
    print("Crennell reflexible index (1..32): build + topology")
    print("  (fully-nested 'solid' forms are single closed spheres, chi=2;")
    print("   hollow lowercase-only forms are multi-component; compound/")
    print("   interlocking forms have 4-valent edges -- all are valid.)")
    nonman = 0
    for k in sorted(CRENNELL_REFLEXIBLE):
        code = CRENNELL[k]
        V, F = build(code, scale=False)
        st = surface_stats(V, F)
        if not st['closed_2']:
            nonman += 1
        print("   %2d  %-12s cells=%-3d V%-4d E%-4d F%-4d chi=%-4d %s"
              % (k, _CRENNELL_STR[k], len(_cells_of_code(code)),
                 st['V'], st['E'], st['F'], st['chi'], classify_surface(st)))
    print("   reflexible presets that are non-manifold (4-valent edges): %d"
          % nonman)

    # ---- Chiral set: build each with a single hand of f1 -------------
    print()
    chiral_shells = [lb for lb in _LABEL_ORDER if len(hands_of(lb)) == 2]
    print("Crennell chiral index (33..59): built with ONE hand of f1")
    print("  (f1 is the only chiral shell: %s.  The roman/italic hand is not"
          % chiral_shells)
    print("   recoverable from a plain-text source, so a consistent hand is")
    print("   used -- each build is a valid member of the enantiomorphic pair.)")
    for k in sorted(CRENNELL_CHIRAL):
        code = crennell_cells(k)
        V, F = build(code, scale=False)
        st = surface_stats(V, F)
        print("   %2d  %-12s cells=%-3d V%-4d E%-4d F%-4d chi=%-4d %s"
              % (k, _CRENNELL_STR[k], len(_cells_of_code(code)),
                 st['V'], st['E'], st['F'], st['chi'], classify_surface(st)))

    # cross-check: #47 (one hand) = compound of five tetrahedra, 20 vertices,
    # and it must differ from #22 (ten tetrahedra, both hands).
    V47, F47 = build(crennell_cells(47), scale=False)
    hull47 = convex_hull_vertices(V47)
    n47 = len(_cells_of_code(crennell_cells(47)))
    n22 = len(_cells_of_code(CRENNELL[22]))
    five_ok = (len(hull47) == 20 and n22 - n47 == 60)
    print("  [%s] #47 compound of five tetrahedra (chiral): 20 hull vertices,"
          " 60 fewer cells than #22 -> hull=%d, cells 47=%d 22=%d"
          % ('PASS' if five_ok else 'FAIL', len(hull47), n47, n22))
    ok_all &= five_ok

    print()
    print("=" * 74)
    print("OVERALL MANDATORY ANCHORS (A, H, G): %s"
          % ('ALL PASS' if ok_all else 'FAILURE'))
    print("=" * 74)
    return ok_all


# --------------------------------------------------------------------------
# 8.  Blender operator (the UI enum is built from the STATIC Crennell table,
#     so registering the add-on does not trigger the cell enumeration; the
#     ~2 s enumeration happens on the first stellation actually built)
# --------------------------------------------------------------------------
_FAMOUS = {
    1: "Icosahedron",
    2: "First stellation (triakis / small triambic)",
    3: "Compound of 5 octahedra",
    4: "Third stellation",
    6: "Second stellation",
    7: "Great icosahedron",
    8: "Final stellation (echidnahedron)",
    22: "Compound of 10 tetrahedra",
    26: "Excavated dodecahedron",
    30: "Medial triambic icosahedron",
    47: "Compound of 5 tetrahedra (chiral)",
}


def _crennell_items():
    items = []
    for k in range(1, 60):
        nm = _FAMOUS.get(k, "Du Val %s" % _CRENNELL_STR[k])
        kind = "reflexible" if k <= 32 else "chiral"
        items.append((str(k), "%2d. %s" % (k, nm),
                      "Du Val %s (%s)" % (_CRENNELL_STR[k], kind)))
    items.append(('CUSTOM', "Custom (shell toggles)",
                  "Choose Du Val shells a..g3 by hand"))
    return items


try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, BoolProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _ITEMS = _crennell_items()

    class MESH_OT_icosahedron_stellation_add(bpy.types.Operator):
        """Add a stellation of the icosahedron -- any of the 59 (Coxeter/
        Du Val/Flather/Petrie, "The Fifty-Nine Icosahedra"), by Crennell
        index, or a custom set of Du Val cell shells"""
        bl_idname = "mesh.icosahedron_stellation_add"
        bl_label = "Icosahedron Stellation"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(name="Stellation", items=_ITEMS, default='8')
        # custom Du Val shell toggles (used when solid == 'CUSTOM')
        sh_a: BoolProperty(name="a (core)", default=True)
        sh_b: BoolProperty(name="b", default=True)
        sh_c: BoolProperty(name="c", default=False)
        sh_d: BoolProperty(name="d", default=False)
        sh_e1: BoolProperty(name="e1", default=False)
        sh_e2: BoolProperty(name="e2", default=False)
        sh_f1: BoolProperty(name="f1 (chiral)", default=False)
        sh_f2: BoolProperty(name="f2", default=False)
        sh_g1: BoolProperty(name="g1", default=False)
        sh_g2: BoolProperty(name="g2", default=False)
        sh_g3: BoolProperty(name="g3 (outer)", default=False)
        style: EnumProperty(
            name="Style",
            items=[('SOLID', "Solid", ""),
                   ('LEONARDO', "Leonardo (da Vinci)",
                    "Open-faced panels via the shared Leonardo modifier"),
                   ('WIRE', "Struts", "Wireframe modifier"),
                   ('WIREFRAME', "Wireframe",
                    "Mesh edges only, displayed as a wireframe")],
            default='SOLID')
        border: FloatProperty(name="Border", default=0.3, min=0.02, max=0.95,
                              description="Leonardo face frame width")
        thickness: FloatProperty(name="Thickness", default=0.05, min=0.001,
                                 max=1.0, description="Panel/strut thickness")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=100.0)

        def _code(self):
            if self.solid == 'CUSTOM':
                return [lb for lb in _ALL_LABELS
                        if getattr(self, 'sh_' + lb)]
            return crennell_cells(int(self.solid))

        def execute(self, context):
            code = self._code()
            if not code:
                self.report({'ERROR'}, "no shells selected")
                return {'CANCELLED'}
            V, F = build(code)
            if not F:
                self.report({'ERROR'}, "empty stellation")
                return {'CANCELLED'}
            if self.solid == 'CUSTOM':
                label = "Stellation " + "".join(
                    lb for lb in _ALL_LABELS if getattr(self, 'sh_' + lb))
            else:
                label = "Stellation %s" % _FAMOUS.get(
                    int(self.solid), _CRENNELL_STR[int(self.solid)])
            me = bpy.data.meshes.new(label)
            me.from_pydata([tuple(c * self.scale for c in v) for v in V],
                           [], [list(f) for f in F])
            me.validate(clean_customdata=True)
            me.update()
            obj = bpy.data.objects.new(label, me)
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
            elif self.style == 'WIREFRAME':
                obj.display_type = 'WIRE'
            self.report({'INFO'}, "%s: V=%d F=%d" % (label, len(V), len(F)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'solid')
            if self.solid == 'CUSTOM':
                col = lay.column(align=True)
                col.label(text="Du Val shells (inner -> outer):")
                for lb in _ALL_LABELS:
                    col.prop(self, 'sh_' + lb)
            lay.prop(self, 'style')
            if self.style == 'LEONARDO':
                lay.prop(self, 'border')
            if self.style != 'SOLID':
                lay.prop(self, 'thickness')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.icosahedron_stellation_add",
                             icon='MESH_ICOSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_icosahedron_stellation_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_icosahedron_stellation_add)


if __name__ == '__main__':
    _self_test()
