# Exact coordinates for the classical solids, from their published constructions.
#
# The exact expression is the PRIMARY artifact here: the floating-point vertex
# table in a record is evaluated FROM these expressions, never the other way
# round. Nothing in this module fits a closed form to an approximation.
#
# Sources for the constructions (not for the numbers -- the numbers are
# computed here from the constructions):
#
# - Euclid, "Elements" XIII: the five regular solids.
# - H. S. M. Coxeter, "Regular Polytopes", 3rd ed., Dover (1973), ch. 3:
#   the golden-ratio coordinates for the icosahedron and dodecahedron.
# - J. Kepler, "Harmonices Mundi" (1619): the Archimedean solids as
#   truncations and rectifications of the regular solids.
# - L. Poinsot (1810) / A. Cauchy (1813): the four regular star polyhedra.
# - E. Catalan (1865): the Archimedean duals, obtained here by exact polar
#   reciprocation in the midsphere.
#
# The truncation parameter is derived, not tabulated. Cutting each vertex of a
# regular solid at a fraction t along every incident edge leaves the original
# edges with length (1 - 2t)L and creates vertex-figure edges of length t*d,
# where d is the distance between two adjacent neighbours of a vertex. Equating
# the two gives
#
#       t = L / (d + 2L)
#
# which is exact and yields 1/3 for the tetrahedron, octahedron and
# icosahedron, 1/(2+sqrt(2)) for the cube and 1/(2+phi) for the dodecahedron.
# Rectification is the same construction at t = 1/2.
#
# The snub cube and snub dodecahedron are deliberately absent: their
# coordinates are roots of an irreducible cubic and sextic respectively and
# have no radical form, so those records carry no exact vertex table.

import itertools
import math

import sympy as sp

PHI = (1 + sp.sqrt(5)) / 2
SQ2 = sp.sqrt(2)


# -- exact vertex patterns for the regular solids (edge length 1) -----------

def _signs(vec, mask=None):
    """All sign choices on the entries flagged in `mask` (default: nonzero)."""
    idx = [i for i in range(3) if (vec[i] != 0 if mask is None else mask[i])]
    out = []
    for combo in itertools.product(*[(1, -1)] * len(idx)):
        v = list(vec)
        for i, s in zip(idx, combo):
            v[i] = v[i] * s
        out.append(tuple(v))
    return out


def _cyclic(vec):
    a, b, c = vec
    return [(a, b, c), (c, a, b), (b, c, a)]


def _dedupe(vs):
    seen, out = set(), []
    for v in vs:
        k = tuple(sp.nsimplify(x) for x in v)
        ks = tuple(str(sp.simplify(x)) for x in k)
        if ks in seen:
            continue
        seen.add(ks)
        out.append(tuple(sp.simplify(x) for x in v))
    return out


def platonic_exact(name):
    """Exact vertices at edge length 1."""
    h = sp.Rational(1, 2)
    if name == "tetrahedron":
        s = sp.sqrt(2) / 4
        return [(s, s, s), (s, -s, -s), (-s, s, -s), (-s, -s, s)]
    if name == "cube":
        return _signs((h, h, h))
    if name == "octahedron":
        r = SQ2 / 2
        out = []
        for base in ((r, 0, 0), (0, r, 0), (0, 0, r)):
            out += _signs(base)
        return out
    if name == "icosahedron":
        out = []
        for base in _cyclic((sp.Integer(0), h, PHI / 2)):
            out += _signs(base)
        return _dedupe(out)
    if name == "dodecahedron":
        out = _signs((PHI / 2, PHI / 2, PHI / 2))
        for base in _cyclic((sp.Integer(0), h, (PHI + 1) / 2)):
            out += _signs(base)
        return _dedupe(out)
    raise KeyError(name)


# -- exact Archimedean patterns that are not simple truncations -------------

def archimedean_pattern(name):
    """Classical exact vertex sets at edge 1 for the cantellated and
    omnitruncated solids, which are not obtained by truncating alone."""
    h = sp.Rational(1, 2)
    if name == "rhombicuboctahedron":
        a = (1 + SQ2) / 2
        out = []
        for base in set(itertools.permutations((h, h, a))):
            out += _signs(base)
        return _dedupe(out)
    if name == "truncated cuboctahedron":
        a, b = (1 + SQ2) / 2, (1 + 2 * SQ2) / 2
        out = []
        for base in set(itertools.permutations((h, a, b))):
            out += _signs(base)
        return _dedupe(out)
    if name == "rhombicosidodecahedron":
        p = PHI
        out = []
        for base in _cyclic((h, h, (p ** 3) / 2)):
            out += _signs(base)
        for base in _cyclic(((p ** 2) / 2, p / 2, 2 * p / 2)):
            out += _signs(base)
        for base in _cyclic(((2 + p) / 2, sp.Integer(0), (p ** 2) / 2)):
            out += _signs(base)
        return _dedupe(out)
    if name == "truncated icosidodecahedron":
        p = PHI
        out = []
        for base in _cyclic((sp.Rational(1, 2) / p, sp.Rational(1, 2) / p,
                             (3 + p) / 2)):
            out += _signs(base)
        for base in _cyclic((sp.Rational(1, 1) / p, p / 2, (1 + 2 * p) / 2)):
            out += _signs(base)
        for base in _cyclic(((2 * p - 1) / 2, sp.Rational(1, 2), (2 + p) / 2)):
            out += _signs(base)
        for base in _cyclic((p / 2, sp.Integer(3) / 2, 2 * p / 2)):
            out += _signs(base)
        for base in _cyclic(((1 + p) / 2, sp.Integer(0), (3 * p - 1) / 2)):
            out += _signs(base)
        return _dedupe(out)
    return None


# -- exact truncation / rectification ---------------------------------------

def _edges_from(V, tol=1e-9):
    n = len(V)
    Vf = [[float(c) for c in v] for v in V]
    dmin = None
    for i in range(n):
        for j in range(i + 1, n):
            d = math.dist(Vf[i], Vf[j])
            dmin = d if dmin is None else min(dmin, d)
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if abs(math.dist(Vf[i], Vf[j]) - dmin) < tol], dmin


def truncate_exact(V, t):
    """Cut every vertex at exact fraction t along each incident edge.
    Returns the exact new vertex list and the (a, b) index pairs it came from."""
    E, L = _edges_from(V)
    adj = {}
    for i, j in E:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    out, origin = [], []
    for i in sorted(adj):
        for j in sorted(adj[i]):
            p = tuple(sp.simplify(V[i][k] + t * (V[j][k] - V[i][k]))
                      for k in range(3))
            out.append(p)
            origin.append((i, j))
    return out, origin


def rectify_exact(V):
    """t = 1/2: the edge midpoints, deduplicated."""
    E, _L = _edges_from(V)
    return [tuple(sp.simplify((V[i][k] + V[j][k]) / 2) for k in range(3))
            for i, j in E]


def truncation_parameter(V):
    """t = L / (d + 2L), exactly, for a regular solid given exactly."""
    E, L = _edges_from(V)
    adj = {}
    for i, j in E:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    i = min(adj)
    nb = sorted(adj[i])
    Vf = [[float(c) for c in v] for v in V]
    # adjacent neighbours of vertex i: the closest pair among its neighbours
    dd = min(math.dist(Vf[a], Vf[b])
             for a, b in itertools.combinations(nb, 2))
    # recover d exactly by matching the numeric value against exact candidates
    cands = {}
    for a, b in itertools.combinations(nb, 2):
        e = sp.simplify(sum((V[a][k] - V[b][k]) ** 2 for k in range(3)))
        cands[round(float(sp.sqrt(e)), 12)] = sp.sqrt(e)
    d = cands[round(dd, 12)]
    Lx = sp.simplify(sp.sqrt(sum((V[E[0][0]][k] - V[E[0][1]][k]) ** 2
                                 for k in range(3))))
    return sp.simplify(Lx / (d + 2 * Lx))


def scale_to_edge_one(V):
    """Rescale an exact vertex set so its shortest edge is exactly 1."""
    _E, L = _edges_from(V)
    Vf = [[float(c) for c in v] for v in V]
    pairs = [(i, j) for i in range(len(V)) for j in range(i + 1, len(V))
             if abs(math.dist(Vf[i], Vf[j]) - L) < 1e-9]
    i, j = pairs[0]
    Lx = sp.sqrt(sp.simplify(sum((V[i][k] - V[j][k]) ** 2 for k in range(3))))
    return [tuple(sp.radsimp(sp.simplify(c / Lx)) for c in v) for v in V]


# -- printing into the record's exact-expression language -------------------

def to_expr(x):
    """Render a sympy scalar into the schema's exact language."""
    e = sp.radsimp(sp.simplify(sp.nsimplify(x)))
    e = sp.sqrtdenest(e)
    s = sp.sstr(sp.simplify(e))
    s = s.replace(" ", "")
    # sympy writes powers; the language has none. Expand small ones.
    while "**" in s:
        s = _expand_pow(s)
    return s


def _expand_pow(s):
    i = s.index("**")
    j = i + 2
    k = j
    while k < len(s) and (s[k].isdigit()):
        k += 1
    exp = int(s[j:k])
    # find the base token
    if s[i - 1] == ")":
        depth = 0
        b = i - 1
        while b >= 0:
            if s[b] == ")":
                depth += 1
            elif s[b] == "(":
                depth -= 1
                if depth == 0:
                    break
            b -= 1
        base = s[b:i]
    else:
        b = i - 1
        while b >= 0 and (s[b].isalnum() or s[b] == "_"):
            b -= 1
        b += 1
        base = s[b:i]
    rep = "*".join([base] * exp)
    if exp > 1:
        rep = "(" + rep + ")"
    return s[:b] + rep + s[k:]
