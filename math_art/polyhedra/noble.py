# Noble polyhedra: the orbit-type method of Connor Hill (2026).
#
# Part of the Math Art polyhedra engine (`math_art/polyhedra/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests headlessly.
#
# A polyhedron is NOBLE when its symmetry group acts transitively on its
# vertices and on its faces.  Hess (1875 on) and Bruckner (1900-1907) found
# the first non-regular ones, Grunbaum named them, and Webb, Mikloweit and
# the Stella community added more after 2008 -- but whether the list was
# complete stayed open until Hill's enumeration: apart from the two
# infinite prismatic families (stephanoids and disphenoids) there are
# exactly 146 noble polyhedra up to similarity.
#
# The idea that makes the search finite, and the one this module
# implements, is to stop thinking about vertex SETS and think about ORBIT
# TYPES.  Every vertex-transitive vertex set is one orbit of a point group.
# Each point group is a normal subgroup of a reflection group *332, *432,
# *532 (or *22n), and a point p of that reflection group's fundamental
# domain is fixed by its distances (a, b, c) to the three mirrors:
#
#     p(a, b, c) = a u1 + b u2 + c u3,
#
# with u_i the unit-distance corner vectors of the domain.  Up to scale the
# last non-zero parameter is 1, so the orbits of one group fall into
# families with zero, one or two free parameters -- Hill's 23 non-prismatic
# ORBIT TYPES (T, O, C, I, ID, D; tT ... rD; sT ... gD).  Moving the
# parameters moves every vertex AFFINELY, so the canonical mapping between
# two members of a type is simply "same group element".
#
# A noble polyhedron is G(F): one polygon F on the orbit, swept round by a
# group G that is transitive on the vertices.  F lies on a plane through
# three or more orbit points, and each side of F must be the whole
# intersection of F's plane with the plane of a neighbouring face -- the
# ADJACENCY GRAPH of the plane.  Faces are cycles of that graph.
#
# Most orbits in a type carry nothing new.  A noble faceting appears only
# where four orbit points that are not coplanar in general BECOME coplanar
# -- the orbit is CRITICAL.  Four points p_0 .. p_3 are coplanar iff
#
#     det [p_i  1]  = 0,
#
# and since each p_i is affine in the parameters, that determinant is a
# polynomial of degree at most three in them.  For a one-parameter type
# the critical orbits are therefore just the positive real roots of a
# finite list of cubics: `critical_orbits()` derives them from scratch.
# Two-parameter types need the intersections of pairs of cubic CURVES,
# which Hill settled with a computer-algebra system; here the catalogue
# carries his locations (as exact minimal polynomials, polished
# numerically) and the engine re-facets them.
#
# References:
# - C. Hill, "The complete set of noble polyhedra", arXiv:2607.28711
#   (2026); code and models at github.com/Plasmath/noble-tools-revised
#   (GPL-3.0).  The orbit types, the adjacency-graph faceting and the
#   stephanoid definitions PC(n,p,q) and AC(n,p,q) follow this paper.
# - E. Hess, "Ueber die zugleich gleicheckigen und gleichflaechigen
#   Polyeder" (1876), and his papers of 1875 and 1877, for the first
#   non-regular noble polyhedra and the stephanoids.
# - M. Bruckner, "Vielecke und Vielflache: Theorie und Geschichte" (1900),
#   and "Ueber die gleicheckig-gleichflaechigen diskontinuierlichen und
#   nichtkonvexen Polyeder", Nova Acta Leopoldina (1906).
# - B. Grunbaum, "Polyhedra with hollow faces", in Polytopes: Abstract,
#   Convex and Computational (1994), 43-70, which named the noble
#   polyhedra.
# - U. Mikloweit, "Exploring Noble Polyhedra With the Program Stella4D",
#   Bridges 2020, 257-264.
# - J. H. Conway, H. Burgiel & C. Goodman-Strauss, "The Symmetries of
#   Things" (2008), for the orbifold names of the point groups.

import itertools
import math

import numpy as np


PHI = (1.0 + math.sqrt(5.0)) / 2.0
_R2 = math.sqrt(2.0)

# --------------------------------------------------------------------
# The three reflection groups, with Hill's generating mirrors (Defs.
# 3.11-3.13) and the corner vectors of their fundamental domains.  Using
# his frame exactly means his published coordinates and ours agree
# vertex for vertex, which is what the catalogue check relies on.
# --------------------------------------------------------------------
_M = lambda rows: np.array(rows, float)                     # noqa: E731

_R_TET = (_M([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
          _M([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
          _M([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))
_R_OCT = (_M([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), _R_TET[0], _R_TET[2])
_R_ICO = (_R_OCT[0], _M([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
          0.5 * _M([[1 - PHI, -PHI, 1], [-PHI, 1, PHI - 1],
                    [1, PHI - 1, PHI]]))

_U_TET = (_M([-_R2 / 2, _R2 / 2, _R2 / 2]), _M([_R2 / 2, _R2 / 2, _R2 / 2]),
          _M([0, 0, _R2]))
_U_OCT = (_M([1, 1, 1]), _M([0, _R2, _R2]), _M([0, 0, _R2]))
_U_ICO = (_M([1, 0, PHI + 1]), _M([0, 1, PHI]), _M([0, 0, 2 * PHI]))


def _gens(name):
    """Generating matrices of a point group in Hill's frame.

    The rotation groups are generated by products of pairs of mirrors;
    the pyritohedral group 3*2 by the x-mirror, the y-mirror and the
    cyclic permutation of the axes.
    """
    r = {'332': _R_TET, '*332': _R_TET, '432': _R_OCT, '*432': _R_OCT,
         '532': _R_ICO, '*532': _R_ICO}.get(name)
    if name.startswith('*'):
        return list(r)
    if name == '3*2':
        r1, r2, r3 = _R_OCT
        return [r1, r2 @ r1 @ r2, r3 @ r2]
    return [r[0] @ r[1], r[0] @ r[2], r[1] @ r[2]]


_GROUP_CACHE = {}


def group(name):
    """All elements of a point group, in a fixed breadth-first order.

    The order matters: it is what makes the vertex numbering of an orbit
    the same for every member of an orbit type (Hill's canonical map).
    """
    if name not in _GROUP_CACHE:
        gens = _gens(name)
        out = [np.eye(3)]
        seen = {_mkey(out[0])}
        frontier = [out[0]]
        while frontier:
            nxt = []
            for g in frontier:
                for s in gens:
                    h = s @ g
                    k = _mkey(h)
                    if k not in seen:
                        seen.add(k)
                        out.append(h)
                        nxt.append(h)
            frontier = nxt
        _GROUP_CACHE[name] = out
    return _GROUP_CACHE[name]


def _mkey(m):
    return tuple(np.round(m, 8).ravel() + 0.0)


#: order of each group -- a cheap guard on the closure above
GROUP_ORDER = {'332': 12, '*332': 24, '3*2': 24, '432': 24, '*432': 48,
               '532': 60, '*532': 120}

#: groups that can act on an orbit, strongest first.  All live in one
#: frame: the tetrahedral and pyritohedral groups here are subgroups of
#: the octahedral ones, and 332 and 3*2 are ALSO subgroups of *532 in
#: this frame (its mirrors are the coordinate planes).
GROUPS = ('*532', '532', '*432', '432', '3*2', '*332', '332')

# --------------------------------------------------------------------
# Hill's 23 non-prismatic orbit types (Table 1)
# symbol -> (orbit group, domain, parameter pattern, vertex count)
# 'a' and 'b' in the pattern are the free parameters.
# --------------------------------------------------------------------
ORBIT_TYPES = {
    'T':  ('*332', 'tet', (1, 0, 0), 4),
    'O':  ('*432', 'oct', (0, 0, 1), 6),
    'CO': ('*432', 'oct', (0, 1, 0), 12),
    'C':  ('*432', 'oct', (1, 0, 0), 8),
    'I':  ('*532', 'ico', (0, 1, 0), 12),
    'ID': ('*532', 'ico', (0, 0, 1), 30),
    'D':  ('*532', 'ico', (1, 0, 0), 20),
    'tT': ('*332', 'tet', (0, 'a', 1), 12),
    'rT': ('*332', 'tet', ('a', 1, 0), 12),
    'rP': ('3*2', 'oct', (0, 'a', 1), 12),
    'tO': ('*432', 'oct', (0, 'a', 1), 24),
    'tC': ('*432', 'oct', ('a', 1, 0), 24),
    'rC': ('*432', 'oct', ('a', 0, 1), 24),
    'tI': ('*532', 'ico', (0, 'a', 1), 60),
    'tD': ('*532', 'ico', ('a', 0, 1), 60),
    'rD': ('*532', 'ico', ('a', 1, 0), 60),
    'sT': ('332', 'tet', ('a', 'b', 1), 12),
    'gT': ('*332', 'tet', ('a', 'b', 1), 24),
    'gP': ('3*2', 'oct', ('a', 'b', 1), 24),
    'sC': ('432', 'oct', ('a', 'b', 1), 24),
    'gC': ('*432', 'oct', ('a', 'b', 1), 48),
    'sD': ('532', 'ico', ('a', 'b', 1), 60),
    'gD': ('*532', 'ico', ('a', 'b', 1), 120),
}

_BASIS = {'tet': _U_TET, 'oct': _U_OCT, 'ico': _U_ICO}


def dof(otype):
    """Number of free parameters of an orbit type (0, 1 or 2)."""
    return sum(1 for c in ORBIT_TYPES[otype][2] if isinstance(c, str))


def generating_point(otype, a=1.0, b=1.0):
    """The orbit's representative point a u1 + b u2 + c u3."""
    _g, dom, pat, _n = ORBIT_TYPES[otype]
    val = {'a': a, 'b': b}
    coef = [val[c] if isinstance(c, str) else float(c) for c in pat]
    U = _BASIS[dom]
    return coef[0] * U[0] + coef[1] * U[1] + coef[2] * U[2]


_ELT_CACHE = {}


def orbit_elements(otype):
    """For each vertex of the orbit, the group element that makes it.

    Chosen at a generic location so no two elements collide, then reused
    for every location: that IS Hill's canonical mapping, and it is why a
    vertex keeps its index as the parameters move.
    """
    if otype not in _ELT_CACHE:
        gname, _dom, _pat, n = ORBIT_TYPES[otype]
        p = generating_point(otype, 0.3141592653589793, 0.2718281828459045)
        out, seen = [], set()
        for g in group(gname):
            k = _vkey(g @ p, 1.0)
            if k not in seen:
                seen.add(k)
                out.append(g)
        assert len(out) == n, (otype, len(out), n)
        _ELT_CACHE[otype] = out
    return _ELT_CACHE[otype]


def orbit(otype, a=1.0, b=1.0):
    """The vertices of the orbit T(a, b), as an (n, 3) array.

    Vertex 0 is the generating point.  The numbering is the same for
    every location in the type.
    """
    p = generating_point(otype, a, b)
    return np.array([g @ p for g in orbit_elements(otype)])


def _vkey(v, scale, nd=7):
    return tuple(np.round(np.asarray(v, float) / scale, nd) + 0.0)


# --------------------------------------------------------------------
# Groups acting on a concrete vertex set, as permutations
# --------------------------------------------------------------------
def permutations(V, gname, tol=1e-7):
    """The group as permutations of V, or None if it does not preserve V."""
    V = np.asarray(V, float)
    scale = float(np.max(np.linalg.norm(V, axis=1))) or 1.0
    nd = max(3, int(-math.log10(tol)) - 1)
    idx = {_vkey(v, scale, nd): i for i, v in enumerate(V)}
    perms = []
    for g in group(gname):
        W = V @ g.T
        p = []
        for w in W:
            j = idx.get(_vkey(w, scale, nd))
            if j is None:
                return None
            p.append(j)
        perms.append(tuple(p))
    return sorted(set(perms), key=perms.index)


def transitive_groups(V):
    """[(name, perms)] of every group in GROUPS that preserves V and is
    transitive on it, strongest first."""
    out = []
    for name in GROUPS:
        P = permutations(V, name)
        if P is None:
            continue
        if len({p[0] for p in P}) == len(V):
            out.append((name, P))
    return out


# --------------------------------------------------------------------
# Faceting (Hill Section 3.1)
# --------------------------------------------------------------------
def _edges(face):
    n = len(face)
    return tuple(sorted(tuple(sorted((face[i], face[(i + 1) % n])))
                        for i in range(n)))


def planes_through_zero(V, tol=1e-7):
    """Every plane through vertex 0 and two others, as sorted index
    tuples of ALL the orbit points on it."""
    V = np.asarray(V, float)
    scale = float(np.max(np.linalg.norm(V, axis=1))) or 1.0
    n = len(V)
    D = V - V[0]
    out = set()
    for i in range(1, n):
        C = np.cross(D[i], D[i + 1:])
        L = np.linalg.norm(C, axis=1)
        for jj in np.nonzero(L > 1e-9 * scale * scale)[0]:
            c = C[jj] / L[jj]
            on = np.nonzero(np.abs(D @ c) < tol * scale)[0]
            if len(on) >= 3:
                out.add(tuple(int(k) for k in on))
    return sorted(out)


def _apply(perm, seq):
    return tuple(perm[i] for i in seq)


def face_orbit(face, perms):
    """The distinct images of a face under the group, deduped by edge
    set (a face met twice, or backwards, counts once)."""
    seen = {}
    for p in perms:
        f = _apply(p, face)
        seen.setdefault(_edges(f), f)
    return list(seen.values())


def is_polyhedron(faces, n):
    """Hill's validity test, plus the abstract-polyhedron checks.

    Every edge in exactly two faces; all n vertices used; the faces
    connected through shared edges (else it is a compound); no two
    faces on the same vertex set (a faithful realisation); and each
    vertex figure a single cycle (the section above a vertex connected).
    """
    mult = {}
    for f in faces:
        for e in _edges(f):
            mult[e] = mult.get(e, 0) + 1
    if not mult or set(mult.values()) != {2}:
        return False
    if len({i for f in faces for i in f}) != n:
        return False
    if len({frozenset(f) for f in faces}) != len(faces):
        return False
    # connected through edges
    by_edge = {}
    for k, f in enumerate(faces):
        for e in _edges(f):
            by_edge.setdefault(e, []).append(k)
    adj = [set() for _ in faces]
    for a, b in by_edge.values():
        adj[a].add(b)
        adj[b].add(a)
    seen, stack = {0}, [0]
    while stack:
        for j in adj[stack.pop()]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    if len(seen) != len(faces):
        return False
    return all(len(c) == 1 for c in vertex_figures(faces, n).values())


def coplanar_neighbours(V, faces, tol=1e-7):
    """True if two faces sharing an edge lie in one plane.

    Hill's Definition 2.3 forbids it (the two would merge into one
    face).  The adjacency graph does not rule it out by itself: where a
    plane holds two faces of the orbit -- the planes of the fissary
    orbits gD-19 and gD-28 hold two pentagons each -- a cycle can weave
    between them and meet its own image along an edge in the same plane.
    """
    V = np.asarray(V, float)
    scale = float(np.max(np.linalg.norm(V, axis=1))) or 1.0
    planes = []
    for f in faces:
        P = V[list(f)]
        c = P.mean(axis=0)
        _u, _s, vt = np.linalg.svd(P - c)
        nrm = vt[-1]
        d = float(nrm @ c)
        if d < 0:
            nrm, d = -nrm, -d
        planes.append((nrm, d))
    by_edge = {}
    for k, f in enumerate(faces):
        for e in _edges(f):
            by_edge.setdefault(e, []).append(k)
    for ks in by_edge.values():
        (n1, d1), (n2, d2) = planes[ks[0]], planes[ks[1]]
        if np.linalg.norm(n1 - n2) < tol and abs(d1 - d2) < tol * scale:
            return True
    return False


def vertex_figures(faces, n):
    """{vertex: [cycle of face indices]} -- one cycle per connected piece
    of the vertex figure; a polyhedron has exactly one per vertex."""
    around = {v: {} for v in range(n)}
    for k, f in enumerate(faces):
        m = len(f)
        for i, v in enumerate(f):
            u, w = f[i - 1], f[(i + 1) % m]
            around[v][k] = (u, w)
    out = {}
    for v, inc in around.items():
        left = set(inc)
        cycles = []
        while left:
            k0 = min(left)
            cyc = [k0]
            left.discard(k0)
            nb = inc[k0][1]
            k = k0
            while True:
                nxt = [j for j in left if nb in inc[j]]
                if not nxt:
                    break
                k = nxt[0]
                left.discard(k)
                cyc.append(k)
                u, w = inc[k]
                nb = w if u == nb else u
            cycles.append(cyc)
        out[v] = cycles
    return out


def _simple_cycles(adj, min_len=3):
    """Every simple cycle of the plane's adjacency graph, once each.

    Each cycle is grown from its SMALLEST vertex and kept in one of its
    two directions.  Hill grows cycles from vertex 0 only, which is
    enough when the plane kept for a class of planes has its face through
    vertex 0 -- but a face need not use every point of its plane, and on
    the representative plane the matching face can miss vertex 0
    altogether (sD-5.2 is the case: a pentagon on six coplanar points).
    """
    out = []
    for s in sorted(adj):
        partial = [[s]]
        while partial:
            nxt = []
            for p in partial:
                for v in adj[p[-1]]:
                    if v == s and len(p) >= min_len:
                        if p[1] < p[-1]:
                            out.append(p)
                    elif v > s and v not in p:
                        nxt.append(p + [v])
            partial = nxt
    return out


def _through_zero(face, perms):
    """An image of the face that passes through vertex 0, starting there."""
    for p in perms:
        f = _apply(p, face)
        if 0 in f:
            k = f.index(0)
            return f[k:] + f[:k]
    return tuple(face)


def facet(V, perms, planes=None, max_face=24):
    """Noble facetings of the vertex set V under one transitive group.

    Returns a list of generating faces (index cycles through vertex 0);
    `face_orbit(face, perms)` gives the whole polyhedron.
    """
    n = len(V)
    if planes is None:
        planes = planes_through_zero(V)
    # one plane per orbit of planes-through-0 under the group
    done, reps = set(), []
    for pl in planes:
        if pl in done:
            continue
        reps.append(pl)
        for p in perms:
            q = tuple(sorted(p[i] for i in pl))
            if 0 in q:
                done.add(q)
    out, sigs = [], set()
    for pl in reps:
        if len(pl) > max_face:
            continue
        S = set(pl)
        graph = set()
        for p in perms:
            inter = S.intersection(p[i] for i in pl)
            if len(inter) == 2:
                graph.add(tuple(sorted(inter)))
        adj = {i: {j for j in pl if tuple(sorted((i, j))) in graph}
               for i in pl}
        for cyc in _simple_cycles(adj):
            faces = face_orbit(cyc, perms)
            sig = frozenset(_edges(f) for f in faces)
            if sig in sigs:
                continue
            if is_polyhedron(faces, n) and not coplanar_neighbours(V, faces):
                sigs.add(sig)
                out.append(_through_zero(tuple(cyc), perms))
    return out


def noble_facetings(V):
    """Every noble faceting of V under every transitive group.

    [(group name, face)], strongest group first; a face set reached
    under a subgroup as well is reported once, under the larger group
    (its true symmetry is at least that).
    """
    V = np.asarray(V, float)
    planes = planes_through_zero(V)
    groups = transitive_groups(V)
    if not groups:
        return []
    # Two facetings the vertex set's own symmetry carries one onto the
    # other are the same polyhedron up to similarity -- in particular a
    # chiral faceting and its mirror image, which a subgroup such as 532
    # finds as two separate face sets on the dodecahedron's vertices.
    full = groups[0][1]
    out, sigs = [], set()
    for name, perms in groups:
        for face in facet(V, perms, planes):
            faces = face_orbit(face, perms)
            sig = frozenset(_edges(f) for f in faces)
            if sig in sigs:
                continue
            for p in full:
                sigs.add(frozenset(_edges(_apply(p, f)) for f in faces))
            out.append((name, face))
    return out


_NATIVE = {}


def native_groups(otype):
    """The groups transitive on a GENERIC member of an orbit type."""
    if otype not in _NATIVE:
        V = orbit(otype, 0.3141592653589793, 0.2718281828459045)
        _NATIVE[otype] = tuple(n for n, _p in transitive_groups(V))
    return _NATIVE[otype]


def type_facetings(otype, a=1.0, b=1.0):
    """Noble facetings of T(a, b) that BELONG to the type T.

    At some locations an orbit gains symmetry and becomes a member of a
    more symmetric type -- rP at a = phi is the icosahedron's twelve
    vertices.  Its facetings are then counted under that type, as Hill
    counts them, and this returns [] with the reason in `coincides()`.
    """
    V = orbit(otype, a, b)
    if coincides(otype, a, b):
        return []
    return noble_facetings(V)


def coincides(otype, a=1.0, b=1.0):
    """The stronger group this orbit has beyond its type's, or None."""
    V = orbit(otype, a, b)
    tg = transitive_groups(V)
    extra = [n for n, _p in tg if n not in native_groups(otype)]
    return extra[0] if extra else None


def build_faces(V, gname, face):
    """The whole face list of G(F)."""
    return [list(f) for f in face_orbit(tuple(face), permutations(V, gname))]


# --------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------
def inradius(V, faces):
    """Distance from the centre to the face planes (all equal: noble)."""
    V = np.asarray(V, float)
    f = faces[0]
    P = V[list(f)]
    c = P.mean(axis=0)
    # best-fit plane normal
    _u, _s, vt = np.linalg.svd(P - c)
    nrm = vt[-1]
    return abs(float(nrm @ c))


def describe(V, faces):
    """dict: V, E, F, chi, p (face sides), q (vertex-figure length),
    in/circumradius ratio."""
    E = set()
    for f in faces:
        E.update(_edges(f))
    n = len(V)
    q = 2 * len(E) // n
    R = float(np.max(np.linalg.norm(np.asarray(V, float), axis=1)))
    return {'V': n, 'E': len(E), 'F': len(faces),
            'chi': n - len(E) + len(faces), 'p': len(faces[0]), 'q': q,
            'ratio': inradius(V, faces) / R}


def fit(V, extent=1.0):
    """Centre (they are already centred) and scale into [-extent, extent]^3."""
    V = np.asarray(V, float)
    m = float(np.max(np.abs(V))) or 1.0
    return V * (extent / m)


def convex_hull(V, tol=1e-7):
    """Faces of the convex hull of an orbit, as ordered index cycles.

    Shown by the orbit explorer when an orbit carries no noble faceting,
    so the vertex set is still visible (Hill draws his orbit types the
    same way).  Vectorised over all triples; fine for up to 120 points.
    """
    V = np.asarray(V, float)
    n = len(V)
    scale = float(np.max(np.linalg.norm(V, axis=1))) or 1.0
    T = np.array(list(itertools.combinations(range(n), 3)))
    Nm = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    L = np.linalg.norm(Nm, axis=1)
    ok = L > 1e-9 * scale * scale
    T, Nm = T[ok], Nm[ok] / L[ok, None]
    d = np.einsum('ij,ij->i', Nm, V[T[:, 0]])
    flip = d < 0
    Nm[flip] *= -1
    d[flip] *= -1
    out, seen = [], set()
    for k in range(0, len(T), 4096):
        P = Nm[k:k + 4096] @ V.T - d[k:k + 4096, None]
        for r in np.nonzero(np.all(P <= tol * scale, axis=1))[0]:
            nrm, dd = Nm[k + r], d[k + r]
            key = _vkey(np.append(nrm, dd / scale), 1.0, 5)
            if key in seen:
                continue
            seen.add(key)
            on = np.nonzero(np.abs(V @ nrm - dd) < tol * scale)[0]
            c = V[on].mean(axis=0)
            u = V[on[0]] - c
            u /= np.linalg.norm(u)
            w = np.cross(nrm, u)
            ang = [math.atan2(float((V[i] - c) @ w), float((V[i] - c) @ u))
                   for i in on]
            out.append([int(on[i]) for i in np.argsort(ang)])
    return out


# --------------------------------------------------------------------
# Critical orbits of a one-parameter type (Hill Section 3.3)
# --------------------------------------------------------------------
def coplanarity_cubics(otype):
    """Coefficients (c0, c1, c2, c3) of det[p0 p_i p_j p_k ; 1] as a cubic
    in a, for every triple with vertex 0 -- one row per triple.

    Each p_i(a) = g_i (a u + w) is affine in a, so the 4x4 determinant
    is a cubic.  It is sampled at four values of a and interpolated,
    which is exact up to rounding.
    """
    assert dof(otype) == 1
    ts = np.array([0.5, 1.0, 1.5, 2.0])
    n = ORBIT_TYPES[otype][3]
    trip = np.array(list(itertools.combinations(range(1, n), 3)))
    vals = []
    for t in ts:
        V = orbit(otype, t)
        D = V[trip] - V[0]                      # (m, 3, 3)
        vals.append(np.linalg.det(D))
    Y = np.array(vals)                          # (4, m)
    A = np.vander(ts, 4, increasing=True)
    return trip, np.linalg.solve(A, Y).T        # (m, 4)


def critical_orbits(otype, a_max=64.0, tol=1e-9):
    """The critical locations of a one-parameter orbit type, derived.

    These are the positive real roots of the coplanarity cubics that
    are not identically zero -- the parameter values where four orbit
    points that are in general position become coplanar.  Only there
    can a noble polyhedron exist beyond those of the generic orbit.
    """
    _trip, C = coplanarity_cubics(otype)
    scale = np.max(np.abs(C), axis=1)
    keep = scale > 1e-9 * float(np.max(scale))
    C = C[keep] / scale[keep, None]
    C[np.abs(C) < 1e-11] = 0.0
    polys = {tuple(np.round(r, 9) + 0.0) for r in C}
    roots = []
    for c in polys:
        r = np.roots(c[::-1]) if any(c[1:]) else []
        for z in r:
            if abs(z.imag) < 1e-7 and 1e-9 < z.real < a_max:
                roots.append(float(z.real))
    roots.sort()
    out = []
    for z in roots:
        if not out or z - out[-1] > 1e-7 * max(1.0, z):
            out.append(z)
    return out


# --------------------------------------------------------------------
# Locations from minimal polynomials
# --------------------------------------------------------------------
def polish_root(coeffs, approx):
    """The real root of the integer polynomial nearest `approx`,
    Newton-polished.  `coeffs` is highest degree first."""
    c = np.array(coeffs, float)
    r = np.roots(c)
    r = [z.real for z in r if abs(z.imag) < 1e-6]
    x = min(r, key=lambda z: abs(z - approx))
    d = np.polyder(c)
    for _ in range(8):
        fx, dx = np.polyval(c, x), np.polyval(d, x)
        if dx == 0:
            break
        x -= fx / dx
    return float(x)


# --------------------------------------------------------------------
# Duals (Hill Theorem 2.6)
# --------------------------------------------------------------------
def dual(V, faces):
    """Polar dual in the circumsphere: face planes become vertices, each
    vertex's cycle of faces becomes a face.

    A face at distance d along unit normal n reciprocates to the point
    R^2 n / d.  Where a vertex figure is not one cycle (Hill's FISSARY
    case) each cycle becomes its own face, and faces that were coplanar
    reciprocate to coinciding vertices -- kept distinct, as the
    degenerate polyhedron he describes.
    """
    V = np.asarray(V, float)
    R = float(np.max(np.linalg.norm(V, axis=1)))
    DV = []
    for f in faces:
        P = V[list(f)]
        c = P.mean(axis=0)
        _u, _s, vt = np.linalg.svd(P - c)
        nrm = vt[-1]
        d = float(nrm @ c)
        if d < 0:
            nrm, d = -nrm, -d
        DV.append(nrm * (R * R / d) if d > 1e-12 else nrm * 1e6)
    DF = []
    for v, cycles in sorted(vertex_figures(faces, len(V)).items()):
        for cyc in cycles:
            DF.append(list(cyc))
    return np.array(DV), DF


# --------------------------------------------------------------------
# The prismatic families (Hill Section 4)
# --------------------------------------------------------------------
def stephanoid(n, p, q, kind='P', height=1.0):
    """Hess's crown polyhedra, in Hill's form.

    kind 'P' -- prismatic PC(n, p, q), 2p - n < 2q < p < n: two regular
      n-gons a_1..a_n (top) and b_1..b_n (bottom), aligned, and the
      crossed quadrilateral a_1, b_{1+q}, a_{1+p}, b_{1+p-q} under *22n.
    kind 'A' -- antiprismatic AC(n, p, q), q odd, 2p - n < q < p < n:
      the 2n antiprism vertices a_1..a_2n in angular order, alternately
      up and down, and the quadrilateral a_1, a_{1+q}, a_{1+2p},
      a_{1+2p-q} under 2*n.

    Returns (V, F).  If n, p, q share a factor the orbit of the face is
    a compound; that is reported by `is_polyhedron` failing.
    """
    h = 0.5 * height
    if kind == 'P':
        V = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n),
              h) for i in range(n)]
        V += [(x, y, -h) for x, y, _z in V]
        A = lambda i: (i - 1) % n                     # noqa: E731
        B = lambda i: n + (i - 1) % n                 # noqa: E731
        base = [A(1), B(1 + q), A(1 + p), B(1 + p - q)]
        # *22n acting on the labels: rotation i -> i+1, the top/bottom
        # swap, and the reflection i -> -i
        def sym(f, k, flip, refl):
            out = []
            for v in f:
                top = v < n
                i = v % n
                i = (-i if refl else i) + k
                if flip:
                    top = not top
                out.append((i % n) if top else n + i % n)
            return out
        F, seen = [], set()
        for k in range(n):
            for flip in (False, True):
                for refl in (False, True):
                    f = sym(base, k, flip, refl)
                    e = _edges(f)
                    if e not in seen:
                        seen.add(e)
                        F.append(f)
        return V, F
    m = 2 * n
    V = [(math.cos(math.pi * i / n), math.sin(math.pi * i / n),
          h if i % 2 == 0 else -h) for i in range(m)]
    a = lambda i: (i - 1) % m                         # noqa: E731
    base = [a(1), a(1 + q), a(1 + 2 * p), a(1 + 2 * p - q)]
    # 2*n: rotation by two steps, and the half-turn-with-flip i -> 1 - i
    # composed with reflection -- on labels, i -> -i shifted by an odd
    # number maps up to down.
    F, seen = [], set()
    for k in range(0, m, 2):
        for refl in (False, True):
            f = [((-v if refl else v) + k + (1 if refl else 0)) % m
                 for v in base]
            e = _edges(f)
            if e not in seen:
                seen.add(e)
                F.append(f)
    return V, F


def disphenoid(x, y, z):
    """A disphenoid: four congruent triangles, vertices at the even sign
    changes of (x, y, z).  Tetragonal when two of x, y, z agree,
    rhombic when all three differ, regular when all agree."""
    V = [(x, y, z), (x, -y, -z), (-x, y, -z), (-x, -y, z)]
    F = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
    return V, F


def stephanoid_ok(n, p, q, kind):
    """Hill's inequalities for PC / AC, and no common factor."""
    if kind == 'P':
        # With n and p both even (q then odd) the face's vertices keep to
        # one parity class -- a_odd with b_even, or a_even with b_odd --
        # and the orbit splits into two crowns on half the vertices each.
        # That is a compound although n, p, q share no factor, so it is
        # excluded here as well (PC(8, 4, 1) is the first case).
        ok = 2 * p - n < 2 * q < p < n and q >= 1             and not (n % 2 == 0 and p % 2 == 0)
    else:
        ok = q % 2 == 1 and 2 * p - n < q < p < n and q >= 1
    return ok and math.gcd(math.gcd(n, p), q) == 1


# --------------------------------------------------------------------
def _planar(V, F, tol=1e-9):
    V = np.asarray(V, float)
    for f in F:
        P = V[list(f)]
        c = P.mean(axis=0)
        s = np.linalg.svd(P - c, compute_uv=False)
        if s[-1] > tol * (1 + s[0]):
            return False
    return True


def _selftest():
    # the groups close up at the right orders
    for name, order in GROUP_ORDER.items():
        assert len(group(name)) == order, (name, len(group(name)))
    # every orbit type has the advertised vertex count (checked inside)
    for t in ORBIT_TYPES:
        orbit(t, 0.7, 0.4)

    # Hill's frame: the tI orbit matches his published coordinates
    a = 0.37
    V = orbit('tI', a)
    assert np.allclose(V[0], [0, a, a * PHI + 2 * PHI]), V[0]

    # The zero-parameter orbits reproduce his counts: T 1, O 1, C 1,
    # I 4, ID 6, D 7, CO none.
    want = {'T': 1, 'O': 1, 'C': 1, 'I': 4, 'ID': 6, 'D': 7, 'CO': 0}
    for t, k in want.items():
        got = noble_facetings(orbit(t))
        assert len(got) == k, (t, len(got), k)
        for g, f in got:
            F = build_faces(orbit(t), g, f)
            assert _planar(orbit(t), F), (t, f)
    print('0-DOF orbit types: %s' % want)

    # one-parameter derivation: the critical orbits of tO include Hill's
    # tO-1 at the real root of a^3 - a^2 - 2a - 1, and faceting there
    # finds exactly his one noble polyhedron
    crit = critical_orbits('tO')
    a1 = polish_root([1, -1, -2, -1], 2.1479)
    assert any(abs(c - a1) < 1e-7 for c in crit), (a1, crit)
    hits = [c for c in crit if noble_facetings(orbit('tO', c))]
    assert len(hits) == 1 and abs(hits[0] - a1) < 1e-7, hits
    print('tO: %d critical orbits, noble facetings only at a = %.12f'
          % (len(crit), hits[0]))

    # Hill's one-parameter counts, re-derived with no input from his
    # data: every critical orbit found from the coplanarity cubics, plus
    # one generic orbit, faceted under every transitive group.  (tI and
    # rD agree too -- 17 and 19 -- but take a minute; the catalogue
    # build checks them.)
    for t, want in (('tT', 0), ('rT', 0), ('rP', 0), ('tC', 1), ('rC', 1),
                    ('tD', 6)):
        got = sum(len(type_facetings(t, a))
                  for a in critical_orbits(t) + [0.123456789])
        assert got == want, (t, got, want)
    print('1-DOF derivation: tT 0, rT 0, rP 0, tO 1, tC 1, rC 1, tD 6')

    # duality: the icosahedron and dodecahedron swap
    V = orbit('I')
    got = noble_facetings(V)
    ico = [build_faces(V, g, f) for g, f in got
           if len(f) == 3 and describe(V, build_faces(V, g, f))['chi'] == 2]
    DV, DF = dual(V, ico[0])
    assert len(DV) == 20 and len(DF) == 12 and {len(f) for f in DF} == {5}
    assert _planar(DV, DF)

    # the prismatic families
    # every admissible crown up to n = 16 is a planar, connected noble
    # polyhedron with 2n crossed quadrilaterals -- including Hill's two
    # figure examples PC(5, 3, 1) and AC(5, 2, 1)
    assert stephanoid_ok(5, 3, 1, 'P') and stephanoid_ok(5, 2, 1, 'A')
    assert not stephanoid_ok(8, 4, 1, 'P')        # the parity compound
    count = 0
    for kind in ('P', 'A'):
        for n in range(3, 17):
            for p, q in itertools.product(range(1, n), repeat=2):
                if not stephanoid_ok(n, p, q, kind):
                    continue
                Vs, Fs = stephanoid(n, p, q, kind)
                assert _planar(Vs, Fs), (kind, n, p, q)
                assert is_polyhedron(Fs, len(Vs)), (kind, n, p, q)
                assert len(Fs) == 2 * n and                     {len(f) for f in Fs} == {4}, (kind, n, p, q)
                count += 1
    print('crowns: %d admissible PC/AC up to n = 16, all valid' % count)
    Vd, Fd = disphenoid(1.0, 0.7, 0.4)
    assert is_polyhedron(Fd, 4)
    L = sorted(round(math.dist(Vd[i], Vd[j]), 9)
               for i, j in itertools.combinations(range(4), 2))
    assert L[0] == L[1] and L[2] == L[3] and L[4] == L[5], L
    print('RESULT: OK')
