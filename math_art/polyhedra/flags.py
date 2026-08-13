# The flag complex of a polyhedron, and Conway operators as flag rewrites.
#
# Part of the Math Art polyhedron engine (`math_art/polyhedra/`).  Python
# and numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly.
#
# A FLAG is a triple (vertex, edge, face) that are mutually incident: a
# corner of a face, together with one of the two edges meeting there.  A
# polyhedron with E edges has exactly 4E flags, because every edge carries
# two vertices and two faces.
#
# Three involutions act on the flags.  Each changes exactly ONE element of
# the triple and keeps the other two:
#
#     s0  the other VERTEX on the same edge, same face
#     s1  the other EDGE at the same vertex, same face
#     s2  the same vertex and edge, in the OTHER face
#
# They generate the connection (flag) group, and they satisfy
#
#     s0^2 = s1^2 = s2^2 = identity        (each is an involution)
#     (s0 s2)^2 = identity                 (s0 and s2 commute)
#
# The second relation is the combinatorial statement that vertex and face
# are "far apart" in the incidence chain, and it is what makes DUALITY a
# relabelling: exchanging s0 with s2 preserves every relation, so it maps
# a polyhedron to its dual.  Every Conway operator is likewise a rule for
# grouping flags into the faces of a new polyhedron, which is why they all
# come from one mechanism instead of eight bespoke constructions.
#
# Geometry rides along by BARYCENTRIC SUBDIVISION: each flag names one
# triangle (its vertex, its edge midpoint, its face centroid), so an
# operator that says which of those points survive, and how flags group,
# determines the output mesh completely.
#
# References:
#   John H. Conway, Heidi Burgiel and Chaim Goodman-Strauss, "The
#     Symmetries of Things" (2008), chapters 20-21 -- flags, the
#     connection group, and operators as flag rewrites.
#   George W. Hart, "Conway Notation for Polyhedra" (georgehart.com) and
#     "Sculpture based on Propellorized Polyhedra", Proceedings of
#     MOSAIC 2000 -- the operator vocabulary implemented here.
#   Egon Schulte and Peter McMullen, "Abstract Regular Polytopes" (2002)
#     -- flags and the flag graph in their general form.
#   A. M. Brondsted, "An Introduction to Convex Polytopes" (1983) for the
#     face lattice the flags enumerate.

import math

import numpy as np


class FlagComplex:
    """The flags of a polygonal mesh, with the three involutions.

    Built from `(V, F)` where `F` lists each face's vertex indices in
    order.  The mesh must be a closed orientable surface -- every edge
    shared by exactly two faces -- which is what makes `s2` total.
    """

    __slots__ = ('V', 'F', 'flags', 's0', 's1', 's2', 'edges', '_index')

    def __init__(self, V, F):
        self.V = [tuple(float(c) for c in v) for v in V]
        self.F = [list(int(i) for i in f) for f in F]

        # a flag is (face, corner, side): corner c of face f, taken with
        # the edge leaving c (side 0) or arriving at c (side 1).  That is
        # the same thing as (vertex, edge, face) but indexes directly.
        self.flags = []
        self._index = {}
        for fi, face in enumerate(self.F):
            n = len(face)
            for c in range(n):
                for side in (0, 1):
                    self._index[(fi, c, side)] = len(self.flags)
                    self.flags.append((fi, c, side))

        # edge -> the (up to two) faces carrying it
        self.edges = {}
        for fi, face in enumerate(self.F):
            n = len(face)
            for c in range(n):
                a, b = face[c], face[(c + 1) % n]
                self.edges.setdefault((a, b) if a < b else (b, a),
                                      []).append((fi, c))

        m = len(self.flags)
        self.s0 = [0] * m
        self.s1 = [0] * m
        self.s2 = [0] * m
        for k, (fi, c, side) in enumerate(self.flags):
            n = len(self.F[fi])
            # s0: same edge and face, the other end of the edge.  The edge
            # of (f, c, 0) is c -> c+1 and of (f, c, 1) is c-1 -> c, so
            # swapping the vertex means moving along that same edge.
            if side == 0:
                self.s0[k] = self._index[(fi, (c + 1) % n, 1)]
            else:
                self.s0[k] = self._index[(fi, (c - 1) % n, 0)]
            # s1: same vertex and face, the other edge at that corner
            self.s1[k] = self._index[(fi, c, 1 - side)]
            # s2: same vertex and edge, the other face on that edge
            if side == 0:
                a, b = self.F[fi][c], self.F[fi][(c + 1) % n]
            else:
                a, b = self.F[fi][(c - 1) % n], self.F[fi][c]
            key = (a, b) if a < b else (b, a)
            pair = self.edges.get(key, [])
            other = [(g, cc) for (g, cc) in pair if g != fi]
            if not other:
                self.s2[k] = k                      # boundary edge: fixed
            else:
                g, cc = other[0]
                gn = len(self.F[g])
                # in face g the same edge runs the other way round
                if self.F[g][cc] == self.vertex_of(k):
                    self.s2[k] = self._index[(g, cc, 0)]
                else:
                    self.s2[k] = self._index[(g, (cc + 1) % gn, 1)]

    # -- what a flag points at ------------------------------------------
    def vertex_of(self, k):
        fi, c, _side = self.flags[k]
        return self.F[fi][c]

    def face_of(self, k):
        return self.flags[k][0]

    def edge_of(self, k):
        fi, c, side = self.flags[k]
        n = len(self.F[fi])
        if side == 0:
            a, b = self.F[fi][c], self.F[fi][(c + 1) % n]
        else:
            a, b = self.F[fi][(c - 1) % n], self.F[fi][c]
        return (a, b) if a < b else (b, a)

    # -- barycentric points ---------------------------------------------
    def vertex_point(self, k):
        return np.asarray(self.V[self.vertex_of(k)], float)

    def edge_point(self, k):
        a, b = self.edge_of(k)
        return 0.5 * (np.asarray(self.V[a], float)
                      + np.asarray(self.V[b], float))

    def face_point(self, k):
        f = self.F[self.face_of(k)]
        return np.mean([self.V[i] for i in f], axis=0)

    # -- the relations that make this a flag complex ---------------------
    def check(self):
        """Verify the defining relations; returns a dict of residuals."""
        m = len(self.flags)
        inv = {}
        for name, s in (('s0', self.s0), ('s1', self.s1), ('s2', self.s2)):
            inv[name] = sum(1 for k in range(m) if s[s[k]] != k)
        # s0 and s2 commute: (s0 s2)^2 = id
        comm = sum(1 for k in range(m)
                   if self.s0[self.s2[self.s0[self.s2[k]]]] != k)
        # no involution may fix a flag (except s2 on a boundary edge)
        fixed = {name: sum(1 for k in range(m) if s[k] == k)
                 for name, s in (('s0', self.s0), ('s1', self.s1),
                                 ('s2', self.s2))}
        return {'flags': m, 'not_involution': inv, 'commutator': comm,
                'fixed': fixed}

    def orbits(self, gens):
        """Partition the flags into orbits under the given involutions."""
        seen = [-1] * len(self.flags)
        out = []
        for k in range(len(self.flags)):
            if seen[k] >= 0:
                continue
            comp, stack = [], [k]
            seen[k] = len(out)
            while stack:
                x = stack.pop()
                comp.append(x)
                for s in gens:
                    y = s[x]
                    if seen[y] < 0:
                        seen[y] = len(out)
                        stack.append(y)
            out.append(comp)
        return out


def _dedupe(points, faces, tol=1e-9):
    """Weld coincident vertices and drop degenerate faces."""
    key = {}
    remap = []
    V = []
    q = 1.0 / max(tol, 1e-15)
    for p in points:
        k = (round(p[0] * q), round(p[1] * q), round(p[2] * q))
        if k not in key:
            key[k] = len(V)
            V.append([float(p[0]), float(p[1]), float(p[2])])
        remap.append(key[k])
    F = []
    for f in faces:
        g = []
        for i in f:
            j = remap[i]
            if not g or g[-1] != j:
                g.append(j)
        if len(g) > 2 and g[0] == g[-1]:
            g.pop()
        if len(g) > 2:
            F.append(g)
    return V, F


def _ordered_face(fc, flag_ids, point_of):
    """Order a set of flags into a face cycle using s0/s1 adjacency."""
    if not flag_ids:
        return []
    s = set(flag_ids)
    start = min(s)
    cycle = [start]
    s.discard(start)
    cur = start
    while s:
        nxt = None
        for step in (fc.s0, fc.s1, fc.s2):
            cand = step[cur]
            if cand in s:
                nxt = cand
                break
        if nxt is None:
            nxt = min(s)
        cycle.append(nxt)
        s.discard(nxt)
        cur = nxt
    return cycle


def dual(V, F):
    """Conway `d`: vertices become faces and faces become vertices.

    On flags this is the relabelling s0 <-> s2 -- the reason duality is
    an involution rather than a construction.  Realised geometrically by
    putting a vertex at each face centroid and walking the faces around
    each original vertex.
    """
    fc = FlagComplex(V, F)
    # one new vertex per old face
    pts = [np.mean([fc.V[i] for i in f], axis=0) for f in fc.F]
    # one new face per old vertex: the faces around it, in order
    around = {}
    for fi, face in enumerate(fc.F):
        n = len(face)
        for c in range(n):
            around.setdefault(face[c], []).append(
                (face[(c - 1) % n], fi, face[(c + 1) % n]))
    out = []
    for v, items in around.items():
        nxt = {a: (fi, b) for (a, fi, b) in items}
        start = items[0][0]
        cyc, cur = [], start
        for _ in range(len(items)):
            if cur not in nxt:
                break
            fi, b = nxt[cur]
            cyc.append(fi)
            cur = b
        if len(cyc) > 2:
            out.append(cyc)
    return _dedupe(pts, out)


def ambo(V, F):
    """Conway `a`: the MEDIAL -- one vertex per edge.

    Faces come in two families: one per original face (its edge
    midpoints in order) and one per original vertex (the midpoints of the
    edges meeting it).  On flags this is the s0 <-> s1 relabelling.
    """
    fc = FlagComplex(V, F)
    eidx, pts = {}, []
    for key in fc.edges:
        eidx[key] = len(pts)
        a, b = key
        pts.append(0.5 * (np.asarray(fc.V[a], float)
                          + np.asarray(fc.V[b], float)))
    out = []
    for face in fc.F:
        n = len(face)
        out.append([eidx[(face[c], face[(c + 1) % n])
                         if face[c] < face[(c + 1) % n]
                         else (face[(c + 1) % n], face[c])]
                    for c in range(n)])
    around = {}
    for fi, face in enumerate(fc.F):
        n = len(face)
        for c in range(n):
            v = face[c]
            prv, nxt = face[(c - 1) % n], face[(c + 1) % n]
            e_in = (v, prv) if v < prv else (prv, v)
            e_out = (v, nxt) if v < nxt else (nxt, v)
            around.setdefault(v, []).append((e_in, e_out))
    for v, pairs in around.items():
        chain = {a: b for (a, b) in pairs}
        start = pairs[0][0]
        cyc, cur = [], start
        for _ in range(len(pairs)):
            cyc.append(eidx[cur])
            if cur not in chain:
                break
            cur = chain[cur]
        if len(cyc) > 2:
            out.append(cyc)
    return _dedupe(pts, out)


def kis(V, F, height=0.25, only_n=0):
    """Conway `k`: raise a pyramid on every face (or every n-gon).

    One new vertex per face, pushed out along the face normal by
    `height` times the face's mean edge length; each face becomes a fan
    of triangles.
    """
    fc = FlagComplex(V, F)
    pts = [list(v) for v in fc.V]
    out = []
    for face in fc.F:
        n = len(face)
        if only_n and n != only_n:
            out.append(list(face))
            continue
        P = np.array([fc.V[i] for i in face], float)
        c = P.mean(axis=0)
        nrm = np.zeros(3)
        for i in range(n):
            nrm += np.cross(P[i] - c, P[(i + 1) % n] - c)
        ln = float(np.linalg.norm(nrm))
        # scale by the mean EDGE LENGTH, not the mean radius from the
        # centroid.  Both are reasonable; this is the convention the
        # generator has always used, and the equivalence gate against it
        # is what surfaced the difference.
        el = float(np.mean([np.linalg.norm(P[i] - P[(i + 1) % n])
                            for i in range(n)]))
        apex = c + (nrm / ln) * (height * el) if ln > 1e-15 else c
        ai = len(pts)
        pts.append([float(x) for x in apex])
        for i in range(n):
            out.append([face[i], face[(i + 1) % n], ai])
    return _dedupe(pts, out)


def _selftest():
    ok = True
    from .seeds import seed_poly

    # THE RELATIONS.  Each s is an involution, s0 and s2 commute, and the
    # flag count is exactly 4E.  If any of these fail the complex is not a
    # flag complex and every operator built on it is meaningless.
    bad = []
    for kind in ('TETRA', 'CUBE', 'OCTA', 'DODECA', 'ICOSA'):
        Vs, Fs = seed_poly(kind)
        fc = FlagComplex(Vs, Fs)
        r = fc.check()
        nE = len(fc.edges)
        if r['flags'] != 4 * nE:
            bad.append(f"{kind}:flags {r['flags']}!=4E={4 * nE}")
        elif any(r['not_involution'].values()):
            bad.append(f"{kind}:involution {r['not_involution']}")
        elif r['commutator']:
            bad.append(f"{kind}:(s0s2)^2 {r['commutator']}")
        elif any(r['fixed'].values()):
            bad.append(f"{kind}:fixed {r['fixed']}")
    good = not bad
    ok &= good
    print(f"flags: s0/s1/s2 are involutions, s0s2 commutes, |flags|=4E on "
          f"all five solids {'OK' if good else 'FAIL ' + '; '.join(bad)}")

    # Euler characteristic of every derived solid must stay 2.
    def chi(Vv, Ff):
        e = set()
        for f in Ff:
            for a, b in zip(f, list(f[1:]) + [f[0]]):
                e.add((a, b) if a < b else (b, a))
        return len(Vv) - len(e) + len(Ff)

    bad = []
    for kind in ('TETRA', 'CUBE', 'OCTA', 'DODECA', 'ICOSA'):
        Vs, Fs = seed_poly(kind)
        for name, fn in (('d', dual), ('a', ambo), ('k', kis)):
            Vo, Fo = fn(Vs, Fs)
            if chi(Vo, Fo) != 2:
                bad.append(f"{name}{kind}:chi={chi(Vo, Fo)}")
    good = not bad
    ok &= good
    print(f"flags: d/a/k preserve Euler characteristic on all five solids "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # dd = identity: the dual of the dual is the original, combinatorially
    bad = []
    for kind in ('TETRA', 'CUBE', 'OCTA', 'DODECA', 'ICOSA'):
        Vs, Fs = seed_poly(kind)
        V1, F1 = dual(Vs, Fs)
        V2, F2 = dual(V1, F1)
        if (len(V2), len(F2)) != (len(Vs), len(Fs)):
            bad.append(f"{kind}:{len(V2)},{len(F2)} vs {len(Vs)},{len(Fs)}")
    good = not bad
    ok &= good
    print(f"flags: dd is the identity on counts "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # the classical dual pairs, by vertex/face counts
    pairs = {'TETRA': (4, 4), 'CUBE': (6, 8), 'OCTA': (8, 6),
             'DODECA': (12, 20), 'ICOSA': (20, 12)}
    bad = []
    for kind, (nv, nf) in pairs.items():
        Vs, Fs = seed_poly(kind)
        Vd, Fd = dual(Vs, Fs)
        if (len(Vd), len(Fd)) != (nv, nf):
            bad.append(f"{kind}->{len(Vd)},{len(Fd)} want {nv},{nf}")
    good = not bad
    ok &= good
    print(f"flags: cube<->octahedron, dodeca<->icosahedron, tetra self-dual "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # ambo of any solid has one vertex per edge
    bad = []
    for kind in ('TETRA', 'CUBE', 'OCTA', 'DODECA', 'ICOSA'):
        Vs, Fs = seed_poly(kind)
        fc = FlagComplex(Vs, Fs)
        Va, Fa = ambo(Vs, Fs)
        if len(Va) != len(fc.edges):
            bad.append(f"{kind}:{len(Va)}!={len(fc.edges)}")
    good = not bad
    ok &= good
    print(f"flags: ambo puts exactly one vertex on every edge "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")


    # EQUIVALENCE WITH THE BESPOKE OPERATORS.  This module exists to
    # replace eight hand-written constructions with one mechanism, so the
    # gate that matters is whether it reproduces them.  Compared on what
    # is intrinsic -- counts, the face-size histogram, and the vertex set
    # as a shape -- because flag order differs from the hand-written
    # vertex order and that difference is meaningless.
    try:
        from .. import conway_operators as _co
    except ImportError:
        import conway_operators as _co
    bad = []
    for kind in ('TETRA', 'CUBE', 'OCTA', 'DODECA', 'ICOSA'):
        Vs, Fs = seed_poly(kind)
        Vs, Fs = _co.orient_outward(Vs, Fs)
        for nm, old, new in (('d', _co.op_dual, dual),
                             ('a', _co.op_ambo, ambo),
                             ('k', _co.op_kis, kis)):
            Vb, Fb = old(Vs, Fs)
            Vf, Ff = new(Vs, Fs)
            hb = sorted((len(f) for f in Fb))
            hf = sorted((len(f) for f in Ff))
            Ab = np.asarray(Vb, float); Ab -= Ab.mean(axis=0)
            Af = np.asarray(Vf, float); Af -= Af.mean(axis=0)
            db = np.sort(np.round(np.linalg.norm(Ab, axis=1), 7))
            df = np.sort(np.round(np.linalg.norm(Af, axis=1), 7))
            if (len(Vb), len(Fb)) != (len(Vf), len(Ff)) or hb != hf:
                bad.append(f"{nm}{kind}:combinatorics")
            elif float(np.max(np.abs(db - df))) > 1e-7:
                bad.append(f"{nm}{kind}:geometry")
    good = not bad
    ok &= good
    print(f"flags: d/a/k reproduce the bespoke operators on all five solids "
          f"(15 combinations) {'OK' if good else 'FAIL ' + ','.join(bad)}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("flags self-test failed")
