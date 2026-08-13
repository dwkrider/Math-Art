# Platonic seed solids and icosahedral face recovery.
#
# Part of the Math Art polyhedron engine (`math_art/polyhedra/`).  Python
# and numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operators stay in their flat generator
# modules and import this package.
#
# The five Platonic solids as explicit vertex and face tables, plus the
# face reconstruction that recovers an icosahedron's triangles from its
# twelve vertices.
#
# Six generators carried their own copy of these -- fractal_polyhedron,
# geodesic, platonic_twist, polylinks, rotegrity and weave.  The
# icosa_faces copies were all equivalent; the seed tables came in two
# normalisations, reconciled here by the `unit` flag.
#
# References:
#   The regular solids and their coordinates are classical (Euclid,
#   Elements XIII).  The golden-ratio vertex coordinates for the
#   icosahedron and dodecahedron are the standard modern presentation,
#   e.g. H. S. M. Coxeter, "Regular Polytopes" (3rd ed., 1973), ch. 3.

import math


#: the golden ratio, which the icosahedral and dodecahedral
#: coordinates are built from
PHI = (1.0 + 5.0 ** 0.5) / 2.0


def _unit(v):
    """Unit vector; a zero vector is returned unchanged rather than
    raising, so a degenerate seed cannot crash the caller."""
    l = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / l for x in v)


def icosa_faces(V):
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


def seed_poly(kind, unit=False):
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
        F = icosa_faces(V)
    elif kind == 'DODECA':
        IV = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                IV += [(0, a, b), (a, b, 0), (b, 0, a)]
        IF = icosa_faces(IV)
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
    V = [list(v) for v in V]
    if unit:
        # normalise to unit circumradius.  Three generators (geodesic,
        # rotegrity, weave) built their seeds this way and three
        # (polylinks, fractal_polyhedron, platonic_twist) used the raw
        # integer/golden coordinates.  The two are EXACTLY related by this
        # scale -- verified vertex-for-vertex on all five solids, with
        # identical face lists -- so one function and one flag covers both.
        r = math.sqrt(sum(c * c for c in V[0]))
        if r > 1e-12:
            V = [[c / r for c in v] for v in V]
    return V, [list(f) for f in F]


def _selftest():
    ok = True

    # Euler's formula V - E + F = 2 holds for every convex polyhedron, and
    # each Platonic solid has its known census.  This checks the tables as
    # SOLIDS, not merely as arrays of the right length.
    census = {'TETRA': (4, 6, 4), 'CUBE': (8, 12, 6), 'OCTA': (6, 12, 8),
              'DODECA': (20, 30, 12), 'ICOSA': (12, 30, 20)}
    bad = []
    for kind, (nv, ne, nf) in census.items():
        V, F = seed_poly(kind)
        edges = set()
        for f in F:
            for a, b in zip(f, list(f[1:]) + [f[0]]):
                edges.add((a, b) if a < b else (b, a))
        if (len(V), len(edges), len(F)) != (nv, ne, nf):
            bad.append(f"{kind}:{len(V)},{len(edges)},{len(F)}")
        elif len(V) - len(edges) + len(F) != 2:
            bad.append(f"{kind}:chi")
    good = not bad
    ok &= good
    print(f"seeds: V-E+F=2 and the census for all five solids "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # Every vertex of a Platonic solid is the same distance from the
    # centre, and every edge the same length -- that is what makes it
    # regular, and it is the check that a mistyped coordinate fails.
    bad = []
    for kind in census:
        V, F = seed_poly(kind)
        A = [math.sqrt(sum(c * c for c in v)) for v in V]
        el = []
        for f in F:
            for a, b in zip(f, list(f[1:]) + [f[0]]):
                el.append(math.dist(V[a], V[b]))
        if max(A) - min(A) > 1e-9 or max(el) - min(el) > 1e-9:
            bad.append(f"{kind}:r{max(A) - min(A):.1e} e{max(el) - min(el):.1e}")
    good = not bad
    ok &= good
    print(f"seeds: vertex-transitive and edge-regular "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # `unit` rescales to circumradius 1 and changes NOTHING else -- the
    # face lists and the shape must be identical.  Three generators want
    # the raw coordinates and three want this; they differ only by it.
    bad = []
    for kind in census:
        V, F = seed_poly(kind)
        Vu, Fu = seed_poly(kind, unit=True)
        r = math.sqrt(sum(c * c for c in V[0]))
        if [list(f) for f in F] != [list(f) for f in Fu]:
            bad.append(f"{kind}:faces")
        elif max(abs(a / r - b) for va, vb in zip(V, Vu)
                 for a, b in zip(va, vb)) > 1e-12:
            bad.append(f"{kind}:scale")
        elif abs(math.sqrt(sum(c * c for c in Vu[0])) - 1.0) > 1e-12:
            bad.append(f"{kind}:radius")
    good = not bad
    ok &= good
    print(f"seeds: unit=True is a pure rescale, faces untouched "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # icosa_faces recovers the 20 triangles from the 12 vertices alone
    V, F = seed_poly('ICOSA')
    rec = icosa_faces([tuple(v) for v in V])
    good = (len(rec) == 20
            and {tuple(sorted(t)) for t in rec} == {tuple(sorted(f))
                                                    for f in F})
    ok &= good
    print(f"seeds: icosa_faces recovers all 20 triangles from the vertices "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("seeds self-test failed")
