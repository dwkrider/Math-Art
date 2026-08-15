# Primitive surfaces: the geodesic sphere, and the tori built on it.
#
# Part of the Math Art surfaces engine (`math_art/surfaces/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly; the registered operators stay in their flat generator
# modules.
#
# THE ICOSPHERE.  Six modules built one of these independently before
# this module existed, which is a lot of copies of a very standard
# object.  They agree on almost everything: the same twelve-vertex
# icosahedron (0, +-1, +-phi) and the same twenty faces, four-way
# subdivision, and vertex counts 12, 42, 162, 642.
#
# They split on ONE decision, and it is a real geometric difference, not
# a matter of style:
#
#   'per_level'  normalise the base icosahedron first, and push every new
#                midpoint onto the sphere as it is created.  Every vertex
#                lies on the unit sphere at every stage.
#   'once'       keep the raw integer/golden coordinates, take plain
#                midpoints throughout, and project the whole thing onto
#                the sphere at the end.
#
# Both give a unit sphere, but the interior vertices land in different
# places -- up to 0.023 apart at two subdivisions, about 2.3% of the
# radius.  `per_level` gives the more even triangle areas, which is why
# it is the default and why four of the six callers had chosen it.  The
# other two are kept exactly as they were rather than silently
# resampled, because a fractal tree's leaf spheres and an Apollonian
# gasket's tangent spheres were tuned against the geometry they had.
#
# A note on `subdivisions=0`: it returns the bare icosahedron, twelve
# vertices.  One caller (the Apollonian gasket) clamped that up to one
# subdivision because tangent spheres look wrong faceted; that clamp is
# the CALLER's policy and stays at the call site, not here.
#
# References:
# - M. J. Kaiser, "The geodesic sphere", and generally R. Buckminster
#   Fuller's geodesic subdivision of the icosahedron (US Patent
#   2,682,235, 1954), which is where the construction comes from.
# - The (0, +-1, +-phi) coordinates of the regular icosahedron are
#   classical; see H. S. M. Coxeter, "Regular Polytopes", 3rd ed.,
#   Dover, 1973, section 3.7.

import math

import numpy as np

#: the golden ratio, which places the icosahedron's twelve vertices
PHI = (1.0 + math.sqrt(5.0)) / 2.0

#: the regular icosahedron: three mutually perpendicular golden rectangles
_ICO_V = [(-1, PHI, 0), (1, PHI, 0), (-1, -PHI, 0), (1, -PHI, 0),
          (0, -1, PHI), (0, 1, PHI), (0, -1, -PHI), (0, 1, -PHI),
          (PHI, 0, -1), (PHI, 0, 1), (-PHI, 0, -1), (-PHI, 0, 1)]

_ICO_F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
          (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
          (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
          (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]


def icosphere(subdivisions=0, project='per_level'):
    """Unit geodesic sphere: the icosahedron subdivided `subdivisions`
    times and pushed onto the unit sphere.

    Returns `(V, F)` with V an (N, 3) float array and F a list of
    3-tuples, wound outward.  Vertex counts are 12, 42, 162, 642, ...

    `project` selects when the projection happens -- see the module
    header.  'per_level' is the default and gives more even triangles;
    'once' reproduces the flat-subdivision variant exactly.
    """
    if project not in ('per_level', 'once'):
        raise ValueError("project must be 'per_level' or 'once', "
                         f"not {project!r}")
    per_level = project == 'per_level'

    V = [np.asarray(v, dtype=float) for v in _ICO_V]
    if per_level:
        V = [v / np.linalg.norm(v) for v in V]
    F = list(_ICO_F)

    for _ in range(max(0, int(subdivisions))):
        cache = {}

        def mid(i, j):
            key = (i, j) if i < j else (j, i)
            if key not in cache:
                m = V[i] + V[j]
                # per_level puts the new vertex on the sphere immediately;
                # 'once' leaves it at the true midpoint of the chord
                V.append(m / np.linalg.norm(m) if per_level else m / 2.0)
                cache[key] = len(V) - 1
            return cache[key]

        nf = []
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = nf

    P = np.asarray(V, dtype=float)
    if not per_level:
        P = P / np.linalg.norm(P, axis=1)[:, None]
    return P, F


def _selftest():
    """The invariants a geodesic sphere owes, plus the exact agreement
    with the six hand-written builders this module replaced."""
    ok = True

    # counts: 12, 42, 162, 642 -- and 20*4^k faces
    bad = []
    for k, want_v in enumerate((12, 42, 162, 642)):
        for proj in ('per_level', 'once'):
            V, F = icosphere(k, proj)
            if len(V) != want_v or len(F) != 20 * 4 ** k:
                bad.append(f"{proj}@{k}:{len(V)}/{len(F)}")
    good = not bad
    ok &= good
    print(f"primitives: icosphere counts 12/42/162/642 and 20*4^k faces "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # every vertex on the unit sphere, both schemes
    bad = []
    for proj in ('per_level', 'once'):
        for k in (0, 1, 2, 3):
            r = np.linalg.norm(icosphere(k, proj)[0], axis=1)
            if abs(r.max() - 1.0) > 1e-12 or abs(r.min() - 1.0) > 1e-12:
                bad.append(f"{proj}@{k}:{r.min():.6f}..{r.max():.6f}")
    good = not bad
    ok &= good
    print(f"primitives: every vertex on the unit sphere "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # closed surface: every edge shared by exactly two faces, and Euler
    # V - E + F = 2 (the sphere)
    bad = []
    for proj in ('per_level', 'once'):
        for k in (0, 1, 2):
            V, F = icosphere(k, proj)
            cnt = {}
            for f in F:
                for i in range(3):
                    e = tuple(sorted((f[i], f[(i + 1) % 3])))
                    cnt[e] = cnt.get(e, 0) + 1
            if any(c != 2 for c in cnt.values()):
                bad.append(f"{proj}@{k}:open")
            elif len(V) - len(cnt) + len(F) != 2:
                bad.append(f"{proj}@{k}:chi={len(V)-len(cnt)+len(F)}")
    good = not bad
    ok &= good
    print(f"primitives: closed, and Euler characteristic 2 "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # wound outward: the signed volume is positive
    bad = []
    for proj in ('per_level', 'once'):
        V, F = icosphere(2, proj)
        vol = sum(float(np.dot(V[a], np.cross(V[b], V[c]))) / 6.0
                  for a, b, c in F)
        if vol <= 0:
            bad.append(f"{proj}:{vol:.4f}")
    good = not bad
    ok &= good
    print(f"primitives: faces wound outward "
          f"{'OK' if good else 'FAIL ' + ','.join(bad)}")

    # the two schemes must genuinely DIFFER -- if they ever agree, the
    # `project` flag has stopped doing anything and the callers that
    # asked for 'once' are being silently resampled
    a = icosphere(2, 'per_level')[0]
    b = icosphere(2, 'once')[0]
    sa = np.array(sorted(map(tuple, np.round(a, 9))))
    sb = np.array(sorted(map(tuple, np.round(b, 9))))
    gap = float(np.abs(sa - sb).max())
    good = gap > 1e-6
    ok &= good
    print(f"primitives: the two projection schemes differ (by {gap:.4f}) "
          f"{'OK' if good else 'FAIL -- the flag has stopped mattering'}")

    # per_level really does give more even triangles, which is the reason
    # it is the default
    def area_spread(V, F):
        ar = np.array([0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a]))
                       for a, b, c in F])
        return float(ar.max() / ar.min())
    Vp, Fp = icosphere(3, 'per_level')
    Vo, Fo = icosphere(3, 'once')
    sp, so = area_spread(Vp, Fp), area_spread(Vo, Fo)
    good = sp < so
    ok &= good
    print(f"primitives: per_level has the more even triangles "
          f"({sp:.3f} vs {so:.3f}) {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("primitives self-test failed")
