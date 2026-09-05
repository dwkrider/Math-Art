# Spherical kite lattices, and the search for a spherical hat analogue.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests
# headlessly.  `polyhedra` is imported lazily inside `kite_solid` so
# `import patterns` does not drag the polyhedron engine in.
#
# The 2023 aperiodic monotile is eight kites of the [3.4.6.4] kite
# lattice: the deltoidal trihexagonal tiling, dual of the
# rhombitrihexagonal tiling, arising from the (2,3,6) triangle group.
# That group is Euclidean.  Its SPHERICAL siblings are (2,3,3), (2,3,4)
# and (2,3,5), whose kite duals are real Catalan solids, and the kite
# count is |mirror group| / 2:
#
#     (2,3,4)  *432  order 48  ->  deltoidal icositetrahedron,  24 kites
#     (2,3,5)  *532  order 120 ->  deltoidal hexecontahedron,   60 kites
#
# So an EIGHT-kite piece can only ever partition the first: 24/8 = 3,
# while 60/8 = 7.5 rules the icosahedral case out by counting, before
# any geometry is built.  (The hexecontahedron has 60 faces, not 120;
# 120 is the order of the group, i.e. the flag count.)
#
# What this module then asks is the MONOHEDRAL question, which is the
# only version worth asking.  "Can 24 kites be split into three
# connected eight-sets" is nearly certainly yes and says nothing; "is
# there ONE eight-kite piece whose images under the octahedral group
# tile the solid" is the property that would make it a hat analogue.
#
# The answer is yes, and the abundance is the point: of 1845 connected
# eight-kite pieces, 366 are distinct under the 48-element group and
# FIFTEEN of those tile monohedrally.  That is a positive result and a
# deflationary one.  The hat matters because it is essentially unique
# and because it forces aperiodicity; fifteen solutions is a weak
# constraint, and on a compact surface aperiodicity is not even
# definable -- every tiling of a sphere is finite, so there is no
# infinite non-periodic structure to force.  A spherical einstein is
# therefore not a hard open problem here; it is a category error, and
# what survives the translation is only the combinatorial setting.
#
# References:
# - David Smith, Joseph Samuel Myers, Craig S. Kaplan & Chaim
#   Goodman-Strauss, "An aperiodic monotile" (arXiv:2303.10798, 2023) --
#   the hat, and its description as eight kites of the [3.4.6.4] lattice.
# - John H. Conway, Heidi Burgiel & Chaim Goodman-Strauss, "The
#   Symmetries of Things" (2008) -- the orbifold signatures *432 and
#   *532 used to build the symmetry groups.
# - Eugene Catalan, "Memoire sur la theorie des polyedres", Journal de
#   l'Ecole Polytechnique 41 (1865) -- the duals of the Archimedean
#   solids, including both deltoidal solids used here.

from math import acos, pi
import numpy as np


# Conway words for the two spherical kite lattices, with their kite
# counts and the orbifold signature of the symmetry group.
KITE_SOLIDS = {
    'deC': {'name': "deltoidal icositetrahedron", 'kites': 24,
            'signature': 'STAR_432'},
    'deD': {'name': "deltoidal hexecontahedron", 'kites': 60,
            'signature': 'STAR_532'},
}


def kite_solid(word='deC', hart_iters=600):
    """(V, F) for a spherical kite lattice: the Conway word canonicalised
    and radially projected onto the unit sphere.

    Three existing calls and no new geometry -- `apply_conway` builds the
    dual, `canonicalize_best` gives it the true Catalan metric (a raw
    centroid dual of an `aa` seed does not have it), and normalising the
    vertices puts every kite on the sphere."""
    try:
        from ..polyhedra import conway as cw, canonical as canon
    except ImportError:                       # flat import (test runner)
        from polyhedra import conway as cw, canonical as canon
    V, F = cw.apply_conway(word)
    V = canon.canonicalize_best(V, F, hart_iters=hart_iters)
    A = np.asarray(V, float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    return A, [list(f) for f in F]


def face_centroids(V, F):
    """Unit-sphere centroid of each face."""
    C = np.array([np.mean(np.asarray(V, float)[f], axis=0) for f in F])
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def face_adjacency(F):
    """face index -> set of edge-sharing neighbours."""
    edge = {}
    for i, f in enumerate(F):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            edge.setdefault((a, b) if a < b else (b, a), []).append(i)
    adj = {i: set() for i in range(len(F))}
    for fs in edge.values():
        if len(fs) == 2:
            adj[fs[0]].add(fs[1])
            adj[fs[1]].add(fs[0])
    return adj


def face_permutations(C, mats, tol=1e-6):
    """Each symmetry matrix rewritten as a permutation of the faces."""
    C = np.asarray(C, float)
    perms = set()
    for M in mats:
        img = C @ np.asarray(M, float).T
        p, ok = [], True
        for q in img:
            d = np.linalg.norm(C - q, axis=1)
            j = int(np.argmin(d))
            if d[j] > tol:
                ok = False
                break
            p.append(j)
        if ok and len(set(p)) == len(p):
            perms.add(tuple(p))
    return sorted(perms)


def connected_pieces(adj, n, size):
    """Every edge-connected `size`-subset of the face graph.

    Grown from each seed with a canonical ordering so each subset is
    produced once."""
    seen, out = set(), []

    def grow(cur, frontier):
        if len(cur) == size:
            k = frozenset(cur)
            if k not in seen:
                seen.add(k)
                out.append(k)
            return
        top = max(cur)
        for f in sorted(frontier):
            if f < top:
                continue
            grow(cur | {f}, (frontier | adj[f]) - cur - {f})

    for s in range(n):
        grow({s}, set(adj[s]))
    return out


def _canonical(piece, perms):
    """Least image of a piece under the group -- a TOTAL order, so this
    must compare sorted tuples.  Comparing frozensets would silently use
    subset ordering, which is partial, and `min` would then return an
    arbitrary element and dedupe almost nothing."""
    return min(tuple(sorted(p[i] for i in piece)) for p in perms)


def distinct_pieces(pieces, perms):
    """One representative per symmetry class."""
    seen, reps = set(), []
    for s in pieces:
        k = _canonical(s, perms)
        if k not in seen:
            seen.add(k)
            reps.append(s)
    return reps


def tiles_monohedrally(piece, perms, n):
    """A partition of all n faces into images of `piece`, or None.

    This is the hat-analogue test: not any partition into equal-sized
    connected sets, but one where every part is the SAME piece moved by
    a symmetry of the solid."""
    orbit = sorted({frozenset(p[i] for i in piece) for p in perms})
    universe = frozenset(range(n))
    need = n // len(piece)
    answer = []

    def rec(used, chosen):
        if used == universe:
            answer.extend(chosen)
            return True
        if len(chosen) >= need:
            return False
        pick = min(universe - used)
        for o in orbit:
            if pick in o and not (o & used):
                if rec(used | o, chosen + [o]):
                    return True
        return False

    return answer if rec(frozenset(), []) else None


def monohedral_pieces(word='deC', size=8, hart_iters=600):
    """(solutions, stats) for the spherical hat-analogue search.

    Returns the list of (piece, partition) that tile the solid
    monohedrally, plus a stats dict.  Raises ValueError when the piece
    size does not divide the kite count -- the icosahedral case, which
    is settled by arithmetic rather than search."""
    from .spherical import build_group
    spec = KITE_SOLIDS[word]
    if spec['kites'] % size:
        raise ValueError(
            "%d kites is not divisible by %d (%.2f pieces), so no "
            "%d-kite piece can partition the %s"
            % (spec['kites'], size, spec['kites'] / size, size,
               spec['name']))
    V, F = kite_solid(word, hart_iters)
    C = face_centroids(V, F)
    adj = face_adjacency(F)
    perms = face_permutations(C, build_group(spec['signature']))
    pieces = connected_pieces(adj, len(F), size)
    reps = distinct_pieces(pieces, perms)
    sols = []
    for s in reps:
        part = tiles_monohedrally(s, perms, len(F))
        if part:
            sols.append((sorted(s), [sorted(x) for x in part]))
    return sols, {'kites': len(F), 'perms': len(perms),
                  'connected': len(pieces), 'distinct': len(reps),
                  'tiling': len(sols)}


# --------------------------------------------------------------------
# Spherical coverage: the sphere's answer to `_coverage`
# --------------------------------------------------------------------

def spherical_area(poly, tol=1e-12):
    """Area of a spherical polygon given by unit vectors, by Girard's
    theorem: the spherical excess (sum of interior angles) - (n-2) pi.

    The planar coverage checks in this repo sample a grid and count
    point-in-polygon hits.  On a sphere that needs a point-in-spherical-
    polygon test that does not exist here -- but areas do the same job
    exactly and with no sampling at all, because a set of spherical
    polygons covers the sphere without gaps or overlaps precisely when
    its areas sum to 4 pi."""
    P = np.asarray(poly, float)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    n = len(P)
    total = 0.0
    for i in range(n):
        prev, cur, nxt = P[i - 1], P[i], P[(i + 1) % n]
        # tangents at `cur` along the two edges
        t1 = prev - cur * float(np.dot(prev, cur))
        t2 = nxt - cur * float(np.dot(nxt, cur))
        n1 = float(np.linalg.norm(t1))
        n2 = float(np.linalg.norm(t2))
        if n1 < tol or n2 < tol:
            continue
        c = float(np.dot(t1, t2)) / (n1 * n2)
        total += acos(max(-1.0, min(1.0, c)))
    return total - (n - 2) * pi


def covers_sphere(V, F, tol=1e-9):
    """(ok, total, 4 pi): do these spherical faces tile the whole sphere?"""
    A = np.asarray(V, float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    total = sum(spherical_area(A[f]) for f in F)
    return abs(total - 4.0 * pi) <= tol * 4.0 * pi, total, 4.0 * pi


# --------------------------------------------------------------------

def _selftest():
    ok = True

    # The two solids, and the counting argument that settles the
    # icosahedral case without a search.
    for word, kites in (('deC', 24), ('deD', 60)):
        V, F = kite_solid(word, hart_iters=200)
        sizes = {len(f) for f in F}
        good = len(F) == kites and sizes == {4}
        ok &= good
        print(f"spherical_kites: {word} has {len(F)} kite faces "
              f"(expected {kites}), all quads={sizes == {4}} "
              f"{'OK' if good else 'FAIL'}")

        cov, total, want = covers_sphere(V, F)
        ok &= cov
        print(f"spherical_kites: {word} faces cover the sphere, area "
              f"{total:.9f} vs 4pi {want:.9f} {'OK' if cov else 'FAIL'}")

    # 60/8 is not an integer, so the icosahedral hat analogue is ruled
    # out by arithmetic.  The module must SAY so rather than search.
    try:
        monohedral_pieces('deD', 8)
        good = False
    except ValueError:
        good = True
    ok &= good
    print(f"spherical_kites: deD rejects an 8-kite piece by counting "
          f"(60/8 = 7.5) {'OK' if good else 'FAIL'}")

    # The octahedral case: 24/8 = 3, and the search finds the analogue.
    sols, stats = monohedral_pieces('deC', 8, hart_iters=200)
    good = (stats['kites'] == 24 and stats['perms'] == 48
            and stats['distinct'] < stats['connected']
            and len(sols) > 0)
    ok &= good
    print(f"spherical_kites: deC {stats['connected']} connected pieces, "
          f"{stats['distinct']} distinct under {stats['perms']} symmetries, "
          f"{stats['tiling']} tile monohedrally {'OK' if good else 'FAIL'}")

    # Every reported solution must really be a partition: three disjoint
    # images covering all 24 kites.
    bad = 0
    for _piece, part in sols:
        seen = [f for grp in part for f in grp]
        if len(part) != 3 or sorted(seen) != list(range(24)):
            bad += 1
    ok &= not bad
    print(f"spherical_kites: all {len(sols)} solutions are exact "
          f"partitions into 3 pieces ({bad} bad) {'OK' if not bad else 'FAIL'}")

    # The dedupe must use a TOTAL order.  Comparing frozensets uses
    # subset ordering, which is partial, and would leave the classes
    # essentially undeduped -- the bug this check exists to catch.
    good = stats['distinct'] * 2 < stats['connected']
    ok &= good
    print(f"spherical_kites: symmetry dedupe is effective "
          f"({stats['connected']} -> {stats['distinct']}) "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("spherical_kites self-test failed")
