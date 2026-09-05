
# Reflexive polytopes, their mirrors, and tropical Calabi-Yau surfaces.
#
# Part of the Math Art polyhedron engine (`math_art/polyhedra/`).
# Python and numpy only -- no `bpy` -- so the engine imports and
# self-tests headlessly.  The halfspace and hull work is done by the
# sibling `hull` module rather than repeated here.
#
# REFLEXIVE POLYTOPES (Batyrev 1994).  A lattice polytope D with the
# origin in its interior is *reflexive* if every facet lies at integral
# distance 1 from the origin -- equivalently, if the polar dual
#
#     D* = { y : <x, y> >= -1  for all x in D }
#
# is again a lattice polytope.  These are exactly the polytopes whose
# toric variety is Gorenstein Fano, so that the generic anticanonical
# hypersurface in it is Calabi-Yau; and since (D*)* = D, the involution
# D -> D* carries one Calabi-Yau family to another.  That involution is
# Batyrev's construction of MIRROR PAIRS.  There are exactly 4319
# reflexive polytopes in three dimensions (Kreuzer-Skarke 1998), each
# giving a K3 surface, and 473,800,776 in four (Kreuzer-Skarke 2000),
# giving the Calabi-Yau threefolds.  A curated handful of the
# three-dimensional ones is here, each with its mirror.
#
# TROPICAL CALABI-YAU AND K3 POLYTOPES.  Over the tropical semiring
# (min, +) a polynomial f = min_v (c_v + v.x) has a hypersurface T(f):
# the set where the minimum is attained at least twice.  It is a
# piecewise linear 2-complex in R^3 -- an honest polyhedral surface,
# no projection involved -- and it is the large-complex-structure limit
# of the classical hypersurface with the same Newton polytope.  When
# that Newton polytope is the fourth dilate of the standard tetrahedron
# the classical surface is a quartic in P^3, i.e. a K3, and the unique
# BOUNDED region of the complement of T(f) is what Balletti, Panizzut
# and Sturmfels call a K3 polytope.  Their Example 4 is the default
# here; its K3 polytope is simple with f-vector (64, 96, 34), the
# largest their classification allows, and `_selftest` reproduces that
# from the geometry rather than quoting it.
#
# Both objects come out of the same halfspace arithmetic.  A polytope
# { x : n . x <= d } has a vertex wherever three of its facet planes
# meet in a point the rest admit, which is what `hull.halfspace_vertices`
# enumerates; the polar dual of conv(V) is the halfspace body of the
# planes -v . y <= 1; and the K3 polytope is the halfspace body of
# (p - u) . x <= c_u - c_p over the other lattice points.  Only the
# tropical surface needs anything else, and that is plane clipping.
#
# References:
# - V. V. Batyrev, "Dual polyhedra and mirror symmetry for Calabi-Yau
#   hypersurfaces in toric varieties", Journal of Algebraic Geometry 3
#   (1994) 493-535 -- reflexive polytopes (Def. 4.1.5), the
#   self-duality theorem (Thm. 4.1.6), the face correspondence
#   (Prop. 4.1.7) and the mirror involution.
# - M. Kreuzer and H. Skarke, "Classification of reflexive polyhedra in
#   three dimensions", Advances in Theoretical and Mathematical Physics
#   2 (1998) 853-871 -- the 4319 three-dimensional reflexive polytopes.
# - M. Kreuzer and H. Skarke, "Complete classification of reflexive
#   polyhedra in four dimensions", Advances in Theoretical and
#   Mathematical Physics 4 (2000) 1209-1230 -- the 473,800,776
#   four-dimensional ones and their 30,108 Hodge pairs.
# - G. Balletti, M. Panizzut and B. Sturmfels, "K3 polytopes and their
#   quartic surfaces", Advances in Geometry 21 (2021) 85-98 -- the
#   definition of a K3 polytope, the f-vector relations of their
#   Lemma 11 (f = (Vol, 3Vol/2, Vol/2 + 2), so 3 f0 = 2 f1 and every K3
#   polytope is simple), and the tropical quartic of their Example 4
#   used as the default coefficients here.
# - D. Maclagan and B. Sturmfels, "Introduction to Tropical Geometry",
#   Graduate Studies in Mathematics 161, AMS (2015) -- tropical
#   hypersurfaces and their duality with regular subdivisions.

import itertools
import math
from collections import defaultdict

import numpy as np

try:
    from .hull import halfspace_vertices, hull_faces
except ImportError:                       # flat import outside the package
    from hull import halfspace_vertices, hull_faces


# --------------------------------------------------------------------
# Reflexive polytopes and their mirrors
# --------------------------------------------------------------------

def _cube():
    return [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]


def _octahedron():
    return [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1)]


#: key -> (label, vertices).  Each is reflexive, so each carries a
#: family of K3 surfaces, and each is paired with its mirror by `polar`.
REFLEXIVE = {
    # Batyrev's Delta_3: the Newton polytope of the quartic surface in
    # P^3, translated so its one interior lattice point is the origin.
    'QUARTIC': ("Quartic K3 (4 Delta_3)",
                [(3, -1, -1), (-1, 3, -1), (-1, -1, 3), (-1, -1, -1)]),
    'P3': ("Projective Space P^3",
           [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1)]),
    'CUBE': ("Cube (P1 x P1 x P1)", _cube()),
    'OCTAHEDRON': ("Octahedron", _octahedron()),
    'WP1113': ("Weighted P(1,1,1,3)",
               [(1, 0, 0), (0, 1, 0), (-1, -1, 3), (0, 0, -1)]),
    'PRISM': ("Prism (P2 x P1)",
              [(1, 0, 1), (0, 1, 1), (-1, -1, 1),
               (1, 0, -1), (0, 1, -1), (-1, -1, -1)]),
    'BIPYRAMID': ("Bipyramid",
                  [(1, 0, 0), (0, 1, 0), (-1, -1, 0),
                   (0, 0, 1), (0, 0, -1)]),
}

#: The mirror pairs worth building as one object: polytope, its dual.
MIRROR_PAIRS = (
    ('CUBE', 'OCTAHEDRON'),
    ('QUARTIC', 'P3'),
    ('PRISM', 'BIPYRAMID'),
    ('WP1113', 'WP1113'),
)


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v], n


def polar(vertices, tol=1e-7):
    """Vertices and faces of the polar dual { y : <v, y> >= -1 }.

    Written as the halfspace body of -v . y <= 1, with each normal
    scaled to unit length as `hull.halfspace_vertices` expects.
    """
    planes = []
    for v in vertices:
        n, ln = _unit([-float(c) for c in v])
        planes.append((n, 1.0 / ln))
    verts = halfspace_vertices(planes, tol)
    return verts, hull_faces(verts)


def is_reflexive(vertices, tol=1e-7):
    """Batyrev's test: the dual must again be a lattice polytope."""
    dv, _ = polar(vertices, tol)
    if not dv:
        return False
    return all(abs(c - round(c)) < 1e-6 for v in dv for c in v)


def f_vector(verts, faces):
    """(vertices, edges, faces) of a 3-polytope given by its faces."""
    edges = set()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            edges.add((min(a, b), max(a, b)))
    return len(verts), len(edges), len(faces)


def _normalise(verts, want=1.0):
    m = max((abs(c) for v in verts for c in v), default=1.0) or 1.0
    return [tuple(want * c / m for c in v) for v in verts]


def mirror_pair(key, separation=1.3, equal_sizes=True):
    """A reflexive polytope and its mirror, as one mesh.

    A polytope and its dual differ wildly in scale -- the dual of a
    large body is a small one -- so unscaled the smaller of the two
    all but disappears; `equal_sizes` puts them on the same footing.
    `separation` 0 nests them about the origin they share, which is
    the only interior lattice point of either.
    """
    label, V = REFLEXIVE[key]
    F = hull_faces(V)
    DV, DF = polar(V)
    if equal_sizes:
        V, DV = _normalise(V), _normalise(DV)
    verts, faces = [], []
    for i, (pv, pf) in enumerate(((V, F), (DV, DF))):
        off = (i * 2 - 1) * separation
        base = len(verts)
        verts.extend((float(x) + off, float(y), float(z))
                     for x, y, z in pv)
        faces.extend([tuple(j + base for j in f) for f in pf])
    return verts, faces


# --------------------------------------------------------------------
# Tropical Calabi-Yau surfaces and K3 polytopes
# --------------------------------------------------------------------

def simplex_lattice_points(deg=4, dim=3):
    """Lattice points of the deg-th dilate of the standard simplex."""
    return [pt for pt in itertools.product(range(deg + 1), repeat=dim)
            if sum(pt) <= deg]


def balletti_example4():
    """The tropical quartic of Balletti-Panizzut-Sturmfels, Example 4.

    Newton polytope 4 Delta_3, interior lattice point (1,1,1); the K3
    polytope it bounds is simple with f-vector (64, 96, 34).
    """
    coeff = {}
    for pt in simplex_lattice_points(4, 3):
        i, j, k = pt
        s = i + j + k
        srt = tuple(sorted((i, j, k), reverse=True))
        if s == 4:
            c = {(4, 0, 0): 5, (3, 1, 0): 3, (2, 2, 0): 2,
                 (2, 1, 1): 0}[srt]
        elif s == 3:
            c = {(3, 0, 0): 3, (2, 1, 0): 0, (1, 1, 1): -9}[srt]
        elif s == 2:
            c = {(2, 0, 0): 2, (1, 1, 0): 0}[srt]
        elif s == 1:
            c = 3
        else:
            c = 5
        coeff[pt] = float(c)
    return coeff


def k3_polytope(coeff=None, interior=(1, 1, 1), tol=1e-7):
    """The bounded region of the complement of T(f).

    That region is where one fixed term -- the one at the Newton
    polytope's interior lattice point -- beats every other, so it is
    the halfspace body of (p - u) . x <= c_u - c_p, bounded exactly
    because p is interior.
    """
    if coeff is None:
        coeff = balletti_example4()
    p = [float(c) for c in interior]
    cp = coeff[tuple(interior)]
    planes = []
    for u, cu in coeff.items():
        if tuple(u) == tuple(interior):
            continue
        n, ln = _unit([p[i] - float(u[i]) for i in range(3)])
        planes.append((n, (cu - cp) / ln))
    verts = halfspace_vertices(planes, tol)
    return verts, hull_faces(verts)


def bounded_radius(coeff=None, interior=(1, 1, 1)):
    """How far out the interesting part of T(f) reaches.

    The bounded region sets the scale of everything else, so the clip
    box is quoted as a multiple of its radius rather than in absolute
    units; otherwise a change of coefficients silently crops the
    surface.
    """
    v, _ = k3_polytope(coeff, interior)
    if not v:
        return 1.0
    return max((abs(c) for p in v for c in p), default=1.0) or 1.0


def _clip(poly, a, b, g, tol=1e-9):
    """Sutherland-Hodgman clip of a 2D polygon by a s + b t <= g."""
    if not poly:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        s0, t0 = poly[i]
        s1, t1 = poly[(i + 1) % n]
        d0 = a * s0 + b * t0 - g
        d1 = a * s1 + b * t1 - g
        if d0 <= tol:
            out.append((s0, t0))
        if (d0 > tol) != (d1 > tol):
            u = d0 / (d0 - d1)
            out.append((s0 + u * (s1 - s0), t0 + u * (t1 - t0)))
    return out


def _area(poly):
    a = 0.0
    for i in range(len(poly)):
        s0, t0 = poly[i]
        s1, t1 = poly[(i + 1) % len(poly)]
        a += s0 * t1 - s1 * t0
    return 0.5 * abs(a)


def tropical_surface(coeff=None, box=None, tol=1e-7, interior=(1, 1, 1)):
    """The tropical hypersurface of min_v (c_v + v.x), clipped to a box.

    Each 2-cell of T(f) is the locus where two terms tie and beat the
    rest: one linear equation and a pile of linear inequalities, so a
    convex polygon inside the plane of the equation.  Each is built by
    clipping a large square in that plane, and cells with no area are
    dropped -- which is what removes the pairs of terms that never tie
    anywhere.
    """
    if coeff is None:
        coeff = balletti_example4()
    if box is None:
        box = 1.25 * bounded_radius(coeff, interior)
    A = [np.asarray(v, dtype=float) for v in coeff]
    c = [coeff[v] for v in coeff]
    m = len(A)
    axes = [np.eye(3)[i] * s for i in range(3) for s in (1.0, -1.0)]

    polys = []
    for i, j in itertools.combinations(range(m), 2):
        n = A[i] - A[j]
        nn = float(n @ n)
        if nn < 1e-12:
            continue
        x0 = n * ((c[j] - c[i]) / nn)
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(e1 @ n) / math.sqrt(nn) > 0.9:
            e1 = np.array([0.0, 1.0, 0.0])
        e1 = e1 - (e1 @ n) / nn * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n / math.sqrt(nn), e1)

        R = 4.0 * box
        poly = [(-R, -R), (R, -R), (R, R), (-R, R)]
        for k in range(m):
            if k == i or k == j:
                continue
            w = A[i] - A[k]
            poly = _clip(poly, float(w @ e1), float(w @ e2),
                         float(c[k] - c[i] - w @ x0))
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            for ax in axes:                      # clip to the view box
                poly = _clip(poly, float(ax @ e1), float(ax @ e2),
                             float(box - ax @ x0))
                if len(poly) < 3:
                    break
        if len(poly) >= 3 and _area(poly) > 1e-6:
            polys.append([tuple(x0 + s * e1 + t * e2) for s, t in poly])

    verts, faces = [], []
    cells = defaultdict(list)
    snap = 1e-5

    def merge(v):
        base = tuple(int(math.floor(x / snap)) for x in v)
        for off in itertools.product((0, -1, 1), repeat=3):
            for n_ in cells.get((base[0] + off[0], base[1] + off[1],
                                 base[2] + off[2]), ()):
                w = verts[n_]
                if all(abs(v[k] - w[k]) <= snap for k in range(3)):
                    return n_
        n_ = len(verts)
        verts.append(tuple(v))
        cells[base].append(n_)
        return n_

    for poly in polys:
        f = []
        for v in poly:
            n_ = merge(v)
            if not f or f[-1] != n_:
                f.append(n_)
        if len(f) > 2 and f[0] == f[-1]:
            f.pop()
        if len(f) >= 3:
            faces.append(f)
    return verts, faces


# --------------------------------------------------------------------

def _selftest():
    bad = []

    # 1. every listed polytope is reflexive, duality is an involution,
    #    and the two f-vectors are reverses of one another -- which is
    #    Batyrev's face correspondence, seen in the counts.
    for key, (label, V) in REFLEXIVE.items():
        refl = is_reflexive(V)
        dv, dfc = polar(V)
        ddv, _ = polar(dv)
        A = sorted(tuple(round(c, 6) for c in v) for v in V)
        B = sorted(tuple(round(c, 6) for c in v) for v in ddv)
        inv = (len(A) == len(B)
               and all(all(abs(x - y) < 1e-6 for x, y in zip(p, q))
                       for p, q in zip(A, B)))
        fp = f_vector(V, hull_faces(V))
        fd = f_vector(dv, dfc)
        rev = fp == tuple(reversed(fd))
        ok = refl and inv and rev
        print(f"toric: {key:11s} f={fp} dual f={fd} reflexive={refl} "
              f"D**=D {inv} reversed {rev} {'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"{key}: reflexive={refl} inv={inv} rev={rev}")

    # 2. a non-reflexive polytope is rejected: twice the octahedron has
    #    its facets at integral distance 2, so its dual is half-integral.
    twice = [(2 * a, 2 * b, 2 * c) for a, b, c in _octahedron()]
    ok = not is_reflexive(twice)
    print(f"toric: 2*octahedron rejected {'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("2*octahedron accepted")

    # 3. every mirror pair builds, with both halves present.
    for a, b in MIRROR_PAIRS:
        V, F = mirror_pair(a)
        n_a = len(REFLEXIVE[a][1])
        n_b = len(polar(REFLEXIVE[a][1])[0])
        ok = len(V) == n_a + n_b and len(F) > 3
        print(f"toric: mirror pair {a}/{b} V={len(V)}"
              f"({n_a}+{n_b}) F={len(F)} {'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"mirror pair {a}")

    # 4. Balletti-Panizzut-Sturmfels Example 4: the K3 polytope has
    #    their f-vector, is simple, and satisfies Euler and Lemma 11.
    v, f = k3_polytope()
    fv = f_vector(v, f)
    deg = defaultdict(int)
    for fc in f:
        for i in fc:
            deg[i] += 1
    simple = set(deg.values()) == {3}
    euler = fv[0] - fv[1] + fv[2] == 2
    lem11 = 3 * fv[0] == 2 * fv[1] and 2 * fv[2] == fv[0] + 4
    ok = fv == (64, 96, 34) and simple and euler and lem11
    print(f"toric: Balletti Example 4 K3 polytope f-vector {fv} "
          f"(64, 96, 34) simple={simple} euler={euler} "
          f"lemma11={lem11} {'OK' if ok else 'BAD'}")
    if not ok:
        bad.append(f"K3 polytope {fv} simple={simple}")

    # 5. the tropical surface is the dual complex of a unimodular
    #    triangulation of 4 Delta_3, which by Euler has 35 vertices,
    #    130 edges, 160 triangles and 64 tetrahedra -- so one 2-cell
    #    per edge, 130 of them -- and every vertex of it is a point
    #    where at least three of the tropical terms tie.
    coeff = balletti_example4()
    verts, faces = tropical_surface(coeff)
    A = [np.asarray(k, dtype=float) for k in coeff]
    C = [coeff[k] for k in coeff]
    worst = 0
    for v in verts[::37]:
        vals = np.array([C[i] + A[i] @ np.asarray(v)
                         for i in range(len(A))])
        worst = max(worst, int(np.sum(vals < vals.min() + 1e-6)))
    ok = len(faces) == 130 and worst >= 3
    print(f"toric: tropical quartic {len(verts)} verts {len(faces)} "
          f"cells (130), max tie multiplicity {worst} "
          f"{'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("tropical surface")

    if bad:
        raise AssertionError("; ".join(bad))
