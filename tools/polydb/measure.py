# Metrics and grouping schemes for the polyhedron database.
#
# Everything here is computed from the vertex/face tables, so a record can
# never carry a measure its own geometry contradicts.
#
# One definition is worth stating because it is easy to get wrong. The
# MIDRADIUS is the distance from the centre to the LINE of an edge -- the
# radius of the sphere the edges are tangent to -- not the distance to the edge
# midpoint. For a uniform polyhedron the two agree, because an edge's endpoints
# are equidistant from the centre. For a Catalan or any other dual they do not:
# an edge there joins vertices of different types, so its tangent point is not
# its midpoint, and using midpoints would silently report a quantity that is
# not the midradius and is not even constant across edges.

import math
from collections import defaultdict


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def norm(a):
    return math.sqrt(dot(a, a))


def newell(vs):
    """Face normal, correct for non-convex and star polygons."""
    n = [0.0, 0.0, 0.0]
    for p, q in zip(vs, vs[1:] + vs[:1]):
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return n


def edges_of(F):
    """Distinct undirected edges, in first-seen order."""
    out, seen = [], set()
    for f in F:
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            k = frozenset((a, b))
            if k not in seen:
                seen.add(k)
                out.append((a, b))
    return out


def centre(V):
    return [sum(v[i] for v in V) / len(V) for i in range(3)]


def recentre(V):
    c = centre(V)
    return [tuple(v[i] - c[i] for i in range(3)) for v in V]


def edge_lengths(V, E):
    return [norm(sub(V[a], V[b])) for a, b in E]


def line_distance(p, a, b):
    """Perpendicular distance from p to the infinite line through a, b."""
    d = sub(b, a)
    L = norm(d)
    if L < 1e-15:
        return norm(sub(p, a))
    ap = sub(p, a)
    cr = [ap[1] * d[2] - ap[2] * d[1],
          ap[2] * d[0] - ap[0] * d[2],
          ap[0] * d[1] - ap[1] * d[0]]
    return norm(cr) / L


def plane_fit_normal(vs):
    """Unit normal of the plane through `vs`, by least squares.

    Needed because the Newell normal VANISHES on a crossed polygon: an
    antiparallelogram's two halves wind opposite ways and their contributions
    cancel exactly, even though the points are perfectly coplanar. Those faces
    are real -- they are the dual faces of the star uniforms whose vertex
    figure is a crossed rectangle (Har'El 1993, section 5) -- so treating a
    zero Newell normal as a degenerate face would wrongly reject them.
    """
    n = len(vs)
    if n < 3:
        return None
    c = [sum(v[i] for v in vs) / n for i in range(3)]
    # 3x3 scatter matrix of the centred points; the normal is the eigenvector
    # of its smallest eigenvalue.
    s = [[0.0] * 3 for _ in range(3)]
    for v in vs:
        d = [v[i] - c[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                s[i][j] += d[i] * d[j]
    try:
        import numpy as np
        w, vec = np.linalg.eigh(np.array(s))
        nrm = [float(x) for x in vec[:, 0]]
    except Exception:
        return None
    L = norm(nrm)
    if L < 1e-14:
        return None
    return [x / L for x in nrm]


def face_plane(V, f):
    """(unit normal, signed offset) of a face's plane.

    Falls back to a least-squares plane when the Newell normal cancels, which
    is what happens on a crossed face.
    """
    vs = [V[i] for i in f]
    n = newell(vs)
    L = norm(n)
    scale = max(norm(sub(vs[i], vs[0])) for i in range(1, len(vs))) or 1.0
    if L < 1e-12 * scale * scale:
        n = plane_fit_normal(vs)
        if n is None:
            return None, None
        return n, dot(n, vs[0])
    n = [c / L for c in n]
    return n, dot(n, vs[0])


def metrics(V, F, tol=1e-9):
    """Numeric metrics. Values that are not well defined come back None."""
    E = edges_of(F)
    L = edge_lengths(V, E)
    radii = [norm(v) for v in V]
    mids = [line_distance([0.0, 0.0, 0.0], V[a], V[b]) for a, b in E]

    out = {
        "edge_lengths_uniform": (max(L) - min(L)) < tol * max(L),
        "edge_length": max(L) if (max(L) - min(L)) < tol * max(L) else None,
        "circumradius": max(radii) if (max(radii) - min(radii)) < tol * max(radii) else None,
        "midradius": max(mids) if (max(mids) - min(mids)) < tol * max(mids) else None,
    }

    # inradius per face size
    per = defaultdict(list)
    for f in F:
        n, d = face_plane(V, f)
        if n is None:
            continue
        per[len(f)].append(abs(d))
    out["inradius"] = {k: (max(v) if (max(v) - min(v)) < tol * max(max(v), 1.0)
                           else None) for k, v in per.items()}

    # surface area and volume: only meaningful when nothing self-intersects
    out["surface_area"] = _area(V, F)
    out["volume"] = _volume(V, F)
    if out["surface_area"] and out["volume"] and out["volume"] > 0:
        A, Vol = out["surface_area"], out["volume"]
        out["isoperimetric_quotient"] = 36 * math.pi * Vol * Vol / (A ** 3)
    else:
        out["isoperimetric_quotient"] = None

    out["dihedral_angles"] = dihedrals(V, F)
    return out


def _area(V, F):
    tot = 0.0
    for f in F:
        vs = [V[i] for i in f]
        tot += 0.5 * norm(newell(vs))
    return tot


def _volume(V, F):
    """Signed volume by the divergence theorem; None if the result is not a
    sensible positive number (self-intersecting solids)."""
    tot = 0.0
    for f in F:
        vs = [V[i] for i in f]
        n = newell(vs)
        tot += dot(vs[0], n) / 6.0
    return abs(tot) if abs(tot) > 1e-12 else None


def dihedrals(V, F, nd=7):
    """Dihedral angles grouped by the (sides, sides) pair of the two faces."""
    inc = defaultdict(list)
    for i, f in enumerate(F):
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            inc[frozenset((a, b))].append(i)
    normals = []
    for f in F:
        n, _d = face_plane(V, f)
        normals.append(n)
    # Rounding is used ONLY to deduplicate; the value kept is full precision.
    # Deriving radians from a rounded degree figure costs ~1e-9 in the cosine,
    # which is enough to defeat exact recognition downstream.
    groups = defaultdict(dict)
    for _e, fis in inc.items():
        if len(fis) != 2:
            continue
        i, j = fis
        if normals[i] is None or normals[j] is None:
            continue
        c = max(-1.0, min(1.0, dot(normals[i], normals[j])))
        rad = math.pi - math.acos(c)
        pair = tuple(sorted((len(F[i]), len(F[j]))))
        groups[pair].setdefault(round(math.degrees(rad), nd), rad)
    out = []
    for pair, vals in sorted(groups.items()):
        for key in sorted(vals):
            rad = vals[key]
            out.append({"sides": list(pair), "degrees": math.degrees(rad),
                        "radians": rad})
    return out


def orient(V, F):
    """Give the faces a coherent winding: every edge traversed once in each
    direction. Returns (faces, orientable).

    Propagation is a breadth-first walk over the face adjacency, flipping each
    neighbour so it crosses the shared edge the opposite way. If the walk comes
    back to an already-oriented face with the wrong sense, the surface is
    ONE-SIDED and no coherent winding exists -- which is the correct answer for
    the hemipolyhedra, not a failure. In that case the faces are returned
    unchanged with orientable=False.

    For an orientable result the whole set is flipped if needed so that
    normals point outward on the majority of faces.
    """
    from collections import deque

    F = [list(f) for f in F]
    inc = defaultdict(list)
    for i, f in enumerate(F):
        for a, b in zip(f, f[1:] + f[:1]):
            inc[frozenset((a, b))].append(i)
    if any(len(v) != 2 for v in inc.values()):
        return F, None                      # not a closed 2-manifold

    out = [None] * len(F)
    orientable = True
    for seed in range(len(F)):
        if out[seed] is not None:
            continue
        out[seed] = F[seed]
        q = deque([seed])
        while q:
            i = q.popleft()
            fi = out[i]
            for a, b in zip(fi, fi[1:] + fi[:1]):
                nb = [x for x in inc[frozenset((a, b))] if x != i]
                if not nb:
                    continue
                j = nb[0]
                g = F[j]
                gd = set(zip(g, g[1:] + g[:1]))
                flipped = g if (b, a) in gd else g[::-1]
                if out[j] is None:
                    out[j] = flipped
                    q.append(j)
                elif out[j] != flipped and out[j] != flipped[::-1]:
                    pass
                elif out[j] != flipped:
                    orientable = False
    if not orientable:
        return F, False

    # verify coherence outright rather than trusting the walk
    seen = set()
    for f in out:
        for a, b in zip(f, f[1:] + f[:1]):
            if (a, b) in seen:
                return F, False
            seen.add((a, b))

    votes = 0
    for f in out:
        vs = [V[i] for i in f]
        n = newell(vs)
        c = [sum(v[k] for v in vs) / len(vs) for k in range(3)]
        votes += 1 if dot(n, c) > 0 else -1
    if votes < 0:
        out = [f[::-1] for f in out]
    return out, True


# -- vertex figures, angles, defects, densities -----------------------------
#
# All of these hang off the VERTEX FIGURE: the cycle of neighbours around a
# vertex, in order. Getting that cycle is the whole job; once it exists the
# face angles, the angular defect, the solid angle and Har'El's vertex density
# all follow.

def vertex_rings(V, F):
    """For each vertex, its neighbours in cyclic order, or None if the link is
    not a single cycle (which happens at a non-manifold vertex)."""
    # (vertex -> list of (prev, next)) taken from each incident face
    step = defaultdict(dict)
    for f in F:
        k = len(f)
        for i, v in enumerate(f):
            nxt = f[(i + 1) % k]
            prv = f[(i - 1) % k]
            step[v][prv] = nxt
    rings = []
    for v in range(len(V)):
        m = step.get(v)
        if not m:
            rings.append(None)
            continue
        start = next(iter(m))
        ring, cur = [start], start
        ok = True
        while True:
            nxt = m.get(cur)
            if nxt is None:
                ok = False
                break
            if nxt == start:
                break
            ring.append(nxt)
            cur = nxt
            if len(ring) > len(m):
                ok = False
                break
        rings.append(ring if ok and len(ring) == len(m) else None)
    return rings


def face_angles(V, F):
    """Interior angle at each corner of each face, in radians.

    For a star face this is the angle at the STAR POINT (36 degrees for a
    pentagram), because the face is stored as its true winding cycle and
    consecutive entries are its actual edges.
    """
    out = []
    for f in F:
        k = len(f)
        angs = []
        for i in range(k):
            p, c, n = V[f[(i - 1) % k]], V[f[i]], V[f[(i + 1) % k]]
            a, b = sub(p, c), sub(n, c)
            la, lb = norm(a), norm(b)
            if la < 1e-14 or lb < 1e-14:
                angs.append(None)
                continue
            angs.append(math.acos(max(-1.0, min(1.0, dot(a, b) / (la * lb)))))
        out.append(angs)
    return out


def angular_defects(V, F):
    """2*pi minus the face angles meeting at each vertex (Descartes).

    Do NOT sum these into a total: by Descartes' theorem the total is 4*pi for
    every genus-0 solid, so it carries no information the Euler characteristic
    does not already. The per-vertex value is what distinguishes solids.
    """
    at = defaultdict(float)
    fa = face_angles(V, F)
    for f, angs in zip(F, fa):
        for v, a in zip(f, angs):
            if a is not None:
                at[v] += a
    return [2 * math.pi - at.get(i, 0.0) for i in range(len(V))]


def _tri_solid_angle(a, b, c):
    """Signed solid angle of the cone on unit vectors a, b, c
    (Van Oosterom & Strackee, 1983)."""
    num = dot(a, [b[1] * c[2] - b[2] * c[1],
                  b[2] * c[0] - b[0] * c[2],
                  b[0] * c[1] - b[1] * c[0]])
    den = 1.0 + dot(a, b) + dot(b, c) + dot(c, a)
    return 2.0 * math.atan2(num, den)


def solid_angles(V, F, rings=None):
    """Solid angle subtended at each vertex, in steradians, by fan-triangulating
    the vertex figure. None where the link is not a cycle.

    Signed, and reported as such: for a star vertex whose faces wind round more
    than once the notion is convention-dependent, so the sign and magnitude are
    given rather than silently taking an absolute value.
    """
    rings = rings or vertex_rings(V, F)
    out = []
    for i, ring in enumerate(rings):
        if not ring or len(ring) < 3:
            out.append(None)
            continue
        us = []
        bad = False
        for j in ring:
            d = sub(V[j], V[i])
            L = norm(d)
            if L < 1e-14:
                bad = True
                break
            us.append([c / L for c in d])
        if bad:
            out.append(None)
            continue
        tot = 0.0
        for k in range(1, len(us) - 1):
            tot += _tri_solid_angle(us[0], us[k], us[k + 1])
        out.append(tot)
    return out


def vertex_densities(V, F, rings=None):
    """Har'El's vertex density d: how many times the vertex figure winds about
    the vertex. 1 for a convex solid; 0 or more for a star vertex.

    Reference: Z. Har'El, "Uniform Solution for Uniform Polyhedra", Geometriae
    Dedicata 47 (1993), section 3 -- "d, the vertex density, is a non-negative
    integer".
    """
    rings = rings or vertex_rings(V, F)
    out = []
    for i, ring in enumerate(rings):
        if not ring or len(ring) < 3:
            out.append(None)
            continue
        axis = V[i]
        L = norm(axis)
        if L < 1e-14:
            out.append(None)
            continue
        axis = [c / L for c in axis]
        # a basis of the plane perpendicular to the vertex's radial direction
        tmp = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]
        e1 = [tmp[k] - dot(tmp, axis) * axis[k] for k in range(3)]
        n1 = norm(e1)
        if n1 < 1e-14:
            out.append(None)
            continue
        e1 = [c / n1 for c in e1]
        e2 = [axis[1] * e1[2] - axis[2] * e1[1],
              axis[2] * e1[0] - axis[0] * e1[2],
              axis[0] * e1[1] - axis[1] * e1[0]]
        angs = []
        for j in ring:
            d = sub(V[j], V[i])
            angs.append(math.atan2(dot(d, e2), dot(d, e1)))
        turn = 0.0
        for k in range(len(angs)):
            da = angs[(k + 1) % len(angs)] - angs[k]
            while da <= -math.pi:
                da += 2 * math.pi
            while da > math.pi:
                da -= 2 * math.pi
            turn += da
        out.append(int(round(abs(turn) / (2 * math.pi))))
    return out


def has_central_face(V, F, tol=1e-9):
    """True when some face's plane passes through the centre.

    This is exactly the hemipolyhedron condition, and it is what makes the
    polar dual unbounded: a face through the centre reciprocates to a vertex at
    infinity. So `dual_unbounded` is computed, not curated.
    """
    for f in F:
        n, d = face_plane(V, f)
        if n is not None and abs(d) < tol:
            return True
    return False


def weld(V, F, nd=7):
    """Merge coincident vertices, remapping the faces.

    Compounds need this before anything indexes vertices: the five cubes
    inscribed in a dodecahedron occupy 40 vertex SLOTS but only 20 distinct
    POINTS, and the ten tetrahedra likewise. Left unwelded, the vertex->index
    map is not a bijection, so symmetry detection cannot map faces to faces
    and reports the trivial group.
    """
    key, remap, out = {}, [], []
    for v in V:
        k = tuple(round(float(c), nd) + 0.0 for c in v)
        if k not in key:
            key[k] = len(out)
            out.append(tuple(float(c) for c in v))
        remap.append(key[k])
    faces = []
    for f in F:
        g = [remap[i] for i in f]
        # drop any degenerate repeats introduced by the merge
        h = [g[i] for i in range(len(g)) if g[i] != g[(i - 1) % len(g)]]
        if len(h) >= 3:
            faces.append(h)
    return out, faces


def hull_faces(points, tol=1e-7):
    """Convex hull of `points` as (vertices, faces) with COPLANAR SIMPLICES
    MERGED into single polygons, wound consistently outward.

    scipy returns a triangulation; a zonohedron's faces are rhombi and larger
    polygons, so the triangles must be merged or every face would be reported
    as a triangle and the record would describe the wrong solid.
    """
    try:
        from scipy.spatial import ConvexHull
        import numpy as np
    except ImportError:
        return None, None
    P = np.asarray(points, dtype=float)
    try:
        h = ConvexHull(P)
    except Exception:
        return None, None

    planes = []          # index -> (normal, offset)
    members = []         # index -> set of vertex indices
    for eq, simplex in zip(h.equations, h.simplices):
        n, d = eq[:3], eq[3]
        hit = None
        for k, (n2, d2) in enumerate(planes):
            if abs(d - d2) < tol and float(np.max(np.abs(n - n2))) < tol:
                hit = k
                break
        if hit is None:
            planes.append((n, d))
            members.append(set())
            hit = len(planes) - 1
        members[hit].update(int(i) for i in simplex)

    used = sorted({i for s in members for i in s})
    remap = {old: new for new, old in enumerate(used)}
    verts = [tuple(float(c) for c in P[i]) for i in used]

    faces = []
    for k, idxs in enumerate(members):
        if len(idxs) < 3:
            continue
        n = np.asarray(planes[k][0], dtype=float)
        pts = sorted(idxs)
        c = P[pts].mean(axis=0)
        # order the face's vertices by angle in the face plane
        ref = P[pts[0]] - c
        ref = ref / (np.linalg.norm(ref) or 1.0)
        e2 = np.cross(n, ref)
        ang = []
        for i in pts:
            d = P[i] - c
            ang.append((math.atan2(float(d @ e2), float(d @ ref)), i))
        ang.sort()
        f = [remap[i] for _a, i in ang]
        # wind outward: the outward normal is -n for scipy's convention
        vs = [verts[i] for i in f]
        nn = newell(vs)
        cc = [sum(v[k2] for v in vs) / len(vs) for k2 in range(3)]
        if dot(nn, cc) < 0:
            f = f[::-1]
        faces.append(f)
    return verts, faces


def convex_hull_counts(V):
    """(vertices, edges, faces) of the convex hull of the vertex set, with
    coplanar facets merged, or None if scipy is unavailable or the hull is
    degenerate."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return None
    import numpy as np
    try:
        h = ConvexHull(np.asarray(V, dtype=float))
    except Exception:
        return None
    # merge coplanar simplices into true faces
    planes, assign = [], []
    for eq in h.equations:
        n, d = eq[:3], eq[3]
        hit = None
        for k, (n2, d2) in enumerate(planes):
            if (abs(d - d2) < 1e-7
                    and all(abs(n[i] - n2[i]) < 1e-7 for i in range(3))):
                hit = k
                break
        if hit is None:
            planes.append((n, d))
            hit = len(planes) - 1
        assign.append(hit)
    nv = len(set(int(i) for s in h.simplices for i in s))
    nf = len(planes)
    # Euler for a convex (genus 0) hull
    return nv, nv + nf - 2, nf


# -- grouping schemes -------------------------------------------------------

def face_groups(V, F, orbit_groups=None, tol=1e-7):
    """Named ways of partitioning the faces.

    symmetry_orbit  faces equivalent under the symmetry group (the canonical
                    grouping; supplied by the caller, which has the group)
    polygon_type    by number of sides
    coplanar        faces sharing a plane -- this is what distinguishes the
                    stellations, whose faces lie in the face planes of a core
                    solid, and the hemipolyhedra, whose faces pass through the
                    centre
    antipodal       faces swapped by the central inversion, when it is present
    """
    out = {}
    if orbit_groups is not None:
        out["symmetry_orbit"] = [list(g) for g in orbit_groups]

    by_sides = defaultdict(list)
    for i, f in enumerate(F):
        by_sides[len(f)].append(i)
    out["polygon_type"] = {str(k): v for k, v in sorted(by_sides.items())}

    planes = []
    assign = []
    for f in F:
        n, d = face_plane(V, f)
        if n is None:
            assign.append(None)
            continue
        hit = None
        for k, (n2, d2) in enumerate(planes):
            if abs(d - d2) < tol and all(abs(n[i] - n2[i]) < tol for i in range(3)):
                hit = k
                break
            if abs(d + d2) < tol and all(abs(n[i] + n2[i]) < tol for i in range(3)):
                hit = k
                break
        if hit is None:
            planes.append((n, d))
            hit = len(planes) - 1
        assign.append(hit)
    grouped = defaultdict(list)
    for i, k in enumerate(assign):
        if k is not None:
            grouped[k].append(i)
    coplanar = [sorted(v) for v in grouped.values() if len(v) > 1]
    if coplanar:
        out["coplanar"] = sorted(coplanar, key=lambda g: g[0])

    # antipodal pairs, when -I maps the vertex set to itself
    key = {tuple(round(c, 6) + 0.0 for c in v): i for i, v in enumerate(V)}
    if all(tuple(round(-c, 6) + 0.0 for c in v) in key for v in V):
        neg = {i: key[tuple(round(-c, 6) + 0.0 for c in V[i])]
               for i in range(len(V))}
        fk = {frozenset(f): i for i, f in enumerate(F)}
        pairs, used = [], set()
        for i, f in enumerate(F):
            if i in used:
                continue
            j = fk.get(frozenset(neg[x] for x in f))
            if j is None or j == i:
                continue
            used.add(i)
            used.add(j)
            pairs.append(sorted((i, j)))
        if pairs:
            out["antipodal"] = sorted(pairs, key=lambda g: g[0])
    return out
