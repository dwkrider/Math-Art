# Spherical substrates: patterns built ON the sphere, not projected onto it.
#
# Part of the Math Art Pattern Engine (`math_art/patterns/`).  Python +
# numpy only -- no `bpy`.  `regular_solids_generator` is imported lazily
# so `import patterns` stays light.
#
# Stereographic projection (see `surfacemap.StereographicSurface`) maps
# the whole INFINITE plane to the sphere minus a point, so a finite patch
# can only ever reach a cap: a square patch at spread 1 covers half the
# sphere, and its corners run further round than its edge midpoints, so
# the rim comes out ragged.  No setting fixes that -- reaching the pole
# needs an infinite patch, and the tile count grows as the square of the
# spread.  It is the right map for a patch with no symmetry to exploit
# (a Penrose or hat patch has none), and the wrong one for a periodic
# pattern that could simply be built on the sphere in the first place.
#
# So this module does what `knot_carpet_generator`'s knot ball does:
# take a SPHERICAL substrate and draw on it directly.  Radially
# projected, the Platonic and Archimedean solids ARE the regular and
# Archimedean tilings of the sphere -- the cube is the spherical 4.4.4,
# the truncated icosahedron (the football) the spherical 5.6.6 -- so a
# pattern laid on their faces covers the whole sphere, uniformly, with
# no puncture and no polar seam.
#
# Curvature does not disappear; it collects at the vertices.  A pattern
# wanting six neighbours everywhere cannot have them on a sphere: by
# Euler's formula a triangulation with only degree-5 and degree-6
# vertices has exactly twelve degree-5 ones, whatever the frequency.
# Those defects are a theorem about the sphere, not a flaw in the
# construction, and they are what a football's twelve pentagons are.
#
# The bridge between the flat pattern builders and the sphere is the
# exponential map at a face centre.  A planar point at polar coordinates
# (rho, theta) in the tangent frame (e1, e2) at unit vector c lands at
#
#     exp_c(rho, theta) = cos(rho) c + sin(rho) (cos(theta) e1
#                                                + sin(theta) e2)
#
# and `log_map` inverts it.  Distances become angles, straight lines
# become geodesics, and every existing per-tile 2D builder keeps working
# unchanged inside a face.
#
# References:
# - R. Buckminster Fuller -- the geodesic subdivision of the icosahedron.
# - D. L. D. Caspar & A. Klug, "Physical principles in the construction
#   of regular viruses", Cold Spring Harbor Symposia on Quantitative
#   Biology 27 (1962) -- the twelve pentagonal defects, and patterns on
#   a sphere built from a triangulated substrate.
# - Leonhard Euler (1758) -- the polyhedron formula V - E + F = 2 that
#   forces those twelve defects.

from math import acos, atan2, cos, sin
import numpy as np


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-300)


# Substrates that are subdivided rather than named: the geodesic sphere
# and its Goldberg dual.  Unlike the 18 fixed solids these take a
# frequency, so the cell count is a dial rather than a menu choice.
SUBDIVIDED = ('GEODESIC', 'GOLDBERG')


def geodesic_faces(freq=3, base='ICOSA', dual=False):
    """Faces of a geodesic sphere, or of its Goldberg dual.

    The Class-I geodesic subdivision of the icosahedron gives 20 freq^2
    triangles; its dual gives 10 freq^2 + 2 faces, all hexagons except
    for exactly TWELVE pentagons -- Euler's formula again, the same
    twelve defects a football has.  The dual is usually the better
    substrate for strapwork, since a hexagonal cell carries a six-fold
    rosette the way the classical patterns do, while a triangular cell
    can only carry a three-fold one."""
    try:
        from .. import geodesic_generator as gg
    except ImportError:
        import geodesic_generator as gg
    V, F = gg.build_sphere(base, max(1, int(freq)), 'I')
    if dual:
        V, F = gg.goldberg_dual(V, F)
    V = _unit(np.asarray(V, float))
    return [V[list(f)] for f in F]


def substrate_faces(kind='TI', freq=3):
    """The spherical substrate for a base-solid id: a named Platonic or
    Archimedean solid, or a geodesic sphere / Goldberg dual at `freq`."""
    if kind == 'GEODESIC':
        return geodesic_faces(freq, dual=False)
    if kind == 'GOLDBERG':
        return geodesic_faces(freq, dual=True)
    return solid_faces(kind)


def solid_items():
    """(id, label, description) for every substrate: the geodesic pair
    first, then every Platonic and Archimedean solid.  Radially
    projected, each IS a spherical uniform tiling."""
    try:
        from .. import regular_solids_generator as rs
    except ImportError:
        import regular_solids_generator as rs
    out = [
        ('GOLDBERG', "Goldberg (Geodesic Dual)",
         "Hexagons plus exactly twelve pentagons, at the chosen "
         "frequency -- the football generalised. Usually the best "
         "strapwork substrate: hexagonal cells carry six-fold rosettes"),
        ('GEODESIC', "Geodesic Icosahedron",
         "20 x frequency^2 triangles, subdivided and projected to the "
         "sphere. Triangular cells, so three-fold rosettes"),
    ]
    for fam, kind in ((rs.PLATONIC, 'regular'),
                      (rs.ARCHIMEDEAN, 'Archimedean')):
        for sid, label, _nota in fam:
            out.append((sid, label,
                        "One patch per face of the %s -- the spherical "
                        "%s tiling" % (label.lower(), kind)))
    return out


def _family_of(sid):
    try:
        from .. import regular_solids_generator as rs
    except ImportError:
        import regular_solids_generator as rs
    for fam, name in ((rs.PLATONIC, 'PLATONIC'),
                      (rs.ARCHIMEDEAN, 'ARCHIMEDEAN')):
        for s, _label, _nota in fam:
            if s == sid:
                return name
    raise ValueError("unknown solid %r" % (sid,))


def solid_faces(sid='ICOSA', canon_iters=250):
    """The faces of a radially projected solid, as spherical polygons.

    Returns a list of (M, 3) arrays of UNIT vectors, one per face, in
    the solid's own vertex order.  This is the spherical tiling the
    pattern is built on."""
    try:
        from .. import regular_solids_generator as rs
    except ImportError:
        import regular_solids_generator as rs
    V, F, _sizes = rs.build_solid(_family_of(sid), sid, 6, 1.0,
                                  canon_iters=canon_iters)
    V = _unit(np.asarray(V, float))
    return [V[list(f)] for f in F]


def face_frame(poly3):
    """(centre, e1, e2): the unit face centre and an orthonormal tangent
    frame there, with e1 toward the face's first vertex so the frame is
    tied to the face rather than to an arbitrary global axis."""
    P = _unit(np.asarray(poly3, float))
    c = _unit(P.mean(axis=0))
    ref = P[0] - c * float(np.dot(P[0], c))
    if np.linalg.norm(ref) < 1e-12:               # degenerate: pick any
        ref = np.array([0.0, 0.0, 1.0]) - c * float(c[2])
        if np.linalg.norm(ref) < 1e-12:
            ref = np.array([1.0, 0.0, 0.0]) - c * float(c[0])
    e1 = _unit(ref)
    return c, e1, np.cross(c, e1)


def log_map(c, e1, e2, pts3):
    """Sphere -> tangent plane at c: the inverse exponential map.

    A point at angular distance rho from c, in tangent direction theta,
    becomes the planar point rho (cos theta, sin theta).  Geodesic
    distance is preserved along rays from c, so a face maps to a planar
    polygon a flat builder can consume."""
    P = _unit(np.asarray(pts3, float).reshape(-1, 3))
    d = np.clip(P @ np.asarray(c, float), -1.0, 1.0)
    rho = np.arccos(d)
    T = P - np.outer(d, c)
    n = np.linalg.norm(T, axis=1, keepdims=True)
    T = T / np.maximum(n, 1e-300)
    th = np.arctan2(T @ np.asarray(e2, float), T @ np.asarray(e1, float))
    return np.column_stack([rho * np.cos(th), rho * np.sin(th)])


def exp_map(c, e1, e2, pts2):
    """Tangent plane at c -> sphere: the exponential map.

    The exact inverse of `log_map`, so a builder can work in the plane
    and have its output land on the sphere as geodesics."""
    P = np.asarray(pts2, float).reshape(-1, 2)
    rho = np.hypot(P[:, 0], P[:, 1])
    th = np.arctan2(P[:, 1], P[:, 0])
    c = np.asarray(c, float)
    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)
    dirs = (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))
    return (np.outer(np.cos(rho), c)
            + dirs * np.sin(rho)[:, None])


def refine_geodesic(poly3, max_edge, closed=True, cap=64):
    """Subdivide a spherical polyline along its geodesics, so no edge
    spans more than `max_edge` radians.  Straight chords between widely
    separated points cut through the sphere; this keeps them on it."""
    P = _unit(np.asarray(poly3, float))
    n = len(P)
    out = []
    last = n if closed else n - 1
    for k in range(last):
        a, b = P[k], P[(k + 1) % n]
        ang = acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))
        m = int(min(cap, max(1, np.ceil(ang / max(max_edge, 1e-9)))))
        for i in range(m):
            t = i / float(m)
            if ang < 1e-12:
                out.append(a)
            else:
                s = sin(ang)
                out.append((sin((1 - t) * ang) * a + sin(t * ang) * b) / s)
    if not closed:
        out.append(P[-1])
    return _unit(np.array(out))


def sphere_ribbon(path3, width, closed, radius=1.0):
    """(left, right) boundaries of a ribbon of angular `width` following a
    spherical polyline, offset perpendicular to the path within the
    sphere's tangent plane.

    The flat builder mitres in the plane; here the offset direction at
    each point is cross(p, tangent), which is the in-surface normal, so
    the ribbon hugs the sphere instead of leaving it."""
    P = _unit(np.asarray(path3, float))
    n = len(P)
    half = 0.5 * float(width)
    left, right = [], []
    for i in range(n):
        if closed:
            a, b = P[i - 1], P[(i + 1) % n]
        else:
            a = P[max(i - 1, 0)]
            b = P[min(i + 1, n - 1)]
        t = b - a
        t = t - P[i] * float(np.dot(t, P[i]))       # project into tangent
        if np.linalg.norm(t) < 1e-12:
            t = np.array([1.0, 0.0, 0.0])
            t = t - P[i] * float(np.dot(t, P[i]))
        s = _unit(np.cross(P[i], _unit(t)))
        left.append(_unit(P[i] + s * half) * radius)
        right.append(_unit(P[i] - s * half) * radius)
    return np.array(left), np.array(right)


def pair_straightest(incident):
    """Pair a node's incident half-edges "straightest-through", in any
    dimension: match the two directions closest to anti-parallel first.

    The planar strapwork solver has its own 2-component copy of this;
    this one dots over whatever components the direction vectors have,
    so the same band topology comes out on the sphere."""
    n = len(incident)
    scored = []
    for i in range(n):
        di = np.asarray(incident[i][1], float)
        for j in range(i + 1, n):
            dj = np.asarray(incident[j][1], float)
            scored.append((float(np.dot(di, dj)), i, j))
    scored.sort()
    taken = [False] * n
    out = {}
    for _dot, i, j in scored:
        if taken[i] or taken[j]:
            continue
        taken[i] = taken[j] = True
        out[incident[i][0]] = incident[j][0]
        out[incident[j][0]] = incident[i][0]
    return out


class _Nodes3:
    """Endpoint welding on the sphere, by a rounded-coordinate hash."""

    def __init__(self, tol=1e-6):
        self.tol = tol
        self.pts = []
        self._ix = {}

    def add(self, p):
        k = tuple(round(float(c) / self.tol) for c in p)
        if k not in self._ix:
            self._ix[k] = len(self.pts)
            self.pts.append(np.asarray(p, float))
        return self._ix[k]


def build_arrangement_3d(segments, tol=1e-6):
    """The 3D twin of the strapwork arrangement: weld segment endpoints
    into shared nodes and pair the half-edges at each node.

    Returns (pts, seg_nodes, pair) in exactly the planar builder's shape,
    so `trace_bands` -- which is purely combinatorial and never looks at
    a coordinate -- runs on the result unchanged."""
    ns = _Nodes3(tol)
    seg_nodes = []
    for a, b in segments:
        na, nb = ns.add(a), ns.add(b)
        seg_nodes.append(None if na == nb else (na, nb))
    incident = {}
    for si, nn in enumerate(seg_nodes):
        if nn is None:
            continue
        na, nb = nn
        pa, pb = ns.pts[na], ns.pts[nb]
        d = _unit(pb - pa)
        incident.setdefault(na, []).append((si, d))
        incident.setdefault(nb, []).append((si, -d))
    pair = {node: pair_straightest(lst) for node, lst in incident.items()}
    return ns.pts, seg_nodes, pair


def _selftest():
    ok = True

    # Every solid must come back as a closed set of spherical faces.
    for sid, faces, want in (('CUBE', solid_faces('CUBE'), 6),
                             ('ICOSA', solid_faces('ICOSA'), 20),
                             ('TI', solid_faces('TI'), 32)):
        onsphere = max(
            float(np.max(np.abs(np.linalg.norm(f, axis=1) - 1.0)))
            for f in faces)
        good = len(faces) == want and onsphere < 1e-9
        ok &= good
        print(f"spherical_substrate: {sid} -> {len(faces)} spherical faces "
              f"(want {want}), on-sphere {onsphere:.1e} "
              f"{'OK' if good else 'FAIL'}")

    # The geodesic pair: face counts are 20 f^2 triangles, and its dual
    # has 10 f^2 + 2 faces of which EXACTLY twelve are pentagons --
    # Euler's theorem, at every frequency.
    for f in (1, 2, 3):
        tri = substrate_faces('GEODESIC', f)
        gol = substrate_faces('GOLDBERG', f)
        pent = sum(1 for x in gol if len(x) == 5)
        hexa = sum(1 for x in gol if len(x) == 6)
        good = (len(tri) == 20 * f * f and len(gol) == 10 * f * f + 2
                and pent == 12 and pent + hexa == len(gol))
        ok &= good
        print(f"spherical_substrate: freq {f} -> {len(tri)} triangles "
              f"(want {20*f*f}), dual {len(gol)} faces = {hexa} hexagons "
              f"+ {pent} pentagons {'OK' if good else 'FAIL'}")

    # Both must also close the sphere exactly.
    from .spherical_kites import spherical_area as _sa
    for kind in ('GEODESIC', 'GOLDBERG'):
        tot = sum(_sa(x) for x in substrate_faces(kind, 2))
        good = abs(tot - 4.0 * np.pi) < 1e-9
        ok &= good
        print(f"spherical_substrate: {kind} freq 2 covers the sphere "
              f"({tot:.9f}) {'OK' if good else 'FAIL'}")

    # The faces must cover the WHOLE sphere -- the property stereographic
    # projection of a finite patch can never have.
    from .spherical_kites import spherical_area
    faces = solid_faces('TI')
    total = sum(spherical_area(f) for f in faces)
    good = abs(total - 4.0 * np.pi) < 1e-9
    ok &= good
    print(f"spherical_substrate: truncated icosahedron covers the whole "
          f"sphere, area {total:.9f} vs 4pi {4*np.pi:.9f} "
          f"{'OK' if good else 'FAIL'}")

    # exp and log must invert each other, or a flat builder's output
    # would not land back where it came from.
    f = solid_faces('ICOSA')[0]
    c, e1, e2 = face_frame(f)
    flat = log_map(c, e1, e2, f)
    back = exp_map(c, e1, e2, flat)
    err = float(np.max(np.abs(back - f)))
    good = err < 1e-12
    ok &= good
    print(f"spherical_substrate: exp(log(face)) == face ({err:.1e}) "
          f"{'OK' if good else 'FAIL'}")

    # The frame is orthonormal and centred on the face.
    orth = max(abs(float(np.dot(c, e1))), abs(float(np.dot(c, e2))),
               abs(float(np.dot(e1, e2))))
    norms = max(abs(float(np.linalg.norm(v)) - 1.0) for v in (c, e1, e2))
    good = orth < 1e-12 and norms < 1e-12
    ok &= good
    print(f"spherical_substrate: face frame orthonormal "
          f"(orth {orth:.1e}, norm {norms:.1e}) {'OK' if good else 'FAIL'}")

    # Adjacent faces must agree on their shared edge midpoints -- this is
    # what lets bands cross from face to face instead of stopping.
    faces = solid_faces('CUBE')
    mids = {}
    for fc in faces:
        for k in range(len(fc)):
            m = _unit(fc[k] + fc[(k + 1) % len(fc)])
            key = tuple(round(float(x), 6) for x in m)
            mids[key] = mids.get(key, 0) + 1
    shared = sum(1 for v in mids.values() if v == 2)
    good = shared == 12 and len(mids) == 12
    ok &= good
    print(f"spherical_substrate: cube edge midpoints shared by exactly 2 "
          f"faces ({shared}/{len(mids)} of 12) {'OK' if good else 'FAIL'}")

    # A geodesic refinement must stay on the sphere and respect max_edge.
    r = refine_geodesic(faces[0], 0.2)
    on = float(np.max(np.abs(np.linalg.norm(r, axis=1) - 1.0)))
    step = max(acos(float(np.clip(np.dot(r[i], r[(i + 1) % len(r)]),
                                  -1, 1))) for i in range(len(r)))
    good = on < 1e-12 and step <= 0.2 + 1e-9
    ok &= good
    print(f"spherical_substrate: geodesic refine stays on the sphere "
          f"({on:.1e}), max step {step:.3f} <= 0.2 "
          f"{'OK' if good else 'FAIL'}")

    # The ribbon must ride ON the sphere, both edges.
    path = refine_geodesic(faces[0], 0.15)
    left, right = sphere_ribbon(path, 0.12, True, 1.0)
    off = max(float(np.max(np.abs(np.linalg.norm(left, axis=1) - 1.0))),
              float(np.max(np.abs(np.linalg.norm(right, axis=1) - 1.0))))
    good = off < 1e-12
    ok &= good
    print(f"spherical_substrate: ribbon edges stay on the sphere "
          f"({off:.1e}) {'OK' if good else 'FAIL'}")

    # The 3D arrangement must weld a shared endpoint into ONE node and
    # pair the two collinear half-edges through it.
    a = np.array([1.0, 0.0, 0.0])
    b = _unit(np.array([1.0, 0.3, 0.0]))
    d = _unit(np.array([1.0, 0.6, 0.0]))
    pts, seg_nodes, pair = build_arrangement_3d([(a, b), (b, d)])
    good = (len(pts) == 3 and seg_nodes == [(0, 1), (1, 2)]
            and pair.get(1, {}).get(0) == 1)
    ok &= good
    print(f"spherical_substrate: 3D arrangement welds and pairs through "
          f"({len(pts)} nodes) {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("spherical_substrate self-test failed")
