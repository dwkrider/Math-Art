# Faceting: new polyhedra on an existing vertex set.
#
# Part of the Math Art polyhedra engine (`math_art/polyhedra/`).  Python +
# numpy only -- no `bpy` -- so the engine imports and self-tests headlessly.
#
# Faceting is the operation dual to stellation.  Stellation extends a
# solid's FACE PLANES until they cut out new cells; faceting keeps the
# VERTICES and runs new faces through them.  Every stellation of a solid
# corresponds to a faceting of its dual, which is exactly why this module
# exists: `polyhedra/hull.polar_dual` reciprocates the visible boundary a
# stellation engine emits, not the true self-intersecting faces, so it
# returns the wrong solid for anything nonconvex.  Faceting reaches those
# duals directly.
#
# The general faceting problem -- every polyhedron on a given vertex set --
# is enormous.  What is implemented here is the case that matters for the
# noble solids, where the search collapses to something small:
#
#   * a polyhedron is ISOGONAL when its vertices are one orbit of its
#     symmetry group, so START from a single orbit and isogonality is free;
#   * it is ISOHEDRAL when its faces are one orbit, so a candidate is
#     determined by ONE face plus the group;
#   * NOBLE means both at once.
#
# So the search is: enumerate the planes through three or more of the
# vertices, read the candidate polygons off each plane, take the orbit of
# each under the group, and keep those whose edges are each used exactly
# twice.  That last test is what separates a polyhedron from a mere pile
# of polygons, and it is cheap.
#
# References:
# - P. Du Val, "Homographies, Quaternions and Rotations" (1964), and
#   H. S. M. Coxeter, P. Du Val, H. T. Flather & J. F. Petrie, "The
#   Fifty-Nine Icosahedra" (1938), for the stellation/faceting duality.
# - B. Grunbaum, "Polyhedra with hollow faces", in Polytopes: Abstract,
#   Convex and Computational (1994), on noble polyhedra.
# - M. Bruckner, "Vielecke und Vielflache" (1900), which first pictured
#   several of the noble solids.

import itertools
import math

import numpy as np


def _key(p, nd=5):
    return tuple(round(float(c), nd) + 0.0 for c in p)


def vertex_planes(V, tol=1e-6):
    """Every plane containing three or more of the vertices.

    Returned as (unit normal, offset, [vertex indices]) with the offset
    non-negative, so a plane and its opposite are one entry.
    """
    n = len(V)
    P = [np.array(v, float) for v in V]
    out = {}
    for a, b, c in itertools.combinations(range(n), 3):
        nx = np.cross(P[b] - P[a], P[c] - P[a])
        ln = float(np.linalg.norm(nx))
        if ln < tol:
            continue                      # collinear
        nx = nx / ln
        d = float(nx @ P[a])
        if d < 0:
            nx, d = -nx, -d
        k = _key(np.append(nx, d))
        if k in out:
            continue
        on = [i for i in range(n) if abs(float(nx @ P[i]) - d) < tol]
        if len(on) >= 3:
            out[k] = (tuple(nx), d, on)
    return list(out.values())


def _cycles_on_plane(V, nx, on):
    """Candidate polygons through the vertices lying on one plane.

    The points are sorted by azimuth about their centroid, which gives
    the convex circuit; stepping that circuit by d gives the star
    polygon {k/d}.  Only steps coprime with k close up into a single
    circuit visiting every point, and {k/d} and {k/(k-d)} are the same
    edge set traversed oppositely, so d runs to k/2.
    """
    P = [np.array(V[i], float) for i in on]
    cen = sum(P) / len(P)
    u = P[0] - cen
    ln = float(np.linalg.norm(u))
    if ln < 1e-9:
        return []
    u = u / ln
    w = np.cross(np.array(nx, float), u)

    def az(t):
        e = P[t] - cen
        return math.atan2(float(e @ w), float(e @ u))
    order = [on[t] for t in sorted(range(len(on)), key=az)]
    k = len(order)
    out = []
    for d in range(1, k // 2 + 1):
        if math.gcd(k, d) != 1:
            continue
        out.append([order[(d * t) % k] for t in range(k)])
    return out


def _edges(face):
    return {tuple(sorted((face[i], face[(i + 1) % len(face)])))
            for i in range(len(face))}


def face_orbit(V, face, G, tol=1e-6):
    """The orbit of one face under the group, as vertex-index cycles.

    Faces are deduped by their EDGE SET, so the same polygon reached by
    two group elements -- or traversed backwards -- counts once.
    """
    idx = {_key(v): i for i, v in enumerate(V)}
    P = [np.array(v, float) for v in V]
    seen = {}
    for M in G:
        f = []
        for i in face:
            j = idx.get(_key(M @ P[i]))
            if j is None:
                return None               # group does not preserve V
            f.append(j)
        if len(set(f)) != len(f):
            return None
        seen.setdefault(frozenset(_edges(f)), f)
    return list(seen.values())


def noble_facetings(V, G, min_sides=3, max_sides=12):
    """Every noble polyhedron on the vertex orbit V under the group G.

    Returns [(faces, plane normal, step), ...].  A candidate survives
    only if every edge lies in exactly two faces and every vertex is
    used -- the two conditions that make the face orbit a closed
    polyhedron rather than an arrangement of polygons.
    """
    # The group must actually preserve the vertex set, or every
    # candidate silently fails its orbit and the search returns nothing.
    # That is exactly what a frame mismatch looks like -- `_dodeca()` is
    # not invariant under this package's icosahedral rotations, and the
    # first run of this function on it reported "0 noble facetings" for
    # a solid that is regular and therefore trivially noble.
    idx = {_key(v) for v in V}
    for M in G:
        if {_key(M @ np.array(v, float)) for v in V} != idx:
            raise ValueError('the group does not preserve the vertex set '
                             '-- wrong frame?')

    out = []
    seen = set()
    for nx, _d, on in vertex_planes(V):
        if not min_sides <= len(on) <= max_sides:
            continue
        for cyc in _cycles_on_plane(V, nx, on):
            faces = face_orbit(V, cyc, G)
            if not faces:
                continue
            sig = frozenset(frozenset(_edges(f)) for f in faces)
            if sig in seen:
                continue
            seen.add(sig)
            mult = {}
            for f in faces:
                for e in _edges(f):
                    mult[e] = mult.get(e, 0) + 1
            if set(mult.values()) != {2}:
                continue
            if len({i for f in faces for i in f}) != len(V):
                continue
            out.append((faces, nx, len(on)))
    return out


def describe(V, faces):
    """(V, E, F, chi) for a face list."""
    E = set()
    for f in faces:
        E |= _edges(f)
    return len(V), len(E), len(faces), len(V) - len(E) + len(faces)


def _selftest():
    from . import compounds as _cmp

    # The icosahedron's twelve vertices carry four REGULAR polyhedra --
    # the icosahedron {3,5}, the great dodecahedron {5,5/2}, the small
    # stellated dodecahedron {5/2,5} and the great icosahedron {3,5/2}.
    # Regular implies noble, so a noble faceting search on that vertex
    # set must find all four; anything fewer means the search is missing
    # candidates, anything more that fails Euler means the edge test is
    # too weak.
    V, _F = _cmp._icosa()
    G = _cmp.GROUPS['Ih']()
    found = noble_facetings(V, G)
    got = sorted((len(f[0]), len(f[0][0])) for f in found)
    print('icosahedral vertex set: %d noble facetings' % len(found))
    for faces, _nx, k in found:
        print('   %2d faces of %d sides -> V=%d E=%d F=%d chi=%d'
              % (len(faces), len(faces[0]), *describe(V, faces)))
    assert len(found) >= 4, ('expected at least the four regulars',
                             len(found))

    # the icosahedron itself: 20 triangles, and the two dodecahedral
    # ones: 12 pentagons or pentagrams
    counts = sorted(len(f[0]) for f in found)
    assert 20 in counts, ('no 20-face solid (the icosahedron)', counts)
    assert counts.count(12) >= 2, \
        ('expected two 12-face solids (great dodecahedron and small '
         'stellated dodecahedron)', counts)

    # every survivor must be a closed surface
    for faces, _nx, _k in found:
        nv, ne, nf, chi = describe(V, faces)
        assert chi in (2, -6, -4, 0), (nv, ne, nf, chi)
        mult = {}
        for f in faces:
            for e in _edges(f):
                mult[e] = mult.get(e, 0) + 1
        assert set(mult.values()) == {2}, 'edge used other than twice'

    # A faceting keeps the vertices: that is the whole definition, and
    # it is what makes this the right tool for the noble duals, where
    # polar reciprocation of an emitted boundary is not.
    for faces, _nx, _k in found:
        assert {i for f in faces for i in f} == set(range(len(V)))

    # The frame guard.  `_dodeca()` is a turned copy, not the dual of
    # `_icosa()` in this frame, so the icosahedral group does not
    # preserve it -- and without the guard that shows up as "0 noble
    # facetings" for a solid that is regular and so trivially noble.
    bad = _cmp._dodeca()[0]
    try:
        noble_facetings(bad, G)
    except ValueError:
        pass
    else:
        raise AssertionError('the frame guard did not fire')

    # The dodecahedron built in the RIGHT frame -- the icosahedron's face
    # centres -- carries five noble facetings, among them the two
    # classical regulars on that vertex set: the dodecahedron {5,3} and
    # the great stellated dodecahedron {5/2,3}, both with chi = 2.
    IV, IF = _cmp._icosa()
    D = []
    for face in IF:
        m = np.array([sum(IV[i][k] for i in face) / len(face)
                      for k in range(3)])
        D.append(tuple(m / np.linalg.norm(m)))
    dfound = noble_facetings(D, G)
    print('dodecahedral vertex set: %d noble facetings' % len(dfound))
    assert len(dfound) >= 4, len(dfound)
    reg = [f for f in dfound if describe(D, f[0])[3] == 2]
    assert len(reg) == 2, ('expected the two regulars', len(reg))
    assert all(len(f[0]) == 12 and len(f[0][0]) == 5 for f in reg), \
        'the regulars should be twelve five-sided faces'
    print('RESULT: OK')
