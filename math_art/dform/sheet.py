# Meshing the flat pieces, and the topology the solver needs.
#
# Part of the Math Art D-form engine (`math_art/dform/`).  Python and
# numpy only -- no `bpy`.
#
# The flat piece is triangulated with CONCENTRIC RINGS: the boundary ring
# (the seam samples), then shrunken copies of it stepping inward, capped
# by a centre vertex.  A Delaunay triangulation would give slightly
# rounder triangles, but rings win here for three reasons: the boundary
# ring is exactly the seam sampling with no extra vertices on it (so the
# two pieces glue vertex-to-vertex by construction), the ring index is a
# ready-made distance-to-boundary field for the initial pop, and the
# whole thing is O(n) with no degeneracy handling.
#
# Every edge's length in this FLAT layout is its rest length: that array
# is the developable metric, and enforcing it is what makes the solved
# surface isometric to paper.

import numpy as np

from .curves import polygon_area, resample_arclength


def _stitch(outer, inner):
    """Triangles between two closed index loops of unequal length.

    Both loops are parameterised over [0,1) starting at the same place;
    the walk advances whichever loop has the nearer next sample, which
    keeps triangles well shaped when the counts differ.
    """
    p, q = len(outer), len(inner)
    tris = []
    i = j = 0
    while i < p or j < q:
        adv_o = ((i + 1) / p) <= ((j + 1) / q) if i < p else False
        if j >= q:
            adv_o = True
        if adv_o:
            tris.append((outer[i % p], outer[(i + 1) % p], inner[j % q]))
            i += 1
        else:
            tris.append((outer[i % p], inner[(j + 1) % q], inner[j % q]))
            j += 1
    return tris


def disc_mesh(boundary, spacing=None):
    """Ring-triangulate a convex region given its boundary ring.

    `boundary` is (n,2), counter-clockwise, uniform in arclength -- the
    seam samples.  Returns (V2, tris, seam, depth, param):
        V2     (m,2) flat vertex positions, boundary first
        tris   (t,3) counter-clockwise triangles
        seam   the boundary ring indices, 0..n-1, in order
        depth  0 on the boundary, 1 at the centre
        param  position around the ring, 0..1 -- with `depth` this is a
               polar coordinate system on the piece, which is what lets
               the solver start the piece somewhere convex
    """
    B = np.asarray(boundary, dtype=float)
    n = len(B)
    c = B.mean(axis=0)
    R = float(np.mean(np.linalg.norm(B - c, axis=1)))
    if spacing is None:
        seg = np.linalg.norm(np.roll(B, -1, axis=0) - B, axis=1)
        spacing = float(np.mean(seg))
    K = max(1, int(round(R / max(spacing, 1e-9))))

    V = [B]
    depth = [np.zeros(n)]
    param = [np.arange(n) / n]
    loops = [np.arange(n)]
    base = n
    for k in range(1, K):
        f = 1.0 - k / K
        nk = max(3, int(round(n * f)))
        ring = resample_arclength(c + f * (B - c), nk)
        V.append(ring)
        depth.append(np.full(nk, k / K))
        param.append(np.arange(nk) / nk)
        loops.append(np.arange(base, base + nk))
        base += nk
    V.append(c[None, :])
    depth.append(np.array([1.0]))
    param.append(np.array([0.0]))
    centre = base

    tris = []
    for a, b in zip(loops[:-1], loops[1:]):
        tris += _stitch(a, b)
    last = loops[-1]
    for i in range(len(last)):
        tris.append((last[i], last[(i + 1) % len(last)], centre))

    return (np.concatenate(V, axis=0), np.asarray(tris, dtype=int),
            np.arange(n), np.concatenate(depth), np.concatenate(param))


def annulus_mesh(outer, inner, spacing=None):
    """Ring-triangulate an annulus; the SEAM is the inner (hole) ring.

    Used by the anti-D-form, where the two pieces are joined along their
    holes and the outer boundaries stay free.  Both rings are resampled
    to a common count and interpolated, so `inner` supplies the seam
    sampling exactly.
    """
    I = np.asarray(inner, dtype=float)
    O = np.asarray(outer, dtype=float)
    n = len(I)
    O = resample_arclength(O, n)
    # align the outer ring to the inner one by centroid direction so the
    # interpolated rings do not shear
    ci, co = I.mean(axis=0), O.mean(axis=0)
    a_i = np.arctan2(I[0, 1] - ci[1], I[0, 0] - ci[0])
    ang = np.arctan2(O[:, 1] - co[1], O[:, 0] - co[0])
    shift = int(np.argmin(np.abs(((ang - a_i) + np.pi) % (2 * np.pi)
                                 - np.pi)))
    O = np.roll(O, -shift, axis=0)

    gap = float(np.mean(np.linalg.norm(O - I, axis=1)))
    if spacing is None:
        seg = np.linalg.norm(np.roll(I, -1, axis=0) - I, axis=1)
        spacing = float(np.mean(seg))
    K = max(1, int(round(gap / max(spacing, 1e-9))))

    V, depth, param, loops = [], [], [], []
    for k in range(K + 1):
        # uniform spacing.  (Grading the rings denser toward the hole
        # was tried for the anti-D-form's seam boundary layer and made
        # everything WORSE -- the skinny first quads destabilise the
        # hinge and the marching -- so: uniform.)
        s = k / K
        V.append((1.0 - s) * I + s * O)
        depth.append(np.full(n, s))
        param.append(np.arange(n) / n)
        loops.append(np.arange(k * n, (k + 1) * n))

    tris = []
    for a, b in zip(loops[:-1], loops[1:]):
        for i in range(n):
            j = (i + 1) % n
            tris.append((a[i], b[i], b[j]))
            tris.append((a[i], b[j], a[j]))

    return (np.concatenate(V, axis=0), np.asarray(tris, dtype=int),
            np.arange(n), np.concatenate(depth), np.concatenate(param))


def edges_from_tris(tris):
    """Unique undirected edges of a triangle array, as (e,2)."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def interior_edge_flaps(tris):
    """(a, b, c, d) for every edge shared by exactly two triangles.

    (a, b, c) is one triangle IN ITS OWN WINDING and (b, a, d) the other,
    so `cross(b-a, c-a)` is that triangle's true normal.  Keeping the
    direction matters: sort the pair into (min, max) and half the flaps
    come back wound backwards, which silently flips the sign of every
    convexity test built on them.
    """
    rec = {}
    for i, j, k in tris:
        for a, b, o in ((i, j, k), (j, k, i), (k, i, j)):
            rec.setdefault((a, b) if a < b else (b, a), []).append((a, b, o))
    flaps = []
    for ent in rec.values():
        if len(ent) != 2:
            continue
        (a, b, c), (a2, _, d) = ent
        if a2 == a:                      # inconsistent winding: skip it
            continue
        flaps.append((a, b, c, d))
    return np.asarray(flaps, dtype=int).reshape(-1, 4)


def boundary_edges(tris):
    """Edges belonging to exactly one triangle."""
    rec = {}
    for i, j, k in tris:
        for a, b in ((i, j), (j, k), (k, i)):
            key = (a, b) if a < b else (b, a)
            rec[key] = rec.get(key, 0) + 1
    return np.asarray([e for e, c in rec.items() if c == 1],
                      dtype=int).reshape(-1, 2)


def rest_lengths(V2, edges):
    """Edge lengths in the flat layout -- the developable metric."""
    d = V2[edges[:, 1]] - V2[edges[:, 0]]
    return np.linalg.norm(d, axis=1)


def angle_defects(V, tris, n_vert):
    """2*pi minus the incident triangle angles, per vertex.

    Zero at an interior vertex of a developable surface; on a D-form all
    of it is concentrated on the seam.
    """
    defect = np.full(n_vert, 2 * np.pi)
    P = V[tris]
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = P[:, b] - P[:, a]
        v = P[:, c] - P[:, a]
        cosang = np.sum(u * v, axis=1) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-15)
        np.subtract.at(defect, tris[:, a],
                       np.arccos(np.clip(cosang, -1.0, 1.0)))
    return defect


def _selftest():
    ok = True
    from .curves import curve_points, resample_arclength as rs

    B = rs(curve_points('ELLIPSE', aspect=0.6), 96)
    V, T, seam, depth, param = disc_mesh(B)

    # the seam ring must be exactly the input boundary: the two pieces
    # glue vertex-to-vertex, so no extra samples may appear on it
    good = (len(seam) == 96
            and float(np.max(np.abs(V[seam] - B))) < 1e-12)
    ok &= good
    print(f"sheet: seam ring is the input boundary "
          f"{'OK' if good else 'FAIL'}")

    # a triangulated disc: every edge is interior (2 faces) except the
    # boundary ring, and Euler characteristic is 1
    E = edges_from_tris(T)
    bnd = boundary_edges(T)
    chi = len(V) - len(E) + len(T)
    good = chi == 1 and len(bnd) == 96
    ok &= good
    print(f"sheet: disc chi={chi}(1) boundary edges={len(bnd)}(96) "
          f"{'OK' if good else 'FAIL'}")

    # all triangles wound counter-clockwise, and their areas must add up
    # to the polygon area -- a stitch bug shows up as either
    P = V[T]
    cross = ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
             - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
    area = 0.5 * float(np.sum(cross))
    err = abs(area - polygon_area(B)) / abs(polygon_area(B))
    good = bool(np.all(cross > 0)) and err < 1e-12
    ok &= good
    print(f"sheet: triangles CCW, area matches polygon (rel {err:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # a flat sheet has zero angle defect at every interior vertex --
    # the developability baseline the solver has to preserve
    V3 = np.concatenate([V, np.zeros((len(V), 1))], axis=1)
    d = angle_defects(V3, T, len(V))
    inner = np.ones(len(V), bool)
    inner[seam] = False
    worst = float(np.max(np.abs(d[inner])))
    good = worst < 1e-12
    ok &= good
    print(f"sheet: flat interior defect zero ({worst:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # ring spacing must track the seam spacing, or the mesh is anisotropic
    L = rest_lengths(V, E)
    ratio = float(L.max() / L.min())
    good = ratio < 6.0
    ok &= good
    print(f"sheet: edge length ratio {ratio:.2f} (<6) "
          f"{'OK' if good else 'FAIL'}")

    # annulus: two free boundary loops, chi = 0
    O = rs(curve_points('ELLIPSE', aspect=0.8), 96)
    I = O * 0.4
    V, T, seam, depth, param = annulus_mesh(O, I)
    E = edges_from_tris(T)
    chi = len(V) - len(E) + len(T)
    good = chi == 0 and len(boundary_edges(T)) == 2 * 96
    ok &= good
    print(f"sheet: annulus chi={chi}(0) boundary loops 2 "
          f"{'OK' if good else 'FAIL'}")

    # flaps: an interior edge of the disc has two opposite vertices
    V, T, seam, depth, param = disc_mesh(B)
    E = edges_from_tris(T)
    flaps = interior_edge_flaps(T)
    good = len(flaps) == len(E) - len(boundary_edges(T))
    ok &= good
    print(f"sheet: interior flaps={len(flaps)} = edges - boundary "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("dform.sheet self-test failed")
