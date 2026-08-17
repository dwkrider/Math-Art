# Unrolling a D-form back to flat, and reading the curvature off its seam.
#
# Part of the Math Art D-form engine (`math_art/dform/`).  Python and
# numpy only -- no `bpy`.
#
# THE DEVELOPMENT IS FREE HERE, AND EXACT.  Unrolling a general mesh is
# an approximation problem (cut it, flatten it, live with the distortion)
# and the repo's net exporter has to solve it that way.  A D-form is the
# opposite case: the flat pieces are the INPUT.  The solver never touches
# them -- it carries each piece's 2D layout and the local->global vertex
# map alongside the 3D state -- so the development is not recovered from
# the solved surface at all, it is simply handed back.  It is therefore
# exact by construction, and the only thing worth checking is that the
# 3D surface still matches it, which is the isometry gate below.
#
# WHAT THE SEAM CARRIES.  All of a D-form's Gaussian curvature sits on
# the seam, and Orduno et al. (Bridges 2016) make that quantitative: at
# a seam point where the two glued boundaries turn through theta_0 and
# theta_1,
#
#     mu(t) = 2*pi - theta_0(t) - theta_1(t)
#     sum over the seam of mu = 4*pi        (Gauss-Bonnet, genus 0)
#
# so moving the join point cannot create or destroy curvature, only
# REDISTRIBUTE it -- which is exactly why the join offset is the
# interesting artistic knob rather than a nuisance parameter.  `mu` is
# returned per seam vertex so the generator can colour the seam by it.
#
# References:
#   R. R. Orduno, N. Winard, S. Bierwagen, D. Shell, N. Kalantar,
#       A. Borhani, E. Akleman, "A Mathematical Approach to Obtain
#       Isoperimetric Shapes for D-Form Construction," Bridges 2016,
#       pp. 277-284 -- Eq. 1 and Eq. 3.
#   T. Wills, "D-Forms: 3D Forms from Two 2D Sheets," Bridges 2006.

import numpy as np

from .curves import interior_angles, polygon_area
from .sheet import angle_defects, edges_from_tris


def flat_pieces(g):
    """Each piece's exact flat layout as (V2, tris), centred on itself."""
    out = []
    for V2, T in g.flat:
        V2 = np.asarray(V2, dtype=float)
        c = 0.5 * (V2.max(axis=0) + V2.min(axis=0))
        out.append((V2 - c, np.asarray(T, dtype=int)))
    return out


def pack(pieces, gap=0.08):
    """Lay the pieces out side by side, left to right, without overlap.

    Bounding-box packing along one row.  A D-form has two pieces (four
    at the very most, once Route II lands), so a shelf packer would be
    ceremony: the row is already the layout a person cuts from.
    """
    out = []
    x = 0.0
    for V2, T in pieces:
        w = float(V2[:, 0].max() - V2[:, 0].min()) if len(V2) else 0.0
        shift = np.array([x - float(V2[:, 0].min()), 0.0]) if len(V2) else 0.0
        out.append((V2 + shift, T))
        x += w + gap
    if out:
        span = 0.5 * (max(float(V[:, 0].max()) for V, _ in out)
                      + min(float(V[:, 0].min()) for V, _ in out))
        out = [(V - np.array([span, 0.0]), T) for V, T in out]
    return out


def development(g, gap=0.08):
    """The flat layout as one 3D mesh in the z=0 plane: (verts, faces).

    Pieces keep their own vertex numbering, concatenated -- the seam is
    NOT welded here, because the whole point of the development is that
    the pieces are separate things to cut out.
    """
    verts = []
    faces = []
    base = 0
    for V2, T in pack(flat_pieces(g), gap):
        verts.append(np.concatenate([V2, np.zeros((len(V2), 1))], axis=1))
        faces.append(np.asarray(T, dtype=int) + base)
        base += len(V2)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return (np.concatenate(verts, axis=0),
            np.concatenate(faces, axis=0))


def isometry_error(g):
    """Worst relative edge-length error between the 3D form and its net.

    `g.rest` IS the flat layout's edge lengths, so this is the honest
    statement of "the paper did not stretch".
    """
    d = g.V[g.edges[:, 1]] - g.V[g.edges[:, 0]]
    L = np.linalg.norm(d, axis=1)
    return float(np.max(np.abs(L / np.maximum(g.rest, 1e-15) - 1.0)))


def area_error(g):
    """Relative difference between the 3D surface area and the flat area."""
    P = g.V[g.tris]
    a3 = 0.5 * float(np.sum(np.linalg.norm(
        np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)))
    a2 = 0.0
    for V2, T in g.flat:
        Q = np.asarray(V2, dtype=float)[np.asarray(T, dtype=int)]
        a2 += 0.5 * float(np.sum(np.abs(
            (Q[:, 1, 0] - Q[:, 0, 0]) * (Q[:, 2, 1] - Q[:, 0, 1])
            - (Q[:, 1, 1] - Q[:, 0, 1]) * (Q[:, 2, 0] - Q[:, 0, 0]))))
    return abs(a3 - a2) / max(a2, 1e-15)


def seam_mu(g):
    """Angle-defect density mu at each seam vertex, from the 3D mesh.

    This is the measured curvature of the solved surface, so it doubles
    as a check on the flat prediction `curves.mu_density`: the two agree
    when the solve has converged.
    """
    return angle_defects(g.V, g.tris, len(g.V))[g.seam]


def _boundary_interior_angles(V2, mp, n):
    """Interior angles of a piece's seam ring, whichever way it winds.

    `interior_angles` is pi minus the turning angle, which is the
    INTERIOR angle only for a counter-clockwise traversal; walk the same
    polygon clockwise and it returns the 2*pi-complements instead, so the
    defects come out summing to 0 rather than 4*pi.  Piece B is walked in
    the order it was glued -- reversed by default -- and whether that
    ends up counter-clockwise depends on whether the Procrustes fit in
    `glue` chose a reflection, which it does only when a reflection fits
    better.  So the winding is checked here rather than assumed.
    """
    Q = np.asarray(V2)[np.argsort(mp)[:n]]
    if polygon_area(Q) >= 0.0:
        return interior_angles(Q)
    return interior_angles(Q[::-1])[::-1]


def mu_from_flat(g):
    """The same mu predicted from the FLAT outlines (Orduno Eq. 1).

    Exact and solver-independent -- it depends only on the two outlines
    and the join point.  For an anti-D-form (`hole_seam`) the material
    lies OUTSIDE the seam polygon, so the material angle at a seam
    vertex is 2*pi minus the polygon's interior angle; the total then
    comes out -4*pi (two free rims each carry 2*pi of boundary turning,
    and Gauss-Bonnet on the chi = 0 collar balances them).
    """
    (A2, _), (B2, _) = g.flat
    n = len(g.seam)
    # global seam index i is local vertex argsort(map)[i] in each piece,
    # so this walks both boundaries in the order they were glued
    ia = _boundary_interior_angles(A2, g.maps[0], n)
    ib = _boundary_interior_angles(B2, g.maps[1], n)
    if getattr(g, 'hole_seam', False):
        ia = 2 * np.pi - ia
        ib = 2 * np.pi - ib
    return 2 * np.pi - ia - ib


def report(g):
    """Everything the generator wants to show the user, in one dict."""
    mu = seam_mu(g)
    return {
        'strain': isometry_error(g),
        'area_error': area_error(g),
        'mu': mu,
        'mu_total': float(np.sum(mu)),
        'interior_defect': g.interior_defect(),
        'aspect': g.aspect(),
        'iterations': g.iterations,
    }


def _selftest():
    ok = True
    from .curves import (curve_points, match_perimeter, perimeter,
                         resample_arclength, seam_correspondence)
    from .sheet import disc_mesh
    from .solve import glue, settle

    n = 60
    A = resample_arclength(curve_points('ELLIPSE', aspect=0.6), n)
    B = resample_arclength(curve_points('EGG', aspect=0.8, egg=0.3), n)
    B, _ = match_perimeter(A, B)
    g = settle(glue(disc_mesh(A), disc_mesh(B), seam_correspondence(n, 0.3)))

    # the development must be the flat input, to the letter: every edge
    # of the net has exactly the length the solver used as a rest length
    pieces = flat_pieces(g)
    worst = 0.0
    for (V2, T), (V0, _) in zip(pieces, g.flat):
        E = edges_from_tris(np.asarray(T, dtype=int))
        L1 = np.linalg.norm(V2[E[:, 1]] - V2[E[:, 0]], axis=1)
        L0 = np.linalg.norm(np.asarray(V0)[E[:, 1]]
                            - np.asarray(V0)[E[:, 0]], axis=1)
        worst = max(worst, float(np.max(np.abs(L1 / L0 - 1.0))))
    good = worst < 1e-12
    ok &= good
    print(f"develop: net is the flat input exactly (rel {worst:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # isometry of the solved form against that net -- the plan's gate
    err = isometry_error(g)
    good = err < 0.01
    ok &= good
    print(f"develop: 3D form isometric to its net (max edge err "
          f"{100*err:.2f}% < 1%) {'OK' if good else 'FAIL'}")

    # area is the integral version of the same statement, and catches a
    # net that is locally right but globally folded over itself
    aerr = area_error(g)
    good = aerr < 0.01
    ok &= good
    print(f"develop: developed area matches ({100*aerr:.2f}% < 1%) "
          f"{'OK' if good else 'FAIL'}")

    # packing must not overlap, or the cut sheet is unusable
    packed = pack(pieces)
    spans = [(float(V[:, 0].min()), float(V[:, 0].max())) for V, _ in packed]
    good = all(spans[i][1] <= spans[i + 1][0] + 1e-12
               for i in range(len(spans) - 1))
    ok &= good
    print(f"develop: packed pieces do not overlap {'OK' if good else 'FAIL'}")

    # the development mesh must carry every triangle of both pieces
    V, F = development(g)
    good = len(F) == sum(len(T) for _, T in g.flat) and len(V) == sum(
        len(v) for v, _ in g.flat)
    ok &= good
    print(f"develop: net mesh has all {len(F)} faces, {len(V)} verts "
          f"{'OK' if good else 'FAIL'}")

    # Gauss-Bonnet on the seam: total mu = 4*pi.  On the SOLVED mesh the
    # interior keeps a little residual defect, so the seam total is 4*pi
    # only to solver tolerance...
    mu = seam_mu(g)
    tot = float(np.sum(mu))
    good = abs(tot - 4 * np.pi) < 0.02 * 4 * np.pi
    ok &= good
    print(f"develop: seam mu sums to 4pi (got {tot:.4f}, "
          f"{100*abs(tot-4*np.pi)/(4*np.pi):.1f}% off) "
          f"{'OK' if good else 'FAIL'}")

    # ... whereas the FLAT prediction is exact, and must be, since it is
    # just the two polygons' interior angles (Orduno Eq. 3)
    tot2 = float(np.sum(mu_from_flat(g)))
    good = abs(tot2 - 4 * np.pi) < 1e-9
    ok &= good
    print(f"develop: flat mu prediction sums to 4pi exactly "
          f"(err {abs(tot2-4*np.pi):.2e}) {'OK' if good else 'FAIL'}")

    # and the two must agree: the solved seam carries the curvature the
    # flat outlines say it should
    rel = float(np.max(np.abs(mu - mu_from_flat(g)))) / float(
        np.max(np.abs(mu_from_flat(g))))
    good = rel < 0.10
    ok &= good
    print(f"develop: measured mu tracks the flat prediction "
          f"(worst {100*rel:.1f}% of peak) {'OK' if good else 'FAIL'}")

    # moving the join REDISTRIBUTES mu without changing the total --
    # the one-slider story the whole feature is built on
    g2 = settle(glue(disc_mesh(A), disc_mesh(B),
                     seam_correspondence(n, 0.0)))
    m1, m2 = mu_from_flat(g), mu_from_flat(g2)
    moved = float(np.max(np.abs(m1 - m2))) / float(np.max(np.abs(m1)))
    same = abs(float(np.sum(m1)) - float(np.sum(m2)))
    good = moved > 0.1 and same < 1e-9
    ok &= good
    print(f"develop: join offset redistributes mu ({100*moved:.0f}% of "
          f"peak) but conserves it (delta {same:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # a sanity check that the flat pieces really are equal-perimeter,
    # which is the precondition the whole construction rests on
    pa = perimeter(np.asarray(g.flat[0][0])[np.argsort(g.maps[0])[:n]])
    pb = perimeter(np.asarray(g.flat[1][0])[np.argsort(g.maps[1])[:n]])
    good = abs(pa - pb) / pa < 1e-9
    ok &= good
    print(f"develop: the two nets have equal perimeter "
          f"(rel {abs(pa-pb)/pa:.2e}) {'OK' if good else 'FAIL'}")

    # areas are positive and the pieces are not degenerate
    good = all(abs(polygon_area(np.asarray(V2)[np.argsort(mp)[:n]])) > 1e-6
               for (V2, _), mp in zip(g.flat, g.maps))
    ok &= good
    print(f"develop: both pieces enclose real area "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("dform.develop self-test failed")
