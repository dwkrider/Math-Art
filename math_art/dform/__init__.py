"""dform -- D-forms: closed solids glued from flat pieces of equal perimeter.

A D-form (Tony Wills, Bridges 2006; popularised by John Sharp) is what you
get by cutting two flat pieces whose outlines have the SAME PERIMETER and
gluing them edge to edge.  The result pops into a closed, smoothly curved
solid whose entire Gaussian curvature lives on the seam, and which is
developable -- literally paper -- everywhere else.

The ANTI mode is Sharp's anti-D-form: cut a HOLE in each of two flat
pieces, match the hole perimeters, and glue hole edge to hole edge with
the outer rims left free.  The result is an open saddle collar carrying
-4*pi of curvature on the seam (the two free rims balance it under
Gauss-Bonnet) -- not convex, so none of the D-form uniqueness theory
applies, and the solver instead works seam-first with a bending energy.

This package is the mathematics, pure Python + numpy and importable
without Blender; `math_art/dform_generator.py` is the operator over it.

    curves.py    the plane-curve library and the equal-perimeter match
    sheet.py     triangulating a flat piece, and the flat rest lengths
    solve.py     gluing, and solving for the shape it pops into
    develop.py   the exact flat development, and the seam's mu(t)

`build_dform` is the whole public surface: one call, one `DForm`.

References:
  T. Wills, "D-Forms: 3D Forms from Two 2D Sheets," Bridges 2006,
      pp. 503-510 -- the invention.
  J. Sharp, "D-Forms and Developable Surfaces," Bridges 2005, pp. 121-128;
      and "D-Forms" (Tarquin, 2009).
  R. R. Orduno et al., "A Mathematical Approach to Obtain Isoperimetric
      Shapes for D-Form Construction," Bridges 2016, pp. 277-284.
  E. Demaine, J. O'Rourke, "Geometric Folding Algorithms" (Cambridge,
      2007), ch. 25 -- the D-form is the convex hull of its seam curve.
"""

import numpy as np

from . import analytic, curves, develop, sheet, solve

# SEAM and ANTI are solved for; the rest are closed-form relatives that
# assemble exactly from cone and cylinder patches (see `analytic.py`)
MODES = ('SEAM', 'ANTI') + analytic.ANALYTIC_KINDS

__all__ = ['DForm', 'MODES', 'analytic', 'build_dform', 'curves', 'develop',
           'sheet', 'solve']


class DForm:
    """A solved D-form: the pieces, the seam, and the flat development.

    `verts`/`faces` are the closed 3D surface; `dev_verts`/`dev_faces`
    are the flat pieces laid out in the z=0 plane, ready to cut.  `mu`
    is the angle defect at each seam vertex -- the shading channel and
    the Gauss-Bonnet check in one array.
    """

    def __init__(self, verts, faces, seam, mu, piece, dev_verts, dev_faces,
                 stats, sharp_edges=None):
        self.verts = verts
        self.faces = faces
        self.seam = seam
        self.mu = mu
        self.piece = piece
        self.dev_verts = dev_verts
        self.dev_faces = dev_faces
        self.stats = stats
        # every edge that should shade sharp.  For a solved D-form that
        # is the seam ring; the closed-form relatives fold the paper as
        # well, so they add their creases here too.
        if sharp_edges is None:
            r = [int(i) for i in seam]
            sharp_edges = [(r[i], r[(i + 1) % len(r)])
                           for i in range(len(r))] if len(r) > 1 else []
        self.sharp_edges = sharp_edges


# Seam resolution above which the solve is done coarse-first.  The
# solver's convergence rate falls off with the mesh width, so past this
# it is both faster AND better to solve small and resample up.
_COARSE = 72


def _solve_at(A, B, n, join_offset, flip, quality):
    """Solve at `n` segments, coarse-first when that is worth doing."""
    def build(m):
        Am = curves.resample_arclength(A, m)
        Bm = curves.resample_arclength(B, m)
        Bm, _ = curves.match_perimeter(Am, Bm)
        return solve.glue(sheet.disc_mesh(Am), sheet.disc_mesh(Bm),
                          curves.seam_correspondence(m, join_offset, flip))

    if n <= _COARSE:
        return solve.settle(build(n), iterations=quality)

    coarse = solve.settle(build(_COARSE), iterations=quality)
    fine = solve.warm_start(build(n), coarse)
    return solve.polish(fine, iterations=max(120, quality // 2))


def _fit(V, scale):
    """Centre on the bounding box and fit a 2*scale cube, as the repo's
    generators all do."""
    if not len(V):
        return V
    ext = float(np.max(V.max(axis=0) - V.min(axis=0)))
    s = (2.0 * scale / ext) if ext > 1e-12 else 1.0
    return (V - 0.5 * (V.max(axis=0) + V.min(axis=0))) * s


def _build_analytic(mode, segments, scale, vesica_u, vesica_h, panels,
                    slide, growth, skew=0.5, hole=0.45, tilt=0.42,
                    lean=1.0, blade_len=1.7, blade_width=0.6, twist=0.0):
    """The closed-form modes: no solver, no iteration, no convergence.

    These carry creases as well as a seam, so `seam` here means "every
    edge that should be shaded sharp" -- the generator marks it the same
    way either route, and for the vesica that includes the two creases.
    """
    def chain(poly):
        return [(int(poly[i]), int(poly[i + 1]))
                for i in range(len(poly) - 1)]

    n = max(24, int(segments))
    if mode == 'VESICA':
        h = None if vesica_h < 0 else vesica_h
        V, F, seam, creases = analytic.build_vesica(
            u=vesica_u, height=h, samples=n, rings=max(4, n // 12))
        sharp = chain(seam)
        for c in creases:
            sharp += chain(c)
        stats = {'u': float(vesica_u),
                 'h_max': analytic.vesica_max_height(vesica_u),
                 'h_circular': analytic.vesica_circular_height(vesica_u),
                 'segments': n}
    elif mode == 'KOMAN_SPIRAL':
        V, F = analytic.build_koman_spiral(
            blades=panels, growth=growth, skew=skew, lean=lean,
            blade_len=blade_len, blade_width=blade_width, twist=twist,
            samples=max(2, n // 14))
        seam, sharp = [], []
        stats = {'blades': int(panels), 'growth': float(growth),
                 'twist': float(twist), 'segments': n}
    else:                                        # KOMAN_CURL
        V, F = analytic.build_koman_curl(
            panels=panels, slide=slide, growth=growth, skew=skew,
            hole=hole, tilt=tilt, arch_samples=max(4, n // 10))
        seam, sharp = [], []
        eff = (analytic.koman_slide_to_close(panels)
               if slide is None or slide < 0 else slide)
        stats = {'turn_per_panel': analytic.koman_turn(eff, 1.0),
                 'slide': eff,
                 'panels': int(panels), 'growth': float(growth),
                 'segments': n}
    stats.update({'strain': 0.0, 'area_error': 0.0, 'interior_defect': 0.0,
                  'mu_total': 4 * np.pi, 'iterations': 0, 'mode': mode})
    V = _fit(np.asarray(V, dtype=float), scale)
    return DForm(V, F, np.asarray(seam, dtype=int), np.zeros(len(seam)),
                 np.zeros(len(V), dtype=int), np.zeros((0, 3)), [],
                 stats, sharp_edges=sharp)


# Seam resolution above which the ANTI solve goes coarse-first.  The
# anti-D-form's convergence falls off with the mesh width even faster
# than the closed solve's (measured: n=96 direct comes out at 7% strain
# where n=64 reaches 1.8%), so past this the collar is solved small and
# resampled up.
_COARSE_ANTI = 64


def _solve_anti(OA, OB, HA, HB, n, join_offset, flip, quality):
    """Solve an anti-D-form at `n` seam segments, coarse-first if big."""
    def build(m):
        HAm = curves.resample_arclength(HA, m)
        HBm = curves.resample_arclength(HB, m)
        HBm, _ = curves.match_perimeter(HAm, HBm)
        OAm = curves.resample_arclength(OA, max(m, 48))
        OBm = curves.resample_arclength(OB, max(m, 48))
        return solve.glue(sheet.annulus_mesh(OAm, HAm),
                          sheet.annulus_mesh(OBm, HBm),
                          curves.seam_correspondence(m, join_offset, flip),
                          closed=False, hole_seam=True)

    # the collar needs a larger budget than the closed solve: the same
    # `quality` knob maps to ~3.8x the iterations (still seconds)
    iters = max(1800, int(3.8 * quality))
    if n <= _COARSE_ANTI:
        return solve.settle_anti(build(n), iterations=iters)

    coarse = solve.settle_anti(build(_COARSE_ANTI), iterations=iters)
    fine = solve.warm_start(build(n), coarse)
    return solve.polish_anti(fine, iterations=max(900, iters // 2))


def build_dform(mode='SEAM', kind_a='ELLIPSE', kind_b='ELLIPSE',
                aspect_a=0.6, aspect_b=1.0, super_n=3.0, egg=0.35,
                sides_a=3, sides_b=5, corner=0.35, cassini=1.6,
                segments=72, join_offset=0.25, flip=False, quality=900,
                scale=1.0, gap=0.08, vesica_u=0.5, vesica_h=-1.0,
                panels=32, slide=-1.0, growth=0.0, skew=0.5,
                hole=0.45, tilt=0.42, lean=1.0, blade_len=1.7,
                blade_width=0.6, twist=0.0,
                hole_kind_a='ELLIPSE', hole_aspect_a=0.6,
                hole_kind_b='ELLIPSE', hole_aspect_b=1.0,
                hole_scale=0.4):
    """Build a D-form and return it.

    `segments` is the seam resolution -- the number of vertices shared by
    the two pieces -- and `quality` the solver iteration budget.  The two
    outlines are matched to a common perimeter automatically (uniform
    scaling of the second, after Orduno et al.), which is the D-form
    precondition and not something the caller has to arrange.

    In ANTI mode `kind_a`/`kind_b` describe the two OUTER rims (free,
    perimeters unconstrained) and `hole_kind_*`/`hole_aspect_*` the two
    holes, whose perimeters are matched -- they become the seam.
    `hole_scale` sizes the holes relative to the rims, clamped so the
    hole always fits inside its piece.
    """
    if mode not in MODES:
        raise ValueError(f"unknown D-form mode {mode!r}")
    if mode in analytic.ANALYTIC_KINDS:
        return _build_analytic(mode, segments, scale, vesica_u=vesica_u,
                               vesica_h=vesica_h, panels=panels,
                               slide=slide, growth=growth, skew=skew,
                               hole=hole, tilt=tilt, lean=lean,
                               blade_len=blade_len,
                               blade_width=blade_width, twist=twist)

    n = max(16, int(segments))
    A = curves.curve_points(kind_a, aspect=aspect_a, super_n=super_n,
                            egg=egg, sides=sides_a, corner=corner,
                            cassini=cassini)
    B = curves.curve_points(kind_b, aspect=aspect_b, super_n=super_n,
                            egg=egg, sides=sides_b, corner=corner,
                            cassini=cassini)
    # stay dense here: `_solve_at` may sample the pair at more than one
    # resolution, and each should come off the curve rather than off an
    # already-decimated copy of it
    A = curves.ensure_ccw(A)
    B = curves.ensure_ccw(B)

    quality = max(120, int(quality))
    if mode == 'ANTI':
        HA = curves.ensure_ccw(curves.curve_points(
            hole_kind_a, aspect=hole_aspect_a, super_n=super_n, egg=egg,
            sides=sides_a, corner=corner, cassini=cassini))
        HB = curves.ensure_ccw(curves.curve_points(
            hole_kind_b, aspect=hole_aspect_b, super_n=super_n, egg=egg,
            sides=sides_b, corner=corner, cassini=cassini))
        # the hole must fit strictly inside its rim, whatever the caller
        # asked for: clamp against the rim's inradius
        hs = min(max(float(hole_scale), 0.1), 0.8)
        for O, H in ((A, HA), (B, HB)):
            r_out = float(np.min(np.linalg.norm(O, axis=1)))
            r_hole = float(np.max(np.linalg.norm(H, axis=1)))
            if r_hole > 1e-12:
                hs = min(hs, 0.8 * r_out / r_hole)
        g = _solve_anti(A, B, HA * hs, HB * hs, n, join_offset, flip,
                        quality)
    else:
        g = _solve_at(A, B, n, join_offset, flip, quality)

    V = g.V
    ext = float(np.max(V.max(axis=0) - V.min(axis=0)))
    s = (2.0 * scale / ext) if ext > 1e-12 else 1.0
    V = (V - 0.5 * (V.max(axis=0) + V.min(axis=0))) * s

    dv, df = develop.development(g, gap=gap)
    if len(dv):
        dv = dv * s

    stats = develop.report(g)
    stats['segments'] = n
    if mode == 'ANTI':
        # the honest extras for an open collar: how folded is it off the
        # seam (pleat watch), and do the two sheets cross each other
        f = g.flaps
        ns = len(g.seam)
        psi = solve.fold_angles(g.V, f,
                                exclude=((f[:, 0] < ns) & (f[:, 1] < ns)))
        deg = np.degrees(np.abs(psi))
        stats['fold_p99'] = float(np.percentile(deg, 99)) if len(deg) else 0.0
        stats['fold_max'] = float(deg.max()) if len(deg) else 0.0
        stats['crossings'] = solve.sheet_crossings(g)
    return DForm(V, g.tris, g.seam, develop.seam_mu(g), g.piece, dv, df,
                 stats)


def _selftest():
    ok = True

    d = build_dform(segments=56, quality=700, kind_b='EGG', aspect_b=0.8,
                    egg=0.3, join_offset=0.3)

    # the headline object: a closed surface, its seam, and a flat net
    good = (len(d.verts) > 0 and len(d.faces) > 0
            and len(d.seam) == 56 and len(d.dev_faces) == len(d.faces))
    ok &= good
    print(f"dform: built V={len(d.verts)} F={len(d.faces)} "
          f"seam={len(d.seam)} net F={len(d.dev_faces)} "
          f"{'OK' if good else 'FAIL'}")

    # the repo's standing gate: fit a 2 m cube, centred, and actually 3D
    ext = d.verts.max(axis=0) - d.verts.min(axis=0)
    cen = 0.5 * (d.verts.max(axis=0) + d.verts.min(axis=0))
    good = (abs(float(np.max(ext)) - 2.0) < 1e-9
            and float(np.max(np.abs(cen))) < 1e-9
            and float(np.min(ext) / np.max(ext)) > 0.1)
    ok &= good
    print(f"dform: centred, fits a 2m cube, aspect "
          f"{float(np.min(ext)/np.max(ext)):.3f} "
          f"{'OK' if good else 'FAIL'}")

    # the net is flat, and scaled with the form
    good = float(np.max(np.abs(d.dev_verts[:, 2]))) < 1e-12
    ok &= good
    print(f"dform: development lies in z=0 {'OK' if good else 'FAIL'}")

    # curvature is on the seam and adds to 4*pi
    good = abs(d.stats['mu_total'] - 4 * np.pi) < 0.02 * 4 * np.pi
    ok &= good
    print(f"dform: seam carries 4pi (got {d.stats['mu_total']:.4f}) "
          f"{'OK' if good else 'FAIL'}")

    # and the paper did not stretch
    good = d.stats['strain'] < 0.01 and d.stats['area_error'] < 0.01
    ok &= good
    print(f"dform: isometric (strain {d.stats['strain']:.4f}, area "
          f"{100*d.stats['area_error']:.2f}%) {'OK' if good else 'FAIL'}")

    # Above `_COARSE` the solve goes coarse-first and resamples up, and
    # that path has to land on the SAME shape -- if it did not, the
    # segment count would silently change the object rather than just
    # its resolution.  Solving the fine mesh directly does not converge
    # at all here (at 240 segments the seam carried 66% of its 4*pi even
    # after 3000 iterations), so this is also the gate on that.
    lo = build_dform(segments=48, quality=700, kind_b='EGG', aspect_b=0.8,
                     egg=0.3, join_offset=0.3)
    hi = build_dform(segments=_COARSE + 40, quality=700, kind_b='EGG',
                     aspect_b=0.8, egg=0.3, join_offset=0.3)
    ext_lo = lo.verts.max(axis=0) - lo.verts.min(axis=0)
    ext_hi = hi.verts.max(axis=0) - hi.verts.min(axis=0)
    shape = float(np.max(np.abs(np.sort(ext_hi) - np.sort(ext_lo))))
    good = (hi.stats['strain'] < 0.015
            and abs(hi.stats['mu_total'] - 4 * np.pi) < 0.03 * 4 * np.pi
            and shape < 0.12)
    ok &= good
    print(f"dform: refined solve agrees with the coarse one "
          f"(extent delta {shape:.3f}, strain {hi.stats['strain']:.4f}, "
          f"seam {hi.stats['mu_total']/(4*np.pi):.3f}x4pi) "
          f"{'OK' if good else 'FAIL'}")

    # ------------------------------------------------------------ ANTI
    a = build_dform(mode='ANTI', segments=56, quality=800,
                    kind_a='ELLIPSE', aspect_a=0.9, kind_b='ELLIPSE',
                    aspect_b=0.9, hole_kind_a='ELLIPSE', hole_aspect_a=0.6,
                    hole_kind_b='ELLIPSE', hole_aspect_b=1.0,
                    join_offset=0.3)

    # the public triple: an open collar, its seam, and a two-annuli net
    good = (len(a.verts) > 0 and len(a.seam) == 56
            and len(a.dev_faces) == len(a.faces))
    ok &= good
    print(f"dform: ANTI built V={len(a.verts)} F={len(a.faces)} "
          f"net F={len(a.dev_faces)} {'OK' if good else 'FAIL'}")

    # the net is two ANNULI: as a disjoint union its Euler
    # characteristic is 0 + 0 -- the check that `develop` really can
    # hand back pieces with two boundary loops
    de = sheet.edges_from_tris(a.dev_faces)
    chi = len(a.dev_verts) - len(de) + len(a.dev_faces)
    good = chi == 0 and float(np.max(np.abs(a.dev_verts[:, 2]))) < 1e-12
    ok &= good
    print(f"dform: ANTI net is two flat annuli (chi={chi}, z=0) "
          f"{'OK' if good else 'FAIL'}")

    # Gauss-Bonnet with free rims: the seam carries -4*pi
    good = abs(a.stats['mu_total'] + 4 * np.pi) < 0.05 * 4 * np.pi
    ok &= good
    print(f"dform: ANTI seam carries -4pi (got {a.stats['mu_total']:.3f}) "
          f"{'OK' if good else 'FAIL'}")

    # isometric, genuinely 3D, centred in the 2m cube
    ext = a.verts.max(axis=0) - a.verts.min(axis=0)
    cen = 0.5 * (a.verts.max(axis=0) + a.verts.min(axis=0))
    good = (a.stats['strain'] < 0.02
            and abs(float(np.max(ext)) - 2.0) < 1e-9
            and float(np.max(np.abs(cen))) < 1e-9
            and float(np.min(ext) / np.max(ext)) > 0.1)
    ok &= good
    print(f"dform: ANTI isometric (strain {a.stats['strain']:.4f}), "
          f"aspect {float(np.min(ext) / np.max(ext)):.3f} "
          f"{'OK' if good else 'FAIL'}")

    # not pleated (the crease budget lives on the seam, not the sheets)
    good = a.stats['fold_p99'] < 65.0 and a.stats['fold_max'] < 135.0
    ok &= good
    print(f"dform: ANTI folds p99 {a.stats['fold_p99']:.1f} max "
          f"{a.stats['fold_max']:.1f} deg; crossings "
          f"{a.stats['crossings']} (reported) {'OK' if good else 'FAIL'}")

    # an unknown mode must fail loudly rather than silently make a pillow
    try:
        build_dform(mode='NOPE', segments=24, quality=60)
        good = False
    except ValueError:
        good = True
    ok &= good
    print(f"dform: unknown mode rejected {'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("dform self-test failed")
