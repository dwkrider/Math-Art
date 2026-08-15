"""dform -- D-forms: closed solids glued from flat pieces of equal perimeter.

A D-form (Tony Wills, Bridges 2006; popularised by John Sharp) is what you
get by cutting two flat pieces whose outlines have the SAME PERIMETER and
gluing them edge to edge.  The result pops into a closed, smoothly curved
solid whose entire Gaussian curvature lives on the seam, and which is
developable -- literally paper -- everywhere else.

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

from . import curves, develop, sheet, solve

MODES = ('SEAM',)

__all__ = ['DForm', 'MODES', 'build_dform', 'curves', 'develop', 'sheet',
           'solve']


class DForm:
    """A solved D-form: the pieces, the seam, and the flat development.

    `verts`/`faces` are the closed 3D surface; `dev_verts`/`dev_faces`
    are the flat pieces laid out in the z=0 plane, ready to cut.  `mu`
    is the angle defect at each seam vertex -- the shading channel and
    the Gauss-Bonnet check in one array.
    """

    def __init__(self, verts, faces, seam, mu, piece, dev_verts, dev_faces,
                 stats):
        self.verts = verts
        self.faces = faces
        self.seam = seam
        self.mu = mu
        self.piece = piece
        self.dev_verts = dev_verts
        self.dev_faces = dev_faces
        self.stats = stats


def build_dform(mode='SEAM', kind_a='ELLIPSE', kind_b='ELLIPSE',
                aspect_a=0.6, aspect_b=1.0, super_n=3.0, egg=0.35,
                sides_a=3, sides_b=5, corner=0.35, cassini=1.6,
                segments=72, join_offset=0.25, flip=False, quality=900,
                scale=1.0, gap=0.08):
    """Build a D-form and return it.

    `segments` is the seam resolution -- the number of vertices shared by
    the two pieces -- and `quality` the solver iteration budget.  The two
    outlines are matched to a common perimeter automatically (uniform
    scaling of the second, after Orduno et al.), which is the D-form
    precondition and not something the caller has to arrange.
    """
    if mode not in MODES:
        raise ValueError(f"unknown D-form mode {mode!r}")

    n = max(16, int(segments))
    A = curves.curve_points(kind_a, aspect=aspect_a, super_n=super_n,
                            egg=egg, sides=sides_a, corner=corner,
                            cassini=cassini)
    B = curves.curve_points(kind_b, aspect=aspect_b, super_n=super_n,
                            egg=egg, sides=sides_b, corner=corner,
                            cassini=cassini)
    A = curves.ensure_ccw(curves.resample_arclength(A, n))
    B = curves.ensure_ccw(curves.resample_arclength(B, n))
    B, _ = curves.match_perimeter(A, B)

    corr = curves.seam_correspondence(n, join_offset, flip)
    g = solve.glue(sheet.disc_mesh(A), sheet.disc_mesh(B), corr)
    solve.settle(g, iterations=max(120, int(quality)))

    V = g.V
    ext = float(np.max(V.max(axis=0) - V.min(axis=0)))
    s = (2.0 * scale / ext) if ext > 1e-12 else 1.0
    V = (V - 0.5 * (V.max(axis=0) + V.min(axis=0))) * s

    dv, df = develop.development(g, gap=gap)
    if len(dv):
        dv = dv * s

    stats = develop.report(g)
    stats['segments'] = n
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
