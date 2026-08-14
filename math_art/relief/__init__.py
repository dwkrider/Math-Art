"""relief -- relief panels whose height is a field h(x, y).

A relief panel is a scalar field sampled on a rectangle (or a masked region of
one) and lifted into a surface: a plate carved by mathematics rather than
assembled from parts.  The engine is deliberately compositional -- a panel is a
*stack* of pattern layers rather than one closed-form surface -- so that the
organic and the geometric can be mixed rather than chosen between.

Pipeline::

    grid      sample the panel outline, with square cells         (grid)
    orient    an optional direction field the layers follow       (warp)
    warp      displace the coordinates fields are evaluated at    (warp)
    field     evaluate the pattern                                (fields)
    transfer  normalise, then reshape the profile                 (transfer)
    depth     scale to the requested relief depth                 (transfer)
    mesh      lift to a sheet or a watertight slab, and fit       (mesh)

Everything here is NumPy only -- no `bpy` -- so the mathematics is testable
headlessly; `relief_generator.py` is the thin Blender layer over it.

The single design idea worth stating up front: **ridges that fan, split and
sweep are an orientation phenomenon, not an amplitude one.**  A wave with one
global direction cannot produce them however its amplitude is modulated, which
is why the direction field and the phase solve are core rather than optional.

References:
  Robert T. Frankot and Rama Chellappa, "A Method for Enforcing Integrability
    in Shape from Shading Algorithms", IEEE Trans. PAMI 10(4), 1988, 439-451.
  Felix Knoeppel, Keenan Crane, Ulrich Pinkall and Peter Schroeder, "Stripe
    Patterns on Surfaces", ACM TOG 34(4) (SIGGRAPH 2015), art. 39.
  Ken Perlin, "An Image Synthesizer", Computer Graphics 19(3), 1985, 287-296.
  Jerry Tessendorf, "Simulating Ocean Water", SIGGRAPH course notes, 1999-2004.
  Mary D. Waller, "Chladni Figures: A Study in Symmetry", G. Bell and Sons,
    1961.
"""

from . import fields, grid, mesh, transfer, warp
from .fields import FIELDS, evaluate
from .grid import border_window, make_grid, mask_for, SHAPES
from .mesh import apply_fit, edge_report, FITS, FORMS, sheet, slab
from .transfer import apply_curve, CURVES, normalize, NORMS, to_depth
from .warp import (domain_warp, orientation_field, ORIENTATIONS,
                   phase_from_direction, smooth_field)

__all__ = [
    "fields", "grid", "mesh", "transfer", "warp",
    "FIELDS", "evaluate",
    "SHAPES", "make_grid", "mask_for", "border_window",
    "CURVES", "NORMS", "normalize", "apply_curve", "to_depth",
    "ORIENTATIONS", "orientation_field", "domain_warp",
    "phase_from_direction", "smooth_field",
    "FORMS", "FITS", "sheet", "slab", "apply_fit", "edge_report",
    "PRESETS", "build_relief",
]


# Named starting points.  Each is a partial parameter dict merged over the
# defaults of `build_relief`; none of them locks anything.
PRESETS = {
    'DRAPERY': dict(
        field='WAVE_TRAIN', count=3, wavelength=0.55, steepness=0.55,
        orient='CURL', orient_freq=0.45, warp=0.22, warp_iters=2,
        curve='RIDGE', depth=0.3, seed=7),
    'DUNES': dict(
        field='FBM', method='FBM', hurst=0.75, warp=0.3, warp_iters=2,
        curve='RIDGE', depth=0.28, seed=3),
    'POND': dict(
        field='RIPPLE', sources=5, wavelength=0.3, warp=0.0,
        curve='NONE', depth=0.18, seed=11),
    'TERRAIN': dict(
        field='FBM', method='WEIERSTRASS', dim=2.35, warp=0.12,
        curve='NONE', depth=0.35, seed=1),
    'BANDS': dict(
        field='WAVE', wavelength=0.4, orient='SPIRAL', swirl=1.6,
        warp=0.08, curve='SCURVE', curve_amount=0.7, depth=0.22, seed=5),
}


def build_relief(**kw):
    """Build one relief panel.  Returns `(verts, faces, info)`.

    Keyword arguments are the union of the panel, pattern, warp, transfer and
    output controls; see `PRESETS` for worked combinations.  `info` reports the
    values actually realised -- resolution and aspect are both subject to
    adjustment (the sample cap and the square-cell rule), and silently
    differing from what was asked is how a generator earns distrust.
    """
    p = dict(
        # panel
        shape='RECT', width=2.0, aspect=1.0, resolution=256, border=0.0,
        # pattern
        field='WAVE_TRAIN', wavelength=0.5, angle=0.0, phase=0.0,
        steepness=0.0, count=3, spread=0.4, sources=3, seed=1,
        method='FBM', hurst=0.7, dim=2.3, octaves=8, lacunarity=2.0,
        modes=240,
        # orientation + warp
        orient='CONSTANT', orient_freq=0.5, swirl=1.0,
        warp=0.0, warp_iters=2, warp_freq=0.6,
        # transfer
        norm='STD', curve='NONE', curve_amount=1.0, levels=6, terrace=0.25,
        # output
        depth=0.25, form='SLAB', base_thickness=0.1,
        fit='FOOTPRINT', scale=1.0, span=2.0,
    )
    p.update(kw)

    X, Y, info = grid.make_grid(p['width'], p['aspect'], p['resolution'])
    mask = grid.mask_for(p['shape'], X, Y)

    Xw, Yw = warp.domain_warp(X, Y, p['warp'], seed=int(p['seed']) + 1013,
                              iterations=p['warp_iters'], freq=p['warp_freq'])

    h = fields.evaluate(p['field'], Xw, Yw, info, p)
    h = transfer.normalize(h, p['norm'])
    h = transfer.apply_curve(h, p['curve'], amount=p['curve_amount'],
                             levels=p['levels'], smooth=p['terrace'])
    h = transfer.normalize(h, 'MINMAX')
    if p['border'] > 0.0:
        h = h * grid.border_window(X, Y, p['border'], p['shape'])
    Z = 0.5 * float(p['depth']) * h

    if p['form'] == 'SHEET':
        verts, faces = mesh.sheet(X, Y, Z, mask)
    else:
        verts, faces = mesh.slab(X, Y, Z, mask, p['base_thickness'])
    verts = mesh.apply_fit(verts, p['fit'], p['scale'], p['span'])

    lam_cells = grid.wavelength_in_cells(p['wavelength'], info)
    info = dict(info)
    info.update(verts=len(verts), faces=len(faces),
                wavelength_cells=lam_cells,
                aliasing=lam_cells < 4.0)
    return verts, faces, info


def _selftest():
    ok = True
    for name, preset in sorted(PRESETS.items()):
        v, f, info = build_relief(resolution=97, **preset)
        open_e, nonman, oriented = mesh.edge_report(f)
        import numpy as _np
        ext = _np.asarray(v).max(axis=0) - _np.asarray(v).min(axis=0)
        print("relief: %-8s V=%-6d F=%-6d open=%d nonman=%d bbox=%s"
              % (name, len(v), len(f), open_e, nonman, _np.round(ext, 3)))
        ok = ok and open_e == 0 and nonman == 0 and oriented
        ok = ok and _np.isfinite(v).all() and len(f) > 0
        ok = ok and abs(max(ext[0], ext[1]) - 2.0) < 1e-9   # FOOTPRINT fit
        ok = ok and ext[2] > 1e-3                            # actual relief

    # A sheet is open; a slab is not.
    _, fs, _ = build_relief(resolution=61, form='SHEET')
    oe, _, _ = mesh.edge_report(fs)
    ok = ok and oe > 0
    print("relief: SHEET open edges = %d (expected, it is not closed)" % oe)

    # The aliasing guard notices a wavelength below the Nyquist limit.
    _, _, i_fine = build_relief(resolution=61, field='WAVE', wavelength=0.02)
    _, _, i_ok = build_relief(resolution=61, field='WAVE', wavelength=0.5)
    print("relief: lambda 0.02 -> %.2f cells (alias=%s); 0.5 -> %.2f cells "
          "(alias=%s)" % (i_fine['wavelength_cells'], i_fine['aliasing'],
                          i_ok['wavelength_cells'], i_ok['aliasing']))
    ok = ok and i_fine['aliasing'] and not i_ok['aliasing']

    print("RESULT:", "OK" if ok else "BAD")
    assert ok
