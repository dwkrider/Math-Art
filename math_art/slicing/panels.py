# Cut layouts for generators that ALREADY produce flat panels.
#
# The slicer earns its layout by cutting a solid into sections.  Some
# generators skip that step entirely: a slide-together is built from
# flat slitted polygons to begin with, and every one of them is
# already a part waiting to be cut.  Such a generator does not need
# slicing at all -- only the second half of the pipeline, which is
# nesting, kerf and the exporters.
#
# So this module is the adapter.  Give it panels as (outline, normal)
# in 3-D and it hands back a Drawing the same exporters understand,
# which means one SVG/DXF path serves both, and a fix to the nesting
# or to the DXF layers reaches every generator that lays parts flat.
#
# The one real piece of work here is FLATTENING.  A panel's outline is
# a 3-D polygon lying in its own plane; the part that gets cut is that
# polygon seen face-on.  Projecting it onto a frame built from its
# normal does that exactly -- distances within the plane are preserved,
# which is the whole requirement for a cut file, since a part that
# comes out of the plane distorted is a part that does not fit.

import math

import numpy as np

from . import sections
from . import parts as _parts
from . import layout as _layout


def flatten(outline, normal):
    """A 3-D planar outline as 2-D points, seen face-on.

    The frame is right-handed with u x v = normal, so every panel is
    laid out viewed from its own +normal side -- the same chirality
    convention the slicer uses, and for the same reason: a flipped
    part is its own mirror image, and a slit cut on the wrong side
    does not meet its partner.
    """
    u, v, n = sections.plane_frame(normal)
    P = np.asarray(outline, dtype=float)
    origin = P.mean(axis=0)
    rel = P - origin
    return [(float(r @ u), float(r @ v)) for r in rel]


def model_extent(panels):
    """Longest dimension of the assembled model, in its own units."""
    pts = np.asarray([p for outline, _n in panels for p in outline],
                     dtype=float)
    if not len(pts):
        return 0.0
    return float((pts.max(axis=0) - pts.min(axis=0)).max())


def panel_layout(panels, target_size=200.0, sheet_width=600.0,
                 sheet_height=400.0, margin=5.0, kerf=0.15,
                 label_height=4.0, name='panels', family='P'):
    """Nest already-flat panels onto sheets.

    `panels` is a sequence of (outline, normal) in 3-D.  `target_size`
    is the longest dimension the ASSEMBLED model should have in
    millimetres -- the panels are scaled together by that one factor,
    so they still fit each other afterwards.  Scaling them
    individually would be the obvious mistake and would quietly
    destroy the model.

    Returns (drawing, report).
    """
    extent = model_extent(panels)
    scale = (target_size / extent) if extent > 0.0 else 1.0

    built = []
    for i, (outline, normal) in enumerate(panels):
        flat = [(x * scale, y * scale) for x, y in flatten(outline, normal)]
        part = _parts.Part(flat, [], family, i, 0, 0.0)
        part.label = f"{family}-{i + 1:02d}"
        built.append(part)

    drawing, report = _layout.nest(built, sheet_width, sheet_height,
                                   margin, kerf, label_height, name)
    report['scale'] = scale
    report['panels'] = len(built)
    return drawing, report


# ------------------------------------------------------------------ #

def _selftest():
    from . import polyclip as pc

    # a unit square in the z = 0 plane, and the same square stood up in
    # the x = 0 plane: both must flatten to the SAME 2-D part
    flat_sq = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
               (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)]
    up_sq = [(0.0, -1.0, -1.0), (0.0, 1.0, -1.0),
             (0.0, 1.0, 1.0), (0.0, -1.0, 1.0)]
    a = flatten(flat_sq, (0.0, 0.0, 1.0))
    b = flatten(up_sq, (1.0, 0.0, 0.0))
    assert abs(pc.area(a) - 4.0) < 1e-9, f"flattened area {pc.area(a)}"
    assert abs(pc.area(b) - 4.0) < 1e-9, \
        "a panel standing in another plane flattens to the same part"

    # flattening preserves distances within the plane -- a part that
    # comes out distorted is a part that does not fit
    for ring, src in ((a, flat_sq), (b, up_sq)):
        for k in range(len(ring)):
            j = (k + 1) % len(ring)
            d2 = math.dist(ring[k], ring[j])
            d3 = math.dist(src[k], src[j])
            assert abs(d2 - d3) < 1e-9, f"edge {k}: {d2} vs {d3}"

    # --- one scale for the whole model ----------------------------
    panels = [(flat_sq, (0.0, 0.0, 1.0)), (up_sq, (1.0, 0.0, 0.0))]
    assert abs(model_extent(panels) - 2.0) < 1e-9, model_extent(panels)
    d, rep = panel_layout(panels, target_size=100.0, sheet_width=400.0,
                          sheet_height=300.0, kerf=0.0, label_height=0.0)
    assert rep['panels'] == 2, rep
    assert abs(rep['scale'] - 50.0) < 1e-9, f"one scale factor: {rep['scale']}"
    assert rep['placed'] == 2 and rep['oversize'] == 0, rep
    assert not d.bounds_ok(), f"panels must fit the sheet: {d.bounds_ok()}"

    cuts = [e for s in d.sheets for e in s.entities if e.layer == 'CUT']
    assert len(cuts) == 2, f"one outline per panel, got {len(cuts)}"
    for e in cuts:
        assert abs(pc.area(e.points) - 100.0 * 100.0) < 1e-6, \
            "each 2x2 square becomes 100 x 100 mm at this scale"

    # a panel too big for the stock is reported, not silently shrunk
    _d2, rep2 = panel_layout(panels, target_size=5000.0,
                             sheet_width=100.0, sheet_height=100.0)
    assert rep2['oversize'] == 2 and rep2['placed'] == 0, rep2

    return True
