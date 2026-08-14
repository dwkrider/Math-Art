"""curve_frames -- orienting a cross-section as it travels along a curve.

The Blender-free moving-frame engine.  Every generator that sweeps a
profile along a space curve -- knots, ruled surfaces, space-filling
curves, tree limbs, interlace cords, tubes of any kind -- needs a moving
reference frame, and the obvious choice is broken: the Frenet frame is
undefined wherever curvature vanishes and flips through 180 degrees at
every inflection.

    tangents   unit tangents of a sampled curve, open or closed
    kernels    the two rotation-minimizing transports: axis-angle
               (second order) and double reflection (fourth order)
    frames     (heading, left, up) frames in three modes, plus the
               closed-curve case and its holonomy
    sweep      profile sweeps, closed tubes, and the welded tube

CLOSED CURVES ARE THE SUBTLE CASE.  A frame carried once around a loop
does not come back to itself: it returns rotated about the tangent by the
curve's holonomy.  Distributed evenly the tube is clean; ignored, the
whole residual lands on one ring join and folds the surface through
itself.  That defect shipped in three separate generators before the
correction was shared, which is the reason this engine exists.

This was `turtle_frame.py` until the knot extraction, after the
differential-turtle-geometry formulation of `frames`.  That name read as
an L-system feature, so several generators grew their own tube sweeps
instead of finding it; the provenance is kept in `frames.py`'s references
where it belongs.
"""

from .frames import (FIXED_UP, FRENET, PARALLEL, closed_frames,
                     closure_holonomy, frames)
from .kernels import transport_normals
from .sweep import closed_tube, sweep, welded_tube
from .tangents import closed_tangents, tangents

__all__ = [
    "tangents",
    "closed_tangents",
    "transport_normals",
    "frames",
    "closed_frames",
    "closure_holonomy",
    "PARALLEL",
    "FRENET",
    "FIXED_UP",
    "sweep",
    "closed_tube",
    "welded_tube",
]

__version__ = "1.0.0"
