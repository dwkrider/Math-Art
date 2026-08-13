"""knots -- braid words, knot tables, rope relaxation, tubes.

The Blender-FREE engine behind the Math Art knot and link generators.
NumPy/stdlib only, so the whole package imports and self-tests headlessly;
the registered operators stay in their flat generator modules and import
this package.  Layout follows `math_art/seifert/` and `math_art/minsurf/`.

This machinery was not written for this package -- it accumulated inside
`prime_knot_generator.py`, and `link_generator`, `torus_knot_generator` and
`hopf_fibration_generator` had come to import that *generator* to reach it,
including reaching into its registered operator class for the tube builder.
Extracting it here makes the dependency a real one.

    tables      the 249 minimum braid words of Gittings' Table 1, each with
                that paper's published Alexander value at t = 10, which
                makes the row self-verifying.
    braid       words, closure component count, and the embedding of a
                braid closure as 3-D polylines (one, or one per component).
    alexander   the Alexander polynomial through the reduced Burau
                representation; at x = -1 it is the determinant.
    resample    uniform-arclength resampling of closed polylines.
    relax       rope relaxation (repulsion + smoothing) of knots and links,
                plus the linking number of two components.
    curves      closed-form curves: the (p, q) torus link.
    tube        a closed tube swept along a knot, with the closure holonomy
                distributed so the seam matches.
    build       the pipeline: word -> closure -> resample -> relax -> fit.

Typical use:

    from math_art import knots
    P = knots.build_knot('AAAbAb')           # the 6_2 knot as a polyline
    verts, faces = knots.closed_tube(P, 0.08, 16)

References for the mathematics are in each submodule's header; the
principal ones are Artin (1947) for the braid groups, Alexander (1923,
1928) for the closure theorem and the polynomial, Burau (1935) for the
representation, and Gittings (arXiv:math/0401051, 2004) for the table.
"""

from .alexander import alexander_at
from .braid import (braid_closure_loops, braid_closure_points,
                    closure_components, parse_letters)
from .build import build_knot
from .curves import torus_link_components
from .relax import linking_number, relax_knot, relax_link
from .resample import resample_closed, resample_loops
from .tables import KNOTS
from .tube import closed_tube

__all__ = [
    # table
    "KNOTS",
    # braid words and closures
    "parse_letters",
    "closure_components",
    "braid_closure_points",
    "braid_closure_loops",
    # invariants
    "alexander_at",
    "linking_number",
    # polyline machinery
    "resample_closed",
    "resample_loops",
    "relax_knot",
    "relax_link",
    # geometry
    "torus_link_components",
    "closed_tube",
    "build_knot",
]

__version__ = "1.0.0"
