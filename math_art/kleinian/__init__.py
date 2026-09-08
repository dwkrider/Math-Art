# Kleinian groups and their limit sets.
#
# Two Moebius transformations generate a group; the closure of an orbit is the
# LIMIT SET, a fractal that ranges from a round circle to a wild Jordan curve to
# a gasket of tangent circles as the two generating traces vary.  Grandma's
# recipe parameterises the whole family by those traces, so the deformation
# space is three complex numbers, and the once-punctured-torus groups are the
# slice on which the commutator is parabolic.
#
# The Apollonian gasket already in `math_art/ifs/inversive.py` is one very
# special member of this family; this package is the general case.
#
# Blender-free: Python only.  The registered operator is
# `kleinian_generator.py`.
#
#   mobius     2x2 complex matrices, classification, fixed points
#   recipes    Grandma's recipe, the Maskit slice, presets
#   limitset   the depth-first search, and circle orbits
#   slice      the Maskit slice boundary: trace recursion, Farey walk, Newton
#
# References:
# - David Mumford, Caroline Series and David Wright, "Indra's Pearls: The Vision
#   of Felix Klein", Cambridge University Press, 2002.
# - Bernard Maskit, "Kleinian Groups", Springer Grundlehren 287, 1988.
# - Alan F. Beardon, "The Geometry of Discrete Groups", Springer GTM 91, 1983.
# - David J. Wright, "Searching for the cusp", in "Spaces of Kleinian Groups",
#   London Mathematical Society Lecture Note Series 329, Cambridge University
#   Press, 2006, pp. 301-336.

from .limitset import circle_orbit, limit_set, limit_set_preset
from .mobius import (ELLIPTIC, IDENT, LOXODROMIC, PARABOLIC, apply,
                     attracting_fixed_point, classify, det, fixed_points, inv,
                     is_parabolic, mat, mul, normalise, trace, word)
from .recipes import (FAMILIES, GRANDMA, MASKIT, PRESETS, commutator_trace,
                      generators, grandma, markov_residual, maskit)
from .slice import boundary, newton_cusp, next_farey, trace_and_rate, trace_poly

__all__ = [
    'mat', 'mul', 'inv', 'det', 'trace', 'normalise', 'apply', 'word',
    'fixed_points', 'attracting_fixed_point', 'classify', 'is_parabolic',
    'IDENT', 'ELLIPTIC', 'PARABOLIC', 'LOXODROMIC',
    'grandma', 'maskit', 'generators', 'commutator_trace', 'markov_residual',
    'PRESETS', 'FAMILIES', 'GRANDMA', 'MASKIT',
    'limit_set', 'limit_set_preset', 'circle_orbit',
    'trace_poly', 'trace_and_rate', 'next_farey', 'newton_cusp', 'boundary',
]


def _selftest():
    import sys

    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    assert not missing, "kleinian: __all__ names missing: %s" % (missing,)

    a, b, t_ab = grandma(*PRESETS['QUASIFUCHSIAN'])
    assert abs(commutator_trace(a, b) + 2.0) < 1e-9

    pts = limit_set_preset('GASKET', epsilon=0.02, max_depth=18)
    assert len(pts) > 500 and max(abs(z) for z in pts) < 1.0 + 1e-6

    cusps = boundary(denom=6)
    assert len(cusps) > 3 and all(m.imag > 1.0 for (_, _, m) in cusps)

    assert 'bpy' not in sys.modules, "kleinian must import without Blender"
    print("kleinian: %d names resolve; parabolic commutator, gasket in the "
          "unit disc, %d slice cusps. RESULT: OK" % (len(__all__), len(cusps)))
