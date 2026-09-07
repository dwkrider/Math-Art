# Circle packing engine for Math Art.
#
# A circle packing realises a prescribed pattern of tangencies: given a
# triangulation, assign a radius to every vertex so that circles at the ends of
# each edge touch.  The radii are not free -- the combinatorics determines the
# geometry, up to the ambient normalisation.  That rigidity is the discrete
# analogue of the Riemann mapping theorem, and it is why packings are used to
# compute conformal structure rather than merely to draw circles.
#
# Blender-free: Python and numpy only, so the whole package imports and
# self-tests headlessly.  The registered operator lives in
# `circle_packing_generator.py`; the subdivision-tiling generator consumes this
# package through `math_art/subdiv/`.
#
#   angles      the three angle-sum kernels, s-radii, the Uniform Neighbour
#               Model closed-form update
#   complexes   oriented triangulations as flowers, plus stock combinatorics
#   thurston    the radius solver (Collins-Stephenson Table 1)
#   layout      radii -> positions, euclidean and hyperbolic incl. horocycles
#   refine      hexagonal refinement
#   sphere      spherical packings, by remove-a-face and projection
#
# References:
# - Kenneth Stephenson, "Introduction to Circle Packing: The Theory of Discrete
#   Analytic Functions", Cambridge University Press, 2005.
# - Charles R. Collins and Kenneth Stephenson, "A circle packing algorithm",
#   Computational Geometry: Theory and Applications 25 (2003), pp. 233-256.
# - Charles Collins, Gerald L. Orick and Kenneth Stephenson, "A linearized
#   circle packing algorithm", Computational Geometry: Theory and Applications
#   64 (2017), pp. 13-29.
# - Burt Rodin and Dennis Sullivan, "The convergence of circle packings to the
#   Riemann mapping", Journal of Differential Geometry 26 (1987), pp. 349-360.
# - William P. Thurston, "The Geometry and Topology of Three-Manifolds",
#   Princeton lecture notes, 1980, chapter 13.

from .angles import (EUCLIDEAN, GEOMETRIES, HYPERBOLIC, SPHERICAL, angle_sum,
                     corner, corner_euclidean, corner_hyperbolic,
                     corner_spherical, unm_update_euclidean,
                     unm_update_hyperbolic)
from .complexes import (PackingComplex, from_faces, hex_disc, octahedron,
                        square_grid, tetrahedron)
from .layout import (euclid_circle_of_hyp, horocycle_from_tangency,
                     layout_euclidean, layout_hyperbolic, mobius_shift,
                     tangency_error)
from .refine import hex_refine, refine_n
from .sphere import pack_sphere, remove_face, stereographic
from .thurston import (FREE, MAXIMAL, PRESCRIBED, PackResult, h_radius, pack,
                       s_value)

__all__ = [
    # geometries and kernels
    'EUCLIDEAN', 'HYPERBOLIC', 'SPHERICAL', 'GEOMETRIES',
    'corner', 'corner_euclidean', 'corner_hyperbolic', 'corner_spherical',
    'angle_sum', 'unm_update_euclidean', 'unm_update_hyperbolic',
    # combinatorics
    'PackingComplex', 'hex_disc', 'square_grid', 'tetrahedron', 'octahedron',
    'from_faces',
    # solving
    'pack', 'PackResult', 'MAXIMAL', 'PRESCRIBED', 'FREE',
    'h_radius', 's_value',
    # layout
    'layout_euclidean', 'layout_hyperbolic', 'tangency_error', 'mobius_shift',
    'euclid_circle_of_hyp', 'horocycle_from_tangency',
    # refinement and the sphere
    'hex_refine', 'refine_n', 'pack_sphere', 'remove_face', 'stereographic',
]


def pack_and_lay_out(K, geom=EUCLIDEAN, boundary=FREE, boundary_radii=None,
                     aims=None, **kw):
    """Solve and place in one call.

    Returns (centres, radii, result).  For HYPERBOLIC the centres and radii are
    euclidean data in the Poincare disc, ready to draw; for EUCLIDEAN they are
    the packing's own coordinates."""
    res = pack(K, geom=geom, boundary=boundary, boundary_radii=boundary_radii,
               aims=aims, **kw)
    if geom == HYPERBOLIC:
        cen, rad = layout_hyperbolic(K, res.radii)
        pos = [None if c is None else (c.real, c.imag) for c in cen]
        return pos, rad, res
    pos = layout_euclidean(K, res.radii)
    return pos, list(res.radii), res


def _selftest():
    """The facade's contract: every name in __all__ resolves, and a packing
    round-trips end to end."""
    import math
    import sys

    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    assert not missing, "packing: __all__ names missing: %s" % (missing,)

    K, xy = hex_disc(4)
    br = [0.0] * K.nv
    for w in K.boundary:
        x, y = xy[w]
        br[w] = 0.4 * (1.0 + 0.5 * math.cos(2.0 * math.atan2(y, x)))
    pos, rad, res = pack_and_lay_out(K, EUCLIDEAN, PRESCRIBED,
                                     boundary_radii=br)
    assert res.converged and all(p is not None for p in pos)
    assert tangency_error(K, pos, rad) < 1e-8

    posh, radh, resh = pack_and_lay_out(K, HYPERBOLIC, MAXIMAL)
    assert resh.converged
    for w in K.boundary:
        assert abs(math.hypot(*posh[w]) + radh[w] - 1.0) < 1e-6

    assert 'bpy' not in sys.modules, \
        "packing must import without Blender"

    print("packing: %d names in __all__ resolve; euclidean and hyperbolic "
          "packings round-trip. RESULT: OK" % len(__all__))
