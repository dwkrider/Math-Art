"""seifert -- Seifert surfaces of braid closures, built and relaxed.

A trimmed, NumPy-only implementation of the Seifert-surface construction of

    J. J. van Wijk and A. M. Cohen, "Visualization of Seifert Surfaces",
    IEEE Transactions on Visualization and Computer Graphics 12(4), 2006.

Pipeline: braid word -> disk/band combinatorics -> quad mesh -> (refine) ->
relax -> fair.  Topology is fixed by the first two steps and every later step is
required to preserve it; :meth:`Mesh.info` is the check.  Seifert's algorithm is
one Kauffman resolution among 2**c; the others give the non-orientable spanning
surfaces of the same link.

This is the copy used by the Math Art Blender add-on.  Only the geometry and
optimisation stages are kept -- they need only NumPy.  The mean-curvature
fairing (:mod:`seifert.fair`) solves the implicit-fairing system with a NumPy
Jacobi iteration, so nothing here needs SciPy or scikit-image.
"""

from .braid import Braid, Crossing, torus_knot
from .build import (
    SurfaceParams, spanning_surface, seifert_surface, state_surface,
)
from .fair import biharmonic_fair, cmcf_fair, minimal_surface, smooth_boundary
from .mesh import Mesh, MeshInfo
from .pipeline import finish, surface
from .relax import RelaxParams, relax
from .states import (
    State, StateSurface, minimal_crosscap_state, seifert_state, state_data,
    turnback_state,
)
from .subdivide import catmull_clark

__all__ = [
    "Braid",
    "Crossing",
    "torus_knot",
    "Mesh",
    "MeshInfo",
    "SurfaceParams",
    "spanning_surface",
    "seifert_surface",
    "state_surface",
    "State",
    "StateSurface",
    "state_data",
    "seifert_state",
    "turnback_state",
    "minimal_crosscap_state",
    "RelaxParams",
    "relax",
    "finish",
    "surface",
    "catmull_clark",
    "minimal_surface",
    "smooth_boundary",
    "biharmonic_fair",
    "cmcf_fair",
]

__version__ = "1.1.0"
