# solver -- shared numerical core for the iterative geometry solvers.
#
# The Blender-FREE machinery common to the relaxation loops scattered
# through `math_art/` (minsurf plateau, seifert fairing, polyhedra
# canonical forms, ...).  NumPy only -- no `bpy` -- so everything here
# imports and self-tests headlessly, exactly like the other engine
# subpackages (`minsurf/`, `seifert/`, `polyhedra/`).
#
# Members (each adopted by a solver only where the benchmark in
# `tests/bench/` showed a win or a tie -- see
# `research/plans/solver-unification-bench-plan.md`):
#
#   cotan    cotangent-Laplacian weight assembly, with intrinsic
#            mollification as the principled replacement for ad-hoc
#            positivity clamps.
#   descent  1-D step-size control: Evolver-style bracketed parabola
#            line search and an adaptive-gain controller.
#   groom    in-loop mesh quality maintenance: Delaunay edge flips and
#            tangential vertex smoothing.
#   volume   volume-constrained evolution: body volumes and gradients on
#            region-pair-labeled meshes, Gram-matrix Lagrange velocity
#            projection, damped Newton volume restore, and the `evolve`
#            driver composing all of the above (Evolver fixvol scheme +
#            LosTopos labels).
#
# The per-solver descent loops themselves deliberately stay bespoke
# where measurement showed the bespoke loop stronger (the Pinkall-
# Polthier solve in `minsurf/plateau.py` is the canonical example).

from . import cotan  # noqa: F401
from . import descent  # noqa: F401
from . import groom  # noqa: F401
from . import volume  # noqa: F401
