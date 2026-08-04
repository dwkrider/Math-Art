"""The default build-to-finished pipeline, with fitted parameters.

The geometric ratios come from a reference surface -- disks of diameter 60
spaced 83 apart, bands 20 wide -- and the finishing constants were fitted by
searching for the smallest Chamfer distance to a reference trefoil surface,
measured after normalising both to unit RMS radius and aligning by principal
axes plus ICP.

The fitted mix is worth noting: a *light* relaxation followed by heavy
mean-curvature fairing.  Long relaxations move away from the reference, because
the particle model's equilibrium is not the same object as a minimal surface.
"""

from __future__ import annotations

from .braid import Braid
from .build import SurfaceParams, seifert_surface, state_surface
from .fair import minimal_surface, smooth_boundary
from .mesh import Mesh
from .relax import RelaxParams, relax
from .states import State, StateSurface
from .subdivide import catmull_clark

__all__ = ["finish", "surface"]


def finish(
    mesh: Mesh,
    *,
    relax_steps: int = 100,
    rim_steps: int = 0,
    levels: int = 2,
    fair_steps: int = 10,
    fair_strength: float = 2.0,
    relax_params: RelaxParams | None = None,
) -> Mesh:
    """Relax, refine and fair a freshly built surface."""
    if relax_steps:
        mesh = relax(mesh, relax_params or RelaxParams(), iterations=relax_steps)
    if rim_steps:
        mesh = smooth_boundary(mesh, iterations=rim_steps)
    if levels:
        mesh = catmull_clark(mesh, levels)
    if fair_steps:
        mesh = minimal_surface(mesh, strength=fair_strength, iterations=fair_steps)
    return mesh


def surface(
    braid: Braid | str,
    state: State | StateSurface | None = None,
    params: SurfaceParams | None = None,
    **kwargs,
) -> Mesh:
    """Build and finish in one call.

    >>> surface("AAA").info().genus
    1
    """
    return finish(state_surface(braid, state, params), **kwargs)
