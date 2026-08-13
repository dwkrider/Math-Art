"""minsurf -- minimal surfaces: Weierstrass-Enneper data, meshing, solvers.

The Blender-FREE engine behind the Math Art minimal-surface generators.
NumPy only, so the whole package imports and self-tests headlessly; the
registered operators stay in the flat `minimal_surface_toolkit.py`, which
imports this package.  Layout follows `math_art/seifert/`.

Four pieces, in dependency order:

    elliptic     Weierstrass P, P' and zeta on a lattice, via Jacobi theta
                 series -- the square (lemniscatic) torus that Costa and
                 Chen-Gackstatter live on.
    domain       shared mesh/domain utilities: torus grids, puncture masks,
                 connected components, bounding-box fit, boundary cleanup.
    weierstrass  the generic Weierstrass-Enneper / Bjorling integrator that
                 turns a small data "spec" into a surface builder.
    zoo          the catalog: ~80 published surfaces as specs for the above.
    parametric   the classic closed-form surfaces, the PARAMETRIC/MESH_PARAM
                 registries (which `zoo` extends at import), and the meshing
                 pipeline `build_parametric`.
    tpms         triply-periodic minimal surfaces as nodal level sets, meshed
                 by marching tetrahedra.
    plateau      discrete area minimization on a pinned boundary, plus the
                 circle-to-torus-knot span.

Typical use:

    from math_art import minsurf
    V, quads = minsurf.build_parametric('COSTA', 64, 64, 3, 1.0, 1.0)

The mathematics is credited in each submodule's header; the principal
references are Weierstrass (1866) and Enneper (1864) for the representation
formula, Schwarz (1890) for the Bjorling problem and the P/D surfaces,
Schoen (NASA TN D-5541, 1970) for the gyroid and relatives, Costa (1982)
with Hoffman-Meeks (1985) for the embedded genus-one example, and
Pinkall-Polthier (1993) for the discrete area flow.
"""

from .parametric import (ANGLE_PARAM, COUNT_PARAM, FAMILIES, MESH_PARAM,
                         PARAMETRIC, PERIODIC_NO_ARRAY, STOREY_PARAM,
                         SURFACE_FAMILY, build_parametric,
                         build_parametric_grid)
from .plateau import (align_loops, build_annulus_grid, build_disk_grid,
                      build_seifert_span_grid, fair_grid_2d,
                      fair_grid_columns, mesh_area, minimize_area,
                      relax_normal_flow, resample_loop, torus_knot)
from .tpms import (TPMS, TPMS2_HEX_LATTICE, TPMS_EXACT, build_tpms,
                   build_tpms_exact, marching_tets, tpms2_LATTICE,
                   tpms2_cell_matrix)

__all__ = [
    # registries
    "PARAMETRIC",
    "MESH_PARAM",
    "COUNT_PARAM",
    "ANGLE_PARAM",
    "STOREY_PARAM",
    "PERIODIC_NO_ARRAY",
    "SURFACE_FAMILY",
    "FAMILIES",
    "TPMS",
    "TPMS_EXACT",
    "TPMS2_HEX_LATTICE",
    "tpms2_LATTICE",
    # parametric surfaces
    "build_parametric",
    "build_parametric_grid",
    # triply periodic
    "build_tpms",
    "build_tpms_exact",
    "marching_tets",
    "tpms2_cell_matrix",
    # Plateau solver and spans
    "minimize_area",
    "relax_normal_flow",
    "mesh_area",
    "fair_grid_2d",
    "fair_grid_columns",
    "resample_loop",
    "align_loops",
    "build_disk_grid",
    "build_annulus_grid",
    "build_seifert_span_grid",
    "torus_knot",
]

__version__ = "1.0.0"
