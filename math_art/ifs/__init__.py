"""ifs -- the fractal engines behind the Math Art fractal generators.

Blender-free: Python and numpy only, so the whole package imports and
self-tests headlessly; the registered operators stay in their flat
generator modules.  Layout follows `math_art/seifert/`, `minsurf/`,
`knots/`, `patterns/` and `polyhedra/`.

    voxel      occupied cells on an integer lattice to a watertight
               mesh: the exterior-face walker, pinhole repair, outward
               orientation, largest-component selection and the fit to
               the 2 m cube.
    affine     contractive maps w_i(x) = A_i x + b_i, Hutchinson's
               attractor, and the chaos game.
    radix      self-affine lattice tiles from an expanding integer
               matrix and a complete residue digit set.
    flow       strange attractors of autonomous ODEs -- Lorenz, Rossler
               and the Sprott catalogue.
    escape     escape-time fractals: Mandelbulb, quaternion Julia,
               Mandelbox.
    inversive  Apollonian gaskets and sphere packings, via the Descartes
               circle theorem.
    menger     Menger-type sponges and Sierpinski solids, as digit rules
               on subdivided cubes.
    spacefill  space-filling cell complexes: the Kelvin cell, the
               rhombic dodecahedron, the tet-oct honeycomb.
    automaton  Reiter's snowflake cellular automaton.
    spectral   fractal relief by spectral synthesis -- fBm and
               Weierstrass sums.

WHAT THIS PACKAGE IS.  A namespace with one submodule per method, not a
common engine.  These nine generators were originally scoped as one
package on the strength of the shared theme; measured, they implement
seven genuinely different methods and the first three share `voxel`
while the rest share nothing but a home.  That is the honest
description, and the reason the submodules below `radix` do not import
from each other.

Two real duplications did turn up and are now gone: `menger` and
`affine` both carried verbatim copies of the tetrahedron and octahedron
tables that `polyhedra.seeds` already owned, and `menger` carried a
second copy of `voxel._FACE_DIRS`.  A third -- the planar similarity
underlying rep-tile inflation -- lives in `patterns/substitution.py`,
because a rep-tile inflation and a Penrose inflation are the same
mathematics.

NOT re-exported below, deliberately:

  * `PRESETS`.  Three submodules define one (`flow`, `escape`,
    `automaton`) and they are different catalogues.  Flattening them
    into the facade would silently give two of the three to whichever
    import ran last, so reach for `ifs.flow.PRESETS` by name.
  * `menger.MAX_LEVEL`, a per-kind cap; `radix.max_level` is a function
    that computes one.  Similar names, unrelated things, so the facade
    keeps the module qualifier on the first.

References for the mathematics are in each submodule's header.
"""

from .affine import (IFS_FACTS, IFS_PRESETS, SEEDS, build_ifs, chaos_game,
                     contractive, format_maps, parse_maps, plane_frame,
                     plane_relief, spectrally_contractive)
from .automaton import build_preset, build_snowflake, simulate_reiter
from .escape import build_escape_fractal
from .flow import build_attractor, integrate, normalize, resample, speeds
from .inversive import Ball, build_apollonian, gasket_2d, packing_3d, reflect
from .menger import (GRID_KINDS, build_corner_sponge, build_grid_sponge,
                     build_sponge, sponge_cells)
from .radix import (RADIX_PRESETS, TWINDRAGON_FACTS, abc_has_14_neighbours,
                    abc_is_ball, attractor_rank, build_radix, companion,
                    default_level, is_expanding, is_residue_system, max_holes,
                    max_level, radix_points, radix_topology, tile_support_bbox)
from .spacefill import block_volume, build_block, build_mesh, spiral_n
from .spectral import (BASES, base_plate, base_sphere, base_torus, eval_field,
                       fbm_modes, weierstrass_modes, build_fractal_surface)
from .voxel import (MAX_CELLS, blur_density, center_fit, edge_stats,
                    fill_pinholes, keep_largest, orient_outward, voxel_surface)

__all__ = [
    # voxel surfacing
    "voxel_surface",
    "orient_outward",
    "fill_pinholes",
    "keep_largest",
    "center_fit",
    "blur_density",
    "edge_stats",
    "MAX_CELLS",
    # affine IFS
    "build_ifs",
    "chaos_game",
    "contractive",
    "spectrally_contractive",
    "parse_maps",
    "format_maps",
    "plane_frame",
    "plane_relief",
    "IFS_PRESETS",
    "IFS_FACTS",
    "SEEDS",
    # radix tiles
    "build_radix",
    "radix_points",
    "companion",
    "is_expanding",
    "is_residue_system",
    "attractor_rank",
    "radix_topology",
    "tile_support_bbox",
    "abc_has_14_neighbours",
    "abc_is_ball",
    "max_holes",
    "max_level",
    "default_level",
    "RADIX_PRESETS",
    "TWINDRAGON_FACTS",
    # strange attractors
    "build_attractor",
    "integrate",
    "normalize",
    "resample",
    "speeds",
    # escape-time
    "build_escape_fractal",
    # inversive geometry
    "build_apollonian",
    "gasket_2d",
    "packing_3d",
    "reflect",
    "Ball",
    # sponges
    "build_sponge",
    "build_grid_sponge",
    "build_corner_sponge",
    "sponge_cells",
    "GRID_KINDS",
    # space-filling complexes
    "build_mesh",
    "build_block",
    "block_volume",
    "spiral_n",
    # snowflake automaton
    "build_snowflake",
    "build_preset",
    "simulate_reiter",
    # spectral relief
    "build_fractal_surface",
    "eval_field",
    "fbm_modes",
    "weierstrass_modes",
    "base_plate",
    "base_sphere",
    "base_torus",
    "BASES",
]

__version__ = "1.1.0"


def _selftest():
    """The facade's contract: every name in __all__ resolves, the engine
    stays headless, and the deliberate omissions stay omitted.

    A facade that drifts from its own __all__ fails at the caller, not
    here.  That is exactly how the pattern engine shipped a broken build
    once, so the check is cheap insurance.
    """
    import sys
    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    ok = not missing
    print(f"ifs: all {len(__all__)} names in __all__ resolve "
          f"{'OK' if ok else 'FAIL missing ' + ', '.join(missing)}")

    leaked = 'bpy' in sys.modules
    ok &= not leaked
    print(f"ifs: importing the facade left bpy out of sys.modules "
          f"{'OK' if not leaked else 'FAIL'}")

    # The colliding names must NOT be flattened into the facade: three
    # submodules define PRESETS and they are different catalogues, so a
    # re-export would hand two of the three to whichever import ran last.
    from . import automaton, escape, flow
    cats = [flow.PRESETS, escape.PRESETS, automaton.PRESETS]
    distinct = len({tuple(sorted(c)) for c in cats}) == 3
    ok &= distinct and 'PRESETS' not in __all__
    print(f"ifs: the three PRESETS catalogues stay distinct and unexported "
          f"{'OK' if distinct and 'PRESETS' not in __all__ else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("ifs facade self-test failed")
