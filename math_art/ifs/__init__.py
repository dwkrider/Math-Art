"""ifs -- the fractal engines: radix tiles, affine attractors, surfacing.

The Blender-free engine behind the Math Art fractal generators.  Python
and numpy only, so the whole package imports and self-tests headlessly;
the registered operators stay in their flat generator modules.  Layout
follows `math_art/seifert/`, `minsurf/`, `knots/` and `patterns/`.

    voxel     occupied cells on an integer lattice to a watertight mesh:
              the exterior-face walker, pinhole repair, outward
              orientation, largest-component selection and the fit to
              the 2 m cube.  Both families below end here.
    affine    contractive affine maps w_i(x) = A_i x + b_i, their
              attractor by Hutchinson's theorem, and the chaos game.
    radix     self-affine lattice tiles from an expanding integer matrix
              and a complete residue digit set -- exact at every level,
              not sampled.

A note on what this package is NOT.  The nine "fractal" generators were
originally scoped as one package on the strength of the shared theme.
Measured, they had zero shared top-level names and no cross-module
reaches: seven genuinely different methods (digit systems, ODE
integration, escape-time fields, a cellular automaton, subdivision
rules, inversive geometry, spectral synthesis).  So this package is a
namespace with one submodule per method, not a common engine -- the
submodules below the first three share nothing but a home, and that is
the honest description.  The one duplication that did exist, the planar
similarity underlying rep-tile inflation, lives in
`patterns/substitution.py`, because a rep-tile inflation and a Penrose
inflation are the same mathematics.

References for the mathematics are in each submodule's header; the
principal ones are Hutchinson (1981) for the attractor of a contractive
IFS, Barnsley (1993) for the chaos game, and Bandt (1991) for the
integer-matrix radix tiles.
"""

from .affine import (IFS_FACTS, IFS_PRESETS, SEEDS, build_ifs, chaos_game,
                     contractive, format_maps, parse_maps, plane_frame,
                     plane_relief, spectrally_contractive)
from .radix import (RADIX_PRESETS, TWINDRAGON_FACTS, abc_has_14_neighbours,
                    abc_is_ball, attractor_rank, build_radix, companion,
                    default_level, is_expanding, is_residue_system, max_holes,
                    max_level, radix_points, radix_topology, tile_support_bbox)
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
]

__version__ = "1.0.0"


def _selftest():
    """The facade's contract: every name in __all__ resolves, and the
    engine stays headless.

    A facade that drifts from its own __all__ fails at the caller, not
    here.  That is exactly how the pattern engine shipped a broken
    build once, so the check is cheap insurance.
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

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("ifs facade self-test failed")
