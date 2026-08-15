"""weaving -- woven polyhedra, polylinks and orderly tangles.

The Blender-free engine behind the weave generators.  Python and numpy
only, so the package imports and self-tests headlessly; the registered
operators stay in their flat generator modules.

    polyhedral  strands routed along a polyhedron's edges or faces.
    sticks      woven straight-rod assemblies.
    stellated   weaves over stellated forms.
    shells      double-shell weaves: two nested surfaces laced together.
    links       Holden's regular polylinks -- rings interlocked with
                polyhedral symmetry.
    tangles     Holden's orderly tangles of interwoven polygons.
    rotegrity   rotational tensegrities of tilted struts.

NOT to be confused with `patterns/overunder.py`, which is the 2-D
over/under SOLVER -- union-find with parity -- and is consumed by the
flat interlace engines and by `polyhedral` here.  That module was called
`patterns/weave.py` until this work renamed it, precisely so the two
subjects stop answering to one name.
"""

from .links import (PHI, build_polylinks)
from .polyhedral import (Pattern, build_flags, build_weave, build_weave_tubes, flag_cross, flag_point, flag_step, geodesic, parse_pattern, relax_strands, seed_poly, sweep_ribbons, sweep_tubes, weave_circuits)
from .rotegrity import (build_rotegrity)
from .shells import (SOLID_ITEMS, bez, build_cells, build_double_shell, edge_pairing, find_crossings, relax_double_shell, shell_lines, solid_np, strand_cycles, strand_paths)
from .stellated import (BEND_INNER_EXT, BEND_MITER_EXT, DODECA_FACES, DODECA_VERTICES, FACE_STAR_RATIO, INV_PHI, build_arms, indexed_ssd_faces)
from .sticks import (PACKINGS, build_polystix)
from .tangles import (build_tangle, compound)

__all__ = [
    "build_polylinks",
    "PHI",
    "seed_poly",
    "geodesic",
    "Pattern",
    "parse_pattern",
    "build_flags",
    "flag_cross",
    "flag_step",
    "flag_point",
    "weave_circuits",
    "sweep_ribbons",
    "build_weave",
    "relax_strands",
    "sweep_tubes",
    "build_weave_tubes",
    "build_rotegrity",
    "solid_np",
    "edge_pairing",
    "strand_cycles",
    "bez",
    "find_crossings",
    "build_double_shell",
    "relax_double_shell",
    "shell_lines",
    "build_cells",
    "strand_paths",
    "SOLID_ITEMS",
    "indexed_ssd_faces",
    "build_arms",
    "INV_PHI",
    "FACE_STAR_RATIO",
    "BEND_MITER_EXT",
    "BEND_INNER_EXT",
    "DODECA_VERTICES",
    "DODECA_FACES",
    "build_polystix",
    "PACKINGS",
    "compound",
    "build_tangle",
]

__version__ = "1.0.0"


def _selftest():
    """The facade's contract: every name in __all__ resolves, and the
    engine stays headless."""
    import sys
    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    ok = not missing
    print(f"weaving: all {len(__all__)} names in __all__ resolve "
          f"{'OK' if ok else 'FAIL missing ' + ', '.join(missing)}")
    leaked = 'bpy' in sys.modules
    ok &= not leaked
    print(f"weaving: importing the facade left bpy out of sys.modules "
          f"{'OK' if not leaked else 'FAIL'}")
    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("weaving facade self-test failed")
