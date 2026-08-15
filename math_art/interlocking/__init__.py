"""interlocking -- assemblies held together by geometry alone.

The Blender-free engine behind the topological-interlocking generator.
Python and numpy only, so the package imports and self-tests headlessly;
the registered operator stays in `interlocking_generator.py`.

    blocks   convex blocks cut so that none can be removed while its
             neighbours are held, with no bonding or fastening.

This is the first member of what the taxonomy calls "Fabrication &
Mechanism".  It is deliberately NOT called `fabrication/`: origami,
net-unfolding and tensegrity generators do not exist yet, and naming a
package for a heading it does not yet fill invites the folder-not-a-
library problem that retired two packages earlier in this work.  Rename
it when there is something to rename it around.
"""

from .blocks import (build_bisquare, build_cells, build_dome, build_escher, build_escher_block, build_hendeca, build_mcs, build_rhom, build_sl, build_tetra, build_tetrocta, build_versatile, cells_to_mesh, cells_to_meshes)

__all__ = [
    "build_bisquare",
    "build_cells",
    "build_dome",
    "build_escher",
    "build_escher_block",
    "build_hendeca",
    "build_mcs",
    "build_rhom",
    "build_sl",
    "build_tetra",
    "build_tetrocta",
    "build_versatile",
    "cells_to_mesh",
    "cells_to_meshes",
]

__version__ = "1.0.0"


def _selftest():
    """The facade's contract: every name in __all__ resolves, and the
    engine stays headless."""
    import sys
    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    ok = not missing
    print(f"interlocking: all {len(__all__)} names in __all__ resolve "
          f"{'OK' if ok else 'FAIL missing ' + ', '.join(missing)}")
    leaked = 'bpy' in sys.modules
    ok &= not leaked
    print(f"interlocking: importing the facade left bpy out of sys.modules "
          f"{'OK' if not leaked else 'FAIL'}")
    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("interlocking facade self-test failed")
