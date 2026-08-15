"""rollers -- bodies that roll without being spheres.

The Blender-free engine behind the roller generators.  Python and numpy
only, so the package imports and self-tests headlessly; the registered
operators stay in their flat generator modules.

    width        bodies of constant width -- Reuleaux polygons and the
                 Meissner tetrahedra.
    developable  the oloid and the sphericon, whose surfaces unroll flat.

Named for the Add menu group, which had already gathered these.
"""

from .developable import (build_antioloid, build_mobius, build_oloid, build_ruled, roller_circles)
from .width import (build_reuleaux_revolution, build_tetra_body)

__all__ = [
    "build_oloid",
    "roller_circles",
    "build_ruled",
    "build_antioloid",
    "build_mobius",
    "build_reuleaux_revolution",
    "build_tetra_body",
]

__version__ = "1.0.0"


def _selftest():
    """The facade's contract: every name in __all__ resolves, and the
    engine stays headless."""
    import sys
    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    ok = not missing
    print(f"rollers: all {len(__all__)} names in __all__ resolve "
          f"{'OK' if ok else 'FAIL missing ' + ', '.join(missing)}")
    leaked = 'bpy' in sys.modules
    ok &= not leaked
    print(f"rollers: importing the facade left bpy out of sys.modules "
          f"{'OK' if not leaked else 'FAIL'}")
    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("rollers facade self-test failed")
