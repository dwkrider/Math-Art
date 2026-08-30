"""polytopes -- regular 4-polytopes and their projection to 3-space.

The Blender-free engine behind the 4-D polytope generator.  Python and
numpy only, so the package imports and self-tests headlessly; the
registered operator stays in `polytope4d_generator.py`.

    regular   the six regular convex 4-polytopes, their cells and
              vertices, rotations in the six coordinate planes, and the
              orthographic and stereographic projections to R^3.

ONE CONSUMER, and that is fine.  This package exists so the mathematics
of the 4-polytopes lives somewhere a reader can check it without opening
Blender, not because several generators share it.  A package with one
submodule is honest; a package whose modules do not speak to each other
is not.

References are in the submodule header; the principal ones are Schlafli
(1852) for the classification and Coxeter, "Regular Polytopes" (1973).
"""

from . import wythoff
from .wythoff import FAMILIES as WYTHOFF_FAMILIES
from .regular import (COUNTS, DUAL_KIND, HALF_TOL, PHI, add_sphere, add_strut, clear_pole, dual_vertices, hopf_ring_cosets, polytope_edges, polytope_faces, polytope_vertices, project_point, ring_cell_points, rotate4)

__all__ = [
    "wythoff",
    "WYTHOFF_FAMILIES",
    "COUNTS",
    "DUAL_KIND",
    "HALF_TOL",
    "PHI",
    "add_sphere",
    "add_strut",
    "clear_pole",
    "dual_vertices",
    "hopf_ring_cosets",
    "polytope_edges",
    "polytope_faces",
    "polytope_vertices",
    "project_point",
    "ring_cell_points",
    "rotate4",
]

__version__ = "1.0.0"


def _selftest():
    """The facade's contract: every name in __all__ resolves, and the
    engine stays headless."""
    import sys
    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    ok = not missing
    print(f"polytopes: all {len(__all__)} names in __all__ resolve "
          f"{'OK' if ok else 'FAIL missing ' + ', '.join(missing)}")
    leaked = 'bpy' in sys.modules
    ok &= not leaked
    print(f"polytopes: importing the facade left bpy out of sys.modules "
          f"{'OK' if not leaked else 'FAIL'}")
    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("polytopes facade self-test failed")
