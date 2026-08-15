"""projection -- projections from higher dimensions and curved spaces.

The Blender-free engine behind the projection generators.  Python and
numpy only, so the package imports and self-tests headlessly; the
registered operators stay in their flat generator modules.

    stereographic   the conformal map from the sphere to the plane, and
                    the curves and shadows it induces.

Named for the taxonomy heading it serves ("Perception & Projection"),
and expected to gain the anamorphic and impossible-figure engines when
those generators are written.
"""

from .stereographic import (build_shell)

__all__ = [
    "build_shell",
]

__version__ = "1.0.0"


def _selftest():
    """The facade's contract: every name in __all__ resolves, and the
    engine stays headless."""
    import sys
    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    ok = not missing
    print(f"projection: all {len(__all__)} names in __all__ resolve "
          f"{'OK' if ok else 'FAIL missing ' + ', '.join(missing)}")
    leaked = 'bpy' in sys.modules
    ok &= not leaked
    print(f"projection: importing the facade left bpy out of sys.modules "
          f"{'OK' if not leaked else 'FAIL'}")
    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("projection facade self-test failed")
