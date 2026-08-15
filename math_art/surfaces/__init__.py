"""surfaces -- surface families that are not minimal surfaces.

The Blender-free engine behind the Math Art surface generators.  Python
and numpy only, so the whole package imports and self-tests headlessly;
the registered operators stay in their flat generator modules.  Layout
follows `math_art/seifert/`, `minsurf/`, `knots/`, `patterns/`,
`polyhedra/` and `ifs/`.

    primitives  the geodesic sphere and the other base surfaces that
                other engines build on.

WHY THIS IS SEPARATE FROM `minsurf/`.  Minimal surfaces are surfaces,
so two sibling packages reads oddly, and the split is historical rather
than mathematical: `minsurf/` was extracted first and carries the
Weierstrass representation, the Plateau solver and the TPMS families.
This package is for the rest -- algebraic and implicit surfaces,
parametric families, and the primitives above.  Folding the two together
is a reasonable future change and a bad simultaneous one: `minsurf/` has
eight modules and nine consumers.

The Add menu calls its group "Minimal", but the group has long held
supershapes, spherical harmonics and algebraic surfaces alongside the
genuinely minimal ones, which is what made this package a real gap
rather than an invented one.
"""

from .primitives import PHI, icosphere

__all__ = [
    "icosphere",
    "PHI",
]

__version__ = "1.0.0"


def _selftest():
    """The facade's contract: every name in __all__ resolves, and the
    engine stays headless."""
    import sys
    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    ok = not missing
    print(f"surfaces: all {len(__all__)} names in __all__ resolve "
          f"{'OK' if ok else 'FAIL missing ' + ', '.join(missing)}")

    leaked = 'bpy' in sys.modules
    ok &= not leaked
    print(f"surfaces: importing the facade left bpy out of sys.modules "
          f"{'OK' if not leaked else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("surfaces facade self-test failed")
