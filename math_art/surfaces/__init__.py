"""surfaces -- surface families that are not minimal surfaces.

The Blender-free engine behind the Math Art surface generators.  Python
and numpy only, so the whole package imports and self-tests headlessly;
the registered operators stay in their flat generator modules.  Layout
follows `math_art/seifert/`, `minsurf/`, `knots/`, `patterns/`,
`polyhedra/` and `ifs/`.

    primitives  the geodesic sphere and the other base surfaces that
                other engines build on.
    supershape  Gielis's superformula and its spherical products.
    harmonics   spherical harmonics as radial displacement.
    algebraic   the classical algebraic surfaces -- Cayley, Clebsch,
                Kummer, Barth -- as zero sets of polynomials.
    calabi_yau  combinatorial skeletons of Calabi-Yau geometry: Ruan's
                SYZ discriminant graph and the Hanson-Sha tessellation
                of the Fermat surfaces in CP3.  Deliberately has no
                operator over it -- see the module header -- so nothing
                is re-exported here; import it by name.

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
from .supershape import (PRESETS, build_from_preset, build_shell, build_superellipsoid, build_supershape_3d, build_supertoroid)
from .harmonics import (BOURKE_PRESETS, FORM_ITEMS, TAU, assoc_legendre, bourke_radius, build_radial_surface, build_spherical_harmonic, center_fit, max_abs_harmonic, real_sph_harm, sph_harm_norm)
from .algebraic import (PRESETS, build_algebraic)

__all__ = [
    "icosphere",
    "PHI",
    "PRESETS",
    "build_from_preset",
    "build_shell",
    "build_superellipsoid",
    "build_supershape_3d",
    "build_supertoroid",
    "BOURKE_PRESETS",
    "FORM_ITEMS",
    "TAU",
    "assoc_legendre",
    "bourke_radius",
    "build_radial_surface",
    "build_spherical_harmonic",
    "center_fit",
    "max_abs_harmonic",
    "real_sph_harm",
    "sph_harm_norm",
    "PRESETS",
    "build_algebraic",
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
