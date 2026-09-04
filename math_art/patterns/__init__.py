"""patterns -- the Pattern Engine: symmetry groups, tilings, relief.

The Blender-free engine behind the Math Art pattern generators: 2D
patterns carrying genuine 3D structure, in the modular-constructivist
tradition of Erwin Hauer and Norman Carlberg.  Python and numpy only, so
the whole package imports and self-tests headlessly; the registered
operators stay in their flat generator modules.  Layout follows
`math_art/seifert/`, `minsurf/` and `knots/`.

    orbifold    the Conway-Thurston signature parser: its cost (the magic
                theorem sum) routes a signature to the spherical,
                Euclidean or hyperbolic world.  One grammar for all three.
    isometry    plane isometries as 3x3 homogeneous matrices.
    groups      the 17 wallpaper groups and 7 friezes, each a lattice
                basis plus coset representatives.
    motifs      the motif library, and colouring each copy by the kind of
                isometry that produced it.
    placed      the placed-tiles interchange format every tiling backend
                emits and every relief mode consumes.
    prisms      Mode-A relief: 2D polygons and segments to watertight
                prisms on a slab, centred and scaled to the 2 m cube.
                Named `prisms` and not `relief` because `math_art/relief/`
                is a different subject -- displacement on closed surfaces
                -- and two unrelated answers to one grep is a bug in the
                naming, not in either module.
    polygon2d   planar predicates (signed area, winding, arclength, line
                intersection) the tiling generators had each grown their
                own copy of.
    ribbon      mitred ribbons, band faces and strand smoothing -- the
                interlace layer, previously private to the Islamic
                pattern generator and reached into by four others.
    overunder   the over/under solver: union-find with parity, and the
                smoothstep z-offset that makes a strand pass under.
    surfacemap  curved-surface mapping: lay a flat tiling on a torus or a
                sphere, offsetting relief along the surface normal instead
                of +Z.  The torus is a genuine quotient (so a
                lattice-commensurate tiling is seamless); the sphere is a
                chart, and says so.
    substitution  prototiles + inflation rules over planar similarities
                (z -> Az + B conj z + C, so reflections are expressible
                and the hat is representable).  Expansion returns
                PLACEMENTS, not baked polygons.
    emit        Blender object builders -- the ONLY module here that
                touches `bpy`, and deliberately not re-exported below, so
                `import patterns` stays headless.

`common` is the Blender-facing facade: this package's engine names plus
the `emit` builders in one namespace, which is what the 22 pattern
generators import.  It exists because `__init__` deliberately leaves the
Blender layer out, and those generators need both halves.

References for the mathematics are in each submodule's header; the
principal ones are Conway, Burgiel and Goodman-Strauss, "The Symmetries
of Things" (2008) for the orbifold notation and the magic theorem, and
Fedorov (1891) for the classification of the plane groups.
"""

from .groups import (FRIEZE_NAMES, FRIEZE_ORDER, frieze_group, group_closes,
                     wallpaper_group, wallpaper_isometries)
from .isometry import Glide, I, Mir, Rot, T, apply
from .motifs import MOTIFS, PALETTE_RGBA, iso_type, kind_of, motif
from .orbifold import (IUC_ORDER, SIG_OF, WALLPAPER_NAMES, geometry_of,
                       orbifold_cost)
from .placed import Tiling
from .polygon2d import (arclen, ensure_ccw, line_intersection,
                        signed_area, to_xy, unit)
from .ribbon import (angle_cut_piece, band_ribbon_faces,
                     band_ribbon_faces_z, catmull_rom, cut_band,
                     cut_cap_on_edge, miter, miter_ribbon)
from .overunder import ParityDSU, weave_zoff
from .substitution import Similarity, Substitution, penrose_p3
from .prisms import (center_scale, center_xy, merge_cells, prisms,
                     ribbon_polys, slab)
from .surfacemap import (DEFAULT_MAX_EDGE, PlaneSurface, SphereSurface,
                         Surface, TorusSurface, canonicalize_corners,
                         edge_points, make_surface, refine_poly,
                         refine_segment, surface_patch, surface_prisms)

__all__ = [
    # orbifold signatures
    "orbifold_cost",
    "geometry_of",
    "WALLPAPER_NAMES",
    "SIG_OF",
    "IUC_ORDER",
    # isometries
    "I",
    "T",
    "Rot",
    "Mir",
    "Glide",
    "apply",
    # groups
    "wallpaper_group",
    "wallpaper_isometries",
    "group_closes",
    "frieze_group",
    "FRIEZE_NAMES",
    "FRIEZE_ORDER",
    # motifs and colouring
    "motif",
    "MOTIFS",
    "iso_type",
    "kind_of",
    "PALETTE_RGBA",
    # interchange format
    "Tiling",
    # planar polygons
    "signed_area",
    "ensure_ccw",
    "to_xy",
    "unit",
    "line_intersection",
    "arclen",
    # ribbons and interlace
    "miter",
    "miter_ribbon",
    "band_ribbon_faces",
    "band_ribbon_faces_z",
    "catmull_rom",
    "cut_band",
    "cut_cap_on_edge",
    "angle_cut_piece",
    "ParityDSU",
    "weave_zoff",
    # substitution tilings
    "Similarity",
    "Substitution",
    "penrose_p3",
    # relief
    "ribbon_polys",
    "prisms",
    "slab",
    "center_scale",
    "center_xy",
    "merge_cells",
    # curved-surface mapping
    "Surface",
    "PlaneSurface",
    "TorusSurface",
    "SphereSurface",
    "make_surface",
    "refine_poly",
    "refine_segment",
    "edge_points",
    "canonicalize_corners",
    "surface_patch",
    "surface_prisms",
    "DEFAULT_MAX_EDGE",
]

__version__ = "1.0.0"


def _selftest():
    """The facade's contract: everything in __all__ is importable from
    the package itself, and nothing Blender-facing leaks in.

    A facade that drifts from its own __all__ fails at the caller, not
    here, so this is cheap insurance -- and it is the check that would
    have caught `emit`'s names going missing from the shim.
    """
    import sys
    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    ok = not missing
    print(f"patterns: all {len(__all__)} names in __all__ resolve "
          f"{'OK' if ok else 'FAIL missing ' + ', '.join(missing)}")

    # importing the package must NOT drag in bpy: emit stays out of the
    # facade precisely so the engine is headless.
    leaked = 'bpy' in sys.modules
    ok &= not leaked
    print(f"patterns: importing the facade left bpy out of sys.modules "
          f"{'OK' if not leaked else 'FAIL'}")

    # `emit` must not be RE-EXPORTED.  Note the check is on __all__, not
    # on hasattr: importing `patterns.emit` anywhere makes Python bind it
    # as an attribute of this package, so hasattr says nothing about the
    # facade's contract.  What matters is that `from patterns import *`
    # does not drag the Blender layer in.
    leaked_names = [n for n in __all__ if n in ('emit', 'build_object',
                                                'register', 'unregister')]
    ok &= not leaked_names
    print(f"patterns: the Blender layer is not in __all__ "
          f"{'OK' if not leaked_names else 'FAIL ' + ','.join(leaked_names)}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("patterns facade self-test failed")
