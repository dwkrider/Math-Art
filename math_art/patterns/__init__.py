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
    relief      Mode-A relief: 2D polygons and segments to watertight
                prisms on a slab, centred and scaled to the 2 m cube.
    polygon2d   planar polygon predicates (signed area, winding, the
                projection to the xy-plane) that the tiling generators
                had each grown their own copy of.
    emit        Blender object builders -- the ONLY module here that
                touches `bpy`, and deliberately not re-exported below, so
                `import patterns` stays headless.

`pattern_common.py` remains as a re-export shim for its 22 importers.
New code should import this package directly.

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
from .polygon2d import ensure_ccw, signed_area, to_xy
from .relief import (center_scale, center_xy, merge_cells, prisms,
                     ribbon_polys, slab)

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
    # relief
    "ribbon_polys",
    "prisms",
    "slab",
    "center_scale",
    "center_xy",
    "merge_cells",
]

__version__ = "1.0.0"
