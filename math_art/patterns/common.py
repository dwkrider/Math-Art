
# Pattern Engine -- shared foundation
#
# A subsystem of generators that produce 2D patterns carrying genuine
# 3D structure: a unit cell (a motif or a saddle module) replicated by
# a symmetry across the plane and given relief, in the modular-
# constructivist tradition of Erwin Hauer and Norman Carlberg.
#
# This module is the Phase-0 foundation shared by every pattern
# generator.  It is pure Python + numpy (no bpy at import time) so the
# math is unit-testable outside Blender; the small Blender-facing
# helpers live behind an `if _IN_BLENDER:` guard at the bottom.
#
# It provides:
#   * the Conway-Thurston ORBIFOLD SIGNATURE parser -- cost ("magic
#     theorem" sum) and the geometry router that sends a signature to
#     the spherical / Euclidean / hyperbolic world;
#   * 2D isometry primitives (rotation, reflection, glide, lattice
#     translation) and a wallpaper-group table built from them;
#   * the common PLACED-TILES interchange format every tiling backend
#     emits, and the replicate/clip helpers of the Layer-3 wrapper;
#   * Mode-A relief builders -- turn 2D polygons or line segments into
#     watertight prisms on a backing slab -- plus centre/scale to the
#     2 m cube the rest of the extension uses;
#   * a Blender object builder shared by the operators.
#
# References:
#   John H. Conway, Heidi Burgiel & Chaim Goodman-Strauss, "The
#     Symmetries of Things" (2008) -- the orbifold signature notation and
#     the "magic theorem" cost accounting used throughout the engine.
#   E. S. Fedorov (1891) -- the classification of the 17 wallpaper groups
#     that the Euclidean table encodes.
#   Erwin Hauer and Norman Carlberg -- the modular-constructivist relief

# ---------------------------------------------------------------------
# This module is now a RE-EXPORT SHIM.
#
# The engine moved to the Blender-free package `math_art/patterns/`, on
# the same pattern as `seifert/`, `minsurf/` and `knots/`.  Twenty-two
# modules import this one, so the name and its public surface are kept
# exactly as they were rather than editing all of them at once.
#
# New code should import the package directly -- `from .patterns import
# wallpaper_group` -- and reach `patterns.emit` explicitly for the
# Blender builders, which are the only part that needs `bpy`.
# ---------------------------------------------------------------------


from .groups import (FRIEZE_NAMES, FRIEZE_ORDER, frieze_group,
                              group_closes, wallpaper_group,
                              wallpaper_isometries)
from .isometry import Glide, I, Mir, Rot, T, apply
from .motifs import (MOTIFS, PALETTE_RGBA, iso_type, kind_of,
                              motif)
from .orbifold import (IUC_ORDER, SIG_OF, WALLPAPER_NAMES,
                                geometry_of, orbifold_cost)
from .placed import Tiling
from .prisms import (center_scale, center_xy, merge_cells, prisms,
                              ribbon_polys, slab)
from .surfacemap import (DEFAULT_MAX_EDGE, PlaneSurface, SphereSurface,
                                   Surface, TorusSurface,
                                   canonicalize_corners, edge_points,
                                   make_surface, refine_poly,
                                   refine_segment, surface_patch,
                                   surface_prisms)

# private names some generators still reach for
from .groups import _cyclic, _dihedral, _HEX, _SQ      # noqa: F401
from .motifs import _comma, _rect                      # noqa: F401
from .prisms import _apply_transform, _global_transform  # noqa: F401


# The Blender layer lives in `patterns.emit`.  It only defines these names
# when `bpy` is importable, so the re-export is guarded: headless callers
# (and the test runner) simply do not see them, exactly as before.
try:
    from .emit import ADD_MENU, build_object, emit, register, unregister, source_mesh
except ImportError:               # headless: no Blender layer to re-export
    pass



def _selftest():
    # Self-test of the orbifold router: all 17 wallpaper signatures
    # must be Euclidean (cost 2), sample point groups spherical, and a
    # {7,3,2} triangle group hyperbolic.
    bad = [s for s in WALLPAPER_NAMES if geometry_of(s) != 'EUCLIDEAN']
    sph = geometry_of('532') == 'SPHERICAL'          # icosahedral
    hyp = geometry_of('732') == 'HYPERBOLIC'
    grp_ok = all(len(wallpaper_isometries(s, 2, 2)) > 0
                 for s in WALLPAPER_NAMES)
    frieze_ok = all(len(frieze_group(n)) > 0 for n in FRIEZE_ORDER)
    motif_ok = all(len(motif(k)) > 0 for k in MOTIFS)
    not_closed = [WALLPAPER_NAMES[s] for s in WALLPAPER_NAMES
                  if not group_closes(s)]
    print("wallpaper Euclidean:", not bad, "| bad:", bad)
    print("532 spherical:", sph, "| 732 hyperbolic:", hyp)
    print("groups build:", grp_ok, "| frieze:", frieze_ok,
          "| motifs:", motif_ok)
    print("group closure:", not not_closed, "| not closed:", not_closed)
    print("RESULT:", "OK" if (not bad and sph and hyp and grp_ok
                              and frieze_ok and motif_ok
                              and not not_closed) else "BAD")
    assert (not bad and sph and hyp and grp_ok and frieze_ok
            and motif_ok and not not_closed)
