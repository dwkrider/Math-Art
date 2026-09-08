# Finite subdivision rules and conformal tilings.
#
# Each tile type subdivides into a fixed pattern of tile types; iterate, and the
# limit is a tiling.  The headline case is Bowers and Stephenson's pentagonal
# rule -- one pentagon into six -- whose tiles are pentagons COMBINATORIALLY but
# have no straight-line realisation in which they are all regular.  The geometry
# therefore comes from a circle packing, which is what makes every tile
# "conformally regular" at every depth.
#
# Blender-free: Python only, so the package imports and self-tests headlessly.
# It depends on `math_art/packing/` for the layout; the registered operator is
# `subdivision_tiling_generator.py`.
#
#   cells        polygonal complexes: tiles, edges, validation
#   rules        the subdivision rules and their iteration
#   triangulate  the bridge to the packer, and the three layouts
#
# References:
# - Philip L. Bowers and Kenneth Stephenson, "A 'regular' pentagonal tiling of
#   the plane", Conformal Geometry and Dynamics 1 (1997), pp. 58-86.
# - J. W. Cannon, W. J. Floyd and W. R. Parry, "Finite subdivision rules",
#   Conformal Geometry and Dynamics 5 (2001), pp. 153-196.
# - Burt Rodin and Dennis Sullivan, "The convergence of circle packings to the
#   Riemann mapping", Journal of Differential Geometry 26 (1987), pp. 349-360.

from .cells import CellComplex, single_polygon
from .rules import (BARYCENTRIC, PENTAGONAL, QUAD, RULE_FACTOR, RULE_SEEDS,
                    RULES, TRIANGLE_QUAD, build, check_edge_consistency,
                    subdivide)
from .triangulate import (COMBINATORIAL, CONFORMAL, LAYOUTS, PLANAR,
                          aspect_spread, barycentric_triangulation,
                          boundary_cycle, realise, tile_polygons,
                          tutte_positions, worst_anisotropy)

__all__ = [
    'CellComplex', 'single_polygon',
    'RULES', 'RULE_SEEDS', 'RULE_FACTOR',
    'PENTAGONAL', 'BARYCENTRIC', 'QUAD', 'TRIANGLE_QUAD',
    'build', 'subdivide', 'check_edge_consistency',
    'LAYOUTS', 'COMBINATORIAL', 'PLANAR', 'CONFORMAL',
    'realise', 'tile_polygons', 'barycentric_triangulation',
    'boundary_cycle', 'tutte_positions', 'aspect_spread', 'worst_anisotropy',
]


def build_tiling(rule=PENTAGONAL, depth=2, layout=CONFORMAL):
    """Subdivide and realise in one call.

    Returns (polygons, complex, info) with `polygons` a list of tiles, each a
    list of (x, y) corners."""
    K = build(rule, depth)
    pos, info = realise(K, layout)
    return tile_polygons(K, pos), K, info


def tile_count(rule, depth):
    """How many tiles a depth would produce -- for warning before building."""
    return RULE_FACTOR[rule] ** depth


def _selftest():
    import sys

    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    assert not missing, "subdiv: __all__ names missing: %s" % (missing,)

    polys, K, info = build_tiling(PENTAGONAL, 2, CONFORMAL)
    assert len(polys) == 36 == tile_count(PENTAGONAL, 2)
    assert all(len(p) == 5 for p in polys)
    assert info['converged'] and info['angle_error'] < 1e-8

    polys_b, Kb, info_b = build_tiling(BARYCENTRIC, 2, COMBINATORIAL)
    assert len(polys_b) == 36 == tile_count(BARYCENTRIC, 2)
    assert all(len(p) == 3 for p in polys_b)

    for rule in RULES:
        assert tile_count(rule, 3) == RULE_FACTOR[rule] ** 3

    assert 'bpy' not in sys.modules, "subdiv must import without Blender"
    print("subdiv: %d names resolve; pentagonal depth 2 gives 36 conformal "
          "pentagons. RESULT: OK" % len(__all__))
