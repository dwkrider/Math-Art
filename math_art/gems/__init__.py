"""gems -- the geometry of cut gemstones.

The Blender-free engine behind the gemstone generator.  Python and numpy
only, so the package imports and self-tests headlessly; the registered
operators stay in the flat generator module.

    design   the faceting-machine model: index gears, mast angles, tiers,
             tangent-ratio conversion, and the plane-through-points rules
             a lapidary actually cuts by.
    facets   half-space intersection -- turning those planes into a solid,
             with least-squares snapping at the meetpoints where many
             facets converge.

A faceted stone is specified as PLANES, not as vertices, which is why this
is a half-space intersection and not a convex hull: the two are dual
problems and neither routine substitutes for the other.

References:
  Marcel Tolkowsky, "Diamond Design: A Study of the Reflection and
    Refraction of Light in a Diamond", E. & F. N. Spon, London, 1919.
  Robert W. Strickland, "GemCad for Windows Version 1.0 User's Guide",
    GemSoft Enterprises.
  Robert H. Long & Norman W. Steele, "Introduction to Meetpoint Faceting",
    Seattle Faceting Books, 1985.
  Ping Zheng, Qinghua Xiao, Wei Xiong, Ying Yang, "Optimization design
    model construction of round faceted gemstone facets based on geometric
    transformation theory", J. Combin. Math. Combin. Comput. 127a (2025)
    1269-1284, doi:10.61091/jcmcc127a-074.
  Jose M. Sasian, Peter Yantzer, Tom Tivol, "The Optical Design of
    Gemstones", Optics & Photonics News 14(4), 2003.
"""

from .design import (SRB_GEMCAD, CutDesign, Tier, expand_indices, facet_plane,
                     index_azimuth, planes, tangent_ratio,
                     tier_from_angle_point, tier_from_points)
from .facets import (Polytope, intersect_halfspaces, polytope_checks, volume)

__all__ = [
    "Tier",
    "CutDesign",
    "SRB_GEMCAD",
    "planes",
    "facet_plane",
    "index_azimuth",
    "expand_indices",
    "tangent_ratio",
    "tier_from_points",
    "tier_from_angle_point",
    "Polytope",
    "intersect_halfspaces",
    "polytope_checks",
    "volume",
]

__version__ = "1.0.0"


def _selftest():
    """The facade's contract: every exported name resolves, the engine
    stays headless, and the two halves compose into a stone."""
    import sys

    import numpy as np

    ok = True
    missing = [n for n in __all__ if not hasattr(sys.modules[__name__], n)]
    good = not missing
    ok &= good
    print(f"gems: all {len(__all__)} names in __all__ resolve "
          f"{'OK' if good else 'missing ' + ', '.join(missing)}")

    leaked = 'bpy' in sys.modules
    ok &= not leaked
    print(f"gems: importing the facade left bpy out of sys.modules "
          f"{'OK' if not leaked else 'bpy leaked in'}")

    # the two halves compose: a published design becomes a closed solid
    N, d, _ = planes(SRB_GEMCAD)
    P = intersect_halfspaces(N, d)
    chk = polytope_checks(P, N, d)
    good = chk["closed"] and chk["convex"] and len(P.faces) == 73
    ok &= good
    print(f"gems: the Standard Round Brilliant builds end to end "
          f"({len(P.V)} vertices, {len(P.faces)} facets, "
          f"volume {volume(P):.6f}) {'OK' if good else 'did not build'}")

    # a tangent-ratio retarget still builds, and still has 73 facets --
    # the plan view is preserved, so no facet is cut away
    conv = tangent_ratio(SRB_GEMCAD, 42.5, 45.0, half="pavilion")
    Nc, dc, _ = planes(conv)
    Pc = intersect_halfspaces(Nc, dc)
    good = polytope_checks(Pc, Nc, dc)["closed"] and len(Pc.faces) == 73
    ok &= good
    print(f"gems: a pavilion retarget to 45 degrees still closes with 73 "
          f"facets (got {len(Pc.faces)}) {'OK' if good else 'lost facets'}")

    # ... and it is deeper, which is the point of steepening the pavilion
    deep_before = float(P.V[:, 2].min())
    deep_after = float(Pc.V[:, 2].min())
    good = deep_after < deep_before - 1e-6
    ok &= good
    print(f"gems: steepening the pavilion deepens the stone "
          f"({deep_before:.4f} -> {deep_after:.4f}) "
          f"{'OK' if good else 'no deeper'}")

    # the girdle outline is untouched by that retarget -- the plan view
    # invariant, measured on the built solid rather than on the design
    r0 = np.sort(np.linalg.norm(P.V[:, :2], axis=1))[-16:]
    r1 = np.sort(np.linalg.norm(Pc.V[:, :2], axis=1))[-16:]
    worst = float(np.max(np.abs(r0 - r1)))
    good = worst < 1e-9
    ok &= good
    print(f"gems: the girdle outline survives the retarget unchanged "
          f"(worst {worst:.2e}) {'OK' if good else 'plan view moved'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems facade self-test failed")
