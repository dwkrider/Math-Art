# The faceting-machine model of a cut gemstone.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python and numpy
# only -- no `bpy` -- so the engine imports and self-tests headlessly.
#
# A faceted stone is the intersection of half-spaces.  A lapidary does not
# think in half-spaces, though: the stone is held on a dop on a mast, and
# two controls place every facet --
#
#   * the mast/protractor angle theta, the tilt of the cutting lap; and
#   * an INDEX GEAR, a toothed wheel quantising rotation about the dop
#     axis into `gear` positions (96 is the trade standard: 3.75 degrees
#     per tooth, and 96 = 2^5 * 3 divides by 2, 3, 4, 6, 8, 12, 16, 24,
#     32 and 48, so it can hold nearly any symmetry a design wants).
#
# so a facet plane has outward unit normal
#
#     n(theta, i) = (sin|t| cos(phi_i), sin|t| sin(phi_i), s cos|t|)
#     phi_i = 2 pi i / gear + offset,        s = +1 crown, -1 pavilion
#
# and the stone is {x : n . x <= d} over every facet, with d the
# CENTRE-TO-FACET distance the cutter sets through mast height.
#
# The data model here is deliberately GemCad's, because that is what tens
# of thousands of published designs are already written in: a design is a
# list of TIERS, each one angle + distance + a list of index positions,
# plus the gear, the symmetry and the refractive index the design was cut
# for.  See `asc.py` for the on-disk form.
#
# References:
#   Marcel Tolkowsky, "Diamond Design: A Study of the Reflection and
#     Refraction of Light in a Diamond", E. & F. N. Spon, London, 1919 --
#     the original derivation of the round brilliant's proportions.
#   Robert W. Strickland, "GemCad for Windows Version 1.0 User's Guide",
#     GemSoft Enterprises -- the .ASC format and the rule that three
#     pieces of information (angle/index/point combinations) determine a
#     facet plane.
#   Robert H. Long & Norman W. Steele, "Introduction to Meetpoint
#     Faceting", Seattle Faceting Books, 1985 -- meetpoint methodology.
#   United States Faceters Guild, "Gemstone Design Conversion Using the
#     Tangent Ratio Method" -- the angle conversion in `tangent_ratio`.

import math
from typing import NamedTuple

import numpy as np

# A tier angle within this many degrees of vertical is a girdle facet.
_GIRDLE_EPS = 1e-9
# Below this the tier is horizontal (table, or culet when distance < 0).
_FLAT_EPS = 1e-12


class Tier(NamedTuple):
    """One row of facets: a mast angle, a distance, and index positions.

    `angle_deg` follows GemCad's sign convention -- positive on the crown,
    negative on the pavilion, +-90 for the girdle, 0 for the table.  A
    CULET facet (a pavilion facet at 0 degrees) is indistinguishable from
    the table by angle alone, so it is written the way GemCad writes it:
    angle 0 with a NEGATIVE `distance`.  `distance` is otherwise the
    positive centre-to-facet distance.
    """

    angle_deg: float
    distance: float
    indices: tuple
    name: str = ""
    note: str = ""


class CutDesign(NamedTuple):
    """A complete cut: gear, symmetry, design refractive index, tiers."""

    name: str = ""
    gear: int = 96
    gear_offset: float = 0.0
    fold: int = 1
    mirror: bool = False
    ri: float = 1.54
    tiers: tuple = ()
    headings: tuple = ()
    footnotes: tuple = ()

    @property
    def facet_count(self):
        return sum(len(t.indices) for t in self.tiers)


def index_azimuth(index, gear, gear_offset=0.0):
    """Azimuth (radians) of an index position on the gear.

    Index numbering runs counter-clockwise seen from +z (the table side),
    with index 0 == index `gear` along +x.  `gear_offset` is taken in
    DEGREES, added to the azimuth.

    The handedness is a convention, not a fact recoverable from a design
    file: mirroring it produces the enantiomorph, which for the mirror-
    symmetric designs in the catalogue is the same solid.  It is pinned
    for real files by the round-trip and mirror-symmetry checks in
    `asc.py` rather than assumed here.
    """
    return 2.0 * math.pi * ((index % gear) / float(gear)) \
        + math.radians(gear_offset)


def facet_plane(angle_deg, distance, phi):
    """(outward unit normal, offset) of one facet at azimuth `phi`.

    Returns `(nx, ny, nz), d` for the half-space `n . x <= d`.
    """
    a = float(angle_deg)
    if abs(abs(a) - 90.0) < _GIRDLE_EPS:            # girdle: vertical facet
        return (math.cos(phi), math.sin(phi), 0.0), abs(distance)
    if abs(a) < _FLAT_EPS and distance < 0.0:       # culet (0 deg pavilion)
        return (0.0, 0.0, -1.0), -distance
    s = -1.0 if a < 0.0 else 1.0                    # crown +, pavilion -
    t = math.radians(abs(a))
    st, ct = math.sin(t), math.cos(t)
    return (st * math.cos(phi), st * math.sin(phi), s * ct), float(distance)


def planes(design):
    """Expand a `CutDesign` to plane arrays.

    Returns `(N, d, tier_id)` with `N` an (M, 3) array of outward unit
    normals, `d` an (M,) array of offsets and `tier_id` an (M,) int array
    naming the tier each plane came from -- one row per (tier, index).
    """
    N, d, tid = [], [], []
    for k, t in enumerate(design.tiers):
        for idx in t.indices:
            phi = index_azimuth(idx, design.gear, design.gear_offset)
            n, off = facet_plane(t.angle_deg, t.distance, phi)
            N.append(n)
            d.append(off)
            tid.append(k)
    if not N:
        return (np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int))
    return (np.asarray(N, dtype=float), np.asarray(d, dtype=float),
            np.asarray(tid, dtype=int))


def expand_indices(fold, mirror, seed, gear=96):
    """Full index list generated by `seed` under the design's symmetry.

    `fold`-fold radial symmetry steps by `gear // fold`; `mirror` adds the
    reflection about index 0.  Returns a sorted tuple of distinct indices
    in `[0, gear)`.

    This is an authoring helper for the catalogue -- published designs
    already list every index explicitly, and `asc.py` keeps those lists
    verbatim rather than re-deriving them.
    """
    if gear % fold:
        raise ValueError(f"{fold}-fold symmetry needs a gear divisible by "
                         f"{fold}, got {gear}")
    step = gear // fold
    out = set()
    for s in seed:
        for k in range(fold):
            out.add((s + k * step) % gear)
            if mirror:
                out.add((-s + k * step) % gear)
    return tuple(sorted(out))


def tangent_ratio(design, old_ref_deg, new_ref_deg, half="pavilion",
                  girdle_z=0.0):
    """Tangent-ratio conversion: re-angle one half, keep the plan view.

    A design cut for one refractive index leaks light in another.  The
    lapidary's fix is to scale the stone vertically, leaving the plan
    (x, y) view untouched, which changes every elevation angle by the
    tangent law

        tan(theta') = T tan(theta),   T = tan(new_ref) / tan(old_ref)

    applied with the SAME T to every facet in that half.  Geometrically
    this is the linear map z -> T (z - girdle_z) + girdle_z, and pushing
    a plane through it gives the unnormalised normal
    (sin t cos phi, sin t sin phi, n_z / T) with offset
    d + n_z girdle_z (1/T - 1); normalising both yields the new angle and
    distance.

    NOTE the distance rule that falls out is d' = d sin(theta') / sin(theta)
    when `girdle_z` is 0 -- NOT cos(theta')/cos(theta).  The invariant
    being preserved is the facet plane's trace on the girdle plane,
    rho = d / sin(theta), which is exactly "the plan view is unchanged";
    the self-test pins it.

    Girdle tiers (+-90 degrees) pass through untouched.  The table scales
    with T, which is the point: the crown gets taller.
    """
    want_crown = (half == "crown")
    T = math.tan(math.radians(new_ref_deg)) / math.tan(math.radians(old_ref_deg))
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"degenerate tangent ratio T={T} from "
                         f"{old_ref_deg} -> {new_ref_deg}")
    out = []
    for t in design.tiers:
        a = float(t.angle_deg)
        is_girdle = abs(abs(a) - 90.0) < _GIRDLE_EPS
        is_culet = abs(a) < _FLAT_EPS and t.distance < 0.0
        on_crown = (a > 0.0) or (abs(a) < _FLAT_EPS and not is_culet)
        if is_girdle or (on_crown != want_crown):
            out.append(t)
            continue
        th = math.radians(abs(a))
        sgn = -1.0 if (a < 0.0 or is_culet) else 1.0
        nz = sgn * math.cos(th)
        m = math.hypot(math.sin(th), nz / T)
        new_a = math.degrees(math.atan2(math.sin(th), abs(nz) / T))
        dist = abs(t.distance) if is_culet else t.distance
        new_d = (dist + nz * girdle_z * (1.0 / T - 1.0)) / m
        if sgn < 0.0:
            new_a = -new_a
            if is_culet:                     # stays a culet: angle 0, d < 0
                new_a, new_d = 0.0, -new_d
        out.append(t._replace(angle_deg=new_a, distance=new_d))
    return design._replace(tiers=tuple(out))


def tier_from_angle_point(gear, gear_offset, index, angle_deg, p):
    """Centre-to-facet distance of the plane at (angle, index) through `p`.

    GemCad's commonest cutting rule -- "angle, index and one point" -- and
    the one meetpoint faceting leans on: the point is a meetpoint where
    already-cut facets come together.
    """
    phi = index_azimuth(index, gear, gear_offset)
    n, _ = facet_plane(angle_deg, 1.0, phi)
    return float(np.dot(np.asarray(n, dtype=float), np.asarray(p, dtype=float)))


def tier_from_points(gear, gear_offset, index, p1, p2):
    """Solve (angle, distance) for the plane at `index` through two points.

    GemCad's "index and two points" rule: the index fixes the azimuth, so
    one degree of freedom -- the elevation -- is left, and two points on
    the plane determine it.

    Raises ValueError if the two points do not constrain the elevation, or
    if the resulting facet would put the origin on its outer side (a
    negative centre-to-facet distance) -- which means the plane belongs at
    `index + gear/2`, not here.

    The two points determine the PLANE, but not which way its normal
    points, and `atan2` alone would let the order of `p1`, `p2` decide.
    A facet at `index` is the one whose outward normal leans along that
    index's azimuth, so the elevation is normalised into [0, pi] --
    sin(a) >= 0 -- which fixes the orientation independently of the
    argument order.
    """
    phi = index_azimuth(index, gear, gear_offset)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p1 - p2
    h = v[0] * math.cos(phi) + v[1] * math.sin(phi)
    if abs(h) < 1e-12 and abs(v[2]) < 1e-12:
        raise ValueError("the two points coincide; elevation undetermined")
    a = math.atan2(-v[2], h)                      # n . v == 0
    if math.sin(a) < 0.0:                         # face outward at `index`
        a += math.pi
    a = a % (2.0 * math.pi)
    n = np.array([math.sin(a) * math.cos(phi), math.sin(a) * math.sin(phi),
                  math.cos(a)])
    d = float(np.dot(n, p1))
    if d < 0.0:
        raise ValueError(f"plane through these points has the origin on its "
                         f"outer side; it belongs at index "
                         f"{(index + gear // 2) % gear}, not {index}")
    deg = math.degrees(a)
    angle_deg = deg if deg <= 90.0 + 1e-9 else deg - 180.0
    return angle_deg, d


# --------------------------------------------------------------------------
# The Standard Round Brilliant, verbatim from the GemCad manual's .ASC
# example (Strickland, "GemCad File Types").  Kept here as the fixture the
# engine is proved against: 16-sided outline, 8-fold mirror symmetry,
# cut for a refractive index of 1.54.
# --------------------------------------------------------------------------

SRB_GEMCAD = CutDesign(
    name="Standard Round Brilliant",
    gear=96, gear_offset=0.0, fold=8, mirror=True, ri=1.54,
    headings=("Standard Round Brilliant", "9/5/02"),
    footnotes=("GemCad for Windows Manual",),
    tiers=(
        Tier(-90.0, 1.02653281,
             (93, 87, 81, 75, 69, 63, 57, 51, 45, 39, 33, 27, 21, 15, 9, 3),
             name="G", note="16-sided outline"),
        Tier(-42.5, 0.61819401,
             (93, 87, 81, 75, 69, 63, 57, 51, 45, 39, 33, 27, 21, 15, 9, 3),
             name="1"),
        Tier(-41.5, 0.61701256, (96, 84, 72, 60, 48, 36, 24, 12), name="2"),
        Tier(34.0, 0.68444470,
             (3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93),
             name="A"),
        Tier(28.0, 0.60896430, (96, 12, 24, 36, 48, 60, 72, 84), name="B"),
        Tier(16.0, 0.50613241, (6, 18, 30, 42, 54, 66, 78, 90), name="C"),
        Tier(0.0, 0.36450932, (96,), name="T"),
    ),
)


def _selftest():
    ok = True

    # --- plane construction ------------------------------------------------
    N, d, tid = planes(SRB_GEMCAD)
    good = N.shape == (73, 3) and d.shape == (73,)
    ok &= good
    print(f"gems.design: SRB expands to 73 facet planes "
          f"(got {N.shape[0]}) {'OK' if good else 'not 73'}")

    norms = np.linalg.norm(N, axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    good = worst < 1e-12
    ok &= good
    print(f"gems.design: every normal is unit length (worst {worst:.2e}) "
          f"{'OK' if good else 'not unit'}")

    # tier senses: girdle horizontal, pavilion down, crown/table up
    gz = N[tid == 0][:, 2]
    pz = N[tid == 1][:, 2]
    cz = N[tid == 3][:, 2]
    tz = N[tid == 6][:, 2]
    good = (np.allclose(gz, 0.0) and np.all(pz < 0) and np.all(cz > 0)
            and np.allclose(tz, 1.0))
    ok &= good
    print(f"gems.design: girdle horizontal, pavilion down, crown and table up "
          f"{'OK' if good else 'wrong sense'}")

    # the table plane is the one horizontal facet, at its stated height
    good = abs(d[tid == 6][0] - 0.36450932) < 1e-12
    ok &= good
    print(f"gems.design: table offset preserved exactly "
          f"{'OK' if good else 'drifted'}")

    # --- the culet convention (angle 0, negative distance) -----------------
    n_c, d_c = facet_plane(0.0, -0.3, 0.0)
    n_t, d_t = facet_plane(0.0, 0.3, 0.0)
    good = n_c == (0.0, 0.0, -1.0) and d_c == 0.3 and n_t == (0.0, 0.0, 1.0)
    ok &= good
    print(f"gems.design: angle 0 with negative distance reads as a culet "
          f"{'OK' if good else 'misread'}")

    # --- symmetry expansion -----------------------------------------------
    got = expand_indices(8, True, (3,), 96)
    want = tuple(range(3, 96, 6))
    good = got == want
    ok &= good
    print(f"gems.design: 8-fold mirror on seed 3 rebuilds the girdle's 16 "
          f"indices {'OK' if good else f'got {got}'}")
    good = (len(expand_indices(8, True, (0,), 96)) == 8
            and len(expand_indices(8, True, (6,), 96)) == 8)
    ok &= good
    print(f"gems.design: seeds on and off the mirror give 8 indices each "
          f"{'OK' if good else 'wrong count'}")

    # --- plane-through-points solvers -------------------------------------
    phi = index_azimuth(12, 96)
    n0, d0 = facet_plane(37.5, 0.7, phi)
    n0 = np.asarray(n0)
    # two independent points on that plane
    u = np.cross(n0, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    w = np.cross(n0, u)
    base = n0 * d0
    a_deg, dist = tier_from_points(96, 0.0, 12, base + 0.3 * u, base + 0.2 * w)
    good = abs(a_deg - 37.5) < 1e-9 and abs(dist - 0.7) < 1e-9
    ok &= good
    print(f"gems.design: tier_from_points recovers (37.5, 0.7) -> "
          f"({a_deg:.9f}, {dist:.9f}) {'OK' if good else 'mismatch'}")

    dist2 = tier_from_angle_point(96, 0.0, 12, 37.5, base + 0.3 * u)
    good = abs(dist2 - 0.7) < 1e-12
    ok &= good
    print(f"gems.design: tier_from_angle_point recovers the distance "
          f"{'OK' if good else 'mismatch'}")

    n1, dd1 = facet_plane(-42.5, 0.6, phi)
    n1 = np.asarray(n1)
    u1 = np.cross(n1, [0, 0, 1.0])
    u1 /= np.linalg.norm(u1)
    w1 = np.cross(n1, u1)
    b1 = n1 * dd1
    a1, r1 = tier_from_points(96, 0.0, 12, b1 + 0.3 * u1, b1 + 0.2 * w1)
    good = abs(a1 + 42.5) < 1e-9 and abs(r1 - 0.6) < 1e-9
    ok &= good
    print(f"gems.design: tier_from_points handles a pavilion tier "
          f"({a1:.6f}) {'OK' if good else 'mismatch'}")

    # the solver must not depend on the order of its two points -- the
    # bug the [0, pi] normalisation exists to kill
    a_swap, r_swap = tier_from_points(96, 0.0, 12, base + 0.2 * w,
                                      base + 0.3 * u)
    good = abs(a_swap - a_deg) < 1e-12 and abs(r_swap - dist) < 1e-12
    ok &= good
    print(f"gems.design: tier_from_points is independent of point order "
          f"{'OK' if good else 'order dependent'}")

    # a plane belonging at the opposite index is refused, not relabelled:
    # translate the same plane past the origin so its outward side holds it
    far = base - 2.5 * n0
    raised = False
    try:
        tier_from_points(96, 0.0, 12, far + 0.3 * u, far + 0.2 * w)
    except ValueError:
        raised = True
    ok &= raised
    print(f"gems.design: a plane belonging at the opposite index is refused "
          f"{'OK' if raised else 'silently accepted'}")

    # --- tangent-ratio conversion -----------------------------------------
    same = tangent_ratio(SRB_GEMCAD, 40.75, 40.75, half="pavilion")
    good = all(abs(a.angle_deg - b.angle_deg) < 1e-9
               and abs(a.distance - b.distance) < 1e-9
               for a, b in zip(same.tiers, SRB_GEMCAD.tiers))
    ok &= good
    print(f"gems.design: T = 1 is the identity "
          f"{'OK' if good else 'perturbed the design'}")

    conv = tangent_ratio(SRB_GEMCAD, 42.5, 45.0, half="pavilion")
    T = math.tan(math.radians(45.0)) / math.tan(math.radians(42.5))
    ratios, rhos = [], []
    for a, b in zip(SRB_GEMCAD.tiers, conv.tiers):
        # only tilted pavilion tiers: the girdle is exempt, the crown and
        # table belong to the other half, and a culet (angle 0) has no
        # tangent to take a ratio of
        if abs(abs(a.angle_deg) - 90.0) < 1e-9 or a.angle_deg >= 0.0:
            continue
        ratios.append(math.tan(math.radians(abs(b.angle_deg)))
                      / math.tan(math.radians(abs(a.angle_deg))))
        rhos.append((a.distance / math.sin(math.radians(abs(a.angle_deg))),
                     b.distance / math.sin(math.radians(abs(b.angle_deg)))))
    good = ratios and max(abs(r - T) for r in ratios) < 1e-12
    ok &= good
    print(f"gems.design: every pavilion tier scales by the same tangent "
          f"ratio T={T:.6f} {'OK' if good else 'inconsistent'}")

    worst = max(abs(x - y) for x, y in rhos)
    good = worst < 1e-12
    ok &= good
    print(f"gems.design: the plan view d/sin(theta) is preserved "
          f"(worst {worst:.2e}) {'OK' if good else 'plan view moved'}")

    # the crown, and the girdle, are untouched by a pavilion conversion
    good = all(abs(a.angle_deg - b.angle_deg) < 1e-12
               for a, b in zip(SRB_GEMCAD.tiers[3:], conv.tiers[3:]))
    good &= abs(conv.tiers[0].distance - SRB_GEMCAD.tiers[0].distance) < 1e-12
    ok &= good
    print(f"gems.design: a pavilion conversion leaves crown and girdle alone "
          f"{'OK' if good else 'touched the other half'}")

    # a culet survives as a culet (angle 0, negative distance)
    culet = SRB_GEMCAD._replace(
        tiers=SRB_GEMCAD.tiers + (Tier(0.0, -0.05, (96,), name="U"),))
    cc = tangent_ratio(culet, 42.5, 45.0, half="pavilion").tiers[-1]
    good = abs(cc.angle_deg) < 1e-12 and cc.distance < 0
    ok &= good
    print(f"gems.design: a culet stays a culet through the conversion "
          f"(d={cc.distance:.6f}) {'OK' if good else 'lost the convention'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.design self-test failed")
