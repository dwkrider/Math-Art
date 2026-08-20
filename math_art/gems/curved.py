# Cabochons: the cuts that are not polyhedra.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python only -- no
# `bpy`, no numpy.
#
# Every other cut in this package is an intersection of half-spaces, and
# is built by `facets.py`.  A cabochon is not: its surface is CURVED, so
# there are no facet planes to intersect and none of that machinery
# applies.  It gets its own module and its own builder, emitting vertices
# and faces directly.
#
# That is not a shortcut around the facet engine, it is the older cut.
# Polishing a dome needs no index gear and no protractor, only a curve,
# which is why the cabochon predates faceting entirely and why the stones
# that are still cut this way are the ones where facets would be pointless
# or destructive: opaque and translucent material that has no light to
# return (turquoise, lapis, jade), and the phenomenal stones whose whole
# display depends on a curved surface -- a star ruby's asterism and a
# cat's-eye's chatoyancy are reflections off oriented inclusions, and both
# need a dome to gather them into a line.
#
# The shapes here are the standard family:
#
#   single      flat base, domed top -- the common cabochon
#   double      domed both sides, for translucent material lit from behind
#   high        a tall dome; the usual choice for asterism and chatoyancy
#   sugarloaf   a square outline drawn up to a point
#   bullet      a round outline drawn up to a point
#   lentil      a shallow symmetric lens
#
# The dome profile is a superellipse exponent rather than a fixed curve,
# so one parameter carries the family from a hemisphere through the
# ordinary flattened cabochon to a sharp sugarloaf, and the named presets
# are points on it rather than special cases in the code.
#
# References:
#   John Sinkankas, "Gem Cutting: A Lapidary's Manual", 2nd ed. --
#     cabochon proportions and the reasons for them.
#   Robert Webster, "Gems: Their Sources, Descriptions and
#     Identification", 5th ed. -- asterism and chatoyancy, and why both
#     require a domed surface correctly oriented to the inclusions.
#   CIBJO Coloured Stone Commission, "The Gemstone Book" (Blue Book) --
#     cutting-style nomenclature.

import math
from typing import NamedTuple


class Cabochon(NamedTuple):
    """A cabochon's shape, before it becomes a mesh."""

    name: str
    lw: float = 1.0             # length-to-width ratio of the outline
    height: float = 0.45        # dome height as a fraction of the WIDTH
    dome: float = 2.0           # superellipse exponent of the profile
    corners: int = 0            # 0 = round outline; 4 = square, 6 = hex ...
    base_dome: float = 0.0      # 0 = flat base; > 0 = belly, as a height
    girdle: float = 0.0         # straight girdle band, fraction of width


def _outline(t, lw, corners):
    """A point on the outline at parameter t in [0, 1)."""
    a = 2.0 * math.pi * t
    if corners < 3:
        return math.cos(a), lw * math.sin(a)
    # a superellipse-ish rounded polygon: the squircle exponent rises with
    # the corner count so a square reads square and a hexagon reads hex
    n = float(corners)
    k = math.pi / n
    # radius of a regular n-gon with a slightly rounded corner
    r = math.cos(k) / max(1e-9, math.cos(((a % (2.0 * k)) - k)))
    r = min(r, 1.6)
    return r * math.cos(a), lw * r * math.sin(a)


def build_cabochon(cab, rings=24, segments=64):
    """(verts, faces) for a cabochon.

    The dome is a superellipse in the vertical section: at a fraction `u`
    of the way out from the apex the height follows (1 - u**dome)**(1/dome)
    for the symmetric case, which gives a hemisphere at dome = 2, a
    flatter cabochon below it and a sharp point above.
    """
    if cab.lw <= 0.0:
        raise ValueError(f"length-to-width must be positive, got {cab.lw}")
    if cab.height <= 0.0:
        raise ValueError(f"height must be positive, got {cab.height}")
    if cab.dome <= 0.0:
        raise ValueError(f"dome exponent must be positive, got {cab.dome}")
    if rings < 2 or segments < 3:
        raise ValueError("need at least 2 rings and 3 segments")

    verts, faces = [], []
    h = cab.height * 2.0            # height as a fraction of the WIDTH (=2)
    gz = cab.girdle * 2.0 * 0.5

    def profile(u):
        """Height of the dome at radial fraction u in [0, 1]."""
        u = min(1.0, max(0.0, u))
        return h * (1.0 - u ** cab.dome) ** (1.0 / cab.dome)

    # --- the dome, apex first ------------------------------------------
    apex = len(verts)
    verts.append((0.0, 0.0, gz + h))
    for r in range(1, rings + 1):
        u = r / rings
        z = gz + profile(u)
        for s in range(segments):
            x, y = _outline(s / segments, cab.lw, cab.corners)
            verts.append((u * x, u * y, z))
    for s in range(segments):
        faces.append((apex, 1 + (s + 1) % segments, 1 + s))
    for r in range(rings - 1):
        a0 = 1 + r * segments
        b0 = 1 + (r + 1) * segments
        for s in range(segments):
            s1 = (s + 1) % segments
            faces.append((a0 + s, a0 + s1, b0 + s1, b0 + s))

    rim = 1 + (rings - 1) * segments        # first vertex of the last ring

    # --- the girdle band, if any ----------------------------------------
    if gz > 0.0:
        low = len(verts)
        for s in range(segments):
            x, y = _outline(s / segments, cab.lw, cab.corners)
            verts.append((x, y, -gz))
        for s in range(segments):
            s1 = (s + 1) % segments
            faces.append((rim + s, rim + s1, low + s1, low + s))
        rim = low

    # --- the base: flat, or a belly -------------------------------------
    if cab.base_dome > 0.0:
        hb = cab.base_dome * 2.0
        for r in range(1, rings + 1):
            u = 1.0 - r / rings
            z = -gz - hb * (1.0 - u ** cab.dome) ** (1.0 / cab.dome)
            row = len(verts)
            if r < rings:
                for s in range(segments):
                    x, y = _outline(s / segments, cab.lw, cab.corners)
                    verts.append((u * x, u * y, z))
                prev = rim if r == 1 else row - segments
                for s in range(segments):
                    s1 = (s + 1) % segments
                    faces.append((prev + s1, prev + s, row + s, row + s1))
            else:
                verts.append((0.0, 0.0, -gz - hb))
                prev = row - segments
                for s in range(segments):
                    s1 = (s + 1) % segments
                    faces.append((prev + s, prev + s1, row))
    else:
        base = len(verts)
        verts.append((0.0, 0.0, -gz))
        for s in range(segments):
            s1 = (s + 1) % segments
            faces.append((rim + s1, rim + s, base))
    return verts, faces


CABOCHONS = {
    "SINGLE": Cabochon("Cabochon", height=0.42, dome=2.4),
    "HIGH": Cabochon("High Cabochon", height=0.75, dome=2.2,
                     girdle=0.02),
    "LENTIL": Cabochon("Lentil Cabochon", height=0.25, dome=2.6,
                       base_dome=0.25),
    "DOUBLE": Cabochon("Double Cabochon", height=0.45, dome=2.4,
                       base_dome=0.30),
    "OVAL": Cabochon("Oval Cabochon", lw=1.35, height=0.40, dome=2.4),
    "SUGARLOAF": Cabochon("Sugarloaf", height=0.85, dome=1.15, corners=4),
    "BULLET": Cabochon("Bullet Cabochon", height=0.90, dome=1.2),
    "HEX": Cabochon("Hexagonal Cabochon", height=0.45, dome=2.2, corners=6),
}


def get(key):
    try:
        return CABOCHONS[key]
    except KeyError:
        raise KeyError(f"no cabochon named {key!r}; known: "
                       f"{', '.join(sorted(CABOCHONS))}") from None


def cabochon_items():
    return [(k, CABOCHONS[k].name,
             f"{'domed both sides' if CABOCHONS[k].base_dome else 'flat base'}"
             f", height {CABOCHONS[k].height * 100:.0f}% of the width")
            for k in sorted(CABOCHONS)]


def _closed(verts, faces):
    """(watertight, euler) for a triangle/quad soup."""
    edges = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            k = (min(a, b), max(a, b))
            edges[k] = edges.get(k, 0) + 1
    return set(edges.values()) == {2}, len(verts) - len(edges) + len(faces)


def _selftest():
    ok = True

    # --- every preset is a closed surface of genus 0 ---------------------
    bad = []
    for key in sorted(CABOCHONS):
        v, f = build_cabochon(get(key))
        closed, chi = _closed(v, f)
        if not closed or chi != 2:
            bad.append(f"{key}(closed={closed}, chi={chi})")
    good = not bad
    ok &= good
    print(f"gems.curved: all {len(CABOCHONS)} cabochons are closed surfaces "
          f"with Euler 2 {'OK' if good else ', '.join(bad)}")

    # --- a flat base really is flat --------------------------------------
    v, f = build_cabochon(get("SINGLE"))
    zmin = min(p[2] for p in v)
    # a flat base puts its whole rim AND its centre on one plane, so the
    # test is that nothing dips below it, not that one vertex is lowest
    flat = [p for p in v if abs(p[2] - zmin) < 1e-12]
    good = len(flat) > 3 and all(abs(p[2] - zmin) < 1e-12 or p[2] > zmin
                                 for p in v)
    ok &= good
    print(f"gems.curved: a single cabochon's base is flat -- {len(flat)} "
          f"vertices on one plane, none below "
          f"{'OK' if good else 'not flat'}")
    good = abs(zmin) < 1e-12
    ok &= good
    print(f"gems.curved: its base sits on z = 0 (got {zmin:+.2e}) "
          f"{'OK' if good else 'off the plane'}")

    # --- a double cabochon is domed BOTH ways ----------------------------
    v2, f2 = build_cabochon(get("DOUBLE"))
    lo = min(p[2] for p in v2)
    n_lo = sum(1 for p in v2 if abs(p[2] - lo) < 1e-9)
    good = n_lo == 1 and lo < -0.1
    ok &= good
    print(f"gems.curved: a double cabochon comes to a point below "
          f"(z {lo:+.3f}) {'OK' if good else 'flat underneath'}")

    # --- height is honoured ----------------------------------------------
    worst = 0.0
    for hgt in (0.25, 0.45, 0.9):
        v3, _ = build_cabochon(Cabochon("t", height=hgt))
        w = max(p[0] for p in v3) - min(p[0] for p in v3)
        got = (max(p[2] for p in v3) - 0.0) / w
        worst = max(worst, abs(got - hgt))
    good = worst < 1e-9
    ok &= good
    print(f"gems.curved: the dome height is a fraction of the WIDTH "
          f"(worst error {worst:.1e}) {'OK' if good else 'wrong'}")

    # --- the dome exponent runs the family -------------------------------
    # low exponent -> pointed (sugarloaf); high -> flat-topped
    def apex_sharpness(dome):
        v4, _ = build_cabochon(Cabochon("t", height=0.5, dome=dome),
                               rings=32)
        top = max(p[2] for p in v4)
        # how far out you must go to drop 10% of the height
        near = [math.hypot(p[0], p[1]) for p in v4 if p[2] > 0.9 * top]
        return max(near)
    sharp, round_, flat = (apex_sharpness(1.2), apex_sharpness(2.0),
                           apex_sharpness(4.0))
    good = sharp < round_ < flat
    ok &= good
    print(f"gems.curved: the dome exponent runs pointed to flat-topped "
          f"({sharp:.3f} < {round_:.3f} < {flat:.3f}) "
          f"{'OK' if good else 'no ordering'}")

    # --- an oval outline is oval -----------------------------------------
    v5, _ = build_cabochon(get("OVAL"))
    w = max(p[0] for p in v5) - min(p[0] for p in v5)
    ln = max(p[1] for p in v5) - min(p[1] for p in v5)
    good = abs(ln / w - 1.35) < 1e-9
    ok &= good
    print(f"gems.curved: the oval measures {ln / w:.3f}:1, asked for 1.350 "
          f"{'OK' if good else 'wrong'}")

    # --- a square outline has four corners --------------------------------
    v6, _ = build_cabochon(get("SUGARLOAF"), segments=64)
    # the base centre sits at radius 0 and would divide the ratio to
    # nothing; the outline is the rim, not the fan's hub
    rr = [math.hypot(p[0], p[1]) for p in v6
          if abs(p[2]) < 1e-9 and math.hypot(p[0], p[1]) > 1e-9]
    ratio = (max(rr) / min(rr)) if rr else 0.0
    good = ratio > 1.25                          # a circle would be 1.0
    ok &= good
    print(f"gems.curved: a sugarloaf's outline is square, not round "
          f"(corner/edge radius {ratio:.2f}) {'OK' if good else 'round'}")

    # --- a girdle band adds a straight wall -------------------------------
    v7, f7 = build_cabochon(get("HIGH"))
    closed, chi = _closed(v7, f7)
    good = closed and chi == 2
    ok &= good
    print(f"gems.curved: a cabochon with a girdle band still closes "
          f"{'OK' if good else 'open'}")

    # --- refusals ----------------------------------------------------------
    bad = 0
    for kw in (dict(lw=0.0), dict(height=0.0), dict(dome=0.0)):
        try:
            build_cabochon(Cabochon("t", **kw))
        except ValueError:
            bad += 1
    try:
        build_cabochon(get("SINGLE"), rings=1)
    except ValueError:
        bad += 1
    good = bad == 4
    ok &= good
    print(f"gems.curved: impossible cabochons are refused ({bad}/4) "
          f"{'OK' if good else 'accepted nonsense'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.curved self-test failed")
