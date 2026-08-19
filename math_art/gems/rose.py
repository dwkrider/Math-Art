# Rose cuts: a domed crown of triangles on a flat base.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python only -- no
# `bpy`, no numpy.
#
# The rose predates the brilliant, and it is the one common cut with NO
# PAVILION at all: a flat base, and above it rows of triangular facets
# rising to a point.  It was cut for candlelight and for flat rough,
# where a brilliant's depth would have been wasted material.
#
# Structurally that makes it the simplest family here -- one horizontal
# plane facing down, a girdle, and crown rows -- but it exercises a
# corner of the engine nothing else does.  Every other cut in the
# catalogue is bounded below by facets that converge; a rose is bounded
# below by a single plane, so if the base were ever dropped as
# "carrying no facet" the solid would be open, and the closure check is
# what would catch it.
#
# Rows are offset by half a step from each other, so each facet spans the
# joint between two below it and comes out triangular -- which for the
# rose is not an approximation but the defining look.
#
# References:
#   Marcel Tolkowsky, "Diamond Design", E. & F. N. Spon, London, 1919,
#     Part I -- the historical sequence table cut -> rose -> brilliant,
#     and Part III, "The Rose", which analyses the cut's optics.
#   Basil Watermeyer, "Diamond Cutting: A Complete Guide to Diamond
#     Processing", 2nd ed. -- rose and antique cutting practice.

import math

try:
    from .design import CutDesign, Tier, tangent_ratio
except ImportError:                     # flat import outside the package
    from design import CutDesign, Tier, tangent_ratio


def _idx(azimuth_deg, gear):
    return int(round(azimuth_deg / 360.0 * gear)) % gear


def rose_cut(n=8, rows=2, crown_angle=45.0, apex_angle=25.0, height=0.55,
             girdle_pct=0.02, gear=96, ri=1.54, name="Rose Cut"):
    """A rose: flat base, `rows` rows of facets rising to a point.

    `n` is the fold, so the crown carries `n * rows` facets: the default
    8-fold, two-row rose has 24, the count quoted for the Dutch (Holland)
    rose.  `height` is the dome's height as a fraction of the diameter.

    Angles run from `crown_angle` at the girdle to `apex_angle` at the
    top; they must decrease, since two rows at one elevation are parallel
    planes and the inner would cut the outer away.
    """
    if n < 3:
        raise ValueError(f"a rose needs at least 3 fold, got {n}")
    if rows < 1:
        raise ValueError(f"a rose needs at least one row, got {rows}")
    if not 0.0 < apex_angle < crown_angle < 90.0:
        raise ValueError(f"angles must decrease from the girdle inward and "
                         f"lie in (0, 90), got {crown_angle} -> {apex_angle}")
    if height <= 0.0:
        raise ValueError(f"height must be positive, got {height}")

    r_g = 1.0
    r_v = r_g / math.cos(math.pi / n)
    step = 360.0 / n
    z_base = -girdle_pct * r_v if girdle_pct > 0.0 else 0.0

    tiers = [
        # the base: a single downward plane.  GemCad's culet convention --
        # angle 0 with a negative distance -- is exactly this facet.
        Tier(0.0, z_base if z_base < 0.0 else -1e-9 - 0.0,
             (_idx(0.0, gear),), name="B", note="flat base"),
        Tier(-90.0, r_g,
             tuple(_idx((k + 0.5) * step, gear) for k in range(n)),
             name="G", note=f"{n}-sided girdle"),
    ]
    if z_base == 0.0:
        # A rose asked for with no girdle band still needs a base plane,
        # but the band's facets then have zero height and survive only as
        # slivers; the default girdle_pct is non-zero for that reason.
        tiers[0] = Tier(0.0, -1e-12, (_idx(0.0, gear),), name="B",
                        note="flat base")

    dp = r_v / rows
    z, shrink = 0.0, 0.0
    for k in range(rows):
        f = k / max(1, rows - 1)
        ang = crown_angle + (apex_angle - crown_angle) * f
        a = math.radians(ang)
        p = r_v - shrink
        if p <= 1e-9:
            break
        off = 0.5 * (k % 2)
        tiers.append(Tier(ang, math.sin(a) * p + math.cos(a) * z,
                          tuple(_idx((j + 0.5 + off) * step, gear)
                                for j in range(n)),
                          name=f"R{k + 1}", note=f"rose row {k + 1}"))
        z += dp * math.tan(a)
        shrink += dp

    out = CutDesign(name=name, gear=gear, fold=n, mirror=True, ri=ri,
                    tiers=tuple(tiers),
                    headings=(name, f"{n}-fold, {rows} rows, "
                                    f"{n * rows} crown facets"))

    # Scale the dome to the requested height.  This is a tangent-ratio
    # conversion, so it must carry the distance rule d' = d sin(t')/sin(t)
    # as well as the angle rule -- scaling the angles alone leaves every
    # facet plane at its old distance and the dome comes out the wrong
    # height.  Reuse the transform in design.py rather than repeat it.
    want = height * 2.0 * r_v
    if z > 1e-9 and abs(want - z) > 1e-12:
        scale = want / z
        out = tangent_ratio(out, 45.0, math.degrees(math.atan(scale)),
                            half="crown", girdle_z=0.0)
    return out


def dutch_rose(**over):
    """The Dutch (Holland) rose: 24 crown facets in two rows.

    24 is the facet count the Dutch rose is quoted at, reached here as a
    uniform 12-fold pair of rows.  Historical roses distribute those 24
    less evenly -- a small ring of star facets over a larger ring -- so
    this matches the cut by count and character rather than reproducing
    any one antique stone.
    """
    kw = dict(n=12, rows=2, name="Dutch Rose")
    kw.update(over)
    return rose_cut(**kw)


def antwerp_rose(**over):
    """The Antwerp rose: 6-fold, a shallower dome and fewer facets."""
    kw = dict(n=6, rows=2, crown_angle=40.0, apex_angle=20.0, height=0.40,
              name="Antwerp Rose")
    kw.update(over)
    return rose_cut(**kw)


def _selftest():
    try:
        from . import facets as _f
        from .design import planes
    except ImportError:
        import facets as _f
        from design import planes

    ok = True

    def build(D):
        N, d, t = planes(D)
        return _f.intersect_halfspaces(N, d), N, d, t

    D = dutch_rose()
    P, N, d, t = build(D)
    chk = _f.polytope_checks(P, N, d)
    good = chk["closed"] and chk["convex"]
    ok &= good
    print(f"gems.rose: the Dutch rose closes and is convex "
          f"({len(P.V)} vertices, {len(P.faces)} facets) "
          f"{'OK' if good else 'did not build'}")

    names = [D.tiers[t[j]].name for j in P.face_plane]
    crown = names.count("R1") + names.count("R2")
    good = crown == 24
    ok &= good
    print(f"gems.rose: 24 crown facets, the Dutch rose's count "
          f"(got {crown}) {'OK' if good else 'wrong count'}")

    # the defining feature: ONE flat plane underneath, and it survives
    good = names.count("B") == 1
    ok &= good
    print(f"gems.rose: the flat base is a single facet "
          f"(got {names.count('B')}) {'OK' if good else 'base lost'}")

    V = P.V.tolist()
    z_min = min(v[2] for v in V)
    base = [v for v in V if abs(v[2] - z_min) < 1e-9]
    good = len(base) >= 6
    ok &= good
    print(f"gems.rose: the base is a polygon of {len(base)} vertices, not a "
          f"point {'OK' if good else 'converged'}")

    # and it rises to a point
    z_max = max(v[2] for v in V)
    tip = [v for v in V if abs(v[2] - z_max) < 1e-9]
    good = len(tip) == 1
    ok &= good
    print(f"gems.rose: the dome rises to a single point "
          f"(got {len(tip)}) {'OK' if good else 'flat on top'}")

    # A rose has NO pavilion, and the test for that is that nothing
    # tapers below the girdle: the flat base spans the stone's full
    # radius, where any pavilion at all would pull it inward.
    r_max = max(math.hypot(x, y) for x, y, _ in V)
    r_base = max(math.hypot(v[0], v[1]) for v in base)
    good = abs(r_base - r_max) < 1e-9
    ok &= good
    print(f"gems.rose: the base spans the full girdle radius "
          f"({r_base:.4f} of {r_max:.4f}) -- no pavilion "
          f"{'OK' if good else 'grew one'}")

    A = antwerp_rose()
    Pa, Na, da, ta = build(A)
    na = [A.tiers[ta[j]].name for j in Pa.face_plane]
    good = _f.polytope_checks(Pa, Na, da)["closed"] \
        and na.count("R1") + na.count("R2") == 12
    ok &= good
    print(f"gems.rose: the Antwerp rose closes with 12 crown facets "
          f"{'OK' if good else 'wrong'}")

    # height is honoured
    for h in (0.3, 0.55, 0.8):
        Ph, _, _, _ = build(dutch_rose(height=h))
        Vh = Ph.V.tolist()
        r = max(math.hypot(x, y) for x, y, _ in Vh)
        got = (max(v[2] for v in Vh) - min(v[2] for v in Vh)) / (2 * r)
        if abs(got - h) > 0.02:
            ok = False
            print(f"gems.rose: height {h} came out {got:.3f}")
            break
    else:
        print("gems.rose: dome heights of 0.30, 0.55 and 0.80 are honoured OK")

    bad = 0
    for kw in (dict(n=2), dict(rows=0), dict(apex_angle=60.0),
               dict(height=0.0)):
        try:
            rose_cut(**kw)
        except ValueError:
            bad += 1
    good = bad == 4
    ok &= good
    print(f"gems.rose: impossible roses are refused ({bad}/4) "
          f"{'OK' if good else 'accepted nonsense'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.rose self-test failed")
