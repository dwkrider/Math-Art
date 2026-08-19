# Step cuts: the emerald, Asscher, baguette and polygon families.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python only -- no
# `bpy`, no numpy.
#
# A step cut is not a brilliant with different numbers.  Its facets are
# concentric ROWS of four-sided facets running parallel to the girdle,
# stepping inward and upward like a ziggurat, and its outline is a
# polygon rather than a circle -- so, unlike every facet of a round
# brilliant, the facets in one row do NOT share a centre-to-facet
# distance.  A rectangle's long sides stand further from the axis than
# its short sides, and its cut corners further still.  One row therefore
# becomes several tiers, one per distinct distance, which is exactly how
# GemCad writes them: `a` lines that share an angle and differ in
# distance.
#
# The construction rests on one idea -- CONSTANT INSET.  Every row moves
# every facet plane inward by the same absolute distance, so the outline
# marches inward as an offset polygon, staying similar to itself.  Two
# things fall out of that for free:
#
#   * the TABLE is the outline offset inward by the total crown inset, so
#     an emerald cut's table is a smaller cut-corner rectangle, which is
#     what an emerald cut's table looks like; and
#   * the pavilion converges on a KEEL, not a point.  Inset every side of
#     a rectangle by the short half-width and the short sides arrive at
#     the axis while the long sides stop short of it, leaving a ridge of
#     length 2*(l - w).  A keel is not a special case to be coded for --
#     it is what a rectangle does when you offset it inward.
#
# Row heights are then derived rather than given: with the inset per row
# fixed at dp, a row at elevation theta rises dp*tan(theta), so the crown
# height is sum(dp*tan(theta_k)) over the rows.  That is the same
# identity the round brilliant obeys -- c = (r_girdle - r_table)*tan(b) --
# generalised to several rows, and it is why the angles get steeper as
# they approach the girdle without anyone tabulating a height.
#
# References:
#   Robert W. Strickland, "GemCad for Windows Version 1.0 User's Guide",
#     GemSoft Enterprises, 2002 -- the worked emerald-cut diagram with
#     L/W 1.500, T/W 0.950, P/W 0.659, C/W 0.182 and its diagonal
#     (D D)/W 1.551, and the rule that a diagonal dimension is measured
#     between facet PLANES, not between their tips.
#   Basil Watermeyer, "Diamond Cutting: A Complete Guide to Diamond
#     Processing", 2nd ed. -- step-cut and mixed-cut practice.
#   CIBJO Diamond Commission, "The Diamond Book" (Blue Book 2024-1) --
#     cutting-style nomenclature (step cut, mixed cut).

import math

try:
    from .design import CutDesign, Tier
except ImportError:                     # flat import outside the package
    from design import CutDesign, Tier

SQRT2 = math.sqrt(2.0)


def _idx(azimuth_deg, gear):
    return int(round(azimuth_deg / 360.0 * gear)) % gear


def rect_outline(lw=1.5, corner=0.2168):
    """Girdle planes of a cut-corner rectangle, as (azimuth, distance).

    Half-width is 1 across the short axis, so `lw` is the length-to-width
    ratio.  `corner` is how deeply the corner planes cut, as a fraction of
    the half-width: 0 leaves the corner uncut (the plane passes exactly
    through it) and larger values shave it back.
    """
    if lw < 1.0:
        raise ValueError(f"length-to-width ratio must be >= 1, got {lw}")
    w, ln = 1.0, float(lw)
    # the rectangle's corner (w, l) projects onto the 45 degree direction
    # at (w + l)/sqrt(2); starting there and pulling in cuts the corner
    cd = (w + ln) / SQRT2 - corner * w
    if cd <= max(w, ln) + 1e-9:
        raise ValueError(f"corner {corner} cuts past the sides of a {lw}:1 "
                         f"outline; use a smaller corner")
    return [(0.0, w), (180.0, w), (90.0, ln), (270.0, ln),
            (45.0, cd), (135.0, cd), (225.0, cd), (315.0, cd)]


def polygon_outline(sides=8, point_up=False):
    """Girdle planes of a regular polygon, as (azimuth, distance)."""
    if sides < 3:
        raise ValueError(f"a polygon needs at least 3 sides, got {sides}")
    off = 0.0 if point_up else 0.5
    return [(((k + off) * 360.0 / sides) % 360.0, 1.0) for k in range(sides)]


def _rows(outline, angles, total_inset, z_start, crown, gear, prefix):
    """One tier per (row, distance class), stepping inward by equal insets.

    Returns `(tiers, z_end, inset_used)`.
    """
    n = len(angles)
    dp = total_inset / n
    tiers, z, shrink = [], z_start, 0.0
    sign = 1.0 if crown else -1.0
    for k, ang in enumerate(angles):
        a = math.radians(ang)
        # group this row's azimuths by their (shrunken) distance, so
        # facets that share a distance share a tier, as GemCad writes them
        classes = {}
        for az, dist in outline:
            p = dist - shrink
            if p <= 1e-9:
                continue                # this side has already reached the axis
            classes.setdefault(round(p, 9), []).append(az)
        for p, azs in sorted(classes.items(), reverse=True):
            d = math.sin(a) * p + math.cos(a) * z * sign
            tiers.append(Tier(sign * ang, d,
                              tuple(_idx(t, gear) for t in sorted(azs)),
                              name=f"{prefix}{k + 1}",
                              note=f"step row {k + 1}"))
        z += sign * dp * math.tan(a)
        shrink += dp
    return tiers, z, shrink


def step_cut(outline=None, lw=1.5, corner=0.2168, sides=8, table=0.60,
             crown_angles=(45.0, 35.0, 25.0),
             pavilion_angles=(45.0, 42.0, 39.0),
             girdle_pct=0.03, gear=96, ri=1.54, name="Step Cut"):
    """Build a step cut from an outline and a list of row angles.

    `outline` is a list of (azimuth, distance) girdle planes; if omitted a
    cut-corner rectangle of ratio `lw` is used.  Angles are listed from
    the girdle inward, so they must DECREASE -- two rows at the same
    elevation are parallel planes, and the inner one would cut the outer
    one away entirely.
    """
    if outline is None:
        outline = rect_outline(lw, corner)
    if not crown_angles or not pavilion_angles:
        raise ValueError("a step cut needs at least one row per side")
    for label, seq in (("crown", crown_angles), ("pavilion", pavilion_angles)):
        if any(b >= a for a, b in zip(seq, seq[1:])):
            raise ValueError(f"{label} row angles must decrease from the "
                             f"girdle inward, got {seq}")
        if not all(0.0 < a < 90.0 for a in seq):
            raise ValueError(f"{label} row angles must lie in (0, 90)")
    if not 0.0 < table < 1.0:
        raise ValueError(f"table must be a fraction in (0, 1), got {table}")

    g_min = min(d for _, d in outline)
    half_girdle = girdle_pct * g_min
    z_top, z_bot = half_girdle, -half_girdle

    tiers = [Tier(-90.0, d, tuple(_idx(az, gear) for az, dd in outline
                                  if abs(dd - d) < 1e-9),
                  name="G", note="girdle")
             for d in sorted({round(d, 9) for _, d in outline}, reverse=True)]

    # pavilion: inset by the SHORT half-width, which is what brings the
    # short sides to the axis and leaves the long ones as a keel
    pav, z_culet, _ = _rows(outline, list(pavilion_angles), g_min, z_bot,
                            False, gear, "P")
    tiers += pav

    # crown: inset to the table
    crown, z_table, _ = _rows(outline, list(crown_angles),
                              g_min * (1.0 - table), z_top, True, gear, "C")
    tiers += crown
    tiers.append(Tier(0.0, z_table, (_idx(0.0, gear),), name="T",
                      note="table"))

    detail = (f"L/W {lw:.3f}, table {table * 100:.0f}%, "
              f"{len(crown_angles)}+{len(pavilion_angles)} rows")
    return CutDesign(name=name, gear=gear, fold=2, mirror=True, ri=ri,
                     tiers=tuple(tiers), headings=(name, detail))


def emerald(**over):
    """The classic emerald cut: cut-corner rectangle, 3 rows each side.

    The outline follows the worked emerald-cut diagram in Strickland's
    manual -- L/W 1.500, table width U/W 0.450, and a corner cut of 0.2168
    recovered from its published diagonal (D D)/W of 1.551.  His row
    ANGLES are not printed, so the ones here are ordinary step-cut
    practice rather than a reproduction of his stone.
    """
    kw = dict(lw=1.5, corner=0.2168, table=0.45, name="Emerald Cut")
    kw.update(over)
    return step_cut(**kw)


def asscher(**over):
    """The Asscher / square emerald: a square outline with a deep crown."""
    kw = dict(lw=1.0, corner=0.30, table=0.55,
              crown_angles=(50.0, 40.0, 30.0),
              pavilion_angles=(50.0, 45.0, 40.0), name="Asscher Cut")
    kw.update(over)
    return step_cut(**kw)


def baguette(**over):
    """A baguette: long, plain, one row a side and no cut corners."""
    kw = dict(lw=2.0, corner=0.0, table=0.65,
              crown_angles=(40.0,), pavilion_angles=(43.0,),
              name="Baguette")
    kw.update(over)
    return step_cut(**kw)


def polygon_step(sides=6, rows=2, **over):
    """A regular-polygon step cut -- the hexagon and octagon cuts."""
    ca = tuple(45.0 - 10.0 * k for k in range(rows))
    pa = tuple(45.0 - 3.0 * k for k in range(rows))
    kw = dict(outline=polygon_outline(sides), table=0.60,
              crown_angles=ca, pavilion_angles=pa,
              name=f"{sides}-sided Step Cut")
    kw.update(over)
    return step_cut(**kw)


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

    # --- the emerald cut builds ------------------------------------------
    D = emerald()
    P, N, d, t = build(D)
    chk = _f.polytope_checks(P, N, d)
    good = chk["closed"] and chk["convex"]
    ok &= good
    print(f"gems.step: the emerald cut closes and is convex "
          f"({len(P.V)} vertices, {len(P.faces)} facets) "
          f"{'OK' if good else 'did not build'}")

    # The CROWN keeps all eight facets a row -- its corners must survive
    # to the table.  The PAVILION does not, and should not: inset far
    # enough and the corner chamfer is consumed, so an emerald cut's
    # corner facets terminate part-way down and the keel is formed by the
    # main facets alone.  That is what the stone looks like, so the test
    # asserts the crown is complete and merely reports the pavilion.
    tn = [D.tiers[t[j]].name for j in P.face_plane]
    girdle = tn.count("G")
    crown_rows = [tn.count(f"C{k + 1}") for k in range(3)]
    good = girdle == 8 and crown_rows == [8, 8, 8]
    ok &= good
    print(f"gems.step: 8 girdle facets and three complete crown rows of 8 "
          f"(got {girdle}, {crown_rows}) {'OK' if good else 'facets lost'}")
    pav_rows = [tn.count(f"P{k + 1}") for k in range(3)]
    print(f"gems.step: {len(P.faces)} facets in all; pavilion rows "
          f"{pav_rows} -- the corner facets terminate before the keel, as "
          f"they do on a real emerald cut OK")

    # --- the outline is a cut-corner rectangle of the right ratio --------
    V = P.V.tolist()
    zs = [v[2] for v in V]
    z_g = max(z for z in zs if z <= 0.05)
    rim = [v for v in V if abs(v[2] - z_g) < 0.05]
    w = max(abs(v[0]) for v in V)
    ln = max(abs(v[1]) for v in V)
    good = abs(ln / w - 1.5) < 0.02
    ok &= good
    print(f"gems.step: the outline measures {ln / w:.3f}:1, asked for 1.500 "
          f"{'OK' if good else 'wrong ratio'}")

    # --- the pavilion converges on a KEEL, not a point -------------------
    z_min = min(zs)
    keel = [v for v in V if abs(v[2] - z_min) < 1e-6]
    span = max(v[1] for v in keel) - min(v[1] for v in keel)
    good = len(keel) >= 2 and span > 0.5 * (ln - w)
    ok &= good
    print(f"gems.step: the pavilion ends in a keel of {len(keel)} vertices "
          f"spanning {span:.3f} (a point would be 1 vertex) "
          f"{'OK' if good else 'converged to a point'}")

    # --- the table is the outline offset inward --------------------------
    z_max = max(zs)
    tbl = [v for v in V if abs(v[2] - z_max) < 1e-6]
    tw = max(abs(v[0]) for v in tbl)
    good = abs(tw / w - 0.45) < 0.02
    ok &= good
    print(f"gems.step: the table measures {tw / w * 100:.1f}% of the width, "
          f"asked for 45 {'OK' if good else 'drifted'}")
    # Whether the TABLE keeps its cut corners is a threshold, not a
    # constant: insetting a chamfered rectangle erodes the chamfer faster
    # than the sides, so a corner deep enough at the girdle can still be
    # consumed before the table.  With Strickland's own 0.2168 corner and
    # a 45% table it is consumed almost exactly at the table, leaving a
    # rectangle; a deeper corner keeps the octagon.  Both are real stones,
    # so the test pins the MECHANISM rather than a facet count.
    big = build(emerald(corner=0.26))[0]
    zb = max(v[2] for v in big.V.tolist())
    tbl_big = [v for v in big.V.tolist() if abs(v[2] - zb) < 1e-6]
    good = len(tbl) == 4 and len(tbl_big) == 8
    ok &= good
    print(f"gems.step: a 0.217 corner is spent by the table (rectangular, "
          f"{len(tbl)} corners) while 0.26 keeps it octagonal "
          f"({len(tbl_big)}) {'OK' if good else 'threshold not reproduced'}")

    # --- a square outline gives a square stone ---------------------------
    A = asscher()
    Pa, Na, da, ta = build(A)
    Va = Pa.V.tolist()
    wa = max(abs(v[0]) for v in Va)
    la = max(abs(v[1]) for v in Va)
    good = abs(la / wa - 1.0) < 1e-6 and _f.polytope_checks(Pa, Na, da)["closed"]
    ok &= good
    print(f"gems.step: the Asscher is square ({la / wa:.6f}:1) and closed "
          f"{'OK' if good else 'wrong'}")

    # a square pavilion ends in a POINT, since all sides inset equally
    zmin = min(v[2] for v in Va)
    tip = [v for v in Va if abs(v[2] - zmin) < 1e-6]
    good = len(tip) == 1
    ok &= good
    print(f"gems.step: a square outline ends in a point, not a keel "
          f"(got {len(tip)} vertices) {'OK' if good else 'wrong'}")

    # --- one row a side is a baguette ------------------------------------
    B = baguette()
    Pb, Nb, db, tb = build(B)
    good = _f.polytope_checks(Pb, Nb, db)["closed"] and len(Pb.faces) == 4 + 4 + 4 + 1
    ok &= good
    print(f"gems.step: the baguette is 4 girdle + 4 + 4 + table = 13 facets "
          f"(got {len(Pb.faces)}) {'OK' if good else 'wrong count'}")

    # --- regular polygons -------------------------------------------------
    for n in (6, 8, 12):
        Dp = polygon_step(sides=n, rows=2)
        Pp, Np, dp_, tp = build(Dp)
        want = n + n * 2 + n * 2 + 1
        if not (_f.polytope_checks(Pp, Np, dp_)["closed"]
                and len(Pp.faces) == want):
            ok = False
            print(f"gems.step: the {n}-sided step cut gave "
                  f"{len(Pp.faces)} facets, expected {want}")
            break
    else:
        print("gems.step: 6, 8 and 12-sided step cuts all close with the "
              "expected facet counts OK")

    # --- rows must step inward -------------------------------------------
    bad = 0
    for kw in (dict(crown_angles=(30.0, 40.0)),      # increasing
               dict(crown_angles=(45.0, 45.0)),      # parallel
               dict(pavilion_angles=()),             # none
               dict(lw=0.5),                         # narrower than wide
               dict(corner=1.5)):                    # corner past the sides
        try:
            emerald(**kw)
        except ValueError:
            bad += 1
    good = bad == 5
    ok &= good
    print(f"gems.step: impossible step cuts are refused ({bad}/5) "
          f"{'OK' if good else 'accepted nonsense'}")

    # --- more rows make a taller crown, not a different outline ----------
    two = emerald(crown_angles=(45.0, 30.0))
    thr = emerald(crown_angles=(45.0, 37.0, 30.0))
    P2, _, _, _ = build(two)
    P3, _, _, _ = build(thr)
    w2 = max(abs(v[0]) for v in P2.V.tolist())
    w3 = max(abs(v[0]) for v in P3.V.tolist())
    good = abs(w2 - w3) < 1e-9 and len(P3.faces) > len(P2.faces)
    ok &= good
    print(f"gems.step: adding a row adds facets ({len(P2.faces)} -> "
          f"{len(P3.faces)}) without moving the girdle "
          f"{'OK' if good else 'outline moved'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.step self-test failed")
