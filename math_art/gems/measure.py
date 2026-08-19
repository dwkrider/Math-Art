# Metrology: reading a cut stone's proportions back off the solid.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python and numpy
# only -- no `bpy`.
#
# Everything here measures the BUILT POLYTOPE, never the design that
# produced it.  That is the whole point: a generator that reported the
# numbers it was handed would agree with itself no matter how wrong the
# geometry was.  Measuring the solid instead makes the round trip
# (proportions -> planes -> solid -> proportions) a real test, and it is
# what lets an imported third-party design be graded at all, since it
# arrives as angles and distances with no proportions attached.
#
# Conventions follow the trade, and they are not all obvious:
#
#   * every percentage is against the girdle DIAMETER;
#   * TABLE % is corner to corner, across the octagon's points;
#   * crown height and pavilion depth EXCLUDE the girdle (Strickland is
#     explicit that the girdle belongs to neither), so total depth is
#     crown + girdle + pavilion;
#   * the CROWN and PAVILION angles quoted on a report are those of the
#     MAINS -- the bezels and the pavilion mains -- not of the steepest
#     tier, which in a standard brilliant is an upper half.
#
# Identifying the mains from geometry alone takes a rule, and the one
# used here is: among the tiers on that side carrying exactly `fold`
# facets (8 in a round brilliant), take the steepest.  On the GemCad
# reference that picks the 28-degree bezels over the 16-degree stars and
# rejects the 34-degree upper halves, which is correct; picking "the
# steepest tier" outright would wrongly return the upper halves.
#
# References:
#   International Diamond Council, "IDC Rules for Grading Polished
#     Diamonds", 6th ed., 2013, section 4.2.3 -- the proportion bands in
#     `idc_grade`, and the fish-eye / culet-in-bezel definitions.
#   Robert W. Strickland, "GemCad for Windows Version 1.0 User's Guide",
#     GemSoft Enterprises, 2002 -- the dimension conventions, and the
#     carat formula carats = Vol/W^3 * W^3 * 0.005 * s.g.
#   CIBJO Diamond Commission, "The Diamond Book" (Blue Book 2024-1).

import math
from typing import NamedTuple

import numpy as np

_FLAT = 1e-6


class Proportions(NamedTuple):
    """Everything a grading report quotes, measured off the solid."""

    diameter: float
    table_pct: float
    crown_angle: float
    pavilion_angle: float
    crown_height_pct: float
    pavilion_depth_pct: float
    girdle_pct: float
    culet_pct: float
    total_depth_pct: float
    facet_count: int
    volume: float


def _tier_angle(N, j):
    """Elevation of a facet plane above the girdle plane, in degrees."""
    n = N[j]
    return math.degrees(math.atan2(math.hypot(n[0], n[1]), abs(n[2])))


def _main_tier(N, tier_id, faces_by_tier, fold, crown):
    """The tier a report means by 'the crown angle' / 'the pavilion angle'."""
    cand = []
    for t, js in faces_by_tier.items():
        n = N[js[0]]
        if abs(n[2]) < _FLAT:                   # girdle
            continue
        if (n[2] > 0) != crown:
            continue
        ang = _tier_angle(N, js[0])
        if ang < _FLAT:                         # table or culet
            continue
        cand.append((len(js), ang, t))
    if not cand:
        return None
    exact = [c for c in cand if c[0] == fold]
    return max(exact or cand, key=lambda c: c[1])[2]


def proportions(P, N, d, tier_id, fold=8):
    """Measure a built stone.  `P` is a Polytope, `N`/`d` its planes."""
    V = P.V
    r = np.hypot(V[:, 0], V[:, 1])
    z = V[:, 2]
    r_v = float(r.max())
    diameter = 2.0 * r_v
    if diameter <= 0:
        raise ValueError("degenerate stone: zero diameter")

    faces_by_tier = {}
    for f, j in zip(P.faces, P.face_plane):
        faces_by_tier.setdefault(int(tier_id[j]), []).append(j)

    # the girdle band: vertices out at the rim
    rim = r >= r_v - 1e-9 * max(1.0, r_v)
    z_top, z_bot = float(z[rim].max()), float(z[rim].min())
    girdle_pct = (z_top - z_bot) / diameter

    # table: the horizontal facet at the top, if there is one
    table_pct = 0.0
    z_table = float(z.max())
    for t, js in faces_by_tier.items():
        if N[js[0]][2] > 1.0 - _FLAT:
            face = [f for f, j in zip(P.faces, P.face_plane) if j == js[0]][0]
            table_pct = 2.0 * float(max(r[list(face)])) / diameter
            z_table = float(z[list(face)].max())
    # culet: a horizontal facet at the bottom
    culet_pct = 0.0
    z_culet = float(z.min())
    for t, js in faces_by_tier.items():
        if N[js[0]][2] < -1.0 + _FLAT:
            face = [f for f, j in zip(P.faces, P.face_plane) if j == js[0]][0]
            culet_pct = 2.0 * float(max(r[list(face)])) / diameter

    ct = _main_tier(N, tier_id, faces_by_tier, fold, crown=True)
    pt = _main_tier(N, tier_id, faces_by_tier, fold, crown=False)
    crown_angle = _tier_angle(N, faces_by_tier[ct][0]) if ct is not None else 0.0
    pav_angle = _tier_angle(N, faces_by_tier[pt][0]) if pt is not None else 0.0

    try:
        from .facets import volume as _vol
    except ImportError:
        from facets import volume as _vol

    return Proportions(
        diameter=diameter,
        table_pct=table_pct,
        crown_angle=crown_angle,
        pavilion_angle=pav_angle,
        crown_height_pct=(z_table - z_top) / diameter,
        pavilion_depth_pct=(z_bot - z_culet) / diameter,
        girdle_pct=girdle_pct,
        culet_pct=culet_pct,
        total_depth_pct=(z_table - z_culet) / diameter,
        facet_count=len(P.faces),
        volume=_vol(P))


# IDC Rules 6th ed. (2013) section 4.2.3, verbatim.  Each row is the
# band edges from Fair (shallow) through Excellent to Fair (deep); a
# value is graded by the first band it falls into.
_IDC = {
    "crown_angle": [(25.9, "Fair"), (27.9, "Good"), (31.9, "Very Good"),
                    (36.0, "Excellent"), (37.7, "Very Good"),
                    (40.0, "Good"), (1e9, "Fair")],
    "pavilion_angle": [(38.4, "Fair"), (39.5, "Good"), (40.5, "Very Good"),
                       (41.8, "Excellent"), (42.1, "Very Good"),
                       (43.1, "Good"), (1e9, "Fair")],
    "table_pct": [(0.49, "Fair"), (0.51, "Good"), (0.53, "Very Good"),
                  (0.62, "Excellent"), (0.66, "Very Good"),
                  (0.70, "Good"), (1e9, "Fair")],
    "crown_height_pct": [(0.085, "Fair"), (0.105, "Good"), (0.115, "Very Good"),
                         (0.160, "Excellent"), (0.180, "Very Good"),
                         (0.195, "Good"), (1e9, "Fair")],
    "pavilion_depth_pct": [(0.395, "Fair"), (0.410, "Good"),
                           (0.425, "Very Good"), (0.445, "Excellent"),
                           (0.450, "Very Good"), (0.465, "Good"),
                           (1e9, "Fair")],
    "girdle_pct": [(0.005, "Fair"), (0.015, "Good"), (0.020, "Very Good"),
                   (0.040, "Excellent"), (0.050, "Very Good"),
                   (0.075, "Good"), (1e9, "Fair")],
    "culet_pct": [(0.009, "Excellent"), (0.019, "Very Good"),
                  (0.039, "Good"), (1e9, "Fair")],
}

_ORDER = ("Fair", "Good", "Very Good", "Excellent")


def idc_grade(props):
    """Per-parameter IDC proportion grades, plus the overall (worst) one.

    The IDC rule is explicit that when measurements fall in different
    categories "the lowest proportion grade is considered to be the
    overall reading", so the overall grade is the worst, not an average.
    """
    out = {}
    for key, bands in _IDC.items():
        v = getattr(props, key)
        for edge, grade in bands:
            if v <= edge:
                out[key] = grade
                break
        else:
            out[key] = "Fair"
    out["overall"] = min(out.values(), key=_ORDER.index)
    return out


def carat(volume_mm3, sg):
    """Carat weight from volume in mm^3 and specific gravity (1 ct = 0.2 g)."""
    return volume_mm3 * 1e-3 * sg * 5.0


def carat_from_diameter(props, diameter_mm, sg):
    """Carat weight of the measured stone scaled to a real diameter."""
    s = diameter_mm / props.diameter
    return carat(props.volume * s ** 3, sg)


def warnings(props, ri=2.417):
    """The proportion pathologies the IDC rules name, as plain messages."""
    out = []
    crit = math.degrees(math.asin(1.0 / ri)) if ri > 1.0 else 24.4
    if props.pavilion_angle and props.pavilion_angle < 39.0 \
            and props.table_pct > 0.60:
        out.append("fish-eye: the girdle reflects in the table (shallow "
                   "pavilion with a large table)")
    if props.pavilion_angle and props.pavilion_angle < crit + 13.0:
        out.append(f"windowing: light passes straight through; the pavilion "
                   f"is shallow for a refractive index of {ri:.3f}")
    if props.pavilion_angle > 43.5:
        out.append("extinction: the pavilion is steep enough to return "
                   "little light (a 'nail head')")
    if props.total_depth_pct > 0.65 and props.crown_angle > 36.0:
        out.append("the culet may be visible through the bezels")
    return out


def _selftest():
    try:
        from . import brilliant as _b
        from . import facets as _f
        from .design import SRB_GEMCAD, planes
    except ImportError:
        import brilliant as _b
        import facets as _f
        from design import SRB_GEMCAD, planes

    ok = True

    def build(D):
        N, d, t = planes(D)
        return _f.intersect_halfspaces(N, d), N, d, t

    # --- the round trip that makes measurement non-circular ---------------
    want = dict(table=0.57, crown_angle=34.5, pavilion_angle=40.75,
                girdle_pct=0.03, star_len=0.55, lower_girdle_len=0.78)
    D = _b.round_brilliant(**want)
    P, N, d, t = build(D)
    pr = proportions(P, N, d, t, fold=D.fold)

    good = abs(pr.table_pct - 0.57) < 5e-3
    ok &= good
    print(f"gems.measure: a 57% table measures back at "
          f"{pr.table_pct * 100:.2f}% {'OK' if good else 'drifted'}")
    good = abs(pr.crown_angle - 34.5) < 0.05
    ok &= good
    print(f"gems.measure: a 34.5 deg crown measures back at "
          f"{pr.crown_angle:.3f} {'OK' if good else 'drifted'}")
    good = abs(pr.pavilion_angle - 40.75) < 0.05
    ok &= good
    print(f"gems.measure: a 40.75 deg pavilion measures back at "
          f"{pr.pavilion_angle:.3f} {'OK' if good else 'drifted'}")
    good = abs(pr.girdle_pct - 0.03) < 2e-3
    ok &= good
    print(f"gems.measure: a 3% girdle measures back at "
          f"{pr.girdle_pct * 100:.2f}% {'OK' if good else 'drifted'}")
    good = abs(pr.total_depth_pct
               - (pr.crown_height_pct + pr.girdle_pct
                  + pr.pavilion_depth_pct)) < 1e-9
    ok &= good
    print(f"gems.measure: total depth = crown + girdle + pavilion "
          f"({pr.total_depth_pct * 100:.2f}%) "
          f"{'OK' if good else 'inconsistent'}")

    # --- the mains, not the steepest tier ---------------------------------
    Ps, Ns, ds, ts = build(SRB_GEMCAD)
    ps = proportions(Ps, Ns, ds, ts, fold=SRB_GEMCAD.fold)
    good = abs(ps.crown_angle - 28.0) < 0.01 \
        and abs(ps.pavilion_angle - 41.5) < 0.01
    ok &= good
    print(f"gems.measure: on the GemCad reference the mains are found -- "
          f"crown {ps.crown_angle:.2f} (28, not the 34 upper halves), "
          f"pavilion {ps.pavilion_angle:.2f} (41.5) "
          f"{'OK' if good else 'picked the wrong tier'}")

    # --- IDC banding ------------------------------------------------------
    g = idc_grade(pr)
    good = g["overall"] == "Excellent"
    ok &= good
    print(f"grades: a well-cut brilliant is Excellent overall "
          f"(got {g['overall']}) {'OK' if good else 'mis-graded'}")

    shallow = _b.round_brilliant(pavilion_angle=38.0, table=0.68,
                                 girdle_pct=0.03)
    Pw, Nw, dw, tw = build(shallow)
    pw = proportions(Pw, Nw, dw, tw, fold=8)
    gw = idc_grade(pw)
    good = gw["pavilion_angle"] == "Fair" and gw["overall"] == "Fair"
    ok &= good
    print(f"grades: a 38 deg pavilion with a 68% table grades Fair "
          f"(pavilion {gw['pavilion_angle']}, table {gw['table_pct']}) "
          f"{'OK' if good else 'mis-graded'}")

    good = min(idc_grade(pr).values(), key=_ORDER.index) == g["overall"]
    ok &= good
    print(f"grades: the overall grade is the worst parameter, as the IDC "
          f"rule states {'OK' if good else 'averaged instead'}")

    # --- the named failure modes fire, and only when they should ---------
    w_ok = warnings(pr)
    w_bad = warnings(pw)
    good = not w_ok and any("fish-eye" in m for m in w_bad)
    ok &= good
    print(f"gems.measure: the ideal stone raises no warning, the shallow one "
          f"raises fish-eye ({len(w_bad)} total) "
          f"{'OK' if good else 'wrong warnings'}")

    steep = _b.round_brilliant(pavilion_angle=44.5, girdle_pct=0.03)
    Pe, Ne, de, te = build(steep)
    we = warnings(proportions(Pe, Ne, de, te, fold=8))
    good = any("extinction" in m for m in we)
    ok &= good
    print(f"gems.measure: a 44.5 deg pavilion is flagged for extinction "
          f"{'OK' if good else 'not flagged'}")

    # --- carat weight -----------------------------------------------------
    # The trade rule of thumb: a 6.5 mm round brilliant in diamond
    # (s.g. 3.52) weighs about a carat.
    ct = carat_from_diameter(pr, 6.5, 3.52)
    good = 0.85 <= ct <= 1.15
    ok &= good
    print(f"gems.measure: a 6.5 mm diamond brilliant weighs {ct:.3f} ct "
          f"(trade rule of thumb: ~1.00) {'OK' if good else 'off'}")
    good = abs(carat_from_diameter(pr, 13.0, 3.52) - 8.0 * ct) < 1e-9
    ok &= good
    print(f"gems.measure: weight scales with the cube of the diameter "
          f"{'OK' if good else 'does not'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.measure self-test failed")
