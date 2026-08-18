# The parametric round brilliant.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python only -- no
# `bpy`, no numpy.
#
# A published design is a fixed table of angles; this builds the standard
# round brilliant from PROPORTIONS instead -- table %, crown angle,
# pavilion angle, girdle thickness, star length, lower-half length -- the
# parameters a grading report actually quotes.  Every tier that is not
# given directly is SOLVED from a meetpoint, in the order a cutter works,
# so no folklore constants appear anywhere below.
#
# Topology (verified against the GemCad Standard Round Brilliant, and
# matching CIBJO's normative arrangement):
#
#   table       1 octagon, its corners over the bezel azimuths
#   bezel       8 kites   at k*45 deg          "crown mains"
#   star        8 triangles at 22.5 + k*45     one edge on the table
#   upper half  16 triangles at 11.25 + k*22.5 one edge on the girdle
#   pavilion    8 kites   at k*45 deg          "pavilion mains"
#   lower half  16 triangles at 11.25 + k*22.5 one edge on the girdle
#   culet       0 (pointed) or 1 octagon
#                                              = 57 or 58 facets
#
# The crown's whole shape turns on ONE meetpoint: the star, two upper
# halves and two bezels all meet at a single vertex M over each star
# azimuth.  Star length places it; everything else follows.  The pavilion
# mirrors this exactly, with the lower-half length placing M'.
#
# Two proportion identities fall out of the construction, and both were
# checked against the reference solid to the last digit before being
# relied on here:
#
#   crown height    c = (r_girdle - r_table_corner) * tan(crown angle)
#   pavilion depth  p = r_girdle * tan(pavilion angle)
#
# They also reproduce Tolkowsky's own published figures: his 53% table at
# 34.5 degrees gives 16.2% crown height, and his 40.75 degree pavilion
# gives 43.1% depth -- the numbers printed in the 1919 table.  That is the
# strongest available check that the parameterisation means what the
# literature means by these words, and `_selftest` asserts it.
#
# TABLE PERCENTAGE is measured corner to corner, across the octagon's
# points (the bezel azimuths), which is what makes the identity above
# reproduce Tolkowsky.  Measuring flat to flat instead would put his
# crown height at 14.7%, not 16.2%.
#
# References:
#   Marcel Tolkowsky, "Diamond Design", E. & F. N. Spon, London, 1919,
#     Part III and the table of best proportions (pp. 97-104).
#   Robert H. Long & Norman W. Steele, "Introduction to Meetpoint
#     Faceting", Seattle Faceting Books, 1985 -- solving a tier through
#     a meetpoint rather than tabulating its angle.
#   CIBJO Diamond Commission, "The Diamond Book" (Blue Book 2024-1),
#     Annex B 7.2 -- the normative 57/58-facet arrangement.
#   Ping Zheng, Qinghua Xiao, Wei Xiong, Ying Yang, "Optimization design
#     model construction of round faceted gemstone facets based on
#     geometric transformation theory", J. Combin. Math. Combin. Comput.
#     127a (2025) 1269-1284 -- an independent closed form for the same
#     solid, used as a cross-check.

import math

try:
    from .design import CutDesign, Tier
except ImportError:                     # flat import outside the package
    from design import CutDesign, Tier

COS_1125 = math.cos(math.radians(11.25))
COS_2250 = math.cos(math.radians(22.5))


def _idx(azimuth_deg, gear):
    """Nearest index-gear position to an azimuth.

    The gear genuinely quantises azimuth -- a facet can only be cut at a
    tooth -- so rounding here is physical, not a numerical compromise.
    For the standard 96 gear every azimuth this module needs (11.25, 22.5
    and 45 degree multiples) lands exactly on a tooth.
    """
    return int(round(azimuth_deg / 360.0 * gear)) % gear


def round_brilliant(table=0.57, crown_angle=34.5, pavilion_angle=40.75,
                    girdle_pct=0.03, culet_pct=0.0, star_len=0.55,
                    lower_girdle_len=0.78, girdle_facets=16, gear=96,
                    ri=2.417, name="Round Brilliant"):
    """Build a standard round brilliant from its proportions.

    All ratios are fractions of the girdle DIAMETER except `star_len` and
    `lower_girdle_len`, which are fractions of the plan-view distance
    from the table edge (resp. the girdle) to the girdle (resp. the
    culet), as the trade measures them.

    `culet_pct` of 0 gives a pointed culet and 57 facets; any larger
    value adds a culet facet, giving 58.
    """
    if not 0.0 < table < 1.0:
        raise ValueError(f"table must be a fraction in (0, 1), got {table}")
    if not 0.0 < crown_angle < 90.0 or not 0.0 < pavilion_angle < 90.0:
        raise ValueError("crown and pavilion angles must lie in (0, 90)")
    if not 0.0 <= star_len < 1.0 or not 0.0 <= lower_girdle_len < 1.0:
        raise ValueError("star and lower-half lengths are fractions in [0, 1)")
    if girdle_facets < 8:
        raise ValueError(f"need at least 8 girdle facets, got {girdle_facets}")

    # A faceted girdle is a POLYGON, so it has two radii and they are not
    # interchangeable.  `r_g` is the facet distance (the polygon's
    # inradius), which is what a girdle tier's centre-to-facet distance
    # means; `r_v` is the circumradius, out at the polygon's corners.
    #
    # Which one a facet is built against depends on how it meets the
    # girdle.  The bezels and pavilion mains come down to a single POINT
    # on the girdle, and that point is a polygon CORNER -- so they are
    # built against r_v.  The upper and lower halves come down to an EDGE
    # lying in a girdle facet, so they are built against r_g.  Build a
    # bezel against r_g instead and its apex falls inside the polygon,
    # where the two girdle facets either side slice it into a pentagon.
    # Every proportion quoted as a percentage is against the DIAMETER,
    # 2*r_v, since that is what calipers measure.
    r_g = 1.0
    r_v = r_g / math.cos(math.pi / girdle_facets)
    half_girdle = girdle_pct * r_v              # girdle_pct * D / 2
    z_top, z_bot = half_girdle, -half_girdle
    b = math.radians(crown_angle)
    a = math.radians(pavilion_angle)

    tiers = []

    # --- girdle -------------------------------------------------------
    gird_az = [(k + 0.5) * 360.0 / girdle_facets for k in range(girdle_facets)]
    tiers.append(Tier(-90.0, r_g,
                      tuple(_idx(t, gear) for t in gird_az),
                      name="G", note=f"{girdle_facets}-sided girdle"))

    # --- pavilion mains: through the girdle and the culet point -------
    d_pav = math.sin(a) * r_v - math.cos(a) * z_bot
    z_culet = z_bot - r_v * math.tan(a)
    tiers.append(Tier(-pavilion_angle, d_pav,
                      tuple(_idx(k * 45.0, gear) for k in range(8)),
                      name="P", note="pavilion mains, meet at the culet"))

    # --- lower halves: solved through the pavilion meetpoint M' -------
    rho_mp = r_v * (1.0 - lower_girdle_len)
    z_mp = (math.sin(a) * rho_mp * COS_2250 - d_pav) / math.cos(a)
    den = r_g - rho_mp * COS_1125
    if abs(den) < 1e-12:
        raise ValueError("lower-half length leaves the meetpoint on the girdle")
    # Both the rise (z_bot - z_M', the meetpoint is below the girdle) and
    # the run are positive here, so the elevation is the FIRST-quadrant
    # root; feeding atan2 the doubly-negated pair instead lands on its
    # reflection 180 degrees away and inverts the facet.
    th = math.atan2(z_bot - z_mp, den)
    lower_angle = math.degrees(th)
    d_low = math.sin(th) * r_g - math.cos(th) * z_bot
    tiers.append(Tier(-lower_angle, d_low,
                      tuple(_idx(11.25 + k * 22.5, gear) for k in range(16)),
                      name="L", note="lower halves, meet the pavilion mains"))

    # --- culet --------------------------------------------------------
    if culet_pct > 0.0:
        z_cu = (culet_pct * r_v * math.sin(a) - d_pav) / math.cos(a)
        # angle 0 with a NEGATIVE distance is GemCad's culet convention
        tiers.append(Tier(0.0, z_cu, (_idx(0.0, gear),), name="C",
                          note="culet"))

    # --- bezels: through the girdle at the crown angle -----------------
    d_bez = math.sin(b) * r_v + math.cos(b) * z_top
    tiers.append(Tier(crown_angle, d_bez,
                      tuple(_idx(k * 45.0, gear) for k in range(8)),
                      name="B", note="bezels / crown mains"))

    # --- table: corner radius is the table percentage ------------------
    a_corner = table * r_v
    crown_h = (r_v - a_corner) * math.tan(b)
    z_table = z_top + crown_h
    tiers.append(Tier(0.0, z_table, (_idx(0.0, gear),), name="T",
                      note="table"))

    # --- crown meetpoint M, then the stars and upper halves ------------
    rho_edge = a_corner * COS_2250              # table edge, at a star azimuth
    rho_m = rho_edge + star_len * (r_v - rho_edge)
    z_m = (d_bez - math.sin(b) * rho_m * COS_2250) / math.cos(b)

    # star: through M and the table corner beside it
    corner = (a_corner * math.cos(0.0), a_corner * math.sin(0.0), z_table)
    m_pt = (rho_m * math.cos(math.radians(22.5)),
            rho_m * math.sin(math.radians(22.5)), z_m)
    star_angle, d_star = _plane_through(22.5, corner, m_pt)
    tiers.append(Tier(star_angle, d_star,
                      tuple(_idx(22.5 + k * 45.0, gear) for k in range(8)),
                      name="S", note="stars, meet the table and the bezels"))

    # upper halves: sit on the girdle and reach M
    den = r_g - rho_m * COS_1125
    if abs(den) < 1e-12:
        raise ValueError("star length leaves the meetpoint on the girdle")
    # as for the lower halves: the first-quadrant root is the facet
    th = math.atan2(z_m - z_top, den)
    upper_angle = math.degrees(th)
    d_up = math.sin(th) * r_g + math.cos(th) * z_top
    tiers.append(Tier(upper_angle, d_up,
                      tuple(_idx(11.25 + k * 22.5, gear) for k in range(16)),
                      name="U", note="upper halves, meet the bezels at M"))

    # The first heading IS the design name in .ASC -- a reader has nowhere
    # else to take it from -- so it must be the bare name, with the
    # proportions on a second line, or the name would not survive a round
    # trip through the file format.
    detail = (f"table {table * 100:.0f}%, crown {crown_angle:.1f} deg, "
              f"pavilion {pavilion_angle:.2f} deg")
    return CutDesign(name=name, gear=gear, fold=8, mirror=True, ri=ri,
                     tiers=tuple(tiers), headings=(name, detail))


def _plane_through(azimuth_deg, p1, p2):
    """(GemCad angle, distance) of the plane at `azimuth` through 2 points."""
    phi = math.radians(azimuth_deg)
    vx, vy, vz = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    h = vx * math.cos(phi) + vy * math.sin(phi)
    if abs(h) < 1e-12 and abs(vz) < 1e-12:
        raise ValueError("degenerate points; the elevation is undetermined")
    t = math.atan2(-vz, h)
    if math.sin(t) < 0.0:                       # face outward at `azimuth`
        t += math.pi
    n = (math.sin(t) * math.cos(phi), math.sin(t) * math.sin(phi),
         math.cos(t))
    d = n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2]
    deg = math.degrees(t % (2.0 * math.pi))
    return (deg if deg <= 90.0 + 1e-9 else deg - 180.0), d


def tolkowsky(**over):
    """Tolkowsky's 1919 proportions: 53% table, 34.5 crown, 40.75 pavilion."""
    kw = dict(table=0.53, crown_angle=34.5, pavilion_angle=40.75,
              girdle_pct=0.0, star_len=0.55, lower_girdle_len=0.75,
              name="Tolkowsky Brilliant")
    kw.update(over)
    return round_brilliant(**kw)


def _selftest():
    try:
        from . import facets as _facets
        from .design import planes
    except ImportError:
        import facets as _facets
        from design import planes

    ok = True

    # --- Tolkowsky's own published table ---------------------------------
    # His 1919 figures, for a stone of diameter 100: table 53.0, thickness
    # above the girdle 16.2, below 43.1.  A parameterisation that does not
    # reproduce these does not mean what the literature means.
    D = tolkowsky()
    crown_h = (1.0 - 0.53) * math.tan(math.radians(34.5))
    pav_d = math.tan(math.radians(40.75))
    good = abs(crown_h / 2.0 * 100 - 16.2) < 0.1
    ok &= good
    print(f"gems.brilliant: Tolkowsky's crown height comes out "
          f"{crown_h / 2 * 100:.2f}% of the diameter, he printed 16.2 "
          f"{'OK' if good else 'mismatch'}")
    good = abs(pav_d / 2.0 * 100 - 43.1) < 0.1
    ok &= good
    print(f"gems.brilliant: Tolkowsky's pavilion depth comes out "
          f"{pav_d / 2 * 100:.2f}% of the diameter, he printed 43.1 "
          f"{'OK' if good else 'mismatch'}")

    # --- it builds, and to the right topology ----------------------------
    N, d, tid = planes(D)
    P = _facets.intersect_halfspaces(N, d)
    chk = _facets.polytope_checks(P, N, d)
    good = chk["closed"] and chk["convex"]
    ok &= good
    print(f"gems.brilliant: the Tolkowsky brilliant closes and is convex "
          f"({len(P.V)} vertices, {len(P.faces)} facets) "
          f"{'OK' if good else 'did not build'}")

    names = [D.tiers[tid[j]].name for j in P.face_plane]
    n_g = names.count("G")
    crown = sum(names.count(x) for x in ("T", "B", "S", "U"))
    pav = sum(names.count(x) for x in ("P", "L", "C"))
    good = (crown, pav) == (33, 24)
    ok &= good
    print(f"gems.brilliant: 33 crown + 24 pavilion facets "
          f"(got {crown} + {pav}, plus {n_g} girdle) "
          f"{'OK' if good else 'wrong arrangement'}")

    # CIBJO's count, and Zheng's decomposition of it
    from collections import Counter
    shape = Counter(len(f) for f, j in zip(P.faces, P.face_plane)
                    if D.tiers[tid[j]].name != "G")
    good = crown + pav == 57 and shape[3] == 40 and shape[4] == 16 \
        and shape[8] == 1
    ok &= good
    print(f"gems.brilliant: 57 facets = 40 triangles + 16 quads + 1 octagon "
          f"(got {dict(shape)}) {'OK' if good else 'wrong decomposition'}")

    # --- a culet turns 57 into 58 ----------------------------------------
    Dc = tolkowsky(culet_pct=0.02)
    Nc, dc, tc = planes(Dc)
    Pc = _facets.intersect_halfspaces(Nc, dc)
    nc = [Dc.tiers[tc[j]].name for j in Pc.face_plane]
    good = sum(nc.count(x) for x in ("T", "B", "S", "U", "P", "L", "C")) == 58 \
        and nc.count("C") == 1
    ok &= good
    print(f"gems.brilliant: a culet facet makes it 58 "
          f"{'OK' if good else 'culet missing'}")

    # --- the proportions are recovered from the built solid --------------
    r = max(math.hypot(x, y) for x, y, _ in P.V.tolist())
    zt = max(z for _, _, z in P.V.tolist())
    zc = min(z for _, _, z in P.V.tolist())
    corner = max(math.hypot(x, y) for x, y, z in P.V.tolist()
                 if abs(z - zt) < 1e-9)
    good = abs(corner / r - 0.53) < 1e-6
    ok &= good
    print(f"gems.brilliant: the built table measures {corner / r * 100:.3f}% "
          f"corner to corner, asked for 53 {'OK' if good else 'drifted'}")
    good = abs((zt - zc) / (2 * r) - (crown_h + pav_d) / 2.0) < 1e-6
    ok &= good
    print(f"gems.brilliant: total depth {(zt - zc) / (2 * r) * 100:.2f}% "
          f"= crown + pavilion {'OK' if good else 'inconsistent'}")

    # --- the parameters actually steer the shape -------------------------
    deep = round_brilliant(pavilion_angle=43.0, girdle_pct=0.0)
    Nd, dd, _ = planes(deep)
    Pd = _facets.intersect_halfspaces(Nd, dd)
    good = min(z for _, _, z in Pd.V.tolist()) < zc - 1e-3
    ok &= good
    print(f"gems.brilliant: a steeper pavilion makes a deeper stone "
          f"{'OK' if good else 'no effect'}")

    small = round_brilliant(table=0.45, girdle_pct=0.0)
    Ns, ds, ts = planes(small)
    Ps = _facets.intersect_halfspaces(Ns, ds)
    rs = max(math.hypot(x, y) for x, y, _ in Ps.V.tolist())
    zts = max(z for _, _, z in Ps.V.tolist())
    cs = max(math.hypot(x, y) for x, y, z in Ps.V.tolist()
             if abs(z - zts) < 1e-9)
    good = abs(cs / rs - 0.45) < 1e-6
    ok &= good
    print(f"gems.brilliant: a 45% table builds a 45% table "
          f"(got {cs / rs * 100:.2f}) {'OK' if good else 'ignored'}")

    # star length must move the crown meetpoint, and nothing else
    for sl in (0.40, 0.55, 0.70):
        Dx = tolkowsky(star_len=sl)
        Nx, dx, tx = planes(Dx)
        Px = _facets.intersect_halfspaces(Nx, dx)
        nx = [Dx.tiers[tx[j]].name for j in Px.face_plane]
        if sum(nx.count(x) for x in ("T", "B", "S", "U", "P", "L")) != 57:
            ok = False
            print(f"gems.brilliant: star length {sl} broke the arrangement")
            break
    else:
        print("gems.brilliant: the arrangement survives star lengths "
              "0.40, 0.55 and 0.70 OK")

    # --- a girdle of a different facet count still closes ----------------
    for gf in (16, 32, 64):
        Dg = tolkowsky(girdle_pct=0.03, girdle_facets=gf)
        Ng, dg, _ = planes(Dg)
        Pg = _facets.intersect_halfspaces(Ng, dg)
        if not _facets.polytope_checks(Pg, Ng, dg)["closed"]:
            ok = False
            print(f"gems.brilliant: a {gf}-facet girdle did not close")
            break
    else:
        print("gems.brilliant: girdles of 16, 32 and 64 facets all close OK")

    # --- refusals ---------------------------------------------------------
    bad = 0
    for kw in (dict(table=1.5), dict(crown_angle=0.0),
               dict(star_len=1.0), dict(girdle_facets=3)):
        try:
            round_brilliant(**kw)
        except ValueError:
            bad += 1
    good = bad == 4
    ok &= good
    print(f"gems.brilliant: impossible proportions are refused ({bad}/4) "
          f"{'OK' if good else 'accepted nonsense'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.brilliant self-test failed")
