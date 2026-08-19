# Mixed cuts: princess, radiant and barion.
#
# Part of the Math Art gem engine (`math_art/gems/`).  Python only -- no
# `bpy`, no numpy.
#
# A MIXED cut carries the two classic facet grammars on one stone: a
# STEP-CUT crown -- concentric rows of four-sided facets parallel to the
# girdle, as on an emerald cut -- over a BRILLIANT-CUT pavilion, whose
# facets radiate from the culet.  It is how a square or rectangular
# outline gets a round brilliant's light return: the crown keeps the
# architectural step look, the pavilion does the optics.
#
# The three cuts here, and where their geometry comes from:
#
#   BARION -- invented by the Johannesburg cutter Basil Watermeyer (1971),
#     who named it for himself and his wife Marion.  A cut-corner square
#     with a step crown and a brilliant pavilion.  Its defining feature is
#     a row of CRESCENT ("half moon") facets hung directly below the
#     girdle: the girdle of a square stone is a straight line where a
#     round brilliant's would be an arc, and the crescents are what absorb
#     that mismatch, letting a radial fan of pavilion facets meet a square
#     outline.  Each crescent's top edge lies on the girdle; its lower
#     boundary is scalloped out by the fan behind it, which is what gives
#     the half-moon look.
#
#   RADIANT -- Henry Grossbard (Radiant Cut Diamond Corporation, 1977):
#     the same idea carried onto a cut-corner RECTANGLE with a
#     length-to-width ratio.  Built here as the barion construction with
#     `lw > 1`; a rectangle's long and short sides stand at different
#     distances from the axis, so every "row" splits into distance
#     classes exactly as a step cut's rows do.  (Grossbard's original
#     also brilliant-faceted the crown; the step crown here follows the
#     mixed-cut convention -- see the honesty notes below.)
#
#   PRINCESS -- the square chevron-pavilion cut developed by Betzalel
#     Ambar and Israel Itzkowitz (Ambar Diamonds, Los Angeles, c. 1980,
#     trademarked "Quadrillion"; grading nomenclature files it as a
#     square modified brilliant, e.g. IDC Rules 5.2.1).  A sharp-cornered
#     square whose pavilion carries CHEVRON facets: rows of V-shaped
#     facet pairs whose spines run down the corner diagonals from girdle
#     to culet, each successive V nested inside the last.  More chevron
#     rows give finer scintillation, fewer give bolder flashes; two to
#     four rows is the normal range, and it is a parameter here.
#
# Construction notes.  The crown is the step-cut machinery: rows stepping
# inward by CONSTANT INSET, one tier per distance class, ending in a
# table.  The pavilions are meetpoint constructions in the brilliant's
# manner:
#
#   * A polygon girdle has two radii.  A facet anchored on a girdle
#     CORNER is built against the circumradius; one anchored on an EDGE
#     against the facet (inradius) distance.  The princess's chevrons
#     start on the square's corners (circumradius w*sqrt(2)); the
#     barion's crescents hang from the side edges (distance w).
#   * The princess chevron pair at diagonal azimuth +-delta exploits a
#     symmetry: a plane at azimuth (45 + delta) and its mirror at
#     (45 - delta) agree everywhere on the diagonal plane, so solving one
#     of the pair through two spine points on the diagonal places both,
#     and the pair meets along the spine -- the chevron's V.
#   * The barion's lower halves are cut STEEPER than the pavilion mains
#     and anchored on the girdle vertices, so they die out on the mains
#     partway down and the mains alone converge on the culet.  That is
#     the round brilliant's own arrangement -- in the GemCad Standard
#     Round Brilliant the halves are at 42.5 degrees to the mains' 41.5 --
#     and getting it backwards truncates the culet.
#
# References:
#   Basil Watermeyer, "Diamond Cutting: A Complete Guide to Diamond
#     Processing", 2nd ed. -- barion and mixed-cut practice, by the
#     barion's inventor.
#   CIBJO Diamond Commission, "The Diamond Book" (Blue Book) --
#     cutting-style nomenclature: brilliant cut, step cut, mixed cut.
#   International Diamond Council, "Rules for Grading Polished Diamonds"
#     (2013), 5.2.1 -- the princess as a square modified brilliant.
#   Robert H. Long & Norman W. Steele, "Introduction to Meetpoint
#     Faceting", Seattle Faceting Books, 1985 -- solving tiers through
#     meetpoints rather than tabulating angles.
#   Robert W. Strickland, "GemCad for Windows Version 1.0 User's Guide",
#     GemSoft Enterprises, 2002 -- the tier/index data model.

import math

try:
    from .design import CutDesign, Tier
    from .step import rect_outline
except ImportError:                     # flat import outside the package
    from design import CutDesign, Tier
    from step import rect_outline

SQRT2 = math.sqrt(2.0)


def _idx(azimuth_deg, gear):
    return int(round(azimuth_deg / 360.0 * gear)) % gear


def _girdle_tiers(outline, gear):
    """One girdle tier per distance class, as `step.step_cut` writes them."""
    return [Tier(-90.0, d, tuple(_idx(az, gear) for az, dd in outline
                                 if abs(dd - d) < 1e-9),
                 name="G", note="girdle")
            for d in sorted({round(d, 9) for _, d in outline}, reverse=True)]


def _step_crown(outline, angles, total_inset, z_start, gear, prefix="C"):
    """Step-cut crown rows by constant inset; returns (tiers, z_table).

    The same construction as `step._rows` on its crown side: every row
    moves every plane inward by the same absolute distance, so the
    outline marches inward staying similar to itself, and facets that
    share a (shrunken) distance share a tier.
    """
    if not angles:
        raise ValueError("a mixed cut needs at least one crown row")
    if any(b >= a for a, b in zip(angles, angles[1:])):
        raise ValueError(f"crown row angles must decrease from the girdle "
                         f"inward, got {tuple(angles)}")
    if not all(0.0 < a < 90.0 for a in angles):
        raise ValueError("crown row angles must lie in (0, 90)")
    n = len(angles)
    dp = total_inset / n
    tiers, z, shrink = [], z_start, 0.0
    for k, ang in enumerate(angles):
        a = math.radians(ang)
        classes = {}
        for az, dist in outline:
            p = dist - shrink
            if p <= 1e-9:
                continue
            classes.setdefault(round(p, 9), []).append(az)
        for p, azs in sorted(classes.items(), reverse=True):
            tiers.append(Tier(ang, math.sin(a) * p + math.cos(a) * z,
                              tuple(_idx(t, gear) for t in sorted(azs)),
                              name=f"{prefix}{k + 1}",
                              note=f"step row {k + 1}"))
        z += dp * math.tan(a)
        shrink += dp
    return tiers, z


def _plane_at(azimuth_deg, p1, p2):
    """(GemCad angle, distance) of the plane at `azimuth` through 2 points.

    The two points fix the one degree of freedom the azimuth leaves (the
    elevation).  The elevation is normalised so the outward normal leans
    along the azimuth -- sin(t) >= 0 -- which makes the result independent
    of the order of the points.
    """
    phi = math.radians(azimuth_deg)
    vx, vy, vz = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    h = vx * math.cos(phi) + vy * math.sin(phi)
    if abs(h) < 1e-12 and abs(vz) < 1e-12:
        raise ValueError("degenerate points; the elevation is undetermined")
    t = math.atan2(-vz, h)
    if math.sin(t) < 0.0:
        t += math.pi
    n = (math.sin(t) * math.cos(phi), math.sin(t) * math.sin(phi),
         math.cos(t))
    d = n[0] * p1[0] + n[1] * p1[1] + n[2] * p1[2]
    deg = math.degrees(t % (2.0 * math.pi))
    return (deg if deg <= 90.0 + 1e-9 else deg - 180.0), d


# --------------------------------------------------------------------------
# The barion/radiant construction: cut-corner rectangle, step crown,
# brilliant pavilion with crescent facets under the girdle.
# --------------------------------------------------------------------------

def mixed_cut(lw=1.0, corner=0.25, table=0.62,
              crown_angles=(42.0, 32.0), pavilion_angle=42.0,
              moon_angle=68.0, moon_fraction=0.35, lower_offset=1.5,
              girdle_pct=0.03, gear=96, ri=2.417, name="Mixed Cut"):
    """A barion-family mixed cut on a cut-corner rectangle.

    Half-width is 1 across the short axis; `lw` is the length-to-width
    ratio and `corner` how deeply the corners are cut (as `rect_outline`).
    The crown is `crown_angles` step rows down to a `table` fraction.  The
    pavilion is a brilliant fan solved from meetpoints:

      * corner MAINS at `pavilion_angle` through the cut-corner girdle
        edges; where they cross the axis is the culet;
      * CRESCENT (moon) facets at `moon_angle` hung on the side girdle
        edges, reaching `moon_fraction` of the pavilion depth at the side
        midline;
      * side mains solved through the crescent's lower midpoint and the
        culet;
      * lower halves anchored on the girdle vertices, `lower_offset`
        degrees steeper than the plane that would just reach the culet,
        so they die out on the mains above it -- the round brilliant's
        own halves-vs-mains arrangement (halves steeper than mains).
    """
    if corner <= 0.0:
        raise ValueError("a barion/radiant needs cut corners; "
                         f"corner must be positive, got {corner}")
    if not 0.0 < table < 1.0:
        raise ValueError(f"table must be a fraction in (0, 1), got {table}")
    if not 0.0 < pavilion_angle < moon_angle < 90.0:
        raise ValueError(f"need 0 < pavilion angle < moon angle < 90, got "
                         f"{pavilion_angle} and {moon_angle}")
    if not 0.0 < moon_fraction < 1.0:
        raise ValueError(f"moon_fraction is a fraction of the pavilion "
                         f"depth in (0, 1), got {moon_fraction}")
    if lower_offset <= 0.0:
        raise ValueError("lower halves must be cut steeper than the mains "
                         f"(lower_offset > 0), got {lower_offset}")

    outline = rect_outline(lw, corner)          # validates lw and corner
    w, ln = 1.0, float(lw)
    cd = next(d for az, d in outline if az == 45.0)
    hg = girdle_pct * w
    z_top, z_bot = hg, -hg

    tiers = _girdle_tiers(outline, gear)

    # --- corner mains: through the cut-corner girdle edges to the axis ---
    t_c = math.radians(pavilion_angle)
    d_cm = math.sin(t_c) * cd - math.cos(t_c) * z_bot
    z_c = -d_cm / math.cos(t_c)                 # the culet depth
    tiers.append(Tier(-pavilion_angle, d_cm,
                      tuple(_idx(45.0 + k * 90.0, gear) for k in range(4)),
                      name="PC", note="corner mains, meet at the culet"))

    # --- crescents and the side mains behind them ------------------------
    t_m = math.radians(moon_angle)
    z_m = z_bot + moon_fraction * (z_c - z_bot)
    side_classes = sorted({round(d, 9) for az, d in outline
                           if az in (0.0, 90.0, 180.0, 270.0)}, reverse=True)
    for dist in side_classes:
        azs = sorted(az for az, d in outline
                     if az in (0.0, 90.0, 180.0, 270.0)
                     and abs(d - dist) < 1e-9)
        idxs = tuple(_idx(az, gear) for az in azs)
        d_mo = math.sin(t_m) * dist - math.cos(t_m) * z_bot
        tiers.append(Tier(-moon_angle, d_mo, idxs, name="M",
                          note="crescent (moon) facets under the girdle"))
        # the crescent's lower midpoint, on the side midline at z_m
        rho_m = (d_mo + math.cos(t_m) * z_m) / math.sin(t_m)
        if rho_m <= 1e-9:
            raise ValueError("moon_fraction reaches past the crescent "
                             "plane; use a smaller fraction or a steeper "
                             "moon_angle")
        t_s = math.atan2(z_m - z_c, rho_m)      # rise and run both positive:
        d_s = -math.cos(t_s) * z_c              # first-quadrant root
        if t_s >= t_m - 1e-9:
            raise ValueError("the side mains come out as steep as the "
                             "crescents; deepen moon_fraction or steepen "
                             "moon_angle")
        # the crescent must survive: the side main's girdle-level trace
        # has to stand outside the girdle, or it swallows the crescent
        rho_g = (d_s + math.cos(t_s) * z_bot) / math.sin(t_s)
        if rho_g <= dist + 1e-9:
            raise ValueError("the side mains cut the crescents away; "
                             "deepen moon_fraction or steepen moon_angle")
        tiers.append(Tier(-math.degrees(t_s), d_s, idxs, name="PS",
                          note="side mains, meet at the culet"))

    # --- lower halves: steeper than the mains, hung on the vertices ------
    # Anchored on a girdle vertex, at `lower_offset` degrees steeper than
    # the plane through that vertex and the culet.  Steeper-than-through-
    # the-culet is the invariant that matters: it is what makes a half
    # die out on the mains instead of truncating the culet, and it is the
    # round brilliant's own arrangement (SRB halves 42.5 to mains 41.5).
    half_classes = {}
    for qc in (45.0, 135.0, 225.0, 315.0):      # each cut corner ...
        for turn in (-45.0, 45.0):              # ... meets two sides
            qs = (qc + turn) % 360.0
            p_s = w if qs in (0.0, 180.0) else ln
            ux, uy = math.cos(math.radians(qs)), math.sin(math.radians(qs))
            s = 1.0 if turn == -45.0 else -1.0  # which way the corner lies
            tx, ty = -s * uy, s * ux            # perpendicular, toward it
            b = cd * SQRT2 - p_s
            vx, vy = p_s * ux + b * tx, p_s * uy + b * ty
            az = (qs + s * 22.5) % 360.0
            phi = math.radians(_idx(az, gear) * 360.0 / gear)
            rho_eff = vx * math.cos(phi) + vy * math.sin(phi)
            # both the rise (z_bot - z_c, the culet is below the girdle)
            # and the run are positive: the first-quadrant root
            t_h = math.atan2(z_bot - z_c, rho_eff) \
                + math.radians(lower_offset)
            if t_h >= t_m:
                raise ValueError("lower_offset pushes the halves past the "
                                 "crescent angle")
            d_h = math.sin(t_h) * rho_eff - math.cos(t_h) * z_bot
            key = (round(math.degrees(t_h), 9), round(d_h, 9))
            half_classes.setdefault(key, []).append(_idx(az, gear))
    for (a_deg, d_h), idxs in sorted(half_classes.items(),
                                     key=lambda kv: -kv[0][1]):
        tiers.append(Tier(-a_deg, d_h, tuple(sorted(set(idxs))), name="LH",
                          note="lower halves, die out on the mains"))

    # --- step crown and table --------------------------------------------
    ct, z_table = _step_crown(outline, tuple(crown_angles),
                              w * (1.0 - table), z_top, gear)
    tiers += ct
    tiers.append(Tier(0.0, z_table, (_idx(0.0, gear),), name="T",
                      note="table"))

    detail = (f"L/W {lw:.3f}, table {table * 100:.0f}%, step crown over "
              f"brilliant pavilion")
    fold = 4 if abs(lw - 1.0) < 1e-12 else 2
    return CutDesign(name=name, gear=gear, fold=fold, mirror=True, ri=ri,
                     tiers=tuple(tiers), headings=(name, detail))


def barion(**over):
    """Watermeyer's barion: cut-corner square, step crown, crescent-moon
    brilliant pavilion."""
    kw = dict(lw=1.0, corner=0.25, table=0.62, crown_angles=(42.0, 32.0),
              pavilion_angle=42.0, moon_angle=68.0, name="Barion Cut")
    kw.update(over)
    return mixed_cut(**kw)


def radiant(**over):
    """Grossbard's radiant: the barion construction on a cut-corner
    rectangle with a length-to-width ratio."""
    kw = dict(lw=1.35, corner=0.25, table=0.62,
              crown_angles=(43.0, 35.0, 27.0), pavilion_angle=45.0,
              moon_angle=70.0, name="Radiant Cut")
    kw.update(over)
    return mixed_cut(**kw)


# --------------------------------------------------------------------------
# The princess: sharp-cornered square, step crown, chevron pavilion.
# --------------------------------------------------------------------------

def princess(chevrons=3, table=0.70, crown_angles=(40.0, 28.0),
             pavilion_angle=50.0, culet_angle=36.0, chevron_offset=7.5,
             girdle_pct=0.03, gear=96, ri=2.417, name="Princess Cut"):
    """A princess cut with `chevrons` rows of chevron facets.

    The pavilion is rows of V-shaped facet pairs whose spines run down
    the square's corner diagonals from girdle to culet.  Row angles run
    from `pavilion_angle` (measured along the diagonal, at the girdle)
    down to `culet_angle` at the culet; the spine they trace is convex,
    which is what lets every row survive.  `chevron_offset` is how far
    (degrees of azimuth) each half of a chevron pair sits off its
    diagonal.
    """
    if not isinstance(chevrons, int) or not 2 <= chevrons <= 6:
        raise ValueError(f"chevron rows must be an integer from 2 to 6 "
                         f"(two to four is the normal range), got "
                         f"{chevrons!r}")
    if not 0.0 < table < 1.0:
        raise ValueError(f"table must be a fraction in (0, 1), got {table}")
    if not 0.0 < culet_angle < pavilion_angle < 90.0:
        raise ValueError(f"chevron angles must decrease from the girdle to "
                         f"the culet within (0, 90), got {pavilion_angle} "
                         f"-> {culet_angle}")
    tooth = 360.0 / abs(gear)
    if not tooth - 1e-9 <= chevron_offset <= 45.0 - tooth + 1e-9:
        raise ValueError(f"chevron_offset must lie between one index tooth "
                         f"({tooth} deg) and {45.0 - tooth} deg, got "
                         f"{chevron_offset}")

    w = 1.0
    r_v = w * SQRT2                             # the square's corner radius
    hg = girdle_pct * w
    z_top, z_bot = hg, -hg
    outline = [(0.0, w), (90.0, w), (180.0, w), (270.0, w)]

    tiers = _girdle_tiers(outline, gear)

    # --- chevron rows down the diagonals ---------------------------------
    # Spine points on the diagonal: equal radial steps, elevations from
    # the (decreasing) row angles -- a convex polyline from the girdle
    # corner to the culet.  Each row is the plane pair at +-offset about
    # the diagonal solved through consecutive spine points; the mirror
    # symmetry of the pair about the diagonal makes one solve place both.
    cosd = math.cos(math.radians(chevron_offset))
    n = chevrons
    rho_prev, z_prev = r_v, z_bot
    for k in range(1, n + 1):
        f = (k - 1) / (n - 1)
        ang = pavilion_angle + (culet_angle - pavilion_angle) * f
        a_eff = math.radians(ang)
        rho_k = r_v * (1.0 - k / n)
        z_k = z_prev - (rho_prev - rho_k) * math.tan(a_eff)
        t = math.atan(math.tan(a_eff) / cosd)   # the pair's plane elevation
        d = math.sin(t) * rho_prev * cosd - math.cos(t) * z_prev
        idxs = sorted(_idx(diag + s * chevron_offset, gear)
                      for diag in (45.0, 135.0, 225.0, 315.0)
                      for s in (1.0, -1.0))
        tiers.append(Tier(-math.degrees(t), d, tuple(idxs), name=f"P{k}",
                          note=f"chevron row {k}"))
        rho_prev, z_prev = rho_k, z_k

    # --- step crown and table --------------------------------------------
    ct, z_table = _step_crown(outline, tuple(crown_angles),
                              w * (1.0 - table), z_top, gear)
    tiers += ct
    tiers.append(Tier(0.0, z_table, (_idx(0.0, gear),), name="T",
                      note="table"))

    detail = (f"{chevrons} chevron rows, table {table * 100:.0f}%, "
              f"step crown")
    return CutDesign(name=name, gear=gear, fold=4, mirror=True, ri=ri,
                     tiers=tuple(tiers), headings=(name, detail))


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
        P = _f.intersect_halfspaces(N, d)
        return P, _f.polytope_checks(P, N, d), N, d, t

    def names(D, P, tid):
        return [D.tiers[tid[j]].name for j in P.face_plane]

    def culet_faces(P):
        """(vertex, face count) at the solid's lowest point."""
        V = P.V.tolist()
        zmin = min(v[2] for v in V)
        low = [i for i, v in enumerate(V) if abs(v[2] - zmin) < 1e-6]
        touching = sum(1 for face in P.faces if set(face) & set(low))
        return [V[i] for i in low], touching

    # ====================== princess =====================================
    D = princess()
    P, chk, N, d, tid = build(D)
    good = chk["closed"] and chk["convex"] and not P.dropped
    ok &= good
    print(f"gems.mixed: the princess closes, is convex and spends no plane "
          f"({len(P.V)} vertices, {len(P.faces)} facets) "
          f"{'OK' if good else 'did not build'}")

    nm = names(D, P, tid)
    rows = [nm.count(f"P{k}") for k in (1, 2, 3)]
    crown = [nm.count(f"C{k}") for k in (1, 2)]
    good = (len(P.faces) == 37 and nm.count("G") == 4 and rows == [8, 8, 8]
            and crown == [4, 4] and nm.count("T") == 1)
    ok &= good
    print(f"gems.mixed: princess counts 4 girdle + chevrons {rows} + crown "
          f"{crown} + table = 37 (got {len(P.faces)}) "
          f"{'OK' if good else 'wrong arrangement'}")

    # the crown is STEP-like: full concentric rows of four-sided facets
    quads = all(len(face) == 4 for face, j in zip(P.faces, P.face_plane)
                if D.tiers[tid[j]].name.startswith("C"))
    ok &= quads
    print(f"gems.mixed: every princess crown facet is four-sided, as a "
          f"step crown must be {'OK' if quads else 'not step-like'}")

    # the pavilion is BRILLIANT-like: the last chevron row's 8 planes
    # converge on a single culet point on the axis
    low, touching = culet_faces(P)
    good = (len(low) == 1 and math.hypot(low[0][0], low[0][1]) < 1e-6
            and touching == 8)
    ok &= good
    print(f"gems.mixed: the princess pavilion converges on one axial culet "
          f"vertex shared by {touching} facets {'OK' if good else 'no'}")

    # square outline, princess-like total depth
    V = P.V.tolist()
    wx = max(abs(v[0]) for v in V)
    wy = max(abs(v[1]) for v in V)
    depth = (max(v[2] for v in V) - min(v[2] for v in V)) / (2.0 * wx)
    good = abs(wy / wx - 1.0) < 1e-9 and 0.55 < depth < 0.95
    ok &= good
    print(f"gems.mixed: the princess is square ({wy / wx:.6f}:1) and "
          f"{depth * 100:.1f}% deep {'OK' if good else 'wrong shape'}")

    # chevron rows are a real dial: each row is 8 facets, so 2 -> 4 rows
    # must add exactly 16
    c2 = build(princess(chevrons=2))[0]
    c4 = build(princess(chevrons=4))[0]
    good = len(c4.faces) - len(c2.faces) == 16 \
        and len(P.faces) - len(c2.faces) == 8
    ok &= good
    print(f"gems.mixed: chevron count steers the stone -- 2/3/4 rows give "
          f"{len(c2.faces)}/{len(P.faces)}/{len(c4.faces)} facets "
          f"{'OK' if good else 'rows did not add facets'}")

    # ====================== barion =======================================
    B = barion()
    Pb, chkb, Nb, db, tb = build(B)
    good = chkb["closed"] and chkb["convex"] and not Pb.dropped
    ok &= good
    print(f"gems.mixed: the barion closes, is convex and spends no plane "
          f"({len(Pb.V)} vertices, {len(Pb.faces)} facets) "
          f"{'OK' if good else 'did not build'}")

    nb = names(B, Pb, tb)
    parts = {x: nb.count(x) for x in ("G", "M", "PC", "PS", "LH", "T")}
    crown_b = [nb.count(f"C{k}") for k in (1, 2)]
    good = (len(Pb.faces) == 45 and parts == {"G": 8, "M": 4, "PC": 4,
                                              "PS": 4, "LH": 8, "T": 1}
            and crown_b == [8, 8])
    ok &= good
    print(f"gems.mixed: barion counts {parts} + crown {crown_b} = 45 "
          f"(got {len(Pb.faces)}) {'OK' if good else 'wrong arrangement'}")

    # the crescents really are crescents: hung on the girdle's lower
    # edge, wider than they are tall, confined to the upper pavilion
    z_bot_b = -0.03
    z_c_b = min(v[2] for v in Pb.V.tolist())
    moon_ok = True
    for face, j in zip(Pb.faces, Pb.face_plane):
        if B.tiers[tb[j]].name != "M":
            continue
        pts = [Pb.V.tolist()[i] for i in face]
        on_girdle = sum(1 for p in pts if abs(p[2] - z_bot_b) < 1e-6)
        span = max(max(p[0] for p in pts) - min(p[0] for p in pts),
                   max(p[1] for p in pts) - min(p[1] for p in pts))
        rise = max(p[2] for p in pts) - min(p[2] for p in pts)
        deepest = min(p[2] for p in pts)
        moon_ok &= (on_girdle >= 2 and span > rise
                    and deepest > 0.6 * z_c_b)
    ok &= moon_ok
    print(f"gems.mixed: every barion crescent hangs on the girdle edge, "
          f"wider than tall, in the upper pavilion "
          f"{'OK' if moon_ok else 'not crescents'}")

    # the pavilion is brilliant-like: 8 mains (corner + side) converge on
    # one axial culet vertex; the halves die out on the mains above it
    low_b, touching_b = culet_faces(Pb)
    good = (len(low_b) == 1 and math.hypot(low_b[0][0], low_b[0][1]) < 1e-6
            and touching_b == 8)
    ok &= good
    print(f"gems.mixed: the barion's 8 mains converge on one axial culet "
          f"vertex ({touching_b} facets there), halves die out above "
          f"{'OK' if good else 'no'}")

    quads = all(len(face) == 4 for face, j in zip(Pb.faces, Pb.face_plane)
                if B.tiers[tb[j]].name.startswith("C"))
    Vb = Pb.V.tolist()
    wxb = max(abs(v[0]) for v in Vb)
    wyb = max(abs(v[1]) for v in Vb)
    good = quads and abs(wyb / wxb - 1.0) < 1e-9
    ok &= good
    print(f"gems.mixed: the barion crown is all four-sided step facets on "
          f"a square outline ({wyb / wxb:.6f}:1) "
          f"{'OK' if good else 'wrong crown'}")

    # ====================== radiant ======================================
    R = radiant()
    Pr, chkr, Nr, dr, tr = build(R)
    good = chkr["closed"] and chkr["convex"] and not Pr.dropped
    ok &= good
    print(f"gems.mixed: the radiant closes, is convex and spends no plane "
          f"({len(Pr.V)} vertices, {len(Pr.faces)} facets) "
          f"{'OK' if good else 'did not build'}")

    nr = names(R, Pr, tr)
    parts_r = {x: nr.count(x) for x in ("G", "M", "PC", "PS", "LH", "T")}
    crown_r = [nr.count(f"C{k}") for k in (1, 2, 3)]
    good = (len(Pr.faces) == 53 and parts_r == {"G": 8, "M": 4, "PC": 4,
                                                "PS": 4, "LH": 8, "T": 1}
            and crown_r == [8, 8, 8])
    ok &= good
    print(f"gems.mixed: radiant counts {parts_r} + crown {crown_r} = 53 "
          f"(got {len(Pr.faces)}) {'OK' if good else 'wrong arrangement'}")

    # a RECTANGLE of the asked-for ratio, not a square that passed
    # topology checks flat -- measure the aspect, never assert it
    Vr = Pr.V.tolist()
    wxr = max(abs(v[0]) for v in Vr)
    wyr = max(abs(v[1]) for v in Vr)
    good = abs(wyr / wxr - 1.35) < 0.02
    ok &= good
    print(f"gems.mixed: the radiant measures {wyr / wxr:.3f}:1, asked for "
          f"1.350 {'OK' if good else 'wrong ratio'}")

    low_r, touching_r = culet_faces(Pr)
    good = (len(low_r) == 1 and math.hypot(low_r[0][0], low_r[0][1]) < 1e-6
            and touching_r == 8)
    ok &= good
    print(f"gems.mixed: the radiant pavilion converges on one axial culet "
          f"vertex ({touching_r} facets) {'OK' if good else 'no'}")

    # crescents survive on BOTH the long and the short sides
    moon_sides = set()
    for face, j in zip(Pr.faces, Pr.face_plane):
        if R.tiers[tr[j]].name == "M":
            pts = [Vr[i] for i in face]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            moon_sides.add("x" if abs(cx) > abs(cy) else "y")
    good = moon_sides == {"x", "y"}
    ok &= good
    print(f"gems.mixed: the radiant carries crescents on both side pairs "
          f"{'OK' if good else 'a side pair lost its crescents'}")

    # ====================== refusals =====================================
    bad = 0
    for fn, kw in ((princess, dict(chevrons=1)),
                   (princess, dict(table=1.2)),
                   (princess, dict(pavilion_angle=36.0, culet_angle=50.0)),
                   (princess, dict(chevron_offset=44.9)),
                   (barion, dict(corner=0.0)),
                   (barion, dict(moon_angle=40.0)),
                   (barion, dict(crown_angles=(30.0, 40.0))),
                   (radiant, dict(lw=0.8))):
        try:
            fn(**kw)
        except ValueError:
            bad += 1
    good = bad == 8
    ok &= good
    print(f"gems.mixed: impossible mixed cuts are refused ({bad}/8) "
          f"{'OK' if good else 'accepted nonsense'}")

    print("RESULT:", "OK" if ok else "FAILURE")
    if not ok:
        raise AssertionError("gems.mixed self-test failed")
