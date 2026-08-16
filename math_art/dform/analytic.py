# Closed-form relatives of the D-form: shapes that need no solver.
#
# Part of the Math Art D-form engine (`math_art/dform/`).  Python and
# numpy only -- no `bpy`.
#
# The D-form proper has to be solved for: two flat outlines determine a
# shape, but only implicitly, so `solve.py` relaxes a mesh into it.  The
# shapes here are the opposite case.  Each is a closed or nearly-closed
# developable assembled from patches -- cones and cylinders -- whose
# parametrisations are known in closed form, so they are built exactly,
# in one pass, with no iteration and no convergence to worry about.
# Their curvature still lives entirely on creases and seams, which is
# what puts them in this package rather than in a surface zoo.
#
# VESICA -- Mundilova and Wills's folded Vesica Piscis.  The development
# is the union of two unit discs whose centres are 2u apart; folded, it
# becomes one cylindrical and two conical patches.  Everything below
# follows their explicit parametrisation: the cone of apex height h has
# ruling lengths inherited from the flat disc, which fixes the radius of
# the (planar) seam curve as
#
#     r(s)^2 = |c_u(s)|^2 - h^2 = 1 + 2u cos s + u^2 - h^2,
#
# and demanding that the seam keep the flat curve's arclength fixes its
# polar angle,
#
#     alpha'(s) = sqrt((1 + u cos s)^2 - h^2) / r(s)^2,
#
# which the paper evaluates as elliptic integrals of the third kind and
# which is integrated numerically here.  The crease sits at the fraction
# t(s) = h / (cos s + u + h) along each ruling.  The whole thing is real
# only for h <= h_max = 1 - u^2, and the developed creases come out
# circular exactly at h = (1 - u^2) / 2u; the classical Vesica Piscis is
# u = 1/2, where those two coincide at h = 3/4.
#
# KOMAN_CURL -- Ilhan Koman's spiral developable sculptures, in the
# reconstruction of Akgun, Kaya, Koman and Akleman (Koman never wrote
# his method down).  A strip is slit into alternating panels; the
# planar ones are rotated into a common plane, each sliding under its
# neighbour by 2d, which shortens the panel between them and forces it
# to arch.  Their Eq. 1 gives the turn per panel as a = arctan(2d/h),
# so n = 2*pi/a panels close into a ring of radius ~ h/2 (their Eq. 2);
# letting d or h vary along the strip opens the ring into a spiral.
#
# NOT HERE, deliberately:
#   * the perpendicular cylinder pair is already shipped exactly, as
#     `steinmetz_generator.py`'s BICYLINDER -- two cylindrical lune
#     patches meeting along a pair of planar ellipse-like edges is the
#     same surface, built watertight there.
#   * the piecewise-conical Mobius band is not here yet; see BACKLOG.
#
# References:
#   K. Mundilova, T. Wills, "Folding the Vesica Piscis," Bridges 2018,
#       pp. 535-538 (local: research/journals/bridges/2018/
#       bridges2018-535/).  The parametrisation used here is their
#       Eq. (1), the alpha' integrand, and Eq. (2).
#   K. Mundilova, "Notes on the Integration of the Angular Function in
#       the Parametrization of the Vesica Piscis," technical report,
#       2018 -- the closed-form elliptic evaluation.
#   T. Akgun, I. Kaya, A. Koman, E. Akleman, "Spiral Developable
#       Sculptures of Ilhan Koman," Bridges 2007, pp. 47-52 (local:
#       research/journals/bridges/2007/bridges2007-47/), Eqs. (1)-(2).
#   T. Akgun, A. Koman, E. Akleman, "Developable Sculptural Forms of
#       Ilhan Koman," Bridges 2006, pp. 343-350 (local:
#       research/journals/bridges/2006/bridges2006-343/).
#   E. Demaine, G. Price, "Generalized D-Forms Have No Spurious
#       Creases," Discrete Comput. Geom. 43 (2009), 179-186 -- why a
#       D-form proper has no interior creases, and these do.

import numpy as np

ANALYTIC_KINDS = ('VESICA', 'KOMAN_CURL', 'KOMAN_SPIRAL')


# ---------------------------------------------------------------- vesica

def vesica_max_height(u):
    """Largest apex height for which the fold exists: h_max = 1 - u^2."""
    return 1.0 - float(u) ** 2


def vesica_circular_height(u):
    """Apex height whose DEVELOPED creases are circular, (1-u^2)/2u.

    Mundilova and Wills's surprise: at this height the crease, unrolled,
    is exactly a circular arc.  It lies below h_max only for u >= 1/2;
    at u = 1/2 the two coincide and the shape is the classical folded
    Vesica Piscis.
    """
    u = float(u)
    return (1.0 - u * u) / (2.0 * u) if u > 1e-9 else float('inf')


def _vesica_profile(u, h, samples):
    """Seam, crease and apex for the x >= 0 half of the development."""
    u = float(u)
    h = float(h)
    s1 = np.arccos(np.clip(-u, -1.0, 1.0))
    s = np.linspace(-s1, s1, samples)

    cs = np.cos(s)
    r2 = 1.0 + 2.0 * u * cs + u * u - h * h
    r = np.sqrt(np.maximum(r2, 0.0))
    # alpha' = sqrt((1 + u cos s)^2 - h^2) / r^2, integrated from the
    # middle outward; the integrand is even in s, so alpha is odd and
    # the seam comes out symmetric about the x-axis without any fitting
    num = np.sqrt(np.maximum((1.0 + u * cs) ** 2 - h * h, 0.0))
    dalpha = num / np.maximum(r2, 1e-15)
    alpha = np.concatenate([[0.0], np.cumsum(
        0.5 * (dalpha[1:] + dalpha[:-1]) * np.diff(s))])
    alpha -= alpha[len(alpha) // 2] if len(alpha) % 2 else 0.0

    seam = np.stack([r * np.cos(alpha), r * np.sin(alpha),
                     np.zeros_like(r)], axis=1)
    apex = np.array([0.0, 0.0, h])
    t = h / np.maximum(cs + u + h, 1e-12)
    crease = (1.0 - t)[:, None] * apex + t[:, None] * seam
    return s, seam, crease, apex, t


def build_vesica(u=0.5, height=None, samples=96, rings=10):
    """The folded Vesica Piscis, and its two-parameter family.

    `u` is the half-distance between the two disc centres (u = 1/2 is
    the Vesica Piscis itself) and `height` the apex height h; it
    defaults to the circular-crease value, clamped to h_max.

    Returns (verts, faces, seam_polyline, crease_polylines).  The
    surface is three
    patches: an upper cone from the seam in to the crease, a vertical
    cylinder from the crease down to its mirror, and the mirrored lower
    cone back out to the seam.  The cylinder really is one -- the two
    creases differ only in the sign of z, so every ruling between them
    is parallel to the z-axis.

    Topologically it is a tube over s that CLOSES on itself in the other
    direction: leaving the seam, over the crease, down the cylinder, up
    the mirrored cone and back to the same seam point.  At the two ends
    of the s-range the crease meets the seam (there t = 1), so those
    columns collapse to single points -- the two places where the
    development's circles cross -- and the tube is capped there.  So the
    mesh is welded, not three separate strips: a crease is still
    developable, and testing that is only meaningful if the vertices
    along it actually carry the faces from both sides.
    """
    u = min(max(float(u), 0.05), 0.95)
    hmax = vesica_max_height(u)
    h = vesica_circular_height(u) if height is None else float(height)
    h = min(max(h, 1e-3), hmax)

    n = max(16, int(samples))
    k = max(2, int(rings))
    s, seam, crease, apex, t = _vesica_profile(u, h, n)
    mirror = np.array([1.0, 1.0, -1.0])

    V = []
    cols = []
    for i in range(1, n - 1):
        A, B, C = seam[i], crease[i], crease[i] * mirror
        base = len(V)
        for j in range(k):                     # seam -> crease (cone)
            V.append(A + (B - A) * (j / k))
        for j in range(k):                     # crease -> mirror (cylinder)
            V.append(B + (C - B) * (j / k))
        for j in range(k):                     # mirror -> seam (cone)
            V.append(C + (A - C) * (j / k))
        cols.append(list(range(base, base + 3 * k)))
    cap0 = len(V)
    V.append(seam[0])
    cap1 = len(V)
    V.append(seam[-1])

    ring = 3 * k
    F = []
    for a, b in zip(cols[:-1], cols[1:]):
        for j in range(ring):
            j2 = (j + 1) % ring
            F.append((a[j], a[j2], b[j2], b[j]))
    for j in range(ring):                      # the two collapsed ends
        F.append((cap0, cols[0][(j + 1) % ring], cols[0][j]))
        F.append((cap1, cols[-1][j], cols[-1][(j + 1) % ring]))

    # ordered polylines, each running from one collapsed end to the
    # other, so the generator can mark them sharp without guessing at
    # connectivity: the seam is a fold between two glued paper edges and
    # the two creases are folds in the paper, and all three shade sharp
    seam_poly = [cap0] + [c[0] for c in cols] + [cap1]
    creases = [[cap0] + [c[k] for c in cols] + [cap1],
               [cap0] + [c[2 * k] for c in cols] + [cap1]]
    return np.asarray(V, dtype=float), F, seam_poly, creases


# ------------------------------------------------------------ Koman curl

def koman_turn(d, h):
    """Turn per panel, a = arctan(2d/h) -- Akgun et al. Eq. (1)."""
    return float(np.arctan2(2.0 * float(d), max(float(h), 1e-9)))


def koman_slide_to_close(n, h=1.0):
    """The slide d that makes `n` panels close the ring exactly.

    Akgun et al. note that a = 2*pi/n closes the arc into a regular
    n-gon with n the number of panels; inverting their Eq. 1 gives the
    slide that does it, d = h tan(a) / 2.  Driving the sculpture from
    the panel count is much the better handle -- pick d directly and
    the ring almost never closes.
    """
    # a = 2*pi/n, and d = h tan(a) / 2.  n >= 5 keeps a under a right
    # angle, where the tangent runs away
    a = 2.0 * np.pi / max(int(n), 5)
    return 0.5 * float(h) * float(np.tan(a))


def build_koman_curl(panels=32, height=1.0, slide=-1.0, width=0.55,
                     arch_samples=10, growth=0.0, skew=0.5, hole=0.45,
                     tilt=0.42, thickness=0.004):
    """Koman's curled strip: a slit sheet coiled into an overlapping ring.

    THE SHEET IS ONE PIECE.  Akgun et al.'s Figure 3 cuts a rectangle
    with slits from alternating edges, which turns it into a single long
    serpentine ribbon; coiling that ribbon is the sculpture.  Every
    segment of it -- a "tongue" -- stays flat and is pressed into a
    common plane, each rotated from the last by
    a = arctan(2d/h) (their Eq. 1), so the tongues overlap like a fanned
    deck of cards around a central hole and `panels` of them sweep a
    ring of radius about h/2 (their Eq. 2).

    The paper between two tongues is the part that cannot stay flat.
    Sliding each tongue under its neighbour by 2d leaves that connector
    longer than the gap it has to bridge, so it buckles into an arch --
    those arches are the only genuinely curved surfaces here, and the
    ribbon being serpentine, they alternate between the outer and the
    inner rim.

    `skew` swings each tongue away from radial, which is what gives the
    pinwheel; `growth` grows the slide along the ribbon so the ring
    opens into a spiral, as Koman's own sculptures do.  `thickness`
    stacks the tongues by a sheet's worth each so the overlaps read as
    layers rather than as coincident faces -- a real coil closes
    slightly out of plane for the same reason.
    """
    n = max(3, int(panels))
    h = max(float(height), 1e-3)
    l = max(float(width), 0.05) * h        # tongue width along the ribbon
    r_in = max(float(hole), 0.02) * h
    m = max(3, int(arch_samples))
    step = float(thickness) * h
    # a negative slide means "whatever closes the ring", which is what
    # anyone actually wants; a chosen d leaves the ring open at some
    # arbitrary angle, as 24 panels at d = 0.06 does (164 degrees)
    if slide is None or float(slide) < 0.0:
        slide = koman_slide_to_close(n, h)

    V = []
    F = []
    rims = []                              # (outer edge, inner edge) per tongue
    slack = []                             # each tongue's own 2d
    ang = 0.0
    d0 = max(float(slide), 1e-5)
    g = 1.0

    for i in range(n):
        # The SPIRAL comes from shrinking h, not from d.  The paper is
        # explicit: keep a constant (which means holding h/d fixed) and
        # vary the size, and Koman himself "shrank h exponentially".
        # Since the ring radius goes as h/2, that is what walks the
        # tongues inward; scaling d alone only changes the turn rate and
        # leaves every tongue on the same circle.
        hi, li, ri, di = h * g, l * g, r_in * g, d0 * g
        c, s = np.cos(ang), np.sin(ang)
        u = np.array([np.cos(ang + skew), np.sin(ang + skew), 0.0])
        # each tongue LEANS: it is lying on top of the ones before it,
        # so its trailing edge rides up on them.  Coplanar tongues merge
        # into one flat annulus in a render and the fanned-deck reading
        # -- which is what the sculpture looks like -- disappears.
        v = np.array([-np.sin(ang + skew) * np.cos(tilt),
                      np.cos(ang + skew) * np.cos(tilt),
                      np.sin(tilt)])
        base = np.array([ri * c, ri * s, i * step])
        p0 = base
        p1 = base + hi * u
        p2 = p1 + li * v
        p3 = base + li * v
        k = len(V)
        V.extend([p0, p1, p2, p3])
        F.append((k, k + 1, k + 2, k + 3))
        rims.append(((k + 1, k + 2), (k, k + 3)))
        slack.append(2.0 * di)
        ang += koman_turn(di, hi)          # constant while h/d is
        g *= (1.0 + float(growth))

    # the connectors: tongue i hands over to tongue i+1 at the outer rim
    # when i is even and at the inner rim when it is odd, because the
    # ribbon runs out, back, out, back through the slit sheet
    for i in range(n - 1):
        side = 0 if i % 2 == 0 else 1
        a0, a1 = rims[i][side]
        b0, b1 = rims[i + 1][side]
        _arch(V, F, np.asarray(V[a0]), np.asarray(V[a1]),
              np.asarray(V[b0]), np.asarray(V[b1]), slack[i], m)

    return np.asarray(V, dtype=float), F


def build_koman_spiral(blades=80, turn=None, growth=0.034, blade_len=1.70,
                       blade_width=0.60, skew=0.70, lean=1.00,
                       twist=0.0, samples=5, core=0.055):
    """Koman's spiral developable: a coiling spine with blades off it.

    THIS IS THE SCULPTURE, as against the closed wreath of
    `build_koman_curl`.  Akgun et al.'s Figure 4 cuts the alternating
    slits into a TRAPEZOID rather than a rectangle, so the ribbon tapers
    and the coil never closes -- it winds outward as a logarithmic
    spiral, which is what Koman's own metal pieces are (their Figures 2
    and 4C).  The paper's account of how: hold the turn per panel `a`
    constant and shrink the size exponentially along the strip, which is
    what they conclude Koman did.

    The uncut part of the sheet is a SPINE that coils, and each slit
    strip is a blade springing from it.  Blades stay planar -- that is
    the paper's own "blue pieces stay planar", and it is what keeps the
    surface developable -- so the twisting look of the metal sculptures
    comes from each blade being turned a little further round the coil
    than the last, not from bending any one of them.

    Every length scales with the local spiral radius, so the form is
    self-similar: `blade_len`, `blade_width` and `spine` are all given
    as multiples of it.  `twist` rotates a blade's far edge against its
    root for the look of the metal pieces; it is off by default because
    a twisted blade is a hyperbolic paraboloid, no longer developable
    and no longer something you can cut from flat stock.
    """
    n = max(3, int(blades))
    a = (2.0 * np.pi / 30.0) if turn is None else float(turn)
    g = 1.0 + float(growth)
    m = max(1, int(samples))
    r = max(float(core), 1e-4)

    V = []
    F = []
    roots = []                              # (inner, outer) of each root edge

    for i in range(n):
        th = i * a
        er = np.array([np.cos(th), np.sin(th), 0.0])
        et = np.array([-np.sin(th), np.cos(th), 0.0])
        u = np.cos(skew) * er + np.sin(skew) * et       # blade long axis
        vp = -np.sin(skew) * er + np.cos(skew) * et     # blade width, in plane
        v = np.cos(lean) * vp + np.sin(lean) * np.array([0.0, 0.0, 1.0])
        B = r * er
        L, W = blade_len * r, blade_width * r

        base = len(V)
        for j in range(m + 1):
            t = j / m
            # the twist turns the width axis about the blade's own long
            # axis as we run out along it (Rodrigues about u)
            ang = twist * t
            vt = (v * np.cos(ang) + np.cross(u, v) * np.sin(ang)
                  + u * float(np.dot(u, v)) * (1.0 - np.cos(ang)))
            root = B + L * t * u
            V.append(root)
            V.append(root + W * vt)
        for j in range(m):
            q = base + 2 * j
            F.append((q, q + 2, q + 3, q + 1))
        roots.append((base, base + 1))
        r *= g

    # the spine: the uncut band of the sheet, joining each blade's root
    # to the next.  It is what makes the whole thing one piece of paper.
    for i in range(n - 1):
        a0, a1 = roots[i]
        b0, b1 = roots[i + 1]
        F.append((a0, b0, b1, a1))

    return np.asarray(V, dtype=float), F


def _arch(V, F, a0, a1, b0, b1, slack, m):
    """A quad strip from edge (a0,a1) to edge (b0,b1), bowed up by
    `slack` of extra paper.

    The two edges are a turn apart, so the straight bridge between them
    is shorter than the paper that has to span it; the surplus goes into
    a circular bow.  Ruling the strip straight across keeps it a
    cylinder patch, so the connector stays developable.
    """
    chord = 0.5 * (float(np.linalg.norm(b0 - a0))
                   + float(np.linalg.norm(b1 - a1)))
    rise = 0.0
    if chord > 1e-9 and slack > 1e-12:
        th = _arc_angle(chord, chord + slack)
        R = chord / (2.0 * max(np.sin(th), 1e-9))
        rise = R * (1.0 - np.cos(th))
    up = np.array([0.0, 0.0, 1.0])
    base = len(V)
    for j in range(m + 1):
        t = j / m
        lift = up * (rise * np.sin(np.pi * t))
        V.append(a0 + t * (b0 - a0) + lift)
        V.append(a1 + t * (b1 - a1) + lift)
    for j in range(m):
        q = base + 2 * j
        F.append((q, q + 2, q + 3, q + 1))


def _arc_angle(chord, length):
    """Half-angle of a circular arc of given chord and arclength.

    Solves chord/length = sin(x)/x by bisection -- it is monotone on
    (0, pi], so there is nothing clever to do and nothing to go wrong.
    """
    ratio = min(max(chord / max(length, 1e-12), 1e-6), 1.0 - 1e-12)
    lo, hi = 1e-6, np.pi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sin(mid) / mid > ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _arc_points(p_in0, p_in1, p_out0, p_out1, theta, m):
    """Sample the arch's ruling endpoints, inner rim to outer rim.

    The arch bows out of the base plane; the rulings run radially, so
    the patch is a cylinder and stays developable.
    """
    tt = np.linspace(0.0, 1.0, m)
    # a circular profile of half-angle theta, rising over the chord
    ph = (2.0 * tt - 1.0) * theta
    lift = (np.cos(ph) - np.cos(theta)) / max(1.0 - np.cos(theta), 1e-12)
    inner = p_in0[None, :] + tt[:, None] * (p_in1 - p_in0)[None, :]
    outer = p_out0[None, :] + tt[:, None] * (p_out1 - p_out0)[None, :]
    rise = np.zeros((m, 3))
    rise[:, 2] = lift * np.linalg.norm(p_out1 - p_out0) * 0.5
    return 0.5 * (inner + outer) + rise


# ---------------------------------------------------------------- checks

def _defects(V, faces):
    """Angle defect at every vertex of a quad/ngon mesh."""
    defect = np.full(len(V), 2 * np.pi)
    for f in faces:
        k = len(f)
        for i in range(k):
            a = V[f[i]]
            u1 = V[f[(i - 1) % k]] - a
            u2 = V[f[(i + 1) % k]] - a
            n1 = np.linalg.norm(u1)
            n2 = np.linalg.norm(u2)
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            defect[f[i]] -= np.arccos(np.clip(
                float(np.dot(u1, u2)) / (n1 * n2), -1.0, 1.0))
    return defect


def _selftest():
    ok = True

    # h_max and the circular-crease height must meet at the Vesica
    # Piscis itself, u = 1/2, h = 3/4 -- the paper's stated special case
    good = (abs(vesica_max_height(0.5) - 0.75) < 1e-12
            and abs(vesica_circular_height(0.5) - 0.75) < 1e-12)
    ok &= good
    print(f"analytic: vesica u=1/2 gives h_max = h_circ = 3/4 "
          f"{'OK' if good else 'FAIL'}")

    # the seam must keep the development's arclength: that is the whole
    # content of the alpha' integrand, and it is what makes the fold an
    # isometry rather than a drawing of one
    worst = 0.0
    for u in (0.3, 0.5, 0.7):
        h = min(vesica_circular_height(u), vesica_max_height(u))
        s, seam, crease, apex, t = _vesica_profile(u, h, 4001)
        flat = np.stack([np.cos(s) + u, np.sin(s)], axis=1)
        L3 = float(np.sum(np.linalg.norm(np.diff(seam, axis=0), axis=1)))
        L2 = float(np.sum(np.linalg.norm(np.diff(flat, axis=0), axis=1)))
        worst = max(worst, abs(L3 - L2) / L2)
    good = worst < 1e-6
    ok &= good
    print(f"analytic: vesica seam keeps the flat arclength "
          f"(rel {worst:.2e}) {'OK' if good else 'FAIL'}")

    # and the rulings must keep their length too: |c - apex| = |c_u|,
    # which is what fixes r(s).  Both together are the isometry.
    worst = 0.0
    for u in (0.3, 0.5, 0.7):
        h = min(vesica_circular_height(u), vesica_max_height(u))
        s, seam, crease, apex, t = _vesica_profile(u, h, 501)
        rul = np.linalg.norm(seam - apex, axis=1)
        flat = np.sqrt(1.0 + 2 * u * np.cos(s) + u * u)
        worst = max(worst, float(np.max(np.abs(rul / flat - 1.0))))
    good = worst < 1e-9
    ok &= good
    print(f"analytic: vesica ruling lengths match the flat disc "
          f"(rel {worst:.2e}) {'OK' if good else 'FAIL'}")

    # the crease must touch the seam plane exactly at the two points
    # where the circles cross, and nowhere else rise below it
    s, seam, crease, apex, t = _vesica_profile(0.5, 0.75, 401)
    good = (abs(crease[0, 2]) < 1e-9 and abs(crease[-1, 2]) < 1e-9
            and float(np.min(crease[1:-1, 2])) > 0.0)
    ok &= good
    print(f"analytic: vesica crease meets z=0 only at the two circle "
          f"crossings {'OK' if good else 'FAIL'}")

    # The built surface must be developable everywhere off the seam --
    # INCLUDING along the two creases.  Folding paper does not stretch
    # it, so a crease carries no Gaussian curvature either; if the angle
    # sum there missed 2*pi the fold would not be realisable in paper,
    # which is exactly the question Mundilova and Wills set out to
    # settle for this shape.
    def defect_split(n, k=8):
        V, F, si, ci = build_vesica(0.5, samples=n, rings=k)
        d = _defects(V, F)
        off = np.ones(len(V), bool)
        off[list(si)] = False
        cr = np.zeros(len(V), bool)
        for poly in ci:
            cr[list(poly)] = True
        # the crease polylines run end to end, so they share the two
        # collapsed points with the seam; those are seam vertices and
        # carry real curvature, so they are not part of the crease test
        cr &= off
        return (float(np.max(np.abs(d[off & ~cr]))),
                float(np.max(np.abs(d[cr]))),
                float(np.sum(d[~off])))

    inner, crease_120, tot_120 = defect_split(120)
    _, crease_240, _ = defect_split(240)
    _, crease_480, tot_480 = defect_split(480)

    # inside a patch the quads are EXACTLY planar -- two rulings of one
    # cone meet at its apex and so span a plane, and the cylinder's
    # rulings are parallel -- so this is machine precision, not a
    # tolerance
    good = inner < 1e-9
    ok &= good
    print(f"analytic: vesica patch interiors exactly developable "
          f"({inner:.2e}) {'OK' if good else 'FAIL'}")

    # The crease is the real question, and the answer has to be read as
    # a convergence rather than a threshold: a smooth crease sampled at
    # n points reports its own O(1/n) truncation, so the test is that
    # refining actually kills it.  It halves as n doubles.
    good = (crease_240 < 0.62 * crease_120
            and crease_480 < 0.62 * crease_240)
    ok &= good
    print(f"analytic: vesica creases developable in the limit "
          f"(defect {crease_120:.2e} -> {crease_240:.2e} -> "
          f"{crease_480:.2e} as n doubles) {'OK' if good else 'FAIL'}")

    # ... and all of the curvature is on the seam, totalling 4*pi: the
    # surface is closed, so this is discrete Gauss-Bonnet, and it also
    # proves nothing leaked into the two collapsed end points
    good = abs(tot_480 - 4 * np.pi) < 0.01 * 4 * np.pi and tot_480 > tot_120
    ok &= good
    print(f"analytic: vesica seam carries 4pi (got {tot_480:.4f} at "
          f"n=480, {tot_120:.4f} at n=120) {'OK' if good else 'FAIL'}")

    # Koman: the turn per panel is Eq. (1), and n panels of it close a
    # ring; Eq. (2) says its radius is about h/2
    good = abs(koman_turn(0.5, 1.0) - np.arctan(1.0)) < 1e-12
    ok &= good
    print(f"analytic: koman turn a = arctan(2d/h) {'OK' if good else 'FAIL'}")

    # the small-angle regime the paper works in: 2d << h makes a small,
    # and then n = 2pi/a panels close a ring of radius ~ h/2
    h = 1.0
    d = 0.02
    a = koman_turn(d, h)
    n_close = 2 * np.pi / a
    r = n_close * d / (2 * np.pi)
    good = abs(r - h / 2) / (h / 2) < 0.01
    ok &= good
    print(f"analytic: koman ring radius ~ h/2 (got {r:.4f} for h/2="
          f"{h/2:.4f}) {'OK' if good else 'FAIL'}")

    # the derived slide must CLOSE the ring, for every panel count.
    # Choosing d by hand almost never does -- the 24 panels at d = 0.06
    # that looked like a D-form generator bug swept only 164 degrees --
    # so this is the gate that keeps the default honest.
    worst = 0.0
    for n in (8, 16, 24, 32, 48, 90):
        d = koman_slide_to_close(n)
        worst = max(worst, abs(koman_turn(d, 1.0) * n - 2 * np.pi))
    good = worst < 1e-9
    ok &= good
    print(f"analytic: koman panels close a full turn (worst {worst:.2e} "
          f"rad) {'OK' if good else 'FAIL'}")

    # ... and the built ring really is an annulus: a central hole, and
    # a flat wreath rather than a tube
    V, F = build_koman_curl(panels=24)
    r = np.linalg.norm(V[:, :2], axis=1)
    ext = V.max(axis=0) - V.min(axis=0)
    good = (len(F) > 0 and np.all(np.isfinite(V))
            and r.min() > 0.2 * r.max()          # a real hole
            and abs(ext[0] - ext[1]) < 0.1 * ext[0]   # round
            and ext[2] < 0.4 * ext[0])           # a wreath, not a tube
    ok &= good
    print(f"analytic: koman curl is a holed wreath (hole "
          f"{r.min()/r.max():.2f} of radius, rise {ext[2]/ext[0]:.2f}) "
          f"{'OK' if good else 'FAIL'}")

    # a positive growth must actually open the ring into a spiral: the
    # tongues have to walk outward instead of returning to the start
    Vs, _ = build_koman_curl(panels=24, growth=0.05)
    rs = np.linalg.norm(Vs[:, :2], axis=1)
    good = np.all(np.isfinite(Vs)) and rs.max() > 1.02 * r.max()
    ok &= good
    print(f"analytic: koman growth opens the ring into a spiral "
          f"({r.max():.2f} -> {rs.max():.2f}) {'OK' if good else 'FAIL'}")

    # --- the spiral (Koman's actual sculpture) ---

    def blade_spine_planarity(tw, nb=40, sm=5):
        V, F = build_koman_spiral(blades=nb, samples=sm, twist=tw)

        def worst(fs):
            w = 0.0
            for f in fs:
                P = V[list(f)]
                nrm = np.cross(P[1] - P[0], P[2] - P[0])
                ln = float(np.linalg.norm(nrm))
                if ln < 1e-14:
                    continue
                w = max(w, abs(float(np.dot(nrm / ln, P[3] - P[0])))
                        / max(float(np.linalg.norm(P[1] - P[0])), 1e-12))
            return w
        return worst(F[:nb * sm]), worst(F[nb * sm:])

    flat_b, _ = blade_spine_planarity(0.0)
    twist_b, _ = blade_spine_planarity(0.8)

    # The blades must be EXACTLY planar untwisted.  That is the paper's
    # own "blue pieces stay planar", and it is what makes the sculpture
    # cuttable from flat stock -- the spiral look comes from each blade
    # being turned further round the coil, never from bending one.
    good = flat_b < 1e-9
    ok &= good
    print(f"analytic: koman spiral blades exactly planar ({flat_b:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # ... and `twist` really does trade that away, which is why it is
    # off by default; asserting it keeps the docstring honest
    good = twist_b > 1e-2
    ok &= good
    print(f"analytic: koman spiral twist forfeits developability "
          f"({twist_b:.2e}) {'OK' if good else 'FAIL'}")

    # the coil is logarithmic: every blade sits at a constant ratio
    # further out than the last, which is the exponential shrink of h
    # the paper attributes to Koman
    gr = 0.05
    Vs, _ = build_koman_spiral(blades=30, growth=gr, samples=1,
                               blade_len=0.0, blade_width=0.0)
    # each blade emits 2*(samples+1) vertices, so this is its root
    rr = np.linalg.norm(Vs[::4, :2], axis=1)
    ratio = rr[1:] / np.maximum(rr[:-1], 1e-15)
    good = float(np.max(np.abs(ratio - (1.0 + gr)))) < 1e-9
    ok &= good
    print(f"analytic: koman spiral is logarithmic (ratio spread "
          f"{float(np.max(np.abs(ratio - (1 + gr)))):.2e}) "
          f"{'OK' if good else 'FAIL'}")

    # and unlike the wreath it does NOT close: it keeps winding, and
    # grows by a large factor doing it
    V2, F2 = build_koman_spiral()
    r2 = np.linalg.norm(V2[:, :2], axis=1)
    good = (len(F2) > 0 and np.all(np.isfinite(V2))
            and r2.max() / max(r2.min(), 1e-12) > 5.0)
    ok &= good
    print(f"analytic: koman spiral winds open (radius x"
          f"{r2.max()/max(r2.min(),1e-12):.0f}) {'OK' if good else 'FAIL'}")

    # the arc solver must invert sin(x)/x correctly -- it is what sets
    # how far each shortened panel bows
    worst = 0.0
    for ratio in (0.99, 0.9, 0.7, 0.5):
        th = _arc_angle(ratio, 1.0)
        worst = max(worst, abs(np.sin(th) / th - ratio))
    good = worst < 1e-9
    ok &= good
    print(f"analytic: arc chord/length inversion (worst {worst:.2e}) "
          f"{'OK' if good else 'FAIL'}")

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("dform.analytic self-test failed")
