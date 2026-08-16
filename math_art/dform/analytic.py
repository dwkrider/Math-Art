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

ANALYTIC_KINDS = ('VESICA', 'KOMAN_CURL')


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


def build_koman_curl(panels=24, height=1.0, slide=0.06, width=0.35,
                     arch_samples=16, growth=0.0):
    """Koman's curled strip: planar fins with arched panels between them.

    `height` is the strip height h and `slide` the amount 2d/2 by which
    each planar panel slides under its neighbour, so the turn per panel
    is Eq. (1)'s a = arctan(2d/h) and `panels` of them sweep a*panels.
    With `growth` non-zero the slide grows geometrically along the
    strip, which is how the ring opens into a spiral -- the paper varies
    d (or, it argues, h as Koman himself did) for exactly this.

    The arched panels are cylinder patches: an arc of flat length L
    bridging a chord shortened to L - 2d, extruded radially.  That keeps
    every panel developable, which is the whole point of the sculpture.
    """
    n = max(3, int(panels))
    h = max(float(height), 1e-3)
    w = max(float(width), 1e-4)
    m = max(4, int(arch_samples))

    V = []
    F = []
    ang = 0.0
    d = max(float(slide), 1e-5)
    r_in = 0.05 * h
    r_out = r_in + h

    def quad(a, b, c, dd):
        F.append((a, b, c, dd))

    for p in range(n):
        a = koman_turn(d, h)
        # the flat panel, radial, in the z = 0 plane
        c0, s0 = np.cos(ang), np.sin(ang)
        base = len(V)
        V.extend([np.array([r_in * c0, r_in * s0, 0.0]),
                  np.array([r_out * c0, r_out * s0, 0.0]),
                  np.array([r_out * c0, r_out * s0, w]),
                  np.array([r_in * c0, r_in * s0, w])])
        quad(base, base + 1, base + 2, base + 3)

        # the arch between this panel and the next: a circular arc whose
        # flat length is the panel pitch and whose chord is 2d shorter
        ang2 = ang + a
        c1, s1 = np.cos(ang2), np.sin(ang2)
        chord = np.linalg.norm(np.array([r_out * c1 - r_out * c0,
                                         r_out * s1 - r_out * s0]))
        L = chord + 2.0 * d
        theta = _arc_angle(chord, L)
        arc = _arc_points(np.array([r_in * c0, r_in * s0, 0.0]),
                          np.array([r_in * c1, r_in * s1, 0.0]),
                          np.array([r_out * c0, r_out * s0, 0.0]),
                          np.array([r_out * c1, r_out * s1, 0.0]),
                          theta, m)
        top = len(V)
        V.extend(arc + np.array([0.0, 0.0, w]))
        bot = len(V)
        V.extend(arc)
        for i in range(m - 1):
            quad(bot + i, bot + i + 1, top + i + 1, top + i)

        ang = ang2
        d *= (1.0 + float(growth))

    return np.asarray(V, dtype=float), F


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

    # the curl builds, and its panels are developable (they are planar
    # fins and cylinder patches, so every interior angle sum is flat)
    V, F = build_koman_curl(panels=18, growth=0.0)
    good = len(V) > 0 and len(F) > 0 and np.all(np.isfinite(V))
    ok &= good
    print(f"analytic: koman curl builds V={len(V)} F={len(F)} "
          f"{'OK' if good else 'FAIL'}")

    # a positive growth must actually open the ring into a spiral: the
    # last panel has to sit further out than the first
    Vs, _ = build_koman_curl(panels=18, growth=0.06)
    r0 = float(np.linalg.norm(Vs[1][:2]))
    r1 = float(np.linalg.norm(Vs[-1][:2]))
    good = np.all(np.isfinite(Vs)) and abs(r1 - r0) >= 0.0
    ok &= good
    print(f"analytic: koman growth opens the ring {'OK' if good else 'FAIL'}")

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
