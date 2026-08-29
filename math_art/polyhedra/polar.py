"""
polar.py -- polar zonohedra: the two-parameter family PZ(n, theta), its
level structure, its surface helices, and the design presets that pick a
pitch for a reason (pure Python; bpy never).

--------------------------------------------------------------------------
The family
--------------------------------------------------------------------------
Take n unit segments -- the "ribs" of a half-open umbrella -- spaced
equally in azimuth and all tilted by the same pitch theta above the
horizontal:

    g_i = (cos(th) cos(360 i/n), cos(th) sin(360 i/n), sin(th))

Their Minkowski sum is the polar zonohedron.  Two parameters, so a tightly
constrained family: theta near 0 is a pancake, theta near 90 a cigar, and
in between the shape of a U.S. football.  Every face is a rhombus, every
edge has the same length, and the two poles are n-fold vertices.

--------------------------------------------------------------------------
Vertices as contiguous rib sums
--------------------------------------------------------------------------
Hart's indexing makes the whole solid combinatorial.  Every vertex is the
sum of a CONTIGUOUS run of ribs, taken cyclically:

    V(i, L) = g_i + g_{i+1} + ... + g_{i+L-1}       (indices mod n)

with L = 0 the bottom pole (the empty sum) and L = n the top pole (every
rib).  There are n runs of each length 1..n-1, so V = n(n-1) + 2, matching
Euler against F = n(n-1) four-sided faces and E = 2F.

The face whose top corner is V(i, L) is

    [ V(i, L), V(i+1, L-1), V(i+1, L-2), V(i, L-1) ],   L = 2 .. n

and its two edge directions are g_i and g_{i+L-1}, an angular separation of
L-1 ribs.  So the faces come in n-1 LEVELS indexed m = L-1, all n faces of
a level congruent, with face angle

    f_m = arccos( cos(360 m/n) cos^2(th) + sin^2(th) ).

That level index is what everything downstream needs: truncating a dome,
colouring by level, cutting one flat template per distinct rhombus.  The
Antiprism port below (kept for the spirallohedra) does not expose it, which
is why this direct construction exists alongside it.

--------------------------------------------------------------------------
Surface helices
--------------------------------------------------------------------------
Leaving a pole along rib i and going "straight across" every 4-way vertex
walks the runs (i,1), (i,2), ... (i,n): n equal chords covering one
revolution, i.e. a discrete helix.  Fixing the run's END index instead of
its start gives the mirror family, so there are n right-handed and n
left-handed helices and every edge lies on exactly one of each.

--------------------------------------------------------------------------
Choosing the pitch
--------------------------------------------------------------------------
Volume is proportional to cos^2(th) sin(th), maximised at th = arctan(1/V2)
~ 35.264 degrees for EVERY n -- the ribs are then a eutactic star, an
orthogonal projection of the n-dimensional coordinate basis, so the solid
is a shadow of the n-cube.  Surface area has no closed form and its optimum
drifts with n (35.26 at n = 3 down towards 33.5 for large n).  Setting two
face angles supplementary makes two levels congruent; at n = 5 that single
condition makes all twenty faces golden rhombi, which is Fedorov's rhombic
icosahedron.

--------------------------------------------------------------------------
Angle convention
--------------------------------------------------------------------------
theta here is Hart's: measured from the HORIZONTAL.  The Blender operator
has long measured its pitch from the AXIS, so pitch = 90 - theta; the
conversion lives in `theta_of` and nowhere else.

References:
- George W. Hart, "The Joy of Polar Zonohedra", Bridges 2021, pp. 7-14
  (counts, surface helices, and the Appendix formulas for face angles,
  dihedrals, height, radius and vertex coordinates used here).
- B. Chilton and H. S. M. Coxeter, "Polar Zonohedra", American
  Mathematical Monthly 70(9) (1963), pp. 946-951.
- C. H. H. Franklin, "Hypersolid Concepts and the Completeness of Things
  and Phenomena", Mathematical Gazette 21 (1937), pp. 360-364 -- polar
  zonohedra as shadows of n-dimensional hypercubes.
- H. S. M. Coxeter, "Regular Polytopes", 3rd ed., Dover (1973), ch. 2.
- E. S. Fedorov (1885); M. Senechal and R. V. Galiulin, "An Introduction
  to the Theory of Figures: The Geometry of E. S. Federov", Structural
  Topology 10 (1984), pp. 5-22 -- the rhombic icosahedron.
- Russell Towle, "Polar Zonohedra", Mathematica Journal 6(2) (1996),
  pp. 8-12; "Rhombic Spirallohedra" (2003).
- Antiprism (Adrian Rossiter), `base/zonohedron.cc` -- the
  `make_polar_zonohedron` port at the foot of this module.
"""

import math
from math import cos, sin, pi, gcd

EPS = 1e-12


# --------------------------------------------------------------------------
# angle convention
# --------------------------------------------------------------------------

def theta_of(pitch, ref='AXIS'):
    """Hart's theta (degrees above the horizontal) from the operator's dial.

    The generator's `pitch` has always been measured from the AXIS, while
    every published formula measures from the horizontal.  One conversion,
    in one place, so the presets can be written the way the papers state
    them.
    """
    return (90.0 - pitch) if ref == 'AXIS' else pitch


def pitch_of(theta, ref='AXIS'):
    """The inverse of `theta_of` -- a preset's answer put back on the dial."""
    return (90.0 - theta) if ref == 'AXIS' else theta


# --------------------------------------------------------------------------
# the star
# --------------------------------------------------------------------------

def ribs(n, theta, ellipticity=1.0, length=1.0):
    """The n umbrella ribs, Hart eq. (1).

    The azimuth runs the opposite way round from Hart's statement (note
    the minus on Y).  That is a mirror image, and it is deliberate: the
    generator's long-standing `polar_star` winds this way, and
    `ifs/spacefill.py` builds its spirallohedra from it, so matching the
    sign keeps one handedness across the whole add-on.  Nothing measured
    here depends on the choice -- face angles, dihedrals, height, radius,
    area and volume are all functions of the angle BETWEEN ribs -- and
    only the chirality of the surface helices and of the spirallohedra
    flips, which is a mirroring, not a different solid.

    `ellipticity` scales X only.  That is a linear map, so it keeps every
    face planar and every pair of opposite edges parallel -- the rhombi
    merely open into general parallelograms and the footprint becomes an
    ellipse.  It is Hart's third design parameter.
    """
    t = math.radians(theta)
    out = []
    for k in range(n):
        a = 2.0 * pi * k / n
        out.append([length * ellipticity * cos(t) * cos(a),
                    -length * cos(t) * sin(a),
                    length * sin(t)])
    return out


def axis_star_pitch(n, theta, ellipticity=1.0):
    """The same ribs in the operator's axis-pitch convention, kept so the
    general zonotope path and the polar path agree vector for vector."""
    return ribs(n, theta, ellipticity)


# --------------------------------------------------------------------------
# closed-form measurements (Hart, Appendix)
# --------------------------------------------------------------------------

def face_angle(n, theta, m):
    """Face angle of the rhombi at level m (1 .. n-1), in degrees."""
    t = math.radians(theta)
    c = cos(2.0 * pi * m / n) * cos(t) ** 2 + sin(t) ** 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def face_angles(n, theta):
    """Every level's face angle, level 1 first."""
    return [face_angle(n, theta, m) for m in range(1, n)]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _unit(a):
    l = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    return [x / l for x in a] if l > EPS else [0.0, 0.0, 0.0]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def dihedral(n, theta, m, ellipticity=1.0):
    """Dihedral angle along the level-m edges, in degrees.

    Hart eqs. (3) and (4).  The rib indices differ at level 1, because the
    two faces meeting there are the pole faces rather than a pair further
    up the same zone -- tracing where the zones cross is what fixes the
    pattern, and getting it wrong silently returns a plausible number.
    """
    g = ribs(n, theta, ellipticity)
    if m <= 1:
        n1 = _unit(_cross(g[1 % n], g[2 % n]))
        n2 = _unit(_cross(g[2 % n], g[3 % n]))
    else:
        n1 = _unit(_cross(g[1 % n], g[m % n]))
        n2 = _unit(_cross(g[1 % n], g[(m + 1) % n]))
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(n1, n2)))))


def dihedrals(n, theta, ellipticity=1.0):
    return [dihedral(n, theta, m, ellipticity) for m in range(1, n)]


def height(n, theta):
    """Pole-to-pole distance for unit ribs: every edge rises sin(theta)."""
    return n * math.sin(math.radians(theta))


def max_radius(n, theta):
    """Widest horizontal radius, for unit ribs.

    A vertex is a run of L ribs, and the horizontal part of that sum is L
    chords of length cos(theta) turning by 360/n each -- an arc of a
    regular n-gon -- so its radius is

        cos(theta) sin(pi L / n) / sin(pi / n),

    maximised at L = floor(n/2).  For even n that is sin(pi/2) = 1 and the
    expression collapses to Hart's cos(theta)/sin(180/n); for odd n no
    vertex reaches the equator and the true maximum is smaller by
    cos(pi/2n), which matters at small n (about 2.5% at n = 7).
    """
    L = n // 2
    return (math.cos(math.radians(theta)) * math.sin(pi * L / n)
            / math.sin(pi / n))


def distinct_shapes(n):
    """How many rhombus shapes a generic PZ(n, .) needs cutting.

    Level m and level n-m are congruent, so an odd n has every shape twice
    and an even n has one unique middle level.
    """
    return n // 2


# --------------------------------------------------------------------------
# pitch presets
# --------------------------------------------------------------------------

#: theta = arctan(1/sqrt(2)); the ribs become a eutactic star and the solid
#: a shadow of the n-cube.  Volume-optimal for every n.
THETA_VOLUME = math.degrees(math.atan(1.0 / math.sqrt(2.0)))


def _area(n, theta):
    """Total surface area for unit ribs: n rhombi per level, each |g0 x gm|."""
    g = ribs(n, theta)
    tot = 0.0
    for m in range(1, n):
        c = _cross(g[0], g[m])
        tot += math.sqrt(_dot(c, c))
    return n * tot


def theta_area_optimal(n, lo=0.5, hi=89.5, iters=200):
    """The area-maximising pitch, by golden-section search.

    Hart reports no analytic result and that the optimum drifts with n --
    35.26 degrees at n = 3, towards about 33.5 for large n -- so this is a
    search, not a formula.  The area is unimodal in theta on (0, 90).
    """
    r = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = _area(n, c), _area(n, d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - r * (b - a)
            fc = _area(n, c)
        else:
            a, c, fc = c, d, fd
            d = a + r * (b - a)
            fd = _area(n, d)
        if b - a < 1e-10:
            break
    return 0.5 * (a + b)


def theta_congruent(n, a=1, b=2, iters=200):
    """The pitch that makes levels `a` and `b` congruent.

    Two rhombi with supplementary vertex angles are the same rhombus turned
    a quarter way round, so setting f_a + f_b = 180 degrees merges two
    levels into one part number.  At theta = 0 the sum is 360(a+b)/n and it
    falls monotonically to 0 at theta = 90, so a root exists exactly when
    a + b > n/2, and bisection finds it.

    n = 5 with (a, b) = (1, 2) is the celebrated case: all twenty faces
    become golden rhombi and the solid is Fedorov's rhombic icosahedron.
    """
    if not (1 <= a < n and 1 <= b < n):
        raise ValueError("congruent levels must lie in 1..n-1")
    if a == b:
        raise ValueError("congruent levels must be two different levels")
    if 2 * (a + b) <= n:
        raise ValueError(
            "levels %d and %d cannot be made congruent on a %d-fold star "
            "(needs a + b > n/2)" % (a, b, n))

    def f(th):
        return face_angle(n, th, a) + face_angle(n, th, b) - 180.0

    lo, hi = 1e-6, 90.0 - 1e-6
    if f(lo) < 0:
        raise ValueError("no congruent pitch for those levels")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


PITCH_PRESETS = ('CUSTOM', 'VOLUME', 'AREA', 'SQUARE45', 'CONGRUENT')


def preset_theta(preset, n, levels=(1, 2), fallback=35.264):
    """Hart's theta for a named preset, in degrees above the horizontal."""
    if preset == 'VOLUME':
        return THETA_VOLUME
    if preset == 'AREA':
        return theta_area_optimal(n)
    if preset == 'SQUARE45':
        return 45.0
    if preset == 'CONGRUENT':
        return theta_congruent(n, levels[0], levels[1])
    return fallback


# --------------------------------------------------------------------------
# the solid, built directly from the contiguous-run indexing
# --------------------------------------------------------------------------

def _run_sums(star):
    """V(i, L) for L = 0..n, as a dict keyed (i, L).

    L = 0 is the empty sum (the bottom pole) and L = n every rib (the top
    pole), both independent of i, so they collapse to one point each.
    """
    n = len(star)
    tot = [0.0, 0.0, 0.0]
    for v in star:
        tot = [tot[c] + v[c] for c in range(3)]
    pts = {}
    for i in range(n):
        pts[(i, 0)] = [0.0, 0.0, 0.0]
        pts[(i, n)] = list(tot)
        acc = [0.0, 0.0, 0.0]
        for L in range(1, n):
            g = star[(i + L - 1) % n]
            acc = [acc[c] + g[c] for c in range(3)]
            pts[(i, L)] = list(acc)
    return pts, tot


def polar_solid(star, levels=0, cap_rim=False):
    """(V, F, level_per_face) for the polar zonohedron of `star`.

    `levels` truncates: keep only faces up to that level, leaving a dome
    (0 or >= n-1 means the whole solid).  `cap_rim` closes the resulting
    zig-zag rim with a fan, which is what turns a dome into a bell.

    Faces are wound so their normals point away from the centre; the
    centre of a zonotope is half the sum of its generators.
    """
    n = len(star)
    if n < 3:
        raise ValueError("a polar zonohedron needs at least 3 ribs")
    top = n - 1 if (levels <= 0 or levels >= n - 1) else int(levels)

    pts, tot = _run_sums(star)
    centre = [0.5 * c for c in tot]

    V = [[0.0, 0.0, 0.0], list(tot)]        # 0 = bottom pole, 1 = top pole
    idx = {}

    def vid(i, L):
        if L <= 0:
            return 0
        if L >= n:
            return 1
        key = (i % n, L)
        j = idx.get(key)
        if j is None:
            j = len(V)
            idx[key] = j
            V.append(list(pts[key]))
        return j

    F, level = [], []
    for L in range(2, top + 2):
        for i in range(n):
            f = [vid(i, L), vid(i + 1, L - 1), vid(i + 1, L - 2), vid(i, L - 1)]
            F.append(_outward(V, f, centre))
            level.append(L - 1)

    if top < n - 1 and cap_rim:
        # the free boundary is the zig-zag through the two longest runs
        rim = []
        for i in range(n):
            rim.append(vid(i, top + 1))
            rim.append(vid(i + 1, top))
        c = [sum(V[k][d] for k in rim) / len(rim) for d in range(3)]
        cid = len(V)
        V.append(c)
        m = len(rim)
        for k in range(m):
            f = [cid, rim[k], rim[(k + 1) % m]]
            # the cap looks the other way from the shell it closes
            F.append(_outward(V, f, centre, inward=True))
            level.append(0)
    return V, F, level


def _outward(V, f, centre, inward=False):
    """Wind a face so its normal points away from (or towards) `centre`."""
    a, b, c = V[f[0]], V[f[1]], V[f[2]]
    nrm = _cross([b[i] - a[i] for i in range(3)],
                 [c[i] - a[i] for i in range(3)])
    mid = [sum(V[k][i] for k in f) / len(f) - centre[i] for i in range(3)]
    d = _dot(nrm, mid)
    if (d < 0) != bool(inward):
        return list(reversed(f))
    return list(f)


def face_zone_pair(n, i, L):
    """The two ribs spanning the face whose top corner is V(i, L)."""
    return (i % n, (i + L - 1) % n)


def zone_pairs(n, levels=0):
    """The spanning rib pair per face, in the order `polar_solid` emits."""
    top = n - 1 if (levels <= 0 or levels >= n - 1) else int(levels)
    out = []
    for L in range(2, top + 2):
        for i in range(n):
            out.append(face_zone_pair(n, i, L))
    return out


# --------------------------------------------------------------------------
# surface helices
# --------------------------------------------------------------------------

def helix_runs(n, handed='CW'):
    """The n pole-to-pole edge paths of one handedness, as (i, L) runs.

    Holding the run's START index fixed and growing it walks the ribs in
    cyclic order; holding its END index fixed walks them the other way.
    Those are the two mirror families -- every edge lies on exactly one
    path of each, except the 2n polar edges, which lie on one of each.
    """
    out = []
    for i in range(n):
        if handed == 'CW':
            out.append([(i, L) for L in range(0, n + 1)])
        else:
            out.append([((i - L + 1) % n, L) for L in range(0, n + 1)])
    return out


def helix_points(star, handed='CW'):
    """Each surface helix as a polyline of points."""
    n = len(star)
    pts, _tot = _run_sums(star)
    out = []
    for run in helix_runs(n, handed):
        out.append([list(pts[(i % n, L)]) for i, L in run])
    return out


def extruded_helices(star, depth=1.0, handed='CW'):
    """Webster's PZ_E(n, theta, d): one handedness of helix, extruded down
    the polar axis into a wall of parallelograms.

    `depth` is in edge lengths, so d = 1 makes the extruded faces rhombi
    and the shape stays independent of overall scale.  The n walls spiral
    round the axis like the arms of a galaxy; large n needs a small d or
    neighbouring walls collide.
    """
    V, F = [], []
    for chain in helix_points(star, handed):
        base = len(V)
        m = len(chain)
        for p in chain:
            V.append(list(p))
        for p in chain:
            V.append([p[0], p[1], p[2] - depth])
        for k in range(m - 1):
            F.append([base + k, base + k + 1,
                      base + m + k + 1, base + m + k])
    return V, F


def helix_collides(n, theta, depth):
    """Does a wall of this depth reach its neighbour?

    Consecutive helices are one rib apart, so the vertical gap between a
    wall and the next is the rise of one edge, sin(theta).  Extruding
    further than that overlaps -- which Webster notes happens quickly as n
    grows, though he treats it as an aesthetic call rather than an error.
    """
    return depth > math.sin(math.radians(theta)) + 1e-9


# --------------------------------------------------------------------------
# flat templates for cutting
# --------------------------------------------------------------------------

def template_rhombi(n, theta, side=1.0):
    """One flat outline per DISTINCT rhombus, laid out in a row.

    Levels m and n-m are congruent, so cutting stops at floor(n/2).  Each
    outline is the rhombus with that level's face angle, on unit (or
    `side`) edges: the parts list for a paper, card or plywood model.
    Returns (polygons, angles, levels).
    """
    polys, angs, lvls = [], [], []
    x = 0.0
    for m in range(1, distinct_shapes(n) + 1):
        a = math.radians(face_angle(n, theta, m))
        dx, dy = side * math.cos(a), side * math.sin(a)
        quad = [(x, 0.0), (x + side, 0.0),
                (x + side + dx, dy), (x + dx, dy)]
        polys.append(quad)
        angs.append(math.degrees(a))
        lvls.append(m)
        x += side + max(dx, 0.0) + 0.6 * side
    return polys, angs, lvls


def template_chain(n, theta, side=1.0):
    """Hart's unfolded chain: the n-1 rhombi of one meridian, joined edge
    to edge in level order.

    Walking the cumulative face angles lays the levels out as a strip that
    folds straight back into a pole-to-pole band of the solid, so it is the
    piece a builder actually tapes together first.
    """
    xs, ys = [0.0], [0.0]
    for m in range(1, n):
        a = math.radians(face_angle(n, theta, m))
        xs.append(xs[-1] + side * math.cos(a))
        ys.append(ys[-1] + side * math.sin(a))
    quads = []
    for k in range(1, n):
        quads.append([(xs[k - 1], ys[k - 1]), (xs[k - 1] + side, ys[k - 1]),
                      (xs[k] + side, ys[k]), (xs[k], ys[k])])
    return quads


# --------------------------------------------------------------------------
# fused lobes: the complete polar zonohedron
# --------------------------------------------------------------------------

def lobe_offsets(star, lobes):
    """Translations stacking `lobes` copies pole onto pole.

    A copy shifted by the full rib sum sits with its bottom pole on the
    previous one's top pole, and because the surface helices of the two
    copies are congruent their edges line up along the seam.  Repeating
    gives a finite piece of what Hart calls the complete polar zonohedron
    -- an infinite chain of lobes, the discrete analogue of a full double
    cone rather than a single dunce cap.
    """
    tot = [0.0, 0.0, 0.0]
    for v in star:
        tot = [tot[c] + v[c] for c in range(3)]
    return [[tot[c] * k for c in range(3)] for k in range(max(1, int(lobes)))]


# --------------------------------------------------------------------------
# Antiprism's polar zonohedron / spirallohedron port
# --------------------------------------------------------------------------
# A direct translation of `make_polar_zonohedron` from Antiprism's
# base/zonohedron.cc (Adrian Rossiter).  Kept because it -- and not the
# level construction above -- is what produces Russell Towle's rhombic
# spirallohedra, and what handles a star taken with a STEP (Antiprism's
# `zono -P n/d`), where a step sharing a factor with n splits the result
# into gcd(n, d) separate lobes.

def _pos_mod(a, b):
    return a % b if b else 0


def _get_idx(P, s, s_step, num_spirals, i, j, V):
    if i < 0:
        if j == 0:
            idx = -2
        elif j > P - s_step:
            idx = ((_pos_mod(s - 1, num_spirals) + 1)
                   * (P - s_step) * s_step) + j - P
        else:
            idx = (_pos_mod(s - 1, num_spirals) * (P - s_step) * s_step) \
                + (j - 1) * s_step
    elif j == s_step:
        if i == P - s_step - 1:
            idx = -1
        elif i >= P - 2 * s_step:
            idx = (_pos_mod(s - 1, num_spirals) * (P - s_step) * s_step) \
                + (P - s_step) * s_step + i - (P - s_step - 1)
        else:
            idx = (_pos_mod(s - 1, num_spirals) * (P - s_step) * s_step) \
                + (i + j) * s_step
    else:
        idx = (s * (P - s_step) * s_step) + (i * s_step) + j
    return V + idx + 2


def _get_face(P, s, s_step, num_spirals, i, j, V):
    return [_get_idx(P, s, s_step, num_spirals, i - 1, j, V),
            _get_idx(P, s, s_step, num_spirals, i, j, V),
            _get_idx(P, s, s_step, num_spirals, i, j + 1, V),
            _get_idx(P, s, s_step, num_spirals, i - 1, j + 1, V)]


def make_polar_zonohedron(star, step=1, spiral_step=0):
    """Antiprism's construction.  Returns (verts, faces).

    `step` is the offset polygon of `zono -P n/d`; `spiral_step` = 0 gives
    the polar zonohedron and a nonzero value a rhombic spirallohedron of
    that spiral width.
    """
    verts = []
    faces = []
    N = len(star)
    D = step
    num_parts = gcd(N, D)
    P = N // num_parts
    P_spiral_step = 1 if spiral_step == 0 else \
        _pos_mod(spiral_step // num_parts, P)
    if P_spiral_step == 0:
        raise ValueError("invalid spiral width for this star")
    P_num_spirals = P // gcd(P, P_spiral_step)

    for p in range(num_parts):
        V = len(verts)
        star_part = [star[(p * P // P_num_spirals + i * D) % N]
                     for i in range(P)]
        verts.append([0.0, 0.0, 0.0])       # initial point
        verts.append([0.0, 0.0, 0.0])       # final point, set later
        A = [0.0, 0.0, 0.0]
        B = [0.0, 0.0, 0.0]
        for s in range(P_num_spirals):
            A = [0.0, 0.0, 0.0]
            for i in range(P - P_spiral_step):
                i_idx = (s * P_spiral_step + i) % P
                A = [A[c] + star_part[i_idx][c] for c in range(3)]
                B = [0.0, 0.0, 0.0]
                for j in range(P_spiral_step):
                    verts.append([A[c] + B[c] for c in range(3)])
                    j_idx = _pos_mod((s - 1) * P_spiral_step + j, P)
                    B = [B[c] + star_part[j_idx][c] for c in range(3)]
                    faces.append(_get_face(P, s, P_spiral_step,
                                           P_num_spirals, i, j, V))
        verts[V + 1] = [A[c] + B[c] for c in range(3)]
    return verts, faces


def polar_parts(n, step):
    """How many separate lobes `zono -P n/step` produces."""
    return gcd(n, max(1, int(step)))


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _selftest():
    def close(a, b, t=1e-7):
        return abs(a - b) <= t * max(1.0, abs(a), abs(b))

    # --- counts, Euler, and the level structure ------------------------
    for n in (3, 5, 8, 12):
        st = ribs(n, 40.0)
        V, F, lev = polar_solid(st)
        assert len(F) == n * (n - 1), (n, len(F))
        assert len(V) == n * (n - 1) + 2, (n, len(V))
        assert all(len(f) == 4 for f in F)
        E = set()
        for f in F:
            for k in range(4):
                a, b = f[k], f[(k + 1) % 4]
                E.add((min(a, b), max(a, b)))
        assert len(E) == 2 * len(F), (n, len(E))
        assert len(V) - len(E) + len(F) == 2, (n, len(V), len(E), len(F))
        # n faces per level, and each face's angle is its level's angle
        for m in range(1, n):
            assert lev.count(m) == n, (n, m, lev.count(m))

    # every face is a rhombus of unit edge, at its level's face angle
    n, th = 9, 37.0
    st = ribs(n, th)
    V, F, lev = polar_solid(st)
    for f, m in zip(F, lev):
        e = [[V[f[(k + 1) % 4]][c] - V[f[k]][c] for c in range(3)]
             for k in range(4)]
        for x in e:
            assert close(math.sqrt(_dot(x, x)), 1.0, 1e-9)
        u, w = _unit(e[0]), _unit([-c for c in e[3]])
        ang = math.degrees(math.acos(max(-1.0, min(1.0, _dot(u, w)))))
        want = face_angle(n, th, m)
        assert close(min(ang, 180.0 - ang),
                     min(want, 180.0 - want), 1e-7), (m, ang, want)

    # --- closed forms --------------------------------------------------
    for n, th in ((7, 33.0), (12, 55.0)):
        st = ribs(n, th)
        V, _F, _l = polar_solid(st)
        hi = max(v[2] for v in V) - min(v[2] for v in V)
        assert close(hi, height(n, th), 1e-9), (n, hi, height(n, th))
        rad = max(math.hypot(v[0], v[1]) for v in V)
        assert close(rad, max_radius(n, th), 1e-9), (n, rad)

    # dihedrals measured off the mesh must match Hart eqs. (3)-(4)
    n, th = 8, 42.0
    st = ribs(n, th)
    V, F, lev = polar_solid(st)
    nrm = []
    for f in F:
        a, b, c = V[f[0]], V[f[1]], V[f[2]]
        nrm.append(_unit(_cross([b[i] - a[i] for i in range(3)],
                                [c[i] - a[i] for i in range(3)])))
    edge_faces = {}
    for fi, f in enumerate(F):
        for k in range(4):
            key = (min(f[k], f[(k + 1) % 4]), max(f[k], f[(k + 1) % 4]))
            edge_faces.setdefault(key, []).append(fi)
    # Hart's dihedral_1 is the angle across a POLAR edge, where two faces
    # of the same level 1 meet; dihedral_m for m > 1 is the angle across an
    # edge separating level m-1 from level m.  Grouping the mesh's edges by
    # the level pair of their two faces has to reproduce exactly that.
    seen = {}
    for key, fs in edge_faces.items():
        if len(fs) != 2:
            continue
        d = math.degrees(math.acos(max(-1.0, min(1.0,
                                                 _dot(nrm[fs[0]], nrm[fs[1]])))))
        pair = tuple(sorted((lev[fs[0]], lev[fs[1]])))
        seen.setdefault(pair, set()).add(round(d, 6))
    for m in range(1, n):
        want = dihedral(n, th, m)
        pair = (1, 1) if m == 1 else (m - 1, m)
        got = seen.get(pair, ())
        assert len(got) == 1 and abs(list(got)[0] - want) < 1e-5, \
            (m, pair, want, sorted(got))
    # the top pole mirrors the bottom, so its own same-level edges repeat
    # dihedral_1 rather than introducing a new value
    top = seen.get((n - 1, n - 1), ())
    assert len(top) == 1 and abs(list(top)[0] - dihedral(n, th, 1)) < 1e-5, top

    # --- presets -------------------------------------------------------
    assert close(THETA_VOLUME, 35.26438968, 1e-8), THETA_VOLUME
    # n = 3 at the volume-optimal pitch is a cube: all edges 1, all
    # dihedrals 90
    st = ribs(3, THETA_VOLUME)
    V, F, _l = polar_solid(st)
    assert (len(V), len(F)) == (8, 6), (len(V), len(F))
    for m in (1, 2):
        assert close(face_angle(3, THETA_VOLUME, m), 90.0, 1e-8)
    # n = 4 gives the rhombic dodecahedron
    V, F, _l = polar_solid(ribs(4, THETA_VOLUME))
    assert (len(V), len(F)) == (14, 12), (len(V), len(F))
    # ... and n a multiple of 3 has square faces at levels n/3 and 2n/3
    for n in (6, 9, 12):
        assert close(face_angle(n, THETA_VOLUME, n // 3), 90.0, 1e-8), n
        assert close(face_angle(n, THETA_VOLUME, 2 * n // 3), 90.0, 1e-8), n

    # area-optimal: 35.26 at n = 3, drifting towards ~33.5 (Hart)
    assert close(theta_area_optimal(3), THETA_VOLUME, 1e-6), \
        theta_area_optimal(3)
    big = theta_area_optimal(48)
    assert 33.3 < big < 33.8, big
    assert theta_area_optimal(48) < theta_area_optimal(8) < theta_area_optimal(3)

    # congruent levels at n = 5: Fedorov's rhombic icosahedron, twenty
    # golden rhombi (acute angle arccos(1/sqrt(5)) = 63.4349 degrees)
    th5 = theta_congruent(5, 1, 2)
    golden = math.degrees(math.acos(1.0 / math.sqrt(5.0)))
    a1, a2 = face_angle(5, th5, 1), face_angle(5, th5, 2)
    assert close(min(a1, 180 - a1), golden, 1e-7), (a1, golden)
    assert close(min(a2, 180 - a2), golden, 1e-7), (a2, golden)
    V, F, _l = polar_solid(ribs(5, th5))
    assert len(F) == 20, len(F)
    # an impossible request must be refused, not silently approximated
    try:
        theta_congruent(12, 1, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("theta_congruent accepted a + b <= n/2")

    # --- truncation ----------------------------------------------------
    n = 10
    for k in (1, 3, 6):
        V, F, lev = polar_solid(ribs(n, 40.0), levels=k)
        assert len(F) == n * k, (k, len(F))
        assert max(lev) == k, (k, max(lev))
        V2, F2, _l2 = polar_solid(ribs(n, 40.0), levels=k, cap_rim=True)
        assert len(F2) == n * k + 2 * n, (k, len(F2))

    # --- helices -------------------------------------------------------
    n = 7
    st = ribs(n, 40.0)
    for handed in ('CW', 'CCW'):
        chains = helix_points(st, handed)
        assert len(chains) == n
        for ch in chains:
            assert len(ch) == n + 1
            # every step is one unit edge, and each path runs pole to pole
            for k in range(n):
                d = [ch[k + 1][c] - ch[k][c] for c in range(3)]
                assert close(math.sqrt(_dot(d, d)), 1.0, 1e-9)
            assert close(ch[0][2], 0.0, 1e-9)
            assert close(ch[-1][2], height(n, 40.0), 1e-9)
        # The projection of a helix onto the XY plane is a regular n-gon --
        # but one centred on its own centroid, NOT on the polar axis, since
        # the path spirals up one side of the solid.  Measuring radii from
        # the origin is the easy mistake and gives six different numbers.
        proj = [(p[0], p[1]) for p in chains[0][:-1]]
        cx = sum(p[0] for p in proj) / len(proj)
        cy = sum(p[1] for p in proj) / len(proj)
        r = {round(math.hypot(p[0] - cx, p[1] - cy), 9) for p in proj}
        assert len(r) == 1, sorted(r)
        # and its circumradius is that of a regular n-gon of side cos(theta)
        want = math.cos(math.radians(40.0)) / (2 * math.sin(pi / n))
        assert close(list(r)[0], want, 1e-9), (list(r)[0], want)
    # the two families together use every edge, the polar ones twice
    V, F, _l = polar_solid(st)
    E = set()
    for f in F:
        for k in range(4):
            E.add((min(f[k], f[(k + 1) % 4]), max(f[k], f[(k + 1) % 4])))
    used = 0
    for handed in ('CW', 'CCW'):
        for run in helix_runs(n, handed):
            used += len(run) - 1
    assert used == 2 * n * n, used
    assert len(E) == 2 * n * (n - 1)

    # --- ellipticity keeps the faces planar ----------------------------
    st = ribs(9, 40.0, ellipticity=1.7)
    V, F, _l = polar_solid(st)
    for f in F:
        a, b, c, d = (V[i] for i in f)
        nr = _cross([b[i] - a[i] for i in range(3)],
                    [c[i] - a[i] for i in range(3)])
        assert abs(_dot(nr, [d[i] - a[i] for i in range(3)])) < 1e-9

    # --- templates -----------------------------------------------------
    n, th = 11, 38.0
    polys, angs, lvls = template_rhombi(n, th)
    assert len(polys) == distinct_shapes(n) == 5, (len(polys),)
    for quad, m in zip(polys, lvls):
        for k in range(4):
            p, q = quad[k], quad[(k + 1) % 4]
            assert close(math.hypot(q[0] - p[0], q[1] - p[1]), 1.0, 1e-9)
    assert len(template_chain(n, th)) == n - 1

    # --- the Antiprism port still agrees on the plain solid ------------
    for n in (5, 9, 12):
        st = ribs(n, 40.0)
        Vp, Fp = make_polar_zonohedron(st, 1, 0)
        assert len(Fp) == n * (n - 1), (n, len(Fp))
    # a step sharing a factor with n splits the model into that many lobes
    assert polar_parts(12, 4) == 4
    Vp, Fp = make_polar_zonohedron(ribs(12, 40.0), 4, 0)
    assert len(Fp) == 4 * 3 * 2, len(Fp)

    print("RESULT: OK")
