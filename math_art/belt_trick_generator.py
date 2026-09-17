# Dirac Belt Trick generator for Blender
#
# A cube spins about a fixed axis with a belt fastened to each of its
# six faces, the far end of every belt pinned to a surrounding cage.
# The belts twist and tangle as the cube turns, never pass through one
# another or through themselves, and after TWO full turns (720
# degrees) every belt is back exactly where it started -- after one
# turn it is not.  That is the belt trick: pi_1(SO(3)) = Z/2, so the
# loop of a single turn is stuck but the loop of a double turn can be
# contracted, and the belts are what carry the contraction.  Drive
# "Turn" from 0 to 720 degrees for one cycle.
#
# WHY THE BELT CANNOT BE A FUNCTION OF THE CUBE'S ORIENTATION.  If it
# were, the belt would be the same at 360 degrees as at 0, since the
# cube is; but a belt that returns after one turn would contract the
# single-turn loop, which is impossible.  So the belt state must depend
# on the turn parameter itself, with period 4 pi, and the untangling
# must be driven by the same parameter as the winding.
#
# THE FRAME FIELD.  Attach a frame to each point of a belt.  A belt is
# then a path s -> F(s,t) in SO(3), s = 0 at the cage (F = identity)
# and s = 1 at the cube (F = the cube's rotation).  Lifted to unit
# quaternions, with psi = turn / 2 in [0, 2 pi]:
#
#   q_T(rho, psi) = ( cos^2 rho + sin^2 rho cos psi,
#                     sin rho cos rho (1 - cos psi) m,
#                     sin rho sin psi n ),        rho = (pi/2) g(s),
#
# n the spin axis and m a fixed unit vector perpendicular to it.  This
# is Hanson's deformation 12.3.2 (credited to Hart, Francis and
# Kauffman) read with the roles of its two parameters swapped, which
# Pengelley and Ramras point out is exactly the candle-dance reading:
# time is the loop parameter, position along the belt is the stage.
# Geometrically the loop psi -> q_T is, at each rho, a circle on the
# 2-sphere spanned by 1, m and n, all these circles tangent to one
# another at the identity; at rho = pi/2 it is the great circle
# through 1 and n, i.e. the double turn about n, and at rho = 0 it is
# the single point 1.  So every belt frame goes round a loop of
# period 2 pi in psi (4 pi in turn), the cube end goes round the
# double turn, and the cage end stays put.  At psi = pi the path
# s -> q_T is a full 2 pi rotation about m: the two belts along m are
# straight with a full twist, the other four are spirals coiled once
# round the cube in the plane perpendicular to m.
#
# The field actually used is q_T conjugated by a rotation about n
# through psi,  q = R_n(psi) q_T R_n(-psi):  the axis m precesses at
# half the cube's rate, as the arm of Holroyd's spinor linkage does.
# Conjugation changes neither boundary (R_n commutes with the cube's
# rotation and fixes 1) nor the period, but it makes the bending axis
# of every belt where it meets the cube a CONSTANT direction (m) in
# the cube's own frame.  Without it the direction the polar belts bend
# in rotates a full turn per cycle relative to the face, and no
# periodic strip can be glued to a face under that motion.
#
# NESTED SPHERES.  The film-makers' construction (Hanson 12.3, the
# "onion" of glass spheres): the point of belt i at parameter s is
#
#   p_i(s,t) = r(s) F(s,t) u_i,
#
# u_i the face normal, r(s) decreasing from the cage radius to the
# cube.  The stage profile g(s) climbs at a capped slope with smooth
# ramps at both ends, because the twist a belt carries at 360 degrees
# is a full turn spread over the untangling zone, and its peak rate is
# the peak of g': a steeper profile makes a tighter helicoid, whose
# facets crease.  ALL SIX belts use the SAME field F, and that is the whole
# reason they never collide: at each s the six belt slices are the six
# arcs at the axis directions of ONE sphere, rigidly rotated, so they
# stay 90 degrees apart on that sphere, and slices at different s lie
# on different spheres.  The ribbon's width is drawn as an arc of that
# same sphere -- a chord of exactly the belt width -- so the argument
# applies to the whole strip, not just its centre line, and a slice
# whose chord subtends 2 lambda leaves the neighbouring arcs a gap of
# 90 - 2 lambda degrees.  The operator reports that gap at the cube,
# where it is smallest.  Holroyd remarks that "arbitrarily many belts
# in arbitrary directions are possible": this is why.
#
# ANY PLATONIC SOLID.  Nothing above used the cube except its face
# normals, so the centre can be any of the five solids (or just two
# opposite faces of the cube, the classic single belt held at both
# ends).  What the cube did fix was the width direction of each belt at
# its face: the conjugated field bends every belt, whatever its face,
# about the SAME body axis m where it meets the solid, so a belt on the
# face with normal u takes its width along m projected into that face,
# and a belt whose normal is along m (which only twists there) takes
# any in-plane direction.  The packing bound generalises too: arcs
# centred on the face normals, at their angular spacing, each of half
# angle lambda -- the operator solves for the widest belt whose arcs
# clear one another by a margin over the thickness and clamps to it,
# saying so.  Twenty faces at 41.8 degrees apart leave far less room
# than six at 90.
#
# The spheres keep the belts off each other, not off the solid: its
# corners reach past the innermost spheres (a cube's to 1.73 times the
# face distance, a tetrahedron's to 3 times), and a belt coiling round
# at those radii would pass through them.  So each belt stays STRAIGHT
# and rigid with the solid out to just beyond its circumradius, and the
# untangling zone begins there; a radial segment from a face centre
# never re-enters a convex solid.  It also leaves each face exactly
# orthogonally.  Beyond the straight run the profile g rises from 1 to
# 0 with g' = 0 at both ends, so the belt departs from straight without
# a kink.  For a solid the last curved slice sits on the sphere through
# the face's chord and a short flat stub in the solid's frame joins the
# arc to the face; with only two belts, which are antipodal and can
# never meet, the cross-section is drawn as a straight chord instead
# of a sphere arc, which is what lets a belt as wide as the cube read
# as a flat strap rather than a trough.
#
# THE WIDTH OF A STRAP.  The frame field alone does not make a belt: a
# strap can bend only about its width and twist about its length, and
# the field bends the belts sideways as well (measured: up to about
# 160 degrees of in-plane bending along a belt, which cannot be
# avoided by any common field with fixed face widths -- the wrap that
# starts every cycle bends the belts across the spin axis exactly in
# their own plane).  Drawn with the frame's width vector such a bend
# shows as shear, the width vector falling toward the centre line
# (74 degrees at worst).  Instead the width is taken PERPENDICULAR to
# the centre line, within the sphere's tangent plane, wherever the
# belt is bending appreciably; where the bending is weak the frame's
# width is kept, since a tiny in-plane bend costs only a tiny shear
# while following it exactly would twist the strap through a right
# angle for nothing.  The sign of the perpendicular is carried
# continuously along the belt from the cube end, and the deviation
# angle -- taken modulo a half turn, since the strip is the same set
# under w -> -w -- is then smoothed along the belt over about half a
# belt width, which caps the twist rate where the bending direction
# swings quickly through an inflection.  The result is continuous in the turn, exactly
# periodic (nothing depends on history), and keeps every slice on its
# sphere.  Residual shear stays below about 40 degrees.
#
# References:
# - Andrew J. Hanson, "Visualizing Quaternions", Morgan Kaufmann
#   (2006), chapter 12.  The nested-sphere ("onion") visualisation of
#   a frame sequence, and the explicit double-twist deformation
#   12.3.2, credited to Hart, Francis and Kauffman, that is the frame
#   field used here.
# - David Pengelley and Daniel Ramras, "How efficiently can one
#   untangle a double-twist?  Waving is believing!", The Mathematical
#   Intelligencer 39 (2017), 27-40; arXiv:1610.04680.  The same
#   nullhomotopy as a closed form, its optimality, and the candle-dance
#   reading in which time is the loop parameter.
# - George K. Francis and Louis H. Kauffman, "Air on the Dirac
#   strings", in The Mathematical Legacy of Wilhelm Magnus,
#   Contemporary Mathematics 169 (1994), 261-276.  The film this
#   construction descends from.
# - Alexander E. Holroyd, "The Spinor Linkage - a mechanical
#   implementation of the plate trick", arXiv:2107.01681 (2021).  The
#   arm that precesses at half the cube's rate, and the remark that
#   any number of belts in any directions can share one untangling.
# - M. H. A. Newman, "On a string problem of Dirac", Journal of the
#   London Mathematical Society s1-17 (1942), 173-177.  The original
#   several-strings model and the proof that an odd number of turns
#   cannot be undone.
# - Jason Hise, "Belt trick" (animation, 2012), the six-belt picture
#   this generator reproduces; construction not published, inferred
#   here from the films' method.

import math

bl_info = {
    "name": "Dirac Belt Trick",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Odds & Ends",
    "description": "A spinning cube with a belt on every face, "
                   "untangling itself every 720 degrees",
    "category": "Add Mesh",
}

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, BoolProperty,
                           EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

from .spinor_generator import _qmul, _qrot, _qaxis, build_cage

TAU = 2.0 * math.pi
FULL_TURN = 2.0 * TAU

# How readily the width follows the bending (see "THE WIDTH OF A
# STRAP" above).  KAPPA sets the bending, relative to the radial
# progress, at which the perpendicular width takes over from the
# frame's; SMOOTH is the smoothing length of the width's deviation
# angle, in belt widths, which caps how fast the width may turn; TAPER
# is the fraction of the belt over which that deviation is eased back
# to zero before the straight run into the face.
KAPPA = 0.25
SMOOTH = 0.35
TAPER = 0.04

_Z = (0.0, 0.0, 1.0)

_M = (0.0, 1.0, 0.0)      # the body axis every belt bends about at its face
_PHI = 0.5 * (1.0 + math.sqrt(5.0))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    m = _norm(a)
    return (a[0] / m, a[1] / m, a[2] / m)


def _signs(*coords):
    """All sign combinations of the non-zero coordinates."""
    out = [()]
    for c in coords:
        out = [o + (v,) for o in out for v in ((c, -c) if c else (0.0,))]
    return sorted(set(out))


def _cyclic(pts):
    return sorted(set(tuple(p[(k + i) % 3] for k in range(3))
                      for p in pts for i in range(3)))


_SOLID_DATA = {
    # vertices, face normals (unnormalised)
    'TETRA': ([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
              [(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]),
    'CUBE': (_signs(1, 1, 1), _cyclic([(1, 0, 0), (-1, 0, 0)])),
    'OCTA': (_cyclic([(1, 0, 0), (-1, 0, 0)]), _signs(1, 1, 1)),
    'DODECA': (_signs(1, 1, 1) + _cyclic(_signs(0, 1.0 / _PHI, _PHI)),
               _cyclic(_signs(0, _PHI, 1))),
    'ICOSA': (_cyclic(_signs(0, _PHI, 1)),
              _signs(1, 1, 1) + _cyclic(_signs(0, 1.0 / _PHI, _PHI))),
}
_SIDES = {'TETRA': 3, 'CUBE': 4, 'OCTA': 3, 'DODECA': 5, 'ICOSA': 3}
SOLID_NAMES = ['TWO', 'TETRA', 'CUBE', 'OCTA', 'DODECA', 'ICOSA']


def solid(kind, half=0.15):
    """The solid at the centre with inradius `half`: (verts, faces,
    belts), belts a list of (face normal u, width direction w, face
    inradius).  'TWO' is the cube with belts on its z faces only."""
    verts0, normals = _SOLID_DATA['CUBE' if kind == 'TWO' else kind]
    normals = [_unit(tuple(float(c) for c in n)) for n in normals]
    inr = max(_dot(n, v) for n in normals for v in verts0)
    verts = [tuple(half * c / inr for c in v) for v in verts0]
    faces, belts = [], []
    for n in normals:
        top = max(_dot(n, v) for v in verts)
        idx = [i for i, v in enumerate(verts)
               if _dot(n, v) > top - 1e-9 * half]
        assert len(idx) == _SIDES['CUBE' if kind == 'TWO' else kind],             "face normal does not match the vertex set"
        c = tuple(sum(verts[i][k] for i in idx) / len(idx) for k in range(3))
        e1 = _unit(tuple(verts[idx[0]][k] - c[k] for k in range(3)))
        e2 = _cross(n, e1)
        idx.sort(key=lambda i: math.atan2(
            _dot(tuple(verts[i][k] - c[k] for k in range(3)), e2),
            _dot(tuple(verts[i][k] - c[k] for k in range(3)), e1)))
        faces.append(idx)
        if kind == 'TWO' and abs(n[2]) < 0.5:
            continue
        # width: the bending axis m projected into the face; a face
        # normal along m only twists there, so take the spin axis instead
        wm = tuple(_M[k] - _dot(_M, n) * n[k] for k in range(3))
        if _norm(wm) < 1e-6:
            wm = tuple(_Z[k] - _dot(_Z, n) * n[k] for k in range(3))
        w = _unit(wm)
        # face inradius: distance from the centre to the nearest edge
        rf = min(_norm(_cross(
            tuple(verts[idx[j]][k] - c[k] for k in range(3)),
            _unit(tuple(verts[idx[(j + 1) % len(idx)]][k] - verts[idx[j]][k]
                        for k in range(3)))))
            for j in range(len(idx)))
        belts.append((n, w, rf))
    return verts, faces, belts


def arc_gap(belts, lam, samples=13):
    """Smallest angular distance (radians) between the width arcs of any
    two belts on one sphere, each arc of half angle lam along its own
    width direction."""
    pts = []
    for u, w, _ in belts:
        arc = []
        for j in range(samples):
            a = lam * (-1.0 + 2.0 * j / (samples - 1))
            arc.append(tuple(math.cos(a) * u[k] + math.sin(a) * w[k]
                             for k in range(3)))
        pts.append(arc)
    best = math.pi
    for i in range(len(belts)):
        for j in range(i + 1, len(belts)):
            ui, uj = belts[i][0], belts[j][0]
            if _dot(ui, uj) < math.cos(min(math.pi, 2.0 * lam + best)):
                continue
            for p in pts[i]:
                for q in pts[j]:
                    d = max(-1.0, min(1.0, _dot(p, q)))
                    a = math.acos(d)
                    if a < best:
                        best = a
    return best


def width_limit(belts, half, thickness, margin=0.02):
    """Widest belt whose arcs on the innermost sphere clear each other
    by max(margin, 2.5 thickness), and which fits its face.  Found by
    bisection on the width; returns (width, gap in radians at it)."""
    need = max(margin, 2.5 * thickness)
    face = 2.0 * min(rf for _, _, rf in belts)
    lo, hi = 0.0, face
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        hs = attach_radius(half, mid)
        lam = math.asin(min(1.0, mid / (2.0 * hs)))
        gap = arc_gap(belts, lam)
        if 2.0 * hs * math.sin(0.5 * gap) >= need:
            lo = mid
        else:
            hi = mid
    hs = attach_radius(half, lo)
    return lo, arc_gap(belts, math.asin(min(1.0, lo / (2.0 * hs))))


# ---------------------------------------------------------------
# the frame field
# ---------------------------------------------------------------


RAMP = 0.35


def profile(x, ramp=None):
    """Stage profile: 0 below 0, 1 above 1, C2, slope 0 at both ends and
    a plateau of slope 1/(1 - ramp) between two ramps of width `ramp`
    whose slope rises as a smoothstep.  Its peak slope (1.54 for the
    default ramp) sets how tightly a belt is twisted at 360 degrees,
    and the ramp length how sharply the bending sets in at the zone's
    outer end; both crease the mesh if pushed (smootherstep's peak of
    1.875 did), and both ease with a longer zone, which is why Reach
    defaults to 0.7."""
    if ramp is None:
        ramp = RAMP
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    if x < ramp:
        u = x / ramp
        y = ramp * u * u * u * (1.0 - 0.5 * u)
    elif x <= 1.0 - ramp:
        y = x - 0.5 * ramp
    else:
        u = (1.0 - x) / ramp
        y = 1.0 - ramp - ramp * u * u * u * (1.0 - 0.5 * u)
    return y / (1.0 - ramp)


def field(g, psi):
    """Unit quaternion of the belt frame at stage g (0 at the cage, 1 at
    the cube) and half-turn psi: R_z(psi) q_T(pi g / 2, psi) R_z(-psi),
    in closed form.  The vector part of q_T is B m + C n with m = y,
    n = z; conjugating by R_z(psi) turns m to (-sin psi, cos psi, 0)."""
    rho = 0.5 * math.pi * g
    c, s = math.cos(rho), math.sin(rho)
    cp, sp = math.cos(psi), math.sin(psi)
    A = c * c + s * s * cp
    B = c * s * (1.0 - cp)
    C = s * sp
    return (A, -B * sp, B * cp, C)


def _field_by_conjugation(g, psi):
    """The same field assembled the long way, for the self-test."""
    rho = 0.5 * math.pi * g
    c, s = math.cos(rho), math.sin(rho)
    qt = (c * c + s * s * math.cos(psi), 0.0,
          c * s * (1.0 - math.cos(psi)), s * math.sin(psi))
    r = _qaxis(_Z, psi)
    rb = (r[0], -r[1], -r[2], -r[3])
    return _qmul(_qmul(r, qt), rb)


# ---------------------------------------------------------------
# one belt
# ---------------------------------------------------------------


def attach_radius(half, width):
    """Radius of the sphere whose chord of length `width` lies in the
    face plane at distance `half`: where the last belt slice sits."""
    return math.sqrt(half * half + 0.25 * width * width)


def belt_rows(psi, u, w, r_out=1.0, half=0.15, width=0.16, reach=0.5,
              ns=160, nlam=11, nstub=1, kappa=KAPPA, smooth=SMOOTH,
              taper=TAPER, r_min=None, flat=False):
    """Cross-section rows of one belt, cage end first, solid end last.

    The belt is straight and rigid with the solid for r <= r_min
    (default: the face) and untangles over the fraction `reach` of the
    belt beyond that.  With flat=False each of the first ns rows is an
    arc of the sphere r(s) with a chord of exactly `width`, and the last
    nstub rows are the flat stub in the solid's frame joining that arc
    to the face; with flat=True every row is a straight chord tangent
    to its sphere and the last row is the face itself.  Returns
    (rows, info) with the centre line, its tangent and the width vector
    for the checks."""
    if r_min is None:
        r_min = half
    hs = half if flat else attach_radius(half, width)
    dr = hs - r_out
    sig1 = min(1.0, (r_out - max(r_min, hs)) / (r_out - hs))
    sig_a = sig1 * (1.0 - reach)
    h = 1.0 / (ns - 1)
    sigma = [i * h for i in range(ns)]
    r = [r_out + dr * s for s in sigma]
    U, W = [], []
    for s in sigma:
        g = profile((s - sig_a) / (sig1 - sig_a))
        q = field(g, psi)
        U.append(_qrot(q, u))
        W.append(_qrot(q, w))
    # tangential heading r dU/ds (central differences; zero at the ends,
    # where g' vanishes)
    Ttan, tn = [], []
    for i in range(ns):
        if i == 0 or i == ns - 1:
            Ttan.append((0.0, 0.0, 0.0))
            tn.append(0.0)
            continue
        a, b = U[i - 1], U[i + 1]
        d = tuple(r[i] * (b[k] - a[k]) / (2.0 * h) for k in range(3))
        Ttan.append(d)
        tn.append(_norm(d))
    # the perpendicular width, sign carried from the solid end
    L = [None] * ns
    i1 = -1
    for i in range(ns):
        if tn[i] >= 1e-13:
            L[i] = _cross(U[i], tuple(c / tn[i] for c in Ttan[i]))
            i1 = i
        else:
            L[i] = W[i]
    if i1 >= 0:
        if _dot(L[i1], W[i1]) < 0.0:
            L[i1] = tuple(-c for c in L[i1])
        for i in range(i1 - 1, -1, -1):
            if tn[i] >= 1e-13 and _dot(L[i], L[i + 1]) < 0.0:
                L[i] = tuple(-c for c in L[i])
    eps = kappa * abs(dr)
    alpha = []
    Fv = [_cross(U[i], W[i]) for i in range(ns)]
    prev = 0.0
    for i in range(ns):
        a = tn[i] * tn[i] / (tn[i] * tn[i] + eps * eps)
        v = tuple(a * L[i][k] + (1.0 - a) * W[i][k] for k in range(3))
        ang = math.atan2(_dot(v, Fv[i]), _dot(v, W[i]))
        # unwrap along the belt MODULO A HALF TURN: the strip is the
        # same set under w -> -w, so only the line of the width matters,
        # and the blend vector's sign flips (it passes near zero where
        # the carried sign is anti-parallel to the frame's width) must
        # not register as half-turn twists
        while ang - prev > 0.5 * math.pi:
            ang -= math.pi
        while ang - prev < -0.5 * math.pi:
            ang += math.pi
        alpha.append(ang)
        prev = ang
    # cap the twist rate: smooth the deviation angle along the belt
    # (Gaussian, std = smooth belt widths).  Smoothing a LINE angle is
    # sound because the unwrap above made it continuous modulo a half
    # turn; smoothing the width vectors themselves would cancel across
    # a twist, and a hard rate cap either delays a twist into the bend
    # or lags every feature met from one side -- both were tried and
    # measured worse.
    if smooth > 0.0:
        sg = smooth * width / abs(dr)
        halfk = int(math.ceil(3.0 * sg / h))
        if halfk >= 1:
            kern = [math.exp(-0.5 * (j * h / sg) ** 2)
                    for j in range(-halfk, halfk + 1)]
            ksum = sum(kern)
            kern = [k / ksum for k in kern]
            sm = []
            for i in range(ns):
                acc = 0.0
                for j, kv in enumerate(kern):
                    idx = i + j - halfk
                    idx = 0 if idx < 0 else (ns - 1 if idx > ns - 1 else idx)
                    acc += kv * alpha[idx]
                sm.append(acc)
            alpha = sm
    # ease the deviation back to the frame's width line (the nearest
    # multiple of a half turn) over the last `taper` of the zone, so the
    # straight run stays rigid with the solid
    k_end = math.pi * round(alpha[-1] / math.pi)
    for i in range(ns):
        tp = (sigma[i] - (sig1 - taper)) / taper
        tp = 0.0 if tp < 0.0 else (1.0 if tp > 1.0 else tp)
        tp = tp * tp * (3.0 - 2.0 * tp)
        alpha[i] = k_end + (1.0 - tp) * (alpha[i] - k_end)
    wv = [tuple(math.cos(alpha[i]) * W[i][k] + math.sin(alpha[i]) * Fv[i][k]
                for k in range(3)) for i in range(ns)]
    # the slices
    rows = []
    lam_last = []
    for i in range(ns):
        row = []
        if flat:
            for j in range(nlam):
                y = 0.5 * width * (-1.0 + 2.0 * j / (nlam - 1))
                row.append(tuple(r[i] * U[i][k] + y * wv[i][k]
                                 for k in range(3)))
        else:
            lam_max = math.asin(min(1.0, width / (2.0 * r[i])))
            for j in range(nlam):
                lam = lam_max * (-1.0 + 2.0 * j / (nlam - 1))
                cl, sl = math.cos(lam), math.sin(lam)
                row.append(tuple(r[i] * (cl * U[i][k] + sl * wv[i][k])
                                 for k in range(3)))
                if i == ns - 1:
                    lam_last.append(lam)
        rows.append(row)
    if not flat:
        # the stub: flat, in the solid's frame, from the arc to the face,
        # ordered across like the last arc (whose width may have come
        # out as minus the face's: same strip, opposite sign)
        rc = _qaxis(_Z, 2.0 * psi)
        sgn = 1.0 if _dot(wv[-1], W[-1]) >= 0.0 else -1.0
        for kk in range(1, nstub + 1):
            t = kk / nstub
            row = []
            for lam in lam_last:
                y = sgn * hs * math.sin(lam)
                x0 = hs * math.cos(lam)
                x = x0 + t * (half - x0)
                row.append(_qrot(rc, tuple(x * u[k] + y * w[k]
                                           for k in range(3))))
            rows.append(row)
    T = [tuple(dr * U[i][k] + Ttan[i][k] for k in range(3)) for i in range(ns)]
    info = dict(sigma=sigma, r=r, U=U, W=W, wv=wv, T=T, tn=tn, hs=hs,
                sig1=sig1)
    return rows, info


def build_belts(psi, belts, r_out=1.0, half=0.15, width=0.16, reach=0.5,
                ns=160, nlam=11, nstub=1, r_min=None, flat=False):
    """All the belts as one mesh: (verts, faces, face_belt_index)."""
    verts, faces, mats = [], [], []
    for bi, (u, w, _) in enumerate(belts):
        rows, _ = belt_rows(psi, u, w, r_out, half, width, reach,
                            ns, nlam, nstub, r_min=r_min, flat=flat)
        base = len(verts)
        for row in rows:
            verts.extend(row)
        nr = len(rows)
        for i in range(nr - 1):
            for j in range(nlam - 1):
                a = base + i * nlam + j
                faces.append([a, a + 1, a + nlam + 1, a + nlam])
                mats.append(bi)
    return verts, faces, mats


def circumradius(verts):
    return max(_norm(v) for v in verts)


CLEAR = 1.03   # the straight run reaches this factor past the circumradius


def cube_gap(half, width, belts=None):
    """Angular gap (radians) between neighbouring belt arcs on the
    innermost sphere, and the same as a distance."""
    if belts is None:
        belts = solid('CUBE', half)[2]
    hs = attach_radius(half, width)
    lam = math.asin(min(1.0, width / (2.0 * hs)))
    gap = arc_gap(belts, lam)
    return gap, 2.0 * hs * math.sin(0.5 * max(0.0, gap))


# ---------------------------------------------------------------
# Blender operator
# ---------------------------------------------------------------

if _IN_BLENDER:

    def _colour(i, n):
        """Evenly spaced hues, alternating light and dark so neighbours
        on the solid stay distinct."""
        h = (i * 0.618033988749895) % 1.0
        v = 0.85 if i % 2 == 0 else 0.6
        k = int(h * 6.0)
        f = h * 6.0 - k
        p, q, t = v * 0.35, v * (1.0 - 0.65 * f), v * (1.0 - 0.65 * (1.0 - f))
        return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v),
                (v, p, q)][k % 6]

    def _material(name, rgb):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*rgb, 1.0)
        mat.use_nodes = True
        node = mat.node_tree.nodes.get("Principled BSDF")
        if node is not None:
            node.inputs["Base Color"].default_value = (*rgb, 1.0)
            node.inputs["Roughness"].default_value = 0.5
        return mat

    class MESH_OT_belt_trick_add(bpy.types.Operator):
        """Add the Dirac belt trick: a spinning solid with a belt on
        every face, the belts twisting and untangling with a period of
        two full turns.  Animate Turn from 0 to 720 degrees"""
        bl_idname = "mesh.belt_trick_add"
        bl_label = "Dirac Belt Trick"
        bl_options = {'REGISTER', 'UNDO'}

        solid: EnumProperty(
            name="Solid",
            description="What spins at the centre, and so how many belts",
            items=[('TWO', "Two Belts",
                    "A cube with belts on its top and bottom faces only: "
                    "the classic belt held at both ends, easiest to "
                    "follow through the cycle"),
                   ('TETRA', "Tetrahedron", "Four belts"),
                   ('CUBE', "Cube", "Six belts, one per face"),
                   ('OCTA', "Octahedron", "Eight belts"),
                   ('DODECA', "Dodecahedron", "Twelve belts"),
                   ('ICOSA', "Icosahedron", "Twenty belts")],
            default='TWO')
        turn: FloatProperty(
            name="Turn", default=math.radians(300.0), min=0.0,
            max=FULL_TURN, subtype='ANGLE',
            description="How far the solid has turned about the vertical "
                        "axis.  The belts return to their starting "
                        "state at 720 degrees, and not at 360; keyframe "
                        "this from 0 to 720 for one cycle")
        size: FloatProperty(
            name="Size", default=0.3, min=0.02, max=1.2,
            description="Diameter of the solid's inscribed sphere: the "
                        "distance between opposite faces, which for the "
                        "cube is its edge")
        belt_width: FloatProperty(
            name="Belt Width", default=0.2, min=0.01, max=1.0,
            description="Width of each belt.  Belts must clear one "
                        "another where they meet the solid, so each "
                        "solid has a widest belt for its size; a wider "
                        "request is clamped to it and reported")
        reach: FloatProperty(
            name="Reach", default=0.7, min=0.1, max=0.95,
            description="Fraction of each belt, measured in from the "
                        "cage to just past the solid's corners, that "
                        "takes part in the untangling; the rest lies "
                        "straight.  Smaller values coil the belts more "
                        "tightly round the solid")
        thickness: FloatProperty(
            name="Thickness", default=0.012, min=0.0, max=0.1,
            description="Belt thickness, applied as a Solidify modifier "
                        "centred on the strip; zero for a bare surface")
        resolution: IntProperty(
            name="Resolution", default=160, min=24, max=800,
            description="Samples along each belt")
        across: IntProperty(
            name="Across", default=11, min=2, max=32,
            description="Samples across each belt.  A belt twisted "
                        "through a full turn is a tight helicoid, and "
                        "too few samples across it show as creases")
        show_solid: BoolProperty(
            name="Solid", default=True,
            description="Add the solid, turned with the belt roots")
        show_cage: BoolProperty(
            name="Cage", default=True,
            description="Add the sphere of great circles the far ends "
                        "are pinned to")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Half-extent of the result, cage and belt "
                        "thickness included; 1.0 fits the 2 m cube")

        def _mesh_from(self, verts, faces, name, smooth=True):
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            # the stub's edge rows meet the face at a point, so weld the
            # coincident vertices that leaves; nothing else is doubled
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-9)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            for p in me.polygons:
                # The belts and the cage are sampled curved surfaces and
                # want smooth normals; the solid is a polyhedron whose
                # faces ARE flat, and smoothing them rounds off the very
                # edges that say which solid it is.
                p.use_smooth = smooth
            me.update()
            return me

        def execute(self, context):
            psi = 0.5 * self.turn
            half = 0.5 * self.size
            sverts, sfaces, belts = solid(self.solid, half)
            limit, _ = width_limit(belts, half, self.thickness)
            width = min(self.belt_width, limit)
            # the cage tube and the belt thickness both stick out past the
            # cage radius; fit them inside the half-extent
            cage_tube = 0.004
            fit = self.scale / (1.0 + max(0.5 * self.thickness, cage_tube))
            verts, faces, mats = build_belts(
                psi, belts, 1.0, half, width, self.reach,
                self.resolution, self.across,
                r_min=CLEAR * circumradius(sverts), flat=(self.solid == 'TWO'))
            verts = [(v[0] * fit, v[1] * fit, v[2] * fit) for v in verts]
            me = self._mesh_from(verts, faces, "Belt Trick")
            for i in range(len(belts)):
                me.materials.append(_material("Belt %d" % (i + 1),
                                              _colour(i, len(belts))))
            for p, mi in zip(me.polygons, mats):
                p.material_index = mi
            obj = bpy.data.objects.new("Belt Trick", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0.0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness * fit
                mod.offset = 0.0
                mod.use_even_offset = True
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

            parts = []
            if self.show_solid:
                rc = _qaxis(_Z, 2.0 * psi)
                sv = [tuple(fit * c for c in _qrot(rc, v)) for v in sverts]
                parts.append(("Belt Trick Solid", (sv, sfaces)))
            if self.show_cage:
                parts.append(("Belt Trick Cage",
                              build_cage(fit, 6, 96, cage_tube * fit)))
            for name, (pv, pf) in parts:
                sub = self._mesh_from(pv, pf, name,
                                      smooth="Solid" not in name)
                so = bpy.data.objects.new(name, sub)
                context.collection.objects.link(so)
                so.parent = obj

            gap, dist = cube_gap(half, width, belts)
            msg = ("V=%d F=%d  turn %.0f deg  %d belts %.3f wide, "
                   "neighbours %.0f deg (%.3f) apart at the solid"
                   % (len(me.vertices), len(me.polygons),
                      math.degrees(self.turn), len(belts), width * fit,
                      math.degrees(gap), dist * fit))
            if width < self.belt_width:
                self.report({'WARNING'}, msg + " - belt width clamped "
                            "from %.3f: wider belts would meet at the "
                            "solid" % (self.belt_width * fit))
            else:
                self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'solid')
            lay.prop(self, 'turn')
            lay.prop(self, 'size')
            lay.prop(self, 'belt_width')
            lay.prop(self, 'reach')
            lay.prop(self, 'thickness')
            lay.prop(self, 'resolution')
            lay.prop(self, 'across')
            lay.prop(self, 'show_solid')
            lay.prop(self, 'show_cage')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator(MESH_OT_belt_trick_add.bl_idname,
                             icon='FORCE_VORTEX')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_belt_trick_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_belt_trick_add)


# ---------------------------------------------------------------
# standalone numeric checks
# ---------------------------------------------------------------


def _selftest():
    tol = 1e-12
    psis = [TAU * k / 24.0 for k in range(25)]
    gs = [k / 16.0 for k in range(17)]

    # 1. the field: unit, boundaries, the closed form equals the
    #    conjugation, period 2 pi in psi (4 pi in turn), NOT pi
    for psi in psis:
        for g in gs:
            q = field(g, psi)
            assert abs(_norm(q[1:]) ** 2 + q[0] ** 2 - 1.0) < tol
            qc = _field_by_conjugation(g, psi)
            assert max(abs(q[k] - qc[k]) for k in range(4)) < 1e-12
            q2 = field(g, psi + TAU)
            assert max(abs(q[k] - q2[k]) for k in range(4)) < 1e-12, \
                "field not 4 pi periodic"
        assert max(abs(c) for c in field(0.0, psi)[1:]) < tol
        assert abs(field(0.0, psi)[0] - 1.0) < tol
        qc = _qaxis(_Z, 2.0 * psi)
        q = field(1.0, psi)
        assert max(abs(q[k] - qc[k]) for k in range(4)) < tol, \
            "cube end is not the cube's rotation"
    worst = max(max(abs(field(g, psi)[k] - field(g, psi + math.pi)[k])
                    for k in range(4))
                for g in gs for psi in psis)
    assert worst > 1.9, "field returns after a single turn"
    # the stage profile: 0 and 1 at the ends, flat there, monotone
    assert profile(0.0) == 0.0 and abs(profile(1.0) - 1.0) < tol
    assert profile(1e-6) < 1e-9 and 1.0 - profile(1.0 - 1e-6) < 1e-9
    last = -1.0
    for k in range(101):
        v = profile(k / 100.0)
        assert v >= last
        last = v

    # 2. the solids: right counts, faces at the inradius, widths in the
    #    face and along the bending axis's projection
    half, width = 0.15, 0.16
    counts = {'TWO': (8, 6, 2), 'TETRA': (4, 4, 4), 'CUBE': (8, 6, 6),
              'OCTA': (6, 8, 8), 'DODECA': (20, 12, 12), 'ICOSA': (12, 20, 20)}
    limits = {}
    for kind in SOLID_NAMES:
        verts, faces, belts = solid(kind, half)
        assert (len(verts), len(faces), len(belts)) == counts[kind], kind
        for u, w, rf in belts:
            assert abs(_norm(u) - 1.0) < tol and abs(_norm(w) - 1.0) < tol
            assert abs(_dot(u, w)) < tol
            mperp = tuple(_M[k] - _dot(_M, u) * u[k] for k in range(3))
            if _norm(mperp) > 1e-6:
                assert abs(abs(_dot(_unit(mperp), w)) - 1.0) < 1e-9
            assert rf > 0.0
        for n in [b[0] for b in belts]:
            top = max(_dot(n, v) for v in verts)
            assert abs(top - half) < 1e-12, "face not at the inradius"
        limits[kind] = width_limit(belts, half, 0.012)[0]
    assert abs(limits['TWO'] - 2.0 * half) < 1e-6      # face-bound only
    assert limits['ICOSA'] < limits['DODECA'] < limits['OCTA'] < \
        limits['CUBE'] < limits['TETRA']
    assert limits['ICOSA'] < width < limits['CUBE']

    # 3. the geometry, per solid: period, non-return, ends,
    #    orthogonality, width
    ns, nlam = 81, 5
    for kind in SOLID_NAMES:
        sverts, _, belts = solid(kind, half)
        w_use = min(width, limits[kind])
        flat = (kind == 'TWO')
        rmin = CLEAR * circumradius(sverts)
        for psi in psis[:12:3]:
            d360_all = 0.0
            for bi, (u, w, _) in enumerate(belts):
                args = (1.0, half, w_use, 0.5, ns, nlam)
                kw = dict(r_min=rmin, flat=flat)
                rows, info = belt_rows(psi, u, w, *args, **kw)
                rows2, _ = belt_rows(psi + TAU, u, w, *args, **kw)
                rows3, _ = belt_rows(psi + math.pi, u, w, *args, **kw)
                rc = _qaxis(_Z, 2.0 * psi)
                nu, nw = _qrot(rc, u), _qrot(rc, w)
                d720 = max(_norm(tuple(a[k] - b[k] for k in range(3)))
                           for ra, rb in zip(rows, rows2)
                           for a, b in zip(ra, rb))
                assert d720 < 1e-11, \
                    "belt not back after 720 degrees: %g" % d720
                # a belt along the loop axis is merely twisted at 360,
                # so its rows move little; the configuration as a whole
                # must move by more than a belt width
                d360_all = max(d360_all, max(
                    _norm(tuple(a[k] - b[k] for k in range(3)))
                    for ra, rb in zip(rows, rows3) for a, b in zip(ra, rb)))
                for i in range(ns):
                    # every slice on its sphere (a flat chord: its
                    # centre), radii strictly decreasing, and the belt
                    # straight and rigid with the solid inside r_min
                    for p in (rows[i][nlam // 2:nlam // 2 + 1] if flat
                              else rows[i]):
                        assert abs(_norm(p) - info['r'][i]) < 1e-12
                    if i:
                        assert info['r'][i] < info['r'][i - 1]
                    if info['r'][i] <= rmin:
                        assert _norm(tuple(info['U'][i][k] - nu[k]
                                           for k in range(3))) < 1e-12
                        assert abs(abs(_dot(info['wv'][i], nw)) - 1.0) < 1e-12
                assert _norm(tuple(rows[0][nlam // 2][k] - u[k]
                                   for k in range(3))) < tol
                for p in rows[-1]:
                    assert abs(_dot(p, nu) - half) < 1e-12, "stub off the face"
                for i in (0, ns - 1, len(rows) - 1):
                    edge = tuple(rows[i][-1][k] - rows[i][0][k]
                                 for k in range(3))
                    assert abs(_norm(edge) - w_use) < 1e-12
                edge = tuple(rows[-1][-1][k] - rows[-1][0][k] for k in range(3))
                assert abs(abs(_dot(_unit(edge), nw)) - 1.0) < 1e-12
                # the belt leaves the face along its normal: the last
                # two slices are exactly radial, and so is the tangent
                lastU, prevU = info['U'][-1], info['U'][-2]
                assert _norm(tuple(lastU[k] - prevU[k] for k in range(3))) < 1e-12
                assert abs(abs(_dot(_unit(info['T'][-1]), nu)) - 1.0) < 1e-12
                for i in range(1, ns - 1):
                    t = _unit(info['T'][i])
                    sh = math.degrees(math.asin(min(1.0, abs(_dot(t, info['wv'][i])))))
                    assert sh < 50.0, "shear %.1f deg" % sh
            assert d360_all > 0.3, "%s back after 360 degrees" % kind

    # 4. non-intersection: on every sphere the arcs stay apart by the
    #    packing gap, for the cube and the icosahedron
    for kind in ('CUBE', 'ICOSA'):
        belts = solid(kind, half)[2]
        w_use = min(width, limits[kind])
        gap, dist = cube_gap(half, w_use, belts)
        assert gap > 0.0
        for psi in psis[:12:2]:
            allrows = [belt_rows(psi, u, w, 1.0, half, w_use, 0.5, ns, 3,
                                 r_min=CLEAR * circumradius(solid(kind, half)[0]))[0]
                       for u, w, _ in belts]
            nb = len(belts)
            for i in range(0, ns, 2):
                for a in range(nb):
                    for b in range(a + 1, nb):
                        if _dot(belts[a][0], belts[b][0]) < 0.3:
                            continue
                        for p in allrows[a][i]:
                            for q in allrows[b][i]:
                                d = _norm(tuple(p[k] - q[k] for k in range(3)))
                                assert d > 0.98 * dist, \
                                    "%s belts %d,%d within %.3f" % (kind, a, b, d)

    # 5. the reported gap: 33.9 degrees for the cube defaults
    gap, dist = cube_gap(half, width)
    assert abs(math.degrees(gap) - 33.9) < 0.3
    print("belt_trick_generator: self-test OK")
