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
# cube.  The stage profile g(s) rises as two smoothstep ramps meeting in
# the middle of the zone, and that shape is the outcome of measuring
# the tightest bend any belt makes over the whole cycle.  In a 2 m box
# a belt across the loop axis coils once round the solid within half a
# metre of radius, so its tangent runs near 85 degrees from radial and
# the ELBOW where it turns from the straight run into that coil is the
# sharpest thing in the cycle: about 0.05 m of radius for the cube at
# 360 degrees, whatever the profile.  Concentrating the frame's turning
# anywhere -- a plateau, ramps shaped in the tilt, constant-curvature
# elbows -- buys a rounder elbow at 360 and pays with sharper S-bends
# at 60 or 630 degrees, where a belt's curvature follows the RATE OF
# CHANGE of the turning; the long smooth ramps are the balance.  The
# consequence for the belt's width is stated in the operator: a belt
# wider than that radius creases at the elbow, so the default width
# is set at the radius.
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
# and rigid with the solid out to past its circumradius, and the
# untangling zone begins there; a radial segment from a face centre
# never re-enters a convex solid.  It also leaves each face exactly
# orthogonally.  Beyond the straight run the profile g rises from 0 to
# 1 with g' = 0 at both ends, so the belt departs from straight without
# a kink.  The cross-section is a straight chord of the belt width,
# perpendicular to the centre line.
#
# THE TARGET AND THE DRAWN CURVE.  The centre line the field gives,
# r(s) F(s,t) u, is topologically right but not smooth enough to look
# like a belt: a full turn inside a 2 m cage runs the coil near 85
# degrees from radial, and the ELBOW where the belt turns from its
# straight run into that coil has a radius of about a third of the
# belt width whatever the stage profile (a plateau, ramps shaped in the
# tilt, constant-curvature elbows and a tilt bump were all measured;
# whatever rounds one elbow sharpens an S-bend at another turn, since
# at small turns a belt's curvature follows the rate of change of the
# frame's turning).  So the field's curve is treated as a TARGET, and
# the curve actually drawn is a penalised fit to it: resampled by arc
# length, then minimising the squared deviation from the target plus
# L^4 times the integral of curvature squared (discretely, the second
# differences), with the samples in both straight runs held fixed --
# which pins each end's position AND tangent, so the belt still meets
# its face and the cage exactly orthogonally.  The fit is a
# deterministic linear solve of the target, so an exactly periodic
# family of targets gives an exactly periodic family of curves, and
# the width vector is carried over by projecting the target's onto the
# fitted tangent.  What the fit cannot promise is the packing proof:
# the fitted curves leave their spheres, so they are swept for
# intersection instead -- a fit that crossed another belt would change
# the homotopy class, which is the one thing that would falsify the
# mathematics -- and the operator warns when the belts come within a
# thickness of one another at the turn shown.  The Smoothing property
# is L.  Measured on the cube in a 2 m cage: L = 0.05 lifts the
# tightest radius in the cycle from 0.068 m (the elbow) to 0.095 m for
# 1.3 cm of drift, after which the S-bend at 120 degrees is the limit;
# L of 0.1 and beyond starts to flatten the coil itself (half a metre
# of drift by 0.15).  Smoothing the direction path on the sphere
# instead, which would have kept the packing proof, does nothing for
# the elbow, since on the sphere the coil is a geodesic and the elbow
# is only a change of speed.
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
SMOOTH = 0.5
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


def solid(kind, half=0.15, n=_Z, m=_M):
    """The solid at the centre with inradius `half`: (verts, faces,
    belts), belts a list of (face normal u, width direction w, face
    inradius), for spin axis n and loop axis m.  'TWO' is the cube with
    belts on its z faces only."""
    verts0, normals = _SOLID_DATA['CUBE' if kind == 'TWO' else kind]
    normals = [_unit(tuple(float(c) for c in n)) for n in normals]
    inr = max(_dot(n, v) for n in normals for v in verts0)
    verts = [tuple(half * c / inr for c in v) for v in verts0]
    faces, belts = [], []
    for nf in normals:
        top = max(_dot(nf, v) for v in verts)
        idx = [i for i, v in enumerate(verts)
               if _dot(nf, v) > top - 1e-9 * half]
        assert len(idx) == _SIDES['CUBE' if kind == 'TWO' else kind],             "face normal does not match the vertex set"
        c = tuple(sum(verts[i][k] for i in idx) / len(idx) for k in range(3))
        e1 = _unit(tuple(verts[idx[0]][k] - c[k] for k in range(3)))
        e2 = _cross(nf, e1)
        idx.sort(key=lambda i: math.atan2(
            _dot(tuple(verts[i][k] - c[k] for k in range(3)), e2),
            _dot(tuple(verts[i][k] - c[k] for k in range(3)), e1)))
        faces.append(idx)
        if kind == 'TWO' and abs(nf[2]) < 0.5:
            continue
        # width: the bending axis m projected into the face; a face
        # normal along m only twists there, so take the spin axis instead
        wm = tuple(m[k] - _dot(m, nf) * nf[k] for k in range(3))
        if _norm(wm) < 1e-6:
            wm = tuple(n[k] - _dot(n, nf) * nf[k] for k in range(3))
        w = _unit(wm)
        # face inradius: distance from the centre to the nearest edge
        rf = min(_norm(_cross(
            tuple(verts[idx[j]][k] - c[k] for k in range(3)),
            _unit(tuple(verts[idx[(j + 1) % len(idx)]][k] - verts[idx[j]][k]
                        for k in range(3)))))
            for j in range(len(idx)))
        belts.append((nf, w, rf))
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


RAMP = 0.5


def profile(x, ramp=None):
    """Stage profile: 0 below 0, 1 above 1, C2, slope 0 at both ends and
    a plateau of slope 1/(1 - ramp) between two ramps of width `ramp`
    whose slope rises as a smoothstep.  The ramp length sets the
    radius of the elbow where a belt turns from radial into its coil
    at 360 degrees (an arctangent of the rate, so most of the turn
    happens early in the ramp), and the peak slope how tightly a belt
    along the loop axis is twisted and how sharp the S-bends elsewhere
    in the cycle are.  Measured over the whole cycle, the tightest
    centre-line radius is largest with no plateau at all (ramp 0.5,
    0.054 m for the cube in a 2 m cage at Reach 0.95); ramps shaped in
    the tilt, constant-curvature elbows and a tilt bump were all tried
    and are all worse somewhere else in the cycle."""
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


def field(g, psi, n=_Z, m=_M):
    """Unit quaternion of the belt frame at stage g (0 at the cage, 1 at
    the solid) and half-turn psi: R_n(psi) q_T(pi g / 2, psi) R_n(-psi),
    in closed form, for spin axis n and loop axis m perpendicular to
    it.  The vector part of q_T is B m + C n; conjugating by R_n(psi)
    turns m to cos psi m + sin psi (n x m)."""
    rho = 0.5 * math.pi * g
    c, s = math.cos(rho), math.sin(rho)
    cp, sp = math.cos(psi), math.sin(psi)
    A = c * c + s * s * cp
    B = c * s * (1.0 - cp)
    C = s * sp
    l = _cross(n, m)
    return (A,
            B * (cp * m[0] + sp * l[0]) + C * n[0],
            B * (cp * m[1] + sp * l[1]) + C * n[1],
            B * (cp * m[2] + sp * l[2]) + C * n[2])


def _field_by_conjugation(g, psi, n=_Z, m=_M):
    """The same field assembled the long way, for the self-test."""
    rho = 0.5 * math.pi * g
    c, s = math.cos(rho), math.sin(rho)
    B, C = c * s * (1.0 - math.cos(psi)), s * math.sin(psi)
    qt = (c * c + s * s * math.cos(psi),
          B * m[0] + C * n[0], B * m[1] + C * n[1], B * m[2] + C * n[2])
    r = _qaxis(n, psi)
    rb = (r[0], -r[1], -r[2], -r[3])
    return _qmul(_qmul(r, qt), rb)


# Spin axis -> (n, m): the loop axis m is perpendicular to the spin axis
# and, for the two-belt case whose belts lie along z, perpendicular to
# the belts as well, so that spinning about x or y bends and wraps them
# rather than merely twisting them.
AXES = {'X': ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        'Y': ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        'Z': ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))}


# ---------------------------------------------------------------
# one belt
# ---------------------------------------------------------------


def attach_radius(half, width):
    """Radius of the sphere whose chord of length `width` lies in the
    face plane at distance `half`: where the last belt slice sits."""
    return math.sqrt(half * half + 0.25 * width * width)


def belt_rows(psi, u, w, r_out=1.0, half=0.15, width=0.16, reach=0.5,
              ns=160, nlam=11, kappa=KAPPA, smooth=SMOOTH, taper=TAPER,
              r_min=None, n=_Z, m=_M, smoothing=None):
    """Cross-section rows of one belt, cage end first, solid end last.

    The belt is straight and rigid with the solid for r <= r_min
    (default: the face) and untangles over the fraction `reach` of the
    belt beyond that.  Every row is a straight chord of length `width`
    across the drawn centre line; the last row is the face itself.  With
    `smoothing` > 0 the drawn centre line is the penalised fit to the
    field's (see the header), resampled by arc length; otherwise it is
    the field's own.  Returns (rows, info): info carries the target
    (sigma, r, U, W, wv, T) and the drawn line (P) for the checks."""
    if r_min is None:
        r_min = half
    if smoothing is None:
        smoothing = SMOOTHING
    hs = half
    dr = hs - r_out
    sig1 = min(1.0, (r_out - max(r_min, hs)) / (r_out - hs))
    sig_a = sig1 * (1.0 - reach)
    h = 1.0 / (ns - 1)
    sigma = [i * h for i in range(ns)]
    r = [r_out + dr * s for s in sigma]
    U, W = [], []
    for s in sigma:
        g = profile((s - sig_a) / (sig1 - sig_a))
        q = field(g, psi, n, m)
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
    # straight run stays rigid with the solid; and to zero over the
    # straight run at the cage, which the smoothing above leaks into
    # when the zone starts close to it
    k_end = math.pi * round(alpha[-1] / math.pi)
    for i in range(ns):
        tp = (sigma[i] - (sig1 - taper)) / taper
        tp = 0.0 if tp < 0.0 else (1.0 if tp > 1.0 else tp)
        tp = tp * tp * (3.0 - 2.0 * tp)
        alpha[i] = k_end + (1.0 - tp) * (alpha[i] - k_end)
        t0 = sigma[i] / max(sig_a, 1e-9)
        t0 = 0.0 if t0 < 0.0 else (1.0 if t0 > 1.0 else t0)
        alpha[i] *= t0 * t0 * (3.0 - 2.0 * t0)
    wv = [tuple(math.cos(alpha[i]) * W[i][k] + math.sin(alpha[i]) * Fv[i][k]
                for k in range(3)) for i in range(ns)]
    # the target centre line, and the drawn one
    target = [tuple(r[i] * U[i][k] for k in range(3)) for i in range(ns)]
    run_out = r_out - (r_out + dr * sig_a)       # straight run at the cage
    run_in = (r_out + dr * sig1) - hs            # straight run at the solid
    if smoothing > 0.0:
        P, wd, fitted_to = smooth_curve(target, wv, run_out, run_in,
                                        smoothing)
    else:
        P, wd, fitted_to = target, wv, target
    # width vectors perpendicular to the drawn tangent
    rows = []
    for i in range(ns):
        a = P[max(i - 1, 0)]
        b = P[min(i + 1, ns - 1)]
        t = _unit(tuple(b[k] - a[k] for k in range(3)))
        d = _dot(wd[i], t)
        wv_i = tuple(wd[i][k] - d * t[k] for k in range(3))
        wv_i = _unit(wv_i) if _norm(wv_i) > 1e-9 else wd[i]
        row = []
        for j in range(nlam):
            y = 0.5 * width * (-1.0 + 2.0 * j / (nlam - 1))
            row.append(tuple(P[i][k] + y * wv_i[k] for k in range(3)))
        rows.append(row)
    T = [tuple(dr * U[i][k] + Ttan[i][k] for k in range(3)) for i in range(ns)]
    info = dict(sigma=sigma, r=r, U=U, W=W, wv=wv, T=T, tn=tn, hs=hs,
                sig1=sig1, P=P, target=fitted_to)
    return rows, info


def _resample(pts, vecs, count):
    """Resample a polyline (and a vector attached to each point) at
    `count` points evenly spaced in arc length; ends are kept exactly."""
    n = len(pts)
    arc = [0.0]
    for i in range(1, n):
        arc.append(arc[-1] + _norm(tuple(pts[i][k] - pts[i - 1][k]
                                         for k in range(3))))
    total = arc[-1]
    out_p, out_v = [], []
    j = 0
    for c in range(count):
        s = total * c / (count - 1)
        while j < n - 2 and arc[j + 1] < s:
            j += 1
        seg = arc[j + 1] - arc[j]
        f = 0.0 if seg <= 0.0 else min(1.0, max(0.0, (s - arc[j]) / seg))
        out_p.append(tuple(pts[j][k] + f * (pts[j + 1][k] - pts[j][k])
                           for k in range(3)))
        v = tuple(vecs[j][k] + f * (vecs[j + 1][k] - vecs[j][k])
                  for k in range(3))
        out_v.append(_unit(v) if _norm(v) > 1e-9 else vecs[j])
    out_p[-1] = pts[-1]
    out_v[-1] = vecs[-1]
    return out_p, out_v, total


def _solve_banded(diag, off1, off2, rhs):
    """Solve a symmetric pentadiagonal system given its diagonal and the
    two off-diagonals (off1[i] couples i and i+1, off2[i] couples i and
    i+2), for several right-hand sides.  Banded Gaussian elimination
    without pivoting; the matrix is positive definite."""
    n = len(diag)
    rows = []
    for i in range(n):
        row = {i: diag[i]}
        if i >= 1:
            row[i - 1] = off1[i - 1]
        if i >= 2:
            row[i - 2] = off2[i - 2]
        if i + 1 < n:
            row[i + 1] = off1[i]
        if i + 2 < n:
            row[i + 2] = off2[i]
        rows.append(row)
    b = [list(r) for r in rhs]
    nrhs = len(b)
    for i in range(n):
        piv = rows[i][i]
        for r in (i + 1, i + 2):
            if r >= n or i not in rows[r]:
                continue
            fac = rows[r][i] / piv
            if fac == 0.0:
                continue
            for c, v in rows[i].items():
                if c >= i:
                    rows[r][c] = rows[r].get(c, 0.0) - fac * v
            for k in range(nrhs):
                b[k][r] -= fac * b[k][i]
    x = [[0.0] * n for _ in range(nrhs)]
    for k in range(nrhs):
        for i in range(n - 1, -1, -1):
            v = b[k][i]
            for c, coef in rows[i].items():
                if c > i:
                    v -= coef * x[k][c]
            x[k][i] = v / rows[i][i]
    return x


def smooth_curve(target, vecs, run_out, run_in, L):
    """The penalised fit of the header: resample `target` by arc length,
    then minimise sum |p - t|^2 + (L/h)^4 sum |second difference|^2
    with the samples inside the straight runs (arc length <= run_out
    from the cage end, >= total - run_in at the solid end) held at the
    target, which pins both ends' positions and tangents.  Returns the
    fitted points and the resampled width vectors."""
    n = len(target)
    t, v, total = _resample(target, vecs, n)
    h = total / (n - 1)
    mu = (L / h) ** 4
    fixed = [False] * n
    for i in range(n):
        s = h * i
        if s <= run_out + 1e-12 or s >= total - run_in - 1e-12:
            fixed[i] = True
    # the second sample from each end is pinned ON the radial line of
    # its straight run (the run may be shorter than one sample), which
    # is what clamps the end tangent to the face normal and the radius
    fixed[0] = fixed[1] = fixed[-1] = fixed[-2] = True
    r0, r1 = _norm(t[0]), _norm(t[-1])
    t[1] = tuple(t[0][k] * (1.0 - h / r0) for k in range(3))
    t[-2] = tuple(t[-1][k] * (1.0 + h / r1) for k in range(3))
    # A = W + mu D^T D, W = identity on the free rows; fixed rows become
    # identities with the target on the right, and their couplings move
    # to the right-hand side of their neighbours
    diag = [0.0] * n
    off1 = [0.0] * (n - 1)
    off2 = [0.0] * (n - 2)
    for i in range(1, n - 1):
        # row of D: (1, -2, 1) at i-1, i, i+1
        diag[i - 1] += mu
        diag[i] += 4.0 * mu
        diag[i + 1] += mu
        off1[i - 1] += -2.0 * mu
        off1[i] += -2.0 * mu
        off2[i - 1] += mu
    rhs = [[0.0] * n for _ in range(3)]
    for i in range(n):
        if not fixed[i]:
            diag[i] += 1.0
            for k in range(3):
                rhs[k][i] += t[i][k]
    # fold the fixed samples into the right-hand side and decouple them
    for i in range(n):
        if fixed[i]:
            for j, coef in ((i - 2, off2[i - 2] if i >= 2 else 0.0),
                            (i - 1, off1[i - 1] if i >= 1 else 0.0),
                            (i + 1, off1[i] if i + 1 < n else 0.0),
                            (i + 2, off2[i] if i + 2 < n else 0.0)):
                if 0 <= j < n and not fixed[j]:
                    for k in range(3):
                        rhs[k][j] -= coef * t[i][k]
    for i in range(n):
        if fixed[i]:
            diag[i] = 1.0
            for k in range(3):
                rhs[k][i] = t[i][k]
            if i >= 1:
                off1[i - 1] = 0.0
            if i + 1 < n:
                off1[i] = 0.0
            if i >= 2:
                off2[i - 2] = 0.0
            if i + 2 < n:
                off2[i] = 0.0
    x = _solve_banded(diag, off1, off2, rhs)
    P = [(x[0][i], x[1][i], x[2][i]) for i in range(n)]
    return P, v, t


def belt_clearance(all_rows, per_belt=20):
    """Rough clearance between distinct belts at one turn: the smallest
    distance between sample points on their strips (both edges and the
    centre of about `per_belt` rows each), so a value below the
    thickness means the belts may touch."""
    best = 1e30
    pts = []
    for rows in all_rows:
        step = max(1, len(rows) // per_belt)
        sub = []
        for row in rows[::step]:
            sub.extend((row[0], row[len(row) // 2], row[-1]))
        pts.append(sub)
    nb = len(pts)
    for a in range(nb):
        for b in range(a + 1, nb):
            for p in pts[a]:
                for q in pts[b]:
                    d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2
                    if d < best:
                        best = d
    return math.sqrt(best)


def build_belts(psi, belts, r_out=1.0, half=0.15, width=0.16, reach=0.5,
                ns=160, nlam=11, r_min=None, n=_Z, m=_M, smoothing=None):
    """All the belts as one mesh: (verts, faces, face_belt_index,
    rows per belt)."""
    verts, faces, mats, all_rows = [], [], [], []
    for bi, (u, w, _) in enumerate(belts):
        rows, _ = belt_rows(psi, u, w, r_out, half, width, reach,
                            ns, nlam, r_min=r_min, n=n, m=m,
                            smoothing=smoothing)
        all_rows.append(rows)
        base = len(verts)
        for row in rows:
            verts.extend(row)
        nr = len(rows)
        for i in range(nr - 1):
            for j in range(nlam - 1):
                a = base + i * nlam + j
                faces.append([a, a + 1, a + nlam + 1, a + nlam])
                mats.append(bi)
    return verts, faces, mats, all_rows


def circumradius(verts):
    return max(_norm(v) for v in verts)


CLEAR = 1.15   # the straight run reaches this factor past the circumradius
SMOOTHING = 0.05   # default length scale of the bending penalty, metres


def min_bend_radius(belts, half, width, reach, r_min, n=_Z, m=_M,
                    smoothing=None, ns=240):
    """Smallest radius of curvature of a drawn centre line, sampled at
    the turns where it is smallest (360 degrees, the elbow into the
    coil; 120 and 255, the S-bends) for the first belt and the belt
    least aligned with it."""
    picks = [belts[0]]
    if len(belts) > 2:
        picks.append(min(belts[1:], key=lambda b: abs(_dot(b[0], belts[0][0]))))
    best = 1e30
    for (u, w, _) in picks:
        for psi in (math.pi, math.pi / 3.0, math.radians(127.5)):
            rows, _ = belt_rows(psi, u, w, 1.0, half, width, reach, ns, 3,
                                r_min=r_min, n=n, m=m, smoothing=smoothing)
            pts = [row[1] for row in rows]
            for i in range(1, ns - 1):
                a, b, c = pts[i - 1], pts[i], pts[i + 1]
                ab = tuple(b[k] - a[k] for k in range(3))
                bc = tuple(c[k] - b[k] for k in range(3))
                ac = tuple(c[k] - a[k] for k in range(3))
                cr = _norm(_cross(ab, bc))
                if cr > 1e-30:
                    best = min(best, _norm(ab) * _norm(bc) * _norm(ac) / (2.0 * cr))
    return best


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
        spin_axis: EnumProperty(
            name="Spin Axis",
            description="The axis the solid turns about.  The two belts "
                        "lie along z, so spinning about z twists them "
                        "about their own length while x or y bends and "
                        "wraps them round the cube",
            items=[('X', "X", "Spin about the x axis"),
                   ('Y', "Y", "Spin about the y axis"),
                   ('Z', "Z", "Spin about the z axis")],
            default='Z')
        turn: FloatProperty(
            name="Turn", default=math.radians(300.0), min=0.0,
            max=FULL_TURN, subtype='ANGLE',
            description="How far the solid has turned about the spin "
                        "axis.  The belts return to their starting "
                        "state at 720 degrees, and not at 360; keyframe "
                        "this from 0 to 720 for one cycle")
        size: FloatProperty(
            name="Size", default=0.16, min=0.02, max=1.2,
            description="How big the solid is, measured across its "
                        "CIRCUMSPHERE so that every solid sits the same "
                        "way inside the cage.  Measuring across the "
                        "faces instead would leave the tetrahedron half "
                        "again too big and the dodecahedron a third too "
                        "small, since the two radii are in a very "
                        "different ratio for each.  For the cube this "
                        "is its edge")
        belt_width: FloatProperty(
            name="Belt Width", default=0.0, min=0.0, max=1.0,
            description="Width of each belt, built exactly as set.  Leave at 0 for the widest belt that still clears its neighbours at the solid, which is the only setting safe for every solid: the limit runs from 0.30 for two belts down to 0.03 for the icosahedron's twenty.  Any positive value is used as given, with a warning if the belts would meet.  Original: Width of each belt.  Belts must clear one "
                        "another where they meet the solid, so each "
                        "solid has a widest belt for its size; a wider "
                        "request is clamped to it and reported.  The "
                        "tightest bend a belt makes over the cycle has "
                        "a radius of about 0.05 m in a 2 m cage whatever "
                        "the settings, so a belt wider than that creases "
                        "there; the operator reports the ratio")
        reach: FloatProperty(
            name="Reach", default=0.95, min=0.1, max=0.95,
            description="Fraction of each belt, measured in from the "
                        "cage to just past the solid's corners, that "
                        "takes part in the untangling; the rest lies "
                        "straight.  Smaller values coil the belts more "
                        "tightly round the solid")
        smoothing: FloatProperty(
            name="Smoothing", default=SMOOTHING, min=0.0, max=0.2,
            subtype='DISTANCE',
            description="Length scale of the bending penalty the drawn "
                        "belt is fitted with: bends of about this size "
                        "and smaller are rounded off, trading exactness "
                        "of the untangling path for a swoopier belt.  "
                        "Around 0.05 rounds the elbows where a belt "
                        "enters its coil for about a centimetre of "
                        "drift; much above 0.1 the fit starts to flatten "
                        "the coils themselves and pulls belts together.  "
                        "Zero draws the field's own path, elbows and "
                        "all")
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
            n, m = AXES[self.spin_axis]
            sverts, sfaces, belts = solid(self.solid, half, n, m)
            # Normalise on the CIRCUMSPHERE, not the inscribed sphere,
            # so that every solid sits the same way inside the cage.
            # The ratio of the two differs sharply between them: at an
            # equal face-to-face size the tetrahedron's circumradius is
            # three times its inradius where the cube's is only 1.73,
            # so the tetrahedron looked half again too big and the
            # dodecahedron and icosahedron nearly a third too small.
            # Scaling to the cube's circumradius leaves the cube, the
            # two-belt cube and the octahedron exactly as they were.
            cr = circumradius(sverts)
            if cr > 1e-9:
                half *= half * math.sqrt(3.0) / cr
                sverts, sfaces, belts = solid(self.solid, half, n, m)
            # The width is the user's to set.  Past `limit` the belts
            # cannot all clear one another where they crowd together at
            # the solid, so a wider belt will pass through its
            # neighbours -- but that is a look to be reported, not a
            # value to be overridden behind the user's back.  Clamping
            # it silently meant the number in the panel was not the
            # number that got built.
            # Width is the user's to set, and is built exactly as asked.
            # Zero asks for the widest belt that still clears its
            # neighbours where they crowd together at the solid, which
            # is the only value that is safe for EVERY solid -- the
            # limit falls from 0.30 for two belts to 0.03 for the
            # icosahedron's twenty, so no single fixed default can be
            # both strap-like and non-overlapping across the enum.
            limit, _ = width_limit(belts, half, self.thickness)
            if self.belt_width > 1e-6:
                width = self.belt_width
            else:
                # Auto: the widest belt that satisfies BOTH constraints
                # -- it must clear its neighbours where they crowd at
                # the solid, and it must not be wider than its own
                # tightest bend, or it creases there.  Taking only the
                # packing limit gives a belt that fits but folds; only
                # the bend radius gives one that is smooth but may
                # overlap.  No fixed number can serve every solid: the
                # packing limit alone runs from 0.30 for two belts to
                # 0.03 for the icosahedron's twenty.
                width = min(limit,
                            min_bend_radius(belts, half, limit,
                                            self.reach,
                                            CLEAR * circumradius(sverts),
                                            n, m, self.smoothing))
            # the cage tube and the belt thickness both stick out past the
            # cage radius; fit them inside the half-extent
            cage_tube = 0.004
            fit = self.scale / (1.0 + max(0.5 * self.thickness, cage_tube))
            verts, faces, mats, all_rows = build_belts(
                psi, belts, 1.0, half, width, self.reach,
                self.resolution, self.across,
                r_min=CLEAR * circumradius(sverts), n=n, m=m,
                smoothing=self.smoothing)
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
                rc = _qaxis(n, 2.0 * psi)
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
            bend = min_bend_radius(belts, half, width, self.reach,
                                   CLEAR * circumradius(sverts), n, m,
                                   self.smoothing) * fit
            clear = belt_clearance(all_rows) * fit
            msg = ("V=%d F=%d  turn %.0f deg  %d belts %.3f wide, "
                   "neighbours %.0f deg (%.3f) apart at the solid, "
                   "tightest bend radius %.3f = %.1f widths"
                   % (len(me.vertices), len(me.polygons),
                      math.degrees(self.turn), len(belts), width * fit,
                      math.degrees(gap), dist * fit, bend,
                      bend / (width * fit)))
            if width > limit:
                self.report({'WARNING'}, msg + " - wider than %.3f, so "
                            "the belts meet one another at the solid; "
                            "narrow it, or use fewer belts"
                            % (limit * fit))
            elif len(belts) > 1 and clear < self.thickness * fit:
                self.report({'WARNING'}, msg + " - belts come within "
                            "%.3f of one another at this turn; lower "
                            "Smoothing or narrow the belt" % clear)
            elif bend < width * fit:
                self.report({'WARNING'}, msg + " - the belt is wider "
                            "than its tightest bend and will crease "
                            "there; raise Smoothing or narrow it")
            else:
                self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'solid')
            lay.prop(self, 'spin_axis')
            lay.prop(self, 'turn')
            lay.prop(self, 'size')
            lay.prop(self, 'belt_width')
            lay.prop(self, 'reach')
            lay.prop(self, 'smoothing')
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

    # 1. the field, for every spin axis: unit, boundaries, the closed
    #    form equals the conjugation, period 2 pi in psi (4 pi in turn),
    #    NOT pi
    for n, m in AXES.values():
        for psi in psis:
            for g in gs:
                q = field(g, psi, n, m)
                assert abs(_norm(q[1:]) ** 2 + q[0] ** 2 - 1.0) < tol
                qc = _field_by_conjugation(g, psi, n, m)
                assert max(abs(q[k] - qc[k]) for k in range(4)) < 1e-12
                q2 = field(g, psi + TAU, n, m)
                assert max(abs(q[k] - q2[k]) for k in range(4)) < 1e-12, \
                    "field not 4 pi periodic"
            assert max(abs(c) for c in field(0.0, psi, n, m)[1:]) < tol
            assert abs(field(0.0, psi, n, m)[0] - 1.0) < tol
            qc = _qaxis(n, 2.0 * psi)
            q = field(1.0, psi, n, m)
            assert max(abs(q[k] - qc[k]) for k in range(4)) < tol, \
                "solid end is not the solid's rotation"
        worst = max(max(abs(field(g, psi, n, m)[k]
                            - field(g, psi + math.pi, n, m)[k])
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

    # 3. the geometry, per solid and axis: period, non-return, ends,
    #    orthogonality, width, the straight runs held, the fit close to
    #    its target
    ns, nlam = 81, 5
    cases = [(kind, 'Z') for kind in SOLID_NAMES] + [('TWO', 'X'), ('TWO', 'Y')]
    for kind, axis in cases:
        n, m = AXES[axis]
        sverts, _, belts = solid(kind, half, n, m)
        w_use = min(width, limits[kind])
        rmin = CLEAR * circumradius(sverts)
        for psi in psis[:12:3]:
            d360_all = 0.0
            for bi, (u, w, _) in enumerate(belts):
                args = (1.0, half, w_use, 0.95, ns, nlam)
                kw = dict(r_min=rmin, n=n, m=m)
                rows, info = belt_rows(psi, u, w, *args, **kw)
                rows2, _ = belt_rows(psi + TAU, u, w, *args, **kw)
                rows3, _ = belt_rows(psi + math.pi, u, w, *args, **kw)
                rc = _qaxis(n, 2.0 * psi)
                nu, nw = _qrot(rc, u), _qrot(rc, w)
                d720 = max(_norm(tuple(a[k] - b[k] for k in range(3)))
                           for ra, rb in zip(rows, rows2)
                           for a, b in zip(ra, rb))
                assert d720 < 1e-7, \
                    "belt not back after 720 degrees: %g" % d720
                d360_all = max(d360_all, max(
                    _norm(tuple(a[k] - b[k] for k in range(3)))
                    for ra, rb in zip(rows, rows3) for a, b in zip(ra, rb)))
                P = info['P']
                # cage end: on the cage, along u, width w; the first two
                # rows radial
                assert _norm(tuple(rows[0][nlam // 2][k] - u[k]
                                   for k in range(3))) < 1e-7
                edge = tuple(rows[0][-1][k] - rows[0][0][k] for k in range(3))
                assert abs(_norm(edge) - w_use) < 1e-7
                assert abs(abs(_dot(_unit(edge), w)) - 1.0) < 1e-7
                step = _unit(tuple(P[1][k] - P[0][k] for k in range(3)))
                assert abs(abs(_dot(step, u)) - 1.0) < 1e-7
                # solid end: the last row is the face's chord, the last
                # two rows radial along the turned normal
                for pt in rows[-1]:
                    assert abs(_dot(pt, nu) - half) < 1e-7, "end off the face"
                edge = tuple(rows[-1][-1][k] - rows[-1][0][k] for k in range(3))
                assert abs(_norm(edge) - w_use) < 1e-7
                assert abs(abs(_dot(_unit(edge), nw)) - 1.0) < 1e-7
                step = _unit(tuple(P[-1][k] - P[-2][k] for k in range(3)))
                assert abs(abs(_dot(step, nu)) - 1.0) < 1e-7
                # every chord is the belt width, perpendicular to the line
                for i in range(1, ns - 1):
                    edge = tuple(rows[i][-1][k] - rows[i][0][k] for k in range(3))
                    assert abs(_norm(edge) - w_use) < 1e-7
                    t = _unit(tuple(P[i + 1][k] - P[i - 1][k] for k in range(3)))
                    assert abs(_dot(_unit(edge), t)) < 1e-6
                # the drawn line stays within a few widths of the target
                # and outside the solid's circumsphere beyond the runs
                tgt = info['target']
                dev = max(_norm(tuple(pp[k] - tt[k] for k in range(3)))
                          for pp, tt in zip(P, tgt))
                assert dev < 0.1, "fit strays %g" % dev
                assert min(_norm(pp) for pp in P) >= half - 1e-9
            assert d360_all > 0.3, "%s %s back after 360 degrees" % (kind, axis)

    # 4. non-intersection, roughly: at every sampled turn the strips of
    #    the cube's and the icosahedron's belts stay more than a
    #    thickness apart (the sweep in the scratch harness is the real
    #    check; this catches a fit that pulls belts together)
    for kind in ('CUBE', 'ICOSA'):
        sverts, _, belts = solid(kind, half)
        w_use = min(width, limits[kind])
        rmin = CLEAR * circumradius(sverts)
        for psi in psis[:12:2]:
            allrows = [belt_rows(psi, u, w, 1.0, half, w_use, 0.95, ns, 3,
                                 r_min=rmin)[0] for u, w, _ in belts]
            assert belt_clearance(allrows, per_belt=40) > 0.012,                 "%s belts touch at %.0f deg" % (kind, math.degrees(2 * psi))

    # 5. the reported gap: 33.9 degrees for the cube defaults
    gap, dist = cube_gap(half, width)
    assert abs(math.degrees(gap) - 33.9) < 0.3
    print("belt_trick_generator: self-test OK")
