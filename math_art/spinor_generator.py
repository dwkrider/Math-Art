# Spinor / Dirac Belt Trick generator for Blender
#
# Static geometry for the belt trick: the fact that a 2*pi rotation
# of an object tethered to its surroundings cannot be undone, while
# a 4*pi rotation can.  Formally pi_1(SO(3)) = Z/2, so the
# double-twist loop in SO(3) is nullhomotopic while the single twist
# is not.  The add-on emits no animation, so the *motion* is
# spatialised here: an untangling is a map of a rectangle
# (s,t) -> SO(3) with the loop t running from the identity back to
# itself, and that rectangle is what gets meshed.
#
# THE HOMOTOPY.  Pengelley and Ramras give a closed-form nullhomotopy
# of the double twist -- their "double-tipping" construction -- as the
# unit quaternion
#
#   Dhat(s,t) = (1 - 2 cos^2 s sin^2(t/2))
#             + I (sin 2s sin^2(t/2))
#             + K (cos s sin t),
#
# for 0 <= s <= pi/2, 0 <= t <= 2pi.  At s = 0 this is cos t + K sin t,
# the double twist about K; at s = pi/2 it is the constant 1; and at
# t = 0 and t = 2pi it is the identity, so every stage is a loop based
# at the identity.
#
# THE STAGE, IN DEGREES.  The parameter s is not a quantity anyone can
# picture, so the UI works in degrees of rotation instead.  The
# rotation angle of Dhat(s,t) is largest at t = pi, where the real
# part is 1 - 2 cos^2 s; the angle there is 2 pi - 4 s, and since the
# real part is symmetric about t = pi the angle rises to that value
# and falls back.  So the TOTAL rotation swept over one cycle is
#
#   Theta(s) = 4 pi - 8 s,        s = (4 pi - Theta) / 8,
#
# exactly linear: 720 degrees at the stuck double twist, 0 degrees
# when the belt is flat.  The billowing variant below tips the axis
# but leaves the real part alone, so this reading is the same for
# both homotopies.
#
# TWO VARIANTS, which is why this generator has a Homotopy setting:
#
#   * PENGELLEY-RAMRAS (efficient).  The formula above.  Its J
#     component vanishes identically, so the untangling uses only
#     rotation axes in the x-z plane.  It is one-to-one except along
#     the edges the boundary conditions force, and onto the RP^2 it
#     lands in; since its axes are coplanar, its image in the ball
#     model is the flat x-z disc.
#   * FRANCIS-KAUFFMAN (billowing).  The older nullhomotopy, which
#     spends all three axial directions.  Pengelley and Ramras give
#     the conversion (up to permuting and negating some coordinates):
#     rotate the K component by s radians toward J, so K(cos s sin t)
#     becomes
#         J (sin s)(cos s sin t) + K (cos s)(cos s sin t),
#     leaving the real and I components alone.  The norm is unchanged,
#     so this is the same unit quaternion field with the axis tipped
#     out of plane -- but the swept image is now three-dimensional.
#
# THE BALL MODEL.  A rotation by angle th about unit axis u is the
# point (th/pi) u of the unit ball, antipodal boundary points being
# identified; the ball is then RP^3 = SO(3) (Stoytchev; Pengelley-
# Ramras Fig. 1).  A unit quaternion q = (w, v) maps in by taking the
# representative with w >= 0 -- q and -q are the same rotation -- and
# th = 2 arccos w.
#
# That choice of representative is discontinuous, and the seam runs
# through the INTERIOR of the parameter rectangle: the real part
# vanishes where 2 cos^2 s sin^2(t/2) = 1, that is along
#
#   t1(s) = 2 arcsin( 1 / (sqrt 2 cos s) )   and   t2 = 2pi - t1,
#
# which is real exactly for s <= pi/4.  Crossing it, the image jumps
# to the antipodal boundary point.  So the surface is built as three
# separate patches -- t in [0,t1], [t1,t2], [t2,2pi] -- and every
# curve is split the same way.  Meshing across the seam would draw
# faces spanning a full diameter of the ball.
#
# THE BELTS.  Attaching a frame to each point along a belt turns the
# belt into a path in SO(3).  Here the belt is developed FROM the
# frame rather than drawn along a fixed line: the core is the running
# integral of R(s,t) applied to K, and the width vector is R(s,t)
# applied to I.  Both come from the same rotation, so the width stays
# exactly perpendicular to the core by construction.  At s = 0 the
# core is straight and the strip carries 4 pi of twist; at s = pi/2 it
# is straight and flat; in between the core bows out, which is the
# "waving" of the title.
#
# WHY THE ENDS OF A BELT MOVE, AND WHY IT CAN MEET ITSELF.  Pengelley
# and Ramras are explicit that only the ROTATIONAL positioning of each
# frame matters here, not the translational component: the untangling
# is a homotopy of loops in SO(3) and says nothing about where the
# belt lies.  Drawing one anyway forces a choice, and the honest
# choice is the one above -- the belt's tangent IS the frame's own
# length vector, which is what makes it a belt rather than a sheared
# ribbon.  The price is that the far end sits at the integral of that
# vector, so it moves as the stage changes: a full belt-length from
# the near end at the double twist and at the flat belt, closing up in
# between, and for the billowing variant meeting it exactly at 360
# degrees.  Around 421 degrees the efficient variant passes through
# itself.  A real belt does neither, so the operator REPORTS the gap
# rather than pretending otherwise.  Staley describes the same thing
# from the other side -- in his deformations one "moves the buckle
# towards the left, and the twist becomes a coil" -- so the coiling is
# authentic to the subject even where the drawn belt is not physical.  The alternatives were tried and
# are worse: pinning the core straight makes the width vector fall
# onto it (0.998 parallel at 450 degrees), and reconstructing from the
# twist angle about a fixed axis jumps by a whole turn between stages.
#
# THE ROSETTE.  Dirac's original model is not one belt but several,
# radiating from a turning hub -- scissors on strings, or a hand
# holding cups.  Mode "Belt Rosette" places the same stage on each of
# n belts spaced around the axis.  Newman (1942) proved this is where
# the content is: with n >= 3 an odd number of turns can never be
# undone, while an even number always can, and with n <= 2 even the
# odd case comes free.
#
# THE LIFT.  Dhat itself is the lift of the homotopy to the double
# cover S^3.  Mode "S3 Lift" stereographically projects it into R^3.
# The pole -1 lies ON the image (at s = 0, t = pi), so the quaternion
# is first turned by a fixed rotation about J.  That moves the pole
# clear of the image for BOTH variants: the turned pole has a
# non-zero J component, and on the image the J and I components
# vanish together, so the two never meet.
#
# References:
# - David Pengelley and Daniel Ramras, "How efficiently can one
#   untangle a double-twist?  Waving is believing!", The Mathematical
#   Intelligencer 39 (2017), 27-40; arXiv:1610.04680.  Source of the
#   closed-form nullhomotopy, of the ball model used here, and of the
#   conversion to the Francis-Kauffman variant.
# - George K. Francis and Louis H. Kauffman, "Air on the Dirac
#   strings", in The Mathematical Legacy of Wilhelm Magnus,
#   Contemporary Mathematics 169 (1994), 261-276.  The billowing
#   nullhomotopy, implemented here in the form Pengelley and Ramras
#   give for it, up to a permutation and negation of coordinates.
# - Orlin Stoytchev, "Topology of SO(3) for Kids", arXiv:2310.19665
#   (2023).  The solid-torus-to-ball derivation of SO(3) = RP^3.
# - Ethan D. Bolker, "The Spinor Spanner", American Mathematical
#   Monthly 80 (1973), 977-984.  The disc model and an explicit
#   tease-apart homotopy.
# - M. H. A. Newman, "On a string problem of Dirac", Journal of the
#   London Mathematical Society s1-17 (1942), 173-177.  Dirac's
#   original scissors-and-string model, and the braid-group proof
#   that an odd number of turns cannot be undone.
# - Andrew J. Hanson and Hui Ma, "Parallel transport approach to
#   curve framing", Indiana University Computer Science Technical
#   Report TR-425 (1995).  The discrete parallel-transport frame used
#   by the tube sweep, and the closed-curve spin correction.
# - Mark Staley, "Understanding quaternions and the Dirac belt
#   trick", European Journal of Physics 31 (2010), 467;
#   arXiv:1001.1778.  The belt-as-parameter-path exposition, and the
#   account of belt motions that turn a twist into a coil.

import math
import pathlib

bl_info = {
    "name": "Spinor Belt Trick",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Math Art > Odds & Ends",
    "description": "Nullhomotopy of the double twist in SO(3): the "
                   "Dirac belt trick as static geometry",
    "category": "Add Mesh",
}

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


TAU = 2.0 * math.pi
HALF_PI = 0.5 * math.pi
FULL_TURN = 2.0 * TAU          # 4 pi: the stuck double twist


# ---------------------------------------------------------------
# quaternion helpers.  q = (w, x, y, z) throughout.
# ---------------------------------------------------------------


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by + ay * bw + az * bx - ax * bz,
            aw * bz + az * bw + ax * by - ay * bx)


def _qrot(q, v):
    """Rotate the 3-vector v by the unit quaternion q (v -> q v qbar)."""
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def _qaxis(axis, angle):
    """Unit quaternion for a rotation of `angle` about a unit axis."""
    h = 0.5 * angle
    sh = math.sin(h)
    return (math.cos(h), sh * axis[0], sh * axis[1], sh * axis[2])


# ---------------------------------------------------------------
# the homotopy itself
# ---------------------------------------------------------------


def nullhomotopy(s, t, billow=False):
    """The unit quaternion Dhat(s,t) of the double-twist nullhomotopy.

    s in [0, pi/2] is the stage -- s = 0 is the double twist, s = pi/2
    the constant loop -- and t in [0, 2pi] runs along the loop.  With
    billow=False this is the Pengelley-Ramras double-tipping homotopy,
    whose J component vanishes identically; with billow=True the K
    component is rotated by s toward J, giving the Francis-Kauffman
    homotopy, which uses all three axial directions.  The real part is
    the same either way.
    """
    cs = math.cos(s)
    ss = math.sin(s)
    h = math.sin(0.5 * t) ** 2
    w = 1.0 - 2.0 * cs * cs * h
    x = math.sin(2.0 * s) * h
    k = cs * math.sin(t)
    if billow:
        return (w, x, ss * k, cs * k)
    return (w, x, 0.0, k)


def single_twist(t):
    """The single 2*pi twist about K -- the loop that CANNOT be
    untangled.  Kept for contrast: it is the generator of
    pi_1(SO(3)) = Z/2, so no nullhomotopy of it exists."""
    h = 0.5 * t
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def stage_from_turn(turn):
    """Stage s of the homotopy that sweeps `turn` radians of rotation
    in total over one cycle.  Theta(s) = 4pi - 8s exactly, so this is
    a straight line: 4pi (720 degrees) is the stuck double twist and 0
    is the flat belt."""
    return min(HALF_PI, max(0.0, (FULL_TURN - turn) / 8.0))


def turn_from_stage(s):
    """Inverse of stage_from_turn: total rotation swept, in radians."""
    return FULL_TURN - 8.0 * s


def stages_for(frames, turn):
    """The stages to emit.  One frame uses the requested turn; more
    than one spans the whole homotopy evenly -- which, the mapping
    being linear, is also evenly spaced in degrees."""
    if frames <= 1:
        return [stage_from_turn(turn)]
    return [HALF_PI * k / (frames - 1) for k in range(frames)]


def ball_point(q, branch=0):
    """Map a unit quaternion to the solid-ball model of SO(3).

    The rotation is placed at (theta/pi) * axis, with theta in [0, pi]
    after choosing the representative with w >= 0.  The identity goes
    to the centre, a half-turn to the unit sphere.

    `branch` forces that choice, which matters on the seam where the
    real part vanishes: there the two representatives are opposite
    points of the boundary sphere, and rounding decides between them.
    A patch that approaches the seam from the positive side must keep
    taking the positive representative all the way to it, or its last
    row jumps a full diameter.  branch = +1 never negates, -1 always
    does, and the default 0 picks by the sign of w."""
    w, x, y, z = q
    if (w < 0.0 if branch == 0 else branch < 0):
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    w = max(-1.0, min(1.0, w))
    r = (2.0 * math.acos(w)) / math.pi
    return (r * x / n, r * y / n, r * z / n)


def seam_t(s):
    """Where the ball model's antipodal identification is crossed.

    The real part of Dhat vanishes at t1 = 2 arcsin(1/(sqrt2 cos s))
    and at t2 = 2pi - t1, which are real only for s <= pi/4.  Past
    that the loop never reaches a half-turn, and both collapse to pi,
    leaving no seam to cross."""
    c = math.cos(s)
    if c <= 1e-12:
        return math.pi, math.pi
    a = 1.0 / (math.sqrt(2.0) * c)
    if a >= 1.0:
        return math.pi, math.pi
    t1 = 2.0 * math.asin(a)
    return t1, TAU - t1


# ---------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------


def _fit_unit_cube(verts, scale=1.0):
    """Centre verts on the origin and scale so the largest half-extent
    is `scale` (the house rule: fit the 2 m cube)."""
    if not verts:
        return verts
    lo = [min(v[i] for v in verts) for i in range(3)]
    hi = [max(v[i] for v in verts) for i in range(3)]
    mid = [0.5 * (lo[i] + hi[i]) for i in range(3)]
    ext = max(max(abs(v[i] - mid[i]) for i in range(3)) for v in verts)
    if ext <= 1e-12:
        return [(0.0, 0.0, 0.0) for _ in verts]
    f = scale / ext
    return [tuple((v[i] - mid[i]) * f for i in range(3)) for v in verts]


def _fit_box(verts, centre, half, scale=1.0):
    """Scale about a GIVEN centre and half-extent, not the vertices' own
    bounding box.

    The ribbon modes must not be fitted to what they happen to span at
    one stage: doing that rescales and re-centres the whole object every
    time the stage changes, so the clamped end drifts and the hub
    breathes even though the geometry is anchored.  Callers pass a box
    computed once over the whole family (see ribbon_box) and every stage
    then sits in the same frame."""
    if not verts or half <= 1e-12:
        return list(verts)
    f = scale / half
    return [tuple((v[i] - centre[i]) * f for i in range(3))
            for v in verts]


def ribbon_box(build, samples=17):
    """Centre and half-extent covering every stage of a ribbon family.

    `build` takes a stage and returns its vertices.  Sampling the whole
    family once gives a frame that is the same at every stage, which is
    what keeps the picture still while the stage is scrubbed."""
    lo = [1e30] * 3
    hi = [-1e30] * 3
    for i in range(samples):
        for v in build(HALF_PI * i / (samples - 1)):
            for k in range(3):
                lo[k] = min(lo[k], v[k])
                hi[k] = max(hi[k], v[k])
    centre = tuple(0.5 * (lo[k] + hi[k]) for k in range(3))
    half = max(max(hi[k] - centre[k], centre[k] - lo[k])
               for k in range(3))
    return centre, half


def _grid_faces(ns, nt, base=0):
    """Quad faces of an ns x nt vertex grid indexed row-major."""
    faces = []
    for i in range(ns - 1):
        for j in range(nt - 1):
            a = base + i * nt + j
            faces.append([a, a + 1, a + nt + 1, a + nt])
    return faces


def _uv_sphere(radius=1.0, seg=48, rings=24):
    """A plain lat-long sphere, used for the ball's boundary."""
    verts, faces = [], []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(seg):
            th = TAU * j / seg
            verts.append((radius * math.sin(phi) * math.cos(th),
                          radius * math.sin(phi) * math.sin(th),
                          radius * math.cos(phi)))
    for i in range(rings):
        for j in range(seg):
            a = i * seg + j
            b = i * seg + (j + 1) % seg
            faces.append([a, b, b + seg, a + seg])
    return verts, faces


def _tangents(path, closed):
    n = len(path)
    out = []
    for i in range(n):
        if closed:
            a, b = path[(i - 1) % n], path[(i + 1) % n]
        else:
            a, b = path[max(i - 1, 0)], path[min(i + 1, n - 1)]
        d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        m = math.sqrt(sum(c * c for c in d))
        out.append((0.0, 0.0, 1.0) if m < 1e-12
                   else (d[0] / m, d[1] / m, d[2] / m))
    return out


def _tube(path, radius, sides=10, closed=False, cap=True):
    """Sweep a circular cross-section along a polyline using a
    parallel-transported frame (Hanson & Ma, TR-425): at each step
    rotate the reference normal by the turn angle about the common
    perpendicular of consecutive tangents, so the tube never spins
    where the path merely straightens.  A closed path gets TR-425's
    spin correction, distributing the holonomy the transport picks up
    around the loop so the last ring meets the first."""
    n = len(path)
    if n < 2:
        return [], []
    tang = _tangents(path, closed)

    t0 = tang[0]
    seed = (0.0, 0.0, 1.0) if abs(t0[2]) < 0.9 else (1.0, 0.0, 0.0)
    nx = (t0[1] * seed[2] - t0[2] * seed[1],
          t0[2] * seed[0] - t0[0] * seed[2],
          t0[0] * seed[1] - t0[1] * seed[0])
    m = math.sqrt(sum(c * c for c in nx))
    start = (1.0, 0.0, 0.0) if m < 1e-12 else tuple(c / m for c in nx)

    normals = [start]
    normal = start
    for i in range(1, n):
        a, b = tang[i - 1], tang[i]
        bx = (a[1] * b[2] - a[2] * b[1],
              a[2] * b[0] - a[0] * b[2],
              a[0] * b[1] - a[1] * b[0])
        bm = math.sqrt(sum(c * c for c in bx))
        if bm > 1e-12:
            axis = tuple(c / bm for c in bx)
            dot = max(-1.0, min(1.0, sum(a[k] * b[k] for k in range(3))))
            normal = _qrot(_qaxis(axis, math.acos(dot)), normal)
        normals.append(normal)

    if closed:
        # TR-425 spin correction: measure the angle the transported
        # normal has drifted after one circuit and unwind it linearly,
        # so the last ring lines up with the first.
        ref = normals[0]
        t_end = tang[0]
        cross = (t_end[1] * ref[2] - t_end[2] * ref[1],
                 t_end[2] * ref[0] - t_end[0] * ref[2],
                 t_end[0] * ref[1] - t_end[1] * ref[0])
        last = normals[-1]
        alpha = math.atan2(sum(last[k] * cross[k] for k in range(3)),
                           sum(last[k] * ref[k] for k in range(3)))
        normals = [_qrot(_qaxis(tang[i], -alpha * i / n), normals[i])
                   for i in range(n)]

    verts, faces = [], []
    for i in range(n):
        t, nrm, c = tang[i], normals[i], path[i]
        bi = (t[1] * nrm[2] - t[2] * nrm[1],
              t[2] * nrm[0] - t[0] * nrm[2],
              t[0] * nrm[1] - t[1] * nrm[0])
        for k in range(sides):
            a = TAU * k / sides
            ca, sa = math.cos(a), math.sin(a)
            verts.append((c[0] + radius * (ca * nrm[0] + sa * bi[0]),
                          c[1] + radius * (ca * nrm[1] + sa * bi[1]),
                          c[2] + radius * (ca * nrm[2] + sa * bi[2])))
    span = n if closed else n - 1
    for i in range(span):
        nxt = ((i + 1) % n) * sides
        for k in range(sides):
            a = i * sides + k
            b = i * sides + (k + 1) % sides
            faces.append([a, b, nxt + (k + 1) % sides, nxt + k])
    if cap and not closed:
        faces.append(list(range(sides - 1, -1, -1)))
        faces.append([(n - 1) * sides + k for k in range(sides)])
    return verts, faces


def _merge(target_v, target_f, verts, faces):
    base = len(target_v)
    target_v.extend(verts)
    target_f.extend([[i + base for i in f] for f in faces])


# ---------------------------------------------------------------
# mode: Dirac Ball
# ---------------------------------------------------------------


def build_ball_surface(res_s=48, res_t=96, billow=False):
    """The nullhomotopy swept out inside the solid-ball model of
    SO(3), as three patches split along the antipodal seam (see
    seam_t).  Flat for Pengelley-Ramras, whose axes are coplanar; a
    billowed solid for Francis-Kauffman."""
    verts, faces = [], []
    for region in (0, 1, 2):
        # The middle patch, where the real part is negative, exists
        # only for s <= pi/4 and tapers to a point there.
        smax = 0.25 * math.pi if region == 1 else HALF_PI
        rows = res_s if region != 1 else max(3, res_s // 2)
        branch = -1 if region == 1 else 1
        base = len(verts)
        for i in range(rows):
            s = smax * i / (rows - 1)
            t1, t2 = seam_t(s)
            a, b = ((0.0, t1), (t1, t2), (t2, TAU))[region]
            for j in range(res_t):
                t = a + (b - a) * j / (res_t - 1)
                verts.append(ball_point(nullhomotopy(s, t, billow),
                                        branch))
        faces.extend(_grid_faces(rows, res_t, base))
    return verts, faces


def loop_pieces(s, res_t=192, billow=False):
    """One stage of the homotopy as polylines in the ball model, split
    wherever the antipodal identification is crossed.  The double
    twist (s = 0) comes back in THREE pieces, not two: it passes
    through the centre at t = pi, between its two boundary jumps."""
    t1, t2 = seam_t(s)
    out = []
    for branch, (a, b) in zip((1, -1, 1),
                              ((0.0, t1), (t1, t2), (t2, TAU))):
        if b - a < 1e-9:
            continue
        n = max(3, int(round(res_t * (b - a) / TAU)) + 1)
        out.append([ball_point(nullhomotopy(s, a + (b - a) * j / (n - 1),
                                            billow), branch)
                    for j in range(n)])
    return out


def single_twist_pieces(res_t=192):
    """The un-untanglable 2*pi twist in the ball: straight out to the
    boundary along +K, in again at -K, back to the centre -- one
    diameter, traversed once."""
    out = []
    for branch, (a, b) in zip((1, -1), ((0.0, math.pi), (math.pi, TAU))):
        n = max(3, res_t // 2)
        out.append([ball_point(single_twist(a + (b - a) * j / (n - 1)),
                               branch)
                    for j in range(n)])
    return out


def build_ball(res_s=48, res_t=96, billow=False, loops=False,
               stages=None, radius=0.02, sides=10):
    """The Dirac Ball mode: either the swept surface, or the family of
    stage loops as tubes."""
    if not loops:
        return build_ball_surface(res_s, res_t, billow)
    verts, faces = [], []
    for s in (stages or [0.0]):
        for piece in loop_pieces(s, max(res_t * 2, 96), billow):
            _merge(verts, faces, *_tube(piece, radius, sides))
    return verts, faces


# ---------------------------------------------------------------
# mode: Belt Filmstrip
# ---------------------------------------------------------------


def belt_frame(s, res_t=120, billow=False, anchor='START'):
    """Core and width vector of the belt at stage s.

    The core is the running integral of R(s,t) applied to K and the
    width vector is R(s,t) applied to I, so the width is exactly
    perpendicular to the core tangent -- a rotation preserves the
    right angle between K and I.  The core has unit speed, hence arc
    length 2pi at every stage.  It is centred on its own mean so the
    belts of a filmstrip line up."""
    ts = [TAU * j / (res_t - 1) for j in range(res_t)]
    qs = [nullhomotopy(s, t, billow) for t in ts]
    dirs = [_qrot(q, (0.0, 0.0, 1.0)) for q in qs]
    core = [(0.0, 0.0, 0.0)]
    acc = [0.0, 0.0, 0.0]
    for j in range(1, res_t):
        dt = ts[j] - ts[j - 1]
        for k in range(3):
            acc[k] += 0.5 * dt * (dirs[j - 1][k] + dirs[j][k])
        core.append(tuple(acc))
    # Anchor on the FIRST point by default: one end of a belt is
    # clamped and stays put, which is how every depiction of the trick
    # is set up.  Centring on the mean instead lets both ends swing as
    # the stage changes, which reads as the whole belt moving.
    if anchor == 'MEAN':
        base = [sum(p[k] for p in core) / len(core) for k in range(3)]
    else:
        base = core[0]
    core = [tuple(p[k] - base[k] for k in range(3)) for p in core]
    return core, [_qrot(q, (1.0, 0.0, 0.0)) for q in qs], dirs


def build_belts(stages, res_t=120, billow=False, length=2.0,
                width=0.32, gap=0.55):
    """The filmstrip: one belt per stage, laid side by side along y.

    Belt 0 is the double twist -- a straight strip carrying 4 pi of
    twist -- and the last belt is straight and flat.  In between the
    core bows out, because the frame's K direction wanders."""
    verts, faces = [], []
    f = length / TAU
    hw = 0.5 * width
    n = len(stages)
    for k, s in enumerate(stages):
        core, wide, _ = belt_frame(s, res_t, billow)
        # Space the row along x.  Stacking along y would pile the
        # belts face to face -- y is the surface normal at the clamped
        # end -- so head on they would all superimpose.
        x0 = (k - 0.5 * (n - 1)) * gap
        base = len(verts)
        for j in range(len(core)):
            c, e = core[j], wide[j]
            verts.append((x0 + f * c[0] + hw * e[0],
                          f * c[1] + hw * e[1],
                          f * c[2] + hw * e[2]))
            verts.append((x0 + f * c[0] - hw * e[0],
                          f * c[1] - hw * e[1],
                          f * c[2] - hw * e[2]))
        for j in range(len(core) - 1):
            a = base + 2 * j
            faces.append([a, a + 1, a + 3, a + 2])
    return verts, faces


def build_rosette(s, belts=4, res_t=120, billow=False, length=4.0,
                  width=0.32, hub=0.35):
    """Dirac's own arrangement: `belts` belts radiating from a central
    hub, each carrying the same stage of the untangling, spaced evenly
    around the axis.

    This is the arrangement of Newman's 1942 scissors-and-string model,
    but not yet its content: each belt here is twisted about its OWN
    radius and the belts never interact, so what is drawn is one
    untangling repeated n times around the axis.  A hub turning about a
    single axis would instead wrap the belts around each other into
    Newman's braid, which is what makes three or more strands
    insoluble for an odd number of turns.  See BACKLOG.

    Each belt is built by the same frame construction as the sequence
    mode, then laid along a radius: its own axis is mapped to the
    outward direction and the copy is spun about the hub axis."""
    core, wide, _ = belt_frame(s, res_t, billow)
    f = length / TAU
    hw = 0.5 * width

    def place(v, ang):
        # local (x,y,z) -> outward x, tangential y, axial z
        x, y, z = v[2] + hub, v[1], -v[0]
        ca, sa = math.cos(ang), math.sin(ang)
        return (ca * x - sa * y, sa * x + ca * y, z)

    verts, faces = [], []
    for k in range(belts):
        ang = TAU * k / belts
        base = len(verts)
        for c, e in zip(core, wide):
            a = [f * c[i] + hw * e[i] for i in range(3)]
            b = [f * c[i] - hw * e[i] for i in range(3)]
            verts.append(place(a, ang))
            verts.append(place(b, ang))
        for j in range(len(core) - 1):
            q = base + 2 * j
            faces.append([q, q + 1, q + 3, q + 2])
    return verts, faces


def belt_self_distance(s, res_t=160, billow=False, length=4.0,
                       width=0.32):
    """Closest approach between non-adjacent points of the belt's core.

    This, not the end-to-end gap, is what says whether the ribbon passes
    through itself: the two ends can meet without the belt crossing, and
    the belt can cross while its ends are far apart (the efficient
    variant does exactly that near 421 degrees)."""
    core, _, _ = belt_frame(s, res_t, billow)
    f = length / TAU
    n = len(core)
    # Skip a window of about two belt widths of arc.  Neighbouring
    # samples are always close, so the window has to exclude them --
    # but sizing it as a fixed fraction of the belt would make the
    # answer scale with belt length and say nothing about the width it
    # has to clear.
    step = length / max(1, n - 1)
    skip = max(2, min(n // 3, int(2.0 * width / step) + 1))
    best = 1e30
    for i in range(0, n, 2):
        for j in range(i + skip, n, 2):
            d = sum((core[i][k] - core[j][k]) ** 2 for k in range(3))
            if d < best:
                best = d
    return f * math.sqrt(best)


def sphere_directions(n):
    """n directions spread over the sphere, by the Fibonacci spiral.

    The demonstrations send the belts out in all directions from the
    ball, not around one equator, so this is the default spread."""
    if n == 1:
        return [(1.0, 0.0, 0.0)]
    out = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for k in range(n):
        z = 1.0 - 2.0 * k / (n - 1) if n > 1 else 0.0
        z *= (n - 1) / (n + 1.0)          # keep off the exact poles
        r = math.sqrt(max(0.0, 1.0 - z * z))
        a = ga * k
        out.append((r * math.cos(a), r * math.sin(a), z))
    return out


def ring_directions(n):
    """n directions evenly around the equator."""
    return [(math.cos(TAU * k / n), math.sin(TAU * k / n), 0.0)
            for k in range(n)]


def _across(u):
    """Two unit vectors spanning the plane across the direction u."""
    seed = (0.0, 0.0, 1.0) if abs(u[2]) < 0.9 else (1.0, 0.0, 0.0)
    a = (u[1] * seed[2] - u[2] * seed[1],
         u[2] * seed[0] - u[0] * seed[2],
         u[0] * seed[1] - u[1] * seed[0])
    m = math.sqrt(sum(c * c for c in a))
    a = tuple(c / m for c in a)
    b = (u[1] * a[2] - u[2] * a[1],
         u[2] * a[0] - u[0] * a[2],
         u[0] * a[1] - u[1] * a[0])
    return a, b


def build_cage(radius=4.35, rings=6, seg=96, tube=0.012, sides=6):
    """The wireframe globe the belts are pinned to.

    In the films the outer ends are fastened to a sphere of great
    circles, and that cage is most of what makes the picture readable:
    it shows at a glance that the far ends are not going anywhere and
    that only the ball in the middle is turning."""
    verts, faces = [], []
    for i in range(rings):
        a = math.pi * i / rings
        ca, sa = math.cos(a), math.sin(a)
        path = [(radius * math.cos(TAU * j / seg) * ca,
                 radius * math.cos(TAU * j / seg) * sa,
                 radius * math.sin(TAU * j / seg)) for j in range(seg)]
        _merge(verts, faces, *_tube(path, tube, sides, closed=True))
    for i in range(1, rings):
        z = radius * math.cos(math.pi * i / rings)
        r = math.sqrt(max(0.0, radius * radius - z * z))
        path = [(r * math.cos(TAU * j / seg), r * math.sin(TAU * j / seg),
                 z) for j in range(seg)]
        _merge(verts, faces, *_tube(path, tube, sides, closed=True))
    return verts, faces


def build_twisted_rosette(hub_turn, belts=4, res_t=120, length=4.0,
                          width=0.32, hub=0.35, dirs=None, strands=1,
                          strand_gap=0.12):
    """The hub spun through `hub_turn`, winding the twist INTO the belts.

    This is the first half of the trick, and the half every animation
    opens with: the outer ends are pinned to something that does not
    move, the ball in the middle turns, and each belt takes up twist in
    proportion to how far round the ball has gone.  Nothing translates
    -- both ends of every belt stay exactly where they are, and only the
    cross-section rotates -- so this is the phase where the ends really
    are fixed.

    The twist runs linearly from none at the pinned outer end to the
    full hub angle at the ball, which is what an evenly-taken-up belt
    does.  At 720 degrees each belt carries the double twist that the
    untangling phase can then remove.

    A caveat worth stating: one rigid rotation about a single axis would
    twist only those belts lying ALONG that axis, and would wrap the
    ones across it instead.  Showing every belt twisting equally is the
    usual convention of the demonstrations rather than strict rigid-body
    motion -- it is the picture of the trick, drawn n times."""
    verts, faces = [], []
    dirs = dirs or ring_directions(belts)
    # Split the strap lengthwise into parallel ribbons.  The films do
    # this, and it is what makes the twist legible: a plain strap seen
    # edge-on is a line, whereas the stripes keep turning.
    hw = 0.5 * width / max(1, strands)
    for u3 in dirs:
        e1, e2 = _across(u3)
        for sidx in range(strands):
            off = ((sidx - 0.5 * (strands - 1))
                   * (1.0 + strand_gap) * 2.0 * hw) if strands > 1 else 0.0
            base = len(verts)
            for j in range(res_t):
                t = j / (res_t - 1)          # 0 pinned end, 1 at the hub
                r = hub + length * (1.0 - t)
                th = hub_turn * t
                c, sn = math.cos(th), math.sin(th)
                w = tuple(c * e1[i] + sn * e2[i] for i in range(3))
                centre = tuple(r * u3[i] + off * w[i] for i in range(3))
                verts.append(tuple(centre[i] + hw * w[i]
                                   for i in range(3)))
                verts.append(tuple(centre[i] - hw * w[i]
                                   for i in range(3)))
            for j in range(res_t - 1):
                q = base + 2 * j
                faces.append([q, q + 1, q + 3, q + 2])
    return verts, faces


def belt_end_gap(s, res_t=240, billow=False, length=4.0):
    """Distance between the two ends of a belt at stage s.

    The untangling is a homotopy of ROTATIONS -- Pengelley and Ramras
    are explicit that only the rotational positioning of each frame
    matters, not the translational component -- so a belt drawn with
    its tangent equal to the frame's own length vector has its far end
    at the integral of that vector, and nothing holds it still.  At the
    double twist and at the flat belt the ends are a full belt-length
    apart; in between they close up, and for the billowing variant they
    meet exactly at 360 degrees.  This is reported, not corrected:
    correcting it would mean drawing frames the homotopy does not have.
    """
    core, _, _ = belt_frame(s, res_t, billow)
    f = length / TAU
    return f * math.sqrt(sum((core[-1][k] - core[0][k]) ** 2
                             for k in range(3)))


# ---------------------------------------------------------------
# mode: S3 Lift
# ---------------------------------------------------------------


def _project(q, v):
    """Turn q by v, then stereographically project S^3 -> R^3 from -1."""
    w, x, y, z = _qmul(v, q)
    d = 1.0 + w
    if d < 1e-9:
        d = 1e-9
    return (x / d, y / d, z / d)


def lift_extent(res_s, res_t, billow, view_turn):
    """Largest and mean radius of the projected image, before fitting.

    The projected lobe grows like cot(view_turn/4), so a small turn
    throws out one spike that swallows the whole 2 m cube once the
    result is scaled to fit.  The ratio of these two is what the
    operator warns on -- it is the thing the user actually sees,
    unlike the 1+w that the pole guard measures."""
    hv = 0.5 * view_turn
    v = (math.cos(hv), 0.0, math.sin(hv), 0.0)
    rmax, total, n = 0.0, 0.0, 0
    for i in range(res_s):
        s = HALF_PI * i / (res_s - 1)
        for j in range(res_t):
            t = TAU * j / (res_t - 1)
            p = _project(nullhomotopy(s, t, billow), v)
            r = math.sqrt(sum(c * c for c in p))
            rmax = max(rmax, r)
            total += r
            n += 1
    return rmax, (total / n if n else 0.0)


def build_lift(res_s=48, res_t=96, billow=False, view_turn=1.7,
               loops=False, stages=None, radius=0.02, sides=10):
    """The lift of the homotopy to S^3, stereographically projected
    into R^3 from the pole -1.  The quaternion is first turned by
    `view_turn` about J, which lifts the image clear of that pole for
    both variants."""
    hv = 0.5 * view_turn
    v = (math.cos(hv), 0.0, math.sin(hv), 0.0)
    if loops:
        verts, faces = [], []
        for s in (stages or [0.0]):
            path = [_project(nullhomotopy(s, TAU * j / res_t, billow), v)
                    for j in range(res_t)]
            _merge(verts, faces, *_tube(path, radius, sides, closed=True))
        return verts, faces
    verts = []
    for i in range(res_s):
        s = HALF_PI * i / (res_s - 1)
        for j in range(res_t):
            t = TAU * j / (res_t - 1)
            verts.append(_project(nullhomotopy(s, t, billow), v))
    return verts, _grid_faces(res_s, res_t)


# ---------------------------------------------------------------
# Blender layer
# ---------------------------------------------------------------

if _IN_BLENDER:

    class MESH_OT_spinor_add(bpy.types.Operator):
        """Add the Dirac belt trick as static geometry: a nullhomotopy
        of the double twist in SO(3), in the ball model, as a strip of
        belts, or lifted to the 3-sphere"""
        bl_idname = "mesh.spinor_add"
        bl_label = "Spinor Belt Trick"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Mode",
            description="What to build from the untangling",
            items=[('BELTS', "Untangling Sequence",
                    "One belt per stage of the untangling, side by "
                    "side: the first is twisted twice, the last is "
                    "flat, and the ones between bow out as the twist "
                    "is traded away"),
                   ('ROSETTE', "Belt Rosette",
                    "The same stage of the untangling on several belts "
                    "radiating from a central hub, in the arrangement "
                    "Dirac used.  Each belt is twisted about its own "
                    "radius, so this shows one untangling repeated "
                    "around the axis rather than the belts braiding "
                    "with each other"),
                   ('BALL', "Dirac Ball",
                    "The untangling inside the solid-ball model of "
                    "SO(3), where the identity is the centre, a "
                    "half-turn is the unit sphere, and opposite "
                    "points of that sphere are the same rotation"),
                   ('LIFT', "S3 Lift",
                    "The untangling lifted to the 3-sphere of unit "
                    "quaternions and stereographically projected "
                    "into space")],
            default='BELTS')
        homotopy: EnumProperty(
            name="Homotopy",
            description="Which untangling to build",
            items=[('FK', "Francis-Kauffman",
                    "The billowing untangling, which spends all three "
                    "axial directions and so sweeps a genuinely "
                    "three-dimensional shape"),
                   ('PR', "Pengelley-Ramras",
                    "The efficient double-tipping untangling: "
                    "one-to-one except along the edges the endpoints "
                    "force, and onto the projective plane it lands "
                    "in.  Its axes are coplanar, so in the ball it is "
                    "a flat disc and its lift is a round bowl")],
            default='FK')
        style: EnumProperty(
            name="Style",
            description="Whether to fill the untangling in or draw it "
                        "as curves",
            items=[('SURFACE', "Surface",
                    "Mesh the whole sheet swept by the untangling"),
                   ('LOOPS', "Loops",
                    "Draw the individual stages as tubes.  Clearer "
                    "than the filled sheet, and the only legible "
                    "choice for the flat Pengelley-Ramras image")],
            default='SURFACE')

        frames: IntProperty(
            name="Stages", default=7, min=1, max=32,
            description="How many stages of the untangling to build, "
                        "spread evenly from the stuck double twist to "
                        "the flat belt.  Set to 1 to build a single "
                        "stage, chosen by Belt Turn")
        belt_turn: FloatProperty(
            name="Belt Turn", default=0.0,
            min=0.0, max=FULL_TURN, subtype='ANGLE',
            description="How far the buckle turns: 0 degrees is the "
                        "untangled belt, 720 degrees the double twist "
                        "that can be undone.  Used by the rosette, and "
                        "by the sequence when Stages is 1")

        res_s: IntProperty(
            name="Sheet Rows", default=48, min=3, max=256,
            description="Samples across the untangling, from the "
                        "double twist to the flat belt")
        res_t: IntProperty(
            name="Loop Samples", default=96, min=8, max=512,
            description="Samples around each loop")

        belt_length: FloatProperty(
            name="Belt Length", default=4.0, min=0.2, max=20.0,
            description="Arc length of each belt before fitting")
        belt_width: FloatProperty(
            name="Belt Width", default=0.32, min=0.01, max=4.0,
            description="Width of each belt")
        phase: EnumProperty(
            name="Phase",
            description="Which half of the trick to show",
            items=[('TWIST', "Twisting Up",
                    "The hub turns and winds twist into the belts, "
                    "their far ends pinned to the cage.  Nothing moves "
                    "but the turn itself"),
                   ('UNTANGLE', "Untangling",
                    "The hub is held still at its original orientation "
                    "while the belts are worked around to shed a double "
                    "twist.  This is the part that can only be done "
                    "from 720 degrees")],
            default='TWIST')
        hub_turn: FloatProperty(
            name="Hub Turn", default=TAU, min=0.0, max=FULL_TURN,
            subtype='ANGLE',
            description="How far the hub has been turned, winding twist "
                        "into every belt.  720 degrees is the double "
                        "twist the untangling phase can remove; 360 is "
                        "the single twist that it cannot")
        spread: EnumProperty(
            name="Spread",
            description="Which way the belts leave the hub",
            items=[('SPHERE', "All Directions",
                    "Spread over the sphere, as the demonstrations "
                    "show them"),
                   ('RING', "Ring",
                    "Evenly around one equator")],
            default='SPHERE')
        strands: IntProperty(
            name="Strands", default=3, min=1, max=12,
            description="Split each belt lengthwise into this many "
                        "parallel ribbons.  A plain strap seen edge-on "
                        "is a line; the stripes keep the twist visible "
                        "from any angle")
        show_cage: BoolProperty(
            name="Cage", default=True,
            description="Add the sphere of great circles the far ends "
                        "are pinned to.  It is what shows at a glance "
                        "that only the hub is turning")
        belts: IntProperty(
            name="Belts", default=4, min=1, max=24,
            description="How many belts radiate from the hub.  Newman "
                        "proved the trick needs three or more to have "
                        "any content: with two, an odd number of turns "
                        "comes undone as well")
        hub_radius: FloatProperty(
            name="Hub Radius", default=0.35, min=0.0, max=4.0,
            description="Clear space at the centre, where the belts "
                        "are anchored to the turning hub")
        show_hub: BoolProperty(
            name="Hub", default=True,
            description="Add a sphere for the object the belts are "
                        "anchored to -- Dirac's scissors, or the "
                        "dancer's hand")
        belt_gap: FloatProperty(
            name="Belt Spacing", default=1.7, min=0.05, max=8.0,
            description="Distance between neighbouring belts.  With "
                        "each belt clamped at the same end, the middle "
                        "stages swing out about two fifths of a belt "
                        "length to each side, so less than that and "
                        "neighbours interleave")

        view_turn: FloatProperty(
            name="View Turn", default=1.7, min=0.5, max=5.7,
            subtype='ANGLE',
            description="Rotation applied before projecting, which "
                        "moves the projection pole off the image.  "
                        "Small values throw one lobe far out and "
                        "flatten everything else")

        tube_radius: FloatProperty(
            name="Tube Radius", default=0.02, min=0.001, max=0.5,
            description="Radius of the drawn curves")
        tube_sides: IntProperty(
            name="Tube Sides", default=10, min=3, max=32,
            description="Cross-section segments of the drawn curves")

        show_loop: BoolProperty(
            name="Double-Twist Loop", default=True,
            description="Add the double twist itself as a tube.  In "
                        "the ball it runs out to the boundary, "
                        "re-enters at the opposite point, passes "
                        "through the centre and does it again")
        show_single: BoolProperty(
            name="Single Twist", default=False,
            description="Add the single 2*pi twist: one diameter, "
                        "traversed once.  This is the loop that "
                        "cannot be untangled, shown for contrast")
        show_ball: BoolProperty(
            name="Boundary Sphere", default=False,
            description="Add the unit sphere bounding the ball model, "
                        "whose opposite points are identified.  Give "
                        "it a transparent material to see inside")

        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Half-extent of the result; 1.0 fits the 2 m "
                        "cube")

        def _mesh_from(self, verts, faces, name, weld=True):
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            if weld:
                bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            me.update()
            return me

        def execute(self, context):
            billow = (self.homotopy == 'FK')
            loops = (self.style == 'LOOPS')
            stages = stages_for(self.frames, self.belt_turn)
            extras, note = [], ""

            if self.mode == 'BALL':
                verts, faces = build_ball(
                    self.res_s, self.res_t, billow, loops, stages,
                    self.tube_radius, self.tube_sides)
                if self.show_loop:
                    for piece in loop_pieces(0.0, max(self.res_t * 2, 96),
                                             billow):
                        extras.append(_tube(piece, self.tube_radius,
                                            self.tube_sides))
                if self.show_single:
                    for piece in single_twist_pieces(
                            max(self.res_t * 2, 96)):
                        extras.append(_tube(piece, self.tube_radius,
                                            self.tube_sides))
                if self.show_ball:
                    extras.append(_uv_sphere(1.0, 48, 24))
            elif self.mode == 'BELTS':
                verts, faces = build_belts(
                    stages, self.res_t, billow, self.belt_length,
                    self.belt_width, self.belt_gap)
                nb = len(stages)
                box_centre, box_half = ribbon_box(
                    lambda st, nb=nb: build_belts(
                        stages_for(nb, turn_from_stage(st)), 48, billow,
                        self.belt_length, self.belt_width,
                        self.belt_gap)[0])
            elif self.mode == 'ROSETTE' and self.phase == 'TWIST':
                dirs = (sphere_directions(self.belts)
                        if self.spread == 'SPHERE'
                        else ring_directions(self.belts))
                verts, faces = build_twisted_rosette(
                    self.hub_turn, self.belts, self.res_t,
                    self.belt_length, self.belt_width, self.hub_radius,
                    dirs, self.strands)
                rim = self.hub_radius + self.belt_length
                box_centre = (0.0, 0.0, 0.0)
                box_half = rim if self.show_cage else                     rim + 0.5 * self.belt_width
                if self.show_cage:
                    extras.append(build_cage(rim, 6, 96,
                                             0.004 * rim))
                if self.show_hub and self.hub_radius > 1e-6:
                    hv, hf = _uv_sphere(self.hub_radius * 0.85, 32, 16)
                    extras.append((hv, hf))
            elif self.mode == 'ROSETTE':
                verts, faces = build_rosette(
                    stage_from_turn(self.belt_turn), self.belts,
                    self.res_t, billow, self.belt_length,
                    self.belt_width, self.hub_radius)
                # The hub is the centre of a rosette, so put it on the
                # origin rather than the family's bounding-box middle:
                # the belts curl to one side, which would otherwise push
                # the hub off centre and tilt the whole figure.
                _, box_half = ribbon_box(
                    lambda st: build_rosette(
                        st, self.belts, 48, billow, self.belt_length,
                        self.belt_width, self.hub_radius)[0])
                box_centre = (0.0, 0.0, 0.0)
                if self.show_hub and self.hub_radius > 1e-6:
                    hv, hf = _uv_sphere(self.hub_radius * 0.85, 32, 16)
                    extras.append((hv, hf))
            else:
                verts, faces = build_lift(
                    self.res_s, self.res_t, billow, self.view_turn,
                    loops, stages, self.tube_radius, self.tube_sides)
                rmax, rmean = lift_extent(24, 48, billow, self.view_turn)
                ratio = rmax / rmean if rmean > 1e-9 else 0.0
                if ratio > 8.0:
                    note = ("projection is %.0fx lopsided; raise View "
                            "Turn" % ratio)

            # Fit the surface and its extras as one rigid body, so they
            # stay registered with each other.
            allv, spans = list(verts), []
            for ev, ef in extras:
                spans.append((len(allv), len(ev), ef))
                allv.extend(ev)
            if self.mode in ('BELTS', 'ROSETTE'):
                # One frame for the whole family, so scrubbing the stage
                # moves the belt and nothing else.  Fitting each stage to
                # its own bounding box instead would rescale and
                # re-centre the object every time, which makes the
                # clamped end and the hub appear to drift.
                allv = _fit_box(allv, box_centre, box_half, self.scale)
            else:
                allv = _fit_unit_cube(allv, self.scale)
            verts = allv[:len(verts)]

            # A ribbon has no duplicate vertices to merge, and welding
            # would fuse its two ends into a closed band at the stage
            # where they happen to touch.
            me = self._mesh_from(verts, faces, "Spinor",
                                 weld=self.mode not in ('BELTS',
                                                        'ROSETTE'))
            obj = bpy.data.objects.new(
                "Spinor %s" % self.mode.title(), me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

            for k, (off, cnt, ef) in enumerate(spans):
                sub = self._mesh_from(allv[off:off + cnt], ef,
                                      "Spinor Part")
                so = bpy.data.objects.new("Spinor Part %d" % k, sub)
                context.collection.objects.link(so)
                so.parent = obj

            if note:
                self.report({'WARNING'}, note)
            elif self.mode in ('BELTS', 'ROSETTE'):
                # The homotopy fixes rotations, not positions, so a
                # belt's far end moves as it is untangled.  Report the
                # gap rather than let it be a surprise: when it drops
                # below a belt width the ribbon is passing through
                # itself, which a real belt cannot do.
                # Report the stage actually shown; for a sequence, the
                # worst stage in it.
                if self.mode == 'ROSETTE' and self.phase == 'TWIST':
                    self.report(
                        {'INFO'},
                        "V=%d F=%d  hub turned %.0f deg  both ends of "
                        "every belt fixed"
                        % (len(me.vertices), len(me.polygons),
                           math.degrees(self.hub_turn)))
                    return {'FINISHED'}
                shown = ([stage_from_turn(self.belt_turn)]
                         if self.mode == 'ROSETTE' else stages)
                near = min(belt_self_distance(st, 160, billow,
                                              self.belt_length,
                                              self.belt_width)
                           for st in shown)
                s0 = shown[0]
                gap = belt_end_gap(s0, 240, billow, self.belt_length)
                msg = ("V=%d F=%d  turn %.0f deg  ends %.2f apart, "
                       "belt clears itself by %.2f"
                       % (len(me.vertices), len(me.polygons),
                          math.degrees(turn_from_stage(s0)), gap, near))
                if near < 0.5 * self.belt_width:
                    self.report({'WARNING'}, msg + " - the ribbon passes "
                                "through itself here; the untangling "
                                "fixes the frames, not where the belt "
                                "lies")
                else:
                    self.report({'INFO'}, msg)
            else:
                turn = math.degrees(turn_from_stage(stages[0]))
                self.report(
                    {'INFO'},
                    "V=%d F=%d  %d stage%s  first turn %.0f deg"
                    % (len(me.vertices), len(me.polygons), len(stages),
                       "" if len(stages) == 1 else "s", turn))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            ribbon = self.mode in ('BELTS', 'ROSETTE')
            lay.prop(self, 'mode')
            lay.prop(self, 'homotopy')
            if not ribbon:
                lay.prop(self, 'style')
            drawn = (not ribbon and self.style == 'LOOPS')

            if self.mode == 'ROSETTE':
                # A rosette is one configuration, so a turn drives it
                # directly and there is no stage count to compete with.
                # Which turn depends on the phase: winding the twist in
                # with the hub, or working it back out again.
                lay.prop(self, 'phase')
                lay.prop(self, 'belts')
                if self.phase == 'TWIST':
                    lay.prop(self, 'spread')
                    lay.prop(self, 'hub_turn')
                    lay.prop(self, 'strands')
                else:
                    lay.prop(self, 'belt_turn')
            elif self.mode == 'BELTS' or drawn:
                lay.prop(self, 'frames')
                sub = lay.row()
                sub.enabled = (self.frames <= 1)
                sub.prop(self, 'belt_turn')
            lay.prop(self, 'res_t')
            if not ribbon and not drawn:
                lay.prop(self, 'res_s')

            if ribbon:
                lay.prop(self, 'belt_length')
                lay.prop(self, 'belt_width')
            if self.mode == 'BELTS':
                lay.prop(self, 'belt_gap')
            if self.mode == 'ROSETTE':
                lay.prop(self, 'hub_radius')
                lay.prop(self, 'show_hub')
                if self.phase == 'TWIST':
                    lay.prop(self, 'show_cage')
            if self.mode == 'LIFT':
                lay.prop(self, 'view_turn')
            if self.mode == 'BALL':
                lay.prop(self, 'show_loop')
                lay.prop(self, 'show_single')
                lay.prop(self, 'show_ball')
            if drawn or (self.mode == 'BALL'
                         and (self.show_loop or self.show_single)):
                lay.prop(self, 'tube_radius')
                lay.prop(self, 'tube_sides')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator(MESH_OT_spinor_add.bl_idname,
                             icon='FORCE_MAGNETIC')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_spinor_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_spinor_add)


# ---------------------------------------------------------------
# standalone numeric checks
# ---------------------------------------------------------------


def _belt_twist(s, res_t=400, billow=False):
    """Total twist of the belt at stage s, and the worst deviation of
    its width vector from perpendicular to the core.

    Measured against the EXACT tangent R(s,t) applied to K, not
    against a chord of the integrated core: the right angle between
    K and I is what a rotation preserves, so it holds to rounding,
    whereas a chord carries the quadrature's own truncation error
    and would only report the sampling step back."""
    core, wide, tang = belt_frame(s, res_t, billow)
    perp = 0.0
    for j in range(len(core)):
        perp = max(perp, abs(sum(tang[j][k] * wide[j][k]
                                 for k in range(3))))
    total = 0.0
    for j in range(1, len(core)):
        a, b, t = wide[j - 1], wide[j], tang[j]
        cross = (a[1] * b[2] - a[2] * b[1],
                 a[2] * b[0] - a[0] * b[2],
                 a[0] * b[1] - a[1] * b[0])
        total += math.atan2(sum(cross[k] * t[k] for k in range(3)),
                            sum(a[k] * b[k] for k in range(3)))
    return abs(total), perp


def _selftest():
    ok = True

    # 1. Dhat is a unit quaternion everywhere, in both variants.
    worst = 0.0
    for billow in (False, True):
        for i in range(41):
            s = HALF_PI * i / 40.0
            for j in range(81):
                q = nullhomotopy(s, TAU * j / 80.0, billow)
                worst = max(worst,
                            abs(math.sqrt(sum(c * c for c in q)) - 1.0))
    print("1. unit norm: max |q|-1 = %.3g" % worst)
    ok = ok and worst < 1e-12

    # 2. Boundary conditions: s = 0 is the double twist about K,
    #    s = pi/2 is the constant loop, every stage is based at 1.
    bad = 0.0
    for j in range(81):
        t = TAU * j / 80.0
        q = nullhomotopy(0.0, t)
        bad = max(bad, abs(q[0] - math.cos(t)), abs(q[3] - math.sin(t)),
                  abs(q[1]), abs(q[2]))
    const = based = 0.0
    for i in range(41):
        s = HALF_PI * i / 40.0
        for billow in (False, True):
            for t in (0.0, TAU):
                q = nullhomotopy(s, t, billow)
                based = max(based, abs(q[0] - 1.0), abs(q[1]),
                            abs(q[2]), abs(q[3]))
            for j in range(81):
                q = nullhomotopy(HALF_PI, TAU * j / 80.0, billow)
                const = max(const, abs(q[0] - 1.0), abs(q[1]),
                            abs(q[2]), abs(q[3]))
    print("2. s=0 double twist %.3g / s=pi/2 constant %.3g / based %.3g"
          % (bad, const, based))
    ok = ok and max(bad, const, based) < 1e-12

    # 3. Pengelley-Ramras is planar in the ball; Francis-Kauffman is
    #    not.  Without this a bug collapsing FK onto PR would pass
    #    every other check here.
    flat = max(abs(ball_point(nullhomotopy(HALF_PI * i / 40.0,
                                           TAU * j / 80.0))[1])
               for i in range(41) for j in range(81))
    solid = max(abs(ball_point(nullhomotopy(HALF_PI * i / 40.0,
                                            TAU * j / 80.0, True))[1])
                for i in range(41) for j in range(81))
    print("3. ball y-extent: PR = %.3g (flat), FK = %.3g (solid)"
          % (flat, solid))
    ok = ok and flat < 1e-12 and solid > 0.1

    # 4. The ball model is a unit ball, and the double twist reaches
    #    its boundary.
    rmax = 0.0
    for i in range(41):
        s = HALF_PI * i / 40.0
        for j in range(81):
            for billow in (False, True):
                p = ball_point(nullhomotopy(s, TAU * j / 80.0, billow))
                rmax = max(rmax, math.sqrt(sum(c * c for c in p)))
    reach = max(math.sqrt(sum(c * c for c in ball_point(
        nullhomotopy(0.0, TAU * j / 400.0)))) for j in range(401))
    print("4. ball radius max |p| = %.6f (<=1), twist reaches %.6f"
          % (rmax, reach))
    ok = ok and rmax <= 1.0 + 1e-9 and reach > 0.999

    # 5. The double twist splits into THREE pieces at TWO sign flips:
    #    it passes through the centre at t = pi between its jumps.
    t1, t2 = seam_t(0.0)
    pieces = loop_pieces(0.0, 400)
    mid = math.sqrt(sum(c * c for c in
                        ball_point(nullhomotopy(0.0, math.pi))))
    print("5. double twist: %d pieces, seam at %.4f / %.4f "
          "(pi/2, 3pi/2), centre at t=pi |p|=%.3g"
          % (len(pieces), t1, t2, mid))
    ok = (ok and len(pieces) == 3
          and abs(t1 - 0.5 * math.pi) < 1e-12
          and abs(t2 - 1.5 * math.pi) < 1e-12 and mid < 1e-12)
    # the single twist crosses once, so it comes back in two pieces
    ok = ok and len(single_twist_pieces(200)) == 2

    # 6. No face of the ball mesh spans the antipodal identification.
    #    The bug this catches joined a boundary point to its opposite,
    #    drawing a face straight across a diameter of the ball.
    #
    #    Two things are asserted, neither of them a magic length.  An
    #    edge "spans the seam" exactly when both ends sit on the
    #    boundary sphere and are opposite each other, so that is tested
    #    directly and is resolution-independent.  The longest edge is
    #    then required to shrink under refinement: a seam crossing
    #    stays pinned near a full diameter however fine the grid gets,
    #    while the honest worst edge -- at the cone point where the
    #    middle patch tapers away -- refines like the square root of
    #    the step.
    for billow in (False, True):
        spans, lengths = 0, []
        for res_s, res_t in ((24, 48), (96, 192)):
            v, f = build_ball_surface(res_s, res_t, billow)
            step = 0.0
            for fc in f:
                for i in range(len(fc)):
                    a, b = v[fc[i]], v[fc[(i + 1) % len(fc)]]
                    step = max(step, math.sqrt(sum((a[k] - b[k]) ** 2
                                                   for k in range(3))))
                    ra = math.sqrt(sum(c * c for c in a))
                    rb = math.sqrt(sum(c * c for c in b))
                    if ra > 0.9 and rb > 0.9 and math.sqrt(
                            sum((a[k] + b[k]) ** 2
                                for k in range(3))) < 0.2:
                        spans += 1
            lengths.append(step)
        print("6. ball %s: %d edges across the identification, "
              "longest edge %.4f -> %.4f on refinement"
              % ("FK" if billow else "PR", spans, lengths[0],
                 lengths[1]))
        ok = ok and spans == 0 and lengths[1] < 0.75 * lengths[0]

    # 7. The belt's twist runs from 4 pi at the double twist to 0 at
    #    the end, and the ribbon's arc length is the same at every
    #    stage -- the core is built from a unit vector, so an error in
    #    the quadrature would show up here.
    #
    #    NOTE what is NOT asserted: that the width is perpendicular to
    #    the tangent.  Both come from the same rotation applied to K and
    #    to I, so their being perpendicular tests only that _qrot is a
    #    rotation.  The quantity that matters is printed instead -- how
    #    much the frame turns about the ribbon's own NORMAL.  A real
    #    belt can twist, and can bend about its width, but cannot bend
    #    in its own plane; this ribbon does, which is why it reads as a
    #    curved strip rather than a strap.  It is the honest limit of
    #    developing a rotation-only homotopy into a surface.
    for billow in (False, True):
        worst_perp, tw0, tw1 = 0.0, 0.0, 0.0
        for k in range(9):
            tw, perp = _belt_twist(HALF_PI * k / 8.0, 400, billow)
            worst_perp = max(worst_perp, perp)
            if k == 0:
                tw0 = tw
            if k == 8:
                tw1 = tw
        print("7. belt %s: twist %.4f -> %.4f (4pi = %.4f)"
              % ("FK" if billow else "PR", tw0, tw1, FULL_TURN))
        ok = ok and abs(tw0 - FULL_TURN) < 1e-6 and tw1 < 1e-9

        # How far the frame turns about the ribbon's own normal, and
        # about its width, per stage.  Reported, not asserted: a belt
        # would have the normal term at zero.
        wj = wi = 0.0
        for k in range(1, 9):
            st = HALF_PI * k / 8.0
            prev = None
            for j in range(401):
                q = nullhomotopy(st, TAU * j / 400.0, billow)
                if prev is not None:
                    d = _qmul((prev[0], -prev[1], -prev[2], -prev[3]), q)
                    wi += abs(d[1])
                    wj += abs(d[2])
                prev = q
        print("7. belt %s: frame turns %.2f about the width, %.2f about "
              "the normal (a strap would have 0 about the normal)"
              % ("FK" if billow else "PR", 2.0 * wi, 2.0 * wj))

        # Arc length is stage-independent: the core integrates a unit
        # vector, so every stage must come out the same length.
        lens = []
        for k in range(5):
            core, _, _ = belt_frame(HALF_PI * k / 4.0, 600, billow)
            lens.append(sum(math.sqrt(sum((core[j + 1][m] - core[j][m])
                                          ** 2 for m in range(3)))
                            for j in range(len(core) - 1)))
        print("7. belt %s: arc length %.4f..%.4f over the stages "
              "(2pi = %.4f)"
              % ("FK" if billow else "PR", min(lens), max(lens), TAU))
        ok = ok and max(lens) - min(lens) < 1e-3 and abs(lens[0] - TAU) < 1e-3

        # The polyline's chords are not the true tangent, and the gap
        # is the quadrature's truncation error rather than a defect --
        # so require that refining actually shrinks it.
        chord = []
        for res in (200, 400, 800):
            core, wide, _ = belt_frame(0.3, res, billow)
            tang = _tangents(core, False)
            chord.append(max(abs(sum(tang[j][k] * wide[j][k]
                                     for k in range(3)))
                             for j in range(len(core))))
        print("7. belt %s: chord deviation %.2e -> %.2e -> %.2e "
              "(refines away)"
              % ("FK" if billow else "PR", chord[0], chord[1],
                 chord[2]))
        ok = (ok and chord[2] < chord[1] < chord[0]
              and chord[2] < 0.5 * chord[0])

    # 7b. The rosette's belts are congruent copies spaced evenly about
    #     the axis, and the reported end gap matches the belt.
    for billow in (False, True):
        v, f = build_rosette(stage_from_turn(FULL_TURN), 5, 60, billow)
        per = len(v) // 5
        rad = []
        for k in range(5):
            block = v[k * per:(k + 1) * per]
            rad.append(sorted(round(math.hypot(q[0], q[1]), 9)
                              for q in block))
        same = all(rad[k] == rad[0] for k in range(5))
        # every belt must START on the hub -- its inner end is buckled
        # to the turning object, so that is the end that stays put
        starts = max(abs(math.hypot(v[k * per][0], v[k * per][1]) - 0.35)
                     for k in range(5))
        gap0 = belt_end_gap(stage_from_turn(FULL_TURN), 240, billow)
        gapm = belt_end_gap(stage_from_turn(0.6 * FULL_TURN), 240,
                            billow)
        print("7b. rosette %s: 5 congruent belts=%s, anchored on the "
              "hub to %.1e, end gap %.2f at 720 deg vs %.2f mid-way"
              % ("FK" if billow else "PR", same, starts, gap0, gapm))
        ok = ok and same and starts < 1e-9 and gap0 > gapm

    # 8. Every mode builds a finite mesh inside the 2 m cube, with
    #    every face index in range -- one stage and several, both
    #    styles.
    cases = []
    for billow in (False, True):
        tag = "FK" if billow else "PR"
        cases += [
            ("ball-surf-" + tag, build_ball(16, 32, billow, False)),
            ("ball-loops-" + tag, build_ball(16, 32, billow, True,
                                             stages_for(5, FULL_TURN))),
            ("ball-one-" + tag, build_ball(16, 32, billow, True,
                                           stages_for(1, 0.6 * FULL_TURN))),
            ("belts-" + tag, build_belts(stages_for(5, FULL_TURN), 40,
                                         billow)),
            ("belt-one-" + tag, build_belts(
                stages_for(1, 0.6 * FULL_TURN), 40, billow)),
            ("rosette-" + tag, build_rosette(
                stage_from_turn(FULL_TURN), 4, 40, billow)),
            ("rosette1-" + tag, build_rosette(
                stage_from_turn(0.0), 1, 40, billow)),
            ("lift-surf-" + tag, build_lift(16, 32, billow, 1.7, False)),
            ("lift-loops-" + tag, build_lift(16, 32, billow, 1.7, True,
                                             stages_for(5, FULL_TURN)))]
    for name, (v, f) in cases:
        v = _fit_unit_cube(v)
        fin = all(all(math.isfinite(c) for c in p) for p in v)
        ext = max(max(abs(c) for c in p) for p in v)
        nmax = max(max(i for i in fc) for fc in f)
        print("8. %-14s verts=%5d faces=%5d finite=%s extent=%.4f"
              % (name, len(v), len(f), fin, ext))
        ok = ok and fin and ext <= 1.0 + 1e-9 and nmax < len(v)

    # 8b. The ribbon frame does not depend on the stage.  This is the
    #     check for the bug the user actually saw: fitting each stage to
    #     its own bounding box rescaled and re-centred the whole object
    #     every time the stage moved, so the clamped end drifted.
    for billow in (False, True):
        c, h = ribbon_box(lambda st: build_rosette(
            st, 4, 48, billow, 4.0, 0.32, 0.35)[0])
        anchors = []
        for k in range(9):
            v = build_rosette(HALF_PI * k / 8.0, 4, 48, billow,
                              4.0, 0.32, 0.35)[0]
            anchors.append(_fit_box(v, c, h, 1.0)[0])
        drift = max(math.sqrt(sum((a[m] - anchors[0][m]) ** 2
                                  for m in range(3)))
                    for a in anchors)
        print("8b. rosette %s: anchor drifts %.2e across the stages "
              "after fitting" % ("FK" if billow else "PR", drift))
        ok = ok and drift < 1e-12

    # 9. Parallel transport is a rotation, so the tube cannot stretch
    #    its cross-section.
    path = [(math.cos(a), math.sin(a), 0.3 * a)
            for a in [TAU * k / 60.0 for k in range(61)]]
    tv, _ = _tube(path, 0.1, 8, cap=False)
    err = 0.0
    for i, c in enumerate(path):
        for k in range(8):
            p = tv[i * 8 + k]
            err = max(err, abs(math.sqrt(sum((p[m] - c[m]) ** 2
                                             for m in range(3))) - 0.1))
    print("9. tube radius error = %.3g" % err)
    ok = ok and err < 1e-9

    # 10. The lift is not lopsided at the default View Turn, and a
    #     small turn is (which is what the operator warns about).
    for billow in (False, True):
        rmax, rmean = lift_extent(24, 48, billow, 1.7)
        bad_max, bad_mean = lift_extent(24, 48, billow, 0.1)
        print("10. lift %s: extent ratio %.2f at 1.7 rad, %.2f at 0.1"
              % ("FK" if billow else "PR", rmax / rmean,
                 bad_max / bad_mean))
        ok = ok and rmax / rmean < 8.0 and bad_max / bad_mean > 8.0

    # 11. Degrees are a linear reading of the stage, and the same
    #     reading for both variants -- the billow tips the axis but
    #     leaves the real part, hence the rotation angle, untouched.
    lin = wdiff = 0.0
    for i in range(201):
        s = HALF_PI * i / 200.0
        best = max(2.0 * math.acos(max(-1.0, min(
            1.0, nullhomotopy(s, TAU * j / 400.0)[0])))
            for j in range(401))
        lin = max(lin, abs(2.0 * best - turn_from_stage(s)))
        for j in range(41):
            t = TAU * j / 40.0
            wdiff = max(wdiff, abs(nullhomotopy(s, t)[0]
                                   - nullhomotopy(s, t, True)[0]))
    rt = max(abs(stage_from_turn(turn_from_stage(HALF_PI * i / 50.0))
                 - HALF_PI * i / 50.0) for i in range(51))
    print("11. turn linear in stage %.3g, PR/FK agree %.3g, "
          "round trip %.3g" % (lin, wdiff, rt))
    ok = ok and lin < 1e-4 and wdiff == 0.0 and rt < 1e-12

    # 12. Every operator property is reachable in the panel.
    #     draw() is hand-written and the properties are declared far
    #     away from it, so a new one is easy to add and then never
    #     show.  That shipped once: the rosette's Phase, Spread, Hub
    #     Turn and Strands were built and wired into execute() but left
    #     out of draw(), so the panel offered a Belt Turn that drove
    #     nothing.  Nothing else in the suite looks at the UI.
    import re
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    declared = set(re.findall(
        r"^        (\w+): (?:Int|Float|Enum|Bool)Property",
        src, re.M))
    body = src[src.index("        def draw(self, context):"):
               src.index("    def _menu_func")]
    shown = set(re.findall(r"prop\(self, '(\w+)'\)", body))
    missing = sorted(declared - shown)
    print("12. %d properties declared, %d reachable in the panel%s"
          % (len(declared), len(declared & shown),
             "" if not missing else "; MISSING: " + ", ".join(missing)))
    ok = ok and not missing

    assert ok
    print("spinor standalone tests passed")
