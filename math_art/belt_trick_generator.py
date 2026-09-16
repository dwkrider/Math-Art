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
# cube.  ALL SIX belts use the SAME field F, and that is the whole
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
# The belt leaves each face orthogonally because g'(1) = 0: the
# sideways part of the tangent is proportional to dF/ds, which
# vanishes there, leaving only the radial part.  The last slice sits
# on the sphere through the face's chord, and a short flat stub in the
# cube's frame joins that arc to the face itself.
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
# angle is then smoothed along the belt over about half a belt width,
# which caps the twist rate where the bending direction swings quickly
# through an inflection.  The result is continuous in the turn, exactly
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
    from bpy.props import (IntProperty, FloatProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

from .spinor_generator import _qmul, _qrot, _qaxis, build_block, build_cage

TAU = 2.0 * math.pi
FULL_TURN = 2.0 * TAU

# How readily the width follows the bending (see "THE WIDTH OF A
# STRAP" above).  KAPPA sets the bending, relative to the radial
# progress, at which the perpendicular width takes over from the
# frame's; SMOOTH is the smoothing length of the deviation angle in
# belt widths; TAPER is the fraction of the belt over which the width
# is eased back to the face's own before the stub.
KAPPA = 0.4
SMOOTH = 0.5
TAPER = 0.04

_Z = (0.0, 0.0, 1.0)

# (face normal, width direction).  The widths are forced by the field:
# at the cube every belt bends about m = y, so the belts along x and z
# take their width along y; the belts along y only twist there and
# take z.  This is also a packing that keeps the arcs 90 - 2 lambda
# apart on every sphere.
FACES = [
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
]


# ---------------------------------------------------------------
# the frame field
# ---------------------------------------------------------------


def smootherstep(x):
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


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


def attach_radius(half, width):
    """Radius of the sphere whose chord of length `width` lies in the
    face plane at distance `half`: where the last belt slice sits."""
    return math.sqrt(half * half + 0.25 * width * width)


def belt_rows(psi, u, w, r_out=1.0, half=0.15, width=0.16, reach=0.5,
              ns=160, nlam=7, nstub=1, kappa=KAPPA, smooth=SMOOTH,
              taper=TAPER):
    """Cross-section rows of one belt, cage end first, cube end last.

    Each of the first ns rows is an arc of the sphere r(s) with a chord
    of exactly `width`; the last nstub rows are the flat stub in the
    cube's frame joining that arc to the face (its edge points coincide
    with the arc's, since the chord lies in the face plane, so one row
    is enough and the edge quads collapse to triangles).  Returns (rows, info)
    where info holds the centre line, its tangent and the width vector
    for the checks."""
    hs = attach_radius(half, width)
    dr = hs - r_out
    sig_a = 1.0 - reach
    h = 1.0 / (ns - 1)
    sigma = [i * h for i in range(ns)]
    r = [r_out + dr * s for s in sigma]
    U, W = [], []
    for s in sigma:
        g = smootherstep((s - sig_a) / reach)
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
    # the perpendicular width, sign carried from the cube end
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
        # unwrap along the belt
        while ang - prev > math.pi:
            ang -= TAU
        while ang - prev < -math.pi:
            ang += TAU
        alpha.append(ang)
        prev = ang
    # cap the twist rate: smooth the deviation angle along the belt
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
            for i in range(ns):
                tp = (sigma[i] - (1.0 - taper)) / taper
                tp = 0.0 if tp < 0.0 else (1.0 if tp > 1.0 else tp)
                tp = tp * tp * (3.0 - 2.0 * tp)
                alpha[i] = (1.0 - tp) * sm[i] + tp * alpha[-1]
    wv = [tuple(math.cos(alpha[i]) * W[i][k] + math.sin(alpha[i]) * Fv[i][k]
                for k in range(3)) for i in range(ns)]
    # the slices
    rows = []
    lam_last = []
    for i in range(ns):
        lam_max = math.asin(min(1.0, width / (2.0 * r[i])))
        row = []
        for j in range(nlam):
            lam = lam_max * (-1.0 + 2.0 * j / (nlam - 1))
            cl, sl = math.cos(lam), math.sin(lam)
            row.append(tuple(r[i] * (cl * U[i][k] + sl * wv[i][k])
                             for k in range(3)))
            if i == ns - 1:
                lam_last.append(lam)
        rows.append(row)
    # the stub: flat, in the cube's frame, from the arc to the face
    rc = _qaxis(_Z, 2.0 * psi)
    for kk in range(1, nstub + 1):
        t = kk / nstub
        row = []
        for lam in lam_last:
            y = hs * math.sin(lam)
            x0 = hs * math.cos(lam)
            x = x0 + t * (half - x0)
            row.append(_qrot(rc, tuple(x * u[k] + y * w[k] for k in range(3))))
        rows.append(row)
    T = [tuple(dr * U[i][k] + Ttan[i][k] for k in range(3)) for i in range(ns)]
    info = dict(sigma=sigma, r=r, U=U, W=W, wv=wv, T=T, tn=tn, hs=hs)
    return rows, info


def build_belts(psi, r_out=1.0, half=0.15, width=0.16, reach=0.5,
                ns=160, nlam=7, nstub=1):
    """All six belts as one mesh: (verts, faces, face_belt_index)."""
    verts, faces, mats = [], [], []
    for bi, (u, w) in enumerate(FACES):
        rows, _ = belt_rows(psi, u, w, r_out, half, width, reach,
                            ns, nlam, nstub)
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


def cube_gap(half, width):
    """Angular gap (radians) between neighbouring belt arcs on the
    innermost sphere, and the same as a distance."""
    hs = attach_radius(half, width)
    lam = math.asin(min(1.0, width / (2.0 * hs)))
    gap = 0.5 * math.pi - 2.0 * lam
    return gap, 2.0 * hs * math.sin(0.5 * max(0.0, gap))


# ---------------------------------------------------------------
# Blender operator
# ---------------------------------------------------------------

if _IN_BLENDER:

    _COLOURS = [(0.85, 0.45, 0.45), (0.45, 0.75, 0.45),
                (0.50, 0.50, 0.90), (0.90, 0.80, 0.30),
                (0.30, 0.80, 0.80), (0.80, 0.40, 0.80)]

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
        """Add the Dirac belt trick: a spinning cube with a belt on
        every face, the belts twisting and untangling with a period of
        two full turns.  Animate Turn from 0 to 720 degrees"""
        bl_idname = "mesh.belt_trick_add"
        bl_label = "Dirac Belt Trick"
        bl_options = {'REGISTER', 'UNDO'}

        turn: FloatProperty(
            name="Turn", default=math.radians(300.0), min=0.0,
            max=FULL_TURN, subtype='ANGLE',
            description="How far the cube has turned about the vertical "
                        "axis.  The belts return to their starting "
                        "state at 720 degrees, and not at 360; keyframe "
                        "this from 0 to 720 for one cycle")
        cube_size: FloatProperty(
            name="Cube Size", default=0.3, min=0.02, max=1.2,
            description="Edge length of the cube the belts are fastened "
                        "to")
        belt_width: FloatProperty(
            name="Belt Width", default=0.16, min=0.01, max=1.0,
            description="Width of each belt.  Must be less than about "
                        "one and a half times the cube's half-edge, or "
                        "neighbouring belts meet at the cube")
        reach: FloatProperty(
            name="Reach", default=0.5, min=0.1, max=0.95,
            description="Fraction of each belt, measured in from the "
                        "cube, that takes part in the untangling; the "
                        "rest lies straight to the cage.  Smaller values "
                        "coil the belts more tightly round the cube")
        thickness: FloatProperty(
            name="Thickness", default=0.012, min=0.0, max=0.1,
            description="Belt thickness, applied as a Solidify modifier "
                        "centred on the strip; zero for a bare surface")
        resolution: IntProperty(
            name="Resolution", default=160, min=24, max=800,
            description="Samples along each belt")
        across: IntProperty(
            name="Across", default=7, min=2, max=24,
            description="Samples across each belt")
        show_cube: BoolProperty(
            name="Cube", default=True,
            description="Add the cube, turned with the belt roots")
        show_cage: BoolProperty(
            name="Cage", default=True,
            description="Add the sphere of great circles the far ends "
                        "are pinned to")
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Radius of the cage; 1.0 fits the 2 m cube")

        def _mesh_from(self, verts, faces, name):
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
                p.use_smooth = True
            me.update()
            return me

        def execute(self, context):
            psi = 0.5 * self.turn
            half = 0.5 * self.cube_size
            verts, faces, mats = build_belts(
                psi, 1.0, half, self.belt_width, self.reach,
                self.resolution, self.across)
            sc = self.scale
            verts = [(v[0] * sc, v[1] * sc, v[2] * sc) for v in verts]
            me = self._mesh_from(verts, faces, "Belt Trick")
            for i, rgb in enumerate(_COLOURS):
                me.materials.append(_material("Belt %d" % (i + 1), rgb))
            for p, mi in zip(me.polygons, mats):
                p.material_index = mi
            obj = bpy.data.objects.new("Belt Trick", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0.0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness * sc
                mod.offset = 0.0
                mod.use_even_offset = True
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

            parts = []
            if self.show_cube:
                parts.append(("Belt Trick Cube",
                              build_block(half * sc, 2.0 * psi, _Z)))
            if self.show_cage:
                parts.append(("Belt Trick Cage",
                              build_cage(sc, 6, 96, 0.004 * sc)))
            for name, (pv, pf) in parts:
                sub = self._mesh_from(pv, pf, name)
                so = bpy.data.objects.new(name, sub)
                context.collection.objects.link(so)
                so.parent = obj

            gap, dist = cube_gap(half, self.belt_width)
            msg = ("V=%d F=%d  turn %.0f deg  neighbouring belts %.0f deg "
                   "(%.3f) apart at the cube"
                   % (len(me.vertices), len(me.polygons),
                      math.degrees(self.turn), math.degrees(gap),
                      dist * sc))
            if gap <= 0.0:
                self.report({'WARNING'}, msg + " - belts overlap; narrow "
                            "the belt or enlarge the cube")
            else:
                self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'turn')
            lay.prop(self, 'cube_size')
            lay.prop(self, 'belt_width')
            lay.prop(self, 'reach')
            lay.prop(self, 'thickness')
            lay.prop(self, 'resolution')
            lay.prop(self, 'across')
            lay.prop(self, 'show_cube')
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

    # 2. the geometry: period, non-return, ends, orthogonality, width
    half, width, ns, nlam = 0.15, 0.16, 81, 5
    hs = attach_radius(half, width)
    for psi in psis[:12]:
        for bi, (u, w) in enumerate(FACES):
            rows, info = belt_rows(psi, u, w, 1.0, half, width, 0.5,
                                   ns, nlam)
            rows2, _ = belt_rows(psi + TAU, u, w, 1.0, half, width, 0.5,
                                 ns, nlam)
            rows3, _ = belt_rows(psi + math.pi, u, w, 1.0, half, width,
                                 0.5, ns, nlam)
            d720 = max(_norm(tuple(a[k] - b[k] for k in range(3)))
                       for ra, rb in zip(rows, rows2) for a, b in zip(ra, rb))
            assert d720 < 1e-11, "belt not back after 720 degrees: %g" % d720
            d360 = max(_norm(tuple(a[k] - b[k] for k in range(3)))
                       for ra, rb in zip(rows, rows3) for a, b in zip(ra, rb))
            if bi == 0:
                assert d360 > 0.5, "belt back after 360 degrees"
            # every sphere row lies on its sphere, radii strictly decrease
            for i in range(ns):
                for p in rows[i]:
                    assert abs(_norm(p) - info['r'][i]) < 1e-12
                if i:
                    assert info['r'][i] < info['r'][i - 1]
            # cage end: identity frame, on the cage, width w
            assert _norm(tuple(rows[0][nlam // 2][k] - u[k]
                               for k in range(3))) < tol
            edge = tuple(rows[0][-1][k] - rows[0][0][k] for k in range(3))
            assert abs(_norm(edge) - width) < 1e-12
            assert abs(abs(_dot(_unit(edge), w)) - 1.0) < 1e-12
            # cube end: the stub's last row lies in the turned face plane,
            # centred on the face, with the face's own width direction
            rc = _qaxis(_Z, 2.0 * psi)
            nu, nw = _qrot(rc, u), _qrot(rc, w)
            for p in rows[-1]:
                assert abs(_dot(p, nu) - half) < 1e-12, "stub off the face"
            edge = tuple(rows[-1][-1][k] - rows[-1][0][k] for k in range(3))
            assert abs(_norm(edge) - width) < 1e-12
            assert abs(abs(_dot(_unit(edge), nw)) - 1.0) < 1e-12
            # width chord exactly `width` on every row, and the belt
            # leaves the face along its normal
            for i in range(ns):
                edge = tuple(rows[i][-1][k] - rows[i][0][k] for k in range(3))
                assert abs(_norm(edge) - width) < 1e-12
            # (the last chord's tilt is a discretisation residual that
            # falls as the cube of the step: 0.05 deg at 81 samples,
            # 0.007 at 160, 0.0008 at 320)
            last = info['U'][-1]
            prev = info['U'][-2]
            assert math.degrees(math.acos(min(1.0, _dot(last, prev)))) < 0.1
            assert abs(abs(_dot(_unit(info['T'][-1]), nu)) - 1.0) < 1e-12
            # shear of the strip stays moderate
            for i in range(1, ns - 1):
                t = _unit(info['T'][i])
                sh = math.degrees(math.asin(min(1.0, abs(_dot(t, info['wv'][i])))))
                assert sh < 50.0, "shear %.1f deg" % sh

    # 3. non-intersection: on every sphere the six arcs stay apart by
    #    the packing gap, at every turn sampled
    gap, dist = cube_gap(half, width)
    assert gap > 0.0
    for psi in psis[:12]:
        allrows = [belt_rows(psi, u, w, 1.0, half, width, 0.5, ns, nlam)[0]
                   for u, w in FACES]
        for i in range(ns):
            for a in range(6):
                for b in range(a + 1, 6):
                    for p in allrows[a][i]:
                        for q in allrows[b][i]:
                            d = _norm(tuple(p[k] - q[k] for k in range(3)))
                            assert d > 0.98 * dist, \
                                "belts %d,%d within %.3f" % (a, b, d)

    # 4. the reported gap: 33.9 degrees for the defaults (lambda = 28.1)
    assert abs(math.degrees(gap) - 33.9) < 0.3
    print("belt_trick_generator: self-test OK")
