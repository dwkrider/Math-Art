
# Delaunay Surfaces Generator for Blender
#
# The DELAUNAY SURFACES are the surfaces of revolution of constant mean
# curvature.  Delaunay (1841) classified them completely and gave a
# beautiful mechanical construction: roll a conic along a line without
# slipping, and the path traced by a FOCUS is the meridian of a CMC
# surface of revolution.  Which conic you roll picks the surface:
#
#     ellipse    ->  UNDULOID   a periodic chain of bulges
#     circle     ->  CYLINDER   (the ellipse's degenerate case)
#     ellipse    ->  SPHERE     (the other degenerate case, e -> 1,
#                               focus on the boundary: a bead chain of
#                               tangent spheres)
#     hyperbola  ->  NODOID     self-intersecting loops
#     parabola   ->  CATENOID   the minimal (H = 0) limit
#
# Why the closed form, when math_art/cmc_generator.py already reaches
# CMC surfaces?  Because that module gets them by constrained area
# minimisation, which is genuine CMC but cannot represent the
# SELF-INTERSECTING nodoid arcs (a minimiser is never self-intersecting)
# and cannot be aimed at a prescribed Delaunay weight.  Delaunay's
# construction is exact, covers the whole family including the nodoids,
# and takes the weight as an input.  The two routes then check each
# other, which is worth more than either alone.
#
# The construction used here is not the 1841 roulette but its modern
# first integral, which is exact, short, and numerically kind.  Write
# the meridian in arclength s as (r(s), z(s)) with tangent angle psi,
# so r' = cos(psi) and z' = sin(psi).  For a surface of revolution the
# principal curvatures are psi' (meridian) and sin(psi)/r (parallel), so
# constant mean curvature H means
#
#     psi' + sin(psi)/r = 2 H .
#
# Multiply by r and notice the left side is a total derivative:
#
#     d/ds ( r sin psi ) = cos psi ( sin psi + r psi' ) = 2 H r r'
#                        = H d/ds ( r^2 ) ,
#
# so the system has the FIRST INTEGRAL
#
#     r sin(psi) = H r^2 + c ,     i.e.    sin(psi) = H r + c / r ,
#
# with c the Delaunay weight (the "flux" through a parallel).  Every
# Delaunay surface is one (H, c); the classification above is just the
# sign of c.  Fixing H = 1/2 -- so all modes here share one mean
# curvature, which is what makes them a single family -- gives
#
#     sin(psi) = r/2 + c/r ,
#
# and |sin psi| <= 1 confines r between the roots of r^2/2 -+ r + c = 0:
#
#     UNDULOID  0 < c < 1/2   r in [1 - sqrt(1-2c), 1 + sqrt(1-2c)]
#     CYLINDER  c = 1/2       r == 1
#     SPHERE    c = 0         r in [0, 2]
#     NODOID    c < 0         r in [-1 + sqrt(1-2c), 1 + sqrt(1-2c)]
#
# Note the two degenerate members have DIFFERENT radii at the same H:
# the cylinder has one principal curvature zero, so H = 1/(2R) and
# R = 1, while the sphere has both equal to 1/R, so H = 1/R and R = 2.
# They are the same family at the same mean curvature all the same.
#
# In terms of the NECK radius (the smallest parallel), which is what the
# operator actually exposes because it is the one number you can see:
#
#     UNDULOID  neck in (0, 1)   bulge = 2 - neck   c =  neck (2 - neck)/2
#     NODOID    neck > 0         bulge = 2 + neck   c = -neck (2 + neck)/2
#
# so neck + bulge = 2 = 1/H for the unduloid and bulge - neck = 2 for the
# nodoid.  That 2 is the rolling conic's major axis 2a = 1/H, and
# neck * bulge = c/H is its semi-minor axis squared -- Delaunay's
# construction, read off the first integral.
#
# The meridian is then a quadrature, and it is integrated in a variable
# chosen to keep the integrand smooth.  dz/dr = sin psi / cos psi has an
# inverse-square-root singularity at both turning radii, where
# |sin psi| = 1; substituting r = r_mid - (delta) cos(theta) makes the
# vanishing factor exactly delta sin(theta), which CANCELS against the
# dr/dtheta = delta sin(theta) from the change of variables.  What is
# left is smooth on the closed interval:
#
#     dz/dtheta = sigma(r) * r / ( H sqrt( (r - o1)(r - o2) ) ) ,
#
# with sigma(r) = H r + c/r and o1, o2 the two roots that are NOT the
# turning radii (both negative in every case here, so the square root is
# real).  Note dz/dtheta changes sign for the nodoid exactly where
# sigma = 0, i.e. r^2 = -c/H: that sign change IS the nodoid's
# backwards loop, and it is the part of the family the variational
# solver cannot see.
#
# The catenoid is handled separately because it is the H = 0 member and
# the normalisation above divides by H; there sin(psi) = c/r integrates
# to the catenary r = c cosh(z/c) outright.
#
# References:
# - Charles-Eugene Delaunay, "Sur la surface de revolution dont la
#   courbure moyenne est constante", J. Math. Pures Appl. 6 (1841),
#   309-314 (the classification and the rolling-conic theorem).
# - M. Sturm, note appended to Delaunay's paper (1841), 315-320 -- the
#   first integral used here.
# - James Eells, "The surfaces of Delaunay", Math. Intelligencer 9
#   (1987), 53-57 (a modern account of the roulette construction).
# - Nicholas J. Korevaar, Rob Kusner, Bruce Solomon, "The structure of
#   complete embedded surfaces with constant mean curvature", J. Diff.
#   Geom. 30 (1989), 465-503 -- Delaunay surfaces as the ends of every
#   complete embedded CMC surface, which is why this family matters
#   beyond its own good looks.

bl_info = {
    "name": "Delaunay Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Delaunay Surface",
    "description": "Unduloid, nodoid, catenoid, cylinder and sphere -- "
                   "the constant-mean-curvature surfaces of revolution",
    "category": "Add Mesh",
}

import math

import numpy as np

# Every mode is built at this mean curvature, so the family is one
# family: the rolling conic's major axis is 2a = 1/H = 2 throughout.
H_FIXED = 0.5

MODES = ('UNDULOID', 'NODOID', 'SPHERE', 'CYLINDER', 'CATENOID')


def weight_for(mode, neck):
    """Delaunay weight c for a mode and neck radius, at H = H_FIXED.

    Inverts neck = 1 - sqrt(1 - 2c) (unduloid) and
    neck = -1 + sqrt(1 - 2c) (nodoid); see the module header."""
    if mode == 'UNDULOID':
        n = min(max(neck, 1e-6), 1.0 - 1e-9)
        return 0.5 * n * (2.0 - n)
    if mode == 'NODOID':
        n = max(neck, 1e-6)
        return -0.5 * n * (2.0 + n)
    if mode == 'SPHERE':
        return 0.0
    if mode == 'CYLINDER':
        return 0.5
    raise ValueError(f"mode {mode!r} has no Delaunay weight")


def turning_radii(c, H=H_FIXED):
    """(r_min, r_max, other_root_1, other_root_2) for the weight c.

    r^2 H -+ r + c = 0 give four roots; two bound the physical interval
    (where |sin psi| = 1) and two are negative and only appear inside
    the smooth square root of the quadrature."""
    disc = 1.0 - 4.0 * H * c
    if disc < 0.0:
        raise ValueError(f"no Delaunay surface for c = {c} at H = {H}")
    s = math.sqrt(disc)
    p_lo, p_hi = (1.0 - s) / (2.0 * H), (1.0 + s) / (2.0 * H)
    q_lo, q_hi = (-1.0 - s) / (2.0 * H), (-1.0 + s) / (2.0 * H)
    if c > 0.0:                      # unduloid: both p roots positive
        return p_lo, p_hi, q_lo, q_hi
    return q_hi, p_hi, p_lo, q_lo    # nodoid / sphere


def half_meridian(c, n=400, H=H_FIXED):
    """One half period of the meridian, neck -> bulge, as (r, z) with
    z(0) = 0.  See the module header for why theta is the right
    variable."""
    r_min, r_max, o1, o2 = turning_radii(c, H)
    r_mid, delta = 0.5 * (r_min + r_max), 0.5 * (r_max - r_min)
    theta = np.linspace(0.0, math.pi, n)

    def dz(th):
        r = r_mid - delta * np.cos(th)
        r = np.maximum(r, 1e-12)
        sigma = H * r + c / r
        root = np.sqrt(np.maximum((r - o1) * (r - o2), 1e-300))
        return sigma * r / (H * root)

    h = float(theta[1] - theta[0])
    y = dz(theta)
    ym = dz(0.5 * (theta[:-1] + theta[1:]))
    seg = (h / 6.0) * (y[:-1] + 4.0 * ym + y[1:])
    z = np.concatenate([[0.0], np.cumsum(seg)])
    return r_mid - delta * np.cos(theta), z


def meridian(mode, neck=0.4, periods=2, n=400, height=2.5):
    """Full meridian (r, z) of a Delaunay surface.

    `periods` counts neck-to-neck repeats for the periodic modes; for
    CATENOID it is ignored and `height` sets the extent instead."""
    if mode == 'CATENOID':
        # H = 0: sin(psi) = c/r integrates to the catenary outright.
        a = max(neck, 1e-6)
        t = np.linspace(-height / (2.0 * a), height / (2.0 * a), n)
        return a * np.cosh(t), a * t
    if mode == 'CYLINDER':
        z = np.linspace(-0.5 * height, 0.5 * height, n)
        return np.full_like(z, 1.0 / (2.0 * H_FIXED)), z

    c = weight_for(mode, neck)
    rh, zh = half_meridian(c, n)
    span = float(zh[-1])
    # a full period is the half arc plus its mirror about the bulge
    r_per = np.concatenate([rh, rh[::-1][1:]])
    z_per = np.concatenate([zh, (2.0 * span - zh[::-1])[1:]])
    r_all, z_all = [r_per], [z_per]
    for k in range(1, max(1, periods)):
        r_all.append(r_per[1:])
        z_all.append(z_per[1:] + 2.0 * span * k)
    r = np.concatenate(r_all)
    z = np.concatenate(z_all)
    return r, z - 0.5 * (z[0] + z[-1])


def mean_curvature_profile(r, z):
    """Mean curvature sampled along a meridian, from the geometry alone.

    H = (psi' + sin(psi)/r)/2 with psi the tangent angle and ' the
    arclength derivative.  Independent of everything the construction
    assumed, so it is a real check rather than a restatement."""
    dr = np.gradient(r)
    dz = np.gradient(z)
    ds = np.hypot(dr, dz)
    psi = np.unwrap(np.arctan2(dz, dr))
    dpsi = np.gradient(psi)
    kappa_meridian = dpsi / np.maximum(ds, 1e-300)
    kappa_parallel = np.sin(psi) / np.maximum(r, 1e-300)
    return 0.5 * (kappa_meridian + kappa_parallel)


def build_surface(mode, neck=0.4, periods=2, ures=200, vres=64,
                  height=2.5, scale=1.0):
    """Revolve the meridian into a quad mesh.  Returns (verts, faces,
    info) with info carrying the numbers worth reporting."""
    r, z = meridian(mode, neck, periods, ures, height)
    phi = np.linspace(0.0, 2.0 * math.pi, vres, endpoint=False)
    R, P = np.meshgrid(r, phi, indexing='ij')
    Z, _ = np.meshgrid(z, phi, indexing='ij')
    V = np.stack([(R * np.cos(P)).ravel(),
                  (R * np.sin(P)).ravel(), Z.ravel()], axis=-1)
    nu, nv = len(r), vres
    faces = []
    for i in range(nu - 1):
        for j in range(nv):
            j1 = (j + 1) % nv
            faces.append((i * nv + j, i * nv + j1,
                          (i + 1) * nv + j1, (i + 1) * nv + j))
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    V = (V - 0.5 * (lo + hi)) * ((2.0 / ext if ext > 1e-9 else 1.0) * scale)
    H_meas = mean_curvature_profile(r, z)
    info = {
        'neck': float(r.min()), 'bulge': float(r.max()),
        'weight': (0.0 if mode == 'CATENOID' else weight_for(mode, neck)),
        'H_median': float(np.median(H_meas[2:-2])),
        'z_span': float(z[-1] - z[0]),
    }
    return V, faces, info


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_delaunay_surface_add(bpy.types.Operator):
        """Add a Delaunay surface: a surface of revolution of constant
        mean curvature, traced by the focus of a rolling conic"""
        bl_idname = "mesh.delaunay_surface_add"
        bl_label = "Delaunay Surface"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Surface",
            items=[('UNDULOID', "Unduloid", "rolling ELLIPSE -- a "
                    "periodic chain of bulges, the generic Delaunay "
                    "surface"),
                   ('NODOID', "Nodoid", "rolling HYPERBOLA -- "
                    "self-intersecting loops; the branch a "
                    "minimisation solver cannot reach"),
                   ('SPHERE', "Sphere Chain", "the degenerate ellipse: "
                    "the meridian closes and the surface becomes a "
                    "string of tangent spheres"),
                   ('CYLINDER', "Cylinder", "rolling CIRCLE -- the "
                    "constant-radius member"),
                   ('CATENOID', "Catenoid", "rolling PARABOLA -- the "
                    "minimal (H = 0) limit of the family")],
            default='UNDULOID')
        neck: FloatProperty(
            name="Neck Radius", default=0.4, min=0.02, max=3.0,
            description="Radius of the narrowest parallel.  The "
                        "unduloid's bulge is then 2 - neck (so neck < 1, "
                        "and neck = 1 is the cylinder); the nodoid's is "
                        "2 + neck; the catenoid's neck is its waist")
        periods: IntProperty(
            name="Periods", default=2, min=1, max=12,
            description="Neck-to-neck repeats (ignored for the "
                        "cylinder and catenoid)")
        ures: IntProperty(name="Profile Samples", default=200, min=16,
                          max=2000)
        vres: IntProperty(name="Revolution Samples", default=64, min=6,
                          max=512)
        height: FloatProperty(
            name="Height", default=2.5, min=0.2, max=40.0,
            description="Extent of the cylinder and catenoid, which "
                        "have no period of their own")
        shade_smooth: BoolProperty(name="Smooth Shading", default=True)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            neck = self.neck
            if self.mode == 'UNDULOID' and neck >= 1.0:
                # neck + bulge = 2, so neck >= 1 is the cylinder or
                # beyond; there is no unduloid there.
                self.report({'WARNING'},
                            f"Unduloid needs neck < 1 (got {neck:.2f}); "
                            f"clamped to 0.98")
                neck = 0.98
            verts, faces, info = build_surface(
                self.mode, neck, self.periods, self.ures, self.vres,
                self.height, self.scale)
            label = dict(UNDULOID="Unduloid", NODOID="Nodoid",
                         SPHERE="Sphere Chain", CYLINDER="Cylinder",
                         CATENOID="Catenoid")[self.mode]
            me = bpy.data.meshes.new(label)
            me.from_pydata([tuple(v) for v in verts], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.shade_smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(label, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report(
                {'INFO'},
                f"{label}: V={len(me.vertices)} F={len(me.polygons)}, "
                f"neck {info['neck']:.4f} bulge {info['bulge']:.4f}, "
                f"Delaunay weight c={info['weight']:+.4f}, "
                f"measured H={info['H_median']:.4f} "
                f"(exact {0.0 if self.mode == 'CATENOID' else H_FIXED})")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode in ('UNDULOID', 'NODOID', 'CATENOID'):
                lay.prop(self, 'neck')
                if self.mode == 'UNDULOID' and self.neck >= 1.0:
                    lay.label(text="Unduloid needs neck < 1",
                              icon='ERROR')
            if self.mode in ('UNDULOID', 'NODOID', 'SPHERE'):
                lay.prop(self, 'periods')
            if self.mode in ('CYLINDER', 'CATENOID'):
                lay.prop(self, 'height')
            lay.prop(self, 'ures')
            lay.prop(self, 'vres')
            lay.prop(self, 'shade_smooth')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.delaunay_surface_add",
                             icon='META_CAPSULE')

    ADD_MENU = True    # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_delaunay_surface_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_delaunay_surface_add)


def _selftest():
    ok_all = True

    # 1) THE gate: mean curvature really is constant along the meridian,
    #    measured from the geometry rather than from the construction.
    for mode, neck, want in (('UNDULOID', 0.4, H_FIXED),
                             ('UNDULOID', 0.15, H_FIXED),
                             ('UNDULOID', 0.85, H_FIXED),
                             ('NODOID', 0.4, H_FIXED),
                             ('NODOID', 1.5, H_FIXED),
                             ('SPHERE', 0.0, H_FIXED),
                             ('CYLINDER', 1.0, H_FIXED),
                             ('CATENOID', 0.5, 0.0)):
        r, z = meridian(mode, neck, periods=1, n=4000)
        Hs = mean_curvature_profile(r, z)[5:-5]     # drop the end stencils
        med = float(np.median(Hs))
        spread = float(np.percentile(Hs, 97.5) - np.percentile(Hs, 2.5))
        ok = abs(med - want) < 2e-3 and spread < 5e-3
        ok_all = ok_all and ok
        print(f"H const {mode:9s} neck={neck:<5.2f}: median {med:+.6f} "
              f"(want {want:+.4f}) 95% spread {spread:.2e} "
              f"{'OK' if ok else 'BAD'}")

    # 2) neck and bulge hit the closed forms neck + bulge = 2 (unduloid)
    #    and bulge - neck = 2 (nodoid) -- the rolling conic's major axis
    for mode, neck in (('UNDULOID', 0.2), ('UNDULOID', 0.6),
                       ('NODOID', 0.3), ('NODOID', 1.2)):
        r, _ = meridian(mode, neck, periods=1, n=4000)
        want_bulge = (2.0 - neck) if mode == 'UNDULOID' else (2.0 + neck)
        ok = (abs(r.min() - neck) < 1e-6
              and abs(r.max() - want_bulge) < 1e-6)
        ok_all = ok_all and ok
        print(f"radii {mode:9s} neck={neck:<4.2f}: measured "
              f"{r.min():.6f} / {r.max():.6f} (want {neck:.6f} / "
              f"{want_bulge:.6f}) {'OK' if ok else 'BAD'}")

    # 3) the degenerate members are exactly what they claim
    # NB the sphere's radius is 1/H = 2, not 1/(2H) = 1: a sphere has
    # BOTH principal curvatures 1/R, so H = 1/R, whereas the cylinder
    # has one of them zero and H = 1/(2R).  Same family, same H,
    # different radii -- an easy place to write the wrong assertion.
    r, z = meridian('SPHERE', 0.0, periods=1, n=4000)
    rad = np.hypot(r, z - 0.5 * (z[0] + z[-1]))
    want_R = 1.0 / H_FIXED
    ok = abs(rad.max() - want_R) < 2e-3 and (rad.max() - rad.min()) < 5e-3
    ok_all = ok_all and ok
    print(f"sphere         : radius in [{rad.min():.5f}, {rad.max():.5f}] "
          f"(want {want_R:.1f} = 1/H) {'OK' if ok else 'BAD'}")

    r, z = meridian('CATENOID', 0.5, n=2000, height=2.5)
    ok = float(np.abs(r - 0.5 * np.cosh(z / 0.5)).max()) < 1e-9
    ok_all = ok_all and ok
    print(f"catenoid       : max|r - a cosh(z/a)| = "
          f"{float(np.abs(r - 0.5 * np.cosh(z / 0.5)).max()):.2e} "
          f"{'OK' if ok else 'BAD'}")

    r, _ = meridian('CYLINDER', 1.0, n=100, height=3.0)
    ok = float(np.abs(r - 1.0).max()) < 1e-12
    ok_all = ok_all and ok
    print(f"cylinder       : max|r - 1| = {float(np.abs(r - 1).max()):.1e} "
          f"{'OK' if ok else 'BAD'}")

    # 4) the nodoid genuinely turns back on itself -- this is the branch
    #    the variational solver in cmc_generator.py cannot produce, so
    #    the self-intersection is the whole point of having this module.
    for mode, neck, want_turn in (('NODOID', 0.4, True),
                                  ('NODOID', 1.2, True),
                                  ('UNDULOID', 0.4, False),
                                  ('UNDULOID', 0.9, False)):
        _, z = meridian(mode, neck, periods=1, n=4000)
        dz = np.diff(z)
        turns = bool((dz.min() < -1e-9) and (dz.max() > 1e-9))
        ok = turns == want_turn
        ok_all = ok_all and ok
        print(f"loop {mode:9s} neck={neck:<4.2f}: meridian turns back "
              f"{str(turns):5s} (want {want_turn}) "
              f"{'OK' if ok else 'BAD'}")

    # 5) the weight matches the first integral: sin(psi) = H r + c/r must
    #    reach exactly +-1 at the turning radii and stay inside between
    for mode, neck in (('UNDULOID', 0.35), ('NODOID', 0.8)):
        c = weight_for(mode, neck)
        r, z = meridian(mode, neck, periods=1, n=4000)
        sigma = H_FIXED * r + c / r
        ok = (abs(abs(sigma).max() - 1.0) < 1e-9
              and float(sigma.max()) <= 1.0 + 1e-9
              and float(sigma.min()) >= -1.0 - 1e-9)
        ok_all = ok_all and ok
        print(f"first integral {mode:9s}: sin(psi) in "
              f"[{sigma.min():+.7f}, {sigma.max():+.7f}] (must reach "
              f"+-1, never exceed) {'OK' if ok else 'BAD'}")

    # 6) meshes build, are finite, and are not collapsed
    for mode in MODES:
        V, F, info = build_surface(mode, 0.4, 2, 120, 48)
        V = np.asarray(V)
        ext = V.max(0) - V.min(0)
        ok = (len(F) > 0 and np.isfinite(V).all()
              and float(ext.min() / ext.max()) > 0.05)
        ok_all = ok_all and ok
        print(f"build {mode:9s}: V={len(V)} F={len(F)} aspect "
              f"{ext.min() / ext.max():.3f} H={info['H_median']:.4f} "
              f"{'OK' if ok else 'BAD'}")

    # 7) THE cross-check: the variational CMC solver in cmc_generator.py
    #    relaxes a liquid bridge by constrained area minimisation, which
    #    shares NO code with the closed form here.  Its equilibrium must
    #    be a Delaunay surface, so the two must draw the same curve.
    #
    #    Three things make the comparison honest rather than circular:
    #      * CMC surfaces scale -- x -> lambda x sends H -> H/lambda --
    #        so a relaxed bridge of curvature H0 rescaled by 2 H0 is a
    #        Delaunay surface at H = 1/2, this module's normalisation.
    #        Nothing is fitted; H0 is measured off the solver's mesh.
    #      * The weight c is then read from the first integral
    #        c = r sin(psi) - H r^2, again measured, not assumed.  Its
    #        SPREAD along the meridian doubles as a convergence monitor:
    #        a surface that is really CMC has c constant.
    #      * The curves are compared by nearest distance, not by
    #        interpolating r(z).  A relaxed bridge is only a SEGMENT of
    #        a Delaunay surface pinned between two rims, and the nodoid
    #        meridian turns back on itself so r is multivalued in z --
    #        an r(z) comparison silently compares different branches.
    try:
        from .cmc_generator import relax_bridge, signed_mean_curvature
    except Exception as exc:                    # noqa: BLE001
        print(f"cross-check     : SKIP (cmc_generator: {exc})")
    else:
        nring, nrow = 48, 17
        for vf, want_mode in ((1.15, 'UNDULOID'), (1.35, 'NODOID')):
            Vb, Tb, _ = relax_bridge(volume_factor=vf, nring=nring,
                                     nrow=nrow, iters=600, squash=0.2)
            rb = np.hypot(Vb[:, 0], Vb[:, 1]).reshape(nrow, nring).mean(1)
            zb = Vb[:, 2].reshape(nrow, nring).mean(1)
            Hb = signed_mean_curvature(Vb, Tb).reshape(nrow, nring).mean(1)
            inner = slice(3, nrow - 3)
            H0 = float(np.median(Hb[inner]))
            lam = 2.0 * abs(H0)
            rr_s, zz_s = rb * lam, zb * lam
            psi = np.unwrap(np.arctan2(np.gradient(zz_s),
                                       np.gradient(rr_s)))
            c_pt = rr_s * np.sin(psi) - H_FIXED * rr_s * rr_s
            c_meas = float(np.median(c_pt[inner]))
            c_spread = float(c_pt[inner].max() - c_pt[inner].min())
            mode = 'UNDULOID' if c_meas > 0 else 'NODOID'
            root = math.sqrt(max(1.0 - 2.0 * c_meas, 0.0))
            neck = (1.0 - root) if c_meas > 0 else (-1.0 + root)
            rc, zc = meridian(mode, neck, periods=2, n=4000)
            mid = nrow // 2
            if rr_s[mid] > rr_s[inner][0]:      # mid-plane is a bulge
                pick = np.where(rc > rc.max() - 1e-9)[0]
            else:                               # mid-plane is a neck
                pick = np.where(rc < rc.min() + 1e-9)[0]
            zc = zc - zc[pick[len(pick) // 2]]
            zz_s = zz_s - zz_s[mid]
            P = np.stack([rr_s[inner], zz_s[inner]], axis=1)
            Q = np.stack([rc, zc], axis=1)
            dist = np.linalg.norm(P[:, None, :] - Q[None, :, :],
                                  axis=2).min(axis=1)
            rel = float(dist.max() / rc.max())
            ok = mode == want_mode and c_spread < 2e-2 and rel < 5e-3
            ok_all = ok_all and ok
            print(f"cross-check vf={vf}: solver H={H0:+.4f} -> c="
                  f"{c_meas:+.5f} (spread {c_spread:.1e}) {mode}, "
                  f"neck {neck:.4f}; curves agree to {rel * 100:.3f}% "
                  f"{'OK' if ok else 'BAD'}")

    assert ok_all
    print("delaunay surface standalone tests passed")
