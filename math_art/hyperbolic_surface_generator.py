
# Hyperbolic Surfaces Generator for Blender
#
# Smooth surfaces of constant negative Gaussian curvature (K = -1) from
# the classical theory of pseudospherical surfaces / the sine-Gordon
# equation:
#
#   PSEUDOSPHERE  the tractricoid -- the surface of revolution of a
#                 tractrix (Beltrami's model of the hyperbolic plane)
#   DINI          Dini's surface -- a pseudosphere sheared into a
#                 twisting helical band (twist exposed as a parameter)
#   KUEN          Kuen's surface -- a bounded K = -1 surface with a
#                 characteristic bulb and cusp
#   MINDING_BULGE and MINDING_SPINDLE -- the other two K = -1 surfaces
#                 of revolution (see below)
#
# The three surfaces of revolution.  Writing a surface of revolution
# with the metric ds^2 = du^2 + f(u)^2 dv^2 (so u is arclength along the
# meridian), the Gauss curvature is K = -f''/f, so K = -1 forces
#     f'' = f ,   hence   f(u) = A cosh u + B sinh u ,
# and the sign of A^2 - B^2 splits the K = -1 surfaces of revolution
# into exactly THREE types -- a classification due to Minding (1839):
#
#   A^2 = B^2  parabolic    f = C e^{-u}    the pseudosphere (above)
#   A^2 > B^2  hyperbolic   f = a cosh u    the MINDING BULGE
#   A^2 < B^2  conic        f = a sinh u    the MINDING SPINDLE
#
# Embedded as r(u) = f(u), z(u) = integral sqrt(1 - f'(u)^2) du, which
# is real only while |f'| <= 1.  That bound is the whole story of these
# surfaces and is exposed rather than hidden: the bulge runs out at
# |u| = asinh(1/a) in a pair of flared rims, and the spindle at
# u = acosh(1/a) in an equator, past which no isometric embedding in
# R^3 continues.  Both therefore realise only a STRIP of the hyperbolic
# plane -- the same Hilbert obstruction as the pseudosphere -- and the
# spindle additionally closes up at two conical points, where it is not
# an immersion at all.  All three are locally isometric to each other
# and to the hyperbolic plane; they differ only in how they are placed
# in R^3, which is Minding's point.
#
# Expect a visible crease around the spindle's equator: it is real, not
# a meshing artifact.  Where |f'| -> 1 the meridian meets that circle
# with a VERTICAL tangent -- writing s for the arclength short of it,
# r_max - r ~ s while z_eq - z ~ s^{3/2}, so dr/dz blows up like
# s^{-1/2}.  The two mirrored halves therefore share a tangent line but
# not a curvature, and the closed spindle is C^1 and not C^2.  The
# bulge shows the same thing, less obtrusively, at its two rims.
#
# By Hilbert's theorem there is no complete C^2 isometric immersion of
# the whole hyperbolic plane in R^3, so each of these covers only a
# patch; the crocheted (ruffled) realisation of the *whole* plane lives
# in the separate Crochet generator. Output is centred and fit to a
# 2 m cube; pair it with the Curvature Colour operator to see the
# constant negative curvature as a uniform tint.
#
# References:
# - Pseudosphere: Eugenio Beltrami, "Saggio di interpretazione della
#   geometria non-euclidea", Giornale di Matematiche 6, 1868.
# - Dini's surface: Ulisse Dini, 1865.
# - Kuen's surface: Theodor Kuen, "Ueber Flaechen von constantem
#   Kruemmungsmass", Sitzungsber. Bayer. Akad. Wiss., 1884.
# - Hilbert's theorem: D. Hilbert, "Ueber Flaechen von constanter
#   Gausscher Kruemmung", Trans. AMS 2, 1901, pp. 87-99.
# - Minding bulge and spindle: Ferdinand Minding, "Wie sich entscheiden
#   laesst, ob zwei gegebene krumme Flaechen auf einander abwickelbar
#   sind oder nicht; nebst Bemerkungen ueber die Flaechen von
#   unveraenderlichem Kruemmungsmasse", J. reine angew. Math. (Crelle)
#   19 (1839), 370-387 -- the classification of the constant-curvature
#   surfaces of revolution into the parabolic, hyperbolic and conic
#   types, and the theorem that all surfaces of equal constant
#   curvature are locally isometric.

bl_info = {
    "name": "Hyperbolic Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Hyperbolic Surface",
    "description": "Pseudosphere, Dini, Kuen and Minding constant-"
                   "negative-curvature surfaces",
    "category": "Add Mesh",
}

import math

import numpy as np


def _pseudosphere(U, V, twist=0.0, a=0.5):
    x = np.cosh(U) ** -1 * np.cos(V)
    y = np.cosh(U) ** -1 * np.sin(V)
    z = U - np.tanh(U)
    return x, y, z


def _dini(U, V, twist=0.2, a=0.5):
    x = np.cos(V) * np.sin(U)
    y = np.sin(V) * np.sin(U)
    z = np.cos(U) + np.log(np.tan(U / 2.0)) + twist * V
    return x, y, z


def _cum_simpson(g, t):
    """Cumulative integral of the callable `g` over the uniform grid
    `t`, Simpson's rule per interval (so O(h^4), with `g` also sampled
    at the interval midpoints).  Returns an array the length of `t`
    starting at 0."""
    h = float(t[1] - t[0])
    y = g(t)
    ym = g(0.5 * (t[:-1] + t[1:]))
    seg = (h / 6.0) * (y[:-1] + 4.0 * ym + y[1:])
    return np.concatenate([[0.0], np.cumsum(seg)])


def _minding_profile(kind, a, n=2001):
    """Meridian of a Minding surface as tables (t, r, z), with the
    normalised profile parameter t running over [-1, 1].

    Both meridians are integrated in a variable that makes the
    integrand SMOOTH, because the naive one is not.  With u the
    arclength, z' = sqrt(1 - f'(u)^2) vanishes like a square root where
    |f'| -> 1, so quadrature in u converges only like h^{3/2} at
    exactly the rim that gives these surfaces their shape.  Substituting
    f'(u) = sin(phi) removes it:

      BULGE   f = a cosh u,  f' = a sinh u = sin(phi),  phi in [-pi/2, pi/2]
              r = sqrt(a^2 + sin^2 phi)
              dz = cos^2(phi) dphi / sqrt(a^2 + sin^2 phi)
              -- smooth on the closed interval; t = 2 phi / pi.

      SPINDLE f = a sinh u,  f' = a cosh u = sin(phi),  phi in [asin a, pi/2]
              r = sqrt(sin^2 phi - a^2)
              dz = cos^2(phi) dphi / sqrt(sin^2 phi - a^2)
              -- now singular at the TIP instead (u |-> phi has a
              critical point there), cured by the second substitution
              phi = phi0 + tau^2, under which dz/dtau and r are both
              smooth and vanish linearly.  The half meridian tip ->
              equator is then mirrored to close the spindle.

    So each surface is integrated in the variable that is smooth for it,
    rather than one shared variable that is smooth for neither."""
    if kind == 'MINDING_BULGE':
        # t in [-1, 1] <-> phi in [-pi/2, pi/2]
        t = np.linspace(-1.0, 1.0, n)
        phi = t * (math.pi / 2.0)
        r = np.sqrt(a * a + np.sin(phi) ** 2)

        def dz_dt(tt):
            p = tt * (math.pi / 2.0)
            s = np.sin(p)
            return (math.pi / 2.0) * np.cos(p) ** 2 / np.sqrt(a * a + s * s)

        z = _cum_simpson(dz_dt, t)
        return t, r, z - 0.5 * (z[0] + z[-1])       # centre the waist

    if kind != 'MINDING_SPINDLE':
        raise ValueError(f"not a Minding surface: {kind!r}")

    phi0 = math.asin(min(max(a, 1e-9), 1.0 - 1e-12))
    span = math.pi / 2.0 - phi0
    if span <= 1e-12:                                # a -> 1: degenerate
        span = 1e-12
    T = math.sqrt(span)
    tau = np.linspace(0.0, T, n)                     # 0 = tip, T = equator

    def _phi(tt):
        return phi0 + tt * tt

    def _r_of(tt):
        s = np.sin(_phi(tt))
        return np.sqrt(np.maximum(s * s - a * a, 0.0))

    def dz_dtau(tt):
        p = _phi(tt)
        s = np.sin(p)
        rad = np.sqrt(np.maximum(s * s - a * a, 0.0))
        # limit at the tip: sin^2 phi - a^2 ~ 2 a cos(phi0) tau^2, so
        # dz/dtau -> 2 cos^2(phi0) / sqrt(2 a cos phi0), a finite value.
        lim = 2.0 * math.cos(phi0) ** 2 / math.sqrt(
            2.0 * a * math.cos(phi0)) if a > 0 else 0.0
        out = np.full_like(np.asarray(tt, dtype=float), lim)
        good = rad > 1e-14
        out[good] = (2.0 * tt[good] * np.cos(p[good]) ** 2 / rad[good])
        return out

    W = _cum_simpson(dz_dtau, tau)                   # height above the tip
    z_half = W[-1] - W                               # height below equator
    r_half = _r_of(tau)
    s_half = 1.0 - tau / T                           # 1 at tip, 0 at equator

    order = np.argsort(s_half)                       # make t increasing
    sp, rp, zp = s_half[order], r_half[order], z_half[order]
    t = np.concatenate([-sp[::-1][:-1], sp])
    r = np.concatenate([rp[::-1][:-1], rp])
    z = np.concatenate([-zp[::-1][:-1], zp])
    return t, r, z


def _minding(kind):
    """Build the (U, V, twist, a) surface function for a Minding type.
    The profile tables are rebuilt per call because they depend on `a`;
    at n = 2001 that is well under a millisecond."""
    def fn(U, V, twist=0.0, a=0.5):
        t, r, z = _minding_profile(kind, a)
        ru = np.interp(U, t, r)
        zu = np.interp(U, t, z)
        return ru * np.cos(V), ru * np.sin(V), zu
    return fn


def _kuen(U, V, twist=0.0, a=0.5):
    denom = 1.0 + (U * np.sin(V)) ** 2
    x = 2.0 * (np.cos(U) + U * np.sin(U)) * np.sin(V) / denom
    y = 2.0 * (np.sin(U) - U * np.cos(U)) * np.sin(V) / denom
    z = np.log(np.tan(V / 2.0)) + 2.0 * np.cos(V) / denom
    return x, y, z


# (label, function, u range, v range, wrap_v)
PRESETS = {
    'PSEUDOSPHERE': ("Pseudosphere", _pseudosphere, (0.0, 3.0),
                     (0.0, 2 * math.pi), True),
    'DINI': ("Dini Surface", _dini, (0.05, 1.5),
             (0.0, 12.0 * math.pi), False),
    'KUEN': ("Kuen Surface", _kuen, (-4.5, 4.5),
             (0.08, math.pi - 0.08), False),
    # u is the NORMALISED profile parameter t of _minding_profile, not
    # arclength -- the arclength extent depends on `a`.  The spindle
    # stops a hair short of +-1 because r -> 0 there: meshing the two
    # conical points exactly would collapse a whole ring of vertices
    # onto one another and hand Blender a fan of degenerate quads.  At
    # 0.999 the remaining opening is ~0.1% of the equatorial radius.
    'MINDING_BULGE': ("Minding Bulge", _minding('MINDING_BULGE'),
                      (-1.0, 1.0), (0.0, 2 * math.pi), True),
    'MINDING_SPINDLE': ("Minding Spindle", _minding('MINDING_SPINDLE'),
                        (-0.999, 0.999), (0.0, 2 * math.pi), True),
}


def _center(V):
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = float((hi - lo).max())
    return (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)


def _grid_faces(nu, nv, wrap_v):
    faces = []
    for i in range(nu - 1):
        for j in range(nv - (0 if wrap_v else 1)):
            j1 = (j + 1) % nv
            faces.append((i * nv + j, (i + 1) * nv + j,
                          (i + 1) * nv + j1, i * nv + j1))
    return faces


def build_surface(kind, ures, vres, twist=0.2, scale=1.0,
                  minding_a=0.5):
    label, fn, (u0, u1), (v0, v1), wrap = PRESETS[kind]
    us = np.linspace(u0, u1, ures)
    vs = (np.linspace(v0, v1, vres, endpoint=False) if wrap
          else np.linspace(v0, v1, vres))
    U, Vv = np.meshgrid(us, vs, indexing='ij')
    x, y, z = fn(U, Vv, twist, minding_a)
    V = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    faces = _grid_faces(ures, vres, wrap)
    return _center(V) * scale, faces


def mean_curvature(V, faces):
    """Median per-vertex Gaussian curvature via angle defect / area;
    ~ -1 for these surfaces. Robust to the stretched triangles the
    parametrisations produce near their singular boundaries."""
    n = len(V)
    ang = np.zeros(n)
    area = np.zeros(n)
    deg = np.zeros(n)
    tris = []
    for f in faces:
        if len(f) == 4:
            tris.append((f[0], f[1], f[2]))
            tris.append((f[0], f[2], f[3]))
        else:
            tris.append((f[0], f[1], f[2]))
    for a, b, c in tris:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            u = V[j] - V[i]
            w = V[k] - V[i]
            cs = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)
                                 + 1e-12)
            ang[i] += math.acos(max(-1.0, min(1.0, cs)))
        ar = 0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a]))
        for i in (a, b, c):
            area[i] += ar / 3.0
            deg[i] += 1
    interior = deg >= 6
    defect = 2 * math.pi - ang
    ki = defect[interior] / np.maximum(area[interior], 1e-12)
    return float(np.median(ki)) if ki.size else 0.0


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

    class MESH_OT_hyperbolic_surface_add(bpy.types.Operator):
        """Add a smooth constant-negative-curvature surface
        (pseudosphere, Dini or Kuen)"""
        bl_idname = "mesh.hyperbolic_surface_add"
        bl_label = "Hyperbolic Surface"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Surface",
            items=[(k, v[0], v[0]) for k, v in PRESETS.items()],
            default='PSEUDOSPHERE')
        ures: IntProperty(name="U Resolution", default=64, min=8,
                          max=400)
        vres: IntProperty(name="V Resolution", default=96, min=8,
                          max=400)
        twist: FloatProperty(
            name="Twist", default=0.2, min=0.0, max=2.0,
            description="Helical shear of Dini's surface "
                        "(curvature = -1/(1+twist^2))")
        minding_a: FloatProperty(
            name="Waist / Girth", default=0.5, min=0.05, max=3.0,
            description="Shape parameter a of the Minding surfaces: "
                        "the bulge is r = a cosh(u), so a is its waist "
                        "radius; the spindle is r = a sinh(u) and needs "
                        "a < 1, its equator sitting at sqrt(1 - a^2)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                            max=100.0)
        shade_smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            label = PRESETS[self.preset][0]
            a = self.minding_a
            if self.preset == 'MINDING_SPINDLE' and a >= 1.0:
                # r = a sinh(u) needs |f'| = a cosh(u) <= 1, which is
                # unsatisfiable for a >= 1: there is no spindle at all.
                self.report({'WARNING'},
                            f"Minding spindle needs a < 1 (got {a:.2f}); "
                            f"clamped to 0.95")
                a = 0.95
            verts, faces = build_surface(self.preset, self.ures,
                                         self.vres, self.twist,
                                         self.scale, a)
            me = bpy.data.meshes.new("HyperbolicSurface")
            me.from_pydata([tuple(v) for v in np.asarray(verts)], [],
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
            self.report({'INFO'},
                        f"{label}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            lay.prop(self, 'ures')
            lay.prop(self, 'vres')
            if self.preset == 'DINI':
                lay.prop(self, 'twist')
            if self.preset in ('MINDING_BULGE', 'MINDING_SPINDLE'):
                lay.prop(self, 'minding_a')
                if (self.preset == 'MINDING_SPINDLE'
                        and self.minding_a >= 1.0):
                    lay.label(text="Spindle needs a < 1", icon='ERROR')
            lay.prop(self, 'scale')
            lay.prop(self, 'shade_smooth')

    def _menu_func(self, context):
        self.layout.operator("mesh.hyperbolic_surface_add",
                             icon='MESH_CAPSULE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_hyperbolic_surface_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_hyperbolic_surface_add)


def _selftest():
    ok_all = True

    # 1) each surface has constant Gaussian curvature ~ -1
    #    (Dini with twist 0.2 -> -1/(1+0.2^2) = -0.96)
    want = {'PSEUDOSPHERE': -1.0, 'DINI': -1.0 / 1.04, 'KUEN': -1.0,
            'MINDING_BULGE': -1.0, 'MINDING_SPINDLE': -1.0}
    for kind in PRESETS:
        fn = PRESETS[kind][1]
        _, _, (u0, u1), (v0, v1), wrap = PRESETS[kind]
        us = np.linspace(u0, u1, 80)
        vs = (np.linspace(v0, v1, 80, endpoint=False) if wrap
              else np.linspace(v0, v1, 80))
        U, Vv = np.meshgrid(us, vs, indexing='ij')
        x, y, z = fn(U, Vv, 0.2, 0.5)
        Vraw = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
        faces = _grid_faces(80, 80, wrap)
        K = mean_curvature(Vraw, faces)
        ok = abs(K - want[kind]) < 0.05
        ok_all = ok_all and ok
        print(f"{kind:16s}: V={len(Vraw)} F={len(faces)} "
              f"meanK={K:+.3f} (want {want[kind]:+.3f}) "
              f"{'OK' if ok else 'BAD'}")

    # 2) The Minding profiles, checked against the ODE they come from
    #    rather than against a curvature estimate.  Re-deriving the
    #    arclength u from the generated (r, z) table and comparing r(u)
    #    with a cosh u / a sinh u exercises the whole substitution and
    #    quadrature chain, and is far sharper than an angle-defect
    #    measurement on a mesh.
    for kind, f_exact in (('MINDING_BULGE', np.cosh),
                          ('MINDING_SPINDLE', np.sinh)):
        for a in (0.3, 0.5, 0.8):
            t, r, z = _minding_profile(kind, a, n=4001)
            if kind == 'MINDING_SPINDLE':
                t, r, z = t[len(t) // 2:], r[len(r) // 2:], z[len(z) // 2:]
            # arclength along the meridian, measured from the waist
            # (bulge) or from the equator (spindle)
            seg = np.hypot(np.diff(r), np.diff(z))
            u = np.concatenate([[0.0], np.cumsum(seg)])
            if kind == 'MINDING_BULGE':
                u = u - np.interp(0.0, t, u)      # u = 0 at the waist
                # the table spans BOTH rims, |u| <= asinh(1/a)
                u_end = 2.0 * math.asinh(1.0 / a)
            else:
                u = u[-1] - u                     # u = 0 at the tip
                u_end = math.acosh(1.0 / a)       # tip -> equator only
            r_exact = a * f_exact(u)
            err = float(np.abs(r - r_exact).max())
            # the arclength extent must also hit the analytic |f'| <= 1
            # bound, which is what makes these strips and not planes
            span_err = abs(abs(u[0] - u[-1]) - u_end)
            ok = err < 5e-4 and span_err < 5e-4
            ok_all = ok_all and ok
            print(f"{kind:16s} a={a:.1f}: max|r - a {f_exact.__name__} u| "
                  f"= {err:.2e}  arclength {abs(u[0] - u[-1]):.5f} vs "
                  f"{u_end:.5f} {'OK' if ok else 'BAD'}")

    # 3) |f'| <= 1 everywhere on the generated profile -- outside that
    #    bound z' = sqrt(1 - f'^2) is imaginary and there is no surface.
    #    Equivalently: the profile is nowhere steeper than 45 degrees in
    #    dr/du, i.e. |dr| <= |ds| along the meridian.
    for kind in ('MINDING_BULGE', 'MINDING_SPINDLE'):
        for a in (0.2, 0.6, 0.9):
            t, r, z = _minding_profile(kind, a, n=4001)
            ds = np.hypot(np.diff(r), np.diff(z))
            slope = float(np.max(np.abs(np.diff(r)) / np.maximum(ds, 1e-15)))
            ok = slope <= 1.0 + 1e-9 and np.isfinite(z).all()
            ok_all = ok_all and ok
            print(f"{kind:16s} a={a:.1f}: max|dr/ds| = {slope:.6f} "
                  f"(must be <= 1) {'OK' if ok else 'BAD'}")

    # 4) the spindle really closes to a point at both ends, and the
    #    bulge really does not (its waist is r = a)
    t, r, _ = _minding_profile('MINDING_SPINDLE', 0.5, n=4001)
    ok = r[0] < 1e-6 and r[-1] < 1e-6 and abs(r.max()
                                              - math.sqrt(1 - 0.25)) < 1e-6
    ok_all = ok_all and ok
    print(f"spindle ends    : r(tips)={r[0]:.2e}/{r[-1]:.2e} "
          f"r(equator)={r.max():.6f} (want {math.sqrt(0.75):.6f}) "
          f"{'OK' if ok else 'BAD'}")
    t, r, _ = _minding_profile('MINDING_BULGE', 0.5, n=4001)
    ok = abs(r.min() - 0.5) < 1e-9 and abs(r.max()
                                           - math.sqrt(1.25)) < 1e-9
    ok_all = ok_all and ok
    print(f"bulge waist/rim : {r.min():.6f} / {r.max():.6f} "
          f"(want 0.500000 / {math.sqrt(1.25):.6f}) "
          f"{'OK' if ok else 'BAD'}")

    # 5) the mesher runs for every preset and produces finite geometry
    for kind in PRESETS:
        V, F = build_surface(kind, 40, 48, 0.2, 1.0, 0.5)
        ok = len(V) == 40 * 48 and len(F) > 0 and np.isfinite(V).all()
        ok_all = ok_all and ok
        print(f"build {kind:16s}: V={len(V)} F={len(F)} "
              f"{'OK' if ok else 'BAD'}")

    assert ok_all
    print("hyperbolic surface standalone tests passed")
