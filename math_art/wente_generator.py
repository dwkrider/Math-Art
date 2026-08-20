
# Wente Torus Generator for Blender -- a closed CMC torus in R^3.
#
# In 1951 Heinz Hopf conjectured that every closed constant-mean-curvature
# surface in R^3 is a round sphere.  Wente disproved it in 1986 by
# constructing a CMC TORUS.  These are those surfaces.
#
# The construction here follows Walter's explicit description rather than
# a theta-functional or spectral-curve one, and it is short enough to
# state completely.
#
# 1. ISOTHERMIC SETUP (Walter Sect. 1).  A CMC surface without umbilics
#    admits isothermic curvature-line coordinates in which the Hopf
#    differential is constant; normalising it to 1 gives
#
#        I  = E (dx^2 + dy^2) ,   E = e^F / H ,
#        II = L dx^2 + N dy^2 ,   L = e^F + 1 ,  N = e^F - 1 ,  M = 0 ,
#
#    so the principal curvatures are H(1 +- e^{-F}) -- mean curvature H by
#    construction -- and the Gauss equation reduces to SINH-GORDON,
#
#        Laplacian F + 4 H sinh(F) = 0 .
#
# 2. THE SOLUTION.  Walter's separable ansatz solves it in Jacobi cn:
#
#        F = 4 artanh( gamma cn_k(alpha x) * gammabar cn_kbar(alphabar y) )
#
#    with, for a shape angle theta and a second angle thetabar,
#
#        k = sin(theta) ,          kbar = sin(thetabar) ,
#        gamma = sqrt(tan theta) , gammabar = sqrt(tan thetabar) ,
#        alpha    = sqrt( 4H sin(2 thetabar) / sin(2(theta + thetabar)) ) ,
#        alphabar = sqrt( 4H sin(2 theta)    / sin(2(theta + thetabar)) ) .
#
#    Note Walter's "Tg" in that formula is the HYPERBOLIC tangent; his
#    capital-letter trig symbols are all hyperbolic, which is an easy way
#    to build the wrong surface.
#
# 3. THE IMMERSION.  Rather than Walter's (6.16) -- whose auxiliary
#    constants b and p are not needed -- the surface is recovered by
#    integrating the frame directly.  With e1 = X_x/sqrt(E),
#    e2 = X_y/sqrt(E), e3 = n, and using L/sqrt(E) = 2 sqrt(H) cosh(F/2),
#    N/sqrt(E) = 2 sqrt(H) sinh(F/2), sigma_x = F_x/2:
#
#      d_x: e1' = -(F_y/2) e2 + 2 sqrt(H) cosh(F/2) e3
#           e2' =  (F_y/2) e1
#           e3' = -2 sqrt(H) cosh(F/2) e1
#      d_y: e1' =  (F_x/2) e2
#           e2' = -(F_x/2) e1 + 2 sqrt(H) sinh(F/2) e3
#           e3' = -2 sqrt(H) sinh(F/2) e2
#      X_x = sqrt(E) e1 ,   X_y = sqrt(E) e2 .
#
#    Along y = 0 the cn in y is at its maximum so F_y = 0, the x-equations
#    reduce to a rotation in the (e1, e3) plane, and e2 is constant -- the
#    x-curve there is PLANAR.  That is Walter's "plane y-curves" property
#    falling out of the frame system, and it doubles as an exact initial
#    condition for the y-integration.
#
# 4. CLOSURE, which is what makes it a torus rather than a strip.  Over
#    one x-period the surface is carried to itself by a rigid motion.
#    Two things have to be true for that motion to generate a closed
#    surface, and both are checked in the self-test rather than assumed:
#
#      * the motion must be a pure ROTATION, not a screw.  This is what
#        fixes thetabar = 65.354955354 degrees, and it holds here to
#        1e-16 -- the screw component along the rotation axis vanishes at
#        every theta.
#      * the rotation angle Omega must be a rational multiple of 2 pi.
#        Omega increases monotonically from 0 to 360 degrees as theta
#        runs over (0, 24.645044646), so each rational
#        l/n = 1 + Omega/360 in the open interval ]1, 2[ is hit exactly
#        once -- which is Walter's (6.A), recovered numerically here.
#        (The two angles sum to 90 degrees, which is not a coincidence.)
#
#    Omega is obtained from the trace of the period rotation, which gives
#    it folded into [0, 180].  The fold point is located by maximising
#    that folded angle -- there Omega = 180 exactly and the branches meet
#    -- and above it Omega = 360 - folded.  Reading the angle off an
#    eigenvector instead does NOT work: the eigenvector's sign is
#    arbitrary and flips Omega to 360 - Omega unpredictably.
#
# References:
# - Henry C. Wente, "Counterexample to a conjecture of H. Hopf", Pacific
#   J. Math. 121 (1986), 193-243 -- the existence proof.
# - Rolf Walter, "Explicit examples to the H-problem of Heinz Hopf",
#   Geom. Dedicata 23 (1987), 187-213 -- the isothermic description,
#   the separable solution (0.2) used here, and the closure condition
#   l/n in ]1,2[ (6.A).
# - Wayne Rossman, "The Morse Index of Wente Tori", arXiv:0804.4193
#   (2008), Lemma 2.2 -- the constants k, kbar, gamma, gammabar, alpha,
#   alphabar in terms of theta and thetabar, used verbatim above.
# - Uwe Abresch, "Constant mean curvature tori in terms of elliptic
#   functions", J. reine angew. Math. 374 (1987), 169-192 -- the same
#   family, indexed by the same rational, assembled from congruent
#   pieces.
# - A. I. Bobenko, "All constant mean curvature tori in R^3, S^3, H^3 in
#   terms of theta-functions", Math. Ann. 290 (1991), 209-245 -- the
#   general theory these are the simplest case of.

bl_info = {
    "name": "Wente Torus",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Wente Torus",
    "description": "Closed constant-mean-curvature tori -- Wente's "
                   "counterexample to the Hopf conjecture",
    "category": "Add Mesh",
}

import math

import numpy as np

from .minsurf.elliptic import ellipk, jacobi_sncndn

H_FIXED = 0.5

# Fixed by the translational period problem (Walter; Rossman Sect. 2).
# Its complement, 24.645044646 degrees, is the upper end of the theta
# range -- the two sum to exactly 90.
THETA_BAR = math.radians(65.354955354)
THETA_MAX = math.radians(24.645044646)


def constants(theta, H=H_FIXED):
    """(k, kbar, gamma, gammabar, alpha, alphabar) -- Rossman Lemma 2.2."""
    tb = THETA_BAR
    k, kb = math.sin(theta), math.sin(tb)
    g, gb = math.sqrt(math.tan(theta)), math.sqrt(math.tan(tb))
    den = math.sin(2.0 * (theta + tb))
    al = math.sqrt(4.0 * H * math.sin(2.0 * tb) / den)
    alb = math.sqrt(4.0 * H * math.sin(2.0 * theta) / den)
    return k, kb, g, gb, al, alb


def periods(theta, H=H_FIXED):
    """(x, y) periods of F: cn has period 4K."""
    k, kb, _, _, al, alb = constants(theta, H)
    return 4.0 * ellipk(k * k) / al, 4.0 * ellipk(kb * kb) / alb


def _F_and_derivs(theta, H=H_FIXED):
    """Callables F, F_x, F_y for Walter's separable solution."""
    k, kb, g, gb, al, alb = constants(theta, H)

    def AB(x, y):
        cx = jacobi_sncndn(al * np.asarray(x, float), k * k)[1]
        cy = jacobi_sncndn(alb * np.asarray(y, float), kb * kb)[1]
        return g * cx, gb * cy

    def dA(x):
        sn, _, dn = jacobi_sncndn(al * np.asarray(x, float), k * k)
        return -g * al * sn * dn

    def dB(y):
        sn, _, dn = jacobi_sncndn(alb * np.asarray(y, float), kb * kb)
        return -gb * alb * sn * dn

    def F(x, y):
        a, b = AB(x, y)
        return 4.0 * np.arctanh(np.clip(a * b, -1.0 + 1e-13, 1.0 - 1e-13))

    def Fx(x, y):
        a, b = AB(x, y)
        p = a * b
        return 4.0 * dA(x) * b / (1.0 - p * p)

    def Fy(x, y):
        a, b = AB(x, y)
        p = a * b
        return 4.0 * a * dB(y) / (1.0 - p * p)

    return F, Fx, Fy


def _rk4(state, t, h, deriv):
    k1 = deriv(state, t)
    s2 = tuple(state[i] + 0.5 * h * k1[i] for i in range(4))
    k2 = deriv(s2, t + 0.5 * h)
    s3 = tuple(state[i] + 0.5 * h * k2[i] for i in range(4))
    k3 = deriv(s3, t + 0.5 * h)
    s4 = tuple(state[i] + h * k3[i] for i in range(4))
    k4 = deriv(s4, t + h)
    return tuple(state[i] + (h / 6.0)
                 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4))


def immersion(theta, x, y, H=H_FIXED):
    """Integrate the frame system and return X of shape (len(x), len(y), 3)."""
    F, Fx, Fy = _F_and_derivs(theta, H)
    rt = math.sqrt(H)

    i0 = int(np.argmin(np.abs(x)))

    def dx_(st, xx):
        e1, e2, e3, _ = st
        f = F(xx, 0.0)
        fy = Fy(xx, 0.0)
        c = 2.0 * rt * np.cosh(f / 2.0)
        se = np.exp(f / 2.0) / rt
        return (-(fy / 2.0) * e2 + c * e3, (fy / 2.0) * e1, -c * e1, se * e1)

    row = [None] * len(x)
    row[i0] = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
               np.array([0.0, 0.0, 1.0]), np.zeros(3))
    for i in range(i0, len(x) - 1):
        row[i + 1] = _rk4(row[i], x[i], x[i + 1] - x[i], dx_)
    for i in range(i0, 0, -1):
        row[i - 1] = _rk4(row[i], x[i], x[i - 1] - x[i], dx_)

    E1 = np.array([r[0] for r in row])
    E2 = np.array([r[1] for r in row])
    E3 = np.array([r[2] for r in row])
    XX = np.array([r[3] for r in row])

    def dy_(st, yy):
        e1, e2, e3, _ = st
        f = F(x, yy)[:, None]
        fx = Fx(x, yy)[:, None]
        sh = 2.0 * rt * np.sinh(f / 2.0)
        se = np.exp(f / 2.0) / rt
        return ((fx / 2.0) * e2, -(fx / 2.0) * e1 + sh * e3, -sh * e2,
                se * e2)

    j0 = int(np.argmin(np.abs(y)))
    out = [None] * len(y)
    out[j0] = XX.copy()
    cur = (E1, E2, E3, XX)
    for j in range(j0, len(y) - 1):
        cur = _rk4(cur, y[j], y[j + 1] - y[j], dy_)
        out[j + 1] = cur[3].copy()
    cur = (E1, E2, E3, XX)
    for j in range(j0, 0, -1):
        cur = _rk4(cur, y[j], y[j - 1] - y[j], dy_)
        out[j - 1] = cur[3].copy()
    return np.transpose(np.array(out), (1, 0, 2))


def period_motion(theta, H=H_FIXED, nx=121, ny=121):
    """(R, screw, folded_angle) of the rigid motion carrying the y-curve
    at x = 0 to the one at x = one x-period."""
    Px, Py = periods(theta, H)
    x = np.linspace(0.0, Px, nx)
    y = np.linspace(-0.5 * Py, 0.5 * Py, ny)
    X = immersion(theta, x, y, H)
    A, B = X[0], X[-1]
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        Vt[-1] *= -1.0
        R = U @ Vt
    w, v = np.linalg.eig(R)
    ax = np.real(v[:, int(np.argmin(np.abs(w - 1.0)))])
    ax /= np.linalg.norm(ax)
    screw = float(np.dot(cb - R.T @ ca, ax))
    folded = math.degrees(math.acos(
        max(-1.0, min(1.0, 0.5 * (np.trace(R) - 1.0)))))
    return R, screw, folded


_FOLD_CACHE = {}


def fold_point(H=H_FIXED):
    """The theta at which the period rotation is exactly 180 degrees.

    Found by maximising the folded angle, because the trace only ever
    gives Omega folded into [0, 180]; the fold point is where the two
    branches meet.  Cached -- it costs about sixty surface integrations."""
    if H in _FOLD_CACHE:
        return _FOLD_CACHE[H]
    gr = 0.5 * (math.sqrt(5.0) - 1.0)
    lo, hi = math.radians(14.0), math.radians(21.0)
    for _ in range(48):
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        if period_motion(c, H)[2] > period_motion(d, H)[2]:
            hi = d
        else:
            lo = c
    _FOLD_CACHE[H] = 0.5 * (lo + hi)
    return _FOLD_CACHE[H]


def rotation_angle(theta, H=H_FIXED):
    """Period rotation in degrees, unfolded onto (0, 360)."""
    folded = period_motion(theta, H)[2]
    return folded if theta <= fold_point(H) else 360.0 - folded


def theta_for(l, n, H=H_FIXED, iters=48):
    """Solve for the shape angle giving a closed torus with l/n in ]1,2[.

    Omega increases monotonically from 0 to 360 degrees across the theta
    range, so plain bisection cannot miss."""
    ratio = l / n
    if not 1.0 < ratio < 2.0:
        raise ValueError(
            f"l/n = {l}/{n} = {ratio:.4f} is outside ]1, 2[; Walter (6.A) "
            f"gives a closed Wente torus only for a ratio in that open "
            f"interval")
    target = 360.0 * (ratio - 1.0)
    lo, hi = math.radians(0.2), THETA_MAX - 1e-4
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rotation_angle(mid, H) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_surface(l=5, n=4, ures=240, vres=120, scale=1.0, H=H_FIXED):
    """Mesh a closed Wente torus: n x-periods by one y-period."""
    theta = theta_for(l, n, H)
    Px, Py = periods(theta, H)
    x = np.linspace(0.0, n * Px, ures, endpoint=False)
    y = np.linspace(-0.5 * Py, 0.5 * Py, vres, endpoint=False)
    X = immersion(theta, x, y, H).reshape(-1, 3)
    faces = []
    for i in range(ures):
        i1 = (i + 1) % ures
        for j in range(vres):
            j1 = (j + 1) % vres
            faces.append((i * vres + j, i * vres + j1,
                          i1 * vres + j1, i1 * vres + j))
    lo, hi = X.min(axis=0), X.max(axis=0)
    ext = float((hi - lo).max())
    X = (X - 0.5 * (lo + hi)) * ((2.0 / ext if ext > 1e-9 else 1.0) * scale)
    return X, faces, {'theta': math.degrees(theta), 'Px': Px, 'Py': Py,
                      'omega': rotation_angle(theta, H)}


def mean_curvature_grid(X, hx, hy):
    """H of a surface sampled on a uniform (x, y) grid, by central
    differences; interior points only."""
    Xu = (X[2:, 1:-1] - X[:-2, 1:-1]) / (2 * hx)
    Xv = (X[1:-1, 2:] - X[1:-1, :-2]) / (2 * hy)
    Xuu = (X[2:, 1:-1] - 2 * X[1:-1, 1:-1] + X[:-2, 1:-1]) / hx ** 2
    Xvv = (X[1:-1, 2:] - 2 * X[1:-1, 1:-1] + X[1:-1, :-2]) / hy ** 2
    Xuv = (X[2:, 2:] - X[2:, :-2] - X[:-2, 2:] + X[:-2, :-2]) / (4 * hx * hy)
    nrm = np.cross(Xu, Xv)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True),
                           1e-300)
    E = (Xu * Xu).sum(-1)
    F = (Xu * Xv).sum(-1)
    G = (Xv * Xv).sum(-1)
    L = (Xuu * nrm).sum(-1)
    M = (Xuv * nrm).sum(-1)
    N = (Xvv * nrm).sum(-1)
    return (E * N - 2 * F * M + G * L) / (2 * (E * G - F * F))


# ==========================================================================
# Blender layer
# ==========================================================================

try:
    import bpy
    from bpy.props import IntProperty, FloatProperty, BoolProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_wente_torus_add(bpy.types.Operator):
        """Add a Wente torus: a closed constant-mean-curvature torus,
        the counterexample to Hopf's conjecture"""
        bl_idname = "mesh.wente_torus_add"
        bl_label = "Wente Torus"
        bl_options = {'REGISTER', 'UNDO'}

        lobes_l: IntProperty(
            name="l", default=5, min=2, max=24,
            description="Numerator of the closure fraction l/n, which "
                        "must lie strictly between 1 and 2.  Walter's "
                        "own figures use 5/4")
        lobes_n: IntProperty(
            name="n", default=4, min=1, max=24,
            description="Denominator of the closure fraction: the torus "
                        "closes after n periods of the profile")
        ures: IntProperty(name="Along Profile", default=240, min=24,
                          max=1200)
        vres: IntProperty(name="Around", default=120, min=12, max=800)
        shade_smooth: BoolProperty(name="Smooth Shading", default=True)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            ratio = self.lobes_l / self.lobes_n
            if not 1.0 < ratio < 2.0:
                self.report(
                    {'ERROR'},
                    f"l/n = {self.lobes_l}/{self.lobes_n} = {ratio:.4f} "
                    f"must lie strictly between 1 and 2 -- outside that "
                    f"range Walter's period condition has no solution")
                return {'CANCELLED'}
            verts, faces, info = build_surface(
                self.lobes_l, self.lobes_n, self.ures, self.vres,
                self.scale)
            me = bpy.data.meshes.new("Wente Torus")
            me.from_pydata([tuple(v) for v in verts], [],
                           [tuple(int(i) for i in f) for f in faces])
            me.validate(clean_customdata=True)
            if self.shade_smooth:
                me.polygons.foreach_set('use_smooth',
                                        [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Wente Torus", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report(
                {'INFO'},
                f"Wente torus l/n = {self.lobes_l}/{self.lobes_n}: "
                f"V={len(me.vertices)} F={len(me.polygons)}, shape angle "
                f"{info['theta']:.5f} deg, period rotation "
                f"{info['omega']:.4f} deg, H = {H_FIXED}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'lobes_l')
            lay.prop(self, 'lobes_n')
            ratio = self.lobes_l / max(self.lobes_n, 1)
            if not 1.0 < ratio < 2.0:
                lay.label(text="l/n must be strictly between 1 and 2",
                          icon='ERROR')
            lay.prop(self, 'ures')
            lay.prop(self, 'vres')
            lay.prop(self, 'shade_smooth')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.wente_torus_add", icon='MESH_TORUS')

    ADD_MENU = True    # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_wente_torus_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_wente_torus_add)


def _selftest():
    ok_all = True

    # 1) Walter's separable F really does solve sinh-Gordon.  This is the
    #    cheapest and sharpest check of the CONSTANTS, which were the
    #    missing piece for years of this plan: Rossman's Lemma 2.2
    #    supplies them and this confirms them.
    for H in (0.5, 1.0):
        for thd in (5.0, 12.0, 18.0, 23.0):
            F, _, _ = _F_and_derivs(math.radians(thd), H)
            rng = np.random.default_rng(11)
            xs = rng.uniform(-1.0, 1.0, 300)
            ys = rng.uniform(-1.0, 1.0, 300)
            prev, rate = None, 0.0
            for h in (2e-3, 1e-3):
                lap = ((F(xs + h, ys) - 2 * F(xs, ys) + F(xs - h, ys))
                       + (F(xs, ys + h) - 2 * F(xs, ys)
                          + F(xs, ys - h))) / (h * h)
                res = float(np.abs(lap + 4.0 * H * np.sinh(F(xs, ys))).max())
                if prev is not None:
                    rate = math.log2(prev / res)
                prev = res
            ok = rate > 1.8
            ok_all = ok_all and ok
            print(f"sinh-Gordon H={H} theta={thd:5.1f}: residual "
                  f"{prev:.2e}, convergence rate {rate:+.2f} (want 2, so "
                  f"it is truncation) {'OK' if ok else 'BAD'}")

    # 2) The period map is a pure ROTATION -- no screw.  This is the
    #    translational period problem, and it is what pins
    #    thetabar = 65.354955354 degrees.  Assumed by the construction,
    #    so it has to be measured.
    for thd in (4.0, 10.0, 16.0, 22.0):
        _, screw, folded = period_motion(math.radians(thd))
        ok = abs(screw) < 1e-9
        ok_all = ok_all and ok
        print(f"pure rotation theta={thd:5.1f}: screw along axis "
              f"{screw:+.2e} (want 0), folded angle {folded:8.4f} "
              f"{'OK' if ok else 'BAD'}")

    # 3) The rotation increases monotonically across the whole theta
    #    range, from 0 to 360 degrees -- so l/n = 1 + Omega/360 sweeps
    #    exactly ]1, 2[ and every admissible fraction is hit once.  That
    #    IS Walter's (6.A), recovered rather than quoted.
    ths = np.linspace(math.radians(1.0), THETA_MAX - 1e-3, 26)
    om = np.array([rotation_angle(t) for t in ths])
    ok = bool(np.all(np.diff(om) > 0.0)) and om[0] < 15.0 and om[-1] > 330.0
    ok_all = ok_all and ok
    print(f"monotone rotation: Omega spans {om[0]:.2f} -> {om[-1]:.2f} deg "
          f"over the theta range, monotone={bool(np.all(np.diff(om) > 0))} "
          f"{'OK' if ok else 'BAD'}")

    # 4) The closure solver hits its target, including l/n = 3/2 which
    #    lands exactly on the fold where the two branches meet.
    for (l, n) in ((5, 4), (4, 3), (3, 2), (5, 3), (7, 4), (8, 5)):
        th = theta_for(l, n)
        got = rotation_angle(th)
        want = 360.0 * (l / n - 1.0)
        ok = abs(got - want) < 1e-6
        ok_all = ok_all and ok
        print(f"closure l/n={l}/{n}: theta={math.degrees(th):9.6f} deg, "
              f"Omega={got:.6f} (want {want:.4f}) "
              f"{'OK' if ok else 'BAD'}")

    # 5) THE gate: constant mean curvature, measured in R^3 off the mesh
    #    with no reference to the construction.
    for (l, n) in ((5, 4), (4, 3), (5, 3)):
        th = theta_for(l, n)
        Px, Py = periods(th)
        nx = ny = 141
        x = np.linspace(0.0, Px, nx)
        y = np.linspace(-0.5 * Py, 0.5 * Py, ny)
        X = immersion(th, x, y)
        Hm = mean_curvature_grid(X, x[1] - x[0], y[1] - y[0])
        Hm = Hm[np.isfinite(Hm)]
        med = float(np.median(Hm))
        q1, q3 = np.percentile(Hm, [25.0, 75.0])
        ok = abs(med - H_FIXED) < 5e-3 and (q3 - q1) < 5e-3
        ok_all = ok_all and ok
        print(f"CMC l/n={l}/{n}: H median {med:+.6f} (want {H_FIXED}) "
              f"IQR {q3 - q1:.1e} {'OK' if ok else 'BAD'}")

    # 6) The torus actually CLOSES: after n x-periods the surface returns
    #    to itself.  This is the whole point -- a Wente strip is easy, a
    #    Wente torus is not.
    for (l, n) in ((5, 4), (4, 3)):
        th = theta_for(l, n)
        Px, Py = periods(th)
        y = np.linspace(-0.5 * Py, 0.5 * Py, 81)
        # a UNIFORM grid spanning the n periods: the frame is integrated
        # along this grid, so a sparse or unequal one would ask RK4 to
        # leap a whole period in a single step
        x = np.linspace(0.0, n * Px, 60 * n + 1)
        A = immersion(th, x, y)
        start, end = A[0], A[-1]
        gap = float(np.abs(start - end).max())
        span = float(np.abs(start).max())
        ok = gap / max(span, 1e-9) < 5e-3
        ok_all = ok_all and ok
        print(f"closes l/n={l}/{n}: |X(0,y) - X({n} periods, y)| = "
              f"{gap:.2e} against a surface of size {span:.3f} "
              f"{'OK' if ok else 'BAD'}")

    # 7) l/n outside ]1,2[ has no solution and must be refused
    refused = 0
    for (l, n) in ((1, 1), (2, 1), (1, 2), (5, 2)):
        try:
            theta_for(l, n)
        except ValueError:
            refused += 1
    ok = refused == 4
    ok_all = ok_all and ok
    print(f"guards: {refused}/4 out-of-range fractions refused "
          f"{'OK' if ok else 'BAD'}")

    assert ok_all
    print("wente torus standalone tests passed")
