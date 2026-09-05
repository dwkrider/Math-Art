
# Calabi-Yau Metrics Generator for Blender
#
# Yau's theorem says a Ricci-flat Kaehler metric exists on every
# compact Calabi-Yau, and gives no formula for it -- which is why the
# other Calabi-Yau generators here draw complex-algebraic SHAPE rather
# than metric.  The non-compact Calabi-Yaus are the exception: a
# handful of them have metrics in closed form, and those are what this
# module draws.  Both modes are two-dimensional slices, but they are
# EXACT ones: the geometry built is isometric to the slice, not a
# sketch of it.
#
# EGUCHI-HANSON SPACE.  The first asymptotically locally Euclidean
# Ricci-flat Kaehler metric, and the simplest non-compact Calabi-Yau
# 2-fold:
#
#     ds^2 = (1 - (a/r)^4)^-1 dr^2
#            + (r^2/4)[ sx^2 + sy^2 + (1 - (a/r)^4) sz^2 ]
#
# on T*S^2, the resolution of C^2/Z2.  Holding the two-sphere
# directions fixed leaves the (r, psi) slice
#
#     ds^2 = (1 - (a/r)^4)^-1 dr^2 + (r^2/4)(1 - (a/r)^4) dpsi^2 / 4
#
# which IS isometrically embeddable in R^3, as a surface of revolution
# with profile
#
#     rho(r) = sqrt(r^4 - a^4) / (2r),
#     dz/dr  = sqrt(3 r^4 + a^4) / (2 r^2).
#
# That z comes out real is not luck: rho'^2 + z'^2 works out to
# 4r^8 / (4 r^4 (r^4 - a^4)) = 1/(1 - (a/r)^4), exactly g_rr, and the
# check is asserted below.  The point of the picture is the tip.  As
# r -> a the psi-circle closes off smoothly, because
# drho/ds = (r^4 + a^4)/(2r^4) -> 1 there -- but only if psi has period
# 2 pi.  Eguchi and Hanson's original psi runs to 4 pi on S^3, so the
# metric is smooth only after restricting "the volume measure to one
# half the range of S^3", leaving RP(3) = SO(3) at infinity.  Take the
# full range and the tip is a cone of angle 4 pi, which no surface in
# R^3 can carry: the Z2 quotient is visible as the difference between a
# smooth cap and an impossible one.
#
# THE CONIFOLD.  The conifold is the quadric cone sum (w^A)^2 = 0 in
# C^4 -- six real dimensions, a cone over S^2 x S^3 -- and its node can
# be repaired two ways: by DEFORMATION, sum (w^A)^2 = eps^2, which
# replaces the point with an S^3, or by a SMALL RESOLUTION, which
# replaces it with an S^2 = P^1.  Passing from one to the other changes
# the topology while staying Calabi-Yau, and the local metrics on both
# sides are known in closed form.  Neither fits in R^3.  What does fit,
# and is the same story one complex dimension down, is the A_1 surface
#
#     x y = z^2  in C^3,   real points  u^2 - v^2 - z^2 = delta
#
# whose node is likewise repaired by a deformation or a resolution,
# both replacing it with a sphere.  delta = 0 is the double cone;
# delta < 0 opens a throat of radius sqrt(-delta) -- the real slice of
# the sphere that replaces the node -- and delta > 0 separates the two
# sheets.  Eguchi-Hanson space is the resolution of exactly this
# singularity, so the two modes of this generator are the same
# geometry told metrically and algebraically.
#
# References:
# - T. Eguchi and A. J. Hanson, "Asymptotically flat self-dual
#   solutions to Euclidean gravity", Physics Letters 74B (1978)
#   249-251 -- the metric (their solution II, Eqs. (9) and (17)), the
#   removal of the apparent singularity at r = a by the change of
#   variable u^2 = r^2(1 - (a/r)^4), and the "corner" footnote on the
#   half-range of S^3.
# - T. Eguchi, P. B. Gilkey and A. J. Hanson, "Gravitation, gauge
#   theories and differential geometry", Physics Reports 66 (1980)
#   213-393 -- the standard review, including the T*S^2 topology.
# - A. J. Hanson, "Visualizing the Eguchi-Hanson Space", Visualization
#   Workshop, University of Tokyo, March 2020 -- the visualization
#   problem this addresses, and an 11-dimensional Nash embedding of
#   the whole 4-manifold.
# - P. Candelas and X. C. de la Ossa, "Comments on conifolds", Nuclear
#   Physics B342 (1990) 246-268 -- the conifold, its deformation
#   Eq. (1.2) and small resolution Eq. (1.4), the Euler-characteristic
#   relation (1.6), and the explicit Ricci-flat Kaehler metrics on
#   both.
# - E. Brieskorn, "Ueber die Aufloesung gewisser Singularitaeten von
#   holomorphen Abbildungen", Mathematische Annalen 166 (1966) 76-102
#   -- the A_1 singularity C^2/Z2 and its resolution.

import math

import numpy as np

bl_info = {
    "name": "Calabi-Yau Metrics",
    "author": "Math Art project (after Eguchi & Hanson / Candelas & "
              "de la Ossa)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Calabi-Yau Metrics",
    "description": "Eguchi-Hanson space and the conifold transition",
    "category": "Add Mesh",
}

try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# --------------------------------------------------------------------
# Eguchi-Hanson space
# --------------------------------------------------------------------

def eh_rho(r, a=1.0):
    """Radius of the psi-circle of the Eguchi-Hanson metric."""
    r = np.asarray(r, dtype=float)
    return np.sqrt(np.maximum(r ** 4 - a ** 4, 0.0)) / (2.0 * r)


def eh_drho(r, a=1.0):
    """d rho / d r."""
    r = np.asarray(r, dtype=float)
    return (r ** 4 + a ** 4) / (2.0 * r ** 2 *
                                np.sqrt(np.maximum(r ** 4 - a ** 4,
                                                   1e-300)))


def eh_dz(r, a=1.0):
    """d z / d r for the isometric surface of revolution.

    Fixed by rho'^2 + z'^2 = g_rr = (1 - (a/r)^4)^-1; the algebra
    collapses to a perfect square, 4 r^8 in the numerator, so
    z' = sqrt(3 r^4 + a^4) / (2 r^2) with no radical left over.
    """
    r = np.asarray(r, dtype=float)
    return np.sqrt(3.0 * r ** 4 + a ** 4) / (2.0 * r ** 2)


def eh_grr(r, a=1.0):
    """The metric coefficient g_rr the embedding has to reproduce."""
    r = np.asarray(r, dtype=float)
    return 1.0 / (1.0 - (a / r) ** 4)


def eh_profile(a=1.0, r_max=3.0, steps=192):
    """(r, rho, z) along the bolt slice, z integrated from the tip.

    The integrand z'(r) is finite everywhere including r = a, so plain
    Simpson on a uniform grid is enough; it is rho' that blows up
    there, and rho itself is known in closed form.
    """
    r = np.linspace(a, r_max, steps)
    dz = eh_dz(r, a)
    z = np.zeros_like(r)
    h = r[1] - r[0]
    z[1:] = np.cumsum(0.5 * h * (dz[:-1] + dz[1:]))
    return r, eh_rho(r, a), z


def eh_surface(a=1.0, r_max=3.0, steps=96, segments=64, bolt=False):
    """The Eguchi-Hanson bolt slice as a surface of revolution."""
    r, rho, z = eh_profile(a, r_max, steps)
    verts, faces = [], []
    # The tip is a single point: rho(a) = 0.
    verts.append((0.0, 0.0, float(z[0])))
    for i in range(1, len(r)):
        for j in range(segments):
            t = 2.0 * math.pi * j / segments
            verts.append((float(rho[i]) * math.cos(t),
                          float(rho[i]) * math.sin(t), float(z[i])))
    for j in range(segments):
        faces.append((0, 1 + j, 1 + (j + 1) % segments))
    for i in range(1, len(r) - 1):
        b0 = 1 + (i - 1) * segments
        b1 = 1 + i * segments
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((b0 + j, b1 + j, b1 + k, b0 + k))
    if bolt:
        # The two-sphere that survives at r = a.  Its metric there is
        # (r^2/16)(dtheta^2 + sin^2 theta dphi^2), so its radius is
        # a/4 -- the "bolt" the space is named for.
        base = len(verts)
        rings, segs = 24, 32
        cz = float(z[0])
        for i in range(rings + 1):
            th = math.pi * i / rings
            for j in range(segs):
                ph = 2.0 * math.pi * j / segs
                verts.append((0.25 * a * math.sin(th) * math.cos(ph),
                              0.25 * a * math.sin(th) * math.sin(ph),
                              cz + 0.25 * a * math.cos(th)))
        for i in range(rings):
            for j in range(segs):
                k = (j + 1) % segs
                faces.append((base + i * segs + j, base + i * segs + k,
                              base + (i + 1) * segs + k,
                              base + (i + 1) * segs + j))
    return verts, faces


def eh_tip_angle(a=1.0, eps=1e-7):
    """The cone angle at r = a, in units of the psi period.

    drho/ds -> 1 means the cap closes smoothly when psi runs over
    2 pi; the value is (r^4 + a^4)/(2 r^4) -> 1 as r -> a.
    """
    r = a * (1.0 + eps)
    return float(eh_drho(r, a) / math.sqrt(eh_grr(r, a)))


# --------------------------------------------------------------------
# The conifold, in its two-dimensional model
# --------------------------------------------------------------------

def conifold_surface(delta=0.0, extent=1.5, steps=64, segments=64):
    """Real points of u^2 - v^2 - z^2 = delta, as a mesh.

    delta = 0 is the double cone; delta < 0 is one connected surface
    with a throat of radius sqrt(-delta) -- the real slice of the
    sphere that replaces the node; delta > 0 is two separate sheets.
    """
    verts, faces = [], []
    s_min = math.sqrt(-delta) if delta < 0.0 else 0.0
    s = np.linspace(s_min, extent, steps)

    if delta < 0.0:
        # One sheet: u runs from -sqrt(s^2+delta) up through the
        # throat at u = 0 and out again, so sweep s down and back.
        u = np.sqrt(np.maximum(s ** 2 + delta, 0.0))
        rs = np.concatenate((s[::-1], s[1:]))
        us = np.concatenate((-u[::-1], u[1:]))
        rings = [(float(rr), float(uu)) for rr, uu in zip(rs, us)]
        for rr, uu in rings:
            for j in range(segments):
                t = 2.0 * math.pi * j / segments
                verts.append((float(uu), rr * math.cos(t),
                              rr * math.sin(t)))
        for i in range(len(rings) - 1):
            for j in range(segments):
                k = (j + 1) % segments
                faces.append((i * segments + j, (i + 1) * segments + j,
                              (i + 1) * segments + k, i * segments + k))
        return verts, faces

    # delta >= 0: two sheets, or -- at delta = 0 exactly -- the two
    # nappes of one cone.  There the apexes coincide at the origin and
    # are emitted as a single vertex, so the mesh is connected through
    # the node the way the variety is.
    u = np.sqrt(s ** 2 + delta)
    node = None
    for sign in (1.0, -1.0):
        base = len(verts)
        if delta == 0.0 and node is not None:
            apex = node
        else:
            apex = base
            verts.append((float(sign * u[0]), 0.0, 0.0))
            if delta == 0.0:
                node = apex
        base = len(verts) - 1
        for i in range(1, len(s)):
            for j in range(segments):
                t = 2.0 * math.pi * j / segments
                verts.append((float(sign * u[i]),
                              float(s[i]) * math.cos(t),
                              float(s[i]) * math.sin(t)))
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((apex, base + 1 + j, base + 1 + k))
        for i in range(1, len(s) - 1):
            b0 = base + 1 + (i - 1) * segments
            b1 = base + 1 + i * segments
            for j in range(segments):
                k = (j + 1) % segments
                faces.append((b0 + j, b1 + j, b1 + k, b0 + k))
    return verts, faces


def conifold_residual(verts, delta):
    """max |u^2 - v^2 - z^2 - delta| over the mesh."""
    if not verts:
        return 0.0
    V = np.asarray(verts, dtype=float)
    return float(np.max(np.abs(V[:, 0] ** 2 - V[:, 1] ** 2
                               - V[:, 2] ** 2 - delta)))


def components(nverts, faces):
    """Number of connected components of a face list."""
    parent = list(range(nverts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in faces:
        for i in range(1, len(f)):
            a, b = find(f[0]), find(f[i])
            if a != b:
                parent[a] = b
    used = {i for f in faces for i in f}
    return len({find(i) for i in used})


def fit(pts, scale=1.0):
    """Centre on the bounding-box midpoint, largest extent 2*scale."""
    if not pts:
        return pts
    A = np.asarray(pts, dtype=float)
    lo, hi = A.min(axis=0), A.max(axis=0)
    ext = float((hi - lo).max())
    s = (2.0 * scale / ext) if ext > 1e-12 else 1.0
    return [tuple(v) for v in (A - 0.5 * (lo + hi)) * s]


# --------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------

if _IN_BLENDER:

    class MESH_OT_calabi_yau_metric_add(bpy.types.Operator):
        """Add an exactly embedded slice of a Ricci-flat metric"""
        bl_idname = "mesh.calabi_yau_metric_add"
        bl_label = "Calabi-Yau Metrics"
        bl_options = {'REGISTER', 'UNDO'}

        mode: EnumProperty(
            name="Space",
            description="Which Ricci-flat geometry to slice",
            items=[('EGUCHI_HANSON', "Eguchi-Hanson Space",
                    "The bolt slice of the Eguchi-Hanson metric, "
                    "embedded isometrically as a surface of "
                    "revolution. The tip closes smoothly only because "
                    "the angle runs over half of the three-sphere"),
                   ('CONIFOLD', "Conifold Transition",
                    "The two-dimensional model of the conifold, "
                    "u^2 - v^2 - z^2 = delta: a node at delta = 0, "
                    "replaced by a sphere on one side and separating "
                    "into two sheets on the other")],
            default='EGUCHI_HANSON')
        bolt_size: FloatProperty(
            name="Bolt Size", default=1.0, min=0.05, max=3.0,
            description="The parameter a: the size at which the "
                        "geometry stops, where the two-sphere sits")
        reach: FloatProperty(
            name="Reach", default=3.0, min=1.05, max=12.0,
            description="How far out along r to build, in units of the "
                        "bolt size. Far out the metric is flat")
        show_bolt: BoolProperty(
            name="Show Bolt", default=True,
            description="Also draw the two-sphere at the tip, at its "
                        "true radius a/4 -- the sphere that replaces "
                        "the singular point of C^2/Z2")
        delta: FloatProperty(
            name="Deformation", default=-0.25, min=-1.0, max=1.0,
            description="Negative opens a throat of radius "
                        "sqrt(-delta), the real slice of the sphere "
                        "that replaces the node; zero is the singular "
                        "cone; positive separates the two sheets")
        extent: FloatProperty(
            name="Extent", default=1.5, min=0.2, max=6.0,
            description="How far out to build the conifold model")
        steps: IntProperty(
            name="Rings", default=96, min=8, max=512,
            description="Samples along the profile")
        segments: IntProperty(
            name="Segments", default=64, min=6, max=256,
            description="Samples around the axis")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size multiplier")

        def execute(self, context):
            if self.mode == 'EGUCHI_HANSON':
                a = self.bolt_size
                verts, faces = eh_surface(a, self.reach * a, self.steps,
                                          self.segments, self.show_bolt)
                name = "Eguchi-Hanson Bolt"
                ang = eh_tip_angle(a)
                msg = ("Eguchi-Hanson: tip closes smoothly "
                       f"(drho/ds = {ang:.6f}) for a psi period of "
                       "2 pi; the full 4 pi range of S^3 would give a "
                       "cone of twice the angle, which R^3 cannot hold")
            else:
                verts, faces = conifold_surface(self.delta, self.extent,
                                                self.steps,
                                                self.segments)
                name = "Conifold"
                res = conifold_residual(verts, self.delta)
                nc = components(len(verts), faces)
                what = ("throat of radius "
                        f"{math.sqrt(-self.delta):.4f}"
                        if self.delta < 0 else
                        "singular cone" if self.delta == 0 else
                        "two separated sheets")
                msg = (f"conifold delta = {self.delta:+.3f}: {what}, "
                       f"{nc} component(s), residual {res:.1e}")

            me = bpy.data.meshes.new(name)
            me.from_pydata(fit(verts, self.scale), [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'mode')
            if self.mode == 'EGUCHI_HANSON':
                lay.prop(self, 'bolt_size')
                lay.prop(self, 'reach')
                lay.prop(self, 'show_bolt')
            else:
                lay.prop(self, 'delta')
                lay.prop(self, 'extent')
            lay.prop(self, 'steps')
            lay.prop(self, 'segments')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.calabi_yau_metric_add",
                             icon='MOD_SIMPLEDEFORM')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_calabi_yau_metric_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_calabi_yau_metric_add)


# --------------------------------------------------------------------

def _selftest():
    bad = []

    # 1. the embedding really is isometric: rho'^2 + z'^2 = g_rr.
    for a in (0.5, 1.0, 2.3):
        r = np.linspace(a * 1.000001, a * 20.0, 4001)
        err = float(np.max(np.abs(eh_drho(r, a) ** 2 + eh_dz(r, a) ** 2
                                  - eh_grr(r, a))))
        rel = err / float(np.max(eh_grr(r, a)))
        ok = rel < 1e-9
        print(f"eguchi_hanson: a={a} isometry |rho'^2+z'^2-g_rr| "
              f"rel = {rel:.2e} {'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"isometry a={a}: {rel:.2e}")

    # 2. the tip closes smoothly: drho/ds -> 1, so the cone angle is
    #    the psi period.  This is the Z2 quotient made visible.
    for a in (0.5, 1.0, 2.3):
        t = eh_tip_angle(a)
        ok = abs(t - 1.0) < 1e-6
        print(f"eguchi_hanson: a={a} tip drho/ds = {t:.9f} (1) "
              f"{'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"tip angle a={a}: {t}")

    # 3. asymptotically flat: rho -> r/2, so the boundary at infinity
    #    is the half-size circle of RP(3) rather than the full S^3.
    a = 1.0
    r = np.array([1e3, 1e4, 1e5])
    rel = float(np.max(np.abs(eh_rho(r, a) / (0.5 * r) - 1.0)))
    ok = rel < 1e-6
    print(f"eguchi_hanson: rho/(r/2) - 1 at large r = {rel:.2e} "
          f"{'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("asymptotics")

    # 4. the built surface is closed at the tip and has the profile
    #    the formulas predict.
    v, f = eh_surface(1.0, 3.0, 64, 32)
    V = np.asarray(v)
    rad = np.hypot(V[:, 0], V[:, 1])
    ok = float(rad.min()) < 1e-12 and len(f) > 0
    print(f"eguchi_hanson: mesh V={len(v)} F={len(f)} tip radius "
          f"{rad.min():.1e} {'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("eh mesh")

    # 5. the conifold model sits exactly on its quadric, and the
    #    component count flips across the transition.
    for delta, want in ((-0.36, 1), (0.0, 1), (0.25, 2)):
        v, f = conifold_surface(delta, 1.5, 48, 48)
        res = conifold_residual(v, delta)
        nc = components(len(v), f)
        ok = res < 1e-12 and nc == want
        print(f"conifold: delta={delta:+.2f} residual={res:.1e} "
              f"components={nc}({want}) {'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"conifold delta={delta}: res={res:.1e} nc={nc}")

    # 6. the throat is exactly the vanishing cycle: its radius is
    #    sqrt(-delta).
    delta = -0.36
    v, _ = conifold_surface(delta, 1.5, 48, 48)
    V = np.asarray(v)
    waist = float(np.min(np.hypot(V[:, 1], V[:, 2])))
    ok = abs(waist - math.sqrt(-delta)) < 1e-9
    print(f"conifold: throat radius {waist:.9f} "
          f"(sqrt(-delta) = {math.sqrt(-delta):.9f}) "
          f"{'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("throat radius")

    # 7. the fit: centred, inside a 2 m cube.
    P = np.asarray(fit(eh_surface(1.0, 3.0, 32, 16)[0]))
    ctr = float(np.abs(0.5 * (P.min(axis=0) + P.max(axis=0))).max())
    ext = float((P.max(axis=0) - P.min(axis=0)).max())
    ok = ctr < 1e-9 and abs(ext - 2.0) < 1e-9
    print(f"cy_metric: fit centre={ctr:.2e} extent={ext:.6f} "
          f"{'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("fit")

    if bad:
        raise AssertionError("; ".join(bad))
