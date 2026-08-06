
# Gomboc Generator for Blender
#
# A gomboc is a convex, homogeneous solid that is "mono-monostatic":
# it has exactly ONE stable and ONE unstable balance point, so -- like
# a self-righting toy, but with no added weight and no hollow -- it
# always rolls back to the same resting pose.  Vladimir Arnold
# conjectured such a body could exist; Gabor Domokos and Peter
# Varkonyi proved it and built the first one in 2006.
#
# The catch for a sculptor is that a TRUE gomboc is startlingly close
# to a sphere (Domokos & Varkonyi's first solution departs from it by
# under 5e-5 in radius).  This generator offers three ways to get a
# gomboc-shaped object:
#
#   * DOMOKOS-VARKONYI (default) -- the ACTUAL construction from their
#     2006 paper (transcribed in
#     research/papers/monostatic-bodies/monostatic_bodies_2006/).  In
#     spherical coordinates (longitude theta, latitude phi in
#     (-pi/2, pi/2)) the boundary is the radial surface
#         R(theta, phi) = 1 + d * dR(theta, phi, c),
#     a small deformation of the unit sphere by the field dR built from
#     a latitude-warping map
#         f(phi, c) = pi * ( (e^{phi/(pi c) + 1/(2c)} - 1)/(e^{1/c} - 1)
#                            - 1/2 ),
#     two meridian profiles f1 = sin f(phi, c),  f2 = -sin f(-phi, c),
#     and a longitude blend weight
#         a = cos^2(theta) cos^2 f(phi, c)
#             / ( cos^2(theta) cos^2 f(phi, c)
#                 + sin^2(theta) cos^2 f(-phi, c) ),
#         dR = a f1 + (1 - a) f2      (dR = +-1 at the poles).
#     d sets the amplitude of the deviation from the sphere; c shapes
#     the "tennis-ball seam" separatrix dR = 0.  The genuine
#     mono-monostatic body needs d < 5e-5 with c ~ 0.275 -- visually a
#     ball -- so, exactly as in the paper's own Figures 3-4, we default
#     to an EXAGGERATED d that makes the class {1,1} shape visible.
#     (Two known typos in the printed equations are corrected here: the
#     physical radius is 1 + d*dR, not (1+d)*dR, and the blend
#     numerator carries cos^2 f(-phi, c).)
#
#   * SCULPTURAL -- the familiar high-domed "turtle shell" LIKENESS:
#     a rounded belly sweeping up to a leaning dorsal ridge, built by
#     morphing the signed-distance field of a sphere toward a thin tall
#     blade (after the libfive sketch by M. Keeter / E. Liberato) and
#     meshed with the project's marching-tetrahedra kernel.  A likeness
#     of the physical object, not a certified balancing body.
#
#   * ANALYTIC (Sloan I / II) -- Robert J. Sloan's 2023 closed-form
#     radial surfaces, exaggerated by a flatness parameter beta:
#         Sloan 1:  r^4 = 1 + 4 beta sin(phi) cos(theta - 5 phi)
#         Sloan 2:  r^4 = 1 + 4 beta sin(phi)
#                          cos( theta - (3pi/2)(cos phi - cos^3 phi/3) )
#
# References:
#   - G. Domokos and P. Varkonyi, "Mono-monostatic bodies: the answer
#     to Arnold's question", The Mathematical Intelligencer 28 (2006),
#     no. 4, 34-38.  (Local copy + Markdown transcription under
#     research/papers/monostatic-bodies/.)
#   - P. L. Varkonyi and G. Domokos, "Static equilibria of rigid bodies:
#     dice, pebbles and the Poincare-Hopf theorem", J. Nonlinear Sci.
#     16 (2006), 255-281.
#   - Robert J. Sloan, "An analytic parameterization of the gomboc"
#     (2023); see also Wolfram MathWorld, "Gomboc",
#     https://mathworld.wolfram.com/Gomboc.html .
#   - libfive gomboc sketch: E. Liberato (2020),
#     eddieliberato.github.io/blog/2020-08-12-gomboc/ (after M. Keeter's
#     libfive f-rep modeller).

bl_info = {
    "name": "Gomboc",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Gomboc",
    "description": "Gomboc: the self-righting mono-monostatic solid "
                   "(after Domokos & Varkonyi; analytic form after Sloan)",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi

import numpy as np


# ---------------------------------------------------------------------
# SCULPTURAL gomboc: sphere-to-blade SDF morph, marched to a mesh
# ---------------------------------------------------------------------
def _box_sdf(X, Y, Z, c, h):
    qx = np.abs(X - c[0]) - h[0]
    qy = np.abs(Y - c[1]) - h[1]
    qz = np.abs(Z - c[2]) - h[2]
    ox, oy, oz = np.maximum(qx, 0.0), np.maximum(qy, 0.0), np.maximum(qz, 0.0)
    outside = np.sqrt(ox * ox + oy * oy + oz * oz)
    inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
    return outside + inside


def _sculptural_field(ridge, lean):
    """SDF whose zero set is the gomboc likeness.  `ridge` (0..~0.5) is
    the sphere->blade morph weight (higher = sharper dorsal crest);
    `lean` shears the crest sideways as it rises, breaking symmetry."""
    def f(X, Y, Z):
        Ys = Y - lean * np.maximum(Z, 0.0)         # lean the ridge over
        sphere = np.sqrt(X * X + Ys * Ys + Z * Z) - 1.0
        blade = _box_sdf(X, Ys, Z, (0.0, 0.0, 0.65), (0.02, 0.8, 0.65))
        return (1.0 - ridge) * sphere + ridge * blade
    return f


def build_gomboc_sculptural(ridge=0.35, lean=0.5, resolution=76):
    """Recognisable gomboc likeness, meshed by marching tetrahedra."""
    try:
        from . import minimal_surface_toolkit as mst
    except ImportError:
        import minimal_surface_toolkit as mst
    bmin = (-1.55, -2.0, -1.5)
    bmax = (1.55, 1.6, 1.7)
    cell = 3.2 / max(24, resolution)
    res = tuple(max(8, int(round((bmax[i] - bmin[i]) / cell)))
                for i in range(3))
    V, F = mst.marching_tets(_sculptural_field(ridge, lean), bmin, bmax, res)
    return V, F


# ---------------------------------------------------------------------
# mesh helper (welds the two spherical poles) -- shared by the radial
# Domokos-Varkonyi and Sloan builders
# ---------------------------------------------------------------------
def _vkey(p):
    return (round(p[0], 7), round(p[1], 7), round(p[2], 7))


class _Mesh:
    """Vertex-welding mesh builder (welds the two poles)."""

    def __init__(self):
        self.verts = []
        self.faces = []
        self._ids = {}

    def vid(self, p):
        k = _vkey(p)
        i = self._ids.get(k)
        if i is None:
            i = len(self.verts)
            self._ids[k] = i
            self.verts.append(p)
        return i

    def quad(self, a, b, c, d):
        ids = []
        for p in (a, b, c, d):
            i = self.vid(p)
            if i not in ids:
                ids.append(i)
        if len(ids) >= 3:
            self.faces.append(ids)


# ---------------------------------------------------------------------
# DOMOKOS-VARKONYI gomboc: the actual 2006 two-parameter construction
# ---------------------------------------------------------------------
def _dv_warp(phi, c):
    """Latitude-warping map f(phi, c): (-pi/2, pi/2) -> (-pi/2, pi/2),
    near-identity for c >> 1, strongly nonlinear as c -> 0."""
    return pi * ((math.exp(phi / (pi * c) + 1.0 / (2.0 * c)) - 1.0)
                 / (math.exp(1.0 / c) - 1.0) - 0.5)


def _dv_delta(theta, phi, c):
    """Deviation field dR(theta, phi, c) in [-1, 1]; dR = 0 is the
    separatrix (tennis-ball seam)."""
    if phi >= 0.5 * pi - 1e-9:
        return 1.0
    if phi <= -0.5 * pi + 1e-9:
        return -1.0
    fp = _dv_warp(phi, c)
    fm = _dv_warp(-phi, c)
    f1 = sin(fp)
    f2 = -sin(fm)
    num = cos(theta) ** 2 * cos(fp) ** 2
    den = num + sin(theta) ** 2 * cos(fm) ** 2
    a = (num / den) if den > 1e-12 else 0.5
    return a * f1 + (1.0 - a) * f2


def build_gomboc_dv(c=0.275, d=0.5, phi_segments=128, theta_segments=176):
    """Watertight Domokos-Varkonyi radial gomboc R = 1 + d*dR about its
    centre.  d is the (here exaggerated) deviation from the unit sphere;
    c shapes the separatrix."""
    m = _Mesh()
    nphi = max(8, phi_segments)
    nth = max(8, theta_segments)

    def pt(i, j):
        phi = -0.5 * pi + pi * i / nphi
        theta = 2.0 * pi * (j % nth) / nth
        r = 1.0 + d * _dv_delta(theta, phi, c)
        return (r * cos(phi) * cos(theta),
                r * cos(phi) * sin(theta),
                r * sin(phi))

    grid = [[pt(i, j) for j in range(nth)] for i in range(nphi + 1)]
    for i in range(nphi):
        for j in range(nth):
            jn = (j + 1) % nth
            m.quad(grid[i][j], grid[i][jn],
                   grid[i + 1][jn], grid[i + 1][j])
    return m.verts, m.faces


# ---------------------------------------------------------------------
# ANALYTIC gomboc: Sloan's closed-form radial surfaces
# ---------------------------------------------------------------------
def _sloan_radius(kind, phi, theta, beta):
    s = sin(phi)
    if kind == 'SLOAN_II':
        c = cos(phi)
        ph = theta - (1.5 * pi) * (c - c * c * c / 3.0)
    else:                                    # 'SLOAN_I'
        ph = theta - 5.0 * phi
    r4 = 1.0 + 4.0 * beta * s * cos(ph)
    if r4 < 1e-9:
        r4 = 1e-9
    return r4 ** 0.25


def build_gomboc_analytic(kind='SLOAN_II', beta=0.17, phi_segments=96,
                          theta_segments=128):
    """Watertight radial Sloan gomboc surface about its centre."""
    m = _Mesh()
    nphi = max(6, phi_segments)
    nth = max(8, theta_segments)

    def pt(i, j):
        phi = pi * i / nphi
        theta = 2.0 * pi * (j % nth) / nth
        r = _sloan_radius(kind, phi, theta, beta)
        return (r * sin(phi) * cos(theta),
                r * sin(phi) * sin(theta),
                r * cos(phi))

    grid = [[pt(i, j) for j in range(nth)] for i in range(nphi + 1)]
    for i in range(nphi):
        for j in range(nth):
            jn = (j + 1) % nth
            m.quad(grid[i][j], grid[i][jn],
                   grid[i + 1][jn], grid[i + 1][j])
    return m.verts, m.faces


try:
    import bpy
    import bmesh
    from bpy.props import IntProperty, FloatProperty, EnumProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_gomboc_add(bpy.types.Operator):
        """Add a gomboc -- the convex, homogeneous self-righting solid """ \
            """with a single stable and single unstable balance point"""
        bl_idname = "mesh.gomboc_add"
        bl_label = "Gomboc"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Form",
            items=[('DV', "Domokos-Varkonyi (paper)",
                    "The actual 2006 construction: R = 1 + d*dR, a "
                    "deformed sphere with a tennis-ball-seam "
                    "separatrix (d exaggerated for visibility)"),
                   ('SCULPTURAL', "Sculptural (turtle-shell likeness)",
                    "The familiar gomboc silhouette: rounded belly "
                    "sweeping up to a leaning dorsal ridge (a likeness, "
                    "meshed from an SDF morph)"),
                   ('SLOAN_II', "Analytic - Sloan II",
                    "Sloan's closed-form radial surface II "
                    "(beta up to ~0.17)"),
                   ('SLOAN_I', "Analytic - Sloan I",
                    "Sloan's closed-form radial surface I; a helically "
                    "swept ridge (beta up to ~0.15)")],
            default='DV')
        dv_c: FloatProperty(
            name="Separatrix c", default=0.275, min=0.1, max=3.0,
            description="Domokos-Varkonyi: shapes the tennis-ball-seam "
                        "separatrix (~0.275 is the paper's value)")
        dv_d: FloatProperty(
            name="Deviation d", default=0.5, min=0.0, max=0.95,
            description="Domokos-Varkonyi: deviation from the sphere. "
                        "The true body needs d < 5e-5 (a ball); larger "
                        "d exaggerates the shape, as in the paper's "
                        "figures")
        ridge: FloatProperty(
            name="Ridge", default=0.35, min=0.05, max=0.6,
            description="Sculptural: sharpness of the dorsal crest "
                        "(sphere-to-blade morph weight)")
        lean: FloatProperty(
            name="Lean", default=0.5, min=0.0, max=1.0,
            description="Sculptural: how far the crest leans over one "
                        "end (breaks the symmetry)")
        resolution: IntProperty(
            name="Resolution", default=76, min=24, max=200,
            description="Sculptural: marching-grid density")
        beta: FloatProperty(
            name="Flatness beta", default=0.17, min=0.0, max=0.25,
            description="Analytic: departure from a sphere (0 = ball). "
                        "Keep <= ~0.15 (Sloan I) / ~0.17 (Sloan II)")
        phi_segments: IntProperty(
            name="Rings", default=96, min=8, max=400,
            description="Analytic: segments pole to pole")
        theta_segments: IntProperty(
            name="Segments", default=128, min=8, max=512,
            description="Analytic: segments around the axis")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            if self.kind == 'DV':
                verts, faces = build_gomboc_dv(
                    self.dv_c, self.dv_d, self.phi_segments,
                    self.theta_segments)
            elif self.kind == 'SCULPTURAL':
                verts, faces = build_gomboc_sculptural(
                    self.ridge, self.lean, self.resolution)
            else:
                verts, faces = build_gomboc_analytic(
                    self.kind, self.beta, self.phi_segments,
                    self.theta_segments)

            verts = [tuple(map(float, v)) for v in np.asarray(verts)]
            faces = [tuple(int(i) for i in f) for f in faces]

            def _fit(vs):
                # centre on the bbox midpoint and scale the largest
                # extent to 2.0 * scale (a ~2 m cube); scale divides out
                # of the ratio, so it is not applied twice.
                if not vs:
                    return vs
                xs = [v[0] for v in vs]
                ys = [v[1] for v in vs]
                zs = [v[2] for v in vs]
                cx = 0.5 * (min(xs) + max(xs))
                cy = 0.5 * (min(ys) + max(ys))
                cz = 0.5 * (min(zs) + max(zs))
                ext = max(max(xs) - min(xs), max(ys) - min(ys),
                          max(zs) - min(zs))
                s = (2.0 * self.scale / ext) if ext > 1e-9 else 1.0
                return [((v[0] - cx) * s, (v[1] - cy) * s,
                         (v[2] - cz) * s) for v in vs]

            me = bpy.data.meshes.new("Gomboc")
            me.from_pydata(_fit(verts), [], faces)
            me.validate(clean_customdata=True)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
            me.polygons.foreach_set('use_smooth',
                                    [True] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new("Gomboc", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            if self.kind == 'DV':
                lay.prop(self, 'dv_c')
                lay.prop(self, 'dv_d')
                lay.prop(self, 'phi_segments')
                lay.prop(self, 'theta_segments')
            elif self.kind == 'SCULPTURAL':
                lay.prop(self, 'ridge')
                lay.prop(self, 'lean')
                lay.prop(self, 'resolution')
            else:
                lay.prop(self, 'beta')
                lay.prop(self, 'phi_segments')
                lay.prop(self, 'theta_segments')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.gomboc_add", icon='MESH_UVSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_gomboc_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_gomboc_add)


def _selftest():
    from collections import Counter

    def _edge_manifold(f):
        cnt = Counter()
        for fc in f:
            n = len(fc)
            for i in range(n):
                a, b = fc[i], fc[(i + 1) % n]
                cnt[(min(a, b), max(a, b))] += 1
        return all(c == 2 for c in cnt.values()), len(cnt)

    # Domokos-Varkonyi: watertight ball (chi = 2), positive radius, and
    # the deformation actually breaks the sphere (R varies with d).
    v, f = build_gomboc_dv(0.275, 0.5, 80, 112)
    man, ne = _edge_manifold(f)
    chi = len(v) - ne + len(f)
    rr = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in v]
    ok = man and chi == 2 and min(rr) > 0.05 and (max(rr) - min(rr)) > 0.3
    print(f"gomboc[DV c=.275 d=.5]: V={len(v)} F={len(f)} manifold={man} "
          f"chi={chi}(2) R=[{min(rr):.3f},{max(rr):.3f}] "
          f"{'OK' if ok else 'BAD'}")
    assert ok
    # d -> 0 relaxes to the unit sphere (all radii ~ 1)
    v0, _ = build_gomboc_dv(0.275, 0.0, 24, 32)
    rr0 = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in v0]
    assert max(abs(r - 1.0) for r in rr0) < 1e-9, "d=0 must be a sphere"

    # sculptural: closed manifold, convex-ish, taller than wide with a
    # single upper crest (top extent above centre exceeds the belly).
    v, f = build_gomboc_sculptural(0.35, 0.5, 56)
    v = [tuple(map(float, p)) for p in np.asarray(v)]
    f = [tuple(int(i) for i in t) for t in f]
    cnt = Counter()
    for fc in f:
        n = len(fc)
        for i in range(n):
            a, b = fc[i], fc[(i + 1) % n]
            cnt[(min(a, b), max(a, b))] += 1
    manifold = all(c == 2 for c in cnt.values())
    zs = [p[2] for p in v]
    ok = manifold and len(v) > 100 and (max(zs) - min(zs)) > 0.5
    print(f"gomboc[sculptural]: V={len(v)} F={len(f)} manifold={manifold} "
          f"z-extent={max(zs) - min(zs):.2f} {'OK' if ok else 'BAD'}")
    assert ok

    # analytic Sloan forms: watertight ball (chi = 2), radius positive.
    for kind, beta in (('SLOAN_I', 0.15), ('SLOAN_II', 0.17)):
        v, f = build_gomboc_analytic(kind, beta, 48, 64)
        cnt = Counter()
        for fc in f:
            n = len(fc)
            for i in range(n):
                a, b = fc[i], fc[(i + 1) % n]
                cnt[(min(a, b), max(a, b))] += 1
        manifold = all(c == 2 for c in cnt.values())
        chi = len(v) - len(cnt) + len(f)
        rmin = min(math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in v)
        ok = manifold and chi == 2 and rmin > 0.1
        print(f"gomboc[{kind}]: V={len(v)} F={len(f)} manifold={manifold} "
              f"chi={chi}(2) rmin={rmin:.3f} {'OK' if ok else 'BAD'}")
        assert ok
