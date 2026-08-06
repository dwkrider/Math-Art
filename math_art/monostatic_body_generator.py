
# Monostatic Body ("Gomboc") Generator for Blender
#
# A gomboc is a convex, homogeneous solid that is "mono-monostatic":
# it has exactly ONE stable and ONE unstable balance point, so -- like
# a self-righting toy, but with no added weight and no hollow -- it
# always rolls back to the same resting pose.  Vladimir Arnold
# conjectured such a body could exist; Gabor Domokos and Peter
# Varkonyi proved it and built the first one in 2006.
#
# There is an important gap between the MATHEMATICS and the ICONIC
# OBJECT, so this generator offers four forms:
#
#   * REFERENCE (default) -- the recognisable FABRICATED gomboc: a
#     rounded body with a smooth belly rising to a single curved ridge
#     and beak.  The makers describe it as a "tennis-ball" assembly of
#     segments of simple surfaces (cylinder, ellipsoid, cone) and
#     planes -- it has NO single closed-form equation.  To reproduce it
#     faithfully, its convex star-shaped boundary was digitised from a
#     reference model as a radial grid R(phi, theta) (see
#     _monostatic_data.py) and bilinearly interpolated.  This is the
#     shape people picture; it departs from a sphere by ~40%.  (A
#     spherical-harmonic fit was tried first but rings near the beak.)
#
#   * DOMOKOS-VARKONYI -- the actual construction from their 2006
#     existence proof (transcribed in
#     research/papers/monostatic-bodies/monostatic_bodies_2006/).  In
#     spherical coordinates the boundary is R = 1 + d*dR, a small
#     deformation of the unit sphere with a "tennis-ball seam"
#     separatrix (dR built from a latitude-warp map -- see _dv_delta).
#     The genuine class {1,1} body needs d < 5e-5 (visually a ball), so
#     -- as in the paper's own figures -- d defaults to an exaggerated
#     value to make the shape visible.
#
#   * SLOAN I / SLOAN II -- M. L. Sloan's two closed-form ANALYTIC
#     gombocs (infinitely differentiable), r^4 = 1 + 4 beta sin(theta)
#     cos(phi - P(theta)) with theta the polar angle and phi the
#     azimuth:
#         Sloan 1:  P = 5 theta                              (beta<=0.15)
#         Sloan 2:  P = (3pi/2)(cos theta - cos^3 theta / 3) (beta<=0.17)
#     These provably have exactly two equilibria, but (footnote 2 of
#     Sloan) are only small perturbations of a sphere -- they do NOT
#     look like the fabricated object.
#
# References:
#   - G. Domokos and P. Varkonyi, "Mono-monostatic bodies: the answer
#     to Arnold's question", The Mathematical Intelligencer 28 (2006),
#     no. 4, 34-38.  (Local copy + Markdown under
#     research/papers/monostatic-bodies/.)
#   - M. L. Sloan, "An Analytical Gomboc" (technical note, with the
#     guidance of G. Domokos); see also Wolfram MathWorld, "Gomboc",
#     https://mathworld.wolfram.com/Gomboc.html .
#   - The fabricated gomboc and its tennis-ball segment construction:
#     G. Domokos & P. Varkonyi, gomboc.eu.

bl_info = {
    "name": "Monostatic Body",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Monostatic Body",
    "description": "Monostatic body: the self-righting gomboc -- the "
                   "fabricated shape plus the Domokos-Varkonyi and Sloan "
                   "analytic constructions",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi

import numpy as np


# ---------------------------------------------------------------------
# mesh helper (welds coincident vertices, e.g. the two spherical poles)
# ---------------------------------------------------------------------
def _vkey(p):
    return (round(p[0], 7), round(p[1], 7), round(p[2], 7))


class _Mesh:
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


def _radial_mesh(radius_grid):
    """Build a watertight radial mesh from R sampled on an
    (nphi+1) x nth grid (phi = colatitude 0..pi rows, theta wraps).
    Poles weld automatically."""
    nphi = radius_grid.shape[0] - 1
    nth = radius_grid.shape[1]
    m = _Mesh()
    ii = np.arange(nphi + 1)[:, None]
    jj = np.arange(nth)[None, :]
    phi = pi * ii / nphi
    th = 2.0 * pi * jj / nth
    R = radius_grid
    X = R * np.sin(phi) * np.cos(th)
    Y = R * np.sin(phi) * np.sin(th)
    Z = R * np.cos(phi) * np.ones((1, nth))
    for i in range(nphi):
        for j in range(nth):
            jn = (j + 1) % nth
            m.quad((X[i, j], Y[i, j], Z[i, j]),
                   (X[i, jn], Y[i, jn], Z[i, jn]),
                   (X[i + 1, jn], Y[i + 1, jn], Z[i + 1, jn]),
                   (X[i + 1, j], Y[i + 1, j], Z[i + 1, j]))
    return m.verts, m.faces


# ---------------------------------------------------------------------
# REFERENCE (fabricated) gomboc: digitised radial grid, bilinearly
# interpolated (a spherical-harmonic fit was tried first but rings)
# ---------------------------------------------------------------------
def _reference_grid():
    try:
        from . import _monostatic_data as md
    except ImportError:
        import _monostatic_data as md
    return (md.GOMBOC_GRID_NPHI, md.GOMBOC_GRID_NTH,
            np.asarray(md.GOMBOC_GRID, dtype=float).reshape(
                md.GOMBOC_GRID_NPHI, md.GOMBOC_GRID_NTH))


def build_gomboc_reference():
    """Mesh the digitised radial grid directly at its native resolution
    (grid rows are the pole-to-pole colatitude samples, columns the
    wrapping azimuth samples), so no interpolation crease is introduced."""
    _gph, _gth, G = _reference_grid()
    return _radial_mesh(G)


# ---------------------------------------------------------------------
# DOMOKOS-VARKONYI gomboc: the 2006 two-parameter construction
# ---------------------------------------------------------------------
def _dv_warp(phi, c):
    return pi * ((math.exp(phi / (pi * c) + 1.0 / (2.0 * c)) - 1.0)
                 / (math.exp(1.0 / c) - 1.0) - 0.5)


def _dv_delta(theta, phi, c):
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
    nphi = max(8, phi_segments)
    nth = max(8, theta_segments)
    R = np.empty((nphi + 1, nth))
    for i in range(nphi + 1):
        phi = -0.5 * pi + pi * i / nphi
        for j in range(nth):
            theta = 2.0 * pi * j / nth
            R[i, j] = 1.0 + d * _dv_delta(theta, phi, c)
    # note: here phi is a LATITUDE (-pi/2..pi/2); remap to colatitude
    # grid rows already run pole-to-pole, so _radial_mesh's colatitude
    # matches after flipping the row order is unnecessary -- rows i=0..n
    # correspond to latitude -pi/2..pi/2 i.e. colatitude pi..0.  Build
    # directly here to keep the latitude convention.
    m = _Mesh()

    def pt(i, j):
        phi = -0.5 * pi + pi * i / nphi
        theta = 2.0 * pi * (j % nth) / nth
        r = R[i, j]
        return (r * cos(phi) * cos(theta),
                r * cos(phi) * sin(theta),
                r * sin(phi))

    for i in range(nphi):
        for j in range(nth):
            jn = (j + 1) % nth
            m.quad(pt(i, j), pt(i, jn), pt(i + 1, jn), pt(i + 1, j))
    return m.verts, m.faces


# ---------------------------------------------------------------------
# SLOAN analytic gombocs: r^4 = 1 + 4 beta sin(theta) cos(phi - P(theta))
# (theta = polar angle, phi = azimuth, per Sloan's note)
# ---------------------------------------------------------------------
def _sloan_radius(kind, polar, azim, beta):
    s = sin(polar)
    if kind == 'SLOAN_II':
        c = cos(polar)
        p = azim - (1.5 * pi) * (c - c * c * c / 3.0)
    else:                                    # 'SLOAN_I'
        p = azim - 5.0 * polar
    r4 = 1.0 + 4.0 * beta * s * cos(p)
    if r4 < 1e-9:
        r4 = 1e-9
    return r4 ** 0.25


def build_gomboc_sloan(kind='SLOAN_I', beta=0.15, phi_segments=96,
                       theta_segments=128):
    m = _Mesh()
    nphi = max(6, phi_segments)
    nth = max(8, theta_segments)

    def pt(i, j):
        polar = pi * i / nphi
        azim = 2.0 * pi * (j % nth) / nth
        r = _sloan_radius(kind, polar, azim, beta)
        return (r * sin(polar) * cos(azim),
                r * sin(polar) * sin(azim),
                r * cos(polar))

    for i in range(nphi):
        for j in range(nth):
            jn = (j + 1) % nth
            m.quad(pt(i, j), pt(i, jn), pt(i + 1, jn), pt(i + 1, j))
    return m.verts, m.faces


try:
    import bpy
    import bmesh
    from bpy.props import IntProperty, FloatProperty, EnumProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_monostatic_body_add(bpy.types.Operator):
        """Add a monostatic body -- the gomboc, a convex homogeneous """ \
            """self-righting solid with one stable and one unstable """ \
            """balance point"""
        bl_idname = "mesh.monostatic_body_add"
        bl_label = "Monostatic Body"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Form",
            items=[('REFERENCE', "Fabricated Gomboc (recognisable)",
                    "The iconic fabricated shape: smooth belly rising "
                    "to a curved ridge and beak (spherical-harmonic "
                    "model of the real object)"),
                   ('DV', "Domokos-Varkonyi (2006)",
                    "The existence-proof construction R = 1 + d*dR "
                    "with a tennis-ball-seam separatrix (d exaggerated "
                    "for visibility)"),
                   ('SLOAN_I', "Analytic - Sloan I",
                    "Sloan's closed-form r^4 = 1 + 4 beta sin(t) "
                    "cos(p - 5t); a near-sphere (beta up to ~0.15)"),
                   ('SLOAN_II', "Analytic - Sloan II",
                    "Sloan's closed-form II; a near-sphere "
                    "(beta up to ~0.17)")],
            default='REFERENCE')
        dv_c: FloatProperty(
            name="Separatrix c", default=0.275, min=0.1, max=3.0,
            description="Domokos-Varkonyi: shapes the tennis-ball-seam "
                        "separatrix (~0.275 is the paper's value)")
        dv_d: FloatProperty(
            name="Deviation d", default=0.5, min=0.0, max=0.95,
            description="Domokos-Varkonyi: deviation from the sphere "
                        "(the true body needs d < 5e-5; larger d "
                        "exaggerates the shape, as in the paper)")
        beta: FloatProperty(
            name="Perturbation beta", default=0.15, min=0.0, max=0.25,
            description="Sloan: perturbation from the sphere. Keep "
                        "<= ~0.15 (Sloan I) / ~0.17 (Sloan II)")
        phi_segments: IntProperty(
            name="Rings", default=128, min=8, max=400,
            description="Segments from pole to pole")
        theta_segments: IntProperty(
            name="Segments", default=176, min=8, max=512,
            description="Segments around the axis")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            if self.kind == 'REFERENCE':
                verts, faces = build_gomboc_reference()
            elif self.kind == 'DV':
                verts, faces = build_gomboc_dv(
                    self.dv_c, self.dv_d, self.phi_segments,
                    self.theta_segments)
            else:
                verts, faces = build_gomboc_sloan(
                    self.kind, self.beta, self.phi_segments,
                    self.theta_segments)

            verts = [tuple(map(float, v)) for v in verts]

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

            me = bpy.data.meshes.new("Monostatic Body")
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
            obj = bpy.data.objects.new("Monostatic Body", me)
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
            elif self.kind in ('SLOAN_I', 'SLOAN_II'):
                lay.prop(self, 'beta')
            if self.kind != 'REFERENCE':      # reference uses its grid res
                lay.prop(self, 'phi_segments')
                lay.prop(self, 'theta_segments')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.monostatic_body_add",
                             icon='MESH_UVSPHERE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_monostatic_body_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_monostatic_body_add)


def _edge_manifold(f):
    from collections import Counter
    cnt = Counter()
    for fc in f:
        n = len(fc)
        for i in range(n):
            a, b = fc[i], fc[(i + 1) % n]
            cnt[(min(a, b), max(a, b))] += 1
    return all(c == 2 for c in cnt.values()), len(cnt)


def _selftest():
    # Reference (SH) gomboc: watertight ball (chi = 2), clearly
    # non-spherical (the fabricated shape deviates ~40% from a sphere).
    v, f = build_gomboc_reference()
    man, ne = _edge_manifold(f)
    chi = len(v) - ne + len(f)
    rr = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in v]
    dev = (max(rr) - min(rr)) / (sum(rr) / len(rr))
    ok = man and chi == 2 and min(rr) > 0.1 and dev > 0.2
    print(f"gomboc[REFERENCE]: V={len(v)} F={len(f)} manifold={man} "
          f"chi={chi}(2) R=[{min(rr):.3f},{max(rr):.3f}] dev={dev:.2f} "
          f"{'OK' if ok else 'BAD'}")
    assert ok

    # Domokos-Varkonyi: watertight, positive radius, d -> 0 is a sphere.
    v, f = build_gomboc_dv(0.275, 0.5, 60, 96)
    man, ne = _edge_manifold(f)
    chi = len(v) - ne + len(f)
    rr = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in v]
    ok = man and chi == 2 and min(rr) > 0.05
    print(f"gomboc[DV]: V={len(v)} F={len(f)} manifold={man} chi={chi}(2) "
          f"R=[{min(rr):.3f},{max(rr):.3f}] {'OK' if ok else 'BAD'}")
    assert ok
    v0, _ = build_gomboc_dv(0.275, 0.0, 24, 32)
    rr0 = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in v0]
    assert max(abs(r - 1.0) for r in rr0) < 1e-9, "d=0 must be a sphere"

    # Sloan analytic forms: watertight ball, near-sphere.
    for kind, beta in (('SLOAN_I', 0.15), ('SLOAN_II', 0.17)):
        v, f = build_gomboc_sloan(kind, beta, 48, 64)
        man, ne = _edge_manifold(f)
        chi = len(v) - ne + len(f)
        rmin = min(math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in v)
        ok = man and chi == 2 and rmin > 0.1
        print(f"gomboc[{kind}]: V={len(v)} F={len(f)} manifold={man} "
              f"chi={chi}(2) rmin={rmin:.3f} {'OK' if ok else 'BAD'}")
        assert ok
