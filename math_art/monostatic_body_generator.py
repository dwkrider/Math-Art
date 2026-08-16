
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
#   * GÖMBÖC (default; internal id FABRICATED) -- the recognisable
#     fabricated gomboc as a
#     SMOOTH, STRICTLY CONVEX body.  The makers describe the object as
#     a "tennis-ball" assembly of segments of simple surfaces
#     (cylinder, ellipsoid, cone) and planes, joined along crease
#     edges -- it has NO single closed-form equation.  Here its
#     digitised SUPPORT function h(u) = max_x x.u (spherical-harmonic
#     coefficients in _monostatic_data.py) is smoothed by the spherical
#     heat kernel, i.e. the coefficients are damped by
#     exp(-l(l+1) sigma^2/2).  Convolving a support function with a
#     nonnegative zonal kernel is a rotational Minkowski average of the
#     body, so the result is PROVABLY again the support function of a
#     convex body -- the crease edges come out rounded at radius ~sigma
#     with no ringing, faceting or pole pinch.  The surface is meshed
#     by the normal parametrisation x(u) = grad H(u) of the
#     1-homogeneous extension H(p) = |p| h(p/|p|), evaluated on an
#     icosphere of directions (this naturally concentrates vertices where the
#     curvature is high, e.g. along the seam and beak).
#     CAVEAT on the physics: this form reproduces the SHAPE (smooth,
#     strictly convex, one mirror plane, ~40% off a sphere), but it is
#     NOT a certified mono-monostatic solid.  Resting energy is
#     V(e) = c.e + h(-e) (c = centre of mass); equilibria are its
#     critical points, and a true gomboc must have exactly one min and
#     one max.  Digitising h from the (low-poly) reference leaves a
#     residual roughness of ~1e-3 * h -- the SAME order as the real
#     gomboc's ~1e-3 tolerance -- so the energy landscape carries extra
#     shallow equilibria and the exact 1+1 count cannot be certified
#     here.  For a PROVABLY mono-monostatic body use the Sloan or
#     Domokos-Varkonyi forms below (which look near-spherical -- that
#     tension is the whole point of the gomboc).
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
#   - Support functions, Minkowski combinations and their convolution
#     by nonnegative zonal kernels: R. Schneider, "Convex Bodies: The
#     Brunn-Minkowski Theory", 2nd ed., Cambridge Univ. Press, 2014
#     (ch. 1.7-1.8; the normal parametrisation x = h u + grad_S h is
#     the classical "reverse spherical image").

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
from math import cos, sin, pi, sqrt

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


try:
    from .surfaces.primitives import icosphere as _icosphere_shared
except ImportError:  # flat import outside the package
    from surfaces.primitives import icosphere as _icosphere_shared


def _icosphere(level=0):
    """Unit icosphere: the icosahedron subdivided `level` times.
    The shared builder is in `surfaces.primitives`.
    """
    return _icosphere_shared(level, 'per_level')


# ---------------------------------------------------------------------
# FABRICATED gomboc: smooth strictly convex body from the heat-damped
# spherical-harmonic expansion of the digitised SUPPORT function.
# Heat flow of a support function = rotational Minkowski average of the
# body (nonnegative zonal kernel), hence again a support function of a
# convex body: the crease edges round off at radius ~sigma with no
# ringing or faceting.  Surface meshed by the classical normal
# parametrisation x(u) = grad H(u), H(p) = |p| h(p/|p|).
# ---------------------------------------------------------------------
def _sh_gomboc_coeffs():
    try:
        from .polyhedra import _monostatic_data as md
    except ImportError:
        from polyhedra import _monostatic_data as md
    L = md.GOMBOC_SH_L
    c = np.zeros((L + 1) * (L + 1))
    c[np.asarray(md.GOMBOC_SH_IDX, dtype=int)] = md.GOMBOC_SH_COEF
    return L, c


def _sh_support(dirs, sigma):
    """Evaluate the heat-damped support function h_sigma at unit
    directions `dirs` (n, 3).  Fully normalised real spherical
    harmonics via the stable Holmes-Featherstone recurrences; memory
    stays O(n) (no basis matrix is materialised)."""
    L, c = _sh_gomboc_coeffs()
    ll = np.arange(L + 1)
    damp = np.exp(-ll * (ll + 1) * sigma * sigma / 2.0)
    # degrees the damping has not annihilated
    lcut = int(max(np.nonzero(damp > 1e-9)[0], default=0)) \
        if damp[-1] <= 1e-9 else L
    lcut = max(2, lcut)
    # highest degree with a surviving coefficient, per order m
    lmax_m = np.full(L + 1, -1, dtype=int)
    for i in np.nonzero(c)[0]:
        l = int(sqrt(i))
        m = abs(i - l * l - l)
        if l > lmax_m[m]:
            lmax_m[m] = l
    ct = np.clip(dirs[:, 2], -1.0, 1.0)
    st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    out = np.zeros(len(dirs))
    r2 = sqrt(2.0)
    pmm = np.full(len(dirs), sqrt(1.0 / (4.0 * pi)))    # P_00
    for m in range(L + 1):
        if m > 0:
            pmm = pmm * st * sqrt((2.0 * m + 1.0) / (2.0 * m))
        ltop = min(lmax_m[m], lcut)
        if ltop < m:
            continue
        cosm = np.cos(m * phi) if m else None
        sinm = np.sin(m * phi) if m else None
        plm2 = None
        plm1 = None
        for l in range(m, ltop + 1):
            if l == m:
                p = pmm
            elif l == m + 1:
                p = ct * sqrt(2.0 * m + 3.0) * pmm
            else:
                a = sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
                b = sqrt(((2.0 * l + 1.0) * (l + m - 1.0) * (l - m - 1.0))
                         / ((2.0 * l - 3.0) * (l * l - m * m)))
                p = a * ct * plm1 - b * plm2
            base = l * l + l
            if m == 0:
                cc = c[base] * damp[l]
                if cc:
                    out += cc * p
            else:
                cc = c[base + m] * damp[l]
                cs = c[base - m] * damp[l]
                if cc:
                    out += (r2 * cc) * (p * cosm)
                if cs:
                    out += (r2 * cs) * (p * sinm)
            plm2, plm1 = plm1, p
    return out


def build_gomboc_fabricated(sigma=0.09, level=6):
    """The fabricated gomboc as a C-infinity strictly convex surface:
    heat-kernel-rounded support function, meshed on an icosphere of
    normal directions via x(u) = grad H(u) (central differences of the
    1-homogeneous extension H)."""
    verts, faces = _icosphere(max(2, min(7, int(level))))
    U = np.asarray(verts)
    d = 1e-4

    def hom(P):
        r = np.sqrt((P * P).sum(axis=1))
        return r * _sh_support(P / r[:, None], sigma)

    X = np.empty_like(U)
    for k in range(3):
        e = np.zeros(3)
        e[k] = d
        X[:, k] = (hom(U + e) - hom(U - e)) / (2.0 * d)
    return ([tuple(map(float, p)) for p in X],
            [list(f) for f in faces])


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
    # here phi is a LATITUDE (-pi/2..pi/2): rows i = 0..nphi run pole to
    # pole and the poles weld automatically.
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
    from .sharp_creases import mark_sharp_by_angle
except ImportError:                     # flat import outside the package
    from sharp_creases import mark_sharp_by_angle

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty,
                           EnumProperty, BoolProperty)
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
            items=[('FABRICATED', "Gömböc",
                    "The iconic fabricated gomboc as a smooth, "
                    "strictly convex surface: spherical heat-kernel "
                    "(Minkowski) rounding of the digitised support "
                    "function -- rounded belly, seam edges and beak "
                    "with no faceting"),
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
            default='FABRICATED')
        smooth_sigma: FloatProperty(
            name="Edge Rounding", default=0.09, min=0.07, max=0.25,
            description="Fabricated: heat-kernel rounding radius "
                        "(radians on the sphere of normals); smaller "
                        "keeps the tennis-ball seam edges crisper "
                        "(below ~0.07 the digitisation noise returns)")
        smooth_level: IntProperty(
            name="Subdivisions", default=6, min=3, max=7,
            description="Fabricated: icosphere subdivisions of the "
                        "normal sphere (6 = 40962 vertices)")
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
        sharp_edges: BoolProperty(
            name="Sharp Edges", default=True,
            description="Mark the solid's fold curves sharp (and "
                        "creased). The Gomboc and its relatives carry a ridge where the two monostatic lobes meet. The surface is smooth "
                        "everywhere else, so shading straight across "
                        "the fold rounds off the one feature that "
                        "defines the shape")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            if self.kind == 'FABRICATED':
                verts, faces = build_gomboc_fabricated(
                    self.smooth_sigma, self.smooth_level)
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
            if self.sharp_edges:
                mark_sharp_by_angle(me, 30.0)
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
            if self.kind == 'FABRICATED':
                lay.prop(self, 'smooth_sigma')
                lay.prop(self, 'smooth_level')
            elif self.kind == 'DV':
                lay.prop(self, 'dv_c')
                lay.prop(self, 'dv_d')
            elif self.kind in ('SLOAN_I', 'SLOAN_II'):
                lay.prop(self, 'beta')
            if self.kind != 'FABRICATED':
                # the Gomboc form meshes on its own normal-sphere
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
    # Fabricated (smooth convex) gomboc: watertight ball (chi = 2),
    # mirror-symmetric about x = 0, clearly non-spherical, and CONVEX
    # (every mesh point lies inside the support-function bound).
    v, f = build_gomboc_fabricated(sigma=0.09, level=4)
    man, ne = _edge_manifold(f)
    chi = len(v) - ne + len(f)
    X = np.asarray(v)
    rr = np.sqrt((X * X).sum(axis=1))
    dev = (rr.max() - rr.min()) / rr.mean()
    # mirror symmetry of the underlying support function
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(256, 3))
    dirs /= np.sqrt((dirs * dirs).sum(axis=1))[:, None]
    hd = _sh_support(dirs, 0.09)
    hm = _sh_support(dirs * np.array([-1.0, 1.0, 1.0]), 0.09)
    asym = np.abs(hd - hm).max()
    # convexity certificate: max_j X_j . u_i <= h(u_i) (+ tolerance)
    U = X / rr[:, None]
    hu = _sh_support(U, 0.09)
    worst = -1e9
    for i0 in range(0, len(U), 512):
        s = (X @ U[i0:i0 + 512].T).max(axis=0) - hu[i0:i0 + 512]
        worst = max(worst, float(s.max()))
    ok = (man and chi == 2 and rr.min() > 0.1 and dev > 0.2
          and asym < 1e-6 and worst < 1e-4)
    print(f"gomboc[FABRICATED]: V={len(v)} F={len(f)} manifold={man} "
          f"chi={chi}(2) dev={dev:.2f} mirror_asym={asym:.1e} "
          f"convexity_excess={worst:.1e} {'OK' if ok else 'BAD'}")
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
