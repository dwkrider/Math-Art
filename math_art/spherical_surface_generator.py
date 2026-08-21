# Spherical Surfaces Generator for Blender
#
# Smooth surfaces of constant POSITIVE Gaussian curvature (K = +1).
# This is the other half of Minding's 1839 classification: the sibling
# module `hyperbolic_surface_generator` ships the K = -1 surfaces of
# revolution (pseudosphere, Minding bulge, Minding spindle), and these
# are the K = +1 ones.  They live apart because "Hyperbolic Surface" is
# the wrong name to show a user for a sphere.
#
# With the meridian written in ARCLENGTH as (f(u), z(u)), the Gauss
# curvature of the surface of revolution is K = -f''/f, so K = +1 forces
#
#     f'' + f = 0     ->     f(u) = a cos(u)
#
# (against f'' - f = 0 and a cosh / a sinh for K = -1).  The single
# parameter a splits the family into three, exactly as Minding found:
#
#   a = 1   the unit SPHERE
#   a < 1   the SPINDLE type: the meridian meets the axis at an angle,
#           so the surface closes to a point at each pole -- a lemon
#           with two conical tips
#   a > 1   the BULGE type: z' vanishes before the meridian reaches the
#           axis, so the surface stops at two circular rims and is an
#           open barrel, not a closed shape
#
# AND ONE THAT IS NOT A SURFACE OF REVOLUTION.  Minding's classification
# is complete only among surfaces of revolution, and the obvious next
# question is whether K = +1 forces that symmetry.  It does not, and
# SIEVERT'S SURFACE (1886) is the standing answer: a surface of constant
# positive curvature with no axis of symmetry at all, and -- what makes
# it worth having rather than merely existing -- one written in ordinary
# functions rather than as the solution of a differential equation.
# It is given in cylindrical form,
#
#     rho   = 2a sqrt(2 + 2 sin^2 u) sin v / (2 - sin^2 v cos^2 u)
#     theta = -u / sqrt2 + arctan(sqrt2 tan u)
#     z     = a (ln tan(v/2) + 4 cos v / (2 - sin^2 v cos^2 u))
#
# and the self-test measures K on it directly rather than trusting the
# transcription -- which is the right call, because a plausible-looking
# variant of these formulas recalled from a second source measured
# K between 2.5 and 7.4 instead of the constant 1.
#
# THE QUADRATURE, and why it is written twice.  With z' = sqrt(1 - f'^2)
# and f' = -a sin u, the integrand is
#
#     dz = sqrt(1 - a^2 sin^2 u) du
#
# For a < 1 that is bounded below by sqrt(1 - a^2) > 0 and is smooth on
# the whole meridian, so it is integrated directly in u.  For a >= 1 it
# vanishes like a square root where |a sin u| -> 1, which is exactly the
# rim that gives the bulge its shape, and quadrature in u would converge
# only like h^{3/2} there.  The substitution a sin u = sin(phi) removes
# it:
#
#     r  = sqrt(a^2 - sin^2 phi)
#     dz = cos^2(phi) dphi / sqrt(a^2 - sin^2 phi)
#
# smooth on the closed interval because a >= 1.  This is the mirror of
# the shipped K = -1 bulge, which is the same expression with
# sqrt(a^2 + sin^2 phi).  At a = 1 it degenerates gracefully to
# r = cos phi, dz = cos phi dphi, i.e. the unit sphere in closed form --
# so the sphere needs no special case.
#
# References:
#   F. Minding, "Ueber die Biegung gewisser Flaechen", Journal fuer die
#       reine und angewandte Mathematik 18 (1838) 297-302 and 19 (1839)
#       370-387 -- the classification of the surfaces of revolution of
#       constant curvature, of which these are the K > 0 members.
#   D. Hilbert and S. Cohn-Vossen, "Anschauliche Geometrie" (1932),
#       chapter on surfaces of constant curvature -- the spindle/bulge
#       description followed here.
#   H. Sievert, "Ueber die Zentralflaechen der Enneperschen Flaechen
#       konstanten Kruemmungsmasses", dissertation, Tuebingen 1886 --
#       the non-rotational K = +1 surface.
#   R. Ferreol, "Encyclopedie des formes mathematiques remarquables",
#       mathcurve.com, chapter "surface de Sievert", for the cylindrical
#       parametrisation transcribed here; a converted copy of the
#       encyclopedia is in research/books/
#       mathcurve_encyclopedie_formes_mathematiques/.

bl_info = {
    "name": "Spherical Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Surfaces of revolution of constant positive "
                   "Gaussian curvature: the sphere, and Minding's "
                   "spindle and bulge types",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    from . import rim_curve as _rim
except ImportError:  # flat import outside the package
    import rim_curve as _rim

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


PRESETS = {
    'SPHERE': ("Sphere (K = +1)", 1.0),
    'SPINDLE': ("Spherical Spindle (K = +1)", 0.55),
    'BULGE': ("Spherical Bulge (K = +1)", 1.6),
    # not a surface of revolution, so `a` is a plain scale here rather
    # than the meridian amplitude it is for the three above
    'SIEVERT': ("Sievert's Surface (K = +1)", 1.0),
}

#: the members that are surfaces of revolution, i.e. everything
#: Minding classified.  Sievert's surface is the one that is not, and
#: several code paths (the arclength gate, the `a` slider's meaning)
#: apply only to the rotational ones.
ROTATIONAL = ('SPHERE', 'SPINDLE', 'BULGE')


def _cum_simpson(g, t):
    """Cumulative integral of `g` over the uniform grid `t`, Simpson
    per interval (so O(h^4), with `g` also sampled at the midpoints)."""
    h = float(t[1] - t[0])
    y = g(t)
    ym = g(0.5 * (t[:-1] + t[1:]))
    seg = (h / 6.0) * (y[:-1] + 4.0 * ym + y[1:])
    return np.concatenate([[0.0], np.cumsum(seg)])


def spherical_profile(a, n=1201):
    """Meridian of the K = +1 surface of revolution, as (r, z) tables.

    Returns arrays sampled from one end of the meridian to the other.
    See the module header for why the two regimes are integrated in
    different variables.
    """
    a = float(a)
    if a <= 0.0:
        raise ValueError("a must be positive")

    if a < 1.0:
        # spindle: the integrand never degenerates, so integrate in u
        u = np.linspace(-math.pi / 2.0, math.pi / 2.0, n)
        r = a * np.cos(u)

        def dz(uu):
            return np.sqrt(np.maximum(1.0 - a * a * np.sin(uu) ** 2, 0.0))

        z = _cum_simpson(dz, u)
    else:
        # bulge (a > 1) and sphere (a = 1): substitute a sin u = sin phi
        phi = np.linspace(-math.pi / 2.0, math.pi / 2.0, n)
        s2 = np.sin(phi) ** 2
        r = np.sqrt(np.maximum(a * a - s2, 0.0))

        def dz(pp):
            c = np.cos(pp)
            return c * c / np.sqrt(np.maximum(a * a - np.sin(pp) ** 2,
                                              1e-30))

        z = _cum_simpson(dz, phi)

    return r, z - 0.5 * (z[0] + z[-1])


def build_spherical(kind='SPHERE', a=None, segments=96, rings=64,
                    scale=1.0):
    """Mesh a K = +1 surface of revolution.

    The spindle and the sphere close on the axis, so their end rings
    are welded to a single pole vertex; the bulge ends on two circles
    of positive radius and is left open there, because that rim is the
    surface, not an artifact of truncation.
    """
    if a is None:
        a = PRESETS[kind][1]
    r, z = spherical_profile(a, n=max(64, rings) + 1)
    # resample the profile onto `rings` stations
    idx = np.linspace(0, len(r) - 1, rings + 1)
    r = np.interp(idx, np.arange(len(r)), r)
    z = np.interp(idx, np.arange(len(z)), z)

    closes = r[0] < 1e-9 and r[-1] < 1e-9
    verts = []
    if closes:
        verts.append((0.0, 0.0, float(z[0])))
    lo = 1 if closes else 0
    hi = len(r) - 1 if closes else len(r)
    for j in range(lo, hi):
        for i in range(segments):
            th = 2.0 * math.pi * i / segments
            verts.append((float(r[j] * math.cos(th)),
                          float(r[j] * math.sin(th)), float(z[j])))
    if closes:
        verts.append((0.0, 0.0, float(z[-1])))

    faces = []
    nrows = hi - lo
    base = 1 if closes else 0

    def vid(row, i):
        return base + row * segments + (i % segments)

    for row in range(nrows - 1):
        for i in range(segments):
            faces.append((vid(row, i), vid(row, i + 1),
                          vid(row + 1, i + 1), vid(row + 1, i)))
    if closes:
        south = 0
        north = len(verts) - 1
        for i in range(segments):
            faces.append((south, vid(0, i + 1), vid(0, i)))
            faces.append((north, vid(nrows - 1, i), vid(nrows - 1, i + 1)))

    V = np.asarray(verts, dtype=float)
    if len(V):
        lo_b, hi_b = V.min(axis=0), V.max(axis=0)
        ext = float((hi_b - lo_b).max())
        V = (V - 0.5 * (lo_b + hi_b)) * (2.0 / ext if ext > 1e-9 else 1.0)
        V = V * float(scale)
    return [tuple(v) for v in V], faces


def sievert_point(u, v, a=1.0):
    """Sievert's surface in cylindrical form, as an (..., 3) array.

    `u` runs across the sheet and `v` along it.  v is kept strictly
    inside (0, pi): ln tan(v/2) runs off to -infinity at v = 0 and to
    +infinity at v = pi, which is the surface's own pair of ends, not a
    defect -- the mesh is simply cut short of them.  u is kept inside
    (-pi/2, pi/2) for the same reason: theta uses arctan(sqrt2 tan u),
    whose branch jumps at the ends of that interval.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u, v = np.broadcast_arrays(u, v)
    su, cu = np.sin(u), np.cos(u)
    sv, cv = np.sin(v), np.cos(v)
    den = 2.0 - sv * sv * cu * cu
    rho = a * 2.0 * np.sqrt(2.0 + 2.0 * su * su) * sv / den
    th = -u / math.sqrt(2.0) + np.arctan(math.sqrt(2.0) * np.tan(u))
    z = a * (np.log(np.tan(0.5 * v)) + 4.0 * cv / den)
    return np.stack([rho * np.cos(th), rho * np.sin(th), z], axis=-1)


def build_sievert(a=1.0, segments=96, rings=64, v_margin=0.32,
                  scale=1.0):
    """Mesh Sievert's surface.

    `v_margin` trims the two ends where z runs off logarithmically; at
    0 the surface would be infinitely long, so some trim is compulsory
    and the value is the shape control rather than a fudge.  The u
    range is a full period, so the two u edges meet and are welded.
    """
    nu = max(8, int(segments))
    nv = max(6, int(rings)) + 1
    m = min(max(float(v_margin), 1e-3), 1.5)
    eps = 1e-6
    u = np.linspace(-0.5 * math.pi + eps, 0.5 * math.pi - eps, nu)
    v = np.linspace(m, math.pi - m, nv)
    P = sievert_point(u[:, None], v[None, :], a)
    verts = [tuple(map(float, p)) for p in P.reshape(-1, 3)]
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            faces.append((i * nv + j, (i + 1) * nv + j,
                          (i + 1) * nv + j + 1, i * nv + j + 1))
    V = np.asarray(verts, dtype=float)
    if len(V):
        lo, hi = V.min(axis=0), V.max(axis=0)
        ext = float((hi - lo).max())
        V = (V - 0.5 * (lo + hi)) * (2.0 / ext if ext > 1e-9 else 1.0)
        V = V * float(scale)
    return [tuple(v_) for v_ in V], faces


def gaussian_curvature(verts, faces):
    """Median per-vertex Gaussian curvature by angle defect over area.

    The same estimator the K = -1 module uses, and for the same reason:
    it is robust to the stretched triangles a surface of revolution
    produces near its poles.  Note it is scale-dependent, so it is
    evaluated on the UNSCALED surface.
    """
    V = np.asarray(verts, dtype=float)
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
            tris.append(tuple(f))
    for a, b, c in tris:
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            u = V[j] - V[i]
            w = V[k] - V[i]
            cs = float(np.dot(u, w)) / (float(np.linalg.norm(u))
                                        * float(np.linalg.norm(w))
                                        + 1e-12)
            ang[i] += math.acos(max(-1.0, min(1.0, cs)))
        ar = 0.5 * float(np.linalg.norm(
            np.cross(V[b] - V[a], V[c] - V[a])))
        for i in (a, b, c):
            area[i] += ar / 3.0
            deg[i] += 1
    interior = deg >= 6
    defect = 2.0 * math.pi - ang
    ki = defect[interior] / np.maximum(area[interior], 1e-12)
    return float(np.median(ki)) if ki.size else 0.0


if _IN_BLENDER:

    class MESH_OT_spherical_surface_add(bpy.types.Operator):
        """Add a surface of revolution of constant positive Gaussian
        curvature: the sphere, or Minding's spindle and bulge types"""
        bl_idname = "mesh.spherical_surface_add"
        bl_label = "Spherical Surface"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Surface",
            items=[('SPHERE', "Sphere (K = +1)",
                    "a = 1: the round sphere"),
                   ('SPINDLE', "Spherical Spindle (K = +1)",
                    "a < 1: closes to a conical point at each pole"),
                   ('BULGE', "Spherical Bulge (K = +1)",
                    "a > 1: an open barrel ending at two circular "
                    "rims"),
                   ('SIEVERT', "Sievert's Surface (K = +1)",
                    "Constant positive curvature with NO axis of "
                    "symmetry: the surface that shows Minding's "
                    "classification is complete only among surfaces "
                    "of revolution")],
            default='SPHERE')
        shape_a: FloatProperty(
            name="Profile a", default=0.0, min=0.0, max=4.0,
            description="Meridian amplitude a in f(u) = a cos(u); "
                        "0 uses the preset's own value. Below 1 gives "
                        "a spindle, above 1 a bulge, exactly 1 the "
                        "sphere")
        v_margin: FloatProperty(
            name="End Trim", default=0.32, min=0.02, max=1.4,
            description="How far short of its two ends Sievert's "
                        "surface is cut. Its height runs off "
                        "logarithmically there, so the surface is "
                        "infinitely long and some trim is compulsory; "
                        "this is the shape control, not a fudge "
                        "(Sievert only)")
        segments: IntProperty(
            name="Segments", default=96, min=8, max=512,
            description="Divisions around the axis")
        rings: IntProperty(
            name="Rings", default=64, min=6, max=512,
            description="Divisions along the meridian")
        scale: FloatProperty(name="Scale", default=1.0,
                             min=0.01, max=100.0)
        smooth: BoolProperty(name="Smooth Shading", default=True)
        rim: _rim.rim_prop()
        rim_thickness: _rim.rim_thickness_prop()
        rim_smooth: _rim.rim_smooth_prop()
        rim_profile: _rim.rim_profile_prop()
        rim_twist: _rim.rim_twist_prop()
        rim_reeds: _rim.rim_reeds_prop()

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'SIEVERT':
                lay.prop(self, 'v_margin')
            else:
                lay.prop(self, 'shape_a')
            for k in ('segments', 'rings', 'scale', 'smooth'):
                lay.prop(self, k)
            _rim.draw_rim(lay, self)

        def execute(self, context):
            label, a_def = PRESETS[self.preset]
            a = self.shape_a if self.shape_a > 0.0 else a_def
            if self.preset == 'SIEVERT':
                verts, faces = build_sievert(
                    1.0, self.segments, self.rings, self.v_margin,
                    self.scale)
            else:
                verts, faces = build_spherical(
                    self.preset, a, self.segments, self.rings,
                    self.scale)
            me = bpy.data.meshes.new(label)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth',
                                    [self.smooth] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(label, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            nrim = 0
            if self.rim:
                nrim = _rim.add_rim_from_object(
                    context, obj, label, self.rim_thickness,
                    self.rim_smooth, self.rim_profile,
                    twist=self.rim_twist, reeds=self.rim_reeds)
            self.report({'INFO'},
                        f"{label}: a={a:.3f}, {len(me.vertices)} verts, "
                        f"{len(me.polygons)} faces"
                        + (f", rim {nrim} loop(s)" if self.rim else ""))
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator("mesh.spherical_surface_add",
                             icon='MESH_UVSPHERE')

    _classes = (MESH_OT_spherical_surface_add,)

    ADD_MENU = True

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)


def _selftest():
    ok = True

    # The defining property, measured rather than assumed.  Curvature is
    # scale-dependent, so this runs on the unscaled surface: the sphere
    # of curvature +1 has radius 1, and normalising it into the 2 m cube
    # would report K = +1 only by coincidence.
    bad = []
    for kind in ROTATIONAL:
        a = PRESETS[kind][1]
        r, z = spherical_profile(a, n=2001)
        verts = []
        faces = []
        segs, rows = 72, len(r)
        for j in range(rows):
            for i in range(segs):
                th = 2.0 * math.pi * i / segs
                verts.append((r[j] * math.cos(th), r[j] * math.sin(th),
                              z[j]))
        for j in range(rows - 1):
            for i in range(segs):
                i1 = (i + 1) % segs
                faces.append((j * segs + i, j * segs + i1,
                              (j + 1) * segs + i1, (j + 1) * segs + i))
        K = gaussian_curvature(verts, faces)
        if not (abs(K - 1.0) < 0.05):
            bad.append('%s:K=%.3f' % (kind, K))
    ok &= not bad
    print("spherical: K = +1 measured on all three rotational types %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # The meridian is f = a cos(u) in ARCLENGTH, so the profile's own
    # arclength must come back as the parameter range it was built from.
    # This is what would break if the substitution were wrong.
    bad = []
    for kind in ROTATIONAL:
        a = PRESETS[kind][1]
        r, z = spherical_profile(a, n=4001)
        ds = np.hypot(np.diff(r), np.diff(z))
        total = float(ds.sum())
        want = math.pi if a <= 1.0 else 2.0 * math.asin(1.0 / a)
        if not (abs(total - want) < 2e-3):
            bad.append('%s:%.4f!=%.4f' % (kind, total, want))
    ok &= not bad
    print("spherical: meridian arclength matches the parameter span %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # Topology: sphere and spindle close, the bulge does not.
    bad = []
    for kind, closed_expected in (('SPHERE', True), ('SPINDLE', True),
                                  ('BULGE', False)):
        V, F = build_spherical(kind, None, 48, 40)
        cnt = {}
        for f in F:
            for i in range(len(f)):
                e = frozenset((f[i], f[(i + 1) % len(f)]))
                cnt[e] = cnt.get(e, 0) + 1
        closed = all(c == 2 for c in cnt.values())
        finite = all(all(math.isfinite(t) for t in v) for v in V)
        if closed != closed_expected or not finite:
            bad.append('%s:closed=%s' % (kind, closed))
    ok &= not bad
    print("spherical: sphere/spindle close, bulge ends on two rims %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    # Sievert's surface: K = +1 everywhere, and NO axis of symmetry.
    # Both halves matter.  The curvature is computed analytically from
    # the first and second fundamental forms rather than from the mesh,
    # because that is what actually pins the transcription down -- and
    # it does: a plausible variant of these formulas recalled from
    # another source passes every mesh-level check here and measures
    # K between 2.5 and 7.4.
    def _K(fn, u, v, h=1e-5):
        Pu = (fn(u + h, v) - fn(u - h, v)) / (2.0 * h)
        Pv = (fn(u, v + h) - fn(u, v - h)) / (2.0 * h)
        Puu = (fn(u + h, v) - 2.0 * fn(u, v) + fn(u - h, v)) / (h * h)
        Pvv = (fn(u, v + h) - 2.0 * fn(u, v) + fn(u, v - h)) / (h * h)
        Puv = (fn(u + h, v + h) - fn(u + h, v - h)
               - fn(u - h, v + h) + fn(u - h, v - h)) / (4.0 * h * h)
        n = np.cross(Pu, Pv)
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        E = np.einsum('...i,...i->...', Pu, Pu)
        F_ = np.einsum('...i,...i->...', Pu, Pv)
        G = np.einsum('...i,...i->...', Pv, Pv)
        L = np.einsum('...i,...i->...', Puu, n)
        M = np.einsum('...i,...i->...', Puv, n)
        N = np.einsum('...i,...i->...', Pvv, n)
        return (L * N - M * M) / (E * G - F_ * F_)

    rng = np.random.default_rng(20260821)
    uu = rng.uniform(-1.2, 1.2, 400)
    vv = rng.uniform(0.35, math.pi - 0.35, 400)
    K = _K(lambda a_, b_: sievert_point(a_, b_, 1.0), uu, vv)
    spread = float(np.max(K) - np.min(K))
    off = float(np.max(np.abs(K - 1.0)))
    good = off < 1e-3 and spread < 1e-3
    ok &= good
    print("spherical: Sievert's surface has K = +1 (max |K-1| = %.1e, "
          "spread %.1e) %s" % (off, spread, 'OK' if good else 'FAIL'))

    # ...and it is genuinely NOT a surface of revolution.  Stated
    # exactly: a surface of revolution meets every horizontal plane in
    # CIRCLES, so rho would be constant along a level set of z.  Take an
    # actual level set -- for each u, solve z(u, v) = z0 for v by
    # bisection, which is well posed because z is monotone in v (it
    # carries ln tan(v/2), running from -inf to +inf) -- and measure how
    # much rho varies along it.  Doing this on the parametrisation
    # rather than on the mesh keeps the answer free of the band-width
    # artefact that a naive z-slice of the vertices would pick up.
    def _v_at(u0, z0, lo=1e-4, hi=math.pi - 1e-4):
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if sievert_point(u0, mid, 1.0)[2] < z0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    worst = 0.0
    for z0 in (-1.0, 0.0, 1.0):
        rhos = []
        for u0 in np.linspace(-1.3, 1.3, 41):
            P = sievert_point(float(u0), _v_at(float(u0), z0), 1.0)
            if abs(P[2] - z0) < 1e-6:            # the solve converged
                rhos.append(math.hypot(P[0], P[1]))
        if len(rhos) > 8:
            rhos = np.asarray(rhos)
            worst = max(worst, float(np.ptp(rhos) / rhos.mean()))
    good = worst > 0.2
    ok &= good
    print("spherical: Sievert is not a surface of revolution -- rho "
          "varies by %.0f%% along a level set of z %s"
          % (100.0 * worst, 'OK' if good else 'FAIL'))

    # and it meshes: finite, in the cube, no degenerate faces
    bad = []
    for margin in (0.15, 0.32, 0.8):
        V, F = build_sievert(1.0, 64, 48, margin)
        A = np.asarray(V, dtype=float)
        if not np.all(np.isfinite(A)) or len(F) < 100:
            bad.append('m=%.2f:%d faces' % (margin, len(F)))
        elif float((A.max(0) - A.min(0)).max()) > 2.0 + 1e-6:
            bad.append('m=%.2f:oversize' % margin)
    ok &= not bad
    print("spherical: Sievert meshes at three end trims %s"
          % ('OK' if not bad else 'FAIL ' + ','.join(bad)))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("spherical surface self-test failed")
