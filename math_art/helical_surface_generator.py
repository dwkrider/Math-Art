
# Helical Surfaces generator for Blender: three classic
# helical / spiral parametric surfaces.
#
#   HYPERBOLIC_HELICOID -- the helicoid's hyperbolic cousin,
#     a twisted band whose points all share the denominator
#     1 + cosh(u) cosh(v); the torsion tau sets how fast the
#     band winds around its axis.
#   SEASHELL -- a conical spiral shell: a circular tube whose
#     radius shrinks linearly to an apex while it winds n
#     whorls around the axis and rises by a fixed height.
#   CORKSCREW -- the "twisted sphere": a sphere stretched
#     along a diameter and sheared upward while it turns, so
#     each meridian circle climbs by b per radian of turn.
#
# Each surface is built by a pure-python function that works
# without bpy, so this file can be run standalone for its
# self-tests.  Seams where a parameter wraps (the seashell's
# tube direction, the corkscrew's meridian circles) are welded
# by index so the meshes are seamless; the seashell's apex,
# where the whole tube collapses to one point, is emitted as a
# single vertex with a triangle fan.
#
# References:
# - The helicoid is a classical minimal and ruled surface
#   (J. B. C. Meusnier, 1776); the hyperbolic helicoid, conical
#   seashell and twisted-sphere forms here are standard parametric
#   surfaces. See A. Gray, E. Abbena, S. Salamon, "Modern
#   Differential Geometry of Curves and Surfaces with Mathematica"
#   (3rd ed., 2006), and J. Meier's gallery (3d-meier.de).

bl_info = {
    "name": "Helical Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Hyperbolic helicoid, seashell and "
                   "corkscrew parametric surfaces",
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
    from bpy.props import (IntProperty, FloatProperty,
                           EnumProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def build_hyperbolic_helicoid(tau=2.5, extent=4.0, res=96):
    """(verts, faces): the hyperbolic helicoid

        x = sinh(v) cos(tau u) / (1 + cosh(u) cosh(v))
        y = sinh(v) sin(tau u) / (1 + cosh(u) cosh(v))
        z = cosh(v) sinh(u) / (1 + cosh(u) cosh(v))

    over u, v in [-extent, extent].  An open twisted band: no
    parameter wraps, so there is no seam to weld.  The shared
    denominator is >= 2 everywhere, so every point is finite."""
    nu = nv = max(2, int(res))
    u = np.linspace(-extent, extent, nu + 1)
    v = np.linspace(-extent, extent, nv + 1)
    U, V = np.meshgrid(u, v, indexing='ij')
    den = 1.0 + np.cosh(U) * np.cosh(V)
    X = np.sinh(V) * np.cos(tau * U) / den
    Y = np.sinh(V) * np.sin(tau * U) / den
    Z = np.cosh(V) * np.sinh(U) / den
    verts = [(X[i, j], Y[i, j], Z[i, j])
             for i in range(nu + 1) for j in range(nv + 1)]
    faces = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            b = (i + 1) * (nv + 1) + j
            faces.append([a, b, b + 1, a + 1])
    return verts, faces


def build_seashell(whorls=2, aspect=0.2, height=1.0,
                   opening=0.1, res=96):
    """(verts, faces): a conical seashell surface

        r(u, v) = (1 - v/2pi) (1 + cos u) + c
        x = r cos(n v)
        y = r sin(n v)
        z = b v / 2pi + a sin(u) (1 - v/2pi)

    with n = whorls, a = aspect (vertical stretch of the tube
    section), b = height (total rise), c = opening (offset
    radius that keeps the mouth open), u, v in [0, 2pi].  The
    tube direction u wraps and is welded by index; at v = 2pi
    the whole tube collapses to the apex (c cos 2pi n,
    c sin 2pi n, b), emitted once and fanned with triangles."""
    nu = nv = max(3, int(res))
    two_pi = 2.0 * math.pi
    n, a, b, c = float(whorls), aspect, height, opening
    verts = []
    for j in range(nv):            # spiral rows, apex excluded
        v = two_pi * j / nv
        taper = 1.0 - v / two_pi
        for i in range(nu):        # tube ring, u wraps
            u = two_pi * i / nu
            r = taper * (1.0 + math.cos(u)) + c
            verts.append((r * math.cos(n * v),
                          r * math.sin(n * v),
                          b * v / two_pi
                          + a * math.sin(u) * taper))
    apex = len(verts)
    verts.append((c * math.cos(two_pi * n),
                  c * math.sin(two_pi * n), b))
    faces = []
    for j in range(nv - 1):
        for i in range(nu):
            i2 = (i + 1) % nu
            faces.append([j * nu + i, j * nu + i2,
                          (j + 1) * nu + i2, (j + 1) * nu + i])
    last = (nv - 1) * nu
    for i in range(nu):
        faces.append([last + i, last + (i + 1) % nu, apex])
    return verts, faces


def build_corkscrew(a=1.0, b=0.3, res=96):
    """(verts, faces): the corkscrew surface (twisted sphere)

        x = a cos(u) cos(v)
        y = a sin(u) cos(v)
        z = a sin(v) + b u

    with a the sphere radius and b the axial rise per radian
    of twist, u, v in [0, 2pi].  Each meridian circle (fixed
    u, v wrapping) is welded by index; u is left open because
    the twist offsets its two ends by 2 pi b in z."""
    nu = nv = max(3, int(res))
    two_pi = 2.0 * math.pi
    verts = []
    for i in range(nu + 1):        # twist direction, open
        u = two_pi * i / nu
        for j in range(nv):        # meridian circle, v wraps
            v = two_pi * j / nv
            verts.append((a * math.cos(u) * math.cos(v),
                          a * math.sin(u) * math.cos(v),
                          a * math.sin(v) + b * u))
    faces = []
    for i in range(nu):
        for j in range(nv):
            j2 = (j + 1) % nv
            faces.append([i * nv + j, i * nv + j2,
                          (i + 1) * nv + j2, (i + 1) * nv + j])
    return verts, faces


_SURFACES = [
    ('HYPERBOLIC_HELICOID', "Hyperbolic Helicoid",
     "Twisted band with torsion tau over the shared "
     "denominator 1 + cosh(u) cosh(v)"),
    ('SEASHELL', "Seashell",
     "Conical spiral shell winding n whorls to an apex"),
    ('CORKSCREW', "Corkscrew",
     "Twisted sphere: a sphere sheared upward as it turns"),
]


if _IN_BLENDER:

    class MESH_OT_helical_surface_add(bpy.types.Operator):
        """Add a classic helical parametric surface: the
        hyperbolic helicoid, a conical seashell, or the
        corkscrew surface (twisted sphere)"""
        bl_idname = "mesh.helical_surface_add"
        bl_label = "Helical Surface"
        bl_options = {'REGISTER', 'UNDO'}
        rim: _rim.rim_prop()
        rim_thickness: _rim.rim_thickness_prop()
        rim_smooth: _rim.rim_smooth_prop()
        rim_profile: _rim.rim_profile_prop()
        rim_twist: _rim.rim_twist_prop()

        surface: EnumProperty(
            name="Surface", items=_SURFACES,
            default='HYPERBOLIC_HELICOID')
        tau: FloatProperty(
            name="Torsion", default=2.5, min=0.0, max=20.0,
            description="Torsion tau of the hyperbolic "
                        "helicoid (turns per unit of the "
                        "spine parameter)")
        extent: FloatProperty(
            name="Extent", default=4.0, min=0.5, max=10.0,
            description="Half-range of both parameters of "
                        "the hyperbolic helicoid")
        whorls: IntProperty(
            name="Whorls", default=2, min=1, max=8,
            description="Number of turns of the shell spiral")
        aspect: FloatProperty(
            name="Tube Aspect", default=1.0, min=0.0, max=2.0,
            description="Vertical stretch of the tube "
                        "cross-section (a)")
        height: FloatProperty(
            name="Height", default=4.0, min=0.0, max=10.0,
            description="Total rise of the shell from mouth "
                        "to apex (b)")
        opening: FloatProperty(
            name="Opening", default=0.37, min=0.0, max=2.0,
            description="Offset radius that keeps the shell "
                        "mouth open (c)")
        cork_a: FloatProperty(
            name="Sphere Radius", default=0.5, min=0.01,
            max=10.0,
            description="Radius a of the twisted sphere")
        cork_b: FloatProperty(
            name="Twist Rise", default=0.3, min=0.0, max=5.0,
            description="Axial rise b per radian of twist")
        resolution: IntProperty(
            name="Resolution", default=96, min=8, max=512,
            description="Segments along each parametric "
                        "direction")
        smooth: BoolProperty(name="Smooth Shading",
                             default=True)
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="Solidify modifier thickness (0 = raw "
                        "surface)")
        scale: FloatProperty(name="Scale", default=1.0,
                             min=0.01, max=100.0)

        def execute(self, context):
            if self.surface == 'HYPERBOLIC_HELICOID':
                verts, faces = build_hyperbolic_helicoid(
                    self.tau, self.extent, self.resolution)
                name = "Hyperbolic Helicoid"
            elif self.surface == 'SEASHELL':
                verts, faces = build_seashell(
                    self.whorls, self.aspect, self.height,
                    self.opening, self.resolution)
                name = "Seashell"
            else:
                verts, faces = build_corkscrew(
                    self.cork_a, self.cork_b, self.resolution)
                name = "Corkscrew"
            # fit (roughly) within a 2 x scale cube at the origin
            lo = [min(v[k] for v in verts) for k in range(3)]
            hi = [max(v[k] for v in verts) for k in range(3)]
            half = max((hi[k] - lo[k]) / 2.0 for k in range(3)) \
                or 1.0
            s = self.scale / half
            verts = [tuple((v[k] - (lo[k] + hi[k]) / 2.0) * s
                           for k in range(3)) for v in verts]
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set(
                'use_smooth', [self.smooth] * len(me.polygons))
            me.update()
            obj = bpy.data.objects.new(name, me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            if self.thickness > 0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'},
                        f"{name}: V={len(me.vertices)} "
                        f"F={len(me.polygons)}")
            if self.rim:
                _ob = context.active_object
                if _ob is not None:
                    _rim.add_rim_from_object(
                        context, _ob, _ob.name,
                        self.rim_thickness, self.rim_smooth,
                        self.rim_profile, twist=self.rim_twist)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'surface')
            if self.surface == 'HYPERBOLIC_HELICOID':
                keys = ('tau', 'extent')
            elif self.surface == 'SEASHELL':
                keys = ('whorls', 'aspect', 'height',
                        'opening')
            else:
                keys = ('cork_a', 'cork_b')
            for k in keys + ('resolution', 'smooth',
                             'thickness', 'scale'):
                lay.prop(self, k)

            _rim.draw_rim(lay, self)
    def _menu_func(self, context):
        self.layout.operator("mesh.helical_surface_add",
                             icon='MOD_SCREW')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_helical_surface_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_helical_surface_add)


def _selftest():
    builds = [
        ("hyperbolic helicoid",
         lambda: build_hyperbolic_helicoid(res=48)),
        ("seashell", lambda: build_seashell(res=48)),
        ("corkscrew", lambda: build_corkscrew(res=48)),
    ]
    for label, fn in builds:
        verts, faces = fn()
        assert len(verts) > 0 and len(faces) > 0
        finite = all(all(math.isfinite(c) for c in v)
                     for v in verts)
        assert finite, f"{label}: non-finite coordinate"
        valid = all(0 <= i < len(verts)
                    for f in faces for i in f)
        assert valid, f"{label}: face index out of range"
        print(f"{label}: V={len(verts)} F={len(faces)} "
              f"finite={finite} indices ok={valid}")
    # the seashell apex (last vertex, where the tube
    # collapses) must be a clean finite point
    verts, faces = build_seashell(res=48)
    apex = verts[-1]
    assert all(math.isfinite(c) for c in apex)
    assert not any(math.isnan(c) for c in apex)
    # every apex triangle references the single apex vertex
    tris = [f for f in faces if len(f) == 3]
    assert len(tris) == 48
    assert all(f[2] == len(verts) - 1 for f in tris)
    print(f"seashell apex {tuple(round(c, 6) for c in apex)}"
          f" welded into {len(tris)} triangles")
    print("helical standalone tests passed")
