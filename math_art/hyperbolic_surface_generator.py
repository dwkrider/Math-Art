
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

bl_info = {
    "name": "Hyperbolic Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Hyperbolic Surface",
    "description": "Pseudosphere, Dini and Kuen constant-negative-"
                   "curvature surfaces",
    "category": "Add Mesh",
}

import math

import numpy as np


def _pseudosphere(U, V, twist=0.0):
    x = np.cosh(U) ** -1 * np.cos(V)
    y = np.cosh(U) ** -1 * np.sin(V)
    z = U - np.tanh(U)
    return x, y, z


def _dini(U, V, twist=0.2):
    x = np.cos(V) * np.sin(U)
    y = np.sin(V) * np.sin(U)
    z = np.cos(U) + np.log(np.tan(U / 2.0)) + twist * V
    return x, y, z


def _kuen(U, V, twist=0.0):
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


def build_surface(kind, ures, vres, twist=0.2, scale=1.0):
    label, fn, (u0, u1), (v0, v1), wrap = PRESETS[kind]
    us = np.linspace(u0, u1, ures)
    vs = (np.linspace(v0, v1, vres, endpoint=False) if wrap
          else np.linspace(v0, v1, vres))
    U, Vv = np.meshgrid(us, vs, indexing='ij')
    x, y, z = fn(U, Vv, twist)
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
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                            max=100.0)
        shade_smooth: BoolProperty(name="Smooth Shading", default=True)

        def execute(self, context):
            label = PRESETS[self.preset][0]
            verts, faces = build_surface(self.preset, self.ures,
                                         self.vres, self.twist,
                                         self.scale)
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
    # each surface has constant Gaussian curvature ~ -1
    # (Dini with twist 0.2 -> -1/(1+0.2^2) = -0.96)
    for kind in PRESETS:
        fn = PRESETS[kind][1]
        _, _, (u0, u1), (v0, v1), wrap = PRESETS[kind]
        us = np.linspace(u0, u1, 80)
        vs = (np.linspace(v0, v1, 80, endpoint=False) if wrap
              else np.linspace(v0, v1, 80))
        U, Vv = np.meshgrid(us, vs, indexing='ij')
        x, y, z = fn(U, Vv, 0.2)
        Vraw = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
        faces = _grid_faces(80, 80, wrap)
        K = mean_curvature(Vraw, faces)
        print(f"{kind:13s}: V={len(Vraw)} F={len(faces)} "
              f"meanK={K:+.3f}")
