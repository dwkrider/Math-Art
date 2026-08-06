
# Gomboc Generator for Blender
#
# A gomboc is a convex, homogeneous solid that is "mono-monostatic":
# it has exactly ONE stable and ONE unstable balance point, so -- like
# a self-righting toy, but with no added weight and no hollow -- it
# always rolls back to the same resting pose.  Vladimir Arnold
# conjectured such a body could exist; Gabor Domokos and Peter
# Varkonyi proved it and built the first one in 2006.  A true gomboc
# is startlingly close to a sphere (their first solution departs from
# the sphere by only ~1e-5), which is exactly what makes it hard to
# picture.
#
# This generator draws the boundary as a radial ("star-shaped") surface
# r = R(theta, phi) about the centre of mass, using Robert J. Sloan's
# 2023 closed-form analytic gomboc surfaces.  The flatness parameter
# beta exaggerates the departure from a sphere so the shape is actually
# visible as a sculpture; at beta -> 0 it relaxes back to a round ball.
#
#   Sloan gomboc 1:  r^4 = 1 + 4 beta sin(phi) cos(theta - 5 phi)
#   Sloan gomboc 2:  r^4 = 1 + 4 beta sin(phi)
#                            cos( theta - (3 pi / 2)(cos phi - cos^3(phi)/3) )
#
# with phi the polar angle (colatitude, 0..pi) and theta the azimuth
# (0..2pi).  Both poles sit at r = 1, so the mesh closes into a
# watertight ball.  (Note: these analytic surfaces reproduce the
# gomboc's characteristic look and its two radial critical points;
# whether a given beta yields a strictly mono-monostatic *solid*
# depends on the centre-of-mass height over all orientations -- a
# finer condition than the surface radius alone.  They are used here
# for their shape, not as a certified balancing object.)
#
# References:
#   - G. Domokos and P. Varkonyi, "Mono-monostatic bodies: the answer
#     to Arnold's question", The Mathematical Intelligencer 28 (2006),
#     no. 4, 34-38.
#   - P. L. Varkonyi and G. Domokos, "Static equilibria of rigid bodies:
#     dice, pebbles and the Poincare-Hopf theorem", J. Nonlinear Sci.
#     16 (2006), 255-281.
#   - Robert J. Sloan, "An analytic parameterization of the gomboc"
#     (2023); see also Wolfram MathWorld, "Gomboc",
#     https://mathworld.wolfram.com/Gomboc.html .

bl_info = {
    "name": "Gomboc",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Gomboc",
    "description": "Mono-monostatic self-righting solid (after Domokos & "
                   "Varkonyi; analytic surface after Sloan)",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi


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


def _radius(kind, phi, theta, beta):
    """Sloan's analytic gomboc radius R = (r^4)^(1/4)."""
    s = sin(phi)
    if kind == 'SECOND':
        c = cos(phi)
        ph = theta - (1.5 * pi) * (c - c * c * c / 3.0)
    else:                                    # 'FIRST'
        ph = theta - 5.0 * phi
    r4 = 1.0 + 4.0 * beta * s * cos(ph)
    if r4 < 1e-9:
        r4 = 1e-9
    return r4 ** 0.25


def build_gomboc(kind='SECOND', beta=0.17, phi_segments=96,
                 theta_segments=128, scale=1.0):
    """Watertight radial gomboc surface about its centre.

    phi (phi_segments) is colatitude 0..pi; theta (theta_segments) is
    the azimuth 0..2pi and wraps.  The poles collapse to single welded
    vertices.
    """
    m = _Mesh()
    nphi = max(6, phi_segments)
    nth = max(8, theta_segments)

    def pt(i, j):
        phi = pi * i / nphi
        theta = 2.0 * pi * (j % nth) / nth
        r = _radius(kind, phi, theta, beta)
        return (r * sin(phi) * cos(theta) * scale,
                r * sin(phi) * sin(theta) * scale,
                r * cos(phi) * scale)

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
        """Add a gomboc -- a convex, homogeneous self-righting solid """ \
            """with a single stable and single unstable balance point"""
        bl_idname = "mesh.gomboc_add"
        bl_label = "Gomboc"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Variant",
            items=[('SECOND', "Sloan II",
                    "r^4 = 1 + 4 beta sin(phi) cos(theta - "
                    "(3pi/2)(cos phi - cos^3 phi / 3)); smooth single "
                    "crest (beta up to ~0.17)"),
                   ('FIRST', "Sloan I",
                    "r^4 = 1 + 4 beta sin(phi) cos(theta - 5 phi); a "
                    "helically swept ridge (beta up to ~0.15)")],
            default='SECOND')
        beta: FloatProperty(
            name="Flatness beta", default=0.17, min=0.0, max=0.25,
            description="Departure from a sphere; 0 is a round ball. "
                        "Keep <= ~0.15 (Sloan I) / ~0.17 (Sloan II)")
        phi_segments: IntProperty(
            name="Rings", default=96, min=8, max=400,
            description="Segments from pole to pole")
        theta_segments: IntProperty(
            name="Segments", default=128, min=8, max=512,
            description="Segments around the axis")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            verts, faces = build_gomboc(
                self.kind, self.beta, self.phi_segments,
                self.theta_segments, self.scale)

            def _fit(vs):
                # centre on the bbox midpoint, scale largest extent to
                # 2.0 * scale (scale divides out of the ratio).
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
    for kind, beta in (('FIRST', 0.15), ('SECOND', 0.17)):
        v, f = build_gomboc(kind, beta, 48, 64)
        for p in v:
            for c in p:
                assert math.isfinite(c)
        # closed manifold: every edge shared by exactly two faces,
        # Euler characteristic of a sphere = 2.
        cnt = Counter()
        for fc in f:
            n = len(fc)
            for i in range(n):
                a, b = fc[i], fc[(i + 1) % n]
                cnt[(min(a, b), max(a, b))] += 1
        manifold = all(c == 2 for c in cnt.values())
        chi = len(v) - len(cnt) + len(f)
        # convexity sanity: radius must stay strictly positive.
        rmin = min(math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
                   for p in v)
        ok = manifold and chi == 2 and rmin > 0.1
        print(f"gomboc[{kind} beta={beta}]: verts={len(v)} faces={len(f)} "
              f"manifold={manifold} chi={chi}(2) rmin={rmin:.3f} "
              f"{'OK' if ok else 'BAD'}")
        assert ok
