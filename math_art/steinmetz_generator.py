
# Steinmetz Solid Generator for Blender
#
# A Steinmetz solid is the intersection of two or three equal-radius
# circular cylinders whose axes meet at right angles.
#
#   * BICYLINDER -- two perpendicular cylinders.  Its surface is two
#     cylindrical "lune" patches meeting along a pair of planar
#     ellipse-like edges; placed on an incline and nudged, it rolls in
#     a straight line (Brecher's "rolloids").
#   * TRICYLINDER -- three mutually perpendicular cylinders; a rounded
#     cube-like solid with twelve curved edges.
#
# The bicylinder's volume 16 r^3 / 3 was found by Archimedes (Method of
# Mechanical Theorems) and independently by the electrical engineer
# Charles Proteus Steinmetz, after whom the solid is named.  The
# bicylinder is built here directly as an exact watertight mesh of its
# two cylinder patches (the patch grids follow the cylinders, so the
# curved edges stay clean); the tricylinder is built by Boolean
# intersection of three cylinders.
#
# References:
#   - Archimedes, "The Method of Mechanical Theorems" (c. 250 BC);
#     T. L. Heath (ed.), "The Works of Archimedes", Cambridge, 1897.
#   - M. Moore, "Symmetrical Intersections of Right Circular Cylinders",
#     The Mathematical Gazette 58 (1974), 181-185.
#   - K. Brecher, "Rolloids", Bridges 2023 Conference Proceedings,
#     345-352 (the bicylinder as a straight-line roller; local copy at
#     research/bridges/2023/bridges2023-345/).

bl_info = {
    "name": "Steinmetz Solid",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Steinmetz Solid",
    "description": "Bicylinder / tricylinder: the intersection of "
                   "perpendicular cylinders (a straight-line roller)",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi, sqrt


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


def build_bicylinder(radius=1.0, segments=64):
    """Exact watertight bicylinder: the two cylinder patches
    (axis X: y^2 + z^2 = r^2, axis Y: x^2 + z^2 = r^2), each clipped to
    the interior of the other cylinder.  Their boundaries coincide on
    the two intersection edges (x = +-y), so they weld into one closed
    surface."""
    m = _Mesh()
    na = max(8, segments)
    ns = max(4, segments // 2)

    def px(a, s):                       # patch on cylinder about X
        ca = cos(a)
        return (s * radius * abs(ca), radius * ca, radius * sin(a))

    def py(b, s):                       # patch on cylinder about Y
        cb = cos(b)
        return (radius * cb, s * radius * abs(cb), radius * sin(b))

    for patch in (px, py):
        for i in range(na):
            a0 = 2.0 * pi * i / na
            a1 = 2.0 * pi * (i + 1) / na
            for j in range(ns):
                s0 = -1.0 + 2.0 * j / ns
                s1 = -1.0 + 2.0 * (j + 1) / ns
                m.quad(patch(a0, s0), patch(a1, s0),
                       patch(a1, s1), patch(a0, s1))
    return m.verts, m.faces


# radial (star-shaped) builder -- used for the headless self-test and
# as the non-Blender fallback; the intersection of infinite cylinders
# is convex and contains the origin, so r(u) = min over cylinders of
# r / |u projected off that cylinder's axis|.
def _steinmetz_radius(kind, u, radius):
    ux, uy, uz = u
    big = 1e18
    txy = uy * uy + uz * uz          # off the X axis
    tyz = ux * ux + uz * uz          # off the Y axis
    tX = radius / sqrt(txy) if txy > 1e-12 else big
    tY = radius / sqrt(tyz) if tyz > 1e-12 else big
    if kind == 'TRICYLINDER':
        tzz = ux * ux + uy * uy      # off the Z axis
        tZ = radius / sqrt(tzz) if tzz > 1e-12 else big
        return min(tX, tY, tZ)
    return min(tX, tY)


def build_steinmetz_radial(kind='TRICYLINDER', radius=1.0,
                           phi_segments=96, theta_segments=160):
    m = _Mesh()
    nphi = max(8, phi_segments)
    nth = max(8, theta_segments)

    def pt(i, j):
        phi = pi * i / nphi
        theta = 2.0 * pi * (j % nth) / nth
        u = (sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi))
        r = _steinmetz_radius(kind, u, radius)
        return (r * u[0], r * u[1], r * u[2])

    grid = [[pt(i, j) for j in range(nth)] for i in range(nphi + 1)]
    for i in range(nphi):
        for j in range(nth):
            jn = (j + 1) % nth
            m.quad(grid[i][j], grid[i][jn], grid[i + 1][jn], grid[i + 1][j])
    return m.verts, m.faces


try:
    from .sharp_creases import mark_sharp_by_angle
except ImportError:                     # flat import outside the package
    from sharp_creases import mark_sharp_by_angle

try:
    import bpy
    import bmesh
    import mathutils
    from bpy.props import (IntProperty, FloatProperty,
                           EnumProperty, BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _cylinder_obj(coll, axis, r, length, seg):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False,
                              segments=seg, radius1=r, radius2=r,
                              depth=length)
        if axis == 'X':
            mat = mathutils.Matrix.Rotation(pi / 2.0, 4, 'Y')
        elif axis == 'Y':
            mat = mathutils.Matrix.Rotation(pi / 2.0, 4, 'X')
        else:
            mat = mathutils.Matrix.Identity(4)
        bmesh.ops.transform(bm, matrix=mat, verts=bm.verts)
        me = bpy.data.meshes.new("sm_tmp")
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new("sm_tmp", me)
        coll.objects.link(o)
        return o

    def _boolean(ctx, a, b):
        md = a.modifiers.new("b", 'BOOLEAN')
        md.operation = 'INTERSECT'
        md.object = b
        md.solver = 'EXACT'
        with ctx.temp_override(object=a, active_object=a,
                               selected_objects=[a]):
            bpy.ops.object.modifier_apply(modifier=md.name)
        bpy.data.objects.remove(b, do_unlink=True)
        return a

    def _build_tricylinder_csg(ctx, r, seg):
        coll = ctx.collection
        length = 4.0 * r
        res = _cylinder_obj(coll, 'X', r, length, seg)
        for axis in ('Y', 'Z'):
            res = _boolean(ctx, res, _cylinder_obj(coll, axis, r,
                                                   length, seg))
        bm = bmesh.new()
        bm.from_mesh(res.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5 * r)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(res.data)
        bm.free()
        verts = [v.co[:] for v in res.data.vertices]
        faces = [list(p.vertices) for p in res.data.polygons]
        bpy.data.objects.remove(res, do_unlink=True)
        return verts, faces

    class MESH_OT_steinmetz_add(bpy.types.Operator):
        """Add a Steinmetz solid -- the intersection of two or three """ \
            """equal cylinders meeting at right angles"""
        bl_idname = "mesh.steinmetz_add"
        bl_label = "Steinmetz Solid"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Shape",
            items=[('BICYLINDER', "Bicylinder (Steinmetz)",
                    "Two perpendicular cylinders -- rolls in a straight "
                    "line down an incline"),
                   ('TRICYLINDER', "Tricylinder",
                    "Three mutually perpendicular cylinders -- a rounded "
                    "cube with twelve curved edges")],
            default='BICYLINDER')
        radius: FloatProperty(
            name="Radius", default=1.0, min=0.05, max=100.0,
            description="Common cylinder radius")
        sharp_edges: BoolProperty(
            name="Sharp Edges", default=True,
            description="Mark the solid's fold curves sharp (and "
                        "creased). Two cylinders meet along a pair of planar ellipse-like edges; three meet along twelve. The surface is smooth "
                        "everywhere else, so shading straight across "
                        "the fold rounds off the one feature that "
                        "defines the shape")
        segments: IntProperty(
            name="Segments", default=64, min=8, max=512,
            description="Cylinder segments (mesh resolution)")

        def execute(self, context):
            if self.kind == 'BICYLINDER':
                verts, faces = build_bicylinder(self.radius, self.segments)
            else:
                verts, faces = _build_tricylinder_csg(context, self.radius,
                                                      self.segments)

            verts = [tuple(map(float, v)) for v in verts]

            def _fit(vs):
                if not vs:
                    return vs
                xs = [v[0] for v in vs]
                ys = [v[1] for v in vs]
                zs = [v[2] for v in vs]
                cx = 0.5 * (min(xs) + max(xs))
                cy = 0.5 * (min(ys) + max(ys))
                cz = 0.5 * (min(zs) + max(zs))
                return [(v[0] - cx, v[1] - cy, v[2] - cz) for v in vs]

            me = bpy.data.meshes.new("Steinmetz")
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
                mark_sharp_by_angle(me, 45.0)
            me.update()
            obj = bpy.data.objects.new(f"Steinmetz {self.kind.title()}", me)
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
            lay.prop(self, 'radius')
            lay.prop(self, 'segments')

    def _menu_func(self, context):
        self.layout.operator("mesh.steinmetz_add", icon='MESH_CYLINDER')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_steinmetz_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_steinmetz_add)


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
    # Bicylinder (exact patch mesh): watertight (chi = 2), and every
    # vertex lies on the surface -- i.e. it is on one cylinder
    # (max(x^2+z^2, y^2+z^2) = r^2) and inside the other.
    r = 1.0
    v, f = build_bicylinder(r, 64)
    man, ne = _edge_manifold(f)
    chi = len(v) - ne + len(f)
    worst = 0.0
    for x, y, z in v:
        cyl = max(x * x + z * z, y * y + z * z)   # binding cylinder
        worst = max(worst, abs(cyl - r * r))
    ok = man and chi == 2 and worst < 1e-9
    print(f"steinmetz[bicylinder]: V={len(v)} F={len(f)} manifold={man} "
          f"chi={chi}(2) max|on-surface err|={worst:.1e} "
          f"{'OK' if ok else 'BAD'}")
    assert ok

    # Tricylinder (radial proxy for the CSG operator build): watertight,
    # convex, radius stays within [r, sqrt2 r].
    v, f = build_steinmetz_radial('TRICYLINDER', 1.0, 64, 96)
    man, ne = _edge_manifold(f)
    chi = len(v) - ne + len(f)
    rr = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in v]
    ok = man and chi == 2 and min(rr) > 0.99 and max(rr) < 1.42
    print(f"steinmetz[tricylinder]: V={len(v)} F={len(f)} manifold={man} "
          f"chi={chi}(2) R=[{min(rr):.3f},{max(rr):.3f}] "
          f"{'OK' if ok else 'BAD'}")
    assert ok
