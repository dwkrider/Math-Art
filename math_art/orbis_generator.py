
# Orbis / Holey Roller Generator for Blender
#
# Two LINKED tori (rings) -- each threaded through the other's hole,
# like two links of a chain -- roll together as a single object with a
# wobbling, meandering path.  Peer Clahsen designed such a two-ring play
# object, the "Orbis", for the Swiss Naef toy company; Kenneth Brecher
# and Randy Rhine built a large wooden sculptural version they call the
# "Holey Roller".
#
# The two ring circles are placed exactly like the oloid's two circles:
# radius R, centres R/2 either side of the origin in perpendicular
# planes, so each circle passes through the other's centre and the rings
# are Hopf-linked.  A tube of radius r is swept around each (r < R/2, so
# a ring passes cleanly through the other's hole without the tubes
# touching); `angle` may tilt the second ring about the shared axis.
# The linked pair is inseparable, so it rolls as one; as with the
# compound polyhedra it is left as an immersed pair, not Boolean-merged.
#
# References:
#   - P. Clahsen, "Objeux", Naef Herstellung und Verlag, 1967/1976.
#   - K. Brecher, "Rolloids", Bridges 2023 Conference Proceedings,
#     345-352 (the "Orbis" and the "Holey Roller"; local copy at
#     research/bridges/2023/bridges2023-345/).

bl_info = {
    "name": "Orbis / Holey Roller",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Orbis / Holey Roller",
    "description": "Two tori tilted about a shared diameter -- a "
                   "two-ring rolling object (Clahsen / Brecher)",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi, radians


def _torus(R, r, nu, nv, center=(0.0, 0.0, 0.0), axis='Z', rot_x=0.0):
    """Tube of radius r around a circle of radius R (in the xy-plane for
    axis 'Z', the xz-plane for axis 'Y'), shifted by `center` and then
    rotated by rot_x radians about the X axis.  Local indices from 0."""
    ca, sa = cos(rot_x), sin(rot_x)
    verts = []
    for i in range(nu):
        u = 2.0 * pi * i / nu
        cu, su = cos(u), sin(u)
        for j in range(nv):
            v = 2.0 * pi * j / nv
            rr = R + r * cos(v)
            zz = r * sin(v)
            if axis == 'Z':
                x, y, z = rr * cu, rr * su, zz       # ring in xy-plane
            else:
                x, y, z = rr * cu, zz, rr * su       # ring in xz-plane
            x += center[0]
            y += center[1]
            z += center[2]
            verts.append((x, y * ca - z * sa, y * sa + z * ca))

    def vid(i, j):
        return (i % nu) * nv + (j % nv)

    faces = []
    for i in range(nu):
        for j in range(nv):
            faces.append([vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)])
    return verts, faces


def build_orbis(major=1.0, tube=0.28, angle=0.0, nu=96, nv=32):
    """Two LINKED tori (a Hopf link of solid rings): the two ring
    circles are placed like the oloid's two circles -- radius R, centres
    R/2 either side of the origin in perpendicular planes -- so each ring
    passes through the other's central hole.  `angle` tilts the second
    ring about the shared X axis.  Combined into one mesh (an immersed,
    inseparable pair that rolls as one)."""
    R = major
    c = 0.5 * R                                # centres 2c = R apart
    vA, fA = _torus(R, tube, nu, nv, center=(-c, 0.0, 0.0), axis='Z')
    vB, fB = _torus(R, tube, nu, nv, center=(c, 0.0, 0.0), axis='Y',
                    rot_x=radians(angle))
    off = len(vA)
    verts = vA + vB
    faces = fA + [[k + off for k in fc] for fc in fB]
    return verts, faces


try:
    import bpy
    import bmesh
    from bpy.props import IntProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_orbis_add(bpy.types.Operator):
        """Add an Orbis / Holey Roller -- two tori tilted about a """ \
            """shared diameter, rolling as one"""
        bl_idname = "mesh.orbis_add"
        bl_label = "Orbis / Holey Roller"
        bl_options = {'REGISTER', 'UNDO'}

        tube: FloatProperty(
            name="Tube Radius", default=0.28, min=0.02, max=0.49,
            description="Ring thickness (fraction of the ring radius; "
                        "< 0.5 so the rings link without the tubes "
                        "touching)")
        angle: FloatProperty(
            name="Ring Tilt", default=0.0, min=-90.0, max=90.0,
            description="Tilt of the second ring about the shared axis "
                        "(0 = perpendicular linked rings)")
        major_segments: IntProperty(
            name="Ring Segments", default=96, min=12, max=512)
        tube_segments: IntProperty(
            name="Tube Segments", default=32, min=6, max=128)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            verts, faces = build_orbis(
                1.0, self.tube, self.angle,
                self.major_segments, self.tube_segments)

            def _fit(vs):
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

            me = bpy.data.meshes.new("Orbis")
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
            obj = bpy.data.objects.new("Orbis", me)
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
            lay.prop(self, 'tube')
            lay.prop(self, 'angle')
            lay.prop(self, 'major_segments')
            lay.prop(self, 'tube_segments')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.orbis_add", icon='MESH_TORUS')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_orbis_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_orbis_add)


def _selftest():
    from collections import Counter
    v, f = build_orbis(1.0, 0.28, 0.0, 64, 24)
    # each torus is a watertight torus (every edge shared by two faces,
    # chi = 0); the linked pair has chi = 0 and two components.
    cnt = Counter()
    for fc in f:
        n = len(fc)
        for i in range(n):
            a, b = fc[i], fc[(i + 1) % n]
            cnt[(min(a, b), max(a, b))] += 1
    manifold = all(c == 2 for c in cnt.values())
    chi = len(v) - len(cnt) + len(f)
    ok = manifold and chi == 0 and len(v) == 2 * 64 * 24
    print(f"orbis: V={len(v)} F={len(f)} manifold={manifold} chi={chi}(0) "
          f"{'OK' if ok else 'BAD'}")
    assert ok
