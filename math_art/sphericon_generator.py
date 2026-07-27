
# Sphericon / Polysphericon Generator for Blender
#
# A sphericon is a developable roller built from a regular polygon:
# take the solid of revolution of a regular n-gon about one of its axes
# of symmetry, slice it in half with a plane through that axis, rotate
# one half so the polygon cross-section maps back onto itself, and
# rejoin.  The classic sphericon is the (4,1) case -- a square swept
# into a bicone, halved and rotated 90 degrees.  The general (n,k)
# "polysphericon" rotates one half by k steps of 360/n degrees.
#
#   - even n: the axis passes through two opposite vertices, so both
#     ends of the profile are cone apexes (n=4 gives the bicone).
#   - odd  n: the only symmetry axis passes through one vertex and the
#     midpoint of the opposite edge, so one end is an apex and the
#     other a (split) disc.  n = 7, k = 1 is the heptagonal sphericon.
#
# The surface is a set of conical bands.  Each half is a half-turn of
# the profile; the second half is turned by k*360/n about the cut
# plane's normal so its cross-section (the regular polygon) lands on
# itself.  Because that rotation carries a profile vertex onto the next
# one -- and, for odd n, the bottom edge-midpoint onto the next
# edge-midpoint -- the profile is sampled at every polygon vertex AND
# every edge midpoint so the two halves weld exactly along the seam.
#
# References:
#   - David Swart, "Arcs on Spheres and Snakes on Planes", Bridges 2024
#     (the sphericon = solid of revolution of a regular polygon, halved
#     and its cross-section rotated onto itself).
#   - I. Sabitov / P. Roberts (polysphericons); H-ITS Polysphericons
#     project, https://www.h-its.org/projects/polysphericons/ .
#   - C. J. Roberts, sphericon; Wikipedia "sphericon".

bl_info = {
    "name": "Sphericon",
    "author": "Math Art project (after Swart / the polysphericon "
              "family)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Sphericon",
    "description": "Sphericon / polysphericon developable rollers from "
                   "regular polygons (incl. the heptagonal sphericon)",
    "category": "Add Mesh",
}

import math
from math import sin, cos, pi


def _vkey(p):
    return (round(p[0], 8), round(p[1], 8), round(p[2], 8))


class _TagMesh:
    """Vertex-welding mesh builder that also tags each face (band id)."""

    def __init__(self):
        self.verts = []
        self.faces = []
        self.tags = []
        self._ids = {}

    def _vid(self, p):
        k = _vkey(p)
        i = self._ids.get(k)
        if i is None:
            i = len(self.verts)
            self._ids[k] = i
            self.verts.append(p)
        return i

    def face(self, pts, tag):
        ids = []
        for p in pts:
            i = self._vid(p)
            if i not in ids:
                ids.append(i)
        if len(ids) >= 3:
            self.faces.append(ids)
            self.tags.append(tag)


def _profile(n, detail=2):
    """Right-hand profile (r >= 0) of a regular n-gon revolved about a
    symmetry axis, top apex to bottom axis-point, as a list of
    (band_index, r, z).  Each polygon edge is split into `detail`
    steps (>=2, even) so vertices *and* midpoints are sampled; the odd
    half-edge at the bottom gets detail/2 steps to keep the sampling
    uniform around the polygon (needed for an exact seam weld)."""
    detail = max(2, detail - (detail % 2))     # force even, >= 2
    odd = (n % 2 == 1)
    last = (n - 1) // 2 if odd else n // 2
    V = [(sin(k * 2 * pi / n), cos(k * 2 * pi / n))
         for k in range(last + 1)]             # V_0 .. V_last, x >= 0
    segs = [(V[k], V[k + 1], detail) for k in range(last)]
    if odd:
        segs.append((V[last], (0.0, -cos(pi / n)), detail // 2))
    pts = [(0, V[0][0], V[0][1])]              # top apex, band 0
    band = 0
    for (p0, p1, steps) in segs:
        for s in range(1, steps + 1):
            t = s / steps
            pts.append((band, p0[0] + (p1[0] - p0[0]) * t,
                        p0[1] + (p1[1] - p0[1]) * t))
        band += 1
    return pts


def build_sphericon(sides=4, steps=1, segments=96, scale=1.0):
    """Watertight (n, k) polysphericon.  `sides` = n (>=3), `steps` = k
    (the half is turned by k*360/n degrees), `segments` = angular
    resolution over a half turn.  Returns (verts, faces, band_ids)."""
    n = max(3, int(sides))
    prof = _profile(n)
    alpha = (int(steps) % n) * 2.0 * pi / n
    ca, sa = cos(alpha), sin(alpha)
    seg = max(6, int(segments))
    m = _TagMesh()

    def vA(pi_idx, j):
        _, r, z = prof[pi_idx]
        th = pi * j / seg
        return (r * cos(th), r * sin(th), z)

    def vB(pi_idx, j):
        _, r, z = prof[pi_idx]
        th = pi + pi * j / seg
        x, y, zz = r * cos(th), r * sin(th), z
        return (x * ca + zz * sa, y, -x * sa + zz * ca)   # rotate y

    for vfn in (vA, vB):
        for k in range(len(prof) - 1):
            band = prof[k + 1][0] if prof[k][1] == 0.0 else prof[k][0]
            for j in range(seg):
                m.face([vfn(k, j), vfn(k, j + 1),
                        vfn(k + 1, j + 1), vfn(k + 1, j)], band)
    return m.verts, m.faces, m.tags


try:
    import bpy
    import bmesh
    from mathutils import Vector
    from bpy.props import (IntProperty, FloatProperty, BoolProperty,
                           EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    _PALETTE = [
        (0.79, 0.52, 0.47), (0.84, 0.76, 0.61), (0.60, 0.72, 0.79),
        (0.73, 0.61, 0.75), (0.71, 0.75, 0.63), (0.83, 0.63, 0.58),
        (0.64, 0.66, 0.81), (0.65, 0.54, 0.67),
    ]

    def _band_materials(me, tags, coloring):
        if coloring == 'NONE':
            mat = bpy.data.materials.new("Sphericon")
            mat.use_nodes = True
            b = mat.node_tree.nodes.get("Principled BSDF")
            b.inputs["Base Color"].default_value = (0.8, 0.8, 0.82, 1)
            b.inputs["Roughness"].default_value = 0.4
            me.materials.append(mat)
            return
        nb = (max(tags) + 1) if tags else 1
        for i in range(nb):
            mat = bpy.data.materials.new("Sphericon Band %d" % i)
            mat.use_nodes = True
            b = mat.node_tree.nodes.get("Principled BSDF")
            col = _PALETTE[i % len(_PALETTE)]
            b.inputs["Base Color"].default_value = (*col, 1)
            b.inputs["Roughness"].default_value = 0.45
            me.materials.append(mat)
        me.polygons.foreach_set("material_index", tags)

    class MESH_OT_sphericon_add(bpy.types.Operator):
        """Add a sphericon / polysphericon: a developable roller made by
        halving the solid of revolution of a regular polygon and turning
        one half so the cross-section maps onto itself"""
        bl_idname = "mesh.sphericon_add"
        bl_label = "Sphericon"
        bl_options = {'REGISTER', 'UNDO'}

        sides: IntProperty(
            name="Polygon Sides", default=4, min=3, max=24,
            description="n: sides of the regular polygon revolved into "
                        "the roller (4 = classic sphericon, 7 = "
                        "heptagonal sphericon)")
        steps: IntProperty(
            name="Rotation Steps", default=1, min=1, max=23,
            description="k: the second half is turned by k x 360/n "
                        "degrees so the polygon cross-section lands on "
                        "itself (1 = the fundamental sphericon)")
        segments: IntProperty(
            name="Segments", default=96, min=6, max=512,
            description="Angular resolution over each half turn")
        coloring: EnumProperty(
            name="Coloring",
            items=[('BANDS', "Per Conical Band",
                    "A different colour per conical band"),
                   ('NONE', "Single Material", "One plain material")],
            default='BANDS')
        smooth_shading: BoolProperty(name="Smooth Shading",
                                     default=True)
        scale: FloatProperty(
            name="Scale", default=1.0, min=0.01, max=100.0,
            description="Multiplier on the normalized size (1.0 = a "
                        "2 m cube, centered on the origin)")

        def execute(self, context):
            verts, faces, tags = build_sphericon(
                self.sides, self.steps, self.segments)

            # centre on the bounding box and fit a 2 m cube * scale
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            zs = [v[2] for v in verts]
            cx = 0.5 * (min(xs) + max(xs))
            cy = 0.5 * (min(ys) + max(ys))
            cz = 0.5 * (min(zs) + max(zs))
            ext = max(max(xs) - min(xs), max(ys) - min(ys),
                      max(zs) - min(zs)) or 1.0
            s = 2.0 * self.scale / ext
            verts = [((v[0] - cx) * s, (v[1] - cy) * s,
                      (v[2] - cz) * s) for v in verts]

            me = bpy.data.meshes.new("Sphericon")
            me.from_pydata(verts, [], faces)
            me.validate(clean_customdata=False)
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            # crease the band boundaries and the cut seam (large dihedral
            # angle) so smooth shading keeps the cones crisp while the
            # circumference stays round -- the developable-roller look
            if self.smooth_shading:
                thr = math.radians(16.0)
                for e in bm.edges:
                    if len(e.link_faces) == 2:
                        try:
                            if e.calc_face_angle(0.0) > thr:
                                e.smooth = False
                        except ValueError:
                            pass
            bm.to_mesh(me)
            bm.free()

            self._tags = tags
            # tags line up with faces as long as validate/remove_doubles
            # did not drop any; guard the lengths
            if len(tags) != len(me.polygons):
                tags = [0] * len(me.polygons)
            _band_materials(me, tags, self.coloring)

            att = me.attributes.new("band", 'INT', 'FACE')
            att.data.foreach_set(
                "value", tags if len(tags) == len(me.polygons)
                else [0] * len(me.polygons))

            me.polygons.foreach_set(
                'use_smooth', [self.smooth_shading] * len(me.polygons))
            me.update()

            obj = bpy.data.objects.new("Sphericon", me)
            context.scene.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report(
                {'INFO'},
                "(%d,%d) sphericon: V=%d F=%d" %
                (self.sides, self.steps, len(me.vertices),
                 len(me.polygons)))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'sides')
            lay.prop(self, 'steps')
            lay.prop(self, 'segments')
            lay.prop(self, 'coloring')
            lay.prop(self, 'smooth_shading')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.sphericon_add", icon='MESH_CAPSULE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_sphericon_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_sphericon_add)


if __name__ == "__main__":
    if _IN_BLENDER:
        register()
    else:
        from collections import Counter

        def check(n, k, seg=48):
            v, f, t = build_sphericon(n, k, seg)
            cnt = Counter()
            for fc in f:
                L = len(fc)
                for i in range(L):
                    a, b = fc[i], fc[(i + 1) % L]
                    cnt[(min(a, b), max(a, b))] += 1
            border = [e for e, c in cnt.items() if c != 2]
            man = not border
            chi = len(v) - len(cnt) + len(f)
            print("(%d,%d): V=%d E=%d F=%d chi=%d watertight=%s %s"
                  % (n, k, len(v), len(cnt), len(f), chi, man,
                     "OK" if man and chi == 2 else "BAD"))
            return man and chi == 2

        allok = True
        for (n, k) in [(4, 1), (3, 1), (5, 1), (6, 1), (7, 1),
                       (8, 1), (8, 3), (9, 1), (12, 1)]:
            allok = check(n, k) and allok
        print("RESULT:", "ALL OK" if allok else "SOME BAD")
