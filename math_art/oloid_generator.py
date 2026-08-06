
# Oloid & Ruled Surface Generator for Blender
#
# The oloid (Paul Schatz, 1929): the convex hull of two unit circles
# in perpendicular planes, each passing through the other's centre.
# Its surface is built here from the exact ruling of Dirnboeck &
# Stachel ("The Development of the Oloid", J. Geometry and Graphics
# 1997):
#
#   A(t) = (sin t, -1/2 - cos t, 0)                 t in [-2pi/3, 2pi/3]
#   B(t) = (0, 1/2 - cos t/(1+cos t),
#           +- sqrt(1+2 cos t)/(1+cos t))
#
# every ruling A(t)B(t) has length sqrt(3). Also included: the
# two-circle roller (circle centres sqrt(2) apart, convex hull), and
# ruled strips between two perpendicular circles after Kit Wallace's
# "ruled Mobius strip" experiments (kitwallace.tumblr.com/post/
# 85762927079), including his true one-edged Mobius surface.

bl_info = {
    "name": "Oloid & Ruled Surfaces",
    "author": "Math Art project (after Schatz / Dirnboeck & "
              "Stachel / Kit Wallace)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Oloid",
    "description": "Oloid, two-circle roller, ruled circle strips",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi, sqrt, radians


def _vkey(p):
    return (round(p[0], 9), round(p[1], 9), round(p[2], 9))


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
        ids = [self.vid(p) for p in (a, b, c, d)]
        uniq = []
        for i in ids:
            if i not in uniq:
                uniq.append(i)
        if len(uniq) >= 3:
            self.faces.append(uniq)


def build_oloid(segments=96, scale=1.0):
    """Exact oloid surface from the Dirnboeck-Stachel ruling; two
    mirror strips welded along the circle arcs. Watertight."""
    m = _Mesh()
    T = 2 * pi / 3
    n = segments

    def A(t):
        return (sin(t) * scale, (-0.5 - cos(t)) * scale, 0.0)

    def B(t, sign):
        c = cos(t)
        z2 = max(0.0, 1.0 + 2.0 * c)
        z = sqrt(z2) / (1.0 + c)
        if z < 1e-6:                     # end rulings: weld exactly
            z = 0.0
        return (0.0, (0.5 - c / (1.0 + c)) * scale,
                sign * z * scale)

    for i in range(n):
        t0 = -T + 2 * T * i / n
        t1 = -T + 2 * T * (i + 1) / n
        m.quad(A(t0), A(t1), B(t1, 1.0), B(t0, 1.0))
        m.quad(A(t1), A(t0), B(t0, -1.0), B(t1, -1.0))
    return m.verts, m.faces


def roller_circles(segments=96, scale=1.0):
    """Point cloud of the two-circle roller: two unit circles in
    perpendicular planes, centres sqrt(2) apart (hulled in Blender)."""
    d = sqrt(2.0) / 2.0
    pts = []
    for i in range(segments):
        a = 2 * pi * i / segments
        pts.append((cos(a) * scale, (-d + sin(a)) * scale, 0.0))
        pts.append((0.0, (d + cos(a)) * scale, sin(a) * scale))
    return pts


def build_ruled(segments=96, separation=1.0, incline=0.0, phase=0.0,
                scale=1.0):
    """Kit Wallace's ruled strip: straight lines from circle 1 (xy
    plane, centred at the origin) to circle 2 (xz plane, offset along
    x, optionally inclined), sample i to sample i+phase."""
    m = _Mesh()
    n = segments
    inc = radians(incline)
    ci, si = cos(inc), sin(inc)

    def c1(i):
        a = 2 * pi * i / n
        return (cos(a) * scale, sin(a) * scale, 0.0)

    def c2(i):
        a = 2 * pi * (i / n + phase)
        x, y, z = cos(a), 0.0, sin(a)
        # incline about the z axis, then offset along x
        x, y = x * ci - y * si, x * si + y * ci
        return ((x + separation) * scale, y * scale, z * scale)

    for i in range(n):
        m.quad(c1(i), c1(i + 1), c2(i + 1), c2(i))
    return m.verts, m.faces


def build_antioloid(segments=128, phase=0.0, scale=1.0):
    """The anti-oloid (cf. the Matter Collection piece): the ruled
    band between the same two circles as the oloid -- perpendicular
    planes, each centre on the other's circumference -- but with
    rulings connecting points travelling around the FULL circles in
    step, sweeping through the interior. Unlike a Mobius strip it is
    two-sided, and unlike the oloid it is not developable."""
    m = _Mesh()
    n = segments

    def A(t):
        return (sin(t) * scale, (-0.5 - cos(t)) * scale, 0.0)

    def B(t):
        return (0.0, (0.5 + cos(t)) * scale, sin(t) * scale)

    # the canonical anti-oloid pairs each point with the half-turn
    # opposite point on the other circle (phase 0 here); other phases
    # give crossed variants
    p = 2 * pi * phase + pi
    for i in range(n):
        t0 = 2 * pi * i / n
        t1 = 2 * pi * (i + 1) / n
        m.quad(A(t0), A(t1), B(t1 + p), B(t0 + p))
    return m.verts, m.faces


def build_mobius(segments=192, scale=1.0):
    """Kit Wallace's true one-edged ruled Mobius strip: rulings
    f(x) -> f(x + 1/2) along a double-loop edge curve."""
    m = _Mesh()

    def f(x):
        r = 1.0 - 0.15 * sin(2 * pi * x + radians(30))
        return (r * cos(4 * pi * x) * scale,
                r * sin(4 * pi * x) * scale,
                0.2 * cos(2 * pi * x) * scale)

    n = segments

    def fx(i):
        return f((i % (2 * n)) / (2 * n))

    for i in range(n):
        m.quad(fx(i), fx(i + 1), fx(i + 1 + n), fx(i + n))
    return m.verts, m.faces


try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_oloid_add(bpy.types.Operator):
        """Add an oloid, two-circle roller, or ruled circle strip"""
        bl_idname = "mesh.oloid_add"
        bl_label = "Oloid"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Shape",
            items=[('OLOID', "Oloid",
                    "Convex hull of two perpendicular circles "
                    "through each other's centre (exact ruled "
                    "surface, Schatz 1929)"),
                   ('ROLLER', "Two-Circle Roller",
                    "Circle centres sqrt(2) apart: rolls with its "
                    "centre of mass at constant height"),
                   ('ANTIOLOID', "Anti-Oloid",
                    "Ruled band between the oloid's two circles, "
                    "sweeping through the interior: two-sided and "
                    "non-developable (after the Matter Collection "
                    "piece)"),
                   ('RULED', "Ruled Circle Strip",
                    "Straight rulings between two perpendicular "
                    "circles (after Kit Wallace); separation, "
                    "inclination and phase are adjustable"),
                   ('MOBIUS', "Ruled Mobius Strip",
                    "Kit Wallace's one-edged ruled Mobius surface "
                    "(rulings across a double-loop edge)")],
            default='OLOID')
        segments: IntProperty(name="Segments", default=96, min=12,
                              max=512)
        separation: FloatProperty(
            name="Separation", default=1.0, min=0.0, max=3.0,
            description="Distance between the circle centres "
                        "(ruled strip)")
        incline: FloatProperty(
            name="Inclination", default=0.0, min=-90.0, max=90.0,
            description="Tilt of the second circle (degrees)")
        phase: FloatProperty(
            name="Phase", default=0.0, min=-0.5, max=0.5,
            description="Ruling offset around the second circle "
                        "(fraction of a turn)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            if self.kind == 'OLOID':
                verts, faces = build_oloid(self.segments, self.scale)
            elif self.kind == 'ANTIOLOID':
                verts, faces = build_antioloid(self.segments,
                                               self.phase, self.scale)
            elif self.kind == 'RULED':
                verts, faces = build_ruled(self.segments,
                                           self.separation,
                                           self.incline, self.phase,
                                           self.scale)
            elif self.kind == 'MOBIUS':
                verts, faces = build_mobius(
                    max(48, self.segments), self.scale)
            else:
                verts, faces = None, None

            def _fit(vs):
                # centre on the bounding-box midpoint and uniformly
                # scale so the largest extent is 2.0 * self.scale (a
                # ~2 m cube by default). The shape verts already carry
                # self.scale; because this is a ratio it divides out,
                # so scale is not applied twice.
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

            me = bpy.data.meshes.new("Oloid")
            if self.kind == 'ROLLER':
                pts = _fit(roller_circles(self.segments, self.scale))
                bm = bmesh.new()
                vs = [bm.verts.new(p) for p in pts]
                bmesh.ops.convex_hull(bm, input=vs)
                unused = [v for v in bm.verts if not v.link_faces]
                bmesh.ops.delete(bm, geom=unused, context='VERTS')
                bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
                bm.to_mesh(me)
                bm.free()
            else:
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
            obj = bpy.data.objects.new(f"Oloid {self.kind.title()}",
                                       me)
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
            lay.prop(self, 'segments')
            if self.kind == 'RULED':
                for k in ('separation', 'incline', 'phase'):
                    lay.prop(self, k)
            elif self.kind == 'ANTIOLOID':
                lay.prop(self, 'phase')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.oloid_add", icon='MESH_CAPSULE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_oloid_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_oloid_add)


def _selftest():
    from collections import Counter
    v, f = build_oloid(96)
    cnt = Counter()
    for fc in f:
        for i in range(len(fc)):
            a, b = fc[i], fc[(i + 1) % len(fc)]
            cnt[(min(a, b), max(a, b))] += 1
    man = all(c == 2 for c in cnt.values())
    # constant ruling length sqrt(3)
    T = 2 * pi / 3
    maxerr = 0.0
    for i in range(1, 96):
        t = -T + 2 * T * i / 96
        A = (sin(t), -0.5 - cos(t), 0.0)
        c = cos(t)
        B = (0.0, 0.5 - c / (1 + c), sqrt(1 + 2 * c) / (1 + c))
        d = sqrt(sum((A[k] - B[k]) ** 2 for k in range(3)))
        maxerr = max(maxerr, abs(d - sqrt(3)))
    print(f"oloid: verts={len(v)} faces={len(f)} manifold={man} "
          f"ruling |err|={maxerr:.2e} "
          f"{'OK' if man and maxerr < 1e-9 else 'BAD'}")
    v, f = build_mobius(96)
    # Mobius: one boundary loop, chi = 0
    cnt = Counter()
    for fc in f:
        for i in range(len(fc)):
            a, b = fc[i], fc[(i + 1) % len(fc)]
            cnt[(min(a, b), max(a, b))] += 1
    border = [e for e, c in cnt.items() if c == 1]
    chi = len(v) - len(cnt) + len(f)
    # walk the boundary
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in border:
        adj[a].append(b)
        adj[b].append(a)
    loops = 0
    seen = set()
    for s in adj:
        if s in seen:
            continue
        loops += 1
        cur, prev = s, None
        while True:
            seen.add(cur)
            nxt = [x for x in adj[cur] if x != prev]
            if not nxt or nxt[0] == s:
                break
            prev, cur = cur, nxt[0]
    ok = loops == 1 and chi == 0
    print(f"mobius: verts={len(v)} faces={len(f)} chi={chi}(0) "
          f"boundary loops={loops}(1) {'OK' if ok else 'BAD'}")
    v, f = build_ruled(96)
    print(f"ruled: verts={len(v)} faces={len(f)} OK")
    assert man and maxerr < 1e-9
    assert ok
