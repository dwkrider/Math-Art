
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
#
# References:
# - P. Schatz, the oloid (1929); see "Rhythmusforschung und Technik"
#   (1975) for his own account of the invertible cube it came from.
# - H. Dirnboeck and H. Stachel, "The Development of the Oloid",
#   J. Geometry and Graphics 1 (1997) 105-118 -- the exact ruling used
#   here, and the result that every ruling has length sqrt(3).
# - K. Wallace, "ruled Mobius strip" experiments,
#   kitwallace.tumblr.com/post/85762927079 -- the two-circle roller and
#   the one-edged ruled surface.

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
# The mathematics lives in the sibling `rollers` engine package;
# this module is the Blender layer over it.
try:
    from .rollers.developable import (build_antioloid, build_mobius,
                                          build_oloid, build_ruled, cos, pi,
                                          roller_circles, sin, sqrt)
except ImportError:  # flat import outside the package
    from rollers.developable import (build_antioloid, build_mobius,
                                         build_oloid, build_ruled, cos, pi,
                                         roller_circles, sin, sqrt)



















try:
    from .sharp_creases import mark_sharp_by_angle
except ImportError:                     # flat import outside the package
    from sharp_creases import mark_sharp_by_angle

try:
    import bpy
    import bmesh
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
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
            description="Which developable / ruled shape to build",
            items=[('OLOID', "Oloid",
                    "Convex hull of two perpendicular circles "
                    "through each other's centre (exact ruled "
                    "surface, Schatz 1929)"),
                   ('ROLLER', "Two-Circle Roller / Wobbler",
                    "Circle centres sqrt(2) apart: rolls with its "
                    "centre of mass at constant height. Set Disc "
                    "Aspect != 1 for Hirsch's elliptical 'ellipsoloid'"),
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
                              max=512,
                              description="Number of segments around the "
                                          "circles")
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
        aspect: FloatProperty(
            name="Disc Aspect", default=1.0, min=0.25, max=4.0,
            description="Two-circle roller: 1.0 = round discs "
                        "(wobbler); != 1 stretches them to ellipses "
                        "(ellipsoloid)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size multiplier")
        sharp_edges: BoolProperty(
            name="Sharp Circle Edges", default=True,
            description="Mark the two circular arcs sharp (and "
                        "creased). The oloid and the two-circle roller "
                        "are convex hulls, so the developable band "
                        "folds back on itself along each circle -- "
                        "those arcs are real edges and shading smooth "
                        "across them rounds the whole form off. The "
                        "ruled strips have no such fold and are left "
                        "alone")

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
                pts = _fit(roller_circles(self.segments, self.scale,
                                          self.aspect))
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
            # Only the closed convex rollers have folds.  ANTIOLOID,
            # RULED and MOBIUS are smooth ruled strips -- measured, no
            # interior edge of any of them bends past 6 degrees -- and
            # the one 176-degree edge the Mobius does report is a
            # degenerate face at its join, not a crease to mark.
            if self.sharp_edges and self.kind in ('OLOID', 'ROLLER'):
                mark_sharp_by_angle(me, 40.0)
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
            elif self.kind == 'ROLLER':
                lay.prop(self, 'aspect')
            lay.prop(self, 'scale')
            if self.kind in ('OLOID', 'ROLLER'):
                lay.prop(self, 'sharp_edges')

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
