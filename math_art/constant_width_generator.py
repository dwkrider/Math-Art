
# Surfaces of Constant Width for Blender
#
# A convex body has "constant width" w if every pair of parallel
# supporting planes is exactly w apart -- it rolls like a sphere under
# a flat board, keeping the board at constant height, yet (except for
# the sphere) it is not round.  This module builds the three classic
# 3-D examples:
#
#   1. Reuleaux SOLIDS OF REVOLUTION.  A Reuleaux polygon (an odd
#      regular n-gon whose sides are replaced by circular arcs of
#      radius = width, each centred on the opposite vertex) is itself a
#      plane curve of constant width.  Revolving it about one of its
#      axes of symmetry sweeps a solid of constant width.  The n = 3
#      case (revolved Reuleaux triangle) has the least volume of any
#      known constant-width surface of revolution.
#
#   2. The REULEAUX TETRAHEDRON: the intersection of four balls of
#      radius w centred at the vertices of a regular tetrahedron of
#      edge w.  Famously this is NOT quite constant width -- its six
#      curved edges bulge, so its width ranges up to w(sqrt3 - 1)/...
#      ~ 1.025 w.  Included as the starting point (and cautionary tale).
#
#   3. The two MEISSNER TETRAHEDRA.  Meissner (1911) fixed the
#      Reuleaux tetrahedron by shaving three of its six edges, each
#      replaced by a patch of the surface of revolution of a circular
#      arc of radius w about the line joining that edge's two end
#      vertices.  Rounding three edges that meet at a vertex gives the
#      body "M_V"; three edges forming a triangle give "M_E".  Both are
#      true constant-width solids and are conjectured (Bonnesen-Fenchel)
#      to minimise volume among all width-w bodies.
#
# The Reuleaux solids of revolution are meshed directly by revolving a
# Reuleaux polygon.  The Reuleaux and Meissner tetrahedra are built by
# Boolean geometry in Blender (intersecting four balls, then paring
# three edges with a wedge and filling them with a spindle -- a
# surface of revolution of a circular arc -- after the OpenSCAD
# construction of ceptimus, 2018): Booleans mesh the sphere patches and
# their rounded joins cleanly, whereas a single lat-long radial mesh
# tears along the patch-boundary seams.  (A pure-Python radial builder
# is retained for the headless self-test, which checks the constant-
# width property numerically.)
#
# References:
#   - Ernst Meissner & Friedrich Schilling, "Drei Gipsmodelle von
#     Flaechen konstanter Breite", Z. Math. Phys. 60 (1912), 92-94.
#   - Bernd Kawohl & Christof Weber, "Meissner's Mysterious Bodies",
#     Math. Intelligencer 33 (2011), no. 3, 94-101.
#   - T. Bonnesen & W. Fenchel, "Theorie der konvexen Koerper",
#     Springer 1934 (the minimal-volume conjecture).
#   - Wikipedia, "Surface of constant width" and "Reuleaux triangle".

bl_info = {
    "name": "Surfaces of Constant Width",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Constant Width",
    "description": "Reuleaux solids of revolution, the Reuleaux "
                   "tetrahedron, and the two Meissner tetrahedra (after "
                   "Meissner & Schilling)",
    "category": "Add Mesh",
}
# The mathematics lives in the sibling `rollers` engine package;
# this module is the Blender layer over it.
try:
    from .rollers.width import (_dot, build_reuleaux_revolution,
                                    build_tetra_body, cos, pi, sin, sqrt)
except ImportError:  # flat import outside the package
    from rollers.width import (_dot, build_reuleaux_revolution,
                                   build_tetra_body, cos, pi, sin, sqrt)































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

    import mathutils
    from math import atan

    # Standard regular-tetrahedron vertices (edge sqrt(8/3)); the tetra
    # bodies are built by Boolean geometry, which meshes the sphere
    # patches and their rounded joins cleanly -- a radial (lat-long)
    # sampling instead tears along the patch-boundary seams.
    _TETRA_STD = [(sqrt(8.0 / 9.0), 0.0, -1.0 / 3.0),
                  (-sqrt(2.0 / 9.0), sqrt(2.0 / 3.0), -1.0 / 3.0),
                  (-sqrt(2.0 / 9.0), -sqrt(2.0 / 3.0), -1.0 / 3.0),
                  (0.0, 0.0, 1.0)]

    def _cw_sphere(coll, center, r, subdiv):
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=r)
        bmesh.ops.translate(bm, verts=bm.verts, vec=center)
        me = bpy.data.meshes.new("cw_tmp")
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new("cw_tmp", me)
        coll.objects.link(o)
        return o

    def _cw_boolean(ctx, a, b, op):
        md = a.modifiers.new("b", 'BOOLEAN')
        md.operation = op
        md.object = b
        md.solver = 'EXACT'
        with ctx.temp_override(object=a, active_object=a,
                               selected_objects=[a]):
            bpy.ops.object.modifier_apply(modifier=md.name)
        bpy.data.objects.remove(b, do_unlink=True)
        return a

    def _cw_spindle(coll, w, nz=64, nth=72):
        # revolve the Meissner arc rho_s(z) = sqrt(w^2 - z^2) - w*sqrt3/2
        # about z (a thin lens with tips at z = -+ w/2), shifted to
        # z in [0, w] so the tips sit on two tetra vertices.
        c3 = w * sqrt(3.0) / 2.0
        bm = bmesh.new()
        rings = []
        for iz in range(nz + 1):
            z = -0.5 * w + w * iz / nz
            rho = max(sqrt(max(w * w - z * z, 0.0)) - c3, 0.0)
            rings.append([bm.verts.new((rho * cos(2 * pi * j / nth),
                                        rho * sin(2 * pi * j / nth),
                                        z + 0.5 * w)) for j in range(nth)])
        for iz in range(nz):
            for j in range(nth):
                jn = (j + 1) % nth
                try:
                    bm.faces.new([rings[iz][j], rings[iz][jn],
                                  rings[iz + 1][jn], rings[iz + 1][j]])
                except ValueError:
                    pass
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new("cw_tmp")
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new("cw_tmp", me)
        coll.objects.link(o)
        return o

    def _cw_wedge(coll, w, alpha):
        # dihedral wedge (triangular prism) that pares the sharp edge
        # back to the spindle; angle 2*alpha, apex on the edge line.
        ca, sa, R, H = cos(alpha), sin(alpha), w / 5.0, 1.2 * w
        pts = [(0, 0, 0), (R * ca, -R * sa, 0), (R * ca, R * sa, 0),
               (0, 0, H), (R * ca, -R * sa, H), (R * ca, R * sa, H)]
        faces = [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1),
                 (1, 4, 5, 2), (2, 5, 3, 0)]
        me = bpy.data.meshes.new("cw_tmp")
        me.from_pydata(pts, [], faces)
        me.update()
        o = bpy.data.objects.new("cw_tmp", me)
        coll.objects.link(o)
        return o

    def _cw_place(o, v_apex, alpha, angle):
        rz = mathutils.Matrix.Rotation(angle, 4, 'Z')
        ry = mathutils.Matrix.Rotation(-alpha, 4, 'Y')
        t = mathutils.Matrix.Translation(mathutils.Vector(v_apex))
        o.data.transform(rz @ t @ ry)
        return o

    def _build_tetra_csg(ctx, kind, width, subdiv):
        """Reuleaux / Meissner tetra by Boolean geometry (after the
        OpenSCAD construction of ceptimus, 2018).  Returns verts, faces
        of a cleaned, watertight mesh."""
        coll = ctx.collection
        k = width / sqrt(8.0 / 3.0)
        vk = [(x * k, y * k, z * k) for (x, y, z) in _TETRA_STD]
        alpha = atan(2.0 * sqrt(2.0)) / 2.0

        res = _cw_sphere(coll, vk[0], width, subdiv)
        for i in (1, 2, 3):
            res = _cw_boolean(ctx, res, _cw_sphere(coll, vk[i], width,
                                                   subdiv), 'INTERSECT')
        if kind == 'MEISSNER':
            for ang in (0.0, 2.0 * pi / 3.0, 4.0 * pi / 3.0):
                res = _cw_boolean(ctx, res,
                                  _cw_place(_cw_wedge(coll, width, alpha),
                                            vk[0], alpha, ang), 'DIFFERENCE')
            for ang in (0.0, 2.0 * pi / 3.0, 4.0 * pi / 3.0):
                res = _cw_boolean(ctx, res,
                                  _cw_place(_cw_spindle(coll, width),
                                            vk[0], alpha, ang), 'UNION')

        bm = bmesh.new()
        bm.from_mesh(res.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5 * width)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(res.data)
        bm.free()
        verts = [v.co[:] for v in res.data.vertices]
        faces = [list(p.vertices) for p in res.data.polygons]
        bpy.data.objects.remove(res, do_unlink=True)
        return verts, faces

    class MESH_OT_constant_width_add(bpy.types.Operator):
        """Add a surface of constant width -- a Reuleaux solid of """ \
            """revolution, the Reuleaux tetrahedron, or a Meissner """ \
            """tetrahedron"""
        bl_idname = "mesh.constant_width_add"
        bl_label = "Constant Width Solid"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Shape",
            items=[('MEISSNER', "Meissner Tetrahedron",
                    "Reuleaux tetrahedron with three edges (meeting at "
                    "one vertex) rounded -- a true constant-width solid"),
                   ('REVOLUTION', "Reuleaux Solid of Revolution",
                    "A Reuleaux polygon revolved about a symmetry axis "
                    "(the triangle case has least known volume)"),
                   ('REULEAUX', "Reuleaux Tetrahedron",
                    "Intersection of four balls; NOT quite constant "
                    "width (its edges bulge) -- the starting point")],
            default='MEISSNER')
        poly_n: IntProperty(
            name="Polygon Sides", default=3, min=3, max=15,
            description="Sides of the Reuleaux polygon (forced odd)")
        width: FloatProperty(
            name="Width", default=2.0, min=0.05, max=100.0,
            description="The constant width (rolling height)")
        sphere_subdiv: IntProperty(
            name="Smoothness", default=6, min=4, max=8,
            description="Icosphere subdivisions for the Boolean tetra "
                        "(higher = smoother, slower; 8 is very heavy)")
        sharp_edges: BoolProperty(
            name="Sharp Edges", default=True,
            description="Mark the solid's fold curves sharp (and "
                        "creased). A Reuleaux rotation or Meissner body carries the sharp edges of the polygon it came from. The surface is smooth "
                        "everywhere else, so shading straight across "
                        "the fold rounds off the one feature that "
                        "defines the shape")
        theta_segments: IntProperty(
            name="Segments", default=160, min=12, max=512,
            description="Revolution: segments around the axis")

        def execute(self, context):
            if self.kind == 'REVOLUTION':
                verts, faces = build_reuleaux_revolution(
                    self.poly_n, self.width, self.theta_segments)
            else:
                verts, faces = _build_tetra_csg(
                    context, self.kind, self.width, self.sphere_subdiv)

            def _fit(vs):
                # centre on the bbox midpoint; the shapes already carry
                # the requested width, so no rescale is applied here.
                if not vs:
                    return vs
                xs = [v[0] for v in vs]
                ys = [v[1] for v in vs]
                zs = [v[2] for v in vs]
                cx = 0.5 * (min(xs) + max(xs))
                cy = 0.5 * (min(ys) + max(ys))
                cz = 0.5 * (min(zs) + max(zs))
                return [(v[0] - cx, v[1] - cy, v[2] - cz) for v in vs]

            me = bpy.data.meshes.new("Constant Width")
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
                mark_sharp_by_angle(me, 25.0)
            me.update()
            obj = bpy.data.objects.new("Constant Width", me)
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
            if self.kind == 'REVOLUTION':
                lay.prop(self, 'poly_n')
            lay.prop(self, 'width')
            if self.kind == 'REVOLUTION':
                lay.prop(self, 'theta_segments')
            else:
                lay.prop(self, 'sphere_subdiv')

    def _menu_func(self, context):
        self.layout.operator("mesh.constant_width_add", icon='MESH_CIRCLE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_constant_width_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_constant_width_add)


def _width_stats(verts, ndir=48):
    """Sampled width range of a point cloud (support-function estimate;
    always slightly under-estimates the true width)."""
    lo = 9e9
    hi = -9e9
    for a in range(ndir):
        for b in range(ndir // 2 + 1):
            th = 2 * pi * a / ndir
            ph = pi * b / (ndir // 2)
            d = (sin(ph) * cos(th), sin(ph) * sin(th), cos(ph))
            proj = [_dot(v, d) for v in verts]
            wdt = max(proj) - min(proj)
            lo = min(lo, wdt)
            hi = max(hi, wdt)
    return lo, hi


def _manifold(faces):
    from collections import Counter
    cnt = Counter()
    for f in faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            cnt[(min(a, b), max(a, b))] += 1
    return all(c == 2 for c in cnt.values()), len(cnt)


def _selftest():
    W = 2.0
    # 1. Reuleaux solids of revolution: watertight + constant width == W.
    for n in (3, 5, 7):
        v, f = build_reuleaux_revolution(n, W, 96, 40)
        man, ne = _manifold(f)
        chi = len(v) - ne + len(f)
        lo, hi = _width_stats(v, 40)
        ok = man and chi == 2 and abs(hi - W) < 0.02 and (hi - lo) < 0.03 * W
        print(f"reuleaux_rev n={n}: V={len(v)} F={len(f)} manifold={man} "
              f"chi={chi}(2) width=[{lo:.3f},{hi:.3f}]~{W} "
              f"{'OK' if ok else 'BAD'}")
        assert ok

    # 2. Reuleaux tetra: watertight, but width EXCEEDS W (the point).
    v, f = build_tetra_body('REULEAUX', W, 64, 96)
    man, ne = _manifold(f)
    chi = len(v) - ne + len(f)
    lo, hi = _width_stats(v, 44)
    over = hi > W * 1.01
    print(f"reuleaux_tetra: V={len(v)} F={len(f)} manifold={man} chi={chi}(2) "
          f"width=[{lo:.3f},{hi:.3f}] over_width={over} "
          f"{'OK' if (man and chi == 2 and over) else 'BAD'}")
    assert man and chi == 2 and over

    # 3. Meissner bodies: watertight AND the excess width is removed
    # (max width drops back to ~W; a true constant-width solid).
    for kind in ('MEISSNER_V', 'MEISSNER_E'):
        v, f = build_tetra_body(kind, W, 72, 120)
        man, ne = _manifold(f)
        chi = len(v) - ne + len(f)
        lo, hi = _width_stats(v, 44)
        ok = man and chi == 2 and hi < W * 1.004
        print(f"{kind.lower()}: V={len(v)} F={len(f)} manifold={man} "
              f"chi={chi}(2) width=[{lo:.3f},{hi:.3f}] max<=W "
              f"{'OK' if ok else 'BAD'}")
        assert ok
