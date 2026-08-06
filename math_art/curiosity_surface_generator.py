
# Miscellaneous Surfaces generator for Blender: three classic
# surfaces from the geometry literature.
#
# 1. Fresnel's elasticity surface (von Seggern 1993, p. 304): the
#    quartic surface r = sqrt(a^2 x'^2 + b^2 y'^2 + c^2 z'^2),
#    where (x', y', z') is the unit direction of the radius
#    vector; in Cartesian form (x^2 + y^2 + z^2)^2 =
#    a^2 x^2 + b^2 y^2 + c^2 z^2.  It is a radial surface over
#    the sphere of directions, so it is meshed as a displaced UV
#    sphere with welded poles and seam (watertight).
#
# 2. The paper bag surface (Robin 2004; plotted after Trott 2004,
#    p. 103): the crimped-bag plot
#        x = v cos u
#        y = (v + b u) sin u
#        z = a v^2
#    for u in [0, 2 pi], v in [0, 2], with the classic constants
#    a = 2.47 (height coefficient) and b = -1.26 (crimp
#    coefficient).  The u = 0 and u = 2 pi boundary curves
#    coincide in space and are welded; the surface stays open at
#    v = 0 and v = 2 like a real bag.
#
# 3. The trihyperboloid (Knill 2017; Villarino and Varilly 2024):
#    the boundary of the solid enclosed by the three hyperboloids
#        x^2 + y^2 - z^2 <= 1
#        y^2 + z^2 - x^2 <= 1
#        z^2 + x^2 - y^2 <= 1
#    (volume ln 256 = 8 ln 2), shaped like a stella octangula
#    with webs hung across adjacent faces.  Along any unit
#    direction (l, m, n) the largest of the three quadratic forms
#    q_max = max(l^2+m^2-n^2, m^2+n^2-l^2, n^2+l^2-m^2) is at
#    least 1/3, so the boundary is the radial graph
#    r = 1/sqrt(q_max) with 1 <= r <= sqrt(3) -- meshed exactly
#    as a displaced UV sphere (watertight).

bl_info = {
    "name": "Miscellaneous Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Fresnel's elasticity surface, the paper bag "
                   "surface, and the trihyperboloid",
    "category": "Add Mesh",
}

import math

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


def build_radial(radius_fn, segments=96, rings=48):
    """(verts, faces): the radial surface r = radius_fn(l, m, n)
    over unit directions (l, m, n), meshed as a displaced UV
    sphere.  Poles and the theta seam are welded, so the result
    is watertight; faces wind outward (positive volume)."""
    segments = max(3, int(segments))
    rings = max(2, int(rings))
    verts = []
    r = radius_fn(0.0, 0.0, 1.0)
    verts.append((0.0, 0.0, r))
    for i in range(1, rings):
        phi = math.pi * i / rings
        sp, cp = math.sin(phi), math.cos(phi)
        for j in range(segments):
            th = 2.0 * math.pi * j / segments
            d = (sp * math.cos(th), sp * math.sin(th), cp)
            r = radius_fn(*d)
            verts.append((r * d[0], r * d[1], r * d[2]))
    r = radius_fn(0.0, 0.0, -1.0)
    verts.append((0.0, 0.0, -r))
    south = len(verts) - 1

    def vid(i, j):
        return 1 + (i - 1) * segments + j % segments

    faces = []
    for j in range(segments):
        faces.append([0, vid(1, j), vid(1, j + 1)])
    for i in range(1, rings - 1):
        for j in range(segments):
            faces.append([vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)])
    for j in range(segments):
        faces.append([vid(rings - 1, j), south,
                      vid(rings - 1, j + 1)])
    return verts, faces


def fresnel_radius(a, b, c):
    """r(l, m, n) = sqrt(a^2 l^2 + b^2 m^2 + c^2 n^2) for a unit
    direction: Fresnel's elasticity surface (von Seggern's
    quartic (x^2+y^2+z^2)^2 = a^2 x^2 + b^2 y^2 + c^2 z^2)."""
    def r(l, m, n):
        return math.sqrt(a * a * l * l + b * b * m * m
                         + c * c * n * n)
    return r


def build_fresnel(a=1.0, b=1.5, c=2.0, segments=96, rings=48):
    """(verts, faces): Fresnel's elasticity surface with
    semi-axes a, b, c, watertight."""
    return build_radial(fresnel_radius(a, b, c), segments, rings)


def trihyperboloid_radius(l, m, n):
    """r(l, m, n) = 1/sqrt(max of the three hyperboloid forms):
    the boundary of the solid x^2+y^2-z^2 <= 1,
    y^2+z^2-x^2 <= 1, z^2+x^2-y^2 <= 1 along a unit direction.
    The maximum is >= 1/3, so r is finite (1 <= r <= sqrt 3)."""
    l2, m2, n2 = l * l, m * m, n * n
    q = max(l2 + m2 - n2, m2 + n2 - l2, n2 + l2 - m2)
    return 1.0 / math.sqrt(q)


def build_trihyperboloid(segments=96, rings=48):
    """(verts, faces): the trihyperboloid boundary surface,
    watertight."""
    return build_radial(trihyperboloid_radius, segments, rings)


def build_paper_bag(a=2.47, b=-1.26, segments=96, rings=48,
                    depth=2.0):
    """(verts, faces): the paper bag surface x = v cos u,
    y = (v + b u) sin u, z = a v^2 for u in [0, 2 pi] and
    v in [0, depth].  The u seam (where both boundary curves
    coincide) is welded; the v boundaries stay open."""
    segments = max(3, int(segments))
    rings = max(1, int(rings))
    verts = []
    for i in range(rings + 1):
        v = depth * i / rings
        for j in range(segments):
            u = 2.0 * math.pi * j / segments
            verts.append((v * math.cos(u),
                          (v + b * u) * math.sin(u),
                          a * v * v))
    faces = []
    for i in range(rings):
        for j in range(segments):
            j2 = (j + 1) % segments
            faces.append([i * segments + j, i * segments + j2,
                          (i + 1) * segments + j2,
                          (i + 1) * segments + j])
    return verts, faces


if _IN_BLENDER:

    class MESH_OT_curiosity_surface_add(bpy.types.Operator):
        """Add a classic miscellaneous surface: Fresnel's
        elasticity surface, the paper bag surface, or the
        trihyperboloid"""
        bl_idname = "mesh.curiosity_surface_add"
        bl_label = "Miscellaneous Surface"
        bl_options = {'REGISTER', 'UNDO'}

        surface: EnumProperty(
            name="Surface",
            items=[('FRESNEL', "Fresnel Elasticity Surface",
                    "The quartic (x^2+y^2+z^2)^2 = "
                    "a^2 x^2 + b^2 y^2 + c^2 z^2"),
                   ('PAPERBAG', "Paper Bag Surface",
                    "The crimped inflated-bag surface "
                    "x = v cos u, y = (v + b u) sin u, "
                    "z = a v^2"),
                   ('TRIHYPERBOLOID', "Trihyperboloid",
                    "Boundary of the solid enclosed by the "
                    "three hyperboloids x^2+y^2-z^2 <= 1 "
                    "(cyclic); volume 8 ln 2")],
            default='FRESNEL')
        semi_a: FloatProperty(
            name="Semi-Axis A", default=1.0, min=0.01, max=10.0,
            description="Fresnel semi-axis along X")
        semi_b: FloatProperty(
            name="Semi-Axis B", default=1.5, min=0.01, max=10.0,
            description="Fresnel semi-axis along Y")
        semi_c: FloatProperty(
            name="Semi-Axis C", default=2.0, min=0.01, max=10.0,
            description="Fresnel semi-axis along Z")
        bag_a: FloatProperty(
            name="Height Coefficient", default=2.47,
            min=0.01, max=10.0,
            description="Coefficient a in z = a v^2 (2.47 in "
                        "the classic plot)")
        bag_b: FloatProperty(
            name="Crimp Coefficient", default=-1.26,
            min=-10.0, max=10.0,
            description="Coefficient b in y = (v + b u) sin u "
                        "(-1.26 in the classic plot)")
        resolution: IntProperty(
            name="Resolution", default=48, min=6, max=256,
            description="Rings across the surface (twice as "
                        "many segments around)")
        smooth: BoolProperty(name="Smooth Shading", default=True)
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=1.0,
            description="Solidify modifier thickness (0 = raw "
                        "surface)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            res = self.resolution
            if self.surface == 'FRESNEL':
                verts, faces = build_fresnel(
                    self.semi_a, self.semi_b, self.semi_c,
                    2 * res, res)
                name = "Fresnel Elasticity Surface"
            elif self.surface == 'PAPERBAG':
                verts, faces = build_paper_bag(
                    self.bag_a, self.bag_b, 2 * res, res)
                name = "Paper Bag Surface"
            else:
                verts, faces = build_trihyperboloid(2 * res, res)
                name = "Trihyperboloid"
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
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'surface')
            if self.surface == 'FRESNEL':
                for k in ('semi_a', 'semi_b', 'semi_c'):
                    lay.prop(self, k)
            elif self.surface == 'PAPERBAG':
                for k in ('bag_a', 'bag_b'):
                    lay.prop(self, k)
            for k in ('resolution', 'smooth', 'thickness',
                      'scale'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator("mesh.curiosity_surface_add",
                             icon='MESH_UVSPHERE')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(MESH_OT_curiosity_surface_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_curiosity_surface_add)


def _selftest():
    def _finite(verts):
        return all(all(math.isfinite(c) for c in v)
                   for v in verts)

    def _valid(verts, faces):
        n = len(verts)
        return all(0 <= i < n for f in faces for i in f)

    def _watertight(faces):
        cnt = {}
        for f in faces:
            for i in range(len(f)):
                e = frozenset((f[i], f[(i + 1) % len(f)]))
                cnt[e] = cnt.get(e, 0) + 1
        return all(c == 2 for c in cnt.values())

    def _volume(verts, faces):
        vol = 0.0
        for f in faces:
            for i in range(1, len(f) - 1):
                (ax, ay, az) = verts[f[0]]
                (bx, by, bz) = verts[f[i]]
                (cx, cy, cz) = verts[f[i + 1]]
                vol += (ax * (by * cz - bz * cy)
                        - ay * (bx * cz - bz * cx)
                        + az * (bx * cy - by * cx)) / 6.0
        return vol

    # ---- Fresnel's elasticity surface -------------------
    a, b, c = 1.0, 1.5, 2.0
    segs, rngs = 32, 16
    verts, faces = build_fresnel(a, b, c, segs, rngs)
    assert _finite(verts) and _valid(verts, faces)
    assert _watertight(faces)
    # every vertex obeys r = sqrt(a^2 l^2 + b^2 m^2
    # + c^2 n^2) for its unit direction (l, m, n)
    for (x, y, z) in verts:
        r = math.sqrt(x * x + y * y + z * z)
        l, m, n = x / r, y / r, z / r
        want = math.sqrt(a * a * l * l + b * b * m * m
                         + c * c * n * n)
        assert abs(r - want) < 1e-9
    # sample directions: the axes hit r = a, b, c
    eq = 1 + (rngs // 2 - 1) * segs      # phi = pi/2, th = 0
    px = verts[eq]
    py = verts[eq + segs // 4]           # theta = pi/2
    pz = verts[0]                        # north pole
    assert abs(px[0] - a) < 1e-9 and abs(px[1]) < 1e-9
    assert abs(py[1] - b) < 1e-9 and abs(abs(py[0])) < 1e-9
    assert abs(pz[2] - c) < 1e-9
    vol = _volume(verts, faces)
    print(f"fresnel: V={len(verts)} F={len(faces)} "
          f"watertight=True axes=({a},{b},{c}) "
          f"vol={vol:.3f}")
    assert vol > 0.0

    # ---- Paper bag surface ------------------------------
    pa, pb, depth = 2.47, -1.26, 2.0
    segs, rngs = 48, 24
    verts, faces = build_paper_bag(pa, pb, segs, rngs)
    assert _finite(verts) and _valid(verts, faces)
    # spot-check the parametrization at a few grid points
    for (i, j) in ((0, 0), (rngs, 0), (7, 11), (rngs, 30)):
        v = depth * i / rngs
        u = 2.0 * math.pi * j / segs
        x, y, z = verts[i * segs + j]
        assert abs(x - v * math.cos(u)) < 1e-9
        assert abs(y - (v + pb * u) * math.sin(u)) < 1e-9
        assert abs(z - pa * v * v) < 1e-9
    print(f"paper bag: V={len(verts)} F={len(faces)} "
          f"open surface, formula verified")

    # ---- Trihyperboloid ---------------------------------
    segs, rngs = 96, 48
    verts, faces = build_trihyperboloid(segs, rngs)
    assert _finite(verts) and _valid(verts, faces)
    assert _watertight(faces)
    # every vertex lies on the boundary: the largest of the
    # three hyperboloid forms equals 1, and 1 <= r <= sqrt 3
    for (x, y, z) in verts:
        x2, y2, z2 = x * x, y * y, z * z
        q = max(x2 + y2 - z2, y2 + z2 - x2, z2 + x2 - y2)
        assert abs(q - 1.0) < 1e-9
        r = math.sqrt(x2 + y2 + z2)
        assert 1.0 - 1e-9 <= r <= math.sqrt(3.0) + 1e-9
    vol = _volume(verts, faces)
    want = 8.0 * math.log(2.0)     # ln 256, the exact volume
    print(f"trihyperboloid: V={len(verts)} F={len(faces)} "
          f"watertight=True vol={vol:.4f} "
          f"(exact 8 ln 2 = {want:.4f})")
    assert abs(vol - want) < 0.15

    print("miscellaneous surfaces standalone tests passed")
