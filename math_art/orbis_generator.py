
# Orbis / Holey Roller Generator for Blender
#
# Two torus rings JOINED at a hinge into a single rigid two-holed solid.
# Peer Clahsen designed the "Orbis" -- two wooden rings fused where they
# meet, their planes folded to an angle of about 45 degrees -- as one of
# his "Objeux" play objects for the Swiss Kurt Naef toy company.
# Kenneth Brecher and the wood craftsman Randy Rhine later built a
# large sculptural version, the "Holey Roller": "two tori made of
# multiple wood segments joined at an angle of about 45 degrees".  The
# object rolls on the outer rims of its two rings with a wobbling,
# meandering path.
#
# Construction: the two ring planes share a hinge line and are folded
# to a dihedral `angle` (default 45 degrees) about it, like a partly
# opened book.  The rings are joined at the SURFACE, not the
# centreline: each ring circle (radius R, tube radius r) is placed a
# distance R + g from the hinge line in its own plane, with
#     g = (2 r - delta) / (2 sin(angle / 2)),
# so the closest approach of the two centre circles -- at the mirrored
# inner-equator points facing each other across the wedge -- is
# 2 r - delta.  The tube surfaces therefore touch and overlap by only
# a shallow depth delta (`overlap` fraction of r) at ONE small contact
# region; the centre circles themselves stay far apart and never
# thread through each other.  A Boolean union of the two solid tori
# (EXACT solver, as in steinmetz_generator) welds them at that contact
# patch into ONE connected, genus-two solid -- two rings glued
# edge-to-edge, not a linked chain.
#
# References:
#   - P. Clahsen, "Objeux", Naef Herstellung und Verlag, 1967/1976
#     (the two-ring "Orbis" play object).
#   - K. Brecher, "Rolloids", Bridges 2023 Conference Proceedings,
#     345-352 (the "Orbis" and the "Holey Roller"; local copy at
#     research/bridges/2023/bridges2023-345/, Figure 11).

bl_info = {
    "name": "Orbis / Holey Roller",
    "author": "Math Art project",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Orbis / Holey Roller",
    "description": "Two torus rings hinged at 45 degrees and fused "
                   "into one rolling solid (Clahsen / Brecher)",
    "category": "Add Mesh",
}

import math
from math import cos, sin, pi, radians, sqrt


def _ring_frame(R, r, angle_deg, weld, sign):
    """Centre and in-plane axes of one ring.  The hinge line is the
    y-axis; each ring plane contains it, splayed sign*(angle/2) from
    the +z bisector direction.  The centre sits R + g from the hinge
    with g = (2r - weld) / (2 sin(angle/2)), so the two centre circles
    come no closer than 2r - weld: the tube SURFACES overlap by the
    shallow depth `weld` at one contact spot, while the centrelines
    stay well apart (no threading)."""
    half = 0.5 * radians(angle_deg)
    dx, dz = sign * sin(half), cos(half)          # in-plane radial axis
    d = (dx, 0.0, dz)
    e = (0.0, 1.0, 0.0)                           # hinge direction
    n = (e[1] * d[2] - e[2] * d[1],               # plane normal e x d
         e[2] * d[0] - e[0] * d[2],
         e[0] * d[1] - e[1] * d[0])
    g = (2.0 * r - weld) / (2.0 * sin(half))      # inner-rim-to-hinge
    a = R + g                                     # centre-to-hinge
    c = (a * d[0], a * d[1], a * d[2])
    return c, d, e, n


def _ring(R, r, angle_deg, weld, sign, nu, nv):
    """Quad mesh of one torus: tube of radius r swept around the ring
    circle given by _ring_frame.  Local vertex indices from 0."""
    c, d, e, n = _ring_frame(R, r, angle_deg, weld, sign)
    verts = []
    for i in range(nu):
        u = 2.0 * pi * i / nu
        cu, su = cos(u), sin(u)
        for j in range(nv):
            v = 2.0 * pi * j / nv
            rr = R + r * cos(v)
            zz = r * sin(v)
            verts.append(tuple(
                c[k] + rr * (cu * d[k] + su * e[k]) + zz * n[k]
                for k in range(3)))

    def vid(i, j):
        return (i % nu) * nv + (j % nv)

    faces = []
    for i in range(nu):
        for j in range(nv):
            faces.append([vid(i, j), vid(i + 1, j),
                          vid(i + 1, j + 1), vid(i, j + 1)])
    return verts, faces


def build_orbis(major=1.0, tube=0.35, angle=45.0, nu=96, nv=32,
                overlap=0.5):
    """Two hinged tori, tube surfaces overlapping by a shallow weld
    depth, as one combined mesh.  The ring planes (dihedral `angle`
    about the y-axis hinge) hold circles of radius `major` placed so
    their closest approach is 2*tube - `overlap`*tube: the tubes touch
    and shallowly overlap at ONE contact spot, centrelines apart.
    This immersed pair is the pure-Python form; in Blender the
    operator Boolean-unions the two rings into one connected solid."""
    R, r = major, tube
    ov = overlap * r
    vA, fA = _ring(R, r, angle, ov, +1, nu, nv)
    vB, fB = _ring(R, r, angle, ov, -1, nu, nv)
    off = len(vA)
    verts = vA + vB
    faces = fA + [[k + off for k in fc] for fc in fB]
    return verts, faces


def _dist_to_circle(p, c, d, e, R):
    """Distance from point p to the circle centre c, radius R, spanned
    by unit axes d, e (tube-centre distance for the solid-torus test)."""
    q = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
    x = q[0] * d[0] + q[1] * d[1] + q[2] * d[2]
    y = q[0] * e[0] + q[1] * e[1] + q[2] * e[2]
    qq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2]
    z2 = max(qq - x * x - y * y, 0.0)
    rho = sqrt(x * x + y * y)
    return sqrt((rho - R) ** 2 + z2)


try:
    import bpy
    import bmesh
    from bpy.props import IntProperty, FloatProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _ring_obj(coll, R, r, angle_deg, weld, sign, nu, nv):
        verts, faces = _ring(R, r, angle_deg, weld, sign, nu, nv)
        me = bpy.data.meshes.new("orbis_tmp")
        me.from_pydata(verts, [], faces)
        me.validate(clean_customdata=True)
        o = bpy.data.objects.new("orbis_tmp", me)
        coll.objects.link(o)
        return o

    def _build_orbis_csg(ctx, R, r, angle_deg, weld, nu, nv):
        """Boolean-UNION the two hinged rings into one connected solid
        (EXACT solver; same pattern as steinmetz_generator)."""
        coll = ctx.collection
        a = _ring_obj(coll, R, r, angle_deg, weld, +1, nu, nv)
        b = _ring_obj(coll, R, r, angle_deg, weld, -1, nu, nv)
        md = a.modifiers.new("u", 'BOOLEAN')
        md.operation = 'UNION'
        md.object = b
        md.solver = 'EXACT'
        with ctx.temp_override(object=a, active_object=a,
                               selected_objects=[a]):
            bpy.ops.object.modifier_apply(modifier=md.name)
        bpy.data.objects.remove(b, do_unlink=True)
        bm = bmesh.new()
        bm.from_mesh(a.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5 * r)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(a.data)
        bm.free()
        verts = [v.co[:] for v in a.data.vertices]
        faces = [list(p.vertices) for p in a.data.polygons]
        bpy.data.objects.remove(a, do_unlink=True)
        return verts, faces

    class MESH_OT_orbis_add(bpy.types.Operator):
        """Add an Orbis / Holey Roller -- two torus rings hinged at """ \
            """45 degrees and fused into one rolling solid"""
        bl_idname = "mesh.orbis_add"
        bl_label = "Orbis / Holey Roller"
        bl_options = {'REGISTER', 'UNDO'}

        tube: FloatProperty(
            name="Tube Radius", default=0.35, min=0.02, max=0.49,
            description="Ring thickness (fraction of the ring radius)")
        angle: FloatProperty(
            name="Hinge Angle", default=45.0, min=5.0, max=180.0,
            description="Dihedral angle between the two ring planes at "
                        "the hinge (about 45 degrees on the original "
                        "Orbis; 180 = coplanar figure-eight)")
        overlap: FloatProperty(
            name="Join Depth", default=0.5, min=0.05, max=1.0,
            description="Depth of the surface weld where the two tube "
                        "surfaces are pushed past tangency at the "
                        "contact spot, as a fraction of the tube "
                        "radius (small = a shallow glued joint)")
        major_segments: IntProperty(
            name="Ring Segments", default=96, min=12, max=512)
        tube_segments: IntProperty(
            name="Tube Segments", default=32, min=6, max=128)
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            r = self.tube
            verts, faces = _build_orbis_csg(
                context, 1.0, r, self.angle, self.overlap * r,
                self.major_segments, self.tube_segments)
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
            lay.prop(self, 'overlap')
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
    R, r, ang, ov = 1.0, 0.35, 45.0, 0.5
    nu, nv = 64, 24
    v, f = build_orbis(R, r, ang, nu, nv, ov)
    # each ring is a watertight torus (every edge shared by two faces,
    # chi = 0 per component, chi = 0 total for the pair).
    cnt = Counter()
    for fc in f:
        n = len(fc)
        for i in range(n):
            a, b = fc[i], fc[(i + 1) % n]
            cnt[(min(a, b), max(a, b))] += 1
    manifold = all(c == 2 for c in cnt.values())
    chi = len(v) - len(cnt) + len(f)
    # the surface weld: the two CENTRE circles must stay well apart
    # (closest approach 2r - weld, so no centreline threading), while
    # the tube surfaces overlap by the shallow weld depth -- i.e. some
    # mesh vertices of ring B lie inside ring A's solid torus (and
    # vice versa), so the Boolean union is one connected genus-2 body
    # joined only at the small contact patch.
    weld = ov * r
    cA, dA, eA, _ = _ring_frame(R, r, ang, weld, +1)
    cB, dB, eB, _ = _ring_frame(R, r, ang, weld, -1)
    dmin = min(_dist_to_circle(
        (cB[0] + R * (cos(t) * dB[0]) + R * (sin(t) * eB[0]),
         cB[1] + R * (cos(t) * dB[1]) + R * (sin(t) * eB[1]),
         cB[2] + R * (cos(t) * dB[2]) + R * (sin(t) * eB[2])),
        cA, dA, eA, R)
        for t in (2.0 * pi * k / 720 for k in range(720)))
    centres_apart = abs(dmin - (2.0 * r - weld)) < 1e-3 and dmin > r
    off = nu * nv
    in_a = sum(1 for p in v[off:]
               if _dist_to_circle(p, cA, dA, eA, R) < r - 1e-9)
    in_b = sum(1 for p in v[:off]
               if _dist_to_circle(p, cB, dB, eB, R) < r - 1e-9)
    ok = (manifold and chi == 0 and len(v) == 2 * nu * nv
          and centres_apart and in_a > 0 and in_b > 0)
    print(f"orbis: V={len(v)} F={len(f)} manifold={manifold} chi={chi}(0) "
          f"centre_gap={dmin:.4f}({2 * r - weld:.4f}) "
          f"weldA={in_a} weldB={in_b} "
          f"{'OK' if ok else 'BAD'}")
    assert ok
