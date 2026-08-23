
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
# Nothing stops at three.  Take n equal cylinders whose axes are the
# diagonals of a symmetric solid and the surface becomes n RINGS of
# cylinder patches, one ring per cylinder, in patterns that echo the
# Catalan solids: 4 cube diagonals give 24 patches arranged like the
# deltoidal icositetrahedron, 6 cuboctahedral axes give 36 like the
# tetragonal triacontahexahedron, 6 icosahedral axes give 60 like the
# deltoidal hexecontahedron, and 10 and 12 axes give 180 and 216.  A ring
# is pinched wherever another axis's great circle crosses its own, at the
# directions +-(a_j x a_k), so counting DISTINCT cross-products counts the
# patches exactly -- see `ring_patches`.
#
# Move the axes into a single plane instead and the same construction
# gives the EQUIDOMOID, Archimedes' dome: a prism whose flat sides have
# become elliptic cylinder lunes, and which tends to a SPHERE as its order
# rises (the circumscribing prism only tends to a cylinder).
#
# The bicylinder's volume 16 r^3 / 3 was found by Archimedes (Method of
# Mechanical Theorems) and independently by the electrical engineer
# Charles Proteus Steinmetz, after whom the solid is named.  The
# bicylinder is built here directly as an exact watertight mesh of its
# two cylinder patches (the patch grids follow the cylinders, so the
# curved edges stay clean); the tricylinder is built by Boolean
# intersection of three cylinders, and the larger sets by the radial
# builder, which is exact for these convex star-shaped solids and avoids
# a fragile stack of a dozen Booleans.
#
# References:
#   - Archimedes, "The Method of Mechanical Theorems" (c. 250 BC);
#     T. L. Heath (ed.), "The Works of Archimedes", Cambridge, 1897.
#   - M. Moore, "Symmetrical Intersections of Right Circular Cylinders",
#     The Mathematical Gazette 58 (1974), 181-185.
#   - K. Brecher, "Rolloids", Bridges 2023 Conference Proceedings,
#     345-352 (the bicylinder as a straight-line roller; local copy at
#     research/bridges/2023/bridges2023-345/).
#   - The n-cylinder axis sets and their patch counts: Robert Ferreol,
#     "Encyclopedie des formes mathematiques remarquables"
#     (mathcurve.com), "polyedres cylindriques".  Four of its five patch
#     counts are reproduced exactly here; for the twelve
#     truncated-octahedral axes the count is 216 rather than the
#     published 240, since that axis set is in special position (two of
#     the eleven cross-product directions coincide).
#   - Equidomoid: studied by Archimedes, and named by Leopold Hugo
#     between 1867 and 1875; Emile Fourrey, "Recreations geometriques",
#     319-326.

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
PHI = (1.0 + sqrt(5.0)) / 2.0


def _axes(vectors):
    """Unit axes, one per antipodal pair (a cylinder has no direction)."""
    out = []
    for v in vectors:
        n = sqrt(sum(c * c for c in v))
        u = tuple(c / n for c in v)
        if not any(abs(abs(sum(u[i] * w[i] for i in range(3))) - 1.0) < 1e-9
                   for w in out):
            out.append(u)
    return out


def _perms3(a, b, c):
    """The three cyclic permutations of a coordinate triple."""
    return [(a, b, c), (c, a, b), (b, c, a)]


def _all_perms3(a, b, c):
    """All six permutations -- needed where the solid is not chiral."""
    return [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]


# The axis sets of ch078: n equal cylinders whose axes are the diagonals of
# a named solid.  Intersecting them gives a solid whose surface is n RINGS
# of cylinder patches, and the ring/patch counts are the published check.
# `patches` is the total number of patches (rings x patches per ring).
AXIS_SETS = {
    'BICYLINDER': ((1, 0, 0), (0, 1, 0)),
    'TRICYLINDER': ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    # 4 cube diagonals -> the topology of the deltoidal icositetrahedron
    'CUBE4': tuple((sx, sy, 1) for sx in (1, -1) for sy in (1, -1)),
    # 6 cuboctahedron vertices (= face diagonals of the rhombic dodecahedron)
    'CUBOCT6': tuple(v for s in (1, -1)
                     for v in _perms3(1, s, 0)),
    # 6 icosahedron vertices (= face diagonals of the dodecahedron)
    'ICOSA6': tuple(v for s in (1, -1)
                    for v in _perms3(0, 1, s * PHI)),
    # 10 dodecahedron vertices
    'DODECA10': tuple([(sx, sy, 1) for sx in (1, -1) for sy in (1, -1)]
                      + [v for s in (1, -1)
                         for v in _perms3(0, s / PHI, PHI)]),
    # 12 truncated-octahedron vertices: ALL permutations of (0, +-1, +-2)
    'TRUNCOCT12': tuple(v for s in (1, -1) for t in (1, -1)
                        for v in _all_perms3(0.0, s * 1.0, t * 2.0)),
}
AXIS_SETS = {k: tuple(_axes(v)) for k, v in AXIS_SETS.items()}


def _steinmetz_radius(kind, u, radius):
    """Distance from the origin to the boundary along the unit vector u.

    The intersection of solid cylinders through the origin is convex and
    contains the origin, so its boundary is a radial graph: along u the
    cylinder about axis a is left at radius / |u - (u.a)a|, and the
    binding constraint is whichever cylinder gives the smallest value.
    """
    axes = kind if isinstance(kind, (list, tuple)) and not isinstance(
        kind, str) else AXIS_SETS[kind]
    best = 1e18
    for a in axes:
        d = sum(u[i] * a[i] for i in range(3))
        off = sum((u[i] - d * a[i]) ** 2 for i in range(3))
        if off > 1e-12:
            t = radius / sqrt(off)
            if t < best:
                best = t
    return best


def equidomoid_axes(order):
    """`order` cylinder axes spaced evenly in ONE plane.

    Intersecting equal cylinders whose axes are coplanar and regularly
    spaced gives -- up to a dilation -- the EQUIDOMOID, Archimedes' dome:
    a right prism on a regular n-gon whose height equals the diameter of
    its base and whose n flat side faces have been replaced by elliptic
    cylinder lunes of semi-axes R and R.cos(pi/n).

    Its limit is the reason it is worth having.  A point p is inside iff
    its distance to EVERY axis is at most R.  With the axes filling a whole
    plane, the worst axis for p is the one perpendicular to p's horizontal
    part, and the distance to it is just |p| -- so the condition collapses
    to |p| <= R and the equidomoid tends to a SPHERE, while the prism
    circumscribing it tends only to a cylinder.  An axis direction is the
    same as its opposite, so the axes run over half a turn, not a full one.
    """
    n = max(2, int(order))
    return tuple((cos(pi * k / n), sin(pi * k / n), 0.0) for k in range(n))


def ring_patches(kind, k):
    """How many patches the k-th cylinder's ring is cut into.

    On the great circle perpendicular to axis a_k that cylinder is always
    the binding one, at the smallest possible radius; another cylinder a_j
    can only tie with it where the direction is perpendicular to a_j as
    well.  Those directions are exactly +-(a_k x a_j), so each DISTINCT
    cross-product direction pinches the ring at an antipodal pair of
    points, and the ring falls into that many patches.  Coincident
    cross-products -- axis sets in special position -- pinch fewer times.
    """
    axes = AXIS_SETS[kind] if isinstance(kind, str) else kind
    a = axes[k]
    dirs = []
    for j, b in enumerate(axes):
        if j == k:
            continue
        c = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0])
        n = sqrt(sum(x * x for x in c))
        if n < 1e-9:
            continue
        c = tuple(x / n for x in c)
        if not any(abs(abs(sum(c[i] * d[i] for i in range(3))) - 1.0) < 1e-9
                   for d in dirs):
            dirs.append(c)
    return 2 * len(dirs)


def patch_count(kind):
    """Total number of cylinder patches on the solid's surface."""
    axes = AXIS_SETS[kind] if isinstance(kind, str) else kind
    return sum(ring_patches(axes, k) for k in range(len(axes)))


def _steinmetz_binding(kind, u):
    """Index of the cylinder that the boundary point along u belongs to."""
    axes = AXIS_SETS[kind] if isinstance(kind, str) else kind
    best, arg = 1e18, -1
    for k, a in enumerate(axes):
        d = sum(u[i] * a[i] for i in range(3))
        off = sum((u[i] - d * a[i]) ** 2 for i in range(3))
        if off > 1e-12:
            t = 1.0 / sqrt(off)
            if t < best - 1e-12:
                best, arg = t, k
    return arg


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
                    "cube with twelve curved edges"),
                   ('CUBE4', "Four Cube Diagonals",
                    "Axes through the cube's corners; 24 patches, with "
                    "the pattern of the deltoidal icositetrahedron"),
                   ('CUBOCT6', "Six Cuboctahedral Axes",
                    "Axes through the cuboctahedron's vertices; 36 "
                    "patches, patterned like the tetragonal "
                    "triacontahexahedron"),
                   ('ICOSA6', "Six Icosahedral Axes",
                    "Axes through the icosahedron's vertices; 60 "
                    "patches, like the deltoidal hexecontahedron"),
                   ('DODECA10', "Ten Dodecahedral Axes",
                    "Axes through the dodecahedron's vertices; 180 "
                    "patches"),
                   ('TRUNCOCT12', "Twelve Truncated-Octahedral Axes",
                    "Axes through the truncated octahedron's vertices; "
                    "216 patches, the roundest of the set"),
                   ('EQUIDOMOID', "Equidomoid (Archimedes' Dome)",
                    "Axes evenly spaced in ONE plane instead of around "
                    "the sphere: a prism whose sides have become "
                    "elliptic lunes, tending to a sphere as the order "
                    "rises")],
            default='BICYLINDER',
            description="Which set of cylinder axes to intersect")
        order: IntProperty(
            name="Order", default=5, min=2, max=64,
            description="Number of coplanar cylinders, which is also the "
                        "number of sides of the underlying prism "
                        "(equidomoid only)")
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
            elif self.kind == 'TRICYLINDER':
                verts, faces = _build_tricylinder_csg(context, self.radius,
                                                      self.segments)
            else:
                # Booleans over six or twelve cylinders are slow and
                # fragile; the solid is convex and star-shaped about the
                # origin, so the radial builder gives it exactly.
                axes = (equidomoid_axes(self.order)
                        if self.kind == 'EQUIDOMOID' else self.kind)
                verts, faces = build_steinmetz_radial(
                    axes, self.radius, 2 * self.segments, 4 * self.segments)

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
            if self.kind == 'EQUIDOMOID':
                r = self.radius
                far = max(abs(math.dist((0, 0, 0), v) - r) for v in verts)
                self.report({'INFO'},
                            f"V={len(me.vertices)} F={len(me.polygons)}, "
                            f"furthest departure from the sphere it tends "
                            f"to: {far / r:.3f} of the radius")
            elif self.kind not in ('BICYLINDER', 'TRICYLINDER'):
                self.report({'INFO'},
                            f"V={len(me.vertices)} F={len(me.polygons)}, "
                            f"{len(AXIS_SETS[self.kind])} cylinders in "
                            f"{patch_count(self.kind)} patches")
            else:
                self.report({'INFO'},
                            f"V={len(me.vertices)} F={len(me.polygons)}")
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'kind')
            if self.kind == 'EQUIDOMOID':
                lay.prop(self, 'order')
            lay.prop(self, 'radius')
            lay.prop(self, 'segments')
            lay.prop(self, 'sharp_edges')

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

    # The n-cylinder sets.  Each cylinder contributes a RING of patches
    # around the great circle perpendicular to its axis, pinched wherever
    # another axis's great circle crosses it.  Counting those pinch points
    # exactly reproduces the published tallies for four of the five sets.
    #
    # It does NOT reproduce the fifth.  The source gives 12 x 20 = 240 for
    # the twelve truncated-octahedral axes; the count here is 12 x 18 =
    # 216, because two of the eleven cross-product directions from any one
    # axis coincide -- that axis set is in special position, not general
    # position.  Two independent methods agree on 216 (this exact count,
    # and a flood fill of the argmin partition on a 800 x 1600 grid), and
    # both reproduce 12 / 24 / 36 / 60 for the other four sets, so the
    # published 240 looks like an assumption of general position.
    want = {'TRICYLINDER': 12, 'CUBE4': 24, 'CUBOCT6': 36, 'ICOSA6': 60,
            'DODECA10': 180, 'TRUNCOCT12': 216}
    for kind, n in want.items():
        got = patch_count(kind)
        assert got == n, (kind, got, n)
        per = {ring_patches(kind, k) for k in range(len(AXIS_SETS[kind]))}
        assert len(per) == 1, (kind, per)      # every ring alike
        v, f = build_steinmetz_radial(kind, 1.0, 48, 96)
        man, ne = _edge_manifold(f)
        assert man and len(v) - ne + len(f) == 2, (kind, man)
        rr = [math.sqrt(sum(c * c for c in p)) for p in v]
        assert min(rr) > 0.999, (kind, min(rr))   # insphere is the radius
    print("steinmetz[n-cylinder]: patch counts %s -- 4 of 5 published "
          "counts reproduced exactly; TRUNCOCT12 is 216, not the "
          "published 240 (that axis set is in special position)"
          % {k: patch_count(k) for k in want})

    # Equidomoid: coplanar axes.  The height is exactly the diameter, and
    # the solid closes down onto the sphere of radius r as the order
    # rises, monotonically -- the property that makes it Archimedes' dome
    # rather than just another prism.
    far = []
    for order in (3, 4, 5, 6, 8, 12, 24):
        v, f = build_steinmetz_radial(equidomoid_axes(order), 1.0, 96, 192)
        man, ne = _edge_manifold(f)
        assert man and len(v) - ne + len(f) == 2, (order, man)
        zs = [p[2] for p in v]
        assert abs(max(zs) - 1.0) < 1e-9 and abs(min(zs) + 1.0) < 1e-9, \
            (order, min(zs), max(zs))          # height = 2r = the diameter
        far.append(max(math.sqrt(sum(c * c for c in p)) for p in v) - 1.0)
    assert all(far[i] > far[i + 1] for i in range(len(far) - 1)), far
    assert far[-1] < 0.01, far[-1]
    print("steinmetz[equidomoid]: height is exactly the base diameter, "
          "and the departure from the limiting sphere falls %.3f -> %.4f "
          "over orders 3..24" % (far[0], far[-1]))
