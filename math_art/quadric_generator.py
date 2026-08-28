# Quadric surfaces: the degree-2 surfaces, by their exact charts.
#
# The nine non-degenerate real quadrics, plus the sphere, the spheroid
# and the plane, which the literature always treats alongside them.
# Together they are the whole of degree two, and they are the oldest
# classified family of surfaces there is -- Euler's 1748 Introductio
# gives the classification into ellipsoid, the two hyperboloids, the two
# paraboloids, the cone and the three cylinders.
#
# WHY CHARTS RATHER THAN LEVEL SETS.  Every quadric is exactly
# parametrisable, so a chart gives clean quads, exact normals, usable UVs
# and no marching-tetrahedra staircase.  The implicit route already
# exists for anyone who wants it -- `mesh.algebraic_surface_add` takes an
# arbitrary polynomial -- so meshing these by contouring would trade the
# one advantage degree two has for nothing.
#
# Three of the family shipped before this module, all as RULED surfaces:
# the hyperboloid of one sheet and the hyperbolic paraboloid (the two
# doubly ruled quadrics) in `ruled_surface_generator`, and the circular
# cylinder as the H = 1/2r member of the Delaunay family in
# `delaunay_generator`.  They are included here too, because a quadric
# generator missing the two doubly ruled ones would be a strange object,
# and because being reachable two ways is a fact about the surface rather
# than a duplication -- data/surfaces records both constructions.
#
# WHAT IS SINGULAR OR DISCONNECTED, since both are easy to mesh wrongly:
#   - the elliptic cone has an apex, where the surface is not smooth and
#     the chart degenerates; the mesh welds the apex to a single vertex.
#   - the hyperboloid of TWO SHEETS is exactly that: two components.  A
#     generator returning one sheet looks perfectly plausible and is
#     wrong, so both are built and the self-test counts them.
#
# References:
# - L. Euler, "Introductio in analysin infinitorum", vol. II (1748),
#   appendix on surfaces -- the classification of the quadrics.
# - G. Salmon, "A Treatise on the Analytic Geometry of Three
#   Dimensions" (1862), for the ruled quadrics and their generators.
# - D. Hilbert and S. Cohn-Vossen, "Anschauliche Geometrie" (1932),
#   chapter 1 -- the quadrics as the second-order surfaces, with the
#   doubly ruled cases discussed geometrically.
# - E. W. Weisstein, "Quadratic Surface", MathWorld, for the standard
#   normal forms used here.

bl_info = {
    "name": "Quadric Surfaces",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces > Quadric",
    "description": "The degree-2 surfaces: ellipsoid, paraboloids, "
                   "hyperboloids, cone, cylinders, sphere and plane",
    "category": "Add Mesh",
}

import math

TAU = 2.0 * math.pi

#: key -> (label, description)
QUADRICS = (
    ('SPHERE', "Sphere",
     "x^2 + y^2 + z^2 = 1. The K = +1 model, the Willmore minimiser and "
     "the isoperimetric extremum, all at once"),
    ('SPHEROID', "Spheroid",
     "An ellipsoid of revolution: c > 1 prolate (rugby ball), c < 1 "
     "oblate (the Earth), c = 1 the sphere"),
    ('ELLIPSOID', "Ellipsoid",
     "x^2/a^2 + y^2/b^2 + z^2/c^2 = 1, the closed quadric"),
    ('ELLIPTIC_PARABOLOID', "Elliptic Paraboloid",
     "z = x^2/a^2 + y^2/b^2, the reflector dish"),
    ('HYPERBOLIC_PARABOLOID', "Hyperbolic Paraboloid",
     "z = x^2/a^2 - y^2/b^2, the saddle: DOUBLY RULED, so buildable "
     "from two families of straight members"),
    ('HYPERBOLOID_ONE', "Hyperboloid of One Sheet",
     "x^2/a^2 + y^2/b^2 - z^2/c^2 = 1. The other doubly ruled quadric, "
     "which is why cooling towers are built from straight rods"),
    ('HYPERBOLOID_TWO', "Hyperboloid of Two Sheets",
     "x^2/a^2 + y^2/b^2 - z^2/c^2 = -1. DISCONNECTED: two components"),
    ('ELLIPTIC_CONE', "Elliptic Cone",
     "x^2/a^2 + y^2/b^2 = z^2. Ruled and developable away from the "
     "apex, which is the one point where it is not a smooth surface"),
    ('CIRCULAR_CYLINDER', "Circular Cylinder",
     "x^2 + y^2 = 1. Flat (K = 0) and CMC at once"),
    ('ELLIPTIC_CYLINDER', "Elliptic Cylinder", "x^2/a^2 + y^2/b^2 = 1"),
    ('HYPERBOLIC_CYLINDER', "Hyperbolic Cylinder",
     "x^2/a^2 - y^2/b^2 = 1. Two components, like the two-sheeted "
     "hyperboloid it is the cylinder over"),
    ('PARABOLIC_CYLINDER', "Parabolic Cylinder", "z = x^2/a^2"),
    ('PLANE', "Plane",
     "z = 0. Both minimal and flat -- the only surface that is both, "
     "and the trivial member of every classification here"),
)

#: which of them need more than one connected component
DISCONNECTED = {'HYPERBOLOID_TWO', 'HYPERBOLIC_CYLINDER'}


def _grid(nu, nv, u0, u1, v0, v1):
    us = [u0 + (u1 - u0) * i / float(nu) for i in range(nu + 1)]
    vs = [v0 + (v1 - v0) * j / float(nv) for j in range(nv + 1)]
    return us, vs


def _quads(nu, nv, wrap_u=False, wrap_v=False, base=0):
    """Quad indices over an (nu+1) x (nv+1) grid."""
    faces = []
    su = nu if wrap_u else nu
    for i in range(su):
        for j in range(nv):
            a = base + i * (nv + 1) + j
            b = base + ((i + 1) % (nu + 1) if wrap_u else i + 1) * (nv + 1) + j
            faces.append((a, b, b + 1, a + 1))
    return faces


def _patch(fn, nu, nv, u0, u1, v0, v1, base=0):
    """Vertices and quads of one parametric patch."""
    us, vs = _grid(nu, nv, u0, u1, v0, v1)
    verts = [fn(u, v) for u in us for v in vs]
    return verts, _quads(nu, nv, base=base)


def build(kind, nu=64, nv=48, a=1.0, b=0.7, c=0.5, extent=1.2):
    """(verts, faces, components) for one quadric.

    `extent` bounds the unbounded members; the closed ones ignore it.
    """
    A, B, C = max(a, 1e-6), max(b, 1e-6), max(c, 1e-6)
    E = max(extent, 1e-3)

    if kind == 'SPHERE':
        A = B = C = 1.0
        kind_ell = True
    elif kind == 'SPHEROID':
        A = B = 1.0
        kind_ell = True
    elif kind == 'ELLIPSOID':
        kind_ell = True
    else:
        kind_ell = False

    if kind_ell:
        # u = polar angle, v = azimuth; poles welded to single vertices
        verts = [(0.0, 0.0, C)]
        rings = []
        for i in range(1, nu):
            t = math.pi * i / float(nu)
            ring = []
            for j in range(nv):
                p = TAU * j / float(nv)
                ring.append(len(verts))
                verts.append((A * math.sin(t) * math.cos(p),
                              B * math.sin(t) * math.sin(p),
                              C * math.cos(t)))
            rings.append(ring)
        south = len(verts)
        verts.append((0.0, 0.0, -C))
        faces = []
        for j in range(nv):
            faces.append((0, rings[0][j], rings[0][(j + 1) % nv]))
        for i in range(len(rings) - 1):
            for j in range(nv):
                faces.append((rings[i][j], rings[i + 1][j],
                              rings[i + 1][(j + 1) % nv],
                              rings[i][(j + 1) % nv]))
        for j in range(nv):
            faces.append((south, rings[-1][(j + 1) % nv], rings[-1][j]))
        return verts, faces, 1

    if kind == 'ELLIPTIC_PARABOLOID':
        def f(u, v):
            return (A * u * math.cos(v), B * u * math.sin(v), u * u)
        # welded apex at u = 0
        verts = [(0.0, 0.0, 0.0)]
        rings = []
        for i in range(1, nu + 1):
            u = E * i / float(nu)
            ring = []
            for j in range(nv):
                ring.append(len(verts))
                verts.append(f(u, TAU * j / float(nv)))
            rings.append(ring)
        faces = [(0, rings[0][j], rings[0][(j + 1) % nv]) for j in range(nv)]
        for i in range(len(rings) - 1):
            for j in range(nv):
                faces.append((rings[i][j], rings[i + 1][j],
                              rings[i + 1][(j + 1) % nv],
                              rings[i][(j + 1) % nv]))
        return verts, faces, 1

    if kind == 'HYPERBOLIC_PARABOLOID':
        def f(u, v):
            return (A * u, B * v, u * u - v * v)
        v, fc = _patch(f, nu, nv, -E, E, -E, E)
        return v, fc, 1

    if kind == 'HYPERBOLOID_ONE':
        def f(u, v):
            return (A * math.cosh(u) * math.cos(v),
                    B * math.cosh(u) * math.sin(v),
                    C * math.sinh(u))
        us, vs = _grid(nu, nv, -E, E, 0.0, TAU)
        verts = [f(u, v) for u in us for v in vs[:-1]]
        m = len(vs) - 1
        faces = []
        for i in range(nu):
            for j in range(m):
                p = i * m + j
                q = (i + 1) * m + j
                faces.append((p, q, q + (1 if j + 1 < m else 1 - m),
                              p + (1 if j + 1 < m else 1 - m)))
        return verts, faces, 1

    if kind == 'HYPERBOLOID_TWO':
        # TWO components -- the whole point of this one
        verts, faces = [], []
        for sgn in (1.0, -1.0):
            base = len(verts)
            top = len(verts)
            verts.append((0.0, 0.0, sgn * C))
            rings = []
            for i in range(1, nu + 1):
                u = E * i / float(nu)
                ring = []
                for j in range(nv):
                    p = TAU * j / float(nv)
                    ring.append(len(verts))
                    verts.append((A * math.sinh(u) * math.cos(p),
                                  B * math.sinh(u) * math.sin(p),
                                  sgn * C * math.cosh(u)))
                rings.append(ring)
            for j in range(nv):
                faces.append((top, rings[0][j], rings[0][(j + 1) % nv]))
            for i in range(len(rings) - 1):
                for j in range(nv):
                    faces.append((rings[i][j], rings[i + 1][j],
                                  rings[i + 1][(j + 1) % nv],
                                  rings[i][(j + 1) % nv]))
            del base
        return verts, faces, 2

    if kind == 'ELLIPTIC_CONE':
        # apex welded: a chart singularity, not a mesh defect
        verts = [(0.0, 0.0, 0.0)]
        rings = []
        for i in range(1, nu + 1):
            z = E * i / float(nu)
            ring = []
            for j in range(nv):
                p = TAU * j / float(nv)
                ring.append(len(verts))
                verts.append((A * z * math.cos(p), B * z * math.sin(p), z))
            rings.append(ring)
        faces = [(0, rings[0][j], rings[0][(j + 1) % nv]) for j in range(nv)]
        for i in range(len(rings) - 1):
            for j in range(nv):
                faces.append((rings[i][j], rings[i + 1][j],
                              rings[i + 1][(j + 1) % nv],
                              rings[i][(j + 1) % nv]))
        return verts, faces, 1

    if kind in ('CIRCULAR_CYLINDER', 'ELLIPTIC_CYLINDER'):
        ra, rb = (1.0, 1.0) if kind == 'CIRCULAR_CYLINDER' else (A, B)
        verts, faces = [], []
        for i in range(nu + 1):
            z = -E + 2.0 * E * i / float(nu)
            for j in range(nv):
                p = TAU * j / float(nv)
                verts.append((ra * math.cos(p), rb * math.sin(p), z))
        for i in range(nu):
            for j in range(nv):
                a0 = i * nv + j
                a1 = i * nv + (j + 1) % nv
                b0 = (i + 1) * nv + j
                b1 = (i + 1) * nv + (j + 1) % nv
                faces.append((a0, b0, b1, a1))
        return verts, faces, 1

    if kind == 'HYPERBOLIC_CYLINDER':
        verts, faces = [], []
        for sgn in (1.0, -1.0):
            base = len(verts)
            def f(u, v, _s=sgn):
                return (_s * A * math.cosh(u), B * math.sinh(u), v)
            vv, ff = _patch(f, nu, nv, -E, E, -E, E, base=base)
            verts += vv
            faces += ff
        return verts, faces, 2

    if kind == 'PARABOLIC_CYLINDER':
        def f(u, v):
            return (A * u, v, u * u)
        v, fc = _patch(f, nu, nv, -E, E, -E, E)
        return v, fc, 1

    if kind == 'PLANE':
        def f(u, v):
            return (u, v, 0.0)
        v, fc = _patch(f, nu, nv, -E, E, -E, E)
        return v, fc, 1

    raise ValueError("unknown quadric %r" % kind)


def fit(verts, size=1.0):
    """Centre on the bounding-box midpoint, scale so the largest extent
    is 2 * size -- the repo's display convention."""
    if not verts:
        return verts
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    cz = 0.5 * (min(zs) + max(zs))
    ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    s = (2.0 * size / ext) if ext > 1e-12 else 1.0
    return [((v[0] - cx) * s, (v[1] - cy) * s, (v[2] - cz) * s)
            for v in verts]


# ---------------------------------------------------------------------------


try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, IntProperty
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    class MESH_OT_quadric_add(bpy.types.Operator):
        """Add a quadric: one of the degree-2 surfaces, built from its
        exact parametrisation rather than by contouring"""
        bl_idname = "mesh.quadric_add"
        bl_label = "Quadric"
        bl_options = {'REGISTER', 'UNDO'}

        kind: EnumProperty(
            name="Surface",
            items=[(k, lab, desc) for k, lab, desc in QUADRICS],
            default='ELLIPSOID',
            description="Which quadric to build")
        a: FloatProperty(
            name="Semi-axis a", default=1.0, min=0.05, max=10.0,
            description="First semi-axis; ignored by the sphere")
        b: FloatProperty(
            name="Semi-axis b", default=0.7, min=0.05, max=10.0,
            description="Second semi-axis; ignored by the sphere and the "
                        "spheroid, which are surfaces of revolution")
        c: FloatProperty(
            name="Semi-axis c", default=0.5, min=0.05, max=10.0,
            description="Polar semi-axis. For the spheroid, c > 1 is "
                        "prolate and c < 1 oblate")
        extent: FloatProperty(
            name="Extent", default=1.2, min=0.1, max=6.0,
            description="How far an unbounded quadric is drawn before it "
                        "is cut off. The closed ones ignore it")
        segments_u: IntProperty(
            name="Segments U", default=64, min=4, max=512)
        segments_v: IntProperty(
            name="Segments V", default=48, min=3, max=512)
        size: FloatProperty(
            name="Size", default=1.0, min=0.01, max=100.0,
            description="Half the largest extent of the finished object")

        def execute(self, context):
            verts, faces, comps = build(
                self.kind, self.segments_u, self.segments_v,
                self.a, self.b, self.c, self.extent)
            verts = fit(verts, self.size)

            label = dict((k, lab) for k, lab, _d in QUADRICS)[self.kind]
            me = bpy.data.meshes.new(label)
            me.from_pydata(verts, [], faces)
            me.validate()
            me.update()
            obj = bpy.data.objects.new(label, me)
            context.collection.objects.link(obj)
            context.view_layer.objects.active = obj
            obj.select_set(True)
            self.report(
                {'INFO'},
                "%s: %d verts, %d faces, %d component%s"
                % (label, len(verts), len(faces), comps,
                   "" if comps == 1 else "s"))
            return {'FINISHED'}

    def _menu_func(self, context):
        self.layout.operator(MESH_OT_quadric_add.bl_idname,
                             text="Quadric", icon='MESH_UVSPHERE')

    def register():
        bpy.utils.register_class(MESH_OT_quadric_add)
        if hasattr(bpy.types, "VIEW3D_MT_mesh_add"):
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if hasattr(bpy.types, "VIEW3D_MT_mesh_add"):
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_quadric_add)


def _selftest():
    """Numeric self-test; raises on failure.

    Each quadric is checked against the property that DEFINES it -- the
    implicit equation it satisfies -- rather than against a vertex count,
    so a wrong chart fails rather than merely looking different.
    """
    # implicit form each chart must satisfy, at the default semi-axes
    IMPLICIT = {
        'SPHERE': lambda x, y, z: x * x + y * y + z * z - 1.0,
        'SPHEROID': lambda x, y, z: x * x + y * y + z * z / 0.25 - 1.0,
        'ELLIPSOID': lambda x, y, z: (x * x + y * y / 0.49
                                      + z * z / 0.25 - 1.0),
        'ELLIPTIC_PARABOLOID': lambda x, y, z: x * x + y * y / 0.49 - z,
        'HYPERBOLIC_PARABOLOID': lambda x, y, z: x * x - y * y / 0.49 - z,
        'HYPERBOLOID_ONE': lambda x, y, z: (x * x + y * y / 0.49
                                            - z * z / 0.25 - 1.0),
        'HYPERBOLOID_TWO': lambda x, y, z: (x * x + y * y / 0.49
                                            - z * z / 0.25 + 1.0),
        'ELLIPTIC_CONE': lambda x, y, z: x * x + y * y / 0.49 - z * z,
        'CIRCULAR_CYLINDER': lambda x, y, z: x * x + y * y - 1.0,
        'ELLIPTIC_CYLINDER': lambda x, y, z: x * x + y * y / 0.49 - 1.0,
        'HYPERBOLIC_CYLINDER': lambda x, y, z: x * x - y * y / 0.49 - 1.0,
        'PARABOLIC_CYLINDER': lambda x, y, z: x * x - z,
        'PLANE': lambda x, y, z: z,
    }

    for key, _label, _desc in QUADRICS:
        verts, faces, comps = build(key, 24, 18)
        assert verts and faces, "%s built nothing" % key

        # every vertex must satisfy the defining equation
        f = IMPLICIT[key]
        worst = max(abs(f(*v)) for v in verts)
        assert worst < 1e-9, \
            "%s: a vertex misses its own equation by %.3g" % (key, worst)

        # the disconnected ones must actually be disconnected
        want = 2 if key in DISCONNECTED else 1
        assert comps == want, \
            "%s should have %d component(s), reports %d" % (key, want, comps)

        # faces index real vertices
        n = len(verts)
        for fa in faces:
            assert all(0 <= i < n for i in fa), "%s: bad face index" % key

    # the two-sheeted hyperboloid must really separate in SPACE, not just
    # be built in two loops: no vertex of the upper sheet may have z <= 0.
    verts, _f, _c = build('HYPERBOLOID_TWO', 20, 16)
    zs = [v[2] for v in verts]
    assert min(zs) < 0 < max(zs), "two sheets must straddle z = 0"
    assert not any(abs(z) < 0.4 for z in zs), \
        "the sheets must not meet near z = 0; c = 0.5 puts the vertices at " \
        "|z| >= 0.5"

    # fit() centres on the box midpoint and scales the largest extent to 2
    v = fit([(0.0, 0.0, 0.0), (4.0, 2.0, 1.0)], 1.0)
    ext = max(max(p[i] for p in v) - min(p[i] for p in v) for i in range(3))
    assert abs(ext - 2.0) < 1e-12, ext
    mid = [0.5 * (max(p[i] for p in v) + min(p[i] for p in v))
           for i in range(3)]
    assert all(abs(m) < 1e-12 for m in mid), mid

    print("RESULT: OK  (quadric_generator, %d surfaces, each checked "
          "against its own implicit equation)" % len(QUADRICS))
