
# Hopf Fibration generator for Blender.
#
# The Hopf fibration is the map h: S^3 -> S^2 whose fibres are great
# circles of the 3-sphere; any two distinct fibres are linked exactly
# once (a Hopf link).  This generator draws a chosen set of fibres,
# stereographically projected from S^3 into R^3, where every fibre
# becomes a circle (a Villarceau circle) -- except the one fibre
# through the projection pole, which becomes the straight central
# axis.  Fibres over a circle of latitude on S^2 fill a torus of
# revolution; sweeping the latitude fills space with nested, pairwise
# linked tori (the canonical "Niles Johnson" picture).
#
# Construction (per base point p = (sin b cos l, sin b sin l, cos b)
# on S^2, colatitude b and longitude l): the fibre h^{-1}(p) is the
# Clifford circle
#     z0 = cos(b/2) e^{i(t + l)},   z1 = sin(b/2) e^{i t},   t in [0,2pi)
# i.e. the S^3 point (Re z0, Im z0, Re z1, Im z1).  One checks
# h(z0,z1) = (2 z0 conj(z1), |z0|^2 - |z1|^2) = p for every t, so the
# whole circle sits over p.  A (p,q) generalisation winds the two
# phases at integer rates -- z0 = cos(b/2) e^{i(P t + l)},
# z1 = sin(b/2) e^{i Q t} -- laying a (P,Q) torus knot/link on the
# same Clifford torus (P=Q=1 is the classical fibre).  Stereographic
# projection from the north pole N=(0,0,0,1) is
#     sigma(x1,x2,x3,x4) = (x1,x2,x3) / (1 - x4).
# Colour convention (after Niles Johnson): a fibre is coloured by its
# base point on S^2 -- hue from the longitude, value from the
# latitude -- so nearby fibres share a colour.
#
# References:
# - Heinz Hopf, "Ueber die Abbildungen der dreidimensionalen Sphaere
#   auf die Kugelflaeche", Math. Ann. 104 (1931), 637-665 (the
#   fibration and its linking invariant).
# - D. W. Lyons, "An Elementary Introduction to the Hopf Fibration",
#   Math. Mag. 76 (2003), 87-98.
# - N. Johnson, "Visualization of the Hopf fibration" (2011),
#   https://nilesjohnson.net/hopf.html (the colour/latitude-torus
#   rendering imitated here).
# - Y. Villarceau (1848): a torus of revolution carries two extra
#   circles through each point, the "Villarceau circles" -- exactly
#   the Hopf fibres of a stereographically projected Clifford torus.

bl_info = {
    "name": "Hopf Fibration",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Curve > Hopf Fibration",
    "description": "Fibres of the Hopf fibration of S^3 as "
                   "stereographically projected circles (Villarceau "
                   "circles), coloured by base point on S^2",
    "category": "Add Curve",
}

import math
from math import gcd, pi, sqrt

try:
    import bpy
    from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                           BoolProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

_PHI = (1.0 + sqrt(5.0)) / 2.0

# A fixed, generic rotation of S^2 applied to every base point so that
# none of the standard base sets (octahedron / cube vertices, latitude
# poles) lands exactly on the south pole, whose fibre would project to
# an infinite line and blow up the scene.  The angles are arbitrary
# and irrational-looking on purpose.
_TILT = (0.3178, 0.2114, 0.1291)


def _rot_matrix(ax, ay, az):
    import numpy as np
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# --------------------------------------------------------------------------
# Kernel: fibre lift to S^3 and stereographic projection to R^3
# --------------------------------------------------------------------------

def fiber_s3(base, samples, P=1, Q=1):
    """The fibre over unit 3-vector `base` = (a,b,c) on S^2, sampled
    at `samples` angles, as an (samples, 4) array on S^3.  With P=Q=1
    this is the Hopf fibre (a great circle); general (P,Q) winds the
    two complex phases at those integer rates, giving a (P,Q) torus
    curve on the same Clifford torus."""
    import numpy as np
    a, b, c = base
    beta = math.acos(max(-1.0, min(1.0, c)))   # colatitude in [0, pi]
    lam = math.atan2(b, a)                      # longitude
    ch, sh = math.cos(beta / 2.0), math.sin(beta / 2.0)
    t = np.linspace(0.0, 2.0 * pi, samples, endpoint=False)
    p0 = P * t + lam
    p1 = Q * t
    return np.stack([ch * np.cos(p0), ch * np.sin(p0),
                     sh * np.cos(p1), sh * np.sin(p1)], axis=1)


def stereographic(X):
    """Stereographic projection S^3 -> R^3 from the north pole
    N = (0,0,0,1): (x1,x2,x3,x4) -> (x1,x2,x3)/(1 - x4).  Points near
    N are pushed far out; the caller keeps base points off the south
    pole so no fibre actually reaches N."""
    import numpy as np
    denom = 1.0 - X[:, 3]
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    return X[:, :3] / denom[:, None]


def project_fiber(base, samples, P=1, Q=1):
    """A single fibre as an (samples, 3) projected polyline."""
    return stereographic(fiber_s3(base, samples, P, Q))


# --------------------------------------------------------------------------
# Base-point providers on S^2 (each returns a list of unit 3-vectors)
# --------------------------------------------------------------------------

def _normalize(v):
    import numpy as np
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def latitudes(n_lat, n_fiber, lat_min_deg=20.0, lat_max_deg=160.0):
    """`n_lat` circles of latitude (colatitude spread between
    lat_min and lat_max degrees), each carrying `n_fiber` fibres
    equally spaced in longitude.  This is the nested-tori picture."""
    import numpy as np
    pts = []
    if n_lat == 1:
        betas = [0.5 * (lat_min_deg + lat_max_deg) * pi / 180.0]
    else:
        betas = np.linspace(lat_min_deg, lat_max_deg, n_lat) * pi / 180.0
    for beta in betas:
        for k in range(n_fiber):
            lam = 2.0 * pi * k / n_fiber
            pts.append((math.sin(beta) * math.cos(lam),
                        math.sin(beta) * math.sin(lam),
                        math.cos(beta)))
    return pts


def flower(n_fiber, ring_deg=35.0):
    """Fibres over a single small circle at colatitude `ring_deg`:
    the projected circles interleave into a flower-like linked
    bundle."""
    return latitudes(1, n_fiber, ring_deg, ring_deg)


def _platonic(kind):
    import numpy as np
    if kind == 'TETRA':
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    elif kind == 'OCTA':
        V = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
             (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif kind == 'CUBE':
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
    elif kind == 'ICOSA':
        p = _PHI
        V = []
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                V += [(0, s1, s2 * p), (s1, s2 * p, 0),
                      (s2 * p, 0, s1)]
    elif kind == 'DODECA':
        p = _PHI
        V = [(x, y, z) for x in (-1, 1) for y in (-1, 1)
             for z in (-1, 1)]
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                V += [(0, s1 / p, s2 * p), (s1 / p, s2 * p, 0),
                      (s2 * p, 0, s1 / p)]
    else:
        raise ValueError(kind)
    return [tuple(_normalize(v)) for v in V]


def fibonacci(n):
    """`n` base points spread over S^2 by the golden-angle spiral --
    a quasi-uniform ball of linked fibres."""
    pts = []
    ga = pi * (3.0 - sqrt(5.0))            # golden angle
    for k in range(n):
        z = 1.0 - (2.0 * k + 1.0) / n      # in (-1, 1)
        r = sqrt(max(0.0, 1.0 - z * z))
        phi = ga * k
        pts.append((r * math.cos(phi), r * math.sin(phi), z))
    return pts


def great_circle(n_fiber, tilt_deg=55.0):
    """Fibres over a great circle of S^2 tilted `tilt_deg` from the
    equator: a single, maximally spread band of linked fibres (its
    two extreme fibres bound a Hopf band)."""
    import numpy as np
    pts = []
    a = tilt_deg * pi / 180.0
    for k in range(n_fiber):
        u = 2.0 * pi * k / n_fiber
        # equator circle rotated about the x-axis by `a`
        x = math.cos(u)
        y = math.sin(u) * math.cos(a)
        z = math.sin(u) * math.sin(a)
        pts.append((x, y, z))
    return pts


_PROVIDERS = {
    'LATITUDES': "Nested tori (circles of latitude)",
    'FLOWER': "Flower (one small circle of base points)",
    'GREATCIRCLE': "Great-circle band",
    'FIBONACCI': "Fibonacci sphere (quasi-uniform)",
    'TETRA': "Tetrahedron vertices",
    'OCTA': "Octahedron vertices",
    'CUBE': "Cube vertices",
    'ICOSA': "Icosahedron vertices",
    'DODECA': "Dodecahedron vertices",
}


def base_points(preset, n_lat, n_fiber, lat_min, lat_max):
    """Dispatch to a base-point provider; returns a list of unit
    3-vectors on S^2 (before the generic tilt is applied)."""
    if preset == 'LATITUDES':
        return latitudes(n_lat, n_fiber, lat_min, lat_max)
    if preset == 'FLOWER':
        return flower(n_fiber, 0.5 * (lat_min + lat_max))
    if preset == 'GREATCIRCLE':
        return great_circle(n_fiber, 0.5 * (lat_min + lat_max))
    if preset == 'FIBONACCI':
        return fibonacci(max(1, n_fiber))
    return _platonic(preset)


# --------------------------------------------------------------------------
# Assembly: project every fibre, tilt/centre/scale to fit the unit cube
# --------------------------------------------------------------------------

def build_fibers(preset='LATITUDES', n_lat=6, n_fiber=24, samples=160,
                 P=1, Q=1, lat_min=20.0, lat_max=160.0,
                 fit_radius=1.0, max_radius=12.0):
    """Return (fibers, bases): `fibers` is a list of (samples, 3)
    projected polylines and `bases` the matching tilted base points
    (for colouring).  The whole set is centred at the origin and
    uniformly scaled so its 95th-percentile point radius is
    `fit_radius`; any fibre whose points blow past `max_radius`
    (a base point near the south pole) is dropped."""
    import numpy as np
    R = _rot_matrix(*_TILT)
    raw = base_points(preset, n_lat, n_fiber, lat_min, lat_max)
    bases = [tuple(R @ np.asarray(b)) for b in raw]

    polylines = [project_fiber(b, samples, P, Q) for b in bases]

    # drop near-line fibres, then centre + scale on what remains
    kept = [(p, b) for p, b in zip(polylines, bases)
            if np.isfinite(p).all()
            and np.linalg.norm(p, axis=1).max() < max_radius]
    if not kept:
        return [], []
    fibers = [p for p, _ in kept]
    bases = [b for _, b in kept]

    allpts = np.concatenate(fibers, axis=0)
    center = 0.5 * (allpts.max(0) + allpts.min(0))
    rad = np.linalg.norm(allpts - center, axis=1)
    ref = np.percentile(rad, 95.0)
    scale = (fit_radius / ref) if ref > 1e-9 else 1.0
    fibers = [(p - center) * scale for p in fibers]
    return fibers, bases


def _fiber_color(base):
    """Standard sphere colouring of a base point: hue from longitude,
    value from latitude (north pole light, south pole dark)."""
    import colorsys
    a, b, c = base
    hue = (math.atan2(b, a) / (2.0 * pi)) % 1.0
    val = 0.35 + 0.6 * (0.5 * (c + 1.0))       # c in [-1,1] -> val
    return colorsys.hsv_to_rgb(hue, 0.72, val)


if _IN_BLENDER:

    class CURVE_OT_hopf_fibration_add(bpy.types.Operator):
        """Add fibres of the Hopf fibration of S^3, stereographically
        projected to R^3 as circles and coloured by base point"""
        bl_idname = "curve.hopf_fibration_add"
        bl_label = "Hopf Fibration"
        bl_options = {'REGISTER', 'UNDO'}

        preset: EnumProperty(
            name="Base Points",
            items=[(k, k.title(), v) for k, v in _PROVIDERS.items()],
            default='LATITUDES')
        n_lat: IntProperty(
            name="Latitudes / Rings", default=6, min=1, max=48,
            description="Circles of latitude (LATITUDES), or the "
                        "point budget for the Fibonacci sphere")
        n_fiber: IntProperty(
            name="Fibres", default=24, min=1, max=200,
            description="Fibres per latitude (LATITUDES), or the total "
                        "number of fibres for FLOWER / GREATCIRCLE / "
                        "FIBONACCI presets")
        samples: IntProperty(
            name="Samples", default=160, min=24, max=1024,
            description="Points per fibre")
        wind_p: IntProperty(
            name="Wind P", default=1, min=1, max=12,
            description="Winding rate of the first phase; (P,Q) != "
                        "(1,1) lays a (P,Q) torus knot/link on each "
                        "fibre's Clifford torus")
        wind_q: IntProperty(
            name="Wind Q", default=1, min=1, max=12,
            description="Winding rate of the second phase")
        lat_min: FloatProperty(
            name="Colat Min", default=20.0, min=1.0, max=179.0,
            description="Smallest colatitude (deg) for latitude / "
                        "flower / great-circle presets")
        lat_max: FloatProperty(
            name="Colat Max", default=160.0, min=1.0, max=179.0,
            description="Largest colatitude (deg)")
        output: EnumProperty(
            name="Output",
            items=[('BEZIER', "Bezier Curve", "auto-smoothed"),
                   ('POLY', "Poly Curve", ""),
                   ('NURBS', "NURBS Curve", ""),
                   ('MESH', "Mesh Tube", "swept tube mesh")],
            default='MESH')
        radius: FloatProperty(
            name="Tube Radius", default=0.02, min=0.0, max=1.0,
            step=1, precision=3,
            description="Curve bevel depth / tube radius")
        resolution: IntProperty(name="Bevel Resolution", default=4,
                                min=1, max=16)
        tube_sides: IntProperty(name="Tube Sides", default=8,
                                min=3, max=32)
        color_fibers: BoolProperty(
            name="Colour by Base Point", default=True,
            description="One material per fibre, hue from longitude "
                        "and value from latitude of its base point")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0)

        def execute(self, context):
            import numpy as np
            fibers, bases = build_fibers(
                self.preset, self.n_lat, self.n_fiber, self.samples,
                self.wind_p, self.wind_q, self.lat_min, self.lat_max)
            if not fibers:
                self.report({'ERROR'}, "No fibres produced")
                return {'CANCELLED'}
            fibers = [np.asarray(f) * self.scale for f in fibers]
            d = len(fibers)
            name = f"Hopf Fibration ({self.preset.title()})"
            color = self.color_fibers

            if self.output == 'MESH':
                try:
                    from . import prime_knot_generator as pk
                except ImportError:
                    import prime_knot_generator as pk
                tube = pk.CURVE_OT_prime_knot_add._tube
                verts, faces, midx = [], [], []
                for k, Pl in enumerate(fibers):
                    v, f = tube(Pl, self.radius, self.tube_sides)
                    base = len(verts)
                    verts.extend(v)
                    faces.extend([[base + i for i in face]
                                  for face in f])
                    midx.extend([k] * len(f))
                me = bpy.data.meshes.new(name)
                me.from_pydata(verts, [], faces)
                me.validate(clean_customdata=True)
                me.polygons.foreach_set(
                    'use_smooth', [True] * len(me.polygons))
                if color and len(me.polygons) == len(midx):
                    me.polygons.foreach_set('material_index', midx)
                me.update()
                obj = bpy.data.objects.new(name, me)
            else:
                cu = bpy.data.curves.new(name, 'CURVE')
                cu.dimensions = '3D'
                for Pl in fibers:
                    if self.output == 'BEZIER':
                        sp = cu.splines.new('BEZIER')
                        sp.bezier_points.add(len(Pl) - 1)
                        for i, pnt in enumerate(Pl):
                            bp = sp.bezier_points[i]
                            bp.co = pnt
                            bp.handle_left_type = 'AUTO'
                            bp.handle_right_type = 'AUTO'
                    else:
                        sp = cu.splines.new(self.output)
                        sp.points.add(len(Pl) - 1)
                        for i, pnt in enumerate(Pl):
                            sp.points[i].co = (pnt[0], pnt[1],
                                               pnt[2], 1.0)
                        if self.output == 'NURBS':
                            sp.order_u = 4
                    sp.use_cyclic_u = True
                cu.bevel_depth = self.radius
                cu.bevel_resolution = self.resolution
                obj = bpy.data.objects.new(name, cu)

            if color:
                for k in range(d):
                    rgb = _fiber_color(bases[k])
                    mat = bpy.data.materials.new(f"{name} F{k + 1}")
                    mat.diffuse_color = (*rgb, 1.0)
                    mat.use_nodes = True
                    node = mat.node_tree.nodes.get("Principled BSDF")
                    if node:
                        node.inputs["Base Color"].default_value = \
                            (*rgb, 1.0)
                    obj.data.materials.append(mat)
                if self.output != 'MESH':
                    for k, sp in enumerate(obj.data.splines):
                        sp.material_index = k

            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report(
                {'INFO'},
                f"{name}: {d} fibres, {self.samples} samples each"
                + ("" if (self.wind_p, self.wind_q) == (1, 1)
                   else f", ({self.wind_p},{self.wind_q}) winding"))
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'preset')
            if self.preset == 'LATITUDES':
                lay.prop(self, 'n_lat')
            if self.preset in ('LATITUDES', 'FLOWER', 'GREATCIRCLE',
                               'FIBONACCI'):
                lay.prop(self, 'n_fiber')
            lay.prop(self, 'samples')
            row = lay.row(align=True)
            row.prop(self, 'wind_p')
            row.prop(self, 'wind_q')
            if self.preset in ('LATITUDES', 'FLOWER', 'GREATCIRCLE'):
                lay.prop(self, 'lat_min')
                lay.prop(self, 'lat_max')
            lay.prop(self, 'output')
            lay.prop(self, 'radius')
            if self.output == 'MESH':
                lay.prop(self, 'tube_sides')
            elif self.radius > 0:
                lay.prop(self, 'resolution')
            lay.prop(self, 'color_fibers')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("curve.hopf_fibration_add",
                             icon='FORCE_MAGNETIC')

    ADD_MENU = True   # the Math Art extension menu sets this False

    def register():
        bpy.utils.register_class(CURVE_OT_hopf_fibration_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_curve_add.remove(_menu_func)
        bpy.utils.unregister_class(CURVE_OT_hopf_fibration_add)


def _selftest():
    import numpy as np
    ok_all = True

    # 1) lifted fibre lies on S^3 and its Hopf image is constant (= base)
    def hopf(X):
        a, b, c, d = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        z0z1_re = 2.0 * (a * c + b * d)          # Re(2 z0 conj z1)
        z0z1_im = 2.0 * (b * c - a * d)          # Im(2 z0 conj z1)
        return np.stack([z0z1_re, z0z1_im,
                         a * a + b * b - c * c - d * d], axis=1)

    for base in [(0.3, 0.5, 0.8), (0.0, 0.0, 1.0), (-0.6, 0.4, -0.7)]:
        b = _normalize(base)
        X = fiber_s3(b, 128)
        on_s3 = abs(np.linalg.norm(X, axis=1) - 1.0).max()
        img = hopf(X)
        img_dev = np.linalg.norm(img - img.mean(0), axis=1).max()
        base_err = np.linalg.norm(img.mean(0) - b)
        ok = on_s3 < 1e-12 and img_dev < 1e-9 and base_err < 1e-9
        ok_all = ok_all and ok
        print(f"base={tuple(round(x,2) for x in b)}: |S3 dev|="
              f"{on_s3:.1e} img const={img_dev:.1e} "
              f"base err={base_err:.1e} {'OK' if ok else 'BAD'}")

    # 2) a projected fibre is a genuine circle: coplanar, and an
    #    algebraic circle fit in that plane leaves ~0 radial residual.
    #    (Stereographic projection spaces the samples non-uniformly, so
    #    the circle's centre is NOT the point centroid -- fit it.)
    fib = project_fiber(_normalize((0.4, 0.2, 0.6)), 200)
    ctr = fib.mean(0)
    U, S, Vt = np.linalg.svd(fib - ctr)
    coplanar = S[2] / S[0]                        # ~0 if planar
    u = (fib - ctr) @ Vt[0]                       # in-plane coords
    v = (fib - ctr) @ Vt[1]
    A = np.stack([2 * u, 2 * v, np.ones_like(u)], axis=1)
    sol, *_ = np.linalg.lstsq(A, u * u + v * v, rcond=None)
    cu_, cv_, cc = sol
    r = np.sqrt(np.maximum(0.0, (u - cu_) ** 2 + (v - cv_) ** 2))
    circ = (r.max() - r.min()) / r.mean()
    ok = coplanar < 1e-6 and circ < 1e-6
    ok_all = ok_all and ok
    print(f"projected fibre: coplanarity={coplanar:.1e} "
          f"circularity={circ:.1e} {'OK' if ok else 'BAD'}")

    # 3) every base-point provider yields unit vectors; assembly builds
    for preset in _PROVIDERS:
        pts = base_points(preset, 4, 12, 20.0, 160.0)
        unit = max(abs(np.linalg.norm(p) - 1.0) for p in pts)
        fibers, bases = build_fibers(preset, 3, 12, 60)
        ok = unit < 1e-9 and len(fibers) > 0 and len(fibers) == len(bases)
        ok_all = ok_all and ok
        print(f"{preset}: {len(pts)} base pts unit dev={unit:.1e} "
              f"-> {len(fibers)} fibres {'OK' if ok else 'BAD'}")

    assert ok_all
    print("hopf fibration standalone tests passed")
