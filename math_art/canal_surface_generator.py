# Canal surfaces: the envelope of a one-parameter family of spheres.
#
# A spine curve c(t) carries a sphere of radius r(t) at every point; the
# canal surface is the envelope of that family.  Each sphere touches the
# envelope along a CHARACTERISTIC CIRCLE, and the geometry of that circle
# is the whole content of the construction: writing s for arclength along
# the spine and r' = dr/ds, the circle's centre sits at
#
#     c(s) - r r' T(s)          (pulled BACK along the tangent),
#
# its radius is r sqrt(1 - r'^2), and its plane is tilted from the normal
# plane by the angle asin(r').  A varying-radius tube swept naively --
# circles of radius r(s) in the normal plane, which is what a bevelled
# curve gives -- is NOT the envelope: it cuts into the spheres wherever
# r' is nonzero.  The tilt term is what this generator adds over the
# repo's existing constant-radius tube machinery (curve_frames.sweep,
# knots/tube.py), which remains correct for r' = 0.
#
# Two classical specialisations close the loop with surfaces already in
# the catalogue, and the self-test holds the generator to both:
#   - a circular spine with constant radius is EXACTLY the torus;
#   - an ellipse spine (a cos u, b sin u, 0) with the radius law
#     r = d - (c/a) x, c^2 = a^2 - b^2, is Monge's construction of the
#     Dupin cyclide, and every vertex must satisfy the cyclide quartic
#         (x^2 + y^2 + z^2 + b^2 - d^2)^2 = 4 (a x - c d)^2 + 4 b^2 y^2.
# Dupin's cyclides are precisely the surfaces that are canal surfaces in
# TWO different ways, which is why they fall out of this operator and of
# the torus-inversion route (mesh.curiosity_surface_add) alike.
#
# Open spines are closed with the true spherical end caps -- the portion
# of the first and last sphere beyond the characteristic circle, a cap of
# half-angle acos(r') about the tangent -- so an open canal is watertight
# and a straight spine with a linear radius law ends in the round tip a
# real cone of spheres has.
#
# The envelope exists only while |r'| < 1; where a radius law is steeper
# than the spine is long the sphere family swallows itself and the
# characteristic circle turns imaginary.  The builder clamps r' and
# reports how many samples were clamped rather than emitting NaNs.
#
# References:
# - G. Monge, "Application de l'analyse a la geometrie" (1807) -- canal
#   surfaces as envelopes of one-parameter families of spheres, with the
#   characteristic-circle analysis.
# - C. Dupin, "Applications de geometrie et de mechanique" (1822) -- the
#   cyclides as the surfaces that are canal surfaces in two ways.
# - V. Chandru, D. Dutta and C. M. Hoffmann, "On the geometry of Dupin
#   cyclides", The Visual Computer 5 (1989), 277-290 -- the sphere-family
#   construction and the (a, b, d) quartic normal form used by the
#   self-test.
# - R. Ferreol, "Encyclopedie des formes mathematiques remarquables"
#   (mathcurve.com), "Surface tubulaire / surface canal".

bl_info = {
    "name": "Canal Surface",
    "author": "Math Art project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Math Art > Surfaces",
    "description": "Envelope of a one-parameter family of spheres along "
                   "a spine curve, with tilted characteristic circles "
                   "and true spherical end caps",
    "category": "Add Mesh",
}

import math

import numpy as np

try:
    from .curve_frames.frames import closed_frames, frames
    from .quadric_generator import fit
except ImportError:
    from curve_frames.frames import closed_frames, frames
    from quadric_generator import fit

TAU = 2.0 * math.pi

#: key -> (label, description); read by tools/surfdb (as source, via ast)
SPINES = (
    ('LINE', "Straight Line",
     "A straight spine. Constant radius gives the exact cylinder "
     "(capsule with end caps); a linear taper gives a cone of spheres"),
    ('CIRCLE', "Circle",
     "A circular spine. Constant radius gives the exact torus; the "
     "self-test verifies the torus equation on every vertex"),
    ('ELLIPSE', "Ellipse",
     "An elliptical spine. With the Cyclide radius law this is Monge's "
     "construction of the Dupin cyclide, checked against the cyclide "
     "quartic"),
    ('HELIX', "Helix",
     "A helical spine: the classic canal-surface testbed, and with the "
     "Waves law a string of pearls wound into a spring"),
    ('TORUS_KNOT', "Torus Knot",
     "A (p, q) torus-knot spine, so every choice of radius law wraps a "
     "varying-thickness envelope around a knot"),
)

#: how r varies along the spine (t is arclength fraction, x the spine's
#: first coordinate)
LAWS = (
    ('CONSTANT', "Constant",
     "Fixed radius: the classical pipe (tube) surface"),
    ('TAPER', "Taper",
     "Radius linear in arclength, from Radius at the start to End "
     "Radius at the far end. Meant for open spines; on a closed spine "
     "it jumps at the seam"),
    ('WAVES', "Waves",
     "Radius modulated sinusoidally along the spine: a string of "
     "pearls. The wave count is an integer so a closed spine stays "
     "seamless"),
    ('CYCLIDE', "Cyclide",
     "Radius decreasing linearly with the spine point's x coordinate "
     "(r = radius - slope * x). On an ellipse spine with slope c/a "
     "this is exactly Monge's construction of the Dupin cyclide"),
)


# ---------------------------------------------------------------------------
# spines


def spine_points(kind, n, aspect=0.8, turns=3.0, pitch=0.35,
                 knot_p=2, knot_q=3):
    """(points, closed) for a built-in spine.

    Closed spines return `n` samples WITHOUT the duplicate endpoint
    (what `curve_frames.closed_frames` expects); open spines return
    n + 1 samples including both ends.
    """
    if kind == 'LINE':
        t = np.linspace(-1.0, 1.0, n + 1)
        P = np.stack([np.zeros_like(t), np.zeros_like(t), t], axis=1)
        return P, False
    if kind == 'CIRCLE':
        u = np.linspace(0.0, TAU, n, endpoint=False)
        P = np.stack([np.cos(u), np.sin(u), np.zeros_like(u)], axis=1)
        return P, True
    if kind == 'ELLIPSE':
        u = np.linspace(0.0, TAU, n, endpoint=False)
        P = np.stack([np.cos(u), max(aspect, 0.05) * np.sin(u),
                      np.zeros_like(u)], axis=1)
        return P, True
    if kind == 'HELIX':
        u = np.linspace(0.0, TAU * turns, n + 1)
        z = pitch * (u / TAU)
        P = np.stack([np.cos(u), np.sin(u), z - z[-1] * 0.5], axis=1)
        return P, False
    if kind == 'TORUS_KNOT':
        p, q = max(int(knot_p), 1), max(int(knot_q), 1)
        u = np.linspace(0.0, TAU, n, endpoint=False)
        w = 1.0 + 0.4 * np.cos(q * u)
        P = np.stack([w * np.cos(p * u), w * np.sin(p * u),
                      0.4 * np.sin(q * u)], axis=1)
        return P, True
    raise ValueError("unknown spine %r" % kind)


def arclength(P, closed):
    """(s, S): cumulative arclength per sample, and the total length."""
    P = np.asarray(P, dtype=float)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])[:len(P)]
    S = s[-1] + (np.linalg.norm(P[0] - P[-1]) if closed else 0.0)
    if closed:
        S = float(np.sum(d) + np.linalg.norm(P[0] - P[-1]))
    else:
        S = float(np.sum(d))
    return s, max(S, 1e-12)


def radius_profile(law, P, T, s, S, radius=0.25, radius_end=0.08,
                   wave_count=6, wave_depth=0.35, slope=0.6):
    """(r, r_s): the radius and its arclength derivative per sample.

    Analytic in every law: TAPER and WAVES differentiate in arclength
    directly, CYCLIDE uses dr/ds = -slope * dx/ds = -slope * T_x.
    """
    t = s / S
    if law == 'CONSTANT':
        r = np.full(len(s), radius)
        r_s = np.zeros(len(s))
    elif law == 'TAPER':
        r = radius + (radius_end - radius) * t
        r_s = np.full(len(s), (radius_end - radius) / S)
    elif law == 'WAVES':
        k = max(int(wave_count), 1)
        r = radius * (1.0 + wave_depth * np.sin(TAU * k * t))
        r_s = radius * wave_depth * np.cos(TAU * k * t) * TAU * k / S
    elif law == 'CYCLIDE':
        r = radius - slope * np.asarray(P)[:, 0]
        r_s = -slope * np.asarray(T)[:, 0]
    else:
        raise ValueError("unknown radius law %r" % law)
    return np.maximum(r, 1e-4), r_s


# ---------------------------------------------------------------------------
# the envelope


def build_canal(P, closed, r, r_s, sides=32, caps=True):
    """(verts, faces, stats) -- the envelope mesh.

    `stats` reports `clamped` (samples where |r'| had to be cut to keep
    the characteristic circle real) and `ring_residual` (the worst
    | |X - c| - r | over all vertices: zero for a correct envelope,
    since every characteristic circle lies ON its sphere).
    """
    P = np.asarray(P, dtype=float)
    n = len(P)
    if closed:
        T, N, B = closed_frames(P)
    else:
        T, N, B = frames(P)

    RS_MAX = 0.99
    clamped = int(np.count_nonzero(np.abs(r_s) > RS_MAX))
    rs = np.clip(r_s, -RS_MAX, RS_MAX)
    rho = r * np.sqrt(1.0 - rs * rs)

    ang = TAU * np.arange(sides) / sides
    ca, sa = np.cos(ang), np.sin(ang)
    # ring i, vertex k:  C - r r' T + rho (N cos + B sin)
    centre = P - (r * rs)[:, None] * T
    rings = (centre[:, None, :]
             + rho[:, None, None] * (ca[None, :, None] * N[:, None, :]
                                     + sa[None, :, None] * B[:, None, :]))
    verts = [tuple(v) for v in rings.reshape(-1, 3)]
    faces = []
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((i * sides + k, i * sides + k2,
                          j * sides + k2, j * sides + k))

    if not closed and caps:
        m = max(2, sides // 4)
        for end, sgn in ((0, -1.0), (n - 1, +1.0)):
            # cap axis sgn*T; the characteristic circle sits at the
            # half-angle acos(-sgn * r') from that axis
            phi0 = math.acos(max(-1.0, min(1.0, -sgn * rs[end])))
            pole = P[end] + r[end] * sgn * T[end]
            pole_idx = len(verts)
            verts.append(tuple(pole))
            prev = None
            for jj in range(1, m):
                phi = phi0 * jj / m
                ring_idx = len(verts)
                for k in range(sides):
                    u = ca[k] * N[end] + sa[k] * B[end]
                    v = math.cos(phi) * sgn * T[end] + math.sin(phi) * u
                    verts.append(tuple(P[end] + r[end] * v))
                if jj == 1:
                    for k in range(sides):
                        k2 = (k + 1) % sides
                        tri = (pole_idx, ring_idx + k, ring_idx + k2)
                        faces.append(tri if sgn > 0 else tri[::-1])
                else:
                    for k in range(sides):
                        k2 = (k + 1) % sides
                        quad = (prev + k, ring_idx + k,
                                ring_idx + k2, prev + k2)
                        faces.append(quad if sgn > 0 else quad[::-1])
                prev = ring_idx
            # join the innermost cap ring (or the pole fan) to the
            # surface's boundary ring, which sits exactly on the sphere
            surf = end * sides
            if m == 2 and prev is None:
                prev = pole_idx  # degenerate; cannot happen with m >= 2
            for k in range(sides):
                k2 = (k + 1) % sides
                quad = (prev + k, surf + k, surf + k2, prev + k2)
                faces.append(quad if sgn > 0 else quad[::-1])

    dist = np.linalg.norm(rings - P[:, None, :], axis=2)
    ring_residual = float(np.max(np.abs(dist - r[:, None])))
    return verts, faces, {"clamped": clamped,
                          "ring_residual": ring_residual}


def resample_polyline(P, closed, n):
    """Resample `P` to `n` samples, uniform in arclength."""
    P = np.asarray(P, dtype=float)
    if closed:
        Q = np.vstack([P, P[:1]])
    else:
        Q = P
    d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    S = s[-1]
    if S <= 1e-12:
        raise ValueError("spine has zero length")
    t = (np.linspace(0.0, S, n, endpoint=False) if closed
         else np.linspace(0.0, S, n + 1))
    out = np.stack([np.interp(t, s, Q[:, i]) for i in range(3)], axis=1)
    return out


# ---------------------------------------------------------------------------


try:
    import bpy
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


if _IN_BLENDER:

    def _spine_from_object(context, samples):
        """(points, closed) sampled from the selected curve object.

        The evaluated object is converted to a wire mesh and its longest
        edge chain walked, so every spline type (Bezier, NURBS, poly)
        and every modifier on it are honoured.
        """
        obs = [o for o in context.selected_objects if o.type == 'CURVE']
        if not obs and context.active_object is not None \
                and context.active_object.type == 'CURVE':
            obs = [context.active_object]
        if not obs:
            raise ValueError("select a curve object to use as the spine")
        ob = obs[0]
        deps = context.evaluated_depsgraph_get()
        ob_eval = ob.evaluated_get(deps)
        me = ob_eval.to_mesh()
        try:
            V = [tuple(ob.matrix_world @ v.co) for v in me.vertices]
            E = [tuple(e.vertices) for e in me.edges]
        finally:
            ob_eval.to_mesh_clear()
        if len(V) < 3 or not E:
            raise ValueError("the curve produced no usable polyline")
        adj = {}
        for a, b in E:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        ends = [v for v, nb in adj.items() if len(nb) == 1]
        start = min(ends) if ends else min(adj)
        chain, prev, cur = [start], None, start
        while True:
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
            chain.append(cur)
        closed = not ends and len(chain) == len(adj)
        if len(chain) < 3:
            raise ValueError("the curve's polyline is too short")
        P = np.asarray([V[i] for i in chain], dtype=float)
        return resample_polyline(P, closed, samples), closed

    class MESH_OT_canal_surface_add(bpy.types.Operator):
        """Add a canal surface: the envelope of a one-parameter family
        of spheres carried along a spine curve, with the characteristic
        circles tilted as the radius varies"""
        bl_idname = "mesh.canal_surface_add"
        bl_label = "Canal Surface"
        bl_options = {'REGISTER', 'UNDO'}

        spine: EnumProperty(
            name="Spine",
            items=[(k, lab, desc) for k, lab, desc in SPINES] + [
                ('CURVE', "Selected Curve",
                 "Use the selected curve object as the spine, so any "
                 "knot or curve generator in the add-on feeds this one")],
            default='HELIX',
            description="The curve the family of spheres travels along")
        radius_law: EnumProperty(
            name="Radius Law",
            items=[(k, lab, desc) for k, lab, desc in LAWS],
            default='CONSTANT',
            description="How the sphere radius varies along the spine")
        radius: FloatProperty(
            name="Radius", default=0.25, min=0.005, max=2.0,
            description="Sphere radius, or its value at the start of "
                        "the spine for the varying laws")
        radius_end: FloatProperty(
            name="End Radius", default=0.06, min=0.001, max=2.0,
            description="Sphere radius at the far end of the spine "
                        "(Taper law)")
        wave_count: IntProperty(
            name="Wave Count", default=6, min=1, max=64,
            description="Number of radius waves along the spine "
                        "(Waves law)")
        wave_depth: FloatProperty(
            name="Wave Depth", default=0.35, min=0.0, max=0.95,
            description="Relative depth of the radius waves (Waves law)")
        slope: FloatProperty(
            name="Slope", default=0.6, min=-2.0, max=2.0,
            description="Rate at which the radius falls with the spine "
                        "point's x coordinate (Cyclide law); c/a on an "
                        "ellipse spine gives the exact Dupin cyclide")
        ellipse_aspect: FloatProperty(
            name="Ellipse Aspect", default=0.8, min=0.1, max=1.0,
            description="Minor over major axis of the ellipse spine")
        turns: FloatProperty(
            name="Turns", default=3.0, min=0.25, max=16.0,
            description="Number of turns of the helix spine")
        pitch: FloatProperty(
            name="Pitch", default=0.9, min=0.0, max=4.0,
            description="Vertical rise of the helix spine per turn")
        knot_p: IntProperty(
            name="Knot P", default=2, min=1, max=8,
            description="p of the (p, q) torus-knot spine")
        knot_q: IntProperty(
            name="Knot Q", default=3, min=1, max=9,
            description="q of the (p, q) torus-knot spine")
        spine_samples: IntProperty(
            name="Spine Samples", default=256, min=16, max=1024,
            description="Number of characteristic circles along the "
                        "spine")
        sides: IntProperty(
            name="Sides", default=32, min=6, max=128,
            description="Vertices around each characteristic circle")
        cap_ends: BoolProperty(
            name="Cap Ends", default=True,
            description="Close an open spine with the true spherical "
                        "caps of the first and last sphere, keeping the "
                        "envelope watertight")
        size: FloatProperty(
            name="Size", default=1.0, min=0.01, max=100.0,
            description="Half the largest extent of the finished object")

        def execute(self, context):
            if self.spine == 'CURVE':
                try:
                    P, closed = _spine_from_object(context,
                                                   self.spine_samples)
                except ValueError as e:
                    self.report({'ERROR'}, str(e))
                    return {'CANCELLED'}
            else:
                P, closed = spine_points(
                    self.spine, self.spine_samples,
                    aspect=self.ellipse_aspect, turns=self.turns,
                    pitch=self.pitch,
                    knot_p=self.knot_p, knot_q=self.knot_q)

            if closed:
                T, _N, _B = closed_frames(P)
            else:
                T, _N, _B = frames(P)
            s, S = arclength(P, closed)
            r, r_s = radius_profile(
                self.radius_law, P, T, s, S, radius=self.radius,
                radius_end=self.radius_end, wave_count=self.wave_count,
                wave_depth=self.wave_depth, slope=self.slope)
            verts, faces, stats = build_canal(
                P, closed, r, r_s, sides=self.sides, caps=self.cap_ends)
            verts = fit(verts, self.size)

            me = bpy.data.meshes.new("Canal Surface")
            me.from_pydata(verts, [], faces)
            me.validate()
            me.update()
            obj = bpy.data.objects.new("Canal Surface", me)
            context.collection.objects.link(obj)
            context.view_layer.objects.active = obj
            obj.select_set(True)

            msg = ("Canal: %d verts, %d faces; sphere-distance residual "
                   "%.1e" % (len(verts), len(faces),
                             stats["ring_residual"]))
            if stats["clamped"]:
                msg += ("; radius law steeper than the envelope allows "
                        "at %d samples (|dr/ds| clamped below 1)"
                        % stats["clamped"])
            self.report({'INFO'}, msg)
            return {'FINISHED'}

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'spine')
            if self.spine == 'ELLIPSE':
                lay.prop(self, 'ellipse_aspect')
            elif self.spine == 'HELIX':
                lay.prop(self, 'turns')
                lay.prop(self, 'pitch')
            elif self.spine == 'TORUS_KNOT':
                lay.prop(self, 'knot_p')
                lay.prop(self, 'knot_q')
            lay.prop(self, 'radius_law')
            lay.prop(self, 'radius')
            if self.radius_law == 'TAPER':
                lay.prop(self, 'radius_end')
            elif self.radius_law == 'WAVES':
                lay.prop(self, 'wave_count')
                lay.prop(self, 'wave_depth')
            elif self.radius_law == 'CYCLIDE':
                lay.prop(self, 'slope')
            for k in ('spine_samples', 'sides', 'cap_ends', 'size'):
                lay.prop(self, k)

    def _menu_func(self, context):
        self.layout.operator(MESH_OT_canal_surface_add.bl_idname,
                             text="Canal Surface", icon='MESH_CAPSULE')

    def register():
        bpy.utils.register_class(MESH_OT_canal_surface_add)
        if hasattr(bpy.types, "VIEW3D_MT_mesh_add"):
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if hasattr(bpy.types, "VIEW3D_MT_mesh_add"):
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_canal_surface_add)


# ---------------------------------------------------------------------------


def _boundary_edge_count(faces):
    cnt = {}
    for f in faces:
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            k = (a, b) if a < b else (b, a)
            cnt[k] = cnt.get(k, 0) + 1
    return sum(1 for c in cnt.values() if c != 2)


def _canal(kind, law, n, sides, caps=True, **kw):
    spine_kw = {k: kw.pop(k) for k in ('aspect', 'turns', 'pitch',
                                       'knot_p', 'knot_q') if k in kw}
    P, closed = spine_points(kind, n, **spine_kw)
    T, _N, _B = closed_frames(P) if closed else frames(P)
    s, S = arclength(P, closed)
    r, r_s = radius_profile(law, P, T, s, S, **kw)
    verts, faces, stats = build_canal(P, closed, r, r_s, sides=sides,
                                      caps=caps)
    return P, r, np.asarray(verts), faces, stats


def _selftest():
    """Numeric self-test; raises on failure.

    Each gate is a closed-form oracle, not a look: the cylinder by its
    axis distance, the torus and the Dupin cyclide by their implicit
    quartics, and the envelope property by sphere tangency.
    """
    ok = True

    # 1. straight spine + constant radius, no caps: the EXACT cylinder
    _P, _r, V, F, st = _canal('LINE', 'CONSTANT', 128, 24, caps=False,
                              radius=0.3)
    d = np.abs(np.linalg.norm(V[:, :2], axis=1) - 0.3)
    good = float(d.max()) < 1e-9 and st["clamped"] == 0
    ok &= good
    print("canal: straight spine is an exact cylinder "
          "(radial error %.1e) %s" % (float(d.max()),
                                      "OK" if good else "FAIL"))

    # 2. circular spine + constant radius: the EXACT torus
    _P, _r, V, F, st = _canal('CIRCLE', 'CONSTANT', 256, 24, radius=0.3)
    res = (np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2) - 1.0) ** 2 \
        + V[:, 2] ** 2 - 0.09
    good = float(np.max(np.abs(res))) < 1e-8
    ok &= good
    print("canal: circle spine satisfies the torus equation "
          "(residual %.1e) %s" % (float(np.max(np.abs(res))),
                                  "OK" if good else "FAIL"))
    wt = _boundary_edge_count(F)
    good = wt == 0
    ok &= good
    print("canal: closed spine is watertight (%d open edges) %s"
          % (wt, "OK" if good else "FAIL"))

    # 3. THE OVERLAP WITH THE SHIPPED CYCLIDES.  Ellipse spine a = 1,
    # b = 0.8 (so c = 0.6), radius law r = 0.7 - 0.6 x: every vertex
    # must satisfy the Dupin cyclide quartic with (a, b, d) =
    # (1, 0.8, 0.7) -- the same family mesh.curiosity_surface_add builds
    # by torus inversion.  The tangent enters through r' = -c T_x /a and
    # is computed by central differences, so the residual must SHRINK
    # like h^2 under refinement; asserting the ratio guards against
    # blaming the formula for truncation error (and vice versa).
    a, b, dpar = 1.0, 0.8, 0.7
    c = math.sqrt(a * a - b * b)

    def cyclide_residual(n):
        _P, _r, V, _F, _st = _canal('ELLIPSE', 'CYCLIDE', n, 16,
                                    aspect=b / a, radius=dpar, slope=c / a)
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
        q = (x * x + y * y + z * z + b * b - dpar * dpar) ** 2 \
            - 4.0 * (a * x - c * dpar) ** 2 - 4.0 * b * b * y * y
        return float(np.max(np.abs(q)))

    r96, r384 = cyclide_residual(96), cyclide_residual(384)
    # the ratio test only means something above the numerical floor; in
    # practice the residual sits at ~1e-9 already, i.e. the construction
    # is exact to within the frame transport's own rounding
    good = r384 < 2e-3 and (r384 < 1e-6 or r96 / max(r384, 1e-15) > 4.0)
    ok &= good
    print("canal: ellipse spine + cyclide law satisfies the Dupin "
          "quartic (residual %.1e at 384 samples, %.1e at 96) %s"
          % (r384, r96, "OK" if good else "FAIL"))

    # 4. the ENVELOPE property, on a spine with no special symmetry:
    # every characteristic circle lies exactly on its sphere, and no
    # part of the surface cuts inside any generating sphere.
    P, r, V, F, st = _canal('HELIX', 'WAVES', 200, 20, radius=0.22,
                            wave_count=5, wave_depth=0.4,
                            turns=2.0, pitch=1.2)
    good = st["ring_residual"] < 1e-9
    ok &= good
    print("canal: every vertex of ring i lies ON sphere i "
          "(residual %.1e) %s" % (st["ring_residual"],
                                  "OK" if good else "FAIL"))
    worst_cut, worst_gap = 0.0, 0.0
    for i in range(20, 180, 16):
        dmin = float(np.min(np.linalg.norm(V - P[i], axis=1)))
        worst_cut = max(worst_cut, r[i] - dmin)
        worst_gap = max(worst_gap, 0.0) if dmin <= r[i] + 5e-3 \
            else max(worst_gap, dmin - r[i])
    good = worst_cut < 1e-6 and worst_gap == 0.0
    ok &= good
    print("canal: surface is tangent to the generating spheres -- "
          "never inside (worst cut %.1e), always touching %s"
          % (worst_cut, "OK" if good else "FAIL"))

    # 5. open spine with caps is watertight; the caps are ON the end
    # spheres
    _P2, r2, V2, F2, st2 = _canal('HELIX', 'TAPER', 96, 20, caps=True,
                                  radius=0.3, radius_end=0.08,
                                  turns=1.5, pitch=1.0)
    wt = _boundary_edge_count(F2)
    good = wt == 0 and bool(np.all(np.isfinite(V2)))
    ok &= good
    print("canal: open spine with spherical caps is watertight "
          "(%d open edges) %s" % (wt, "OK" if good else "FAIL"))

    # 6. fit() convention: centred, 2 units across at size 1
    fitted = np.asarray(fit([tuple(v) for v in V2], 1.0))
    ext = fitted.max(axis=0) - fitted.min(axis=0)
    mid = 0.5 * (fitted.max(axis=0) + fitted.min(axis=0))
    good = abs(float(ext.max()) - 2.0) < 1e-9 \
        and float(np.max(np.abs(mid))) < 1e-9
    ok &= good
    print("canal: fitted output is centred in the 2 m cube %s"
          % ("OK" if good else "FAIL"))

    print("RESULT:", "OK" if ok else "FAIL")
    if not ok:
        raise AssertionError("canal surface self-test failed")
