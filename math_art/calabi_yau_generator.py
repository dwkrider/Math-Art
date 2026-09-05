
# Calabi-Yau Cross-Section Generator for Blender
#
# The object every "Calabi-Yau manifold" picture actually shows.  A
# Calabi-Yau threefold has SIX real dimensions and does not fit in
# R^3; what is drawn is a two-real-dimensional CROSS-SECTION of one,
# projected from R^4 to R^3.  Dehomogenising the quintic threefold
#
#     z0^5 + z1^5 + z2^5 + z3^5 + z4^5 = 0     in CP^4
#
# and holding two of the coordinates fixed leaves the complex curve
# z1^5 + z2^5 = 1 in C^2 = R^4 -- a real 2-manifold.  Hanson (1994)
# parametrised it by extending the superquadric trick to the complex
# domain: with the complex sine and cosine
#
#     u1(t,x) = cosh(x + i t),   u2(t,x) = -i sinh(x + i t)
#
# obeying u1^2 + u2^2 = 1, the n^2 maps
#
#     z1 = e^{2 pi i k1 / n} u1^{2/n},   z2 = e^{2 pi i k2 / n} u2^{2/n}
#
# for 0 <= t <= pi/2, |x| <= x_max satisfy z1^n + z2^n = 1 identically,
# and piece together into the surface.  The 4D -> 3D projection is
#
#     (Re z1, Re z2, cos a Im z1 + sin a Im z2)
#
# with the discarded fourth coordinate sin a Im z1 - cos a Im z2 kept
# here as a mesh attribute, since it is the only trace the projection
# leaves of the dimension it threw away.
#
# WHAT THE SURFACE IS.  z1^p + z2^q = 1 is the Milnor fibre of the
# Brieskorn-Pham singularity z1^p + z2^q at the origin, so the
# truncated surface is a genus-minimising Seifert surface of the (p,q)
# TORUS LINK: Euler characteristic 1 - (p-1)(q-1), gcd(p,q) boundary
# circles, genus ((p-1)(q-1) - gcd(p,q) + 1)/2.  For p = q = 5 that is
# genus 6 with five boundary circles, chi = -15; for (2,3) it is the
# familiar once-punctured torus spanning the trefoil.  Both counts are
# checked against the finished mesh in `_selftest`, the same way
# `seifert_surface_generator` checks its braid closures.
#
# A NOTE ON THE PHASE CONVENTION.  Hanson puts the factor -i INSIDE
# the 2/n power (see the Mathematica listing in Table 1 of the 1994
# paper: `cSin[theta_,xi_] := (-.5 I)(E^(xi + I theta) - E^(-xi - I
# theta))`, then `cSin[...]^(2.0/n)`).  Widely copied MATLAB
# transcriptions instead write `(1/i) * sinh(z)^(2/n)`, with the phase
# OUTSIDE the power; that only satisfies z1^n + z2^n = 1 for n = 2.
# The residual max |z1^p + z2^q - e^{2 pi i s}| over the whole sample
# grid is asserted below, which catches exactly this mistake.
#
# References:
# - A. J. Hanson, "A Construction for Computer Visualization of
#   Certain Complex Curves", Notices of the American Mathematical
#   Society 41, no. 9 (1994) 1156-1163 -- the parametrisation, the
#   n^2-patch structure, the 4D -> 3D projection, the genus formula
#   g = (n-1)(n-2)/2, the n asymptotic boundary circles, and the
#   torus-knot generalisation z1^n1 + z2^n2 = 1 of its Eqs. (10)-(14).
# - A. H. Barr, "Superquadrics and Angle-Preserving Transformations",
#   IEEE Computer Graphics and Applications 1 (1981) 11-23 -- the real
#   superquadric construction Hanson complexifies.
# - J. Milnor, "Singular Points of Complex Hypersurfaces", Annals of
#   Mathematics Studies 61, Princeton (1968) -- the Milnor fibration,
#   whose fibre for z1^p + z2^q is the surface built here and whose
#   boundary is the (p,q) torus link.
# - E. Brieskorn, "Beispiele zur Differentialtopologie von
#   Singularitaeten", Inventiones Mathematicae 2 (1966) 1-14 -- the
#   Brieskorn-Pham singularities z1^p + z2^q.
# - P. Candelas, G. Horowitz, A. Strominger and E. Witten, "Vacuum
#   configurations for superstrings", Nuclear Physics B258 (1985)
#   46-74 -- why the quintic threefold is the standard example.

import cmath
import itertools
import math
from collections import defaultdict

import numpy as np

_OFF4 = tuple(itertools.product((0, -1, 1), repeat=4))

bl_info = {
    "name": "Calabi-Yau Cross-Section",
    "author": "Math Art project (after Hanson 1994)",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Add > Mesh > Calabi-Yau Cross-Section",
    "description": "Hanson's cross-section of the quintic; Milnor "
                   "fibres of Brieskorn singularities",
    "category": "Add Mesh",
}

try:
    import bpy
    import bmesh
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                           IntProperty)
    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False


# --------------------------------------------------------------------
# The complex charts
# --------------------------------------------------------------------

def _grid(theta_steps, xi_steps, xi_max):
    """Sample grids for the chart, exact where exactness matters.

    Two floating-point details decide whether the patches weld into
    one surface at all, and both are handled here rather than left to
    the library:

    * cos(pi/2) evaluates to 6.1e-17, not 0.  At theta = pi/2 the
      chart wants u1 = 0 exactly, because there z1 = 0 for EVERY k1
      and the p patches sharing a k2 have to meet at that one point.
      Left alone, u1^(2/p) turns 6.1e-17 into 1.3e-7 for p = 5 -- big
      enough that the fixed point splits into p separate vertices and
      the boundary comes out with 2 gcd(p,q) circles instead of
      gcd(p,q).  So cos and sin are pinned at both ends of the range.
    * the xi grid is built symmetrically from one half, so that
      xi = 0 is hit exactly and xi = -t is the exact negative of
      xi = +t.  The seam identification runs xi -> -xi, and only
      exact symmetry lets the two halves weld.
    """
    th = np.linspace(0.0, 0.5 * math.pi, theta_steps)
    c, s = np.cos(th), np.sin(th)
    c[0], s[0] = 1.0, 0.0
    c[-1], s[-1] = 0.0, 1.0
    half = (xi_steps - 1) // 2
    t = np.arange(half + 1, dtype=float) / max(half, 1) * xi_max
    xi = np.concatenate((-t[::-1], t[1:]))
    return c, s, xi


def _u1_u2(theta_steps, xi_steps, xi_max):
    """Hanson's complex cosine and sine, with u1^2 + u2^2 = 1.

    u1 = cosh(xi + i theta) = cosh(xi) cos(theta) + i sinh(xi) sin(theta)
    u2 = -i sinh(xi + i theta) = cosh(xi) sin(theta) - i sinh(xi) cos(theta)

    written out in real and imaginary parts so the pinned endpoints of
    `_grid` survive.  Both have non-negative real part over
    theta in [0, pi/2], so the principal branch of the fractional
    power never crosses its cut.
    """
    c, s, xi = _grid(theta_steps, xi_steps, xi_max)
    ch, sh = np.cosh(xi)[:, None], np.sinh(xi)[:, None]
    c, s = c[None, :], s[None, :]
    return ch * c + 1j * sh * s, ch * s - 1j * sh * c


def patch(p, q, k1, k2, theta_steps, xi_steps, xi_max, phase=0.0):
    """One of the p*q phase-related patches of z1^p + z2^q = c.

    Returns (z1, z2) as complex arrays of shape (xi_steps,
    theta_steps).  `phase` s slides the whole family along
    c = exp(2 pi i s): z1 picks up exp(2 pi i s / p) and z2
    exp(2 pi i s / q), so at s = 1 patch (k1, k2) has become
    (k1 + 1, k2 + 1) and the surface has returned to itself.
    """
    u1, u2 = _u1_u2(theta_steps, xi_steps, xi_max)
    s1 = np.exp(2j * math.pi * (k1 + phase) / p)
    s2 = np.exp(2j * math.pi * (k2 + phase) / q)
    return s1 * u1 ** (2.0 / p), s2 * u2 ** (2.0 / q)


def residual(p, q, theta_steps=17, xi_steps=17, xi_max=1.0, phase=0.0):
    """max |z1^p + z2^q - exp(2 pi i phase)| over every patch.

    The one number that catches a wrong phase convention.
    """
    want = cmath.exp(2j * math.pi * phase)
    worst = 0.0
    for k1 in range(p):
        for k2 in range(q):
            z1, z2 = patch(p, q, k1, k2, theta_steps, xi_steps,
                           xi_max, phase)
            worst = max(worst,
                        float(np.max(np.abs(z1 ** p + z2 ** q - want))))
    return worst


def expected_topology(p, q):
    """(chi, boundary components, genus) of the Milnor fibre.

    The fibre of z1^p + z2^q has first Betti number mu = (p-1)(q-1)
    (the Milnor number) and d = gcd(p, q) boundary circles, so
    chi = 1 - mu and g = (mu - d + 1)/2.
    """
    mu = (p - 1) * (q - 1)
    d = math.gcd(p, q)
    return 1 - mu, d, (mu - d + 1) // 2


# --------------------------------------------------------------------
# Meshing
# --------------------------------------------------------------------

def cross_section(p=5, q=5, theta_steps=17, xi_steps=17, xi_max=1.0,
                  phase=0.0, angle=45.0, weld=True):
    """Build the whole surface.

    Returns (pts4, faces, patch_id), where `pts4` is a list of
    (X, Y, Z, W) -- the R^4 point, with W the coordinate the
    projection discards -- `faces` a list of quads, and `patch_id`
    one integer k1 + p * k2 per face.

    Welding is done on the R^4 coordinates, never on the projected
    R^3 ones: two sheets that cross in the picture are generally far
    apart on the surface, and merging them there would fuse the
    projection's self-intersections into the topology.
    """
    if xi_steps % 2 == 0:                     # must sample xi = 0
        xi_steps += 1
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))

    pts4, faces, patch_id = [], [], []
    cells = defaultdict(list)                 # welding grid
    tol = 1e-7

    def merge(v):
        """Index of an existing point within `tol`, else a new one.

        Quantising to a grid and probing the 3^4 neighbouring cells
        rather than trusting a single rounded key: two points that
        agree to 1e-16 can still straddle a quantisation boundary,
        and when they do the surface silently comes apart at a seam.
        """
        base = tuple(int(math.floor(x / tol)) for x in v)
        for off in _OFF4:
            for n in cells.get((base[0] + off[0], base[1] + off[1],
                                base[2] + off[2], base[3] + off[3]), ()):
                w = pts4[n]
                if all(abs(v[k] - w[k]) <= tol for k in range(4)):
                    return n
        n = len(pts4)
        pts4.append(v)
        cells[base].append(n)
        return n

    for k2 in range(q):
        for k1 in range(p):
            z1, z2 = patch(p, q, k1, k2, theta_steps, xi_steps,
                           xi_max, phase)
            X, Y = z1.real, z2.real
            Z = ca * z1.imag + sa * z2.imag
            W = sa * z1.imag - ca * z2.imag
            idx = np.empty((xi_steps, theta_steps), dtype=np.int64)
            for i in range(xi_steps):
                for j in range(theta_steps):
                    v = (float(X[i, j]), float(Y[i, j]),
                         float(Z[i, j]), float(W[i, j]))
                    # Only the theta = 0 and theta = pi/2 columns can
                    # coincide with another patch: away from them |u1|
                    # and |u2| pin (theta, |xi|), and the sheets are
                    # separated by the roots of unity.  Welding just
                    # those two columns keeps the cost independent of
                    # the interior resolution.
                    if weld and j in (0, theta_steps - 1):
                        n = merge(v)
                    else:
                        n = len(pts4)
                        pts4.append(v)
                    idx[i, j] = n
            pid = k1 + p * k2
            for i in range(xi_steps - 1):
                for j in range(theta_steps - 1):
                    a, b = int(idx[i, j]), int(idx[i, j + 1])
                    c, d = int(idx[i + 1, j + 1]), int(idx[i + 1, j])
                    if len({a, b, c, d}) < 4:     # degenerate at a seam
                        continue
                    faces.append((a, b, c, d))
                    patch_id.append(pid)
    return pts4, faces, patch_id


def _edge_use(faces):
    use = defaultdict(int)
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            use[(min(a, b), max(a, b))] += 1
    return use


def boundary_loops(faces):
    """Index loops around the boundary edges, longest first."""
    use = _edge_use(faces)
    adj = defaultdict(list)
    for (a, b), n in use.items():
        if n == 1:
            adj[a].append(b)
            adj[b].append(a)
    loops, seen = [], set()
    for start in list(adj):
        if start in seen:
            continue
        loop, cur, prev = [start], start, None
        seen.add(start)
        while True:
            nxt = [x for x in adj[cur] if x != prev and x not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) > 2:
            loops.append(loop)
    loops.sort(key=len, reverse=True)
    return loops


def topology(pts4, faces):
    """(V, E, F, chi, boundary loop count) of a welded mesh."""
    use = _edge_use(faces)
    used = {i for f in faces for i in f}
    v, e, f = len(used), len(use), len(faces)
    return v, e, f, v - e + f, len(boundary_loops(faces))


def fit(pts, scale=1.0):
    """Centre on the bounding-box midpoint, largest extent 2*scale."""
    if not pts:
        return pts
    A = np.asarray(pts, dtype=float)
    lo, hi = A.min(axis=0), A.max(axis=0)
    ext = float((hi - lo).max())
    s = (2.0 * scale / ext) if ext > 1e-12 else 1.0
    return [tuple(v) for v in (A - 0.5 * (lo + hi)) * s]


# --------------------------------------------------------------------
# Blender layer
# --------------------------------------------------------------------

if _IN_BLENDER:

    _PALETTE = ((0.90, 0.24, 0.22), (0.95, 0.55, 0.15),
                (0.92, 0.82, 0.20), (0.35, 0.72, 0.30),
                (0.20, 0.62, 0.78), (0.28, 0.35, 0.72),
                (0.55, 0.32, 0.70), (0.85, 0.40, 0.62),
                (0.45, 0.45, 0.45), (0.70, 0.55, 0.35))

    class MESH_OT_calabi_yau_add(bpy.types.Operator):
        """Add Hanson's cross-section of the Calabi-Yau quintic"""
        bl_idname = "mesh.calabi_yau_add"
        bl_label = "Calabi-Yau Cross-Section"
        bl_options = {'REGISTER', 'UNDO'}

        family: EnumProperty(
            name="Family",
            description="Which complex curve to section",
            items=[('FERMAT', "Fermat",
                    "z1^n + z2^n = 1, the cross-section of the "
                    "degree-n Fermat surface. n = 5 is the quintic "
                    "threefold's cross-section -- the picture "
                    "reproduced everywhere as 'a Calabi-Yau manifold'"),
                   ('BRIESKORN', "Brieskorn",
                    "z1^p + z2^q = 1 with the two powers independent: "
                    "the Milnor fibre of a Brieskorn singularity, "
                    "spanning the (p, q) torus link")],
            default='FERMAT')
        degree: IntProperty(
            name="Degree", default=5, min=2, max=12,
            description="Degree n of the Fermat curve z1^n + z2^n = 1. "
                        "The surface is built from n x n patches and "
                        "has genus (n-1)(n-2)/2")
        power_p: IntProperty(
            name="First Power", default=2, min=2, max=9,
            description="Exponent p in z1^p + z2^q = 1")
        power_q: IntProperty(
            name="Second Power", default=3, min=2, max=9,
            description="Exponent q in z1^p + z2^q = 1. With p, the "
                        "boundary is the (p, q) torus link")
        angle: FloatProperty(
            name="Projection Angle", default=45.0, min=-180.0,
            max=180.0,
            description="Which shadow of the 4-dimensional surface to "
                        "look at: the height is cos(angle) Im z1 + "
                        "sin(angle) Im z2. Turning it moves the "
                        "self-intersections, which belong to the "
                        "projection and not to the surface")
        phase: FloatProperty(
            name="Phase", default=0.0, min=-1.0, max=1.0,
            description="Slides along the family z1^p + z2^q = "
                        "exp(2 pi i phase). The patches rotate into "
                        "one another and the surface returns to itself "
                        "at phase = 1")
        extent: FloatProperty(
            name="Extent", default=1.0, min=0.1, max=3.0,
            description="How far the barbs run out towards the curve's "
                        "points at infinity. The surface is unbounded; "
                        "this is where it is cut off")
        theta_steps: IntProperty(
            name="Patch Segments", default=17, min=3, max=97,
            description="Samples across each patch")
        xi_steps: IntProperty(
            name="Patch Rings", default=17, min=3, max=97,
            description="Samples along each patch. Forced odd so the "
                        "surface passes through the fixed points at "
                        "the middle of every patch")
        colour: EnumProperty(
            name="Colour",
            description="What to paint on the surface",
            items=[('NONE', "None", "Leave the mesh unpainted"),
                   ('PATCH', "Patch",
                    "One colour per (k1, k2) patch: shows the "
                    "complex phase of each patch relative to the "
                    "basis patch"),
                   ('SYMMETRY', "Symmetry",
                    "Colour by (k1 + k2) mod n, which makes the "
                    "n-fold symmetry of the surface obvious"),
                   ('FOURTH', "Fourth Dimension",
                    "Colour by the coordinate the projection throws "
                    "away -- the only way it survives into the render")],
            default='PATCH')
        weld: BoolProperty(
            name="Weld Patches", default=True,
            description="Join the patches into one surface, welding on "
                        "the 4D coordinates so that crossings in the "
                        "projection are not fused")
        boundary_curve: BoolProperty(
            name="Boundary Link", default=False,
            description="Also emit the boundary as a bevelled curve. "
                        "It is the (p, q) torus link")
        thickness: FloatProperty(
            name="Thickness", default=0.0, min=0.0, max=0.5,
            description="Solidify the sheet for printing (0 = leave it "
                        "a surface)")
        scale: FloatProperty(name="Scale", default=1.0, min=0.01,
                             max=100.0,
                             description="Overall size multiplier")

        def _powers(self):
            if self.family == 'FERMAT':
                return self.degree, self.degree
            return self.power_p, self.power_q

        def execute(self, context):
            p, q = self._powers()
            nxi = self.xi_steps + (1 - self.xi_steps % 2)
            pts4, faces, pid = cross_section(
                p, q, self.theta_steps, nxi, self.extent,
                self.phase, self.angle, self.weld)
            if not faces:
                self.report({'ERROR'}, "empty surface")
                return {'CANCELLED'}

            w = [v[3] for v in pts4]
            pts = fit([v[:3] for v in pts4], self.scale)

            me = bpy.data.meshes.new("CalabiYau")
            me.from_pydata(pts, [], faces)
            me.validate(clean_customdata=True)
            me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))

            att = me.attributes.new("fourth_coordinate", 'FLOAT', 'POINT')
            att.data.foreach_set('value', w)

            if self.colour != 'NONE':
                col = me.color_attributes.new(name="patch",
                                              type='FLOAT_COLOR',
                                              domain='CORNER')
                if self.colour == 'FOURTH':
                    lo, hi = min(w), max(w)
                    rng = (hi - lo) or 1.0
                    buf = []
                    for poly in me.polygons:
                        for vi in poly.vertices:
                            t = (w[vi] - lo) / rng
                            buf += [0.15 + 0.8 * t, 0.35 + 0.2 * abs(0.5 - t),
                                    0.95 - 0.7 * t, 1.0]
                else:
                    buf = []
                    for f_i, poly in enumerate(me.polygons):
                        k = pid[f_i]
                        if self.colour == 'SYMMETRY':
                            k = ((k % p) + (k // p)) % max(p, q)
                        r, g, b = _PALETTE[k % len(_PALETTE)]
                        shade = 0.72 + 0.28 * (((k * 7) % 5) / 4.0)
                        buf += [r * shade, g * shade, b * shade, 1.0] * \
                            len(poly.vertices)
                col.data.foreach_set('color', buf)

            me.update()
            obj = bpy.data.objects.new("Calabi-Yau Cross-Section", me)
            context.collection.objects.link(obj)
            obj.location = context.scene.cursor.location

            if self.thickness > 0.0:
                mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
                mod.thickness = self.thickness
                mod.offset = 0.0

            if self.boundary_curve:
                self._add_boundary(context, obj, pts, faces)

            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

            chi, nb, genus = expected_topology(p, q)
            V, E, F, got, loops = topology(pts4, faces)
            ok = "OK" if (not self.weld or
                          (got == chi and loops == nb)) else "??"
            self.report({'INFO'},
                        f"({p},{q}) torus link: genus {genus}, "
                        f"{nb} boundary circles, chi {chi} "
                        f"[mesh chi {got}, {loops} loops {ok}]")
            return {'FINISHED'}

        def _add_boundary(self, context, parent, pts, faces):
            loops = boundary_loops(faces)
            if not loops:
                return
            cu = bpy.data.curves.new("CalabiYauLink", 'CURVE')
            cu.dimensions = '3D'
            cu.bevel_depth = 0.02
            cu.bevel_resolution = 4
            for loop in loops:
                sp = cu.splines.new('POLY')
                sp.points.add(len(loop) - 1)
                for i, vi in enumerate(loop):
                    sp.points[i].co = (*pts[vi], 1.0)
                sp.use_cyclic_u = True
            ob = bpy.data.objects.new("Boundary Link", cu)
            context.collection.objects.link(ob)
            ob.location = parent.location
            ob.parent = parent
            ob.matrix_parent_inverse = parent.matrix_world.inverted()

        def draw(self, context):
            lay = self.layout
            lay.use_property_split = True
            lay.prop(self, 'family')
            if self.family == 'FERMAT':
                lay.prop(self, 'degree')
            else:
                lay.prop(self, 'power_p')
                lay.prop(self, 'power_q')
            lay.prop(self, 'angle')
            lay.prop(self, 'phase')
            lay.prop(self, 'extent')
            lay.prop(self, 'theta_steps')
            lay.prop(self, 'xi_steps')
            lay.prop(self, 'colour')
            lay.prop(self, 'weld')
            lay.prop(self, 'boundary_curve')
            lay.prop(self, 'thickness')
            lay.prop(self, 'scale')

    def _menu_func(self, context):
        self.layout.operator("mesh.calabi_yau_add",
                             icon='SURFACE_NSURFACE')

    ADD_MENU = True

    def register():
        bpy.utils.register_class(MESH_OT_calabi_yau_add)
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.append(_menu_func)

    def unregister():
        if ADD_MENU:
            bpy.types.VIEW3D_MT_mesh_add.remove(_menu_func)
        bpy.utils.unregister_class(MESH_OT_calabi_yau_add)


# --------------------------------------------------------------------

def _selftest():
    bad = []

    # 1. the defining equation, over every patch of several families,
    #    with the phase turned on.  This is the check that catches the
    #    (1/i) sinh^(2/n) transcription error.
    for (p, q, s) in ((5, 5, 0.0), (5, 5, 0.37), (3, 3, 0.0),
                      (2, 3, 0.0), (4, 7, 0.19), (8, 8, 0.5)):
        r = residual(p, q, 13, 13, 1.2, s)
        print(f"calabi_yau: |z1^{p} + z2^{q} - e(2 pi i {s})| "
              f"max = {r:.2e} {'OK' if r < 1e-9 else 'BAD'}")
        if r >= 1e-9:
            bad.append(f"residual ({p},{q}) = {r:.2e}")

    # 2. the topology of the welded mesh against the Milnor fibre.
    for (p, q) in ((2, 2), (2, 3), (3, 3), (2, 5), (4, 6), (5, 5)):
        chi, nb, genus = expected_topology(p, q)
        pts4, faces, _ = cross_section(p, q, 13, 13, 1.0, 0.0, 45.0,
                                       True)
        V, E, F, got, loops = topology(pts4, faces)
        ok = got == chi and loops == nb
        print(f"calabi_yau: ({p},{q}) V={V} E={E} F={F} "
              f"chi={got}({chi}) loops={loops}({nb}) genus={genus} "
              f"{'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(f"topology ({p},{q}): chi {got} != {chi} or "
                       f"loops {loops} != {nb}")

    # 3. the fit: centred, and inside a 2 m cube.
    pts4, faces, _ = cross_section(5, 5, 9, 9, 1.0)
    P = np.asarray(fit([v[:3] for v in pts4]))
    ctr = float(np.abs(0.5 * (P.min(axis=0) + P.max(axis=0))).max())
    ext = float((P.max(axis=0) - P.min(axis=0)).max())
    ok = ctr < 1e-9 and abs(ext - 2.0) < 1e-9
    print(f"calabi_yau: fit centre={ctr:.2e} extent={ext:.6f} "
          f"{'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("fit")

    # 4. xi = 0 is sampled, so the patches meet at their fixed points:
    #    an even ring count is bumped to odd inside cross_section.
    a = cross_section(3, 3, 5, 8, 1.0)[0]
    b = cross_section(3, 3, 5, 9, 1.0)[0]
    ok = len(a) == len(b)
    print(f"calabi_yau: even ring count bumped to odd "
          f"{'OK' if ok else 'BAD'}")
    if not ok:
        bad.append("xi parity")

    if bad:
        raise AssertionError("; ".join(bad))
